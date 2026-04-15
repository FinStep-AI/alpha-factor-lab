"""
因子: 成交额均线离散度 (Amount MA Dispersion) v1
----------------------------------------------
构造:
  1. 计算多周期成交额均线: MA5, MA10, MA20, MA40, MA60
  2. 归一化: MA_x / MA20 (以20日均线为基准)
  3. 截面标准差: std(MA5/MA20, MA10/MA20, MA20/MA20, MA40/MA20, MA60/MA20)
  4. 成交额OLS中性化 + MAD winsorize + z-score

逻辑:
  - ma_disp成功(t=4.25)说明"离散度"这个框架在CSI1000非常有效
  - 本因子将同样的框架从价格均线迁移到成交额均线
  - 高离散度 = 成交额在不同时间尺度上差异大 = 资金流趋势明确
    (短期放量+长期缩量 或 短期缩量+长期放量)
  - 低离散度 = 各周期成交额一致 = 资金流稳定
  - 可能方向: 正向(资金流趋势明确=更多alpha) 或 负向

Barra风格: Liquidity/Sentiment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def mad_winsorize(s, n_mad=5):
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    upper = med + n_mad * 1.4826 * mad
    lower = med - n_mad * 1.4826 * mad
    return s.clip(lower, upper)

def zscore(s):
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - mean) / std

def main():
    base = Path(__file__).resolve().parent.parent
    
    print("读取K线数据...")
    df = pd.read_csv(base / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算多周期成交额均线
    print("计算多周期成交额均线...")
    for w in [5, 10, 20, 40, 60]:
        df[f'amt_ma{w}'] = df.groupby('stock_code')['amount'].transform(
            lambda x: x.rolling(w, min_periods=int(w*0.75)).mean()
        )
    
    # 归一化: 以MA20为基准
    for w in [5, 10, 40, 60]:
        df[f'amt_ratio_{w}'] = df[f'amt_ma{w}'] / df['amt_ma20']
    df['amt_ratio_20'] = 1.0  # MA20/MA20 = 1
    
    # 截面标准差(跨周期)
    ratio_cols = ['amt_ratio_5', 'amt_ratio_10', 'amt_ratio_20', 'amt_ratio_40', 'amt_ratio_60']
    df['amt_disp'] = df[ratio_cols].std(axis=1)
    
    # 20日成交额均值(中性化用)
    df['log_amount_20d'] = np.log(df['amt_ma20'] + 1)
    
    # 过滤
    factor_df = df[['date', 'stock_code', 'amt_disp', 'log_amount_20d']].dropna()
    print(f"原始因子行数: {len(factor_df)}")
    
    # OLS中性化
    print("成交额OLS中性化...")
    factor_df = factor_df.set_index(['date', 'stock_code'])
    raw = factor_df['amt_disp']
    neutral_var = factor_df['log_amount_20d']
    
    neutralized = pd.Series(np.nan, index=raw.index)
    for dt in raw.index.get_level_values('date').unique():
        mask = raw.index.get_level_values('date') == dt
        y = raw[mask].dropna()
        x = neutral_var[mask].dropna()
        common = y.index.intersection(x.index)
        if len(common) < 30:
            continue
        y_c = y[common].values
        x_c = x[common].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            resid = y_c - X @ beta
            neutralized[common] = resid
        except:
            continue
    
    factor_df['factor_neutralized'] = neutralized
    factor_df = factor_df.dropna(subset=['factor_neutralized'])
    
    # MAD winsorize + z-score
    print("MAD winsorize + z-score...")
    final_values = []
    for dt in factor_df.index.get_level_values('date').unique():
        mask = factor_df.index.get_level_values('date') == dt
        vals = factor_df.loc[mask, 'factor_neutralized']
        vals = mad_winsorize(vals)
        vals = zscore(vals)
        final_values.append(vals)
    
    factor_df['factor'] = pd.concat(final_values)
    
    output = factor_df[['factor']].reset_index()
    output.columns = ['date', 'stock_code', 'factor']
    output = output.dropna()
    
    out_path = base / 'data' / 'factor_amt_disp_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子输出到: {out_path}")
    print(f"总行数: {len(output)}")
    print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"股票数: {output['stock_code'].nunique()}")

if __name__ == '__main__':
    main()
