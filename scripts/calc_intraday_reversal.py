"""
因子: 日内反转强度 (Intraday Reversal Strength) v1
----------------------------------------------
构造:
  1. gap = (open - prev_close) / prev_close  # 跳空方向
  2. intraday = (close - open) / open  # 日内走势
  3. reversal_signal = -gap * intraday  # 跳空和日内走势方向相反时为正
     - gap_up + close<open → 正值（高开低走反转）
     - gap_down + close>open → 正值（低开高走反转）
  4. factor = MA20(reversal_signal) 的z-score
  5. 成交额OLS中性化 + MAD winsorize + z-score

逻辑:
  - 高日内反转强度 = 开盘定价频繁被日内交易纠正 = 散户/噪声交易者影响开盘
  - 这种模式说明：(a)集合竞价定价效率低 (b)日内有理性资金介入纠正
  - 累积来看，经常被纠正的股票可能处于信息不确定期
  - 可能方向：正向(反转力量强=market microstructure alpha) 或 负向(价格发现效率低)

Barra风格: 微观结构
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
    
    # 计算前一日收盘
    print("计算日内反转信号...")
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    
    # gap和日内收益
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['intraday'] = (df['close'] - df['open']) / df['open']
    
    # 反转信号: gap方向和日内方向相反时为正
    df['reversal_signal'] = -df['gap'] * df['intraday']
    
    # 20日滚动均值
    df['reversal_ma20'] = df.groupby('stock_code')['reversal_signal'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    
    # 20日成交额均值(用于中性化)
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20).mean() + 1)
    )
    
    # 过滤
    factor_df = df[['date', 'stock_code', 'reversal_ma20', 'log_amount_20d']].dropna()
    print(f"原始因子行数: {len(factor_df)}")
    
    # OLS中性化
    print("成交额OLS中性化...")
    factor_df = factor_df.set_index(['date', 'stock_code'])
    raw = factor_df['reversal_ma20']
    neutral_var = factor_df['log_amount_20d']
    
    neutralized = pd.Series(np.nan, index=raw.index)
    dates = raw.index.get_level_values('date').unique()
    for dt in dates:
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
    
    # 输出
    output = factor_df[['factor']].reset_index()
    output.columns = ['date', 'stock_code', 'factor']
    output = output.dropna()
    
    out_path = base / 'data' / 'factor_intraday_reversal_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子输出到: {out_path}")
    print(f"总行数: {len(output)}")
    print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"股票数: {output['stock_code'].nunique()}")
    
    print("\n截面因子统计 (最后一日):")
    last = output[output['date'] == output['date'].max()]
    print(f"  均值: {last['factor'].mean():.4f}")
    print(f"  标准差: {last['factor'].std():.4f}")

if __name__ == '__main__':
    main()
