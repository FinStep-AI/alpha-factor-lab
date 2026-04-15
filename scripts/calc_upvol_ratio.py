"""
因子: 上行波动率比率 (Upside Volatility Ratio) v1
----------------------------------------------
构造:
  1. 20日窗口内, 分别计算正收益日和负收益日的标准差
     upvol = std(ret where ret>0, 20d)
     downvol = std(|ret| where ret<0, 20d) 
  2. ratio = upvol / (upvol + downvol) ∈ (0, 1)
     >0.5 = 上涨波动 > 下跌波动 = 正偏态
  3. 成交额OLS中性化 + MAD winsorize + z-score

逻辑:
  - 传统波动率因子(idio_vol, amp_level)只看总波动, 不区分方向
  - 本因子分离上行/下行波动率, 捕捉波动率的方向不对称性
  - 上行波动大(正偏态) = 涨的时候涨得猛, 跌的时候跌幅温和
  - 这种不对称性可能来自: 基本面改善初期(大涨小回调), 知情交易者买入等
  - 文献: Ang, Chen & Xing (2006) 关注下行风险, 本因子关注上行/下行比率

Barra风格: Volatility (细分方向)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def rolling_updown_vol_ratio(rets, window=20, min_obs=5):
    """计算滚动上行/下行波动率比率"""
    n = len(rets)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        w = rets[i - window + 1: i + 1]
        valid = w[~np.isnan(w)]
        up = valid[valid > 0]
        down = valid[valid < 0]
        
        if len(up) < min_obs or len(down) < min_obs:
            continue
        
        up_vol = np.std(up, ddof=1)
        down_vol = np.std(np.abs(down), ddof=1)
        
        total = up_vol + down_vol
        if total > 0:
            result[i] = up_vol / total
    
    return result

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
    
    # 计算日收益率
    print("计算日收益率...")
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 计算20日成交额均值(用于中性化)
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20).mean() + 1)
    )
    
    # 计算上行/下行波动率比率
    print("计算上行/下行波动率比率(20d)...")
    results = []
    for stk, gdf in df.groupby('stock_code'):
        rets = gdf['ret'].values
        dates = gdf['date'].values
        log_amt = gdf['log_amount_20d'].values
        
        ratio = rolling_updown_vol_ratio(rets, window=20, min_obs=5)
        
        for i in range(len(dates)):
            if not np.isnan(ratio[i]) and not np.isnan(log_amt[i]):
                results.append({
                    'date': dates[i],
                    'stock_code': stk,
                    'up_vol_ratio': ratio[i],
                    'log_amount_20d': log_amt[i]
                })
    
    factor_df = pd.DataFrame(results)
    print(f"原始因子行数: {len(factor_df)}")
    
    # OLS中性化
    print("成交额OLS中性化...")
    factor_df = factor_df.set_index(['date', 'stock_code'])
    raw = factor_df['up_vol_ratio']
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
    
    out_path = base / 'data' / 'factor_upvol_ratio_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子输出到: {out_path}")
    print(f"总行数: {len(output)}")
    print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"股票数: {output['stock_code'].nunique()}")
    
    print("\n截面因子统计 (最后一日):")
    last = output[output['date'] == output['date'].max()]
    print(f"  均值: {last['factor'].mean():.4f}")
    print(f"  标准差: {last['factor'].std():.4f}")
    print(f"  分位数: {last['factor'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()}")

if __name__ == '__main__':
    main()
