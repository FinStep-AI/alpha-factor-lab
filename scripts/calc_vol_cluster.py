"""
因子: 波动率聚类强度 (Volatility Clustering) v1
----------------------------------------------
构造:
  1. 计算20日窗口内 |ret_t| 和 |ret_{t-1}| 的滚动相关系数
     高值 = 波动率聚类强(GARCH效应) = 大波动倾向于连续出现
  2. 成交额OLS中性化 + MAD winsorize + z-score

逻辑:
  - 波动率聚类是GARCH模型的核心特征之一
  - 强聚类 = 信息释放缓慢, 不确定性持续 = 可能的风险溢价
  - 弱聚类 = 波动随机, 信息快速消化
  - 这个角度与现有波动率因子(amp_level, idio_vol)不同:
    它们看波动率水平, 本因子看波动率的时间自相关性

Barra风格: Volatility (波动率动态特征)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def rolling_vol_cluster(abs_ret, window=20, min_obs=15):
    """滚动|ret_t|和|ret_{t-1}|的相关系数"""
    n = len(abs_ret)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        w = abs_ret[i - window: i + 1]
        if np.sum(~np.isnan(w)) < min_obs + 1:
            continue
        x = w[:-1]  # |ret_{t-1}|
        y = w[1:]   # |ret_t|
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < min_obs:
            continue
        x_v = x[valid]
        y_v = y[valid]
        if x_v.std() == 0 or y_v.std() == 0:
            continue
        corr = np.corrcoef(x_v, y_v)[0, 1]
        result[i] = corr
    
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
    
    # 日收益率
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df['abs_ret'] = df['ret'].abs()
    
    # 成交额(中性化用)
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20).mean() + 1)
    )
    
    print("计算波动率聚类强度(20d)...")
    results = []
    for stk, gdf in df.groupby('stock_code'):
        abs_ret = gdf['abs_ret'].values
        dates = gdf['date'].values
        log_amt = gdf['log_amount_20d'].values
        
        vc = rolling_vol_cluster(abs_ret, window=20, min_obs=15)
        
        for i in range(len(dates)):
            if not np.isnan(vc[i]) and not np.isnan(log_amt[i]):
                results.append({
                    'date': dates[i],
                    'stock_code': stk,
                    'vol_cluster': vc[i],
                    'log_amount_20d': log_amt[i]
                })
    
    factor_df = pd.DataFrame(results)
    print(f"原始因子行数: {len(factor_df)}")
    
    # OLS中性化
    print("成交额OLS中性化...")
    factor_df = factor_df.set_index(['date', 'stock_code'])
    raw = factor_df['vol_cluster']
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
    
    out_path = base / 'data' / 'factor_vol_cluster_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子输出到: {out_path}")
    print(f"总行数: {len(output)}")
    print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"股票数: {output['stock_code'].nunique()}")

if __name__ == '__main__':
    main()
