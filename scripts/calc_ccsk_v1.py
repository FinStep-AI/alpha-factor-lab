#!/usr/bin/env python3
"""
条件协偏度因子 (Conditional Coskewness, CCSK)
================================================

公式: CCSK_i = -Cov(r_i · I{r_i>0} - r_i · I{r_i<0}, r_mkt²) / Var(r_mkt), 60日滚动

理论:
  Harvey & Siddique (2000): Conditional Skewness Asset Pricing
  Boyer, Mitton & Vorkink (2010): Expected Idiosyncratic Skewness, RFS
  
  高条件协偏度(正值)=股票收益在市场方差高的日子更负→更差的极端风险暴露
  低条件协偏度(负值)=在市场高波动时反而上涨→防御性强

构造:
  1. 分市场状态: r_mkt > 0 (牛市) vs r_mkt < 0 (熊市)
  2. 计算分段协偏度: CCSK_up = Cov(r_i | r_mkt>0, r_mkt² | r_mkt>0)
  3. 最终因子 = -(CSK_up - CSK_down) 即牛市协偏 - 熊市协偏

简化版(日频可行):
  CCSK_i = -cov(r_i * sign(r_mkt), r_mkt²), 60日滚动窗口
  
  当市场大跌(rmkt<0)且波动大(rmkt²高): sign(rmkt)=-1
    → r_i*(-1) 与 rmkt²正协方差 → CCSK为负(低因子值=差)
  当市场大跌但波动小: rmkt²低 → 影响小

预期方向: 正向(高CCSK = 在市场波动大时表现好/受保护 = 高收益)

Barra风格: Risk
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def rolling_coskewness(stock_ret, mkt_sq, window=60, min_periods=40):
    """
    计算滚动窗口内 r_i 与 r_mkt² 的协方差。
    使用在线算法: cov(x,y) = E[xy] - E[x]E[y]
    """
    n = len(stock_ret)
    result = np.full(n, np.nan)
    
    # Use rolling window via pandas for efficiency
    s = pd.Series(stock_ret, index=range(n))
    m = pd.Series(mkt_sq, index=range(n))
    
    for i in range(window-1, n):
        start = i - window + 1
        end = i + 1
        x = s.iloc[start:end].values
        y = m.iloc[start:end].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_periods:
            continue
        x_clean = x[mask]
        y_clean = y[mask]
        cov_val = np.cov(x_clean, y_clean, ddof=0)[0, 1]
        if np.isfinite(cov_val):
            result[i] = cov_val
    
    return result


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    # Load returns
    print("读取收益率数据...")
    ret = pd.read_csv(data_dir / "csi1000_returns.csv", parse_dates=['date'])
    ret_wide = ret.pivot(index='date', columns='stock_code', values='return')
    ret_wide = ret_wide.sort_index()
    
    # Load market returns
    print("读取市场收益率...")
    mkt_ret = pd.read_csv(data_dir / "mkt_ret_daily.csv", parse_dates=['date'], index_col='date')
    mkt_ret = mkt_ret['mkt_return']
    mkt_ret = mkt_ret.reindex(ret_wide.index)
    
    # Market squared returns
    mkt_sq = mkt_ret ** 2
    
    # Align
    common_dates = ret_wide.index.intersection(mkt_sq.index)
    ret_wide = ret_wide.loc[common_dates]
    mkt_sq = mkt_sq.loc[common_dates]
    
    print(f"数据: {ret_wide.index.min().date()} ~ {ret_wide.index.max().date()}")
    print(f"股票数: {ret_wide.shape[1]:d}, 交易日: {len(ret_wide):d}")
    
    # Compute rolling coskewness for each stock
    window = 60
    min_periods = 40
    results = []
    
    stocks = ret_wide.columns.tolist()
    total_stocks = len(stocks)
    
    for idx, stock in enumerate(stocks):
        if (idx + 1) % 100 == 0:
            print(f"  处理 {idx+1}/{total_stocks}...")
        
        stock_ret = ret_wide[stock].values.astype(float)
        
        # Rolling covariance of stock_ret with mkt_sq
        cov_vals = rolling_coskewness(stock_ret, mkt_sq.values, window, min_periods)
        
        # CCSK = -cov(r_i, r_mkt²) (negative: want stocks that benefit from market vol)
        ccsk_raw = -cov_vals
        
        # Store
        for date_idx, date in enumerate(ret_wide.index):
            if np.isfinite(ccsk_raw[date_idx]):
                results.append({
                    "date": date,
                    "stock_code": stock,
                    "ccsk_raw": ccsk_raw[date_idx]
                })
    
    print(f"\n原始因子值: {len(results)} 行")
    df_raw = pd.DataFrame(results)
    
    # --- 成交额中性化 (amount neutralization) ---
    print("读取成交额数据...")
    kline = pd.read_csv(data_dir / "csi1000_kline_raw.csv", dtype={'stock_code': str})
    kline['date'] = pd.to_datetime(kline['date'])
    # log(20日均成交额)
    g = kline.groupby('stock_code')
    kline['log_amt_20'] = g['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    amt_df = kline[['date', 'stock_code', 'log_amt_20']].dropna()
    
    # Ensure consistent dtypes before merge
    df_raw['stock_code'] = df_raw['stock_code'].astype(str)
    amt_df['stock_code'] = amt_df['stock_code'].astype(str)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    amt_df['date'] = pd.to_datetime(amt_df['date'])
    
    # Merge
    df_raw = df_raw.merge(amt_df, on=['date', 'stock_code'], how='inner')
    df_raw = df_raw.dropna(subset=['ccsk_raw', 'log_amt_20'])
    
    print(f"合并成交额后: {len(df_raw)} 行")
    
    # --- 截面OLS中性化 + MAD + z-score ---
    final_results = []
    dates = sorted(df_raw['date'].unique())
    
    for date in dates:
        sub = df_raw[df_raw['date'] == date]
        if len(sub) < 50:
            continue
        
        y = sub['ccsk_raw'].values.astype(float)
        x = sub['log_amt_20'].values.astype(float)
        
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        y_clean = y[mask]
        
        try:
            beta = np.linalg.solve(X.T @ X + 1e-10 * np.eye(2), X.T @ y_clean)
            residual = np.full_like(y, np.nan)
            residual[mask] = y_clean - X @ beta
        except Exception as e:
            continue
        
        # MAD winsorize
        valid = np.isfinite(residual)
        if valid.sum() < 30:
            continue
        residual_valid = residual[valid]
        med = np.nanmedian(residual_valid)
        mad = np.nanmedian(np.abs(residual_valid - med))
        if mad > 1e-10:
            scaled = 1.4826 * mad
            residual = np.clip(residual, med - 5 * scaled, med + 5 * scaled)
        
        # z-score
        std = np.nanstd(residual)
        if std > 1e-10:
            residual = (residual - np.nanmean(residual)) / std
        
        sub_idx = sub.index[valid]
        for i, idx in enumerate(sub_idx):
            row = sub.loc[idx]
            final_results.append({
                "date": date,
                "stock_code": row['stock_code'],
                "factor_value": residual[valid][i]
            })
    
    result_df = pd.DataFrame(final_results)
    result_df = result_df.dropna(subset=['factor_value'])
    
    print(f"\n最终因子: {len(result_df)} 行, {result_df['date'].nunique()} 截面")
    print(f"日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
    print(f"因子分布: mean={result_df['factor_value'].mean():.4f}, "
          f"std={result_df['factor_value'].std():.4f}, "
          f"min={result_df['factor_value'].min():.4f}, "
          f"max={result_df['factor_value'].max():.4f}")
    
    # Save
    output_file = data_dir / "factor_ccsk_v1.csv"
    result_df["date_str"] = result_df["date"].dt.strftime("%Y-%m-%d")
    result_df[["date_str", "stock_code", "factor_value"]].rename(
        columns={"date_str": "date"}
    ).to_csv(output_file, index=False)
    print(f"\n已保存: {output_file}")


if __name__ == "__main__":
    main()
