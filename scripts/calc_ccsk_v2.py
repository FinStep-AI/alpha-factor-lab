#!/usr/bin/env python3
"""
分段协偏度 v2 (Segmented Coskewness)
================================================

理论:
  - 在「市场波动大的日子」中，股价越敏感(负收益越多)者属于高风险资产
  - 若某股票在市场高波动日反而正收益(抗跌)，说明有独特alpha
  - 这是Harvey & Siddique (2000) Conditional Skewness的简化版日频代理

构造 (Piecewise Coskewness):
  1. 标记市场高波动日: r_mkt² > 60日均值 → 高波动期
  2. 计算在高波动期的协偏度: CCSK_up = cov(r_i, r_mkt) 在高波动期
     → 衡量「市场高波动时股票是否同步下跌」
  3. 因子值 = -CCSK_up (高负协方=抗跌=正alpha)

简化版 (更稳定):
  CCSK = -corr(r_i · I(r_mkt<0), r_mkt²) × sign(r_i 在熊市中的平均收益)
  
  用correlation而非cov避免量纲问题，用sign信息保留方向
"""

import pandas as pd
import numpy as np
from pathlib import Path

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
    mkt_ret = mkt_ret['mkt_return'].reindex(ret_wide.index)
    
    mkt_sq = mkt_ret ** 2
    # Market high-vol threshold (60d MA of squared returns)
    mkt_sq_ma60 = mkt_sq.rolling(60, min_periods=40).mean()
    high_vol_mask = mkt_sq > mkt_sq_ma60
    
    # Sign of market return (1=up/-1=down) * high-vol indicator
    # When r_mkt < 0 AND |r_mkt| is large (high vol): negative contribution
    r_mkt_sign_vol = np.sign(mkt_ret.values) * (mkt_sq.values / (mkt_sq_ma60.values + 1e-10))
    
    dates = ret_wide.index
    stocks = ret_wide.columns.tolist()
    
    print(f"数据: {dates[0].date()} ~ {dates[-1].date()}")
    print(f"股票数: {len(stocks)}, 交易日: {len(dates)}")
    print(f"高波动日比例: {high_vol_mask.sum()}/{len(high_vol_mask)} ({100*high_vol_mask.mean():.1f}%)")
    
    # Compute factor
    window = 60
    min_p = 40
    results = []
    
    for idx, stock in enumerate(stocks):
        if (idx + 1) % 200 == 0:
            print(f"  处理 {idx+1}/{len(stocks)}...")
        
        stock_ret = ret_wide[stock].values.astype(float)
        
        # Rolling correlation: corr(r_i, sign(r_mkt) * |r_mkt|/~) over 60d
        # This is calculated in rolling windows
        factor_vals = np.full(len(dates), np.nan)
        
        s = pd.Series(stock_ret, index=range(len(dates)))
        m = pd.Series(r_mkt_sign_vol, index=range(len(dates)))
        
        for t in range(window-1, len(dates)):
            start = t - window + 1
            y = s.iloc[start:t+1].values
            x = m.iloc[start:t+1].values
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < min_p:
                continue
            y_c = y[mask]
            x_c = x[mask]
            
            # Correlation
            if np.std(y_c) < 1e-10 or np.std(x_c) < 1e-10:
                continue
            corr_val = np.corrcoef(y_c, x_c)[0, 1]
            if np.isfinite(corr_val):
                # Factor = -correlation (anticorrelated = good)
                factor_vals[t] = -corr_val
        
        # Store results
        for t in range(len(dates)):
            if np.isfinite(factor_vals[t]):
                results.append({
                    "date": dates[t],
                    "stock_code": stock,
                    "factor_raw": factor_vals[t]
                })
    
    print(f"\n原始因子值: {len(results)} 行")
    df_raw = pd.DataFrame(results)
    
    # --- 成交额中性化 ---
    print("合并成交额数据...")
    kline = pd.read_csv(data_dir / "csi1000_kline_raw.csv", dtype={'stock_code': str})
    kline['date'] = pd.to_datetime(kline['date'])
    g = kline.groupby('stock_code')
    kline['log_amt_20'] = g['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    amt_df = kline[['date', 'stock_code', 'log_amt_20']].dropna()
    
    df_raw['stock_code'] = df_raw['stock_code'].astype(str)
    amt_df['stock_code'] = amt_df['stock_code'].astype(str)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    amt_df['date'] = pd.to_datetime(amt_df['date'])
    
    df_raw = df_raw.merge(amt_df, on=['date', 'stock_code'], how='inner')
    df_raw = df_raw.dropna(subset=['factor_raw', 'log_amt_20'])
    print(f"合并后: {len(df_raw)} 行")
    
    # --- 截面OLS中性化 + MAD + z-score ---
    final_results = []
    for date, sub in df_raw.groupby('date'):
        if len(sub) < 50:
            continue
        
        y = sub['factor_raw'].values.astype(float)
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
        except:
            continue
        
        # MAD winsorize
        valid = np.isfinite(residual)
        if valid.sum() < 30:
            continue
        res_valid = residual[valid]
        med = np.nanmedian(res_valid)
        mad = np.nanmedian(np.abs(res_valid - med))
        if mad > 1e-10:
            residual = np.clip(residual, med - 5*1.4826*mad, med + 5*1.4826*mad)
        
        # z-score
        std = np.nanstd(residual)
        if std > 1e-10:
            residual = (residual - np.nanmean(residual)) / std
        
        sub_valid = sub.index[valid]
        for i, idx in enumerate(sub_valid):
            row = sub.loc[idx]
            final_results.append({
                "date": date,
                "stock_code": row['stock_code'],
                "factor_value": residual[valid][i]
            })
    
    result_df = pd.DataFrame(final_results).dropna(subset=['factor_value'])
    
    print(f"\n最终因子: {len(result_df)} 行, {result_df['date'].nunique()} 截面")
    print(f"因子分布: mean={result_df['factor_value'].mean():.4f}, "
          f"std={result_df['factor_value'].std():.4f}")
    
    output_file = data_dir / "factor_ccsk_v2.csv"
    result_df['date_str'] = result_df['date'].dt.strftime("%Y-%m-%d")
    result_df[['date_str', 'stock_code', 'factor_value']].rename(
        columns={'date_str': 'date'}
    ).to_csv(output_file, index=False)
    print(f"\n已保存: {output_file}")


if __name__ == "__main__":
    main()
