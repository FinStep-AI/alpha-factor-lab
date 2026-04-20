#!/usr/bin/env python3
"""Volume-Price Efficiency Factor Calculator v2

核心思路：放量日的平均收益率 vs 缩量日的平均收益率差值（波动率标准化）。
- 正向使用：高值 = 放量日涨跌幅显著优于缩量日 = 信息在交易活跃日集中释放

分子：20日窗口内成交量top 1/3天平均回报 - 成交量bottom 1/3天平均回报
分母：20日窗口内回报标准差（波动率标准化）
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def calc_factor():
    df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # Daily return
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df = df.dropna(subset=['ret']).reset_index(drop=True)

    window = 20
    results = []

    for stock, g in df[['date', 'stock_code', 'volume', 'ret']].groupby('stock_code'):
        g = g.sort_values('date').reset_index(drop=True)
        n = len(g)
        if n < window + 5:
            continue

        dates = g['date'].values
        volumes = g['volume'].values
        rets = g['ret'].values

        vals = []
        for i in range(window - 1, n):
            v_win = volumes[i - window + 1 : i + 1]
            r_win = rets[i - window + 1 : i + 1]

            # Sort by volume to get tertile breakpoints
            if np.sum(~np.isnan(v_win)) < window * 0.9:
                vals.append(np.nan)
                continue

            valid_mask = ~np.isnan(v_win)
            v_valid = v_win[valid_mask]
            r_valid = r_win[valid_mask]

            if len(v_valid) < window // 2:
                vals.append(np.nan)
                continue

            # Tertile breakpoints
            try:
                q1 = np.percentile(v_valid, 33.33)
                q3 = np.percentile(v_valid, 66.66)
            except:
                vals.append(np.nan)
                continue

            high_mask = v_valid >= q3
            low_mask = v_valid <= q1

            if high_mask.sum() < 2 or low_mask.sum() < 2:
                vals.append(np.nan)
                continue

            r_high = np.mean(r_valid[high_mask])
            r_low = np.mean(r_valid[low_mask])
            sigma = np.std(r_valid)

            if sigma < 1e-8:
                vals.append(np.nan)
                continue

            raw_val = (r_high - r_low) / sigma
            vals.append(raw_val)

        if vals:
            g_out = g.iloc[window - 1:].copy()
            g_out['factor_raw'] = vals[:len(g_out)]
            g_out = g_out.dropna(subset=['factor_raw'])
            if len(g_out) > 0:
                results.append(g_out[['date', 'stock_code', 'factor_raw']])

    if not results:
        print("ERROR: No factor values")
        return

    df_factor = pd.concat(results, ignore_index=True)

    # Log transform
    df_factor['factor_log'] = np.log(df_factor['factor_raw'].clip(lower=0.001) + 0.001 + 0.02)

    # Market cap proxy: log(20d avg amount)
    df_amount = df[['date', 'stock_code', 'amount']].copy()
    df_amount['log_amount_20d'] = df_amount.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    df_factor = df_factor.merge(df_amount[['date', 'stock_code', 'log_amount_20d']], on=['date', 'stock_code'], how='left')
    df_factor = df_factor.dropna(subset=['factor_log', 'log_amount_20d'])

    # OLS neutralize
    output_rows = []
    for date, day_df in df_factor.groupby('date'):
        d = day_df.copy()
        if len(d) < 100:
            continue
        X = np.column_stack([np.ones(len(d)), d['log_amount_20d'].values])
        y = d['factor_log'].values
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            d['factor_neut'] = y - X @ beta
        except:
            continue

        med = d['factor_neut'].median()
        mad = (d['factor_neut'] - med).abs().median() * 1.4826
        if mad < 1e-6:
            continue
        d['factor_neut'] = d['factor_neut'].clip(med - 5.2*mad, med + 5.2*mad)

        std = d['factor_neut'].std()
        if std < 1e-6:
            continue
        d['factor_value'] = (d['factor_neut'] - d['factor_neut'].mean()) / std
        output_rows.append(d[['date', 'stock_code', 'factor_value']])

    if not output_rows:
        print("ERROR: No final rows")
        return

    result = pd.concat(output_rows, ignore_index=True)
    outpath = DATA_DIR / "factor_pv_efficiency_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")
    print(f"   Dates: {result['date'].min()} ~ {result['date'].max()}")
    print(f"   Stocks: {result['stock_code'].nunique()}")
    print(f"   stats: mean={result['factor_value'].mean():.4f}, std={result['factor_value'].std():.4f}")


if __name__ == "__main__":
    calc_factor()
