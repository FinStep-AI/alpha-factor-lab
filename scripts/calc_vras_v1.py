#!/usr/bin/env python3
"""
Volume-Return Asymmetry Strength v1 (VRAS v1)
Vectorized v2: 使用pandas rolling rank + rolling corr
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    output_file = data_dir / "factor_vras_v1.csv"
    
    print("加载数据...")
    df = pd.read_csv(kline_file, dtype={"stock_code": str}, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    n_stocks = df["stock_code"].nunique()
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {n_stocks}只股票")
    
    WINDOW = 20
    MIN_PERIODS = 10
    
    # Precompute |ret| 
    print("预处理: |ret|...")
    df["abs_ret"] = df["pct_change"].abs()
    df = df.dropna(subset=["abs_ret"])
    df = df[df["abs_ret"] > 0].copy()  # 避免rank degeneracy
    
    # 横截面rank (每日rank, 使跨截面可比)
    print("计算每日横截rank(|ret|) and rank(volume)...")
    df["rank_abs_ret"] = df.groupby("date")["abs_ret"].rank(pct=True)
    df["rank_volume"] = df.groupby("date")["volume"].rank(pct=True)
    
    print("pivot to matrices...")
    rank_ret_matrix = df.pivot_table(index="date", columns="stock_code", values="rank_abs_ret")
    rank_vol_matrix = df.pivot_table(index="date", columns="stock_code", values="rank_volume")
    
    print("计算滚动20日 Rolling Spearman Corr...")
    # Spearman = Pearson of ranks. We have daily rank matrices.
    # Rolling Pearson on rank matrices:
    
    # Efficient rolling Pearson using exponentially-weighted covariance
    # But simpler: just use pd.Series.rolling().corr() for each stock
    
    # Since we already have rank-transformed data, Pearson corr on ranks = Spearman
    vras_list = []
    
    for i, stock in enumerate(rank_ret_matrix.columns):
        if (i+1) % 200 == 0:
            print(f"  股票 {i+1}/{len(rank_ret_matrix.columns)} ({stock})...")
        
        r_ret = rank_ret_matrix[stock]
        r_vol = rank_vol_matrix[stock]
        
        # Rolling Pearson = Spearman (since we already ranked)
        rolling_corr = r_ret.rolling(WINDOW, min_periods=MIN_PERIODS).corr(r_vol)
        
        tmp = pd.DataFrame({
            "date": r_ret.index,
            "stock_code": stock,
            "vras_raw": rolling_corr.values
        })
        vras_list.append(tmp)
    
    result_df = pd.concat(vras_list, ignore_index=True)
    result_df = result_df.dropna(subset=["vras_raw"])
    
    print(f"\n原始VRAS: {len(result_df)} records")
    print(f"日期: {result_df['date'].min().date()} ~ {result_df['date'].max().date()}")
    print(f"VRAS分布: mean={result_df['vras_raw'].mean():.4f}, std={result_df['vras_raw'].std():.4f}")
    
    # ========================================
    # Neutralization: log(20d avg amount) OLS
    # ========================================
    print("\n中性化处理...")
    
    WINDOW_AMT = 20
    amt_df = df[["date", "stock_code", "amount"]].copy()
    amt_df["log_amount_20d"] = amt_df.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(WINDOW_AMT, min_periods=10).mean() + 1)
    )
    
    result_df = result_df.merge(
        amt_df[["date", "stock_code", "log_amount_20d"]],
        on=["date", "stock_code"],
        how="left"
    )
    result_df = result_df.dropna(subset=["vras_raw", "log_amount_20d"])
    
    t0 = time.time()
    final_records = []
    dates_done = 0
    
    for date, group in result_df.groupby("date"):
        if len(group) < 50:
            continue
        dates_done += 1
        
        y = group["vras_raw"].values.astype(float)
        x = group["log_amount_20d"].values.astype(float)
        
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        try:
            beta = np.linalg.solve(X.T @ X + 1e-10 * np.eye(2), X.T @ y[mask])
            residual = y[mask] - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.nanmedian(residual)
        mad = np.nanmedian(np.abs(residual - med))
        if mad > 1e-10:
            scaled = 1.4826 * mad
            residual = np.clip(residual, med - 3*scaled, med + 3*scaled)
        
        # z-score
        std = np.nanstd(residual)
        if std > 1e-10:
            residual = (residual - np.nanmean(residual)) / std
        
        sub = group.iloc[np.where(mask)[0]].copy()
        sub.loc[:, "factor_value"] = residual
        final_records.append(sub[["date", "stock_code", "factor_value"]])
    
    print(f" Neutralization: {dates_done} dates in {time.time()-t0:.1f}s")
    
    if not final_records:
        print("ERROR: Neutralization failed!")
        return
    
    output_df = pd.concat(final_records, ignore_index=True)
    output_df = output_df.dropna(subset=["factor_value"])
    output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime("%Y-%m-%d")
    
    print(f"\n✓ 最终因子:")
    print(f"  日期: {output_df['date'].min()} ~ {output_df['date'].max()}")
    print(f"  记录: {len(output_df)}")
    print(f"  截面数: {output_df['date'].nunique()}")
    print(f"  值分布: mean={output_df['factor_value'].mean():.4f}, std={output_df['factor_value'].std():.4f}")
    print(f"  范围: [{output_df['factor_value'].min():.4f}, {output_df['factor_value'].max():.4f}]")
    
    daily_n = output_df.groupby("date")["stock_code"].count()
    print(f"  日均覆盖: {daily_n.mean():.0f}只")
    
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ 保存: {output_file}")

if __name__ == "__main__":
    main()
