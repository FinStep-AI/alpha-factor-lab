#!/usr/bin/env python3
"""
因子：Overnight_Squeeze v1 - 修复版
核心：只统计跌日的隔夜收益均值，宽松门槛，不做过度处理
"""
import numpy as np
import pandas as pd
import sys

def main():
    data_path = "data/csi1000_kline_raw.csv"
    output_path = "data/factor_overnight_squeeze_v1.csv"
    
    print(f"[信息] 加载: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    group = df.groupby("stock_code")
    
    # overnight return
    df["prev_close"] = group["close"].shift(1)
    df["overnight_ret"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    
    # intraday return
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    
    # Mask: only use overnight returns on down days
    df["overnight_on_down"] = np.where(df["intraday_ret"] < 0, df["overnight_ret"], np.nan)
    
    # 20-day rolling mean with only 10 down days required
    print("[信息] 滚动计算20日均值 (min_periods=10)...")
    df["factor_raw"] = group["overnight_on_down"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    
    # Log amount for neutralization
    df["log_amt_20d"] = group["amount"].transform(
        lambda s: np.log(s.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    )
    
    n_valid_raw = df["factor_raw"].notna().sum()
    print(f"[统计] raw_valid: {n_valid_raw}/{len(df)} ({n_valid_raw/len(df)*100:.1f}%)")
    
    # Simple cross-sectional z-score (no neutralize first, too sparse)
    # Try neutralize for dates with enough cross-section
    def neutralize_and_std(group_df):
        y = group_df["factor_raw"].values
        x = group_df["log_amt_20d"].values
        mask = np.isfinite(y) & np.isfinite(x)
        
        if mask.sum() < 10:
            # Not enough to neutralize, just z-score
            valid_y = y[np.isfinite(y)]
            if len(valid_y) > 5:
                median = np.nanmedian(valid_y)
                std = np.nanstd(valid_y)
                if std > 1e-8:
                    y_new = (y - median) / std
                else:
                    y_new = y - median
                return pd.Series(y_new, index=group_df.index)
            return pd.Series(y, index=group_df.index)
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        residual = np.full(len(group_df), np.nan)
        residual[mask] = y[mask] - X @ beta
        
        # z-score
        valid_res = residual[np.isfinite(residual)]
        if len(valid_res) > 5:
            median = np.nanmedian(valid_res)
            std = np.nanstd(valid_res)
            if std > 1e-8:
                residual = (residual - median) / std
            else:
                residual  = residual - median
        return pd.Series(residual, index=group_df.index)
    
    print("[信息] 处理每日期截面...")
    df["factor_final"] = df.groupby("date", group_keys=False).apply(neutralize_and_std)
    
    out = df[["date", "stock_code", "factor_final"]].copy()
    out.columns = ["date", "stock_code", "factor_overnight_squeeze_v1"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    
    fv = out["factor_overnight_squeeze_v1"]
    n = fv.notna().sum()
    print(f"\n[结果] {output_path}")
    print(f"[统计] 有效: {n}/{len(fv)} ({n/len(fv)*100:.1f}%)")
    if n > 0:
        print(f"[统计] 均值={fv.mean():.4f}, std={fv.std():.4f}")
    
    # Per-date coverage
    cov = df.groupby("date")["factor_final"].notna().sum()
    print(f"[覆盖] 日均: {cov.mean():.0f}, 范围[{cov.min():.0f}-{cov.max():.0f}]")

if __name__ == "__main__":
    main()
