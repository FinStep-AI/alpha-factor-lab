#!/usr/bin/env python3
"""
因子：Volume_Extreme_Return_Contrast v1
来源灵感：学术文献中"volume-driven price signals"

构造：20日滚动, 按当日成交额分位数(70%分位 vs 30%分位)
计算极端高量日vs极端低量日的未来收益差
factor_raw = avg(forward_return where volume in top 30%) - avg(forward_return where vol in bottom 30%)

逻辑：高量日=信息驱动日，其方向能预测后续；低量日=噪音日，信号弱
预期：高量日收益>低量日收益 →正值因子 →做多G5(高量+信号强) / 做空G1(低量+信号弱)
"""
import numpy as np
import pandas as pd
import sys

def main():
    data_path = "data/csi1000_kline_raw.csv"
    output_path = "data/factor_vol_extreme_contrast_v1.csv"
    
    print(f"[信息] 加载: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Daily return (for computing forward return contrast)
    df["ret"] = df["close"].pct_change()
    
    # Forward return (5-day) computed later, but we need to align properly
    # Actually we'll compute the volume quantile each day and correlate with future 5d return
    group = df.groupby("stock_code")
    
    # 20-day rolling volume percentiles
    window = 20
    high_q = 0.7   # top 30%
    low_q = 0.3    # bottom 30%
    min_p = 15      # minimum observations for rolling computations
    
    # Rolling quantile of volume (20-day)
    print("[信息] 计算20日成交额分位数...")
    df["vol_70pct"] = group["amount"].transform(
        lambda s: s.rolling(window, min_periods=min_p).quantile(high_q)
    )
    df["vol_30pct"] = group["amount"].transform(
        lambda s: s.rolling(window, min_periods=min_p).quantile(low_q)
    )
    
    # Flag: is today in top 30% or bottom 30% volume?
    df["is_high_vol"] = (df["amount"] >= df["vol_70pct"]).astype(float)
    df["is_low_vol"] = (df["amount"] <= df["vol_30pct"]).astype(float)
    
    # 5-day forward return
    df["ret_fwd5d"] = group["ret"].transform(
        lambda s: s.rolling(5).sum().shift(-5)  # 5-day forward return
    )
    
    # Rolling: avg forward return on high-vol days minus avg on low-vol days
    print("[信息] 滚动计算极端量日收益差...")
    df["high_vol_fwd_ret"] = group["ret_fwd5d"].transform(
        lambda s: s.where(df.loc[s.index, "is_high_vol"].astype(bool), np.nan)
                   .rolling(window, min_periods=10).mean()
    )
    
    df["low_vol_fwd_ret"] = group["ret_fwd5d"].transform(
        lambda s: s.where(df.loc[s.index, "is_low_vol"].astype(bool), np.nan)
                   .rolling(window, min_periods=10).mean()
    )
    
    # Contrast = high_vol_fwd - low_vol_fwd
    # Positive means: high-volume days predict better forward returns → information hypothesis
    df["factor_raw"] = df["high_vol_fwd_ret"] - df["low_vol_fwd_ret"]
    
    # Log amount 20d for neutralize
    df["log_amt_20d"] = group["amount"].transform(
        lambda s: np.log(s.replace(0, np.nan)).rolling(20, min_periods=12).mean()
    )
    
    n_valid = df["factor_raw"].notna().sum()
    print(f"[统计] raw_valid: {n_valid}/{len(df)} ({n_valid/len(df)*100:.1f}%)")
    
    # Cross-sectional z-score (per date)
    def per_date_zscore(g):
        y = g["factor_raw"].values.astype(float)
        valid = np.isfinite(y)
        if valid.sum() < 20:
            return pd.Series(np.nan, index=g.index)
        v = y[valid]
        med = np.nanmedian(v)
        std = np.nanstd(v)
        if std < 1e-8:
            return pd.Series(np.nan, index=g.index)
        result = np.full(len(y), np.nan)
        result[valid] = (v - med) / std
        return pd.Series(result, index=g.index)
    
    print("[信息] 截面z-score...")
    df["factor_final"] = df.groupby("date", group_keys=False).apply(per_date_zscore)
    
    out = df[["date", "stock_code", "factor_final"]].copy()
    out.columns = ["date", "stock_code", "factor_vol_extreme_contrast_v1"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    
    fv = out["factor_vol_extreme_contrast_v1"]
    n = fv.notna().sum()
    print(f"\n[结果] {output_path}")
    print(f"[统计] 有效: {n}/{len(fv)} ({n/len(fv)*100:.1f}%)")
    if n > 0:
        print(f"[统计] 均值={fv.mean():.4f}, std={fv.std():.4f}")

if __name__ == "__main__":
    main()
