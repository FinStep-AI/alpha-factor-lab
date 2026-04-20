#!/usr/bin/env python3
"""
因子：Vol_Ret_Weight_v1 (Volume-Weighted Return)

方向：Volume-as-Signal (Lee & Swaminathan 2000; Gervais et al. 2001)
核心思想：成交量加权的累计收益率，分离"低量反弹"(噪音) vs "放量上涨"(真实信息)

构造方法：
1. 20日窗口
2. 计算日均收益率中，成交量加权的方向偏度
3. yield = (Σ(volume_i × return_i) / Σ(volume_i)) × 20
   → 等价于成交量加权的累计收益率
4. 交叉截面：市值中性化 + z-score

预期：高VWR → 成交量集中在上涨日 → 正向后5日收益（知情交易溢价）
"""
import numpy as np
import pandas as pd
import sys

def main():
    data_path = "data/csi1000_kline_raw.csv"
    output_path = "data/factor_vol_ret_weight_v1.csv"
    
    print(f"[信息] 加载: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Daily return
    df["ret"] = df["close"].pct_change()
    
    # Volume-weighted cumulative return over 20 days
    window = 20
    group = df.groupby("stock_code")
    
    # Numerator: rolling sum of (volume × return)
    df["vol_x_ret"] = df["amount"] * df["ret"]
    df["numer"] = group["vol_x_ret"].transform(
        lambda s: s.rolling(window, min_periods=15).sum()
    )
    
    # Denominator: rolling sum of volume
    df["denom"] = group["amount"].transform(
        lambda s: s.rolling(window, min_periods=15).sum()
    )
    
    # Raw factor: volume-weighted average return × window
    # = Σ(vol×ret)/Σ(vol) × window
    # Equivalent to: weighted cumulative return
    df["factor_raw"] = (df["numer"] / df["denom"]) * window
    
    # Log amount for neutralize
    df["log_amt_20d"] = group["amount"].transform(
        lambda s: np.log(s.replace(0, np.nan)).rolling(20, min_periods=12).mean()
    )
    
    n_valid_raw = df["factor_raw"].notna().sum()
    print(f"[统计] raw_valid: {n_valid_raw}/{len(df)} ({n_valid_raw/len(df)*100:.1f}%)")
    
    def per_date_process(group_df):
        y = group_df["factor_raw"].values.astype(float)
        x = group_df["log_amt_20d"].values.astype(float)
        
        valid_y = y[np.isfinite(y)]
        if len(valid_y) < 20:
            return pd.Series(np.nan, index=group_df.index)
        
        # Cross-sectional neutralize: divide by std (market-cap neutralized proxy)
        median = np.nanmedian(valid_y)
        std = np.nanstd(valid_y)
        if std < 1e-8:
            return pd.Series(np.nan, index=group_df.index)
        z = (y - median) / std
        
        # Also do OLS neutralize if enough cross-section
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 20:
            return pd.Series(z, index=group_df.index)
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        residual = np.full(len(y), np.nan)
        residual[mask] = y[mask] - X @ beta
        
        # z-score residual
        r_std = np.nanstd(residual)
        if r_std < 1e-8:
            return pd.Series(z, index=group_df.index)
        residual = (residual - np.nanmedian(residual)) / r_std
        return pd.Series(residual, index=group_df.index)
    
    print("[信息] 截面处理...")
    df["factor_final"] = df.groupby("date", group_keys=False).apply(per_date_process)
    
    out = df[["date", "stock_code", "factor_final"]].copy()
    out.columns = ["date", "stock_code", "factor_vol_ret_weight_v1"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    
    fv = out["factor_vol_ret_weight_v1"]
    n = fv.notna().sum()
    print(f"\n[结果] {output_path}")
    print(f"[统计] 有效: {n}/{len(fv)} ({n/len(fv)*100:.1f}%)")
    if n > 0:
        print(f"[统计] 均值={fv.mean():.4f}, std={fv.std():.4f}")

if __name__ == "__main__":
    main()
