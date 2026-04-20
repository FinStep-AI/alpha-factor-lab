#!/usr/bin/env python3
"""
因子：Cond_Overnight v1 (Conditional Overnight Return after Negative Day)
论文启发：Song & Zhang (2022) MDPI "The Asymmetric Overnight Return Anomaly in the Chinese Stock Market"
方向：负面过度反应后的反转收益

构造：(20日内(日内跌日的隔夜收益均值)) → 正向 = 跌后恐慌→反转
关键：只统计日内下跌日的隔夜收益，日间上涨日的隔夜收益不考虑

注意：overnight_momentum_v1 已经入库，但他用了[(隔夜-日内)20日净值]
本因子角度不同：只看负面情境下的隔夜效果
"""
import numpy as np
import pandas as pd
import sys

def main():
    data_path = "data/csi1000_kline_raw.csv"
    output_path = "data/factor_cond_overnight_v1.csv"
    
    print(f"[信息] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    required = {"stock_code", "date", "open", "close"}
    missing = required - set(df.columns)
    if missing:
        print(f"[错误] 缺少列: {missing}")
        sys.exit(1)
    
    # Sort by stock_code then date
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Compute overnight return: (open_t - close_{t-1}) / close_{t-1}
    df["_prev_close"] = df.groupby("stock_code")["close"].shift(1)
    df["overnight_ret"] = (df["open"] - df["_prev_close"]) / df["_prev_close"]
    
    # Compute intraday return: (close_t - open_t) / open_t
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    
    # Boolean: was intraday return negative?
    df["is_down_day"] = (df["intraday_ret"] < 0).astype(float)
    
    # 20-day rolling: mean of overnight_ret where is_down_day==1
    window = 20
    group = df.groupby("stock_code")
    
    # Numerator: sum of overnight_ret on down days, rolling
    df["_down_overnight_sum"] = group["overnight_ret"].transform(
        lambda s: s.where(df.loc[s.index, "is_down_day"].astype(bool), np.nan)
                   .rolling(window, min_periods=15).sum()
    )
    
    # Denominator: count of down days, rolling
    df["_down_day_count"] = group["is_down_day"].transform(
        lambda s: s.rolling(window, min_periods=15).sum()
    )
    
    # Conditional overnight: mean overnight return on down days
    df["factor_raw"] = df["_down_overnight_sum"] / df["_down_day_count"]
    
    # Neutralize by transaction amount (proxy for market cap correlation)
    # use log_amount (approx log_market_cap)
    if "amount" in df.columns:
        df["_log_amount"] = np.log(df["amount"].replace(0, np.nan))
        df["_log_amount_20d"] = df.groupby("stock_code")["_log_amount"].transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )
    else:
        print("[警告] 无amount列, 跳过市值中性化")
        df["_log_amount_20d"] = 0
    
    # Cross-sectional neutralization (per date)
    def neutralize(group_df, col="factor_raw"):
        """OLS neutralize against log_amount_20d"""
        y = group_df[col].values.astype(float)
        x = group_df["_log_amount_20d"].values.astype(float)
        
        # Filter valid
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 5:
            return pd.Series(np.nan, index=group_df.index)
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        residual = np.full(len(group_df), np.nan)
        residual[mask] = y[mask] - X @ beta
        return pd.Series(residual, index=group_df.index)
    
    print("[信息] 横截面市值中性化...")
    df["factor_neutral"] = df.groupby("date", group_keys=False).apply(
        lambda g: neutralize(g, "factor_raw")
    )
    
    # MAD winsorize + z-score
    def mad_zscore(group_series):
        s = group_series.copy()
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return s
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        s = s.clip(lower, upper)
        std = s.std()
        if std > 0:
            s = (s - s.mean()) / std
        return s
    
    print("[信息] MAD Winsorize + Z-score...")
    df["factor_final"] = df.groupby("date")["factor_neutral"].transform(mad_zscore)
    
    # Output
    out = df[["date", "stock_code", "factor_final"]].copy()
    out.columns = ["date", "stock_code", "factor_cond_overnight_v1"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    
    fv = out["factor_cond_overnight_v1"]
    valid = fv.notna().sum()
    print(f"\n[结果] 因子值已保存: {output_path}")
    print(f"[统计] 有效值: {valid}/{len(fv)} ({valid/len(fv)*100:.1f}%)")
    print(f"[统计] 均值: {fv.mean():.4f}")
    print(f"[统计] 标准差: {fv.std():.4f}")
    
    # Quality check: how many stocks have at least 15 down-days in window?
    n_stocks_with_data = df[df["_down_day_count"] >= 15]["stock_code"].nunique()
    print(f"[质量] 有足够数据的股票: {n_stocks_with_data}/999")

if __name__ == "__main__":
    main()
