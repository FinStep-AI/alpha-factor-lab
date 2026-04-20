#!/usr/bin/env python3
"""
因子：Overnight_Drop_Intensity v1 (ODI)
灵感：Paper "Asymmetric Overnight Return Anomaly" (MDPI 2022)
论文发现：A股日内跌→隔夜继续跌（散户恐慌），但幅度因股而异

构造方法：
1. 只考虑intraday_return < 0的天
2. 计算 overnight_ret / intraday_ret → 隔夜跌幅占日内跌幅的比例
3. 20日滚动均值
4. 正向：高比率=隔夜跟跌幅度大=散户恐慌→反转？

但避免 min_periods 门槛问题：宽松到 min_periods=8 (20天8天)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    data_path = "data/csi1000_kline_raw.csv"
    output_path = "data/factor_overnight_intensity_v1.csv"
    
    print(f"[信息] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    required = {"stock_code", "date", "open", "close"}
    if missing := required - set(df.columns):
        print(f"[错误] 缺少列: {missing}")
        sys.exit(1)
    
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Compute overnight and intraday returns
    group = df.groupby("stock_code")
    df["_prev_close"] = group["close"].shift(1)
    df["overnight_ret"] = (df["open"] - df["_prev_close"]) / df["_prev_close"]
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    df["_log_amt_20d"] = group["amount"].transform(
        lambda s: np.log(s.replace(0, np.nan)).rolling(20, min_periods=12).mean()
    )
    
    # On down days: overnight/intraday (both negative → positive ratio)
    # On up days: NaN (ignore)
    # Logic: overnight_ret / intraday_ret when intraday_ret < 0
    # 比率越接近0：隔夜大幅跟跌（恐慌）
    # 比率越接近1：隔夜反弹（抄底买入）
    mask_down = df["intraday_ret"] < 0
    df["drop_ratio"] = np.where(mask_down, 
                                 df["overnight_ret"] / df["intraday_ret"], 
                                 np.nan)
    
    # Clip to [0, 2] to filter outliers
    df["drop_ratio"] = df["drop_ratio"].clip(0, 2)
    
    window = 20
    min_p = 8  # 20天8天下跌日已经很宽松
    df["factor_raw"] = group["drop_ratio"].transform(
        lambda s: s.rolling(window, min_periods=min_p).mean()
    )
    
    print(f"[统计] raw valid: {df['factor_raw'].notna().sum()}/{len(df)}")
    
    # Neutralize
    valid_mask = np.isfinite(df["factor_raw"].values) & np.isfinite(df["_log_amt_20d"].values)
    print(f"[统计] neutralizable: {valid_mask.sum()}")
    if valid_mask.sum() < 100:
        print("[警告] 可中性化数据太少，尝试跳过中性化直接归一化")
        df["factor_final"] = df["factor_raw"]
    else:
        def ols_residual(y, x, idx):
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 5:
                return np.full(len(y), np.nan)
            X = np.column_stack([np.ones(mask.sum()), x[mask]])
            beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
            res = np.full(len(y), np.nan)
            res[mask] = y[mask] - X @ beta
            return pd.Series(res, index=idx)
        
        print("[信息] 横截面市值中性化...")
        df["_fn"] = df.groupby("date", group_keys=False).apply(
            lambda g: ols_residual(g["factor_raw"].values, g["_log_amt_20d"].values, g.index)
        )
        
        # MAD + z-score
        def mad_zscore(s):
            s = s.copy()
            med = s.median()
            mad = (s - med).abs().median()
            if mad == 0 or np.isnan(mad):
                return s
            s = s.clip(med - 5*1.4826*mad, med + 5*1.4826*mad)
            std = s.std()
            if std > 0:
                s = (s - s.mean()) / std
            return s
        
        print("[信息] MAD Winsorize + Z-score...")
        df["factor_final"] = df.groupby("date")["_fn"].transform(mad_zscore)
    
    # Output
    out = df[["date", "stock_code", "factor_final"]].copy()
    out.columns = ["date", "stock_code", "factor_overnight_intensity_v1"]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    
    fv = out["factor_overnight_intensity_v1"]
    valid_n = fv.notna().sum()
    print(f"\n[结果] 因子值: {output_path}")
    print(f"[统计] 有效值: {valid_n}/{len(fv)} ({valid_n/len(fv)*100:.1f}%)")
    if valid_n > 0:
        print(f"[统计] 均值: {fv.mean():.4f}, 标准差: {fv.std():.4f}")
    
    # Coverage stats
    n_dates = df["date"].nunique()
    n_stocks_per_date = df.groupby("date")["factor_final"].notna().sum()
    print(f"[覆盖] 日均有效数: {n_stocks_per_date.mean():.0f}, "
          f"范围[{n_stocks_per_date.min():.0f}-{n_stocks_per_date.max():.0f}]")

if __name__ == "__main__":
    main()
