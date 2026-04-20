#!/usr/bin/env python3
"""
因子：Overnight_Asym_v1 (Overnight Asymmetry Ratio)
论文启发：Song & Zhang (2022) MDPI "Asymmetric Overnight Return Anomaly"

构造：20日滚动 (E[overnight|down_day] - E[overnight|up_day]) / std(all_overnight, 20d)
含义：隔夜对负面消息的反应比正面消息放大了多少

逻辑：A股散户对负面消息过度反应（恐慌抛售隔夜更惨），这种不对称性越大：
- 方向1（反转）：过度反应→后续反弹 → 正向因子
- 方向2（信息/质量）：反映信息不对称严重 → 高不对称=差质量 → 负向因子

先测试原始方向（高比率=正向），如无效再反向。
"""
import numpy as np
import pandas as pd
import sys

def compute_overnight_asym(df_raw, window=20, min_periods=15):
    """Compute (mean overnight on down days - mean overnight on up days) / std"""
    df = df_raw.copy()
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Overnight return
    df["_prev_close"] = df.groupby("stock_code")["close"].shift(1)
    df["overnight_ret"] = (df["open"] - df["_prev_close"]) / df["_prev_close"]
    
    # Intraday return (for dividing into down/up days)
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    
    group = df.groupby("stock_code")
    
    # Mean overnight on down days (intraday_ret < 0)
    df["_ovn_down_sum"] = group["overnight_ret"].transform(
        lambda s: s.where(df.loc[s.index, "intraday_ret"] < 0, np.nan)
                   .rolling(window, min_periods=min_periods).mean()
    )
    
    # Mean overnight on up days (intraday_ret >= 0)
    df["_ovn_up_sum"] = group["overnight_ret"].transform(
        lambda s: s.where(df.loc[s.index, "intraday_ret"] >= 0, np.nan)
                   .rolling(window, min_periods=min_periods).mean()
    )
    
    # Std of all overnight (for normalization)
    df["_ovn_std"] = group["overnight_ret"].transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    
    # Asymmetry ratio
    df["factor_raw"] = (df["_ovn_down_sum"] - df["_ovn_up_sum"]) / df["_ovn_std"]
    return df

def neutralize_and_zscore(df, factor_col="factor_raw",
                          neutralize_col="_log_amt_20d",
                          output_col="factor_final"):
    """Cross-sectional OLS neutralize + MAD + z-score per date"""
    
    if "amount" in df.columns and neutralize_col not in df.columns:
        df["_log_amt"] = np.log(df["amount"].replace(0, np.nan))
        df[neutralize_col] = df.groupby("stock_code")["_log_amt"].transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )
    
    def ols_residual(y, x):
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 5:
            return np.full_like(y, np.nan)
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        res = np.full_like(y, np.nan)
        res[mask] = y[mask] - X @ beta
        return res
    
    print("[信息] 横截面市值中性化...")
    df["_neutralized"] = (
        df.groupby("date", group_keys=False)
          .apply(lambda g: pd.Series(
              ols_residual(g[factor_col].values.astype(float),
                           g[neutralize_col].values.astype(float)),
              index=g.index
          ))
    )
    
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
    df[output_col] = df.groupby("date")["_neutralized"].transform(mad_zscore)
    return df

def main():
    data_path = "data/csi1000_kline_raw.csv"
    output_path = "data/factor_overnight_asym_v1.csv"
    
    print(f"[信息] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    print("[信息] 计算隔夜不对称比率...")
    df = compute_overnight_asym(df, window=20, min_periods=15)
    
    df = neutralize_and_zscore(df, factor_col="factor_raw",
                               output_col="factor_overnight_asym_v1")
    
    out = df[["date", "stock_code", "factor_overnight_asym_v1"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)
    
    fv = out["factor_overnight_asym_v1"]
    valid = fv.notna().sum()
    print(f"\n[结果] 因子值: {output_path}")
    print(f"[统计] 有效值: {valid}/{len(fv)} ({valid/len(fv)*100:.1f}%)")
    print(f"[统计] 均值: {fv.mean():.4f}, 标准差: {fv.std():.4f}")

if __name__ == "__main__":
    main()
