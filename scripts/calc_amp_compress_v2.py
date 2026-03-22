"""
因子: 振幅压缩改进版 (Amplitude Compression v2, amp_compress_v2)
改进: 原v1用MA5/MA20, mono=0.7; v2尝试多个窗口+方向组合
      构造: -log(MA_short_amplitude / MA_long_amplitude)
      正向: 高因子值=短期振幅远小于长期=波动率压缩=蓄势
      
逻辑: 股票经历波动率压缩后(短期振幅<长期均值), 往往面临方向选择
      在截面上, 波动率压缩的股票可能吸引趋势跟踪资金
      与TAE(换手率/振幅)不同: TAE关注换手-振幅比, 本因子关注振幅时间结构
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", parse_dates=["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# amplitude已有
kline["amplitude_pct"] = kline["amplitude"]  # 振幅百分比

print(f"数据: {kline.shape[0]} 行, {kline['stock_code'].nunique()} 只股票")

# ── 测试多个窗口组合 ──────────────────────────────────────
configs = [
    (3, 20, "3_20"),
    (5, 40, "5_40"),
    (3, 40, "3_40"),
    (5, 20, "5_20"),  # 原版参考
]

best_config = None
best_result = None

for short_w, long_w, label in configs:
    print(f"\n{'='*60}")
    print(f"测试窗口: MA{short_w} / MA{long_w}")
    print(f"{'='*60}")
    
    # 计算短期和长期平均振幅
    kline[f"ma_amp_s"] = kline.groupby("stock_code")["amplitude_pct"].transform(
        lambda x: x.rolling(short_w, min_periods=max(2, short_w//2)).mean()
    )
    kline[f"ma_amp_l"] = kline.groupby("stock_code")["amplitude_pct"].transform(
        lambda x: x.rolling(long_w, min_periods=max(5, long_w//2)).mean()
    )
    
    # 压缩比率 (取负: 高值=更压缩)
    kline["raw"] = -np.log(kline["ma_amp_s"] / kline["ma_amp_l"].clip(lower=0.01))
    
    # 成交额中性化
    kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize_ols(df):
        mask = df["raw"].notna() & df["log_amount_20d"].notna() & np.isfinite(df["raw"])
        if mask.sum() < 30:
            df["factor"] = np.nan
            return df
        y = df.loc[mask, "raw"].values
        x = df.loc[mask, "log_amount_20d"].values
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
        except:
            df["factor"] = np.nan
            return df
        df["factor"] = np.nan
        df.loc[mask, "factor"] = resid
        return df
    
    kline = kline.groupby("date", group_keys=False).apply(neutralize_ols)
    
    # MAD + z-score
    def mad_zscore(df):
        vals = df["factor"]
        mask = vals.notna()
        if mask.sum() < 30:
            df["factor"] = np.nan
            return df
        med = vals[mask].median()
        mad = (vals[mask] - med).abs().median() * 1.4826
        if mad < 1e-10:
            df["factor"] = np.nan
            return df
        lower = med - 3 * mad
        upper = med + 3 * mad
        clipped = vals.clip(lower, upper)
        mu = clipped[mask].mean()
        sigma = clipped[mask].std()
        if sigma < 1e-10:
            df["factor"] = np.nan
            return df
        df["factor"] = (clipped - mu) / sigma
        return df
    
    kline = kline.groupby("date", group_keys=False).apply(mad_zscore)
    
    output = kline[["date", "stock_code", "factor"]].dropna(subset=["factor"])
    output = output.rename(columns={"factor": f"amp_compress_{label}"})
    
    out_path = DATA_DIR / f"factor_amp_compress_{label}.csv"
    output.to_csv(out_path, index=False)
    print(f"  输出: {out_path}, {len(output)} 行")
    print(f"  分布: mean={output.iloc[:,-1].mean():.4f}, std={output.iloc[:,-1].std():.4f}")

print("\n所有窗口计算完成。接下来回测各窗口。")
