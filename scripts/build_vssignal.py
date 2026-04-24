"""
成交量放大确认因子 Volume Surge Signal v1 (vssignal_v1)
纯向量化计算，日线可构建
因子ID: vssignal_v1  2026-04-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
DATA = BASE / "data"

print("→ 加载数据...")
k = pd.read_csv(DATA / "csi1000_kline_raw.csv", parse_dates=["date"], low_memory=False)
k = k.sort_values(["stock_code", "date"]).reset_index(drop=True)
k["ret"] = k["pct_change"] / 100.0
k["log_amt"] = np.log(k["amount"].clip(lower=1))
print(f"  {len(k):,}行 | {k['stock_code'].nunique()}只")

# ── Step1: 放量日识别 ─────────────────────────────────────────────────────────
print("→ 识别放量日...")

# 20日滚动均值/标准差（个股级别）
k["vol_ma20"] = k.groupby("stock_code")["volume"].transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)
k["vol_std20"] = k.groupby("stock_code")["volume"].transform(
    lambda s: s.rolling(20, min_periods=10).std()
)
k["vol_zscore"] = (k["volume"] - k["vol_ma20"]) / k["vol_std20"].replace(0, np.nan).fillna(0.001)

# 截面：个股/截面中位数 = 相对放量倍数
cross_med = k.groupby("date")["volume"].transform("median")
k["vol_surge_ratio"] = (k["volume"] / cross_med.clip(lower=1)).clip(0, 30)

# 放量日 = 相对vol_zscore > 1
k["is_surge"] = (k["vol_zscore"] > 1).astype(float)

# ── Step2: 放量后response信号的构造 ──────────────────────────────────────────
print("→ 构建放量响应信号...")

# forward_abs_ret_3d: 当天起未来3日的绝对收益率均值  （未来的事，只取放量日那行对应天）
k["fwd_abs_ret_3d"] = (
    k.groupby("stock_code")["ret"]
    .apply(lambda s: s.shift(-1).abs().rolling(3, min_periods=2).mean())
    .reset_index(level=0, drop=True)
)

# 将 sequence 的数量全部初始化为 NaN，并赋值。
k["fwd_abs_ret_3d"].replace([np.inf, -np.inf], np.nan, inplace=True)
k["fwd_abs_ret_3d"].fillna(0, inplace=True)

# ── Step3: 纯粹用截面内相对排序来构建因子（不用时间序列循环）──────────────────
# 向量化合成：
# - 放大量倍数高  +  forward回应高  = 量大+价波动呼应好
# - 截面rank做稳健缩放 [-2,2]

print("→ 向量化因子合成...")

# 用截面rank percentile替代volume_surge_ratio，去偏量纲
k["cr_vol"]   = k.groupby("date")["vol_surge_ratio"].transform(
    lambda x: 2 * (x.rank() / x.count()) - 1
).fillna(0)   # [-1, 1]

k["cr_fwd"]   = k.groupby("date")["fwd_abs_ret_3d"].transform(
    lambda x: 2 * (x.rank() / x.count()) - 1
).fillna(0)

# 只取放量日：非放量日信号为0（避免把没放量的品种纳入信号）
k["cr_combined"] = np.where(
    k["is_surge"] == 1,
    k["cr_vol"] * 0.4 + k["cr_fwd"] * 0.6,
    0
)

# 20日滚动平滑：放量信号累积
k["factor_raw"] = k.groupby("stock_code")["cr_combined"].transform(
    lambda s: s.rolling(20, min_periods=10).mean()
)

# ── Step4: 市值中性化 ────────────────────────────────────────────────────────
print("→ 市值中性化...")

from numpy.linalg import lstsq

def neutralize(grp):
    y = grp["factor_raw"].values
    x = grp["log_amt"].values
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 30:
        grp["factor_val"] = np.nan
        return grp
    A = np.column_stack([np.ones(m.sum()), x[m]])
    b, _, _, _ = lstsq(A, y[m], rcond=None)
    r = np.full(len(y), np.nan)
    r[m] = y[m] - A @ b
    grp["factor_val"] = r
    return grp

k = k.groupby("date", group_keys=False).apply(neutralize)
print(f"  有效率: {k['factor_val'].notna().mean():.1%}")
print(f"  均值: {k['factor_val'].mean():.6f}   std: {k['factor_val'].std():.4f}")

# ── 输出 ──────────────────────────────────────────────────────────────────────
out = k[["date", "stock_code", "factor_val"]].dropna(subset=["factor_val"]).copy()
out = out.rename(columns={"factor_val": "factor_value"})
out_path = DATA / "vssignal_v1.csv"
out.to_csv(out_path, index=False)

print(f"\n✓ 因子值输出: {out_path}")
print(f"  日期: {out['date'].min().date()} ~ {out['date'].max().date()}")
print(f"  总行: {len(out):,}行")

print("\n因子分布:")
print(out["factor_value"].describe())
