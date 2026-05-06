"""
gap_reversal_strength_v1: Gap+Close-Location复合因子

  1. gap = open/prev_close - 1          # 隔夜跳空
  2. close_recov = close/open - 1        # 日内反转幅度
  3. gap_rev = -gap * close_recov_gap  # 低开→弱反弹→负值均衡
  4. 10日 rolling mean → 20d MA成交额OLS中性化

  正向使用：复合值越高→低开+日内风格强烈→后续出现正alpha
"""

import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. 读K线数据 ────────────────────────────────────────────────────────────────
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", parse_dates=["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ── 2. gap = open/prev_close - 1 ───────────────────────────────────────────────
kline["prev_close"] = kline.groupby("stock_code")["close"].shift(1)
kline["gap"] = kline["open"] / kline["prev_close"] - 1

# ── 3. 日内修复 close_recov = close/open - 1 ─────────────────────────────────
kline["close_recov"] = kline["close"] / kline["open"] - 1

# ── 4. 复合：gap_rev = -(gap - recov) ──────────────────────────────────────────
# gap越小（负越多）→ 低开 → 看close_recov是否足够
# gap_rev越高: gap很负但close_recov也很负→gap_rev低（未反弹）
# gap_rev低: gap很负但close_recov为正→已修复
kline["gap_compound"] = - (kline["gap"] - kline["close_recov"])

# ── 5. 10日 rolling mean ───────────────────────────────────────────────────────
kline["factor_raw"] = (
    kline.groupby("stock_code")["gap_compound"]
    .transform(lambda s: s.rolling(10, min_periods=7).mean())
)

# ── 6. 成交额中性化（OLS中性化） ─────────────────────────────────────────────
amount = kline[["date", "stock_code", "amount", "factor_raw", "close"]].dropna(subset=["factor_raw", "amount"])
amount["log_amount"] = np.log(amount["amount"] + 1)
amount["log_close"] = np.log(amount["close"] + 1)

def neutralize_cross_section(group: pd.DataFrame) -> pd.Series:
    """截面OLS线性中性化：factor_raw ~ log_amount"""
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(len(group)), group["log_amount"].values])
    y = group["factor_raw"].values.astype(float)
    try:
        beta, *_ = lstsq(X, y, rcond=None)
        resid = y - X @ beta
    except Exception:
        resid = y
    return pd.Series(resid, index=group.index)

amount["factor_neutral"] = amount.groupby("date", group_keys=False).apply(neutralize_cross_section)

# ── 7. MAD winsorize + z-score ─────────────────────────────────────────────────
def mad_zscore(s: pd.Series, n_sigma: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=s.index)
    upper = med + n_sigma * 1.4826 * mad
    lower = med - n_sigma * 1.4826 * mad
    s_clip = s.clip(lower, upper)
    return (s_clip - s_clip.mean()) / (s_clip.std() + 1e-8)

amount["factor_final"] = amount.groupby("date", group_keys=False)["factor_neutral"].transform(mad_zscore)

# ── 8. 输出因子CSV ─────────────────────────────────────────────────────────────
out = amount[["date", "stock_code", "factor_final"]].rename(columns={"factor_final": "factor_value"})
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out.to_csv(OUT_DIR / "factor_gap_reversal_strength_v1.csv", index=False)
print(f"Done. Written {len(out)} rows to factor_gap_reversal_strength_v1.csv")
print(f"Date range: {out['date'].min()} ~ {out['date'].max()}")
print(f"Cross-sections: {out['date'].nunique()}")
