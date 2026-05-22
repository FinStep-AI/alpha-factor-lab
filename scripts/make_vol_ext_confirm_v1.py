"""
vol_ext_confirm_v1 —— 成交量极值事件收益确认因子
-------------------------------------------------
20d 滚动窗口，将每日截面 volume 做 pct-rank，取 hi≥70th / lo≤30th 两簇。
若一只股票窗口内同时能拿出 ≥4 天的高量记录和 ≥4 天低量记录，
则计算两簇未来 5 日总收益均值之差，否则为 NaN。
输出 → data/factor_vol_ext_confirm_v1.csv
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

BASE  = Path(__file__).resolve().parent.parent
kline = pd.read_csv(BASE/"data"/"csi1000_kline_raw.csv", parse_dates=["date"])
ret   = pd.read_csv(BASE/"data"/"csi1000_returns.csv", parse_dates=["date"])

# ── pivot ────────────────────────────────────────────────────────────────
vol  = kline.pivot_table(index="date", columns="stock_code", values="volume").sort_index()
ret  = ret.pivot_table(index="date",  columns="stock_code", values="return").sort_index()
amt  = kline.pivot_table(index="date", columns="stock_code", values="amount").sort_index()

# ── 5d 前瞻收益（截面日 t 对应的 forward 收益从 t+1 到 t+5 的累计）───────
log_ret  = np.log1p(ret.clip(-0.999))
fwd5 = np.expm1(log_ret.cumsum().shift(-5) - log_ret.cumsum())

DATES = vol.index; STOCKS = vol.columns
V = vol.values.astype(float);   R = ret.values.astype(float)
F5 = fwd5.values.astype(float)
T, N = V.shape

# ── per-day safe pct-rank ─────────────────────────────────────────────────
ranks = np.empty((T, N), np.float64)
for t in range(T):
    col = V[t].copy(); nm = np.isnan(col)
    if nm.all(): ranks[t] = np.nan; continue
    r = pd.Series(col[~nm]).rank(method="average").values / col[~nm].size
    row = np.full(N, np.nan); row[~nm] = r; ranks[t] = row

# ── rolling window: 放量 / 缩量日之未来 5 日平均收益差 ─────────────────────
W, HI, LO, HI_MIN, LO_MIN = 20, 0.70, 0.30, 4, 4

raw = []
for i in range(W, T):
    pct = ranks[i - W : i]                  # (W, N)
    valid = np.isfinite(pct)
    hi_ok = (pct >= HI) & valid
    lo_ok = (pct <= LO) & valid
    hc = hi_ok.sum(0);  lc = lo_ok.sum(0)
    can = (hc >= HI_MIN) & (lc >= LO_MIN)  # both-side stocks

    r = np.full(N, np.nan)
    if can.any():
        # einsum: sum over W axis only for matching flags
        hi_sum = np.einsum("ij,ij->j", pct, hi_ok.astype(float))
        lo_sum = np.einsum("ij,ij->j", pct, lo_ok.astype(float))
        hi_r5  = np.einsum("ij,ij->j", F5[i-W:i], hi_ok.astype(float))
        lo_r5  = np.einsum("ij,ij->j", F5[i-W:i], lo_ok.astype(float))

        # only overwrite valid-side stocks
        hi_sum[~can] = np.nan;  lo_sum[~can] = np.nan
        hi_r5 [~can] = np.nan;  lo_r5 [~can] = np.nan
        cnt_h = np.where(can, hc, 1);  cnt_l = np.where(can, lc, 1)
        vol_diff     = hi_sum / cnt_h - lo_sum / cnt_l            # volume-rank 差
        ret_diff     = hi_r5  / cnt_h - lo_r5  / cnt_l            # 未来 5d 收益差
        r[can] = ret_diff[can]                       # 因子 raw = 收益差
    raw.append(pd.Series(r, index=STOCKS, name=DATES[i]))

fac = pd.DataFrame(raw)                            # (T-W, N)

# ── 成交额 OLS 中性化 ──────────────────────────────────────────────────────
amtr = np.log(amt.rolling(20).mean().reindex(fac.index)[STOCKS] + 1)

res_list = []
for i in range(len(fac)):
    y = fac.values[i];  x = amtr.values[i]
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 30:
        res_list.append(np.full(N, np.nan));  continue
    yy, xx = y[m], x[m];  dx = xx - xx.mean()
    b = (dx * yy).sum() / ((dx ** 2).sum() + 1e-12)
    a = yy.mean() - b * xx.mean()
    r = np.full(N, np.nan);  r[m] = yy - (a + b * xx)
    res_list.append(r)

resid = pd.DataFrame(res_list, index=fac.index, columns=STOCKS)

# ── MAD winsorize + z-score ───────────────────────────────────────────────
def _madz(s):
    med = s.median();  mad = (s - med).abs().median() * 1.4826
    if mad < 1e-10:  return pd.Series(0.0, index=s.index)
    c = s.clip(med - 3.5 * mad, med + 3.5 * mad)
    return (c - c.mean()) / (c.std() + 1e-10)

final = resid.apply(_madz, axis=1)

out = (final.stack().reset_index().rename(columns={0: "factor_value"})
       .dropna(subset=["factor_value"]))
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out["date"]       = out["date"].astype(str)
out.to_csv(BASE/"data"/"factor_vol_ext_confirm_v1.csv", index=False)

t = out["factor_value"]
print(f"rows={len(out)}  mean={t.mean():+.5f}  std={t.std():.4f}"
      f"  min={t.min():.3f}  max={t.max():.3f}")
