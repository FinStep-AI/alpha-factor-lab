"""
roe_qual_persist_v2 — ROE 时序持续性因子（日频 output）
========================================================
对每个 available trade_date 做截面：
  1. 取当时全部已发布季报中每支股票最近 KEEP=8 季 ROE
  2. 截面 Rank+时间加权求和 → factor_raw
  3. 对 log(MA20 amount) OLS 残差 → neutralized
  4. 截面 MAD winsorize + z-score
"""
import os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from numpy.linalg import lstsq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(DATA, "factor_roe_qual_persist.csv")

KEEP = 8

# ── 读取 ──
fd = pd.read_csv(os.path.join(DATA, "csi1000_fundamental_cache.csv"),
                 parse_dates=["report_date"])
fd = fd[fd["roe"].notna()].copy()
fd["stock_code"] = fd["stock_code"].astype(str).str.zfill(6)
fd = fd.sort_values(["stock_code","report_date"])

kline = pd.read_csv(os.path.join(DATA, "csi1000_kline_raw.csv"),
                    parse_dates=["date"],
                    usecols=["date","stock_code","amount"])
kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)
kline = kline.sort_values(["stock_code","date"])

amt20 = (kline.groupby("stock_code")["amount"]
         .transform(lambda s: s.rolling(20, min_periods=5).mean())
         .rename("amt20"))
kline = kline.join(amt20)

# 时间窗口的权重（近大远小，4 档分段）
sub_w = np.linspace(0.1, 1.0, KEEP);  sub_w /= sub_w.sum()

# ── 逐步超时：只处理报告期 <= 截面前、且最后一条 eff_date <= 截面日 的股票 ──
print("[INFO] building factor …")

results = []

def build_one(rpt_idx: int):
    """处理第 rpt_idx 个 report_date：找下一个 report_date 做 eff_end，emit 所有有效 trade_date。"""
    rpts_all = sorted(fd["report_date"].unique())
    if rpt_idx >= len(rpts_all) - 1:
        return
    rpt_cur  = rpts_all[rpt_idx]
    rpt_next = rpts_all[rpt_idx + 1]   # 下一个财报季基准点，用 effect start
    eff_start = rpt_cur + pd.Timedelta(days=45)
    eff_end   = rpt_next + pd.Timedelta(days=45)

    # 取 eff_start 之前已发布的所有季报，按 stock_code 取最近 KEEP 条
    sub = fd[fd["report_date"] <= rpt_cur].copy()
    top8 = (sub.groupby("stock_code")
            .apply(lambda g: g.sort_values("report_date").tail(KEEP))
            .reset_index(drop=True))
    if len(top8) < KEEP:
        return

    # 截面 Rank % within each (report_date × stock_code) — actually just cross-sectional on latest
    top8["ord"] = top8.groupby("stock_code").cumcount(ascending=False)
    # Rank on the latest ROE across all stocks at the cross-section date rpt_cur
    # We take each stock's most recent ROE that's already in the top8 window
    latest_per_stock = (top8[top8["ord"] == 0]
                        .set_index("stock_code")["roe"])
    if len(latest_per_stock) < 100:
        return
    rank_map = latest_per_stock.rank(pct=True)   # vector indexed by stock_code

    # For each stock beyond the latest_ROE_rank, we also include prior-period rank as components.
    # Simpler approach: build the seq = [rank at pos 0, rank at pos 1, …, rank at KEEP-1]
    poss = sorted(top8["ord"].unique(), reverse=False)
    ranked = {}
    for pos in range(KEEP):
        subp = top8[top8["ord"] == pos].set_index("stock_code")["roe"]
        rk = subp.rank(pct=True)
        ranked[pos] = rk

    # Only stocks that have KEEP consecutive data points
    ok_stocks = set.intersection(*[set(ranked[p].index) for p in range(KEEP)])
    if len(ok_stocks) < 200:
        return

    # weighted rank sum = factor value, at eff_start date signal holds
    stack = pd.DataFrame({p: ranked[p] for p in range(KEEP)})
    stack = stack.loc[list(ok_stocks)]
    stack["factor_raw"] = (stack.values * sub_w).sum(axis=1)

    # find trade dates in [eff_start, eff_end)
    td0 = pd.Timestamp(eff_start).normalize()
    td1 = pd.Timestamp(eff_end).normalize()
    trade_dates = np.sort(kline["date"].unique())
    tmask = (trade_dates >= td0) & (trade_dates < td1)
    tdates = trade_dates[tmask]
    if len(tdates) == 0:
        return

    stock_set = sorted(ok_stocks)
    for td in tdates:
        amt_row = kline.loc[kline["date"] == td, ["stock_code","amount","amt20"]].drop_duplicates("stock_code")
        if amt_row.empty:
            continue
        sig = stack["factor_raw"].reindex(stock_set)
        combined = pd.DataFrame({
            "stock_code": stock_set,
            "factor_raw": sig.values
        }).merge(amt_row, on="stock_code", how="inner")
        combined = combined.dropna(subset=["factor_raw","amt20"])
        if len(combined) < 80:
            continue

        la = np.log(np.maximum(combined["amt20"].values, 1))
        x  = np.column_stack([np.ones(len(combined)), la])
        y  = combined["factor_raw"].values
        coef = lstsq(x, y, rcond=None)[0]
        resid = y - x @ coef

        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) * 1.4826
        if mad < 1e-9:
            z = np.zeros(len(resid))
        else:
            c = np.clip(resid, med-5.2*mad, med+5.2*mad)
            z = (c - c.mean()) / (c.std()+1e-9)

        for sc, v in zip(combined["stock_code"].values, z):
            results.append({"date": td, "stock_code": sc, "factor": float(v)})

rpts = sorted(fd["report_date"].unique())
for i in range(len(rpts)):
    build_one(i)

out = pd.DataFrame(results)
out.to_csv(OUT, index=False)
print(f"✅ {OUT}")
print(f"   rows={len(out)}  dates={out['date'].dt.date.min()}~{out['date'].dt.date.max()}")
print(f"   uniq_dates={out['date'].nunique()}  stocks={out['stock_code'].nunique()}")
print(out["factor"].describe())
