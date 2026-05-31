"""
factor_roe_accel_v2.py
----------------------
ROE 加速度因子 v2 — 更贴近 Hou-Xue-Zhang (2015) 构造

构造步骤
~~~~~~~~
1. 读 csi1000_fundamental_cache.csv，取得 stock_code / report_date / roe
2. 将季度 ROE 展开到日度（report 日向前展，报告日后 45 天生效）
3. 构造：
     roe_yoy     = ROE_t - ROE_{t-4}       # 同比变化（4个季度前）
     roe_accel   = roe_yoy_t - roe_yoy_{t-4}  # 加速度（yoy 环比变化）
4. 成交额 OLS 中性化 + MAD 缩尾 + zscore
5. 输出 factor_roe_accel_v2.csv

与 roe_accel_v1 的区别
~~~~~~~~~~~~~~~~~~~~~~~
v1 用的是 diff(ma(ROE, 4)) 做近似，拾取的是 ROE 趋势斜率；
v2 严格区分两步：先算同比变化率的序列，再对变化率本身做一阶差分，
   符合 HXZ 论文中" profitability acceleration / change in profitability "语义。
"""

import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")

FUND  = "data/csi1000_fundamental_cache.csv"
KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_roe_accel_v2.csv"

REPORT_LAG = 45   # 报告日后多少天因子生效

# ── 1. 加载基本面 ────────────────────────────────────────────────────────────
print("Loading fundamentals …")
fund = pd.read_csv(FUND, parse_dates=["report_date"])
fund["stock_code"] = fund["stock_code"].astype(str).str.zfill(6)
fund = fund.dropna(subset=["roe"]).sort_values(["stock_code","report_date"])

# ── 2. 季度 ROE → 季度 yoy (diff 4 periods) ─────────────────────────────
fund["roe_yoy"] = fund.groupby("stock_code")["roe"].diff(4)

# ── 3. 季度 yoy → 加速度（yoy 环比差分）───────────────────────────────────
fund["roe_accel_raw"] = fund.groupby("stock_code")["roe_yoy"].diff(1)

# ── 4. 展开到日度 ──────────────────────────────────────────────────────────
print("Expanding to daily …")
all_dates = pd.read_csv(KLINE, usecols=["date"], parse_dates=["date"])["date"].unique()
all_dates = np.sort(all_dates)
stocks = sorted(fund["stock_code"].unique())

rows = []
for code, grp in fund.groupby("stock_code"):
    grp = grp.sort_values("report_date")
    eff_dates = grp["report_date"] + pd.Timedelta(days=REPORT_LAG)
    # 只取未来的日期
    future = grp[eff_dates <= all_dates.max()].copy()
    if future.empty:
        continue
    future["eff_date"] = pd.to_datetime(eff_dates[eff_dates <= all_dates.max()].values)
    future = future[["eff_date","stock_code","roe_accel_raw"]].rename(columns={"eff_date":"date"})
    # 将季度数据前向填充到日频
    daily_idx = pd.DataFrame({"date": all_dates})
    daily_idx["stock_code"] = code
    merged = daily_idx.merge(future, on=["date","stock_code"], how="left")
    merged["roe_accel_raw"] = merged["roe_accel_raw"].ffill()
    rows.append(merged)

daily_roe = pd.concat(rows, ignore_index=True)
print(f"  daily ROE rows: {len(daily_roe)}")

# ── 5. 合并 Kline 成交额 ───────────────────────────────────────────────────
print("Merging kline …")
kline = pd.read_csv(KLINE, parse_dates=["date"],
                    usecols=["date","stock_code","amount"])
kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)
kline["log_amount_20d"] = (
    kline.groupby("stock_code")["amount"]
    .transform(lambda x: np.log(x.rolling(20,min_periods=10).mean()+1))
)
kline = kline.dropna(subset=["log_amount_20d"])

panel = kline.merge(daily_roe, on=["date","stock_code"], how="left")
panel = panel.dropna(subset=["roe_accel_raw","log_amount_20d"])

# ── 6. 截面中和 ────────────────────────────────────────────────────────────
results = []
for dt, p in panel.groupby("date"):
    if len(p) < 30: continue
    y = p["roe_accel_raw"].values
    X = np.column_stack([np.ones(len(p)), p["log_amount_20d"].values])
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
    except Exception:
        continue
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    if mad < 1e-8: continue
    r = np.clip(r, med - 5.2*mad, med + 5.2*mad)
    s = r.std()
    if s < 1e-8: continue
    z = (r - r.mean()) / s
    results.append(pd.DataFrame({
        "date": dt, "stock_code": p["stock_code"].values,
        "factor_roe_accel_v2": z}))

out = pd.concat(results, ignore_index=True).sort_values(["date","stock_code"]).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"Done. {len(out)} rows → {OUT}")
print(f"Range: {out['date'].min()} ~ {out['date'].max()}")
print(f"Median stocks/day: {out.groupby('date')['stock_code'].count().median():.0f}")
print(out["factor_roe_accel_v2"].describe())
