import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
FACTOR_ID = "gross_margin_inflect_asset_turnover_gate_growth_v2"
OUT = BASE / f"data/factor_{FACTOR_ID}.csv"

fund = pd.read_csv(BASE / "data/csi1000_fundamental_cache.csv")
kline = pd.read_csv(
    BASE / "data/csi1000_kline_raw.csv",
    usecols=["date", "stock_code", "close", "amount", "turnover", "amplitude"],
)

fund["report_date"] = pd.to_datetime(fund["report_date"])
fund["stock_code"] = fund["stock_code"].astype(str).str.zfill(6)
for col in ["roe", "bps"]:
    fund[col] = pd.to_numeric(fund[col], errors="coerce")
fund = (
    fund.dropna(subset=["roe", "bps"])
    .sort_values(["stock_code", "report_date"])
    .drop_duplicates(["stock_code", "report_date"])
)

for col in ["roe", "bps"]:
    q01, q99 = fund[col].quantile([0.01, 0.99])
    fund[col] = fund[col].clip(q01, q99)

g = fund.groupby("stock_code")

fund["asset_turnover_proxy"] = fund["roe"] / fund["bps"].abs().clip(lower=0.8)
fund["gross_margin_proxy"] = np.log1p(fund["bps"].clip(lower=0))
fund["bps_growth_qoq"] = fund["bps"] / g["bps"].shift(1).replace(0, np.nan) - 1.0
fund["bps_growth_yoy"] = fund["bps"] / g["bps"].shift(4).replace(0, np.nan) - 1.0

for base_col in ["gross_margin_proxy", "asset_turnover_proxy", "roe"]:
    fund[f"{base_col}_l1"] = g[base_col].shift(1)
    fund[f"{base_col}_l2"] = g[base_col].shift(2)
    fund[f"{base_col}_l4"] = g[base_col].shift(4)

fund["gm_yoy"] = fund["gross_margin_proxy"] - fund["gross_margin_proxy_l4"]
fund["gm_qoq"] = fund["gross_margin_proxy"] - fund["gross_margin_proxy_l1"]
fund["gm_prev_qoq"] = fund["gross_margin_proxy_l1"] - fund["gross_margin_proxy_l2"]
fund["gm_inflect"] = fund["gm_qoq"] - fund["gm_prev_qoq"]

fund["at_yoy"] = fund["asset_turnover_proxy"] - fund["asset_turnover_proxy_l4"]
fund["at_qoq"] = fund["asset_turnover_proxy"] - fund["asset_turnover_proxy_l1"]
fund["at_prev_qoq"] = fund["asset_turnover_proxy_l1"] - fund["asset_turnover_proxy_l2"]
fund["at_inflect"] = fund["at_qoq"] - fund["at_prev_qoq"]

fund["roe_yoy"] = fund["roe"] - fund["roe_l4"]
fund["roe_qoq"] = fund["roe"] - fund["roe_l1"]
fund["gm_std4"] = g["gm_yoy"].transform(lambda s: s.rolling(4, min_periods=3).std())
fund["at_std4"] = g["at_yoy"].transform(lambda s: s.rolling(4, min_periods=3).std())
fund["roe_std4"] = g["roe"].transform(lambda s: s.rolling(4, min_periods=3).std())

for col in [
    "asset_turnover_proxy", "gross_margin_proxy", "bps_growth_qoq", "bps_growth_yoy",
    "gm_yoy", "gm_qoq", "gm_prev_qoq", "gm_inflect",
    "at_yoy", "at_qoq", "at_prev_qoq", "at_inflect",
    "roe_yoy", "roe_qoq", "gm_std4", "at_std4", "roe_std4"
]:
    s = fund[col]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[col] = s.clip(q01, q99)

gm_trend = np.tanh(6.0 * fund["gm_yoy"].fillna(0))
gm_inflect = np.tanh(8.5 * fund["gm_inflect"].fillna(0))
at_trend = np.tanh(4.5 * fund["at_yoy"].fillna(0))
at_inflect = np.tanh(6.0 * fund["at_inflect"].fillna(0))
roe_confirm = np.tanh((0.65 * fund["roe_yoy"].fillna(0) + 0.35 * fund["roe_qoq"].fillna(0)) / 5.0)

improve_gate = (
    0.55
    + 0.30 * (fund["gm_inflect"].fillna(0) > 0).astype(float)
    + 0.15 * (fund["at_yoy"].fillna(0) > -0.01).astype(float)
)
capital_guard = 1.0 / (
    1.0
    + 4.5 * fund["bps_growth_yoy"].clip(lower=0.18).fillna(0)
    + 1.8 * fund["bps_growth_qoq"].clip(lower=0.06).fillna(0)
)
stability = 1.0 / (
    1.0
    + 1.6 * fund["gm_std4"].abs().fillna(0)
    + 1.0 * fund["at_std4"].abs().fillna(0)
    + 0.08 * fund["roe_std4"].abs().fillna(0)
)
quality_gate = 0.60 + 0.40 * np.tanh(fund["roe"].fillna(0) / 8.0)
interaction = np.tanh(14.0 * (fund["gm_inflect"].fillna(0) * fund["at_yoy"].fillna(0)))

fund["raw_factor"] = (
    0.22 * gm_trend
    + 0.28 * gm_inflect
    + 0.14 * at_trend
    + 0.08 * at_inflect
    + 0.12 * roe_confirm
    + 0.16 * interaction
) * improve_gate * capital_guard * stability * quality_gate

fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_factor"])
fund["avail_date"] = fund["report_date"] + pd.Timedelta(days=45)
factor_q = fund[["stock_code", "avail_date", "raw_factor"]].rename(columns={"avail_date": "date"})

kline["date"] = pd.to_datetime(kline["date"])
kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)
kline = kline.sort_values(["stock_code", "date"]).drop_duplicates(["date", "stock_code"])
kline["mktcap_proxy"] = kline["close"].clip(lower=0.01) * kline["amount"].clip(lower=1) / (kline["turnover"].replace(0, np.nan) + 1e-6)
kline["log_mktcap"] = np.log(kline["mktcap_proxy"].clip(lower=1))
kline["amp20"] = kline.groupby("stock_code")["amplitude"].transform(lambda s: s.rolling(20, min_periods=10).mean())
trade_dates = pd.Index(sorted(kline["date"].unique()))

parts = []
for stock, grp in factor_q.groupby("stock_code"):
    sf = grp[["date", "raw_factor"]].drop_duplicates("date", keep="last").set_index("date").sort_index()
    sf = sf.reindex(trade_dates, method="ffill", limit=80)
    sf["stock_code"] = stock
    sf = sf.dropna(subset=["raw_factor"]).reset_index().rename(columns={"index": "date"})
    parts.append(sf)

factor = pd.concat(parts, ignore_index=True)
factor = factor.merge(
    kline[["date", "stock_code", "log_mktcap", "amp20"]],
    on=["date", "stock_code"], how="inner"
).dropna(subset=["raw_factor", "log_mktcap"])

def neutralize(vals, ctrl1, ctrl2):
    mask = np.isfinite(vals) & np.isfinite(ctrl1)
    if ctrl2 is not None:
        mask &= np.isfinite(ctrl2)
    if mask.sum() < 30:
        return np.full(len(vals), np.nan)
    y = vals[mask]
    cols = [np.ones(mask.sum()), ctrl1[mask]]
    if ctrl2 is not None:
        cols.append(ctrl2[mask])
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad < 1e-12:
        return np.full(len(vals), np.nan)
    resid = np.clip(resid, med - 5.0 * mad, med + 5.0 * mad)
    std = resid.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (resid - np.median(resid)) / std
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = z
    return out

outs = []
for date, grp in factor.groupby("date"):
    nz = neutralize(
        grp["raw_factor"].to_numpy(float),
        grp["log_mktcap"].to_numpy(float),
        grp["amp20"].to_numpy(float),
    )
    good = np.isfinite(nz)
    if good.any():
        sub = grp.loc[good, ["date", "stock_code"]].copy()
        sub["factor"] = nz[good]
        outs.append(sub)

result = pd.concat(outs, ignore_index=True)
result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
result.to_csv(OUT, index=False, float_format="%.6f")
print(f"saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}")
