"""
因子：量价方向协偏度 (Volume-Return Co-skewness, VRC)

背景灵感：
  从Chordia & Subrahmanyam (2004) Order Imbalance + Harvey & Siddique (1999) Co-skewness 出发，
  但采用更精确的协偏度定义替代简单相关性。

本因子构造:
  1. ret = change / 100
  2. vol_chg = volume / MA20(volume) - 1
  3. 20日滚动量价协偏度 = mean(ret^2 * sign(vol_chg), 20d)
  4. 按日做成交额中性化，输出标准化残差

理论逻辑:
  - 高协偏度: 大波动日更常出现在放量方向一致的交易环境中
  - 低协偏度: 大波动更多出现在缩量或噪音交易环境中

Barra风格: 微观结构
"""

import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
DATA = WORKSPACE / "data"
OUT_DIR = DATA / "factor_vol_ret_coskew_v1"
OUT_DIR.mkdir(exist_ok=True)

ROLL = 20

print("加载 kline 数据...")
kl = pd.read_csv(DATA / "csi1000_kline_raw.csv")
kl["date"] = pd.to_datetime(kl["date"])

print("计算基础矩阵...")
ret_mat = kl.pivot_table(index="date", columns="stock_code", values="change").sort_index() / 100.0
vol_mat = kl.pivot_table(index="date", columns="stock_code", values="volume").sort_index().astype(float)
amt_mat = kl.pivot_table(index="date", columns="stock_code", values="amount").sort_index().astype(float)

print(f"计算 {ROLL} 日成交量偏离...")
vol_chg = vol_mat / vol_mat.rolling(ROLL).mean() - 1.0

print(f"计算 {ROLL} 日滚动量价协偏度...")
result_dict = {}
dates_all = ret_mat.index.intersection(vol_chg.index)
ret_mat = ret_mat.loc[dates_all]
vol_chg = vol_chg.loc[dates_all]

for idx in range(ROLL - 1, len(dates_all)):
    date = dates_all[idx]
    r_win = ret_mat.iloc[idx - ROLL + 1: idx + 1]
    v_win = vol_chg.iloc[idx - ROLL + 1: idx + 1]
    coskew_raw = ((r_win * r_win) * np.sign(v_win)).mean()
    result_dict[date] = coskew_raw

coskew_df = pd.DataFrame(result_dict).T
coskew_df.columns.name = "stock_code"

factor_raw = coskew_df.stack(dropna=True).reset_index()
factor_raw.columns = ["date", "stock_code", "factor_raw"]
factor_raw["factor_raw"] = factor_raw["factor_raw"].replace([np.inf, -np.inf], np.nan)
factor_raw.to_csv(OUT_DIR / "factor_raw.csv", index=False)
print(f"原始协偏度保存: {len(factor_raw)} 条")

print("按日做 winsorize + z-score + 成交额中性化...")
final_frames = []
amount_long = amt_mat.stack(dropna=True).reset_index()
amount_long.columns = ["date", "stock_code", "amount"]
merged = factor_raw.merge(amount_long, on=["date", "stock_code"], how="left")

for d, grp in merged.groupby("date", sort=True):
    grp = grp[["date", "stock_code", "factor_raw", "amount"]].dropna().copy()
    if len(grp) < 50:
        continue

    vals = grp["factor_raw"].values
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        continue
    lower = med - 3.0 * mad
    upper = med + 3.0 * mad
    grp["factor_w"] = grp["factor_raw"].clip(lower, upper)

    std = grp["factor_w"].std()
    if not np.isfinite(std) or std == 0:
        continue
    grp["factor_zs"] = (grp["factor_w"] - grp["factor_w"].mean()) / std

    grp["log_amount"] = np.log(grp["amount"].replace(0, np.nan))
    grp = grp.dropna(subset=["factor_zs", "log_amount"])
    if len(grp) < 50:
        continue

    x = grp["log_amount"].values
    y = grp["factor_zs"].values
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    if denom == 0:
        continue
    beta = (x_c * (y - y.mean())).sum() / denom
    alpha = y.mean() - beta * x.mean()
    grp["factor_value"] = y - (alpha + beta * x)

    resid_std = grp["factor_value"].std()
    if not np.isfinite(resid_std) or resid_std == 0:
        continue
    grp["factor_value"] = (grp["factor_value"] - grp["factor_value"].mean()) / resid_std
    final_frames.append(grp[["date", "stock_code", "factor_value"]])

if not final_frames:
    raise RuntimeError("中性化后没有有效因子值，请检查数据对齐")

out_df = pd.concat(final_frames, ignore_index=True)
out_path = DATA / "factor_vol_ret_coskew_v1.csv"
out_df.to_csv(out_path, index=False)
print(f"保存因子值: {out_path}, 共 {len(out_df)} 条")
print("完成！")
