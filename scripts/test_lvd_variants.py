#!/usr/bin/env python3
"""缩量下跌因子变体测试 - 优化IC"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

warnings.filterwarnings("ignore")

BASE = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
DATA_DIR = BASE / "data"

def mad_winsorize(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median() * 1.4826
    return s.clip(median - n_mad * mad, median + n_mad * mad)

def neutralize_ols(df, factor_col, neutral_col):
    result = df[[factor_col]].copy()
    result["residual"] = np.nan
    for date, group in df.groupby("date"):
        y = group[factor_col].values
        x = group[neutral_col].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, _, _, _ = lstsq(X, y[mask], rcond=None)
        resid = y.copy()
        resid[mask] = y[mask] - X @ beta
        resid[~mask] = np.nan
        result.loc[group.index, "residual"] = resid
    return result["residual"]

def calc_quick_ic(df, ret_pivot, forward_days=5):
    """快速IC"""
    valid = df.dropna(subset=["factor_value"])
    factor_pivot = valid.pivot_table(index="date", columns="stock_code", values="factor_value")
    common_dates = factor_pivot.index.intersection(ret_pivot.index)
    common_stocks = factor_pivot.columns.intersection(ret_pivot.columns)
    fp = factor_pivot.loc[common_dates, common_stocks]
    rp = ret_pivot.loc[:, common_stocks]
    fwd = rp.rolling(forward_days).sum().shift(-forward_days)
    fwd_aligned = fwd.loc[common_dates]
    ic_list = []
    for dt in common_dates:
        f_row = fp.loc[dt].dropna()
        r_row = fwd_aligned.loc[dt].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < 50:
            continue
        ic = f_row[common].corr(r_row[common])
        if np.isfinite(ic):
            ic_list.append(ic)
    ic_arr = np.array(ic_list)
    ic_mean = ic_arr.mean()
    ic_t = ic_mean / (ic_arr.std() / np.sqrt(len(ic_arr)))
    return ic_mean, ic_t, len(ic_arr)

# 读取数据
print("读取数据...")
df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df["ret"] = df.groupby("stock_code")["close"].pct_change()
df["log_amount_20d"] = np.log(df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()).clip(lower=1))

df["vol_ma20"] = df.groupby("stock_code")["volume"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df["vol_ma10"] = df.groupby("stock_code")["volume"].transform(
    lambda x: x.rolling(10, min_periods=5).mean()
)

returns = pd.read_csv(DATA_DIR / "csi1000_returns.csv")
returns["date"] = pd.to_datetime(returns["date"])
ret_pivot = returns.pivot(index="date", columns="stock_code", values="return")

# 变体测试
variants = {
    # 基准
    "lvd_10d_base": {"desc": "基准:缩量下跌10d频率", "window": 10,
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < d["vol_ma20"])).astype(float)},
    
    # 窗口变体
    "lvd_5d": {"desc": "缩量下跌5d", "window": 5,
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < d["vol_ma20"])).astype(float)},
    "lvd_15d": {"desc": "缩量下跌15d", "window": 15,
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < d["vol_ma20"])).astype(float)},
    "lvd_20d": {"desc": "缩量下跌20d", "window": 20,
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < d["vol_ma20"])).astype(float)},
    
    # 阈值变体
    "lvd_strong_10d": {"desc": "强缩量下跌(vol<0.7*MA20)10d", "window": 10,
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < 0.7 * d["vol_ma20"])).astype(float)},
    "lvd_deep_10d": {"desc": "深缩量下跌(ret<-1% & vol<MA20)10d", "window": 10,
        "cond": lambda d: ((d["ret"] < -0.01) & (d["volume"] < d["vol_ma20"])).astype(float)},
    
    # 加权变体：不看频率，看缩量下跌的成交量/MA20比值
    "lvd_vol_weighted_10d": {"desc": "缩量下跌日vol/MA20均值(weighted)10d", "window": 10,
        "cond": lambda d: np.where((d["ret"] < 0) & (d["volume"] < d["vol_ma20"]),
                                    d["volume"] / d["vol_ma20"], np.nan)},
    
    # 成交量萎缩幅度加权
    "lvd_shrink_10d": {"desc": "缩量下跌日(1-vol/MA20)求和10d", "window": 10,
        "cond": lambda d: np.where((d["ret"] < 0) & (d["volume"] < d["vol_ma20"]),
                                    1 - d["volume"] / d["vol_ma20"], 0)},
    
    # 用MA10而非MA20
    "lvd_ma10_10d": {"desc": "缩量下跌(vol<MA10)10d", "window": 10,
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < d["vol_ma10"])).astype(float)},
}

print(f"\n{'名称':<25} {'描述':<40} {'IC':>8} {'t':>8} {'N':>5}")
print("-" * 90)

for name, cfg in variants.items():
    # 计算条件
    df["signal"] = cfg["cond"](df)
    df.loc[df["ret"].isna() | df["vol_ma20"].isna(), "signal"] = np.nan
    
    w = cfg["window"]
    df["freq"] = df.groupby("stock_code")["signal"].transform(
        lambda x: x.rolling(w, min_periods=max(3, int(w*0.7))).mean()
    )
    
    df["f_win"] = df.groupby("date")["freq"].transform(mad_winsorize)
    df["f_neutral"] = neutralize_ols(df, "f_win", "log_amount_20d")
    df["factor_value"] = df.groupby("date")["f_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    ic_mean, ic_t, n = calc_quick_ic(df, ret_pivot, forward_days=5)
    star = " ★" if ic_mean > 0.02 and abs(ic_t) > 2 else " ✓" if abs(ic_t) > 2 else ""
    print(f"  {name:<23} {cfg['desc']:<40} {ic_mean:>7.4f} {ic_t:>7.2f} {n:>5}{star}")

# 也测试10d forward
print(f"\n--- 10d forward return ---")
for name in ["lvd_10d_base", "lvd_5d", "lvd_shrink_10d"]:
    cfg = variants[name]
    df["signal"] = cfg["cond"](df)
    df.loc[df["ret"].isna() | df["vol_ma20"].isna(), "signal"] = np.nan
    w = cfg["window"]
    df["freq"] = df.groupby("stock_code")["signal"].transform(
        lambda x: x.rolling(w, min_periods=max(3, int(w*0.7))).mean()
    )
    df["f_win"] = df.groupby("date")["freq"].transform(mad_winsorize)
    df["f_neutral"] = neutralize_ols(df, "f_win", "log_amount_20d")
    df["factor_value"] = df.groupby("date")["f_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    ic_mean, ic_t, n = calc_quick_ic(df, ret_pivot, forward_days=10)
    star = " ★" if ic_mean > 0.02 and abs(ic_t) > 2 else " ✓" if abs(ic_t) > 2 else ""
    print(f"  {name:<23} 10d_fwd  {ic_mean:>7.4f} {ic_t:>7.2f} {n:>5}{star}")

print("\n完成！")
