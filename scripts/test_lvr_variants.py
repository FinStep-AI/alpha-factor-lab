#!/usr/bin/env python3
"""快速测试缩量反弹因子的不同窗口和变体"""
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

# 读取数据
print("读取数据...")
df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df["ret"] = df.groupby("stock_code")["close"].pct_change()
df["log_amount_20d"] = np.log(df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()).clip(lower=1))

# 测试变体
variants = {
    "lvr_20d": {
        "desc": "缩量反弹 20日窗口",
        "cond": lambda d: ((d["ret"] > 0) & (d["volume"] < d["vol_ma20"])).astype(float),
        "vol_ma": 20, "window": 20
    },
    "lvr_strong_10d": {
        "desc": "强缩量反弹(ret>0.5% & vol<0.7*MA20) 10日",
        "cond": lambda d: ((d["ret"] > 0.005) & (d["volume"] < 0.7 * d["vol_ma20"])).astype(float),
        "vol_ma": 20, "window": 10
    },
    "hvd_10d": {
        "desc": "放量下跌 10日(反向: 做空放量跌)",
        "cond": lambda d: -((d["ret"] < 0) & (d["volume"] > d["vol_ma20"])).astype(float),
        "vol_ma": 20, "window": 10
    },
    "lvd_10d": {
        "desc": "缩量下跌 10日(卖压枯竭: 不跌+没量)",
        "cond": lambda d: ((d["ret"] < 0) & (d["volume"] < d["vol_ma20"])).astype(float),
        "vol_ma": 20, "window": 10
    },
}

# 预计算vol_ma20
df["vol_ma20"] = df.groupby("stock_code")["volume"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

returns = pd.read_csv(DATA_DIR / "csi1000_returns.csv")
returns["date"] = pd.to_datetime(returns["date"])
ret_pivot = returns.pivot(index="date", columns="stock_code", values="return")

for name, cfg in variants.items():
    print(f"\n{'='*50}")
    print(f"测试: {name} - {cfg['desc']}")
    
    # 计算条件
    df["signal"] = cfg["cond"](df)
    df.loc[df["ret"].isna() | df["vol_ma20"].isna(), "signal"] = np.nan
    
    # 滚动频率
    w = cfg["window"]
    df["freq"] = df.groupby("stock_code")["signal"].transform(
        lambda x: x.rolling(w, min_periods=int(w*0.8)).mean()
    )
    
    # 中性化
    df["f_win"] = df.groupby("date")["freq"].transform(mad_winsorize)
    df["f_neutral"] = neutralize_ols(df, "f_win", "log_amount_20d")
    df["factor_value"] = df.groupby("date")["f_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # 快速IC计算
    valid = df.dropna(subset=["factor_value"])
    factor_pivot = valid.pivot_table(index="date", columns="stock_code", values="factor_value")
    
    # 对齐
    common_dates = factor_pivot.index.intersection(ret_pivot.index)
    common_stocks = factor_pivot.columns.intersection(ret_pivot.columns)
    
    fp = factor_pivot.loc[common_dates, common_stocks]
    
    # 5日forward return
    rp = ret_pivot.loc[:, common_stocks]
    fwd5 = rp.rolling(5).sum().shift(-5)
    fwd5_aligned = fwd5.loc[common_dates]
    
    # IC序列
    ic_list = []
    for dt in common_dates:
        f_row = fp.loc[dt].dropna()
        r_row = fwd5_aligned.loc[dt].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < 50:
            continue
        ic = f_row[common].corr(r_row[common])
        if np.isfinite(ic):
            ic_list.append(ic)
    
    if len(ic_list) > 10:
        ic_arr = np.array(ic_list)
        ic_mean = ic_arr.mean()
        ic_t = ic_mean / (ic_arr.std() / np.sqrt(len(ic_arr)))
        print(f"  IC均值: {ic_mean:.4f}  t-stat: {ic_t:.2f}  N={len(ic_arr)}")
    else:
        print(f"  IC计算失败 (N={len(ic_list)})")

print("\n完成所有变体测试！")
