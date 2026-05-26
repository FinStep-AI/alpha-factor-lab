#!/usr/bin/env python3
"""
因子: volshock_sigd_v1  --  Volume-Shock × Signed Overnight Direction (clean two-way)
论文: Hameed & Tseng 2025 "Volume Shocks and Overnight Returns"
     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5156605

动机:
  volshock_od_v1 引入 close_pos 后顶端组单调性断裂 (G4 > G5)。
  这里回到 SSRN 论文的纯核心里程碑:
    Given yesterday's return sign, did high-volume days amplify that signal?
    即 vol_shock 和历史 overnight 收益的符号是否一致。

公式 (日频、隔夜 lag1):
  vs(t)   = sign( vol(t-1) / MA20_vol(t-1) - 1 )
  o_ret(t)= close(t-1)/close(t-2) - 1          (隔夜收益, lag1)
  sign_o  = sign( o_ret(t-1) )
  raw     = vs(t) × sign_o

  放量 + 收益正 → +1 (今天继续上涨)
  放量 + 收益负 → -1 (今天继续下跌)
  缩量 + 收益正 → -1 (今天可能回落)
  缩量 + 收益负 → +1 (今天可能反弹, 缩量止跌)

中性化: 成交额 OLS 残差 → MAD → zscore
输出: data/factor_volshock_sigd_v1.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

EPS   = 1e-9
LV_W  = 20
WIN   = 3.0
COST  = 0.003     # NOT used here, just in report
FWD5  = 5          # primary horizon
DATA  = Path(__file__).parent.parent / "data"
OUT   = DATA / "factor_volshock_sigd_v1.csv"

# ── 1. 加载 ────────────────────────────────────────────────────────────────
print("[1/4] 加载 …")
kl = pd.read_csv(DATA / "csi1000_kline_raw.csv", parse_dates=["date"])
kl = kl.sort_values(["stock_code","date"]).reset_index(drop=True)

# tomorrow-return = close shifted back = today's overnight direction
# overnight_ret_yday = close_yesterday / close_2days - 1
kl["close_lag1"] = kl.groupby("stock_code")["close"].shift(1)
kl["close_lag2"] = kl.groupby("stock_code")["close"].shift(2)
kl["overnight_ret_yday"] = kl["close_lag1"] / kl["close_lag2"] - 1.0

# leave only rows where both closes exist
kl = kl.dropna(subset=["close_lag1","close_lag2","volume","amount"]).copy()

# ── 2. vol_shock + sign  ──────────────────────────────────────────────────
print("[2/4] 滚动计算 vol_shock …")

def roll(g):
    g = g.sort_values("date").copy()
    ma = g["volume"].rolling(LV_W, min_periods=LV_W).mean()
    g["vol_shock"] = g["volume"] / (ma + EPS) - 1.0
    return g

kl = kl.groupby("stock_code", group_keys=False).apply(roll)
kl = kl.dropna(subset=["vol_shock"]).copy()

kl["vs"]     = np.sign(kl["vol_shock"])        # +1 or -1
kl["sign_o"] = np.sign(kl["overnight_ret_yday"])  # +1 or -1

# lag1: on day t, yesterday's (vs × sign_o) drives today's signal
for c in ["vs","sign_o"]:
    kl[c+"_l"] = kl.groupby("stock_code")[c].shift(1)

kl = kl.dropna(subset=["vs_l","sign_o_l"]).copy()
kl["factor_raw"] = kl["vs_l"] * kl["sign_o_l"]   # {-1, 0 -> rare, +1}

# ── 3. 成交额中性化 + MAD + z-score ──────────────────────────────────────
print("[3/4] 中性化 & 截面标准化 …")

def neutral(g):
    ym = g[["factor_raw","amount"]].dropna()
    ym = ym[ym["amount"] > 0]
    if len(ym) < 30 or ym["amount"].nunique() < 2:
        ranks = ym["factor_raw"].rank(method="average")
        n = len(ym) + EPS
        z = (ranks - (n+1)/2) / ((n**2-1)/12 + EPS)
        z = np.clip(z, -WIN, WIN)
    else:
        y, x = ym["factor_raw"].values, ym["amount"].values
        try:
            b0,b1,_,_,_ = sp_stats.linregress(x,y)
        except Exception:
            g.loc[:,"factor_value"] = np.nan
            return g
        r = y - (b0 + b1*x)
        med = np.median(r)
        mad = np.median(np.abs(r-med)) + EPS
        z = (r-med) / (1.4826*mad)
        z = np.clip(z,-WIN,WIN)
    std = z.std()
    if std < EPS:
        g.loc[ym.index,"factor_value"] = 0.0
    else:
        g.loc[ym.index,"factor_value"] = (z - z.mean()) / std
    return g

kl = kl.groupby("date",group_keys=False).apply(neutral)
kl = kl.dropna(subset=["factor_value"])

# ── 4. 输出 ───────────────────────────────────────────────────────────────
print("[4/4] 写 CSV …")
out = kl[["date","stock_code","factor_value"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out = out.sort_values(["date","stock_code"]).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"  行数={len(out)}  跨度={out['date'].min()}~{out['date'].max()}")
print(f"  截面股票均值={out.groupby('date')['stock_code'].count().mean():.0f}")
