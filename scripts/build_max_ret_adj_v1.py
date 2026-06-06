#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: max_ret_adj_v1 — 波动率调整彩票溢价（Bali-Cakici-Whitelaw 2011 本土优化）

论文来源
========
  Bali, Brown, Murray, Tang (2017, JFQA)  "A Lottery-Demand-Based Explanation of the Beta Anomaly",
      https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/
          lotterydemandbased-explanation-of-the-beta-anomaly/B5B9F0A65256E6E86B45D72AE0A256C4

  Bali, Cakici, Whitelaw (2011, JFE) "Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns"
      https://www.jstor.org/stable/26590484

现有量价因子中：
  tail_risk_cvar_v1  用 CVaR(10d, bottom-2) 做尾部风险； MAX 与 CVaR 截面相关性约 0.55
  两者均是"极端收益"代理，但方向不同：
    CVaR 走高 → 极端亏损更频繁； MAX 走高 → 极端盈利/彩票偏好；
  不能相互替代。

优化动机
========
原始 MAX 因子（上月单日最大收益）在美股有强 IC，在 A 股因为涨跌停板原因
原始 MAX 分布会因 price-limit 压缩成不连续；简单取 MAX_20d 在中证1000上漂移也大。
考虑到此，改良公式在吻合原文献的"彩票偏好"精神下，聚焦测显 ROBUSTNESS 结构：

  因子 = - MAX_20d / σ_20d
   (negative:  MAX 高 → 更多"LOTTERY"溢价要求 → 低预期收益)
  除以 20日波动率 → 消除极端一天的随机性，使信号稳定

方向：高值 = 彩票溢价低 = 高预期收益（与 Bali et al 原始方向相反）

构造:
  1. MAX_20d = rolling 20d 个股 max(pct_change)
  2. SIGMA_20d = rolling 20d std(ret)
  3. factor_raw = - MAX_20d / SIGMA_20d
  4. OLS 成交额中性化 MAD winsorize z-score

Barra: 本因子定位 Liquidity/MICRO（已有 tail_risk_cvar_v1（尾部风险）/vol_extreme）不直接占用 Growth/Quality）。
来源归 Momentum 混合-Lottery 特征。
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

BASE   = Path(__file__).resolve().parent.parent
DPATH  = BASE / 'data' / 'csi1000_kline_raw.csv'
OUT    = BASE / 'data' / 'factor_max_ret_adj_v1.csv'
KLINE  = BASE / 'data' / 'csi1000_kline_raw.csv'
W      = 20
MIN_P  = 12

# ── data ──────────────────────────────────────────────
print('[1] load kline…')
df = pd.read_csv(DPATH, dtype={'stock_code': str})
df['stock_code'] = df['stock_code'].str.zfill(6)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code','date']).reset_index(drop=True)

# 若 pct_change 含 NaN 则重算
if df['pct_change'].isna().mean() > 0.2:
    close = df.pivot_table(index='date', columns='stock_code', values='close')
    df['pct_change'] = (close.pct_change().reindex(df['date']).lookup(df.index, df['stock_code']))*100

cpiv = df.pivot_table(index='date', columns='stock_code', values='close').sort_index()
apiv = df.pivot_table(index='date', columns='stock_code', values='amount').sort_index()
ret  = cpiv.pct_change()

# ── raw factor ────────────────────────────────────────
roll_max  = ret.rolling(W, min_periods=MIN_P).max()
roll_std  = ret.rolling(W, min_periods=MIN_P).std()
raw = -roll_max / roll_std.replace(0, np.nan)
print('[2] raw factor:\n', raw.stack().describe())

# ── winsorize + neutralize + z-score ──────────────────
log_amt = np.log(apiv.rolling(W, min_periods=10).mean().clip(lower=1))

def winsorize_mad_row(fr):
    fr = np.asarray(fr, dtype=float)
    m  = np.isfinite(fr)
    if m.sum() < 30: return fr
    med = np.nanmedian(fr[m])
    mad = np.nanmedian(np.abs(fr[m] - med))
    if mad < 1e-12: return fr
    return np.clip(fr, med - 5*1.4826*mad, med + 5*1.4826*mad)

def neutralize_ols(fr, am):
    fr, am = np.asarray(fr, float), np.asarray(am, float)
    ok = np.isfinite(fr) & np.isfinite(am)
    out = np.full_like(fr, np.nan)
    if ok.sum() < 50: return out
    x = np.column_stack([np.ones(ok.sum()), am[ok]])
    y = fr[ok]
    try:
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        r = y - x @ b
    except Exception:
        return out
    mu, s = r.mean(), r.std()
    if s < 1e-12: return out
    out[ok] = (r - mu) / s
    return out

records = []
dates   = raw.index
stocks  = raw.columns
for i, dt in enumerate(dates):
    fr = raw.loc[dt].values
    am = log_amt.loc[dt].values
    if not np.isfinite(am).any(): continue
    fr = winsorize_mad_row(fr)
    z  = neutralize_ols(fr, am)
    if not np.isfinite(z).any(): continue
    m  = np.isfinite(z)
    records += [{'date': dt, 'stock_code': stocks[j], 'factor_value': float(z[j])}
                 for j in np.where(m)[0]]
    if (i+1) % 100 == 0: print(f'  {i+1}/{len(dates)}')

fout = pd.DataFrame(records)
fout['date'] = pd.to_datetime(fout['date'])
fout.to_csv(OUT, index=False)
print(f'[3] saved {len(fout)} rows → {OUT}')
print(f'    {fout["date"].min().date()} ~ {fout["date"].max().date()}')
print(f'    mean={fout["factor_value"].mean():.4f}  std={fout["factor_value"].std():.4f}')
