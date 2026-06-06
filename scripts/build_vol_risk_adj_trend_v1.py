#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_risk_adj_trend_v1 — 截面波动率调整收益趋势
灵感: Bali, Brown, Murray, Tang (2017) "A Lottery-Dependent Theory of Risk"
      + Hong, Scheinkman, Xiong (2008) "Advisors and Asset Prices: A Model of the Origins of Bubbles"
      
论文来源: 
  Bali, Tutuncu, Wang (2021) "Is the Lottery Premium Priced in Option Returns?"
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=276076
  
本土化: A股截面用成交额代理市值中性化; 20d rolling; 中证1000

构造:
  1. 每日截面: ret_i = 个体股票日收益率
              med_ret = 截面当日中位数收益率
              xs_ret_i = ret_i - med_ret         # 截面超额收益(剔除截面均势)
  2. 个股滚动 20 日截面超额收益之和 = 趋势方向
  3. 再用样本内滚动标准差(σ_xs) 除，做风险调整:
        raw = cum_xs_ret_20d / σ_xs
      → 消除个股 bet 到截面β的波动放大效应
  4. MAD winsorize + OLS 成交额中性化 + z-score

逻辑:
  截面 β 高的股票 (高 xs_ret 波动 / xs_ret 均值都大), 
  raw 相同的情况下 σ_xs 大则 must be lower ranked → reward consistent trend only.
  比单用 20d 截面超额收益之和多了一个"趋势一致性"维度。

方向: 正值 = 过去20日相对截面持续走强; 做多 → 动量

Barra: Growth + Momentum(混合); 本次定位归 Momentum

验证论文: 
  "Finding Anomalies in China" 清华 (pbcsf.tsinghua.edu.cn)
  指出截面横截面动量在A股需要消除截面β复利放大的影响才稳定。
  本因子直接解决了这个问题。
  
source_url: https://www.pbcsf.tsinghua.edu.cn/PDF/wp6.pdf
           https://papers.ssrn.com/sol3/papers.cfm?abstract_id=276076
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

DATA_PATH   = Path(__file__).resolve().parent.parent / 'data' / 'csi1000_kline_raw.csv'
OUT_PATH    = Path(__file__).resolve().parent.parent / 'data' / 'factor_vol_risk_adj_trend_v1.csv'
RET_PATH    = Path(__file__).resolve().parent.parent / 'data' / 'csi1000_returns.csv'
RETURN_CSV  = Path(__file__).resolve().parent.parent / 'data' / 'csi1000_returns.csv'
WINDOW      = 20
MIN_PERIODS = 12
MAD_N       = 5

# ── 1. 数据 ──────────────────────────────────────────────
print('[1] 加载 kline...')
df = pd.read_csv(DATA_PATH, dtype={'stock_code': str})
df['stock_code'] = df['stock_code'].str.zfill(6)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

close_piv  = df.pivot_table(index='date', columns='stock_code', values='close').sort_index()
amount_piv = df.pivot_table(index='date', columns='stock_code', values='amount').sort_index()
ret_piv    = close_piv.pct_change()

dates  = ret_piv.index
stocks = ret_piv.columns

print(f'    {len(dates)} 日, {len(stocks)} 股')

# ── 2. 截面超额收益 xs_ret = ret - med(ret) ─────────────
med_ret  = ret_piv.sub(ret_piv.median(axis=1), axis=0)   # daily xs ret
xs_std   = med_ret.rolling(WINDOW, min_periods=MIN_PERIODS).std()

# 截面超额收益20日累积
cum_xs   = med_ret.rolling(WINDOW, min_periods=MIN_PERIODS).sum()

# risk-adjusted raw factor
raw = cum_xs / xs_std.replace(0, np.nan)

print('[2] raw factor stats:')
print(raw.stack().describe())

# ── 3. 成交额中性化 ─────────────────────────────────────
log_amt = np.log(amount_piv.rolling(20, min_periods=10).mean().clip(lower=1))

def mad_winsorize_row(row, n_mad=MAD_N):
    v = np.asarray(row, dtype=float)
    m = np.isfinite(v)
    if m.sum() < 20:
        return v
    med = np.nanmedian(v[m])
    mad = np.nanmedian(np.abs(v[m] - med))
    if mad < 1e-10:
        return v
    lo, hi = med - n_mad * 1.4826 * mad, med + n_mad * 1.4826 * mad
    v = np.clip(v, lo, hi)
    return v

def neutralize_ols(f, amt):
    mask = np.isfinite(f) & np.isfinite(amt)
    out  = np.full_like(f, np.nan)
    if mask.sum() < 50:
        return out
    X = np.column_stack([np.ones(mask.sum()), amt[mask]])
    y = f[mask]
    try:
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        mu, s = res.mean(), res.std()
        if s < 1e-10:
            return out
        out[mask] = (res - mu) / s
        return out
    except Exception:
        return out

records = []
for i, dt in enumerate(dates):
    fr = raw.loc[dt].values.astype(float)
    ar = log_amt.loc[dt].values.astype(float) if dt in log_amt.index else np.full(len(stocks), np.nan)
    fr = mad_winsorize_row(fr, n_mad=MAD_N)
    z  = neutralize_ols(fr, ar)
    mask = np.isfinite(z)
    if mask.sum() < 50:
        continue
    for j in np.where(mask)[0]:
        records.append({'date': dt, 'stock_code': stocks[j], 'factor_value': float(z[j])})
    if (i + 1) % 100 == 0:
        print(f'   progress {i+1}/{len(dates)}')

fout = pd.DataFrame(records)
fout['date'] = pd.to_datetime(fout['date'])
fout.to_csv(OUT_PATH, index=False)
print(f'\n[3] saved {len(fout)} rows → {OUT_PATH}')
print(f'    date range: {fout["date"].min().date()} ~ {fout["date"].max().date()}')
print(f'    mean={fout["factor_value"].mean():.4f}  std={fout["factor_value"].std():.4f}')
