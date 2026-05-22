"""
因子：遗憾规避恢复（Regret Recovery）
ID: regret_recovery_v1

来源：Wang & Yang (2025) "Regret aversion and asset pricing anomalies in the
      Chinese stock market", *International Review of Finance*, vol.25(1).
      东财证券 2023-03 《基于遗憾规避理论的量化选股因子》。

逻辑（日K线可观测约化版，核心机制保留）：
  遗憾规避的选股效应 = 当投资者被迫确认已持有多头 "认亏出局" 后，
  后续该股走势出现分化 → 那些经历深度回撤但当日结构已转强的股票，
  正是 regret seller 已清仓、新冷静买家已接手的信号，此后 tend to rebound。

构造：
  ① regret_body_t
     = -(close_t - open_t) / (high_t - low_t)        (负数 = 大阴线)
     × I(close_t 位于当日下 1/3 区间)
     clip 到 (−1, 0]
     过去 6 个交易日求和 → regret_intensity

  ② close_pos_above_mid_freq
     过去 6 个交易日中 close 位于 (open+high)/2 以上的天数占比
     = 近 6 日内投资者看多 / 维持信心信号的天数比例

  ③ regime_proxy
     过去 20 日收益率均值（截面中性化后相当于相对市场状态）
     遗憾规避在熊市下更显著（IRF 2025 结论），veto bear-regime 股票

  因子值 composite（等权）：
      regret_comp = zscore(regret_intensity)
                   + zscore(close_pos_above_mid_freq)
                   + 0.5 × zscore(regime_proxy)

  值高 → "刚经历过深度遗憾但已恢复上攻意愿" 的股票

  ※ 因子库已有极端负收益日频率(extreme_neg_day_freq_v1) →
    本因子额外引入 close-position 结构与 market-regime 分量，不是简单重复。

Barra 风格：Reversal / Sentiment
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'skills/alpha-factor-lab/scripts')

import numpy as np
import pandas as pd
from factor_calculator import neutralize_cross_section

KLINE = 'data/csi1000_kline_raw.csv'
OUT   = 'data/regret_recovery_v1.csv'

# ── load ────────────────────────────────────────────────────────────
print("Loading …")
df = pd.read_csv(KLINE, usecols=['date','stock_code','open','close','high','low','amount','pct_change'])
df['date']     = pd.to_datetime(df['date'])
for c in ['open','close','high','low','amount','pct_change']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['close','high','low'])
df = df.sort_values(['stock_code','date']).reset_index(drop=True)

# ── daily helpers ───────────────────────────────────────────────────
rng    = (df['high'] - df['low']).clip(lower=df['close'] * 0.001)
cp     = ((df['close'] - df['low']) / rng).clip(0, 1)
# please-clip: I(close <= (open+high)/2)  →  1 = 收盘 ≤ 日中位线
below_mid   = (df['close'] <= (df['open'] + df['high']) / 2).astype(float)

# black-body intensity  (negative: big black candle)
body_proxy  = -np.where(
    rng < df['close'] * 0.001,
    0.0,
    (df['close'] - df['open']) / rng   # (<0 black, >0 white)
).clip(-1, 0)

# ══ rolling 6-day windows ═══════════════════════════════════════════
WIN  = 6
WIN2 = 20

def roll_mean(g, col, w, mp):
    return g[col].transform(lambda s: s.rolling(w, min_periods=mp).mean())

def roll_sum(g, col, w, mp):
    return g[col].transform(lambda s: s.rolling(w, min_periods=mp).sum())

print("Rolling 6d regret intensity …")
df['body_proxy'] = body_proxy
df['below_mid']  = below_mid

df['regret_intensity'] = (
    df.groupby('stock_code')
      .apply(lambda g: roll_sum(g, 'body_proxy', WIN, 4))
      .reset_index(level=0, drop=True)
)
# regret_intensity is negative; negate so HIGHER = more regret
df['regret_intensity'] = -df['regret_intensity']

print("Rolling 6d close-above-mid frequency …")
df['pos_signal'] = 1 - below_mid    # 1 = close > mid = bullish on that day
df['close_above_mid_freq'] = (
    df.groupby('stock_code')
      .apply(lambda g: roll_mean(g, 'pos_signal', WIN, 4))
      .reset_index(level=0, drop=True)
)

# ── market regime proxy ══════════════════════════════════════════════
print("Rolling 20d return proxy …")
df['ret_5d'] = df.groupby('stock_code')['pct_change'].transform(
    lambda s: s.rolling(5, min_periods=3).sum()
)
df['regime_proxy'] = (
    df.groupby('stock_code')
      .apply(lambda g: roll_mean(g, 'ret_5d', WIN2, 10))
      .reset_index(level=0, drop=True)
)

# ── cross-sectional z per date ════════════════════════════════════════
def cs_z(g, col, out_col):
    v = g[col].values.astype(float)
    m = np.isfinite(v)
    if m.sum() < 5:
        g[out_col] = np.nan; return g
    mu, sd = v[m].mean(), v[m].std()
    r = np.where(m, np.where(sd > 1e-12, (v - mu) / sd, 0.0), np.nan)
    g[out_col] = r
    return g

print("Cross-sectional z-score …")
for col, out in [('regret_intensity','z_regret'),
                  ('close_above_mid_freq','z_midfreq'),
                  ('regime_proxy','z_regime')]:
    df = df.groupby('date', group_keys=False).apply(lambda g: cs_z(g, col, out))

df['raw_factor'] = df['z_regret'] + df['z_midfreq'] + 0.5 * df['z_regime']

# market-cap neutralise
print("Neutralizing …")
df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
    lambda s: np.log(s.rolling(20, min_periods=10).mean().clip(lower=1))
)

ready = df[['date','stock_code','raw_factor','log_amount_20d']].dropna().copy()
ready = ready.sort_values(['date','stock_code']).reset_index(drop=True)
ready['factor_value'] = neutralize_cross_section(
    ready, 'raw_factor', neutralize_cols=['log_amount_20d']
)

out = ready[['date','stock_code','factor_value']].dropna()
out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
out.to_csv(OUT, index=False)
print(f"Done  {len(out)} rows → {OUT}")
print(f"  {out['date'].min()}  ~  {out['date'].max()}")
print(out.tail(5).to_string(index=False))
