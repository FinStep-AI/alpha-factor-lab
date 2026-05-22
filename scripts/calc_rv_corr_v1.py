"""
因子：滚动量价相关系数（Return–Volume Correlation, RV_Corr）
ID: rv_corr_v1

逻辑（跟清华 2025「A股零售交易者行为」呼应，但焦点转向截面横截面选股）：

  传统观点：高换手率 = 情绪高 → 反转
  本因子把"情绪"细化为「涨/跌阶段到底是放量还是缩量」：
    rv_corr = corr(ret_t, vol_t)  过去 20 个自然日

  正 rv_corr  → 上涨放量、下跌缩量 → 市场参与者以买入接盘为主
  负 rv_corr  → 上涨缩量、下跌放量 → 参与者以卖出止损为主

  在中证 1000 这样的中小盘里：
    · 负 rv_corr 往往意味着近期总是"跌放量、涨缩量"；
      市场主要以卖出救济止损的方式交易，价格发现已钝化 →
      此后这类股票如果放量停止，更可能出现补涨。
      这就是新的"放量滞跌→后续反转"信号。

  值高 = rv_corr 高（正量价协同），期待继续动量延续
  值低 = rv_corr 低/负（放量滞跌），期待反转
  
  按遗憾规避解释：能接受亏损的投资者已卖出，接着看涨跌量能是否反转融合。

Barra 风格：Sentiment / Micro-structure
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'skills/alpha-factor-lab/scripts')

import numpy as np
import pandas as pd
from factor_calculator import neutralize_cross_section

KLINE = 'data/csi1000_kline_raw.csv'
OUT   = 'data/rv_corr_v1.csv'
WIN   = 20

# ── load ────────────────────────────────────────────────────────────
print("Loading …")
df = pd.read_csv(KLINE, usecols=['date','stock_code','open','close','high','low','volume','pct_change'])
df['date']  = pd.to_datetime(df['date'])
for c in ['close','volume','pct_change']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['close','volume'])
df = df.sort_values(['stock_code','date']).reset_index(drop=True)

# ── rolling rv_corr ─────────────────────────────────────────────────
print(f"Rolling {WIN}-d corr(ret, vol) …")

def rolling_rv_corr(ret_s: pd.Series, vol_s: pd.Series, win: int) -> pd.Series:
    """rolling corr of two aligned float series, min_periods = win*0.7"""
    r = ret_s.rolling(win, min_periods=max(int(win*0.7), 4)).corr(vol_s)
    return r

results = []
for code, g in df.groupby('stock_code', sort=False):
    g = g.sort_values('date').copy()
    rv = rolling_rv_corr(g['pct_change'], g['volume'], WIN)
    tmp = g[['date','stock_code']].copy()
    tmp['rv_corr_raw'] = rv.values
    results.append(tmp)

corr_df = pd.concat(results, ignore_index=True)
df = df.merge(corr_df, on=['date','stock_code'], how='left')

# ── 5-d mini-MA smoother ────────────────────────────────────────────
df['raw_factor'] = df.groupby('stock_code')['rv_corr_raw'].transform(
    lambda s: s.rolling(5, min_periods=3).mean()
)

# ── neutralize ──────────────────────────────────────────────────────
print("Neutralizing …")
df['log_amount_20d'] = df.groupby('stock_code')['volume'].transform(
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
