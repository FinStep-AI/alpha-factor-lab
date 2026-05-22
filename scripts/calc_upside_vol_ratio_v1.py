"""
因子：上涨日量能占比（Upside Volume Ratio / UVR）
ID: upside_vol_ratio_v1

逻辑：
  区分两类成交信号：
    · 上涨日成交量 → 买方主导 / 信仰注入（smart buyer 逐步建仓）
    · 下跌日成交量 → 卖方主导 / regret／恐慌卖出
  本因子度量「过去 20 日里，上涨日成交量占总成交量的比例」。
  比例越高 → 增量资金以买入方式入场 → 价格发现向上推进

  与现有因子的增量信息：
    - 换手率水平(v1)               → 绝对换手量级，不区分方向
    - 换手率水平v1                 → 同上
    - 换手率-量价量残差因子        → 总量维度残差，仍无涨跌分工
    - ★本因子是第一个区分「涨日 vs 跌日 量能分工」的选股因子

Barra 风格：Liquidity / 行为金融
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'skills/alpha-factor-lab/scripts')

import numpy as np
import pandas as pd
from factor_calculator import neutralize_cross_section

KLINE = 'data/csi1000_kline_raw.csv'
OUT   = 'data/upside_vol_ratio_v1.csv'
WIN   = 20

# ── 1. load ──────────────────────────────────────────────────────────
print("Loading …")
df = pd.read_csv(KLINE, usecols=['date','stock_code','open','close','high','low','volume','amount','pct_change'])
df['date']   = pd.to_datetime(df['date'])
for c in ['close','volume','amount','pct_change']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['close','volume'])
df = df.sort_values(['stock_code','date']).reset_index(drop=True)

# ── 2. up-day bool + up-volume / dn-volume ───────────────────────────
print("Computing up/down day volumes …")
df['is_up']   = (df['pct_change'] > 0).astype(float)
df['up_vol']  = df['volume'] * df['is_up']
df['dn_vol']  = df['volume'] * (1 - df['is_up'])

# ── 3. rolling sums ──────────────────────────────────────────────────
print(f"Rolling {WIN}-d sums …")
def rsum(g, col):
    return g[col].transform(lambda s: s.rolling(WIN, min_periods=int(WIN*0.7)).sum())

df['up_vol_sum']  = df.groupby('stock_code').apply(lambda g: rsum(g,'up_vol')).reset_index(level=0,drop=True)
df['dn_vol_sum']  = df.groupby('stock_code').apply(lambda g: rsum(g,'dn_vol')).reset_index(level=0,drop=True)
df['vol_sum']     = df.groupby('stock_code').apply(lambda g: rsum(g,'volume')).reset_index(level=0,drop=True)

# ── 4. upside volume ratio ───────────────────────────────────────────
df['raw_factor'] = np.where(
    df['vol_sum'] > 0,
    (df['up_vol_sum'] - df['dn_vol_sum']) / df['vol_sum'].clip(lower=1),
    0.0
)
# raw_factor  ∈ (−1, 1) :  +1 = all up-day volume, −1 = all down-day volume

# WIN5 期均值平滑
df['raw_factor'] = df.groupby('stock_code')['raw_factor'].transform(
    lambda s: s.rolling(5, min_periods=3).mean()
)

# ── 5. neutralize by log_amount_20d ──────────────────────────────────
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
