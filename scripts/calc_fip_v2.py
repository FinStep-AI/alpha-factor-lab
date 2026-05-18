#!/usr/bin/env python3
"""
fip_v2 — 信息离散度因子（Frog-in-the-Pan, 本土化升级版）

论文
  原始: Da, Gurun & Warachka (2014) "Frog in the Pan: Continuous Information and
        Momentum", Review of Financial Studies, 27(7), 2171-2218.
       https://academic.oup.com/rfs/article-abstract/27/7/2171/1578455
  市场状态扩展: Galvani (2024) "Frog in the Pan and the market-state effect on
        momentum", Finance Research Letters, 63(C).
       https://ideas.repec.org/a/eee/finlet/v63y2024ics1544612324004045.html

核心改进（vs v1）
  (1) 引入 20-d 市场状态过滤：只在 UP state 下做 FIP 信号，
      DOWN state 下翻转符号（Galvani 2024 关键结论：FIP 效应只在上涨市场存在）。
  (2) 用 frac_pos（0~3% 小幅正收益天数）作为连续信息代理；
      不限制上界，让"渐近小幅增量"贡献天然累积。
  (3) 信号归一化：除以 20d 总天数，乘以市场状态方向，再统一截面标准化。

构造
  raw_fip = sign(market_cum20) × (frac_small_pos_20d − frac_small_neg_20d)
  mkt_state = sign(20d_cum_log_mkt_ret)          # +1 UP / −1 DOWN
  factor = neutralize( mkt_state × raw_fip, log_amount_20d, MAD+z-score )

Column output: date | stock_code | factor_value
"""
import sys, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('fip_v2')

# ─── paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
KLINE  = BASE / 'data' / 'csi1000_kline_raw.csv'
RET    = BASE / 'data' / 'csi1000_returns.csv'
OUT    = BASE / 'data' / 'factor_fip_v2.csv'
OUT_META = BASE / 'output' / 'fip_v2' / 'meta.json'

# ─── config ─────────────────────────────────────────────────────────────────
WINDOW     = 20      # rolling window for daily FIP signal
MKT_WINDOW = 20      # market state look-back (trading days)
LOW_THR    = 0.0     # 底部阈值 = 0%（只排除零收益日）
HIGH_THR   = 0.03    # 顶部阈值 = 3%（>3% 视为大张，不属于"小幅"传播）
NEUTRAL    = 'log_amount_20d'


# ─── step 1 load ─────────────────────────────────────────────────────────────
log.info('loading kline …')
kline = pd.read_csv(KLINE, parse_dates=['date'])
kline = kline[kline['amount'] > 0].copy()          # 去掉零成交行
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)

# 防止 pct_change 异常值（涨跌停那天 pct_change 会被读到极大/极小值）
pct = kline['pct_change'].clip(-21, 21)            # A 股 ±20% 限制
kline['ret'] = pct / 100.0

# ─── step 2 market state ────────────────────────────────────────────────────
log.info('computing market state (120d log cum-return) …')
mkt_ret = kline.groupby('date')['ret'].mean().rename('mkt_ret').reset_index()
mkt_ret['log_ret'] = np.log1p(mkt_ret['mkt_ret'])
mkt_ret['mkt_cum'] = mkt_ret['log_ret'].rolling(MKT_WINDOW, min_periods=MKT_WINDOW).sum()
# 用120d cumulative 市场状态——Galvani (2024) 用的是 protracted（持续）市场走向
mkt_ret['mkt_cum120'] = mkt_ret['log_ret'].rolling(120, min_periods=120).sum()
mkt_ret['mkt_state_up'] = (mkt_ret['mkt_cum120'] > 0).astype(int)   # 1=UP, 0=DOWN
log.info(f'market state UP ratio = {mkt_ret["mkt_state_up"].mean():.3f}')
up_days = int(mkt_ret['mkt_state_up'].sum())
dn_days = int((1 - mkt_ret['mkt_state_up']).sum())
log.info(f'UP={up_days} days, DOWN={dn_days} days')

# ─── step 3 per-stock small-win-day fraction ────────────────────────────────
log.info(f'computing per-stock FIP signal (window={WINDOW}, LOW_THR={LOW_THR}, HIGH_THR={HIGH_THR}) …')

def _small_frac(group: pd.DataFrame) -> pd.Series:
    """Compute frac_small_pos and frac_small_neg for one stock, 20d rolling."""
    r = group['ret'].values
    out = pd.DataFrame(index=group.index)
    out['small_pos'] = 0.0
    out['small_neg'] = 0.0
    n = len(r)
    for i in range(WINDOW - 1, n):
        win = r[i - WINDOW + 1: i + 1]
        out.iloc[i, out.columns.get_loc('small_pos')] = np.mean(
            (win >  LOW_THR) & (win <  HIGH_THR))
        out.iloc[i, out.columns.get_loc('small_neg')] = np.mean(
            (win < -LOW_THR) & (win > -HIGH_THR))
    return out

grp = kline.groupby('stock_code', sort=False)
parts = []
for i, (code, g) in enumerate(grp):
    g = g.sort_values('date')
    ss = _small_frac(g)
    tmp = g[['date', 'stock_code', 'amount', 'ret']].copy()
    tmp['frac_pos_20d'] = ss['small_pos'].values
    tmp['frac_neg_20d'] = ss['small_neg'].values
    parts.append(tmp)
    if (i + 1) % 100 == 0:
        log.info(f'  stocks done: {i + 1}/1000')

kline_sig = pd.concat(parts, ignore_index=True)
kline_sig['raw_fip'] = kline_sig['frac_pos_20d'] - kline_sig['frac_neg_20d']     # direction = + small_wins
log.info('raw FIP stats:\n' +
         kline_sig['raw_fip'].describe().to_string())

# ─── step 4 merge market state ───────────────────────────────────────────────
kline_sig = kline_sig.merge(
    mkt_ret[['date', 'mkt_state_up']], on='date', how='left')

# ─── step 5 market-state-adjusted factor sign ────────────────────────────────
# UP   → keep raw sign  (FIP 效应存在，连续小幅正收益 = 正向信号)
# DOWN → flip sign      (Galvani 2024: FIP 效应在 DOWN 消失/反转)
kline_sig['state_fip'] = kline_sig['raw_fip'] * np.where(
    kline_sig['mkt_state_up'] == 1, 1.0, -1.0)

# ─── step 6 neutralize + standardize ────────────────────────────────────────
log.info('neutralizing (log_amount OLS) + MAD + z-score …')

kline_sig['log_amount_20d'] = (
    kline_sig.groupby('stock_code')['amount']
    .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
)

def cross_section_neutralize(daily: pd.DataFrame) -> pd.Series:
    """OLS neutralize factor against log_amount, then MAD + z-score."""
    vals = daily['state_fip'].values
    neutralizer = daily['log_amount_20d'].values
    mask = np.isfinite(vals) & np.isfinite(neutralizer)
    out = np.full(len(vals), np.nan)
    if mask.sum() < 30:
        return pd.Series(out, index=daily.index)
    x = neutralizer[mask]
    y = vals[mask]
    slope, intercept, _, _, _ = stats.linregress(x, y)
    resid = y - (intercept + slope * x)
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med)) * 1.4826
    if mad < 1e-9:
        out[mask] = 0.0
    else:
        clipped = np.clip((resid - med) / mad, -5, 5)
        s = np.nanstd(clipped)
        out[mask] = 0.0 if s < 1e-9 else (clipped - np.nanmean(clipped)) / s
    return pd.Series(out, index=daily.index)

results = []
for dt, grp_d in kline_sig.groupby('date', sort=False):
    f = cross_section_neutralize(grp_d)
    sub = grp_d[['date', 'stock_code']].copy()
    sub['factor_value'] = f.values
    results.append(sub.dropna(subset=['factor_value']))

factor_df = pd.concat(results, ignore_index=True)
log.info(f'factor rows: {len(factor_df)}  dates: {factor_df["date"].nunique()}')

# ─── step 7 persist ─────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
factor_df.to_csv(OUT, index=False)

meta = {
    'factor_id': 'fip_v2',
    'factor_name': '信息离散度(FIP) 市场状态适应版',
    'source_type': '论文复现+本土化',
    'source_title': 'Frog in the Pan: Continuous Information and Momentum + '
                    'market-state extension (Galvani 2024)',
    'source_url': [
        'https://academic.oup.com/rfs/article-abstract/27/7/2171/1578455',
        'https://ideas.repec.org/a/eee/finlet/v63y2024ics1544612324004045.html'
    ],
    'source_authors': 'Da, Gurun & Warachka (2014); Galvani (2024)',
    'description': (
        '20日小幅正/负收益天数的截面差，乘以120日市场状态方向。'
        'Galvani(2024)表明FIP效应只存在于UP市场，因此DOWN期间翻转符号。'
        f'窗口={WINDOW}d, 低阈值={LOW_THR}, 高阈值={HIGH_THR}, 市场状态={MKT_WINDOW}/120d.'
    ),
    'formula': (
        'raw_fip = frac_pos_{TH} − frac_neg_{TH}; '
        'mkt_state = sign(cum_log_ret_120d); '
        'factor = neutralize(mkt_state × raw_fip, log_amt_20d)'
    ),
    'construction': {
        'window': WINDOW, 'low_thr': LOW_THR, 'high_thr': HIGH_THR,
        'mkt_window': MKT_WINDOW, 'mkt_cum_long': 120,
        'neutralizer': NEUTRAL, 'direction': 'context-dependent (UP=+, DOWN=−)'
    },
    'data_range': {'start': str(kline_sig['date'].min().date()),
                   'end':   str(kline_sig['date'].max().date())},
    'market_state_up_ratio': round(float(kline_sig['mkt_state_up'].mean().item()), 3),
    'rows': len(factor_df), 'dates': int(factor_df['date'].nunique()),
}
OUT_META.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_META, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

log.info(f'saved → {OUT}')
log.info(f'meta  → {OUT_META}')
