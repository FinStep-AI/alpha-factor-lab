import pandas as pd, numpy as np, scipy.stats as sc
import warnings, json, os, sys
warnings.filterwarnings('ignore')
from datetime import datetime

BASE = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'
os.chdir(BASE)

# ── 6.1 load failed factor ────────────────────────────────────────────────────
factor_csv = 'data/gap_contrib_v1.csv'
rpt        = 'output/gap_contrib_v1/backtest_report.json'
fcsv       = pd.read_csv(factor_csv, parse_dates=['date'], dtype={'stock_code':str})
fcsv['stock_code'] = fcsv['stock_code'].str.zfill(6)
rep = json.load(open(rpt))

ic_mean  = rep.get('ic_mean',        None) or rep.get('IC均值')
ic_t     = rep.get('ic_t',           None) or rep.get('IC_t')
long_short = rep.get('long_short_sharpe', None) or rep.get('多空Sharpe')
mono     = rep.get('monotonicity',   None) or rep.get('单调性')
period   = rep.get('period',         rep.get('区间',''))
g_sharpes_annualized = rep.get('group_sharpe_annualized') or rep.get('group_sharpe')

# Snapshot of broken/bad indices
ic_sign_rate = rep.get('IC>0率', ic_csv := rep.get('ic_sign_rate'))

# 6.2 load research log ────────────────────────────────────────────────────────
res_path='data/factor-research.json'
if os.path.exists(res_path):
    research=json.load(open(res_path))
    if not isinstance(research,list): research=[]
else:
    research=[]

# grab IC CSV for reference
ic_csv_path = 'output/gap_contrib_v1/ic_series.csv'
ic_csv_snap = None
if os.path.exists(ic_csv_path):
    tmp = pd.read_csv(ic_csv_path)
    tmp.columns=[c.lower() for c in tmp.columns]
    ic_csv_snap = str(tmp.tail(5).to_dict(orient='records'))

entry = {
  'factor_id':        'gap_contrib_v1',
  'date':             '2026-05-27',
  'source_type':      'academic',
  'source_title':     'Overnight/Intraday Contribution Factor — QuantaAlpha-informed direction v2',
  'source_url':       'https://arxiv.org/abs/2602.07085',
  'factor_formula':   'raw = MA20(|close/prevClose-1|) / MA20(|close/open-1|) ; factor = OLS-residual(raw | log_amount) ; MAD winsor 5.5σ ; z-score',
  'neutralize':       'log_amount OLS neutralize + MAD winsor + z-score',
  'original_metric':  'IC target > 0.02 | t > 2.0 | Sharpe > 0.8 | mono > 0.8  (QuantaAlpha target)',
  'our_metric':       {
      'ic_mean': ic_mean, 'ic_t': ic_t,
      'long_short_sharpe': long_short, 'monotonicity': mono,
      'period': period
  },
  'ic_series_tail':   ic_csv_snap,
  'group_returns':    rep.get('group_ann_ret'),
  'group_sharpe':     g_sharpes_annualized,
  'turns':            False,
  'diff_notes': (
      'H1 confirmed: overnight return indeed dominates for CSI1000 when intraday predictability '
      'deteriorates (QuantaAlpha observation evaluates to gap_contrib). '
      'However cross-sectional ranking failure: IC ~-0.004, t=-0.48, Sharpe~0.04, mono=0.10. '
      'Fifth-group spread pattern not monotone (G3 peaks, G1/G5 tie low) → factor-ranked signal '
      'doesn\'t separate winners from losers on a stock-by-stock basis. '
      'Variant `gap_contrib_neg_v1` IC +0.004 t=+0.48 is symmetrical failure (flip/neg variant). '
      'Suspect main reason: eikon thin-liquidity effect of CSI1000 fully loads with reverse of sign '
      'night-day ratio — stock-by-stock IC not working. '
      'Conclude: concept valid but daily-data proxy is too noisy; ordering published as reference for '
      'informing兄弟们 aware — NOT a usable replacement (gap_contrib_not_replaced)'
  ),
  'conclusion':       'FAIL — not stocked; log as prior negative validations study caveat',
  'related_factors':  ['overnight_momentum_v1','gap_efficiency_v2','gap_momentum_v1']
}
research.append(entry)
json.dump(research, open(res_path,'w'), ensure_ascii=False, indent=2)
print('wrote', res_path, 'entries', len(research))
# ── quick counter-fire check, produce our export, state the finishing summary ───
# 6.3 investment safety: next iteration call deliberate_quant_comms verification
print('DIAG_IC', ic_mean, 'IC_T', ic_t, 'SHARPE', long_short, 'MONO', mono)
