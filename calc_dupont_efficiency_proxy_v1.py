import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
KLINE = BASE/'data/csi1000_kline_raw.csv'
FUND = BASE/'data/csi1000_fundamental_cache.csv'
OUT = BASE/'data/factor_dupont_efficiency_proxy_v1.csv'

k = pd.read_csv(KLINE)
f = pd.read_csv(FUND)

k['date'] = pd.to_datetime(k['date'])
f['report_date'] = pd.to_datetime(f['report_date'])
k['stock_code'] = k['stock_code'].astype(str)
f['stock_code'] = f['stock_code'].astype(str)

# 市值代理：amount / turnover
k['turnover'] = pd.to_numeric(k['turnover'], errors='coerce')
k['amount'] = pd.to_numeric(k['amount'], errors='coerce')
k['close'] = pd.to_numeric(k['close'], errors='coerce')
k['open'] = pd.to_numeric(k['open'], errors='coerce')
k['high'] = pd.to_numeric(k['high'], errors='coerce')
k['low'] = pd.to_numeric(k['low'], errors='coerce')
k['pct_change'] = pd.to_numeric(k['pct_change'], errors='coerce')/100.0

k['mktcap_proxy'] = k['amount'] / (k['turnover']/100.0).replace(0, np.nan)
k['log_mktcap'] = np.log(k['mktcap_proxy'].clip(lower=1))

# 价格端低噪声确认
k = k.sort_values(['stock_code','date']).copy()
g = k.groupby('stock_code', group_keys=False)
k['ret_5d'] = g['close'].pct_change(5)
k['vol_20'] = g['pct_change'].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
k['amp_20'] = g['amplitude'].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
k['intraday_eff'] = (k['close'] - k['open']) / (k['high'] - k['low']).replace(0, np.nan)
k['intraday_eff_10'] = g['intraday_eff'].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)

# 基本面衍生：用 BPS 增长近似权益扩张，用 ROE 改善代理盈利效率改善
f = f.sort_values(['stock_code','report_date']).copy()
fg = f.groupby('stock_code', group_keys=False)
f['roe_lag4'] = fg['roe'].shift(4)
f['roe_lag1'] = fg['roe'].shift(1)
f['bps_lag4'] = fg['bps'].shift(4)
f['bps_lag1'] = fg['bps'].shift(1)

f['roe_yoy'] = f['roe'] - f['roe_lag4']
f['roe_qoq'] = f['roe'] - f['roe_lag1']
f['bps_yoy'] = f['bps'] / f['bps_lag4'].replace(0, np.nan) - 1
f['bps_qoq'] = f['bps'] / f['bps_lag1'].replace(0, np.nan) - 1

# 杜邦/效率代理：ROE改善 but 资本扩张克制；叠加稳定性
f['eff_proxy'] = (
    0.50*np.tanh(f['roe_yoy']/4.0) +
    0.25*np.tanh(f['roe_qoq']/2.0) -
    0.30*np.tanh(f['bps_yoy']/0.20) -
    0.10*np.tanh(f['bps_qoq']/0.08)
)

# 财报按45天滞后生效
f['effective_date'] = f['report_date'] + pd.Timedelta(days=45)
use_cols = ['stock_code','effective_date','eff_proxy','roe_yoy','roe_qoq','bps_yoy','bps_qoq']
f2 = f[use_cols].sort_values(['stock_code','effective_date'])

# merge_asof 映射到日频
res = []
for sc, kd in k.groupby('stock_code'):
    fd = f2[f2['stock_code']==sc].sort_values('effective_date')
    if fd.empty:
        continue
    merged = pd.merge_asof(
        kd.sort_values('date'),
        fd.drop(columns=['stock_code']).sort_values('effective_date'),
        left_on='date', right_on='effective_date', direction='backward'
    )
    merged['stock_code'] = sc
    res.append(merged)

df = pd.concat(res, ignore_index=True)

# 组合信号：效率改善 + 价格低噪声确认 + 短期不过热
raw = (
    df['eff_proxy']
    + 0.15*np.tanh(df['intraday_eff_10']/0.25)
    - 0.12*np.tanh(df['vol_20']/0.04)
    - 0.08*np.tanh(df['amp_20']/6.0)
    - 0.10*np.tanh(df['ret_5d']/0.12)
)
df['raw_factor'] = raw

# 横截面市值中性化

def neutralize(sub):
    sub = sub[['stock_code','raw_factor','log_mktcap']].dropna().copy()
    if len(sub) < 20:
        return pd.DataFrame({'stock_code': sub['stock_code'], 'factor': np.nan})
    y = sub['raw_factor'].values
    x = sub['log_mktcap'].values
    mask = np.isfinite(y) & np.isfinite(x)
    sub = sub.loc[mask].copy()
    if len(sub) < 20:
        return pd.DataFrame({'stock_code': sub['stock_code'], 'factor': np.nan})
    X = np.column_stack([np.ones(len(sub)), sub['log_mktcap'].values])
    beta = np.linalg.lstsq(X, sub['raw_factor'].values, rcond=None)[0]
    resid = sub['raw_factor'].values - X @ beta
    mu = np.nanmean(resid)
    sd = np.nanstd(resid)
    z = (resid - mu) / sd if sd and np.isfinite(sd) and sd > 1e-8 else resid*0
    return pd.DataFrame({'stock_code': sub['stock_code'].values, 'factor': z})

out = []
for dt, sub in df.groupby('date'):
    nz = neutralize(sub)
    nz['date'] = dt
    out.append(nz)

factor = pd.concat(out, ignore_index=True)[['date','stock_code','factor']].dropna()
factor['date'] = factor['date'].dt.strftime('%Y-%m-%d')
factor.to_csv(OUT, index=False)
print('saved', OUT, 'rows', len(factor), 'dates', factor['date'].nunique(), 'stocks', factor['stock_code'].nunique())
print(factor.head().to_string())
