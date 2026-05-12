"""
Factor: Amplitude Spike Kurtosis (Volatility clustering via spike persistence)
ID: amp_spike_kurt_v1   Barra: Volatility

Logic (new direction):
  Existing vol factors use rolling std -> smooth.
  This factor quantifies spike persistence: exogenous bursts in amplitude rank
  show up as consecutive high-rank-amplitude days.
  Signal = 5-day moving average of daily cross-sectional rank(log_amplitude)
  High signal = recent amplitude rank is consistently high -> volatility clustering
  Neutralized by log_amount_ma20 OLS
"""
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_amp_spike_kurt_v1.csv'):
    df = pd.read_csv(kline_file)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 1. cross-sectional daily rank of log_amplitude
    df['log_amp'] = np.log(df['amplitude'].clip(lower=0.1) + 1)
    df['amp_rank'] = df.groupby('date')['log_amp'].transform(
        lambda x: x.rank(pct=True, method='average'))

    # 2. 5-day MA of rank -> spike persistence
    df['spike_strength'] = df.groupby('stock_code')['amp_rank'].transform(
        lambda x: x.rolling(5, min_periods=3).mean())

    # 3. cross-sectional MAD + z-score
    def zscoref(grp):
        vals = grp['spike_strength'].values.astype(float)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad < 1e-12:
            grp['factor_value'] = 0.0; return grp
        up, lo = med + 5.2*1.4826*mad, med - 5.2*1.4826*mad
        vc = np.clip(vals, lo, up)
        m, s = np.nanmean(vc), np.nanstd(vc)
        grp['factor_value'] = 0.0 if s < 1e-12 else (vc - m) / s
        return grp
    df = df.groupby('date', group_keys=False).apply(zscoref)
    df = df.dropna(subset=['factor_value'])

    # 4. OLS neutralization by log_amount_ma20
    df['log_amount'] = np.log(df['amount'].clip(lower=1) + 1)
    df['log_amount_ma20'] = df.groupby('stock_code')['log_amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    out = df[['date','stock_code','factor_value','log_amount_ma20']].dropna(
        subset=['factor_value','log_amount_ma20']).copy()
    res = []
    for dt, grp in out.groupby('date'):
        x = grp['log_amount_ma20'].values.astype(float)
        y = grp['factor_value'].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            grp['factor_neutral'] = 0.0; res.append(grp[['date','stock_code','factor_neutral']]); continue
        X = np.column_stack([np.ones(m.sum()), x[m]])
        beta, _, _, _ = np.linalg.lstsq(X, y[m], rcond=None)
        r = np.full(len(x), np.nan); r[m] = y[m] - X @ beta
        r_med, r_std = np.nanmedian(r[m]), np.nanstd(r[m])
        grp['factor_neutral'] = 0.0 if r_std < 1e-12 else (
            np.where(m, (r-r_med)/r_std, np.nan))
        res.append(grp[['date','stock_code','factor_neutral']])
    result = pd.concat(res, ignore_index=True).dropna(subset=['factor_neutral'])
    result = result.rename(columns={'factor_neutral':'factor_value'})
    result[['date','stock_code','factor_value']].sort_values(
        ['date','stock_code']).to_csv(output_file, index=False)
    print(f"Done: {len(result)} rows -> {output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kline',  default='data/csi1000_kline_raw.csv')
    p.add_argument('--output', default='data/factor_amp_spike_kurt_v1.csv')
    a = p.parse_args()
    compute_factor(a.kline, a.output)
