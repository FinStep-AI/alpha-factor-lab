#!/usr/bin/env python3
import numpy as np
import pandas as pd

KLINE = 'data/csi1000_kline_raw.csv'
OUT = 'data/factor_boll_breakout_intensity_v1.csv'


def cs_winsorize(s, n=5.0):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    bound = 1.4826 * mad * n
    return s.clip(med - bound, med + bound)


def cs_neutralize(df, factor_col, x_col):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby('date'):
        y = g[factor_col].astype(float)
        x = g[x_col].astype(float)
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 20:
            continue
        yy = y[mask].values
        xx = x[mask].values
        X = np.column_stack([np.ones(len(xx)), xx])
        beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        resid = yy - X @ beta
        out.loc[g.index[mask]] = resid
    return out


def cs_zscore(s):
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def rolling_pct_rank(x, window=60, minp=30):
    def _last_rank(arr):
        s = pd.Series(arr)
        return s.rank(pct=True).iloc[-1]
    return x.rolling(window, min_periods=minp).apply(_last_rank, raw=False)


def main():
    df = pd.read_csv(KLINE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    grouped = df.groupby('stock_code', group_keys=False)

    df['ma20'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df['std20'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=15).std())
    df['upper'] = df['ma20'] + 2 * df['std20']
    df['bbw'] = (4 * df['std20']) / df['ma20'].replace(0, np.nan)
    df['bbw_pct'] = grouped['bbw'].transform(lambda x: rolling_pct_rank(x, 60, 30))

    df['ret5'] = grouped['close'].pct_change(5)
    df['amt20'] = grouped['amount'].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df['vol20'] = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df['amt_ratio'] = df['amount'] / df['amt20'].replace(0, np.nan)

    breakout = (df['close'] / df['upper'].replace(0, np.nan)) - 1.0
    squeeze = 1.0 - df['bbw_pct']
    raw = squeeze * breakout * np.log1p(df['amt_ratio'].clip(lower=0))
    raw = raw * np.tanh(df['ret5'].fillna(0) / 0.12)

    out = df[['date', 'stock_code']].copy()
    out['raw'] = raw
    out['log_mktcap_proxy'] = np.log(df['amt20'].clip(lower=1.0))
    out['raw_w'] = out.groupby('date')['raw'].transform(lambda s: cs_winsorize(s, 5.0))
    out['neutral'] = cs_neutralize(out, 'raw_w', 'log_mktcap_proxy')
    out['neutral_w'] = out.groupby('date')['neutral'].transform(lambda s: cs_winsorize(s, 5.0))
    out['factor_boll_breakout_intensity_v1'] = out.groupby('date')['neutral_w'].transform(cs_zscore)
    out = out[['date', 'stock_code', 'factor_boll_breakout_intensity_v1']]
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(OUT, index=False)
    print('saved', OUT, 'rows', len(out), 'valid', out.iloc[:,2].notna().sum())

if __name__ == '__main__':
    main()
