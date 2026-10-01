#!/usr/bin/env python3
import numpy as np
import pandas as pd

KLINE = 'data/csi1000_kline_raw.csv'
OUT = 'data/factor_boll_squeeze_breakout_confirm_v1.csv'


def cs_winsorize(s, n=5.0):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    bound = 1.4826 * mad * n
    return s.clip(med - bound, med + bound)


def cs_zscore(s):
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def cs_neutralize(df, factor_col, x_col):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby('date'):
        y = g[factor_col].astype(float)
        x = g[x_col].astype(float)
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        yy = y[mask].values
        xx = x[mask].values
        X = np.column_stack([np.ones(len(xx)), xx])
        beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        resid = yy - X @ beta
        out.loc[g.index[mask]] = resid
    return out


def rolling_pct_rank(x, window=60, minp=30):
    def _last_rank(arr):
        s = pd.Series(arr)
        return s.rank(pct=True).iloc[-1]
    return x.rolling(window, min_periods=minp).apply(_last_rank, raw=False)


def main():
    df = pd.read_csv(KLINE)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    g = df.groupby('stock_code', group_keys=False)

    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df['std20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=15).std())
    df['upper'] = df['ma20'] + 2.0 * df['std20']
    df['lower'] = df['ma20'] - 2.0 * df['std20']
    df['bbw'] = (df['upper'] - df['lower']) / df['ma20'].replace(0, np.nan)
    df['bbw_pct'] = g['bbw'].transform(lambda x: rolling_pct_rank(x, 80, 40))

    df['amt20'] = g['amount'].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df['amt5'] = g['amount'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['ret3'] = g['close'].pct_change(3)
    df['ret5'] = g['close'].pct_change(5)
    df['max_close_10'] = g['close'].transform(lambda x: x.shift(1).rolling(10, min_periods=7).max())

    squeeze_gate = (1.0 - df['bbw_pct']).clip(lower=0, upper=1)
    breakout_upper = ((df['close'] / df['upper'].replace(0, np.nan)) - 1.0).clip(lower=-0.3, upper=0.3)
    breakout_range = ((df['close'] / df['max_close_10'].replace(0, np.nan)) - 1.0).clip(lower=-0.3, upper=0.3)
    volume_confirm = np.log1p((df['amt5'] / df['amt20'].replace(0, np.nan)).clip(lower=0, upper=5))
    trend_confirm = np.tanh(df['ret5'].fillna(0) / 0.10)
    overheat_penalty = np.abs(np.tanh(df['ret3'].fillna(0) / 0.12))

    raw = squeeze_gate * (0.65 * breakout_upper + 0.35 * breakout_range) * (1.0 + volume_confirm) * trend_confirm
    raw = raw * (1.0 - 0.35 * overheat_penalty)

    out = df[['date', 'stock_code']].copy()
    out['raw'] = raw.replace([np.inf, -np.inf], np.nan)
    out['log_mktcap_proxy'] = np.log((df['close'] * df['amount'] / (df['turnover'].replace(0, np.nan) + 1e-6)).clip(lower=1.0))
    out['raw_w'] = out.groupby('date')['raw'].transform(lambda s: cs_winsorize(s, 5.0))
    out['neutral'] = cs_neutralize(out, 'raw_w', 'log_mktcap_proxy')
    out['neutral_w'] = out.groupby('date')['neutral'].transform(lambda s: cs_winsorize(s, 5.0))
    out['factor'] = out.groupby('date')['neutral_w'].transform(cs_zscore)

    out = out[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(OUT, index=False, float_format='%.6f')
    print('saved', OUT, 'rows', len(out), 'dates', out['date'].min(), out['date'].max())

if __name__ == '__main__':
    main()
