
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

"""
因子：ROE改善估值约束残差 v2
ID: roe_improve_valuation_resid_v2

逻辑：
  用季度ROE同比改善（growth/quality）减去估值拥挤（PB）的线性解释部分，
  提取“ROE在改善，但估值尚未充分反映”的截面残差。
  再做市值代理（log amount）中性化，得到更偏 Growth/Quality 的纯因子。

构造：
  1) 对每个股票按财报序列计算 roe_yoy = roe_t - roe_t-4
  2) 以 bps × close 构造 PB = close / bps
  3) 每个交易日使用最近已披露财报（report_date <= trade_date）
  4) factor_raw = residual of (roe_yoy ~ log(pb)) cross-sectionally
  5) 再对 factor_raw 做 (factor_raw ~ log_amount) OLS 中性化 + MAD 缩尾 + zscore
"""


def cs_residual(y, x, min_n=30):
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < min_n:
        return np.full(len(y), np.nan)
    yy = y[mask]
    xx = x[mask]
    X = np.column_stack([np.ones(len(xx)), xx])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    resid = yy - X @ beta
    out = np.full(len(y), np.nan)
    out[np.where(mask)[0]] = resid
    return out


def neutralize(values, control, min_n=30):
    values = np.asarray(values, dtype=float)
    control = np.asarray(control, dtype=float)
    mask = np.isfinite(values) & np.isfinite(control)
    if mask.sum() < min_n:
        return np.full(len(values), np.nan)

    v = values[mask]
    c = control[mask]
    X = np.column_stack([np.ones(len(c)), c])
    beta = np.linalg.lstsq(X, v, rcond=None)[0]
    r = v - X @ beta

    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad < 1e-12:
        return np.full(len(values), np.nan)
    r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
    std = r.std()
    if std < 1e-12:
        return np.full(len(values), np.nan)

    z = (r - np.median(r)) / std
    out = np.full(len(values), np.nan)
    out[np.where(mask)[0]] = z
    return out


def main():
    print('Loading kline...')
    k = pd.read_csv('data/csi1000_kline_raw.csv', usecols=['date','stock_code','close','amount'])
    k['date'] = pd.to_datetime(k['date'])
    k = k.sort_values(['date','stock_code']).reset_index(drop=True)

    print('Loading fundamentals...')
    f = pd.read_csv('data/csi1000_fundamental_cache.csv', usecols=['stock_code','report_date','bps','roe'])
    f['report_date'] = pd.to_datetime(f['report_date'])
    f = f.sort_values(['report_date','stock_code']).reset_index(drop=True)

    print('Computing quarterly ROE YoY improvement...')
    f['roe_yoy'] = f.groupby('stock_code')['roe'].transform(lambda s: s - s.shift(4))

    # use quarter-end close as price anchor for PB on report date, then carry to trading dates via merge_asof
    qpx = k[['date','stock_code','close']].rename(columns={'date':'report_date','close':'trade_close'}).sort_values(['report_date','stock_code']).reset_index(drop=True)
    merged_q = pd.merge_asof(
        f,
        qpx,
        by='stock_code',
        on='report_date',
        direction='backward'
    )
    merged_q['pb'] = merged_q['trade_close'] / merged_q['bps']
    feat = merged_q[['stock_code','report_date','roe_yoy','pb']].sort_values(['report_date','stock_code']).reset_index(drop=True)

    print('Aligning latest available report to each trading day...')
    kd = k[['stock_code','date','close','amount']].sort_values(['date','stock_code']).reset_index(drop=True)
    panel = pd.merge_asof(
        kd,
        feat,
        by='stock_code',
        left_on='date',
        right_on='report_date',
        direction='backward'
    )

    panel['log_pb'] = np.log(panel['pb'].clip(lower=0.05))
    panel['log_amount'] = np.log(panel['amount'].clip(lower=1))

    print('Cross-sectional residualization and market-cap neutralization...')
    out = []
    ndates = panel['date'].nunique()
    for i, (dt, g) in enumerate(panel.groupby('date'), 1):
        if i % 200 == 0:
            print(f'  {i}/{ndates} {dt.date()}')
        g = g.copy()
        y = g['roe_yoy'].to_numpy(dtype=float)
        x = g['log_pb'].to_numpy(dtype=float)
        raw = cs_residual(y, x, min_n=50)
        g['factor_raw'] = raw
        g = g.dropna(subset=['factor_raw','log_amount'])
        if len(g) < 50:
            continue
        g['factor_value'] = neutralize(g['factor_raw'].to_numpy(dtype=float), g['log_amount'].to_numpy(dtype=float), min_n=50)
        g = g.dropna(subset=['factor_value'])
        out.append(g[['date','stock_code','factor_value']])

    res = pd.concat(out, ignore_index=True)
    out_path = 'data/factor_roe_improve_valuation_resid_v2.csv'
    res.to_csv(out_path, index=False)
    print(f'Saved {out_path}, shape={res.shape}, dates={res.date.nunique()}, stocks={res.stock_code.nunique()}')
    print(res['factor_value'].describe())


if __name__ == '__main__':
    main()
