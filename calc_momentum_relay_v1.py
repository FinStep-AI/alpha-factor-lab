"""
因子：短中期动量接力 (Short-Intermediate Momentum Relay)
ID: momentum_relay_v1   Barra: Momentum

逻辑（全新方向，不与均线离散度重叠）：
  均线离散度(Dispersion) = MA均线间距离的波动；衡量的是"均线密集程度"
  接力因子(Momentum Relay) = 5日 vs 30日 的纯价格动能对比
  
  若某股票过去5日累积涨幅 > 过去30日累积涨幅 → 说明近5日涨速 > 整个30日日均唱速
  → 动量在加速，更多资金在涌入 → 短期可能继续上涨
  反之则动量减弱

信号：relay = rank_5d - rank_30d (截面rank差分)
中性化：log_amount_ma20 OLS
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_momentum_relay_v1.csv'):
    df = pd.read_csv(kline_file)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 5日 & 30日 累积收益
    df['ret_5d'] = df.groupby('stock_code')['pct_change'].transform(
        lambda x: x.rolling(5, min_periods=3).sum() / 100)
    df['ret_30d'] = df.groupby('stock_code')['pct_change'].transform(
        lambda x: x.rolling(30, min_periods=15).sum() / 100)

    # 截面 Rank (0~1)
    df['rank_5d'] = df.groupby('date')['ret_5d'].transform(
        lambda x: x.rank(pct=True, na_option='keep', method='average'))
    df['rank_30d'] = df.groupby('date')['ret_30d'].transform(
        lambda x: x.rank(pct=True, na_option='keep', method='average'))

    # 接力信号 = 短期rank - 长期rank
    df['relay'] = df['rank_5d'] - df['rank_30d']
    df['factor_raw'] = df.groupby('date')['relay'].transform(
        lambda x: x.rank(pct=True, na_option='keep', method='average'))

    # MAD + z-score
    def zscoref(grp):
        vals = grp['factor_raw'].values.astype(float)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad < 1e-12:
            grp['factor_value'] = 0.0
            return grp
        up, lo = med + 5.2*1.4826*mad, med - 5.2*1.4826*mad
        vc = np.clip(vals, lo, up)
        m, s = np.nanmean(vc), np.nanstd(vc)
        grp['factor_value'] = 0.0 if s < 1e-12 else (vc - m) / s
        return grp
    df = df.groupby('date', group_keys=False).apply(zscoref)
    df = df.dropna(subset=['factor_value'])

    # 市值代理
    df['log_amount'] = np.log(df['amount'].clip(lower=1) + 1)
    df['log_amount_ma20'] = df.groupby('stock_code')['log_amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    out = df[['date','stock_code','factor_value','log_amount_ma20']].dropna(
        subset=['factor_value','log_amount_ma20']).copy()

    residuals = []
    for dt, grp in out.groupby('date'):
        x = grp['log_amount_ma20'].values.astype(float)
        y = grp['factor_value'].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            grp['factor_neutral'] = 0.0
            residuals.append(grp[['date','stock_code','factor_neutral']]); continue
        X = np.column_stack([np.ones(m.sum()), x[m]])
        beta, _, _, _ = np.linalg.lstsq(X, y[m], rcond=None)
        r = np.full(len(x), np.nan); r[m] = y[m] - X @ beta
        r_med, r_std = np.nanmedian(r[m]), np.nanstd(r[m])
        grp['factor_neutral'] = 0.0 if r_std < 1e-12 else (
            np.where(m, (r - r_med) / r_std, np.nan)
        )
        residuals.append(grp[['date','stock_code','factor_neutral']])

    result = pd.concat(residuals, ignore_index=True).dropna(subset=['factor_neutral'])
    result = result.rename(columns={'factor_neutral':'factor_value'})
    result[['date','stock_code','factor_value']].sort_values(
        ['date','stock_code']).to_csv(output_file, index=False)
    print(f"Done: {len(result)} rows -> {output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    p.add_argument('--output', default='data/factor_momentum_relay_v1.csv')
    a = p.parse_args()
    compute_factor(a.kline, a.output)
