"""
因子：日均振幅顶部突破强度 (Daily Range Top Touch Strength)
ID: range_top_touch_v1   Barra: Momentum

逻辑（全新方向）：
  close 在 high 附近 → 多头当日占优 → 未来短期继续上涨概率高
  定义 α = (high-close)/amplitude，α越小=收盘越接近日内最高=多头越强
  取 20d roll α 均值的反向，rank 后做截面 rank → factor

前导：中证1000资金二次驱动，突破信号预测力优于大盘
市值中性化：log_amount_ma20 OLS 中性化
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

N_DAYS = 20


def compute_factor(kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_range_top_hit_v1.csv'):
    df = pd.read_csv(kline_file)
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 1) 日度 α: 收盘离日内最高价的距离/振幅 (0=收盘=high, 1=收盘=low)
    amp_safe = df['amplitude'].clip(lower=0.1)  # 振幅≥0.1%
    df['pos_ratio'] = (df['high'] - df['close']) / amp_safe  # 0= leverage
    df['pos_ratio'] = df['pos_ratio'].clip(0, 1)

    # 2) 20d 均值 α → 多头越强则均值 α 越低，故作反向
    df['alpha_ma'] = (
        df.groupby('stock_code')['pos_ratio']
          .transform(lambda x: x.rolling(N_DAYS, min_periods=10).mean())
    )
    # 反向：均值 α 越低 → 因子值越高
    df['momentum_signal'] = 1.0 - df['alpha_ma']

    # 3) 每日截面 rank
    df['factor_value'] = (
        df.groupby('date')['momentum_signal']
          .transform(lambda x: x.rank(pct=True, na_option='keep'))
    )

    # 4) 截面上 MAD + z-score 去极
    def zscore(grp):
        vals = grp['factor_value'].values.astype(float)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad < 1e-12:
            grp['factor_value'] = 0.0
            return grp
        up, lo = med + 5.2 * 1.4826 * mad, med - 5.2 * 1.4826 * mad
        vc = np.clip(vals, lo, up)
        m, s = np.nanmean(vc), np.nanstd(vc)
        grp['factor_value'] = 0.0 if s < 1e-12 else (vc - m) / s
        return grp

    df = df.groupby('date', group_keys=False).apply(zscore)

    # 5) 市值代理: log(amount_ma20) → OLS 中性化
    df['log_amount'] = np.log(df['amount'].clip(lower=1) + 1)
    df['log_amount_ma20'] = (
        df.groupby('stock_code')['log_amount']
          .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )

    out = df[['date', 'stock_code', 'factor_value', 'log_amount_ma20']].dropna(
        subset=['factor_value', 'log_amount_ma20']
    ).copy()

    residuals = []
    for dt, grp in out.groupby('date'):
        x = grp['log_amount_ma20'].values.astype(float)
        y = grp['factor_value'].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            grp['factor_neutral'] = 0.0
            residuals.append(grp[['date', 'stock_code', 'factor_neutral']])
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
        resid = np.full(len(x), np.nan)
        resid[mask] = y[mask] - X @ beta
        r_med, r_std = np.nanmedian(resid[mask]), np.nanstd(resid[mask])
        grp['factor_neutral'] = 0.0 if r_std < 1e-12 else (
            np.where(mask, (resid - r_med) / r_std, np.nan)
        )
        residuals.append(grp[['date', 'stock_code', 'factor_neutral']])

    result = pd.concat(residuals, ignore_index=True).dropna(subset=['factor_neutral'])
    result = result.rename(columns={'factor_neutral': 'factor_value'})
    result[['date', 'stock_code', 'factor_value']].sort_values(
        ['date', 'stock_code']
    ).to_csv(output_file, index=False)
    print(f"Done: {len(result)} rows → {output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    p.add_argument('--output', default='data/factor_range_top_hit_v1.csv')
    a = p.parse_args()
    compute_factor(a.kline, a.output)
