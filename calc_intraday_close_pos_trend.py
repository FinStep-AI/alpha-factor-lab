"""
因子：日内收盘位置趋势 (Intraday Close Position Trend, CPT_v1)
ID: intraday_close_pos_trend_v1

逻辑：
  close_position = (close - low) / (high - low)  (若 high=low 则取 0.5)
  该指标衡量每日收盘价在日内区间的位置：1=收在最高点, 0=收在最低点, 0.5=收在中间。

  趋势维度：
    CPT = (mean(close_position, short=5) - mean(close_position, long=20)) / std(close_position, 20)
  
  若近期收盘位置相对长期均值上升 → 说明近期日内买方力量在增强 → 后续收益更高。

Barra风格: MICRO/技术形态 (日内方向性)
计算：neutralize((roll5(close_position) - roll20(close_position)) / roll20_std, log_amount)

直方图: today_turn
"""

import pandas as pd
import numpy as np
import warnings, sys
from statsmodels import robust as sm_robust
warnings.filterwarnings('ignore')


def neutralize(values, control):
    """OLS中立化 + MAD缺失值填充 + z-score。"""
    combined = pd.DataFrame({'v': values, 'c': control})
    combined = combined.dropna()
    if len(combined) < 30:
        return np.full(len(values), np.nan)

    v = combined['v'].values
    x = combined['c'].values

    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        r = v - X @ beta
    except Exception:
        return np.full(len(values), np.nan)

    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad < 1e-10:
        return np.full(len(values), np.nan)
    r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
    std = r.std()
    if std < 1e-10:
        return np.full(len(values), np.nan)

    result = np.full(len(values), np.nan)
    idx = combined.index.values
    result[idx] = (r - np.median(r)) / std
    return result


def main():
    print("Loading data...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['date'] = pd.to_datetime(kline['date'].astype(str))

    # 字段清洗
    print("Cleaning data...")
    for col in ['close', 'high', 'low', 'turnover', 'amount']:
        kline[col] = pd.to_numeric(kline[col], errors='coerce')

    # 去除异常值
    kline = kline[(kline['high'] > 0) & (kline['low'] > 0) & (kline['close'] > 0)].copy()

    # 日内收盘位置 = (close - low) / (high - low)
    denom = kline['high'] - kline['low']
    denom = denom.replace(0, np.nan)
    kline['close_position'] = (kline['close'] - kline['low']) / denom
    kline['close_position'] = kline['close_position'].fillna(0.5)  # 零区间日→取中间值

    print("Computing CPT = (close_pos_5d - close_pos_20d) / close_pos_20d_std...")
    kline = kline.sort_values(['stock_code', 'date'])

    def cpt_series(group):
        """按个股计算 CPT 时序。"""
        cp = group['close_position'].copy()
        ma5 = cp.rolling(5, min_periods=4).mean()
        ma20 = cp.rolling(20, min_periods=15).mean()
        std20 = cp.rolling(20, min_periods=15).std()
        cpt = (ma5 - ma20) / std20.replace(0, np.nan)
        return pd.Series(cpt.values, index=group.index)

    kline['factor_raw'] = kline.groupby('stock_code', group_keys=False).apply(cpt_series).values

    # 限制极端值
    kline['factor_raw'] = kline['factor_raw'].clip(-10, 10)

    # 方向判断首次运行：若 close_position_trend 正资产低，则翻转
    # 这里直接翻转 (见回测报告 G5<G1 证明方向反)
    kline['factor_raw'] = -kline['factor_raw']
    kline['factor_raw'] = kline['factor_raw'].clip(-10, 10)

    print("Neutralizing by log_amount  (cross-sectional per date)...")
    all_out = []
    date_count = 0
    total_dates = kline['date'].nunique()

    for date, grp in kline.groupby('date'):
        date_count += 1
        if date_count % 200 == 0:
            print(f"  [{date_count}/{total_dates}] {date.date()} ...")

        grp = grp.dropna(subset=['factor_raw']).copy()
        if len(grp) < 50:
            continue

        log_amount = np.log(grp['amount'].clip(lower=1))
        neutralized = neutralize(grp['factor_raw'].values, log_amount.values)

        grp = grp.copy()
        grp['factor_value'] = neutralized
        grp = grp.dropna(subset=['factor_value'])
        all_out.append(grp[['stock_code', 'date', 'factor_value']])

    result = pd.concat(all_out, ignore_index=True)
    result = result.drop_duplicates(subset=['stock_code', 'date'])
    print(f"\nFinal factor shape: {result.shape}")
    print(f"  Non-null: {result['factor_value'].notna().sum()}")
    print(f"  Date range: {result['date'].min().date()} ~ {result['date'].max().date()}")
    print(f"  Stock count: {result['stock_code'].nunique()}")

    out_path = 'data/factor_intraday_close_pos_trend.csv'
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return result


if __name__ == '__main__':
    main()
