#!/usr/bin/env python3
"""
因子: 均线趋势加速度 7/21 v1 (Moving Average Trend Acceleration 7/21)
factor_id: mom_accel_7_21_v1

逻辑:
  定义 close/MA_k - close/MA_K 为其"价格偏离修正积分"（近期弯矩新兴暗示 这是一种阻抗趋势加速）。
  用 k=7（近方向短柱方向的准轴方向），K=21（实盘附近基线轴）组合出比 趋势差动量比例。
  平移括号到这里，k 不选5，是目的防 短期离奇徘徊 滤掉新保守的7日均线，以避免同/ma_bias重复确认，不与 close/ma 系列高度相关（与 ma_bias 同切入点但量级形态统计相关小）。

  虽然接近 ma_disp（也是均线距差算散度/差价），但 ma_disp 是对 均线之间之间的偏差作均线离散度，这里是对 close 而非均线间差，并且是短期均线一端的偏离——几何流向上有别，但做 rcue  相关性粗查低（选7这短段独立方向）。

  这里选用 close_price/k 而不是 ma_k 两者 高阶维差 作为粘贴；测试假设中5日近短窗口的迭代周期须为 7或短数列加速快捷。

最后公式: factor = close/MA7 - close/MA21，成交额OLS中性化。
"""
import numpy as np, pandas as pd
from pathlib import Path

def main():
    D = Path("data")
    print("loading...")
    df = pd.read_csv(D/"csi1000_kline_raw.csv",  # dtype省开销
                     usecols=['date','stock_code','close','amount'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code','date']).reset_index(drop=True)
    n_stk = df['stock_code'].nunique()
    print(f"stocks={n_stk}, {df['date'].min().date()} ~ {df['date'].max().date()}")

    print("rolling MA7 / MA21...")
    df['ma7']  = df.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(7,  min_periods=5).mean())
    df['ma21'] = df.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(21, min_periods=15).mean())

    print("raw factor = close/ma7 - close/ma21...")
    raw = df['close']/df['ma7'] - df['close']/df['ma21']      # price deviation term
    df['raw_factor'] = raw
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))

    res = df[['date','stock_code','raw_factor','log_amount']].dropna()
    print(f"non-null: {len(res):,}")

    # --- 5% MAD winsorize per cross-section ---
    parts = []
    for dt, g in res.groupby('date', sort=False):
        v = g['raw_factor'].values
        lo, hi = np.nanquantile(v, .025), np.nanquantile(v, .975)
        g2 = g.copy(); g2['raw_factor'] = np.clip(v, lo, hi)
        parts.append(g2)
    res = pd.concat(parts, ignore_index=True)

    # --- cross-section z-score ---
    res['z'] = res.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))

    # --- amount OLS neutralization ---
    parts2 = []
    for dt, g in res.groupby('date', sort=False):
        g = g.dropna(subset=['z','log_amount'])
        if len(g) < 20:
            g['n'] = np.nan; parts2.append(g[['n']]); continue
        x = g['log_amount'].values; y = g['z'].values
        xm=np.nanmean(x); ym=np.nanmean(y)
        b = np.nansum((x-xm)*(y-ym)) / (np.nansum((x-xm)**2)+1e-10)
        a = ym - b*xm
        g = g.copy(); g['n'] = y - (a+b*x); parts2.append(g[['n']])
    neu = pd.concat(parts2, ignore_index=False)
    res['z2'] = neu['n']
    res = res.dropna(subset=['z2'])

    # --- final cross-section z-score + clip ---
    res['factor_value'] = res.groupby('date')['z2'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12)).clip(-3, 3)

    out = res[['date','stock_code','factor_value']].dropna()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out_path = D / "factor_mom_accel_7_21_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}  ({len(out):,} rows, {out['date'].min()}~{out['date'].max()})")
    print(out['factor_value'].describe())
    print(out.groupby('date')['factor_value'].std().describe())

if __name__ == '__main__':
    main()
