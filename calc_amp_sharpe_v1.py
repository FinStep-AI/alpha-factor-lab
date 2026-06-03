"""
calc_amp_sharpe_v1
==================
因子名: amp_sharpe_v1
来源: 2026-05 知乎研报方向《基于类夏普比动量约束与低波反转的量化选股》
       核心思路: 振幅具有"小幅稳定后反转/惯性"的类Sharpe特性，
       用 10日振幅的 均值/标准差（收益/波动Sharpe比）刻画一段窗口内的"价格区间稳定度+廉价度"，
       而不是单看振幅均值（后者在中证1000已被振幅水平 v2覆盖）。

构造:
   amp_daily = (high - low) / close * 100          # 日振幅（%）
   amp_mean_10d   = mean(amp_daily[-10:])
   amp_std_10d    = std(amp_daily[-10:], ddof=1)
   amp_sharpe_raw = amp_mean_10d / amp_std_10d      # 类似Sharpe比：报酬/波动
   → 样本内高 amp_sharpe = 振幅大且稳定（低CV），可能对应前期放宽动量带

市值中性化: OLS 残差（默认: --neutralize market_cap）

输出: data/amp_sharpe_v1.csv  [date, stock_code, factor_value]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(__file__)          # alpha-factor-lab/
DATA_DIR  = os.path.join(BASE_DIR, 'data')


def load_kline() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'csi1000_kline_raw.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    return df


def compute_raw(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df['amp'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan) * 100

    g = df.groupby('stock_code', sort=False)
    df['amp_mean'] = g['amp'].transform(lambda x: x.rolling(window, min_periods=5).mean())
    df['amp_std'] = g['amp'].transform(lambda x: x.rolling(window, min_periods=3).std())

    df['factor_raw'] = df['amp_mean'] / df['amp_std'].replace(0, np.nan)
    # 极端值 winsorize by group
    lower = df['factor_raw'].quantile(0.005)
    upper = df['factor_raw'].quantile(0.995)
    df['factor_raw'] = df['factor_raw'].clip(lower, upper)

    df = df.dropna(subset=['factor_raw', 'amp_mean', 'amp_std'])
    return df[['date', 'stock_code', 'factor_raw']].rename(columns={'factor_raw': 'factor_value'})


def neutralize_market_cap(df_factor: pd.DataFrame, df_kline: pd.DataFrame) -> pd.DataFrame:
    """OLS neutralize by market_cap proxy: close * volume as size proxy."""
    merged = df_factor.merge(
        df_kline[['date', 'stock_code', 'close', 'volume']], on=['date', 'stock_code'], how='left')
    merged['size_proxy'] = merged['close'] * merged['volume'].replace(0, np.nan)

    def neutralize_group(g):
        y = g['factor_value'].values
        x = g['size_proxy'].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            g['factor_value_neutral'] = np.nan
            return g
        yy, xx = y[mask], x[mask]
        # log-scale x
        xx = np.log1p(xx)
        X = np.column_stack([np.ones(len(xx)), xx])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
            resid = np.full(len(y), np.nan)
            resid[mask] = yy - X @ beta
        except Exception:
            resid = np.full(len(y), np.nan)
        g['factor_value_neutral'] = resid
        return g

    result = merged.groupby('date', group_keys=False).apply(neutralize_group)
    result = result.dropna(subset=['factor_value_neutral'])
    return result[['date', 'stock_code', 'factor_value_neutral']].rename(
        columns={'factor_value_neutral': 'factor_value'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--neutralize', action='store_true', default=True,
                    help='Market-cap neutralize (OLS residual)')
    ap.add_argument('--no-neutralize', dest='neutralize', action='store_false')
    ap.add_argument('--output', default=os.path.join(DATA_DIR, 'amp_sharpe_v1.csv'))
    args = ap.parse_args()

    print(f'Loading kline...')
    df = load_kline()
    print(f'  {len(df):,} rows, {df["stock_code"].nunique()} stocks, {df["date"].nunique()} dates')

    print(f'Computing amp_sharpe_v1 (window={args.window})...')
    raw = compute_raw(df, window=args.window)
    print(f'  raw factor rows: {len(raw):,}')

    if args.neutralize:
        print('Neutralizing by market cap (OLS residual)...')
        out = neutralize_market_cap(raw, df)
        print(f'  neutralized rows: {len(out):,}')
    else:
        out = raw

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False, float_format='%.6f')
    print(f'Saved → {args.output}')

    # summary
    g = out.groupby('date')['factor_value'].agg(['mean','std','count']).tail(5)
    print(f'\nLast 5 dates summary:\n{g.to_string()}')


if __name__ == '__main__':
    main()
