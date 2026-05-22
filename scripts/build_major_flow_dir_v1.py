"""
因子：主力资金净流入方向指示因子 (major_flow_dir_v1)

构造逻辑（来源于 A 股高频资金流研究 + 中证1000样本构造）：
  1. 对每只股票，从 fintool 拉取主力资金净流入序列（major_net_flow_in），
     单位为元，用日线 amount（成交额，元）做分母。
  2. 计算截面日频净流入比率：flow_ratio = major_net_flow_in / amount
     （加入前/后 20 日均值，当前值 / 均值，避免绝对量偏置）。
  3. 5 日均值 → raw_factor = mean(flow_ratio, 5)
  4. 做 log(1+|x|) * sign(x) 变换，再对数市值中性化，最后 z-score。

预期方向：正向（主力持续净流入多的股票，后续收益更高）
Barra 风格：微观结构

来源逻辑参考：
  - i研报：大单与小单资金流的 alpha 能力（major_net_flow IC 0.054, IR 3.96）
  - 知乎：集成订单流不平衡因子研究（Cont, 2023 QF）
"""
import os, sys, json, argparse, warnings
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scripts.fintool_client as fintool

warnings.filterwarnings('ignore')

KLINE  = os.path.join('data', 'csi1000_kline_raw.csv')
OUT    = os.path.join('data',  'major_flow_dir_v1.csv')
CACHE  = os.path.join('data',  'major_flow_cache.parquet')

# ── helpers ──────────────────────────────────────────────────────────────────

def normalize_code(raw) -> str:
    s = str(int(raw)).strip()
    return s.zfill(6)


def load_kline():
    df = pd.read_csv(KLINE, parse_dates=['date'])
    df['stock_code'] = df['stock_code'].apply(normalize_code)
    df['amount']     = pd.to_numeric(df['amount'], errors='coerce')
    df['volume']     = pd.to_numeric(df['volume'], errors='coerce')
    return df


def fetch_flow(codes, start, end):
    """拉取资金流；优先读缓存，缺失部分补拉。"""
    cache = {}
    if os.path.exists(CACHE):
        try:
            cached = pd.read_parquet(CACHE)
            cache = {c: g for c, g in cached.groupby('stock_code')}
            print(f"  [cache] loaded {len(cache)} codes")
        except Exception:
            pass

    need = [c for c in codes if c not in cache]
    print(f"  fintool need fetch: {len(need)} codes")
    ok = err = skip = 0
    for i, code in enumerate(need):
        try:
            rows = fintool.get_net_flow(code, start_date=start, end_date=end)
            if not rows:
                skip += 1
                continue
            sub = pd.DataFrame(rows)
            sub['stock_code'] = code
            sub['date'] = pd.to_datetime(sub['date'])
            sub['major_net_flow_in'] = pd.to_numeric(sub['major_net_flow_in'], errors='coerce')
            cache[code] = sub
            ok += 1
        except Exception as e:
            err += 1
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(need)}] ok={ok} err={err} skip={skip}")
    print(f"  fintool done: ok={ok} err={err} skip={skip}")

    # persist cache
    all_df = pd.concat(cache.values(), ignore_index=True)
    all_df.to_parquet(CACHE, index=False)
    print(f"  [cache] saved {len(cache)} codes → {CACHE}")
    return cache


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor-name', default='major_flow_dir_v1')
    ap.add_argument('--lookback-days', type=int, default=120,
                    help='fintool 拉取天数窗口（默认 120）')
    ap.add_argument('--smooth-window', type=int, default=5,
                    help='主力净流入比率均线窗口（默认 5）')
    ap.add_argument('--neutralize', default='log_market_cap',
                    help='中性化变量，逗号分隔')
    args = ap.parse_args()

    print(f'\n=== {args.factor_name} ===')
    df = load_kline()
    codes = sorted(df['stock_code'].unique())
    print(f'kline: {len(codes)} codes, {df["date"].min().date()} ~ {df["date"].max().date()}')

    end   = df['date'].max().strftime('%Y-%m-%d')
    start = (df['date'].max() - pd.Timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')
    flow_cache = fetch_flow(codes, start, end)

    # Build flow frame
    flow_parts = []
    for c, g in flow_cache.items():
        tmp = g[['date', 'stock_code', 'major_net_flow_in']].copy()
        flow_parts.append(tmp)
    if not flow_parts:
        print('ERROR: no flow data'); sys.exit(1)
    flow = pd.concat(flow_parts, ignore_index=True)
    print(f'flow rows: {len(flow)}  dates: {sorted(flow.date.unique())[:3]} ... {sorted(flow.date.unique())[-3:]}')

    # merge volume proxy (20d avg amount per stock)
    k20 = (df.groupby('stock_code', as_index=False)
             .apply(lambda g: g.assign(amount_20d=g['amount'].rolling(20, min_periods=5).mean()))
             .reset_index(drop=True))
    k20['date'] = pd.to_datetime(k20['date'])
    merged = flow.merge(k20[['date','stock_code','amount_20d']], on=['date','stock_code'], how='inner')
    merged = merged.dropna(subset=['major_net_flow_in','amount_20d'])
    merged = merged[merged['amount_20d'] > 0]

    # ratio = ratio_t / avg_ratio_20d  (cross-sectional flow intensity)
    merged['ratio'] = merged['major_net_flow_in'] / merged['amount_20d']
    ratio_mean = merged.groupby('stock_code')['ratio'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    merged['ratio_rel'] = merged['ratio'] / ratio_mean.replace(0, np.nan)
    merged['ratio_rel'] = merged['ratio_rel'].clip(-50, 50)  # cap outliers

    # smooth
    merged = merged.sort_values(['stock_code','date'])
    merged['raw_factor'] = (merged.groupby('stock_code')['ratio_rel']
                              .transform(lambda x: x.rolling(args.smooth_window, min_periods=3).mean()))

    # last available date per stock
    out = (merged.dropna(subset=['raw_factor'])
                  .groupby('stock_code', as_index=False)
                  .last()[['date','stock_code','raw_factor']]
                  .rename(columns={'date':'asof_date'}))
    out['asof_date'] = out['asof_date'].dt.strftime('%Y-%m-%d')
    print(f'raw factor rows (last avail): {len(out)}')

    # log-transform
    out['factor_raw'] = np.sign(out['raw_factor']) * np.log1p(out['raw_factor'].abs())

    # neutralize by chosen variable
    for nvar in args.neutralize.split(','):
        nvar = nvar.strip()
        if nvar == 'log_market_cap':
            # use amount_20d as proxy for market cap (highly correlated for CSI1000)
            ctor = k20[['date','stock_code','amount_20d']].copy()
            ctor['log_mktcap'] = np.log(ctor['amount_20d'].clip(lower=1))
            ctor = (ctor.groupby('stock_code', as_index=False)
                         .last()[['stock_code','log_mktcap']])
            out = out.merge(ctor, on='stock_code', how='left')
            nvar = 'log_mktcap'
        from numpy.linalg import lstsq
        mask = out[nvar].notna() & np.isfinite(out[nvar]) & np.isfinite(out['factor_raw'])
        X = np.column_stack([np.ones(mask.sum()), out.loc[mask, nvar].values])
        y = out.loc[mask, 'factor_raw'].values
        coef, *_ = lstsq(X, y, rcond=None)
        out.loc[mask, 'factor_raw'] = y - X @ coef

    # z-score
    mu = out['factor_raw'].mean()
    sd = out['factor_raw'].std()
    out['factor_value'] = (out['factor_raw'] - mu) / (sd if sd > 1e-12 else 1.0)

    out_out = out[['asof_date','stock_code','factor_value']].copy()
    out_out.columns = ['date', 'stock_code', 'factor_value']
    out_out.to_csv(OUT, index=False)
    print(f'factor saved → {OUT}  ({len(out_out)} rows)')
    print(out_out.describe())


if __name__ == '__main__':
    main()
