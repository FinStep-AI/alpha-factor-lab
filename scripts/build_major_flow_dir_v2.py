"""
因子：主力资金净流入方向指示因子 v2 (major_flow_dir_v2)
========================================================
修复：v1 仅输出最新 1 日截面，无法时序回测。
v2  改用每周五截面，采集 2026-01-23 ~ 2026-05-15 共 17 期。

构造逻辑
--------
每周五截面：
  1. fintool → major_net_flow_in，已缓存 data/major_flow_cache.parquet
  2. 日频 net_flow / 20 日成交额均值  → flow_ratio
  3. 个股 60 日均值做 baseline  → intensity = ratio / roll_mean(ratio,60)
  4. 5 日平滑均值
  5. sign*log1p + 对数成交额中性化 + z-score

来源：i研报《大单与小单资金流的 alpha 能力》（IC 0.054，IR 3.96）
"""
import os, sys, argparse, warnings
import pandas as pd
import numpy as np
from numpy.linalg import lstsq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scripts.fintool_client as fintool

warnings.filterwarnings('ignore')

KLINE  = os.path.join('data', 'csi1000_kline_raw.csv')
OUT    = os.path.join('data',  'major_flow_dir_v2.csv')
CACHE  = os.path.join('data',  'major_flow_cache.parquet')

FINTOL_FIRST = pd.Timestamp('2026-01-19')   # fintool 净资金流起始日期


# ── helpers ──────────────────────────────────────────────────────────────────

def norm_code(x) -> str: return str(int(x)).strip().zfill(6)

def load_kline():
    df = pd.read_csv(KLINE, parse_dates=['date'])
    df['stock_code'] = df['stock_code'].apply(norm_code)
    df['amount']     = pd.to_numeric(df['amount'], errors='coerce')
    return df

def load_flow(codes, start, end):
    """从缓存读，缺的从 fintool 补。"""
    if os.path.exists(CACHE):
        cached = pd.read_parquet(CACHE)
        have = set(cached.stock_code.unique()) if 'stock_code' in cached.columns else set()
        need = [c for c in codes if c not in have]
    else:
        cached, need = pd.DataFrame(), list(codes)

    if need:
        print(f'  ⏬ fintool fetch: {len(need)} codes …')
        ok = err = skip = 0
        parts = []
        for i, code in enumerate(need):
            try:
                rows = fintool.get_net_flow(code, start_date=start, end_date=end)
                if rows:
                    sub = pd.DataFrame(rows)
                    sub['stock_code']          = code
                    sub['date']                = pd.to_datetime(sub['date'])
                    sub['major_net_flow_in']   = pd.to_numeric(sub['major_net_flow_in'], errors='coerce')
                    parts.append(sub)
                    ok += 1
                else:
                    skip += 1
            except Exception:
                err += 1
            if (i + 1) % 200 == 0:
                print(f'    [{i+1}/{len(need)}] ok={ok} err={err} skip={skip}')
        print(f'  ✅ ok={ok} err={err} skip={skip}')
        if parts:
            new_df = pd.concat(parts, ignore_index=True)
            cached  = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df

    cached.to_parquet(CACHE, index=False)
    print(f'  💾 cache: {len(cached):,} rows / {cached.stock_code.nunique()} codes')
    return cached


# ── core ─────────────────────────────────────────────────────────────────────

def compute_friday_cross_sections(flow, kline, smooth=5):
    """
    返回 DataFrame[date, stock_code, factor_value]，
    date 限定在 2026-01-23 ~ 2026-05-15 的每周五，且与 kline 收益有交叠。
    """
    kline = kline.copy()
    kline['date'] = pd.to_datetime(kline['date'])

    # --- 20 日成交额均值（市值代理） ---
    k20 = (kline.groupby('stock_code', as_index=False, group_keys=False)
              .apply(lambda g: g.assign(amount_20d=
                  g['amount'].rolling(20, min_periods=6).mean())))

    # --- 资金流日频 ratio ---
    merged = (flow.merge(k20[['date','stock_code','amount_20d']],
                         on=['date','stock_code'], how='inner')
                .dropna(subset=['major_net_flow_in','amount_20d'])
                .loc[lambda df: df['amount_20d'] > 1e3])
    merged['ratio'] = merged['major_net_flow_in'] / merged['amount_20d']

    # --- intensity = ratio / 60 日个股均线 ---
    merged = merged.sort_values(['stock_code','date'])
    merged['ratio_base'] = (
        merged.groupby('stock_code')['ratio']
              .transform(lambda x: x.shift(1).rolling(60, min_periods=12).mean()))
    merged['intensity']  = (merged['ratio'] / merged['ratio_base'].replace(0, np.nan)).clip(-50, 50)

    # --- 5 日平滑 ---
    merged['raw'] = (
        merged.groupby('stock_code')['intensity']
              .transform(lambda x: x.rolling(smooth, min_periods=2).mean()))

    # --- 收盘价可用性 mask ---
    price_last = (kline.groupby(['stock_code','date'], as_index=False)['close']
                      .last()
                      .rename(columns={'date':'price_date'}))
    price_last['price_date'] = pd.to_datetime(price_last['price_date'])

    merged = merged.merge(price_last, on='stock_code', how='left')
    valid_dates = set(pd.to_datetime(kline.date.unique()))

    # --- 每周五截面 ---
    all_biz   = pd.bdate_range('2026-01-01', '2026-06-01')
    fridays   = sorted(all_biz[(all_biz.weekday==4) & (all_biz>=FINTOL_FIRST)])
    print(f'  📅 eligible Fridays: {len(fridays)}  {fridays[0].date()} ~ {fridays[-1].date()}')

    # market-cap proxy (cross-sectional last)
    mktcap_last = (k20.sort_values('date')
                       .groupby('stock_code', as_index=False)['amount_20d'].last()
                       .rename(columns={'amount_20d':'log_mktcap'}))
    mktcap_last['log_mktcap'] = np.log(mktcap_last['log_mktcap'].clip(lower=1))

    rows = []
    for fri_d in fridays:
        fri_d_ts = pd.Timestamp(fri_d)
        # code must have flow data by fri and have kline close on fri or earlier
        sub = (merged[merged['date'] <= fri_d_ts]
                    .groupby('stock_code', as_index=False)['raw'].last()
                    .dropna(subset=['raw']))
        # keep only codes with tradable close onfri (or within next biz day if fri holiday)
        avail = sorted(kline[kline['date']<=fri_d_ts].stock_code.unique())
        sub   = sub[sub['stock_code'].isin(avail)]
        sub   = sub.merge(mktcap_last, on='stock_code', how='left').dropna(subset=['log_mktcap'])
        if len(sub) < 200:
            print(f'    ─ fri {fri_d.date()}: only {len(sub)} codes, skip')
            continue

        # neutralize by log_mktcap
        X = np.column_stack([np.ones(len(sub)), sub['log_mktcap'].values])
        y = sub['raw'].values
        coef, *_ = lstsq(X, y, rcond=None)
        resid = y - X @ coef
        mu, sd = resid.mean(), resid.std()
        sub = sub.assign(factor_value=(resid - mu) / (sd if sd > 1e-9 else 1.0))
        sub['date'] = fri_d_ts.strftime('%Y-%m-%d')
        rows.append(sub[['date','stock_code','factor_value']])

    if not rows:
        print('  ERROR: no valid sections'); sys.exit(1)
    return pd.concat(rows, ignore_index=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor-name',    default='major_flow_dir_v2')
    ap.add_argument('--lookback-days',  type=int, default=400)
    ap.add_argument('--smooth-window',  type=int, default=5)
    args = ap.parse_args()

    print(f'\n=== {args.factor_name} ===')
    kline    = load_kline()
    codes    = sorted(kline.stock_code.unique())
    end      = kline.date.max().strftime('%Y-%m-%d')
    start    = (kline.date.max() - pd.Timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')

    flow     = load_flow(codes, start, end)
    result   = compute_friday_cross_sections(flow, kline, smooth=args.smooth_window)

    result.to_csv(OUT, index=False)
    print(f'\n  ✅  {len(result):,} rows  dates={result.date.nunique()}  → {OUT}')
    print(result.groupby('date').size().describe())


if __name__ == '__main__':
    main()
