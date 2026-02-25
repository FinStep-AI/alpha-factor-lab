#!/usr/bin/env python3
"""构建因子1(预期修正加速度) + 因子2(情绪价量复合) 并回测"""
import pandas as pd
import numpy as np
import json, os, sys, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUTPUT = os.path.join(BASE, 'output')
SCRIPTS = os.path.join(BASE, 'scripts')

print("=" * 60)
print("  加载K线数据...")
print("=" * 60)
kline = pd.read_csv(os.path.join(DATA, 'csi300_kline_raw.csv'))
kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code', 'date'])
print(f"K线: {kline['stock_code'].nunique()} 只, {len(kline)} 行")

# 日收益率
kline['returns'] = kline.groupby('stock_code')['close'].pct_change()

# 等权市场日收益
market_ret = kline.groupby('date')['returns'].mean().rename('market_return')
kline = kline.merge(market_ret, on='date', how='left')
kline['excess_return'] = kline['returns'] - kline['market_return']

# ============================================================
# 因子1: 分析师预期修正加速度 (超额收益加速度代理)
# ============================================================
print("\n" + "=" * 60)
print("  因子1: 预期修正加速度")
print("=" * 60)

f1_results = []
for code, g in kline.groupby('stock_code'):
    g = g.set_index('date').sort_index()
    if len(g) < 50:
        continue
    # 20日累计超额收益
    cum_ex_20 = g['excess_return'].rolling(20, min_periods=10).sum()
    # 前20日累计超额
    cum_ex_prev20 = cum_ex_20.shift(20)
    # 加速度
    accel = cum_ex_20 - cum_ex_prev20
    # 标准化
    ex_vol = g['excess_return'].rolling(40, min_periods=20).std()
    factor = accel / (ex_vol * np.sqrt(20)).replace(0, np.nan)
    
    temp = pd.DataFrame({'date': g.index, 'stock_code': code, 'factor': factor.values})
    f1_results.append(temp)

f1_df = pd.concat(f1_results, ignore_index=True).dropna(subset=['factor'])

# 截面winsorize+zscore
def cs_process(group):
    v = group['factor'].copy()
    med = v.median()
    mad = (v - med).abs().median()
    if mad > 0:
        sm = 1.4826 * mad
        v = v.clip(med - 3*sm, med + 3*sm)
    s = v.std()
    if s > 0:
        v = (v - v.mean()) / s
    group['factor'] = v
    return group

f1_df = f1_df.groupby('date', group_keys=False).apply(cs_process)
print(f"因子1: {f1_df['factor'].notna().sum()} 有效值, {f1_df['stock_code'].nunique()} 只")
print(f"日期: {f1_df['date'].min()} ~ {f1_df['date'].max()}")

# 保存
f1_path = os.path.join(DATA, 'analyst_revision_accel_factor.csv')
f1_out = f1_df.copy()
f1_out['date'] = f1_out['date'].dt.strftime('%Y-%m-%d') if hasattr(f1_out['date'].iloc[0], 'strftime') else f1_out['date']
f1_out.to_csv(f1_path, index=False)

# ============================================================
# 因子2: 情绪价量复合因子
# ============================================================
print("\n" + "=" * 60)
print("  因子2: 情绪价量复合因子")
print("=" * 60)

f2_results = []
for code, g in kline.groupby('stock_code'):
    g = g.set_index('date').sort_index()
    if len(g) < 30:
        continue
    
    # 换手率z-score
    t_ma5 = g['turnover'].rolling(5, min_periods=3).mean()
    t_ma20 = g['turnover'].rolling(20, min_periods=10).mean()
    t_std20 = g['turnover'].rolling(20, min_periods=10).std()
    t_zscore = (t_ma5 - t_ma20) / t_std20.replace(0, np.nan)
    
    # 量价相关性
    pv_corr = g['volume'].rolling(5, min_periods=3).corr(g['close'])
    
    # 10日平均振幅
    amp_10d = g['amplitude'].rolling(10, min_periods=5).mean()
    
    # 复合 = t_zscore × |pv_corr| × amp_10d
    factor = t_zscore * pv_corr.abs() * amp_10d
    
    temp = pd.DataFrame({'date': g.index, 'stock_code': code, 'factor': factor.values})
    f2_results.append(temp)

f2_df = pd.concat(f2_results, ignore_index=True).dropna(subset=['factor'])
f2_df = f2_df.groupby('date', group_keys=False).apply(cs_process)
print(f"因子2: {f2_df['factor'].notna().sum()} 有效值, {f2_df['stock_code'].nunique()} 只")
print(f"日期: {f2_df['date'].min()} ~ {f2_df['date'].max()}")

f2_path = os.path.join(DATA, 'sentiment_pv_composite_factor.csv')
f2_out = f2_df.copy()
f2_out['date'] = f2_out['date'].dt.strftime('%Y-%m-%d') if hasattr(f2_out['date'].iloc[0], 'strftime') else f2_out['date']
f2_out.to_csv(f2_path, index=False)

# ============================================================
# 准备收益率文件
# ============================================================
returns_path = os.path.join(DATA, 'csi300_returns.csv')
ret_df = kline[['date', 'stock_code', 'returns']].dropna()
ret_df['date'] = ret_df['date'].dt.strftime('%Y-%m-%d')
ret_df.to_csv(returns_path, index=False)
print(f"\n收益率: {returns_path} ({len(ret_df)} 行)")

# ============================================================
# 回测
# ============================================================
factors = [
    ('analyst_revision_accel', f1_path, 5, 5),   # 周频调仓
    ('sentiment_pv_composite', f2_path, 5, 5),    # 周频调仓
]

for fid, fpath, rebal, fwd in factors:
    out_dir = os.path.join(OUTPUT, fid)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'backtest_report.json')
    
    cmd = [
        sys.executable, os.path.join(SCRIPTS, 'factor_backtest.py'),
        '--factor', fpath,
        '--returns', returns_path,
        '--n-groups', '5',
        '--rebalance-freq', str(rebal),
        '--forward-days', str(fwd),
        '--cost', '0.002',
        '--output-report', report_path,
        '--output-dir', out_dir,
        '--factor-name', fid,
    ]
    
    print(f"\n{'='*60}")
    print(f"  回测: {fid} (调仓{rebal}d, 前瞻{fwd}d)")
    print(f"{'='*60}")
    
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.returncode != 0:
        print(f"[ERROR] {r.stderr[:500]}")

print("\n\n✅ 因子1+因子2 构建与回测完成!")
