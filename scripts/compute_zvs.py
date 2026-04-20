#!/usr/bin/env python3
"""
因子: Z值成交量动量 (ZVS v1)
逻辑: vol_zscore × daily_ret，20日滚动均值

discriminative approach:
  当日量能程度(z-score) x 当日价格发现方向 = 成交异常日的
  收益累加值 = 大成交量推动价格向某个方向的累计效应

高累加值→连续放量上涨/连续放量下跌的持续性信号
Barra: Momentum/MICRO
"""

import numpy as np, pandas as pd, subprocess, json

def compute_zvs(fwd_days=5):
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code','date']).reset_index(drop=True)
    g = df.groupby('stock_code')
    
    df['ret'] = g['close'].pct_change()
    df['vol_z'] = g['volume'].transform(
        lambda x: (x - x.rolling(20,min_periods=10).mean()) / (x.rolling(20,min_periods=10).std()+1e-10)
    ).clip(-3, 3)
    
    # 量能加权当日收益
    df['zvs_daily'] = df['vol_z'] * df['ret']  # sign置入
    
    # 20日累积
    df['factor_raw'] = g['zvs_daily'].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    
    # 20日前瞻: shift(-fwd_days)
    # 但采用await的做法先计算因子值，再对factor_raw做shift(-fwd)    
    
    df['factor_shifted'] = g['factor_raw'].shift(-fwd_days)
    
    df['log_amount_20d'] = np.log(df['amount'].rolling(20,min_periods=10).mean()+1)
    
    rows = []
    for date, group in df.groupby('date'):
        gr = group.dropna(subset=['factor_shifted','log_amount_20d'])
        if len(gr) < 30: continue
        x,y = gr['log_amount_20d'].values, gr['factor_shifted'].values
        X = np.column_stack([np.ones(len(x)), x])
        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            res = y - X@b
        except: continue
        if len(res)==0 or np.std(res)<1e-10: continue
        med = float(np.median(res))
        mad = float(np.median(np.abs(np.array(res)-med)))*1.4826
        if mad<1e-10: continue
        res = np.clip(res, med-3*mad, med+3*mad)
        mu, sg = res.mean(), res.std()
        if sg<1e-10: continue
        gr = gr.copy(); gr['fn'] = (res-mu)/sg
        rows.append(gr[['date','stock_code','fn']])
    
    if not rows:
        print("ERROR: no data"); return
    
    fp = f'data/factor_zvs_v1_fwd{fwd_days}.csv'
    pd.concat(rows, ignore_index=True).to_csv(fp, index=False)
    print(f"[INFO] ZVS fwd={fwd_days}: {pd.read_csv(fp)['date'].min()}~{pd.read_csv(fp)['date'].max()}, "
          f"{pd.read_csv(fp)['stock_code'].nunique()} stocks")
    
    cost = 0.003 if fwd_days <= 5 else 0.002
    od = f'output/zvs_v1_fwd{fwd_days}'
    subprocess.run([
        'python3','skills/alpha-factor-lab/scripts/factor_backtest.py',
        '--factor', fp, '--returns', 'data/csi1000_returns.csv',
        '--n-groups','5','--rebalance-freq','20',
        '--forward-days', str(fwd_days), '--cost', str(cost),
        '--output-report', f'{od}/report.json',
        '--output-dir', od, '--factor-name', f'zvs_v1_fwd{fwd_days}'
    ], capture_output=True, text=True)
    with open(f'{od}/report.json') as f: r = json.load(f)
    m = r.get('metrics', {})
    print(f"ZVS fwd={fwd_days}: IC={m.get('ic_mean',0):.4f} t={m.get('ic_t_stat',0):.2f} "
          f"Sharpe={m.get('long_short_sharpe',0):.2f} Mono={m.get('monotonicity',0):.2f} "
          f"TO={m.get('turnover_mean',0):.2f}")

if __name__ == '__main__':
    compute_zvs(fwd_days=5)
    compute_zvs(fwd_days=20)
