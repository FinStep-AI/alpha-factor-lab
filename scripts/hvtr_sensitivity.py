#!/usr/bin/env python3
"""HVTR灵敏度测试: 不同spike阈值 + 20d前瞻"""
import numpy as np, pandas as pd, subprocess, sys, json

base = pd.read_csv('data/factor_hvtr_v1.csv')
returns = pd.read_csv('data/csi1000_returns.csv')

def run_backtest(factor_csv, rebal, fwd, cost, out_dir, name):
    subprocess.run([
        'python3','skills/alpha-factor-lab/scripts/factor_backtest.py',
        '--factor', factor_csv, '--returns', 'data/csi1000_returns.csv',
        '--n-groups','5', '--rebalance-freq', str(rebal),
        '--forward-days', str(fwd), '--cost', str(cost),
        '--output-report', f'{out_dir}/report.json',
        '--output-dir', f'{out_dir}/', '--factor-name', name
    ], capture_output=True, text=True)
    with open(f'{out_dir}/report.json') as f:
        r = json.load(f)
    return {
        'ic': r.get('ic_mean'), 't': r.get('ic_t_stat'),
        'sharpe': r.get('long_short_sharpe'),
        'mono': r.get('monotonicity'), 'turnover': r.get('turnover_mean'),
        'g5_sharpe': r.get('group_sharpe', [None]*5)[4] if r.get('group_sharpe') else None
    }

# 测试不同spike阈值
for k in [0.5, 0.75, 1.0, 1.25, 1.5]:
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code','date']).reset_index(drop=True)
    g = df.groupby('stock_code')
    df['fwd_ret_20d'] = g['close'].transform(lambda x: x.shift(-20)/x - 1)
    df['vol_ma_20'] = g['volume'].transform(lambda x: x.rolling(20,min_periods=10).mean())
    df['vol_std_20'] = g['volume'].transform(lambda x: x.rolling(20,min_periods=10).std().fillna(0))
    df['is_spike'] = (df['volume'] > df['vol_ma_20'] + k*df['vol_std_20']).astype(int)
    
    def calc(group):
        sf = group['fwd_ret_20d'].where(group['is_spike']==1)
        nf = group['fwd_ret_20d'].where(group['is_spike']==0)
        return sf.rolling(20,min_periods=8).mean() - nf.rolling(20,min_periods=8).mean()
    
    df['fv'] = g.apply(calc).reset_index(level=0, drop=True)
    df['la20'] = np.log(df['amount'].rolling(20,min_periods=10).mean()+1)
    
    rows = []
    for date, gr in df.groupby('date'):
        gr = gr.dropna(subset=['fv','la20'])
        if len(gr)<30: continue
        x,y = gr['la20'].values, gr['fv'].values
        X = np.column_stack([np.ones(len(x)), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = y - X@b
        if len(r2) == 0: continue
        med = np.median(r2)
        mad = np.median(np.abs(r2-med))*1.4826
        if mad<1e-10: continue
        r2 = np.clip(r2, med-3*mad, med+3*mad)
        mu,sg = r2.mean(), r2.std()
        if sg<1e-10: continue
        z = (r2-mu)/sg
        gr=gr.copy(); gr['fn']=z
        rows.append(gr[['date','stock_code','fn']])
    
    if not rows: continue
    out = pd.concat(rows, ignore_index=True)
    fp = f'data/hvtr_k{k:.2f}.csv'
    out.to_csv(fp, index=False)
    
    r = run_backtest(fp, 20, 20, 0.002, f'output/hvtr_k{k:.2f}_20d', f'hvtr_k{k:.2f}')
    spike_rate = (df['is_spike']==1).sum()/len(df)
    print(f"k={k:.2f}: IC={r['ic']:.4f if r['ic'] else 'NaN'} t={r['t']:.2f if r['t'] else 'NaN'} "
          f"Shar={r['sharpe']:.2f if r['sharpe'] else 'NaN'} "
          f"Mono={r['mono']:.2f if r['mono'] else 'NaN'} TO={r['turnover']:.2f if r['turnover'] else 'NaN'} "
          f"G5S={r['g5_sharpe']:.2f if r['g5_sharpe'] else 'NaN'} Spike={spike_rate:.1%} N={N}")
