#!/usr/bin/env python3
"""
HVTR v2: 改进版 - 加入spike强度加权 + 更长前瞻 + 使用激增累积强度
"""
import numpy as np, pandas as pd, subprocess, json, sys

def compute_hvtr_v2(k_spike=1.0, fwd_days=20):
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code','date']).reset_index(drop=True)
    g = df.groupby('stock_code')
    df['fwd_ret'] = g['close'].transform(lambda x: x.shift(-fwd_days)/x - 1)
    
    # volume stats for spike detection
    df['vol_ma_20'] = g['volume'].transform(lambda x: x.rolling(20,min_periods=10).mean())
    df['vol_std_20'] = g['volume'].transform(lambda x: x.rolling(20,min_periods=10).std().fillna(0))
    df['spike_thresh'] = df['vol_ma_20'] + k_spike * df['vol_std_20']
    
    # spike strength: how far above threshold
    df['spike_strength'] = np.maximum(0, df['volume'] - df['spike_thresh']) / (df['vol_std_20'] + 1)
    df['spike_strength'] = df['spike_strength'].clip(upper=3)  # cap
    
    # weighted forward return: spike_strength * fwd_ret, 20d rolling
    df['weighted_fwd'] = df['spike_strength'] * df['fwd_ret']
    df['weight_sum'] = df['spike_strength']
    
    def calc_weighted(group):
        wsum = group['weighted_fwd'].rolling(20, min_periods=8).sum()
        wcount = group['weight_sum'].rolling(20, min_periods=8).sum()
        return wsum / (wcount + 1e-10)
    
    df['factor_raw'] = g.apply(calc_weighted).reset_index(level=0, drop=True)
    df['log_amount_20d'] = np.log(df['amount'].rolling(20, min_periods=10).mean() + 1)
    
    rows = []
    for date, group in df.groupby('date'):
        gr = group.dropna(subset=['factor_raw','log_amount_20d'])
        if len(gr) < 30: continue
        x, y = gr['log_amount_20d'].values, gr['factor_raw'].values
        X = np.column_stack([np.ones(len(x)), x])
        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            res = y - X@b
        except: continue
        if len(res)==0: continue
        med = np.median(res); mad = np.median(np.abs(res-med))*1.4826
        if mad<1e-10: continue
        res = np.clip(res, med-3*mad, med+3*mad)
        mu, sg = res.mean(), res.std()
        if sg<1e-10: continue
        gr = gr.copy(); gr['fn'] = (res-mu)/sg
        rows.append(gr[['date','stock_code','fn']])
    
    if not rows:
        print(f"ERROR k={k_spike} fwd={fwd_days}: no data", file=sys.stderr)
        return
    
    fp = f'data/factor_hvtr_v2_k{k_spike}_fwd{fwd_days}.csv'
    pd.concat(rows, ignore_index=True).to_csv(fp, index=False)
    
    od = f'output/hvtr_v2_k{k_spike}_fwd{fwd_days}'
    cost = 0.002 if fwd_days >= 20 else 0.003
    subprocess.run([
        'python3','skills/alpha-factor-lab/scripts/factor_backtest.py',
        '--factor', fp, '--returns', 'data/csi1000_returns.csv',
        '--n-groups','5','--rebalance-freq','20',
        '--forward-days', str(fwd_days), '--cost', str(cost),
        '--output-report', f'{od}/report.json',
        '--output-dir', od, '--factor-name', f'hvtr_v2_k{k_spike}_fwd{fwd_days}'
    ], capture_output=True, text=True)
    with open(f'{od}/report.json') as f: r = json.load(f)
    m = r.get('metrics', {})
    print(f"HVTRv2 k={k_spike} fwd={fwd_days}: "
          f"IC={m.get('ic_mean',0):.4f} t={m.get('ic_t_stat',0):.2f} "
          f"LS_S={m.get('long_short_sharpe',0):.2f} Mono={m.get('monotonicity',0):.2f} "
          f"Turnover={m.get('turnover_mean',0):.2f}")

if __name__ == '__main__':
    # 重点测试: 高IC高Sharpe+低换手
    compute_hvtr_v2(k_spike=0.75, fwd_days=20)
    compute_hvtr_v2(k_spike=1.0, fwd_days=20)
    compute_hvtr_v2(k_spike=1.25, fwd_days=20)
    compute_hvtr_v2(k_spike=0.75, fwd_days=10)
    compute_hvtr_v2(k_spike=1.0, fwd_days=10)
