#!/usr/bin/env python3
"""
因子: 成交量激增后收益增强 (HVTR v3, corrected)
修正未来信息泄露 - 收益率用lag 1

构造:
  spike_strength(D) × fwd_ret(D-1)  # lag 1避免前视
    
论文: Gervais, Kaniel & Mingelgrin (2001) JF
"""

import numpy as np, pandas as pd, subprocess, json

def compute_hvtr_v3(k_spike=1.0, fwd_days=20):
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code','date']).reset_index(drop=True)
    g = df.groupby('stock_code')
    
    # fwd_ret 用lag 1: date D 用 D-1 的spike强度 + D-1 的fwd_ret
    # 但要先算fwd_ret(D) = close(D+fwd)/close(D) - 1 (未lag)
    df['fwd_ret_raw'] = g['close'].transform(lambda x: x.shift(-fwd_days)/x - 1)
    # lag 1天: date D 用 fwd_ret(D-1) = close(D-1+fwd)/close(D-1) - 1
    df['fwd_ret'] = g['fwd_ret_raw'].shift(1)
    
    # spike_strength (未lag, date D 用 D 的成交量特征)
    df['vol_ma_20'] = g['volume'].transform(lambda x: x.rolling(20,min_periods=10).mean())
    df['vol_std_20'] = g['volume'].transform(lambda x: x.rolling(20,min_periods=10).std().fillna(0))
    df['spike_strength'] = np.maximum(0, df['volume'] - (df['vol_ma_20'] + k_spike*df['vol_std_20']))
    df['spike_strength'] = df['spike_strength'] / (df['vol_std_20'] + 1)
    df['spike_strength'] = df['spike_strength'].clip(upper=3)
    
    # weighted fwd_ret
    df['w_fwd'] = df['spike_strength'] * df['fwd_ret']
    df['w_sum'] = df['spike_strength']
    
    def calc_weighted(gr):
        ws = gr['w_fwd'].rolling(20,min_periods=8).sum()
        wc = gr['w_sum'].rolling(20,min_periods=8).sum()
        return ws / (wc + 1e-10)
    
    df['factor_raw'] = g.apply(calc_weighted).reset_index(level=0, drop=True)
    df['log_amount_20d'] = np.log(df['amount'].rolling(20,min_periods=10).mean() + 1)
    
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
        med_val = float(np.median(res))
        mad_val = float(np.median(np.abs(np.array(res)-med_val))) * 1.4826
        if mad_val<1e-10 or np.std(res)<1e-10: continue
        res = np.clip(res, med_val-3*mad_val, med_val+3*mad_val)
        mu, sg = res.mean(), res.std()
        if sg<1e-10: continue
        gr = gr.copy(); gr['fn'] = (res-mu)/sg
        rows.append(gr[['date','stock_code','fn']])
    
    if not rows:
        print(f"ERROR k={k_spike}: no data")
        return
    
    fp = f'data/factor_hvtr_v3_k{k_spike}_fwd{fwd_days}.csv'
    pd.concat(rows, ignore_index=True).to_csv(fp, index=False)
    
    od = f'output/hvtr_v3_k{k_spike}_fwd{fwd_days}'
    cost = 0.002 if fwd_days >= 20 else 0.003
    subprocess.run([
        'python3','skills/alpha-factor-lab/scripts/factor_backtest.py',
        '--factor', fp, '--returns', 'data/csi1000_returns.csv',
        '--n-groups','5','--rebalance-freq','20',
        '--forward-days', str(fwd_days), '--cost', str(cost),
        '--output-report', f'{od}/report.json',
        '--output-dir', od, '--factor-name', f'hvtr_v3_k{k_spike}_fwd{fwd_days}'
    ], capture_output=True, text=True)
    with open(f'{od}/report.json') as f: r = json.load(f)
    m = r.get('metrics', {})
    print(f"HVTRv3 k={k_spike} fwd={fwd_days}: "
          f"IC={m.get('ic_mean',0):.4f} t={m.get('ic_t_stat',0):.2f} "
          f"LS_S={m.get('long_short_sharpe',0):.2f} Mono={m.get('monotonicity',0):.2f} "
          f"TO={m.get('turnover_mean',0):.2f}")
    return m

if __name__ == '__main__':
    compute_hvtr_v3(k_spike=0.75, fwd_days=20)
    compute_hvtr_v3(k_spike=1.0, fwd_days=20)
    compute_hvtr_v3(k_spike=1.25, fwd_days=20)
