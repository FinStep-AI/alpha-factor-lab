#!/usr/bin/env python3
"""
跳空/日内振幅比 参数优化版：测试不同窗口期
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def compute_factor(kline, lookback=20):
    kline = kline.sort_values(['stock_code', 'date']).copy()
    results = []
    
    for code, grp in kline.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        prev_close = grp['close'].shift(1)
        gap = np.abs(grp['open'].values / prev_close.values - 1)
        intra_range = (grp['high'].values - grp['low'].values) / prev_close.values + 0.001
        
        ratio = pd.Series(gap / intra_range, index=grp.index)
        factor_raw = ratio.rolling(lookback, min_periods=max(lookback//2, 5)).mean()
        
        sub = grp[['date', 'stock_code']].copy()
        sub['raw_factor'] = factor_raw.values
        results.append(sub)
    
    return pd.concat(results, ignore_index=True)


def neutralize(df, kline, amt_window=20):
    df = df.copy()
    kline_amt = kline[['date', 'stock_code', 'amount']].copy()
    kline_amt['date'] = pd.to_datetime(kline_amt['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    kline_amt = kline_amt.sort_values(['stock_code', 'date'])
    w = max(amt_window, 20)
    kline_amt['log_amount'] = kline_amt.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(w, min_periods=w//2).mean() + 1)
    )
    
    df = df.merge(kline_amt[['date', 'stock_code', 'log_amount']], on=['date', 'stock_code'], how='left')
    df['log_factor'] = np.log(df['raw_factor'].clip(lower=1e-10))
    
    output = []
    for dt, cs in df.groupby('date'):
        cs = cs.dropna(subset=['log_factor', 'log_amount']).copy()
        if len(cs) < 50:
            continue
        
        X = np.column_stack([np.ones(len(cs)), cs['log_amount'].values])
        y = cs['log_factor'].values
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y - X @ beta
        except:
            continue
        
        cs['r'] = residual
        med = np.nanmedian(cs['r'])
        mad = np.nanmedian(np.abs(cs['r'] - med))
        if mad > 1e-10:
            upper = med + 5.0 * 1.4826 * mad
            lower = med - 5.0 * 1.4826 * mad
            cs['r'] = cs['r'].clip(lower=lower, upper=upper)
        
        mean, std = cs['r'].mean(), cs['r'].std()
        cs['factor'] = (cs['r'] - mean) / std if std > 1e-10 else 0.0
        
        output.append(cs[['date', 'stock_code', 'factor']])
    
    return pd.concat(output, ignore_index=True)


if __name__ == '__main__':
    import subprocess, json
    
    print("加载数据...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    
    results_table = []
    
    for lookback in [10, 20, 40, 60]:
        print(f"\n=== Lookback={lookback}d ===")
        raw = compute_factor(kline, lookback=lookback)
        final = neutralize(raw, kline, amt_window=lookback)
        
        fname = f'data/factor_gap_range_ratio_lb{lookback}.csv'
        final.to_csv(fname, index=False)
        
        for fwd in [5, 10, 20]:
            outdir = f'output/gap_range_ratio_lb{lookback}_fwd{fwd}d'
            report = f'{outdir}/report.json'
            
            cmd = [
                'python3', 'skills/alpha-factor-lab/scripts/factor_backtest.py',
                '--factor', fname,
                '--returns', 'data/csi1000_returns.csv',
                '--n-groups', '5', '--forward-days', str(fwd),
                '--cost', '0.003',
                '--output-dir', outdir + '/',
                '--output-report', report
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if r.returncode == 0:
                with open(report) as f:
                    rpt = json.load(f)
                ic = rpt.get('ic_mean', 0)
                ic_t = rpt.get('ic_t', 0)
                mono = rpt.get('monotonicity', 0)
                sharpe = rpt.get('long_short_sharpe', 0)
                g5_sharpe = rpt.get('group_sharpe', [0,0,0,0,0])[4]
                mdd = rpt.get('long_short_mdd', 0)
                
                flag = "✓" if abs(ic) > 0.02 and abs(ic_t) > 2.0 and mono > 0.8 else " "
                results_table.append({
                    'lb': lookback, 'fwd': fwd, 'ic': ic, 'ic_t': ic_t,
                    'mono': mono, 'sharpe': sharpe, 'g5_sharpe': g5_sharpe,
                    'mdd': mdd, 'pass': flag
                })
                print(f"  fwd={fwd}d: IC={ic:.4f} t={ic_t:.2f} mono={mono:.1f} sharpe={sharpe:.2f} g5_sharpe={g5_sharpe:.2f} {flag}")
    
    print("\n=== 汇总 ===")
    for r in results_table:
        print(f"lb={r['lb']:2d} fwd={r['fwd']:2d}d | IC={r['ic']:.4f} t={r['ic_t']:.2f} | mono={r['mono']:.1f} | LS_sharpe={r['sharpe']:.2f} | G5_sharpe={r['g5_sharpe']:.2f} | MDD={r['mdd']:.1%} {r['pass']}")
