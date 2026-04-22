"""
因子：振幅衰减率 (Amplitude Decay Rate, ADR_v1)
ID: amp_decay_rate_v1

逻辑：
  振幅 = (high-low)/close，反映日内波动程度。
  若最近5日振幅出现明显的指数衰减（趋势下降），说明波动正在收缩，
  市场开始进入整理/突破阶段。对小盘股而言，波动率收缩常伴随趋势加速。

  对最近N日振幅做指数衰减拟合 y = a·exp(-b·t) + c，取衰减速率 b。
  b > 0 → 振幅下降 = 波动率收缩
  b < 0 → 振幅上升 = 波动率扩散

假设：波动率快速收缩的股票后续有更强的趋势alpha。

Barra风格: Volatility/趋势
计算：neutralize(exp_fit_decay(amplitude, 5), log_amount)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def exp_decay_rate(series, window=5):
    """
    Fit y = a*exp(-b*t) + c by nonlinear least squares.
    Returns decay rate b (positive = decaying amplitude).
    """
    vals = np.asarray(series.values, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    
    t = np.arange(window, dtype=float)
    t_c = t - t.mean()
    X_dt = np.sum(t_c**2)
    
    for i in range(window-1, n):
        y = vals[i-window+1:i+1]
        key = np.isfinite(y)
        y_clean = y[key]
        if len(y_clean) < 4:
            continue
        
        # Non-linear least squares for b using iterative Gauss-Newton
        # Start with log-linear approximation: ln(y-c) = ln(a) - b*t
        # Estimate c as the minimum y
        c0 = np.percentile(y_clean, 10)  # floor estimate
        # Ensure c < min(y)
        c0 = min(c0, y_clean.min() * 0.95)
        c0 = max(c0, 1e-8)
        
        resid_prev = float('inf')
        c = c0
        for _iter in range(10):
            y_adj = y_clean - c
            mask = y_adj > 0
            if mask.sum() < 3:
                break
            ln_y = np.log(y_adj[mask])
            t_use = t[:len(y_clean)][mask]
            
            # Simple linear regression of ln_y ~ -t
            X = np.column_stack([np.ones(len(t_use)), -t_use])
            try:
                coef = np.linalg.lstsq(X, ln_y, rcond=None)[0]
                ln_a, b_est = float(coef[0]), float(coef[1])
                a_est = np.exp(ln_a)
                # Update c via linearization
                y_pred = a_est * np.exp(-b_est * t) + c
                residual = np.sum((y_clean - y_pred)**2)
                if residual > resid_prev * 1.1:  # diverged
                    break
                resid_prev = residual
                c = c * 0.9 + np.percentile(y_clean, 5) * 0.1  # slightly adjust
            except:
                break
        
        out[i] = b_est if np.isfinite(b_est) and b_est != float('inf') else np.nan
    
    return out


def neutralize_vectorized(df):
    """Vectorized neutralize per date."""
    out_rows = []
    for date, g in df.groupby('date'):
        g = g.dropna(subset=['factor_raw', 'log_amount']).copy()
        if len(g) < 30:
            continue
        
        vals = g['factor_raw'].values.astype(float)
        ctrl = g['log_amount'].values.astype(float)
        
        mask = np.isfinite(vals) & np.isfinite(ctrl)
        v, x = vals[mask], ctrl[mask]
        
        if len(v) < 30:
            continue
        
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, v, rcond=None)[0]
            r = v - X @ beta
        except Exception:
            continue
        
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10:
            continue
        r = np.clip(r, med - 5.2*mad, med + 5.2*mad)
        std = r.std()
        if std < 1e-10:
            continue
        z = (r - np.median(r)) / std
        
        g_out = g.iloc[np.where(mask)[0]][['stock_code','date']].copy()
        g_out['factor_value'] = z
        out_rows.append(g_out)
    
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def main():
    print("Loading kline data...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"Shape: {kline.shape}")
    
    # Compute amplitude
    print("Computing amplitude...")
    price_range = kline['high'] - kline['low']
    price_range = price_range.replace(0, np.nan)
    kline['amplitude'] = price_range / kline['close'].clip(lower=0.01)
    
    # Rolling decay rate
    print("Computing 5-day amplitude decay rate...")
    results = []
    stock_count = 0
    for code, grp in kline.groupby('stock_code'):
        stock_count += 1
        if stock_count % 200 == 0:
            print(f"  Processed {stock_count}/1000...")
        grp = grp.sort_values('date').reset_index(drop=True)
        decay = exp_decay_rate(grp['amplitude'], window=5)
        grp_out = grp[['stock_code','date','amount']].copy()
        grp_out['factor_raw'] = decay
        results.append(grp_out)
    
    combined = pd.concat(results, ignore_index=True)
    print(f"Raw factor: {combined.shape}, non-null: {combined['factor_raw'].notna().sum()}")
    
    # Neutralize
    combined = combined.dropna(subset=['factor_raw', 'amount']).copy()
    combined['log_amount'] = np.log(combined['amount'].clip(lower=1))
    
    print("Neutralizing...")
    result = neutralize_vectorized(combined[['date','stock_code','factor_raw','log_amount']])
    
    print(f"Final: {result.shape}, mean={result['factor_value'].mean():.4f}, std={result['factor_value'].std():.4f}")
    result.to_csv('data/factor_amp_decay_rate_v1.csv', index=False)
    print("Saved.")


if __name__ == '__main__':
    main()
