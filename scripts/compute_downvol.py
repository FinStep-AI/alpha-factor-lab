"""
下行量价共振因子 (Downside Volume-Price Co-movement) v1
========================================================
逻辑：
- 结合CVaR(关注下行尾部)和PV相关(量价共振)的思路
- 关注"恐慌性抛售"：下跌日的成交量集中程度
- 如果近期下跌日都伴随放量 = 恐慌抛售严重 = 可能超卖反弹

构造：
  对近20天中的下跌日(ret<0)：
    down_vol_share = sum(volume_down_days) / sum(volume_all_days)
    即下跌日成交量占总成交量的比例
  
  factor = -down_vol_share  (负号：高恐慌抛售→因子低→可能反弹)
  
  也可以理解为：factor = 1 - down_vol_share = up_vol_share
  即上涨日成交量占比，高 = 资金流入为主

改进版：
  v2: 量加权下行收益 = sum(ret_neg * vol) / sum(vol)  (类VolumeWeightedCVaR)  
  v3: 下行日均量/上行日均量 的倒数
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    window = 20
    
    results = []
    for stock_code, gdf in df.groupby('stock_code'):
        gdf = gdf.sort_values('date').reset_index(drop=True)
        n = len(gdf)
        
        v1_vals, v2_vals, v3_vals, dates = [], [], [], []
        
        for i in range(window - 1, n):
            w = gdf.iloc[i - window + 1: i + 1]
            rets = w['daily_ret'].values
            vols = w['volume'].values
            
            valid = ~(np.isnan(rets) | np.isnan(vols))
            if valid.sum() < 10:
                v1_vals.append(np.nan); v2_vals.append(np.nan); v3_vals.append(np.nan)
                dates.append(gdf.iloc[i]['date'])
                continue
            
            r = rets[valid]
            v = vols[valid]
            
            down_mask = r < 0
            up_mask = r > 0
            
            # v1: down_vol_share → 取反
            total_vol = v.sum()
            if total_vol > 0:
                down_vol_share = v[down_mask].sum() / total_vol
                v1_vals.append(-down_vol_share)  # 负号: 高下行量占比 = 低因子值
            else:
                v1_vals.append(np.nan)
            
            # v2: volume-weighted average downside return
            if down_mask.sum() >= 3:
                vw_down_ret = np.average(r[down_mask], weights=v[down_mask])
                v2_vals.append(-vw_down_ret)  # 负号: 大幅下跌 = 低因子值
            else:
                v2_vals.append(np.nan)
            
            # v3: down_day_avg_vol / up_day_avg_vol (取反)
            if down_mask.sum() >= 3 and up_mask.sum() >= 3:
                down_avg_vol = v[down_mask].mean()
                up_avg_vol = v[up_mask].mean()
                ratio = down_avg_vol / (up_avg_vol + 1)
                v3_vals.append(-np.log(ratio + 0.01))  # 低下行量比 = 高因子值
            else:
                v3_vals.append(np.nan)
            
            dates.append(gdf.iloc[i]['date'])
        
        results.append(pd.DataFrame({
            'date': dates, 'stock_code': stock_code,
            'factor_v1': v1_vals, 'factor_v2': v2_vals, 'factor_v3': v3_vals
        }))
    
    factor_df = pd.concat(results, ignore_index=True)
    print(f"Computed: {factor_df.shape}")
    
    # Amount for neutralization
    amt = df[['date', 'stock_code', 'amount']].copy()
    amt['avg_amount_20d'] = amt.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    amt['log_amount'] = np.log(amt['avg_amount_20d'].clip(lower=1))
    factor_df = factor_df.merge(amt[['date', 'stock_code', 'log_amount']], on=['date', 'stock_code'], how='left')
    
    for version, col in [('v1', 'factor_v1'), ('v2', 'factor_v2'), ('v3', 'factor_v3')]:
        fdf = factor_df[['date', 'stock_code', col, 'log_amount']].copy()
        fdf = fdf.rename(columns={col: 'factor_raw'})
        fdf = fdf.dropna(subset=['factor_raw', 'log_amount'])
        
        def neutralize(g):
            y, x = g['factor_raw'].values, g['log_amount'].values
            v = ~(np.isnan(y)|np.isnan(x))
            if v.sum()<30: g['factor']=np.nan; return g
            X = np.column_stack([np.ones(v.sum()), x[v]])
            try:
                b = np.linalg.lstsq(X, y[v], rcond=None)[0]
                r = np.full(len(y), np.nan); r[v] = y[v] - X@b; g['factor'] = r
            except: g['factor'] = np.nan
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(neutralize)
        
        def winsorize_mad(g):
            v = g['factor'].values; m = ~np.isnan(v)
            if m.sum()<10: return g
            med = np.nanmedian(v); mad = np.nanmedian(np.abs(v[m]-med))
            if mad<1e-10: return g
            g['factor'] = np.clip(v, med-3*1.4826*mad, med+3*1.4826*mad)
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(winsorize_mad)
        
        def zscore(g):
            v = g['factor'].values; m = ~np.isnan(v)
            if m.sum()<10: return g
            mn, s = np.nanmean(v[m]), np.nanstd(v[m])
            g['factor'] = (v-mn)/s if s>1e-10 else 0.0
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(zscore)
        out = fdf[['date','stock_code','factor']].dropna(subset=['factor'])
        out.to_csv(f'data/factor_downvol_{version}.csv', index=False)
        print(f"Downvol {version}: {out.shape}")

if __name__ == '__main__':
    compute_factor()
