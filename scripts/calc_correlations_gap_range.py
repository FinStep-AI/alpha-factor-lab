#!/usr/bin/env python3
"""计算新因子与所有入库因子的IC相关性"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

# 加载新因子
new = pd.read_csv('data/factor_gap_range_ratio_lb60.csv')
new['date'] = pd.to_datetime(new['date'])
new = new.rename(columns={'factor': 'gap_range_ratio_v1'})

# 加载所有入库因子
existing_factors = {
    'amihud_illiq_v2': 'data/factor_amihud_illiq_v2.csv',
    'shadow_pressure_v1': 'data/factor_shadow_pressure.csv',
    'overnight_momentum_v1': 'data/factor_overnight_momentum.csv',
    'gap_momentum_v1': 'data/factor_gap_momentum.csv',
    'tail_risk_cvar_v1': 'data/factor_tail_risk_cvar.csv',
    'neg_day_freq_v1': 'data/factor_neg_day_freq.csv',
    'turnover_level_v1': 'data/factor_turnover_level.csv',
    'tae_v1': 'data/factor_tae.csv',
    'pv_corr_v1': 'data/factor_pv_corr.csv',
    'amp_level_v2': 'data/factor_amp_level_v2.csv',
    'ma_disp_v1': 'data/factor_ma_disp_v6.csv',
    'vol_cv_neg_v1': 'data/factor_vol_cv_neg.csv',
    'turnover_decel_v1': 'data/factor_turnover_mom_neg.csv',
}

correlations = {}
for name, path in existing_factors.items():
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'factor': name})
        
        merged = new.merge(df[['date', 'stock_code', name]], on=['date', 'stock_code'], how='inner')
        
        # 截面相关性的时序平均
        corrs = []
        for dt, cs in merged.groupby('date'):
            cs = cs.dropna(subset=['gap_range_ratio_v1', name])
            if len(cs) > 50:
                c = cs['gap_range_ratio_v1'].corr(cs[name])
                if not np.isnan(c):
                    corrs.append(c)
        
        avg_corr = np.mean(corrs) if corrs else None
        correlations[name] = round(avg_corr, 3) if avg_corr is not None else None
        print(f"  {name:30s}: corr = {avg_corr:.3f} (N={len(corrs)})")
    except Exception as e:
        correlations[name] = None
        print(f"  {name:30s}: ERROR - {e}")

print(f"\nMax |corr| = {max(abs(v) for v in correlations.values() if v is not None):.3f}")
print(f"冗余检查 (>0.7): {'无冗余' if max(abs(v) for v in correlations.values() if v is not None) < 0.7 else '⚠️ 有冗余!'}")

# 保存
with open('output/gap_range_ratio_v1_lb60_20d/correlations.json', 'w') as f:
    json.dump(correlations, f, indent=2)
print(f"\n相关性已保存到 output/gap_range_ratio_v1_lb60_20d/correlations.json")
