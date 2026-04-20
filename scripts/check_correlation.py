#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sync = pd.read_csv("data/factor_price_sync_v3.csv")
sync["date"] = pd.to_datetime(sync["date"])

pivot = sync.pivot_table(index="date", columns="stock_code", values="factor_value")

factors_map = {
    "amihud_illiq_v2": "data/factor_amihud_illiq_v2.csv",
    "shadow_pressure_v1": "data/factor_shadow_pressure_v1.csv",
    "overnight_momentum_v1": "data/factor_overnight_momentum_v1.csv",
    "gap_momentum_v1": "data/factor_gap_momentum_v1.csv",
    "tail_risk_cvar_v1": "data/factor_cvar_v1.csv",
    "neg_day_freq_v1": "data/factor_neg_day_freq_v1.csv",
    "turnover_level_v1": "data/factor_turnover_level_v1.csv",
    "tae_v1": "data/factor_tae_v1.csv",
    "amp_level_v2": "data/factor_amp_level_v2.csv",
    "ma_disp_v1": "data/factor_ma_disp_v1.csv",
    "vol_cv_neg_v1": "data/factor_vol_cv_neg.csv",
    "turnover_decel_v1": "data/factor_turnover_decel_v1.csv",
    "informed_flow_v1": "data/factor_informed_flow_v1.csv",
    "price_mom_5d_v1": "data/factor_price_mom_5d_v1.csv",
    "vol_ret_align_v1": "data/factor_vol_ret_align_rank_20d.csv",
    "vwap_dev_v1": "data/factor_vwap_dev_v1.csv",
}

header = "{:<25} {:>10}".format("Factor", "IC_corr")
print(header)
print("-" * 37)

corrs = {}
for name, path in factors_map.items():
    try:
        f = pd.read_csv(path)
        f["date"] = pd.to_datetime(f["date"])
        f_pivot = f.pivot_table(index="date", columns="stock_code", values="factor_value")
        common_dates = pivot.index.intersection(f_pivot.index)
        common_stocks = pivot.columns.intersection(f_pivot.columns)
        if len(common_dates) > 10 and len(common_stocks) > 10:
            p_sub = pivot.loc[common_dates, common_stocks]
            f_sub = f_pivot.loc[common_dates, common_stocks]
            corrs_list = []
            for date in common_dates:
                p_vals = p_sub.loc[date].dropna()
                f_vals = f_sub.loc[date].dropna()
                common = p_vals.index.intersection(f_vals.index)
                if len(common) > 30:
                    corr = np.corrcoef(p_vals[common].values, f_vals[common].values)[0, 1]
                    if np.isfinite(corr):
                        corrs_list.append(corr)
            if corrs_list:
                mean_corr = np.mean(corrs_list)
                corrs[name] = mean_corr
                marker = "  [REDUNDANT]" if abs(mean_corr) > 0.7 else ""
                print("{:<25} {:>+10.3f}{}".format(name, mean_corr, marker))
    except Exception as e:
        print("{:<25} ERROR: {}".format(name, e))

print()
if corrs:
    max_corr = max(abs(v) for v in corrs.values())
    print("Max |corr| = {:.3f}".format(max_corr))
