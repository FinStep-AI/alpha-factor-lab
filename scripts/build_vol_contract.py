# -*- coding: utf-8 -*-
"""快速因子：波动率收缩信号 (Volatility Contraction)
因子ID: vol_contract_v1
逻辑：过去20日波动率标准差 < 过去40日波动率标准差 → 波动收缩 → 正alpha
期限：5日前瞻，20日调仓。
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
kline = pd.read_csv(PROJ / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
kline['pct_change'] = pd.to_numeric(kline['pct_change'], errors='coerce')

print("构建波动收缩因子...")
# 20日/40日波动率
kline['vol_20d'] = kline.groupby('stock_code')['pct_change'].transform(
    lambda x: x.rolling(20, min_periods=10).std()
)
kline['vol_40d'] = kline.groupby('stock_code')['pct_change'].transform(
    lambda x: x.rolling(40, min_periods=20).std()
)
kline['vol_spread'] = kline['vol_20d'] - kline['vol_40d']  # 正值=收缩

# 成交额中性化
kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

kline['neutral_factor'] = np.nan
for date, grp in kline.groupby('date'):
    idx = grp.index
    f  = grp['vol_spread'].values.astype(float)
    na = grp['amount_20d'].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(na) & (na > 0)
    if mask.sum() < 30:
        continue
    f_m, n_m = f[mask], na[mask]
    med_f, mad_f = np.median(f_m), np.median(np.abs(f_m - np.median(f_m)))
    if mad_f > 0:
        f_m = np.clip(f_m, med_f - 5.5*1.4826*mad_f, med_f + 5.5*1.4826*mad_f)
    X = np.column_stack([np.ones(len(n_m)), np.log(n_m + 1)])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, f_m, rcond=None)
        resid = f_m - X @ beta
        std = resid.std()
        if std > 0:
            resid = (resid - resid.mean()) / std
        kline.loc[idx[mask], 'neutral_factor'] = resid
    except Exception:
        pass

out = kline[['date', 'stock_code', 'neutral_factor']].dropna(subset=['neutral_factor'])
outPath = PROJ / 'data' / 'factor_vol_contract_v1.csv'
out[['date', 'stock_code', 'neutral_factor']].rename(columns={'neutral_factor':'factor_value'}).to_csv(outPath, index=False)
print(f"因子CSV: {outPath} ({len(out)} 行)")
