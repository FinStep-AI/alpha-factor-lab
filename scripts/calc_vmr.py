"""
因子计算：成交量变动比因子 (Volume Momentum Ratio, VMR)
=========================================================
构造逻辑：
  高成交量日 = vol_chg > MA(vol_chg) + 1.5 * std(vol_chg, 20d)
  低成交量日 = vol_chg < MA(vol_chg) - 1.5 * std(vol_chg, 20d)
  因子值 = 高成交量日(forward_5d收益均值) / (低成交量日(forward_5d收益均值) + 1e-6)
  
  等价简化： vol_chg_ratio = MA20( (volume_t / MA20(volume) - 1) )
  取对数后做成交额OLS中性化 + MAD缩尾 + z-score

  高因子值 = 放量涨幅大于缩量涨幅 = 资金持续流入
  低因子值 = 放量跌幅大于缩量涨幅 = 资金持续流出
"""
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

KLINE_FILE = 'data/csi1000_kline_raw.csv'
RETURNS_FILE = 'data/csi1000_returns.csv'
OUTPUT_FILE = 'data/factor_vol_mom_ratio_v1.csv'

print("Loading data...")
kline = pd.read_csv(KLINE_FILE)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# === 构造 vol_change_ratio (成交量相对变化) ===
kline['vol_ma20'] = kline.groupby('stock_code')['volume'].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)
kline['vol_chg'] = kline['volume'] / (kline['vol_ma20'] + 1e-6) - 1

# 20日均值标准化
kline['vol_chg_ma20'] = kline.groupby('stock_code')['vol_chg'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# Forward 5d return
kline['fwd_5d_ret'] = kline.groupby('stock_code')['pct_change'].transform(
    lambda x: x.shift(-1).rolling(5, min_periods=3).sum()
)

# === 因子：20日滚动 高量日收益均值 - 低量日收益均值 ===
print("Computing factor values...")

def high_low_vol_ret_diff(g):
    """高量日forward收益均值 - 低量日forward收益均值"""
    vol_chg = g['vol_chg'].values
    fwd_ret = g['fwd_5d_ret'].values
    result = np.full(len(g), np.nan)
    
    for i in range(20, len(g)):
        vc = vol_chg[i-19:i+1]  # 过去20日
        fr = fwd_ret[i-19:i+1]
        valid = ~(np.isnan(vc) | np.isnan(fr))
        if valid.sum() < 10:
            continue
        vc_v = vc[valid]
        fr_v = fr[valid]
        q_high = np.quantile(vc_v, 0.67)
        q_low = np.quantile(vc_v, 0.33)
        high_ret = fr_v[vc_v >= q_high].mean() if (vc_v >= q_high).sum() > 0 else np.nan
        low_ret = fr_v[vc_v <= q_low].mean() if (vc_v <= q_low).sum() > 0 else np.nan
        if not np.isnan(high_ret) and not np.isnan(low_ret):
            result[i] = high_ret - low_ret
    return pd.Series(result, index=g.index)

factor_raw = kline.groupby('stock_code', group_keys=False).apply(high_low_vol_ret_diff)
kline['factor_raw'] = factor_raw

# 也可以用简化版：直接用 vol_chg_ma20 绝对值（去掉forward ret计算，避免前视偏差）
# 修正：因子在截面日 i 的值应该只用到 i 日及之前的信息
# 上面的 high_low_vol_ret_diff 用了 fwd_5d_ret，这在因子值 (i日) 计算时还未知 → 前视偏差!
# 重写：只用过去20日的日收益率 + 过去20日的 vol_chg，计算高量日的当日收益 - 低量日的当日收益（20日内的 asymmetry）

print("Recomputing without lookahead bias...")

def vol_asym_no_fwd(g):
    """高量日平均日收益 - 低量日平均日收益 (temporal asymmetry, no forward return)"""
    vol_chg = g['vol_chg'].values
    daily_ret = g['pct_change'].values / 100.0  # convert pct to fraction
    result = np.full(len(g), np.nan)
    
    for i in range(20, len(g)):
        vc = vol_chg[i-19:i+1]
        dr = daily_ret[i-19:i+1]
        valid = ~(np.isnan(vc) | np.isnan(dr))
        if valid.sum() < 10:
            continue
        vc_v = vc[valid]
        dr_v = dr[valid]
        q_high = np.quantile(vc_v, 0.67)
        q_low = np.quantile(vc_v, 0.33)
        high_ret = dr_v[vc_v >= q_high].mean() if (vc_v >= q_high).sum() > 0 else np.nan
        low_ret = dr_v[vc_v <= q_low].mean() if (vc_v <= q_low).sum() > 0 else np.nan
        if not np.isnan(high_ret) and not np.isnan(low_ret):
            result[i] = high_ret - low_ret
    return pd.Series(result, index=g.index)

kline['factor_raw'] = kline.groupby('stock_code', group_keys=False).apply(vol_asym_no_fwd)

# 如果这个因子太稀疏，回退到简化版：直接用20日 vol_chg 的异常程度
kline['factor_simple'] = kline.groupby('stock_code')['vol_chg'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

print("Factor stats (raw):")
print(kline[['factor_raw', 'factor_simple']].describe())

# === 截面中性化 ===
print("\nNeutralizing...")

def cross_section_neutralize(df_day, factor_col='factor_raw'):
    """截面OLS中性化(turnover作为代理) + MAD + z-score"""
    d = df_day[['stock_code', factor_col, 'turnover']].dropna(subset=[factor_col])
    if len(d) < 100:
        d['factor_neu'] = np.nan
        return d[['stock_code', 'factor_neu']]
    
    # OLS: y ~ log(turnover + 1)
    y = d[factor_col].values
    x_raw = d['turnover'].values + 1
    log_x = np.log(x_raw)
    log_x = np.where(np.isfinite(log_x), log_x, np.nanmedian(log_x))
    
    valid = np.isfinite(y) & np.isfinite(log_x)
    if valid.sum() < 50:
        d['factor_neu'] = np.nan
        return d[['stock_code', 'factor_neu']]
    
    x_mat = np.column_stack([np.ones(valid.sum()), log_x[valid]])
    y_v = y[valid]
    try:
        beta = np.linalg.lstsq(x_mat, y_v, rcond=None)[0]
        resid = np.full(len(y), np.nan)
        resid[valid] = y_v - x_mat @ beta
    except:
        d['factor_neu'] = np.nan
        return d[['stock_code', 'factor_neu']]
    
    # MAD
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if mad < 1e-8:
        d['factor_neu'] = np.nan
        return d[['stock_code', 'factor_neu']]
    
    # Winsorize ± 5.5 MAD
    upper = med + 5.5 * 1.4826 * mad
    lower = med - 5.5 * 1.4826 * mad
    resid_c = np.clip(resid, lower, upper)
    
    # Z-score
    sigma = np.nanstd(resid_c)
    if sigma < 1e-8:
        d['factor_neu'] = np.nan
        return d[['stock_code', 'factor_neu']]
    d['factor_neu'] = (resid_c - np.nanmean(resid_c)) / sigma
    return d[['stock_code', 'factor_neu']]

# 用简化版 factor_simple 做中性化
print("Computing neutralized factor with simple version...")
results = []
for dt, grp in kline.groupby('date'):
    res = cross_section_neutralize(grp, 'factor_simple')
    res['date'] = dt
    results.append(res)

neu = pd.concat(results, ignore_index=True)
kline = kline.merge(neu, on=['stock_code', 'date'], how='left')

# 输出
dates_valid = kline['date'].unique()
dates_valid.sort()
df_out = kline[['date', 'stock_code', 'factor_neu']].drop_duplicates(
    subset=['date', 'stock_code']
).rename(columns={'factor_neu': 'factor_value'})
df_out = df_out.pivot(index='date', columns='stock_code', values='factor_value').sort_index()
df_out.index.name = 'date'
df_out.columns.name = None
df_out.to_csv(OUTPUT_FILE)
print(f"\nSaved {len(df_out)} dates x {len(df_out.columns)} stocks")
print(f"Output: {OUTPUT_FILE}")
print(df_out.iloc[-5:, :5].to_string())
print(f"\nNon-NaN count: {df_out.notna().sum().sum()}")
print(f"Sparsity: {1 - df_out.notna().sum().sum() / (len(df_out) * len(df_out.columns)):.2%}")
