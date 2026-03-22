"""
因子: 波动率不对称性 (Volatility Asymmetry, vol_asym_v1)
构造: log(upside_vol / downside_vol), 20日滚动窗口, 成交额中性化
逻辑: 上行波动>下行波动 的股票 = 上涨弹性好/下跌抗性强 → Quality代理/Growth代理
      高vol_asym = 涨多跌少特征 → 市场对其乐观但不是短期超买
方向: 正向 (高因子值=高预期收益)
Barra风格: Volatility / Quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── 读数据 ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", parse_dates=["date"])
print(f"K线数据: {kline.shape[0]} 行, {kline['stock_code'].nunique()} 只股票")
print(f"日期范围: {kline['date'].min()} ~ {kline['date'].max()}")

# ── 计算日收益率 ────────────────────────────────────────
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
kline["ret"] = kline.groupby("stock_code")["close"].pct_change()

# ── 20日滚动上行/下行波动率 ──────────────────────────────
def calc_vol_asym(group, window=20, min_periods=10):
    """计算上行和下行波动率的比值"""
    ret = group["ret"].values
    dates = group["date"].values
    n = len(ret)
    
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        start = max(0, i - window + 1)
        rets = ret[start:i+1]
        valid = rets[~np.isnan(rets)]
        
        if len(valid) < min_periods:
            continue
        
        up_rets = valid[valid > 0]
        down_rets = valid[valid < 0]
        
        # 至少各需3个正/负收益日
        if len(up_rets) < 3 or len(down_rets) < 3:
            continue
        
        up_vol = np.std(up_rets, ddof=1)
        down_vol = np.std(down_rets, ddof=1)
        
        if down_vol < 1e-10:
            continue
        
        result[i] = np.log(up_vol / down_vol)
    
    group["vol_asym_raw"] = result
    return group

print("计算波动率不对称性...")
kline = kline.groupby("stock_code", group_keys=False).apply(calc_vol_asym)
print(f"  非空因子值: {kline['vol_asym_raw'].notna().sum()} / {len(kline)}")

# ── 成交额中性化 ────────────────────────────────────────
# 20日平均成交额(对数)
kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

def neutralize_ols(df):
    """截面OLS中性化: 因子对log_amount回归取残差"""
    mask = df["vol_asym_raw"].notna() & df["log_amount_20d"].notna()
    if mask.sum() < 30:
        df["factor"] = np.nan
        return df
    
    y = df.loc[mask, "vol_asym_raw"].values
    x = df.loc[mask, "log_amount_20d"].values
    X = np.column_stack([np.ones(len(x)), x])
    
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
    except:
        df["factor"] = np.nan
        return df
    
    df["factor"] = np.nan
    df.loc[mask, "factor"] = resid
    return df

print("截面中性化...")
kline = kline.groupby("date", group_keys=False).apply(neutralize_ols)

# ── MAD winsorize + z-score ──────────────────────────────
def mad_zscore(df):
    vals = df["factor"]
    mask = vals.notna()
    if mask.sum() < 30:
        df["factor"] = np.nan
        return df
    
    med = vals[mask].median()
    mad = (vals[mask] - med).abs().median() * 1.4826
    if mad < 1e-10:
        df["factor"] = np.nan
        return df
    
    # Winsorize at ±3 MAD
    lower = med - 3 * mad
    upper = med + 3 * mad
    clipped = vals.clip(lower, upper)
    
    # z-score
    mu = clipped[mask].mean()
    sigma = clipped[mask].std()
    if sigma < 1e-10:
        df["factor"] = np.nan
        return df
    
    df["factor"] = (clipped - mu) / sigma
    return df

print("MAD winsorize + z-score...")
kline = kline.groupby("date", group_keys=False).apply(mad_zscore)

# ── 输出 ────────────────────────────────────────────────
output = kline[["date", "stock_code", "factor"]].dropna(subset=["factor"])
output = output.rename(columns={"factor": "vol_asym_v1"})
output_path = DATA_DIR / "factor_vol_asym_v1.csv"
output.to_csv(output_path, index=False)
print(f"\n输出: {output_path}")
print(f"  行数: {len(output)}")
print(f"  日期: {output['date'].min()} ~ {output['date'].max()}")
print(f"  股票: {output['stock_code'].nunique()}")
print(f"  因子分布: mean={output['vol_asym_v1'].mean():.4f}, std={output['vol_asym_v1'].std():.4f}")
print(f"  分位数: {output['vol_asym_v1'].quantile([0.01, 0.25, 0.50, 0.75, 0.99]).to_dict()}")
