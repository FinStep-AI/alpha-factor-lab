#!/usr/bin/env python3
"""
成交量趋势斜率因子 (Volume Slope Factor)

思路：
对每只股票近20日成交量做OLS线性回归 vol(t) = a + b*t，取斜率b。
再对斜率做成交额OLS中性化 + MAD winsorize + z-score。

高斜率 = 成交量持续放大 = 投资者关注度上升 = 信息扩散加速
低斜率 = 成交量持续萎缩 = 关注度下降

与已有因子区别：
  turnover_level: 换手率水平（高低）
  vol_cv_neg: 成交量稳定性（波动大小）
  turnover_decel: 短/长期换手比（5d vs 20d ratio）
  vol_slope: 成交量线性趋势方向和强度（时间维度斜率）
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---- 参数 ----
LOOKBACK = 20
OUTPUT_PATH = "data/factor_vol_slope_v1.csv"

# ---- 读取数据 ----
print("读取K线数据...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ---- 计算成交量斜率 ----
print(f"计算{LOOKBACK}日成交量线性回归斜率...")

def volume_slope(vol_series):
    """对成交量序列做OLS线性回归，返回标准化斜率"""
    n = len(vol_series)
    if n < LOOKBACK:
        return np.nan
    vol = vol_series.values[-LOOKBACK:]
    if np.any(np.isnan(vol)) or np.all(vol == 0):
        return np.nan
    # 标准化：斜率除以均值，得到相对变化率
    mean_vol = np.mean(vol)
    if mean_vol <= 0:
        return np.nan
    t = np.arange(LOOKBACK, dtype=float)
    # OLS: b = cov(t, vol) / var(t)
    t_mean = t.mean()
    vol_mean = vol.mean()
    cov_tv = np.sum((t - t_mean) * (vol - vol_mean))
    var_t = np.sum((t - t_mean) ** 2)
    if var_t == 0:
        return np.nan
    slope = cov_tv / var_t
    # 标准化斜率：斜率/均值，衡量每日相对于均值的变化率
    return slope / mean_vol

results = []
for code, grp in df.groupby("stock_code"):
    grp = grp.sort_values("date")
    dates = grp["date"].values
    volumes = grp["volume"].values
    
    for i in range(LOOKBACK - 1, len(grp)):
        vol_window = volumes[i - LOOKBACK + 1:i + 1]
        mean_vol = np.mean(vol_window)
        if mean_vol <= 0 or np.any(np.isnan(vol_window)):
            continue
        t = np.arange(LOOKBACK, dtype=float)
        t_mean = t.mean()
        vol_mean = vol_window.mean()
        cov_tv = np.sum((t - t_mean) * (vol_window - vol_mean))
        var_t = np.sum((t - t_mean) ** 2)
        if var_t == 0:
            continue
        slope = cov_tv / var_t
        norm_slope = slope / mean_vol  # 标准化斜率
        results.append({
            "date": dates[i],
            "stock_code": code,
            "factor_raw": norm_slope
        })

factor_df = pd.DataFrame(results)
print(f"原始因子值: {len(factor_df)} 条")

# ---- 计算20日平均成交额（用于中性化）----
print("计算20日平均成交额...")
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=15).mean() + 1)
)
amount_map = df[["date", "stock_code", "log_amount_20d"]].copy()
factor_df = factor_df.merge(amount_map, on=["date", "stock_code"], how="left")

# ---- 截面处理：成交额OLS中性化 + MAD winsorize + z-score ----
print("截面处理: OLS中性化 + MAD winsorize + z-score...")

def cross_section_process(group):
    """单日截面处理"""
    vals = group["factor_raw"].values.copy()
    neutralizer = group["log_amount_20d"].values.copy()
    
    # 去除nan
    valid = ~(np.isnan(vals) | np.isnan(neutralizer))
    if valid.sum() < 50:
        group["factor_value"] = np.nan
        return group
    
    # OLS中性化: factor = a + b*log_amount + residual
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(valid.sum()), neutralizer[valid]])
    y = vals[valid]
    try:
        beta, _, _, _ = lstsq(X, y, rcond=None)
        residuals = y - X @ beta
    except:
        group["factor_value"] = np.nan
        return group
    
    # MAD winsorize
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-10:
        mad = np.std(residuals)
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    residuals = np.clip(residuals, lower, upper)
    
    # z-score
    std = np.std(residuals)
    if std < 1e-10:
        group["factor_value"] = np.nan
        return group
    z = (residuals - np.mean(residuals)) / std
    
    result = np.full(len(vals), np.nan)
    result[valid] = z
    group["factor_value"] = result
    return group

factor_df = factor_df.groupby("date", group_keys=False).apply(cross_section_process)

# ---- 输出 ----
output = factor_df[["date", "stock_code", "factor_value"]].dropna(subset=["factor_value"])
output.to_csv(OUTPUT_PATH, index=False)
print(f"因子输出: {len(output)} 条 → {OUTPUT_PATH}")

# 统计
print("\n=== 因子统计 ===")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"截面股票数: {output.groupby('date')['stock_code'].count().mean():.0f} (平均)")
print(f"因子值分布: mean={output['factor_value'].mean():.4f}, std={output['factor_value'].std():.4f}")
print(f"偏度: {output['factor_value'].skew():.4f}, 峰度: {output['factor_value'].kurtosis():.4f}")
