#!/usr/bin/env python3
"""
收益率峰度因子 (Return Kurtosis Factor)

思路：
计算每只股票过去20日日收益率的超额峰度(excess kurtosis)，取负值。
高因子值 = 低峰度 = 收益分布接近正态 = 交易模式稳定 = Quality代理

与已有因子区别：
  idio_vol: 衡量收益率标准差（波动幅度）
  vol_cv_neg: 衡量成交量稳定性
  本因子: 衡量收益率分布形态（尾部厚度），是更高阶的矩

也测试正向（高峰度→高收益，极端事件风险溢价逻辑）
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

LOOKBACK = 20
OUTPUT_PATH = "data/factor_ret_kurtosis_v1.csv"

print("读取K线数据...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 计算日收益率
print("计算日收益率...")
df["ret"] = df.groupby("stock_code")["close"].pct_change()

# 计算滚动峰度
print(f"计算{LOOKBACK}日滚动峰度...")
results = []
for code, grp in df.groupby("stock_code"):
    grp = grp.sort_values("date")
    dates = grp["date"].values
    rets = grp["ret"].values
    
    for i in range(LOOKBACK - 1, len(grp)):
        ret_window = rets[i - LOOKBACK + 1:i + 1]
        valid = ~np.isnan(ret_window)
        if valid.sum() < 15:
            continue
        ret_clean = ret_window[valid]
        # excess kurtosis (Fisher's definition, normal=0)
        kurt = sp_stats.kurtosis(ret_clean, fisher=True)
        if np.isnan(kurt) or np.isinf(kurt):
            continue
        results.append({
            "date": dates[i],
            "stock_code": code,
            "factor_raw": kurt  # 先用正向（高峰度），回测时看方向
        })

factor_df = pd.DataFrame(results)
print(f"原始因子值: {len(factor_df)} 条")

# 计算20日平均成交额（用于中性化）
print("计算20日平均成交额...")
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=15).mean() + 1)
)
amount_map = df[["date", "stock_code", "log_amount_20d"]].copy()
factor_df = factor_df.merge(amount_map, on=["date", "stock_code"], how="left")

# 截面处理
print("截面处理: OLS中性化 + MAD winsorize + z-score...")

def cross_section_process(group):
    vals = group["factor_raw"].values.copy()
    neutralizer = group["log_amount_20d"].values.copy()
    
    valid = ~(np.isnan(vals) | np.isnan(neutralizer))
    if valid.sum() < 50:
        group["factor_value"] = np.nan
        return group
    
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

# 输出
output = factor_df[["date", "stock_code", "factor_value"]].dropna(subset=["factor_value"])
output.to_csv(OUTPUT_PATH, index=False)
print(f"因子输出: {len(output)} 条 → {OUTPUT_PATH}")

# 同时输出反向版本
output_neg = output.copy()
output_neg["factor_value"] = -output_neg["factor_value"]
output_neg.to_csv("data/factor_ret_kurtosis_neg_v1.csv", index=False)
print(f"反向因子输出: {len(output_neg)} 条 → data/factor_ret_kurtosis_neg_v1.csv")

print("\n=== 因子统计 ===")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"截面股票数: {output.groupby('date')['stock_code'].count().mean():.0f} (平均)")
print(f"因子值分布: mean={output['factor_value'].mean():.4f}, std={output['factor_value'].std():.4f}")
print(f"偏度: {output['factor_value'].skew():.4f}, 峰度: {output['factor_value'].kurtosis():.4f}")
