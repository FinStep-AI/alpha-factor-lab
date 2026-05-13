#!/usr/bin/env python3
"""
因子: 趋势日比率因子 v1 (Trend Day Ratio v1)
factor_id: trend_day_ratio_v1

构造逻辑:
  1. 每日趋势效率 = |close - open| / (high - low + eps)
     值域 [0,1]：1 = 完美趋势日（单边行情），0 = 完美震荡日（十字星）
  2. 20日滚动均值 = ts_mean(trend_eff, 20)
  3. 成交额中性化 → MAD缩尾 → z-score

理论依据:
  - 日内趋势效率高 → 信息有序释放，非噪声交易主导
  - 主力资金方向感明确 → 动量延续性强
  - Chordia, Roll & Subrahmanyam (2011): 价格发现效率与定价效率正相关
  - Amihud & Mendelson (1987): 日内价格形态反映市场微观结构质量

假设: 趋势效率均值较高 → 5日预期收益更高（正向）
"""

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ───── 1. 构造原始 column: trend_eff = |close-open|/(high-low+eps) ─────

print("=" * 60)
print("Step 1: 计算趋势效率 & 20日均值趋势日比率")
print("=" * 60)

kline = BASE / "data" / "csi1000_kline_raw.csv"
factor_raw = BASE / "data" / "factor_trend_day_ratio_raw.csv"

code = f"""
import pandas as pd, numpy as np, sys
from pathlib import Path

BASE = Path("{BASE}")
kline = BASE / "data/csi1000_kline_raw.csv"
factor_raw = BASE / "data/factor_trend_day_ratio_raw.csv"

df = pd.read_csv(kline)
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# 趋势效率 |close-open|/(high-low+eps)
df["trend_eff"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 0.01)

# 20日滚动均值（按 stock 分组）
g = df.groupby("stock_code")
df["trend_day_ratio"] = g["trend_eff"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

out = df[["date", "stock_code", "trend_day_ratio"]].dropna()
out.to_csv(factor_raw, index=False)
print(f"Done: {{len(out)}} rows")
"""

result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(BASE))
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])


# ───── 2. 中性化 & 标准化 ─────

print()
print("=" * 60)
print("Step 2: 成交额中性化 + MAD缩尾 + z-score")
print("=" * 60)

kline_file = BASE / "data" / "csi1000_kline_raw.csv"
factor_output = BASE / "data" / "factor_trend_day_ratio_v1.csv"

neutralize_code = f"""
import pandas as pd, numpy as np, sys
from pathlib import Path
from scipy import stats as sp_stats

BASE = Path("{BASE}")
kline = BASE / "data/csi1000_kline_raw.csv"
factor_raw = BASE / "data/factor_trend_day_ratio_raw.csv"
factor_out = BASE / "data/factor_trend_day_ratio_v1.csv"

raw = pd.read_csv(factor_raw, parse_dates=["date"])
kline2 = pd.read_csv(kline, parse_dates=["date"])
# log_amount_20d
g = kline2.sort_values(["stock_code","date"]).groupby("stock_code")
kline2["log_amount_20d"] = g["amount"].transform(
    lambda x: np.log(x.rolling(20,min_periods=10).mean().replace(0, np.nan) + 1)
)
log_amt = kline2[["date","stock_code","log_amount_20d"]].dropna()

df = raw.merge(log_amt, on=["date","stock_code"], how="left").dropna()

# 按截面成交额OLS中性化
from numpy.linalg import lstsq
def neutralize(group):
    y = group["trend_day_ratio"].values
    x = group["log_amount_20d"].values
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 3:
        group["_factor"] = np.nan
        return group
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    try:
        b, _, _, _ = lstsq(X, y[mask], rcond=None)
        resid = np.full(len(group), np.nan)
        resid[mask] = y[mask] - X @ b
    except:
        group["_factor"] = y
        return group
    group["_factor"] = resid
    return group

df = df.groupby("date", group_keys=False).apply(neutralize)

# MAD缩尾 (5 sigma)
def mad_winsorize(x, n_mad=5.0):
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad): return x
    lower = med - n_mad * 1.4826 * mad
    upper = med + n_mad * 1.4826 * mad
    return x.clip(lower, upper)

# 只对有效截面做，然后交叉处理
for grp_date, grp in df.groupby("date"):
    grp2 = grp.dropna(subset=["_factor"])
    if len(grp2) < 5: continue
    med = grp2["_factor"].median()
    mad = (grp2["_factor"] - med).abs().median()
    if mad == 0 or np.isnan(mad): continue
    lower = med - 5*1.4826*mad
    upper = med + 5*1.4826*mad
    df.loc[grp2.index, "_factor"] = grp2["_factor"].clip(lower, upper)

# z-score 横截面
def zscore(group):
    mean = group["_factor"].mean()
    std = group["_factor"].std()
    if std == 0 or np.isnan(std):
        group["factor_value"] = 0
    else:
        group["factor_value"] = (group["_factor"] - mean) / std
    return group

df = df.groupby("date", group_keys=False).apply(zscore)

out = df[["date","stock_code","factor_value"]].dropna()
out.to_csv(factor_out, index=False)
print(f"Factor row count during production: {{len(out)}}")
print(f"Factor written to: {{factor_out}}")
"""

result2 = subprocess.run([sys.executable, "-c", neutralize_code], capture_output=True, text=True, cwd=str(BASE))
print(result2.stdout[-500:] if result2.stdout else "")
if result2.stderr:
    print("STDERR:", result2.stderr[-500:])

print()
result3 = subprocess.run([sys.executable, "-c", f"import pandas as pd; df = pd.read_csv('{BASE}/data/factor_trend_day_ratio_v1.csv'); print(df.shape); print(df.head(2))"], capture_output=True, text=True, cwd=str(BASE))
print(result3.stdout)
if result3.stderr:
    print("STDERR:", result3.stderr[-300:])
