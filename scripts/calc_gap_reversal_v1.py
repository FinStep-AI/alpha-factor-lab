#!/usr/bin/env python3
"""计算 Intraday Gap Reversal 因子。

因子定义：20日滚动隔夜缺口与日内收益的相关性的负值。
- overnight_gap = (open - close_lag_1) / close_lag_1
- intraday_ret = (close - open) / open
- gap_reversal_strength = -corr(overnight_gap, intraday_ret, 20日)

正向预期：高 gap_reversal = 隔夜被comment回调多 = 隔夜过度反应 → 后续回吐更多
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "skills/alpha-factor-lab/scripts")
from factor_calculator import neutralize_cross_section, log_transform

def main():
    # Load OHLCV data
    print("[1/4] 加载数据...")
    df = pd.read_csv("data/csi1000_kline_raw.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    grouped = df.groupby("stock_code")
    
    # 计算 intraday return 和 overnight gap
    print("[2/4] 计算日内收益和隔夜跳空...")
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    
    # overnight_gap: today's open vs yesterday's close
    df["prev_close"] = grouped["close"].shift(1)
    df["overnight_gap"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    
    # 20日滚动相关: corr(overnight_gap, intraday_ret)
    print("[3/4] 计算20日滚动相关（隔夜gap vs 日内收益）...")
    def rolling_corr(group, col1, col2, window):
        return group[[col1, col2]].rolling(window, min_periods=15).corr().unstack().iloc[:, 0]
    
    df["gap_intraday_corr"] = grouped.apply(
        lambda g: rolling_corr(g, "overnight_gap", "intraday_ret", 20)
    ).reset_index(level=0, drop=True)
    
    # 因子 = -correlation (反转信号)
    df["factor_raw"] = -df["gap_intraday_corr"]
    
    # 市值中性化：使用 amount 20日均值作为市值代理
    print("[3.5/4] 市值中性化...")
    df["amount_20d"] = grouped["amount"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    df["log_amount_20d"] = np.log(df["amount_20d"].replace(0, np.nan))
    
    df["factor_neutral"] = neutralize_cross_section(
        df, "factor_raw", neutralize_cols=["log_amount_20d"]
    )
    
    # MAD winsorize + z-score
    from factor_calculator import winsorize_mad, zscore_cross_section
    print("[4/4] 标准化 & 输出...")
    df["factor_value"] = df["factor_neutral"].clip(lower=-3, upper=3)
    df["factor_value"] = df.groupby("date")["factor_value"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
    )
    
    # Output
    out = df[["date", "stock_code", "factor_value"]].dropna()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv("data/factor_gap_reversal_v1.csv", index=False)
    print(f"Done! {len(out)} rows. Output: data/factor_gap_reversal_v1.csv")
    print(f"Factor stats: mean={out['factor_value'].mean():.4f} std={out['factor_value'].std():.4f}")

if __name__ == "__main__":
    main()
