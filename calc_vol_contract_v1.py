"""
波动率收缩因子 v1 (Volatility Contraction)
short: std(ret 5d) / std(ret 20d) → 取对数 → 成交额中性化 → MAD → z-score
高因子值 = 短期波动率相对于长期收缩 = 市场稳定性恢复
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

def main():
    data_path = sys.argv[1]  # csi1000_kline_raw.csv
    output_path = sys.argv[2]  # output CSV
    print(f"[INFO] 加载数据: {data_path}")
    
    df = pd.read_csv(data_path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    # 计算日收益率
    df["ret"] = df.groupby("stock_code")["close"].pct_change()
    
    # 滚动标准差 5日 / 20日
    df["std_5d"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(5, min_periods=3).std()
    )
    df["std_20d"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    
    # 20日平均成交额 (用于中性化)
    df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
    )
    
    # 因子原始值: -log(std_5d / std_20d)
    # 高值 = 短期波动率收缩 (std_5d < std_20d → log < 0 → -log > 0)
    df["raw_factor"] = -np.log(df["std_5d"] / df["std_20d"]).replace([-np.inf, np.inf], np.nan)
    
    # MAD winsorize
    def mad_winsorize(x):
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return x
        scaled_mad = 1.4826 * mad
        return x.clip(med - 3.0 * scaled_mad, med + 3.0 * scaled_mad)
    
    df["raw_factor"] = df.groupby("date")["raw_factor"].transform(mad_winsorize)
    
    # 成交额OLS中性化 (截面回归残差)
    from numpy.linalg import lstsq
    
    def neutralize_cs(group, factor_col, neutralizer_col):
        y = group[factor_col].values.astype(float)
        x_col = group[neutralizer_col].values.astype(float)
        valid = np.isfinite(y) & np.isfinite(x_col)
        if valid.sum() < 5:
            return pd.Series(np.nan, index=group.index)
        
        y_c = y[valid]
        x_c = x_col[valid].reshape(-1, 1)
        X = np.column_stack([np.ones(len(x_c)), x_c])
        
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y_c, rcond=None)
            residual_full = np.full(len(y), np.nan)
            residual_full[valid] = y_c - X @ beta
        except:
            return pd.Series(np.nan, index=group.index)
        
        return pd.Series(residual_full, index=group.index)
    
    df["neutralized"] = df.groupby("date", group_keys=False).apply(
        lambda g: neutralize_cs(g, "raw_factor", "log_amount_20d")
    )
    
    # 二次MAD winsorize
    df["neutralized"] = df.groupby("date")["neutralized"].transform(mad_winsorize)
    
    # 截面z-score
    def zscore_cs(x):
        valid = x.dropna()
        if len(valid) < 3 or valid.std() == 0:
            return x
        return (x - valid.mean()) / valid.std()
    
    df["factor_value"] = df.groupby("date")["neutralized"].transform(zscore_cs)
    
    # 输出
    out = df[["date", "stock_code", "factor_value"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"[DONE] 因子值已保存: {output_path}")
    print(f"[STATS] 有效值: {out['factor_value'].notna().sum()} / {len(out)}")
    print(f"[STATS] 均值: {out['factor_value'].mean():.4f}, std: {out['factor_value'].std():.4f}")
    print(f"[STATS] 最小: {out['factor_value'].min():.4f}, 最大: {out['factor_value'].max():.4f}")
    print(f"[STATS] 偏度: {out['factor_value'].skew():.4f}")

if __name__ == "__main__":
    main()
