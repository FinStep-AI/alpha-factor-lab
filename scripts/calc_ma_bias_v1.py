"""
因子: 均线乖离率 (MA Bias, ma_bias_v1)
构造: (close - MA20) / MA20, 成交额中性化
       正向: 做多价格高于均线(动量确认)
       反向: 做多价格低于均线(反转/超卖)
逻辑: 均线乖离是经典技术指标。在截面上:
       - 正向使用: 强势股票(价格>MA20)动量延续
       - 反向使用: 弱势股票(价格<MA20)均值回复
       与5d/10d reversal不同, 这里以20日均线为锚点
Barra风格: Momentum / Reversal
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", parse_dates=["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"数据: {kline.shape[0]} 行, {kline['stock_code'].nunique()} 只股票")

# ── 多个均线窗口 ────────────────────────────────────────
for ma_win in [10, 20, 60]:
    print(f"\n计算 MA{ma_win} 乖离率...")
    kline[f"ma{ma_win}"] = kline.groupby("stock_code")["close"].transform(
        lambda x: x.rolling(ma_win, min_periods=max(5, ma_win//2)).mean()
    )
    kline[f"bias_{ma_win}_raw"] = (kline["close"] - kline[f"ma{ma_win}"]) / kline[f"ma{ma_win}"]

# ── 成交额中性化 (20日) ──────────────────────────────────
kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

for ma_win in [10, 20, 60]:
    col_raw = f"bias_{ma_win}_raw"
    col_out = f"ma_bias_{ma_win}"
    
    def neutralize_ols(df, col=col_raw):
        mask = df[col].notna() & df["log_amount_20d"].notna() & np.isfinite(df[col])
        if mask.sum() < 30:
            df["factor"] = np.nan
            return df
        y = df.loc[mask, col].values
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
    
    print(f"  中性化 bias_{ma_win}...")
    kline = kline.groupby("date", group_keys=False).apply(lambda df: neutralize_ols(df, col_raw))
    
    # MAD + z-score
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
        lower = med - 3 * mad
        upper = med + 3 * mad
        clipped = vals.clip(lower, upper)
        mu = clipped[mask].mean()
        sigma = clipped[mask].std()
        if sigma < 1e-10:
            df["factor"] = np.nan
            return df
        df["factor"] = (clipped - mu) / sigma
        return df
    
    kline = kline.groupby("date", group_keys=False).apply(mad_zscore)
    
    # 输出正向(动量)和反向(反转)
    output = kline[["date", "stock_code", "factor"]].dropna(subset=["factor"])
    
    # 正向 (高bias=高因子值=做多动量)
    output_pos = output.copy()
    output_pos = output_pos.rename(columns={"factor": col_out})
    out_path = DATA_DIR / f"factor_{col_out}.csv"
    output_pos.to_csv(out_path, index=False)
    print(f"  → {out_path}, {len(output_pos)} 行")
    
    # 反向 (取负: 高因子值=超卖=做多反转)
    output_neg = output.copy()
    output_neg["factor"] = -output_neg["factor"]
    output_neg = output_neg.rename(columns={"factor": f"{col_out}_rev"})
    out_path_rev = DATA_DIR / f"factor_{col_out}_rev.csv"
    output_neg.to_csv(out_path_rev, index=False)
    print(f"  → {out_path_rev}")

print("\n完成！")
