#!/usr/bin/env python3
"""
流动性创新因子 (Liquidity Innovation / Illiquidity Shock)

学术来源:
- Amihud (2002) "Illiquidity and Stock Returns" JFM
- Pastor & Stambaugh (2003) "Liquidity Risk and Expected Stock Returns" JFE
- Acharya & Pedersen (2005) "Asset Pricing with Liquidity Risk" JFE

构造逻辑:
1. 计算日度Amihud非流动性: ILLIQ_d = |ret_d| / amount_d
2. 短期均值: ILLIQ_short = MA(ILLIQ, 5日)
3. 长期均值: ILLIQ_long = MA(ILLIQ, 60日)
4. 流动性创新: LIQ_INNOV = -(ILLIQ_short / ILLIQ_long - 1)
   负号: 流动性改善(短期ILLIQ下降) → 因子值高 → 预期正alpha
   
原理: 
- 流动性突然改善(短期ILLIQ < 长期ILLIQ)可能意味着:
  a) 新的机构投资者进入(关注度提升)
  b) 信息不对称降低(更多知情交易)
  c) 做市商竞争加剧(流动性供给增加)
- 这些都是正面信号，预示后续正超额收益

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def calc_factor():
    # 加载数据
    kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # 计算日度Amihud ratio: |pct_change| / amount
    # pct_change 是百分比值，转成小数
    kline["ret"] = kline["pct_change"].astype(float) / 100.0
    kline["amihud_d"] = kline["ret"].abs() / kline["amount"]
    
    # 去掉无效值
    kline.loc[kline["amount"] <= 0, "amihud_d"] = np.nan
    
    # 滚动均值
    kline["illiq_short"] = kline.groupby("stock_code")["amihud_d"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    kline["illiq_long"] = kline.groupby("stock_code")["amihud_d"].transform(
        lambda x: x.rolling(60, min_periods=40).mean()
    )
    
    # 流动性创新: -(short/long - 1) = -(short - long) / long
    # 负号: 流动性改善(short < long) → 因子值为正
    kline["raw_factor"] = -(kline["illiq_short"] / kline["illiq_long"] - 1)
    
    # 成交额中性化
    kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    # OLS中性化: factor = alpha + beta * log_amount + residual
    def neutralize_group(df):
        """按日期截面做OLS中性化"""
        results = []
        for date, group in df.groupby("date"):
            sub = group[["stock_code", "raw_factor", "log_amount_20d"]].dropna()
            if len(sub) < 50:
                continue
            x = sub["log_amount_20d"].values
            y = sub["raw_factor"].values
            
            # OLS回归
            x_dm = x - x.mean()
            beta = np.sum(x_dm * (y - y.mean())) / (np.sum(x_dm**2) + 1e-10)
            alpha = y.mean() - beta * x.mean()
            residual = y - (alpha + beta * x)
            
            sub = sub.copy()
            sub["factor_neutral"] = residual
            results.append(sub[["stock_code", "factor_neutral"]].assign(date=date))
        
        return pd.concat(results, ignore_index=True)
    
    neutral = neutralize_group(kline[["date", "stock_code", "raw_factor", "log_amount_20d"]])
    
    # MAD Winsorize + Z-score
    def mad_zscore(df):
        results = []
        for date, group in df.groupby("date"):
            vals = group["factor_neutral"].values.copy()
            median = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - median))
            mad_e = mad * 1.4826  # equivalent to std
            if mad_e < 1e-10:
                continue
            # winsorize at 3 MAD
            upper = median + 3 * mad_e
            lower = median - 3 * mad_e
            vals = np.clip(vals, lower, upper)
            # z-score
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std < 1e-10:
                continue
            vals = (vals - mean) / std
            sub = group.copy()
            sub["factor"] = vals
            results.append(sub[["date", "stock_code", "factor"]])
        return pd.concat(results, ignore_index=True)
    
    final = mad_zscore(neutral)
    
    # 输出为 pivot 格式 (date x stock_code)
    pivot = final.pivot(index="date", columns="stock_code", values="factor")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    
    output_path = DATA_DIR / "factor_liq_innovation_v1.csv"
    pivot.to_csv(output_path)
    print(f"✅ 因子已保存: {output_path}")
    print(f"   日期范围: {pivot.index.min().date()} ~ {pivot.index.max().date()}")
    print(f"   股票数: {pivot.shape[1]}")
    print(f"   有效值比例: {pivot.notna().sum().sum() / pivot.size:.2%}")
    
    # 抽样检查
    sample = pivot.iloc[-1].dropna()
    print(f"\n最新截面统计:")
    print(f"   均值={sample.mean():.4f}, 标准差={sample.std():.4f}")
    print(f"   最小={sample.min():.4f}, 最大={sample.max():.4f}")
    print(f"   有效股票数={len(sample)}")
    
    return pivot


if __name__ == "__main__":
    calc_factor()
