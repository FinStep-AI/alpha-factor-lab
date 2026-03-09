#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Price-to-52-Week-High (PTH) Factor

复现论文:
- George & Hwang (2004) "The 52-Week High and Momentum Investing" 
  Journal of Finance, 59(5), 2145-2176.
- George, Hwang & Li (2018) "The 52-Week High, Q Theory, and the 
  Cross Section of Stock Returns" JFE, 128(1), 148-163.

因子定义:
  PTH = Close_t / max(Close_{t-249}, ..., Close_t)
  
  即: 当前收盘价 / 过去250个交易日最高收盘价
  
  PTH ∈ (0, 1]
  PTH = 1 → 创52周新高
  PTH 接近0 → 远离52周高点

经济学逻辑:
  1. 锚定效应(Anchoring): 交易者用52周高点作为心理锚点
  2. 接近高点时，信息虽然利好但投资者犹豫不决(under-reaction)
  3. 最终价格调整到位 → 高PTH股票后续跑赢低PTH股票
  4. George et al. (2018): PTH还包含未来增长预期信息(q-theory)

本土化:
  - 股票池: 中证1000
  - 250交易日回望(约1年)
  - 市值中性化
  - 正向使用: 高PTH → 做多
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_pth(kline_path: str, output_path: str, lookback: int = 250):
    """
    计算Price-to-52-Week-High因子。
    
    Parameters
    ----------
    kline_path : str
        K线数据路径
    output_path : str
        输出因子CSV路径
    lookback : int
        回望期天数 (默认250≈52周)
    """
    print(f"{'='*60}")
    print(f"Price-to-52-Week-High Factor (lookback={lookback})")
    print(f"George & Hwang (2004 JF) + George, Hwang & Li (2018 JFE)")
    print(f"{'='*60}")
    
    # 1. Load
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    print(f"    日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # 2. Compute rolling 52-week high and PTH
    print(f"\n[2] 计算{lookback}日滚动最高价和PTH...")
    df['high_52w'] = df.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(lookback, min_periods=lookback // 2).max()
    )
    
    df['pth'] = df['close'] / df['high_52w']
    df['pth'] = df['pth'].replace([np.inf, -np.inf], np.nan)
    
    # Check basic stats
    valid_pth = df['pth'].dropna()
    print(f"    PTH范围: [{valid_pth.min():.4f}, {valid_pth.max():.4f}]")
    print(f"    PTH均值: {valid_pth.mean():.4f}")
    print(f"    PTH中位数: {valid_pth.median():.4f}")
    
    # 3. Market cap neutralization
    print("\n[3] 市值中性化 (log_amount_20d)...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize(group):
        y = group['pth']
        x = group['log_amount_20d']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c = y[mask].values
        x_c = x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            residuals = y_c - X @ beta
            result = pd.Series(np.nan, index=group.index)
            result[mask] = residuals
            return result
        except Exception:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    # 4. Winsorize (MAD 3x)
    print("[4] Winsorize (MAD 3x)...")
    def winsorize_mad(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10:
            return v
        c = v[m]
        med = c.median()
        mad = np.median(np.abs(c - med))
        if mad < 1e-10:
            return v
        lower = med - 3 * 1.4826 * mad
        upper = med + 3 * 1.4826 * mad
        return v.clip(lower, upper)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # 5. Z-score
    print("[5] 截面z-score标准化...")
    def zscore_cs(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10:
            return v
        mu = v[m].mean()
        s = v[m].std()
        if s < 1e-10:
            return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore_cs)
    
    # 6. Output
    print("[6] 输出因子值...")
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    n_valid = output.shape[0]
    n_dates_valid = output['date'].nunique()
    avg_stocks = output.groupby('date')['stock_code'].nunique().median()
    
    print(f"\n✅ PTH因子计算完成!")
    print(f"   输出: {output_path}")
    print(f"   有效记录: {n_valid:,}")
    print(f"   有效日期: {n_dates_valid}")
    print(f"   平均每日股票数: {avg_stocks:.0f}")
    
    # Stats
    desc = output['factor_value'].describe()
    print(f"   均值: {desc['mean']:.4f}, 标准差: {desc['std']:.4f}")
    print(f"   范围: [{desc['min']:.4f}, {desc['max']:.4f}]")
    
    return output


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_pth_52w_v1.csv"
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    
    compute_pth(kline_path, output_path, lookback)
