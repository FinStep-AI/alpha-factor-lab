#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量价加权短期反转因子 (Volume-Weighted Short-Term Reversal v1)
==============================================================

核心思路:
  经典短期反转 = 做多近期输家, 做空近期赢家
  改进: 用成交量变化加权，放大"放量异动"的反转信号

  对过去5天的每日收益率，乘以当天的异常成交量比率:
    vol_ratio_t = volume_t / mean(volume, 20d)
    weighted_ret = sum(ret_t * vol_ratio_t, t-5 to t-1) / sum(vol_ratio_t)
    
  factor = -weighted_ret  (反向: 做多近期放量下跌股，做空近期放量上涨股)
  
  经济学直觉:
  - 放量上涨 = 过度追捧 → 反转概率更高
  - 放量下跌 = 恐慌抛售 → 反转概率更高
  - 成交量放大了价格过度反应的信号
  
  在中证1000中:
  - 散户主导，情绪驱动强 → 短期反转效应显著
  - 放量异动后的反转更为可靠(信息含量高)

文献参考:
  - Jegadeesh (1990) "Evidence of Predictable Behavior of Security Returns" JF
  - Lehmann (1990) "Fads, Martingales, and Market Efficiency" QJE
  - Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium" JF

处理:
  1. 计算5日量价加权收益
  2. 取反(做多近期弱势)
  3. 5%缩尾
  4. OLS log_amount_20d 中性化
  5. 截面z-score标准化
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_vol_weighted_reversal(kline_path: str, output_path: str, 
                                   short_window: int = 5, vol_window: int = 20):
    print(f"{'='*60}")
    print(f"Volume-Weighted Short-Term Reversal (short={short_window}, vol_ref={vol_window})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Daily returns and volume ratio
    print("[2] 计算收益率和异常成交量比率...")
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df['vol_ma'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(vol_window, min_periods=int(vol_window*0.75)).mean()
    )
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-8)
    # Clip extreme volume ratios
    df['vol_ratio'] = df['vol_ratio'].clip(0.1, 10)
    
    # 3. Volume-weighted short-term return
    print(f"[3] 计算{short_window}日量价加权收益...")
    def calc_vw_ret(group):
        group = group.sort_values('date').copy()
        ret = group['ret'].values
        vol_r = group['vol_ratio'].values
        n = len(ret)
        vw_ret = np.full(n, np.nan)
        
        for i in range(short_window, n):
            r = ret[i-short_window:i]
            vr = vol_r[i-short_window:i]
            valid = ~np.isnan(r) & ~np.isnan(vr) & (vr > 0)
            if valid.sum() < short_window * 0.6:
                continue
            vw_ret[i] = np.sum(r[valid] * vr[valid]) / np.sum(vr[valid])
        
        group['vw_ret'] = vw_ret
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_vw_ret)
    
    # 4. Negate for reversal (做多近期弱势)
    df['vw_ret'] = -df['vw_ret']
    
    # 5. log_amount for neutralization
    print("[4] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(vol_window, min_periods=int(vol_window*0.75)).mean() + 1)
    )
    
    # 6. Winsorize 5%
    print("[5] 5%缩尾...")
    def winsorize_cs(group):
        v = group['vw_ret']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['vw_ret'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 7. OLS neutralization
    print("[6] OLS市值中性化...")
    def neutralize(group):
        y = group['vw_ret']
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
    
    # 8. Cross-sectional z-score
    print("[7] 截面z-score标准化...")
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
    
    # 9. Output
    print("[8] 输出因子值...")
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    n_valid = output.shape[0]
    n_dates_valid = output['date'].nunique()
    avg_stocks = output.groupby('date')['stock_code'].nunique().median()
    
    print(f"\n✅ 因子计算完成!")
    print(f"   输出: {output_path}")
    print(f"   有效记录: {n_valid:,}")
    print(f"   有效日期: {n_dates_valid}")
    print(f"   平均每日股票数: {avg_stocks:.0f}")
    
    desc = output['factor_value'].describe()
    print(f"\n   均值: {desc['mean']:.4f}")
    print(f"   标准差: {desc['std']:.4f}")
    print(f"   范围: [{desc['min']:.4f}, {desc['max']:.4f}]")


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_vol_weighted_reversal_v1.csv"
    
    compute_vol_weighted_reversal(kline_path, output_path)
