#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVaR (Expected Shortfall) 因子 - 尾部风险回避 (tail_risk_cvar_v1)

灵感来源:
- Bali, Cakici & Whitelaw (2011) "Maxing Out: Stocks as Lotteries" JFE
  → MAX/MIN effect: 极端收益率预测未来截面收益
- Ang, Chen & Xing (2006) "Downside Risk" RFS
  → 下行风险要求更高补偿
- CVaR/Expected Shortfall 是Basel III推荐的尾部风险度量

因子定义:
  raw = -mean(bottom_2(daily_returns, past 10 days))
  
  即: 过去10个交易日中，最差2天收益率的平均值，取负值。
  
  做多方向: 高因子值 = 近期未经历极端下跌（尾部风险低）
  做空方向: 低因子值 = 近期有极端下跌（高尾部风险）

经济学直觉:
  A股中证1000（小盘股）中，近期经历极端下跌的股票:
  1. 可能面临基本面恶化/退市风险 → 继续走差
  2. 流动性危机引发连锁抛售 → 短期内持续承压
  3. 机构风控止损 → 卖压持续
  而未经历极端下跌的股票处于"安全区"，后续表现更稳健。
  
  这与"动量"在尾部的表现一致：极端负收益有惯性。

处理步骤:
  1. 计算日收益率
  2. 10日窗口取最差2天平均值，取负
  3. OLS市值(log_amount_20d)中性化
  4. MAD 3x Winsorize
  5. 截面z-score标准化

回测结果 (CSI1000, 2022.10-2026.03, 5日前瞻):
  IC均值: 0.0202 (t=2.30)
  多空Sharpe: 2.04
  多空年化: 28.92%
  单调性: 1.0 (完美!)
  G5年化: 25.51%, G1年化: -1.72%
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_tail_risk_cvar(kline_path: str, output_path: str, 
                            window: int = 10, k_worst: int = 2):
    """
    计算CVaR/Expected Shortfall尾部风险因子。
    
    Parameters
    ----------
    kline_path : str
        K线数据路径 (date, stock_code, open, close, high, low, volume, amount)
    output_path : str
        输出因子CSV路径
    window : int
        滚动窗口天数 (默认10)
    k_worst : int
        取最差k天的平均值 (默认2)
    """
    print(f"{'='*60}")
    print(f"Tail Risk CVaR Factor (window={window}, k_worst={k_worst})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Daily returns
    print("[2] 计算日收益率...")
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 3. Rolling CVaR: average of worst k returns in window
    print(f"[3] 计算{window}日滚动CVaR (底部{k_worst}天均值)...")
    
    def calc_cvar(group):
        ret = group['ret']
        def bottom_k_avg(x):
            if len(x) < window // 2:
                return np.nan
            sorted_x = np.sort(x)[:k_worst]
            return sorted_x.mean()
        return ret.rolling(window, min_periods=window // 2).apply(
            bottom_k_avg, raw=True
        )
    
    df['raw_cvar'] = df.groupby('stock_code', group_keys=False).apply(calc_cvar)
    
    # Negate: low CVaR (extreme drops) → low factor value
    df['raw_cvar'] = -df['raw_cvar']
    
    # 4. Market cap neutralization
    print("[4] 市值中性化 (log_amount_20d)...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize(group):
        y = group['raw_cvar']
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
    
    # 5. Winsorize (MAD 3x)
    print("[5] Winsorize (MAD 3x)...")
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
    
    # 6. Cross-sectional z-score
    print("[6] 截面z-score标准化...")
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
    
    # 7. Output
    print("[7] 输出因子值...")
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
    
    # Basic stats
    desc = output['factor_value'].describe()
    print(f"\n   均值: {desc['mean']:.4f}")
    print(f"   标准差: {desc['std']:.4f}")
    print(f"   范围: [{desc['min']:.4f}, {desc['max']:.4f}]")
    
    return output


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_tail_risk_cvar_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    k_worst = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    
    compute_tail_risk_cvar(kline_path, output_path, window, k_worst)
