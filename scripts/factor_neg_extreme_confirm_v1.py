#!/usr/bin/env python3
"""
因子: 反转强度×成交确认 (neg_extreme_confirm_v1)
逻辑: 近期有极端下跌(>=2%)的天数 × 下跌日放量/上涨日缩量比率(卖压已耗尽)
Barra: Reversal (短期反转,爆量已消化)
方向: 正向(高因子值=高预期反弹收益)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compute_factor(kline_path: str, output_path: str, 
                   extreme_threshold: float = -0.02,
                   lookback_extreme: int = 10,
                   lookback_vol_ratio: int = 20) -> None:
    """
    计算反转强度×成交确认因子
    
    Parameters
    ----------
    kline_path : str
        日线CSV路径，含 date,stock_code,open,close,high,low,volume,amount,turnover
    output_path : str
        输出因子CSV路径 (date, stock_code, factor_value)
    extreme_threshold : float
        极端下跌阈值（收益率下限），默认 -0.02 = -2%
    lookback_extreme : int
        极端下跌天数统计窗口，默认 10日
    lookback_vol_ratio : int
        涨跌成交量比窗口，默认 20日
    """
    print(f"[因子计算] 读取数据: {kline_path}")
    df = pd.read_csv(kline_path)
    
    # 基本字段
    df['date'] = pd.to_datetime(df['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # === 计算日收益率 ===
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # === 信号1: 极端下跌天数（lookback_extreme日内ret<=extreme_threshold的天数）===
    df['is_extreme_down'] = (df['ret'] <= extreme_threshold).astype(float)
    df['extreme_down_count'] = df.groupby('stock_code')['is_extreme_down'].transform(
        lambda x: x.rolling(lookback_extreme, min_periods=5).sum()
    )
    
    # === 信号2: 跌日放量/涨日缩量 ratio ===
    # 上涨日平均成交量 / 下跌日平均成交量
    df['is_up'] = (df['ret'] > 0).astype(float)
    df['is_down'] = (df['ret'] < 0).astype(float)
    
    # 纯滚动均值（不区分涨跌日的简单移动平均版本）
    # 涨日均量：用groupby + apply计算滚动窗口
    def _rolling_up_vol_ratio(group):
        ret = group['ret'].values
        vol = group['volume'].values
        n = len(ret)
        result = np.full(n, np.nan)
        
        for i in range(lookback_vol_ratio - 1, n):
            start = max(0, i - lookback_vol_ratio + 1)
            window_ret = ret[start:i+1]
            window_vol = vol[start:i+1]
            up_mask = window_ret > 0
            down_mask = window_ret < 0
            up_vol = window_vol[up_mask].mean() if up_mask.sum() > 0 else np.nan
            down_vol = window_vol[down_mask].mean() if down_mask.sum() > 0 else np.nan
            if not np.isnan(up_vol) and not np.isnan(down_vol) and down_vol > 0:
                result[i] = up_vol / down_vol
            else:
                result[i] = np.nan
        return pd.Series(result, index=group.index)
    
    print(f"[因子计算] 计算涨跌成交量比 (lookback={lookback_vol_ratio}日)...")
    df['vol_ratio'] = df.groupby('stock_code', group_keys=False).apply(_rolling_up_vol_ratio)
    
    # === 因子合成 ===
    # 将两个信号归一化到[0,1]或常规z-score空间后相乘
    # 方法：极端下跌天数归一化(除以窗口长度) → [0,1]
    # vol_ratio直接使用（>1=涨日放量/跌日缩量；<1=涨日缩量/跌日放量）
    # 正逻辑：跌日放量+涨日缩量 → ratio < 1 → 反转因子值要高 → 取 (1 - ratio)
    df['reversal_strength'] = df['extreme_down_count'] / lookback_extreme  # [0,1]
    df['volume_confirm'] = 1 - df['vol_ratio']  # high=跌日放量确认
    
    df['factor_raw'] = df['reversal_strength'] * df['volume_confirm']
    
    # === 成交额OLS中性化 ===
    # 构建对数成交额代理
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=5).mean() + 1)
    )
    
    print("[因子计算] 截面OLS中性化(对log_amount_20d)...")
    
    from numpy.linalg import lstsq
    
    neutralized = []
    for dt, grp in df.groupby('date'):
        mask = grp['factor_raw'].notna() & grp['log_amount_20d'].notna()
        sub = grp[mask].copy()
        if len(sub) < 30:
            continue
        
        y = sub['factor_raw'].values
        x = sub['log_amount_20d'].values
        # OLS: y = a + b*x + residual
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = lstsq(X, y, rcond=None)
        residual = y - X @ beta
        
        # MAD winsorize (3σ)
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            continue
        upper = med + 3 * 1.4826 * mad
        lower = med - 3 * 1.4826 * mad
        residual = np.clip(residual, lower, upper)
        
        # z-score
        std = residual.std()
        if std < 1e-10:
            continue
        z = (residual - residual.mean()) / std
        
        tmp = sub[['date', 'stock_code']].copy()
        tmp['factor_value'] = z
        neutralized.append(tmp)
    
    result = pd.concat(neutralized, ignore_index=True)
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    
    # 统计信息
    n_dates = result['date'].nunique()
    n_stocks = result['stock_code'].nunique()
    valid = result.dropna(subset=['factor_value'])
    print(f"[因子计算] 完成！")
    print(f"  日期数: {n_dates}  | 股票数: {n_stocks}")
    print(f"  有效观测: {len(valid)}  | 缺失: {len(result) - len(valid)}")
    print(f"  因子均值: {valid['factor_value'].mean():.4f}  std: {valid['factor_value'].std():.4f}")
    print(f"  输出路径: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_neg_extreme_confirm_v1.csv')
    parser.add_argument('--threshold', type=float, default=-0.02, help='极端下跌阈值(default=-0.02)')
    parser.add_argument('--lookback-extreme', type=int, default=10)
    parser.add_argument('--lookback-vol', type=int, default=20)
    args = parser.parse_args()
    
    compute_factor(
        kline_path=args.kline,
        output_path=args.output,
        extreme_threshold=args.threshold,
        lookback_extreme=args.lookback_extreme,
        lookback_vol_ratio=args.lookback_vol,
    )
