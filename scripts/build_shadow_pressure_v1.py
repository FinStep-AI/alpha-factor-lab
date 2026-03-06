"""
上影线卖压因子 (Shadow Pressure v1)
====================================
逻辑：上影线长度相对于K线实体的比率，衡量日内卖压。
- 上影线长 → 盘中冲高但收低，卖压重
- 下影线长 → 盘中下探但收高，买盘托底

因子 = 20日均值( (high - max(open,close)) / (high - low + 1e-8) )
     - 20日均值( (min(open,close) - low) / (high - low + 1e-8) )
上影线占比越大，因子值越正 → 卖压重
下影线占比越大，因子值越负 → 买压强

方向：反向（高卖压→低收益），做多低卖压股票。
市值中性化。

学术支持：
- Blau & Whitby (2015) 上影线反映知情卖出
- A股日内T+0交易限制下，集合竞价和尾盘撮合含信息量更大
"""

import pandas as pd
import numpy as np
import sys

def compute_shadow_pressure(kline_path, output_path, window=20, min_periods=15):
    """计算上影线卖压因子"""
    print(f"[1/4] 读取数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    # K线范围
    df['range'] = df['high'] - df['low']
    df['body_top'] = df[['open', 'close']].max(axis=1)
    df['body_bot'] = df[['open', 'close']].min(axis=1)
    
    # 上影线比率 和 下影线比率
    df['upper_shadow_ratio'] = (df['high'] - df['body_top']) / (df['range'] + 1e-8)
    df['lower_shadow_ratio'] = (df['body_bot'] - df['low']) / (df['range'] + 1e-8)
    
    # 卖压指标 = 上影线比率 - 下影线比率
    df['shadow_pressure'] = df['upper_shadow_ratio'] - df['lower_shadow_ratio']
    
    print(f"[2/4] 计算{window}日滚动均值...")
    
    results = []
    stocks = df['stock_code'].unique()
    total = len(stocks)
    for i, stock in enumerate(stocks):
        if (i+1) % 200 == 0:
            print(f"  处理: {i+1}/{total}")
        stock_data = df[df['stock_code'] == stock].copy()
        stock_data['factor_raw'] = stock_data['shadow_pressure'].rolling(
            window=window, min_periods=min_periods
        ).mean()
        results.append(stock_data[['date', 'stock_code', 'factor_raw', 'amount']].dropna(subset=['factor_raw']))
    
    factor_df = pd.concat(results, ignore_index=True)
    
    print(f"[3/4] 市值中性化...")
    factor_df['log_amount'] = np.log(factor_df['amount'].clip(lower=1))
    
    def neutralize_cross_section(group):
        y = group['factor_raw']
        x = group['log_amount']
        valid = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor_value'] = np.nan
            return group
        y_v, x_v = y[valid], x[valid]
        x_dm = x_v - x_v.mean()
        beta = (x_dm * y_v).sum() / (x_dm ** 2).sum() if (x_dm ** 2).sum() > 0 else 0
        alpha = y_v.mean() - beta * x_v.mean()
        residuals = pd.Series(np.nan, index=group.index)
        residuals[valid] = y_v - (alpha + beta * x_v)
        std = residuals.std()
        if std > 0:
            residuals = residuals / std
        group['factor_value'] = residuals
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    
    output = factor_df[['date', 'stock_code', 'factor_value']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    
    print(f"[4/4] 输出因子: {output_path}")
    print(f"  总行数: {len(output)}, 日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  因子统计: mean={output['factor_value'].mean():.4f}, std={output['factor_value'].std():.4f}")
    
    output.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    kline_path = 'alpha-factor-lab/data/csi1000_kline_raw.csv'
    output_path = 'alpha-factor-lab/data/factor_shadow_pressure_v1.csv'
    compute_shadow_pressure(kline_path, output_path)
