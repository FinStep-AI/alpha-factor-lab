"""
收益率自相关因子 (Return Autocorrelation v1)
============================================
逻辑：20日滚动收益率一阶自相关系数。
- 高自相关 → 价格趋势持续(动量)，但也意味着定价效率低
- 低/负自相关 → 价格均值回复，定价效率高
文献：Heston, Sadka & Ronnie (2010) 发现低自相关股票后续表现更好
A股小盘可能相反（动量持续性=知情交易，正向因子）

做法：
1. 计算每只股票20日滚动收益率一阶自相关 autocorr(ret, 20)
2. 市值中性化
3. 输出因子CSV

方向先设为正向（猜测A股小盘高自相关=趋势延续=正收益），
回测后根据结果调整。
"""

import pandas as pd
import numpy as np
import sys
import os

def compute_return_autocorrelation(kline_path, output_path, window=20, min_periods=15):
    """计算收益率自相关因子"""
    print(f"[1/4] 读取数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    
    # 计算日收益率
    df = df.sort_values(['stock_code', 'date'])
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    print(f"[2/4] 计算{window}日滚动自相关系数...")
    
    def rolling_autocorr(group, window, min_periods):
        """计算滚动一阶自相关"""
        ret = group['ret']
        # 使用pandas rolling + apply
        autocorr_values = ret.rolling(window=window, min_periods=min_periods).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= min_periods else np.nan,
            raw=True
        )
        return autocorr_values
    
    results = []
    stocks = df['stock_code'].unique()
    total = len(stocks)
    for i, stock in enumerate(stocks):
        if (i+1) % 200 == 0:
            print(f"  处理: {i+1}/{total}")
        stock_data = df[df['stock_code'] == stock].copy()
        stock_data['factor_raw'] = rolling_autocorr(stock_data, window, min_periods)
        results.append(stock_data[['date', 'stock_code', 'factor_raw']].dropna())
    
    factor_df = pd.concat(results, ignore_index=True)
    
    print(f"[3/4] 市值中性化...")
    # 用log(amount)作为市值代理（没有市值数据时）
    # 合并原始数据获取amount
    amount_df = df[['date', 'stock_code', 'amount']].copy()
    factor_df = factor_df.merge(amount_df, on=['date', 'stock_code'], how='left')
    factor_df['log_amount'] = np.log(factor_df['amount'].clip(lower=1))
    
    # 截面回归中性化
    def neutralize_cross_section(group):
        y = group['factor_raw']
        x = group['log_amount']
        valid = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor_value'] = np.nan
            return group
        y_v, x_v = y[valid], x[valid]
        # OLS residual
        x_dm = x_v - x_v.mean()
        beta = (x_dm * y_v).sum() / (x_dm ** 2).sum() if (x_dm ** 2).sum() > 0 else 0
        alpha = y_v.mean() - beta * x_v.mean()
        residuals = pd.Series(np.nan, index=group.index)
        residuals[valid] = y_v - (alpha + beta * x_v)
        # 标准化
        std = residuals.std()
        if std > 0:
            residuals = residuals / std
        group['factor_value'] = residuals
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor_value']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    
    print(f"[4/4] 输出因子: {output_path}")
    print(f"  总行数: {len(output)}, 日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  因子统计: mean={output['factor_value'].mean():.4f}, std={output['factor_value'].std():.4f}")
    
    output.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    kline_path = 'alpha-factor-lab/data/csi1000_kline_raw.csv'
    output_path = 'alpha-factor-lab/data/factor_ret_autocorr_v1.csv'
    compute_return_autocorrelation(kline_path, output_path)
