"""
因子：开盘定价一致率 (Opening Price Consistency, OPC)
ID: open_consistency_v1

逻辑：
  每天判断开盘跳空方向(open/prev_close-1)与日内走势方向(close/open-1)是否一致。
  20日滚动计算一致率。
  高一致率 = 开盘定价正确率高 = 价格发现效率高 = Quality代理
  低一致率 = 开盘频繁被反转 = 噪声交易多

  补充维度：加权版本，用|gap|加权（大幅跳空的一致性更重要）

方向：正向（高一致率 = 高预期收益）
中性化：成交额OLS中性化
"""

import pandas as pd
import numpy as np
import sys

def calc_open_consistency(kline_path, output_path, window=20):
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算前收盘价
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    
    # 开盘跳空
    df['gap'] = df['open'] / df['prev_close'] - 1
    
    # 日内收益
    df['intraday'] = df['close'] / df['open'] - 1
    
    # 方向一致性：sign(gap) == sign(intraday)，gap=0视为不一致
    df['consistent'] = ((df['gap'] > 0) & (df['intraday'] > 0)) | \
                       ((df['gap'] < 0) & (df['intraday'] < 0))
    df['consistent'] = df['consistent'].astype(float)
    
    # 对gap=0或intraday=0的情况设为NaN
    df.loc[(df['gap'].abs() < 1e-8) | (df['intraday'].abs() < 1e-8), 'consistent'] = np.nan
    
    print("Calculating rolling consistency rate...")
    # 20日滚动一致率
    df['raw_factor'] = df.groupby('stock_code')['consistent'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.7)).mean()
    )
    
    # 准备成交额中性化
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    # 截面中性化 + MAD + z-score
    print("Cross-section neutralization...")
    
    factor_df = df[['date', 'stock_code', 'raw_factor', 'log_amount_20d']].dropna().copy()
    
    def process_date(group):
        y = group['raw_factor'].values
        x = group['log_amount_20d'].values
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        # OLS neutralize
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            residuals[valid] = y[valid] - X @ beta
        except:
            group['factor'] = np.nan
            return group
        
        # MAD winsorize
        vals = residuals[valid]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad < 1e-8:
            group['factor'] = 0.0
            return group
        
        bound = 5 * 1.4826 * mad
        residuals = np.clip(residuals, med - bound, med + bound)
        
        # z-score
        vals_valid = residuals[valid]
        mean = np.mean(vals_valid)
        std = np.std(vals_valid)
        if std < 1e-8:
            group['factor'] = 0.0
            return group
        
        group['factor'] = (residuals - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(process_date)
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output = output.rename(columns={'factor': 'factor_value'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"Factor saved to {output_path}")
    print(f"  Rows: {len(output)}")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  Stocks per date (mean): {output.groupby('date')['stock_code'].count().mean():.0f}")
    print(f"\nFactor distribution:")
    print(f"  Mean: {output['factor_value'].mean():.4f}")
    print(f"  Std:  {output['factor_value'].std():.4f}")
    
    return output

if __name__ == '__main__':
    calc_open_consistency('data/csi1000_kline_raw.csv', 'data/factor_open_consistency_v1.csv')
