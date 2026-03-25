"""
因子：换手率动量 (Turnover Momentum)
ID: turnover_mom_v1

逻辑：
  MA5(turnover) - MA15(turnover) 的标准化版本
  衡量换手率的短期变化趋势
  高值 = 换手率在加速提升 = 投资者关注度增加
  
  两个版本一起测：
  v1: 做多换手率加速（正向）
  v2: 做多换手率减速（反向）

中性化：成交额OLS中性化
"""

import pandas as pd
import numpy as np
import sys

def calc_turnover_momentum(kline_path, output_path, window_short=5, window_long=20):
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print("Calculating turnover momentum...")
    
    # MA5 and MA20 of turnover
    df['turnover_ma_short'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(window_short, min_periods=3).mean()
    )
    df['turnover_ma_long'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(window_long, min_periods=10).mean()
    )
    
    # Turnover momentum = log(short/long)
    df['raw_factor'] = np.log(
        (df['turnover_ma_short'] / df['turnover_ma_long']).clip(lower=0.01, upper=100)
    )
    
    # 准备成交额中性化
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    print("Cross-section neutralization...")
    factor_df = df[['date', 'stock_code', 'raw_factor', 'log_amount_20d']].dropna().copy()
    
    def process_date(group):
        y = group['raw_factor'].values
        x = group['log_amount_20d'].values
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            residuals[valid] = y[valid] - X @ beta
        except:
            group['factor'] = np.nan
            return group
        
        vals = residuals[valid]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad < 1e-8:
            group['factor'] = 0.0
            return group
        
        bound = 5 * 1.4826 * mad
        residuals = np.clip(residuals, med - bound, med + bound)
        
        vals_valid = residuals[valid]
        mean = np.mean(vals_valid)
        std = np.std(vals_valid)
        if std < 1e-8:
            group['factor'] = 0.0
            return group
        
        group['factor'] = (residuals - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(process_date)
    
    # 输出正向（做多换手率加速）
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output = output.rename(columns={'factor': 'factor_value'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    # 也输出反向版本
    output_neg = output.copy()
    output_neg['factor_value'] = -output_neg['factor_value']
    neg_path = output_path.replace('.csv', '_neg.csv')
    output_neg.to_csv(neg_path, index=False)
    
    print(f"Factor saved to {output_path} (positive direction)")
    print(f"Factor saved to {neg_path} (negative direction)")
    print(f"  Rows: {len(output)}")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  Stocks per date: {output.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    calc_turnover_momentum('data/csi1000_kline_raw.csv', 'data/factor_turnover_mom_v1.csv')
