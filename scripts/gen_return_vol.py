"""
return_volatility_v1 - 收益波动率因子 (Quality代理)
---------------------------------------------------
理论: Bali et al. (2016) EMP, Novy-Marx (2014) Quality
低残差波动率 = 盈利稳定/信息质量高 = 未来更好

构造:
  1. 计算过去60日日收益率的已实现波动率: sqrt(sum(ret^2))
  2. 取负号: -std(ret_60d) — 低波动=高因子值
  3. 成交额OLS中性化
  4. MAD缩尾 + z-score标准化

Barra风格: Quality
方向: 正向（低波动=高预期收益）
"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def calc_return_volatility_factor(kline_path, returns_path, output_path, window=60):
    """计算return_volatility_v1因子"""
    
    print(f"=== return_volatility_v1 因子计算 ===")
    print(f"K线数据: {kline_path}")
    print(f"收益数据: {returns_path}")
    print(f"窗口: {window}日")
    
    # 加载收益数据
    rets = pd.read_csv(returns_path)
    rets['date'] = pd.to_datetime(rets['date'])
    rets = rets.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"收益数据: {rets.shape[0]}条, {rets['stock_code'].nunique()}只股票")
    print(f"日期范围: {rets['date'].min()} ~ {rets['date'].max()}")
    
    # 计算滚动 realized volatility
    print(f"计算{window}日已实现波动率...")
    rets['ret_sq'] = rets['return'] ** 2
    
    # Rolling realized volatility
    factor_raw = rets.groupby('stock_code')['ret_sq'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).sum().apply(np.sqrt)
    )
    rets['factor_raw'] = -factor_raw  # 负号：低波动=高因子值
    
    # 加载成交额用于中性化
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['log_amount_20d'] = np.log(kline.groupby('stock_code')['amount']
                                      .transform(lambda x: x.rolling(20, min_periods=15).mean()) + 1)
    
    # 合并
    df = rets[['date', 'stock_code', 'factor_raw']].merge(
        kline[['date', 'stock_code', 'log_amount_20d']], 
        on=['date', 'stock_code'], 
        how='inner'
    )
    df = df.dropna(subset=['factor_raw', 'log_amount_20d'])
    
    # 成交额OLS中性化 (每截面)
    print("成交额OLS中性化...")
    from sklearn.linear_model import LinearRegression
    dates = df['date'].unique()
    results = []
    
    for dt in dates:
        mask = df['date'] == dt
        sub = df.loc[mask, ['date', 'stock_code', 'factor_raw', 'log_amount_20d']].copy()
        sub = sub.dropna(subset=['factor_raw', 'log_amount_20d'])
        if len(sub) < 30:
            continue
        
        # OLS regression
        reg = LinearRegression()
        X = sub['log_amount_20d'].values.reshape(-1, 1)
        y = sub['factor_raw'].values
        reg.fit(X, y)
        residuals = y - reg.predict(X)
        
        out = sub[['date', 'stock_code']].copy()
        out['factor_neutral'] = residuals
        results.append(out)
    
    df = pd.concat(results, ignore_index=True)
    
    # MAD缩尾 + z-score
    print("MAD缩尾 + z-score标准化...")
    def mad_zscore(group):
        vals = group['factor_neutral'].values
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            group['factor_value'] = 0
            return group
        upper = median + 5.5 * mad
        lower = median - 5.5 * mad
        vals_clipped = np.clip(vals, lower, upper)
        mean = np.mean(vals_clipped)
        std = np.std(vals_clipped)
        if std == 0:
            group['factor_value'] = 0
            return group
        group['factor_value'] = (vals_clipped - mean) / std
        return group
    
    df = df.groupby('date', group_keys=False).apply(mad_zscore)
    
    # 输出
    result = df[['date', 'stock_code', 'factor_value']].dropna()
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    result.to_csv(output_path, index=False)
    
    print(f"因子数据: {result.shape[0]}条")
    print(f"日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"因子均值: {result['factor_value'].mean():.4f}")
    print(f"因子标准差: {result['factor_value'].std():.4f}")
    print(f"因子max: {result['factor_value'].max():.4f}, min: {result['factor_value'].min():.4f}")
    print(f"输出: {output_path}")
    print("因子计算完成!")
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    BASE_DIR = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'
    KLINE_PATH = f'{BASE_DIR}/data/csi1000_kline_raw.csv'
    RETURNS_PATH = f'{BASE_DIR}/data/csi1000_returns.csv'
    OUTPUT_PATH = args.output or f'{BASE_DIR}/data/factor_return_vol_60d.csv'
    
    calc_return_volatility_factor(KLINE_PATH, RETURNS_PATH, OUTPUT_PATH, args.window)
