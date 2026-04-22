"""
return_consistency_v1 - 收益方向一致性因子 (Quality代理)
--------------------------------------------------------
理论: Novy-Marx (2014) "Is Momentum Really Momentum?" JFE
Heston et al. (2010) "Seasonality in the Cross-Section of Stock Returns"

收益方向一致性 = 高盈利质量/信息质量 → 股价走势更连贯

构造:
  1. 算sign(ret): 当日收益率的符号 (1/-1/0)
  2. 滚40日: majority vote判断主导方向, 计算同向天数占比
     方向一致性 = sum(sign(ret) == sign(dominant_direction)) / 40
     值域[0.5, 1.0] (50%=随机分歧, 100%=完全一致)
  3. -log(1-x)变换 -> logit变换增强差异
  4. 成交额OLS中性化 + MAD缩尾 + z-score

Barra风格: Quality
方向: 正向（高一致性=高预期收益）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def calc_consistency_factor(returns_path, output_path, window=40):
    """计算收益方向一致性因子"""
    
    print(f"=== return_consistency_v1 因子计算 ===")
    
    # 加载收益数据
    rets = pd.read_csv(returns_path)
    rets['date'] = pd.to_datetime(rets['date'])
    rets = rets.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"收益数据: {rets.shape[0]}条, {rets['stock_code'].nunique()}只股票")
    print(f"日期范围: {rets['date'].min()} ~ {rets['date'].max()}")
    
    # sign(ret)
    rets['sign_ret'] = np.sign(rets['return'])
    
    # Rolling方向一致性
    print(f"计算{window}日方向一致性...")
    
    def calc_consistency(sign_series, w=window):
        """
        计算滚动一致性: 在窗口内, 与主导方向同向的天数占比
        sign_series: pandas Series of sign values
        """
        # 截面一致性计算 (逐日滚动)
        consistency = sign_series.rolling(w, min_periods=int(w*0.8)).apply(
            lambda x: float(np.sum(np.sign(np.sum(x)) * x == 1)) / len(x) if len(x) > 0 and np.sum(x) != 0 else 0.5,
            raw=False
        )
        # 值域映射到[0.5, 1]: 占比最低0%对应0.5, 100%对应1.0
        return 0.5 + 0.5 * consistency
    
    rets['consistency'] = rets.groupby('stock_code')['sign_ret'].transform(
        lambda x: calc_consistency(x)
    )
    
    # logit变换增强差异
    eps = 1e-6
    rets['factor_raw'] = -np.log(1 - rets['consistency'] + eps)
    # 负号 Transformation: 单调递减，值域[0, ∞)
    # 一致性低(<=50%) → |factor|小; 一致性高(>=75%) → |factor|大
    # 不用负号: 一致性高→因子值大→正向使用
    
    # 加载成交额
    kline = pd.read_csv('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data/csi1000_kline_raw.csv')
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
    
    # 成交额OLS中性化
    print("成交额OLS中性化...")
    dates = sorted(df['date'].unique())
    results = []
    
    for dt in dates:
        mask = df['date'] == dt
        sub = df.loc[mask, ['date', 'stock_code', 'factor_raw', 'log_amount_20d']].copy()
        sub = sub.dropna(subset=['factor_raw', 'log_amount_20d'])
        if len(sub) < 30:
            continue
        
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=40)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    BASE_DIR = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'
    RETURNS_PATH = f'{BASE_DIR}/data/csi1000_returns.csv'
    OUTPUT_PATH = args.output or f'{BASE_DIR}/data/factor_ret_consistency_v2.csv'
    
    calc_consistency_factor(RETURNS_PATH, OUTPUT_PATH, args.window)
