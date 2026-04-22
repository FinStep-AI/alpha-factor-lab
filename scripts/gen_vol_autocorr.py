"""
volume_autocorr_v1 - 成交量自相关因子 (Quality代理)
---------------------------------------------------
理论: Brennan, Huh & Subrahmanyam (2013) "An Analysis of the Amihud Illiquidity Premium"
Chordia & Sarkar (2005) "Pricing the Factor Risk Premia"

成交量自相关捕捉持续性交易行为：
  高自相关 = 持续的资金流入/关注度/信息释放 = 优质资产
  低自相关 = 偶然性成交量/噪音交易 = 低质资产

构造:
  1. log(volume + 1) 对20日滚动求自相关系数 (5日滞后)
  2. 成交额OLS中性化
  3. MAD缩尾 + z-score

Barra风格: Quality (新增)
方向: 正向（高自相关=高预期收益）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def calc_vol_autocorr_factor(kline_path, output_path, window=20, lag=5):
    """计算成交量自相关因子"""
    
    print(f"=== volume_autocorr_v1 因子计算 ===")
    print(f"窗口={window}, 滞后={lag}")
    
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['code_str'] = kline['stock_code'].astype(str).str.zfill(6)
    
    print(f"K线数据: {kline.shape[0]}条, {kline['code_str'].nunique()}只股票")
    print(f"日期范围: {kline['date'].min()} ~ {kline['date'].max()}")
    
    # log volume
    kline['log_vol'] = np.log(kline['volume'] + 1)
    
    # Rolling autocorrelation
    print(f"计算{window}日滚动自相关(lag={lag})...")
    
    def rolling_autocorr(series, w=window, L=lag):
        """Rolling autocorrelation with given lag"""
        if len(series) < w:
            return np.nan
        window_data = series.iloc[-w:]
        if len(window_data) <= L:
            return np.nan
        return window_data.autocorr(lag=L)
    
    kline['vol_autocorr'] = kline.groupby('code_str')['log_vol'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).apply(
            lambda s: pd.Series(s).autocorr(lag=lag) if len(s) > lag else np.nan,
            raw=False
        )
    )
    
    # 成交额中性化变量
    kline['log_amount_20d'] = np.log(kline.groupby('code_str')['amount']
                                       .transform(lambda x: x.rolling(20, min_periods=15).mean()) + 1)
    
    # 准备neutralization
    df = kline[['date', 'code_str', 'vol_autocorr', 'log_amount_20d']].copy()
    df = df.dropna(subset=['vol_autocorr', 'log_amount_20d'])
    
    print(f"Neutralization前数据: {df.shape[0]}条")
    
    # OLS neutralization
    print("成交额OLS中性化...")
    dates = sorted(df['date'].unique())
    results = []
    
    for dt in dates:
        mask = df['date'] == dt
        sub = df.loc[mask, ['date', 'code_str', 'vol_autocorr', 'log_amount_20d']].copy()
        sub = sub.dropna(subset=['vol_autocorr', 'log_amount_20d'])
        if len(sub) < 30:
            continue
        
        reg = LinearRegression()
        X = sub['log_amount_20d'].values.reshape(-1, 1)
        y = sub['vol_autocorr'].values
        reg.fit(X, y)
        residuals = y - reg.predict(X)
        
        out = sub[['date', 'code_str']].copy()
        out['factor_neutral'] = residuals
        results.append(out)
    
    df_neutral = pd.concat(results, ignore_index=True)
    
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
    
    df_neutral = df_neutral.groupby('date', group_keys=False).apply(mad_zscore)
    
    # output
    result = df_neutral[['date', 'code_str', 'factor_value']].dropna()
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    result = result.rename(columns={'code_str': 'stock_code'})
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
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--lag', type=int, default=5)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    BASE_DIR = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'
    KLINE_PATH = f'{BASE_DIR}/data/csi1000_kline_raw.csv'
    OUTPUT_PATH = args.output or f'{BASE_DIR}/data/factor_vol_autocorr_v1.csv'
    
    calc_vol_autocorr_factor(KLINE_PATH, OUTPUT_PATH, args.window, args.lag)
