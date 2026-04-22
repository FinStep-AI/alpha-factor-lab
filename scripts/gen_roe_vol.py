"""
roe_vol_adjusted_v1 - ROE波动率因子 (Quality代理)
----------------------------------------------------
理论: Novy-Marx (2017) "Is Momentum Really Momentum?"
Penman et al. (2018) "Fundamental Analysis"
Fama & French (2015) "A Five-Factor Asset Pricing Model"

低ROE波动率 = 稳定的盈利能力 = 高质量公司 = 风险溢价补偿更低 = 未来收益更好。

注意: ROE波动率与常规波动率反向(高RV=高因子值), 与return_vol反转。
构造:
  1. 计算每只股票8个季度ROE的rolling标准差(年化调整)
  2. 取负号: -std(ROE_8q) — 低波动=高因子值
  3. 市值中性化 (总资产=PB*股价 → 用收盘价×股数 proxy)
  4. MAD缩尾 + z-score

Barra风格: Quality
方向: 正向（低ROE波动=高预期收益）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def calc_roe_vol_factor(fundamental_path, kline_path, output_path, window_quarters=8):
    """计算ROE波动率因子"""
    
    print(f"=== roe_vol_adjusted_v1 因子计算 ===")
    
    # 加载基本面数据
    fund = pd.read_csv(fundamental_path)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund['code_str'] = fund['stock_code'].astype(str).str.zfill(6)
    
    # 加载K线获取交易日期和价格
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['code_str'] = kline['stock_code'].astype(str).str.zfill(6)
    
    # 市值proxy (用成交额20日均值)
    kline['log_amount_20d'] = np.log(kline.groupby('stock_code')['amount']
                                       .transform(lambda x: x.rolling(20, min_periods=15).mean()) + 1)
    
    print(f"基本面: {fund.shape[0]}条, {fund['code_str'].nunique()}只股票")
    print(f"ROE数据: {fund['roe'].min():.2f} ~ {fund['roe'].max():.2f}")
    
    # 每只股票/报告期的ROE值
    # 使用rolling std (window_quarters = 8 quarters = 2 years)
    print(f"计算{window_quarters}季度ROE滚动标准差...")
    
    # 按股票+时间排序
    fund = fund.sort_values(['code_str', 'report_date']).reset_index(drop=True)
    
    # 计算每只股票的rolling std of ROE
    fund['roe_std'] = fund.groupby('code_str')['roe'].transform(
        lambda x: x.shift(1).rolling(window_quarters, min_periods=6).std()
    )
    # shift(1): 防止前视偏差(当前期财报包含未来信息)
    
    # 因子值 = -roe_std (低波动→高因子值)
    fund['factor_raw'] = -fund['roe_std']
    
    # Expand到交易日期: 每份财报有效到下一份财报发布日(或まで)
    # 为简化, 将财报日期映射到当日/次日
    # 实际操作: 财报发布后有信息滞后, 此处简单用report_date作为有效起始日
    
    # 选择最新财报 (每个财报期取最新的)
    fund_latest = fund.sort_values(['code_str', 'report_date', 'roe_std'], 
                                    ascending=[True, True, False])
    fund_latest = fund_latest.drop_duplicates(['code_str', 'report_date'], keep='first')
    
    # Map to K-line dates
    print("映射财报到交易日期...")
    # For each trade_date, use the latest report_date <= trade_date
    trade_dates = kline[['date', 'code_str', 'log_amount_20d']].copy()
    trade_dates = trade_dates.drop_duplicates(['date', 'code_str'])
    trade_dates = trade_dates.sort_values(['date', 'code_str'])
    
    # Simple approach: forward-fill roe_std per stock
    # Get stock-level report_data
    stock_reports = fund.sort_values(['code_str', 'report_date'])[['code_str', 'report_date', 'factor_raw']].copy()
    stock_reports = stock_reports.dropna(subset=['factor_raw'])
    
    # Create all combinations and forward fill
    all_stocks = kline[['date', 'code_str', 'log_amount_20d']].drop_duplicates('code_str')[['code_str']]
    all_dates = sorted(kline['date'].unique())
    
    # For each stock, merge with reports and forward-fill
    results = []
    stock_list = all_stocks['code_str'].unique()
    
    for i, stock in enumerate(stock_list):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(stock_list)} stocks...")
        
        stock_reports_sub = stock_reports[stock_reports['code_str'] == stock].copy()
        if len(stock_reports_sub) == 0:
            continue
        
        stock_trades = trade_dates[trade_dates['code_str'] == stock].copy()
        stock_trades = stock_trades.sort_values('date')
        
        # Merge with reports (left join on date)
        merged = stock_trades.merge(stock_reports_sub, on='code_str', how='left')
        merged = merged[merged['report_date'] <= merged['date']]
        
        if len(merged) == 0:
            continue
        
        # Keep latest report_date per trade_date
        merged = merged.sort_values(['date', 'report_date'], ascending=[True, False])
        merged = merged.drop_duplicates(['date', 'code_str'], keep='first')
        
        results.append(merged[['date', 'code_str', 'factor_raw', 'log_amount_20d']])
    
    df = pd.concat(results, ignore_index=True)
    df = df.dropna(subset=['factor_raw'])
    
    print(f"Factor data before neutralization: {df.shape[0]} rows")
    
    # 市值OLS中性化
    print("成交额OLS中性化...")
    dates = sorted(df['date'].unique())
    neutral_results = []
    
    for dt in dates:
        mask = df['date'] == dt
        sub = df.loc[mask, ['date', 'code_str', 'factor_raw', 'log_amount_20d']].copy()
        sub = sub.dropna(subset=['factor_raw', 'log_amount_20d'])
        if len(sub) < 30:
            continue
        
        reg = LinearRegression()
        X = sub['log_amount_20d'].values.reshape(-1, 1)
        y = sub['factor_raw'].values
        reg.fit(X, y)
        residuals = y - reg.predict(X)
        
        out = sub[['date', 'code_str']].copy()
        out['factor_neutral'] = residuals
        neutral_results.append(out)
    
    df_neutral = pd.concat(neutral_results, ignore_index=True)
    
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
    parser.add_argument('--window', type=int, default=8)
    args = parser.parse_args()
    
    BASE_DIR = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'
    FUND_PATH = f'{BASE_DIR}/data/csi1000_fundamental_cache.csv'
    KLINE_PATH = f'{BASE_DIR}/data/csi1000_kline_raw.csv'
    OUTPUT_PATH = f'{BASE_DIR}/data/factor_roe_vol_v1.csv'
    
    calc_roe_vol_factor(FUND_PATH, KLINE_PATH, OUTPUT_PATH, args.window)
