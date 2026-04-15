"""
ROE加速度因子 (roe_accel_v1)
---------------------------------
逻辑：ROE环比变化的变化（二阶导数）
捕捉盈利改善加速/恶化加速的公司
正值 = 盈利改善在加速（看多信号）

构造：
1. 计算每季度ROE环比变化 delta_ROE(t) = ROE(t) - ROE(t-1)
2. 计算加速度 accel(t) = delta_ROE(t) - delta_ROE(t-1)
3. 财报数据按发布节奏映射到交易日（延迟1个月确保数据可得）
4. 市值中性化（对log(市值)回归取残差）

Barra风格: Growth
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    
    # Sort
    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # --- Step 1: Compute ROE acceleration per stock per report_date ---
    fund['roe_prev'] = fund.groupby('stock_code')['roe'].shift(1)
    fund['roe_delta'] = fund['roe'] - fund['roe_prev']
    fund['roe_delta_prev'] = fund.groupby('stock_code')['roe_delta'].shift(1)
    fund['roe_accel'] = fund['roe_delta'] - fund['roe_delta_prev']
    
    # Drop rows without acceleration
    fund_valid = fund.dropna(subset=['roe_accel'])[['stock_code', 'report_date', 'roe_accel']].copy()
    
    # --- Step 2: Map report_date to effective trading dates ---
    # Conservative: data available ~1 month after report_date
    # Q1 (03-31) -> available 05-01
    # Q2 (06-30) -> available 09-01 (mid-report deadline 08-31)
    # Q3 (09-30) -> available 11-01
    # Q4 (12-31) -> available 05-01 next year (annual report deadline 04-30)
    
    report_to_avail = {
        '03-31': lambda y: f'{y}-05-01',
        '06-30': lambda y: f'{y}-09-01',
        '09-30': lambda y: f'{y}-11-01',
        '12-31': lambda y: f'{int(y)+1}-05-01',
    }
    
    def get_available_date(report_date):
        y = report_date[:4]
        md = report_date[5:]
        if md in report_to_avail:
            return report_to_avail[md](y)
        return None
    
    fund_valid['avail_date'] = fund_valid['report_date'].apply(get_available_date)
    fund_valid = fund_valid.dropna(subset=['avail_date'])
    
    # --- Step 3: For each trading date, get the latest available factor value ---
    trade_dates = sorted(kline['date'].unique())
    stocks = sorted(kline['stock_code'].unique())
    
    # Build lookup: for each stock, sorted list of (avail_date, roe_accel)
    stock_factors = {}
    for sc, grp in fund_valid.groupby('stock_code'):
        records = grp.sort_values('avail_date')[['avail_date', 'roe_accel']].values.tolist()
        stock_factors[sc] = records
    
    # For each trade date, for each stock, find latest available roe_accel
    results = []
    for td in trade_dates:
        for sc in stocks:
            if sc not in stock_factors:
                continue
            records = stock_factors[sc]
            # Find latest record where avail_date <= td
            val = np.nan
            for avail_d, accel in records:
                if avail_d <= td:
                    val = accel
                else:
                    break
            if not np.isnan(val):
                results.append({'date': td, 'stock_code': sc, 'factor_raw': val})
    
    factor_df = pd.DataFrame(results)
    print(f"Raw factor rows: {len(factor_df)}, dates: {factor_df['date'].nunique()}, stocks per date avg: {len(factor_df)/factor_df['date'].nunique():.0f}")
    
    # --- Step 4: Market-cap neutralization ---
    # Use amount/turnover as proxy for market cap (amount = price * volume, turnover = volume/total_shares)
    # So market_cap_proxy = amount / turnover (roughly proportional to market cap)
    kline['mktcap_proxy'] = np.where(kline['turnover'] > 0, kline['amount'] / kline['turnover'], np.nan)
    kline['log_mktcap'] = np.log(kline['mktcap_proxy'])
    
    mktcap = kline[['date', 'stock_code', 'log_mktcap']].copy()
    factor_df = factor_df.merge(mktcap, on=['date', 'stock_code'], how='left')
    
    # Cross-sectional regression: factor_raw ~ log_mktcap, take residual
    def neutralize(group):
        valid = group.dropna(subset=['factor_raw', 'log_mktcap'])
        if len(valid) < 30:
            group['factor'] = np.nan
            return group
        x = valid['log_mktcap'].values
        y = valid['factor_raw'].values
        
        # Winsorize extremes (1/99 percentile)
        y_low, y_high = np.percentile(y, [1, 99])
        y = np.clip(y, y_low, y_high)
        
        # OLS
        x_dm = x - x.mean()
        beta = np.sum(x_dm * y) / np.sum(x_dm ** 2) if np.sum(x_dm ** 2) > 0 else 0
        residual = y - beta * x_dm - y.mean()
        
        # Z-score
        std = residual.std()
        if std > 0:
            residual = residual / std
        
        group.loc[valid.index, 'factor'] = residual
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize)
    
    # --- Step 5: Output ---
    output = factor_df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    output.to_csv('data/factor_roe_accel_v1.csv', index=False)
    print(f"\nOutput: data/factor_roe_accel_v1.csv")
    print(f"Rows: {len(output)}, Dates: {output['date'].nunique()}, Stocks/date: {len(output)/output['date'].nunique():.0f}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"\nFactor stats:")
    print(output['factor'].describe())

if __name__ == '__main__':
    main()
