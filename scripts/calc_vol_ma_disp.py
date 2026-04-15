#!/usr/bin/env python3
"""
因子：成交量均线离散度 (Volume MA Dispersion)
ID: vol_ma_disp_v1

灵感来源：ma_disp_v1（价格均线离散度，IC=0.026, t=4.25）表现极好。
将同样的离散度思想应用到成交量维度。

构造：
  1. 计算5条成交量均线：MA5, MA10, MA20, MA40, MA60
  2. 每条均线除以MA20（归一化到同一尺度）
  3. 5个归一化值的截面标准差 = 成交量均线离散度
  4. 对数变换 + 成交额OLS中性化 + MAD winsorize + z-score

逻辑：
  - 高离散度 = 成交量多周期均线发散 = 成交量趋势明确（放量or缩量中）
  - 低离散度 = 成交量均线收敛 = 交易量平稳
  - 如果高离散度→高收益：说明成交量趋势变化本身包含alpha信号
  - 如果低离散度→高收益：说明交易平稳的股票更优

Barra风格目标：Liquidity/Sentiment（补充换手率维度）
"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

def compute_factor(df):
    """计算成交量均线离散度因子"""
    df = df.copy()
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    results = []
    
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        vol = grp['volume'].values
        vol_s = pd.Series(vol)
        
        # 5条成交量均线
        ma5 = vol_s.rolling(5, min_periods=3).mean()
        ma10 = vol_s.rolling(10, min_periods=5).mean()
        ma20 = vol_s.rolling(20, min_periods=10).mean()
        ma40 = vol_s.rolling(40, min_periods=20).mean()
        ma60 = vol_s.rolling(60, min_periods=30).mean()
        
        # 归一化：每条均线除以MA20
        r5 = ma5 / ma20
        r10 = ma10 / ma20
        r20 = ma20 / ma20  # = 1.0
        r40 = ma40 / ma20
        r60 = ma60 / ma20
        
        # 5个比值的标准差 = 离散度
        ratios = pd.DataFrame({
            'r5': r5, 'r10': r10, 'r20': r20, 'r40': r40, 'r60': r60
        })
        disp = ratios.std(axis=1)
        
        grp['raw_factor'] = disp.values
        results.append(grp[['date', 'stock_code', 'raw_factor']])
    
    result = pd.concat(results, ignore_index=True)
    return result


def neutralize_and_standardize(df, kline_df):
    """成交额OLS中性化 + MAD winsorize + z-score"""
    df = df.copy()
    
    # Merge amount for neutralization
    amt = kline_df[['date', 'stock_code', 'amount']].copy()
    amt['date'] = pd.to_datetime(amt['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 计算20日平均成交额
    amt = amt.sort_values(['stock_code', 'date'])
    amt['log_amount_20d'] = amt.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    df = df.merge(amt[['date', 'stock_code', 'log_amount_20d']], on=['date', 'stock_code'], how='left')
    
    # 对数变换
    df['log_factor'] = np.log(df['raw_factor'].clip(lower=1e-10))
    
    # 按日截面中性化
    output = []
    for dt, cross in df.groupby('date'):
        cs = cross.dropna(subset=['log_factor', 'log_amount_20d']).copy()
        if len(cs) < 50:
            continue
        
        # OLS neutralize
        X = cs['log_amount_20d'].values
        y = cs['log_factor'].values
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            residual = y - X_with_const @ beta
        except:
            continue
        
        cs['factor_neutral'] = residual
        
        # MAD winsorize
        med = np.nanmedian(cs['factor_neutral'])
        mad = np.nanmedian(np.abs(cs['factor_neutral'] - med))
        if mad < 1e-10:
            cs['factor_win'] = cs['factor_neutral']
        else:
            upper = med + 5.0 * 1.4826 * mad
            lower = med - 5.0 * 1.4826 * mad
            cs['factor_win'] = cs['factor_neutral'].clip(lower=lower, upper=upper)
        
        # z-score
        mean = cs['factor_win'].mean()
        std = cs['factor_win'].std()
        if std < 1e-10:
            cs['factor'] = 0.0
        else:
            cs['factor'] = (cs['factor_win'] - mean) / std
        
        output.append(cs[['date', 'stock_code', 'factor']])
    
    return pd.concat(output, ignore_index=True)


if __name__ == '__main__':
    print("加载数据...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    
    print("计算成交量均线离散度...")
    raw = compute_factor(kline)
    
    print("中性化 + 标准化...")
    final = neutralize_and_standardize(raw, kline)
    
    # 保存
    output_path = 'data/factor_vol_ma_disp.csv'
    final.to_csv(output_path, index=False)
    
    print(f"因子计算完成! 保存到 {output_path}")
    print(f"  日期范围: {final['date'].min()} ~ {final['date'].max()}")
    print(f"  股票数: {final['stock_code'].nunique()}")
    print(f"  总记录: {len(final)}")
    print(f"  因子统计: mean={final['factor'].mean():.4f}, std={final['factor'].std():.4f}")
    print(f"  非空比例: {final['factor'].notna().mean():.2%}")
