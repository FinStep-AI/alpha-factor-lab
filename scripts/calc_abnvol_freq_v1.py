#!/usr/bin/env python3
"""
因子: 异常成交日频率 (Abnormal Volume Day Frequency) v1
公式: count(vol_i > 2 * MA60(vol)) / 20, 过去20日中成交量超过60日均量2倍的天数占比
      成交额OLS中性化 + MAD winsorize + z-score

逻辑:
- 异常高成交量日反映重大信息事件(财报、公告、资金进出)
- 高频率 = 近期信息事件密集 = 市场对该股关注度高
- 与vol_cv_neg不同: vol_cv_neg看波动稳定性, 本因子看极端日频率
- 与vol_hhi不同: vol_hhi用HHI衡量集中度(可能被少数极端日驱动), 本因子用固定阈值(2x)计数
- vol_hhi已失败(IC=0.006), 但可能是因为HHI构造问题, 阈值法可能不同

假说1(正向): 信息事件密集的股票信息传播加速,价格发现更充分→正alpha
假说2(反向): 信息事件密集=不确定性大→负alpha(被超额卖出后反弹)

同时输出正反两个方向的因子文件
"""

import numpy as np
import pandas as pd
import sys, os

def compute_factor():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'csi1000_kline_raw.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"数据: {len(df)} 行, {df['stock_code'].nunique()} 股, {df['date'].nunique()} 日")
    
    g = df.groupby('stock_code')
    
    # MA60 of volume
    df['vol_ma60'] = g['volume'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    
    # Abnormal volume flag: volume > 2 * MA60
    df['abnormal_vol'] = (df['volume'] > 2.0 * df['vol_ma60']).astype(float)
    df.loc[df['vol_ma60'].isna(), 'abnormal_vol'] = np.nan
    
    # 20-day rolling count of abnormal volume days
    df['raw_factor'] = g['abnormal_vol'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # log_amount_60d for neutralization
    df['log_amount_60d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(60, min_periods=30).mean().replace(0, np.nan))
    )
    
    print(f"原始因子有效值: {df['raw_factor'].notna().sum()}/{len(df)} ({df['raw_factor'].notna().mean()*100:.1f}%)")
    
    # Quick stats on raw factor
    valid = df['raw_factor'].dropna()
    print(f"原始因子分布: mean={valid.mean():.4f}, std={valid.std():.4f}, "
          f"min={valid.min():.4f}, max={valid.max():.4f}")
    
    # Cross-sectional processing
    results = []
    for date, group in df.groupby('date'):
        sub = group[['stock_code', 'raw_factor', 'log_amount_60d']].dropna()
        if len(sub) < 30:
            continue
        
        y = sub['raw_factor'].values.copy()
        x = sub['log_amount_60d'].values
        
        # MAD winsorize
        med = np.median(y)
        mad_val = np.median(np.abs(y - med))
        if mad_val > 0:
            bound = 3 * 1.4826 * mad_val
            y = np.clip(y, med - bound, med + bound)
        
        # OLS neutralize
        X = np.column_stack([np.ones(len(x)), x])
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 10:
            continue
        try:
            XtX = X[mask].T @ X[mask]
            Xty = X[mask].T @ y[mask]
            beta = np.linalg.solve(XtX + 1e-10 * np.eye(2), Xty)
            resid = y - X @ beta
        except:
            resid = y
        
        # Second MAD winsorize
        med2 = np.median(resid)
        mad2 = np.median(np.abs(resid - med2))
        if mad2 > 0:
            bound2 = 3 * 1.4826 * mad2
            resid = np.clip(resid, med2 - bound2, med2 + bound2)
        
        # Z-score
        std_val = np.std(resid)
        if std_val > 0:
            zscore = (resid - np.mean(resid)) / std_val
        else:
            zscore = resid * 0
        
        for i, idx in enumerate(sub.index):
            results.append({
                'date': date,
                'stock_code': sub.loc[idx, 'stock_code'],
                'factor': zscore[i]
            })
    
    result_df = pd.DataFrame(results)
    result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
    
    # Save positive direction
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'factor_abnvol_freq_v1.csv')
    result_df.to_csv(out_path, index=False)
    
    # Save negative direction
    result_neg = result_df.copy()
    result_neg['factor'] = -result_neg['factor']
    out_neg = os.path.join(os.path.dirname(__file__), '..', 'data', 'factor_abnvol_freq_neg_v1.csv')
    result_neg.to_csv(out_neg, index=False)
    
    print(f"\n因子已保存: {out_path} (正向) / {out_neg} (反向)")
    print(f"有效值: {result_df['factor'].notna().sum()}")
    print(f"均值: {result_df['factor'].mean():.4f}")
    print(f"标准差: {result_df['factor'].std():.4f}")
    print(f"日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")

if __name__ == '__main__':
    compute_factor()
