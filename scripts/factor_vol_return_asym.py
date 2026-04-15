#!/usr/bin/env python3
"""
Volume-Return Asymmetry (VRA) Factor
------------------------------------
VRA = log(mean_volume_up_days / mean_volume_down_days), 20日窗口
上涨日=收益率>0; 下跌日=收益率<0
正值=上涨时放量、下跌时缩量=资金看多
成交额OLS中性化

文献: Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium"
       Llorente et al. (2002) "Dynamic Volume-Return Relation"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compute_vra(df_raw, lookback=20):
    """计算VRA因子"""
    records = []
    
    for code, gdf in df_raw.groupby('code'):
        gdf = gdf.sort_values('date').copy()
        
        # 日收益率
        gdf['ret'] = gdf['close'].pct_change()
        # 成交量
        vol = gdf['volume'].values
        ret = gdf['ret'].values
        dates = gdf['date'].values
        close = gdf['close'].values
        amount = gdf['amount'].values if 'amount' in gdf.columns else vol * close
        
        for i in range(lookback, len(gdf)):
            window_ret = ret[i-lookback+1:i+1]
            window_vol = vol[i-lookback+1:i+1]
            
            up_mask = window_ret > 0
            down_mask = window_ret < 0
            
            n_up = up_mask.sum()
            n_down = down_mask.sum()
            
            if n_up < 3 or n_down < 3:
                continue
            
            mean_vol_up = window_vol[up_mask].mean()
            mean_vol_down = window_vol[down_mask].mean()
            
            if mean_vol_down <= 0 or mean_vol_up <= 0:
                continue
            
            vra = np.log(mean_vol_up / mean_vol_down)
            
            # 同时记录20日成交额均值用于中性化
            log_amount_20d = np.log(amount[i-lookback+1:i+1].mean() + 1)
            
            records.append({
                'date': dates[i],
                'code': code,
                'factor_raw': vra,
                'log_amount_20d': log_amount_20d
            })
    
    factor_df = pd.DataFrame(records)
    if factor_df.empty:
        return factor_df
    
    # 成交额OLS中性化
    from sklearn.linear_model import LinearRegression
    result = []
    for date, group in factor_df.groupby('date'):
        if len(group) < 30:
            continue
        X = group[['log_amount_20d']].values
        y = group['factor_raw'].values
        
        # winsorize
        median = np.median(y)
        mad = np.median(np.abs(y - median))
        if mad > 0:
            y_clip = np.clip(y, median - 5*1.4826*mad, median + 5*1.4826*mad)
        else:
            y_clip = y
        
        lr = LinearRegression()
        lr.fit(X, y_clip)
        residual = y_clip - lr.predict(X)
        
        # z-score
        std = residual.std()
        if std > 0:
            residual = residual / std
        
        tmp = group.copy()
        tmp['factor'] = residual
        result.append(tmp[['date', 'code', 'factor']])
    
    return pd.concat(result, ignore_index=True)


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent / 'data'
    
    print("读取K线数据...")
    df_raw = pd.read_csv(data_dir / 'csi1000_kline_raw.csv')
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    # 标准化列名
    if 'stock_code' in df_raw.columns:
        df_raw.rename(columns={'stock_code': 'code'}, inplace=True)
    
    # 确保有amount列
    if 'amount' not in df_raw.columns:
        df_raw['amount'] = df_raw['volume'] * df_raw['close']
    
    print(f"股票数: {df_raw['code'].nunique()}, 行数: {len(df_raw)}")
    
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"计算VRA因子 (lookback={lookback})...")
    
    factor_df = compute_vra(df_raw, lookback=lookback)
    
    out_path = data_dir / f'factor_vol_return_asym_{lookback}d.csv'
    factor_df.to_csv(out_path, index=False)
    print(f"保存到 {out_path}, 行数: {len(factor_df)}")
    print(f"日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"因子统计: mean={factor_df['factor'].mean():.4f}, std={factor_df['factor'].std():.4f}")
