"""
因子: 收盘偏向趋势 (Close Bias Trend) v1
----------------------------------------------
构造:
  1. daily_bias = (close - mid) / range, 其中 mid=(high+low)/2, range=high-low
     映射到 [-1, 1], 收在最高=1, 收在最低=-1
  2. 对每只股票近20日 daily_bias 做OLS回归(y=bias, x=1..20), 取斜率
     斜率>0 = 买方力量在增强(收盘位置持续改善)
  3. 成交额OLS中性化 + MAD winsorize + z-score

逻辑:
  - 收盘位置(CLV)是经典的日内多空力量指标
  - 但CLV水平值已被多个因子使用(shadow_pressure, CMF等)
  - 本因子捕捉CLV的**变化趋势**, 即买方力量是否在加速
  - 趋势为正 = 每天收盘位置都在改善 = 渐进式买入 = 知情资金逐步建仓
  - 类比: turnover_decel看换手率的加减速, 本因子看收盘强度的加减速

Barra风格: Sentiment (买方力量趋势)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def compute_ols_slope(arr):
    """对arr做OLS回归, 返回斜率。arr中NaN过多则返回NaN。"""
    valid = ~np.isnan(arr)
    n = valid.sum()
    if n < 10:  # 至少10个有效点
        return np.nan
    x = np.arange(len(arr), dtype=float)
    x_valid = x[valid]
    y_valid = arr[valid]
    x_mean = x_valid.mean()
    y_mean = y_valid.mean()
    ss_xy = ((x_valid - x_mean) * (y_valid - y_mean)).sum()
    ss_xx = ((x_valid - x_mean) ** 2).sum()
    if ss_xx == 0:
        return np.nan
    return ss_xy / ss_xx

def neutralize_ols(df_factor, df_neutral_var):
    """OLS中性化: factor = alpha + beta * neutral_var + residual, 返回residual"""
    result = pd.Series(np.nan, index=df_factor.index)
    dates = df_factor.index.get_level_values('date').unique()
    for dt in dates:
        if dt not in df_factor.index.get_level_values('date'):
            continue
        y = df_factor.loc[dt]
        x = df_neutral_var.loc[dt] if dt in df_neutral_var.index.get_level_values('date') else None
        if x is None:
            continue
        common = y.dropna().index.intersection(x.dropna().index)
        if len(common) < 30:
            continue
        y_c = y[common].values
        x_c = x[common].values
        # OLS
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            resid = y_c - X @ beta
            for i, stk in enumerate(common):
                result.loc[(dt, stk)] = resid[i]
        except:
            continue
    return result

def mad_winsorize(s, n_mad=5):
    """MAD Winsorize"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    upper = med + n_mad * 1.4826 * mad
    lower = med - n_mad * 1.4826 * mad
    return s.clip(lower, upper)

def zscore(s):
    """截面z-score"""
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - mean) / std

def main():
    base = Path(__file__).resolve().parent.parent
    
    # 读取数据
    print("读取K线数据...")
    df = pd.read_csv(base / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'])
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算 daily_bias = (close - mid) / range
    print("计算日内收盘偏向...")
    df['mid'] = (df['high'] + df['low']) / 2
    df['range'] = df['high'] - df['low']
    df['daily_bias'] = np.where(
        df['range'] > 0,
        (df['close'] - df['mid']) / df['range'],
        0  # 涨跌停一字板, range=0
    )
    # daily_bias ∈ [-1, 1] 理论上, 但由于(close-mid)/range的定义是 ∈ [-0.5, 0.5]
    # 修正: (close - mid) / range 的范围是 [-0.5, 0.5], 乘2归一化到[-1,1]
    df['daily_bias'] = df['daily_bias'] * 2  # 现在 ∈ [-1, 1]
    
    # 对每只股票近20日daily_bias做OLS斜率
    print("计算20日收盘偏向趋势斜率...")
    window = 20
    
    slopes = []
    for stk, gdf in df.groupby('stock_code'):
        bias = gdf['daily_bias'].values
        dates = gdf['date'].values
        stk_slopes = np.full(len(bias), np.nan)
        for i in range(window - 1, len(bias)):
            window_data = bias[i - window + 1: i + 1]
            stk_slopes[i] = compute_ols_slope(window_data)
        for i in range(len(dates)):
            slopes.append({
                'date': dates[i],
                'stock_code': stk,
                'close_bias_trend': stk_slopes[i]
            })
    
    factor_df = pd.DataFrame(slopes)
    factor_df = factor_df.dropna(subset=['close_bias_trend'])
    print(f"原始因子行数: {len(factor_df)}")
    
    # 合并成交额(用于中性化)
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20).mean() + 1)
    )
    
    factor_df = factor_df.merge(amt_df[['date', 'stock_code', 'log_amount_20d']], on=['date', 'stock_code'], how='left')
    factor_df = factor_df.dropna(subset=['log_amount_20d'])
    
    # 设置MultiIndex
    factor_df = factor_df.set_index(['date', 'stock_code'])
    
    # OLS中性化
    print("成交额OLS中性化...")
    raw = factor_df['close_bias_trend']
    neutral_var = factor_df['log_amount_20d']
    
    neutralized = pd.Series(np.nan, index=raw.index)
    dates = raw.index.get_level_values('date').unique()
    for dt in dates:
        mask = raw.index.get_level_values('date') == dt
        y = raw[mask].dropna()
        x = neutral_var[mask].dropna()
        common = y.index.intersection(x.index)
        if len(common) < 30:
            continue
        y_c = y[common].values
        x_c = x[common].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            resid = y_c - X @ beta
            neutralized[common] = resid
        except:
            continue
    
    factor_df['factor_neutralized'] = neutralized
    factor_df = factor_df.dropna(subset=['factor_neutralized'])
    
    # MAD winsorize + z-score (截面)
    print("MAD winsorize + z-score...")
    final_values = []
    for dt in factor_df.index.get_level_values('date').unique():
        mask = factor_df.index.get_level_values('date') == dt
        vals = factor_df.loc[mask, 'factor_neutralized']
        vals = mad_winsorize(vals)
        vals = zscore(vals)
        final_values.append(vals)
    
    factor_df['factor'] = pd.concat(final_values)
    
    # 输出
    output = factor_df[['factor']].reset_index()
    output.columns = ['date', 'stock_code', 'factor']
    output = output.dropna()
    
    out_path = base / 'data' / 'factor_close_bias_trend_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子输出到: {out_path}")
    print(f"总行数: {len(output)}")
    print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"股票数: {output['stock_code'].nunique()}")
    
    # 基础统计
    print("\n截面因子统计 (最后一日):")
    last = output[output['date'] == output['date'].max()]
    print(f"  均值: {last['factor'].mean():.4f}")
    print(f"  标准差: {last['factor'].std():.4f}")
    print(f"  分位数: {last['factor'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()}")

if __name__ == '__main__':
    main()
