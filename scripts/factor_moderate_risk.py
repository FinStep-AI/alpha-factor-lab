"""
适度冒险因子 (moderate_risk_v1)
来源：方正金工《成交量激增时刻蕴含的alpha信息》2022

日频近似版本（原始研报使用分钟频数据）：
核心逻辑：成交量激增后价格的"适度波动"→ 过小(反应不足)或过大(反应过度)都不可取

构建步骤（日频重构）：
1. 识别成交量激增日：当日volume > 过去20日均值 + 1倍标准差
2. 计算成交额比例：volume_surge_ratio = volume / MA20(volume) - 1
3. 计算激增日的近5日绝对收益标准差（反应过度？）& 近5日收益绝对均值（反应方向）
4. 适度因子 = |激增日收益均值 - 截面均值|（希望适度）
5. 月频合成 = MA20(适度因子) / std20(适度因子)
6. 市值中性化
"""
import numpy as np
import pandas as pd

def calc_moderate_risk_factor(df_kline: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算适度冒险因子
    
    Parameters:
    - df_kline: DataFrame with columns [date, stock_code, open, close, high, low, volume, amount, pct_change]
    - window: 回看窗口，默认20日
    
    Returns:
    - DataFrame with columns [date, stock_code, factor_value]
    """
    df = df_kline.copy()
    
    # 1. 成交量特征
    df['volume_ma20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    df['volume_std20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )
    
    # 成交量偏离度（日频激增代理）
    # volume_ma20 > 0 才有意义，避免除零
    df['volume_surge_ratio'] = np.where(
        df['volume_ma20'] > 0,
        df['volume'] / df['volume_ma20'] - 1,
        np.nan
    )
    
    # 2. 激增识别：volume_surge_ratio > 1 （即成交量>均值+1倍std的近似）
    df['is_surge'] = df['volume_surge_ratio'] > 1.0
    
    # 3. 在激增日，看后续的收益行为
    # 计算20日滚动收益std（对应"耀眼波动率"）
    df['ret_abs'] = df['pct_change'].abs() / 100  # 日绝对收益
    df['ret_std_20'] = df.groupby('stock_code')['ret_abs'].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )
    
    # 4. 计算日频"适度因子"
    # 当成交量大时，我们希望收益波动适中
    # 适度 = |绝对收益 - 截面均值| （偏离越大，越不"适度"）
    # 但原文是"越小越好"，所以我们的因子方向需要测试
    # 价格变化后的绝对值偏离度：|abs(ret) - median_ret_abs|
    # 这反映激增日价格是否反应"过度"或"不足"
    df['abs_return'] = df['pct_change'].abs() / 100
    
    # 5. 核心因子构造：
    # F1: volume-driven volatility = volume_surge_ratio * ret_std_20
    # 高成交量且高波动 → 过度冒险
    df['f_vol_vol'] = df['volume_surge_ratio'] * df['ret_std_20']
    
    # F2: normality deviation = abs(volume_surge_ratio - 1) * abs_return
    # 成交量偏离越大+价格变动越大 → 信号越强
    df['f_vol_dev'] = np.abs(df['volume_surge_ratio'] - 1) * df['abs_return']
    
    # 6. 取滚动20日均值合成月频因子
    for fcol in ['f_vol_vol', 'f_vol_dev']:
        df[f'{fcol}_ma20'] = df.groupby('stock_code')[fcol].transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        )
    
    # 因子1和因子2组合 → 适度冒险因子（等权）
    # 因子值越低越好(IC为负表明低值→高收益)
    df['moderate_risk_raw'] = -(df['f_vol_vol_ma20'] + df['f_vol_dev_ma20']) / 2
    
    # 7. 对数市值中性化
    # 需要amount作为市值的代理(中证1000的成分股市值)
    # 或者用 amount * 某个因子
    df['log_amount'] = np.log(df['amount'].clip(lower=1e6))
    
    # 按截面做中性化
    factor_col = 'moderate_risk_raw'
    
    def neutralize(group):
        y = group[factor_col].values
        x = group['log_amount'].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            group[f'{factor_col}_neu'] = np.nan
            return group
        slope = np.cov(x[mask], y[mask])[0, 1] / (np.var(x[mask]) + 1e-10)
        intercept = np.mean(y[mask]) - slope * np.mean(x[mask])
        residual = y.copy()
        residual[mask] = y[mask] - (slope * x[mask] + intercept)
        group[f'{factor_col}_neu'] = residual
        return group
    
    df = df.groupby('date', group_keys=False).apply(neutralize)
    
    # 输出
    out_cols = ['date', 'stock_code', f'{factor_col}_neu', 'volume_surge_ratio', 'is_surge']
    result = df[out_cols].dropna(subset=[f'{factor_col}_neu']).copy()
    result.columns = ['date', 'stock_code', 'factor_value', 'volume_surge_ratio', 'is_surge_flag']
    result['date'] = pd.to_datetime(result['date'])
    return result

if __name__ == '__main__':
    import sys
    
    kline_path = 'data/csi1000_kline_raw.csv'
    output_path = 'data/moderate_risk_v1.csv'
    
    print(f"读取K线数据: {kline_path}")
    df = pd.read_csv(kline_path)
    print(f"原始数据: {df.shape[0]} 行, {df['stock_code'].nunique()} 只股票")
    
    print(f"\n计算适度冒险因子(moderate_risk_v1)...")
    result = calc_moderate_risk_factor(df, window=20)
    
    print(f"\n因子值形状: {result.shape}")
    print(f"日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"非空因子值: {result['factor_value'].notna().sum()}")
    print(f"激增日标记: {result['is_surge_flag'].sum()} 条")
    print(f"\n因子值统计:")
    print(result['factor_value'].describe())
    
    # 保存
    result.to_csv(output_path, index=False)
    print(f"\n因子已保存到: {output_path}")
