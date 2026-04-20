"""
适度冒险因子 v2 (moderate_risk_v2)
优化版本：基于更严谨的日频近似

核心逻辑（基于研报原文的简化版）：
- 成交量激增日：volume > MA20(volume) + 1*std_volume
- 在激增日，计算"耀眼波动率"：当日收益率对过去20日波动率的偏离
  = abs(ret) / std20(ret_abs)
- 这个值越低 → 激增但价格稳定（反应不足/适度）
- 这个值越高 → 激增但价格剧烈变动（反应过度）
- 适度冒险 = 1 - 标准化后的"过度"程度

简化版：
factor = -std20(|ret|)  # 波动率越低越好（反应越稳定）
     × volume_spike  # 激增日的成交量放大程度
然后在滚动20日上做了中和:

研报原文因子ICRank=-8.89%，说明因子与未来收益钝角相关。负值意味着低的因子值与高收益配对。
对于我们的回测数据，分组显示Monotonic: -0.9000以及IC=-0.023 (t=1.46)，说明因子看起来完全是无信号的。

但对于"是IC与分组表现的正负界定有误"的可能性，若Monotonic接近-1，实际上暗含因子与未来收益的关系。
"""
import numpy as np
import pandas as pd

def calc_moderate_risk_v2(df_kline: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    适度冒险因子 v2 - 纯粹信号版
    
    简化构造：直接取"成交量增强 × 波动率压缩"的交互
    volume_spike = volume / MA20(volume)
    ret_vol = std20(|ret|)
    factor = -(volume_spike × ret_vol) 
    
    逻辑：交易量大但波动小 = "适度冒险" = 后验收益高
    """
    df = df_kline.copy()
    
    # 1. 成交量增强比率
    df['volume_ma20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    df['volume_spike'] = np.where(
        df['volume_ma20'] > 0,
        df['volume'] / df['volume_ma20'],
        np.nan
    )
    
    # 2. 20日滚动收益波动率
    df['ret_abs'] = df['pct_change'].abs() / 100
    df['ret_std20'] = df.groupby('stock_code')['ret_abs'].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )
    
    # 3. 核心：成交量激增 × 波动率
    # 狂热但稳定 = 适度冒险（正期待收益）
    # 所以负号表示：值越低越好
    df['factor_raw'] = -df['volume_spike'] * df['ret_std20']
    
    # 4. 20日滚动平均（降频到月频）
    df['factor_ma20'] = df.groupby('stock_code')['factor_raw'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    
    # 5. 市值中性化
    df['log_amount'] = np.log(df['amount'].clip(lower=1e6))
    
    factor_col = 'factor_ma20'
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
    out_cols = ['date', 'stock_code', f'{factor_col}_neu', 'volume_spike', 'ret_std20']
    result = df[out_cols].dropna(subset=[f'{factor_col}_neu']).copy()
    result.columns = ['date', 'stock_code', 'factor_value', 'volume_spike', 'ret_std20']
    result['date'] = pd.to_datetime(result['date'])
    return result

if __name__ == '__main__':
    import sys
    
    kline_path = 'data/csi1000_kline_raw.csv'
    output_path = 'data/moderate_risk_v2.csv'
    
    df = pd.read_csv(kline_path)
    print(f"原始数据: {df.shape[0]} 行, {df['stock_code'].nunique()} 只股票")
    
    result = calc_moderate_risk_v2(df, window=20)
    print(f"\n因子值形状: {result.shape}")
    print(f"日期范围: {result['date'].min()} ~ {result['date'].max()}")
    print(f"因子统计:\n{result['factor_value'].describe()}")
    
    result.to_csv(output_path, index=False)
    print(f"\n已保存: {output_path}")
