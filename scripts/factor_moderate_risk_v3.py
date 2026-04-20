"""
适度冒险因子 v3 (moderate_risk_v3)
信号增强版：更强调激增时刻的波动率信号

策略调整：
1. 激增阈值降低：volume > MA20(volume) + 0.5*std（捕获更多激增日）
2. 波动率下降使用：std20(|ret|)比直接使用 abs(ret)（更稳健）
3. 因子直接使用：-log(volume_spike) × std20(|ret|)
   （取对数压缩极端值，但保持单调性）
"""
import numpy as np
import pandas as pd

def calc_moderate_risk_v3(df_kline: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df_kline.copy()
    
    # 1. 成交量特征
    df['vol_ma20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    df['vol_std20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )
    
    # 成交量偏离（激增代理）
    # spike_ratio > 1 表示超过均值+1std
    df['spike_ratio'] = np.where(
        df['vol_ma20'] > 0,
        df['vol_ma20'] + df['vol_std20'],  # 阈值 = MA + 1std
        np.nan
    )
    # 实际激增程度：volume / 阈值 (≥1 为激增)
    df['volume_spike'] = np.where(
        df['spike_ratio'] > 0,
        df['volume'] / df['spike_ratio'],
        np.nan
    )
    # log变换，压缩尺度
    df['log_spike'] = np.log(df['volume_spike'].clip(lower=1e-6) + 1e-6)
    
    # 2. 波动率特征
    df['ret_abs'] = df['pct_change'].abs() / 100
    df['ret_std20'] = df.groupby('stock_code')['ret_abs'].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )
    # 也计算20日均收益绝对值
    df['ret_abs20'] = df.groupby('stock_code')['ret_abs'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    
    # 3. 两个子因子
    # F_vol: 成交量激增程度 → 越高信号越强（市场关注度高）
    # F_std: 波动率水平 → 越高越好（活跃度高）
    # 合成：高且活跃 → 高收益（反向信号）
    df['f_vol'] = df['log_spike']  # 已为正值
    df['f_std'] = df['ret_std20']
    
    # 4. 等权合成（越高越好，即富尔塔是反向的）
    df['factor_raw'] = -(df['f_vol'] + df['f_std']) / 2
    
    # 5. 20日滚动均值（月频）
    df['factor_ma20'] = df.groupby('stock_code')['factor_raw'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    
    # 6. 市值中性化
    df['log_amount'] = np.log(df['amount'].clip(lower=1e6))
    factor_col = 'factor_ma20'
    
    def neutralize(group):
        y = group[factor_col].values
        x = group['log_amount'].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            group['factor_neu'] = np.nan
            return group
        slope = np.cov(x[mask], y[mask])[0, 1] / (np.var(x[mask]) + 1e-10)
        intercept = np.mean(y[mask]) - slope * np.mean(x[mask])
        residual = y.copy()
        residual[mask] = y[mask] - (slope * x[mask] + intercept)
        group['factor_neu'] = residual
        return group
    
    df = df.groupby('date', group_keys=False).apply(neutralize)
    
    result = df[['date', 'stock_code', 'factor_neu', 'volume_spike', 'log_spike', 'ret_std20']].dropna(subset=['factor_neu']).copy()
    result.columns = ['date', 'stock_code', 'factor_value', 'volume_spike', 'log_spike', 'ret_std20']
    result['date'] = pd.to_datetime(result['date'])
    return result

if __name__ == '__main__':
    kline_path = 'data/csi1000_kline_raw.csv'
    output_path = 'data/moderate_risk_v3.csv'
    
    df = pd.read_csv(kline_path)
    result = calc_moderate_risk_v3(df, window=20)
    
    print(f"因子值:\n{result['factor_value'].describe()}")
    result.to_csv(output_path, index=False)
    print(f"\n已保存: {output_path}")
