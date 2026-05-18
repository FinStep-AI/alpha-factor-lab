"""
APB Factor (东方证券 买卖压力量价背离因子 v1)
=============================================

核心思路 (东方证券 APB 指标):
  理性投资者从均衡建仓、逢低买入、精准出场。
  建仓后价格低位处成交量最大 → 均值系统上便会底部成交量压缩,
  解构竟尔: 低价格, 高低量, 也是VWAP向低价,
  
  用量均价即:
  
  APB = ln( mean(VWAP, N) / VWAP_avg_weighted )
  
  当APB > 0:
  → 价格均价 > 成交量加权均价
  → 价格上涨时成交量较小, 价格下跌时成交量较大
  → 智慧资金低吸证据 → 正面 → 未来正预期
  → 说明近期正在"低吸式建仓", 后续走强概率更大

  当APB < 0:
  → 价格均价 < 成交量加权均价
  → 价格上涨时成交活跃 → 追涨/出货行为
  → 智慧资金出货证据 → 负面 → 未来负预期

因子ID: apb_v1
空头: 东方证券《基于量价关系度量股票的买卖压力》(2019)
http://www.wufls.com/20191029.html

Barra风格: MICRO (价量供求结构因子)
"""

import numpy as np
import pandas as pd
import warnings
from numpy.linalg import lstsq
warnings.filterwarnings('ignore')


def compute_vwap(gdf: pd.DataFrame) -> pd.Series:
    """单只股票逐日VWAP = 成交额 / 成交量"""
    vwap = gdf['amount'] / gdf['volume'].clip(1)  # 防零
    vwap.name = 'vwap'
    return vwap


def compute_apb(gdf: pd.DataFrame, N: int = 20, min_valid: int = 12) -> pd.Series:
    """单个股票序列: 滚动N日APB因子"""
    vwap = compute_vwap(gdf)
    
    # Simple last-N-day average of daily VWAP
    avg_simple = vwap.rolling(N, min_periods=min_valid).mean()
    
    # Volume-weighted average of daily VWAP over N days
    # = sum(vol_i × VWAP_i) / sum(vol_i)
    vol = gdf['volume'].values.astype(float)
    vwap_vals = vwap.values.astype(float)
    
    # 滚动 computation
    n = len(gdf)
    vwap_weighted = np.full(n, np.nan)
    
    for i in range(N-1, n):
        idx_start = i - N + 1
        valid = np.isfinite(vol[idx_start:i+1]) & np.isfinite(vwap_vals[idx_start:i+1])
        if valid.sum() >= min_valid:
            vs = vol[idx_start:i+1][valid]
            ws = vwap_vals[idx_start:i+1][valid]
            total_vol = vs.sum()
            if total_vol > 0:
                vwap_weighted[i] = (vs * ws).sum() / total_vol
    
    vwap_weighted_s = pd.Series(vwap_weighted, index=gdf.index)
    
    # APB = ln(simple_avg / volume_weighted_avg)
    # Handle short periods where one side has fewer days
    # Use only valid rows where both are available
    apb_raw = np.log(avg_simple / vwap_weighted_s.clip(1e-9))
    
    return apb_raw


def neutralize_ols(residual_col: str, neutralizer: pd.Series) -> pd.Series:
    """OLScross-sectional residuals: neutralize residual_col vs neutralizer (mati per day)"""
    residuals = np.full(len(residual_col), np.nan)
    for dt, idxs in residual_col.groupby(level=0).groups.items():
        y = residual_col.iloc[idxs].values.astype(float)
        x = neutralizer.iloc[idxs].values.astype(float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() >= 20:
            xm = x[mask].mean()
            ym = y[mask].mean()
            xd = x[mask] - xm
            denom = (xd**2).sum()
            if denom > 1e-12:
                beta = (xd * (y[mask] - ym)).sum() / denom
                alpha = ym - beta * xm
                y_out = np.full(len(y), np.nan)
                y_out[mask] = y[mask] - (alpha + beta * x[mask])
        
    return pd.Series(residuals, index=residual_col.index)


def main():
    import sys
    from pathlib import Path
    
    BASE = Path(__file__).resolve().parent.parent
    KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
    OUTPUT_PATH = BASE / "data" / "csi1000_apb_v1.csv"
    
    print(f"[1] Loading data: {KLINE_PATH}")
    df = pd.read_csv(KLINE_PATH, parse_dates=['date'])
    print(f"    {df['stock_code'].nunique()} stocks, {df['date'].nunique()} trading days")
    print(f"    Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Compute log_market_cap proxy for neutralization
    # amount is total transaction value; use sqrt amount as rough size proxy
    df['log_amount_20d'] = np.log(
        df.groupby('stock_code')['amount'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ).clip(1)
    )
    
    N = 20
    MIN_VALID = 12
    
    print(f"[2] Computing APB factor (N={N} days, min_periods={MIN_VALID})...")
    results = []
    
    for code, gdf in df.groupby('stock_code', sort=True):
        gdf = gdf.sort_values('date')
        apb = compute_apb(gdf, N=N, min_valid=MIN_VALID)
        # Take last row (most recent) for this stock
        # Actually collect all valid rows
        tmp = pd.DataFrame({
            'date': gdf['date'].values,
            'stock_code': code,
            'apb_raw': apb.values,
            'log_amount_20d': gdf['log_amount_20d'].values,
        }).dropna(subset=['apb_raw'])
        results.append(tmp)
    
    all_data = pd.concat(results, ignore_index=True)
    print(f"    Raw factor: {len(all_data)} rows, {all_data['date'].nunique()} dates")
    
    # Winsorize 5% per day
    print("[3] Winsorize (5%)...")
    def winsorize(series):
        lower = series.quantile(0.05)
        upper = series.quantile(0.95)
        return series.clip(lower, upper)
    
    all_data['apb_winsor'] = all_data.groupby('date')['apb_raw'].transform(winsorize)
    
    # OLS cross-sectional neutralize vs log_amount_20d
    print("[4] Market-cap neutralization (OLS)...")
    
    neutralized = np.full(len(all_data), np.nan)
    for dt, mask in all_data.groupby('date').groups.items():
        idx = list(mask)
        y = all_data.loc[idx, 'apb_winsor'].values.astype(float)
        x = all_data.loc[idx, 'log_amount_20d'].values.astype(float)
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 20:
            continue
        xm = x[valid].mean()
        ym = y[valid].mean()
        xd = x[valid] - xm
        denom = (xd**2).sum()
        if denom > 1e-10:
            beta = (xd * (y[valid] - ym)).sum() / denom
            alpha = ym - beta * xm
            y_out = np.full(len(valid), np.nan)
            y_out[valid] = y[valid] - (alpha + beta * x[valid])
            # Fill all rows (including invalid)
            full_out = np.full(len(y), np.nan)
            full_out[valid] = y_out[valid]
            neutralized[idx] = full_out
    
    all_data['apb_neutral'] = neutralized
    
    # Second winsorize for residuals
    all_data['apb_neutral'] = all_data.groupby('date')['apb_neutral'].transform(
        lambda x: x.clip(x.quantile(0.05), x.quantile(0.95))
    )
    
    # Z-score normalize
    all_data['apb_neutral'] = all_data.groupby('date')['apb_neutral'].transform(
        lambda x: (x - x.mean()) / x.std().clip(1e-9)
    )
    
    # Final output  
    output = all_data[['date', 'stock_code', 'apb_neutral']].rename(
        columns={'apb_neutral': 'factor_value'}
    )
    output = output.dropna(subset=['factor_value'])
    output.to_csv(OUTPUT_PATH, index=False)
    
    print(f"[5] Output: {OUTPUT_PATH}")
    print(f"    {len(output)} rows, date range: {output['date'].min().date()} ~ {output['date'].max().date()}")
    print(f"    Mean: {output['factor_value'].mean():.4f}, Std: {output['factor_value'].std():.4f}")
    print(f"    Per-day stocks: {output.groupby('date')['stock_code'].count().describe().round(1).to_dict()}")
    print("Done!")


if __name__ == '__main__':
    main()
