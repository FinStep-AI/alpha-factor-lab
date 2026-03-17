#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量重跑全部因子（构建+回测），输出到 output/{factor_id}/
使用最新K线数据（截止到data中最新日期）

步骤：
1. 加载K线数据，预计算所有需要的衍生字段
2. 逐个计算每个因子的factor_values.csv
3. 对每个因子调用 factor_backtest.py 做回测
4. 更新 factors.json 中的 metrics 和 period
"""

import os, sys, time, json, warnings, subprocess
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
BACKTEST_SCRIPT = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts" / "factor_backtest.py"

# ============================================================
# 数据加载
# ============================================================
print("=" * 60)
print("批量因子重算 & 回测")
print("=" * 60)

print("\n[1] 加载K线数据...")
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 加载成分股
codes = set(pd.read_csv(DATA_DIR / "csi1000_codes.csv")['stock_code'].astype(str).str.zfill(6))
kline = kline[kline['stock_code'].isin(codes)].copy()

print(f"  股票: {kline['stock_code'].nunique()}, 交易日: {kline['date'].nunique()}, "
      f"日期范围: {kline['date'].min().date()} ~ {kline['date'].max().date()}, 总行数: {len(kline)}")

# ============================================================
# 预计算衍生字段
# ============================================================
print("\n[2] 预计算衍生字段...")

# 日收益率
kline['return'] = kline['pct_change'].astype(float) / 100.0

# 市值代理
kline['mktcap_proxy'] = kline['amount'] / (kline['turnover'].clip(lower=0.01) / 100)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))

# 成交额对数（用于部分因子的中性化）
kline['log_amount'] = np.log(kline['amount'].clip(lower=1))
kline['log_amount_20d'] = kline.groupby('stock_code')['log_amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 绝对收益率
kline['abs_ret'] = kline['return'].abs()

# 加载指数数据
index_df = pd.read_csv(DATA_DIR / "csi1000_index_daily.csv")
index_df['date'] = pd.to_datetime(index_df['date'])
index_df['mkt_return'] = index_df['pct_change'].astype(float) / 100.0
mkt_map = index_df.set_index('date')['mkt_return'].to_dict()
kline['mkt_return'] = kline['date'].map(mkt_map)

# 基本面数据（如果存在）
fund_path = DATA_DIR / "csi1000_fundamental_cache.csv"
has_fundamental = fund_path.exists()
if has_fundamental:
    fund = pd.read_csv(fund_path)
    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    print(f"  基本面数据: {len(fund)} 行")

print("  衍生字段计算完成")

# ============================================================
# 辅助函数
# ============================================================
def neutralize(factor_s, control_s):
    """OLS回归取残差做中性化"""
    mask = factor_s.notna() & control_s.notna()
    if mask.sum() < 10:
        return factor_s
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(mask.sum()), control_s[mask].values])
    y = factor_s[mask].values
    coef, _, _, _ = lstsq(X, y, rcond=None)
    resid = pd.Series(np.nan, index=factor_s.index)
    resid[mask] = y - X @ coef
    return resid

def cross_section_zscore(s):
    """截面标准化"""
    m, sd = s.mean(), s.std()
    if sd < 1e-10:
        return s * 0
    return (s - m) / sd

def winsorize_mad(s, n=5):
    """MAD缩尾"""
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    lo, hi = med - n * mad, med + n * mad
    return s.clip(lo, hi)

def build_factor_by_date(kline_df, raw_col, neutral_col='log_mktcap', factor_name='factor'):
    """按日截面：缩尾→中性化→标准化，返回 (date, stock_code, factor_value) DataFrame"""
    results = []
    for dt, gdf in kline_df.groupby('date'):
        vals = gdf[raw_col].copy()
        # 缩尾
        vals = winsorize_mad(vals)
        # 中性化
        ctrl = gdf[neutral_col] if neutral_col in gdf.columns else gdf['log_mktcap']
        resid = neutralize(vals, ctrl)
        # 标准化
        resid = cross_section_zscore(resid)
        tmp = pd.DataFrame({'date': dt, 'stock_code': gdf['stock_code'].values, 'factor_value': resid.values})
        results.append(tmp)
    return pd.concat(results, ignore_index=True)


# ============================================================
# 因子定义 & 计算
# ============================================================
FACTORS_CONFIG = {
    'amihud_illiq_v2': {'rebalance': 5, 'forward': 5, 'cost': 0.003, 'direction': 1},
    'shadow_pressure_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'overnight_momentum_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'gap_momentum_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'turnover_decay_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': -1},
    'idio_vol_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'close_location_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'pb_roe_residual_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': -1},
    'beta_elasticity_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'close_vwap_dev_v1': {'rebalance': 20, 'forward': 5, 'cost': 0.003, 'direction': -1},
    'tail_risk_cvar_v1': {'rebalance': 20, 'forward': 5, 'cost': 0.003, 'direction': 1},
    'pv_corr_v1': {'rebalance': 20, 'forward': 20, 'cost': 0.002, 'direction': 1},
    'neg_day_freq_v1': {'rebalance': 10, 'forward': 10, 'cost': 0.003, 'direction': 1},
}


def compute_factor(kline_df, factor_id):
    """计算单个因子的原始值并加到kline_df上"""
    t0 = time.time()
    df = kline_df  # alias
    
    if factor_id == 'amihud_illiq_v2':
        # |ret| / amount(亿), 20日均值, log变换
        df['_amihud_raw'] = df['abs_ret'] / (df['amount'].clip(lower=1) / 1e8)
        df['_factor_raw'] = df.groupby('stock_code')['_amihud_raw'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        df['_factor_raw'] = np.log(df['_factor_raw'].clip(lower=1e-10))
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'shadow_pressure_v1':
        # 上影线比 - 下影线比, 20日均值
        df['_upper_shadow'] = (df['high'] - df[['open','close']].max(axis=1)) / (df['high'] - df['low']).clip(lower=0.01)
        df['_lower_shadow'] = (df[['open','close']].min(axis=1) - df['low']) / (df['high'] - df['low']).clip(lower=0.01)
        df['_shadow_diff'] = df['_upper_shadow'] - df['_lower_shadow']
        df['_factor_raw'] = df.groupby('stock_code')['_shadow_diff'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        neutral_col = 'log_amount_20d'
        
    elif factor_id == 'overnight_momentum_v1':
        # 隔夜收益 = open/prev_close - 1; 日内收益 = close/open - 1
        df['_prev_close'] = df.groupby('stock_code')['close'].shift(1)
        df['_overnight_ret'] = df['open'] / df['_prev_close'] - 1
        df['_intraday_ret'] = df['close'] / df['open'] - 1
        df['_on_sum20'] = df.groupby('stock_code')['_overnight_ret'].transform(
            lambda x: x.rolling(20, min_periods=10).sum())
        df['_id_sum20'] = df.groupby('stock_code')['_intraday_ret'].transform(
            lambda x: x.rolling(20, min_periods=10).sum())
        df['_factor_raw'] = df['_on_sum20'] - df['_id_sum20']
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'gap_momentum_v1':
        # 跳空方向+幅度+时效性的Z-score综合
        df['_prev_close'] = df.groupby('stock_code')['close'].shift(1)
        df['_gap'] = df['open'] / df['_prev_close'] - 1
        df['_gap_dir'] = np.sign(df['_gap'])
        df['_gap_amp'] = df['_gap'].abs()
        # 20日：方向均值 + 幅度均值 + 近期权重(指数衰减)
        df['_gap_dir_20'] = df.groupby('stock_code')['_gap_dir'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        df['_gap_amp_20'] = df.groupby('stock_code')['_gap_amp'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        # 近5日跳空均值 vs 20日：时效性
        df['_gap_5'] = df.groupby('stock_code')['_gap'].transform(
            lambda x: x.rolling(5, min_periods=3).mean())
        df['_gap_20'] = df.groupby('stock_code')['_gap'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        # 复合
        df['_factor_raw'] = df['_gap_dir_20'] + df['_gap_amp_20'] + (df['_gap_5'] - df['_gap_20'])
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'turnover_decay_v1':
        # -log(MA5_turnover / MA20_turnover)
        df['_turn_ma5'] = df.groupby('stock_code')['turnover'].transform(
            lambda x: x.rolling(5, min_periods=3).mean())
        df['_turn_ma20'] = df.groupby('stock_code')['turnover'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        df['_factor_raw'] = -np.log((df['_turn_ma5'] / df['_turn_ma20'].clip(lower=0.01)).clip(lower=0.01))
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'idio_vol_v1':
        # 个股对市场回归残差的20日滚动波动率
        df['_excess'] = df['return'] - df['mkt_return'].fillna(0)
        # 简化：用excess return的20日std近似idio vol
        df['_factor_raw'] = df.groupby('stock_code')['_excess'].transform(
            lambda x: x.rolling(20, min_periods=15).std())
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'close_location_v1':
        # (close - low) / (high - low), 20日均值
        df['_cl'] = (df['close'] - df['low']) / (df['high'] - df['low']).clip(lower=0.01)
        df['_factor_raw'] = df.groupby('stock_code')['_cl'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'pb_roe_residual_v1':
        # 需要基本面数据，截面回归 ln(PB) ~ ROE 残差
        if not has_fundamental:
            print(f"  ⚠️ {factor_id}: 缺少基本面数据，跳过")
            return None
        # 将最新可用的PB/ROE merge到K线
        fund_latest = fund.sort_values('report_date').groupby('stock_code').last().reset_index()
        fund_latest = fund_latest[['stock_code', 'bps', 'roe']].copy()
        fund_latest = fund_latest.dropna(subset=['bps', 'roe'])
        # 近似PB: close / bps
        # 需要每日merge
        df_merged = df.merge(fund_latest, on='stock_code', how='left')
        df_merged['_pb'] = df_merged['close'] / df_merged['bps'].clip(lower=0.01)
        df_merged['_ln_pb'] = np.log(df_merged['_pb'].clip(lower=0.01))
        df_merged['_roe'] = df_merged['roe']
        # 每日截面回归
        results = []
        for dt, gdf in df_merged.groupby('date'):
            mask = gdf['_ln_pb'].notna() & gdf['_roe'].notna()
            sub = gdf[mask]
            if len(sub) < 30:
                tmp = pd.DataFrame({'date': dt, 'stock_code': gdf['stock_code'].values, 
                                   'factor_value': np.nan})
                results.append(tmp)
                continue
            X = np.column_stack([np.ones(len(sub)), sub['_roe'].values])
            y = sub['_ln_pb'].values
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid_vals = y - X @ coef
            # 中性化 by mktcap
            resid_s = pd.Series(resid_vals, index=sub.index)
            resid_n = neutralize(resid_s, sub['log_mktcap'])
            resid_z = cross_section_zscore(resid_n)
            tmp = pd.DataFrame({'date': dt, 'stock_code': sub['stock_code'].values,
                               'factor_value': resid_z.values})
            results.append(tmp)
        elapsed = time.time() - t0
        print(f"  {factor_id}: {elapsed:.1f}s")
        return pd.concat(results, ignore_index=True)
        
    elif factor_id == 'beta_elasticity_v1':
        # Beta_20d - Beta_60d
        # 用滚动协方差/方差近似
        df['_cov_20'] = df.groupby('stock_code').apply(
            lambda g: g['return'].rolling(20, min_periods=15).cov(g['mkt_return'])).reset_index(level=0, drop=True)
        df['_var_20'] = df.groupby('stock_code')['mkt_return'].transform(
            lambda x: x.rolling(20, min_periods=15).var())
        df['_beta_20'] = df['_cov_20'] / df['_var_20'].clip(lower=1e-10)
        
        df['_cov_60'] = df.groupby('stock_code').apply(
            lambda g: g['return'].rolling(60, min_periods=40).cov(g['mkt_return'])).reset_index(level=0, drop=True)
        df['_var_60'] = df.groupby('stock_code')['mkt_return'].transform(
            lambda x: x.rolling(60, min_periods=40).var())
        df['_beta_60'] = df['_cov_60'] / df['_var_60'].clip(lower=1e-10)
        
        df['_factor_raw'] = df['_beta_20'] - df['_beta_60']
        neutral_col = 'log_mktcap'
        
    elif factor_id == 'close_vwap_dev_v1':
        # VWAP = amount / volume; 偏离 = (close - vwap) / vwap, 20日均值
        df['_vwap'] = df['amount'] / df['volume'].clip(lower=1)
        df['_dev'] = (df['close'] - df['_vwap']) / df['_vwap'].clip(lower=0.01)
        df['_factor_raw'] = df.groupby('stock_code')['_dev'].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        neutral_col = 'log_amount_20d'
        
    elif factor_id == 'tail_risk_cvar_v1':
        # 10日内最差2天日收益均值，取负（高CVaR=高尾部风险）
        def bottom_2_mean(x):
            if len(x.dropna()) < 5:
                return np.nan
            return x.nsmallest(2).mean()
        df['_cvar_raw'] = df.groupby('stock_code')['return'].transform(
            lambda x: x.rolling(10, min_periods=5).apply(bottom_2_mean, raw=False))
        df['_factor_raw'] = -df['_cvar_raw']  # 取负：高值=高风险
        neutral_col = 'log_amount_20d'
        
    elif factor_id == 'pv_corr_v1':
        # 日收益与成交量变化的20日滚动相关
        df['_vol_pct'] = df.groupby('stock_code')['volume'].pct_change()
        df['_factor_raw'] = df.groupby('stock_code').apply(
            lambda g: g['return'].rolling(20, min_periods=10).corr(g['_vol_pct'])).reset_index(level=0, drop=True)
        neutral_col = 'log_amount_20d'
        
    elif factor_id == 'neg_day_freq_v1':
        # 10日内 return <= -3% 的天数占比
        df['_neg_flag'] = (df['return'] <= -0.03).astype(float)
        df['_factor_raw'] = df.groupby('stock_code')['_neg_flag'].transform(
            lambda x: x.rolling(10, min_periods=5).mean())
        neutral_col = 'log_amount_20d'
    
    else:
        print(f"  ⚠️ 未知因子: {factor_id}")
        return None
    
    # 通用后处理：截面缩尾→中性化→标准化
    result = build_factor_by_date(df, '_factor_raw', neutral_col, factor_id)
    elapsed = time.time() - t0
    print(f"  {factor_id}: {elapsed:.1f}s, {result['factor_value'].notna().sum()} 有效值")
    
    # 清理临时列
    temp_cols = [c for c in df.columns if c.startswith('_')]
    df.drop(columns=temp_cols, inplace=True, errors='ignore')
    
    return result


# ============================================================
# 主流程
# ============================================================
factor_ids = list(FACTORS_CONFIG.keys())
print(f"\n[3] 开始构建 {len(factor_ids)} 个因子...")

# 准备returns数据（close价格数据，pivot成 date × stock）
print("  准备收益率矩阵...")
returns_path = DATA_DIR / "csi1000_returns_matrix.csv"
ret_pivot = kline.pivot_table(index='date', columns='stock_code', values='close')
ret_pivot.index = ret_pivot.index.strftime('%Y-%m-%d')
ret_pivot.to_csv(returns_path)
print(f"  收益率矩阵: {ret_pivot.shape}")

all_results = {}
failed = []

for i, fid in enumerate(factor_ids, 1):
    print(f"\n--- [{i}/{len(factor_ids)}] {fid} ---")
    try:
        factor_df = compute_factor(kline, fid)
        if factor_df is None:
            failed.append(fid)
            continue
        
        # 保存因子值
        fv_path = DATA_DIR / f"factor_{fid}.csv"
        factor_df.to_csv(fv_path, index=False)
        
        # 转为回测引擎需要的格式（pivot: date × stock_code）
        fv_pivot_path = OUTPUT_DIR / fid / "factor_values_pivot.csv"
        os.makedirs(OUTPUT_DIR / fid, exist_ok=True)
        
        fv_pivot = factor_df.pivot_table(index='date', columns='stock_code', values='factor_value')
        fv_pivot.index = pd.to_datetime(fv_pivot.index).strftime('%Y-%m-%d')
        fv_pivot.to_csv(fv_pivot_path)
        
        all_results[fid] = {'factor_path': str(fv_pivot_path), 'rows': len(factor_df)}
        
    except Exception as e:
        print(f"  ❌ {fid} 失败: {e}")
        import traceback; traceback.print_exc()
        failed.append(fid)

print(f"\n\n[4] 因子构建完成: 成功 {len(all_results)}, 失败 {len(failed)}")
if failed:
    print(f"  失败列表: {failed}")

# ============================================================
# 批量回测
# ============================================================
print(f"\n[5] 开始批量回测...")

backtest_results = {}

for fid, info in all_results.items():
    cfg = FACTORS_CONFIG[fid]
    out_dir = OUTPUT_DIR / fid
    report_path = out_dir / "backtest_report.json"
    
    cmd = [
        sys.executable, str(BACKTEST_SCRIPT),
        '--factor', info['factor_path'],
        '--returns', str(returns_path),
        '--is-price',
        '--n-groups', '5',
        '--rebalance-freq', str(cfg['rebalance']),
        '--forward-days', str(cfg['forward']),
        '--cost', str(cfg['cost']),
        '--factor-name', fid,
        '--output-report', str(report_path),
        '--output-dir', str(out_dir),
    ]
    
    print(f"\n  回测 {fid} (rebal={cfg['rebalance']}, fwd={cfg['forward']}, cost={cfg['cost']})...")
    t0 = time.time()
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        
        if r.returncode != 0:
            print(f"  ❌ 回测失败 ({elapsed:.0f}s): {r.stderr[-500:]}")
            failed.append(f"{fid}_backtest")
            continue
        
        # 读取回测结果
        with open(report_path) as f:
            rpt = json.load(f)
        
        metrics = rpt.get('metrics', {})
        ic = metrics.get('ic_mean', 0)
        ic_t = metrics.get('ic_t_stat', 0)
        mono = metrics.get('monotonicity', 0)
        ls_sharpe = metrics.get('long_short_sharpe', 0)
        g5_sharpe = metrics.get('group_metrics', {}).get('group_5', {}).get('sharpe_ratio', 0)
        
        print(f"  ✅ {fid}: IC={ic:.4f}(t={ic_t:.2f}), mono={mono:.2f}, LS_Sharpe={ls_sharpe:.2f}, G5_Sharpe={g5_sharpe:.2f} ({elapsed:.0f}s)")
        
        backtest_results[fid] = {
            'report': rpt,
            'metrics': metrics,
        }
    except subprocess.TimeoutExpired:
        print(f"  ❌ 回测超时 (>600s)")
        failed.append(f"{fid}_backtest")
    except Exception as e:
        print(f"  ❌ 回测异常: {e}")
        failed.append(f"{fid}_backtest")

# ============================================================
# 更新 factors.json
# ============================================================
print(f"\n[6] 更新 factors.json...")

factors_path = BASE_DIR / "factors.json"
with open(factors_path) as f:
    factors = json.load(f)

date_max = kline['date'].max().strftime('%Y-%m-%d')

for fc in factors:
    fid = fc['id']
    if fid not in backtest_results:
        continue
    
    rpt = backtest_results[fid]['report']
    metrics = rpt['metrics']
    
    # 更新 period
    fc['period'] = rpt.get('period', fc.get('period', ''))
    fc['updated'] = time.strftime('%Y-%m-%d')
    
    # 更新 metrics
    fc['metrics'] = {
        'ic_mean': metrics.get('ic_mean', 0),
        'ic_std': metrics.get('ic_std', 0),
        'ic_t': metrics.get('ic_t_stat', 0),
        'ic_positive_ratio': metrics.get('ic_positive_pct', 0),
        'rank_ic': metrics.get('rank_ic_mean', 0),
        'rank_ic_std': metrics.get('rank_ic_std', 0),
        'ir': metrics.get('ir', 0),
        'ic_count': metrics.get('ic_count', 0),
        'long_short_sharpe': metrics.get('long_short_sharpe', 0),
        'long_short_mdd': metrics.get('long_short_mdd', 0),
        'monotonicity': metrics.get('monotonicity', 0),
    }
    
    # 提取分组指标
    gm = metrics.get('group_metrics', {})
    fc['metrics']['group_sharpe'] = [
        gm.get(f'group_{i}', {}).get('sharpe_ratio', 0) for i in range(1, 6)
    ]
    fc['metrics']['group_returns_annualized'] = [
        gm.get(f'group_{i}', {}).get('annualized_return', 0) for i in range(1, 6)
    ]
    fc['metrics']['group_mdd'] = [
        gm.get(f'group_{i}', {}).get('max_drawdown', 0) for i in range(1, 6)
    ]
    
    # NAV / IC 数据路径
    fc['nav_data'] = f"output/{fid}/cumulative_returns.json"
    fc['ic_data'] = f"output/{fid}/ic_series.json"

with open(factors_path, 'w') as f:
    json.dump(factors, f, indent=2, ensure_ascii=False)

print(f"  factors.json 已更新 ({len(backtest_results)} 个因子)")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print(f"  K线截止: {date_max}")
print(f"  因子构建成功: {len(all_results)}")
print(f"  回测成功: {len(backtest_results)}")
print(f"  失败: {failed if failed else '无'}")
print()
for fid, info in backtest_results.items():
    m = info['metrics']
    ic = m.get('ic_mean', 0)
    ic_t = m.get('ic_t_stat', 0)
    mono = m.get('monotonicity', 0)
    ls = m.get('long_short_sharpe', 0)
    g5 = m.get('group_metrics', {}).get('group_5', {}).get('sharpe_ratio', 0)
    print(f"  {fid:25s} IC={ic:+.4f}(t={ic_t:5.2f}) mono={mono:.2f} LS_Sharpe={ls:.2f} G5_Sharpe={g5:.2f}")

print("\n✅ 批量重跑完成！")