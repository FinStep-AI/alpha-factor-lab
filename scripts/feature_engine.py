#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程 Pipeline (Feature Engineering Engine)

把原始K线 + 因子CSV → 统一特征矩阵 + 标签
供 ML/DL 选股模型使用

输出格式: DataFrame with columns:
  [date, stock_code, feat_1, feat_2, ..., label_5d, label_10d, label_20d]

Features:
  1. 原始因子值 (10个)
  2. 因子截面 rank (百分位)
  3. 因子滞后值 (T-5, T-20)
  4. 因子变化率 (T vs T-5, T vs T-20)
  5. K线衍生特征 (技术指标 + 滚动统计量)
  6. 因子交互特征 (top因子 ratio/product)

Labels:
  未来N日超额收益 (相对中证1000指数)

Usage:
  python3 feature_engine.py --output data/ml_features.pkl [--refresh]
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 活跃因子列表 (从 factors.json 读取)
FACTORS_JSON = os.path.join(BASE_DIR, 'factors.json')

# 数据文件
KLINE_FILE = os.path.join(DATA_DIR, 'csi1000_kline_raw.csv')
RETURNS_FILE = os.path.join(DATA_DIR, 'csi1000_returns.csv')
INDEX_FILE = os.path.join(DATA_DIR, 'csi1000_index_daily.csv')

# 标签前瞻期
FORWARD_DAYS = [5, 10, 20]

# 滞后期
LAG_DAYS = [5, 20]

# 滚动窗口
ROLLING_WINDOWS = [5, 10, 20, 60]


def load_active_factors():
    """从 factors.json 加载活跃因子ID和对应CSV路径"""
    with open(FACTORS_JSON) as f:
        factors = json.load(f)
    
    factor_map = {}
    for fac in factors:
        fid = fac.get('id', fac.get('factor_id', fac.get('name', '')))
        if not fid:
            continue
        csv_path = os.path.join(DATA_DIR, f'factor_{fid}.csv')
        if os.path.exists(csv_path):
            factor_map[fid] = csv_path
        else:
            print(f"  ⚠️  因子 {fid} 的CSV不存在，跳过")
    
    return factor_map


def load_factor_data(factor_map):
    """加载所有因子数据并合并"""
    print(f"📊 加载 {len(factor_map)} 个因子...")
    
    merged = None
    for fid, csv_path in factor_map.items():
        df = pd.read_csv(csv_path)
        # 统一列名: 第三列为因子值
        val_col = [c for c in df.columns if c not in ('date', 'stock_code')][0]
        df = df.rename(columns={val_col: f'f_{fid}'})
        df = df[['date', 'stock_code', f'f_{fid}']]
        
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=['date', 'stock_code'], how='outer')
    
    merged = merged.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"  因子矩阵: {merged.shape}, 日期范围: {merged['date'].min()} ~ {merged['date'].max()}")
    return merged


def load_kline_data():
    """加载K线数据"""
    print("📈 加载K线数据...")
    kl = pd.read_csv(KLINE_FILE)
    kl = kl.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"  K线: {kl.shape}, 股票数: {kl['stock_code'].nunique()}")
    return kl


def load_index_data():
    """加载指数日线数据"""
    print("📉 加载中证1000指数...")
    idx = pd.read_csv(INDEX_FILE)
    idx = idx.sort_values('date').reset_index(drop=True)
    idx = idx.rename(columns={'close': 'index_close', 'pct_change': 'index_ret'})
    idx['index_ret'] = idx['index_ret'] / 100.0  # 百分比转小数
    return idx[['date', 'index_close', 'index_ret']]


def build_kline_features(kl):
    """从K线数据构造技术特征"""
    print("🔧 构建K线衍生特征...")
    
    df = kl.copy()
    df = df.sort_values(['stock_code', 'date'])
    
    features = df[['date', 'stock_code']].copy()
    
    # --- 基础价量特征 ---
    # 日收益率
    features['ret_1d'] = df.groupby('stock_code')['close'].pct_change()
    
    # 换手率
    features['turnover'] = df['turnover']
    
    # 振幅
    features['amplitude'] = df['amplitude']
    
    # 成交额(对数)
    features['log_amount'] = np.log1p(df['amount'])
    
    # --- 滚动统计量 ---
    for w in ROLLING_WINDOWS:
        grp = df.groupby('stock_code')
        
        # 收益率滚动均值/标准差/偏度
        ret = grp['pct_change'].transform(lambda x: x / 100.0)
        features[f'ret_mean_{w}d'] = ret.groupby(df['stock_code']).transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 3)).mean()
        )
        features[f'ret_std_{w}d'] = ret.groupby(df['stock_code']).transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 3)).std()
        )
        if w >= 20:
            features[f'ret_skew_{w}d'] = ret.groupby(df['stock_code']).transform(
                lambda x: x.rolling(w, min_periods=w//2).skew()
            )
            features[f'ret_kurt_{w}d'] = ret.groupby(df['stock_code']).transform(
                lambda x: x.rolling(w, min_periods=w//2).kurt()
            )
        
        # 换手率滚动均值
        features[f'turnover_mean_{w}d'] = df.groupby('stock_code')['turnover'].transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 3)).mean()
        )
        
        # 成交额滚动均值
        features[f'amount_mean_{w}d'] = df.groupby('stock_code')['amount'].transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 3)).mean()
        )
    
    # --- 技术指标 ---
    # RSI(14)
    delta = df.groupby('stock_code')['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df['stock_code']).transform(lambda x: x.rolling(14, min_periods=7).mean())
    avg_loss = loss.groupby(df['stock_code']).transform(lambda x: x.rolling(14, min_periods=7).mean())
    features['rsi_14'] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
    
    # MACD
    ema12 = df.groupby('stock_code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = df.groupby('stock_code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    dif = ema12 - ema26
    dea = dif.groupby(df['stock_code']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    features['macd_dif'] = dif
    features['macd_dea'] = dea
    features['macd_hist'] = 2 * (dif - dea)
    
    # 布林带宽度
    ma20 = df.groupby('stock_code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    std20 = df.groupby('stock_code')['close'].transform(lambda x: x.rolling(20, min_periods=10).std())
    features['boll_width'] = (2 * std20) / ma20.replace(0, np.nan)
    features['boll_pct'] = (df['close'] - (ma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    
    # 均线乖离率
    for ma_n in [5, 10, 20, 60]:
        ma = df.groupby('stock_code')['close'].transform(
            lambda x: x.rolling(ma_n, min_periods=max(ma_n//2, 3)).mean()
        )
        features[f'bias_{ma_n}'] = (df['close'] - ma) / ma.replace(0, np.nan)
    
    # 量比 (今日成交量 / 过去5日均量)
    vol_ma5 = df.groupby('stock_code')['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    features['vol_ratio'] = df['volume'] / vol_ma5.replace(0, np.nan)
    
    # 价量相关性 (20日)
    features['pv_corr_20'] = df.groupby('stock_code').apply(
        lambda g: g['close'].pct_change().rolling(20, min_periods=10).corr(g['volume'].pct_change())
    ).reset_index(level=0, drop=True)
    
    # 最高/最低价位置
    high_max_20 = df.groupby('stock_code')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
    low_min_20 = df.groupby('stock_code')['low'].transform(lambda x: x.rolling(20, min_periods=10).min())
    price_range = (high_max_20 - low_min_20).replace(0, np.nan)
    features['price_position_20'] = (df['close'] - low_min_20) / price_range
    
    n_features = len([c for c in features.columns if c not in ('date', 'stock_code')])
    print(f"  K线衍生特征: {n_features} 个")
    return features


def add_factor_derived_features(df, factor_cols):
    """因子衍生特征: 截面rank、滞后、变化率"""
    print("🔧 构建因子衍生特征...")
    
    n_before = len([c for c in df.columns if c not in ('date', 'stock_code')])
    
    for fcol in factor_cols:
        # 截面百分位 rank
        df[f'{fcol}_rank'] = df.groupby('date')[fcol].rank(pct=True)
        
        # 滞后值和变化率
        for lag in LAG_DAYS:
            df[f'{fcol}_lag{lag}'] = df.groupby('stock_code')[fcol].shift(lag)
            # 变化率 = (当前 - 滞后) / |滞后|
            lag_val = df[f'{fcol}_lag{lag}']
            df[f'{fcol}_chg{lag}'] = (df[fcol] - lag_val) / lag_val.abs().replace(0, np.nan)
    
    n_after = len([c for c in df.columns if c not in ('date', 'stock_code')])
    print(f"  因子衍生特征: +{n_after - n_before} 个 (截面rank + 滞后 + 变化率)")
    return df


def add_interaction_features(df, factor_cols):
    """因子交互特征: top因子对的 ratio 和 product"""
    print("🔧 构建因子交互特征...")
    
    # 选最重要的因子对做交互 (避免特征爆炸)
    # 优先选不同 Barra 风格的因子
    key_factors = [c for c in factor_cols if any(k in c for k in 
                   ['amihud', 'shadow', 'overnight', 'gap', 'turnover'])][:5]
    
    n_added = 0
    for i in range(len(key_factors)):
        for j in range(i+1, len(key_factors)):
            f1, f2 = key_factors[i], key_factors[j]
            # 使用 rank 版本做交互更稳定
            r1 = f'{f1}_rank'
            r2 = f'{f2}_rank'
            if r1 in df.columns and r2 in df.columns:
                df[f'inter_{f1}x{f2}_prod'] = df[r1] * df[r2]
                df[f'inter_{f1}x{f2}_diff'] = df[r1] - df[r2]
                n_added += 2
    
    print(f"  交互特征: +{n_added} 个")
    return df


def build_labels(kl, index_df):
    """构建未来N日超额收益标签
    
    修复(2026-03-14): 加1天gap避免前视偏差。
    修复(2026-03-15): 用T+1开盘价作为买入价（更贴近实际成交），
    标签收益 = close(T+1+n) / open(T+1) - 1
    
    信号在T日收盘产生，T+1开盘买入，持有n天后以收盘价卖出。
    """
    print("🏷️  构建标签 (未来超额收益, T+1开盘买入)...")
    
    df = kl[['date', 'stock_code', 'open', 'close', 'pct_change', 'volume']].copy()
    df = df.sort_values(['stock_code', 'date'])
    
    # T+1开盘价 = open.shift(-1)
    df['next_open'] = df.groupby('stock_code')['open'].shift(-1)
    # T+1涨跌幅(用于判断涨跌停)
    df['next_pct'] = df.groupby('stock_code')['pct_change'].shift(-1)
    # T+1成交量(用于判断停牌)
    df['next_volume'] = df.groupby('stock_code')['volume'].shift(-1)
    
    # 个股未来收益: T+1开盘买入，T+1+n收盘卖出
    # return = close(T+1+n) / open(T+1) - 1
    for n in FORWARD_DAYS:
        df[f'fwd_ret_{n}d'] = df.groupby('stock_code')['close'].transform(
            lambda x, _n=n: x.shift(-(_n + 1))
        ) / df['next_open'] - 1
    
    # 标记不可交易: T+1涨停(买不进) / T+1跌停(卖不出) / T+1停牌
    # 涨停: pct_change >= 9.8% (考虑A股10%/20%涨跌幅，9.8%是保守阈值)
    # 跌停: pct_change <= -9.8%
    # 停牌: volume == 0
    df['next_limit_up'] = df['next_pct'].fillna(0) >= 9.8  # T+1涨停，买不进
    df['next_suspended'] = df['next_volume'].fillna(0) == 0  # T+1停牌
    df['untradable'] = df['next_limit_up'] | df['next_suspended']
    
    # 不可交易的标签设为 NaN（回测时自动排除）
    for n in FORWARD_DAYS:
        df.loc[df['untradable'], f'fwd_ret_{n}d'] = np.nan
    
    n_untradable = df['untradable'].sum()
    n_limit_up = df['next_limit_up'].sum()
    n_suspended = df['next_suspended'].sum()
    print(f"  不可交易过滤: 涨停{n_limit_up}条 + 停牌{n_suspended}条 = {n_untradable}条 "
          f"({n_untradable/len(df)*100:.2f}%)")
    
    # 合并指数收益 (同样用T+1开盘买入)
    idx = index_df.copy()
    idx = idx.sort_values('date')
    idx['idx_next_open'] = idx['index_close'].shift(-1)  # 指数无开盘价，用前收近似
    # 注: 如果有指数开盘价数据更好，这里用T+1收盘近似（指数不存在涨跌停）
    for n in FORWARD_DAYS:
        idx[f'idx_fwd_{n}d'] = idx['index_close'].shift(-(n + 1)) / idx['index_close'].shift(-1) - 1
    
    df = df.merge(idx[['date'] + [f'idx_fwd_{n}d' for n in FORWARD_DAYS]], on='date', how='left')
    
    # 超额收益 = 个股 - 指数
    for n in FORWARD_DAYS:
        df[f'label_{n}d'] = df[f'fwd_ret_{n}d'] - df[f'idx_fwd_{n}d']
    
    label_cols = [f'label_{n}d' for n in FORWARD_DAYS]
    result = df[['date', 'stock_code', 'untradable'] + label_cols]
    
    for n in FORWARD_DAYS:
        col = f'label_{n}d'
        valid = result[col].dropna()
        print(f"  label_{n}d: mean={valid.mean():.4f}, std={valid.std():.4f}, count={len(valid)}")
    
    return result


def winsorize(series, limits=(0.01, 0.99)):
    """截面缩尾"""
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower, upper)


def preprocess_features(df, feature_cols):
    """特征预处理: 截面缩尾 + 截面标准化 (向量化版本)"""
    print("🧹 特征预处理 (缩尾 + 标准化)...")
    
    # 按日期分组，批量处理所有特征列
    grouped = df.groupby('date')
    
    # 截面缩尾: 1%/99%
    lower = grouped[feature_cols].transform('quantile', 0.01)
    upper = grouped[feature_cols].transform('quantile', 0.99)
    df[feature_cols] = df[feature_cols].clip(lower, upper, axis=1)
    
    # 截面 z-score
    means = grouped[feature_cols].transform('mean')
    stds = grouped[feature_cols].transform('std')
    stds = stds.replace(0, np.nan)
    df[feature_cols] = (df[feature_cols] - means) / stds
    
    # 填充 NaN → 0 (截面均值)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # 极端值 clip
    df[feature_cols] = df[feature_cols].clip(-5, 5)
    
    print(f"  预处理完成: {len(feature_cols)} 列")
    return df


def build_feature_matrix(output_path=None, refresh=False):
    """主流程: 构建完整特征矩阵"""
    
    if output_path and os.path.exists(output_path) and not refresh:
        print(f"✅ 特征矩阵已存在: {output_path}")
        print("   使用 --refresh 强制重建")
        return pd.read_pickle(output_path)
    
    print("=" * 60)
    print("🚀 特征工程 Pipeline 启动")
    print("=" * 60)
    t0 = datetime.now()
    
    # 1. 加载数据
    factor_map = load_active_factors()
    factor_df = load_factor_data(factor_map)
    kl = load_kline_data()
    index_df = load_index_data()
    
    # 2. K线衍生特征
    kline_features = build_kline_features(kl)
    
    # 3. 合并因子 + K线特征
    print("🔗 合并特征...")
    df = factor_df.merge(kline_features, on=['date', 'stock_code'], how='inner')
    print(f"  合并后: {df.shape}")
    
    # 4. 因子衍生特征
    factor_cols = [c for c in df.columns if c.startswith('f_')]
    df = add_factor_derived_features(df, factor_cols)
    
    # 5. 交互特征
    df = add_interaction_features(df, factor_cols)
    
    # 6. 标签
    labels = build_labels(kl, index_df)
    df = df.merge(labels, on=['date', 'stock_code'], how='left')
    
    # 7. 特征列 (排除标签、元数据、不可交易标记)
    label_cols = [f'label_{n}d' for n in FORWARD_DAYS]
    exclude_cols = ['date', 'stock_code', 'untradable'] + label_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 8. 预处理
    df = preprocess_features(df, feature_cols)
    
    # 9. 分离：有标签(train) + 无标签(predict)，都保留
    # 不再删除无标签行，predict模式需要最新数据
    n_with_label = df['label_5d'].notna().sum()
    n_no_label = df['label_5d'].isna().sum()
    print(f"  有标签行: {n_with_label:,}, 无标签行(最近): {n_no_label:,}")
    df = df.reset_index(drop=True)
    
    # 10. 统计
    print("\n" + "=" * 60)
    print("📊 特征矩阵摘要")
    print("=" * 60)
    print(f"  总行数: {len(df):,}")
    print(f"  股票数: {df['stock_code'].nunique()}")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  交易日数: {df['date'].nunique()}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  特征列表:")
    for i, c in enumerate(feature_cols):
        print(f"    [{i+1:3d}] {c}")
    
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  ⏱️  耗时: {elapsed:.1f}s")
    
    # 11. 保存
    if output_path:
        df.to_pickle(output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  💾 保存: {output_path} ({size_mb:.1f}MB)")
    
    # 保存特征名列表
    meta = {
        'feature_cols': feature_cols,
        'label_cols': label_cols,
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'date_range': [df['date'].min(), df['date'].max()],
        'built_at': datetime.now().isoformat(),
    }
    meta_path = output_path.replace('.pkl', '_meta.json') if output_path else None
    if meta_path:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  📝 元数据: {meta_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='特征工程Pipeline')
    parser.add_argument('--output', default=os.path.join(DATA_DIR, 'ml_features.pkl'),
                       help='输出路径 (pickle)')
    parser.add_argument('--refresh', action='store_true', help='强制重建')
    args = parser.parse_args()
    
    build_feature_matrix(args.output, args.refresh)


if __name__ == '__main__':
    main()
