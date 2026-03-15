#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 选股模型 (ML Stock Selector)

Pipeline:
  1. 加载特征矩阵 (feature_engine.py 产出)
  2. 滚动时序训练 (Walk-Forward)
  3. LightGBM 训练 + Optuna 超参搜索
  4. 生成每日预测收益率 → 选TOP N
  5. 回测: 与旧线性模型对比

Usage:
  # 训练+回测
  python3 ml_stock_selector.py --mode backtest --output models/lgbm_v1/

  # 预测当日选股
  python3 ml_stock_selector.py --mode predict --date 2026-03-07 --topn 25

  # Optuna 调参
  python3 ml_stock_selector.py --mode tune --n-trials 50
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import ndcg_score

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FEATURES_PKL = os.path.join(DATA_DIR, 'ml_features.pkl')
FEATURES_META = os.path.join(DATA_DIR, 'ml_features_meta.json')

# 默认标签
DEFAULT_LABEL = 'label_5d'

# 标签前瞻期 (与 feature_engine.py 一致)
FORWARD_DAYS = [5, 10, 20]

# Walk-Forward 参数
TRAIN_DAYS = 504      # 训练窗口 ~2年
PREDICT_DAYS = 63     # 预测窗口 ~1季度
STEP_DAYS = 21        # 步进 ~1月
MIN_TRAIN_DAYS = 252  # 最少需要1年数据才开始训练

# 默认 LightGBM 参数
DEFAULT_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 500,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

# 选股参数
TOP_N = 25

# 交易成本明细 (A股)
COMMISSION_RATE = 0.00025   # 佣金 万2.5 (单边)
STAMP_TAX_RATE = 0.0005     # 印花税 万5 (卖方单边)
SLIPPAGE_RATE = 0.001       # 滑点/冲击成本 (单边估计)
# 买入成本 = 佣金 + 滑点 = 0.125%
# 卖出成本 = 佣金 + 印花税 + 滑点 = 0.175%
# 单次换仓双边 = 买入 + 卖出 ≈ 0.3% (与旧 COST_RATE 一致，但更精确)
BUY_COST = COMMISSION_RATE + SLIPPAGE_RATE
SELL_COST = COMMISSION_RATE + STAMP_TAX_RATE + SLIPPAGE_RATE

# 最少上市天数 (过滤次新股/新股，缓解幸存者偏差)
MIN_LISTING_DAYS = 120


def load_features():
    """加载特征矩阵"""
    print("📊 加载特征矩阵...")
    df = pd.read_pickle(FEATURES_PKL)
    
    with open(FEATURES_META) as f:
        meta = json.load(f)
    
    feature_cols = meta['feature_cols']
    print(f"  样本: {len(df):,}, 特征: {len(feature_cols)}, 日期: {df['date'].nunique()}")
    return df, feature_cols


def walk_forward_split(dates, train_days=TRAIN_DAYS, predict_days=PREDICT_DAYS,
                       step_days=STEP_DAYS, gap_days=None):
    """
    Walk-Forward 滚动切分
    
    修复(2026-03-14): 新增 gap_days 参数，训练集与测试集之间留出 gap
    防止标签的前瞻期（如 label_5d 需要未来 5 天价格）与测试期重叠。
    默认 gap = max(FORWARD_DAYS) = 20 天。
    
    Returns: list of (train_dates, test_dates)
    """
    if gap_days is None:
        gap_days = max(FORWARD_DAYS) if FORWARD_DAYS else 20
    
    unique_dates = sorted(dates.unique())
    n = len(unique_dates)
    
    splits = []
    start = 0
    
    while start + train_days + gap_days + predict_days <= n:
        train_end = start + train_days
        test_start = train_end + gap_days  # gap 跳过，避免标签泄漏
        test_end = min(test_start + predict_days, n)
        
        train_dates = unique_dates[start:train_end]
        test_dates = unique_dates[test_start:test_end]
        
        splits.append((train_dates, test_dates))
        start += step_days
    
    # 如果剩余数据不足一个完整窗口，但有足够训练数据
    if len(splits) == 0 and n >= MIN_TRAIN_DAYS + gap_days + 5:
        train_end = n - gap_days - 5
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[train_end + gap_days:]
        splits.append((train_dates, test_dates))
    
    return splits


def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    """训练单个 LightGBM 模型"""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    n_est = params.pop('n_estimators', 500)
    
    model = lgb.LGBMRegressor(n_estimators=n_est, **params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    
    return model


def calc_ic(pred, actual):
    """计算 Rank IC (Spearman)"""
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() < 10:
        return np.nan
    from scipy.stats import spearmanr
    ic, _ = spearmanr(pred[mask], actual[mask])
    return ic


def backtest_ml(df, feature_cols, label_col=DEFAULT_LABEL, params=None, top_n=TOP_N):
    """
    Walk-Forward 回测
    
    Returns: dict with backtest results
    """
    print("\n" + "=" * 60)
    print("🚀 Walk-Forward 回测")
    print("=" * 60)
    
    splits = walk_forward_split(df['date'])
    print(f"  滚动窗口: {len(splits)} 期")
    print(f"  训练窗口: {TRAIN_DAYS}天, 预测: {PREDICT_DAYS}天, 步进: {STEP_DAYS}天")
    print(f"  标签: {label_col}, TOP-{top_n}")
    
    all_predictions = []
    fold_metrics = []
    
    for i, (train_dates, test_dates) in enumerate(splits):
        # 划分数据
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, label_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, label_col].values
        
        # 验证集: 训练集最后20%，但去掉末尾 label_gap 天防止标签泄漏
        # label_5d 需要未来5天价格，验证集末尾5天的标签会与测试期重叠
        label_gap = int(DEFAULT_LABEL.replace('label_', '').replace('d', ''))
        # 训练集按日期排序，找出需要排除的末尾样本
        train_dates_sorted = sorted(df.loc[train_mask, 'date'].unique())
        gap_cutoff_dates = set(train_dates_sorted[-label_gap:]) if len(train_dates_sorted) > label_gap else set()
        
        # 验证集从训练集末尾取20%，但排除最后 label_gap 天
        usable_train_mask = train_mask & ~df['date'].isin(gap_cutoff_dates)
        X_train_usable = df.loc[usable_train_mask, feature_cols].values
        y_train_usable = df.loc[usable_train_mask, label_col].values
        
        val_size = max(int(len(X_train_usable) * 0.2), 1000)
        X_val = X_train_usable[-val_size:]
        y_val = y_train_usable[-val_size:]
        X_train_sub = X_train_usable[:-val_size]
        y_train_sub = y_train_usable[:-val_size]
        
        # 训练
        model = train_lgbm(X_train_sub, y_train_sub, X_val, y_val, params)
        
        # 预测
        pred = model.predict(X_test)
        
        # IC
        ic = calc_ic(pred, y_test)
        
        # 保存预测
        test_df = df.loc[test_mask, ['date', 'stock_code', label_col]].copy()
        test_df['pred'] = pred
        all_predictions.append(test_df)
        
        fold_metrics.append({
            'fold': i + 1,
            'train_start': str(train_dates[0]),
            'train_end': str(train_dates[-1]),
            'test_start': str(test_dates[0]),
            'test_end': str(test_dates[-1]),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'ic': ic,
            'best_iter': model.best_iteration_ if hasattr(model, 'best_iteration_') else -1,
        })
        
        print(f"  Fold {i+1:2d}/{len(splits)}: "
              f"train {train_dates[0]}~{train_dates[-1]} | "
              f"test {test_dates[0]}~{test_dates[-1]} | "
              f"IC={ic:.4f}" if not np.isnan(ic) else f"  Fold {i+1}: IC=NaN")
    
    # 合并所有预测
    pred_df = pd.concat(all_predictions, ignore_index=True)
    
    # 计算回测指标
    results = evaluate_predictions(pred_df, label_col, top_n)
    results['fold_metrics'] = fold_metrics
    results['model_params'] = params or DEFAULT_PARAMS
    
    # 保存最后一个模型作为当前模型
    results['last_model'] = model
    results['last_fold_train_dates'] = list(train_dates)
    
    return results, pred_df


def evaluate_predictions(pred_df, label_col, top_n):
    """
    评估预测结果
    
    修复(2026-03-15):
    - 涨跌停/停牌标的在选股时排除（label=NaN的自动排除）
    - 真实换手率计算交易成本（逐期追踪持仓变动）
    - 次新股过滤（上市不满MIN_LISTING_DAYS天的排除）
    
    注意: label_5d 是未来5日累计超额收益，不是日度收益。
    回测逻辑: 每5个交易日调仓一次 (周度调仓)，持仓期收益 = label_5d。
    """
    print("\n" + "=" * 60)
    print("📊 回测结果评估")
    print("=" * 60)
    
    # 提取持仓周期 (5/10/20) 
    fwd_match = label_col.replace('label_', '').replace('d', '')
    hold_days = int(fwd_match)
    print(f"  持仓周期: {hold_days} 天")
    
    # 过滤: 只保留可交易样本 (label非NaN说明可交易)
    tradable_df = pred_df.dropna(subset=[label_col]).copy()
    n_filtered = len(pred_df) - len(tradable_df)
    print(f"  不可交易过滤: {n_filtered} 条 ({n_filtered/len(pred_df)*100:.2f}%)")
    
    # 次新股过滤: 每只股票在样本中出现不到 MIN_LISTING_DAYS 天的前 MIN_LISTING_DAYS 个观测不参与选股
    stock_date_rank = tradable_df.groupby('stock_code')['date'].rank(method='first')
    ipo_mask = stock_date_rank <= MIN_LISTING_DAYS
    n_ipo_filtered = ipo_mask.sum()
    tradable_df = tradable_df[~ipo_mask].copy()
    print(f"  次新股过滤: {n_ipo_filtered} 条 (上市<{MIN_LISTING_DAYS}天)")
    
    # 1. 总体 IC (每日截面IC)
    daily_ic = tradable_df.groupby('date').apply(
        lambda g: calc_ic(g['pred'].values, g[label_col].values),
        include_groups=False
    )
    
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_t = ic_mean / (ic_std / np.sqrt(len(daily_ic))) if ic_std > 0 else 0
    ic_positive_rate = (daily_ic > 0).mean()
    
    print(f"  Rank IC: mean={ic_mean:.4f}, std={ic_std:.4f}, IR={ic_ir:.4f}, t={ic_t:.2f}")
    print(f"  IC>0 比例: {ic_positive_rate:.1%}")
    
    # 2. 分组收益 (5组) — 用每个调仓日的分组
    tradable_df['group'] = tradable_df.groupby('date')['pred'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=['G1(低)', 'G2', 'G3', 'G4', 'G5(高)'])
    )
    
    # 只取调仓日 (每 hold_days 天取一次)
    unique_dates = sorted(tradable_df['date'].unique())
    rebalance_dates = unique_dates[::hold_days]  # 每 hold_days 天调仓一次
    rebal_df = tradable_df[tradable_df['date'].isin(rebalance_dates)]
    
    group_ret = rebal_df.groupby(['date', 'group'])[label_col].mean().unstack()
    group_avg = group_ret.mean()
    print(f"\n  分组 {hold_days}日超额收益 (每期均值, bp):")
    for g in group_avg.index:
        print(f"    {g}: {group_avg[g]*10000:.1f} bp")
    
    # 日化分组收益 (用于年化)
    group_daily_avg = group_avg / hold_days
    print(f"  分组日化超额 (bp):")
    for g in group_daily_avg.index:
        print(f"    {g}: {group_daily_avg[g]*10000:.1f} bp/day")
    
    # 3. TOP-N 组合收益 (周度调仓) — 含真实换手率交易成本
    prev_holdings = set()  # 上期持仓股票代码
    period_returns_list = []
    period_dates_list = []
    turnover_list = []
    cost_list = []
    
    for dt in rebalance_dates:
        day_df = rebal_df[rebal_df['date'] == dt]
        top_stocks = day_df.nlargest(top_n, 'pred')
        
        if len(top_stocks) == 0:
            continue
        
        # 当期持仓
        curr_holdings = set(top_stocks['stock_code'].values)
        
        # 真实换手率计算
        if len(prev_holdings) > 0:
            # 卖出: 上期有但本期没有的
            sold = prev_holdings - curr_holdings
            # 买入: 本期有但上期没有的
            bought = curr_holdings - prev_holdings
            # 换手率 = (卖出 + 买入) / (2 * 持仓数)
            turnover = (len(sold) + len(bought)) / (2 * max(len(curr_holdings), 1))
            # 交易成本 = 卖出成本 + 买入成本 (按持仓等权)
            sell_weight = len(sold) / max(len(prev_holdings), 1)
            buy_weight = len(bought) / max(len(curr_holdings), 1)
            period_cost = sell_weight * SELL_COST + buy_weight * BUY_COST
        else:
            # 首次建仓: 全部买入
            turnover = 1.0
            period_cost = BUY_COST  # 只有买入成本
        
        period_ret = top_stocks[label_col].mean()
        # 扣除交易成本
        period_ret_net = period_ret - period_cost
        
        period_returns_list.append(period_ret_net)
        period_dates_list.append(dt)
        turnover_list.append(turnover)
        cost_list.append(period_cost)
        
        prev_holdings = curr_holdings
    
    period_returns = pd.Series(period_returns_list, index=period_dates_list)
    # 也计算不扣费的用于对比
    period_returns_gross = pd.Series(
        [rebal_df[rebal_df['date'] == dt].nlargest(top_n, 'pred')[label_col].mean() 
         for dt in period_dates_list],
        index=period_dates_list
    )
    
    avg_turnover = np.mean(turnover_list[1:]) if len(turnover_list) > 1 else 0  # 排除首次建仓
    avg_cost = np.mean(cost_list[1:]) if len(cost_list) > 1 else 0
    
    # 累计收益 (净值)
    nav = (1 + period_returns).cumprod()
    nav_gross = (1 + period_returns_gross).cumprod()
    total_ret = nav.iloc[-1] - 1
    total_ret_gross = nav_gross.iloc[-1] - 1
    n_periods = len(nav)
    periods_per_year = 252 / hold_days
    ann_ret = (1 + total_ret) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0
    ann_ret_gross = (1 + total_ret_gross) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0
    
    # 最大回撤
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = dd.min()
    
    # Sharpe (用净收益的period return)
    period_mean = period_returns.mean()
    period_std = period_returns.std()
    sharpe = (period_mean / period_std) * np.sqrt(periods_per_year) if period_std > 0 else 0
    
    # Sharpe (毛收益)
    sharpe_gross = (period_returns_gross.mean() / period_returns_gross.std()) * np.sqrt(periods_per_year) \
        if period_returns_gross.std() > 0 else 0
    
    print(f"\n  TOP-{top_n} 组合 (每{hold_days}天调仓):")
    print(f"    调仓次数: {n_periods}")
    print(f"    平均换手率: {avg_turnover:.1%} (每期)")
    print(f"    平均交易成本: {avg_cost*10000:.1f} bp/期")
    print(f"    年化交易成本: {avg_cost * periods_per_year:.2%}")
    print(f"    每期平均超额(毛): {period_returns_gross.mean():.4f} ({period_returns_gross.mean()*10000:.1f}bp)")
    print(f"    每期平均超额(净): {period_mean:.4f} ({period_mean*10000:.1f}bp)")
    print(f"    累计超额(毛): {total_ret_gross:.2%}")
    print(f"    累计超额(净): {total_ret:.2%}")
    print(f"    年化超额(毛): {ann_ret_gross:.2%}")
    print(f"    年化超额(净): {ann_ret:.2%}")
    print(f"    Sharpe(毛): {sharpe_gross:.2f}")
    print(f"    Sharpe(净): {sharpe:.2f}")
    print(f"    最大回撤: {max_dd:.2%}")
    
    # Bottom-N (空头)
    bottom_period = pd.Series(
        [rebal_df[rebal_df['date'] == dt].nsmallest(top_n, 'pred')[label_col].mean() 
         for dt in period_dates_list],
        index=period_dates_list
    )
    
    # 多空
    ls_period = period_returns_gross - bottom_period
    ls_nav = (1 + ls_period).cumprod()
    ls_sharpe = (ls_period.mean() / ls_period.std()) * np.sqrt(periods_per_year) if ls_period.std() > 0 else 0
    
    print(f"\n  多空 (TOP{top_n} - BOTTOM{top_n}):")
    print(f"    累计: {ls_nav.iloc[-1] - 1:.2%}")
    print(f"    Sharpe: {ls_sharpe:.2f}")
    
    # G5 Sharpe
    g5_period = rebal_df[rebal_df['group'] == 'G5(高)'].groupby('date')[label_col].mean()
    g5_sharpe = (g5_period.mean() / g5_period.std()) * np.sqrt(periods_per_year) if g5_period.std() > 0 else 0
    print(f"\n  📊 G5(高)组 Sharpe: {g5_sharpe:.2f} (旧线性模型参考: 0.8~0.9)")
    
    results = {
        'ic_mean': float(ic_mean),
        'ic_std': float(ic_std),
        'ic_ir': float(ic_ir),
        'ic_t': float(ic_t),
        'ic_positive_rate': float(ic_positive_rate),
        'top_n': top_n,
        'hold_days': hold_days,
        'n_rebalance': int(n_periods),
        'avg_turnover': float(avg_turnover),
        'avg_cost_per_period_bp': float(avg_cost * 10000),
        'annual_cost': float(avg_cost * periods_per_year),
        'period_mean_excess': float(period_mean),
        'period_mean_excess_gross': float(period_returns_gross.mean()),
        'total_excess_return': float(total_ret),
        'total_excess_return_gross': float(total_ret_gross),
        'annual_excess_return': float(ann_ret),
        'annual_excess_return_gross': float(ann_ret_gross),
        'sharpe': float(sharpe),
        'sharpe_gross': float(sharpe_gross),
        'max_drawdown': float(max_dd),
        'ls_sharpe': float(ls_sharpe),
        'g5_sharpe': float(g5_sharpe),
        'n_untradable_filtered': int(n_filtered),
        'n_ipo_filtered': int(n_ipo_filtered),
        'group_returns_per_period': {str(k): float(v) for k, v in group_avg.items()},
        'group_returns_daily': {str(k): float(v) for k, v in group_daily_avg.items()},
    }
    
    return results


def predict_today(df, feature_cols, model_path=None, date=None, top_n=TOP_N):
    """预测指定日期的选股结果"""
    
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'lgbm_v1', 'model.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先运行 --mode backtest 训练模型")
        return None
    
    model = joblib.load(model_path)
    
    if date is None:
        date = df['date'].max()
    
    today_df = df[df['date'] == date].copy()
    if len(today_df) == 0:
        print(f"❌ 日期 {date} 无数据")
        return None
    
    X = today_df[feature_cols].values
    pred = model.predict(X)
    today_df['pred_return'] = pred
    
    # 选 TOP-N
    top = today_df.nlargest(top_n, 'pred_return')
    
    print(f"\n📋 {date} TOP-{top_n} 选股:")
    print(f"{'排名':>4} {'股票代码':>10} {'预测超额':>10}")
    print("-" * 30)
    for i, (_, row) in enumerate(top.iterrows()):
        print(f"{i+1:4d} {int(row['stock_code']):>10} {row['pred_return']:>10.4f}")
    
    return top[['date', 'stock_code', 'pred_return']]


def tune_hyperparams(df, feature_cols, label_col=DEFAULT_LABEL, n_trials=30):
    """
    Optuna 超参搜索（多 fold 平均 IC）
    
    修复(2026-03-14): 用最后 N 个 fold 的平均 IC 作为目标，
    避免单 fold 过拟合。默认用最后 3 个 fold（或全部 fold 如果不足 3 个）。
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("\n" + "=" * 60)
    print("🔍 Optuna 超参搜索 (多fold平均IC)")
    print(f"   trials: {n_trials}")
    print("=" * 60)
    
    splits = walk_forward_split(df['date'])
    if len(splits) < 2:
        print("❌ 数据不足以进行调参")
        return DEFAULT_PARAMS
    
    # 用最后 N 个 fold 做交叉验证（避免单 fold 过拟合）
    n_tune_folds = min(3, len(splits))
    tune_splits = splits[-n_tune_folds:]
    print(f"  使用最后 {n_tune_folds} 个 fold 做调参")
    
    # 预构建每个 fold 的数据（避免重复索引）
    fold_data = []
    label_gap = int(label_col.replace('label_', '').replace('d', ''))
    for train_dates, test_dates in tune_splits:
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)
        
        X_train_full = df.loc[train_mask, feature_cols].values
        y_train_full = df.loc[train_mask, label_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, label_col].values
        
        # 验证集: 训练集末尾20%，去掉最后 label_gap 天
        train_dates_sorted = sorted(df.loc[train_mask, 'date'].unique())
        gap_cutoff = set(train_dates_sorted[-label_gap:]) if len(train_dates_sorted) > label_gap else set()
        usable_mask = train_mask & ~df['date'].isin(gap_cutoff)
        X_train_usable = df.loc[usable_mask, feature_cols].values
        y_train_usable = df.loc[usable_mask, label_col].values
        
        val_size = max(int(len(X_train_usable) * 0.2), 1000)
        X_val = X_train_usable[-val_size:]
        y_val = y_train_usable[-val_size:]
        X_train_sub = X_train_usable[:-val_size]
        y_train_sub = y_train_usable[:-val_size]
        
        fold_data.append((X_train_sub, y_train_sub, X_val, y_val, X_test, y_test))
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }
        
        # 多 fold 平均 IC
        ics = []
        for X_tr, y_tr, X_v, y_v, X_te, y_te in fold_data:
            model = lgb.LGBMRegressor(n_estimators=500, **params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_v, y_v)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            pred = model.predict(X_te)
            ic = calc_ic(pred, y_te)
            if not np.isnan(ic):
                ics.append(ic)
        
        return np.mean(ics) if ics else 0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n  最优平均IC: {study.best_value:.4f} (across {n_tune_folds} folds)")
    print(f"  最优参数:")
    best = study.best_params
    for k, v in best.items():
        print(f"    {k}: {v}")
    
    # 转成完整参数
    full_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'n_estimators': 500,
        **best,
    }
    
    return full_params


def save_model_and_results(model, results, pred_df, output_dir, feature_cols):
    """保存模型、结果、预测"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"  💾 模型: {model_path}")
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    imp_path = os.path.join(output_dir, 'feature_importance.csv')
    importance.to_csv(imp_path, index=False)
    print(f"  📊 特征重要性: {imp_path}")
    print(f"  TOP-10 特征:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:40s} {int(row['importance']):6d}")
    
    # 回测结果
    results_clean = {k: v for k, v in results.items() 
                     if k not in ('last_model', 'last_fold_train_dates')}
    results_path = os.path.join(output_dir, 'backtest_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    print(f"  📝 回测结果: {results_path}")
    
    # 预测结果
    pred_path = os.path.join(output_dir, 'predictions.pkl')
    pred_df.to_pickle(pred_path)
    print(f"  📦 预测数据: {pred_path}")
    
    # NAV 曲线 (供前端) — 按调仓周期计算，含交易成本
    hold_days = results.get('hold_days', 5)
    top_n_val = results.get('top_n', TOP_N)
    label_col_name = 'label_5d'
    
    # 过滤不可交易
    tradable_pred = pred_df.dropna(subset=[label_col_name]).copy()
    # 次新股过滤
    stock_date_rank = tradable_pred.groupby('stock_code')['date'].rank(method='first')
    tradable_pred = tradable_pred[stock_date_rank > MIN_LISTING_DAYS]
    
    unique_dates = sorted(tradable_pred['date'].unique())
    rebalance_dates = unique_dates[::hold_days]
    rebal_pred = tradable_pred[tradable_pred['date'].isin(rebalance_dates)]
    
    # 含真实换手率的净收益 NAV
    prev_holdings = set()
    nav_list = []
    cum_nav = 1.0
    for dt in rebalance_dates:
        day_df = rebal_pred[rebal_pred['date'] == dt]
        top_stocks = day_df.nlargest(top_n_val, 'pred')
        if len(top_stocks) == 0:
            continue
        curr_holdings = set(top_stocks['stock_code'].values)
        
        if len(prev_holdings) > 0:
            sold = prev_holdings - curr_holdings
            bought = curr_holdings - prev_holdings
            sell_weight = len(sold) / max(len(prev_holdings), 1)
            buy_weight = len(bought) / max(len(curr_holdings), 1)
            period_cost = sell_weight * SELL_COST + buy_weight * BUY_COST
        else:
            period_cost = BUY_COST
        
        period_ret = top_stocks[label_col_name].mean() - period_cost
        cum_nav *= (1 + period_ret)
        nav_list.append({'date': str(dt), 'nav': float(cum_nav)})
        prev_holdings = curr_holdings
    
    nav_path = os.path.join(output_dir, 'nav_curve.json')
    with open(nav_path, 'w') as f:
        json.dump(nav_list, f)
    print(f"  📈 NAV曲线: {nav_path} ({len(nav_list)}期)")


def main():
    parser = argparse.ArgumentParser(description='LightGBM 选股模型')
    parser.add_argument('--mode', choices=['backtest', 'predict', 'tune'],
                       default='backtest', help='运行模式')
    parser.add_argument('--output', default=os.path.join(MODELS_DIR, 'lgbm_v1'),
                       help='输出目录')
    parser.add_argument('--date', help='预测日期 (predict模式)')
    parser.add_argument('--topn', type=int, default=TOP_N, help='选股数量')
    parser.add_argument('--label', default=DEFAULT_LABEL, help='标签列')
    parser.add_argument('--n-trials', type=int, default=30, help='Optuna trials数')
    parser.add_argument('--params', help='参数JSON文件')
    args = parser.parse_args()
    
    # 加载数据
    df, feature_cols = load_features()
    
    if args.mode == 'tune':
        # 超参搜索
        best_params = tune_hyperparams(df, feature_cols, args.label, args.n_trials)
        
        # 保存参数
        os.makedirs(args.output, exist_ok=True)
        params_path = os.path.join(args.output, 'best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  💾 参数已保存: {params_path}")
        
    elif args.mode == 'backtest':
        # 加载自定义参数
        params = None
        if args.params and os.path.exists(args.params):
            with open(args.params) as f:
                params = json.load(f)
            print(f"  使用自定义参数: {args.params}")
        
        # 回测
        results, pred_df = backtest_ml(df, feature_cols, args.label, params, args.topn)
        
        # 保存
        save_model_and_results(results['last_model'], results, pred_df, args.output, feature_cols)
        
    elif args.mode == 'predict':
        predict_today(df, feature_cols, date=args.date, top_n=args.topn)


if __name__ == '__main__':
    main()
