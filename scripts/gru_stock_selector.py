#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU 选股模型 (Deep Learning Stock Selector)

用 GRU + Temporal Attention 捕捉时序模式，与 LightGBM 做正面对比。

核心设计:
  - 输入: 每只股票过去 SEQ_LEN 天的特征序列 → (batch, seq_len, n_features)
  - 输出: 预测未来5日超额收益 (标量)
  - Walk-Forward 滚动回测，与 LightGBM 完全对齐
  - Apple Silicon MPS 加速

Usage:
  # 回测 (vs LightGBM)
  python3 gru_stock_selector.py --mode backtest --output models/gru_v1/

  # 预测
  python3 gru_stock_selector.py --mode predict --topn 25
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FEATURES_PKL = os.path.join(DATA_DIR, 'ml_features.pkl')
FEATURES_META = os.path.join(DATA_DIR, 'ml_features_meta.json')

# GRU 超参数
SEQ_LEN = 20         # 序列长度 (回看20天)
HIDDEN_SIZE = 128     # GRU隐藏层大小
NUM_LAYERS = 2        # GRU层数
DROPOUT = 0.3         # Dropout
FC_HIDDEN = 64        # 全连接隐藏层
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 5          # Early stopping

# Walk-Forward (与 LightGBM 对齐)
DEFAULT_LABEL = 'label_5d'
TRAIN_DAYS = 504
PREDICT_DAYS = 63
STEP_DAYS = 21

TOP_N = 25
COST_RATE = 0.003


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ===== Dataset =====
class StockSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ===== Model =====
class GRUStockModel(nn.Module):
    """
    GRU + Temporal Attention + FC

    GRU(n_features, hidden) → Attention加权聚合 → LayerNorm → FC → 1
    """
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, fc_hidden=FC_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)                        # (B, T, H)
        w = torch.softmax(self.attn(gru_out), dim=1)    # (B, T, 1)
        ctx = (gru_out * w).sum(dim=1)                  # (B, H)
        return self.fc(ctx).squeeze(-1)                  # (B,)


# ===== 数据 =====
def load_features():
    print("📊 加载特征矩阵...")
    df = pd.read_pickle(FEATURES_PKL)
    with open(FEATURES_META) as f:
        meta = json.load(f)
    feature_cols = meta['feature_cols']
    print(f"  样本: {len(df):,}, 特征: {len(feature_cols)}, 日期: {df['date'].nunique()}")
    return df, feature_cols


def build_all_sequences(df, feature_cols, label_col):
    """
    一次性构造所有 (date, stock) 的序列，返回索引字典。
    避免每个Fold重复遍历。
    
    Returns:
        all_seqs: np.array (N, SEQ_LEN, n_features)
        all_labels: np.array (N,)
        all_dates: np.array (N,) — 每个样本的日期
        all_stocks: np.array (N,) — 每个样本的stock_code
        date_index: dict {date: list of indices} — 按日期快速查询
    """
    print("  🔧 预构造所有序列（一次性）...")
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_features = len(feature_cols)
    sequences, labels, dates_arr, stocks_arr = [], [], [], []
    
    for stock, sdf in df.groupby('stock_code'):
        if len(sdf) < SEQ_LEN + 1:
            continue
        feats = sdf[feature_cols].values.astype(np.float32)
        lbls = sdf[label_col].values
        dts = sdf['date'].values
        
        # 向量化NaN检查
        nan_counts = np.isnan(feats).sum(axis=1)  # (T,)
        
        for i in range(SEQ_LEN, len(sdf)):
            if np.isnan(lbls[i]):
                continue
            # 窗口内NaN占比 > 30% 跳过
            window_nans = nan_counts[i-SEQ_LEN:i].sum()
            if window_nans > SEQ_LEN * n_features * 0.3:
                continue
            sequences.append(np.nan_to_num(feats[i-SEQ_LEN:i], nan=0.0))
            labels.append(lbls[i])
            dates_arr.append(dts[i])
            stocks_arr.append(stock)
    
    all_seqs = np.array(sequences, dtype=np.float32)
    all_labels = np.array(labels, dtype=np.float32)
    all_dates = np.array(dates_arr)
    all_stocks = np.array(stocks_arr)
    
    # 建日期索引
    date_index = {}
    for idx, d in enumerate(all_dates):
        date_index.setdefault(d, []).append(idx)
    
    print(f"  ✅ 共 {len(all_seqs):,} 个序列, {len(date_index)} 个日期")
    return all_seqs, all_labels, all_dates, all_stocks, date_index


def get_sequences_by_dates(all_seqs, all_labels, all_dates, all_stocks, date_index, target_dates):
    """从预构造的缓存中按日期提取序列（O(1)查询）"""
    indices = []
    for d in target_dates:
        indices.extend(date_index.get(d, []))
    
    if not indices:
        n_feat = all_seqs.shape[2] if len(all_seqs.shape) == 3 else 0
        return np.zeros((0, SEQ_LEN, n_feat), dtype=np.float32), np.array([]), pd.DataFrame()
    
    indices = np.array(indices)
    seqs = all_seqs[indices]
    lbls = all_labels[indices]
    meta = pd.DataFrame({'date': all_dates[indices], 'stock_code': all_stocks[indices]})
    return seqs, lbls, meta


# ===== 训练 =====
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0, 0
    all_p, all_y = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        p = model(X)
        total_loss += criterion(p, y).item(); n += 1
        all_p.append(p.cpu().numpy())
        all_y.append(y.cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(all_p), np.concatenate(all_y)


def train_gru(train_seqs, train_labels, val_seqs, val_labels, n_features, device):
    train_loader = DataLoader(StockSequenceDataset(train_seqs, train_labels),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockSequenceDataset(val_seqs, val_labels),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = GRUStockModel(n_features=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_loss, best_state, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_p, v_y = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        from scipy.stats import spearmanr
        mask = ~(np.isnan(v_p) | np.isnan(v_y))
        ic = spearmanr(v_p[mask], v_y[mask])[0] if mask.sum() > 10 else 0

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    Epoch {ep+1:2d}/{EPOCHS}: "
                  f"train={t_loss:.6f} val={v_loss:.6f} IC={ic:.4f}")

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop @ epoch {ep+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def calc_ic(pred, actual):
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() < 10:
        return np.nan
    from scipy.stats import spearmanr
    return spearmanr(pred[mask], actual[mask])[0]


# ===== Walk-Forward =====
def walk_forward_split(dates):
    ud = sorted(dates.unique())
    n = len(ud)
    splits, start = [], 0
    while start + TRAIN_DAYS + PREDICT_DAYS <= n:
        te = start + TRAIN_DAYS
        splits.append((ud[start:te], ud[te:min(te + PREDICT_DAYS, n)]))
        start += STEP_DAYS
    return splits


def backtest_gru(df, feature_cols, label_col=DEFAULT_LABEL, top_n=TOP_N):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"🧠 GRU Walk-Forward 回测")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  SEQ_LEN={SEQ_LEN}, HIDDEN={HIDDEN_SIZE}, LAYERS={NUM_LAYERS}, "
          f"DROPOUT={DROPOUT}, LR={LR}, EPOCHS={EPOCHS}")

    # 一次性预构造所有序列（核心优化：避免每Fold重复遍历）
    all_seqs, all_labels, all_dates, all_stocks, date_index = \
        build_all_sequences(df, feature_cols, label_col)

    splits = walk_forward_split(df['date'])
    print(f"  滚动窗口: {len(splits)} 期\n")

    n_features = len(feature_cols)
    all_preds_list = []
    fold_metrics = []
    last_model = None

    for i, (train_dates, test_dates) in enumerate(splits):
        print(f"  Fold {i+1}/{len(splits)}: "
              f"train {train_dates[0]}~{train_dates[-1]} → "
              f"test {test_dates[0]}~{test_dates[-1]}")

        # 从缓存中按日期切片（O(1)）
        tr_s, tr_l, _ = get_sequences_by_dates(
            all_seqs, all_labels, all_dates, all_stocks, date_index, train_dates)
        if len(tr_s) == 0:
            print("    ⚠️ skip (empty)")
            continue

        val_n = max(int(len(tr_s) * 0.2), 500)
        model = train_gru(tr_s[:-val_n], tr_l[:-val_n],
                          tr_s[-val_n:], tr_l[-val_n:], n_features, device)
        last_model = model

        # 测试
        te_s, te_l, te_m = get_sequences_by_dates(
            all_seqs, all_labels, all_dates, all_stocks, date_index, test_dates)
        if len(te_s) == 0:
            continue

        model.eval()
        preds = []
        with torch.no_grad():
            for j in range(0, len(te_s), BATCH_SIZE):
                batch = torch.FloatTensor(te_s[j:j+BATCH_SIZE]).to(device)
                preds.append(model(batch).cpu().numpy())
        preds = np.concatenate(preds)

        ic = calc_ic(preds, te_l)
        print(f"    samples={len(te_s)}, IC={ic:.4f}" if not np.isnan(ic) else f"    samples={len(te_s)}, IC=NaN")

        te_m_df = te_m.copy()
        te_m_df['pred'] = preds
        te_m_df[label_col] = te_l
        all_preds_list.append(te_m_df)

        fold_metrics.append({
            'fold': i + 1,
            'train_start': str(train_dates[0]), 'train_end': str(train_dates[-1]),
            'test_start': str(test_dates[0]), 'test_end': str(test_dates[-1]),
            'train_size': len(tr_s) - val_n, 'test_size': len(te_s),
            'ic': float(ic) if not np.isnan(ic) else None,
        })

    if not all_preds_list:
        print("❌ 无有效预测")
        return None, None, None

    pred_df = pd.concat(all_preds_list, ignore_index=True)
    results = evaluate_predictions(pred_df, label_col, top_n)
    results['fold_metrics'] = fold_metrics
    results['model_type'] = 'GRU'
    results['hyperparams'] = {
        'seq_len': SEQ_LEN, 'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
        'fc_hidden': FC_HIDDEN, 'lr': LR, 'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
    }
    return results, pred_df, last_model


def evaluate_predictions(pred_df, label_col, top_n):
    print(f"\n{'='*60}")
    print("📊 GRU 回测结果")
    print(f"{'='*60}")

    hold_days = int(label_col.replace('label_', '').replace('d', ''))

    # IC
    daily_ic = pred_df.groupby('date').apply(
        lambda g: calc_ic(g['pred'].values, g[label_col].values),
        include_groups=False
    )
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_t = ic_mean / (ic_std / np.sqrt(len(daily_ic))) if ic_std > 0 else 0
    ic_pos = (daily_ic > 0).mean()

    print(f"  Rank IC: mean={ic_mean:.4f}, std={ic_std:.4f}, IR={ic_ir:.4f}, t={ic_t:.2f}")
    print(f"  IC>0: {ic_pos:.1%}")

    # 分组
    pdf = pred_df.copy()
    pdf['group'] = pdf.groupby('date')['pred'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=['G1(低)', 'G2', 'G3', 'G4', 'G5(高)'])
    )

    ud = sorted(pdf['date'].unique())
    reb_dates = ud[::hold_days]
    reb = pdf[pdf['date'].isin(reb_dates)]

    g_avg = reb.groupby(['date', 'group'])[label_col].mean().unstack().mean()
    print(f"\n  分组 {hold_days}日超额 (bp):")
    for g in g_avg.index:
        print(f"    {g}: {g_avg[g]*10000:.1f} bp")

    # TOP-N
    pr = reb.groupby('date').apply(
        lambda g: g.nlargest(top_n, 'pred')[label_col].mean(),
        include_groups=False
    )
    nav = (1 + pr).cumprod()
    total = nav.iloc[-1] - 1
    n_per = len(nav)
    ppy = 252 / hold_days
    ann = (1 + total) ** (ppy / n_per) - 1 if n_per > 0 else 0
    mdd = ((nav - nav.cummax()) / nav.cummax()).min()
    sh = (pr.mean() / pr.std()) * np.sqrt(ppy) if pr.std() > 0 else 0

    turnover_cost = 0.5 * COST_RATE * 2 * ppy
    adj_ann = ann - turnover_cost

    print(f"\n  TOP-{top_n} ({hold_days}天调仓):")
    print(f"    调仓: {n_per}次")
    print(f"    每期超额: {pr.mean()*10000:.1f} bp")
    print(f"    累计超额: {total:.2%}")
    print(f"    年化超额: {ann:.2%}")
    print(f"    年化(扣费): {adj_ann:.2%}")
    print(f"    Sharpe: {sh:.2f}")
    print(f"    MaxDD: {mdd:.2%}")

    # 多空
    br = reb.groupby('date').apply(
        lambda g: g.nsmallest(top_n, 'pred')[label_col].mean(),
        include_groups=False
    )
    ls = pr - br
    ls_sh = (ls.mean() / ls.std()) * np.sqrt(ppy) if ls.std() > 0 else 0

    # G5
    g5 = reb[reb['group'] == 'G5(高)'].groupby('date')[label_col].mean()
    g5_sh = (g5.mean() / g5.std()) * np.sqrt(ppy) if g5.std() > 0 else 0

    print(f"\n  多空 Sharpe: {ls_sh:.2f}")
    print(f"  G5 Sharpe: {g5_sh:.2f}")

    return {
        'ic_mean': float(ic_mean), 'ic_std': float(ic_std),
        'ic_ir': float(ic_ir), 'ic_t': float(ic_t),
        'ic_positive_rate': float(ic_pos),
        'top_n': top_n, 'hold_days': hold_days,
        'n_rebalance': int(n_per),
        'period_mean_excess': float(pr.mean()),
        'total_excess_return': float(total),
        'annual_excess_return': float(ann),
        'annual_excess_return_adj': float(adj_ann),
        'sharpe': float(sh), 'max_drawdown': float(mdd),
        'ls_sharpe': float(ls_sh), 'g5_sharpe': float(g5_sh),
        'group_returns_per_period': {str(k): float(v) for k, v in g_avg.items()},
    }


def save_results(results, pred_df, output_dir, model=None, n_features=None):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Predictions
    pred_df.to_pickle(os.path.join(output_dir, 'predictions.pkl'))

    # NAV
    hold = results.get('hold_days', 5)
    tn = results.get('top_n', TOP_N)
    lc = 'label_5d'
    ud = sorted(pred_df['date'].unique())
    rd = ud[::hold]
    rp = pred_df[pred_df['date'].isin(rd)]
    pr = rp.groupby('date').apply(
        lambda g: g.nlargest(tn, 'pred')[lc].mean(), include_groups=False)
    nav = (1 + pr).cumprod()
    with open(os.path.join(output_dir, 'nav_curve.json'), 'w') as f:
        json.dump([{'date': str(d), 'nav': float(v)} for d, v in nav.items()], f)

    # IC series
    daily_ic = pred_df.groupby('date').apply(
        lambda g: calc_ic(g['pred'].values, g[lc].values), include_groups=False)
    with open(os.path.join(output_dir, 'ic_series.json'), 'w') as f:
        json.dump([{'date': str(d), 'ic': float(v)} for d, v in daily_ic.items()
                    if not np.isnan(v)], f)

    # Model
    if model is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_features': n_features,
            'hyperparams': results.get('hyperparams', {}),
        }, os.path.join(output_dir, 'model.pt'))

    print(f"  💾 结果已保存: {output_dir}")


def compare_with_lgbm(gru_results, output_dir):
    """对比 GRU vs LightGBM"""
    lgbm_path = os.path.join(MODELS_DIR, 'lgbm_v1', 'backtest_results.json')
    if not os.path.exists(lgbm_path):
        print("\n  ⚠️ LightGBM 结果不存在，跳过对比")
        return

    with open(lgbm_path) as f:
        lgbm = json.load(f)

    print(f"\n{'='*60}")
    print("⚔️  GRU vs LightGBM 对比")
    print(f"{'='*60}")

    metrics = [
        ('IC均值', 'ic_mean', '.4f'),
        ('IC_t', 'ic_t', '.2f'),
        ('IC>0比例', 'ic_positive_rate', '.1%'),
        ('年化超额', 'annual_excess_return', '.2%'),
        ('年化(扣费)', 'annual_excess_return_adj', '.2%'),
        ('Sharpe', 'sharpe', '.2f'),
        ('最大回撤', 'max_drawdown', '.2%'),
        ('多空Sharpe', 'ls_sharpe', '.2f'),
        ('G5 Sharpe', 'g5_sharpe', '.2f'),
    ]

    print(f"  {'指标':<12} {'LightGBM':>12} {'GRU':>12} {'胜出':>8}")
    print(f"  {'-'*48}")

    comparison = {}
    gru_wins = 0
    lgbm_wins = 0

    for name, key, fmt in metrics:
        lv = lgbm.get(key, 0)
        gv = gru_results.get(key, 0)

        # 回撤越小越好
        if key == 'max_drawdown':
            winner = 'GRU' if gv > lv else 'LGBM'  # 回撤是负数，越大(接近0)越好
        else:
            winner = 'GRU' if gv > lv else 'LGBM'

        if winner == 'GRU':
            gru_wins += 1
        else:
            lgbm_wins += 1

        flag = '🟢' if winner == 'GRU' else '🔵'
        lf = f"{lv:{fmt}}"
        gf = f"{gv:{fmt}}"
        print(f"  {name:<12} {lf:>12} {gf:>12} {flag} {winner}")
        comparison[key] = {'lgbm': lv, 'gru': gv, 'winner': winner}

    print(f"\n  总分: GRU {gru_wins} : {lgbm_wins} LightGBM")

    # 保存对比
    with open(os.path.join(output_dir, 'vs_lgbm.json'), 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='GRU 选股模型')
    parser.add_argument('--mode', choices=['backtest', 'predict'],
                        default='backtest')
    parser.add_argument('--output', default=os.path.join(MODELS_DIR, 'gru_v1'))
    parser.add_argument('--topn', type=int, default=TOP_N)
    parser.add_argument('--label', default=DEFAULT_LABEL)
    args = parser.parse_args()

    df, feature_cols = load_features()

    if args.mode == 'backtest':
        results, pred_df, model = backtest_gru(
            df, feature_cols, args.label, args.topn)
        if results:
            save_results(results, pred_df, args.output, model, len(feature_cols))
            compare_with_lgbm(results, args.output)

    elif args.mode == 'predict':
        # 加载最新模型预测
        model_path = os.path.join(args.output, 'model.pt')
        if not os.path.exists(model_path):
            print("❌ 模型不存在，先跑 --mode backtest")
            return
        ckpt = torch.load(model_path, map_location='cpu')
        device = get_device()
        model = GRUStockModel(n_features=ckpt['n_features']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        latest_date = df['date'].max()
        print(f"\n📋 预测日期: {latest_date}")

        # 构造序列
        all_seqs, all_labels, all_dates, all_stocks, date_index = \
            build_all_sequences(df, feature_cols, args.label)
        seqs, lbls, meta = get_sequences_by_dates(
            all_seqs, all_labels, all_dates, all_stocks, date_index, [latest_date])
        if len(seqs) == 0:
            print("❌ 无数据")
            return

        with torch.no_grad():
            preds = model(torch.FloatTensor(seqs).to(device)).cpu().numpy()

        meta['pred_return'] = preds
        top = meta.nlargest(args.topn, 'pred_return')
        print(f"\nTOP-{args.topn}:")
        for i, (_, r) in enumerate(top.iterrows()):
            print(f"  {i+1:3d}  {int(r['stock_code']):>6d}  {r['pred_return']:.4f}")


if __name__ == '__main__':
    main()
