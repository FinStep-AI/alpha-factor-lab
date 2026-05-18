"""
Volatility Skewness v1 — 收益分布偏度因子
===========================================
过去20日日收益率的 skewness（三阶中心矩），成交额中性化+MAD缩尾+z-score。

假设：负偏(厚左尾)股票历史上后续表现较差(持续暴跌风险)，正偏(厚右尾)有零星支撑效应。
Boyer, Mitton & Vorkink (2010) "Idiosyncratic Risk" RFS。

20日窗口，中证1000，成交额OLS中性化，5%MAD缩尾，5组分层，5%手续费。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# ── 参数 ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data')
OUTPUT_DIR = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/output/volatility_skewness_v1')
WINDOW     = 20          # 因子计算窗口
FORWARD_D  = 5           # 未来收益天数
COST       = 0.003       # 双边手续费 0.3%
N_GROUPS   = 5           # 分层组数
WINSORIZE  = 0.05        # MAD 5%缩尾
START_DATE = '2023-01-01'
END_DATE   = '2026-04-30'

print("=" * 60)
print("Volatility Skewness v1 回测")
print("=" * 60)

# ── 1. 读取数据 ──────────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_DIR / 'csi1000_kline_raw.csv',
                  parse_dates=['date']).sort_values(['stock_code','date'])

# 构建日收益率
raw['ret'] = raw.groupby('stock_code')['close'].pct_change()

# 取对数成交额
raw['log_amount'] = np.log(raw['amount'].clip(lower=1))

print(f"[1/6] 数据加载完成: {len(raw):,} rows, {raw['stock_code'].nunique()} stocks, "
      f"{raw['date'].min().date()} ~ {raw['date'].max().date()}")

# ── 2. 计算因子(raw): 20日收益率偏度 ──────────────────────────────────────────
def rolling_skew(group):
    return group['ret'].rolling(WINDOW, min_periods=WINDOW).skew()

raw['factor_raw'] = raw.groupby('stock_code', group_keys=False).apply(rolling_skew)

# 去除窗口不足的NaN
raw = raw.dropna(subset=['factor_raw']).copy()
print(f"[2/6] 窗口滚动偏度完成，有效截面: "
      f"{raw['date'].nunique()} days, {raw.groupby('date')['factor_raw'].count().min():.0f} ~ "
      f"{raw.groupby('date')['factor_raw'].count().max():.0f} stocks")

# ── 3. 截面中性化 + MAD缩尾 + z-score ────────────────────────────────────────
def neutralize(panel, factor_name='factor_raw', neutralize_col='log_amount'):
    """
    截面 OLS 中性化：对 factor_raw ~ neutralizer 回归，取残差，
    再 MAD 缩尾 + z-score。
    """
    panel = panel.copy()

    # OLS 中性化 (只保留有效数据)
    mask = panel[factor_name].notna() & panel[neutralize_col].notna() & np.isfinite(panel[neutralize_col])
    sub = panel[mask].copy()
    X = np.column_stack([np.ones(len(sub)), sub[neutralize_col].values])
    y = sub[factor_name].values
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
        panel.loc[mask, 'factor_neutral'] = residual
    except Exception:
        panel['factor_neutral'] = panel[factor_name]

    # MAD 缩尾
    med = panel['factor_neutral'].median()
    mad = (panel['factor_neutral'] - med).abs().median()
    if mad < 1e-10:
        upper = lower = med
    else:
        lower = med - WINSORIZE * 1.4826 * mad
        upper = med + WINSORIZE * 1.4826 * mad
    panel['factor_neutral'] = panel['factor_neutral'].clip(lower, upper)

    # z-score
    mu = panel['factor_neutral'].mean()
    sig = panel['factor_neutral'].std()
    if sig < 1e-10:
        panel['factor_zscore'] = 0
    else:
        panel['factor_zscore'] = (panel['factor_neutral'] - mu) / sig

    return panel


results = []
for dt, grp in raw.groupby('date'):
    grp2 = neutralize(grp)
    for _, row in grp2[['stock_code','factor_zscore']].dropna().iterrows():
        results.append({'date': dt, 'stock_code': row['stock_code'],
                        'factor_value': row['factor_zscore']})

panel = pd.DataFrame(results).set_index(['date','stock_code']).sort_index()
print(f"[3/6] 截面中性化完成，面板: {panel.shape}")

# ── 4. 构建截面因子值表 + 计算未来收益 ─────────────────────────────────────────
pivot = panel['factor_value'].unstack('stock_code')

raw2 = raw[['date','stock_code','ret','log_amount','amount']].dropna(subset=['ret'])
all_dates = pivot.index.sort_values()

# 构建 future return
fwd_map = {}
for sc, grp in raw2.groupby('stock_code'):
    grp = grp.sort_values('date').rolling(FORWARD_D+1)['ret'].sum().shift(-FORWARD_D)
    for dt, val in grp.items():
        if dt in fwd_map:
            fwd_map[dt] = fwd_map[dt].append(pd.Series({sc: val}), ignore_index=False) if isinstance(fwd_map[dt], pd.Series) else fwd_map[dt]
        else:
            fwd_map[dt] = pd.Series({sc: val}) if not isinstance(val, float) or not np.isnan(val) else None

# 用更简洁的方式构建 future return
fwd_ret_dict = {}
for dt in all_dates:
    valid = {}
    for sc in pivot.columns:
        # 找该日期起 FORWARD_D 天的收益
        sub = raw2[raw2['stock_code']==sc].set_index('date').reindex(pd.date_range(dt, periods=FORWARD_D+1, freq='B')[1:])['ret']
        if len(sub.dropna()) == FORWARD_D:
            valid[sc] = sub.sum()
    fwd_ret_dict[dt] = pd.Series(valid)

print("[4/6] 未来收益计算完成，" + str(len(fwd_ret_dict)) + " 天")

# ── 5. 分层 + 回测 ────────────────────────────────────────────────────────────
records = []
group_series = {i: [] for i in range(1, N_GROUPS+1)}
long_short_rets = {}

for dt in sorted(fwd_ret_dict.keys()):
    if dt not in pivot.index:
        continue
    fac = pivot.loc[dt].dropna()
    fr  = fwd_ret_dict[dt].dropna()
    common = fac.index.intersection(fr.index)
    if len(common) < N_GROUPS * 10:
        continue

    fac_c = fac[common]
    fr_c  = fr[common]

    # 分组 quintile (1=最小组)
    groups = pd.qcut(fac_c.rank(method='first'), N_GROUPS, labels=list(range(1, N_GROUPS+1)))
    rets = fr_c.groupby(groups).mean().sort_index()
    for g, r in rets.items():
        if not np.isnan(r):
            group_series[int(g)].append((dt, r - COST))
            records.append({'date': dt, 'group': int(g), 'ret': r - COST})

    # 多空
    ls_ret = rets.iloc[-1] - rets.iloc[0] - 2*COST
    long_short_rets[dt] = ls_ret

rt_df = pd.DataFrame(records)
print(f"[5/6] 分层回测完成，{len(rt_df)} 条记录")

# ── 6. 评估指标 ──────────────────────────────────────────────────────────────
def evaluate(rets):
    if len(rets) < 10:
        return {}
    mu  = rets.mean()
    std = rets.std()
    t   = mu / (std / np.sqrt(len(rets))) if std > 1e-12 else 0
    ir  = mu / std if std > 1e-12 else 0
    ann = (1 + mu) ** 252 - 1
    sharpe = ir * np.sqrt(252)
    cum = (1 + rets).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return {'mean': float(mu), 'std': float(std), 'an_ret': float(ann),
            'ir': float(ir), 't': float(t), 'sharpe': float(sharpe), 'mdd': float(mdd)}

group_ret_series = {g: pd.Series([r for _, r in sr],
                                  index=pd.DatetimeIndex([d for d, _ in sr]))
                    for g, sr in group_series.items()}

g_metrics = {}
for g in range(1, N_GROUPS+1):
    m = evaluate(group_ret_series[g])
    g_metrics[g] = m

ls_series = pd.Series(long_short_rets)
ls_metrics = evaluate(ls_series)

# 截面IC均值 (因子值与未来收益的 rank correlation)
ics = []
for dt in fwd_ret_dict:
    if dt in pivot.index:
        fac = pivot.loc[dt].dropna()
        fr  = fwd_ret_dict[dt].dropna()
        com = fac.index.intersection(fr.index)
        if len(com) > 10:
            ics.append(fac[com].corr(fr[com], method='spearman'))

ic_arr = np.array([x for x in ics if not np.isnan(x)])
ic_mean  = ic_arr.mean()
ic_std   = ic_arr.std(ddof=1)
ic_pos   = (ic_arr > 0).mean()
ic_t     = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0
# Pre-compute facets for ranks_ic (valid cross-section pairs)
_facets = {}
for _dt in fwd_ret_dict:
    if _dt in pivot.index:
        _fac = pivot.loc[_dt].dropna()
        _fr  = fwd_ret_dict[_dt].dropna()
        _com = _fac.index.intersection(_fr.index)
        if len(_com) > 1:
            _facets[_dt] = (_fac[_com], _fr[_com])

ranks_ic = np.array([_fa.rank().corr(_fr.rank())
                     for _dt, (_fa, _fr) in _facets.items()])

print("\n" + "=" * 60)
print("回测结果摘要")
print("=" * 60)

for g in range(1, N_GROUPS+1):
    m = g_metrics[g]
    print(f"  G{g}: 年化={m.get('an_ret',0):.1%} | Sharpe={m.get('sharpe',0):.2f} |"
          f" t={m.get('t',0):.2f}")

print(f"\n  IC均值={ic_mean:.4f} | std={ic_std:.4f} | t值={ic_t:.2f} | IC正比例={ic_pos:.1%}")
print(f"  Long-Short Sharpe={ls_metrics.get('sharpe',0):.2f} | MDD={ls_metrics.get('mdd',0):.1%}")

# 单调性评分
g_sharpes = [g_metrics[g].get('sharpe', np.nan) for g in range(1, N_GROUPS+1)]
mono = np.all(np.diff(g_sharpes) >= 0)
mono_pct = sum(np.diff(g_sharpes) >= 0) / (N_GROUPS - 1)
print(f"  单调性: 完全={mono} | 部分={mono_pct:.0%} ({sum(np.diff(g_sharpes)>=0)}/{N_GROUPS-1})")

print("\n" + "=" * 60)
print("通过标准: |IC|>0.015, t>2, Sharpe>0.5, mono≥0.6")
print("=" * 60)

passed  = (abs(ic_mean) > 0.015) and (ic_t > 2) and (ls_metrics.get('sharpe',0) > 0.5) and (mono_pct >= 0.6)
str_pass = "✅ 通过！入库" if passed else "❌ 未达标"
print(f"  判断结果: {str_pass}")

# 保存结果
report = {
    'factor_name': 'volatility_skewness_v1',
    'metrics': {
        'ic_mean': float(ic_mean),
        'ic_std': float(ic_std),
        'ic_t': float(ic_t),
        'ic_positive_ratio': float(ic_pos),
        'long_short_sharpe': float(ls_metrics.get('sharpe', 0)),
        'long_short_mdd': float(ls_metrics.get('mdd', 0)),
        'monotonicity': float(mono_pct),
        'ic_count': int(len(ic_arr)),
    },
    'group_metrics': g_metrics,
    'check': {
        'ic_pass': bool(abs(ic_mean) > 0.015),
        't_pass': bool(ic_t > 2),
        'sharpe_pass': bool(ls_metrics.get('sharpe', 0) > 0.5),
        'mono_pass': bool(mono_pct >= 0.6),
        'overall': bool(passed)
    },
    'n_groups': N_GROUPS,
    'rebalance_freq': 5,
    'cost': COST,
    'output_dir': str(OUTPUT_DIR),
}

# 输出 IC 序列
ic_df = pd.DataFrame({'ic': ic_arr}, index=range(len(ic_arr)))
ic_df.to_csv(OUTPUT_DIR / 'ic_series.csv')

# 输出累积收益
nm = (1 + ls_series).cumprod()
nm.index.name = 'date'
nm.to_frame('nav').to_csv(str(OUTPUT_DIR / 'cumulative_returns.csv'))

with open(OUTPUT_DIR / 'report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[6/6] 结果已写入 {OUTPUT_DIR}/")
print(report['check'])
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
