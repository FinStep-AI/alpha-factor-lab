#!/usr/bin/env python3
"""
因子: 净买方主导比 (Net Buyer Dominance) v1
factor_id: net_buyer_dom_v1

逻辑 (更新思路):
  对过去20天按涨跌分组：
    - 涨日平均量 = mean(amount_i | ret_i > 0)
    - 跌日平均量 = mean(amount_i | ret_i <= 0)
  净买方比 = 涨日均量 / (涨日均量 + 跌日均量)
  
  或者等效视角 **净流入占比**（更紧凑、更稳定）：
    每天以 ret 正负作为 funding 方向代理，成交额作为强度，
    net_buyer = (sum_up amount - sum_down amount) / sum_all amount
  这在截面比均值形式的「涨日量/(涨日量+跌日量)」方差更低，统计力更强。

  再启发性增益：用真 ret 收益带号做收益方向（而非只看涨跌方向），
  net_buyer = sum(ret_i * amount_i) / sum(|ret_i| * amount_i)
  类似资金流加权，但只用日线 OHLCV 可算。
  
  ★ 用最后一步最紧凑等式：把 sign(ret)×amount 当作净流入代理。
"""
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("data")

    print("Loading data...")
    df = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    print(f"  stocks={df['stock_code'].nunique()}, "
          f"dates={df['date'].min().date()}~{df['date'].max().date()}")

    # ---- 基础字段 ----
    df['ret'] = df.groupby('stock_code')['close'].pct_change()

    # 净买方代理(日频): sign(ret) * amount ≈ 近似知情资金流
    df['net_flow_proxy'] = np.sign(df['ret']) * df['amount']

    # ---- 20日滚动净买方比例 ----
    #   net_buyer = sum(sign(ret)*amount, 20d) / sum(|amount|, 20d)
    # 用日收益做方向、成交额做强度；分母用|amount|比sum(amount)更能抵抗方向偏差
    df['nb_num'] = df.groupby('stock_code')['net_flow_proxy'].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    df['nb_den'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.abs().rolling(20, min_periods=10).sum()
    )
    raw = df['nb_num'] / df['nb_den'].replace(0, np.nan)

    # ---- 成交额代理（中性化控制变量） ----
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))

    result = df[['date', 'stock_code']].copy()
    result['raw_factor'] = raw
    result['log_amount'] = df['log_amount']
    result = result.dropna(subset=['raw_factor', 'log_amount']).copy()

    # ---- MAD 5% winsorize per cross-section ----
    print("5% winsorize...")
    parts = []
    for dt, g in result.groupby('date', sort=False):
        v = g['raw_factor'].values
        lo, hi = np.nanquantile(v, 0.025), np.nanquantile(v, 0.975)
        g2 = g.copy(); g2['raw_factor'] = np.clip(v, lo, hi)
        parts.append(g2)
    result = pd.concat(parts, ignore_index=True)

    # ---- 截面 z-score ----
    result['factor_zscore'] = result.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # ---- 成交额OLS中性化 ----
    print("OLS neutralization...")
    parts2 = []
    for dt, g in result.groupby('date', sort=False):
        g = g.dropna(subset=['factor_zscore', 'log_amount'])
        if len(g) < 20:
            g['factor_neutral'] = np.nan
            parts2.append(g[['factor_neutral']]); continue
        x = g['log_amount'].values; y = g['factor_zscore'].values
        xm = np.nanmean(x); ym = np.nanmean(y)
        b = np.nansum((x-xm)*(y-ym)) / (np.nansum((x-xm)**2) + 1e-10)
        a = ym - b*xm
        g = g.copy(); g['factor_neutral'] = y - (a + b*x)
        parts2.append(g[['factor_neutral']])
    neutralized = pd.concat(parts2, ignore_index=False)
    result = result.copy()
    result['factor_neutral'] = neutralized['factor_neutral']

    result = result.dropna(subset=['factor_neutral'])

    # ---- 最终截面 z-score ----
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)

    out = result[['date','stock_code','factor_value']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')

    out_path = data_dir / "factor_net_buyer_dom_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(out):,} rows, "
          f"{out['date'].min()}~{out['date'].max()})")
    print(out['factor_value'].describe())

if __name__ == "__main__":
    main()
