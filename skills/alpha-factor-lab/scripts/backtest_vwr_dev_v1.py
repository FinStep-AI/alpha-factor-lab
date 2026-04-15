#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 成交量加权收益偏差因子 (Volume-Weighted Return Deviation) v1

逻辑:
  VWR_dev = 成交量加权日收益均值 - 等权日收益均值 (过去20天)
  
  正值: 高成交量日倾向上涨 → 知情买入主导 → 信息持续释放 → 后续正alpha
  负值: 高成交量日倾向下跌 → 恐慌抛售主导 → 卖压持续 → 后续负alpha
  
  本质: 衡量成交量与收益方向的协同性(volume-return covariation)
  与传统量价相关不同——这里直接用成交量作为权重加权收益率，
  避免相关系数对离群值的敏感性。
  
  成交额OLS中性化 + MAD winsorize + z-score

参考: Llorente et al. (2002), Campbell, Grossman & Wang (1993)
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
RETURNS_PATH = BASE_DIR / "data" / "returns.csv"

WINDOW = 20
WINSOR_PCT = 0.05
N_GROUPS = 5
COST = 0.003

CONFIGS = [
    {"name": "vwr_dev_v1_5d", "rebalance": 5, "forward": 5},
    {"name": "vwr_dev_v1_10d", "rebalance": 10, "forward": 10},
    {"name": "vwr_dev_v1_20d", "rebalance": 20, "forward": 20},
]


def winsorize_mad(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad < 1e-12:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)


def load_and_compute():
    print("[1/4] Loading data & computing factor...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
    df["daily_ret"] = df["close"] / df["prev_close"] - 1
    
    # Log amount for neutralization
    df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    # Compute VWR deviation per stock rolling
    results = []
    for stock, grp in df.groupby("stock_code"):
        grp = grp.sort_values("date").copy()
        ret = grp["daily_ret"].values
        vol = grp["volume"].values.astype(float)
        dates = grp["date"].values
        log_amt = grp["log_amount_20d"].values
        
        n = len(grp)
        for i in range(WINDOW - 1, n):
            r_w = ret[i - WINDOW + 1: i + 1]
            v_w = vol[i - WINDOW + 1: i + 1]
            
            valid = np.isfinite(r_w) & np.isfinite(v_w) & (v_w > 0)
            if valid.sum() < WINDOW * 0.7:
                continue
            
            r_v = r_w[valid]
            v_v = v_w[valid]
            
            # Volume-weighted mean return
            v_sum = v_v.sum()
            if v_sum <= 0:
                continue
            vwr = (r_v * v_v).sum() / v_sum
            
            # Equal-weighted mean return
            ewr = r_v.mean()
            
            # Deviation
            factor_val = vwr - ewr
            
            results.append({
                "date": dates[i],
                "stock_code": stock,
                "factor_raw": factor_val,
                "log_amount_20d": log_amt[i]
            })
    
    factor_df = pd.DataFrame(results)
    print(f"  Computed {len(factor_df)} factor values")
    return df, factor_df


def neutralize(factor_df):
    print("[2/4] Neutralizing...")
    results = []
    for date, grp in factor_df.groupby("date"):
        vals = grp[["stock_code", "factor_raw", "log_amount_20d"]].dropna()
        if len(vals) < 50:
            continue
        
        lo = vals["factor_raw"].quantile(WINSOR_PCT)
        hi = vals["factor_raw"].quantile(1 - WINSOR_PCT)
        vals["factor_w"] = vals["factor_raw"].clip(lower=lo, upper=hi)
        
        y = vals["factor_w"].values
        x = vals["log_amount_20d"].values
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            continue
        
        yv, xv = y[valid], x[valid]
        x_mat = np.column_stack([np.ones(len(xv)), xv])
        try:
            beta_ols = np.linalg.lstsq(x_mat, yv, rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            residuals[valid] = yv - x_mat @ beta_ols
        except np.linalg.LinAlgError:
            continue
        
        vals["factor_neutral"] = residuals
        vals["factor_neutral"] = winsorize_mad(vals["factor_neutral"])
        
        mu = vals["factor_neutral"].mean()
        sigma = vals["factor_neutral"].std()
        if sigma < 1e-10:
            continue
        vals["factor_z"] = (vals["factor_neutral"] - mu) / sigma
        
        for _, row in vals.iterrows():
            results.append({
                "date": date,
                "stock_code": row["stock_code"],
                "factor_value": row["factor_z"]
            })
    
    out = pd.DataFrame(results)
    print(f"  Output: {len(out)} rows, {out['date'].nunique()} dates")
    return out


def run_backtest(factor_df, config):
    name = config["name"]
    rebal = config["rebalance"]
    fwd = config["forward"]
    
    factor_path = BASE_DIR / "data" / f"factor_{name}.csv"
    output_dir = BASE_DIR / "output" / name
    report_path = output_dir / "backtest_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    factor_df.to_csv(factor_path, index=False)
    
    engine_path = Path(__file__).parent / "factor_backtest.py"
    import subprocess
    cmd = [
        sys.executable, str(engine_path),
        "--factor", str(factor_path),
        "--returns", str(RETURNS_PATH),
        "--n-groups", str(N_GROUPS),
        "--rebalance-freq", str(rebal),
        "--forward-days", str(fwd),
        "--cost", str(COST),
        "--output-report", str(report_path),
        "--output-dir", str(output_dir),
        "--factor-name", name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print last part of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(f"  {line}")
    
    if not report_path.exists():
        return None
    
    with open(report_path) as f:
        report = json.load(f)
    
    m = report.get("metrics", {})
    return {
        "name": name,
        "ic_mean": m.get("ic_mean", 0) or 0,
        "ic_t": m.get("ic_t_stat", 0) or 0,
        "sharpe": m.get("long_short_sharpe", 0) or 0,
        "mono": m.get("monotonicity", 0) or 0,
        "mdd": m.get("long_short_mdd", 0) or 0,
        "turnover": m.get("turnover_mean", 0) or 0,
        "group_rets": m.get("group_returns_annualized", []),
        "report": report,
    }


def main():
    df, factor_raw = load_and_compute()
    factor_df = neutralize(factor_raw)
    
    # Ensure returns CSV
    if not RETURNS_PATH.exists():
        df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
        df["return"] = df["close"] / df["prev_close"] - 1
        df[["date", "stock_code", "return"]].dropna().to_csv(RETURNS_PATH, index=False)
    
    print("\n[3/4] Running backtests...")
    best = None
    all_results = []
    
    for config in CONFIGS:
        print(f"\n--- {config['name']} ---")
        
        # Original direction
        result = run_backtest(factor_df, config)
        if result:
            passed = abs(result["ic_mean"]) > 0.015 and abs(result["ic_t"]) > 2 and abs(result["sharpe"]) > 0.5
            result["passed"] = passed
            all_results.append(result)
            print(f"  → IC={result['ic_mean']:.4f}, t={result['ic_t']:.2f}, "
                  f"Sharpe={result['sharpe']:.4f}, Mono={result['mono']:.2f} {'✅' if passed else '❌'}")
            if passed and (best is None or abs(result["sharpe"]) > abs(best["sharpe"])):
                best = result
        
        # Reverse direction
        rev_df = factor_df.copy()
        rev_df["factor_value"] = -rev_df["factor_value"]
        rev_config = {**config, "name": config["name"] + "_rev"}
        rev_result = run_backtest(rev_df, rev_config)
        if rev_result:
            rev_passed = abs(rev_result["ic_mean"]) > 0.015 and abs(rev_result["ic_t"]) > 2 and abs(rev_result["sharpe"]) > 0.5
            rev_result["passed"] = rev_passed
            all_results.append(rev_result)
            print(f"  → [REV] IC={rev_result['ic_mean']:.4f}, t={rev_result['ic_t']:.2f}, "
                  f"Sharpe={rev_result['sharpe']:.4f}, Mono={rev_result['mono']:.2f} {'✅' if rev_passed else '❌'}")
            if rev_passed and (best is None or abs(rev_result["sharpe"]) > abs(best["sharpe"])):
                best = rev_result
    
    print(f"\n[4/4] Summary")
    print("="*60)
    for r in all_results:
        tag = "✅" if r.get("passed") else "❌"
        print(f"  {tag} {r['name']:30s} IC={r['ic_mean']:+.4f} t={r['ic_t']:+.2f} "
              f"Sharpe={r['sharpe']:+.4f} Mono={r['mono']:+.2f}")
    print("="*60)
    
    if best and best.get("passed"):
        print(f"  BEST: {best['name']} → Sharpe={best['sharpe']:.4f}")
    else:
        print("  No config passed all thresholds")
    
    return best


if __name__ == "__main__":
    result = main()
    sys.exit(0 if (result and result.get("passed")) else 1)
