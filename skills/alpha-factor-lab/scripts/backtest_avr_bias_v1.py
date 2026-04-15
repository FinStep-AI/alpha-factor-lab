#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 异常成交量收益偏向因子 (Abnormal Volume Return Bias) v1

逻辑:
  过去20日中，将每天按成交量分为两类:
    - 高成交量日: 当日成交量 > 20日MA * 1.5
    - 正常日: 其余
  
  因子 = mean(高成交量日的收益率) - mean(正常日的收益率)
  
  正向: 高成交量日上涨 → 知情交易者买入信号 → 后续信息持续释放 → 正alpha
  负向: 高成交量日下跌 → 恐慌性抛售 → 后续持续承压
  
  成交额OLS中性化 + MAD winsorize + z-score

参考: Llorente, Michaely, Saar & Wang (2002) JFQA
      "Dynamic Volume-Return Relation of Individual Stocks"
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
RETURNS_PATH = BASE_DIR / "data" / "returns.csv"

WINDOW = 20
VOLUME_THRESHOLD = 1.5  # abnormal = > 1.5x MA
WINSOR_PCT = 0.05
N_GROUPS = 5
COST = 0.003

# Try multiple configurations
CONFIGS = [
    {"name": "avr_bias_v1_5d", "rebalance": 5, "forward": 5},
    {"name": "avr_bias_v1_20d", "rebalance": 20, "forward": 20},
]


def winsorize_mad(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad < 1e-12:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)


def load_data():
    print("[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
    df["daily_ret"] = df["close"] / df["prev_close"] - 1
    
    # 20d rolling mean volume
    df["vol_ma20"] = df.groupby("stock_code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # Is abnormal volume day?
    df["is_abnormal_vol"] = (df["volume"] > df["vol_ma20"] * VOLUME_THRESHOLD).astype(float)
    
    # Log amount for neutralization
    df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    print(f"  Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks")
    return df


def compute_factor(df):
    """Compute abnormal volume return bias."""
    print("[2/5] Computing factor...")
    
    results = []
    for stock, grp in df.groupby("stock_code"):
        grp = grp.sort_values("date").copy()
        
        ret = grp["daily_ret"].values
        is_abn = grp["is_abnormal_vol"].values
        dates = grp["date"].values
        log_amt = grp["log_amount_20d"].values
        
        n = len(grp)
        for i in range(WINDOW - 1, n):
            window_ret = ret[i - WINDOW + 1: i + 1]
            window_abn = is_abn[i - WINDOW + 1: i + 1]
            
            valid = np.isfinite(window_ret) & np.isfinite(window_abn)
            if valid.sum() < WINDOW * 0.7:
                continue
            
            wr = window_ret[valid]
            wa = window_abn[valid]
            
            abn_mask = wa > 0.5
            norm_mask = ~abn_mask
            
            n_abn = abn_mask.sum()
            n_norm = norm_mask.sum()
            
            if n_abn < 2 or n_norm < 5:
                continue
            
            abn_ret_mean = wr[abn_mask].mean()
            norm_ret_mean = wr[norm_mask].mean()
            
            factor_val = abn_ret_mean - norm_ret_mean
            
            results.append({
                "date": dates[i],
                "stock_code": stock,
                "factor_raw": factor_val,
                "log_amount_20d": log_amt[i]
            })
    
    factor_df = pd.DataFrame(results)
    print(f"  Computed {len(factor_df)} factor values")
    return factor_df


def neutralize(factor_df):
    """OLS neutralize + winsorize + z-score."""
    print("[3/5] Neutralizing...")
    
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


def run_backtest_config(factor_df, config):
    """Run backtest for a specific config."""
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
    print(result.stdout[-800:] if len(result.stdout) > 800 else result.stdout)
    
    if not report_path.exists():
        return None
    
    with open(report_path) as f:
        report = json.load(f)
    
    metrics = report.get("metrics", {})
    return {
        "name": name,
        "ic_mean": metrics.get("ic_mean", 0) or 0,
        "ic_t": metrics.get("ic_t_stat", 0) or 0,
        "sharpe": metrics.get("long_short_sharpe", 0) or 0,
        "mono": metrics.get("monotonicity", 0) or 0,
        "mdd": metrics.get("long_short_mdd", 0) or 0,
        "turnover": metrics.get("turnover_mean", 0) or 0,
        "group_rets": metrics.get("group_returns_annualized", []),
        "report": report,
    }


def try_reverse(factor_df, config):
    """Try reversed factor."""
    rev_df = factor_df.copy()
    rev_df["factor_value"] = -rev_df["factor_value"]
    rev_config = {
        "name": config["name"] + "_rev",
        "rebalance": config["rebalance"],
        "forward": config["forward"]
    }
    return run_backtest_config(rev_df, rev_config)


def main():
    df = load_data()
    
    # Prepare returns
    if not RETURNS_PATH.exists():
        df_ret = df[["date", "stock_code", "daily_ret"]].dropna()
        df_ret = df_ret.rename(columns={"daily_ret": "return"})
        df_ret.to_csv(RETURNS_PATH, index=False)
    
    factor_raw = compute_factor(df)
    factor_df = neutralize(factor_raw)
    
    print("\n[4/5] Running backtests (multiple configs)...")
    
    best = None
    for config in CONFIGS:
        print(f"\n--- Config: {config['name']} (rebal={config['rebalance']}, fwd={config['forward']}) ---")
        
        result = run_backtest_config(factor_df, config)
        if result:
            passed = abs(result["ic_mean"]) > 0.015 and abs(result["ic_t"]) > 2 and abs(result["sharpe"]) > 0.5
            print(f"  IC={result['ic_mean']:.4f}, t={result['ic_t']:.2f}, "
                  f"Sharpe={result['sharpe']:.4f}, Mono={result['mono']:.2f}")
            
            if passed:
                result["passed"] = True
                if best is None or abs(result["sharpe"]) > abs(best["sharpe"]):
                    best = result
        
        # Try reverse
        rev_result = try_reverse(factor_df, config)
        if rev_result:
            rev_passed = abs(rev_result["ic_mean"]) > 0.015 and abs(rev_result["ic_t"]) > 2 and abs(rev_result["sharpe"]) > 0.5
            print(f"  [REV] IC={rev_result['ic_mean']:.4f}, t={rev_result['ic_t']:.2f}, "
                  f"Sharpe={rev_result['sharpe']:.4f}, Mono={rev_result['mono']:.2f}")
            
            if rev_passed:
                rev_result["passed"] = True
                if best is None or abs(rev_result["sharpe"]) > abs(best["sharpe"]):
                    best = rev_result
    
    print(f"\n[5/5] Summary")
    if best and best.get("passed"):
        print(f"  ✅ BEST: {best['name']}, IC={best['ic_mean']:.4f}, t={best['ic_t']:.2f}, "
              f"Sharpe={best['sharpe']:.4f}")
    else:
        print("  ❌ No configuration passed thresholds")
        # Print best attempt
        all_results = []
        for config in CONFIGS:
            for suffix in ["", "_rev"]:
                rp = BASE_DIR / "output" / (config["name"] + suffix) / "backtest_report.json"
                if rp.exists():
                    with open(rp) as f:
                        r = json.load(f)
                    m = r.get("metrics", {})
                    all_results.append({
                        "name": config["name"] + suffix,
                        "ic": m.get("ic_mean", 0),
                        "t": m.get("ic_t_stat", 0),
                        "sharpe": m.get("long_short_sharpe", 0),
                        "mono": m.get("monotonicity", 0),
                    })
        if all_results:
            best_attempt = max(all_results, key=lambda x: abs(x.get("sharpe", 0) or 0))
            print(f"  Best attempt: {best_attempt}")
    
    return best


if __name__ == "__main__":
    result = main()
    sys.exit(0 if (result and result.get("passed")) else 1)
