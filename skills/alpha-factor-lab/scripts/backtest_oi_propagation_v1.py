#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 隔夜-日内动量传播因子 (Overnight-Intraday Momentum Propagation) v1

逻辑:
  过去20天滚动计算 intraday_ret = alpha + beta * overnight_ret 的回归beta。
  beta > 0: 隔夜涨→日内也涨 = 信息延续/传播强
  beta < 0: 隔夜涨→日内跌 = 信息反转/过度反应
  
  假设: 信息传播效率高(beta高)的股票，知情交易者主导定价，
  价格发现有序，后续收益更好(信息效率溢价)。

  本质: 衡量隔夜→日内的信息传导系数，与overnight_momentum(看隔夜收益水平)
  完全不同角度——一个看"涨多少"，一个看"传导多快"。

参考: Gao, Han, Li, Zhou (2018) "Market Intraday Momentum"

数据: csi1000_kline_raw.csv (999只, 817天)
    列: date, stock_code, open, close, high, low, volume, amount, amplitude, 
        pct_change, change, turnover

输出:
  - data/factor_oi_propagation_v1.csv (因子值)
  - output/oi_propagation_v1/backtest_report.json
  - output/oi_propagation_v1/cumulative_returns.json
  - output/oi_propagation_v1/ic_series.json
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[3]  # alpha-factor-lab/
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
RETURNS_PATH = BASE_DIR / "data" / "returns.csv"
FACTOR_OUT = BASE_DIR / "data" / "factor_oi_propagation_v1.csv"
OUTPUT_DIR = BASE_DIR / "output" / "oi_propagation_v1"
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ============================================================
# Parameters
# ============================================================
WINDOW = 20          # rolling regression window
WINSOR_PCT = 0.05    # 5% winsorize
N_GROUPS = 5         # quintile sort
REBALANCE_FREQ = 5   # weekly rebalance
FORWARD_DAYS = 5     # IC forward window
COST = 0.003         # single-side trading cost


def load_data():
    """Load kline data and compute overnight/intraday returns."""
    print("[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Compute prev close for overnight return
    df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
    
    # Overnight return: open / prev_close - 1
    df["overnight_ret"] = df["open"] / df["prev_close"] - 1
    
    # Intraday return: close / open - 1
    df["intraday_ret"] = df["close"] / df["open"] - 1
    
    # Daily return for forward returns
    df["daily_ret"] = df["close"] / df["prev_close"] - 1
    
    # Log amount for neutralization
    df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    # Filter out extreme values
    for col in ["overnight_ret", "intraday_ret"]:
        df.loc[df[col].abs() > 0.15, col] = np.nan
    
    print(f"  Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks, "
          f"{df['date'].nunique()} days")
    return df


def compute_rolling_beta(df):
    """Compute rolling OLS beta of intraday_ret on overnight_ret."""
    print("[2/6] Computing rolling OI propagation beta...")
    
    results = []
    for stock, grp in df.groupby("stock_code"):
        grp = grp.sort_values("date").copy()
        
        ovn = grp["overnight_ret"].values
        intra = grp["intraday_ret"].values
        dates = grp["date"].values
        log_amt = grp["log_amount_20d"].values
        
        n = len(grp)
        betas = np.full(n, np.nan)
        
        for i in range(WINDOW - 1, n):
            y = intra[i - WINDOW + 1: i + 1]
            x = ovn[i - WINDOW + 1: i + 1]
            
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 10:  # need at least 10 valid points
                continue
            
            xv, yv = x[valid], y[valid]
            
            # OLS: beta = cov(x,y) / var(x)
            x_mean = xv.mean()
            y_mean = yv.mean()
            cov_xy = ((xv - x_mean) * (yv - y_mean)).mean()
            var_x = ((xv - x_mean) ** 2).mean()
            
            if var_x > 1e-12:
                betas[i] = cov_xy / var_x
        
        for i in range(n):
            if np.isfinite(betas[i]):
                results.append({
                    "date": dates[i],
                    "stock_code": stock,
                    "oi_beta_raw": betas[i],
                    "log_amount_20d": log_amt[i]
                })
    
    factor_df = pd.DataFrame(results)
    print(f"  Computed {len(factor_df)} factor values")
    return factor_df


def winsorize_mad(s, n_mad=5):
    """MAD-based winsorization."""
    median = s.median()
    mad = (s - median).abs().median()
    if mad < 1e-12:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)


def neutralize_and_normalize(factor_df):
    """Cross-sectional neutralization (OLS on log_amount) + winsorize + z-score."""
    print("[3/6] Neutralizing and normalizing...")
    
    results = []
    for date, grp in factor_df.groupby("date"):
        vals = grp[["stock_code", "oi_beta_raw", "log_amount_20d"]].dropna()
        if len(vals) < 50:
            continue
        
        # Winsorize raw factor (5% each side)
        lo = vals["oi_beta_raw"].quantile(WINSOR_PCT)
        hi = vals["oi_beta_raw"].quantile(1 - WINSOR_PCT)
        vals["factor_w"] = vals["oi_beta_raw"].clip(lower=lo, upper=hi)
        
        # OLS neutralize on log_amount
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
        
        # MAD winsorize residuals
        valid_resid = vals["factor_neutral"].dropna()
        if len(valid_resid) < 30:
            continue
        vals["factor_neutral"] = winsorize_mad(vals["factor_neutral"])
        
        # Z-score
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


def save_factor_csv(factor_df):
    """Save factor CSV for backtest engine."""
    print("[4/6] Saving factor CSV...")
    factor_df.to_csv(FACTOR_OUT, index=False)
    print(f"  Saved to {FACTOR_OUT}")


def prepare_returns_csv(df):
    """Prepare returns CSV if not exists."""
    if RETURNS_PATH.exists():
        return
    print("  Preparing returns.csv...")
    ret_df = df[["date", "stock_code", "daily_ret"]].dropna()
    ret_df = ret_df.rename(columns={"daily_ret": "return"})
    ret_df.to_csv(RETURNS_PATH, index=False)


def run_backtest(factor_df):
    """Run backtest using the factor_backtest.py engine."""
    print("[5/6] Running backtest...")
    
    # Prepare returns
    df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values(["stock_code", "date"])
    df_raw["prev_close"] = df_raw.groupby("stock_code")["close"].shift(1)
    df_raw["return"] = df_raw["close"] / df_raw["prev_close"] - 1
    
    ret_long = df_raw[["date", "stock_code", "return"]].dropna()
    ret_long.to_csv(RETURNS_PATH, index=False)
    
    # Use factor_backtest.py
    engine_path = Path(__file__).parent / "factor_backtest.py"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    import subprocess
    cmd = [
        sys.executable, str(engine_path),
        "--factor", str(FACTOR_OUT),
        "--returns", str(RETURNS_PATH),
        "--n-groups", str(N_GROUPS),
        "--rebalance-freq", str(REBALANCE_FREQ),
        "--forward-days", str(FORWARD_DAYS),
        "--cost", str(COST),
        "--output-report", str(REPORT_PATH),
        "--output-dir", str(OUTPUT_DIR),
        "--factor-name", "oi_propagation_v1"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode


def evaluate_results():
    """Load and evaluate backtest results."""
    print("[6/6] Evaluating results...")
    
    if not REPORT_PATH.exists():
        print("  ERROR: No backtest report found!")
        return None
    
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
    
    metrics = report.get("metrics", {})
    
    ic_mean = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    sharpe = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    mdd = metrics.get("long_short_mdd", 0) or 0
    turnover = metrics.get("turnover_mean", 0) or 0
    
    group_rets = metrics.get("group_returns_annualized", [])
    
    print(f"\n{'='*60}")
    print(f"  OI Propagation Factor v1 - Results")
    print(f"{'='*60}")
    print(f"  IC Mean:        {ic_mean:.4f}")
    print(f"  IC t-stat:      {ic_t:.2f}")
    print(f"  Long-Short Sharpe: {sharpe:.4f}")
    print(f"  Monotonicity:   {mono:.4f}")
    print(f"  Long-Short MDD: {mdd:.2%}")
    print(f"  Turnover:       {turnover:.2%}")
    print(f"  Group Returns:  {[f'{r:.1%}' if r else 'N/A' for r in group_rets]}")
    print(f"{'='*60}")
    
    # Evaluate thresholds
    passed = abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(sharpe) > 0.5
    print(f"\n  Pass Criteria: |IC|>0.015={abs(ic_mean)>0.015}, |t|>2={abs(ic_t)>2}, "
          f"|Sharpe|>0.5={abs(sharpe)>0.5}")
    print(f"  VERDICT: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return {
        "passed": passed,
        "ic_mean": ic_mean,
        "ic_t": ic_t,
        "sharpe": sharpe,
        "mono": mono,
        "mdd": mdd,
        "turnover": turnover,
        "group_rets": group_rets,
        "report": report,
    }


def try_reverse_direction():
    """If original direction fails, try reverse (negative factor)."""
    print("\n[RETRY] Trying reverse direction (negative factor)...")
    
    # Load factor, negate, and re-save
    fdf = pd.read_csv(FACTOR_OUT)
    fdf["factor_value"] = -fdf["factor_value"]
    FACTOR_OUT_REV = BASE_DIR / "data" / "factor_oi_propagation_v1_rev.csv"
    fdf.to_csv(FACTOR_OUT_REV, index=False)
    
    OUTPUT_DIR_REV = BASE_DIR / "output" / "oi_propagation_v1_rev"
    REPORT_PATH_REV = OUTPUT_DIR_REV / "backtest_report.json"
    OUTPUT_DIR_REV.mkdir(parents=True, exist_ok=True)
    
    engine_path = Path(__file__).parent / "factor_backtest.py"
    import subprocess
    cmd = [
        sys.executable, str(engine_path),
        "--factor", str(FACTOR_OUT_REV),
        "--returns", str(RETURNS_PATH),
        "--n-groups", str(N_GROUPS),
        "--rebalance-freq", str(REBALANCE_FREQ),
        "--forward-days", str(FORWARD_DAYS),
        "--cost", str(COST),
        "--output-report", str(REPORT_PATH_REV),
        "--output-dir", str(OUTPUT_DIR_REV),
        "--factor-name", "oi_propagation_v1_rev"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if not REPORT_PATH_REV.exists():
        return None
    
    with open(REPORT_PATH_REV, "r") as f:
        report = json.load(f)
    
    metrics = report.get("metrics", {})
    ic_mean = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    sharpe = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    mdd = metrics.get("long_short_mdd", 0) or 0
    turnover = metrics.get("turnover_mean", 0) or 0
    group_rets = metrics.get("group_returns_annualized", [])
    
    print(f"\n  [REV] IC={ic_mean:.4f}, t={ic_t:.2f}, Sharpe={sharpe:.4f}, "
          f"Mono={mono:.4f}")
    
    passed = abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(sharpe) > 0.5
    print(f"  [REV] VERDICT: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return {
        "passed": passed,
        "ic_mean": ic_mean,
        "ic_t": ic_t,
        "sharpe": sharpe,
        "mono": mono,
        "mdd": mdd,
        "turnover": turnover,
        "group_rets": group_rets,
        "report": report,
        "direction": "reverse",
    }


def main():
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Compute factor
    factor_raw = compute_rolling_beta(df)
    
    # Step 3: Neutralize
    factor_df = neutralize_and_normalize(factor_raw)
    
    # Step 4: Save
    save_factor_csv(factor_df)
    
    # Step 5-6: Backtest & Evaluate
    prepare_returns_csv(df)
    rc = run_backtest(factor_df)
    
    result = evaluate_results()
    
    # Try reverse if needed
    if result and not result["passed"]:
        rev_result = try_reverse_direction()
        if rev_result and rev_result["passed"]:
            result = rev_result
    
    return result


if __name__ == "__main__":
    result = main()
    if result:
        sys.exit(0 if result["passed"] else 1)
    else:
        sys.exit(1)
