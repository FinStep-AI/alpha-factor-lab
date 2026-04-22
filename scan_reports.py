#!/usr/bin/env python3
"""Scan all backtest reports for factors NOT yet in factors.json,
print IC, t-stat, Sharpe, mono, G5 return. Sort by IC."""
import json, os, glob
from pathlib import Path

base = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")

# Load existing factor IDs
with open(base / "factors.json") as f:
    existing = json.load(f)
existing_ids = set()
for x in existing:
    existing_ids.add(x.get("id", "") or x.get("factor_id", ""))

results = []
for rp in glob.glob(str(base / "output" / "*/backtest_report.json")):
    fid = Path(rp).parent.name
    if fid in existing_ids:
        continue
    with open(rp) as f:
        r = json.load(f)
    m = r.get("metrics", {})
    if not m:
        # some reports nest metrics differently
        m = r
    ic = m.get("ic_mean", 0)
    t = m.get("ic_t_stat", 0)
    sh = m.get("long_short_sharpe", 0)
    mono = m.get("monotonicity", 0)
    gr = m.get("group_returns_annualized", [])
    g5 = gr[-1] if gr else 0
    g1 = gr[0] if gr else 0
    name = r.get("factor_name", r.get("name", fid))
    results.append((abs(ic), ic, t, sh, mono, g5, g1, fid, name))

results.sort(reverse=True)
print(f"{'factor_id':30s} {'name':35s} {'IC':>8s}  {'t':>6s}  {'Sharpe':>7s}  {'Mono':>6s}  {'G5':>8s}  {'G1':>8s}")
print("-"*120)
for abs_ic, ic, t, sh, mono, g5, g1, fid, name in results[:30]:
    g5s = f"{g5:.2%}" if g5 else "N/A"
    g1s = f"{g1:.2%}" if g1 else "N/A"
    flag = "★" if abs(ic) > 0.015 and abs(t) > 2 and abs(sh) > 0.5 else " "
    print(f"{fid:30s} {name:35s} {ic:+8.4f}  {t:+6.2f}  {sh:+7.2f}  {mono:+6.2f}  {g5s:>8s}  {g1s:>8s} {flag}")

print(f"\nTotal scanned: {len(results)} factors not in registry")
print("★ = passes criteria (|IC|>0.015, |t|>2, |Sharpe|>0.5)")
