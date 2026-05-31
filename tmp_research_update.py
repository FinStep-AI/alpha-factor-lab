import json

with open("factor-research.json") as f:
    d = json.load(f)

entry = {
  "factor_id": "pv_wsync_v1",
  "name": "量价信号持续自相关因子 (Volume-Weighted Signed Return Autocorrelation)",
  "date": "2026-05-31",
  "source_type": "SSRN_PAPER + PKU",
  "source_title": "The Overnight Drift (SSRN-3546173) \u4e0e\u5317\u5927\u5149\u534e\u201c\u4e2d\u56fd\u80a1\u7968\u5e02\u573a\u7684\u4fe1\u606f\u4f20\u5bfc\u4e0e\u6d41\u52a8\u6027\u9700\u6c42\u201d",
  "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3546173",
  "source_author": "Bali, Brown, Murray (Overnight Drift); PKU Guanghua (info transmission)",
  "source_year": "2024 / 2022",
  "source_journal": "SSRN / PKU CCJ",
  "barra_style": "Liquidity + Momentum",
  "original_metric": "signal = sign(ret_{t-1}) * ln(1 + turnover_t / MA20(turnover)); rolling 60d lag-1 ACF, min_periods=40",
  "our_metric": "5d IC=0.0001 (t=0.02, failed); LS Sharpe=0.291; turnover=150.8%; mon=0.80 but G5<G4 non-monotonic",
  "diff_notes": "CSI1000 panel has heavy microstructure noise; lag-1 signal continuity blunted by retail trading autocorr competition; cross-sectional translation of overnight-drift mechanism not directly portable to A-share panel",
  "local_factor_id": "pv_wsync_v1",
  "formula": "sv_t = sign(ret_{t-1}) * ln(1 + turnover_t / MA20(turnover)); factor = 60d rolling lag-1 corr(sv_t, sv_{t-1}) neutralized by log(amount) MAD3 z-score",
  "conclusion": "failed / not entering corpus",
  "status": "failed",
  "verification_gates_passed": False,
  "gates_failed": ["ic_mean 0.0001 = 0.02", "ic_t 0.02 > 2.0", "g5_sharpe not reached", "monotonicity=0.80 but G5<G4 non-monotone"],
  "metrics": {
    "ic_mean_5d": 0.0001,
    "ic_t_5d": 0.02,
    "long_short_sharpe": 0.291,
    "monotonicity": 0.80,
    "group_ann_returns_pct": [-0.56, -0.75, 1.21, 11.15, 6.18],
    "turnover_mean": 1.508
  }
}

d.append(entry)
with open("factor-research.json", "w") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"factor-research.json now has {len(d)} entries")
