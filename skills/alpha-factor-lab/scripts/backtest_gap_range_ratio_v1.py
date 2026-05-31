#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gap_range_ratio_v1 skill wrapper

Runs the canonical root-level backtest script located at
`<repo>/scripts/backtest_gap_range_ratio_v1.py`, then exits with the
same return code so callers / cron pipelines see an accurate pass/fail.
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]               # alpha-factor-lab root
TARGET = REPO / "scripts" / "backtest_gap_range_ratio_v1.py"

if not TARGET.exists():
    print(f"[gap_range_ratio_v1] canonical script not found: {TARGET}", file=sys.stderr)
    sys.exit(1)

# run the canonical script under __name__ == "__main__" so its sys.exit propagates
sys.argv = [str(TARGET)] + sys.argv[1:]
runpy.run_path(str(TARGET), run_name="__main__")
