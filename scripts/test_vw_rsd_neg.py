#!/usr/bin/env python3
"""
VW-RSD v1b: 反向使用 (-factor), 测试20d调仓+20d前瞻
"""

import numpy as np
import pandas as pd
import subprocess, json, sys, os

# 读取原始因子并取反
df = pd.read_csv('data/factor_vw_rsd_v1.csv')
df['factor_neutral'] = -df['factor_neutral']  # 反向
df.to_csv('data/factor_vw_rsd_v1_neg.csv', index=False)

print("[INFO] 反向因子已保存")

# 运行回测
result = subprocess.run([
    'python3', 'skills/alpha-factor-lab/scripts/factor_backtest.py',
    '--factor', 'data/factor_vw_rsd_v1_neg.csv',
    '--returns', 'data/csi1000_returns.csv',
    '--n-groups', '5',
    '--rebalance-freq', '20',
    '--forward-days', '20',
    '--cost', '0.002',
    '--output-report', 'output/vw_rsd_v1_neg_20d/report.json',
    '--output-dir', 'output/vw_rsd_v1_neg_20d/',
    '--factor-name', 'vw_rsd_v1_neg'
], capture_output=True, text=True, cwd='/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
