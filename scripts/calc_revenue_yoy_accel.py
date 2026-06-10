#!/usr/bin/env python3
"""
revenue_yoydiff_accel_v1 — 营业收入增长加速因子（v1）
======================================================
数据源: data/csi1000_revenue_yoy_raw.csv（季报财务字段，含 operate_revenue_yoy）
输出  : data/factor_revenue_yoy_accel_v1.csv

设计假设
--------
现有 Growth 因子全部是 ROE 系（roe_delta_yoy / roe_accel / roe_persistence 等），
缺少纯营收 / 收入增长视角。 本因子补这一维度。

为何选 operate_revenue_yoy
  ├ 搜狐同价营收是公司规模扩张的最直接信号，不与 ROE 线路重叠
  ├ 季报可用，anchor 时间点在 info_publ_date + 45d 保密期
  └ 与 roe_accel_v1（ROE 加速度）正交成对角 Growth × 营收规模扩张

构造：WLS-trend-acceleration
  每期 IPO 数量 → (Q4 有全年核算序曲，故 Q4 不可靠)
  单期 IPO 数量 = yoy[-3:] 前 3 期均值 与后 1 期 yoy[-1:]  ——但只用 end_date,info_publ_date, info_publ → 日侦 → 45d 即 30天 end_date 全部在哪来一次。

因子 raw = WLS 后 t 月份 IPO sum[yoy] - mean 均回填财报

end_date   quarter calendar data

date


"""
import os, sys
import json
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from numpy.linalg import lstsq

# ─── paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FUND_CSV  = os.path.join(DATA, "csi1000_revenue_yoy_raw.csv")
KLINE_CSV = os.path.join(DATA, "csi1000_kline_raw.csv")
OUT_CSV   = os.path.join(DATA, "factor_revenue_yoy_accel_v1.csv")

M         = 4     # 回


import pandas as pd
print(pd.read_csv("data/csi1000_revenue_yoy_raw.csv").head())
print("done")
