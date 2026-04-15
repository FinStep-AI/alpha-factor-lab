#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子：ROE同比变化 (roe_yoy_chg_v1)
逻辑：当季ROE - 去年同季ROE，捕捉盈利改善趋势
Barra风格：Growth
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    # 1. 读数据
    fund = pd.read_csv(DATA_DIR / "csi1000_fundamental_cache.csv")
    kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    
    # 标准化stock_code为字符串，去掉前导零对齐
    fund["stock_code"] = fund["stock_code"].astype(str).str.zfill(6)
    kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)
    
    # 2. 计算ROE同比变化
    fund["report_date"] = pd.to_datetime(fund["report_date"])
    fund = fund.sort_values(["stock_code", "report_date"])
    
    # 提取季度标识(月份)，用于同比匹配
    fund["quarter_month"] = fund["report_date"].dt.month
    fund["year"] = fund["report_date"].dt.year
    
    # 同比：当期ROE - 去年同季ROE
    fund_pivot = fund.pivot_table(index=["stock_code", "quarter_month"], columns="year", values="roe")
    
    records = []
    for (stock, qm), row in fund_pivot.iterrows():
        years = sorted([y for y in row.index if pd.notna(row[y])])
        for i in range(1, len(years)):
            curr_year = years[i]
            prev_year = years[i-1]
            # 确保是同比（差一年）
            if curr_year - prev_year == 1:
                roe_chg = row[curr_year] - row[prev_year]
                # 报告日期
                rd = pd.Timestamp(year=curr_year, month=qm, day=28 if qm != 2 else 28)
                # 实际设最后一天
                if qm == 3:
                    rd = pd.Timestamp(year=curr_year, month=3, day=31)
                elif qm == 6:
                    rd = pd.Timestamp(year=curr_year, month=6, day=30)
                elif qm == 9:
                    rd = pd.Timestamp(year=curr_year, month=9, day=30)
                elif qm == 12:
                    rd = pd.Timestamp(year=curr_year, month=12, day=31)
                records.append({
                    "stock_code": stock,
                    "report_date": rd,
                    "roe_yoy_chg": roe_chg
                })
    
    roe_chg_df = pd.DataFrame(records)
    print(f"ROE同比变化记录数: {len(roe_chg_df)}, 股票数: {roe_chg_df['stock_code'].nunique()}")
    print(f"report_date范围: {roe_chg_df['report_date'].min()} ~ {roe_chg_df['report_date'].max()}")
    print(f"roe_yoy_chg统计:\n{roe_chg_df['roe_yoy_chg'].describe()}")
    
    # 3. 映射到交易日
    # 财报发布延迟假设：
    # Q1(3/31) -> 4/30发布, Q2(6/30) -> 8/31发布, Q3(9/30) -> 10/31发布, Q4(12/31) -> 次年4/30发布
    def get_available_date(report_date):
        m = report_date.month
        y = report_date.year
        if m == 3:   # Q1 -> 4月底可用
            return pd.Timestamp(year=y, month=5, day=1)
        elif m == 6: # Q2 -> 8月底可用
            return pd.Timestamp(year=y, month=9, day=1)
        elif m == 9: # Q3 -> 10月底可用
            return pd.Timestamp(year=y, month=11, day=1)
        elif m == 12: # Q4 -> 次年4月底可用
            return pd.Timestamp(year=y+1, month=5, day=1)
        return report_date
    
    roe_chg_df["available_date"] = roe_chg_df["report_date"].apply(get_available_date)
    
    # 获取所有交易日
    kline["date"] = pd.to_datetime(kline["date"])
    trade_dates = sorted(kline["date"].unique())
    
    # 对每个交易日，取每只股票最新的available因子值
    all_stocks = kline["stock_code"].unique()
    
    # 构建每个股票的因子时间序列
    factor_records = []
    
    for stock in all_stocks:
        stock_fund = roe_chg_df[roe_chg_df["stock_code"] == stock].sort_values("available_date")
        if len(stock_fund) == 0:
            continue
        
        # 对交易日做forward fill
        stock_dates = kline[kline["stock_code"] == stock]["date"].unique()
        
        for td in stock_dates:
            # 找最新的available因子
            valid = stock_fund[stock_fund["available_date"] <= td]
            if len(valid) == 0:
                continue
            latest = valid.iloc[-1]
            factor_records.append({
                "date": td.strftime("%Y-%m-%d"),
                "stock_code": stock,
                "raw_factor": latest["roe_yoy_chg"]
            })
    
    factor_df = pd.DataFrame(factor_records)
    print(f"\n映射到交易日后: {len(factor_df)} 行, {factor_df['stock_code'].nunique()} 只股票")
    print(f"日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    
    # 4. 市值中性化
    # 用close * volume作为市值代理
    kline["log_mktcap"] = np.log(kline["close"] * kline["volume"] + 1)
    mktcap = kline[["date", "stock_code", "log_mktcap"]].copy()
    mktcap["date"] = mktcap["date"].dt.strftime("%Y-%m-%d")
    
    factor_df = factor_df.merge(mktcap, on=["date", "stock_code"], how="left")
    
    # 横截面中性化：每天对raw_factor做log_mktcap回归取残差
    def neutralize_cross_section(group):
        y = group["raw_factor"].values
        x = group["log_mktcap"].values
        
        # 去掉NaN
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 30:
            group["factor_value"] = np.nan
            return group
        
        y_clean = y[mask]
        x_clean = x[mask]
        
        # OLS回归取残差
        x_mat = np.column_stack([np.ones(len(x_clean)), x_clean])
        try:
            beta = np.linalg.lstsq(x_mat, y_clean, rcond=None)[0]
            residuals = y_clean - x_mat @ beta
        except:
            group["factor_value"] = np.nan
            return group
        
        # Winsorize (MAD 3倍)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) * 1.4826
        if mad > 0:
            lower = med - 3 * mad
            upper = med + 3 * mad
            residuals = np.clip(residuals, lower, upper)
        
        # Z-score标准化
        std = residuals.std()
        if std > 0:
            residuals = (residuals - residuals.mean()) / std
        
        result = np.full(len(y), np.nan)
        result[mask] = residuals
        group["factor_value"] = result
        return group
    
    factor_df = factor_df.groupby("date", group_keys=False).apply(neutralize_cross_section)
    
    # 5. 输出
    output = factor_df[["date", "stock_code", "factor_value"]].copy()
    # stock_code去掉前导零（与其他因子保持一致 - 检查格式）
    # 先检查returns的格式
    ret_sample = pd.read_csv(DATA_DIR / "csi1000_returns.csv", nrows=3)
    ret_sc = str(ret_sample["stock_code"].iloc[0])
    print(f"\nreturns stock_code样本: '{ret_sc}' (len={len(ret_sc)})")
    
    # 如果returns用的是整数格式
    if len(ret_sc) <= 4 or not ret_sc.startswith("0"):
        output["stock_code"] = output["stock_code"].apply(lambda x: str(int(x)))
    
    output_path = DATA_DIR / "factor_roe_yoy_chg_v1.csv"
    output.to_csv(output_path, index=False)
    
    # 统计
    valid = output["factor_value"].notna()
    print(f"\n最终输出: {len(output)} 行, 有效: {valid.sum()} ({valid.mean()*100:.1f}%)")
    print(f"因子值统计:\n{output.loc[valid, 'factor_value'].describe()}")
    print(f"\n保存到: {output_path}")

if __name__ == "__main__":
    main()
