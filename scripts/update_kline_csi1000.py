#!/usr/bin/env python3
"""增量更新中证1000 K线数据（腾讯数据源）
用法: python3 scripts/update_kline_csi1000.py [--days 30]
"""
import pandas as pd
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_kline import fetch_kline_tencent

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
KLINE_FILE = os.path.join(DATA_DIR, 'csi1000_kline_raw.csv')
CODES_FILE = os.path.join(DATA_DIR, 'csi1000_codes.csv')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='每只股票拉取的K线天数（默认30，覆盖增量）')
    parser.add_argument('--batch-size', type=int, default=50, help='每批保存的股票数')
    parser.add_argument('--delay', type=float, default=0.15, help='每次请求间隔(秒)')
    args = parser.parse_args()

    # 读取现有数据
    print(f"读取现有K线数据: {KLINE_FILE}")
    existing = pd.read_csv(KLINE_FILE)
    existing['stock_code'] = existing['stock_code'].astype(str).str.zfill(6)
    old_max_date = existing['date'].max()
    old_rows = len(existing)
    print(f"  现有: {existing['stock_code'].nunique()} 只, {old_rows} 行, 最新日期: {old_max_date}")

    # 读取股票列表
    codes = pd.read_csv(CODES_FILE)['stock_code'].tolist()
    codes = [str(c).zfill(6) for c in codes]
    print(f"  股票池: {len(codes)} 只")

    # 增量拉取 — 先存到临时CSV，边拉边写，避免进程中断丢数据
    tmp_file = KLINE_FILE + '.incremental.csv'
    new_records = []
    fail_count = 0
    success_count = 0
    empty_count = 0
    total_new = 0

    for i, code in enumerate(codes):
        try:
            klines = fetch_kline_tencent(code, args.days)
            if klines:
                for kl in klines:
                    if kl['date'] > old_max_date:
                        new_records.append({
                            'date': kl['date'],
                            'stock_code': code,
                            'open': kl['open'],
                            'close': kl['close'],
                            'high': kl['high'],
                            'low': kl['low'],
                            'volume': kl['volume'],
                            'amount': 0,
                            'amplitude': 0,
                            'pct_change': 0,
                            'change': 0,
                            'turnover': 0,
                        })
                success_count += 1
            else:
                empty_count += 1
            fail_count = 0
        except Exception as e:
            fail_count += 1
            if fail_count >= 10:
                print(f"  [!] 连续失败10次，等15秒... (last: {code}: {str(e)[:60]})")
                time.sleep(15)
                fail_count = 0
            else:
                time.sleep(0.5)
            continue

        # 每100只落盘一次增量数据
        if (i + 1) % args.batch_size == 0 and new_records:
            batch_df = pd.DataFrame(new_records)
            if os.path.exists(tmp_file):
                batch_df.to_csv(tmp_file, mode='a', header=False, index=False)
            else:
                batch_df.to_csv(tmp_file, index=False)
            total_new += len(new_records)
            new_dates = set(r['date'] for r in new_records)
            print(f"  [{i+1}/{len(codes)}] 成功{success_count} 空{empty_count} 累计新增{total_new}条 新日期: {sorted(new_dates)[-3:] if new_dates else '无'}")
            new_records = []

        time.sleep(args.delay)

    # 最后一批
    if new_records:
        batch_df = pd.DataFrame(new_records)
        if os.path.exists(tmp_file):
            batch_df.to_csv(tmp_file, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(tmp_file, index=False)
        total_new += len(new_records)

    if total_new == 0 and not os.path.exists(tmp_file):
        print(f"\n无新数据（数据已是最新: {old_max_date}）")
        return

    # 合并
    print(f"\n合并数据...")
    if os.path.exists(tmp_file):
        new_df = pd.read_csv(tmp_file)
        new_df['stock_code'] = new_df['stock_code'].astype(str).str.zfill(6)
    else:
        print("无增量文件，退出")
        return

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'stock_code'], keep='last')
    combined = combined.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 补算 pct_change
    print(f"补算涨跌幅...")
    def calc_pct_change(group):
        group = group.sort_values('date')
        group['pct_change'] = group['close'].pct_change() * 100
        return group
    
    combined = combined.groupby('stock_code', group_keys=False).apply(calc_pct_change)

    # 保存
    combined.to_csv(KLINE_FILE, index=False)
    new_max_date = combined['date'].max()
    new_dates = sorted(new_df['date'].unique())
    print(f"\n完成!")
    print(f"  更新前: {old_rows} 行, 最新 {old_max_date}")
    print(f"  更新后: {len(combined)} 行, 最新 {new_max_date}")
    print(f"  新增: {total_new} 条, 覆盖 {len(new_dates)} 个交易日")
    print(f"  新日期: {new_dates}")

    # 清理临时文件
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        print(f"  已清理临时文件")

    # 同步更新 returns 文件
    returns_file = os.path.join(DATA_DIR, 'csi1000_returns.csv')
    if os.path.exists(returns_file):
        print(f"\n更新收益率文件: {returns_file}")
        ret_df = combined[['date', 'stock_code', 'close']].copy()
        ret_df = ret_df.sort_values(['stock_code', 'date'])
        ret_df['return'] = ret_df.groupby('stock_code')['close'].pct_change()
        ret_df = ret_df.dropna(subset=['return'])
        ret_df[['date', 'stock_code', 'return']].to_csv(returns_file, index=False)
        print(f"  收益率文件已更新: {len(ret_df)} 行, 最新 {ret_df['date'].max()}")


if __name__ == '__main__':
    main()
