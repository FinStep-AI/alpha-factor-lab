#!/usr/bin/env python3
"""增量更新中证1000 K线数据
数据源优先级: fintool（聚源）> 腾讯行情（fallback）
用法: python3 scripts/update_kline_csi1000.py [--days 10] [--source fintool|tencent|auto]
"""
import pandas as pd
import time
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
KLINE_FILE = os.path.join(DATA_DIR, 'csi1000_kline_raw.csv')
CODES_FILE = os.path.join(DATA_DIR, 'csi1000_codes.csv')


def fetch_kline_fintool(code, end_date, num=30):
    """fintool K线（聚源数据源，单次最多100条）"""
    from fintool_client import get_kline
    return get_kline(code, end_date, num)


def fetch_kline_tencent_fallback(code, days=30):
    """腾讯K线（fallback）"""
    from fetch_kline import fetch_kline_tencent
    return fetch_kline_tencent(code, days)


def _is_index_contaminated(code, klines):
    """检测fintool是否返回了指数数据而非股票数据
    fintool bug: 000xxx代码与指数代码冲突时可能返回指数行情
    检测方式: 股价>500或单日涨幅>50%视为异常（中证1000成分股不可能）
    """
    for kl in klines:
        close = kl.get('close', 0)
        pct = abs(kl.get('pct_change', 0))
        if close > 500 or pct > 50:
            return True
    return False


def fetch_one(code, end_date, num, source):
    """获取单只K线，支持 fallback"""
    if source in ('fintool', 'auto'):
        try:
            klines = fetch_kline_fintool(code, end_date, num)
            if klines:
                if _is_index_contaminated(code, klines):
                    # fintool返回了指数数据，强制fallback到腾讯
                    pass
                else:
                    return code, klines, 'fintool'
        except Exception as e:
            if source == 'fintool':
                return code, [], f'fintool_error: {str(e)[:60]}'

    # fallback to tencent
    if source in ('tencent', 'auto'):
        try:
            klines = fetch_kline_tencent_fallback(code, num)
            if klines:
                return code, klines, 'tencent'
        except Exception as e:
            return code, [], f'tencent_error: {str(e)[:60]}'

    return code, [], 'empty'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=10,
                        help='每只拉取K线天数（fintool=条数, tencent=天数）')
    parser.add_argument('--source', choices=['auto', 'fintool', 'tencent'], default='auto',
                        help='数据源（默认auto: fintool优先+腾讯fallback）')
    parser.add_argument('--workers', type=int, default=10,
                        help='并发线程数（仅fintool有效，无速率限制）')
    parser.add_argument('--refresh-codes', action='store_true',
                        help='从fintool重新拉取成分股列表')
    parser.add_argument('--end-date', default=None,
                        help='截止日期（默认今天）')
    args = parser.parse_args()

    if args.end_date is None:
        from datetime import date
        args.end_date = date.today().strftime('%Y-%m-%d')

    # ===== 1. 成分股列表 =====
    if args.refresh_codes:
        print("从fintool刷新中证1000成分股列表...")
        from fintool_client import get_constituents
        stocks = get_constituents("中证1000")
        codes = [s['security_code'] for s in stocks]
        pd.DataFrame({'stock_code': codes}).to_csv(CODES_FILE, index=False)
        print(f"  成分股: {len(codes)} 只, 已更新 {CODES_FILE}")
    else:
        codes_df = pd.read_csv(CODES_FILE)
        codes = [str(c).zfill(6) for c in codes_df['stock_code'].tolist()]
        print(f"股票池: {len(codes)} 只（本地列表）")

    # ===== 2. 读取现有数据 =====
    print(f"读取现有K线: {KLINE_FILE}")
    existing = pd.read_csv(KLINE_FILE)
    existing['stock_code'] = existing['stock_code'].astype(str).str.zfill(6)
    old_max_date = existing['date'].max()
    old_rows = len(existing)
    print(f"  现有: {existing['stock_code'].nunique()} 只, {old_rows} 行, 最新: {old_max_date}")

    # ===== 3. 批量拉取 =====
    new_records = []
    source_stats = {'fintool': 0, 'tencent': 0, 'empty': 0, 'error': 0}
    t0 = time.time()

    if args.source in ('fintool', 'auto') and args.workers > 1:
        # 多线程模式（fintool无速率限制）
        print(f"多线程拉取（{args.workers}线程, {args.source}模式）...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_one, c, args.end_date, args.days, args.source): c
                for c in codes
            }
            done = 0
            for future in as_completed(futures):
                code, klines, src = future.result()
                done += 1

                if 'error' in src:
                    source_stats['error'] += 1
                elif src == 'empty' or not klines:
                    source_stats['empty'] += 1
                else:
                    source_stats[src] = source_stats.get(src, 0) + 1
                    for kl in klines:
                        kl_date = kl.get('date', '')
                        if kl_date > old_max_date:
                            new_records.append({
                                'date': kl_date,
                                'stock_code': code,
                                'open': kl['open'],
                                'close': kl['close'],
                                'high': kl['high'],
                                'low': kl['low'],
                                'volume': kl.get('volume', 0),
                                'amount': kl.get('amount', 0),
                                'amplitude': kl.get('amplitude', 0),
                                'pct_change': kl.get('pct_change', 0),
                                'change': 0,
                                'turnover': kl.get('turnover', 0),
                            })

                if done % 100 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{len(codes)}] {rate:.1f}/s | "
                          f"fintool={source_stats.get('fintool',0)} "
                          f"tencent={source_stats.get('tencent',0)} "
                          f"empty={source_stats['empty']} "
                          f"err={source_stats['error']} | "
                          f"新增{len(new_records)}条")
    else:
        # 单线程模式（腾讯需要限速）
        print(f"单线程拉取（{args.source}模式）...")
        for i, code in enumerate(codes):
            code, klines, src = fetch_one(code, args.end_date, args.days, args.source)

            if 'error' in src:
                source_stats['error'] += 1
            elif src == 'empty' or not klines:
                source_stats['empty'] += 1
            else:
                source_stats[src] = source_stats.get(src, 0) + 1
                for kl in klines:
                    kl_date = kl.get('date', '')
                    if kl_date > old_max_date:
                        new_records.append({
                            'date': kl_date,
                            'stock_code': code,
                            'open': kl['open'],
                            'close': kl['close'],
                            'high': kl['high'],
                            'low': kl['low'],
                            'volume': kl.get('volume', 0),
                            'amount': kl.get('amount', 0),
                            'amplitude': kl.get('amplitude', 0),
                            'pct_change': kl.get('pct_change', 0),
                            'change': 0,
                            'turnover': kl.get('turnover', 0),
                        })

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(codes)}] {src} | 新增{len(new_records)}条")

            if args.source == 'tencent':
                time.sleep(0.15)

    elapsed = time.time() - t0
    print(f"\n拉取完成: {elapsed:.1f}秒")
    print(f"  数据源统计: {json.dumps(source_stats)}")
    print(f"  新增记录: {len(new_records)} 条")

    # ===== 4. 合并 =====
    if not new_records:
        print(f"无新数据（已是最新: {old_max_date}）")
        return

    print("合并数据...")
    new_df = pd.DataFrame(new_records)
    new_df['stock_code'] = new_df['stock_code'].astype(str).str.zfill(6)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'stock_code'], keep='last')
    combined = combined.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 补算 pct_change（覆盖fintool原始值，保证一致性）
    print("补算涨跌幅...")
    def calc_pct_change(group):
        group = group.sort_values('date')
        group['pct_change'] = group['close'].pct_change() * 100
        return group

    combined = combined.groupby('stock_code', group_keys=False).apply(calc_pct_change)

    # 保存
    combined.to_csv(KLINE_FILE, index=False)
    new_max_date = combined['date'].max()
    new_dates = sorted(new_df['date'].unique())
    print(f"\n✅ 完成!")
    print(f"  更新前: {old_rows} 行, 最新 {old_max_date}")
    print(f"  更新后: {len(combined)} 行, 最新 {new_max_date}")
    print(f"  新增: {len(new_records)} 条, 覆盖 {len(new_dates)} 个交易日")
    print(f"  新日期: {new_dates}")

    # 同步收益率文件
    returns_file = os.path.join(DATA_DIR, 'csi1000_returns.csv')
    if os.path.exists(returns_file):
        print(f"\n更新收益率文件...")
        ret_df = combined[['date', 'stock_code', 'close']].copy()
        ret_df = ret_df.sort_values(['stock_code', 'date'])
        ret_df['return'] = ret_df.groupby('stock_code')['close'].pct_change()
        ret_df = ret_df.dropna(subset=['return'])
        ret_df[['date', 'stock_code', 'return']].to_csv(returns_file, index=False)
        print(f"  收益率: {len(ret_df)} 行, 最新 {ret_df['date'].max()}")


if __name__ == '__main__':
    main()
