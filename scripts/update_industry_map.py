#!/usr/bin/env python3
"""从fintool刷新中证1000申万行业映射（替代新浪55%覆盖率 → 聚源100%覆盖）
用法: python3 scripts/update_industry_map.py [--output data/csi1000_industry_sws.json]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fintool_client import get_industry_sws

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CODES_FILE = os.path.join(DATA_DIR, 'csi1000_codes.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'csi1000_industry_sws.json')


def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=OUTPUT_FILE)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()

    # 读取股票列表
    codes_df = pd.read_csv(CODES_FILE)
    codes = [str(c).zfill(6) for c in codes_df['stock_code'].tolist()]
    print(f"股票池: {len(codes)} 只")

    # 批量获取行业
    results = {}
    errors = []
    t0 = time.time()

    def _fetch(code):
        try:
            info = get_industry_sws(code)
            return code, info, None
        except Exception as e:
            return code, None, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, c): c for c in codes}
        done = 0
        for future in as_completed(futures):
            code, info, err = future.result()
            done += 1
            if err:
                errors.append((code, err))
            elif info:
                results[code] = {
                    'name': info.get('security_name', ''),
                    'industry_standard': info.get('industry_standard', ''),
                    'first': info.get('first_industry_name', ''),
                    'second': info.get('second_industry_name', ''),
                    'third': info.get('third_industry_name', ''),
                }

            if done % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{done}/{len(codes)}] {elapsed:.1f}s | 成功={len(results)} 失败={len(errors)}")

    elapsed = time.time() - t0
    coverage = len(results) / len(codes) * 100

    # 统计行业分布
    industry_count = {}
    for code, info in results.items():
        ind = info.get('first', '未知')
        industry_count[ind] = industry_count.get(ind, 0) + 1

    print(f"\n完成: {elapsed:.1f}秒")
    print(f"覆盖率: {len(results)}/{len(codes)} ({coverage:.1f}%)")
    print(f"一级行业: {len(industry_count)} 个")
    print(f"失败: {len(errors)} 只")

    # TOP10 行业
    sorted_ind = sorted(industry_count.items(), key=lambda x: -x[1])[:10]
    for ind, cnt in sorted_ind:
        print(f"  {ind}: {cnt}")

    # 保存
    output = {
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total': len(codes),
        'covered': len(results),
        'coverage_pct': round(coverage, 1),
        'industries': len(industry_count),
        'industry_distribution': industry_count,
        'stocks': results,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {args.output}")

    if errors:
        print(f"\n⚠️ 失败股票({len(errors)}): {[e[0] for e in errors[:10]]}")


if __name__ == '__main__':
    main()
