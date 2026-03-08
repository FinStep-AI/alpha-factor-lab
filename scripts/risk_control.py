#!/usr/bin/env python3
"""
第三期改造：风控模块
- ST过滤：调仓时自动排除ST/*ST股票
- 减持预警：检测持仓/候选标的的股东减持计划
- 行业中性约束：限制单行业暴露不超过阈值
- 组合风控：个股集中度、组合止损

用法:
  # 过滤ST股
  python3 scripts/risk_control.py --action filter-st --codes 000001,000004,300579

  # 检查减持预警
  python3 scripts/risk_control.py --action reduction-alert --codes 301177,300579,688232

  # 行业中性检查
  python3 scripts/risk_control.py --action industry-check --codes 301177,300579,688232 --max-weight 0.30

  # 完整风控（对候选股票列表做全套检查）
  python3 scripts/risk_control.py --action full-check --codes 301177,300579,688232

  # 检查当前持仓的风险状态
  python3 scripts/risk_control.py --action portfolio-risk --data paper-trading-data.json --player quant

数据源: fintool API（聚源）
"""
import sys
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fintool_client import (get_st_stocks, get_transfer_plan, _call)

# ============================================================
# 配置
# ============================================================
# 行业暴露上限（单个申万一级行业占组合权重上限）
MAX_INDUSTRY_WEIGHT = 0.25  # 25%
# 个股集中度上限
MAX_SINGLE_STOCK_WEIGHT = 0.10  # 10%
# 组合最大回撤止损线
MAX_DRAWDOWN_STOP = 0.15  # 15%
# 减持计划预警天数（距离减持结束日期还有N天以内视为活跃减持）
REDUCTION_ALERT_DAYS = 90
# 减持比例预警线
REDUCTION_RATIO_ALERT = 0.02  # 2%

# ============================================================
# ST过滤
# ============================================================
_st_cache = None
_st_cache_time = 0
ST_CACHE_TTL = 3600  # 1小时缓存

def get_st_list(force_refresh=False) -> Set[str]:
    """获取当前所有ST/*ST股票代码集合（带缓存）"""
    global _st_cache, _st_cache_time
    
    now = time.time()
    if not force_refresh and _st_cache is not None and (now - _st_cache_time) < ST_CACHE_TTL:
        return _st_cache
    
    st_codes = set()
    
    # fintool的get_security_change返回raw dict，需要手动解析
    try:
        result = _call('company_info', 'get_security_change', {})
        if isinstance(result, dict):
            data = result.get('data', result)
            items = data.get('items', [])
            total_pages = data.get('total_pages', 1)
            
            for item in items:
                code = item.get('security_code', '')
                if code:
                    st_codes.add(code)
            
            # 拉取剩余页
            for page in range(2, total_pages + 1):
                try:
                    r2 = _call('company_info', 'get_security_change', {'page': page})
                    if isinstance(r2, dict):
                        d2 = r2.get('data', r2)
                        for item in d2.get('items', []):
                            code = item.get('security_code', '')
                            if code:
                                st_codes.add(code)
                except:
                    pass
        elif isinstance(result, list):
            for item in result:
                code = item.get('security_code', '') if isinstance(item, dict) else ''
                if code:
                    st_codes.add(code)
    except Exception as e:
        print(f"[WARN] 获取ST列表失败: {e}")
    
    _st_cache = st_codes
    _st_cache_time = now
    print(f"[风控] ST列表: {len(st_codes)} 只")
    return st_codes


def filter_st(codes: List[str]) -> Tuple[List[str], List[str]]:
    """
    过滤ST股票
    返回: (通过的代码列表, 被过滤的ST代码列表)
    """
    st_set = get_st_list()
    
    passed = []
    filtered = []
    for code in codes:
        # 去掉后缀
        bare = code.split('.')[0]
        if bare in st_set:
            filtered.append(code)
        else:
            passed.append(code)
    
    if filtered:
        print(f"[风控] ST过滤: 移除 {len(filtered)} 只 → {filtered}")
    return passed, filtered


# ============================================================
# 减持预警
# ============================================================
def check_reduction_alert(codes: List[str], max_workers: int = 5) -> List[Dict]:
    """
    批量检查减持计划
    返回: 有活跃减持计划的股票列表
    """
    alerts = []
    today = datetime.now()
    
    def check_one(code: str):
        bare = code.split('.')[0]
        try:
            plans = get_transfer_plan(bare)
            if not plans:
                return None
            
            active_plans = []
            for plan in plans:
                plan_type = plan.get('transfer_plan_type', '')
                # 只关注减持计划
                if '减持' not in plan_type:
                    continue
                
                # 检查是否在活跃期内
                end_date_str = plan.get('promise_end_date', '')
                begin_date_str = plan.get('promise_begin_date', '')
                procedure = plan.get('event_procedure', '') or ''
                
                # 已完成的跳过
                if '完成' in procedure:
                    continue
                
                # 解析结束日期
                is_active = False
                end_date = None
                if end_date_str:
                    try:
                        end_date = datetime.strptime(end_date_str[:10], '%Y-%m-%d')
                        # 还在减持期内或刚结束不久
                        if end_date >= today - timedelta(days=30):
                            is_active = True
                    except:
                        pass
                
                if not end_date_str:
                    # 无结束日期的承诺（如"不减持"承诺），跳过
                    continue
                
                if is_active:
                    # 获取减持比例
                    ratio = plan.get('reduce_ratio_ceiling')
                    try:
                        ratio = float(ratio) if ratio else None
                    except:
                        ratio = None
                    
                    active_plans.append({
                        'shareholder': plan.get('shareholder_name', ''),
                        'type': plan_type,
                        'begin': begin_date_str[:10] if begin_date_str else '',
                        'end': end_date_str[:10] if end_date_str else '',
                        'ratio': ratio,
                        'size': plan.get('reduce_size', ''),
                        'method': plan.get('trade_type_statment', '') or plan.get('trade_type', ''),
                    })
            
            if active_plans:
                return {
                    'code': code,
                    'name': plans[0].get('security_name', ''),
                    'plans': active_plans
                }
            return None
        except Exception as e:
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(check_one, c): c for c in codes}
        for fut in as_completed(futs):
            result = fut.result()
            if result:
                alerts.append(result)
    
    if alerts:
        print(f"[风控] 减持预警: {len(alerts)} 只有活跃减持计划")
        for a in alerts:
            for p in a['plans']:
                level = "⚠️高" if (p['ratio'] and p['ratio'] >= REDUCTION_RATIO_ALERT) else "ℹ️低"
                print(f"  {level} {a['code']} {a['name']}: {p['shareholder'][:20]} "
                      f"减持{p['ratio']*100:.1f}% ({p['begin']}~{p['end']})" 
                      if p['ratio'] else
                      f"  {level} {a['code']} {a['name']}: {p['shareholder'][:20]} "
                      f"减持 ({p['begin']}~{p['end']})")
    else:
        print(f"[风控] 减持预警: 无活跃减持计划")
    
    return alerts


# ============================================================
# 行业中性约束
# ============================================================
def load_industry_map(industry_file: str = None) -> Dict[str, str]:
    """加载申万行业映射（code -> 申万一级行业名）"""
    if industry_file is None:
        # 默认路径
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        industry_file = os.path.join(base, 'data', 'csi1000_industry_sws.json')
    
    if not os.path.exists(industry_file):
        print(f"[WARN] 行业映射文件不存在: {industry_file}")
        return {}
    
    with open(industry_file) as f:
        data = json.load(f)
    
    stocks = data.get('stocks', data)
    if not isinstance(stocks, dict):
        return {}
    
    mapping = {}
    for code, info in stocks.items():
        if isinstance(info, dict):
            mapping[code] = info.get('first', '')
        elif isinstance(info, str):
            mapping[code] = info
    
    return mapping


def check_industry_concentration(codes: List[str], weights: Dict[str, float] = None,
                                  max_weight: float = None, industry_map: Dict[str, str] = None) -> Dict:
    """
    检查行业集中度
    codes: 股票代码列表
    weights: {code: weight} 权重映射，None则等权
    max_weight: 单行业最大权重（默认用全局配置）
    industry_map: 行业映射，None自动加载
    
    返回: {
        'passed': True/False,
        'industry_weights': {industry: weight},
        'violations': [{'industry': ..., 'weight': ..., 'codes': [...]}],
        'suggestions': [{'action': 'remove', 'code': ..., 'reason': ...}]
    }
    """
    if max_weight is None:
        max_weight = MAX_INDUSTRY_WEIGHT
    
    if industry_map is None:
        industry_map = load_industry_map()
    
    n = len(codes)
    if n == 0:
        return {'passed': True, 'industry_weights': {}, 'violations': [], 'suggestions': []}
    
    # 计算每只股票的权重
    if weights is None:
        w = 1.0 / n
        stock_weights = {c: w for c in codes}
    else:
        stock_weights = weights
    
    # 按行业汇总
    industry_detail = {}  # industry -> [(code, weight)]
    unknown = []
    
    for code in codes:
        bare = code.split('.')[0]
        industry = industry_map.get(bare, '')
        if not industry:
            unknown.append(code)
            industry = '未知'
        
        if industry not in industry_detail:
            industry_detail[industry] = []
        industry_detail[industry].append((code, stock_weights.get(code, 1.0/n)))
    
    # 计算行业权重
    industry_weights = {}
    for ind, stocks in industry_detail.items():
        industry_weights[ind] = sum(w for _, w in stocks)
    
    # 检查违规
    violations = []
    for ind, weight in sorted(industry_weights.items(), key=lambda x: -x[1]):
        if weight > max_weight:
            codes_in_ind = [c for c, _ in industry_detail[ind]]
            violations.append({
                'industry': ind,
                'weight': round(weight, 4),
                'limit': max_weight,
                'codes': codes_in_ind,
                'count': len(codes_in_ind)
            })
    
    # 生成建议
    suggestions = []
    if violations:
        for v in violations:
            excess = v['weight'] - max_weight
            # 建议移除超标行业中权重最小的股票（影响最小）
            ind_stocks = industry_detail[v['industry']]
            ind_stocks_sorted = sorted(ind_stocks, key=lambda x: x[1])  # 按权重从小到大
            
            removed_weight = 0
            for code, w in ind_stocks_sorted:
                if removed_weight >= excess:
                    break
                suggestions.append({
                    'action': 'remove',
                    'code': code,
                    'industry': v['industry'],
                    'reason': f"{v['industry']}超配({v['weight']*100:.1f}%>{max_weight*100:.0f}%)"
                })
                removed_weight += w
    
    passed = len(violations) == 0
    
    if not passed:
        print(f"[风控] 行业集中度: ❌ {len(violations)} 个行业超标")
        for v in violations:
            print(f"  ⚠️ {v['industry']}: {v['weight']*100:.1f}% > {max_weight*100:.0f}% "
                  f"({v['count']}只: {','.join(v['codes'][:5])})")
    else:
        # 打印行业分布
        print(f"[风控] 行业集中度: ✅ 通过 ({len(industry_weights)}个行业)")
        top3 = sorted(industry_weights.items(), key=lambda x: -x[1])[:3]
        for ind, w in top3:
            print(f"  {ind}: {w*100:.1f}%")
    
    return {
        'passed': passed,
        'industry_weights': {k: round(v, 4) for k, v in industry_weights.items()},
        'violations': violations,
        'suggestions': suggestions,
        'unknown_codes': unknown
    }


# ============================================================
# 组合风控
# ============================================================
def check_portfolio_risk(data: dict, player_id: str) -> Dict:
    """
    检查选手的组合风险指标
    返回: {
        'max_drawdown': float,
        'stop_loss_triggered': bool,
        'concentration': {'max_stock_weight': ..., 'top_stock': ...},
        'industry': {...},
        'reduction_alerts': [...],
        'st_holdings': [...],
        'risk_level': 'low'|'medium'|'high'|'critical'
    }
    """
    player = data['players'][player_id]
    portfolio = player['portfolio']
    positions = portfolio.get('positions', {})
    total_value = portfolio.get('total_value', 0)
    
    result = {
        'risk_level': 'low',
        'warnings': [],
        'actions': []
    }
    
    if not positions or total_value <= 0:
        result['risk_level'] = 'low'
        result['warnings'].append('无持仓')
        return result
    
    # 1. 个股集中度
    stock_weights = {}
    for code, pos in positions.items():
        mv = pos.get('market_value', 0) or (pos.get('current_price', 0) * pos.get('volume', 0))
        stock_weights[code] = mv / total_value if total_value > 0 else 0
    
    max_stock = max(stock_weights.items(), key=lambda x: x[1])
    if max_stock[1] > MAX_SINGLE_STOCK_WEIGHT:
        result['warnings'].append(
            f"个股集中度超标: {max_stock[0]}({positions[max_stock[0]].get('name','')}) "
            f"占比{max_stock[1]*100:.1f}% > {MAX_SINGLE_STOCK_WEIGHT*100:.0f}%"
        )
    
    result['concentration'] = {
        'max_stock_weight': round(max_stock[1], 4),
        'max_stock_code': max_stock[0],
        'max_stock_name': positions[max_stock[0]].get('name', ''),
    }
    
    # 2. ST检查
    codes = list(positions.keys())
    _, st_holdings = filter_st(codes)
    if st_holdings:
        result['st_holdings'] = st_holdings
        result['warnings'].append(f"持有ST股票: {st_holdings}")
        result['actions'].append({
            'action': 'sell',
            'codes': st_holdings,
            'reason': 'ST风险'
        })
    
    # 3. 行业集中度
    industry_result = check_industry_concentration(
        codes, weights=stock_weights
    )
    result['industry'] = industry_result
    if not industry_result['passed']:
        for v in industry_result['violations']:
            result['warnings'].append(
                f"行业超配: {v['industry']} {v['weight']*100:.1f}%"
            )
    
    # 4. 减持预警
    reduction_alerts = check_reduction_alert(codes)
    result['reduction_alerts'] = reduction_alerts
    for alert in reduction_alerts:
        for p in alert['plans']:
            if p.get('ratio') and p['ratio'] >= REDUCTION_RATIO_ALERT:
                result['warnings'].append(
                    f"大额减持预警: {alert['code']}({alert['name']}) "
                    f"{p['shareholder'][:15]} 减持≤{p['ratio']*100:.1f}%"
                )
    
    # 5. 最大回撤
    stats = player.get('stats', {})
    mdd = stats.get('max_drawdown_pct', 0) / 100
    result['max_drawdown'] = mdd
    if mdd > MAX_DRAWDOWN_STOP:
        result['warnings'].append(
            f"组合回撤超止损线: {mdd*100:.1f}% > {MAX_DRAWDOWN_STOP*100:.0f}%"
        )
        result['actions'].append({
            'action': 'reduce_position',
            'reason': f'最大回撤{mdd*100:.1f}%超过{MAX_DRAWDOWN_STOP*100:.0f}%止损线'
        })
    
    # 6. 综合风险评级
    n_warnings = len(result['warnings'])
    has_st = bool(st_holdings) if 'st_holdings' in result else False
    has_stop_loss = mdd > MAX_DRAWDOWN_STOP
    
    if has_st or has_stop_loss:
        result['risk_level'] = 'critical'
    elif n_warnings >= 3:
        result['risk_level'] = 'high'
    elif n_warnings >= 1:
        result['risk_level'] = 'medium'
    else:
        result['risk_level'] = 'low'
    
    return result


# ============================================================
# 调仓前完整风控检查
# ============================================================
def pre_rebalance_check(candidate_codes: List[str], 
                         max_industry_weight: float = None,
                         industry_map: Dict[str, str] = None) -> Dict:
    """
    调仓前对候选标的做完整风控检查
    
    返回: {
        'passed_codes': [...],  # 通过风控的代码
        'rejected': [...],      # 被拒绝的代码及原因
        'warnings': [...],      # 警告但不拒绝
        'industry_adjustment': {...},  # 行业调整建议
    }
    """
    if max_industry_weight is None:
        max_industry_weight = MAX_INDUSTRY_WEIGHT
    if industry_map is None:
        industry_map = load_industry_map()
    
    print(f"\n{'='*60}")
    print(f"[风控] 调仓前检查: {len(candidate_codes)} 只候选")
    print(f"{'='*60}")
    
    result = {
        'passed_codes': [],
        'rejected': [],
        'warnings': [],
        'industry_adjustment': None,
    }
    
    # Step 1: ST过滤
    non_st, st_codes = filter_st(candidate_codes)
    for code in st_codes:
        result['rejected'].append({
            'code': code,
            'reason': 'ST股票',
            'severity': 'critical'
        })
    
    # Step 2: 减持预警
    reduction_alerts = check_reduction_alert(non_st)
    high_risk_codes = set()
    for alert in reduction_alerts:
        for plan in alert['plans']:
            if plan.get('ratio') and plan['ratio'] >= REDUCTION_RATIO_ALERT:
                # 大额减持 → 直接剔除
                high_risk_codes.add(alert['code'])
                result['rejected'].append({
                    'code': alert['code'],
                    'name': alert['name'],
                    'reason': f"大额减持预警: {plan['shareholder'][:15]} 减持≤{plan['ratio']*100:.1f}%",
                    'severity': 'high'
                })
            else:
                # 小额减持 → 警告
                result['warnings'].append({
                    'code': alert['code'],
                    'name': alert['name'],
                    'message': f"有减持计划: {plan['shareholder'][:15]} ({plan['begin']}~{plan['end']})",
                    'severity': 'medium'
                })
    
    # 过滤大额减持
    remaining = [c for c in non_st if c not in high_risk_codes]
    
    # Step 3: 行业集中度检查
    industry_result = check_industry_concentration(
        remaining, max_weight=max_industry_weight, industry_map=industry_map
    )
    
    if not industry_result['passed']:
        # 有行业超标，需要调整
        result['industry_adjustment'] = industry_result
        
        # 按建议移除超标行业的股票
        remove_codes = set(s['code'] for s in industry_result.get('suggestions', []))
        for suggestion in industry_result.get('suggestions', []):
            result['rejected'].append({
                'code': suggestion['code'],
                'reason': suggestion['reason'],
                'severity': 'medium'
            })
        remaining = [c for c in remaining if c not in remove_codes]
        
        # 重新检查调整后的行业分布
        if remaining:
            final_check = check_industry_concentration(
                remaining, max_weight=max_industry_weight, industry_map=industry_map
            )
            result['industry_adjustment']['after_adjustment'] = final_check
    
    result['passed_codes'] = remaining
    
    # 汇总
    print(f"\n[风控] 结果汇总:")
    print(f"  输入: {len(candidate_codes)} 只")
    print(f"  通过: {len(result['passed_codes'])} 只")
    print(f"  剔除: {len(result['rejected'])} 只")
    print(f"  警告: {len(result['warnings'])} 条")
    
    if result['rejected']:
        print(f"\n  剔除明细:")
        for r in result['rejected']:
            print(f"    ❌ {r['code']} {r.get('name','')} — {r['reason']}")
    
    if result['warnings']:
        print(f"\n  警告明细:")
        for w in result['warnings']:
            print(f"    ⚠️ {w['code']} {w.get('name','')} — {w['message']}")
    
    return result


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="因子猎人风控模块")
    parser.add_argument("--action", required=True,
                       choices=["filter-st", "reduction-alert", "industry-check", 
                               "full-check", "portfolio-risk"],
                       help="风控动作")
    parser.add_argument("--codes", help="股票代码列表（逗号分隔）")
    parser.add_argument("--data", default="paper-trading-data.json", help="竞赛数据文件")
    parser.add_argument("--player", default="quant", help="选手ID")
    parser.add_argument("--max-weight", type=float, default=0.25, help="行业最大权重")
    parser.add_argument("--workers", type=int, default=5, help="并发数")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    args = parser.parse_args()
    
    codes = [c.strip() for c in args.codes.split(',')] if args.codes else []
    
    if args.action == "filter-st":
        passed, filtered = filter_st(codes)
        if args.json:
            print(json.dumps({'passed': passed, 'filtered': filtered}, ensure_ascii=False))
        else:
            print(f"通过: {passed}")
            print(f"过滤: {filtered}")
    
    elif args.action == "reduction-alert":
        alerts = check_reduction_alert(codes, args.workers)
        if args.json:
            print(json.dumps(alerts, ensure_ascii=False, indent=2))
    
    elif args.action == "industry-check":
        result = check_industry_concentration(codes, max_weight=args.max_weight)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.action == "full-check":
        result = pre_rebalance_check(codes, max_industry_weight=args.max_weight)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    
    elif args.action == "portfolio-risk":
        if not os.path.exists(args.data):
            print(f"文件不存在: {args.data}")
            sys.exit(1)
        with open(args.data) as f:
            data = json.load(f)
        result = check_portfolio_risk(data, args.player)
        
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            emoji = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
            print(f"\n{'='*60}")
            print(f"[风控] 组合风险报告 — {args.player}")
            print(f"{'='*60}")
            print(f"风险等级: {emoji.get(result['risk_level'], '❓')} {result['risk_level'].upper()}")
            
            if result.get('concentration'):
                c = result['concentration']
                print(f"\n个股集中度:")
                print(f"  最大持仓: {c['max_stock_code']}({c['max_stock_name']}) "
                      f"{c['max_stock_weight']*100:.1f}%")
            
            if result.get('max_drawdown') is not None:
                print(f"\n最大回撤: {result['max_drawdown']*100:.1f}% "
                      f"(止损线: {MAX_DRAWDOWN_STOP*100:.0f}%)")
            
            if result.get('warnings'):
                print(f"\n警告 ({len(result['warnings'])}):")
                for w in result['warnings']:
                    print(f"  ⚠️ {w}")
            
            if result.get('actions'):
                print(f"\n建议操作:")
                for a in result['actions']:
                    print(f"  → {a['action']}: {a.get('codes', '')} {a['reason']}")
            
            if not result.get('warnings'):
                print(f"\n✅ 无风险警告")


if __name__ == "__main__":
    main()
