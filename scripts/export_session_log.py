#!/usr/bin/env python3
"""
从 OpenClaw session JSONL 文件导出会话日志（markdown格式）
格式参考 logs/2026-02-20.md

用法: python3 export_session_log.py <jsonl_file> <output_file> [date_label]
"""
import json, sys, os
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))

def format_ts(ts_str):
    """ISO时间字符串 → 本地时间"""
    if not ts_str:
        return "?"
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.astimezone(CST).strftime("%H:%M:%S")
    except:
        return "?"

def extract_text(content):
    """从 content 数组提取文本和工具调用"""
    parts = []
    tools = []
    if isinstance(content, str):
        return content, tools
    if not isinstance(content, list):
        return str(content), tools
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            t = item.get('type', '')
            if t == 'text':
                text = item.get('text', '')
                if text.strip():
                    parts.append(text.strip())
            elif t == 'toolCall':
                name = item.get('name', '?')
                args = item.get('arguments', {})
                arg_str = ', '.join(f'{k}={repr(v)[:80]}' for k, v in (args or {}).items()) if args else ''
                tools.append(f"🔧 `{name}({arg_str})`")
    return '\n'.join(parts), tools

def main():
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <jsonl_file> <output_file> [date_label]")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    output_path = sys.argv[2]
    date_label = sys.argv[3] if len(sys.argv) > 3 else "Unknown"
    
    # 读取所有事件
    events = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # 统计
    total_calls = 0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cost = 0.0
    models_used = set()
    entries = []
    
    for evt in events:
        if evt.get('type') != 'message':
            continue
        
        msg = evt.get('message', {})
        role = msg.get('role', '')
        ts = evt.get('timestamp') or msg.get('timestamp')
        content = msg.get('content', '')
        
        if role == 'assistant':
            model = msg.get('model', '')
            if model:
                models_used.add(model)
            
            usage = msg.get('usage', {})
            inp = usage.get('input', 0) or 0
            out = usage.get('output', 0) or 0
            cache_r = usage.get('cacheRead', 0) or 0
            total_tok = usage.get('totalTokens', 0) or (inp + out + cache_r)
            cost_info = usage.get('cost', {})
            cost_total = cost_info.get('total', 0) if cost_info else 0
            
            if total_tok > 0:
                total_calls += 1
                total_input += inp
                total_output += out
                total_cache_read += cache_r
                total_cost += (cost_total or 0)
            
            text, tools = extract_text(content)
            
            token_info = ""
            if total_tok > 0:
                token_info = f" | `{model}` | {total_tok:,} tokens (out:{out:,})"
            
            entry = f"### 🤖 助手 — {format_ts(ts)}{token_info}\n\n"
            if text:
                if len(text) > 500:
                    text = text[:500] + "\n..."
                entry += text + "\n"
            if tools:
                entry += '\n'.join(tools) + "\n"
            entries.append(entry)
        
        elif role == 'user':
            text, _ = extract_text(content)
            if text:
                if len(text) > 300:
                    text = text[:300] + "\n..."
                entry = f"### 👤 用户 — {format_ts(ts)}\n\n{text}\n"
                entries.append(entry)
        
        elif role == 'toolResult':
            # 简要记录工具结果
            name = msg.get('toolName', '?')
            is_err = msg.get('isError', False)
            text, _ = extract_text(content)
            status = "❌" if is_err else "✅"
            if text and len(text) > 200:
                text = text[:200] + "..."
            entries.append(f"### 🔧 {status} {name} — {format_ts(ts)}\n\n{text}\n")
    
    # 生成输出
    total_all = total_input + total_output + total_cache_read
    with open(output_path, 'w') as f:
        f.write(f"# 会话日志 — {date_label}\n\n")
        f.write("| 统计 | 数值 |\n|------|------|\n")
        f.write(f"| 模型调用次数 | {total_calls} |\n")
        f.write(f"| 总 tokens | {total_all:,} |\n")
        f.write(f"| 输入 tokens | {total_input:,} |\n")
        f.write(f"| 输出 tokens | {total_output:,} |\n")
        f.write(f"| 缓存读取 | {total_cache_read:,} |\n")
        if total_cost > 0:
            f.write(f"| 总费用 | ${total_cost:.4f} |\n")
        f.write(f"| 使用模型 | {', '.join(sorted(models_used)) or 'unknown'} |\n")
        f.write("\n---\n\n")
        
        for entry in entries:
            f.write(entry)
            f.write("\n---\n\n")
    
    print(f"导出完成: {output_path}")
    print(f"  调用次数: {total_calls}")
    print(f"  总tokens: {total_all:,}")
    print(f"  输出tokens: {total_output:,}")
    print(f"  缓存读取: {total_cache_read:,}")
    print(f"  消息条数: {len(entries)}")

if __name__ == '__main__':
    main()
