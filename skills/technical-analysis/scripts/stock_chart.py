#!/usr/bin/env python3
"""
股票技术分析图表生成器 - 多市场版
包含：K线、均线、BOLL、成交量、OBV、MACD、KDJ、RSI、买卖点标记
支持 A股、美股、港股数据格式
"""

import json
import sys
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
COLORS = {
    'up': '#ef5350', 'down': '#26a69a',
    'ma5': '#ff9800', 'ma10': '#2196f3', 'ma20': '#9c27b0', 'ma60': '#4caf50', 'ma120': '#795548',
    'boll_upper': '#e91e63', 'boll_lower': '#00bcd4', 'boll_mid': '#607d8b',
    'volume_up': '#ffcdd2', 'volume_down': '#b2dfdb',
    'obv': '#ff5722',
    'macd_positive': '#ef5350', 'macd_negative': '#26a69a',
    'dif': '#ff9800', 'dea': '#2196f3',
    'k_line': '#ff9800', 'd_line': '#2196f3', 'j_line': '#9c27b0',
    'rsi6': '#ff9800', 'rsi12': '#2196f3',
    'buy_signal': '#ff1744', 'sell_signal': '#00e676',
    'support': '#26a69a', 'resistance': '#ef5350',
}


def parse_kline_data(kline_json):
    """解析K线数据为DataFrame，兼容多种格式"""
    if isinstance(kline_json, str):
        data = json.loads(kline_json)
    else:
        data = kline_json

    # 统一格式: fetch_kline.py 输出
    if isinstance(data, dict) and 'data' in data:
        records = data['data']
    # 旧格式: MCP 嵌套 JSON
    elif isinstance(data, dict) and 'result' in data:
        inner = json.loads(data['result']) if isinstance(data['result'], str) else data['result']
        records = inner.get('data', inner)
    elif isinstance(data, list):
        records = data
    else:
        records = data

    df = pd.DataFrame(records)
    
    # 兼容旧字段名
    rename_map = {}
    if 'trade_date' in df.columns:
        rename_map['trade_date'] = 'date'
    if 'open_price' in df.columns:
        rename_map.update({'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close', 'trade_amount': 'volume'})
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # 解析日期
    if df['date'].dtype == 'int64' or (df['date'].dtype == 'object' and df['date'].str.match(r'^\d{8}$').all()):
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 确保数值类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_indicators(df):
    """计算所有技术指标"""
    # 均线
    for period in [5, 10, 20, 60, 120]:
        col = f'ma{period}'
        if col not in df.columns or df[col].isna().all():
            df[col] = df['close'].rolling(window=period).mean()

    # BOLL布林带
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['close'].rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

    # MACD (12, 26, 9)
    if 'dea' not in df.columns or df['dea'].isna().all():
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = 2 * (df['dif'] - df['dea'])
    else:
        # 旧格式兼容
        df['dif'] = df.get('macd', pd.Series(dtype=float)) / 2 + df['dea'] if 'macd' in df.columns else df['dea']
        df['macd_hist'] = df.get('macd', pd.Series(0, index=df.index))

    # KDJ
    low_list = df['low'].rolling(window=9, min_periods=9).min()
    high_list = df['high'].rolling(window=9, min_periods=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # RSI
    delta = df['close'].diff()
    gain6 = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss6 = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['rsi6'] = 100 - (100 / (1 + gain6 / loss6.replace(0, np.nan)))
    gain12 = (delta.where(delta > 0, 0)).rolling(window=12).mean()
    loss12 = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
    df['rsi12'] = 100 - (100 / (1 + gain12 / loss12.replace(0, np.nan)))

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    # 成交量均线
    df['vol_ma5'] = df['volume'].rolling(window=5).mean()
    df['vol_ma10'] = df['volume'].rolling(window=10).mean()

    return df


def find_signals(df):
    """识别买卖点信号"""
    signals = {'buy': [], 'sell': []}

    for i in range(2, len(df)):
        # MACD金叉/死叉
        if 'dif' in df.columns and 'dea' in df.columns:
            dif_prev, dif_curr = df['dif'].iloc[i-1], df['dif'].iloc[i]
            dea_prev, dea_curr = df['dea'].iloc[i-1], df['dea'].iloc[i]
            if pd.notna(dif_prev) and pd.notna(dif_curr):
                if dif_prev <= dea_prev and dif_curr > dea_curr:
                    signals['buy'].append((i, df['low'].iloc[i], 'MACD金叉'))
                elif dif_prev >= dea_prev and dif_curr < dea_curr:
                    signals['sell'].append((i, df['high'].iloc[i], 'MACD死叉'))

        # KDJ
        if 'kdj_k' in df.columns and 'kdj_d' in df.columns:
            k_prev, k_curr = df['kdj_k'].iloc[i-1], df['kdj_k'].iloc[i]
            d_prev, d_curr = df['kdj_d'].iloc[i-1], df['kdj_d'].iloc[i]
            if pd.notna(k_prev) and pd.notna(k_curr):
                if k_prev <= d_prev and k_curr > d_curr and k_curr < 30:
                    signals['buy'].append((i, df['low'].iloc[i], 'KDJ低位金叉'))
                elif k_prev >= d_prev and k_curr < d_curr and k_curr > 70:
                    signals['sell'].append((i, df['high'].iloc[i], 'KDJ高位死叉'))

        # 均线交叉
        if 'ma5' in df.columns and 'ma10' in df.columns:
            ma5_prev, ma5_curr = df['ma5'].iloc[i-1], df['ma5'].iloc[i]
            ma10_prev, ma10_curr = df['ma10'].iloc[i-1], df['ma10'].iloc[i]
            if pd.notna(ma5_prev) and pd.notna(ma10_prev):
                if ma5_prev <= ma10_prev and ma5_curr > ma10_curr:
                    signals['buy'].append((i, df['low'].iloc[i], 'MA5上穿MA10'))
                elif ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                    signals['sell'].append((i, df['high'].iloc[i], 'MA5下穿MA10'))

        # BOLL突破
        if 'boll_lower' in df.columns and 'boll_upper' in df.columns:
            if pd.notna(df['boll_lower'].iloc[i-1]) and pd.notna(df['boll_lower'].iloc[i]):
                if df['close'].iloc[i-1] <= df['boll_lower'].iloc[i-1] and df['close'].iloc[i] > df['boll_lower'].iloc[i]:
                    signals['buy'].append((i, df['low'].iloc[i], 'BOLL下轨反弹'))
                if df['close'].iloc[i-1] >= df['boll_upper'].iloc[i-1] and df['close'].iloc[i] < df['boll_upper'].iloc[i]:
                    signals['sell'].append((i, df['high'].iloc[i], 'BOLL上轨回落'))

    return signals


def draw_candlestick(ax, df):
    """绘制K线图和布林带"""
    width = 0.6
    for i, row in df.iterrows():
        color = COLORS['up'] if row['close'] >= row['open'] else COLORS['down']
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open']) or (row['close'] * 0.001)
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color if row['close'] >= row['open'] else 'white',
                         edgecolor=color, linewidth=0.8)
        ax.add_patch(rect)

    # 均线
    for col, color, label in [('ma5', COLORS['ma5'], 'MA5'), ('ma10', COLORS['ma10'], 'MA10'),
                               ('ma20', COLORS['ma20'], 'MA20'), ('ma60', COLORS['ma60'], 'MA60'),
                               ('ma120', COLORS['ma120'], 'MA120')]:
        if col in df.columns and df[col].notna().any():
            ax.plot(range(len(df)), df[col], color=color, linewidth=1, label=label, alpha=0.8)

    # BOLL
    if 'boll_upper' in df.columns:
        ax.plot(range(len(df)), df['boll_upper'], color=COLORS['boll_upper'], linewidth=1, linestyle='--', label='BOLL上', alpha=0.6)
        ax.plot(range(len(df)), df['boll_mid'], color=COLORS['boll_mid'], linewidth=1, linestyle='-', label='BOLL中', alpha=0.6)
        ax.plot(range(len(df)), df['boll_lower'], color=COLORS['boll_lower'], linewidth=1, linestyle='--', label='BOLL下', alpha=0.6)
        ax.fill_between(range(len(df)), df['boll_upper'], df['boll_lower'], alpha=0.05, color='gray')

    ax.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
    ax.set_xlim(-1, len(df))
    ax.set_ylabel('价格', fontsize=10)
    ax.grid(True, alpha=0.3)


def draw_signals(ax, df, signals):
    """绘制买卖点标记"""
    price_range = df['high'].max() - df['low'].min()
    for idx, price, label in signals.get('buy', []):
        ax.annotate('B', xy=(idx, price), xytext=(idx, price - price_range * 0.05),
                   fontsize=8, color=COLORS['buy_signal'], fontweight='bold', ha='center', va='top')
        ax.scatter(idx, price, marker='^', color=COLORS['buy_signal'], s=50, zorder=5)
    for idx, price, label in signals.get('sell', []):
        ax.annotate('S', xy=(idx, price), xytext=(idx, price + price_range * 0.05),
                   fontsize=8, color=COLORS['sell_signal'], fontweight='bold', ha='center', va='bottom')
        ax.scatter(idx, price, marker='v', color=COLORS['sell_signal'], s=50, zorder=5)


def draw_volume_obv(ax, df):
    """绘制成交量和OBV"""
    colors = [COLORS['volume_up'] if df.iloc[i]['close'] >= df.iloc[i]['open']
              else COLORS['volume_down'] for i in range(len(df))]
    
    # 自适应单位
    max_vol = df['volume'].max()
    if max_vol > 1e8:
        vol_div, vol_unit = 1e8, '亿'
    elif max_vol > 1e4:
        vol_div, vol_unit = 1e4, '万'
    else:
        vol_div, vol_unit = 1, ''

    ax.bar(range(len(df)), df['volume'] / vol_div, color=colors, width=0.6, alpha=0.7)
    if 'vol_ma5' in df.columns:
        ax.plot(range(len(df)), df['vol_ma5'] / vol_div, color=COLORS['ma5'], linewidth=1, label='MA5')
    if 'vol_ma10' in df.columns:
        ax.plot(range(len(df)), df['vol_ma10'] / vol_div, color=COLORS['ma10'], linewidth=1, label='MA10')
    ax.set_ylabel(f'成交量({vol_unit})', fontsize=9)
    ax.set_xlim(-1, len(df))
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # OBV副轴
    ax2 = ax.twinx()
    obv_div = 1e8 if df['obv'].abs().max() > 1e8 else 1e4
    ax2.plot(range(len(df)), df['obv'] / obv_div, color=COLORS['obv'], linewidth=1, label='OBV')
    ax2.set_ylabel(f'OBV({"亿" if obv_div == 1e8 else "万"})', fontsize=9, color=COLORS['obv'])
    ax2.tick_params(axis='y', labelcolor=COLORS['obv'])


def draw_macd(ax, df):
    """绘制MACD指标"""
    hist = df.get('macd_hist', pd.Series(0, index=df.index))
    colors = [COLORS['macd_positive'] if v >= 0 else COLORS['macd_negative'] for v in hist]
    ax.bar(range(len(df)), hist, color=colors, width=0.6, alpha=0.7)
    if 'dif' in df.columns:
        ax.plot(range(len(df)), df['dif'], color=COLORS['dif'], linewidth=1, label='DIF')
    if 'dea' in df.columns:
        ax.plot(range(len(df)), df['dea'], color=COLORS['dea'], linewidth=1, label='DEA')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=7)
    ax.set_ylabel('MACD', fontsize=9)
    ax.set_xlim(-1, len(df))
    ax.grid(True, alpha=0.3)


def draw_kdj(ax, df):
    """绘制KDJ指标"""
    ax.plot(range(len(df)), df['kdj_k'], color=COLORS['k_line'], linewidth=1, label='K')
    ax.plot(range(len(df)), df['kdj_d'], color=COLORS['d_line'], linewidth=1, label='D')
    ax.plot(range(len(df)), df['kdj_j'], color=COLORS['j_line'], linewidth=1, label='J')
    ax.axhline(y=80, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=20, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.fill_between(range(len(df)), 80, 100, alpha=0.1, color='red')
    ax.fill_between(range(len(df)), 0, 20, alpha=0.1, color='green')
    ax.legend(loc='upper left', fontsize=7)
    ax.set_ylabel('KDJ', fontsize=9)
    ax.set_xlim(-1, len(df))
    ax.set_ylim(-10, 110)
    ax.grid(True, alpha=0.3)


def draw_rsi(ax, df):
    """绘制RSI指标"""
    ax.plot(range(len(df)), df['rsi6'], color=COLORS['rsi6'], linewidth=1, label='RSI6')
    ax.plot(range(len(df)), df['rsi12'], color=COLORS['rsi12'], linewidth=1, label='RSI12')
    ax.axhline(y=80, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=20, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
    ax.fill_between(range(len(df)), 80, 100, alpha=0.1, color='red')
    ax.fill_between(range(len(df)), 0, 20, alpha=0.1, color='green')
    ax.legend(loc='upper left', fontsize=7)
    ax.set_ylabel('RSI', fontsize=9)
    ax.set_xlim(-1, len(df))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)


def generate_chart(kline_json, stock_name, stock_code, output_path, show_days=60):
    """生成完整的技术分析图表"""
    df = parse_kline_data(kline_json)
    if len(df) > show_days:
        df = df.tail(show_days).reset_index(drop=True)

    df = calculate_indicators(df)
    signals = find_signals(df)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 1, 0.8, 0.8, 0.3], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    draw_candlestick(ax1, df)
    draw_signals(ax1, df, signals)
    
    last_close = df.iloc[-1]['close']
    change_pct = df.iloc[-1].get('change_pct', 0)
    if isinstance(change_pct, (int, float)) and abs(change_pct) < 1:
        change_pct *= 100  # 小数格式转百分比
    title_color = COLORS['up'] if change_pct >= 0 else COLORS['down']
    ax1.set_title(f'{stock_name}（{stock_code}）技术分析  最新价: {last_close:.2f}  涨跌幅: {change_pct:+.2f}%',
                  fontsize=14, fontweight='bold', color=title_color)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    draw_volume_obv(ax2, df)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    draw_macd(ax3, df)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    draw_kdj(ax4, df)

    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    draw_rsi(ax5, df)

    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.set_xlim(-1, len(df))
    ax6.axis('off')

    # X轴日期
    date_labels = df['date'].dt.strftime('%m/%d')
    tick_positions = list(range(0, len(df), max(1, len(df) // 10)))
    ax5.set_xticks(tick_positions)
    ax5.set_xticklabels([date_labels.iloc[i] for i in tick_positions], rotation=45, fontsize=8)
    for ax in [ax1, ax2, ax3, ax4]:
        plt.setp(ax.get_xticklabels(), visible=False)

    date_range = f"{df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}"
    fig.text(0.99, 0.01, date_range, ha='right', va='bottom', fontsize=9, color='gray')

    signal_text = []
    for idx, price, label in signals.get('buy', [])[-3:]:
        signal_text.append(f"B: {label} ({df['date'].iloc[idx].strftime('%m/%d')})")
    for idx, price, label in signals.get('sell', [])[-3:]:
        signal_text.append(f"S: {label} ({df['date'].iloc[idx].strftime('%m/%d')})")
    if signal_text:
        fig.text(0.01, 0.01, '近期信号: ' + ' | '.join(signal_text[-4:]),
                ha='left', va='bottom', fontsize=8, color='gray')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Chart saved: {output_path}")
    
    # 输出信号摘要
    signal_summary = {
        'buy': [(df['date'].iloc[idx].strftime('%Y-%m-%d'), label) for idx, _, label in signals.get('buy', [])[-5:]],
        'sell': [(df['date'].iloc[idx].strftime('%Y-%m-%d'), label) for idx, _, label in signals.get('sell', [])[-5:]],
        'last_close': float(last_close),
        'change_pct': float(change_pct),
        'ma5': float(df['ma5'].iloc[-1]) if pd.notna(df['ma5'].iloc[-1]) else None,
        'ma10': float(df['ma10'].iloc[-1]) if pd.notna(df['ma10'].iloc[-1]) else None,
        'ma20': float(df['ma20'].iloc[-1]) if pd.notna(df['ma20'].iloc[-1]) else None,
        'ma60': float(df['ma60'].iloc[-1]) if 'ma60' in df.columns and pd.notna(df['ma60'].iloc[-1]) else None,
        'rsi6': float(df['rsi6'].iloc[-1]) if pd.notna(df['rsi6'].iloc[-1]) else None,
        'kdj_k': float(df['kdj_k'].iloc[-1]) if pd.notna(df['kdj_k'].iloc[-1]) else None,
        'boll_upper': float(df['boll_upper'].iloc[-1]) if pd.notna(df['boll_upper'].iloc[-1]) else None,
        'boll_lower': float(df['boll_lower'].iloc[-1]) if pd.notna(df['boll_lower'].iloc[-1]) else None,
    }
    
    summary_path = output_path.rsplit('.', 1)[0] + '_signals.json'
    with open(summary_path, 'w') as f:
        json.dump(signal_summary, f, ensure_ascii=False, indent=2)
    print(f"Signals saved: {summary_path}")
    
    return output_path, signals


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 stock_chart.py <kline_json_file> <stock_name> <stock_code> [output_path]")
        sys.exit(1)

    kline_file = sys.argv[1]
    stock_name = sys.argv[2]
    stock_code = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else f"{stock_code}_chart.png"

    with open(kline_file, 'r') as f:
        kline_json = f.read()

    generate_chart(kline_json, stock_name, stock_code, output_path)
