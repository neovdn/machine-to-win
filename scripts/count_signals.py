import os, sys, pandas as pd
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.rule_engine import evaluate_entry
from engine.backtester import merge_h1_to_m5

m5 = load_candles_csv('data/historical/XAUUSD_M5_2026-01-01_2026-07-25.csv')
h1 = load_candles_csv('data/historical/XAUUSD_H1_2026-01-01_2026-07-25.csv')
m5_ind = run_all_indicators(m5)
h1_ind = run_all_indicators(h1)
df = merge_h1_to_m5(m5_ind, h1_ind)

signals = 0
for i in range(100, len(df)):
    if pd.isna(df['trend_h1'].iloc[i]): continue
    r = df.iloc[i]
    dec = evaluate_entry({
        'time': str(df.index[i]),
        'close': float(r['close']),
        'ema_9': float(r['ema_9']),
        'ema_21': float(r['ema_21']),
        'rsi_14': float(r['rsi_14']),
        'trend': str(r['trend']),
        'ema_gap_pct': float(r['ema_gap_pct']),
        'trend_h1': str(r['trend_h1'])
    })
    if dec['keputusan'] in ['BUY', 'SELL']:
        signals += 1

print(f"Total RAW Signals Tanpa Position Blocking: {signals}")
