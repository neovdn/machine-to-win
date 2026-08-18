import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from engine.regime_detector import _detect_swing_sequence

# Simulate 4-cycle data
n = 80
base = 2000.0
opens = np.zeros(n); highs = np.zeros(n)
lows = np.zeros(n); closes = np.zeros(n)

impulse_up = 4.0; correction = 1.5; cycle_len = 4
price = base
for i in range(n):
    phase = i % cycle_len
    cycle_num = i // cycle_len
    if phase == 0:
        step_mult = 1.0 + cycle_num * 0.03
        opens[i] = price; closes[i] = price + impulse_up * step_mult
        highs[i] = closes[i] + 1.0; lows[i] = price - 0.2; price = closes[i]
    elif phase == 1:
        opens[i] = price; closes[i] = price + impulse_up * 0.4
        highs[i] = closes[i] + 0.5; lows[i] = price - 0.2; price = closes[i]
    elif phase == 2:
        opens[i] = price; closes[i] = price - correction * 1.2
        highs[i] = price + 0.3; lows[i] = closes[i] - 0.8; price = closes[i]
    else:
        opens[i] = price; closes[i] = price - correction * 0.3
        highs[i] = price + 0.2; lows[i] = closes[i] - 0.3; price = closes[i]

dates = pd.date_range('2026-01-01', periods=n, freq='15min', tz='UTC')
df = pd.DataFrame({
    'open': opens, 'high': highs, 'low': lows, 'close': closes,
    'tick_volume': 100.0, 'atr_14': 5.0,
    'ema_9': closes-1.0, 'ema_21': closes-4.0,
    'ema_gap_pct': 0.15, 'trend': 'UPTREND', 'volume_ratio': 1.0
}, index=dates)

idx = n-1
swings = _detect_swing_sequence(df, idx, lookback=20, wing=2)
highs_sw = [s for s in swings if s['tipe']=='HIGH']
lows_sw = [s for s in swings if s['tipe']=='LOW']
print(f'idx={idx}, lookback=20, wing=2')
print(f'Swing HIGH: {len(highs_sw)} -> posisi {[s["posisi"] for s in highs_sw]}')
print(f'Swing LOW:  {len(lows_sw)} -> posisi {[s["posisi"] for s in lows_sw]}')

# Count consecutive pairs
hh_pairs = sum(1 for i in range(1, len(highs_sw)) if highs_sw[i]['harga'] > highs_sw[i-1]['harga'])
hl_pairs = sum(1 for i in range(1, len(lows_sw)) if lows_sw[i]['harga'] > lows_sw[i-1]['harga'])
print(f'HH pairs={hh_pairs}, HL pairs={hl_pairs}')

# Also check what highs/lows look like in last 20 candles
print()
print('Last 20 candle highs/lows:')
for j in range(idx-19, idx+1):
    phase = j % cycle_len
    print(f'  j={j} phase={phase} high={highs[j]:.2f} low={lows[j]:.2f}')
