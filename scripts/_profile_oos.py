"""Diagnostic script to profile OOS backtest speed."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import merge_h1_to_m5, WARM_UP_CANDLES
from scripts.run_param_sweep import run_fast_backtest
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m5_path = os.path.join(ROOT_DIR, 'data', 'historical', 'XAUUSD_M5_2026-01-01_2026-07-25.csv')
h1_path = os.path.join(ROOT_DIR, 'data', 'historical', 'XAUUSD_H1_2026-01-01_2026-07-25.csv')

print('Loading...')
df_m5 = load_candles_csv(m5_path)
df_h1 = load_candles_csv(h1_path)

# Filter calibration period
ts_from = pd.Timestamp('2026-01-01', tz='UTC')
ts_to   = pd.Timestamp('2026-04-30 23:59:59', tz='UTC')
df_m5_c = df_m5.loc[(df_m5.index >= ts_from) & (df_m5.index <= ts_to)].copy()
df_h1_c = df_h1.loc[(df_h1.index >= ts_from) & (df_h1.index <= ts_to)].copy()
print(f'Calib M5: {len(df_m5_c)} candles, H1: {len(df_h1_c)} candles')

print('Computing indicators...')
t0 = time.time()
df_m5_ind = run_all_indicators(df_m5_c.copy())
df_h1_ind = run_all_indicators(df_h1_c.copy())
df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)
print(f'Indicators done in {time.time()-t0:.2f}s')

print('Running ONE backtest combo...')
t1 = time.time()
trades, summary = run_fast_backtest(
    df_m5_ind=df_m5_ind, df_merged=df_merged,
    atr_mult=0.9, lookback=15, wing=3, rrr_min=1.3,
    warm_up=WARM_UP_CANDLES
)
elapsed = time.time() - t1
print(f'ONE combo done in {elapsed:.2f}s, trades={summary["total_trades"]}')
print(f'Estimated for 192 combos: {elapsed*192:.1f}s = {elapsed*192/60:.1f} min')
