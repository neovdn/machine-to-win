"""Run Fase 1 parameters on validation period and compare vs calibration."""
import pandas as pd, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import merge_h1_to_m5, WARM_UP_CANDLES, MAX_FORWARD_CANDLES, DEFAULT_SPREAD_PTS
from scripts.run_param_sweep import run_fast_backtest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m5_path = os.path.join(ROOT_DIR, 'data', 'historical', 'XAUUSD_M5_2026-01-01_2026-07-25.csv')
h1_path = os.path.join(ROOT_DIR, 'data', 'historical', 'XAUUSD_H1_2026-01-01_2026-07-25.csv')

df_m5 = load_candles_csv(m5_path)
df_h1 = load_candles_csv(h1_path)

def filter_period(df, date_from, date_to):
    ts_from = pd.Timestamp(date_from, tz='UTC')
    ts_to   = pd.Timestamp(date_to, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= ts_from) & (df.index <= ts_to)].copy()

# Fase 1 params
ATR, LOOK, WING, RRR = 0.9, 15, 3, 1.3

print("=== EVALUASI PARAMETER FASE 1 DI PERIODE VALIDASI ===")
print(f"  atr={ATR}, lookback={LOOK}, wing={WING}, rrr_min={RRR}")

# Validation period
df_m5_val = filter_period(df_m5, '2026-05-01', '2026-07-25')
df_h1_val = filter_period(df_h1, '2026-05-01', '2026-07-25')
df_m5_val_ind = run_all_indicators(df_m5_val.copy())
df_h1_val_ind = run_all_indicators(df_h1_val.copy())
df_val_merged = merge_h1_to_m5(df_m5_val_ind, df_h1_val_ind)

t0 = time.time()
val_trades, val_summary = run_fast_backtest(
    df_m5_ind=df_m5_val_ind, df_merged=df_val_merged,
    atr_mult=ATR, lookback=LOOK, wing=WING, rrr_min=RRR,
    warm_up=WARM_UP_CANDLES
)
print(f"Done in {time.time()-t0:.1f}s")
print()
print(f"VALIDASI  - Total trades:    {val_summary['total_trades']}")
print(f"VALIDASI  - win_rate:        {(val_summary['win_rate'] or 0)*100:.2f}%")
print(f"VALIDASI  - avg_rrr_real:    {val_summary['avg_rrr_realized']:+.4f}")
print(f"VALIDASI  - no_hit_rate:     {(val_summary['no_hit_rate'] or 0)*100:.2f}%")
print(f"VALIDASI  - total_pnl_net:   {val_summary['total_pnl_net']:+.2f}")
print(f"VALIDASI  - max_drawdown_net:{val_summary['max_drawdown_net']:+.2f}")
