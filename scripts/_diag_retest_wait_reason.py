"""Investigate: mengapa evaluate_entry WAIT padahal trigger_source=BOTH?"""
import sys
sys.path.insert(0, r'C:\Users\mercy\AppData\Local\machine-to-win')
from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators, get_latest_signals
from engine.backtester import merge_h1_to_m5, WARM_UP_CANDLES
from engine.rule_engine import evaluate_entry

df_m5 = load_candles_csv(r'C:\Users\mercy\AppData\Local\machine-to-win\data\historical\XAUUSD_M5_2026-01-01_2026-07-25.csv')
df_h1 = load_candles_csv(r'C:\Users\mercy\AppData\Local\machine-to-win\data\historical\XAUUSD_H1_2026-01-01_2026-07-25.csv')
df_m5_ind = run_all_indicators(df_m5.copy())
df_h1_ind = run_all_indicators(df_h1.copy())
df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind, h1_min_ema_gap_pct=0.02)

for idx in [206, 207, 234]:
    signals = get_latest_signals(df_m5_ind.iloc[:idx+1])
    row_merged = df_merged.iloc[idx]
    signals["trend_h1"] = row_merged.get("trend_h1", "SIDEWAYS")
    decision = evaluate_entry(signals, df=df_m5_ind, idx=idx, enable_retest_trigger=True)
    keputusan = decision["keputusan"]
    ts = decision["trigger_source"]
    rsi_cond = decision["kondisi_detail"].get("rsi_filter", {})
    vol_cond = decision["kondisi_detail"].get("volume_filter", {})
    rsi_val = signals.get("rsi_14", 0)
    print(f"idx={idx}  keputusan={keputusan}  trigger_source={ts}")
    print(f"  rsi_14={rsi_val:.2f}  rsi_filter.terpenuhi={rsi_cond.get('terpenuhi')}  keterangan={rsi_cond.get('keterangan','')[:80]}")
    print(f"  volume_filter.terpenuhi={vol_cond.get('terpenuhi')}  keterangan={vol_cond.get('keterangan','')[:80]}")
    print(f"  alasan_wait: {decision['alasan_wait']}")
    print()
