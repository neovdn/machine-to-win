"""Spot check: mengapa evaluate_entry() retest mode menghasilkan WAIT padahal _check_retest_trigger terpenuhi?"""
import os, sys
sys.path.insert(0, r'C:\Users\mercy\AppData\Local\machine-to-win')

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators, get_latest_signals
from engine.backtester import merge_h1_to_m5, WARM_UP_CANDLES
from engine.rule_engine import _check_retest_trigger, evaluate_entry

df_m5 = load_candles_csv(r'C:\Users\mercy\AppData\Local\machine-to-win\data\historical\XAUUSD_M5_2026-01-01_2026-07-25.csv')
df_h1 = load_candles_csv(r'C:\Users\mercy\AppData\Local\machine-to-win\data\historical\XAUUSD_H1_2026-01-01_2026-07-25.csv')
df_m5_ind = run_all_indicators(df_m5.copy())
df_h1_ind = run_all_indicators(df_h1.copy())
df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind, h1_min_ema_gap_pct=0.02)

# Cek 20 titik pertama di mana _check_retest_trigger terpenuhi
print("Mencari titik di mana retest terpenuhi...")
found = 0
for idx in range(WARM_UP_CANDLES, min(2000, len(df_m5_ind))):
    result = _check_retest_trigger(df_m5_ind, idx=idx)
    if result["terpenuhi"]:
        # Cek apa yang terjadi di evaluate_entry
        signals = get_latest_signals(df_m5_ind.iloc[:idx+1])
        # Ambil trend_h1 dari df_merged
        if idx < len(df_merged):
            row_merged = df_merged.iloc[idx]
            signals["trend_h1"] = row_merged.get("trend_h1", "SIDEWAYS")
            signals["ema_9_h1"] = row_merged.get("ema_9_h1", None)
            signals["ema_21_h1"] = row_merged.get("ema_21_h1", None)

        decision = evaluate_entry(signals, df=df_m5_ind, idx=idx, enable_retest_trigger=True)
        retest_cond = decision["kondisi_detail"].get("retest_trigger", {})
        bias_cond   = decision["kondisi_detail"].get("bias_h1", {})

        print(f"\n--- idx={idx} ---")
        print(f"  _check_retest_trigger: terpenuhi={result['terpenuhi']}, arah={result['arah']}")
        print(f"  evaluate_entry: keputusan={decision['keputusan']}, trigger_source={decision['trigger_source']}")
        print(f"  bias_h1: terpenuhi={bias_cond.get('terpenuhi')}, arah={bias_cond.get('arah')}")
        print(f"  retest_trigger via eval: terpenuhi={retest_cond.get('terpenuhi')}, arah={retest_cond.get('arah')}")
        print(f"  arah_kandidat dari eval: {decision['kondisi_detail']['bias_h1'].get('arah_kandidat', '?')}")
        print(f"  signals trend={signals.get('trend')}, trend_h1={signals.get('trend_h1')}")
        print(f"  keterangan bias_h1: {bias_cond.get('keterangan', '')[:120]}")
        print(f"  alasan_wait: {decision['alasan_wait'][:3]}")
        found += 1
        if found >= 10:
            break

if found == 0:
    print("Tidak ada titik retest terpenuhi dalam candle 100-2000!")
    # Cek candle yang lebih jauh
    n_found = 0
    for idx in range(WARM_UP_CANDLES, len(df_m5_ind), 3):
        r = _check_retest_trigger(df_m5_ind, idx=idx)
        if r["terpenuhi"]:
            n_found += 1
            if n_found <= 3:
                print(f"  Ditemukan di idx={idx}: {r['keterangan'][:100]}")
    print(f"Total ditemukan di full dataset (step=3): {n_found}")
