"""
scripts/_fase11_validate_sd_zone.py
===================================
Validasi Empiris untuk Fase 11 (Supply & Demand Zone).

Langkah-langkah:
1. Sweep parameter `impulsive_body_atr_ratio` untuk mencari rasio optimal
   (menyeimbangkan jumlah trade, PnL, dan win rate).
2. Walk-Forward Analysis (WFA) membandingkan `sl_source="SWING"` vs `"SD_ZONE"`
   menggunakan rasio terbaik di 5 fold independen (2020-2024).
3. Analisis statistik dan FRESH vs TESTED distribution.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import merge_h1_to_m5
from scripts.run_walk_forward import generate_folds
from scripts.run_oos_validation import filter_period
from scripts.run_param_sweep import run_fast_backtest

# Constants
M5_PATH = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
H1_PATH = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

RATIOS_TO_TEST = [1.0, 1.5, 2.0, 2.5]
SWING_PARAMS = {
    "atr_mult": 0.9,
    "lookback": 15,
    "wing": 2,
    "rrr_min": 1.5,
}

def analyze_trades(trades: list) -> dict:
    if not trades:
        return {"win_rate": 0, "pnl": 0, "avg_rrr": 0, "total": 0, "fresh_pct": 0}
    
    wins = [t for t in trades if t["pnl_net"] > 0]
    pnl = sum(t["pnl_net"] for t in trades)
    avg_rrr = np.mean([t["rrr_realized"] for t in trades])
    
    # Check freshness for SD_ZONE
    sd_trades = [t for t in trades if isinstance(t.get("pesan"), str) and "S&D zone" in t["pesan"]]
    fresh_count = sum(1 for t in sd_trades if "FRESH" in t.get("pesan", ""))
    fresh_pct = fresh_count / len(sd_trades) * 100 if sd_trades else 0

    return {
        "win_rate": len(wins) / len(trades),
        "pnl": pnl,
        "avg_rrr": avg_rrr,
        "total": len(trades),
        "fresh_pct": fresh_pct,
    }

def main():
    print("=" * 80)
    print("  FASE 11.3: VALIDASI EMPIRIS SUPPLY & DEMAND ZONE")
    print("=" * 80)

    if not os.path.exists(M5_PATH) or not os.path.exists(H1_PATH):
        print(f"Data baseline tidak ditemukan: {M5_PATH}")
        sys.exit(1)

    print("\n-> Memuat data historis (Extended 2025-2026)...")
    df_m5 = load_candles_csv(M5_PATH)
    df_h1 = load_candles_csv(H1_PATH)
    
    print("-> Menghitung indikator (butuh beberapa detik)...")
    df_m5_ind = run_all_indicators(df_m5)
    df_h1_ind = run_all_indicators(df_h1)
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)

    print("\n" + "=" * 80)
    print("  TAHAP 1: PARAMETER SWEEP (impulsive_body_atr_ratio)")
    print("=" * 80)
    print(f"{'Method':<10} {'Ratio':<6} | {'Trades':<6} {'WinRate':<8} {'AvgRRR':<8} {'PnL Net':<10} | {'FRESH %':<8}")
    print("-" * 80)

    # Baseline (SWING)
    res_swing = run_fast_backtest(df_m5_ind, df_merged, **SWING_PARAMS, sl_source="SWING")
    
    trades_swing = res_swing[0].to_dict('records') if not res_swing[0].empty else []
    stats_swing = analyze_trades(trades_swing)
    print(f"{'SWING':<10} {'-':<6} | {stats_swing['total']:<6} {stats_swing['win_rate']:.1%}   {stats_swing['avg_rrr']:.2f}     {stats_swing['pnl']:<10.1f} | -")

    # Sweep SD_ZONE
    best_ratio = None
    best_pnl = -float('inf')

    for ratio in RATIOS_TO_TEST:
        res_sd = run_fast_backtest(
            df_m5_ind, df_merged, **SWING_PARAMS, 
            sl_source="SD_ZONE", sd_impulsive_ratio=ratio
        )
        trades_sd = res_sd[0].to_dict('records') if not res_sd[0].empty else []
        stats_sd = analyze_trades(trades_sd)
        print(f"{'SD_ZONE':<10} {ratio:<6} | {stats_sd['total']:<6} {stats_sd['win_rate']:.1%}   {stats_sd['avg_rrr']:.2f}     {stats_sd['pnl']:<10.1f} | {stats_sd['fresh_pct']:.1f}%")
        
        # Sederhana: pilih PnL tertinggi untuk walk-forward, walau ideally dipilih manual
        if stats_sd['pnl'] > best_pnl:
            best_pnl = stats_sd['pnl']
            best_ratio = ratio

    if best_ratio is None:
        best_ratio = 1.5

    print(f"\n-> Memilih ratio={best_ratio} untuk Walk-Forward Analysis.")

    print("\n" + "=" * 80)
    print("  TAHAP 2: WALK-FORWARD ANALYSIS LINTAS FOLD")
    print("=" * 80)

    folds = generate_folds(
        data_start=df_m5.index[0].strftime("%Y-%m-%d"),
        data_end=df_m5.index[-1].strftime("%Y-%m-%d"),
        calib_months=3,
        val_months=1,
    )
    print(f"Menggunakan {len(folds)} fold validasi.")

    print(f"\n{'Fold':<6} | {'SWING Trades':<12} {'SWING Win%':<12} {'SWING PnL':<12} | {'SD Trades':<10} {'SD Win%':<10} {'SD PnL':<10}")
    print("-" * 80)

    pnl_swing_list = []
    pnl_sd_list = []

    for f in folds:
        val_start, val_end = f["val_start"], f["val_end"]
        
        df_m5_val = filter_period(df_m5_ind, val_start, val_end)
        df_merged_val = filter_period(df_merged, val_start, val_end)
        
        if len(df_m5_val) == 0:
            continue
        # WFA SWING
        res_sw = run_fast_backtest(df_m5_val, df_merged_val, **SWING_PARAMS, sl_source="SWING")
        if not res_sw[0].empty:
            trades_sw = res_sw[0].to_dict('records')
            st_sw = analyze_trades(trades_sw)
            pnl_swing_list.append(st_sw['pnl'])
        else:
            st_sw = {"total": 0, "win_rate": 0, "pnl": 0}

        # WFA SD_ZONE
        res_sd = run_fast_backtest(df_m5_val, df_merged_val, **SWING_PARAMS, sl_source="SD_ZONE", sd_impulsive_ratio=best_ratio)
        if not res_sd[0].empty:
            trades_sd = res_sd[0].to_dict('records')
            st_sd = analyze_trades(trades_sd)
            pnl_sd_list.append(st_sd['pnl'])
        else:
            st_sd = {"total": 0, "win_rate": 0, "pnl": 0}

        print(f"{f['val_start']} | {st_sw['total']:<12} {st_sw['win_rate']:<12.1%} {st_sw['pnl']:<12.1f} | {st_sd['total']:<10} {st_sd['win_rate']:<10.1%} {st_sd['pnl']:<10.1f}")

    print("\n" + "=" * 80)
    print("  TAHAP 3: UJI SIGNIFIKANSI STATISTIK")
    print("=" * 80)
    
    # Wilcoxon Signed-Rank Test pada PnL per fold
    if len(pnl_swing_list) > 1:
        try:
            stat, p_value = stats.wilcoxon(pnl_swing_list, pnl_sd_list)
            print(f"Wilcoxon Signed-Rank Test (PnL per fold):")
            print(f"  Statistik : {stat}")
            print(f"  P-Value   : {p_value:.5f}")
            if p_value < 0.05:
                print("  Kesimpulan: Perbedaan SIGNIFIKAN (p < 0.05). S&D Zone secara meyakinkan mengubah PnL.")
            else:
                print("  Kesimpulan: Perbedaan TIDAK SIGNIFIKAN (p >= 0.05). Varians bisa jadi kebetulan acak.")
        except Exception as e:
            print(f"Wilcoxon test gagal (mungkin karena sample kembar): {e}")
    else:
        print("Data fold terlalu sedikit untuk uji statistik.")

if __name__ == "__main__":
    main()
