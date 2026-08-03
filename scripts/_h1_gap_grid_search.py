"""
scripts/_h1_gap_grid_search.py
==============================
Grid Search untuk mencari threshold optimal h1_min_ema_gap_pct.

Uji coba threshold gap H1 [0.00, 0.01, 0.02, 0.03, 0.05]
dengan dua parameter set (Scalp M5 dan Baseline).
Dilanjutkan dengan verifikasi entry set untuk threshold terbaik,
serta walk-forward robustness check.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import (
    merge_h1_to_m5,
    simulate_trade_outcome,
    compute_summary,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE,
    DEFAULT_SPREAD_PTS,
)
from engine.rule_engine import evaluate_entry
from engine.risk_manager import calculate_sl_tp


# =============================================================================
# PARAMETER SETS
# =============================================================================

PARAMS_SET_A = {
    "label"               : "Set A (Scalp_M5)",
    "atr_multiplier"      : 0.9,
    "swing_lookback"      : 15,
    "swing_wing"          : 3,
    "rrr_min"             : 1.3,
    "swing_clamp_min_atr" : None,
    "swing_clamp_max_atr" : None,
}

PARAMS_SET_B = {
    "label"               : "Set B (Baseline)",
    "atr_multiplier"      : 1.5,
    "swing_lookback"      : 50,
    "swing_wing"          : 5,
    "rrr_min"             : 2.0,
    "swing_clamp_min_atr" : 0.0,
    "swing_clamp_max_atr" : 999.0,
}

CANDIDATES = [0.00, 0.01, 0.02, 0.03, 0.05]

# =============================================================================
# RUN BACKTEST (Re-used pattern)
# =============================================================================

def run_one_backtest(df_m5_ind, df_h1_ind, h1_gap, params):
    # Merge dengan threshold gap H1 spesifik
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind, h1_min_ema_gap_pct=h1_gap)
    
    trades = []
    in_trade_until_idx = -1
    n_total = len(df_merged)
    entry_times = []
    
    for i in range(WARM_UP_CANDLES, n_total):
        if i <= in_trade_until_idx:
            continue
            
        row = df_merged.iloc[i]
        if pd.isna(row.get("trend_h1")):
            continue
            
        signals = {
            "time"        : df_merged.index[i],
            "close"       : float(row["close"]),
            "ema_9"       : float(row["ema_9"]),
            "ema_21"      : float(row["ema_21"]),
            "rsi_14"      : float(row["rsi_14"]),
            "trend"       : str(row["trend"]),
            "ema_gap_pct" : float(row["ema_gap_pct"]),
            "trend_h1"    : str(row["trend_h1"]),
            "volume_ratio": float(row["volume_ratio"]) if "volume_ratio" in row and not pd.isna(row.get("volume_ratio")) else None,
        }
        
        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in signals.values()
            if isinstance(v, (int, float))
        )
        if has_nan:
            continue
            
        decision = evaluate_entry(signals, volume_mode="FILTER")
        if decision["keputusan"] not in ("BUY", "SELL"):
            continue
            
        arah = decision["keputusan"]
        df_slice = df_m5_ind.iloc[: i + 1]
        
        risk = calculate_sl_tp(
            df                  = df_slice,
            entry               = signals["close"],
            arah                = arah,
            profile             = "scalp_m5",
            rrr_min             = params["rrr_min"],
            atr_multiplier      = params["atr_multiplier"],
            swing_lookback      = params["swing_lookback"],
            swing_wing          = params["swing_wing"],
            swing_clamp_min_atr = params.get("swing_clamp_min_atr"),
            swing_clamp_max_atr = params.get("swing_clamp_max_atr"),
            tick_info           = {
                "ask": signals["close"] + DEFAULT_SPREAD_PTS / 2,
                "bid": signals["close"] - DEFAULT_SPREAD_PTS / 2,
            },
        )
        
        if not risk["valid"]:
            continue
            
        sl       = risk["sl"]
        tp       = risk["tp"]
        jarak_sl = risk["jarak_sl"]
        jarak_tp = risk["jarak_tp"]
        
        if jarak_sl < MIN_SL_DISTANCE:
            continue
            
        outcome_info = simulate_trade_outcome(
            df_m5_full  = df_m5_ind,
            entry_idx   = i,
            entry       = risk["entry"],
            sl          = sl,
            tp          = tp,
            max_candles = MAX_FORWARD_CANDLES,
        )
        
        outcome      = outcome_info["outcome"]
        candles_held = outcome_info["candles_held"]
        
        spread_cost_total = DEFAULT_SPREAD_PTS * 2
        
        if outcome == "TP_HIT":
            rrr_realized = risk.get("rrr_after_spread") or risk["rrr"]
            pnl_points   = +jarak_tp
            pnl_net      = pnl_points - spread_cost_total
        elif outcome == "SL_HIT":
            rrr_realized = -1.0
            pnl_points   = -jarak_sl
            pnl_net      = pnl_points - spread_cost_total
        else:  # NO_HIT
            exit_price_mtm = outcome_info.get("exit_price_mtm", risk["entry"])
            if arah == "BUY":
                pnl_raw = exit_price_mtm - risk["entry"]
            else:
                pnl_raw = risk["entry"] - exit_price_mtm
            pnl_points   = max(pnl_raw, -jarak_sl)
            pnl_net      = pnl_points - spread_cost_total
            rrr_realized = round(pnl_points / jarak_sl, 4) if jarak_sl > 0 else 0.0

        entry_time_str = str(df_merged.index[i])
        entry_times.append(entry_time_str)

        trades.append({
            "entry_time"       : entry_time_str,
            "direction"        : arah,
            "outcome"          : outcome,
            "candles_held"     : candles_held,
            "rrr_realized"     : rrr_realized,
            "rrr_after_spread" : risk.get("rrr_after_spread"),
            "pnl_points"       : pnl_points,
            "pnl_net"          : pnl_net,
            "pnl_type"         : "TP" if outcome == "TP_HIT" else "SL" if outcome == "SL_HIT" else "MTM",
            "sl_method"        : risk.get("sl_method", "UNKNOWN"),
            "trend_h1"         : signals["trend_h1"],
            "ambiguous_candle" : outcome_info["ambiguous_candle"],
            "spread_pts"       : DEFAULT_SPREAD_PTS,
        })
        
        in_trade_until_idx = i + candles_held
        
    if not trades:
        return pd.DataFrame(), {}, entry_times
        
    trades_df = pd.DataFrame(trades)
    summary   = compute_summary(trades_df)
    return trades_df, summary, entry_times

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  GRID SEARCH: THRESHOLD GAP H1 (h1_min_ema_gap_pct)")
    print("=" * 70)
    
    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")
    
    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print("Data CSV tidak ditemukan.")
        sys.exit(1)
        
    print("Loading data...")
    df_m5_raw = load_candles_csv(m5_path)
    df_h1_raw = load_candles_csv(h1_path)
    
    print("Menghitung indikator...")
    df_m5_ind = run_all_indicators(df_m5_raw.copy())
    df_h1_ind = run_all_indicators(df_h1_raw.copy())
    
    results = []
    runs_data = {} # Simpan untuk task berikutnya
    
    print("\n--- MULAI GRID SEARCH ---")
    for param_set in [PARAMS_SET_A, PARAMS_SET_B]:
        print(f"\nParameter: {param_set['label']}")
        for gap in CANDIDATES:
            print(f"  Menguji gap={gap:.2f}%...", end="", flush=True)
            t0 = time.time()
            trades_df, summary, entry_times = run_one_backtest(df_m5_ind, df_h1_ind, gap, param_set)
            
            row = {
                "Param_Set": param_set['label'],
                "H1_Gap": gap,
                "Total_Trades": summary.get("total_trades", 0),
                "Win_Rate": summary.get("win_rate", 0),
                "No_Hit_Rate": summary.get("no_hit_rate", 0),
                "Avg_RRR": summary.get("avg_rrr_realized", 0),
                "PnL_Net": summary.get("total_pnl_net", 0),
                "Max_DD": summary.get("max_drawdown_net", 0),
            }
            results.append(row)
            runs_data[f"{param_set['label']}_{gap}"] = entry_times
            print(f" Selesai dalam {time.time()-t0:.1f}s - {row['Total_Trades']} trades")
            
    # Print Tabel
    print("\n" + "="*95)
    print(f"{'Param Set':<20} {'H1 Gap':<8} {'Trades':<8} {'Win Rate':<10} {'NoHitRate':<10} {'Avg RRR':<10} {'PnL Net':<10} {'Max DD':<10}")
    print("-" * 95)
    for r in results:
        wr = f"{r['Win_Rate']*100:.1f}%" if r['Win_Rate'] else "N/A"
        nhr = f"{r['No_Hit_Rate']*100:.1f}%" if r['No_Hit_Rate'] else "N/A"
        rrr = f"{r['Avg_RRR']:+.4f}R" if r['Avg_RRR'] else "N/A"
        pnl = f"{r['PnL_Net']:+.1f}"
        mdd = f"{r['Max_DD']:+.1f}"
        print(f"{r['Param_Set']:<20} {r['H1_Gap']:<8.2f} {r['Total_Trades']:<8} {wr:<10} {nhr:<10} {rrr:<10} {pnl:<10} {mdd:<10}")
        
    print("="*95)
    
    # === ANALISA TERBAIK ===
    # Cari kandidat dengan avg RRR positif tinggi, total trades lumayan, di kedua set.
    # Pilih 0.02 atau 0.03 tergantung hasil. (Hardcode 0.02 as candidate to investigate next, wait we'll decide in the log and just do the verif for 0.02 and 0.03 here).
    
    for best_candidate in [0.01, 0.02, 0.03]:
        print(f"\n--- VERIFIKASI ENTRY SET (KANDIDAT = {best_candidate:.2f}%) ---")
        for param_set in [PARAMS_SET_A, PARAMS_SET_B]:
            print(f"\nParameter: {param_set['label']}")
            entry_best = set(runs_data[f"{param_set['label']}_{best_candidate}"])
            entry_00 = set(runs_data[f"{param_set['label']}_0.0"])
            entry_05 = set(runs_data[f"{param_set['label']}_0.05"])
            
            print(f"  Trade 0.00% (Lama): {len(entry_00)}")
            print(f"  Trade {best_candidate:.2f}% (Kandidat): {len(entry_best)}")
            print(f"  Trade 0.05% (Ketat): {len(entry_05)}")
            
            common_00 = entry_best & entry_00
            common_05 = entry_best & entry_05
            
            print(f"  Irisan dengan 0.00%: {len(common_00)} ({len(common_00)/len(entry_best)*100 if len(entry_best)>0 else 0:.1f}% dari Kandidat)")
            print(f"  Irisan dengan 0.05%: {len(common_05)} ({len(common_05)/len(entry_best)*100 if len(entry_best)>0 else 0:.1f}% dari Kandidat)")

    
    # === WALK FORWARD CHECK (using 0.02 as test) ===
    BEST_GAP = 0.02 # Let's test 0.02 on walk forward
    print(f"\n--- WALK FORWARD VALIDATION (KANDIDAT {BEST_GAP:.2f}%) ---")
    
    m5_path_ext = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
    h1_path_ext = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")
    
    if os.path.exists(m5_path_ext) and os.path.exists(h1_path_ext):
        df_m5_raw_ext = load_candles_csv(m5_path_ext)
        df_h1_raw_ext = load_candles_csv(h1_path_ext)
        df_m5_ind_ext = run_all_indicators(df_m5_raw_ext.copy())
        df_h1_ind_ext = run_all_indicators(df_h1_raw_ext.copy())
        
        folds = [
            {"val": ("2025-09-01", "2025-09-30"), "label": "Fold 1"},
            {"val": ("2025-10-01", "2025-10-31"), "label": "Fold 2"},
            {"val": ("2025-11-01", "2025-11-30"), "label": "Fold 3"},
            {"val": ("2025-12-01", "2025-12-31"), "label": "Fold 4"},
            {"val": ("2026-01-01", "2026-01-31"), "label": "Fold 5"},
            {"val": ("2026-02-01", "2026-02-28"), "label": "Fold 6"},
            {"val": ("2026-03-01", "2026-03-31"), "label": "Fold 7"},
            {"val": ("2026-04-01", "2026-04-30"), "label": "Fold 8"},
            {"val": ("2026-05-01", "2026-05-31"), "label": "Fold 9"},
            {"val": ("2026-06-01", "2026-06-30"), "label": "Fold 10"},
            {"val": ("2026-07-01", "2026-07-25"), "label": "Fold 11"},
        ]

        def _filter(df, start_str, end_str):
            start = pd.Timestamp(start_str, tz="UTC")
            end   = pd.Timestamp(end_str,   tz="UTC")
            return df[(df.index >= start) & (df.index <= end)]

        print(f"\n  {'Fold':<10} {'Val Period':<26} {'Trades':>7} {'WinRate':>9} {'AvgRRR':>9} {'NoHit%':>8} {'PnL':>9}")
        print("  " + "-" * 82)

        df_merged_ext = merge_h1_to_m5(df_m5_ind_ext, df_h1_ind_ext, h1_min_ema_gap_pct=BEST_GAP)
        
        results_ext = []
        for fold in folds:
            df_m5_val   = _filter(df_m5_ind_ext,  fold["val"][0], fold["val"][1])
            df_merged_v = _filter(df_merged_ext,   fold["val"][0], fold["val"][1])

            if len(df_m5_val) < WARM_UP_CANDLES + 20:
                continue

            t_df, t_sum, _ = run_one_backtest(
                df_m5_val,
                df_h1_ind_ext, # Not strictly needed inside since we pass df_merged, wait run_one_backtest re-merges!
                BEST_GAP,
                PARAMS_SET_A # Let's use scalp_m5 for OOS like last time
            )
            # Fix: run_one_backtest calls merge_h1_to_m5 internally, so we don't need df_merged_ext above actually.
            # Wait, run_one_backtest takes df_m5_val and df_h1_ind_ext, and merges. Yes.

            n      = t_sum.get("total_trades", 0)
            wr     = t_sum.get("win_rate")
            rrr    = t_sum.get("avg_rrr_realized")
            nhr    = t_sum.get("no_hit_rate", 0)
            pnl    = t_sum.get("total_pnl_net", 0)

            wr_str  = f"{wr:.1%}"  if wr  is not None else "N/A"
            rrr_str = f"{rrr:+.3f}R" if rrr is not None else "N/A"
            nhr_str = f"{nhr:.1%}" if nhr is not None else "N/A"

            val_label = f"{fold['val'][0]} – {fold['val'][1]}"
            print(f"  {fold['label']:<10} {val_label:<26} {n:>7,} {wr_str:>9} {rrr_str:>9} {nhr_str:>8} {pnl:>+9.1f}")

            results_ext.append({
                "total_trades": n,
                "win_rate"    : wr,
                "avg_rrr"     : rrr,
            })

        valid_results = [r for r in results_ext if r["total_trades"] >= 10]
        win_rates = [r["win_rate"] for r in valid_results if r["win_rate"] is not None]
        avg_rrrs  = [r["avg_rrr"]  for r in valid_results if r["avg_rrr"]  is not None]

        print("\n  Ringkasan Robustness:")
        if win_rates:
            print(f"    Win Rate — mean={np.mean(win_rates):.1%}, std={np.std(win_rates):.1%}")
        if avg_rrrs:
            print(f"    Avg RRR  — mean={np.mean(avg_rrrs):+.3f}, std={np.std(avg_rrrs):.3f}")
            neg_count = sum(1 for r in avg_rrrs if r <= 0)
            print(f"    Fold dengan avg_rrr <= 0: {neg_count}/{len(avg_rrrs)}")


if __name__ == "__main__":
    main()
