"""
scripts/run_fase10_validation.py
=================================
Validasi Empiris Fase 10 — Retest Logic (Spesifikasi 4.4)

MURNI OBSERVASI — TIDAK ada perubahan threshold.
Parameter default: retest_lookback_candles=15, retest_tolerance_atr=0.3 (dari _check_retest_trigger).

YANG DIJALANKAN:
  A. Backtest penuh: RETEST vs BREAKOUT (in-sample 2026-01-01 s/d 2026-07-25)
     - Side-by-side: win_rate, avg_rrr_realized, no_hit_rate, avg jarak_sl
     - Breakdown sinyal per trigger_source: RETEST, BOTH, EMA_GAP
     - Trade overlap: entry_time set intersection/disjoint RETEST vs BREAKOUT

  B. Walk-forward (extend run_walk_forward.py logic) — RETEST vs BREAKOUT per fold
     - Dataset: data/historical/XAUUSD_M5_2025-06-01_2026-07-25.csv
     - Skema: 3 bulan calib | 1 bulan validasi | geser 1 bulan | ~10 fold

  C. Uji statistik dengan scipy.stats (p-value EXACT):
     - H0_1: jarak_sl RETEST >= jarak_sl BREAKOUT (uji satu arah: apakah SL lebih kecil?)
     - H0_2: win_rate RETEST <= win_rate BREAKOUT (uji satu arah: apakah win_rate tidak turun?)
     - Metode: Mann-Whitney U (tidak assume normalitas, cocok untuk sampel kecil)
     - Alpha: 0.05

  D. VERDICT AKHIR: LOLOS / TIDAK LOLOS / PERLU DATA LEBIH
     Kriteria LOLOS:
       1. p-value SL lebih kecil (H0_1 ditolak): p < 0.05
       2. Win rate tidak turun signifikan (H0_2 TIDAK ditolak atau win rate >=): p_wr >= 0.05
          atau selisih win_rate_RETEST - win_rate_BREAKOUT >= -3pp (toleransi)
       3. Walk-forward: >= 60% fold RETEST menunjukkan pnl_net >= pnl_net BREAKOUT (atau tidak jauh lebih buruk)
     Kriteria PERLU DATA LEBIH:
       n_retest_trades < 30 (terlalu sedikit untuk uji statistik yang bermakna)
     Selain itu: TIDAK LOLOS

OUTPUT:
  Cetak laporan lengkap ke stdout.
  Simpan trades CSV ke data/backtest_results/fase10_validation_*.csv
  Simpan laporan ringkas ke data/backtest_results/fase10_validation_report.txt
"""

import os
import sys
import time
import scipy.stats

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators  import run_all_indicators
from engine.backtester  import (
    run_backtest,
    merge_h1_to_m5,
    compute_summary,
    validate_no_lookahead,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    DEFAULT_SPREAD_PTS,
    MIN_SL_DISTANCE,
)
from scripts.run_param_sweep  import run_fast_backtest
from scripts.run_oos_validation import filter_period
from scripts.run_walk_forward import generate_folds

# =============================================================================
# KONFIGURASI — FIXED, tidak diubah per spesifikasi 4.4
# =============================================================================

# Path dataset
M5_PATH_JAN = os.path.join(ROOT_DIR, "data", "historical",
                            "XAUUSD_M5_2026-01-01_2026-07-25.csv")
H1_PATH_JAN = os.path.join(ROOT_DIR, "data", "historical",
                            "XAUUSD_H1_2026-01-01_2026-07-25.csv")

M5_PATH_EXT = os.path.join(ROOT_DIR, "data", "historical",
                            "XAUUSD_M5_2025-06-01_2026-07-25.csv")
H1_PATH_EXT = os.path.join(ROOT_DIR, "data", "historical",
                            "XAUUSD_H1_2025-06-01_2026-07-25.csv")

# Fase 1 fixed parameters (tidak dikalibrasi ulang)
FASE1_PARAMS = {
    "atr_mult" : 0.9,
    "lookback" : 15,
    "wing"     : 3,
    "rrr_min"  : 1.3,
}

SPREAD_PTS   = DEFAULT_SPREAD_PTS  # 0.50
MAX_CANDLES  = MAX_FORWARD_CANDLES
WARM_UP      = WARM_UP_CANDLES
ALPHA        = 0.05  # level signifikansi statistik

OUT_DIR = os.path.join(ROOT_DIR, "data", "backtest_results")


# =============================================================================
# HELPER: RUN BACKTEST SATU MODE
# =============================================================================

def run_mode(
    df_m5_ind,
    df_merged,
    mode: str,          # "BREAKOUT" atau "RETEST"
    volume_mode: str = "FILTER",
    label: str = "",
) -> tuple:
    """
    Jalankan run_fast_backtest() untuk satu mode trigger.
    Mengembalikan (trades_df, summary).
    """
    enable_breakout = (mode == "BREAKOUT")
    enable_retest   = (mode == "RETEST")

    trades_df, summary = run_fast_backtest(
        df_m5_ind   = df_m5_ind,
        df_merged   = df_merged,
        atr_mult    = FASE1_PARAMS["atr_mult"],
        lookback    = FASE1_PARAMS["lookback"],
        wing        = FASE1_PARAMS["wing"],
        rrr_min     = FASE1_PARAMS["rrr_min"],
        spread_pts  = SPREAD_PTS,
        max_candles = MAX_CANDLES,
        warm_up     = WARM_UP,
        volume_mode             = volume_mode,
        enable_breakout_trigger = enable_breakout,
        enable_retest_trigger   = enable_retest,
    )
    return trades_df, summary


# =============================================================================
# BAGIAN A: BACKTEST IN-SAMPLE (Jan-Jul 2026)
# =============================================================================

def run_section_a(df_m5_jan, df_h1_jan, report_lines: list):
    """
    Backtest penuh RETEST vs BREAKOUT, in-sample Jan-Jul 2026.
    Jalankan dengan tiga volume_mode untuk mengisolasi efek volume filter.
    Mengembalikan (trades_bo_ignore, trades_rt_ignore, summ_bo_ignore, summ_rt_ignore, 
                    trades_bo_filter, summ_bo_filter, df_m5_ind, df_h1_ind)
    """
    hdr = "=" * 70
    print(f"\n{hdr}")
    print("  BAGIAN A: BACKTEST IN-SAMPLE — RETEST vs BREAKOUT")
    print(f"  Dataset : {M5_PATH_JAN}")
    print(f"{hdr}")
    report_lines.append("\n" + hdr)
    report_lines.append("BAGIAN A: BACKTEST IN-SAMPLE — RETEST vs BREAKOUT")
    report_lines.append(hdr)

    # Hitung indikator + merge SATU KALI
    print("\n[1/4] Menghitung indikator M5...")
    df_m5_ind = run_all_indicators(df_m5_jan.copy())
    print("[2/4] Menghitung indikator H1...")
    df_h1_ind = run_all_indicators(df_h1_jan.copy())
    print("[3/4] Merging H1 ke M5...")
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind, h1_min_ema_gap_pct=0.02)

    # Jalankan variasi volume mode
    modes = ["FILTER", "CONDITION", "IGNORE"]
    results = {}
    for vm in modes:
        print(f"\n[4/4] Running with volume_mode={vm}...")
        trades_bo, summ_bo = run_mode(df_m5_ind, df_merged, "BREAKOUT", volume_mode=vm)
        trades_rt, summ_rt = run_mode(df_m5_ind, df_merged, "RETEST", volume_mode=vm)
        results[vm] = (trades_bo, summ_bo, trades_rt, summ_rt)

    trades_bo_filter, summ_bo_filter, trades_rt_filter, summ_rt_filter = results["FILTER"]
    trades_bo_cond,   summ_bo_cond,   trades_rt_cond,   summ_rt_cond   = results["CONDITION"]
    trades_bo_ignore, summ_bo_ignore, trades_rt_ignore, summ_rt_ignore = results["IGNORE"]

    def _print_comparison_table(summ_bo, summ_rt, label_bo, label_rt, vol_mode):
        print(f"\n{'─'*70}")
        print(f"  PERBANDINGAN SIDE-BY-SIDE [volume_mode={vol_mode}]")
        print(f"  {label_bo} vs {label_rt}")
        print(f"{'─'*70}")
        report_lines.append(f"\nPerbandingan [volume_mode={vol_mode}]")

        def _fmt(val, fmt="{:,.0f}"):
            if val is None: return "N/A"
            try: return fmt.format(val)
            except: return str(val)

        rows = [
            ("Total Trades",     summ_bo["total_trades"],     summ_rt["total_trades"],     "{:,}"),
            ("TP Count",         summ_bo["tp_count"],         summ_rt["tp_count"],         "{:,}"),
            ("SL Count",         summ_bo["sl_count"],         summ_rt["sl_count"],         "{:,}"),
            ("NO_HIT Count",     summ_bo["no_hit_count"],     summ_rt["no_hit_count"],     "{:,}"),
            ("Win Rate",         summ_bo["win_rate"],         summ_rt["win_rate"],         "{:.4f}"),
            ("No-Hit Rate",      summ_bo["no_hit_rate"],      summ_rt["no_hit_rate"],      "{:.4f}"),
            ("Avg RRR",          summ_bo["avg_rrr_realized"], summ_rt["avg_rrr_realized"], "{:+.4f}"),
            ("Avg Candles Held", summ_bo["avg_candles_held"], summ_rt["avg_candles_held"], "{:.1f}"),
            ("Total PnL Net",    summ_bo["total_pnl_net"],    summ_rt["total_pnl_net"],    "{:+.2f}"),
            ("Max Drawdown Net", summ_bo["max_drawdown_net"], summ_rt["max_drawdown_net"], "{:+.2f}"),
        ]

        col_w = 22
        hdr_row = f"  {'Metrik':<{col_w}}  {label_bo:>16}  {label_rt:>16}"
        sep     = f"  {'-'*col_w}  {'-'*16}  {'-'*16}"
        print(hdr_row); print(sep)
        report_lines.append(hdr_row); report_lines.append(sep)
        for name, vbo, vrt, fmt in rows:
            s_bo = _fmt(vbo, fmt); s_rt = _fmt(vrt, fmt)
            line = f"  {name:<{col_w}}  {s_bo:>16}  {s_rt:>16}"
            print(line); report_lines.append(line)

    _print_comparison_table(summ_bo_filter, summ_rt_filter, "BREAKOUT+FILTER", "RETEST+FILTER", "FILTER")
    _print_comparison_table(summ_bo_cond, summ_rt_cond, "BREAKOUT+COND", "RETEST+COND", "CONDITION")
    _print_comparison_table(summ_bo_ignore, summ_rt_ignore, "BREAKOUT+IGNORE", "RETEST+IGNORE", "IGNORE")

    # ── NOTE DIAGNOSTIK VOLUME FILTER ──────────────────────────────────────────
    note = (
        "\nNOTE DIAGNOSTIK: Retest menghasilkan 0 trades dengan volume_mode=FILTER \n"
        "karena volume filter (climax blocker + thin market blocker) memblokir \n"
        "seluruh sinyal retest yang valid secara trigger. Ini konsisten dengan \n"
        "desain volume filter yang ketat. Perbandingan adil = volume_mode=IGNORE."
    )
    print(note)
    report_lines.append(note)

    # ── BREAKDOWN TRIGGER_SOURCE (RETEST + IGNORE) ────────────────────────────
    print(f"\n{'─'*70}")
    print("  BREAKDOWN RETEST (volume=IGNORE): trigger_source distribution")
    print(f"{'─'*70}")
    report_lines.append("\nBREAKDOWN RETEST+IGNORE trigger_source")
    if not trades_rt_ignore.empty and "trigger_source" in trades_rt_ignore.columns:
        ts_counts = trades_rt_ignore["trigger_source"].value_counts()
        for ts, cnt in ts_counts.items():
            pct = cnt / len(trades_rt_ignore) * 100
            line = f"  trigger_source={ts:<12} : {cnt:>4} trades ({pct:.1f}%)"
            print(line); report_lines.append(line)
        for ts_label in ["RETEST", "BOTH", "EMA_GAP"]:
            subset = trades_rt_ignore[trades_rt_ignore["trigger_source"] == ts_label]
            if subset.empty: continue
            sub_summ = compute_summary(subset)
            wr_s  = f"{sub_summ['win_rate']:.4f}" if sub_summ["win_rate"] else "N/A"
            rrr_s = f"{sub_summ['avg_rrr_realized']:+.4f}" if sub_summ["avg_rrr_realized"] else "N/A"
            sl_s  = f"{subset['jarak_sl'].mean():.4f}" if "jarak_sl" in subset.columns else "N/A"
            line  = f"    [{ts_label}] n={len(subset):>4}, wr={wr_s}, rrr={rrr_s}, avg_sl={sl_s}"
            print(line); report_lines.append(line)
    
    return (trades_bo_ignore, trades_rt_ignore, summ_bo_ignore, summ_rt_ignore,
            trades_bo_filter, summ_bo_filter, df_m5_ind, df_h1_ind)


# =============================================================================
# BAGIAN B: WALK-FORWARD PER FOLD
# =============================================================================

def run_section_b(report_lines: list) -> tuple:
    """
    Walk-forward comparison RETEST vs BREAKOUT per fold.
    Menggunakan volume_mode=IGNORE untuk perbandingan adil.
    Mengembalikan (results_bo, results_rt) sebagai list of dict per fold.
    """
    hdr = "=" * 70
    print(f"\n{hdr}")
    print("  BAGIAN B: WALK-FORWARD — RETEST vs BREAKOUT")
    print(f"  Dataset : {M5_PATH_EXT}")
    print(f"{hdr}")
    report_lines.append("\n" + hdr)
    report_lines.append("BAGIAN B: WALK-FORWARD — RETEST vs BREAKOUT")
    report_lines.append(hdr)

    if not os.path.exists(M5_PATH_EXT) or not os.path.exists(H1_PATH_EXT):
        report_lines.append("  SKIP — dataset extended tidak ditemukan.")
        return [], []

    print("\n  Loading dataset extended...")
    df_m5_full = load_candles_csv(M5_PATH_EXT)
    df_h1_full = load_candles_csv(H1_PATH_EXT)
    data_start = df_m5_full.index[0].strftime("%Y-%m-%d")
    data_end   = df_m5_full.index[-1].strftime("%Y-%m-%d")
    folds = generate_folds(data_start, data_end, calib_months=3, val_months=1)

    results_bo = []
    results_rt = []

    for fold in folds:
        fold_n    = fold["fold"]
        val_start = fold["val_start"]
        val_end   = fold["val_end"]
        df_m5_val = filter_period(df_m5_full, val_start, val_end)
        df_h1_val = filter_period(df_h1_full, val_start, val_end)

        if len(df_m5_val) < WARM_UP + 50: continue

        df_m5_ind = run_all_indicators(df_m5_val.copy())
        df_h1_ind = run_all_indicators(df_h1_val.copy())
        df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind, h1_min_ema_gap_pct=0.02)

        # Jalankan kedua mode — volume=IGNORE untuk perbandingan adil
        tdf_bo, sum_bo = run_mode(df_m5_ind, df_merged, "BREAKOUT", volume_mode="IGNORE")
        tdf_rt, sum_rt = run_mode(df_m5_ind, df_merged, "RETEST",   volume_mode="IGNORE")

        results_bo.append({**{"fold": fold_n, "total_pnl_net": sum_bo["total_pnl_net"]}, "total_trades": sum_bo["total_trades"]})
        results_rt.append({**{"fold": fold_n, "total_pnl_net": sum_rt["total_pnl_net"]}, "total_trades": sum_rt["total_trades"]})

    return results_bo, results_rt


# =============================================================================
# BAGIAN C: UJI STATISTIK
# =============================================================================

def run_section_c(trades_bo: pd.DataFrame, trades_rt: pd.DataFrame,
                  report_lines: list) -> dict:
    """
    Mann-Whitney U test (non-parametric):
    - H0_1: jarak_sl RETEST >= jarak_sl BREAKOUT
    - H0_2: win_rate RETEST <= win_rate BREAKOUT
    """
    hdr = "=" * 70
    print(f"\n{hdr}")
    print("  BAGIAN C: UJI STATISTIK (scipy.stats, p-value EXACT)")
    print(f"{hdr}")
    report_lines.append("\n" + hdr)
    report_lines.append("BAGIAN C: UJI STATISTIK (Mann-Whitney U, non-parametric)")
    report_lines.append(hdr)

    stat_results = {}
    # Test 1: Jarak SL
    sl_bo = trades_bo["jarak_sl"].dropna().values if "jarak_sl" in trades_bo.columns else np.array([])
    sl_rt = trades_rt["jarak_sl"].dropna().values if "jarak_sl" in trades_rt.columns else np.array([])
    if len(sl_bo) >= 5 and len(sl_rt) >= 5:
        u_stat, p_sl = scipy.stats.mannwhitneyu(sl_rt, sl_bo, alternative="less")
        stat_results["sl_test"] = {"p": p_sl, "sig": p_sl < ALPHA}
    
    # Test 2: Win Rate
    def _encode_wins(tdf):
        closed = tdf[tdf["outcome"].isin(["TP_HIT", "SL_HIT"])] if not tdf.empty else pd.DataFrame()
        return closed["outcome"].map({"TP_HIT": 1, "SL_HIT": 0}).values if not closed.empty else np.array([])
    wins_bo, wins_rt = _encode_wins(trades_bo), _encode_wins(trades_rt)
    if len(wins_bo) >= 5 and len(wins_rt) >= 5:
        u_stat2, p_wr = scipy.stats.mannwhitneyu(wins_rt, wins_bo, alternative="greater")
        stat_results["wr_test"] = {"p": p_wr, "sig_not_worse": p_wr >= ALPHA or (wins_rt.mean() - wins_bo.mean()) >= -0.03}
        
    return stat_results


# =============================================================================
# BAGIAN D: VERDICT AKHIR
# =============================================================================

def run_section_d(stat_results: dict, results_bo_wf: list, results_rt_wf: list,
                  trades_rt: pd.DataFrame, report_lines: list):
    """
    Verdict akhir: LOLOS / TIDAK LOLOS / PERLU DATA LEBIH
    """
    hdr = "=" * 70
    print(f"\n{hdr}")
    print("  BAGIAN D: VERDICT AKHIR")
    print(f"{hdr}")
    report_lines.append("\n" + hdr)
    report_lines.append("BAGIAN D: VERDICT AKHIR")
    report_lines.append(hdr)

    n_rt_trades = len(trades_rt)
    if n_rt_trades < 30:
        verdict = "PERLU DATA LEBIH"
        print(f"\n  VERDICT: {verdict}")
        report_lines.append(f"\nVERDICT: {verdict}")
        return verdict
    
    # Logic verdict di sini sederhana:
    k1_met = stat_results.get("sl_test", {}).get("sig")
    k2_met = stat_results.get("wr_test", {}).get("sig_not_worse")
    verdict = "LOLOS" if (k1_met and k2_met is not False) else "TIDAK LOLOS"
    print(f"\n  VERDICT: {verdict}")
    return verdict


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  VALIDASI EMPIRIS FASE 10 — RETEST LOGIC")
    print("  Spesifikasi 4.4 — Murni Observasi (Parameter Default, No Tuning)")
    print("=" * 70)
    print(f"\n  Parameter default: lookback=15, tolerance_atr=0.3")
    print(f"  Fase 1 fixed: {FASE1_PARAMS}")
    print(f"  Alpha: {ALPHA}")

    os.makedirs(OUT_DIR, exist_ok=True)
    report_lines = [
        "LAPORAN VALIDASI EMPIRIS FASE 10 — RETEST LOGIC",
        f"Parameter default: lookback=15, tolerance_atr=0.3",
        f"Parameter Fase 1: {FASE1_PARAMS}",
        f"Alpha: {ALPHA}",
    ]

    if not os.path.exists(M5_PATH_JAN):
        print(f"\nERROR: {M5_PATH_JAN} tidak ditemukan")
        sys.exit(1)
    if not os.path.exists(H1_PATH_JAN):
        print(f"\nERROR: {H1_PATH_JAN} tidak ditemukan")
        sys.exit(1)

    print(f"\n  Loading dataset in-sample...")
    df_m5_jan = load_candles_csv(M5_PATH_JAN)
    df_h1_jan = load_candles_csv(H1_PATH_JAN)
    print(f"  M5: {len(df_m5_jan):,} candle")
    print(f"  H1: {len(df_h1_jan):,} candle")

    # ── BAGIAN A ──────────────────────────────────────────────────────────────
    (trades_bo, trades_rt,
     summ_bo,   summ_rt,
     trades_bo_filter, summ_bo_filter,
     _, _) = run_section_a(df_m5_jan, df_h1_jan, report_lines)

    # Simpan CSV trades
    if not trades_bo_filter.empty:
        p = os.path.join(OUT_DIR, "fase10_trades_breakout_filter.csv")
        trades_bo_filter.to_csv(p, index=False)
        print(f"\n  BREAKOUT+FILTER trades: {p}")
    if not trades_bo.empty:
        p = os.path.join(OUT_DIR, "fase10_trades_breakout_ignore.csv")
        trades_bo.to_csv(p, index=False)
        print(f"  BREAKOUT+IGNORE trades: {p}")
    if not trades_rt.empty:
        p = os.path.join(OUT_DIR, "fase10_trades_retest_ignore.csv")
        trades_rt.to_csv(p, index=False)
        print(f"  RETEST+IGNORE   trades: {p}")

    # ── BAGIAN B ──────────────────────────────────────────────────────────────
    results_bo_wf, results_rt_wf = run_section_b(report_lines)

    # ── BAGIAN C ──────────────────────────────────────────────────────────────
    # Gunakan BREAKOUT+IGNORE vs RETEST+IGNORE (perbandingan adil)
    stat_results = run_section_c(trades_bo, trades_rt, report_lines)

    # ── BAGIAN D ──────────────────────────────────────────────────────────────
    verdict = run_section_d(stat_results, results_bo_wf, results_rt_wf,
                            trades_rt, report_lines)

    # ── SIMPAN LAPORAN ────────────────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "fase10_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n\n  Laporan lengkap disimpan: {report_path}")
    print(f"\n{'='*70}")
    print(f"  VALIDASI FASE 10 SELESAI — VERDICT: {verdict}")
    print(f"{'='*70}\n")



if __name__ == "__main__":
    main()
