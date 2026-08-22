"""
scripts/run_walk_forward_regime.py
====================================
Walk-Forward Validation per Regime/Strategy — Fase 22.

TUJUAN:
    Gate validasi definitif untuk ketiga strategi regime-based (TREND_FOLLOWING,
    RANGE_REVERSAL, BREAKOUT_RETEST). Walk-forward adalah gate definitif, bukan
    backtest single-window. Hasil Fase 21 (208 trade, 14 bulan) adalah INDIKASI
    awal, bukan bukti edge — fase ini yang menjawab apakah masing-masing strategi
    punya edge konsisten lintas waktu.

METODOLOGI:
    - Skema fold: 3 bulan kalibrasi + 1 bulan validasi, geser 1 bulan (~10 fold)
    - Sistem baru (regime-based): run_regime_backtest() per fold
    - Baseline pembanding: run_backtest() (sistem lama) 1x pada dataset penuh 14 bulan
    - Statistik: Wilcoxon signed-rank test (1-sisi), korelasi Pearson+Spearman
    - Koreksi multiple comparison: Bonferroni DAN Benjamini-Hochberg FDR GABUNGAN
      (9 hipotesis total: 3 Wilcoxon + 3 Pearson + 3 Spearman)

REUSE (TIDAK DITULIS ULANG):
    - generate_folds()        dari scripts/run_walk_forward.py
    - filter_period()         dari scripts/run_oos_validation.py
    - run_regime_backtest()   dari engine/backtester_regime.py
    - run_backtest()          dari engine/backtester.py

HARD CONSTRAINTS:
    - TIDAK ada perubahan pada file manapun dari Fase 12-21 atau sistem lama
    - Murni if-else, tidak ada AI/ML
    - Verdict per strategi INDEPENDEN satu sama lain
    - Verdict HANYA: LOLOS / TIDAK LOLOS / PERLU DATA LEBIH (tidak ada bersyarat)

FASE 20 (NEWS FILTER) DI-SKIP atas keputusan pemilik proyek.
"""

import os
import sys
import time
from datetime import timezone

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.backtester import run_backtest
from engine.backtester_regime import run_regime_backtest
from scripts.run_walk_forward import generate_folds
from scripts.run_oos_validation import filter_period


# =============================================================================
# KONSTANTA — DOKUMENTASI WAJIB (justifikasi setiap nilai)
# =============================================================================

FOLD_MIN_TRADES = 3
"""
Minimum trade per fold per strategi agar fold dianggap "eligible" masuk statistik.
Fold hanya 1 bulan — ambang ini sengaja kecil (3) karena kita tidak ingin membuang
fold hanya karena aktivitas rendah dalam 1 bulan. Angka 3 dipilih sebagai titik awal
pragmatis; BELUM DIKALIBRASI — perlu review setelah data lebih banyak tersedia.
"""

STRATEGY_MIN_TOTAL_TRADES = 30
"""
Total trade gabungan SEMUA fold eligible minimum agar strategi layak mendapat verdict
LOLOS/TIDAK LOLOS (bukan PERLU DATA LEBIH). Nilai ini SAMA dengan DEFAULT_MIN_TRADES
di run_oos_validation.py — sengaja disamakan untuk konsistensi konvensi proyek.
"""

ALPHA = 0.05
"""
Tingkat signifikansi. Dipakai SETELAH koreksi multiple comparison (BH-FDR).
"""

# Parameter Fase 1 untuk baseline (sistem lama) — SAMA persis seperti run_walk_forward.py
FASE1_PARAMS_OLD_SYSTEM = {
    "atr_multiplier": 0.9,
    "swing_lookback": 15,
    "swing_wing": 3,
    "rrr_min": 1.3,
}

STRATEGY_NAMES = ["RANGE_REVERSAL", "BREAKOUT_RETEST", "TREND_FOLLOWING"]

DATA_START = "2025-06-01"
DATA_END   = "2026-07-25"

RESULTS_DIR = os.path.join(ROOT_DIR, "data", "backtest_results")
REPORT_PATH = os.path.join(RESULTS_DIR, "walk_forward_regime_report.txt")

M5_PATH  = os.path.join(ROOT_DIR, "data", "historical", f"XAUUSD_M5_{DATA_START}_{DATA_END}.csv")
M15_PATH = os.path.join(ROOT_DIR, "data", "historical", f"XAUUSD_M15_{DATA_START}_{DATA_END}.csv")
H1_PATH  = os.path.join(ROOT_DIR, "data", "historical", f"XAUUSD_H1_{DATA_START}_{DATA_END}.csv")


# =============================================================================
# FUNGSI LOGIKA INTI — TESTABLE, TERPISAH DARI main()
# =============================================================================

def filter_eligible_folds(
    fold_results: list,
    min_trades: int = FOLD_MIN_TRADES,
) -> list:
    """
    Filter fold yang eligible masuk statistik (total_trades >= min_trades).

    Fold yang tidak eligible TETAP ADA di fold_results asli (untuk laporan),
    tapi TIDAK dikembalikan oleh fungsi ini. Keduanya harus dilaporkan terpisah.

    Parameter:
        fold_results : list of dict — satu dict per fold, dengan key "total_trades"
        min_trades   : Minimum trade agar fold dianggap eligible

    Return:
        list[dict] — hanya fold yang eligible
    """
    return [f for f in fold_results if (f.get("total_trades") or 0) >= min_trades]


def compute_fold_positive_pct(eligible_folds: list) -> float:
    """
    Hitung persentase fold dengan avg_rrr_realized > 0 (fold "positif").

    Ini adalah kriteria utama #1 untuk LOLOS: >= 60% fold harus positif.

    Parameter:
        eligible_folds : list of dict dengan key "avg_rrr_realized"

    Return:
        float — persentase fold positif (0.0-100.0), atau 0.0 jika tidak ada fold
    """
    if not eligible_folds:
        return 0.0
    positif = sum(
        1 for f in eligible_folds
        if (f.get("avg_rrr_realized") or 0.0) > 0
    )
    return 100.0 * positif / len(eligible_folds)


def run_wilcoxon_test(eligible_folds: list) -> dict:
    """
    Jalankan Wilcoxon signed-rank test 1-sisi (alternative="greater") pada
    array avg_rrr_realized dari semua fold eligible, uji terhadap median = 0.

    Catatan: p-value mentah yang dikembalikan BELUM dikoreksi multiple comparison.
    Koreksi Bonferroni/BH-FDR dilakukan di apply_multiple_comparison_correction(),
    gabungan dengan semua hipotesis dari ketiga strategi.

    Parameter:
        eligible_folds : list of dict dengan key "avg_rrr_realized"

    Return:
        dict dengan key:
            "statistic"   : float — test statistic Wilcoxon
            "p_raw"       : float — p-value mentah (1-sisi, belum dikoreksi)
            "n"           : int   — jumlah fold yang masuk uji
            "rrr_values"  : list  — array avg_rrr_realized yang diuji
            "error"       : str | None — pesan error jika uji tidak bisa dijalankan
    """
    rrr_values = [
        float(f.get("avg_rrr_realized") or 0.0)
        for f in eligible_folds
    ]

    if len(rrr_values) < 2:
        return {
            "statistic" : None,
            "p_raw"     : 1.0,
            "n"         : len(rrr_values),
            "rrr_values": rrr_values,
            "error"     : f"Tidak cukup fold untuk Wilcoxon (n={len(rrr_values)}, min=2)",
        }

    # Periksa apakah semua nilai sama (tidak bisa Wilcoxon)
    if len(set(rrr_values)) == 1:
        return {
            "statistic" : 0.0,
            "p_raw"     : 1.0,
            "n"         : len(rrr_values),
            "rrr_values": rrr_values,
            "error"     : f"Semua avg_rrr_realized sama ({rrr_values[0]:.4f}), Wilcoxon tidak informatif",
        }

    try:
        stat, p = stats.wilcoxon(rrr_values, alternative="greater")
        return {
            "statistic" : float(stat),
            "p_raw"     : float(p),
            "n"         : len(rrr_values),
            "rrr_values": rrr_values,
            "error"     : None,
        }
    except Exception as e:
        return {
            "statistic" : None,
            "p_raw"     : 1.0,
            "n"         : len(rrr_values),
            "rrr_values": rrr_values,
            "error"     : str(e),
        }


def run_correlation_diagnostics(eligible_folds: list) -> dict:
    """
    Diagnostik korelasi temporal: nomor fold (urutan kronologis) vs avg_rrr_realized.

    Pearson DAN Spearman WAJIB dilaporkan bersamaan. Ini INFORMATIF, bukan
    kriteria verdict — dipakai untuk mendeteksi drift performa strategi seiring waktu.

    Parameter:
        eligible_folds : list of dict dengan key "fold" (nomor fold) dan "avg_rrr_realized"

    Return:
        dict dengan key:
            "pearson_r"    : float
            "pearson_p"    : float
            "spearman_r"   : float
            "spearman_p"   : float
            "n"            : int
            "error"        : str | None
    """
    fold_order = [f.get("fold", i+1) for i, f in enumerate(eligible_folds)]
    rrr_values = [float(f.get("avg_rrr_realized") or 0.0) for f in eligible_folds]

    if len(fold_order) < 3:
        return {
            "pearson_r"  : None,
            "pearson_p"  : None,
            "spearman_r" : None,
            "spearman_p" : None,
            "n"          : len(fold_order),
            "error"      : f"Tidak cukup fold untuk korelasi (n={len(fold_order)}, min=3)",
        }

    try:
        pr, pp = stats.pearsonr(fold_order, rrr_values)
        sr, sp = stats.spearmanr(fold_order, rrr_values)
        return {
            "pearson_r"  : float(pr),
            "pearson_p"  : float(pp),
            "spearman_r" : float(sr),
            "spearman_p" : float(sp),
            "n"          : len(fold_order),
            "error"      : None,
        }
    except Exception as e:
        return {
            "pearson_r"  : None,
            "pearson_p"  : None,
            "spearman_r" : None,
            "spearman_p" : None,
            "n"          : len(fold_order),
            "error"      : str(e),
        }


def _bh_fdr(p_values: list, alpha: float = ALPHA) -> list:
    """
    Benjamini-Hochberg FDR correction.

    Pola implementasi dari scripts/_fase7_full_validation.py (bh_fdr()).
    Ditulis ulang sebagai fungsi lokal karena _fase7_full_validation.py
    adalah script privat (prefixed _), bukan modul produksi.

    Return:
        list[bool] — True jika hipotesis ke-i signifikan setelah koreksi
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    for k, (orig_idx, p) in enumerate(indexed):
        if p <= (k + 1) / n * alpha:
            rejected[orig_idx] = True
    # BH monotone enforcement: jika hipotesis ke-k ditolak, semua yang lebih kecil juga ditolak
    max_k_rejected = -1
    for k, (orig_idx, _p) in enumerate(indexed):
        if rejected[orig_idx]:
            max_k_rejected = k
    for k, (orig_idx, _p) in enumerate(indexed):
        if k <= max_k_rejected:
            rejected[orig_idx] = True
    return rejected


def apply_multiple_comparison_correction(
    p_values: list,
    alpha: float = ALPHA,
) -> dict:
    """
    Terapkan koreksi Bonferroni DAN Benjamini-Hochberg FDR pada semua p-value
    yang diberikan SEKALIGUS (satu koreksi gabungan, bukan per-strategi terpisah).

    Spesifikasi: p_values adalah list dari 9 hipotesis gabungan:
        - 3 Wilcoxon (satu per strategi)
        - 3 Pearson (satu per strategi)
        - 3 Spearman (satu per strategi)

    Parameter:
        p_values : list[float] — p-value mentah dari semua hipotesis
        alpha    : float       — tingkat signifikansi

    Return:
        dict dengan key:
            "bonferroni_rejected" : list[bool] — True jika signifikan setelah Bonferroni
            "bh_fdr_rejected"     : list[bool] — True jika signifikan setelah BH-FDR
            "bonferroni_threshold": float — threshold Bonferroni (alpha / n)
            "n_hypotheses"        : int
    """
    n = len(p_values)
    if n == 0:
        return {
            "bonferroni_rejected" : [],
            "bh_fdr_rejected"     : [],
            "bonferroni_threshold": None,
            "n_hypotheses"        : 0,
        }

    bonf_threshold = alpha / n
    bonf_rejected  = [p <= bonf_threshold for p in p_values]
    bh_rejected    = _bh_fdr(p_values, alpha=alpha)

    return {
        "bonferroni_rejected" : bonf_rejected,
        "bh_fdr_rejected"     : bh_rejected,
        "bonferroni_threshold": bonf_threshold,
        "n_hypotheses"        : n,
    }


def determine_verdict(
    total_eligible_trades : int,
    pct_positive_folds    : float,
    wilcoxon_bh_rejected  : bool,
    min_total_trades      : int = STRATEGY_MIN_TOTAL_TRADES,
    min_pct_positive      : float = 60.0,
) -> str:
    """
    Tentukan verdict akhir per strategi secara INDEPENDEN.

    Hierarki keputusan (berurutan):
        1. Jika total_eligible_trades < min_total_trades -> "PERLU DATA LEBIH"
           (tidak perlu periksa kriteria lain — sampel tidak cukup)
        2. Jika KEDUA kriteria terpenuhi:
               - pct_positive_folds >= min_pct_positive (konsistensi arah)
               - wilcoxon_bh_rejected is True (signifikan setelah BH-FDR, arah positif)
           -> "LOLOS"
        3. Selain itu (sampel cukup tapi salah satu/kedua kriteria gagal) -> "TIDAK LOLOS"

    Parameter:
        total_eligible_trades : Total trade dari semua fold eligible
        pct_positive_folds    : % fold dengan avg_rrr_realized > 0
        wilcoxon_bh_rejected  : True jika Wilcoxon p < alpha setelah BH-FDR
        min_total_trades      : Batas minimum total trade
        min_pct_positive      : Batas minimum % fold positif

    Return:
        str — "LOLOS" | "TIDAK LOLOS" | "PERLU DATA LEBIH"
    """
    if total_eligible_trades < min_total_trades:
        return "PERLU DATA LEBIH"

    if pct_positive_folds >= min_pct_positive and wilcoxon_bh_rejected:
        return "LOLOS"

    return "TIDAK LOLOS"


# =============================================================================
# HELPER INTERNAL LAPORAN
# =============================================================================

def _fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v:.1f}%"

def _fmt_r(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"

def _fmt_p(v):
    if v is None:
        return "N/A"
    return f"{v:.6f}"

def _fmt_pts(v):
    if v is None:
        return "N/A"
    return f"{v:+.2f} pts"


# =============================================================================
# MAIN — LANGKAH 1-7
# =============================================================================

def main():
    t_start = time.time()
    lines = []  # Semua output dikumpulkan dulu, lalu dicetak + disimpan

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("  FASE 22: WALK-FORWARD VALIDATION PER REGIME/STRATEGY")
    emit("  XAUUSD M5/M15/H1 — Sistem Rule-Based")
    emit("=" * 70)
    emit(f"  Data        : {DATA_START} -> {DATA_END}")
    emit(f"  Fold min trades: {FOLD_MIN_TRADES} (per fold per strategi)")
    emit(f"  Strat min total: {STRATEGY_MIN_TOTAL_TRADES} (total semua fold eligible)")
    emit(f"  Alpha       : {ALPHA}")
    emit(f"  Koreksi     : Bonferroni + Benjamini-Hochberg FDR (gabungan 9 hipotesis)")
    emit()

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    emit("[1/7] Memuat data historis...")
    df_m5_raw  = load_candles_csv(M5_PATH)
    df_m15_raw = load_candles_csv(M15_PATH)
    df_h1_raw  = load_candles_csv(H1_PATH)
    emit(f"   M5 : {len(df_m5_raw):,} candle ({df_m5_raw.index[0]} -> {df_m5_raw.index[-1]})")
    emit(f"   M15: {len(df_m15_raw):,} candle ({df_m15_raw.index[0]} -> {df_m15_raw.index[-1]})")
    emit(f"   H1 : {len(df_h1_raw):,} candle ({df_h1_raw.index[0]} -> {df_h1_raw.index[-1]})")
    emit()

    # -------------------------------------------------------------------------
    # LANGKAH 2 — Baseline sistem lama (1x pada full dataset)
    # -------------------------------------------------------------------------
    emit("[2/7] Menjalankan baseline sistem LAMA (single-strategy, full 14 bulan)...")
    emit("   (Ini mungkin memakan waktu beberapa menit)")
    t_bl = time.time()
    _trades_old_df, summary_old = run_backtest(
        df_m5_raw.copy(),
        df_h1_raw.copy(),
        atr_multiplier=FASE1_PARAMS_OLD_SYSTEM["atr_multiplier"],
        swing_lookback=FASE1_PARAMS_OLD_SYSTEM["swing_lookback"],
        swing_wing=FASE1_PARAMS_OLD_SYSTEM["swing_wing"],
        rrr_min=FASE1_PARAMS_OLD_SYSTEM["rrr_min"],
        verbose=False,
    )
    elapsed_bl = time.time() - t_bl
    emit(f"   Selesai: {summary_old.get('total_trades', 0)} trade ({elapsed_bl:.1f}s)")
    emit()

    # -------------------------------------------------------------------------
    # LANGKAH 1 — Generate folds & jalankan backtest regime per fold
    # -------------------------------------------------------------------------
    emit("[3/7] Generate walk-forward folds...")
    folds = generate_folds(
        data_start=DATA_START,
        data_end=DATA_END,
        calib_months=3,
        val_months=1,
    )
    emit(f"   Jumlah fold: {len(folds)}")
    for f in folds:
        emit(f"   Fold {f['fold']:2d}: calib={f['calib_start']}~{f['calib_end']}, "
             f"val={f['val_start']}~{f['val_end']}")
    emit()

    emit("[4/7] Menjalankan backtest regime per fold (ini bisa memakan waktu 10-30 menit)...")
    emit()

    # Inisialisasi koleksi per strategi
    fold_results_per_strategy = {s: [] for s in STRATEGY_NAMES}
    fold_results_aggregate    = []

    for fold_info in folds:
        fold_n    = fold_info["fold"]
        val_start = fold_info["val_start"]
        val_end   = fold_info["val_end"]

        emit(f"   Fold {fold_n:2d} [{val_start} ~ {val_end}]...")

        # Filter ke window validasi (reuse filter_period)
        df_m5_val  = filter_period(df_m5_raw,  val_start, val_end)
        df_m15_val = filter_period(df_m15_raw, val_start, val_end)
        df_h1_val  = filter_period(df_h1_raw,  val_start, val_end)

        if len(df_m5_val) < 200:
            emit(f"   >> SKIP: M5 terlalu sedikit ({len(df_m5_val)} candle < 200)")
            for s in STRATEGY_NAMES:
                fold_results_per_strategy[s].append({
                    "fold"             : fold_n,
                    "val_start"        : val_start,
                    "val_end"          : val_end,
                    "total_trades"     : 0,
                    "win_rate"         : None,
                    "avg_rrr_realized" : None,
                    "total_pnl_net"    : None,
                    "skip_reason"      : "SKIP_TOO_FEW_CANDLES",
                })
            fold_results_aggregate.append({
                "fold"        : fold_n,
                "val_start"   : val_start,
                "val_end"     : val_end,
                "total_trades": 0,
                "win_rate"    : None,
                "pnl_net"     : None,
                "skip_reason" : "SKIP_TOO_FEW_CANDLES",
            })
            continue

        t_fold = time.time()
        trades_df, seg_summary = run_regime_backtest(
            df_m5_val.copy(),
            df_m15_val.copy(),
            df_h1_val.copy(),
            verbose=False,
        )
        elapsed_fold = time.time() - t_fold

        n_trades = len(trades_df)
        emit(f"      -> {n_trades} trade ({elapsed_fold:.1f}s)")

        per_strat = seg_summary.get("per_strategy", {})
        overall   = seg_summary.get("overall", {})

        fold_results_aggregate.append({
            "fold"        : fold_n,
            "val_start"   : val_start,
            "val_end"     : val_end,
            "total_trades": n_trades,
            "win_rate"    : overall.get("win_rate_pct"),
            "avg_rrr"     : overall.get("avg_rrr_realized"),
            "pnl_net"     : overall.get("total_pnl_net"),
            "skip_reason" : None,
        })

        for strat_name in STRATEGY_NAMES:
            s_data = per_strat.get(strat_name, {})
            fold_results_per_strategy[strat_name].append({
                "fold"             : fold_n,
                "val_start"        : val_start,
                "val_end"          : val_end,
                "total_trades"     : s_data.get("total_trades", 0) or 0,
                "win_rate"         : s_data.get("win_rate_pct"),
                "avg_rrr_realized" : s_data.get("avg_rrr_realized"),
                "total_pnl_net"    : s_data.get("total_pnl_net"),
                "skip_reason"      : None,
            })

    emit()
    emit("[5/7] Menghitung statistik per strategi...")
    emit()

    # -------------------------------------------------------------------------
    # LANGKAH 3 — Statistik per strategi
    # -------------------------------------------------------------------------
    strat_stats = {}

    for strat_name in STRATEGY_NAMES:
        fold_list  = fold_results_per_strategy[strat_name]
        eligible   = filter_eligible_folds(fold_list, min_trades=FOLD_MIN_TRADES)
        n_eligible = len(eligible)
        total_trade = sum(f.get("total_trades", 0) or 0 for f in eligible)
        pct_pos    = compute_fold_positive_pct(eligible)
        wilcoxon   = run_wilcoxon_test(eligible)
        corr       = run_correlation_diagnostics(eligible)

        strat_stats[strat_name] = {
            "fold_list"               : fold_list,
            "eligible_folds"          : eligible,
            "n_eligible"              : n_eligible,
            "total_eligible_trades"   : total_trade,
            "pct_positive_folds"      : pct_pos,
            "wilcoxon"                : wilcoxon,
            "correlation"             : corr,
        }

    # Susun 9 p-value dalam urutan tetap untuk koreksi gabungan:
    # [RR_wilcox, BR_wilcox, TF_wilcox, RR_pearson, BR_pearson, TF_pearson,
    #  RR_spearman, BR_spearman, TF_spearman]
    all_p_values = []
    hyp_labels   = []

    for strat_name in STRATEGY_NAMES:
        st = strat_stats[strat_name]
        all_p_values.append(st["wilcoxon"].get("p_raw", 1.0) or 1.0)
        hyp_labels.append(f"{strat_name}_wilcoxon")
    for strat_name in STRATEGY_NAMES:
        st = strat_stats[strat_name]
        all_p_values.append(st["correlation"].get("pearson_p") or 1.0)
        hyp_labels.append(f"{strat_name}_pearson")
    for strat_name in STRATEGY_NAMES:
        st = strat_stats[strat_name]
        all_p_values.append(st["correlation"].get("spearman_p") or 1.0)
        hyp_labels.append(f"{strat_name}_spearman")

    # -------------------------------------------------------------------------
    # LANGKAH 4 — Koreksi multiple comparison (gabungan 9 hipotesis)
    # -------------------------------------------------------------------------
    correction    = apply_multiple_comparison_correction(all_p_values, alpha=ALPHA)
    bonf_rejected = correction["bonferroni_rejected"]
    bh_rejected   = correction["bh_fdr_rejected"]

    # Distribusikan hasil koreksi kembali ke masing-masing strategi
    for i, strat_name in enumerate(STRATEGY_NAMES):
        strat_stats[strat_name]["wilcoxon_bonf_rejected"]    = bonf_rejected[i]
        strat_stats[strat_name]["wilcoxon_bh_rejected"]      = bh_rejected[i]
        strat_stats[strat_name]["wilcoxon_p_corrected_bonf"] = (
            min(all_p_values[i] * len(all_p_values), 1.0)
        )
        strat_stats[strat_name]["pearson_bh_rejected"]  = bh_rejected[3 + i]
        strat_stats[strat_name]["spearman_bh_rejected"] = bh_rejected[6 + i]

    # -------------------------------------------------------------------------
    # LANGKAH 5 — Verdict per strategi (INDEPENDEN)
    # -------------------------------------------------------------------------
    for strat_name in STRATEGY_NAMES:
        st = strat_stats[strat_name]
        verdict = determine_verdict(
            total_eligible_trades=st["total_eligible_trades"],
            pct_positive_folds=st["pct_positive_folds"],
            wilcoxon_bh_rejected=st["wilcoxon_bh_rejected"],
        )
        strat_stats[strat_name]["verdict"] = verdict

    # -------------------------------------------------------------------------
    # LANGKAH 6 — Sistem baru (full dataset) untuk tabel perbandingan
    # -------------------------------------------------------------------------
    emit("[6/7] Menjalankan backtest sistem BARU (full 14 bulan) untuk perbandingan...")
    t_new = time.time()
    _trades_new_df, seg_full = run_regime_backtest(
        df_m5_raw.copy(),
        df_m15_raw.copy(),
        df_h1_raw.copy(),
        verbose=False,
    )
    elapsed_new = time.time() - t_new
    new_overall = seg_full.get("overall", {})
    emit(f"   Selesai: {new_overall.get('total_trades', 0)} trade ({elapsed_new:.1f}s)")
    emit()

    # -------------------------------------------------------------------------
    # LANGKAH 7 — Cetak & simpan laporan
    # -------------------------------------------------------------------------
    emit("[7/7] Menyusun laporan...")
    emit()

    # Metrik sistem lama
    old_total = summary_old.get("total_trades", 0)
    old_wr    = summary_old.get("win_rate_pct")
    old_rrr   = summary_old.get("avg_rrr_realized")
    old_pnl   = summary_old.get("total_pnl_net")
    old_dd    = summary_old.get("max_drawdown_net")

    # Metrik sistem baru
    new_total = new_overall.get("total_trades", 0)
    new_wr    = new_overall.get("win_rate_pct")
    new_rrr   = new_overall.get("avg_rrr_realized")
    new_pnl   = new_overall.get("total_pnl_net")
    new_dd    = new_overall.get("max_drawdown_net")

    emit()
    emit("=" * 70)
    emit("  HASIL WALK-FORWARD VALIDATION — FASE 22")
    emit("=" * 70)
    emit(f"  Dataset         : {DATA_START} -> {DATA_END} (14 bulan)")
    emit(f"  Jumlah fold     : {len(folds)}")
    emit(f"  FOLD_MIN_TRADES : {FOLD_MIN_TRADES}")
    emit(f"  STRAT_MIN_TOTAL : {STRATEGY_MIN_TOTAL_TRADES}")
    emit(f"  ALPHA           : {ALPHA}")
    emit(f"  Hipotesis total : {len(all_p_values)} (Bonferroni threshold = {correction['bonferroni_threshold']:.6f})")
    emit()

    # Tabel per fold (overview)
    emit("-" * 70)
    emit("  TABEL PER FOLD (SEMUA STRATEGI, OVERVIEW)")
    emit("-" * 70)
    emit(f"  {'Fold':>4}  {'Validasi':>21}  {'Total':>5}  {'RR':>5}  {'BR':>5}  {'TF':>5}")
    emit(f"  {'-'*4}  {'-'*21}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for fold_info in folds:
        fn = fold_info["fold"]
        fa = next((a for a in fold_results_aggregate if a["fold"] == fn), {})
        rr = next((f for f in fold_results_per_strategy["RANGE_REVERSAL"] if f["fold"] == fn), {})
        br = next((f for f in fold_results_per_strategy["BREAKOUT_RETEST"] if f["fold"] == fn), {})
        tf = next((f for f in fold_results_per_strategy["TREND_FOLLOWING"] if f["fold"] == fn), {})
        val_label = f"{fold_info['val_start']}~{fold_info['val_end']}"
        emit(
            f"  {fn:>4}  {val_label:>21}  "
            f"{fa.get('total_trades', 0):>5}  "
            f"{rr.get('total_trades', 0):>5}  "
            f"{br.get('total_trades', 0):>5}  "
            f"{tf.get('total_trades', 0):>5}"
        )
    emit()

    # Detail per strategi
    for strat_name in STRATEGY_NAMES:
        st      = strat_stats[strat_name]
        wilcox  = st["wilcoxon"]
        corr    = st["correlation"]
        verdict = st["verdict"]

        emit("-" * 70)
        emit(f"  STRATEGI: {strat_name}")
        emit("-" * 70)

        # Tabel fold detail
        emit(f"  {'Fold':>4}  {'Validasi':>21}  {'Trade':>5}  {'WinRate':>7}  {'AvgRRR':>7}  {'Eligible':>9}")
        emit(f"  {'-'*4}  {'-'*21}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*9}")
        for fr in st["fold_list"]:
            fn = fr["fold"]
            is_elig = (fr.get("total_trades") or 0) >= FOLD_MIN_TRADES
            val_label = f"{fr['val_start']}~{fr['val_end']}"
            wr_s  = _fmt_pct(fr.get("win_rate"))
            rrr_s = _fmt_r(fr.get("avg_rrr_realized"))
            elig_s = "Ya" if is_elig else f"Tidak (<{FOLD_MIN_TRADES})"
            emit(
                f"  {fn:>4}  {val_label:>21}  "
                f"{fr.get('total_trades', 0):>5}  "
                f"{wr_s:>7}  "
                f"{rrr_s:>7}  "
                f"{elig_s:>9}"
            )

        emit()
        emit(f"  Fold eligible (>= {FOLD_MIN_TRADES} trade): {st['n_eligible']} dari {len(st['fold_list'])}")
        emit(f"  Total trade (fold eligible)            : {st['total_eligible_trades']}")

        if st["total_eligible_trades"] < STRATEGY_MIN_TOTAL_TRADES:
            emit(f"  >> SAMPEL TIDAK CUKUP ({st['total_eligible_trades']} < {STRATEGY_MIN_TOTAL_TRADES})")
            emit(f"     Uji statistik dilewati (verdict langsung: PERLU DATA LEBIH)")
        else:
            emit(f"  % fold positif (avg_rrr > 0)           : {_fmt_pct(st['pct_positive_folds'])} (syarat >= 60%)")
            emit()
            emit(f"  Wilcoxon signed-rank (1-sisi, alternative=greater, terhadap median=0):")
            if wilcox.get("error"):
                emit(f"    Error: {wilcox['error']}")
            else:
                emit(f"    n fold     : {wilcox['n']}")
                emit(f"    Statistic  : {wilcox['statistic']:.4f}")
                emit(f"    p mentah   : {_fmt_p(wilcox['p_raw'])}")
                emit(f"    p Bonf.*   : {_fmt_p(st.get('wilcoxon_p_corrected_bonf'))} "
                     f"(sig: {'Ya' if st['wilcoxon_bonf_rejected'] else 'Tidak'})")
                emit(f"    BH-FDR     : rejected={'Ya' if st['wilcoxon_bh_rejected'] else 'Tidak'}")
                emit(f"    (*) Bonferroni: p_raw * {len(all_p_values)} hipotesis, capped at 1.0")
            emit()
            emit(f"  Korelasi temporal (fold-order vs avg_rrr) [INFORMATIF, bukan kriteria verdict]:")
            if corr.get("error"):
                emit(f"    Error: {corr['error']}")
            else:
                emit(f"    Pearson : r={_fmt_r(corr['pearson_r'])}, p={_fmt_p(corr['pearson_p'])} "
                     f"(sig BH-FDR: {'Ya' if st['pearson_bh_rejected'] else 'Tidak'})")
                emit(f"    Spearman: r={_fmt_r(corr['spearman_r'])}, p={_fmt_p(corr['spearman_p'])} "
                     f"(sig BH-FDR: {'Ya' if st['spearman_bh_rejected'] else 'Tidak'})")

        emit()
        emit("  " + "=" * 50)
        emit(f"  VERDICT {strat_name}: {verdict}")
        emit("  " + "=" * 50)
        emit()

    # Tabel perbandingan sistem lama vs baru
    emit("=" * 70)
    emit("  TABEL PERBANDINGAN SISTEM LAMA VS BARU (dataset 14 bulan penuh)")
    emit("=" * 70)
    emit()
    emit(f"  {'Metrik':<22}  {'Sistem LAMA':>18}  {'Sistem BARU':>18}")
    emit(f"  {'-'*22}  {'-'*18}  {'-'*18}")
    emit(f"  {'Total trade':<22}  {str(old_total):>18}  {str(new_total):>18}")
    emit(f"  {'Win Rate':<22}  {_fmt_pct(old_wr):>18}  {_fmt_pct(new_wr):>18}")
    emit(f"  {'Avg RRR realized':<22}  {(_fmt_r(old_rrr)+' R'):>18}  {(_fmt_r(new_rrr)+' R'):>18}")
    emit(f"  {'Total PnL Net':<22}  {_fmt_pts(old_pnl):>18}  {_fmt_pts(new_pnl):>18}")
    emit(f"  {'Max Drawdown':<22}  {_fmt_pts(old_dd):>18}  {_fmt_pts(new_dd):>18}")
    emit()
    emit("  Catatan:")
    emit("  - Sistem LAMA: run_backtest() (single-strategy, rule_engine.py)")
    emit(f"    Parameter: ATR={FASE1_PARAMS_OLD_SYSTEM['atr_multiplier']}, "
         f"lookback={FASE1_PARAMS_OLD_SYSTEM['swing_lookback']}, "
         f"wing={FASE1_PARAMS_OLD_SYSTEM['swing_wing']}, "
         f"rrr_min={FASE1_PARAMS_OLD_SYSTEM['rrr_min']}")
    emit("  - Sistem BARU: run_regime_backtest() (3 strategi per regime, rrr_min=None)")
    emit()

    # Ringkasan verdict akhir
    emit("=" * 70)
    emit("  RINGKASAN VERDICT AKHIR")
    emit("=" * 70)
    for strat_name in STRATEGY_NAMES:
        verdict  = strat_stats[strat_name]["verdict"]
        n_trade  = strat_stats[strat_name]["total_eligible_trades"]
        pct_pos  = strat_stats[strat_name]["pct_positive_folds"]
        bh_sig   = strat_stats[strat_name]["wilcoxon_bh_rejected"]
        emit(f"  {strat_name:<20}: {verdict}")
        emit(f"    n_trade_eligible={n_trade}, "
             f"pct_positif={_fmt_pct(pct_pos)}, "
             f"Wilcoxon BH-FDR={'sig' if bh_sig else 'tidak sig'}")
    emit()

    # Tabel koreksi multiple comparison lengkap
    emit("-" * 70)
    emit("  DETAIL KOREKSI MULTIPLE COMPARISON (gabungan 9 hipotesis)")
    emit("-" * 70)
    emit(f"  {'#':>2}  {'Hipotesis':<32}  {'p_raw':>10}  {'Bonf.':>6}  {'BH-FDR':>6}")
    emit(f"  {'-'*2}  {'-'*32}  {'-'*10}  {'-'*6}  {'-'*6}")
    for i, (label, p, bonf_r, bh_r) in enumerate(
        zip(hyp_labels, all_p_values, bonf_rejected, bh_rejected)
    ):
        emit(
            f"  {i+1:>2}  {label:<32}  {p:>10.6f}  "
            f"{'Ya' if bonf_r else 'Tidak':>6}  {'Ya' if bh_r else 'Tidak':>6}"
        )
    emit()

    t_elapsed = time.time() - t_start
    emit(f"  Total waktu eksekusi: {t_elapsed:.1f}s ({t_elapsed/60:.1f} menit)")
    emit()

    # Simpan laporan ke file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    print(f"\nLaporan tersimpan di: {REPORT_PATH}")


if __name__ == "__main__":
    main()
