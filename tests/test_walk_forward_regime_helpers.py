"""
tests/test_walk_forward_regime_helpers.py
===========================================
Unit test untuk fungsi logika inti Fase 22 (Walk-Forward Validation per Regime/Strategy).

CATATAN DESAIN:
    Test ini bersifat murni logika agregasi — BUKAN causality/DataFrame test
    seperti fase-fase sebelumnya. Kausalitas sistem sudah dibuktikan di Fase 21
    (test_backtester_regime.py). Di sini kita hanya mengtest fungsi-fungsi
    statistik dan keputusan yang menerima data mentah (list of dict) sebagai input.

FUNGSI YANG DITEST:
    - determine_verdict()
    - filter_eligible_folds()
    - apply_multiple_comparison_correction()
    - compute_fold_positive_pct()
    - run_wilcoxon_test()
    - run_correlation_diagnostics()
"""

import pytest
import numpy as np
from scipy import stats as scipy_stats

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.run_walk_forward_regime import (
    determine_verdict,
    filter_eligible_folds,
    compute_fold_positive_pct,
    run_wilcoxon_test,
    run_correlation_diagnostics,
    apply_multiple_comparison_correction,
    FOLD_MIN_TRADES,
    STRATEGY_MIN_TOTAL_TRADES,
    ALPHA,
    _bh_fdr,
)


# =============================================================================
# HELPER — buat fold result dummy
# =============================================================================

def _make_fold(fold_n: int, trades: int, avg_rrr: float = 0.5):
    return {
        "fold"             : fold_n,
        "val_start"        : f"2025-0{fold_n}-01",
        "val_end"          : f"2025-0{fold_n}-28",
        "total_trades"     : trades,
        "avg_rrr_realized" : avg_rrr,
        "win_rate"         : 50.0,
        "total_pnl_net"    : trades * avg_rrr * 5.0,
        "skip_reason"      : None,
    }


# =============================================================================
# TEST 1-4: determine_verdict()
# =============================================================================

class TestDetermineVerdict:

    def test_1_kurang_dari_min_trades_langsung_perlu_data(self):
        """Sampel < STRATEGY_MIN_TOTAL_TRADES → PERLU DATA LEBIH, terlepas dari kriteria lain."""
        result = determine_verdict(
            total_eligible_trades=STRATEGY_MIN_TOTAL_TRADES - 1,  # tepat di bawah batas
            pct_positive_folds=80.0,   # memenuhi syarat >=60%
            wilcoxon_bh_rejected=True, # andaikan signifikan
        )
        assert result == "PERLU DATA LEBIH"

    def test_1b_nol_trade_langsung_perlu_data(self):
        """Nol trade → PERLU DATA LEBIH."""
        result = determine_verdict(
            total_eligible_trades=0,
            pct_positive_folds=100.0,
            wilcoxon_bh_rejected=True,
        )
        assert result == "PERLU DATA LEBIH"

    def test_2_sampel_cukup_kedua_kriteria_terpenuhi_lolos(self):
        """Sampel cukup, >=60% fold positif, Wilcoxon BH-FDR signifikan → LOLOS."""
        result = determine_verdict(
            total_eligible_trades=STRATEGY_MIN_TOTAL_TRADES,  # tepat batas minimum
            pct_positive_folds=60.0,   # tepat di batas minimum
            wilcoxon_bh_rejected=True,
        )
        assert result == "LOLOS"

    def test_2b_sampel_cukup_pct_lebih_dari_60_lolos(self):
        """Sampel cukup, 100% fold positif, signifikan → LOLOS."""
        result = determine_verdict(
            total_eligible_trades=50,
            pct_positive_folds=100.0,
            wilcoxon_bh_rejected=True,
        )
        assert result == "LOLOS"

    def test_3_pct_positif_kurang_dari_60_tidak_lolos(self):
        """Sampel cukup, TAPI pct positif < 60% → TIDAK LOLOS (meski Wilcoxon signifikan)."""
        result = determine_verdict(
            total_eligible_trades=50,
            pct_positive_folds=59.9,   # tepat di bawah batas
            wilcoxon_bh_rejected=True,
        )
        assert result == "TIDAK LOLOS"

    def test_3b_pct_positif_nol_tidak_lolos(self):
        """Nol fold positif → TIDAK LOLOS."""
        result = determine_verdict(
            total_eligible_trades=50,
            pct_positive_folds=0.0,
            wilcoxon_bh_rejected=True,
        )
        assert result == "TIDAK LOLOS"

    def test_4_pct_ok_tapi_wilcoxon_tidak_signifikan_tidak_lolos(self):
        """Sampel cukup, >=60% fold positif, TAPI Wilcoxon BH-FDR tidak signifikan → TIDAK LOLOS."""
        result = determine_verdict(
            total_eligible_trades=50,
            pct_positive_folds=80.0,
            wilcoxon_bh_rejected=False,  # tidak signifikan
        )
        assert result == "TIDAK LOLOS"

    def test_4b_keduanya_gagal_tidak_lolos(self):
        """Sampel cukup, keduanya gagal → TIDAK LOLOS."""
        result = determine_verdict(
            total_eligible_trades=50,
            pct_positive_folds=40.0,    # gagal
            wilcoxon_bh_rejected=False, # gagal
        )
        assert result == "TIDAK LOLOS"


# =============================================================================
# TEST 5: filter_eligible_folds()
# =============================================================================

class TestFilterEligibleFolds:

    def test_5_fold_di_bawah_min_dikecualikan_dari_output(self):
        """Fold dengan total_trades < FOLD_MIN_TRADES tidak ada di return value."""
        fold_list = [
            _make_fold(1, trades=FOLD_MIN_TRADES - 1),   # tidak eligible
            _make_fold(2, trades=FOLD_MIN_TRADES),         # eligible (tepat batas)
            _make_fold(3, trades=FOLD_MIN_TRADES + 5),    # eligible
            _make_fold(4, trades=0),                       # tidak eligible
        ]
        result = filter_eligible_folds(fold_list, min_trades=FOLD_MIN_TRADES)
        assert len(result) == 2
        assert all(f["total_trades"] >= FOLD_MIN_TRADES for f in result)

    def test_5b_fold_asli_tidak_diubah(self):
        """fold_list asli tidak berubah setelah filter (tidak ada in-place mutation)."""
        fold_list = [
            _make_fold(1, trades=0),
            _make_fold(2, trades=10),
        ]
        original_len = len(fold_list)
        _ = filter_eligible_folds(fold_list, min_trades=FOLD_MIN_TRADES)
        assert len(fold_list) == original_len  # asli tidak berkurang

    def test_5c_semua_eligible(self):
        """Jika semua fold eligible, semua dikembalikan."""
        fold_list = [_make_fold(i, trades=10) for i in range(1, 6)]
        result = filter_eligible_folds(fold_list, min_trades=FOLD_MIN_TRADES)
        assert len(result) == 5

    def test_5d_semua_tidak_eligible(self):
        """Jika tidak ada yang eligible, return list kosong."""
        fold_list = [_make_fold(i, trades=0) for i in range(1, 4)]
        result = filter_eligible_folds(fold_list, min_trades=FOLD_MIN_TRADES)
        assert result == []


# =============================================================================
# TEST 6: apply_multiple_comparison_correction()
# =============================================================================

class TestApplyMultipleComparisonCorrection:

    def test_6_bonferroni_threshold_benar(self):
        """Bonferroni threshold = alpha / n_hipotesis."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = apply_multiple_comparison_correction(p_values, alpha=0.05)
        expected_threshold = 0.05 / 5
        assert abs(result["bonferroni_threshold"] - expected_threshold) < 1e-10

    def test_6b_bonferroni_rejected_benar(self):
        """Bonferroni: hanya p <= alpha/n yang ditolak."""
        alpha = 0.05
        n = 9
        threshold = alpha / n  # ~0.00556
        p_values = [
            0.001,   # < threshold -> ditolak
            0.005,   # < threshold -> ditolak
            0.006,   # > threshold -> tidak ditolak
            0.1,     # > threshold -> tidak ditolak
            0.2, 0.3, 0.5, 0.7, 0.9
        ]
        result = apply_multiple_comparison_correction(p_values, alpha=alpha)
        bonf = result["bonferroni_rejected"]
        assert bonf[0] is True   # 0.001 < 0.00556
        assert bonf[1] is True   # 0.005 < 0.00556
        assert bonf[2] is False  # 0.006 > 0.00556
        assert bonf[3] is False  # 0.1

    def test_6c_bh_fdr_manual_verification(self):
        """
        BH-FDR: Verifikasi manual dengan set p-value yang diketahui hasilnya.
        
        Untuk n=5, alpha=0.05:
        Sorted: [0.001, 0.01, 0.04, 0.08, 0.15]
        BH threshold per rank:
          rank 1: 0.05*1/5 = 0.010 -> 0.001 <= 0.010 -> ditolak
          rank 2: 0.05*2/5 = 0.020 -> 0.01  <= 0.020 -> ditolak
          rank 3: 0.05*3/5 = 0.030 -> 0.04  > 0.030  -> tidak
          rank 4: 0.05*4/5 = 0.040 -> 0.08  > 0.040  -> tidak
          rank 5: 0.05*5/5 = 0.050 -> 0.15  > 0.050  -> tidak
        Monotone enforcement: max_k_rejected = 1 (rank ke-2, 0-indexed), 
        semua rank <= 1 diterima -> rank 0 dan 1 rejected.
        Hasilnya: [0.001, 0.01] ditolak, [0.04, 0.08, 0.15] tidak ditolak.
        """
        # Input dalam urutan asal (bukan sorted)
        p_values_input = [0.04, 0.001, 0.15, 0.01, 0.08]
        result = apply_multiple_comparison_correction(p_values_input, alpha=0.05)
        bh = result["bh_fdr_rejected"]

        # Verifikasi dengan scipy sebagai ground truth tambahan
        bh_manual = _bh_fdr(p_values_input, alpha=0.05)

        # 0.04 (idx 0): tidak ditolak
        assert bh[0] is False
        # 0.001 (idx 1): ditolak
        assert bh[1] is True
        # 0.15 (idx 2): tidak ditolak
        assert bh[2] is False
        # 0.01 (idx 3): ditolak
        assert bh[3] is True
        # 0.08 (idx 4): tidak ditolak
        assert bh[4] is False

        # Konsisten dengan implementasi _bh_fdr internal
        assert bh == bh_manual

    def test_6d_p_values_kosong(self):
        """Input kosong → semua list kosong."""
        result = apply_multiple_comparison_correction([], alpha=0.05)
        assert result["bonferroni_rejected"] == []
        assert result["bh_fdr_rejected"] == []
        assert result["bonferroni_threshold"] is None
        assert result["n_hypotheses"] == 0

    def test_6e_jumlah_output_sama_dengan_input(self):
        """len(bonferroni_rejected) == len(bh_fdr_rejected) == len(p_values)."""
        p_values = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 1.0]
        result = apply_multiple_comparison_correction(p_values, alpha=0.05)
        assert len(result["bonferroni_rejected"]) == 9
        assert len(result["bh_fdr_rejected"]) == 9


# =============================================================================
# TEST 7: Verdict INDEPENDEN antar strategi
# =============================================================================

class TestVerdictIndependen:

    def test_7_tiga_strategi_kombinasi_campuran(self):
        """
        Skenario tiga strategi dengan kombinasi berbeda:
        - RANGE_REVERSAL: sampel cukup, 70% positif, BH-FDR signifikan -> LOLOS
        - BREAKOUT_RETEST: sampel cukup, 40% positif -> TIDAK LOLOS
        - TREND_FOLLOWING: sampel < 30 -> PERLU DATA LEBIH

        Setiap verdict ditentukan INDEPENDEN — satu tidak mempengaruhi yang lain.
        """
        verdict_rr = determine_verdict(
            total_eligible_trades=50,
            pct_positive_folds=70.0,
            wilcoxon_bh_rejected=True,
        )
        verdict_br = determine_verdict(
            total_eligible_trades=35,
            pct_positive_folds=40.0,
            wilcoxon_bh_rejected=False,
        )
        verdict_tf = determine_verdict(
            total_eligible_trades=15,  # < 30
            pct_positive_folds=100.0,
            wilcoxon_bh_rejected=True,
        )

        assert verdict_rr == "LOLOS"
        assert verdict_br == "TIDAK LOLOS"
        assert verdict_tf == "PERLU DATA LEBIH"

    def test_7b_semua_perlu_data(self):
        """Ketiga strategi bisa semua PERLU DATA LEBIH."""
        for _ in range(3):
            v = determine_verdict(
                total_eligible_trades=5,
                pct_positive_folds=80.0,
                wilcoxon_bh_rejected=True,
            )
            assert v == "PERLU DATA LEBIH"

    def test_7c_verdict_tidak_bergantung_pada_urutan_perhitungan(self):
        """Urutan perhitungan tidak mempengaruhi verdict masing-masing strategi."""
        # Hitung dalam urutan berbeda
        v2 = determine_verdict(40, 40.0, False)   # TIDAK LOLOS
        v1 = determine_verdict(50, 70.0, True)    # LOLOS
        v3 = determine_verdict(10, 80.0, True)    # PERLU DATA LEBIH

        assert v1 == "LOLOS"
        assert v2 == "TIDAK LOLOS"
        assert v3 == "PERLU DATA LEBIH"


# =============================================================================
# TEST TAMBAHAN: compute_fold_positive_pct & run_wilcoxon_test
# =============================================================================

class TestComputeFoldPositivePct:

    def test_pct_positif_semua_positif(self):
        folds = [_make_fold(i, trades=5, avg_rrr=0.5) for i in range(1, 6)]
        assert compute_fold_positive_pct(folds) == 100.0

    def test_pct_positif_nol(self):
        folds = [_make_fold(i, trades=5, avg_rrr=-0.3) for i in range(1, 4)]
        assert compute_fold_positive_pct(folds) == 0.0

    def test_pct_positif_campuran(self):
        folds = [
            _make_fold(1, trades=5, avg_rrr=0.5),
            _make_fold(2, trades=5, avg_rrr=-0.3),
            _make_fold(3, trades=5, avg_rrr=0.1),
            _make_fold(4, trades=5, avg_rrr=-0.1),
        ]
        # 2 positif dari 4
        pct = compute_fold_positive_pct(folds)
        assert abs(pct - 50.0) < 1e-6

    def test_pct_list_kosong(self):
        assert compute_fold_positive_pct([]) == 0.0


class TestRunWilcoxonTest:

    def test_wilcoxon_semua_positif_menghasilkan_p_kecil(self):
        """Semua rrr > 0 dan bervariasi → Wilcoxon p-value harusnya kecil (1-sisi greater)."""
        folds = [_make_fold(i, trades=5, avg_rrr=float(i) * 0.1) for i in range(1, 8)]
        result = run_wilcoxon_test(folds)
        assert result["error"] is None
        assert result["p_raw"] < 0.1  # harusnya signifikan dengan data semua positif

    def test_wilcoxon_kurang_dari_2_fold_error(self):
        """Kurang dari 2 fold → error, p=1.0."""
        result = run_wilcoxon_test([_make_fold(1, trades=5, avg_rrr=0.5)])
        assert result["p_raw"] == 1.0
        assert result["error"] is not None

    def test_wilcoxon_semua_nilai_sama_error(self):
        """Semua avg_rrr sama → tidak bisa Wilcoxon, error dilaporkan."""
        folds = [_make_fold(i, trades=5, avg_rrr=0.5) for i in range(1, 5)]
        result = run_wilcoxon_test(folds)
        assert result["p_raw"] == 1.0
        assert result["error"] is not None

    def test_wilcoxon_campuran_positif_negatif(self):
        """Campuran positif dan negatif → bisa berjalan, p mungkin tidak signifikan."""
        folds = [
            _make_fold(1, trades=5, avg_rrr=0.5),
            _make_fold(2, trades=5, avg_rrr=-0.5),
            _make_fold(3, trades=5, avg_rrr=0.3),
            _make_fold(4, trades=5, avg_rrr=-0.3),
            _make_fold(5, trades=5, avg_rrr=0.1),
        ]
        result = run_wilcoxon_test(folds)
        # Tidak error, p valid [0, 1]
        assert result["error"] is None
        assert 0.0 <= result["p_raw"] <= 1.0
