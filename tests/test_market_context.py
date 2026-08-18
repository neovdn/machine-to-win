"""
tests/test_market_context.py
=============================
Unit & Causality Test untuk engine/market_context.py — Fase 12 H1 Context Layer.

PENGUJIAN (9 kelompok, semua data sintetis):

  1. test_bullish_strong:
     Bias BULLISH dengan strength STRONG (gap >= 0.15%).

  2. test_bullish_moderate:
     Bias BULLISH dengan strength MODERATE (0.05% <= gap < 0.15%).

  3. test_bullish_weak:
     Bias BULLISH dengan strength WEAK (gap < 0.05%).

  4. test_bearish:
     Bias BEARISH (EMA9 < EMA21, close < EMA21, gap cukup besar).

  5. test_neutral_sideways:
     Bias NEUTRAL dari kondisi SIDEWAYS H1.

  6. test_idx_out_of_range:
     idx di luar range (idx=999 pada DataFrame 10 baris) → return context kosong,
     TIDAK BOLEH crash/raise exception.

  7. test_empty_dataframe:
     DataFrame kosong (0 baris) → return context kosong, TIDAK BOLEH crash.

  8. test_negative_idx_identical_to_absolute:
     idx=-1 menghasilkan hasil identik dengan idx=n-1 (posisi absolut yang sama).

  9. test_causality_no_lookahead:
     Mutasi ekstrem candle setelah idx → hasil 100% identik dengan sebelum mutasi.
     Ini membuktikan zero look-ahead secara empiris, bukan sekadar asumsi.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Tambahkan root directory ke sys.path (pola sama dengan test_zone_detector.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.market_context import (
    get_h1_context,
    get_h1_context_from_precomputed,
    H1_STRENGTH_STRONG_THRESHOLD,
    H1_STRENGTH_MODERATE_THRESHOLD,
)


# =============================================================================
# HELPER: Pembuatan DataFrame H1 Sintetis
# =============================================================================

def _make_h1_df(
    n: int = 30,
    base_price: float = 3000.0,
    ema9_offset: float = 0.0,
    ema21_offset: float = 0.0,
    close_offset: float = 0.0,
) -> pd.DataFrame:
    """
    Buat DataFrame H1 sintetis dengan n baris.

    Untuk mengontrol bias dan gap dengan tepat, kita langsung set kolom
    ema_9, ema_21, close sebagai nilai konstan yang sudah kita tentukan.
    Ini menghindari ketergantungan pada proses kalkulasi EMA dari data
    OHLC mentah (yang membutuhkan warmup period), sehingga skenario test
    lebih deterministik dan mudah diverifikasi manual.

    Parameter:
        n            : Jumlah baris.
        base_price   : Harga dasar untuk close dan EMA.
        ema9_offset  : Offset EMA9 dari base_price (positif = EMA9 di atas base).
        ema21_offset : Offset EMA21 dari base_price.
        close_offset : Offset close dari base_price.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="1H", tz="UTC")
    close_val = base_price + close_offset
    ema9_val  = base_price + ema9_offset
    ema21_val = base_price + ema21_offset

    return pd.DataFrame({
        "open"    : base_price,
        "high"    : base_price + 1.0,
        "low"     : base_price - 1.0,
        "close"   : close_val,
        "ema_9"   : ema9_val,
        "ema_21"  : ema21_val,
        "atr_14"  : 5.0,
    }, index=dates)


# =============================================================================
# TEST SUITE
# =============================================================================

class TestMarketContext(unittest.TestCase):
    """Test suite untuk get_h1_context()."""

    # ─── Test 1: BULLISH STRONG ───────────────────────────────────────────────

    def test_bullish_strong(self):
        """
        Skenario BULLISH STRONG: EMA9 jauh di atas EMA21, close di atas EMA21.

        Konstruksi:
            base_price = 3000.0
            ema21 = 3000.0  (base)
            ema9  = 3000.0 + (0.20% * 3000.0) = 3000.0 + 6.0 = 3006.0
            close = 3004.0  (di atas ema21)

            gap_pct = (ema9 - ema21) / ema21 * 100
                    = (3006.0 - 3000.0) / 3000.0 * 100
                    = 6.0 / 3000.0 * 100
                    = 0.20%

            0.20% >= H1_STRENGTH_STRONG_THRESHOLD (0.15%) → STRONG
            EMA9 > EMA21 DAN close > EMA21 → UPTREND → BULLISH
        """
        ema21_val = 3000.0
        gap_target_pct = 0.20  # 0.20% > 0.15% threshold STRONG
        ema9_val  = ema21_val * (1 + gap_target_pct / 100)
        close_val = ema21_val + 4.0  # di atas ema21

        df = _make_h1_df(
            n=10,
            base_price=ema21_val,
            ema9_offset=ema9_val - ema21_val,
            ema21_offset=0.0,
            close_offset=close_val - ema21_val,
        )

        ctx = get_h1_context(df, idx=-1)

        self.assertEqual(ctx["bias"], "BULLISH")
        self.assertEqual(ctx["strength_zone"], "STRONG")
        self.assertGreaterEqual(ctx["strength"], H1_STRENGTH_STRONG_THRESHOLD)
        self.assertGreater(ctx["ema_gap_pct"], 0)  # positif untuk bullish
        self.assertIsNotNone(ctx["close"])
        self.assertIsNotNone(ctx["time"])
        self.assertIn("BULLISH", ctx["keterangan"])
        self.assertIn("STRONG", ctx["keterangan"])

    # ─── Test 2: BULLISH MODERATE ─────────────────────────────────────────────

    def test_bullish_moderate(self):
        """
        Skenario BULLISH MODERATE: gap EMA antara 0.05% dan 0.15%.

        Konstruksi:
            gap_target = 0.10%  (di antara MODERATE_THRESHOLD dan STRONG_THRESHOLD)
            ema9 = ema21 * (1 + 0.10/100)
            close di atas ema21

            Ekspektasi:
                bias = "BULLISH"
                strength_zone = "MODERATE"
                strength >= 0.05 dan < 0.15
        """
        ema21_val = 3000.0
        gap_target_pct = 0.10  # antara 0.05% dan 0.15%
        ema9_val  = ema21_val * (1 + gap_target_pct / 100)
        close_val = ema21_val + 2.0

        df = _make_h1_df(
            n=10,
            base_price=ema21_val,
            ema9_offset=ema9_val - ema21_val,
            ema21_offset=0.0,
            close_offset=close_val - ema21_val,
        )

        ctx = get_h1_context(df, idx=-1)

        self.assertEqual(ctx["bias"], "BULLISH")
        self.assertEqual(ctx["strength_zone"], "MODERATE")
        self.assertGreaterEqual(ctx["strength"], H1_STRENGTH_MODERATE_THRESHOLD)
        self.assertLess(ctx["strength"], H1_STRENGTH_STRONG_THRESHOLD)
        self.assertIn("BULLISH", ctx["keterangan"])
        self.assertIn("MODERATE", ctx["keterangan"])

    # ─── Test 3: BULLISH WEAK ─────────────────────────────────────────────────

    def test_bullish_weak(self):
        """
        Skenario BULLISH WEAK: gap EMA < 0.05% tapi masih UPTREND.

        Konstruksi:
            gap_target = 0.03%  (di bawah MODERATE_THRESHOLD 0.05%)
            ema9 sedikit di atas ema21, close sedikit di atas ema21

            Karena min_ema_gap_pct=0.0 (default), bias UPTREND tetap muncul
            meski gap kecil. Ini membedakan detect_bias_h1() dari detect_trend().

            Ekspektasi:
                bias = "BULLISH"
                strength_zone = "WEAK"
                strength < 0.05
        """
        ema21_val = 3000.0
        gap_target_pct = 0.03  # di bawah threshold MODERATE
        ema9_val  = ema21_val * (1 + gap_target_pct / 100)
        close_val = ema21_val + 0.5  # sedikit di atas ema21

        df = _make_h1_df(
            n=10,
            base_price=ema21_val,
            ema9_offset=ema9_val - ema21_val,
            ema21_offset=0.0,
            close_offset=close_val - ema21_val,
        )

        ctx = get_h1_context(df, idx=-1)

        self.assertEqual(ctx["bias"], "BULLISH")
        self.assertEqual(ctx["strength_zone"], "WEAK")
        self.assertLess(ctx["strength"], H1_STRENGTH_MODERATE_THRESHOLD)
        self.assertGreater(ctx["ema_gap_pct"], 0)
        self.assertIn("BULLISH", ctx["keterangan"])
        self.assertIn("WEAK", ctx["keterangan"])

    # ─── Test 4: BEARISH ──────────────────────────────────────────────────────

    def test_bearish(self):
        """
        Skenario BEARISH: EMA9 di bawah EMA21, close di bawah EMA21.

        Konstruksi:
            gap_target = -0.20% (negatif → bearish)
            ema9 = ema21 * (1 - 0.20/100)  → ema9 di bawah ema21
            close di bawah ema21

            Ekspektasi:
                bias = "BEARISH"
                ema_gap_pct < 0 (negatif, karena ema9 < ema21)
                strength = abs(ema_gap_pct) = 0.20% → STRONG
                strength_zone = "STRONG"
        """
        ema21_val = 3000.0
        gap_target_pct = -0.20  # negatif → ema9 di bawah ema21
        ema9_val  = ema21_val * (1 + gap_target_pct / 100)
        close_val = ema21_val - 5.0  # di bawah ema21

        df = _make_h1_df(
            n=10,
            base_price=ema21_val,
            ema9_offset=ema9_val - ema21_val,
            ema21_offset=0.0,
            close_offset=close_val - ema21_val,
        )

        ctx = get_h1_context(df, idx=-1)

        self.assertEqual(ctx["bias"], "BEARISH")
        self.assertEqual(ctx["strength_zone"], "STRONG")
        self.assertGreaterEqual(ctx["strength"], H1_STRENGTH_STRONG_THRESHOLD)
        self.assertLess(ctx["ema_gap_pct"], 0)  # negatif untuk bearish
        self.assertIn("BEARISH", ctx["keterangan"])

    # ─── Test 5: NEUTRAL (SIDEWAYS) ───────────────────────────────────────────

    def test_neutral_sideways(self):
        """
        Skenario NEUTRAL: close di antara EMA9 dan EMA21 (tidak ada bias).

        Konstruksi:
            ema21 = 3000.0
            ema9  = 3006.0  (ema9 > ema21, tapi close di bawah ema21)
            close = 2995.0  (di bawah ema21)

            Logika detect_bias_h1():
            - EMA9 > EMA21 → syarat UPTREND (kondisi 1) ✓
            - close < EMA21 → syarat UPTREND (kondisi 2) ✗
            - Tidak memenuhi UPTREND maupun DOWNTREND → SIDEWAYS

            Ekspektasi: bias = "NEUTRAL"
        """
        ema21_val = 3000.0
        ema9_val  = 3006.0   # ema9 > ema21 (seolah bullish)
        close_val = 2995.0   # tapi close DI BAWAH ema21 → tidak konsisten → SIDEWAYS

        df = _make_h1_df(
            n=10,
            base_price=ema21_val,
            ema9_offset=ema9_val - ema21_val,
            ema21_offset=0.0,
            close_offset=close_val - ema21_val,
        )

        ctx = get_h1_context(df, idx=-1)

        self.assertEqual(ctx["bias"], "NEUTRAL")
        self.assertIn("NEUTRAL", ctx["keterangan"])
        # strength dan strength_zone tetap ada meski NEUTRAL
        self.assertIsNotNone(ctx["strength"])
        self.assertIsNotNone(ctx["strength_zone"])

    # ─── Test 6: idx di luar range ────────────────────────────────────────────

    def test_idx_out_of_range(self):
        """
        idx=999 pada DataFrame 10 baris → return context kosong tanpa crash.

        Setelah normalisasi (idx positif sudah), 999 >= 10 → di luar range.
        Fungsi HARUS return _empty_context() tanpa raise exception apapun.
        """
        df = _make_h1_df(n=10)

        # Tidak boleh raise exception
        ctx = get_h1_context(df, idx=999)

        self.assertEqual(ctx["bias"], "NEUTRAL")
        self.assertIsNone(ctx["strength"])
        self.assertIsNone(ctx["strength_zone"])
        self.assertIsNone(ctx["ema_gap_pct"])
        self.assertIsNone(ctx["close"])
        self.assertIsNone(ctx["time"])
        self.assertIsNotNone(ctx["keterangan"])  # ada penjelasan
        self.assertIn("999", ctx["keterangan"])  # keterangan menyebut idx bermasalah

    # ─── Test 7: DataFrame kosong ─────────────────────────────────────────────

    def test_empty_dataframe(self):
        """
        DataFrame dengan 0 baris → return context kosong tanpa crash.

        Fungsi HARUS return _empty_context() tanpa raise exception apapun,
        bahkan dengan idx default -1.
        """
        df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "ema_9", "ema_21"])

        ctx = get_h1_context(df_empty, idx=-1)

        self.assertEqual(ctx["bias"], "NEUTRAL")
        self.assertIsNone(ctx["strength"])
        self.assertIsNone(ctx["strength_zone"])
        self.assertIsNone(ctx["ema_gap_pct"])
        self.assertIsNone(ctx["close"])
        self.assertIsNone(ctx["time"])
        self.assertIsNotNone(ctx["keterangan"])

    # ─── Test 8: idx negatif identik dengan idx absolut ──────────────────────

    def test_negative_idx_identical_to_absolute(self):
        """
        get_h1_context(df, idx=-1) harus menghasilkan dict yang IDENTIK
        dengan get_h1_context(df, idx=n-1).

        Ini membuktikan normalisasi idx negatif berjalan benar.
        Semua field dibandingkan (kecuali 'time' yang dibanding via equality,
        bukan assertAlmostEqual, karena timestamp bukan float).
        """
        n = 15
        df = _make_h1_df(n=n, ema9_offset=6.0, close_offset=4.0)  # BULLISH STRONG

        ctx_neg = get_h1_context(df, idx=-1)
        ctx_abs = get_h1_context(df, idx=n - 1)

        # Semua field harus identik persis
        self.assertEqual(ctx_neg["bias"],          ctx_abs["bias"])
        self.assertEqual(ctx_neg["strength_zone"], ctx_abs["strength_zone"])
        self.assertEqual(ctx_neg["ema_gap_pct"],   ctx_abs["ema_gap_pct"])
        self.assertEqual(ctx_neg["close"],         ctx_abs["close"])
        self.assertEqual(ctx_neg["time"],          ctx_abs["time"])
        self.assertEqual(ctx_neg["keterangan"],    ctx_abs["keterangan"])
        self.assertAlmostEqual(ctx_neg["strength"], ctx_abs["strength"], places=10)

    # ─── Test 9: Kausalitas (WAJIB — zero look-ahead) ────────────────────────

    def test_causality_no_lookahead(self):
        """
        Mutasi EKSTREM candle setelah idx TIDAK BOLEH mengubah hasil apapun.

        Ini adalah bukti empiris zero look-ahead, bukan sekadar asumsi.
        Mengikuti pola test_causality_no_lookahead di test_zone_detector.py
        dan causality test di test_supply_demand.py.

        Prosedur:
          1. Buat DataFrame sintetis 35 baris dengan kondisi BULLISH STRONG.
          2. Pilih t = 17 (tengah data, ada cukup data di kiri dan kanan).
          3. Panggil get_h1_context(df, idx=t) → simpan sebagai result_original.
          4. Copy df, mutasi EKSTREM semua baris setelah t:
               - close  → 99999.0   (naik ekstrem)
               - ema_9  → 50000.0   (naik ekstrem)
               - ema_21 → 1.0       (turun ekstrem)
             Ini akan mengubah bias menjadi BULLISH dengan gap sangat besar
             jika fungsi membaca data setelah t.
          5. Panggil get_h1_context(df_mutated, idx=t) → simpan sebagai result_mutated.
          6. Assert semua field identik: bias, strength, strength_zone,
             ema_gap_pct, close, keterangan.

        Jika fungsi causal: baris setelah t tidak dibaca → hasil identik.
        Jika fungsi TIDAK causal: mutasi baris t+1 dst akan mengubah
            ema_gap_pct_h1_raw[t] (karena kalkulasi EMA sebelum t
            dipengaruhi oleh nilai setelah t) → hasil BERBEDA → test GAGAL.

        CATATAN PENTING tentang detect_bias_h1() dan kausalitas EMA:
            calculate_ema() di indicators.py menggunakan pandas .ewm(adjust=False)
            yang bersifat rekursif murni: nilai EMA di baris t hanya bergantung
            pada nilai EMA di baris t-1 dan close di baris t. Nilai EMA di
            baris setelah t TIDAK mempengaruhi EMA di baris t atau sebelumnya.
            Kausalitas ini dijamin oleh pandas dan dibuktikan oleh test ini.
        """
        n = 35
        t = 17  # titik evaluasi di tengah data

        # ── Bangun DataFrame dengan kondisi BULLISH STRONG yang konsisten ────
        # Kita set ema9_offset dan close_offset langsung agar hasilnya jelas.
        ema21_base = 3000.0
        gap_pct    = 0.20   # 0.20% → STRONG
        ema9_delta = ema21_base * (gap_pct / 100)   # 6.0
        close_delta = 4.0   # close di atas ema21

        df = _make_h1_df(
            n=n,
            base_price=ema21_base,
            ema9_offset=ema9_delta,
            ema21_offset=0.0,
            close_offset=close_delta,
        )

        # ── Evaluasi sebelum mutasi ──────────────────────────────────────────
        result_original = get_h1_context(df.copy(), idx=t)

        # ── Mutasi EKSTREM semua baris SETELAH t ────────────────────────────
        df_mutated = df.copy()
        for col, nilai_ekstrem in [("close", 99999.0), ("ema_9", 50000.0), ("ema_21", 1.0)]:
            df_mutated.iloc[t + 1:, df_mutated.columns.get_loc(col)] = nilai_ekstrem

        # ── Evaluasi setelah mutasi ──────────────────────────────────────────
        result_mutated = get_h1_context(df_mutated, idx=t)

        # ── Semua field harus 100% identik ──────────────────────────────────
        self.assertEqual(
            result_original["bias"],
            result_mutated["bias"],
            msg="KAUSALITAS GAGAL: 'bias' berubah setelah mutasi data masa depan",
        )
        self.assertAlmostEqual(
            result_original["strength"],
            result_mutated["strength"],
            places=10,
            msg="KAUSALITAS GAGAL: 'strength' berubah setelah mutasi data masa depan",
        )
        self.assertEqual(
            result_original["strength_zone"],
            result_mutated["strength_zone"],
            msg="KAUSALITAS GAGAL: 'strength_zone' berubah setelah mutasi data masa depan",
        )
        self.assertAlmostEqual(
            result_original["ema_gap_pct"],
            result_mutated["ema_gap_pct"],
            places=10,
            msg="KAUSALITAS GAGAL: 'ema_gap_pct' berubah setelah mutasi data masa depan",
        )
        self.assertAlmostEqual(
            result_original["close"],
            result_mutated["close"],
            places=10,
            msg="KAUSALITAS GAGAL: 'close' berubah setelah mutasi data masa depan",
        )
        self.assertEqual(
            result_original["keterangan"],
            result_mutated["keterangan"],
            msg="KAUSALITAS GAGAL: 'keterangan' berubah setelah mutasi data masa depan",
        )


# =============================================================================
# TEST SUITE: get_h1_context_from_precomputed (varian efisiensi backtest)
# =============================================================================

class TestMarketContextPrecomputed(unittest.TestCase):
    """
    Test suite untuk get_h1_context_from_precomputed().

    Membuktikan bahwa varian precomputed menghasilkan output yang IDENTIK
    dengan get_h1_context() untuk kasus normal, dan menangani edge case
    dengan benar.
    """

    def test_precomputed_identical_to_main(self):
        """
        get_h1_context_from_precomputed() harus menghasilkan dict identik
        dengan get_h1_context() untuk input yang sama.
        """
        from engine.indicators import detect_bias_h1

        df = _make_h1_df(n=10, ema9_offset=6.0, close_offset=4.0)

        # Hitung precomputed
        df_with_bias = detect_bias_h1(df.copy(), min_ema_gap_pct=0.0)

        ctx_main        = get_h1_context(df, idx=-1)
        ctx_precomputed = get_h1_context_from_precomputed(df_with_bias, idx=-1)

        self.assertEqual(ctx_main["bias"],          ctx_precomputed["bias"])
        self.assertEqual(ctx_main["strength_zone"], ctx_precomputed["strength_zone"])
        self.assertAlmostEqual(ctx_main["strength"],     ctx_precomputed["strength"],     places=10)
        self.assertAlmostEqual(ctx_main["ema_gap_pct"],  ctx_precomputed["ema_gap_pct"],  places=10)
        self.assertAlmostEqual(ctx_main["close"],        ctx_precomputed["close"],        places=10)
        self.assertEqual(ctx_main["time"],          ctx_precomputed["time"])
        self.assertEqual(ctx_main["keterangan"],    ctx_precomputed["keterangan"])

    def test_precomputed_empty_dataframe(self):
        """DataFrame kosong → return context kosong tanpa crash."""
        df_empty = pd.DataFrame(columns=["close", "bias_h1", "ema_gap_pct_h1_raw"])
        ctx = get_h1_context_from_precomputed(df_empty, idx=-1)
        self.assertEqual(ctx["bias"], "NEUTRAL")
        self.assertIsNone(ctx["strength"])

    def test_precomputed_missing_columns(self):
        """Kolom bias_h1 tidak ada → return context kosong, tidak crash."""
        df = _make_h1_df(n=10)  # tidak punya 'bias_h1'
        ctx = get_h1_context_from_precomputed(df, idx=-1)
        self.assertEqual(ctx["bias"], "NEUTRAL")
        self.assertIsNone(ctx["strength"])
        self.assertIn("bias_h1", ctx["keterangan"])


if __name__ == "__main__":
    unittest.main()
