"""
tests/test_zone_detector.py
============================
Unit test untuk engine/zone_detector.py — deteksi zona konsolidasi M5.

PENGUJIAN (6 kasus, semua data sintetis):
  1. test_clear_consolidation:
     Data flat 30 candle → is_valid=True, duration=20 (capped at lookback).
  2. test_clear_trending:
     Data trending kuat → is_valid=False (duration < min_duration).
  3. test_duration_below_minimum:
     8 candle flat (terbaru) + 12 trending (lama) → duration=8 < 10.
  4. test_insufficient_data:
     idx < lookback → is_valid=False, semua numerik None, tidak crash.
  5. test_duration_calculation:
     15 candle flat (terbaru) + 5 wild (lama) → duration=15 tepat.
  6. test_causality_no_lookahead:
     Mutasi candle setelah idx → hasil 100% identik.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Tambahkan root directory ke sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.zone_detector import detect_consolidation_zone


class TestZoneDetector(unittest.TestCase):
    """Test suite untuk detect_consolidation_zone()."""

    # ─── Helper: buat DataFrame sintetis ─────────────────────────────────────

    @staticmethod
    def _make_flat_df(n: int, base_price: float = 2000.0,
                      spread: float = 1.0, atr: float = 5.0) -> pd.DataFrame:
        """
        Buat n candle flat (konsolidasi jelas).
        high = base + spread, low = base - spread, range = 2*spread.
        """
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({
            "open": base_price,
            "high": base_price + spread,
            "low": base_price - spread,
            "close": base_price + 0.1,
            "atr_14": atr,
        }, index=dates)

    @staticmethod
    def _make_trending_df(n: int, base_price: float = 2000.0,
                          step: float = 3.0, atr: float = 2.0) -> pd.DataFrame:
        """
        Buat n candle trending naik kuat.
        Setiap candle naik `step`, sehingga range window cepat melebihi ATR.
        """
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        prices = base_price + np.arange(n) * step
        return pd.DataFrame({
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices + 0.5,
            "atr_14": atr,
        }, index=dates)

    # ─── Test 1: Konsolidasi jelas ───────────────────────────────────────────

    def test_clear_consolidation(self):
        """
        30 candle flat: high=2001, low=1999, range=2.0, ATR=5.0.
        ratio = 2.0/5.0 = 0.4 (jauh di bawah threshold 2.5).
        Duration = 20 (capped at lookback).
        Karena data flat di seluruh window, duration-window dan lookback-window
        menghasilkan resistance/support yang sama.
        """
        df = self._make_flat_df(n=30, base_price=2000.0, spread=1.0, atr=5.0)
        result = detect_consolidation_zone(df, idx=-1, lookback=20)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["duration"], 20)
        self.assertAlmostEqual(result["resistance"], 2001.0)
        self.assertAlmostEqual(result["support"], 1999.0)
        self.assertAlmostEqual(result["range_zone"], 2.0)
        self.assertAlmostEqual(result["range_atr_ratio"], 0.4)
        self.assertIn("KONSOLIDASI VALID", result["keterangan"])

    # ─── Test 2: Trending jelas ──────────────────────────────────────────────

    def test_clear_trending(self):
        """
        30 candle trending naik (step=3.0 per candle), ATR=2.0.
        Threshold = 2.5 * 2.0 = 5.0.
        Bahkan 2 candle saja: range = (base+3+1) - (base-1) = 5.0 → pas di batas.
        3 candle: range = (base+6+1) - (base-1) = 8.0 > 5.0.
        Jadi duration akan kecil (2-3), jauh di bawah min_duration=10.
        """
        df = self._make_trending_df(n=30, step=3.0, atr=2.0)
        result = detect_consolidation_zone(df, idx=-1, lookback=20)

        self.assertFalse(result["is_valid"])
        self.assertLess(result["duration"], 10)
        self.assertIn("TIDAK VALID", result["keterangan"])

    # ─── Test 3: Durasi di bawah minimum ─────────────────────────────────────

    def test_duration_below_minimum(self):
        """
        Konstruksi: 12 candle trending (lama) + 8 candle flat (terbaru).
        Ekspansi mundur dari idx: 8 candle flat valid, lalu candle ke-9 (trending)
        akan break threshold → duration=8 < min_duration=10 → is_valid=False.

        Resistance/support dihitung dari window 8-candle (flat).

        Matematis:
        - Trending: step=10, candle terakhir (idx=11): high=2111, low=2109
        - Flat: high=2131, low=2129, range flat=2
        - Threshold: 2.5 * 3.0 = 7.5
        - 8 candle flat: range=2 < 7.5 ✓
        - Tambah candle trending idx=11 (low=2109): range=2131-2109=22 > 7.5 ✗
        - → duration=8
        """
        # 12 candle trending (step besar supaya low jauh dari flat zone)
        df_trending = self._make_trending_df(n=12, base_price=2000.0, step=10.0, atr=3.0)
        # 8 candle flat (di atas ujung trending, tapi range kecil)
        dates_flat = pd.date_range(
            df_trending.index[-1] + pd.Timedelta(minutes=5),
            periods=8, freq="5min", tz="UTC"
        )
        df_flat = pd.DataFrame({
            "open": 2130.0,
            "high": 2131.0,
            "low": 2129.0,
            "close": 2130.1,
            "atr_14": 3.0,
        }, index=dates_flat)

        df = pd.concat([df_trending, df_flat]).reset_index(drop=True)

        result = detect_consolidation_zone(df, idx=-1, lookback=20)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["duration"], 8)
        # Resistance/support dari 8-candle flat window
        self.assertAlmostEqual(result["resistance"], 2131.0)
        self.assertAlmostEqual(result["support"], 2129.0)
        self.assertAlmostEqual(result["range_zone"], 2.0)
        self.assertIn("TIDAK VALID", result["keterangan"])

    # ─── Test 4: Data tidak cukup ────────────────────────────────────────────

    def test_insufficient_data(self):
        """
        idx=5 dengan lookback=20 → butuh minimal idx >= 19.
        Harus return is_valid=False, semua numerik None, tanpa crash.
        """
        df = self._make_flat_df(n=10, atr=5.0)
        result = detect_consolidation_zone(df, idx=5, lookback=20)

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["resistance"])
        self.assertIsNone(result["support"])
        self.assertIsNone(result["range_zone"])
        self.assertIsNone(result["range_atr_ratio"])
        self.assertEqual(result["duration"], 0)
        self.assertIn("Data tidak cukup", result["keterangan"])

    # ─── Test 5: Perhitungan duration tepat ──────────────────────────────────

    def test_duration_calculation(self):
        """
        Konstruksi: 5 candle wild (lama) + 15 candle flat (terbaru).
        Ekspansi mundur dari idx: 15 candle flat valid, lalu candle ke-16 (wild)
        akan break threshold → duration=15 tepat.

        Resistance/support dihitung dari window 15-candle flat.
        """
        # 5 candle wild: range per candle sangat besar
        dates_wild = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
        wild_prices = np.array([2000.0, 2050.0, 1950.0, 2080.0, 1920.0])
        df_wild = pd.DataFrame({
            "open": wild_prices,
            "high": wild_prices + 20.0,
            "low": wild_prices - 20.0,
            "close": wild_prices + 1.0,
            "atr_14": 4.0,
        }, index=dates_wild)

        # 15 candle flat
        dates_flat = pd.date_range(
            dates_wild[-1] + pd.Timedelta(minutes=5),
            periods=15, freq="5min", tz="UTC"
        )
        df_flat = pd.DataFrame({
            "open": 2000.0,
            "high": 2002.0,
            "low": 1998.0,
            "close": 2000.5,
            "atr_14": 4.0,
        }, index=dates_flat)

        df = pd.concat([df_wild, df_flat]).reset_index(drop=True)

        result = detect_consolidation_zone(df, idx=-1, lookback=20)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["duration"], 15)
        # Resistance/support dari 15-candle flat window
        self.assertAlmostEqual(result["resistance"], 2002.0)
        self.assertAlmostEqual(result["support"], 1998.0)
        self.assertAlmostEqual(result["range_zone"], 4.0)
        self.assertAlmostEqual(result["range_atr_ratio"], 1.0)  # 4.0 / 4.0

    # ─── Test 6: Kausalitas (no look-ahead) ──────────────────────────────────

    def test_causality_no_lookahead(self):
        """
        Mutasi candle setelah idx TIDAK BOLEH mengubah hasil deteksi.
        Ini membuktikan bahwa fungsi hanya baca data <= idx.
        """
        df = self._make_flat_df(n=40, base_price=2000.0, spread=1.0, atr=5.0)

        # Evaluasi di idx=29 (ada 30 candle valid sebelumnya)
        t = 29
        result_original = detect_consolidation_zone(df.copy(), idx=t, lookback=20)

        # Mutasi candle t+1 s/d t+10 secara ekstrem
        df_mutated = df.copy()
        for col, val in [("high", 9999.0), ("low", 1.0), ("close", 5000.0)]:
            df_mutated.iloc[t + 1:, df_mutated.columns.get_loc(col)] = val

        result_mutated = detect_consolidation_zone(df_mutated, idx=t, lookback=20)

        # Semua field harus identik
        self.assertEqual(result_original["is_valid"], result_mutated["is_valid"])
        self.assertEqual(result_original["duration"], result_mutated["duration"])
        self.assertAlmostEqual(result_original["resistance"], result_mutated["resistance"])
        self.assertAlmostEqual(result_original["support"], result_mutated["support"])
        self.assertAlmostEqual(result_original["range_zone"], result_mutated["range_zone"])
        self.assertAlmostEqual(result_original["range_atr_ratio"], result_mutated["range_atr_ratio"])


if __name__ == "__main__":
    unittest.main()
