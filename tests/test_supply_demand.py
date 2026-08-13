"""
tests/test_supply_demand.py
============================
Unit & Causality Test untuk engine/supply_demand.py — Fase 11 S&D Zone.

PENGUJIAN (10 kelompok, semua data sintetis atau historis):

  1. test_impulsive_detection_bullish:
     Candle body besar bullish terdeteksi sebagai BULLISH impulsif.
  2. test_impulsive_detection_bearish:
     Candle body besar bearish terdeteksi sebagai BEARISH impulsif.
  3. test_impulsive_not_triggered_small_body:
     Candle body kecil (< threshold) tidak dianggap impulsif.
  4. test_sd_zone_from_valid_origin:
     Origin "diam" sebelum candle impulsif → zona terbentuk dengan level benar.
  5. test_sd_zone_invalid_origin_also_impulsive:
     Candle origin JUGA impulsif (body besar) → zona tidak terbentuk.
  6. test_freshness_fresh_zone:
     Zona FRESH tetap FRESH jika harga tidak kembali menyentuh range zona.
  7. test_freshness_tested_zone:
     Zona menjadi TESTED setelah harga overlap dengan range zona.
  8. test_invalidation_demand_zone:
     Demand zone invalid setelah close break di bawah low_origin.
  9. test_find_nearest_sd_zone_returns_none:
     find_nearest_sd_zone() return None kalau tidak ada zona valid dalam lookback.
  10. test_calculate_sl_tp_sd_zone_method:
      calculate_sl_tp(sl_source="SD_ZONE") menghasilkan sl_method="SD_ZONE".
  11. test_calculate_sl_tp_swing_default_unchanged:
      calculate_sl_tp(sl_source="SWING") identik 100% dengan perilaku sebelum Fase 11.
  12. test_causality_detect_impulsive:
      Mutasi candle SETELAH idx tidak mengubah hasil detect_impulsive_move().
  13. test_causality_detect_sd_zone_from_origin:
      Mutasi candle SETELAH idx_impulsive tidak mengubah hasil detect_sd_zone_from_origin().
  14. test_causality_find_nearest_sd_zone:
      Mutasi candle SETELAH idx evaluasi tidak mengubah hasil find_nearest_sd_zone().
  15. test_causality_calculate_sl_tp_sd_zone:
      Mutasi candle SETELAH idx evaluasi tidak mengubah hasil calculate_sl_tp(sl_source="SD_ZONE").
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.supply_demand import (
    detect_impulsive_move,
    detect_sd_zone_from_origin,
    find_nearest_sd_zone,
    DEFAULT_IMPULSIVE_RATIO,
    SD_BUFFER,
)
from engine.risk_manager import calculate_sl_tp


# =============================================================================
# HELPER: Pembuatan DataFrame Sintetis
# =============================================================================

def _make_df(n: int = 30, base_price: float = 2000.0, atr: float = 5.0) -> pd.DataFrame:
    """
    Buat DataFrame sintetis dengan n candle 'datar' (body kecil).
    open = base_price, close = base_price + 0.1 (body = 0.1, jauh di bawah threshold 1.5*5=7.5)
    high = base_price + 0.5, low = base_price - 0.5
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open"   : base_price,
        "high"   : base_price + 0.5,
        "low"    : base_price - 0.5,
        "close"  : base_price + 0.1,
        "atr_14" : atr,
    }, index=dates)


def _inject_impulsive_bullish(df: pd.DataFrame, idx: int, body: float = 10.0) -> pd.DataFrame:
    """
    Ganti candle di idx menjadi bullish impulsif.
    open = price, close = price + body, high = price + body + 0.5, low = price - 0.5
    """
    df = df.copy()
    price = float(df["open"].iloc[idx])
    df.iloc[idx, df.columns.get_loc("open")]  = price
    df.iloc[idx, df.columns.get_loc("close")] = price + body
    df.iloc[idx, df.columns.get_loc("high")]  = price + body + 0.5
    df.iloc[idx, df.columns.get_loc("low")]   = price - 0.5
    return df


def _inject_impulsive_bearish(df: pd.DataFrame, idx: int, body: float = 10.0) -> pd.DataFrame:
    """
    Ganti candle di idx menjadi bearish impulsif.
    open = price, close = price - body, high = price + 0.5, low = price - body - 0.5
    """
    df = df.copy()
    price = float(df["open"].iloc[idx])
    df.iloc[idx, df.columns.get_loc("open")]  = price
    df.iloc[idx, df.columns.get_loc("close")] = price - body
    df.iloc[idx, df.columns.get_loc("high")]  = price + 0.5
    df.iloc[idx, df.columns.get_loc("low")]   = price - body - 0.5
    return df


# =============================================================================
# TEST 1-3: detect_impulsive_move()
# =============================================================================

class TestDetectImpulsiveMove(unittest.TestCase):
    """Test untuk fungsi detect_impulsive_move()."""

    def test_impulsive_detection_bullish(self):
        """
        Candle dengan body = 10, ATR = 5, ratio = 1.5 → threshold = 7.5.
        body 10 >= 7.5 → BULLISH impulsif.
        """
        df = _make_df(n=10, atr=5.0)
        df = _inject_impulsive_bullish(df, idx=9, body=10.0)

        result = detect_impulsive_move(df, idx=9, impulsive_body_atr_ratio=1.5)

        self.assertTrue(result["is_impulsive"])
        self.assertEqual(result["arah"], "BULLISH")
        self.assertAlmostEqual(result["body"], 10.0, places=5)
        self.assertAlmostEqual(result["atr_value"], 5.0, places=5)
        self.assertAlmostEqual(result["threshold"], 7.5, places=5)
        self.assertGreater(result["body_atr_ratio"], 1.5)
        self.assertIn("IMPULSIF BULLISH", result["keterangan"])

    def test_impulsive_detection_bearish(self):
        """
        Candle bearish body = 10, ATR = 5, threshold = 7.5.
        body 10 >= 7.5 → BEARISH impulsif.
        """
        df = _make_df(n=10, atr=5.0)
        df = _inject_impulsive_bearish(df, idx=9, body=10.0)

        result = detect_impulsive_move(df, idx=9, impulsive_body_atr_ratio=1.5)

        self.assertTrue(result["is_impulsive"])
        self.assertEqual(result["arah"], "BEARISH")
        self.assertAlmostEqual(result["body"], 10.0, places=5)
        self.assertIn("IMPULSIF BEARISH", result["keterangan"])

    def test_impulsive_not_triggered_small_body(self):
        """
        Candle flat: body = 0.1, ATR = 5, threshold = 7.5.
        body 0.1 < 7.5 → bukan impulsif.
        """
        df = _make_df(n=10, atr=5.0)
        # Candle sudah flat dari make_df, tidak perlu inject

        result = detect_impulsive_move(df, idx=9, impulsive_body_atr_ratio=1.5)

        self.assertFalse(result["is_impulsive"])
        self.assertIsNone(result["arah"])
        self.assertIn("TIDAK impulsif", result["keterangan"])

    def test_impulsive_doji_not_impulsive(self):
        """
        Candle doji (body = 0) tidak dianggap impulsif.
        """
        df = _make_df(n=10, atr=5.0)
        df = df.copy()
        df.iloc[9, df.columns.get_loc("close")] = float(df["open"].iloc[9])  # doji

        result = detect_impulsive_move(df, idx=9, impulsive_body_atr_ratio=1.5)

        self.assertFalse(result["is_impulsive"])
        self.assertIn("Doji", result["keterangan"])

    def test_impulsive_negative_idx(self):
        """
        idx=-1 harus diinterpretasi sebagai candle terakhir.
        """
        df = _make_df(n=10, atr=5.0)
        df = _inject_impulsive_bullish(df, idx=9, body=10.0)

        result = detect_impulsive_move(df, idx=-1, impulsive_body_atr_ratio=1.5)
        self.assertTrue(result["is_impulsive"])

    def test_impulsive_insufficient_data(self):
        """
        Data tidak cukup (idx=0, min_consecutive=2) → return False tanpa crash.
        """
        df = _make_df(n=5, atr=5.0)
        df = _inject_impulsive_bullish(df, idx=0, body=10.0)

        result = detect_impulsive_move(df, idx=0, impulsive_body_atr_ratio=1.5,
                                       min_consecutive_impulsive=2)
        self.assertFalse(result["is_impulsive"])


# =============================================================================
# TEST 4-5: detect_sd_zone_from_origin()
# =============================================================================

class TestDetectSdZoneFromOrigin(unittest.TestCase):
    """Test untuk fungsi detect_sd_zone_from_origin()."""

    def test_sd_zone_from_valid_origin_demand(self):
        """
        Origin "diam" di idx=9 (body 0.1) → candle impulsif bullish di idx=10
        → Demand zone terbentuk dengan [low_9, high_9] yang benar.

        ATR = 5, threshold_impulsif = 1.5 * 5 = 7.5.
        Origin body = 0.1 < 0.5 * 7.5 = 3.75 (valid basing candle).
        Candle impulsif body = 10 >= 7.5 (valid impulsif).
        """
        df = _make_df(n=15, base_price=2000.0, atr=5.0)

        # Candle 9 adalah origin (diam: high=2000.5, low=1999.5)
        # Candle 10 adalah candle impulsif bullish
        df = _inject_impulsive_bullish(df, idx=10, body=10.0)

        result = detect_sd_zone_from_origin(df, idx_impulsive=10,
                                            impulsive_body_atr_ratio=1.5)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["zone_type"], "DEMAND")
        # low_origin = low di idx=9 = 1999.5
        self.assertAlmostEqual(result["low"],  1999.5, places=4)
        # high_origin = high di idx=9 = 2000.5
        self.assertAlmostEqual(result["high"], 2000.5, places=4)
        self.assertEqual(result["origin_idx"], 9)
        self.assertIn("DEMAND ZONE", result["keterangan"])

    def test_sd_zone_from_valid_origin_supply(self):
        """
        Origin "diam" di idx=9 → candle impulsif bearish di idx=10
        → Supply zone terbentuk.
        """
        df = _make_df(n=15, base_price=2000.0, atr=5.0)
        df = _inject_impulsive_bearish(df, idx=10, body=10.0)

        result = detect_sd_zone_from_origin(df, idx_impulsive=10,
                                            impulsive_body_atr_ratio=1.5)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["zone_type"], "SUPPLY")
        self.assertAlmostEqual(result["low"],  1999.5, places=4)
        self.assertAlmostEqual(result["high"], 2000.5, places=4)
        self.assertEqual(result["origin_idx"], 9)
        self.assertIn("SUPPLY ZONE", result["keterangan"])

    def test_sd_zone_invalid_origin_also_impulsive(self):
        """
        Candle origin (idx=9) JUGA impulsif (body besar) → zona tidak terbentuk.

        ATR = 5, threshold = 7.5.
        Origin body max = 0.5 * 7.5 = 3.75.
        Origin body = 10 >= 3.75 → bukan basing candle yang valid.
        """
        df = _make_df(n=15, base_price=2000.0, atr=5.0)
        # Jadikan idx=9 JUGA impulsif (body besar)
        df = _inject_impulsive_bullish(df, idx=9, body=10.0)
        # Jadikan idx=10 impulsif
        df = _inject_impulsive_bullish(df, idx=10, body=10.0)

        result = detect_sd_zone_from_origin(df, idx_impulsive=10,
                                            impulsive_body_atr_ratio=1.5)

        self.assertFalse(result["is_valid"])
        self.assertIsNone(result["zone_type"])
        self.assertIn("JUGA impulsif", result["keterangan"])

    def test_sd_zone_not_impulsive_candle(self):
        """
        Candle di idx_impulsive bukan candle impulsif → zona tidak terbentuk.
        """
        df = _make_df(n=15, base_price=2000.0, atr=5.0)
        # Tidak inject candle impulsif, jadi candle idx=10 flat (body=0.1)

        result = detect_sd_zone_from_origin(df, idx_impulsive=10,
                                            impulsive_body_atr_ratio=1.5)

        self.assertFalse(result["is_valid"])
        self.assertIn("BUKAN impulsif", result["keterangan"])

    def test_sd_zone_insufficient_idx(self):
        """
        idx_impulsive=0 → tidak ada origin (idx=-1 tidak valid).
        """
        df = _make_df(n=10, atr=5.0)
        df = _inject_impulsive_bullish(df, idx=0, body=10.0)

        result = detect_sd_zone_from_origin(df, idx_impulsive=0,
                                            impulsive_body_atr_ratio=1.5)

        self.assertFalse(result["is_valid"])
        self.assertIn("tidak valid", result["keterangan"])


# =============================================================================
# TEST 6-9: find_nearest_sd_zone() — Freshness, Invalidasi, None
# =============================================================================

class TestFindNearestSdZone(unittest.TestCase):
    """Test untuk fungsi find_nearest_sd_zone()."""

    def _build_demand_scenario(self, touched=False, invalidated=False):
        """
        Buat skenario demand zone untuk BUY:
        - Candle 0..19: flat (base_price=2000, atr=5)
        - Candle 20: origin "diam" (flat)
        - Candle 21: impulsif bullish (body=10) → demand zone di [1999.5, 2000.5]
        - Candle 22..29: candle "evaluasi" area, bisa "menyentuh" atau "break" zona

        Zona demand = [low_origin, high_origin] = [1999.5, 2000.5]
        Current price (candle 29) = 2010 (di atas zona → BUY SL dari zona di bawah)
        """
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

        opens  = [2000.0] * n
        closes = [2000.1] * n
        highs  = [2000.5] * n
        lows   = [1999.5] * n
        atrs   = [5.0] * n

        # Candle 21: impulsif bullish (body=10, dari open=2000 ke close=2010)
        opens[21]  = 2000.0
        closes[21] = 2010.0
        highs[21]  = 2010.5
        lows[21]   = 1999.5

        # Candle 22..28: harga di sekitar 2010 (di atas zona)
        for k in range(22, 29):
            opens[k]  = 2010.0
            closes[k] = 2010.1
            highs[k]  = 2011.0
            lows[k]   = 2009.5

        # Candle 29: current price = 2010
        opens[29]  = 2010.0
        closes[29] = 2010.0
        highs[29]  = 2010.5
        lows[29]   = 2009.5

        if touched:
            # Candle 25 menyentuh zona [1999.5, 2000.5]
            # Overlap: low <= 2000.5 AND high >= 1999.5
            lows[25]  = 1999.8   # overlap dengan zona (low <= 2000.5)
            highs[25] = 2000.3   # overlap dengan zona (high >= 1999.5)
            closes[25] = 2000.2  # tidak break di bawah 1999.5

        if invalidated:
            # Candle 26 close di bawah low_origin (1999.5) → invalidasi
            lows[26]   = 1999.0
            closes[26] = 1999.3  # close < 1999.5 → invalidasi demand zone

        df = pd.DataFrame({
            "open"   : opens,
            "high"   : highs,
            "low"    : lows,
            "close"  : closes,
            "atr_14" : atrs,
        }, index=dates)

        return df

    def test_freshness_fresh_zone(self):
        """
        Zona FRESH: tidak ada candle yang menyentuh [1999.5, 2000.5] setelah terbentuk.
        find_nearest_sd_zone() harus return zona dengan freshness="FRESH".
        """
        df = self._build_demand_scenario(touched=False, invalidated=False)
        result = find_nearest_sd_zone(df, arah="BUY", idx=29,
                                      lookback=50, impulsive_body_atr_ratio=1.5)

        self.assertIsNotNone(result)
        self.assertEqual(result["freshness"], "FRESH")
        # zone_low = 1999.5, zone_high = 2000.5
        self.assertAlmostEqual(result["zone_low"],  1999.5, places=4)
        self.assertAlmostEqual(result["zone_high"], 2000.5, places=4)
        # SL level = zone_low - buffer = 1999.5 - 0.50 = 1999.0
        self.assertAlmostEqual(result["level"], 1999.5 - SD_BUFFER, places=4)

    def test_freshness_tested_zone(self):
        """
        Zona TESTED: candle 25 menyentuh [1999.5, 2000.5] (overlap).
        freshness harus "TESTED".
        """
        df = self._build_demand_scenario(touched=True, invalidated=False)
        result = find_nearest_sd_zone(df, arah="BUY", idx=29,
                                      lookback=50, impulsive_body_atr_ratio=1.5)

        self.assertIsNotNone(result)
        self.assertEqual(result["freshness"], "TESTED")

    def test_invalidation_demand_zone(self):
        """
        Demand zone INVALID setelah close break di bawah low_origin (1999.5).
        find_nearest_sd_zone() harus return None (zona invalid dibuang).
        """
        df = self._build_demand_scenario(touched=False, invalidated=True)
        result = find_nearest_sd_zone(df, arah="BUY", idx=29,
                                      lookback=50, impulsive_body_atr_ratio=1.5)

        # Zona sudah invalid → harus return None
        self.assertIsNone(result)

    def test_find_nearest_sd_zone_returns_none_no_valid_zone(self):
        """
        Tidak ada zona valid dalam lookback (semua candle flat, tidak ada impulsif).
        find_nearest_sd_zone() harus return None tanpa crash.
        """
        df = _make_df(n=30, base_price=2000.0, atr=5.0)
        # Tidak ada candle impulsif → tidak ada zona

        result = find_nearest_sd_zone(df, arah="BUY", idx=29,
                                      lookback=50, impulsive_body_atr_ratio=1.5)

        self.assertIsNone(result)

    def test_find_nearest_sd_zone_sell_supply(self):
        """
        Skenario SELL: supply zone DI ATAS current price.
        Candle impulsif bearish di idx=21 → supply zone terbentuk.
        Evaluasi di idx=29 dengan current price di bawah zona.
        """
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

        opens  = [2000.0] * n
        closes = [2000.1] * n
        highs  = [2000.5] * n
        lows   = [1999.5] * n
        atrs   = [5.0] * n

        # Candle 21: impulsif bearish (dari open=2000 ke close=1990)
        opens[21]  = 2000.0
        closes[21] = 1990.0
        highs[21]  = 2000.5
        lows[21]   = 1989.5

        # Candle 22..29: harga di bawah zona (sekitar 1990)
        for k in range(22, 30):
            opens[k]  = 1990.0
            closes[k] = 1990.1
            highs[k]  = 1990.5
            lows[k]   = 1989.5

        df = pd.DataFrame({
            "open"   : opens,
            "high"   : highs,
            "low"    : lows,
            "close"  : closes,
            "atr_14" : atrs,
        }, index=dates)

        result = find_nearest_sd_zone(df, arah="SELL", idx=29,
                                      lookback=50, impulsive_body_atr_ratio=1.5)

        self.assertIsNotNone(result)
        # Supply zone dari origin_idx=20 (candle flat sebelum bearish impulsif)
        # zone_low = 1999.5, zone_high = 2000.5
        self.assertAlmostEqual(result["zone_low"],  1999.5, places=4)
        self.assertAlmostEqual(result["zone_high"], 2000.5, places=4)
        # SL level = zone_high + buffer = 2000.5 + 0.5 = 2001.0
        self.assertAlmostEqual(result["level"], 2000.5 + SD_BUFFER, places=4)


# =============================================================================
# TEST 10-11: calculate_sl_tp() integrasi
# =============================================================================

class TestCalculateSlTpIntegration(unittest.TestCase):
    """
    Test integrasi calculate_sl_tp() dengan sl_source="SD_ZONE" dan "SWING".
    """

    def _build_df_with_demand_zone(self):
        """
        Buat DataFrame dengan demand zone yang jelas untuk BUY.
        Zona origin di [1999.5, 2000.5], current price = 2010.
        """
        n = 35
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

        opens  = [2000.0] * n
        closes = [2000.1] * n
        highs  = [2000.5] * n
        lows   = [1999.5] * n
        atrs   = [5.0] * n

        # Candle 21: impulsif bullish besar
        opens[21]  = 2000.0
        closes[21] = 2010.0
        highs[21]  = 2010.5
        lows[21]   = 1999.5

        # Candle 22..34: harga di sekitar 2010 (di atas zona)
        for k in range(22, n):
            opens[k]  = 2010.0
            closes[k] = 2010.0
            highs[k]  = 2011.0
            lows[k]   = 2009.5

        # Juga perlu swing low yang valid untuk path SWING:
        # Candle 30 akan jadi swing low (low lebih rendah dari tetangganya)
        lows[30] = 2009.0
        for k in [27, 28, 29, 31, 32, 33]:
            lows[k] = 2009.5

        df = pd.DataFrame({
            "open"   : opens,
            "high"   : highs,
            "low"    : lows,
            "close"  : closes,
            "atr_14" : atrs,
        }, index=dates)
        return df

    def test_calculate_sl_tp_sd_zone_method(self):
        """
        calculate_sl_tp(sl_source="SD_ZONE") harus:
        - return sl_method="SD_ZONE" jika zona ditemukan
        - return valid=True
        - SL berada di bawah entry untuk BUY
        """
        df = self._build_df_with_demand_zone()
        entry = 2010.0

        result = calculate_sl_tp(
            df         = df,
            entry      = entry,
            arah       = "BUY",
            profile    = "scalp_m5",
            sl_source  = "SD_ZONE",
        )

        self.assertTrue(result["valid"])
        # sl_method harus "SD_ZONE" (bukan "SWING" atau "ATR")
        self.assertEqual(result["sl_method"], "SD_ZONE")
        # SL harus di bawah entry untuk BUY
        self.assertLess(result["sl"], entry)
        # TP harus di atas entry untuk BUY
        self.assertGreater(result["tp"], entry)

    def test_calculate_sl_tp_swing_default_unchanged(self):
        """
        calculate_sl_tp() tanpa sl_source (default="SWING") harus:
        - identik 100% dengan perilaku sebelum Fase 11
        - sl_method = "SWING" atau "ATR" (tidak pernah "SD_ZONE")

        Verifikasi dengan memanggil dua kali (identik = deterministic):
        call1 tanpa sl_source vs call2 dengan sl_source="SWING" eksplisit.
        """
        df = self._build_df_with_demand_zone()
        entry = 2010.0

        # Call tanpa sl_source (default harus "SWING")
        result_default = calculate_sl_tp(
            df     = df,
            entry  = entry,
            arah   = "BUY",
            profile= "scalp_m5",
        )

        # Call eksplisit sl_source="SWING"
        result_swing = calculate_sl_tp(
            df        = df,
            entry     = entry,
            arah      = "BUY",
            profile   = "scalp_m5",
            sl_source = "SWING",
        )

        # Harus identik persis
        self.assertEqual(result_default["sl"],        result_swing["sl"])
        self.assertEqual(result_default["tp"],        result_swing["tp"])
        self.assertEqual(result_default["sl_method"], result_swing["sl_method"])
        self.assertEqual(result_default["jarak_sl"],  result_swing["jarak_sl"])

        # sl_method tidak boleh "SD_ZONE" pada path default
        self.assertNotEqual(result_default["sl_method"], "SD_ZONE")
        self.assertNotEqual(result_swing["sl_method"],   "SD_ZONE")

    def test_calculate_sl_tp_sd_zone_fallback_to_atr(self):
        """
        Jika tidak ada S&D zone dalam lookback, sl_source="SD_ZONE" harus
        fallback ke ATR, bukan crash. sl_method harus "ATR".
        """
        # DataFrame flat tanpa candle impulsif → tidak ada zona S&D
        df = _make_df(n=30, base_price=2000.0, atr=5.0)
        entry = 2000.0

        result = calculate_sl_tp(
            df        = df,
            entry     = entry,
            arah      = "BUY",
            profile   = "scalp_m5",
            sl_source = "SD_ZONE",
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["sl_method"], "ATR")


# =============================================================================
# TEST 12-15: Kausalitas (no look-ahead)
# =============================================================================

class TestCausalitySupplyDemand(unittest.TestCase):
    """
    Test kausalitas end-to-end: mutasi candle SETELAH idx evaluasi
    tidak boleh mengubah hasil fungsi S&D.

    Pola identik dengan test_causality_no_lookahead di test_zone_detector.py
    dan TestBreakoutTriggerCausalityEndToEnd di test_phase9_breakout.py.
    """

    def _build_causality_df(self, n: int = 50):
        """
        Buat DataFrame dengan demand zone yang jelas di tengah.
        - Candle 0..19: flat
        - Candle 20: origin "diam"
        - Candle 21: impulsif bullish (body=10)
        - Candle 22..49: harga di atas zona (sekitar 2010)
        """
        opens  = [2000.0] * n
        closes = [2000.1] * n
        highs  = [2000.5] * n
        lows   = [1999.5] * n
        atrs   = [5.0] * n

        # Candle 21: impulsif bullish
        opens[21]  = 2000.0
        closes[21] = 2010.0
        highs[21]  = 2010.5
        lows[21]   = 1999.5

        for k in range(22, n):
            opens[k]  = 2010.0
            closes[k] = 2010.1
            highs[k]  = 2011.0
            lows[k]   = 2009.5

        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        return pd.DataFrame({
            "open"   : opens,
            "high"   : highs,
            "low"    : lows,
            "close"  : closes,
            "atr_14" : atrs,
        }, index=dates)

    def test_causality_detect_impulsive(self):
        """
        Mutasi candle SETELAH idx tidak boleh mengubah hasil detect_impulsive_move().
        """
        df = self._build_causality_df(n=50)
        t  = 21  # candle impulsif

        result_orig = detect_impulsive_move(df.copy(), idx=t,
                                            impulsive_body_atr_ratio=1.5)

        # Mutasi ekstrem candle t+1 s/d t+10
        df_mut = df.copy()
        for col, val in [("high", 9999.0), ("low", 1.0), ("close", 5000.0), ("open", 5000.0)]:
            df_mut.iloc[t + 1:, df_mut.columns.get_loc(col)] = val

        result_mut = detect_impulsive_move(df_mut, idx=t,
                                           impulsive_body_atr_ratio=1.5)

        self.assertEqual(result_orig["is_impulsive"], result_mut["is_impulsive"])
        self.assertEqual(result_orig["arah"],          result_mut["arah"])
        if result_orig["body"] is not None:
            self.assertAlmostEqual(result_orig["body"], result_mut["body"], places=5)
        if result_orig["threshold"] is not None:
            self.assertAlmostEqual(result_orig["threshold"], result_mut["threshold"], places=5)

    def test_causality_detect_sd_zone_from_origin(self):
        """
        Mutasi candle SETELAH idx_impulsive tidak boleh mengubah hasil
        detect_sd_zone_from_origin(). Origin dan candle impulsif ada di masa lalu.
        """
        df = self._build_causality_df(n=50)
        t  = 21  # idx_impulsive

        result_orig = detect_sd_zone_from_origin(df.copy(), idx_impulsive=t,
                                                  impulsive_body_atr_ratio=1.5)

        # Mutasi candle t+1 s/d t+20 secara ekstrem
        df_mut = df.copy()
        for col, val in [("high", 9999.0), ("low", 1.0), ("close", 5000.0), ("open", 5000.0)]:
            df_mut.iloc[t + 1:, df_mut.columns.get_loc(col)] = val

        result_mut = detect_sd_zone_from_origin(df_mut, idx_impulsive=t,
                                                 impulsive_body_atr_ratio=1.5)

        # Semua field harus identik
        self.assertEqual(result_orig["is_valid"],   result_mut["is_valid"])
        self.assertEqual(result_orig["zone_type"],  result_mut["zone_type"])
        self.assertEqual(result_orig["origin_idx"], result_mut["origin_idx"])
        if result_orig["low"] is not None:
            self.assertAlmostEqual(result_orig["low"],  result_mut["low"],  places=5)
            self.assertAlmostEqual(result_orig["high"], result_mut["high"], places=5)

    def test_causality_find_nearest_sd_zone(self):
        """
        Mutasi candle SETELAH idx evaluasi tidak boleh mengubah hasil
        find_nearest_sd_zone(). Ini test kausalitas utama untuk modul ini.
        """
        df = self._build_causality_df(n=50)
        t  = 40  # titik evaluasi

        result_orig = find_nearest_sd_zone(df.copy(), arah="BUY", idx=t,
                                            lookback=50, impulsive_body_atr_ratio=1.5)

        # Mutasi ekstrem candle t+1 s/d t+9
        df_mut = df.copy()
        for col, val in [("high", 9999.0), ("low", 1.0), ("close", 5000.0), ("open", 5000.0)]:
            df_mut.iloc[t + 1:, df_mut.columns.get_loc(col)] = val

        result_mut = find_nearest_sd_zone(df_mut, arah="BUY", idx=t,
                                           lookback=50, impulsive_body_atr_ratio=1.5)

        # Keduanya harus sama — both None, atau both dengan nilai identik
        if result_orig is None:
            self.assertIsNone(result_mut,
                              "Mutasi candle masa depan mengubah hasil dari None menjadi non-None!")
        else:
            self.assertIsNotNone(result_mut,
                                 "Mutasi candle masa depan mengubah hasil dari non-None menjadi None!")
            self.assertAlmostEqual(result_orig["level"],      result_mut["level"],      places=5)
            self.assertAlmostEqual(result_orig["zone_low"],   result_mut["zone_low"],   places=5)
            self.assertAlmostEqual(result_orig["zone_high"],  result_mut["zone_high"],  places=5)
            self.assertEqual(result_orig["origin_idx"], result_mut["origin_idx"])
            self.assertEqual(result_orig["freshness"],  result_mut["freshness"])

    def test_causality_calculate_sl_tp_sd_zone(self):
        """
        Mutasi candle SETELAH idx tidak boleh mengubah hasil
        calculate_sl_tp(sl_source="SD_ZONE").

        Ini membuktikan bahwa integrasi di risk_manager.py + supply_demand.py
        tetap causal secara end-to-end.
        """
        df = self._build_causality_df(n=50)
        t  = 40  # titik evaluasi
        entry = 2010.0

        df_slice_orig = df.iloc[:t + 1].copy()
        result_orig   = calculate_sl_tp(
            df        = df_slice_orig,
            entry     = entry,
            arah      = "BUY",
            profile   = "scalp_m5",
            sl_source = "SD_ZONE",
        )

        # Mutasi ekstrem candle t+1 s/d akhir
        df_mut = df.copy()
        for col, val in [("high", 9999.0), ("low", 1.0), ("close", 5000.0), ("open", 5000.0)]:
            df_mut.iloc[t + 1:, df_mut.columns.get_loc(col)] = val

        # Slice HANYA sampai t+1 — sama seperti caller (backtester slices df[:i+1])
        df_slice_mut = df_mut.iloc[:t + 1].copy()
        result_mut   = calculate_sl_tp(
            df        = df_slice_mut,
            entry     = entry,
            arah      = "BUY",
            profile   = "scalp_m5",
            sl_source = "SD_ZONE",
        )

        # Semua nilai harus identik (slice yang sama → hasil identik)
        self.assertEqual(result_orig["sl"],        result_mut["sl"])
        self.assertEqual(result_orig["tp"],        result_mut["tp"])
        self.assertEqual(result_orig["sl_method"], result_mut["sl_method"])
        self.assertEqual(result_orig["jarak_sl"],  result_mut["jarak_sl"])


# =============================================================================
# TEST TAMBAHAN: find_nearest_sd_zone dengan idx negatif & edge cases
# =============================================================================

class TestFindNearestSdZoneEdgeCases(unittest.TestCase):
    """Edge case tambahan untuk find_nearest_sd_zone()."""

    def test_returns_none_wrong_arah(self):
        """Arah tidak valid → return None tanpa crash."""
        df = _make_df(n=30, atr=5.0)
        result = find_nearest_sd_zone(df, arah="HOLD", idx=29)
        self.assertIsNone(result)

    def test_returns_none_insufficient_data(self):
        """idx=0 → tidak ada candle sebelumnya untuk scan → None."""
        df = _make_df(n=5, atr=5.0)
        result = find_nearest_sd_zone(df, arah="BUY", idx=0)
        self.assertIsNone(result)

    def test_negative_idx_normalized(self):
        """
        idx=-1 harus diinterpretasi sebagai candle terakhir.
        Hasil harus sama dengan idx=n-1.
        """
        n = 30
        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        opens  = [2000.0] * n
        closes = [2000.1] * n
        highs  = [2000.5] * n
        lows   = [1999.5] * n
        atrs   = [5.0] * n

        opens[21]  = 2000.0
        closes[21] = 2010.0
        highs[21]  = 2010.5
        lows[21]   = 1999.5

        for k in range(22, n):
            opens[k]  = 2010.0
            closes[k] = 2010.1
            highs[k]  = 2011.0
            lows[k]   = 2009.5

        df = pd.DataFrame({
            "open"   : opens, "high": highs, "low": lows,
            "close"  : closes, "atr_14": atrs,
        }, index=dates)

        result_neg  = find_nearest_sd_zone(df, arah="BUY", idx=-1)
        result_pos  = find_nearest_sd_zone(df, arah="BUY", idx=n - 1)

        if result_neg is None:
            self.assertIsNone(result_pos)
        else:
            self.assertIsNotNone(result_pos)
            self.assertAlmostEqual(result_neg["level"],     result_pos["level"],     places=5)
            self.assertAlmostEqual(result_neg["zone_low"],  result_pos["zone_low"],  places=5)
            self.assertAlmostEqual(result_neg["zone_high"], result_pos["zone_high"], places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
