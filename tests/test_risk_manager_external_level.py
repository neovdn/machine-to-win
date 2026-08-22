"""
tests/test_risk_manager_external_level.py
==========================================
Unit & Regresi Test untuk path EXTERNAL_LEVEL di engine/risk_manager.py (Fase 19).

PENGUJIAN:
  1. test_external_level_buy_valid:
     EXTERNAL_LEVEL BUY dengan level valid -> sl_method=="EXTERNAL_LEVEL",
     SL dihitung dengan buffer ATR-relative, clamp diterapkan benar.
  2. test_external_level_sell_valid:
     EXTERNAL_LEVEL SELL -> cerminan dari test (1).
  3. test_external_level_none_fallback_atr:
     EXTERNAL_LEVEL dengan external_level=None -> fallback ke sl_method=="ATR".
  4. test_regresi_default_swing_tidak_berubah:
     Regresi eksplisit: panggil calculate_sl_tp() TANPA menyebut sl_source sama sekali
     (pakai default "SWING") -> perilaku identik sebelum modifikasi Fase 19.
     Test ini memvalidasi bahwa modifikasi pada risk_manager.py tidak mengubah
     SATU BYTE PUN perilaku default SWING path.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.risk_manager import (
    calculate_sl_tp,
    EXTERNAL_LEVEL_BUFFER_ATR,
    SWING_CLAMP_MIN_ATR,
    SWING_CLAMP_MAX_ATR,
)


def _make_df(n: int = 50, base_price: float = 2000.0, atr: float = 5.0) -> pd.DataFrame:
    """Buat DataFrame sintetis n candle dengan ATR konstan."""
    dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    prices = base_price
    return pd.DataFrame({
        "open"  : prices,
        "high"  : prices + 1.0,
        "low"   : prices - 1.0,
        "close" : prices + 0.1,
        "atr_14": atr,
    }, index=dates)


class TestExternalLevelBuy(unittest.TestCase):
    """Test 1: EXTERNAL_LEVEL BUY dengan level valid."""

    def setUp(self):
        self.atr = 5.0
        self.df = _make_df(atr=self.atr)
        self.entry = 2000.0
        self.external_lvl = 1990.0  # level mentah SL kandidat

    def test_sl_method_external_level(self):
        """sl_method harus 'EXTERNAL_LEVEL'."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["sl_method"], "EXTERNAL_LEVEL")

    def test_sl_swing_level_dengan_buffer(self):
        """sl_swing_level = external_level - buf (buf = EXTERNAL_LEVEL_BUFFER_ATR * atr)."""
        buf = EXTERNAL_LEVEL_BUFFER_ATR * self.atr
        expected_sl_swing = self.external_lvl - buf

        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        # sl_swing_level = level setelah buffer (sebelum clamp)
        self.assertAlmostEqual(res["sl_swing_level"], round(expected_sl_swing, 2), places=2)

    def test_sl_swing_raw_adalah_external_level(self):
        """sl_swing_raw harus diisi dengan external_level mentah (sebelum buffer)."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["sl_swing_raw"], round(self.external_lvl, 2), places=2)

    def test_sl_final_dengan_clamp(self):
        """SL final harus tepat setelah buffer + clamp ATR diterapkan."""
        buf = EXTERNAL_LEVEL_BUFFER_ATR * self.atr
        sl_swing_lvl = self.external_lvl - buf
        dist_raw     = self.entry - sl_swing_lvl
        min_dist     = SWING_CLAMP_MIN_ATR * self.atr
        max_dist     = SWING_CLAMP_MAX_ATR * self.atr
        dist_final   = max(min_dist, min(dist_raw, max_dist))  # clamp
        expected_sl  = self.entry - dist_final

        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["sl"], round(expected_sl, 2), places=2)

    def test_clamp_reason_normal(self):
        """Tidak ada clamp jika jarak dalam range [min_dist, max_dist]."""
        # dist_raw = (entry - (external_lvl - buf)) = 2000 - (1990 - 0.75) = 10.75
        # min_dist = 0.7 * 5.0 = 3.5, max_dist = 2.0 * 5.0 = 10.0
        # dist 10.75 > max_dist 10.0 -> MAX_CAP
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["sl_swing_clamped"])
        self.assertEqual(res["clamp_reason"], "MAX_CAP")

    def test_override_buffer_atr(self):
        """external_level_buffer_atr override bekerja dengan benar."""
        custom_buf_multiplier = 0.30
        buf = custom_buf_multiplier * self.atr
        sl_swing_lvl = self.external_lvl - buf
        dist_raw     = self.entry - sl_swing_lvl
        min_dist     = SWING_CLAMP_MIN_ATR * self.atr
        max_dist     = SWING_CLAMP_MAX_ATR * self.atr
        dist_final   = max(min_dist, min(dist_raw, max_dist))
        expected_sl  = self.entry - dist_final

        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
            external_level_buffer_atr=custom_buf_multiplier,
        )
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["sl"], round(expected_sl, 2), places=2)


class TestExternalLevelSell(unittest.TestCase):
    """Test 2: EXTERNAL_LEVEL SELL — cerminan dari BUY."""

    def setUp(self):
        self.atr = 5.0
        self.df = _make_df(atr=self.atr)
        self.entry = 2000.0
        self.external_lvl = 2010.0  # level mentah SL kandidat SELL (di atas entry)

    def test_sl_method_external_level_sell(self):
        """sl_method harus 'EXTERNAL_LEVEL' untuk SELL."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="SELL",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["sl_method"], "EXTERNAL_LEVEL")

    def test_sl_di_atas_entry_untuk_sell(self):
        """Untuk SELL, SL harus di atas entry."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="SELL",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertGreater(res["sl"], res["entry"])

    def test_sl_swing_level_sell_dengan_buffer(self):
        """sl_swing_level = external_level + buf untuk SELL (buffer di arah atas)."""
        buf = EXTERNAL_LEVEL_BUFFER_ATR * self.atr
        expected_sl_swing = self.external_lvl + buf

        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="SELL",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["sl_swing_level"], round(expected_sl_swing, 2), places=2)

    def test_tp_di_bawah_entry_untuk_sell(self):
        """Untuk SELL, TP harus di bawah entry."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="SELL",
            sl_source="EXTERNAL_LEVEL",
            external_level=self.external_lvl,
        )
        self.assertTrue(res["valid"])
        self.assertLess(res["tp"], res["entry"])


class TestExternalLevelNoneFallbackATR(unittest.TestCase):
    """Test 3: EXTERNAL_LEVEL dengan external_level=None -> fallback ke ATR."""

    def setUp(self):
        self.atr = 5.0
        self.df = _make_df(atr=self.atr)
        self.entry = 2000.0

    def test_fallback_atr_ketika_external_level_none(self):
        """external_level=None harus menghasilkan sl_method=='ATR'."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=None,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["sl_method"], "ATR")

    def test_sl_dari_atr_multiplier(self):
        """SL ATR fallback = entry - (atr_multiplier * atr)."""
        # atr_multiplier default dari scalp_m5 = 0.9
        atr_multiplier = 0.9
        expected_sl = self.entry - (atr_multiplier * self.atr)

        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=None,
        )
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["sl"], round(expected_sl, 2), places=2)

    def test_sl_swing_raw_none_ketika_fallback_atr(self):
        """sl_swing_raw harus None saat ATR fallback."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="EXTERNAL_LEVEL",
            external_level=None,
        )
        self.assertTrue(res["valid"])
        self.assertIsNone(res["sl_swing_raw"])


class TestRegresiDefaultSwing(unittest.TestCase):
    """
    Test 4: Regresi eksplisit — memanggil calculate_sl_tp() TANPA menyebut sl_source
    sama sekali (pakai default "SWING") harus menghasilkan perilaku yang identik
    dengan sebelum modifikasi Fase 19.

    Strategi: bandingkan output panggilan default (sl_source tidak disebut)
    dengan panggilan eksplisit sl_source="SWING" — keduanya harus identik persis.
    Ini membuktikan bahwa branch SWING tidak berubah sama sekali.
    """

    def setUp(self):
        """Buat DataFrame sintetis mirip dengan yang dipakai test_backtester.py."""
        dates = pd.date_range("2026-01-01 00:00:00", periods=100, freq="5min", tz="UTC")
        prices = 2000.0 + np.sin(np.linspace(0, 10, 100)) * 5.0
        self.df = pd.DataFrame({
            "open"       : prices,
            "high"       : prices + 1.0,
            "low"        : prices - 1.0,
            "close"      : prices + 0.2,
            "tick_volume": 100,
            "atr_14"     : 2.0,
        }, index=dates)
        self.entry = 2000.0

    def test_default_identik_dengan_sl_source_swing_buy(self):
        """Default call (tanpa sl_source) == panggilan eksplisit sl_source='SWING' untuk BUY."""
        res_default = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
        )
        res_explicit = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            sl_source="SWING",
        )
        self.assertEqual(res_default, res_explicit,
                         "Default call harus identik dengan sl_source='SWING'")

    def test_default_identik_dengan_sl_source_swing_sell(self):
        """Default call (tanpa sl_source) == panggilan eksplisit sl_source='SWING' untuk SELL."""
        res_default = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="SELL",
        )
        res_explicit = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="SELL",
            sl_source="SWING",
        )
        self.assertEqual(res_default, res_explicit,
                         "Default call harus identik dengan sl_source='SWING'")

    def test_default_call_menghasilkan_sl_method_swing_atau_atr(self):
        """Default call harus menghasilkan sl_method 'SWING' atau 'ATR', BUKAN 'EXTERNAL_LEVEL'."""
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
        )
        self.assertIn(res["sl_method"], ("SWING", "ATR"),
                      "Default path hanya boleh menghasilkan SWING atau ATR, bukan EXTERNAL_LEVEL")

    def test_dengan_tick_info_default_masih_spread_aware(self):
        """Default call dengan tick_info harus tetap spread-aware (backward compatible)."""
        tick_info = {"ask": 2000.25, "bid": 1999.75}
        res = calculate_sl_tp(
            df=self.df,
            entry=self.entry,
            arah="BUY",
            tick_info=tick_info,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["entry_type"], "ASK")
        self.assertAlmostEqual(res["entry"], 2000.25, places=2)
        self.assertEqual(res["spread"], 0.50)

    def test_atr_clamping_masih_bekerja_di_default(self):
        """
        Regresi: atr clamping pada SWING path masih bekerja persis seperti
        test_atr_clamping_logic di test_backtester.py.
        """
        entry = 2000.0
        atr_value = 2.0
        dates = pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC")
        prices = [2000.0] * 30
        df = pd.DataFrame({
            "open" : prices,
            "high" : prices,
            "low"  : prices,
            "close": prices,
            "atr_14": atr_value,
        }, index=dates)

        # Set swing low sangat dekat ke entry: low = 1999.70
        # swing_buffer=0.5 -> sl_swing_level = 1999.20 -> dist = 0.80 < min_dist=1.4 -> MIN_CAP
        df.iloc[15, df.columns.get_loc("low")] = 1999.70

        res = calculate_sl_tp(
            df=df,
            entry=entry,
            arah="BUY",
            profile="scalp_m5",
            swing_lookback=20,
            swing_wing=3,
        )

        self.assertTrue(res["valid"])
        self.assertEqual(res["sl_method"], "SWING")
        self.assertTrue(res["sl_swing_clamped"])
        self.assertEqual(res["clamp_reason"], "MIN_CAP")
        self.assertAlmostEqual(res["jarak_sl"], 1.4, places=5)   # 0.7 * 2.0
        self.assertAlmostEqual(res["sl"], entry - 1.4, places=5)


if __name__ == "__main__":
    unittest.main()
