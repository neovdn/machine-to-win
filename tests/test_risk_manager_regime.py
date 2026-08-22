"""
tests/test_risk_manager_regime.py
==================================
Unit Test untuk engine/risk_manager_regime.py (Fase 19).

PENGUJIAN (8 skenario wajib per spec):
  1. RANGE_REVERSAL dengan invalidation_level valid, TP tidak overshoot boundary
     seberang -> valid=True, tp_capped=False.
  2. RANGE_REVERSAL dengan TP overshoot boundary seberang TAPI RRR setelah cap
     masih layak -> valid=True, tp_capped=True, tp sama dengan boundary seberang.
  3. RANGE_REVERSAL dengan TP overshoot DAN RRR setelah cap TIDAK layak ->
     valid=False, skip_reason menjelaskan RRR.
  4. BREAKOUT_RETEST dengan invalidation_level_sl valid -> valid=True,
     tp_capped=False (cap TIDAK berlaku meskipun zone diberikan).
  5. TREND_FOLLOWING dengan invalidation_level_sl valid -> valid=True.
  6. Level referensi None/hilang di strategy_result -> valid=False, tidak crash.
  7. strategy_name tidak dikenal -> valid=False, tidak crash.
  8. zone tidak diberikan (None) untuk RANGE_REVERSAL -> Langkah 3 dilewati
     secara graceful (tidak crash, tp_capped=False).
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.risk_manager_regime import (
    calculate_regime_sl_tp,
    STRATEGY_LEVEL_FIELD_MAP,
    REGIME_RISK_MIN_RRR_AFTER_CAP,
)


# =============================================================================
# HELPER
# =============================================================================

def _make_df(n: int = 50, base_price: float = 2000.0, atr: float = 5.0) -> pd.DataFrame:
    """Buat DataFrame sintetis n candle dengan ATR konstan."""
    dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open"  : base_price,
        "high"  : base_price + 1.0,
        "low"   : base_price - 1.0,
        "close" : base_price + 0.1,
        "atr_14": atr,
    }, index=dates)


def _make_zone(support: float, resistance: float) -> dict:
    return {"support": support, "resistance": resistance}


def _make_range_reversal_result(invalidation_level: float) -> dict:
    """Simulasi hasil evaluate_range_reversal() yang terpenuhi."""
    return {
        "terpenuhi"          : True,
        "arah"               : "BUY",
        "boundary_referensi" : "support",
        "sweep_terpenuhi"    : True,
        "rejection_terpenuhi": True,
        "invalidation_level" : invalidation_level,
        "keterangan"         : "sintetis untuk test",
    }


def _make_breakout_result(invalidation_level_sl: float, arah: str = "BUY") -> dict:
    """Simulasi hasil evaluate_breakout_retest() yang terpenuhi."""
    return {
        "terpenuhi"            : True,
        "arah"                 : arah,
        "invalidation_level_sl": invalidation_level_sl,
        "keterangan"           : "sintetis untuk test",
    }


def _make_trend_following_result(invalidation_level_sl: float, arah: str = "BUY") -> dict:
    """Simulasi hasil evaluate_trend_following() yang terpenuhi."""
    return {
        "terpenuhi"            : True,
        "arah"                 : arah,
        "invalidation_level_sl": invalidation_level_sl,
        "keterangan"           : "sintetis untuk test",
    }


# =============================================================================
# TEST CLASS 1: RANGE_REVERSAL
# =============================================================================

class TestRangeReversalNoOvershoot(unittest.TestCase):
    """
    Test 1: RANGE_REVERSAL dengan invalidation_level valid, TP tidak overshoot
    boundary seberang -> valid=True, tp_capped=False.
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        # Range: support=1990, resistance=2050 (lebar 60 poin)
        # BUY: invalidation_level di bawah support, TP target seberang = resistance=2050
        # Dengan ATR=5, RRR default ~1.3 -> TP tidak jauh, tidak melampaui 2050
        self.df = _make_df(atr=self.atr, base_price=self.entry)
        # Pilih invalidation_level sangat dekat entry supaya jarak SL kecil -> TP kecil
        # Setelah clamp min (0.7*5=3.5) -> jarak_sl=3.5 -> TP=entry+(3.5*1.3)=2004.55 < resistance=2050
        self.invalidation_level = 1998.0   # sangat dekat -> akan di-clamp ke min
        self.zone = _make_zone(support=1990.0, resistance=2050.0)
        self.strategy_result = _make_range_reversal_result(self.invalidation_level)

    def test_valid_true(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
        )
        self.assertTrue(res["valid"], f"Harusnya valid=True. skip_reason: {res.get('skip_reason')}")

    def test_tp_capped_false(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
        )
        self.assertFalse(res["tp_capped"], "TP tidak seharusnya di-cap")

    def test_tp_original_none(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
        )
        self.assertIsNone(res["tp_original"], "tp_original harus None jika tidak di-cap")

    def test_sl_method_external_level(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
        )
        self.assertEqual(res["sl_method"], "EXTERNAL_LEVEL")

    def test_tp_lebih_besar_dari_entry_buy(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
        )
        self.assertGreater(res["tp"], res["entry"])


class TestRangeReversalOvershootRRRLayak(unittest.TestCase):
    """
    Test 2: RANGE_REVERSAL dengan TP overshoot boundary seberang, RRR setelah
    cap masih layak -> valid=True, tp_capped=True, tp == boundary seberang.
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        # BUY: resistance=2003 (sangat dekat — TP pasti overshoot)
        # invalidation_level sangat jauh di bawah -> SL jauh -> jarak_sl besar -> TP besar
        # Tapi resistance hanya 2003 -> TP akan di-cap ke 2003
        # RRR setelah cap: (2003 - entry) / jarak_sl harus >= REGIME_RISK_MIN_RRR_AFTER_CAP
        # Kita pastikan RRR layak dengan atur jarak_sl kecil
        # Caranya: invalidation_level dekat entry, clamp ke min_dist=3.5
        # jarak_sl = 3.5 -> TP original = entry + 3.5*1.3 = 2004.55 > resistance=2003
        # achievable_rrr = (2003 - entry) / 3.5 = 3/3.5 = 0.857 -> < 1.0 NOT LAYAK!
        # Ganti: resistance=2008, jarak_sl=3.5
        # achievable_rrr = (2008 - 2000) / 3.5 = 2.29 -> layak
        self.df = _make_df(atr=self.atr, base_price=self.entry)
        self.invalidation_level = 1998.0   # sangat dekat -> clamp ke min -> jarak_sl=3.5
        # TP original = 2000 + 3.5*1.3 = 2004.55
        # resistance = 2002 -> overshoot -> cap ke 2002
        # achievable_rrr = (2002 - 2000) / 3.5 = 0.57 -> TIDAK layak
        # Pakai resistance=2008 agar achievable_rrr=(2008-2000)/3.5=2.29 -> layak
        # dan TP original = 2004.55 < 2008? NO! Tidak overshoot!
        # Perlu TP overshoot: resistance harus < TP original = 2004.55
        # Gunakan resistance=2002 -> achievable_rrr=(2002-2000)/3.5=0.57 TIDAK layak
        # Gunakan jarak_sl kecil: invalidation harus menghasilkan jarak_sl sedikit
        # Dengan atr=5, min_dist=3.5. Agar achievable_rrr layak, resistance cukup jauh.
        # Kita set resistance=2006 -> achievable_rrr=(2006-2000)/3.5=1.71 -> layak
        # TP original = 2000+3.5*1.3=2004.55 < 2006 -> TIDAK overshoot!
        # Solusi: pakai rrr_min lebih besar supaya TP original overshoot resistance
        # rrr_min=3.0 -> TP=2000+3.5*3=2010.5 > resistance=2006 -> overshoot
        # achievable_rrr=(2006-2000)/3.5=1.71 >= 1.0 -> layak
        self.zone = _make_zone(support=1990.0, resistance=2006.0)
        self.strategy_result = _make_range_reversal_result(self.invalidation_level)
        self.rrr_min = 3.0  # agar TP original besar dan pasti overshoot

    def test_valid_true(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertTrue(res["valid"], f"Harusnya valid=True. skip_reason: {res.get('skip_reason')}")

    def test_tp_capped_true(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertTrue(res["tp_capped"], "TP seharusnya di-cap ke boundary seberang")

    def test_tp_sama_dengan_resistance(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertAlmostEqual(res["tp"], self.zone["resistance"], places=2,
                                msg="TP harus sama dengan resistance (boundary seberang)")

    def test_tp_original_tersimpan(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertIsNotNone(res["tp_original"], "tp_original harus tersimpan saat di-cap")
        # TP original harus lebih besar dari resistance (TP before cap > boundary)
        self.assertGreater(res["tp_original"], self.zone["resistance"])


class TestRangeReversalOvershootRRRTidakLayak(unittest.TestCase):
    """
    Test 3: RANGE_REVERSAL dengan TP overshoot DAN RRR setelah cap TIDAK layak
    -> valid=False, skip_reason menjelaskan RRR.
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        # BUY: resistance=2001.5 (sangat dekat entry)
        # invalidation_level jauh di bawah -> setelah clamp min, jarak_sl=3.5
        # TP original=2000+3.5*3=2010.5 > resistance=2001.5 -> overshoot
        # achievable_rrr = (2001.5-2000)/3.5 = 0.43 < 1.0 -> TIDAK LAYAK
        self.df = _make_df(atr=self.atr, base_price=self.entry)
        self.invalidation_level = 1998.0
        self.zone = _make_zone(support=1990.0, resistance=2001.5)
        self.strategy_result = _make_range_reversal_result(self.invalidation_level)
        self.rrr_min = 3.0

    def test_valid_false(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertFalse(res["valid"],
                         "Harusnya valid=False karena RRR tidak layak setelah cap")

    def test_skip_reason_menyebut_rrr(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])
        self.assertIn("RRR", res["skip_reason"],
                      "skip_reason harus menyebut RRR tidak layak")

    def test_tp_capped_true_meskipun_tidak_valid(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=self.zone,
            rrr_min=self.rrr_min,
        )
        self.assertFalse(res["valid"])
        self.assertTrue(res["tp_capped"])


# =============================================================================
# TEST CLASS 2: BREAKOUT_RETEST
# =============================================================================

class TestBreakoutRetestValid(unittest.TestCase):
    """
    Test 4: BREAKOUT_RETEST dengan invalidation_level_sl valid -> valid=True,
    tp_capped=False (cap TIDAK berlaku meskipun zone diberikan).
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        self.df = _make_df(atr=self.atr, base_price=self.entry)
        # Untuk BREAKOUT_RETEST, SL biasanya di bawah level breakout
        self.invalidation_level_sl = 1994.0  # level SL mentah
        self.strategy_result = _make_breakout_result(self.invalidation_level_sl)
        # Zone diberikan dengan resistance kecil — tapi cap TIDAK berlaku untuk strategi ini
        self.zone = _make_zone(support=1990.0, resistance=2003.0)

    def test_valid_true(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="BREAKOUT_RETEST",
            strategy_result=self.strategy_result,
            zone=self.zone,   # diberikan tapi diabaikan untuk cap
        )
        self.assertTrue(res["valid"],
                        f"Harusnya valid=True. skip_reason: {res.get('skip_reason')}")

    def test_tp_capped_false_meskipun_zone_diberikan(self):
        """Cap TP ke boundary seberang TIDAK berlaku untuk BREAKOUT_RETEST."""
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="BREAKOUT_RETEST",
            strategy_result=self.strategy_result,
            zone=self.zone,
        )
        self.assertTrue(res["valid"])
        self.assertFalse(res["tp_capped"],
                         "tp_capped harus False untuk BREAKOUT_RETEST (cap tidak berlaku)")

    def test_sl_method_external_level(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="BREAKOUT_RETEST",
            strategy_result=self.strategy_result,
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["sl_method"], "EXTERNAL_LEVEL")


# =============================================================================
# TEST CLASS 3: TREND_FOLLOWING
# =============================================================================

class TestTrendFollowingValid(unittest.TestCase):
    """
    Test 5: TREND_FOLLOWING dengan invalidation_level_sl valid -> valid=True.
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        self.df = _make_df(atr=self.atr, base_price=self.entry)
        self.invalidation_level_sl = 1993.0
        self.strategy_result = _make_trend_following_result(self.invalidation_level_sl)

    def test_valid_true(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="TREND_FOLLOWING",
            strategy_result=self.strategy_result,
        )
        self.assertTrue(res["valid"],
                        f"Harusnya valid=True. skip_reason: {res.get('skip_reason')}")

    def test_tp_capped_false(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="TREND_FOLLOWING",
            strategy_result=self.strategy_result,
        )
        self.assertTrue(res["valid"])
        self.assertFalse(res["tp_capped"])

    def test_semua_field_wajib_ada(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="TREND_FOLLOWING",
            strategy_result=self.strategy_result,
        )
        self.assertTrue(res["valid"])
        # Semua field wajib dari calculate_sl_tp() harus ada
        for field in ("entry", "sl", "tp", "rrr", "jarak_sl", "jarak_tp",
                      "sl_method", "atr_value", "pesan"):
            self.assertIn(field, res, f"Field '{field}' harus ada di return dict")


# =============================================================================
# TEST CLASS 4: ERROR HANDLING
# =============================================================================

class TestLevelReferensiNoneAtauHilang(unittest.TestCase):
    """
    Test 6: Level referensi None/hilang di strategy_result -> valid=False, tidak crash.
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        self.df = _make_df(atr=self.atr, base_price=self.entry)

    def test_invalidation_level_none_range_reversal(self):
        """invalidation_level=None -> valid=False, tidak crash."""
        strategy_result = {
            "terpenuhi"      : True,
            "invalidation_level": None,   # None
        }
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=strategy_result,
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])

    def test_field_hilang_di_strategy_result(self):
        """Field tidak ada sama sekali -> valid=False, tidak crash."""
        strategy_result = {
            "terpenuhi": True,
            # "invalidation_level" tidak ada sama sekali
        }
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=strategy_result,
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])

    def test_invalidation_level_sl_none_breakout(self):
        """invalidation_level_sl=None untuk BREAKOUT_RETEST -> valid=False, tidak crash."""
        strategy_result = {
            "terpenuhi"            : True,
            "invalidation_level_sl": None,
        }
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="BREAKOUT_RETEST",
            strategy_result=strategy_result,
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])

    def test_strategy_result_dict_kosong(self):
        """strategy_result kosong -> valid=False, tidak crash."""
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="TREND_FOLLOWING",
            strategy_result={},
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])


class TestStrategyNameTidakDikenal(unittest.TestCase):
    """
    Test 7: strategy_name tidak dikenal -> valid=False, tidak crash.
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        self.df = _make_df(atr=self.atr, base_price=self.entry)

    def test_strategy_name_tidak_dikenal(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="UNKNOWN_STRATEGY",
            strategy_result={"invalidation_level": 1990.0},
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])

    def test_skip_reason_menyebut_strategy_name(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="SCALPING_V99",
            strategy_result={"invalidation_level": 1990.0},
        )
        self.assertFalse(res["valid"])
        self.assertIn("SCALPING_V99", res["skip_reason"],
                      "skip_reason harus menyebut strategy_name yang tidak dikenal")

    def test_string_kosong_strategy_name(self):
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="",
            strategy_result={},
        )
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["skip_reason"])


class TestZoneNoneRangeReversal(unittest.TestCase):
    """
    Test 8: zone tidak diberikan (None) untuk RANGE_REVERSAL -> Langkah 3
    dilewati secara graceful (tidak crash, tp_capped=False).
    """

    def setUp(self):
        self.atr = 5.0
        self.entry = 2000.0
        self.df = _make_df(atr=self.atr, base_price=self.entry)
        self.invalidation_level = 1994.0
        self.strategy_result = _make_range_reversal_result(self.invalidation_level)

    def test_zone_none_tidak_crash(self):
        """zone=None untuk RANGE_REVERSAL harus berjalan tanpa crash."""
        try:
            res = calculate_regime_sl_tp(
                df_m5=self.df, entry=self.entry, arah="BUY",
                strategy_name="RANGE_REVERSAL",
                strategy_result=self.strategy_result,
                zone=None,  # zone tidak diberikan
            )
        except Exception as e:
            self.fail(f"zone=None menyebabkan crash: {e}")

    def test_zone_none_valid_true(self):
        """zone=None -> kalkulasi tetap valid (hanya Langkah 3 yang dilewati)."""
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=None,
        )
        self.assertTrue(res["valid"],
                        f"Harusnya valid=True meski zone=None. skip_reason: {res.get('skip_reason')}")

    def test_zone_none_tp_capped_false(self):
        """zone=None -> tp_capped harus False (Langkah 3 dilewati)."""
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=None,
        )
        self.assertTrue(res["valid"])
        self.assertFalse(res["tp_capped"],
                         "tp_capped harus False jika zone=None")

    def test_zone_none_keterangan_regime_ada(self):
        """zone=None -> keterangan_regime tetap terisi."""
        res = calculate_regime_sl_tp(
            df_m5=self.df, entry=self.entry, arah="BUY",
            strategy_name="RANGE_REVERSAL",
            strategy_result=self.strategy_result,
            zone=None,
        )
        self.assertTrue(res["valid"])
        self.assertIsNotNone(res["keterangan_regime"])
        self.assertIn("zone=None", res["keterangan_regime"])


# =============================================================================
# TEST CLASS 5: STRATEGY_LEVEL_FIELD_MAP
# =============================================================================

class TestStrategyLevelFieldMap(unittest.TestCase):
    """Verifikasi mapping field sesuai spec."""

    def test_range_reversal_field(self):
        self.assertEqual(STRATEGY_LEVEL_FIELD_MAP["RANGE_REVERSAL"], "invalidation_level")

    def test_breakout_retest_field(self):
        self.assertEqual(STRATEGY_LEVEL_FIELD_MAP["BREAKOUT_RETEST"], "invalidation_level_sl")

    def test_trend_following_field(self):
        self.assertEqual(STRATEGY_LEVEL_FIELD_MAP["TREND_FOLLOWING"], "invalidation_level_sl")

    def test_tiga_strategi_terdaftar(self):
        self.assertEqual(len(STRATEGY_LEVEL_FIELD_MAP), 3)


if __name__ == "__main__":
    unittest.main()
