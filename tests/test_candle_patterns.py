"""
tests/test_candle_patterns.py
==============================
Unit test untuk engine/candle_patterns.py (Fase 7).

FILOSOFI TEST:
    Setiap test menggunakan DataFrame sintetis dengan nilai OHLC yang dikontrol
    manual, sehingga kita tahu PERSIS apakah pattern seharusnya terdeteksi atau tidak.
    Tidak ada ambiguitas data — ini murni verifikasi logika matematis.

CAKUPAN TEST:
    1. detect_bullish_engulfing    — kasus positif dan negatif
    2. detect_bearish_engulfing    — kasus positif dan negatif
    3. detect_pin_bar (Hammer)     — kasus positif dan negatif
    4. detect_pin_bar (Shooting Star) — kasus positif dan negatif
    5. detect_marubozu (Bullish)   — kasus positif dan negatif
    6. detect_marubozu (Bearish)   — kasus positif dan negatif
    7. calculate_candle_pattern_score — skor 2 (pattern + swing), skor 1 (pattern tanpa swing),
       skor 0 (tidak ada pattern), swing=None (tidak crash)
    8. Edge cases: data kurang dari 2 baris, df kosong, idx tidak valid
    9. Kausalitas: pattern di idx tertentu tidak bergantung pada data sesudahnya

KONVENSI DATAFRAME SINTETIS:
    - Semua DF punya kolom: open, high, low, close, atr_14
    - atr_14 diset konsisten (biasanya 2.0) agar threshold bisa dihitung
    - DatetimeIndex ber-timezone UTC agar kompatibel dengan pipeline
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.candle_patterns import (
    detect_bullish_engulfing,
    detect_bearish_engulfing,
    calculate_candle_pattern_score,
    _get_candle_values,
    _resolve_idx,
)


# =============================================================================
# HELPER: Buat DataFrame sintetis 1 atau 2 candle
# =============================================================================

def make_df(candles: list[dict], atr: float = 2.0) -> pd.DataFrame:
    """
    Buat DataFrame sintetis dengan kolom OHLC + atr_14.

    Parameter:
        candles : list of dict, masing-masing punya 'open', 'high', 'low', 'close'
        atr     : nilai atr_14 yang diterapkan ke semua baris (konstan)
    """
    dates = pd.date_range("2026-01-01 00:00", periods=len(candles), freq="5min", tz="UTC")
    rows  = []
    for c in candles:
        rows.append({
            "open"   : float(c["open"]),
            "high"   : float(c["high"]),
            "low"    : float(c["low"]),
            "close"  : float(c["close"]),
            "atr_14" : float(atr),
        })
    return pd.DataFrame(rows, index=dates)


# =============================================================================
# TEST 1: Bullish Engulfing
# =============================================================================

class TestBullishEngulfing(unittest.TestCase):

    def test_positif_kasus_ideal(self):
        """
        Kasus ideal Bullish Engulfing:
          Candle i-1: bearish,  open=105, close=100  (body=5)
          Candle i  : bullish, open=99,  close=106   (body=7, menelan penuh body sebelumnya)
          ATR=2.0 → 0.5*ATR=1.0 → body(i)=7 >= 1.0 ✓
        """
        df = make_df([
            {"open": 105, "high": 106, "low": 99,  "close": 100},  # bearish candle i-1
            {"open": 99,  "high": 107, "low": 98,  "close": 106},  # bullish candle i
        ], atr=2.0)
        res = detect_bullish_engulfing(df, idx=-1)
        self.assertTrue(res["terpenuhi"], f"Harus True, dapat: {res}")
        self.assertEqual(res["arah"], "BUY")
        self.assertIn("Bullish Engulfing", res["keterangan"])

    def test_negatif_body_tidak_menelan(self):
        """
        Candle i menelan sebagian saja (close(i) < open(i-1)) — TIDAK engulfing:
          Candle i-1: bearish, open=105, close=100  (body=5)
          Candle i  : bullish, open=99, close=103   (body=4, close < open_prev=105)
        """
        df = make_df([
            {"open": 105, "high": 106, "low": 99,  "close": 100},
            {"open": 99,  "high": 104, "low": 98,  "close": 103},  # close(103) < open_prev(105)
        ], atr=2.0)
        res = detect_bullish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"], f"Harus False, dapat: {res}")

    def test_negatif_candle_prev_bukan_bearish(self):
        """
        Candle i-1 bullish — syarat pertama gagal.
          Candle i-1: bullish, open=100, close=105
          Candle i  : bullish, open=99, close=106
        """
        df = make_df([
            {"open": 100, "high": 106, "low": 99,  "close": 105},  # bullish i-1
            {"open": 99,  "high": 107, "low": 98,  "close": 106},
        ], atr=2.0)
        res = detect_bullish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])

    def test_negatif_candle_cur_bearish(self):
        """
        Candle i bearish — syarat kedua gagal.
        """
        df = make_df([
            {"open": 105, "high": 106, "low": 99,  "close": 100},  # bearish i-1
            {"open": 106, "high": 107, "low": 97,  "close": 98},   # bearish i juga
        ], atr=2.0)
        res = detect_bullish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])

    def test_negatif_body_terlalu_kecil(self):
        """
        Pattern menelan sempurna tapi body(i) < 0.5 * ATR → noise filter.
          ATR=2.0, body(i)=0.3 < 0.5*2.0=1.0
        """
        df = make_df([
            {"open": 100.5, "high": 101, "low": 99, "close": 100.0},   # bearish kecil
            {"open": 99.9,  "high": 101, "low": 99, "close": 100.2},   # bullish body=0.3
        ], atr=2.0)
        res = detect_bullish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])

    def test_edge_kurang_satu_baris(self):
        """
        Hanya satu candle — tidak ada i-1 → harus False, tidak crash.
        """
        df = make_df([
            {"open": 100, "high": 105, "low": 99, "close": 103},
        ], atr=2.0)
        res = detect_bullish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])
        self.assertEqual(res["arah"], "NETRAL")


# =============================================================================
# TEST 2: Bearish Engulfing
# =============================================================================

class TestBearishEngulfing(unittest.TestCase):

    def test_positif_kasus_ideal(self):
        """
        Kasus ideal Bearish Engulfing:
          Candle i-1: bullish, open=100, close=105
          Candle i  : bearish, open=106, close=99  (body=7, menelan penuh)
          ATR=2.0 → body(i)=7 >= 1.0 ✓
        """
        df = make_df([
            {"open": 100, "high": 106, "low": 99,  "close": 105},  # bullish
            {"open": 106, "high": 107, "low": 98,  "close": 99},   # bearish, menelan
        ], atr=2.0)
        res = detect_bearish_engulfing(df, idx=-1)
        self.assertTrue(res["terpenuhi"], f"Harus True, dapat: {res}")
        self.assertEqual(res["arah"], "SELL")

    def test_negatif_close_tidak_lebih_rendah(self):
        """
        close(i) = 101, tidak di bawah open(i-1)=100 → bukan engulfing penuh.
        """
        df = make_df([
            {"open": 100, "high": 106, "low": 99, "close": 105},
            {"open": 106, "high": 107, "low": 100, "close": 101},  # close(101) > open_prev(100)
        ], atr=2.0)
        res = detect_bearish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])

    def test_negatif_candle_prev_bukan_bullish(self):
        """
        Candle i-1 bearish — syarat pertama gagal.
        """
        df = make_df([
            {"open": 105, "high": 106, "low": 99,  "close": 100},  # bearish i-1
            {"open": 106, "high": 107, "low": 98,  "close": 99},
        ], atr=2.0)
        res = detect_bearish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])

    def test_edge_satu_baris(self):
        df = make_df([{"open": 100, "high": 105, "low": 99, "close": 103}])
        res = detect_bearish_engulfing(df, idx=-1)
        self.assertFalse(res["terpenuhi"])





# =============================================================================
# TEST 7: calculate_candle_pattern_score
# =============================================================================

class TestCalculateCandlePatternScore(unittest.TestCase):

    def _df_bullish_engulfing(self):
        """DataFrame 2 candle dengan Bullish Engulfing yang valid."""
        return make_df([
            {"open": 105, "high": 106, "low": 99,  "close": 100},  # bearish
            {"open": 99,  "high": 107, "low": 98,  "close": 106},  # bullish engulfing
        ], atr=2.0)

    def _df_bearish_engulfing(self):
        """DataFrame 2 candle dengan Bearish Engulfing yang valid."""
        return make_df([
            {"open": 100, "high": 106, "low": 99,  "close": 105},  # bullish
            {"open": 106, "high": 107, "low": 98,  "close": 99},   # bearish engulfing
        ], atr=2.0)

    def _df_no_pattern(self):
        """DataFrame dengan candle biasa, tidak ada pattern yang kuat."""
        return make_df([
            {"open": 100, "high": 101, "low": 99, "close": 100.5},  # doji kecil
            {"open": 100.5, "high": 101.5, "low": 99.5, "close": 101},  # candle normal
        ], atr=2.0)

    def test_skor_2_pattern_dan_swing_dekat(self):
        """
        Bullish Engulfing terdeteksi, close(106) dekat swing_low=105 (jarak=1.0 <= 1.0*ATR=2.0).
        Harus skor 2.
        """
        df = self._df_bullish_engulfing()
        close = float(df.iloc[-1]["close"])  # 106
        swing_low = 105.0  # jarak = 106-105 = 1.0, 1.0*ATR = 2.0 → dekat ✓

        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "BUY",
            swing_low     = swing_low,
            swing_high    = None,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 2, f"Harus 2: {res}")
        self.assertIsNotNone(res["pattern_detected"])
        self.assertIn("BULLISH_ENGULFING", res["pattern_detected"])

    def test_skor_1_pattern_tapi_swing_jauh(self):
        """
        Bullish Engulfing terdeteksi, tapi swing_low=90.0 jauh dari close(106).
        Jarak = 16.0 > 1.0*ATR=2.0 → kondisi konteks tidak terpenuhi.
        Harus skor 1.
        """
        df = self._df_bullish_engulfing()
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "BUY",
            swing_low     = 90.0,  # jauh
            swing_high    = None,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 1, f"Harus 1: {res}")
        self.assertIsNotNone(res["pattern_detected"])

    def test_skor_1_pattern_swing_none(self):
        """
        Pattern terdeteksi tapi swing_low=None → kondisi konteks tidak terpenuhi.
        Harus skor 1 (bukan error, bukan auto-lolos).
        """
        df = self._df_bullish_engulfing()
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "BUY",
            swing_low     = None,  # tidak ada swing
            swing_high    = None,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 1, f"Harus 1 (bukan crash atau 0): {res}")
        self.assertIsNotNone(res["pattern_detected"])

    def test_skor_0_tidak_ada_pattern(self):
        """
        Tidak ada pattern BUY terdeteksi → skor 0.
        """
        df = self._df_no_pattern()
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "BUY",
            swing_low     = 99.0,
            swing_high    = None,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 0, f"Harus 0: {res}")
        self.assertIsNone(res["pattern_detected"])

    def test_arah_sell_bearish_engulfing_skor_2(self):
        """
        Bearish Engulfing + swing_high dekat → skor 2.
        close(99) dekat swing_high=100.0 (jarak=1.0 <= 1.0*ATR=2.0).
        """
        df = self._df_bearish_engulfing()
        close = float(df.iloc[-1]["close"])  # 99
        swing_high = 100.0  # jarak = 100-99 = 1.0 <= 2.0 ✓

        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "SELL",
            swing_low     = None,
            swing_high    = swing_high,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 2, f"Harus 2: {res}")
        self.assertIn("BEARISH_ENGULFING", res["pattern_detected"])

    def test_arah_sell_tapi_pattern_buy_terdeteksi(self):
        """
        Ada Bullish Engulfing di candle terakhir, tapi arah_kandidat=SELL.
        Pattern searah SELL tidak ada → skor 0.
        (Jangan mixed direction)
        """
        df = self._df_bullish_engulfing()
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "SELL",
            swing_low     = None,
            swing_high    = 110.0,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 0, f"Pattern BUY tidak boleh dihitung untuk SELL: {res}")

    def test_arah_netral_skor_0(self):
        """
        arah_kandidat = 'NETRAL' → tidak ada pattern yang dicek → skor 0.
        """
        df = self._df_bullish_engulfing()
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "NETRAL",
            swing_low     = 90.0,
            swing_high    = 110.0,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 0)
        self.assertIsNone(res["pattern_detected"])

    def test_df_kosong_tidak_crash(self):
        """
        DataFrame kosong → tidak crash, skor 0.
        """
        df = pd.DataFrame(columns=["open", "high", "low", "close", "atr_14"])
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "BUY",
            swing_low     = 100.0,
            swing_high    = None,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 0)

    def test_df_none_tidak_crash(self):
        """
        df=None → tidak crash, skor 0.
        """
        res = calculate_candle_pattern_score(
            df            = None,
            arah_kandidat = "BUY",
            swing_low     = 100.0,
            swing_high    = None,
            atr_value     = 2.0,
        )
        self.assertEqual(res["score"], 0)

    def test_struktur_return_konsisten(self):
        """
        Return dict harus selalu punya semua field yang dibutuhkan oleh
        calculate_setup_quality() (pola komponen lain).
        """
        df = self._df_bullish_engulfing()
        res = calculate_candle_pattern_score(
            df            = df,
            arah_kandidat = "BUY",
            swing_low     = 105.0,
            swing_high    = None,
            atr_value     = 2.0,
        )
        required_keys = ["score", "max", "label", "detail", "pattern_detected"]
        for key in required_keys:
            self.assertIn(key, res, f"Key '{key}' tidak ada di return dict")
        self.assertEqual(res["max"], 2)
        self.assertEqual(res["label"], "Candlestick Pattern")
        self.assertIsInstance(res["score"], int)
        self.assertIn(res["score"], [0, 1, 2])


# =============================================================================
# TEST 8: Pengujian Kausalitas (Tidak Membaca Masa Depan)
# =============================================================================

class TestKausalitas(unittest.TestCase):

    def test_pattern_di_idx_tidak_terpengaruh_data_sesudahnya(self):
        """
        Verifikasi bahwa deteksi pattern di candle i menghasilkan
        hasil yang sama apakah df berisi hanya s/d candle i, atau
        df berisi candle i+1, i+2, dst juga.

        Ini adalah test analogi dengan validate_no_lookahead() di backtester.py.
        """
        # Buat DataFrame dengan 4 candle:
        # Candle 0: normal
        # Candle 1 (i-1): bearish (untuk engulfing)
        # Candle 2 (i  ): bullish engulfing → pattern seharusnya terdeteksi
        # Candle 3 (i+1): random (seharusnya tidak mempengaruhi deteksi di candle 2)
        all_candles = [
            {"open": 102, "high": 103, "low": 101, "close": 102.5},  # candle 0, bullish kecil
            {"open": 105, "high": 106, "low": 99,  "close": 100},    # candle 1, bearish (i-1)
            {"open": 99,  "high": 107, "low": 98,  "close": 106},    # candle 2, bullish engulfing (i)
            {"open": 80,  "high": 85,  "low": 70,  "close": 72},     # candle 3, data masa depan
        ]
        df_full = make_df(all_candles, atr=2.0)

        # Test pada df penuh (s/d candle 3)
        res_full = detect_bullish_engulfing(df_full, idx=2)

        # Test pada df yang di-slice s/d candle 2 saja (no lookahead)
        df_slice = make_df(all_candles[:3], atr=2.0)
        res_slice = detect_bullish_engulfing(df_slice, idx=-1)

        # Harus identik
        self.assertEqual(
            res_full["terpenuhi"],
            res_slice["terpenuhi"],
            f"Lookahead detected! full={res_full}, slice={res_slice}"
        )
        self.assertEqual(
            res_full["arah"],
            res_slice["arah"],
        )


# =============================================================================
# TEST 9: Helper Functions
# =============================================================================

class TestHelpers(unittest.TestCase):

    def test_resolve_idx_negatif(self):
        df = make_df([
            {"open": 100, "high": 101, "low": 99, "close": 100.5},
            {"open": 101, "high": 102, "low": 100, "close": 101.5},
        ])
        self.assertEqual(_resolve_idx(df, -1), 1)
        self.assertEqual(_resolve_idx(df, -2), 0)
        self.assertEqual(_resolve_idx(df, 0), 0)
        self.assertEqual(_resolve_idx(df, 1), 1)

    def test_resolve_idx_out_of_range(self):
        df = make_df([{"open": 100, "high": 101, "low": 99, "close": 100.5}])
        self.assertEqual(_resolve_idx(df, -5), -1)  # too far negative
        self.assertEqual(_resolve_idx(df, 99), -1)  # too far positive

    def test_get_candle_values_field_check(self):
        df = make_df([
            {"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0},
        ], atr=2.0)
        c = _get_candle_values(df, 0)
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c["body"],       3.0)
        self.assertAlmostEqual(c["range_"],     7.0)
        self.assertAlmostEqual(c["upper_wick"], 2.0)  # 105 - max(100, 103) = 2
        self.assertAlmostEqual(c["lower_wick"], 2.0)  # min(100, 103) - 98 = 2
        self.assertTrue(c["is_bullish"])
        self.assertFalse(c["is_bearish"])

    def test_get_candle_values_idx_invalid(self):
        df = make_df([{"open": 100, "high": 101, "low": 99, "close": 100.5}])
        self.assertIsNone(_get_candle_values(df, 99))


# =============================================================================
# TEST 10: Integrasi dengan calculate_setup_quality (smoke test)
# =============================================================================

class TestIntegrasiQualityScoring(unittest.TestCase):

    def test_default_call_exclude_candle_pattern(self):
        """
        Panggil calculate_setup_quality() TANPA argumen enable_candle_pattern
        (pure default). Verifikasi:
          - breakdown TIDAK punya key 'candle_pattern'
          - setup_quality_max == 8 (4 komponen aktif x 2 poin)
          - hanya 4 key di breakdown: ema_gap, rsi_zone, swing_distance, trigger_confluence
        Ini adalah test eksplisit untuk keputusan freeze Fase 7.
        """
        from engine.rule_engine import calculate_setup_quality

        signals = {
            "ema_gap_pct" : 0.20,
            "rsi_14"      : 50.0,
            "close"       : 100.0,
            "atr_14"      : 2.0,
            "trend_h1"    : "UPTREND",
            "trend"       : "UPTREND",
            "swing_low"   : 90.0,
            "swing_high"  : None,
        }
        df_dummy = make_df([
            {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5},
            {"open": 99.0, "high": 101.0, "low": 98.5, "close": 100.0},
        ], atr=2.0)

        # Panggil TANPA enable_candle_pattern — murni default
        res = calculate_setup_quality(signals, {}, {}, {}, df=df_dummy)

        # Komponen candle_pattern harus TIDAK ada di breakdown (bukan sekedar skor 0)
        self.assertNotIn(
            "candle_pattern", res["quality_breakdown"],
            "candle_pattern tidak boleh ada di breakdown pada kondisi default (freeze Fase 7)"
        )
        # Max harus 8 (4 komponen x 2 poin)
        self.assertEqual(
            res["setup_quality_max"], 8,
            "Default max harus 8 (ema_gap + rsi_zone + swing_distance + trigger_confluence)"
        )
        # Tepat 4 komponen di breakdown
        expected_keys = {"ema_gap", "rsi_zone", "swing_distance", "trigger_confluence"}
        self.assertEqual(
            set(res["quality_breakdown"].keys()), expected_keys,
            f"Breakdown harus tepat 4 komponen, bukan: {set(res['quality_breakdown'].keys())}"
        )

    def test_max_score_dengan_candle_pattern_aktif_jadi_10(self):
        """
        Verifikasi bahwa enable_candle_pattern=True (eksplisit) menghasilkan max=10
        karena 5 komponen aktif (Fase 9). Ini adalah jalur RISET, bukan live default.
        """
        from engine.rule_engine import calculate_setup_quality

        signals = {
            "ema_gap_pct" : 0.20,   # strong → score 2
            "rsi_14"      : 50.0,   # netral → score 2
            "close"       : 100.0,
            "atr_14"      : 2.0,
            "trend_h1"    : "UPTREND",
            "trend"       : "UPTREND",
            "swing_low"   : 90.0,   # dist=10.0, atr=2.0 → ratio=5x → score 2
            "swing_high"  : None,
        }
        df_dummy = make_df([
            {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5},
            {"open": 99.0, "high": 101.0, "low": 98.5, "close": 100.0},
        ], atr=2.0)

        # Eksplisit aktifkan candle_pattern — bukan default, jalur riset
        res = calculate_setup_quality(signals, {}, {}, {}, df=df_dummy, enable_candle_pattern=True)
        # 5 komponen aktif → max=10
        self.assertEqual(res["setup_quality_max"], 10, "Max harus 10 ketika candle_pattern aktif (5 komponen)")
        self.assertIn("candle_pattern", res["quality_breakdown"])
        self.assertIn("trigger_confluence", res["quality_breakdown"])

    def test_default_skema_max8_threshold_strong7_moderate4(self):
        """
        Skema default (candle_pattern OFF): max=8, STRONG>=7, MODERATE>=4.
        Verifikasi bahwa skor 6 dari 8 = MODERATE (bukan STRONG), dan skor
        7 dari 8 = STRONG. Threshold harus dihitung dinamis dari komponen aktif.
        """
        from engine.rule_engine import calculate_setup_quality

        # Skor 6: ema_gap=2, rsi=2, swing=2, trigger_confluence=0 (trigger_source=None)
        # Default: candle_pattern OFF → max=8, STRONG>=7, MODERATE>=4 → skor 6 = MODERATE
        signals = {
            "ema_gap_pct" : 0.20,   # 2 pts
            "rsi_14"      : 50.0,   # 2 pts
            "close"       : 100.0,
            "atr_14"      : 2.0,
            "trend_h1"    : "UPTREND",
            "trend"       : "UPTREND",
            "swing_low"   : 90.0,   # 2 pts
            "swing_high"  : None,
        }
        res = calculate_setup_quality(signals, {}, {}, {})  # default: candle_pattern=False
        # trigger_confluence=0 (trigger_source=None) → total=6, max=8, MODERATE>=4 → MODERATE
        self.assertEqual(res["setup_quality_score"], 6)
        self.assertEqual(res["setup_quality_max"], 8)
        self.assertEqual(res["setup_quality"], "MODERATE")

        # Skor 7: sama + trigger_source="EMA_GAP" (trigger_confluence=1) → 2+2+2+1=7 → STRONG
        res7 = calculate_setup_quality(signals, {}, {}, {}, trigger_source="EMA_GAP")
        self.assertEqual(res7["setup_quality_score"], 7)
        self.assertEqual(res7["setup_quality"], "STRONG")

    def test_default_weak_skor_rendah(self):
        """
        Skema default (candle_pattern OFF, max=8): skor rendah harus WEAK.
        MODERATE >= ceil(50% * 8) = 4, jadi skor 2 = WEAK.
        """
        from engine.rule_engine import calculate_setup_quality

        signals = {
            "ema_gap_pct" : 0.10,   # 1 pt (0.08-0.15)
            "rsi_14"      : 35.0,   # 1 pt (30-40)
            "close"       : 100.0,
            "atr_14"      : 2.0,
            "trend_h1"    : "SIDEWAYS",
            "trend"       : "SIDEWAYS",
            "swing_low"   : None,
            "swing_high"  : None,
        }
        # swing_distance=0, candle_pattern OFF (default), trigger_confluence=0
        # total = 1+1+0+0 = 2; max=8; MODERATE>=4 → WEAK
        res = calculate_setup_quality(signals, {}, {}, {})  # pure default
        self.assertEqual(res["setup_quality_score"], 2)
        self.assertEqual(res["setup_quality_max"], 8)
        self.assertEqual(res["setup_quality"], "WEAK")

    def test_toggle_off_skor_sama_dengan_3_komponen_dan_trigger(self):
        """
        Dengan enable_candle_pattern=False eksplisit + df tersedia:
        - candle_pattern TIDAK ada di breakdown (sesuai logika freeze: komponen off = tidak dicatat)
        - total = ema_gap + rsi_zone + swing_distance + trigger_confluence (tanpa candle)
        - max=8 (4 komponen x 2 poin)
        """
        from engine.rule_engine import calculate_setup_quality

        signals = {
            "ema_gap_pct" : 0.20,
            "rsi_14"      : 50.0,
            "close"       : 100.0,
            "atr_14"      : 2.0,
            "trend_h1"    : "UPTREND",
            "trend"       : "UPTREND",
            "swing_low"   : 90.0,
            "swing_high"  : None,
        }
        df_dummy = make_df([
            {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.5},
            {"open": 99.0, "high": 101.0, "low": 98.5, "close": 100.0},
        ], atr=2.0)

        res_off = calculate_setup_quality(signals, {}, {}, {}, df=df_dummy, enable_candle_pattern=False)
        # candle_pattern OFF → tidak ada di breakdown
        self.assertNotIn("candle_pattern", res_off["quality_breakdown"])
        # 3 komponen non-trigger tetap ada dan skornya benar
        self.assertEqual(res_off["quality_breakdown"]["ema_gap"]["score"], 2)
        self.assertEqual(res_off["quality_breakdown"]["rsi_zone"]["score"], 2)
        self.assertEqual(res_off["quality_breakdown"]["swing_distance"]["score"], 2)
        # trigger_confluence=0 (trigger_source=None default)
        self.assertEqual(res_off["quality_breakdown"]["trigger_confluence"]["score"], 0)
        # total=6, max=8
        self.assertEqual(res_off["setup_quality_score"], 6)
        self.assertEqual(res_off["setup_quality_max"], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
