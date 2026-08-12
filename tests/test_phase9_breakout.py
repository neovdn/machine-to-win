"""
tests/test_phase9_breakout.py
==============================
Test suite Fase 9: Breakout Confirmation Trigger.

TAHAP 1A — Unit test _check_breakout_trigger() (terisolasi)
TAHAP 1B — Unit test evaluate_entry() skenario baru
TAHAP 1C — Kompatibilitas kondisi_detail key names (wajib stabil untuk web/app.py)
TAHAP 1D — Kausalitas end-to-end (mutasi candle masa depan tidak mengubah keputusan)

CATATAN PENTING:
    Test 1D membuktikan jalur penuh evaluate_entry() dengan breakout aktif tetap
    causal. Ini BERBEDA dari test Fase 8 yang menguji detect_consolidation_zone()
    secara terisolasi. Di sini kita membuktikan titik integrasi idx=i-1 di pipeline
    sesungguhnya tidak menyebabkan lookahead.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rule_engine import _check_breakout_trigger, evaluate_entry
from engine.zone_detector import detect_consolidation_zone
from engine.indicators import run_all_indicators


# =============================================================================
# HELPER: buat signals dict minimal yang valid
# =============================================================================

def _make_signals(
    trend_h1="UPTREND",
    trend_m5="UPTREND",
    close=2000.0,
    open_price=1999.0,
    ema_9=2001.0,
    ema_21=1998.0,
    ema_gap_pct=0.15,
    rsi_14=50.0,
    atr_14=2.0,
    volume_ratio=1.0,
) -> dict:
    """Buat signals dict minimal dengan nilai default yang menghasilkan BUY."""
    return {
        "time"        : datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        "trend"       : trend_m5,
        "trend_h1"    : trend_h1,
        "close"       : close,
        "open"        : open_price,
        "ema_9"       : ema_9,
        "ema_21"      : ema_21,
        "ema_gap_pct" : ema_gap_pct,
        "rsi_14"      : rsi_14,
        "atr_14"      : atr_14,
        "volume_ratio": volume_ratio,
    }


def _make_valid_zone(resistance=2001.0, support=1999.0, duration=12) -> dict:
    """Buat zone dict minimal yang valid."""
    return {
        "is_valid"       : True,
        "resistance"     : resistance,
        "support"        : support,
        "range_zone"     : resistance - support,
        "range_atr_ratio": (resistance - support) / 2.0,
        "duration"       : duration,
        "keterangan"     : f"KONSOLIDASI VALID: duration={duration} candle",
    }


def _make_invalid_zone() -> dict:
    """Buat zone dict dengan is_valid=False."""
    return {
        "is_valid"       : False,
        "resistance"     : None,
        "support"        : None,
        "range_zone"     : None,
        "range_atr_ratio": None,
        "duration"       : 5,
        "keterangan"     : "TIDAK VALID: duration=5 candle (< min 10)",
    }


# =============================================================================
# TAHAP 1A: Unit test _check_breakout_trigger()
# =============================================================================

class TestCheckBreakoutTrigger(unittest.TestCase):
    """
    Tahap 1A: unit test _check_breakout_trigger() secara terisolasi.
    """

    def test_buy_valid_volume_confirmation(self):
        """BUY valid: zona valid, close > resistance, volume_ratio >= 1.2"""
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        signals = _make_signals(close=2002.0, open_price=2001.5, atr_14=2.0, volume_ratio=1.3)
        result = _check_breakout_trigger(signals, zone)

        self.assertTrue(result["terpenuhi"])
        self.assertEqual(result["arah"], "BUY")
        self.assertTrue(result["konfirmasi_volume"])
        self.assertIn("BUY", result["keterangan"])
        self.assertIn("2002.00", result["keterangan"])
        self.assertIn("2001.00", result["keterangan"])

    def test_buy_valid_body_confirmation(self):
        """BUY valid: zona valid, close > resistance, body >= 0.8 * atr_14 (tanpa volume)"""
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # body = close - open = 2002.0 - 2000.2 = 1.8 >= 0.8 * 2.0 = 1.6
        # (gunakan 1.8 bukan 1.6 untuk menghindari float precision edge case)
        signals = _make_signals(close=2002.0, open_price=2000.2, atr_14=2.0, volume_ratio=0.5)
        result = _check_breakout_trigger(signals, zone)

        self.assertTrue(result["terpenuhi"])
        self.assertEqual(result["arah"], "BUY")
        self.assertFalse(result["konfirmasi_volume"])  # volume rendah, tidak konfirmasi
        self.assertTrue(result["konfirmasi_body"])     # body konfirmasi

    def test_buy_or_logic_not_and(self):
        """
        Konfirmasi OR bukan AND: hanya body cukup (tanpa volume) → tetap BUY valid.
        """
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # volume rendah (0.5 < 1.2), body = 2.0 >= 0.8 * 2.0 = 1.6
        signals = _make_signals(close=2003.0, open_price=2001.0, atr_14=2.0, volume_ratio=0.5)
        result = _check_breakout_trigger(signals, zone)

        self.assertTrue(result["terpenuhi"])
        self.assertEqual(result["arah"], "BUY")
        # Hanya body konfirmasi, volume tidak
        self.assertFalse(result["konfirmasi_volume"])
        self.assertTrue(result["konfirmasi_body"])

    def test_buy_fail_wick_only_not_close(self):
        """
        Gagal: high > resistance tapi close <= resistance.
        Breakout harus via CLOSE, bukan sekedar high/wick.
        """
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # close = 2000.9 TIDAK menembus resistance=2001.0
        # (high bisa saja di 2002.0 tapi kita tidak melihat high di _check_breakout_trigger)
        signals = _make_signals(close=2000.9, open_price=2000.0, atr_14=2.0, volume_ratio=1.5)
        result = _check_breakout_trigger(signals, zone)

        self.assertFalse(result["terpenuhi"])
        self.assertEqual(result["arah"], "NETRAL")

    def test_buy_fail_invalid_zone(self):
        """Gagal: zone is_valid=False → trigger otomatis False tanpa cek harga."""
        zone = _make_invalid_zone()
        signals = _make_signals(close=2005.0, volume_ratio=2.0)  # harga jauh di atas, tapi zona invalid
        result = _check_breakout_trigger(signals, zone)

        self.assertFalse(result["terpenuhi"])
        self.assertEqual(result["arah"], "NETRAL")
        self.assertIn("tidak valid", result["keterangan"].lower())

    def test_buy_fail_no_confirmation(self):
        """
        Gagal: close > resistance tapi tidak ada konfirmasi.
        (volume_ratio < 1.2 DAN body < 0.8 * atr_14)
        """
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # body = 2001.1 - 2001.05 = 0.05, jauh di bawah 0.8 * 2.0 = 1.6
        signals = _make_signals(close=2001.1, open_price=2001.05, atr_14=2.0, volume_ratio=0.5)
        result = _check_breakout_trigger(signals, zone)

        self.assertFalse(result["terpenuhi"])
        self.assertEqual(result["arah"], "NETRAL")
        self.assertIn("tidak ada konfirmasi", result["keterangan"])

    def test_sell_valid_volume_confirmation(self):
        """SELL valid: close < support, volume_ratio >= 1.2"""
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # close menembus support ke bawah
        signals = _make_signals(close=1998.0, open_price=1998.5, atr_14=2.0, volume_ratio=1.5)
        result = _check_breakout_trigger(signals, zone)

        self.assertTrue(result["terpenuhi"])
        self.assertEqual(result["arah"], "SELL")
        self.assertTrue(result["konfirmasi_volume"])

    def test_sell_valid_body_confirmation(self):
        """SELL valid: close < support, body >= 0.8 * atr_14 (tanpa volume)"""
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # body = 2000.5 - 1998.0 = 2.5 >= 0.8 * 2.0 = 1.6
        signals = _make_signals(close=1998.0, open_price=2000.5, atr_14=2.0, volume_ratio=0.3)
        result = _check_breakout_trigger(signals, zone)

        self.assertTrue(result["terpenuhi"])
        self.assertEqual(result["arah"], "SELL")
        self.assertFalse(result["konfirmasi_volume"])
        self.assertTrue(result["konfirmasi_body"])

    def test_inside_zone_netral(self):
        """Close masih di dalam zona → NETRAL."""
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        signals = _make_signals(close=2000.0, open_price=1999.5, atr_14=2.0, volume_ratio=2.0)
        result = _check_breakout_trigger(signals, zone)

        self.assertFalse(result["terpenuhi"])
        self.assertEqual(result["arah"], "NETRAL")

    def test_volume_ratio_none_graceful(self):
        """volume_ratio=None tidak crash, hanya body bisa jadi konfirmasi."""
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        # body besar, volume_ratio None
        signals = _make_signals(close=2003.0, open_price=2000.0, atr_14=2.0, volume_ratio=None)
        result = _check_breakout_trigger(signals, zone)

        # body = 3.0 >= 0.8 * 2.0 = 1.6 → konfirmasi body saja sudah cukup
        self.assertTrue(result["terpenuhi"])
        self.assertFalse(result["konfirmasi_volume"])
        self.assertTrue(result["konfirmasi_body"])

    def test_return_keys_complete(self):
        """Return dict selalu punya semua key yang diharapkan."""
        zone = _make_valid_zone()
        signals = _make_signals(close=2002.0, open_price=2001.0, volume_ratio=1.3)
        result = _check_breakout_trigger(signals, zone)

        expected_keys = {"terpenuhi", "arah", "keterangan", "konfirmasi_volume",
                         "konfirmasi_body", "zone_resistance", "zone_support"}
        self.assertTrue(expected_keys.issubset(result.keys()))


# =============================================================================
# TAHAP 1B: evaluate_entry() skenario baru (trigger_source)
# =============================================================================

class TestEvaluateEntryPhase9(unittest.TestCase):
    """
    Tahap 1B: evaluate_entry() dengan arsitektur Fase 9.
    Fokus pada trigger_source dan logika bias_h1 sebagai prasyarat wajib.
    """

    def _signals_buy(self, **kwargs) -> dict:
        """Signals default untuk BUY (bias H1 UP, EMA M5 UP)."""
        return _make_signals(trend_h1="UPTREND", trend_m5="UPTREND", **kwargs)

    def _signals_sell(self, **kwargs) -> dict:
        """Signals default untuk SELL (bias H1 DOWN, EMA M5 DOWN)."""
        return _make_signals(
            trend_h1="DOWNTREND", trend_m5="DOWNTREND",
            ema_9=1998.0, ema_21=2001.0, ema_gap_pct=-0.15, **kwargs
        )

    def test_ema_only_trigger_source(self):
        """Hanya EMA cocok → trigger_source='EMA_GAP', keputusan BUY"""
        signals = self._signals_buy(close=2000.0, open_price=1999.0, atr_14=2.0, volume_ratio=1.0)
        # Zone invalid → breakout tidak aktif
        zone = _make_invalid_zone()
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        # EMA M5 UPTREND searah dengan bias H1 UPTREND
        self.assertEqual(result["keputusan"], "BUY")
        self.assertEqual(result["trigger_source"], "EMA_GAP")

    def test_breakout_only_trigger_source(self):
        """
        Hanya Breakout cocok → trigger_source='BREAKOUT'.
        EMA M5 SIDEWAYS (tidak nyala), tapi breakout BUY valid.

        Gunakan volume_mode='IGNORE' untuk mengisolasi behavior breakout trigger
        tanpa distraksi volume filter (volume filter ada di lapisan berbeda).
        volume_ratio=1.25 untuk konfirmasi breakout (>= 1.2).
        """
        signals = _make_signals(
            trend_h1="UPTREND",
            trend_m5="SIDEWAYS",  # EMA tidak nyala
            close=2002.0,  # menembus resistance 2001.0
            open_price=2001.0,
            ema_9=2000.5, ema_21=2000.3,
            ema_gap_pct=0.01,  # gap terlalu tipis → SIDEWAYS
            rsi_14=50.0,
            atr_14=2.0,
            volume_ratio=1.25,  # konfirmasi breakout; dalam window filter (< 1.278)
        )
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        self.assertEqual(result["keputusan"], "BUY")
        self.assertEqual(result["trigger_source"], "BREAKOUT")

    def test_both_triggers_active(self):
        """Keduanya cocok → trigger_source='BOTH', keputusan BUY"""
        # volume_ratio=1.25: cukup untuk konfirmasi breakout (>= 1.2)
        # dan TIDAK di-blokir volume filter (upper bound = 1.278)
        signals = self._signals_buy(
            close=2002.0,   # menembus resistance 2001.0
            open_price=2001.0,
            atr_14=2.0,
            volume_ratio=1.25,
        )
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        self.assertEqual(result["keputusan"], "BUY")
        self.assertEqual(result["trigger_source"], "BOTH")

    def test_breakout_opposite_direction_does_not_block(self):
        """
        Breakout berlawanan arah dengan bias_h1 TIDAK memblokir entry dari EMA.
        bias_h1=UPTREND, ema nyala BUY, breakout SELL (close < support).
        Hasil: entry BUY dari EMA, breakout SELL diabaikan.

        Catatan: close=1998.0 (di bawah support 1999.0) artinya breakout SELL,
        tapi signals["trend"]="UPTREND" dan ema_9 > ema_21 → EMA cocok BUY.
        Bias H1 UPTREND → arah_kandidat=BUY → breakout SELL berlawanan → diabaikan.
        """
        signals = _make_signals(
            trend_h1="UPTREND",
            trend_m5="UPTREND",   # EMA M5 UPTREND → ema_cocok=True untuk BUY
            close=1998.0,         # menembus support 1999.0 ke bawah → breakout SELL
            open_price=1998.5,
            ema_9=2001.0,
            ema_21=1998.0,
            ema_gap_pct=0.15,
            atr_14=2.0,
            volume_ratio=1.25,    # konfirmasi breakout SELL; volume dalam window filter
            rsi_14=50.0,
        )
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        # bias_h1=UPTREND → arah_kandidat=BUY
        # EMA nyala BUY (searah) → ema_cocok=True
        # Breakout nyala SELL (berlawanan) → tidak memblokir, tidak masuk trigger_source
        self.assertEqual(result["keputusan"], "BUY")
        self.assertEqual(result["trigger_source"], "EMA_GAP")

    def test_bias_h1_sideways_always_wait(self):
        """
        bias_h1 SIDEWAYS → WAIT apapun kondisi trigger.
        Ini membuktikan bias_h1 adalah prasyarat wajib yang TIDAK bisa dibypass trigger.
        """
        signals = _make_signals(
            trend_h1="SIDEWAYS",  # prasyarat tidak terpenuhi
            trend_m5="UPTREND",   # EMA nyala
            close=2002.0,
            open_price=2001.0,
            atr_14=2.0,
            volume_ratio=1.5,
        )
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        self.assertEqual(result["keputusan"], "WAIT")
        self.assertIsNone(result["trigger_source"])

    def test_enable_breakout_false_zone_ignored(self):
        """
        enable_breakout_trigger=False → breakout tidak dievaluasi meski zone valid.
        Untuk Tahap 0 regression: output sama dengan baseline EMA-only.
        """
        signals = self._signals_buy(
            close=2002.0, open_price=2001.0, atr_14=2.0, volume_ratio=1.5
        )
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)

        result_with = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)
        result_without = evaluate_entry(signals, zone=zone, enable_breakout_trigger=False)

        # Keputusan sama (EMA masih nyala BUY)
        self.assertEqual(result_without["keputusan"], result_with["keputusan"])
        # Tapi trigger_source dengan enable=False tidak menangkap BOTH
        self.assertEqual(result_without["trigger_source"], "EMA_GAP")
        # Dengan enable=True dan breakout juga nyala → BOTH
        self.assertEqual(result_with["trigger_source"], "BOTH")

    def test_no_trigger_valid_gives_wait(self):
        """
        Tidak ada trigger valid (EMA SIDEWAYS, breakout zone invalid) → WAIT.
        """
        signals = _make_signals(
            trend_h1="UPTREND",
            trend_m5="SIDEWAYS",  # EMA tidak nyala
            close=2000.0,
            open_price=1999.5,
            ema_9=2000.1, ema_21=2000.05,
            ema_gap_pct=0.003,
        )
        zone = _make_invalid_zone()
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        self.assertEqual(result["keputusan"], "WAIT")
        self.assertIsNone(result["trigger_source"])

    def test_konfirmasi_semantics(self):
        """
        konfirmasi_terpenuhi = jumlah trigger cocok arah (0-2).
        konfirmasi_dibutuhkan = selalu 1.
        """
        # Skenario BOTH: 2 trigger cocok
        signals = self._signals_buy(close=2002.0, open_price=2001.0, atr_14=2.0, volume_ratio=1.5)
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        if result["trigger_source"] == "BOTH":
            self.assertEqual(result["konfirmasi_terpenuhi"], 2)
        self.assertEqual(result["konfirmasi_dibutuhkan"], 1)

    def test_sell_scenario_both_triggers(self):
        """SELL scenario: bias_h1 DOWN, EMA DOWN, breakout ke bawah → BOTH"""
        # volume_ratio=1.25: konfirmasi breakout (>= 1.2) dan tidak blokir volume filter (< 1.278)
        signals = _make_signals(
            trend_h1="DOWNTREND",
            trend_m5="DOWNTREND",
            close=1997.0,   # menembus support 1999.0
            open_price=1998.5,
            ema_9=1998.0, ema_21=2001.0,
            ema_gap_pct=-0.15,
            rsi_14=50.0,
            atr_14=2.0,
            volume_ratio=1.25,
        )
        zone = _make_valid_zone(resistance=2001.0, support=1999.0)
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        self.assertEqual(result["keputusan"], "SELL")
        self.assertEqual(result["trigger_source"], "BOTH")


# =============================================================================
# TAHAP 1C: Kompatibilitas kondisi_detail key names
# =============================================================================

class TestKondisiDetailKeyCompatibility(unittest.TestCase):
    """
    Tahap 1C: memverifikasi kondisi_detail selalu punya key yang diharapkan.
    Ini kritis untuk web/app.py::_build_result() yang membaca by-name.
    """

    REQUIRED_KEYS = {"bias_h1", "ema_trigger_m5", "rsi_filter"}

    def _assert_required_keys(self, result, msg=""):
        """Assert semua required keys ada di kondisi_detail."""
        detail = result.get("kondisi_detail", {})
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, detail, f"Key '{key}' hilang dari kondisi_detail. {msg}")

    def test_keys_present_with_breakout_enabled_valid_zone(self):
        """enable_breakout_trigger=True + zone valid → required keys tetap ada + breakout_trigger ada"""
        signals = _make_signals(trend_h1="UPTREND", trend_m5="UPTREND")
        zone = _make_valid_zone()
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=True)

        self._assert_required_keys(result, "enable=True, zone valid")
        # breakout_trigger BOLEH ada (zone valid, trigger aktif)
        self.assertIn("breakout_trigger", result["kondisi_detail"])

    def test_keys_present_with_breakout_disabled(self):
        """enable_breakout_trigger=False → required keys ada, breakout_trigger TIDAK ada"""
        signals = _make_signals(trend_h1="UPTREND", trend_m5="UPTREND")
        zone = _make_valid_zone()
        result = evaluate_entry(signals, zone=zone, enable_breakout_trigger=False)

        self._assert_required_keys(result, "enable=False")
        self.assertNotIn("breakout_trigger", result["kondisi_detail"])

    def test_keys_present_with_zone_none(self):
        """zone=None → required keys ada, breakout_trigger TIDAK ada"""
        signals = _make_signals(trend_h1="UPTREND", trend_m5="UPTREND")
        result = evaluate_entry(signals, zone=None, enable_breakout_trigger=True)

        self._assert_required_keys(result, "zone=None")
        self.assertNotIn("breakout_trigger", result["kondisi_detail"])

    def test_keys_present_sideways_wait(self):
        """bias_h1 SIDEWAYS (→ WAIT) → required keys tetap ada"""
        signals = _make_signals(trend_h1="SIDEWAYS", trend_m5="SIDEWAYS")
        result = evaluate_entry(signals, zone=None, enable_breakout_trigger=True)

        self.assertEqual(result["keputusan"], "WAIT")
        self._assert_required_keys(result, "WAIT scenario")

    def test_breakout_trigger_key_conditional(self):
        """
        breakout_trigger ada HANYA jika enable=True DAN zone tidak None.
        Semua kombinasi lain: breakout_trigger tidak ada.
        """
        signals = _make_signals(trend_h1="UPTREND", trend_m5="UPTREND")

        # enable=True, zone=None → tidak ada
        r1 = evaluate_entry(signals, zone=None, enable_breakout_trigger=True)
        self.assertNotIn("breakout_trigger", r1["kondisi_detail"])

        # enable=False, zone=valid → tidak ada
        r2 = evaluate_entry(signals, zone=_make_valid_zone(), enable_breakout_trigger=False)
        self.assertNotIn("breakout_trigger", r2["kondisi_detail"])

        # enable=True, zone=valid → ada
        r3 = evaluate_entry(signals, zone=_make_valid_zone(), enable_breakout_trigger=True)
        self.assertIn("breakout_trigger", r3["kondisi_detail"])

    def test_bias_h1_key_structure(self):
        """bias_h1 dict tetap punya field terpenuhi, arah, keterangan."""
        signals = _make_signals(trend_h1="UPTREND", trend_m5="UPTREND")
        result = evaluate_entry(signals)

        c_h1 = result["kondisi_detail"]["bias_h1"]
        self.assertIn("terpenuhi", c_h1)
        self.assertIn("arah", c_h1)
        self.assertIn("keterangan", c_h1)

    def test_ema_trigger_m5_key_structure(self):
        """ema_trigger_m5 dict tetap punya field terpenuhi, arah, keterangan."""
        signals = _make_signals(trend_h1="UPTREND", trend_m5="UPTREND")
        result = evaluate_entry(signals)

        c_m5 = result["kondisi_detail"]["ema_trigger_m5"]
        self.assertIn("terpenuhi", c_m5)
        self.assertIn("arah", c_m5)
        self.assertIn("keterangan", c_m5)

    def test_rsi_filter_key_structure(self):
        """rsi_filter dict tetap punya field memblokir, keterangan."""
        signals = _make_signals()
        result = evaluate_entry(signals)

        c_rsi = result["kondisi_detail"]["rsi_filter"]
        self.assertIn("memblokir", c_rsi)
        self.assertIn("keterangan", c_rsi)


# =============================================================================
# TAHAP 1D: Kausalitas end-to-end
# =============================================================================

class TestBreakoutTriggerCausalityEndToEnd(unittest.TestCase):
    """
    Tahap 1D: membuktikan jalur penuh evaluate_entry() dengan breakout aktif
    tetap causal. Mutasi candle masa depan (setelah titik evaluasi) tidak boleh
    mengubah keputusan atau trigger_source di titik evaluasi.

    Ini mengikuti pola tests/test_no_lookahead.py::test_future_candle_mutation_signal_immutability
    tapi mencakup titik integrasi detect_consolidation_zone() + evaluate_entry().
    """

    @classmethod
    def setUpClass(cls):
        """Buat DataFrame sintetis yang cukup panjang untuk evaluasi breakout."""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2026-01-01 00:00:00", periods=n, freq="5min", tz="UTC")

        # Buat harga dengan konsolidasi di awal lalu trend naik
        prices = np.zeros(n)
        prices[:150] = 2000.0 + np.random.randn(150) * 0.3  # konsolidasi sempit
        prices[150:] = 2002.0 + np.cumsum(np.random.randn(150) * 0.2)  # trend naik

        cls.df_raw = pd.DataFrame({
            "open"       : prices - 0.1,
            "high"       : prices + 0.5,
            "low"        : prices - 0.5,
            "close"      : prices,
            "tick_volume": np.random.randint(50, 200, n),
            "spread"     : 10,
            "real_volume": 0,
        }, index=dates)

    def _build_signals_at(self, df_ind: pd.DataFrame, i: int, trend_h1: str = "UPTREND") -> dict:
        """Buat signals dict dari baris ke-i DataFrame yang sudah dihitung indikatornya."""
        row = df_ind.iloc[i]
        return {
            "time"        : df_ind.index[i],
            "close"       : float(row["close"]),
            "open"        : float(row["open"]),
            "ema_9"       : float(row["ema_9"]),
            "ema_21"      : float(row["ema_21"]),
            "ema_gap_pct" : float(row["ema_gap_pct"]),
            "rsi_14"      : float(row["rsi_14"]),
            "atr_14"      : float(row["atr_14"]),
            "trend"       : str(row["trend"]),
            "trend_h1"    : trend_h1,
            "volume_ratio": float(row["volume_ratio"]) if not pd.isna(row.get("volume_ratio", float("nan"))) else 1.0,
        }

    def test_future_candle_mutation_does_not_change_decision(self):
        """
        Mutasi candle t+1 sampai t+30 tidak mengubah keputusan DAN trigger_source
        pada candle t. Ini membuktikan integrasi idx=i-1 di zona benar-benar causal.
        """
        t = 160  # titik evaluasi — setelah konsolidasi, dalam periode trend naik

        # Hitung indikator dari data original (hanya sampai t)
        df_ind_orig = run_all_indicators(self.df_raw.iloc[:t + 1].copy())

        # Hitung zona dari candle t-1 (seperti yang dilakukan backtester)
        zone_orig = detect_consolidation_zone(
            df_ind_orig, idx=t - 1,
            lookback=20,
            max_range_atr_ratio=2.5,
            min_duration_candles=10,
        )

        signals_orig = self._build_signals_at(df_ind_orig, t)
        decision_orig = evaluate_entry(
            signals_orig,
            zone=zone_orig,
            enable_breakout_trigger=True,
        )

        # Mutasi ekstrem: ubah candle t+1 sampai t+30
        df_mutated = self.df_raw.iloc[:t + 31].copy()
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("close")] = 9999.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("high")]  = 10000.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("low")]   = 1.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("open")]  = 9999.0

        # Hitung indikator dari data yang dimutasi
        df_ind_mutated = run_all_indicators(df_mutated)

        # Hitung zona dari candle t-1 di dataset yang dimutasi
        # (t-1 belum dimutasi, harus identik dengan orig)
        zone_mutated = detect_consolidation_zone(
            df_ind_mutated, idx=t - 1,
            lookback=20,
            max_range_atr_ratio=2.5,
            min_duration_candles=10,
        )

        # Verifikasi zona dari t-1 identik (candle t-1 tidak dimutasi)
        self.assertEqual(zone_orig["is_valid"], zone_mutated["is_valid"])
        if zone_orig["resistance"] is not None:
            self.assertAlmostEqual(
                zone_orig["resistance"], zone_mutated["resistance"], places=5,
                msg="resistance zona berubah setelah mutasi candle masa depan — ada lookahead!"
            )
            self.assertAlmostEqual(
                zone_orig["support"], zone_mutated["support"], places=5,
                msg="support zona berubah setelah mutasi candle masa depan — ada lookahead!"
            )

        # Ambil signals candle t dari dataset yang dimutasi
        signals_mutated = self._build_signals_at(df_ind_mutated, t)
        decision_mutated = evaluate_entry(
            signals_mutated,
            zone=zone_mutated,
            enable_breakout_trigger=True,
        )

        # KRITIS: keputusan dan trigger_source harus identik
        self.assertEqual(
            decision_orig["keputusan"],
            decision_mutated["keputusan"],
            f"Keputusan berubah setelah mutasi candle masa depan!\n"
            f"Original: {decision_orig['keputusan']}\n"
            f"Mutasi  : {decision_mutated['keputusan']}"
        )
        self.assertEqual(
            decision_orig["trigger_source"],
            decision_mutated["trigger_source"],
            f"trigger_source berubah setelah mutasi candle masa depan!\n"
            f"Original: {decision_orig['trigger_source']}\n"
            f"Mutasi  : {decision_mutated['trigger_source']}"
        )

    def test_indicator_at_t_unchanged_after_future_mutation(self):
        """
        Indikator (EMA, RSI, ATR) di candle t tidak berubah setelah mutasi candle t+1, t+2, ...
        Ini adalah prerequirement kausalitas yang sudah dibuktikan oleh validate_no_lookahead()
        tapi kita verifikasi ulang dalam konteks Fase 9.
        """
        t = 160
        df_ind_orig = run_all_indicators(self.df_raw.iloc[:t + 1].copy())

        df_mutated = self.df_raw.iloc[:t + 31].copy()
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("close")] = 9999.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("high")]  = 10000.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("low")]   = 1.0
        df_ind_mutated = run_all_indicators(df_mutated)

        row_orig    = df_ind_orig.iloc[t]
        row_mutated = df_ind_mutated.iloc[t]

        for col in ["ema_9", "ema_21", "rsi_14", "atr_14"]:
            self.assertAlmostEqual(
                float(row_orig[col]),
                float(row_mutated[col]),
                places=5,
                msg=f"Kolom {col} berubah di candle t setelah mutasi candle masa depan!"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
