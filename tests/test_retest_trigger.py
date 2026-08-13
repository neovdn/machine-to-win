"""
tests/test_retest_trigger.py
============================
Test suite Fase 10: Retest Trigger.

STRUKTUR (staging pre-registered sebelum backtest):

    TAHAP 1A — Unit test _check_retest_trigger() (terisolasi, synthetic df)
        1. TestRetestValid_BUY        — retest BUY lengkap (breakout→touch→confirm)
        2. TestRetestInvalidasi       — harga kembali masuk zona → breakout invalid
        3. TestRetestWindowHabis      — window lookback habis tanpa retest touch
        4. TestRetestTidakAdaBreakout — tidak ada breakout event di window
        5. TestRetestBodyKecil        — touch ada tapi body idx terlalu kecil

    TAHAP 1B — Unit test evaluate_entry() dengan enable_retest_trigger=True
        6. TestEvaluateEntryRetestMode — integrate _check_retest_trigger ke pipeline

    TAHAP 1C — Key compatibility
        7. TestRetestKondisiDetailKeys — "retest_trigger" muncul saat enable_retest_trigger=True

    TAHAP 1D — Kausalitas end-to-end
        8. TestRetestTriggerCausalityEndToEnd — mutasi candle masa depan tidak mengubah keputusan

    TAHAP 1E — Regresi baseline
        9. TestRetestBaselineRegression — 249 trades identik saat enable_retest_trigger=False

PRINSIP:
    - Test kausalitas (1D) wajib pass sebelum backtest kalibrasi (Fase 11).
    - Baseline regression (1E) wajib pass — enable_retest_trigger=False harus
      menghasilkan angka persis sama dengan Fase 9 (249 trades, semua angka identik).
    - Semua synthetic df dibangun menggunakan atr_14 eksplisit untuk determinisme
      dan kecepatan. run_all_indicators() hanya dipakai untuk causality test (1D).
"""

import os
import sys
import math
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.rule_engine import (
    _check_retest_trigger,
    _RETEST_SWING_BUFFER,
    _RETEST_BODY_MIN_RATIO,
    evaluate_entry,
)


# =============================================================================
# HELPER FACTORY — membangun DataFrame sintetis deterministik
# =============================================================================

def _make_base_df(n=80, atr=2.0, price_base=2000.0, volume_ratio=1.5, freq="5min"):
    """
    DataFrame sintetis dengan kolom minimal yang dibutuhkan oleh
    _check_retest_trigger dan detect_consolidation_zone.

    Candle default: konsolidasi ketat di sekitar price_base.
        high = price_base + 0.5
        low  = price_base - 0.5
        open = price_base
        close= price_base
        atr_14 = atr (konstan)
        volume_ratio = volume_ratio (konstan)
    """
    dates = pd.date_range("2026-01-01 00:00", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "open":         [price_base] * n,
        "high":         [price_base + 0.5] * n,
        "low":          [price_base - 0.5] * n,
        "close":        [price_base] * n,
        "atr_14":       [atr] * n,
        "volume_ratio": [volume_ratio] * n,
        "tick_volume":  [1000] * n,
    }, index=dates)
    return df


def _set_candle(df, idx, *, open_=None, high=None, low=None, close=None,
                atr_14=None, volume_ratio=None):
    """Helper: atur nilai satu candle di index idx secara in-place."""
    if open_        is not None: df.at[df.index[idx], "open"]         = open_
    if high         is not None: df.at[df.index[idx], "high"]         = high
    if low          is not None: df.at[df.index[idx], "low"]          = low
    if close        is not None: df.at[df.index[idx], "close"]        = close
    if atr_14       is not None: df.at[df.index[idx], "atr_14"]       = atr_14
    if volume_ratio is not None: df.at[df.index[idx], "volume_ratio"] = volume_ratio


# =============================================================================
# TAHAP 1A — Unit test _check_retest_trigger()
# =============================================================================

class TestRetestValid_BUY(unittest.TestCase):
    """
    Skenario: retest BUY yang valid lengkap.

    Layout (80 candle, ATR=2.0):
        idx 0-24  : konsolidasi ketat (high=2000.5, low=1999.5, range=1.0 ≤ 5.0*ATR ✓)
        idx 25    : BREAKOUT BUY
                    open=2001.0, close=2005.0, high=2005.5, low=2001.0
                    body = 4.0 >= 0.8*2.0=1.6 ✓; close > resistance=2000.5 ✓
        idx 26-28 : pullback ke atas resistance (low=2001.3 ≈ resistance, tidak invalid)
        idx 29    : konfirmasi retest
                    open=2001.5, close=2003.5, high=2004.0, low=2001.4
                    touch: low=2001.4 ∈ [2000.5 - 0.6, 2000.5 + 0.6] = [1999.9, 2001.1]?
                    Tunggu — ATR=2.0, tol=0.3*2.0=0.6, resistance=2000.5
                    range = [1999.9, 2001.1]
                    low=2001.4 > 2001.1 → BELUM touch di idx=29.
                    Coba geser low pullback ke 2000.7 (dalam range).

    Koreksi: idx 26-28 punya low=2000.7 → touch di idx=26.
    idx 29: konfirmasi: close=2003.5 > 2000.5 ✓, body=|2003.5-2001.5|=2.0 >= 0.3*2.0=0.6 ✓.
    """

    @classmethod
    def setUpClass(cls):
        """Build synthetic df dan tempatkan breakout/retest pattern."""
        df = _make_base_df(n=80, atr=2.0, price_base=2000.0)

        # Candle 0-24: konsolidasi ketat (sudah dibuat oleh _make_base_df)
        # Resistance = max(high[0:25]) = 2000.5

        # Candle 25: breakout BUY
        _set_candle(df, 25,
                    open_=2001.0, high=2005.5, low=2001.0, close=2005.0)

        # Candle 26-28: pullback dengan low=2000.7 (within [1999.9, 2001.1])
        for i in [26, 27, 28]:
            _set_candle(df, i,
                        open_=2002.5, high=2003.0, low=2000.7, close=2002.5)

        # Candle 29: konfirmasi (close > resistance, body besar)
        _set_candle(df, 29,
                    open_=2001.5, high=2004.0, low=2001.4, close=2003.5)

        cls.df = df

    def test_terpenuhi_true(self):
        """Valid BUY retest harus menghasilkan terpenuhi=True."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertTrue(
            result["terpenuhi"],
            f"Expected terpenuhi=True, got False. keterangan={result['keterangan']}"
        )

    def test_arah_buy(self):
        """Arah harus BUY."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertEqual(result["arah"], "BUY")

    def test_breakout_idx_ditemukan(self):
        """breakout_idx harus ditemukan (not None)."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertIsNotNone(result["breakout_idx"])

    def test_breakout_level_adalah_resistance(self):
        """breakout_level harus sekitar resistance = 2000.5."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertAlmostEqual(result["breakout_level"], 2000.5, places=2)

    def test_candles_since_breakout(self):
        """candles_since_breakout = idx - breakout_idx."""
        result = _check_retest_trigger(self.df, idx=29)
        if result["breakout_idx"] is not None:
            expected = 29 - result["breakout_idx"]
            self.assertEqual(result["candles_since_breakout"], expected)

    def test_keterangan_tidak_kosong(self):
        """keterangan harus informatif."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertIn("BUY", result["keterangan"])
        self.assertGreater(len(result["keterangan"]), 20)


class TestRetestInvalidasi(unittest.TestCase):
    """
    Skenario: harga kembali masuk zona setelah breakout → breakout invalid.

    Layout sama dengan TestRetestValid_BUY, tapi candle 26 punya
    close = 1999.0 < resistance - 0.50 = 2000.5 - 0.50 = 2000.0 → INVALID.
    Karena tidak ada breakout lain dalam window yang valid, harus return False.
    """

    @classmethod
    def setUpClass(cls):
        df = _make_base_df(n=80, atr=2.0, price_base=2000.0)

        # Breakout di idx=25
        _set_candle(df, 25,
                    open_=2001.0, high=2005.5, low=2001.0, close=2005.0)

        # Candle 26: close=1999.0 < 2000.5-0.50=2000.0 → INVALIDASI
        _set_candle(df, 26,
                    open_=2000.5, high=2001.0, low=1998.5, close=1999.0)

        # Candle 27-28: pullback biasa
        for i in [27, 28]:
            _set_candle(df, i,
                        open_=2002.5, high=2003.0, low=2000.7, close=2002.5)

        # Candle 29: konfirmasi (tidak akan dipakai karena breakout sudah invalid)
        _set_candle(df, 29,
                    open_=2001.5, high=2004.0, low=2001.4, close=2003.5)

        cls.df = df

    def test_terpenuhi_false_karena_invalidasi(self):
        """Breakout yang ter-invalidasi harus menghasilkan terpenuhi=False."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertFalse(
            result["terpenuhi"],
            f"Expected terpenuhi=False (invalidated), got True. "
            f"keterangan={result['keterangan']}"
        )

    def test_arah_netral(self):
        """Arah harus NETRAL saat tidak ada retest valid."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertEqual(result["arah"], "NETRAL")


class TestRetestWindowHabis(unittest.TestCase):
    """
    Skenario: breakout ditemukan tapi tidak ada retest touch dalam window.

    Setelah breakout di idx=25, harga langsung naik ke 2010 tanpa pullback.
    Tidak ada candle yang low mendekati resistance=2000.5 (semuanya >> 2000.5+tol).
    """

    @classmethod
    def setUpClass(cls):
        df = _make_base_df(n=80, atr=2.0, price_base=2000.0)

        # Breakout di idx=25
        _set_candle(df, 25,
                    open_=2001.0, high=2006.0, low=2001.0, close=2005.5)

        # Candle 26-29: harga naik langsung, tidak ada pullback
        for i in [26, 27, 28, 29]:
            _set_candle(df, i,
                        open_=2008.0, high=2012.0, low=2007.5, close=2010.0)

        cls.df = df

    def test_terpenuhi_false_karena_no_touch(self):
        """Tanpa retest touch harus terpenuhi=False."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertFalse(result["terpenuhi"])

    def test_tidak_crash(self):
        """Fungsi tidak boleh raise exception."""
        try:
            result = _check_retest_trigger(self.df, idx=29)
        except Exception as e:
            self.fail(f"_check_retest_trigger raised exception: {e}")

    def test_return_dict_lengkap(self):
        """Return dict harus selalu punya semua key yang diperlukan."""
        result = _check_retest_trigger(self.df, idx=29)
        required_keys = {"terpenuhi", "arah", "keterangan", "breakout_idx",
                         "breakout_level", "candles_since_breakout"}
        self.assertEqual(set(result.keys()), required_keys)


class TestRetestTidakAdaBreakout(unittest.TestCase):
    """
    Skenario: tidak ada breakout event dalam window lookback.

    Seluruh window hanya konsolidasi — tidak ada candle yang menembus zona.
    Dengan lookback=5, window sangat sempit dan tidak mungkin ada breakout.
    """

    @classmethod
    def setUpClass(cls):
        # Hanya konsolidasi, tidak ada breakout
        cls.df = _make_base_df(n=80, atr=2.0, price_base=2000.0)

    def test_terpenuhi_false_karena_no_breakout(self):
        """Tanpa breakout event harus terpenuhi=False."""
        result = _check_retest_trigger(self.df, idx=30, retest_lookback_candles=5)
        self.assertFalse(result["terpenuhi"])

    def test_breakout_idx_none(self):
        """breakout_idx harus None saat tidak ada breakout."""
        result = _check_retest_trigger(self.df, idx=30, retest_lookback_candles=5)
        self.assertIsNone(result["breakout_idx"])

    def test_tidak_crash(self):
        """Fungsi tidak boleh raise exception."""
        try:
            _check_retest_trigger(self.df, idx=30, retest_lookback_candles=5)
        except Exception as e:
            self.fail(f"_check_retest_trigger raised exception: {e}")


class TestRetestBodyKecil(unittest.TestCase):
    """
    Skenario: retest touch ada, tapi body candle konfirmasi (idx) terlalu kecil.

    Breakout di 25, touch di 26, tapi candle 29 punya body=0.05 < 0.3*2.0=0.60.
    Harus return terpenuhi=False dengan penjelasan body terlalu kecil.
    """

    @classmethod
    def setUpClass(cls):
        df = _make_base_df(n=80, atr=2.0, price_base=2000.0)

        # Breakout di idx=25
        _set_candle(df, 25,
                    open_=2001.0, high=2005.5, low=2001.0, close=2005.0)

        # Candle 26: retest touch (low mendekati resistance=2000.5)
        _set_candle(df, 26,
                    open_=2002.5, high=2003.0, low=2000.7, close=2002.5)

        # Candle 27-28: normal di atas resistance
        for i in [27, 28]:
            _set_candle(df, i,
                        open_=2002.5, high=2003.0, low=2001.5, close=2002.5)

        # Candle 29: konfirmasi GAGAL — body terlalu kecil
        # close=2002.6, open=2002.55, body=0.05 < 0.3*2.0=0.6
        # close=2002.6 > resistance=2000.5 ✓ (close_ok)
        # body=0.05 < 0.6 ✗ (body_ok GAGAL)
        _set_candle(df, 29,
                    open_=2002.55, high=2002.8, low=2002.4, close=2002.6)

        cls.df = df

    def test_terpenuhi_false_karena_body_kecil(self):
        """Body terlalu kecil harus membuat terpenuhi=False."""
        result = _check_retest_trigger(self.df, idx=29)
        self.assertFalse(
            result["terpenuhi"],
            f"Expected terpenuhi=False (body kecil), got True. "
            f"keterangan={result['keterangan']}"
        )

    def test_keterangan_menyebut_body(self):
        """keterangan harus menyebut kegagalan body."""
        result = _check_retest_trigger(self.df, idx=29)
        # Keterangan harus menyebut 'body' atau 'ATR'
        self.assertTrue(
            "body" in result["keterangan"].lower() or
            "atr" in result["keterangan"].lower(),
            f"Expected 'body' atau 'ATR' in keterangan, got: {result['keterangan']}"
        )


# =============================================================================
# TAHAP 1B — evaluate_entry() dengan enable_retest_trigger=True
# =============================================================================

class TestEvaluateEntryRetestMode(unittest.TestCase):
    """
    Verifikasi bahwa evaluate_entry() berjalan tanpa error saat
    enable_retest_trigger=True, dan menghasilkan trigger_source yang valid.
    """

    @classmethod
    def setUpClass(cls):
        """Build df dengan pola retest BUY (sama seperti TestRetestValid_BUY)."""
        df = _make_base_df(n=80, atr=2.0, price_base=2000.0)

        # Sama persis dengan TestRetestValid_BUY — breakout di 25, touch di 26-28, confirm di 29
        _set_candle(df, 25, open_=2001.0, high=2005.5, low=2001.0, close=2005.0)
        for i in [26, 27, 28]:
            _set_candle(df, i, open_=2002.5, high=2003.0, low=2000.7, close=2002.5)
        _set_candle(df, 29, open_=2001.5, high=2004.0, low=2001.4, close=2003.5)

        cls.df = df

    def _build_signals_at_idx(self, idx):
        """Build signals dict dari candle idx."""
        row = self.df.iloc[idx]
        return {
            "time"         : self.df.index[idx],
            "close"        : float(row["close"]),
            "open"         : float(row["open"]),
            "high"         : float(row["high"]),
            "low"          : float(row["low"]),
            "ema_9"        : float(row["close"]) - 0.5,  # sintetis: ema_9 < ema_21 untuk WAIT
            "ema_21"       : float(row["close"]) + 0.5,  # sintetis
            "ema_gap_pct"  : -0.05,
            "rsi_14"       : 55.0,
            "atr_14"       : float(row["atr_14"]),
            "trend"        : "UPTREND",
            "trend_h1"     : "UPTREND",
            "ema_gap_pct"  : 0.1,
            "volume_ratio" : float(row["volume_ratio"]),
        }

    def test_tidak_crash_saat_retest_true(self):
        """evaluate_entry() tidak boleh raise exception saat enable_retest_trigger=True."""
        signals = self._build_signals_at_idx(29)
        try:
            result = evaluate_entry(
                signals,
                df=self.df,
                idx=29,
                enable_retest_trigger=True,
            )
        except Exception as e:
            self.fail(f"evaluate_entry() raised exception: {e}")

    def test_return_keys_lengkap(self):
        """Return dict harus punya semua key yang diperlukan."""
        signals = self._build_signals_at_idx(29)
        result = evaluate_entry(
            signals,
            df=self.df,
            idx=29,
            enable_retest_trigger=True,
        )
        required = {"keputusan", "arah", "trigger_source", "alasan_entry",
                    "alasan_wait", "kondisi_detail", "konfirmasi_terpenuhi",
                    "konfirmasi_dibutuhkan"}
        for k in required:
            self.assertIn(k, result, f"Missing key: {k}")

    def test_trigger_source_valid_ketika_retest_aktif(self):
        """trigger_source harus None, EMA_GAP, RETEST, atau BOTH — bukan BREAKOUT."""
        signals = self._build_signals_at_idx(29)
        result = evaluate_entry(
            signals,
            df=self.df,
            idx=29,
            enable_retest_trigger=True,
        )
        valid_sources = {None, "EMA_GAP", "RETEST", "BOTH"}
        self.assertIn(
            result["trigger_source"], valid_sources,
            f"trigger_source '{result['trigger_source']}' bukan salah satu dari {valid_sources}"
        )

    def test_trigger_source_tidak_breakout_saat_retest_mode(self):
        """Ketika mode retest aktif, trigger_source TIDAK boleh 'BREAKOUT'."""
        signals = self._build_signals_at_idx(29)
        result = evaluate_entry(
            signals,
            df=self.df,
            idx=29,
            enable_retest_trigger=True,
        )
        self.assertNotEqual(
            result["trigger_source"], "BREAKOUT",
            "trigger_source tidak boleh 'BREAKOUT' saat enable_retest_trigger=True "
            "(mode REPLACE — BREAKOUT bukan trigger penentu)"
        )


# =============================================================================
# TAHAP 1C — Key compatibility: kondisi_detail
# =============================================================================

class TestRetestKondisiDetailKeys(unittest.TestCase):
    """
    Verifikasi bahwa:
    - Saat enable_retest_trigger=True dan df tersedia: "retest_trigger" ADA di kondisi_detail.
    - Saat enable_retest_trigger=False (default): "retest_trigger" TIDAK ada di kondisi_detail.
    - "breakout_trigger" masih muncul saat enable_breakout_trigger=True (tanpa retest mode).
    """

    def _make_signals(self):
        """Signals sintetis minimal."""
        return {
            "time"         : pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
            "close"        : 2000.0,
            "open"         : 1999.8,
            "high"         : 2000.5,
            "low"          : 1999.5,
            "ema_9"        : 2000.5,
            "ema_21"       : 1999.5,
            "ema_gap_pct"  : 0.05,
            "rsi_14"       : 55.0,
            "atr_14"       : 2.0,
            "trend"        : "UPTREND",
            "trend_h1"     : "UPTREND",
            "volume_ratio" : 1.5,
        }

    def _make_df(self, n=50):
        """Minimal df dengan konsolidasi saja (tidak ada breakout/retest event)."""
        return _make_base_df(n=n, atr=2.0, price_base=2000.0)

    def test_retest_trigger_ada_saat_retest_mode(self):
        """'retest_trigger' harus ada di kondisi_detail saat enable_retest_trigger=True."""
        df  = self._make_df()
        idx = 30
        result = evaluate_entry(
            self._make_signals(),
            df=df,
            idx=idx,
            enable_retest_trigger=True,
        )
        self.assertIn(
            "retest_trigger", result["kondisi_detail"],
            "Key 'retest_trigger' tidak ditemukan di kondisi_detail saat enable_retest_trigger=True"
        )

    def test_retest_trigger_tidak_ada_saat_mode_normal(self):
        """'retest_trigger' TIDAK boleh ada saat enable_retest_trigger=False (default)."""
        result = evaluate_entry(self._make_signals())
        self.assertNotIn(
            "retest_trigger", result["kondisi_detail"],
            "Key 'retest_trigger' tidak boleh ada di kondisi_detail saat mode normal"
        )

    def test_keputusan_selalu_ada(self):
        """Keputusan harus selalu ada (BUY/SELL/WAIT)."""
        df  = self._make_df()
        result = evaluate_entry(
            self._make_signals(),
            df=df,
            idx=30,
            enable_retest_trigger=True,
        )
        self.assertIn(result["keputusan"], {"BUY", "SELL", "WAIT"})


# =============================================================================
# TAHAP 1D — Kausalitas end-to-end
# =============================================================================

class TestRetestTriggerCausalityEndToEnd(unittest.TestCase):
    """
    Membuktikan secara empiris bahwa _check_retest_trigger() dan evaluate_entry()
    adalah CAUSAL — keputusan di candle t TIDAK berubah jika candle t+1, t+2, ...
    dimutasi secara ekstrem.

    Pola test identik dengan TestBreakoutTriggerCausalityEndToEnd di
    tests/test_phase9_breakout.py:

        1. Buat df dengan run_all_indicators() (indikator realistis).
        2. Jalankan _check_retest_trigger(df, t=T) dan evaluate_entry() di t=T.
        3. Mutasi candle T+1..T+50 dengan nilai ekstrem (close=9999, high=10000, low=1).
        4. Hitung ulang — hasilnya HARUS identik 100%.

    Perbedaan dengan test_no_lookahead.py yang menguji indikator:
        Test ini fokus pada _check_retest_trigger() dan evaluate_entry() dalam mode
        retest. Kita verifikasi bahwa backward-search loop tidak membaca candle
        masa depan secara tidak sengaja.
    """

    @classmethod
    def setUpClass(cls):
        from engine.indicators import run_all_indicators

        # Dataset sintetis: 200 candle dengan fluktuasi acak + pola breakout-pullback
        np.random.seed(42)
        n = 200
        t = 120  # titik evaluasi

        # Bangun harga: 0-80 konsolidasi, 81-100 breakout area, 101+ trend
        prices = np.zeros(n)
        prices[:80]  = 2000.0 + np.random.randn(80) * 0.3
        prices[80:]  = 2003.0 + np.cumsum(np.random.randn(120) * 0.5)

        dates = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
        df_raw = pd.DataFrame({
            "open":        prices,
            "high":        prices + 1.5,
            "low":         prices - 1.5,
            "close":       prices + 0.2,
            "tick_volume": 1000,
            "spread":      10,
            "real_volume": 0,
        }, index=dates)

        cls.df_ind = run_all_indicators(df_raw.copy())
        cls.t      = t

    def _result_retest_at_t(self, df):
        """Jalankan _check_retest_trigger di t pada df yang diberikan."""
        return _check_retest_trigger(df, idx=self.t)

    def _result_entry_at_t(self, df):
        """Jalankan evaluate_entry() dalam mode retest di t."""
        from engine.indicators import get_latest_signals
        df_slice = df.iloc[:self.t + 1]
        signals  = get_latest_signals(df_slice)
        signals["trend_h1"] = "UPTREND"  # inject H1 bias
        return evaluate_entry(
            signals,
            df=df,
            idx=self.t,
            enable_retest_trigger=True,
        )

    def test_retest_causal_setelah_mutasi(self):
        """
        _check_retest_trigger(df, t) HARUS identik sebelum dan sesudah
        mutasi ekstrem pada candle t+1..t+50.
        """
        import copy

        # Jalankan pada df original
        result_orig = self._result_retest_at_t(self.df_ind.copy())

        # Mutasi candle t+1..t+50 dengan nilai ekstrem
        df_mutated = self.df_ind.copy()
        t_end = min(self.t + 51, len(df_mutated))
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("close")] = 9999.0
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("high")]  = 10000.0
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("low")]   = 1.0
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("open")]  = 5000.0

        # Jalankan pada df yang dimutasi
        result_mutated = self._result_retest_at_t(df_mutated)

        # Verifikasi identik
        self.assertEqual(
            result_orig["terpenuhi"], result_mutated["terpenuhi"],
            f"terpenuhi berubah setelah mutasi candle masa depan! "
            f"orig={result_orig['terpenuhi']}, mutated={result_mutated['terpenuhi']}"
        )
        self.assertEqual(
            result_orig["arah"], result_mutated["arah"],
            f"arah berubah setelah mutasi candle masa depan! "
            f"orig={result_orig['arah']}, mutated={result_mutated['arah']}"
        )
        self.assertEqual(
            result_orig["breakout_idx"], result_mutated["breakout_idx"],
            f"breakout_idx berubah setelah mutasi! "
            f"orig={result_orig['breakout_idx']}, mutated={result_mutated['breakout_idx']}"
        )

    def test_evaluate_entry_causal_retest_mode(self):
        """
        evaluate_entry() dalam mode retest HARUS menghasilkan keputusan identik
        sebelum dan sesudah mutasi candle masa depan.
        """
        # Jalankan pada df original
        result_orig = self._result_entry_at_t(self.df_ind.copy())

        # Mutasi candle t+1..t+50
        df_mutated = self.df_ind.copy()
        t_end = min(self.t + 51, len(df_mutated))
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("close")] = 9999.0
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("high")]  = 10000.0
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("low")]   = 1.0
        df_mutated.iloc[self.t + 1 : t_end, df_mutated.columns.get_loc("open")]  = 5000.0

        # Jalankan pada df yang dimutasi
        result_mutated = self._result_entry_at_t(df_mutated)

        self.assertEqual(
            result_orig["keputusan"], result_mutated["keputusan"],
            f"Keputusan berubah setelah mutasi candle masa depan! "
            f"orig={result_orig['keputusan']}, mutated={result_mutated['keputusan']}"
        )
        self.assertEqual(
            result_orig["trigger_source"], result_mutated["trigger_source"],
            f"trigger_source berubah setelah mutasi! "
            f"orig={result_orig['trigger_source']}, mutated={result_mutated['trigger_source']}"
        )


# =============================================================================
# TAHAP 1E — Regresi baseline (249 trades)
# =============================================================================

class TestRetestBaselineRegression(unittest.TestCase):
    """
    Verifikasi bahwa run_backtest() dengan enable_retest_trigger=False (default)
    menghasilkan angka baseline persis sama seperti Fase 9 (249 trades).

    Ini adalah REGRESSION GATE — jika test ini fail, berarti Fase 10 telah
    mengubah perilaku saat retest NONAKTIF, yang merupakan bug.

    Angka baseline: 249 trades (sama persis dengan test_backtester.py).
    """

    def test_enable_retest_false_menghasilkan_baseline_249(self):
        """
        run_backtest() dengan enable_retest_trigger=False HARUS menghasilkan
        persis 249 trades — identik dengan hasil sebelum Fase 10.
        """
        m5_path = os.path.join(ROOT_DIR, "data", "historical",
                               "XAUUSD_M5_2026-01-01_2026-07-25.csv")
        h1_path = os.path.join(ROOT_DIR, "data", "historical",
                               "XAUUSD_H1_2026-01-01_2026-07-25.csv")

        if not os.path.exists(m5_path) or not os.path.exists(h1_path):
            self.skipTest(
                "File cache historis tidak ditemukan — skip baseline verification test."
            )

        from engine.backtester import run_backtest

        df_m5 = pd.read_csv(m5_path)
        df_m5["time"] = pd.to_datetime(df_m5["time"])
        df_m5.set_index("time", inplace=True)

        df_h1 = pd.read_csv(h1_path)
        df_h1["time"] = pd.to_datetime(df_h1["time"])
        df_h1.set_index("time", inplace=True)

        trades_df, summary = run_backtest(
            df_m5=df_m5,
            df_h1=df_h1,
            warm_up=100,
            max_candles=288,
            spread_pts=0.50,
            atr_multiplier=1.5,
            swing_lookback=50,
            swing_wing=5,
            rrr_min=2.0,
            swing_clamp_min_atr=0.0,
            swing_clamp_max_atr=999.0,
            enable_breakout_trigger=False,
            enable_retest_trigger=False,   # Fase 10: WAJIB False untuk regression test
            verbose=False,
        )

        self.assertEqual(
            summary["total_trades"], 249,
            f"REGRESSION FAILED: total_trades={summary['total_trades']}, expected=249. "
            f"Fase 10 telah mengubah trade set saat retest=False — ini bug!"
        )
        self.assertEqual(summary["tp_count"],     76)
        self.assertEqual(summary["sl_count"],    126)
        self.assertEqual(summary["no_hit_count"], 47)
        self.assertAlmostEqual(summary["win_rate"],         0.376, places=3)
        self.assertAlmostEqual(summary["no_hit_rate"],      0.189, places=3)
        self.assertAlmostEqual(summary["avg_rrr_realized"], 0.10,  places=2)
        self.assertAlmostEqual(summary["avg_candles_held"], 93.3,  places=1)
        self.assertAlmostEqual(summary["total_pnl_net"],    1674.5, places=1)
        self.assertAlmostEqual(summary["max_drawdown_net"], -957.1, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
