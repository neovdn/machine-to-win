"""
tests/test_range_reversal.py
============================
Unit test untuk engine/strategies/range_reversal.py — Fase 15.

Cakupan test:
    1.  BUY valid (sweep+reclaim+rejection terpenuhi)
    2.  SELL valid (cerminan BUY di resistance)
    3.  Sweep tanpa reclaim (breakdown asli, bukan reversal)
    4.  Reclaim tanpa sweep cukup dalam (depth di bawah minimum)
    5.  Sweep+reclaim OK tapi rejection lemah (wick pendek / close posisi salah)
    6.  Entry di tengah range (tidak menyentuh boundary manapun)
    7.  Zone = None → tidak crash
    8.  Zone dengan resistance/support = None → tidak crash
    9.  Kolom df_m5 tidak lengkap → tidak crash
    10. Test kausalitas (WAJIB): mutasi baris SETELAH idx tidak mempengaruhi hasil

Semua DataFrame sintetis dibangun minimal — hanya kolom yang dibutuhkan fungsi.
"""

import copy
import pytest
import pandas as pd

from engine.strategies.range_reversal import (
    evaluate_range_reversal,
    _detect_rejection_wick,
    RANGE_REVERSAL_MIN_SWEEP_DEPTH_ATR,
    RANGE_REVERSAL_MIN_REJECTION_WICK_ATR,
    RANGE_REVERSAL_MIN_CLOSE_POSITION,
)


# =============================================================================
# HELPER BUILDER
# =============================================================================

def _buat_df_satu_candle(
    open_: float,
    high: float,
    low: float,
    close: float,
    atr: float,
) -> pd.DataFrame:
    """Buat DataFrame M5 dengan satu baris candle sintetis."""
    return pd.DataFrame([{
        "open"   : open_,
        "high"   : high,
        "low"    : low,
        "close"  : close,
        "atr_14" : atr,
    }])


def _buat_zone(support: float, resistance: float) -> dict:
    """Buat dict zone minimal dengan support dan resistance."""
    return {"support": support, "resistance": resistance}


def _buat_df_multi_candle(baris: list[dict]) -> pd.DataFrame:
    """
    Buat DataFrame multi-baris dari list dict.
    Tiap dict: {"open", "high", "low", "close", "atr_14"}
    """
    return pd.DataFrame(baris)


# =============================================================================
# PARAMETER CANDLE REFERENSI UNTUK BUY VALID
# =============================================================================
# Boundary:
#   support    = 2000.00
#   resistance = 2010.00
#   atr        = 5.00
#
# Syarat sweep:
#   low < support                 → low = 1999.00   ✓ (menembus 1.00 poin)
#   depth >= 0.1 * 5.0 = 0.5     → depth = 1.00    ✓
#   close >= support              → close = 2001.00 ✓
#
# Syarat rejection candle (BUY):
#   lower_wick = min(open, close) - low = min(2001.5, 2001.0) - 1999.0 = 2.0
#   batas_wick = 0.5 * 5.0 = 2.5  → GAGAL jika 2.0 < 2.5
#
# Agar lower_wick >= 2.5:
#   open = 2004.0, high = 2005.0, low = 1999.0, close = 2002.0
#   lower_wick = min(2004.0, 2002.0) - 1999.0 = 2002.0 - 1999.0 = 3.0 >= 2.5  ✓
#   range_total = 2005.0 - 1999.0 = 6.0
#   close_posisi = (2002.0 - 1999.0) / 6.0 = 3.0 / 6.0 = 0.5 >= 0.5          ✓

ATR_REF    = 5.0
SUPPORT    = 2000.0
RESISTANCE = 2010.0

BUY_OPEN  = 2004.0
BUY_HIGH  = 2005.0
BUY_LOW   = 1999.0     # 1.0 di bawah support → depth=1.0 >= 0.5 ✓
BUY_CLOSE = 2002.0


# =============================================================================
# TEST 1: BUY VALID
# =============================================================================

class TestBuyValid:
    """Candle yang memenuhi semua syarat BUY Range Reversal."""

    def test_terpenuhi_true(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is True

    def test_arah_buy(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["arah"] == "BUY"

    def test_boundary_referensi_support(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["boundary_referensi"] == "support"

    def test_sweep_terpenuhi(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_terpenuhi"] is True

    def test_rejection_terpenuhi(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["rejection_terpenuhi"] is True

    def test_invalidation_level_sama_dengan_low(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["invalidation_level"] == pytest.approx(BUY_LOW)

    def test_sweep_depth_positif(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_depth"] is not None
        assert hasil["sweep_depth"] > 0

    def test_keterangan_mengandung_buy(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert "BUY" in hasil["keterangan"]

    def test_idx_negatif_dikenali(self):
        """idx_m5 = -1 harus disamakan dengan idx 0 pada DataFrame satu baris."""
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, -1, zone)
        assert hasil["terpenuhi"] is True
        assert hasil["arah"] == "BUY"


# =============================================================================
# TEST 2: SELL VALID
# =============================================================================
# Cerminan BUY di resistance:
#   resistance = 2010.0, support = 2000.0, atr = 5.0
#   high > resistance → high = 2011.0  (depth=1.0 >= 0.5 ✓)
#   close <= resistance → close = 2008.0 ✓
#   open = 2006.0  (open lebih rendah dari close → bearish)
#   upper_wick = high - max(open, close) = 2011.0 - max(2006.0, 2008.0) = 2011.0 - 2008.0 = 3.0 >= 2.5 ✓
#   range_total = 2011.0 - 2005.0 = 6.0
#   close_posisi = (high - close) / range_total = (2011.0 - 2008.0) / 6.0 = 0.5 >= 0.5 ✓

SELL_OPEN  = 2006.0
SELL_HIGH  = 2011.0
SELL_LOW   = 2005.0
SELL_CLOSE = 2008.0


class TestSellValid:
    """Candle yang memenuhi semua syarat SELL Range Reversal."""

    def test_terpenuhi_true(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is True

    def test_arah_sell(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["arah"] == "SELL"

    def test_boundary_referensi_resistance(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["boundary_referensi"] == "resistance"

    def test_sweep_terpenuhi(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_terpenuhi"] is True

    def test_rejection_terpenuhi(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["rejection_terpenuhi"] is True

    def test_invalidation_level_sama_dengan_high(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["invalidation_level"] == pytest.approx(SELL_HIGH)

    def test_keterangan_mengandung_sell(self):
        df   = _buat_df_satu_candle(SELL_OPEN, SELL_HIGH, SELL_LOW, SELL_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert "SELL" in hasil["keterangan"]


# =============================================================================
# TEST 3: SWEEP TANPA RECLAIM (breakdown asli)
# =============================================================================
# low < support, depth cukup, TAPI close juga di bawah support → breakdown

class TestSweepTanpaReclaim:
    """Low menembus support cukup dalam tapi close tidak kembali ke atas support."""

    def test_terpenuhi_false(self):
        # low = 1999.0 → depth = 1.0 >= 0.5 ✓
        # close = 1998.0 < support=2000.0 → reclaim GAGAL
        df   = _buat_df_satu_candle(2001.0, 2002.0, 1999.0, 1998.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_arah_netral(self):
        df   = _buat_df_satu_candle(2001.0, 2002.0, 1999.0, 1998.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["arah"] == "NETRAL"

    def test_keterangan_mengandung_breakdown(self):
        df   = _buat_df_satu_candle(2001.0, 2002.0, 1999.0, 1998.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert "breakdown" in hasil["keterangan"].lower()


# =============================================================================
# TEST 4: RECLAIM TANPA SWEEP CUKUP DALAM
# =============================================================================
# low sedikit di bawah support tapi depth < RANGE_REVERSAL_MIN_SWEEP_DEPTH_ATR * atr

class TestReclaimTanpaSweepCukup:
    """Wick menembus support tapi terlalu dangkal (depth < minimum)."""

    def test_terpenuhi_false(self):
        # min_sweep_depth = 0.1 * 5.0 = 0.5
        # low = 1999.9 → depth = 0.1 < 0.5 → sweep GAGAL
        df   = _buat_df_satu_candle(2002.0, 2003.0, 1999.9, 2001.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_sweep_terpenuhi_false(self):
        df   = _buat_df_satu_candle(2002.0, 2003.0, 1999.9, 2001.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_terpenuhi"] is False

    def test_sweep_depth_none(self):
        """Karena sweep tidak terpenuhi, sweep_depth harus None."""
        df   = _buat_df_satu_candle(2002.0, 2003.0, 1999.9, 2001.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_depth"] is None

    def test_keterangan_menyebutkan_depth(self):
        df   = _buat_df_satu_candle(2002.0, 2003.0, 1999.9, 2001.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert "depth" in hasil["keterangan"].lower()


# =============================================================================
# TEST 5: SWEEP + RECLAIM OK TAPI REJECTION LEMAH
# =============================================================================
# Sweep valid, reclaim valid, tapi wick pendek atau close posisi tengah/bawah

class TestSweepOkTapiRejectionLemah:
    """Sweep+reclaim OK tapi candle tidak menunjukkan rejection yang kuat."""

    def _candle_no_rejection(self):
        # open=2001.0, high=2003.0, low=1999.0, close=2000.5, atr=5.0
        # sweep: low=1999.0 < support=2000.0, depth=1.0 >= 0.5 ✓, close=2000.5 >= 2000.0 ✓
        # lower_wick = min(2001.0, 2000.5) - 1999.0 = 2000.5 - 1999.0 = 1.5
        # batas_wick = 0.5 * 5.0 = 2.5  → 1.5 < 2.5 → REJECTION GAGAL (wick pendek)
        return _buat_df_satu_candle(2001.0, 2003.0, 1999.0, 2000.5, ATR_REF)

    def test_terpenuhi_false(self):
        df   = self._candle_no_rejection()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_sweep_terpenuhi_true(self):
        """Sweep harus tetap terpenuhi — hanya rejection yang gagal."""
        df   = self._candle_no_rejection()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_terpenuhi"] is True

    def test_rejection_terpenuhi_false(self):
        df   = self._candle_no_rejection()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["rejection_terpenuhi"] is False

    def test_rejection_detail_ada(self):
        """rejection_detail harus berupa dict berisi keterangan kenapa gagal."""
        df   = self._candle_no_rejection()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert isinstance(hasil["rejection_detail"], dict)
        assert "keterangan" in hasil["rejection_detail"]

    def test_invalidation_level_none(self):
        """Tidak ada entry → invalidation_level harus None."""
        df   = self._candle_no_rejection()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["invalidation_level"] is None

    def test_keterangan_menyebutkan_rejection_gagal(self):
        df   = self._candle_no_rejection()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert "rejection" in hasil["keterangan"].lower()

    def test_close_posisi_rendah_juga_gagal(self):
        """
        Wick cukup panjang tapi close di posisi bawah candle → close_position rendah.
        open=2000.2, high=2003.0, low=1999.0, close=2000.1 (close sangat rendah)
        lower_wick = min(2000.2, 2000.1) - 1999.0 = 1.1 → < 2.5 jadi sudah gagal di wick
        Buat candle di mana lower_wick cukup tapi close_position terlalu rendah:
        open=1999.5, high=2004.0, low=1996.0, close=2000.1, atr=5.0
        depth = 2000.0 - 1996.0 = 4.0 >= 0.5 ✓ (reclaim: close=2000.1 >= 2000.0 ✓)
        lower_wick = min(1999.5, 2000.1) - 1996.0 = 1999.5 - 1996.0 = 3.5 >= 2.5 ✓
        range_total = 2004.0 - 1996.0 = 8.0
        close_posisi = (2000.1 - 1996.0) / 8.0 = 4.1 / 8.0 = 0.5125 >= 0.5 ✓
        → HARUSNYA LULUS! Ubah close agar close_posisi jatuh di bawah 0.5:
        close = 1999.9 → close < support → reclaim GAGAL
        Perlu konstruksi berbeda: wick panjang tapi posisi rendah dalam range
        open=2002.5, high=2005.5, low=1996.0, close=2000.1
        depth = 2000.0 - 1996.0 = 4.0 >= 0.5 ✓, close=2000.1 >= 2000.0 ✓ (reclaim ✓)
        lower_wick = min(2002.5, 2000.1) - 1996.0 = 2000.1 - 1996.0 = 4.1 >= 2.5 ✓
        range_total = 2005.5 - 1996.0 = 9.5
        close_posisi = (2000.1 - 1996.0) / 9.5 = 4.1 / 9.5 ≈ 0.431 < 0.5 → GAGAL
        """
        df   = _buat_df_satu_candle(2002.5, 2005.5, 1996.0, 2000.1, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False
        assert hasil["sweep_terpenuhi"] is True
        assert hasil["rejection_terpenuhi"] is False


# =============================================================================
# TEST 6: ENTRY DI TENGAH RANGE
# =============================================================================
# low/high jauh dari kedua boundary → tidak menyentuh support maupun resistance

class TestEntryTengahRange:
    """Candle yang sama sekali tidak menyentuh boundary mana pun."""

    def test_terpenuhi_false(self):
        # support=2000.0, resistance=2010.0
        # candle ditengah range: open=2004.0, high=2006.0, low=2003.0, close=2005.0
        df   = _buat_df_satu_candle(2004.0, 2006.0, 2003.0, 2005.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_arah_netral(self):
        df   = _buat_df_satu_candle(2004.0, 2006.0, 2003.0, 2005.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["arah"] == "NETRAL"

    def test_boundary_referensi_none(self):
        df   = _buat_df_satu_candle(2004.0, 2006.0, 2003.0, 2005.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["boundary_referensi"] is None

    def test_sweep_terpenuhi_false(self):
        df   = _buat_df_satu_candle(2004.0, 2006.0, 2003.0, 2005.0, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["sweep_terpenuhi"] is False


# =============================================================================
# TEST 7: ZONE = NONE
# =============================================================================

class TestZoneNone:
    """Zone adalah None — tidak boleh crash."""

    def test_tidak_crash(self):
        df = _buat_df_satu_candle(2004.0, 2006.0, 1999.0, 2001.0, ATR_REF)
        hasil = evaluate_range_reversal(df, 0, None)
        assert hasil["terpenuhi"] is False

    def test_arah_netral(self):
        df = _buat_df_satu_candle(2004.0, 2006.0, 1999.0, 2001.0, ATR_REF)
        hasil = evaluate_range_reversal(df, 0, None)
        assert hasil["arah"] == "NETRAL"

    def test_keterangan_menyebutkan_zone_none(self):
        df = _buat_df_satu_candle(2004.0, 2006.0, 1999.0, 2001.0, ATR_REF)
        hasil = evaluate_range_reversal(df, 0, None)
        assert "none" in hasil["keterangan"].lower() or "tidak tersedia" in hasil["keterangan"].lower()


# =============================================================================
# TEST 8: ZONE DENGAN RESISTANCE/SUPPORT = NONE
# =============================================================================

class TestZoneFieldNone:
    """Zone dict ada tapi resistance atau support bernilai None."""

    def test_resistance_none_tidak_crash(self):
        df   = _buat_df_satu_candle(2004.0, 2011.0, 1999.0, 2008.0, ATR_REF)
        zone = {"resistance": None, "support": SUPPORT}
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_support_none_tidak_crash(self):
        df   = _buat_df_satu_candle(2004.0, 2005.0, 1999.0, 2001.0, ATR_REF)
        zone = {"resistance": RESISTANCE, "support": None}
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_keduanya_none_tidak_crash(self):
        df   = _buat_df_satu_candle(2004.0, 2006.0, 1999.0, 2001.0, ATR_REF)
        zone = {"resistance": None, "support": None}
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_zone_kosong_tidak_crash(self):
        """Zone dict ada tapi tidak ada key resistance/support sama sekali."""
        df   = _buat_df_satu_candle(2004.0, 2006.0, 1999.0, 2001.0, ATR_REF)
        zone = {}
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False


# =============================================================================
# TEST 9: KOLOM DF_M5 TIDAK LENGKAP
# =============================================================================

class TestKolomTidakLengkap:
    """DataFrame tidak memiliki kolom yang dibutuhkan."""

    def test_tanpa_atr14_tidak_crash(self):
        df = pd.DataFrame([{"open": 2004.0, "high": 2005.0, "low": 1999.0, "close": 2002.0}])
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_tanpa_high_tidak_crash(self):
        df = pd.DataFrame([{"open": 2004.0, "low": 1999.0, "close": 2002.0, "atr_14": ATR_REF}])
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_tanpa_low_tidak_crash(self):
        df = pd.DataFrame([{"open": 2004.0, "high": 2005.0, "close": 2002.0, "atr_14": ATR_REF}])
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_df_kosong_tidak_crash(self):
        df = pd.DataFrame()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert hasil["terpenuhi"] is False

    def test_keterangan_menyebutkan_kolom_hilang(self):
        df = pd.DataFrame([{"open": 2004.0, "high": 2005.0, "low": 1999.0, "close": 2002.0}])
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 0, zone)
        assert "kolom" in hasil["keterangan"].lower() or "atr" in hasil["keterangan"].lower()


# =============================================================================
# TEST 10: KAUSALITAS — MUTASI BARIS SETELAH IDX TIDAK MEMPENGARUHI HASIL
# =============================================================================

class TestKausalitasSingleCandle:
    """
    Membuktikan bahwa evaluate_range_reversal() hanya membaca df_m5.iloc[idx_m5].
    Mutasi ekstrem pada semua baris SETELAH idx tidak boleh mengubah hasil
    evaluasi di idx.
    """

    def _buat_df_multi(self) -> pd.DataFrame:
        """
        DataFrame 5 baris. idx evaluasi = 2 (baris tengah).
        Baris 0,1: data normal (sebelum idx — seharusnya tidak dibaca).
        Baris 2: candle BUY valid (index evaluasi).
        Baris 3,4: data normal (setelah idx — TIDAK BOLEH dibaca).
        """
        baris = [
            {"open": 2003.0, "high": 2004.0, "low": 2001.0, "close": 2002.0, "atr_14": ATR_REF},
            {"open": 2002.0, "high": 2003.5, "low": 2001.5, "close": 2003.0, "atr_14": ATR_REF},
            # Baris evaluasi (idx=2) — candle BUY valid
            {"open": BUY_OPEN, "high": BUY_HIGH, "low": BUY_LOW, "close": BUY_CLOSE, "atr_14": ATR_REF},
            {"open": 2003.0, "high": 2004.0, "low": 2001.0, "close": 2002.0, "atr_14": ATR_REF},
            {"open": 2004.0, "high": 2005.0, "low": 2003.0, "close": 2004.5, "atr_14": ATR_REF},
        ]
        return pd.DataFrame(baris)

    def test_baseline_buy_valid(self):
        """Tanpa mutasi, idx=2 harus menghasilkan BUY."""
        df   = self._buat_df_multi()
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 2, zone)
        assert hasil["terpenuhi"] is True
        assert hasil["arah"] == "BUY"

    def test_mutasi_setelah_idx_tidak_berpengaruh(self):
        """
        Setelah mutasi ekstrem pada baris 3 dan 4 (setelah idx=2),
        hasil evaluasi di idx=2 harus IDENTIK dengan baseline.
        """
        df_ori    = self._buat_df_multi()
        zone      = _buat_zone(SUPPORT, RESISTANCE)
        hasil_ori = evaluate_range_reversal(df_ori, 2, zone)

        # Buat salinan dan mutasi baris 3 & 4 secara ekstrem
        df_mutasi = df_ori.copy()
        for kolom in ["open", "high", "low", "close", "atr_14"]:
            df_mutasi.loc[3, kolom] = 9999.0
            df_mutasi.loc[4, kolom] = 0.0001

        hasil_mutasi = evaluate_range_reversal(df_mutasi, 2, zone)

        # Semua field harus identik
        assert hasil_mutasi["terpenuhi"]          == hasil_ori["terpenuhi"]
        assert hasil_mutasi["arah"]               == hasil_ori["arah"]
        assert hasil_mutasi["boundary_referensi"] == hasil_ori["boundary_referensi"]
        assert hasil_mutasi["sweep_terpenuhi"]    == hasil_ori["sweep_terpenuhi"]
        assert hasil_mutasi["rejection_terpenuhi"]== hasil_ori["rejection_terpenuhi"]
        assert hasil_mutasi["invalidation_level"] == pytest.approx(hasil_ori["invalidation_level"])

    def test_mutasi_sebelum_idx_tidak_berpengaruh(self):
        """
        Mutasi pada baris 0 dan 1 (sebelum idx=2) juga tidak boleh berpengaruh —
        fungsi ini hanya membaca idx=2, bukan lookback.
        """
        df_ori    = self._buat_df_multi()
        zone      = _buat_zone(SUPPORT, RESISTANCE)
        hasil_ori = evaluate_range_reversal(df_ori, 2, zone)

        df_mutasi = df_ori.copy()
        for kolom in ["open", "high", "low", "close", "atr_14"]:
            df_mutasi.loc[0, kolom] = 1.0
            df_mutasi.loc[1, kolom] = 1.0

        hasil_mutasi = evaluate_range_reversal(df_mutasi, 2, zone)

        assert hasil_mutasi["terpenuhi"]          == hasil_ori["terpenuhi"]
        assert hasil_mutasi["arah"]               == hasil_ori["arah"]
        assert hasil_mutasi["boundary_referensi"] == hasil_ori["boundary_referensi"]
        assert hasil_mutasi["sweep_terpenuhi"]    == hasil_ori["sweep_terpenuhi"]
        assert hasil_mutasi["rejection_terpenuhi"]== hasil_ori["rejection_terpenuhi"]
        assert hasil_mutasi["invalidation_level"] == pytest.approx(hasil_ori["invalidation_level"])


# =============================================================================
# TEST TAMBAHAN: HELPER _detect_rejection_wick SECARA LANGSUNG
# =============================================================================

class TestDetectRejectionWick:
    """Unit test langsung untuk helper _detect_rejection_wick."""

    def test_buy_rejection_valid(self):
        # lower_wick = min(2004,2002) - 1999 = 3.0 >= 2.5 ✓
        # close_posisi = (2002-1999)/6 = 0.5 >= 0.5 ✓
        hasil = _detect_rejection_wick(2004.0, 2005.0, 1999.0, 2002.0, ATR_REF, "BUY")
        assert hasil["terpenuhi"] is True
        assert hasil["wick_length"] == pytest.approx(3.0)
        assert hasil["close_position"] == pytest.approx(0.5)

    def test_sell_rejection_valid(self):
        # upper_wick = 2011 - max(2006, 2008) = 2011 - 2008 = 3.0 >= 2.5 ✓
        # close_posisi = (2011-2008)/6 = 0.5 >= 0.5 ✓
        hasil = _detect_rejection_wick(2006.0, 2011.0, 2005.0, 2008.0, ATR_REF, "SELL")
        assert hasil["terpenuhi"] is True
        assert hasil["wick_length"] == pytest.approx(3.0)

    def test_buy_wick_pendek_gagal(self):
        # lower_wick = min(2001,2000.5) - 1999 = 1.5 < 2.5
        hasil = _detect_rejection_wick(2001.0, 2003.0, 1999.0, 2000.5, ATR_REF, "BUY")
        assert hasil["terpenuhi"] is False

    def test_doji_range_nol(self):
        """Candle doji sempurna (high==low) tidak boleh crash."""
        hasil = _detect_rejection_wick(2000.0, 2000.0, 2000.0, 2000.0, ATR_REF, "BUY")
        assert hasil["terpenuhi"] is False
        assert "range candle" in hasil["keterangan"].lower() or "doji" in hasil["keterangan"].lower()

    def test_return_dict_field_lengkap(self):
        """Return dict harus selalu memiliki field wajib."""
        hasil = _detect_rejection_wick(2004.0, 2005.0, 1999.0, 2002.0, ATR_REF, "BUY")
        assert "terpenuhi"      in hasil
        assert "wick_length"    in hasil
        assert "close_position" in hasil
        assert "keterangan"     in hasil


# =============================================================================
# TEST EDGE CASE: IDX TIDAK VALID
# =============================================================================

class TestIdxTidakValid:
    """idx_m5 di luar range DataFrame tidak boleh crash."""

    def test_idx_terlalu_besar(self):
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, 99, zone)
        assert hasil["terpenuhi"] is False

    def test_idx_negatif_terlalu_besar(self):
        """idx=-99 pada DataFrame 1 baris harus gracefully return False."""
        df   = _buat_df_satu_candle(BUY_OPEN, BUY_HIGH, BUY_LOW, BUY_CLOSE, ATR_REF)
        zone = _buat_zone(SUPPORT, RESISTANCE)
        hasil = evaluate_range_reversal(df, -99, zone)
        assert hasil["terpenuhi"] is False
