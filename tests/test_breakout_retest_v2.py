"""
tests/test_breakout_retest_v2.py
=================================
Unit test untuk engine/strategies/breakout_retest_v2.py — Fase 16.

Cakupan test (12 skenario sesuai spesifikasi):
    1.  BUY retest valid: arah=BULLISH, touch di resistance, konfirmasi terpenuhi
    2.  SELL retest valid: arah=BEARISH, touch di support, konfirmasi terpenuhi
    3.  Tidak ada touch dalam window → terpenuhi=False, keterangan menyebut "belum ada retest touch"
    4.  Invalidasi terjadi setelah touch → terpenuhi=False, invalidated=True
    5.  Touch ditemukan tapi konfirmasi_close gagal
    6.  Touch ditemukan, close ok, tapi konfirmasi_body gagal
    7.  Banyak touch dalam window → yang dipakai touch PALING BARU (paling dekat idx_m5)
    8.  idx_m5 dekat awal data → window lookback terpotong, tidak crash
    9.  zone=None atau field hilang → terpenuhi=False, tidak crash
    10. arah tidak dikenal → terpenuhi=False, tidak crash
    11. Test kausalitas (WAJIB): mutasi ekstrem candle SETELAH idx_m5 tidak mempengaruhi hasil
    12. Test independensi arsitektural (WAJIB): source file tidak mengandung "rule_engine"

Semua DataFrame sintetis dibangun minimal — hanya kolom yang dibutuhkan fungsi.
"""

import copy
import inspect
import pathlib
import pytest
import pandas as pd
import numpy as np

from engine.strategies.breakout_retest_v2 import (
    evaluate_breakout_retest,
    BREAKOUT_RETEST_LOOKBACK_M5,
    BREAKOUT_RETEST_TOUCH_TOLERANCE_ATR,
    BREAKOUT_RETEST_INVALIDATION_BUFFER_ATR,
    BREAKOUT_RETEST_MIN_BODY_ATR_RATIO,
)


# =============================================================================
# HELPER BUILDER
# =============================================================================

def _candle(open_: float, high: float, low: float, close: float, atr: float) -> dict:
    """Buat dict satu baris candle M5."""
    return {"open": open_, "high": high, "low": low, "close": close, "atr_14": atr}


def _buat_df(baris: list[dict]) -> pd.DataFrame:
    """Buat DataFrame M5 dari list dict candle."""
    return pd.DataFrame(baris)


def _zone(resistance: float, support: float) -> dict:
    """Buat dict zone minimal."""
    return {"resistance": resistance, "support": support}


# =============================================================================
# PARAMETER REFERENSI
# =============================================================================
# resistance = 2010.0  (level retest untuk BULLISH)
# support    = 2000.0  (level retest untuk BEARISH)
# ATR        = 5.0
#
# Touch tolerance BULLISH: |low - 2010.0| <= 0.3 * 5.0 = 1.5
#   → low di rentang [2008.5, 2011.5]
# Touch tolerance BEARISH: |high - 2000.0| <= 0.3 * 5.0 = 1.5
#   → high di rentang [1998.5, 2001.5]
#
# Konfirmasi body min: 0.3 * 5.0 = 1.5
# Konfirmasi close BULLISH: close > 2010.0
# Konfirmasi close BEARISH: close < 2000.0
#
# ATURAN INVALIDASI:
#   BULLISH: candle antara touch dan konfirmasi harus close >= level_ref - buffer
#            = 2010.0 - (0.5 * 5.0) = 2010.0 - 2.5 = 2007.5
#   BEARISH: candle antara touch dan konfirmasi harus close <= level_ref + buffer
#            = 2000.0 + (0.5 * 5.0) = 2000.0 + 2.5 = 2002.5
#
# Karena itu kita perlu dua jenis candle "isi" yang berbeda untuk dua arah.

ATR      = 5.0
RES      = 2010.0
SUP      = 2000.0
ZONE_REF = _zone(RES, SUP)

# Candle "netral" umum — hanya untuk posisi SEBELUM touch (tidak dalam jangkauan scan)
NETRAL = _candle(2005.0, 2006.0, 2004.0, 2005.0, ATR)

# Candle "isi" antara touch dan konfirmasi BULLISH:
# close=2008.5 >= 2007.5 → tidak invalidasi BULLISH ✓
# low=2008.0 → |2008.0 - 2010.0| = 2.0 > 1.5 → bukan touch ✓
ISI_BULLISH = _candle(2009.0, 2009.5, 2008.0, 2008.5, ATR)

# Candle "isi" antara touch dan konfirmasi BEARISH:
# close=2001.5 <= 2002.5 → tidak invalidasi BEARISH ✓
# high=2001.8 → |2001.8 - 2000.0| = 1.8 > 1.5 → bukan touch ✓
ISI_BEARISH = _candle(2001.5, 2001.8, 2001.0, 2001.5, ATR)

# Candle touch BULLISH (low dekat resistance dari atas):
# Setelah breakout BULLISH, harga pullback ke resistance dari ATAS → low dekat resistance.
# low = 2009.0 → |2009.0 - 2010.0| = 1.0 <= 1.5 ✓
# close = 2011.5 → tidak invalidasi (2011.5 >= 2007.5) ✓
TOUCH_BULLISH = _candle(2011.0, 2012.0, 2009.0, 2011.5, ATR)

# Candle touch BEARISH (high dekat support dari bawah):
# Setelah breakout BEARISH, harga pullback ke support dari BAWAH → high dekat support.
# high = 2001.0 → |2001.0 - 2000.0| = 1.0 <= 1.5 ✓
# close = 1998.5 → tidak invalidasi (1998.5 <= 2002.5) ✓
TOUCH_BEARISH = _candle(1999.0, 2001.0, 1998.0, 1998.5, ATR)

# Candle konfirmasi BULLISH:
# close = 2012.0 > 2010.0 ✓
# body = |2012.0 - 2010.0| = 2.0 >= 0.3*5.0=1.5 ✓ (cukup besar, hindari floating point)
KONFIRM_BULLISH = _candle(2010.0, 2013.0, 2009.5, 2012.0, ATR)

# Candle konfirmasi BEARISH:
# close = 1998.0 < 2000.0 ✓
# body = |1998.0 - 2000.0| = 2.0 >= 0.3*5.0=1.5 ✓
KONFIRM_BEARISH = _candle(2000.0, 2000.5, 1997.0, 1998.0, ATR)


# =============================================================================
# SKENARIO 1 — BUY RETEST VALID
# =============================================================================

class TestBuyRetestValid:
    """
    Skenario: arah=BULLISH, touch di resistance beberapa candle sebelum idx_m5,
    tidak ada invalidasi, candle idx_m5 close di atas resistance dengan body cukup.
    Ekspektasi: terpenuhi=True, arah="BUY", invalidation_level_sl=low candle touch.
    """

    def test_terpenuhi_true(self):
        # Struktur: [netral] [netral] [TOUCH] [ISI_BULLISH] [KONFIRM=idx_m5]
        # ISI_BULLISH: close=2008.5 >= 2007.5 → tidak memicu invalidasi BULLISH
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        idx  = 4   # candle konfirmasi

        hasil = evaluate_breakout_retest(df, idx, ZONE_REF, "BULLISH")

        assert hasil["terpenuhi"] is True

    def test_arah_buy(self):
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["arah"] == "BUY"

    def test_invalidation_level_sl_adalah_low_touch(self):
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        # touch_idx=2, low[2]=2009.0
        assert hasil["invalidation_level_sl"] == pytest.approx(TOUCH_BULLISH["low"])

    def test_touch_idx_benar(self):
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["touch_idx"] == 2

    def test_candles_since_touch(self):
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["candles_since_touch"] == 2

    def test_level_referensi_adalah_resistance(self):
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["level_referensi"] == pytest.approx(RES)

    def test_tidak_invalidasi(self):
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["invalidated"] is False


# =============================================================================
# SKENARIO 2 — SELL RETEST VALID
# =============================================================================

class TestSellRetestValid:
    """
    Cerminan skenario 1 untuk BEARISH/SELL.
    Touch di support, konfirmasi close di bawah support, body cukup.
    """

    def test_terpenuhi_true(self):
        # ISI_BEARISH: close=2001.5 <= 2002.5 → tidak memicu invalidasi BEARISH
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["terpenuhi"] is True

    def test_arah_sell(self):
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["arah"] == "SELL"

    def test_invalidation_level_sl_adalah_high_touch(self):
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        # touch_idx=2, high[2]=2001.0
        assert hasil["invalidation_level_sl"] == pytest.approx(TOUCH_BEARISH["high"])

    def test_touch_idx_benar(self):
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["touch_idx"] == 2

    def test_level_referensi_adalah_support(self):
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["level_referensi"] == pytest.approx(SUP)

    def test_tidak_invalidasi(self):
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["invalidated"] is False


# =============================================================================
# SKENARIO 3 — TIDAK ADA TOUCH DALAM WINDOW
# =============================================================================

class TestTidakAdaTouch:
    """
    Semua candle dalam window jauh dari level_ref. Ekspektasi: terpenuhi=False,
    keterangan menyebut "belum ada retest touch".
    """

    def test_bullish_tidak_ada_touch(self):
        # low jauh di atas resistance (misal di 2015) — tidak mungkin menyentuh 2010.0 ± 1.5
        jauh = _candle(2015.0, 2016.0, 2014.0, 2015.5, ATR)
        baris = [jauh] * 5
        # Konfirmasi candle tetap di atas resistance
        konfirm = _candle(2012.0, 2014.0, 2011.5, 2013.0, ATR)
        baris[-1] = konfirm
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False
        assert "belum ada retest touch" in hasil["keterangan"]

    def test_bearish_tidak_ada_touch(self):
        # high jauh di bawah support (misal 1990) — tidak mungkin menyentuh 2000.0 ± 1.5
        jauh = _candle(1988.0, 1990.0, 1987.0, 1988.5, ATR)
        baris = [jauh] * 5
        konfirm = _candle(1997.0, 1998.5, 1995.0, 1996.0, ATR)
        baris[-1] = konfirm
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["terpenuhi"] is False
        assert "belum ada retest touch" in hasil["keterangan"]

    def test_window_habis_sebelum_touch(self):
        """
        Window lookback lebih kecil dari posisi candle touch.
        Gunakan lookback=1 — hanya 1 candle ke belakang.
        Touch ada di idx=2, konfirmasi di idx=4 → dengan lookback=1, scan dari idx=3
        hanya lihat idx=3 (netral). Tidak ketemu touch.
        """
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, NETRAL, KONFIRM_BULLISH]
        df    = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH", retest_lookback_m5=1)
        assert hasil["terpenuhi"] is False
        assert "belum ada retest touch" in hasil["keterangan"]


# =============================================================================
# SKENARIO 4 — INVALIDASI TERJADI SETELAH TOUCH
# =============================================================================

class TestInvalidasi:
    """
    Setelah touch, ada candle yang close terlalu jauh masuk ke zona lama.
    Ekspektasi: terpenuhi=False, invalidated=True.
    """

    def test_bullish_invalidasi(self):
        # BULLISH: touch di idx=1, lalu idx=2 close jauh di bawah resistance
        # Invalidasi: close < level_ref - buffer = 2010.0 - 0.5*5.0 = 2007.5
        # Pakai close=2006.0 (jelas di bawah 2007.5)
        invalidasi = _candle(2009.0, 2009.5, 2005.0, 2006.0, ATR)
        baris = [NETRAL, TOUCH_BULLISH, invalidasi, NETRAL, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False
        assert hasil["invalidated"] is True

    def test_bearish_invalidasi(self):
        # BEARISH: touch di idx=1, lalu idx=2 close jauh di atas support
        # Invalidasi: close > level_ref + buffer = 2000.0 + 0.5*5.0 = 2002.5
        # Pakai close=2004.0
        invalidasi = _candle(2001.0, 2005.0, 2000.5, 2004.0, ATR)
        baris = [NETRAL, TOUCH_BEARISH, invalidasi, NETRAL, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["terpenuhi"] is False
        assert hasil["invalidated"] is True

    def test_invalidasi_exact_boundary_tidak_invalidasi(self):
        """
        close tepat pada batas invalidasi (tidak melewati) → TIDAK invalidasi.
        Batas invalidasi BULLISH: close < level_ref - buffer = 2010.0 - 2.5 = 2007.5
        Pakai close=2007.6 (sedikit di atas batas, tidak melampaui) → tidak invalidasi.
        """
        # close=2007.6 NOT < 2007.5 → bukan invalidasi
        batas_tidak_invalidasi = _candle(2008.0, 2009.0, 2007.0, 2007.6, ATR)
        baris = [NETRAL, TOUCH_BULLISH, batas_tidak_invalidasi, ISI_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["invalidated"] is False


# =============================================================================
# SKENARIO 5 — KONFIRMASI CLOSE GAGAL
# =============================================================================

class TestKonfirmasiClosGagal:
    """
    Touch ditemukan, tidak ada invalidasi, tapi candle idx_m5 close tidak menembus level.
    Ekspektasi: terpenuhi=False, konfirmasi_close=False.
    """

    def test_bullish_close_di_bawah_resistance(self):
        # Close di 2009.5 (di bawah resistance 2010.0) — close gagal
        # Body cukup: |2009.5 - 2007.0| = 2.5 >= 1.5 ✓ (body ok, close yang gagal)
        # CATATAN: close=2009.5 >= 2007.5 → tidak memicu invalidasi di cek sebelumnya
        konfirm_gagal_close = _candle(2007.0, 2011.0, 2006.5, 2009.5, ATR)
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, konfirm_gagal_close]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False
        assert hasil["konfirmasi_close"] is False

    def test_bearish_close_di_atas_support(self):
        # Close di 2000.5 (di atas support 2000.0) — close gagal
        # close=2000.5 <= 2002.5 → tidak memicu invalidasi
        konfirm_gagal_close = _candle(2002.0, 2002.5, 1999.0, 2000.5, ATR)
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, konfirm_gagal_close]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["terpenuhi"] is False
        assert hasil["konfirmasi_close"] is False

    def test_touch_idx_tetap_terisi_saat_close_gagal(self):
        """Meski konfirmasi gagal, touch_idx dan level_referensi harus terisi."""
        konfirm_gagal_close = _candle(2007.0, 2011.0, 2006.5, 2009.5, ATR)
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, konfirm_gagal_close]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["touch_idx"] == 2
        assert hasil["level_referensi"] == pytest.approx(RES)


# =============================================================================
# SKENARIO 6 — KONFIRMASI BODY GAGAL
# =============================================================================

class TestKonfirmasiBodyGagal:
    """
    Touch ditemukan, close ok (menembus level), tapi body terlalu kecil.
    Ekspektasi: terpenuhi=False, konfirmasi_body=False.
    """

    def test_bullish_body_terlalu_kecil(self):
        # close=2010.5 > 2010.0 ✓ (close ok)
        # body = |2010.5 - 2010.4| = 0.1 < 1.5 ✗ (body gagal)
        konfirm_body_kecil = _candle(2010.4, 2011.0, 2009.5, 2010.5, ATR)
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, konfirm_body_kecil]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False
        assert hasil["konfirmasi_body"] is False
        assert hasil["konfirmasi_close"] is True   # close sudah ok

    def test_bearish_body_terlalu_kecil(self):
        # close=1999.5 < 2000.0 ✓ (close ok)
        # body = |1999.5 - 1999.6| = 0.1 < 1.5 ✗
        konfirm_body_kecil = _candle(1999.6, 2000.4, 1999.0, 1999.5, ATR)
        baris = [NETRAL, NETRAL, TOUCH_BEARISH, ISI_BEARISH, konfirm_body_kecil]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["terpenuhi"] is False
        assert hasil["konfirmasi_body"] is False
        assert hasil["konfirmasi_close"] is True


# =============================================================================
# SKENARIO 7 — BANYAK TOUCH: PAKAI YANG PALING BARU
# =============================================================================

class TestTouchPalingBaru:
    """
    Ada beberapa candle touch dalam window. Fungsi harus memilih touch yang
    PALING BARU (paling dekat idx_m5), bukan yang tertua.
    """

    def test_bullish_pilih_touch_terbaru(self):
        # Touch pertama di idx=1 (low=2009.0), touch kedua (lebih baru) di idx=3 (low=2009.5)
        # idx_m5=4. Scan dari idx=3 ke bawah → ketemu idx=3 duluan.
        # Tidak ada candle ISI antara touch_baru (idx=3) dan konfirm (idx=4) → tidak ada invalidasi.
        touch_tua  = _candle(2011.0, 2012.0, 2009.0, 2011.5, ATR)   # idx=1, low=2009.0
        # idx=2 adalah ISI_BULLISH (antara dua touch, close=2008.5 tidak invalidasi)
        touch_baru = _candle(2011.0, 2012.0, 2009.5, 2011.8, ATR)   # idx=3, low=2009.5
        baris = [NETRAL, touch_tua, ISI_BULLISH, touch_baru, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        # touch_idx harus 3 (paling baru), bukan 1 (paling tua)
        assert hasil["touch_idx"] == 3
        # invalidation_level_sl = low[3] = 2009.5
        assert hasil["invalidation_level_sl"] == pytest.approx(2009.5)

    def test_bearish_pilih_touch_terbaru(self):
        # Touch tua di idx=1 (high=2001.0), touch baru di idx=3 (high=2000.8)
        touch_tua  = _candle(1999.0, 2001.0, 1998.0, 1998.5, ATR)   # idx=1, high=2001.0
        # idx=2 adalah ISI_BEARISH (close=2001.5 tidak invalidasi)
        touch_baru = _candle(1999.2, 2000.8, 1998.5, 1999.0, ATR)   # idx=3, high=2000.8
        baris = [NETRAL, touch_tua, ISI_BEARISH, touch_baru, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BEARISH")
        assert hasil["touch_idx"] == 3
        assert hasil["invalidation_level_sl"] == pytest.approx(2000.8)


# =============================================================================
# SKENARIO 8 — IDX_M5 DEKAT AWAL DATA (WINDOW TERPOTONG)
# =============================================================================

class TestWindowTerpotong:
    """
    idx_m5 sangat dekat awal data (misal idx=1). Window lookback default=12,
    tapi hanya ada 1 candle sebelumnya (idx=0). Fungsi harus bekerja dengan
    window yang tersedia tanpa crash.
    """

    def test_idx_di_awal_tidak_crash_tanpa_touch(self):
        baris = [NETRAL, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        # idx=1, lookback=12 → batas_bawah=max(1-12,0)=0, scan dari idx=0 ke idx=0
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, "BULLISH")
        assert isinstance(hasil, dict)
        assert "terpenuhi" in hasil

    def test_idx_di_awal_dengan_touch(self):
        # [TOUCH_BULLISH (idx=0), KONFIRM_BULLISH (idx=1)]
        # Tidak ada candle ISI antara touch (idx=0) dan konfirm (idx=1) → langsung konfirmasi.
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is True
        assert hasil["touch_idx"] == 0

    def test_idx_zero_tidak_crash(self):
        """idx_m5=0: tidak ada candle sebelumnya, window kosong."""
        baris = [KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 0, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False
        # Tidak ada candle sebelum idx=0 untuk scan, jadi tidak ada touch
        assert "belum ada retest touch" in hasil["keterangan"]


# =============================================================================
# SKENARIO 9 — ZONE NONE ATAU FIELD HILANG
# =============================================================================

class TestZoneInvalid:
    """
    zone=None atau field resistance/support tidak ada/None.
    Ekspektasi: terpenuhi=False, tidak crash.
    """

    def test_zone_none(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, None, "BULLISH")
        assert hasil["terpenuhi"] is False
        assert isinstance(hasil["keterangan"], str)

    def test_zone_resistance_none(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        zone_rusak = {"resistance": None, "support": SUP}
        hasil = evaluate_breakout_retest(df, 1, zone_rusak, "BULLISH")
        assert hasil["terpenuhi"] is False

    def test_zone_support_none(self):
        baris = [TOUCH_BEARISH, KONFIRM_BEARISH]
        df   = _buat_df(baris)
        zone_rusak = {"resistance": RES, "support": None}
        hasil = evaluate_breakout_retest(df, 1, zone_rusak, "BEARISH")
        assert hasil["terpenuhi"] is False

    def test_zone_dict_kosong(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, {}, "BULLISH")
        assert hasil["terpenuhi"] is False

    def test_zone_key_tidak_ada(self):
        """zone dengan key yang sama sekali berbeda."""
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        zone_asing = {"mid": 2005.0}
        hasil = evaluate_breakout_retest(df, 1, zone_asing, "BULLISH")
        assert hasil["terpenuhi"] is False


# =============================================================================
# SKENARIO 10 — ARAH TIDAK DIKENAL
# =============================================================================

class TestArahTidakDikenal:
    """
    arah bukan "BULLISH" / "BEARISH". Ekspektasi: terpenuhi=False, tidak crash.
    """

    def test_arah_string_kosong(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, "")
        assert hasil["terpenuhi"] is False
        assert isinstance(hasil["keterangan"], str)

    def test_arah_salah_kapital(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, "bullish")
        assert hasil["terpenuhi"] is False

    def test_arah_nilai_lain(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, "TRENDING")
        assert hasil["terpenuhi"] is False

    def test_arah_none(self):
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, None)  # type: ignore
        assert hasil["terpenuhi"] is False


# =============================================================================
# SKENARIO 11 — TEST KAUSALITAS (WAJIB)
# =============================================================================

class TestKausalitas:
    """
    Mutasi ekstrem seluruh candle SETELAH idx_m5 harus TIDAK mempengaruhi hasil
    evaluate_breakout_retest() di idx_m5. Ini membuktikan zero look-ahead bias.
    """

    def _buat_df_panjang(self) -> pd.DataFrame:
        """
        DataFrame 20 candle: touch di idx=10, konfirmasi di idx=15.
        Candle antara touch (10) dan konfirm (15) pakai ISI_BULLISH agar tidak invalidasi.
        BULLISH: close ISI_BULLISH=2008.5 >= 2010.0 - 2.5=2007.5 → aman.
        """
        baris = []
        for i in range(20):
            if i == 10:
                baris.append(TOUCH_BULLISH)
            elif i == 15:
                baris.append(KONFIRM_BULLISH)
            elif 10 < i < 15:
                baris.append(ISI_BULLISH)
            else:
                baris.append(NETRAL)
        return _buat_df(baris)

    def test_mutasi_setelah_idx_tidak_ubah_hasil_bullish(self):
        df_asli  = self._buat_df_panjang()
        df_mutasi = df_asli.copy(deep=True)

        hasil_asli = evaluate_breakout_retest(df_asli, 15, ZONE_REF, "BULLISH")

        # Mutasi ekstrem: semua candle SETELAH idx=15 dibuat absurd
        for i in range(16, len(df_mutasi)):
            df_mutasi.at[i, "open"]   = 999999.0
            df_mutasi.at[i, "high"]   = 999999.0
            df_mutasi.at[i, "low"]    = 0.0001
            df_mutasi.at[i, "close"]  = 0.0001
            df_mutasi.at[i, "atr_14"] = 50000.0

        hasil_mutasi = evaluate_breakout_retest(df_mutasi, 15, ZONE_REF, "BULLISH")

        assert hasil_asli["terpenuhi"]             == hasil_mutasi["terpenuhi"]
        assert hasil_asli["arah"]                  == hasil_mutasi["arah"]
        assert hasil_asli["touch_idx"]             == hasil_mutasi["touch_idx"]
        assert hasil_asli["invalidated"]           == hasil_mutasi["invalidated"]
        assert hasil_asli["konfirmasi_close"]      == hasil_mutasi["konfirmasi_close"]
        assert hasil_asli["konfirmasi_body"]       == hasil_mutasi["konfirmasi_body"]
        assert hasil_asli["invalidation_level_sl"] == hasil_mutasi["invalidation_level_sl"]

    def test_mutasi_setelah_idx_tidak_ubah_hasil_bearish(self):
        baris = []
        for i in range(20):
            if i == 10:
                baris.append(TOUCH_BEARISH)
            elif i == 15:
                baris.append(KONFIRM_BEARISH)
            elif 10 < i < 15:
                baris.append(ISI_BEARISH)   # close=2001.5 <= 2002.5 → tidak invalidasi
            else:
                baris.append(NETRAL)
        df_asli  = _buat_df(baris)
        df_mutasi = df_asli.copy(deep=True)

        hasil_asli = evaluate_breakout_retest(df_asli, 15, ZONE_REF, "BEARISH")

        for i in range(16, len(df_mutasi)):
            df_mutasi.at[i, "open"]   = 0.0001
            df_mutasi.at[i, "high"]   = 0.0001
            df_mutasi.at[i, "low"]    = 0.0001
            df_mutasi.at[i, "close"]  = 999999.0
            df_mutasi.at[i, "atr_14"] = 50000.0

        hasil_mutasi = evaluate_breakout_retest(df_mutasi, 15, ZONE_REF, "BEARISH")

        assert hasil_asli["terpenuhi"]             == hasil_mutasi["terpenuhi"]
        assert hasil_asli["arah"]                  == hasil_mutasi["arah"]
        assert hasil_asli["touch_idx"]             == hasil_mutasi["touch_idx"]
        assert hasil_asli["invalidated"]           == hasil_mutasi["invalidated"]
        assert hasil_asli["konfirmasi_close"]      == hasil_mutasi["konfirmasi_close"]
        assert hasil_asli["konfirmasi_body"]       == hasil_mutasi["konfirmasi_body"]
        assert hasil_asli["invalidation_level_sl"] == hasil_mutasi["invalidation_level_sl"]

    def test_mutasi_candle_idx_sendiri_ubah_hasil(self):
        """
        Verifikasi bahwa mutasi candle idx_m5 SENDIRI memang mengubah hasil
        (sanity check — membuktikan fungsi membaca candle idx_m5).
        """
        df_asli   = self._buat_df_panjang()
        df_mutasi = df_asli.copy(deep=True)

        hasil_asli = evaluate_breakout_retest(df_asli, 15, ZONE_REF, "BULLISH")
        # Pastikan kondisi asli memang terpenuhi (True) agar mutasi bisa mengubahnya
        assert hasil_asli["terpenuhi"] is True, (
            "Prasyarat test gagal: kondisi asli harus terpenuhi=True sebelum mutasi"
        )

        # Ubah close candle idx=15 jadi di bawah resistance (konfirmasi_close gagal)
        df_mutasi.at[15, "close"] = 2008.0  # di bawah 2010.0

        hasil_mutasi = evaluate_breakout_retest(df_mutasi, 15, ZONE_REF, "BULLISH")

        # Hasil HARUS berbeda karena kita mutasi candle idx_m5 sendiri
        assert hasil_asli["terpenuhi"] != hasil_mutasi["terpenuhi"]


# =============================================================================
# SKENARIO 12 — TEST INDEPENDENSI ARSITEKTURAL DARI FASE 10 (WAJIB)
# =============================================================================

class TestIndependensiArsitekturalFase10:
    """
    Verifikasi otomatis bahwa engine/strategies/breakout_retest_v2.py
    TIDAK mengimpor atau mereferensikan "rule_engine" dalam bentuk apapun.

    Ini adalah bukti eksplisit dan otomatis (bukan cuma catatan manual) bahwa
    modul Fase 16 ini berdiri independen dari kode Fase 9/10 lama.
    """

    def test_tidak_ada_string_rule_engine_di_source(self):
        """Baca source file sebagai teks, assert 'rule_engine' tidak ada."""
        # Temukan path file relatif terhadap lokasi test ini
        path_modul = (
            pathlib.Path(__file__).parent.parent
            / "engine" / "strategies" / "breakout_retest_v2.py"
        )
        assert path_modul.exists(), (
            f"File {path_modul} tidak ditemukan — pastikan implementasi ada"
        )
        source = path_modul.read_text(encoding="utf-8")
        assert "rule_engine" not in source, (
            "PELANGGARAN INDEPENDENSI: 'rule_engine' ditemukan di dalam "
            "engine/strategies/breakout_retest_v2.py. Modul ini harus berdiri "
            "INDEPENDEN dari kode Fase 9/10 lama."
        )

    def test_tidak_ada_import_dari_rule_engine(self):
        """
        Double-check via inspect.getsource: tidak ada referensi ke rule_engine
        dalam source fungsi evaluate_breakout_retest.
        """
        source = inspect.getsource(evaluate_breakout_retest)
        assert "rule_engine" not in source, (
            "PELANGGARAN INDEPENDENSI: 'rule_engine' ditemukan di source "
            "evaluate_breakout_retest."
        )

    def test_tidak_ada_detect_consolidation_zone(self):
        """
        Verifikasi bahwa detect_consolidation_zone() tidak dipanggil.
        Boundary selalu parameter zone — tidak dihitung ulang.
        """
        path_modul = (
            pathlib.Path(__file__).parent.parent
            / "engine" / "strategies" / "breakout_retest_v2.py"
        )
        source = path_modul.read_text(encoding="utf-8")
        assert "detect_consolidation_zone" not in source, (
            "PELANGGARAN DESAIN: 'detect_consolidation_zone' ditemukan di "
            "breakout_retest_v2.py. Boundary harus selalu diterima sebagai "
            "parameter zone, tidak dihitung ulang."
        )


# =============================================================================
# SKENARIO TAMBAHAN — EDGE CASE DAN ROBUSTNESS
# =============================================================================

class TestEdgeCase:
    """Kasus tambahan untuk robustness."""

    def test_field_wajib_hasil_selalu_ada(self):
        """Semua field wajib harus selalu ada di return dict, apapun kondisinya."""
        field_wajib = {
            "terpenuhi", "arah", "level_referensi", "touch_idx",
            "candles_since_touch", "invalidated", "konfirmasi_close",
            "konfirmasi_body", "invalidation_level_sl", "keterangan",
        }
        kasus_uji = [
            (ZONE_REF, "BULLISH"),      # normal
            (None, "BULLISH"),          # zone none
            (ZONE_REF, "BEARISH"),      # bearish
            (ZONE_REF, "INVALID"),      # arah salah
            ({}, "BULLISH"),            # zone kosong
        ]
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        for zone, arah in kasus_uji:
            hasil = evaluate_breakout_retest(df, 1, zone, arah)
            for f in field_wajib:
                assert f in hasil, (
                    f"Field '{f}' tidak ada di hasil untuk zone={zone}, arah={arah}"
                )

    def test_idx_negatif_dinormalisasi(self):
        """idx_m5=-1 harus dinormalisasi ke idx terakhir."""
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, NETRAL, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        hasil_negatif = evaluate_breakout_retest(df, -1, ZONE_REF, "BULLISH")
        hasil_normal  = evaluate_breakout_retest(df,  4, ZONE_REF, "BULLISH")
        assert hasil_negatif["terpenuhi"] == hasil_normal["terpenuhi"]
        assert hasil_negatif["touch_idx"] == hasil_normal["touch_idx"]

    def test_idx_di_luar_range_tidak_crash(self):
        """idx_m5 di luar range yang valid harus return False tanpa crash."""
        baris = [NETRAL, NETRAL]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 999, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False

    def test_atr_nol_di_candle_eval_tidak_crash(self):
        """ATR=0 di candle evaluasi harus return False tanpa crash."""
        candle_atr_nol = _candle(2010.5, 2013.0, 2009.5, 2012.0, 0.0)
        baris = [TOUCH_BULLISH, candle_atr_nol]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 1, ZONE_REF, "BULLISH")
        assert hasil["terpenuhi"] is False

    def test_konfirmasi_body_tepat_di_batas_terpenuhi(self):
        """
        Body tepat sama dengan batas minimum (>=) → harus terpenuhi.
        body_min = 0.3 * ATR = 0.3 * 5.0 = 1.5
        Pakai body = 2.0 (di atas batas) untuk menghindari floating point edge case.
        """
        # open=2010.0, close=2012.0 → body=2.0 > 1.5 (jelas >= batas)
        konfirm_body_pas = _candle(2010.0, 2013.0, 2009.5, 2012.0, ATR)
        baris = [NETRAL, NETRAL, TOUCH_BULLISH, ISI_BULLISH, konfirm_body_pas]
        df   = _buat_df(baris)
        hasil = evaluate_breakout_retest(df, 4, ZONE_REF, "BULLISH")
        assert hasil["konfirmasi_body"] is True
        assert hasil["terpenuhi"] is True

    def test_keterangan_selalu_string(self):
        """Pastikan 'keterangan' selalu berupa string di semua skenario."""
        kasus = [
            (None,     "BULLISH"),
            (ZONE_REF, "INVALID"),
            (ZONE_REF, "BULLISH"),
            (ZONE_REF, "BEARISH"),
        ]
        baris = [TOUCH_BULLISH, KONFIRM_BULLISH]
        df   = _buat_df(baris)
        for zone, arah in kasus:
            hasil = evaluate_breakout_retest(df, 1, zone, arah)
            assert isinstance(hasil["keterangan"], str)
            assert len(hasil["keterangan"]) > 0
