"""
engine/market_context.py
========================
Modul H1 Context Layer — fondasi deteksi regime market (Fase 12).

TUJUAN:
    Membungkus output fungsi detect_bias_h1() yang sudah ada di
    engine/indicators.py menjadi satu dict context H1 yang terstruktur.
    Context ini akan dikonsumsi oleh:
        - Regime Detector M15 (Fase 13, belum dikerjakan)
        - Strategy Layer (Fase 14 dan seterusnya, belum dikerjakan)

    Fase 12 adalah INFRASTRUKTUR murni — tidak ada keputusan BUY/SELL/WAIT
    di sini. Anggap modul ini seperti session_filter.py yang sudah ada:
    informatif, read-only, tidak mengubah apapun di pipeline keputusan.

KARAKTER MODUL:
    - BUKAN sinyal entry dan BUKAN komponen scoring.
    - TIDAK diintegrasikan ke evaluate_entry() atau rule_engine.py manapun
      di Fase 12. Integrasi ke pipeline trading adalah pekerjaan Fase 13+.
    - TIDAK mengubah engine/indicators.py, web/app.py, atau backtester.py.
    - Berdiri INDEPENDEN — hanya butuh pandas dan engine.indicators.
    - Tidak ada state global — recompute bersih setiap panggilan.

CAUSALITY (ZERO LOOK-AHEAD):
    get_h1_context(df_h1_ind, idx) HANYA membaca data pada dan sebelum idx.
    Candle df_h1_ind.iloc[idx+1:] tidak pernah disentuh dalam bentuk apapun.

    Cara kerja:
        detect_bias_h1() bekerja row-wise menggunakan ema_9, ema_21, close
        dari baris yang sama (tidak ada .shift(-N) atau akses masa depan).
        Setelah detect_bias_h1() selesai, get_h1_context() hanya membaca
        baris ke-idx (dan tidak menyentuh baris manapun setelahnya).
        Kausalitas terbukti secara konstruksi dan diverifikasi via test
        mutasi ekstrem di tests/test_market_context.py.

CATATAN PARAMETER (BELUM DIKALIBRASI):
    H1_STRENGTH_STRONG_THRESHOLD   = 0.15 %
    H1_STRENGTH_MODERATE_THRESHOLD = 0.05 %

    Kedua angka di atas adalah STARTING POINT yang masuk akal secara
    struktural, BUKAN hasil kalibrasi backtest. Pola ini sama seperti
    parameter di zone_detector.py (lookback=20, max_range_atr_ratio=2.5,
    min_duration=10) dan supply_demand.py (DEFAULT_IMPULSIVE_RATIO=1.5,
    ORIGIN_BODY_MAX_RATIO=0.5) yang semuanya ditandai "BELUM DIKALIBRASI"
    dan menunggu validasi empiris di fase-fase mendatang.
    JANGAN ubah nilai ini tanpa approval eksplisit setelah validasi empiris.
"""

import pandas as pd
from typing import Any

from engine.indicators import detect_bias_h1


# =============================================================================
# PARAMETER THRESHOLD STRENGTH (BELUM DIKALIBRASI — lihat docstring modul)
# =============================================================================

H1_STRENGTH_STRONG_THRESHOLD   = 0.15  # gap pct >= 0.15% → STRONG.  BELUM dikalibrasi.
H1_STRENGTH_MODERATE_THRESHOLD = 0.05  # gap pct >= 0.05% → MODERATE. BELUM dikalibrasi.
#
# Alasan starting point:
#   0.05% adalah batas yang sudah dipakai di detect_trend() (M5) sebagai
#   sinyal "gap sudah cukup lebar untuk dianggap trending". Karena H1
#   bergerak lebih lambat dari M5, kita pakai 0.05% sebagai batas bawah
#   MODERATE (bukan WEAK) — artinya 0.05% di H1 sudah cukup meaningful.
#   0.15% dipilih sebagai batas STRONG (3× MODERATE) — gap sebesar ini
#   di H1 biasanya menandakan trend H1 yang sudah mapan dan kuat.
#   Kedua angka ini akan dikalibrasi via validasi empiris setelah regime
#   detector (Fase 13) selesai dan ada data backtest yang cukup.


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _empty_context(keterangan: str) -> dict:
    """
    Return dict standar untuk kasus data tidak cukup / error / edge case.

    APA INI:
        Helper internal yang dipakai oleh get_h1_context() dan
        get_h1_context_from_precomputed() kapanpun kondisi input tidak
        memenuhi syarat (DataFrame kosong, idx di luar range, kolom hilang).

    KENAPA bias="NEUTRAL" sebagai default:
        Ketika kita tidak punya informasi arah H1 yang valid, anggapan
        paling aman adalah "tidak ada bias" (NEUTRAL), bukan BULLISH/BEARISH.
        Ini mencegah regime detector (Fase 13) mengambil keputusan salah
        berdasarkan data tidak valid.

    Pola ini mengikuti _empty_result() di engine/zone_detector.py dan
    _empty_zone() di engine/supply_demand.py.
    """
    return {
        "bias"         : "NEUTRAL",
        "strength"     : None,
        "strength_zone": None,
        "ema_gap_pct"  : None,
        "close"        : None,
        "time"         : None,
        "keterangan"   : keterangan,
    }


# =============================================================================
# HELPER INTERNAL: KATEGORISASI STRENGTH
# =============================================================================

def _kategorisasi_strength(strength: float) -> str:
    """
    Kategorikan nilai strength (float, >= 0.0) menjadi label zona.

    Threshold menggunakan konstanta modul yang sudah didefinisikan di atas.
    Fungsi ini dipakai oleh get_h1_context() dan get_h1_context_from_precomputed()
    agar logika kategorisasi tidak duplikat.
    """
    if strength >= H1_STRENGTH_STRONG_THRESHOLD:
        return "STRONG"
    elif strength >= H1_STRENGTH_MODERATE_THRESHOLD:
        return "MODERATE"
    else:
        return "WEAK"


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def get_h1_context(df_h1_ind: pd.DataFrame, idx: int = -1) -> dict:
    """
    Menghasilkan dict context H1 dari satu candle evaluasi.

    APA INI:
        Membungkus detect_bias_h1() menjadi output terstruktur berisi
        bias arah, kekuatan (magnitude gap EMA), dan metadata candle H1
        di posisi idx. Output ini akan dikonsumsi oleh regime detector M15
        (Fase 13) dan strategy layer (Fase 14+).

    KENAPA DIPISAH DARI detect_bias_h1():
        detect_bias_h1() menghasilkan kolom di DataFrame (untuk pipeline
        backtest/live yang butuh kolom 'bias_h1' per baris). Fungsi ini
        menghasilkan dict satu titik waktu yang kaya informasi — format
        yang lebih cocok untuk dikonsumsi oleh komponen downstream yang
        butuh "konteks H1 saat ini" bukan seluruh kolom historis.

    CAUSALITY:
        Fungsi ini hanya membaca df_h1_ind pada dan sebelum idx.
        detect_bias_h1() sendiri bersifat causal (row-wise, tidak ada
        look-ahead). Setelah itu, kita hanya membaca baris ke-idx.
        Baris idx+1 dan seterusnya tidak pernah disentuh.

    Parameter:
        df_h1_ind : DataFrame H1 yang SUDAH melewati run_all_indicators()
                    dari engine/indicators.py — harus punya kolom:
                    'close', 'ema_9', 'ema_21' (butuh untuk detect_bias_h1).
                    Kolom 'ema_gap_pct_h1_raw' akan ditambahkan otomatis
                    oleh detect_bias_h1() jika belum ada.
        idx       : Index candle H1 evaluasi.
                    Default -1 = candle H1 terbaru (closed terakhir).
                    Boleh negatif — normalisasi mengikuti pola di
                    zone_detector.py dan supply_demand.py.

    Return:
        dict dengan field:
            bias         : str   — "BULLISH" / "BEARISH" / "NEUTRAL"
            strength     : float — abs(ema_gap_pct) di idx, atau None jika error
            strength_zone: str   — "STRONG" / "MODERATE" / "WEAK", atau None
            ema_gap_pct  : float — nilai raw ema_gap_pct_h1_raw di idx (bertanda),
                                   atau None jika error
            close        : float — harga close candle H1 di idx, atau None
            time         : Any   — timestamp df_h1_ind.index[idx], atau None
            keterangan   : str   — penjelasan ringkas hasil context

    Catatan edge case:
        - DataFrame kosong → return _empty_context() tanpa crash.
        - idx di luar range setelah normalisasi → return _empty_context().
        - Kolom yang dibutuhkan tidak ada → return _empty_context() dengan
          keterangan yang menyebutkan kolom mana yang hilang. Tidak pernah
          raise exception — fungsi ini aman dipanggil di jalur live.
    """
    # ── Validasi DataFrame kosong ────────────────────────────────────────────
    if len(df_h1_ind) == 0:
        return _empty_context("DataFrame H1 kosong (0 baris) — tidak ada data untuk dievaluasi")

    # ── Normalisasi idx negatif ──────────────────────────────────────────────
    n = len(df_h1_ind)
    if idx < 0:
        idx = n + idx

    # ── Validasi idx dalam range ─────────────────────────────────────────────
    if idx < 0 or idx >= n:
        return _empty_context(
            f"idx={idx} di luar range valid [0, {n - 1}] — "
            f"DataFrame punya {n} baris"
        )

    # ── Validasi kolom wajib ────────────────────────────────────────────────
    required_cols = {"close", "ema_9", "ema_21"}
    missing = required_cols - set(df_h1_ind.columns)
    if missing:
        return _empty_context(
            f"Kolom tidak lengkap — kolom berikut tidak ada di DataFrame: "
            f"{sorted(missing)}. Pastikan df_h1_ind sudah melewati "
            f"run_all_indicators() dan calculate_ema()."
        )

    # ── Panggil detect_bias_h1() untuk mendapatkan kolom bias_h1 ────────────
    # Fungsi ini bersifat causal (row-wise) dan mengembalikan DataFrame baru
    # dengan kolom 'bias_h1' dan 'ema_gap_pct_h1_raw' ditambahkan.
    # Kita pakai min_ema_gap_pct=0.0 (default) sesuai spesifikasi Fase 12.
    df_with_bias = detect_bias_h1(df_h1_ind, min_ema_gap_pct=0.0)

    # ── Baca nilai di posisi idx (HANYA baris ini) ───────────────────────────
    bias_h1_raw = df_with_bias["bias_h1"].iloc[idx]
    ema_gap_pct = float(df_with_bias["ema_gap_pct_h1_raw"].iloc[idx])
    close_val   = float(df_with_bias["close"].iloc[idx])
    time_val    = df_with_bias.index[idx]

    # ── Petakan bias_h1 raw → label standar context ──────────────────────────
    _peta_bias = {
        "UPTREND"  : "BULLISH",
        "DOWNTREND": "BEARISH",
        "SIDEWAYS" : "NEUTRAL",
    }
    bias = _peta_bias.get(bias_h1_raw, "NEUTRAL")

    # ── Hitung strength = magnitude gap EMA (sudah absolut) ─────────────────
    strength      = abs(ema_gap_pct)
    strength_zone = _kategorisasi_strength(strength)

    # ── Format string keterangan ─────────────────────────────────────────────
    tanda = "+" if ema_gap_pct >= 0 else ""   # tanda eksplisit untuk positif
    keterangan = (
        f"H1 Context: {bias} ({strength_zone}, gap={tanda}{ema_gap_pct:.2f}%) "
        f"-- candle H1 @ {time_val}, close={close_val:.2f}"
    )

    return {
        "bias"         : bias,
        "strength"     : strength,
        "strength_zone": strength_zone,
        "ema_gap_pct"  : ema_gap_pct,
        "close"        : close_val,
        "time"         : time_val,
        "keterangan"   : keterangan,
    }


# =============================================================================
# FUNGSI OPSIONAL: EFISIENSI BACKTEST (precomputed)
# =============================================================================

def get_h1_context_from_precomputed(df_h1_with_bias: pd.DataFrame, idx: int = -1) -> dict:
    """
    Sama seperti get_h1_context(), tapi mengasumsikan df_h1_with_bias SUDAH
    punya kolom 'bias_h1' dan 'ema_gap_pct_h1_raw' (sudah melewati
    detect_bias_h1() sebelumnya).

    APA INI:
        Varian efisien dari get_h1_context() untuk keperluan backtest.
        Saat backtest, caller menghitung detect_bias_h1() SEKALI di awal
        untuk seluruh DataFrame, lalu memanggil fungsi ini berkali-kali
        per candle tanpa overhead recompute detect_bias_h1().

    KENAPA DIPISAH:
        detect_bias_h1() menghitung ulang kolom untuk SELURUH DataFrame
        setiap kali dipanggil — O(N) per panggilan. Saat backtest dengan
        ribuan candle dan get_h1_context() dipanggil per-candle, overhead
        ini bisa signifikan. Dengan varian ini, overhead jadi O(1) per
        panggilan karena kolom sudah ada.

    CAUSALITY:
        Fungsi ini tidak memanggil detect_bias_h1() sama sekali — ia
        langsung membaca kolom yang sudah ada. Kausalitas dijamin oleh
        caller yang memastikan df_h1_with_bias dihitung secara causal
        sebelum dipassing ke sini. Fungsi ini sendiri hanya membaca
        baris ke-idx dan tidak menyentuh baris setelahnya.

    Parameter:
        df_h1_with_bias : DataFrame H1 yang SUDAH punya kolom:
                          'bias_h1', 'ema_gap_pct_h1_raw', 'close'.
                          Kolom ini dihasilkan oleh detect_bias_h1().
        idx             : Index candle H1 evaluasi (default -1).
                          Normalisasi idx negatif dipakai sama seperti
                          get_h1_context().

    Return:
        dict dengan field yang IDENTIK dengan get_h1_context().

    Catatan:
        Jika kolom 'bias_h1' atau 'ema_gap_pct_h1_raw' tidak ada di
        df_h1_with_bias, return _empty_context() dengan penjelasan yang
        menyarankan caller memanggil detect_bias_h1() terlebih dahulu.
    """
    # ── Validasi DataFrame kosong ────────────────────────────────────────────
    if len(df_h1_with_bias) == 0:
        return _empty_context("DataFrame H1 kosong (0 baris) — tidak ada data untuk dievaluasi")

    # ── Normalisasi idx negatif ──────────────────────────────────────────────
    n = len(df_h1_with_bias)
    if idx < 0:
        idx = n + idx

    # ── Validasi idx dalam range ─────────────────────────────────────────────
    if idx < 0 or idx >= n:
        return _empty_context(
            f"idx={idx} di luar range valid [0, {n - 1}] — "
            f"DataFrame punya {n} baris"
        )

    # ── Validasi kolom wajib (termasuk kolom precomputed) ───────────────────
    required_cols = {"close", "bias_h1", "ema_gap_pct_h1_raw"}
    missing = required_cols - set(df_h1_with_bias.columns)
    if missing:
        return _empty_context(
            f"Kolom tidak lengkap — kolom berikut tidak ada: {sorted(missing)}. "
            f"Pastikan df_h1_with_bias sudah melewati detect_bias_h1() "
            f"sebelum dipassing ke get_h1_context_from_precomputed()."
        )

    # ── Baca nilai di posisi idx (HANYA baris ini) ───────────────────────────
    bias_h1_raw = df_h1_with_bias["bias_h1"].iloc[idx]
    ema_gap_pct = float(df_h1_with_bias["ema_gap_pct_h1_raw"].iloc[idx])
    close_val   = float(df_h1_with_bias["close"].iloc[idx])
    time_val    = df_h1_with_bias.index[idx]

    # ── Petakan bias_h1 raw → label standar context ──────────────────────────
    _peta_bias = {
        "UPTREND"  : "BULLISH",
        "DOWNTREND": "BEARISH",
        "SIDEWAYS" : "NEUTRAL",
    }
    bias = _peta_bias.get(bias_h1_raw, "NEUTRAL")

    # ── Hitung strength = magnitude gap EMA (sudah absolut) ─────────────────
    strength      = abs(ema_gap_pct)
    strength_zone = _kategorisasi_strength(strength)

    # ── Format string keterangan ─────────────────────────────────────────────
    tanda = "+" if ema_gap_pct >= 0 else ""
    keterangan = (
        f"H1 Context: {bias} ({strength_zone}, gap={tanda}{ema_gap_pct:.2f}%) "
        f"-- candle H1 @ {time_val}, close={close_val:.2f}"
    )

    return {
        "bias"         : bias,
        "strength"     : strength,
        "strength_zone": strength_zone,
        "ema_gap_pct"  : ema_gap_pct,
        "close"        : close_val,
        "time"         : time_val,
        "keterangan"   : keterangan,
    }
