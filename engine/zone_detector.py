"""
engine/zone_detector.py
========================
Modul deteksi zona konsolidasi (range/sideways) untuk XAUUSD M5.

TUJUAN:
    Mendeteksi apakah harga sedang berada dalam zona konsolidasi — yaitu
    range harga yang relatif sempit dibanding volatilitas saat ini (ATR).
    Modul ini adalah INFRASTRUKTUR FONDASI untuk Fase 9 (breakout trigger).
    "Breakout dari APA?" membutuhkan zona yang terdefinisi secara presisi.

KARAKTER MODUL:
    - BUKAN komponen scoring dan BUKAN sinyal entry.
    - TIDAK diintegrasikan ke evaluate_entry() atau calculate_setup_quality().
    - Berdiri INDEPENDEN — hanya butuh DataFrame dengan kolom OHLC + atr_14.
    - Tidak import apapun dari rule_engine.py, risk_manager.py, atau modul
      engine lainnya.

DEFINISI KONSOLIDASI:
    Window mundur dari candle evaluasi (idx) dianggap konsolidasi jika:
        range_zone = max(high) - min(low) dalam window
        range_zone <= max_range_atr_ratio * atr_14[idx]

    Duration dihitung dengan ekspansi mundur candle demi candle dari idx:
    mulai dari [idx, idx] (1 candle), perlebar ke [idx-1, idx] (2 candle), dst.
    Berhenti saat range melebihi threshold atau mencapai lookback.

    Semua output (resistance, support, range_zone, range_atr_ratio) dihitung
    dari window sebesar duration (bukan lookback penuh).

    Zona dianggap valid jika duration >= min_duration_candles.

CATATAN PARAMETER:
    Ketiga parameter default (lookback=20, max_range_atr_ratio=2.5,
    min_duration_candles=10) BELUM dikalibrasi via backtest — ini nilai awal
    yang masuk akal secara struktural (sama seperti h1_min_ema_gap_pct=0.02
    sebelum walk-forward). Kalibrasi angka-angka ini, kalau diperlukan,
    adalah kerja terpisah setelah validasi akurasi deteksi (8.3) menunjukkan
    definisi dasarnya sudah masuk akal.

CATATAN KONSEPTUAL:
    Level resistance/support di sini adalah batas luar zona konsolidasi untuk
    keperluan deteksi breakout di Fase 9. Ini BUKAN level presisi seperti
    swing high/low (find_nearest_swing() di risk_manager.py, yang butuh
    konfirmasi wing kiri-kanan). Dua konsep ini beda tujuan meski sama-sama
    soal level harga — jangan reuse atau modifikasi find_nearest_swing().

CAUSALITY:
    Fungsi hanya membaca data pada dan sebelum index idx.
    Candle setelah idx TIDAK PERNAH disentuh.
"""

import pandas as pd
import numpy as np


# =============================================================================
# PARAMETER DEFAULT (BELUM DIKALIBRASI — lihat docstring modul)
# =============================================================================

DEFAULT_LOOKBACK = 20               # ~1 jam 40 menit di M5
DEFAULT_MAX_RANGE_ATR_RATIO = 2.5   # range zona maks 2.5x ATR saat ini
DEFAULT_MIN_DURATION_CANDLES = 10   # minimum panjang zona valid


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def detect_consolidation_zone(
    df: pd.DataFrame,
    idx: int = -1,
    lookback: int = DEFAULT_LOOKBACK,
    max_range_atr_ratio: float = DEFAULT_MAX_RANGE_ATR_RATIO,
    min_duration_candles: int = DEFAULT_MIN_DURATION_CANDLES,
) -> dict:
    """
    Deteksi zona konsolidasi pada candle ke-idx.

    Parameter:
        df                   : DataFrame dengan kolom high, low, close, atr_14.
                               Semua data harus candle CLOSED (sesuai konvensi
                               seluruh codebase — tidak menyentuh candle berjalan).
        idx                  : Index candle evaluasi (default -1 = candle terakhir).
                               Hanya data pada dan sebelum idx yang dibaca (causal).
        lookback             : Jumlah candle maksimum untuk ekspansi mundur.
                               Default 20 (~1j40m M5). BELUM dikalibrasi.
        max_range_atr_ratio  : Rasio maksimum range/ATR agar dianggap konsolidasi.
                               Default 2.5. BELUM dikalibrasi.
        min_duration_candles : Jumlah candle minimum agar zona dianggap valid.
                               Default 10. BELUM dikalibrasi.

    Return:
        dict dengan field:
            is_valid        : bool   — zona konsolidasi valid & cukup durasi
            resistance      : float | None — max(high) dalam window duration
            support         : float | None — min(low) dalam window duration
            range_zone      : float | None — resistance - support
            range_atr_ratio : float | None — range_zone / atr_14[idx]
            duration        : int    — berapa candle mundur zona tetap valid
                                       (0 jika data tidak cukup)
            keterangan      : str    — penjelasan ringkas hasil deteksi

    Catatan:
        - Jika data tidak cukup (idx < lookback - 1 setelah normalisasi),
          return is_valid=False dengan semua numerik=None, tanpa crash.
        - resistance/support/range_zone/range_atr_ratio dihitung dari window
          sebesar duration (bukan lookback penuh).
    """
    # ── Normalisasi idx negatif ──────────────────────────────────────────────
    n = len(df)
    if idx < 0:
        idx = n + idx

    # ── Validasi: data cukup? ────────────────────────────────────────────────
    required_cols = {"high", "low", "close", "atr_14"}
    missing = required_cols - set(df.columns)
    if missing:
        return _empty_result(
            duration=0,
            keterangan=f"Kolom tidak lengkap: {sorted(missing)} tidak ada di DataFrame",
        )

    if idx < lookback - 1 or idx >= n:
        return _empty_result(
            duration=0,
            keterangan=(
                f"Data tidak cukup: idx={idx}, lookback={lookback}, "
                f"butuh minimal idx >= {lookback - 1}, total baris={n}"
            ),
        )

    # ── Validasi ATR ─────────────────────────────────────────────────────────
    atr_value = df["atr_14"].iloc[idx]
    if pd.isna(atr_value) or atr_value <= 0:
        return _empty_result(
            duration=0,
            keterangan=f"atr_14 tidak valid di idx={idx}: {atr_value}",
        )

    # ── Hitung threshold ─────────────────────────────────────────────────────
    threshold = max_range_atr_ratio * atr_value

    # ── Ekspansi mundur: hitung duration ─────────────────────────────────────
    # Mulai dari 1 candle (hanya idx), perlebar ke belakang candle demi candle.
    # Di setiap langkah, cek range sub-window <= threshold.
    # Berhenti saat range > threshold atau sudah mencapai lookback.
    #
    # Untuk efisiensi, kita track running max(high) dan min(low) secara
    # inkremental — menambah satu candle ke belakang hanya perlu cek apakah
    # candle baru itu memperluas range.

    highs = df["high"].values  # numpy array untuk kecepatan akses
    lows = df["low"].values

    running_max_high = highs[idx]
    running_min_low = lows[idx]
    duration = 1  # minimal 1 candle (idx sendiri)

    # Batas mundur: idx - lookback + 1 (tapi tidak boleh < 0)
    earliest = max(idx - lookback + 1, 0)

    for k in range(idx - 1, earliest - 1, -1):
        candidate_max = max(running_max_high, highs[k])
        candidate_min = min(running_min_low, lows[k])
        candidate_range = candidate_max - candidate_min

        if candidate_range > threshold:
            break  # range melebihi threshold, berhenti ekspansi

        # Masih valid — update running values
        running_max_high = candidate_max
        running_min_low = candidate_min
        duration += 1

    # ── Hitung output dari window duration ───────────────────────────────────
    resistance = float(running_max_high)
    support = float(running_min_low)
    range_zone = resistance - support
    range_atr_ratio = range_zone / float(atr_value)

    # ── Cek minimum durasi ───────────────────────────────────────────────────
    is_valid = duration >= min_duration_candles

    # ── Keterangan ───────────────────────────────────────────────────────────
    if is_valid:
        keterangan = (
            f"KONSOLIDASI VALID: duration={duration} candle "
            f"(>= min {min_duration_candles}), "
            f"range={range_zone:.2f} ({range_atr_ratio:.2f}x ATR), "
            f"R={resistance:.2f}, S={support:.2f}"
        )
    else:
        keterangan = (
            f"TIDAK VALID: duration={duration} candle "
            f"(< min {min_duration_candles}), "
            f"range={range_zone:.2f} ({range_atr_ratio:.2f}x ATR), "
            f"R={resistance:.2f}, S={support:.2f}"
        )

    return {
        "is_valid": is_valid,
        "resistance": resistance,
        "support": support,
        "range_zone": range_zone,
        "range_atr_ratio": range_atr_ratio,
        "duration": duration,
        "keterangan": keterangan,
    }


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _empty_result(duration: int, keterangan: str) -> dict:
    """Return dict standar untuk kasus data tidak cukup / error."""
    return {
        "is_valid": False,
        "resistance": None,
        "support": None,
        "range_zone": None,
        "range_atr_ratio": None,
        "duration": duration,
        "keterangan": keterangan,
    }
