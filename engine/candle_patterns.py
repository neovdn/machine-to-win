"""
engine/candle_patterns.py
=========================
Modul deteksi pola candlestick untuk sistem trading XAUUSD M5.

PERAN DALAM ARSITEKTUR:
    Modul ini adalah bagian dari LAPIS KEPERCAYAAN (confidence scoring), bukan
    lapis keputusan. Tidak ada fungsi di sini yang mempengaruhi BUY/SELL/WAIT.
    Semua fungsi menghasilkan sinyal informatif yang dikonsumsi oleh
    calculate_candle_pattern_score(), yang kemudian dipanggil dari
    calculate_setup_quality() di engine/rule_engine.py.

PRINSIP KAUSALITAS (Zero Lookahead):
    Semua deteksi HANYA membaca data pada index yang dicek (idx) dan candle
    SEBELUMNYA (idx-1, idx-2, ...). Tidak ada pembacaan data setelah index tsb.
    Ini konsisten dengan filosofi causal yang dibuktikan oleh validate_no_lookahead()
    di engine/backtester.py.

POLA YANG DIDUKUNG (Fase 7 Opsi B):
    (A) Bullish / Bearish Engulfing — momentum reversal dari body yang menelan
    (Pin Bar dan Marubozu dihapus pada iterasi Opsi B karena jarang aktif di entry M5)

SKEMA SKOR (dari calculate_candle_pattern_score()):
    2 poin : pattern searah terdeteksi DAN dekat swing level (konteks valid)
    1 poin : pattern searah terdeteksi TAPI jauh dari swing / swing None
    0 poin : tidak ada pattern searah terdeteksi

ASUMSI TERTULIS (Fase 7):
    - atr_value dilewatkan secara eksplisit dari calculate_setup_quality(),
      yang mengambilnya dari signals.get("atr_14", 1.5). Modul ini tidak
      membaca atr_14 langsung dari df untuk menghindari dependensi implisit.
    - swing_low dan swing_high dilewatkan dari signals (sudah di-inject oleh
      pipeline backtester dan web/app.py). Modul ini tidak memanggil
      find_nearest_swing() sendiri.
    - Jika df kurang dari 2 baris (tidak ada candle sebelumnya), semua deteksi
      akan mengembalikan terpenuhi=False. Ini graceful degradation, bukan error.

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.
"""

import pandas as pd


# =============================================================================
# HELPER INTERNAL — Ekstraksi nilai OHLC dari satu baris
# =============================================================================

def _get_candle_values(df: pd.DataFrame, idx: int) -> dict | None:
    """
    Ekstrak nilai OHLC + ATR dari baris tertentu di DataFrame.

    Parameter:
        df  : DataFrame yang sudah punya kolom open, high, low, close, atr_14
        idx : Posisi baris (iloc index, integer positif)

    Return:
        dict berisi open, high, low, close, body, range_, upper_wick,
        lower_wick, is_bullish, is_bearish
        atau None jika idx tidak valid.

    CAUSAL NOTE: Fungsi ini hanya membaca satu baris pada idx yang diminta.
    Caller bertanggung jawab untuk tidak melewatkan idx > candle yang sedang dicek.
    """
    if idx < 0 or idx >= len(df):
        return None

    row   = df.iloc[idx]
    o     = float(row["open"])
    h     = float(row["high"])
    l     = float(row["low"])
    c     = float(row["close"])

    body        = abs(c - o)
    range_      = h - l
    upper_wick  = h - max(o, c)
    lower_wick  = min(o, c) - l
    is_bullish  = c > o
    is_bearish  = c < o

    return {
        "open"       : o,
        "high"       : h,
        "low"        : l,
        "close"      : c,
        "body"       : body,
        "range_"     : range_,
        "upper_wick" : upper_wick,
        "lower_wick" : lower_wick,
        "is_bullish" : is_bullish,
        "is_bearish" : is_bearish,
    }


def _resolve_idx(df: pd.DataFrame, idx: int) -> int:
    """
    Konversi idx (mungkin negatif seperti -1) ke posisi integer absolut.

    Return:
        Integer index positif, atau -1 jika tidak valid.
    """
    n = len(df)
    if idx < 0:
        pos = n + idx   # -1 → n-1, -2 → n-2, dst
    else:
        pos = idx

    if pos < 0 or pos >= n:
        return -1
    return pos


# =============================================================================
# (A) DETEKSI ENGULFING
# =============================================================================

def detect_bullish_engulfing(df: pd.DataFrame, idx: int = -1) -> dict:
    """
    Deteksi pola Bullish Engulfing pada candle index idx.

    DEFINISI MATEMATIS:
        Candle i   = candle yang dicek (idx)
        Candle i-1 = candle sebelumnya

        Syarat:
          1. is_bearish(i-1) — candle sebelumnya merah/turun
          2. is_bullish(i)   — candle saat ini hijau/naik
          3. open(i)  <= close(i-1) — buka di bawah atau sama dengan tutup candle sebelumnya
          4. close(i) >= open(i-1)  — tutup di atas atau sama dengan buka candle sebelumnya
             [body candle i menelan penuh body candle i-1]
          5. body(i) >= 0.3 * atr_14(i) — bukan candle terlalu kecil / noise
             (Opsi A kalibrasi: diturunkan dari 0.5 → 0.3 agar lebih banyak candle
              yang lolos filter, target coverage 25-45% dari sebelumnya 9%)

    CAUSAL: Hanya membaca candle idx dan idx-1.

    Parameter:
        df  : DataFrame dengan kolom open, high, low, close, atr_14
        idx : Index candle yang dicek (default -1 = candle terakhir)

    Return dict:
        {
            "terpenuhi"  : bool   — True jika pola terdeteksi
            "arah"       : str    — "BUY" jika terpenuhi, "NETRAL" jika tidak
            "keterangan" : str    — penjelasan singkat
        }
    """
    pos = _resolve_idx(df, idx)

    if pos < 1:
        return {
            "terpenuhi"  : False,
            "arah"       : "NETRAL",
            "keterangan" : "Data tidak cukup (butuh minimal 2 candle untuk Engulfing)",
        }

    c_cur  = _get_candle_values(df, pos)
    c_prev = _get_candle_values(df, pos - 1)

    if c_cur is None or c_prev is None:
        return {
            "terpenuhi"  : False,
            "arah"       : "NETRAL",
            "keterangan" : "Gagal membaca data candle",
        }

    # Cek semua syarat Bullish Engulfing
    syarat_bearish_prev = c_prev["is_bearish"]
    syarat_bullish_cur  = c_cur["is_bullish"]
    syarat_open         = c_cur["open"]  <= c_prev["close"]
    syarat_close        = c_cur["close"] >= c_prev["open"]

    # Baca atr_14 dari df untuk filter noise (causal: baca dari baris idx)
    atr_val = float(df.iloc[pos].get("atr_14", 0.0)) if "atr_14" in df.columns else 0.0
    syarat_body = c_cur["body"] >= 0.3 * atr_val if atr_val > 0 else True  # Opsi A: 0.5 → 0.3

    if (syarat_bearish_prev and syarat_bullish_cur
            and syarat_open and syarat_close and syarat_body):
        return {
            "terpenuhi"  : True,
            "arah"       : "BUY",
            "keterangan" : (
                f"Bullish Engulfing: body {c_cur['body']:.2f} menelan body sebelumnya "
                f"({c_prev['open']:.2f}→{c_prev['close']:.2f}), "
                f"candle saat ini {c_cur['open']:.2f}→{c_cur['close']:.2f}"
            ),
        }

    # Bangun keterangan mengapa tidak terpenuhi (audit)
    alasan = []
    if not syarat_bearish_prev:
        alasan.append("candle sebelumnya tidak bearish")
    if not syarat_bullish_cur:
        alasan.append("candle saat ini tidak bullish")
    if not syarat_open:
        alasan.append(f"open({c_cur['open']:.2f}) > close_prev({c_prev['close']:.2f})")
    if not syarat_close:
        alasan.append(f"close({c_cur['close']:.2f}) < open_prev({c_prev['open']:.2f})")
    if not syarat_body:
        alasan.append(f"body({c_cur['body']:.2f}) < 0.5*ATR({0.5*atr_val:.2f})")

    return {
        "terpenuhi"  : False,
        "arah"       : "NETRAL",
        "keterangan" : "Bukan Bullish Engulfing: " + "; ".join(alasan),
    }


def detect_bearish_engulfing(df: pd.DataFrame, idx: int = -1) -> dict:
    """
    Deteksi pola Bearish Engulfing pada candle index idx.

    DEFINISI MATEMATIS (kebalikan dari Bullish Engulfing):
        Candle i   = candle yang dicek (idx)
        Candle i-1 = candle sebelumnya

        Syarat:
          1. is_bullish(i-1) — candle sebelumnya hijau/naik
          2. is_bearish(i)   — candle saat ini merah/turun
          3. open(i)  >= close(i-1) — buka di atas atau sama dengan tutup candle sebelumnya
          4. close(i) <= open(i-1)  — tutup di bawah atau sama dengan buka candle sebelumnya
             [body candle i menelan penuh body candle i-1]
          5. body(i) >= 0.3 * atr_14(i) — bukan candle terlalu kecil / noise
             (Opsi A kalibrasi: diturunkan dari 0.5 → 0.3)

    CAUSAL: Hanya membaca candle idx dan idx-1.

    Return dict:
        {
            "terpenuhi"  : bool   — True jika pola terdeteksi
            "arah"       : str    — "SELL" jika terpenuhi, "NETRAL" jika tidak
            "keterangan" : str    — penjelasan singkat
        }
    """
    pos = _resolve_idx(df, idx)

    if pos < 1:
        return {
            "terpenuhi"  : False,
            "arah"       : "NETRAL",
            "keterangan" : "Data tidak cukup (butuh minimal 2 candle untuk Engulfing)",
        }

    c_cur  = _get_candle_values(df, pos)
    c_prev = _get_candle_values(df, pos - 1)

    if c_cur is None or c_prev is None:
        return {
            "terpenuhi"  : False,
            "arah"       : "NETRAL",
            "keterangan" : "Gagal membaca data candle",
        }

    syarat_bullish_prev = c_prev["is_bullish"]
    syarat_bearish_cur  = c_cur["is_bearish"]
    syarat_open         = c_cur["open"]  >= c_prev["close"]
    syarat_close        = c_cur["close"] <= c_prev["open"]

    atr_val = float(df.iloc[pos].get("atr_14", 0.0)) if "atr_14" in df.columns else 0.0
    syarat_body = c_cur["body"] >= 0.3 * atr_val if atr_val > 0 else True  # Opsi A: 0.5 → 0.3

    if (syarat_bullish_prev and syarat_bearish_cur
            and syarat_open and syarat_close and syarat_body):
        return {
            "terpenuhi"  : True,
            "arah"       : "SELL",
            "keterangan" : (
                f"Bearish Engulfing: body {c_cur['body']:.2f} menelan body sebelumnya "
                f"({c_prev['open']:.2f}→{c_prev['close']:.2f}), "
                f"candle saat ini {c_cur['open']:.2f}→{c_cur['close']:.2f}"
            ),
        }

    alasan = []
    if not syarat_bullish_prev:
        alasan.append("candle sebelumnya tidak bullish")
    if not syarat_bearish_cur:
        alasan.append("candle saat ini tidak bearish")
    if not syarat_open:
        alasan.append(f"open({c_cur['open']:.2f}) < close_prev({c_prev['close']:.2f})")
    if not syarat_close:
        alasan.append(f"close({c_cur['close']:.2f}) > open_prev({c_prev['open']:.2f})")
    if not syarat_body:
        alasan.append(f"body({c_cur['body']:.2f}) < 0.5*ATR({0.5*atr_val:.2f})")

    return {
        "terpenuhi"  : False,
        "arah"       : "NETRAL",
        "keterangan" : "Bukan Bearish Engulfing: " + "; ".join(alasan),
    }





# =============================================================================
# FUNGSI AGREGAT: CALCULATE CANDLE PATTERN SCORE
# =============================================================================

def calculate_candle_pattern_score(
    df            : pd.DataFrame,
    arah_kandidat : str,
    swing_low     : float | None,
    swing_high    : float | None,
    atr_value     : float,
) -> dict:
    """
    Hitung skor komponen candlestick pattern (0-2 poin) untuk arah kandidat tertentu.

    FUNGSI INI ADALAH AGGREGATOR — dipanggil dari calculate_setup_quality() di
    engine/rule_engine.py. Hasil dikembalikan sebagai satu dict komponen yang
    konsisten dengan pola komponen lain di fungsi tersebut.

    ALUR KERJA:
        1. Deteksi pattern yang searah dengan arah_kandidat di candle terakhir (iloc[-1])
        2. Jika pattern ditemukan, cek konteks swing (jarak ke swing level)
        3. Tentukan skor: 2 (pattern + swing dekat), 1 (pattern saja), 0 (tidak ada)

    ATURAN ARAH:
        arah_kandidat == "BUY"  → hanya cek pattern Bullish Engulfing
        arah_kandidat == "SELL" → hanya cek pattern Bearish Engulfing
        arah_kandidat == "NETRAL" atau lainnya → skor 0 (tidak ada pattern relevan untuk dicek)

    KONTEKS SWING (skor 2 vs 1):
        BUY  → pattern valid jika close(i) berada dalam 1.0 * atr_14 dari swing_low
        SELL → pattern valid jika close(i) berada dalam 1.0 * atr_14 dari swing_high
        Jika swing_low/swing_high == None → kondisi jarak TIDAK terpenuhi (skor 1, bukan 2)

    MULTIPLE PATTERN:
        (Tidak relevan lagi pada Opsi B karena hanya Engulfing yang dideteksi, tapi field 
        "pattern_detected" dipertahankan untuk kompatibilitas audit)

    CAUSAL:
        Semua deteksi di sini menggunakan df.iloc[-1] dan df.iloc[-2].
        Caller harus memastikan df sudah di-slice hingga index i sebelum memanggil
        fungsi ini (df = df_m5_ind.iloc[:i+1]).

    Parameter:
        df            : DataFrame M5 yang sudah melewati run_all_indicators(),
                        di-slice hingga candle yang sedang dievaluasi.
                        Harus punya kolom open, high, low, close, atr_14.
        arah_kandidat : "BUY", "SELL", atau string lain (→ skor 0)
        swing_low     : Nilai swing low dari signals (None jika tidak tersedia)
        swing_high    : Nilai swing high dari signals (None jika tidak tersedia)
        atr_value     : Nilai ATR dari signals (dipakai untuk cek jarak swing)

    Return dict (mengikuti pola komponen lain di calculate_setup_quality):
        {
            "score"            : int    — 0, 1, atau 2
            "max"              : 2
            "label"            : "Candlestick Pattern"
            "detail"           : str    — penjelasan pattern + status swing
            "pattern_detected" : str | None  — nama pattern untuk audit
        }
    """
    # ── Guard: DataFrame kosong atau kurang dari 1 baris ──────────────────────
    if df is None or len(df) == 0:
        return {
            "score"            : 0,
            "max"              : 2,
            "label"            : "Candlestick Pattern",
            "detail"           : "DataFrame tidak tersedia — skor 0",
            "pattern_detected" : None,
        }

    # ── Guard: arah_kandidat tidak valid untuk pattern check ─────────────────
    if arah_kandidat not in ("BUY", "SELL"):
        return {
            "score"            : 0,
            "max"              : 2,
            "label"            : "Candlestick Pattern",
            "detail"           : f"Arah kandidat '{arah_kandidat}' — tidak ada pattern dicek",
            "pattern_detected" : None,
        }

    # ── Deteksi semua pattern yang relevan untuk arah kandidat ───────────────
    # Hanya cek pattern yang searah — jangan mixed direction
    nama_pattern_ditemukan: list[str] = []
    keterangan_pattern: list[str] = []

    if arah_kandidat == "BUY":
        # Pattern bullish yang dicek
        r_engulf = detect_bullish_engulfing(df, idx=-1)

        if r_engulf["terpenuhi"]:
            nama_pattern_ditemukan.append("BULLISH_ENGULFING")
            keterangan_pattern.append(r_engulf["keterangan"])

    else:  # SELL
        # Pattern bearish yang dicek
        r_engulf = detect_bearish_engulfing(df, idx=-1)

        if r_engulf["terpenuhi"]:
            nama_pattern_ditemukan.append("BEARISH_ENGULFING")
            keterangan_pattern.append(r_engulf["keterangan"])

    # ── Tidak ada pattern terdeteksi → skor 0 ────────────────────────────────
    if not nama_pattern_ditemukan:
        return {
            "score"            : 0,
            "max"              : 2,
            "label"            : "Candlestick Pattern",
            "detail"           : f"Tidak ada pattern {arah_kandidat} terdeteksi di candle terakhir",
            "pattern_detected" : None,
        }

    # ── Ada pattern → cek konteks swing (apakah dekat swing level?) ──────────
    close_price = float(df.iloc[-1]["close"])
    konteks_swing_terpenuhi = False
    keterangan_swing = ""

    if arah_kandidat == "BUY":
        if swing_low is not None and atr_value > 0:
            jarak_ke_swing = abs(close_price - swing_low)
            batas_jarak    = 1.0 * atr_value
            konteks_swing_terpenuhi = jarak_ke_swing <= batas_jarak
            keterangan_swing = (
                f"Jarak ke swing_low={swing_low:.2f}: {jarak_ke_swing:.2f} "
                f"({'<=' if konteks_swing_terpenuhi else '>'} 1.0*ATR {batas_jarak:.2f})"
            )
        else:
            keterangan_swing = (
                "swing_low tidak tersedia" if swing_low is None
                else "ATR=0, tidak bisa hitung jarak"
            )
    else:  # SELL
        if swing_high is not None and atr_value > 0:
            jarak_ke_swing = abs(swing_high - close_price)
            batas_jarak    = 1.0 * atr_value
            konteks_swing_terpenuhi = jarak_ke_swing <= batas_jarak
            keterangan_swing = (
                f"Jarak ke swing_high={swing_high:.2f}: {jarak_ke_swing:.2f} "
                f"({'<=' if konteks_swing_terpenuhi else '>'} 1.0*ATR {batas_jarak:.2f})"
            )
        else:
            keterangan_swing = (
                "swing_high tidak tersedia" if swing_high is None
                else "ATR=0, tidak bisa hitung jarak"
            )

    # ── Tentukan skor final ───────────────────────────────────────────────────
    nama_gabungan = "+".join(nama_pattern_ditemukan)
    if konteks_swing_terpenuhi:
        score  = 2
        detail = (
            f"{nama_gabungan} terdeteksi, dekat swing ({keterangan_swing}) → skor 2"
        )
    else:
        score  = 1
        detail = (
            f"{nama_gabungan} terdeteksi, tapi jauh dari swing ({keterangan_swing}) → skor 1"
        )

    return {
        "score"            : score,
        "max"              : 2,
        "label"            : "Candlestick Pattern",
        "detail"           : detail,
        "pattern_detected" : nama_gabungan,
    }
