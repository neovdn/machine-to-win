"""
engine/risk_manager.py
======================
Modul kalkulasi Stop Loss, Take Profit, dan Risk-Reward Ratio.

PENDEKATAN: Hybrid ATR + Swing (default)
    1. ATR → menghitung batas MINIMUM jarak SL (melindungi dari noise candle)
    2. Swing High/Low → mencari level struktur market terdekat (lebih natural)
    3. Pilih yang lebih jauh dari entry (lebih konservatif):
         - BUY : SL = min(sl_atr, sl_swing)  ← lebih RENDAH = lebih jauh di bawah entry
         - SELL: SL = max(sl_atr, sl_swing)  ← lebih TINGGI = lebih jauh di atas entry
    4. TP = entry ± (jarak_SL × rrr_min)

KENAPA LOGIKA min/max PENTING:
    Untuk BUY, SL harus di BAWAH entry. "Lebih jauh" = lebih rendah = min().
    Kalau pakai max() untuk BUY, kita malah pilih SL yang LEBIH DEKAT ke entry
    (lebih tinggi), yang artinya SL lebih mudah kena hit → salah!

    Sebaliknya untuk SELL: SL di ATAS entry. "Lebih jauh" = lebih tinggi = max().

FALLBACK:
    Kalau swing tidak ditemukan dalam lookback window → pakai ATR saja (tidak error).

LOGIKA MURNI — TIDAK ADA AI / MACHINE LEARNING.
"""

import pandas as pd
import numpy as np


# =============================================================================
# KONSTANTA KONFIGURASI DEFAULT
# =============================================================================

ATR_PERIOD        = 14    # Periode ATR (harus sama dengan yang di indicators.py)
ATR_MULTIPLIER    = 1.5   # SL minimum = 1.5 × ATR dari entry
RRR_MIN_DEFAULT   = 2.0   # TP = entry ± (jarak_SL × RRR minimum)
SWING_LOOKBACK    = 50    # Berapa candle ke belakang untuk cari swing
SWING_WING        = 5     # Berapa candle kiri & kanan untuk konfirmasi swing
                          # (diubah dari 3 → 5 setelah analisis data nyata)
                          #
                          # TRADE-OFF WING SIZE DI XAUUSD M5:
                          #   wing=3 : window 35 menit. Banyak swing ditemukan tapi
                          #            sebagian adalah noise — lembah/puncak kecil yang
                          #            tidak terlihat jelas di chart manual.
                          #   wing=5 : window 55 menit (~1 jam). Menyaring swing kecil
                          #            yang terlalu mirip satu sama lain, tapi masih
                          #            menemukan swing yang benar-benar terlihat di chart.
                          #            SL yang dihasilkan lebih bermakna secara visual.
                          #   wing=8 : window 85 menit. Terlalu ketat — bisa melewatkan
                          #            swing yang jelas terlihat (misal 30 menit lalu)
                          #            dan lebih sering fallback ke ATR.
                          #
                          # KESIMPULAN: wing=5 adalah sweet spot untuk M5 XAUUSD.
SWING_BUFFER      = 0.50  # Buffer dollar di luar swing (agar SL tidak persis di level)


# =============================================================================
# FUNGSI 1: DETEKSI SWING HIGH / SWING LOW
# =============================================================================

def find_nearest_swing(
    df         : pd.DataFrame,
    arah       : str,
    lookback   : int = SWING_LOOKBACK,
    wing       : int = SWING_WING,
) -> float | None:
    """
    Mencari swing high (untuk SELL) atau swing low (untuk BUY) terdekat.

    APA ITU SWING HIGH/LOW:
        Swing LOW  = candle yang nilai LOW-nya lebih rendah dari `wing` candle
                     di kiri DAN kanan. Ini titik "lembah" lokal pada chart.
        Swing HIGH = candle yang nilai HIGH-nya lebih tinggi dari `wing` candle
                     di kiri DAN kanan. Ini titik "puncak" lokal pada chart.

        Contoh swing low dengan wing=3:
              High ─ ─ ─ ─ ─ ─ ─
                         │ │
            ─ ─ ─ ─ ─ ─ │ └── swing low: candle ini low-nya lebih rendah dari
                         │     3 candle di kiri dan 3 candle di kanan
              Low  ─ ─ ─ ┘

    CARA KERJA:
        1. Ambil `lookback` candle terakhir dari DataFrame (sudah dijamin closed semua
           karena get_candles() menggunakan start_pos=1).
        2. Namun candle PALING AKHIR di window tetap dikecualikan karena alasan teknikal
           swing: candle paling akhir tidak punya candle di sebelah kanannya, sehingga
           tidak bisa memenuhi syarat "lebih rendah/tinggi dari wing candle di kanan".
           Ini bukan karena belum closed, tapi karena keterbatasan definisi swing.
        3. Cari candle yang low/high-nya adalah minimum/maksimum lokal
        4. Kembalikan swing terdekat (yang paling baru) — iterate dari akhir

    Parameter:
        df       : DataFrame yang sudah punya kolom 'low' dan 'high'
                   (semua candle sudah closed, jaminan dari get_candles())
        arah     : "BUY"  → cari swing LOW (untuk referensi SL di bawah entry)
                   "SELL" → cari swing HIGH (untuk referensi SL di atas entry)
        lookback : Jumlah candle ke belakang yang ditelusuri
        wing     : Jumlah candle di kiri & kanan untuk konfirmasi swing

    Return:
        float → harga swing yang ditemukan (nilai low atau high)
        None  → tidak ditemukan swing dalam lookback window
    """
    # Kita butuh minimal (lookback + 2*wing) candle agar ada cukup data
    min_data = lookback + wing * 2 + 1

    if len(df) < min_data:
        # Data tidak cukup — kembalikan None agar caller pakai fallback ATR
        return None

    # Ambil window data:
    # - Kecualikan candle paling akhir ([-1]) karena alasan teknikal swing:
    #   Sebuah candle bisa jadi swing HANYA jika ada `wing` candle di kiri
    #   DAN kanan yang lebih tinggi/rendah. Candle terakhir di DataFrame
    #   tidak punya candle di sebelah kanannya sama sekali, sehingga
    #   tidak bisa memenuhi definisi swing.
    #   (Bukan karena belum closed — data sudah dijamin closed semua dari
    #   get_candles() dengan start_pos=1.)
    # - Ambil lebih banyak dari lookback agar candle di tepi window tetap
    #   punya cukup tetangga untuk validasi swing
    data = df.iloc[-(lookback + wing * 2):-1].copy()
    n    = len(data)

    # Pilih kolom yang relevan berdasarkan arah
    col = "low" if arah == "BUY" else "high"

    # ─────────────────────────────────────────────────────────────────────────
    # Iterasi dari candle TERBARU ke TERLAMA (mencari swing terdekat dulu)
    # Valid range: wing ≤ i ≤ (n - 1 - wing)
    # → butuh wing candle di kiri (i-wing:i) dan kanan (i+1:i+wing+1)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(n - 1 - wing, wing - 1, -1):
        # Ambil nilai candle ke-i dan tetangganya
        window_slice = data[col].iloc[i - wing : i + wing + 1]
        val          = data[col].iloc[i]

        if arah == "BUY":
            # Swing LOW: nilai ini adalah minimum lokal dalam window
            if val == window_slice.min():
                return float(val)

        else:  # SELL
            # Swing HIGH: nilai ini adalah maksimum lokal dalam window
            if val == window_slice.max():
                return float(val)

    # Tidak ada swing ditemukan dalam lookback window
    return None


# =============================================================================
# FUNGSI 2: KALKULASI SL DAN TP (FUNGSI UTAMA)
# =============================================================================

def calculate_sl_tp(
    df             : pd.DataFrame,
    entry          : float,
    arah           : str,
    rrr_min        : float = RRR_MIN_DEFAULT,
    atr_multiplier : float = ATR_MULTIPLIER,
    swing_lookback : int   = SWING_LOOKBACK,
    swing_buffer   : float = SWING_BUFFER,
) -> dict:
    """
    Hitung SL dan TP menggunakan pendekatan Hybrid ATR + Swing.

    ALUR KALKULASI:
        1. Ambil nilai ATR terbaru dari DataFrame (sudah dihitung di indicators.py)
        2. Hitung SL versi ATR:
             BUY : sl_atr = entry − (atr_multiplier × atr)
             SELL: sl_atr = entry + (atr_multiplier × atr)
        3. Cari swing terdekat (swing low untuk BUY, swing high untuk SELL)
        4. Hitung SL versi Swing (jika swing ditemukan):
             BUY : sl_swing = swing_level − swing_buffer
             SELL: sl_swing = swing_level + swing_buffer
        5. Pilih SL final yang LEBIH JAUH dari entry (lebih konservatif):
             BUY : sl_final = min(sl_atr, sl_swing)  ← lebih RENDAH = lebih jauh
             SELL: sl_final = max(sl_atr, sl_swing)  ← lebih TINGGI = lebih jauh
        6. Hitung TP: TP = entry ± (jarak_SL × rrr_min)

    Parameter:
        df             : DataFrame yang sudah melewati run_all_indicators()
                         — harus punya kolom 'atr_14', 'high', 'low'
        entry          : Harga entry (biasanya close candle terbaru)
        arah           : "BUY" atau "SELL"
        rrr_min        : RRR minimum. TP = entry ± (jarak_SL × rrr_min)
        atr_multiplier : Pengali ATR untuk SL minimum
        swing_lookback : Berapa candle ditelusuri untuk cari swing
        swing_buffer   : Buffer dollar di luar swing level

    Return:
        dict berisi:
            "valid"         : bool   — False jika data tidak mencukupi
            "entry"         : float  — harga entry
            "sl"            : float  — harga Stop Loss final
            "tp"            : float  — harga Take Profit
            "rrr"           : float  — RRR aktual yang dihasilkan
            "jarak_sl"      : float  — jarak absolut entry ke SL (dalam dollar)
            "jarak_tp"      : float  — jarak absolut entry ke TP (dalam dollar)
            "sl_method"     : str    — "SWING" atau "ATR" (dari mana SL berasal)
            "atr_value"     : float  — nilai ATR aktual
            "sl_atr_level"  : float  — SL versi ATR (selalu dihitung, untuk referensi)
            "sl_swing_raw"  : float|None — harga swing yang ditemukan (sebelum buffer)
            "sl_swing_level": float|None — SL versi swing (setelah buffer)
            "pesan"         : str    — penjelasan singkat bagaimana SL dihitung
    """
    _validate_inputs(df, entry, arah, rrr_min)

    atr_col = f"atr_{ATR_PERIOD}"

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 1: Ambil nilai ATR terbaru
    # ─────────────────────────────────────────────────────────────────────────
    # Gunakan iloc[-1] = nilai dari candle paling baru
    atr_value = float(df[atr_col].iloc[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2: Hitung SL versi ATR (selalu dihitung sebagai fallback)
    # ─────────────────────────────────────────────────────────────────────────
    if arah == "BUY":
        sl_atr = entry - (atr_multiplier * atr_value)
    else:  # SELL
        sl_atr = entry + (atr_multiplier * atr_value)

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 3: Cari swing terdekat dan hitung SL versi Swing
    # ─────────────────────────────────────────────────────────────────────────
    swing_raw   = find_nearest_swing(df, arah, lookback=swing_lookback)
    sl_swing    = None

    if swing_raw is not None:
        # Tambah buffer di luar swing level
        # BUY : SL sedikit di BAWAH swing low → swing_raw - buffer
        # SELL: SL sedikit di ATAS swing high → swing_raw + buffer
        if arah == "BUY":
            sl_swing = swing_raw - swing_buffer
        else:  # SELL
            sl_swing = swing_raw + swing_buffer

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 4: Pilih SL final — yang LEBIH JAUH dari entry
    # ─────────────────────────────────────────────────────────────────────────
    #
    # ⚠️  LOGIKA KRITIS — baca ini dulu sebelum mengubah kode:
    #
    #   Untuk BUY:  SL ada di BAWAH entry (nilai lebih kecil dari entry)
    #               "Lebih jauh" = lebih rendah = nilai LEBIH KECIL → pakai min()
    #               min(sl_atr, sl_swing) memilih level yang LEBIH RENDAH
    #
    #   Untuk SELL: SL ada di ATAS entry (nilai lebih besar dari entry)
    #               "Lebih jauh" = lebih tinggi = nilai LEBIH BESAR → pakai max()
    #               max(sl_atr, sl_swing) memilih level yang LEBIH TINGGI
    #
    #   Menggunakan max() untuk BUY adalah SALAH — itu justru memilih SL
    #   yang lebih dekat ke entry (lebih tinggi dari sl_atr), lebih mudah
    #   kena hit, dan tidak menghormati struktur market.

    if sl_swing is not None:
        if arah == "BUY":
            # Pilih yang lebih RENDAH (lebih jauh di bawah entry)
            sl_final   = min(sl_atr, sl_swing)
            sl_method  = "SWING" if sl_swing <= sl_atr else "ATR"
        else:  # SELL
            # Pilih yang lebih TINGGI (lebih jauh di atas entry)
            sl_final   = max(sl_atr, sl_swing)
            sl_method  = "SWING" if sl_swing >= sl_atr else "ATR"
    else:
        # Tidak ada swing — fallback ke ATR
        sl_final  = sl_atr
        sl_method = "ATR"

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 5: Hitung TP berdasarkan RRR minimum
    # ─────────────────────────────────────────────────────────────────────────
    jarak_sl = abs(entry - sl_final)   # selalu positif

    if arah == "BUY":
        tp = entry + (jarak_sl * rrr_min)
    else:  # SELL
        tp = entry - (jarak_sl * rrr_min)

    jarak_tp = abs(tp - entry)         # selalu positif
    rrr      = jarak_tp / jarak_sl if jarak_sl > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Susun pesan audit
    # ─────────────────────────────────────────────────────────────────────────
    if sl_method == "SWING" and swing_raw is not None:
        pesan = (
            f"SL dari swing {'low' if arah == 'BUY' else 'high'} @ {swing_raw:.2f} "
            f"{'−' if arah == 'BUY' else '+'} buffer {swing_buffer:.2f} = {sl_final:.2f} "
            f"(ATR level {sl_atr:.2f} lebih dekat ke entry, swing dipakai)"
        )
    else:
        alasan_fallback = "tidak ada swing ditemukan" if swing_raw is None else \
                          f"ATR lebih jauh dari swing ({sl_atr:.2f} vs {sl_swing:.2f})"
        pesan = (
            f"SL dari ATR: {entry:.2f} "
            f"{'−' if arah == 'BUY' else '+'} ({atr_multiplier}×{atr_value:.2f}) = {sl_final:.2f} "
            f"({alasan_fallback})"
        )

    return {
        "valid"          : True,
        "entry"          : round(entry,    2),
        "sl"             : round(sl_final, 2),
        "tp"             : round(tp,       2),
        "rrr"            : round(rrr,      2),
        "jarak_sl"       : round(jarak_sl, 2),
        "jarak_tp"       : round(jarak_tp, 2),
        "sl_method"      : sl_method,
        "atr_value"      : round(atr_value, 2),
        "sl_atr_level"   : round(sl_atr,    2),
        "sl_swing_raw"   : round(swing_raw, 2) if swing_raw is not None else None,
        "sl_swing_level" : round(sl_swing,  2) if sl_swing  is not None else None,
        "pesan"          : pesan,
    }


# =============================================================================
# HELPER INTERNAL — Validasi Input
# =============================================================================

def _validate_inputs(df: pd.DataFrame, entry: float, arah: str, rrr_min: float) -> None:
    """
    Validasi input sebelum kalkulasi SL/TP.

    Raise:
        TypeError  : jika df bukan DataFrame
        ValueError : jika kolom kurang, arah tidak valid, atau rrr_min tidak masuk akal
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Parameter 'df' harus pandas DataFrame, bukan {type(df).__name__}."
        )

    required = ["high", "low", "close", f"atr_{ATR_PERIOD}"]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut tidak ada di DataFrame: {missing}\n"
            f"Pastikan df sudah melewati run_all_indicators() yang memanggil "
            f"calculate_atr() juga."
        )

    if arah not in ("BUY", "SELL"):
        raise ValueError(
            f"Parameter 'arah' harus 'BUY' atau 'SELL', bukan '{arah}'."
        )

    if rrr_min <= 0:
        raise ValueError(
            f"Parameter 'rrr_min' harus lebih besar dari 0, dapat: {rrr_min}"
        )

    if df.empty:
        raise ValueError("DataFrame kosong — tidak ada data untuk dihitung.")
