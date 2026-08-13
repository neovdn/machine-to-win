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

SPREAD-AWARE ENTRY (versi terbaru):
    Jika caller memberikan tick_info (dict berisi 'ask' dan 'bid'), harga entry
    yang dipakai bukan lagi close candle tapi harga eksekusi nyata:
        BUY  → entry = tick_info['ask']  (kita membeli di harga ask)
        SELL → entry = tick_info['bid']  (kita menjual di harga bid)
    Spread = ask - bid, dicatat di output.
    rrr_after_spread = RRR yang sudah memperhitungkan cost spread di kedua sisi.
    Jika tick_info = None, fallback ke parameter 'entry' (biasanya close) —
    output backward-compatible.

LOGIKA MURNI — TIDAK ADA AI / MACHINE LEARNING.
"""

import pandas as pd
import numpy as np

# Import fungsi S&D (Fase 11) — lazy path via function-level import untuk
# menghindari circular import dan menjaga modul ini tetap berdiri sendiri.
# find_nearest_sd_zone diimport di dalam calculate_sl_tp saat sl_source="SD_ZONE".


# =============================================================================
# KONSTANTA KONFIGURASI DEFAULT (Profile: scalp_m5)
# =============================================================================

ATR_PERIOD          = 14    # Periode ATR (harus sama dengan yang di indicators.py)
ATR_MULTIPLIER      = 0.9   # Pengali ATR default untuk scalp_m5
RRR_MIN_DEFAULT     = 1.3   # TP = entry ± (jarak_SL × RRR minimum)
SWING_LOOKBACK      = 15    # Berapa candle ke belakang untuk cari swing (~1 jam 15 min untuk M5)
SWING_WING          = 3     # Window konfirmasi wing (±15 min)
SWING_BUFFER        = 0.50  # Buffer dollar di luar swing (agar SL tidak persis di level)
SWING_CLAMP_MIN_ATR = 0.7   # Batas minimum jarak SL terhadap ATR saat clamp
SWING_CLAMP_MAX_ATR = 2.0   # Batas maksimum jarak SL terhadap ATR saat clamp


# =============================================================================
# PROFIL RISIKO (RISK PROFILES)
# =============================================================================

RISK_PROFILES = {
    "scalp_m5": {
        "atr_multiplier"     : 0.9,
        "swing_lookback"     : 15,
        "swing_wing"         : 3,
        "rrr_min"            : 1.3,
        "swing_clamp_min_atr": 0.7,
        "swing_clamp_max_atr": 2.0,
        "swing_buffer"       : 0.50,
    },
    # Profil lain dapat ditambahkan di sini jika dibutuhkan (misal "swing_m15")
}


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

    Parameter:
        df       : DataFrame yang sudah punya kolom 'low' dan 'high'
        arah     : "BUY"  → cari swing LOW
                   "SELL" → cari swing HIGH
        lookback : Jumlah candle ke belakang yang ditelusuri
        wing     : Jumlah candle di kiri & kanan untuk konfirmasi swing

    Return:
        float → harga swing yang ditemukan (nilai low atau high)
        None  → tidak ditemukan swing dalam lookback window
    """
    min_data = lookback + wing * 2 + 1

    if len(df) < min_data:
        return None

    data = df.iloc[-(lookback + wing * 2):-1].copy()
    n    = len(data)
    col  = "low" if arah == "BUY" else "high"

    for i in range(n - 1 - wing, wing - 1, -1):
        window_slice = data[col].iloc[i - wing : i + wing + 1]
        val          = data[col].iloc[i]

        if arah == "BUY":
            if val == window_slice.min():
                return float(val)
        else:  # SELL
            if val == window_slice.max():
                return float(val)

    return None


# =============================================================================
# FUNGSI 2: KALKULASI SL DAN TP (FUNGSI UTAMA DEGAN ATR CLAMP)
# =============================================================================

def calculate_sl_tp(
    df                  : pd.DataFrame,
    entry               : float,
    arah                : str,
    profile             : str = "scalp_m5",
    rrr_min             : float | None = None,
    atr_multiplier      : float | None = None,
    swing_lookback      : int | None = None,
    swing_wing          : int | None = None,
    swing_clamp_min_atr : float | None = None,
    swing_clamp_max_atr : float | None = None,
    swing_buffer        : float | None = None,
    tick_info           : dict | None = None,
    sl_source           : str = "SWING",
    sd_impulsive_ratio  : float | None = None,
                                        # Fase 11.3: Threshold untuk zone origin.
                                        # None = gunakan default dari module supply_demand.
                                        # Sumber referensi SL.
                                        # "SWING" (default) → perilaku lama, identik 100%
                                        #   dengan sebelum Fase 11.
                                        # "SD_ZONE" → gunakan Supply & Demand zone sebagai
                                        #   referensi SL (Fase 11). Default HARUS "SWING"
                                        #   sampai divalidasi dan disetujui eksplisit.
) -> dict:
    """
    Hitung SL dan TP menggunakan pendekatan ATR-Clamped Swing atau S&D Zone.

    ALUR KALKULASI:
        1. Muat parameter default dari RISK_PROFILES[profile]. Parameter yang
           dilewatkan secara eksplisit akan me-override nilai default profile.
        2. Tentukan harga entry eksekusi (ask untuk BUY, bid untuk SELL jika tick_info ada).
        3a. Jika sl_source="SWING" (default):
              Cari level swing raw (swing low untuk BUY, swing high untuk SELL).
              Jika ditemukan:
                - Jarak raw = entry - (swing_low - buffer) [BUY]
                            atau (swing_high + buffer) - entry [SELL]
                - Clamp jarak raw ke [min_dist, max_dist] di mana:
                    min_dist = swing_clamp_min_atr × ATR
                    max_dist = swing_clamp_max_atr × ATR
                - SL final = entry ± dist_clamped
              Jika tidak ditemukan: fallback ke ATR.
        3b. Jika sl_source="SD_ZONE" (Fase 11):
              Panggil find_nearest_sd_zone() dari engine.supply_demand.
              Jika zona ditemukan:
                - Level SL raw = zona.level (sudah include buffer)
                - Hitung jarak raw dari entry ke level SL raw
                - Clamp dengan range ATR yang SAMA (SWING_CLAMP_MIN_ATR / MAX_ATR)
                - SL final = entry ± dist_clamped
              Jika tidak ditemukan: fallback ke ATR (identik dengan swing fallback).
              sl_method = "SD_ZONE" di output (bukan "SWING") agar audit trail jelas.
        4. TP = entry ± (jarak_sl × rrr_min)

    Parameter baru (Fase 11):
        sl_source : str — "SWING" (default) atau "SD_ZONE".
                    Default harus "SWING" sampai validasi Fase 11.3 selesai
                    dan disetujui eksplisit.

    Return:
        dict audit lengkap berisi SL, TP, RRR, audit clamp, dan status metode SL.
        Field sl_method: "SWING" / "ATR" (perilaku lama) atau "SD_ZONE" / "ATR" (Fase 11).
    """
    # ── Resolve parameter dari Profile & Overrides ───────────────────────────
    p = RISK_PROFILES.get(profile, RISK_PROFILES["scalp_m5"])

    rrr_min             = rrr_min             if rrr_min             is not None else p.get("rrr_min", RRR_MIN_DEFAULT)
    atr_multiplier      = atr_multiplier      if atr_multiplier      is not None else p.get("atr_multiplier", ATR_MULTIPLIER)
    swing_lookback      = swing_lookback      if swing_lookback      is not None else p.get("swing_lookback", SWING_LOOKBACK)
    swing_wing          = swing_wing          if swing_wing          is not None else p.get("swing_wing", SWING_WING)
    swing_clamp_min_atr = swing_clamp_min_atr if swing_clamp_min_atr is not None else p.get("swing_clamp_min_atr", SWING_CLAMP_MIN_ATR)
    swing_clamp_max_atr = swing_clamp_max_atr if swing_clamp_max_atr is not None else p.get("swing_clamp_max_atr", SWING_CLAMP_MAX_ATR)
    swing_buffer        = swing_buffer        if swing_buffer        is not None else p.get("swing_buffer", SWING_BUFFER)

    _validate_inputs(df, entry, arah, rrr_min)

    atr_col = f"atr_{ATR_PERIOD}"

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 1: Tentukan harga entry eksekusi
    # ─────────────────────────────────────────────────────────────────────────
    spread     = None
    entry_type = "CLOSE"

    if tick_info is not None:
        ask = tick_info.get("ask")
        bid = tick_info.get("bid")

        if ask is not None and bid is not None and ask > 0 and bid > 0:
            spread = round(ask - bid, 5)
            if arah == "BUY":
                entry = float(ask)
                entry_type = "ASK"
            else:
                entry = float(bid)
                entry_type = "BID"

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2: Ambil ATR & Hitung level referensi ATR
    # ─────────────────────────────────────────────────────────────────────────
    atr_value = float(df[atr_col].iloc[-1])

    if arah == "BUY":
        sl_atr = entry - (atr_multiplier * atr_value)
    else:
        sl_atr = entry + (atr_multiplier * atr_value)

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 3: Deteksi Level SL & Logika ATR Clamp
    # Cabang sl_source="SWING" (default) atau sl_source="SD_ZONE" (Fase 11).
    # ─────────────────────────────────────────────────────────────────────────
    sl_swing          = None
    sl_swing_clamped  = False
    clamp_reason      = None
    swing_raw         = None   # level raw swing (diisi saat SWING path)
    sd_zone_info      = None   # dict zona S&D (diisi saat SD_ZONE path)

    min_dist = swing_clamp_min_atr * atr_value
    max_dist = swing_clamp_max_atr * atr_value

    if sl_source == "SD_ZONE":
        # ── PATH S&D ZONE (Fase 11) ──────────────────────────────────────────
        # Import di dalam fungsi untuk menghindari circular import.
        from engine.supply_demand import find_nearest_sd_zone, DEFAULT_IMPULSIVE_RATIO  # noqa: PLC0415

        ratio = sd_impulsive_ratio if sd_impulsive_ratio is not None else DEFAULT_IMPULSIVE_RATIO

        sd_zone_info = find_nearest_sd_zone(
            df       = df,
            arah     = arah,
            idx      = -1,   # selalu pakai candle terakhir df (df sudah di-slice oleh caller)
            lookback = 50,   # SD_LOOKBACK default — BELUM dikalibrasi (Fase 11)
            impulsive_body_atr_ratio = ratio,
            buffer   = swing_buffer,
        )

        if sd_zone_info is not None:
            # level sudah include buffer (dihitung di find_nearest_sd_zone)
            sl_sd_level = sd_zone_info["level"]

            if arah == "BUY":
                dist_raw = entry - sl_sd_level
            else:  # SELL
                dist_raw = sl_sd_level - entry

            # ATR clamp IDENTIK dengan SWING path — tidak buat clamp range baru.
            if dist_raw < min_dist:
                dist_final       = min_dist
                sl_swing_clamped = True
                clamp_reason     = "MIN_CAP"
            elif dist_raw > max_dist:
                dist_final       = max_dist
                sl_swing_clamped = True
                clamp_reason     = "MAX_CAP"
            else:
                dist_final       = dist_raw
                sl_swing_clamped = False
                clamp_reason     = None

            sl_method = "SD_ZONE"  # audit trail: sumber SL adalah S&D zone
            if arah == "BUY":
                sl_final = entry - dist_final
            else:
                sl_final = entry + dist_final

            # sl_swing diisi dengan level zona (analogus dengan swing_raw - buffer)
            sl_swing = sl_sd_level
        else:
            # Fallback ke ATR jika tidak ada zona S&D valid ditemukan
            dist_final       = atr_multiplier * atr_value
            sl_final         = sl_atr
            sl_method        = "ATR"
            sl_swing_clamped = False
            clamp_reason     = None

    else:
        # ── PATH SWING (default, perilaku IDENTIK sebelum Fase 11) ───────────
        swing_raw = find_nearest_swing(df, arah, lookback=swing_lookback, wing=swing_wing)

        if swing_raw is not None:
            if arah == "BUY":
                sl_swing = swing_raw - swing_buffer
                dist_raw = entry - sl_swing
            else:  # SELL
                sl_swing = swing_raw + swing_buffer
                dist_raw = sl_swing - entry

            # Clamp distance ke range [min_dist, max_dist]
            if dist_raw < min_dist:
                dist_final       = min_dist
                sl_swing_clamped = True
                clamp_reason     = "MIN_CAP"
            elif dist_raw > max_dist:
                dist_final       = max_dist
                sl_swing_clamped = True
                clamp_reason     = "MAX_CAP"
            else:
                dist_final       = dist_raw
                sl_swing_clamped = False
                clamp_reason     = None

            sl_method = "SWING"
            if arah == "BUY":
                sl_final = entry - dist_final
            else:
                sl_final = entry + dist_final
        else:
            # Fallback ke ATR jika swing tidak ditemukan
            dist_final       = atr_multiplier * atr_value
            sl_final         = sl_atr
            sl_method        = "ATR"
            sl_swing_clamped = False
            clamp_reason     = None

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 4: Hitung TP & RRR
    # ─────────────────────────────────────────────────────────────────────────
    jarak_sl = abs(entry - sl_final)

    if arah == "BUY":
        tp = entry + (jarak_sl * rrr_min)
    else:
        tp = entry - (jarak_sl * rrr_min)

    jarak_tp = abs(tp - entry)
    rrr      = jarak_tp / jarak_sl if jarak_sl > 0 else 0.0

    rrr_after_spread = None
    if spread is not None and jarak_sl > 0:
        effective_profit = max(0.0, jarak_tp - spread)
        effective_risk   = jarak_sl + spread
        rrr_after_spread = round(effective_profit / effective_risk, 2) if effective_risk > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Susun pesan audit
    # ─────────────────────────────────────────────────────────────────────────
    if sl_method == "SWING" and swing_raw is not None:
        clamp_str = f" [CLAMPED: {clamp_reason} ({min_dist:.2f} - {max_dist:.2f} USD)]" if sl_swing_clamped else ""
        pesan = (
            f"SL dari swing {'low' if arah == 'BUY' else 'high'} @ {swing_raw:.2f} "
            f"{'−' if arah == 'BUY' else '+'} buffer {swing_buffer:.2f} = {sl_final:.2f}{clamp_str} "
            f"(ATR {atr_value:.2f}, clamp range: [{min_dist:.2f}, {max_dist:.2f}])"
        )
    elif sl_method == "SD_ZONE" and sd_zone_info is not None:
        clamp_str = f" [CLAMPED: {clamp_reason} ({min_dist:.2f} - {max_dist:.2f} USD)]" if sl_swing_clamped else ""
        freshness = sd_zone_info.get("freshness", "?")
        pesan = (
            f"SL dari S&D zone [{sd_zone_info['zone_low']:.2f}, {sd_zone_info['zone_high']:.2f}] "
            f"({freshness}) origin_idx={sd_zone_info['origin_idx']}, "
            f"level={sd_zone_info['level']:.2f} = {sl_final:.2f}{clamp_str} "
            f"(ATR {atr_value:.2f}, clamp range: [{min_dist:.2f}, {max_dist:.2f}])"
        )
    else:
        src = "swing" if sl_source == "SWING" else "S&D zone"
        pesan = (
            f"SL dari ATR: {entry:.2f} "
            f"{'−' if arah == 'BUY' else '+'} ({atr_multiplier}×{atr_value:.2f}) = {sl_final:.2f} "
            f"(tidak ada {src} ditemukan dalam lookback)"
        )

    # sl_swing_raw: untuk kompatibilitas backward dengan backtester (inject ke signals)
    # Pada path SD_ZONE, kita isi dengan zone_low (BUY) atau zone_high (SELL)
    # agar caller yang membaca sl_swing_raw untuk quality scoring tetap bisa berjalan.
    if sl_method == "SWING":
        sl_swing_raw_out = round(swing_raw, 2) if swing_raw is not None else None
    elif sl_method == "SD_ZONE" and sd_zone_info is not None:
        # Beri referensi level struktural (tanpa buffer) untuk quality scoring.
        if arah == "BUY":
            sl_swing_raw_out = round(sd_zone_info["zone_low"], 2)
        else:
            sl_swing_raw_out = round(sd_zone_info["zone_high"], 2)
    else:
        sl_swing_raw_out = None

    return {
        "valid"            : True,
        "entry"            : round(entry,    2),
        "entry_type"       : entry_type,
        "sl"               : round(sl_final, 2),
        "tp"               : round(tp,       2),
        "rrr"              : round(rrr,      2),
        "jarak_sl"         : round(jarak_sl, 2),
        "jarak_tp"         : round(jarak_tp, 2),
        "sl_method"        : sl_method,
        "sl_swing_clamped" : sl_swing_clamped,
        "clamp_reason"     : clamp_reason,
        "atr_value"        : round(atr_value, 2),
        "sl_atr_level"     : round(sl_atr,    2),
        "sl_swing_raw"     : sl_swing_raw_out,
        "sl_swing_level"   : round(sl_swing,  2) if sl_swing  is not None else None,
        "spread"           : round(spread, 5) if spread is not None else None,
        "rrr_after_spread" : rrr_after_spread,
        "pesan"            : pesan,
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
