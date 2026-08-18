"""
engine/strategies/trend_following_v2.py
=======================================
Modul evaluasi strategi Trend Following v2 untuk XAUUSD M5.

TUJUAN:
    Mengevaluasi peluang entry trend following saat regime M15 terdeteksi sebagai TRENDING.
    Strategi ini menggabungkan sinyal EMA cross (momentum) dengan struktur harga
    (pullback ke swing terdekat dan penembusan struktur minor).

KEPUTUSAN REUSE EMA-trigger vs tidak-reuse H1-bias:
    - _check_ema_trigger_m5() DIREUSE LANGSUNG dari rule_engine.py karena ini adalah
      komponen momentum yang valid dan sudah teruji. Ini dikombinasikan (AND) dengan
      filter lokasi dan struktur (bukan satu-satunya sumber keputusan).
    - Fungsi pengecekan bias H1 dan pengambilan konteks H1 TIDAK DIREUSE, karena arah trend secara
      makro sudah ditentukan oleh regime M15 (TRENDING) yang diterima sebagai parameter,
      sehingga kita tidak lagi mengandalkan bias H1 secara mentah.

CAUSALITY:
    Zero look-ahead bias. Fungsi ini hanya membaca baris DataFrame M5 pada dan
    sebelum index evaluasi (df_m5.iloc[:idx_m5+1]).

CATATAN:
    - BELUM DIKALIBRASI: Nilai TREND_PULLBACK_PROXIMITY_ATR adalah nilai awal.
    - Lokasi/pullback proximity adalah pengecekan WAJIB, bukan opsional. Entry
      hanya valid jika harga saat ini dekat dengan area pullback struktural, bukan
      setelah harga sudah jauh melesat.
"""

import pandas as pd
from engine.rule_engine import _check_ema_trigger_m5
from engine.risk_manager import find_nearest_swing, SWING_LOOKBACK, SWING_WING

# Toleransi jarak close saat ini ke swing level pullback, dalam satuan ATR.
# BELUM DIKALIBRASI
TREND_PULLBACK_PROXIMITY_ATR = 1.0


def evaluate_trend_following(
    df_m5: pd.DataFrame,
    idx_m5: int,
    arah: str,                              # "BULLISH" | "BEARISH"
    swing_lookback: int = SWING_LOOKBACK,
    swing_wing: int = SWING_WING,
    pullback_proximity_atr: float = TREND_PULLBACK_PROXIMITY_ATR,
) -> dict:
    """
    Evaluasi peluang entry trend following di timeframe M5.
    """

    def _kosong(keterangan: str, d_ema=False, d_pb=False, d_pb_lvl=None, d_pb_dist=None, d_sb=False) -> dict:
        return {
            "terpenuhi"             : False,
            "arah"                  : "NETRAL",
            "ema_trigger_ok"        : d_ema,
            "pullback_ok"           : d_pb,
            "pullback_swing_level"  : d_pb_lvl,
            "pullback_distance"     : d_pb_dist,
            "structure_break_ok"    : d_sb,
            "invalidation_level_sl" : None,
            "keterangan"            : keterangan,
        }

    # ── Langkah 1 — Tentukan arah kandidat ──────────────────────────────────
    if arah == "BULLISH":
        arah_kandidat = "BUY"
    elif arah == "BEARISH":
        arah_kandidat = "SELL"
    else:
        return _kosong(f"arah tidak dikenal: {arah}")

    # Normalisasi idx negatif
    n = len(df_m5)
    if idx_m5 < 0:
        idx_m5 = n + idx_m5
        
    if idx_m5 < 0 or idx_m5 >= n:
        return _kosong(f"idx_m5 {idx_m5} di luar batas [0, {n-1}]")

    row_m5 = df_m5.iloc[idx_m5]

    # ── Langkah 2 — Konfirmasi EMA trigger (reuse) ──────────────────────────
    signals = {
        "trend"       : row_m5.get("trend"),
        "ema_9"       : row_m5.get("ema_9"),
        "ema_21"      : row_m5.get("ema_21"),
        "ema_gap_pct" : row_m5.get("ema_gap_pct"),
    }
    c_m5 = _check_ema_trigger_m5(signals)
    ema_trigger_ok = bool(c_m5["terpenuhi"] and c_m5["arah"] == arah_kandidat)
    ema_ket = "OK" if ema_trigger_ok else f"GAGAL ({c_m5['keterangan']})"

    # ── Langkah 3 — Konfirmasi pullback struktural (reuse) ──────────────────
    swing_level = find_nearest_swing(
        df_m5.iloc[:idx_m5+1], arah=arah_kandidat, lookback=swing_lookback, wing=swing_wing
    )
    
    pullback_ok = False
    pullback_distance = None
    pb_ket = "GAGAL (swing tidak ditemukan)"
    
    if swing_level is not None:
        close_now = float(row_m5["close"])
        atr_now   = float(row_m5["atr_14"])
        
        # Pengecekan lokasi (pullback proximity)
        pullback_distance = abs(close_now - swing_level)
        limit_dist = pullback_proximity_atr * atr_now
        
        if pullback_distance <= limit_dist:
            pullback_ok = True
            pb_ket = f"OK (dist={pullback_distance:.4f} <= limit={limit_dist:.4f})"
        else:
            pullback_ok = False
            pb_ket = f"GAGAL (lokasi terlalu jauh/sudah impulsif, dist={pullback_distance:.4f} > limit={limit_dist:.4f})"

    # ── Langkah 4 — Konfirmasi break of minor structure ─────────────────────
    structure_break_ok = False
    sb_ket = ""
    
    if idx_m5 < 2:
        sb_ket = "GAGAL (data tidak cukup untuk cek struktur minor, butuh 2 candle sblmnya)"
    else:
        close_now = float(row_m5["close"])
        high_1 = float(df_m5["high"].iloc[idx_m5-1])
        high_2 = float(df_m5["high"].iloc[idx_m5-2])
        low_1  = float(df_m5["low"].iloc[idx_m5-1])
        low_2  = float(df_m5["low"].iloc[idx_m5-2])
        
        if arah_kandidat == "BUY":
            max_high = max(high_1, high_2)
            structure_break_ok = bool(close_now > max_high)
            sb_ket = f"OK (close {close_now:.4f} > max_high {max_high:.4f})" if structure_break_ok else f"GAGAL (close {close_now:.4f} <= max_high {max_high:.4f})"
        else:
            min_low = min(low_1, low_2)
            structure_break_ok = bool(close_now < min_low)
            sb_ket = f"OK (close {close_now:.4f} < min_low {min_low:.4f})" if structure_break_ok else f"GAGAL (close {close_now:.4f} >= min_low {min_low:.4f})"

    # ── Langkah 5 — Gabungkan (AND semua) ───────────────────────────────────
    terpenuhi = ema_trigger_ok and pullback_ok and structure_break_ok
    invalidation_level_sl = swing_level if terpenuhi else None
    
    if terpenuhi:
        keterangan = (
            f"Trend Following v2 {arah_kandidat} TERPENUHI. "
            f"EMA: {ema_ket}, Pullback: {pb_ket}, Structure: {sb_ket}"
        )
    else:
        keterangan = (
            f"Trend Following v2 {arah_kandidat} GAGAL. "
            f"EMA: {ema_ket}, Pullback: {pb_ket}, Structure: {sb_ket}"
        )

    return {
        "terpenuhi"             : terpenuhi,
        "arah"                  : arah_kandidat if terpenuhi else "NETRAL",
        "ema_trigger_ok"        : ema_trigger_ok,
        "pullback_ok"           : pullback_ok,
        "pullback_swing_level"  : swing_level,
        "pullback_distance"     : pullback_distance,
        "structure_break_ok"    : structure_break_ok,
        "invalidation_level_sl" : invalidation_level_sl,
        "keterangan"            : keterangan,
    }
