"""
engine/supply_demand.py
=======================
Modul deteksi zona Supply & Demand (S&D) untuk XAUUSD M5 — Fase 11.

TUJUAN:
    Mengidentifikasi zona origin (level harga di mana supply/demand kuat
    terbentuk sebelum pergerakan impulsif) sebagai kandidat alternatif
    referensi SL di risk_manager.py.

    Modul ini adalah INFRASTRUKTUR RISK MANAGEMENT murni:
        - BUKAN sinyal entry independen.
        - BUKAN komponen scoring (calculate_setup_quality()).
        - TIDAK diintegrasikan ke evaluate_entry().
        - Satu-satunya titik integrasi: calculate_sl_tp() di risk_manager.py
          via parameter sl_source="SD_ZONE".

KONSEP INTI — S&D Zone vs Consolidation Zone:
    zone_detector.py (Fase 8) mendeteksi zona KONSOLIDASI/RANGE — area di
    mana harga bergerak sideways (dipakai untuk breakout). Modul ini mendeteksi
    zona ORIGIN — satu candle "basing" yang relatif diam sebelum pergerakan
    impulsif keluar darinya. Dua konsep berbeda meski sama-sama soal "level harga".

DEFINISI ZONA:
    1. Candle impulsif: body >= impulsive_body_atr_ratio * ATR.
       Default ratio 1.5 (lihat catatan kalibrasi di bawah).
    2. Candle origin (base candle): candle tepat SEBELUM candle impulsif.
       Origin harus "diam" — body-nya < 0.5 * impulsive_body_atr_ratio * ATR.
    3. Zona = range [low_origin, high_origin] dari candle origin.
    4. Demand zone: di bawah bullish impulsive move (SL BUY dari bawah zona).
       Supply zone: di atas bearish impulsive move (SL SELL dari atas zona).

FRESHNESS:
    FRESH  = harga belum pernah kembali menyentuh zona sejak terbentuk.
    TESTED = sudah pernah disentuh minimal satu kali. Masih valid dipakai.
    Keduanya bisa dipakai sebagai referensi SL. Label ini untuk analisis
    FRESH vs TESTED di validasi empiris (Fase 11.3, bagian 4) — bukan
    untuk filter saat ini.

INVALIDASI:
    Demand zone invalid jika ada candle k dengan close_k < low_origin
    (harga sudah break ke bawah zona, bukan sekadar retest).
    Supply zone invalid jika ada candle k dengan close_k > high_origin.
    Zona invalid tidak dikembalikan sebagai kandidat SL.

CATATAN PARAMETER (BELUM DIKALIBRASI):
    Semua nilai default di bawah adalah STARTING POINT yang masuk akal
    secara struktural, BUKAN hasil kalibrasi backtest. Pola ini sama seperti
    zone_detector.py (lookback=20, max_range_atr_ratio=2.5, min_duration=10)
    dan risk_manager.py (ATR_MULTIPLIER=0.9, SWING_LOOKBACK=15) sebelum
    walk-forward dilakukan.

    impulsive_body_atr_ratio = 1.5
        Alasan starting point: harus lebih ketat dari threshold body candle
        pattern biasa (0.3×ATR). "Impulsif" secara definisi harus jauh lebih
        kuat dari candle rata-rata. 1.5×ATR adalah batas moderat — tidak terlalu
        longgar (1.0×ATR) sehingga terlalu banyak candle lolos, tidak terlalu
        ketat (2.5×ATR) sehingga zona sangat jarang. Parameter sweep di validasi
        empiris (11.3 bagian 2) akan menguji 1.0/1.5/2.0/2.5.
        JANGAN ubah nilai ini tanpa approval eksplisit setelah validasi empiris.

    SD_LOOKBACK = 50
        Berapa candle ke belakang dicari zona valid. Lebih lebar dari
        SWING_LOOKBACK=15 karena origin candle bisa cukup jauh di belakang
        candle impulsif, dan candle impulsif bisa beberapa candle di belakang idx.
        BELUM dikalibrasi — dilaporkan balik jika terlalu lebar/sempit setelah
        validasi frekuensi zona ditemukan.

    ORIGIN_BODY_MAX_RATIO = 0.5
        Batas atas body candle origin sebagai fraksi dari
        impulsive_body_atr_ratio * ATR. Origin harus "diam" dibanding
        candle impulsif. 0.5 artinya body origin < separuh threshold impulsif.
        BELUM dikalibrasi.

    SD_BUFFER = 0.50
        Buffer dollar di luar zona (sama dengan SWING_BUFFER di risk_manager.py).
        Reuse nilai yang sudah ada agar tidak ada parameter baru yang perlu
        dikalibrasi terpisah. Ubah hanya jika ada alasan jelas dari validasi.

CAUSALITY:
    Semua fungsi hanya membaca data pada dan sebelum index idx.
    Candle setelah idx TIDAK PERNAH disentuh.
    Tidak ada state eksternal — recompute penuh setiap panggilan.
"""

import pandas as pd
import numpy as np


# =============================================================================
# PARAMETER DEFAULT (BELUM DIKALIBRASI — lihat docstring modul)
# =============================================================================

SD_LOOKBACK                 = 50     # Candle ke belakang untuk cari zona
DEFAULT_IMPULSIVE_RATIO     = 1.5    # body >= 1.5 * ATR → impulsif
ORIGIN_BODY_MAX_RATIO       = 0.5    # body origin < 0.5 * threshold impulsif
DEFAULT_MIN_CONSECUTIVE     = 1      # minimum candle impulsif berturut-turut
SD_BUFFER                   = 0.50   # Buffer dollar di luar zona (= SWING_BUFFER)


# =============================================================================
# FUNGSI 1: DETEKSI CANDLE IMPULSIF
# =============================================================================

def detect_impulsive_move(
    df                       : pd.DataFrame,
    idx                      : int,
    impulsive_body_atr_ratio : float = DEFAULT_IMPULSIVE_RATIO,
    min_consecutive_impulsive: int   = DEFAULT_MIN_CONSECUTIVE,
) -> dict:
    """
    Deteksi apakah candle pada idx merupakan candle impulsif.

    DEFINISI IMPULSIF:
        body_i = abs(close_i - open_i)
        body_i >= impulsive_body_atr_ratio * atr_14_i → impulsif

    MULTI-CANDLE (min_consecutive_impulsive > 1):
        Jika min_consecutive_impulsive=N, candle di idx HANYA dianggap impulsif
        jika N candle berturut-turut ending di idx semuanya impulsif.
        Default 1: satu candle impulsif sudah cukup.
        Opsi ini disediakan untuk keperluan parameter sweep masa depan.

    ARAH:
        close_i > open_i → BULLISH (kandidat demand zone di bawahnya)
        close_i < open_i → BEARISH (kandidat supply zone di atasnya)
        close_i == open_i → DOJI, BUKAN impulsif (body = 0)

    Parameter:
        df                        : DataFrame dengan kolom open, close, atr_14.
        idx                       : Index candle evaluasi (harus >= 0 setelah normalisasi).
                                    Hanya data pada dan sebelum idx yang dibaca (causal).
        impulsive_body_atr_ratio  : Threshold body/ATR. Default 1.5 (BELUM dikalibrasi).
        min_consecutive_impulsive : Minimum candle impulsif berturut-turut. Default 1.

    Return:
        dict dengan field:
            is_impulsive   : bool   — True jika memenuhi syarat impulsif
            arah           : str    — "BULLISH" / "BEARISH" / None
            body           : float  — abs(close - open) di idx
            atr_value      : float  — atr_14 di idx
            body_atr_ratio : float  — body / atr_value (ratio aktual)
            threshold      : float  — threshold minimum body untuk dianggap impulsif
            keterangan     : str    — penjelasan ringkas
    """
    # ── Normalisasi idx negatif ──────────────────────────────────────────────
    n = len(df)
    if idx < 0:
        idx = n + idx

    # ── Validasi kolom ──────────────────────────────────────────────────────
    required_cols = {"open", "close", "atr_14"}
    missing = required_cols - set(df.columns)
    if missing:
        return {
            "is_impulsive"   : False,
            "arah"           : None,
            "body"           : None,
            "atr_value"      : None,
            "body_atr_ratio" : None,
            "threshold"      : None,
            "keterangan"     : f"Kolom tidak lengkap: {sorted(missing)}",
        }

    # ── Validasi idx dalam range ─────────────────────────────────────────────
    if idx < 0 or idx >= n:
        return {
            "is_impulsive"   : False,
            "arah"           : None,
            "body"           : None,
            "atr_value"      : None,
            "body_atr_ratio" : None,
            "threshold"      : None,
            "keterangan"     : f"idx={idx} di luar range [0, {n-1}]",
        }

    # ── Pastikan ada cukup candle untuk min_consecutive ─────────────────────
    start_idx = idx - min_consecutive_impulsive + 1
    if start_idx < 0:
        return {
            "is_impulsive"   : False,
            "arah"           : None,
            "body"           : None,
            "atr_value"      : None,
            "body_atr_ratio" : None,
            "threshold"      : None,
            "keterangan"     : (
                f"Data tidak cukup untuk min_consecutive={min_consecutive_impulsive}: "
                f"idx={idx}, butuh start_idx >= 0"
            ),
        }

    # ── Evaluasi candle di idx ───────────────────────────────────────────────
    close_val  = float(df["close"].iloc[idx])
    open_val   = float(df["open"].iloc[idx])
    atr_value  = float(df["atr_14"].iloc[idx])

    if pd.isna(atr_value) or atr_value <= 0:
        return {
            "is_impulsive"   : False,
            "arah"           : None,
            "body"           : None,
            "atr_value"      : atr_value,
            "body_atr_ratio" : None,
            "threshold"      : None,
            "keterangan"     : f"atr_14 tidak valid di idx={idx}: {atr_value}",
        }

    body      = abs(close_val - open_val)
    threshold = impulsive_body_atr_ratio * atr_value

    if body == 0.0:
        # Doji — bukan impulsif
        return {
            "is_impulsive"   : False,
            "arah"           : None,
            "body"           : body,
            "atr_value"      : atr_value,
            "body_atr_ratio" : 0.0,
            "threshold"      : threshold,
            "keterangan"     : f"Doji (body=0): tidak impulsif",
        }

    body_atr_ratio = body / atr_value
    arah_candle    = "BULLISH" if close_val > open_val else "BEARISH"

    # Cek min_consecutive (semua candle dari start_idx s/d idx harus impulsif
    # dan searah)
    if min_consecutive_impulsive > 1:
        all_consecutive_ok = True
        for k in range(start_idx, idx):  # candle sebelum idx
            c_k    = float(df["close"].iloc[k])
            o_k    = float(df["open"].iloc[k])
            atr_k  = float(df["atr_14"].iloc[k])
            if pd.isna(atr_k) or atr_k <= 0:
                all_consecutive_ok = False
                break
            body_k = abs(c_k - o_k)
            if body_k < impulsive_body_atr_ratio * atr_k:
                all_consecutive_ok = False
                break
            arah_k = "BULLISH" if c_k > o_k else "BEARISH"
            if arah_k != arah_candle:
                all_consecutive_ok = False
                break

        if not all_consecutive_ok:
            return {
                "is_impulsive"   : False,
                "arah"           : arah_candle,
                "body"           : body,
                "atr_value"      : atr_value,
                "body_atr_ratio" : body_atr_ratio,
                "threshold"      : threshold,
                "keterangan"     : (
                    f"Candle di idx={idx} impulsif tapi tidak memenuhi "
                    f"min_consecutive={min_consecutive_impulsive} candle berturut-turut"
                ),
            }

    # ── Evaluasi threshold ───────────────────────────────────────────────────
    is_impulsive = body >= threshold

    if is_impulsive:
        keterangan = (
            f"IMPULSIF {arah_candle}: body={body:.4f} >= {impulsive_body_atr_ratio}×ATR "
            f"({threshold:.4f}), ratio={body_atr_ratio:.2f}x"
        )
    else:
        keterangan = (
            f"TIDAK impulsif: body={body:.4f} < {impulsive_body_atr_ratio}×ATR "
            f"({threshold:.4f}), ratio={body_atr_ratio:.2f}x"
        )

    return {
        "is_impulsive"   : is_impulsive,
        "arah"           : arah_candle if is_impulsive else None,
        "body"           : body,
        "atr_value"      : atr_value,
        "body_atr_ratio" : body_atr_ratio,
        "threshold"      : threshold,
        "keterangan"     : keterangan,
    }


# =============================================================================
# FUNGSI 2: DETEKSI ZONA DARI CANDLE ORIGIN
# =============================================================================

def detect_sd_zone_from_origin(
    df                       : pd.DataFrame,
    idx_impulsive            : int,
    impulsive_body_atr_ratio : float = DEFAULT_IMPULSIVE_RATIO,
) -> dict:
    """
    Bentuk zona S&D dari candle origin (candle tepat sebelum candle impulsif).

    ALUR:
        1. Verifikasi bahwa candle di idx_impulsive memang impulsif.
        2. Ambil candle origin = candle di idx_impulsive - 1.
        3. Verifikasi bahwa candle origin BUKAN candle impulsif sendiri
           (basing candle harus relatif "diam").
        4. Batas zona = [low_origin, high_origin] dari candle origin.

    SYARAT CANDLE ORIGIN (BUKAN impulsif):
        body_origin < ORIGIN_BODY_MAX_RATIO * impulsive_body_atr_ratio * atr_origin
        Di mana ORIGIN_BODY_MAX_RATIO = 0.5 (BELUM dikalibrasi).
        Artinya body origin harus < separuh threshold impulsif.

    JENIS ZONA:
        Bullish impulsive move → DEMAND zone (SL BUY = low_origin - buffer)
        Bearish impulsive move → SUPPLY zone (SL SELL = high_origin + buffer)

    Parameter:
        df                        : DataFrame dengan kolom open, high, low, close, atr_14.
        idx_impulsive             : Index candle impulsif (harus >= 1 agar ada origin).
        impulsive_body_atr_ratio  : Threshold body/ATR. Default 1.5 (BELUM dikalibrasi).

    Return:
        dict dengan field:
            is_valid    : bool   — True jika zona terbentuk dengan valid
            zone_type   : str    — "DEMAND" / "SUPPLY" / None
            low         : float  — low_origin (batas bawah zona)
            high        : float  — high_origin (batas atas zona)
            origin_idx  : int    — integer index candle origin di df
            keterangan  : str    — penjelasan ringkas
    """
    # ── Normalisasi idx_impulsive ────────────────────────────────────────────
    n = len(df)
    if idx_impulsive < 0:
        idx_impulsive = n + idx_impulsive

    # ── Validasi kolom ──────────────────────────────────────────────────────
    required_cols = {"open", "high", "low", "close", "atr_14"}
    missing = required_cols - set(df.columns)
    if missing:
        return _empty_zone(keterangan=f"Kolom tidak lengkap: {sorted(missing)}")

    # ── Validasi idx_impulsive ───────────────────────────────────────────────
    if idx_impulsive < 1 or idx_impulsive >= n:
        return _empty_zone(
            keterangan=(
                f"idx_impulsive={idx_impulsive} tidak valid: butuh 1 <= idx < {n}"
            )
        )

    # ── Verifikasi candle impulsif ───────────────────────────────────────────
    imp_result = detect_impulsive_move(df, idx=idx_impulsive,
                                       impulsive_body_atr_ratio=impulsive_body_atr_ratio)
    if not imp_result["is_impulsive"]:
        return _empty_zone(
            keterangan=(
                f"Candle idx={idx_impulsive} BUKAN impulsif: "
                f"{imp_result['keterangan']}"
            )
        )

    arah_impulsive = imp_result["arah"]  # "BULLISH" atau "BEARISH"

    # ── Ambil candle origin (idx_impulsive - 1) ──────────────────────────────
    # CAUSALITY: origin selalu berada sebelum idx_impulsive → aman.
    origin_idx = idx_impulsive - 1

    low_origin  = float(df["low"].iloc[origin_idx])
    high_origin = float(df["high"].iloc[origin_idx])
    open_origin = float(df["open"].iloc[origin_idx])
    close_origin = float(df["close"].iloc[origin_idx])
    atr_origin  = float(df["atr_14"].iloc[origin_idx])

    if pd.isna(atr_origin) or atr_origin <= 0:
        return _empty_zone(
            keterangan=f"atr_14 tidak valid di origin_idx={origin_idx}: {atr_origin}"
        )

    # ── Verifikasi candle origin BUKAN impulsif (harus "diam") ───────────────
    body_origin = abs(close_origin - open_origin)
    # Batas: body origin harus < ORIGIN_BODY_MAX_RATIO * threshold impulsif
    origin_max_body = ORIGIN_BODY_MAX_RATIO * impulsive_body_atr_ratio * atr_origin

    if body_origin >= origin_max_body:
        return _empty_zone(
            keterangan=(
                f"Candle origin idx={origin_idx} JUGA impulsif (body={body_origin:.4f} "
                f">= {ORIGIN_BODY_MAX_RATIO}×threshold={origin_max_body:.4f}): "
                f"bukan basing candle yang valid, zona tidak terbentuk"
            )
        )

    # ── Zona terbentuk ───────────────────────────────────────────────────────
    zone_type = "DEMAND" if arah_impulsive == "BULLISH" else "SUPPLY"

    keterangan = (
        f"{zone_type} ZONE terbentuk dari origin idx={origin_idx}: "
        f"[{low_origin:.2f}, {high_origin:.2f}] "
        f"(origin body={body_origin:.4f} < max={origin_max_body:.4f}, "
        f"impulsif {arah_impulsive} di idx={idx_impulsive})"
    )

    return {
        "is_valid"  : True,
        "zone_type" : zone_type,
        "low"       : low_origin,
        "high"      : high_origin,
        "origin_idx": origin_idx,
        "keterangan": keterangan,
    }


# =============================================================================
# FUNGSI 3: CARI ZONA S&D TERDEKAT
# =============================================================================

def find_nearest_sd_zone(
    df                       : pd.DataFrame,
    arah                     : str,
    idx                      : int,
    lookback                 : int   = SD_LOOKBACK,
    impulsive_body_atr_ratio : float = DEFAULT_IMPULSIVE_RATIO,
    buffer                   : float = SD_BUFFER,
) -> dict | None:
    """
    Cari zona S&D valid terdekat ke harga saat ini untuk keperluan SL.

    MENGIKUTI POLA find_nearest_swing() di risk_manager.py:
        - Scan mundur dari idx - 1 ke batas lookback.
        - Untuk setiap candle impulsif yang ditemukan, coba bentuk zona dari
          candle origin (candle sebelumnya).
        - Cek validitas (freshness check + invalidation check) zona tersebut.
        - Return zona TERDEKAT yang valid (tidak perlu yang terbaru — yang
          paling dekat ke harga current).

    ARAH & LEVEL SL:
        BUY  → cari DEMAND zone terdekat DI BAWAH harga current (close[idx]).
               SL level = low_origin - buffer
        SELL → cari SUPPLY zone terdekat DI ATAS harga current (close[idx]).
               SL level = high_origin + buffer

    FRESHNESS CHECK (per 3.1.3):
        Zona FRESH jika tidak ada candle k di (idx_impulsive, idx] dengan
        overlap zona: low_k <= high_origin AND high_k >= low_origin.
        Zona TESTED jika sudah pernah disentuh. Keduanya dikembalikan.

    INVALIDATION CHECK (per 3.1.4):
        Demand zone invalid jika ada candle k di (idx_impulsive, idx] dengan
        close_k < low_origin. (Harga sudah break LEWAT zona ke bawah.)
        Supply zone invalid jika ada candle k dengan close_k > high_origin.
        Zona invalid TIDAK dikembalikan.

    JIKA TIDAK ADA ZONA VALID:
        Return None — sama seperti find_nearest_swing() return None.
        Caller (calculate_sl_tp) harus fallback ke ATR.

    Parameter:
        df                        : DataFrame dengan kolom open, high, low, close, atr_14.
                                    Hanya data s/d idx yang dibaca (causal).
        arah                      : "BUY" (cari demand zone) atau "SELL" (cari supply zone).
        idx                       : Index candle evaluasi saat ini.
        lookback                  : Jumlah candle ke belakang. Default 50 (BELUM dikalibrasi).
        impulsive_body_atr_ratio  : Threshold impulsif. Default 1.5 (BELUM dikalibrasi).
        buffer                    : Buffer dollar di luar zona. Default 0.50 (= SWING_BUFFER).

    Return:
        dict dengan field:
            level      : float  — level SL kandidat (low_origin - buffer atau high_origin + buffer)
            zone_low   : float  — batas bawah zona origin
            zone_high  : float  — batas atas zona origin
            origin_idx : int    — index candle origin
            freshness  : str    — "FRESH" atau "TESTED"
            keterangan : str    — penjelasan ringkas
        ATAU None jika tidak ada zona valid dalam lookback.

    Catatan Kausalitas:
        Hanya membaca df.iloc[:idx+1]. Candle setelah idx tidak pernah disentuh.
    """
    # ── Normalisasi idx ──────────────────────────────────────────────────────
    n = len(df)
    if idx < 0:
        idx = n + idx

    # ── Validasi dasar ───────────────────────────────────────────────────────
    if arah not in ("BUY", "SELL"):
        return None

    if idx < 1 or idx >= n:
        return None

    required_cols = {"open", "high", "low", "close", "atr_14"}
    if not required_cols.issubset(set(df.columns)):
        return None

    # ── Harga current (candle idx) ───────────────────────────────────────────
    current_price = float(df["close"].iloc[idx])

    # ── Tentukan range scan ──────────────────────────────────────────────────
    # Kita scan candle impulsif dari idx-1 mundur ke batas lookback.
    # Candle idx sendiri tidak discan sebagai candle impulsif: zona terbentuk
    # dari pergerakan impulsif DI MASA LALU, bukan candle saat ini.
    # Minimum: idx_impulsive harus >= 1 (butuh origin di idx_impulsive - 1).
    scan_start = idx - 1                      # mulai dari candle sebelum idx
    scan_end   = max(idx - lookback, 1)       # batas mundur (tidak boleh < 1)

    # Koleksi zona valid untuk memilih yang terdekat ke current_price
    candidate_zones = []

    for idx_imp in range(scan_start, scan_end - 1, -1):
        # CAUSALITY: idx_imp <= idx - 1 < idx → selalu di masa lalu.
        # detect_sd_zone_from_origin hanya membaca idx_imp dan idx_imp-1.

        zone = detect_sd_zone_from_origin(
            df, idx_impulsive=idx_imp,
            impulsive_body_atr_ratio=impulsive_body_atr_ratio,
        )

        if not zone["is_valid"]:
            continue

        # ── Filter arah zona sesuai arah trade ──────────────────────────────
        # BUY → butuh DEMAND zone (bullish impulsive) DI BAWAH harga current
        # SELL → butuh SUPPLY zone (bearish impulsive) DI ATAS harga current
        zone_type  = zone["zone_type"]
        zone_low   = zone["low"]
        zone_high  = zone["high"]
        origin_idx = zone["origin_idx"]

        if arah == "BUY":
            if zone_type != "DEMAND":
                continue
            # Demand zone harus di BAWAH harga current
            if zone_high >= current_price:
                continue
        else:  # SELL
            if zone_type != "SUPPLY":
                continue
            # Supply zone harus di ATAS harga current
            if zone_low <= current_price:
                continue

        # ── Cek invalidasi dan freshness ─────────────────────────────────────
        # Scan candle k di (idx_imp, idx] — range setelah zona terbentuk
        # CAUSALITY: scan hanya sampai idx (tidak melewati idx).
        is_invalid  = False
        is_touched  = False  # untuk freshness check

        # Scan mulai dari candle SETELAH candle impulsif
        for k in range(idx_imp + 1, idx + 1):
            low_k  = float(df["low"].iloc[k])
            high_k = float(df["high"].iloc[k])
            close_k = float(df["close"].iloc[k])

            # Invalidation check
            if arah == "BUY":
                # Demand zone invalid jika close sudah break di bawah zone_low
                if close_k < zone_low:
                    is_invalid = True
                    break
            else:  # SELL
                # Supply zone invalid jika close sudah break di atas zone_high
                if close_k > zone_high:
                    is_invalid = True
                    break

            # Freshness/touched check: apakah candle ini overlap dengan zona?
            # Overlap = low_k <= zone_high AND high_k >= zone_low
            if not is_touched:
                if low_k <= zone_high and high_k >= zone_low:
                    is_touched = True

        if is_invalid:
            continue

        freshness = "FRESH" if not is_touched else "TESTED"

        # ── Hitung level SL kandidat ─────────────────────────────────────────
        if arah == "BUY":
            level = zone_low - buffer
        else:  # SELL
            level = zone_high + buffer

        candidate_zones.append({
            "level"      : level,
            "zone_low"   : zone_low,
            "zone_high"  : zone_high,
            "origin_idx" : origin_idx,
            "freshness"  : freshness,
            "keterangan" : (
                f"{zone_type} zone dari origin_idx={origin_idx} "
                f"[{zone_low:.2f}, {zone_high:.2f}] {freshness}, "
                f"SL level = {level:.2f} (buffer={buffer:.2f})"
            ),
        })

    if not candidate_zones:
        return None

    # ── Pilih zona TERDEKAT ke harga current ─────────────────────────────────
    # BUY  → SL harus di bawah current_price, pilih yang PALING TINGGI (terdekat)
    # SELL → SL harus di atas current_price, pilih yang PALING RENDAH (terdekat)
    if arah == "BUY":
        # Pilih zona dengan zone_high tertinggi (terdekat ke current_price dari bawah)
        best = max(candidate_zones, key=lambda z: z["zone_high"])
    else:
        # Pilih zona dengan zone_low terendah (terdekat ke current_price dari atas)
        best = min(candidate_zones, key=lambda z: z["zone_low"])

    return best


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _empty_zone(keterangan: str) -> dict:
    """Return dict standar untuk zona tidak valid."""
    return {
        "is_valid"  : False,
        "zone_type" : None,
        "low"       : None,
        "high"      : None,
        "origin_idx": None,
        "keterangan": keterangan,
    }
