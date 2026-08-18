"""
engine/strategies/breakout_retest_v2.py
========================================
Modul evaluasi entry trigger strategi Breakout Retest V2 untuk XAUUSD M5.

TUJUAN:
    Mendeteksi sinyal entry continuation (breakout retest) di timeframe M5
    ketika strategy_router (Fase 14) memilih "BREAKOUT_RETEST" — baik lewat:
    - source="DIRECT"      : regime M15 saat ini = BREAKOUT_TRANSITION
    - source="GRACE_WINDOW": regime M15 saat ini = CHOP, tapi breakout valid
                              terjadi beberapa candle lalu (grace window Fase 14)

    Prinsip inti: breakout sudah DIKONFIRMASI oleh regime detector M15 (Fase 13).
    Modul ini HANYA mencari pola touch (retest ke level breakout) → confirm
    (continuation setelah retest) di M5, menggunakan level boundary dari M15.

KAITAN DAN PERBEDAAN DENGAN FASE 9/10 LAMA:
    Proyek ini sebelumnya punya mekanisme retest di modul engine lama (Fase 9/10):
    - _check_breakout_trigger() : trigger M5, MASIH AKTIF di sistem lama.
    - _check_retest_trigger()   : trigger M5, FASE 10, STATUS: CLOSED — TIDAK
      LOLOS. Kegagalan: zero signal di bawah volume filter produksi, RRR realized
      negatif, SL terlalu sempit karena noise M5.

    Modul ini BERBEDA SECARA ARSITEKTURAL dari _check_retest_trigger() lama:

    1. GATING REGIME: Fase 10 lama mencoba retest di SEMUA kondisi market (trigger
       tambahan OR dengan EMA trigger, tanpa gating regime). Modul ini HANYA
       aktif ketika M15 regime detector (Fase 13) sudah mengonfirmasi
       BREAKOUT_TRANSITION — breakout divalidasi di level M15 yang jauh lebih
       robust dari cek breakout instan per-candle M5.

    2. LEVEL BOUNDARY: Fase 10 lama scan mundur M5 untuk menemukan candle breakout
       sendiri (karena breakout adalah trigger M5 yang belum tentu ada). Modul
       ini MEMPERCAYAI bahwa breakout sudah terjadi dan menerima level boundary
       (resistance/support) sebagai parameter dari M15 — tidak mencari breakout
       dari nol.

    3. TOLERANSI ATR-RELATIVE: Fase 10 lama pakai angka dollar tetap
       (_RETEST_SWING_BUFFER=0.50, _RETEST_BODY_MIN_RATIO=0.3 — tapi dalam
       satuan price, bukan ATR). Modul ini pakai satuan ATR-relative, konsisten
       dengan modul baru lain di upgrade ini (zone_detector.py, supply_demand.py,
       regime_detector.py).

    4. TIDAK ADA FILTER VOLUME di layer ini. Fase 10 gagal sebagian karena volume
       filter produksi memblokir semua sinyal retest. Volume filter — jika
       diperlukan — adalah keputusan terpisah di Fase 18/19.

    PENTING: Modul ini adalah VALIDASI BARU DARI NOL. Status LOLOS Fase 13
    (regime detector) TIDAK otomatis berarti strategi ini akan LOLOS backtest.
    Verifikasi empiris dilakukan gabungan Fase 21/22.

CAUSALITY (ZERO LOOK-AHEAD):
    Fungsi ini hanya membaca df_m5.iloc[:idx_m5+1] (candle pada dan SEBELUM
    idx_m5). Baris manapun sesudah idx_m5 TIDAK PERNAH disentuh. Ini dibuktikan
    secara eksplisit via test mutasi ekstrem di tests/test_breakout_retest_v2.py.

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.

CATATAN "BELUM DIKALIBRASI":
    Semua konstanta di modul ini adalah nilai awal struktural. Kalibrasi empiris
    berbasis backtest dilakukan di Fase 21/22.
"""

import pandas as pd


# =============================================================================
# KONSTANTA (BELUM DIKALIBRASI — nilai awal struktural, bukan hasil backtest)
# =============================================================================

BREAKOUT_RETEST_LOOKBACK_M5 = 12
# Jumlah candle M5 ke belakang untuk mencari retest touch (~1 jam).
# Sepadan dengan grace window M15 di Fase 14 (4 candle M15 = 12 candle M5).
# BELUM dikalibrasi.

BREAKOUT_RETEST_TOUCH_TOLERANCE_ATR = 0.3
# Toleransi "menyentuh" level boundary dalam satuan ATR.
# Candle dianggap "touch" jika |low - level_ref| <= toleransi * ATR (BULLISH)
# atau |high - level_ref| <= toleransi * ATR (BEARISH).
# BELUM dikalibrasi.

BREAKOUT_RETEST_INVALIDATION_BUFFER_ATR = 0.5
# Seberapa jauh harga boleh masuk kembali ke zona lama setelah touch sebelum
# dianggap invalidasi. Diukur dari level referensi dalam satuan ATR.
# BELUM dikalibrasi.

BREAKOUT_RETEST_MIN_BODY_ATR_RATIO = 0.3
# Body minimum candle konfirmasi relatif terhadap ATR.
# Body = abs(close - open). Terlalu kecil → candle ragu-ragu, bukan konfirmasi.
# BELUM dikalibrasi.


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def evaluate_breakout_retest(
    df_m5: pd.DataFrame,
    idx_m5: int,
    zone: dict,
    arah: str,
    retest_lookback_m5: int = BREAKOUT_RETEST_LOOKBACK_M5,
) -> dict:
    """
    Evaluasi apakah kondisi di sekitar candle M5 idx_m5 memenuhi syarat entry
    Breakout Retest V2.

    Strategi ini aktif ketika strategy_router memilih "BREAKOUT_RETEST"
    (regime M15 = BREAKOUT_TRANSITION, atau dalam grace window Fase 14).
    Fungsi ini mencari pola retest touch → konfirmasi continuation di M5,
    menggunakan level boundary yang sudah ditentukan oleh detect_market_regime()
    di level M15 — TIDAK menghitung atau mencari breakout sendiri.

    CAUSALITY: Hanya membaca df_m5.iloc[:idx_m5+1]. Baris sesudah idx_m5
    TIDAK pernah disentuh dalam bentuk apapun.

    Parameter:
        df_m5              : DataFrame M5 dengan kolom OHLC + atr_14 (sudah
                             melewati run_all_indicators()). Kolom wajib:
                             open, high, low, close, atr_14.
        idx_m5             : Index candle M5 evaluasi (boleh negatif, akan
                             dinormalisasi).
        zone               : dict boundary dari detect_market_regime() field
                             "zone" saat regime == "BREAKOUT_TRANSITION".
                             Minimal berisi: {"resistance": float, "support": float}
                             Jika None atau key bernilai None → return
                             terpenuhi=False, tidak crash.
        arah               : "BULLISH" (breakout ke atas, cari BUY retest) atau
                             "BEARISH" (breakout ke bawah, cari SELL retest).
                             Diterima dari field "arah" hasil detect_market_regime()
                             — TIDAK dihitung ulang di modul ini.
        retest_lookback_m5 : Jumlah candle M5 ke belakang untuk scan touch.
                             Default = BREAKOUT_RETEST_LOOKBACK_M5.

    Return dict (field wajib):
        {
            "terpenuhi"            : bool   — True jika semua syarat terpenuhi
            "arah"                 : str    — "BUY", "SELL", atau "NETRAL"
            "level_referensi"      : float|None — level yang dievaluasi
            "touch_idx"            : int|None   — index candle touch terbaru
            "candles_since_touch"  : int|None   — selisih idx_m5 - touch_idx
            "invalidated"          : bool   — True jika ada invalidasi setelah touch
            "konfirmasi_close"     : bool   — close candle idx_m5 menembus level
            "konfirmasi_body"      : bool   — body candle idx_m5 cukup besar
            "invalidation_level_sl": float|None — level SL mentah untuk Fase 19
            "keterangan"           : str    — penjelasan lengkap + audit trail
        }
    """
    # ── Normalisasi idx negatif ───────────────────────────────────────────────
    n = len(df_m5)
    if idx_m5 < 0:
        idx_m5 = n + idx_m5

    # ── Return kosong standar (helper lokal) ──────────────────────────────────
    def _kosong(keterangan: str, invalidated: bool = False) -> dict:
        return {
            "terpenuhi"            : False,
            "arah"                 : "NETRAL",
            "level_referensi"      : None,
            "touch_idx"            : None,
            "candles_since_touch"  : None,
            "invalidated"          : invalidated,
            "konfirmasi_close"     : False,
            "konfirmasi_body"      : False,
            "invalidation_level_sl": None,
            "keterangan"           : keterangan,
        }

    # ── Guard 1: Validasi idx_m5 dalam range ─────────────────────────────────
    if idx_m5 < 0 or idx_m5 >= n:
        return _kosong(
            f"Breakout Retest V2: idx_m5={idx_m5} di luar range valid [0, {n - 1}]"
        )

    # ── Guard 2: Validasi kolom df_m5 ────────────────────────────────────────
    kolom_wajib = {"open", "high", "low", "close", "atr_14"}
    kolom_hilang = kolom_wajib - set(df_m5.columns)
    if kolom_hilang:
        return _kosong(
            f"Breakout Retest V2: kolom tidak lengkap — "
            f"{sorted(kolom_hilang)} tidak ada di df_m5"
        )

    # ── Guard 3: Validasi zone ────────────────────────────────────────────────
    if zone is None:
        return _kosong(
            "Breakout Retest V2: zone=None — boundary tidak tersedia, tidak bisa evaluasi"
        )

    resistance = zone.get("resistance")
    support    = zone.get("support")

    # ── Guard 4: Validasi arah dan tentukan level referensi ───────────────────
    if arah == "BULLISH":
        if resistance is None:
            return _kosong(
                "Breakout Retest V2: arah=BULLISH tapi zone['resistance']=None — "
                "level referensi tidak tersedia"
            )
        level_ref  = float(resistance)
        arah_entry = "BUY"
    elif arah == "BEARISH":
        if support is None:
            return _kosong(
                "Breakout Retest V2: arah=BEARISH tapi zone['support']=None — "
                "level referensi tidak tersedia"
            )
        level_ref  = float(support)
        arah_entry = "SELL"
    else:
        return _kosong(
            f"Breakout Retest V2: arah='{arah}' tidak dikenal — "
            f"harus 'BULLISH' atau 'BEARISH'"
        )

    # ── Guard 5: Validasi ATR di candle evaluasi ──────────────────────────────
    atr_eval = float(df_m5.iloc[idx_m5]["atr_14"])
    if pd.isna(atr_eval) or atr_eval <= 0:
        return _kosong(
            f"Breakout Retest V2: atr_14 tidak valid di idx_m5={idx_m5}: {atr_eval}"
        )

    # =========================================================================
    # LANGKAH 2 — SCAN MUNDUR: CARI RETEST TOUCH PALING BARU
    # =========================================================================
    #
    # Scan k dari (idx_m5 - 1) sampai max(idx_m5 - retest_lookback_m5, 0),
    # mundur (dari yang paling dekat idx_m5 ke yang paling jauh).
    # Ambil touch PERTAMA yang ditemukan (= paling baru), lalu berhenti.
    #
    # BULLISH: touch jika |low[k] - level_ref| <= TOUCH_TOLERANCE_ATR * atr_14[k]
    # BEARISH: touch jika |high[k] - level_ref| <= TOUCH_TOLERANCE_ATR * atr_14[k]

    touch_idx   = None
    batas_bawah = max(idx_m5 - retest_lookback_m5, 0)

    for k in range(idx_m5 - 1, batas_bawah - 1, -1):
        baris_k = df_m5.iloc[k]
        atr_k   = float(baris_k["atr_14"])

        # Lewati candle dengan ATR tidak valid dalam scan (jangan crash)
        if pd.isna(atr_k) or atr_k <= 0:
            continue

        toleransi_k = BREAKOUT_RETEST_TOUCH_TOLERANCE_ATR * atr_k

        if arah == "BULLISH":
            low_k = float(baris_k["low"])
            if abs(low_k - level_ref) <= toleransi_k:
                touch_idx = k
                break
        else:  # BEARISH
            high_k = float(baris_k["high"])
            if abs(high_k - level_ref) <= toleransi_k:
                touch_idx = k
                break

    if touch_idx is None:
        return _kosong(
            f"Breakout Retest V2: belum ada retest touch dalam window "
            f"{retest_lookback_m5} candle ke belakang dari idx_m5={idx_m5}. "
            f"level_ref={level_ref:.4f}, arah={arah}"
        )

    candles_since_touch = idx_m5 - touch_idx

    # =========================================================================
    # LANGKAH 3 — CEK INVALIDASI (dari touch_idx+1 sampai idx_m5-1)
    # =========================================================================
    #
    # Setelah touch ditemukan, cek apakah ada candle yang close terlalu jauh
    # kembali ke dalam zona lama — pertanda breakout gagal (false breakout).
    #
    # BULLISH: invalidasi jika close[m] < level_ref - (INVALIDATION_BUFFER_ATR * atr_14[m])
    # BEARISH: invalidasi jika close[m] > level_ref + (INVALIDATION_BUFFER_ATR * atr_14[m])

    for m in range(touch_idx + 1, idx_m5):
        baris_m = df_m5.iloc[m]
        atr_m   = float(baris_m["atr_14"])

        if pd.isna(atr_m) or atr_m <= 0:
            continue

        buffer_m = BREAKOUT_RETEST_INVALIDATION_BUFFER_ATR * atr_m

        if arah == "BULLISH":
            close_m = float(baris_m["close"])
            if close_m < level_ref - buffer_m:
                return _kosong(
                    f"Breakout Retest V2: INVALIDASI setelah touch di idx={touch_idx}. "
                    f"Candle idx={m}: close={close_m:.4f} < "
                    f"level_ref={level_ref:.4f} - buffer={buffer_m:.4f} "
                    f"= {level_ref - buffer_m:.4f}. "
                    f"Breakout dianggap gagal (false breakout).",
                    invalidated=True,
                )
        else:  # BEARISH
            close_m = float(baris_m["close"])
            if close_m > level_ref + buffer_m:
                return _kosong(
                    f"Breakout Retest V2: INVALIDASI setelah touch di idx={touch_idx}. "
                    f"Candle idx={m}: close={close_m:.4f} > "
                    f"level_ref={level_ref:.4f} + buffer={buffer_m:.4f} "
                    f"= {level_ref + buffer_m:.4f}. "
                    f"Breakout dianggap gagal (false breakout).",
                    invalidated=True,
                )

    # =========================================================================
    # LANGKAH 4 — CEK KONFIRMASI di candle idx_m5
    # =========================================================================
    #
    # body = abs(close - open)
    #
    # BULLISH:
    #   konfirmasi_close = close[idx_m5] > level_ref
    #   konfirmasi_body  = body >= BREAKOUT_RETEST_MIN_BODY_ATR_RATIO * atr_14[idx_m5]
    #
    # BEARISH:
    #   konfirmasi_close = close[idx_m5] < level_ref
    #   konfirmasi_body  = body >= BREAKOUT_RETEST_MIN_BODY_ATR_RATIO * atr_14[idx_m5]
    #
    # terpenuhi = True HANYA jika konfirmasi_close AND konfirmasi_body.

    baris_eval  = df_m5.iloc[idx_m5]
    open_eval   = float(baris_eval["open"])
    high_eval   = float(baris_eval["high"])
    low_eval    = float(baris_eval["low"])
    close_eval  = float(baris_eval["close"])

    body        = abs(close_eval - open_eval)
    body_min    = BREAKOUT_RETEST_MIN_BODY_ATR_RATIO * atr_eval

    if arah == "BULLISH":
        konfirmasi_close = close_eval > level_ref
        konfirmasi_body  = body >= body_min
    else:  # BEARISH
        konfirmasi_close = close_eval < level_ref
        konfirmasi_body  = body >= body_min

    # ── Bangun keterangan audit untuk konfirmasi ──────────────────────────────
    if konfirmasi_close and konfirmasi_body:
        # ── Terpenuhi: ambil invalidation_level_sl dari candle touch ──────────
        baris_touch = df_m5.iloc[touch_idx]
        if arah == "BULLISH":
            invalidation_level_sl = float(baris_touch["low"])
        else:  # BEARISH
            invalidation_level_sl = float(baris_touch["high"])

        ket = (
            f"Breakout Retest V2 {arah_entry}: "
            f"touch di idx={touch_idx} ({candles_since_touch} candle lalu), "
            f"level_ref={level_ref:.4f}, "
            f"konfirmasi close={'OK' if konfirmasi_close else 'GAGAL'} "
            f"(close={close_eval:.4f} {'>' if arah == 'BULLISH' else '<'} {level_ref:.4f}), "
            f"konfirmasi body={'OK' if konfirmasi_body else 'GAGAL'} "
            f"(body={body:.4f} >= {BREAKOUT_RETEST_MIN_BODY_ATR_RATIO}*ATR={atr_eval:.4f} "
            f"= {body_min:.4f}), "
            f"invalidation_level_sl={invalidation_level_sl:.4f}"
        )

        return {
            "terpenuhi"            : True,
            "arah"                 : arah_entry,
            "level_referensi"      : level_ref,
            "touch_idx"            : touch_idx,
            "candles_since_touch"  : candles_since_touch,
            "invalidated"          : False,
            "konfirmasi_close"     : True,
            "konfirmasi_body"      : True,
            "invalidation_level_sl": invalidation_level_sl,
            "keterangan"           : ket,
        }

    # ── Konfirmasi gagal — audit komponen mana yang tidak terpenuhi ───────────
    komponen_gagal = []
    if not konfirmasi_close:
        if arah == "BULLISH":
            komponen_gagal.append(
                f"konfirmasi_close GAGAL: close={close_eval:.4f} <= level_ref={level_ref:.4f} "
                f"(butuh close > level_ref untuk BUY)"
            )
        else:
            komponen_gagal.append(
                f"konfirmasi_close GAGAL: close={close_eval:.4f} >= level_ref={level_ref:.4f} "
                f"(butuh close < level_ref untuk SELL)"
            )
    if not konfirmasi_body:
        komponen_gagal.append(
            f"konfirmasi_body GAGAL: body={body:.4f} < "
            f"{BREAKOUT_RETEST_MIN_BODY_ATR_RATIO}*ATR={atr_eval:.4f} = {body_min:.4f}"
        )

    ket = (
        f"Breakout Retest V2: touch ditemukan di idx={touch_idx} "
        f"({candles_since_touch} candle lalu), level_ref={level_ref:.4f}, "
        f"TAPI konfirmasi GAGAL — " + "; ".join(komponen_gagal)
    )

    return {
        "terpenuhi"            : False,
        "arah"                 : "NETRAL",
        "level_referensi"      : level_ref,
        "touch_idx"            : touch_idx,
        "candles_since_touch"  : candles_since_touch,
        "invalidated"          : False,
        "konfirmasi_close"     : konfirmasi_close,
        "konfirmasi_body"      : konfirmasi_body,
        "invalidation_level_sl": None,
        "keterangan"           : ket,
    }
