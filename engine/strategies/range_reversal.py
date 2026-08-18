"""
engine/strategies/range_reversal.py
=====================================
Modul evaluasi entry trigger strategi Range Reversal untuk XAUUSD M5.

TUJUAN:
    Mendeteksi sinyal entry mean-reversion di timeframe M5 ketika market regime
    M15 terdeteksi sebagai RANGING oleh detect_market_regime() (Fase 13).
    Prinsip inti: "Trade the boundaries, not the middle" — entry HANYA di dekat
    boundary range (support/resistance) yang sudah ditentukan M15, TIDAK PERNAH
    di tengah range.

KARAKTER MODUL (SINGLE-CANDLE STATELESS):
    Fungsi utama evaluate_range_reversal() membaca SATU baris DataFrame M5
    (df_m5.iloc[idx_m5]) — tidak ada lookback ke belakang, tidak ada state
    antar-pemanggilan. Setiap pemanggilan identik (df_m5, idx_m5, zone) akan
    selalu menghasilkan output yang sama (deterministik).

CAUSALITY (ZERO LOOK-AHEAD):
    Fungsi ini hanya membaca df_m5.iloc[idx_m5]. Baris manapun di luar idx_m5
    TIDAK PERNAH disentuh — baik sebelum maupun sesudahnya. Ini dibuktikan
    secara eksplisit via test mutasi ekstrem di tests/test_range_reversal.py
    (TestKausalitasSingleCandle).

KONFIRMASI ENTRY (AND — KEDUANYA WAJIB):
    Entry terpenuhi HANYA jika DUA komponen terpenuhi BERSAMAAN:
    (a) Sweep + Reclaim : wick menembus boundary dengan kedalaman cukup,
                          close kembali ke dalam range.
    (b) Rejection Candle: bentuk candle menunjukkan penolakan harga di boundary
                          (lower wick panjang untuk BUY, upper wick untuk SELL,
                          close pada posisi yang tepat).
    Alasan AND bukan OR: filosofi proyek "lebih baik kehilangan peluang daripada
    entry lemah." Jika backtest Fase 21/22 menunjukkan ini terlalu ketat, dapat
    dilonggarkan ke OR — keputusan kalibrasi berbasis data, bukan sekarang.

SL/TP:
    Modul ini TIDAK menghitung SL, TP, atau RRR. Hanya mengembalikan
    invalidation_level (level referensi mentah) yang akan digunakan oleh
    Fase 19 (Regime-Aware Risk Management) untuk meneruskan ke calculate_sl_tp()
    di engine/risk_manager.py.

INPUT BOUNDARY:
    Boundary range (resistance, support) diterima sebagai parameter zone —
    TIDAK dihitung ulang di sini. Caller (nantinya Fase 21) meneruskan field
    "zone" dari detect_market_regime() secara langsung.

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.
"""

import pandas as pd


# =============================================================================
# KONSTANTA (BELUM DIKALIBRASI — nilai awal struktural, bukan hasil backtest)
# =============================================================================

RANGE_REVERSAL_MIN_SWEEP_DEPTH_ATR = 0.1
# Minimal penetrasi wick ke luar boundary dalam satuan ATR.
# Contoh: ATR=5.0, maka wick harus menembus minimal 0.5 poin di luar boundary.
# BELUM dikalibrasi.

RANGE_REVERSAL_MIN_REJECTION_WICK_ATR = 0.5
# Minimal panjang wick penolakan (lower wick untuk BUY, upper wick untuk SELL)
# dalam satuan ATR. Ini memastikan ada tekanan harga yang signifikan.
# BELUM dikalibrasi.

RANGE_REVERSAL_MIN_CLOSE_POSITION = 0.5
# Posisi close dalam range candle (0.0 = di low, 1.0 = di high).
# Untuk BUY: close harus di atas 50% range candle (close dekat high candle).
# Untuk SELL: (high - close) / range harus >= 0.5 (close dekat low candle).
# BELUM dikalibrasi.


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _detect_rejection_wick(
    open_: float,
    high: float,
    low: float,
    close: float,
    atr: float,
    arah: str,
) -> dict:
    """
    Deteksi pola rejection candle berdasarkan panjang wick dan posisi close.

    Didesain SELF-CONTAINED untuk konteks rejection di LOKASI boundary range —
    BUKAN reuse dari engine/candle_patterns.py (yang dikalibrasi untuk konteks
    berbeda, yaitu scoring umum di entry M5 generik).

    Parameter:
        open_ : float — harga open candle
        high  : float — harga high candle
        low   : float — harga low candle
        close : float — harga close candle
        atr   : float — nilai ATR-14 candle
        arah  : str   — "BUY" (cek lower wick) atau "SELL" (cek upper wick)

    Return:
        dict berisi:
            "terpenuhi"      : bool  — True jika rejection terkonfirmasi
            "wick_length"    : float — panjang wick yang dievaluasi
            "close_position" : float — posisi close dalam range candle (0.0–1.0)
            "keterangan"     : str   — penjelasan ringkas

    Kriteria BUY (lower wick):
        lower_wick   = min(open_, close) - low
        close_posisi = (close - low) / (high - low)
        terpenuhi    = lower_wick >= MIN_REJECTION_WICK_ATR * atr
                       AND close_posisi >= MIN_CLOSE_POSITION

    Kriteria SELL (upper wick — cerminan BUY):
        upper_wick   = high - max(open_, close)
        close_posisi = (high - close) / (high - low)   [close dekat low candle]
        terpenuhi    = upper_wick >= MIN_REJECTION_WICK_ATR * atr
                       AND close_posisi >= MIN_CLOSE_POSITION
    """
    range_total = high - low

    # Hindari divisi nol pada candle doji sempurna (range = 0)
    if range_total <= 0:
        return {
            "terpenuhi"      : False,
            "wick_length"    : 0.0,
            "close_position" : 0.5,
            "keterangan"     : "Rejection: range candle = 0 (doji sempurna), tidak bisa evaluasi",
        }

    batas_wick = RANGE_REVERSAL_MIN_REJECTION_WICK_ATR * atr

    if arah == "BUY":
        lower_wick    = min(open_, close) - low
        close_posisi  = (close - low) / range_total
        wick_length   = lower_wick

        wick_ok       = lower_wick >= batas_wick
        posisi_ok     = close_posisi >= RANGE_REVERSAL_MIN_CLOSE_POSITION
        terpenuhi     = wick_ok and posisi_ok

        if terpenuhi:
            ket = (
                f"Rejection BUY: lower_wick={lower_wick:.4f} >= "
                f"{RANGE_REVERSAL_MIN_REJECTION_WICK_ATR}*ATR({atr:.4f})={batas_wick:.4f}, "
                f"close_posisi={close_posisi:.3f} >= {RANGE_REVERSAL_MIN_CLOSE_POSITION}"
            )
        elif not wick_ok:
            ket = (
                f"Rejection BUY GAGAL: lower_wick={lower_wick:.4f} < "
                f"{RANGE_REVERSAL_MIN_REJECTION_WICK_ATR}*ATR({atr:.4f})={batas_wick:.4f}"
            )
        else:
            ket = (
                f"Rejection BUY GAGAL: close_posisi={close_posisi:.3f} < "
                f"{RANGE_REVERSAL_MIN_CLOSE_POSITION} (close terlalu rendah dalam candle)"
            )

    else:  # arah == "SELL"
        upper_wick    = high - max(open_, close)
        close_posisi  = (high - close) / range_total
        wick_length   = upper_wick

        wick_ok       = upper_wick >= batas_wick
        posisi_ok     = close_posisi >= RANGE_REVERSAL_MIN_CLOSE_POSITION
        terpenuhi     = wick_ok and posisi_ok

        if terpenuhi:
            ket = (
                f"Rejection SELL: upper_wick={upper_wick:.4f} >= "
                f"{RANGE_REVERSAL_MIN_REJECTION_WICK_ATR}*ATR({atr:.4f})={batas_wick:.4f}, "
                f"close_posisi={close_posisi:.3f} >= {RANGE_REVERSAL_MIN_CLOSE_POSITION}"
            )
        elif not wick_ok:
            ket = (
                f"Rejection SELL GAGAL: upper_wick={upper_wick:.4f} < "
                f"{RANGE_REVERSAL_MIN_REJECTION_WICK_ATR}*ATR({atr:.4f})={batas_wick:.4f}"
            )
        else:
            ket = (
                f"Rejection SELL GAGAL: close_posisi={close_posisi:.3f} < "
                f"{RANGE_REVERSAL_MIN_CLOSE_POSITION} (close terlalu tinggi dalam candle)"
            )

    return {
        "terpenuhi"      : terpenuhi,
        "wick_length"    : float(wick_length),
        "close_position" : float(close_posisi),
        "keterangan"     : ket,
    }


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def evaluate_range_reversal(
    df_m5: pd.DataFrame,
    idx_m5: int,
    zone: dict,
) -> dict:
    """
    Evaluasi apakah candle M5 pada idx_m5 memenuhi syarat entry Range Reversal.

    Strategi ini aktif ketika strategy_router memilih "RANGE_REVERSAL"
    (regime M15 = RANGING). Fungsi ini mencari entry trigger di M5 berdasarkan
    boundary range yang sudah ditentukan M15 — TIDAK menghitung boundary dari nol.

    CAUSALITY: Hanya membaca df_m5.iloc[idx_m5] — satu baris. Baris lain
    di luar idx_m5 TIDAK disentuh dalam bentuk apapun.

    Parameter:
        df_m5  : DataFrame M5 dengan kolom OHLC + atr_14 (sudah melewati
                 run_all_indicators()). Kolom wajib: open, high, low, close, atr_14.
        idx_m5 : Index candle M5 evaluasi (boleh negatif, akan dinormalisasi).
        zone   : dict boundary dari detect_market_regime() field "zone" saat
                 regime == "RANGING". Minimal berisi:
                     {"resistance": float, "support": float}
                 Jika zone is None atau key bernilai None → return terpenuhi=False,
                 tidak crash.

    Return dict (field wajib):
        {
            "terpenuhi"          : bool   — True jika BUY atau SELL terpenuhi
            "arah"               : str    — "BUY", "SELL", atau "NETRAL"
            "boundary_referensi" : str|None — "support", "resistance", atau None
            "sweep_terpenuhi"    : bool   — apakah sweep+reclaim terpenuhi
            "sweep_depth"        : float|None — kedalaman wick menembus boundary
            "rejection_terpenuhi": bool   — apakah rejection candle terpenuhi
            "rejection_detail"   : dict   — output dari _detect_rejection_wick()
            "invalidation_level" : float|None — level referensi SL mentah (Fase 19)
            "keterangan"         : str    — penjelasan lengkap + audit trail
        }
    """
    # ── Normalisasi idx negatif ──────────────────────────────────────────────
    n = len(df_m5)
    if idx_m5 < 0:
        idx_m5 = n + idx_m5

    # ── Return kosong standar (helper lokal) ─────────────────────────────────
    def _kosong(keterangan: str) -> dict:
        return {
            "terpenuhi"          : False,
            "arah"               : "NETRAL",
            "boundary_referensi" : None,
            "sweep_terpenuhi"    : False,
            "sweep_depth"        : None,
            "rejection_terpenuhi": False,
            "rejection_detail"   : {
                "terpenuhi"      : False,
                "wick_length"    : None,
                "close_position" : None,
                "keterangan"     : "Tidak dievaluasi",
            },
            "invalidation_level" : None,
            "keterangan"         : keterangan,
        }

    # ── Guard 1: Validasi zone ───────────────────────────────────────────────
    if zone is None:
        return _kosong("Range Reversal: zone=None — boundary tidak tersedia, tidak bisa evaluasi")

    resistance = zone.get("resistance")
    support    = zone.get("support")

    if resistance is None or support is None:
        return _kosong(
            f"Range Reversal: zone tidak lengkap — "
            f"resistance={resistance}, support={support} (salah satu atau keduanya None)"
        )

    resistance = float(resistance)
    support    = float(support)

    # ── Guard 2: Validasi kolom df_m5 ───────────────────────────────────────
    kolom_wajib = {"open", "high", "low", "close", "atr_14"}
    kolom_hilang = kolom_wajib - set(df_m5.columns)
    if kolom_hilang:
        return _kosong(
            f"Range Reversal: kolom tidak lengkap — {sorted(kolom_hilang)} tidak ada di df_m5"
        )

    # ── Guard 3: Validasi idx_m5 dalam range ────────────────────────────────
    if idx_m5 < 0 or idx_m5 >= n:
        return _kosong(
            f"Range Reversal: idx_m5={idx_m5} di luar range valid [0, {n - 1}]"
        )

    # ── Baca SATU baris (kausalitas terjamin) ────────────────────────────────
    baris  = df_m5.iloc[idx_m5]
    open_  = float(baris["open"])
    high   = float(baris["high"])
    low    = float(baris["low"])
    close  = float(baris["close"])
    atr    = float(baris["atr_14"])

    # ── Guard 4: Validasi ATR ────────────────────────────────────────────────
    if pd.isna(atr) or atr <= 0:
        return _kosong(
            f"Range Reversal: atr_14 tidak valid di idx_m5={idx_m5}: {atr}"
        )

    min_sweep_depth = RANGE_REVERSAL_MIN_SWEEP_DEPTH_ATR * atr

    # =========================================================================
    # EVALUASI BUY — entry di support
    # =========================================================================
    #
    # (a) Sweep + Reclaim:
    #     1. low < support                    → wick menembus ke bawah boundary
    #     2. support - low >= min_sweep_depth → penetrasi cukup dalam
    #     3. close >= support                 → close kembali ke dalam range
    #
    # (b) Rejection Candle (lower wick panjang, close posisi tinggi dalam candle)
    #
    # BUY TERPENUHI hanya jika (a) AND (b) keduanya True.

    sweep_buy        = False
    sweep_depth_buy  = None

    if low < support:
        kedalaman = support - low
        if kedalaman >= min_sweep_depth and close >= support:
            sweep_buy       = True
            sweep_depth_buy = float(kedalaman)

    if sweep_buy:
        rejection_buy = _detect_rejection_wick(open_, high, low, close, atr, "BUY")
        if rejection_buy["terpenuhi"]:
            return {
                "terpenuhi"          : True,
                "arah"               : "BUY",
                "boundary_referensi" : "support",
                "sweep_terpenuhi"    : True,
                "sweep_depth"        : sweep_depth_buy,
                "rejection_terpenuhi": True,
                "rejection_detail"   : rejection_buy,
                "invalidation_level" : low,   # SL kandidat mentah — buffer ditambah Fase 19
                "keterangan"         : (
                    f"Range Reversal BUY: sweep support={support:.4f} "
                    f"(depth={sweep_depth_buy:.4f} >= {min_sweep_depth:.4f}), "
                    f"close={close:.4f} >= support, "
                    f"rejection: {rejection_buy['keterangan']}"
                ),
            }
        else:
            # Sweep ok tapi rejection gagal → tidak entry, tapi audit lengkap
            return {
                "terpenuhi"          : False,
                "arah"               : "NETRAL",
                "boundary_referensi" : "support",
                "sweep_terpenuhi"    : True,
                "sweep_depth"        : sweep_depth_buy,
                "rejection_terpenuhi": False,
                "rejection_detail"   : rejection_buy,
                "invalidation_level" : None,
                "keterangan"         : (
                    f"Range Reversal: sweep support OK (depth={sweep_depth_buy:.4f}), "
                    f"TAPI rejection GAGAL — {rejection_buy['keterangan']}"
                ),
            }

    # =========================================================================
    # EVALUASI SELL — entry di resistance (hanya jika BUY tidak terpenuhi)
    # =========================================================================
    #
    # (a) Sweep + Reclaim:
    #     1. high > resistance                      → wick menembus ke atas boundary
    #     2. high - resistance >= min_sweep_depth   → penetrasi cukup dalam
    #     3. close <= resistance                    → close kembali ke dalam range
    #
    # (b) Rejection Candle (upper wick panjang, close posisi rendah dalam candle)

    sweep_sell        = False
    sweep_depth_sell  = None

    if high > resistance:
        kedalaman = high - resistance
        if kedalaman >= min_sweep_depth and close <= resistance:
            sweep_sell       = True
            sweep_depth_sell = float(kedalaman)

    if sweep_sell:
        rejection_sell = _detect_rejection_wick(open_, high, low, close, atr, "SELL")
        if rejection_sell["terpenuhi"]:
            return {
                "terpenuhi"          : True,
                "arah"               : "SELL",
                "boundary_referensi" : "resistance",
                "sweep_terpenuhi"    : True,
                "sweep_depth"        : sweep_depth_sell,
                "rejection_terpenuhi": True,
                "rejection_detail"   : rejection_sell,
                "invalidation_level" : high,   # SL kandidat mentah — buffer ditambah Fase 19
                "keterangan"         : (
                    f"Range Reversal SELL: sweep resistance={resistance:.4f} "
                    f"(depth={sweep_depth_sell:.4f} >= {min_sweep_depth:.4f}), "
                    f"close={close:.4f} <= resistance, "
                    f"rejection: {rejection_sell['keterangan']}"
                ),
            }
        else:
            return {
                "terpenuhi"          : False,
                "arah"               : "NETRAL",
                "boundary_referensi" : "resistance",
                "sweep_terpenuhi"    : True,
                "sweep_depth"        : sweep_depth_sell,
                "rejection_terpenuhi": False,
                "rejection_detail"   : rejection_sell,
                "invalidation_level" : None,
                "keterangan"         : (
                    f"Range Reversal: sweep resistance OK (depth={sweep_depth_sell:.4f}), "
                    f"TAPI rejection GAGAL — {rejection_sell['keterangan']}"
                ),
            }

    # =========================================================================
    # TIDAK ADA TRIGGER — audit kenapa
    # =========================================================================
    #
    # Diagnosa lebih detail: apakah setidaknya ada wick menembus? atau sama sekali
    # tidak menyentuh boundary?

    menyentuh_support    = low < support
    menyentuh_resistance = high > resistance

    if not menyentuh_support and not menyentuh_resistance:
        ket = (
            f"Range Reversal: NETRAL — candle tidak menyentuh boundary mana pun. "
            f"low={low:.4f} >= support={support:.4f}, "
            f"high={high:.4f} <= resistance={resistance:.4f} "
            f"(entry hanya di dekat boundary, bukan tengah range)"
        )
    elif menyentuh_support and not sweep_buy:
        # low < support tapi sweep gagal: bisa depth kurang atau close tidak reclaim
        kedalaman_aktual = support - low
        if kedalaman_aktual < min_sweep_depth:
            ket = (
                f"Range Reversal: sweep support GAGAL — depth={kedalaman_aktual:.4f} "
                f"< min={min_sweep_depth:.4f} ({RANGE_REVERSAL_MIN_SWEEP_DEPTH_ATR}*ATR={atr:.4f})"
            )
        else:
            # depth cukup tapi close di bawah support (breakdown, bukan reversal)
            ket = (
                f"Range Reversal: sweep support ada (depth={kedalaman_aktual:.4f}), "
                f"TAPI close={close:.4f} < support={support:.4f} — ini breakdown, bukan reversal"
            )
    elif menyentuh_resistance and not sweep_sell:
        kedalaman_aktual = high - resistance
        if kedalaman_aktual < min_sweep_depth:
            ket = (
                f"Range Reversal: sweep resistance GAGAL — depth={kedalaman_aktual:.4f} "
                f"< min={min_sweep_depth:.4f} ({RANGE_REVERSAL_MIN_SWEEP_DEPTH_ATR}*ATR={atr:.4f})"
            )
        else:
            ket = (
                f"Range Reversal: sweep resistance ada (depth={kedalaman_aktual:.4f}), "
                f"TAPI close={close:.4f} > resistance={resistance:.4f} — ini breakout, bukan reversal"
            )
    else:
        ket = (
            f"Range Reversal: NETRAL — tidak ada sweep valid di boundary mana pun "
            f"(low={low:.4f}, high={high:.4f}, support={support:.4f}, resistance={resistance:.4f})"
        )

    return {
        "terpenuhi"          : False,
        "arah"               : "NETRAL",
        "boundary_referensi" : None,
        "sweep_terpenuhi"    : False,
        "sweep_depth"        : None,
        "rejection_terpenuhi": False,
        "rejection_detail"   : {
            "terpenuhi"      : False,
            "wick_length"    : None,
            "close_position" : None,
            "keterangan"     : "Tidak dievaluasi — sweep tidak terpenuhi",
        },
        "invalidation_level" : None,
        "keterangan"         : ket,
    }
