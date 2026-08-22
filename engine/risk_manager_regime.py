"""
engine/risk_manager_regime.py
==============================
Modul Fase 19: Regime-Aware Risk Management.

TUJUAN:
    Menjadi satu pintu kalkulasi SL/TP untuk ketiga strategi regime-based
    (Range Reversal, Breakout Retest v2, Trend Following v2). Modul ini
    menerjemahkan level referensi mentah (invalidation_level / invalidation_level_sl)
    dari strategi manapun menjadi kalkulasi SL/TP/RRR yang lengkap, dengan
    TETAP memakai infrastruktur calculate_sl_tp() yang sudah matang di
    engine/risk_manager.py — bukan menulis ulang.

KARAKTER MODUL (PENERJEMAH, BUKAN PENGGANTI):
    Modul ini TIDAK menghitung SL/TP dari nol. Ia hanya:
    1. Membaca field level referensi mentah dari hasil evaluate_*() strategi.
    2. Meneruskan level tersebut ke calculate_sl_tp() via sl_source="EXTERNAL_LEVEL".
    3. Menerapkan logika sanity check khusus per-strategi (khususnya cap RRR
       untuk Range Reversal di Langkah 3).
    4. Menyusun dict return yang lengkap untuk caller (nantinya Fase 21).

CATATAN FIELD MAPPING TIDAK SERAGAM:
    Ketiga strategi menghasilkan field nama berbeda untuk level referensi SL mentah:
    - Range Reversal  : "invalidation_level"    (field berbeda dari dua lainnya)
    - Breakout Retest : "invalidation_level_sl"
    - Trend Following : "invalidation_level_sl"
    Ini BUKAN bug — field name sudah dikunci di modul strategi masing-masing yang
    sudah SELESAI. Perbedaan ini ditangani via STRATEGY_LEVEL_FIELD_MAP di bawah.
    JANGAN mengubah nama field di modul strategi yang sudah selesai.

CATATAN CAP RRR HANYA UNTUK RANGE_REVERSAL:
    Langkah 3 (sanity check ceiling TP terhadap boundary seberang range) HANYA
    berlaku untuk RANGE_REVERSAL. Untuk BREAKOUT_RETEST dan TREND_FOLLOWING,
    langkah ini dilewati sepenuhnya — kedua strategi tersebut tidak punya
    boundary seberang yang jelas sebagai target alami.

INTEGRASI:
    Modul ini BELUM diintegrasikan ke web/app.py atau engine/backtester.py.
    Integrasi penuh adalah Fase 21.

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.
"""

import pandas as pd

from engine.risk_manager import calculate_sl_tp


# =============================================================================
# MAPPING FIELD LEVEL REFERENSI PER STRATEGI
# =============================================================================

STRATEGY_LEVEL_FIELD_MAP = {
    "RANGE_REVERSAL"  : "invalidation_level",
    "BREAKOUT_RETEST"  : "invalidation_level_sl",
    "TREND_FOLLOWING"  : "invalidation_level_sl",
}
# CATATAN: Nama field tidak seragam antar strategi — ini disengaja (lihat
# docstring modul). Jangan ubah mapping ini tanpa mengubah modul strategi
# yang sudah selesai (melanggar prinsip proyek).


# =============================================================================
# KONSTANTA
# =============================================================================

REGIME_RISK_MIN_RRR_AFTER_CAP = 1.0
# Minimum RRR yang masih dianggap layak setelah TP di-cap ke boundary seberang
# range (hanya berlaku untuk RANGE_REVERSAL).
# BELUM DIKALIBRASI — nilai awal struktural, bukan hasil backtest.


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def calculate_regime_sl_tp(
    df_m5          : pd.DataFrame,
    entry          : float,
    arah           : str,
    strategy_name  : str,
    strategy_result: dict,
    zone           : dict | None = None,
    rrr_min        : float | None = None,
    tick_info      : dict | None = None,
) -> dict:
    """
    Kalkulasi SL/TP regime-aware untuk ketiga strategi (satu pintu).

    Parameter:
        df_m5           : DataFrame M5 sudah di-slice caller sampai idx_m5
                          (pola: df_m5.iloc[:idx_m5+1]). Harus sudah punya kolom
                          OHLC + atr_14.
        entry           : float — harga entry kandidat (biasanya close candle idx_m5).
        arah            : str   — "BUY" | "SELL".
        strategy_name   : str   — "RANGE_REVERSAL" | "BREAKOUT_RETEST" | "TREND_FOLLOWING".
        strategy_result : dict  — hasil evaluate_*() dari strategi terkait.
                          Harus berisi field level referensi SL mentah sesuai
                          STRATEGY_LEVEL_FIELD_MAP.
        zone            : dict | None — boundary range dari detect_market_regime().
                          HANYA dibutuhkan untuk RANGE_REVERSAL (Langkah 3 cap TP).
                          Format minimal: {"resistance": float, "support": float}.
                          None -> Langkah 3 dilewati secara graceful (tp_capped=False).
        rrr_min         : float | None — override RRR minimum. None -> pakai default
                          dari RISK_PROFILES["scalp_m5"] di calculate_sl_tp().
        tick_info       : dict | None — spread info untuk spread-aware entry.
                          Diteruskan langsung ke calculate_sl_tp().

    Return:
        dict berisi:
            "valid"             : bool   — True jika kalkulasi berhasil dan RRR layak.
            "skip_reason"       : str | None — penjelasan jika valid=False.
            # Semua field dari calculate_sl_tp() jika valid=True:
            # entry, sl, tp, rrr, jarak_sl, jarak_tp, sl_method, sl_swing_clamped,
            # clamp_reason, atr_value, sl_atr_level, sl_swing_raw, sl_swing_level,
            # spread, rrr_after_spread, pesan, entry_type.
            "tp_capped"         : bool   — True jika TP di-cap ke boundary seberang range.
            "tp_original"       : float | None — nilai TP sebelum di-cap (jika tp_capped=True).
            "keterangan_regime" : str    — ringkasan audit regime risk management.
    """

    # ── Langkah 1: Ambil level referensi mentah via STRATEGY_LEVEL_FIELD_MAP ──

    # Validasi strategy_name
    if strategy_name not in STRATEGY_LEVEL_FIELD_MAP:
        return {
            "valid"             : False,
            "skip_reason"       : f"strategy_name tidak dikenal: '{strategy_name}'. "
                                   f"Pilihan valid: {list(STRATEGY_LEVEL_FIELD_MAP.keys())}",
            "tp_capped"         : False,
            "tp_original"       : None,
            "keterangan_regime" : f"SKIP: strategy_name tidak dikenal ({strategy_name})",
        }

    field_name = STRATEGY_LEVEL_FIELD_MAP[strategy_name]

    # Validasi field ada di strategy_result
    if field_name not in strategy_result:
        return {
            "valid"             : False,
            "skip_reason"       : f"Field '{field_name}' tidak ditemukan di strategy_result "
                                   f"untuk strategi '{strategy_name}'. "
                                   f"Pastikan evaluate_*() dipanggil dengan benar.",
            "tp_capped"         : False,
            "tp_original"       : None,
            "keterangan_regime" : f"SKIP: field '{field_name}' hilang di strategy_result",
        }

    raw_level = strategy_result[field_name]

    # Validasi nilai level tidak None
    if raw_level is None:
        return {
            "valid"             : False,
            "skip_reason"       : f"Field '{field_name}' bernilai None di strategy_result "
                                   f"untuk strategi '{strategy_name}'. "
                                   f"Strategi mungkin tidak terpenuhi (terpenuhi=False).",
            "tp_capped"         : False,
            "tp_original"       : None,
            "keterangan_regime" : f"SKIP: '{field_name}' = None (strategi tidak terpenuhi)",
        }

    raw_level = float(raw_level)

    # ── Langkah 2: Panggil calculate_sl_tp() dengan sl_source="EXTERNAL_LEVEL" ─

    sl_tp_result = calculate_sl_tp(
        df              = df_m5,
        entry           = entry,
        arah            = arah,
        rrr_min         = rrr_min,
        tick_info       = tick_info,
        sl_source       = "EXTERNAL_LEVEL",
        external_level  = raw_level,
    )

    # calculate_sl_tp() selalu return valid=True jika input valid.
    # Jika ada exception, ia propagate ke caller — sesuai desain.

    # Ekstrak nilai-nilai kunci untuk Langkah 3
    tp         = sl_tp_result["tp"]
    jarak_sl   = sl_tp_result["jarak_sl"]

    # Inisialisasi variabel output cap
    tp_capped   = False
    tp_original = None

    # ── Langkah 3: Sanity check RRR khusus RANGE_REVERSAL ─────────────────────
    # Cap TP ke boundary seberang range jika TP melampaui target alami.
    # HANYA berlaku untuk RANGE_REVERSAL — strategi lain dilewati sepenuhnya.

    if strategy_name == "RANGE_REVERSAL" and zone is not None:
        resistance = zone.get("resistance")
        support    = zone.get("support")

        if resistance is not None and support is not None:
            resistance = float(resistance)
            support    = float(support)

            if arah == "BUY":
                realistic_ceiling = resistance
                overshoot = tp > realistic_ceiling
            else:  # SELL
                realistic_ceiling = support
                overshoot = tp < realistic_ceiling

            if overshoot:
                tp_original    = tp
                tp_capped      = True
                tp             = realistic_ceiling
                jarak_tp_baru  = abs(tp - sl_tp_result["entry"])

                if jarak_sl > 0:
                    achievable_rrr = jarak_tp_baru / jarak_sl
                else:
                    achievable_rrr = 0.0

                if achievable_rrr < REGIME_RISK_MIN_RRR_AFTER_CAP:
                    return {
                        "valid"             : False,
                        "skip_reason"       : (
                            f"RRR tidak layak setelah TP dibatasi ke boundary seberang range "
                            f"({achievable_rrr:.2f} < {REGIME_RISK_MIN_RRR_AFTER_CAP})"
                        ),
                        "tp_capped"         : True,
                        "tp_original"       : tp_original,
                        "keterangan_regime" : (
                            f"SKIP: TP di-cap ke {realistic_ceiling:.2f} "
                            f"(boundary {'resistance' if arah == 'BUY' else 'support'}), "
                            f"achievable_rrr={achievable_rrr:.2f} < "
                            f"REGIME_RISK_MIN_RRR_AFTER_CAP={REGIME_RISK_MIN_RRR_AFTER_CAP}"
                        ),
                    }

                # RRR setelah cap masih layak — perbarui nilai
                jarak_tp_final = jarak_tp_baru
                rrr_final      = round(achievable_rrr, 2)
                keterangan_cap = (
                    f"TP di-cap ke boundary {'resistance' if arah == 'BUY' else 'support'} "
                    f"= {realistic_ceiling:.2f} (dari {tp_original:.2f}), "
                    f"achievable_rrr={rrr_final:.2f}"
                )
            else:
                # TP tidak melampaui ceiling — tidak perlu cap
                jarak_tp_final = sl_tp_result["jarak_tp"]
                rrr_final      = sl_tp_result["rrr"]
                keterangan_cap = "TP dalam batas boundary seberang range, tidak di-cap"
        else:
            # zone ada tapi resistance/support None — lewati cap secara graceful
            jarak_tp_final = sl_tp_result["jarak_tp"]
            rrr_final      = sl_tp_result["rrr"]
            keterangan_cap = "zone.resistance atau zone.support None — cap dilewati"
    else:
        # Bukan RANGE_REVERSAL, atau zone=None — lewati Langkah 3 sepenuhnya
        jarak_tp_final = sl_tp_result["jarak_tp"]
        rrr_final      = sl_tp_result["rrr"]
        if strategy_name == "RANGE_REVERSAL":
            keterangan_cap = "zone=None — Langkah 3 dilewati (tidak crash)"
        else:
            keterangan_cap = f"Cap TP tidak berlaku untuk strategi {strategy_name}"

    # ── Langkah 4: Susun return dict ──────────────────────────────────────────

    keterangan_regime = (
        f"calculate_regime_sl_tp: strategy={strategy_name}, arah={arah}, "
        f"raw_level={raw_level:.2f}, entry={sl_tp_result['entry']:.2f}, "
        f"sl={sl_tp_result['sl']:.2f}, tp={round(tp, 2):.2f}, "
        f"rrr={rrr_final:.2f}, sl_method={sl_tp_result['sl_method']}. "
        f"{keterangan_cap}"
    )

    result = {
        "valid"             : True,
        "skip_reason"       : None,
        # ── Semua field asli dari calculate_sl_tp() ──
        "entry"             : sl_tp_result["entry"],
        "entry_type"        : sl_tp_result["entry_type"],
        "sl"                : sl_tp_result["sl"],
        "tp"                : round(tp, 2),
        "rrr"               : rrr_final,
        "jarak_sl"          : sl_tp_result["jarak_sl"],
        "jarak_tp"          : round(jarak_tp_final, 2),
        "sl_method"         : sl_tp_result["sl_method"],
        "sl_swing_clamped"  : sl_tp_result["sl_swing_clamped"],
        "clamp_reason"      : sl_tp_result["clamp_reason"],
        "atr_value"         : sl_tp_result["atr_value"],
        "sl_atr_level"      : sl_tp_result["sl_atr_level"],
        "sl_swing_raw"      : sl_tp_result["sl_swing_raw"],
        "sl_swing_level"    : sl_tp_result["sl_swing_level"],
        "spread"            : sl_tp_result["spread"],
        "rrr_after_spread"  : sl_tp_result["rrr_after_spread"],
        "pesan"             : sl_tp_result["pesan"],
        # ── Field tambahan khusus regime risk management ──
        "tp_capped"         : tp_capped,
        "tp_original"       : round(tp_original, 2) if tp_original is not None else None,
        "keterangan_regime" : keterangan_regime,
    }

    return result
