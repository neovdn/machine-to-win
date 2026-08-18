"""
engine/location_filter.py
=========================
Modul untuk mengevaluasi kualitas lokasi entry dan keselarasan dengan bias H1.

TUJUAN:
    Menilai seberapa bagus lokasi entry yang dihasilkan oleh strategi, serta
    keselarasannya dengan bias timeframe H1. Modul ini bertindak sebagai layer
    scoring tambahan SEBELUM risk management.

KEPUTUSAN SOFT SCORING vs HARD VETO:
    Modul ini mengadopsi pendekatan SOFT SCORING, BUKAN HARD VETO.
    Alasannya: ketiga strategi (Range Reversal, Breakout Retest, Trend Following)
    SUDAH memiliki hard gate lokasi di dalam logika mereka masing-masing.
    (Misalnya, syarat pullback distance, depth of sweep, dsb). Jika modul ini
    menambahkan veto lagi, itu akan menduplikasi logika. Oleh karena itu, modul
    ini murni memberikan SKOR KUALITAS tambahan. Keputusan (field 'terpenuhi')
    dari strategi tidak pernah dibatalkan.

KARAKTER MODUL:
    - Tidak ada akses DataFrame.
    - Murni fungsi dari dict/scalar ke dict (read-only pada input).
    - Tidak memiliki efek samping.

CATATAN:
    - BELUM DIKALIBRASI: Threshold skoring (baik lokasi maupun komponen H1)
      serta arah hipotesis (terutama asumsi untuk BREAKOUT_RETEST) adalah asumsi
      awal yang WAJIB diuji dan dikalibrasi lewat backtest Fase 21/22. Arah
      hipotesis mungkin terbalik pada praktiknya.
"""

import math

# =============================================================================
# KONSTANTA LOKASI (BELUM DIKALIBRASI)
# =============================================================================

# RANGE_REVERSAL (RR)
LOCATION_RR_DEPTH_STRONG_ATR   = 0.3
LOCATION_RR_DEPTH_MODERATE_ATR = 0.15

# BREAKOUT_RETEST (BR)
# Hipotesis arah: retest cepat mengindikasikan momentum kuat (bisa salah, perlu diuji)
LOCATION_BR_FRESH_CANDLES    = 3
LOCATION_BR_MODERATE_CANDLES = 8

# TREND_FOLLOWING (TF)
LOCATION_TF_PULLBACK_STRONG_ATR   = 0.3
LOCATION_TF_PULLBACK_MODERATE_ATR = 0.7

# KONSTANTA SETUP QUALITY (POLA DARI calculate_setup_quality)
STRONG_PCT   = 0.875
MODERATE_PCT = 0.500

def _kosong_lokasi(keterangan: str, label: str = "Location", metric_used: str = "none") -> dict:
    return {
        "score"        : 0,
        "max"          : 2,
        "label"        : label,
        "detail"       : keterangan,
        "metric_used"  : metric_used,
        "metric_value" : None,
    }

def calculate_location_score(strategy_name: str, strategy_result: dict, atr: float) -> dict:
    if not isinstance(strategy_result, dict):
        return _kosong_lokasi("strategy_result bukan dict")
        
    if not strategy_result.get("terpenuhi"):
        return _kosong_lokasi("Strategi tidak terpenuhi (terpenuhi=False)")
        
    if atr is None or atr <= 0:
        return _kosong_lokasi("ATR tidak valid atau nol")

    if strategy_name == "RANGE_REVERSAL":
        val = strategy_result.get("sweep_depth")
        if val is None:
            return _kosong_lokasi("Field 'sweep_depth' tidak ditemukan", label="RR Location", metric_used="sweep_depth")
            
        try:
            depth = float(val)
        except (TypeError, ValueError):
            return _kosong_lokasi("Field 'sweep_depth' tidak valid", label="RR Location", metric_used="sweep_depth")
            
        ratio = depth / atr
        if ratio >= LOCATION_RR_DEPTH_STRONG_ATR:
            score = 2
            ket = "sweep dalam, indikasi liquidity grab jelas"
        elif ratio >= LOCATION_RR_DEPTH_MODERATE_ATR:
            score = 1
            ket = "sweep cukup"
        else:
            score = 0
            ket = "sweep minimal, dekat batas ambang"
            
        return {
            "score"        : score,
            "max"          : 2,
            "label"        : "RR Location",
            "detail"       : ket,
            "metric_used"  : "sweep_depth",
            "metric_value" : depth,
        }
        
    elif strategy_name == "BREAKOUT_RETEST":
        val = strategy_result.get("candles_since_touch")
        if val is None:
            return _kosong_lokasi("Field 'candles_since_touch' tidak ditemukan", label="BR Location", metric_used="candles_since_touch")
            
        try:
            candles = int(val)
        except (TypeError, ValueError):
            return _kosong_lokasi("Field 'candles_since_touch' tidak valid", label="BR Location", metric_used="candles_since_touch")
            
        if candles <= LOCATION_BR_FRESH_CANDLES:
            score = 2
            ket = "retest cepat, momentum breakout masih kuat (hipotesis)"
        elif candles <= LOCATION_BR_MODERATE_CANDLES:
            score = 1
            ket = "retest sedang"
        else:
            score = 0
            ket = "retest lambat, momentum berpotensi melemah (hipotesis)"
            
        return {
            "score"        : score,
            "max"          : 2,
            "label"        : "BR Location",
            "detail"       : ket,
            "metric_used"  : "candles_since_touch",
            "metric_value" : candles,
        }
        
    elif strategy_name == "TREND_FOLLOWING":
        val = strategy_result.get("pullback_distance")
        if val is None:
            return _kosong_lokasi("Field 'pullback_distance' tidak ditemukan", label="TF Location", metric_used="pullback_distance")
            
        try:
            distance = float(val)
        except (TypeError, ValueError):
            return _kosong_lokasi("Field 'pullback_distance' tidak valid", label="TF Location", metric_used="pullback_distance")
            
        ratio = distance / atr
        if ratio <= LOCATION_TF_PULLBACK_STRONG_ATR:
            score = 2
            ket = "pullback sangat dekat swing, lokasi optimal"
        elif ratio <= LOCATION_TF_PULLBACK_MODERATE_ATR:
            score = 1
            ket = "pullback cukup dekat"
        else:
            score = 0
            ket = "pullback mendekati batas toleransi"
            
        return {
            "score"        : score,
            "max"          : 2,
            "label"        : "TF Location",
            "detail"       : ket,
            "metric_used"  : "pullback_distance",
            "metric_value" : distance,
        }
        
    else:
        return _kosong_lokasi(f"strategy_name '{strategy_name}' tidak dikenal")


def calculate_h1_confluence_score(strategy_result: dict, h1_context: dict) -> dict:
    def _kosong_h1(keterangan: str, h1_bias=None) -> dict:
        return {
            "score"      : 0,
            "max"        : 2,
            "label"      : "H1 Confluence",
            "detail"     : keterangan,
            "h1_bias"    : h1_bias,
            "arah_cocok" : False,
        }

    if not isinstance(strategy_result, dict) or not isinstance(h1_context, dict):
        return _kosong_h1("Input bukan dictionary")
        
    arah_strat = strategy_result.get("arah")
    if arah_strat not in ("BUY", "SELL"):
        return _kosong_h1("Arah strategi tidak valid atau bukan BUY/SELL")
        
    bias_h1 = h1_context.get("bias")
    if not bias_h1:
        return _kosong_h1("Bias H1 tidak ditemukan")

    arah_cocok = (arah_strat == "BUY" and bias_h1 == "BULLISH") or (arah_strat == "SELL" and bias_h1 == "BEARISH")
    
    if not arah_cocok:
        return {
            "score"      : 0,
            "max"        : 2,
            "label"      : "H1 Confluence",
            "detail"     : "Arah strategi tidak sejalan dengan bias H1 atau H1 NEUTRAL",
            "h1_bias"    : bias_h1,
            "arah_cocok" : False,
        }
        
    strength = h1_context.get("strength_zone")
    if strength == "STRONG":
        score = 2
        ket = "Arah cocok dan strength_zone H1 STRONG"
    elif strength == "MODERATE":
        score = 1
        ket = "Arah cocok dan strength_zone H1 MODERATE"
    elif strength == "WEAK":
        score = 0
        ket = "Arah cocok tapi strength_zone H1 WEAK"
    else:
        score = 0
        ket = f"Arah cocok tapi strength_zone H1 tidak dikenal ({strength})"
        
    return {
        "score"      : score,
        "max"        : 2,
        "label"      : "H1 Confluence",
        "detail"     : ket,
        "h1_bias"    : bias_h1,
        "arah_cocok" : True,
    }


def calculate_confluence_summary(
    strategy_name: str, strategy_result: dict, atr: float, h1_context: dict
) -> dict:
    
    loc = calculate_location_score(strategy_name, strategy_result, atr)
    h1 = calculate_h1_confluence_score(strategy_result, h1_context)
    
    total_score = loc["score"] + h1["score"]
    max_score = 4
    
    threshold_strong = math.ceil(STRONG_PCT * max_score)
    threshold_moderate = math.ceil(MODERATE_PCT * max_score)
    
    if total_score >= threshold_strong:
        q_label = "STRONG"
        ket = "Setup memiliki konfluensi LOKASI dan H1 yang sangat kuat."
    elif total_score >= threshold_moderate:
        q_label = "MODERATE"
        ket = "Setup memiliki konfluensi LOKASI dan H1 menengah."
    else:
        q_label = "WEAK"
        ket = "Setup memiliki konfluensi LOKASI dan H1 yang lemah."
        
    return {
        "total_score"  : total_score,
        "max_score"    : max_score,
        "quality_label": q_label,
        "breakdown"    : {
            "location"     : loc,
            "h1_confluence": h1,
        },
        "keterangan"   : ket,
    }
