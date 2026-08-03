import sys

with open('engine/rule_engine.py', encoding='utf-8') as f:
    content = f.read()

start_marker = '# =============================================================================\n# HELPER INTERNAL'
end_marker   = '\n\n\n# =============================================================================\n# HELPER INTERNAL \u2014 Validasi Input'

start_idx = content.find(start_marker)
end_idx   = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print('MARKERS NOT FOUND')
    sys.exit(1)

old_section = content[start_idx:end_idx]
print(f'Replacing {len(old_section)} chars ({old_section.count(chr(10))} lines) starting at {start_idx}')

new_section = '''# =============================================================================
# HELPER INTERNAL \u2014 Setup Quality Scoring (Auditable 0-6 Poin)
# =============================================================================
# Fase 4.3 (2026-08-03): alignment dihapus (tautologi struktural),
# swing_distance diperbaiki (data kini di-feed dari pipeline sebelum evaluate_entry).
# Skor maks: 8 -> 6. Threshold: STRONG >=5, MODERATE >=3, WEAK <3.

def calculate_setup_quality(signals: dict, c_h1: dict, c_m5: dict, c_rsi: dict) -> dict:
    """
    Hitung setup_quality ("STRONG", "MODERATE", "WEAK") berbasis point-based scoring (0-6 Poin).

    3 Komponen Scoring (setelah Fase 4.3):
        1. EMA Gap Strength  (0-2 pts): Kekuatan trend M5 dari gap EMA
        2. RSI Zone          (0-2 pts): RSI di zona netral vs ekstrem
        3. Swing Distance    (0-2 pts): Jarak harga ke swing terdekat vs ATR
                                        (data via signals["swing_low"]/["swing_high"]
                                        yang di-feed pipeline sebelum evaluate_entry)

    CATATAN ARSITEKTUR (Fase 4.3):
        Komponen alignment H1-M5 dihapus -- terbukti tautologi secara struktural:
        evaluate_entry() mensyaratkan H1 dan M5 searah (MINIMUM_CONDITIONS_MET=2)
        sebelum trade bisa terjadi, sehingga alignment SELALU 2 untuk setiap trade.
        Zero variance, tidak bisa membedakan setup apapun. Bukan bug data, tapi
        konsekuensi matematis dari arsitektur filter entry.

    Threshold (proporsional dari skema 0-8 lama, max 6):
        STRONG   >= 5  (dari >=6/8=75% -> 75%*6=4.5 -> dibulatkan ke 5)
        MODERATE >= 3  (dari >=4/8=50% -> 50%*6=3.0)
        WEAK      < 3
    """
    breakdown   = {}
    total_score = 0

    # 1. EMA Gap Strength (0-2)
    ema_gap_pct = abs(signals.get("ema_gap_pct", 0.0))
    if ema_gap_pct >= 0.15:
        score_gap  = 2
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Kuat >= 0.15%)"
    elif ema_gap_pct >= 0.08:
        score_gap  = 1
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Sedang 0.08-0.15%)"
    else:
        score_gap  = 0
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Tipis < 0.08%)"
    total_score += score_gap
    breakdown["ema_gap"] = {
        "score": score_gap, "max": 2,
        "label": "Kekuatan Trend M5 (EMA Gap)", "detail": detail_gap,
    }

    # 2. RSI Zone (0-2)
    rsi = signals.get("rsi_14", 50.0)
    if 40.0 <= rsi <= 60.0:
        score_rsi  = 2
        detail_rsi = f"RSI {rsi:.1f} (Zona Optimum Netral 40-60)"
    elif (30.0 <= rsi < 40.0) or (60.0 < rsi <= 70.0):
        score_rsi  = 1
        detail_rsi = f"RSI {rsi:.1f} (Zona Waspada 30-40 / 60-70)"
    else:
        score_rsi  = 0
        detail_rsi = f"RSI {rsi:.1f} (Zona Ekstrem <30 / >70)"
    total_score += score_rsi
    breakdown["rsi_zone"] = {
        "score": score_rsi, "max": 2,
        "label": "Zona RSI M5", "detail": detail_rsi,
    }

    # 3. Swing Distance (0-2)
    # signals["swing_low"] / ["swing_high"] di-feed oleh caller (backtester & app.py)
    # sebelum memanggil evaluate_entry() -- diperbaiki di Fase 4.3.
    trend_h1    = signals.get("trend_h1", "SIDEWAYS")
    trend_m5    = signals.get("trend",    "SIDEWAYS")
    close_price = signals.get("close", 0.0)
    atr         = signals.get("atr_14", 1.5)
    sw_low      = signals.get("swing_low")
    sw_high     = signals.get("swing_high")

    swing_dist = None
    if trend_h1 == "UPTREND" or trend_m5 == "UPTREND":
        if sw_low is not None:
            swing_dist = abs(close_price - sw_low)
    elif trend_h1 == "DOWNTREND" or trend_m5 == "DOWNTREND":
        if sw_high is not None:
            swing_dist = abs(sw_high - close_price)

    # Fallback: pakai swing yang tersedia jika arah tidak terdeteksi
    if swing_dist is None:
        if sw_low is not None:
            swing_dist = abs(close_price - sw_low)
        elif sw_high is not None:
            swing_dist = abs(sw_high - close_price)

    if swing_dist is not None and atr > 0:
        atr_ratio = swing_dist / atr
        if atr_ratio >= 1.5:
            score_swing  = 2
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR -- Luas)"
        elif atr_ratio >= 0.8:
            score_swing  = 1
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR -- Cukup)"
        else:
            score_swing  = 0
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR -- Sempit)"
    else:
        score_swing  = 0
        detail_swing = "Swing Tidak Ditemukan / Data Tidak Tersedia"
    total_score += score_swing
    breakdown["swing_distance"] = {
        "score": score_swing, "max": 2,
        "label": "Jarak ke Swing Structure", "detail": detail_swing,
    }

    # Penentuan Label Quality (max=6, threshold proporsional dari skema 0-8 lama)
    if total_score >= 5:
        quality_label = "STRONG"
    elif total_score >= 3:
        quality_label = "MODERATE"
    else:
        quality_label = "WEAK"

    return {
        "setup_quality"       : quality_label,
        "setup_quality_score" : total_score,
        "setup_quality_max"   : 6,
        "quality_breakdown"   : breakdown,
    }'''

new_content = content[:start_idx] + new_section + content[end_idx:]

with open('engine/rule_engine.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f'Done. File rewritten ({len(new_content)} chars).')
print('Verifying...')
with open('engine/rule_engine.py', encoding='utf-8') as f:
    verify = f.read()
if 'alignment' not in verify[verify.find('def calculate_setup_quality'):verify.find('def _validate_signals')]:
    print('OK: alignment komponen sudah dihapus dari calculate_setup_quality')
else:
    print('WARN: masih ada teks alignment di dalam calculate_setup_quality')
if 'setup_quality_max\"   : 6' in verify:
    print('OK: max score = 6')
if 'total_score >= 5' in verify:
    print('OK: STRONG threshold = 5')
if 'total_score >= 3' in verify:
    print('OK: MODERATE threshold = 3')
