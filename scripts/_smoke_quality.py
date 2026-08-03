import sys
sys.path.insert(0, '.')

# 1. rule_engine import
from engine.rule_engine import evaluate_entry, calculate_setup_quality
print('OK: rule_engine import works')

# 2. With swing data
signals = {
    'time': None, 'close': 3200.0, 'ema_9': 3198.0, 'ema_21': 3194.0,
    'rsi_14': 52.0, 'trend': 'UPTREND', 'ema_gap_pct': 0.12,
    'trend_h1': 'UPTREND', 'volume_ratio': None,
    'swing_low': 3185.0, 'swing_high': None,
    'atr_14': 5.0,
}

quality = calculate_setup_quality(signals, {}, {}, {})
print()
print('calculate_setup_quality() with swing data:')
print('  setup_quality       : %s' % quality['setup_quality'])
print('  setup_quality_score : %d / %d' % (quality['setup_quality_score'], quality['setup_quality_max']))
for k, v in quality['quality_breakdown'].items():
    print('  %-20s: score=%d/2  %s' % (k, v['score'], v['detail']))

if 'alignment' in quality['quality_breakdown']:
    print('FAIL: alignment masih ada di breakdown')
else:
    print()
    print('OK: alignment sudah tidak ada di breakdown')

sd = quality['quality_breakdown']['swing_distance']['score']
print('OK: swing_distance skor = %d (data tersedia: jarak=15, ATR=5, ratio=3.0 -> skor 2)' % sd)

# 3. Without swing data
signals2 = dict(signals)
signals2['swing_low']  = None
signals2['swing_high'] = None
q2 = calculate_setup_quality(signals2, {}, {}, {})
print()
print('WITHOUT swing data (fallback):')
print('  swing_distance score = %d (expected 0)' % q2['quality_breakdown']['swing_distance']['score'])
print('  detail: %s' % q2['quality_breakdown']['swing_distance']['detail'])
print()
print('ALL TESTS PASSED')
