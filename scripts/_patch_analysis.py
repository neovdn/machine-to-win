content = open('scripts/analyze_quality_score_phase43.py', encoding='utf-8').read()

old_cols = '''COMPONENT_COLS = [
    "score_ema_gap",
    "score_alignment",
    "score_rsi_zone",
    "score_swing_distance",
]

COMPONENT_LABELS = {
    "score_ema_gap"        : "EMA Gap Strength",
    "score_alignment"      : "H1-M5 Alignment",
    "score_rsi_zone"       : "RSI Zone",
    "score_swing_distance" : "Swing Distance",
}'''

new_cols = '''COMPONENT_COLS = [
    "score_ema_gap",
    "score_rsi_zone",
    "score_swing_distance",
]

COMPONENT_LABELS = {
    "score_ema_gap"        : "EMA Gap Strength",
    "score_rsi_zone"       : "RSI Zone",
    "score_swing_distance" : "Swing Distance",
}
# NOTE: score_alignment dihapus di Fase 4.3 (tautologi struktural).
# score_swing_distance sekarang terisi data melalui perbaikan pipeline.'''

if old_cols in content:
    content = content.replace(old_cols, new_cols)
    print('OK: COMPONENT_COLS updated')
else:
    print('Not found. Searching...')
    idx = content.find('COMPONENT_COLS')
    print(repr(content[idx:idx+400]))

# Also update required_cols check
old_req = '    required_cols = ["setup_quality", "setup_quality_score"] + COMPONENT_COLS'
new_req = '    required_cols = ["setup_quality", "setup_quality_score"] + COMPONENT_COLS'
# That is the same -- check for score_alignment in required_cols
if '"score_alignment"' in content:
    content = content.replace('"score_alignment"', '')
    print('OK: residual score_alignment reference removed')

with open('scripts/analyze_quality_score_phase43.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('File written.')
