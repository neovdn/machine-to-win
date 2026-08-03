content = open('scripts/analyze_quality_score_phase43.py', encoding='utf-8').read()

old_labels = '''COMPONENT_LABELS = {
    "score_ema_gap"        : "EMA Gap Strength",
    
    "score_swing_distance" : "Swing Distance",
}'''

new_labels = '''COMPONENT_LABELS = {
    "score_ema_gap"        : "EMA Gap Strength",
    "score_rsi_zone"       : "RSI Zone",
    "score_swing_distance" : "Swing Distance",
}'''

if old_labels in content:
    content = content.replace(old_labels, new_labels)
    print('OK: COMPONENT_LABELS fixed')
else:
    print('Pattern not found')
    idx = content.find('COMPONENT_LABELS')
    print(repr(content[idx:idx+300]))

with open('scripts/analyze_quality_score_phase43.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
