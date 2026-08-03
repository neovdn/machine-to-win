import re

with open('engine/rule_engine.py', encoding='utf-8') as f:
    content = f.read()

# Find boundaries of calculate_setup_quality function
start_marker = '# =============================================================================\n# HELPER INTERNAL'
end_marker = '\n\n\n# =============================================================================\n# HELPER INTERNAL — Validasi Input'

start_idx = content.find(start_marker)
end_idx   = content.find(end_marker)

print(f'start_idx: {start_idx}')
print(f'end_idx: {end_idx}')
print()
print('--- First 200 chars of target section ---')
print(repr(content[start_idx:start_idx+200]))
print()
print('--- Last 100 chars before end marker ---')
print(repr(content[end_idx-100:end_idx]))
