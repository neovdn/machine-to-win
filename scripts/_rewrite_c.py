import re

content = open('scripts/analyze_quality_score_phase43.py', encoding='utf-8').read()

# Find the function boundaries
fn_start = content.find('\ndef analyze_inter_component_correlation(')
fn_end_search = content.find('\ndef analyze_score_vs_total(')

if fn_start == -1 or fn_end_search == -1:
    print('Boundaries not found')
    print('fn_start:', fn_start, 'fn_end:', fn_end_search)
else:
    old_fn = content[fn_start:fn_end_search]
    print('Found function (%d chars)' % len(old_fn))

    new_fn = '''
def analyze_inter_component_correlation(df):
    print()
    print("=" * 70)
    print("  BAGIAN C: KORELASI ANTAR KOMPONEN (3-KOMPONEN PASCA PERBAIKAN)")
    print("=" * 70)
    print("  Catatan: alignment dihapus (tautologi). swing_distance sekarang terisi data.")

    comp_df = df[COMPONENT_COLS].dropna().astype(float)
    if comp_df.empty:
        print("  Data komponen kosong!")
        return

    pearson_corr  = comp_df.corr(method="pearson")
    spearman_corr = comp_df.corr(method="spearman")

    print()
    print("  Pearson Correlation Matrix:")
    header = "  " + " " * 28 + "".join("%-17s" % COMPONENT_LABELS[c] for c in COMPONENT_COLS)
    print(header)
    for r in COMPONENT_COLS:
        row = "  %-28s" % COMPONENT_LABELS[r]
        for c in COMPONENT_COLS:
            val = pearson_corr.loc[r, c]
            row += "%+17.4f" % val
        print(row)

    print()
    print("  Spearman Correlation Matrix:")
    print(header)
    for r in COMPONENT_COLS:
        row = "  %-28s" % COMPONENT_LABELS[r]
        for c in COMPONENT_COLS:
            val = spearman_corr.loc[r, c]
            row += "%+17.4f" % val
        print(row)

    pairs = [
        ("score_ema_gap",   "score_rsi_zone",       "ema_gap vs rsi_zone"),
        ("score_ema_gap",   "score_swing_distance",  "ema_gap vs swing_distance"),
        ("score_rsi_zone",  "score_swing_distance",  "rsi_zone vs swing_distance"),
    ]
    print()
    print("  Pairwise:")
    for col_a, col_b, label in pairs:
        if col_a not in comp_df or col_b not in comp_df:
            continue
        # Check for zero variance (skip if one is constant)
        if comp_df[col_a].std() == 0 or comp_df[col_b].std() == 0:
            print("  %s: zero variance -- skip" % label)
            continue
        sp_r, sp_p = stats.spearmanr(comp_df[col_a], comp_df[col_b])
        pe_r, pe_p = stats.pearsonr(comp_df[col_a],  comp_df[col_b])
        print("  %s:" % label)
        print("    Pearson  r = %+.4f, p = %.6f %s" % (pe_r, pe_p, _sig(pe_p)))
        print("    Spearman r = %+.4f, p = %.6f %s" % (sp_r, sp_p, _sig(sp_p)))

    print()
    print("  Crosstab ema_gap vs swing_distance (counts):")
    ct = pd.crosstab(df["score_ema_gap"], df["score_swing_distance"])
    print(ct.to_string(index=True))

'''

    new_content = content[:fn_start] + new_fn + content[fn_end_search:]
    with open('scripts/analyze_quality_score_phase43.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Done. Verifying syntax...')
    import subprocess
    result = subprocess.run(['python', '-m', 'py_compile', 'scripts/analyze_quality_score_phase43.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print('Syntax OK')
    else:
        print('Syntax error:')
        print(result.stderr)
