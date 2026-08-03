content = open('scripts/analyze_quality_score_phase43.py', encoding='utf-8').read()

# Find and replace the inter-component correlation section (Bagian C)
old_c = '''    print()
    print("  FOKUS: ema_gap vs alignment (dugaan redundansi 4.1):")
    sp_r, sp_p = stats.spearmanr(comp_df["score_ema_gap"], comp_df[
    print(f"    Pearson  r = {pe_r:+.4f}, p = {pe_p:.6f} {_sig(pe_p)}")
    print(f"    Spearman r = {sp_r:+.4f}, p = {sp_p:.6f} {_sig(sp_p)}")

    if abs(sp_r) > 0.7:
        print(f"    -> KORELASI TINGGI (r={sp_r:.3f} > 0.7): ema_gap dan alignment")
        print(f"       TERBUKTI redundan secara statistik -- penggabungan 4.1 JUSTIFIED.")
    elif abs(sp_r) > 0.4:
        print(f"    -> KORELASI SEDANG (r={sp_r:.3f}, 0.4-0.7): ada tumpang tindih")
        print(f"       tapi tidak sepenuhnya redundan -- perlu pertimbangan lebih.")
    else:
        print(f"    -> KORELASI RENDAH (r={sp_r:.3f} < 0.4): ema_gap dan alignment")
        print(f"       TIDAK redundan secara statistik -- penggabungan 4.1 perlu justifikasi lain.")

    print()
    print("  Crosstab ema_gap score vs alignment score (counts):")
    ct = pd.crosstab(df["score_ema_gap"], df["score_alignment"])
    print(ct.to_string(index=True))'''

new_c = '''    print()
    print("  FOKUS: ema_gap vs rsi_zone (anti-korelasi yang sudah ditemukan di run pertama):")
    sp_r, sp_p = stats.spearmanr(comp_df["score_ema_gap"], comp_df["score_rsi_zone"])
    pe_r, pe_p = stats.pearsonr(comp_df["score_ema_gap"], comp_df["score_rsi_zone"])
    print("    Pearson  r = %+.4f, p = %.6f %s" % (pe_r, pe_p, _sig(pe_p)))
    print("    Spearman r = %+.4f, p = %.6f %s" % (sp_r, sp_p, _sig(sp_p)))

    print()
    print("  FOKUS: ema_gap vs swing_distance (ada korelasi struktural?):")
    sp_r2, sp_p2 = stats.spearmanr(comp_df["score_ema_gap"], comp_df["score_swing_distance"])
    pe_r2, pe_p2 = stats.pearsonr(comp_df["score_ema_gap"], comp_df["score_swing_distance"])
    print("    Pearson  r = %+.4f, p = %.6f %s" % (pe_r2, pe_p2, _sig(pe_p2)))
    print("    Spearman r = %+.4f, p = %.6f %s" % (sp_r2, sp_p2, _sig(sp_p2)))

    print()
    print("  FOKUS: rsi_zone vs swing_distance:")
    sp_r3, sp_p3 = stats.spearmanr(comp_df["score_rsi_zone"], comp_df["score_swing_distance"])
    pe_r3, pe_p3 = stats.pearsonr(comp_df["score_rsi_zone"], comp_df["score_swing_distance"])
    print("    Pearson  r = %+.4f, p = %.6f %s" % (pe_r3, pe_p3, _sig(pe_p3)))
    print("    Spearman r = %+.4f, p = %.6f %s" % (sp_r3, sp_p3, _sig(sp_p3)))

    print()
    print("  Crosstab ema_gap vs swing_distance (counts):")
    ct = pd.crosstab(df["score_ema_gap"], df["score_swing_distance"])
    print(ct.to_string(index=True))'''

if old_c in content:
    content = content.replace(old_c, new_c)
    print('OK: inter-component section updated')
else:
    # partial match
    idx = content.find('FOKUS: ema_gap vs alignment')
    print('Partial idx:', idx)
    if idx >= 0:
        print(repr(content[idx-200:idx+500]))

with open('scripts/analyze_quality_score_phase43.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Written')
