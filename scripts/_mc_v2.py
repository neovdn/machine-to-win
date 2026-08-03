import sys, os, glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

files = sorted(glob.glob('data/backtest_results/phase43_trades_*.csv'))
csv_path = files[-1]
print('Loading: %s' % csv_path)
df = pd.read_csv(csv_path)
df_closed = df[df['outcome'].isin(['TP_HIT', 'SL_HIT'])].copy()
df_closed['is_win'] = (df_closed['outcome'] == 'TP_HIT').astype(int)
print('Trades: %d' % len(df))
print()

all_tests = []
def add(name, p):
    if p is not None and not np.isnan(p):
        all_tests.append((name, float(p)))

# ANOVA per bucket
for metric in ['pnl_net', 'rrr_realized']:
    groups = [df_closed[df_closed['setup_quality']==b][metric].dropna() for b in ['STRONG','MODERATE','WEAK']]
    groups = [g for g in groups if len(g)>=5]
    if len(groups)>=2:
        F, p = stats.f_oneway(*groups)
        add('ANOVA_%s' % metric[:3], p)

# Pairwise t-tests
for a, b in [('STRONG','WEAK'),('STRONG','MODERATE'),('MODERATE','WEAK')]:
    for metric in ['rrr_realized','pnl_net']:
        ga = df_closed[df_closed['setup_quality']==a][metric].dropna()
        gb = df_closed[df_closed['setup_quality']==b][metric].dropna()
        if len(ga)>=5 and len(gb)>=5:
            _, p = stats.ttest_ind(ga, gb, equal_var=False)
            add('t_%svs%s_%s' % (a[:3],b[:3],metric[:3]), p)

# Per component (only 3 now)
for comp in ['score_ema_gap', 'score_rsi_zone', 'score_swing_distance']:
    x_all = df[comp].astype(float)
    x_cl  = df_closed[comp].astype(float)
    # vs is_win
    mask = x_cl.notna() & df_closed['is_win'].notna()
    if mask.sum()>=10 and x_cl[mask].std()>0:
        _, pp = stats.pearsonr(x_cl[mask], df_closed['is_win'][mask].astype(float))
        _, sp = stats.spearmanr(x_cl[mask], df_closed['is_win'][mask].astype(float))
        add('Pear_%s_win' % comp[6:14], pp)
        add('Spea_%s_win' % comp[6:14], sp)
    # vs pnl_net
    mask = x_all.notna() & df['pnl_net'].notna()
    if mask.sum()>=10 and x_all[mask].std()>0:
        _, pp = stats.pearsonr(x_all[mask], df['pnl_net'][mask])
        _, sp = stats.spearmanr(x_all[mask], df['pnl_net'][mask])
        add('Pear_%s_pnl' % comp[6:14], pp)
        add('Spea_%s_pnl' % comp[6:14], sp)
    # vs rrr
    mask = x_cl.notna() & df_closed['rrr_realized'].notna()
    if mask.sum()>=10 and x_cl[mask].std()>0:
        _, pp = stats.pearsonr(x_cl[mask], df_closed['rrr_realized'][mask])
        _, sp = stats.spearmanr(x_cl[mask], df_closed['rrr_realized'][mask])
        add('Pear_%s_rrr' % comp[6:14], pp)
        add('Spea_%s_rrr' % comp[6:14], sp)
    # t-test skor 2 vs 0
    g2 = df_closed[df_closed[comp]==2]['pnl_net'].dropna()
    g0 = df_closed[df_closed[comp]==0]['pnl_net'].dropna()
    if len(g2)>=5 and len(g0)>=5:
        _, p = stats.ttest_ind(g2, g0, equal_var=False)
        add('t_%s_2vs0' % comp[6:14], p)

# Inter-component
for col_a, col_b, lbl in [
    ('score_ema_gap','score_rsi_zone','eg_rz'),
    ('score_ema_gap','score_swing_distance','eg_sw'),
    ('score_rsi_zone','score_swing_distance','rz_sw'),
]:
    x = df[col_a].astype(float)
    y = df[col_b].astype(float)
    mask = x.notna() & y.notna()
    if mask.sum()>=10 and x[mask].std()>0 and y[mask].std()>0:
        _, pp = stats.pearsonr(x[mask], y[mask])
        _, sp = stats.spearmanr(x[mask], y[mask])
        add('Pear_%s' % lbl, pp)
        add('Spea_%s' % lbl, sp)

# Total score
x_tot_all = df['setup_quality_score'].astype(float)
x_tot_cl  = df_closed['setup_quality_score'].astype(float)
for name, x, y in [
    ('Pear_tot_win', x_tot_cl, df_closed['is_win'].astype(float)),
    ('Pear_tot_pnl', x_tot_all, df['pnl_net']),
    ('Pear_tot_rrr', x_tot_cl, df_closed['rrr_realized']),
]:
    mask = x.notna() & y.notna()
    if mask.sum()>=10 and x[mask].std()>0:
        _, p = stats.pearsonr(x[mask], y[mask])
        add(name, p)

# STRONG vs WEAK (score-based, Bagian D equivalent)
strong_d = df_closed[df_closed['setup_quality_score']>=5]
weak_d   = df_closed[df_closed['setup_quality_score']<3]
for metric in ['pnl_net','rrr_realized']:
    ga = strong_d[metric].dropna()
    gb = weak_d[metric].dropna()
    if len(ga)>=5 and len(gb)>=5:
        _, p = stats.ttest_ind(ga, gb, equal_var=False)
        add('t_STRONG_WEAK_%s' % metric[:3], p)

m = len(all_tests)
bonf = 0.05 / m
print('Total tests: %d | Bonferroni threshold: %.6f' % (m, bonf))
print()

sorted_tests = sorted(all_tests, key=lambda x: x[1])
bh_thresh    = [(i+1)*0.05/m for i in range(m)]

largest_k = 0
for k, (n, p) in enumerate(sorted_tests):
    if p <= bh_thresh[k]:
        largest_k = k + 1

print('=== FULL P-VALUE TABLE (sorted, m=%d tests) ===' % m)
hdr = '  %4s  %-38s  %-10s  %-10s  %-4s  %-4s'
print(hdr % ('Rank', 'Test', 'p-value', 'BH_thr', 'BH', 'Bonf'))
for k, (n, p) in enumerate(sorted_tests):
    bh_ok  = 'YES' if k < largest_k else 'no'
    bon_ok = 'YES' if p < bonf else 'no'
    row = '  %4d  %-38s  %-10.6f  %-10.6f  %-4s  %-4s'
    print(row % (k+1, n, p, bh_thresh[k], bh_ok, bon_ok))

print()
print('=== SUMMARY ===')
print('Surviving Bonferroni (%d tests, alpha=%.6f):' % (m, bonf))
surv_b = [(n,p) for n,p in all_tests if p < bonf]
for n,p in sorted(surv_b, key=lambda x: x[1]):
    print('  %-38s p=%.8f' % (n, p))
print()
print('Surviving BH FDR (q=0.05, largest_k=%d):' % largest_k)
for n,p in sorted_tests[:largest_k]:
    print('  %-38s p=%.8f' % (n, p))

# Extra: swing_distance details
print()
print('=== SWING DISTANCE vs RRR_REALIZED (detail) ===')
x = df_closed['score_swing_distance'].astype(float)
y = df_closed['rrr_realized']
mask = x.notna() & y.notna()
sp_r, sp_p = stats.spearmanr(x[mask], y[mask])
pe_r, pe_p = stats.pearsonr(x[mask], y[mask])
print('n=%d | Pearson r=%+.4f p=%.8f | Spearman r=%+.4f p=%.8f' % (mask.sum(), pe_r, pe_p, sp_r, sp_p))
print()
print('Mean rrr_realized per swing_distance score:')
for sc in [0, 1, 2]:
    sub = df_closed[df_closed['score_swing_distance']==sc]['rrr_realized']
    print('  Skor %d: mean=%+.4f, n=%d' % (sc, sub.mean() if len(sub)>0 else float('nan'), len(sub)))
