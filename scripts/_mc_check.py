import sys, os, glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Load trade log
files = sorted(glob.glob('data/backtest_results/phase43_trades_*.csv'))
csv_path = files[-1]
print(f'Loading: {csv_path}')
df = pd.read_csv(csv_path)
print(f'Total trades: {len(df)}')
df_closed = df[df['outcome'].isin(['TP_HIT', 'SL_HIT'])].copy()
df_closed['is_win'] = (df_closed['outcome'] == 'TP_HIT').astype(int)

# ===== STRONG BUCKET DETAIL =====
strong = df[df['setup_quality'] == 'STRONG']['pnl_net']
print()
print('=== STRONG BUCKET (n=%d) pnl_net ===' % len(strong))
print('  Mean   : %+.4f' % strong.mean())
print('  Median : %+.4f' % strong.median())
print('  Std Dev: %.4f' % strong.std())
print('  Min    : %+.4f' % strong.min())
print('  Max    : %+.4f' % strong.max())
print('  Q25    : %+.4f' % strong.quantile(0.25))
print('  Q75    : %+.4f' % strong.quantile(0.75))
print()
print('  pnl_net > 0  : %d (%.1f%%)' % ((strong > 0).sum(), (strong > 0).mean()*100))
print('  pnl_net <= 0 : %d (%.1f%%)' % ((strong <= 0).sum(), (strong <= 0).mean()*100))
print('  pnl_net > 10 : %d (%.1f%%)' % ((strong > 10).sum(), (strong > 10).mean()*100))
print('  pnl_net > 20 : %d (%.1f%%)' % ((strong > 20).sum(), (strong > 20).mean()*100))
print()
print('  Top 10 pnl_net values:')
for v in sorted(strong.values, reverse=True)[:10]:
    print('    %+.4f' % v)
print('  Bottom 5 pnl_net values:')
for v in sorted(strong.values)[:5]:
    print('    %+.4f' % v)

# ===== ALL P-VALUES =====
print()
print('=== COMPUTING ALL P-VALUES ===')
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

# Pairwise t-tests per bucket
for a, b in [('STRONG','WEAK'),('STRONG','MODERATE'),('MODERATE','WEAK')]:
    for metric in ['rrr_realized','pnl_net']:
        ga = df_closed[df_closed['setup_quality']==a][metric].dropna()
        gb = df_closed[df_closed['setup_quality']==b][metric].dropna()
        if len(ga)>=5 and len(gb)>=5:
            _, p = stats.ttest_ind(ga, gb, equal_var=False)
            add('t_%svs%s_%s' % (a[:3],b[:3],metric[:3]), p)

# Per component correlations (only functioning: ema_gap, rsi_zone)
for comp in ['score_ema_gap', 'score_rsi_zone']:
    x_all = df[comp].astype(float)
    x_cl  = df_closed[comp].astype(float)
    # vs is_win
    mask = x_cl.notna() & df_closed['is_win'].notna()
    if mask.sum()>=10:
        _, pp = stats.pearsonr(x_cl[mask], df_closed['is_win'][mask].astype(float))
        _, sp = stats.spearmanr(x_cl[mask], df_closed['is_win'][mask].astype(float))
        add('Pear_%s_win' % comp[6:12], pp)
        add('Spea_%s_win' % comp[6:12], sp)
    # vs pnl_net
    mask = x_all.notna() & df['pnl_net'].notna()
    if mask.sum()>=10:
        _, pp = stats.pearsonr(x_all[mask], df['pnl_net'][mask])
        _, sp = stats.spearmanr(x_all[mask], df['pnl_net'][mask])
        add('Pear_%s_pnl' % comp[6:12], pp)
        add('Spea_%s_pnl' % comp[6:12], sp)
    # vs rrr_realized
    mask = x_cl.notna() & df_closed['rrr_realized'].notna()
    if mask.sum()>=10:
        _, pp = stats.pearsonr(x_cl[mask], df_closed['rrr_realized'][mask])
        _, sp = stats.spearmanr(x_cl[mask], df_closed['rrr_realized'][mask])
        add('Pear_%s_rrr' % comp[6:12], pp)
        add('Spea_%s_rrr' % comp[6:12], sp)
    # t-test skor 2 vs skor 0
    g2 = df_closed[df_closed[comp]==2]['pnl_net'].dropna()
    g0 = df_closed[df_closed[comp]==0]['pnl_net'].dropna()
    if len(g2)>=5 and len(g0)>=5:
        _, p = stats.ttest_ind(g2, g0, equal_var=False)
        add('t_%s_2vs0' % comp[6:12], p)

# Inter-component: ema_gap vs rsi_zone
x_eg = df['score_ema_gap'].astype(float)
x_rz = df['score_rsi_zone'].astype(float)
mask = x_eg.notna() & x_rz.notna()
if mask.sum()>=10:
    _, pp = stats.pearsonr(x_eg[mask], x_rz[mask])
    _, sp = stats.spearmanr(x_eg[mask], x_rz[mask])
    add('Pear_emag_vs_rsi', pp)
    add('Spea_emag_vs_rsi', sp)

# Total score vs outcome
x_tot_all = df['setup_quality_score'].astype(float)
x_tot_cl  = df_closed['setup_quality_score'].astype(float)
for name, x, y in [
    ('Pear_total_win', x_tot_cl, df_closed['is_win'].astype(float)),
    ('Pear_total_pnl', x_tot_all, df['pnl_net']),
    ('Pear_total_rrr', x_tot_cl, df_closed['rrr_realized']),
]:
    mask = x.notna() & y.notna()
    if mask.sum()>=10:
        _, p = stats.pearsonr(x[mask], y[mask])
        add(name, p)

m = len(all_tests)
bonf_thresh = 0.05 / m
print('Total tests: %d' % m)
print('Bonferroni threshold: %.6f' % bonf_thresh)

# BH FDR
sorted_tests = sorted(all_tests, key=lambda x: x[1])
bh_thresholds = [(i+1)*0.05/m for i in range(m)]
largest_k = 0
for k, (name, p) in enumerate(sorted_tests):
    if p <= bh_thresholds[k]:
        largest_k = k + 1

print()
print('=== FULL P-VALUE TABLE (sorted, m=%d tests) ===' % m)
header = '  Rank  %-42s p-value    BH_thr     BH  Bonf'
print(header % 'Test')
for k, (name, p) in enumerate(sorted_tests):
    bh_ok = 'YES' if k < largest_k else 'no'
    bonf_ok = 'YES' if p < bonf_thresh else 'no'
    row = '  %4d  %-42s %.6f   %.6f   %-3s  %s'
    print(row % (k+1, name, p, bh_thresholds[k], bh_ok, bonf_ok))

print()
print('=== SUMMARY ===')
print('Surviving Bonferroni (%d tests, alpha=%.6f):' % (m, bonf_thresh))
surv_bonf = [(n,p) for n,p in all_tests if p < bonf_thresh]
if surv_bonf:
    for n,p in sorted(surv_bonf, key=lambda x: x[1]):
        print('  %s: p=%.8f' % (n, p))
else:
    print('  NONE')
print()
print('Surviving BH FDR (q=0.05, largest_k=%d):' % largest_k)
if largest_k > 0:
    for n,p in sorted_tests[:largest_k]:
        print('  %s: p=%.8f' % (n, p))
else:
    print('  NONE')
