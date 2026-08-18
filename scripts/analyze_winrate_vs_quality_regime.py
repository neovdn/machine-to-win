import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
import math
import os

# CONFIGURATION
CSV_FILE = "data/backtest_results/backtest_XAUUSD_M5_20260803_112829.csv"
OUTPUT_DIR = "data/backtest_results/"
ALPHA = 0.05

def wilson_score_interval(successes, n, confidence=0.95):
    """Calculate the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    return center - spread, center + spread

def reconciliation_check(total_raw, total_closed, df_closed, categories, col_name):
    """Verifies that the breakdown accounts for all trades."""
    sum_breakdown = df_closed.shape[0]
    if sum_breakdown != total_closed:
        raise ValueError(f"Reconciliation failed: df_closed has {df_closed.shape[0]} trades, expected {total_closed}")
    sum_cats = df_closed[col_name].isin(categories).sum()
    if sum_cats != total_closed:
        raise ValueError(f"Reconciliation failed: breakdown by {col_name} sum is {sum_cats}, expected {total_closed}")
    return True

def analyze_winrate_vs_quality():
    print("=" * 80)
    print("ANALISIS WIN RATE VS SETUP QUALITY VS REGIME PASAR")
    print("=" * 80)
    
    # TAHAP 1: PRE-REGISTERED CRITERIA
    print("\n[TAHAP 1] SPESIFIKASI & KRITERIA SUKSES")
    print(f"File Sumber: {CSV_FILE}")
    print("Alasan: File CSV ini adalah log backtest representatif untuk konfigurasi Fase 1 (ATR=0.9, Lookback=15, Wing=3, RRR_min=1.3) yang telah diverifikasi di codebase saat ini.")
    print("\nKriteria Sukses Pertanyaan 1 (Setup Quality):")
    print("- LOLOS jika ada perbedaan win rate STRONG vs WEAK yang signifikan (p < alpha setelah koreksi FDR/Bonferroni) DAN arah perbedaannya STRONG > WEAK.")
    print("- Jika n_trades di bucket < 30, dianggap PERLU DATA LEBIH.")
    print("\nKriteria Sukses Pertanyaan 2 (Regime/Trend Strength):")
    print("- LOLOS jika korelasi |ema_gap_pct| vs is_win atau pnl_net signifikan (p < alpha setelah koreksi) DAN arahnya masuk akal (gap besar -> performa lebih baik).")
    print("- Jika n_trades < 30, dianggap PERLU DATA LEBIH.")
    print("\nMinimum total trades = 100.")
    print("-" * 80)

    # Load Data
    full_path = os.path.join(os.getcwd(), CSV_FILE)
    if not os.path.exists(full_path):
        print(f"ERROR: File {full_path} tidak ditemukan!")
        return

    df_raw = pd.read_csv(full_path)
    total_raw = len(df_raw)
    
    # Filter only closed trades
    df_closed = df_raw[df_raw['outcome'].isin(['TP_HIT', 'SL_HIT'])].copy()
    total_closed = len(df_closed)
    
    df_closed['is_win'] = (df_closed['outcome'] == 'TP_HIT').astype(int)
    df_closed['abs_ema_gap_pct'] = df_raw['ema_gap_pct'].abs()
    
    # Make sure we use raw df for all-trades metrics where appropriate, but usually we just calculate it.
    # We will compute avg pnl_net (all) by grouping on raw df as well.
    # Wait, df_raw is needed for "all trades" metric for PnL.
    
    if total_closed < 100:
        print(f"PERLU DATA LEBIH: Total trade closed hanya {total_closed} (< 100). Analisis dihentikan.")
        return

    # TAHAP 2: BAGIAN D - RECONCILIATION
    try:
        reconciliation_check(total_raw, total_closed, df_closed, ['STRONG', 'MODERATE', 'WEAK'], 'setup_quality')
        print(f"[RECONCILIATION CHECK] Lolos. Total Raw: {total_raw}, Total Closed: {total_closed}")
    except ValueError as e:
        print(f"[RECONCILIATION CHECK] GAGAL: {e}")
        return
        
    p_values_collection = [] # list of dicts: {'hypothesis': str, 'p_value': float, 'effect_size': str}

    # BAGIAN A: SETUP QUALITY
    print("\n" + "=" * 80)
    print("BAGIAN A: BREAKDOWN PER SETUP QUALITY")
    print("=" * 80)
    
    categories = ['STRONG', 'MODERATE', 'WEAK']
    quality_stats = []
    
    for cat in categories:
        df_cat_closed = df_closed[df_closed['setup_quality'] == cat]
        df_cat_all = df_raw[df_raw['setup_quality'] == cat]
        
        n_closed = len(df_cat_closed)
        n_all = len(df_cat_all)
        
        if n_closed > 0:
            wins = df_cat_closed['is_win'].sum()
            win_rate = wins / n_closed
            ci_lower, ci_upper = wilson_score_interval(wins, n_closed)
            avg_rrr = df_cat_closed['rrr_realized'].mean()
        else:
            win_rate, ci_lower, ci_upper, avg_rrr = 0.0, 0.0, 0.0, 0.0
            
        avg_pnl_all = df_cat_all['pnl_net'].mean() if n_all > 0 else 0.0
        
        quality_stats.append({
            'Quality': cat,
            'N_Closed': n_closed,
            'N_All': n_all,
            'WinRate': win_rate,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Avg_RRR_Closed': avg_rrr,
            'Avg_PnL_All': avg_pnl_all
        })
    
    df_quality = pd.DataFrame(quality_stats)
    print("\nTabel Setup Quality:")
    print(df_quality.to_string(index=False))
    df_quality.to_csv(os.path.join(OUTPUT_DIR, "winrate_quality_analysis_partA_table.csv"), index=False)

    # ANOVA / Chi-Square
    strong = df_closed[df_closed['setup_quality'] == 'STRONG']
    moderate = df_closed[df_closed['setup_quality'] == 'MODERATE']
    weak = df_closed[df_closed['setup_quality'] == 'WEAK']
    
    # ANOVA avg_rrr_realized
    f_stat, p_anova_rrr = stats.f_oneway(strong['rrr_realized'], moderate['rrr_realized'], weak['rrr_realized'])
    p_values_collection.append({'hypothesis': 'ANOVA RRR_Realized (Quality)', 'p_value': p_anova_rrr, 'effect_size': f'F={f_stat:.3f}'})
    
    # Chi-Square Win Rate
    contingency = [
        [strong['is_win'].sum(), len(strong) - strong['is_win'].sum()],
        [moderate['is_win'].sum(), len(moderate) - moderate['is_win'].sum()],
        [weak['is_win'].sum(), len(weak) - weak['is_win'].sum()]
    ]
    # Filter out empty rows for chi2
    valid_contingency = [row for row in contingency if sum(row) > 0]
    if len(valid_contingency) > 1:
        chi2, p_chi2, dof, ex = stats.chi2_contingency(valid_contingency)
        p_values_collection.append({'hypothesis': 'Chi-Square Win Rate (Quality)', 'p_value': p_chi2, 'effect_size': f'Chi2={chi2:.3f}'})
    
    # Pairwise t-test for STRONG vs WEAK (Welch)
    if len(strong) >= 2 and len(weak) >= 2:
        t_stat, p_t_sw = stats.ttest_ind(strong['rrr_realized'], weak['rrr_realized'], equal_var=False)
        # Cohen's d
        s_pooled = np.sqrt(((len(strong)-1)*strong['rrr_realized'].var() + (len(weak)-1)*weak['rrr_realized'].var()) / (len(strong)+len(weak)-2))
        d_val = (strong['rrr_realized'].mean() - weak['rrr_realized'].mean()) / (s_pooled if s_pooled > 0 else 1)
        p_values_collection.append({'hypothesis': 't-test STRONG vs WEAK RRR', 'p_value': p_t_sw, 'effect_size': f"Cohen's d={d_val:.3f}"})
        
        # Prop Z-test for Win Rate STRONG vs WEAK
        count = np.array([strong['is_win'].sum(), weak['is_win'].sum()])
        nobs = np.array([len(strong), len(weak)])
        if np.all(nobs > 0):
            z_stat, p_z_sw = proportions_ztest(count, nobs)
            wr_diff = (strong['is_win'].mean() - weak['is_win'].mean()) * 100
            p_values_collection.append({'hypothesis': 'z-test STRONG vs WEAK WinRate', 'p_value': p_z_sw, 'effect_size': f'Diff={wr_diff:.1f}%'})

    # Correlation components
    components = ['score_ema_gap', 'score_rsi_zone', 'score_swing_distance']
    for comp in components:
        if comp in df_closed.columns:
            p_corr, p_pval = stats.pearsonr(df_closed[comp], df_closed['is_win'])
            s_corr, s_pval = stats.spearmanr(df_closed[comp], df_closed['is_win'])
            p_values_collection.append({'hypothesis': f'Pearson {comp} vs is_win', 'p_value': p_pval, 'effect_size': f'r={p_corr:.3f}'})
            p_values_collection.append({'hypothesis': f'Spearman {comp} vs is_win', 'p_value': s_pval, 'effect_size': f'rho={s_corr:.3f}'})
            
            p_corr_pnl, p_pval_pnl = stats.pearsonr(df_raw[comp], df_raw['pnl_net'])
            s_corr_pnl, s_pval_pnl = stats.spearmanr(df_raw[comp], df_raw['pnl_net'])
            p_values_collection.append({'hypothesis': f'Pearson {comp} vs pnl_net', 'p_value': p_pval_pnl, 'effect_size': f'r={p_corr_pnl:.3f}'})
            p_values_collection.append({'hypothesis': f'Spearman {comp} vs pnl_net', 'p_value': s_pval_pnl, 'effect_size': f'rho={s_corr_pnl:.3f}'})

    # BAGIAN B: EMA GAP PCT (TREND STRENGTH)
    print("\n" + "=" * 80)
    print("BAGIAN B: BREAKDOWN PER EMA GAP (TREND STRENGTH)")
    print("=" * 80)
    
    df_raw['abs_ema_gap_pct'] = df_raw['ema_gap_pct'].abs()
    
    try:
        df_closed['ema_quartile'] = pd.qcut(df_closed['abs_ema_gap_pct'], 4, labels=['Q1(Terlemah)', 'Q2', 'Q3', 'Q4(Terkuat)'])
        df_raw['ema_quartile'] = pd.qcut(df_raw['abs_ema_gap_pct'], 4, labels=['Q1(Terlemah)', 'Q2', 'Q3', 'Q4(Terkuat)'])
        
        trend_stats = []
        for q in ['Q1(Terlemah)', 'Q2', 'Q3', 'Q4(Terkuat)']:
            df_q_closed = df_closed[df_closed['ema_quartile'] == q]
            df_q_all = df_raw[df_raw['ema_quartile'] == q]
            
            n_closed = len(df_q_closed)
            n_all = len(df_q_all)
            
            if n_closed > 0:
                wins = df_q_closed['is_win'].sum()
                win_rate = wins / n_closed
                ci_lower, ci_upper = wilson_score_interval(wins, n_closed)
                avg_rrr = df_q_closed['rrr_realized'].mean()
            else:
                win_rate, ci_lower, ci_upper, avg_rrr = 0.0, 0.0, 0.0, 0.0
                
            avg_pnl_all = df_q_all['pnl_net'].mean() if n_all > 0 else 0.0
            
            trend_stats.append({
                'Quartile': q,
                'N_Closed': n_closed,
                'N_All': n_all,
                'WinRate': win_rate,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Avg_RRR_Closed': avg_rrr,
                'Avg_PnL_All': avg_pnl_all
            })
            
        df_trend = pd.DataFrame(trend_stats)
        print("\nTabel Trend Strength Quartiles:")
        print(df_trend.to_string(index=False))
        df_trend.to_csv(os.path.join(OUTPUT_DIR, "winrate_quality_analysis_partB_table.csv"), index=False)
        
        # Correlations for Part B
        p_corr_ema, p_pval_ema = stats.pearsonr(df_closed['abs_ema_gap_pct'], df_closed['is_win'])
        s_corr_ema, s_pval_ema = stats.spearmanr(df_closed['abs_ema_gap_pct'], df_closed['is_win'])
        p_values_collection.append({'hypothesis': 'Pearson |ema_gap| vs is_win', 'p_value': p_pval_ema, 'effect_size': f'r={p_corr_ema:.3f}'})
        p_values_collection.append({'hypothesis': 'Spearman |ema_gap| vs is_win', 'p_value': s_pval_ema, 'effect_size': f'rho={s_corr_ema:.3f}'})
        
        p_corr_emapnl, p_pval_emapnl = stats.pearsonr(df_raw['abs_ema_gap_pct'], df_raw['pnl_net'])
        s_corr_emapnl, s_pval_emapnl = stats.spearmanr(df_raw['abs_ema_gap_pct'], df_raw['pnl_net'])
        p_values_collection.append({'hypothesis': 'Pearson |ema_gap| vs pnl_net', 'p_value': p_pval_emapnl, 'effect_size': f'r={p_corr_emapnl:.3f}'})
        p_values_collection.append({'hypothesis': 'Spearman |ema_gap| vs pnl_net', 'p_value': s_pval_emapnl, 'effect_size': f'rho={s_corr_emapnl:.3f}'})
    except ValueError as e:
        print("Gagal membuat kuartil:", e)

    # Trigger source if available
    if 'trigger_source' in df_closed.columns:
        print("\nKolom 'trigger_source' ditemukan. Menghitung breakdown...")
        # Lakukan breakdown di sini (tidak diimplementasikan karena file saat ini tidak punya kolom ini)
    else:
        print("\nKolom 'trigger_source' TIDAK ditemukan. Melewati breakdown trigger_source.")

    # BAGIAN C: MULTIPLE COMPARISON
    print("\n" + "=" * 80)
    print("BAGIAN C: MULTIPLE COMPARISON CORRECTION")
    print("=" * 80)
    
    df_pvals = pd.DataFrame(p_values_collection)
    
    # Bonferroni
    reject_bonf, pvals_corrected_bonf, _, _ = multipletests(df_pvals['p_value'], alpha=ALPHA, method='bonferroni')
    df_pvals['p_bonferroni'] = pvals_corrected_bonf
    df_pvals['sig_bonferroni'] = reject_bonf
    
    # Benjamini-Hochberg FDR
    reject_fdr, pvals_corrected_fdr, _, _ = multipletests(df_pvals['p_value'], alpha=ALPHA, method='fdr_bh')
    df_pvals['p_fdr_bh'] = pvals_corrected_fdr
    df_pvals['sig_fdr_bh'] = reject_fdr
    
    print("\nTabel P-Value dan Signifikansi:")
    print(df_pvals.to_string(index=False))
    df_pvals.to_csv(os.path.join(OUTPUT_DIR, "winrate_quality_analysis_partC_pvals.csv"), index=False)

    # BAGIAN D: VERDICT
    print("\n" + "=" * 80)
    print("BAGIAN D: VERDICT (KESIMPULAN)")
    print("=" * 80)
    
    # Verdict 1: Setup Quality
    z_test_row = df_pvals[df_pvals['hypothesis'] == 'z-test STRONG vs WEAK WinRate']
    if not z_test_row.empty:
        is_sig = z_test_row['sig_fdr_bh'].values[0] or z_test_row['sig_bonferroni'].values[0]
        # check direction
        diff_str = z_test_row['effect_size'].values[0]
        diff_val = float(diff_str.replace('Diff=', '').replace('%', ''))
        n_strong = df_quality[df_quality['Quality']=='STRONG']['N_Closed'].values[0]
        n_weak = df_quality[df_quality['Quality']=='WEAK']['N_Closed'].values[0]
        
        if n_strong < 30 or n_weak < 30:
            print("VERDICT PERTANYAAN 1 (Setup Quality Prediktif): PERLU DATA LEBIH")
            print(f"Alasan: Jumlah trade di kategori STRONG ({n_strong}) atau WEAK ({n_weak}) kurang dari batas minimum (30).")
        elif is_sig and diff_val > 0:
            print("VERDICT PERTANYAAN 1 (Setup Quality Prediktif): LOLOS")
            print(f"Alasan: Terdapat perbedaan Win Rate yang signifikan antara STRONG vs WEAK (p_fdr < {ALPHA}), dan arahnya sesuai (STRONG lebih baik {diff_val:.1f}%).")
        else:
            print("VERDICT PERTANYAAN 1 (Setup Quality Prediktif): TIDAK LOLOS")
            if is_sig and diff_val <= 0:
                print(f"Alasan: Perbedaan signifikan tapi arahnya terbalik (WEAK lebih baik atau sama). Efek: {diff_val:.1f}%.")
            else:
                p_val_fdr = z_test_row['p_fdr_bh'].values[0]
                print(f"Alasan: Tidak ada perbedaan signifikan antara STRONG vs WEAK (p_fdr = {p_val_fdr:.4f} >= {ALPHA}). Scoring saat ini tidak prediktif.")
    else:
        print("VERDICT PERTANYAAN 1 (Setup Quality Prediktif): PERLU DATA LEBIH (Gagal Hitung Z-Test)")
        
    print("")
    
    # Verdict 2: Trend Strength (EMA Gap)
    corr_ema_win_row = df_pvals[df_pvals['hypothesis'] == 'Spearman |ema_gap| vs is_win']
    corr_ema_pnl_row = df_pvals[df_pvals['hypothesis'] == 'Spearman |ema_gap| vs pnl_net']
    
    if not corr_ema_win_row.empty and not corr_ema_pnl_row.empty:
        is_sig_win = corr_ema_win_row['sig_fdr_bh'].values[0]
        is_sig_pnl = corr_ema_pnl_row['sig_fdr_bh'].values[0]
        
        eff_win = float(corr_ema_win_row['effect_size'].values[0].replace('rho=', ''))
        eff_pnl = float(corr_ema_pnl_row['effect_size'].values[0].replace('rho=', ''))
        
        n_q1 = df_trend[df_trend['Quartile']=='Q1(Terlemah)']['N_Closed'].values[0]
        n_q4 = df_trend[df_trend['Quartile']=='Q4(Terkuat)']['N_Closed'].values[0]
        
        if n_q1 < 30 or n_q4 < 30:
            print("VERDICT PERTANYAAN 2 (Regime/Trend Strength Prediktif): PERLU DATA LEBIH")
            print(f"Alasan: N trade per kuartil (Q1={n_q1}, Q4={n_q4}) kurang dari 30. Terlalu sedikit data untuk menarik kesimpulan rezim.")
        elif (is_sig_win and eff_win > 0) or (is_sig_pnl and eff_pnl > 0):
            print("VERDICT PERTANYAAN 2 (Regime/Trend Strength Prediktif): LOLOS")
            print(f"Alasan: |ema_gap| berkorelasi signifikan secara statistik dengan Win Rate atau PnL (rho_win={eff_win:.3f}, rho_pnl={eff_pnl:.3f}). Tren kuat berasosiasi dengan performa lebih baik.")
        else:
            print("VERDICT PERTANYAAN 2 (Regime/Trend Strength Prediktif): TIDAK LOLOS")
            print(f"Alasan: |ema_gap| tidak memiliki korelasi positif yang signifikan secara statistik dengan performa (rho_win={eff_win:.3f}, rho_pnl={eff_pnl:.3f}). Filter rezim berbasis EMA gap tidak cukup prediktif.")
    else:
         print("VERDICT PERTANYAAN 2 (Regime/Trend Strength Prediktif): PERLU DATA LEBIH (Gagal Hitung Korelasi)")

if __name__ == "__main__":
    analyze_winrate_vs_quality()
