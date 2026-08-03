import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
# using _h1_gap_grid_search run_one_backtest helper for quick access
from _h1_gap_grid_search import run_one_backtest, PARAMS_SET_B

def main():
    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")
    
    df_m5_raw = load_candles_csv(m5_path)
    df_h1_raw = load_candles_csv(h1_path)
    
    df_m5_ind = run_all_indicators(df_m5_raw.copy())
    df_h1_ind = run_all_indicators(df_h1_raw.copy())
    
    print("Menjalankan backtest untuk 0.01%, 0.02%, 0.03% pada Set B (Baseline)...")
    
    # Kumpulkan rrr_realized untuk masing-masing
    gaps = [0.01, 0.02, 0.03]
    results = {}
    
    for g in gaps:
        trades_df, _, _ = run_one_backtest(df_m5_ind, df_h1_ind, g, PARAMS_SET_B)
        if trades_df.empty:
            results[g] = []
        else:
            # pastikan hanya trades closed (ada rrr_realized)
            closed_trades = trades_df.dropna(subset=['rrr_realized'])
            results[g] = closed_trades['rrr_realized'].values

    print("\n--- T-TEST (Independent 2-sample t-test, equal_var=False) ---")
    
    pairs = [(0.01, 0.02), (0.02, 0.03), (0.01, 0.03)]
    for g1, g2 in pairs:
        a = results[g1]
        b = results[g2]
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        print(f"Gap {g1:.2f}% (Mean={mean_a:+.4f}, N={len(a)}) vs Gap {g2:.2f}% (Mean={mean_b:+.4f}, N={len(b)})")
        print(f"   t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            print("   -> BEDA SIGNIFIKAN secara statistik (p < 0.05)")
        else:
            print("   -> BEDA TIDAK SIGNIFIKAN (p >= 0.05) - variasi bisa karena noise")

if __name__ == "__main__":
    main()
