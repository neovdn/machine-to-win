import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from scripts.run_walk_forward import generate_folds, FASE1_PARAMS
from scripts.run_param_sweep import run_fast_backtest
from engine.indicators import run_all_indicators
from engine.backtester import merge_h1_to_m5, WARM_UP_CANDLES, MAX_FORWARD_CANDLES, DEFAULT_SPREAD_PTS
from scripts.run_oos_validation import filter_period

def main():
    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")
    
    print("Memuat data...")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)

    folds = generate_folds(
        data_start="2025-06-01",
        data_end="2026-07-25",
        calib_months=3,
        val_months=1,
    )

    baseline_trades_list = []
    varian1_trades_list = []

    print(f"Menjalankan ulang backtest untuk {len(folds)} folds...")
    for f in folds:
        val_start = f["val_start"]
        val_end = f["val_end"]
        
        df_m5_val = filter_period(df_m5, val_start, val_end)
        df_h1_val = filter_period(df_h1, val_start, val_end)

        if len(df_m5_val) < WARM_UP_CANDLES + 50:
            continue

        df_m5_val_ind = run_all_indicators(df_m5_val.copy())
        df_h1_val_ind = run_all_indicators(df_h1_val.copy())
        df_val_merged = merge_h1_to_m5(df_m5_val_ind, df_h1_val_ind)

        trades_bl, _ = run_fast_backtest(
            df_m5_ind=df_m5_val_ind, df_merged=df_val_merged, **FASE1_PARAMS, volume_mode="OFF"
        )
        trades_v1, _ = run_fast_backtest(
            df_m5_ind=df_m5_val_ind, df_merged=df_val_merged, **FASE1_PARAMS, volume_mode="FILTER"
        )
        
        baseline_trades_list.append(trades_bl)
        varian1_trades_list.append(trades_v1)

    print("Menggabungkan data trades...")
    all_baseline = pd.concat(baseline_trades_list, ignore_index=True)
    all_varian1 = pd.concat(varian1_trades_list, ignore_index=True)

    # Identifikasi trade yang dibuang
    baseline_entry_times = set(all_baseline["entry_time"])
    varian1_entry_times = set(all_varian1["entry_time"])

    filtered_out_times = baseline_entry_times - varian1_entry_times
    passed_times = varian1_entry_times

    filtered_out_trades = all_baseline[all_baseline["entry_time"].isin(filtered_out_times)]
    passed_trades = all_baseline[all_baseline["entry_time"].isin(passed_times)]

    print("\n" + "="*50)
    print("HASIL ANALISIS LEVEL TRADE (FILTER Q25/Q75)")
    print("="*50)
    print(f"Total trades baseline: {len(all_baseline)}")
    print(f"Total trades varian 1: {len(all_varian1)}")
    print(f"Total trades dibuang : {len(filtered_out_trades)}")
    print(f"Total trades lolos   : {len(passed_trades)}")

    def analyze_group(name, group):
        pnl = group["pnl_net"]
        mean = pnl.mean()
        std = pnl.std()
        n = len(pnl)
        wins = len(group[group["pnl_net"] > 0])
        win_rate = wins / n * 100 if n > 0 else 0
        
        if n > 1:
            t_stat, p_val = stats.ttest_1samp(pnl, 0)
        else:
            t_stat, p_val = np.nan, np.nan
            
        print(f"\n--- {name} ---")
        print(f"N trades    : {n}")
        print(f"Mean PnL    : {mean:.2f}")
        print(f"Std Dev PnL : {std:.2f}")
        print(f"Win Rate    : {win_rate:.1f}% ({wins}/{n})")
        print(f"t-stat      : {t_stat:.3f}")
        print(f"p-value     : {p_val:.4e} ({'Signifikan' if p_val < 0.05 else 'Tidak Signifikan'})")

    analyze_group("TRADE YANG DIBUANG OLEH FILTER", filtered_out_trades)
    analyze_group("TRADE YANG LOLOS FILTER", passed_trades)
    
    wins_bl = len(all_baseline[all_baseline["pnl_net"] > 0])
    wr_bl = wins_bl / len(all_baseline) * 100
    print(f"\nOverall Baseline Win Rate: {wr_bl:.1f}%")

if __name__ == "__main__":
    main()
