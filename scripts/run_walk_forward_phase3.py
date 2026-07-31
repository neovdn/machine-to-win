"""
scripts/run_walk_forward_phase3.py
===================================
Script untuk membandingkan 3 varian rule engine di Fase 3.2:
- Baseline : Mode OFF (Volume diabaikan)
- Varian 1 : Mode FILTER (Volume = Filter)
- Varian 2 : Mode CONDITION (Volume = Kondisi 3, MIN_CONDITIONS = 3)
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from scripts.run_walk_forward import generate_folds, run_one_fold_validation, FASE1_PARAMS

def main():
    print("=" * 70)
    print("  WALK-FORWARD TESTING FASE 3.2 — VOLUME PARTICIPATION")
    print("=" * 70)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"❌ File data historis tidak ditemukan di {m5_path} / {h1_path}")
        sys.exit(1)

    print("-> Loading data M5 dan H1 dari CSV...")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)

    folds = generate_folds(
        data_start="2025-06-01",
        data_end="2026-07-25",
        calib_months=3,
        val_months=1,
    )

    print(f"-> Ditemukan {len(folds)} folds (11 fold diharapkan).")

    variants = [
        {"name": "BASELINE", "mode": "OFF"},
        {"name": "VARIAN 1", "mode": "FILTER"},
    ]

    results = {v["name"]: [] for v in variants}

    for i, f in enumerate(folds):
        fold_n = f["fold"]
        print(f"\n--- FOLD {fold_n}: Validasi {f['val_start']} s/d {f['val_end']} ---")

        for var in variants:
            res = run_one_fold_validation(
                df_m5_full=df_m5,
                df_h1_full=df_h1,
                val_start=f["val_start"],
                val_end=f["val_end"],
                params=FASE1_PARAMS,
                fold_label=f"Fold {fold_n}",
                volume_mode=var["mode"],
            )
            
            pnl = res.get("total_pnl_net")
            trd = res.get("total_trades", 0)
            wr  = res.get("win_rate")
            rrr = res.get("avg_rrr_realized")
            
            if pnl is not None:
                results[var["name"]].append({
                    "fold": fold_n,
                    "val_start": f["val_start"],
                    "val_end": f["val_end"],
                    "trades": trd,
                    "pnl": pnl,
                    "win_rate": wr,
                    "avg_rrr": rrr,
                    "max_drawdown": res.get("max_drawdown_net"),
                })
                
            wr_s = f"{wr:.1f}%" if wr is not None else "N/A"
            pnl_s = f"{pnl:+.2f}" if pnl is not None else "N/A"
            print(f"  [{var['name']:10s}] trades: {trd:3d} | WR: {wr_s:6s} | PnL: {pnl_s}")

    print("\n" + "=" * 70)
    print("  KESIMPULAN WALK-FORWARD FASE 3.2")
    print("=" * 70)

    # Simpan hasil per-fold ke CSV
    out_dir = os.path.join(ROOT_DIR, "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "walk_forward_phase3_results.csv")
    
    flat_results = []
    for var_name, var_results in results.items():
        for r in var_results:
            r_copy = r.copy()
            r_copy["variant"] = var_name
            flat_results.append(r_copy)
            
    if flat_results:
        results_df = pd.DataFrame(flat_results)
        results_df.to_csv(out_csv, index=False)
        print(f"Hasil per-fold disimpan ke: {out_csv}\n")
    
    
    print(f"{'Varian':10s} | {'Avg PnL':10s} | {'Std Dev':10s} | {'t-stat':8s} | {'Pos Folds':10s} | {'Avg WR':8s} | {'Avg RRR':8s} | {'Total Trades'}")
    print("-" * 95)

    for var in variants:
        res_list = results[var["name"]]
        if not res_list:
            continue
            
        pnls = [r["pnl"] for r in res_list]
        wrs = [r["win_rate"] for r in res_list if r["win_rate"] is not None]
        rrrs = [r["avg_rrr"] for r in res_list if r["avg_rrr"] is not None]
        trades = sum(r["trades"] for r in res_list)
        
        n_folds = len(pnls)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1) if n_folds > 1 else 0
        t_stat = (mean_pnl / (std_pnl / np.sqrt(n_folds))) if std_pnl > 0 else 0
        pos_folds = sum(1 for p in pnls if p > 0)
        
        mean_wr = np.mean(wrs) if wrs else 0
        mean_rrr = np.mean(rrrs) if rrrs else 0
        
        print(f"{var['name']:10s} | {mean_pnl:>10.2f} | {std_pnl:>10.2f} | {t_stat:>8.3f} | {pos_folds:2d}/{n_folds:<7d} | {mean_wr:>7.1f}% | {mean_rrr:>8.4f} | {trades:5d}")


if __name__ == "__main__":
    main()
