"""
scripts/_diag_pullback_walkforward.py
========================================
DIAGNOSTIK FASE 1e -- Walk-Forward Validation Trigger Pullback + Volume Filter
=================================================================================
LATAR BELAKANG:
    _diag_pullback_volume_filter_sweep.py menunjukkan bahwa trigger pullback-to-EMA
    yang difilter volume_ratio >= threshold BERPOTENSI profitable (PnL positif,
    AvgRRR positif, MaxDD jauh lebih kecil dari baseline). TAPI ini baru diuji
    di SATU window data (7 bulan) yang sama dengan window discovery-nya --
    risiko overfitting tinggi.

TUJUAN SCRIPT INI:
    Uji trigger ini di banyak fold (rolling window) yang TIDAK overlap satu
    sama lain, memakai metodologi IDENTIK dengan run_walk_forward.py yang
    sudah kamu pakai untuk validasi Fase 2.3. Kalau hasilnya konsisten
    positif di mayoritas fold -- ini edge asli. Kalau cuma menang di 1-2
    fold saja -- itu overfitting, bukan edge.

THRESHOLD YANG DIUJI:
    volume_min = 1.0  (titik mulai positif, sample besar -- lower bound)
    volume_min = 1.10 (titik PnL terbaik di sweep sebelumnya)
    volume_min = 1.278 (konstanta VOLUME_RATIO_HIGH_THRESHOLD yang SUDAH ADA
                        di rule_engine.py -- bukan angka baru yang ditambang
                        khusus untuk strategi ini, jadi lebih rendah risiko
                        overfit secara desain)

CARA PAKAI:
    python scripts/_diag_pullback_walkforward.py
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
from engine.backtester import merge_h1_to_m5, WARM_UP_CANDLES
from scripts._diag_pullback_candidates import find_pullback_candidates, RISK_PARAMS
from scripts._diag_pullback_volume_filter_sweep import backtest_with_volume_filter
from scripts.run_walk_forward import generate_folds
from scripts.run_oos_validation import filter_period

CHOSEN_EMA_PROXIMITY_PCT = 0.05
VOLUME_THRESHOLDS_TO_VALIDATE = [1.0, 1.10, 1.278]


def main():
    print("=" * 78)
    print("  DIAGNOSTIK FASE 1e -- WALK-FORWARD: TRIGGER PULLBACK + VOLUME FILTER")
    print("=" * 78)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"\n  Dataset extended tidak ditemukan:\n    {m5_path}\n    {h1_path}")
        print("  Jalankan dulu: python scripts/fetch_extended_data.py (butuh MT5 aktif)")
        sys.exit(1)

    print("\n-> Loading dataset extended (Jun 2025 - Jul 2026)...")
    df_m5_full = load_candles_csv(m5_path)
    df_h1_full = load_candles_csv(h1_path)
    print(f"   M5: {len(df_m5_full):,} candle ({df_m5_full.index[0]} -> {df_m5_full.index[-1]})")
    print(f"   H1: {len(df_h1_full):,} candle")

    print("\n-> Menghitung indikator (satu kali untuk seluruh dataset)...")
    df_m5_ind_full = run_all_indicators(df_m5_full.copy())
    df_h1_ind_full = run_all_indicators(df_h1_full.copy())
    df_merged_full = merge_h1_to_m5(df_m5_ind_full, df_h1_ind_full)

    print("-> Menandai candidate pullback di seluruh dataset...")
    df_tagged_full = find_pullback_candidates(df_merged_full, CHOSEN_EMA_PROXIMITY_PCT)

    folds = generate_folds(
        data_start = df_m5_full.index[0].strftime("%Y-%m-%d"),
        data_end   = df_m5_full.index[-1].strftime("%Y-%m-%d"),
        calib_months = 3,
        val_months   = 1,
    )
    print(f"\n-> {len(folds)} fold dihasilkan (validasi bulanan, tidak overlap).")

    for vmin in VOLUME_THRESHOLDS_TO_VALIDATE:
        print("\n" + "=" * 78)
        print(f"  THRESHOLD volume_min = {vmin}")
        print("=" * 78)
        print(f"\n  {'Fold':<6} {'Val Period':<24} {'Trades':>7} {'WinRate':>9} {'AvgRRR':>9} {'PnL Net':>10}")
        print(f"  {'-'*6} {'-'*24} {'-'*7} {'-'*9} {'-'*9} {'-'*10}")

        fold_results = []
        for f in folds:
            val_start, val_end = f["val_start"], f["val_end"]

            df_m5_val   = filter_period(df_m5_ind_full,  val_start, val_end)
            df_tagged_v = filter_period(df_tagged_full,   val_start, val_end)

            if len(df_m5_val) < WARM_UP_CANDLES + 20:
                print(f"  {f['fold']:<6} {val_start} -> {val_end:<12} SKIP (data kurang)")
                continue

            trades_df, summary = backtest_with_volume_filter(df_m5_ind_full, df_tagged_v, vmin)

            n   = summary.get("total_trades", 0)
            wr  = summary.get("win_rate")
            rrr = summary.get("avg_rrr_realized")
            pnl = summary.get("total_pnl_net")

            wr_s  = f"{wr*100:.1f}%" if wr is not None else "N/A"
            rrr_s = f"{rrr:+.4f}" if rrr is not None else "N/A"
            pnl_s = f"{pnl:+.1f}" if pnl is not None else "N/A"

            print(f"  {f['fold']:<6} {val_start} -> {val_end:<12} {n:>7} {wr_s:>9} {rrr_s:>9} {pnl_s:>10}")

            if n > 0:
                fold_results.append({"fold": f["fold"], "trades": n, "win_rate": wr,
                                      "avg_rrr": rrr, "pnl_net": pnl})

        # ── Ringkasan lintas fold (sama gaya dengan run_walk_forward.py) ────────
        if fold_results:
            pnls = [r["pnl_net"] for r in fold_results if r["pnl_net"] is not None]
            rrrs = [r["avg_rrr"] for r in fold_results if r["avg_rrr"] is not None]
            n_folds = len(pnls)
            n_pos   = sum(1 for p in pnls if p > 0)
            mean_pnl = np.mean(pnls) if pnls else 0
            std_pnl  = np.std(pnls, ddof=1) if len(pnls) > 1 else 0
            t_stat   = (mean_pnl / (std_pnl / np.sqrt(n_folds))) if std_pnl > 0 else 0

            print(f"\n  Ringkasan {n_folds} fold valid:")
            print(f"    Fold PnL positif : {n_pos}/{n_folds} ({n_pos/n_folds*100:.0f}%)")
            print(f"    Mean PnL/fold    : {mean_pnl:+.2f}  (std={std_pnl:.2f})")
            print(f"    t-stat           : {t_stat:+.3f}")
            if rrrs:
                print(f"    Mean AvgRRR/fold : {np.mean(rrrs):+.4f}  (std={np.std(rrrs):.4f})")
                neg_rrr_folds = sum(1 for r in rrrs if r <= 0)
                print(f"    Fold avg_rrr<=0  : {neg_rrr_folds}/{len(rrrs)}")

    print("\n" + "=" * 78)
    print("  CARA BACA & KEPUTUSAN")
    print("=" * 78)
    print("""
  Untuk tiap threshold, lihat:
    - Berapa % fold yang PnL-nya positif? Kalau >= 70-80% fold positif dan
      t-stat cukup besar (>2 kira-kira) -> ini edge yang konsisten, layak
      dipertimbangkan masuk produksi.
    - Kalau cuma 50-60% fold positif, atau ada beberapa fold dengan avg_rrr
      sangat negatif -> ini TIDAK stabil, kemungkinan besar overfitting
      terhadap window discovery-nya. JANGAN dimasukkan ke rule_engine.py dulu.

  Kirim balik seluruh tabel (ketiga threshold) untuk kita putuskan bersama
  apakah trigger pullback+volume ini siap diintegrasikan sebagai trigger
  kedua di rule_engine.py (OR logic dengan trigger M5-trend yang sudah ada),
  dan threshold mana yang paling robust untuk dipakai.
""")


if __name__ == "__main__":
    main()