"""
scripts/run_walk_forward.py
============================
Walk-Forward Testing — Fase 2.3

METODOLOGI:
    Jalankan backtest dengan parameter FIXED (Fase 1) di rolling window validasi
    yang tidak overlap satu sama lain. Tidak ada grid search per fold —
    tujuannya mengukur konsistensi edge parameter Fase 1 lintas waktu.

SKEMA WINDOW:
    Dataset  : 14 bulan (Jun 2025 - Jul 2026)
    Kalibrasi: 3 bulan (tidak dipakai untuk tuning — hanya referensi konteks)
    Validasi : 1 bulan (langsung setelah window kalibrasi)
    Geser    : 1 bulan per fold
    ~10 fold total

    Fold 1: Calib Jun-Aug 2025  | Val Sep 2025
    Fold 2: Calib Jul-Sep 2025  | Val Oct 2025
    ...
    Fold 10: Calib Apr-Jun 2026 | Val Jul 2026

PARAMETER FIXED (Fase 1):
    atr_multiplier = 0.9
    swing_lookback = 15
    swing_wing     = 3
    rrr_min        = 1.3

REUSE:
    run_fast_backtest() dari scripts/run_param_sweep.py
    filter_period() dari scripts/run_oos_validation.py
    (tidak ada logika backtest yang ditulis ulang)
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import (
    merge_h1_to_m5,
    validate_no_lookahead,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    DEFAULT_SPREAD_PTS,
)
from scripts.run_param_sweep import run_fast_backtest
from scripts.run_oos_validation import filter_period


# =============================================================================
# PARAMETER FASE 1 — FIXED, tidak diubah
# =============================================================================
FASE1_PARAMS = {
    "atr_mult" : 0.9,
    "lookback" : 15,
    "wing"     : 3,
    "rrr_min"  : 1.3,
}

# =============================================================================
# GENERATE WINDOW FOLD
# =============================================================================

def generate_folds(
    data_start     : str,
    data_end       : str,
    calib_months   : int = 3,
    val_months     : int = 1,
) -> list:
    """
    Generate rolling window folds untuk walk-forward testing.

    Setiap fold terdiri dari:
        calib_start → calib_end  (3 bulan)
        val_start   → val_end    (1 bulan, langsung setelah calib)

    Window validasi antar fold TIDAK overlap satu sama lain.
    Window kalibrasi antar fold BOLEH overlap (ini yang dimaksud
    "rolling" / overlapping calibration window).

    Parameter:
        data_start   : 'YYYY-MM-DD' — awal dataset keseluruhan
        data_end     : 'YYYY-MM-DD' — akhir dataset keseluruhan
        calib_months : Lama window kalibrasi dalam bulan
        val_months   : Lama window validasi dalam bulan

    Return:
        List of dict: [
            {
                "fold"        : int,
                "calib_start" : str,
                "calib_end"   : str,
                "val_start"   : str,
                "val_end"     : str,
            },
            ...
        ]
    """
    import calendar

    # Parse tanggal awal/akhir
    start = pd.Timestamp(data_start, tz="UTC")
    end   = pd.Timestamp(data_end,   tz="UTC")

    folds = []
    fold_n = 1
    calib_start = start

    while True:
        # Hitung akhir window kalibrasi (akhir bulan ke-calib_months)
        calib_end_raw = calib_start + relativedelta(months=calib_months) - pd.Timedelta(days=1)

        # Awal dan akhir window validasi
        val_start_raw = calib_start + relativedelta(months=calib_months)
        val_end_raw   = val_start_raw + relativedelta(months=val_months) - pd.Timedelta(days=1)

        # Pastikan window validasi masih dalam dataset
        if val_start_raw > end:
            break

        # Clamp val_end ke akhir dataset jika melewati batas
        val_end_clamped = min(val_end_raw, end)

        folds.append({
            "fold"        : fold_n,
            "calib_start" : calib_start.strftime("%Y-%m-%d"),
            "calib_end"   : calib_end_raw.strftime("%Y-%m-%d"),
            "val_start"   : val_start_raw.strftime("%Y-%m-%d"),
            "val_end"     : val_end_clamped.strftime("%Y-%m-%d"),
        })

        # Geser 1 bulan ke depan
        calib_start = calib_start + relativedelta(months=val_months)
        fold_n += 1

    return folds


# =============================================================================
# JALANKAN SATU FOLD VALIDASI
# =============================================================================

def run_one_fold_validation(
    df_m5_full : pd.DataFrame,
    df_h1_full : pd.DataFrame,
    val_start  : str,
    val_end    : str,
    params     : dict,
    fold_label : str = "",
    volume_mode: str = "FILTER",
) -> dict:
    """
    Jalankan backtest parameter fixed di window validasi satu fold.

    Parameter:
        df_m5_full : DataFrame M5 mentah, seluruh dataset
        df_h1_full : DataFrame H1 mentah, seluruh dataset
        val_start  : 'YYYY-MM-DD' awal window validasi
        val_end    : 'YYYY-MM-DD' akhir window validasi
        params     : dict dengan key atr_mult, lookback, wing, rrr_min
        fold_label : Label untuk logging

    Return:
        dict ringkasan metrik validasi fold ini
    """
    # Filter ke window validasi
    df_m5_val = filter_period(df_m5_full, val_start, val_end)
    df_h1_val = filter_period(df_h1_full, val_start, val_end)

    if len(df_m5_val) < WARM_UP_CANDLES + 50:
        return {
            "status"           : "SKIP_TOO_FEW_CANDLES",
            "n_candles"        : len(df_m5_val),
            "total_trades"     : 0,
            "win_rate"         : None,
            "avg_rrr_realized" : None,
            "no_hit_rate"      : None,
            "total_pnl_net"    : None,
            "max_drawdown_net" : None,
        }

    # Hitung indikator & merge
    df_m5_val_ind = run_all_indicators(df_m5_val.copy())
    df_h1_val_ind = run_all_indicators(df_h1_val.copy())
    df_val_merged = merge_h1_to_m5(df_m5_val_ind, df_h1_val_ind)

    # Jalankan backtest dengan parameter fixed
    trades_df, summary = run_fast_backtest(
        df_m5_ind   = df_m5_val_ind,
        df_merged   = df_val_merged,
        atr_mult    = params["atr_mult"],
        lookback    = params["lookback"],
        wing        = params["wing"],
        rrr_min     = params["rrr_min"],
        volume_mode = volume_mode,
        spread_pts  = DEFAULT_SPREAD_PTS,
        max_candles = MAX_FORWARD_CANDLES,
        warm_up     = WARM_UP_CANDLES,
    )

    wr  = summary.get("win_rate")
    nhr = summary.get("no_hit_rate")

    return {
        "status"           : "OK",
        "n_candles"        : len(df_m5_val),
        "total_trades"     : summary.get("total_trades", 0),
        "win_rate"         : round(wr * 100, 2)  if wr  is not None else None,
        "avg_rrr_realized" : summary.get("avg_rrr_realized"),
        "no_hit_rate"      : round(nhr * 100, 2) if nhr is not None else None,
        "total_pnl_net"    : summary.get("total_pnl_net"),
        "max_drawdown_net" : summary.get("max_drawdown_net"),
        "tp_count"         : summary.get("tp_count", 0),
        "sl_count"         : summary.get("sl_count", 0),
        "no_hit_count"     : summary.get("no_hit_count", 0),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Testing Fase 2.3 — Parameter Fase 1 Fixed",
    )
    parser.add_argument(
        "--data-m5",
        default=os.path.join(ROOT_DIR, "data", "historical",
                             "XAUUSD_M5_2025-06-01_2026-07-25.csv"),
        help="Path ke file M5 extended dataset",
    )
    parser.add_argument(
        "--data-h1",
        default=os.path.join(ROOT_DIR, "data", "historical",
                             "XAUUSD_H1_2025-06-01_2026-07-25.csv"),
        help="Path ke file H1 extended dataset",
    )
    parser.add_argument(
        "--calib-months", type=int, default=3,
        help="Lama window kalibrasi dalam bulan (default: 3)",
    )
    parser.add_argument(
        "--val-months", type=int, default=1,
        help="Lama window validasi dalam bulan (default: 1)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  WALK-FORWARD TESTING — FASE 2.3")
    print("  Parameter Fase 1 FIXED (tanpa grid search per fold)")
    print("=" * 70)
    print(f"\n  Parameter Fase 1 fixed:")
    for k, v in FASE1_PARAMS.items():
        print(f"    {k:<14} = {v}")
    print(f"\n  Window kalibrasi : {args.calib_months} bulan (referensi)")
    print(f"  Window validasi  : {args.val_months} bulan (tidak overlap antar fold)")

    # Load data
    print(f"\n-> Loading data dari:")
    print(f"   M5: {args.data_m5}")
    print(f"   H1: {args.data_h1}")

    if not os.path.exists(args.data_m5):
        print(f"\nERROR: File M5 tidak ditemukan: {args.data_m5}")
        print("Jalankan dulu: python scripts/fetch_extended_data.py")
        sys.exit(1)

    if not os.path.exists(args.data_h1):
        print(f"\nERROR: File H1 tidak ditemukan: {args.data_h1}")
        sys.exit(1)

    df_m5_full = load_candles_csv(args.data_m5)
    df_h1_full = load_candles_csv(args.data_h1)

    data_start = df_m5_full.index[0].strftime("%Y-%m-%d")
    data_end   = df_m5_full.index[-1].strftime("%Y-%m-%d")

    print(f"\n   M5: {len(df_m5_full):,} candle ({data_start} -> {data_end})")
    print(f"   H1: {len(df_h1_full):,} candle")

    # Validasi no-lookahead
    print(f"\n-> Validasi zero look-ahead...")
    val_check = validate_no_lookahead(df_m5_full, n_samples=5)
    if not val_check["passed"]:
        raise RuntimeError(f"Look-ahead validation GAGAL!\n{val_check['message']}")
    print(f"   {val_check['message']}")

    # Generate folds
    folds = generate_folds(
        data_start   = data_start,
        data_end     = data_end,
        calib_months = args.calib_months,
        val_months   = args.val_months,
    )

    print(f"\n-> {len(folds)} fold dihasilkan:")
    print(f"   {'Fold':<5} {'Calib':<24} {'Validasi':<24}")
    print(f"   {'-'*5} {'-'*24} {'-'*24}")
    for f in folds:
        print(f"   {f['fold']:<5} {f['calib_start']} -> {f['calib_end']}  "
              f"{f['val_start']} -> {f['val_end']}")

    # ─── JALANKAN SEMUA FOLD ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EKSEKUSI WALK-FORWARD ({len(folds)} fold × Fase 1 fixed)")
    print(f"{'='*70}")

    all_results = []
    t_total = time.time()

    for fold in folds:
        fold_n    = fold["fold"]
        val_start = fold["val_start"]
        val_end   = fold["val_end"]

        print(f"\n[Fold {fold_n}/{len(folds)}] Validasi: {val_start} -> {val_end}")
        t0 = time.time()

        result = run_one_fold_validation(
            df_m5_full = df_m5_full,
            df_h1_full = df_h1_full,
            val_start  = val_start,
            val_end    = val_end,
            params     = FASE1_PARAMS,
            fold_label = f"Fold {fold_n}",
        )

        elapsed = time.time() - t0
        status  = result["status"]

        if status == "SKIP_TOO_FEW_CANDLES":
            print(f"   SKIP — terlalu sedikit candle ({result['n_candles']})")
        else:
            wr_str  = f"{result['win_rate']:.1f}%"   if result['win_rate']   is not None else "N/A"
            rrr_str = f"{result['avg_rrr_realized']:+.4f}" if result['avg_rrr_realized'] is not None else "N/A"
            pnl_str = f"{result['total_pnl_net']:+.2f}"   if result['total_pnl_net']    is not None else "N/A"
            dd_str  = f"{result['max_drawdown_net']:+.2f}" if result['max_drawdown_net'] is not None else "N/A"
            print(f"   trades={result['total_trades']}, win={wr_str}, "
                  f"rrr={rrr_str}, pnl={pnl_str}, dd={dd_str} [{elapsed:.1f}s]")

        record = {
            "fold"             : fold_n,
            "calib_start"      : fold["calib_start"],
            "calib_end"        : fold["calib_end"],
            "val_start"        : val_start,
            "val_end"          : val_end,
            "n_candles_val"    : result["n_candles"],
            "status"           : status,
            "total_trades"     : result.get("total_trades", 0),
            "tp_count"         : result.get("tp_count", 0),
            "sl_count"         : result.get("sl_count", 0),
            "no_hit_count"     : result.get("no_hit_count", 0),
            "win_rate_pct"     : result.get("win_rate"),
            "avg_rrr_realized" : result.get("avg_rrr_realized"),
            "no_hit_rate_pct"  : result.get("no_hit_rate"),
            "total_pnl_net"    : result.get("total_pnl_net"),
            "max_drawdown_net" : result.get("max_drawdown_net"),
            "elapsed_s"        : round(elapsed, 1),
        }
        all_results.append(record)

    total_elapsed = time.time() - t_total
    print(f"\n=> Semua fold selesai dalam {total_elapsed:.1f} detik.")

    # ─── TABEL PER-FOLD ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    valid_df   = results_df[results_df["status"] == "OK"].copy()

    print(f"\n{'='*90}")
    print(f"  TABEL HASIL PER-FOLD — Parameter Fase 1 Fixed")
    print(f"  (atr=0.9, lookback=15, wing=3, rrr_min=1.3)")
    print(f"{'='*90}")
    header = (f"  {'Fold':<5} {'Val Period':<24} {'Trades':<7} {'WinRate':>7} "
              f"{'AvgRRR':>8} {'NoHit%':>7} {'PnL Net':>10} {'MaxDD':>9}")
    print(header)
    print(f"  {'-'*5} {'-'*24} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*10} {'-'*9}")

    for _, row in results_df.iterrows():
        if row["status"] != "OK":
            print(f"  {int(row['fold']):<5} {row['val_start']} -> {row['val_end']:<12} SKIP")
            continue

        wr_s  = f"{row['win_rate_pct']:.1f}%" if row['win_rate_pct']   is not None else "N/A"
        rrr_s = f"{row['avg_rrr_realized']:+.4f}" if row['avg_rrr_realized'] is not None else "N/A"
        nhr_s = f"{row['no_hit_rate_pct']:.1f}%" if row['no_hit_rate_pct'] is not None else "N/A"
        pnl_s = f"{row['total_pnl_net']:+.2f}"  if row['total_pnl_net']    is not None else "N/A"
        dd_s  = f"{row['max_drawdown_net']:+.2f}" if row['max_drawdown_net'] is not None else "N/A"
        pnl_flag = " +" if (row['total_pnl_net'] or 0) > 0 else " -"

        print(f"  {int(row['fold']):<5} {row['val_start']} -> {row['val_end']:<12} "
              f"{int(row['total_trades']):<7} {wr_s:>7} {rrr_s:>8} {nhr_s:>7} "
              f"{pnl_s:>10} {dd_s:>9}{pnl_flag}")

    # ─── AGREGAT LINTAS FOLD ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  AGREGAT LINTAS FOLD — Parameter Fase 1 Fixed")
    print(f"{'='*70}")

    if not valid_df.empty:
        n_valid_folds  = len(valid_df)
        n_pnl_positive = int((valid_df["total_pnl_net"] > 0).sum())
        n_pnl_negative = int((valid_df["total_pnl_net"] <= 0).sum())

        rrr_vals = valid_df["avg_rrr_realized"].dropna()
        pnl_vals = valid_df["total_pnl_net"].dropna()
        wr_vals  = valid_df["win_rate_pct"].dropna()

        print(f"\n  Fold valid (status=OK)  : {n_valid_folds}")
        print(f"  Fold PnL positif        : {n_pnl_positive} / {n_valid_folds} "
              f"({'%.0f' % (n_pnl_positive/n_valid_folds*100)}%)")
        print(f"  Fold PnL negatif        : {n_pnl_negative} / {n_valid_folds}")
        print()
        print(f"  {'Metrik':<26} {'Rata-rata':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'Win Rate (%)':<26} {wr_vals.mean():>10.2f} {wr_vals.std():>10.2f} "
              f"{wr_vals.min():>10.2f} {wr_vals.max():>10.2f}")
        print(f"  {'Avg RRR Realized':<26} {rrr_vals.mean():>+10.4f} {rrr_vals.std():>10.4f} "
              f"{rrr_vals.min():>+10.4f} {rrr_vals.max():>+10.4f}")
        print(f"  {'Total PnL Net ($)':<26} {pnl_vals.mean():>+10.2f} {pnl_vals.std():>10.2f} "
              f"{pnl_vals.min():>+10.2f} {pnl_vals.max():>+10.2f}")

        # ─── ANALISIS POLA TEMPORAL ───────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  ANALISIS POLA TEMPORAL")
        print(f"{'='*70}")

        # Tandai fold positif vs negatif
        valid_df = valid_df.copy()
        valid_df["val_month"] = pd.to_datetime(valid_df["val_start"]).dt.month
        valid_df["val_year"]  = pd.to_datetime(valid_df["val_start"]).dt.year
        valid_df["pnl_sign"]  = valid_df["total_pnl_net"].apply(
            lambda x: "POSITIF" if x > 0 else "NEGATIF"
        )

        print(f"\n  Fold-fold yang PnL NEGATIF:")
        neg_df = valid_df[valid_df["pnl_sign"] == "NEGATIF"]
        if neg_df.empty:
            print("  (tidak ada — semua fold positif)")
        else:
            for _, r in neg_df.iterrows():
                print(f"    Fold {int(r['fold'])}: {r['val_start']} -> {r['val_end']} | "
                      f"pnl={r['total_pnl_net']:+.2f}, wr={r['win_rate_pct']:.1f}%, "
                      f"rrr={r['avg_rrr_realized']:+.4f}")

        print(f"\n  Fold-fold yang PnL POSITIF:")
        pos_df = valid_df[valid_df["pnl_sign"] == "POSITIF"]
        if pos_df.empty:
            print("  (tidak ada — semua fold negatif)")
        else:
            for _, r in pos_df.iterrows():
                print(f"    Fold {int(r['fold'])}: {r['val_start']} -> {r['val_end']} | "
                      f"pnl={r['total_pnl_net']:+.2f}, wr={r['win_rate_pct']:.1f}%, "
                      f"rrr={r['avg_rrr_realized']:+.4f}")

        # Apakah fold negatif berkerumun di periode tertentu?
        if not neg_df.empty:
            neg_months = neg_df["val_month"].tolist()
            print(f"\n  Bulan validasi yang negatif: "
                  f"{[f'{y}/{m:02d}' for y, m in zip(neg_df['val_year'], neg_df['val_month'])]}")

            # Cek korelasi dengan trades count (proxy volatilitas — lebih sedikit = sideways)
            print(f"\n  Rata-rata trades di fold POSITIF vs NEGATIF:")
            print(f"    Positif: {pos_df['total_trades'].mean():.1f} trades/fold")
            print(f"    Negatif: {neg_df['total_trades'].mean():.1f} trades/fold")

    # ─── SIMPAN HASIL ─────────────────────────────────────────────────────────
    out_dir = os.path.join(ROOT_DIR, "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "walk_forward_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Hasil lengkap disimpan: {out_csv}")

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD SELESAI")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
