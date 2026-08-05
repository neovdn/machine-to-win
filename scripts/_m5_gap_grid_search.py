"""
scripts/_m5_gap_grid_search.py
===============================
DIAGNOSTIK FASE 1b -- Threshold Gap EMA M5 (bukan H1)
=======================================================
TUJUAN:
    1. Ukur trade-off frekuensi vs kualitas kalau threshold min_ema_gap_pct
       untuk M5 (di detect_trend(), saat ini default 0.05%) diturunkan.
       Sama metodologinya dengan _h1_gap_grid_search.py yang sudah pernah
       dipakai untuk kalibrasi H1 -- sekarang giliran M5.

    2. Konfirmasi hipotesis "late confirmation + RSI compounding":
       ambil SEMUA trade yang lolos sistem SEKARANG (threshold 0.05%, default),
       lalu lihat distribusi ema_gap_pct dan rsi_at_entry di titik entry-nya.
       Kalau hipotesis benar, kita akan lihat entry ter-cluster di gap yang
       sudah lumayan lebar (bukan di dekat 0.05%) dan RSI sudah condong ke
       salah satu sisi (bukan di tengah 40-60).

CATATAN PENTING:
    Ini TIDAK mengubah engine/indicators.py atau engine/rule_engine.py.
    Threshold M5 hanya di-override secara lokal di script ini untuk eksplorasi.

CARA PAKAI:
    python scripts/_m5_gap_grid_search.py
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import (
    calculate_ema, calculate_rsi, calculate_atr, calculate_volume_ratio,
    detect_trend,
)
from engine.backtester import merge_h1_to_m5, DEFAULT_SPREAD_PTS, WARM_UP_CANDLES, MAX_FORWARD_CANDLES
from scripts.run_param_sweep import run_fast_backtest


# Threshold M5 yang diuji -- termasuk 0.05% (baseline saat ini) sebagai pembanding
M5_GAP_CANDIDATES = [0.02, 0.03, 0.05, 0.08, 0.10]

RISK_PARAMS = {"atr_mult": 0.9, "lookback": 15, "wing": 3, "rrr_min": 1.3}


def build_m5_indicators_raw(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Hitung EMA/RSI/ATR/volume TANPA memanggil detect_trend() -- supaya
    threshold gap bisa di-override berkali-kali tanpa hitung ulang EMA/RSI/ATR."""
    df = calculate_ema(df_m5.copy(), periods=[9, 21])
    df = calculate_rsi(df, period=14)
    df = calculate_atr(df, period=14)
    df = calculate_volume_ratio(df)
    return df


def main():
    print("=" * 78)
    print("  DIAGNOSTIK FASE 1b -- THRESHOLD GAP EMA M5 + DISTRIBUSI ENTRY")
    print("=" * 78)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"File tidak ditemukan:\n  {m5_path}\n  {h1_path}")
        sys.exit(1)

    print("\n-> Loading data...")
    df_m5_raw = load_candles_csv(m5_path)
    df_h1_raw = load_candles_csv(h1_path)

    print("-> Menghitung EMA/RSI/ATR/Volume M5 (satu kali, threshold trend belum diterapkan)...")
    df_m5_base = build_m5_indicators_raw(df_m5_raw)

    print("-> Menghitung indikator H1 penuh (termasuk bias_h1, sekali saja)...")
    from engine.indicators import run_all_indicators
    df_h1_ind = run_all_indicators(df_h1_raw.copy())

    # =========================================================================
    # BAGIAN A: GRID SEARCH THRESHOLD M5
    # =========================================================================
    print("\n" + "=" * 78)
    print("  BAGIAN A: TRADE-OFF THRESHOLD GAP M5 (frekuensi vs kualitas)")
    print("=" * 78)
    print(f"\n  {'M5 Gap':>8} {'Trades':>8} {'WinRate':>9} {'AvgRRR':>9} {'NoHit%':>8} {'PnL Net':>10} {'MaxDD':>9}")
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*10} {'-'*9}")

    results = []
    all_trades_by_threshold = {}

    for gap in M5_GAP_CANDIDATES:
        df_m5_thr = detect_trend(df_m5_base.copy(), min_ema_gap_pct=gap)
        df_merged = merge_h1_to_m5(df_m5_thr, df_h1_ind)  # H1 threshold tetap default 0.02

        trades_df, summary = run_fast_backtest(
            df_m5_ind = df_m5_thr,
            df_merged = df_merged,
            atr_mult  = RISK_PARAMS["atr_mult"],
            lookback  = RISK_PARAMS["lookback"],
            wing      = RISK_PARAMS["wing"],
            rrr_min   = RISK_PARAMS["rrr_min"],
            spread_pts  = DEFAULT_SPREAD_PTS,
            max_candles = MAX_FORWARD_CANDLES,
            warm_up     = WARM_UP_CANDLES,
        )
        all_trades_by_threshold[gap] = trades_df

        n    = summary.get("total_trades", 0)
        wr   = summary.get("win_rate")
        rrr  = summary.get("avg_rrr_realized")
        nhr  = summary.get("no_hit_rate")
        pnl  = summary.get("total_pnl_net")
        mdd  = summary.get("max_drawdown_net")

        wr_s  = f"{wr*100:.1f}%" if wr is not None else "N/A"
        rrr_s = f"{rrr:+.4f}" if rrr is not None else "N/A"
        nhr_s = f"{(nhr or 0)*100:.1f}%"
        pnl_s = f"{pnl:+.1f}" if pnl is not None else "N/A"
        mdd_s = f"{mdd:+.1f}" if mdd is not None else "N/A"

        flag = "  <- baseline saat ini" if gap == 0.05 else ""
        print(f"  {gap:>7.2f}% {n:>8,} {wr_s:>9} {rrr_s:>9} {nhr_s:>8} {pnl_s:>10} {mdd_s:>9}{flag}")

        results.append({"gap": gap, "total_trades": n, "win_rate": wr,
                         "avg_rrr": rrr, "no_hit_rate": nhr, "pnl_net": pnl, "max_dd": mdd})

    # =========================================================================
    # BAGIAN B: DISTRIBUSI ema_gap_pct DAN RSI DI TITIK ENTRY (baseline 0.05%)
    # =========================================================================
    print("\n" + "=" * 78)
    print("  BAGIAN B: DISTRIBUSI ema_gap_pct & RSI DI ENTRY AKTUAL (threshold 0.05%, baseline)")
    print("=" * 78)

    baseline_trades = all_trades_by_threshold.get(0.05)
    if baseline_trades is None or baseline_trades.empty:
        print("\n  Tidak ada trade di threshold baseline -- tidak bisa analisis distribusi.")
    else:
        gap_vals = baseline_trades["ema_gap_pct"].abs()
        rsi_vals = baseline_trades["rsi_at_entry"]

        print(f"\n  N trade = {len(baseline_trades)}")
        print(f"\n  Distribusi |ema_gap_pct| di titik entry:")
        print(f"    min={gap_vals.min():.4f}%  Q25={gap_vals.quantile(0.25):.4f}%  "
              f"median={gap_vals.median():.4f}%  Q75={gap_vals.quantile(0.75):.4f}%  max={gap_vals.max():.4f}%")

        print(f"\n  Berapa % trade yang gap-nya masih di bawah 0.15% (RSI_STRONG_TREND_THRESHOLD)?")
        below_015 = (gap_vals < 0.15).mean() * 100
        print(f"    {below_015:.1f}% trade entry dengan gap < 0.15% -> RSI filter MASIH mode TREND_LEMAH (veto ketat)")

        print(f"\n  Distribusi RSI di titik entry:")
        print(f"    min={rsi_vals.min():.1f}  Q25={rsi_vals.quantile(0.25):.1f}  "
              f"median={rsi_vals.median():.1f}  Q75={rsi_vals.quantile(0.75):.1f}  max={rsi_vals.max():.1f}")

        near_extreme = ((rsi_vals >= 60) | (rsi_vals <= 40)).mean() * 100
        print(f"\n  Berapa % entry dengan RSI sudah di luar zona netral (40-60)? {near_extreme:.1f}%")
        print(f"  (Ini proxy: makin tinggi angka ini, makin sering entry terjadi setelah momentum")
        print(f"   sudah lumayan matang -- mendukung hipotesis 'entry telat'.)")

    # =========================================================================
    # RINGKASAN
    # =========================================================================
    print("\n" + "=" * 78)
    print("  CARA BACA HASIL")
    print("=" * 78)
    print("""
  Bagian A -- kalau threshold diturunkan (mis. 0.02% atau 0.03%):
    - total_trades naik (frekuensi lebih tinggi, sesuai dugaan)
    - PERIKSA: apakah win_rate/avg_rrr TURUN signifikan, atau tetap sebanding?
      Kalau tetap sebanding/lebih baik -> gap 0.05% memang terlalu ketat, aman diturunkan.
      Kalau win_rate/avg_rrr anjlok -> gap rendah membawa banyak noise/false signal juga.

  Bagian B -- kalau below_015% tinggi (misal >70%) DAN near_extreme tinggi (misal >50%):
    -> Ini KONFIRMASI KUAT hipotesis compounding: sebagian besar entry sistem sekarang
       terjadi saat RSI sudah condong ekstrem TAPI gap belum cukup lebar untuk masuk
       mode TREND_KUAT -- artinya entry yang LOLOS adalah entry yang situasinya paling
       "berisiko sempit" (gap kecil tapi RSI sudah jauh), sementara entry yang lebih sehat
       (RSI netral, gap masih tipis, baru mulai pullback) DIBUANG karena gap belum cukup.

  Kirim balik tabel Bagian A dan angka-angka Bagian B untuk kita putuskan bersama
  apakah solusinya: (1) turunkan threshold M5, (2) ganti mekanisme trigger M5 jadi
  pullback-based (lihat _diag_pullback_candidates.py), atau (3) kombinasi keduanya.
""")


if __name__ == "__main__":
    main()