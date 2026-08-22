"""
scripts/run_backtest_regime.py
==============================
Script CLI untuk menjalankan backtest regime-based historis XAUUSD M5/M15/H1.

CARA PAKAI:
    # Jalankan dengan dataset default (path sudah tersedia di data/historical/)
    python scripts/run_backtest_regime.py

    # Custom path CSV
    python scripts/run_backtest_regime.py \\
        --m5-file data/historical/XAUUSD_M5_2025-06-01_2026-07-25.csv \\
        --m15-file data/historical/XAUUSD_M15_2025-06-01_2026-07-25.csv \\
        --h1-file data/historical/XAUUSD_H1_2025-06-01_2026-07-25.csv

    # Filter subset tanggal (untuk smoke test lebih cepat)
    python scripts/run_backtest_regime.py \\
        --start 2025-06-01 --end 2025-09-30

    # Semua opsi:
    python scripts/run_backtest_regime.py --help

OUTPUT:
    1. Laporan tersegmentasi di terminal:
       - Overall summary
       - Per regime (TRENDING/RANGING/BREAKOUT_TRANSITION/CHOP)
       - Per strategi (TREND_FOLLOWING/RANGE_REVERSAL/BREAKOUT_RETEST)
       - Per sesi (LONDON_NY/LONDON/ASIA/NY_ONLY)
       - Status rekonsiliasi
    2. CSV hasil trade di data/backtest_results/
       backtest_regime_XAUUSD_YYYYMMDD_HHMMSS.csv

CATATAN:
    Modul ini tidak mengubah engine/backtester.py atau modul manapun dari Fase 12-19.
    Fase 20 (News Filter) di-skip atas keputusan pemilik proyek.
"""

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

# ── Tambahkan root folder ke Python path ────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.backtester_regime import (
    run_regime_backtest,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    DEFAULT_SPREAD_PTS,
    REGIME_BREAKOUT_GRACE_CANDLES,
)


# =============================================================================
# PATH DEFAULT DATA
# =============================================================================

DEFAULT_M5_FILE  = os.path.join(ROOT_DIR, "data", "historical",
                                 "XAUUSD_M5_2025-06-01_2026-07-25.csv")
DEFAULT_M15_FILE = os.path.join(ROOT_DIR, "data", "historical",
                                 "XAUUSD_M15_2025-06-01_2026-07-25.csv")
DEFAULT_H1_FILE  = os.path.join(ROOT_DIR, "data", "historical",
                                 "XAUUSD_H1_2025-06-01_2026-07-25.csv")


# =============================================================================
# ARGPARSE
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run_backtest_regime.py",
        description = "Backtest regime-based engine XAUUSD M5/M15/H1 -- tanpa AI/ML.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Contoh:
  # Jalankan dengan dataset default (14 bulan)
  python scripts/run_backtest_regime.py

  # Subset tanggal untuk smoke test cepat
  python scripts/run_backtest_regime.py --start 2025-06-01 --end 2025-09-30

  # Custom file CSV
  python scripts/run_backtest_regime.py \\
      --m5-file data/historical/XAUUSD_M5_2025-06-01_2026-07-25.csv \\
      --m15-file data/historical/XAUUSD_M15_2025-06-01_2026-07-25.csv \\
      --h1-file data/historical/XAUUSD_H1_2025-06-01_2026-07-25.csv
        """,
    )

    # ── Sumber data ────────────────────────────────────────────────────────────
    src_group = parser.add_argument_group("Sumber Data")
    src_group.add_argument(
        "--m5-file",
        metavar = "PATH",
        default = DEFAULT_M5_FILE,
        help    = f"Path CSV M5. (default: {DEFAULT_M5_FILE})",
    )
    src_group.add_argument(
        "--m15-file",
        metavar = "PATH",
        default = DEFAULT_M15_FILE,
        help    = f"Path CSV M15. (default: {DEFAULT_M15_FILE})",
    )
    src_group.add_argument(
        "--h1-file",
        metavar = "PATH",
        default = DEFAULT_H1_FILE,
        help    = f"Path CSV H1. (default: {DEFAULT_H1_FILE})",
    )
    src_group.add_argument(
        "--start",
        metavar = "YYYY-MM-DD",
        default = None,
        help    = "Tanggal mulai backtest (filter dari CSV). "
                  "Contoh: 2025-06-01",
    )
    src_group.add_argument(
        "--end",
        metavar = "YYYY-MM-DD",
        default = None,
        help    = "Tanggal akhir backtest (filter dari CSV). "
                  "Contoh: 2025-12-31",
    )

    # ── Parameter backtest ─────────────────────────────────────────────────────
    bt_group = parser.add_argument_group("Parameter Backtest")
    bt_group.add_argument(
        "--max-candles",
        type    = int,
        default = MAX_FORWARD_CANDLES,
        metavar = "N",
        help    = f"Batas candle forward per trade. (default: {MAX_FORWARD_CANDLES})",
    )
    bt_group.add_argument(
        "--warm-up",
        type    = int,
        default = WARM_UP_CANDLES,
        metavar = "N",
        help    = f"Candle warm-up untuk indikator. (default: {WARM_UP_CANDLES})",
    )
    bt_group.add_argument(
        "--rrr-min",
        type    = float,
        default = None,
        metavar = "RRR",
        help    = "Minimum Risk-to-Reward Ratio. None = pakai default calculate_sl_tp().",
    )
    bt_group.add_argument(
        "--grace-candles",
        type    = int,
        default = REGIME_BREAKOUT_GRACE_CANDLES,
        metavar = "N",
        help    = f"Grace window candle M15 untuk BREAKOUT_TRANSITION. "
                  f"(default: {REGIME_BREAKOUT_GRACE_CANDLES})",
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output",
        metavar = "DIR",
        default = os.path.join(ROOT_DIR, "data", "backtest_results"),
        help    = "Folder tujuan output CSV hasil trade. "
                  "(default: data/backtest_results/)",
    )
    out_group.add_argument(
        "--no-save-csv",
        action  = "store_true",
        default = False,
        help    = "Jangan simpan CSV hasil trade.",
    )

    return parser


# =============================================================================
# LOAD DATA DARI CSV
# =============================================================================

def _load_data(args) -> tuple:
    """
    Load M5, M15, H1 dari file CSV dan terapkan filter tanggal jika ada.

    Return:
        (df_m5, df_m15, df_h1) atau (None, None, None) jika gagal
    """
    # Cek file tersedia
    for label, path in [("M5", args.m5_file), ("M15", args.m15_file), ("H1", args.h1_file)]:
        if not os.path.exists(path):
            print(f"❌ File {label} tidak ditemukan: {path}")
            return None, None, None

    print(f"→ Loading M5  dari: {args.m5_file}")
    df_m5 = load_candles_csv(args.m5_file)
    if df_m5 is None or df_m5.empty:
        print("❌ Gagal load M5")
        return None, None, None

    print(f"→ Loading M15 dari: {args.m15_file}")
    df_m15 = load_candles_csv(args.m15_file)
    if df_m15 is None or df_m15.empty:
        print("❌ Gagal load M15")
        return None, None, None

    print(f"→ Loading H1  dari: {args.h1_file}")
    df_h1 = load_candles_csv(args.h1_file)
    if df_h1 is None or df_h1.empty:
        print("❌ Gagal load H1")
        return None, None, None

    # Filter tanggal jika diminta
    if args.start or args.end:
        try:
            if args.start:
                dt_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                df_m5  = df_m5[df_m5.index >= dt_start]
                df_m15 = df_m15[df_m15.index >= dt_start]
                df_h1  = df_h1[df_h1.index >= dt_start]
            if args.end:
                dt_end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                df_m5  = df_m5[df_m5.index <= dt_end]
                df_m15 = df_m15[df_m15.index <= dt_end]
                df_h1  = df_h1[df_h1.index <= dt_end]
            print(f"   Filter tanggal diterapkan:")
            print(f"   M5 : {len(df_m5):,} candle ({df_m5.index[0]} → {df_m5.index[-1]})")
            print(f"   M15: {len(df_m15):,} candle ({df_m15.index[0]} → {df_m15.index[-1]})")
            print(f"   H1 : {len(df_h1):,} candle ({df_h1.index[0]} → {df_h1.index[-1]})")
        except (ValueError, IndexError) as e:
            print(f"⚠️  Filter tanggal gagal: {e} — memakai seluruh data CSV")

    return df_m5, df_m15, df_h1


# =============================================================================
# PRINT LAPORAN TERSEGMENTASI
# =============================================================================

def _fmt_wr(win_rate) -> str:
    """Format win rate sebagai persentase dengan bar visual."""
    if win_rate is None:
        return "N/A"
    pct = win_rate * 100
    bar_len = int(pct / 5)
    bar = "=" * bar_len + "-" * (20 - bar_len)
    return f"{pct:6.1f}%  [{bar}]"


def _print_segment_summary(label: str, summary: dict) -> None:
    """Cetak ringkasan satu segmen (overall / satu regime / satu strategi)."""
    total  = summary.get("total_trades", 0)
    tp_n   = summary.get("tp_count", 0)
    sl_n   = summary.get("sl_count", 0)
    no_n   = summary.get("no_hit_count", 0)
    wr     = summary.get("win_rate")
    pf     = summary.get("profit_factor")
    exp    = summary.get("expectancy")
    rrr    = summary.get("avg_rrr_realized")
    dd     = summary.get("max_drawdown_net", 0)
    pnl    = summary.get("total_pnl_net", 0)

    print(f"\n  [{label}]")
    print(f"    Trade    : {total:>6,}  (TP={tp_n}, SL={sl_n}, NO_HIT={no_n})")
    print(f"    Win Rate : {_fmt_wr(wr)}")
    if rrr is not None:
        sign = "+" if rrr >= 0 else ""
        print(f"    Avg RRR  : {sign}{rrr:.2f}R")
    if pf is not None:
        print(f"    Profit F : {pf:.2f}")
    elif pf is None and total > 0:
        print(f"    Profit F : N/A (semua trade menang atau tidak ada SL)")
    if exp is not None:
        sign = "+" if exp >= 0 else ""
        print(f"    Expectancy: {sign}{exp:.2f} pts/trade")
    print(f"    PnL Net  : {pnl:>+10.2f} pts")
    print(f"    Max DD   : {dd:>+10.2f} pts")


def _print_full_report(
    segmented_summary: dict,
    trades_df,
    output_csv_path: str,
) -> None:
    """
    Cetak laporan tersegmentasi lengkap ke terminal.
    """
    print()
    print("=" * 70)
    print("  RINGKASAN BACKTEST REGIME-BASED — XAUUSD M5/M15/H1")
    print("=" * 70)

    recon = segmented_summary.get("reconciliation", {})
    total = recon.get("total_trades", 0)

    if total == 0:
        print("\n  WARNING: Tidak ada trade yang ditemukan.")
        print("  Kemungkinan: warm-up terlalu besar, data terlalu sedikit,")
        print("  atau tidak ada sinyal yang memenuhi syarat dalam periode ini.")
        return

    # ── OVERALL ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  OVERALL")
    print("=" * 40)
    _print_segment_summary("OVERALL", segmented_summary.get("overall", {}))

    # ── PER REGIME ────────────────────────────────────────────────────────────
    per_regime = segmented_summary.get("per_regime", {})
    if per_regime:
        print("\n" + "=" * 40)
        print("  BREAKDOWN PER REGIME")
        print("=" * 40)
        for regime_name, summary in sorted(per_regime.items()):
            _print_segment_summary(regime_name, summary)

    # ── PER STRATEGI ──────────────────────────────────────────────────────────
    per_strategy = segmented_summary.get("per_strategy", {})
    if per_strategy:
        print("\n" + "=" * 40)
        print("  BREAKDOWN PER STRATEGI")
        print("=" * 40)
        for strat_name, summary in sorted(per_strategy.items()):
            _print_segment_summary(strat_name, summary)

    # ── PER SESI ──────────────────────────────────────────────────────────────
    per_session = segmented_summary.get("per_session", {})
    if per_session:
        print("\n" + "=" * 40)
        print("  BREAKDOWN PER SESI")
        print("=" * 40)
        for sess_name, summary in sorted(per_session.items()):
            _print_segment_summary(sess_name, summary)

    # ── DISTRIBUSI TRADE ──────────────────────────────────────────────────────
    if not trades_df.empty:
        print("\n" + "=" * 40)
        print("  DISTRIBUSI TRADE")
        print("=" * 40)
        if "regime" in trades_df.columns:
            print("\n  Per Regime:")
            for r, cnt in trades_df["regime"].value_counts().items():
                pct = cnt / total * 100
                print(f"    {str(r):25s}: {cnt:>5,} ({pct:.1f}%)")
        if "strategy" in trades_df.columns:
            print("\n  Per Strategi:")
            for s, cnt in trades_df["strategy"].value_counts().items():
                pct = cnt / total * 100
                print(f"    {str(s):25s}: {cnt:>5,} ({pct:.1f}%)")
        if "strategy_source" in trades_df.columns:
            print("\n  Per Source:")
            for src, cnt in trades_df["strategy_source"].value_counts().items():
                pct = cnt / total * 100
                print(f"    {str(src):25s}: {cnt:>5,} ({pct:.1f}%)")
        if "session" in trades_df.columns:
            print("\n  Per Sesi:")
            for sess, cnt in trades_df["session"].value_counts().items():
                pct = cnt / total * 100
                print(f"    {str(sess):25s}: {cnt:>5,} ({pct:.1f}%)")

    # ── STATUS REKONSILIASI ───────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  STATUS REKONSILIASI")
    print("=" * 40)
    reconciled = recon.get("reconciled", False)
    icon = "✅" if reconciled else "❌"
    print(f"\n  Status  : {icon} {'PASSED' if reconciled else 'FAILED'}")
    print(f"  Total   : {recon.get('total_trades', 0):,}")
    print(f"  ∑ Regime: {recon.get('sum_per_regime', 0):,}")
    print(f"  ∑ Strat : {recon.get('sum_per_strategy', 0):,}")
    print(f"  ∑ Sesi  : {recon.get('sum_per_session', 0):,}")
    print(f"  Ket     : {recon.get('keterangan', '')}")

    # ── OUTPUT FILE ───────────────────────────────────────────────────────────
    print(f"\n  OUTPUT:")
    print(f"  CSV hasil trade: {output_csv_path}")
    print(f"  Baris          : {len(trades_df):,} trade")
    print()
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # Normalize argparse underscore/hyphen
    args.no_save_csv  = getattr(args, "no_save_csv", False)
    args.max_candles  = getattr(args, "max_candles", MAX_FORWARD_CANDLES)
    args.warm_up      = getattr(args, "warm_up", WARM_UP_CANDLES)
    args.grace_candles = getattr(args, "grace_candles", REGIME_BREAKOUT_GRACE_CANDLES)
    args.m5_file      = getattr(args, "m5_file", DEFAULT_M5_FILE)
    args.m15_file     = getattr(args, "m15_file", DEFAULT_M15_FILE)
    args.h1_file      = getattr(args, "h1_file", DEFAULT_H1_FILE)

    print("=" * 70)
    print("  run_backtest_regime.py — XAUUSD Regime-Based Backtest CLI")
    print("=" * 70)
    print(f"  M5 file    : {args.m5_file}")
    print(f"  M15 file   : {args.m15_file}")
    print(f"  H1 file    : {args.h1_file}")
    print(f"  Max candles: {args.max_candles}")
    print(f"  Warm-up    : {args.warm_up} candle")
    print(f"  Grace M15  : {args.grace_candles} candle")
    print(f"  RRR min    : {args.rrr_min}")
    print(f"  Output dir : {args.output}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    df_m5, df_m15, df_h1 = _load_data(args)
    if df_m5 is None:
        sys.exit(1)
    print()

    # ── Jalankan backtest ─────────────────────────────────────────────────────
    try:
        trades_df, segmented_summary = run_regime_backtest(
            df_m5         = df_m5,
            df_m15        = df_m15,
            df_h1         = df_h1,
            warm_up       = args.warm_up,
            max_candles   = args.max_candles,
            rrr_min       = args.rrr_min,
            grace_candles = args.grace_candles,
            verbose       = True,
        )
    except Exception as e:
        print(f"\n❌ FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Simpan CSV ─────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename  = f"backtest_regime_XAUUSD_{timestamp_tag}.csv"
    out_path      = os.path.join(args.output, out_filename)

    if not args.no_save_csv and not trades_df.empty:
        trades_df.to_csv(out_path, index=False)
        print(f"\n✅ CSV disimpan: {out_path}")
    elif trades_df.empty:
        print("\n⚠️  Tidak ada trade — CSV tidak disimpan.")

    # ── Print laporan ─────────────────────────────────────────────────────────
    _print_full_report(segmented_summary, trades_df, out_path)


if __name__ == "__main__":
    main()
