"""
scripts/run_backtest.py
=======================
Script CLI untuk menjalankan backtest historis XAUUSD M5 - Rule-Based Engine.

CARA PAKAI:
    # Tarik data dari MT5, simpan ke CSV, jalankan backtest
    python scripts/run_backtest.py --start 2026-01-01 --end 2026-07-25

    # Jalankan ulang dari CSV cache (tanpa perlu MT5 aktif)
    python scripts/run_backtest.py \
        --source csv \
        --m5-file data/historical/XAUUSD_M5_2026-01-01_2026-07-25.csv \
        --h1-file data/historical/XAUUSD_H1_2026-01-01_2026-07-25.csv

    # Custom output folder dan max forward candles
    python scripts/run_backtest.py \
        --start 2026-01-01 --end 2026-07-25 \
        --max-candles 576 \
        --output data/backtest_results/run_01/

    # Semua opsi tersedia:
    python scripts/run_backtest.py --help

PARAMETER:
    --start       YYYY-MM-DD   Tanggal mulai backtest (inklusif)
    --end         YYYY-MM-DD   Tanggal akhir backtest (inklusif)
    --source      mt5/csv      Sumber data (default: mt5)
    --m5-file     PATH         Path CSV M5 (hanya jika --source csv)
    --h1-file     PATH         Path CSV H1 (hanya jika --source csv)
    --output      PATH         Folder output CSV hasil trade (default: data/backtest_results/)
    --max-candles INT          Batas candle forward per trade (default: 288 = ~24 jam)
    --warm-up     INT          Candle warm-up untuk indikator (default: 100)
    --symbol      STRING       Simbol MT5 (default dari .env: XAUUSD)
    --no-save-csv              Jangan simpan cache CSV saat source=mt5

OUTPUT:
    1. Ringkasan agregat di terminal
    2. CSV trade-by-trade di --output folder:
       backtest_XAUUSD_M5_YYYYMMDD_HHMMSS.csv

REPRODUCIBILITY:
    Saat source=mt5, data OHLC mentah otomatis disimpan ke data/historical/
    sebagai cache CSV. Run berikutnya bisa pakai --source csv --m5-file <path>
    untuk mendapatkan hasil yang identik tanpa perlu MT5 aktif.
"""

import sys
import os
import argparse
from datetime import datetime, timezone

# ── Tambahkan root folder ke Python path ────────────────────────────────────
# Script dijalankan dari root project (machine-to-win/), tapi pastikan
# Python bisa menemukan folder engine/ dengan benar.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import (
    initialize_mt5,
    shutdown_mt5,
    get_candles_range,
    save_candles_csv,
    load_candles_csv,
)
from engine.backtester import run_backtest, compute_summary


# =============================================================================
# ARGPARSE — Definisi Parameter CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run_backtest.py",
        description = "Backtest rule-based engine XAUUSD M5 -- tanpa AI/ML.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Contoh:
  # Pull dari MT5, simpan cache, jalankan backtest
  python scripts/run_backtest.py --start 2026-01-01 --end 2026-07-25

  # Pakai CSV cache (tanpa MT5 aktif)
  python scripts/run_backtest.py \\
      --source csv \\
      --m5-file data/historical/XAUUSD_M5_2026-01-01_2026-07-25.csv \\
      --h1-file data/historical/XAUUSD_H1_2026-01-01_2026-07-25.csv
        """,
    )

    # ── Sumber data ───────────────────────────────────────────────────────────
    src_group = parser.add_argument_group("Sumber Data")
    src_group.add_argument(
        "--source",
        choices = ["mt5", "csv"],
        default = "mt5",
        help    = "Sumber data historis. 'mt5' = tarik langsung (butuh MT5 aktif). "
                  "'csv' = pakai file cache yang sudah ada. (default: mt5)",
    )
    src_group.add_argument(
        "--start",
        metavar = "YYYY-MM-DD",
        default = None,
        help    = "Tanggal mulai backtest, inklusif. "
                  "Wajib jika --source mt5. Contoh: 2026-01-01",
    )
    src_group.add_argument(
        "--end",
        metavar = "YYYY-MM-DD",
        default = None,
        help    = "Tanggal akhir backtest, inklusif. "
                  "Wajib jika --source mt5. Contoh: 2026-07-25",
    )
    src_group.add_argument(
        "--m5-file",
        metavar = "PATH",
        default = None,
        help    = "Path ke file CSV M5. Wajib jika --source csv.",
    )
    src_group.add_argument(
        "--h1-file",
        metavar = "PATH",
        default = None,
        help    = "Path ke file CSV H1. Wajib jika --source csv.",
    )
    src_group.add_argument(
        "--symbol",
        default = None,
        help    = "Simbol MT5 (default dari .env: XAUUSD). Hanya relevan jika --source mt5.",
    )
    src_group.add_argument(
        "--no-save-csv",
        action  = "store_true",
        default = False,
        help    = "Jangan simpan cache CSV saat source=mt5. "
                  "(default: simpan otomatis ke data/historical/)",
    )

    # ── Parameter backtest ────────────────────────────────────────────────────
    bt_group = parser.add_argument_group("Parameter Backtest")
    bt_group.add_argument(
        "--max-candles",
        type    = int,
        default = 288,
        metavar = "N",
        help    = "Batas candle forward per trade untuk SL/TP simulation. "
                  "288 M5 = ~24 jam trading time. (default: 288)",
    )
    bt_group.add_argument(
        "--rrr-min",
        type    = float,
        default = 1.3,
        help    = "Minimum Risk-to-Reward Ratio (default: 1.3).",
    )
    bt_group.add_argument(
        "--warm-up",
        type    = int,
        default = 100,
        metavar = "N",
        help    = "Jumlah candle awal yang dilewati (warm-up indikator). (default: 100)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "--output",
        metavar = "DIR",
        default = os.path.join(ROOT_DIR, "data", "backtest_results"),
        help    = "Folder tujuan output CSV hasil trade. "
                  "(default: data/backtest_results/)",
    )

    return parser


# =============================================================================
# LOAD DATA — MT5 atau CSV
# =============================================================================

def _load_from_mt5(args) -> tuple:
    """
    Tarik data M5 dan H1 dari MT5 untuk rentang tanggal yang diminta.

    Setelah berhasil, simpan raw OHLC ke data/historical/ sebagai cache CSV
    untuk reproducibility run berikutnya (kecuali --no-save-csv).

    Return:
        (df_m5, df_h1) — tuple DataFrame mentah siap dipakai run_backtest()
        (None, None) jika gagal
    """
    if args.start is None or args.end is None:
        print("❌ --start dan --end wajib diisi jika --source mt5")
        print("   Contoh: --start 2026-01-01 --end 2026-07-25")
        return None, None

    try:
        date_from = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        date_to   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"❌ Format tanggal tidak valid: {e}")
        print("   Gunakan format YYYY-MM-DD (contoh: 2026-01-01)")
        return None, None

    if date_from >= date_to:
        print("❌ --start harus lebih awal dari --end")
        return None, None

    print(f"→ Menghubungkan ke MT5...")
    if not initialize_mt5():
        return None, None

    try:
        # ── Tarik M5 ──────────────────────────────────────────────────────────
        print()
        df_m5 = get_candles_range(
            date_from     = date_from,
            date_to       = date_to,
            symbol        = args.symbol,
            timeframe_str = "M5",
        )
        if df_m5 is None or df_m5.empty:
            print("❌ Gagal menarik data M5")
            return None, None

        # ── Tarik H1 (dengan warm-up tambahan di awal untuk EMA convergence) ──
        # Kita butuh H1 yang dimulai jauh sebelum date_from agar EMA sudah
        # fully converged saat backtest dimulai.
        # Tambahkan 30 hari H1 sebelum date_from sebagai warm-up buffer.
        from datetime import timedelta
        h1_date_from = date_from - timedelta(days=30)

        print()
        df_h1 = get_candles_range(
            date_from     = h1_date_from,
            date_to       = date_to,
            symbol        = args.symbol,
            timeframe_str = "H1",
        )
        if df_h1 is None or df_h1.empty:
            print("❌ Gagal menarik data H1")
            return None, None

        # ── Simpan ke CSV cache ───────────────────────────────────────────────
        if not args.no_save_csv:
            hist_dir   = os.path.join(ROOT_DIR, "data", "historical")
            date_tag   = f"{args.start}_{args.end}"
            symbol_tag = (args.symbol or "XAUUSD").replace("/", "")

            m5_path = os.path.join(hist_dir, f"{symbol_tag}_M5_{date_tag}.csv")
            h1_path = os.path.join(hist_dir, f"{symbol_tag}_H1_{date_tag}.csv")

            print()
            save_candles_csv(df_m5, m5_path)
            save_candles_csv(df_h1, h1_path)
            print(f"\n💡 Untuk run berikutnya tanpa MT5:")
            print(f"   python scripts/run_backtest.py \\")
            print(f"       --source csv \\")
            print(f"       --m5-file {m5_path} \\")
            print(f"       --h1-file {h1_path}")

        return df_m5, df_h1

    finally:
        shutdown_mt5()


def _load_from_csv(args) -> tuple:
    """
    Load data M5 dan H1 dari file CSV cache lokal.

    Return:
        (df_m5, df_h1) — tuple DataFrame mentah
        (None, None) jika gagal
    """
    if args.m5_file is None or args.h1_file is None:
        print("❌ --m5-file dan --h1-file wajib diisi jika --source csv")
        print("   Contoh: --m5-file data/historical/XAUUSD_M5_2026-01-01_2026-07-25.csv")
        return None, None

    print("→ Loading data M5 dari CSV...")
    df_m5 = load_candles_csv(args.m5_file)
    if df_m5 is None:
        return None, None

    print("→ Loading data H1 dari CSV...")
    df_h1 = load_candles_csv(args.h1_file)
    if df_h1 is None:
        return None, None

    # Filter rentang tanggal jika --start/--end diberikan bersama --source csv
    if args.start is not None or args.end is not None:
        try:
            if args.start:
                start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                df_m5 = df_m5[df_m5.index >= start_dt]
                df_h1 = df_h1[df_h1.index >= start_dt]
            if args.end:
                end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                df_m5 = df_m5[df_m5.index <= end_dt]
                df_h1 = df_h1[df_h1.index <= end_dt]
            print(f"   Filter tanggal diterapkan:")
            print(f"   M5: {len(df_m5):,} candle ({df_m5.index[0]} → {df_m5.index[-1]})")
            print(f"   H1: {len(df_h1):,} candle ({df_h1.index[0]} → {df_h1.index[-1]})")
        except (ValueError, IndexError) as e:
            print(f"⚠️  Filter tanggal gagal: {e} — memakai seluruh data CSV")

    return df_m5, df_h1


# =============================================================================
# PRINT RINGKASAN TERMINAL
# =============================================================================

def _print_summary(summary: dict, trades_df, output_csv_path: str) -> None:
    """
    Cetak ringkasan hasil backtest ke terminal dalam format yang mudah dibaca.

    Semua metrik ditampilkan - termasuk NO_HIT rate yang sering diabaikan.
    """
    print()
    print("=" * 60)
    print("  RINGKASAN HASIL BACKTEST")
    print("=" * 60)

    total  = summary["total_trades"]
    closed = summary["closed_count"]
    tp_n   = summary["tp_count"]
    sl_n   = summary["sl_count"]
    no_n   = summary["no_hit_count"]

    if total == 0:
        print("  WARNING: Tidak ada trade yang ditemukan dalam rentang data.")
        print("     Kemungkinan: warm-up terlalu besar, data terlalu sedikit,")
        print("     atau tidak ada sinyal BUY/SELL dalam periode ini.")
        return

    print(f"\n  STATISTIK TRADE")
    print(f"  {'Total trade':30s}: {total:>6,}")
    print(f"  {'Trade resolved (TP+SL)':30s}: {closed:>6,}")
    print(f"    {'-> TP HIT (profit)':28s}: {tp_n:>6,}")
    print(f"    {'-> SL HIT (loss)':28s}: {sl_n:>6,}")
    print(f"  {'NO HIT (tidak resolve)':30s}: {no_n:>6,}")
    print()

    # ── Win rate dan no-hit rate ──────────────────────────────────────────────
    win_rate    = summary["win_rate"]
    no_hit_rate = summary["no_hit_rate"]

    print(f"  PERFORMA")
    if win_rate is not None:
        wr_pct  = win_rate * 100
        bar_len = int(wr_pct / 5)
        bar     = "=" * bar_len + "-" * (20 - bar_len)
        print(f"  {'Win Rate (TP/closed)':30s}: {wr_pct:>6.1f}%  [{bar}]")
    else:
        print(f"  {'Win Rate (TP/closed)':30s}: N/A (tidak ada trade closed)")

    if no_hit_rate is not None:
        no_pct = no_hit_rate * 100
        flag   = " [WARNING: TINGGI - periksa max_candles]" if no_pct > 15 else ""
        print(f"  {'NO HIT Rate':30s}: {no_pct:>6.1f}%{flag}")

    avg_rrr = summary["avg_rrr_realized"]
    if avg_rrr is not None:
        sign = "+" if avg_rrr >= 0 else ""
        print(f"  {'Avg RRR Realized':30s}: {sign}{avg_rrr:>5.2f}R")

    avg_c = summary["avg_candles_held"]
    if avg_c is not None:
        print(f"  {'Avg Candles Held':30s}: {avg_c:>6.1f} "
              f"(~{avg_c * 5 / 60:.1f} jam trading time)")
    print()

    # ── Risiko ───────────────────────────────────────────────────────────────
    print(f"  RISIKO")
    print(f"  {'Max Consecutive Losses':30s}: {summary['max_consec_loss']:>6}")
    print(f"  {'Max Drawdown':30s}: {summary['max_drawdown_pts']:>+8.2f} pts")
    print(f"  {'Total P&L':30s}: {summary['total_pnl_points']:>+8.2f} pts")
    print()

    # ── Kualitas data ─────────────────────────────────────────────────────────
    ambig_rate = summary.get("ambiguous_rate", 0) or 0
    ambig_n    = summary.get("ambiguous_count", 0)
    print(f"  KUALITAS DATA")
    print(f"  {'BUY trade':30s}: {summary['buy_count']:>6,}")
    print(f"  {'SELL trade':30s}: {summary['sell_count']:>6,}")
    sl_method  = summary.get("sl_method_breakdown", {})
    for method, cnt in sl_method.items():
        print(f"  {'SL Method - ' + method:30s}: {cnt:>6,}")
    flag_ambig = " [WARNING: TINGGI - pertimbangkan data M1]" if ambig_rate > 0.1 else ""
    print(f"  {'Ambiguous Candle':30s}: {ambig_n:>6,} ({ambig_rate*100:.1f}%){flag_ambig}")
    print()

    # ── Output ───────────────────────────────────────────────────────────────
    print(f"  OUTPUT")
    print(f"  CSV hasil trade: {output_csv_path}")
    print(f"  Baris: {len(trades_df):,} trade")
    print()
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # Normalize atribut dengan tanda hubung (argparse konversi ke underscore)
    args.no_save_csv = getattr(args, "no_save_csv", False)
    args.max_candles = getattr(args, "max_candles", 288)
    args.warm_up     = getattr(args, "warm_up", 100)
    args.m5_file     = getattr(args, "m5_file", None)
    args.h1_file     = getattr(args, "h1_file", None)

    print("=" * 60)
    print("  run_backtest.py - XAUUSD M5 Backtest CLI")
    print("=" * 60)
    print(f"  Source      : {args.source}")
    print(f"  Max candles : {args.max_candles} (~{args.max_candles*5/60:.0f} jam)")
    print(f"  Warm-up     : {args.warm_up} candle")
    print(f"  Output dir  : {args.output}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.source == "mt5":
        df_m5, df_h1 = _load_from_mt5(args)
    else:
        df_m5, df_h1 = _load_from_csv(args)

    if df_m5 is None or df_h1 is None:
        sys.exit(1)

    print()

    # ── Jalankan backtest ─────────────────────────────────────────────────────
    try:
        trades_df, summary = run_backtest(
            df_m5       = df_m5,
            df_h1       = df_h1,
            warm_up     = args.warm_up,
            max_candles = args.max_candles,
            verbose     = True,
        )
    except RuntimeError as e:
        # validate_no_lookahead gagal — hentikan dengan pesan jelas
        print(f"\n❌ FATAL: {e}")
        sys.exit(1)

    # ── Simpan CSV hasil trade ─────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_tag    = (args.symbol or "XAUUSD").replace("/", "")
    out_filename  = f"backtest_{symbol_tag}_M5_{timestamp_tag}.csv"
    out_path      = os.path.join(args.output, out_filename)

    if not trades_df.empty:
        trades_df.to_csv(out_path, index=False)

    # ── Print ringkasan ───────────────────────────────────────────────────────
    _print_summary(summary, trades_df, out_path)


if __name__ == "__main__":
    main()
