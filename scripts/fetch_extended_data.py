"""
scripts/fetch_extended_data.py
================================
Script untuk menarik data XAUUSD M5 dan H1 dari 2025-06-01 s/d 2026-07-25
dan menyimpannya ke file BARU — tidak pernah menyentuh file cache lama.
Mendukung chunked fetching karena limit max bars MT5 per request.

File yang dibuat:
  data/historical/XAUUSD_M5_2025-06-01_2026-07-25.csv
  data/historical/XAUUSD_H1_2025-06-01_2026-07-25.csv

File yang TIDAK BOLEH disentuh (dan tidak akan disentuh oleh script ini):
  data/historical/XAUUSD_M5_2026-01-01_2026-07-25.csv
  data/historical/XAUUSD_H1_2026-01-01_2026-07-25.csv
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import (
    initialize_mt5,
    get_candles_range,
    save_candles_csv,
    shutdown_mt5,
)

# Target file paths — NAMA BARU, bukan overwrite file lama
M5_OUT = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
H1_OUT = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

# File lama yang TIDAK BOLEH disentuh
M5_OLD = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
H1_OLD = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

def chunked_fetch(date_from, date_to, timeframe_str):
    """Fetch in chunks to avoid MT5 limit"""
    chunks = [
        (datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc)),
        (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 25, tzinfo=timezone.utc))
    ]
    dfs = []
    for d_from, d_to in chunks:
        df = get_candles_range(date_from=d_from, date_to=d_to, timeframe_str=timeframe_str)
        if df is not None and not df.empty:
            dfs.append(df)
        else:
            print(f"ERROR: Chunk {d_from} -> {d_to} gagal")
            return None
    return pd.concat(dfs).drop_duplicates()

def main():
    print("=" * 65)
    print("  FETCH EXTENDED DATA — Jun 2025 s/d Jul 2026 (CHUNKED)")
    print("=" * 65)

    # Verifikasi file lama masih ada dan tidak akan disentuh
    print("\n[CHECK] Verifikasi file lama tidak akan disentuh:")
    for f in [M5_OLD, H1_OLD]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            mtime = os.path.getmtime(f)
            import time
            print(f"  OK — {os.path.basename(f)} ({size:,} bytes, last-modified: "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})")
        else:
            print(f"  WARNING: {os.path.basename(f)} tidak ditemukan!")

    # Cek apakah output sudah ada
    for f, label in [(M5_OUT, "M5"), (H1_OUT, "H1")]:
        if os.path.exists(f):
            print(f"\n  File {label} baru sudah ada: {os.path.basename(f)}")
            if f == M5_OLD or f == H1_OLD:
                print("  ERROR: Path sama dengan file lama! Script dibatalkan.")
                sys.exit(1)

    # Koneksi MT5
    print("\n[1/4] Menghubungkan ke MetaTrader 5...")
    if not initialize_mt5():
        print("ERROR: MT5 tidak tersedia.")
        sys.exit(1)

    try:
        # Tarik M5
        if not os.path.exists(M5_OUT):
            print("\n[2/4] Menarik data M5 (Jun 2025 - Jul 2026) in chunks...")
            df_m5 = chunked_fetch(None, None, "M5")
            if df_m5 is None or df_m5.empty:
                print("ERROR: Data M5 tidak tersedia.")
                sys.exit(1)
            print(f"  Total M5 ditarik: {len(df_m5):,} candle "
                  f"({df_m5.index[0]} -> {df_m5.index[-1]})")
            save_candles_csv(df_m5, M5_OUT)
        else:
            print(f"\n[2/4] M5 sudah ada, skip.")

        # Tarik H1
        if not os.path.exists(H1_OUT):
            print("\n[3/4] Menarik data H1 (Jun 2025 - Jul 2026) in chunks...")
            df_h1 = chunked_fetch(None, None, "H1")
            if df_h1 is None or df_h1.empty:
                print("ERROR: Data H1 tidak tersedia.")
                sys.exit(1)
            print(f"  Total H1 ditarik: {len(df_h1):,} candle "
                  f"({df_h1.index[0]} -> {df_h1.index[-1]})")
            save_candles_csv(df_h1, H1_OUT)
        else:
            print(f"\n[3/4] H1 sudah ada, skip.")

    finally:
        print("\n[4/4] Menutup koneksi MT5...")
        shutdown_mt5()

if __name__ == "__main__":
    main()
