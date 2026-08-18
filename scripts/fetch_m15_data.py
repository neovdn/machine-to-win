"""
scripts/fetch_m15_data.py
==========================
Script untuk menarik data XAUUSD M15 dari 2025-06-01 s/d 2026-07-25
dan menyimpannya ke file baru — tidak pernah menyentuh file cache lama.
Mendukung chunked fetching karena limit max bars MT5 per request.

TUJUAN:
    Menyediakan data historis M15 untuk validasi empiris Fase 13
    (scripts/_diag_regime_detector.py). Tanpa file ini, diagnostik
    tidak bisa dijalankan.

FILE YANG DIBUAT:
    data/historical/XAUUSD_M15_2025-06-01_2026-07-25.csv

FILE YANG TIDAK BOLEH DISENTUH:
    data/historical/XAUUSD_M5_2025-06-01_2026-07-25.csv
    data/historical/XAUUSD_H1_2025-06-01_2026-07-25.csv
    data/historical/XAUUSD_M5_2026-01-01_2026-07-25.csv
    data/historical/XAUUSD_H1_2026-01-01_2026-07-25.csv

CATATAN:
    Script ini BUTUH MetaTrader 5 aktif dan terhubung untuk dijalankan.
    Jangan jalankan script ini tanpa MT5 aktif di komputer.

    Chunked fetch dibagi dua periode untuk menghindari limit max bars MT5:
        Chunk 1: 2025-06-01 s/d 2025-12-31
        Chunk 2: 2026-01-01 s/d 2026-07-25

USAGE:
    python scripts/fetch_m15_data.py

    (harus dijalankan dengan MT5 aktif dan terhubung)
"""

import os
import sys
import time
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

# =============================================================================
# PATH KONFIGURASI
# =============================================================================

# Target — file M15 baru yang akan dibuat
M15_OUT = os.path.join(
    ROOT_DIR, "data", "historical", "XAUUSD_M15_2025-06-01_2026-07-25.csv"
)

# File-file lama yang TIDAK BOLEH disentuh
FILE_TIDAK_BOLEH_DISENTUH = [
    os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv"),
    os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv"),
    os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv"),
    os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv"),
]

# =============================================================================
# FUNGSI CHUNKED FETCH
# =============================================================================

def chunked_fetch_m15():
    """
    Fetch data M15 dalam dua chunk untuk menghindari limit bar MT5.

    Chunk 1: 2025-06-01 → 2025-12-31
    Chunk 2: 2026-01-01 → 2026-07-25

    Return: DataFrame gabungan, atau None jika salah satu chunk gagal.
    """
    chunks = [
        (
            datetime(2025, 6,  1,  0, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 12, 31, 23, 59, 0, tzinfo=timezone.utc),
            "2025-06-01 → 2025-12-31",
        ),
        (
            datetime(2026, 1,  1,  0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 7,  25, 23, 59, 0, tzinfo=timezone.utc),
            "2026-01-01 → 2026-07-25",
        ),
    ]

    dfs = []
    for d_from, d_to, label in chunks:
        print(f"   Chunk [{label}] ...")
        df_chunk = get_candles_range(
            date_from      = d_from,
            date_to        = d_to,
            timeframe_str  = "M15",
        )
        if df_chunk is None or df_chunk.empty:
            print(f"   ERROR: Chunk [{label}] gagal — tidak ada data dikembalikan.")
            return None

        print(f"   → {len(df_chunk):,} candle M15 "
              f"({df_chunk.index[0]} → {df_chunk.index[-1]})")
        dfs.append(df_chunk)

    # Gabungkan dan hapus duplikat (bisa terjadi di batas chunk)
    df_gabung = pd.concat(dfs).drop_duplicates()
    df_gabung = df_gabung.sort_index()

    return df_gabung


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  FETCH M15 DATA — Fase 13 Validasi Empiris")
    print("  Rentang: Jun 2025 s/d Jul 2026 (CHUNKED)")
    print("=" * 65)

    # ── Verifikasi output path BUKAN salah satu file lama ────────────────────
    print("\n[CHECK] Verifikasi tidak akan menimpa file lama:")
    for f in FILE_TIDAK_BOLEH_DISENTUH:
        basename = os.path.basename(f)
        if os.path.exists(f):
            size  = os.path.getsize(f)
            mtime = os.path.getmtime(f)
            print(f"  OK — {basename} ada ({size:,} bytes, "
                  f"modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})")
        else:
            print(f"  INFO: {basename} tidak ditemukan (tidak masalah)")

    # Pastikan M15_OUT bukan salah satu file lama
    if M15_OUT in FILE_TIDAK_BOLEH_DISENTUH:
        print("\n  ERROR: Path output sama dengan file lama! Script dibatalkan.")
        sys.exit(1)

    # ── Cek apakah output sudah ada ──────────────────────────────────────────
    if os.path.exists(M15_OUT):
        size  = os.path.getsize(M15_OUT)
        mtime = os.path.getmtime(M15_OUT)
        print(f"\n  File sudah ada: {os.path.basename(M15_OUT)}")
        print(f"  ({size:,} bytes, modified: "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})")
        print("  Lewati fetch — gunakan --force untuk overwrite (belum diimplementasi).")
        print("\n  → Selesai (tidak ada yang dilakukan).")
        return

    # ── Pastikan direktori output ada ────────────────────────────────────────
    os.makedirs(os.path.dirname(M15_OUT), exist_ok=True)

    # ── Koneksi MT5 ──────────────────────────────────────────────────────────
    print("\n[1/3] Menghubungkan ke MetaTrader 5 ...")
    if not initialize_mt5():
        print("  ERROR: MT5 tidak tersedia atau tidak bisa diinisialisasi.")
        print("  Pastikan MT5 sudah berjalan dan terhubung ke broker sebelum")
        print("  menjalankan script ini.")
        sys.exit(1)
    print("  Koneksi MT5 berhasil.")

    try:
        # ── Fetch M15 dalam chunks ────────────────────────────────────────────
        print("\n[2/3] Menarik data M15 (Jun 2025 → Jul 2026) dalam 2 chunk ...")
        df_m15 = chunked_fetch_m15()

        if df_m15 is None or df_m15.empty:
            print("  ERROR: Data M15 tidak berhasil ditarik.")
            sys.exit(1)

        print(f"\n  Total M15 ditarik: {len(df_m15):,} candle")
        print(f"  Rentang: {df_m15.index[0]} → {df_m15.index[-1]}")

        # ── Simpan ke CSV ─────────────────────────────────────────────────────
        print(f"\n[3/3] Menyimpan ke: {os.path.basename(M15_OUT)} ...")
        save_candles_csv(df_m15, M15_OUT)
        size = os.path.getsize(M15_OUT)
        print(f"  → Tersimpan: {size:,} bytes")

    finally:
        # ── Tutup koneksi MT5 ─────────────────────────────────────────────────
        print("\nMenutup koneksi MT5 ...")
        shutdown_mt5()

    print("\n" + "=" * 65)
    print("  SELESAI — Data M15 siap untuk validasi empiris Fase 13.")
    print(f"  Jalankan: python scripts/_diag_regime_detector.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
