"""
scripts/fetch_candles.py
========================
Script untuk menguji koneksi ke MT5 dan melihat data candle OHLC.

CARA JALANKAN:
    python scripts/fetch_candles.py

PRASYARAT:
    - MetaTrader 5 desktop sudah TERBUKA dan SUDAH LOGIN ke akun broker
    - Library sudah terinstall: pip install -r requirements.txt

YANG DILAKUKAN SCRIPT INI:
    1. Hubungkan Python ke MT5 desktop
    2. Ambil 500 candle XAUUSD M5 terakhir
    3. Validasi kualitas data
    4. Tampilkan ringkasan dan preview data
    5. Tutup koneksi MT5
"""

import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# Tambahkan folder root project ke Python path
# ─────────────────────────────────────────────────────────────────────────────
# Kenapa perlu ini?
# Script ini ada di folder /scripts/, tapi ingin import dari folder /engine/
# Python perlu tahu di mana mencari modul 'engine'
# Kita tambahkan folder parent (/machine-to-win/) ke sys.path secara manual
# ─────────────────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from engine.data_fetcher import initialize_mt5, shutdown_mt5, get_candles, validate_data


def display_summary(df):
    """
    Menampilkan ringkasan statistik dari DataFrame candle.

    Fungsi ini hanya untuk keperluan tampilan di terminal (bukan bagian dari engine).
    """
    from tabulate import tabulate

    # ──────────────────────────────────────────────────
    # Tampilkan 10 candle TERBARU (paling bawah = terbaru)
    # ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  📋  10 CANDLE TERBARU (XAUUSD M5)")
    print("="*65)

    # Format DataFrame untuk tampilan: reset index agar 'time' muncul sebagai kolom
    df_display = df.tail(10).reset_index()

    # Format kolom waktu agar lebih ringkas (hapus timezone info)
    df_display["time"] = df_display["time"].dt.strftime("%Y-%m-%d %H:%M")

    # Format harga dengan 2 desimal (standar Gold/XAUUSD)
    for col in ["open", "high", "low", "close"]:
        df_display[col] = df_display[col].map("{:.2f}".format)

    print(tabulate(
        df_display,
        headers=["Waktu (UTC)", "Open", "High", "Low", "Close", "Vol"],
        tablefmt="rounded_outline",  # Tabel dengan border yang rapi
        showindex=False              # Jangan tampilkan index nomor baris
    ))

    # ──────────────────────────────────────────────────
    # Tampilkan statistik keseluruhan data
    # ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  📊  STATISTIK DATA")
    print("="*65)

    # Waktu candle pertama dan terakhir dalam data
    first_time = df.index[0].strftime("%Y-%m-%d %H:%M UTC")
    last_time  = df.index[-1].strftime("%Y-%m-%d %H:%M UTC")

    stats = [
        ["Jumlah candle",    len(df)],
        ["Rentang waktu",    f"{first_time}  →  {last_time}"],
        ["Harga tertinggi",  f"{df['high'].max():.2f}"],
        ["Harga terendah",   f"{df['low'].min():.2f}"],
        ["Close terakhir",   f"{df['close'].iloc[-1]:.2f}"],
        ["Kolom tersedia",   ", ".join(df.columns.tolist())],
    ]

    print(tabulate(stats, tablefmt="simple", colalign=("right", "left")))
    print()


def main():
    """
    Fungsi utama — alur eksekusi script dari awal sampai akhir.
    """
    print("\n" + "="*65)
    print("  🚀  MACHINE-TO-WIN — Test Koneksi MT5 & Data XAUUSD M5")
    print("="*65 + "\n")

    # ─────────────────────────────────────────
    # STEP 1: Inisialisasi koneksi ke MT5
    # ─────────────────────────────────────────
    print("[ STEP 1 ] Menghubungkan ke MetaTrader 5...")
    if not initialize_mt5():
        # Jika koneksi gagal, hentikan script (tidak ada yang bisa dilakukan)
        print("\n❌ Script dihentikan karena koneksi MT5 gagal.")
        sys.exit(1)  # Exit code 1 = ada error

    print()

    try:
        # ─────────────────────────────────────────
        # STEP 2: Ambil data candle
        # ─────────────────────────────────────────
        print("[ STEP 2 ] Mengambil data candle...")
        df = get_candles()  # Pakai nilai default dari .env

        if df is None:
            print("\n❌ Script dihentikan karena gagal mengambil data candle.")
            sys.exit(1)

        print()

        # ─────────────────────────────────────────
        # STEP 3: Validasi kualitas data
        # ─────────────────────────────────────────
        print("[ STEP 3 ] Validasi data...")
        validate_data(df)

        # ─────────────────────────────────────────
        # STEP 4: Tampilkan hasil
        # ─────────────────────────────────────────
        print("\n[ STEP 4 ] Menampilkan data...")
        display_summary(df)

        print("✅ SUKSES! Data candle XAUUSD M5 berhasil ditarik dan ditampilkan.")
        print("   Siap untuk Step 2: Kalkulasi Indikator (EMA, RSI, S/R)\n")

    finally:
        # ─────────────────────────────────────────
        # STEP 5: Selalu tutup koneksi MT5
        # ─────────────────────────────────────────
        # 'finally' memastikan koneksi SELALU ditutup, bahkan jika ada error di atas
        print("[ STEP 5 ] Menutup koneksi MT5...")
        shutdown_mt5()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: hanya jalankan main() jika script dijalankan langsung
# (bukan di-import oleh modul lain)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
