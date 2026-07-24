"""
scripts/test_indicators.py
==========================
Script untuk memvalidasi hasil kalkulasi indikator di engine/indicators.py.

CARA PAKAI:
    1. Pastikan MT5 sudah terbuka dan login
    2. Jalankan dari root project:
       python scripts/test_indicators.py

CARA MEMBANDINGKAN DENGAN MT5:
    1. Di MT5, buka chart XAUUSD M5
    2. Tambahkan EMA 9 dan EMA 21 (Insert → Indicators → Trend → Moving Average)
       - Period: 9, Method: Exponential, Apply to: Close
       - Period: 21, Method: Exponential, Apply to: Close
    3. Tambahkan RSI 14 (Insert → Indicators → Oscillators → RSI)
       - Period: 14
    4. Bandingkan nilai yang muncul di sini dengan nilai di MT5 pada candle terakhir

KENAPA BISA SEDIKIT BERBEDA:
    Perbedaan kecil (< 0.5) di candle terakhir adalah NORMAL karena:
    - MT5 memakai seluruh history akun sebagai "seed" nilai awal EMA
    - Script ini hanya pakai 500 candle
    - Makin banyak candle yang diambil, makin kecil perbedaannya
    - Solusi: naikkan MT5_CANDLE_COUNT di .env ke 1000 atau 2000
"""

import sys
import os

# Paksa terminal Windows untuk pakai encoding UTF-8 agar emoji tampil benar
# Ini tidak wajib di Linux/Mac tapi diperlukan di Windows terminal tertentu
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Tambahkan root project ke Python path agar import engine.* bisa jalan
# Ini diperlukan karena script dijalankan dari folder scripts/,
# tapi modul ada di engine/ yang sejajar dengan scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.data_fetcher import initialize_mt5, get_candles, shutdown_mt5
from engine.indicators import run_all_indicators, get_latest_signals


# ─────────────────────────────────────────────────────────────────────────────
# TAMPILAN HEADER
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  \U0001f9ea  TEST INDIKATOR \u2014 EMA 9/21, RSI 14, Trend Detection")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Hubungkan ke MT5 dan ambil data
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ STEP 1 ] Menghubungkan ke MT5...")
if not initialize_mt5():
    print("❌ Tidak bisa lanjut — MT5 tidak terhubung.")
    sys.exit(1)

print("\n[ STEP 2 ] Mengambil data candle XAUUSD M5...")
df = get_candles()

if df is None:
    print("❌ Gagal mengambil data. Tutup koneksi MT5.")
    shutdown_mt5()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Hitung semua indikator
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ STEP 3 ] Menghitung indikator (EMA, RSI, Trend)...")
df = run_all_indicators(df)
print("✅ Kalkulasi selesai!")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Tampilkan 10 candle terakhir dengan semua indikator
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  📋  10 CANDLE TERAKHIR + INDIKATOR")
print("=" * 65)

# Pilih kolom yang mau ditampilkan
display_cols = ["close", "ema_9", "ema_21", "rsi_14", "trend"]
df_display = df[display_cols].tail(10).copy()

# Format angka agar rapi
df_display["close"]  = df_display["close"].round(2)
df_display["ema_9"]  = df_display["ema_9"].round(2)
df_display["ema_21"] = df_display["ema_21"].round(2)
df_display["rsi_14"] = df_display["rsi_14"].round(1)

try:
    from tabulate import tabulate
    # Format waktu jadi lebih pendek untuk tampilan
    df_display.index = df_display.index.strftime("%m-%d %H:%M")
    print(tabulate(df_display, headers="keys", tablefmt="rounded_outline",
                   floatfmt=".2f"))
except ImportError:
    print(df_display.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Tampilkan ringkasan nilai terbaru (candle paling kanan di chart)
# ─────────────────────────────────────────────────────────────────────────────

signals = get_latest_signals(df)

print()
print("=" * 65)
print("  📊  KONDISI MARKET SAAT INI (CANDLE TERBARU)")
print("=" * 65)
print(f"  Waktu candle : {signals['time']}")
print(f"  Close        : {signals['close']:.2f}")
print()
print(f"  EMA 9        : {signals['ema_9']:.2f}")
print(f"  EMA 21       : {signals['ema_21']:.2f}")
print(f"  EMA Gap %    : {signals['ema_gap_pct']:+.4f}%")
print()
print(f"  RSI 14       : {signals['rsi_14']:.1f}")

# Interpretasi RSI untuk kemudahan membaca
rsi_val = signals["rsi_14"]
if rsi_val > 70:
    rsi_label = "⚠️  OVERBOUGHT — momentum bullish kuat, waspadai reversal"
elif rsi_val < 30:
    rsi_label = "⚠️  OVERSOLD   — momentum bearish kuat, waspadai reversal"
elif rsi_val >= 50:
    rsi_label = "✅ Zona bullish netral"
else:
    rsi_label = "🔽 Zona bearish netral"
print(f"               {rsi_label}")

print()
# Tampilkan trend dengan emoji yang intuitif
trend_icons = {
    "UPTREND"   : "📈 UPTREND   — EMA9 > EMA21, close di atas EMA21",
    "DOWNTREND" : "📉 DOWNTREND — EMA9 < EMA21, close di bawah EMA21",
    "SIDEWAYS"  : "➡️  SIDEWAYS  — EMA belum konfirmasi arah jelas",
}
print(f"  Trend        : {trend_icons.get(signals['trend'], signals['trend'])}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Panduan verifikasi manual
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  ✅  CARA VERIFIKASI MANUAL DI MT5")
print("=" * 65)
print("  1. Buka chart XAUUSD M5 di MT5")
print("  2. Tambah EMA 9 (Exponential, Apply to Close)")
print("  3. Tambah EMA 21 (Exponential, Apply to Close)")
print("  4. Tambah RSI 14")
print("  5. Hover ke candle TERAKHIR, bandingkan nilai dengan output di atas")
print()
print("  💡 Perbedaan kecil (± 1–2 poin) di EMA/RSI adalah NORMAL karena")
print("     script ini hanya pakai 500 candle sebagai history.")
print("     Untuk hasil lebih akurat, naikkan MT5_CANDLE_COUNT di .env ke 2000.")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Tutup koneksi MT5 dengan bersih
# ─────────────────────────────────────────────────────────────────────────────

print("[ STEP 4 ] Menutup koneksi MT5...")
shutdown_mt5()
print("\n✅ Test selesai!")
