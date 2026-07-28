"""
scripts/test_indicators.py
==========================
Script untuk memvalidasi hasil kalkulasi indikator di engine/indicators.py.

CARA PAKAI:
    1. Pastikan MT5 sudah terbuka dan login
    2. Jalankan dari root project:
       python scripts/test_indicators.py

VALIDASI CANDLE CLOSED:
    Script ini juga memverifikasi bahwa sistem menggunakan candle yang sudah
    CLOSED, bukan candle yang sedang berjalan. Caranya:
    - Tampilkan timestamp candle terbaru
    - Hitung selisih vs waktu sekarang
    - Jika selisih >= 5 menit (1 timeframe M5), berarti BENAR — sistem skip
      candle aktif dan menggunakan candle closed terakhir.

CARA MEMBANDINGKAN DENGAN MT5:
    1. Di MT5, buka chart XAUUSD M5
    2. Tambahkan EMA 9 dan EMA 21 (Insert → Indicators → Trend → Moving Average)
       - Period: 9, Method: Exponential, Apply to: Close
       - Period: 21, Method: Exponential, Apply to: Close
    3. Tambahkan RSI 14 (Insert → Indicators → Oscillators → RSI)
       - Period: 14
    4. Arahkan ke candle KEDUA dari kanan (bukan yang paling kanan!) —
       karena yang paling kanan adalah candle aktif yang kita skip.
    5. Bandingkan nilai yang muncul di sini dengan nilai di MT5 pada candle tersebut.

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
import MetaTrader5 as mt5  # untuk ambil waktu broker sebagai referensi validasi
from datetime import datetime, timezone  # untuk validasi waktu UTC sejati


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

# Ambil waktu sekarang dalam UTC sejati SEBELUM shutdown.
# Setelah fix timezone di data_fetcher, df.index sudah UTC sejati.
# Maka perbandingan "sekarang vs candle terbaru" harus pakai UTC sejati juga.
# Kita tidak lagi membutuhkan broker_now_unix — waktu UTC komputer sudah cukup.
now_utc = datetime.now(timezone.utc)


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
print("  📃  10 CANDLE TERAKHIR + INDIKATOR (semua sudah closed)")
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
# STEP 4: Tampilkan ringkasan nilai terbaru (candle closed terakhir)
# ─────────────────────────────────────────────────────────────────────────────

signals = get_latest_signals(df)

print()
print("=" * 65)
print("  📊  KONDISI MARKET — CANDLE CLOSED TERAKHIR")
print("=" * 65)
print(f"  Waktu candle : {signals['time']}")
print(f"  Close        : {signals['close']:.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# VALIDASI CANDLE CLOSED
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  ✅  VALIDASI: APAKAH SISTEM SUDAH SKIP CANDLE AKTIF?")
print("=" * 65)

# ── Cara kerja validasi ini (setelah fix timezone) ───────────────────────────
# df.index[-1] sekarang berisi waktu UTC sejati (sudah dikoreksi offset broker
# di data_fetcher.py). Kita bandingkan langsung dengan now_utc yang juga UTC.
#
# now_utc        : waktu sekarang (UTC sejati, diambil sebelum shutdown)
# candle_open    : waktu pembukaan candle (UTC sejati, dari df.index[-1])
# candle_close   : candle_open + 300 detik (5 menit × 60)
# df.index[-1] sudah UTC sejati (setelah fix timezone di data_fetcher.py)
# Kita pakai .timestamp() untuk konversi ke detik integer — ini benar karena
# pandas Timestamp yang tz-aware mengembalikan epoch UTC saat dipanggil .timestamp()
candle_open_ts    = df.index[-1]
candle_open_unix  = int(candle_open_ts.timestamp())
TF_SECONDS = 5 * 60  # timeframe M5 = 300 detik
candle_close_unix = candle_open_unix + TF_SECONDS
now_unix          = int(now_utc.timestamp())

# Selisih dalam detik (positif = sudah lewat, negatif = belum terjadi)
sudah_closed_detik = now_unix - candle_close_unix
sudah_closed_menit = sudah_closed_detik / 60

print(f"  Referensi waktu sekarang (UTC)  : {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"  Candle terbaru open (UTC sejati) : {candle_open_ts}")
print(f"  Perkiraan candle close           : unix = {candle_close_unix}  (open + {TF_SECONDS}s)")
print(f"  Selisih (now - close)            : {sudah_closed_detik} detik  = {sudah_closed_menit:.1f} menit")
print()

if sudah_closed_detik > 0:
    print(f"  ✅ TERBUKTI CLOSED! Candle terbaru sudah tutup {sudah_closed_menit:.1f} menit yang lalu.")
    print(f"     Waktu UTC sekarang ({now_unix}) > waktu tutup candle ({candle_close_unix}).")
    print(f"     Sistem TIDAK menganalisis candle yang sedang berjalan.")
    print(f"     Keputusan trading tidak akan repainting.")
elif sudah_closed_detik == 0:
    print(f"  ✅ Candle baru saja closed (tepat di waktu tutupnya).")
else:
    print(f"  ⚠️  PERINGATAN: Candle terbaru belum closed ({abs(sudah_closed_menit):.1f} menit lagi).")
    print(f"     Ini tidak normal jika start_pos=1 sudah diterapkan di get_candles().")
    print(f"     Periksa data_fetcher.py.")

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
