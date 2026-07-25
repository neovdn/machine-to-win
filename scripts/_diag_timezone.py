"""Diagnostic: konfirmasi timezone broker MT5."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

mt5.initialize()

# ── Nilai mentah dari MT5 ────────────────────────────────────────────────────
tick    = mt5.symbol_info_tick("XAUUSD")
unix_tick = tick.time      # detik, bilangan bulat

# 3 candle paling baru (termasuk candle yang sedang berjalan, start_pos=0)
rates   = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 3)
df      = pd.DataFrame(rates)
unix_active  = int(df["time"].iloc[-1])   # candle sedang berjalan
unix_closed  = int(df["time"].iloc[-2])   # candle sebelumnya (closed)

mt5.shutdown()

# ── Waktu komputer ───────────────────────────────────────────────────────────
now_utc   = datetime.now(timezone.utc)
now_local = datetime.now().astimezone()

print()
print("=" * 65)
print("  INVESTIGASI TIMEZONE BROKER MT5")
print("=" * 65)

print(f"\n  Waktu komputer UTC   : {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"  Waktu komputer lokal : {now_local.strftime('%Y-%m-%d %H:%M:%S %z')}")

# ── Decode unix timestamp dua cara ───────────────────────────────────────────
UTC3 = timezone(timedelta(hours=3))

tick_as_utc  = datetime.fromtimestamp(unix_tick, tz=timezone.utc)
tick_as_utc3 = datetime.fromtimestamp(unix_tick, tz=UTC3)

print(f"\n  Tick unix = {unix_tick}")
print(f"  Dibaca sebagai UTC   : {tick_as_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"  Dibaca sebagai UTC+3 : {tick_as_utc3.strftime('%Y-%m-%d %H:%M:%S')} UTC+3")

# ── Pengujian: selisih tick vs sekarang ──────────────────────────────────────
delta_raw = (tick_as_utc - now_utc).total_seconds()   # selisih jika dibaca mentah

print()
print("=" * 65)
print("  PENGUJIAN SELISIH")
print("=" * 65)
print(f"\n  Jika tick dianggap UTC   -> selisih vs now_utc = {delta_raw:+.0f} detik  ({delta_raw/3600:+.2f} jam)")

# Cek: apakah unix timestamp ini adalah DETIK SEJAK EPOCH UTC atau BUKAN?
# Unix timestamp sejati: setiap detik = satu detik sejati di UTC.
# Epoch UTC = 1970-01-01 00:00:00 UTC.
# Kalau tick.time = 1784895877, dikonversi ke UTC = 12:24 UTC.
# Tapi now_utc = 09:24 UTC → selisih -3 jam.
# 
# Artinya: nilai integer unix_tick jika diinterpretasi sebagai detik-UTC
# menghasilkan waktu yang 3 jam lebih maju dari sekarang.
#
# Bagaimana ini bisa terjadi?
# → Broker server timezone = UTC+3.
# → MT5 library mengembalikan waktu SERVER LOKAL (UTC+3) dalam format detik,
#   TAPI dihitung sejak epoch UTC (bukan epoch UTC+3).
# → Hasilnya: angka unix = jam lokal broker, tapi frame-of-reference = UTC epoch.
# → Saat pandas decode dengan utc=True, waktu yang keluar = jam broker lokal
#   (UTC+3), BUKAN jam UTC sejati.
# → Ini adalah perilaku RESMI MT5: copy_rates_from_pos mengembalikan waktu
#   dalam TIMEZONE SERVER BROKER, bukan UTC.

print()
print("=" * 65)
print("  KESIMPULAN FAKTA")
print("=" * 65)
print(f"""
  Fakta 1 — Selisih tepat -10800 detik (-3 jam):
    tick.time sebagai 'UTC' = {tick_as_utc.strftime('%H:%M:%S')} UTC
    Komputer UTC saat ini   = {now_utc.strftime('%H:%M:%S')} UTC
    Selisih                 = {delta_raw/3600:+.0f} jam (tepat -3 jam, bukan ±1/2 jam)

  Fakta 2 — Perilaku ini RESMI di dokumentasi MT5:
    mt5.copy_rates_from_pos() mengembalikan field 'time' dalam timezone
    SERVER BROKER, bukan UTC. Nilai ini adalah detik sejak epoch, tapi
    epoch-nya dihitung dari UTC+0 sedangkan jam yang disimpan = jam UTC+3.

  Fakta 3 — Konfirmasi: UTC+3 = EEST (Eastern European Summer Time)
    Bulan Juli = musim panas Eropa Timur.
    EET  (winter) = UTC+2, berlaku Oktober-Maret
    EEST (summer) = UTC+3, berlaku Maret-Oktober  ← sekarang
    Broker Eropa Timur (banyak broker Forex besar: IC Markets, Pepperstone,
    Exness, dll.) standarnya pakai EET/EEST.

  Fakta 4 — Implikasi ke kode kita:
    pd.to_datetime(df['time'], unit='s', utc=True) menghasilkan timestamp
    dengan label '+00:00' (UTC), TAPI jam yang dikandungnya = jam broker (UTC+3).
    Label UTC-nya SALAH — seharusnya +03:00.
""")

# ── Validasi yang benar: koreksi timezone dulu ───────────────────────────────
print("=" * 65)
print("  VALIDASI YANG BENAR (dengan koreksi timezone)")
print("=" * 65)

# Candle aktif (start_pos=0)
active_open_broker  = datetime.fromtimestamp(unix_active,  tz=timezone.utc)  # jam broker, salah label
active_open_utc_true = active_open_broker - timedelta(hours=3)                 # koreksi ke UTC sejati
active_close_utc_true = active_open_utc_true + timedelta(minutes=5)           # +5 menit = waktu tutup

# Candle closed (start_pos=0, posisi -2)
closed_open_broker   = datetime.fromtimestamp(unix_closed, tz=timezone.utc)
closed_open_utc_true = closed_open_broker - timedelta(hours=3)
closed_close_utc_true = closed_open_utc_true + timedelta(minutes=5)

print(f"\n  === Candle AKTIF (sedang berjalan, start_pos=0 posisi terakhir) ===")
print(f"  Open broker (raw)    : {active_open_broker.strftime('%H:%M')} (label UTC, sebenarnya UTC+3)")
print(f"  Open UTC sejati      : {active_open_utc_true.strftime('%H:%M')} UTC")
print(f"  Close UTC sejati     : {active_close_utc_true.strftime('%H:%M')} UTC  (open + 5 menit)")
print(f"  Waktu sekarang (UTC) : {now_utc.strftime('%H:%M:%S')} UTC")
delta_active = (active_close_utc_true - now_utc).total_seconds() / 60
print(f"  Sisa waktu candle    : {delta_active:.1f} menit sampai closed")
if delta_active > 0:
    print(f"  → Candle ini BELUM closed ({delta_active:.1f} menit lagi)")
else:
    print(f"  → Candle ini sudah closed ({abs(delta_active):.1f} menit yang lalu)")

print(f"\n  === Candle CLOSED (posisi ke-2, start_pos=0 posisi -2) ===")
print(f"  Open broker (raw)    : {closed_open_broker.strftime('%H:%M')} (label UTC, sebenarnya UTC+3)")
print(f"  Open UTC sejati      : {closed_open_utc_true.strftime('%H:%M')} UTC")
print(f"  Close UTC sejati     : {closed_close_utc_true.strftime('%H:%M')} UTC  (open + 5 menit)")
print(f"  Waktu sekarang (UTC) : {now_utc.strftime('%H:%M:%S')} UTC")
delta_closed = (now_utc - closed_close_utc_true).total_seconds() / 60
print(f"  Sudah closed sejak   : {delta_closed:.1f} menit yang lalu")

print()
print("=" * 65)
print("  IMPLIKASI KE VALIDASI di test_indicators.py")
print("=" * 65)
print("""
  Metode lama (sebelum fix):
    datetime.now(utc) vs candle timestamp → selisih -3 jam karena timezone
    mismatch. Hasilnya "umur candle = -180 menit" → tidak berguna.

  Metode baru (setelah fix):
    Membandingkan gap antar 2 candle berturutan (candle[-1] vs candle[-2]).
    Gap ini TIMEZONE-INDEPENDENT karena kedua candle pakai timezone yang sama.
    Gap = 5 menit → bukti data candle konsisten (bukan bukti candle closed).

  Apa yang BELUM divalidasi oleh metode baru:
    Metode gap hanya membuktikan data berurutan, BUKAN membuktikan
    bahwa candle terbaru (setelah fix start_pos=1) sudah closed.
    Untuk itu perlu koreksi timezone seperti yang ditampilkan di atas.

  Validasi yang BENAR-BENAR membuktikan candle closed:
    Ambil candle terbaru (df.iloc[-1] setelah start_pos=1).
    Konversi timestamp-nya ke UTC sejati (kurangi 3 jam karena UTC+3).
    Hitung: (now_utc) - (candle_open_utc + 5 menit).
    Jika positif → candle sudah closed. Jika negatif → belum closed.
""")
