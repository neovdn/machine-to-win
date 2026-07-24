"""
scripts/test_rule_engine.py
===========================
Script validasi end-to-end: MT5 → Indikator → Rule Engine → Keputusan.

CARA PAKAI:
    1. Pastikan MT5 sudah terbuka dan login
    2. Jalankan dari root project:
       python scripts/test_rule_engine.py

OUTPUT YANG DITAMPILKAN:
    1. Nilai indikator terbaru (close, EMA, RSI, trend)
    2. Keputusan akhir (BUY / SELL / WAIT) beserta alasannya
    3. Breakdown tiap kondisi untuk audit
    4. Tabel simulasi skenario berbeda (untuk memastikan logika benar)

CARA MEMBANDINGKAN DENGAN ANALISIS MANUAL:
    - Buka chart XAUUSD M5 di MT5
    - Perhatikan posisi EMA 9 vs EMA 21 dan nilai RSI
    - Bandingkan dengan keputusan yang muncul di sini
    - Jika keputusan berbeda dari analisis manual kamu, lihat bagian
      "breakdown kondisi" untuk tahu kondisi mana yang tidak terpenuhi
"""

import sys
import os

# Setup encoding UTF-8 agar emoji tampil di terminal Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Tambahkan root project ke Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.data_fetcher  import initialize_mt5, get_candles, shutdown_mt5
from engine.indicators    import run_all_indicators, get_latest_signals
from engine.rule_engine   import evaluate_entry


# =============================================================================
# HELPERS TAMPILAN
# =============================================================================

def _garis(char="=", lebar=65):
    print(char * lebar)

def _cetak_keputusan(hasil: dict):
    """Tampilkan keputusan akhir dengan format yang jelas dan mudah dibaca."""

    keputusan = hasil["keputusan"]

    # Pilih warna/simbol berdasarkan keputusan
    if keputusan == "BUY":
        ikon = "📈 BUY  (LONG)"
        warna_garis = "─"
    elif keputusan == "SELL":
        ikon = "📉 SELL (SHORT)"
        warna_garis = "─"
    else:
        ikon = "⏳ WAIT — Tidak ada setup yang valid saat ini"
        warna_garis = "·"

    print()
    _garis()
    print(f"  KEPUTUSAN AKHIR: {ikon}")
    _garis()

    print(f"  Waktu evaluasi : {hasil['waktu_evaluasi']}")
    print(f"  Harga close    : {hasil['close']:.2f}")
    print(f"  Konfirmasi     : {hasil['konfirmasi_terpenuhi']} / "
          f"{hasil['konfirmasi_dibutuhkan']} kondisi terpenuhi")

    # Tampilkan alasan entry
    if hasil["alasan_entry"]:
        print()
        print("  ALASAN ENTRY:")
        for alasan in hasil["alasan_entry"]:
            print(f"    ✅ {alasan}")

    # Tampilkan alasan wait
    if hasil["alasan_wait"]:
        print()
        print("  ALASAN WAIT / DITOLAK:")
        for alasan in hasil["alasan_wait"]:
            # Tandai dengan ikon berbeda tergantung apakah ini kondisi ok atau blokir
            if alasan.startswith("[Kondisi OK]"):
                print(f"    ✅ {alasan}")
            elif alasan.startswith("[RSI BLOKIR]") or alasan.startswith("[Terpenuhi]"):
                print(f"    🚫 {alasan}")
            else:
                print(f"    ❌ {alasan}")


def _cetak_breakdown(hasil: dict):
    """Tampilkan breakdown detail setiap kondisi untuk audit."""

    print()
    _garis("-")
    print("  BREAKDOWN KONDISI (AUDIT)")
    _garis("-")

    detail = hasil["kondisi_detail"]

    # ── Kondisi 1: Trend + EMA ────────────────────────────────────────────
    c = detail["trend_and_ema"]
    status = "✅ TERPENUHI" if c["terpenuhi"] else "❌ TIDAK TERPENUHI"
    print(f"\n  [1] Trend + EMA Alignment  →  {status}")
    print(f"      Trend label : {c['_trend_label']}")
    print(f"      EMA cross   : {c['_ema_cross']}")
    print(f"      Arah        : {c['arah']}")
    print(f"      Keterangan  : {c['keterangan']}")

    # ── Filter: RSI ───────────────────────────────────────────────────────
    r = detail["rsi_filter"]
    if r["memblokir"]:
        rsi_status = "🚫 MEMBLOKIR ENTRY"
    else:
        rsi_status = "✅ TIDAK MEMBLOKIR"
    print(f"\n  [F] RSI Filter             →  {rsi_status}")
    print(f"      Nilai RSI   : {r['rsi']:.1f}")
    print(f"      Zona        : {r['zona']}")
    print(f"      Keterangan  : {r['keterangan']}")


def _cetak_skenario(nama: str, signals_tiruan: dict):
    """
    Jalankan rule engine dengan data tiruan untuk memverifikasi logika.
    Berguna untuk memastikan logika benar tanpa harus menunggu kondisi market tertentu.
    """
    hasil = evaluate_entry(signals_tiruan)
    keputusan = hasil["keputusan"]

    ikon_map = {"BUY": "📈 BUY ", "SELL": "📉 SELL", "WAIT": "⏳ WAIT"}
    ikon = ikon_map.get(keputusan, keputusan)

    # Ambil alasan utama
    alasan = (hasil["alasan_entry"] + hasil["alasan_wait"])
    alasan_str = alasan[0] if alasan else "-"
    # Potong jika terlalu panjang
    if len(alasan_str) > 55:
        alasan_str = alasan_str[:52] + "..."

    print(f"  {nama:<30} │ {ikon} │ {alasan_str}")


# =============================================================================
# MAIN
# =============================================================================

print()
_garis()
print("  🔬  TEST RULE ENGINE — Evaluasi Kondisi Entry XAUUSD M5")
_garis()


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 1: EVALUASI DATA REAL DARI MT5
# ─────────────────────────────────────────────────────────────────────────────

print("\n[ STEP 1 ] Menghubungkan ke MT5...")
if not initialize_mt5():
    print("❌ MT5 tidak terhubung. Pastikan MT5 sudah terbuka dan login.")
    sys.exit(1)

print("\n[ STEP 2 ] Mengambil dan memproses data...")
df = get_candles()
if df is None:
    shutdown_mt5()
    sys.exit(1)

df      = run_all_indicators(df)
signals = get_latest_signals(df)

print("\n[ STEP 3 ] Menjalankan rule engine...")
hasil = evaluate_entry(signals)

# Tampilkan ringkasan indikator terbaru
print()
_garis()
print("  📊  KONDISI MARKET TERBARU (INPUT KE RULE ENGINE)")
_garis()
print(f"  Waktu    : {signals['time']}")
print(f"  Close    : {signals['close']:.2f}")
print(f"  EMA 9    : {signals['ema_9']:.2f}")
print(f"  EMA 21   : {signals['ema_21']:.2f}")
print(f"  RSI 14   : {signals['rsi_14']:.1f}")
print(f"  Trend    : {signals['trend']}")
print(f"  EMA Gap% : {signals['ema_gap_pct']:+.4f}%")

# Tampilkan keputusan akhir
_cetak_keputusan(hasil)

# Tampilkan breakdown kondisi
_cetak_breakdown(hasil)


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 2: SIMULASI SKENARIO (untuk verifikasi logika)
# ─────────────────────────────────────────────────────────────────────────────
# Kita buat beberapa skenario tiruan dengan nilai yang kita kontrol sendiri.
# Tujuannya: memastikan logika rule engine benar tanpa menunggu kondisi
# market tertentu secara nyata.

from datetime import datetime, timezone

def _buat_signals(trend, ema_9, ema_21, rsi, close=4000.0):
    """Helper singkat untuk buat dict signals tiruan."""
    return {
        "time"        : datetime.now(tz=timezone.utc),
        "close"       : close,
        "ema_9"       : ema_9,
        "ema_21"      : ema_21,
        "rsi_14"      : rsi,
        "trend"       : trend,
        "ema_gap_pct" : (ema_9 - ema_21) / ema_21 * 100,
    }

print()
_garis()
print("  🧪  SIMULASI SKENARIO (Verifikasi Logika)")
_garis()
print()
print(f"  {'Skenario':<30} │ {'Keputusan':<9} │ {'Alasan Utama'}")
print(f"  {'-'*30}─┼─{'─'*9}─┼─{'─'*50}")

# Skenario 1: Uptrend normal — harus BUY
_cetak_skenario(
    "1. Uptrend, RSI netral (55)",
    _buat_signals("UPTREND",   ema_9=4030.0, ema_21=4025.0, rsi=55.0)
)

# Skenario 2: Downtrend normal — harus SELL
_cetak_skenario(
    "2. Downtrend, RSI netral (45)",
    _buat_signals("DOWNTREND", ema_9=4020.0, ema_21=4025.0, rsi=45.0)
)

# Skenario 3: Uptrend + RSI overbought — harus WAIT (RSI blokir)
_cetak_skenario(
    "3. Uptrend, RSI overbought (75)",
    _buat_signals("UPTREND",   ema_9=4030.0, ema_21=4025.0, rsi=75.0)
)

# Skenario 4: Downtrend + RSI oversold — harus WAIT (RSI blokir)
_cetak_skenario(
    "4. Downtrend, RSI oversold (25)",
    _buat_signals("DOWNTREND", ema_9=4020.0, ema_21=4025.0, rsi=25.0)
)

# Skenario 5: Sideways — harus WAIT
_cetak_skenario(
    "5. Sideways, RSI netral (50)",
    _buat_signals("SIDEWAYS",  ema_9=4025.0, ema_21=4025.0, rsi=50.0)
)

# Skenario 6: RSI tepat di batas (70.0) — harus BUY (batas TIDAK memblokir)
_cetak_skenario(
    "6. Uptrend, RSI tepat di batas (70)",
    _buat_signals("UPTREND",   ema_9=4030.0, ema_21=4025.0, rsi=70.0)
)

# Skenario 7: RSI sedikit di atas batas (70.1) — harus WAIT
_cetak_skenario(
    "7. Uptrend, RSI 70.1 (blokir)",
    _buat_signals("UPTREND",   ema_9=4030.0, ema_21=4025.0, rsi=70.1)
)

print()
print("  💡 Skenario 6 & 7 memverifikasi batas RSI: > 70 diblokir, tepat 70 tidak.")

# ─────────────────────────────────────────────────────────────────────────────
# Tutup koneksi MT5
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[ STEP 4 ] Menutup koneksi MT5...")
shutdown_mt5()
print()
_garis()
print("  ✅  Test selesai!")
_garis()
print()
