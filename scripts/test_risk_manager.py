"""
scripts/test_risk_manager.py
============================
Script validasi end-to-end: MT5 → Indikator → Rule Engine → SL/TP/RRR.

CARA PAKAI:
    1. Pastikan MT5 sudah terbuka dan login
    2. Jalankan dari root project:
       python scripts/test_risk_manager.py

OUTPUT YANG DITAMPILKAN:
    1. Kondisi market terbaru (dari indikator)
    2. Keputusan rule engine (BUY/SELL/WAIT)
    3. Level SL, TP, dan RRR (jika BUY atau SELL)
    4. Simulasi skenario BUY dan SELL untuk verifikasi logika min/max

CARA MEMBANDINGKAN DENGAN CHART MT5:
    - Tandai level entry, SL, dan TP di chart XAUUSD M5
    - Perhatikan: apakah SL berada di belakang swing low/high terdekat?
    - Perhatikan: apakah jarak SL masuk akal (tidak terlalu sempit/lebar)?
"""

import sys
import os

# Encoding UTF-8 agar emoji tampil di Windows terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Tambahkan root project ke Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.data_fetcher  import initialize_mt5, get_candles, shutdown_mt5
from engine.indicators    import run_all_indicators, get_latest_signals
from engine.rule_engine   import evaluate_entry
from engine.risk_manager  import calculate_sl_tp, find_nearest_swing


# =============================================================================
# HELPERS TAMPILAN
# =============================================================================

def _garis(char="=", lebar=65):
    print(char * lebar)


def _cetak_sl_tp(risk: dict, arah: str):
    """Tampilkan hasil kalkulasi SL/TP secara terstruktur."""

    ikon_arah = "📈 BUY (LONG)" if arah == "BUY" else "📉 SELL (SHORT)"

    print()
    _garis()
    print(f"  💰  LEVEL SL / TP / RRR  —  {ikon_arah}")
    _garis()

    print(f"  Entry        : {risk['entry']:.2f}")
    print()

    # Tampilkan SL dengan arah yang jelas
    sl_dir = "↓ di bawah entry" if arah == "BUY" else "↑ di atas entry"
    print(f"  Stop Loss    : {risk['sl']:.2f}  ({sl_dir})")
    print(f"  Jarak SL     : {risk['jarak_sl']:.2f} dollar")
    print(f"  Metode SL    : {risk['sl_method']}")

    print()
    tp_dir = "↑ di atas entry" if arah == "BUY" else "↓ di bawah entry"
    print(f"  Take Profit  : {risk['tp']:.2f}  ({tp_dir})")
    print(f"  Jarak TP     : {risk['jarak_tp']:.2f} dollar")
    print()
    print(f"  RRR          : 1 : {risk['rrr']:.1f}")

    print()
    _garis("-")
    print("  AUDIT SL (dari mana angka ini berasal)")
    _garis("-")
    print(f"  ATR(14)      : {risk['atr_value']:.2f}")
    print(f"  SL via ATR   : {risk['sl_atr_level']:.2f}")

    if risk["sl_swing_raw"] is not None:
        swing_label = "Swing Low" if arah == "BUY" else "Swing High"
        print(f"  {swing_label:<13}: {risk['sl_swing_raw']:.2f}  (raw, sebelum buffer)")
        print(f"  SL via Swing : {risk['sl_swing_level']:.2f}  (setelah buffer)")
    else:
        print(f"  Swing        : tidak ditemukan dalam lookback window")

    print()
    print(f"  Dipilih      : {risk['sl_method']}")
    print(f"  Alasan       : {risk['pesan']}")


def _cetak_skenario_sltp(label, entry, arah, df_subset):
    """Jalankan kalkulasi SL/TP dengan data tiruan dan tampilkan ringkasannya."""
    risk = calculate_sl_tp(df_subset, entry=entry, arah=arah, rrr_min=2.0)
    sl   = risk["sl"]
    tp   = risk["tp"]
    rrr  = risk["rrr"]
    met  = risk["sl_method"]

    # Verifikasi arah: SL harus di sisi yang benar dari entry
    sl_valid = (arah == "BUY" and sl < entry) or (arah == "SELL" and sl > entry)
    tp_valid = (arah == "BUY" and tp > entry) or (arah == "SELL" and tp < entry)
    arah_ok  = "✅" if (sl_valid and tp_valid) else "❌ ARAH SALAH!"

    print(f"  {label:<34} │ SL={sl:.2f}  TP={tp:.2f}  RRR={rrr:.1f}  [{met}] {arah_ok}")


# =============================================================================
# MAIN — TEST DENGAN DATA REAL MT5
# =============================================================================

print()
_garis()
print("  🔬  TEST RISK MANAGER — SL / TP / RRR  (XAUUSD M5)")
_garis()

print("\n[ STEP 1 ] Menghubungkan ke MT5...")
if not initialize_mt5():
    print("❌ MT5 tidak terhubung.")
    sys.exit(1)

print("\n[ STEP 2 ] Mengambil dan memproses data...")
df = get_candles()
if df is None:
    shutdown_mt5()
    sys.exit(1)

df_h1 = get_candles(timeframe_str="H1", count=100)
if df_h1 is None:
    shutdown_mt5()
    sys.exit(1)

df    = run_all_indicators(df)      # → tambah ema_9, ema_21, rsi_14, trend, atr_14
df_h1 = run_all_indicators(df_h1)   # → H1 indicators

print("\n[ STEP 3 ] Evaluasi kondisi entry...")
signals = get_latest_signals(df)
signals["trend_h1"] = get_latest_signals(df_h1)["trend"]
hasil   = evaluate_entry(signals)


# ── Tampilkan ringkasan kondisi market ──────────────────────────────────────
print()
_garis()
print("  📊  KONDISI MARKET TERBARU")
_garis()
print(f"  Waktu    : {signals['time']}")
print(f"  Close    : {signals['close']:.2f}")
print(f"  EMA 9    : {signals['ema_9']:.2f}")
print(f"  EMA 21   : {signals['ema_21']:.2f}")
print(f"  RSI 14   : {signals['rsi_14']:.1f}")
print(f"  Trend    : {signals['trend']}")
print(f"  ATR 14   : {df['atr_14'].iloc[-1]:.2f}")

# ── Keputusan rule engine ────────────────────────────────────────────────────
keputusan = hasil["keputusan"]
arah      = hasil["arah"]   # "LONG", "SHORT", atau None

print()
_garis()
print(f"  ⚖️   KEPUTUSAN RULE ENGINE: {keputusan}")
_garis()
for alasan in (hasil["alasan_entry"] + hasil["alasan_wait"]):
    ikon = "✅" if alasan.startswith("[Terpenuhi]") or not alasan.startswith("[") else \
           "🚫" if "BLOKIR" in alasan else \
           "✅" if "tidak memblokir" in alasan.lower() or "terpenuhi" in alasan.lower() else "❌"
    print(f"  {ikon} {alasan}")

# ── Kalkulasi SL/TP hanya jika ada entry ────────────────────────────────────
if keputusan in ("BUY", "SELL"):

    print(f"\n[ STEP 4 ] Menghitung SL/TP untuk {keputusan}...")
    entry = signals["close"]

    risk = calculate_sl_tp(
        df             = df,
        entry          = entry,
        arah           = keputusan,
        rrr_min        = 2.0,
        atr_multiplier = 1.5,
        swing_lookback = 50,
        swing_buffer   = 0.50,
    )

    _cetak_sl_tp(risk, keputusan)

else:
    print()
    print("  ⏳  WAIT — tidak ada setup valid. SL/TP tidak dihitung.")
    print("       Jalankan script ini lagi saat ada BUY atau SELL signal.")


# =============================================================================
# SIMULASI SKENARIO — Verifikasi Logika min() dan max()
# =============================================================================
# Kita paksa skenario BUY dan SELL menggunakan data real dari MT5,
# hanya ubah parameter 'arah' untuk memastikan logika SL benar.
#
# Yang diverifikasi:
#   1. BUY : SL harus di BAWAH entry, TP di ATAS entry
#   2. SELL: SL harus di ATAS entry, TP di BAWAH entry
#   3. BUY menggunakan min() untuk pilih SL, bukan max()
#   4. SELL menggunakan max() untuk pilih SL, bukan min()

print()
_garis()
print("  🧪  SIMULASI SKENARIO (Verifikasi Arah SL/TP)")
_garis()
print()
print(f"  {'Skenario':<34} │ Detail")
print(f"  {'─'*34}─┼─{'─'*40}")

entry_sim = signals["close"]

_cetak_skenario_sltp(
    f"BUY  @ {entry_sim:.2f}  (SL harus < entry)",
    entry=entry_sim, arah="BUY", df_subset=df
)
_cetak_skenario_sltp(
    f"SELL @ {entry_sim:.2f}  (SL harus > entry)",
    entry=entry_sim, arah="SELL", df_subset=df
)

# Verifikasi RRR: TP harus = entry ± (jarak_SL × 2)
r_buy  = calculate_sl_tp(df, entry=entry_sim, arah="BUY",  rrr_min=2.0)
r_sell = calculate_sl_tp(df, entry=entry_sim, arah="SELL", rrr_min=2.0)

buy_rrr_ok  = abs(r_buy["jarak_tp"]  - r_buy["jarak_sl"]  * 2.0) < 0.01
sell_rrr_ok = abs(r_sell["jarak_tp"] - r_sell["jarak_sl"] * 2.0) < 0.01
print()
print(f"  RRR BUY  check (jarak_TP = 2 × jarak_SL): {'✅ BENAR' if buy_rrr_ok  else '❌ SALAH'}")
print(f"  RRR SELL check (jarak_TP = 2 × jarak_SL): {'✅ BENAR' if sell_rrr_ok else '❌ SALAH'}")

# ── Informasi swing yang ditemukan ──────────────────────────────────────────
print()
_garis("-")
print("  📌  SWING LEVEL YANG DITEMUKAN (lookback 50 candle)")
_garis("-")
swing_low  = find_nearest_swing(df, "BUY")
swing_high = find_nearest_swing(df, "SELL")
print(f"  Swing Low  (referensi SL BUY)  : "
      f"{swing_low:.2f}"  if swing_low  is not None else "  Swing Low  : tidak ditemukan")
print(f"  Swing High (referensi SL SELL) : "
      f"{swing_high:.2f}" if swing_high is not None else "  Swing High : tidak ditemukan")
print(f"  Entry (close terbaru)          : {entry_sim:.2f}")

if swing_low and swing_high:
    print()
    print(f"  ✅ Swing low  ada DI BAWAH entry: {swing_low  < entry_sim}")
    print(f"  ✅ Swing high ada DI ATAS entry : {swing_high > entry_sim}")

# ── Penutup ──────────────────────────────────────────────────────────────────
print()
print("[ STEP 5 ] Menutup koneksi MT5...")
shutdown_mt5()
print()
_garis()
print("  ✅  Test selesai!")
_garis()
print()
print("  CARA VERIFIKASI MANUAL DI CHART MT5:")
print("  1. Buka chart XAUUSD M5")
print("  2. Tandai level entry, SL, dan TP dari output di atas")
print("  3. Periksa: apakah SL berada di belakang swing low/high yang terlihat?")
print("  4. Periksa: apakah jarak SL masuk akal vs kondisi volatilitas saat ini?")
print()
