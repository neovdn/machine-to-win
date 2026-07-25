"""
scripts/compare_swing_wing.py
==============================
Analisis perbandingan nilai SWING_WING (3 vs 5 vs 8) menggunakan data real MT5.

TUJUAN:
    Sebelum mengubah nilai default SWING_WING di risk_manager.py, kita perlu
    melihat dampak nyata dari setiap pilihan terhadap:
    1. Level swing yang berhasil ditemukan (harga dan posisi candle-nya)
    2. Apakah swing ditemukan atau fallback ke ATR
    3. Selisih level SL final yang dihasilkan (dalam dollar)

INI BUKAN TEST REGULER:
    Script ini dijalankan satu kali untuk membantu pengambilan keputusan parameter.
    Setelah keputusan diambil, script ini tidak perlu dijalankan lagi secara rutin.
    (Berbeda dengan test_risk_manager.py yang dijalankan setiap ada perubahan kode.)

CARA PAKAI:
    python scripts/compare_swing_wing.py
"""

import sys
import os

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from engine.data_fetcher  import initialize_mt5, get_candles, shutdown_mt5
from engine.indicators    import run_all_indicators, get_latest_signals
from engine.risk_manager  import find_nearest_swing, calculate_sl_tp, SWING_LOOKBACK, SWING_BUFFER


# =============================================================================
# HELPERS TAMPILAN
# =============================================================================

def _garis(char="=", lebar=70):
    print(char * lebar)

def _hdr(judul):
    print()
    _garis()
    print(f"  {judul}")
    _garis()


# =============================================================================
# FUNGSI ANALISIS: satu wing, dua arah (BUY & SELL)
# =============================================================================

def _analisis_satu_wing(df: pd.DataFrame, entry: float, wing: int) -> dict:
    """
    Jalankan find_nearest_swing() dan calculate_sl_tp() untuk satu nilai wing.

    APA YANG DIHITUNG:
        - Swing low (untuk BUY): level harga terdekat yang bisa jadi referensi SL
        - Swing high (untuk SELL): level harga terdekat yang bisa jadi referensi SL
        - Untuk masing-masing: hitung juga posisi candle-nya (ke berapa dari akhir)
        - Hitung SL final (dengan buffer) dan jarak SL dari entry

    Return: dict dengan hasil BUY dan SELL
    """
    hasil = {}

    for arah in ("BUY", "SELL"):
        col = "low" if arah == "BUY" else "high"

        # ── Cari swing level ─────────────────────────────────────────────────
        swing_level = find_nearest_swing(df, arah, lookback=SWING_LOOKBACK, wing=wing)

        # ── Cari posisi candle dari belakang ─────────────────────────────────
        posisi_candle = None
        if swing_level is not None:
            # Window yang dicari oleh find_nearest_swing (sama persis dengan implementasinya)
            data_window = df.iloc[-(SWING_LOOKBACK + wing * 2):-1].copy()
            n = len(data_window)

            # Iterasi dari akhir (sama dengan find_nearest_swing)
            for i in range(n - 1 - wing, wing - 1, -1):
                val = data_window[col].iloc[i]
                if abs(val - swing_level) < 0.001:   # float comparison dengan toleransi
                    # Posisi relatif dari candle terbaru: 0 = df.iloc[-1], 1 = df.iloc[-2], dst.
                    # data_window[-1] = df.iloc[-2] (karena window pakai :-1)
                    # data_window[i]  = df.iloc[-(n - i) - 1]
                    posisi_dari_terakhir = (n - 1 - i) + 1  # +1 karena window exclude df.iloc[-1]
                    posisi_candle = posisi_dari_terakhir
                    break

        # ── Hitung SL final (sama logika dengan calculate_sl_tp) ─────────────
        if swing_level is not None:
            if arah == "BUY":
                sl_swing = swing_level - SWING_BUFFER
                sl_final = sl_swing   # simplified: asumsikan swing < ATR (biasanya benar untuk wing besar)
            else:
                sl_swing = swing_level + SWING_BUFFER
                sl_final = sl_swing

            jarak_sl = abs(entry - sl_final)
        else:
            sl_swing  = None
            sl_final  = None
            jarak_sl  = None

        # ── Ambil SL final yang sebenarnya (dengan ATR comparison) ───────────
        risk = calculate_sl_tp(df, entry=entry, arah=arah, swing_lookback=SWING_LOOKBACK,
                               swing_buffer=SWING_BUFFER)
        # Override dengan hasil resmi dari calculate_sl_tp (sudah handle ATR vs swing)
        sl_final_resmi  = risk["sl"]
        jarak_sl_resmi  = risk["jarak_sl"]
        metode          = risk["sl_method"]

        hasil[arah] = {
            "swing_level"    : swing_level,           # harga swing (sebelum buffer), None jika tidak ada
            "sl_swing"       : sl_swing,               # swing level + buffer
            "sl_final"       : sl_final_resmi,         # SL final (memenangkan ATR vs swing)
            "jarak_sl"       : jarak_sl_resmi,         # jarak SL dari entry dalam dollar
            "metode"         : metode,                 # "SWING" atau "ATR"
            "posisi_candle"  : posisi_candle,          # berapa candle dari terakhir
            "ditemukan"      : swing_level is not None,
        }

    return hasil


# =============================================================================
# MAIN
# =============================================================================

_hdr("PERBANDINGAN SWING_WING: 3 vs 5 vs 8  —  Data Real XAUUSD M5")

# ── Sambungkan ke MT5 dan ambil data ─────────────────────────────────────────
print("\n[ STEP 1 ] Menghubungkan ke MT5...")
if not initialize_mt5():
    print("❌ MT5 tidak terhubung.")
    sys.exit(1)

print("[ STEP 2 ] Mengambil dan memproses data...")
df = get_candles()
if df is None:
    shutdown_mt5()
    sys.exit(1)

df      = run_all_indicators(df)
signals = get_latest_signals(df)
entry   = signals["close"]

print(f"\n  Entry (close candle terbaru) : {entry:.2f}")
print(f"  ATR 14 saat ini              : {df['atr_14'].iloc[-1]:.2f}")
print(f"  Jumlah candle data           : {len(df)}")
print(f"  Lookback yang digunakan      : {SWING_LOOKBACK} candle")


# ── Jalankan analisis untuk ketiga wing ──────────────────────────────────────
WING_CANDIDATES = [3, 5, 8]
semua_hasil = {}

for wing in WING_CANDIDATES:
    semua_hasil[wing] = _analisis_satu_wing(df, entry, wing)


# =============================================================================
# TAMPILAN PERBANDINGAN — BUY (Swing Low)
# =============================================================================

_hdr("PERBANDINGAN SWING LOW  (Referensi SL untuk BUY)")

print(f"""
  Penjelasan kolom:
    Wing       : Nilai SWING_WING yang diuji
    Artinya    : Swing butuh N candle kiri + N kanan lebih tinggi/rendah
    Swing Low  : Harga "lembah" terendah yang ditemukan (sebelum buffer)
    SL Final   : Harga Stop Loss yang dipakai (swing/ATR mana yang lebih jauh)
    Jarak SL   : Jarak SL dari entry dalam dollar (makin besar = SL lebih jauh)
    Posisi     : Candle tersebut ada berapa candle ke belakang dari candle terbaru
    Metode     : SWING atau ATR (ATR = swing tidak ditemukan, pakai fallback)
""")

print(f"  {'Wing':<6} {'Swing Low':>10} {'SL Final':>10} {'Jarak SL':>10} "
      f"{'Posisi':>8} {'Metode':<8} {'Window (menit)'}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*15}")

for wing in WING_CANDIDATES:
    h = semua_hasil[wing]["BUY"]
    swing_str  = f"{h['swing_level']:.2f}"  if h["ditemukan"] else "tidak ada"
    sl_str     = f"{h['sl_final']:.2f}"     if h["sl_final"] is not None else "-"
    jarak_str  = f"{h['jarak_sl']:.2f}"     if h["jarak_sl"] is not None else "-"
    posisi_str = f"-{h['posisi_candle']}c"  if h["posisi_candle"] is not None else "N/A"
    window_mnt = (wing * 2 + 1) * 5         # jumlah candle × 5 menit
    print(f"  {wing:<6} {swing_str:>10} {sl_str:>10} {jarak_str:>10} "
          f"{posisi_str:>8} {h['metode']:<8} ({window_mnt} menit = {window_mnt/60:.1f} jam)")


# =============================================================================
# TAMPILAN PERBANDINGAN — SELL (Swing High)
# =============================================================================

_hdr("PERBANDINGAN SWING HIGH  (Referensi SL untuk SELL)")

print(f"  {'Wing':<6} {'Swing High':>10} {'SL Final':>10} {'Jarak SL':>10} "
      f"{'Posisi':>8} {'Metode':<8} {'Window (menit)'}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*15}")

for wing in WING_CANDIDATES:
    h = semua_hasil[wing]["SELL"]
    swing_str  = f"{h['swing_level']:.2f}"  if h["ditemukan"] else "tidak ada"
    sl_str     = f"{h['sl_final']:.2f}"     if h["sl_final"] is not None else "-"
    jarak_str  = f"{h['jarak_sl']:.2f}"     if h["jarak_sl"] is not None else "-"
    posisi_str = f"-{h['posisi_candle']}c"  if h["posisi_candle"] is not None else "N/A"
    window_mnt = (wing * 2 + 1) * 5
    print(f"  {wing:<6} {swing_str:>10} {sl_str:>10} {jarak_str:>10} "
          f"{posisi_str:>8} {h['metode']:<8} ({window_mnt} menit = {window_mnt/60:.1f} jam)")


# =============================================================================
# PERBANDINGAN DAMPAK KE SL: selisih jarak antar wing
# =============================================================================

_hdr("DAMPAK KE JARAK SL  (Selisih Dollar Antar Wing)")

print("  Berapa dollar perbedaan SL jika ganti wing?")
print()

for arah in ("BUY", "SELL"):
    print(f"  === {arah} ===")
    jarak = {w: semua_hasil[w][arah]["jarak_sl"] for w in WING_CANDIDATES}

    # Selisih relatif dari wing=3 sebagai baseline
    baseline = jarak[3]
    if baseline is not None:
        print(f"  Wing 3 → SL jarak {baseline:.2f} USD  (baseline)")
        for w in [5, 8]:
            if jarak[w] is not None:
                selisih = jarak[w] - baseline
                sign = "+" if selisih >= 0 else ""
                pct = selisih / baseline * 100
                print(f"  Wing {w} → SL jarak {jarak[w]:.2f} USD  ({sign}{selisih:.2f} USD, {sign}{pct:.1f}% dari wing 3)")
            else:
                print(f"  Wing {w} → N/A")
    print()


# =============================================================================
# SCAN SEMUA SWING YANG ADA DI LOOKBACK — untuk tiap wing
# =============================================================================
# Ini yang paling penting: berapa banyak swing TOTAL yang bisa ditemukan
# dalam 50 candle lookback untuk setiap nilai wing?

_hdr("JUMLAH SWING DALAM 50 CANDLE LOOKBACK")

print("  Menghitung semua swing (tidak hanya yang terdekat)...\n")

def _hitung_semua_swing(df: pd.DataFrame, arah: str, wing: int,
                        lookback: int = SWING_LOOKBACK) -> list:
    """
    Hitung SEMUA swing dalam lookback window (bukan hanya terdekat).
    Return: list of (index_posisi, harga_swing)
    """
    col = "low" if arah == "BUY" else "high"
    data = df.iloc[-(lookback + wing * 2):-1].copy()
    n    = len(data)

    swings_found = []
    for i in range(n - 1 - wing, wing - 1, -1):
        val          = data[col].iloc[i]
        window_slice = data[col].iloc[i - wing : i + wing + 1]

        if arah == "BUY" and val == window_slice.min():
            candle_posisi = (n - 1 - i) + 1
            swings_found.append((candle_posisi, float(val)))
        elif arah == "SELL" and val == window_slice.max():
            candle_posisi = (n - 1 - i) + 1
            swings_found.append((candle_posisi, float(val)))

    return swings_found

print(f"  {'Wing':<6} {'Swing Low (BUY)':>20} {'Swing High (SELL)':>20}")
print(f"  {'-'*6} {'-'*20} {'-'*20}")

swing_data = {}
for wing in WING_CANDIDATES:
    swings_low  = _hitung_semua_swing(df, "BUY",  wing)
    swings_high = _hitung_semua_swing(df, "SELL", wing)
    swing_data[wing] = {"BUY": swings_low, "SELL": swings_high}
    print(f"  {wing:<6} {len(swings_low):>3} swing ditemukan   {len(swings_high):>3} swing ditemukan")

# Detail swing untuk setiap wing
for wing in WING_CANDIDATES:
    print()
    window_mnt = (wing * 2 + 1) * 5
    print(f"  ── Wing = {wing}  (konfirmasi {wing} candle kiri+kanan, window {window_mnt} menit) ──")

    for arah, label in [("BUY", "Swing Low "), ("SELL", "Swing High")]:
        swings = swing_data[wing][arah]
        if not swings:
            print(f"     {label}: tidak ada dalam {SWING_LOOKBACK} candle lookback → FALLBACK ATR")
        else:
            for posisi, harga in swings[:5]:  # tampilkan maks 5 swing
                jarak_dari_entry = abs(entry - harga)
                print(f"     {label}: {harga:.2f}  (candle -{posisi:>2} dari sekarang, "
                      f"jarak {jarak_dari_entry:.2f} dari entry)")
            if len(swings) > 5:
                print(f"     {label}: ... dan {len(swings)-5} swing lagi")


# =============================================================================
# REKOMENDASI BERDASARKAN DATA
# =============================================================================

_hdr("REKOMENDASI BERDASARKAN DATA NYATA")

# Hitung statistik untuk dasar rekomendasi
n3_low  = len(swing_data[3]["BUY"])
n5_low  = len(swing_data[5]["BUY"])
n8_low  = len(swing_data[8]["BUY"])
n3_high = len(swing_data[3]["SELL"])
n5_high = len(swing_data[5]["SELL"])
n8_high = len(swing_data[8]["SELL"])

# Tentukan rekomendasi berdasarkan ketersediaan swing
def _pilih_rekomendasi():
    # Kriteria:
    # 1. Minimal harus ada 1 swing ditemukan untuk BUY dan SELL
    # 2. Lebih besar wing = lebih signifikan, tapi lebih sedikit swing
    kandidat = []
    for wing in WING_CANDIDATES:
        low_count  = len(swing_data[wing]["BUY"])
        high_count = len(swing_data[wing]["SELL"])
        if low_count >= 1 and high_count >= 1:
            kandidat.append(wing)

    if not kandidat:
        return 3, "Tidak ada wing yang berhasil menemukan swing untuk kedua arah. Pakai wing=3 (paling sensitif)."
    elif len(kandidat) == 1:
        w = kandidat[0]
        return w, f"Hanya wing={w} yang berhasil menemukan swing untuk BUY dan SELL."
    else:
        # Pilih wing terbesar yang masih menemukan swing
        best = max(kandidat)
        return best, f"Wing={best} adalah yang paling signifikan secara visual dan masih menemukan swing."

wing_rekomendasi, alasan = _pilih_rekomendasi()

print(f"""
  DATA YANG DIKUMPULKAN:
  ┌──────────┬────────────────────┬─────────────────────┐
  │ Wing     │  Swing Low (BUY)   │  Swing High (SELL)  │
  ├──────────┼────────────────────┼─────────────────────┤
  │ wing = 3 │  {n3_low:>2} swing           │   {n3_high:>2} swing           │
  │ wing = 5 │  {n5_low:>2} swing           │   {n5_high:>2} swing           │
  │ wing = 8 │  {n8_low:>2} swing           │   {n8_high:>2} swing           │
  └──────────┴────────────────────┴─────────────────────┘

  REKOMENDASI: SWING_WING = {wing_rekomendasi}
  Alasan     : {alasan}

  CATATAN PENTING:
  - Ini adalah snapshot data satu waktu — kondisi market saat ini.
  - Di kondisi market berbeda (sideways vs trending), jumlah swing bisa berbeda.
  - Wing yang lebih besar = SL yang lebih bermakna secara visual, tapi
    jika swing tidak ditemukan, sistem fallback ke ATR (masih aman).
  - Setelah mengubah default, selalu jalankan test_risk_manager.py.
""")

print("[ STEP 3 ] Menutup koneksi MT5...")
shutdown_mt5()
_garis()
print("  ✅  Analisis selesai!")
_garis()
