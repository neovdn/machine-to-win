"""
engine/rule_engine.py
=====================
Modul evaluasi kondisi entry berdasarkan framework trading XAUUSD M5.

CARA KERJA:
    1. Terima dict dari get_latest_signals() (hasil indikator terbaru)
    2. Evaluasi setiap kondisi secara terpisah — masing-masing jadi fungsi kecil
    3. Gabungkan semua hasil → tentukan keputusan BUY / SELL / WAIT
    4. Kembalikan dict lengkap berisi keputusan + breakdown tiap kondisi

FILOSOFI DESAIN ("Checklist Kondisi"):
    Daripada satu if-else besar yang sulit dibaca, tiap kondisi dipecah menjadi
    fungsi kecil sendiri (_check_...). Setiap fungsi mengembalikan dict berisi:
        - "terpenuhi" : bool   → apakah kondisi ini sudah terpenuhi?
        - "arah"      : str    → "BUY", "SELL", "NETRAL", atau None
        - "keterangan": str    → penjelasan singkat kenapa terpenuhi/tidak
        - (+ field tambahan sesuai kondisi)

    Keuntungan desain ini:
    - Audit mudah: kamu tahu PERSIS kondisi mana yang terpenuhi/tidak
    - Mudah diextend: tambah kondisi baru = tambah satu fungsi baru
    - Mudah ditest: setiap fungsi bisa ditest secara terpisah

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.

KONDISI YANG DIEVALUASI SAAT INI:
    [1] _check_bias_h1        : Arah bias dari timeframe H1
    [2] _check_ema_trigger_m5 : EMA cross timing dari M5
    [F] _check_rsi_filter     : RSI sebagai filter/veto kontekstual (bukan kondisi entry)

RSI FILTER KONTEKSTUAL (versi terbaru):
    Perilaku RSI filter berbeda tergantung kekuatan trend (ema_gap_pct):
    - Trend KUAT  (|ema_gap_pct| >= RSI_STRONG_TREND_EMA_GAP_THRESHOLD):
        RSI overbought/oversold TIDAK memblokir entry — treat sebagai potensi
        continuation. Warning tetap dicatat di field "rsi_warning" untuk audit.
    - Trend LEMAH (|ema_gap_pct| < threshold):
        Veto ketat seperti sebelumnya — RSI ekstrem MEMBATALKAN entry.

CONTEXT WARNINGS (session_filter):
    evaluate_entry() juga menjalankan session_filter untuk mendeteksi:
    - Apakah sesi sedang low liquidity (di luar London/NY overlap)
    - Apakah mendekati market close Jumat (risiko gap weekend)
    Hasil berupa list string di field "context_warnings" — TIDAK mengubah
    keputusan BUY/SELL/WAIT, hanya ditampilkan di UI sebagai informasi konteks.

CARA TAMBAH KONDISI BARU NANTI:
    1. Buat fungsi baru: def _check_NAMAKONDISI(signals): ...
    2. Panggil di dalam evaluate_entry()
    3. Tambahkan hasilnya ke kondisi_detail
    Tidak perlu ubah logika yang sudah ada.
"""

import math
from datetime import datetime, timezone

# session_filter diimport secara lazy di dalam evaluate_entry() untuk menghindari
# circular import dan agar module ini tetap bisa ditest tanpa session_filter.

# candle_patterns diimport secara lazy di dalam calculate_setup_quality() untuk
# menjaga agar modul ini tetap bisa ditest tanpa dependensi candle_patterns.
# (Fase 7: komponen ke-4 confidence scoring)


# =============================================================================
# KONSTANTA KONFIGURASI
# =============================================================================
# Konfigurasi threshold diletakkan di atas, bukan dikubur di dalam fungsi,
# supaya mudah diubah tanpa perlu cari-cari ke dalam kode.

# Threshold RSI untuk zona ekstrem
RSI_OVERBOUGHT = 70.0   # RSI > nilai ini = kondisi overbought → blokir BUY
RSI_OVERSOLD   = 30.0   # RSI < nilai ini = kondisi oversold   → blokir SELL

# Threshold kekuatan trend untuk RSI filter KONTEKSTUAL
#
# Jika |ema_gap_pct| >= nilai ini → trend dianggap KUAT:
#   RSI overbought/oversold TIDAK memblokir entry (potensi continuation).
#   Warning tetap dicatat di breakdown untuk audit.
#
# Jika |ema_gap_pct| < nilai ini → trend LEMAH:
#   Veto ketat seperti biasa — RSI ekstrem membatalkan entry.
#
# Contoh untuk XAUUSD ~$3300:
#   ema_gap_pct = 0.15%  →  jarak EMA ≈ $4.95  (trend yang sudah cukup jelas)
#   ema_gap_pct = 0.05%  →  jarak EMA ≈ $1.65  (masih terlalu tipis)
# Nilai 0.15% bisa dikalibrasi ulang setelah observasi data nyata.
RSI_STRONG_TREND_EMA_GAP_THRESHOLD = 0.15  # dalam persen (abs value)

# Jumlah kondisi ENTRY yang harus terpenuhi untuk menghasilkan BUY/SELL
# (kondisi entry = semua kondisi _check_... KECUALI filter seperti RSI)
#
# Dinaikkan ke 2 setelah split kondisi menjadi dua sumber independen:
#   [1] _check_bias_h1()        — arah dari H1 (timeframe lebih besar)
#   [2] _check_ema_trigger_m5() — timing dari EMA cross M5
# Keduanya harus searah sebelum engine boleh output BUY atau SELL.
#
# ⚠️  DEPRECATED sejak Fase 9 — konstanta ini TIDAK LAGI menentukan logika
# entry. Fase 9 mengganti voting mayoritas dengan arsitektur:
#   bias_h1.terpenuhi AND trigger_valid (minimal 1 dari trigger_group cocok arah)
# Konstanta dipertahankan agar tidak break existing scripts yang mengimpornya,
# tetapi evaluate_entry() tidak menggunakan nilainya untuk keputusan entry.
MINIMUM_CONDITIONS_MET = 2  # DEPRECATED: tidak dipakai di evaluate_entry() sejak Fase 9

# =============================================================================
# KONFIGURASI KONDISI KETIGA: VOLUME PARTICIPATION (Fase 3.2)
# =============================================================================
#
# volume_ratio = tick_volume_candle / rolling50_mean(tick_volume)
# Sumber: kolom 'volume_ratio' dari indicators.calculate_volume_ratio().
#
# FILOSOFI (berbeda dari EMA/RSI):
#   Volume tidak mengandung informasi ARAH (long atau short). Volume hanya
#   mengandung informasi KUALITAS PARTISIPASI pasar.
#   Oleh karena itu dikimplementasi sebagai FILTER/VETO, bukan kondisi entry
#   beridentitas BUY/SELL. Entry hanya diizinkan saat volume_ratio berada
#   di zona medium (bukan ekstrem rendah maupun tinggi):
#   - Terlalu rendah (< LOW_THRESHOLD) = pasar tipis, noise lebih mendominasi.
#   - Terlalu tinggi (> HIGH_THRESHOLD) = potensi climax/exhaustion, bukan kelanjutan.
#   - Zona medium = partisipasi normal, sinyal lebih dapat dipercaya.
#
# THRESHOLD:
#   Nilai berikut diturunkan dari distribusi 14 bulan data (80,958 candle M5):
#     Q25 volume_ratio ≈ 0.71, Q75 ≈ 1.28
#   Threshold dipilih SEDIKIT LEBIH LEBAR dari Q25/Q75 (0.50 dan 1.80) untuk
#   menghindari over-filtering — hanya menolak candle yang benar-benar ekstrem.
#   Sesuai prinsip Fase 3: jangan jadikan bucket kecil (<30 trade) sebagai
#   dasar kalibrasi, gunakan threshold yang robust secara distribusi.
#
# VOLUME_MODE mengontrol cara volume diintegrasikan di evaluate_entry():
#   "FILTER"     : veto saja (tidak menambah konfirmasi), MINIMUM_CONDITIONS_MET=2
#   "CONDITION"  : kondisi ke-3 penuh, bisa menaikkan konfirmasi_terpenuhi ke 3
#
# Untuk walk-forward Fase 3.2: dua varian diuji lewat scripts/run_walk_forward_phase3.py
# yang memanggil evaluate_entry() dengan mode yang berbeda via override.
VOLUME_RATIO_LOW_THRESHOLD  = 0.708  # Batas bawah: Q25 dari distribusi (sebelum walk-forward)
VOLUME_RATIO_HIGH_THRESHOLD = 1.278  # Batas atas : Q75 dari distribusi (sebelum walk-forward)

# Mode integrasi volume di evaluate_entry() -- bisa di-override dari luar.
# JANGAN ubah nilai default ini untuk produksi -- gunakan parameter saat call.
# Mode yang tersedia:
#   "FILTER"    : volume sebagai veto saja (tidak menambah konfirmasi)
#   "CONDITION" : volume sebagai kondisi ke-3 penuh
#   "IGNORE"    : volume tidak dipakai sama sekali (untuk analisis/backtest murni)
VOLUME_MODE_DEFAULT = "FILTER"  # "FILTER", "CONDITION", atau "IGNORE"


# =============================================================================
# KONDISI 1: BIAS ARAH DARI H1
# =============================================================================
# Sumber: detect_trend() dijalankan pada candle H1 (timeframe lebih besar).
# Fungsi ini membaca signals["trend_h1"] yang sudah diisi oleh caller
# (app.py atau script) sebelum memanggil evaluate_entry().
#
# Peran: menentukan ARAH BIAS market secara makro. Timeframe H1 bergerak
# lebih lambat sehingga tren di sini lebih bermakna dan lebih susah dipalsukan
# oleh noise intraday M5.
#
# Catatan: kondisi ini INDEPENDEN dari kondisi EMA M5 di bawah — sumbernya
# berbeda timeframe, bukan turunan satu sama lain.

def _check_bias_h1(signals: dict) -> dict:
    """
    Evaluasi arah bias dari trend H1.

    Membaca signals["trend_h1"] yang merupakan label trend dari candle H1,
    dihitung menggunakan detect_trend() yang sama dengan M5 tapi pada
    DataFrame H1 yang berbeda.

    LOGIKA:
        BUY   ← trend_h1 == "UPTREND"    (bias makro bullish)
        SELL  ← trend_h1 == "DOWNTREND"  (bias makro bearish)
        NETRAL ← trend_h1 == "SIDEWAYS"  (pasar konsolidasi di H1 — tunggu)

    Return dict:
        {
            "terpenuhi"  : bool  — True jika ada arah jelas (UP atau DOWN)
            "arah"       : str   — "BUY", "SELL", atau "NETRAL"
            "keterangan" : str   — penjelasan ringkas
            "trend_h1"   : str   — nilai mentah dari signals["trend_h1"]
        }
    """
    trend_h1 = signals["trend_h1"]

    if trend_h1 == "UPTREND":
        terpenuhi  = True
        arah_final = "BUY"
        keterangan = f"Bias H1: UPTREND (EMA9 > EMA21, Close > EMA21) — bias makro bullish"
    elif trend_h1 == "DOWNTREND":
        terpenuhi  = True
        arah_final = "SELL"
        keterangan = f"Bias H1: DOWNTREND (EMA9 < EMA21, Close < EMA21) — bias makro bearish"
    else:  # SIDEWAYS
        terpenuhi  = False
        arah_final = "NETRAL"
        keterangan = f"Bias H1: SIDEWAYS — pasar konsolidasi di timeframe H1, TUNGGU"

    return {
        "terpenuhi"  : terpenuhi,
        "arah"       : arah_final,
        "keterangan" : keterangan,
        "trend_h1"   : trend_h1,
    }


# =============================================================================
# KONDISI 2: EMA TRIGGER DARI M5
# =============================================================================
# Sumber: signals["trend"] (label trend M5) + signals["ema_9"] / signals["ema_21"]
# dari candle M5 terbaru.
#
# Peran: menentukan TIMING ENTRY — apakah EMA M5 sudah cross ke arah yang
# benar dan harga sudah di posisi yang tepat? Ini "pemicu" yang memvalidasi
# bahwa momentum M5 selaras dengan bias H1.
#
# PENTING — tidak sirkular:
#   signals["trend"] = label trend M5 dari detect_trend(df_m5)
#   EMA cross dievaluasi ULANG dari nilai ema_9 / ema_21 yang sama.
#   Memang derive dari sumber yang sama, TAPI kondisi ini hanya lolos
#   ke output BUY/SELL jika JUGA searah dengan bias_h1 (kondisi 1).
#   Dua kondisi dari sumber berbeda (H1 dan M5) harus konsisten — itulah
#   yang mencegah false confidence.

def _check_ema_trigger_m5(signals: dict) -> dict:
    """
    Evaluasi apakah EMA M5 memberikan sinyal timing entry yang valid.

    Memeriksa trend label M5 (dari detect_trend() pada candle M5) sekaligus
    mengkonfirmasi posisi EMA9 vs EMA21 secara eksplisit untuk audit.

    LOGIKA:
        BUY  ← trend == "UPTREND"   (EMA9 > EMA21, Close > EMA21, gap cukup lebar)
        SELL ← trend == "DOWNTREND" (EMA9 < EMA21, Close < EMA21, gap cukup lebar)
        NETRAL ← trend == "SIDEWAYS" (EMA terlalu dekat atau harga di antara EMA)

    Return dict:
        {
            "terpenuhi"  : bool  — True jika ada arah jelas (UP atau DOWN)
            "arah"       : str   — "BUY", "SELL", atau "NETRAL"
            "keterangan" : str   — penjelasan ringkas termasuk nilai EMA
            "trend_m5"   : str   — nilai mentah dari signals["trend"]
            "_ema_cross"  : str  — "EMA9>EMA21", "EMA9<EMA21", atau "SAMA"
            "ema_gap_pct" : float — gap persen EMA9 vs EMA21 saat ini
        }
    """
    trend   = signals["trend"]        # label trend M5 dari detect_trend(df_m5)
    ema_9   = signals["ema_9"]
    ema_21  = signals["ema_21"]
    gap_pct = signals["ema_gap_pct"]  # sudah dihitung di detect_trend()

    # Konfirmasi arah EMA cross secara eksplisit (untuk audit)
    if ema_9 > ema_21:
        ema_cross = "EMA9>EMA21"
    elif ema_9 < ema_21:
        ema_cross = "EMA9<EMA21"
    else:
        ema_cross = "SAMA"

    # Tentukan arah dan apakah kondisi terpenuhi berdasarkan trend label M5
    # (detect_trend() sudah menerapkan threshold gap — kalau SIDEWAYS di sini
    #  artinya gap terlalu tipis atau harga di antara EMA)
    if trend == "UPTREND":
        terpenuhi  = True
        arah_final = "BUY"
        keterangan = (
            f"Trigger M5: UPTREND — {ema_cross} "
            f"(EMA9={ema_9:.2f} vs EMA21={ema_21:.2f}, gap={gap_pct:+.4f}%)"
        )
    elif trend == "DOWNTREND":
        terpenuhi  = True
        arah_final = "SELL"
        keterangan = (
            f"Trigger M5: DOWNTREND — {ema_cross} "
            f"(EMA9={ema_9:.2f} vs EMA21={ema_21:.2f}, gap={gap_pct:+.4f}%)"
        )
    else:  # SIDEWAYS
        terpenuhi  = False
        arah_final = "NETRAL"
        keterangan = (
            f"Trigger M5: SIDEWAYS — {ema_cross} "
            f"(EMA9={ema_9:.2f} vs EMA21={ema_21:.2f}, gap={gap_pct:+.4f}%) "
            f"— EMA terlalu dekat atau harga di antara EMA, TUNGGU"
        )

    return {
        "terpenuhi"  : terpenuhi,
        "arah"       : arah_final,
        "keterangan" : keterangan,
        "trend_m5"   : trend,
        "_ema_cross"  : ema_cross,
        "ema_gap_pct" : gap_pct,
    }


# =============================================================================
# TRIGGER ALTERNATIF: BREAKOUT DARI ZONA KONSOLIDASI (Fase 9)
# =============================================================================
# Trigger ini sejajar dengan _check_ema_trigger_m5() — keduanya menjawab
# "kapan momen entry yang tepat?", BUKAN "ke arah mana?"
# Arah selalu dari bias_h1 — trigger ini hanya soal timing/konfirmasi.
#
# Kausalitas kritis: zona HARUS dihitung dari candle SEBELUM candle evaluasi.
# Caller (backtester/app.py) melewatkan zone yang sudah dihitung dari idx=i-1.
# Kalau zone dihitung dari idx=i (termasuk candle breakout itu sendiri),
# range zona akan otomatis melebar — circular, bukan breakout sesungguhnya.
#
# Return dict persis pola _check_ema_trigger_m5():
#   terpenuhi  : bool
#   arah       : "BUY" / "SELL" / "NETRAL"
#   keterangan : str (level resistance/support, close, konfirmasi yang terpenuhi)

def _check_breakout_trigger(signals: dict, zone: dict) -> dict:
    """
    Evaluasi apakah candle saat ini menembus zona konsolidasi dengan konfirmasi.

    KRITERIA BREAKOUT VALID (kandidat BUY):
        1. zone["is_valid"] == True
           (zona dihitung dari candle SEBELUM candle ini — anti-circular)
        2. close > zone["resistance"]
           (close menembus — high saja TIDAK cukup, harus close)
        3. Konfirmasi tambahan (OR — salah satu cukup, bukan AND):
           a. volume_ratio >= 1.2  (volume partisipasi tinggi), ATAU
           b. body >= 0.8 * atr_14 (candle punya momentum/body besar)

    KRITERIA BREAKOUT VALID (kandidat SELL):
        Kebalikan arah: close < zone["support"], konfirmasi sama.

    CATATAN PARAMETER:
        Angka 1.2 (volume_ratio) dan 0.8 (body/ATR ratio) adalah titik awal
        kalibrasi — BELUM dioptimasi via backtest. Sama seperti h1_min_ema_gap_pct
        dan parameter Fase 8, nilai ini bisa dikalibrasi setelah validasi edge
        dasarnya terbukti. Tulis ulang di docstring ini jika dikalibrasi.

    Parameter:
        signals : dict dari pipeline — dibutuhkan field:
                  "close"        : float — harga close candle evaluasi
                  "open"         : float — harga open candle evaluasi
                  "atr_14"       : float — ATR 14-period candle evaluasi
                  "volume_ratio" : float | None — volume ratio candle evaluasi
                  (body dihitung di sini: abs(close - open))
        zone    : dict dari detect_consolidation_zone() — harus punya:
                  "is_valid", "resistance", "support"

    Return dict:
        {
            "terpenuhi"          : bool   — True jika breakout valid
            "arah"               : str    — "BUY", "SELL", atau "NETRAL"
            "keterangan"         : str    — penjelasan lengkap
            "konfirmasi_volume"  : bool   — apakah volume_ratio >= 1.2
            "konfirmasi_body"    : bool   — apakah body >= 0.8 * atr_14
            "zone_resistance"    : float | None
            "zone_support"       : float | None
        }
    """
    # ── Guard: zona tidak valid → tidak ada yang bisa di-breakout ──────────────
    if not zone.get("is_valid", False):
        return {
            "terpenuhi"         : False,
            "arah"              : "NETRAL",
            "keterangan"        : (
                f"Breakout Trigger: zona konsolidasi tidak valid — "
                f"{zone.get('keterangan', 'zona tidak tersedia')}"
            ),
            "konfirmasi_volume" : False,
            "konfirmasi_body"   : False,
            "zone_resistance"   : zone.get("resistance"),
            "zone_support"      : zone.get("support"),
        }

    close      = float(signals["close"])
    open_price = float(signals["open"])
    atr_14     = float(signals.get("atr_14", 0.0))
    vr         = signals.get("volume_ratio")
    resistance = float(zone["resistance"])
    support    = float(zone["support"])

    # Hitung body candle (abs karena bearish = open > close)
    body = abs(close - open_price)

    # Evaluasi konfirmasi (OR — salah satu cukup)
    # Threshold 1.2 dan 0.8 adalah titik awal kalibrasi — lihat docstring.
    konfirmasi_volume = (vr is not None) and (float(vr) >= 1.2)
    konfirmasi_body   = (atr_14 > 0) and (body >= 0.8 * atr_14)
    ada_konfirmasi    = konfirmasi_volume or konfirmasi_body

    # Bangun string penjelasan konfirmasi
    konfirmasi_parts = []
    if konfirmasi_volume:
        konfirmasi_parts.append(f"volume_ratio={vr:.3f} >= 1.2")
    if konfirmasi_body:
        konfirmasi_parts.append(f"body={body:.2f} >= 0.8*ATR({atr_14:.2f})={0.8*atr_14:.2f}")
    konfirmasi_str = ", ".join(konfirmasi_parts) if konfirmasi_parts else "tidak ada konfirmasi"

    # ── Evaluasi arah breakout ─────────────────────────────────────────────────
    if close > resistance:
        # Kandidat BUY — close menembus resistance
        if ada_konfirmasi:
            return {
                "terpenuhi"         : True,
                "arah"              : "BUY",
                "keterangan"        : (
                    f"Breakout Trigger: BUY — close={close:.2f} > R={resistance:.2f} "
                    f"(zona {zone['duration']} candle), konfirmasi: {konfirmasi_str}"
                ),
                "konfirmasi_volume" : konfirmasi_volume,
                "konfirmasi_body"   : konfirmasi_body,
                "zone_resistance"   : resistance,
                "zone_support"      : support,
            }
        else:
            return {
                "terpenuhi"         : False,
                "arah"              : "NETRAL",
                "keterangan"        : (
                    f"Breakout Trigger: close={close:.2f} > R={resistance:.2f} "
                    f"(zona {zone['duration']} candle) TAPI tidak ada konfirmasi "
                    f"(volume_ratio={vr if vr is not None else 'N/A'}, "
                    f"body={body:.2f} < 0.8*ATR={0.8*atr_14:.2f}) — breakout lemah, diabaikan"
                ),
                "konfirmasi_volume" : False,
                "konfirmasi_body"   : False,
                "zone_resistance"   : resistance,
                "zone_support"      : support,
            }

    elif close < support:
        # Kandidat SELL — close menembus support
        if ada_konfirmasi:
            return {
                "terpenuhi"         : True,
                "arah"              : "SELL",
                "keterangan"        : (
                    f"Breakout Trigger: SELL — close={close:.2f} < S={support:.2f} "
                    f"(zona {zone['duration']} candle), konfirmasi: {konfirmasi_str}"
                ),
                "konfirmasi_volume" : konfirmasi_volume,
                "konfirmasi_body"   : konfirmasi_body,
                "zone_resistance"   : resistance,
                "zone_support"      : support,
            }
        else:
            return {
                "terpenuhi"         : False,
                "arah"              : "NETRAL",
                "keterangan"        : (
                    f"Breakout Trigger: close={close:.2f} < S={support:.2f} "
                    f"(zona {zone['duration']} candle) TAPI tidak ada konfirmasi "
                    f"(volume_ratio={vr if vr is not None else 'N/A'}, "
                    f"body={body:.2f} < 0.8*ATR={0.8*atr_14:.2f}) — breakout lemah, diabaikan"
                ),
                "konfirmasi_volume" : False,
                "konfirmasi_body"   : False,
                "zone_resistance"   : resistance,
                "zone_support"      : support,
            }

    else:
        # Close masih di dalam zona — bukan breakout
        return {
            "terpenuhi"         : False,
            "arah"              : "NETRAL",
            "keterangan"        : (
                f"Breakout Trigger: NETRAL — close={close:.2f} masih di dalam zona "
                f"[S={support:.2f}, R={resistance:.2f}] (zona {zone['duration']} candle)"
            ),
            "konfirmasi_volume" : konfirmasi_volume,
            "konfirmasi_body"   : konfirmasi_body,
            "zone_resistance"   : resistance,
            "zone_support"      : support,
        }


# =============================================================================
# FILTER: RSI
# =============================================================================
# Filter berbeda dari kondisi entry — ini adalah VETO.
# Filter tidak membantu entry terjadi, tapi bisa MEMBATALKAN entry
# yang sudah terpenuhi dari kondisi lain.

def _check_rsi_filter(signals: dict, arah_kandidat: str) -> dict:
    """
    Evaluasi apakah RSI memblokir entry ke arah tertentu — dengan mode kontekstual.

    RSI dipakai sebagai FILTER, bukan kondisi entry utama.
    Artinya: RSI yang bagus tidak MENDORONG entry, tapi RSI ekstrem bisa
    MEMBATALKAN entry yang sudah dikonfirmasi kondisi lain.

    LOGIKA KONTEKSTUAL (berdasarkan kekuatan trend):

    Trend KUAT (|ema_gap_pct| >= RSI_STRONG_TREND_EMA_GAP_THRESHOLD):
        RSI overbought/oversold TIDAK otomatis veto — dalam trend kuat, RSI
        bisa tetap ekstrem untuk waktu yang cukup lama (momentum continuation).
        Entry tetap diizinkan, tapi warning dicatat di field "rsi_warning"
        agar trader tetap sadar kondisi RSI saat ini.
        mode_kontekstual = "TREND_KUAT"

    Trend LEMAH (|ema_gap_pct| < threshold):
        Veto ketat seperti sebelumnya.
        - arah BUY  + RSI > 70 → BLOKIR (overbought, risiko reversal tinggi)
        - arah SELL + RSI < 30 → BLOKIR (oversold, risiko reversal tinggi)
        mode_kontekstual = "TREND_LEMAH"

    Parameter:
        signals        : dict dari get_latest_signals() — harus ada "rsi_14", "ema_gap_pct"
        arah_kandidat  : "BUY", "SELL", atau "NETRAL"

    Return dict:
        {
            "memblokir"        : bool     — True berarti entry DIBATALKAN oleh RSI
            "rsi"              : float    — nilai RSI terbaru
            "zona"             : str      — "OVERBOUGHT", "OVERSOLD", atau "NETRAL"
            "keterangan"       : str      — penjelasan utama
            "rsi_warning"      : str|None — pesan peringatan jika RSI ekstrem tapi
                                            tidak memblokir (mode TREND_KUAT)
            "mode_kontekstual" : str      — "TREND_KUAT" atau "TREND_LEMAH"
            "ema_gap_pct"      : float    — nilai gap yang dipakai untuk mode check
        }
    """
    rsi         = signals["rsi_14"]
    ema_gap_pct = signals["ema_gap_pct"]

    # Tentukan zona RSI
    if rsi > RSI_OVERBOUGHT:
        zona = "OVERBOUGHT"
    elif rsi < RSI_OVERSOLD:
        zona = "OVERSOLD"
    else:
        zona = "NETRAL"

    # Tentukan mode kontekstual berdasarkan kekuatan trend
    # (abs() karena ema_gap_pct negatif untuk downtrend)
    is_strong_trend = abs(ema_gap_pct) >= RSI_STRONG_TREND_EMA_GAP_THRESHOLD
    mode_kontekstual = "TREND_KUAT" if is_strong_trend else "TREND_LEMAH"

    # ── Evaluasi veto berdasarkan mode kontekstual ────────────────────────────

    rsi_warning = None  # default: tidak ada warning tambahan

    if arah_kandidat == "BUY" and zona == "OVERBOUGHT":
        if is_strong_trend:
            # Trend kuat — RSI ekstrem kemungkinan besar momentum continuation
            # TIDAK blokir, tapi catat sebagai warning
            memblokir = False
            keterangan = (
                f"RSI {rsi:.1f} (OVERBOUGHT) — trend KUAT (gap EMA {ema_gap_pct:+.4f}%), "
                f"RSI tidak memblokir (potensi continuation)"
            )
            rsi_warning = (
                f"⚠️ RSI {rsi:.1f} > {RSI_OVERBOUGHT} (OVERBOUGHT) — "
                f"dalam trend kuat ini dianggap continuation, bukan reversal signal. "
                f"Pantau price action dengan hati-hati."
            )
        else:
            # Trend lemah — veto normal
            memblokir = True
            keterangan = (
                f"RSI {rsi:.1f} > {RSI_OVERBOUGHT} (OVERBOUGHT) — "
                f"trend lemah (gap EMA {ema_gap_pct:+.4f}%), blokir BUY"
            )

    elif arah_kandidat == "SELL" and zona == "OVERSOLD":
        if is_strong_trend:
            # Trend kuat — oversold dalam downtrend bisa continuation
            memblokir = False
            keterangan = (
                f"RSI {rsi:.1f} (OVERSOLD) — trend KUAT (gap EMA {ema_gap_pct:+.4f}%), "
                f"RSI tidak memblokir (potensi continuation)"
            )
            rsi_warning = (
                f"⚠️ RSI {rsi:.1f} < {RSI_OVERSOLD} (OVERSOLD) — "
                f"dalam trend kuat ini dianggap continuation, bukan reversal signal. "
                f"Pantau price action dengan hati-hati."
            )
        else:
            # Trend lemah — veto normal
            memblokir = True
            keterangan = (
                f"RSI {rsi:.1f} < {RSI_OVERSOLD} (OVERSOLD) — "
                f"trend lemah (gap EMA {ema_gap_pct:+.4f}%), blokir SELL"
            )

    else:
        # RSI netral, atau arah NETRAL — tidak ada konflik
        memblokir = False
        keterangan = (
            f"RSI {rsi:.1f} — zona {zona} "
            f"[{mode_kontekstual}, gap EMA {ema_gap_pct:+.4f}%], "
            f"tidak memblokir entry {arah_kandidat}"
        )

    return {
        "memblokir"        : memblokir,
        "rsi"              : rsi,
        "zona"             : zona,
        "keterangan"       : keterangan,
        "rsi_warning"      : rsi_warning,
        "mode_kontekstual" : mode_kontekstual,
        "ema_gap_pct"      : ema_gap_pct,
    }


# =============================================================================
# KONDISI 3 / FILTER 2: VOLUME PARTICIPATION (Fase 3.2)
# =============================================================================
# Sumber: signals["volume_ratio"] dari indicators.calculate_volume_ratio().
#
# Peran: menilai KUALITAS PARTISIPASI pasar pada candle entry.
# Volume tidak korelasi dengan arah (EMA), sehingga memberikan informasi
# independen: apakah ada partisipasi pasar yang cukup untuk mendukung
# pergerakan yang signifikan?
#
# Logika: tolak entry di candle dengan volume ekstrem (terlalu tipis atau
# terlalu jenuh) — zona medium menunjukkan kondisi pasar paling kondusif.

def _check_volume_participation(signals: dict) -> dict:
    """
    Evaluasi apakah volume candle saat ini berada di zona partisipasi yang sehat.

    LOGIKA:
        volume_ratio < VOLUME_RATIO_LOW_THRESHOLD  (α=0.50):
            Volume terlalu rendah — pasar tipis, sinyal berisiko noise.
            memblokir = True  (tolak entry)

        volume_ratio > VOLUME_RATIO_HIGH_THRESHOLD (β=1.80):
            Volume terlalu tinggi — potensi exhaustion/climax.
            memblokir = True  (tolak entry)

        VOLUME_RATIO_LOW_THRESHOLD ≤ volume_ratio ≤ VOLUME_RATIO_HIGH_THRESHOLD:
            Zona medium — partisipasi normal, kondisi kondusif.
            memblokir = False

    CATATAN GRACEFUL DEGRADATION:
        Jika 'volume_ratio' tidak ada di signals (misalnya data lama yang tidak
        punya tick_volume, atau mode live tanpa kolom ini), fungsi ini NOT BLOCK
        dan mencatat keterangan bahwa data tidak tersedia. Ini mencegah crash
        bila rule engine dipanggil dari konteks tanpa data volume.

    Dalam mode CONDITION (kondisi ke-3 penuh):
        - terpenuhi = True  jika volume di zona medium
        - arah = arah kandidat (diteruskan dari konteks luar, karena volume
          tidak punya informasi arah sendiri)
        - Caller (evaluate_entry) yang mengontrol apakah ini masuk hitungan
          kondisi_terpenuhi atau veto saja.

    Return dict:
        {
            "memblokir"   : bool   — True jika volume ekstrem (veto entry)
            "terpenuhi"   : bool   — True jika volume di zona medium
            "zona"        : str    — "RENDAH", "NORMAL", "TINGGI"
            "volume_ratio": float  — nilai volume_ratio terbaru
            "keterangan"  : str    — penjelasan ringkas
            "data_tersedia": bool  — False jika signals tidak punya 'volume_ratio'
        }
    """
    # Graceful degradation jika volume_ratio tidak tersedia
    if "volume_ratio" not in signals or signals["volume_ratio"] is None:
        return {
            "memblokir"    : False,   # tidak blokir — jangan crash tanpa data
            "terpenuhi"    : False,   # tidak dihitung sebagai konfirmasi
            "arah"         : "NETRAL",
            "zona"         : "UNKNOWN",
            "volume_ratio" : None,
            "keterangan"   : "Volume_ratio tidak tersedia — filter dinonaktifkan",
            "data_tersedia": False,
        }

    vr = float(signals["volume_ratio"])

    if vr < VOLUME_RATIO_LOW_THRESHOLD:
        zona      = "RENDAH"
        terpenuhi = False
        memblokir = True
        keterangan = (
            f"Volume ratio {vr:.3f} < {VOLUME_RATIO_LOW_THRESHOLD} (RENDAH) — "
            f"pasar terlalu tipis, sinyal berisiko noise"
        )
    elif vr > VOLUME_RATIO_HIGH_THRESHOLD:
        zona      = "TINGGI"
        terpenuhi = False
        memblokir = True
        keterangan = (
            f"Volume ratio {vr:.3f} > {VOLUME_RATIO_HIGH_THRESHOLD} (TINGGI) — "
            f"potensi volume climax / exhaustion, hindari entry"
        )
    else:
        zona      = "NORMAL"
        terpenuhi = True
        memblokir = False
        keterangan = (
            f"Volume ratio {vr:.3f} [{VOLUME_RATIO_LOW_THRESHOLD}–{VOLUME_RATIO_HIGH_THRESHOLD}] "
            f"(NORMAL) — partisipasi pasar kondusif untuk entry"
        )

    return {
        "memblokir"    : memblokir,
        "terpenuhi"    : terpenuhi,
        "arah"         : "NETRAL",  # volume tidak berisi informasi arah
        "zona"         : zona,
        "volume_ratio" : round(vr, 4),
        "keterangan"   : keterangan,
        "data_tersedia": True,
    }


# =============================================================================
# FUNGSI UTAMA: EVALUATE ENTRY
# =============================================================================

def evaluate_entry(
    signals                  : dict,
    volume_mode              : str          = VOLUME_MODE_DEFAULT,
    df                       = None,        # pd.DataFrame | None — DataFrame M5 yang di-slice hingga candle saat ini
                                            # (Fase 7) Diteruskan ke calculate_setup_quality() untuk candle pattern
                                            # detection. Default None = tidak ada candle pattern scoring.
    zone                     : dict | None  = None,
                                            # dict dari detect_consolidation_zone() — dihitung dari candle
                                            # SEBELUM candle ini (idx=i-1) untuk menghindari circular lookahead.
                                            # None = breakout trigger tidak bisa dievaluasi (zona tidak tersedia).
    enable_breakout_trigger  : bool         = True,
                                            # Toggle breakout trigger (Fase 9).
                                            # True  = aktifkan breakout sebagai trigger alternatif (default).
                                            # False = nonaktifkan; dipakai untuk Tahap 0 regression check
                                            #         (sistem harus menghasilkan trade identik dengan baseline).
) -> dict:
    """
    Fungsi utama rule engine — evaluasi semua kondisi dan beri keputusan akhir.

    ARSITEKTUR FASE 9 (berbeda dari fase sebelumnya):
        Sebelumnya: bias_h1 + ema_trigger_m5 di-AND-kan (voting mayoritas, MINIMUM_CONDITIONS_MET=2)
        Sekarang  : bias_h1 WAJIB sebagai prasyarat terpisah, trigger_group berisi EMA + Breakout (OR)

        Struktur baru:
            trigger_valid = (ema_trigger_m5.arah == bias_h1.arah)
                         OR (breakout_trigger.arah == bias_h1.arah)  [jika enabled]
            ENTRY jika: bias_h1.terpenuhi AND trigger_valid AND tidak ada filter yang blokir

        Breakout yang berlawanan arah dengan bias_h1 DIABAIKAN (tidak memblokir).
        Arah keputusan selalu dari bias_h1 — trigger hanya soal timing/konfirmasi.

    ALUR KERJA:
        1. Evaluasi bias_h1 sebagai prasyarat wajib
        2. Evaluasi trigger_group (EMA + Breakout opsional)
        3. Tentukan trigger_valid: ada minimal 1 trigger searah dengan bias_h1
        4. Cek filter RSI dan Volume terhadap arah bias_h1
        5. Tentukan keputusan final: BUY / SELL / WAIT
        6. Kumpulkan context warnings dari session_filter
        7. Hitung setup quality (9.3 jika aktif)
        8. Susun output lengkap

    Parameter:
        signals                 : dict dari get_latest_signals() — minimal harus punya:
                                  "trend", "trend_h1", "ema_9", "ema_21", "ema_gap_pct",
                                  "rsi_14", "close", "time"
                                  Untuk breakout trigger juga dibutuhkan:
                                  "open", "atr_14", "volume_ratio"
        volume_mode             : str — "FILTER", "CONDITION", atau "IGNORE".
        zone                    : dict dari detect_consolidation_zone() atau None.
        enable_breakout_trigger : bool — toggle breakout trigger.

    Return:
        dict berisi:
            "keputusan"              : "BUY" / "SELL" / "WAIT"
            "arah"                   : "LONG" / "SHORT" / None
            "trigger_source"         : str | None — sumber trigger yang menyebabkan entry:
                                       "EMA_GAP"  = hanya EMA trigger yang cocok arah
                                       "BREAKOUT" = hanya breakout trigger yang cocok arah
                                       "BOTH"     = keduanya cocok arah (confluence)
                                       None       = WAIT (tidak ada trigger valid)
            "alasan_entry"           : list[str] — penjelasan ringkas mengapa entry
            "alasan_wait"            : list[str] — penjelasan mengapa WAIT/ditolak
            "kondisi_detail"         : dict — breakdown tiap kondisi:
                                       SELALU ada: "bias_h1", "ema_trigger_m5", "rsi_filter"
                                       Opsional  : "breakout_trigger" (jika enabled & zone ada)
                                                   "volume_filter" (jika volume_mode=FILTER)
            "konfirmasi_terpenuhi"   : int — jumlah trigger yang cocok arah bias_h1 (0-2)
                                       (semantik berbeda dari sebelumnya — bukan lagi hitungan
                                        kondisi_entry, tapi jumlah trigger yang searah)
            "konfirmasi_dibutuhkan"  : int — selalu 1 (minimal 1 trigger valid)
            "close"                  : float — harga close terbaru
            "waktu_evaluasi"         : str — timestamp evaluasi
            "context_warnings"       : list[str] — peringatan konteks (TIDAK mengubah keputusan)
    """
    _validate_signals(signals)

    alasan_entry = []
    alasan_wait  = []

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 1: Prasyarat Wajib — Bias H1
    # ─────────────────────────────────────────────────────────────────────────
    # bias_h1 dipisah eksplisit dari trigger_group — bukan bagian dari OR logic.
    # Entry HANYA bisa terjadi jika bias_h1.terpenuhi == True.
    # bias_h1 juga menjadi sumber arah_kandidat yang tunggal.

    c_h1 = _check_bias_h1(signals)
    arah_kandidat = c_h1["arah"]  # "BUY", "SELL", atau "NETRAL"

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 2: Evaluasi Trigger Group (OR logic)
    # ─────────────────────────────────────────────────────────────────────────
    # Trigger group menjawab "kapan momen entry yang tepat?" — bukan "ke arah mana?"
    # Minimal 1 trigger harus searah dengan bias_h1 untuk entry terjadi.
    # Trigger berlawanan arah DIABAIKAN (tidak memblokir entry dari trigger lain).

    c_m5  = _check_ema_trigger_m5(signals)
    c_vol = _check_volume_participation(signals)

    trigger_group = [("ema_trigger_m5", c_m5)]
    c_breakout = None  # default: tidak ada breakout trigger

    if enable_breakout_trigger and zone is not None:
        c_breakout = _check_breakout_trigger(signals, zone)
        trigger_group.append(("breakout_trigger", c_breakout))

    # Tentukan trigger mana yang cocok arah dengan bias_h1
    # Trigger berlawanan arah hanya diabaikan — tidak memblokir
    ema_cocok      = c_m5["terpenuhi"] and c_m5["arah"] == arah_kandidat
    breakout_cocok = (
        c_breakout is not None
        and c_breakout["terpenuhi"]
        and c_breakout["arah"] == arah_kandidat
    )

    trigger_valid = ema_cocok or breakout_cocok

    # Tentukan trigger_source untuk audit
    if ema_cocok and breakout_cocok:
        trigger_source = "BOTH"
    elif ema_cocok:
        trigger_source = "EMA_GAP"
    elif breakout_cocok:
        trigger_source = "BREAKOUT"
    else:
        trigger_source = None

    # Hitung konfirmasi_terpenuhi — jumlah trigger yang cocok arah bias_h1 (0, 1, atau 2)
    # Semantik berbeda dari sebelumnya: bukan lagi hitungan kondisi_entry,
    # tapi jumlah trigger yang secara aktif mendukung keputusan ini.
    konfirmasi_terpenuhi = (1 if ema_cocok else 0) + (1 if breakout_cocok else 0)
    konfirmasi_dibutuhkan = 1  # minimal 1 trigger valid

    # Volume mode CONDITION: volume bisa jadi kondisi ke-3
    # (semantik lama dipertahankan untuk kompatibilitas mode ini)
    if volume_mode == "CONDITION" and c_vol["terpenuhi"]:
        # Dalam mode CONDITION, volume menambah konfirmasi tapi bukan bagian trigger_group
        # (volume tidak punya arah, tidak dihitung untuk trigger_valid)
        pass  # konfirmasi_terpenuhi tidak berubah untuk trigger_valid logic

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 3: Evaluasi Filter RSI dan Volume terhadap arah kandidat
    # ─────────────────────────────────────────────────────────────────────────

    c_rsi = _check_rsi_filter(signals, arah_kandidat)

    vol_memblokir = False
    if volume_mode == "FILTER":
        vol_memblokir = c_vol["memblokir"]
    # mode "IGNORE": volume tidak dipakai sama sekali (vol_memblokir tetap False)
    # mode "CONDITION": volume sebagai tambahan skor, bukan veto di sini

    filter_memblokir = c_rsi["memblokir"] or vol_memblokir

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 4: Tentukan keputusan final
    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY jika: bias_h1.terpenuhi AND trigger_valid AND NOT filter_memblokir
    # (MINIMUM_CONDITIONS_MET tidak lagi dipakai — lihat komentar deprecation)

    if c_h1["terpenuhi"] and trigger_valid and not filter_memblokir:
        # ✅ ENTRY — semua syarat terpenuhi
        keputusan  = arah_kandidat   # "BUY" atau "SELL"
        arah_label = "LONG" if arah_kandidat == "BUY" else "SHORT"

        alasan_entry.append(c_h1["keterangan"])
        for nama, hasil in trigger_group:
            if hasil["terpenuhi"] and hasil["arah"] == arah_kandidat:
                alasan_entry.append(hasil["keterangan"])
        alasan_entry.append(c_rsi["keterangan"])
        if volume_mode == "FILTER":
            alasan_entry.append(f"[Volume OK] {c_vol['keterangan']}")

    elif c_h1["terpenuhi"] and trigger_valid and filter_memblokir:
        # ⏳ WAIT — trigger ada tapi filter memblokir
        keputusan  = "WAIT"
        arah_label = None

        alasan_wait.append(f"[Prasyarat OK] {c_h1['keterangan']}")
        for nama, hasil in trigger_group:
            if hasil["terpenuhi"] and hasil["arah"] == arah_kandidat:
                alasan_wait.append(f"[Trigger OK] {hasil['keterangan']}")
        if c_rsi["memblokir"]:
            alasan_wait.append(f"[RSI BLOKIR] {c_rsi['keterangan']}")
        if vol_memblokir:
            alasan_wait.append(f"[VOL BLOKIR] {c_vol['keterangan']}")

    else:
        # ⏳ WAIT — bias_h1 tidak terpenuhi ATAU tidak ada trigger valid
        keputusan  = "WAIT"
        arah_label = None

        if not c_h1["terpenuhi"]:
            alasan_wait.append(c_h1["keterangan"])
        else:
            alasan_entry.append(f"[Prasyarat OK] {c_h1['keterangan']}")

        for nama, hasil in trigger_group:
            if hasil["terpenuhi"] and hasil["arah"] == arah_kandidat:
                alasan_entry.append(f"[Trigger OK] {hasil['keterangan']}")
            elif hasil["terpenuhi"] and hasil["arah"] != arah_kandidat:
                alasan_wait.append(f"[Trigger berlawanan arah] {hasil['keterangan']}")
            else:
                alasan_wait.append(hasil["keterangan"])

        if c_rsi["memblokir"]:
            alasan_wait.append(c_rsi["keterangan"])
        elif arah_kandidat != "NETRAL":
            alasan_entry.append(c_rsi["keterangan"])

        if volume_mode == "FILTER":
            if vol_memblokir:
                alasan_wait.append(c_vol["keterangan"])
            elif arah_kandidat != "NETRAL":
                alasan_entry.append(c_vol["keterangan"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 5: Kumpulkan context warnings dari session_filter
    # ─────────────────────────────────────────────────────────────────────────
    # Lazy import — agar evaluate_entry() tetap bisa ditest tanpa session_filter.
    # Jika import gagal (lingkungan minimal), context_warnings cukup kosong.
    context_warnings = []
    try:
        from engine import session_filter as _sf

        sess = _sf.is_high_liquidity_session(signals["time"])
        if sess["warning"] is not None:
            context_warnings.append(sess["warning"])

        close_check = _sf.is_near_market_close(signals["time"])
        if close_check["warning"] is not None:
            context_warnings.append(close_check["warning"])

    except Exception:
        # Gagal import atau gagal proses → tidak crash, context_warnings tetap kosong
        pass

    # Tambahkan rsi_warning ke context_warnings jika ada (dari RSI filter kontekstual)
    if c_rsi.get("rsi_warning") is not None:
        context_warnings.append(c_rsi["rsi_warning"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 5b: Hitung Confidence / Setup Quality Scoring
    # ─────────────────────────────────────────────────────────────────────────
    quality_res = calculate_setup_quality(
        signals, c_h1, c_m5, c_rsi,
        df=df,
        trigger_source=trigger_source,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 6: Susun output lengkap
    # ─────────────────────────────────────────────────────────────────────────
    # kondisi_detail: key-name dipertahankan persis untuk kompatibilitas web/app.py
    # Key yang SELALU ada: "bias_h1", "ema_trigger_m5", "rsi_filter"
    # Key baru opsional : "breakout_trigger" (hanya jika enabled dan zone tidak None)
    # Key volume opsional: "volume_filter" (hanya jika volume_mode=FILTER)
    kondisi_detail = {
        "bias_h1"        : c_h1,
        "ema_trigger_m5" : c_m5,
        "rsi_filter"     : c_rsi,
    }
    if c_breakout is not None:
        kondisi_detail["breakout_trigger"] = c_breakout
    if volume_mode == "FILTER":
        kondisi_detail["volume_filter"] = c_vol

    return {
        # Keputusan akhir
        "keputusan"  : keputusan,
        "arah"       : arah_label,

        # Sumber trigger yang aktif (Fase 9) — untuk audit dan breakdown per trigger
        "trigger_source" : trigger_source,

        # Confidence / Setup Quality Scoring (auditable breakdown)
        "setup_quality"       : quality_res["setup_quality"],
        "setup_quality_score" : quality_res["setup_quality_score"],
        "setup_quality_max"   : quality_res["setup_quality_max"],
        "quality_breakdown"   : quality_res["quality_breakdown"],

        # Alasan — untuk UI dan audit
        "alasan_entry" : alasan_entry,
        "alasan_wait"  : alasan_wait,

        # Breakdown detail tiap kondisi — untuk audit mendalam
        # Key yang SELALU ada: bias_h1, ema_trigger_m5, rsi_filter
        # Key baru opsional : breakout_trigger (jika enabled & zone ada)
        "kondisi_detail" : kondisi_detail,

        # Statistik konfirmasi (semantik baru sejak Fase 9):
        # konfirmasi_terpenuhi = jumlah trigger yang cocok arah bias_h1 (0-2)
        # konfirmasi_dibutuhkan = 1 (minimal 1 trigger valid)
        "konfirmasi_terpenuhi"  : konfirmasi_terpenuhi,
        "konfirmasi_dibutuhkan" : konfirmasi_dibutuhkan,

        # Context market saat evaluasi
        "close"           : signals["close"],
        "waktu_evaluasi"  : str(signals["time"]),

        # Peringatan konteks — TIDAK mengubah keputusan, hanya informasi untuk trader
        "context_warnings" : context_warnings,
    }


# =============================================================================
# HELPER INTERNAL — Setup Quality Scoring (Auditable 0-8 Poin)
# =============================================================================
# Fase 4.3 (2026-08-03): alignment dihapus (tautologi struktural),
# swing_distance diperbaiki (data kini di-feed dari pipeline sebelum evaluate_entry).
# Skor maks setelah Fase 4.3: 6. Threshold: STRONG >=5, MODERATE >=3, WEAK <3.
#
# Fase 7 (2026-08-06): candle_pattern ditambah sebagai komponen ke-4 (0-2 poin).
# Skor maks setelah Fase 7: 8. Threshold disesuaikan proporsional:
#   STRONG lama  : >=5/6 = 83.3% -> 83.3% * 8 = 6.67 -> dibulatkan ke >=7
#   MODERATE lama: >=3/6 = 50.0% -> 50.0% * 8 = 4.0  -> >=4
#
# Fase 9 (2026-08-11): trigger_confluence ditambah sebagai komponen ke-5 (0-2 poin).
# Skor maks setelah Fase 9: 10. Threshold disesuaikan proporsional dari skema Fase 7:
#   STRONG lama  : >=7/8 = 87.5% -> 87.5% * 10 = 8.75 -> dibulatkan ke >=9
#   MODERATE lama: >=4/8 = 50.0% -> 50.0% * 10 = 5.0  -> >=5
#   WEAK         : <5

def calculate_setup_quality(
    signals              : dict,
    c_h1                 : dict,
    c_m5                 : dict,
    c_rsi                : dict,
    df                   = None,    # pd.DataFrame | None
                                    # DataFrame M5 di-slice hingga candle saat ini.
                                    # Dibutuhkan untuk candle pattern detection (Fase 7).
                                    # None = komponen candle_pattern tidak dihitung (skor 0).
    enable_candle_pattern: bool = False,
                                    # Toggle komponen ke-4 (Fase 7).
                                    # DEFAULT False: keputusan freeze Fase 7 — komponen TIDAK LOLOS
                                    # 3 siklus validasi, dikecualikan dari jalur live secara default.
                                    # True  = aktifkan manual untuk riset/validasi future out-of-sample.
                                    # False = (default live) hanya 4 komponen yang dihitung.
    trigger_source       : str | None = None,
                                    # Sumber trigger aktif dari evaluate_entry() (Fase 9).
                                    # "EMA_GAP", "BREAKOUT", "BOTH", atau None.
                                    # Dipakai untuk komponen ke-5 "Trigger Confluence" (0-2 poin).
                                    # None = tidak ada trigger valid atau zone tidak tersedia -> skor 0.
) -> dict:
    """
    Hitung setup_quality ("STRONG", "MODERATE", "WEAK") berbasis point-based scoring.

    Komponen yang SELALU aktif (4 komponen, total max=8, skema live default):
        1. EMA Gap Strength        (0-2 pts): Kekuatan trend M5 dari gap EMA
        2. RSI Zone                (0-2 pts): RSI di zona netral vs ekstrem
        3. Swing Distance          (0-2 pts): Jarak harga ke swing terdekat vs ATR
                                              (data via signals["swing_low"]/["swing_high"]
                                              yang di-feed pipeline sebelum evaluate_entry)
        5. Trigger Confluence      (0-2 pts): Apakah hanya EMA, hanya Breakout, atau keduanya
                                              yang memberikan sinyal searah bias H1.
                                              (Fase 9: menangkap kekuatan gabungan tanpa
                                               mengubah logika entry itu sendiri)

    Komponen OPSIONAL (default OFF di jalur live):
        4. Candlestick Pattern     (0-2 pts): Pattern OHLC bullish/bearish di candle terakhir.
                                              Dikecualikan dari jalur live (freeze Fase 7 —
                                              tidak lolos 3 siklus validasi korelasi).
                                              Aktifkan manual via enable_candle_pattern=True
                                              hanya untuk riset dengan data out-of-sample baru.

    CATATAN ARSITEKTUR (Fase 4.3):
        Komponen alignment H1-M5 dihapus -- terbukti tautologi secara struktural:
        evaluate_entry() mensyaratkan H1 dan M5 searah (MINIMUM_CONDITIONS_MET=2)
        sebelum trade bisa terjadi, sehingga alignment SELALU 2 untuk setiap trade.
        Zero variance, tidak bisa membedakan setup apapun. Bukan bug data, tapi
        konsekuensi matematis dari arsitektur filter entry.

    CATATAN ARSITEKTUR (Fase 7 — komponen ke-4, status FREEZE):
        candle_pattern dikecualikan dari scoring live setelah TIDAK LOLOS 3 siklus
        validasi (Fase 7 Opsi B). Kode tetap ada untuk kemungkinan diaktifkan
        manual di riset masa depan dengan data out-of-sample baru.

    CATATAN ARSITEKTUR (Fase 9 — komponen ke-5):
        trigger_confluence didesain sebagai SCORING, bukan logika entry.
        Dua trigger aktif bersamaan (BOTH) tidak mengubah keputusan entry --
        entry tetap terjadi kalau salah satu saja cocok. Skor BOTH = 2 hanya
        mencerminkan "setup ini lebih terkonfirmasi" di level scoring, bukan
        di level binary BUY/SELL/WAIT.

    CARA HITUNG max DAN THRESHOLD (dinamis, bukan hardcode):
        max_score = 2 per komponen yang benar-benar aktif.
        Threshold menggunakan persentase tetap yang disepakati:
            STRONG   >= ceil(87.5% * max_score)  — dari kesepakatan Fase 7: >=7/8
            MODERATE >= ceil(50.0% * max_score)  — dari kesepakatan awal: >=3/6
        Dengan skema default (4 komponen, max=8):
            STRONG   >= ceil(0.875 * 8) = ceil(7.0) = 7
            MODERATE >= ceil(0.500 * 8) = ceil(4.0) = 4
        Jika candle_pattern aktif (5 komponen, max=10):
            STRONG   >= ceil(0.875 * 10) = ceil(8.75) = 9
            MODERATE >= ceil(0.500 * 10) = ceil(5.0)  = 5

    Parameter:
        signals               : dict dari get_latest_signals() / pipeline backtester
        c_h1, c_m5, c_rsi     : dict hasil _check_*() (saat ini tidak dipakai langsung
                                di scoring, tapi dipertahankan signature-nya untuk
                                konsistensi dengan caller)
        df                    : DataFrame M5 di-slice s/d candle saat ini (bisa None)
        enable_candle_pattern : bool — toggle komponen ke-4 (default False = jalur live)
        trigger_source        : str | None — dari evaluate_entry(), untuk komponen ke-5
    """
    breakdown   = {}
    total_score = 0

    # 1. EMA Gap Strength (0-2)
    ema_gap_pct = abs(signals.get("ema_gap_pct", 0.0))
    if ema_gap_pct >= 0.15:
        score_gap  = 2
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Kuat >= 0.15%)"
    elif ema_gap_pct >= 0.08:
        score_gap  = 1
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Sedang 0.08-0.15%)"
    else:
        score_gap  = 0
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Tipis < 0.08%)"
    total_score += score_gap
    breakdown["ema_gap"] = {
        "score": score_gap, "max": 2,
        "label": "Kekuatan Trend M5 (EMA Gap)", "detail": detail_gap,
    }

    # 2. RSI Zone (0-2)
    rsi = signals.get("rsi_14", 50.0)
    if 40.0 <= rsi <= 60.0:
        score_rsi  = 2
        detail_rsi = f"RSI {rsi:.1f} (Zona Optimum Netral 40-60)"
    elif (30.0 <= rsi < 40.0) or (60.0 < rsi <= 70.0):
        score_rsi  = 1
        detail_rsi = f"RSI {rsi:.1f} (Zona Waspada 30-40 / 60-70)"
    else:
        score_rsi  = 0
        detail_rsi = f"RSI {rsi:.1f} (Zona Ekstrem <30 / >70)"
    total_score += score_rsi
    breakdown["rsi_zone"] = {
        "score": score_rsi, "max": 2,
        "label": "Zona RSI M5", "detail": detail_rsi,
    }

    # 3. Swing Distance (0-2)
    # signals["swing_low"] / ["swing_high"] di-feed oleh caller (backtester & app.py)
    # sebelum memanggil evaluate_entry() -- diperbaiki di Fase 4.3.
    trend_h1    = signals.get("trend_h1", "SIDEWAYS")
    trend_m5    = signals.get("trend",    "SIDEWAYS")
    close_price = signals.get("close", 0.0)
    atr         = signals.get("atr_14", 1.5)
    sw_low      = signals.get("swing_low")
    sw_high     = signals.get("swing_high")

    swing_dist = None
    if trend_h1 == "UPTREND" or trend_m5 == "UPTREND":
        if sw_low is not None:
            swing_dist = abs(close_price - sw_low)
    elif trend_h1 == "DOWNTREND" or trend_m5 == "DOWNTREND":
        if sw_high is not None:
            swing_dist = abs(sw_high - close_price)

    # Fallback: pakai swing yang tersedia jika arah tidak terdeteksi
    if swing_dist is None:
        if sw_low is not None:
            swing_dist = abs(close_price - sw_low)
        elif sw_high is not None:
            swing_dist = abs(sw_high - close_price)

    if swing_dist is not None and atr > 0:
        atr_ratio = swing_dist / atr
        if atr_ratio >= 1.5:
            score_swing  = 2
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR -- Luas)"
        elif atr_ratio >= 0.8:
            score_swing  = 1
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR -- Cukup)"
        else:
            score_swing  = 0
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR -- Sempit)"
    else:
        score_swing  = 0
        detail_swing = "Swing Tidak Ditemukan / Data Tidak Tersedia"
    total_score += score_swing
    breakdown["swing_distance"] = {
        "score": score_swing, "max": 2,
        "label": "Jarak ke Swing Structure", "detail": detail_swing,
    }

    # 4. Candlestick Pattern (0-2) — Fase 7
    # Sumber: OHLC candle terakhir di df (murni dari price action, bukan EMA/RSI/swing)
    # Arah kandidat: ikuti pola penentuan arah yang sama dengan komponen swing_distance
    # (cek trend_h1 dulu, lalu trend_m5 sebagai fallback — bukan duplikat logic,
    #  tapi referensi yang sama agar semua komponen menilai arah yang konsisten).
    if enable_candle_pattern and df is not None:
        # Tentukan arah_kandidat untuk pattern detection — pola identik dengan swing_distance
        if trend_h1 == "UPTREND" or (trend_h1 == "SIDEWAYS" and trend_m5 == "UPTREND"):
            arah_cek = "BUY"
        elif trend_h1 == "DOWNTREND" or (trend_h1 == "SIDEWAYS" and trend_m5 == "DOWNTREND"):
            arah_cek = "SELL"
        else:
            arah_cek = "NETRAL"  # keduanya SIDEWAYS — tidak ada pattern yang dicek

        try:
            # Import lazy untuk mencegah circular import dan dependensi tidak perlu
            from engine.candle_patterns import calculate_candle_pattern_score
            cp_result = calculate_candle_pattern_score(
                df            = df,
                arah_kandidat = arah_cek,
                swing_low     = sw_low,
                swing_high    = sw_high,
                atr_value     = atr,
            )
        except Exception as cp_err:
            # Graceful degradation: jika candle_patterns gagal diimport atau error,
            # jangan crash seluruh scoring — catat skor 0 dengan keterangan error
            cp_result = {
                "score"            : 0,
                "max"              : 2,
                "label"            : "Candlestick Pattern",
                "detail"           : f"Error: {cp_err}",
                "pattern_detected" : None,
            }

        total_score += cp_result["score"]
        breakdown["candle_pattern"] = cp_result
    else:
        # Komponen candle_pattern tidak aktif dalam dua skenario:
        # (A) enable_candle_pattern=False (default live / freeze Fase 7):
        #     Komponen tidak dihitung dan TIDAK ditulis ke breakdown.
        #     Breakdown hanya berisi 4 komponen aktif (ema_gap, rsi_zone,
        #     swing_distance, trigger_confluence) — mencerminkan skema live.
        # (B) enable_candle_pattern=True tapi df=None:
        #     Toggle aktif tapi data tidak tersedia — catat di breakdown
        #     sebagai skor 0 agar caller tahu data missing, bukan bug.
        if enable_candle_pattern and df is None:
            breakdown["candle_pattern"] = {
                "score"            : 0,
                "max"              : 2,
                "label"            : "Candlestick Pattern",
                "detail"           : "DataFrame tidak tersedia — candle pattern tidak dihitung",
                "pattern_detected" : None,
            }
            # total_score tidak bertambah (skor 0 implisit)
        # else: enable_candle_pattern=False → tidak ditulis ke breakdown sama sekali


    # 5. Trigger Confluence (0-2) — Fase 9
    # trigger_source dari evaluate_entry() menjadi komponen ke-5.
    # Ini menangkap kekuatan gabungan trigger tanpa mengubah logika entry.
    if trigger_source == "BOTH":
        score_tconf  = 2
        detail_tconf = f"Trigger Confluence: BOTH (EMA + Breakout searah bias H1) — konfluensi penuh"
    elif trigger_source in ("EMA_GAP", "BREAKOUT"):
        score_tconf  = 1
        detail_tconf = f"Trigger Confluence: {trigger_source} (satu trigger aktif searah bias H1)"
    else:
        score_tconf  = 0
        detail_tconf = (
            "Trigger Confluence: tidak ada trigger valid "
            + ("(zone tidak tersedia)" if trigger_source is None else f"(trigger_source={trigger_source})")
        )
    total_score += score_tconf
    breakdown["trigger_confluence"] = {
        "score"          : score_tconf,
        "max"            : 2,
        "label"          : "Trigger Confluence (EMA + Breakout)",
        "detail"         : detail_tconf,
        "trigger_source" : trigger_source,
    }

    # Penentuan max_score dan threshold secara DINAMIS dari komponen yang aktif.
    # Formula (disepakati Fase 7, dipertahankan semua fase):
    #   STRONG   >= ceil(87.5% * max_score)  — dari >=7/8 Fase 7
    #   MODERATE >= ceil(50.0% * max_score)  — dari >=3/6 awal
    # Skema default (candle_pattern OFF, 4 komponen):
    #   max=8  → STRONG>=7, MODERATE>=4
    # Skema riset (candle_pattern ON, 5 komponen):
    #   max=10 → STRONG>=9, MODERATE>=5
    STRONG_PCT   = 0.875
    MODERATE_PCT = 0.500
    component_max_total  = 2 + 2 + 2        # ema_gap + rsi_zone + swing_distance (selalu aktif)
    if enable_candle_pattern and df is not None:
        component_max_total += 2            # candle_pattern (opsional)
    component_max_total += 2               # trigger_confluence (selalu aktif)

    strong_threshold   = math.ceil(STRONG_PCT   * component_max_total)
    moderate_threshold = math.ceil(MODERATE_PCT * component_max_total)

    if total_score >= strong_threshold:
        quality_label = "STRONG"
    elif total_score >= moderate_threshold:
        quality_label = "MODERATE"
    else:
        quality_label = "WEAK"

    return {
        "setup_quality"       : quality_label,
        "setup_quality_score" : total_score,
        "setup_quality_max"   : component_max_total,
        "quality_breakdown"   : breakdown,
    }


# =============================================================================
# HELPER INTERNAL — Validasi Input
# =============================================================================

def _validate_signals(signals: dict) -> None:
    """
    Pastikan dict signals punya semua field yang dibutuhkan rule engine.

    Raise:
        TypeError  : jika signals bukan dict
        ValueError : jika ada field yang kurang
    """
    if not isinstance(signals, dict):
        raise TypeError(
            f"Input harus dictionary, bukan {type(signals).__name__}.\n"
            f"Pastikan kamu memanggil get_latest_signals() terlebih dahulu."
        )

    required = ["trend", "trend_h1", "ema_9", "ema_21", "rsi_14", "close", "time", "ema_gap_pct"]
    missing  = [k for k in required if k not in signals]

    if missing:
        raise ValueError(
            f"Field berikut tidak ada di signals: {missing}\n"
            f"Field tersedia: {list(signals.keys())}\n"
            f"Pastikan:\n"
            f"  1. signals berasal dari get_latest_signals(df_m5) yang sudah melewati run_all_indicators()\n"
            f"  2. signals['trend_h1'] sudah diisi dari hasil get_latest_signals(df_h1)['trend']"
        )
