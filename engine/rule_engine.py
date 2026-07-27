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

from datetime import datetime, timezone

# session_filter diimport secara lazy di dalam evaluate_entry() untuk menghindari
# circular import dan agar module ini tetap bisa ditest tanpa session_filter.


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
MINIMUM_CONDITIONS_MET = 2


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
# FUNGSI UTAMA: EVALUATE ENTRY
# =============================================================================

def evaluate_entry(signals: dict) -> dict:
    """
    Fungsi utama rule engine — evaluasi semua kondisi dan beri keputusan akhir.

    ALUR KERJA:
        1. Jalankan semua kondisi entry → kumpulkan hasilnya
        2. Hitung berapa kondisi yang terpenuhi, dan ke arah apa
        3. Cek filter RSI kontekstual terhadap arah kandidat
        4. Tentukan keputusan final: BUY / SELL / WAIT
        5. Kumpulkan context warnings dari session_filter (tidak mempengaruhi keputusan)
        6. Kumpulkan semua info ke dalam satu dict output

    Parameter:
        signals : dict dari get_latest_signals() — minimal harus punya:
                  "trend", "trend_h1", "ema_9", "ema_21", "ema_gap_pct",
                  "rsi_14", "close", "time"

    Return:
        dict berisi:
            "keputusan"              : "BUY" / "SELL" / "WAIT"
            "arah"                   : "LONG" / "SHORT" / None
            "alasan_entry"           : list[str] — penjelasan ringkas mengapa entry
            "alasan_wait"            : list[str] — penjelasan mengapa WAIT/ditolak
            "kondisi_detail"         : dict      — breakdown tiap kondisi
            "konfirmasi_terpenuhi"   : int       — berapa kondisi entry terpenuhi
            "konfirmasi_dibutuhkan"  : int       — minimum yang dibutuhkan
            "close"                  : float     — harga close terbaru
            "waktu_evaluasi"         : str       — timestamp evaluasi
            "context_warnings"       : list[str] — peringatan konteks dari session filter
                                                   (TIDAK mengubah keputusan BUY/SELL/WAIT)
    """
    _validate_signals(signals)

    alasan_entry = []
    alasan_wait  = []

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 1: Evaluasi semua kondisi entry
    # ─────────────────────────────────────────────────────────────────────────
    # Untuk tambah kondisi baru: tambah pemanggilan fungsi di sini
    # dan masukkan hasilnya ke dalam `kondisi_entry`

    c_h1  = _check_bias_h1(signals)
    c_m5  = _check_ema_trigger_m5(signals)

    # Kumpulkan semua hasil kondisi entry ke dalam list
    # Format: (nama_kondisi, hasil_dict)
    #
    # [1] bias_h1       — sumber independen: trend dari timeframe H1
    # [2] ema_trigger_m5 — sumber independen: EMA cross pada timeframe M5
    # Keduanya harus searah (kedua-duanya BUY atau kedua-duanya SELL).
    kondisi_entry = [
        ("bias_h1",        c_h1),
        ("ema_trigger_m5", c_m5),
        # ("sr_level",    _check_sr_level(signals)),    # ← tambah di sini nanti
        # ("candle_pattern", _check_candle(signals)),   # ← atau ini
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 2: Hitung berapa kondisi terpenuhi dan tentukan arah kandidat
    # ─────────────────────────────────────────────────────────────────────────

    # Hitung berapa kondisi yang terpenuhi
    kondisi_terpenuhi = sum(1 for _, hasil in kondisi_entry if hasil["terpenuhi"])

    # Tentukan arah mayoritas dari kondisi yang terpenuhi
    # (jika semua kondisi konsisten, ini mudah — kalau nanti ada konflik, perlu voting)
    arah_votes = [hasil["arah"] for _, hasil in kondisi_entry if hasil["terpenuhi"]]

    if len(set(arah_votes)) == 1 and arah_votes:
        # Semua kondisi yang terpenuhi menunjuk arah yang sama
        arah_kandidat = arah_votes[0]
    elif not arah_votes:
        arah_kandidat = "NETRAL"
    else:
        # Kondisi konflik (tidak akan terjadi saat ini, relevan nanti setelah split)
        arah_kandidat = "NETRAL"

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 3: Evaluasi filter RSI terhadap arah kandidat
    # ─────────────────────────────────────────────────────────────────────────

    c_rsi = _check_rsi_filter(signals, arah_kandidat)

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 4: Tentukan keputusan final
    # ─────────────────────────────────────────────────────────────────────────

    cukup_kondisi = kondisi_terpenuhi >= MINIMUM_CONDITIONS_MET

    if cukup_kondisi and arah_kandidat != "NETRAL" and not c_rsi["memblokir"]:
        # ✅ ENTRY — semua syarat terpenuhi
        keputusan = arah_kandidat                      # "BUY" atau "SELL"
        arah_label = "LONG" if arah_kandidat == "BUY" else "SHORT"

        # Kumpulkan alasan entry dari semua kondisi yang terpenuhi
        for _, hasil in kondisi_entry:
            if hasil["terpenuhi"]:
                alasan_entry.append(hasil["keterangan"])
        alasan_entry.append(c_rsi["keterangan"])

    elif cukup_kondisi and arah_kandidat != "NETRAL" and c_rsi["memblokir"]:
        # ⏳ WAIT — kondisi entry terpenuhi tapi RSI memblokir
        keputusan  = "WAIT"
        arah_label = None

        for _, hasil in kondisi_entry:
            if hasil["terpenuhi"]:
                alasan_wait.append(f"[Kondisi OK] {hasil['keterangan']}")
        alasan_wait.append(f"[RSI BLOKIR] {c_rsi['keterangan']}")

    else:
        # ⏳ WAIT — kondisi entry tidak cukup terpenuhi
        keputusan  = "WAIT"
        arah_label = None

        for _, hasil in kondisi_entry:
            if not hasil["terpenuhi"]:
                alasan_wait.append(hasil["keterangan"])
            else:
                alasan_entry.append(f"[Terpenuhi] {hasil['keterangan']}")
        alasan_wait.append(c_rsi["keterangan"])

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
    # Ini memastikan warning RSI muncul di UI meski RSI tidak memblokir entry
    if c_rsi.get("rsi_warning") is not None:
        context_warnings.append(c_rsi["rsi_warning"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 5b: Hitung Confidence / Setup Quality Scoring
    # ─────────────────────────────────────────────────────────────────────────
    quality_res = calculate_setup_quality(signals, c_h1, c_m5, c_rsi)

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 6: Susun output lengkap
    # ─────────────────────────────────────────────────────────────────────────

    return {
        # Keputusan akhir
        "keputusan"  : keputusan,
        "arah"       : arah_label,

        # Confidence / Setup Quality Scoring (auditable breakdown)
        "setup_quality"       : quality_res["setup_quality"],
        "setup_quality_score" : quality_res["setup_quality_score"],
        "setup_quality_max"   : quality_res["setup_quality_max"],
        "quality_breakdown"   : quality_res["quality_breakdown"],

        # Alasan — untuk UI dan audit
        "alasan_entry" : alasan_entry,
        "alasan_wait"  : alasan_wait,

        # Breakdown detail tiap kondisi — untuk audit mendalam
        "kondisi_detail" : {
            "bias_h1"        : c_h1,
            "ema_trigger_m5" : c_m5,
            "rsi_filter"     : c_rsi,
        },

        # Statistik konfirmasi
        "konfirmasi_terpenuhi"  : kondisi_terpenuhi,
        "konfirmasi_dibutuhkan" : MINIMUM_CONDITIONS_MET,

        # Context market saat evaluasi
        "close"           : signals["close"],
        "waktu_evaluasi"  : str(signals["time"]),

        # Peringatan konteks — TIDAK mengubah keputusan, hanya informasi untuk trader
        # Bisa berisi: warning sesi low-liquidity, warning gap weekend, warning RSI kontekstual
        "context_warnings" : context_warnings,
    }


# =============================================================================
# HELPER INTERNAL — Setup Quality Scoring (Auditable 0-8 Poin)
# =============================================================================

def calculate_setup_quality(signals: dict, c_h1: dict, c_m5: dict, c_rsi: dict) -> dict:
    """
    Hitung setup_quality ("STRONG", "MODERATE", "WEAK") berbasis point-based scoring (0-8 Poin).

    4 Komponen Scoring:
        1. EMA Gap Strength (0-2 pts): Kekuatan trend M5 dari gap EMA
        2. Alignment H1 & M5 (0-2 pts): Keselarasan bias makro H1 dan trigger M5
        3. RSI Zone (0-2 pts): RSI netral optimum vs ekstrem
        4. Swing Distance (0-2 pts): Jarak harga ke struktur swing terdekat vs ATR
    """
    breakdown = {}
    total_score = 0

    # 1. EMA Gap Strength (0-2)
    ema_gap_pct = abs(signals.get("ema_gap_pct", 0.0))
    if ema_gap_pct >= 0.15:
        score_gap = 2
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Kuat >= 0.15%)"
    elif ema_gap_pct >= 0.08:
        score_gap = 1
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Sedang 0.08-0.15%)"
    else:
        score_gap = 0
        detail_gap = f"Gap EMA {ema_gap_pct:+.4f}% (Trend Tipis < 0.08%)"
    total_score += score_gap
    breakdown["ema_gap"] = {
        "score": score_gap, "max": 2, "label": "Kekuatan Trend M5 (EMA Gap)", "detail": detail_gap
    }

    # 2. Timeframe Alignment (0-2)
    trend_h1 = signals.get("trend_h1", "SIDEWAYS")
    trend_m5 = signals.get("trend", "SIDEWAYS")
    if trend_h1 == trend_m5 and trend_h1 in ("UPTREND", "DOWNTREND"):
        score_align = 2
        detail_align = f"Searah — H1 {trend_h1} & M5 {trend_m5}"
    elif (trend_h1 in ("UPTREND", "DOWNTREND") and trend_m5 == "SIDEWAYS") or \
         (trend_m5 in ("UPTREND", "DOWNTREND") and trend_h1 == "SIDEWAYS"):
        score_align = 1
        detail_align = f"Parsial — H1 {trend_h1} vs M5 {trend_m5}"
    else:
        score_align = 0
        detail_align = f"Konflik / Netral — H1 {trend_h1} vs M5 {trend_m5}"
    total_score += score_align
    breakdown["alignment"] = {
        "score": score_align, "max": 2, "label": "Keselarasan Timeframe H1 & M5", "detail": detail_align
    }

    # 3. RSI Zone (0-2)
    rsi = signals.get("rsi_14", 50.0)
    if 40.0 <= rsi <= 60.0:
        score_rsi = 2
        detail_rsi = f"RSI {rsi:.1f} (Zona Optimum Netral 40-60)"
    elif (30.0 <= rsi < 40.0) or (60.0 < rsi <= 70.0):
        score_rsi = 1
        detail_rsi = f"RSI {rsi:.1f} (Zona Waspada 30-40 / 60-70)"
    else:
        score_rsi = 0
        detail_rsi = f"RSI {rsi:.1f} (Zona Ekstrem <30 / >70)"
    total_score += score_rsi
    breakdown["rsi_zone"] = {
        "score": score_rsi, "max": 2, "label": "Zona RSI M5", "detail": detail_rsi
    }

    # 4. Swing Distance (0-2)
    close_price = signals.get("close", 0.0)
    atr = signals.get("atr_14", 1.5)
    sw_low = signals.get("swing_low")
    sw_high = signals.get("swing_high")

    swing_dist = None
    if trend_h1 == "UPTREND" or trend_m5 == "UPTREND":
        if sw_low is not None:
            swing_dist = abs(close_price - sw_low)
    elif trend_h1 == "DOWNTREND" or trend_m5 == "DOWNTREND":
        if sw_high is not None:
            swing_dist = abs(sw_high - close_price)

    if swing_dist is None:
        if sw_low is not None:
            swing_dist = abs(close_price - sw_low)
        elif sw_high is not None:
            swing_dist = abs(sw_high - close_price)

    if swing_dist is not None and atr > 0:
        atr_ratio = swing_dist / atr
        if atr_ratio >= 1.5:
            score_swing = 2
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR — Luas)"
        elif atr_ratio >= 0.8:
            score_swing = 1
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR — Cukup)"
        else:
            score_swing = 0
            detail_swing = f"Jarak Swing ${swing_dist:.2f} ({atr_ratio:.1f}x ATR — Sempit)"
    else:
        score_swing = 0
        detail_swing = "Swing Tidak Ditemukan / Sempit (Fallback ATR)"
    total_score += score_swing
    breakdown["swing_distance"] = {
        "score": score_swing, "max": 2, "label": "Jarak ke Swing Structure", "detail": detail_swing
    }

    # Penentuan Label Quality
    if total_score >= 6:
        quality_label = "STRONG"
    elif total_score >= 4:
        quality_label = "MODERATE"
    else:
        quality_label = "WEAK"

    return {
        "setup_quality"       : quality_label,
        "setup_quality_score" : total_score,
        "setup_quality_max"   : 8,
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
