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
    [1] _check_trend_and_ema  : Trend direction + EMA 9/21 alignment (digabung)
    [2] _check_rsi_filter     : RSI sebagai filter/veto (bukan kondisi entry)

CARA TAMBAH KONDISI BARU NANTI:
    1. Buat fungsi baru: def _check_NAMAKONDISI(signals): ...
    2. Panggil di dalam evaluate_entry()
    3. Tambahkan hasilnya ke kondisi_detail
    Tidak perlu ubah logika yang sudah ada.
"""

from datetime import datetime, timezone


# =============================================================================
# KONSTANTA KONFIGURASI
# =============================================================================
# Konfigurasi threshold diletakkan di atas, bukan dikubur di dalam fungsi,
# supaya mudah diubah tanpa perlu cari-cari ke dalam kode.

# Threshold RSI untuk zona ekstrem
RSI_OVERBOUGHT = 70.0   # RSI > nilai ini = kondisi overbought → blokir BUY
RSI_OVERSOLD   = 30.0   # RSI < nilai ini = kondisi oversold   → blokir SELL

# Jumlah kondisi ENTRY yang harus terpenuhi untuk menghasilkan BUY/SELL
# (kondisi entry = semua kondisi _check_... KECUALI filter seperti RSI)
MINIMUM_CONDITIONS_MET = 1   # saat ini hanya 1 kondisi entry (_check_trend_and_ema)


# =============================================================================
# KONDISI 1: TREND + EMA ALIGNMENT
# =============================================================================
# STATUS: DIGABUNG (sumber yang sama — detect_trend() sudah pakai EMA cross)
#
# CATATAN UNTUK PENGEMBANGAN NANTI:
#   Fungsi ini mengevaluasi DUA aspek sekaligus:
#     A) Trend label dari detect_trend()  → apakah UPTREND / DOWNTREND?
#     B) EMA cross dari nilai ema_9/ema_21 → apakah EMA cepat > EMA lambat?
#
#   Saat ini keduanya selalu sejalan karena detect_trend() menggunakan
#   EMA cross sebagai salah satu syaratnya.
#
#   SPLIT POINT: Saat kamu nanti punya detect_trend() yang pakai timeframe
#   lebih besar (misal H1), pisahkan fungsi ini menjadi:
#     - _check_trend(signals)       → evaluasi trend label dari H1
#     - _check_ema_alignment(signals) → evaluasi EMA cross dari M5
#   Dan ubah MINIMUM_CONDITIONS_MET = 2

def _check_trend_and_ema(signals: dict) -> dict:
    """
    Evaluasi apakah arah trend dan EMA alignment mendukung entry.

    LOGIKA:
        BUY  ← trend == "UPTREND"   (artinya: ema_9 > ema_21 DAN close > ema_21)
        SELL ← trend == "DOWNTREND" (artinya: ema_9 < ema_21 DAN close < ema_21)
        WAIT ← trend == "SIDEWAYS"  (tidak ada konfirmasi arah jelas)

    Return dict:
        {
            "terpenuhi"  : bool   — True jika ada arah jelas (UP atau DOWN)
            "arah"       : str    — "BUY", "SELL", atau "NETRAL"
            "keterangan" : str    — penjelasan ringkas
            "trend"      : str    — nilai mentah dari signals["trend"]
            # Detail internal (berguna saat split nanti):
            "_trend_label"  : str  — nilai dari signals["trend"]
            "_ema_cross"    : str  — "EMA9>EMA21", "EMA9<EMA21", atau "SAMA"
        }
    """
    trend   = signals["trend"]
    ema_9   = signals["ema_9"]
    ema_21  = signals["ema_21"]

    # ── Aspek A: Evaluasi trend label ──────────────────────────────────────
    # (ini yang akan jadi fungsi _check_trend() sendiri saat split)
    if trend == "UPTREND":
        arah_dari_trend = "BUY"
    elif trend == "DOWNTREND":
        arah_dari_trend = "SELL"
    else:
        arah_dari_trend = "NETRAL"

    # ── Aspek B: Evaluasi EMA cross ────────────────────────────────────────
    # (ini yang akan jadi fungsi _check_ema_alignment() sendiri saat split)
    if ema_9 > ema_21:
        ema_cross = "EMA9>EMA21"
        arah_dari_ema = "BUY"
    elif ema_9 < ema_21:
        ema_cross = "EMA9<EMA21"
        arah_dari_ema = "SELL"
    else:
        ema_cross = "SAMA"
        arah_dari_ema = "NETRAL"

    # ── Gabungkan keduanya ─────────────────────────────────────────────────
    # Saat ini: kedua aspek harus SAMA ARAH untuk kondisi terpenuhi
    # (nanti saat dipisah, masing-masing punya logika independen)
    if arah_dari_trend == arah_dari_ema and arah_dari_trend != "NETRAL":
        terpenuhi = True
        arah_final = arah_dari_trend
        keterangan = (
            f"Trend {trend} + {ema_cross} "
            f"(EMA9={ema_9:.2f} vs EMA21={ema_21:.2f}) — searah {arah_final}"
        )
    elif arah_dari_trend == "NETRAL":
        terpenuhi = False
        arah_final = "NETRAL"
        keterangan = f"Trend SIDEWAYS — tidak ada arah jelas, TUNGGU"
    else:
        # Kondisi konflik: trend dan EMA menunjuk arah berbeda
        # (Ini tidak akan terjadi saat ini, tapi bisa terjadi setelah split)
        terpenuhi = False
        arah_final = "NETRAL"
        keterangan = (
            f"Konflik: trend menunjuk {arah_dari_trend} tapi EMA menunjuk {arah_dari_ema} "
            f"— tidak cukup keyakinan untuk entry"
        )

    return {
        "terpenuhi"    : terpenuhi,
        "arah"         : arah_final,
        "keterangan"   : keterangan,
        "trend"        : trend,
        # Field internal — berguna untuk debugging dan saat split nanti
        "_trend_label" : trend,
        "_ema_cross"   : ema_cross,
    }


# =============================================================================
# FILTER: RSI
# =============================================================================
# Filter berbeda dari kondisi entry — ini adalah VETO.
# Filter tidak membantu entry terjadi, tapi bisa MEMBATALKAN entry
# yang sudah terpenuhi dari kondisi lain.

def _check_rsi_filter(signals: dict, arah_kandidat: str) -> dict:
    """
    Evaluasi apakah RSI memblokir entry ke arah tertentu.

    RSI dipakai sebagai FILTER, bukan kondisi entry utama.
    Artinya: RSI yang bagus tidak MENDORONG entry, tapi RSI ekstrem bisa
    MEMBATALKAN entry yang sudah dikonfirmasi kondisi lain.

    Logika veto:
        - arah BUY  + RSI > 70 → BLOKIR (overbought, risiko reversal tinggi)
        - arah SELL + RSI < 30 → BLOKIR (oversold, risiko reversal tinggi)
        - Semua kondisi lain   → TIDAK memblokir (entry tetap bisa jalan)

    Parameter:
        signals        : dict dari get_latest_signals()
        arah_kandidat  : "BUY" atau "SELL" (arah yang mau dicheck)

    Return dict:
        {
            "memblokir"  : bool  — True berarti entry DIBATALKAN oleh RSI
            "rsi"        : float — nilai RSI terbaru
            "zona"       : str   — "OVERBOUGHT", "OVERSOLD", atau "NETRAL"
            "keterangan" : str   — penjelasan
        }
    """
    rsi = signals["rsi_14"]

    # Tentukan zona RSI
    if rsi > RSI_OVERBOUGHT:
        zona = "OVERBOUGHT"
    elif rsi < RSI_OVERSOLD:
        zona = "OVERSOLD"
    else:
        zona = "NETRAL"

    # Tentukan apakah RSI memblokir arah yang dimaksud
    if arah_kandidat == "BUY" and zona == "OVERBOUGHT":
        memblokir = True
        keterangan = (
            f"RSI {rsi:.1f} > {RSI_OVERBOUGHT} (OVERBOUGHT) — "
            f"harga sudah terlalu tinggi, blokir BUY"
        )
    elif arah_kandidat == "SELL" and zona == "OVERSOLD":
        memblokir = True
        keterangan = (
            f"RSI {rsi:.1f} < {RSI_OVERSOLD} (OVERSOLD) — "
            f"harga sudah terlalu rendah, blokir SELL"
        )
    else:
        memblokir = False
        keterangan = (
            f"RSI {rsi:.1f} — zona {zona}, "
            f"tidak memblokir entry {arah_kandidat}"
        )

    return {
        "memblokir"  : memblokir,
        "rsi"        : rsi,
        "zona"       : zona,
        "keterangan" : keterangan,
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
        3. Cek filter RSI terhadap arah kandidat
        4. Tentukan keputusan final: BUY / SELL / WAIT
        5. Kumpulkan semua info ke dalam satu dict output

    Parameter:
        signals : dict dari get_latest_signals() — minimal harus punya:
                  "trend", "ema_9", "ema_21", "rsi_14", "close", "time"

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
    """
    _validate_signals(signals)

    alasan_entry = []
    alasan_wait  = []

    # ─────────────────────────────────────────────────────────────────────────
    # TAHAP 1: Evaluasi semua kondisi entry
    # ─────────────────────────────────────────────────────────────────────────
    # Untuk tambah kondisi baru: tambah pemanggilan fungsi di sini
    # dan masukkan hasilnya ke dalam `kondisi_entry`

    c_trend = _check_trend_and_ema(signals)

    # Kumpulkan semua hasil kondisi entry ke dalam list
    # Format: (nama_kondisi, hasil_dict)
    kondisi_entry = [
        ("trend_and_ema", c_trend),
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
    # TAHAP 5: Susun output lengkap
    # ─────────────────────────────────────────────────────────────────────────

    return {
        # Keputusan akhir
        "keputusan"  : keputusan,
        "arah"       : arah_label,

        # Alasan — untuk UI dan audit
        "alasan_entry" : alasan_entry,
        "alasan_wait"  : alasan_wait,

        # Breakdown detail tiap kondisi — untuk audit mendalam
        "kondisi_detail" : {
            "trend_and_ema" : c_trend,
            "rsi_filter"    : c_rsi,
        },

        # Statistik konfirmasi
        "konfirmasi_terpenuhi"  : kondisi_terpenuhi,
        "konfirmasi_dibutuhkan" : MINIMUM_CONDITIONS_MET,

        # Context market saat evaluasi
        "close"           : signals["close"],
        "waktu_evaluasi"  : str(signals["time"]),
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

    required = ["trend", "ema_9", "ema_21", "rsi_14", "close", "time"]
    missing  = [k for k in required if k not in signals]

    if missing:
        raise ValueError(
            f"Field berikut tidak ada di signals: {missing}\n"
            f"Field tersedia: {list(signals.keys())}\n"
            f"Pastikan signals berasal dari get_latest_signals() yang sudah "
            f"melewati run_all_indicators()."
        )
