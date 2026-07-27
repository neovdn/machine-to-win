"""
engine/session_filter.py
========================
Modul deteksi sesi trading dan potensi risiko kontekstual (gap weekend, likuiditas).

PERAN MODUL INI:
    Memberikan KONTEKS situasional kepada trader — BUKAN mengubah keputusan
    BUY/SELL/WAIT. Setiap warning di sini bersifat informatif semata.

    Filosofi desain: keputusan entry tetap murni dari rule harga (trend, EMA, RSI).
    Session filter hanya menambahkan lapisan transparansi — "kamu boleh entry, tapi
    ketahui bahwa kondisi likuiditas saat ini tipis / mendekati weekend gap".

FUNGSI YANG TERSEDIA:
    is_high_liquidity_session(timestamp) → dict
        Cek apakah timestamp berada di jam overlap London/NY (likuiditas tinggi).
        Beri warning jika di luar jam tersebut.

    is_near_market_close(timestamp, minutes_before=60) → dict
        Deteksi apakah timestamp mendekati penutupan pasar Jumat.
        Warning eksplisit tentang risiko gap weekend.

CATATAN TIMEZONE:
    Semua timestamp yang masuk ke fungsi ini diproses dalam UTC.
    Timestamp dari MT5 (via data_fetcher) sudah dalam UTC (meski dengan
    label timezone yang kadang menyesatkan — lihat README Known Considerations #2).
    Kita pakai .utcoffset() atau konversi eksplisit ke UTC jika perlu.

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.
"""

from datetime import datetime, timezone, timedelta


# =============================================================================
# KONSTANTA KONFIGURASI
# =============================================================================
# Letakkan di atas agar mudah diubah tanpa perlu cari-cari ke dalam kode.

# Jam overlap London + New York (UTC) — jam likuiditas tertinggi untuk XAUUSD
# London buka: ~07:00 UTC | NY buka: ~12:00 UTC | London tutup: ~16:00 UTC
# Overlap aktif: 12:00–16:00 UTC (likuiditas tertinggi)
# NY sesi penuh: 12:00–20:00 UTC (masih sangat likuid)
# Kita pakai 12:00–20:00 UTC sebagai definisi "high liquidity window"
LONDON_NY_OVERLAP_START_UTC = 12   # jam mulai (inklusif), dalam UTC
LONDON_NY_OVERLAP_END_UTC   = 20   # jam selesai (eksklusif), dalam UTC

# Definisi batas setiap sesi (jam UTC, inklusif mulai)
# Dipakai untuk label sesi yang informatif di output
SESSION_BOUNDARIES = [
    # (start_hour_utc, end_hour_utc_exclusive, label)
    (22, 24, "ASIA"),         # Asia session: 22:00–00:00 UTC (Minggu malam / Senin pagi)
    ( 0,  7, "ASIA"),         # Asia session: 00:00–07:00 UTC
    ( 7, 12, "LONDON"),       # London-only: 07:00–12:00 UTC
    (12, 20, "LONDON_NY"),    # London + NY overlap: 12:00–20:00 UTC (high liquidity)
    (20, 22, "NY_ONLY"),      # NY sesi akhir: 20:00–22:00 UTC
]

# Market Forex/Gold tutup hari Jumat malam
# Standar umum broker: sekitar 21:00–23:00 UTC
# Kita pakai 22:00 UTC sebagai estimasi market close (bisa disesuaikan per broker)
MARKET_CLOSE_FRIDAY_UTC = 22   # jam market close Jumat (estimasi)

# Toleransi: berapa jam sebelum market close dianggap "mendekati" (bukan menit)
# default = 1 jam sebelum close = candle di atas jam 21:00 UTC pada hari Jumat
NEAR_CLOSE_HOURS_BEFORE = 1


# =============================================================================
# FUNGSI 1: DETEKSI SESI LIKUIDITAS TINGGI
# =============================================================================

def is_high_liquidity_session(timestamp) -> dict:
    """
    Cek apakah timestamp berada di jam trading dengan likuiditas tinggi.

    Definisi "high liquidity" untuk XAUUSD:
        Jam overlap London + New York = 12:00–20:00 UTC
        Di luar jam ini, spread cenderung lebih lebar dan pergerakan harga
        bisa lebih "spikey" karena volume order yang lebih tipis.

    PENTING: Ini BUKAN hard block — hanya menghasilkan warning string.
    Trader tetap bisa entry di sesi apapun; ini hanya konteks risiko.

    Parameter:
        timestamp : Timestamp candle terakhir. Bisa berupa:
                    - pandas Timestamp (dengan atau tanpa timezone)
                    - datetime object
                    - string yang bisa di-parse

    Return:
        dict berisi:
            "is_high_liquidity" : bool   — True jika di dalam jam 12:00–20:00 UTC
            "session_label"     : str    — label sesi saat ini ("LONDON_NY", "ASIA", dst)
            "hour_utc"          : int    — jam UTC saat ini (dari timestamp)
            "warning"           : str|None — pesan warning jika di luar high liquidity,
                                             None jika kondisi normal
    """
    # Konversi timestamp ke datetime dengan timezone UTC
    dt_utc = _to_utc_datetime(timestamp)
    hour_utc = dt_utc.hour

    # Tentukan label sesi berdasarkan jam UTC
    session_label = _get_session_label(hour_utc)

    # Cek apakah masuk window high liquidity
    is_high_liq = LONDON_NY_OVERLAP_START_UTC <= hour_utc < LONDON_NY_OVERLAP_END_UTC

    # Buat pesan warning jika di luar jam likuiditas tinggi
    if is_high_liq:
        warning = None
    else:
        warning = (
            f"Sesi {session_label} ({hour_utc:02d}:xx UTC) — di luar jam overlap "
            f"London/NY ({LONDON_NY_OVERLAP_START_UTC:02d}:00–{LONDON_NY_OVERLAP_END_UTC:02d}:00 UTC). "
            f"Spread bisa lebih lebar, likuiditas lebih rendah dari biasanya."
        )

    return {
        "is_high_liquidity" : is_high_liq,
        "session_label"     : session_label,
        "hour_utc"          : hour_utc,
        "warning"           : warning,
    }


# =============================================================================
# FUNGSI 2: DETEKSI MENDEKATI MARKET CLOSE (JUMAT)
# =============================================================================

def is_near_market_close(timestamp, hours_before: int = NEAR_CLOSE_HOURS_BEFORE) -> dict:
    """
    Deteksi apakah timestamp mendekati penutupan pasar Jumat.

    Konteks risiko yang dimaksud:
        Saat pasar menutup Jumat malam, posisi yang masih terbuka akan menghadapi
        "gap weekend" — harga pembukaan Senin bisa sangat berbeda dari penutupan Jumat
        karena berita yang muncul saat pasar tutup (geopolitik, data ekonomi weekend).

        Untuk XAUUSD, gap weekend bisa mencapai puluhan dollar — SL bisa terlewati
        (slippage) atau bahkan terbuka dengan harga yang jauh lebih buruk dari target.

    Definisi "mendekati close":
        Hari Jumat, dalam window `hours_before` jam sebelum MARKET_CLOSE_FRIDAY_UTC.
        Contoh default: Jumat antara jam 21:00–22:00 UTC (1 jam sebelum close 22:00).

    PENTING: Ini BUKAN hard block — hanya menghasilkan warning eksplisit.

    Parameter:
        timestamp    : Timestamp candle terakhir (sama format seperti fungsi di atas)
        hours_before : Berapa jam sebelum market close dianggap "mendekati".
                       Default: NEAR_CLOSE_HOURS_BEFORE (1 jam)

    Return:
        dict berisi:
            "is_near_close"     : bool   — True jika Jumat dan mendekati market close
            "minutes_to_close"  : int|None — estimasi menit ke market close,
                                             None jika bukan hari Jumat
            "day_name"          : str    — nama hari ("Friday", "Monday", dst) dalam English
            "warning"           : str|None — pesan warning jika mendekati close,
                                             None jika kondisi normal
    """
    dt_utc = _to_utc_datetime(timestamp)
    hour_utc     = dt_utc.hour
    minute_utc   = dt_utc.minute
    weekday      = dt_utc.weekday()   # 0=Senin, 4=Jumat, 5=Sabtu, 6=Minggu
    day_name     = dt_utc.strftime("%A")  # "Friday", "Monday", dll

    # Hanya relevan untuk hari Jumat
    if weekday != 4:  # 4 = Jumat
        return {
            "is_near_close"    : False,
            "minutes_to_close" : None,
            "day_name"         : day_name,
            "warning"          : None,
        }

    # Hitung window awal "mendekati close"
    # Contoh: close=22:00, hours_before=1 → window mulai 21:00
    window_start_hour = MARKET_CLOSE_FRIDAY_UTC - hours_before

    # Sudah lewat close? (jam sudah >= close → market sudah tutup atau mau close)
    # Kondisi: jam UTC antara window_start dan close
    is_near = window_start_hour <= hour_utc < MARKET_CLOSE_FRIDAY_UTC

    if not is_near:
        return {
            "is_near_close"    : False,
            "minutes_to_close" : None,
            "day_name"         : day_name,
            "warning"          : None,
        }

    # Hitung estimasi menit tersisa ke market close
    # Close = MARKET_CLOSE_FRIDAY_UTC:00 UTC
    # Sisa = (close_hour - current_hour) jam - current_minute menit
    minutes_to_close = (
        (MARKET_CLOSE_FRIDAY_UTC - hour_utc) * 60
    ) - minute_utc

    warning = (
        f"⚠️ JUMAT MENDEKATI MARKET CLOSE: sekitar {minutes_to_close} menit lagi "
        f"pasar menutup (~{MARKET_CLOSE_FRIDAY_UTC:02d}:00 UTC). "
        f"Risiko gap weekend tinggi — harga pembukaan Senin bisa jauh dari close Jumat. "
        f"Pertimbangkan untuk tidak membuka posisi baru."
    )

    return {
        "is_near_close"    : True,
        "minutes_to_close" : max(0, minutes_to_close),
        "day_name"         : day_name,
        "warning"          : warning,
    }


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _to_utc_datetime(timestamp) -> datetime:
    """
    Konversi berbagai format timestamp ke datetime object UTC yang konsisten.

    Mendukung:
        - pandas Timestamp (tz-aware maupun tz-naive)
        - datetime object (tz-aware maupun tz-naive)
        - string ISO 8601

    Catatan: timestamp dari MT5 via data_fetcher sudah UTC (lihat README #2).
    Jika timestamp tz-naive, kita asumsikan UTC (sesuai konvensi codebase).
    """
    # Kalau sudah datetime atau pandas Timestamp
    if hasattr(timestamp, "tzinfo"):
        # Jika ada timezone info, konversi ke UTC
        if timestamp.tzinfo is not None:
            # Konversi ke UTC
            if hasattr(timestamp, "to_pydatetime"):
                # pandas Timestamp → python datetime
                dt = timestamp.to_pydatetime()
            else:
                dt = timestamp
            # Konversi ke UTC
            dt_utc = dt.astimezone(timezone.utc)
        else:
            # tz-naive → asumsikan UTC (konvensi MT5 timestamps di codebase ini)
            if hasattr(timestamp, "to_pydatetime"):
                dt = timestamp.to_pydatetime()
            else:
                dt = timestamp
            dt_utc = dt.replace(tzinfo=timezone.utc)
        return dt_utc

    # Kalau string → parse dulu
    if isinstance(timestamp, str):
        # Coba parse berbagai format umum
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(timestamp, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        raise ValueError(
            f"Tidak bisa parse timestamp: {timestamp!r}. "
            f"Gunakan format ISO 8601 atau pandas Timestamp."
        )

    raise TypeError(
        f"Format timestamp tidak dikenal: {type(timestamp).__name__}. "
        f"Gunakan pandas Timestamp atau datetime object."
    )


def _get_session_label(hour_utc: int) -> str:
    """
    Dapatkan label sesi berdasarkan jam UTC.

    Return: "LONDON_NY", "LONDON", "ASIA", "NY_ONLY", atau "UNKNOWN"
    """
    for start, end, label in SESSION_BOUNDARIES:
        if start <= hour_utc < end:
            return label
    return "UNKNOWN"
