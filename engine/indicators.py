"""
engine/indicators.py
====================
Modul kalkulasi indikator teknikal untuk sistem trading XAUUSD M5.

CARA KERJA UMUM:
    Semua fungsi di sini menerima DataFrame dari data_fetcher.get_candles()
    dan mengembalikan DataFrame yang sama tapi sudah ditambah kolom baru
    berisi nilai indikator.

    Contoh alur pemakaian:
        df = get_candles()           # ambil data dari MT5 (semua sudah closed)
        df = calculate_ema(df)       # tambah kolom ema_9 dan ema_21
        df = calculate_rsi(df)       # tambah kolom rsi_14
        df = detect_trend(df)        # tambah kolom trend

    Untuk melihat hasil terakhir (candle paling baru yang sudah closed):
        df.iloc[-1]                  # baris terakhir = candle CLOSED terakhir

    CATATAN PENTING:
        get_candles() menggunakan start_pos=1 di MT5, sehingga candle yang
        sedang terbentuk (belum closed) TIDAK ADA di DataFrame ini.
        df.iloc[-1] selalu = candle yang sudah selesai (closed), bukan
        candle yang masih berjalan. Ini mencegah masalah "repainting signal".

CATATAN LIBRARY:
    Kita TIDAK pakai pandas-ta karena tidak support Python 3.10 di PyPI.
    Sebagai gantinya, kita pakai:
    - pandas built-in: .ewm() untuk EMA, rolling() untuk perhitungan window
    - library 'ta' (Technical Analysis): untuk RSI dan indikator lain
    Kedua metode ini menghasilkan nilai yang IDENTIK dengan MT5.
"""

import pandas as pd
import numpy as np
import ta  # library Technical Analysis (pip install ta)


# =============================================================================
# BAGIAN 1: EMA — Exponential Moving Average
# =============================================================================

def calculate_ema(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """
    Menghitung EMA untuk satu atau lebih periode dan menambahkannya ke DataFrame.

    APA ITU EMA:
        EMA (Exponential Moving Average) adalah rata-rata harga yang memberi
        bobot lebih besar ke harga yang lebih baru. Berbeda dengan SMA (Simple MA)
        yang memberi bobot sama ke semua harga.

        Kenapa EMA lebih berguna untuk trading:
        - Lebih responsif terhadap pergerakan harga terbaru
        - Sinyal lebih cepat dibanding SMA
        - Yang paling umum dipakai trader: EMA 9 (cepat) dan EMA 21 (lambat)

    KENAPA NILAINYA SAMA DENGAN MT5:
        MT5 menghitung EMA dengan rumus eksponensial standar menggunakan
        multiplier = 2 / (periode + 1). Pandas .ewm(span=N, adjust=False)
        menggunakan rumus yang IDENTIK. Hasilnya akan sangat mirip dengan
        MT5, tapi mungkin berbeda sedikit di awal karena perbedaan "seed"
        (MT5 mungkin pakai SMA periode pertama sebagai nilai awal).

    Parameter:
        df      : DataFrame dari get_candles() — harus punya kolom 'close'
        periods : List periode EMA yang mau dihitung.
                  Default: [9, 21] (EMA 9 dan EMA 21)

    Return:
        DataFrame yang sama + kolom baru 'ema_9', 'ema_21', dst.
        Beberapa baris pertama akan NaN karena butuh data minimal N candle
        untuk menghitung EMA periode N.

    Contoh output kolom baru:
        ema_9  : nilai EMA periode 9 di setiap candle
        ema_21 : nilai EMA periode 21 di setiap candle
    """
    if periods is None:
        periods = [9, 21]

    # Validasi DataFrame
    _check_dataframe(df, required_columns=["close"])

    # Hitung EMA untuk setiap periode yang diminta
    for period in periods:
        column_name = f"ema_{period}"

        # .ewm(span=N) = setup window eksponensial dengan periode N
        # adjust=False  = gunakan rumus rekursif standar (sama dengan MT5)
        # .mean()        = hitung rata-rata tertimbang eksponensial
        df[column_name] = df["close"].ewm(span=period, adjust=False).mean()

    return df


# =============================================================================
# BAGIAN 2: RSI — Relative Strength Index
# =============================================================================

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Menghitung RSI dan menambahkan kolom 'rsi_14' ke DataFrame.

    APA ITU RSI:
        RSI (Relative Strength Index) adalah indikator momentum yang mengukur
        kecepatan dan besarnya pergerakan harga, dengan skala 0–100.

        Cara membacanya:
        - RSI > 70  → overbought (harga sudah terlalu tinggi, potensi turun)
        - RSI < 30  → oversold  (harga sudah terlalu rendah, potensi naik)
        - RSI ~ 50  → netral    (tidak ada momentum kuat ke arah tertentu)

        Dalam framework trading kita, RSI dipakai sebagai FILTER — bukan
        sinyal utama. Artinya, kita tidak entry HANYA karena RSI > 70 atau < 30,
        tapi kita pakai RSI untuk mengkonfirmasi setup yang sudah ada.

    KENAPA NILAINYA SAMA DENGAN MT5:
        MT5 menghitung RSI dengan metode Wilder's Smoothing (SMMA/RMA).
        Library 'ta' menggunakan ta.momentum.RSIIndicator yang mengimplementasikan
        metode yang sama. Nilai akan sangat mirip setelah N candle pertama.

    Parameter:
        df     : DataFrame dari get_candles() — harus punya kolom 'close'
        period : Periode RSI. Default 14 (standar industri, sama dengan MT5 default)

    Return:
        DataFrame yang sama + kolom baru f'rsi_{period}' (misal 'rsi_14')

    Contoh nilai dan artinya:
        rsi_14 = 72.5  → overbought, momentum bullish kuat
        rsi_14 = 28.3  → oversold, momentum bearish kuat
        rsi_14 = 51.0  → netral
    """
    _check_dataframe(df, required_columns=["close"])

    column_name = f"rsi_{period}"

    # ta.momentum.RSIIndicator: implementasi RSI dengan Wilder's Smoothing
    # fillna=False: biarkan NaN untuk candle yang belum cukup datanya
    rsi_indicator = ta.momentum.RSIIndicator(
        close=df["close"],
        window=period,
        fillna=False
    )

    df[column_name] = rsi_indicator.rsi()

    return df


# =============================================================================
# BAGIAN 3: DETEKSI TREND
# =============================================================================

def detect_trend(df: pd.DataFrame, min_ema_gap_pct: float = 0.05) -> pd.DataFrame:
    """
    Mendeteksi kondisi trend berdasarkan posisi EMA 9, EMA 21, dan harga close.

    LOGIKA YANG DIPAKAI (framework trading manual kamu):

        UPTREND (trend naik):
            - EMA 9 > EMA 21 (garis cepat di atas garis lambat = momentum naik)
            - Close > EMA 21 (harga di atas garis lambat = buyer in control)
            - abs(ema_gap_pct) >= min_ema_gap_pct (jarak EMA cukup lebar)

        DOWNTREND (trend turun):
            - EMA 9 < EMA 21 (garis cepat di bawah = momentum turun)
            - Close < EMA 21 (harga di bawah = seller in control)
            - abs(ema_gap_pct) >= min_ema_gap_pct (jarak EMA cukup lebar)

        SIDEWAYS / KONSOLIDASI:
            - Semua kondisi lain — termasuk:
              a) EMA terlalu dekat (|gap| < min_ema_gap_pct): zona choppy/konsolidasi
              b) Harga di antara EMA: arah tidak jelas
            - Ini kondisi "tunggu" — tidak ada sinyal yang bisa dipercaya

    THRESHOLD KEKUATAN TREND (min_ema_gap_pct):
        Kalau EMA 9 dan EMA 21 hampir menyentuh satu sama lain, label
        "UPTREND" atau "DOWNTREND" menyesatkan — secara visual chart terlihat
        choppy, bukan trending. Threshold ini memaksa label ke SIDEWAYS jika
        jarak EMA terlalu tipis untuk dianggap sebagai tren yang valid.

        Contoh untuk XAUUSD di ~$3300:
            min_ema_gap_pct = 0.05%  →  jarak minimum ≈ $1.65
            gap = -0.0132%  →  |gap| = 0.0132% < 0.05%  →  paksa SIDEWAYS
            gap = -0.18%    →  |gap| = 0.18%   ≥ 0.05%  →  DOWNTREND valid

        Nilai 0.05% adalah titik awal kalibrasi — bisa disesuaikan setelah
        observasi data nyata. Semakin tinggi threshold, semakin ketat filter.

    TAMBAHAN: Kekuatan trend
        Selain label trend, kita juga hitung 'ema_gap_pct' — jarak antara
        EMA 9 dan EMA 21 dalam persen. Makin lebar jarak, makin kuat trendnya.

    Parameter:
        df              : DataFrame yang sudah melewati calculate_ema() — butuh
                          kolom 'close', 'ema_9', 'ema_21'
        min_ema_gap_pct : Threshold minimum |ema_gap_pct| agar tren dianggap
                          valid (bukan SIDEWAYS). Default: 0.05%.
                          Buat lebih besar untuk filter yang lebih ketat.

    Return:
        DataFrame yang sama + kolom baru:
            'trend'       : string "UPTREND", "DOWNTREND", atau "SIDEWAYS"
            'ema_gap_pct' : jarak EMA 9 vs EMA 21 dalam persen (float)
                            Positif = EMA9 di atas EMA21 (bullish)
                            Negatif = EMA9 di bawah EMA21 (bearish)
    """
    _check_dataframe(df, required_columns=["close", "ema_9", "ema_21"])

    # ─────────────────────────────────────────────────────────────────────────
    # Hitung jarak relatif antara EMA 9 dan EMA 21 (dalam persen)
    # Ini menunjukkan SEBERAPA KUAT trendnya
    # Rumus: (EMA9 - EMA21) / EMA21 * 100
    # Positif = uptrend, negatif = downtrend
    # ─────────────────────────────────────────────────────────────────────────
    df["ema_gap_pct"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"] * 100

    # ─────────────────────────────────────────────────────────────────────────
    # Cek apakah jarak EMA cukup lebar untuk dianggap tren valid
    # Ini adalah prasyarat tambahan di atas syarat EMA cross + price position
    # ─────────────────────────────────────────────────────────────────────────
    gap_cukup = df["ema_gap_pct"].abs() >= min_ema_gap_pct

    # ─────────────────────────────────────────────────────────────────────────
    # Tentukan label trend menggunakan np.select (lebih efisien dari apply/loop)
    # np.select: mirip if-elif-else tapi untuk seluruh kolom sekaligus
    # ─────────────────────────────────────────────────────────────────────────

    # Definisikan kondisi (urutan penting! kondisi pertama yang match = dipakai)
    conditions = [
        # Kondisi UPTREND: EMA cepat di atas EMA lambat, harga di atas EMA lambat,
        # DAN jarak EMA cukup lebar (tidak sekadar menyentuh satu sama lain)
        (df["ema_9"] > df["ema_21"]) & (df["close"] > df["ema_21"]) & gap_cukup,

        # Kondisi DOWNTREND: EMA cepat di bawah EMA lambat, harga di bawah EMA lambat,
        # DAN jarak EMA cukup lebar
        (df["ema_9"] < df["ema_21"]) & (df["close"] < df["ema_21"]) & gap_cukup,
    ]

    # Nilai yang dikembalikan untuk setiap kondisi (urutan sama dengan conditions)
    choices = ["UPTREND", "DOWNTREND"]

    # default = SIDEWAYS jika tidak ada kondisi yang match —
    # termasuk saat EMA sudah cross tapi gapnya terlalu tipis (choppy zone)
    df["trend"] = np.select(conditions, choices, default="SIDEWAYS")

    return df


# =============================================================================
# BAGIAN 4: ATR — Average True Range (Volatilitas)
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Menghitung ATR dan menambahkan kolom 'atr_14' ke DataFrame.

    APA ITU ATR:
        ATR (Average True Range) mengukur rata-rata VOLATILITAS candle dalam
        N periode. Semakin besar ATR, semakin "liar" pergerakan harga.

        Berguna untuk:
        - Menentukan jarak SL yang "wajar" berdasarkan kondisi market saat ini
        - Market volatile (ATR besar) → SL lebih jauh agar tidak kena noise
        - Market tenang (ATR kecil) → SL bisa lebih dekat

    KOMPONEN:
        True Range (TR) = nilai terbesar dari:
            1. high − low              (rentang candle itu sendiri)
            2. |high − close sebelumnya| (gap naik dari close sebelumnya)
            3. |low  − close sebelumnya| (gap turun dari close sebelumnya)
        ATR = rata-rata TR selama N periode (dengan Wilder's Smoothing)

    KENAPA NILAINYA SAMA DENGAN MT5:
        MT5 ATR menggunakan Wilder's Smoothing: alpha = 1 / period.
        Di pandas, ini setara dengan .ewm(alpha=1/period, adjust=False).

    Parameter:
        df     : DataFrame dari get_candles() — butuh kolom high, low, close
        period : Periode ATR. Default 14 (standar industri)

    Return:
        DataFrame yang sama + kolom baru f'atr_{period}' (misal 'atr_14')

    Contoh nilai untuk XAUUSD M5:
        atr_14 = 5.20 → rata-rata pergerakan candle ±5.20 dollar dalam 14 candle
    """
    _check_dataframe(df, required_columns=["high", "low", "close"])

    column_name = f"atr_{period}"

    # ── Hitung True Range ───────────────────────────────────────────────────
    high        = df["high"]
    low         = df["low"]
    prev_close  = df["close"].shift(1)  # close candle sebelumnya

    # True Range = max dari 3 komponen di atas
    tr1 = high - low                   # rentang candle sendiri
    tr2 = (high - prev_close).abs()    # gap naik
    tr3 = (low  - prev_close).abs()    # gap turun

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ── Smoothing: Wilder's Smoothing (identik dengan MT5) ─────────────────
    # alpha = 1/period → ini adalah Wilder's Smoothing Method
    # adjust=False → gunakan rumus rekursif (sama seperti MT5)
    df[column_name] = true_range.ewm(alpha=1.0 / period, adjust=False).mean()

    return df


# =============================================================================
# BAGIAN 5: FUNGSI UTAMA — Jalankan Semua Indikator Sekaligus
# =============================================================================

def run_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shortcut untuk menjalankan semua kalkulasi indikator dalam satu panggilan.

    Fungsi ini memanggil calculate_ema → calculate_rsi → detect_trend → calculate_atr
    secara berurutan. Urutan penting karena detect_trend butuh hasil EMA.

    ASUMSI INPUT:
        df yang diterima sudah berisi candle CLOSED semua (jaminan dari get_candles()).
        Tidak perlu buang baris terakhir di sini karena data sudah bersih.

    Parameter:
        df : DataFrame mentah dari get_candles() — semua baris sudah closed

    Return:
        DataFrame lengkap dengan semua kolom indikator:
        - ema_9, ema_21  (dari calculate_ema)
        - rsi_14         (dari calculate_rsi)
        - trend          (dari detect_trend): "UPTREND" / "DOWNTREND" / "SIDEWAYS"
        - ema_gap_pct    (dari detect_trend): kekuatan trend dalam persen
        - atr_14         (dari calculate_atr): volatilitas rata-rata per candle

    Contoh penggunaan:
        from engine.data_fetcher import initialize_mt5, get_candles, shutdown_mt5
        from engine.indicators import run_all_indicators

        initialize_mt5()
        df = get_candles()           # hanya candle closed
        df = run_all_indicators(df)
        print(df.iloc[-1])   # lihat nilai terkini dari candle closed terakhir
        shutdown_mt5()
    """
    df = calculate_ema(df, periods=[9, 21])
    df = calculate_rsi(df, period=14)
    df = detect_trend(df)
    df = calculate_atr(df, period=14)
    return df


# =============================================================================
# BAGIAN 5: FUNGSI BANTU — Ringkasan Nilai Terkini
# =============================================================================

def get_latest_signals(df: pd.DataFrame) -> dict:
    """
    Mengekstrak nilai indikator dari candle CLOSED terakhir sebagai dictionary.

    "Candle CLOSED terakhir" = candle paling kanan di chart yang sudah selesai
    terbentuk (high/low/close-nya sudah final, tidak akan berubah lagi).

    KENAPA BUKAN CANDLE YANG SEDANG BERJALAN:
        Data dari get_candles() sudah dijamin hanya berisi candle closed
        (start_pos=1 di MT5). Jadi df.iloc[-1] secara otomatis = candle
        closed terakhir — bukan candle yang sedang terbentuk.

    Fungsi ini berguna untuk:
    - Menampilkan ringkasan kondisi market saat ini
    - Mengirim data ke rule_engine
    - Logging dan debugging

    Parameter:
        df : DataFrame yang sudah melalui run_all_indicators()

    Return:
        Dictionary berisi nilai-nilai dari candle closed terakhir, contoh:
        {
            "time"        : Timestamp("2026-07-24 08:15:00+00:00"),  # waktu OPEN candle
            "close"       : 4029.26,    # harga penutupan (FINAL)
            "ema_9"       : 4031.45,
            "ema_21"      : 4038.72,
            "rsi_14"      : 42.3,
            "trend"       : "DOWNTREND",
            "ema_gap_pct" : -0.18
        }
    """
    required = ["close", "ema_9", "ema_21", "rsi_14", "trend", "ema_gap_pct"]
    _check_dataframe(df, required_columns=required)

    # iloc[-1] = ambil baris TERAKHIR = candle CLOSED terakhir
    # (bukan candle yang sedang berjalan, karena get_candles() sudah skip itu)
    last = df.iloc[-1]

    res = {
        "time"        : df.index[-1],           # waktu candle terbaru
        "close"       : round(float(last["close"]),       2),
        "ema_9"       : round(float(last["ema_9"]),       2),
        "ema_21"      : round(float(last["ema_21"]),      2),
        "rsi_14"      : round(float(last["rsi_14"]),      2),
        "trend"       : str(last["trend"]),
        "ema_gap_pct" : round(float(last["ema_gap_pct"]), 4),
    }

    if "atr_14" in df.columns and not pd.isna(last["atr_14"]):
        res["atr_14"] = round(float(last["atr_14"]), 2)

    try:
        from engine.risk_manager import find_nearest_swing
        sw_low  = find_nearest_swing(df, "BUY")
        sw_high = find_nearest_swing(df, "SELL")
        if sw_low is not None:
            res["swing_low"] = round(float(sw_low), 2)
        if sw_high is not None:
            res["swing_high"] = round(float(sw_high), 2)
    except Exception:
        pass

    return res



# =============================================================================
# BAGIAN 6: HELPER INTERNAL — Validasi Input
# =============================================================================

def _check_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validasi DataFrame sebelum diproses.

    Ini fungsi INTERNAL (diawali underscore = tidak perlu dipanggil dari luar).
    Dipanggil di awal setiap fungsi kalkulasi untuk memberikan pesan error
    yang jelas jika input salah.

    Raise:
        ValueError : jika df kosong atau kolom yang dibutuhkan tidak ada
        TypeError  : jika df bukan pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Input harus pandas DataFrame, bukan {type(df).__name__}.\n"
            f"Pastikan kamu memanggil get_candles() terlebih dahulu."
        )

    if df.empty:
        raise ValueError(
            "DataFrame kosong — tidak ada data untuk dihitung.\n"
            "Pastikan get_candles() berhasil mengembalikan data."
        )

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di DataFrame: {missing}\n"
            f"Kolom tersedia: {list(df.columns)}\n"
            f"Kemungkinan kamu belum memanggil fungsi sebelumnya "
            f"(misalnya detect_trend() butuh calculate_ema() dulu)."
        )
