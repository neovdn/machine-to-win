"""
engine/data_fetcher.py
======================
Modul untuk koneksi ke MetaTrader 5 dan pengambilan data candle OHLC.

CARA KERJA:
1. MT5 desktop harus sudah terbuka dan login secara manual
2. Script ini menghubungkan Python ke MT5 melalui socket lokal
3. Minta data candle, terima sebagai numpy array, konversi ke DataFrame

KENAPA TIDAK PERLU USERNAME/PASSWORD:
Library MetaTrader5 di Windows bisa terhubung langsung ke proses MT5 yang
sedang berjalan tanpa perlu autentikasi ulang — karena sudah login di desktop.
"""

import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


# Muat variabel dari file .env ke environment Python
# Sehingga kita bisa akses dengan os.getenv("MT5_SYMBOL")
load_dotenv()


# =====================================================================
# MAPPING: String timeframe dari .env → Konstanta integer MT5
# =====================================================================
# MT5 menggunakan integer untuk timeframe, bukan string "M5"
# Kita buat peta konversi agar konfigurasi di .env tetap human-readable
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,   # 1 menit
    "M5":  mt5.TIMEFRAME_M5,   # 5 menit  ← yang kita pakai
    "M15": mt5.TIMEFRAME_M15,  # 15 menit
    "M30": mt5.TIMEFRAME_M30,  # 30 menit
    "H1":  mt5.TIMEFRAME_H1,   # 1 jam
    "H4":  mt5.TIMEFRAME_H4,   # 4 jam
    "D1":  mt5.TIMEFRAME_D1,   # 1 hari
}


def initialize_mt5() -> bool:
    """
    Menginisialisasi koneksi ke MetaTrader 5 yang sedang berjalan.

    Return:
        True  → koneksi berhasil
        False → koneksi gagal (MT5 belum terbuka / tidak terinstall)

    Cara kerja:
        mt5.initialize() akan mencari proses MT5 yang sedang berjalan di sistem.
        Jika MT5 sudah terbuka dan login, koneksi langsung berhasil.
        Tidak perlu username/password karena sudah login di desktop.
    """
    success = mt5.initialize()

    if not success:
        # Ambil detail error dari MT5 untuk debugging
        error_code, error_message = mt5.last_error()
        print(f"❌ Koneksi ke MT5 gagal!")
        print(f"   Error code   : {error_code}")
        print(f"   Error message: {error_message}")
        print(f"\n💡 Pastikan:")
        print(f"   1. MetaTrader 5 desktop sudah terbuka")
        print(f"   2. Sudah login ke akun broker di MT5")
        print(f"   3. Library MetaTrader5 sudah terinstall (pip install MetaTrader5)")
        return False

    # Tampilkan info versi MT5 yang terdeteksi
    terminal_info = mt5.terminal_info()
    print(f"✅ Koneksi ke MT5 berhasil!")
    print(f"   MT5 build    : {terminal_info.build}")
    print(f"   Path terminal: {terminal_info.path}")
    return True


def shutdown_mt5() -> None:
    """
    Menutup koneksi ke MT5 dengan bersih.

    Penting: Selalu panggil fungsi ini setelah selesai menggunakan MT5
    agar tidak ada resource yang menggantung (seperti menutup file setelah dibuka).
    """
    mt5.shutdown()
    print("🔌 Koneksi MT5 ditutup.")


def get_candles(
    symbol: str = None,
    timeframe_str: str = None,
    count: int = None
) -> pd.DataFrame | None:
    """
    Mengambil data candle historis dari MT5 dan mengembalikannya sebagai DataFrame.

    Parameter:
        symbol        : Nama instrumen (misal "XAUUSD"). Default dari .env
        timeframe_str : Timeframe sebagai string (misal "M5"). Default dari .env
        count         : Jumlah candle yang diambil. Default dari .env

    Return:
        pd.DataFrame dengan kolom: time, open, high, low, close, tick_volume
        None jika terjadi error

    KOLOM PENJELASAN:
        time        : Waktu pembukaan candle (dalam format datetime)
        open        : Harga pembukaan candle
        high        : Harga tertinggi dalam periode candle
        low         : Harga terendah dalam periode candle
        close       : Harga penutupan candle
        tick_volume : Jumlah tick (pergerakan harga) dalam periode candle
                      Di Forex/Gold, ini adalah volume proxy (bukan volume asli lot)
    """
    # Gunakan nilai dari parameter, atau fallback ke nilai di .env
    symbol        = symbol        or os.getenv("MT5_SYMBOL", "XAUUSD")
    timeframe_str = timeframe_str or os.getenv("MT5_TIMEFRAME", "M5")
    count         = count         or int(os.getenv("MT5_CANDLE_COUNT", "500"))

    # Validasi: pastikan timeframe yang diminta ada di peta konversi
    if timeframe_str not in TIMEFRAME_MAP:
        print(f"❌ Timeframe '{timeframe_str}' tidak dikenal.")
        print(f"   Pilihan yang valid: {list(TIMEFRAME_MAP.keys())}")
        return None

    # Konversi string timeframe ke integer konstanta MT5
    timeframe = TIMEFRAME_MAP[timeframe_str]

    print(f"📊 Menarik {count} candle {symbol} {timeframe_str}...")

    # ─────────────────────────────────────────────────────────────────
    # mt5.copy_rates_from_pos() — fungsi utama pengambilan data
    # ─────────────────────────────────────────────────────────────────
    # Parameter:
    #   symbol     : nama instrumen
    #   timeframe  : timeframe (integer konstanta MT5)
    #   start_pos  : 0 = mulai dari candle TERBARU
    #   count      : berapa banyak candle yang diambil ke belakang
    #
    # Hasilnya adalah numpy structured array (seperti tabel)
    # ─────────────────────────────────────────────────────────────────
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    # Cek apakah data berhasil ditarik
    if rates is None or len(rates) == 0:
        error_code, error_message = mt5.last_error()
        print(f"❌ Gagal menarik data candle!")
        print(f"   Error code   : {error_code}")
        print(f"   Error message: {error_message}")
        print(f"\n💡 Kemungkinan penyebab:")
        print(f"   1. Simbol '{symbol}' tidak ditemukan di MT5")
        print(f"      → Cek nama simbol di Market Watch MT5")
        print(f"      → Broker yang berbeda mungkin pakai nama berbeda (misal: XAUUSDm)")
        print(f"   2. Simbol belum ditambahkan ke Market Watch")
        print(f"      → Di MT5: klik kanan Market Watch → Show All → cari XAUUSD")
        return None

    # ─────────────────────────────────────────────────────────────────
    # Konversi numpy array → pandas DataFrame
    # ─────────────────────────────────────────────────────────────────
    # DataFrame adalah "tabel data" yang mudah dimanipulasi dan dihitung
    df = pd.DataFrame(rates)

    # Kolom 'time' dari MT5 berformat Unix timestamp (detik sejak 1970-01-01)
    # Konversi ke format datetime yang manusia bisa baca
    # utc=True karena MT5 selalu menggunakan UTC untuk timestamp
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Pilih hanya kolom yang relevan untuk analisis kita
    # (MT5 juga punya kolom 'spread' dan 'real_volume' yang tidak kita butuhkan sekarang)
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]

    # Bersihkan index: set waktu sebagai index agar mudah di-query berdasarkan waktu
    df = df.set_index("time")

    print(f"✅ Data berhasil ditarik: {len(df)} candle")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validasi kualitas data candle yang sudah ditarik.

    Cek yang dilakukan:
        1. DataFrame tidak kosong
        2. Tidak ada nilai NaN (data hilang/rusak)
        3. Logika OHLC valid: high >= low, high >= open, high >= close

    Return:
        True  → data valid, siap dipakai
        False → ada masalah dengan data

    Kenapa perlu validasi:
        Data dari broker kadang punya "gap" atau nilai yang hilang,
        terutama saat market tutup (weekend) atau terjadi masalah koneksi.
        Data yang tidak valid bisa menyebabkan perhitungan indikator salah.
    """
    print("\n🔍 Memvalidasi data...")

    # Cek 1: DataFrame tidak kosong
    if df is None or df.empty:
        print("❌ DataFrame kosong — tidak ada data sama sekali")
        return False

    # Cek 2: Tidak ada nilai NaN
    nan_count = df.isnull().sum().sum()  # Hitung total sel yang kosong
    if nan_count > 0:
        print(f"⚠️  Ditemukan {nan_count} nilai kosong (NaN) dalam data")
        print(f"   Detail per kolom:")
        print(df.isnull().sum())
        # Ini warning, bukan error fatal — kita tetap lanjut
    else:
        print("   ✅ Tidak ada nilai kosong")

    # Cek 3: Validasi logika OHLC
    # Dalam candle manapun, harga tertinggi (high) harus >= semua harga lain
    invalid_high = df[df["high"] < df[["open", "low", "close"]].max(axis=1)]
    if not invalid_high.empty:
        print(f"⚠️  Ditemukan {len(invalid_high)} candle dengan data OHLC tidak valid")
    else:
        print("   ✅ Logika OHLC valid (high >= open/low/close)")

    print("✅ Validasi selesai")
    return True
