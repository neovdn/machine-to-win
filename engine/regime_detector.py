"""
engine/regime_detector.py
=========================
Modul deteksi market regime untuk XAUUSD M15 — Fase 13.

TUJUAN:
    Mengklasifikasikan kondisi market M15 ke dalam salah satu dari empat
    kategori regime sebelum strategi trading dipilih:

        - TRENDING          : harga bergerak searah dengan struktur swing
                              terurut (HH/HL atau LL/LH) DAN EMA gap signifikan
                              DAN mayoritas candle di sisi EMA yang benar.
        - RANGING           : harga bergerak bolak-balik di antara dua boundary
                              yang terdefinisi jelas dengan minimal 2 sentuhan
                              tiap sisi.
        - BREAKOUT_TRANSITION: candle saat ini baru saja menembus zona konsolidasi
                              yang valid (terbentuk SEBELUM candle ini) dengan
                              konfirmasi body atau volume.
        - CHOP              : kondisi default — tidak ada bukti cukup untuk
                              salah satu dari tiga kategori di atas.

KARAKTER MODUL:
    - BUKAN sinyal entry dan BUKAN komponen scoring.
    - TIDAK diintegrasikan ke evaluate_entry() atau pipeline live/backtest.
      Integrasi ke strategy router adalah pekerjaan Fase 14.
    - STATELESS murni: tidak ada state/cache antar pemanggilan. Setiap
      pemanggilan detect_market_regime(df_m15, idx) recompute PENUH dari
      df_m15.iloc[:idx+1]. Hasil identik untuk (df_m15, idx) yang sama
      tidak peduli urutan atau riwayat pemanggilan sebelumnya.
    - Berdiri INDEPENDEN terhadap rule_engine.py, backtester.py, dan
      market_context.py. Modul ini hanya consumer read-only.

URUTAN EVALUASI (WATERFALL):
    1. Cek BREAKOUT_TRANSITION terlebih dahulu.
       Alasan: breakout adalah kondisi paling spesifik — harus ada zona
       sebelumnya yang valid, dan candle saat ini baru menemubus zona itu.
       Kalau tidak dicek duluan, candle breakout bisa salah kelabel sebagai
       RANGING (zona masih terlihat di lookback) atau TRENDING (EMA baru
       mulai bergerak).
    2. Cek RANGING — zona flat yang masih aktif.
    3. Cek TRENDING — struktur HH/HL atau LL/LH yang terkonfirmasi.
    4. Default ke CHOP jika semua gagal.

CAUSALITY (ZERO LOOK-AHEAD):
    Semua fungsi hanya membaca df_m15.iloc[:idx+1].
    Candle setelah idx TIDAK PERNAH disentuh dalam bentuk apapun.
    detect_consolidation_zone() dipanggil dengan idx atau idx-1,
    tidak pernah dengan index > idx.
    Ini dibuktikan via test mutasi ekstrem di tests/test_regime_detector.py.

CATATAN PARAMETER (BELUM DIKALIBRASI):
    Semua konstanta di bagian KONSTANTA KONFIGURASI di bawah adalah
    STARTING POINT struktural yang masuk akal secara konseptual —
    BUKAN hasil kalibrasi backtest. Pola ini sama seperti:
        - zone_detector.py  : lookback=20, max_range_atr_ratio=2.5, min_duration=10
        - supply_demand.py  : DEFAULT_IMPULSIVE_RATIO=1.5, ORIGIN_BODY_MAX_RATIO=0.5
        - market_context.py : H1_STRENGTH_STRONG_THRESHOLD=0.15
    Kalibrasi via backtest dan walk-forward adalah kerja terpisah SETELAH
    validasi empiris (_diag_regime_detector.py) menunjukkan distribusi regime
    masuk akal. JANGAN ubah nilai ini tanpa approval eksplisit setelah
    validasi empiris.

DEPENDENSI:
    - engine/indicators.py   : detect_trend() (dipanggil dengan min_ema_gap_pct
                               berbeda dari default M5)
    - engine/zone_detector.py: detect_consolidation_zone() (timeframe-agnostic,
                               dipanggil dengan parameter berbeda untuk M15)
    Input df_m15 sudah harus melewati run_all_indicators() — kolom wajib ada:
    open, high, low, close, ema_9, ema_21, ema_gap_pct, trend, atr_14, volume_ratio.
"""

import pandas as pd
import numpy as np

from engine.indicators import detect_trend
from engine.zone_detector import detect_consolidation_zone


# =============================================================================
# KONSTANTA KONFIGURASI (BELUM DIKALIBRASI — lihat docstring modul)
# =============================================================================

# --- TRENDING ---
REGIME_TREND_LOOKBACK               = 20
# Candle M15 ke belakang untuk scan swing sequence (~5 jam market aktif).
# BELUM dikalibrasi.

REGIME_TREND_MIN_EMA_GAP_PCT        = 0.10
# Threshold EMA gap (%) untuk M15 — sengaja 2× lebih ketat dari default
# M5 (0.05%) karena M15 lebih sedikit noise; kalau M15 menunjukkan
# UPTREND/DOWNTREND dengan gap kecil, sinyal seringkali masih choppy.
# BELUM dikalibrasi.

REGIME_TREND_SWING_WING             = 2
# Konfirmasi swing: candle di idx dianggap swing high/low jika high/low-nya
# merupakan ekstrem dari window [idx-wing, idx+wing]. Wing=2 = ±30 menit
# di M15. Lebih kecil dari SWING_WING=3 di risk_manager.py karena timeframe
# lebih panjang (satu candle M15 = 3 candle M5). BELUM dikalibrasi.

REGIME_TREND_MIN_SWING_PAIRS        = 2
# Minimum pasangan swing terurut (HH+HL pasang untuk bullish, LL+LH untuk
# bearish) yang harus ditemukan sebelum dinyatakan TRENDING. Nilai 2 berarti
# harus ada setidaknya 2 higher-low DAN 2 higher-high untuk bullish.
# BELUM dikalibrasi.

REGIME_TREND_DIRECTIONAL_MIN_RATIO  = 0.65
# Proporsi minimum candle dalam lookback yang close-nya di sisi EMA 21
# yang benar (di atas untuk bullish, di bawah untuk bearish). 65% berarti
# setidaknya 13 dari 20 candle harus searah. BELUM dikalibrasi.

# --- RANGING ---
REGIME_RANGE_LOOKBACK               = 20
# Candle M15 ke belakang untuk detect_consolidation_zone(). BELUM dikalibrasi.

REGIME_RANGE_MAX_ATR_RATIO          = 3.0
# Batas atas range/ATR untuk dianggap konsolidasi di M15. Sedikit lebih
# longgar dari default zone_detector.py (2.5) karena range M15 secara
# struktural lebih lebar (3 candle M5 per candle M15). BELUM dikalibrasi.

REGIME_RANGE_MIN_DURATION           = 8
# Minimum durasi konsolidasi dalam candle M15. Lebih pendek dari default
# zone_detector.py (10) karena 8 candle M15 = ~2 jam konsolidasi yang
# sudah cukup berarti. BELUM dikalibrasi.

REGIME_RANGE_MIN_TOUCHES_PER_SIDE   = 2
# Minimum sentuhan ke masing-masing boundary (resistance DAN support)
# di dalam window konsolidasi. Ini membuktikan "bolak-balik" — bukan
# hanya range statis sempit. BELUM dikalibrasi.

REGIME_RANGE_TOUCH_TOLERANCE_ATR    = 0.3
# Toleransi sentuhan boundary dalam satuan ATR. Sentuhan ke resistance =
# candle dengan high >= resistance - (tolerance * ATR). Nilai 0.3 artinya
# candle hanya perlu masuk dalam 30% ATR dari boundary. BELUM dikalibrasi.

# --- BREAKOUT_TRANSITION ---
REGIME_BREAKOUT_LOOKBACK_ZONE       = 20
# Candle M15 ke belakang untuk detect_consolidation_zone di idx-1 (zona
# sebelum candle breakout). BELUM dikalibrasi.

REGIME_BREAKOUT_MAX_RANGE_ATR_RATIO = 2.5
# Threshold range/ATR untuk zona prior yang valid. Sama dengan default
# zone_detector.py. BELUM dikalibrasi.

REGIME_BREAKOUT_MIN_DURATION        = 10
# Minimum durasi zona prior. Sama dengan default zone_detector.py.
# BELUM dikalibrasi.

REGIME_BREAKOUT_MIN_BODY_ATR_RATIO  = 0.8
# Body candle breakout minimal 0.8× ATR untuk konfirmasi. Lebih rendah dari
# supply_demand.py (1.5×ATR) karena candle breakout M15 tidak harus
# sekuat candle impulsif S&D — cukup punya body yang substansial.
# BELUM dikalibrasi.

REGIME_BREAKOUT_MIN_VOLUME_RATIO    = 1.2
# Volume candle breakout minimal 1.2× rata-rata (volume_ratio >= 1.2).
# Ini konfirmasi OPSIONAL (OR dengan body) — jika kolom volume_ratio
# tidak tersedia, syarat ini dilewati secara graceful. BELUM dikalibrasi.


# =============================================================================
# PARAMETER SET CYCLE 2 (untuk audit trail — dipakai via override, bukan
# overwrite konstanta default di atas yang tetap sebagai "Cycle 1 starting point")
# =============================================================================
# Catatan: gunakan dict ini sebagai **kwargs saat memanggil detect_market_regime().
# Contoh: detect_market_regime(df, idx, **REGIME_PARAMS_V2)

REGIME_PARAMS_V2 = {
    # --- TRENDING: beri ruang lebih untuk swing pairs ---
    "trend_lookback"       : 30,   # 20 -> 30: lebih banyak candle untuk scan swing
    "trend_min_swing_pairs": 1,    # 2  -> 1: satu pasang sudah cukup dengan EMA+konsistensi
    # (EMA gap pct dan directional min ratio tidak diubah di Cycle 2)
    # --- RANGING: turunkan min sentuhan agar range baru bisa terdeteksi ---
    "range_min_touches"    : 1,    # 2 -> 1: minimal 1 sentuhan tiap sisi
}
# Justifikasi dan validasi: lihat output _diag_regime_detector_root_cause.py
# dan laporan rekalibrasi Cycle 2 di walkthrough.md.


def _empty_regime(keterangan: str) -> dict:
    """
    Return dict standar untuk kasus data tidak cukup atau error.
    Regime default ke CHOP — kondisi paling aman (tidak ada sinyal).
    """
    return {
        "regime"     : "CHOP",
        "arah"       : None,
        "zone"       : None,
        "detail"     : {
            "breakout_check": {
                "terpenuhi": False,
                "arah": None,
                "zone": None,
                "konfirmasi_body": False,
                "konfirmasi_volume": False,
                "keterangan": "data tidak cukup",
            },
            "ranging_check": {
                "terpenuhi": False,
                "zone": None,
                "touches_resistance": 0,
                "touches_support": 0,
                "keterangan": "data tidak cukup",
            },
            "trending_check": {
                "terpenuhi": False,
                "arah": None,
                "ema_ok": False,
                "struktur_ok": False,
                "konsistensi_ok": False,
                "konsistensi_ratio": 0.0,
                "swing_count_confirm": 0,
                "keterangan": "data tidak cukup",
            },
        },
        "keterangan" : keterangan,
    }


# =============================================================================
# HELPER DETEKSI SWING SEQUENCE
# =============================================================================

def _detect_swing_sequence(
    df_m15  : pd.DataFrame,
    idx     : int,
    lookback: int,
    wing    : int,
) -> list:
    """
    Deteksi semua titik swing high dan swing low terkonfirmasi dalam window
    [idx - lookback + 1, idx - wing] dari df_m15.

    DEFINISI SWING (meniru pola find_nearest_swing() di risk_manager.py):
        Swing LOW di posisi k: low[k] == min(low[k-wing : k+wing+1])
                               yaitu low[k] lebih rendah dari atau sama dengan
                               semua candle dalam window ±wing di sekitarnya.
        Swing HIGH di posisi k: high[k] == max(high[k-wing : k+wing+1])

    CATATAN KAUSALITAS:
        Window konfirmasi membutuhkan candle di kanan (k+wing). Oleh karena itu
        scan hanya dilakukan sampai idx - wing (bukan sampai idx) agar candle
        konfirmasi kanan masih tersedia di df_m15.iloc[:idx+1].
        Candle di idx sendiri TIDAK masuk sebagai swing karena tidak ada
        konfirmasi kanan yang cukup.

    Parameter:
        df_m15  : DataFrame M15 yang sudah punya kolom high, low.
        idx     : Index candle evaluasi (sudah ternormalisasi, >= 0).
        lookback: Jumlah candle maksimum ke belakang untuk scan.
        wing    : Konfirmasi candle kiri-kanan (sama dengan pola risk_manager.py).

    Return:
        list of dict, terurut dari posisi terlama ke terbaru:
            [{"posisi": int, "harga": float, "tipe": "HIGH" | "LOW"}, ...]
        List kosong jika tidak ada swing yang terdeteksi.
    """
    # Batas scan: dari idx-lookback+1 sampai idx-wing
    # Kiri: pastikan ada cukup candle untuk wing kiri
    scan_start = max(idx - lookback + 1, wing)
    # Kanan: pastikan ada cukup candle untuk wing kanan (candle k+wing <= idx)
    scan_end = idx - wing

    if scan_start > scan_end:
        return []

    highs = df_m15["high"].values
    lows  = df_m15["low"].values

    swings = []

    for k in range(scan_start, scan_end + 1):
        # Ambil window ±wing di sekitar k
        window_high = highs[k - wing : k + wing + 1]
        window_low  = lows[k - wing  : k + wing + 1]

        # Swing HIGH: high[k] adalah tertinggi di window
        if highs[k] == np.max(window_high):
            swings.append({
                "posisi": k,
                "harga" : float(highs[k]),
                "tipe"  : "HIGH",
            })

        # Swing LOW: low[k] adalah terendah di window
        # (bisa saja satu candle menjadi swing HIGH sekaligus swing LOW
        #  jika doji sempurna — kasus ekstrem, tapi kita catat keduanya)
        if lows[k] == np.min(window_low):
            swings.append({
                "posisi": k,
                "harga" : float(lows[k]),
                "tipe"  : "LOW",
            })

    # Urutkan dari posisi terlama ke terbaru
    swings.sort(key=lambda x: x["posisi"])

    return swings


# =============================================================================
# SUB-FUNGSI 1: CEK TRENDING
# =============================================================================

def _check_trending(
    df_m15              : pd.DataFrame,
    idx                 : int,
    trend_lookback      : int   | None = None,
    trend_min_ema_gap   : float | None = None,
    trend_swing_wing    : int   | None = None,
    trend_min_swing_pairs: int  | None = None,
    trend_dir_min_ratio : float | None = None,
) -> dict:
    """
    Cek apakah kondisi market di idx memenuhi kriteria TRENDING.

    Parameter opsional (None = gunakan konstanta modul):
        trend_lookback        : override REGIME_TREND_LOOKBACK
        trend_min_ema_gap     : override REGIME_TREND_MIN_EMA_GAP_PCT
        trend_swing_wing      : override REGIME_TREND_SWING_WING
        trend_min_swing_pairs : override REGIME_TREND_MIN_SWING_PAIRS
        trend_dir_min_ratio   : override REGIME_TREND_DIRECTIONAL_MIN_RATIO

    TRENDING terpenuhi HANYA jika KETIGA syarat berikut terpenuhi (AND):

    (a) Konfirmasi EMA:
        Panggil detect_trend() dengan min_ema_gap_pct=trend_min_ema_gap.
        Kolom trend di idx harus "UPTREND" atau "DOWNTREND", bukan "SIDEWAYS".
        detect_trend() sudah dipanggil oleh run_all_indicators() — kolom
        'trend' sudah ada di df_m15. NAMUN, default min_ema_gap_pct di
        run_all_indicators() adalah 0.05 (M5 default), BEDA dari threshold M15
        (0.10). Oleh karena itu kita panggil detect_trend() lagi di sini
        dengan threshold M15 yang benar — hasilnya disimpan lokal, tidak
        mengubah kolom asli.

    (b) Konfirmasi struktur — rangkaian swing:
        Deteksi semua swing high dan low dalam window lookback menggunakan
        _detect_swing_sequence(). Dari rangkaian ini:
        - Bullish: minimal trend_min_swing_pairs higher-low berturut
          DAN minimal trend_min_swing_pairs higher-high berturut.
        - Bearish: minimal trend_min_swing_pairs lower-low berturut
          DAN minimal trend_min_swing_pairs lower-high berturut.

    (c) Konfirmasi konsistensi arah:
        Dalam window trend_lookback candle, proporsi candle dengan
        close di sisi EMA 21 yang benar >= trend_dir_min_ratio.

    Arah dari (a), (b), (c) HARUS konsisten — kalau EMA bilang UPTREND tapi
    struktur swing menunjukkan pola bearish, TRENDING tidak terpenuhi.

    Parameter:
        df_m15 : DataFrame M15 yang sudah punya kolom ema_gap_pct, ema_21,
                 close, high, low, serta kolom dari detect_trend().
        idx    : Index candle evaluasi (sudah ternormalisasi, >= 0).

    Return:
        dict dengan field:
            terpenuhi          : bool
            arah               : "BULLISH" | "BEARISH" | None
            ema_ok             : bool
            struktur_ok        : bool
            konsistensi_ok     : bool
            konsistensi_ratio  : float
            swing_count_confirm: int  — total pasang swing terurut yang ditemukan
            keterangan         : str
    """
    # ── Resolve parameter (None → konstanta modul) ────────────────────────────
    _lookback    = trend_lookback       if trend_lookback       is not None else REGIME_TREND_LOOKBACK
    _ema_gap     = trend_min_ema_gap    if trend_min_ema_gap    is not None else REGIME_TREND_MIN_EMA_GAP_PCT
    _wing        = trend_swing_wing     if trend_swing_wing     is not None else REGIME_TREND_SWING_WING
    _min_pairs   = trend_min_swing_pairs if trend_min_swing_pairs is not None else REGIME_TREND_MIN_SWING_PAIRS
    _dir_ratio   = trend_dir_min_ratio  if trend_dir_min_ratio  is not None else REGIME_TREND_DIRECTIONAL_MIN_RATIO

    # ── Hasil default ─────────────────────────────────────────────────────────
    hasil = {
        "terpenuhi"           : False,
        "arah"                : None,
        "ema_ok"              : False,
        "struktur_ok"         : False,
        "konsistensi_ok"      : False,
        "konsistensi_ratio"   : 0.0,
        "swing_count_confirm" : 0,
        "keterangan"          : "",
    }

    # ── (a) Konfirmasi EMA ────────────────────────────────────────────────────
    # Panggil detect_trend() dengan threshold M15 (berbeda dari default M5).
    # Gunakan salinan df untuk tidak mengubah kolom asli.
    df_trend = detect_trend(
        df_m15.iloc[:idx + 1].copy(),
        min_ema_gap_pct=_ema_gap
    )
    trend_val = str(df_trend["trend"].iloc[-1])
    # Kolom 'trend' di df_trend berisi label dari threshold M15.

    ema_ok   = trend_val in ("UPTREND", "DOWNTREND")
    arah_ema = None
    if trend_val == "UPTREND":
        arah_ema = "BULLISH"
    elif trend_val == "DOWNTREND":
        arah_ema = "BEARISH"

    hasil["ema_ok"] = ema_ok

    if not ema_ok:
        hasil["keterangan"] = (
            f"EMA gagal: trend M15 = '{trend_val}' "
            f"(threshold gap {_ema_gap}% tidak terpenuhi)"
        )
        return hasil

    # ── (b) Konfirmasi struktur — rangkaian swing ─────────────────────────────
    swings = _detect_swing_sequence(
        df_m15,
        idx     = idx,
        lookback= _lookback,
        wing    = _wing,
    )

    highs_sw = [s for s in swings if s["tipe"] == "HIGH"]
    lows_sw  = [s for s in swings if s["tipe"] == "LOW"]

    # Hitung pasangan terurut searah EMA
    swing_confirm = 0
    struktur_ok   = False
    arah_struktur = None

    if arah_ema == "BULLISH":
        # Hitung berapa consecutive higher-high di daftar swing high
        hh_pairs = 0
        for i in range(1, len(highs_sw)):
            if highs_sw[i]["harga"] > highs_sw[i - 1]["harga"]:
                hh_pairs += 1
        # Hitung berapa consecutive higher-low di daftar swing low
        hl_pairs = 0
        for i in range(1, len(lows_sw)):
            if lows_sw[i]["harga"] > lows_sw[i - 1]["harga"]:
                hl_pairs += 1

        # Pasangan minimal: min dari HH dan HL pairs (keduanya harus cukup)
        swing_confirm = min(hh_pairs, hl_pairs)
        if (hh_pairs >= _min_pairs and
                hl_pairs >= _min_pairs):
            struktur_ok   = True
            arah_struktur = "BULLISH"

    else:  # arah_ema == "BEARISH"
        # Hitung berapa consecutive lower-low di daftar swing low
        ll_pairs = 0
        for i in range(1, len(lows_sw)):
            if lows_sw[i]["harga"] < lows_sw[i - 1]["harga"]:
                ll_pairs += 1
        # Hitung berapa consecutive lower-high di daftar swing high
        lh_pairs = 0
        for i in range(1, len(highs_sw)):
            if highs_sw[i]["harga"] < highs_sw[i - 1]["harga"]:
                lh_pairs += 1

        swing_confirm = min(ll_pairs, lh_pairs)
        if (ll_pairs >= _min_pairs and
                lh_pairs >= _min_pairs):
            struktur_ok   = True
            arah_struktur = "BEARISH"

    hasil["struktur_ok"]            = struktur_ok
    hasil["swing_count_confirm"]    = swing_confirm

    if not struktur_ok:
        hasil["keterangan"] = (
            f"Struktur swing gagal: arah EMA={arah_ema}, "
            f"swing HIGH ditemukan={len(highs_sw)}, "
            f"swing LOW ditemukan={len(lows_sw)}, "
            f"pasang terurut={swing_confirm} "
            f"(butuh >= {_min_pairs})"
        )
        return hasil

    # ── Cek konsistensi arah antara EMA dan struktur ──────────────────────────
    if arah_ema != arah_struktur:
        hasil["keterangan"] = (
            f"Konflik arah: EMA={arah_ema} tapi struktur swing={arah_struktur}. "
            f"TRENDING tidak terpenuhi."
        )
        return hasil

    # ── (c) Konfirmasi konsistensi arah ──────────────────────────────────────
    window_start = max(0, idx - _lookback + 1)
    window_slice = df_m15.iloc[window_start : idx + 1]
    n_candle     = len(window_slice)

    ema21_vals = window_slice["ema_21"].values
    close_vals = window_slice["close"].values

    if arah_ema == "BULLISH":
        # Proporsi candle dengan close > ema_21
        n_searah = int(np.sum(close_vals > ema21_vals))
    else:
        # Proporsi candle dengan close < ema_21
        n_searah = int(np.sum(close_vals < ema21_vals))

    konsistensi_ratio = n_searah / n_candle if n_candle > 0 else 0.0
    konsistensi_ok    = konsistensi_ratio >= _dir_ratio

    hasil["konsistensi_ok"]    = konsistensi_ok
    hasil["konsistensi_ratio"] = float(konsistensi_ratio)

    if not konsistensi_ok:
        hasil["keterangan"] = (
            f"Konsistensi arah gagal: {n_searah}/{n_candle} candle di sisi EMA21 "
            f"yang benar = {konsistensi_ratio:.2%} "
            f"(butuh >= {_dir_ratio:.0%})"
        )
        return hasil

    # ── Semua syarat terpenuhi ─────────────────────────────────────────────────
    hasil["terpenuhi"] = True
    hasil["arah"]      = arah_ema
    hasil["keterangan"] = (
        f"TRENDING {arah_ema}: EMA gap OK (trend={trend_val}), "
        f"swing_confirm={swing_confirm} pasang (>= {_min_pairs}), "
        f"konsistensi={konsistensi_ratio:.2%} (>= {_dir_ratio:.0%})"
    )

    return hasil


# =============================================================================
# SUB-FUNGSI 2: CEK RANGING
# =============================================================================

def _check_ranging(
    df_m15                  : pd.DataFrame,
    idx                     : int,
    range_lookback          : int   | None = None,
    range_max_atr_ratio     : float | None = None,
    range_min_duration      : int   | None = None,
    range_touch_tolerance   : float | None = None,
    range_min_touches       : int   | None = None,
) -> dict:
    """
    Cek apakah kondisi market di idx memenuhi kriteria RANGING.

    Parameter opsional (None = gunakan konstanta modul):
        range_lookback        : override REGIME_RANGE_LOOKBACK
        range_max_atr_ratio   : override REGIME_RANGE_MAX_ATR_RATIO
        range_min_duration    : override REGIME_RANGE_MIN_DURATION
        range_touch_tolerance : override REGIME_RANGE_TOUCH_TOLERANCE_ATR
        range_min_touches     : override REGIME_RANGE_MIN_TOUCHES_PER_SIDE

    RANGING terpenuhi jika:
    1. detect_consolidation_zone() dengan idx=idx (bukan idx-1) mengembalikan
       is_valid=True — zona konsolidasi aktif saat ini.
    2. Dalam window durasi zona, ada minimal range_min_touches
       sentuhan ke boundary atas (resistance) DAN ke boundary bawah (support).
       Sentuhan dihitung dengan toleransi range_touch_tolerance * ATR.

    Catatan idx=idx (bukan idx-1):
        RANGING menilai kondisi SAAT INI — zona harus mencakup candle idx itu
        sendiri. Berbeda dengan BREAKOUT_TRANSITION yang memeriksa zona SEBELUM
        candle breakout (idx-1).

    Parameter:
        df_m15 : DataFrame M15 dengan kolom high, low, atr_14.
        idx    : Index candle evaluasi (sudah ternormalisasi, >= 0).

    Return:
        dict dengan field:
            terpenuhi          : bool
            zone               : dict  — hasil detect_consolidation_zone() mentah
            touches_resistance : int
            touches_support    : int
            keterangan         : str
    """
    # ── Resolve parameter (None → konstanta modul) ────────────────────────────
    _lookback   = range_lookback        if range_lookback        is not None else REGIME_RANGE_LOOKBACK
    _max_atr    = range_max_atr_ratio   if range_max_atr_ratio   is not None else REGIME_RANGE_MAX_ATR_RATIO
    _min_dur    = range_min_duration    if range_min_duration    is not None else REGIME_RANGE_MIN_DURATION
    _tolerance  = range_touch_tolerance if range_touch_tolerance is not None else REGIME_RANGE_TOUCH_TOLERANCE_ATR
    _min_touch  = range_min_touches     if range_min_touches     is not None else REGIME_RANGE_MIN_TOUCHES_PER_SIDE

    # ── Panggil detect_consolidation_zone dengan parameter M15 ────────────────
    zone = detect_consolidation_zone(
        df_m15,
        idx                 = idx,
        lookback            = _lookback,
        max_range_atr_ratio = _max_atr,
        min_duration_candles= _min_dur,
    )

    if not zone["is_valid"]:
        return {
            "terpenuhi"          : False,
            "zone"               : zone,
            "touches_resistance" : 0,
            "touches_support"    : 0,
            "keterangan"         : (
                f"Zona konsolidasi tidak valid: {zone['keterangan']}"
            ),
        }

    # ── Hitung sentuhan ke boundary ───────────────────────────────────────────
    resistance = zone["resistance"]
    support    = zone["support"]
    duration   = zone["duration"]

    atr_val    = float(df_m15["atr_14"].iloc[idx])
    toleransi  = _tolerance * atr_val

    # Window sentuhan: [idx - duration + 1, idx]
    window_start = max(0, idx - duration + 1)
    window_slice = df_m15.iloc[window_start : idx + 1]

    high_vals = window_slice["high"].values
    low_vals  = window_slice["low"].values

    # Sentuhan ke resistance: high >= resistance - toleransi
    touches_resistance = int(np.sum(high_vals >= resistance - toleransi))
    # Sentuhan ke support: low <= support + toleransi
    touches_support    = int(np.sum(low_vals  <= support    + toleransi))

    terpenuhi = (
        touches_resistance >= _min_touch and
        touches_support    >= _min_touch
    )

    if terpenuhi:
        keterangan = (
            f"RANGING VALID: zona [{support:.2f}, {resistance:.2f}], "
            f"sentuhan resistance={touches_resistance} "
            f"(>= {_min_touch}), "
            f"sentuhan support={touches_support} "
            f"(>= {_min_touch})"
        )
    else:
        keterangan = (
            f"RANGING gagal: zona ada tapi sentuhan tidak cukup — "
            f"resistance={touches_resistance}/{_min_touch}, "
            f"support={touches_support}/{_min_touch}"
        )

    return {
        "terpenuhi"          : terpenuhi,
        "zone"               : zone,
        "touches_resistance" : touches_resistance,
        "touches_support"    : touches_support,
        "keterangan"         : keterangan,
    }


# =============================================================================
# SUB-FUNGSI 3: CEK BREAKOUT_TRANSITION
# =============================================================================

def _check_breakout_transition(df_m15: pd.DataFrame, idx: int) -> dict:
    """
    Cek apakah candle di idx merupakan breakout dari zona konsolidasi sebelumnya.

    BREAKOUT_TRANSITION terpenuhi jika:
    1. Ada zona konsolidasi valid di idx-1 (zona SEBELUM candle yang dicek).
       detect_consolidation_zone() dipanggil dengan idx=idx-1 untuk menghindari
       circular (candle breakout tidak boleh ikut mendefinisikan zonanya sendiri).
    2. Close di idx menembus resistance (bullish) atau support (bearish) zona.
    3. Konfirmasi (salah satu — OR):
       - Body candle >= REGIME_BREAKOUT_MIN_BODY_ATR_RATIO * ATR, ATAU
       - volume_ratio >= REGIME_BREAKOUT_MIN_VOLUME_RATIO (jika kolom tersedia).

    Catatan idx-1:
        Pola identik dengan cara _check_breakout_trigger() dipakai di
        rule_engine.py dan backtester.py: zona dihitung SEBELUM candle breakout
        agar candle itu sendiri tidak ikut memperlebar definisi zona.

    Parameter:
        df_m15 : DataFrame M15 dengan kolom open, high, low, close, atr_14,
                 dan opsional volume_ratio.
        idx    : Index candle evaluasi (sudah ternormalisasi, >= 1).

    Return:
        dict dengan field:
            terpenuhi         : bool
            arah              : "BULLISH" | "BEARISH" | None
            zone              : dict  — zone_prior mentah dari detect_consolidation_zone()
            konfirmasi_body   : bool
            konfirmasi_volume : bool
            keterangan        : str
    """
    # ── Validasi idx minimal 1 ────────────────────────────────────────────────
    if idx < 1:
        return {
            "terpenuhi"         : False,
            "arah"              : None,
            "zone"              : None,
            "konfirmasi_body"   : False,
            "konfirmasi_volume" : False,
            "keterangan"        : f"idx={idx} terlalu kecil untuk cek breakout (butuh >= 1)",
        }

    # ── Zona prior (idx-1) ────────────────────────────────────────────────────
    # CAUSALITY: idx-1 < idx → selalu membaca data masa lalu.
    zone_prior = detect_consolidation_zone(
        df_m15,
        idx                 = idx - 1,
        lookback            = REGIME_BREAKOUT_LOOKBACK_ZONE,
        max_range_atr_ratio = REGIME_BREAKOUT_MAX_RANGE_ATR_RATIO,
        min_duration_candles= REGIME_BREAKOUT_MIN_DURATION,
    )

    if not zone_prior["is_valid"]:
        return {
            "terpenuhi"         : False,
            "arah"              : None,
            "zone"              : zone_prior,
            "konfirmasi_body"   : False,
            "konfirmasi_volume" : False,
            "keterangan"        : (
                f"Tidak ada zona prior valid di idx-1={idx-1}: "
                f"{zone_prior['keterangan']}"
            ),
        }

    resistance_prior = zone_prior["resistance"]
    support_prior    = zone_prior["support"]

    # ── Cek breakout di candle idx ────────────────────────────────────────────
    close_idx = float(df_m15["close"].iloc[idx])
    open_idx  = float(df_m15["open"].iloc[idx])
    atr_idx   = float(df_m15["atr_14"].iloc[idx])

    breakout_bullish = close_idx > resistance_prior
    breakout_bearish = close_idx < support_prior

    if not breakout_bullish and not breakout_bearish:
        return {
            "terpenuhi"         : False,
            "arah"              : None,
            "zone"              : zone_prior,
            "konfirmasi_body"   : False,
            "konfirmasi_volume" : False,
            "keterangan"        : (
                f"Tidak ada breakout: close={close_idx:.2f} "
                f"masih di dalam zona [{support_prior:.2f}, {resistance_prior:.2f}]"
            ),
        }

    arah_breakout = "BULLISH" if breakout_bullish else "BEARISH"

    # ── Konfirmasi body ───────────────────────────────────────────────────────
    body_idx       = abs(close_idx - open_idx)
    konfirmasi_body = (
        atr_idx > 0 and
        body_idx >= REGIME_BREAKOUT_MIN_BODY_ATR_RATIO * atr_idx
    )

    # ── Konfirmasi volume (graceful — lewati jika kolom tidak ada) ────────────
    konfirmasi_volume = False
    if "volume_ratio" in df_m15.columns:
        vol_ratio = df_m15["volume_ratio"].iloc[idx]
        if not pd.isna(vol_ratio):
            konfirmasi_volume = float(vol_ratio) >= REGIME_BREAKOUT_MIN_VOLUME_RATIO

    # ── Konfirmasi terpenuhi jika salah satu OK (OR) ──────────────────────────
    konfirmasi_ok = konfirmasi_body or konfirmasi_volume

    if not konfirmasi_ok:
        return {
            "terpenuhi"         : False,
            "arah"              : arah_breakout,
            "zone"              : zone_prior,
            "konfirmasi_body"   : konfirmasi_body,
            "konfirmasi_volume" : konfirmasi_volume,
            "keterangan"        : (
                f"Breakout {arah_breakout} ditemukan tapi konfirmasi lemah: "
                f"body={body_idx:.2f} (butuh >= {REGIME_BREAKOUT_MIN_BODY_ATR_RATIO}×ATR "
                f"= {REGIME_BREAKOUT_MIN_BODY_ATR_RATIO * atr_idx:.2f}), "
                f"volume_ok={konfirmasi_volume} "
                f"(ratio={float(df_m15['volume_ratio'].iloc[idx]) if 'volume_ratio' in df_m15.columns else 'N/A'})"
            ),
        }

    return {
        "terpenuhi"         : True,
        "arah"              : arah_breakout,
        "zone"              : zone_prior,
        "konfirmasi_body"   : konfirmasi_body,
        "konfirmasi_volume" : konfirmasi_volume,
        "keterangan"        : (
            f"BREAKOUT_TRANSITION {arah_breakout}: "
            f"close={close_idx:.2f} menembus "
            f"{'resistance' if arah_breakout == 'BULLISH' else 'support'}"
            f"={resistance_prior if arah_breakout == 'BULLISH' else support_prior:.2f}, "
            f"body_ok={konfirmasi_body}, volume_ok={konfirmasi_volume}"
        ),
    }


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def detect_market_regime(
    df_m15                  : pd.DataFrame,
    idx                     : int = -1,
    # --- Parameter override TRENDING (None = konstanta modul) ---
    trend_lookback          : int   | None = None,
    trend_min_ema_gap       : float | None = None,
    trend_swing_wing        : int   | None = None,
    trend_min_swing_pairs   : int   | None = None,
    trend_dir_min_ratio     : float | None = None,
    # --- Parameter override RANGING (None = konstanta modul) ---
    range_lookback          : int   | None = None,
    range_max_atr_ratio     : float | None = None,
    range_min_duration      : int   | None = None,
    range_touch_tolerance   : float | None = None,
    range_min_touches       : int   | None = None,
) -> dict:
    """
    Deteksi market regime di candle ke-idx dari df_m15.

    PARAMETER OVERRIDE:
        Semua parameter di atas bersifat opsional (default None).
        Jika None, masing-masing sub-fungsi menggunakan konstanta modul.
        Jika diberikan nilai, nilai itu menggantikan konstanta modul untuk
        pemanggilan ini saja — konstanta modul TIDAK berubah.
        Pola identik dengan calculate_sl_tp() di risk_manager.py.

        Gunakan REGIME_PARAMS_V2 (dict di atas) sebagai **kwargs untuk
        memanggil dengan parameter set Cycle 2:
            detect_market_regime(df, idx, **REGIME_PARAMS_V2)

    ALUR:
        1. Normalisasi idx negatif (pola sama seperti zone_detector.py).
        2. Validasi kolom dan data cukup — jika tidak, return CHOP.
        3. Hitung semua tiga pengecekan secara paralel konseptual:
               breakout_check = _check_breakout_transition(df_m15, idx)
               ranging_check  = _check_ranging(df_m15, idx, ...)
               trending_check = _check_trending(df_m15, idx, ...)
        4. Waterfall seleksi regime:
               BREAKOUT_TRANSITION > RANGING > TRENDING > CHOP
        5. Susun field "zone" sesuai regime terpilih.
        6. Return dict lengkap dengan semua field wajib.

    STATELESS:
        Fungsi ini TIDAK menerima riwayat regime sebelumnya dan TIDAK
        menyimpan state apapun antar pemanggilan. Setiap pemanggilan untuk
        (df_m15, idx) yang sama akan menghasilkan output yang sama persis
        tidak peduli urutan atau riwayat pemanggilan sebelumnya.

    CAUSALITY:
        Hanya membaca df_m15.iloc[:idx+1]. Candle setelah idx tidak pernah
        disentuh, termasuk saat memanggil detect_consolidation_zone().

    Parameter:
        df_m15 : DataFrame M15 yang SUDAH melewati run_all_indicators().
                 Kolom wajib: open, high, low, close, ema_9, ema_21,
                 ema_gap_pct, trend, atr_14.
                 Kolom opsional: volume_ratio (untuk konfirmasi breakout).
        idx    : Index candle evaluasi. Default -1 = candle terakhir.
                 Nilai negatif dinormalisasi sesuai pola standar.

    Return:
        dict dengan field:
            regime     : str        — "TRENDING" / "RANGING" / "BREAKOUT_TRANSITION" / "CHOP"
            arah       : str | None — "BULLISH" / "BEARISH" (hanya untuk TRENDING &
                                      BREAKOUT_TRANSITION), None untuk RANGING & CHOP.
            zone       : dict | None— hasil detect_consolidation_zone() yang relevan
                                      (untuk RANGING dan BREAKOUT_TRANSITION), None lainnya.
            detail     : dict       — breakdown audit SEMUA pengecekan (selalu ada,
                                      termasuk yang tidak memenangkan waterfall).
            keterangan : str        — penjelasan ringkas mengapa regime ini dipilih.
    """
    # ── 1. Normalisasi idx negatif ────────────────────────────────────────────
    n = len(df_m15)
    if idx < 0:
        idx = n + idx

    # ── 2. Validasi kolom wajib ───────────────────────────────────────────────
    required_cols = {"open", "high", "low", "close", "ema_9", "ema_21",
                     "ema_gap_pct", "trend", "atr_14"}
    missing = required_cols - set(df_m15.columns)
    if missing:
        return _empty_regime(
            keterangan=(
                f"Kolom tidak lengkap: {sorted(missing)} tidak ada di DataFrame. "
                f"Pastikan df_m15 sudah melewati run_all_indicators()."
            )
        )

    # ── 2b. Validasi idx dalam range dan data cukup untuk lookback terbesar ───
    # Resolve nilai lookback yang dipakai (override atau konstanta)
    _t_lookback  = trend_lookback    if trend_lookback    is not None else REGIME_TREND_LOOKBACK
    _t_wing      = trend_swing_wing  if trend_swing_wing  is not None else REGIME_TREND_SWING_WING
    _r_lookback  = range_lookback    if range_lookback    is not None else REGIME_RANGE_LOOKBACK

    # Lookback terbesar yang dibutuhkan: max dari semua lookback + wing konfirmasi
    # zone_detector memerlukan idx >= lookback-1 untuk RANGING (idx=idx)
    # zone_detector memerlukan idx-1 >= lookback_zone-1 untuk BREAKOUT (idx-1=idx-1)
    # swing sequence memerlukan idx >= lookback + wing - 1
    min_required_idx = max(
        _r_lookback - 1,                          # untuk ranging
        REGIME_BREAKOUT_LOOKBACK_ZONE,            # untuk breakout (idx-1 >= lookback-1)
        _t_lookback + _t_wing - 1,                # untuk swing scan
    )

    if idx < min_required_idx or idx >= n:
        return _empty_regime(
            keterangan=(
                f"Data tidak cukup: idx={idx}, butuh minimal idx >= {min_required_idx}, "
                f"total baris={n}. Kembalikan CHOP secara defensif."
            )
        )

    # ── 3. Hitung semua tiga pengecekan ──────────────────────────────────────
    # Semua tiga SELALU dihitung dan SELALU ada di detail (pola evaluate_entry()).
    breakout_check = _check_breakout_transition(df_m15, idx)
    ranging_check  = _check_ranging(
        df_m15, idx,
        range_lookback        = range_lookback,
        range_max_atr_ratio   = range_max_atr_ratio,
        range_min_duration    = range_min_duration,
        range_touch_tolerance = range_touch_tolerance,
        range_min_touches     = range_min_touches,
    )
    trending_check = _check_trending(
        df_m15, idx,
        trend_lookback        = trend_lookback,
        trend_min_ema_gap     = trend_min_ema_gap,
        trend_swing_wing      = trend_swing_wing,
        trend_min_swing_pairs = trend_min_swing_pairs,
        trend_dir_min_ratio   = trend_dir_min_ratio,
    )

    detail = {
        "breakout_check": breakout_check,
        "ranging_check" : ranging_check,
        "trending_check": trending_check,
    }

    # ── 4. Waterfall seleksi regime ───────────────────────────────────────────
    if breakout_check["terpenuhi"]:
        regime     = "BREAKOUT_TRANSITION"
        arah       = breakout_check["arah"]
        zone       = breakout_check["zone"]
        keterangan = (
            f"BREAKOUT_TRANSITION terpilih: {breakout_check['keterangan']}"
        )

    elif ranging_check["terpenuhi"]:
        regime     = "RANGING"
        arah       = None
        zone       = ranging_check["zone"]
        keterangan = (
            f"RANGING terpilih: {ranging_check['keterangan']}"
        )

    elif trending_check["terpenuhi"]:
        regime     = "TRENDING"
        arah       = trending_check["arah"]
        zone       = None
        keterangan = (
            f"TRENDING terpilih: {trending_check['keterangan']}"
        )

    else:
        regime     = "CHOP"
        arah       = None
        zone       = None
        keterangan = (
            "CHOP: semua pengecekan gagal. "
            f"Breakout: {breakout_check['keterangan']}. "
            f"Ranging: {ranging_check['keterangan']}. "
            f"Trending: {trending_check['keterangan']}."
        )

    # ── 5. Return dict lengkap ────────────────────────────────────────────────
    return {
        "regime"     : regime,
        "arah"       : arah,
        "zone"       : zone,
        "detail"     : detail,
        "keterangan" : keterangan,
    }
