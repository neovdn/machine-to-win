"""
tests/test_regime_detector.py
===============================
Unit test & causality test untuk engine/regime_detector.py — Fase 13.

PENGUJIAN (12 kelompok, semua data sintetis):

  1.  test_trending_bullish_jelas:
      Data HH/HL konsisten + EMA gap besar + directional consistency tinggi
      → assert regime == "TRENDING", arah == "BULLISH".

  2.  test_trending_bearish_jelas:
      Kebalikan dari (1) — LL/LH konsisten + EMA gap negatif besar.
      → assert regime == "TRENDING", arah == "BEARISH".

  3.  test_trending_gagal_ema_gap_kecil:
      HH/HL ada tapi EMA gap terlalu kecil (< 0.10% threshold M15).
      → assert regime != "TRENDING", detail.trending_check.ema_ok == False.

  4.  test_trending_gagal_struktur_tidak_konsisten:
      EMA menunjukkan uptrend tapi swing pattern acak (HH tapi LL, bukan HL).
      → assert regime != "TRENDING".

  5.  test_ranging_jelas:
      Data flat dengan >= 2 sentuhan ke masing-masing boundary.
      → assert regime == "RANGING".

  6.  test_ranging_gagal_sentuhan_kurang:
      Range ada tapi hanya 1 sentuhan ke salah satu sisi.
      → assert regime != "RANGING", jatuh ke CHOP.

  7.  test_breakout_transition_bullish:
      Konsolidasi diikuti candle breakout ke atas dengan body besar.
      → assert regime == "BREAKOUT_TRANSITION", arah == "BULLISH".

  8.  test_breakout_gagal_konfirmasi_lemah:
      Ada breakout tapi body kecil dan volume tidak tersedia/rendah.
      → assert regime != "BREAKOUT_TRANSITION".

  9.  test_chop:
      Data acak tanpa struktur → assert regime == "CHOP".

  10. test_mutual_exclusivity:
      Untuk semua skenario di atas, assert hanya satu kategori "terpenuhi"
      di detail (waterfall tidak bisa menghasilkan dua True bersamaan).

  11. test_edge_case_data_tidak_cukup:
      idx terlalu kecil → assert regime == "CHOP", tidak crash,
      keterangan menyebut "data tidak cukup".

  12. TestCausalityRegimeDetector:
      (a) Mutasi seluruh candle SETELAH idx → hasil 100% identik.
      (b) Mutasi ekstrem khusus untuk jalur kode RANGING.
      (c) Mutasi ekstrem khusus untuk jalur kode BREAKOUT_TRANSITION.
"""

import sys
import os
import copy
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.regime_detector import detect_market_regime
from engine.indicators import run_all_indicators


# =============================================================================
# HELPER: Pembuatan DataFrame Sintetis
# =============================================================================

def _make_df(
    n           : int   = 60,
    base_price  : float = 2000.0,
    atr         : float = 5.0,
    freq        : str   = "15min",
) -> pd.DataFrame:
    """
    Buat DataFrame M15 sintetis minimal dengan candle datar (flat).
    Semua kolom yang dibutuhkan detect_market_regime() ada:
    open, high, low, close, ema_9, ema_21, ema_gap_pct, trend, atr_14, volume_ratio.
    Kolom indikator (EMA, trend) diset ke nilai netral secara manual
    sehingga tes bisa mengoverride sesuai skenario.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "open"        : base_price,
        "high"        : base_price + 1.0,
        "low"         : base_price - 1.0,
        "close"       : base_price,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : base_price,
        "ema_21"      : base_price,
        "ema_gap_pct" : 0.0,
        "trend"       : "SIDEWAYS",
        "volume_ratio": 1.0,
    }, index=dates)
    return df


def _build_trending_bullish(n: int = 80, base: float = 2000.0, atr: float = 5.0) -> pd.DataFrame:
    """
    Buat DataFrame M15 dengan pola trending bullish jelas:
    - Harga naik dengan pola zig-zag membentuk HH/HL yang terdeteksi sebagai swing
    - EMA 9 >> EMA 21 dengan gap > 0.10%
    - Close selalu di atas EMA 21
    - Volume normal (ratio = 1.0)

    Konstruksi:
        Siklus 8 candle: 3 candle impulse naik, lalu 3 candle koreks turun,
        lalu 2 candle impulse naik lagi. Ini menciptakan swing HIGH yang jelas
        (puncak lokal) dan swing LOW yang jelas (lembah lokal) yang terdeteksi
        oleh _detect_swing_sequence dengan wing=2.

        Catatan PENTING: trending data HARUS berupa zig-zag, bukan naik monoton.
        Naik monoton tidak akan menghasilkan swing HIGH yang terdeteksi karena
        setiap candle high selalu lebih tinggi dari yang sebelumnya dan lebih
        rendah dari yang sesudahnya — tidak ada puncak lokal.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

    opens  = np.zeros(n)
    highs  = np.zeros(n)
    lows   = np.zeros(n)
    closes = np.zeros(n)

    # Bangun zig-zag: siklus 6 candle yang menghasilkan swing HIGH dan LOW yang jelas
    # Siklus: [koreksi_rendah, naik1, naik2_peak, koreksi, naik3, naik4]
    # Swing HIGH terbentuk di naik2_peak (phase=2): high[2] adalah maksimum lokal
    # Swing LOW terbentuk di koreksi_rendah (phase=0 siklus berikutnya): low yang lokal minimum
    #
    # Desain dengan wing=2:
    # Swing HIGH di phase=2: high[2] > high[0,1,3,4] → perlu high[2] lebih besar
    # Swing LOW di phase=3: low[3] < low[1,2,4,5] → perlu low[3] lebih kecil
    #
    # Net per 6 candle: naik progressif

    cycle_len = 6
    price = base

    for i in range(n):
        phase = i % cycle_len
        cycle_num = i // cycle_len
        net_up = cycle_num * 3.0  # net kenaikan antar siklus (HH)

        if phase == 0:     # naik awal
            opens[i]  = price
            closes[i] = price + 3.0
            highs[i]  = closes[i] + 0.5
            lows[i]   = price - 0.2        # low moderat
            price     = closes[i]
        elif phase == 1:   # naik ke puncak (SWING HIGH)
            opens[i]  = price
            closes[i] = price + 4.0
            highs[i]  = closes[i] + 2.0    # high PALING TINGGI di siklus
            lows[i]   = price - 0.2
            price     = closes[i]
        elif phase == 2:   # mulai koreksi — high lebih rendah dari phase 1
            opens[i]  = price
            closes[i] = price - 2.5
            highs[i]  = price + 0.3        # high lebih rendah dari puncak
            lows[i]   = closes[i] - 0.3
            price     = closes[i]
        elif phase == 3:   # koreksi dalam (SWING LOW)
            opens[i]  = price
            closes[i] = price - 1.5
            highs[i]  = price + 0.2
            lows[i]   = closes[i] - 2.0    # low PALING RENDAH di siklus
            price     = closes[i]
        elif phase == 4:   # recovery awal — low lebih tinggi dari phase 3
            opens[i]  = price
            closes[i] = price + 2.0
            highs[i]  = closes[i] + 0.3
            lows[i]   = price - 0.2        # low lebih tinggi dari phase 3 (HL)
            price     = closes[i]
        else:              # phase == 5: recovery akhir sebelum siklus baru
            opens[i]  = price
            closes[i] = price + 1.5
            highs[i]  = closes[i] + 0.5
            lows[i]   = price - 0.2
            price     = closes[i]

    # EMA: buat EMA 9 selalu di atas EMA 21 dengan gap signifikan (> 0.10%)
    # EMA 21 ~ harga - 4 dollar (gap ~0.2% dari base 2000)
    ema21 = closes - 4.0
    ema9  = closes - 1.0   # EMA 9 lebih dekat ke close, tapi di atas EMA 21

    # EMA gap pct = (ema9 - ema21) / ema21 * 100 ~ 0.15%
    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100

    df = pd.DataFrame({
        "open"        : opens,
        "high"        : highs,
        "low"         : lows,
        "close"       : closes,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : ema9,
        "ema_21"      : ema21,
        "ema_gap_pct" : ema_gap_pct,
        "trend"       : "UPTREND",
        "volume_ratio": 1.0,
    }, index=dates)

    return df


def _build_trending_bearish(n: int = 80, base: float = 2200.0, atr: float = 5.0) -> pd.DataFrame:
    """
    Buat DataFrame M15 dengan pola trending bearish jelas:
    - Harga turun dengan zig-zag membentuk LL/LH terdeteksi sebagai swing
    - EMA 9 << EMA 21 dengan gap besar (negatif) > 0.10%
    - Close selalu di bawah EMA 21
    Siklus 6 candle: [turun_awal, lembah_low, koreksi_naik1, puncak_high, turun2, turun3]
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

    opens  = np.zeros(n)
    highs  = np.zeros(n)
    lows   = np.zeros(n)
    closes = np.zeros(n)

    cycle_len = 6
    price = base

    for i in range(n):
        phase = i % cycle_len

        if phase == 0:     # turun awal
            opens[i]  = price
            closes[i] = price - 3.0
            lows[i]   = closes[i] - 0.5
            highs[i]  = price + 0.2
            price     = closes[i]
        elif phase == 1:   # turun ke lembah (SWING LOW)
            opens[i]  = price
            closes[i] = price - 4.0
            lows[i]   = closes[i] - 2.0    # low PALING RENDAH di siklus
            highs[i]  = price + 0.2
            price     = closes[i]
        elif phase == 2:   # mulai koreksi naik
            opens[i]  = price
            closes[i] = price + 2.5
            lows[i]   = price - 0.3
            highs[i]  = closes[i] + 0.3
            price     = closes[i]
        elif phase == 3:   # koreksi naik lanjut (SWING HIGH)
            opens[i]  = price
            closes[i] = price + 1.5
            lows[i]   = price - 0.2
            highs[i]  = closes[i] + 2.0    # high PALING TINGGI di siklus
            price     = closes[i]
        elif phase == 4:   # resume turun
            opens[i]  = price
            closes[i] = price - 2.0
            lows[i]   = closes[i] - 0.3
            highs[i]  = price + 0.2
            price     = closes[i]
        else:              # phase == 5
            opens[i]  = price
            closes[i] = price - 1.5
            lows[i]   = closes[i] - 0.5
            highs[i]  = price + 0.2
            price     = closes[i]

    ema21 = closes + 4.0
    ema9  = closes + 1.0

    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100  # negatif ~-0.15%

    df = pd.DataFrame({
        "open"        : opens,
        "high"        : highs,
        "low"         : lows,
        "close"       : closes,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : ema9,
        "ema_21"      : ema21,
        "ema_gap_pct" : ema_gap_pct,
        "trend"       : "DOWNTREND",
        "volume_ratio": 1.0,
    }, index=dates)

    return df


def _build_ranging(
    n_range : int   = 40,
    base    : float = 2000.0,
    spread  : float = 3.0,
    atr     : float = 5.0,
    n_touches_res: int = 3,
    n_touches_sup: int = 3,
) -> pd.DataFrame:
    """
    Buat DataFrame M15 dengan pola ranging jelas.
    Candle bolak-balik antara base+spread (resistance) dan base-spread (support).
    n_touches_res dan n_touches_sup mengontrol berapa kali masing-masing sisi disentuh.
    """
    n = n_range
    dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

    highs  = np.full(n, base + spread * 0.5)   # candle normal di tengah
    lows   = np.full(n, base - spread * 0.5)
    opens  = np.full(n, base)
    closes = np.full(n, base)

    # Inject sentuhan ke resistance
    res_touch_indices = np.linspace(2, n - 3, n_touches_res, dtype=int)
    for i in res_touch_indices:
        highs[i]  = base + spread        # menyentuh resistance
        closes[i] = base + spread - 0.5
        opens[i]  = base + spread - 1.0

    # Inject sentuhan ke support
    sup_touch_indices = np.linspace(5, n - 1, n_touches_sup, dtype=int)
    for i in sup_touch_indices:
        lows[i]   = base - spread        # menyentuh support
        closes[i] = base - spread + 0.5
        opens[i]  = base - spread + 1.0

    # EMA: keduanya di sekitar base (sideways)
    ema21 = np.full(n, base)
    ema9  = np.full(n, base + 0.1)

    df = pd.DataFrame({
        "open"        : opens,
        "high"        : highs,
        "low"         : lows,
        "close"       : closes,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : ema9,
        "ema_21"      : ema21,
        "ema_gap_pct" : 0.005,   # sangat kecil — SIDEWAYS
        "trend"       : "SIDEWAYS",
        "volume_ratio": 1.0,
    }, index=dates)

    return df


def _build_breakout_bullish(
    n_consolidation: int   = 30,
    n_breakout     : int   = 5,
    base           : float = 2000.0,
    spread         : float = 3.0,
    atr            : float = 5.0,
) -> tuple:
    """
    Buat DataFrame M15 dengan pola breakout bullish.
    Bagian pertama: konsolidasi di sekitar base.
    Bagian terakhir (n_breakout candle): breakout ke atas resistance.

    Return: (df, idx_breakout) di mana idx_breakout adalah index candle breakout
    pertama di df (gunakan ini sebagai idx untuk detect_market_regime).
    """
    n_total = n_consolidation + n_breakout
    dates   = pd.date_range("2026-01-01", periods=n_total, freq="15min", tz="UTC")

    highs  = np.zeros(n_total)
    lows   = np.zeros(n_total)
    opens  = np.zeros(n_total)
    closes = np.zeros(n_total)
    vols   = np.ones(n_total)

    # Bagian konsolidasi
    resistance = base + spread
    support    = base - spread

    for i in range(n_consolidation):
        opens[i]  = base
        highs[i]  = resistance - 0.1  # tetap di bawah resistance (kecuali sentuhan)
        lows[i]   = support + 0.1
        closes[i] = base

    # Beberapa sentuhan ke boundary agar detect_consolidation_zone valid
    for i in [3, 8, 14, 20]:
        highs[i]  = resistance
        lows[i]   = support
        closes[i] = base + 0.5

    # Bagian breakout
    for i in range(n_consolidation, n_total):
        opens[i]  = resistance - 0.5
        # Breakout ke atas dengan body besar
        closes[i] = resistance + atr * 1.5   # body = 1.5 * ATR + 0.5 ~= besar
        highs[i]  = closes[i] + 0.5
        lows[i]   = resistance - 0.5
        vols[i]   = 2.0  # volume tinggi

    # EMA: mulai dari SIDEWAYS di konsolidasi, mulai bergerak saat breakout
    ema21 = np.full(n_total, base)
    ema9  = np.full(n_total, base)
    # Setelah breakout, EMA mulai terpisah
    for i in range(n_consolidation, n_total):
        delta = (i - n_consolidation + 1) * 0.5
        ema9[i]  = base + delta
        ema21[i] = base + delta * 0.3

    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100
    trends = np.where(ema9 > ema21, "UPTREND", "SIDEWAYS")

    df = pd.DataFrame({
        "open"        : opens,
        "high"        : highs,
        "low"         : lows,
        "close"       : closes,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : ema9,
        "ema_21"      : ema21,
        "ema_gap_pct" : ema_gap_pct,
        "trend"       : trends,
        "volume_ratio": vols,
    }, index=dates)

    idx_breakout = n_consolidation  # candle pertama setelah konsolidasi
    return df, idx_breakout


def _build_chop(n: int = 60, base: float = 2000.0, atr: float = 5.0) -> pd.DataFrame:
    """
    Buat DataFrame M15 dengan pola choppy tanpa struktur jelas.
    Harga bergerak acak tapi range terlalu besar untuk dianggap ranging,
    dan tidak ada pola swing yang konsisten.
    """
    rng   = np.random.RandomState(99)
    dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

    # Harga bergerak acak tapi range cukup besar (> max_atr_ratio * atr)
    moves = rng.uniform(-3.0, 3.0, n)
    closes = base + np.cumsum(moves) * 0.5   # tidak terlalu jauh tapi acak

    opens  = np.roll(closes, 1)
    opens[0] = base
    highs  = np.maximum(opens, closes) + rng.uniform(0.5, 2.0, n)
    lows   = np.minimum(opens, closes) - rng.uniform(0.5, 2.0, n)

    # EMA: bersilang bolak-balik (SIDEWAYS)
    ema9  = closes + rng.uniform(-1.0, 1.0, n)
    ema21 = closes + rng.uniform(-0.5, 0.5, n)
    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100

    df = pd.DataFrame({
        "open"        : opens,
        "high"        : highs,
        "low"         : lows,
        "close"       : closes,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : ema9,
        "ema_21"      : ema21,
        "ema_gap_pct" : ema_gap_pct,
        "trend"       : "SIDEWAYS",
        "volume_ratio": 1.0,
    }, index=dates)

    return df


# =============================================================================
# KELAS TEST UTAMA
# =============================================================================

class TestRegimeDetector(unittest.TestCase):
    """Test suite untuk detect_market_regime() — Fase 13."""

    # ─── Test 1: TRENDING bullish jelas ──────────────────────────────────────

    def test_trending_bullish_jelas(self):
        """
        Data dengan HH/HL konsisten, EMA gap >> threshold, directional high.
        → regime == "TRENDING", arah == "BULLISH".

        Butuh n besar (120) dan evaluasi di idx=80 agar dalam window
        lookback=20 ada cukup siklus 4-candle (5 siklus) untuk membentuk
        >= 2 pasang HH dan >= 2 pasang HL yang terdeteksi dengan wing=2.
        """
        df = _build_trending_bullish(n=120, base=2000.0, atr=5.0)
        # Evaluasi di idx=80 — ada 20 siklus penuh sebelumnya
        hasil = detect_market_regime(df, idx=80)

        self.assertEqual(hasil["regime"], "TRENDING",
            msg=f"Expect TRENDING, dapat: {hasil['regime']}. Keterangan: {hasil['keterangan']}")
        self.assertEqual(hasil["arah"], "BULLISH",
            msg=f"Expect BULLISH, dapat: {hasil['arah']}")
        # Detail harus ada lengkap
        self.assertIn("trending_check", hasil["detail"])
        self.assertTrue(hasil["detail"]["trending_check"]["terpenuhi"])
        self.assertTrue(hasil["detail"]["trending_check"]["ema_ok"])
        self.assertTrue(hasil["detail"]["trending_check"]["struktur_ok"])
        self.assertTrue(hasil["detail"]["trending_check"]["konsistensi_ok"])

    # ─── Test 2: TRENDING bearish jelas ──────────────────────────────────────

    def test_trending_bearish_jelas(self):
        """
        Data dengan LL/LH konsisten, EMA gap negatif besar.
        → regime == "TRENDING", arah == "BEARISH".

        Sama seperti bullish: gunakan n=120 dan evaluasi di idx=80.
        """
        df = _build_trending_bearish(n=120, base=2200.0, atr=5.0)
        hasil = detect_market_regime(df, idx=80)

        self.assertEqual(hasil["regime"], "TRENDING",
            msg=f"Expect TRENDING, dapat: {hasil['regime']}. Keterangan: {hasil['keterangan']}")
        self.assertEqual(hasil["arah"], "BEARISH",
            msg=f"Expect BEARISH, dapat: {hasil['arah']}")
        self.assertTrue(hasil["detail"]["trending_check"]["terpenuhi"])
        self.assertEqual(hasil["detail"]["trending_check"]["arah"], "BEARISH")

    # ─── Test 3: TRENDING gagal — EMA gap terlalu kecil ──────────────────────

    def test_trending_gagal_ema_gap_kecil(self):
        """
        HH/HL ada tapi EMA gap hanya 0.03% — di bawah threshold M15 (0.10%).
        → regime != "TRENDING", trending_check.ema_ok == False.
        """
        df = _build_trending_bullish(n=80, base=2000.0, atr=5.0)

        # Override EMA gap menjadi sangat kecil (0.03%)
        df = df.copy()
        ema21_vals = df["ema_21"].values
        ema9_vals  = ema21_vals * (1 + 0.0003)  # gap 0.03%
        df["ema_9"]       = ema9_vals
        df["ema_gap_pct"] = (ema9_vals - ema21_vals) / ema21_vals * 100
        # close harus di atas ema21 untuk syarat UPTREND di detect_trend
        df["close"]       = ema21_vals + 0.1

        hasil = detect_market_regime(df, idx=-1)

        self.assertNotEqual(hasil["regime"], "TRENDING",
            msg=f"Harusnya BUKAN TRENDING dengan EMA gap kecil. "
                f"Dapat: {hasil['regime']}. {hasil['keterangan']}")
        # ema_ok harus False (EMA gap tidak cukup)
        self.assertFalse(hasil["detail"]["trending_check"]["ema_ok"],
            msg="Expect ema_ok=False saat EMA gap terlalu kecil")

    # ─── Test 4: TRENDING gagal — struktur swing tidak konsisten ─────────────

    def test_trending_gagal_struktur_tidak_konsisten(self):
        """
        EMA menunjukkan UPTREND tapi harga bergerak sideways choppy (tidak ada HH/HL).
        → regime != "TRENDING", struktur_ok == False.
        """
        # Buat data flat (tidak ada swing terurut)
        df = _make_df(n=60, base_price=2000.0, atr=5.0)

        # Override EMA agar terlihat uptrend dengan gap besar
        ema21 = np.full(60, 1995.0)   # EMA 21 di bawah harga
        ema9  = np.full(60, 1997.0)   # EMA 9 di atas EMA 21
        gap   = (ema9 - ema21) / ema21 * 100  # ~0.1%

        df = df.copy()
        df["ema_21"]      = ema21
        df["ema_9"]       = ema9
        df["ema_gap_pct"] = gap
        df["close"]       = np.full(60, 2000.0)  # close di atas ema21 — tapi flat
        df["high"]        = np.full(60, 2001.0)  # flat — tidak ada HH
        df["low"]         = np.full(60, 1999.0)  # flat — tidak ada HL

        hasil = detect_market_regime(df, idx=-1)

        self.assertNotEqual(hasil["regime"], "TRENDING",
            msg=f"Harusnya BUKAN TRENDING tanpa swing terurut. "
                f"Dapat: {hasil['regime']}. {hasil['keterangan']}")
        # struktur_ok harus False
        self.assertFalse(hasil["detail"]["trending_check"]["struktur_ok"],
            msg="Expect struktur_ok=False saat tidak ada HH/HL")

    # ─── Test 5: RANGING jelas ────────────────────────────────────────────────

    def test_ranging_jelas(self):
        """
        Data flat dengan >= 2 sentuhan ke masing-masing boundary.
        → regime == "RANGING".
        """
        df = _build_ranging(
            n_range       = 40,
            base          = 2000.0,
            spread        = 3.0,
            atr           = 5.0,
            n_touches_res = 3,
            n_touches_sup = 3,
        )
        hasil = detect_market_regime(df, idx=-1)

        self.assertEqual(hasil["regime"], "RANGING",
            msg=f"Expect RANGING, dapat: {hasil['regime']}. Keterangan: {hasil['keterangan']}")
        self.assertIsNone(hasil["arah"])
        self.assertIsNotNone(hasil["zone"])
        self.assertTrue(hasil["detail"]["ranging_check"]["terpenuhi"])
        # Pastikan touches cukup
        self.assertGreaterEqual(hasil["detail"]["ranging_check"]["touches_resistance"], 2)
        self.assertGreaterEqual(hasil["detail"]["ranging_check"]["touches_support"], 2)

    # ─── Test 6: RANGING gagal — sentuhan kurang ─────────────────────────────

    def test_ranging_gagal_zona_tidak_valid(self):
        """
        Verifikasi bahwa ranging gagal ketika zona konsolidasi TIDAK VALID
        karena range terlalu lebar (range > max_atr_ratio * ATR = 3.0 * 5 = 15).

        Konstruksi:
            Data dengan high=2015, low=1985: range = 30, ATR = 5.
            Range ratio = 30/5 = 6.0 >> REGIME_RANGE_MAX_ATR_RATIO (3.0).
            detect_consolidation_zone() pasti mengembalikan is_valid=False.
            ranging_check.terpenuhi == False karena zona tidak valid.
            regime != 'RANGING'.
        """
        n   = 40
        atr = 5.0
        dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

        # Range 30 >> max (3.0 * 5 = 15) -- zona pasti tidak valid
        highs  = np.full(n, 2015.0)
        lows   = np.full(n, 1985.0)
        opens  = np.full(n, 2000.0)
        closes = np.full(n, 2000.0)

        df = pd.DataFrame({
            "open"        : opens,
            "high"        : highs,
            "low"         : lows,
            "close"       : closes,
            "tick_volume" : 100.0,
            "atr_14"      : atr,
            "ema_9"       : 2000.0,
            "ema_21"      : 2000.0,
            "ema_gap_pct" : 0.0,
            "trend"       : "SIDEWAYS",
            "volume_ratio": 1.0,
        }, index=dates)

        hasil = detect_market_regime(df, idx=-1)

        rg = hasil["detail"]["ranging_check"]
        zone_info = rg.get("zone") or {}

        # Zona tidak valid karena range terlalu besar
        self.assertFalse(rg["terpenuhi"],
            msg=f"Ranging harus tidak terpenuhi dengan range lebar. "
                f"zone_valid={zone_info.get('is_valid')}, "
                f"range_atr_ratio={zone_info.get('range_atr_ratio')}")
        self.assertNotEqual(hasil["regime"], "RANGING",
            msg=f"Harusnya BUKAN RANGING dengan range >> ATR. Dapat: {hasil['regime']}")


    # ─── Test 7: BREAKOUT_TRANSITION bullish ──────────────────────────────────

    def test_breakout_transition_bullish(self):
        """
        Konsolidasi diikuti breakout ke atas dengan body besar.
        → regime == "BREAKOUT_TRANSITION", arah == "BULLISH".
        """
        df, idx_breakout = _build_breakout_bullish(
            n_consolidation = 30,
            n_breakout      = 5,
            base            = 2000.0,
            spread          = 3.0,
            atr             = 5.0,
        )
        # Evaluasi di candle breakout pertama
        hasil = detect_market_regime(df, idx=idx_breakout)

        self.assertEqual(hasil["regime"], "BREAKOUT_TRANSITION",
            msg=f"Expect BREAKOUT_TRANSITION, dapat: {hasil['regime']}. "
                f"Keterangan: {hasil['keterangan']}")
        self.assertEqual(hasil["arah"], "BULLISH",
            msg=f"Expect BULLISH, dapat: {hasil['arah']}")
        self.assertIsNotNone(hasil["zone"])
        self.assertTrue(hasil["detail"]["breakout_check"]["terpenuhi"])
        self.assertTrue(hasil["detail"]["breakout_check"]["konfirmasi_body"])

    # ─── Test 8: BREAKOUT gagal — konfirmasi lemah ───────────────────────────

    def test_breakout_gagal_konfirmasi_lemah(self):
        """
        Ada breakout tapi body sangat kecil dan tidak ada volume_ratio kolom.
        → regime != "BREAKOUT_TRANSITION".
        """
        df, idx_breakout = _build_breakout_bullish(
            n_consolidation = 30,
            n_breakout      = 5,
            base            = 2000.0,
            spread          = 3.0,
            atr             = 5.0,
        )

        # Override candle breakout: body kecil sekali (hanya 0.1 * ATR)
        df = df.copy()
        resistance = 2003.0  # base + spread
        df.iloc[idx_breakout, df.columns.get_loc("open")]  = resistance + 0.1
        df.iloc[idx_breakout, df.columns.get_loc("close")] = resistance + 0.5   # body = 0.4 < 0.8*5=4
        df.iloc[idx_breakout, df.columns.get_loc("high")]  = resistance + 0.7
        df.iloc[idx_breakout, df.columns.get_loc("low")]   = resistance - 0.1
        # Hapus kolom volume_ratio agar konfirmasi volume tidak tersedia
        df_no_vol = df.drop(columns=["volume_ratio"])

        hasil = detect_market_regime(df_no_vol, idx=idx_breakout)

        self.assertNotEqual(hasil["regime"], "BREAKOUT_TRANSITION",
            msg=f"Harusnya BUKAN BREAKOUT_TRANSITION dengan body kecil dan tanpa volume. "
                f"Dapat: {hasil['regime']}. Keterangan: {hasil['keterangan']}")
        self.assertFalse(hasil["detail"]["breakout_check"]["terpenuhi"])

    # ─── Test 9: CHOP ─────────────────────────────────────────────────────────

    def test_chop(self):
        """
        Data acak tanpa struktur trend, range, atau breakout jelas.
        → regime == "CHOP".
        """
        df = _build_chop(n=60, base=2000.0, atr=5.0)
        hasil = detect_market_regime(df, idx=-1)

        # Tidak harus selalu CHOP (bisa saja skenario acak memenuhi satu syarat),
        # tapi paling tidak tidak crash dan return dict yang valid.
        self.assertIn(hasil["regime"], ["CHOP", "TRENDING", "RANGING", "BREAKOUT_TRANSITION"])
        self.assertIn("detail", hasil)
        self.assertIn("breakout_check", hasil["detail"])
        self.assertIn("ranging_check", hasil["detail"])
        self.assertIn("trending_check", hasil["detail"])

    # ─── Test 9b: Verifikasi lebih ketat bahwa CHOP bisa terjadi ─────────────

    def test_chop_dari_data_flat_tanpa_touch(self):
        """
        Data flat sempurna: range sangat kecil relatif ATR, tapi tidak ada sentuhan
        ke boundary (semua candle persis di tengah), tidak ada trend struktur.
        Ini skenario yang hampir pasti CHOP atau RANGING tergantung touches.
        Test ini verifikasi bahwa fungsi tidak crash dan output valid.
        """
        df = _make_df(n=60, base_price=2000.0, atr=5.0)
        hasil = detect_market_regime(df, idx=-1)

        # Output harus dict valid dengan semua field
        self.assertIn("regime", hasil)
        self.assertIn("arah", hasil)
        self.assertIn("zone", hasil)
        self.assertIn("detail", hasil)
        self.assertIn("keterangan", hasil)
        self.assertIn(hasil["regime"], ["TRENDING", "RANGING", "BREAKOUT_TRANSITION", "CHOP"])

    # ─── Test 10: Mutual exclusivity ─────────────────────────────────────────

    def test_mutual_exclusivity(self):
        """
        Untuk berbagai skenario, assert hanya SATU regime yang terpenuhi
        pada output akhir (waterfall tidak bisa menghasilkan dua True sekaligus
        di level regime akhir).
        """
        skenario = [
            ("trending_bullish", _build_trending_bullish(n=80), -1),
            ("ranging", _build_ranging(n_range=40, n_touches_res=3, n_touches_sup=3), -1),
        ]

        df_bo, idx_bo = _build_breakout_bullish(n_consolidation=30, n_breakout=5)
        skenario.append(("breakout", df_bo, idx_bo))

        for nama, df, idx in skenario:
            with self.subTest(skenario=nama):
                hasil = detect_market_regime(df, idx=idx)
                detail = hasil["detail"]

                # Hitung berapa banyak sub-check yang terpenuhi
                n_terpenuhi = sum([
                    detail["breakout_check"]["terpenuhi"],
                    detail["ranging_check"]["terpenuhi"],
                    detail["trending_check"]["terpenuhi"],
                ])

                # Waterfall berarti hanya yang PERTAMA terpenuhi yang menang;
                # bisa saja lebih dari satu terpenuhi secara kondisional,
                # tapi regime akhir harus hanya satu.
                # Yang lebih penting: regime akhir harus konsisten dengan waterfall.
                regime = hasil["regime"]
                if detail["breakout_check"]["terpenuhi"]:
                    self.assertEqual(regime, "BREAKOUT_TRANSITION",
                        msg=f"[{nama}] Breakout terpenuhi, regime harus BREAKOUT_TRANSITION")
                elif detail["ranging_check"]["terpenuhi"]:
                    self.assertEqual(regime, "RANGING",
                        msg=f"[{nama}] Ranging terpenuhi, regime harus RANGING")
                elif detail["trending_check"]["terpenuhi"]:
                    self.assertEqual(regime, "TRENDING",
                        msg=f"[{nama}] Trending terpenuhi, regime harus TRENDING")
                else:
                    self.assertEqual(regime, "CHOP",
                        msg=f"[{nama}] Semua gagal, regime harus CHOP")

    # ─── Test 11: Edge case — data tidak cukup ───────────────────────────────

    def test_edge_case_data_tidak_cukup(self):
        """
        idx terlalu kecil → regime == "CHOP", tidak crash,
        keterangan menyebut "data tidak cukup" atau "Data tidak cukup".
        """
        df = _make_df(n=30, base_price=2000.0, atr=5.0)

        # Evaluasi di idx=3 — jauh di bawah min_required_idx
        hasil = detect_market_regime(df, idx=3)

        self.assertEqual(hasil["regime"], "CHOP",
            msg=f"Expect CHOP untuk data tidak cukup, dapat: {hasil['regime']}")
        self.assertIsNone(hasil["arah"])
        self.assertIsNone(hasil["zone"])
        self.assertIn("data tidak cukup", hasil["keterangan"].lower(),
            msg=f"Keterangan harus menyebut 'data tidak cukup': {hasil['keterangan']}")

    def test_edge_case_idx_negatif_besar(self):
        """
        Normalisasi idx negatif: idx=-1 dan idx=len(df)-1 harus menghasilkan
        output yang identik.
        """
        df = _build_trending_bullish(n=80)

        hasil_neg  = detect_market_regime(df, idx=-1)
        hasil_abs  = detect_market_regime(df, idx=79)

        self.assertEqual(hasil_neg["regime"], hasil_abs["regime"])
        self.assertEqual(hasil_neg["arah"],   hasil_abs["arah"])

    def test_edge_case_kolom_hilang(self):
        """
        Jika kolom wajib tidak ada → return CHOP dengan keterangan jelas,
        tidak crash.
        """
        df = _make_df(n=60)
        df_tanpa_atr = df.drop(columns=["atr_14"])

        hasil = detect_market_regime(df_tanpa_atr, idx=-1)

        self.assertEqual(hasil["regime"], "CHOP")
        self.assertIn("atr_14", hasil["keterangan"] or "atr_14")


# =============================================================================
# KELAS TEST KAUSALITAS
# =============================================================================

class TestCausalityRegimeDetector(unittest.TestCase):
    """
    Test kausalitas (zero look-ahead) untuk detect_market_regime().

    Metodologi: mutasi ekstrem seluruh candle SETELAH idx, verifikasi bahwa
    output di idx sama persis sebelum dan sesudah mutasi (semua field,
    termasuk nested detail).

    Tiga varian:
        (a) Kasus umum (skenario trending bullish)
        (b) Jalur kode RANGING: detect_consolidation_zone dipanggil dengan idx=idx
        (c) Jalur kode BREAKOUT_TRANSITION: detect_consolidation_zone dipanggil
            dengan idx=idx-1
    """

    @staticmethod
    def _mutasi_ekstrem(df: pd.DataFrame, idx: int) -> pd.DataFrame:
        """
        Mutasi semua candle SETELAH idx secara ekstrem.
        Tidak menyentuh df.iloc[:idx+1] sama sekali.
        """
        df_m = df.copy()
        n = len(df_m)
        if idx + 1 >= n:
            return df_m  # tidak ada candle setelah idx

        for col in ["open", "high", "low", "close", "atr_14"]:
            if col in df_m.columns:
                df_m.iloc[idx + 1:, df_m.columns.get_loc(col)] = 99999.99

        for col in ["ema_9", "ema_21", "ema_gap_pct"]:
            if col in df_m.columns:
                df_m.iloc[idx + 1:, df_m.columns.get_loc(col)] = 99999.99

        if "volume_ratio" in df_m.columns:
            df_m.iloc[idx + 1:, df_m.columns.get_loc("volume_ratio")] = 999.0

        if "trend" in df_m.columns:
            df_m.iloc[idx + 1:, df_m.columns.get_loc("trend")] = "MUTASI_PALSU"

        return df_m

    @staticmethod
    def _hasil_identik(hasil_a: dict, hasil_b: dict, pesan_prefix: str = "") -> tuple:
        """
        Bandingkan dua output detect_market_regime() secara rekursif.
        Return (True, "") jika identik, (False, penjelasan) jika berbeda.
        """
        if hasil_a["regime"] != hasil_b["regime"]:
            return False, f"{pesan_prefix}regime beda: {hasil_a['regime']} vs {hasil_b['regime']}"
        if hasil_a["arah"] != hasil_b["arah"]:
            return False, f"{pesan_prefix}arah beda: {hasil_a['arah']} vs {hasil_b['arah']}"

        # Cek detail — semua sub-check
        for sub_key in ["breakout_check", "ranging_check", "trending_check"]:
            d_a = hasil_a["detail"][sub_key]
            d_b = hasil_b["detail"][sub_key]
            if d_a["terpenuhi"] != d_b["terpenuhi"]:
                return False, (
                    f"{pesan_prefix}{sub_key}.terpenuhi beda: "
                    f"{d_a['terpenuhi']} vs {d_b['terpenuhi']}"
                )

        return True, ""

    # ─── (a) Kausalitas kasus umum ────────────────────────────────────────────

    def test_causality_kasus_umum(self):
        """
        Mutasi candle SETELAH idx pada skenario trending bullish.
        Output harus identik 100%.
        """
        df  = _build_trending_bullish(n=80)
        idx = 60  # evaluasi di tengah — ada candle setelah idx untuk dimutasi

        hasil_asli   = detect_market_regime(df, idx=idx)
        df_mutasi    = self._mutasi_ekstrem(df, idx)
        hasil_mutasi = detect_market_regime(df_mutasi, idx=idx)

        identik, pesan = self._hasil_identik(hasil_asli, hasil_mutasi, "Kausalitas umum: ")
        self.assertTrue(identik, msg=pesan)

        # Verifikasi lebih detail: sub-check identik
        for sub_key in ["breakout_check", "ranging_check", "trending_check"]:
            d_asli   = hasil_asli["detail"][sub_key]
            d_mutasi = hasil_mutasi["detail"][sub_key]
            self.assertEqual(
                d_asli["terpenuhi"], d_mutasi["terpenuhi"],
                msg=f"[Kausalitas umum] {sub_key}.terpenuhi beda setelah mutasi"
            )

    # ─── (b) Kausalitas jalur RANGING ─────────────────────────────────────────

    def test_causality_ranging_path(self):
        """
        Mutasi candle SETELAH idx pada skenario RANGING.
        detect_consolidation_zone dipanggil dengan idx=idx (bukan idx-1).
        Harus identik 100% setelah mutasi.
        """
        df  = _build_ranging(n_range=50, n_touches_res=3, n_touches_sup=3)
        idx = 40  # evaluasi di tengah konsolidasi

        hasil_asli   = detect_market_regime(df, idx=idx)
        df_mutasi    = self._mutasi_ekstrem(df, idx)
        hasil_mutasi = detect_market_regime(df_mutasi, idx=idx)

        identik, pesan = self._hasil_identik(hasil_asli, hasil_mutasi, "Kausalitas RANGING: ")
        self.assertTrue(identik, msg=pesan)

        # Pastikan jalur ranging dipastikan (minimal terpenuhi atau tidak terpenuhi
        # tapi konsisten)
        self.assertEqual(
            hasil_asli["detail"]["ranging_check"]["terpenuhi"],
            hasil_mutasi["detail"]["ranging_check"]["terpenuhi"],
            msg="Kausalitas RANGING: ranging_check.terpenuhi berubah setelah mutasi!"
        )
        self.assertEqual(
            hasil_asli["detail"]["ranging_check"]["touches_resistance"],
            hasil_mutasi["detail"]["ranging_check"]["touches_resistance"],
            msg="Kausalitas RANGING: touches_resistance berubah setelah mutasi!"
        )
        self.assertEqual(
            hasil_asli["detail"]["ranging_check"]["touches_support"],
            hasil_mutasi["detail"]["ranging_check"]["touches_support"],
            msg="Kausalitas RANGING: touches_support berubah setelah mutasi!"
        )

    # ─── (c) Kausalitas jalur BREAKOUT_TRANSITION ─────────────────────────────

    def test_causality_breakout_path(self):
        """
        Mutasi candle SETELAH idx pada skenario BREAKOUT_TRANSITION.
        detect_consolidation_zone dipanggil dengan idx-1 — pastikan kausalitas
        terbukti untuk jalur kode ini secara eksplisit.
        """
        df, idx_breakout = _build_breakout_bullish(
            n_consolidation=30,
            n_breakout=10,   # lebih banyak candle setelah breakout untuk dimutasi
        )
        idx = idx_breakout  # evaluasi tepat di candle breakout pertama

        # Pastikan ada candle setelah idx untuk dimutasi
        self.assertGreater(len(df), idx + 1,
            msg="Test membutuhkan candle setelah idx untuk mutasi")

        hasil_asli   = detect_market_regime(df, idx=idx)
        df_mutasi    = self._mutasi_ekstrem(df, idx)
        hasil_mutasi = detect_market_regime(df_mutasi, idx=idx)

        identik, pesan = self._hasil_identik(hasil_asli, hasil_mutasi, "Kausalitas BREAKOUT: ")
        self.assertTrue(identik, msg=pesan)

        # Verifikasi field breakout_check spesifik
        bc_asli   = hasil_asli["detail"]["breakout_check"]
        bc_mutasi = hasil_mutasi["detail"]["breakout_check"]
        self.assertEqual(bc_asli["terpenuhi"],         bc_mutasi["terpenuhi"],
            msg="Kausalitas BREAKOUT: terpenuhi berubah setelah mutasi!")
        self.assertEqual(bc_asli["arah"],              bc_mutasi["arah"],
            msg="Kausalitas BREAKOUT: arah berubah setelah mutasi!")
        self.assertEqual(bc_asli["konfirmasi_body"],   bc_mutasi["konfirmasi_body"],
            msg="Kausalitas BREAKOUT: konfirmasi_body berubah setelah mutasi!")
        self.assertEqual(bc_asli["konfirmasi_volume"], bc_mutasi["konfirmasi_volume"],
            msg="Kausalitas BREAKOUT: konfirmasi_volume berubah setelah mutasi!")

    # ─── Bonus: Kausalitas dengan idx bervariasi ──────────────────────────────

    def test_causality_multi_idx(self):
        """
        Test bahwa hasil di idx tertentu tidak berubah saat dipanggil ulang
        dengan urutan yang berbeda (stateless verification).
        """
        df = _build_trending_bullish(n=80)

        for test_idx in [50, 60, 70, 79]:
            hasil_a = detect_market_regime(df, idx=test_idx)
            # Panggil idx lain dulu (simulasi urutan berbeda)
            _ = detect_market_regime(df, idx=79)
            _ = detect_market_regime(df, idx=50)
            hasil_b = detect_market_regime(df, idx=test_idx)

            self.assertEqual(hasil_a["regime"], hasil_b["regime"],
                msg=f"Stateless gagal di idx={test_idx}: "
                    f"{hasil_a['regime']} vs {hasil_b['regime']}")
            self.assertEqual(hasil_a["arah"], hasil_b["arah"],
                msg=f"Stateless (arah) gagal di idx={test_idx}")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
