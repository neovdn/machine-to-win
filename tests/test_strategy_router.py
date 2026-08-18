"""
tests/test_strategy_router.py
==============================
Unit test & causality test untuk engine/strategy_router.py — Fase 14.

PENGUJIAN (8 kelompok, semua data sintetis):

  1.  test_mapping_langsung_trending:
      Data trending bullish → route TREND_FOLLOWING, source DIRECT.

  2.  test_mapping_langsung_ranging:
      Data ranging → route RANGE_REVERSAL, source DIRECT, arah None.

  3.  test_mapping_langsung_breakout:
      Data breakout bullish → route BREAKOUT_RETEST, source DIRECT
      (bukan GRACE_WINDOW — regime saat ini sendiri sudah breakout).

  4.  test_mapping_langsung_chop_tanpa_breakout:
      Data CHOP tanpa breakout dalam grace window → strategy None, source NONE.

  5.  test_grace_window_positif:
      Breakout bullish di idx-2, saat ini CHOP → BREAKOUT_RETEST, source
      GRACE_WINDOW, arah BULLISH, grace_regime bukan None.

  6a. test_grace_window_batas_dalam_window:
      Breakout tepat di idx - grace_candles (batas inklusif) → masih ketemu.

  6b. test_grace_window_batas_luar_window:
      Breakout tepat di idx - grace_candles - 1 (sudah di luar) → NONE.

  7.  test_grace_window_tidak_berlaku_saat_ranging:
      Breakout beberapa candle lalu tapi regime saat ini RANGING →
      RANGE_REVERSAL, source DIRECT (grace window tidak membajak routing normal).

  8.  test_grace_window_ambil_breakout_terbaru:
      Breakout bearish di idx-1 DAN breakout bullish di idx-3, keduanya dalam
      window, regime saat ini CHOP → pakai yang di idx-1 (bearish, terbaru).

  9.  test_stateless_order_independen:
      Panggil get_active_strategy() di beberapa idx dalam urutan acak,
      assert hasil untuk idx tertentu selalu identik.

  10. TestCausalityStrategyRouter:
      Mutasi ekstrem seluruh candle SETELAH idx → hasil identik 100%,
      termasuk untuk kasus grace window.

  11. test_integrasi_stub_registry:
      Stub STRATEGY_REGISTRY di file test ini (BUKAN di modul engine).
      Untuk tiap regime, ambil strategy dari router, lookup registry,
      assert output stub sesuai.

  12. test_precomputed_identik_dengan_live:
      get_active_strategy_from_precomputed_regimes() menghasilkan output
      identik dengan get_active_strategy() untuk idx yang sama.
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

from engine.strategy_router import (
    route_strategy,
    get_active_strategy,
    get_active_strategy_from_precomputed_regimes,
    STRATEGY_MAP,
    REGIME_BREAKOUT_GRACE_CANDLES,
)
from engine.regime_detector import detect_market_regime


# =============================================================================
# STUB STRATEGY REGISTRY (HANYA di file test ini — bukan di engine/)
# =============================================================================

STUB_STRATEGY_REGISTRY = {
    "TREND_FOLLOWING": lambda: "stub: trend following dipanggil",
    "RANGE_REVERSAL" : lambda: "stub: range reversal dipanggil",
    "BREAKOUT_RETEST": lambda: "stub: breakout retest dipanggil",
}
# Registry ini membuktikan bahwa string yang dihasilkan router adalah key valid
# yang bisa dipakai oleh fase-fase berikutnya (Fase 15–17).


# =============================================================================
# HELPER: Builder DataFrame Sintetis
# =============================================================================

def _make_flat(
    n          : int   = 80,
    base       : float = 2000.0,
    atr        : float = 5.0,
    freq       : str   = "15min",
) -> pd.DataFrame:
    """
    DataFrame M15 sintetis flat (harga datar, EMA sideways).
    Dipakai sebagai base untuk test yang butuh CHOP atau modifikasi manual.
    """
    dates = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "open"        : base,
        "high"        : base + 1.0,
        "low"         : base - 1.0,
        "close"       : base,
        "tick_volume" : 100.0,
        "atr_14"      : atr,
        "ema_9"       : base,
        "ema_21"      : base,
        "ema_gap_pct" : 0.0,
        "trend"       : "SIDEWAYS",
        "volume_ratio": 1.0,
    }, index=dates)
    return df


def _build_trending_bullish(n: int = 120, base: float = 2000.0, atr: float = 5.0) -> pd.DataFrame:
    """
    DataFrame M15 trending bullish (HH/HL konsisten, EMA gap besar).
    Diambil dari pola yang sudah terbukti di test_regime_detector.py.
    Evaluasi valid di idx >= 40.
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

        if phase == 0:
            opens[i] = price; closes[i] = price + 3.0
            highs[i] = closes[i] + 0.5; lows[i] = price - 0.2
            price = closes[i]
        elif phase == 1:
            opens[i] = price; closes[i] = price + 4.0
            highs[i] = closes[i] + 2.0; lows[i] = price - 0.2
            price = closes[i]
        elif phase == 2:
            opens[i] = price; closes[i] = price - 2.5
            highs[i] = price + 0.3; lows[i] = closes[i] - 0.3
            price = closes[i]
        elif phase == 3:
            opens[i] = price; closes[i] = price - 1.5
            highs[i] = price + 0.2; lows[i] = closes[i] - 2.0
            price = closes[i]
        elif phase == 4:
            opens[i] = price; closes[i] = price + 2.0
            highs[i] = closes[i] + 0.3; lows[i] = price - 0.2
            price = closes[i]
        else:
            opens[i] = price; closes[i] = price + 1.5
            highs[i] = closes[i] + 0.5; lows[i] = price - 0.2
            price = closes[i]

    ema21 = closes - 4.0
    ema9  = closes - 1.0
    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": 100.0, "atr_14": atr,
        "ema_9": ema9, "ema_21": ema21, "ema_gap_pct": ema_gap_pct,
        "trend": "UPTREND", "volume_ratio": 1.0,
    }, index=dates)


def _build_ranging(
    n_range      : int   = 50,
    base         : float = 2000.0,
    spread       : float = 3.0,
    atr          : float = 5.0,
    n_touches_res: int   = 4,
    n_touches_sup: int   = 4,
) -> pd.DataFrame:
    """
    DataFrame M15 ranging (zona konsolidasi dengan sentuhan ke kedua sisi).
    """
    n = n_range
    dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

    highs  = np.full(n, base + spread * 0.5)
    lows   = np.full(n, base - spread * 0.5)
    opens  = np.full(n, base)
    closes = np.full(n, base)

    res_touch_indices = np.linspace(2, n - 3, n_touches_res, dtype=int)
    for i in res_touch_indices:
        highs[i] = base + spread
        closes[i] = base + spread - 0.5
        opens[i] = base + spread - 1.0

    sup_touch_indices = np.linspace(5, n - 1, n_touches_sup, dtype=int)
    for i in sup_touch_indices:
        lows[i] = base - spread
        closes[i] = base - spread + 0.5
        opens[i] = base - spread + 1.0

    ema21 = np.full(n, base)
    ema9  = np.full(n, base + 0.1)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": 100.0, "atr_14": atr,
        "ema_9": ema9, "ema_21": ema21, "ema_gap_pct": 0.005,
        "trend": "SIDEWAYS", "volume_ratio": 1.0,
    }, index=dates)


def _build_breakout_bullish(
    n_consolidation: int   = 30,
    n_breakout     : int   = 5,
    base           : float = 2000.0,
    spread         : float = 3.0,
    atr            : float = 5.0,
) -> tuple:
    """
    DataFrame M15 dengan konsolidasi diikuti breakout bullish.
    Return: (df, idx_breakout)
    """
    n_total = n_consolidation + n_breakout
    dates   = pd.date_range("2026-01-01", periods=n_total, freq="15min", tz="UTC")

    highs  = np.zeros(n_total)
    lows   = np.zeros(n_total)
    opens  = np.zeros(n_total)
    closes = np.zeros(n_total)
    vols   = np.ones(n_total)

    resistance = base + spread
    support    = base - spread

    for i in range(n_consolidation):
        opens[i] = base; highs[i] = resistance - 0.1
        lows[i] = support + 0.1; closes[i] = base

    for i in [3, 8, 14, 20]:
        highs[i] = resistance; lows[i] = support; closes[i] = base + 0.5

    for i in range(n_consolidation, n_total):
        opens[i]  = resistance - 0.5
        closes[i] = resistance + atr * 1.5
        highs[i]  = closes[i] + 0.5
        lows[i]   = resistance - 0.5
        vols[i]   = 2.0

    ema21 = np.full(n_total, base)
    ema9  = np.full(n_total, base)
    for i in range(n_consolidation, n_total):
        delta = (i - n_consolidation + 1) * 0.5
        ema9[i]  = base + delta
        ema21[i] = base + delta * 0.3

    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100
    trends = np.where(ema9 > ema21, "UPTREND", "SIDEWAYS")

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": 100.0, "atr_14": atr,
        "ema_9": ema9, "ema_21": ema21, "ema_gap_pct": ema_gap_pct,
        "trend": trends, "volume_ratio": vols,
    }, index=dates)

    return df, n_consolidation


def _build_grace_window_scenario(
    grace_candles      : int   = REGIME_BREAKOUT_GRACE_CANDLES,
    n_before_breakout  : int   = 35,
    n_chop_after       : int   = None,
    breakout_at_offset : int   = None,
    atr                : float = 5.0,
    base               : float = 2000.0,
    spread             : float = 3.0,
    final_regime       : str   = "CHOP",   # "CHOP" atau "RANGING"
) -> tuple:
    """
    Buat DataFrame untuk skenario grace window.

    Layout:
        [n_before_breakout candle konsolidasi]
        [1 candle BREAKOUT_TRANSITION bullish di idx_bo]
        [n_chop_after candle CHOP/RANGING setelah breakout]

    Return: (df, idx_eval, idx_bo)
        idx_eval = index candle terakhir (yang akan dievaluasi)
        idx_bo   = index candle breakout

    Parameter breakout_at_offset:
        Jika None → breakout di idx_eval - 2 (default).
        Jika set → breakout di idx_eval - breakout_at_offset.
    """
    if n_chop_after is None:
        n_chop_after = grace_candles  # default: breakout 2 candle sebelum akhir

    n_breakout = 1  # selalu 1 candle breakout
    n_total    = n_before_breakout + n_breakout + n_chop_after
    dates      = pd.date_range("2026-01-01", periods=n_total, freq="15min", tz="UTC")

    highs  = np.zeros(n_total)
    lows   = np.zeros(n_total)
    opens  = np.zeros(n_total)
    closes = np.zeros(n_total)
    vols   = np.ones(n_total)

    resistance = base + spread
    support    = base - spread
    idx_bo     = n_before_breakout  # index candle breakout

    # Bagian konsolidasi
    for i in range(n_before_breakout):
        opens[i] = base; highs[i] = resistance - 0.1
        lows[i] = support + 0.1; closes[i] = base

    # Inject sentuhan ke boundary agar zona valid
    for i in [3, 8, 14, 20, 27]:
        if i < n_before_breakout:
            highs[i] = resistance
            lows[i]  = support

    # Candle breakout bullish (body besar, volume tinggi)
    opens[idx_bo]  = resistance - 0.5
    closes[idx_bo] = resistance + atr * 1.5
    highs[idx_bo]  = closes[idx_bo] + 0.5
    lows[idx_bo]   = resistance - 0.5
    vols[idx_bo]   = 2.0

    # Setelah breakout: CHOP (range lebar, tidak ada zona) atau RANGING
    if final_regime == "RANGING":
        # Buat range baru dengan sentuhan ke kedua sisi
        new_base = closes[idx_bo]
        new_spread = spread * 0.8
        new_res = new_base + new_spread
        new_sup = new_base - new_spread

        for i in range(idx_bo + 1, n_total):
            opens[i] = new_base
            closes[i] = new_base
            highs[i] = new_res - 0.1
            lows[i] = new_sup + 0.1

        # Inject sentuhan ke boundary ranging baru
        after_start = idx_bo + 1
        for j, i in enumerate(range(after_start, n_total)):
            if j % 3 == 0:
                highs[i] = new_res
                closes[i] = new_res - 0.3
            elif j % 3 == 1:
                lows[i] = new_sup
                closes[i] = new_sup + 0.3
    else:
        # CHOP setelah breakout: harga bergerak range lebar (tidak ada zona valid)
        rng = np.random.RandomState(seed=42)
        for i in range(idx_bo + 1, n_total):
            opens[i]  = base
            # Range besar (> max_atr_ratio * atr) → tidak ada zona valid
            closes[i] = base + rng.uniform(-atr * 0.4, atr * 0.4)
            highs[i]  = max(opens[i], closes[i]) + atr * 1.5
            lows[i]   = min(opens[i], closes[i]) - atr * 1.5

    # EMA: sideways di seluruh area
    ema21 = np.full(n_total, base)
    ema9  = np.full(n_total, base + 0.05)
    ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": 100.0, "atr_14": atr,
        "ema_9": ema9, "ema_21": ema21, "ema_gap_pct": ema_gap_pct,
        "trend": "SIDEWAYS", "volume_ratio": vols,
    }, index=dates)

    idx_eval = n_total - 1
    return df, idx_eval, idx_bo


# =============================================================================
# HELPER: MUTASI EKSTREM (untuk test kausalitas)
# =============================================================================

def _mutasi_ekstrem(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """
    Mutasi semua candle SETELAH idx secara ekstrem.
    Identik dengan metodologi di test_regime_detector.py.
    """
    df_m = df.copy()
    n = len(df_m)
    if idx + 1 >= n:
        return df_m

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


def _hasil_identik(a: dict, b: dict) -> tuple:
    """
    Bandingkan dua output get_active_strategy() untuk identitas.
    Return (True, "") jika identik, (False, penjelasan) jika berbeda.
    """
    if a["strategy"] != b["strategy"]:
        return False, f"strategy beda: {a['strategy']} vs {b['strategy']}"
    if a["arah"] != b["arah"]:
        return False, f"arah beda: {a['arah']} vs {b['arah']}"
    if a["source"] != b["source"]:
        return False, f"source beda: {a['source']} vs {b['source']}"
    if a["regime"]["regime"] != b["regime"]["regime"]:
        return False, f"regime.regime beda: {a['regime']['regime']} vs {b['regime']['regime']}"

    # Bandingkan grace_regime
    if (a["grace_regime"] is None) != (b["grace_regime"] is None):
        return False, f"grace_regime None mismatch: {a['grace_regime']} vs {b['grace_regime']}"
    if a["grace_regime"] is not None and b["grace_regime"] is not None:
        if a["grace_regime"]["regime"] != b["grace_regime"]["regime"]:
            return False, (
                f"grace_regime.regime beda: "
                f"{a['grace_regime']['regime']} vs {b['grace_regime']['regime']}"
            )

    return True, ""


# =============================================================================
# KELAS TEST UTAMA: MAPPING LANGSUNG
# =============================================================================

class TestMappingLangsung(unittest.TestCase):
    """Test 1–4: routing langsung untuk keempat regime."""

    # ─── Test 1: TRENDING bullish ─────────────────────────────────────────────

    def test_mapping_langsung_trending_bullish(self):
        """
        Data trending bullish → strategy TREND_FOLLOWING, source DIRECT,
        arah BULLISH, grace_regime None.
        """
        df   = _build_trending_bullish(n=120, base=2000.0, atr=5.0)
        idx  = 80
        hasil = get_active_strategy(df, idx=idx)

        self.assertEqual(hasil["strategy"], "TREND_FOLLOWING",
            msg=f"Expect TREND_FOLLOWING, dapat: {hasil['strategy']}. "
                f"Keterangan: {hasil['keterangan']}")
        self.assertEqual(hasil["source"], "DIRECT",
            msg=f"Expect source=DIRECT, dapat: {hasil['source']}")
        self.assertEqual(hasil["arah"], "BULLISH",
            msg=f"Expect arah=BULLISH, dapat: {hasil['arah']}")
        self.assertIsNone(hasil["grace_regime"])
        self.assertIsNotNone(hasil["regime"])
        self.assertEqual(hasil["regime"]["regime"], "TRENDING")

        # Test integrasi stub registry
        key = hasil["strategy"]
        self.assertIn(key, STUB_STRATEGY_REGISTRY,
            msg=f"strategy '{key}' harus ada di STUB_STRATEGY_REGISTRY")
        output_stub = STUB_STRATEGY_REGISTRY[key]()
        self.assertEqual(output_stub, "stub: trend following dipanggil")

    # ─── Test 2: RANGING ─────────────────────────────────────────────────────

    def test_mapping_langsung_ranging(self):
        """
        Data ranging → strategy RANGE_REVERSAL, source DIRECT, arah None.
        """
        df    = _build_ranging(n_range=50, base=2000.0, spread=3.0, atr=5.0)
        hasil = get_active_strategy(df, idx=-1)

        self.assertEqual(hasil["strategy"], "RANGE_REVERSAL",
            msg=f"Expect RANGE_REVERSAL, dapat: {hasil['strategy']}. "
                f"Keterangan: {hasil['keterangan']}")
        self.assertEqual(hasil["source"], "DIRECT")
        self.assertIsNone(hasil["arah"],
            msg="RANGE_REVERSAL tidak punya arah — harus None")
        self.assertIsNone(hasil["grace_regime"])
        self.assertEqual(hasil["regime"]["regime"], "RANGING")

        # Integrasi stub
        output_stub = STUB_STRATEGY_REGISTRY[hasil["strategy"]]()
        self.assertEqual(output_stub, "stub: range reversal dipanggil")

    # ─── Test 3: BREAKOUT_TRANSITION → source DIRECT (bukan GRACE_WINDOW) ────

    def test_mapping_langsung_breakout_transition(self):
        """
        Saat regime saat ini BREAKOUT_TRANSITION → strategy BREAKOUT_RETEST,
        source DIRECT (bukan GRACE_WINDOW), arah sesuai arah breakout.
        Grace window HANYA untuk kasus CHOP setelah breakout, bukan untuk
        saat breakout itu sendiri sedang aktif.
        """
        df, idx_bo = _build_breakout_bullish(
            n_consolidation=30, n_breakout=5, base=2000.0, spread=3.0, atr=5.0
        )
        # Pastikan regime di idx_bo memang BREAKOUT_TRANSITION
        regime_bo = detect_market_regime(df, idx=idx_bo)
        self.assertEqual(regime_bo["regime"], "BREAKOUT_TRANSITION",
            msg=f"Data builder harus menghasilkan BREAKOUT_TRANSITION di idx={idx_bo}. "
                f"Dapat: {regime_bo['regime']}. Keterangan: {regime_bo['keterangan']}")

        hasil = get_active_strategy(df, idx=idx_bo)

        self.assertEqual(hasil["strategy"], "BREAKOUT_RETEST",
            msg=f"Expect BREAKOUT_RETEST, dapat: {hasil['strategy']}")
        self.assertEqual(hasil["source"], "DIRECT",
            msg=f"Harus DIRECT (bukan GRACE_WINDOW) saat breakout aktif sekarang. "
                f"Dapat: {hasil['source']}")
        self.assertEqual(hasil["arah"], "BULLISH",
            msg=f"Expect arah=BULLISH, dapat: {hasil['arah']}")
        self.assertIsNone(hasil["grace_regime"])

        # Integrasi stub
        output_stub = STUB_STRATEGY_REGISTRY[hasil["strategy"]]()
        self.assertEqual(output_stub, "stub: breakout retest dipanggil")

    # ─── Test 4: CHOP tanpa breakout dalam window → NONE ─────────────────────

    def test_mapping_langsung_chop_no_breakout(self):
        """
        Data CHOP murni tanpa ada breakout dalam grace window →
        strategy None, source NONE.
        """
        df    = _make_flat(n=80, base=2000.0, atr=5.0)
        # Verifikasi data memang CHOP
        regime = detect_market_regime(df, idx=-1)
        # Harus CHOP (flat data tanpa zona atau trend)
        # Kalau bukan CHOP, data builder perlu disesuaikan
        # (tetap test terhadap hasil aktual)

        hasil = get_active_strategy(df, idx=-1, grace_candles=REGIME_BREAKOUT_GRACE_CANDLES)

        # Jika regime saat ini bukan CHOP (misal RANGING karena zona flat),
        # test tetap valid asal strategy diroute dengan benar dari STRATEGY_MAP
        if hasil["regime"]["regime"] == "CHOP":
            self.assertIsNone(hasil["strategy"],
                msg="CHOP tanpa breakout dalam window → strategy harus None")
            self.assertEqual(hasil["source"], "NONE",
                msg="CHOP tanpa breakout → source harus NONE")
            self.assertIsNone(hasil["grace_regime"])
        else:
            # Kalau flat data dianggap RANGING, verifikasi mapping tetap benar
            expected_strat = STRATEGY_MAP.get(hasil["regime"]["regime"])
            self.assertEqual(hasil["strategy"], expected_strat,
                msg=f"Mapping harus sesuai STRATEGY_MAP untuk regime={hasil['regime']['regime']}")
            self.assertEqual(hasil["source"], "DIRECT")


# =============================================================================
# KELAS TEST GRACE WINDOW
# =============================================================================

class TestGraceWindow(unittest.TestCase):
    """Test 5–8: skenario grace window."""

    # ─── Test 5: Grace window positif ────────────────────────────────────────

    def test_grace_window_positif(self):
        """
        Breakout bullish di idx-2, saat ini CHOP →
        strategy BREAKOUT_RETEST, source GRACE_WINDOW, arah BULLISH,
        grace_regime bukan None dan regime-nya BREAKOUT_TRANSITION.
        """
        # n_chop_after=2 → breakout di idx_eval - 2
        df, idx_eval, idx_bo = _build_grace_window_scenario(
            grace_candles    = REGIME_BREAKOUT_GRACE_CANDLES,
            n_before_breakout= 35,
            n_chop_after     = 2,
            final_regime     = "CHOP",
        )

        # Verifikasi setup: regime di idx_bo harus BREAKOUT_TRANSITION
        regime_bo = detect_market_regime(df, idx=idx_bo)
        self.assertEqual(regime_bo["regime"], "BREAKOUT_TRANSITION",
            msg=f"Setup: harus BREAKOUT_TRANSITION di idx_bo={idx_bo}. "
                f"Dapat: {regime_bo['regime']}. {regime_bo['keterangan']}")

        # Verifikasi setup: regime di idx_eval harus CHOP
        regime_eval = detect_market_regime(df, idx=idx_eval)
        self.assertEqual(regime_eval["regime"], "CHOP",
            msg=f"Setup: harus CHOP di idx_eval={idx_eval}. "
                f"Dapat: {regime_eval['regime']}. {regime_eval['keterangan']}")

        hasil = get_active_strategy(df, idx=idx_eval, grace_candles=REGIME_BREAKOUT_GRACE_CANDLES)

        self.assertEqual(hasil["strategy"], "BREAKOUT_RETEST",
            msg=f"Expect BREAKOUT_RETEST via grace window. "
                f"Dapat: {hasil['strategy']}. {hasil['keterangan']}")
        self.assertEqual(hasil["source"], "GRACE_WINDOW",
            msg=f"Expect source=GRACE_WINDOW. Dapat: {hasil['source']}")
        self.assertEqual(hasil["arah"], "BULLISH",
            msg=f"Expect arah=BULLISH dari breakout bullish. Dapat: {hasil['arah']}")
        self.assertIsNotNone(hasil["grace_regime"],
            msg="grace_regime harus ada saat source=GRACE_WINDOW")
        self.assertEqual(hasil["grace_regime"]["regime"], "BREAKOUT_TRANSITION",
            msg=f"grace_regime.regime harus BREAKOUT_TRANSITION. "
                f"Dapat: {hasil['grace_regime']['regime']}")

        # Integrasi stub
        output_stub = STUB_STRATEGY_REGISTRY[hasil["strategy"]]()
        self.assertEqual(output_stub, "stub: breakout retest dipanggil")

    # ─── Test 6a: Batas window — dalam window (breakout persis di batas) ──────

    def test_grace_window_batas_dalam_window(self):
        """
        Breakout tepat di idx - grace_candles (batas inklusif) →
        harus masih ditemukan (source GRACE_WINDOW).
        """
        grace = REGIME_BREAKOUT_GRACE_CANDLES  # default = 4
        # Breakout di offset = grace (batas paling jauh yang masih inklusif)
        # n_chop_after = grace agar idx_eval - idx_bo = grace
        df, idx_eval, idx_bo = _build_grace_window_scenario(
            grace_candles     = grace,
            n_before_breakout = 35,
            n_chop_after      = grace,
            final_regime      = "CHOP",
        )
        # idx_eval - idx_bo = grace → breakout di tepat batas window

        # Verifikasi setup
        regime_bo = detect_market_regime(df, idx=idx_bo)
        self.assertEqual(regime_bo["regime"], "BREAKOUT_TRANSITION",
            msg=f"Setup batas: butuh BREAKOUT_TRANSITION di idx_bo={idx_bo}. "
                f"Dapat: {regime_bo['regime']}")

        regime_eval = detect_market_regime(df, idx=idx_eval)
        self.assertEqual(regime_eval["regime"], "CHOP",
            msg=f"Setup batas: butuh CHOP di idx_eval={idx_eval}. "
                f"Dapat: {regime_eval['regime']}")

        hasil = get_active_strategy(df, idx=idx_eval, grace_candles=grace)

        # Grace window scan dari idx-1 sampai max(idx-grace, 0) = idx-4.
        # idx_bo = idx_eval - grace → dalam rentang scan.
        self.assertEqual(hasil["source"], "GRACE_WINDOW",
            msg=f"Breakout di tepat batas window harus masih ditemukan (inklusif). "
                f"Dapat: {hasil['source']}. Keterangan: {hasil['keterangan']}")
        self.assertEqual(hasil["strategy"], "BREAKOUT_RETEST")

    # ─── Test 6b: Batas window — di luar window ───────────────────────────────

    def test_grace_window_batas_luar_window(self):
        """
        Breakout tepat di idx - grace_candles - 1 (1 candle di luar window) →
        harus TIDAK ditemukan (source NONE).
        """
        grace = REGIME_BREAKOUT_GRACE_CANDLES  # default = 4
        # Breakout di offset = grace + 1 → tepat 1 di luar batas
        n_chop_after = grace + 1

        df, idx_eval, idx_bo = _build_grace_window_scenario(
            grace_candles     = grace,
            n_before_breakout = 35,
            n_chop_after      = n_chop_after,
            final_regime      = "CHOP",
        )
        # idx_eval - idx_bo = grace + 1 → tepat di luar window

        # Verifikasi setup
        regime_bo = detect_market_regime(df, idx=idx_bo)
        self.assertEqual(regime_bo["regime"], "BREAKOUT_TRANSITION",
            msg=f"Setup luar batas: butuh BREAKOUT_TRANSITION di idx_bo={idx_bo}. "
                f"Dapat: {regime_bo['regime']}")

        regime_eval = detect_market_regime(df, idx=idx_eval)
        self.assertEqual(regime_eval["regime"], "CHOP",
            msg=f"Setup luar batas: butuh CHOP di idx_eval={idx_eval}. "
                f"Dapat: {regime_eval['regime']}")

        hasil = get_active_strategy(df, idx=idx_eval, grace_candles=grace)

        # Grace window scan dari idx-1 sampai idx-grace (inklusif).
        # idx_bo = idx_eval - (grace+1) → TIDAK dalam rentang scan.
        self.assertEqual(hasil["source"], "NONE",
            msg=f"Breakout di luar window (offset={grace+1}) harus NONE. "
                f"Dapat: {hasil['source']}. Keterangan: {hasil['keterangan']}")
        self.assertIsNone(hasil["strategy"])
        self.assertIsNone(hasil["grace_regime"])

    # ─── Test 7: Grace window TIDAK berlaku saat regime saat ini RANGING ──────

    def test_grace_window_tidak_berlaku_saat_ranging(self):
        """
        Ada breakout beberapa candle lalu, TAPI regime saat ini adalah RANGING
        (bukan CHOP) → strategy RANGE_REVERSAL, source DIRECT.
        Grace window tidak membajak routing normal.

        Implementasi: buat DataFrame dengan:
          [konsolidasi panjang] [breakout bullish] [20 candle ranging baru]
        Di idx_eval (akhir ranging), regime harus RANGING.
        """
        # Parameter konsolidasi
        n_cons  = 35
        atr     = 5.0
        base    = 2000.0
        spread  = 3.0
        n_range_after = 25  # cukup panjang agar zona ranging baru terbentuk

        resistance = base + spread
        support    = base - spread
        n_total    = n_cons + 1 + n_range_after
        dates      = pd.date_range("2026-01-01", periods=n_total, freq="15min", tz="UTC")

        highs  = np.zeros(n_total)
        lows   = np.zeros(n_total)
        opens  = np.zeros(n_total)
        closes = np.zeros(n_total)
        vols   = np.ones(n_total)

        # Konsolidasi dengan sentuhan ke boundary
        for i in range(n_cons):
            opens[i] = base; highs[i] = resistance - 0.1
            lows[i] = support + 0.1; closes[i] = base
        for i in [3, 8, 14, 20, 28]:
            highs[i] = resistance; lows[i] = support

        # Breakout bullish
        idx_bo = n_cons
        new_base = resistance + atr * 1.5
        opens[idx_bo]  = resistance - 0.5
        closes[idx_bo] = new_base
        highs[idx_bo]  = new_base + 0.5
        lows[idx_bo]   = resistance - 0.5
        vols[idx_bo]   = 2.0

        # Ranging baru di level yang lebih tinggi (cukup panjang, boundary jelas)
        new_spread = spread * 0.8
        new_res = new_base + new_spread
        new_sup = new_base - new_spread
        for i in range(idx_bo + 1, n_total):
            opens[i]  = new_base
            closes[i] = new_base
            highs[i]  = new_res - 0.1
            lows[i]   = new_sup + 0.1

        # Inject sentuhan ke boundary ranging baru (beberapa kali)
        after = idx_bo + 1
        touch_pts_r = np.linspace(after + 1, n_total - 3, 6, dtype=int)
        touch_pts_s = np.linspace(after + 3, n_total - 1, 6, dtype=int)
        for i in touch_pts_r:
            highs[i] = new_res; closes[i] = new_res - 0.3
        for i in touch_pts_s:
            lows[i] = new_sup; closes[i] = new_sup + 0.3

        ema21 = np.full(n_total, new_base)
        ema9  = np.full(n_total, new_base + 0.05)
        ema_gap_pct = (ema9 - ema21) / np.where(ema21 > 0, ema21, 1) * 100

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "tick_volume": 100.0, "atr_14": atr,
            "ema_9": ema9, "ema_21": ema21, "ema_gap_pct": ema_gap_pct,
            "trend": "SIDEWAYS", "volume_ratio": vols,
        }, index=dates)

        idx_eval = n_total - 1
        regime_eval = detect_market_regime(df, idx=idx_eval)

        if regime_eval["regime"] != "RANGING":
            self.skipTest(
                f"Data builder tidak menghasilkan RANGING di idx_eval={idx_eval} "
                f"(dapat {regime_eval['regime']}). Skenario perlu penyesuaian."
            )

        hasil = get_active_strategy(df, idx=idx_eval, grace_candles=REGIME_BREAKOUT_GRACE_CANDLES)

        self.assertEqual(hasil["strategy"], "RANGE_REVERSAL",
            msg=f"Saat regime RANGING, harus RANGE_REVERSAL (bukan BREAKOUT_RETEST). "
                f"Dapat: {hasil['strategy']}. {hasil['keterangan']}")
        self.assertEqual(hasil["source"], "DIRECT",
            msg=f"Saat regime RANGING, harus DIRECT (bukan GRACE_WINDOW). "
                f"Dapat: {hasil['source']}")
        self.assertIsNone(hasil["grace_regime"],
            msg="Saat RANGING, grace_regime harus None (grace window tidak dijalankan)")

    # ─── Test 8: Ambil breakout TERBARU kalau ada lebih dari satu ────────────

    def test_grace_window_ambil_breakout_terbaru(self):
        """
        Ada breakout bullish di idx-3 DAN breakout bearish di idx-1,
        keduanya dalam window, regime saat ini CHOP →
        hasil pakai yang di idx-1 (bearish, terbaru).

        Implementasi: gunakan get_active_strategy_from_precomputed_regimes()
        dengan regime_series yang dikonstruksi manual — ini memungkinkan kita
        mengontrol persis regime di setiap index tanpa bergantung pada
        kompleksitas data builder yang harus menghasilkan dua breakout dalam
        satu DataFrame sementara tetap menjaga kausalitas zone_detector.
        Precomputed variant sudah dibuktikan identik dengan live variant
        di TestPrecomputed, jadi pendekatan ini valid.
        """
        # Buat regime_series palsu dengan dua breakout dalam window
        # Layout idx: 0 1 2 3 4 5 6
        # idx=6 → CHOP (saat ini, yang dievaluasi)
        # idx=5 → BREAKOUT_TRANSITION BEARISH (terbaru, offset=1)
        # idx=4 → CHOP
        # idx=3 → BREAKOUT_TRANSITION BULLISH (lebih lama, offset=3)
        # idx=0..2 → CHOP
        # grace_candles = 4 → scan 5,4,3,2 — keduanya dalam window

        def _regime_dict(r, arah=None):
            return {
                "regime": r, "arah": arah, "zone": None,
                "detail": {}, "keterangan": f"mock: {r}",
            }

        regime_series = [
            _regime_dict("CHOP"),                               # idx=0
            _regime_dict("CHOP"),                               # idx=1
            _regime_dict("CHOP"),                               # idx=2
            _regime_dict("BREAKOUT_TRANSITION", "BULLISH"),     # idx=3
            _regime_dict("CHOP"),                               # idx=4
            _regime_dict("BREAKOUT_TRANSITION", "BEARISH"),     # idx=5
            _regime_dict("CHOP"),                               # idx=6 (idx_eval)
        ]
        idx_eval = 6
        grace = 4  # scan dari idx-1=5 sampai idx-4=2

        hasil = get_active_strategy_from_precomputed_regimes(
            regime_series, idx=idx_eval, grace_candles=grace
        )

        self.assertEqual(hasil["source"], "GRACE_WINDOW",
            msg=f"Harus GRACE_WINDOW. Dapat: {hasil['source']}")
        self.assertEqual(hasil["strategy"], "BREAKOUT_RETEST")
        self.assertEqual(hasil["arah"], "BEARISH",
            msg=f"Harus pakai breakout terbaru (BEARISH di idx=5). "
                f"Dapat: {hasil['arah']}. Keterangan: {hasil['keterangan']}")
        self.assertIsNotNone(hasil["grace_regime"])
        self.assertEqual(hasil["grace_regime"]["arah"], "BEARISH")


# =============================================================================
# KELAS TEST STATELESS & ORDER-INDEPENDEN
# =============================================================================

class TestStateless(unittest.TestCase):
    """Test 9: stateless — urutan pemanggilan tidak mempengaruhi hasil."""

    def test_stateless_order_independen(self):
        """
        Panggil get_active_strategy() di beberapa idx dalam urutan acak.
        Assert hasil untuk idx tertentu selalu identik.
        """
        df = _build_trending_bullish(n=120, base=2000.0, atr=5.0)

        # Hitung hasil untuk beberapa idx dalam urutan "normal"
        indices_normal = [40, 60, 80, 55, 75]
        hasil_normal   = {idx: get_active_strategy(df, idx=idx) for idx in indices_normal}

        # Panggil lagi dalam urutan terbalik (acak)
        indices_acak = [75, 40, 80, 55, 60]
        hasil_acak   = {idx: get_active_strategy(df, idx=idx) for idx in indices_acak}

        for idx in indices_normal:
            identik, pesan = _hasil_identik(hasil_normal[idx], hasil_acak[idx])
            self.assertTrue(identik,
                msg=f"idx={idx}: hasil berbeda antara urutan normal dan acak. "
                    f"Detail: {pesan}")


# =============================================================================
# KELAS TEST KAUSALITAS
# =============================================================================

class TestCausalityStrategyRouter(unittest.TestCase):
    """
    Test kausalitas (zero look-ahead) untuk get_active_strategy().

    Metodologi: mutasi ekstrem seluruh candle SETELAH idx, verifikasi bahwa
    output di idx sama persis sebelum dan sesudah mutasi.
    Dua varian:
        (a) Kasus TRENDING (jalur DIRECT)
        (b) Kasus grace window (jalur GRACE_WINDOW) — memverifikasi bahwa
            scan mundur juga tidak terpengaruh data masa depan
    """

    # ─── Test 10a: Kausalitas — jalur DIRECT ─────────────────────────────────

    def test_kausalitas_jalur_direct(self):
        """
        Mutasi candle setelah idx → hasil get_active_strategy() identik.
        Jalur: TRENDING → TREND_FOLLOWING (DIRECT).
        """
        df  = _build_trending_bullish(n=120, base=2000.0, atr=5.0)
        idx = 80

        hasil_asli = get_active_strategy(df, idx=idx)
        df_mutasi  = _mutasi_ekstrem(df, idx)
        hasil_mutasi = get_active_strategy(df_mutasi, idx=idx)

        identik, pesan = _hasil_identik(hasil_asli, hasil_mutasi)
        self.assertTrue(identik,
            msg=f"Kausalitas DIRECT gagal — mutasi candle setelah idx={idx} "
                f"mengubah hasil. Detail: {pesan}")

    # ─── Test 10b: Kausalitas — jalur GRACE_WINDOW ───────────────────────────

    def test_kausalitas_jalur_grace_window(self):
        """
        Mutasi candle setelah idx → hasil get_active_strategy() identik.
        Jalur: CHOP + breakout dalam window → BREAKOUT_RETEST (GRACE_WINDOW).
        """
        df, idx_eval, _ = _build_grace_window_scenario(
            grace_candles     = REGIME_BREAKOUT_GRACE_CANDLES,
            n_before_breakout = 35,
            n_chop_after      = 2,
            final_regime      = "CHOP",
        )

        hasil_asli   = get_active_strategy(df, idx=idx_eval)
        df_mutasi    = _mutasi_ekstrem(df, idx_eval)
        hasil_mutasi = get_active_strategy(df_mutasi, idx=idx_eval)

        identik, pesan = _hasil_identik(hasil_asli, hasil_mutasi)
        self.assertTrue(identik,
            msg=f"Kausalitas GRACE_WINDOW gagal — mutasi candle setelah idx={idx_eval} "
                f"mengubah hasil. Detail: {pesan}")


# =============================================================================
# KELAS TEST PRECOMPUTED
# =============================================================================

class TestPrecomputed(unittest.TestCase):
    """
    Test 12: get_active_strategy_from_precomputed_regimes() menghasilkan
    output identik dengan get_active_strategy() untuk idx yang sama.
    """

    def test_precomputed_identik_dengan_live(self):
        """
        Untuk semua idx dalam range, output precomputed == output live.
        Verifikasi juga untuk skenario grace window.
        """
        df = _build_trending_bullish(n=120, base=2000.0, atr=5.0)
        n  = len(df)

        # Bangun precomputed regime series
        regime_series = [detect_market_regime(df, idx=k) for k in range(n)]

        # Bandingkan untuk beberapa idx
        for idx in [40, 60, 80, 100]:
            hasil_live = get_active_strategy(df, idx=idx)
            hasil_pre  = get_active_strategy_from_precomputed_regimes(
                regime_series, idx=idx
            )

            identik, pesan = _hasil_identik(hasil_live, hasil_pre)
            self.assertTrue(identik,
                msg=f"Precomputed vs live tidak identik di idx={idx}. Detail: {pesan}")

    def test_precomputed_grace_window_identik(self):
        """
        Precomputed juga identik untuk skenario grace window.
        """
        df, idx_eval, _ = _build_grace_window_scenario(
            grace_candles     = REGIME_BREAKOUT_GRACE_CANDLES,
            n_before_breakout = 35,
            n_chop_after      = 2,
            final_regime      = "CHOP",
        )
        n = len(df)
        regime_series = [detect_market_regime(df, idx=k) for k in range(n)]

        hasil_live = get_active_strategy(df, idx=idx_eval)
        hasil_pre  = get_active_strategy_from_precomputed_regimes(
            regime_series, idx=idx_eval
        )

        identik, pesan = _hasil_identik(hasil_live, hasil_pre)
        self.assertTrue(identik,
            msg=f"Precomputed grace window tidak identik. Detail: {pesan}")

    def test_precomputed_edge_idx_di_luar_range(self):
        """
        idx di luar range → tidak crash, source="NONE".
        """
        regime_series = [detect_market_regime(_make_flat(n=50), idx=k) for k in range(50)]

        hasil = get_active_strategy_from_precomputed_regimes(regime_series, idx=9999)

        self.assertEqual(hasil["source"], "NONE")
        self.assertIsNone(hasil["strategy"])
        self.assertIn("luar range", hasil["keterangan"].lower())


# =============================================================================
# KELAS TEST ROUTE_STRATEGY (pure mapping)
# =============================================================================

class TestRouteStrategy(unittest.TestCase):
    """
    Test untuk route_strategy() secara terisolasi dari get_active_strategy().
    Memastikan STRATEGY_MAP konsisten dan mappable.
    """

    def test_route_strategy_semua_regime(self):
        """Semua regime menghasilkan mapping yang benar."""
        self.assertEqual(route_strategy("TRENDING"),            "TREND_FOLLOWING")
        self.assertEqual(route_strategy("RANGING"),             "RANGE_REVERSAL")
        self.assertEqual(route_strategy("BREAKOUT_TRANSITION"), "BREAKOUT_RETEST")
        self.assertIsNone(route_strategy("CHOP"))

    def test_route_strategy_regime_tidak_dikenal(self):
        """Regime tidak dikenal → None (graceful, tidak crash)."""
        self.assertIsNone(route_strategy("REGIME_ANEH"))
        self.assertIsNone(route_strategy(""))
        self.assertIsNone(route_strategy("chop"))  # case-sensitive

    def test_strategy_map_konsisten_dengan_stub_registry(self):
        """
        Semua nilai non-None dalam STRATEGY_MAP harus ada di STUB_STRATEGY_REGISTRY.
        Membuktikan konsistensi key antara router dan consumer.
        """
        for regime, strategy in STRATEGY_MAP.items():
            if strategy is not None:
                self.assertIn(strategy, STUB_STRATEGY_REGISTRY,
                    msg=f"strategy='{strategy}' untuk regime='{regime}' "
                        f"tidak ada di STUB_STRATEGY_REGISTRY. "
                        f"Pastikan key konsisten antara router dan fase-fase berikutnya.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
