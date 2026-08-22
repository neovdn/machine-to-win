"""
engine/backtester_regime.py
============================
Orkestrator Backtest Regime-Based — Fase 21.

TUJUAN:
    Menghubungkan SELURUH komponen Fase 12–19 ke dalam satu loop backtest
    end-to-end, lalu memecah hasilnya per regime, per strategi, dan per sesi
    untuk pelaporan tersegmentasi. Ini adalah "integrasi besar" pertama di
    proyek ini — semua komponen sebelumnya berdiri sendiri sampai fase ini.

KARAKTER MODUL (ORKESTRATOR):
    - Tidak mendefinisikan ulang logika yang sudah ada di modul lain.
    - Memanggil fungsi-fungsi dari Fase 12–19 secara langsung (reuse maksimal).
    - Tidak mengubah engine/backtester.py atau modul manapun dari Fase 12–19.
    - Hanya menambahkan lapisan orkestrasi dan pelaporan di atas komponen yang ada.

CAUSALITY (ZERO LOOK-AHEAD):
    Dijamin lewat dua mekanisme:
    1. merge_asof(direction="backward") untuk kedua merge timeframe (M15→M5 dan H1→M5):
       setiap candle M5 di waktu t hanya mendapat regime/context dari candle M15/H1
       yang sudah CLOSED sebelum atau tepat pada t.
    2. Loop backtest membaca dari df_merged (sudah di-merge) — tidak ada akses
       ke candle masa depan dalam bentuk apapun.
    Kausalitas dibuktikan via test mutasi ekstrem di tests/test_backtester_regime.py,
    bukan hanya diklaim.

FASE YANG DI-REUSE:
    - Fase 12 (market_context)    : get_h1_context()
    - Fase 13 (regime_detector)   : detect_market_regime()
    - Fase 14 (strategy_router)   : get_active_strategy_from_precomputed_regimes()
    - Fase 15 (range_reversal)    : evaluate_range_reversal()
    - Fase 16 (breakout_retest)   : evaluate_breakout_retest()
    - Fase 17 (trend_following)   : evaluate_trend_following()
    - Fase 18 (location_filter)   : calculate_confluence_summary()
    - Fase 19 (risk_manager_regime): calculate_regime_sl_tp()
    - Backtester lama             : simulate_trade_outcome(), compute_summary(),
                                    WARM_UP_CANDLES, MAX_FORWARD_CANDLES,
                                    DEFAULT_SPREAD_PTS, MIN_SL_DISTANCE

CATATAN DUPLIKASI PNL FORMULA (DISENGAJA):
    Formula PNL (spread cost, klem MTM untuk NO_HIT) di modul ini MENIRU PERSIS
    formula yang ada di run_backtest() di engine/backtester.py — ditulis ulang
    sebagai kode baru di sini. Ini BUKAN kesalahan arsitektural — ini disengaja
    agar engine/backtester.py (yang sudah selesai dan teruji) tidak dimodifikasi
    sama sekali. Duplikasi kecil ini sah dan terdokumentasi.

FASE 20 (NEWS FILTER):
    Di-skip atas keputusan pemilik proyek. Tidak ada logika news filter di modul ini.

LOGIKA MURNI IF-ELSE — TIDAK ADA AI / MACHINE LEARNING.
"""

import numpy as np
import pandas as pd
from typing import Optional

from engine.indicators import run_all_indicators
from engine.session_filter import is_high_liquidity_session

# Reuse dari backtester lama (tidak memanggil run_backtest() — hanya helpers)
from engine.backtester import (
    simulate_trade_outcome,
    compute_summary,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    DEFAULT_SPREAD_PTS,
    MIN_SL_DISTANCE,
)

# Fase 12 — H1 Context
from engine.market_context import get_h1_context

# Fase 13 — Regime Detector M15
from engine.regime_detector import detect_market_regime

# Fase 14 — Strategy Router
from engine.strategy_router import (
    get_active_strategy_from_precomputed_regimes,
    REGIME_BREAKOUT_GRACE_CANDLES,
)

# Fase 15–17 — Strategi
from engine.strategies.range_reversal import evaluate_range_reversal
from engine.strategies.breakout_retest_v2 import evaluate_breakout_retest
from engine.strategies.trend_following_v2 import evaluate_trend_following

# Fase 18 — Location Filter (confluence scoring — TIDAK PERNAH membatalkan trade)
from engine.location_filter import calculate_confluence_summary

# Fase 19 — Regime Risk Manager
from engine.risk_manager_regime import calculate_regime_sl_tp


# =============================================================================
# FUNGSI 1: MERGE REGIME M15 → TIMELINE M5
# =============================================================================

def merge_regime_to_m5(
    df_m5_ind    : pd.DataFrame,
    df_m15_ind   : pd.DataFrame,
    grace_candles: int = REGIME_BREAKOUT_GRACE_CANDLES,
) -> pd.DataFrame:
    """
    Attach regime M15 dan keputusan strategi ke setiap baris M5 via merge_asof backward.

    ANTI-LOOKAHEAD (direction="backward"):
        Setiap candle M5 di waktu t mendapat regime dari candle M15 terakhir
        yang SUDAH CLOSED sebelum atau tepat pada t:
            m15_attached = max(m15.time) WHERE m15.time <= m5.time

        Contoh:
            M5 candle 13:05 → attach M15 candle 13:00 (bukan 13:15)
            M5 candle 13:14 → attach M15 candle 13:00 (bukan 13:15)
            M5 candle 13:15 → attach M15 candle 13:15

        M15 candle masa depan TIDAK PERNAH bocor ke data M5.

    PRECOMPUTE (O(N) — satu kali untuk seluruh dataset):
        1. detect_market_regime() dipanggil SEKALI untuk setiap index M15 → regime_series
        2. get_active_strategy_from_precomputed_regimes() dipanggil SEKALI per M15 → strategy_series
        Hasilnya di-merge ke M5 via merge_asof, bukan per-candle M5 secara langsung.

    Parameter:
        df_m5_ind     : DataFrame M5 yang sudah melewati run_all_indicators()
        df_m15_ind    : DataFrame M15 yang sudah melewati run_all_indicators()
        grace_candles : Jumlah candle M15 ke belakang untuk grace window breakout.
                        Default = REGIME_BREAKOUT_GRACE_CANDLES dari strategy_router.

    Return:
        DataFrame M5 dengan kolom tambahan:
            m15_regime   : str    — label regime ("TRENDING"/"RANGING"/"BREAKOUT_TRANSITION"/"CHOP")
            m15_strategy : str|None — nama strategi atau None
            m15_arah     : str|None — "BULLISH"/"BEARISH" atau None
            m15_source   : str    — "DIRECT"/"GRACE_WINDOW"/"NONE"
            m15_zone     : object — dict mentah zone dari detect_market_regime(), boleh None
    """
    n_m15 = len(df_m15_ind)

    # ── Langkah 1: Precompute regime untuk SETIAP index M15 (O(N)) ─────────────
    # CAUSALITY: detect_market_regime(df_m15_ind, i) hanya membaca [:i+1]
    if n_m15 == 0:
        # Edge case: data M15 kosong — kembalikan M5 dengan kolom default None
        df_out = df_m5_ind.copy()
        df_out["m15_regime"]   = "CHOP"
        df_out["m15_strategy"] = None
        df_out["m15_arah"]     = None
        df_out["m15_source"]   = "NONE"
        df_out["m15_zone"]     = None
        return df_out

    regime_series = [
        detect_market_regime(df_m15_ind, i)
        for i in range(n_m15)
    ]

    # ── Langkah 2: Precompute strategy untuk SETIAP index M15 (O(N)) ───────────
    # Menggunakan list regime yang sudah dihitung — tidak memanggil detect_market_regime() lagi
    strategy_series = [
        get_active_strategy_from_precomputed_regimes(regime_series, i, grace_candles)
        for i in range(n_m15)
    ]

    # ── Langkah 3: Susun DataFrame M15 auxiliary ────────────────────────────────
    m15_aux_rows = []
    for i in range(n_m15):
        reg  = regime_series[i]
        strat = strategy_series[i]
        m15_aux_rows.append({
            "time"        : df_m15_ind.index[i],
            "m15_regime"  : reg.get("regime", "CHOP"),
            "m15_strategy": strat.get("strategy"),
            "m15_arah"    : strat.get("arah"),
            "m15_source"  : strat.get("source", "NONE"),
            "m15_zone"    : reg.get("zone"),   # dict mentah, bisa None
        })

    df_m15_aux = pd.DataFrame(m15_aux_rows)

    # ── Langkah 4: merge_asof backward — anti-lookahead ────────────────────────
    m5_reset = df_m5_ind.reset_index()   # 'time' jadi kolom reguler
    m5_reset = m5_reset.sort_values("time").reset_index(drop=True)
    df_m15_aux = df_m15_aux.sort_values("time").reset_index(drop=True)

    merged = pd.merge_asof(
        m5_reset,
        df_m15_aux,
        on        = "time",
        direction = "backward",   # WAJIB: anti-lookahead
    )

    # Kembalikan ke DatetimeIndex
    merged = merged.set_index("time")
    return merged


# =============================================================================
# FUNGSI 2: MERGE H1 CONTEXT → TIMELINE M5
# =============================================================================

def merge_h1_context_to_m5(
    df_m5_ind : pd.DataFrame,
    df_h1_ind : pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach H1 context ke setiap baris M5 via merge_asof backward.

    ANTI-LOOKAHEAD (direction="backward"):
        Sama persis dengan merge_regime_to_m5() — setiap candle M5 di waktu t
        hanya mendapat H1 context dari candle H1 yang SUDAH CLOSED sebelum/pada t.

    PRECOMPUTE (O(N)):
        get_h1_context() dipanggil SEKALI untuk setiap index H1 → h1_context_series
        Dict context penuh disimpan di kolom 'h1_context' (object dtype) di setiap
        baris M5 — bukan exploded jadi kolom terpisah, agar tidak menyulitkan
        pembacaan kembali per candle.

    MENGAPA BUKAN REUSE merge_h1_to_m5() LAMA:
        merge_h1_to_m5() di engine/backtester.py menghasilkan kolom 'trend_h1'
        berbasis detect_bias_h1() untuk sistem single-strategy lama. Format itu
        beda struktur dari get_h1_context() Fase 12 yang punya field
        bias/strength/strength_zone. Fungsi ini khusus untuk H1 context Fase 12.

    Parameter:
        df_m5_ind : DataFrame M5 yang sudah melewati run_all_indicators()
                    (boleh sudah punya kolom dari merge_regime_to_m5())
        df_h1_ind : DataFrame H1 yang sudah melewati run_all_indicators()

    Return:
        DataFrame M5 dengan kolom tambahan:
            h1_context : object — dict penuh dari get_h1_context() (bias, strength,
                         strength_zone, ema_gap_pct, close, time, keterangan)
    """
    n_h1 = len(df_h1_ind)

    # Edge case: data H1 kosong
    if n_h1 == 0:
        df_out = df_m5_ind.copy()
        df_out["h1_context"] = None
        return df_out

    # ── Langkah 1: Precompute H1 context untuk SETIAP index H1 (O(N)) ──────────
    # CAUSALITY: get_h1_context(df_h1_ind, i) hanya membaca [:i+1]
    h1_context_series = [
        get_h1_context(df_h1_ind, i)
        for i in range(n_h1)
    ]

    # ── Langkah 2: Susun DataFrame H1 auxiliary ─────────────────────────────────
    h1_aux_rows = []
    for i in range(n_h1):
        h1_aux_rows.append({
            "time"      : df_h1_ind.index[i],
            "h1_context": h1_context_series[i],   # dict penuh
        })

    df_h1_aux = pd.DataFrame(h1_aux_rows)

    # ── Langkah 3: merge_asof backward — anti-lookahead ────────────────────────
    # df_m5_ind bisa sudah punya DatetimeIndex atau sudah dalam bentuk reset
    if "time" in df_m5_ind.columns:
        # Sudah reset (bukan kasus normal, tapi bersifat defensif)
        m5_reset = df_m5_ind.copy().sort_values("time").reset_index(drop=True)
        already_reset = True
    else:
        m5_reset = df_m5_ind.reset_index()
        m5_reset = m5_reset.sort_values("time").reset_index(drop=True)
        already_reset = False

    df_h1_aux = df_h1_aux.sort_values("time").reset_index(drop=True)

    merged = pd.merge_asof(
        m5_reset,
        df_h1_aux,
        on        = "time",
        direction = "backward",   # WAJIB: anti-lookahead
    )

    # Kembalikan ke DatetimeIndex
    merged = merged.set_index("time")
    return merged


# =============================================================================
# FUNGSI 3: LOOP BACKTEST UTAMA
# =============================================================================

def run_regime_backtest(
    df_m5        : pd.DataFrame,
    df_m15       : pd.DataFrame,
    df_h1        : pd.DataFrame,
    warm_up      : int   = WARM_UP_CANDLES,
    max_candles  : int   = MAX_FORWARD_CANDLES,
    spread_pts   : float = DEFAULT_SPREAD_PTS,
    rrr_min      : Optional[float] = None,
    grace_candles: int   = REGIME_BREAKOUT_GRACE_CANDLES,
    verbose      : bool  = True,
) -> tuple:
    """
    Jalankan backtest regime-based end-to-end untuk seluruh DataFrame historis.

    PERBEDAAN DARI run_backtest() LAMA:
        run_backtest() lama menggunakan single-strategy (rule_engine.py) dan
        bias H1 flat via merge_h1_to_m5(). Fungsi ini menggunakan:
        - Regime M15 untuk pemilihan strategi (Fase 13–14)
        - Tiga strategi terpisah yang dipilih per regime (Fase 15–17)
        - Location filter sebagai informational skor (Fase 18)
        - Regime-aware SL/TP (Fase 19)
        Fungsi ini TIDAK memanggil run_backtest() lama.

    GLOBAL POSITION BLOCKING:
        Hanya SATU posisi terbuka di seluruh sistem pada satu waktu
        (in_trade_until_idx). Trade dari strategi apapun memblok slot.
        Sama dengan desain run_backtest() lama — bukan per-strategi.

    ALUR (urutan wajib):
        1. run_all_indicators() untuk M5, M15, H1
        2. merge_regime_to_m5() — attach regime M15 ke M5 (anti-lookahead)
        3. merge_h1_context_to_m5() — attach H1 context ke M5 (anti-lookahead)
        4. Loop candle i dari warm_up ke akhir
        5. compute_segmented_summary() → breakdown per regime/strategi/sesi
        6. Return (trades_df, segmented_summary)

    Parameter:
        df_m5         : DataFrame M5 mentah (sebelum run_all_indicators)
        df_m15        : DataFrame M15 mentah
        df_h1         : DataFrame H1 mentah
        warm_up       : Candle awal yang dilewati (indikator warm-up)
        max_candles   : Batas forward scan per trade
        spread_pts    : Spread per trade dalam USD (cost model)
        rrr_min       : Override RRR minimum. None = pakai default calculate_sl_tp()
        grace_candles : Candle M15 ke belakang untuk grace window breakout
        verbose       : Print progress ke terminal

    Return:
        tuple (trades_df, segmented_summary):
            trades_df        : pd.DataFrame — satu baris per trade
            segmented_summary: dict — breakdown per regime, strategi, sesi + rekonsiliasi
    """
    if verbose:
        print("=" * 60)
        print("  BACKTEST ENGINE REGIME — XAUUSD M5/M15/H1 Rule-Based")
        print("=" * 60)
        print(f"  M5  candle : {len(df_m5):,} ({df_m5.index[0]} -> {df_m5.index[-1]})")
        print(f"  M15 candle : {len(df_m15):,} ({df_m15.index[0]} -> {df_m15.index[-1]})")
        print(f"  H1  candle : {len(df_h1):,} ({df_h1.index[0]} -> {df_h1.index[-1]})")
        print(f"  Warm-up    : {warm_up} candle")
        print(f"  Max forward: {max_candles} candle (~{max_candles * 5 / 60:.0f} jam trading time)")
        print(f"  Spread     : {spread_pts:.2f} USD (cost total: {spread_pts * 2:.2f} USD)")
        print(f"  Grace M15  : {grace_candles} candle")
        print()

    # ── Langkah 1: Hitung indikator SATU KALI untuk seluruh histori (O(N)) ─────
    if verbose:
        print("[1/5] Menghitung indikator M5...")
    df_m5_ind  = run_all_indicators(df_m5.copy())

    if verbose:
        print("[2/5] Menghitung indikator M15 + precompute regime series...")
    df_m15_ind = run_all_indicators(df_m15.copy())

    if verbose:
        print("[3/5] Menghitung indikator H1...")
    df_h1_ind  = run_all_indicators(df_h1.copy())

    # ── Langkah 2: Merge regime M15 ke M5 (anti-lookahead) ────────────────────
    if verbose:
        print("[4/5] Merge regime M15 → M5 (merge_asof backward)...")
    df_merged = merge_regime_to_m5(df_m5_ind, df_m15_ind, grace_candles)

    # ── Langkah 3: Merge H1 context ke M5 (anti-lookahead) ────────────────────
    if verbose:
        print("[5/5] Merge H1 context → M5 (merge_asof backward)...")
    df_merged = merge_h1_context_to_m5(df_merged, df_h1_ind)

    if verbose:
        nan_m15 = int(df_merged["m15_strategy"].isna().sum())
        nan_h1  = int(df_merged["h1_context"].isna().sum())
        print(f"   M5 candle tanpa M15 reference: {nan_m15:,}")
        print(f"   M5 candle tanpa H1 reference : {nan_h1:,}")
        print()

    # ── Langkah 4: Loop utama — evaluasi sinyal per candle ────────────────────
    if verbose:
        n_scan = len(df_merged) - warm_up
        print(f"Scanning {n_scan:,} candle M5 (index {warm_up} s/d {len(df_merged)-1})...")

    trades             = []
    in_trade_until_idx = -1   # GLOBAL position blocking — tidak per-strategi
    n_evaluated        = 0
    n_skip_no_strategy = 0
    n_skip_not_terpenuhi = 0
    n_skip_invalid_risk  = 0
    n_skip_sl_too_small  = 0
    n_total = len(df_merged)

    for i in range(warm_up, n_total):

        # ── Progress report ─────────────────────────────────────────────────
        if verbose and i % 1000 == 0 and i > warm_up:
            pct = (i - warm_up) / (n_total - warm_up) * 100
            print(f"   Progress: {i:,}/{n_total:,} ({pct:.0f}%) "
                  f"- {len(trades)} trade valid")

        # ── a. Global position blocking ─────────────────────────────────────
        if i <= in_trade_until_idx:
            continue

        row = df_merged.iloc[i]

        # ── b. Baca regime dan strategi dari baris i ─────────────────────────
        m15_strategy = row.get("m15_strategy")
        m15_arah     = row.get("m15_arah")
        m15_source   = row.get("m15_source", "NONE")
        m15_zone     = row.get("m15_zone")     # dict atau None
        m15_regime   = row.get("m15_regime", "CHOP")
        h1_context   = row.get("h1_context")   # dict atau None

        # Pastikan h1_context adalah dict (bukan NaN dari pandas)
        if h1_context is None or (isinstance(h1_context, float) and np.isnan(h1_context)):
            h1_context = {"bias": "NEUTRAL", "strength": None, "strength_zone": None,
                          "ema_gap_pct": None, "close": None, "time": None,
                          "keterangan": "h1_context tidak tersedia (NaN dari merge)"}

        # m15_zone: bisa NaN dari pandas saat value aslinya None di DataFrame object column
        if m15_zone is not None and isinstance(m15_zone, float) and np.isnan(m15_zone):
            m15_zone = None

        n_evaluated += 1

        # ── c. Skip jika tidak ada strategi aktif (CHOP tanpa grace window) ──
        if m15_strategy is None or (isinstance(m15_strategy, float) and np.isnan(m15_strategy)):
            n_skip_no_strategy += 1
            continue

        # ── d. Dispatch ke fungsi evaluate_*() yang sesuai ───────────────────
        close_now = float(row["close"])

        if m15_strategy == "RANGE_REVERSAL":
            # RANGE_REVERSAL: tidak ada parameter arah — menentukan sendiri dari price action
            result = evaluate_range_reversal(
                df_m5  = df_m5_ind,
                idx_m5 = i,
                zone   = m15_zone,
            )

        elif m15_strategy == "BREAKOUT_RETEST":
            result = evaluate_breakout_retest(
                df_m5  = df_m5_ind,
                idx_m5 = i,
                zone   = m15_zone,
                arah   = m15_arah,
            )

        elif m15_strategy == "TREND_FOLLOWING":
            result = evaluate_trend_following(
                df_m5  = df_m5_ind,
                idx_m5 = i,
                arah   = m15_arah,
            )

        else:
            # Strategi tidak dikenal — skip secara graceful
            n_skip_no_strategy += 1
            continue

        # ── e. Skip jika sinyal tidak terpenuhi ─────────────────────────────
        if result.get("terpenuhi") is not True:
            n_skip_not_terpenuhi += 1
            continue

        # Ambil arah dari result (RANGE_REVERSAL menentukan arah sendiri)
        arah_trade = result.get("arah")
        if arah_trade not in ("BUY", "SELL"):
            n_skip_not_terpenuhi += 1
            continue

        # ── f. Hitung entry dengan spread-aware tick_info ────────────────────
        # Pola PERSIS sama dengan run_backtest() lama
        tick_info = {
            "ask": close_now + spread_pts / 2,
            "bid": close_now - spread_pts / 2,
        }

        # ── g. Panggil calculate_regime_sl_tp() ─────────────────────────────
        risk = calculate_regime_sl_tp(
            df_m5          = df_m5_ind.iloc[:i + 1],   # slice anti-lookahead
            entry          = close_now,
            arah           = arah_trade,
            strategy_name  = m15_strategy,
            strategy_result= result,
            zone           = m15_zone,
            rrr_min        = rrr_min,
            tick_info      = tick_info,
        )

        # ── h. Skip jika kalkulasi risk tidak valid ──────────────────────────
        if not risk.get("valid"):
            n_skip_invalid_risk += 1
            continue

        jarak_sl = risk["jarak_sl"]
        jarak_tp = risk["jarak_tp"]

        # ── i. Skip jika jarak SL terlalu kecil (guard anomali data) ─────────
        # Pola PERSIS sama dengan run_backtest() lama
        if jarak_sl < MIN_SL_DISTANCE:
            n_skip_sl_too_small += 1
            continue

        # ── j. Hitung skor confluence (informational — TIDAK PERNAH veto) ────
        atr_now = float(row.get("atr_14", 0)) if not pd.isna(row.get("atr_14", 0)) else 0.0
        confluence = calculate_confluence_summary(
            strategy_name   = m15_strategy,
            strategy_result = result,
            atr             = atr_now,
            h1_context      = h1_context,
        )

        # ── k. Simulasi outcome trade (reuse langsung dari backtester lama) ───
        outcome_info = simulate_trade_outcome(
            df_m5_full  = df_m5_ind,
            entry_idx   = i,
            entry       = risk["entry"],
            sl          = risk["sl"],
            tp          = risk["tp"],
            max_candles = max_candles,
        )

        outcome      = outcome_info["outcome"]
        candles_held = outcome_info["candles_held"]
        ambiguous    = outcome_info["ambiguous_candle"]

        # ── l. Hitung PNL — MENIRU PERSIS formula run_backtest() lama ─────────
        # CATATAN: duplikasi formula ini disengaja (lihat docstring modul)
        spread_cost_total = spread_pts * 2

        if outcome == "TP_HIT":
            rrr_realized = risk.get("rrr_after_spread") or risk["rrr"]
            pnl_points   = +jarak_tp
            pnl_net      = pnl_points - spread_cost_total
            pnl_type     = "TP"

        elif outcome == "SL_HIT":
            rrr_realized = -1.0
            pnl_points   = -jarak_sl
            pnl_net      = pnl_points - spread_cost_total
            pnl_type     = "SL"

        else:  # NO_HIT — mark-to-market, klem ke -jarak_sl (pola SAMA dengan lama)
            exit_price_mtm = outcome_info.get("exit_price_mtm", risk["entry"])

            if arah_trade == "BUY":
                pnl_raw = exit_price_mtm - risk["entry"]
            else:  # SELL
                pnl_raw = risk["entry"] - exit_price_mtm

            pnl_points   = max(pnl_raw, -jarak_sl)
            pnl_net      = pnl_points - spread_cost_total
            rrr_realized = round(pnl_points / jarak_sl, 4) if jarak_sl > 0 else 0.0
            pnl_type     = "MTM"

        # ── m. Susun record trade lengkap ────────────────────────────────────
        # Kolom wajib (identik dengan compute_summary() lama — harus persis sama namanya)
        # + Kolom tambahan khusus fase ini

        # Tentukan label sesi dari entry_time
        entry_time_ts = df_merged.index[i]
        try:
            sess_info     = is_high_liquidity_session(entry_time_ts)
            session_label = sess_info.get("session_label", "UNKNOWN")
        except Exception:
            session_label = "UNKNOWN"

        trades.append({
            # ── Kolom WAJIB (untuk compute_summary() lama) ─────────────────
            "entry_time"      : str(entry_time_ts),
            "exit_time"       : outcome_info["exit_time"],
            "direction"       : arah_trade,
            "entry_price"     : risk["entry"],
            "sl"              : risk["sl"],
            "tp"              : risk["tp"],
            "sl_method"       : risk["sl_method"],
            "outcome"         : outcome,
            "candles_held"    : candles_held,
            "rrr_realized"    : rrr_realized,
            "spread_pts"      : spread_pts,
            "jarak_sl"        : jarak_sl,
            "jarak_tp"        : jarak_tp,
            "pnl_points"      : pnl_points,
            "pnl_net"         : pnl_net,
            "pnl_type"        : pnl_type,
            "ambiguous_candle": ambiguous,
            # ── Kolom TAMBAHAN khusus Fase 21 (segmentasi) ─────────────────
            "regime"           : m15_regime,
            "strategy"         : m15_strategy,
            "strategy_source"  : m15_source,
            "confluence_score" : confluence.get("total_score"),
            "confluence_label" : confluence.get("quality_label"),
            "tp_capped"        : risk.get("tp_capped", False),
            "session"          : session_label,
            # ── Kolom tambahan informatif ───────────────────────────────────
            "atr_value"       : risk.get("atr_value"),
            "rrr_planned"     : risk.get("rrr"),
            "rrr_after_spread": risk.get("rrr_after_spread"),
        })

        # ── n. Update global position blocking ──────────────────────────────
        in_trade_until_idx = i + candles_held

    # ── Langkah 5: Susun output ───────────────────────────────────────────────
    if verbose:
        print(f"\nScan selesai:")
        print(f"   Candle dievaluasi    : {n_evaluated:,}")
        print(f"   Skip (tanpa strategi): {n_skip_no_strategy:,}")
        print(f"   Skip (tidak terpenuhi): {n_skip_not_terpenuhi:,}")
        print(f"   Skip (risk invalid)  : {n_skip_invalid_risk:,}")
        print(f"   Skip (SL terlalu kecil): {n_skip_sl_too_small:,}")
        print(f"   Trade valid          : {len(trades):,}")

    if not trades:
        empty_seg = {
            "overall"       : _empty_summary_regime(),
            "per_regime"    : {},
            "per_strategy"  : {},
            "per_session"   : {},
            "reconciliation": {
                "total_trades"    : 0,
                "sum_per_regime"  : 0,
                "sum_per_strategy": 0,
                "sum_per_session" : 0,
                "reconciled"      : True,
                "keterangan"      : "Tidak ada trade — tidak ada yang perlu direkonsiliasi.",
            },
        }
        return pd.DataFrame(), empty_seg

    trades_df = pd.DataFrame(trades)

    # ── Langkah 6: Compute segmented summary ─────────────────────────────────
    segmented_summary = compute_segmented_summary(trades_df)

    return trades_df, segmented_summary


# =============================================================================
# FUNGSI 4: METRIK TAMBAHAN
# =============================================================================

def compute_profit_factor(trades_df: pd.DataFrame) -> Optional[float]:
    """
    Hitung profit factor = gross_profit / abs(gross_loss).

    Return:
        float — profit factor, atau None jika gross_loss == 0 (semua menang
        atau tidak ada trade SL). None lebih informatif dari infinity.

    Catatan:
        Gross profit  = sum semua pnl_net > 0
        Gross loss    = sum semua pnl_net < 0 (nilai negatif)
        Jika gross_loss == 0 → tidak bisa dibagi → return None
    """
    if trades_df.empty or "pnl_net" not in trades_df.columns:
        return None

    gross_profit = float(trades_df.loc[trades_df["pnl_net"] > 0, "pnl_net"].sum())
    gross_loss   = float(trades_df.loc[trades_df["pnl_net"] < 0, "pnl_net"].sum())

    if gross_loss == 0:
        return None   # Semua positif atau tidak ada loss — tidak bisa hitung PF

    return round(gross_profit / abs(gross_loss), 4)


def compute_expectancy(trades_df: pd.DataFrame) -> Optional[float]:
    """
    Hitung expectancy = rata-rata pnl_net per trade.

    Return:
        float — expectancy dalam USD per trade, atau None jika tidak ada trade.

    Catatan:
        Expectancy positif = sistem menguntungkan rata-rata per trade.
        Ini lebih informatif dari win_rate saja karena memperhitungkan
        ukuran win vs loss (bukan hanya frekuensi).
    """
    if trades_df.empty or "pnl_net" not in trades_df.columns:
        return None

    return round(float(trades_df["pnl_net"].mean()), 4)


# =============================================================================
# FUNGSI 5: RINGKASAN TERSEGMENTASI + VERIFIKASI REKONSILIASI
# =============================================================================

def compute_segmented_summary(trades_df: pd.DataFrame) -> dict:
    """
    Hitung ringkasan tersegmentasi per regime, per strategi, per sesi, plus overall.

    RECONCILIATION (WAJIB):
        Memverifikasi bahwa:
            sum(trades per regime)   == total_trades
            sum(trades per strategi) == total_trades
            sum(trades per sesi)     == total_trades

        Jika ada ketidakcocokan → reconciled=False dan keterangan menjelaskan
        penyebabnya (kemungkinan ada trade dengan kolom segmentasi bernilai None —
        jika ini terjadi, itu BUG yang harus dilaporkan, bukan diabaikan).

    Parameter:
        trades_df : pd.DataFrame output dari run_regime_backtest()

    Return:
        dict berisi:
            "overall"        : dict — compute_summary() + profit_factor + expectancy
            "per_regime"     : {regime_name: {...}, ...}
            "per_strategy"   : {strategy_name: {...}, ...}
            "per_session"    : {session_label: {...}, ...}
            "reconciliation" : {total_trades, sum_per_regime, sum_per_strategy,
                                sum_per_session, reconciled, keterangan}
    """
    if trades_df.empty:
        return {
            "overall"       : _empty_summary_regime(),
            "per_regime"    : {},
            "per_strategy"  : {},
            "per_session"   : {},
            "reconciliation": {
                "total_trades"    : 0,
                "sum_per_regime"  : 0,
                "sum_per_strategy": 0,
                "sum_per_session" : 0,
                "reconciled"      : True,
                "keterangan"      : "Tidak ada trade — tidak ada yang perlu direkonsiliasi.",
            },
        }

    total = len(trades_df)

    # ── Overall ───────────────────────────────────────────────────────────────
    overall_base = compute_summary(trades_df)
    overall = {
        **overall_base,
        "profit_factor": compute_profit_factor(trades_df),
        "expectancy"   : compute_expectancy(trades_df),
    }

    # ── Per Regime ────────────────────────────────────────────────────────────
    per_regime = {}
    if "regime" in trades_df.columns:
        for regime_val, group in trades_df.groupby("regime", dropna=False):
            key = str(regime_val) if regime_val is not None else "__NONE__"
            base = compute_summary(group)
            per_regime[key] = {
                **base,
                "profit_factor": compute_profit_factor(group),
                "expectancy"   : compute_expectancy(group),
            }

    # ── Per Strategy ─────────────────────────────────────────────────────────
    per_strategy = {}
    if "strategy" in trades_df.columns:
        for strat_val, group in trades_df.groupby("strategy", dropna=False):
            key = str(strat_val) if strat_val is not None else "__NONE__"
            base = compute_summary(group)
            per_strategy[key] = {
                **base,
                "profit_factor": compute_profit_factor(group),
                "expectancy"   : compute_expectancy(group),
            }

    # ── Per Session ───────────────────────────────────────────────────────────
    per_session = {}
    if "session" in trades_df.columns:
        for sess_val, group in trades_df.groupby("session", dropna=False):
            key = str(sess_val) if sess_val is not None else "__NONE__"
            base = compute_summary(group)
            per_session[key] = {
                **base,
                "profit_factor": compute_profit_factor(group),
                "expectancy"   : compute_expectancy(group),
            }

    # ── Reconciliation Check ─────────────────────────────────────────────────
    sum_per_regime   = sum(v["total_trades"] for v in per_regime.values())
    sum_per_strategy = sum(v["total_trades"] for v in per_strategy.values())
    sum_per_session  = sum(v["total_trades"] for v in per_session.values())

    reconciled = (
        sum_per_regime   == total and
        sum_per_strategy == total and
        sum_per_session  == total
    )

    if reconciled:
        ket_recon = (
            f"REKONSILIASI OK: total={total}, "
            f"sum_regime={sum_per_regime}, sum_strategy={sum_per_strategy}, "
            f"sum_session={sum_per_session}. Semua identik."
        )
    else:
        issues = []
        if sum_per_regime != total:
            n_none_regime = int(trades_df["regime"].isna().sum()) if "regime" in trades_df.columns else -1
            issues.append(
                f"sum_per_regime={sum_per_regime} != total={total} "
                f"(trade dengan regime=None: {n_none_regime})"
            )
        if sum_per_strategy != total:
            n_none_strat = int(trades_df["strategy"].isna().sum()) if "strategy" in trades_df.columns else -1
            issues.append(
                f"sum_per_strategy={sum_per_strategy} != total={total} "
                f"(trade dengan strategy=None: {n_none_strat})"
            )
        if sum_per_session != total:
            n_none_sess = int(trades_df["session"].isna().sum()) if "session" in trades_df.columns else -1
            issues.append(
                f"sum_per_session={sum_per_session} != total={total} "
                f"(trade dengan session=None: {n_none_sess})"
            )
        ket_recon = "BUG — REKONSILIASI GAGAL: " + "; ".join(issues)

    reconciliation = {
        "total_trades"    : total,
        "sum_per_regime"  : sum_per_regime,
        "sum_per_strategy": sum_per_strategy,
        "sum_per_session" : sum_per_session,
        "reconciled"      : reconciled,
        "keterangan"      : ket_recon,
    }

    return {
        "overall"       : overall,
        "per_regime"    : per_regime,
        "per_strategy"  : per_strategy,
        "per_session"   : per_session,
        "reconciliation": reconciliation,
    }


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _empty_summary_regime() -> dict:
    """
    Return ringkasan kosong yang kompatibel dengan struktur compute_summary() lama,
    ditambah field khusus Fase 21 (profit_factor, expectancy).

    Pola mengikuti _empty_summary() di engine/backtester.py.
    """
    from engine.backtester import _empty_summary as _base_empty
    base = _base_empty()
    return {
        **base,
        "profit_factor": None,
        "expectancy"   : None,
    }
