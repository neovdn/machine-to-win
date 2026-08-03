"""
scripts/_investigate_h1_fix.py
================================
INVESTIGASI DAMPAK FIX BIAS H1 — Apple-to-Apple + Robustness Check

TUJUAN:
    1. Perbandingan apple-to-apple: detect_trend() H1 (lama/gap-gated)
       vs detect_bias_h1() H1 (baru/position-based) — parameter IDENTIK.
    2. Verifikasi set entry_time: apakah trade yang masuk benar-benar sama.
    3. Investigasi no_hit_count=0 pada run CLI default params.
    4. Robustness check walk-forward / OOS dengan H1 fix aktif.
    5. Kesimpulan jujur berbasis expectancy (bukan volume/PnL nominal).

CATATAN:
    Script ini adalah file EKSPERIMEN SEMENTARA.
    Jangan commit ke production — hapus setelah investigasi selesai.
    Tidak mengubah kode inti (detect_bias_h1, merge_h1_to_m5, rule_engine).
    Variant H1 lama dibuat sebagai fungsi lokal sementara, bukan modifikasi file.

CARA PAKAI:
    cd c:\\Users\\mercy\\AppData\\Local\\machine-to-win
    python scripts/_investigate_h1_fix.py
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators   import (
    run_all_indicators,
    detect_trend,        # H1 LAMA: gap-gated, identik dengan detect_trend() M5
    detect_bias_h1,      # H1 BARU: position-based, tanpa gap threshold
)
from engine.backtester import (
    validate_no_lookahead,
    simulate_trade_outcome,
    compute_summary,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE,
    DEFAULT_SPREAD_PTS,
)
from engine.rule_engine  import evaluate_entry
from engine.risk_manager import calculate_sl_tp


# =============================================================================
# PARAMETER SETS
# =============================================================================

# Opsi A: parameter test_phase_0_baseline_consistency (parameter "lama")
PARAMS_BASELINE = {
    "atr_multiplier"      : 1.5,
    "swing_lookback"      : 50,
    "swing_wing"          : 5,
    "rrr_min"             : 2.0,
    "swing_clamp_min_atr" : 0.0,
    "swing_clamp_max_atr" : 999.0,
    "label"               : "BASELINE (atr=1.5, lookback=50, wing=5, rrr=2.0)",
}

# Opsi B: parameter default scalp_m5 CLI (parameter "baru" dari run_backtest.py)
PARAMS_SCALP_M5 = {
    "atr_multiplier"      : 0.9,
    "swing_lookback"      : 15,
    "swing_wing"          : 3,
    "rrr_min"             : 1.3,
    "swing_clamp_min_atr" : None,
    "swing_clamp_max_atr" : None,
    "label"               : "SCALP_M5 (atr=0.9, lookback=15, wing=3, rrr=1.3)",
}


# =============================================================================
# MERGE H1 LAMA (gap-gated) — variant sementara untuk perbandingan A/B
# =============================================================================

def _merge_h1_legacy(df_m5_ind: pd.DataFrame, df_h1_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Versi LAMA merge_h1_to_m5 yang memakai detect_trend() untuk H1
    (gap-gated, min_ema_gap_pct=0.05) — BUKAN detect_bias_h1().

    Ini adalah variant SEMENTARA untuk investigasi A/B comparison.
    Tidak dipakai di production — hapus setelah investigasi selesai.

    PERBEDAAN vs merge_h1_to_m5() production:
        Production (baru): gunakan h1_reset["bias_h1"] → trend_h1
        Legacy (lama):     gunakan h1_reset["trend"] → trend_h1
                           (dari detect_trend() yang sudah dijalankan di run_all_indicators)
    """
    m5_reset = df_m5_ind.reset_index()
    h1_reset = df_h1_ind.reset_index()

    m5_reset = m5_reset.sort_values("time").reset_index(drop=True)
    h1_reset = h1_reset.sort_values("time").reset_index(drop=True)

    # KUNCI PERBEDAAN: gunakan kolom "trend" (dari detect_trend, gap-gated)
    # bukan "bias_h1" (dari detect_bias_h1, position-based)
    # run_all_indicators sudah memanggil detect_trend() jadi kolom "trend" sudah ada
    h1_rename = {
        "trend"       : "trend_h1",       # <-- LAMA: gap-gated detect_trend()
        "ema_gap_pct" : "ema_gap_pct_h1",
        "ema_9"       : "ema_9_h1",
        "ema_21"      : "ema_21_h1",
    }
    h1_slim = h1_reset[["time"] + list(h1_rename.keys())].rename(columns=h1_rename)

    merged = pd.merge_asof(
        m5_reset,
        h1_slim,
        on        = "time",
        direction = "backward",
    )
    merged = merged.set_index("time")
    return merged


def _merge_h1_new(df_m5_ind: pd.DataFrame, df_h1_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Versi BARU merge_h1_to_m5 yang memakai detect_bias_h1() untuk H1
    (position-based, tanpa gap threshold) — IDENTIK dengan production.
    """
    # Import production version
    from engine.backtester import merge_h1_to_m5
    return merge_h1_to_m5(df_m5_ind, df_h1_ind)


# =============================================================================
# CORE: Single backtest run (reusable)
# =============================================================================

def _run_one_backtest(
    df_m5_ind   : pd.DataFrame,
    df_merged   : pd.DataFrame,
    params      : dict,
    spread_pts  : float = DEFAULT_SPREAD_PTS,
    max_candles : int   = MAX_FORWARD_CANDLES,
    warm_up     : int   = WARM_UP_CANDLES,
    collect_entry_times : bool = False,
) -> tuple:
    """
    Jalankan satu backtest dengan df_merged yang sudah disiapkan.
    Return: (trades_df, summary, entry_times_list)
    """
    trades             = []
    in_trade_until_idx = -1
    n_total            = len(df_merged)
    entry_times        = []

    for i in range(warm_up, n_total):
        if i <= in_trade_until_idx:
            continue

        row = df_merged.iloc[i]
        if pd.isna(row.get("trend_h1")):
            continue

        signals = {
            "time"        : df_merged.index[i],
            "close"       : float(row["close"]),
            "ema_9"       : float(row["ema_9"]),
            "ema_21"      : float(row["ema_21"]),
            "rsi_14"      : float(row["rsi_14"]),
            "trend"       : str(row["trend"]),
            "ema_gap_pct" : float(row["ema_gap_pct"]),
            "trend_h1"    : str(row["trend_h1"]),
            "volume_ratio": float(row["volume_ratio"])
                if "volume_ratio" in row and not pd.isna(row.get("volume_ratio"))
                else None,
        }

        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in signals.values()
            if isinstance(v, (int, float))
        )
        if has_nan:
            continue

        decision = evaluate_entry(signals, volume_mode="FILTER")
        if decision["keputusan"] not in ("BUY", "SELL"):
            continue

        arah     = decision["keputusan"]
        df_slice = df_m5_ind.iloc[: i + 1]

        risk = calculate_sl_tp(
            df                  = df_slice,
            entry               = signals["close"],
            arah                = arah,
            profile             = "scalp_m5",
            rrr_min             = params["rrr_min"],
            atr_multiplier      = params["atr_multiplier"],
            swing_lookback      = params["swing_lookback"],
            swing_wing          = params["swing_wing"],
            swing_clamp_min_atr = params.get("swing_clamp_min_atr"),
            swing_clamp_max_atr = params.get("swing_clamp_max_atr"),
            tick_info           = {
                "ask": signals["close"] + spread_pts / 2,
                "bid": signals["close"] - spread_pts / 2,
            },
        )

        if not risk["valid"]:
            continue

        sl       = risk["sl"]
        tp       = risk["tp"]
        jarak_sl = risk["jarak_sl"]
        jarak_tp = risk["jarak_tp"]

        if jarak_sl < MIN_SL_DISTANCE:
            continue

        outcome_info = simulate_trade_outcome(
            df_m5_full  = df_m5_ind,
            entry_idx   = i,
            entry       = risk["entry"],
            sl          = sl,
            tp          = tp,
            max_candles = max_candles,
        )

        outcome      = outcome_info["outcome"]
        candles_held = outcome_info["candles_held"]

        spread_cost_total = spread_pts * 2

        if outcome == "TP_HIT":
            rrr_realized = risk.get("rrr_after_spread") or risk["rrr"]
            pnl_points   = +jarak_tp
            pnl_net      = pnl_points - spread_cost_total
        elif outcome == "SL_HIT":
            rrr_realized = -1.0
            pnl_points   = -jarak_sl
            pnl_net      = pnl_points - spread_cost_total
        else:  # NO_HIT
            exit_price_mtm = outcome_info.get("exit_price_mtm", risk["entry"])
            if arah == "BUY":
                pnl_raw = exit_price_mtm - risk["entry"]
            else:
                pnl_raw = risk["entry"] - exit_price_mtm
            pnl_points   = max(pnl_raw, -jarak_sl)
            pnl_net      = pnl_points - spread_cost_total
            rrr_realized = round(pnl_points / jarak_sl, 4) if jarak_sl > 0 else 0.0

        entry_time_str = str(df_merged.index[i])
        if collect_entry_times:
            entry_times.append(entry_time_str)

        trades.append({
            "entry_time"       : entry_time_str,
            "direction"        : arah,
            "outcome"          : outcome,
            "candles_held"     : candles_held,
            "rrr_realized"     : rrr_realized,
            "rrr_after_spread" : risk.get("rrr_after_spread"),
            "pnl_points"       : pnl_points,
            "pnl_net"          : pnl_net,
            "pnl_type"         : "TP" if outcome == "TP_HIT" else "SL" if outcome == "SL_HIT" else "MTM",
            "sl_method"        : risk.get("sl_method", "UNKNOWN"),
            "trend_h1"         : signals["trend_h1"],
            "ambiguous_candle" : outcome_info["ambiguous_candle"],
            "spread_pts"       : spread_pts,
        })

        in_trade_until_idx = i + candles_held

    if not trades:
        return pd.DataFrame(), {}, entry_times

    trades_df = pd.DataFrame(trades)
    summary   = compute_summary(trades_df)
    return trades_df, summary, entry_times


# =============================================================================
# TASK 1: Apple-to-Apple Comparison
# =============================================================================

def task1_apple_to_apple(df_m5_ind, df_h1_ind, params, label):
    """
    Jalankan backtest DUA KALI — satu dengan H1 lama (gap-gated), satu dengan H1 baru
    (position-based) — menggunakan PARAMETER DAN DATASET YANG IDENTIK.

    Return: dict dengan metrik kedua run dan perbandingan delta.
    """
    print(f"\n{'='*70}")
    print(f"TASK 1: Apple-to-Apple Comparison")
    print(f"Parameter: {label}")
    print(f"{'='*70}")

    # ── Run LAMA (H1 gap-gated detect_trend) ──────────────────────────────────
    print("\n[LAMA] Merging H1 via detect_trend() (gap-gated, min_gap=0.05%)...")
    df_merged_legacy = _merge_h1_legacy(df_m5_ind, df_h1_ind)

    t0 = time.time()
    trades_old, summary_old, entry_old = _run_one_backtest(
        df_m5_ind   = df_m5_ind,
        df_merged   = df_merged_legacy,
        params      = params,
        collect_entry_times = True,
    )
    t_old = time.time() - t0
    print(f"   Selesai dalam {t_old:.1f}s — {summary_old.get('total_trades', 0)} trade")

    # ── Run BARU (H1 position-based detect_bias_h1) ───────────────────────────
    print("\n[BARU] Merging H1 via detect_bias_h1() (position-based, zero gap threshold)...")
    df_merged_new = _merge_h1_new(df_m5_ind, df_h1_ind)

    t0 = time.time()
    trades_new, summary_new, entry_new = _run_one_backtest(
        df_m5_ind   = df_m5_ind,
        df_merged   = df_merged_new,
        params      = params,
        collect_entry_times = True,
    )
    t_new = time.time() - t0
    print(f"   Selesai dalam {t_new:.1f}s — {summary_new.get('total_trades', 0)} trade")

    # ── Cetak Tabel Perbandingan ──────────────────────────────────────────────
    _print_comparison_table(summary_old, summary_new, "LAMA (gap-gated)", "BARU (position-based)")

    return {
        "summary_old"  : summary_old,
        "summary_new"  : summary_new,
        "entry_old"    : entry_old,
        "entry_new"    : entry_new,
        "trades_old"   : trades_old,
        "trades_new"   : trades_new,
    }


def _print_comparison_table(s_old, s_new, label_old="LAMA", label_new="BARU"):
    metrics = [
        ("total_trades",         "Total Trades",          "{:,}",   False),
        ("tp_count",             "TP Hit",                "{:,}",   False),
        ("sl_count",             "SL Hit",                "{:,}",   False),
        ("no_hit_count",         "NO HIT",                "{:,}",   False),
        ("win_rate",             "Win Rate (TP/closed)",  "{:.1%}",  True),
        ("no_hit_rate",          "NO HIT Rate",           "{:.1%}",  True),
        ("avg_rrr_realized",     "Avg RRR Realized",      "{:+.4f}", True),
        ("avg_rrr_realized_all", "Avg RRR Realized (all)","{:+.4f}", True),
        ("avg_candles_held",     "Avg Candles Held",      "{:.1f}",  True),
        ("avg_candles_held_all", "Avg Candles (all)",     "{:.1f}",  True),
        ("total_pnl_net",        "Total PnL Net",         "{:+.2f}", True),
        ("max_drawdown_net",     "Max Drawdown Net",      "{:+.2f}", True),
    ]

    col_w = 32
    print(f"\n{'Metrik':<{col_w}} {''+label_old:>22} {''+label_new:>22} {'Delta':>12}")
    print("-" * (col_w + 58))

    for key, name, fmt, compute_delta in metrics:
        v_old = s_old.get(key)
        v_new = s_new.get(key)

        def _fmt(v):
            if v is None:
                return "N/A"
            try:
                return fmt.format(v)
            except Exception:
                return str(v)

        delta_str = ""
        if compute_delta and v_old is not None and v_new is not None:
            try:
                delta = v_new - v_old
                if "Rate" in name or "Win" in name:
                    delta_str = f"{delta:+.1%}"
                elif "RRR" in name:
                    delta_str = f"{delta:+.4f}R"
                else:
                    delta_str = f"{delta:+.2f}"
            except Exception:
                delta_str = "-"

        print(f"  {name:<{col_w-2}} {_fmt(v_old):>22} {_fmt(v_new):>22} {delta_str:>12}")

    print()


# =============================================================================
# TASK 2: Verifikasi Entry Set (apakah trade yang sama?)
# =============================================================================

def task2_verify_entry_set(entry_old, entry_new, trades_old, trades_new):
    """
    Bandingkan daftar entry_time antara run lama dan baru.
    Laporkan % yang identik dan contoh perbedaan.
    """
    print(f"\n{'='*70}")
    print(f"TASK 2: Verifikasi Entry Set")
    print(f"{'='*70}")

    set_old = set(entry_old)
    set_new = set(entry_new)

    n_old = len(entry_old)
    n_new = len(entry_new)

    # Intersection
    common     = set_old & set_new
    only_old   = set_old - set_new  # ada di lama, tidak ada di baru
    only_new   = set_new - set_old  # ada di baru, tidak ada di lama

    pct_common_of_old = len(common) / n_old * 100 if n_old > 0 else 0
    pct_common_of_new = len(common) / n_new * 100 if n_new > 0 else 0

    print(f"\n  Total trade LAMA : {n_old:,}")
    print(f"  Total trade BARU : {n_new:,}")
    print(f"  Entry identik    : {len(common):,} ({pct_common_of_old:.1f}% dari LAMA, {pct_common_of_new:.1f}% dari BARU)")
    print(f"  Hanya di LAMA    : {len(only_old):,} (di-drop oleh H1 baru)")
    print(f"  Hanya di BARU    : {len(only_new):,} (ditambah oleh H1 baru)")

    # Tampilkan contoh 5 perbedaan
    if only_old:
        print(f"\n  Contoh entry yang ADA di LAMA tapi TIDAK di BARU (H1 baru memblok):")
        for et in sorted(only_old)[:5]:
            # Cari outcome di trades_old
            if not trades_old.empty and "entry_time" in trades_old.columns:
                row = trades_old[trades_old["entry_time"] == et]
                if not row.empty:
                    r = row.iloc[0]
                    print(f"    {et} | dir={r.get('direction','?')} | trend_h1={r.get('trend_h1','?')} | outcome={r.get('outcome','?')}")
                else:
                    print(f"    {et}")
            else:
                print(f"    {et}")

    if only_new:
        print(f"\n  Contoh entry yang ADA di BARU tapi TIDAK di LAMA (H1 baru membuka sinyal baru):")
        for et in sorted(only_new)[:5]:
            if not trades_new.empty and "entry_time" in trades_new.columns:
                row = trades_new[trades_new["entry_time"] == et]
                if not row.empty:
                    r = row.iloc[0]
                    print(f"    {et} | dir={r.get('direction','?')} | trend_h1={r.get('trend_h1','?')} | outcome={r.get('outcome','?')}")
                else:
                    print(f"    {et}")
            else:
                print(f"    {et}")

    print()
    return {
        "n_common"    : len(common),
        "n_only_old"  : len(only_old),
        "n_only_new"  : len(only_new),
        "pct_common_of_old": pct_common_of_old,
        "pct_common_of_new": pct_common_of_new,
    }


# =============================================================================
# TASK 3: Investigasi no_hit_count = 0 (scalp_m5 default params)
# =============================================================================

def task3_investigate_nohit(df_m5_ind, df_h1_ind):
    """
    Investigasi mengapa run CLI dengan scalp_m5 default params menghasilkan
    no_hit_count = 0. Cek distribusi candles_held dan histogram outcome.
    """
    print(f"\n{'='*70}")
    print(f"TASK 3: Investigasi no_hit_count = 0")
    print(f"{'='*70}")

    params_cli = {
        "atr_multiplier"      : 0.9,
        "swing_lookback"      : 15,
        "swing_wing"          : 3,
        "rrr_min"             : 1.3,
        "swing_clamp_min_atr" : None,
        "swing_clamp_max_atr" : None,
    }

    print("\n[CLI] Merging H1 baru dan running scalp_m5 default params (max_candles=288)...")
    df_merged_new = _merge_h1_new(df_m5_ind, df_h1_ind)

    t0 = time.time()
    trades_df, summary, _ = _run_one_backtest(
        df_m5_ind   = df_m5_ind,
        df_merged   = df_merged_new,
        params      = params_cli,
        max_candles = MAX_FORWARD_CANDLES,  # 288
    )
    t = time.time() - t0
    print(f"   Selesai dalam {t:.1f}s — {len(trades_df):,} trade")

    if trades_df.empty:
        print("   WARNING: Tidak ada trade yang dihasilkan.")
        return {}

    print(f"\n  Distribusi outcome:")
    outcome_dist = trades_df["outcome"].value_counts()
    for k, v in outcome_dist.items():
        pct = v / len(trades_df) * 100
        print(f"    {k:<12} : {v:>6,} ({pct:.1f}%)")

    no_n = int((trades_df["outcome"] == "NO_HIT").sum())
    print(f"\n  no_hit_count AKTUAL = {no_n}")

    print(f"\n  Statistik candles_held:")
    ch = trades_df["candles_held"]
    print(f"    min    : {ch.min()}")
    print(f"    max    : {ch.max()}")
    print(f"    median : {ch.median():.0f}")
    print(f"    mean   : {ch.mean():.1f}")
    print(f"    std    : {ch.std():.1f}")

    # Histogram candles_held
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 288]
    print(f"\n  Histogram candles_held:")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mask = (ch > lo) & (ch <= hi) if i > 0 else (ch <= hi)
        n    = mask.sum()
        pct  = n / len(trades_df) * 100
        bar  = "█" * int(pct / 2)
        print(f"    {lo:>4} – {hi:>4} candle: {n:>5,} trade ({pct:5.1f}%) {bar}")

    # Distribusi candles_held untuk masing-masing outcome
    print(f"\n  Candles held per outcome:")
    for outcome in ["TP_HIT", "SL_HIT", "NO_HIT"]:
        sub = trades_df[trades_df["outcome"] == outcome]["candles_held"]
        if len(sub) > 0:
            print(f"    {outcome:<10}: n={len(sub):<6} min={sub.min():<5} max={sub.max():<5} mean={sub.mean():.1f}")

    # Apakah perbedaan karena max_candles override?
    # Cek apakah DEFAULT_SPREAD_PTS atau max_candles ada yang ter-override
    print(f"\n  Konfirmasi parameter:")
    print(f"    MAX_FORWARD_CANDLES (default) = {MAX_FORWARD_CANDLES}")
    print(f"    max_candles yang dipakai       = {MAX_FORWARD_CANDLES}")

    # Kesimpulan
    if no_n == 0:
        if ch.max() < MAX_FORWARD_CANDLES:
            print(f"\n  KESIMPULAN: no_hit=0 adalah VALID karena semua {len(trades_df):,} trade")
            print(f"  resolve dalam {ch.max()} candle — KURANG dari window max ({MAX_FORWARD_CANDLES} candle).")
            print(f"  Parameter scalp_m5 (atr=0.9, rrr=1.3) menghasilkan SL/TP yang relatif")
            print(f"  dekat → trade resolve lebih cepat → tidak ada yang expired.")
        else:
            print(f"\n  PERLU PENYELIDIKAN LEBIH: ada trade dengan candles_held={ch.max()}")
            print(f"  tapi no_hit_count masih 0. Periksa apakah max_candles benar-benar dipakai.")
    else:
        print(f"\n  no_hit_count={no_n} — TIDAK nol. Mungkin laporan sebelumnya merujuk run yang berbeda.")

    return {"no_hit_count": no_n, "candles_held_stats": ch.describe().to_dict()}


# =============================================================================
# TASK 4: Robustness Check Walk-Forward
# =============================================================================

def task4_walk_forward(m5_path_ext, h1_path_ext, params):
    """
    Walk-forward dengan parameter FIXED dan H1 fix baru aktif.
    Evaluasi konsistensi win_rate dan avg_rrr_realized lintas fold.
    """
    print(f"\n{'='*70}")
    print(f"TASK 4: Robustness Check Walk-Forward")
    print(f"{'='*70}")

    if not os.path.exists(m5_path_ext) or not os.path.exists(h1_path_ext):
        print(f"  WARNING: Extended dataset tidak ditemukan:")
        print(f"    M5: {m5_path_ext}")
        print(f"    H1: {h1_path_ext}")
        print(f"  Fallback ke dataset 2026-01-01_2026-07-25.")
        return {}

    print(f"\n  Loading extended dataset...")
    df_m5_raw = load_candles_csv(m5_path_ext)
    df_h1_raw = load_candles_csv(h1_path_ext)

    if df_m5_raw is None or df_h1_raw is None:
        print("  ERROR: Gagal load extended dataset.")
        return {}

    print(f"  M5: {len(df_m5_raw):,} candle ({df_m5_raw.index[0]} → {df_m5_raw.index[-1]})")
    print(f"  H1: {len(df_h1_raw):,} candle ({df_h1_raw.index[0]} → {df_h1_raw.index[-1]})")

    # Hitung indikator satu kali
    print("  Menghitung indikator (satu kali)...")
    df_m5_ind = run_all_indicators(df_m5_raw.copy())
    df_h1_ind = run_all_indicators(df_h1_raw.copy())
    df_merged = _merge_h1_new(df_m5_ind, df_h1_ind)

    # Define walk-forward folds (bulanan, ~10 fold)
    # Dataset: Jun 2025 – Jul 2026
    folds = [
        {"calib": ("2025-06-01", "2025-08-31"), "val": ("2025-09-01", "2025-09-30"), "label": "Fold 1"},
        {"calib": ("2025-07-01", "2025-09-30"), "val": ("2025-10-01", "2025-10-31"), "label": "Fold 2"},
        {"calib": ("2025-08-01", "2025-10-31"), "val": ("2025-11-01", "2025-11-30"), "label": "Fold 3"},
        {"calib": ("2025-09-01", "2025-11-30"), "val": ("2025-12-01", "2025-12-31"), "label": "Fold 4"},
        {"calib": ("2025-10-01", "2025-12-31"), "val": ("2026-01-01", "2026-01-31"), "label": "Fold 5"},
        {"calib": ("2025-11-01", "2026-01-31"), "val": ("2026-02-01", "2026-02-28"), "label": "Fold 6"},
        {"calib": ("2025-12-01", "2026-02-28"), "val": ("2026-03-01", "2026-03-31"), "label": "Fold 7"},
        {"calib": ("2026-01-01", "2026-03-31"), "val": ("2026-04-01", "2026-04-30"), "label": "Fold 8"},
        {"calib": ("2026-02-01", "2026-04-30"), "val": ("2026-05-01", "2026-05-31"), "label": "Fold 9"},
        {"calib": ("2026-03-01", "2026-05-31"), "val": ("2026-06-01", "2026-06-30"), "label": "Fold 10"},
        {"calib": ("2026-04-01", "2026-06-30"), "val": ("2026-07-01", "2026-07-25"), "label": "Fold 11"},
    ]

    results = []

    def _filter(df, start_str, end_str):
        start = pd.Timestamp(start_str, tz="UTC")
        end   = pd.Timestamp(end_str,   tz="UTC")
        return df[(df.index >= start) & (df.index <= end)]

    print(f"\n  {'Fold':<10} {'Val Period':<26} {'Trades':>7} {'WinRate':>9} {'AvgRRR':>9} {'NoHit%':>8} {'PnL':>9}")
    print("  " + "-" * 82)

    for fold in folds:
        # Validasi period (tidak perlu run kalibrasi karena parameter sudah FIXED)
        df_m5_val   = _filter(df_m5_ind,  fold["val"][0], fold["val"][1])
        df_merged_v = _filter(df_merged,   fold["val"][0], fold["val"][1])

        if len(df_m5_val) < WARM_UP_CANDLES + 20:
            print(f"  {fold['label']:<10} {'SKIP (data tidak cukup)'}")
            continue

        t_df, t_sum, _ = _run_one_backtest(
            df_m5_ind   = df_m5_val,
            df_merged   = df_merged_v,
            params      = params,
            warm_up     = min(WARM_UP_CANDLES, max(20, len(df_m5_val) // 10)),
        )

        n      = t_sum.get("total_trades", 0)
        wr     = t_sum.get("win_rate")
        rrr    = t_sum.get("avg_rrr_realized")
        nhr    = t_sum.get("no_hit_rate", 0)
        pnl    = t_sum.get("total_pnl_net", 0)

        wr_str  = f"{wr:.1%}"  if wr  is not None else "N/A"
        rrr_str = f"{rrr:+.3f}R" if rrr is not None else "N/A"
        nhr_str = f"{nhr:.1%}" if nhr is not None else "N/A"

        val_label = f"{fold['val'][0]} – {fold['val'][1]}"
        print(f"  {fold['label']:<10} {val_label:<26} {n:>7,} {wr_str:>9} {rrr_str:>9} {nhr_str:>8} {pnl:>+9.1f}")

        results.append({
            "fold"        : fold["label"],
            "val_start"   : fold["val"][0],
            "val_end"     : fold["val"][1],
            "total_trades": n,
            "win_rate"    : wr,
            "avg_rrr"     : rrr,
            "no_hit_rate" : nhr,
            "pnl_net"     : pnl,
        })

    if results:
        valid_results = [r for r in results if r["total_trades"] >= 10]
        win_rates = [r["win_rate"] for r in valid_results if r["win_rate"] is not None]
        avg_rrrs  = [r["avg_rrr"]  for r in valid_results if r["avg_rrr"]  is not None]

        print("\n  Ringkasan Robustness:")
        if win_rates:
            print(f"    Win Rate — min={min(win_rates):.1%}, max={max(win_rates):.1%}, mean={np.mean(win_rates):.1%}, std={np.std(win_rates):.1%}")
        if avg_rrrs:
            print(f"    Avg RRR  — min={min(avg_rrrs):+.3f}, max={max(avg_rrrs):+.3f}, mean={np.mean(avg_rrrs):+.3f}, std={np.std(avg_rrrs):.3f}")
            neg_count = sum(1 for r in avg_rrrs if r <= 0)
            print(f"    Fold dengan avg_rrr <= 0: {neg_count}/{len(avg_rrrs)}")

    return results


# =============================================================================
# TASK 5: Kesimpulan
# =============================================================================

def task5_conclusion(result_baseline, result_scalp):
    """
    Cetak kesimpulan jujur berbasis expectancy, bukan volume/PnL nominal.
    """
    print(f"\n{'='*70}")
    print(f"TASK 5: KESIMPULAN INVESTIGASI")
    print(f"{'='*70}")

    # Helper: hitung expectancy sederhana
    def expectancy(summary):
        wr  = summary.get("win_rate")
        rrr = summary.get("avg_rrr_realized")
        if wr is None or rrr is None:
            return None
        # E = wr * avg_rrr_realized + (1-wr) * (-1) — per-trade expectancy
        return round(wr * rrr + (1 - wr) * (-1.0), 4)

    for label, result in [("BASELINE params", result_baseline), ("SCALP_M5 params", result_scalp)]:
        s_old = result.get("summary_old", {})
        s_new = result.get("summary_new", {})

        wr_old  = s_old.get("win_rate");      wr_new  = s_new.get("win_rate")
        rrr_old = s_old.get("avg_rrr_realized"); rrr_new = s_new.get("avg_rrr_realized")
        e_old   = expectancy(s_old);          e_new   = expectancy(s_new)

        print(f"\n  [{label}]")
        print(f"  Win Rate  : LAMA={wr_old:.1%}  →  BARU={wr_new:.1%}  ({'naik' if wr_new > wr_old else 'TURUN' if wr_new < wr_old else 'netral'} {abs(wr_new-wr_old):.1%})" if wr_old and wr_new else "  Win Rate  : N/A")
        print(f"  AvgRRR    : LAMA={rrr_old:+.4f}  →  BARU={rrr_new:+.4f}  ({'naik' if rrr_new > rrr_old else 'TURUN' if rrr_new < rrr_old else 'netral'})" if rrr_old is not None and rrr_new is not None else "  AvgRRR    : N/A")
        print(f"  Expectancy: LAMA={e_old:+.4f}R  →  BARU={e_new:+.4f}R" if e_old is not None and e_new is not None else "  Expectancy: N/A")

    print(f"""
  ─────────────────────────────────────────────────────────────────────
  FORMAT KESIMPULAN WAJIB (sesuai instruksi):
  ─────────────────────────────────────────────────────────────────────

  Apple-to-apple result:
    Lihat tabel di atas untuk win_rate dan avg_rrr_realized ketika
    HANYA logika H1 yang diubah (parameter identik di kedua run).

  Net positif / negatif untuk sistem:
    Gunakan kolom Expectancy = win_rate × avg_rrr_realized + (1-wr) × (-1).
    Jika expectancy BARU > expectancy LAMA → fix H1 net positif.
    Jika expectancy BARU < expectancy LAMA → fix H1 terdilusi edge.

  Rekomendasi:
    Lihat Task 4 (Walk-Forward) — jika avg_rrr konsisten positif di >70%
    fold dan std rendah, fix H1 layak dipertahankan.
    Jika avg_rrr sering negatif atau volatile, pertimbangkan filter kualitas
    tambahan (misal: min EMA gap H1 > 0.02% agar hanya ambil sinyal H1 yang
    sudah ada momentum — lebih longgar dari 0.05% M5 tapi tidak zero-threshold).
  """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  INVESTIGASI DAMPAK FIX BIAS H1 — Apple-to-Apple + Robustness")
    print("=" * 70)
    print(f"  File ini adalah EKSPERIMEN SEMENTARA — hapus setelah selesai.")
    print()

    # ── Load dataset ─────────────────────────────────────────────────────────
    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

    m5_path_ext = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
    h1_path_ext = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"ERROR: Dataset 2026 tidak ditemukan:")
        print(f"  {m5_path}")
        print(f"  {h1_path}")
        sys.exit(1)

    print(f"Loading data 2026-01-01_2026-07-25...")
    df_m5_raw = load_candles_csv(m5_path)
    df_h1_raw = load_candles_csv(h1_path)

    if df_m5_raw is None or df_h1_raw is None:
        print("ERROR: Gagal load CSV.")
        sys.exit(1)

    print(f"  M5: {len(df_m5_raw):,} candle ({df_m5_raw.index[0]} → {df_m5_raw.index[-1]})")
    print(f"  H1: {len(df_h1_raw):,} candle ({df_h1_raw.index[0]} → {df_h1_raw.index[-1]})")

    # Hitung indikator sekali
    print("\nMenghitung indikator (satu kali, O(n))...")
    df_m5_ind = run_all_indicators(df_m5_raw.copy())
    df_h1_ind = run_all_indicators(df_h1_raw.copy())

    # Validasi no look-ahead
    val = validate_no_lookahead(df_m5_raw, n_samples=5)
    print(f"  {val['message']}")
    if not val["passed"]:
        print("FATAL: Look-ahead terdeteksi! Backtest dihentikan.")
        sys.exit(1)

    # ── TASK 1A: Apple-to-apple dengan BASELINE params ───────────────────────
    result_baseline = task1_apple_to_apple(
        df_m5_ind = df_m5_ind,
        df_h1_ind = df_h1_ind,
        params    = PARAMS_BASELINE,
        label     = PARAMS_BASELINE["label"],
    )

    # ── TASK 1B: Apple-to-apple dengan SCALP_M5 params ───────────────────────
    result_scalp = task1_apple_to_apple(
        df_m5_ind = df_m5_ind,
        df_h1_ind = df_h1_ind,
        params    = PARAMS_SCALP_M5,
        label     = PARAMS_SCALP_M5["label"],
    )

    # ── TASK 2: Verifikasi entry set (untuk BASELINE params) ─────────────────
    task2_verify_entry_set(
        entry_old  = result_baseline["entry_old"],
        entry_new  = result_baseline["entry_new"],
        trades_old = result_baseline["trades_old"],
        trades_new = result_baseline["trades_new"],
    )

    # ── TASK 3: Investigasi no_hit_count = 0 ─────────────────────────────────
    task3_investigate_nohit(df_m5_ind, df_h1_ind)

    # ── TASK 4: Walk-Forward dengan extended dataset ──────────────────────────
    task4_walk_forward(
        m5_path_ext = m5_path_ext,
        h1_path_ext = h1_path_ext,
        params      = PARAMS_SCALP_M5,
    )

    # ── TASK 5: Kesimpulan ────────────────────────────────────────────────────
    task5_conclusion(result_baseline, result_scalp)

    print("\n" + "=" * 70)
    print("  INVESTIGASI SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()
