"""
scripts/_fase7_full_validation.py
===================================
Validasi lengkap Fase 7 — Steps 7.3.2 sampai 7.3.5.

  7.3.2: ON vs OFF — verifikasi trade set identik (entry_time 100% sama)
  7.3.3: Analisis korelasi score_candle_pattern vs is_win, pnl_net
         + uji redundansi vs 3 komponen existing
         + koreksi multiple comparison (Bonferroni + BH-FDR)
         + breakdown per jenis pattern
  7.3.4: Walk-forward — distribusi STRONG/MODERATE/WEAK lintas fold
  7.3.5: Kesimpulan LOLOS / TIDAK LOLOS / PERLU DATA LEBIH

Dataset: data/historical/XAUUSD_{M5,H1}_2025-06-01_2026-07-25.csv (14 bulan)
Parameter fixed (Fase 1): atr=0.9, lookback=15, wing=3, rrr=1.3

Cara pakai:
    python scripts/_fase7_full_validation.py
"""

import os
import sys
import time
from datetime import timezone

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.indicators   import run_all_indicators
from engine.rule_engine  import evaluate_entry, calculate_setup_quality
from engine.risk_manager import calculate_sl_tp, find_nearest_swing
from engine.backtester   import (
    merge_h1_to_m5, validate_no_lookahead, simulate_trade_outcome,
    compute_summary, WARM_UP_CANDLES, MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE, DEFAULT_SPREAD_PTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
M5_PATH  = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
H1_PATH  = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")
SPREAD   = 0.50

FASE1 = dict(atr_mult=0.9, lookback=15, wing=3, rrr_min=1.3)

# Walk-forward: 3 bulan calib, 1 bulan validasi, geser 1 bulan
WF_CALIB_MONTHS = 3
WF_VAL_MONTHS   = 1


# =============================================================================
# BAGIAN 0: LOAD DATA
# =============================================================================

def load_and_prepare():
    """Load + hitung indikator + merge. Dilakukan 1x untuk semua validasi."""
    print("  Memuat data historis...")
    df_m5 = pd.read_csv(M5_PATH)
    df_m5["time"] = pd.to_datetime(df_m5["time"])
    df_m5.set_index("time", inplace=True)
    if df_m5.index.tzinfo is None:
        df_m5.index = df_m5.index.tz_localize("UTC")

    df_h1 = pd.read_csv(H1_PATH)
    df_h1["time"] = pd.to_datetime(df_h1["time"])
    df_h1.set_index("time", inplace=True)
    if df_h1.index.tzinfo is None:
        df_h1.index = df_h1.index.tz_localize("UTC")

    print(f"  M5: {len(df_m5):,} candle ({df_m5.index[0]} -> {df_m5.index[-1]})")
    print(f"  H1: {len(df_h1):,} candle ({df_h1.index[0]} -> {df_h1.index[-1]})")

    print("  Menghitung indikator M5 + H1...")
    df_m5_ind = run_all_indicators(df_m5.copy())
    df_h1_ind = run_all_indicators(df_h1.copy())

    print("  Merging H1 → M5 (backward)...")
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)

    print("  Validasi zero lookahead...")
    val = validate_no_lookahead(df_m5, n_samples=5)
    status = "✅ PASSED" if val["passed"] else "❌ FAILED"
    print(f"  {status} — {val['message']}")
    if not val["passed"]:
        raise RuntimeError("Zero lookahead validation FAILED — hentikan validasi.")

    return df_m5_ind, df_merged


# =============================================================================
# BAGIAN 1: INNER LOOP BACKTEST — dengan candle_pattern scoring
# =============================================================================

def run_backtest_with_quality(
    df_m5_ind     : pd.DataFrame,
    df_merged     : pd.DataFrame,
    enable_cp     : bool  = True,   # enable_candle_pattern
    warm_up       : int   = WARM_UP_CANDLES,
    max_candles   : int   = MAX_FORWARD_CANDLES,
    spread_pts    : float = SPREAD,
    atr_mult      : float = FASE1["atr_mult"],
    lookback      : int   = FASE1["lookback"],
    wing          : int   = FASE1["wing"],
    rrr_min       : float = FASE1["rrr_min"],
) -> pd.DataFrame:
    """
    Backtest lengkap dengan scoring candle_pattern ON atau OFF.

    Parameter enable_cp mengontrol apakah candle_pattern dihitung (True)
    atau di-set 0 (False) — tanpa mengubah trade yang di-generate.

    Return: trades_df dengan kolom score_candle_pattern dan pattern_detected.
    """
    trades             = []
    in_trade_until_idx = -1
    n_total            = len(df_merged)

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
            "volume_ratio": (float(row["volume_ratio"])
                             if "volume_ratio" in row and not pd.isna(row.get("volume_ratio"))
                             else None),
        }

        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in signals.values() if isinstance(v, (int, float))
        )
        if has_nan:
            continue

        # ENTRY DECISION — tidak berubah apapun nilai enable_cp
        decision = evaluate_entry(signals)
        if decision["keputusan"] not in ("BUY", "SELL"):
            continue

        arah     = decision["keputusan"]
        df_slice = df_m5_ind.iloc[: i + 1]

        risk = calculate_sl_tp(
            df             = df_slice,
            entry          = signals["close"],
            arah           = arah,
            rrr_min        = rrr_min,
            atr_multiplier = atr_mult,
            swing_lookback = lookback,
            swing_wing     = wing,
            tick_info      = {
                "ask": signals["close"] + spread_pts / 2,
                "bid": signals["close"] - spread_pts / 2,
            },
        )
        if not risk["valid"]:
            continue

        # Inject swing ke signals (seperti backtester.py Fase 4.3)
        sw_raw = risk.get("sl_swing_raw")
        if sw_raw is not None:
            signals["swing_low"]  = sw_raw if arah == "BUY" else None
            signals["swing_high"] = sw_raw if arah == "SELL" else None
        else:
            signals["swing_low"]  = None
            signals["swing_high"] = None

        signals["atr_14"] = risk["atr_value"]

        # QUALITY SCORING — ini satu-satunya yang dipengaruhi enable_cp
        quality = calculate_setup_quality(
            signals               = signals,
            c_h1                  = {},
            c_m5                  = {},
            c_rsi                 = {},
            df                    = df_slice,
            enable_candle_pattern = enable_cp,
        )

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

        spread_cost = spread_pts * 2
        if outcome == "TP_HIT":
            pnl_points = +jarak_tp
            pnl_net    = pnl_points - spread_cost
        elif outcome == "SL_HIT":
            pnl_points = -jarak_sl
            pnl_net    = pnl_points - spread_cost
        else:
            exit_mtm = outcome_info.get("exit_price_mtm", risk["entry"])
            raw      = exit_mtm - risk["entry"] if arah == "BUY" else risk["entry"] - exit_mtm
            pnl_points = max(raw, -jarak_sl)
            pnl_net    = pnl_points - spread_cost

        qbd = quality.get("quality_breakdown", {})
        trades.append({
            "entry_time"          : str(df_merged.index[i]),
            "outcome"             : outcome,
            "pnl_net"             : pnl_net,
            "setup_quality"       : quality["setup_quality"],
            "setup_quality_score" : quality["setup_quality_score"],
            "score_ema_gap"       : qbd.get("ema_gap",        {}).get("score"),
            "score_rsi_zone"      : qbd.get("rsi_zone",       {}).get("score"),
            "score_swing_distance": qbd.get("swing_distance", {}).get("score"),
            "score_candle_pattern": qbd.get("candle_pattern", {}).get("score"),
            "pattern_detected"    : qbd.get("candle_pattern", {}).get("pattern_detected"),
        })

        in_trade_until_idx = i + candles_held

    return pd.DataFrame(trades)


# =============================================================================
# BAGIAN 2: 7.3.2 — VERIFIKASI TRADE SET IDENTIK
# =============================================================================

def validate_on_off_identical(df_m5_ind, df_merged):
    """
    Verifikasi bahwa entry_time yang di-generate ON dan OFF identik 100%.
    Ini membuktikan tidak ada bocor dari scoring ke entry decision.
    """
    print("\n" + "=" * 65)
    print("  7.3.2 — VERIFIKASI TRADE SET ON vs OFF")
    print("=" * 65)

    t0 = time.time()
    print("  Menjalankan backtest ON...")
    df_on  = run_backtest_with_quality(df_m5_ind, df_merged, enable_cp=True)
    print("  Menjalankan backtest OFF...")
    df_off = run_backtest_with_quality(df_m5_ind, df_merged, enable_cp=False)
    elapsed = time.time() - t0

    set_on  = set(df_on["entry_time"].tolist())
    set_off = set(df_off["entry_time"].tolist())

    identik = (set_on == set_off)
    print(f"\n  Trade ON : {len(df_on):,}")
    print(f"  Trade OFF: {len(df_off):,}")
    print(f"  Set identik: {'✅ YA — tidak ada bocor ke entry logic' if identik else '❌ TIDAK — ADA BUG!'}")

    if not identik:
        only_on  = set_on  - set_off
        only_off = set_off - set_on
        if only_on:
            print(f"  Hanya di ON  ({len(only_on)}): {list(only_on)[:3]}")
        if only_off:
            print(f"  Hanya di OFF ({len(only_off)}): {list(only_off)[:3]}")

    print(f"  Waktu: {elapsed:.1f}s")
    return df_on, df_off, identik


# =============================================================================
# BAGIAN 3: 7.3.3 — ANALISIS KORELASI + REDUNDANSI
# =============================================================================

def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR correction.
    Return list[bool]: True jika hipotesis signifikan setelah koreksi.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    for k, (orig_idx, p) in enumerate(indexed):
        if p <= (k + 1) / n * alpha:
            rejected[orig_idx] = True
    # BH: jika hipotesis ke-k ditolak, semua hipotesis lebih kecil juga ditolak
    # (monotone enforcement)
    max_k_rejected = -1
    for k, (orig_idx, p) in enumerate(indexed):
        if rejected[orig_idx]:
            max_k_rejected = k
    for k, (orig_idx, p) in enumerate(indexed):
        if k <= max_k_rejected:
            rejected[orig_idx] = True
    return rejected


def analyze_correlation(df_on: pd.DataFrame):
    """
    7.3.3: Korelasi score_candle_pattern vs is_win, pnl_net,
    dan uji redundansi vs komponen existing.
    """
    print("\n" + "=" * 65)
    print("  7.3.3 — ANALISIS KORELASI & REDUNDANSI")
    print("=" * 65)

    # Filter closed trades saja (TP atau SL) untuk is_win
    closed = df_on[df_on["outcome"].isin(["TP_HIT", "SL_HIT"])].copy()
    closed["is_win"] = (closed["outcome"] == "TP_HIT").astype(int)

    all_trades = df_on.copy()
    all_trades["is_win"] = (all_trades["outcome"] == "TP_HIT").astype(int)

    print(f"\n  Dataset: {len(df_on):,} trade total, {len(closed):,} closed (TP/SL)")
    print(f"  Win rate (closed): {closed['is_win'].mean():.1%}")

    scp = closed["score_candle_pattern"].dropna()
    if scp.nunique() < 2:
        print("\n  ⚠️  score_candle_pattern: variance=0 — semua skor sama.")
        print("      Kemungkinan: tidak ada candle pattern terdeteksi dalam dataset ini.")
        print("      Kesimpulan: PERLU DATA LEBIH / REVIEW THRESHOLD PATTERN")
        return None

    print(f"\n  Distribusi score_candle_pattern (closed trades):")
    dist = scp.value_counts().sort_index()
    for score, count in dist.items():
        pct = count / len(scp) * 100
        print(f"    Skor {score}: {count:4d} trade ({pct:.1f}%)")

    # ── Kumpulkan semua p-value untuk multiple comparison correction ──────────
    all_hypotheses = []   # (label, r, p, r_sp, p_sp)
    all_p_values   = []

    def corr_pair(col_x, col_y, label, data):
        valid = data[[col_x, col_y]].dropna()
        if len(valid) < 10 or valid[col_x].nunique() < 2 or valid[col_y].nunique() < 2:
            return None
        r, p     = stats.pearsonr(valid[col_x],  valid[col_y])
        rsp, psp = stats.spearmanr(valid[col_x], valid[col_y])
        return (label, r, p, rsp, psp)

    # Hipotesis 1: vs is_win (closed)
    h1 = corr_pair("score_candle_pattern", "is_win", "vs is_win (closed)", closed)
    # Hipotesis 2: vs pnl_net (semua trade)
    h2 = corr_pair("score_candle_pattern", "pnl_net", "vs pnl_net (all)", df_on)
    # Hipotesis 3-5: vs 3 komponen existing (closed)
    h3 = corr_pair("score_candle_pattern", "score_ema_gap",        "vs score_ema_gap",       closed)
    h4 = corr_pair("score_candle_pattern", "score_rsi_zone",       "vs score_rsi_zone",      closed)
    h5 = corr_pair("score_candle_pattern", "score_swing_distance", "vs score_swing_distance",closed)

    hypotheses = [h for h in [h1, h2, h3, h4, h5] if h is not None]
    p_values   = [h[2] for h in hypotheses]  # pakai Pearson p

    # ── Bonferroni correction ─────────────────────────────────────────────────
    alpha = 0.05
    n_hyp = len(p_values)
    alpha_bonf = alpha / n_hyp if n_hyp > 0 else alpha

    # ── BH-FDR correction ────────────────────────────────────────────────────
    bh_sig = bh_fdr(p_values, alpha=alpha)

    print(f"\n  Multiple comparison: {n_hyp} hipotesis")
    print(f"  Alpha Bonferroni: {alpha_bonf:.4f} (={alpha}/{n_hyp})")
    print(f"  Alpha BH-FDR:     {alpha:.2f} (adjusted per BH)")

    print(f"\n  {'Label':<35} {'Pearson r':>10} {'p-value':>10} {'Bonf':>6} {'BH':>6} {'Sp.r':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*8}")

    for idx, (label, r, p, rsp, psp) in enumerate(hypotheses):
        sig_bonf = "✅" if p < alpha_bonf else "  "
        sig_bh   = "✅" if bh_sig[idx]    else "  "
        flag_red = " ⚠️ REDUNDAN?" if abs(r) > 0.7 and label.startswith("vs score") else ""
        print(f"  {label:<35} {r:>+10.4f} {p:>10.4f} {sig_bonf:>6} {sig_bh:>6} {rsp:>+8.4f}{flag_red}")

    # ── Breakdown per jenis pattern ────────────────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  Breakdown per jenis pattern (closed trades):")
    print(f"  {'='*65}")
    if "pattern_detected" in df_on.columns:
        pattern_col = closed["pattern_detected"].fillna("NONE")
        pat_counts  = pattern_col.value_counts()

        print(f"  {'Pattern':<30} {'N':>5} {'Win Rate':>10} {'Avg PnL':>10}")
        print(f"  {'-'*30} {'-'*5} {'-'*10} {'-'*10}")
        for pattern in pat_counts.index:
            mask = pattern_col == pattern
            n    = mask.sum()
            if n >= 3:
                wr      = closed.loc[mask, "is_win"].mean()
                avg_pnl = closed.loc[mask, "pnl_net"].mean()
                print(f"  {pattern:<30} {n:>5} {wr:>10.1%} {avg_pnl:>+10.2f}")
            else:
                print(f"  {pattern:<30} {n:>5} {'(n<3)':>10}")
    else:
        print("  Kolom pattern_detected tidak tersedia.")

    # ── Win rate per skor candle pattern ─────────────────────────────────────
    print(f"\n  Win rate per score_candle_pattern (closed):")
    print(f"  {'Skor':>5} {'N':>5} {'Win Rate':>10} {'Avg PnL':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10}")
    for score in sorted(closed["score_candle_pattern"].dropna().unique()):
        mask    = closed["score_candle_pattern"] == score
        n       = mask.sum()
        wr      = closed.loc[mask, "is_win"].mean()
        avg_pnl = closed.loc[mask, "pnl_net"].mean()
        print(f"  {int(score):>5} {n:>5} {wr:>10.1%} {avg_pnl:>+10.2f}")

    return hypotheses, p_values, alpha_bonf, bh_sig


# =============================================================================
# BAGIAN 4: 7.3.4 — WALK-FORWARD
# =============================================================================

def run_walk_forward(df_m5_ind, df_merged):
    """
    7.3.4: Walk-forward — distribusi STRONG/MODERATE/WEAK lintas fold.
    Parameter fixed (Fase 1): atr=0.9, lookback=15, wing=3, rrr=1.3.
    """
    print("\n" + "=" * 65)
    print("  7.3.4 — WALK-FORWARD (Fase 1 params, candle_pattern ON)")
    print("=" * 65)

    # Generate fold boundaries dari DatetimeIndex df_merged
    idx    = df_merged.index
    start  = idx[WARM_UP_CANDLES]
    end    = idx[-1]

    from dateutil.relativedelta import relativedelta

    folds = []
    calib_start = start
    while True:
        val_start = calib_start + relativedelta(months=WF_CALIB_MONTHS)
        val_end   = val_start   + relativedelta(months=WF_VAL_MONTHS)
        if val_end > end:
            break
        folds.append((calib_start, val_start, val_end))
        calib_start += relativedelta(months=1)

    print(f"  Fold window: {WF_CALIB_MONTHS} bulan calib + {WF_VAL_MONTHS} bulan val, geser 1 bulan")
    print(f"  Total fold: {len(folds)}")

    results = []
    for fold_idx, (cs, vs, ve) in enumerate(folds):
        # Filter df_merged ke window validasi saja
        mask_val = (df_merged.index >= vs) & (df_merged.index < ve)
        df_val   = df_merged[mask_val]

        if len(df_val) < 50:
            continue

        # Buat df_merged slice yang dimulai dari warm_up sebelum val_start
        # (butuh calib candle agar indikator converged)
        mask_full = df_merged.index < ve
        df_full_slice = df_merged[mask_full]

        trades_fold = run_backtest_with_quality(
            df_m5_ind = df_m5_ind,
            df_merged = df_full_slice,
            enable_cp = True,
            warm_up   = WARM_UP_CANDLES,
        )

        # Filter hanya trade di window validasi
        trades_fold = trades_fold[trades_fold["entry_time"] >= str(vs)]
        trades_fold = trades_fold[trades_fold["entry_time"] < str(ve)]

        if trades_fold.empty:
            n_strong = n_mod = n_weak = 0
            win_rate = float("nan")
        else:
            closed_f = trades_fold[trades_fold["outcome"].isin(["TP_HIT", "SL_HIT"])]
            win_rate = (closed_f["outcome"] == "TP_HIT").mean() if not closed_f.empty else float("nan")
            vc = trades_fold["setup_quality"].value_counts()
            n_strong = vc.get("STRONG", 0)
            n_mod    = vc.get("MODERATE", 0)
            n_weak   = vc.get("WEAK", 0)

        results.append({
            "fold"       : fold_idx + 1,
            "val_start"  : str(vs.date()),
            "val_end"    : str(ve.date()),
            "n_trades"   : len(trades_fold),
            "n_strong"   : n_strong,
            "n_moderate" : n_mod,
            "n_weak"     : n_weak,
            "win_rate"   : win_rate,
        })

    df_res = pd.DataFrame(results)

    if df_res.empty:
        print("  Tidak ada fold valid — data mungkin terlalu pendek.")
        return df_res

    print(f"\n  {'Fold':>4} {'Val Start':>12} {'Trades':>7} {'STRONG':>7} {'MODERATE':>9} {'WEAK':>6} {'WinRate':>8}")
    print(f"  {'─'*4} {'─'*12} {'─'*7} {'─'*7} {'─'*9} {'─'*6} {'─'*8}")
    for _, row in df_res.iterrows():
        wr_str = f"{row['win_rate']:.1%}" if not pd.isna(row["win_rate"]) else "  N/A"
        print(f"  {int(row['fold']):>4} {row['val_start']:>12} {int(row['n_trades']):>7} "
              f"{int(row['n_strong']):>7} {int(row['n_moderate']):>9} "
              f"{int(row['n_weak']):>6} {wr_str:>8}")

    # Statistik agregat lintas fold
    print(f"\n  Agregat lintas fold:")
    total_strong   = df_res["n_strong"].sum()
    total_moderate = df_res["n_moderate"].sum()
    total_weak     = df_res["n_weak"].sum()
    total_trades   = df_res["n_trades"].sum()
    if total_trades > 0:
        pct_s = total_strong   / total_trades * 100
        pct_m = total_moderate / total_trades * 100
        pct_w = total_weak     / total_trades * 100
        print(f"    STRONG  : {total_strong:4d} ({pct_s:.1f}%)")
        print(f"    MODERATE: {total_moderate:4d} ({pct_m:.1f}%)")
        print(f"    WEAK    : {total_weak:4d} ({pct_w:.1f}%)")

    # Cek konsistensi distribusi lintas fold (std rendah = konsisten)
    pct_strong_per_fold = (df_res["n_strong"] / df_res["n_trades"].replace(0, np.nan) * 100).dropna()
    if len(pct_strong_per_fold) >= 3:
        print(f"\n  Konsistensi %STRONG lintas fold:")
        print(f"    Mean : {pct_strong_per_fold.mean():.1f}%")
        print(f"    Std  : {pct_strong_per_fold.std():.1f}%  (rendah = konsisten)")
        print(f"    Min  : {pct_strong_per_fold.min():.1f}%")
        print(f"    Max  : {pct_strong_per_fold.max():.1f}%")

    return df_res


# =============================================================================
# BAGIAN 5: 7.3.5 — KESIMPULAN
# =============================================================================

def print_conclusion(identik, corr_result, wf_df):
    """
    7.3.5: Kesimpulan LOLOS / TIDAK LOLOS / PERLU DATA LEBIH.
    """
    print("\n" + "=" * 65)
    print("  7.3.5 — KESIMPULAN VALIDASI FASE 7")
    print("=" * 65)

    issues = []

    # Cek 1: ON/OFF identik
    if not identik:
        issues.append("❌ Trade set ON vs OFF TIDAK identik — ada bocor scoring → entry logic")
    else:
        print("  ✅ 7.3.2: Trade set ON=OFF (tidak ada bocor ke entry logic)")

    # Cek 2: Korelasi
    if corr_result is None:
        issues.append("⚠️  Korelasi tidak bisa dihitung — variance=0 pada score_candle_pattern")
        print("  ⚠️  7.3.3: Korelasi tidak bisa dihitung (variance=0)")
    else:
        hypotheses, p_values, alpha_bonf, bh_sig = corr_result

        # Hipotesis 1 = vs is_win, Hipotesis 2 = vs pnl_net
        h_iswin = hypotheses[0] if len(hypotheses) > 0 else None
        h_pnl   = hypotheses[1] if len(hypotheses) > 1 else None

        # Hipotesis 3-5 = vs komponen existing
        h_existing = hypotheses[2:] if len(hypotheses) > 2 else []

        # Signifikansi setelah koreksi (gunakan BH-FDR sebagai primer)
        if h_iswin:
            sig_iswin = bh_sig[0]
            r_iswin   = h_iswin[1]
            p_iswin   = h_iswin[2]
            if sig_iswin and r_iswin > 0:
                print(f"  ✅ 7.3.3: Korelasi vs is_win: r={r_iswin:+.4f}, p={p_iswin:.4f} (BH-sig) — sinyal positif")
            elif sig_iswin and r_iswin < 0:
                issues.append(f"⚠️  Korelasi vs is_win NEGATIF: r={r_iswin:+.4f} — pattern berlawanan arah?")
            else:
                print(f"  ℹ️  7.3.3: Korelasi vs is_win: r={r_iswin:+.4f}, p={p_iswin:.4f} (tidak signifikan setelah BH-FDR)")

        # Cek redundansi
        redundan = []
        for idx, h in enumerate(h_existing):
            if abs(h[1]) > 0.7:
                redundan.append(h[0])

        if redundan:
            issues.append(f"⚠️  Redundansi tinggi terdeteksi: {redundan} — mirip komponen lain?")
        else:
            print(f"  ✅ 7.3.3: Tidak ada redundansi struktural dengan komponen existing (|r| < 0.7)")

    # Cek 3: Walk-forward
    if wf_df is None or wf_df.empty:
        issues.append("⚠️  Walk-forward: tidak ada fold valid")
    else:
        pct_s = (wf_df["n_strong"] / wf_df["n_trades"].replace(0, np.nan)).mean()
        std_s = (wf_df["n_strong"] / wf_df["n_trades"].replace(0, np.nan)).std()
        if std_s < 0.15:
            print(f"  ✅ 7.3.4: Walk-forward konsisten (std %STRONG: {std_s:.1%})")
        else:
            print(f"  ℹ️  7.3.4: Walk-forward kurang konsisten (std %STRONG: {std_s:.1%} — > 15%)")

    print()
    if issues:
        print("  ─── TEMUAN YANG PERLU DIPERHATIKAN ───")
        for iss in issues:
            print(f"    {iss}")
        print()

    # Kesimpulan akhir
    hard_fail = any("❌" in i for i in issues)
    soft_warn = any("⚠️" in i for i in issues) and not hard_fail

    if hard_fail:
        print("  ╔══════════════════════════════════╗")
        print("  ║  HASIL: TIDAK LOLOS              ║")
        print("  ║  (ada bug / bocor entry logic)   ║")
        print("  ╚══════════════════════════════════╝")
    elif corr_result is None:
        print("  ╔══════════════════════════════════╗")
        print("  ║  HASIL: PERLU DATA LEBIH         ║")
        print("  ║  (variance=0 di score)           ║")
        print("  ╚══════════════════════════════════╝")
    elif soft_warn:
        print("  ╔══════════════════════════════════╗")
        print("  ║  HASIL: LOLOS BERSYARAT          ║")
        print("  ║  (ada warning, cek detail)       ║")
        print("  ╚══════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════╗")
        print("  ║  HASIL: LOLOS                    ║")
        print("  ╚══════════════════════════════════╝")

    print("""
  Rekomendasi langkah selanjutnya:
    - Jika LOLOS: komponen candle_pattern bisa dipertahankan di scoring
    - Jika LOLOS BERSYARAT: tinjau warning, pertimbangkan penyederhanaan pattern
    - Jika TIDAK LOLOS: perbaiki bug yang teridentifikasi
    - Jika PERLU DATA LEBIH: tunggu lebih banyak live trade terkumpul

  Ingat: validasi ini hanya untuk SCORING (bukan entry logic).
  Implementasi ke production scoring tidak memerlukan threshold tambahan.
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  VALIDASI FASE 7 — Candlestick Pattern Component (7.3.2-7.3.5)")
    print("=" * 65)
    print(f"  Dataset: 2025-06-01 s/d 2026-07-25 (14 bulan)")
    print(f"  Parameter Fase 1: atr={FASE1['atr_mult']}, lookback={FASE1['lookback']}, "
          f"wing={FASE1['wing']}, rrr={FASE1['rrr_min']}")
    print()

    # 0. Load & prepare
    t_total = time.time()
    df_m5_ind, df_merged = load_and_prepare()
    print()

    # 7.3.2: ON vs OFF
    df_on, df_off, identik = validate_on_off_identical(df_m5_ind, df_merged)

    # 7.3.3: Korelasi & redundansi
    corr_result = analyze_correlation(df_on)

    # 7.3.4: Walk-forward
    wf_df = run_walk_forward(df_m5_ind, df_merged)

    # 7.3.5: Kesimpulan
    print_conclusion(identik, corr_result, wf_df)

    print(f"\n  Total waktu validasi: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
