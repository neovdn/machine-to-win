"""
scripts/_diag_remaining_triggers_walkforward.py
==================================================
DIAGNOSTIK FASE 1f -- Uji 3 Trigger Tersisa via Walk-Forward Langsung
=========================================================================
LATAR BELAKANG:
    Trigger pullback-to-EMA sudah diuji dan TIDAK terbukti punya edge
    konsisten di walk-forward (t-stat jauh di bawah signifikan di semua
    threshold volume). Dari 4 kandidat trigger awal, tersisa 3:
        1. Swing/S-R bounce
        2. Momentum candle pattern (engulfing, hammer/shooting star)
        3. Micro-breakout (breakout N-candle range)

    Karena backtest single-period sudah terbukti TIDAK bisa dipercaya
    sendirian (rawan overfitting -- lihat kasus pullback kemarin), script
    ini LANGSUNG menguji ketiganya via walk-forward multi-fold (metodologi
    sama dengan run_walk_forward.py), tanpa basa-basi single-period dulu.
    Kalau ada yang t-stat-nya jelas signifikan dan konsisten, BARU kita
    perdalam dengan feature analysis & threshold tuning seperti pullback.

CARA PAKAI:
    python scripts/_diag_remaining_triggers_walkforward.py
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import (
    merge_h1_to_m5, simulate_trade_outcome, compute_summary,
    WARM_UP_CANDLES, MAX_FORWARD_CANDLES, MIN_SL_DISTANCE, DEFAULT_SPREAD_PTS,
)
from engine.risk_manager import calculate_sl_tp
from scripts.run_walk_forward import generate_folds
from scripts.run_oos_validation import filter_period

RISK_PARAMS = {"atr_multiplier": 0.9, "swing_lookback": 15, "swing_wing": 3, "rrr_min": 1.3}

SWING_LOOKBACK   = 15   # sama dengan risk manager, untuk proxy proximity
SWING_PROX_ATR   = 0.5  # candidate valid kalau low/high candle dalam 0.5x ATR dari swing proxy
BREAKOUT_LOOKBACK = 10  # micro-breakout: N candle terakhir


# =============================================================================
# DEFINISI 3 TRIGGER (vectorized tagging di seluruh dataset)
# =============================================================================

def tag_swing_bounce(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Swing/S-R bounce (proxy vektorized -- SL/TP presisi tetap pakai
    find_nearest_swing() asli saat backtest, ini hanya untuk tagging cepat).

    Proxy: rolling min(low) / max(high) dari swing_lookback candle SEBELUM
    candle saat ini (exclude current candle) sebagai referensi level S/R.
    Candidate valid kalau:
        - bias H1 jelas
        - low candle saat ini (BUY) / high candle (SELL) berada dalam
          SWING_PROX_ATR x ATR dari level referensi tsb (harga sempat
          menyentuh/dekat zona itu)
        - candle close menjauh dari level dengan rejection (close jauh
          dari low candle untuk BUY, jauh dari high candle untuk SELL)
        - M5 belum searah H1 (supaya candidate baru, bukan overlap sistem sekarang)
    """
    df = df_merged.copy()
    atr = df["atr_14"]

    ref_low  = df["low"].shift(1).rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).min()
    ref_high = df["high"].shift(1).rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).max()

    bias_up   = df["trend_h1"] == "UPTREND"
    bias_down = df["trend_h1"] == "DOWNTREND"

    near_swing_low  = (df["low"]  - ref_low).abs()  <= (SWING_PROX_ATR * atr)
    near_swing_high = (df["high"] - ref_high).abs() <= (SWING_PROX_ATR * atr)

    # rejection: close jauh dari extreme candle (menandakan mantul, bukan cuma numpang)
    rejection_buy  = (df["close"] - df["low"])  >= (0.4 * atr)
    rejection_sell = (df["high"]  - df["close"]) >= (0.4 * atr)

    m5_not_up   = df["trend"] != "UPTREND"
    m5_not_down = df["trend"] != "DOWNTREND"

    cand_buy  = bias_up   & near_swing_low  & rejection_buy  & m5_not_up
    cand_sell = bias_down & near_swing_high & rejection_sell & m5_not_down

    df["is_candidate"]   = cand_buy | cand_sell
    df["candidate_arah"] = np.select([cand_buy, cand_sell], ["BUY", "SELL"], default=None)
    return df


def tag_candle_pattern(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Bullish/bearish engulfing + hammer/shooting star, searah bias H1."""
    df = df_merged.copy()

    op, cl, hi, lo = df["open"], df["close"], df["high"], df["low"]
    prev_op, prev_cl = op.shift(1), cl.shift(1)

    body  = (cl - op).abs()
    rng   = (hi - lo)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)

    prev_bearish = prev_cl < prev_op
    prev_bullish = prev_cl > prev_op
    curr_bullish = cl > op
    curr_bearish = cl < op

    bullish_engulf = prev_bearish & curr_bullish & (op <= prev_cl) & (cl >= prev_op)
    bearish_engulf = prev_bullish & curr_bearish & (op >= prev_cl) & (cl <= prev_op)

    hammer = (rng > 0) & (lower_wick >= 2 * body) & (upper_wick <= 0.3 * rng) & (body > 0)
    shooting_star = (rng > 0) & (upper_wick >= 2 * body) & (lower_wick <= 0.3 * rng) & (body > 0)

    bias_up   = df["trend_h1"] == "UPTREND"
    bias_down = df["trend_h1"] == "DOWNTREND"
    m5_not_up   = df["trend"] != "UPTREND"
    m5_not_down = df["trend"] != "DOWNTREND"

    cand_buy  = bias_up   & (bullish_engulf | hammer)        & m5_not_up
    cand_sell = bias_down & (bearish_engulf | shooting_star) & m5_not_down

    df["is_candidate"]   = cand_buy | cand_sell
    df["candidate_arah"] = np.select([cand_buy, cand_sell], ["BUY", "SELL"], default=None)
    return df


def tag_micro_breakout(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Breakout dari range N-candle terakhir, searah bias H1."""
    df = df_merged.copy()

    recent_high = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).max()
    recent_low  = df["low"].shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).min()

    bias_up   = df["trend_h1"] == "UPTREND"
    bias_down = df["trend_h1"] == "DOWNTREND"
    m5_not_up   = df["trend"] != "UPTREND"
    m5_not_down = df["trend"] != "DOWNTREND"

    cand_buy  = bias_up   & (df["close"] > recent_high) & m5_not_up
    cand_sell = bias_down & (df["close"] < recent_low)  & m5_not_down

    df["is_candidate"]   = cand_buy | cand_sell
    df["candidate_arah"] = np.select([cand_buy, cand_sell], ["BUY", "SELL"], default=None)
    return df


# =============================================================================
# BACKTEST GENERIK (index-safe, sudah pakai fix dari kasus pullback kemarin)
# =============================================================================

def backtest_candidates(df_m5_ind: pd.DataFrame, df_merged: pd.DataFrame) -> tuple:
    trades = []
    in_trade_until_idx = -1
    candidate_rows = df_merged[df_merged["is_candidate"]]

    for idx in candidate_rows.index:
        i = df_m5_ind.index.get_loc(idx)      # posisi relatif dataset PENUH
        row = df_merged.loc[idx]               # row via .loc (aman meski df_merged dipotong)

        if i <= in_trade_until_idx or i < WARM_UP_CANDLES:
            continue

        arah = row["candidate_arah"]
        if arah not in ("BUY", "SELL"):
            continue

        df_slice = df_m5_ind.iloc[: i + 1]
        entry_price = float(row["close"])

        risk = calculate_sl_tp(
            df=df_slice, entry=entry_price, arah=arah, profile="scalp_m5",
            rrr_min=RISK_PARAMS["rrr_min"], atr_multiplier=RISK_PARAMS["atr_multiplier"],
            swing_lookback=RISK_PARAMS["swing_lookback"], swing_wing=RISK_PARAMS["swing_wing"],
            tick_info={"ask": entry_price + DEFAULT_SPREAD_PTS/2, "bid": entry_price - DEFAULT_SPREAD_PTS/2},
        )
        if not risk["valid"]:
            continue

        sl, tp = risk["sl"], risk["tp"]
        jarak_sl, jarak_tp = risk["jarak_sl"], risk["jarak_tp"]
        if jarak_sl < MIN_SL_DISTANCE:
            continue

        outcome_info = simulate_trade_outcome(
            df_m5_full=df_m5_ind, entry_idx=i, entry=risk["entry"],
            sl=sl, tp=tp, max_candles=MAX_FORWARD_CANDLES,
        )
        outcome, candles_held = outcome_info["outcome"], outcome_info["candles_held"]
        ambiguous = outcome_info["ambiguous_candle"]
        spread_cost_total = DEFAULT_SPREAD_PTS * 2

        if outcome == "TP_HIT":
            rrr_realized = risk.get("rrr_after_spread") or risk["rrr"]
            pnl_points = +jarak_tp
        elif outcome == "SL_HIT":
            rrr_realized = -1.0
            pnl_points = -jarak_sl
        else:
            exit_price_mtm = outcome_info.get("exit_price_mtm", risk["entry"])
            pnl_raw = (exit_price_mtm - risk["entry"]) if arah == "BUY" else (risk["entry"] - exit_price_mtm)
            pnl_points = max(pnl_raw, -jarak_sl)
            rrr_realized = round(pnl_points / jarak_sl, 4) if jarak_sl > 0 else 0.0
        pnl_net = pnl_points - spread_cost_total

        trades.append({
            "entry_time": str(idx), "direction": arah, "outcome": outcome,
            "candles_held": candles_held, "rrr_realized": rrr_realized,
            "pnl_points": pnl_points, "pnl_net": pnl_net, "sl_method": risk["sl_method"],
            "ambiguous_candle": ambiguous, "spread_pts": DEFAULT_SPREAD_PTS,
        })
        in_trade_until_idx = i + candles_held

    if not trades:
        return pd.DataFrame(), compute_summary(pd.DataFrame())
    trades_df = pd.DataFrame(trades)
    return trades_df, compute_summary(trades_df)


# =============================================================================
# WALK-FORWARD RUNNER
# =============================================================================

def run_walkforward(label, df_m5_ind_full, df_tagged_full, folds):
    print("\n" + "=" * 78)
    print(f"  TRIGGER: {label}")
    print("=" * 78)
    print(f"\n  {'Fold':<6} {'Val Period':<24} {'Trades':>7} {'WinRate':>9} {'AvgRRR':>9} {'PnL Net':>10}")
    print(f"  {'-'*6} {'-'*24} {'-'*7} {'-'*9} {'-'*9} {'-'*10}")

    fold_results = []
    for f in folds:
        val_start, val_end = f["val_start"], f["val_end"]
        df_m5_val   = filter_period(df_m5_ind_full, val_start, val_end)
        df_tagged_v = filter_period(df_tagged_full,  val_start, val_end)

        if len(df_m5_val) < WARM_UP_CANDLES + 20:
            print(f"  {f['fold']:<6} {val_start} -> {val_end:<12} SKIP (data kurang)")
            continue

        trades_df, summary = backtest_candidates(df_m5_ind_full, df_tagged_v)

        n   = summary.get("total_trades", 0)
        wr  = summary.get("win_rate")
        rrr = summary.get("avg_rrr_realized")
        pnl = summary.get("total_pnl_net")

        wr_s  = f"{wr*100:.1f}%" if wr is not None else "N/A"
        rrr_s = f"{rrr:+.4f}" if rrr is not None else "N/A"
        pnl_s = f"{pnl:+.1f}" if pnl is not None else "N/A"

        print(f"  {f['fold']:<6} {val_start} -> {val_end:<12} {n:>7} {wr_s:>9} {rrr_s:>9} {pnl_s:>10}")

        if n > 0:
            fold_results.append({"trades": n, "win_rate": wr, "avg_rrr": rrr, "pnl_net": pnl})

    if not fold_results:
        print("\n  Tidak ada fold dengan trade valid.")
        return None

    pnls = [r["pnl_net"] for r in fold_results if r["pnl_net"] is not None]
    rrrs = [r["avg_rrr"] for r in fold_results if r["avg_rrr"] is not None]
    n_folds = len(pnls)
    n_pos   = sum(1 for p in pnls if p > 0)
    mean_pnl = np.mean(pnls) if pnls else 0
    std_pnl  = np.std(pnls, ddof=1) if len(pnls) > 1 else 0
    t_stat   = (mean_pnl / (std_pnl / np.sqrt(n_folds))) if std_pnl > 0 else 0
    mean_rrr = np.mean(rrrs) if rrrs else None
    total_trades = sum(r["trades"] for r in fold_results)

    print(f"\n  Ringkasan {n_folds} fold valid:")
    print(f"    Total trade seluruh fold : {total_trades}")
    print(f"    Fold PnL positif         : {n_pos}/{n_folds} ({n_pos/n_folds*100:.0f}%)")
    print(f"    Mean PnL/fold            : {mean_pnl:+.2f}  (std={std_pnl:.2f})")
    print(f"    t-stat                   : {t_stat:+.3f}")
    if mean_rrr is not None:
        print(f"    Mean AvgRRR/fold         : {mean_rrr:+.4f}")

    return {
        "label": label, "n_folds": n_folds, "n_pos": n_pos,
        "mean_pnl": mean_pnl, "std_pnl": std_pnl, "t_stat": t_stat,
        "mean_rrr": mean_rrr, "total_trades": total_trades,
    }


def main():
    print("=" * 78)
    print("  DIAGNOSTIK FASE 1f -- 3 TRIGGER TERSISA VIA WALK-FORWARD LANGSUNG")
    print("=" * 78)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")
    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"\n  Dataset extended tidak ditemukan:\n    {m5_path}\n    {h1_path}")
        sys.exit(1)

    print("\n-> Loading & menghitung indikator (satu kali)...")
    df_m5_full = load_candles_csv(m5_path)
    df_h1_full = load_candles_csv(h1_path)
    df_m5_ind_full = run_all_indicators(df_m5_full.copy())
    df_h1_ind_full = run_all_indicators(df_h1_full.copy())
    df_merged_full = merge_h1_to_m5(df_m5_ind_full, df_h1_ind_full)

    folds = generate_folds(
        data_start = df_m5_full.index[0].strftime("%Y-%m-%d"),
        data_end   = df_m5_full.index[-1].strftime("%Y-%m-%d"),
        calib_months = 3, val_months = 1,
    )
    print(f"-> {len(folds)} fold dihasilkan.")

    triggers = [
        ("SWING/S-R BOUNCE",       tag_swing_bounce),
        ("MOMENTUM CANDLE PATTERN", tag_candle_pattern),
        ("MICRO-BREAKOUT",          tag_micro_breakout),
    ]

    summary_all = []
    for label, tag_fn in triggers:
        print(f"\n-> Menandai candidate untuk: {label}...")
        df_tagged = tag_fn(df_merged_full)
        n_total_candidates = int(df_tagged["is_candidate"].sum())
        print(f"   Total candidate mentah (sebelum position blocking): {n_total_candidates:,}")

        result = run_walkforward(label, df_m5_ind_full, df_tagged, folds)
        if result:
            summary_all.append(result)

    # ── Tabel perbandingan akhir ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  PERBANDINGAN AKHIR KETIGA TRIGGER")
    print("=" * 78)
    print(f"\n  {'Trigger':<28} {'Trades':>8} {'Fold+%':>8} {'Mean PnL':>10} {'Std':>9} {'t-stat':>8} {'MeanRRR':>9}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*9}")
    for r in summary_all:
        pos_pct = r["n_pos"] / r["n_folds"] * 100
        rrr_s = f"{r['mean_rrr']:+.4f}" if r["mean_rrr"] is not None else "N/A"
        print(f"  {r['label']:<28} {r['total_trades']:>8} {pos_pct:>7.0f}% {r['mean_pnl']:>+10.2f} "
              f"{r['std_pnl']:>9.2f} {r['t_stat']:>+8.3f} {rrr_s:>9}")

    print("""
  Patokan baca t-stat (kasar): |t| >= ~2.0 mulai layak dianggap signifikan
  untuk n=11 fold. Kalau SEMUA trigger di bawah itu (seperti pullback
  kemarin), itu artinya redesain trigger M5 belum menemukan edge tambahan
  yang robust -- dan itu jawaban valid: sistem sekarang sudah cukup
  well-calibrated relatif terhadap alternatif yang kita uji.

  Kirim balik seluruh output (termasuk tabel per-fold tiap trigger) untuk
  kita putuskan bersama langkah selanjutnya.
""")


if __name__ == "__main__":
    main()