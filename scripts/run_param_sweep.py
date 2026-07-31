"""
scripts/run_param_sweep.py
==========================
Script grid search parameter sweep TEROPTIMASI untuk kalibrasi Risk Management Fase 1.

OPTIMASI KINERJA:
  Indikator M5 & H1 + merge_asof + validate_no_lookahead dihitung 1x di awal O(n),
  sehingga 192 kombinasi grid search berjalan murni di memori dalam < 30 detik.
"""

import os
import sys
import itertools
import time
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.rule_engine import evaluate_entry
from engine.risk_manager import calculate_sl_tp
from engine.backtester import (
    merge_h1_to_m5,
    validate_no_lookahead,
    simulate_trade_outcome,
    compute_summary,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE,
    DEFAULT_SPREAD_PTS,
)


def run_fast_backtest(
    df_m5_ind: pd.DataFrame,
    df_merged: pd.DataFrame,
    atr_mult: float,
    lookback: int,
    wing: int,
    rrr_min: float,
    spread_pts: float = DEFAULT_SPREAD_PTS,
    max_candles: int = MAX_FORWARD_CANDLES,
    warm_up: int = WARM_UP_CANDLES,
    volume_mode: str = "FILTER",
) -> tuple:
    """
    Eksekusi backtest super cepat tanpa menghitung ulang indikator.
    """
    trades = []
    in_trade_until_idx = -1
    n_total = len(df_merged)

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
            "volume_ratio": float(row["volume_ratio"]) if "volume_ratio" in row and not pd.isna(row.get("volume_ratio")) else None,
        }

        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in signals.values()
            if isinstance(v, (int, float))
        )
        if has_nan:
            continue

        decision = evaluate_entry(signals, volume_mode=volume_mode)
        if decision["keputusan"] not in ("BUY", "SELL"):
            continue

        arah = decision["keputusan"]
        df_slice = df_m5_ind.iloc[: i + 1]

        risk = calculate_sl_tp(
            df                  = df_slice,
            entry               = signals["close"],
            arah                = arah,
            profile             = "scalp_m5",
            rrr_min             = rrr_min,
            atr_multiplier      = atr_mult,
            swing_lookback      = lookback,
            swing_wing          = wing,
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
        ambiguous    = outcome_info["ambiguous_candle"]

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

        else:  # NO_HIT
            exit_price_mtm = outcome_info.get("exit_price_mtm", risk["entry"])
            if arah == "BUY":
                pnl_raw = exit_price_mtm - risk["entry"]
            else:
                pnl_raw = risk["entry"] - exit_price_mtm

            pnl_points   = max(pnl_raw, -jarak_sl)
            pnl_net      = pnl_points - spread_cost_total
            rrr_realized = round(pnl_points / jarak_sl, 4) if jarak_sl > 0 else 0.0
            pnl_type     = "MTM"

        trades.append({
            "entry_time"       : str(df_merged.index[i]),
            "exit_time"        : outcome_info["exit_time"],
            "direction"        : arah,
            "entry_price"      : risk["entry"],
            "sl"               : sl,
            "tp"               : tp,
            "sl_method"        : risk["sl_method"],
            "sl_swing_clamped" : risk.get("sl_swing_clamped", False),
            "clamp_reason"     : risk.get("clamp_reason"),
            "atr_value"        : risk["atr_value"],
            "outcome"          : outcome,
            "candles_held"     : candles_held,
            "rrr_planned"      : risk["rrr"],
            "rrr_realized"     : rrr_realized,
            "rrr_after_spread" : risk.get("rrr_after_spread"),
            "spread_pts"       : spread_pts,
            "jarak_sl"         : jarak_sl,
            "jarak_tp"         : jarak_tp,
            "pnl_points"       : pnl_points,
            "pnl_net"          : pnl_net,
            "pnl_type"         : pnl_type,
            "ambiguous_candle" : ambiguous,
            "trend_m5"         : signals["trend"],
            "trend_h1"         : signals["trend_h1"],
            "rsi_at_entry"     : round(signals["rsi_14"], 2),
            "ema_gap_pct"      : round(signals["ema_gap_pct"], 4),
        })

        in_trade_until_idx = i + candles_held

    if not trades:
        return pd.DataFrame(), compute_summary(pd.DataFrame())

    trades_df = pd.DataFrame(trades)
    summary   = compute_summary(trades_df)
    return trades_df, summary


def main():
    print("=" * 70)
    print("  PARAMETER SWEEP GRID SEARCH — FASE 1 RISK MANAGEMENT CALIBRATION")
    print("=" * 70)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"❌ File data historis tidak ditemukan di {m5_path} / {h1_path}")
        sys.exit(1)

    print("-> Loading data M5 dan H1 dari CSV...")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)

    print("-> Menghitung indikator M5 dan H1 (1x di awal)...")
    df_m5_ind = run_all_indicators(df_m5.copy())
    df_h1_ind = run_all_indicators(df_h1.copy())

    print("-> Merging H1 bias ke M5 (merge_asof backward)...")
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)

    print("-> Validasi zero look-ahead...")
    val = validate_no_lookahead(df_m5, n_samples=5)
    if not val["passed"]:
        raise RuntimeError("Look-ahead validation failed!")
    print(f"   {val['message']}")

    # Grid search space
    atr_mults = [0.7, 0.9, 1.1, 1.3]
    lookbacks = [10, 15, 20, 30]
    wings     = [2, 3, 4]
    rrr_mins  = [1.2, 1.3, 1.5, 1.8]

    grid = list(itertools.product(atr_mults, lookbacks, wings, rrr_mins))
    total_combos = len(grid)

    print(f"\n-> Memulai Grid Search 192 kombinasi parameter...")
    print(f"   ATR Multipliers : {atr_mults}")
    print(f"   Swing Lookbacks : {lookbacks}")
    print(f"   Swing Wings     : {wings}")
    print(f"   Min RRRs        : {rrr_mins}")
    print("-" * 70)

    results = []
    start_time = time.time()

    for idx, (atr_mult, lookback, wing, rrr_min) in enumerate(grid, 1):
        trades_df, summary = run_fast_backtest(
            df_m5_ind=df_m5_ind,
            df_merged=df_merged,
            atr_mult=atr_mult,
            lookback=lookback,
            wing=wing,
            rrr_min=rrr_min,
            spread_pts=0.50,
            max_candles=288,
            warm_up=100,
        )

        sl_breakdown = summary.get("sl_method_breakdown", {})
        sl_swing_n   = sl_breakdown.get("SWING", 0)
        sl_atr_n     = sl_breakdown.get("ATR", 0)
        clamped_n    = int(trades_df["sl_swing_clamped"].sum()) if not trades_df.empty and "sl_swing_clamped" in trades_df.columns else 0

        wr = summary.get("win_rate") or 0.0
        be_wr = 1.0 / (1.0 + rrr_min)
        theo_expectancy_r = round((wr * rrr_min) - ((1.0 - wr) * 1.0), 4)

        results.append({
            "atr_multiplier"      : atr_mult,
            "swing_lookback"      : lookback,
            "swing_wing"          : wing,
            "rrr_min"             : rrr_min,
            "be_win_rate_pct"     : round(be_wr * 100, 2),
            "total_trades"        : summary["total_trades"],
            "tp_count"            : summary["tp_count"],
            "sl_count"            : summary["sl_count"],
            "no_hit_count"        : summary["no_hit_count"],
            "win_rate_pct"        : round(wr * 100, 2),
            "no_hit_rate_pct"     : round((summary.get("no_hit_rate") or 0.0) * 100, 2),
            "avg_rrr_realized"    : summary.get("avg_rrr_realized"),
            "avg_rrr_realized_all": summary.get("avg_rrr_realized_all"),
            "avg_candles_held"    : summary.get("avg_candles_held"),
            "avg_candles_held_all": summary.get("avg_candles_held_all"),
            "total_pnl_net"       : summary.get("total_pnl_net"),
            "max_drawdown_net"    : summary.get("max_drawdown_net"),
            "theo_expectancy_r"   : theo_expectancy_r,
            "sl_swing_count"      : sl_swing_n,
            "sl_atr_count"        : sl_atr_n,
            "sl_clamped_count"    : clamped_n,
        })

    elapsed_total = time.time() - start_time
    print(f"✅ Grid search 192 kombinasi selesai dalam {elapsed_total:.2f} detik.")

    res_df = pd.DataFrame(results)

    # Sort berdasarkan Total PnL Net descending
    res_df = res_df.sort_values(by="total_pnl_net", ascending=False).reset_index(drop=True)

    out_dir = os.path.join(ROOT_DIR, "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "param_sweep_results.csv")
    res_df.to_csv(out_csv, index=False)

    print(f"\n📂 Hasil lengkap 192 kombinasi disimpan ke:\n   {out_csv}\n")

    # Print Top 15 Kombinasi
    print("=" * 95)
    print("  TOP 15 KOMBINASI PARAMETER TERBAIK (Berdasarkan Total P&L Net Spread)")
    print("=" * 95)
    header = (
        f"{'Rank':<5} | {'ATR':<5} | {'Look':<4} | {'Wing':<4} | {'RRR':<4} | "
        f"{'WinRate':<7} | {'NoHit%':<6} | {'AvgRRR':<6} | {'AvgCand':<7} | {'Net PnL':<9} | {'MaxDD':<8}"
    )
    print(header)
    print("-" * 95)

    for i in range(min(15, len(res_df))):
        row = res_df.iloc[i]
        print(
            f"{i+1:<5} | {row['atr_multiplier']:<5.1f} | {int(row['swing_lookback']):<4} | "
            f"{int(row['swing_wing']):<4} | {row['rrr_min']:<4.1f} | {row['win_rate_pct']:>6.1f}% | "
            f"{row['no_hit_rate_pct']:>5.1f}% | {row['avg_rrr_realized']:>+6.2f} | "
            f"{row['avg_candles_held']:>6.1f} | {row['total_pnl_net']:>+9.1f} | {row['max_drawdown_net']:>8.1f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
