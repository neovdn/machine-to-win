"""
scripts/_diag_pullback_candidates.py
=====================================
DIAGNOSTIK FASE 1 -- Redesain Entry M5 untuk Scalping
======================================================
TUJUAN:
    Mengukur seberapa sering situasi berikut muncul di data historis:
        - H1 punya bias jelas (UPTREND/DOWNTREND, dari detect_bias_h1)
        - M5 SAAT INI SIDEWAYS atau berlawanan arah (ditolak oleh rule engine sekarang)
        - Harga M5 sedang dekat EMA9/EMA21 (candidate pullback zone)
        - Candle M5 menutup dengan momentum searah bias H1 (close > open untuk BUY, dst)

    Lalu, untuk tiap candidate, simulasikan outcome-nya (pakai calculate_sl_tp +
    simulate_trade_outcome yang SAMA PERSIS dengan yang dipakai backtester utama)
    supaya hasilnya bisa dibandingkan apple-to-apple dengan baseline sistem sekarang.

INI SCRIPT EXPLORATORY -- TIDAK MENGUBAH engine/rule_engine.py SAMA SEKALI.
Tujuannya murni mengumpulkan data sebelum kita putuskan redesain apa yang divalidasi.

CARA PAKAI:
    python scripts/_diag_pullback_candidates.py

OUTPUT:
    1. Jumlah candidate pullback per bulan (frekuensi)
    2. Overlap check: berapa % candidate ini SUDAH tercakup oleh sinyal sistem
       sekarang vs benar-benar baru
    3. Backtest ringkas: win rate, avg RRR realized, no_hit rate, total pnl_net
       dari HANYA entry candidate pullback ini (tanpa filter RSI/Volume dulu,
       supaya kita lihat edge mentahnya)
    4. Breakdown per kedalaman pullback (seberapa dekat ke EMA) vs performa
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators, detect_bias_h1
from engine.backtester import (
    merge_h1_to_m5,
    simulate_trade_outcome,
    compute_summary,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE,
    DEFAULT_SPREAD_PTS,
)
from engine.risk_manager import calculate_sl_tp
from engine.rule_engine import evaluate_entry


# =============================================================================
# KONFIGURASI DIAGNOSTIK
# =============================================================================

# Seberapa dekat harga dianggap "dekat EMA" -- dalam persen dari harga.
# Contoh: 0.05% dari XAUUSD ~$3300 = sekitar $1.65
EMA_PROXIMITY_PCT_CANDIDATES = [0.03, 0.05, 0.08, 0.10]

# Pakai parameter risk manager Fase 1 (yang sudah dikalibrasi) supaya apple-to-apple
RISK_PARAMS = {
    "atr_multiplier" : 0.9,
    "swing_lookback" : 15,
    "swing_wing"     : 3,
    "rrr_min"        : 1.3,
}


def find_pullback_candidates(df_merged: pd.DataFrame, ema_proximity_pct: float) -> pd.DataFrame:
    """
    Tandai candle mana saja yang termasuk candidate pullback-to-EMA.

    KRITERIA (semua harus terpenuhi):
        1. trend_h1 (bias H1) != "SIDEWAYS"  -> ada bias arah jelas
        2. Harga (close) berada dalam jarak ema_proximity_pct dari EMA9 ATAU EMA21
           searah bias -- artinya harga baru saja pullback mendekati EMA
        3. Candle M5 saat ini menutup dengan momentum SEARAH bias H1:
              - Bias UPTREND -> close > open (bullish candle)
              - Bias DOWNTREND -> close < open (bearish candle)
        4. trend (M5 label) BUKAN searah H1 (kalau sudah searah, itu sudah
           tercakup oleh rule engine sekarang -- bukan candidate baru)

    Return:
        DataFrame df_merged dengan kolom tambahan:
            "is_candidate" : bool
            "candidate_arah" : "BUY"/"SELL"/None
            "dist_to_ema_pct" : float (jarak ke EMA terdekat searah bias, dalam %)
    """
    df = df_merged.copy()

    close = df["close"]
    ema9  = df["ema_9"]
    ema21 = df["ema_21"]
    op    = df["open"]

    dist_to_ema9_pct  = (close - ema9).abs()  / close * 100
    dist_to_ema21_pct = (close - ema21).abs() / close * 100
    dist_to_ema_min   = pd.concat([dist_to_ema9_pct, dist_to_ema21_pct], axis=1).min(axis=1)

    bias_up   = df["trend_h1"] == "UPTREND"
    bias_down = df["trend_h1"] == "DOWNTREND"

    near_ema     = dist_to_ema_min <= ema_proximity_pct
    bullish_bar  = close > op
    bearish_bar  = close < op

    # M5 belum searah H1 (kalau sudah searah, itu domain rule engine sekarang)
    m5_not_aligned_up   = df["trend"] != "UPTREND"
    m5_not_aligned_down = df["trend"] != "DOWNTREND"

    cand_buy  = bias_up   & near_ema & bullish_bar & m5_not_aligned_up
    cand_sell = bias_down & near_ema & bearish_bar & m5_not_aligned_down

    df["is_candidate"]    = cand_buy | cand_sell
    df["candidate_arah"]  = np.select([cand_buy, cand_sell], ["BUY", "SELL"], default=None)
    df["dist_to_ema_pct"] = dist_to_ema_min

    return df


def check_overlap_with_current_system(df_merged: pd.DataFrame) -> dict:
    """
    Untuk tiap candidate, cek apakah evaluate_entry() versi SEKARANG juga akan
    menghasilkan BUY/SELL di candle yang sama (overlap) atau tidak (benar-benar baru).
    """
    candidates = df_merged[df_merged["is_candidate"]]
    n_total = len(candidates)
    if n_total == 0:
        return {"n_total": 0, "n_overlap": 0, "n_new": 0}

    n_overlap = 0
    for idx, row in candidates.iterrows():
        if pd.isna(row.get("trend_h1")):
            continue
        signals = {
            "time"        : idx,
            "close"       : float(row["close"]),
            "ema_9"       : float(row["ema_9"]),
            "ema_21"      : float(row["ema_21"]),
            "rsi_14"      : float(row["rsi_14"]),
            "trend"       : str(row["trend"]),
            "ema_gap_pct" : float(row["ema_gap_pct"]),
            "trend_h1"    : str(row["trend_h1"]),
            "volume_ratio": None,
        }
        try:
            dec = evaluate_entry(signals, volume_mode="IGNORE")
            if dec["keputusan"] == row["candidate_arah"]:
                n_overlap += 1
        except Exception:
            continue

    return {
        "n_total"   : n_total,
        "n_overlap" : n_overlap,
        "n_new"     : n_total - n_overlap,
    }


def backtest_candidates(df_m5_ind: pd.DataFrame, df_merged: pd.DataFrame) -> tuple:
    """
    Jalankan simulasi entry HANYA pada candidate pullback (tanpa filter RSI/Volume),
    pakai risk manager dan simulate_trade_outcome yang identik dengan backtester utama.
    Position blocking diterapkan sama seperti run_fast_backtest (tidak overlap trade).
    """
    trades = []
    in_trade_until_idx = -1
    n_total = len(df_merged)

    candidate_rows = df_merged[df_merged["is_candidate"]]

    for idx in candidate_rows.index:
        i = df_merged.index.get_loc(idx)
        if i <= in_trade_until_idx:
            continue
        if i < WARM_UP_CANDLES:
            continue

        row  = df_merged.iloc[i]
        arah = row["candidate_arah"]
        if arah not in ("BUY", "SELL"):
            continue

        df_slice = df_m5_ind.iloc[: i + 1]
        entry_price = float(row["close"])

        risk = calculate_sl_tp(
            df             = df_slice,
            entry          = entry_price,
            arah           = arah,
            profile        = "scalp_m5",
            rrr_min        = RISK_PARAMS["rrr_min"],
            atr_multiplier = RISK_PARAMS["atr_multiplier"],
            swing_lookback = RISK_PARAMS["swing_lookback"],
            swing_wing     = RISK_PARAMS["swing_wing"],
            tick_info      = {
                "ask": entry_price + DEFAULT_SPREAD_PTS / 2,
                "bid": entry_price - DEFAULT_SPREAD_PTS / 2,
            },
        )
        if not risk["valid"]:
            continue

        sl, tp = risk["sl"], risk["tp"]
        jarak_sl, jarak_tp = risk["jarak_sl"], risk["jarak_tp"]
        if jarak_sl < MIN_SL_DISTANCE:
            continue

        outcome_info = simulate_trade_outcome(
            df_m5_full  = df_m5_ind,
            entry_idx   = i,
            entry       = risk["entry"],
            sl          = sl,
            tp          = tp,
            max_candles = MAX_FORWARD_CANDLES,
        )
        outcome      = outcome_info["outcome"]
        candles_held = outcome_info["candles_held"]
        spread_cost_total = DEFAULT_SPREAD_PTS * 2

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
            pnl_raw = (exit_price_mtm - risk["entry"]) if arah == "BUY" else (risk["entry"] - exit_price_mtm)
            pnl_points   = max(pnl_raw, -jarak_sl)
            pnl_net      = pnl_points - spread_cost_total
            rrr_realized = round(pnl_points / jarak_sl, 4) if jarak_sl > 0 else 0.0

        trades.append({
            "entry_time"       : str(df_merged.index[i]),
            "direction"        : arah,
            "outcome"          : outcome,
            "candles_held"     : candles_held,
            "rrr_realized"     : rrr_realized,
            "pnl_points"       : pnl_points,
            "pnl_net"          : pnl_net,
            "dist_to_ema_pct"  : round(float(row["dist_to_ema_pct"]), 4),
            "sl_method"        : risk["sl_method"],
            "ambiguous_candle" : outcome_info["ambiguous_candle"],
            "spread_pts"       : DEFAULT_SPREAD_PTS,
        })
        in_trade_until_idx = i + candles_held

    if not trades:
        return pd.DataFrame(), compute_summary(pd.DataFrame())

    trades_df = pd.DataFrame(trades)
    summary   = compute_summary(trades_df)
    return trades_df, summary


def main():
    print("=" * 78)
    print("  DIAGNOSTIK FASE 1 -- CANDIDATE PULLBACK-TO-EMA (M5) YANG TERLEWAT")
    print("=" * 78)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"File data historis tidak ditemukan:\n  {m5_path}\n  {h1_path}")
        print("Jika kamu punya dataset extended (Jun 2025 - Jul 2026), ganti path di atas.")
        sys.exit(1)

    print("\n-> Loading data...")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)
    print(f"   M5: {len(df_m5):,} candle ({df_m5.index[0]} -> {df_m5.index[-1]})")
    print(f"   H1: {len(df_h1):,} candle")

    print("\n-> Menghitung indikator...")
    df_m5_ind = run_all_indicators(df_m5.copy())
    df_h1_ind = run_all_indicators(df_h1.copy())
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)  # pakai default h1_min_ema_gap_pct=0.02

    print("\n" + "=" * 78)
    print("  BAGIAN A: FREKUENSI CANDIDATE PER THRESHOLD JARAK EMA")
    print("=" * 78)
    print(f"\n  {'Threshold':>10} {'N Candidate':>12} {'N Overlap':>10} {'N Baru':>10} {'% Baru':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    results_by_threshold = {}
    for pct in EMA_PROXIMITY_PCT_CANDIDATES:
        df_tagged = find_pullback_candidates(df_merged, pct)
        overlap   = check_overlap_with_current_system(df_tagged)
        pct_baru  = (overlap["n_new"] / overlap["n_total"] * 100) if overlap["n_total"] > 0 else 0
        results_by_threshold[pct] = {"df": df_tagged, "overlap": overlap}
        print(f"  {pct:>9.2f}% {overlap['n_total']:>12,} {overlap['n_overlap']:>10,} "
              f"{overlap['n_new']:>10,} {pct_baru:>7.1f}%")

    print("\n  Catatan: 'N Baru' = candidate yang TIDAK tercakup evaluate_entry() versi sekarang.")
    print("  Ini kandidat sinyal scalping yang sekarang selalu WAIT tapi berpotensi valid.")

    # ── Pilih satu threshold representatif untuk backtest detail (misal 0.05%) ──
    chosen_pct = 0.05
    print("\n" + "=" * 78)
    print(f"  BAGIAN B: BACKTEST MENTAH CANDIDATE (threshold {chosen_pct}%, TANPA filter RSI/Volume)")
    print("=" * 78)

    df_chosen = results_by_threshold[chosen_pct]["df"]
    trades_df, summary = backtest_candidates(df_m5_ind, df_chosen)

    if trades_df.empty:
        print("\n  Tidak ada trade candidate yang valid untuk threshold ini.")
    else:
        print(f"\n  Total trade candidate : {summary['total_trades']:,}")
        print(f"  TP / SL / NO_HIT      : {summary['tp_count']} / {summary['sl_count']} / {summary['no_hit_count']}")
        wr_s  = f"{summary['win_rate']*100:.1f}%" if summary['win_rate'] is not None else "N/A"
        rrr_s = f"{summary['avg_rrr_realized']:+.4f}" if summary['avg_rrr_realized'] is not None else "N/A"
        print(f"  Win Rate (closed)     : {wr_s}")
        print(f"  Avg RRR Realized      : {rrr_s}")
        print(f"  No-Hit Rate           : {(summary.get('no_hit_rate') or 0)*100:.1f}%")
        print(f"  Total PnL Net         : {summary['total_pnl_net']:+.2f}")
        print(f"  Max Drawdown Net      : {summary['max_drawdown_net']:+.2f}")
        print(f"  SL Method Breakdown   : {summary['sl_method_breakdown']}")

        print("\n  -- Breakdown performa per kedalaman pullback (dist_to_ema_pct) --")
        trades_df["ema_dist_bucket"] = pd.cut(
            trades_df["dist_to_ema_pct"],
            bins=[0, 0.02, 0.035, 0.05],
            labels=["0-0.02% (sangat dekat)", "0.02-0.035% (dekat)", "0.035-0.05% (agak jauh)"]
        )
        for bucket, grp in trades_df.groupby("ema_dist_bucket", observed=True):
            if len(grp) == 0:
                continue
            closed = grp[grp["outcome"].isin(["TP_HIT", "SL_HIT"])]
            wr = (closed["outcome"] == "TP_HIT").mean() if len(closed) > 0 else None
            wr_str = f"{wr*100:.1f}%" if wr is not None else "N/A"
            print(f"    {bucket}: n={len(grp):>4}, win_rate={wr_str}, "
                  f"avg_pnl_net={grp['pnl_net'].mean():+.3f}")

        print("\n  -- Distribusi bulanan (frekuensi per bulan) --")
        trades_df["entry_month"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("M")
        monthly = trades_df.groupby("entry_month").size()
        for month, cnt in monthly.items():
            print(f"    {month}: {cnt} trade")

    # ── Bandingkan dengan baseline sistem sekarang (dari test_phase_0_baseline_consistency) ──
    print("\n" + "=" * 78)
    print("  BAGIAN C: KONTEKS PEMBANDING (baseline sistem SEKARANG, dari test suite)")
    print("=" * 78)
    print("""
  Baseline resmi (tests/test_backtester.py::test_phase_0_baseline_consistency):
    total_trades=257, win_rate=40.8%, avg_rrr_realized=+0.20, no_hit_rate=21.8%
    total_pnl_net=+2117.76, max_drawdown_net=-689.48

  Bandingkan angka backtest candidate pullback di atas dengan baseline ini:
    - Apakah win_rate/avg_rrr candidate pullback lebih baik, sebanding, atau lebih buruk?
    - Apakah frekuensinya (total_trades) menambah signifikan jika DIGABUNG ke sistem existing?
""")

    print("=" * 78)
    print("  SELESAI -- salin seluruh output di atas dan kirim balik untuk dianalisis bersama.")
    print("=" * 78)


if __name__ == "__main__":
    main()