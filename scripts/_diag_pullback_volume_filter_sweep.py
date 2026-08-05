"""
scripts/_diag_pullback_volume_filter_sweep.py
================================================
DIAGNOSTIK FASE 1d -- Sweep Threshold Volume Ratio untuk Trigger Pullback
============================================================================
LATAR BELAKANG:
    _diag_pullback_feature_analysis.py menemukan bahwa volume_ratio adalah
    SATU-SATUNYA fitur yang konsisten signifikan (Pearson, Spearman, tertile
    t-test semua sepakat) dalam membedakan trade pullback yang untung vs rugi.
    Tertile atas rata-rata pnl_net POSITIF (+0.294), tertile bawah sangat
    negatif (-1.053).

    TAPI analisis tertile itu post-hoc (filter setelah backtest selesai) --
    tidak proper karena tidak memperhitungkan efek position-blocking yang
    berubah kalau candidate volume rendah di-skip SEBELUM decision diambil
    (candidate berikutnya yang sebelumnya "terhalang" trade lain jadi punya
    kesempatan masuk).

TUJUAN SCRIPT INI:
    Terapkan filter volume_ratio >= threshold LANGSUNG DI DALAM LOOP backtest
    (sebelum entry diambil, sebelum position blocking dihitung) -- persis
    seperti cara volume filter bekerja di rule_engine.py yang sudah ada
    (_check_volume_participation). Uji beberapa threshold untuk melihat
    trade-off frekuensi vs profitabilitas yang SEBENARNYA.

THRESHOLD YANG DIUJI:
    Termasuk threshold yang SUDAH ada konstanta-nya di rule_engine.py
    (VOLUME_RATIO_HIGH_THRESHOLD = 1.278, dari Q75 M5) sebagai referensi,
    plus beberapa titik lain untuk melihat kurva trade-off.

CARA PAKAI:
    python scripts/_diag_pullback_volume_filter_sweep.py
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
    merge_h1_to_m5,
    simulate_trade_outcome,
    compute_summary,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE,
    DEFAULT_SPREAD_PTS,
)
from engine.risk_manager import calculate_sl_tp
from scripts._diag_pullback_candidates import find_pullback_candidates, RISK_PARAMS

CHOSEN_EMA_PROXIMITY_PCT = 0.05

# Threshold volume_ratio yang diuji -- termasuk 0.0 (tanpa filter, baseline
# pembanding dari script sebelumnya) dan 1.278 (konstanta existing di rule_engine)
VOLUME_THRESHOLD_CANDIDATES = [0.0, 0.9, 1.0, 1.1, 1.2, 1.278, 1.4]


def backtest_with_volume_filter(df_m5_ind: pd.DataFrame, df_merged: pd.DataFrame,
                                  volume_min: float) -> tuple:
    """
    Sama seperti backtest_candidates() sebelumnya, tapi candidate dengan
    volume_ratio < volume_min di-SKIP SEBELUM position blocking dicek --
    sehingga candidate berikutnya tetap punya kesempatan wajar untuk masuk.
    """
    trades = []
    in_trade_until_idx = -1
    candidate_rows = df_merged[df_merged["is_candidate"]]

    for idx in candidate_rows.index:
        # PENTING: i harus posisi di df_m5_ind (dataset yang dipakai untuk
        # slicing ATR/swing dan forward-scan simulate_trade_outcome), BUKAN
        # posisi di df_merged -- karena df_merged bisa sudah dipotong per-fold
        # (walk-forward) sementara df_m5_ind tetap dataset penuh (perlu histori
        # lengkap untuk konteks indikator & forward window yang benar).
        # Row tetap diambil dari df_merged, tapi pakai .loc (by timestamp),
        # bukan .iloc (by posisi) -- supaya benar walau df_merged sudah dipotong.
        i = df_m5_ind.index.get_loc(idx)
        row = df_merged.loc[idx]

        # ── Filter volume DULU, sebelum cek apapun yang lain ────────────────
        vr = row.get("volume_ratio")
        if pd.isna(vr) or float(vr) < volume_min:
            continue

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
            "volume_ratio_at_entry": round(float(vr), 4),
        })
        in_trade_until_idx = i + candles_held

    if not trades:
        return pd.DataFrame(), compute_summary(pd.DataFrame())
    trades_df = pd.DataFrame(trades)
    return trades_df, compute_summary(trades_df)


def main():
    print("=" * 78)
    print("  DIAGNOSTIK FASE 1d -- SWEEP THRESHOLD VOLUME UNTUK TRIGGER PULLBACK")
    print("=" * 78)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")
    if not os.path.exists(m5_path):
        m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
        h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

    print("\n-> Loading & menghitung indikator...")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)
    df_m5_ind = run_all_indicators(df_m5.copy())
    df_h1_ind = run_all_indicators(df_h1.copy())
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)
    df_tagged = find_pullback_candidates(df_merged, CHOSEN_EMA_PROXIMITY_PCT)

    print("\n" + "=" * 78)
    print("  HASIL SWEEP THRESHOLD VOLUME_RATIO")
    print("=" * 78)
    print(f"\n  {'VolMin':>8} {'Trades':>8} {'WinRate':>9} {'AvgRRR':>9} {'NoHit%':>8} {'PnL Net':>10} {'MaxDD':>9}")
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*10} {'-'*9}")

    for vmin in VOLUME_THRESHOLD_CANDIDATES:
        trades_df, summary = backtest_with_volume_filter(df_m5_ind, df_tagged, vmin)
        n = summary.get("total_trades", 0)
        wr = summary.get("win_rate")
        rrr = summary.get("avg_rrr_realized")
        nhr = summary.get("no_hit_rate")
        pnl = summary.get("total_pnl_net")
        mdd = summary.get("max_drawdown_net")

        wr_s = f"{wr*100:.1f}%" if wr is not None else "N/A"
        rrr_s = f"{rrr:+.4f}" if rrr is not None else "N/A"
        nhr_s = f"{(nhr or 0)*100:.1f}%"
        pnl_s = f"{pnl:+.1f}" if pnl is not None else "N/A"
        mdd_s = f"{mdd:+.1f}" if mdd is not None else "N/A"
        flag = "  <- tanpa filter (baseline)" if vmin == 0.0 else ("  <- konstanta HIGH existing" if vmin == 1.278 else "")

        print(f"  {vmin:>7.2f} {n:>8,} {wr_s:>9} {rrr_s:>9} {nhr_s:>8} {pnl_s:>10} {mdd_s:>9}{flag}")

    print("\n" + "=" * 78)
    print("  CARA BACA & LANGKAH SELANJUTNYA")
    print("=" * 78)
    print("""
  Cari threshold di mana PnL Net berbalik POSITIF dan AvgRRR > 0, TAPI jumlah
  trade masih cukup besar (jangan pilih threshold yang cuma menyisakan
  segelintir trade -- itu overfitting terhadap sampel kecil, bukan edge nyata).

  Kirim balik tabel ini. Kalau ada threshold yang jelas profitable dengan
  jumlah trade masih memadai (>=100-150 di rentang data ini), langkah
  berikutnya: validasi threshold itu dengan WALK-FORWARD (seperti kamu
  kalibrasi h1_min_ema_gap_pct dulu) sebelum kita integrasikan ke
  rule_engine.py sebagai trigger resmi kedua.
""")


if __name__ == "__main__":
    main()