"""
scripts/_diag_pullback_feature_analysis.py
============================================
DIAGNOSTIK FASE 1c -- Feature Analysis pada Candidate Pullback Mentah
=======================================================================
LATAR BELAKANG:
    _diag_pullback_candidates.py menunjukkan bahwa trigger pullback-to-EMA
    mentah (hanya syarat: dekat EMA + candle momentum searah bias) itu RUGI
    (avg_rrr_realized -0.0409, total pnl -927.44). Definisi trigger-nya
    terlalu lemah -- menangkap banyak false bounce.

TUJUAN SCRIPT INI:
    Daripada menebak filter tambahan, kita ukur dulu: di antara 1.982 candidate
    mentah tadi, fitur APA yang secara STATISTIK membedakan trade yang untung
    vs rugi. Metodologinya SAMA PERSIS dengan analyze_quality_score_phase43.py
    (t-test, korelasi Pearson/Spearman) -- supaya keputusan filter tambahan
    berbasis bukti, bukan feeling.

FITUR YANG DIUKUR DI TITIK ENTRY:
    1. body_pct_atr     : ukuran body candle (|close-open|) relatif terhadap ATR.
                          Hipotesis: bounce candle yang "meyakinkan" (body besar)
                          lebih reliable daripada body kecil/doji.
    2. dist_to_swing_pct: jarak candle ke swing low/high terdekat SEARAH bias
                          (pakai find_nearest_swing yang sama dengan risk_manager),
                          dinormalisasi ke ATR. Hipotesis: bounce yang terjadi
                          dekat swing structure (bukan cuma dekat EMA) lebih valid.
    3. rsi_at_entry     : RSI di titik entry.
                          Hipotesis: RSI yang TIDAK mendukung arah (misal BUY saat
                          RSI < 40) menandakan bounce palsu / melawan momentum besar.
    4. volume_ratio     : partisipasi volume di candle entry.
                          Hipotesis: bounce dengan volume rendah = kurang meyakinkan.
    5. wick_rejection_ratio : untuk arah BUY, seberapa panjang lower wick relatif
                          terhadap total range candle (menandakan penolakan harga
                          dari bawah). Untuk SELL, upper wick. Hipotesis: rejection
                          wick panjang = bounce lebih valid daripada sekadar close>open.

CARA PAKAI:
    python scripts/_diag_pullback_feature_analysis.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import (
    merge_h1_to_m5,
    simulate_trade_outcome,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    MIN_SL_DISTANCE,
    DEFAULT_SPREAD_PTS,
)
from engine.risk_manager import calculate_sl_tp, find_nearest_swing

# Reuse fungsi candidate finder dari script sebelumnya
from scripts._diag_pullback_candidates import find_pullback_candidates, RISK_PARAMS


CHOSEN_EMA_PROXIMITY_PCT = 0.05


def backtest_with_features(df_m5_ind: pd.DataFrame, df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Sama seperti backtest_candidates() di script sebelumnya, tapi sekaligus
    menangkap fitur tambahan di titik entry untuk dianalisis.
    """
    trades = []
    in_trade_until_idx = -1

    candidate_rows = df_merged[df_merged["is_candidate"]]

    for idx in candidate_rows.index:
        i = df_merged.index.get_loc(idx)
        if i <= in_trade_until_idx or i < WARM_UP_CANDLES:
            continue

        row  = df_merged.iloc[i]
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
        spread_cost_total = DEFAULT_SPREAD_PTS * 2

        if outcome == "TP_HIT":
            pnl_points = +jarak_tp
        elif outcome == "SL_HIT":
            pnl_points = -jarak_sl
        else:
            exit_price_mtm = outcome_info.get("exit_price_mtm", risk["entry"])
            pnl_raw = (exit_price_mtm - risk["entry"]) if arah == "BUY" else (risk["entry"] - exit_price_mtm)
            pnl_points = max(pnl_raw, -jarak_sl)
        pnl_net = pnl_points - spread_cost_total

        # ── Fitur tambahan di titik entry ────────────────────────────────────
        atr_val = float(row.get("atr_14", np.nan))
        open_p, close_p, high_p, low_p = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])

        body_pct_atr = abs(close_p - open_p) / atr_val if atr_val and atr_val > 0 else None

        total_range = high_p - low_p
        if total_range > 0:
            if arah == "BUY":
                lower_wick = min(open_p, close_p) - low_p
                wick_rejection_ratio = lower_wick / total_range
            else:
                upper_wick = high_p - max(open_p, close_p)
                wick_rejection_ratio = upper_wick / total_range
        else:
            wick_rejection_ratio = None

        try:
            swing_ref = find_nearest_swing(df_slice, arah=arah,
                                            lookback=RISK_PARAMS["swing_lookback"],
                                            wing=RISK_PARAMS["swing_wing"])
            if swing_ref is not None and atr_val and atr_val > 0:
                dist_to_swing_pct = abs(entry_price - swing_ref) / atr_val
            else:
                dist_to_swing_pct = None
        except Exception:
            dist_to_swing_pct = None

        rsi_val = float(row.get("rsi_14", np.nan))
        vol_ratio = float(row.get("volume_ratio", np.nan)) if not pd.isna(row.get("volume_ratio")) else None

        trades.append({
            "entry_time"          : str(df_merged.index[i]),
            "direction"           : arah,
            "outcome"             : outcome,
            "pnl_net"             : pnl_net,
            "is_win"              : 1 if outcome == "TP_HIT" else 0,
            "body_pct_atr"        : body_pct_atr,
            "wick_rejection_ratio": wick_rejection_ratio,
            "dist_to_swing_atr"   : dist_to_swing_pct,
            "rsi_at_entry"        : rsi_val,
            "volume_ratio"        : vol_ratio,
            "dist_to_ema_pct"     : round(float(row["dist_to_ema_pct"]), 4),
        })
        in_trade_until_idx = i + candles_held

    return pd.DataFrame(trades)


def _sig(p):
    if p is None or np.isnan(p): return "?"
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def analyze_feature(df: pd.DataFrame, col: str, label: str):
    """Korelasi fitur vs pnl_net/is_win (closed trades), + t-test top vs bottom tertile."""
    sub = df.dropna(subset=[col, "pnl_net"])
    if len(sub) < 20:
        print(f"\n  -- {label} ({col}): data terlalu sedikit (n={len(sub)}), skip")
        return

    x = sub[col].astype(float)
    y_pnl = sub["pnl_net"].astype(float)
    y_win = sub["is_win"].astype(float)

    pr, pp = stats.pearsonr(x, y_pnl)
    sr, sp = stats.spearmanr(x, y_pnl)
    prw, ppw = stats.pearsonr(x, y_win)

    print(f"\n  -- {label} ({col}), n={len(sub)} --")
    print(f"     vs pnl_net : Pearson r={pr:+.4f} p={pp:.4f} {_sig(pp)} | Spearman r={sr:+.4f} p={sp:.4f} {_sig(sp)}")
    print(f"     vs is_win  : Pearson r={prw:+.4f} p={ppw:.4f} {_sig(ppw)}")

    # t-test: tertile atas vs bawah
    q33, q67 = x.quantile(0.33), x.quantile(0.67)
    low_grp  = sub.loc[x <= q33, "pnl_net"]
    high_grp = sub.loc[x >= q67, "pnl_net"]
    if len(low_grp) >= 10 and len(high_grp) >= 10:
        t, p = stats.ttest_ind(high_grp, low_grp, equal_var=False)
        print(f"     Tertile ATAS (n={len(high_grp)}, mean_pnl={high_grp.mean():+.3f}) vs "
              f"TERTILE BAWAH (n={len(low_grp)}, mean_pnl={low_grp.mean():+.3f}): "
              f"t={t:+.3f} p={p:.4f} {_sig(p)}")


def main():
    print("=" * 78)
    print("  DIAGNOSTIK FASE 1c -- FEATURE ANALYSIS PADA CANDIDATE PULLBACK MENTAH")
    print("=" * 78)

    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")
    if not os.path.exists(m5_path):
        # fallback ke dataset extended jika ada
        m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2025-06-01_2026-07-25.csv")
        h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2025-06-01_2026-07-25.csv")

    print("\n-> Loading & menghitung indikator...")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)
    df_m5_ind = run_all_indicators(df_m5.copy())
    df_h1_ind = run_all_indicators(df_h1.copy())
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)

    print("-> Menandai candidate pullback...")
    df_tagged = find_pullback_candidates(df_merged, CHOSEN_EMA_PROXIMITY_PCT)

    print("-> Menjalankan backtest + menangkap fitur di titik entry...")
    trades_df = backtest_with_features(df_m5_ind, df_tagged)

    if trades_df.empty:
        print("Tidak ada trade. Berhenti.")
        sys.exit(1)

    closed_df = trades_df[trades_df["outcome"].isin(["TP_HIT", "SL_HIT"])].copy()
    print(f"\n  Total trade: {len(trades_df)} | Closed (TP+SL): {len(closed_df)}")
    print(f"  Overall win rate: {closed_df['is_win'].mean()*100:.1f}%")
    print(f"  Overall avg pnl_net: {trades_df['pnl_net'].mean():+.4f}")

    out_dir = os.path.join(ROOT_DIR, "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pullback_candidate_features.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"\n  Trade log lengkap disimpan ke: {out_path}")

    print("\n" + "=" * 78)
    print("  ANALISIS PER FITUR (t-test tertile atas vs bawah, korelasi vs pnl_net/is_win)")
    print("=" * 78)

    analyze_feature(closed_df, "body_pct_atr",         "Ukuran Body Candle relatif ATR")
    analyze_feature(closed_df, "wick_rejection_ratio",  "Rasio Wick Rejection")
    analyze_feature(closed_df, "dist_to_swing_atr",     "Jarak ke Swing terdekat (dalam ATR)")
    analyze_feature(closed_df, "rsi_at_entry",          "RSI di titik entry")
    analyze_feature(closed_df, "volume_ratio",          "Volume Ratio di titik entry")

    print("\n" + "=" * 78)
    print("  CARA BACA & LANGKAH SELANJUTNYA")
    print("=" * 78)
    print("""
  Untuk tiap fitur: perhatikan p-value tertile atas vs bawah, dan arah korelasinya.
  - Kalau p < 0.05 DAN mean_pnl tertile atas > tertile bawah secara jelas:
    fitur ini KANDIDAT KUAT untuk jadi syarat tambahan di definisi trigger pullback.
  - Kalau p >= 0.05 (ns): fitur ini TIDAK terbukti membedakan -- jangan dipakai
    sebagai filter, meski secara teori "kelihatan penting".

  Kirim balik seluruh output di atas. Dari situ kita susun definisi trigger
  pullback v2 HANYA dengan syarat yang terbukti signifikan, lalu backtest ulang
  untuk lihat apakah versi barunya sudah profitable berdiri sendiri.
""")


if __name__ == "__main__":
    main()