"""
scripts/analyze_quality_score_phase43.py
=========================================
Fase 4.3 - Validasi Empiris Confidence Score (Setup Quality)

TUJUAN:
    Sebelum redesain komponen confidence score (4.1/4.2), kita UKUR DULU
    apakah skema yang ada sekarang (ema_gap, alignment, rsi_zone, swing_distance)
    punya sinyal nyata yang terbukti secara statistik.

ALUR:
    1. Jalankan backtest 14 bulan (Jun 2025 - Jul 2026) dengan CSV yang sudah ada.
    2. Analisis per bucket (STRONG / MODERATE / WEAK):
       - win_rate, avg_rrr_realized, avg_pnl_net
       - t-test / ANOVA antar bucket (p-value)
    3. Analisis per komponen individual:
       - Korelasi Pearson & Spearman vs is_win / pnl_net
       - t-test: skor 0 vs skor 2
    4. Korelasi antar komponen (ema_gap vs alignment).

CARA PAKAI:
    python scripts/analyze_quality_score_phase43.py
    python scripts/analyze_quality_score_phase43.py --save-trades
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from scipy import stats

from engine.data_fetcher import load_candles_csv
from engine.backtester   import run_backtest

DEFAULT_M5 = os.path.join(ROOT_DIR, "data", "historical",
                           "XAUUSD_M5_2025-06-01_2026-07-25.csv")
DEFAULT_H1 = os.path.join(ROOT_DIR, "data", "historical",
                           "XAUUSD_H1_2025-06-01_2026-07-25.csv")

COMPONENT_COLS = [
    "score_ema_gap",
    "score_rsi_zone",
    "score_swing_distance",
]

COMPONENT_LABELS = {
    "score_ema_gap"        : "EMA Gap Strength",
    "score_rsi_zone"       : "RSI Zone",
    "score_swing_distance" : "Swing Distance",
}


def _build_parser():
    p = argparse.ArgumentParser(
        prog        = "analyze_quality_score_phase43.py",
        description = "Fase 4.3: Validasi empiris confidence score.",
    )
    p.add_argument("--m5-file",    default=DEFAULT_M5)
    p.add_argument("--h1-file",    default=DEFAULT_H1)
    p.add_argument("--save-trades", action="store_true")
    p.add_argument("--out-dir",    default=os.path.join(ROOT_DIR, "data", "backtest_results"))
    return p


def _ttest_two_groups(group_a, group_b, label_a, label_b, metric):
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 5 or len(b) < 5:
        return {"ok": False, "reason": f"sample kecil ({len(a)}/{len(b)})"}
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return {
        "ok": True, "label_a": label_a, "label_b": label_b, "metric": metric,
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "t_stat": float(t_stat), "p_val": float(p_val),
        "sig": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")),
    }


def _print_ttest(res, indent="    "):
    if not res.get("ok"):
        print(f"{indent}  skip: {res.get('reason')}")
        return
    diff = res["mean_a"] - res["mean_b"]
    print(f"{indent}{res['label_a']} vs {res['label_b']} -- {res['metric']}")
    print(f"{indent}  n={res['n_a']} vs n={res['n_b']}")
    print(f"{indent}  mean: {res['mean_a']:+.4f} vs {res['mean_b']:+.4f} (diff={diff:+.4f})")
    print(f"{indent}  t={res['t_stat']:+.3f}, p={res['p_val']:.4f} {res['sig']}")


def _anova_groups(groups, metric):
    cleaned = {k: v.dropna().values for k, v in groups.items() if len(v.dropna()) >= 5}
    if len(cleaned) < 2:
        return {"ok": False, "reason": "< 2 grup valid"}
    f_stat, p_val = stats.f_oneway(*cleaned.values())
    return {
        "ok": True, "metric": metric,
        "groups": {k: {"n": len(v), "mean": float(v.mean())} for k, v in cleaned.items()},
        "f_stat": float(f_stat), "p_val": float(p_val),
        "sig": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")),
    }


def _corr_series(x, y):
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None, "n": len(x)}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": len(x)}


def _sig(p):
    if p is None: return "?"
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def analyze_quality_buckets(df):
    print()
    print("=" * 70)
    print("  BAGIAN A: ANALISIS PER BUCKET SETUP QUALITY (STRONG/MODERATE/WEAK)")
    print("=" * 70)

    df_closed = df[df["outcome"].isin(["TP_HIT", "SL_HIT"])].copy()
    df_closed["is_win"] = (df_closed["outcome"] == "TP_HIT").astype(int)

    buckets = ["STRONG", "MODERATE", "WEAK"]
    bucket_stats = {}

    print(f"\n  {'Bucket':<12} {'N total':>8} {'N closed':>9} {'Win Rate':>9} {'Avg RRR':>9} {'Avg PnL net':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*12}")

    for b in buckets:
        mask_all    = df["setup_quality"] == b
        mask_closed = df_closed["setup_quality"] == b
        n_all    = mask_all.sum()
        n_closed = mask_closed.sum()

        if n_closed == 0:
            print(f"  {b:<12} {n_all:>8,} {n_closed:>9,}  {'N/A':>8}  {'N/A':>8}  {'N/A':>11}")
            bucket_stats[b] = None
            continue

        wr      = df_closed.loc[mask_closed, "is_win"].mean()
        avg_rrr = df_closed.loc[mask_closed, "rrr_realized"].mean()
        avg_pnl = df.loc[mask_all, "pnl_net"].mean()

        bucket_stats[b] = {
            "n_all": n_all, "n_closed": n_closed, "wr": wr, "avg_rrr": avg_rrr, "avg_pnl": avg_pnl,
            "rrr_series": df_closed.loc[mask_closed, "rrr_realized"],
            "pnl_series": df.loc[mask_all, "pnl_net"],
            "win_series": df_closed.loc[mask_closed, "is_win"].astype(float),
        }
        print(f"  {b:<12} {n_all:>8,} {n_closed:>9,} {wr*100:>8.1f}% {avg_rrr:>+8.3f}R {avg_pnl:>+11.4f}")

    print()
    print("  Distribusi setup_quality_score (0-8):")
    score_dist = df["setup_quality_score"].value_counts().sort_index()
    for sc, cnt in score_dist.items():
        pct = cnt / len(df) * 100
        bar = "x" * int(pct / 2)
        print(f"    Skor {sc}: {cnt:>5,} ({pct:4.1f}%) {bar}")

    print()
    print("  ANOVA -- perbedaan antar bucket:")
    for metric_key, metric_label in [("rrr_series", "avg_rrr_realized"), ("pnl_series", "avg_pnl_net")]:
        groups_dict = {b: bucket_stats[b][metric_key] for b in buckets if bucket_stats.get(b)}
        anova_res = _anova_groups(groups_dict, metric_label)
        if anova_res.get("ok"):
            print(f"    {metric_label}: F={anova_res['f_stat']:.3f}, p={anova_res['p_val']:.4f} {anova_res['sig']}")
        else:
            print(f"    {metric_label}: {anova_res.get('reason')}")

    print()
    print("  Pairwise t-test (Welch's, unequal variance):")
    pairs = [("STRONG", "WEAK"), ("STRONG", "MODERATE"), ("MODERATE", "WEAK")]
    for a, b in pairs:
        if bucket_stats.get(a) and bucket_stats.get(b):
            print(f"\n    {a} vs {b}:")
            for metric_key, metric_label in [("rrr_series", "rrr_realized"), ("pnl_series", "pnl_net")]:
                res = _ttest_two_groups(
                    bucket_stats[a][metric_key], bucket_stats[b][metric_key],
                    a, b, metric_label)
                _print_ttest(res, indent="      ")


def analyze_per_component(df):
    print()
    print("=" * 70)
    print("  BAGIAN B: ANALISIS PER KOMPONEN INDIVIDUAL")
    print("=" * 70)

    df_closed = df[df["outcome"].isin(["TP_HIT", "SL_HIT"])].copy()
    df_closed["is_win"] = (df_closed["outcome"] == "TP_HIT").astype(int)

    for col in COMPONENT_COLS:
        label = COMPONENT_LABELS[col]
        print()
        print(f"  -- {label} ({col}) -----")

        if col not in df.columns:
            print(f"    kolom tidak ditemukan!")
            continue

        vc = df[col].value_counts().sort_index()
        print(f"    Distribusi skor (0/1/2):")
        for sc, cnt in vc.items():
            pct = cnt / len(df) * 100
            print(f"      Skor {sc}: {cnt:>5,} ({pct:4.1f}%)")

        corr_win = _corr_series(df_closed[col].astype(float), df_closed["is_win"].astype(float))
        corr_pnl = _corr_series(df[col].astype(float), df["pnl_net"])
        corr_rrr = _corr_series(df_closed[col].astype(float), df_closed["rrr_realized"])

        print(f"    Korelasi vs is_win   (n={corr_win['n']:,}): "
              f"Pearson r={corr_win['pearson_r']:+.4f} p={corr_win['pearson_p']:.4f} {_sig(corr_win['pearson_p'])} | "
              f"Spearman r={corr_win['spearman_r']:+.4f} p={corr_win['spearman_p']:.4f} {_sig(corr_win['spearman_p'])}")
        print(f"    Korelasi vs pnl_net  (n={corr_pnl['n']:,}): "
              f"Pearson r={corr_pnl['pearson_r']:+.4f} p={corr_pnl['pearson_p']:.4f} {_sig(corr_pnl['pearson_p'])} | "
              f"Spearman r={corr_pnl['spearman_r']:+.4f} p={corr_pnl['spearman_p']:.4f} {_sig(corr_pnl['spearman_p'])}")
        print(f"    Korelasi vs rrr_real (n={corr_rrr['n']:,}): "
              f"Pearson r={corr_rrr['pearson_r']:+.4f} p={corr_rrr['pearson_p']:.4f} {_sig(corr_rrr['pearson_p'])} | "
              f"Spearman r={corr_rrr['spearman_r']:+.4f} p={corr_rrr['spearman_p']:.4f} {_sig(corr_rrr['spearman_p'])}")

        group_0 = df_closed.loc[df_closed[col] == 0, "pnl_net"]
        group_2 = df_closed.loc[df_closed[col] == 2, "pnl_net"]
        if len(group_0) >= 5 and len(group_2) >= 5:
            res = _ttest_two_groups(group_2, group_0, "Skor 2", "Skor 0", "pnl_net")
            print(f"    t-test (skor 2 vs skor 0) pada pnl_net:")
            _print_ttest(res, indent="      ")

        print(f"    Mean pnl_net per skor value:")
        for sc in [0, 1, 2]:
            sub = df.loc[df[col] == sc, "pnl_net"]
            if len(sub) > 0:
                print(f"      Skor {sc}: mean={sub.mean():+.4f}, n={len(sub):,}")

        print(f"    Win rate per skor (closed trades):")
        for sc in [0, 1, 2]:
            sub = df_closed.loc[df_closed[col] == sc]
            if len(sub) > 0:
                wr = sub["is_win"].mean()
                print(f"      Skor {sc}: win_rate={wr:.1%}, n={len(sub):,}")


def analyze_inter_component_correlation(df):
    print()
    print("=" * 70)
    print("  BAGIAN C: KORELASI ANTAR KOMPONEN (3-KOMPONEN PASCA PERBAIKAN)")
    print("=" * 70)
    print("  Catatan: alignment dihapus (tautologi). swing_distance sekarang terisi data.")

    comp_df = df[COMPONENT_COLS].dropna().astype(float)
    if comp_df.empty:
        print("  Data komponen kosong!")
        return

    pearson_corr  = comp_df.corr(method="pearson")
    spearman_corr = comp_df.corr(method="spearman")

    print()
    print("  Pearson Correlation Matrix:")
    header = "  " + " " * 28 + "".join("%-17s" % COMPONENT_LABELS[c] for c in COMPONENT_COLS)
    print(header)
    for r in COMPONENT_COLS:
        row = "  %-28s" % COMPONENT_LABELS[r]
        for c in COMPONENT_COLS:
            val = pearson_corr.loc[r, c]
            row += "%+17.4f" % val
        print(row)

    print()
    print("  Spearman Correlation Matrix:")
    print(header)
    for r in COMPONENT_COLS:
        row = "  %-28s" % COMPONENT_LABELS[r]
        for c in COMPONENT_COLS:
            val = spearman_corr.loc[r, c]
            row += "%+17.4f" % val
        print(row)

    pairs = [
        ("score_ema_gap",   "score_rsi_zone",       "ema_gap vs rsi_zone"),
        ("score_ema_gap",   "score_swing_distance",  "ema_gap vs swing_distance"),
        ("score_rsi_zone",  "score_swing_distance",  "rsi_zone vs swing_distance"),
    ]
    print()
    print("  Pairwise:")
    for col_a, col_b, label in pairs:
        if col_a not in comp_df or col_b not in comp_df:
            continue
        # Check for zero variance (skip if one is constant)
        if comp_df[col_a].std() == 0 or comp_df[col_b].std() == 0:
            print("  %s: zero variance -- skip" % label)
            continue
        sp_r, sp_p = stats.spearmanr(comp_df[col_a], comp_df[col_b])
        pe_r, pe_p = stats.pearsonr(comp_df[col_a],  comp_df[col_b])
        print("  %s:" % label)
        print("    Pearson  r = %+.4f, p = %.6f %s" % (pe_r, pe_p, _sig(pe_p)))
        print("    Spearman r = %+.4f, p = %.6f %s" % (sp_r, sp_p, _sig(sp_p)))

    print()
    print("  Crosstab ema_gap vs swing_distance (counts):")
    ct = pd.crosstab(df["score_ema_gap"], df["score_swing_distance"])
    print(ct.to_string(index=True))


def analyze_score_vs_total(df):
    print()
    print("=" * 70)
    print("  BAGIAN D: SKOR TOTAL (0-8) VS OUTCOME")
    print("=" * 70)

    df_closed = df[df["outcome"].isin(["TP_HIT", "SL_HIT"])].copy()
    df_closed["is_win"] = (df_closed["outcome"] == "TP_HIT").astype(int)

    corr_win = _corr_series(df_closed["setup_quality_score"].astype(float), df_closed["is_win"].astype(float))
    corr_pnl = _corr_series(df["setup_quality_score"].astype(float), df["pnl_net"])
    corr_rrr = _corr_series(df_closed["setup_quality_score"].astype(float), df_closed["rrr_realized"])

    print(f"\n  Skor total (0-8) vs is_win   (n={corr_win['n']:,}): "
          f"Pearson r={corr_win['pearson_r']:+.4f} p={corr_win['pearson_p']:.4f} {_sig(corr_win['pearson_p'])}")
    print(f"  Skor total (0-8) vs pnl_net  (n={corr_pnl['n']:,}): "
          f"Pearson r={corr_pnl['pearson_r']:+.4f} p={corr_pnl['pearson_p']:.4f} {_sig(corr_pnl['pearson_p'])}")
    print(f"  Skor total (0-8) vs rrr_real (n={corr_rrr['n']:,}): "
          f"Pearson r={corr_rrr['pearson_r']:+.4f} p={corr_rrr['pearson_p']:.4f} {_sig(corr_rrr['pearson_p'])}")

    print()
    print("  Mean pnl_net dan win_rate per skor total:")
    print(f"  {'Skor':>5} {'Bucket':<12} {'N total':>8} {'N closed':>9} {'Win%':>7} {'Mean PnL':>10} {'Mean RRR':>9}")
    print(f"  {'--':>5} {'------':<12} {'-------':>8} {'--------':>9} {'----':>7} {'--------':>10} {'--------':>9}")
    for sc in range(9):
        mask_all    = df["setup_quality_score"] == sc
        mask_closed = df_closed["setup_quality_score"] == sc
        n_all    = mask_all.sum()
        n_closed = mask_closed.sum()
        if n_all == 0: continue
        wr  = df_closed.loc[mask_closed, "is_win"].mean() if n_closed > 0 else float("nan")
        pnl = df.loc[mask_all, "pnl_net"].mean()
        rrr = df_closed.loc[mask_closed, "rrr_realized"].mean() if n_closed > 0 else float("nan")
        qtag = "STRONG" if sc >= 6 else ("MODERATE" if sc >= 4 else "WEAK")
        print(f"  {sc:>5} {qtag:<12} {n_all:>8,} {n_closed:>9,} "
              f"{wr*100:>6.1f}% {pnl:>+10.4f} {rrr:>+9.4f}")

    print()
    print("  t-test: STRONG (>=6) vs WEAK (<4) -- pnl_net:")
    res = _ttest_two_groups(
        df.loc[df["setup_quality_score"] >= 6, "pnl_net"],
        df.loc[df["setup_quality_score"] <  4, "pnl_net"],
        "STRONG (>=6)", "WEAK (<4)", "pnl_net")
    _print_ttest(res, indent="    ")

    print()
    print("  t-test: STRONG (>=6) vs WEAK (<4) -- rrr_realized:")
    res2 = _ttest_two_groups(
        df_closed.loc[df_closed["setup_quality_score"] >= 6, "rrr_realized"],
        df_closed.loc[df_closed["setup_quality_score"] <  4, "rrr_realized"],
        "STRONG (>=6)", "WEAK (<4)", "rrr_realized")
    _print_ttest(res2, indent="    ")


def print_summary_conclusion(df):
    print()
    print("=" * 70)
    print("  RINGKASAN EKSEKUTIF -- TEMUAN FASE 4.3")
    print("=" * 70)
    print()
    df_closed = df[df["outcome"].isin(["TP_HIT", "SL_HIT"])].copy()
    df_closed["is_win"] = (df_closed["outcome"] == "TP_HIT").astype(int)

    total  = len(df)
    closed = len(df_closed)
    print(f"  Dataset: {total:,} trade total, {closed:,} closed (TP+SL)")
    print()

    c = _corr_series(df_closed["setup_quality_score"].astype(float), df_closed["is_win"].astype(float))
    print(f"  Skor total vs win: Pearson r={c['pearson_r']:+.4f} ({_sig(c['pearson_p'])})")

    print()
    print("  Komponen vs pnl_net (Spearman r, seluruh trade):")
    for col in COMPONENT_COLS:
        c2 = _corr_series(df[col].astype(float), df["pnl_net"])
        print(f"    {COMPONENT_LABELS[col]:30s}: r={c2['spearman_r']:+.4f} ({_sig(c2['spearman_p'])})")

    x_eg = df["score_ema_gap"].astype(float)
    x_rz = df["score_rsi_zone"].astype(float)
    mask = x_eg.notna() & x_rz.notna()
    if mask.sum() >= 10 and x_eg[mask].std() > 0 and x_rz[mask].std() > 0:
        sp_r, sp_p = stats.spearmanr(x_eg[mask], x_rz[mask])
        print("  Korelasi ema_gap vs rsi_zone: Spearman r=%+.4f (%s)" % (sp_r, _sig(sp_p)))
    x_sw = df["score_swing_distance"].astype(float)
    mask2 = x_eg.notna() & x_sw.notna()
    if mask2.sum() >= 10 and x_eg[mask2].std() > 0 and x_sw[mask2].std() > 0:
        sp_r2, sp_p2 = stats.spearmanr(x_eg[mask2], x_sw[mask2])
        print("  Korelasi ema_gap vs swing_distance: Spearman r=%+.4f (%s)" % (sp_r2, _sig(sp_p2)))
    print()
    print("  Tunggu review user sebelum lanjut ke Langkah 3 (redesain 4.1/4.2).")


def main():
    parser = _build_parser()
    args   = parser.parse_args()
    args.m5_file     = getattr(args, "m5_file",     DEFAULT_M5)
    args.h1_file     = getattr(args, "h1_file",     DEFAULT_H1)
    args.save_trades = getattr(args, "save_trades", False)
    args.out_dir     = getattr(args, "out_dir",     os.path.join(ROOT_DIR, "data", "backtest_results"))

    print("=" * 70)
    print("  FASE 4.3 -- Validasi Empiris Confidence Score (Setup Quality)")
    print("=" * 70)
    print(f"  M5 file: {args.m5_file}")
    print(f"  H1 file: {args.h1_file}")
    print()

    if not os.path.exists(args.m5_file):
        print(f"  M5 file tidak ditemukan: {args.m5_file}")
        sys.exit(1)
    if not os.path.exists(args.h1_file):
        print(f"  H1 file tidak ditemukan: {args.h1_file}")
        sys.exit(1)

    print("-> Loading data CSV...")
    df_m5 = load_candles_csv(args.m5_file)
    df_h1 = load_candles_csv(args.h1_file)

    if df_m5 is None or df_h1 is None:
        print("  Gagal load data CSV")
        sys.exit(1)

    print(f"   M5: {len(df_m5):,} candle ({df_m5.index[0]} -> {df_m5.index[-1]})")
    print(f"   H1: {len(df_h1):,} candle ({df_h1.index[0]} -> {df_h1.index[-1]})")
    print()

    print("-> Menjalankan backtest (volume_mode=IGNORE -- sesuai hasil Fase 3.2)...")
    print()

    # volume_mode="IGNORE" agar volume tidak dipakai sebagai filter/condition
    # Tapi evaluate_entry() hanya kenal "FILTER" dan "CONDITION"
    # Kita pakai "FILTER" tapi pastikan volume_ratio=None agar filter tidak aktif
    trades_df, summary = run_backtest(
        df_m5        = df_m5,
        df_h1        = df_h1,
        warm_up      = 100,
        max_candles  = 288,
        spread_pts   = 0.50,
        profile      = "scalp_m5",
        volume_mode  = "IGNORE",
        verbose      = True,
    )

    if trades_df.empty:
        print("  Tidak ada trade yang ditemukan!")
        sys.exit(1)

    required_cols = ["setup_quality", "setup_quality_score"] + COMPONENT_COLS
    missing = [c for c in required_cols if c not in trades_df.columns]
    if missing:
        print(f"  Kolom tidak ditemukan di trade log: {missing}")
        print("  Pastikan engine/backtester.py sudah diperbarui (Langkah 1).")
        sys.exit(1)

    df = trades_df.dropna(subset=["setup_quality", "setup_quality_score"]).copy()
    print(f"\n  Trade dengan quality data: {len(df):,} dari {len(trades_df):,}")

    if args.save_trades:
        os.makedirs(args.out_dir, exist_ok=True)
        from datetime import datetime as dt
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out_dir, f"phase43_trades_{ts}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"  Trade log disimpan: {out_path}")

    analyze_quality_buckets(df)
    analyze_per_component(df)
    analyze_inter_component_correlation(df)
    analyze_score_vs_total(df)
    print_summary_conclusion(df)

    print()
    print("=" * 70)
    print("  SELESAI -- Tunggu review sebelum lanjut ke Langkah 3 (redesain).")
    print("=" * 70)


if __name__ == "__main__":
    main()


