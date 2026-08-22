"""
scripts/_diag_trend_following_zero.py
======================================
Script DIAGNOSTIK murni — TIDAK mengubah kode apapun.

TUJUAN:
    Menyelidiki mengapa strategi TREND_FOLLOWING menghasilkan NOL trade dalam
    backtest 14 bulan Fase 21, padahal regime TRENDING adalah 10.2% dari
    total candle M15.

    Metodologi sama seperti rekalibrasi Fase 13 cycle 2: kumpulkan bukti lewat
    breakdown angka mentah, baru kemudian di prompt terpisah diputuskan apakah
    ini bug wiring atau perlu revisi desain strategi.

INSTRUKSI:
    Jalankan TANPA argumen tambahan:
        python scripts/_diag_trend_following_zero.py

    Akan mencetak laporan lengkap ke terminal. Tidak menyimpan file apapun.
    Tidak mengubah kode apapun.

CATATAN:
    Script ini HANYA MEMBACA dan MENGANALISIS. Zero side-effect.
"""

import sys
import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester_regime import (
    merge_regime_to_m5,
    merge_h1_context_to_m5,
)
from engine.strategies.trend_following_v2 import evaluate_trend_following

# ── Path data sama persis dengan run_backtest_regime.py ──────────────────────
M5_PATH  = os.path.join(ROOT_DIR, "data", "historical",
                         "XAUUSD_M5_2025-06-01_2026-07-25.csv")
M15_PATH = os.path.join(ROOT_DIR, "data", "historical",
                         "XAUUSD_M15_2025-06-01_2026-07-25.csv")
H1_PATH  = os.path.join(ROOT_DIR, "data", "historical",
                         "XAUUSD_H1_2025-06-01_2026-07-25.csv")

WARM_UP = 100  # sama dengan run_regime_backtest()


def _bar(pct: float, width: int = 30) -> str:
    """Buat visual bar untuk persentase."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def main():
    print("=" * 70)
    print("  DIAGNOSTIK TREND_FOLLOWING NOL TRADE — Fase 21 Post-Mortem")
    print("=" * 70)
    print()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading data historis...")
    for label, path in [("M5", M5_PATH), ("M15", M15_PATH), ("H1", H1_PATH)]:
        if not os.path.exists(path):
            print(f"  ERROR: File {label} tidak ditemukan: {path}")
            sys.exit(1)

    df_m5  = load_candles_csv(M5_PATH)
    df_m15 = load_candles_csv(M15_PATH)
    df_h1  = load_candles_csv(H1_PATH)
    print(f"  M5 : {len(df_m5):,} candle")
    print(f"  M15: {len(df_m15):,} candle")
    print(f"  H1 : {len(df_h1):,} candle")
    print()

    # ── 2. Hitung indikator (sama persis dengan run_regime_backtest) ──────────
    print("[2/5] Menghitung indikator (identik dengan run_regime_backtest)...")
    df_m5_ind  = run_all_indicators(df_m5.copy())
    df_m15_ind = run_all_indicators(df_m15.copy())
    df_h1_ind  = run_all_indicators(df_h1.copy())

    # ── 3. Merge timeframe (sama persis) ─────────────────────────────────────
    print("[3/5] Merge regime M15 → M5 (merge_asof backward)...")
    df_merged = merge_regime_to_m5(df_m5_ind, df_m15_ind)

    print("[4/5] Merge H1 context → M5 (merge_asof backward)...")
    df_merged = merge_h1_context_to_m5(df_merged, df_h1_ind)

    # Statistik populasi
    n_total = len(df_merged)
    n_tf_active = (df_merged["m15_strategy"] == "TREND_FOLLOWING").sum()
    n_after_warmup = (
        (df_merged["m15_strategy"] == "TREND_FOLLOWING") &
        (pd.Series(range(n_total), index=df_merged.index) >= WARM_UP)
    ).sum()

    print(f"\n  Total candle M5        : {n_total:,}")
    print(f"  TREND_FOLLOWING aktif  : {n_tf_active:,} ({n_tf_active/n_total*100:.1f}%)")
    print(f"  ...setelah warm-up={WARM_UP}: {n_after_warmup:,}")
    print()

    # ── 4. Evaluasi setiap candle TREND_FOLLOWING ─────────────────────────────
    print("[5/5] Evaluasi evaluate_trend_following() pada semua candle populasi...")
    print("  (ini memakan waktu beberapa menit karena O(N) dengan find_nearest_swing)")
    print()

    # Filter: hanya candle SETELAH warm-up (sama dengan loop backtest)
    tf_mask = (df_merged["m15_strategy"] == "TREND_FOLLOWING")
    tf_indices_all = df_merged.index[tf_mask]
    # Hanya yang lewat warm-up
    positional_idx_all = [df_merged.index.get_loc(t) for t in tf_indices_all]
    tf_data = [
        (ts, pos_i)
        for ts, pos_i in zip(tf_indices_all, positional_idx_all)
        if pos_i >= WARM_UP
    ]

    if not tf_data:
        print("  TIDAK ADA candle TREND_FOLLOWING setelah warm-up — stop.")
        sys.exit(0)

    n_pop = len(tf_data)
    print(f"  Populasi yang dianalisis: {n_pop:,} candle")
    print()

    # Proses setiap candle
    records = []
    PROGRESS_INTERVAL = max(1, n_pop // 20)

    for step_num, (ts, pos_i) in enumerate(tf_data):
        if step_num % PROGRESS_INTERVAL == 0:
            pct = step_num / n_pop * 100
            print(f"  {pct:5.1f}%  ({step_num:,}/{n_pop:,})...", end="\r", flush=True)

        row = df_merged.iloc[pos_i]
        arah_m15 = row.get("m15_arah")

        # Guard: jika arah None/NaN, skip secara graceful
        if not arah_m15 or (isinstance(arah_m15, float) and np.isnan(arah_m15)):
            records.append({
                "ts": ts, "pos_i": pos_i, "arah": None,
                "ema_trigger_ok": False, "pullback_ok": False,
                "pullback_swing_level": None, "pullback_distance": None,
                "structure_break_ok": False, "terpenuhi": False,
                "error": "arah_m15 kosong/None",
            })
            continue

        # Panggil evaluate_trend_following() sama persis seperti di backtester_regime.py
        try:
            result = evaluate_trend_following(
                df_m5   = df_m5_ind,
                idx_m5  = pos_i,
                arah    = arah_m15,
            )
            records.append({
                "ts"                    : ts,
                "pos_i"                 : pos_i,
                "arah"                  : arah_m15,
                "ema_trigger_ok"        : result["ema_trigger_ok"],
                "pullback_ok"           : result["pullback_ok"],
                "pullback_swing_level"  : result["pullback_swing_level"],
                "pullback_distance"     : result["pullback_distance"],
                "structure_break_ok"    : result["structure_break_ok"],
                "terpenuhi"             : result["terpenuhi"],
                "error"                 : None,
                # ATR untuk analisis rasio
                "atr_value"             : float(df_m5_ind.iloc[pos_i].get("atr_14", np.nan)),
                "close_value"           : float(df_m5_ind.iloc[pos_i]["close"]),
                # candle sebelumnya untuk sample analysis
                "high_m1"               : float(df_m5_ind.iloc[pos_i - 1]["high"]) if pos_i >= 1 else None,
                "low_m1"                : float(df_m5_ind.iloc[pos_i - 1]["low"])  if pos_i >= 1 else None,
                "close_m1"              : float(df_m5_ind.iloc[pos_i - 1]["close"]) if pos_i >= 1 else None,
                "high_m2"               : float(df_m5_ind.iloc[pos_i - 2]["high"]) if pos_i >= 2 else None,
                "low_m2"                : float(df_m5_ind.iloc[pos_i - 2]["low"])  if pos_i >= 2 else None,
                "close_m2"              : float(df_m5_ind.iloc[pos_i - 2]["close"]) if pos_i >= 2 else None,
                # data candle saat ini (idx)
                "open_now"              : float(df_m5_ind.iloc[pos_i]["open"]),
                "high_now"              : float(df_m5_ind.iloc[pos_i]["high"]),
                "low_now"               : float(df_m5_ind.iloc[pos_i]["low"]),
            })
        except Exception as e:
            records.append({
                "ts": ts, "pos_i": pos_i, "arah": arah_m15,
                "ema_trigger_ok": False, "pullback_ok": False,
                "pullback_swing_level": None, "pullback_distance": None,
                "structure_break_ok": False, "terpenuhi": False,
                "error": str(e),
            })

    print(f"\n  Selesai. Total diproses: {len(records):,}")
    print()

    df_rec = pd.DataFrame(records)

    # Pisahkan error dari analisis utama
    df_err  = df_rec[df_rec["error"].notna()]
    df_good = df_rec[df_rec["error"].isna()].copy()

    n_good = len(df_good)
    n_err  = len(df_err)

    # ── 5. LAPORAN ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  === DIAGNOSTIK TREND_FOLLOWING ZERO TRADE ===")
    print("=" * 70)
    print()
    print(f"  Total candle populasi TREND_FOLLOWING aktif : {n_pop:,}")
    print(f"  Berhasil dievaluasi (tanpa error)           : {n_good:,}")
    print(f"  Error saat evaluasi                         : {n_err:,}")
    if n_err > 0:
        print(f"  Contoh error: {df_err['error'].value_counts().head(3).to_dict()}")
    print()

    if n_good == 0:
        print("  Tidak ada data valid untuk dianalisis — stop.")
        sys.exit(0)

    # ── 5a. Breakdown independen per komponen ─────────────────────────────────
    n_ema = int(df_good["ema_trigger_ok"].sum())
    n_pb  = int(df_good["pullback_ok"].sum())
    n_sb  = int(df_good["structure_break_ok"].sum())
    n_ok  = int(df_good["terpenuhi"].sum())

    pct_ema = n_ema / n_good * 100
    pct_pb  = n_pb  / n_good * 100
    pct_sb  = n_sb  / n_good * 100
    pct_ok  = n_ok  / n_good * 100

    print("─" * 70)
    print("  BREAKDOWN INDEPENDEN PER KOMPONEN:")
    print("─" * 70)
    print(f"  ema_trigger_ok=True     : {n_ema:>6,} / {n_good:,} = {pct_ema:6.2f}%  {_bar(pct_ema)}")
    print(f"  pullback_ok=True        : {n_pb:>6,} / {n_good:,} = {pct_pb:6.2f}%  {_bar(pct_pb)}")
    print(f"  structure_break_ok=True : {n_sb:>6,} / {n_good:,} = {pct_sb:6.2f}%  {_bar(pct_sb)}")
    print(f"  terpenuhi=True (irisan) : {n_ok:>6,} / {n_good:,} = {pct_ok:6.2f}%  {_bar(pct_ok)}")
    print()

    # ── 5b. Breakdown arah (BULLISH vs BEARISH) ──────────────────────────────
    print("─" * 70)
    print("  BREAKDOWN PER ARAH:")
    print("─" * 70)
    for arah_val in ["BULLISH", "BEARISH"]:
        df_arah = df_good[df_good["arah"] == arah_val]
        if df_arah.empty:
            continue
        n_a   = len(df_arah)
        n_ema_a = int(df_arah["ema_trigger_ok"].sum())
        n_pb_a  = int(df_arah["pullback_ok"].sum())
        n_sb_a  = int(df_arah["structure_break_ok"].sum())
        n_ok_a  = int(df_arah["terpenuhi"].sum())
        print(f"  [{arah_val}] n={n_a:,}")
        print(f"    ema_trigger_ok     : {n_ema_a:>5,} ({n_ema_a/n_a*100:.1f}%)")
        print(f"    pullback_ok        : {n_pb_a:>5,} ({n_pb_a/n_a*100:.1f}%)")
        print(f"    structure_break_ok : {n_sb_a:>5,} ({n_sb_a/n_a*100:.1f}%)")
        print(f"    terpenuhi          : {n_ok_a:>5,} ({n_ok_a/n_a*100:.1f}%)")
    print()

    # ── 5c. Cross-tab kritis: ema_ok AND pullback_ok ──────────────────────────
    print("─" * 70)
    print("  CROSS-TAB KRITIS (ema_trigger_ok=True AND pullback_ok=True):")
    print("─" * 70)
    df_both = df_good[df_good["ema_trigger_ok"] & df_good["pullback_ok"]]
    n_both = len(df_both)

    if n_both == 0:
        print("  Tidak ada candle yang punya KEDUANYA ema_trigger_ok=True DAN pullback_ok=True.")
        print("  → Kedua kondisi TIDAK PERNAH terpenuhi bersamaan di seluruh populasi.")
    else:
        n_both_sb_ok   = int(df_both["structure_break_ok"].sum())
        n_both_sb_fail = n_both - n_both_sb_ok
        pct_both_sb_ok = n_both_sb_ok / n_both * 100

        print(f"  Candle dengan ema_ok=True AND pullback_ok=True : {n_both:,}")
        print(f"    + structure_break_ok=True juga  : {n_both_sb_ok:>5,} ({pct_both_sb_ok:.2f}%)  {_bar(pct_both_sb_ok)}")
        print(f"    + structure_break_ok=False       : {n_both_sb_fail:>5,} ({100-pct_both_sb_ok:.2f}%)  {_bar(100-pct_both_sb_ok)}")
        print()
        if n_both_sb_ok == 0:
            print("  → KONFIRMASI: Ketiga syarat (AND) TIDAK PERNAH terpenuhi bersamaan.")
            print("    Ini adalah bukti langsung bahwa terpenuhi=True mustahil dicapai.")
    print()

    # ── 5d. Breakdown penyebab pullback_ok gagal ──────────────────────────────
    print("─" * 70)
    print("  BREAKDOWN PENYEBAB pullback_ok GAGAL:")
    print("─" * 70)
    df_pb_fail = df_good[~df_good["pullback_ok"]]
    n_pb_fail  = len(df_pb_fail)

    # Penyebab 1: swing tidak ditemukan (swing_level is None)
    n_swing_none = int(df_pb_fail["pullback_swing_level"].isna().sum())
    # Penyebab 2: swing ditemukan tapi jarak terlalu jauh
    n_swing_found_but_far = n_pb_fail - n_swing_none

    pct_swing_none = n_swing_none / n_good * 100 if n_good > 0 else 0
    pct_swing_far  = n_swing_found_but_far / n_good * 100 if n_good > 0 else 0

    print(f"  Total candle pullback_ok=False: {n_pb_fail:,}")
    print(f"  Penyebab A — swing tidak ditemukan (pullback_swing_level=None):")
    print(f"    {n_swing_none:>6,} ({n_swing_none/n_pb_fail*100:.1f}% dari yang gagal, "
          f"{pct_swing_none:.1f}% dari total populasi)")
    print(f"  Penyebab B — swing ditemukan tapi jarak > toleransi (1.0 × ATR):")
    print(f"    {n_swing_found_but_far:>6,} ({n_swing_found_but_far/n_pb_fail*100:.1f}% dari yang gagal, "
          f"{pct_swing_far:.1f}% dari total populasi)")
    print()

    # Tambahan: breakdown untuk candle yang ema_trigger_ok=True (subset lebih relevan)
    df_ema_true = df_good[df_good["ema_trigger_ok"]]
    if not df_ema_true.empty:
        df_ema_pb_fail = df_ema_true[~df_ema_true["pullback_ok"]]
        n_ema_swing_none = int(df_ema_pb_fail["pullback_swing_level"].isna().sum())
        n_ema_swing_far  = len(df_ema_pb_fail) - n_ema_swing_none
        print(f"  Sama tapi khusus subset ema_trigger_ok=True ({len(df_ema_true):,} candle):")
        print(f"  Penyebab A — swing tidak ditemukan : {n_ema_swing_none:,} "
              f"({n_ema_swing_none/len(df_ema_true)*100:.1f}%)")
        print(f"  Penyebab B — jarak > toleransi    : {n_ema_swing_far:,} "
              f"({n_ema_swing_far/len(df_ema_true)*100:.1f}%)")
    print()

    # ── 5e. Distribusi pullback_distance/ATR ─────────────────────────────────
    print("─" * 70)
    print("  DISTRIBUSI pullback_distance/ATR (candle dengan swing ditemukan):")
    print("─" * 70)

    # Hanya candle dengan swing ditemukan (pullback_swing_level is not None)
    df_has_swing = df_good[df_good["pullback_swing_level"].notna()].copy()
    n_has_swing  = len(df_has_swing)
    print(f"  Candle dengan swing ditemukan : {n_has_swing:,} ({n_has_swing/n_good*100:.1f}% dari populasi)")

    if not df_has_swing.empty and "atr_value" in df_has_swing.columns:
        df_has_swing["dist_atr_ratio"] = df_has_swing["pullback_distance"] / df_has_swing["atr_value"]
        df_has_swing["dist_atr_ratio"] = df_has_swing["dist_atr_ratio"].replace([np.inf, -np.inf], np.nan)
        df_has_swing = df_has_swing.dropna(subset=["dist_atr_ratio"])

        if not df_has_swing.empty:
            ratios = df_has_swing["dist_atr_ratio"]
            print(f"\n  Statistik pullback_distance / ATR (n={len(ratios):,}):")
            print(f"    Min     : {ratios.min():.4f}")
            print(f"    p10     : {ratios.quantile(0.10):.4f}")
            print(f"    p25     : {ratios.quantile(0.25):.4f}")
            print(f"    Median  : {ratios.median():.4f}")
            print(f"    p75     : {ratios.quantile(0.75):.4f}")
            print(f"    p90     : {ratios.quantile(0.90):.4f}")
            print(f"    p95     : {ratios.quantile(0.95):.4f}")
            print(f"    p99     : {ratios.quantile(0.99):.4f}")
            print(f"    Max     : {ratios.max():.4f}")
            print(f"    Mean    : {ratios.mean():.4f}")
            print(f"    Std     : {ratios.std():.4f}")
            print()
            print(f"  Catatan: batas toleransi saat ini = 1.0 × ATR")
            n_dalam  = (ratios <= 1.0).sum()
            n_luar   = (ratios > 1.0).sum()
            print(f"    Dalam toleransi (≤ 1.0×ATR): {n_dalam:,} ({n_dalam/len(ratios)*100:.1f}%)")
            print(f"    Luar toleransi  (> 1.0×ATR): {n_luar:,} ({n_luar/len(ratios)*100:.1f}%)")
            print()

            # Histogram distribusi
            print("  Histogram distribusi dist/ATR (binwidth = 0.5):")
            bins = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, float("inf")]
            labels = ["0.0–0.5", "0.5–1.0", "1.0–1.5", "1.5–2.0",
                      "2.0–3.0", "3.0–4.0", "4.0–5.0", ">5.0"]
            counts, _ = np.histogram(ratios.values, bins=bins)
            total_hist = counts.sum()
            for lbl, cnt in zip(labels, counts):
                bar = "█" * int(cnt / total_hist * 40)
                pct_h = cnt / total_hist * 100
                marker = " ← batas 1.0×ATR" if lbl == "0.5–1.0" else ""
                print(f"    [{lbl:>8s}]: {cnt:>6,} ({pct_h:5.1f}%)  {bar}{marker}")
    print()

    # ── 5f. Sample 10 candle: ema_ok=True, pullback_ok=True, structure_break=False ──
    print("─" * 70)
    print("  SAMPLE CANDLE (ema_ok=True, pullback_ok=True, structure_break_ok=False):")
    print("─" * 70)

    if n_both == 0:
        print("  Tidak ada candle yang memenuhi kriteria sample (tidak ada ema+pullback bersamaan).")
        print()
    else:
        df_sample_pool = df_both[~df_both["structure_break_ok"]].copy()
        n_sample_pool  = len(df_sample_pool)
        print(f"  Pool sample: {n_sample_pool:,} candle")

        if n_sample_pool == 0:
            print("  Tidak ada (semua yang ema+pullback terpenuhi juga punya structure_break_ok=True).")
        else:
            rng = np.random.default_rng(42)
            sample_n = min(10, n_sample_pool)
            sample_indices = rng.choice(n_sample_pool, size=sample_n, replace=False)
            df_sample = df_sample_pool.iloc[sorted(sample_indices)].reset_index(drop=True)

            print(f"  Menampilkan {sample_n} candle sampel acak (seed=42):")
            print()

            cols_display = [
                "pos_i", "ts", "arah",
                "pullback_swing_level", "pullback_distance",
                "close_m2", "high_m2", "low_m2",
                "close_m1", "high_m1", "low_m1",
                "open_now", "high_now", "low_now", "close_value",
            ]
            for i_s, row_s in df_sample.iterrows():
                pos_i_s = int(row_s["pos_i"])
                print(f"  [{i_s+1:2d}] idx={pos_i_s}, ts={row_s['ts']}, arah={row_s['arah']}")
                print(f"       pullback_swing={row_s['pullback_swing_level']:.4f}, "
                      f"dist={row_s['pullback_distance']:.4f}, atr={row_s.get('atr_value', 'N/A')}")
                print(f"       idx-2: close={row_s['close_m2']:.4f}, "
                      f"high={row_s['high_m2']:.4f}, low={row_s['low_m2']:.4f}")
                print(f"       idx-1: close={row_s['close_m1']:.4f}, "
                      f"high={row_s['high_m1']:.4f}, low={row_s['low_m1']:.4f}")
                print(f"       idx-0: open={row_s['open_now']:.4f}, "
                      f"high={row_s['high_now']:.4f}, low={row_s['low_now']:.4f}, "
                      f"close={row_s['close_value']:.4f}")

                # Cek: apakah close ≤ max(high_m1, high_m2) untuk BUY? (struktur belum tembus)
                if row_s["arah"] == "BULLISH":
                    max_h = max(row_s["high_m1"], row_s["high_m2"])
                    gap = row_s["close_value"] - max_h
                    print(f"       Structure: close({row_s['close_value']:.4f}) vs max_high({max_h:.4f}) "
                          f"→ gap={gap:+.4f} (negatif = belum tembus)")
                else:  # BEARISH
                    min_l = min(row_s["low_m1"], row_s["low_m2"])
                    gap = min_l - row_s["close_value"]
                    print(f"       Structure: close({row_s['close_value']:.4f}) vs min_low({min_l:.4f}) "
                          f"→ gap={gap:+.4f} (negatif = belum tembus)")
                print()

    # ── 5g. Analisis tambahan: co-occurrence matrix semua tiga kondisi ────────
    print("─" * 70)
    print("  CO-OCCURRENCE MATRIX (3 kondisi):")
    print("─" * 70)
    combos = {
        "(F,F,F)": (~df_good["ema_trigger_ok"] & ~df_good["pullback_ok"] & ~df_good["structure_break_ok"]).sum(),
        "(T,F,F)": ( df_good["ema_trigger_ok"] & ~df_good["pullback_ok"] & ~df_good["structure_break_ok"]).sum(),
        "(F,T,F)": (~df_good["ema_trigger_ok"] &  df_good["pullback_ok"] & ~df_good["structure_break_ok"]).sum(),
        "(F,F,T)": (~df_good["ema_trigger_ok"] & ~df_good["pullback_ok"] &  df_good["structure_break_ok"]).sum(),
        "(T,T,F)": ( df_good["ema_trigger_ok"] &  df_good["pullback_ok"] & ~df_good["structure_break_ok"]).sum(),
        "(T,F,T)": ( df_good["ema_trigger_ok"] & ~df_good["pullback_ok"] &  df_good["structure_break_ok"]).sum(),
        "(F,T,T)": (~df_good["ema_trigger_ok"] &  df_good["pullback_ok"] &  df_good["structure_break_ok"]).sum(),
        "(T,T,T)": ( df_good["ema_trigger_ok"] &  df_good["pullback_ok"] &  df_good["structure_break_ok"]).sum(),
    }
    print("  (E=ema_trigger, P=pullback, S=structure_break)")
    print(f"  {'Kombinasi (E,P,S)':<20} {'Jumlah':>8}  {'%':>7}   Bar")
    print("  " + "-" * 60)
    for combo, cnt in sorted(combos.items(), key=lambda x: -x[1]):
        pct_c = cnt / n_good * 100 if n_good > 0 else 0
        bar   = "█" * int(pct_c * 0.5)
        print(f"  {combo:<20} {cnt:>8,}  {pct_c:>6.2f}%   {bar}")
    print()

    # ── 5h. Analisis temporalitas: berapa candle setelah pullback_ok sebelum structure_break_ok?──
    print("─" * 70)
    print("  ANALISIS TEMPORALITAS — struktur break datang SETELAH pullback berapa candle?")
    print("─" * 70)
    print("  (Untuk subset ema_trigger_ok=True AND pullback_ok=True, n=" + str(n_both) + ")")
    print()

    if n_both > 0 and n_both_sb_fail > 0:
        print("  Untuk setiap candle ema+pullback=True tapi structure_break=False,")
        print("  cek berapa candle ke depan (max 10) baru structure break terjadi:")
        print("  (PERHATIAN: ini hanya untuk analisis visual — BUKAN look-ahead di engine)")
        print()

        gaps = []
        df_sample_temp = df_both[~df_both["structure_break_ok"]].copy()
        sample_temp_n  = min(200, len(df_sample_temp))  # batasi untuk kecepatan
        df_sample_temp = df_sample_temp.sample(n=sample_temp_n, random_state=42)

        for _, row_t in df_sample_temp.iterrows():
            pos_t = int(row_t["pos_i"])
            arah_t = row_t["arah"]
            found_at = None

            for delta in range(1, 11):
                future_i = pos_t + delta
                if future_i >= len(df_m5_ind):
                    break
                row_f = df_m5_ind.iloc[future_i]

                # Cek apakah di future_i, structure break terjadi
                # (berdasarkan candle sebelum future_i)
                if future_i < 2:
                    continue
                close_f = float(df_m5_ind.iloc[future_i]["close"])
                high_f1 = float(df_m5_ind.iloc[future_i - 1]["high"])
                high_f2 = float(df_m5_ind.iloc[future_i - 2]["high"])
                low_f1  = float(df_m5_ind.iloc[future_i - 1]["low"])
                low_f2  = float(df_m5_ind.iloc[future_i - 2]["low"])

                if arah_t == "BULLISH" and close_f > max(high_f1, high_f2):
                    found_at = delta
                    break
                elif arah_t == "BEARISH" and close_f < min(low_f1, low_f2):
                    found_at = delta
                    break

            gaps.append(found_at)

        gaps_arr = [g for g in gaps if g is not None]
        gaps_never = [g for g in gaps if g is None]

        print(f"  Sampel: {sample_temp_n} candle diperiksa")
        print(f"  Structure break ditemukan dalam 10 candle ke depan: {len(gaps_arr)} ({len(gaps_arr)/sample_temp_n*100:.1f}%)")
        print(f"  Tidak ditemukan dalam 10 candle ke depan           : {len(gaps_never)} ({len(gaps_never)/sample_temp_n*100:.1f}%)")

        if gaps_arr:
            gaps_series = pd.Series(gaps_arr)
            print(f"\n  Distribusi 'candle ke-berapa structure break terjadi setelah pullback':")
            for delta in range(1, 11):
                cnt_d = (gaps_series == delta).sum()
                pct_d = cnt_d / len(gaps_arr) * 100 if gaps_arr else 0
                bar = "█" * int(pct_d * 0.5)
                print(f"    Delta +{delta:2d}: {cnt_d:>5,} ({pct_d:5.1f}%)  {bar}")
    print()

    # ── KESIMPULAN SEMENTARA ──────────────────────────────────────────────────
    print("=" * 70)
    print("  KESIMPULAN SEMENTARA (murni observasi — BUKAN keputusan fix):")
    print("=" * 70)
    print()
    print(f"  1. Dari {n_good:,} candle populasi TREND_FOLLOWING aktif:")
    print(f"     - EMA trigger terpenuhi        : {pct_ema:.1f}%")
    print(f"     - Pullback struktural terpenuhi : {pct_pb:.1f}%")
    print(f"     - Structure break terpenuhi    : {pct_sb:.1f}%")
    print(f"     - Ketiganya sekaligus (terpenuhi): {pct_ok:.1f}%")
    print()

    if n_ok == 0:
        print("  2. terpenuhi=True = NOL. Sistem TIDAK pernah entry TREND_FOLLOWING.")
        print("     Ini KONFIRMASI konsistensi dengan hasil smoke test Fase 21.")
    print()

    # Diagnosa utama
    if n_both == 0:
        print("  3. TEMUAN UTAMA: ema_trigger_ok DAN pullback_ok tidak pernah terpenuhi")
        print("     BERSAMAAN di candle yang sama. Bottleneck ada di dua kondisi pertama,")
        print("     bahkan sebelum menyentuh structure_break_ok.")
    elif n_both > 0 and n_both_sb_ok == 0:
        print("  3. TEMUAN UTAMA: Ada candle dengan ema+pullback terpenuhi bersamaan,")
        print("     TAPI tidak satupun yang JUGA punya structure_break_ok=True.")
        print("     → Dugaan 'dua fase berbeda' TERBUKTI secara empiris pada dataset ini.")
        print("     → pullback (harga dekat swing/masih diam) dan structure break")
        print("       (harga sudah menembus high/low sebelumnya) memang terjadi di")
        print("       candle yang berbeda, bukan di candle yang sama.")
    elif n_both > 0 and n_both_sb_ok > 0:
        print(f"  3. TEMUAN: Ada {n_both_sb_ok} candle yang SEMUA tiga kondisi terpenuhi,")
        print(f"     tapi tidak menghasilkan trade. Perlu investigasi lebih lanjut di")
        print(f"     level wiring backtester_regime.py.")
    print()
    print("  *** Keputusan desain (jika ada fix yang diperlukan) akan diputuskan")
    print("      di prompt TERPISAH setelah melihat laporan ini. ***")
    print()
    print("=" * 70)
    print("  DIAGNOSTIK SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()
