"""
scripts/_diag_zone_detector.py
===============================
Script diagnostik untuk Fase 8: Deteksi Zona Konsolidasi.

TUJUAN:
    Jalankan detect_consolidation_zone() di seluruh dataset historis M5,
    lalu laporkan:
    1. Frekuensi deteksi (% candle dengan is_valid=True) — gate: 5%-60%
    2. Distribusi duration untuk zona valid (histogram text-based)
    3. Distribusi range_atr_ratio untuk zona valid (statistik deskriptif)
    4. Spot-check visual: 10 sample timestamp acak (seed=42) dengan tabel
       OHLC candle di sekitar zona, untuk cross-check manual di MT5

USAGE:
    python scripts/_diag_zone_detector.py
"""

import sys
import os

# ── Setup encoding & path ────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from engine.zone_detector import detect_consolidation_zone
from engine.indicators import calculate_atr


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = os.path.join(
    ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv"
)
SEED = 42
N_SPOT_CHECK = 10
CONTEXT_CANDLES = 5  # candle sebelum/sesudah zona untuk spot-check


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FASE 8 — DIAGNOSTIK ZONA KONSOLIDASI")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ File tidak ditemukan: {DATA_PATH}")
        print("   Jalankan fetch_candles.py terlebih dahulu.")
        return

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Hitung ATR_14 (fungsi dari indicators.py)
    df = calculate_atr(df, period=14)

    print(f"\nDataset: {os.path.basename(DATA_PATH)}")
    print(f"Total candle: {len(df):,}")
    print(f"Rentang waktu: {df.index[0]} → {df.index[-1]}")

    # ── Jalankan deteksi di seluruh dataset ──────────────────────────────────
    # Skip candle awal yang belum punya cukup lookback (20 candle)
    lookback = 20
    start_idx = lookback - 1  # idx pertama yang bisa dievaluasi
    total_evaluated = 0
    total_valid = 0

    results = []  # simpan (idx, result) untuk analisis lanjutan

    print(f"\nMenjalankan deteksi (lookback={lookback}, "
          f"max_range_atr_ratio=2.5, min_duration=10)...")

    for i in range(start_idx, len(df)):
        # Skip jika ATR belum valid (NaN di awal)
        if pd.isna(df["atr_14"].iloc[i]):
            continue

        result = detect_consolidation_zone(df, idx=i, lookback=lookback)
        total_evaluated += 1

        if result["is_valid"]:
            total_valid += 1
            results.append((i, result))

    # ── 1. Frekuensi deteksi ─────────────────────────────────────────────────
    freq_pct = (total_valid / total_evaluated * 100) if total_evaluated > 0 else 0

    print("\n" + "─" * 60)
    print("1. FREKUENSI DETEKSI")
    print("─" * 60)
    print(f"   Candle dievaluasi : {total_evaluated:,}")
    print(f"   Zona valid        : {total_valid:,}")
    print(f"   Frekuensi         : {freq_pct:.1f}%")

    if 5 <= freq_pct <= 60:
        print(f"   Status            : ✅ DALAM RENTANG WAJAR (5%-60%)")
    else:
        print(f"   Status            : ❌ DI LUAR RENTANG WAJAR (5%-60%)")
        if freq_pct < 5:
            print(f"   → Terlalu jarang — parameter mungkin terlalu ketat")
        else:
            print(f"   → Terlalu sering — parameter mungkin terlalu longgar")

    # ── 2. Distribusi duration ───────────────────────────────────────────────
    if results:
        durations = [r["duration"] for _, r in results]
        durations_arr = np.array(durations)

        print("\n" + "─" * 60)
        print("2. DISTRIBUSI DURATION (zona valid)")
        print("─" * 60)
        print(f"   Min      : {durations_arr.min()}")
        print(f"   Max      : {durations_arr.max()}")
        print(f"   Mean     : {durations_arr.mean():.1f}")
        print(f"   Median   : {np.median(durations_arr):.1f}")
        print(f"   Std      : {durations_arr.std():.1f}")

        # Histogram text-based
        bins = [(10, 12), (13, 15), (16, 18), (19, 20)]
        print(f"\n   Histogram:")
        total_in_bins = 0
        for lo, hi in bins:
            count = np.sum((durations_arr >= lo) & (durations_arr <= hi))
            total_in_bins += count
            pct = count / len(durations_arr) * 100
            bar = "█" * int(pct / 2)
            print(f"   {lo:2d}-{hi:2d}: {count:5d} ({pct:5.1f}%) {bar}")

        # Cek apakah didominasi batas minimum
        at_minimum = np.sum(durations_arr == 10)
        at_min_pct = at_minimum / len(durations_arr) * 100
        near_minimum = np.sum((durations_arr >= 10) & (durations_arr <= 12))
        near_min_pct = near_minimum / len(durations_arr) * 100

        print(f"\n   Duration == 10 (tepat minimum)    : {at_minimum:,} ({at_min_pct:.1f}%)")
        print(f"   Duration 10-12 (dekat minimum)    : {near_minimum:,} ({near_min_pct:.1f}%)")

        if near_min_pct > 70:
            print(f"   ⚠️  PERINGATAN: {near_min_pct:.1f}% zona di/dekat batas minimum")
            print(f"      → Kemungkinan threshold terlalu longgar (banyak false positive)")
        else:
            print(f"   ✅ Distribusi tidak didominasi batas minimum")

    # ── 3. Distribusi range_atr_ratio ────────────────────────────────────────
    if results:
        ratios = [r["range_atr_ratio"] for _, r in results]
        ratios_arr = np.array(ratios)

        print("\n" + "─" * 60)
        print("3. DISTRIBUSI RANGE/ATR RATIO (zona valid)")
        print("─" * 60)
        print(f"   Min      : {ratios_arr.min():.3f}")
        print(f"   Max      : {ratios_arr.max():.3f}")
        print(f"   Mean     : {ratios_arr.mean():.3f}")
        print(f"   Median   : {np.median(ratios_arr):.3f}")
        print(f"   Std      : {ratios_arr.std():.3f}")

        # Histogram
        ratio_bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]
        print(f"\n   Histogram:")
        for lo, hi in ratio_bins:
            count = np.sum((ratios_arr >= lo) & (ratios_arr < hi))
            pct = count / len(ratios_arr) * 100
            bar = "█" * int(pct / 2)
            print(f"   {lo:.1f}-{hi:.1f}: {count:5d} ({pct:5.1f}%) {bar}")

    # ── 4. Spot-check visual ─────────────────────────────────────────────────
    if results:
        print("\n" + "─" * 60)
        print(f"4. SPOT-CHECK VISUAL ({N_SPOT_CHECK} sample acak, seed={SEED})")
        print("─" * 60)
        print("   Gunakan timestamp ini untuk cross-check di chart MT5.")
        print("   Periksa apakah zona terlihat seperti konsolidasi secara visual.\n")

        rng = np.random.RandomState(SEED)
        sample_indices = rng.choice(len(results), size=min(N_SPOT_CHECK, len(results)), replace=False)
        sample_indices.sort()

        for sample_num, si in enumerate(sample_indices, 1):
            df_idx, res = results[si]

            # Rentang window zona
            zone_start = df_idx - res["duration"] + 1
            zone_end = df_idx

            # Context: beberapa candle sebelum dan sesudah zona
            ctx_start = max(zone_start - CONTEXT_CANDLES, 0)
            ctx_end = min(zone_end + CONTEXT_CANDLES, len(df) - 1)

            print(f"   ┌─── Sample #{sample_num} ───────────────────────────────")
            print(f"   │ Zona: candle idx {zone_start} → {zone_end} "
                  f"({df.index[zone_start]} → {df.index[zone_end]})")
            print(f"   │ Duration: {res['duration']} candle")
            print(f"   │ Resistance: {res['resistance']:.2f}")
            print(f"   │ Support:    {res['support']:.2f}")
            print(f"   │ Range:      {res['range_zone']:.2f} "
                  f"({res['range_atr_ratio']:.2f}x ATR)")
            print(f"   │")
            print(f"   │ OHLC context (idx {ctx_start} → {ctx_end}):")
            print(f"   │ {'Idx':>5} {'Time':>22} {'Open':>9} {'High':>9} "
                  f"{'Low':>9} {'Close':>9} {'ATR14':>7} {'InZone':>6}")
            print(f"   │ {'─'*5} {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*6}")

            for j in range(ctx_start, ctx_end + 1):
                row = df.iloc[j]
                in_zone = "  ◄ " if zone_start <= j <= zone_end else ""
                atr_str = f"{row['atr_14']:.2f}" if not pd.isna(row.get("atr_14", np.nan)) else "  N/A"
                print(f"   │ {j:5d} {str(df.index[j]):>22} "
                      f"{row['open']:9.2f} {row['high']:9.2f} "
                      f"{row['low']:9.2f} {row['close']:9.2f} "
                      f"{atr_str:>7} {in_zone}")

            print(f"   └{'─' * 55}")
            print()

    # ── RINGKASAN ────────────────────────────────────────────────────────────
    print("=" * 80)
    print("RINGKASAN EVALUASI")
    print("=" * 80)
    print(f"  Frekuensi deteksi : {freq_pct:.1f}% ", end="")
    freq_ok = 5 <= freq_pct <= 60
    print("✅" if freq_ok else "❌")

    if results:
        duration_ok = near_min_pct <= 70
        print(f"  Distribusi durasi : {near_min_pct:.1f}% di/dekat minimum ", end="")
        print("✅" if duration_ok else "❌")
    else:
        duration_ok = False
        print(f"  Distribusi durasi : Tidak ada zona valid ❌")

    print(f"  Spot-check visual : Perlu evaluasi manual di MT5")
    print()

    if freq_ok and duration_ok:
        print("  → GATE KUANTITATIF: LOLOS (frekuensi + distribusi)")
        print("  → Masih butuh konfirmasi visual manual dari spot-check")
    else:
        print("  → GATE KUANTITATIF: TIDAK LOLOS")
        if not freq_ok:
            print(f"    Frekuensi {freq_pct:.1f}% di luar 5%-60%")
        if not duration_ok:
            print(f"    Distribusi durasi: {near_min_pct:.1f}% di/dekat minimum (> 70%)")

    print("=" * 80)


if __name__ == "__main__":
    main()
