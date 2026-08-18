"""
scripts/_diag_regime_detector.py
==================================
Script diagnostik untuk Fase 13: Deteksi Market Regime M15.

TUJUAN:
    Jalankan detect_market_regime() di seluruh dataset historis M15,
    lalu laporkan:
    1. Distribusi regime: persentase TRENDING / RANGING /
       BREAKOUT_TRANSITION / CHOP dari total candle dievaluasi.
    2. Rata-rata durasi regime tetap (berapa candle berturut-turut regime
       sama sebelum berubah) — indikasi flicker jika terlalu pendek.
    3. Spot-check visual: 15 sample acak (seed=42) per kategori,
       format mudah dibaca untuk cross-check manual di chart MT5.

PRE-REGISTERED SUCCESS CRITERIA (WAJIB dicek, tidak boleh diubah setelah lihat hasil):
    1. Tidak ada kategori yang persentasenya > 70% atau < 5%.
    2. Rata-rata durasi regime tetap >= 4 candle M15.
    3. (Manual) Spot-check visual masuk akal secara intuitif di chart.

USAGE:
    python scripts/_diag_regime_detector.py           # Cycle 1 (default/V1 constants)
    python scripts/_diag_regime_detector.py --v2      # Cycle 2 (REGIME_PARAMS_V2 override)

    Data M15 harus sudah ada di:
        data/historical/XAUUSD_M15_2025-06-01_2026-07-25.csv
    (Jalankan fetch_m15_data.py terlebih dahulu jika belum ada)
"""

import sys
import os
import warnings
import argparse

# ── Setup encoding & path ────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from engine.indicators import run_all_indicators
from engine.regime_detector import detect_market_regime, REGIME_PARAMS_V2

# ── Parse args ───────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--v2", action="store_true",
                     help="Gunakan REGIME_PARAMS_V2 (Cycle 2 override)")
_args, _ = _parser.parse_known_args()
USE_V2   = _args.v2
REGIME_KWARGS = REGIME_PARAMS_V2 if USE_V2 else {}


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = os.path.join(
    ROOT_DIR, "data", "historical", "XAUUSD_M15_2025-06-01_2026-07-25.csv"
)

SEED          = 42
N_SPOT_CHECK  = 15   # sample per kategori untuk spot-check visual
MIN_IDX_START = 30   # skip candle awal yang belum punya cukup data lookback

# Kriteria sukses pre-registered (JANGAN ubah setelah lihat hasil)
KRITERIA_MIN_PCT  = 5.0   # tidak ada kategori < 5%
KRITERIA_MAX_PCT  = 70.0  # tidak ada kategori > 70%
KRITERIA_MIN_DUR  = 4.0   # durasi rata-rata >= 4 candle


# =============================================================================
# MAIN
# =============================================================================

def main():
    cycle_label = "CYCLE 2 (REGIME_PARAMS_V2)" if USE_V2 else "CYCLE 1 (default constants)"
    print("=" * 80)
    print(f"FASE 13 — DIAGNOSTIK MARKET REGIME DETECTOR (M15) — {cycle_label}")
    print("=" * 80)
    if USE_V2:
        print(f"\nParameter override aktif (REGIME_PARAMS_V2):")
        for k, v in REGIME_PARAMS_V2.items():
            print(f"  {k} = {v}")

    # ── Load data M15 ────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ File tidak ditemukan: {DATA_PATH}")
        print("   Jalankan scripts/fetch_m15_data.py terlebih dahulu.")
        print("   (Butuh MetaTrader 5 aktif)")
        return

    print(f"\nMemuat data dari: {os.path.basename(DATA_PATH)} ...")
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw.set_index("time", inplace=True)

    print(f"  Total candle raw   : {len(df_raw):,}")
    print(f"  Rentang waktu      : {df_raw.index[0]} → {df_raw.index[-1]}")

    # ── Jalankan semua indikator ─────────────────────────────────────────────
    print("\nMenghitung indikator (run_all_indicators) ...")
    df = run_all_indicators(df_raw.copy())
    print(f"  Kolom tersedia     : {list(df.columns)}")

    # ── Jalankan detect_market_regime di seluruh dataset ─────────────────────
    print(f"\nMenjalankan detect_market_regime() "
          f"(skip {MIN_IDX_START} candle pertama) ...")

    total_evaluated = 0
    results = []   # list of (idx, timestamp, regime, arah, detail)

    for i in range(MIN_IDX_START, len(df)):
        # Skip jika ATR belum valid
        if pd.isna(df["atr_14"].iloc[i]):
            continue

        try:
            hasil = detect_market_regime(df, idx=i, **REGIME_KWARGS)
        except Exception as e:
            print(f"  WARNING: Error di idx={i}: {e}")
            continue

        # Skip CHOP akibat data tidak cukup (idx terlalu kecil)
        if "data tidak cukup" in hasil.get("keterangan", "").lower():
            continue

        total_evaluated += 1
        results.append({
            "idx"        : i,
            "timestamp"  : df.index[i],
            "regime"     : hasil["regime"],
            "arah"       : hasil["arah"],
            "keterangan" : hasil["keterangan"],
            "detail"     : hasil["detail"],
        })

    if total_evaluated == 0:
        print("\n❌ Tidak ada candle yang berhasil dievaluasi!")
        return

    print(f"  Candle dievaluasi  : {total_evaluated:,}")

    # ── 1. DISTRIBUSI REGIME ─────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("1. DISTRIBUSI REGIME")
    print("─" * 80)

    distribusi = {}
    for r in results:
        reg = r["regime"]
        distribusi[reg] = distribusi.get(reg, 0) + 1

    semua_kategori = ["TRENDING", "RANGING", "BREAKOUT_TRANSITION", "CHOP"]
    distribusi_pct = {}

    kriteria1_ok = True
    for kategori in semua_kategori:
        count = distribusi.get(kategori, 0)
        pct   = count / total_evaluated * 100 if total_evaluated > 0 else 0.0
        distribusi_pct[kategori] = pct

        bar = "█" * max(0, int(pct / 2))
        status = ""
        if pct > KRITERIA_MAX_PCT or (pct < KRITERIA_MIN_PCT and count > 0):
            status = " ⚠️"
            kriteria1_ok = False
        elif count == 0:
            status = " ⚠️ (tidak ada data)"
            kriteria1_ok = False

        print(f"   {kategori:<25}: {count:6,} candle ({pct:5.1f}%)  {bar}{status}")

    print()
    if kriteria1_ok:
        print(f"   ✅ KRITERIA 1: Semua kategori dalam rentang [{KRITERIA_MIN_PCT}%, {KRITERIA_MAX_PCT}%]")
    else:
        print(f"   ❌ KRITERIA 1: Ada kategori di luar rentang [{KRITERIA_MIN_PCT}%, {KRITERIA_MAX_PCT}%]")

    # ── 2. RATA-RATA DURASI REGIME TETAP ─────────────────────────────────────
    print("\n" + "─" * 80)
    print("2. DURASI REGIME TETAP")
    print("─" * 80)

    # Hitung "run length" — berapa candle berturut-turut regime sama
    run_lengths = []
    if results:
        current_regime = results[0]["regime"]
        current_length = 1

        for r in results[1:]:
            if r["regime"] == current_regime:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_regime = r["regime"]
                current_length = 1
        run_lengths.append(current_length)   # run terakhir

    if run_lengths:
        arr        = np.array(run_lengths)
        rata_rata  = float(arr.mean())
        median_dur = float(np.median(arr))
        min_dur    = int(arr.min())
        max_dur    = int(arr.max())
        n_runs     = len(arr)
        n_short    = int(np.sum(arr <= 2))  # run sangat pendek (flicker)

        print(f"   Total pergantian regime  : {n_runs:,}")
        print(f"   Durasi rata-rata         : {rata_rata:.1f} candle M15")
        print(f"   Durasi median            : {median_dur:.1f} candle M15")
        print(f"   Durasi minimum           : {min_dur} candle M15")
        print(f"   Durasi maksimum          : {max_dur} candle M15")
        print(f"   Run sangat pendek (<= 2) : {n_short:,} "
              f"({n_short / n_runs * 100:.1f}% dari total run)")

        # Histogram distribusi run length
        bins_run = [(1, 1), (2, 3), (4, 6), (7, 12), (13, 20), (21, 99999)]
        print(f"\n   Histogram durasi:")
        for lo, hi in bins_run:
            label = f"{lo}+" if hi == 99999 else f"{lo}-{hi}"
            count = int(np.sum((arr >= lo) & (arr <= hi)))
            pct   = count / n_runs * 100
            bar   = "█" * max(0, int(pct / 2))
            print(f"   {label:>8}: {count:5,} ({pct:5.1f}%)  {bar}")

        print()
        kriteria2_ok = rata_rata >= KRITERIA_MIN_DUR
        if kriteria2_ok:
            print(f"   ✅ KRITERIA 2: Durasi rata-rata {rata_rata:.1f} >= {KRITERIA_MIN_DUR} candle M15")
        else:
            print(f"   ❌ KRITERIA 2: Durasi rata-rata {rata_rata:.1f} < {KRITERIA_MIN_DUR} candle M15 "
                  f"(indikasi flicker)")
    else:
        rata_rata    = 0.0
        kriteria2_ok = False
        print("   ❌ Tidak ada data untuk menghitung durasi")

    # ── 3. SPOT-CHECK VISUAL (15 sample per kategori) ─────────────────────────
    print("\n" + "─" * 80)
    print(f"3. SPOT-CHECK VISUAL ({N_SPOT_CHECK} sample acak per kategori, seed={SEED})")
    print("─" * 80)
    print("   Gunakan timestamp ini untuk cross-check manual di chart MT5.")
    print("   Periksa apakah klasifikasi regime masuk akal secara visual.\n")

    rng = np.random.RandomState(SEED)

    for kategori in semua_kategori:
        kandidat = [r for r in results if r["regime"] == kategori]
        print(f"   {'─' * 70}")
        print(f"   REGIME: {kategori} ({len(kandidat):,} candle)")
        print(f"   {'─' * 70}")

        if not kandidat:
            print(f"   (tidak ada candle dengan regime ini)\n")
            continue

        n_sample = min(N_SPOT_CHECK, len(kandidat))
        indices  = rng.choice(len(kandidat), size=n_sample, replace=False)
        indices.sort()

        for rank, si in enumerate(indices, 1):
            r = kandidat[si]
            ts    = r["timestamp"]
            arah  = r["arah"] or "-"
            detail = r["detail"]

            # Ringkasan sub-check untuk spot-check
            bo = detail["breakout_check"]
            rg = detail["ranging_check"]
            tr = detail["trending_check"]

            print(f"\n   [{rank:2d}] idx={r['idx']:5d} | {ts} | arah={arah}")
            print(f"        Breakout: terpenuhi={bo['terpenuhi']}, "
                  f"body={bo['konfirmasi_body']}, vol={bo['konfirmasi_volume']}")
            print(f"        Ranging : terpenuhi={rg['terpenuhi']}, "
                  f"touches_R={rg['touches_resistance']}, "
                  f"touches_S={rg['touches_support']}")
            print(f"        Trending: terpenuhi={tr['terpenuhi']}, "
                  f"ema_ok={tr['ema_ok']}, "
                  f"struktur_ok={tr['struktur_ok']}, "
                  f"konsistensi={tr['konsistensi_ratio']:.2%}")
            print(f"        Keterangan: {r['keterangan'][:100]}...")

        print()

    # ── RINGKASAN KRITERIA ────────────────────────────────────────────────────
    print("=" * 80)
    print("RINGKASAN KRITERIA SUKSES (PRE-REGISTERED)")
    print("=" * 80)

    print(f"\nDistribusi regime:")
    for kat, pct in distribusi_pct.items():
        cnt = distribusi.get(kat, 0)
        print(f"  {kat:<25}: {pct:5.1f}%  ({cnt:,} candle)")

    print(f"\nKriteria 1 (tidak ada >70% atau <5%)  : "
          f"{'✅ PASSED' if kriteria1_ok else '❌ FAILED'}")
    if 'rata_rata' in dir():
        print(f"Kriteria 2 (durasi rata-rata >= 4)     : "
              f"{'✅ PASSED' if kriteria2_ok else '❌ FAILED'} "
              f"(actual: {rata_rata:.1f} candle)")
    print(f"Kriteria 3 (spot-check visual manual)  : Perlu evaluasi manual di MT5")

    print()
    all_ok = kriteria1_ok and kriteria2_ok
    if all_ok:
        print("→ GATE KUANTITATIF: LOLOS")
        print("  (Masih butuh konfirmasi visual manual dari spot-check)")
    else:
        print("→ GATE KUANTITATIF: TIDAK LOLOS")
        if not kriteria1_ok:
            print(f"  Kriteria 1 gagal: ada kategori di luar [{KRITERIA_MIN_PCT}%, {KRITERIA_MAX_PCT}%]")
        if not kriteria2_ok:
            print(f"  Kriteria 2 gagal: durasi rata-rata {rata_rata:.1f} < {KRITERIA_MIN_DUR}")

    print("=" * 80)


if __name__ == "__main__":
    main()
