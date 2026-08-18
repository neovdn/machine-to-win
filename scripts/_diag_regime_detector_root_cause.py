"""
scripts/_diag_regime_detector_root_cause.py
============================================
Diagnostik TAHAP 1 untuk rekalibrasi Fase 13 (Cycle 2).

Tujuan: mengumpulkan 4 metrik root-cause SEBELUM mengubah konstanta apapun.
Script ini TIDAK memodifikasi engine/regime_detector.py.

Metrik yang dihitung:
  1. Histogram swing_confirm (jumlah pasang swing terurut) untuk seluruh candle.
  2. Cross-tab kegagalan trending pada candle dengan ema_ok=True.
  3. Histogram touches_R dan touches_S pada zona yang is_valid=True.
  4. Statistik episode breakout vs episode ranging.
"""

import sys
import os
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.indicators import run_all_indicators
from engine.regime_detector import (
    detect_market_regime,
    _detect_swing_sequence,
    _check_trending,
    _check_ranging,
    _check_breakout_transition,
    REGIME_TREND_LOOKBACK,
    REGIME_TREND_SWING_WING,
    REGIME_TREND_MIN_SWING_PAIRS,
    REGIME_TREND_MIN_EMA_GAP_PCT,
    REGIME_RANGE_LOOKBACK,
    REGIME_RANGE_MAX_ATR_RATIO,
    REGIME_RANGE_MIN_DURATION,
    REGIME_RANGE_TOUCH_TOLERANCE_ATR,
    REGIME_RANGE_MIN_TOUCHES_PER_SIDE,
)
from engine.zone_detector import detect_consolidation_zone

# ── Konstanta ─────────────────────────────────────────────────────────────────
DATA_FILE   = "data/historical/XAUUSD_M15_2025-06-01_2026-07-25.csv"
MIN_IDX     = 30   # skip candle awal (sama dengan _diag_regime_detector.py)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df


def main():
    sep = "=" * 80

    print(sep)
    print("FASE 13 — DIAGNOSTIK ROOT CAUSE (Cycle 2, Tahap 1)")
    print(sep)

    # ── Load dan prepare data ─────────────────────────────────────────────────
    print(f"\nMemuat data dari: {DATA_FILE} ...")
    df_raw = load_data(DATA_FILE)
    print(f"  Total candle raw   : {len(df_raw):,}")
    print(f"  Rentang waktu      : {df_raw.index[0]} -> {df_raw.index[-1]}")

    print("\nMenghitung indikator (run_all_indicators) ...")
    df = run_all_indicators(df_raw.copy())
    n  = len(df)
    print(f"  Candle setelah indikator: {n:,}")

    # ── Jalankan deteksi per candle ───────────────────────────────────────────
    # Kumpulkan data mentah untuk keempat metrik.
    # Agar efisien, kita panggil sub-fungsi individual (bukan detect_market_regime)
    # karena kita butuh data internal yang tidak diekspos di output utama.

    eval_indices = list(range(MIN_IDX, n))
    total_eval   = len(eval_indices)

    print(f"\nMenjalankan sub-fungsi pada {total_eval:,} candle ...")
    print("  (ini membutuhkan waktu beberapa menit — mohon tunggu)")

    # Struktur akumulator
    swing_confirm_dist   = []   # list int — swing_count_confirm per candle
    ema_ok_failures      = {"struktur_saja": 0, "konsistensi_saja": 0, "keduanya": 0, "total_ema_ok": 0}

    zone_valid_touches_R = []   # int — touches_resistance per zona valid
    zone_valid_touches_S = []   # int — touches_support per zona valid

    # Untuk metrik 4: episode
    #   Episode ranging = run berturut-turut regime==RANGING yang baru dimulai
    #   Episode breakout = run berturut-turut regime==BREAKOUT_TRANSITION yang baru dimulai
    #   Kita track lewat perubahan state regime dari full detect_market_regime()
    regime_sequence = []   # list of str — regime per candle
    ranging_episode_count   = 0
    breakout_episode_count  = 0
    prev_regime = None

    # Progress reporting
    report_every = total_eval // 20   # setiap 5%

    for i, idx in enumerate(eval_indices):
        if report_every > 0 and i % report_every == 0:
            pct = i / total_eval * 100
            print(f"  ... {pct:.0f}% ({i:,}/{total_eval:,})", flush=True)

        # ── Metrik 1 & 2: trending diagnostics ───────────────────────────────
        # Panggil _check_trending untuk mendapat detail internal
        tr = _check_trending(df, idx)

        swing_confirm_dist.append(tr.get("swing_count_confirm", 0))

        if tr["ema_ok"]:
            ema_ok_failures["total_ema_ok"] += 1
            if not tr["terpenuhi"]:
                # Tentukan mana yang gagal: struktur atau konsistensi
                s_gagal = not tr["struktur_ok"]
                k_gagal = not tr["konsistensi_ok"]
                if s_gagal and k_gagal:
                    ema_ok_failures["keduanya"] += 1
                elif s_gagal:
                    ema_ok_failures["struktur_saja"] += 1
                elif k_gagal:
                    ema_ok_failures["konsistensi_saja"] += 1

        # ── Metrik 3: ranging zone touches ───────────────────────────────────
        rg = _check_ranging(df, idx)
        zone_info = rg.get("zone") or {}
        if zone_info.get("is_valid", False):
            zone_valid_touches_R.append(rg["touches_resistance"])
            zone_valid_touches_S.append(rg["touches_support"])

        # ── Metrik 4: episode tracking (via full detect) ──────────────────────
        result = detect_market_regime(df, idx)
        regime = result["regime"]
        regime_sequence.append(regime)

        if prev_regime != "RANGING" and regime == "RANGING":
            ranging_episode_count += 1
        if prev_regime != "BREAKOUT_TRANSITION" and regime == "BREAKOUT_TRANSITION":
            breakout_episode_count += 1
        prev_regime = regime

    print(f"  ... 100% ({total_eval:,}/{total_eval:,})")

    # ── LAPORAN METRIK 1: Histogram swing_confirm ─────────────────────────────
    print(f"\n{'-' * 80}")
    print("METRIK 1: HISTOGRAM swing_confirm (seluruh candle dievaluasi)")
    print(f"{'-' * 80}")
    sc_arr   = np.array(swing_confirm_dist)
    n_total  = len(sc_arr)

    bins     = [(0, 0), (1, 1), (2, 2), (3, 999)]
    labels   = ["0 pasang", "1 pasang", "2 pasang", "3+ pasang"]

    print(f"  Total candle dievaluasi: {n_total:,}")
    print()
    for (lo, hi), label in zip(bins, labels):
        count = int(np.sum((sc_arr >= lo) & (sc_arr <= hi)))
        pct   = count / n_total * 100
        bar   = "█" * int(pct / 2)
        print(f"  {label:12s}: {count:6,}  ({pct:5.1f}%)  {bar}")

    print()
    print(f"  Mean swing_confirm : {sc_arr.mean():.3f}")
    print(f"  Median             : {float(np.median(sc_arr)):.1f}")
    print(f"  Max                : {int(sc_arr.max())}")
    print()
    pct_ge2 = float(np.sum(sc_arr >= REGIME_TREND_MIN_SWING_PAIRS)) / n_total * 100
    print(f"  % candle dengan swing_confirm >= {REGIME_TREND_MIN_SWING_PAIRS} "
          f"(threshold saat ini): {pct_ge2:.1f}%")
    pct_ge1 = float(np.sum(sc_arr >= 1)) / n_total * 100
    print(f"  % candle dengan swing_confirm >= 1: {pct_ge1:.1f}%")

    # ── LAPORAN METRIK 2: Cross-tab kegagalan trending ────────────────────────
    print(f"\n{'-' * 80}")
    print("METRIK 2: CROSS-TAB KEGAGALAN TRENDING (dari candle dengan ema_ok=True)")
    print(f"{'-' * 80}")
    total_ema_ok = ema_ok_failures["total_ema_ok"]
    # Candle ema_ok=True yang LULUS (trending terpenuhi) = total_ema_ok - yang gagal
    n_gagal_setelah_ema = (
        ema_ok_failures["struktur_saja"]
        + ema_ok_failures["konsistensi_saja"]
        + ema_ok_failures["keduanya"]
    )
    n_lulus = total_ema_ok - n_gagal_setelah_ema

    print(f"  Candle dengan ema_ok=True : {total_ema_ok:,}  "
          f"({total_ema_ok/total_eval*100:.1f}% dari total)")
    print(f"  Dari yang ema_ok=True:")
    if total_ema_ok > 0:
        print(f"    Lulus (terpenuhi=True)     : {n_lulus:6,}  "
              f"({n_lulus/total_ema_ok*100:.1f}%)")
        print(f"    Gagal karena struktur SAJA : {ema_ok_failures['struktur_saja']:6,}  "
              f"({ema_ok_failures['struktur_saja']/total_ema_ok*100:.1f}%)")
        print(f"    Gagal karena konsistensi SAJA: {ema_ok_failures['konsistensi_saja']:6,}  "
              f"({ema_ok_failures['konsistensi_saja']/total_ema_ok*100:.1f}%)")
        print(f"    Gagal karena KEDUANYA      : {ema_ok_failures['keduanya']:6,}  "
              f"({ema_ok_failures['keduanya']/total_ema_ok*100:.1f}%)")
    else:
        print("    (Tidak ada candle dengan ema_ok=True)")

    # ── LAPORAN METRIK 3: Histogram touches_R dan touches_S ──────────────────
    print(f"\n{'-' * 80}")
    print("METRIK 3: HISTOGRAM touches_R / touches_S (dari zona is_valid=True)")
    print(f"{'-' * 80}")
    n_valid_zones = len(zone_valid_touches_R)
    print(f"  Candle dengan zona is_valid=True : {n_valid_zones:,}  "
          f"({n_valid_zones/total_eval*100:.1f}% dari total)")

    if n_valid_zones > 0:
        r_arr = np.array(zone_valid_touches_R)
        s_arr = np.array(zone_valid_touches_S)

        touch_bins   = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 999)]
        touch_labels = ["0", "1", "2", "3-4", "5+"]

        print(f"\n  touches_RESISTANCE (threshold saat ini >= {REGIME_RANGE_MIN_TOUCHES_PER_SIDE}):")
        for (lo, hi), label in zip(touch_bins, touch_labels):
            count = int(np.sum((r_arr >= lo) & (r_arr <= hi)))
            pct   = count / n_valid_zones * 100
            bar   = "█" * int(pct / 3)
            print(f"    {label:5s}: {count:6,}  ({pct:5.1f}%)  {bar}")
        pct_r_ge2 = float(np.sum(r_arr >= REGIME_RANGE_MIN_TOUCHES_PER_SIDE)) / n_valid_zones * 100
        pct_r_ge1 = float(np.sum(r_arr >= 1)) / n_valid_zones * 100
        print(f"    % zona valid dgn touches_R >= {REGIME_RANGE_MIN_TOUCHES_PER_SIDE}: {pct_r_ge2:.1f}%")
        print(f"    % zona valid dgn touches_R >= 1: {pct_r_ge1:.1f}%")

        print(f"\n  touches_SUPPORT (threshold saat ini >= {REGIME_RANGE_MIN_TOUCHES_PER_SIDE}):")
        for (lo, hi), label in zip(touch_bins, touch_labels):
            count = int(np.sum((s_arr >= lo) & (s_arr <= hi)))
            pct   = count / n_valid_zones * 100
            bar   = "█" * int(pct / 3)
            print(f"    {label:5s}: {count:6,}  ({pct:5.1f}%)  {bar}")
        pct_s_ge2 = float(np.sum(s_arr >= REGIME_RANGE_MIN_TOUCHES_PER_SIDE)) / n_valid_zones * 100
        pct_s_ge1 = float(np.sum(s_arr >= 1)) / n_valid_zones * 100
        print(f"    % zona valid dgn touches_S >= {REGIME_RANGE_MIN_TOUCHES_PER_SIDE}: {pct_s_ge2:.1f}%")
        print(f"    % zona valid dgn touches_S >= 1: {pct_s_ge1:.1f}%")

        # Bottleneck: berapa % zona valid yang gagal HANYA karena salah satu sisi kurang?
        both_ok      = int(np.sum((r_arr >= REGIME_RANGE_MIN_TOUCHES_PER_SIDE) &
                                  (s_arr >= REGIME_RANGE_MIN_TOUCHES_PER_SIDE)))
        r_ok_s_fail  = int(np.sum((r_arr >= REGIME_RANGE_MIN_TOUCHES_PER_SIDE) &
                                  (s_arr <  REGIME_RANGE_MIN_TOUCHES_PER_SIDE)))
        r_fail_s_ok  = int(np.sum((r_arr <  REGIME_RANGE_MIN_TOUCHES_PER_SIDE) &
                                  (s_arr >= REGIME_RANGE_MIN_TOUCHES_PER_SIDE)))
        both_fail    = int(np.sum((r_arr <  REGIME_RANGE_MIN_TOUCHES_PER_SIDE) &
                                  (s_arr <  REGIME_RANGE_MIN_TOUCHES_PER_SIDE)))

        print(f"\n  Cross-tab (zona valid, threshold = {REGIME_RANGE_MIN_TOUCHES_PER_SIDE}):")
        print(f"    Keduanya cukup (RANGING lulus)       : {both_ok:6,}  ({both_ok/n_valid_zones*100:.1f}%)")
        print(f"    R ok, S kurang (gagal karena S)      : {r_ok_s_fail:6,}  ({r_ok_s_fail/n_valid_zones*100:.1f}%)")
        print(f"    R kurang, S ok (gagal karena R)      : {r_fail_s_ok:6,}  ({r_fail_s_ok/n_valid_zones*100:.1f}%)")
        print(f"    Keduanya kurang                      : {both_fail:6,}  ({both_fail/n_valid_zones*100:.1f}%)")

        # Dengan threshold >= 1 (hipotesis rekalibrasi)
        both_ok_v2 = int(np.sum((r_arr >= 1) & (s_arr >= 1)))
        print(f"\n  Simulasi threshold = 1 (hipotesis rekalibrasi):")
        print(f"    Zona valid yang akan LULUS RANGING   : {both_ok_v2:6,}  ({both_ok_v2/n_valid_zones*100:.1f}%)")
    else:
        print("  (Tidak ada zona valid yang ditemukan)")

    # ── LAPORAN METRIK 4: Episode breakout vs ranging ────────────────────────
    print(f"\n{'-' * 80}")
    print("METRIK 4: STATISTIK EPISODE BREAKOUT vs EPISODE RANGING")
    print(f"{'-' * 80}")

    # Hitung episode dari regime_sequence
    regime_arr = regime_sequence
    ep_ranging  = 0
    ep_breakout = 0
    prev_r = None
    for r in regime_arr:
        if r == "RANGING" and prev_r != "RANGING":
            ep_ranging += 1
        if r == "BREAKOUT_TRANSITION" and prev_r != "BREAKOUT_TRANSITION":
            ep_breakout += 1
        prev_r = r

    print(f"  Total candle RANGING            : {regime_arr.count('RANGING'):,}")
    print(f"  Total candle BREAKOUT_TRANSITION: {regime_arr.count('BREAKOUT_TRANSITION'):,}")
    print(f"  Total candle TRENDING           : {regime_arr.count('TRENDING'):,}")
    print(f"  Total candle CHOP               : {regime_arr.count('CHOP'):,}")
    print()
    print(f"  Episode RANGING valid (baru mulai)     : {ep_ranging:,}")
    print(f"  Episode BREAKOUT_TRANSITION (baru mulai): {ep_breakout:,}")

    if ep_ranging > 0:
        rasio = ep_breakout / ep_ranging * 100
        print(f"  Rasio breakout/ranging episode          : {ep_breakout}/{ep_ranging} = {rasio:.1f}%")
        print()
        print(f"  Interpretasi: dari setiap ~{ep_ranging/ep_breakout:.1f} episode ranging,")
        print(f"  rata-rata ada ~1 breakout yang lolos konfirmasi.")
    else:
        print("  (Tidak ada episode ranging)")

    # Statistik durasi episode
    print(f"\n  Durasi episode BREAKOUT_TRANSITION (distribusi):")
    bt_durations = []
    cur_dur = 0
    for r in regime_arr:
        if r == "BREAKOUT_TRANSITION":
            cur_dur += 1
        else:
            if cur_dur > 0:
                bt_durations.append(cur_dur)
                cur_dur = 0
    if cur_dur > 0:
        bt_durations.append(cur_dur)

    if bt_durations:
        bt_arr = np.array(bt_durations)
        print(f"    Durasi rata-rata  : {bt_arr.mean():.2f} candle")
        print(f"    Durasi median     : {float(np.median(bt_arr)):.1f} candle")
        print(f"    Durasi min/max    : {int(bt_arr.min())} / {int(bt_arr.max())} candle")
        print(f"    Episode 1 candle  : {int(np.sum(bt_arr == 1)):,} "
              f"({np.sum(bt_arr == 1)/len(bt_arr)*100:.1f}%)")
        print(f"    Episode 2-3 candle: {int(np.sum((bt_arr >= 2) & (bt_arr <= 3))):,} "
              f"({np.sum((bt_arr >= 2) & (bt_arr <= 3))/len(bt_arr)*100:.1f}%)")
        print(f"    Episode 4+ candle : {int(np.sum(bt_arr >= 4)):,} "
              f"({np.sum(bt_arr >= 4)/len(bt_arr)*100:.1f}%)")
    else:
        print("    (Tidak ada episode breakout)")

    print(f"\n{sep}")
    print("SELESAI — Gunakan hasil ini untuk menentukan rekalibrasi Cycle 2.")
    print(sep)


if __name__ == "__main__":
    main()
