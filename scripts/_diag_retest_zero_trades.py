"""
scripts/_diag_retest_zero_trades.py
=====================================
Diagnostic: mengapa _check_retest_trigger() menghasilkan 0 trades?

Menelusuri candle per candle dan menghitung:
  - Berapa candle yang punya zona konsolidasi valid di j-1?
  - Berapa yang punya breakout event di candle j?
  - Berapa breakout yang tidak diinvalidasi?
  - Berapa yang punya retest touch?
  - Berapa yang punya konfirmasi body di idx?

Output: breakdown funnel lengkap + contoh kasus nyata.
"""

import os
import sys
import pandas as pd
import numpy as np
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators  import run_all_indicators, get_latest_signals
from engine.backtester  import merge_h1_to_m5, WARM_UP_CANDLES
from engine.zone_detector import detect_consolidation_zone
from engine.rule_engine import _check_breakout_trigger, _check_retest_trigger, _RETEST_SWING_BUFFER, _RETEST_BODY_MIN_RATIO
import math

M5_PATH = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
H1_PATH = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

def main():
    print("=" * 70)
    print("  DIAGNOSTIC: Mengapa RETEST = 0 trades?")
    print("=" * 70)

    df_m5 = load_candles_csv(M5_PATH)
    df_h1 = load_candles_csv(H1_PATH)

    print(f"\n  Loading & computing indicators...")
    df_m5_ind = run_all_indicators(df_m5.copy())
    df_h1_ind = run_all_indicators(df_h1.copy())
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind, h1_min_ema_gap_pct=0.02)

    n = len(df_m5_ind)
    warm_up = WARM_UP_CANDLES
    print(f"  Candle total: {n}, warm_up={warm_up}")
    print(f"  Scanning candle {warm_up} - {n-1}...\n")

    # ── FUNNEL DIAGNOSTIC ──────────────────────────────────────────────────
    # Untuk setiap idx dalam range, jalankan _check_retest_trigger dan 
    # lacak alasan kegagalan per step.
    counts = {
        "total_evaluated"   : 0,
        "no_zone_in_window" : 0,      # tidak ada zona valid sama sekali di window
        "zone_found"        : 0,      # ada zona valid di minimal 1 candle j
        "breakout_found"    : 0,      # ada breakout event
        "breakout_invalid"  : 0,      # breakout ditemukan tapi diinvalidasi
        "no_touch"          : 0,      # breakout valid tapi belum ada retest touch
        "touch_no_confirm"  : 0,      # touch ada tapi konfirmasi candle idx gagal
        "terpenuhi"         : 0,      # retest complete!
        "close_ok_fail"     : 0,      # touch ada, confirm gagal karena close
        "body_ok_fail"      : 0,      # touch ada, confirm gagal karena body
    }

    # Contoh kasus untuk tiap kategori
    examples = {}

    # Juga tracking: distribution zone_j per lookback position
    zone_hit_positions = []   # berapa candle mundur dari idx ketika zona ditemukan
    breakout_hit_positions = []

    # Sampling — jangan scan semua 26K candle (terlalu lama)
    # Scan 2000 candle: setiap 10 candle dari warm_up
    step_size = 5
    sample_indices = list(range(warm_up, min(n, warm_up + 5000), step_size))
    print(f"  Sampling {len(sample_indices)} indices (step={step_size})...")

    t0 = time.time()
    for idx in sample_indices:
        counts["total_evaluated"] += 1

        # Jalankan _check_retest_trigger secara langsung tapi dengan logging internal
        # Kita reproduksi logika internal untuk dapat funnel breakdown

        earliest_j = max(1, idx - 15)  # retest_lookback_candles=15
        retest_tol = 0.3

        found_zone_in_window = False
        found_breakout       = False
        found_valid_breakout = False
        found_touch          = False
        found_confirm        = False

        row_idx   = df_m5_ind.iloc[idx]
        close_idx = float(row_idx["close"])
        open_idx  = float(row_idx["open"])
        atr_idx   = float(row_idx["atr_14"])
        body_idx  = abs(close_idx - open_idx)

        for j in range(idx - 1, earliest_j - 1, -1):
            zone_j = detect_consolidation_zone(
                df_m5_ind, idx=j - 1,
                lookback=20, max_range_atr_ratio=2.5, min_duration_candles=10,
            )
            if not zone_j.get("is_valid", False):
                continue

            found_zone_in_window = True
            zone_hit_positions.append(idx - j)  # candle mundur

            row_j = df_m5_ind.iloc[j]
            vr_j  = None
            if "volume_ratio" in df_m5_ind.columns:
                try:
                    vr_f = float(row_j.get("volume_ratio", float("nan")))
                    if not math.isnan(vr_f):
                        vr_j = vr_f
                except: pass

            signals_j = {
                "close"       : float(row_j["close"]),
                "open"        : float(row_j["open"]),
                "atr_14"      : float(row_j["atr_14"]),
                "volume_ratio": vr_j,
            }
            c_bo = _check_breakout_trigger(signals_j, zone_j)
            if not c_bo["terpenuhi"]:
                continue

            found_breakout = True
            breakout_hit_positions.append(idx - j)
            arah_bo   = c_bo["arah"]
            resistance = float(zone_j["resistance"])
            support    = float(zone_j["support"])
            level_ref  = resistance if arah_bo == "BUY" else support

            # Cek invalidasi
            invalidated = False
            for m in range(j + 1, idx):
                close_m = float(df_m5_ind.iloc[m]["close"])
                if arah_bo == "BUY"  and close_m < resistance - _RETEST_SWING_BUFFER:
                    invalidated = True; break
                if arah_bo == "SELL" and close_m > support    + _RETEST_SWING_BUFFER:
                    invalidated = True; break

            if invalidated:
                counts["breakout_invalid"] += 1
                continue

            found_valid_breakout = True

            # Cek retest touch
            found_touch_inner = False
            for k in range(j + 1, idx + 1):
                row_k  = df_m5_ind.iloc[k]
                low_k  = float(row_k["low"])
                high_k = float(row_k["high"])
                atr_k  = float(row_k["atr_14"])
                tol    = retest_tol * atr_k
                if arah_bo == "BUY"  and (resistance - tol) <= low_k  <= (resistance + tol):
                    found_touch_inner = True; break
                if arah_bo == "SELL" and (support    - tol) <= high_k <= (support    + tol):
                    found_touch_inner = True; break

            if not found_touch_inner:
                counts["no_touch"] += 1
                if "no_touch" not in examples:
                    examples["no_touch"] = {
                        "idx": idx, "j": j, "arah": arah_bo,
                        "level_ref": level_ref, "resistance": resistance,
                        "support": support,
                        "close_idx": close_idx, "atr_idx": atr_idx,
                        "tol": retest_tol * atr_idx,
                        "candles_since_bo": idx - j,
                    }
                break  # breakout termuda valid

            found_touch = True

            # Cek konfirmasi di idx
            close_ok = (arah_bo == "BUY" and close_idx > resistance) or \
                       (arah_bo == "SELL" and close_idx < support)
            body_ok  = atr_idx > 0 and body_idx >= _RETEST_BODY_MIN_RATIO * atr_idx

            if close_ok and body_ok:
                found_confirm = True
                counts["terpenuhi"] += 1
                if "terpenuhi" not in examples:
                    examples["terpenuhi"] = {
                        "idx": idx, "j": j, "arah": arah_bo, "level_ref": level_ref,
                        "close_idx": close_idx, "body_idx": body_idx, "atr_idx": atr_idx,
                    }
                break

            # Touch tapi konfirmasi gagal
            counts["touch_no_confirm"] += 1
            if not close_ok:
                counts["close_ok_fail"] += 1
                if "touch_close_fail" not in examples:
                    examples["touch_close_fail"] = {
                        "idx": idx, "j": j, "arah": arah_bo, "level_ref": level_ref,
                        "close_idx": close_idx, "atr_idx": atr_idx,
                    }
            if not body_ok:
                counts["body_ok_fail"] += 1
                if "touch_body_fail" not in examples:
                    examples["touch_body_fail"] = {
                        "idx": idx, "j": j, "arah": arah_bo,
                        "body_idx": body_idx, "atr_idx": atr_idx,
                        "threshold": _RETEST_BODY_MIN_RATIO * atr_idx,
                    }
            break

        if not found_zone_in_window:
            counts["no_zone_in_window"] += 1
        elif found_zone_in_window and not found_breakout:
            pass  # zona ada tapi tidak ada breakout candle — sudah terhitung di loop
        elif found_breakout and not found_valid_breakout:
            pass  # terhitung di breakout_invalid
        elif found_valid_breakout and not found_touch:
            pass  # terhitung di no_touch
        elif found_touch and not found_confirm:
            pass  # terhitung di touch_no_confirm

        if found_zone_in_window:
            counts["zone_found"] += 1
        if found_breakout:
            counts["breakout_found"] += 1

    elapsed = time.time() - t0
    print(f"  Selesai dalam {elapsed:.1f}s\n")

    # ── PRINT FUNNEL ──────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"  FUNNEL BREAKDOWN (sample {len(sample_indices)} indices, step={step_size})")
    print(f"{'─'*70}")

    tot = counts["total_evaluated"]
    def pct(n, d=None):
        d = d or tot
        return f"{n:>6} ({n/d*100:5.1f}%)" if d > 0 else f"{n:>6} (N/A)"

    print(f"  Candle dievaluasi          : {tot}")
    print(f"  Tidak ada zona di window   : {pct(counts['no_zone_in_window'])}")
    print(f"  Ada zona di window         : {pct(counts['zone_found'])}")
    print(f"  Ada breakout event         : {pct(counts['breakout_found'])}")
    print(f"  Breakout diinvalidasi      : {pct(counts['breakout_invalid'])}")
    print(f"  Tidak ada retest touch     : {pct(counts['no_touch'])}")
    print(f"  Touch tapi confirm gagal   : {pct(counts['touch_no_confirm'])}")
    print(f"    -> close_ok gagal        : {pct(counts['close_ok_fail'])}")
    print(f"    -> body_ok gagal         : {pct(counts['body_ok_fail'])}")
    print(f"  RETEST TERPENUHI           : {pct(counts['terpenuhi'])}")

    # ── ANALISIS DISTRIBUSI ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DISTRIBUSI: Posisi zona ditemukan (candle mundur dari idx)")
    print(f"{'─'*70}")
    if zone_hit_positions:
        arr = np.array(zone_hit_positions)
        print(f"  count={len(arr)}, mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
              f"min={arr.min()}, max={arr.max()}")
        # Histogram sederhana
        bins = [1, 3, 6, 10, 15]
        for lo, hi in zip(bins[:-1], bins[1:]):
            cnt = ((arr >= lo) & (arr < hi)).sum()
            print(f"    [{lo:2d}-{hi:2d} candle mundur]: {cnt}")
    else:
        print("  (tidak ada zona ditemukan)")

    print(f"\n  DISTRIBUSI: Posisi breakout (candle mundur dari idx)")
    if breakout_hit_positions:
        arr = np.array(breakout_hit_positions)
        print(f"  count={len(arr)}, mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
              f"min={arr.min()}, max={arr.max()}")
    else:
        print("  (tidak ada breakout ditemukan)")

    # ── ANALISIS ZONA ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  ANALISIS ZONA: berapa banyak zona valid secara global?")
    print(f"{'─'*70}")
    n_zone_valid = 0
    n_bo_valid   = 0
    for check_idx in range(warm_up, min(n, warm_up + 1000), 20):
        zone_check = detect_consolidation_zone(df_m5_ind, idx=check_idx,
                        lookback=20, max_range_atr_ratio=2.5, min_duration_candles=10)
        if zone_check.get("is_valid"):
            n_zone_valid += 1
            # Cek apakah candle check_idx+1 adalah breakout
            if check_idx + 1 < n:
                row_j = df_m5_ind.iloc[check_idx + 1]
                vr = None
                try:
                    vr_f = float(row_j.get("volume_ratio", float("nan")))
                    if not math.isnan(vr_f):
                        vr = vr_f
                except: pass
                sig_j = {"close": float(row_j["close"]), "open": float(row_j["open"]),
                         "atr_14": float(row_j["atr_14"]), "volume_ratio": vr}
                c = _check_breakout_trigger(sig_j, zone_check)
                if c["terpenuhi"]:
                    n_bo_valid += 1

    n_checked = len(range(warm_up, min(n, warm_up + 1000), 20))
    print(f"  Diperiksa {n_checked} posisi (step=20)")
    print(f"  Zona valid              : {n_zone_valid} ({100*n_zone_valid/n_checked:.1f}%)")
    print(f"  Breakout setelah zona   : {n_bo_valid} ({100*n_bo_valid/n_checked:.1f}%)")

    # ── CONTOH KASUS ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  CONTOH KASUS PER KATEGORI")
    print(f"{'─'*70}")
    for cat, ex in examples.items():
        print(f"\n  [{cat}]")
        for k, v in ex.items():
            print(f"    {k}: {v}")

    # ── DIAGNOSIS KUNCI ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DIAGNOSIS KUNCI")
    print(f"{'─'*70}")

    no_zone_pct  = counts["no_zone_in_window"] / tot * 100 if tot > 0 else 0
    no_touch_pct = counts["no_touch"] / max(counts["breakout_found"], 1) * 100
    invalid_pct  = counts["breakout_invalid"] / max(counts["breakout_found"], 1) * 100

    print(f"\n  1. Zona tidak ditemukan di {no_zone_pct:.1f}% kasus")
    print(f"     -> detect_consolidation_zone (lookback=20, min_duration=10, max_range=2.5x ATR)")
    print(f"     -> Kemungkinan: XAUUSD M5 2026 terlalu volatile/trendy, jarang konsolidasi 10+ candle")
    print(f"     -> Cek: perlu min_duration lebih pendek atau max_range_atr_ratio lebih besar?")

    print(f"\n  2. Breakout diinvalidasi: {invalid_pct:.1f}% dari breakout yang ditemukan")
    print(f"     -> SWING_BUFFER = {_RETEST_SWING_BUFFER} — harga kembali masuk zona dalam {invalid_pct:.1f}% kasus")

    print(f"\n  3. Tidak ada retest touch: {no_touch_pct:.1f}% dari breakout valid yang ditemukan")
    print(f"     -> retest_tolerance_atr = 0.3 * ATR mungkin terlalu ketat")
    print(f"     -> Rata-rata ATR M5: {df_m5_ind['atr_14'].mean():.3f}")
    print(f"     -> Rata-rata tol: {0.3 * df_m5_ind['atr_14'].mean():.3f}")

    print(f"\n  4. Touch tapi konfirmasi gagal: {counts['touch_no_confirm']} kasus")
    print(f"     -> close_ok fail: {counts['close_ok_fail']}, body_ok fail: {counts['body_ok_fail']}")

    print(f"\n  KESIMPULAN DIAGNOSTIC:")
    if no_zone_pct > 70:
        print(f"  => BOTTLENECK UTAMA: Zona konsolidasi sangat jarang terdeteksi ({100-no_zone_pct:.1f}% ada zona)")
        print(f"     Kemungkinan penyebab: parameter zona (lookback=20, min_duration=10) terlalu ketat")
        print(f"     untuk data XAUUSD M5 2026 yang cenderung trending.")
        print(f"     Rekomendasi: pertimbangkan relaksasi min_duration_candles atau min_range_atr_ratio")
        print(f"     di kalibrasi Fase 11 — BUKAN sekarang (observasi dulu).")
    elif counts["no_touch"] > counts["breakout_found"] * 0.5:
        print(f"  => BOTTLENECK UTAMA: Breakout ditemukan tapi tidak ada retest touch")
        print(f"     retest_tolerance_atr=0.3 mungkin terlalu ketat.")
    else:
        print(f"  => Bottleneck tersebar. Perlu analisis lebih detail.")

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SELESAI")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
