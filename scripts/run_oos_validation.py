"""
scripts/run_oos_validation.py
==============================
Validasi Out-of-Sample (OOS) — Fase 2 Anti-Overfitting.

TUJUAN:
    Membuktikan bahwa parameter Risk Management hasil Fase 1 bukan sekadar
    overfit ke window data historis tertentu, tapi generalizable ke data baru
    yang belum pernah "dilihat" selama proses kalibrasi.

METODOLOGI:
    1. Split data historis menjadi dua periode yang TIDAK TUMPANG TINDIH:
         - Periode KALIBRASI : default 2026-01-01 s/d 2026-04-30
         - Periode VALIDASI  : default 2026-05-01 s/d 2026-07-25
    2. Jalankan grid search HANYA di periode kalibrasi
    3. Pilih parameter terbaik berdasarkan scoring komposit (bukan hanya PnL)
    4. Evaluasi parameter terpilih SATU KALI di periode validasi (tanpa tuning ulang)
    5. Bandingkan metrik kalibrasi vs validasi — tandai jika overfitting terdeteksi

FLAGGING OVERFITTING:
    Jika win_rate ATAU avg_rrr_realized di periode validasi turun >30% relatif
    terhadap periode kalibrasi (atau berbalik negatif), tampilkan peringatan
    "KEMUNGKINAN OVERFITTING" di terminal.

REUSE:
    Script ini memanfaatkan penuh fungsi dari run_param_sweep.py dan engine/:
    - run_fast_backtest() dari run_param_sweep (tidak tulis ulang dari nol)
    - run_all_indicators, merge_h1_to_m5, validate_no_lookahead, simulate_trade_outcome,
      compute_summary dari engine/

CLI ARGS:
    --calib-end YYYY-MM-DD   : Tanggal akhir periode kalibrasi (default: 2026-04-30)
    --val-start YYYY-MM-DD   : Tanggal mulai periode validasi (default: 2026-05-01)
    --min-trades INT         : Minimum total_trades agar kombinasi dianggap valid (default: 30)
    --overfitting-threshold  : Penurunan relatif (0-1) yang memicu flag overfitting (default: 0.30)
"""

import os
import sys
import argparse
import time
import itertools
import pandas as pd
import numpy as np
from datetime import timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.backtester import (
    merge_h1_to_m5,
    validate_no_lookahead,
    WARM_UP_CANDLES,
    MAX_FORWARD_CANDLES,
    DEFAULT_SPREAD_PTS,
)

# Reuse run_fast_backtest dari run_param_sweep — TIDAK duplikasi logika
from scripts.run_param_sweep import run_fast_backtest


# =============================================================================
# KONSTANTA DEFAULT
# =============================================================================

DEFAULT_CALIB_END  = "2026-04-30"
DEFAULT_VAL_START  = "2026-05-01"
DEFAULT_MIN_TRADES = 30
DEFAULT_OVF_THRESH = 0.30  # 30% penurunan relatif → flag overfitting

# Grid parameter — IDENTIK dengan run_param_sweep.py
ATR_MULTS = [0.7, 0.9, 1.1, 1.3]
LOOKBACKS  = [10, 15, 20, 30]
WINGS      = [2, 3, 4]
RRR_MINS   = [1.2, 1.3, 1.5, 1.8]


# =============================================================================
# SCORING KOMPOSIT (anti pure-PnL selection)
# =============================================================================

def compute_composite_score(row: dict, min_trades: int = DEFAULT_MIN_TRADES) -> float:
    """
    Hitung skor komposit dari metrik backtest untuk pemilihan kandidat terbaik.

    KOMPONEN SKOR (semuanya dinormalisasi ke skala yang sama):
        1. win_rate          : Proporsi trade TP di antara trade resolved (TP+SL)
                               Bobot tinggi — ini ukuran konsistensi edge.
        2. avg_rrr_realized  : Rata-rata RRR realized (TP+SL only). Harus positif.
        3. total_pnl_net     : Total P&L bersih setelah spread. Ukuran magnitude profit.
        4. max_drawdown_net  : Drawdown terbesar (negatif). Makin kecil magnitude = lebih baik.
        5. no_hit_rate_pct   : Persentase trade NO_HIT. Terlalu tinggi = sinyal tidak decisive.
                               Penalti jika > 15%.

    FILTER HARD:
        - Jika total_trades < min_trades → skor = -inf (dikecualikan dari seleksi)
        - Jika avg_rrr_realized <= 0 → skor = -inf (edge negatif, tidak berguna)
        - Jika win_rate is None → skor = -inf (tidak ada trade resolved)

    BOBOT SKOR:
        Komposit dirancang agar trade yang menang lebih konsisten (win_rate tinggi)
        dan tidak terlalu bergantung pada beberapa trade besar saja (via drawdown penalty).

    Return:
        float — skor komposit. Lebih tinggi = lebih baik.
        float("-inf") jika tidak memenuhi filter hard.
    """
    total_trades      = row.get("total_trades", 0)
    win_rate_pct      = row.get("win_rate_pct", 0) or 0.0
    avg_rrr_realized  = row.get("avg_rrr_realized") or 0.0
    total_pnl_net     = row.get("total_pnl_net") or 0.0
    max_drawdown_net  = row.get("max_drawdown_net") or 0.0
    no_hit_rate_pct   = row.get("no_hit_rate_pct", 0) or 0.0

    # Hard filters
    if total_trades < min_trades:
        return float("-inf")
    if avg_rrr_realized <= 0:
        return float("-inf")
    if win_rate_pct <= 0:
        return float("-inf")

    # Komponen skor positif
    score_wr    = win_rate_pct * 2.0          # bobot 2x untuk win_rate
    score_rrr   = avg_rrr_realized * 30.0     # skala ke ~rentang win_rate
    score_pnl   = total_pnl_net * 0.05        # pengaruh kecil agar tidak dominan
    score_dd    = abs(max_drawdown_net) * (-0.1)  # penalti drawdown

    # Penalti NO_HIT rate jika > 15%
    if no_hit_rate_pct > 15.0:
        penalty_no_hit = (no_hit_rate_pct - 15.0) * (-0.5)
    else:
        penalty_no_hit = 0.0

    return score_wr + score_rrr + score_pnl + score_dd + penalty_no_hit


# =============================================================================
# FILTER DATA BERDASARKAN PERIODE
# =============================================================================

def filter_period(df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    """
    Filter DataFrame berdasarkan rentang tanggal [date_from, date_to] (inklusif).

    Parameter:
        df        : DataFrame dengan DatetimeIndex ber-timezone UTC
        date_from : String 'YYYY-MM-DD' — tanggal mulai (inklusif)
        date_to   : String 'YYYY-MM-DD' — tanggal akhir (inklusif, hingga akhir hari)

    Return:
        DataFrame yang sudah di-filter
    """
    # Parse ke Timestamp dengan timezone UTC agar kompatibel dengan index
    ts_from = pd.Timestamp(date_from, tz="UTC")
    ts_to   = pd.Timestamp(date_to,   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= ts_from) & (df.index <= ts_to)].copy()


# =============================================================================
# JALANKAN SWEEP DI SATU PERIODE
# =============================================================================

def run_sweep_on_period(
    df_m5_ind : pd.DataFrame,
    df_merged : pd.DataFrame,
    label     : str,
    spread_pts: float = DEFAULT_SPREAD_PTS,
    max_candles: int  = MAX_FORWARD_CANDLES,
    warm_up   : int   = WARM_UP_CANDLES,
) -> pd.DataFrame:
    """
    Jalankan grid search 192 kombinasi pada DataFrame yang sudah di-filter dan di-proses.

    Parameter:
        df_m5_ind  : M5 dengan indikator (sudah di-filter ke periode)
        df_merged  : M5 + H1 merged (sudah di-filter ke periode)
        label      : Label periode untuk logging (misal "KALIBRASI", "VALIDASI")

    Return:
        DataFrame hasil sweep dengan semua metrik per kombinasi parameter.
    """
    grid = list(itertools.product(ATR_MULTS, LOOKBACKS, WINGS, RRR_MINS))
    total_combos = len(grid)

    print(f"\n-> Memulai Grid Search {total_combos} kombinasi di periode {label}...")
    print(f"   ATR Multipliers : {ATR_MULTS}")
    print(f"   Swing Lookbacks : {LOOKBACKS}")
    print(f"   Swing Wings     : {WINGS}")
    print(f"   Min RRRs        : {RRR_MINS}")
    print("-" * 70)

    results    = []
    start_time = time.time()

    for idx, (atr_mult, lookback, wing, rrr_min) in enumerate(grid, 1):
        trades_df, summary = run_fast_backtest(
            df_m5_ind   = df_m5_ind,
            df_merged   = df_merged,
            atr_mult    = atr_mult,
            lookback    = lookback,
            wing        = wing,
            rrr_min     = rrr_min,
            spread_pts  = spread_pts,
            max_candles = max_candles,
            warm_up     = warm_up,
        )

        wr             = summary.get("win_rate") or 0.0
        no_hit_rate    = summary.get("no_hit_rate") or 0.0
        avg_rrr        = summary.get("avg_rrr_realized")
        avg_rrr_all    = summary.get("avg_rrr_realized_all")
        total_pnl_net  = summary.get("total_pnl_net") or 0.0
        max_dd_net     = summary.get("max_drawdown_net") or 0.0
        total_trades   = summary.get("total_trades", 0)
        tp_count       = summary.get("tp_count", 0)
        sl_count       = summary.get("sl_count", 0)
        no_hit_count   = summary.get("no_hit_count", 0)

        row_dict = {
            "atr_multiplier"   : atr_mult,
            "swing_lookback"   : lookback,
            "swing_wing"       : wing,
            "rrr_min"          : rrr_min,
            "total_trades"     : total_trades,
            "tp_count"         : tp_count,
            "sl_count"         : sl_count,
            "no_hit_count"     : no_hit_count,
            "win_rate_pct"     : round(wr * 100, 2),
            "no_hit_rate_pct"  : round(no_hit_rate * 100, 2),
            "avg_rrr_realized" : avg_rrr,
            "avg_rrr_realized_all": avg_rrr_all,
            "total_pnl_net"    : total_pnl_net,
            "max_drawdown_net" : max_dd_net,
        }
        row_dict["composite_score"] = compute_composite_score(row_dict)
        results.append(row_dict)

    elapsed = time.time() - start_time
    print(f"✅ Sweep {label} selesai dalam {elapsed:.2f} detik ({total_combos} kombinasi).")
    return pd.DataFrame(results)


# =============================================================================
# CETAK TABEL PERBANDINGAN KALIBRASI vs VALIDASI
# =============================================================================

def print_comparison_table(
    calib_summary  : dict,
    val_summary    : dict,
    best_params    : dict,
    ovf_threshold  : float,
) -> None:
    """
    Cetak tabel perbandingan metrik antara periode kalibrasi dan validasi.
    Juga evaluasi dan flag potensi overfitting.
    """
    print("\n" + "=" * 70)
    print("  HASIL PERBANDINGAN: KALIBRASI vs VALIDASI (Out-of-Sample)")
    print("=" * 70)

    # Print parameter terpilih
    print(f"\n  Parameter terpilih dari kalibrasi:")
    print(f"    atr_multiplier = {best_params['atr_multiplier']}")
    print(f"    swing_lookback = {best_params['swing_lookback']}")
    print(f"    swing_wing     = {best_params['swing_wing']}")
    print(f"    rrr_min        = {best_params['rrr_min']}")
    print(f"    composite_score (kalibrasi) = {best_params.get('composite_score', 'N/A'):.4f}")

    # Metrik yang dibandingkan
    metrics = [
        ("win_rate",          "Win Rate (%)",         "win_rate",          100.0),
        ("avg_rrr_realized",  "Avg RRR Realized",     "avg_rrr_realized",  1.0),
        ("no_hit_rate",       "No-Hit Rate (%)",      "no_hit_rate",       100.0),
        ("total_pnl_net",     "Total PnL Net ($)",    "total_pnl_net",     1.0),
        ("max_drawdown_net",  "Max Drawdown Net ($)", "max_drawdown_net",  1.0),
        ("total_trades",      "Total Trades",         "total_trades",      1.0),
    ]

    print(f"\n  {'Metrik':<26} {'KALIBRASI':>14} {'VALIDASI':>14}")
    print(f"  {'-'*26} {'-'*14} {'-'*14}")

    calib_wr  = calib_summary.get("win_rate")
    val_wr    = val_summary.get("win_rate")
    calib_rrr = calib_summary.get("avg_rrr_realized")
    val_rrr   = val_summary.get("avg_rrr_realized")

    for key, label, sum_key, multiplier in metrics:
        c_val = calib_summary.get(sum_key)
        v_val = val_summary.get(sum_key)

        c_str = f"{c_val * multiplier:>+.2f}" if isinstance(c_val, (int, float)) and c_val is not None else "N/A"
        v_str = f"{v_val * multiplier:>+.2f}" if isinstance(v_val, (int, float)) and v_val is not None else "N/A"

        # Untuk total_trades: non-negative, format tanpa tanda +
        if sum_key == "total_trades":
            c_str = f"{c_val:>14}" if c_val is not None else "N/A"
            v_str = f"{v_val:>14}" if v_val is not None else "N/A"
            print(f"  {label:<26} {c_str:>14} {v_str:>14}")
        else:
            print(f"  {label:<26} {c_str:>14} {v_str:>14}")

    print(f"  {'='*56}")

    # ─── EVALUASI OVERFITTING ─────────────────────────────────────────────────
    print(f"\n  ── EVALUASI OVERFITTING (threshold: {ovf_threshold*100:.0f}% penurunan relatif) ──")

    overfitting_flags = []

    def check_metric(label, calib_val, val_val, threshold, lower_is_worse=True):
        """Cek apakah metrik turun lebih dari threshold secara relatif."""
        if calib_val is None or val_val is None:
            print(f"  ⚠️  {label}: data tidak tersedia untuk perbandingan")
            return False

        if lower_is_worse:
            # Metrik: lebih tinggi = lebih baik (win_rate, avg_rrr_realized)
            if calib_val <= 0:
                print(f"  ⚠️  {label}: nilai kalibrasi <= 0, tidak bisa hitung penurunan relatif")
                return False

            if val_val <= 0:
                # Berbalik negatif = overfitting kuat
                print(f"  🔴 {label}: VALIDASI BERBALIK NEGATIF "
                      f"(kalibrasi={calib_val:+.4f}, validasi={val_val:+.4f})")
                return True

            rel_drop = (calib_val - val_val) / abs(calib_val)
            if rel_drop > threshold:
                print(f"  🔴 {label}: turun {rel_drop*100:.1f}% "
                      f"(kalibrasi={calib_val:+.4f}, validasi={val_val:+.4f}, "
                      f"threshold={threshold*100:.0f}%)")
                return True
            else:
                print(f"  🟢 {label}: turun {rel_drop*100:.1f}% — dalam batas wajar "
                      f"(kalibrasi={calib_val:+.4f}, validasi={val_val:+.4f})")
                return False
        else:
            return False

    flag_wr  = check_metric("Win Rate",         calib_wr,  val_wr,  ovf_threshold, lower_is_worse=True)
    flag_rrr = check_metric("Avg RRR Realized", calib_rrr, val_rrr, ovf_threshold, lower_is_worse=True)

    overfitting_flags = [flag_wr, flag_rrr]

    print()
    if any(overfitting_flags):
        print("  " + "!" * 66)
        print("  !!  ⚠️  KEMUNGKINAN OVERFITTING TERDETEKSI                          !!")
        print("  !!                                                                  !!")
        print("  !!  Satu atau lebih metrik kunci turun signifikan (>30%) di         !!")
        print("  !!  periode validasi. Parameter Fase 1 mungkin terlalu fit ke       !!")
        print("  !!  data kalibrasi saja.                                            !!")
        print("  !!                                                                  !!")
        print("  !!  Rekomendasi: tinjau kembali parameter dengan data lebih panjang  !!")
        print("  !!  atau walk-forward testing sebelum deploy live.                  !!")
        print("  " + "!" * 66)
    else:
        print("  " + "✅" + " " * 62)
        print("  ✅  TIDAK ADA OVERFITTING SIGNIFIKAN TERDETEKSI")
        print("  ✅  Metrik kunci tidak turun lebih dari batas threshold di validasi.")
        print("  ✅  Parameter menunjukkan tanda-tanda generalisasi yang memadai.")
        print("  " + "-" * 64)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validasi Out-of-Sample Fase 2 — Anti-Overfitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calib-end",
        default=DEFAULT_CALIB_END,
        help=f"Tanggal akhir periode kalibrasi (YYYY-MM-DD, default: {DEFAULT_CALIB_END})",
    )
    parser.add_argument(
        "--val-start",
        default=DEFAULT_VAL_START,
        help=f"Tanggal mulai periode validasi (YYYY-MM-DD, default: {DEFAULT_VAL_START})",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=DEFAULT_MIN_TRADES,
        help=f"Minimum total_trades agar kombinasi masuk seleksi (default: {DEFAULT_MIN_TRADES})",
    )
    parser.add_argument(
        "--overfitting-threshold",
        type=float,
        default=DEFAULT_OVF_THRESH,
        help=f"Penurunan relatif (0-1) yang memicu flag overfitting (default: {DEFAULT_OVF_THRESH})",
    )
    args = parser.parse_args()

    calib_start = "2026-01-01"
    calib_end   = args.calib_end
    val_start   = args.val_start
    val_end     = "2026-07-25"
    min_trades  = args.min_trades
    ovf_thresh  = args.overfitting_threshold

    print("=" * 70)
    print("  OUT-OF-SAMPLE VALIDATION — FASE 2 ANTI-OVERFITTING")
    print("=" * 70)
    print(f"  Periode KALIBRASI : {calib_start} s/d {calib_end}")
    print(f"  Periode VALIDASI  : {val_start} s/d {val_end}")
    print(f"  Min trades filter : {min_trades}")
    print(f"  Overfitting flag  : penurunan relatif > {ovf_thresh*100:.0f}%")

    # ─── LOAD DATA ────────────────────────────────────────────────────────────
    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

    if not os.path.exists(m5_path) or not os.path.exists(h1_path):
        print(f"\n❌ File data historis tidak ditemukan:")
        print(f"   M5: {m5_path}")
        print(f"   H1: {h1_path}")
        sys.exit(1)

    print(f"\n-> Loading data M5 dan H1 dari CSV...")
    df_m5_raw = load_candles_csv(m5_path)
    df_h1_raw = load_candles_csv(h1_path)

    print(f"   M5 total: {len(df_m5_raw):,} candle ({df_m5_raw.index[0]} → {df_m5_raw.index[-1]})")
    print(f"   H1 total: {len(df_h1_raw):,} candle ({df_h1_raw.index[0]} → {df_h1_raw.index[-1]})")

    # ─── FILTER KE PERIODE MASING-MASING ─────────────────────────────────────
    df_m5_calib = filter_period(df_m5_raw, calib_start, calib_end)
    df_h1_calib = filter_period(df_h1_raw, calib_start, calib_end)
    df_m5_val   = filter_period(df_m5_raw, val_start,   val_end)
    df_h1_val   = filter_period(df_h1_raw, val_start,   val_end)

    print(f"\n-> Setelah filter periode:")
    print(f"   M5 kalibrasi  : {len(df_m5_calib):,} candle ({df_m5_calib.index[0]} → {df_m5_calib.index[-1]})")
    print(f"   H1 kalibrasi  : {len(df_h1_calib):,} candle ({df_h1_calib.index[0]} → {df_h1_calib.index[-1]})")
    print(f"   M5 validasi   : {len(df_m5_val):,} candle ({df_m5_val.index[0]} → {df_m5_val.index[-1]})")
    print(f"   H1 validasi   : {len(df_h1_val):,} candle ({df_h1_val.index[0]} → {df_h1_val.index[-1]})")

    # ─── HITUNG INDIKATOR & MERGE (sekali per periode) ────────────────────────
    print(f"\n-> Menghitung indikator M5 & H1 untuk periode KALIBRASI (1x)...")
    df_m5_calib_ind = run_all_indicators(df_m5_calib.copy())
    df_h1_calib_ind = run_all_indicators(df_h1_calib.copy())
    df_calib_merged = merge_h1_to_m5(df_m5_calib_ind, df_h1_calib_ind)

    print(f"-> Menghitung indikator M5 & H1 untuk periode VALIDASI (1x)...")
    df_m5_val_ind = run_all_indicators(df_m5_val.copy())
    df_h1_val_ind = run_all_indicators(df_h1_val.copy())
    df_val_merged = merge_h1_to_m5(df_m5_val_ind, df_h1_val_ind)

    # ─── VALIDASI ZERO LOOK-AHEAD ─────────────────────────────────────────────
    print(f"\n-> Validasi zero look-ahead (kalibrasi)...")
    val_check = validate_no_lookahead(df_m5_calib, n_samples=5)
    if not val_check["passed"]:
        raise RuntimeError(f"Look-ahead validation GAGAL di periode kalibrasi!\n{val_check['message']}")
    print(f"   {val_check['message']}")

    # ─── GRID SEARCH DI PERIODE KALIBRASI ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  LANGKAH 1: GRID SEARCH DI PERIODE KALIBRASI ({calib_start} → {calib_end})")
    print(f"{'='*70}")

    calib_results_df = run_sweep_on_period(
        df_m5_ind   = df_m5_calib_ind,
        df_merged   = df_calib_merged,
        label       = "KALIBRASI",
    )

    # Sort berdasarkan composite_score descending
    calib_results_df = calib_results_df.sort_values(
        by="composite_score", ascending=False
    ).reset_index(drop=True)

    # ─── PILIH KANDIDAT TERBAIK ───────────────────────────────────────────────
    # Filter: hanya kombinasi yang memenuhi min_trades dan composite_score > -inf
    valid_mask     = calib_results_df["composite_score"] > float("-inf")
    valid_results  = calib_results_df[valid_mask].reset_index(drop=True)

    print(f"\n   Dari {len(calib_results_df)} kombinasi, {len(valid_results)} memenuhi filter "
          f"(min_trades={min_trades}, avg_rrr>0).")

    if valid_results.empty:
        print("❌ Tidak ada kombinasi yang lolos filter. Coba turunkan --min-trades.")
        sys.exit(1)

    best_row    = valid_results.iloc[0].to_dict()
    best_params = {
        "atr_multiplier" : best_row["atr_multiplier"],
        "swing_lookback" : int(best_row["swing_lookback"]),
        "swing_wing"     : int(best_row["swing_wing"]),
        "rrr_min"        : best_row["rrr_min"],
        "composite_score": best_row["composite_score"],
    }

    print(f"\n  TOP 10 KOMBINASI TERBAIK DI KALIBRASI (berdasarkan Composite Score):")
    print(f"  {'Rank':<5} {'ATR':<5} {'Look':<4} {'Wing':<4} {'RRR':<4} "
          f"{'WinRate':>7} {'AvgRRR':>7} {'NoHit%':>7} {'NetPnL':>9} {'MaxDD':>9} {'Trades':>6} {'Score':>8}")
    print(f"  {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*6} {'-'*8}")

    for i in range(min(10, len(valid_results))):
        r = valid_results.iloc[i]
        avg_rrr_str = f"{r['avg_rrr_realized']:+.4f}" if r['avg_rrr_realized'] is not None else "  N/A"
        print(
            f"  {i+1:<5} {r['atr_multiplier']:<5.1f} {int(r['swing_lookback']):<4} "
            f"{int(r['swing_wing']):<4} {r['rrr_min']:<4.1f} "
            f"{r['win_rate_pct']:>6.1f}% {avg_rrr_str:>7} {r['no_hit_rate_pct']:>6.1f}% "
            f"{r['total_pnl_net']:>+9.1f} {r['max_drawdown_net']:>+9.1f} "
            f"{int(r['total_trades']):>6} {r['composite_score']:>8.4f}"
        )

    print(f"\n  ★ KANDIDAT TERPILIH (Rank 1 Composite Score):")
    print(f"    atr_multiplier = {best_params['atr_multiplier']}")
    print(f"    swing_lookback = {best_params['swing_lookback']}")
    print(f"    swing_wing     = {best_params['swing_wing']}")
    print(f"    rrr_min        = {best_params['rrr_min']}")

    # Ekstrak ringkasan kalibrasi untuk parameter terpilih
    calib_summary = {
        "win_rate"         : (best_row["win_rate_pct"] / 100.0) if best_row["win_rate_pct"] else None,
        "avg_rrr_realized" : best_row["avg_rrr_realized"],
        "no_hit_rate"      : (best_row["no_hit_rate_pct"] / 100.0) if best_row["no_hit_rate_pct"] else None,
        "total_pnl_net"    : best_row["total_pnl_net"],
        "max_drawdown_net" : best_row["max_drawdown_net"],
        "total_trades"     : int(best_row["total_trades"]),
    }

    # ─── EVALUASI SATU KALI DI PERIODE VALIDASI ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  LANGKAH 2: EVALUASI DI PERIODE VALIDASI ({val_start} → {val_end})")
    print(f"  [Parameter: atr={best_params['atr_multiplier']}, lookback={best_params['swing_lookback']}, "
          f"wing={best_params['swing_wing']}, rrr={best_params['rrr_min']}]")
    print(f"  CATATAN: Parameter ini TIDAK diubah selama evaluasi validasi.")
    print(f"{'='*70}")

    val_trades_df, val_summary_raw = run_fast_backtest(
        df_m5_ind   = df_m5_val_ind,
        df_merged   = df_val_merged,
        atr_mult    = best_params["atr_multiplier"],
        lookback    = best_params["swing_lookback"],
        wing        = best_params["swing_wing"],
        rrr_min     = best_params["rrr_min"],
        spread_pts  = DEFAULT_SPREAD_PTS,
        max_candles = MAX_FORWARD_CANDLES,
        warm_up     = WARM_UP_CANDLES,
    )

    val_wr = val_summary_raw.get("win_rate")
    val_nhr = val_summary_raw.get("no_hit_rate")

    val_summary = {
        "win_rate"         : val_wr,
        "avg_rrr_realized" : val_summary_raw.get("avg_rrr_realized"),
        "no_hit_rate"      : val_nhr,
        "total_pnl_net"    : val_summary_raw.get("total_pnl_net"),
        "max_drawdown_net" : val_summary_raw.get("max_drawdown_net"),
        "total_trades"     : val_summary_raw.get("total_trades", 0),
    }

    print(f"\n   ✅ Evaluasi validasi selesai: {val_summary['total_trades']} trade.")

    # ─── TABEL PERBANDINGAN & FLAG OVERFITTING ────────────────────────────────
    print_comparison_table(
        calib_summary = calib_summary,
        val_summary   = val_summary,
        best_params   = best_params,
        ovf_threshold = ovf_thresh,
    )

    # ─── SIMPAN HASIL KE CSV ──────────────────────────────────────────────────
    out_dir = os.path.join(ROOT_DIR, "data", "backtest_results")
    os.makedirs(out_dir, exist_ok=True)

    calib_out = os.path.join(out_dir, "oos_calib_sweep_results.csv")
    calib_results_df.to_csv(calib_out, index=False)
    print(f"\n📂 Hasil sweep kalibrasi ({len(calib_results_df)} kombinasi) → {calib_out}")

    if not val_trades_df.empty:
        val_out = os.path.join(out_dir, "oos_validation_trades.csv")
        val_trades_df.to_csv(val_out, index=False)
        print(f"📂 Detail trade validasi ({len(val_trades_df)} trade) → {val_out}")

    print("\n" + "=" * 70)
    print("  OOS VALIDATION SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()
