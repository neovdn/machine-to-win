"""
engine/backtester.py
====================
Backtest engine rule-based untuk evaluasi performa historis sinyal XAUUSD M5.

FILOSOFI DESAIN — O(n), BUKAN O(n²):
    Indikator EMA (.ewm(adjust=False)), RSI (Wilder's Smoothing via ta library),
    dan ATR (.ewm(alpha=1/period, adjust=False)) semuanya CAUSAL by construction:
    nilai di baris ke-i hanya bergantung pada data di index ≤ i.

    Ini bukan asumsi — ini fakta dari cara pandas .ewm() bekerja:
    .ewm(adjust=False) menggunakan rumus rekursif forward-pass:
        EMA[i] = alpha * price[i] + (1-alpha) * EMA[i-1]
    Nilai di baris i sama sekali tidak dipengaruhi oleh baris i+1, i+2, dst.
    Berbeda hanya jika center=True atau shift(-1) dipakai — keduanya tidak ada di sini.

    Konsekuensinya: run_all_indicators() cukup dipanggil SATU KALI untuk seluruh
    DataFrame historis. Nilai df.iloc[i] identik persis dengan hasil recompute dari
    df.iloc[:i+1]. Ini mengubah kompleksitas dari O(n²) → O(n).

    Untuk H1: hitung sekali, lalu gunakan pd.merge_asof(direction="backward")
    untuk attach nilai H1 terakhir yang sudah closed ke tiap baris M5 berdasarkan
    timestamp. direction="backward" menjamin tidak ada leak dari H1 candle yang
    closing-nya setelah waktu M5 candle tersebut.

SATU-SATUNYA EXCEPTION — find_nearest_swing():
    Swing detection menggunakan custom loop (bukan rolling pandas standar) dan
    harus dipanggil per-sinyal dengan potongan data df.iloc[:i+1].
    Tapi ini hanya dijalankan saat ada sinyal BUY/SELL — bukan tiap candle —
    sehingga dampak performa tetap kecil.

ZERO LOOK-AHEAD GUARANTEE:
    Dibuktikan secara empiris oleh validate_no_lookahead(), bukan hanya diklaim.
    Fungsi tersebut memverifikasi bahwa nilai indikator di baris i tidak berubah
    ketika df dipotong hingga i+1 vs df penuh (sampling 5 titik acak).

ALUR BACKTEST:
    1. run_all_indicators(df_m5_full)  — satu kali, O(n)
    2. run_all_indicators(df_h1_full)  — satu kali, O(n)
    3. merge_asof(df_m5, df_h1, direction="backward")  — attach H1 ke M5, O(n log n)
    4. Loop tiap candle i (dari warm_up ke end):
         a. Baca df_merged.iloc[i]          — nilai indikator sudah ada, causal
         b. Bangun signals dict             — persis format live pipeline
         c. evaluate_entry(signals)        — rule engine identik dengan live
         d. Jika BUY/SELL:
              find_nearest_swing(df.iloc[:i+1])  — per-sinyal, hanya saat diperlukan
              calculate_sl_tp(...)               — SL/TP identik dengan live
              simulate_trade_outcome(...)        — forward-scan OHLC
    5. compute_summary()               — agregasi semua metrik

TIDAK ADA AI / MACHINE LEARNING — murni if-else rule-based, identik dengan live.
"""

import numpy as np
import pandas as pd

from engine.indicators   import run_all_indicators
from engine.rule_engine  import evaluate_entry
from engine.risk_manager import calculate_sl_tp, find_nearest_swing


# =============================================================================
# KONSTANTA KONFIGURASI DEFAULT
# =============================================================================

# Jumlah candle awal yang dilewati agar indikator fully converged.
# EMA 21 butuh minimal 21 candle, tapi 100 lebih aman (EWM konvergen lebih baik).
WARM_UP_CANDLES = 100

# Batas forward scan per trade: 288 M5 candle = 24 jam "trading time".
# CATATAN: ini adalah jumlah CANDLE, bukan jam kalender.
# Jika sinyal muncul Jumat sore, 288 candle ke depan bisa merentang sampai
# Senin/Selasa karena candle dihitung per open market, bukan per jam nyata.
# Dokumentasikan ini saat membaca "candles_held" di output CSV.
MAX_FORWARD_CANDLES = 288

# Jarak SL minimum yang dianggap valid (dollar).
# Trade dengan jarak SL di bawah ini diabaikan — kemungkinan anomali data.
MIN_SL_DISTANCE = 0.10


# =============================================================================
# BAGIAN 1: VALIDASI ZERO LOOK-AHEAD (EMPIRIS)
# =============================================================================

def validate_no_lookahead(
    df_m5     : pd.DataFrame,
    n_samples : int = 5,
    seed      : int = 42,
) -> dict:
    """
    Validasi empiris bahwa indikator bersifat causal (zero look-ahead bias).

    CARA KERJA:
        Untuk n_samples titik acak i, bandingkan nilai indikator dari:
          A) run_all_indicators(df_full).iloc[i]       — dihitung dari seluruh data
          B) run_all_indicators(df[:i+1]).iloc[-1]     — dihitung hanya dari data ≤ i

        Jika A == B (dalam toleransi floating point 1e-6), indikator terbukti causal.

    KENAPA TOLERANSI 1e-6:
        Perbedaan sangat kecil bisa muncul dari floating point rounding order
        (bukan look-ahead). Toleransi 1e-6 jauh lebih kecil dari tick size
        XAUUSD ($0.01) sehingga aman untuk trading purposes.

    KOLOM YANG DIVALIDASI:
        ema_9, ema_21, rsi_14, atr_14
        Semua dihitung via .ewm() atau Wilder's Smoothing — causal by construction.

    Parameter:
        df_m5     : DataFrame M5 MENTAH (sebelum run_all_indicators)
                    — dipakai untuk compute indikator secara independen
        n_samples : Berapa titik acak yang diuji (default: 5)
        seed      : Random seed untuk reproducibility (default: 42)

    Return:
        dict berisi:
            "passed"   : bool  — True jika semua sample lolos (diff ≤ 1e-6)
            "n_tested" : int   — berapa titik yang diuji
            "details"  : list  — detail per sample [{index, time, ok, diffs}, ...]
            "message"  : str   — ringkasan hasil satu baris
    """
    rng     = np.random.default_rng(seed)
    details = []
    all_ok  = True

    indicator_cols = ["ema_9", "ema_21", "rsi_14", "atr_14"]
    tolerance      = 1e-6

    # Hitung indikator untuk seluruh data (cara yang dipakai backtest)
    df_full_ind = run_all_indicators(df_m5.copy())

    # Pilih titik uji dari tengah data — hindari edge kiri (warm-up) dan kanan (-)
    min_idx = WARM_UP_CANDLES + 10
    max_idx = len(df_m5) - 10

    if max_idx <= min_idx:
        return {
            "passed"   : False,
            "n_tested" : 0,
            "details"  : [],
            "message"  : (
                f"Data terlalu sedikit untuk validasi "
                f"(butuh minimal {min_idx + 10} candle, ada {len(df_m5)})"
            ),
        }

    sample_indices = rng.integers(min_idx, max_idx, size=n_samples)

    for raw_i in sample_indices:
        i = int(raw_i)

        # Hitung ulang dari potongan data s/d baris i saja
        df_slice_ind = run_all_indicators(df_m5.iloc[: i + 1].copy())

        val_slice = df_slice_ind.iloc[-1]   # baris i, dihitung dari potongan
        val_full  = df_full_ind.iloc[i]     # baris i, dihitung dari data penuh

        sample_ok = True
        diffs     = {}

        for col in indicator_cols:
            if col not in df_slice_ind.columns or col not in df_full_ind.columns:
                continue

            v_slice = float(val_slice[col])
            v_full  = float(val_full[col])
            diff    = abs(v_slice - v_full)

            if np.isnan(diff) or diff > tolerance:
                sample_ok = False
                all_ok    = False

            diffs[col] = {
                "slice": round(v_slice, 8),
                "full" : round(v_full,  8),
                "diff" : diff,
            }

        details.append({
            "index" : i,
            "time"  : str(df_m5.index[i]),
            "ok"    : sample_ok,
            "diffs" : diffs,
        })

    n_passed = sum(1 for d in details if d["ok"])
    passed   = all_ok and n_passed == n_samples

    if passed:
        msg = (
            f"validate_no_lookahead: {n_passed}/{n_samples} titik lolos "
            f"(max diff <= {tolerance}). PASSED -- indikator terbukti causal."
        )
    else:
        msg = (
            f"validate_no_lookahead: {n_passed}/{n_samples} titik lolos. "
            f"FAILED -- ada perbedaan nilai indikator antara full vs sliced!"
        )

    return {
        "passed"   : passed,
        "n_tested" : n_samples,
        "details"  : details,
        "message"  : msg,
    }


# =============================================================================
# BAGIAN 2: MERGE H1 → M5 (ANTI-LOOKAHEAD via merge_asof)
# =============================================================================

def merge_h1_to_m5(
    df_m5_ind : pd.DataFrame,
    df_h1_ind : pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach nilai indikator H1 ke setiap baris M5 berdasarkan timestamp.

    PRINSIP ANTI-LOOKAHEAD (direction="backward"):
        Untuk setiap candle M5 dengan waktu t, kita attach nilai dari H1 candle
        yang SUDAH CLOSED sebelum atau tepat pada t:
            h1_attached = max(h1.time) WHERE h1.time <= m5.time

        pd.merge_asof dengan direction="backward" melakukan ini secara efisien.

        Contoh:
            M5 candle waktu 12:05 → attach H1 candle waktu 12:00 (bukan 13:00)
            M5 candle waktu 12:55 → attach H1 candle waktu 12:00 (bukan 13:00)
            M5 candle waktu 13:05 → attach H1 candle waktu 13:00

        H1 candle masa depan TIDAK PERNAH bocor ke data M5.

    KOLOM H1 YANG DIATTACH:
        trend_h1, ema_gap_pct_h1, ema_9_h1, ema_21_h1
        (direname agar tidak clash dengan kolom M5 yang sama namanya)

    Parameter:
        df_m5_ind : DataFrame M5 yang sudah melewati run_all_indicators()
                    — harus punya DatetimeIndex ber-timezone
        df_h1_ind : DataFrame H1 yang sudah melewati run_all_indicators()
                    — harus punya DatetimeIndex ber-timezone

    Return:
        DataFrame M5 dengan kolom tambahan dari H1 yang sudah diattach
    """
    # merge_asof butuh kolom, bukan DatetimeIndex
    m5_reset = df_m5_ind.reset_index()   # 'time' jadi kolom reguler
    h1_reset = df_h1_ind.reset_index()

    # Pastikan sorted ascending — syarat merge_asof
    m5_reset = m5_reset.sort_values("time").reset_index(drop=True)
    h1_reset = h1_reset.sort_values("time").reset_index(drop=True)

    # Pilih kolom H1 yang relevan, rename agar tidak clash
    h1_rename = {
        "trend"       : "trend_h1",
        "ema_gap_pct" : "ema_gap_pct_h1",
        "ema_9"       : "ema_9_h1",
        "ema_21"      : "ema_21_h1",
    }
    h1_slim = h1_reset[["time"] + list(h1_rename.keys())].rename(columns=h1_rename)

    # merge_asof: untuk setiap baris M5, cari baris H1 terbaru dengan h1.time <= m5.time
    # direction="backward" = anti-lookahead
    merged = pd.merge_asof(
        m5_reset,
        h1_slim,
        on        = "time",
        direction = "backward",
    )

    # Kembalikan ke DatetimeIndex
    merged = merged.set_index("time")
    return merged


# =============================================================================
# BAGIAN 3: SIMULASI OUTCOME TRADE
# =============================================================================

def simulate_trade_outcome(
    df_m5_full  : pd.DataFrame,
    entry_idx   : int,
    entry       : float,
    sl          : float,
    tp          : float,
    max_candles : int = MAX_FORWARD_CANDLES,
) -> dict:
    """
    Simulasikan apakah SL atau TP yang pertama kali tersentuh setelah entry.

    CARA KERJA:
        Iterasi candle ke-(entry_idx+1), (entry_idx+2), ... secara berurutan.
        Untuk setiap candle, cek:
            low  ≤ sl  →  SL_HIT (posisi terhenti, loss)
            high ≥ tp  →  TP_HIT (posisi mencapai target, profit)

        Berhenti di candle pertama yang memenuhi kondisi.

    AMBIGUOUS CANDLE (SL dan TP kena di candle yang sama):
        Jika low ≤ sl DAN high ≥ tp di candle yang sama, kita tidak tahu mana
        yang pertama tanpa data tick atau M1. Kebijakan: KONSERVATIF = SL_HIT.
        Ini worst-case assumption — lebih baik underestimate profit daripada
        overestimate. Candle ini ditandai ambiguous_candle=True di output.

        Jika ambiguous_rate > 5-10% dari total trade, ini sinyal bahwa granularity
        M5 kurang presisi untuk menilai kualitas rule engine secara adil.

    BATAS FORWARD SCAN (max_candles):
        Jika tidak ada hit dalam max_candles candle → NO_HIT.
        NO_HIT rate tinggi (>15%) = sinyal penting — bisa berarti:
        - max_candles terlalu kecil
        - Rule engine menghasilkan sinyal yang tidak decisive (harga sideways)
        Jangan abaikan NO_HIT — ia justru salah satu metrik paling informatif.

    Parameter:
        df_m5_full  : DataFrame M5 LENGKAP dengan kolom high, low
                      (sudah melewati run_all_indicators — hanya pakai OHLC di sini)
        entry_idx   : Integer index (iloc) candle tempat sinyal muncul
        entry       : Harga entry (untuk dokumentasi, tidak dipakai di logika cek)
        sl          : Level Stop Loss
        tp          : Level Take Profit
        max_candles : Batas candle forward yang di-scan per trade

    Return:
        dict berisi:
            "outcome"          : "TP_HIT" / "SL_HIT" / "NO_HIT"
            "candles_held"     : int   — candle yang di-scan (0 jika langsung hit)
            "exit_time"        : str   — waktu candle saat exit (None jika NO_HIT)
            "exit_price"       : float — estimasi harga exit: sl, tp, atau None
            "ambiguous_candle" : bool  — True jika SL dan TP kena candle yang sama
    """
    n_total    = len(df_m5_full)
    scan_start = entry_idx + 1
    scan_end   = min(entry_idx + 1 + max_candles, n_total)

    for j in range(scan_start, scan_end):
        row    = df_m5_full.iloc[j]
        low    = float(row["low"])
        high   = float(row["high"])

        if sl < tp:
            # BUY position: SL is below entry, TP is above entry
            sl_hit = low  <= sl
            tp_hit = high >= tp
        else:
            # SELL position: SL is above entry, TP is below entry
            sl_hit = high >= sl
            tp_hit = low  <= tp

        if sl_hit and tp_hit:
            # Ambiguous: keduanya dalam range candle yang sama
            # Konservatif: anggap SL hit (worst case)
            return {
                "outcome"          : "SL_HIT",
                "candles_held"     : j - entry_idx,
                "exit_time"        : str(df_m5_full.index[j]),
                "exit_price"       : sl,
                "ambiguous_candle" : True,
            }

        if sl_hit:
            return {
                "outcome"          : "SL_HIT",
                "candles_held"     : j - entry_idx,
                "exit_time"        : str(df_m5_full.index[j]),
                "exit_price"       : sl,
                "ambiguous_candle" : False,
            }

        if tp_hit:
            return {
                "outcome"          : "TP_HIT",
                "candles_held"     : j - entry_idx,
                "exit_time"        : str(df_m5_full.index[j]),
                "exit_price"       : tp,
                "ambiguous_candle" : False,
            }

    # Tidak ada hit dalam window max_candles
    return {
        "outcome"          : "NO_HIT",
        "candles_held"     : scan_end - scan_start,
        "exit_time"        : None,
        "exit_price"       : None,
        "ambiguous_candle" : False,
    }


# =============================================================================
# BAGIAN 4: FUNGSI UTAMA BACKTEST
# =============================================================================

def run_backtest(
    df_m5        : pd.DataFrame,
    df_h1        : pd.DataFrame,
    warm_up      : int  = WARM_UP_CANDLES,
    max_candles  : int  = MAX_FORWARD_CANDLES,
    verbose      : bool = True,
) -> tuple:
    """
    Jalankan backtest rule-based untuk seluruh DataFrame historis.

    ALUR LENGKAP:
        1. Hitung indikator M5 dan H1 masing-masing SATU KALI (O(n))
        2. Merge H1 ke M5 via merge_asof backward (O(n log n))
        3. Validasi zero look-ahead secara empiris (sampling acak)
        4. Loop tiap candle i (dari warm_up ke end):
             - Baca df_merged.iloc[i] — nilai indikator sudah causal
             - Bangun signals dict identik dengan format live pipeline
             - evaluate_entry(signals) — rule engine persis sama dengan live
             - Jika BUY/SELL:
                 a. find_nearest_swing(df.iloc[:i+1]) via calculate_sl_tp()
                 b. simulate_trade_outcome() — forward-scan OHLC
             - Skip candle yang masih dalam trade (satu posisi satu waktu)
        5. compute_summary() — agregasi semua metrik

    SATU POSISI PER WAKTU:
        Selama ada trade yang masih "berjalan" (belum hit SL/TP/NO_HIT),
        sinyal baru diabaikan. Ini mencerminkan kondisi trading nyata.
        Untuk NO_HIT: posisi dianggap tidak ada (force-close di akhir window),
        sehingga sinyal berikutnya tetap bisa dievaluasi.

    Parameter:
        df_m5       : DataFrame M5 MENTAH (belum ada kolom indikator)
                      — harus punya DatetimeIndex ber-timezone
                      — kolom: open, high, low, close, tick_volume
        df_h1       : DataFrame H1 MENTAH (format sama dengan M5)
        warm_up     : Candle awal yang dilewati (indikator belum fully converged)
        max_candles : Batas forward scan per trade untuk SL/TP simulation
        verbose     : Jika True, print progress tiap 500 candle

    Return:
        (trades_df, summary)
        trades_df  : pd.DataFrame — satu baris per trade (skema lengkap di bawah)
        summary    : dict — ringkasan agregat (win_rate, drawdown, dll)

    SKEMA KOLOM trades_df:
        entry_time        : Timestamp candle tempat sinyal muncul
        exit_time         : Timestamp candle ketika SL/TP hit (None jika NO_HIT)
        direction         : "BUY" / "SELL"
        entry_price       : Harga entry dari calculate_sl_tp() (= close, entry_type=CLOSE)
        sl                : Level Stop Loss
        tp                : Level Take Profit
        sl_method         : "SWING" / "ATR" — metode SL yang dipakai
        atr_value         : Nilai ATR saat entry
        outcome           : "TP_HIT" / "SL_HIT" / "NO_HIT"
        candles_held      : Berapa candle trade terbuka
        rrr_planned       : RRR yang direncanakan saat entry
        rrr_realized      : RRR aktual berdasarkan outcome
        jarak_sl          : Jarak SL dalam poin dollar
        jarak_tp          : Jarak TP dalam poin dollar
        pnl_points        : Profit/Loss dalam poin dollar
        ambiguous_candle  : True jika SL dan TP kena candle yang sama
        trend_m5          : Label trend M5 saat entry
        trend_h1          : Label trend H1 saat entry
        rsi_at_entry      : Nilai RSI saat entry
        ema_gap_pct       : Jarak EMA 9 vs EMA 21 (%) saat entry
    """
    print("=" * 60)
    print("  BACKTEST ENGINE — XAUUSD M5 Rule-Based")
    print("=" * 60)
    print(f"  M5  candle : {len(df_m5):,} ({df_m5.index[0]} -> {df_m5.index[-1]})")
    print(f"  H1  candle : {len(df_h1):,} ({df_h1.index[0]} -> {df_h1.index[-1]})")
    print(f"  Warm-up    : {warm_up} candle (indikator warm-up window)")
    print(f"  Max forward: {max_candles} candle per trade "
          f"(~{max_candles * 5 / 60:.0f} jam trading time)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 1: Hitung semua indikator SATU KALI untuk seluruh histori
    # ─────────────────────────────────────────────────────────────────────────
    # EMA/RSI/ATR semuanya causal — nilai di baris i sama persis antara
    # run(df_full) dan run(df[:i+1]). Dibuktikan oleh validate_no_lookahead().
    print("[1/5] Menghitung indikator M5 (satu kali, O(n))...")
    df_m5_ind = run_all_indicators(df_m5.copy())

    print("[2/5] Menghitung indikator H1 (satu kali, O(n))...")
    df_h1_ind = run_all_indicators(df_h1.copy())

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2: Merge H1 ke M5 (anti-lookahead via direction="backward")
    # ─────────────────────────────────────────────────────────────────────────
    print("[3/5] Merging H1 bias ke M5 (merge_asof backward)...")
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)

    nan_h1_count = int(df_merged["trend_h1"].isna().sum())
    if nan_h1_count > 0:
        print(f"   WARNING: {nan_h1_count} baris M5 tidak punya H1 reference "
              f"(sebelum H1 data dimulai) -- akan diskip di loop")

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 3: Validasi zero look-ahead (bukti empiris, bukan sekadar klaim)
    # ─────────────────────────────────────────────────────────────────────────
    print("[4/5] Validasi zero look-ahead (5 titik acak, toleransi 1e-6)...")
    val = validate_no_lookahead(df_m5, n_samples=5)
    print(f"   {val['message']}")

    if not val["passed"]:
        raise RuntimeError(
            "validate_no_lookahead GAGAL -- ada look-ahead bias terdeteksi!\n"
            "Backtest dihentikan. Periksa detail di val['details']."
        )
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 4: Loop utama — evaluasi sinyal per candle
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[5/5] Scanning {len(df_merged) - warm_up:,} candle "
          f"(index {warm_up} s/d {len(df_merged)-1})...")

    trades             = []
    n_evaluated        = 0   # candle yang masuk evaluasi rule engine
    n_signals          = 0   # sinyal BUY/SELL yang ditemukan
    in_trade_until_idx = -1  # index terakhir trade yang masih "terbuka"

    n_total = len(df_merged)

    for i in range(warm_up, n_total):

        if verbose and i % 500 == 0 and i > warm_up:
            pct = (i - warm_up) / (n_total - warm_up) * 100
            print(f"   Progress: {i:,}/{n_total:,} ({pct:.0f}%) "
                  f"- {n_signals} sinyal, {len(trades)} trade valid")

        # ── Skip jika masih dalam trade yang belum selesai ───────────────────
        # Satu posisi satu waktu — mencerminkan kondisi trading nyata.
        if i <= in_trade_until_idx:
            continue

        row = df_merged.iloc[i]

        # ── Skip jika H1 data belum tersedia (sebelum H1 histori dimulai) ────
        if pd.isna(row.get("trend_h1")):
            continue

        # ── Bangun signals dict persis seperti format live pipeline ──────────
        # Ini IDENTIK dengan struktur yang dipakai evaluate_entry() di app.py.
        # Nilai-nilai ini causal — df_merged.iloc[i] hanya bergantung pada candle ≤ i.
        signals = {
            "time"        : df_merged.index[i],
            "close"       : float(row["close"]),
            "ema_9"       : float(row["ema_9"]),
            "ema_21"      : float(row["ema_21"]),
            "rsi_14"      : float(row["rsi_14"]),
            "trend"       : str(row["trend"]),
            "ema_gap_pct" : float(row["ema_gap_pct"]),
            "trend_h1"    : str(row["trend_h1"]),
        }

        # ── Skip jika ada NaN di signals (edge candle setelah warm_up) ───────
        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in signals.values()
            if isinstance(v, (int, float))
        )
        if has_nan:
            continue

        n_evaluated += 1

        # ── Evaluasi rule engine — IDENTIK dengan live pipeline ──────────────
        decision = evaluate_entry(signals)

        if decision["keputusan"] not in ("BUY", "SELL"):
            continue

        arah = decision["keputusan"]
        n_signals += 1

        # ── Hitung SL/TP dengan data s/d candle i saja ───────────────────────
        # df_m5_ind.iloc[:i+1] adalah potongan yang tidak mengandung candle masa depan.
        # find_nearest_swing() yang ada di dalam calculate_sl_tp() menggunakan
        # potongan ini — satu-satunya bagian yang per-sinyal, bukan divectorize.
        df_slice = df_m5_ind.iloc[: i + 1]

        risk = calculate_sl_tp(
            df    = df_slice,
            entry = signals["close"],
            arah  = arah,
            # tick_info=None di backtest (tidak ada live MT5) → entry_type="CLOSE"
        )

        if not risk["valid"]:
            continue

        sl       = risk["sl"]
        tp       = risk["tp"]
        jarak_sl = risk["jarak_sl"]
        jarak_tp = risk["jarak_tp"]

        # Guard: abaikan trade dengan jarak SL terlalu kecil (anomali data)
        if jarak_sl < MIN_SL_DISTANCE:
            continue

        # ── Simulasi outcome (forward scan OHLC) ─────────────────────────────
        # Gunakan df_m5_ind (DataFrame lengkap) untuk forward scan — kita SEDANG
        # membaca candle masa depan di sini, tapi ini memang tujuannya: simulasi.
        # Yang dilarang adalah memakai data masa depan saat MEMBUAT keputusan entry.
        outcome_info = simulate_trade_outcome(
            df_m5_full  = df_m5_ind,
            entry_idx   = i,
            entry       = risk["entry"],
            sl          = sl,
            tp          = tp,
            max_candles = max_candles,
        )

        outcome      = outcome_info["outcome"]
        candles_held = outcome_info["candles_held"]
        ambiguous    = outcome_info["ambiguous_candle"]

        # ── Hitung P&L dan RRR realized ───────────────────────────────────────
        if outcome == "TP_HIT":
            rrr_realized = risk["rrr"]   # RRR planned = RRR realized saat TP
            pnl_points   = +jarak_tp
        elif outcome == "SL_HIT":
            rrr_realized = -1.0          # loss 1R
            pnl_points   = -jarak_sl
        else:  # NO_HIT
            rrr_realized = 0.0
            pnl_points   = 0.0

        # ── Catat trade ke list hasil ─────────────────────────────────────────
        trades.append({
            "entry_time"       : str(df_merged.index[i]),
            "exit_time"        : outcome_info["exit_time"],
            "direction"        : arah,
            "entry_price"      : risk["entry"],
            "sl"               : sl,
            "tp"               : tp,
            "sl_method"        : risk["sl_method"],
            "atr_value"        : risk["atr_value"],
            "outcome"          : outcome,
            "candles_held"     : candles_held,
            "rrr_planned"      : risk["rrr"],
            "rrr_realized"     : rrr_realized,
            "jarak_sl"         : jarak_sl,
            "jarak_tp"         : jarak_tp,
            "pnl_points"       : pnl_points,
            "ambiguous_candle" : ambiguous,
            # Kolom konteks untuk audit mendalam
            "trend_m5"         : signals["trend"],
            "trend_h1"         : signals["trend_h1"],
            "rsi_at_entry"     : round(signals["rsi_14"], 2),
            "ema_gap_pct"      : round(signals["ema_gap_pct"], 4),
        })

        # ── Update pointer "sedang dalam trade" ──────────────────────────────
        # Untuk TP_HIT dan SL_HIT: skip evaluasi sampai trade selesai.
        # Untuk NO_HIT: tidak ada posisi terbuka formal → sinyal berikutnya boleh masuk.
        if outcome in ("TP_HIT", "SL_HIT"):
            in_trade_until_idx = i + candles_held

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 5: Susun output
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nScan selesai:")
    print(f"   Candle dievaluasi : {n_evaluated:,}")
    print(f"   Sinyal BUY/SELL   : {n_signals:,}")
    print(f"   Trade valid       : {len(trades):,}")

    if not trades:
        return pd.DataFrame(), _empty_summary()

    trades_df = pd.DataFrame(trades)
    summary   = compute_summary(trades_df)

    return trades_df, summary


# =============================================================================
# BAGIAN 5: RINGKASAN AGREGAT
# =============================================================================

def compute_summary(trades_df: pd.DataFrame) -> dict:
    """
    Hitung statistik agregat dari DataFrame hasil trade.

    METRIK YANG DILAPORKAN:

    Metrik utama:
        total_trades     : Total trade seluruhnya (TP + SL + NO_HIT)
        tp_count         : Jumlah trade yang hit TP
        sl_count         : Jumlah trade yang hit SL
        no_hit_count     : Jumlah trade yang tidak resolve dalam max_candles
        closed_count     : tp_count + sl_count (trade yang benar-benar resolved)
        win_rate         : tp_count / closed_count (hanya trade resolved)
        no_hit_rate      : no_hit_count / total_trades — JANGAN DIABAIKAN
        avg_rrr_realized : Rata-rata RRR realized dari trade resolved

    Metrik risiko:
        max_consec_loss  : Streak kalah berturut-turut terpanjang (SL_HIT only)
        max_drawdown_pts : Drawdown terbesar dari equity curve (dalam poin dollar)
        total_pnl_points : Total P&L dalam poin dollar

    Metrik kualitas data:
        ambiguous_rate   : Proporsi trade yang SL/TP ambigu di candle yang sama
                           Jika > 10%, pertimbangkan menggunakan data M1 untuk disambiguasi

    TENTANG NO_HIT RATE:
        NO_HIT rate tinggi (>15%) = sinyal penting yang tidak boleh hilang dari laporan:
        - Bisa berarti max_candles terlalu kecil
        - Bisa berarti rule engine menghasilkan sinyal tidak decisive (harga sideways)
        Laporan ini SELALU menyertakan no_hit_rate secara eksplisit.

    Parameter:
        trades_df : pd.DataFrame output dari run_backtest()

    Return:
        dict ringkasan dengan semua metrik di atas
    """
    if trades_df.empty:
        return _empty_summary()

    total  = len(trades_df)
    tp_n   = int((trades_df["outcome"] == "TP_HIT").sum())
    sl_n   = int((trades_df["outcome"] == "SL_HIT").sum())
    no_n   = int((trades_df["outcome"] == "NO_HIT").sum())
    closed = tp_n + sl_n
    ambig  = int(trades_df["ambiguous_candle"].sum())

    win_rate    = round(tp_n / closed, 4) if closed > 0 else None
    no_hit_rate = round(no_n  / total, 4) if total  > 0 else 0.0
    ambig_rate  = round(ambig / total, 4) if total  > 0 else 0.0

    # Average RRR realized — hanya dari trade resolved (bukan NO_HIT)
    closed_df  = trades_df[trades_df["outcome"].isin(["TP_HIT", "SL_HIT"])]
    avg_rrr    = round(float(closed_df["rrr_realized"].mean()), 4) if not closed_df.empty else None

    # Max consecutive losses
    max_consec = _max_consecutive_losses(trades_df["outcome"].tolist())

    # Equity curve dan max drawdown
    equity      = trades_df["pnl_points"].cumsum()
    running_max = equity.cummax()
    drawdown    = equity - running_max
    max_dd      = round(float(drawdown.min()), 2) if not drawdown.empty else 0.0

    # Average candles held (hanya closed trade)
    avg_candles = (
        round(float(closed_df["candles_held"].mean()), 1)
        if not closed_df.empty else None
    )

    return {
        # ── Metrik utama ───────────────────────────────────────────────
        "total_trades"       : total,
        "tp_count"           : tp_n,
        "sl_count"           : sl_n,
        "no_hit_count"       : no_n,
        "closed_count"       : closed,
        "win_rate"           : win_rate,         # TP/(TP+SL) — tidak termasuk NO_HIT
        "no_hit_rate"        : no_hit_rate,      # NO_HIT/total — selalu laporkan ini
        "avg_rrr_realized"   : avg_rrr,

        # ── Metrik risiko ──────────────────────────────────────────────
        "max_consec_loss"    : max_consec,
        "max_drawdown_pts"   : max_dd,
        "total_pnl_points"   : round(float(trades_df["pnl_points"].sum()), 2),
        "avg_candles_held"   : avg_candles,

        # ── Metrik kualitas data ───────────────────────────────────────
        "ambiguous_count"    : ambig,
        "ambiguous_rate"     : ambig_rate,

        # ── Breakdown ─────────────────────────────────────────────────
        "buy_count"          : int((trades_df["direction"] == "BUY").sum()),
        "sell_count"         : int((trades_df["direction"] == "SELL").sum()),
        "sl_method_breakdown": trades_df["sl_method"].value_counts().to_dict(),
    }


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _max_consecutive_losses(outcomes: list) -> int:
    """
    Hitung streak SL_HIT berturut-turut terpanjang.

    SL_HIT = kalah.
    Setiap trade non-SL_HIT (TP_HIT atau NO_HIT) menghentikan streak kekalahan.
    """
    max_streak  = 0
    curr_streak = 0

    for o in outcomes:
        if o == "SL_HIT":
            curr_streak += 1
            max_streak   = max(max_streak, curr_streak)
        else:
            curr_streak = 0

    return max_streak


def _empty_summary() -> dict:
    """Return ringkasan kosong jika tidak ada trade yang valid."""
    return {
        "total_trades"       : 0,
        "tp_count"           : 0,
        "sl_count"           : 0,
        "no_hit_count"       : 0,
        "closed_count"       : 0,
        "win_rate"           : None,
        "no_hit_rate"        : None,
        "avg_rrr_realized"   : None,
        "max_consec_loss"    : 0,
        "max_drawdown_pts"   : 0.0,
        "total_pnl_points"   : 0.0,
        "avg_candles_held"   : None,
        "ambiguous_count"    : 0,
        "ambiguous_rate"     : None,
        "buy_count"          : 0,
        "sell_count"         : 0,
        "sl_method_breakdown": {},
    }
