"""
engine/strategy_router.py
==========================
Modul Strategy Router — titik orkestrasi Fase 14.

TUJUAN:
    Memetakan regime pasar M15 (output dari engine/regime_detector.py) ke
    strategi trading yang sesuai dan menegakkan hierarki pemilihan strategi
    secara eksplisit. Modul ini adalah satu-satunya jembatan antara lapisan
    deteksi regime (Fase 13) dan lapisan strategi (Fase 15–17).

    Modul ini BUKAN implementasi strategi — ia hanya menentukan strategi mana
    yang aktif dan mengapa (audit trail). Logika Trend Following, Range Reversal,
    dan Breakout Retest adalah pekerjaan Fase 15–17.

KARAKTER MODUL:
    - ORKESTRASI MURNI: tidak ada logika BUY/SELL/WAIT, tidak ada kalkulasi
      teknikal. Hanya routing dan grace window scan.
    - STATELESS: tidak ada state yang di-pass antar candle. Setiap pemanggilan
      get_active_strategy(df_m15, idx) yang sama menghasilkan output identik
      terlepas dari urutan pemanggilan sebelumnya.
    - TIDAK MEMODIFIKASI: tidak mengubah engine/regime_detector.py,
      engine/indicators.py, web/app.py, atau modul lain yang sudah ada.
      Modul ini murni read-only consumer.

CAUSALITY (ZERO LOOK-AHEAD):
    get_active_strategy(df_m15, idx) hanya membaca df_m15.iloc[:idx+1].
    Di dalam grace window scan, hanya index <= idx-1 yang dibaca (selalu
    lebih kecil dari idx). Tidak ada akses ke candle setelah idx dalam
    bentuk apapun.

CATATAN GRACE WINDOW:
    Grace window KHUSUS untuk transisi BREAKOUT_TRANSITION → CHOP.
    Dari diagnostik empiris Fase 13, BREAKOUT_TRANSITION rata-rata hanya
    berlangsung 1 candle M15 sebelum berubah ke CHOP (karena syarat
    TRENDING belum terbentuk). Tanpa grace window, strategi Breakout Retest
    (Fase 16) tidak punya waktu menunggu retest di M5. Grace window TIDAK
    berlaku untuk transisi ke RANGING — RANGING setelah breakout mengindikasikan
    breakout gagal, biarkan router memilih RANGE_REVERSAL secara normal.

CATATAN PARAMETER (BELUM DIKALIBRASI):
    REGIME_BREAKOUT_GRACE_CANDLES = 4

    Nilai ini adalah STARTING POINT struktural (sekitar 1 jam pada M15),
    bukan hasil kalibrasi backtest. Dipilih agar kira-kira setara dengan
    retest_lookback_candles=15 pada M5 (~75 menit) dari sistem retest M5
    versi lama (Fase 10, CLOSED). Akan dikalibrasi via validasi empiris
    setelah Fase 16 selesai dan ada data backtest yang cukup.
    JANGAN ubah nilai ini tanpa approval eksplisit setelah validasi empiris.
"""

import pandas as pd

from engine.regime_detector import detect_market_regime


# =============================================================================
# KONSTANTA DAN MAPPING STRATEGI
# =============================================================================

STRATEGY_MAP: dict = {
    "TRENDING"           : "TREND_FOLLOWING",
    "RANGING"            : "RANGE_REVERSAL",
    "BREAKOUT_TRANSITION": "BREAKOUT_RETEST",
    "CHOP"               : None,
}
# Mapping tetap dari regime M15 ke nama strategi.
# CHOP → None berarti tidak ada strategi aktif (NO TRADE).
# Konstanta ini diekspos agar bisa diaudit dan diuji secara independen.

REGIME_BREAKOUT_GRACE_CANDLES: int = 4
# Jumlah candle M15 ke belakang yang dicek saat regime saat ini CHOP,
# untuk mendeteksi apakah ada BREAKOUT_TRANSITION yang baru saja terjadi.
# Nilai 4 candle M15 ≈ 1 jam — BELUM DIKALIBRASI, ini starting point.
# Lihat docstring modul untuk justifikasi dan rencana kalibrasi.


# =============================================================================
# FUNGSI 1: PURE MAPPING (tanpa grace window)
# =============================================================================

def route_strategy(regime: str) -> "str | None":
    """
    Peta regime M15 ke nama strategi menggunakan STRATEGY_MAP.

    Fungsi ini adalah lapisan tipis di atas STRATEGY_MAP — dipisahkan
    agar mapping inti bisa diuji dan diaudit secara independen dari
    logika grace window yang lebih kompleks.
    Pola ini mengikuti pemisahan _check_bias_h1() dari evaluate_entry()
    di engine/rule_engine.py.

    Parameter:
        regime : str — salah satu dari "TRENDING" / "RANGING" /
                 "BREAKOUT_TRANSITION" / "CHOP". Regime lain di luar
                 empat ini menghasilkan None (graceful, tidak crash).

    Return:
        str | None — nama strategi, atau None untuk CHOP / regime tidak dikenal.
    """
    return STRATEGY_MAP.get(regime)


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _build_result(
    strategy    : "str | None",
    arah        : "str | None",
    source      : str,
    regime      : dict,
    grace_regime: "dict | None",
    keterangan  : str,
) -> dict:
    """
    Susun dict output standar get_active_strategy().

    Helper internal agar tidak ada duplikasi struktur dict di antara
    dua jalur kode (DIRECT dan GRACE_WINDOW / NONE).
    """
    return {
        "strategy"    : strategy,
        "arah"        : arah,
        "source"      : source,
        "regime"      : regime,
        "grace_regime": grace_regime,
        "keterangan"  : keterangan,
    }


# =============================================================================
# FUNGSI 2: ORKESTRASI UTAMA (dengan grace window)
# =============================================================================

def get_active_strategy(
    df_m15       : pd.DataFrame,
    idx          : int = -1,
    grace_candles: int = REGIME_BREAKOUT_GRACE_CANDLES,
) -> dict:
    """
    Tentukan strategi aktif di candle ke-idx dari df_m15.

    Menggabungkan deteksi regime (detect_market_regime) dengan grace window
    untuk BREAKOUT_TRANSITION → CHOP, menghasilkan satu dict orkestrasi
    yang kaya audit trail.

    ALGORITMA (stateless, urutan wajib):
        1. Normalisasi idx negatif.
        2. Panggil detect_market_regime(df_m15, idx) → regime saat ini.
        3. Peta regime ke strategi via route_strategy().
        4. Jika strategi bukan None → return DIRECT (termasuk kasus di mana
           regime saat ini sudah BREAKOUT_TRANSITION sendiri).
        5. Jika strategi None (artinya CHOP) → scan grace window:
           - Scan mundur dari idx-1 sampai max(idx - grace_candles, 0).
           - Tiap k: panggil detect_market_regime(df_m15, k).
           - Begitu ditemukan k dengan regime BREAKOUT_TRANSITION → return
             GRACE_WINDOW dengan arah dari k tersebut. Berhenti di sini
             (pakai breakout TERBARU, bukan tertua).
           - Kalau tidak ditemukan dalam window → return NONE (CHOP murni).

    BATASAN GRACE WINDOW:
        Grace window HANYA aktif ketika regime saat ini adalah CHOP.
        Kalau regime saat ini RANGING, TRENDING, atau BREAKOUT_TRANSITION,
        fungsi ini mengembalikan DIRECT tanpa menyentuh grace window.
        Ini memastikan grace window tidak "membajak" routing normal.

    CAUSALITY:
        detect_market_regime(df_m15, idx) hanya membaca [:idx+1].
        Scan grace window menggunakan k <= idx-1, selalu di masa lalu.

    STATELESS:
        Tidak ada state eksternal yang di-pass. Setiap pemanggilan untuk
        (df_m15, idx) yang sama menghasilkan output identik.

    Parameter:
        df_m15        : DataFrame M15 yang sudah melewati run_all_indicators().
        idx           : Index candle evaluasi. Default -1 = candle terakhir.
                        Nilai negatif dinormalisasi.
        grace_candles : Jumlah candle ke belakang untuk scan grace window.
                        Default = REGIME_BREAKOUT_GRACE_CANDLES (BELUM DIKALIBRASI).

    Return:
        dict dengan field:
            strategy     : str | None — "TREND_FOLLOWING" / "RANGE_REVERSAL" /
                           "BREAKOUT_RETEST" / None
            arah         : str | None — "BULLISH" / "BEARISH" / None
                           (None untuk RANGE_REVERSAL & NO TRADE)
            source       : str — "DIRECT" / "GRACE_WINDOW" / "NONE"
            regime       : dict — hasil detect_market_regime() penuh di idx saat ini
            grace_regime : dict | None — hasil detect_market_regime() di k
                           (HANYA jika source="GRACE_WINDOW"), None untuk source lain
            keterangan   : str — penjelasan ringkas keputusan routing
    """
    # ── 1. Normalisasi idx negatif ────────────────────────────────────────────
    n = len(df_m15)
    if idx < 0:
        idx = n + idx

    # ── 2. Deteksi regime saat ini ────────────────────────────────────────────
    # CAUSALITY: detect_market_regime hanya membaca df_m15.iloc[:idx+1]
    current = detect_market_regime(df_m15, idx)

    # ── 3. Pure mapping regime → strategi ────────────────────────────────────
    strategy = route_strategy(current["regime"])

    # ── 4. Strategi langsung tersedia (DIRECT) ────────────────────────────────
    # Mencakup: TRENDING, RANGING, BREAKOUT_TRANSITION (sudah aktif sekarang)
    if strategy is not None:
        arah = current["arah"]
        keterangan = (
            f"DIRECT: regime={current['regime']} → strategi={strategy}"
            + (f", arah={arah}" if arah else "")
            + f". {current['keterangan']}"
        )
        return _build_result(
            strategy     = strategy,
            arah         = arah,
            source       = "DIRECT",
            regime       = current,
            grace_regime = None,
            keterangan   = keterangan,
        )

    # ── 5. CHOP: cek grace window untuk BREAKOUT_TRANSITION terbaru ───────────
    # Satu-satunya kasus di sini: strategy is None ↔ current["regime"] == "CHOP"
    # CAUSALITY: scan hanya k <= idx-1
    batas_bawah = max(idx - grace_candles, 0)

    for k in range(idx - 1, batas_bawah - 1, -1):
        # CAUSALITY: k < idx selalu terpenuhi karena range mulai dari idx-1
        regime_k = detect_market_regime(df_m15, k)

        if regime_k["regime"] == "BREAKOUT_TRANSITION":
            # Breakout terbaru ditemukan — pakai ini (berhenti scan)
            arah_k = regime_k["arah"]
            keterangan = (
                f"GRACE_WINDOW: regime saat ini CHOP di idx={idx}, "
                f"ditemukan BREAKOUT_TRANSITION {arah_k} di idx={k} "
                f"(dalam window {grace_candles} candle) "
                f"→ strategi=BREAKOUT_RETEST"
            )
            return _build_result(
                strategy     = "BREAKOUT_RETEST",
                arah         = arah_k,
                source       = "GRACE_WINDOW",
                regime       = current,
                grace_regime = regime_k,
                keterangan   = keterangan,
            )

    # ── 6. Tidak ada breakout dalam window — CHOP murni (NO TRADE) ───────────
    keterangan = (
        f"NONE: regime={current['regime']} di idx={idx}, "
        f"tidak ada BREAKOUT_TRANSITION dalam {grace_candles} candle terakhir "
        f"→ NO TRADE"
    )
    return _build_result(
        strategy     = None,
        arah         = None,
        source       = "NONE",
        regime       = current,
        grace_regime = None,
        keterangan   = keterangan,
    )


# =============================================================================
# FUNGSI 3 (OPSIONAL): EFISIENSI BACKTEST — precomputed regimes
# =============================================================================

def get_active_strategy_from_precomputed_regimes(
    regime_series: "list[dict]",
    idx          : int,
    grace_candles: int = REGIME_BREAKOUT_GRACE_CANDLES,
) -> dict:
    """
    Sama seperti get_active_strategy(), tapi membaca dari list regime yang
    sudah dihitung sebelumnya oleh caller.

    APA INI:
        Varian efisien dari get_active_strategy() untuk keperluan backtest.
        Saat backtest, caller menghitung detect_market_regime() SEKALI untuk
        semua index 0..N-1, simpan hasilnya sebagai list, lalu panggil
        fungsi ini per candle tanpa overhead recompute detect_market_regime().
        Pola identik dengan get_h1_context_from_precomputed() di
        engine/market_context.py (Fase 12).

    CAUSALITY:
        Kausalitas dijamin oleh caller yang harus memastikan regime_series
        dihitung secara causal sebelum dipassing ke sini (setiap elemen
        regime_series[k] hanya boleh menggunakan data df_m15.iloc[:k+1]).
        Fungsi ini sendiri hanya membaca regime_series[k] untuk k <= idx
        dan tidak menyentuh indeks lebih besar dari idx.

    KENAPA DIPISAH:
        detect_market_regime() menghitung ulang semua sub-fungsi untuk seluruh
        window lookback setiap kali dipanggil — overhead signifikan saat
        backtest ribuan candle. Dengan varian ini, overhead jadi O(1) per
        candle karena regime sudah precomputed.

    STATELESS:
        Tidak ada state yang di-pass. Output identik untuk (regime_series, idx)
        yang sama terlepas dari urutan pemanggilan.

    Parameter:
        regime_series : list[dict] — list hasil detect_market_regime() untuk
                        setiap index 0..len(regime_series)-1, sudah dihitung
                        sekali di awal oleh caller. Harus punya panjang >= idx+1.
        idx           : Index candle evaluasi (absolut, >= 0). Tidak di-normalisasi
                        karena caller bertanggung jawab menyediakan list yang benar.
        grace_candles : Jumlah candle ke belakang untuk scan grace window.
                        Default = REGIME_BREAKOUT_GRACE_CANDLES.

    Return:
        dict dengan field identik dengan get_active_strategy():
            strategy, arah, source, regime, grace_regime, keterangan.

    Edge case:
        - idx di luar range regime_series → keterangan error, strategy=None,
          source="NONE". Tidak crash.
    """
    n = len(regime_series)

    # ── Validasi idx ──────────────────────────────────────────────────────────
    if idx < 0 or idx >= n:
        return _build_result(
            strategy     = None,
            arah         = None,
            source       = "NONE",
            regime       = {"regime": "CHOP", "arah": None, "zone": None,
                            "detail": {}, "keterangan": "regime_series idx di luar range"},
            grace_regime = None,
            keterangan   = (
                f"ERROR: idx={idx} di luar range [0, {n - 1}] "
                f"untuk regime_series dengan panjang {n}"
            ),
        )

    # ── Regime saat ini dari precomputed list ──────────────────────────────────
    current  = regime_series[idx]
    strategy = route_strategy(current["regime"])

    # ── DIRECT: strategi tersedia langsung ────────────────────────────────────
    if strategy is not None:
        arah = current["arah"]
        keterangan = (
            f"DIRECT (precomputed): regime={current['regime']} → strategi={strategy}"
            + (f", arah={arah}" if arah else "")
        )
        return _build_result(
            strategy     = strategy,
            arah         = arah,
            source       = "DIRECT",
            regime       = current,
            grace_regime = None,
            keterangan   = keterangan,
        )

    # ── CHOP: scan grace window dari precomputed list ─────────────────────────
    batas_bawah = max(idx - grace_candles, 0)

    for k in range(idx - 1, batas_bawah - 1, -1):
        regime_k = regime_series[k]

        if regime_k["regime"] == "BREAKOUT_TRANSITION":
            arah_k = regime_k["arah"]
            keterangan = (
                f"GRACE_WINDOW (precomputed): regime saat ini CHOP di idx={idx}, "
                f"ditemukan BREAKOUT_TRANSITION {arah_k} di idx={k} "
                f"(dalam window {grace_candles} candle) "
                f"→ strategi=BREAKOUT_RETEST"
            )
            return _build_result(
                strategy     = "BREAKOUT_RETEST",
                arah         = arah_k,
                source       = "GRACE_WINDOW",
                regime       = current,
                grace_regime = regime_k,
                keterangan   = keterangan,
            )

    # ── CHOP murni — tidak ada breakout dalam window ──────────────────────────
    keterangan = (
        f"NONE (precomputed): regime={current['regime']} di idx={idx}, "
        f"tidak ada BREAKOUT_TRANSITION dalam {grace_candles} candle terakhir "
        f"→ NO TRADE"
    )
    return _build_result(
        strategy     = None,
        arah         = None,
        source       = "NONE",
        regime       = current,
        grace_regime = None,
        keterangan   = keterangan,
    )
