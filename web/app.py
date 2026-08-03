"""
web/app.py
==========
Backend Flask untuk antarmuka web mesin analisis trading XAUUSD M5.

CARA KERJA FLASK (penjelasan untuk pemula):
    Flask adalah "server" mini yang berjalan di komputer kamu.
    Ketika kamu buka browser ke http://localhost:5000, Flask menerima
    permintaan tersebut dan mengirim balik halaman HTML.

    Dua URL yang kita punya:
        GET  /         → tampilkan halaman utama (kosong, belum ada hasil)
        POST /analyze  → jalankan analisis, tampilkan hasil

    "GET" = browser minta data (kamu buka alamat di browser)
    "POST" = browser kirim data (kamu klik tombol submit)

ALUR ANALISIS (di fungsi analyze()):
    initialize_mt5()                    ← sambungkan ke MT5 yang sedang berjalan
    get_candles(timeframe="M5")         ← tarik 500 candle XAUUSD M5
    get_candles(timeframe="H1", n=100)  ← tarik 100 candle XAUUSD H1 (bias makro)
    run_all_indicators(df_m5)           ← hitung EMA 9/21, RSI, ATR, trend M5
    run_all_indicators(df_h1)           ← hitung EMA 9/21, trend H1
    get_latest_signals(df_m5)           ← ambil nilai terbaru sebagai dict
    signals["trend_h1"] = ...           ← inject bias H1 ke signals M5
    evaluate_entry(signals)             ← rule engine: BUY / SELL / WAIT + breakdown
    calculate_sl_tp()                   ← hitung SL, TP, RRR (hanya jika BUY/SELL)
    shutdown_mt5()                      ← putuskan koneksi dengan bersih
"""

import sys
import os

# ── Tambahkan root folder ke Python path ───────────────────────────────────
# Ini diperlukan agar Python bisa menemukan folder 'engine/'
# ketika kita jalankan dari dalam folder 'web/'
#
# Penjelasan: sys.path adalah daftar folder tempat Python mencari modul.
# Kita tambahkan folder induk (machine-to-win/) ke daftar tersebut.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import MetaTrader5 as mt5
from flask import Flask, render_template, request, jsonify

from engine.data_fetcher  import initialize_mt5, get_candles, shutdown_mt5
from engine.indicators    import run_all_indicators, get_latest_signals, detect_bias_h1
from engine.rule_engine   import evaluate_entry
from engine.risk_manager  import calculate_sl_tp, find_nearest_swing
from engine.history_logger import init_db, log_analysis, get_history, update_outcome


# =============================================================================
# SETUP FLASK & DATABASE
# =============================================================================
# __name__ adalah variabel Python yang berisi nama file ini.
# Flask memakainya untuk menemukan folder templates/ dan static/ secara otomatis.
# template_folder dan static_folder harus relatif terhadap file ini.

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Inisialisasi tabel SQLite history.db saat server dinyalakan
init_db()



# =============================================================================
# ROUTE 1: Halaman Utama
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Tampilkan halaman utama tanpa hasil analisis.

    Penjelasan decorator @app.route:
        Ini memberitahu Flask: "jika ada request ke URL '/',
        jalankan fungsi index() ini".
        methods=["GET"] artinya hanya terima request GET
        (yaitu saat kamu buka alamat di browser).
    """
    return render_template("index.html", result=None, error=None)


# =============================================================================
# ROUTE 2: Endpoint Analisis
# =============================================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Jalankan seluruh pipeline analisis dan tampilkan hasilnya.

    Dipanggil ketika tombol "Analisis Sekarang" diklik.
    Form HTML mengirim POST request ke URL ini.

    Return:
        Render index.html dengan:
        - result : dict berisi semua data hasil analisis (atau None jika error)
        - error  : string pesan error (atau None jika sukses)
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Seluruh alur dibungkus try-except agar error apapun ditampilkan
    # di UI dengan pesan yang ramah — bukan crash dengan traceback panjang.
    # ─────────────────────────────────────────────────────────────────────────
    mt5_connected = False  # flag untuk pastikan shutdown dipanggil jika connect berhasil

    try:
        # ── LANGKAH 1: Sambungkan ke MT5 ─────────────────────────────────────
        print("→ Menghubungkan ke MT5...")
        if not initialize_mt5():
            return render_template(
                "index.html",
                result=None,
                error=(
                    "❌ Tidak bisa terhubung ke MetaTrader 5. "
                    "Pastikan MT5 sudah terbuka dan kamu sudah login ke akun broker."
                ),
            )
        mt5_connected = True

        # ── LANGKAH 2: Tarik data candle M5 ─────────────────────────────────
        print("→ Menarik data candle M5...")
        df_m5 = get_candles(timeframe_str="M5")  # default symbol dan count dari .env
        if df_m5 is None or df_m5.empty:
            return render_template(
                "index.html",
                result=None,
                error=(
                    "❌ Gagal menarik data candle XAUUSD M5. "
                    "Pastikan simbol XAUUSD tersedia di Market Watch MT5."
                ),
            )

        # ── LANGKAH 2b: Tarik data candle H1 (untuk bias makro) ─────────────
        # 100 candle H1 ≈ 4 hari data — cukup untuk EMA 9/21 warm-up
        print("→ Menarik data candle H1 (bias makro)...")
        df_h1 = get_candles(timeframe_str="H1", count=100)
        if df_h1 is None or df_h1.empty:
            return render_template(
                "index.html",
                result=None,
                error=(
                    "❌ Gagal menarik data candle XAUUSD H1. "
                    "Pastikan simbol XAUUSD tersedia di Market Watch MT5."
                ),
            )

        # ── LANGKAH 3: Hitung semua indikator (M5) ───────────────────────────
        print("→ Menghitung indikator M5...")
        df_m5 = run_all_indicators(df_m5)

        # ── LANGKAH 3b: Hitung indikator H1 (untuk bias) ─────────────────────
        print("→ Menghitung indikator H1...")
        df_h1 = run_all_indicators(df_h1)

        # ── LANGKAH 4: Ambil nilai terbaru (dari M5) ─────────────────────────
        print("→ Mengambil sinyal terbaru...")
        signals = get_latest_signals(df_m5)

        # ── LANGKAH 4b: Inject bias H1 ke signals ────────────────────────────
        # signals["trend_h1"] adalah sumber independen dari timeframe berbeda.
        # Ini memastikan rule engine punya dua input yang benar-benar terpisah:
        #   - signals["trend"]    = trend M5 (trigger timing, gap-gated via detect_trend)
        #   - signals["trend_h1"] = bias H1 (arah makro, position-based via detect_bias_h1)
        #
        # detect_bias_h1() TIDAK mewajibkan gap minimum — berbeda dari detect_trend().
        # Ini sengaja: H1 hanya perlu menjawab "sisi mana?", bukan "sudah kuat trending?".
        df_h1 = detect_bias_h1(df_h1)
        signals["trend_h1"] = df_h1["bias_h1"].iloc[-1]
        print(f"   H1 bias: {signals['trend_h1']} | M5 trigger: {signals['trend']}")

        # ── Fase 4.3: Feed swing data ke signals sebelum evaluate_entry ──────
        # Diperlukan agar calculate_setup_quality() bisa menghitung swing_distance.
        # Panggil find_nearest_swing untuk kedua arah agar scoring bekerja apapun
        # keputusan yang nanti dihasilkan evaluate_entry().
        try:
            signals["swing_low"]  = find_nearest_swing(df_m5, arah="BUY")
            signals["swing_high"] = find_nearest_swing(df_m5, arah="SELL")
            print(f"   Swing low: {signals['swing_low']} | Swing high: {signals['swing_high']}")
        except Exception as sw_err:
            signals["swing_low"]  = None
            signals["swing_high"] = None
            print(f"   ⚠️ Swing detection gagal ({sw_err}) -- swing_distance akan 0")

        # ── LANGKAH 5: Evaluasi kondisi entry (rule engine) ──────────────────
        print("→ Mengevaluasi kondisi entry...")
        decision = evaluate_entry(signals)

        # ── LANGKAH 6: Hitung SL/TP (BUY, SELL, atau Acuan Proyeksi jika WAIT) ───
        print(f"→ Menghitung acuan SL/TP & Manajemen Risiko ({decision['keputusan']})...")
        symbol    = os.getenv("MT5_SYMBOL", "XAUUSD")
        tick_info = None
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None and tick.ask > 0 and tick.bid > 0:
                tick_info = {"ask": tick.ask, "bid": tick.bid}
                print(f"   Tick real-time: ask={tick.ask:.2f} bid={tick.bid:.2f} "
                      f"spread={tick.ask - tick.bid:.5f}")
            else:
                print("   ⚠️ Tick tidak tersedia — fallback ke close price")
        except Exception as e:
            print(f"   ⚠️ Gagal ambil tick ({e}) — fallback ke close price")

        # Tentukan arah kalkulasi risiko: jika BUY/SELL gunakan keputusan, jika WAIT gunakan bias/trend
        risk_direction = decision["keputusan"]
        if risk_direction not in ("BUY", "SELL"):
            h1_bias = signals.get("trend_h1", "SIDEWAYS")
            m5_trend = signals.get("trend", "SIDEWAYS")
            if h1_bias == "DOWNTREND" or m5_trend == "DOWNTREND":
                risk_direction = "SELL"
            else:
                risk_direction = "BUY"

        risk = calculate_sl_tp(
            df        = df_m5,
            entry     = signals["close"],
            arah      = risk_direction,
            profile   = "scalp_m5",
            tick_info = tick_info,
        )
        if risk and isinstance(risk, dict):
            risk["is_estimate"] = (decision["keputusan"] == "WAIT")


        # ── LANGKAH 7: Susun semua data untuk dikirim ke template HTML ────────
        # Template HTML tidak bisa langsung akses dict nested yang kompleks,
        # jadi kita susun ulang menjadi struktur yang lebih flat dan mudah dipakai.
        result = _build_result(signals, decision, risk)

        # ── LANGKAH 7b: Catat histori ke SQLite database ─────────────────────
        try:
            rowid = log_analysis(decision, signals, risk)
            result["history_id"] = rowid
            print(f"💾 Histori analisis tersimpan ke database (ID: {rowid})")
        except Exception as log_err:
            print(f"⚠️ Gagal menyimpan histori ke SQLite: {log_err}")

        print(f"✅ Analisis selesai: {decision['keputusan']}")
        return render_template("index.html", result=result, error=None)

    except Exception as e:
        # Tangkap error yang tidak terduga dan tampilkan pesan yang informatif
        print(f"❌ Error tidak terduga: {e}")
        return render_template(
            "index.html",
            result=None,
            error=f"❌ Terjadi error tidak terduga: {str(e)}",
        )

    finally:
        # ── LANGKAH 8: Selalu putuskan koneksi MT5 ───────────────────────────
        # 'finally' selalu dijalankan, baik ada error maupun tidak.
        # Ini penting agar koneksi MT5 tidak dibiarkan menggantung.
        if mt5_connected:
            shutdown_mt5()
            print("→ Koneksi MT5 ditutup.")


# =============================================================================
# ROUTE 3: Endpoint Histori Analisis & API Outcome
# =============================================================================

@app.route("/history", methods=["GET"])
def history_page():
    """
    Tampilkan halaman histori analisis trading dari SQLite.
    """
    records = get_history(limit=100)
    return render_template("history.html", history=records)


@app.route("/api/history/outcome", methods=["POST"])
def update_history_outcome():
    """
    API endpoint untuk memperbarui status outcome trade (WIN / LOSS / EXPIRED / MANUAL_CLOSE).
    """
    data = request.get_json(silent=True) or request.form
    record_id = data.get("record_id")
    outcome = data.get("outcome")
    notes = data.get("notes", "")

    if not record_id or not outcome:
        return jsonify({"success": False, "error": "record_id dan outcome wajib diisi"}), 400

    try:
        success = update_outcome(int(record_id), str(outcome), str(notes))
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# HELPER: Susun Dict Hasil Analisis untuk Template
# =============================================================================

def _build_result(signals: dict, decision: dict, risk: dict | None) -> dict:
    """
    Susun semua data ke dalam satu dict yang siap dipakai di template HTML.

    Struktur output:
        result["keputusan"]           : "BUY" / "SELL" / "WAIT"
        result["setup_quality"]       : "STRONG" / "MODERATE" / "WEAK"
        result["setup_quality_score"] : int (0-8)
        result["quality_breakdown"]   : dict breakdown 4 komponen
        result["signals"]             : dict nilai indikator terbaru
        result["kondisi"]             : list kondisi dengan status terpenuhi/tidak
        result["alasan_entry"]        : list string alasan entry
        result["alasan_wait"]         : list string alasan wait
        result["risk"]                : dict SL/TP/RRR (None jika WAIT)
        result["waktu"]               : string waktu evaluasi
    """
    keputusan = decision["keputusan"]

    # ── Susun breakdown kondisi sebagai list yang mudah di-loop di HTML ───────
    kondisi_list = []

    # Kondisi 1: Bias H1 (sumber independen — timeframe lebih besar)
    c_h1 = decision["kondisi_detail"]["bias_h1"]
    kondisi_list.append({
        "nama"       : "Bias Arah H1",
        "terpenuhi"  : c_h1["terpenuhi"],
        "keterangan" : c_h1["keterangan"],
        "arah"       : c_h1["arah"],
        "tipe"       : "entry",  # kondisi entry (bukan filter)
    })

    # Kondisi 2: EMA Trigger M5 (timing entry dari timeframe trading)
    c_m5 = decision["kondisi_detail"]["ema_trigger_m5"]
    kondisi_list.append({
        "nama"       : "EMA Trigger M5",
        "terpenuhi"  : c_m5["terpenuhi"],
        "keterangan" : c_m5["keterangan"],
        "arah"       : c_m5["arah"],
        "tipe"       : "entry",  # kondisi entry (bukan filter)
    })

    # Filter RSI
    c_rsi = decision["kondisi_detail"]["rsi_filter"]
    kondisi_list.append({
        "nama"       : "Filter RSI",
        "terpenuhi"  : not c_rsi["memblokir"],  # "terpenuhi" = RSI TIDAK memblokir
        "keterangan" : c_rsi["keterangan"],
        "arah"       : "NETRAL",
        "tipe"       : "filter",  # ini filter (veto), bukan kondisi entry
    })

    # ── Format waktu evaluasi ─────────────────────────────────────────────────
    waktu_raw = str(decision["waktu_evaluasi"])
    try:
        waktu = waktu_raw.replace("+00:00", " UTC").replace("T", " ")[:20] + " UTC"
    except Exception:
        waktu = waktu_raw

    return {
        "keputusan"           : keputusan,
        "arah"                : decision.get("arah"),            # "LONG" / "SHORT" / None
        "setup_quality"       : decision.get("setup_quality", "WEAK"),
        "setup_quality_score" : decision.get("setup_quality_score", 0),
        "setup_quality_max"   : decision.get("setup_quality_max", 8),
        "quality_breakdown"   : decision.get("quality_breakdown", {}),
        "signals"             : signals,
        "kondisi"             : kondisi_list,
        "alasan_entry"        : decision.get("alasan_entry", []),
        "alasan_wait"         : decision.get("alasan_wait",  []),
        "konfirmasi"          : {
            "terpenuhi"  : decision["konfirmasi_terpenuhi"],
            "dibutuhkan" : decision["konfirmasi_dibutuhkan"],
        },
        "risk"                : risk,    # None jika WAIT
        "waktu"               : waktu,
        "context_warnings"    : decision.get("context_warnings", []),
    }



# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Mesin Analisis XAUUSD M5 — Web Interface")
    print("=" * 60)
    print("  Buka browser → http://localhost:5000")
    print("  Tekan Ctrl+C untuk menghentikan server")
    print("=" * 60)

    # debug=True → server otomatis restart jika kode berubah
    # Berguna saat development. Matikan (False) jika sudah production.
    app.run(debug=True, host="127.0.0.1", port=5000)
