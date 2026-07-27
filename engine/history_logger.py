"""
engine/history_logger.py
========================
Modul pengelolaan histori analisis trading menggunakan SQLite (data/history.db).

FUNGSI UTAMA:
    - init_db()         : Inisialisasi tabel SQLite jika belum ada
    - log_analysis()    : Menyimpan setiap hasil evaluate_entry() dari web/app.py
    - get_history()     : Mengambil daftar histori analisis terbaru untuk UI
    - update_outcome()  : Mengubah status outcome trade (PENDING, WIN, LOSS, EXPIRED, dll.)

REPRODUCIBILITY & TRANSPARENCY:
    Semua data disimpan secara terstruktur (termasuk JSON breakdown sinyal & quality scoring)
    sehingga setiap keputusan historis dapat diaudit kembali kapan saja.
"""

import os
import sqlite3
import json
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "history.db"
)


def _get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Membuat dan mengembalikan koneksi SQLite DB.
    Memastikan folder data/ ada sebelum membuka DB.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Mengembalikan hasil query sebagai dict-like Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Inisialisasi tabel SQLite `analysis_history` jika belum tersedia.
    """
    conn = _get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    close_price REAL NOT NULL,
                    decision TEXT NOT NULL,
                    arah TEXT,
                    setup_quality TEXT,
                    quality_score INTEGER,
                    quality_breakdown_json TEXT,
                    signals_json TEXT NOT NULL,
                    sl_price REAL,
                    tp_price REAL,
                    rrr REAL,
                    context_warnings_json TEXT,
                    outcome TEXT DEFAULT 'PENDING',
                    outcome_notes TEXT,
                    created_at TEXT NOT NULL
                )
            """)
    finally:
        conn.close()


def log_analysis(
    decision_dict: dict,
    signals: dict,
    risk_dict: dict | None = None,
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Menyimpan hasil analisis lengkap dari evaluate_entry() dan calculate_sl_tp().

    Parameter:
        decision_dict : output dari evaluate_entry()
        signals       : dict indikator dari get_latest_signals()
        risk_dict     : output dari calculate_sl_tp() (atau None jika WAIT)
        symbol        : nama instrumen ("XAUUSD")
        timeframe     : timeframe ("M5")
        db_path       : path ke file history.db

    Return:
        int : ID baris (primary key) yang baru dimasukkan.
    """
    init_db(db_path)

    # Persiapkan data serialisasi
    timestamp = str(decision_dict.get("waktu_evaluasi", datetime.now(timezone.utc).isoformat()))
    close_price = float(decision_dict.get("close", signals.get("close", 0.0)))
    decision = str(decision_dict.get("keputusan", "WAIT"))
    arah = decision_dict.get("arah")

    setup_quality = decision_dict.get("setup_quality", "WEAK")
    quality_score = decision_dict.get("setup_quality_score", 0)
    quality_breakdown_json = json.dumps(decision_dict.get("quality_breakdown", {}), ensure_ascii=False)

    # Bersihkan signals agar serializable
    clean_signals = {}
    for k, v in signals.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean_signals[k] = v
        else:
            clean_signals[k] = str(v)
    signals_json = json.dumps(clean_signals, ensure_ascii=False)

    sl_price = None
    tp_price = None
    rrr = None
    if risk_dict and isinstance(risk_dict, dict):
        sl_price = risk_dict.get("sl")
        tp_price = risk_dict.get("tp")
        rrr = risk_dict.get("rrr")

    context_warnings_json = json.dumps(decision_dict.get("context_warnings", []), ensure_ascii=False)
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history (
                    timestamp, symbol, timeframe, close_price, decision, arah,
                    setup_quality, quality_score, quality_breakdown_json, signals_json,
                    sl_price, tp_price, rrr, context_warnings_json, outcome, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (
                    timestamp, symbol, timeframe, close_price, decision, arah,
                    setup_quality, quality_score, quality_breakdown_json, signals_json,
                    sl_price, tp_price, rrr, context_warnings_json, created_at
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_history(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Mengambil daftar histori analisis terbaru dari database SQLite.

    Return:
        list of dict berisi rekam data histori analisis.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, timestamp, symbol, timeframe, close_price, decision, arah,
                   setup_quality, quality_score, quality_breakdown_json, signals_json,
                   sl_price, tp_price, rrr, context_warnings_json, outcome, outcome_notes, created_at
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            # Parse JSON fields
            try:
                row_dict["quality_breakdown"] = json.loads(row_dict.get("quality_breakdown_json") or "{}")
            except Exception:
                row_dict["quality_breakdown"] = {}

            try:
                row_dict["signals"] = json.loads(row_dict.get("signals_json") or "{}")
            except Exception:
                row_dict["signals"] = {}

            try:
                row_dict["context_warnings"] = json.loads(row_dict.get("context_warnings_json") or "[]")
            except Exception:
                row_dict["context_warnings"] = []

            result.append(row_dict)
        return result
    finally:
        conn.close()


def update_outcome(
    record_id: int,
    outcome: str,
    notes: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """
    Memperbarui status outcome suatu entri histori (misal: "WIN", "LOSS", "EXPIRED", "MANUAL_CLOSE").

    Return:
        bool : True jika berhasil diperbarui.
    """
    init_db(db_path)
    conn = _get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                UPDATE analysis_history
                SET outcome = ?, outcome_notes = ?
                WHERE id = ?
                """,
                (outcome, notes, record_id),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()
