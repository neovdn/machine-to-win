"""
tests/test_data_fetcher.py
==========================
Unit test untuk modul engine/data_fetcher.py

CARA JALANKAN:
    python -m pytest tests/ -v          # Semua test
    python -m pytest tests/ -v -k "not mt5"  # Hanya test yang tidak butuh MT5

CATATAN:
    Test yang butuh MT5 nyata akan di-skip jika MT5 tidak tersedia.
    Test yang menggunakan mock_df bisa dijalankan kapan saja tanpa MT5.
"""

import sys
import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from datetime import timedelta, timezone, datetime as dt

# Tambahkan root project ke path agar bisa import modul engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_fetcher import validate_data, TIMEFRAME_MAP, get_broker_utc_offset


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: Data dummy untuk testing (tidak butuh koneksi MT5 nyata)
# ─────────────────────────────────────────────────────────────────────────────
# Fixture adalah data atau objek yang disiapkan sebelum test dijalankan
# @pytest.fixture berarti: "ini adalah data yang bisa dipakai oleh test manapun"
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_df():
    """
    Buat DataFrame candle palsu (dummy) untuk testing validasi.
    Formatnya sama persis dengan data nyata dari MT5.
    """
    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=10,
        freq="5min",  # Setiap 5 menit (M5)
        tz="UTC"
    )
    # Setiap baris dibuat konsisten: low = min(open,close)-0.5, high = max(open,close)+0.5
    # Sehingga selalu: high >= open, high >= close, low <= open, low <= close
    opens  = [2050.0, 2051.5, 2049.8, 2051.0, 2050.5,
               2051.0, 2051.5, 2054.5, 2052.0, 2054.0]
    closes = [2051.5, 2049.8, 2051.0, 2050.5, 2051.8,
               2051.0, 2052.0, 2055.0, 2053.5, 2056.0]
    highs  = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows   = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    data = {
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "tick_volume": [120, 95, 145, 88, 167, 112, 134, 98, 201, 156],
    }
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "time"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Validasi data
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_data_valid(mock_df):
    """Data dummy yang valid harus lulus validasi."""
    assert validate_data(mock_df) is True


def test_validate_data_empty():
    """DataFrame kosong harus gagal validasi."""
    empty_df = pd.DataFrame()
    assert validate_data(empty_df) is False


def test_validate_data_none():
    """None harus gagal validasi."""
    assert validate_data(None) is False


def test_validate_data_with_nan(mock_df):
    """DataFrame dengan NaN tetap harus return True (warning, bukan error fatal)."""
    mock_df_with_nan = mock_df.copy()
    mock_df_with_nan.iloc[0, 0] = np.nan  # Masukkan NaN ke baris pertama kolom pertama
    result = validate_data(mock_df_with_nan)
    assert result is True  # validate_data hanya print warning, tidak return False untuk NaN


# ─────────────────────────────────────────────────────────────────────────────
# TEST: TIMEFRAME_MAP
# ─────────────────────────────────────────────────────────────────────────────

def test_timeframe_map_contains_m5():
    """Pastikan M5 ada di peta timeframe."""
    assert "M5" in TIMEFRAME_MAP


def test_timeframe_map_all_timeframes():
    """Pastikan semua timeframe standar tersedia."""
    expected_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    for tf in expected_timeframes:
        assert tf in TIMEFRAME_MAP, f"Timeframe {tf} tidak ditemukan di TIMEFRAME_MAP"


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Data struktur
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_df_has_required_columns(mock_df):
    """DataFrame harus punya semua kolom yang diperlukan."""
    required_columns = ["open", "high", "low", "close", "tick_volume"]
    for col in required_columns:
        assert col in mock_df.columns, f"Kolom '{col}' tidak ditemukan"


def test_mock_df_ohlc_logic(mock_df):
    """Logika OHLC: high selalu >= open, low, close."""
    assert (mock_df["high"] >= mock_df["open"]).all(),  "Ada high < open"
    assert (mock_df["high"] >= mock_df["low"]).all(),   "Ada high < low"
    assert (mock_df["high"] >= mock_df["close"]).all(), "Ada high < close"
    assert (mock_df["low"]  <= mock_df["open"]).all(),  "Ada low > open"
    assert (mock_df["low"]  <= mock_df["close"]).all(), "Ada low > close"


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TEST: get_broker_utc_offset (Item 1 Fase 0 — Fix Timezone)
# ─────────────────────────────────────────────────────────────────────────────
# Test ini menggunakan mock agar tidak membutuhkan koneksi MT5 nyata.
# Mock mensimulasikan tick.time yang 3 jam LEBIH MAJU dari UTC (EEST = UTC+3).
# ─────────────────────────────────────────────────────────────────────────────

def test_get_broker_utc_offset_detects_eest():
    """
    get_broker_utc_offset() harus mendeteksi UTC+3 (EEST) dengan benar.
    Simulasi: tick.time = now_utc_unix + 10800 (3 jam = EEST).
    """
    fixed_utc = dt(2026, 7, 28, 9, 0, 0, tzinfo=timezone.utc)
    tick_unix  = int(fixed_utc.timestamp()) + 3 * 3600  # broker 3 jam lebih maju

    mock_tick = MagicMock()
    mock_tick.time = tick_unix

    with patch("engine.data_fetcher.mt5.symbol_info_tick", return_value=mock_tick), \
         patch("engine.data_fetcher.datetime") as mock_dt:
        # Mock datetime.now(UTC) agar return fixed_utc
        mock_dt.now.return_value = fixed_utc
        mock_dt.side_effect = lambda *args, **kwargs: dt(*args, **kwargs)

        offset = get_broker_utc_offset("XAUUSD")

    assert isinstance(offset, timedelta), "offset harus bertipe timedelta"
    assert offset == timedelta(hours=3), (
        f"Expected timedelta(hours=3), got {offset} "
        f"— get_broker_utc_offset tidak mendeteksi EEST (UTC+3) dengan benar"
    )


def test_get_broker_utc_offset_fallback_when_tick_none():
    """
    get_broker_utc_offset() harus return timedelta(0) jika tick tidak tersedia,
    bukan crash — ini menjaga kompatibilitas jika MT5 tidak aktif.
    """
    with patch("engine.data_fetcher.mt5.symbol_info_tick", return_value=None):
        offset = get_broker_utc_offset("XAUUSD")

    assert offset == timedelta(0), (
        f"Expected timedelta(0) saat tick None, got {offset} "
        f"— fallback harus aman (tidak crash dan return offset=0)"
    )
