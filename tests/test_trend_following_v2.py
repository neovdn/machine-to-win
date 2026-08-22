"""
Tests for Phase 17: Trend Following v2 Strategy (Revisi Temporal Windowing).
"""

import os
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np

from engine.strategies.trend_following_v2 import (
    evaluate_trend_following,
    TREND_PULLBACK_PROXIMITY_ATR,
    TREND_CONFIRMATION_WINDOW_M5
)
from engine.risk_manager import SWING_LOOKBACK, SWING_WING

def _create_mock_df() -> pd.DataFrame:
    # Buat 40 candle untuk memberikan cukup ruang (window + lookback).
    data = []
    for i in range(40):
        data.append({
            "open": 100.0,
            "high": 105.0 - (i * 0.01),
            "low": 95.0 + (i * 0.01),
            "close": 102.0,
            "atr_14": 2.0,
            "trend": "UPTREND",
            "ema_9": 100.0,
            "ema_21": 90.0,
            "ema_gap_pct": 0.5,
        })
    df = pd.DataFrame(data)
    
    # idx evaluasi konfirmasi = 39
    df.loc[37, "high"] = 101.0
    df.loc[38, "high"] = 102.0
    df.loc[39, "close"] = 103.0 # break minor structure for BUY
    
    # Buat swing low di idx 25 (agar masuk dalam lookback=15 dari k=29..38)
    df.loc[22:28, "low"] = [95, 94, 93, 90, 93, 94, 95] # Swing low = 90 di idx 25
    
    return df

def test_1_buy_valid():
    df = _create_mock_df()
    
    # Confirmation di 39 (BUY)
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 90.8
    df.loc[39, "close"] = 91.5 # break minor high (90.8)
    
    # Pullback setup di idx 36 (k = 36) -> dalam window 10 (39 - 10 = 29)
    # distance = 91.5 - 90.0 = 1.5. ATR = 2.0. Limit = 2.0.
    df.loc[36, "close"] = 91.5 
    
    res = evaluate_trend_following(df, 39, "BULLISH", pullback_proximity_atr=1.0)
    print("DEBUG TEST 1:", res)
    assert res["terpenuhi"] is True
    assert res["arah"] == "BUY"
    assert res["ema_trigger_ok"] is True
    assert res["pullback_ok"] is True
    assert res["structure_break_ok"] is True
    assert res["invalidation_level_sl"] == 90.0
    assert res["pullback_swing_level"] == 90.0
    assert res["pullback_idx"] == 36
    assert res["candles_since_pullback"] == 3

def test_2_sell_valid():
    df = _create_mock_df()
    # Setup for SELL
    df["trend"] = "DOWNTREND"
    df["ema_9"] = 90.0
    df["ema_21"] = 100.0
    df["ema_gap_pct"] = -0.5
    
    # Swing high di idx 25
    df.loc[22:28, "high"] = [105, 106, 107, 110, 107, 106, 105] 
    
    # Confirmation di 39 (SELL)
    df.loc[37, "low"] = 109.0
    df.loc[38, "low"] = 109.5
    df.loc[39, "close"] = 108.5 # break minor low
    
    # Pullback setup di idx 35 (k = 35)
    # distance = 110.0 - 108.5 = 1.5 <= 2.0
    df.loc[35, "close"] = 108.5
    
    res = evaluate_trend_following(df, 39, "BEARISH", pullback_proximity_atr=1.0)
    
    assert res["terpenuhi"] is True
    assert res["arah"] == "SELL"
    assert res["pullback_idx"] == 35
    assert res["invalidation_level_sl"] == 110.0

@patch("engine.strategies.trend_following_v2.find_nearest_swing")
def test_3_ema_trigger_not_match(mock_find_swing):
    df = _create_mock_df()
    df.loc[39, "trend"] = "SIDEWAYS" # trigger doesn't match
    
    res = evaluate_trend_following(df, 39, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["ema_trigger_ok"] is False
    
    # Membuktikan short-circuit: scan mundur TIDAK dijalankan sama sekali
    mock_find_swing.assert_not_called()

def test_4_swing_not_found():
    df = _create_mock_df()
    # Remove swing low by making it strictly decreasing
    df["low"] = np.linspace(100, 50, len(df))
    
    res = evaluate_trend_following(df, 39, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["pullback_ok"] is False
    assert res["pullback_swing_level"] is None
    assert res["pullback_idx"] is None

def test_5_pullback_too_far():
    df = _create_mock_df()
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 90.8
    df.loc[39, "close"] = 95.0 # Confirmation OK
    
    # Jauhkan close untuk semua candle dalam window (29 s/d 38)
    df.loc[29:38, "close"] = 95.0 # distance = 5.0 > 2.0
    
    res = evaluate_trend_following(df, 39, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["pullback_ok"] is False
    # window exhausted
    assert res["pullback_idx"] is None
    assert "tidak ada pullback valid" in res["keterangan"]

@patch("engine.strategies.trend_following_v2.find_nearest_swing")
def test_6_structure_break_fails(mock_find_swing):
    df = _create_mock_df()
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 92.0
    df.loc[39, "close"] = 91.5 # Does not break high of 92.0
    
    res = evaluate_trend_following(df, 39, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["structure_break_ok"] is False
    
    # Membuktikan short-circuit: scan mundur TIDAK dijalankan sama sekali
    mock_find_swing.assert_not_called()

def test_7_arah_unknown():
    df = _create_mock_df()
    res = evaluate_trend_following(df, 39, "UNKNOWN")
    assert res["terpenuhi"] is False
    assert res["arah"] == "NETRAL"

def test_8_idx_too_small():
    df = _create_mock_df()
    res = evaluate_trend_following(df, 1, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["structure_break_ok"] is False

def test_9_causality():
    df = _create_mock_df()
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 90.8
    df.loc[39, "close"] = 91.5
    df.loc[36, "close"] = 91.5 # valid pullback
    
    # Append future candles
    future_data = []
    for i in range(10):
        future_data.append({
            "open": 999.0,
            "high": 999.0,
            "low": 999.0,
            "close": 999.0,
            "atr_14": 999.0,
            "trend": "SIDEWAYS",
            "ema_9": 999.0,
            "ema_21": 999.0,
            "ema_gap_pct": 999.0,
        })
    df_extended = pd.concat([df, pd.DataFrame(future_data)], ignore_index=True)
    
    res1 = evaluate_trend_following(df, 39, "BULLISH", pullback_proximity_atr=1.0)
    res2 = evaluate_trend_following(df_extended, 39, "BULLISH", pullback_proximity_atr=1.0)
    
    assert res1 == res2

def test_10_proof_of_reuse():
    filepath = os.path.join("engine", "strategies", "trend_following_v2.py")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "_check_ema_trigger_m5" in content, "Harus reuse _check_ema_trigger_m5"
    assert "find_nearest_swing" in content, "Harus reuse find_nearest_swing"

def test_11_pullback_at_window_boundary():
    df = _create_mock_df()
    # Confirmation di 39 (BUY)
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 90.8
    df.loc[39, "close"] = 91.5 # break minor high
    
    # Boundary: idx_m5 - TREND_CONFIRMATION_WINDOW_M5 = 39 - 10 = 29
    df.loc[29:38, "close"] = 95.0 # too far
    df.loc[29, "close"] = 91.5 # EXACTLY at boundary, distance = 1.5 <= 2.0
    
    res = evaluate_trend_following(df, 39, "BULLISH", pullback_proximity_atr=1.0)
    assert res["terpenuhi"] is True
    assert res["pullback_idx"] == 29

def test_12_pullback_outside_window_boundary():
    df = _create_mock_df()
    # Confirmation di 39 (BUY)
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 90.8
    df.loc[39, "close"] = 91.5 
    
    # Boundary adalah 29. Letakkan pullback di 28.
    df.loc[29:38, "close"] = 95.0 # too far
    df.loc[28, "close"] = 91.5 # OUTSIDE window
    
    res = evaluate_trend_following(df, 39, "BULLISH", pullback_proximity_atr=1.0)
    assert res["terpenuhi"] is False
    assert res["pullback_idx"] is None
    assert "tidak ada pullback valid" in res["keterangan"]

def test_13_multiple_pullbacks_takes_most_recent():
    df = _create_mock_df()
    df.loc[37, "high"] = 90.5
    df.loc[38, "high"] = 90.8
    df.loc[39, "close"] = 91.5 
    
    # Ada 2 pullback dalam window
    df.loc[30, "close"] = 91.5 # Pullback lama
    df.loc[35, "close"] = 91.5 # Pullback LEBIH BARU
    
    res = evaluate_trend_following(df, 39, "BULLISH", pullback_proximity_atr=1.0)
    assert res["terpenuhi"] is True
    # Harus memilih yang paling baru
    assert res["pullback_idx"] == 35
    assert res["candles_since_pullback"] == 4
