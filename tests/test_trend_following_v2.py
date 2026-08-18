"""
Tests for Phase 17: Trend Following v2 Strategy.
"""

import os
import pytest
import pandas as pd
import numpy as np

from engine.strategies.trend_following_v2 import (
    evaluate_trend_following,
    TREND_PULLBACK_PROXIMITY_ATR
)
from engine.risk_manager import SWING_LOOKBACK, SWING_WING

def _create_mock_df() -> pd.DataFrame:
    # We need len(df) >= lookback + wing*2 + 1
    # 15 + 6 + 1 = 22 candles minimum.
    # Let's create 30 candles. index 0 to 29.
    
    data = []
    for i in range(30):
        data.append({
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "atr_14": 2.0,
            "trend": "UPTREND",
            "ema_9": 100.0,
            "ema_21": 90.0,
            "ema_gap_pct": 0.5,
        })
    df = pd.DataFrame(data)
    
    # We want index 29 to be the evaluation candle.
    # It must break minor structure (BUY): close > max(high[28], high[27])
    df.loc[27, "high"] = 101.0
    df.loc[28, "high"] = 102.0
    df.loc[29, "close"] = 103.0 # breaks minor structure
    
    # We need a swing low for BUY in the lookback window.
    # lookback=15, wing=3.
    # The lookback window starts at 29-15 = 14 to 28.
    # So we can put the swing at index 24.
    # Window: 21, 22, 23 (left), 24 (center), 25, 26, 27 (right)
    df.loc[21:27, "low"] = [95, 94, 93, 90, 93, 94, 95] # 90 at index 24
    
    return df

def test_1_buy_valid():
    df = _create_mock_df()
    
    # close_now at index 29 will be 91.5
    df.loc[27, "high"] = 90.5
    df.loc[28, "high"] = 90.8
    df.loc[29, "close"] = 91.5 # breaks minor high (90.8)
    # distance = 91.5 - 90.0 = 1.5. ATR = 2.0. Limit = 1.0 * 2.0 = 2.0. 1.5 <= 2.0 -> OK.
    
    res = evaluate_trend_following(df, 29, "BULLISH", pullback_proximity_atr=1.0)
    
    assert res["terpenuhi"] is True
    assert res["arah"] == "BUY"
    assert res["ema_trigger_ok"] is True
    assert res["pullback_ok"] is True
    assert res["structure_break_ok"] is True
    assert res["invalidation_level_sl"] == 90.0
    assert res["pullback_swing_level"] == 90.0

def test_2_sell_valid():
    df = _create_mock_df()
    # Setup for SELL
    df["trend"] = "DOWNTREND"
    df["ema_9"] = 90.0
    df["ema_21"] = 100.0
    df["ema_gap_pct"] = -0.5
    
    # Needs swing high
    df.loc[21:27, "high"] = [105, 106, 107, 110, 107, 106, 105] # swing high at 24 is 110
    
    # break minor structure for SELL: close_now < min(low_1, low_2)
    df.loc[27, "low"] = 109.0
    df.loc[28, "low"] = 109.5
    df.loc[29, "close"] = 108.5 # breaks minor low (109.0)
    
    # pullback distance = 110.0 - 108.5 = 1.5 <= 2.0 -> OK.
    
    res = evaluate_trend_following(df, 29, "BEARISH", pullback_proximity_atr=1.0)
    
    assert res["terpenuhi"] is True
    assert res["arah"] == "SELL"
    assert res["invalidation_level_sl"] == 110.0
    assert res["pullback_swing_level"] == 110.0

def test_3_ema_trigger_not_match():
    df = _create_mock_df()
    df.loc[29, "trend"] = "SIDEWAYS" # trigger doesn't match
    
    res = evaluate_trend_following(df, 29, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["ema_trigger_ok"] is False

def test_4_swing_not_found():
    df = _create_mock_df()
    # Remove swing low by making it strictly decreasing
    df["low"] = np.linspace(100, 50, len(df))
    
    res = evaluate_trend_following(df, 29, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["pullback_ok"] is False
    assert res["pullback_swing_level"] is None

def test_5_pullback_too_far():
    df = _create_mock_df()
    df.loc[27, "high"] = 90.5
    df.loc[28, "high"] = 90.8
    # distance will be 95.0 - 90.0 = 5.0. ATR is 2.0. Limit is 2.0. 5.0 > 2.0 -> FAIL.
    df.loc[29, "close"] = 95.0 # Breaks minor structure but is too far
    
    res = evaluate_trend_following(df, 29, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["pullback_ok"] is False
    assert res["pullback_swing_level"] == 90.0

def test_6_structure_break_fails():
    df = _create_mock_df()
    df.loc[27, "high"] = 90.5
    df.loc[28, "high"] = 92.0
    df.loc[29, "close"] = 91.5 # Does not break high of 92.0
    
    res = evaluate_trend_following(df, 29, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["structure_break_ok"] is False

def test_7_arah_unknown():
    df = _create_mock_df()
    res = evaluate_trend_following(df, 29, "UNKNOWN")
    assert res["terpenuhi"] is False
    assert res["arah"] == "NETRAL"
    assert "arah tidak dikenal" in res["keterangan"]

def test_8_idx_too_small():
    df = _create_mock_df()
    res = evaluate_trend_following(df, 1, "BULLISH")
    assert res["terpenuhi"] is False
    assert res["structure_break_ok"] is False
    assert "data tidak cukup untuk cek struktur minor" in res["keterangan"]

def test_9_causality():
    df = _create_mock_df()
    df.loc[27, "high"] = 90.5
    df.loc[28, "high"] = 90.8
    df.loc[29, "close"] = 91.5
    
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
    
    res1 = evaluate_trend_following(df, 29, "BULLISH", pullback_proximity_atr=1.0)
    res2 = evaluate_trend_following(df_extended, 29, "BULLISH", pullback_proximity_atr=1.0)
    
    assert res1 == res2

def test_10_proof_of_reuse():
    filepath = os.path.join("engine", "strategies", "trend_following_v2.py")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "_check_ema_trigger_m5" in content, "Harus reuse _check_ema_trigger_m5"
    assert "find_nearest_swing" in content, "Harus reuse find_nearest_swing"
    
    assert "_check_bias_h1" not in content, "TIDAK BOLEH reuse _check_bias_h1"
    assert "get_h1_context" not in content, "TIDAK BOLEH reuse get_h1_context"
