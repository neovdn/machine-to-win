"""
Tests for Phase 18: Location & Confluence Filter
"""

import copy
import pytest
from engine.location_filter import (
    calculate_location_score,
    calculate_h1_confluence_score,
    calculate_confluence_summary,
    LOCATION_RR_DEPTH_STRONG_ATR,
    LOCATION_RR_DEPTH_MODERATE_ATR,
    LOCATION_BR_FRESH_CANDLES,
    LOCATION_BR_MODERATE_CANDLES,
    LOCATION_TF_PULLBACK_STRONG_ATR,
    LOCATION_TF_PULLBACK_MODERATE_ATR
)

def test_1_rr_depth_large():
    res = calculate_location_score(
        "RANGE_REVERSAL",
        {"terpenuhi": True, "sweep_depth": 3.5},
        atr=10.0
    ) # ratio = 0.35 >= 0.3 -> 2
    assert res["score"] == 2

def test_2_rr_depth_threshold():
    res = calculate_location_score(
        "RANGE_REVERSAL",
        {"terpenuhi": True, "sweep_depth": 1.5},
        atr=10.0
    ) # ratio = 0.15 >= 0.15 -> 1
    assert res["score"] == 1
    
    res2 = calculate_location_score(
        "RANGE_REVERSAL",
        {"terpenuhi": True, "sweep_depth": 1.0},
        atr=10.0
    ) # ratio = 0.10 < 0.15 -> 0
    assert res2["score"] == 0

def test_3_br_touch_small():
    res = calculate_location_score(
        "BREAKOUT_RETEST",
        {"terpenuhi": True, "candles_since_touch": 2},
        atr=10.0
    )
    assert res["score"] == 2

def test_4_br_touch_large():
    res = calculate_location_score(
        "BREAKOUT_RETEST",
        {"terpenuhi": True, "candles_since_touch": 10},
        atr=10.0
    )
    assert res["score"] == 0

def test_5_tf_pullback_close():
    res = calculate_location_score(
        "TREND_FOLLOWING",
        {"terpenuhi": True, "pullback_distance": 2.5},
        atr=10.0
    ) # ratio = 0.25 <= 0.3 -> 2
    assert res["score"] == 2

def test_6_tf_pullback_far():
    res = calculate_location_score(
        "TREND_FOLLOWING",
        {"terpenuhi": True, "pullback_distance": 9.5},
        atr=10.0
    ) # ratio = 0.95 > 0.7 -> 0
    assert res["score"] == 0

def test_7_unknown_strategy():
    res = calculate_location_score(
        "UNKNOWN_STRAT",
        {"terpenuhi": True, "sweep_depth": 5.0},
        atr=10.0
    )
    assert res["score"] == 0

def test_8_terpenuhi_false():
    strat_res = {"terpenuhi": False, "sweep_depth": 5.0}
    strat_res_orig = copy.deepcopy(strat_res)
    
    res = calculate_location_score("RANGE_REVERSAL", strat_res, atr=10.0)
    
    assert res["score"] == 0
    # ensure it is unmodified
    assert strat_res == strat_res_orig

def test_9_atr_invalid():
    res = calculate_location_score(
        "RANGE_REVERSAL",
        {"terpenuhi": True, "sweep_depth": 5.0},
        atr=0.0
    )
    assert res["score"] == 0

def test_10_h1_strong():
    res = calculate_h1_confluence_score(
        {"arah": "BUY"},
        {"bias": "BULLISH", "strength_zone": "STRONG"}
    )
    assert res["score"] == 2
    assert res["arah_cocok"] is True

def test_11_h1_weak():
    res = calculate_h1_confluence_score(
        {"arah": "BUY"},
        {"bias": "BULLISH", "strength_zone": "WEAK"}
    )
    assert res["score"] == 0
    assert res["arah_cocok"] is True

def test_12_h1_opposite():
    res = calculate_h1_confluence_score(
        {"arah": "BUY"},
        {"bias": "BEARISH", "strength_zone": "STRONG"}
    )
    assert res["score"] == 0
    assert res["arah_cocok"] is False

def test_13_h1_neutral():
    res = calculate_h1_confluence_score(
        {"arah": "BUY"},
        {"bias": "NEUTRAL", "strength_zone": "STRONG"}
    )
    assert res["score"] == 0
    assert res["arah_cocok"] is False

def test_14_h1_none():
    res = calculate_h1_confluence_score(
        {"arah": "BUY"},
        None
    )
    assert res["score"] == 0

def test_15_confluence_summary():
    strat_strong = {"terpenuhi": True, "arah": "BUY", "sweep_depth": 3.5}
    h1_strong = {"bias": "BULLISH", "strength_zone": "STRONG"}
    
    # 2 + 2 = 4 (STRONG) (ceil(0.875 * 4) = 4)
    res_strong = calculate_confluence_summary("RANGE_REVERSAL", strat_strong, 10.0, h1_strong)
    assert res_strong["total_score"] == 4
    assert res_strong["quality_label"] == "STRONG"
    
    # 0 + 0 = 0 (WEAK)
    strat_weak = {"terpenuhi": True, "arah": "BUY", "sweep_depth": 1.0}
    h1_weak = {"bias": "BULLISH", "strength_zone": "WEAK"}
    res_weak = calculate_confluence_summary("RANGE_REVERSAL", strat_weak, 10.0, h1_weak)
    assert res_weak["total_score"] == 0
    assert res_weak["quality_label"] == "WEAK"
    
    # 2 + 0 = 2 (MODERATE) (ceil(0.5 * 4) = 2)
    res_mod = calculate_confluence_summary("RANGE_REVERSAL", strat_strong, 10.0, h1_weak)
    assert res_mod["total_score"] == 2
    assert res_mod["quality_label"] == "MODERATE"
    
    # 1 + 2 = 3 (MODERATE)
    strat_mod = {"terpenuhi": True, "arah": "BUY", "sweep_depth": 1.5}
    res_mod2 = calculate_confluence_summary("RANGE_REVERSAL", strat_mod, 10.0, h1_strong)
    assert res_mod2["total_score"] == 3
    assert res_mod2["quality_label"] == "MODERATE"

def test_16_purity():
    strat_res = {"terpenuhi": True, "arah": "SELL", "candles_since_touch": 5}
    h1_ctx = {"bias": "BEARISH", "strength_zone": "MODERATE"}
    
    res1 = calculate_confluence_summary("BREAKOUT_RETEST", copy.deepcopy(strat_res), 10.0, copy.deepcopy(h1_ctx))
    res2 = calculate_confluence_summary("BREAKOUT_RETEST", copy.deepcopy(strat_res), 10.0, copy.deepcopy(h1_ctx))
    
    assert res1 == res2
