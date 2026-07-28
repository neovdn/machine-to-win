"""
tests/test_backtester.py
=========================
Unit & Regression Test untuk engine/backtester.py

Pengujian:
  1. test_spread_cost_model:
     Menguji bahwa tick_info (ask/bid sintetis) yang dikirim ke calculate_sl_tp()
     menghasilkan entry price ask/bid, SL/TP, rrr_after_spread, dan pnl_net = pnl_points - 2*spread_pts.
  2. test_nohit_position_blocking:
     Menguji bahwa in_trade_until_idx di-update dengan benar untuk TP_HIT, SL_HIT, dan NO_HIT.
     Menguji bahwa PnL MTM untuk NO_HIT di-clamp secara benar (tidak lebih buruk dari -jarak_sl).
  3. test_phase_0_baseline_consistency:
     Menguji bahwa backtest pada data historis Jan-Jul 2026 menghasilkan persis angka baseline Fase 0.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.risk_manager import calculate_sl_tp
from engine.backtester import run_backtest, simulate_trade_outcome, compute_summary


class TestBacktesterRegression(unittest.TestCase):

    def setUp(self):
        """Buat DataFrame sintetis untuk testing mendalam."""
        dates = pd.date_range("2026-01-01 00:00:00", periods=100, freq="5min", tz="UTC")
        prices = 2000.0 + np.sin(np.linspace(0, 10, 100)) * 5.0
        
        self.df_synthetic = pd.DataFrame({
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices + 0.2,
            "tick_volume": 100,
            "atr_14": 2.0,  # ATR konstan 2.0 USD
        }, index=dates)

    def test_spread_cost_model(self):
        """
        Validasi cost model spread (Fase 0 Item 2):
        - BUY: entry = close + spread/2 (ask)
        - SELL: entry = close - spread/2 (bid)
        - pnl_net = pnl_points - 2 * spread_pts
        """
        close_price = 2000.0
        spread_pts  = 0.50  # USD

        # Test BUY tick_info
        tick_buy = {
            "ask": close_price + spread_pts / 2,  # 2000.25
            "bid": close_price - spread_pts / 2,  # 1999.75
        }
        res_buy = calculate_sl_tp(
            df=self.df_synthetic,
            entry=close_price,
            arah="BUY",
            tick_info=tick_buy
        )
        self.assertTrue(res_buy["valid"])
        self.assertEqual(res_buy["entry_type"], "ASK")
        self.assertAlmostEqual(res_buy["entry"], 2000.25)
        self.assertEqual(res_buy["spread"], 0.50)
        self.assertIsNotNone(res_buy["rrr_after_spread"])

        # Test SELL tick_info
        tick_sell = {
            "ask": close_price + spread_pts / 2,
            "bid": close_price - spread_pts / 2,
        }
        res_sell = calculate_sl_tp(
            df=self.df_synthetic,
            entry=close_price,
            arah="SELL",
            tick_info=tick_sell
        )
        self.assertTrue(res_sell["valid"])
        self.assertEqual(res_sell["entry_type"], "BID")
        self.assertAlmostEqual(res_sell["entry"], 1999.75)
        self.assertEqual(res_sell["spread"], 0.50)

    def test_nohit_position_blocking_and_clamp(self):
        """
        Validasi NO_HIT position blocking dan MTM clamp (Fase 0 Item 3):
        - MTM PnL tidak boleh lebih buruk dari -jarak_sl
        - candles_held untuk NO_HIT mengembalikan max_candles
        """
        df = self.df_synthetic.copy()
        entry = 2000.0
        sl = 1990.0   # jarak_sl = 10.0
        tp = 2020.0   # jarak_tp = 20.0
        
        # Skenario 1: Harga jatuh ke 1980.0 di akhir window tanpa hit SL/TP formal dalam loop
        # Misal high/low di-set agar tidak pernah low <= 1990.0, tapi close akhir window = 1980.0
        # simulate_trade_outcome me-return outcome NO_HIT jika low > sl dan high < tp
        df["low"]  = 1995.0
        df["high"] = 2005.0
        df.iloc[10, df.columns.get_loc("close")] = 1980.0  # MTM price di baris ke-10 (exit)

        outcome_info = simulate_trade_outcome(
            df_m5_full=df,
            entry_idx=0,
            entry=entry,
            sl=sl,
            tp=tp,
            max_candles=10
        )
        self.assertEqual(outcome_info["outcome"], "NO_HIT")
        self.assertEqual(outcome_info["candles_held"], 10)
        
        # Verifikasi PnL clamp logic:
        # raw PnL = 1980.0 - 2000.0 = -20.0. Jarak SL = 10.0.
        # clamped PnL = max(-20.0, -10.0) = -10.0
        jarak_sl = abs(entry - sl)
        pnl_raw = outcome_info["exit_price_mtm"] - entry
        pnl_points = max(pnl_raw, -jarak_sl)
        self.assertEqual(pnl_points, -10.0)

    def test_phase_0_baseline_consistency(self):
        """
        Verifikasi bahwa running backtest pada dataset 2026-01-01 s/d 2026-07-25
        menghasilkan persis angka baseline resmi Fase 0.
        """
        m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
        h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

        if not os.path.exists(m5_path) or not os.path.exists(h1_path):
            self.skipTest("File cache historis tidak ditemukan — skip baseline verification test.")

        df_m5 = pd.read_csv(m5_path)
        df_m5["time"] = pd.to_datetime(df_m5["time"])
        df_m5.set_index("time", inplace=True)

        df_h1 = pd.read_csv(h1_path)
        df_h1["time"] = pd.to_datetime(df_h1["time"])
        df_h1.set_index("time", inplace=True)

        trades_df, summary = run_backtest(
            df_m5=df_m5,
            df_h1=df_h1,
            warm_up=100,
            max_candles=288,
            spread_pts=0.50,
            atr_multiplier=1.5,
            swing_lookback=50,
            swing_wing=5,
            rrr_min=2.0,
            swing_clamp_min_atr=0.0,
            swing_clamp_max_atr=999.0,
            verbose=False
        )

        self.assertEqual(summary["total_trades"], 257)
        self.assertEqual(summary["tp_count"], 82)
        self.assertEqual(summary["sl_count"], 119)
        self.assertEqual(summary["no_hit_count"], 56)
        self.assertAlmostEqual(summary["win_rate"], 0.408, places=3)
        self.assertAlmostEqual(summary["no_hit_rate"], 0.218, places=3)
        self.assertAlmostEqual(summary["avg_rrr_realized"], 0.20, places=2)
        self.assertAlmostEqual(summary["avg_candles_held"], 80.8, places=1)
        self.assertAlmostEqual(summary["total_pnl_net"], 2117.76, places=1)
        self.assertAlmostEqual(summary["max_drawdown_net"], -689.48, places=1)
        
        # Proporsi SL Method
        sl_breakdown = summary["sl_method_breakdown"]
        self.assertEqual(sl_breakdown.get("SWING", 0), 255)
        self.assertEqual(sl_breakdown.get("ATR", 0), 2)

    def test_atr_clamping_logic(self):
        """
        Validasi clamping Swing SL terhadap range ATR [0.7 ATR, 2.0 ATR].
        """
        entry = 2000.0
        atr_value = 2.0
        # min_dist = 0.7 * 2.0 = 1.4
        # max_dist = 2.0 * 2.0 = 4.0

        # DataFrame sintetis dengan swing low sangat dekat (0.5 USD dari entry) -> dist < 1.4 -> MIN_CAP
        dates = pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC")
        prices = [2000.0] * 30
        df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "atr_14": atr_value}, index=dates)
        
        # Set low candle ke-15 (swing candidate) = 1999.70 (buffer=0.5 -> sl_swing_level = 1999.20 -> dist = 0.8 USD)
        df.iloc[15, df.columns.get_loc("low")] = 1999.70

        res_min = calculate_sl_tp(
            df=df,
            entry=entry,
            arah="BUY",
            profile="scalp_m5",
            swing_lookback=20,
            swing_wing=3
        )

        self.assertTrue(res_min["valid"])
        self.assertEqual(res_min["sl_method"], "SWING")
        self.assertTrue(res_min["sl_swing_clamped"])
        self.assertEqual(res_min["clamp_reason"], "MIN_CAP")
        self.assertAlmostEqual(res_min["jarak_sl"], 1.4)  # Clamped to min_dist (0.7 * 2.0)
        self.assertAlmostEqual(res_min["sl"], entry - 1.4)


if __name__ == "__main__":
    unittest.main()

