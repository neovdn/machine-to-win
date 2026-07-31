"""
tests/test_no_lookahead.py
==========================
Unit test untuk memverifikasi bahwa backtester dan indikator engine
bebas dari look-ahead bias (zero look-ahead guarantee).

PENGUJIAN:
  1. test_validate_no_lookahead_causal:
     Menguji fungsi validate_no_lookahead() dari backtester.py pada sample acak.
  2. test_future_candle_mutation_signal_immutability:
     Membuktikan secara langsung bahwa jika candle t+1, t+2, ... diubah (mutasi ekstrem),
     sinyal, indikator, keputusan, dan quality score pada candle t TETAP 100% SAMA.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Tambahkan root directory ke sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.indicators import run_all_indicators, get_latest_signals
from engine.rule_engine import evaluate_entry
from engine.backtester import validate_no_lookahead


class TestNoLookaheadBias(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Muat dataset M5 dan H1 dari data/historical/ untuk pengujian.
        """
        cls.m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
        cls.h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")

        if os.path.exists(cls.m5_path):
            cls.df_m5 = pd.read_csv(cls.m5_path)
            cls.df_m5["time"] = pd.to_datetime(cls.df_m5["time"])
            cls.df_m5.set_index("time", inplace=True)
        else:
            # Generate dummy dataframe jika CSV tidak ditemukan
            dates = pd.date_range("2026-01-01", periods=500, freq="5min", tz="UTC")
            np.random.seed(42)
            prices = 2000.0 + np.cumsum(np.random.randn(500))
            cls.df_m5 = pd.DataFrame({
                "open": prices,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices + 0.1,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0
            }, index=dates)

        if os.path.exists(cls.h1_path):
            cls.df_h1 = pd.read_csv(cls.h1_path)
            cls.df_h1["time"] = pd.to_datetime(cls.df_h1["time"])
            cls.df_h1.set_index("time", inplace=True)
        else:
            dates_h1 = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
            np.random.seed(42)
            prices_h1 = 2000.0 + np.cumsum(np.random.randn(100))
            cls.df_h1 = pd.DataFrame({
                "open": prices_h1,
                "high": prices_h1 + 2.0,
                "low": prices_h1 - 2.0,
                "close": prices_h1 + 0.2,
                "tick_volume": 500,
                "spread": 10,
                "real_volume": 0
            }, index=dates_h1)

    def test_validate_no_lookahead_causal(self):
        """
        Uji fungsi validate_no_lookahead() untuk memastikan korespodensi persis
        antara perhitung per-candle vs full batch.
        """
        res = validate_no_lookahead(self.df_m5.copy(), n_samples=5, seed=42)
        self.assertTrue(res["passed"], f"validate_no_lookahead gagal: {res.get('message')}")

    def test_future_candle_mutation_signal_immutability(self):
        """
        Membuktikan bahwa mengubah candle t+1 s/d t+50 TIDAK MENGUBAH
        sinyal, indikator, keputusan, dan setup quality pada candle t.
        """
        t = 200  # Titik candle t
        df_sliced_original = self.df_m5.iloc[:t + 1].copy()

        # Hitung indikator & keputusan pada candle t asli
        df_ind_orig = run_all_indicators(df_sliced_original)
        signals_orig = get_latest_signals(df_ind_orig)
        signals_orig["trend_h1"] = "UPTREND"  # Inject H1 bias
        decision_orig = evaluate_entry(signals_orig)

        # Buat copy data sampai t+50, tapi mutasi candle t+1 s/d t+50 secara ekstrem
        df_mutated = self.df_m5.iloc[:t + 51].copy()

        # Mutasi ekstrem pada candle t+1 s/d t+50
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("close")] = 9999.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("high")] = 10000.0
        df_mutated.iloc[t + 1:, df_mutated.columns.get_loc("low")] = 1.0

        # Hitung indikator pada dataset yang dimutasi
        df_ind_mutated = run_all_indicators(df_mutated)

        # Ambil nilai candle t (baris index ke-t) dari dataset yang dimutasi
        val_t_mutated = df_ind_mutated.iloc[t]

        # Verifikasi bahwa nilai indikator pada candle t TIDAK BERUBAH
        val_t_orig = df_ind_orig.iloc[t]
        self.assertAlmostEqual(float(val_t_orig["ema_9"]), float(val_t_mutated["ema_9"]), places=5)
        self.assertAlmostEqual(float(val_t_orig["ema_21"]), float(val_t_mutated["ema_21"]), places=5)
        self.assertAlmostEqual(float(val_t_orig["rsi_14"]), float(val_t_mutated["rsi_14"]), places=5)
        self.assertAlmostEqual(float(val_t_orig["volume_ratio"]), float(val_t_mutated["volume_ratio"]), places=5)


        # Evaluasi ulang keputusan pada candle t
        signals_mutated_at_t = get_latest_signals(df_ind_mutated.iloc[:t + 1])
        signals_mutated_at_t["trend_h1"] = "UPTREND"
        decision_mutated = evaluate_entry(signals_mutated_at_t)

        self.assertEqual(decision_orig["keputusan"], decision_mutated["keputusan"])
        self.assertEqual(decision_orig["setup_quality"], decision_mutated["setup_quality"])
        self.assertEqual(decision_orig["setup_quality_score"], decision_mutated["setup_quality_score"])


if __name__ == "__main__":
    unittest.main()
