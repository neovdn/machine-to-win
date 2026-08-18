import pytest
import pandas as pd
import numpy as np
import os
import sys

# Tambahkan root directory ke path agar bisa import dari scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.analyze_winrate_vs_quality_regime import (
    wilson_score_interval,
    reconciliation_check
)

def test_wilson_score_interval():
    # Test cases with known results
    # For n=100, p=0.5, z=1.96 (approx)
    # The interval should be roughly [0.4, 0.6]
    lower, upper = wilson_score_interval(50, 100)
    assert 0.39 < lower < 0.41
    assert 0.59 < upper < 0.61

    # For edge cases
    # n=10, p=0 (0 successes)
    lower, upper = wilson_score_interval(0, 10)
    assert lower == 0.0
    assert upper > 0.0

    # n=10, p=1 (10 successes)
    lower, upper = wilson_score_interval(10, 10)
    assert lower < 1.0
    assert upper >= 0.999

    # n=0, should handle gracefully or return 0,0
    lower, upper = wilson_score_interval(0, 0)
    assert lower == 0.0
    assert upper == 0.0

def test_reconciliation_check():
    # Buat dataframe dummy
    df_raw = pd.DataFrame({
        'outcome': ['TP_HIT', 'SL_HIT', 'TP_HIT', 'NO_HIT'],
        'setup_quality': ['STRONG', 'MODERATE', 'WEAK', 'STRONG']
    })
    
    # df_closed hanya berisi TP_HIT dan SL_HIT (3 baris)
    df_closed = df_raw[df_raw['outcome'].isin(['TP_HIT', 'SL_HIT'])]
    
    # Fungsi seharusnya pass tanpa melempar ValueError jika angkanya sama
    # Total raw: 4, Total closed: 3
    try:
        reconciliation_check(len(df_raw), len(df_closed), df_closed, ['STRONG', 'MODERATE', 'WEAK'], 'setup_quality')
    except ValueError:
        pytest.fail("reconciliation_check gagal untuk kondisi yang benar")
        
    # Test mismatch: kita hapus satu baris dari df_closed
    with pytest.raises(ValueError, match="Reconciliation failed"):
        reconciliation_check(len(df_raw), len(df_closed) - 1, df_closed, ['STRONG', 'MODERATE', 'WEAK'], 'setup_quality')
