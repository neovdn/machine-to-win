"""Analyze Phase 1 parameters in calibration sweep results."""
import pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'backtest_results', 'oos_calib_sweep_results.csv'))

# Phase 1 params
f1 = df[(df['atr_multiplier']==0.9) & (df['swing_lookback']==15) & (df['swing_wing']==3) & (df['rrr_min']==1.3)]
print('=== PARAMETER FASE 1 DI KALIBRASI (atr=0.9, look=15, wing=3, rrr=1.3) ===')
if f1.empty:
    print('Tidak ditemukan!')
else:
    r = f1.iloc[0]
    print(f'win_rate:         {r.win_rate_pct:.2f}%')
    print(f'avg_rrr_realized: {r.avg_rrr_realized:+.4f}')
    print(f'no_hit_rate:      {r.no_hit_rate_pct:.2f}%')
    print(f'total_pnl_net:    {r.total_pnl_net:+.2f}')
    print(f'max_drawdown_net: {r.max_drawdown_net:+.2f}')
    print(f'total_trades:     {int(r.total_trades)}')
    print(f'composite_score:  {r.composite_score:.4f}')

    ranked = df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    mask = (ranked.atr_multiplier==0.9) & (ranked.swing_lookback==15) & (ranked.swing_wing==3) & (ranked.rrr_min==1.3)
    idx = ranked[mask].index[0]
    print(f'Rank (composite): #{idx+1} dari {len(df)}')

# Also show Phase 1 validation result
print()
print('=== TOP 5 TERBAIK CALIB (bukan parameter terpilih, tapi untuk konteks) ===')
ranked = df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
for i in range(5):
    r = ranked.iloc[i]
    print(f'#{i+1}: atr={r.atr_multiplier} look={int(r.swing_lookback)} wing={int(r.swing_wing)} rrr={r.rrr_min} | wr={r.win_rate_pct:.1f}% rrr_real={r.avg_rrr_realized:+.4f} pnl={r.total_pnl_net:+.1f} score={r.composite_score:.4f}')
