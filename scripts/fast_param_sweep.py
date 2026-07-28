import os, sys, time, itertools
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.rule_engine import evaluate_entry
from engine.backtester import merge_h1_to_m5, compute_summary

def main():
    m5_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_M5_2026-01-01_2026-07-25.csv")
    h1_path = os.path.join(ROOT_DIR, "data", "historical", "XAUUSD_H1_2026-01-01_2026-07-25.csv")
    df_m5 = load_candles_csv(m5_path)
    df_h1 = load_candles_csv(h1_path)
    df_m5_ind = run_all_indicators(df_m5)
    df_h1_ind = run_all_indicators(df_h1)
    df_merged = merge_h1_to_m5(df_m5_ind, df_h1_ind)

    # Convert to fast numpy arrays
    n_total = len(df_merged)
    closes = df_merged["close"].values
    highs = df_merged["high"].values
    lows = df_merged["low"].values
    emas9 = df_merged["ema_9"].values
    emas21 = df_merged["ema_21"].values
    rsis = df_merged["rsi_14"].values
    trends = df_merged["trend"].values
    ema_gaps = df_merged["ema_gap_pct"].values
    trends_h1 = df_merged["trend_h1"].values
    atrs = df_merged["atr_14"].values

    atr_mults = [0.7, 0.9, 1.1, 1.3]
    lookbacks = [10, 15, 20, 30]
    wings     = [2, 3, 4]
    rrr_mins  = [1.2, 1.3, 1.5, 1.8]
    grid = list(itertools.product(atr_mults, lookbacks, wings, rrr_mins))

    # Precompute signals
    valid_signals = []
    for i in range(100, n_total):
        if pd.isna(trends_h1[i]): continue
        sig = {
            "time": df_merged.index[i],
            "close": closes[i], "ema_9": emas9[i], "ema_21": emas21[i],
            "rsi_14": rsis[i], "trend": trends[i], "ema_gap_pct": ema_gaps[i], "trend_h1": trends_h1[i]
        }
        dec = evaluate_entry(sig)
        if dec["keputusan"] in ("BUY", "SELL"):
            valid_signals.append((i, dec["keputusan"]))

    results = []
    start_time = time.time()
    
    for (atr_mult, lookback, wing, rrr_min) in grid:
        trades = []
        in_trade_until = -1
        
        for (i, arah) in valid_signals:
            if i <= in_trade_until: continue
            
            # Risk calc fast
            atr_val = atrs[i]
            entry_price = closes[i]
            sl_method = "ATR"
            sl_price = entry_price - (atr_val * atr_mult) if arah == "BUY" else entry_price + (atr_val * atr_mult)
            clamped = False
            
            # Swing calc
            start_idx = max(0, i - lookback + 1)
            # Find swing
            best_swing = None
            if arah == "BUY":
                # Swing low
                for j in range(start_idx + wing, i - wing):
                    if lows[j] == np.min(lows[j-wing:j+wing+1]):
                        best_swing = lows[j]
            else:
                for j in range(start_idx + wing, i - wing):
                    if highs[j] == np.max(highs[j-wing:j+wing+1]):
                        best_swing = highs[j]
                        
            if best_swing is not None:
                swing_dist = abs(entry_price - best_swing)
                max_dist = atr_val * 2.0
                if swing_dist > max_dist:
                    sl_price = entry_price - max_dist if arah == "BUY" else entry_price + max_dist
                    clamped = True
                    sl_method = "SWING"
                else:
                    sl_price = best_swing
                    sl_method = "SWING"

            jarak_sl = abs(entry_price - sl_price)
            if jarak_sl < 0.50: continue # min distance
            
            tp_price = entry_price + (jarak_sl * rrr_min) if arah == "BUY" else entry_price - (jarak_sl * rrr_min)
            
            # Simulate outcome fast
            outcome = "NO_HIT"
            candles_held = 288
            exit_price = closes[min(i+288, n_total-1)]
            
            for f in range(1, 289):
                idx = i + f
                if idx >= n_total:
                    candles_held = f - 1
                    exit_price = closes[-1]
                    break
                
                cur_h = highs[idx]
                cur_l = lows[idx]
                
                hit_tp = False
                hit_sl = False
                
                if arah == "BUY":
                    if cur_l <= sl_price: hit_sl = True
                    if cur_h >= tp_price: hit_tp = True
                else:
                    if cur_h >= sl_price: hit_sl = True
                    if cur_l <= tp_price: hit_tp = True
                    
                if hit_tp and hit_sl:
                    outcome = "SL_HIT" # conservative
                    candles_held = f
                    break
                elif hit_tp:
                    outcome = "TP_HIT"
                    candles_held = f
                    break
                elif hit_sl:
                    outcome = "SL_HIT"
                    candles_held = f
                    break
                    
            pnl_pts = 0
            if outcome == "TP_HIT":
                pnl_pts = abs(entry_price - tp_price)
            elif outcome == "SL_HIT":
                pnl_pts = -jarak_sl
            else:
                if arah == "BUY": pnl_pts = exit_price - entry_price
                else: pnl_pts = entry_price - exit_price
                pnl_pts = max(pnl_pts, -jarak_sl)
                
            pnl_net = pnl_pts - 1.0 # 0.5 spread * 2
            
            trades.append({
                "outcome": outcome, "pnl_points": pnl_pts, "pnl_net": pnl_net,
                "sl_method": sl_method, "candles_held": candles_held,
                "jarak_sl": jarak_sl, "rrr_realized": pnl_pts/jarak_sl if jarak_sl>0 else 0
            })
            in_trade_until = i + candles_held
            
        # summarize
        if not trades: continue
        tdf = pd.DataFrame(trades)
        total = len(tdf)
        tp_n = len(tdf[tdf["outcome"]=="TP_HIT"])
        no_n = len(tdf[tdf["outcome"]=="NO_HIT"])
        sl_n = total - tp_n - no_n
        
        wr = tp_n / (tp_n+sl_n) if (tp_n+sl_n)>0 else 0
        pnl = tdf["pnl_net"].sum()
        equity = tdf["pnl_net"].cumsum()
        max_dd = (equity - equity.cummax()).min()
        
        results.append({
            "atr": atr_mult, "look": lookback, "wing": wing, "rrr": rrr_min,
            "win_rate": wr*100, "no_hit_pct": no_n/total*100,
            "avg_rrr": tdf[tdf["outcome"]!="NO_HIT"]["rrr_realized"].mean() if (tp_n+sl_n)>0 else 0,
            "avg_cand": tdf["candles_held"].mean(),
            "pnl": pnl, "dd": max_dd,
            "sl_swing": len(tdf[tdf["sl_method"]=="SWING"]),
            "total": total
        })
        
    df_res = pd.DataFrame(results).sort_values("pnl", ascending=False)
    print("TOP 10 BY PNL:")
    print(df_res.head(10).to_string())
    print("\nTOP 10 BY DD:")
    print(df_res.sort_values("dd", ascending=False).head(10).to_string())

if __name__ == "__main__":
    main()
