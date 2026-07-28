import os, sys, time
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from engine.data_fetcher import load_candles_csv
from engine.indicators import run_all_indicators
from engine.rule_engine import evaluate_entry
from engine.backtester import merge_h1_to_m5

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

    # Test SINGLE combination
    atr_mult = 0.9
    lookback = 15
    wing     = 3
    rrr_min  = 1.3

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
        jarak_tp = abs(tp_price - entry_price)

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
            pnl_pts = jarak_tp
        elif outcome == "SL_HIT":
            pnl_pts = -jarak_sl
        else:
            if arah == "BUY": pnl_pts = exit_price - entry_price
            else: pnl_pts = entry_price - exit_price
            pnl_pts = max(pnl_pts, -jarak_sl)
            
        spread = 0.50
        pnl_net = pnl_pts - 1.0 # spread * 2
        
        # Calculate RRR realizing (Gross vs Net)
        rrr_gross = round(pnl_pts / jarak_sl, 4) if jarak_sl > 0 else 0.0
        
        eff_profit = max(0.0, jarak_tp - spread)
        eff_risk = jarak_sl + spread
        rrr_net = 0.0
        if outcome == "TP_HIT":
            rrr_net = eff_profit / eff_risk
        elif outcome == "SL_HIT":
            rrr_net = -1.0
        else:
            rrr_net = rrr_gross # fallback
            
        trades.append({
            "outcome": outcome, "pnl_points": pnl_pts, "pnl_net": pnl_net,
            "sl_method": sl_method, "candles_held": candles_held,
            "jarak_sl": jarak_sl, "rrr_gross": rrr_gross, "rrr_net": rrr_net
        })
        in_trade_until = i + candles_held
        
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    tp_n = len(tdf[tdf["outcome"]=="TP_HIT"])
    no_n = len(tdf[tdf["outcome"]=="NO_HIT"])
    sl_n = total - tp_n - no_n
    
    wr = tp_n / (tp_n+sl_n) if (tp_n+sl_n)>0 else 0
    pnl = tdf["pnl_net"].sum()
    equity = tdf["pnl_net"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    
    closed_tdf = tdf[tdf["outcome"]!="NO_HIT"]
    avg_rrr_gross = closed_tdf["rrr_gross"].mean()
    avg_rrr_net = closed_tdf["rrr_net"].mean()
    
    sl_swing = len(tdf[tdf["sl_method"]=="SWING"])
    sl_atr = total - sl_swing

    print("="*50)
    print("HASIL FAST VECTORIZED (NUMPY)")
    print(f"Total Trade      : {total}")
    print(f"TP HIT           : {tp_n}")
    print(f"SL HIT           : {sl_n}")
    print(f"NO HIT           : {no_n}")
    print(f"Win Rate         : {wr*100:.1f}%")
    print(f"NO HIT Rate      : {no_n/total*100:.1f}%")
    print(f"Avg RRR (Gross)  : {avg_rrr_gross:+.2f}R")
    print(f"Avg RRR (Net)    : {avg_rrr_net:+.2f}R")
    print(f"Total PNL (Net)  : {pnl:+.2f}")
    print(f"Max DD (Net)     : {max_dd:.2f}")
    print(f"SL SWING         : {sl_swing}")
    print(f"SL ATR           : {sl_atr}")
    print("="*50)

if __name__ == "__main__":
    main()
