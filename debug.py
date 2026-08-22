import pandas as pd
from engine.strategies.trend_following_v2 import evaluate_trend_following
from tests.test_trend_following_v2 import _create_mock_df

df = _create_mock_df()
df.loc[37, "high"] = 90.5
df.loc[38, "high"] = 90.8
df.loc[39, "close"] = 91.5 # break minor high (90.8)

# Jauhkan close untuk semua candle dalam window
df.loc[29:38, "close"] = 95.0
res = evaluate_trend_following(df, 39, "BULLISH", pullback_proximity_atr=1.0)
for k, v in res.items():
    print(f"{k}: {v}")
