import pandas as pd
from .base import Strategy


class MACrossStrategy(Strategy):
    STRATEGY_NAME = "移動平均クロス"
    PARAM_SCHEMA = [
        {"key": "short_window", "label": "短期MA (日)", "type": "int", "default": 25, "min": 5,  "max": 50},
        {"key": "long_window",  "label": "長期MA (日)", "type": "int", "default": 75, "min": 20, "max": 200},
    ]

    def __init__(self, short_window: int = 25, long_window: int = 75):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["short_ma"] = out["Close"].rolling(self.short_window).mean()
        out["long_ma"]  = out["Close"].rolling(self.long_window).mean()

        out["signal"] = 0
        cross_up   = (out["short_ma"] > out["long_ma"]) & (out["short_ma"].shift(1) <= out["long_ma"].shift(1))
        cross_down = (out["short_ma"] < out["long_ma"]) & (out["short_ma"].shift(1) >= out["long_ma"].shift(1))
        out.loc[cross_up,   "signal"] = 1
        out.loc[cross_down, "signal"] = -1
        return out
