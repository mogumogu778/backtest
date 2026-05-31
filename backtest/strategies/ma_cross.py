import pandas as pd
from .base import Strategy


class MACrossStrategy(Strategy):
    def __init__(self, short_window: int = 25, long_window: int = 75):
        self.short_window = short_window
        self.long_window = long_window

    @property
    def name(self) -> str:
        return "移動平均クロス"

    @property
    def param_labels(self) -> dict:
        return {"short_window": "短期MA", "long_window": "長期MA"}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["short_ma"] = out["Close"].rolling(self.short_window).mean()
        out["long_ma"] = out["Close"].rolling(self.long_window).mean()

        out["signal"] = 0
        cross_up = (out["short_ma"] > out["long_ma"]) & (
            out["short_ma"].shift(1) <= out["long_ma"].shift(1)
        )
        cross_down = (out["short_ma"] < out["long_ma"]) & (
            out["short_ma"].shift(1) >= out["long_ma"].shift(1)
        )
        out.loc[cross_up, "signal"] = 1
        out.loc[cross_down, "signal"] = -1
        return out
