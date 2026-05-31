import pandas as pd
from .base import Strategy


class BollingerStrategy(Strategy):
    STRATEGY_NAME = "ボリンジャーバンド"
    PARAM_SCHEMA = [
        {"key": "window",  "label": "期間 (日)",        "type": "int",   "default": 20,  "min": 5,   "max": 50},
        {"key": "num_std", "label": "標準偏差の倍数", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1},
    ]

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bb_mid"]   = out["Close"].rolling(self.window).mean()
        std             = out["Close"].rolling(self.window).std()
        out["bb_upper"] = out["bb_mid"] + self.num_std * std
        out["bb_lower"] = out["bb_mid"] - self.num_std * std

        out["signal"] = 0
        buy  = (out["Close"] < out["bb_lower"]) & (out["Close"].shift(1) >= out["bb_lower"].shift(1))
        sell = (out["Close"] > out["bb_upper"]) & (out["Close"].shift(1) <= out["bb_upper"].shift(1))
        out.loc[buy,  "signal"] = 1
        out.loc[sell, "signal"] = -1
        return out
