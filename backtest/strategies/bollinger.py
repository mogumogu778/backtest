import pandas as pd
from .base import Strategy


class BollingerStrategy(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    @property
    def name(self) -> str:
        return "ボリンジャーバンド"

    @property
    def param_labels(self) -> dict:
        return {"window": "期間", "num_std": "標準偏差の倍数"}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bb_mid"] = out["Close"].rolling(self.window).mean()
        std = out["Close"].rolling(self.window).std()
        out["bb_upper"] = out["bb_mid"] + self.num_std * std
        out["bb_lower"] = out["bb_mid"] - self.num_std * std

        out["signal"] = 0
        buy = (out["Close"] < out["bb_lower"]) & (
            out["Close"].shift(1) >= out["bb_lower"].shift(1)
        )
        sell = (out["Close"] > out["bb_upper"]) & (
            out["Close"].shift(1) <= out["bb_upper"].shift(1)
        )
        out.loc[buy, "signal"] = 1
        out.loc[sell, "signal"] = -1
        return out
