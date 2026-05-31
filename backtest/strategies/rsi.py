import pandas as pd
from .base import Strategy


class RSIStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return "RSI"

    @property
    def param_labels(self) -> dict:
        return {"period": "期間", "oversold": "売られすぎ閾値", "overbought": "買われすぎ閾値"}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        delta = out["Close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        out["rsi"] = 100 - (100 / (1 + rs))

        out["signal"] = 0
        buy = (out["rsi"] < self.oversold) & (out["rsi"].shift(1) >= self.oversold)
        sell = (out["rsi"] > self.overbought) & (out["rsi"].shift(1) <= self.overbought)
        out.loc[buy, "signal"] = 1
        out.loc[sell, "signal"] = -1
        return out
