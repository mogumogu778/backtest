import pandas as pd
from .base import Strategy


class MACDStrategy(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    @property
    def name(self) -> str:
        return "MACD"

    @property
    def param_labels(self) -> dict:
        return {"fast": "短期EMA", "slow": "長期EMA", "signal_period": "シグナル期間"}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ema_fast = out["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = out["Close"].ewm(span=self.slow, adjust=False).mean()
        out["macd"] = ema_fast - ema_slow
        out["macd_signal"] = out["macd"].ewm(span=self.signal_period, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        out["signal"] = 0
        cross_up = (out["macd"] > out["macd_signal"]) & (
            out["macd"].shift(1) <= out["macd_signal"].shift(1)
        )
        cross_down = (out["macd"] < out["macd_signal"]) & (
            out["macd"].shift(1) >= out["macd_signal"].shift(1)
        )
        out.loc[cross_up, "signal"] = 1
        out.loc[cross_down, "signal"] = -1
        return out
