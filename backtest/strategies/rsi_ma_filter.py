import pandas as pd
from .base import Strategy


class RSIMAFilterStrategy(Strategy):
    """
    RSI + 移動平均フィルター複合戦略

    買い条件（両方を同時に満たした翌日に購入）:
        - RSI が売られすぎ閾値を下抜けた
        - 終値が長期MA より上（上昇トレンド確認）

    売り条件（どちらか一方で売却）:
        - RSI が買われすぎ閾値を上抜けた
        - 終値が長期MA を下抜けた（トレンド崩壊）
    """

    STRATEGY_NAME = "RSI+MAフィルター"
    PARAM_SCHEMA = [
        {"key": "rsi_period",   "label": "RSI期間 (日)",    "type": "int", "default": 14,  "min": 5,  "max": 30},
        {"key": "oversold",     "label": "売られすぎ閾値",  "type": "int", "default": 30,  "min": 10, "max": 45},
        {"key": "overbought",   "label": "買われすぎ閾値",  "type": "int", "default": 70,  "min": 55, "max": 90},
        {"key": "ma_window",    "label": "長期MA期間 (日)", "type": "int", "default": 75,  "min": 20, "max": 200},
    ]

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        ma_window: int = 75,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.ma_window = ma_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # RSI（Wilder EMA法）
        delta = out["Close"].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        out["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

        # 長期移動平均
        out["long_ma"] = out["Close"].rolling(self.ma_window).mean()

        out["signal"] = 0

        # 買い: RSIが売られすぎ閾値を下抜け AND 終値が長期MA上（トレンド確認）
        rsi_oversold = (out["rsi"] < self.oversold) & (out["rsi"].shift(1) >= self.oversold)
        above_ma     = out["Close"] > out["long_ma"]
        out.loc[rsi_oversold & above_ma, "signal"] = 1

        # 売り: RSIが買われすぎ閾値を上抜け OR 終値が長期MAを下抜け
        rsi_overbought = (out["rsi"] > self.overbought) & (out["rsi"].shift(1) <= self.overbought)
        below_ma       = (out["Close"] < out["long_ma"]) & (out["Close"].shift(1) >= out["long_ma"].shift(1))
        out.loc[rsi_overbought | below_ma, "signal"] = -1

        return out
