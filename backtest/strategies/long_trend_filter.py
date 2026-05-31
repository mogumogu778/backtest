import pandas as pd
from .base import Strategy


class LongTrendFilterStrategy(Strategy):
    """
    長期トレンド順張り + 中期押し目 + RSIフィルター（低リスク）

    買い条件（全て同時に満たす）:
        - 終値 > SMA_long（長期上昇トレンド確認）
        - 乖離率(SMA_mid)が dev_buy_min〜dev_buy_max（一時的な押し目）
        - RSI が 40〜50（過熱でも過売りでもない中立域）

    売り条件（いずれか1つ）:
        - 乖離率 >= dev_exit（利確目標到達）
        - RSI >= rsi_exit（買われすぎによる利確）
        - 終値 < SMA_long（トレンド崩壊による損切り）
    """

    STRATEGY_NAME = "長期トレンド順張り（低リスク）"
    PARAM_SCHEMA = [
        {"key": "sma_long",    "label": "長期SMA (日)",        "type": "int",   "default": 200,  "min": 150,  "max": 250},
        {"key": "sma_mid",     "label": "中期SMA (日)",        "type": "int",   "default": 50,   "min": 30,   "max": 75},
        {"key": "rsi_period",  "label": "RSI期間 (日)",        "type": "int",   "default": 14,   "min": 10,   "max": 21},
        {"key": "dev_buy_min", "label": "乖離率 買い下限 (%)", "type": "float", "default": -3.0, "min": -5.0, "max": -1.0, "step": 0.5},
        {"key": "dev_buy_max", "label": "乖離率 買い上限 (%)", "type": "float", "default": -1.0, "min": -3.0, "max":  0.0, "step": 0.5},
        {"key": "dev_exit",    "label": "乖離率 利確閾値 (%)", "type": "float", "default":  3.0, "min":  1.5, "max":  5.0, "step": 0.5},
        {"key": "rsi_exit",    "label": "RSI 利確閾値",        "type": "int",   "default": 70,   "min": 65,   "max": 80},
    ]

    def __init__(
        self,
        sma_long: int = 200,
        sma_mid: int = 50,
        rsi_period: int = 14,
        dev_buy_min: float = -3.0,
        dev_buy_max: float = -1.0,
        dev_exit: float = 3.0,
        rsi_exit: int = 70,
    ):
        self.sma_long = sma_long
        self.sma_mid = sma_mid
        self.rsi_period = rsi_period
        self.dev_buy_min = dev_buy_min
        self.dev_buy_max = dev_buy_max
        self.dev_exit = dev_exit
        self.rsi_exit = rsi_exit

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["sma_long"]  = out["Close"].rolling(self.sma_long).mean()
        out["sma_mid"]   = out["Close"].rolling(self.sma_mid).mean()
        out["deviation"] = (out["Close"] - out["sma_mid"]) / out["sma_mid"] * 100

        delta = out["Close"].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        out["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

        out["signal"] = 0

        sell = (
            (out["deviation"] >= self.dev_exit) |
            (out["rsi"] >= self.rsi_exit) |
            (out["Close"] < out["sma_long"])
        )

        # RSI買い範囲は 40〜50 固定（中立域での押し目を捉える）
        buy = (
            (out["Close"] > out["sma_long"]) &
            (out["deviation"] >= self.dev_buy_min) &
            (out["deviation"] <= self.dev_buy_max) &
            (out["rsi"] >= 40) &
            (out["rsi"] <= 50)
        )

        out.loc[sell, "signal"] = -1
        out.loc[buy & ~sell, "signal"] = 1

        return out
