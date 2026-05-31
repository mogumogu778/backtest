import pandas as pd
from .base import Strategy


class EMATrendATRStrategy(Strategy):
    """
    EMAゴールデンクロス + RSIモメンタム + ATRボラティリティフィルター（中リスク）

    買い条件（全て同時に満たす）:
        - EMAゴールデンクロス（短期EMAが長期EMAを上抜け）
        - RSI が 50〜70（モメンタム健全、過熱前）
        - ATR比率が atr_min_pct〜atr_max_pct（適正ボラティリティ帯）
        - 終値 > EMA_long（トレンド方向との整合）

    売り条件（いずれか1つ）:
        - EMAデッドクロス（短期EMAが長期EMAを下抜け）
        - RSI > 70（過熱による利確）
        - ATR比率 > atr_max_pct（異常ボラティリティ回避）
    """

    STRATEGY_NAME = "EMAトレンド×ATRフィルター（中リスク）"
    PARAM_SCHEMA = [
        {"key": "ema_short",   "label": "短期EMA (日)",        "type": "int",   "default": 12,  "min": 5,   "max": 25},
        {"key": "ema_long",    "label": "長期EMA (日)",        "type": "int",   "default": 26,  "min": 20,  "max": 60},
        {"key": "rsi_period",  "label": "RSI期間 (日)",        "type": "int",   "default": 14,  "min": 7,   "max": 21},
        {"key": "atr_period",  "label": "ATR期間 (日)",        "type": "int",   "default": 14,  "min": 7,   "max": 21},
        {"key": "atr_min_pct", "label": "ATR比率 最小値 (%)", "type": "float", "default": 0.5, "min": 0.3, "max": 1.0, "step": 0.1},
        {"key": "atr_max_pct", "label": "ATR比率 最大値 (%)", "type": "float", "default": 4.0, "min": 3.0, "max": 6.0, "step": 0.5},
    ]

    def __init__(
        self,
        ema_short: int = 12,
        ema_long: int = 26,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_min_pct: float = 0.5,
        atr_max_pct: float = 4.0,
    ):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_min_pct = atr_min_pct
        self.atr_max_pct = atr_max_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["ema_short"] = out["Close"].ewm(span=self.ema_short, adjust=False).mean()
        out["ema_long"]  = out["Close"].ewm(span=self.ema_long,  adjust=False).mean()

        delta = out["Close"].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        out["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

        tr = pd.concat([
            out["High"] - out["Low"],
            (out["High"] - out["Close"].shift(1)).abs(),
            (out["Low"]  - out["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"]     = tr.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        out["atr_pct"] = out["atr"] / out["Close"] * 100

        cross_up   = (out["ema_short"] > out["ema_long"]) & (out["ema_short"].shift(1) <= out["ema_long"].shift(1))
        cross_down = (out["ema_short"] < out["ema_long"]) & (out["ema_short"].shift(1) >= out["ema_long"].shift(1))

        out["signal"] = 0

        sell = (
            cross_down |
            (out["rsi"] > 70) |
            (out["atr_pct"] > self.atr_max_pct)
        )

        buy = (
            cross_up &
            (out["rsi"] >= 50) & (out["rsi"] <= 70) &
            (out["atr_pct"] >= self.atr_min_pct) & (out["atr_pct"] <= self.atr_max_pct) &
            (out["Close"] > out["ema_long"])
        )

        out.loc[sell, "signal"] = -1
        out.loc[buy & ~sell, "signal"] = 1

        return out
