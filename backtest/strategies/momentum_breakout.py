import pandas as pd
from .base import Strategy


class MomentumBreakoutStrategy(Strategy):
    """
    急騰モメンタム・ブレイクアウト戦略（ハイリスク）

    RSIを「売られすぎ/買われすぎ」ではなく「モメンタム加速」として順張りに使う。
    4条件が同時に揃ったときのみエントリーするが、各閾値は積極的な設定。

    買い条件（全て同時に満たす）:
        - 終値 > 前日までの短期チャネル高値（ブレイクアウト）
        - 短期RSI > rsi_buy_threshold（モメンタム加速確認）
        - ATR > 過去20日ATR平均 × 1.2（ボラティリティ拡大）
        - 出来高 > 過去20日平均 × volume_multiplier（大口参入確認）

    売り条件（いずれか1つ）:
        - 終値 < 前日までの短期チャネル安値（ブレイクダウン）
        - 短期RSI < rsi_sell_threshold かつ RSI低下中（モメンタム消失）
    """

    STRATEGY_NAME = "急騰ブレイクアウト（ハイリスク）"
    PARAM_SCHEMA = [
        {"key": "rsi_period",         "label": "RSI期間 (日)",      "type": "int",   "default": 6,   "min": 3,   "max": 14},
        {"key": "rsi_buy_threshold",  "label": "RSI 買い閾値",      "type": "int",   "default": 70,  "min": 60,  "max": 80},
        {"key": "rsi_sell_threshold", "label": "RSI 売り閾値",      "type": "int",   "default": 50,  "min": 40,  "max": 60},
        {"key": "channel_period",     "label": "チャネル期間 (日)", "type": "int",   "default": 5,   "min": 3,   "max": 10},
        {"key": "atr_period",         "label": "ATR期間 (日)",      "type": "int",   "default": 10,  "min": 5,   "max": 20},
        {"key": "volume_multiplier",  "label": "出来高急増倍率",    "type": "float", "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1},
    ]

    def __init__(
        self,
        rsi_period: int = 6,
        rsi_buy_threshold: int = 70,
        rsi_sell_threshold: int = 50,
        channel_period: int = 5,
        atr_period: int = 10,
        volume_multiplier: float = 1.5,
    ):
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.volume_multiplier = volume_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 短期RSI（モメンタム加速検出：逆張りではなく順張り用途）
        delta = out["Close"].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        out["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))

        # ATR（ボラティリティ拡大判定）
        tr = pd.concat([
            out["High"] - out["Low"],
            (out["High"] - out["Close"].shift(1)).abs(),
            (out["Low"]  - out["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        atr_avg    = out["atr"].rolling(20).mean()

        # 価格チャネル（当日を含まない＝ルックアヘッドバイアス回避）
        out["channel_high"] = out["High"].shift(1).rolling(self.channel_period).max()
        out["channel_low"]  = out["Low"].shift(1).rolling(self.channel_period).min()

        # 出来高フィルター
        vol_avg = out["Volume"].rolling(20).mean()

        out["signal"] = 0

        rsi_falling  = out["rsi"] < out["rsi"].shift(1)
        atr_expanding = out["atr"] > atr_avg * 1.2
        volume_surge  = out["Volume"] > vol_avg * self.volume_multiplier

        sell = (
            (out["Close"] < out["channel_low"]) |
            ((out["rsi"] < self.rsi_sell_threshold) & rsi_falling)
        )

        buy = (
            (out["Close"] > out["channel_high"]) &
            (out["rsi"] > self.rsi_buy_threshold) &
            atr_expanding &
            volume_surge
        )

        out.loc[sell, "signal"] = -1
        out.loc[buy & ~sell, "signal"] = 1

        return out
