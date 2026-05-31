from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Base class for all strategies.

    Subclasses must set these class attributes:

        STRATEGY_NAME: str
            Display name shown in the UI.

        PARAM_SCHEMA: list[dict]
            Parameter definitions used to auto-generate sidebar sliders.
            Each entry is a dict with:
                key     : str            - matches the __init__ parameter name
                label   : str            - display label in the UI
                type    : "int"|"float"  - slider type
                default : int|float      - default value
                min     : int|float      - minimum value
                max     : int|float      - maximum value
                step    : int|float      - optional (default: 1 for int, 0.1 for float)

    Example
    -------
        STRATEGY_NAME = "マイ戦略"
        PARAM_SCHEMA = [
            {"key": "period", "label": "期間 (日)", "type": "int",
             "default": 14, "min": 5, "max": 60},
        ]

        def __init__(self, period: int = 14):
            self.period = period

        def generate_signals(self, df):
            ...
    """

    STRATEGY_NAME: str = ""
    PARAM_SCHEMA: list[dict] = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df copy with a 'signal' column: 1=buy, -1=sell, 0=hold."""
        pass

    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    @property
    def param_labels(self) -> dict:
        return {p["key"]: p["label"] for p in self.PARAM_SCHEMA}
