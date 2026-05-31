from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Base class for all strategies.
    generate_signals() must return a DataFrame with a 'signal' column:
      1 = buy, -1 = sell, 0 = hold
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def param_labels(self) -> dict:
        """Human-readable labels for parameters used in display."""
        pass
