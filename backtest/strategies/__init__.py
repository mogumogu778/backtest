import pkgutil
import importlib
import inspect
from pathlib import Path

from .base import Strategy


def discover_strategies() -> dict[str, type[Strategy]]:
    """
    strategies/ フォルダ内の全 .py ファイルを自動スキャンし、
    Strategy のサブクラスを検出して返す。

    新しい戦略を追加するには:
        1. backtest/strategies/ に新しい .py ファイルを作成する
        2. Strategy を継承したクラスを定義し、
           STRATEGY_NAME と PARAM_SCHEMA を設定する
        3. generate_signals() を実装する
    ─── それだけで自動的にアプリに表示される ───

    Returns
    -------
    dict[str, type[Strategy]]
        STRATEGY_NAME → クラス のマッピング（検出順）
    """
    found: dict[str, type[Strategy]] = {}
    package_dir = Path(__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name == "base":
            continue
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(cls, Strategy)
                    and cls is not Strategy
                    and cls.STRATEGY_NAME
                    and cls.STRATEGY_NAME not in found
                ):
                    found[cls.STRATEGY_NAME] = cls
        except Exception:
            pass

    return found


# 後方互換のための直接インポート
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .bollinger import BollingerStrategy
from .macd import MACDStrategy
from .rsi_ma_filter import RSIMAFilterStrategy

__all__ = [
    "Strategy",
    "discover_strategies",
    "MACrossStrategy",
    "RSIStrategy",
    "BollingerStrategy",
    "MACDStrategy",
    "RSIMAFilterStrategy",
]
