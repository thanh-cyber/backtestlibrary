"""I/O utilities for backtest outputs."""

from pathlib import Path
from typing import Union

from .bt_types import RunResult


def write_trades_csv(result: RunResult, path: Union[str, Path]) -> None:
    """Write backtest trades to CSV, including all exit columns.

    Args:
        result: RunResult from ChronologicalBacktestEngine (trades attribute
                includes core fields plus Col_MaxFavorableExcursion_R,
                Col_DistToInitialStop_R, Col_ATR14_Exit, Col_VWAP_Exit, etc.).
        path: Output file path (e.g. "trades.csv" or Path("backtest_results/trades.csv")).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
