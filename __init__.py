"""Reusable backtesting backbone components.

This package intentionally excludes strategy logic. Strategies should be
implemented separately and plugged into the engine via the Strategy protocol.
"""

from .data import GapperDataLoader, LoaderConfig
from .engine import ChronologicalBacktestEngine, BacktestConfig
from .metrics import build_full_metrics
from .monte_carlo import run_monte_carlo
from .exit_metrics import (
    attach_exit_metrics_from_intraday_data,
    attach_exit_metrics_from_minute_bars,
    build_daily_exit_metrics,
    build_daily_exit_metrics_from_minute_bars,
    build_daily_exit_metrics_from_wide_minute_columns,
    merge_exit_metrics_into_backtest_data,
)
from .bt_types import (
    EntryCandidate,
    ExitSignal,
    Position,
    RunResult,
    Strategy,
    TradeRecord,
)
from .io import write_trades_csv

__all__ = [
    "BacktestConfig",
    "ChronologicalBacktestEngine",
    "EntryCandidate",
    "ExitSignal",
    "GapperDataLoader",
    "LoaderConfig",
    "Position",
    "RunResult",
    "Strategy",
    "TradeRecord",
    "build_full_metrics",
    "run_monte_carlo",
    "build_daily_exit_metrics_from_minute_bars",
    "build_daily_exit_metrics_from_wide_minute_columns",
    "build_daily_exit_metrics",
    "merge_exit_metrics_into_backtest_data",
    "attach_exit_metrics_from_minute_bars",
    "attach_exit_metrics_from_intraday_data",
    "write_trades_csv",
]
