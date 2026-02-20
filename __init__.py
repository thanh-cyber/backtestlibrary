"""Reusable backtesting backbone components.

This package intentionally excludes strategy logic. Strategies should be
implemented separately and plugged into the engine via the Strategy protocol.
"""

from .data import GapperDataLoader, LoaderConfig
from .engine import ChronologicalBacktestEngine, BacktestConfig
from .metrics import build_full_metrics
from .monte_carlo import run_monte_carlo
from .types import (
    EntryCandidate,
    ExitSignal,
    Position,
    RunResult,
    Strategy,
    TradeRecord,
)

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
]
