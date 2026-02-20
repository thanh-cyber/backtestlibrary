from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Optional, Protocol

import pandas as pd


@dataclass
class EntryCandidate:
    ticker: str
    row_index: Any
    entry_time: time
    entry_price: float
    side: str  # "long" or "short"
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitSignal:
    exit_price: float
    reason: str


@dataclass
class Position:
    ticker: str
    row_index: Any
    side: str
    shares: int
    entry_price: float
    entry_time: time
    stop_price: Optional[float]
    target_price: Optional[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    ticker: str
    date: pd.Timestamp
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    shares: int
    hold_minutes: int
    gross_pnl: float
    commission: float
    sec_taf_fee: float
    slippage: float
    gst: float
    borrow_cost: float
    net_pnl: float
    account_balance_after: float


@dataclass
class RunResult:
    final_balance: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    trades: pd.DataFrame


class Strategy(Protocol):
    """Plug-in strategy interface for the backbone engine."""

    def find_entries_for_day(
        self,
        day_df: pd.DataFrame,
        timeline: list[time],
        get_price,
    ) -> list[EntryCandidate]:
        """Return candidate entries for this day (any side, any session)."""

    def check_exit(
        self,
        position: Position,
        row: pd.Series,
        current_time: time,
        get_price,
    ) -> Optional[ExitSignal]:
        """Return ExitSignal when position should be closed, else None."""
