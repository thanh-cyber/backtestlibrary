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
    # Runtime tracking for elite exit analytics
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_to_mfe: int = 0
    bars_to_mae: int = 0
    max_dd_from_mfe: float = 0.0
    peak_pl_r: float = 0.0
    bars_since_entry: int = 0
    starting_account: float = 0.0
    unrealized_pl_1000: float = 0.0
    unrealized_pl_1030: float = 0.0
    unrealized_pl_1100: float = 0.0
    unrealized_pl_1130: float = 0.0
    unrealized_pl_1200: float = 0.0
    unrealized_pl_1230: float = 0.0
    unrealized_pl_1300: float = 0.0
    unrealized_pl_1330: float = 0.0
    unrealized_pl_1400: float = 0.0
    unrealized_pl_1430: float = 0.0
    unrealized_pl_1500: float = 0.0
    unrealized_pl_1530: float = 0.0
    unrealized_pl_1600: float = 0.0


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
    # ====================== FINAL ELITE EXIT COLUMNS ======================
    # Core MFE/MAE Family
    Col_MaxFavorableExcursion_R: float = 0.0
    Col_DistToInitialStop_R: float = 0.0
    Col_BarsToMFE: int = 0
    Col_BarsToMAE: int = 0
    Col_MaxDrawdownFromMFE_R: float = 0.0
    Col_FinalPL_R: float = 0.0
    Col_HoldMinutes: int = 0
    Col_ExitHourNumeric: float = 0.0
    # 30-minute block unrealized P&L snapshots
    Col_UnrealizedPL_1000: float = 0.0
    Col_UnrealizedPL_1030: float = 0.0
    Col_UnrealizedPL_1100: float = 0.0
    Col_UnrealizedPL_1130: float = 0.0
    Col_UnrealizedPL_1200: float = 0.0
    Col_UnrealizedPL_1230: float = 0.0
    Col_UnrealizedPL_1300: float = 0.0
    Col_UnrealizedPL_1330: float = 0.0
    Col_UnrealizedPL_1400: float = 0.0
    Col_UnrealizedPL_1430: float = 0.0
    Col_UnrealizedPL_1500: float = 0.0
    Col_UnrealizedPL_1530: float = 0.0
    Col_UnrealizedPL_1600: float = 0.0
    # Bonus high-value columns
    Col_ExitVWAPDeviation_ATR: float = 0.0
    Col_BarsSinceEntry: int = 0
    Col_PosSize_PctAccount: float = 0.0
    # Metrics at exit bar (for exit columns / downstream use)
    Col_ATR14_Exit: float = 0.0
    Col_VWAP_Exit: float = 0.0


@dataclass
class RunResult:
    final_balance: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    trades: pd.DataFrame
    daily_equity: list[float] = field(default_factory=list)  # [start_bal, end_day1, end_day2, ...] for calendar metrics


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
