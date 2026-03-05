from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, Optional, Protocol

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
    _unrealized_captured: set = field(default_factory=set)  # keys captured for "first bar >= target"
    # Continuous_Col_* tracking (Entry/Exit/Max/Min/At30min/At60min) during backtest
    _continuous_entry: Dict[str, float] = field(default_factory=dict)
    _continuous_max: Dict[str, float] = field(default_factory=dict)
    _continuous_min: Dict[str, float] = field(default_factory=dict)
    _continuous_at_30: Dict[str, float] = field(default_factory=dict)
    _continuous_at_60: Dict[str, float] = field(default_factory=dict)
    _continuous_exit: Dict[str, float] = field(default_factory=dict)  # set at exit bar
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
    entry_column_snapshot: Optional[dict] = None  # Entry_Col_* from librarycolumn, captured at entry time when use_library_columns is True


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
    Col_MAE_R: float = 0.0  # Max Adverse Excursion (worst adverse move from entry, in R)
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
    Exit_Col_ATR14: float = 0.0
    Exit_Col_VWAP: float = 0.0


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
    analyzers: Dict[str, Any] = field(default_factory=dict)  # modular analyzer outputs: name -> result


class Analyzer(Protocol):
    """Base for all analyzers. Drop-in, zero core changes."""

    def analyze(self, result: RunResult) -> Any:
        """Return anything (dict, DF, float, object)."""
        ...


class SizerConfig(Protocol):
    """Minimal config needed by sizers; satisfied by BacktestConfig."""

    risk_pct_per_trade: Optional[float]
    fixed_risk_per_trade: Optional[float]
    float_col: str
    float_cap_pct: float
    equity_cap_pct: float
    absolute_cap_value: float


class Sizer(Protocol):
    """Pluggable position sizer (backtrader-style). Returns (shares, entry_value); engine applies caps."""

    def size(
        self,
        entry_price: float,
        row: pd.Series,
        account_balance: float,
        stop_price: Optional[float],
        side: str,
        config: SizerConfig,
    ) -> tuple[int, float]:
        """Return (shares, entry_value). Engine will apply float/equity/absolute caps."""
        ...


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
