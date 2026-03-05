"""I/O utilities for backtest outputs."""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .bt_types import RunResult

# Core trade fields only (no Col_*, Entry_*, Exit_*, Continuous_*). Main CSV gets only these when split_entry_exit=True.
_CORE_TRADE_COLUMNS = (
    "ticker", "date", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
    "shares", "hold_minutes", "gross_pnl", "commission", "sec_taf_fee", "slippage", "gst",
    "borrow_cost", "net_pnl", "account_balance_after",
)

# Engine-computed exit-only columns (written in _close_trade). Included in exit CSV so exit has more columns than entry.
_ENGINE_EXIT_ONLY_COLUMNS = (
    "Col_MaxFavorableExcursion_R", "Col_MAE_R", "Col_BarsToMFE", "Col_BarsToMAE",
    "Col_MaxDrawdownFromMFE_R", "Col_FinalPL_R", "Col_HoldMinutes", "Col_ExitHourNumeric",
    "Col_UnrealizedPL_1000", "Col_UnrealizedPL_1030", "Col_UnrealizedPL_1100", "Col_UnrealizedPL_1130",
    "Col_UnrealizedPL_1200", "Col_UnrealizedPL_1230", "Col_UnrealizedPL_1300", "Col_UnrealizedPL_1330",
    "Col_UnrealizedPL_1400", "Col_UnrealizedPL_1430", "Col_UnrealizedPL_1500", "Col_UnrealizedPL_1530",
    "Col_UnrealizedPL_1600",
    "Col_ExitVWAPDeviation_ATR", "Col_BarsSinceEntry", "Col_PosSize_PctAccount",
)


def get_engine_exit_only_columns() -> List[str]:
    """Return list of engine-computed exit-only column names (added in _close_trade, not from enriched long)."""
    return list(_ENGINE_EXIT_ONLY_COLUMNS)


def write_trades_csv(
    result: RunResult,
    path: Union[str, Path],
    *,
    enriched_long_df: Optional[pd.DataFrame] = None,
    split_entry_exit: bool = False,
) -> None:
    """Write backtest trades to a single CSV (core + entry + exit + continuous columns).

    Writes the full result.trades DataFrame to path so the file can be passed directly
    to Strategy Cruncher (one combined CSV with net_pnl and all Entry_Col_*, Exit_Col_*,
    Continuous_Col_*). Run Phase 2 (enrich_results) to get Entry/Exit/Continuous columns.
    enriched_long_df is accepted for API compatibility but ignored.
    split_entry_exit is ignored; a single combined CSV is always written (kept for API compatibility).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = result.trades if result.trades is not None else pd.DataFrame()
    to_write.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def write_trades_excel(
    result: RunResult,
    path: Union[str, Path],
    *,
    enriched_long_df: Optional[pd.DataFrame] = None,
) -> None:
    """Write backtest trades to Excel. Same column labelling as write_trades_csv.

    No fallback: only writes result.trades as-is. Run Phase 2 to get Continuous_Col_*.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = result.trades if result.trades is not None else pd.DataFrame()
    # No fallback: only write what is in result.trades.
    to_write.to_excel(path, index=False, engine="openpyxl")
