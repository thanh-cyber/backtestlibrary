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

# Engine-computed path/exit analytics (written in _close_trade; export names are Exit_Col_*).
_ENGINE_EXIT_ONLY_COLUMNS = (
    "Exit_Col_MaxFavorableExcursion_R", "Exit_Col_MAE_R", "Exit_Col_BarsToMFE", "Exit_Col_BarsToMAE",
    "Exit_Col_MaxDrawdownFromMFE_R", "Exit_Col_FinalPL_R", "Exit_Col_HoldMinutes", "Exit_Col_ExitHourNumeric",
    "Exit_Col_UnrealizedPL_1000", "Exit_Col_UnrealizedPL_1030", "Exit_Col_UnrealizedPL_1100", "Exit_Col_UnrealizedPL_1130",
    "Exit_Col_UnrealizedPL_1200", "Exit_Col_UnrealizedPL_1230", "Exit_Col_UnrealizedPL_1300", "Exit_Col_UnrealizedPL_1330",
    "Exit_Col_UnrealizedPL_1400", "Exit_Col_UnrealizedPL_1430", "Exit_Col_UnrealizedPL_1500", "Exit_Col_UnrealizedPL_1530",
    "Exit_Col_UnrealizedPL_1600",
    "Exit_Col_ExitVWAPDeviation_ATR", "Exit_Col_BarsSinceEntry", "Exit_Col_PosSize_PctAccount",
)


def get_engine_exit_only_columns() -> List[str]:
    """Return engine-computed Exit_Col_* names (path metrics + exit VWAP/ATR fields from _close_trade)."""
    return list(_ENGINE_EXIT_ONLY_COLUMNS)


def ordered_trades_export_columns(columns: list) -> List[str]:
    """
    Column order for exports: all non-snapshot columns first (original order), then Entry_Col_*,
    Exit_Col_*, Continuous_Col_* (each group sorted alphabetically).
    """
    cols = list(columns)
    entry = sorted(c for c in cols if str(c).startswith("Entry_"))
    exit_c = sorted(c for c in cols if str(c).startswith("Exit_"))
    cont = sorted(c for c in cols if str(c).startswith("Continuous_"))
    tagged = set(entry) | set(exit_c) | set(cont)
    rest = [c for c in cols if c not in tagged]
    return rest + entry + exit_c + cont


def reorder_trades_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns for Excel/CSV export (see ``ordered_trades_export_columns``)."""
    if df is None or getattr(df, "empty", True):
        return df
    order = ordered_trades_export_columns(list(df.columns))
    return df[order].copy()


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
