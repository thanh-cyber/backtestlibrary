"""
Column module: applies librarycolumn's entry / exit / continuous sections to the backtest.

When the engine is run with config.use_library_columns=True:
- Entry: metrics are captured at entry time (apply_entry_columns on the entry bar row) and stored on the position; at close they are merged into the trade as Entry_Col_*.
- Exit: metrics are captured at exit time (apply_exit_columns on the exit bar row) and added to the trade as Col_*_Exit.
- Continuous: run after the backtest via attach_continuous_tracking(result, enriched_long_df), or by passing enriched_long_df to write_trades_csv/write_trades_excel so Cont_Col_*_Entry/Exit/Max/Min/At30min/At60min are added.

When use_library_columns=False (default), the engine does not run entry/exit column logic.

Requires: pip install librarycolumn  (or backtestlibrary[librarycolumn])
If librarycolumn is not installed, get_entry_columns/get_exit_columns return minimal defaults and
attach_continuous_tracking raises a clear error.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .bt_types import RunResult

# Defaults when librarycolumn is not installed (minimal exit set for backward compat)
_DEFAULT_ENTRY_COLUMNS: List[str] = []
_DEFAULT_EXIT_COLUMNS: List[str] = ["Col_ATR14", "Col_VWAP"]
_DEFAULT_CONTINUOUS_COLUMNS: List[str] = []


def _lib() -> Optional[object]:
    """Import librarycolumn if available."""
    try:
        import column_library as lib
        return lib
    except ImportError:
        return None


def get_entry_columns() -> List[str]:
    """Return list of column names available at entry (from librarycolumn.ENTRY_COLUMNS)."""
    lib = _lib()
    if lib is not None and hasattr(lib, "ENTRY_COLUMNS"):
        return list(getattr(lib, "ENTRY_COLUMNS", _DEFAULT_ENTRY_COLUMNS))
    return list(_DEFAULT_ENTRY_COLUMNS)


def get_exit_columns() -> List[str]:
    """Return list of column names to snapshot at exit (from librarycolumn.EXIT_SNAPSHOT_COLUMNS)."""
    lib = _lib()
    if lib is not None and hasattr(lib, "EXIT_SNAPSHOT_COLUMNS"):
        return list(getattr(lib, "EXIT_SNAPSHOT_COLUMNS", _DEFAULT_EXIT_COLUMNS))
    return list(_DEFAULT_EXIT_COLUMNS)


def get_continuous_columns() -> List[str]:
    """Return list of columns to track during the trade (from librarycolumn.CONTINUOUS_TRACKING_COLUMNS)."""
    lib = _lib()
    if lib is not None and hasattr(lib, "CONTINUOUS_TRACKING_COLUMNS"):
        return list(getattr(lib, "CONTINUOUS_TRACKING_COLUMNS", _DEFAULT_CONTINUOUS_COLUMNS))
    return list(_DEFAULT_CONTINUOUS_COLUMNS)


# Column name prefixes/suffixes so Excel clearly shows Entry vs Exit vs Continuous
ENTRY_COLUMN_PREFIX = "Entry_"   # Entry_Col_ATR14 = entry snapshot
EXIT_COLUMN_SUFFIX = "_Exit"     # Col_ATR14_Exit = exit snapshot
CONTINUOUS_COLUMN_PREFIX = "Cont_"  # Cont_Col_RSI14_Entry, Cont_Col_RSI14_Max, etc.


def apply_entry_columns(trade_dict: dict, row: pd.Series) -> None:
    """
    Mutate trade_dict to add Entry_Col_X for each X in get_entry_columns() from row.
    Use the entry bar row (same row in wide format). Labels: Entry_Col_ATR14, etc.
    """
    for col in get_entry_columns():
        try:
            val = row.get(col)
        except Exception:
            continue
        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            continue
        try:
            v = float(val)
            if v != v or abs(v) == float("inf"):
                continue
        except (TypeError, ValueError):
            continue
        trade_dict[f"{ENTRY_COLUMN_PREFIX}{col}"] = v


def apply_exit_columns(trade_dict: dict, row: pd.Series) -> None:
    """
    Mutate trade_dict to add Col_X_Exit for each X in get_exit_columns() from row.
    Call this with the exit bar row before appending the trade to trades_list.
    """
    for col in get_exit_columns():
        try:
            val = row.get(col)
        except Exception:
            continue
        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            continue
        try:
            v = float(val)
            if v != v or abs(v) == float("inf"):  # NaN or inf
                continue
        except (TypeError, ValueError):
            continue
        trade_dict[f"{col}{EXIT_COLUMN_SUFFIX}"] = v


def _trade_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Build datetime from date (Timestamp) and time (str 'HH:MM')."""
    def one(d, t):
        try:
            if pd.isna(d) or pd.isna(t):
                return pd.NaT
            d = pd.Timestamp(d).normalize()
            parts = str(t).strip().split(":")
            h = int(parts[0]) if len(parts) > 0 else 0
            m = int(parts[1]) if len(parts) > 1 else 0
            return d + pd.Timedelta(hours=h, minutes=m)
        except Exception:
            return pd.NaT
    return pd.Series([one(d, t) for d, t in zip(date_series, time_series)], index=date_series.index)


def attach_continuous_tracking(
    result: RunResult,
    enriched_long_df: pd.DataFrame,
    *,
    entry_time_col: str = "EntryTime",
    exit_time_col: str = "ExitTime",
    ticker_col: str = "Ticker",
    datetime_col: str = "datetime",
    columns: Optional[List[str]] = None,
    at_minutes: Optional[List[int]] = None,
) -> RunResult:
    """
    Attach continuous intra-trade tracking (Entry/Exit/Max/Min/At30min/At60min) to result.trades
    using librarycolumn.add_continuous_tracking. enriched_long_df must be minute-level long format
    with Ticker, datetime and the Col_* columns.

    result.trades must have date, entry_time, exit_time, ticker (or already EntryTime, ExitTime, Ticker).
    Returns a new RunResult with result.trades replaced by the enriched trades DataFrame.
    """
    lib = _lib()
    if lib is None or not hasattr(lib, "add_continuous_tracking"):
        raise ImportError(
            "attach_continuous_tracking requires librarycolumn. Install with: pip install librarycolumn"
        )
    trades = result.trades
    if trades is None or trades.empty:
        return result
    tr = trades.copy()
    if "EntryTime" not in tr.columns and "date" in tr.columns and "entry_time" in tr.columns:
        tr["EntryTime"] = _trade_datetime(tr["date"], tr["entry_time"])
    if "ExitTime" not in tr.columns and "date" in tr.columns and "exit_time" in tr.columns:
        tr["ExitTime"] = _trade_datetime(tr["date"], tr["exit_time"])
    if "Ticker" not in tr.columns and "ticker" in tr.columns:
        tr["Ticker"] = tr["ticker"]
    enriched = getattr(lib, "add_continuous_tracking")(
        enriched_long_df,
        tr,
        entry_time_col=entry_time_col,
        exit_time_col=exit_time_col,
        ticker_col=ticker_col,
        datetime_col=datetime_col,
        columns=columns,
        at_minutes=at_minutes,
    )
    # Rename continuous columns so they're clearly labelled: Cont_Col_X_Entry, Cont_Col_X_Exit, etc.
    cont_cols = columns if columns is not None else get_continuous_columns()
    at_mins = at_minutes if at_minutes is not None else [30, 60]
    renames = {}
    for col in cont_cols:
        if col not in enriched_long_df.columns:
            continue
        for suffix in ["_Entry", "_Exit", "_Max", "_Min"]:
            old_name = f"{col}{suffix}"
            if old_name in enriched.columns:
                renames[old_name] = f"{CONTINUOUS_COLUMN_PREFIX}{col}{suffix}"
        for m in at_mins:
            old_name = f"{col}_At{m}min"
            if old_name in enriched.columns:
                renames[old_name] = f"{CONTINUOUS_COLUMN_PREFIX}{col}_At{m}min"
    if renames:
        enriched = enriched.rename(columns=renames)
    return RunResult(
        final_balance=result.final_balance,
        total_return_pct=result.total_return_pct,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        total_pnl=result.total_pnl,
        trades=enriched,
        daily_equity=result.daily_equity,
        analyzers=result.analyzers,
    )


def has_librarycolumn() -> bool:
    """Return True if librarycolumn is installed and usable."""
    return _lib() is not None
