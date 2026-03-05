"""
Column module: applies entry / exit / continuous column lists to the backtest.

When the engine is run with config.use_library_columns=True:
- Entry: metrics are captured at entry time (apply_entry_columns on the entry bar row) and stored on the position; at close they are merged into the trade as Entry_Col_*.
- Exit: metrics are captured at exit time (apply_exit_columns on the exit bar row) and added to the trade as Exit_Col_*.
- Continuous: filled during the backtest when enriched_long_df is provided (Entry/Exit/Max/Min/At30min/At60min per get_continuous_columns()). If the engine was run without enriched_long_df, you can add them later by passing enriched_long_df to write_trades_csv/write_trades_excel, which calls attach_continuous_tracking.

Entry/exit/continuous column lists live in backtestlibrary.column_definitions (ENTRY_COLUMNS, EXIT_SNAPSHOT_COLUMNS, CONTINUOUS_TRACKING_COLUMNS). A project may override by defining those names in its column_library module.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .bt_types import RunResult
from . import column_definitions as _builtin


def _lib() -> Optional[object]:
    """Use librarycolumn package when available (pip install librarycolumn); else any column_library with ENTRY/EXIT/CONTINUOUS; else None (use built-in)."""
    try:
        import column_library as lib
    except ImportError:
        return None
    # Full librarycolumn: add_continuous_tracking + 20 continuous columns
    if hasattr(lib, "add_continuous_tracking") and hasattr(lib, "CONTINUOUS_TRACKING_COLUMNS"):
        cont = getattr(lib, "CONTINUOUS_TRACKING_COLUMNS", [])
        if len(cont) >= 10:
            return lib
    # Project override: has entry/exit or continuous defined
    if hasattr(lib, "ENTRY_COLUMNS") or hasattr(lib, "EXIT_SNAPSHOT_COLUMNS") or hasattr(lib, "CONTINUOUS_TRACKING_COLUMNS"):
        return lib
    return None


def get_entry_columns() -> List[str]:
    """Return list of column names to snapshot at entry. Uses column_library.ENTRY_COLUMNS if defined and non-empty, else backtestlibrary.column_definitions.ENTRY_COLUMNS."""
    lib = _lib()
    if lib is not None and hasattr(lib, "ENTRY_COLUMNS"):
        entry = list(getattr(lib, "ENTRY_COLUMNS"))
        if entry:
            return entry
    return list(_builtin.ENTRY_COLUMNS)


def get_exit_columns() -> List[str]:
    """Return list of column names to snapshot at exit. Uses column_library.EXIT_SNAPSHOT_COLUMNS if defined and non-empty, else backtestlibrary.column_definitions.EXIT_SNAPSHOT_COLUMNS."""
    lib = _lib()
    if lib is not None and hasattr(lib, "EXIT_SNAPSHOT_COLUMNS"):
        exit_cols = list(getattr(lib, "EXIT_SNAPSHOT_COLUMNS"))
        if exit_cols:
            return exit_cols
    return list(_builtin.EXIT_SNAPSHOT_COLUMNS)


def get_continuous_columns() -> List[str]:
    """Return list of columns to track during the trade. Uses column_library.CONTINUOUS_TRACKING_COLUMNS if defined, else backtestlibrary.column_definitions.CONTINUOUS_TRACKING_COLUMNS. May be empty if project overrides with []."""
    lib = _lib()
    if lib is not None and hasattr(lib, "CONTINUOUS_TRACKING_COLUMNS"):
        return list(getattr(lib, "CONTINUOUS_TRACKING_COLUMNS"))
    return list(_builtin.CONTINUOUS_TRACKING_COLUMNS)


# Column name prefixes (congruent: Entry_Col_X, Exit_Col_X, Continuous_Col_X_*)
ENTRY_COLUMN_PREFIX = "Entry_"       # Entry_Col_ATR14 = entry snapshot
EXIT_COLUMN_PREFIX = "Exit_"         # Exit_Col_ATR14 = exit snapshot
CONTINUOUS_COLUMN_PREFIX = "Continuous_"  # Continuous_Col_RSI14_Entry, Continuous_Col_RSI14_Max, etc.


def apply_entry_columns(trade_dict: dict, row: pd.Series) -> None:
    """
    Mutate trade_dict to add Entry_Col_X for each X in get_entry_columns() from row.
    Use the entry bar row (same row in wide format). Labels: Entry_Col_ATR14, etc.
    Always adds every expected column so output has a full set; uses float('nan') when value is missing/invalid.
    """
    for col in get_entry_columns():
        key = f"{ENTRY_COLUMN_PREFIX}{col}"
        val = row.get(col)
        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            trade_dict[key] = float("nan")
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            trade_dict[key] = float("nan")
            continue
        if v != v or abs(v) == float("inf"):
            trade_dict[key] = float("nan")
        else:
            trade_dict[key] = v


def apply_exit_columns(trade_dict: dict, row: pd.Series) -> None:
    """
    Mutate trade_dict to add Exit_Col_X for each X in get_exit_columns() from row.
    Call this with the exit bar row before appending the trade to trades_list.
    Always adds every expected column so output has a full set; uses float('nan') when value is missing/invalid.
    """
    for col in get_exit_columns():
        key = f"{EXIT_COLUMN_PREFIX}{col}"
        val = row.get(col)
        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            trade_dict[key] = float("nan")
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            trade_dict[key] = float("nan")
            continue
        if v != v or abs(v) == float("inf"):
            trade_dict[key] = float("nan")
        else:
            trade_dict[key] = v


def _trade_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Build datetime from date (Timestamp) and time (str 'HH:MM'). No fallback; raises on invalid input."""
    def one(d, t):
        if pd.isna(d) or pd.isna(t):
            return pd.NaT
        d = pd.Timestamp(d).normalize()
        parts = str(t).strip().split(":")
        h = int(parts[0]) if len(parts) > 0 else 0
        m = int(parts[1]) if len(parts) > 1 else 0
        return d + pd.Timedelta(hours=h, minutes=m)
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
    # Normalize date column: TradeRecord uses "date"; support "Date" from other sources
    if "date" not in tr.columns and "Date" in tr.columns:
        tr["date"] = tr["Date"]
    if "EntryTime" not in tr.columns and "date" in tr.columns and "entry_time" in tr.columns:
        tr["EntryTime"] = _trade_datetime(tr["date"], tr["entry_time"])
    if "ExitTime" not in tr.columns and "date" in tr.columns and "exit_time" in tr.columns:
        tr["ExitTime"] = _trade_datetime(tr["date"], tr["exit_time"])
    if "Ticker" not in tr.columns and "ticker" in tr.columns:
        tr["Ticker"] = tr["ticker"]
    # Normalize ticker to match enriched_long (wide_to_long uppercases tickers)
    tr[ticker_col] = tr[ticker_col].astype(str).str.upper().str.strip()
    # Ensure tz-naive datetimes so library key lookups match (avoid tz-aware vs naive mismatch)
    def _to_naive(ser: pd.Series) -> pd.Series:
        ser = pd.to_datetime(ser)
        if getattr(ser.dt, "tz", None) is None:
            return ser
        return ser.apply(lambda x: x.replace(tzinfo=None) if getattr(x, "tzinfo", None) else x)
    if entry_time_col in tr.columns or exit_time_col in tr.columns:
        tr = tr.copy()
        if entry_time_col in tr.columns:
            tr[entry_time_col] = _to_naive(tr[entry_time_col])
        if exit_time_col in tr.columns:
            tr[exit_time_col] = _to_naive(tr[exit_time_col])
    el_df = enriched_long_df.copy()
    if datetime_col in el_df.columns:
        el_df[datetime_col] = _to_naive(el_df[datetime_col])
    enriched = getattr(lib, "add_continuous_tracking")(
        el_df,
        tr,
        entry_time_col=entry_time_col,
        exit_time_col=exit_time_col,
        ticker_col=ticker_col,
        datetime_col=datetime_col,
        columns=columns,
        at_minutes=at_minutes,
    )
    # Rename continuous columns: Continuous_Col_X_Entry, Continuous_Col_X_Exit, etc.
    # Use library output (enriched) so we rename whenever the library added a column, even if
    # enriched_long_df had a different column set (e.g. chunk subset).
    cont_cols = columns if columns is not None else get_continuous_columns()
    at_mins = at_minutes if at_minutes is not None else [30, 60]
    renames = {}
    for col in cont_cols:
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
