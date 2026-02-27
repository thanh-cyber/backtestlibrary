"""
Post-backtest trade enrichment: add Entry_Col_*, Exit_Col_*, and Continuous_Col_* to trades
after the backtest run, by loading wide data per (ticker, date), converting to long, enriching,
and applying entry/exit/continuous column logic.

Used when the engine runs backtest first without enriched_long; then this module enriches
trades in a single post-pass. Supports both in-memory DataFrames and Path (streaming) data sources.
"""
from __future__ import annotations

import re
from datetime import time
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .bt_types import RunResult
from .librarycolumn_enrichment import (
    enrich_long_with_library_columns,
    get_row_at_time,
    wide_to_long,
)
from .columns import apply_entry_columns, apply_exit_columns, attach_continuous_tracking


def _enrich_long_per_ticker_date(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run librarycolumn enrichment per (Ticker, Date) group, since add_all_columns
    expects DatetimeIndex (single time series per call).
    """
    if long_df.empty:
        return long_df
    try:
        import column_library as lib
    except ImportError:
        try:
            from backtestlibrary.librarycolumn import column_library as lib
        except ImportError:
            return long_df
    if not hasattr(lib, "add_all_columns"):
        return long_df
    date_col = "Date" if "Date" in long_df.columns else "date"
    ticker_col = "Ticker" if "Ticker" in long_df.columns else "ticker"
    if date_col not in long_df.columns or ticker_col not in long_df.columns:
        return long_df
    parts = []
    for (ticker, dt), grp in long_df.groupby([ticker_col, date_col]):
        g = grp.copy()
        if "datetime" not in g.columns:
            continue
        g = g.set_index("datetime")
        if not isinstance(g.index, pd.DatetimeIndex):
            g.index = pd.to_datetime(g.index, errors="coerce")
        try:
            out = lib.add_all_columns(g, inplace=False)
        except Exception:
            parts.append(grp)
            continue
        out = out.reset_index()
        parts.append(out)
    if not parts:
        return long_df
    return pd.concat(parts, ignore_index=True)


# Parse "H:MM", "HH:MM", "H:MM:SS", "HH:MM:SS" -> time
_TIME_STR_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def _parse_time_str(s: Any) -> Optional[time]:
    """Parse entry_time/exit_time from string 'H:MM' or 'HH:MM' to datetime.time."""
    if s is None:
        return None
    if isinstance(s, time):
        return s
    s = str(s).strip()
    m = _TIME_STR_RE.match(s)
    if m:
        h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if 0 <= h <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            return time(h, mm, ss)
    return None


def _load_wide_for_dates(
    year: str,
    dates: list[pd.Timestamp],
    tickers: set[str],
    cleaned_year_data: dict,
    wide_path_by_year: dict,
) -> Optional[pd.DataFrame]:
    """
    Load wide data for the given year and dates.
    - If cleaned_year_data[year] is DataFrame: filter to dates and tickers.
    - If Path: read_parquet with filters for dates.
    """
    data = cleaned_year_data.get(year)
    if data is None:
        data = cleaned_year_data.get(str(year))
    path = wide_path_by_year.get(year)
    if path is None:
        path = wide_path_by_year.get(str(year))
    if isinstance(data, pd.DataFrame):
        if data.empty or "Date" not in data.columns:
            return None
        df = data.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        date_set = set(pd.Timestamp(d).normalize() for d in dates)
        df = df[df["Date"].isin(date_set)]
        ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
        if ticker_col in df.columns and tickers:
            df = df[df[ticker_col].astype(str).str.upper().isin({t.upper() for t in tickers})]
        return df if not df.empty else None
    # Path / str
    src_path = path if path is not None else (data if isinstance(data, (Path, str)) else None)
    if src_path is None:
        return None
    path_str = str(Path(src_path).resolve())
    d_min = min(dates)
    d_max = max(dates)
    try:
        df = pd.read_parquet(
            path_str,
            filters=[("Date", ">=", d_min), ("Date", "<=", d_max)],
        )
    except Exception:
        df = pd.read_parquet(path_str)
        df = df[
            (pd.to_datetime(df["Date"], errors="coerce").dt.normalize() >= pd.Timestamp(d_min))
            & (pd.to_datetime(df["Date"], errors="coerce").dt.normalize() <= pd.Timestamp(d_max))
        ]
    if df.empty or "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
    if ticker_col in df.columns and tickers:
        df = df[df[ticker_col].astype(str).str.upper().isin({t.upper() for t in tickers})]
    return df if not df.empty else None


def _filter_long_to_session(
    long_df: pd.DataFrame,
    session_start: time,
    session_end_time: time,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Filter long DataFrame to bars from session_start through session_end_time."""
    if long_df.empty or datetime_col not in long_df.columns:
        return long_df
    dt = pd.to_datetime(long_df[datetime_col])
    start_minutes = session_start.hour * 60 + session_start.minute
    end_minutes = session_end_time.hour * 60 + session_end_time.minute
    bar_minutes = dt.dt.hour * 60 + dt.dt.minute
    mask = (bar_minutes >= start_minutes) & (bar_minutes <= end_minutes)
    return long_df.loc[mask].copy()


def enrich_trades_post_backtest(
    result: RunResult,
    cleaned_year_data: dict,
    wide_path_by_year: dict,
    session_start: time,
    session_end: time,
    config: Any,
) -> RunResult:
    """
    Enrich trades with Entry_Col_*, Exit_Col_*, and Continuous_Col_* after the backtest run.

    For each unique (year, ticker, date) in trades:
    - Load wide data (from DataFrame or Path via pd.read_parquet with filters)
    - Filter to bars from session_start through max exit_time for that (ticker, date)
    - wide_to_long, enrich_long_with_library_columns
    For each trade:
    - get_row_at_time for entry_time -> apply_entry_columns
    - get_row_at_time for exit_time -> apply_exit_columns
    Then attach_continuous_tracking for Continuous_Col_*.

    Handles: empty trades, missing columns, entry_time/exit_time as "H:MM" or "HH:MM".
    """
    if not getattr(config, "use_library_columns", True):
        return result
    trades = result.trades
    if trades is None or trades.empty:
        return result

    # Normalize column names (trade dicts may use date/entry_time/exit_time/ticker)
    tr = trades.copy()
    date_col = "date" if "date" in tr.columns else "Date"
    entry_time_col = "entry_time" if "entry_time" in tr.columns else "EntryTime"
    exit_time_col = "exit_time" if "exit_time" in tr.columns else "ExitTime"
    ticker_col = "ticker" if "ticker" in tr.columns else "Ticker"
    if date_col not in tr.columns or entry_time_col not in tr.columns or exit_time_col not in tr.columns:
        return result
    tr["date"] = pd.to_datetime(tr[date_col], errors="coerce").dt.normalize()

    # Collect unique (year, ticker, date) and max exit_time per (ticker, date)
    year_from_date = tr["date"].dt.year.astype(str)
    tr["_year"] = year_from_date
    unique_keys: list[tuple[str, str, pd.Timestamp]] = []
    seen = set()
    max_exit_by_key: dict[tuple[str, pd.Timestamp], time] = {}
    for _, row in tr.iterrows():
        y = str(row["_year"])
        t = str(row[ticker_col]).strip()
        d = row["date"]
        if pd.isna(d):
            continue
        key = (y, t, d)
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
        exit_t = _parse_time_str(row.get(exit_time_col))
        if exit_t is not None:
            k = (t, d)
            cur = max_exit_by_key.get(k)
            if cur is None or (exit_t.hour * 60 + exit_t.minute) > (cur.hour * 60 + cur.minute):
                max_exit_by_key[k] = exit_t

    # Group by year for loading
    by_year: dict[str, list[tuple[str, pd.Timestamp]]] = {}
    for y, t, d in unique_keys:
        by_year.setdefault(y, []).append((t, d))

    # Load wide -> long -> enrich per year
    enriched_parts: list[pd.DataFrame] = []
    for year, ticker_dates in by_year.items():
        dates = sorted(set(d for _, d in ticker_dates))
        tickers = {t for t, _ in ticker_dates}
        wide_df = _load_wide_for_dates(year, dates, tickers, cleaned_year_data, wide_path_by_year)
        if wide_df is None or wide_df.empty:
            continue
        long_df = wide_to_long(wide_df, ticker_col="Ticker", date_col="Date")
        if long_df.empty:
            continue
        # column_library expects capitalized OHLCV; some versions also expect DatetimeIndex
        rename_map = {c: c.capitalize() for c in ["open", "high", "low", "close", "volume"] if c in long_df.columns}
        if rename_map:
            long_df = long_df.rename(columns=rename_map)
        # Filter to session_start through max exit_time for this year's ticker-dates
        ticker_date_set = set(ticker_dates)
        max_exit = None
        for (t, d), et in max_exit_by_key.items():
            if (t, d) in ticker_date_set:
                if max_exit is None or (et.hour * 60 + et.minute) > (max_exit.hour * 60 + max_exit.minute):
                    max_exit = et
        end_time = max_exit if max_exit is not None else session_end
        long_df = _filter_long_to_session(long_df, session_start, end_time)
        if long_df.empty:
            continue
        try:
            enriched = enrich_long_with_library_columns(long_df)
        except (ValueError, TypeError):
            try:
                enriched = _enrich_long_per_ticker_date(long_df)
            except Exception:
                enriched = long_df
        if not enriched.empty:
            enriched_parts.append(enriched)

    if not enriched_parts:
        return result

    enriched_long = pd.concat(enriched_parts, ignore_index=True)

    # Apply entry/exit columns to each trade
    trades_list = tr.to_dict("records")
    for tdict in trades_list:
        ticker = str(tdict.get(ticker_col, tdict.get("ticker", ""))).strip()
        date_val = tdict.get("date", tdict.get("Date"))
        if pd.isna(date_val):
            continue
        date_ts = pd.Timestamp(date_val).normalize()
        entry_t = _parse_time_str(tdict.get(entry_time_col, tdict.get("entry_time")))
        exit_t = _parse_time_str(tdict.get(exit_time_col, tdict.get("exit_time")))
        if entry_t is not None:
            row_entry = get_row_at_time(enriched_long, ticker, date_ts, entry_t)
            if row_entry is not None:
                apply_entry_columns(tdict, row_entry)
        if exit_t is not None:
            row_exit = get_row_at_time(enriched_long, ticker, date_ts, exit_t)
            if row_exit is not None:
                apply_exit_columns(tdict, row_exit)

    enriched_trades_df = pd.DataFrame(trades_list)
    # Drop internal columns
    if "_year" in enriched_trades_df.columns:
        enriched_trades_df = enriched_trades_df.drop(columns=["_year"])

    # Attach Continuous_Col_*
    try:
        result = attach_continuous_tracking(
            RunResult(
                final_balance=result.final_balance,
                total_return_pct=result.total_return_pct,
                total_trades=result.total_trades,
                winning_trades=result.winning_trades,
                losing_trades=result.losing_trades,
                total_pnl=result.total_pnl,
                trades=enriched_trades_df,
                daily_equity=result.daily_equity,
                analyzers=result.analyzers,
            ),
            enriched_long,
        )
    except Exception as e:
        import warnings
        warnings.warn(
            f"attach_continuous_tracking failed: {e}. Continuous_Col_* columns will be missing.",
            UserWarning,
            stacklevel=1,
        )
        result = RunResult(
            final_balance=result.final_balance,
            total_return_pct=result.total_return_pct,
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            losing_trades=result.losing_trades,
            total_pnl=result.total_pnl,
            trades=enriched_trades_df,
            daily_equity=result.daily_equity,
            analyzers=result.analyzers,
        )
    return result
