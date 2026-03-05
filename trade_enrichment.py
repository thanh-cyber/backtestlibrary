"""
Post-backtest trade enrichment: add Entry_Col_*, Exit_Col_*, and Continuous_Col_* to trades
after the backtest run, by loading wide data per (ticker, date), converting to long, enriching,
and applying entry/exit/continuous column logic.

Used when the engine runs backtest first without enriched_long; then this module enriches
trades in a single post-pass. Supports both in-memory DataFrames and Path (streaming) data sources.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from datetime import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .bt_types import RunResult
from .librarycolumn_enrichment import (
    enrich_long_with_library_columns,
    get_row_at_time,
    wide_to_long,
)
from .columns import apply_entry_columns, apply_exit_columns, attach_continuous_tracking

# Unrealized P&L snapshot times (must match engine)
_UNREALIZED_SNAPSHOT_TARGETS: list[tuple[int, int, str]] = [
    (10, 0, "1000"), (10, 30, "1030"), (11, 0, "1100"), (11, 30, "1130"),
    (12, 0, "1200"), (12, 30, "1230"), (13, 0, "1300"), (13, 30, "1330"),
    (14, 0, "1400"), (14, 30, "1430"), (15, 0, "1500"), (15, 30, "1530"),
    (16, 0, "1600"),
]


def _pl_r_for_side(side: str, entry_price: float, current_price: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if str(side).lower() == "short":
        return (entry_price - current_price) / atr
    return (current_price - entry_price) / atr


def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and (x != x or abs(x) == float("inf"))):
        return None
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _get_bars_slice(
    enriched_long: pd.DataFrame,
    ticker: str,
    date_ts: pd.Timestamp,
    entry_t: time,
    exit_t: time,
    *,
    ticker_col: str = "Ticker",
    datetime_col: str = "datetime",
) -> Optional[pd.DataFrame]:
    """Return rows for (ticker, date) from entry_t to exit_t inclusive, sorted by datetime."""
    if enriched_long.empty or datetime_col not in enriched_long.columns:
        return None
    df = enriched_long
    dt = pd.to_datetime(df[datetime_col])
    date_norm = pd.Timestamp(date_ts).normalize()
    ticker_upper = str(ticker).strip().upper()
    mask_td = (
        (df[ticker_col].astype(str).str.strip().str.upper() == ticker_upper)
        & (dt.dt.normalize() == date_norm)
    )
    subset = df.loc[mask_td].copy()
    if subset.empty:
        return None
    subset["_time"] = pd.to_datetime(subset[datetime_col]).dt.time
    entry_min = entry_t.hour * 60 + entry_t.minute
    exit_min = exit_t.hour * 60 + exit_t.minute

    def in_range(t: time) -> bool:
        m = t.hour * 60 + t.minute
        return entry_min <= m <= exit_min

    subset = subset[subset["_time"].apply(in_range)].drop(columns=["_time"], errors="ignore")
    if subset.empty:
        return None
    subset = subset.sort_values(datetime_col).reset_index(drop=True)
    return subset


# Column names used on enriched_long from wide_to_long + librarycolumn (for index key consistency)
_EL_TICKER_COL = "Ticker"
_EL_DATE_COL = "Date"
_EL_DATETIME_COL = "datetime"


def _build_enriched_long_index(
    enriched_long: pd.DataFrame,
    *,
    ticker_col: str = _EL_TICKER_COL,
    date_col: str = _EL_DATE_COL,
    datetime_col: str = _EL_DATETIME_COL,
) -> dict[tuple[str, pd.Timestamp], pd.DataFrame]:
    """
    Build (ticker_upper, date_normalized) -> DataFrame index for O(1) lookups.
    Key format matches lookup calls: ticker stripped/uppered, date normalized.
    Returns empty dict if required columns are missing (caller falls back to full-scan).
    """
    if enriched_long.empty or datetime_col not in enriched_long.columns:
        return {}
    df = enriched_long
    if ticker_col not in df.columns:
        return {}
    # Use date column for key when present; otherwise normalize datetime
    if date_col in df.columns:
        date_key = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    else:
        date_key = pd.to_datetime(df[datetime_col], errors="coerce").dt.normalize()
    ticker_key = df[ticker_col].astype(str).str.strip().str.upper()
    grouped = df.groupby([ticker_key, date_key], sort=False)
    # Normalize keys to (str, pd.Timestamp) so lookups match (groupby can yield numpy datetime64)
    result: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    for (t, d), grp in grouped:
        dn = pd.Timestamp(d).normalize()
        if pd.isna(dn):
            continue
        result[(str(t), dn)] = grp
    return result


def _get_row_at_time_indexed(
    index: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    ticker: str,
    date_ts: pd.Timestamp,
    t: time,
    *,
    datetime_col: str = _EL_DATETIME_COL,
) -> Optional[pd.Series]:
    """Return row for (ticker, date, time) using prebuilt index. Same semantics as get_row_at_time."""
    key = (str(ticker).strip().upper(), pd.Timestamp(date_ts).normalize())
    df = index.get(key)
    if df is None or df.empty or datetime_col not in df.columns:
        return None
    dt = pd.to_datetime(df[datetime_col])
    mask = (dt.dt.hour == t.hour) & (dt.dt.minute == t.minute)
    subset = df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _get_bars_slice_indexed(
    index: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    ticker: str,
    date_ts: pd.Timestamp,
    entry_t: time,
    exit_t: time,
    *,
    datetime_col: str = _EL_DATETIME_COL,
) -> Optional[pd.DataFrame]:
    """Return rows for (ticker, date) from entry_t to exit_t inclusive using index. Same semantics as _get_bars_slice."""
    key = (str(ticker).strip().upper(), pd.Timestamp(date_ts).normalize())
    df = index.get(key)
    if df is None or df.empty or datetime_col not in df.columns:
        return None
    subset = df.copy()
    subset["_time"] = pd.to_datetime(subset[datetime_col]).dt.time
    entry_min = entry_t.hour * 60 + entry_t.minute
    exit_min = exit_t.hour * 60 + exit_t.minute

    def in_range(t: time) -> bool:
        m = t.hour * 60 + t.minute
        return entry_min <= m <= exit_min

    subset = subset[subset["_time"].apply(in_range)].drop(columns=["_time"], errors="ignore")
    if subset.empty:
        return None
    subset = subset.sort_values(datetime_col).reset_index(drop=True)
    return subset


def _apply_elite_exit_from_slice(
    tdict: dict,
    slice_df: pd.DataFrame,
    entry_price: float,
    exit_price: float,
    net_pnl: float,
    side: str,
    *,
    close_col: str = "Close",
    atr_col: str = "Col_ATR14",
    datetime_col: str = "datetime",
) -> None:
    """
    Compute MFE/MAE/unrealized/Col_FinalPL_R/Col_ExitVWAPDeviation_ATR from bar slice and set on tdict.
    Matches engine logic so these columns are populated even when engine had no enriched_long.
    """
    if slice_df.empty or atr_col not in slice_df.columns:
        return
    close_col_use = close_col if close_col in slice_df.columns else ("close" if "close" in slice_df.columns else None)
    if close_col_use is None:
        return
    side = str(side).lower() if side else "long"
    closes = slice_df[close_col_use]
    atrs = slice_df[atr_col]
    mfe_r = 0.0
    mae_r = 0.0
    peak_pl_r = 0.0
    max_dd_from_mfe = 0.0
    bars_to_mfe = 0
    bars_to_mae = 0
    unrealized: dict[str, float] = {k: 0.0 for (_, _, k) in _UNREALIZED_SNAPSHOT_TARGETS}
    unrealized_captured: set[str] = set()

    for bar_idx, (_, row) in enumerate(slice_df.iterrows()):
        close = _to_float(row.get(close_col_use))
        atr = _to_float(row.get(atr_col))
        if close is None or atr is None or atr <= 0:
            continue
        pl_r = _pl_r_for_side(side, entry_price, close, atr)
        if pl_r > mfe_r:
            mfe_r = pl_r
            bars_to_mfe = bar_idx
        if pl_r < mae_r:
            mae_r = pl_r
            bars_to_mae = bar_idx
        peak_pl_r = max(peak_pl_r, pl_r)
        max_dd_from_mfe = min(max_dd_from_mfe, pl_r - peak_pl_r)

        dt_val = row.get(datetime_col)
        if dt_val is not None:
            try:
                ts = pd.Timestamp(dt_val)
                current_min = ts.hour * 60 + ts.minute
            except Exception:
                current_min = -1
        else:
            current_min = -1
        for h, m, key in _UNREALIZED_SNAPSHOT_TARGETS:
            target_min = h * 60 + m
            if current_min >= target_min and key not in unrealized_captured:
                unrealized[key] = pl_r
                unrealized_captured.add(key)

    tdict["Col_MaxFavorableExcursion_R"] = float(mfe_r)
    tdict["Col_MAE_R"] = float(mae_r)
    tdict["Col_BarsToMFE"] = int(bars_to_mfe)
    tdict["Col_BarsToMAE"] = int(bars_to_mae)
    tdict["Col_MaxDrawdownFromMFE_R"] = float(max_dd_from_mfe)
    for key, val in unrealized.items():
        tdict[f"Col_UnrealizedPL_{key}"] = float(val)

    # Col_FinalPL_R and Col_ExitVWAPDeviation_ATR from last bar (exit bar)
    last = slice_df.iloc[-1]
    atr_exit = _to_float(last.get(atr_col))
    if atr_exit is not None and atr_exit > 0:
        entry_value = abs(tdict.get("shares", 0) * entry_price)
        if entry_value > 0:
            risk_dollar_unit = entry_value / atr_exit
            if risk_dollar_unit > 0:
                tdict["Col_FinalPL_R"] = float(net_pnl / risk_dollar_unit)
        vwap = _to_float(last.get("Col_VWAP"))
        if vwap is None and "Volume" in slice_df.columns and close_col_use:
            o = slice_df.get("Open", slice_df.get("open", pd.Series(dtype=float)))
            h = slice_df.get("High", slice_df.get("high", pd.Series(dtype=float)))
            l_ = slice_df.get("Low", slice_df.get("low", pd.Series(dtype=float)))
            c = slice_df[close_col_use]
            if len(o) and len(h) and len(l_) and len(c):
                typical = (o.astype(float) + h.astype(float) + l_.astype(float) + c.astype(float)) / 4.0
                vol = slice_df["Volume"].astype(float)
                cum_pv = (typical * vol).cumsum()
                cum_v = vol.cumsum()
                if cum_v.iloc[-1] > 0:
                    vwap = float(cum_pv.iloc[-1] / cum_v.iloc[-1])
        if vwap is not None and atr_exit > 0:
            if side == "short":
                tdict["Col_ExitVWAPDeviation_ATR"] = float((vwap - exit_price) / atr_exit)
            else:
                tdict["Col_ExitVWAPDeviation_ATR"] = float((exit_price - vwap) / atr_exit)


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
        g = g.sort_index()  # VWAP/pandas_ta require ordered DatetimeIndex
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


def _wide_columns_for_enrichment(schema_names: list[str]) -> list[str]:
    """Return column names needed for wide_to_long (exclude ATR14/VWAP to save memory)."""
    keep = []
    for c in schema_names:
        s = str(c).strip()
        if s.startswith("ATR14 ") or s.startswith("VWAP "):
            continue
        keep.append(c)
    return keep if keep else schema_names


def _load_wide_for_dates(
    year: str,
    dates: list[pd.Timestamp],
    tickers: set[str],
    cleaned_year_data: dict,
    wide_path_by_year: dict,
    *,
    columns: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load wide data for the given year and dates.
    - If cleaned_year_data[year] is DataFrame: filter to dates and tickers.
    - If Path: read_parquet with filters for dates; optionally only columns (to reduce memory).
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
    read_kw: dict = {"filters": [("Date", ">=", d_min), ("Date", "<=", d_max)], "engine": "pyarrow"}
    if columns is not None:
        read_kw["columns"] = columns
    df = pd.read_parquet(path_str, **read_kw)
    if df is None or df.empty or "Date" not in df.columns:
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


def _get_enrich_columns_for_year_static(
    yr: str,
    cleaned_year_data: dict,
    wide_path_by_year: dict,
    *,
    load_full_columns: bool = False,
) -> Optional[list[str]]:
    """Return column list for parquet read. If load_full_columns=False, exclude ATR14/VWAP to save memory."""
    if load_full_columns:
        return None  # None = load all columns
    data = cleaned_year_data.get(yr) or cleaned_year_data.get(str(yr))
    path = wide_path_by_year.get(yr) or wide_path_by_year.get(str(yr))
    if path is None and isinstance(data, (Path, str)):
        path = data
    if path is None:
        return None
    import pyarrow.parquet as pq
    schema = pq.read_schema(str(Path(path).resolve()))
    return _wide_columns_for_enrichment(schema.names)


def _build_enriched_long_for_chunk(
    year: str,
    chunk_dates: list,
    tickers: set,
    end_time: time,
    session_start: time,
    session_end: time,
    cleaned_year_data: dict,
    wide_path_by_year: dict,
    load_full_columns: bool,
    *,
    cache_path_obj: Optional[Path] = None,
    base_cache_key: Optional[str] = None,
    chunk_start_idx: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Build enriched long for one date chunk only. Optionally write to cache as .../base_cache_key_chunk{i}.parquet.
    Returns enriched DataFrame or None.
    """
    if not chunk_dates:
        return None
    cols = None
    if isinstance(cleaned_year_data.get(year), (Path, str)) or wide_path_by_year.get(year):
        cols = _get_enrich_columns_for_year_static(
            year, cleaned_year_data, wide_path_by_year, load_full_columns=load_full_columns
        )
    wide_df = _load_wide_for_dates(
        year, chunk_dates, tickers, cleaned_year_data, wide_path_by_year, columns=cols
    )
    if wide_df is None or wide_df.empty:
        return None
    long_df = wide_to_long(wide_df, ticker_col="Ticker", date_col="Date")
    del wide_df
    if long_df.empty:
        return None
    rename_map = {c: c.capitalize() for c in ["open", "high", "low", "close", "volume"] if c in long_df.columns}
    if rename_map:
        long_df = long_df.rename(columns=rename_map)
    long_df = _filter_long_to_session(long_df, session_start, end_time)
    if long_df.empty:
        return None
    if "datetime" in long_df.columns:
        long_df = long_df.copy()
        dt = pd.to_datetime(long_df["datetime"])
        long_df["datetime"] = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
        enriched = enrich_long_with_library_columns(long_df)
    del long_df
    if enriched.empty:
        return None
    if cache_path_obj is not None and base_cache_key is not None:
        cache_path_obj.mkdir(parents=True, exist_ok=True)
        chunk_file = cache_path_obj / f"{base_cache_key}_chunk{chunk_start_idx}.parquet"
        enriched.to_parquet(chunk_file, index=False, engine="pyarrow")
    return enriched


def _build_enriched_long_for_year(args: tuple) -> tuple[str, list[pd.DataFrame]]:
    """
    Build enriched long for one year (for parallel execution).
    args: (year, ticker_dates, end_time, session_start, session_end, cache_dir, chunk_days,
          cleaned_year_data, wide_path_by_year, load_full_columns)
    Returns (year, list of DataFrames for that year).
    """
    (year, ticker_dates, end_time, session_start, session_end, cache_dir, chunk_days,
     cleaned_year_data, wide_path_by_year, load_full_columns) = args
    _cache_dir = cache_dir if (cache_dir and isinstance(cache_dir, (str, Path)) and str(cache_dir).strip()) else None
    cache_path_obj = Path(_cache_dir).resolve() if _cache_dir else None
    dates = sorted(set(d for _, d in ticker_dates))
    tickers = {t for t, _ in ticker_dates}
    col_suffix = "full" if load_full_columns else "restr"
    cache_key = f"{year}_{hashlib.md5(str(dates).encode()).hexdigest()[:16]}_{col_suffix}.parquet"
    cache_file = cache_path_obj / cache_key if cache_path_obj is not None else None
    if cache_file is not None and cache_file.is_file():
        cached = pd.read_parquet(cache_file, engine="pyarrow")
        if not cached.empty:
            return (year, [cached])
    year_parts: list[pd.DataFrame] = []
    cols = None
    if isinstance(cleaned_year_data.get(year), (Path, str)) or wide_path_by_year.get(year):
        cols = _get_enrich_columns_for_year_static(
            year, cleaned_year_data, wide_path_by_year, load_full_columns=load_full_columns
        )
    for chunk_start in range(0, len(dates), chunk_days):
        chunk_dates = dates[chunk_start : chunk_start + chunk_days]
        if not chunk_dates:
            continue
        wide_df = _load_wide_for_dates(
            year, chunk_dates, tickers, cleaned_year_data, wide_path_by_year, columns=cols
        )
        if wide_df is None or wide_df.empty:
            continue
        long_df = wide_to_long(wide_df, ticker_col="Ticker", date_col="Date")
        del wide_df
        if long_df.empty:
            continue
        rename_map = {c: c.capitalize() for c in ["open", "high", "low", "close", "volume"] if c in long_df.columns}
        if rename_map:
            long_df = long_df.rename(columns=rename_map)
        long_df = _filter_long_to_session(long_df, session_start, end_time)
        if long_df.empty:
            continue
        # Force datetime to tz-naive so librarycolumn never sees tz-naive vs tz-aware comparison
        if "datetime" in long_df.columns:
            long_df = long_df.copy()
            dt = pd.to_datetime(long_df["datetime"])
            long_df["datetime"] = dt.dt.tz_localize(None) if dt.dt.tz is not None else dt
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
            enriched = enrich_long_with_library_columns(long_df)
        del long_df
        if not enriched.empty:
            year_parts.append(enriched)
    if year_parts and cache_file is not None and cache_path_obj is not None:
        cache_path_obj.mkdir(parents=True, exist_ok=True)
        pd.concat(year_parts, ignore_index=True).to_parquet(cache_file, index=False, engine="pyarrow")
    return (year, year_parts)


def enrich_trades_post_backtest(
    result: RunResult,
    cleaned_year_data: dict,
    wide_path_by_year: dict,
    session_start: time,
    session_end: time,
    config: Any,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    load_full_columns: bool = False,
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

    If cache_dir is set, enriched long DataFrames are saved per year (keyed by year + hash of
    date set) and reused on subsequent runs to skip wide_to_long + librarycolumn for that year.
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

    # Process year-by-year, and within each year by date chunk (never hold full year enriched long in memory)
    CHUNK_DAYS = 21 if load_full_columns else 50
    _cache_dir = cache_dir if (cache_dir and isinstance(cache_dir, (str, Path)) and str(cache_dir).strip()) else None
    cache_path_obj = Path(_cache_dir).resolve() if _cache_dir else None
    if cache_path_obj is not None:
        cache_path_obj.mkdir(parents=True, exist_ok=True)
    tqdm = None
    if show_progress:
        from tqdm.auto import tqdm

    tr["_orig_idx"] = np.arange(len(tr))
    pbar = tqdm(total=100, desc="Phase 2", unit="%") if (tqdm and show_progress) else None
    per_year_enriched: list[pd.DataFrame] = []
    col_suffix = "full" if load_full_columns else "restr"

    try:
        year_keys = sorted(by_year.keys())
        for year_idx, year in enumerate(year_keys):
            ticker_dates = by_year[year]
            dates = sorted(set(d for _, d in ticker_dates))
            tickers = {t for t, _ in ticker_dates}
            ticker_date_set = set(ticker_dates)
            max_exit = None
            for (t, d), et in max_exit_by_key.items():
                if (t, d) in ticker_date_set:
                    if max_exit is None or (et.hour * 60 + et.minute) > (max_exit.hour * 60 + max_exit.minute):
                        max_exit = et
            end_time = max_exit if max_exit is not None else session_end

            base_cache_key = f"{year}_{hashlib.md5(str(dates).encode()).hexdigest()[:16]}_{col_suffix}"
            cache_file = cache_path_obj / f"{base_cache_key}.parquet" if cache_path_obj else None
            full_year_cached = cache_file is not None and cache_file.is_file()

            tr_year = tr[tr["_year"] == year]
            if tr_year.empty:
                if pbar is not None:
                    pbar.update(int(100 * (year_idx + 1) / len(year_keys)))
                continue

            year_chunk_results: list[pd.DataFrame] = []

            for chunk_start in range(0, len(dates), CHUNK_DAYS):
                chunk_dates = dates[chunk_start : chunk_start + CHUNK_DAYS]
                if not chunk_dates:
                    continue
                chunk_dates_set = set(pd.Timestamp(d).normalize() for d in chunk_dates)
                tr_chunk = tr_year[tr_year["date"].dt.normalize().isin(chunk_dates_set)]
                if tr_chunk.empty:
                    continue

                # Load enriched long for this chunk only (from full-year cache with filter, chunk cache file, or build)
                enriched_long = None
                if full_year_cached:
                    d_min, d_max = min(chunk_dates), max(chunk_dates)
                    enriched_long = pd.read_parquet(
                        cache_file,
                        filters=[("Date", ">=", d_min), ("Date", "<=", d_max)],
                        engine="pyarrow",
                    )
                elif cache_path_obj is not None:
                    chunk_file = cache_path_obj / f"{base_cache_key}_chunk{chunk_start}.parquet"
                    if chunk_file.is_file():
                        enriched_long = pd.read_parquet(chunk_file, engine="pyarrow")
                if enriched_long is None or enriched_long.empty:
                    enriched_long = _build_enriched_long_for_chunk(
                        year,
                        chunk_dates,
                        tickers,
                        end_time,
                        session_start,
                        session_end,
                        cleaned_year_data,
                        wide_path_by_year,
                        load_full_columns,
                        cache_path_obj=cache_path_obj,
                        base_cache_key=base_cache_key,
                        chunk_start_idx=chunk_start,
                    )

                if enriched_long is None or enriched_long.empty:
                    unchunk = tr_chunk.drop(columns=["_year"], errors="ignore")
                    if "_orig_idx" in tr_chunk.columns:
                        unchunk["_orig_idx"] = tr_chunk["_orig_idx"].values
                    year_chunk_results.append(unchunk)
                    continue

                el_index = _build_enriched_long_index(enriched_long)
                use_index = len(el_index) > 0

                trades_list = tr_chunk.to_dict("records")
                for tdict in trades_list:
                    ticker = str(tdict.get(ticker_col, tdict.get("ticker", ""))).strip()
                    date_val = tdict.get("date", tdict.get("Date"))
                    if pd.isna(date_val):
                        continue
                    date_ts = pd.Timestamp(date_val).normalize()
                    entry_t = _parse_time_str(tdict.get(entry_time_col, tdict.get("entry_time")))
                    exit_t = _parse_time_str(tdict.get(exit_time_col, tdict.get("exit_time")))
                    if entry_t is not None:
                        row_entry = (
                            _get_row_at_time_indexed(el_index, ticker, date_ts, entry_t)
                            if use_index
                            else get_row_at_time(enriched_long, ticker, date_ts, entry_t)
                        )
                        if row_entry is not None:
                            apply_entry_columns(tdict, row_entry)
                    if exit_t is not None:
                        row_exit = (
                            _get_row_at_time_indexed(el_index, ticker, date_ts, exit_t)
                            if use_index
                            else get_row_at_time(enriched_long, ticker, date_ts, exit_t)
                        )
                        if row_exit is not None:
                            apply_exit_columns(tdict, row_exit)
                    if entry_t is not None and exit_t is not None:
                        slice_df = (
                            _get_bars_slice_indexed(el_index, ticker, date_ts, entry_t, exit_t)
                            if use_index
                            else _get_bars_slice(enriched_long, ticker, date_ts, entry_t, exit_t)
                        )
                        if slice_df is not None and not slice_df.empty:
                            entry_price = tdict.get("entry_price")
                            exit_price = tdict.get("exit_price")
                            net_pnl = tdict.get("net_pnl", 0.0)
                            side = "long" if (tdict.get("shares") or 0) > 0 else "short"
                            if entry_price is not None and exit_price is not None:
                                _apply_elite_exit_from_slice(
                                    tdict, slice_df, float(entry_price), float(exit_price),
                                    float(net_pnl), side,
                                )

                enriched_trades_chunk = pd.DataFrame(trades_list)

                # Continuous_Col_* for this date-chunk's trades
                CONTINUOUS_TRACKING_CHUNK_SIZE = 20
                chunk_trades_list = []
                n_trades = len(tr_chunk)
                for start in range(0, n_trades, CONTINUOUS_TRACKING_CHUNK_SIZE):
                    end = min(start + CONTINUOUS_TRACKING_CHUNK_SIZE, n_trades)
                    tr_cc = tr_chunk.iloc[start:end]
                    trades_cc = enriched_trades_chunk.iloc[start:end]
                    slices_cc = []
                    for _, row in tr_cc.iterrows():
                        ticker = str(row.get(ticker_col, row.get("ticker", ""))).strip()
                        date_val = row.get("date", row.get("Date"))
                        if pd.isna(date_val):
                            continue
                        date_ts = pd.Timestamp(date_val).normalize()
                        entry_t = _parse_time_str(row.get(entry_time_col, row.get("entry_time")))
                        exit_t = _parse_time_str(row.get(exit_time_col, row.get("exit_time")))
                        if entry_t is None or exit_t is None:
                            continue
                        sl = (
                            _get_bars_slice_indexed(el_index, ticker, date_ts, entry_t, exit_t)
                            if use_index
                            else _get_bars_slice(enriched_long, ticker, date_ts, entry_t, exit_t)
                        )
                        if sl is not None and not sl.empty:
                            slices_cc.append(sl)
                    el_cc = pd.concat(slices_cc, ignore_index=True) if slices_cc else pd.DataFrame()
                    if not el_cc.empty and "datetime" in el_cc.columns:
                        dt = pd.to_datetime(el_cc["datetime"])
                        if getattr(dt.dt, "tz", None) is not None:
                            el_cc = el_cc.copy()
                            el_cc["datetime"] = dt.apply(
                                lambda x: x.replace(tzinfo=None) if getattr(x, "tzinfo", None) else x
                            )
                    if not el_cc.empty:
                        cr = attach_continuous_tracking(
                            RunResult(
                                final_balance=result.final_balance,
                                total_return_pct=result.total_return_pct,
                                total_trades=result.total_trades,
                                winning_trades=result.winning_trades,
                                losing_trades=result.losing_trades,
                                total_pnl=result.total_pnl,
                                trades=trades_cc,
                                daily_equity=result.daily_equity,
                                analyzers=result.analyzers,
                            ),
                            el_cc,
                        )
                        chunk_trades_list.append(cr.trades)
                    else:
                        chunk_trades_list.append(trades_cc)

                chunk_result_df = pd.concat(chunk_trades_list, ignore_index=True) if chunk_trades_list else enriched_trades_chunk
                if "_orig_idx" in tr_chunk.columns:
                    chunk_result_df = chunk_result_df.copy()
                    chunk_result_df["_orig_idx"] = tr_chunk["_orig_idx"].values
                year_chunk_results.append(chunk_result_df)
                del enriched_long, el_index

            year_enriched = pd.concat(year_chunk_results, ignore_index=True) if year_chunk_results else tr_year.drop(columns=["_year"], errors="ignore")
            if "_year" in year_enriched.columns:
                year_enriched = year_enriched.drop(columns=["_year"], errors="ignore")
            # _orig_idx already set per chunk; ensure it exists for un-enriched fallback
            if "_orig_idx" not in year_enriched.columns and "_orig_idx" in tr_year.columns:
                year_enriched["_orig_idx"] = tr_year["_orig_idx"].values
            per_year_enriched.append(year_enriched)

            if pbar is not None:
                pbar.update(int(100 * (year_idx + 1) / len(year_keys)))

        # Reassemble in original trade order
        if not per_year_enriched:
            return result
        enriched_trades_with_continuous = pd.concat(per_year_enriched, ignore_index=True)
        if "_orig_idx" in enriched_trades_with_continuous.columns:
            enriched_trades_with_continuous = enriched_trades_with_continuous.sort_values("_orig_idx").drop(columns=["_orig_idx"])
        result = RunResult(
            final_balance=result.final_balance,
            total_return_pct=result.total_return_pct,
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            losing_trades=result.losing_trades,
            total_pnl=result.total_pnl,
            trades=enriched_trades_with_continuous,
            daily_equity=result.daily_equity,
            analyzers=result.analyzers,
        )
        return result
    finally:
        if pbar is not None:
            pbar.close()


def enrich_results(
    raw_results: dict,
    cleaned_year_data: dict,
    config: Any,
    *,
    session_start: Optional[time] = None,
    session_end: Optional[time] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_full_columns: bool = False,
) -> None:
    """
    Run column phase (Entry_Col_*, Exit_Col_*, Continuous_Col_*) on raw_results in place.
    Use after engine.run(..., defer_column_phase=True).
    If cache_dir is set, enriched long DataFrames are cached there per year for reuse.
    If load_full_columns=True, load all wide columns (including ATR14/VWAP) so enrichment
    can compute every Col_* and you get the full 148 entry / 171 exit / 120 continuous columns.
    Uses more memory and disk; cache key includes 'full' vs 'restr' so they don't mix.
    """
    if not getattr(config, "use_library_columns", True):
        return
    start = session_start if session_start is not None else getattr(config, "session_start")
    end = session_end if session_end is not None else getattr(config, "session_end")
    if start is None or end is None:
        return
    wide_path_by_year = {
        k: Path(v) for k, v in cleaned_year_data.items()
        if isinstance(v, (Path, str))
    }
    items = [(y, a) for y, by_acct in raw_results.items() for a in by_acct]
    n = len(items)
    tqdm = None
    if n:
        from tqdm.auto import tqdm
    pbar = tqdm(total=100, desc="Phase 2", unit="%") if (tqdm and n) else None
    try:
        for year, by_acct in raw_results.items():
            for acct, res in by_acct.items():
                enriched = enrich_trades_post_backtest(
                    res,
                    cleaned_year_data,
                    wide_path_by_year,
                    session_start=start,
                    session_end=end,
                    config=config,
                    cache_dir=cache_dir,
                    show_progress=False,
                    load_full_columns=load_full_columns,
                )
                raw_results[year][acct] = enriched
                if pbar is not None:
                    pbar.update(100 / n)
    finally:
        if pbar is not None:
            pbar.close()
