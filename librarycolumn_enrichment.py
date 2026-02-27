"""
Librarycolumn enrichment for backtestlibrary.

When use_library_columns=True, the engine calls:
  - enrich_cleaned_year_data(cleaned_year_data) -> dict[year, enriched_long_df]
  - get_row_at_time(enriched_long_df, ticker, date, time) -> pd.Series | None

so that Entry_Col_* and Exit_Col_* use the correct bar (entry bar vs exit bar).
"""

from __future__ import annotations

import re
from datetime import time
from typing import Optional

import pandas as pd


# Time column pattern: "9:30", "09:30", "16:00"
_TIME_COL_RE = re.compile(r"^(?:\d{1,2}):(\d{2})(?::\d{2})?$")
_VOL_PREFIX_RE = re.compile(r"^(?:Vol|Volume)\s+(\d{1,2}:\d{2}(?::\d{2})?)$", re.I)


def _extract_time_columns(columns: list) -> list[str]:
    out = []
    for c in columns:
        s = str(c).strip()
        if _TIME_COL_RE.match(s):
            out.append(s)
    return sorted(out, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))


def _extract_volume_time_map(columns: list) -> dict[str, str]:
    """Map normalized time 'H:MM' -> column name for volume."""
    vol_map = {}
    for c in columns:
        s = str(c).strip()
        m = _VOL_PREFIX_RE.match(s)
        if m:
            t = m.group(1)
            parts = t.split(":")
            h, mm = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            key = f"{h}:{mm:02d}"
            vol_map[key] = c
    return vol_map


def wide_to_long(
    wide_df: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Convert wide backtest DataFrame to long (one row per ticker/date/time).

    Wide: one row per (Date, Ticker), columns "9:30", "9:31", ... for price,
    optional "Vol 9:30", ... for volume.
    Long: Ticker, Date, datetime, open, high, low, close, volume.
    If only one price column per time, open=high=low=close.
    """
    if wide_df.empty or date_col not in wide_df.columns:
        return pd.DataFrame()
    tc = ticker_col if ticker_col in wide_df.columns else ("ticker" if "ticker" in wide_df.columns else None)
    if tc is None:
        return pd.DataFrame()

    time_cols = _extract_time_columns(wide_df.columns.tolist())
    if not time_cols:
        return pd.DataFrame()

    vol_map = _extract_volume_time_map(wide_df.columns.tolist())
    value_vars = [c for c in time_cols if c in wide_df.columns]
    if not value_vars:
        return pd.DataFrame()

    long_list = []
    wide_df = wide_df.dropna(subset=[date_col]).copy()
    wide_df[date_col] = pd.to_datetime(wide_df[date_col], errors="coerce").dt.normalize()

    for _, r in wide_df.iterrows():
        ticker = r[tc]
        dt_date = r[date_col]
        if pd.isna(dt_date):
            continue
        for t_str in value_vars:
            close = r.get(t_str)
            try:
                close = float(close)
            except (TypeError, ValueError):
                continue
            if pd.isna(close) or close <= 0:
                continue
            parts = t_str.split(":")
            h, mm = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            dt_time = pd.Timestamp(dt_date) + pd.Timedelta(hours=h, minutes=mm)
            vol_col = vol_map.get(t_str) or vol_map.get(f"{h}:{mm:02d}")
            vol = float(r.get(vol_col, 1.0)) if vol_col else 1.0
            if pd.isna(vol) or vol < 0:
                vol = 1.0
            long_list.append({
                "Ticker": str(ticker).upper() if tc else "",
                date_col: dt_date,
                "datetime": dt_time,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": vol,
            })

    if not long_list:
        return pd.DataFrame(
            columns=["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]
        )
    out = pd.DataFrame(long_list)
    out = out.sort_values(["Ticker", "datetime"]).reset_index(drop=True)
    return out


def enrich_long_with_library_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run librarycolumn's add_all_columns (or add_all_missing_indicators) on long DataFrame.

    Expects columns: Ticker, datetime, open, high, low, close, volume (lowercase).
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

    if hasattr(lib, "add_all_columns"):
        return lib.add_all_columns(long_df, inplace=False)
    if hasattr(lib, "add_all_missing_indicators"):
        return lib.add_all_missing_indicators(long_df)
    return long_df


def enrich_cleaned_year_data(
    cleaned_year_data: dict[str, pd.DataFrame],
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
) -> dict[str, pd.DataFrame]:
    """
    For each year with in-memory DataFrame, convert wide->long and run librarycolumn.
    Returns dict with same key type as cleaned_year_data (year or year_str) -> enriched_long_df.
    """
    out = {}
    for year, df in cleaned_year_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if date_col not in df.columns:
            continue
        if ticker_col not in df.columns and "ticker" not in df.columns:
            continue
        long_df = wide_to_long(df, ticker_col=ticker_col, date_col=date_col)
        if long_df.empty:
            continue
        enriched = enrich_long_with_library_columns(long_df)
        if not enriched.empty:
            out[year] = enriched  # preserve key type so engine lookup matches cleaned_year_data
    return out


def get_row_at_time(
    enriched_long_df: Optional[pd.DataFrame],
    ticker: str,
    date: pd.Timestamp,
    t: time,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    datetime_col: str = "datetime",
) -> Optional[pd.Series]:
    """
    Return the row (pd.Series) for (ticker, date, time) from enriched long DataFrame.
    Used by the engine for apply_entry_columns and apply_exit_columns.
    """
    if enriched_long_df is None or enriched_long_df.empty:
        return None
    if datetime_col not in enriched_long_df.columns:
        return None
    df = enriched_long_df
    date_norm = pd.Timestamp(date).normalize()
    dt = pd.to_datetime(df[datetime_col])
    ticker_upper = str(ticker).upper()
    mask = (
        (df[ticker_col].astype(str).str.upper() == ticker_upper)
        & (dt.dt.normalize() == date_norm)
        & (dt.dt.hour == t.hour)
        & (dt.dt.minute == t.minute)
    )
    subset = df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


__all__ = [
    "wide_to_long",
    "enrich_long_with_library_columns",
    "enrich_cleaned_year_data",
    "get_row_at_time",
]
