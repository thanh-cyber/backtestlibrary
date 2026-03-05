"""
Librarycolumn enrichment for backtestlibrary.

When use_library_columns=True, the engine calls:
  - enrich_cleaned_year_data(cleaned_year_data) -> dict[year, enriched_long_df]
  - get_row_at_time(enriched_long_df, ticker, date, time) -> pd.Series | None

so that Entry_Col_* and Exit_Col_* use the correct bar (entry bar vs exit bar).
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time
from typing import Any, Optional

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


def _normalize_time_str(t_str: str) -> str:
    """Normalize '9:30' or '09:30' -> '9:30' for merge keys."""
    parts = t_str.split(":")
    h = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    return f"{h}:{mm:02d}"


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

    Uses vectorized melt instead of iterrows for speed.
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

    work = wide_df.dropna(subset=[date_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work[work[date_col].notna()]

    # Melt price columns (vectorized)
    price_melt = work.melt(
        id_vars=[tc, date_col],
        value_vars=value_vars,
        var_name="time_str",
        value_name="close",
    )
    price_melt["close"] = pd.to_numeric(price_melt["close"], errors="coerce")
    price_melt = price_melt[price_melt["close"].notna() & (price_melt["close"] > 0)]

    if price_melt.empty:
        return pd.DataFrame(
            columns=["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]
        )

    # Merge volume (vectorized) if we have vol columns
    vol_cols = [c for c in vol_map.values() if c in work.columns]
    if vol_cols:
        vol_reverse = {v: _normalize_time_str(k) for k, v in vol_map.items()}
        vol_melt = work.melt(
            id_vars=[tc, date_col],
            value_vars=vol_cols,
            var_name="vol_col",
            value_name="volume",
        )
        vol_melt["time_key"] = vol_melt["vol_col"].map(vol_reverse)
        vol_melt = vol_melt.drop(columns=["vol_col"])
        t_parts = price_melt["time_str"].astype(str).str.split(":", expand=True)
        t_h = pd.to_numeric(t_parts[0], errors="coerce").fillna(0).astype("int64")
        t_mm = pd.to_numeric(t_parts[1], errors="coerce").fillna(0).astype("int64")
        price_melt["time_key"] = t_h.astype(str) + ":" + t_mm.astype(str).str.zfill(2)
        merged = price_melt.merge(
            vol_melt,
            on=[tc, date_col, "time_key"],
            how="left",
        )
        merged = merged.drop(columns=["time_key"])
    else:
        merged = price_melt.copy()
        merged["volume"] = 1.0

    merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce").fillna(1.0)
    merged.loc[merged["volume"] < 0, "volume"] = 1.0

    # Parse time_str -> datetime (vectorized)
    parts = merged["time_str"].astype(str).str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int64")
    mm = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int64")
    merged["datetime"] = merged[date_col] + pd.to_timedelta(h * 60 + mm, unit="m")
    merged["open"] = merged["close"]
    merged["high"] = merged["close"]
    merged["low"] = merged["close"]

    # Output uses "Ticker" (normalized to upper) for engine compatibility
    merged["Ticker"] = merged[tc].astype(str).str.upper()
    out = merged[["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]]
    out = out.sort_values(["Ticker", "datetime"]).reset_index(drop=True)
    return out


def enrich_long_with_library_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run librarycolumn full pipeline so the long DataFrame gets all Col_* for
    ENTRY_COLUMNS (148), EXIT_SNAPSHOT_COLUMNS (171), CONTINUOUS_TRACKING_COLUMNS (20 base).
    No fallback: uses add_full_enrichment, or add_all_columns + add_all_missing_indicators;
    raises if enrichment fails.

    Expects columns: Ticker, datetime, open, high, low, close, volume (lowercase).
    """
    if long_df.empty:
        return long_df
    try:
        import column_library as lib
    except ImportError as e:
        raise ImportError("enrich_long_with_library_columns requires column_library (librarycolumn). Install with: pip install librarycolumn") from e

    if hasattr(lib, "add_full_enrichment"):
        return lib.add_full_enrichment(long_df)
    if hasattr(lib, "add_all_columns") and hasattr(lib, "add_all_missing_indicators"):
        out = lib.add_all_columns(long_df, inplace=False)
        return lib.add_all_missing_indicators(out)
    if hasattr(lib, "add_all_columns"):
        return lib.add_all_columns(long_df, inplace=False)
    raise RuntimeError("column_library has no add_full_enrichment, add_all_columns, or add_all_missing_indicators")


def _enrich_one_year(
    year: Any,
    df: pd.DataFrame,
    ticker_col: str,
    date_col: str,
) -> tuple[Any, Optional[pd.DataFrame]]:
    """
    Worker for parallel enrich_cleaned_year_data.
    Returns (year, enriched_df) or (year, None) if empty/skip.
    Preserves original year key for engine lookup.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return (year, None)
    if date_col not in df.columns:
        return (year, None)
    if ticker_col not in df.columns and "ticker" not in df.columns:
        return (year, None)
    long_df = wide_to_long(df, ticker_col=ticker_col, date_col=date_col)
    if long_df.empty:
        return (year, None)
    enriched = enrich_long_with_library_columns(long_df)
    if enriched.empty:
        return (year, None)
    return (year, enriched)


def enrich_cleaned_year_data(
    cleaned_year_data: dict[str, pd.DataFrame],
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    max_workers: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    For each year with in-memory DataFrame, convert wide->long and run librarycolumn.
    Returns dict with same key type as cleaned_year_data (year or year_str) -> enriched_long_df.

    Uses ProcessPoolExecutor to process years in parallel when max_workers > 1.
    """
    items = [
        (year, df, ticker_col, date_col)
        for year, df in cleaned_year_data.items()
        if isinstance(df, pd.DataFrame)
        and not df.empty
        and date_col in df.columns
        and (ticker_col in df.columns or "ticker" in df.columns)
    ]
    if not items:
        return {}

    n_years = len(items)
    workers = max_workers if max_workers is not None else min(n_years, (os.cpu_count() or 4))
    out: dict[str, pd.DataFrame] = {}

    if workers <= 1:
        for year, df, tc, dc in items:
            _, enriched = _enrich_one_year(year, df, tc, dc)
            if enriched is not None:
                out[year] = enriched  # preserve key type for engine lookup
        return out

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_enrich_one_year, year, df, tc, dc): year for year, df, tc, dc in items}
        for future in as_completed(futures):
            year_key, enriched = future.result()
            if enriched is not None:
                out[year_key] = enriched  # preserve key type
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


def report_entry_exit_column_gaps(
    enriched_long_columns: list,
    *,
    verbose: bool = True,
    include_continuous: bool = True,
) -> dict:
    """
    Compare expected entry/exit/continuous columns (from column_library) to the columns
    present in the enriched long DataFrame. Use before running the backtest to
    see which columns enrichment did not produce (so they will be NaN in output).

    Note: "Entry" and "Exit" here are only the library snapshot columns (Entry_Col_*, Exit_Col_*)
    taken from the enriched long at entry/exit time. Engine-added exit-only columns
    (e.g. Col_MaxFavorableExcursion_R, Col_MAE_R) are computed by the engine and are not in this list.

    Args:
        enriched_long_columns: list of column names from the enriched long (e.g. df.columns.tolist()).
        verbose: if True, print a short summary to stdout.
        include_continuous: if True, also report continuous-tracking columns expected vs present.

    Returns:
        dict with keys: entry_expected, exit_expected, entry_missing_from_long, exit_missing_from_long,
        entry_count, exit_count, entry_missing_count, exit_missing_count; and if include_continuous:
        continuous_expected, continuous_missing_from_long, continuous_count, continuous_missing_count;
        engine_exit_only_columns, engine_exit_only_count.
    """
    from .columns import get_entry_columns, get_exit_columns, get_continuous_columns
    from .io import get_engine_exit_only_columns

    set_long = set(c for c in enriched_long_columns if isinstance(c, str))
    entry_expected = get_entry_columns()
    exit_expected = get_exit_columns()
    engine_exit_only = get_engine_exit_only_columns()
    entry_missing = [c for c in entry_expected if c not in set_long]
    exit_missing = [c for c in exit_expected if c not in set_long]
    out = {
        "entry_expected": entry_expected,
        "exit_expected": exit_expected,
        "entry_missing_from_long": entry_missing,
        "exit_missing_from_long": exit_missing,
        "entry_count": len(entry_expected),
        "exit_count": len(exit_expected),
        "entry_missing_count": len(entry_missing),
        "exit_missing_count": len(exit_missing),
        "engine_exit_only_columns": engine_exit_only,
        "engine_exit_only_count": len(engine_exit_only),
    }
    if include_continuous:
        continuous_expected = get_continuous_columns()
        continuous_missing = [c for c in continuous_expected if c not in set_long]
        out["continuous_expected"] = continuous_expected
        out["continuous_missing_from_long"] = continuous_missing
        out["continuous_count"] = len(continuous_expected)
        out["continuous_missing_count"] = len(continuous_missing)
    if verbose:
        entry_tip = (entry_missing[:15] + ["..."]) if len(entry_missing) > 15 else entry_missing
        exit_tip = (exit_missing[:15] + ["..."]) if len(exit_missing) > 15 else exit_missing
        lines = [
            "Entry/Exit/Continuous column gaps (expected vs present in enriched long):",
            "  (Entry/Exit = library snapshot columns only; engine exit-only columns are added at close.)",
            f"  Entry: {len(entry_expected)} expected, {len(entry_missing)} missing from long"
            + (f" → {entry_tip}" if entry_missing else ""),
            f"  Exit:  {len(exit_expected)} expected, {len(exit_missing)} missing from long"
            + (f" → {exit_tip}" if exit_missing else ""),
        ]
        if include_continuous:
            cont_tip = (continuous_missing[:15] + ["..."]) if len(continuous_missing) > 15 else continuous_missing
            # 6 output columns per base: Entry, Exit, Max, Min, At30min, At60min (engine + Phase 2)
            n_continuous_output = len(continuous_expected) * 6
            lines.append(
                f"  Continuous: {len(continuous_expected)} base expected, {len(continuous_missing)} missing from long"
                + (f" → {cont_tip}" if continuous_missing else "")
                + f" (→ {n_continuous_output} output columns: Entry/Exit/Max/Min/At30min/At60min each)"
            )
        # Engine exit-only: not in long; added by engine at close → total exit cols = exit_count + engine_exit_only_count
        lines.append(
            f"  Engine exit-only (added at close, not in long): {len(engine_exit_only)} columns"
            f" → total exit columns = {len(exit_expected)} + {len(engine_exit_only)} = {len(exit_expected) + len(engine_exit_only)}"
        )
        lines.append("  " + ", ".join(engine_exit_only[:8]) + (" ..." if len(engine_exit_only) > 8 else ""))
        print("\n".join(lines))
    return out


__all__ = [
    "wide_to_long",
    "enrich_long_with_library_columns",
    "enrich_cleaned_year_data",
    "get_row_at_time",
    "report_entry_exit_column_gaps",
]
