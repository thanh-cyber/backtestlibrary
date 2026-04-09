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
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from .columns import _lib as _column_lib


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


def _extract_prefixed_time_map(columns: list, prefix: str) -> dict[str, str]:
    """Map normalized time 'H:MM' -> column name for a prefixed minute field (e.g. Open/High/Low)."""
    out: dict[str, str] = {}
    pref = f"{prefix} "
    for c in columns:
        s = str(c).strip()
        if not s.lower().startswith(pref.lower()):
            continue
        t = s[len(pref) :].strip()
        if _TIME_COL_RE.match(t):
            out[_normalize_time_str(t)] = c
    return out


def _normalize_time_str(t_str: str) -> str:
    """Normalize '9:30' or '09:30' -> '9:30' for merge keys."""
    parts = t_str.split(":")
    h = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    return f"{h}:{mm:02d}"


def _time_str_to_bar_minutes(t_str: str) -> Optional[int]:
    """Minutes from midnight for a bare time column label like '9:30' or '09:30'."""
    s = str(t_str).strip()
    if not _TIME_COL_RE.match(s):
        return None
    parts = s.split(":")
    h = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    return h * 60 + mm


def _session_minute_predicate(
    melt_session_start: Optional[time],
    melt_session_end: Optional[time],
) -> Callable[[Optional[int]], bool]:
    """Inclusive bar-minute filter; both bounds None => keep all."""
    if melt_session_start is None or melt_session_end is None:
        return lambda m: True
    lo = melt_session_start.hour * 60 + melt_session_start.minute
    hi = melt_session_end.hour * 60 + melt_session_end.minute

    def _ok(m: Optional[int]) -> bool:
        if m is None:
            return False
        if lo <= hi:
            return lo <= m <= hi
        return m >= lo or m <= hi

    return _ok


@lru_cache(maxsize=1024)
def _yf_daily(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return None
    def _shape_hist(hist: pd.DataFrame) -> Optional[pd.DataFrame]:
        if hist is None or hist.empty:
            return None
        out = pd.DataFrame(index=pd.to_datetime(hist.index).tz_localize(None).normalize())
        for c in ("Open", "Close"):
            s = None
            if isinstance(hist.columns, pd.MultiIndex):
                lvl0 = hist.columns.get_level_values(0)
                if c in lvl0:
                    sub = hist[c]
                    s = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            elif c in hist.columns:
                s = hist[c]
            if s is not None:
                out[c] = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy()
        if not out.columns.tolist():
            return None
        return out

    try:
        hist = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        shaped = _shape_hist(hist)
        if shaped is not None:
            return shaped
    except Exception:
        pass

    # Fallback path for symbols where yf.download intermittently returns empty.
    try:
        tk = yf.Ticker(str(symbol))
        hist2 = tk.history(start=start_date, end=end_date, interval="1d", auto_adjust=False, actions=False)
        return _shape_hist(hist2)
    except Exception:
        return None


@lru_cache(maxsize=512)
def _yf_info(ticker: str) -> dict:
    try:
        import yfinance as yf
    except Exception:
        return {}
    try:
        tk = yf.Ticker(str(ticker))
        info = getattr(tk, "info", None) or {}
        fast = getattr(tk, "fast_info", None) or {}
        out = {}
        if isinstance(info, dict):
            out.update(info)
        if isinstance(fast, dict):
            out.update(fast)
        return out
    except Exception:
        return {}


def _sector_to_etf(sector: Optional[str]) -> Optional[str]:
    if not sector:
        return None
    s = str(sector).strip().lower()
    m = {
        "technology": "XLK",
        "financial services": "XLF",
        "financial": "XLF",
        "healthcare": "XLV",
        "consumer cyclical": "XLY",
        "consumer defensive": "XLP",
        "industrials": "XLI",
        "energy": "XLE",
        "utilities": "XLU",
        "real estate": "XLRE",
        "basic materials": "XLB",
        "communication services": "XLC",
    }
    return m.get(s)


def _attach_yfinance_context(long_df: pd.DataFrame) -> pd.DataFrame:
    """Attach external benchmark/fundamental series used by column_library market-context columns."""
    if long_df.empty or "Ticker" not in long_df.columns or "datetime" not in long_df.columns:
        return long_df
    out = long_df.copy()
    # yfinance context is intentionally disabled to isolate Phase 2 runtime bottlenecks.
    for col in ("SPY_Close", "SPX_Close", "Col_QQQPremarketChange_Pct"):
        if col not in out.columns:
            out[col] = np.nan

    # Ensure columns exist even when fundamentals enrichment is disabled.
    for col in ("Sector_Close", "Col_MarketCap", "Col_FloatShares", "Col_ShortInterestPctFloat"):
        if col not in out.columns:
            out[col] = np.nan

    # NOTE: Per-ticker yfinance fundamentals are intentionally disabled for now.
    # Large universes trigger heavy rate-limit pressure and can stall Phase 2.
    # Market/fundamental fallback is handled downstream in trade_enrichment.
    return out


def _wide_to_long_from_work(
    work: pd.DataFrame,
    tc: str,
    date_col: str,
    value_vars: list[str],
    vol_map: dict[str, str],
    open_map: dict[str, str],
    high_map: dict[str, str],
    low_map: dict[str, str],
) -> pd.DataFrame:
    """Melt one wide slice (already date-normalized) to long; unsorted rows."""
    empty_cols = ["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]
    if work.empty or not value_vars:
        return pd.DataFrame(columns=empty_cols)

    price_melt = work.melt(
        id_vars=[tc, date_col],
        value_vars=value_vars,
        var_name="time_str",
        value_name="close",
    )
    price_melt["close"] = pd.to_numeric(price_melt["close"], errors="coerce")
    price_melt = price_melt[price_melt["close"].notna() & (price_melt["close"] > 0)]

    if price_melt.empty:
        return pd.DataFrame(columns=empty_cols)

    t_parts = price_melt["time_str"].astype(str).str.split(":", expand=True)
    t_h = pd.to_numeric(t_parts[0], errors="coerce").fillna(0).astype("int64")
    t_mm = pd.to_numeric(t_parts[1], errors="coerce").fillna(0).astype("int64")
    price_melt["time_key"] = t_h.astype(str) + ":" + t_mm.astype(str).str.zfill(2)

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
        merged = price_melt.merge(
            vol_melt,
            on=[tc, date_col, "time_key"],
            how="left",
        )
    else:
        merged = price_melt.copy()
        merged["volume"] = 1.0

    merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce").fillna(1.0)
    merged.loc[merged["volume"] < 0, "volume"] = 1.0

    def _merge_prefixed(base: pd.DataFrame, pmap: dict[str, str], out_name: str) -> pd.DataFrame:
        pcols = [c for c in pmap.values() if c in work.columns]
        if not pcols:
            base[out_name] = base["close"]
            return base
        rev = {v: _normalize_time_str(k) for k, v in pmap.items()}
        pm = work.melt(
            id_vars=[tc, date_col],
            value_vars=pcols,
            var_name=f"{out_name}_col",
            value_name=out_name,
        )
        pm["time_key"] = pm[f"{out_name}_col"].map(rev)
        pm = pm.drop(columns=[f"{out_name}_col"])
        out = base.merge(pm, on=[tc, date_col, "time_key"], how="left")
        out[out_name] = pd.to_numeric(out[out_name], errors="coerce")
        out[out_name] = out[out_name].where(out[out_name].notna(), out["close"])
        return out

    merged = _merge_prefixed(merged, open_map, "open")
    merged = _merge_prefixed(merged, high_map, "high")
    merged = _merge_prefixed(merged, low_map, "low")

    parts = merged["time_str"].astype(str).str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int64")
    mm = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int64")
    merged["datetime"] = merged[date_col] + pd.to_timedelta(h * 60 + mm, unit="m")

    merged["Ticker"] = merged[tc].astype(str).str.upper()
    return merged[["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]]


def wide_to_long(
    wide_df: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    melt_session_start: Optional[time] = None,
    melt_session_end: Optional[time] = None,
    ticker_melt_workers: Optional[int] = None,
    after_each_ticker: Optional[Callable[[], None]] = None,
) -> pd.DataFrame:
    """
    Convert wide backtest DataFrame to long (one row per ticker/date/time).

    Wide: one row per (Date, Ticker), columns "9:30", "9:31", ... for price,
    optional "Vol 9:30", ... for volume.
    Long: Ticker, Date, datetime, open, high, low, close, volume.
    If only one price column per time, open=high=low=close.

    Uses vectorized melt instead of iterrows for speed.

    When ``melt_session_start`` and ``melt_session_end`` are set, only minute columns
    whose bar time falls in that inclusive window are melted (faster when the wide
    file spans more than the backtest session, e.g. PM-only runs on full-day cache).

    Parallel melt: ``ticker_melt_workers`` > 1 or env ``BT_WIDE_TO_LONG_TICKER_WORKERS``.
    Used only when average rows per ticker is high (see env
    ``BT_WIDE_TO_LONG_PARALLEL_MIN_AVG_ROWS``); otherwise one vectorized melt is faster
    (many small per-ticker melts + GIL ≪ single melt on the full frame).

    When ``after_each_ticker`` is set, always melts one ticker at a time and invokes
    the callback after each (for progress bars); skips threaded parallel melt.
    """
    if wide_df.empty or date_col not in wide_df.columns:
        return pd.DataFrame()
    tc = ticker_col if ticker_col in wide_df.columns else ("ticker" if "ticker" in wide_df.columns else None)
    if tc is None:
        return pd.DataFrame()

    _keep_m = _session_minute_predicate(melt_session_start, melt_session_end)

    time_cols = _extract_time_columns(wide_df.columns.tolist())
    if not time_cols:
        return pd.DataFrame()

    cols_all = wide_df.columns.tolist()
    vol_map = _extract_volume_time_map(cols_all)
    open_map = _extract_prefixed_time_map(cols_all, "Open")
    high_map = _extract_prefixed_time_map(cols_all, "High")
    low_map = _extract_prefixed_time_map(cols_all, "Low")
    if melt_session_start is not None and melt_session_end is not None:
        vol_map = {k: v for k, v in vol_map.items() if _keep_m(_time_str_to_bar_minutes(k))}
        open_map = {k: v for k, v in open_map.items() if _keep_m(_time_str_to_bar_minutes(k))}
        high_map = {k: v for k, v in high_map.items() if _keep_m(_time_str_to_bar_minutes(k))}
        low_map = {k: v for k, v in low_map.items() if _keep_m(_time_str_to_bar_minutes(k))}

    value_vars = [c for c in time_cols if c in wide_df.columns]
    if melt_session_start is not None and melt_session_end is not None:
        value_vars = [c for c in value_vars if _keep_m(_time_str_to_bar_minutes(c))]
    if not value_vars:
        return pd.DataFrame()

    work = wide_df.dropna(subset=[date_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work = work[work[date_col].notna()]

    tw = ticker_melt_workers
    if tw is None:
        try:
            tw = int(os.getenv("BT_WIDE_TO_LONG_TICKER_WORKERS", "1"))
        except ValueError:
            tw = 1
    tw = max(1, min(int(tw), 32))
    n_sym = int(work[tc].nunique(dropna=True))
    try:
        _min_avg = int(os.getenv("BT_WIDE_TO_LONG_PARALLEL_MIN_AVG_ROWS", "80"))
    except ValueError:
        _min_avg = 80
    _min_avg = max(1, _min_avg)
    _avg_rows = (len(work) / n_sym) if n_sym else 0.0
    # Many tickers × few rows each: threaded melts are far slower than one melt.
    use_ticker_parallel = tw > 1 and n_sym > 1 and len(work) >= _min_avg * 2 and _avg_rows >= float(_min_avg)

    out: pd.DataFrame
    if after_each_ticker is not None:
        parts_cb: list[pd.DataFrame] = []
        for _tk, w in work.groupby(tc, sort=False):
            if w.empty:
                after_each_ticker()
                continue
            parts_cb.append(
                _wide_to_long_from_work(
                    w,
                    tc,
                    date_col,
                    value_vars,
                    vol_map,
                    open_map,
                    high_map,
                    low_map,
                )
            )
            after_each_ticker()
        parts_cb = [p for p in parts_cb if p is not None and not p.empty]
        if not parts_cb:
            return pd.DataFrame(
                columns=["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]
            )
        out = pd.concat(parts_cb, ignore_index=True)
    elif use_ticker_parallel:
        groups = list(work.groupby(tc, sort=False))
        pool = min(tw, len(groups))
        with ThreadPoolExecutor(max_workers=pool) as ex:
            futs = [
                ex.submit(
                    _wide_to_long_from_work,
                    w,
                    tc,
                    date_col,
                    value_vars,
                    vol_map,
                    open_map,
                    high_map,
                    low_map,
                )
                for _tk, w in groups
            ]
            parts = [fu.result() for fu in futs]
        parts = [p for p in parts if p is not None and not p.empty]
        if not parts:
            return pd.DataFrame(
                columns=["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]
            )
        out = pd.concat(parts, ignore_index=True)
    else:
        out = _wide_to_long_from_work(
            work, tc, date_col, value_vars, vol_map, open_map, high_map, low_map
        )

    if out.empty:
        return pd.DataFrame(
            columns=["Ticker", date_col, "datetime", "open", "high", "low", "close", "volume"]
        )
    return out.sort_values(["Ticker", "datetime"]).reset_index(drop=True)


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
    lib = _column_lib()
    if lib is None:
        raise ImportError("enrich_long_with_library_columns requires vendored backtestlibrary.column_library")
    long_df = _attach_yfinance_context(long_df)

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


def _nearest_bar_row_same_day(
    day_df: pd.DataFrame,
    t: time,
    *,
    datetime_col: str = "datetime",
    max_minute_delta: int = 5,
) -> Optional[pd.Series]:
    """
    Prefer exact (hour, minute) bar; else closest bar by clock-minute distance on the same day.
    Prevents all-NaN snapshots when trades align to a minute that has no bar (sparse tape / alignment).
    """
    if day_df.empty or datetime_col not in day_df.columns:
        return None
    dt = pd.to_datetime(day_df[datetime_col])
    target_min = t.hour * 60 + t.minute
    bar_mins = dt.dt.hour * 60 + dt.dt.minute
    dist = (bar_mins - target_min).abs()
    if (dist == 0).any():
        return day_df.loc[dist == 0].iloc[0]
    imin = int(dist.values.argmin())
    if dist.iloc[imin] > max_minute_delta:
        return None
    return day_df.iloc[imin]


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
    ticker_key = str(ticker).strip().upper()
    mask_td = (df[ticker_col].astype(str).str.strip().str.upper() == ticker_key) & (
        dt.dt.normalize() == date_norm
    )
    day_df = df.loc[mask_td]
    if day_df.empty:
        return None
    mask_exact = (dt.loc[day_df.index].dt.hour == t.hour) & (dt.loc[day_df.index].dt.minute == t.minute)
    subset = day_df.loc[mask_exact]
    if not subset.empty:
        return subset.iloc[0]
    return _nearest_bar_row_same_day(day_df, t, datetime_col=datetime_col)


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
    taken from the enriched long at entry/exit time. Engine path metrics (e.g. Exit_Col_MAE_R,
    Exit_Col_MaxFavorableExcursion_R) are computed at close and are not sourced from the long column list.

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
