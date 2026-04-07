"""
Pandas data feed contract and resampling for the backtest engine.

- DataFeedContract: required columns and shape for engine-ready data.
- validate_feed / normalize_feed: check and normalize a DataFrame to the contract.
- PandasDataFeed: wrap a DataFrame or dict[str, DataFrame] and expose to_cleaned_year_data().
- resample_wide_intraday: resample wide-format minute bars to a coarser timeframe (e.g. 5m).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------- Contract ----------
REQUIRED_FEED_COLUMNS = ("Date", "Ticker")
"""Minimum columns the engine expects: Date (date), Ticker (identifier per row)."""

# Time column pattern: "9:30", "09:30", "9:30:00" (price bars); "Vol 9:30" for volume.
_TIME_COL_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_VOL_COL_RE = re.compile(r"^Vol\s+(\d{1,2}):(\d{2})(?::(\d{2}))?$", re.I)


def _is_time_column(name: str) -> bool:
    """True if column name is a time slot (e.g. '9:30' or '9:30:00')."""
    if not isinstance(name, str) or name in REQUIRED_FEED_COLUMNS:
        return False
    return _TIME_COL_RE.match(name.strip()) is not None


def _is_vol_column(name: str) -> bool:
    """True if column name is 'Vol H:MM' or 'Vol H:MM:SS'."""
    if not isinstance(name, str):
        return False
    return _VOL_COL_RE.match(name.strip()) is not None


def _time_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are time slots (price bars), sorted by time."""
    candidates = [c for c in df.columns if _is_time_column(c)]
    if not candidates:
        return []
    # Sort by (hour, minute, second)
    def key(s: str) -> tuple[int, int, int]:
        m = _TIME_COL_RE.match(s.strip())
        if not m:
            return (0, 0, 0)
        h, mm = int(m.group(1)), int(m.group(2))
        ss = int(m.group(3)) if m.lastindex and m.group(3) else 0
        return (h, mm, ss)
    return sorted(candidates, key=key)


def _parse_time_str(s: str) -> Optional[tuple[int, int, int]]:
    """Return (hour, minute, second) or None."""
    m = _TIME_COL_RE.match(s.strip()) if isinstance(s, str) else None
    if not m:
        return None
    h, mm = int(m.group(1)), int(m.group(2))
    ss = int(m.group(3)) if m.lastindex and m.group(3) else 0
    return (h, mm, ss)


def _minute_of_day(s: str) -> int:
    """Minute-of-day (0..1439) for '9:30' or '9:30:00'. Returns -1 if invalid."""
    p = _parse_time_str(s)
    return (p[0] * 60 + p[1]) if p else -1


# ---------- Validate / Normalize ----------
def validate_feed(df: pd.DataFrame, require_time_columns: bool = True) -> None:
    """Raise ValueError if DataFrame does not meet the feed contract."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Feed must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Feed DataFrame is empty")
    for col in REQUIRED_FEED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Feed must have column '{col}'")
    if require_time_columns:
        time_cols = _time_columns(df)
        if not time_cols:
            raise ValueError(
                "Feed must have at least one time-column (e.g. '9:30', '9:31') for intraday bars"
            )


def normalize_feed(
    df: pd.DataFrame,
    session_start: tuple[int, int] = (9, 30),
    session_end: tuple[int, int] = (16, 0),
    drop_extra_columns: bool = False,
) -> pd.DataFrame:
    """Normalize Date and Ticker; optionally restrict to session time columns."""
    validate_feed(df, require_time_columns=True)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["Date"])
    if "Ticker" in out.columns:
        out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
        out = out[out["Ticker"].notna() & (out["Ticker"] != "")]
    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if drop_extra_columns:
        time_cols = _time_columns(out)
        vol_cols = [c for c in out.columns if _is_vol_column(c)]
        keep = [c for c in ["Date", "Ticker"] + time_cols + vol_cols if c in out.columns]
        out = out[[c for c in out.columns if c in keep or c in REQUIRED_FEED_COLUMNS]]
    return out


# ---------- PandasDataFeed ----------
@dataclass
class DataFeedConfig:
    """Optional config for PandasDataFeed normalization."""
    session_start: tuple[int, int] = (9, 30)
    session_end: tuple[int, int] = (16, 0)
    drop_extra_columns: bool = False
    resample_minutes: Optional[int] = None  # If set, resample to this bar size (e.g. 5 for 5m)


class PandasDataFeed:
    """
    Flexible Pandas data feed for the backtest engine.

    Accepts a single DataFrame (one or more years) or a dict[str, DataFrame]
    (year -> DataFrame). Produces engine-ready cleaned_year_data via
    to_cleaned_year_data().
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        config: Optional[DataFeedConfig] = None,
    ):
        self._data = data
        self._config = config or DataFeedConfig()

    def to_cleaned_year_data(self) -> dict[str, pd.DataFrame]:
        """Return dict[year_str, DataFrame] suitable for engine.run(..., cleaned_year_data)."""
        if isinstance(self._data, dict):
            out = {}
            for year, df in self._data.items():
                if df is None or df.empty:
                    continue
                validate_feed(df, require_time_columns=True)
                df = normalize_feed(
                    df,
                    session_start=self._config.session_start,
                    session_end=self._config.session_end,
                    drop_extra_columns=self._config.drop_extra_columns,
                )
                if self._config.resample_minutes is not None and self._config.resample_minutes > 1:
                    df = resample_wide_intraday(
                        df,
                        rule_minutes=self._config.resample_minutes,
                        session_start=self._config.session_start,
                        session_end=self._config.session_end,
                    )
                if not df.empty:
                    out[str(year)] = df
            return out

        # Single DataFrame: split by year
        validate_feed(self._data, require_time_columns=True)
        df = normalize_feed(
            self._data,
            session_start=self._config.session_start,
            session_end=self._config.session_end,
            drop_extra_columns=self._config.drop_extra_columns,
        )
        if df.empty:
            return {}
        if self._config.resample_minutes is not None and self._config.resample_minutes > 1:
            df = resample_wide_intraday(
                df,
                rule_minutes=self._config.resample_minutes,
                session_start=self._config.session_start,
                session_end=self._config.session_end,
            )
        out = {}
        for year_val, g in df.groupby(df["Date"].dt.year):
            out[str(int(year_val))] = g.reset_index(drop=True)
        return out


# ---------- Resampling ----------
def resample_wide_intraday(
    df_wide: pd.DataFrame,
    rule_minutes: int = 5,
    session_start: tuple[int, int] = (9, 30),
    session_end: tuple[int, int] = (16, 0),
    price_agg: str = "last",
    volume_agg: str = "sum",
) -> pd.DataFrame:
    """
    Resample wide-format intraday bars to a coarser timeframe.

    Input: DataFrame with Date, Ticker, and minute columns (e.g. '9:30', '9:31', ...)
    and optional 'Vol 9:30', ... . Each row is one (Ticker, Date); each column is
    one minute bar (price or volume).

    Output: Same structure with fewer columns, one per rule_minutes (e.g. 9:30, 9:35, 9:40).
    Price columns use price_agg ('last' = close, 'first' = open, etc.); volume uses volume_agg.

    The engine should use timeline_step_seconds = rule_minutes * 60 when running on
    resampled data.
    """
    if df_wide.empty or rule_minutes < 1:
        return df_wide.copy()
    validate_feed(df_wide, require_time_columns=True)
    time_cols = _time_columns(df_wide)
    if not time_cols:
        return df_wide.copy()

    vol_cols = [c for c in df_wide.columns if _is_vol_column(c)]
    base_cols = [c for c in ["Date", "Ticker"] if c in df_wide.columns]
    # Build target bar times (session_start to session_end, every rule_minutes)
    start_m = session_start[0] * 60 + session_start[1]
    end_m = session_end[0] * 60 + session_end[1]
    target_minutes = list(range(start_m, end_m + 1, rule_minutes))
    target_labels = [f"{m // 60}:{m % 60:02d}" for m in target_minutes]

    # Map each source time column to minute-of-day (integer) for bucket assignment
    def col_to_minute(s: str) -> int:
        p = _parse_time_str(s)
        return (p[0] * 60 + p[1]) if p else -1

    # For each target bar, which source columns fall in [t, t+rule_minutes)?
    target_to_sources: dict[str, list[str]] = {lb: [] for lb in target_labels}
    for tc in time_cols:
        m = col_to_minute(tc)
        if m < 0:
            continue
        for t_label in target_labels:
            th, tm = _parse_time_str(t_label)[:2]
            t_m = th * 60 + tm
            if t_m <= m < t_m + rule_minutes:
                target_to_sources[t_label].append(tc)
                break
    # Precompute volume bucket membership once (shared by all rows).
    vol_minutes = {c: _minute_of_day(_vol_time_part(c)) for c in vol_cols}
    target_to_vol_sources: dict[str, list[str]] = {lb: [] for lb in target_labels}
    for i, t_label in enumerate(target_labels):
        t_m = target_minutes[i]
        target_to_vol_sources[t_label] = [
            c for c, m in vol_minutes.items()
            if t_m <= m < t_m + rule_minutes
        ]

    out_rows = []
    for idx, row in df_wide.iterrows():
        row_dict = {c: row[c] for c in base_cols}
        for t_label in target_labels:
            srcs = target_to_sources[t_label]
            if not srcs:
                row_dict[t_label] = np.nan
                continue
            vals = row[srcs].astype(float)
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                row_dict[t_label] = np.nan
            elif price_agg == "last":
                row_dict[t_label] = vals.iloc[-1]
            elif price_agg == "first":
                row_dict[t_label] = vals.iloc[0]
            elif price_agg == "max":
                row_dict[t_label] = vals.max()
            elif price_agg == "min":
                row_dict[t_label] = vals.min()
            elif price_agg == "mean":
                row_dict[t_label] = vals.mean()
            else:
                row_dict[t_label] = vals.iloc[-1]
        for t_label in target_labels:
            vol_srcs = target_to_vol_sources[t_label]
            if not vol_srcs:
                if vol_cols:
                    row_dict[f"Vol {t_label}"] = np.nan
                continue
            vvals = row[vol_srcs].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
            row_dict[f"Vol {t_label}"] = float(vvals.sum() if volume_agg == "sum" else vvals.mean())
        out_rows.append(row_dict)

    out = pd.DataFrame(out_rows)
    # Preserve dtypes for Date/Ticker
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    return out


def _vol_time_part(vol_col: str) -> str:
    """Extract time part from 'Vol 9:30' -> '9:30'."""
    m = _VOL_COL_RE.match(vol_col.strip()) if isinstance(vol_col, str) else None
    if not m:
        return ""
    h, mm = m.group(1), m.group(2)
    ss = m.group(3) if m.lastindex >= 3 and m.group(3) else None
    return f"{int(h)}:{int(mm):02d}" + (f":{int(ss):02d}" if ss else "")
