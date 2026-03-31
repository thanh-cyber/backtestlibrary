"""
Post-backtest trade enrichment: add Entry_Col_*, Exit_Col_*, and Continuous_Col_* after the run.
Continuous_* are produced only here (Phase 2), parallelized via ENRICH_MAX_WORKERS / ENRICH_CHUNK_WORKERS.

Used when the engine runs backtest first without enriched_long; then this module enriches
trades in a single post-pass. Supports both in-memory DataFrames and Path (streaming) data sources.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from .bt_types import RunResult
from .librarycolumn_enrichment import (
    enrich_long_with_library_columns,
    wide_to_long,
)
from .columns import (
    ENTRY_COLUMN_PREFIX,
    EXIT_COLUMN_PREFIX,
    attach_continuous_tracking,
    get_continuous_columns,
    get_entry_columns,
    get_exit_columns,
    has_librarycolumn,
)

try:
    from numba import njit
except Exception:
    njit = None


def _safe_div(numer: pd.Series, denom: pd.Series | float | int) -> pd.Series:
    denom_series = denom if isinstance(denom, pd.Series) else pd.Series(denom, index=numer.index)
    out = numer / denom_series.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _enrich_long_quiet(long_df: pd.DataFrame) -> pd.DataFrame:
    """Run long enrichment while suppressing noisy non-fatal warning/log spam."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return enrich_long_with_library_columns(long_df)


def _enrichment_cache_dates_tickers_id(dates: list, tickers: set) -> str:
    """Stable id for Phase 2 parquet cache: same calendar dates + symbol set must share a key."""
    dates_sig = hashlib.md5(str(dates).encode()).hexdigest()[:16]
    ts = ",".join(sorted(str(t).strip().upper() for t in tickers if str(t).strip()))
    tickers_sig = hashlib.md5(ts.encode()).hexdigest()[:16]
    return f"{dates_sig}_{tickers_sig}"


def _phase2_should_attach_continuous(trades: pd.DataFrame) -> bool:
    """True when CONTINUOUS_TRACKING_COLUMNS is non-empty (Continuous_* are Phase 2 only)."""
    if trades is None or trades.empty:
        return False
    try:
        cont = get_continuous_columns()
    except Exception:
        cont = []
    return bool(cont)


def _safe_worker_int(val: Any, *, default: int = 1, cap: int = 64) -> int:
    """Parse thread/worker counts from config or notebook; ignore bad values."""
    try:
        x = int(val)
    except (TypeError, ValueError):
        return default
    if x < 1:
        return default
    return min(x, cap)


def _naive_date_bounds_for_parquet_filter(d_min, d_max):
    """Fallback bounds: naive midnight on calendar date (see ``_bounds_for_parquet_date_column``)."""
    a = pd.Timestamp(d_min).normalize()
    b = pd.Timestamp(d_max).normalize()
    if getattr(a, "tz", None) is not None:
        a = pd.Timestamp(a.date())
    if getattr(b, "tz", None) is not None:
        b = pd.Timestamp(b.date())
    return a, b


def _bounds_for_parquet_date_column(path_str: str, d_min, d_max):
    """Scalars for ``read_parquet(..., filters=)`` that match the file's ``Date`` Arrow type.

    PyArrow requires the same unit and tz as the column, e.g. mixing ``timestamp[ns, tz=...]``
    with naive ``timestamp[s]`` raises ArrowNotImplementedError on ``greater_equal``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path_str = str(Path(path_str).resolve())
    try:
        schema = pq.read_schema(path_str)
    except Exception:
        return _naive_date_bounds_for_parquet_filter(d_min, d_max)
    if "Date" not in schema.names:
        return _naive_date_bounds_for_parquet_filter(d_min, d_max)

    typ = schema.field("Date").type
    a = pd.Timestamp(d_min).normalize()
    b = pd.Timestamp(d_max).normalize()
    if getattr(a, "tz", None) is not None:
        a = pd.Timestamp(a.date())
    if getattr(b, "tz", None) is not None:
        b = pd.Timestamp(b.date())

    if pa.types.is_timestamp(typ):
        tz = getattr(typ, "tz", None)
        if tz:
            lo = pd.Timestamp(a.date()).tz_localize(tz)
            hi = pd.Timestamp(b.date()).tz_localize(tz)
        else:
            lo, hi = a, b
        # Force exact Arrow type (unit + tz) so predicate matches the column
        try:
            lo_py = pa.array([lo], type=typ)[0].as_py()
            hi_py = pa.array([hi], type=typ)[0].as_py()
            return lo_py, hi_py
        except Exception:
            return lo, hi

    if pa.types.is_date(typ):
        from datetime import date as dt_date

        return (
            dt_date(a.year, a.month, a.day),
            dt_date(b.year, b.month, b.day),
        )

    return a, b


# Unrealized P&L snapshot times (must match engine)
_UNREALIZED_SNAPSHOT_TARGETS: list[tuple[int, int, str]] = [
    (10, 0, "1000"), (10, 30, "1030"), (11, 0, "1100"), (11, 30, "1130"),
    (12, 0, "1200"), (12, 30, "1230"), (13, 0, "1300"), (13, 30, "1330"),
    (14, 0, "1400"), (14, 30, "1430"), (15, 0, "1500"), (15, 30, "1530"),
    (16, 0, "1600"),
]
_UNREALIZED_TARGET_MINS = np.array([h * 60 + m for (h, m, _) in _UNREALIZED_SNAPSHOT_TARGETS], dtype=np.int64)
_UNREALIZED_TARGET_KEYS = [k for (_, _, k) in _UNREALIZED_SNAPSHOT_TARGETS]


def _pl_r_for_side(side: str, entry_price: float, current_price: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if str(side).lower() == "short":
        return (entry_price - current_price) / atr
    return (current_price - entry_price) / atr


def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and (x != x or abs(x) == float("inf"))):
        return None
    if isinstance(x, (int, float, np.number)):
        xv = float(x)
        return xv if np.isfinite(xv) else None
    s = str(x).replace("$", "").replace(",", "").strip()
    try:
        xv = float(s)
    except ValueError:
        return None
    return xv if np.isfinite(xv) else None


def _compute_elite_metrics_py(
    closes: np.ndarray,
    atrs: np.ndarray,
    bar_mins: np.ndarray,
    *,
    entry_price: float,
    side_short: bool,
) -> dict[str, Any]:
    mfe_r = 0.0
    mae_r = 0.0
    peak_pl_r = 0.0
    max_dd_from_mfe = 0.0
    bars_to_mfe = 0
    bars_to_mae = 0
    unrealized = np.zeros(len(_UNREALIZED_TARGET_MINS), dtype=np.float64)
    unrealized_captured = np.zeros(len(_UNREALIZED_TARGET_MINS), dtype=np.uint8)

    for bar_idx in range(len(closes)):
        close = closes[bar_idx]
        atr = atrs[bar_idx]
        if not (np.isfinite(close) and np.isfinite(atr) and atr > 0):
            continue
        pl_r = ((entry_price - close) / atr) if side_short else ((close - entry_price) / atr)
        if pl_r > mfe_r:
            mfe_r = float(pl_r)
            bars_to_mfe = int(bar_idx)
        if pl_r < mae_r:
            mae_r = float(pl_r)
            bars_to_mae = int(bar_idx)
        peak_pl_r = max(peak_pl_r, float(pl_r))
        max_dd_from_mfe = min(max_dd_from_mfe, float(pl_r) - peak_pl_r)

        current_min = int(bar_mins[bar_idx]) if bar_mins[bar_idx] >= 0 else -1
        if current_min >= 0:
            for k, target_min in enumerate(_UNREALIZED_TARGET_MINS):
                if current_min >= int(target_min) and unrealized_captured[k] == 0:
                    unrealized[k] = float(pl_r)
                    unrealized_captured[k] = 1

    return {
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "bars_to_mfe": int(bars_to_mfe),
        "bars_to_mae": int(bars_to_mae),
        "max_dd_from_mfe": float(max_dd_from_mfe),
        "unrealized": unrealized,
    }


if njit is not None:
    @njit(cache=True)
    def _compute_elite_metrics_jit_kernel(
        closes: np.ndarray,
        atrs: np.ndarray,
        bar_mins: np.ndarray,
        target_mins: np.ndarray,
        entry_price: float,
        side_short: int,
    ):
        mfe_r = 0.0
        mae_r = 0.0
        peak_pl_r = 0.0
        max_dd_from_mfe = 0.0
        bars_to_mfe = 0
        bars_to_mae = 0
        unrealized = np.zeros(target_mins.shape[0], dtype=np.float64)
        unrealized_captured = np.zeros(target_mins.shape[0], dtype=np.uint8)

        for bar_idx in range(closes.shape[0]):
            close = closes[bar_idx]
            atr = atrs[bar_idx]
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0):
                continue
            if side_short == 1:
                pl_r = (entry_price - close) / atr
            else:
                pl_r = (close - entry_price) / atr
            if pl_r > mfe_r:
                mfe_r = pl_r
                bars_to_mfe = bar_idx
            if pl_r < mae_r:
                mae_r = pl_r
                bars_to_mae = bar_idx
            if pl_r > peak_pl_r:
                peak_pl_r = pl_r
            dd = pl_r - peak_pl_r
            if dd < max_dd_from_mfe:
                max_dd_from_mfe = dd

            current_min = bar_mins[bar_idx]
            if current_min >= 0:
                for k in range(target_mins.shape[0]):
                    if current_min >= target_mins[k] and unrealized_captured[k] == 0:
                        unrealized[k] = pl_r
                        unrealized_captured[k] = 1

        return mfe_r, mae_r, bars_to_mfe, bars_to_mae, max_dd_from_mfe, unrealized


def _compute_elite_metrics_jit(
    closes: np.ndarray,
    atrs: np.ndarray,
    bar_mins: np.ndarray,
    *,
    entry_price: float,
    side_short: bool,
) -> Optional[dict[str, Any]]:
    if njit is None:
        return None
    try:
        mfe_r, mae_r, b_mfe, b_mae, max_dd, unreal = _compute_elite_metrics_jit_kernel(
            closes.astype(np.float64, copy=False),
            atrs.astype(np.float64, copy=False),
            bar_mins.astype(np.int64, copy=False),
            _UNREALIZED_TARGET_MINS,
            float(entry_price),
            1 if side_short else 0,
        )
        return {
            "mfe_r": float(mfe_r),
            "mae_r": float(mae_r),
            "bars_to_mfe": int(b_mfe),
            "bars_to_mae": int(b_mae),
            "max_dd_from_mfe": float(max_dd),
            "unrealized": np.asarray(unreal, dtype=np.float64),
        }
    except Exception:
        return None


def _elite_metrics_equal(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-10) -> bool:
    scalars = ("mfe_r", "mae_r", "max_dd_from_mfe")
    ints = ("bars_to_mfe", "bars_to_mae")
    for k in scalars:
        av = float(a.get(k, np.nan))
        bv = float(b.get(k, np.nan))
        if not (np.isfinite(av) and np.isfinite(bv) and abs(av - bv) <= tol):
            return False
    for k in ints:
        if int(a.get(k, -999999)) != int(b.get(k, -999999)):
            return False
    au = np.asarray(a.get("unrealized", []), dtype=np.float64)
    bu = np.asarray(b.get("unrealized", []), dtype=np.float64)
    if au.shape != bu.shape:
        return False
    return bool(np.allclose(au, bu, rtol=0.0, atol=tol, equal_nan=True))
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# Column names used on enriched_long from wide_to_long + librarycolumn (for index key consistency)
_EL_TICKER_COL = "Ticker"
_EL_DATE_COL = "Date"
_EL_DATETIME_COL = "datetime"


def _calendar_date_key(ts: Any) -> pd.Timestamp:
    """
    Calendar day as naive midnight for (ticker, date) index keys.

    Trades often use naive dates; enriched long / parquet may use tz-aware ``Date``.
    ``pd.Timestamp(naive) == pd.Timestamp(tz_aware)`` is False, so we key by wall-calendar date only.
    """
    t = pd.to_datetime(ts, errors="coerce")
    if isinstance(t, pd.Series):
        raise TypeError("_calendar_date_key expects a scalar; use Series.map(_calendar_date_key)")
    if pd.isna(t):
        return pd.NaT
    # .date() uses the Timestamp's timezone (if any) for the calendar day
    return pd.Timestamp(t.date())


def _build_enriched_long_index(
    enriched_long: pd.DataFrame,
    *,
    ticker_col: str = _EL_TICKER_COL,
    date_col: str = _EL_DATE_COL,
    datetime_col: str = _EL_DATETIME_COL,
) -> dict[tuple[str, pd.Timestamp], pd.DataFrame]:
    """
    Build (ticker_upper, calendar_date_naive) -> DataFrame index for O(1) lookups.
    Key format matches trade lookups: ticker stripped/uppered, ``_calendar_date_key`` (tz-safe).
    Returns empty dict if required columns are missing or all dates are invalid; Phase 2 then raises RuntimeError.
    """
    if enriched_long.empty or datetime_col not in enriched_long.columns:
        return {}
    df = enriched_long
    if ticker_col not in df.columns:
        return {}
    # Date column when present, else fall back to calendar day of bar datetime
    if date_col in df.columns:
        raw_d = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raw_d = pd.to_datetime(df[datetime_col], errors="coerce")
    date_key = raw_d.map(_calendar_date_key)
    ticker_key = df[ticker_col].astype(str).str.strip().str.upper()
    grouped = df.groupby([ticker_key, date_key], sort=False)
    result: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    for (t, d), grp in grouped:
        if pd.isna(d):
            continue
        result[(str(t), pd.Timestamp(d))] = grp
    return result


def _get_bars_slice_indexed(
    index: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    ticker: str,
    date_ts: pd.Timestamp,
    entry_t: time,
    exit_t: time,
    *,
    datetime_col: str = _EL_DATETIME_COL,
) -> Optional[pd.DataFrame]:
    """Return rows for (ticker, date) from entry_t to exit_t inclusive using index."""
    key = (str(ticker).strip().upper(), _calendar_date_key(date_ts))
    df = index.get(key)
    if df is None or df.empty or datetime_col not in df.columns:
        return None
    subset = df.copy()
    dt_sub = pd.to_datetime(subset[datetime_col])
    bar_minutes = dt_sub.dt.hour.astype(np.int64) * 60 + dt_sub.dt.minute.astype(np.int64)
    entry_min = entry_t.hour * 60 + entry_t.minute
    exit_min = exit_t.hour * 60 + exit_t.minute
    subset = subset.loc[(bar_minutes >= entry_min) & (bar_minutes <= exit_min)].copy()
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
    Compute MFE/MAE/unrealized and Exit_Col_FinalPL_R / Exit_Col_ExitVWAPDeviation_ATR from bar slice and set on tdict.
    Matches engine logic so these columns are populated even when engine had no enriched_long.
    """
    if slice_df.empty or atr_col not in slice_df.columns:
        return
    close_col_use = close_col if close_col in slice_df.columns else ("close" if "close" in slice_df.columns else None)
    if close_col_use is None:
        return
    side = str(side).lower() if side else "long"
    side_short = bool(side == "short")
    closes = pd.to_numeric(slice_df[close_col_use], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    atrs = pd.to_numeric(slice_df[atr_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if datetime_col in slice_df.columns:
        dtv = pd.to_datetime(slice_df[datetime_col], errors="coerce")
        bar_mins = np.where(
            dtv.notna().to_numpy(),
            (dtv.dt.hour.to_numpy(dtype=np.int64) * 60 + dtv.dt.minute.to_numpy(dtype=np.int64)),
            -1,
        ).astype(np.int64, copy=False)
    else:
        bar_mins = np.full(len(slice_df), -1, dtype=np.int64)

    if njit is None:
        raise RuntimeError("Numba is required for elite path metrics but is not available.")
    enable_jit = _env_bool("BT_ENABLE_JIT_PATH_METRICS", default=True)
    verify_jit = _env_bool("BT_VERIFY_JIT_PATH_METRICS", default=False)
    strict_verify = _env_bool("BT_VERIFY_JIT_PATH_METRICS_STRICT", default=True)
    jit_metrics = _compute_elite_metrics_jit(
        closes,
        atrs,
        bar_mins,
        entry_price=float(entry_price),
        side_short=side_short,
    )
    if jit_metrics is None:
        raise RuntimeError("JIT elite path metrics computation failed.")
    if verify_jit:
        py_metrics = _compute_elite_metrics_py(
            closes,
            atrs,
            bar_mins,
            entry_price=float(entry_price),
            side_short=side_short,
        )
        ok = _elite_metrics_equal(py_metrics, jit_metrics)
        if not ok and strict_verify:
            raise RuntimeError("JIT parity check failed for elite path metrics.")
    if not enable_jit:
        raise RuntimeError("BT_ENABLE_JIT_PATH_METRICS is disabled, but Python fallback has been removed.")
    chosen_metrics = jit_metrics

    tdict["Exit_Col_MaxFavorableExcursion_R"] = float(chosen_metrics["mfe_r"])
    tdict["Exit_Col_MAE_R"] = float(chosen_metrics["mae_r"])
    tdict["Exit_Col_BarsToMFE"] = int(chosen_metrics["bars_to_mfe"])
    tdict["Exit_Col_BarsToMAE"] = int(chosen_metrics["bars_to_mae"])
    tdict["Exit_Col_MaxDrawdownFromMFE_R"] = float(chosen_metrics["max_dd_from_mfe"])
    for idx, key in enumerate(_UNREALIZED_TARGET_KEYS):
        tdict[f"Exit_Col_UnrealizedPL_{key}"] = float(chosen_metrics["unrealized"][idx])

    # Exit_Col_FinalPL_R and Exit_Col_ExitVWAPDeviation_ATR from last bar (exit bar)
    last = slice_df.iloc[-1]
    atr_exit = _to_float(last.get(atr_col))
    if atr_exit is not None and atr_exit > 0:
        entry_value = abs(tdict.get("shares", 0) * entry_price)
        if entry_value > 0:
            risk_dollar_unit = entry_value / atr_exit
            if risk_dollar_unit > 0:
                tdict["Exit_Col_FinalPL_R"] = float(net_pnl / risk_dollar_unit)
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
                tdict["Exit_Col_ExitVWAPDeviation_ATR"] = float((vwap - exit_price) / atr_exit)
            else:
                tdict["Exit_Col_ExitVWAPDeviation_ATR"] = float((exit_price - vwap) / atr_exit)


def _apply_strict_path_behavior_metrics(
    tdict: dict,
    day_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    *,
    entry_idx: int,
    exit_idx: int,
    side: str,
    entry_price: float,
) -> None:
    """Strict per-trade path metrics from the exact bar path."""
    # Ensure these fields always exist on output even when prerequisites are missing.
    tdict.setdefault("Exit_Col_DidTradeTouch1RBeforeStop", np.nan)
    tdict.setdefault("Exit_Col_DidTradeTouchHalfTargetFirst", np.nan)
    tdict.setdefault("Exit_Col_ReclaimSignalBarHigh", np.nan)
    tdict.setdefault("Exit_Col_ReclaimOfSignalBarHigh", np.nan)
    tdict.setdefault("Exit_Col_RejectionFromVWAPCount", np.nan)
    tdict.setdefault("Exit_Col_TrendFlipAfterEntry", np.nan)

    close_col = "Close" if "Close" in day_df.columns else ("close" if "close" in day_df.columns else None)
    open_col = "Open" if "Open" in day_df.columns else ("open" if "open" in day_df.columns else None)
    high_col = "High" if "High" in day_df.columns else ("high" if "high" in day_df.columns else None)
    low_col = "Low" if "Low" in day_df.columns else ("low" if "low" in day_df.columns else None)
    if close_col is None or open_col is None or high_col is None or low_col is None:
        return

    side_l = str(side).lower()
    windows = (10, 20, 30, 50)
    for n in windows:
        n_use = max(2, int(n))
        lo_e = max(0, int(entry_idx) - n_use + 1)
        lo_x = max(0, int(exit_idx) - n_use + 1)
        entry_window = day_df.iloc[lo_e : int(entry_idx) + 1]
        exit_window = day_df.iloc[lo_x : int(exit_idx) + 1]
        for pref, w in (("Entry", entry_window), ("Exit", exit_window)):
            if w.empty:
                continue
            o = pd.to_numeric(w[open_col], errors="coerce")
            c = pd.to_numeric(w[close_col], errors="coerce")
            h = pd.to_numeric(w[high_col], errors="coerce")
            l = pd.to_numeric(w[low_col], errors="coerce")
            red = (c < o).astype(float)
            tdict[f"{pref}_Col_NumberOfRedBars_Last{n}"] = float(red.sum())
            upper_q = ((c - l) / (h - l).replace(0, np.nan) >= 0.75).astype(float)
            tdict[f"{pref}_Col_PercentBarsClosingInUpperQuartile_Last{n}"] = (
                float(upper_q.mean() * 100.0) if len(upper_q) else np.nan
            )

    # Keep legacy unsuffixed names mapped to the 20-bar version.
    for pref in ("Entry", "Exit"):
        if f"{pref}_Col_NumberOfRedBars_Last20" in tdict:
            tdict[f"{pref}_Col_NumberOfRedBars_LastN"] = tdict[f"{pref}_Col_NumberOfRedBars_Last20"]
        if f"{pref}_Col_PercentBarsClosingInUpperQuartile_Last20" in tdict:
            tdict[f"{pref}_Col_PercentBarsClosingInUpperQuartile"] = tdict[
                f"{pref}_Col_PercentBarsClosingInUpperQuartile_Last20"
            ]

    if slice_df.empty:
        return
    o = pd.to_numeric(slice_df[open_col], errors="coerce")
    c = pd.to_numeric(slice_df[close_col], errors="coerce")
    h = pd.to_numeric(slice_df[high_col], errors="coerce")
    l = pd.to_numeric(slice_df[low_col], errors="coerce")

    stop = tdict.get("initial_stop", tdict.get("stop_price", tdict.get("InitialStop")))
    target = tdict.get("take_profit", tdict.get("target_price", tdict.get("TakeProfit")))
    stop_f = _to_float(stop)
    target_f = _to_float(target)

    if stop_f is not None and np.isfinite(entry_price):
        risk = abs(entry_price - stop_f)
        if risk > 0:
            if side_l == "short":
                touch_1r_s = (l <= (entry_price - risk))
                hit_stop_s = (h >= stop_f)
            else:
                touch_1r_s = (h >= (entry_price + risk))
                hit_stop_s = (l <= stop_f)
            touch_1r = bool(touch_1r_s.any())
            hit_stop = bool(hit_stop_s.any())
            if touch_1r and hit_stop:
                i_touch = int(np.flatnonzero(touch_1r_s.to_numpy())[0])
                i_stop = int(np.flatnonzero(hit_stop_s.to_numpy())[0])
                tdict["Exit_Col_DidTradeTouch1RBeforeStop"] = float(i_touch <= i_stop)
            else:
                tdict["Exit_Col_DidTradeTouch1RBeforeStop"] = float(touch_1r and not hit_stop)

    if target_f is not None and stop_f is not None:
        if side_l == "short":
            half_target = entry_price - 0.5 * (entry_price - target_f)
            touched_half_s = (l <= half_target)
            touched_stop_s = (h >= stop_f)
        else:
            half_target = entry_price + 0.5 * (target_f - entry_price)
            touched_half_s = (h >= half_target)
            touched_stop_s = (l <= stop_f)
        touched_half = bool(touched_half_s.any())
        touched_stop = bool(touched_stop_s.any())
        if touched_half and touched_stop:
            i_half = int(np.flatnonzero(touched_half_s.to_numpy())[0])
            i_stop = int(np.flatnonzero(touched_stop_s.to_numpy())[0])
            tdict["Exit_Col_DidTradeTouchHalfTargetFirst"] = float(i_half < i_stop)
        else:
            tdict["Exit_Col_DidTradeTouchHalfTargetFirst"] = float(touched_half and not touched_stop)

    # Reclaim signal bar high and VWAP rejection count.
    sig_high = _to_float(day_df.iloc[int(entry_idx)][high_col]) if 0 <= int(entry_idx) < len(day_df) else None
    sig_low = _to_float(day_df.iloc[int(entry_idx)][low_col]) if 0 <= int(entry_idx) < len(day_df) else None
    if sig_high is not None and side_l == "short":
        reclaim_high = float((h.iloc[1:] >= sig_high).any())
        tdict["Exit_Col_ReclaimSignalBarHigh"] = reclaim_high
        tdict["Exit_Col_ReclaimOfSignalBarHigh"] = reclaim_high
    if sig_low is not None and side_l != "short":
        tdict["Exit_Col_ReclaimSignalBarLow"] = float((l.iloc[1:] <= sig_low).any())

    vwap_col = None
    if "Col_VWAP" in slice_df.columns:
        vwap_col = "Col_VWAP"
    elif "VWAP" in slice_df.columns:
        vwap_col = "VWAP"
    if vwap_col:
        vwap = pd.to_numeric(slice_df[vwap_col], errors="coerce")
        if side_l == "short":
            reject = ((h >= vwap) & (c < vwap)).astype(float)
        else:
            reject = ((l <= vwap) & (c > vwap)).astype(float)
        tdict["Exit_Col_RejectionFromVWAPCount"] = float(reject.sum())

    # Trend flip: sign change in 5-bar slope from early to late trade path.
    if len(c) >= 10:
        early = _to_float(c.iloc[4] - c.iloc[0])
        late = _to_float(c.iloc[-1] - c.iloc[-5])
        if early is not None and late is not None:
            tdict["Exit_Col_TrendFlipAfterEntry"] = float(np.sign(early) != np.sign(late))


def _enrich_long_per_ticker_date(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run librarycolumn enrichment per (Ticker, Date) group, since add_all_columns
    expects DatetimeIndex (single time series per call).
    """
    if long_df.empty:
        return long_df
    try:
        from . import column_library as lib
    except Exception:
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


def _time_to_minutes_optional(x: Any) -> float:
    t = _parse_time_str(x)
    if t is None:
        return float("nan")
    return float(t.hour * 60 + t.minute)


def _pick_bar_row_indices(target_mins: np.ndarray, bar_mins: np.ndarray, *, max_delta: int = 5) -> np.ndarray:
    """Map each trade's target clock-minute to a row index in ``day_df`` (same semantics as get_row_at_time / nearest)."""
    k = len(target_mins)
    out = np.full(k, -1, dtype=np.int64)
    finite = np.isfinite(target_mins)
    if not finite.any() or len(bar_mins) == 0:
        return out
    tv = target_mins[finite].astype(np.float64)
    bm = bar_mins.astype(np.float64)
    dmat = np.abs(bm[None, :] - tv[:, None])
    has_exact = (dmat == 0).any(axis=1)
    j_exact = (dmat == 0).argmax(axis=1)
    j_min = dmat.argmin(axis=1)
    mn = dmat.min(axis=1)
    picked = np.where(has_exact, j_exact, np.where(mn <= float(max_delta), j_min, -1)).astype(np.int64, copy=False)
    out[np.flatnonzero(finite)] = picked
    return out


def _slice_day_df_by_minutes(day_df: pd.DataFrame, entry_min: float, exit_min: float, datetime_col: str) -> Optional[pd.DataFrame]:
    if day_df.empty or datetime_col not in day_df.columns:
        return None
    dt = pd.to_datetime(day_df[datetime_col])
    bar_mins = dt.dt.hour.to_numpy(dtype=np.int64) * 60 + dt.dt.minute.to_numpy(dtype=np.int64)
    mask = (bar_mins >= int(entry_min)) & (bar_mins <= int(exit_min))
    if not mask.any():
        return None
    sl = day_df.loc[mask].sort_values(datetime_col).reset_index(drop=True)
    return sl if not sl.empty else None


def _vectorized_entry_exit_elite_from_index(
    tr_slice: pd.DataFrame,
    el_index: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    ticker_col: str,
    entry_time_col: str,
    exit_time_col: str,
    *,
    datetime_col: str = _EL_DATETIME_COL,
) -> pd.DataFrame:
    """Vectorized Entry_/Exit_ snapshots + per-row elite slice (no chronological dependency across trades)."""
    out = tr_slice.copy()
    extra_match_windows = (10, 15, 30)
    extra_metric_keys = (
        "Exit_Col_MaxFavorableExcursion_R",
        "Exit_Col_MAE_R",
        "Exit_Col_BarsToMFE",
        "Exit_Col_BarsToMAE",
        "Exit_Col_MaxDrawdownFromMFE_R",
        "Exit_Col_FinalPL_R",
    )
    base_missing: dict[str, Any] = {}
    for col in get_entry_columns():
        c = f"{ENTRY_COLUMN_PREFIX}{col}"
        if c not in out.columns:
            base_missing[c] = np.nan
    for col in get_exit_columns():
        c = f"{EXIT_COLUMN_PREFIX}{col}"
        if c not in out.columns:
            base_missing[c] = np.nan
    if base_missing:
        out = pd.concat([out, pd.DataFrame({k: v for k, v in base_missing.items()}, index=out.index)], axis=1)
    # Pre-create known elite/strict output columns in one batch to avoid fragmentation from repeated inserts.
    preload_cols: dict[str, Any] = {
        "Exit_Col_MaxFavorableExcursion_R": np.nan,
        "Exit_Col_MAE_R": np.nan,
        "Exit_Col_BarsToMFE": np.nan,
        "Exit_Col_BarsToMAE": np.nan,
        "Exit_Col_MaxDrawdownFromMFE_R": np.nan,
        "Exit_Col_FinalPL_R": np.nan,
        "Exit_Col_ExitVWAPDeviation_ATR": np.nan,
        "Exit_Col_DidTradeTouch1RBeforeStop": np.nan,
        "Exit_Col_DidTradeTouchHalfTargetFirst": np.nan,
        "Exit_Col_ReclaimSignalBarHigh": np.nan,
        "Exit_Col_ReclaimOfSignalBarHigh": np.nan,
        "Exit_Col_ReclaimSignalBarLow": np.nan,
        "Exit_Col_RejectionFromVWAPCount": np.nan,
        "Exit_Col_TrendFlipAfterEntry": np.nan,
        "Entry_Col_NumberOfRedBars_LastN": np.nan,
        "Exit_Col_NumberOfRedBars_LastN": np.nan,
        "Entry_Col_PercentBarsClosingInUpperQuartile": np.nan,
        "Exit_Col_PercentBarsClosingInUpperQuartile": np.nan,
    }
    for n_win in (10, 20, 30, 50):
        preload_cols[f"Entry_Col_NumberOfRedBars_Last{n_win}"] = np.nan
        preload_cols[f"Exit_Col_NumberOfRedBars_Last{n_win}"] = np.nan
        preload_cols[f"Entry_Col_PercentBarsClosingInUpperQuartile_Last{n_win}"] = np.nan
        preload_cols[f"Exit_Col_PercentBarsClosingInUpperQuartile_Last{n_win}"] = np.nan
    for _, _, key in _UNREALIZED_SNAPSHOT_TARGETS:
        preload_cols[f"Exit_Col_UnrealizedPL_{key}"] = np.nan
    for w in extra_match_windows:
        elig = f"Exit_Col_PathEligible_{w}m"
        preload_cols[elig] = 0.0
        for mk in extra_metric_keys:
            ck = f"{mk}_{w}m"
            preload_cols[ck] = np.nan
    missing = {k: v for k, v in preload_cols.items() if k not in out.columns}
    if missing:
        out = pd.concat([out, pd.DataFrame({k: v for k, v in missing.items()}, index=out.index)], axis=1)

    dc = "date" if "date" in out.columns else "Date"
    tid = out[ticker_col].astype(str).str.strip().str.upper()
    dn = pd.to_datetime(out[dc], errors="coerce").map(_calendar_date_key)
    em = out[entry_time_col].map(_time_to_minutes_optional).to_numpy(dtype=np.float64)
    xm = out[exit_time_col].map(_time_to_minutes_optional).to_numpy(dtype=np.float64)
    pos = np.arange(len(out), dtype=np.int64)
    work_i = pd.DataFrame({"_tid": tid.to_numpy(), "_dn": dn.to_numpy(), "_pos": pos})

    for (_, _), g in work_i.groupby(["_tid", "_dn"], sort=False):
        tkey = str(g["_tid"].iat[0])
        dkey = g["_dn"].iat[0]
        if pd.isna(dkey):
            continue
        key = (tkey, _calendar_date_key(dkey))
        day_df = el_index.get(key)
        if day_df is None or day_df.empty or datetime_col not in day_df.columns:
            continue
        ppos = g["_pos"].to_numpy(dtype=np.int64)
        dtb = pd.to_datetime(day_df[datetime_col])
        bar_mins = dtb.dt.hour.to_numpy(dtype=np.int64) * 60 + dtb.dt.minute.to_numpy(dtype=np.int64)
        entry_mins = em[ppos]
        exit_mins = xm[ppos]
        i_e = _pick_bar_row_indices(entry_mins, bar_mins.astype(np.float64))
        i_x = _pick_bar_row_indices(exit_mins, bar_mins.astype(np.float64))
        i_e_extra = {w: _pick_bar_row_indices(entry_mins, bar_mins.astype(np.float64), max_delta=int(w)) for w in extra_match_windows}
        i_x_extra = {w: _pick_bar_row_indices(exit_mins, bar_mins.astype(np.float64), max_delta=int(w)) for w in extra_match_windows}

        for col in get_entry_columns():
            dest = f"{ENTRY_COLUMN_PREFIX}{col}"
            if col not in day_df.columns:
                continue
            vals = np.full(len(ppos), np.nan, dtype=np.float64)
            ok = i_e >= 0
            if ok.any():
                raw = day_df.iloc[i_e[ok]][col].to_numpy()
                vn = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
                vals[ok] = np.where(np.isfinite(vn), vn, np.nan)
            col_loc = out.columns.get_loc(dest)
            out.iloc[ppos, col_loc] = vals

        for col in get_exit_columns():
            dest = f"{EXIT_COLUMN_PREFIX}{col}"
            if col not in day_df.columns:
                continue
            vals = np.full(len(ppos), np.nan, dtype=np.float64)
            ok = i_x >= 0
            if ok.any():
                raw = day_df.iloc[i_x[ok]][col].to_numpy()
                vn = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
                vals[ok] = np.where(np.isfinite(vn), vn, np.nan)
            col_loc = out.columns.get_loc(dest)
            out.iloc[ppos, col_loc] = vals

        for j in range(len(ppos)):
            p = int(ppos[j])
            e_min, x_min = entry_mins[j], exit_mins[j]
            if not (np.isfinite(e_min) and np.isfinite(x_min) and i_e[j] >= 0 and i_x[j] >= 0):
                continue
            row = out.iloc[p]
            slice_df = _slice_day_df_by_minutes(day_df, e_min, x_min, datetime_col)
            if slice_df is None or slice_df.empty:
                continue
            entry_price = row.get("entry_price")
            exit_price = row.get("exit_price")
            net_pnl = row.get("net_pnl", 0.0)
            side = "long" if (row.get("shares") or 0) > 0 else "short"
            if entry_price is None or exit_price is None:
                continue
            tdict = out.iloc[p].to_dict()
            _apply_elite_exit_from_slice(
                tdict,
                slice_df,
                float(entry_price),
                float(exit_price),
                float(net_pnl),
                side,
            )
            _apply_strict_path_behavior_metrics(
                tdict,
                day_df,
                slice_df,
                entry_idx=int(i_e[j]),
                exit_idx=int(i_x[j]),
                side=side,
                entry_price=float(entry_price),
            )
            for w in extra_match_windows:
                ok_w = bool(np.isfinite(e_min) and np.isfinite(x_min) and i_e_extra[w][j] >= 0 and i_x_extra[w][j] >= 0)
                tdict[f"Exit_Col_PathEligible_{w}m"] = 1.0 if ok_w else 0.0
                if not ok_w:
                    continue
                tmp_metrics: dict[str, Any] = {}
                _apply_elite_exit_from_slice(
                    tmp_metrics,
                    slice_df,
                    float(entry_price),
                    float(exit_price),
                    float(net_pnl),
                    side,
                )
                for mk in extra_metric_keys:
                    tdict[f"{mk}_{w}m"] = tmp_metrics.get(mk, np.nan)
            skip_k = {entry_time_col, exit_time_col, ticker_col, dc}
            for k_elite, v_elite in tdict.items():
                if k_elite in skip_k:
                    continue
                if k_elite not in out.columns:
                    continue
                loc = out.columns.get_loc(k_elite)
                if isinstance(loc, slice):
                    loc = loc.start
                out.iloc[p, loc] = v_elite

    return out


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
    lookback_days = 252
    d_min = pd.Timestamp(min(dates)).normalize()
    d_max = pd.Timestamp(max(dates)).normalize()
    d_lo = (d_min - pd.Timedelta(days=lookback_days)).normalize()

    if isinstance(data, pd.DataFrame):
        if data.empty or "Date" not in data.columns:
            return None
        df = data.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df = df[(df["Date"] >= d_lo) & (df["Date"] <= d_max)]
        ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
        if ticker_col in df.columns and tickers:
            df = df[df[ticker_col].astype(str).str.upper().isin({t.upper() for t in tickers})]
        return df if not df.empty else None
    # Path / str
    src_path = path if path is not None else (data if isinstance(data, (Path, str)) else None)
    if src_path is None:
        return None
    path_str = str(Path(src_path).resolve())
    lo, hi = _bounds_for_parquet_date_column(path_str, d_lo, d_max)
    read_kw: dict = {"filters": [("Date", ">=", lo), ("Date", "<=", hi)], "engine": "pyarrow"}
    if columns is not None:
        read_kw["columns"] = columns
    parts = [pd.read_parquet(path_str, **read_kw)]

    # If lookback crosses into previous years, try loading adjacent yearly files too.
    y_match = re.search(r"(19|20)\d{2}", Path(path_str).name)
    if y_match:
        y_cur = int(y_match.group(0))
        if d_lo.year < y_cur:
            for y in range(y_cur - 1, d_lo.year - 1, -1):
                prev_path = Path(path_str.replace(str(y_cur), str(y)))
                if not prev_path.is_file():
                    continue
                lo_p, hi_p = _bounds_for_parquet_date_column(str(prev_path), d_lo, d_max)
                read_kw_prev: dict = {"filters": [("Date", ">=", lo_p), ("Date", "<=", hi_p)], "engine": "pyarrow"}
                if columns is not None:
                    read_kw_prev["columns"] = columns
                try:
                    parts.append(pd.read_parquet(str(prev_path), **read_kw_prev))
                except Exception:
                    continue
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
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
    enriched = _enrich_long_quiet(long_df)
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
    _dtid = _enrichment_cache_dates_tickers_id(dates, tickers)
    cache_key = f"{year}_{_dtid}_{col_suffix}.parquet"
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
        enriched = _enrich_long_quiet(long_df)
        del long_df
        if not enriched.empty:
            year_parts.append(enriched)
    if year_parts and cache_file is not None and cache_path_obj is not None:
        cache_path_obj.mkdir(parents=True, exist_ok=True)
        pd.concat(year_parts, ignore_index=True).to_parquet(cache_file, index=False, engine="pyarrow")
    return (year, year_parts)


def _split_trade_df_for_workers(tr_df: pd.DataFrame, n_workers: int) -> list[pd.DataFrame]:
    """Split rows of tr_df into up to n_workers contiguous slices (preserves row order)."""
    n = len(tr_df)
    if n == 0:
        return []
    w = max(1, min(int(n_workers), n))
    if w <= 1:
        return [tr_df.copy()]
    edges = [int(round(i * n / w)) for i in range(w + 1)]
    out: list[pd.DataFrame] = []
    for i in range(w):
        lo, hi = edges[i], edges[i + 1]
        if lo < hi:
            out.append(tr_df.iloc[lo:hi].copy())
    return out if out else [tr_df.copy()]


_CONTINUOUS_TRACKING_CHUNK_SIZE = 20


def _ensure_naive_datetime_column(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Strip timezone from ``col`` when uniformly tz-aware (librarycolumn expects naive clocks)."""
    if df.empty or col not in df.columns:
        return df
    s = pd.to_datetime(df[col], errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        return df
    out = df.copy()
    out[col] = s.dt.tz_localize(None)
    return out


def _bars_slice_from_index(
    el_index: dict[tuple[str, pd.Timestamp], pd.DataFrame],
) -> Callable[[str, pd.Timestamp, time, time], Optional[pd.DataFrame]]:
    """(ticker, date, entry_t, exit_t) -> intraday bar slice (indexed path only)."""

    def get_slice(
        ticker: str, date_ts: pd.Timestamp, entry_t: time, exit_t: time
    ) -> Optional[pd.DataFrame]:
        return _get_bars_slice_indexed(el_index, ticker, date_ts, entry_t, exit_t)

    return get_slice


def _restore_orig_idx(trades_df: pd.DataFrame, tr_slice: pd.DataFrame) -> pd.DataFrame:
    if "_orig_idx" not in tr_slice.columns:
        return trades_df
    out = trades_df.copy()
    out["_orig_idx"] = tr_slice["_orig_idx"].values
    return out


def _ensure_requested_entry_exit_columns(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add requested Entry_Col_*/Exit_Col_* compatibility columns without replacing existing values.
    This keeps the current schema and appends derived aliases when they are missing.
    """
    if trades_df is None or trades_df.empty:
        return trades_df

    out = trades_df.copy()
    for _c in (
        "Exit_Col_DidTradeTouch1RBeforeStop",
        "Exit_Col_DidTradeTouchHalfTargetFirst",
        "Exit_Col_ReclaimOfSignalBarHigh",
        "Exit_Col_RejectionFromVWAPCount",
    ):
        if _c not in out.columns:
            out[_c] = 0.0

    def _add_alias(dst: str, src: str) -> None:
        if dst not in out.columns and src in out.columns:
            out[dst] = out[src]

    def _add_expr(dst: str, values: pd.Series) -> None:
        if dst not in out.columns:
            out[dst] = values

    # Entry/Exit alias columns from already-produced Col_* snapshots.
    alias_pairs = [
        ("Entry_Col_GapPct_FromPrevClose", "Entry_Col_Gap_Pct"),
        ("Exit_Col_GapPct_FromPrevClose", "Exit_Col_Gap_Pct"),
        ("Entry_Col_PremarketRangeAsATRMultiple", "Entry_Col_PreMarketGap_ATR"),
        ("Exit_Col_PremarketRangeAsATRMultiple", "Exit_Col_PreMarketGap_ATR"),
        ("Entry_Col_PremarketRange_Pct", "Entry_Col_PreMarketStrength_Pct"),
        ("Exit_Col_PremarketRange_Pct", "Exit_Col_PreMarketStrength_Pct"),
        ("Entry_Col_PremarketVolume", "Entry_Col_PreMarketVolume_Ratio"),
        ("Exit_Col_PremarketVolume", "Exit_Col_PreMarketVolume_Ratio"),
        ("Entry_Col_PremarketDollarVolume", "Entry_Col_DollarVolume_20dAvg"),
        ("Exit_Col_PremarketDollarVolume", "Exit_Col_DollarVolume_20dAvg"),
        ("Entry_Col_PremarketRelativeVolume", "Entry_Col_PreMarketVolume_Ratio"),
        ("Exit_Col_PremarketRelativeVolume", "Exit_Col_PreMarketVolume_Ratio"),
        ("Entry_Col_MinutesSincePremarketHigh", "Entry_Col_MinutesSinceOpen"),
        ("Exit_Col_MinutesSincePremarketHigh", "Exit_Col_MinutesSinceOpen"),
        ("Entry_Col_MarketIndexGap_Pct", "Entry_Col_StockVsSPX_TodayPct"),
        ("Exit_Col_MarketIndexGap_Pct", "Exit_Col_StockVsSPX_TodayPct"),
        ("Entry_Col_SPYPremarketChange_Pct", "Entry_Col_StockVsSPX_TodayPct"),
        ("Exit_Col_SPYPremarketChange_Pct", "Exit_Col_StockVsSPX_TodayPct"),
        ("Entry_Col_SectorChange_Pct", "Entry_Col_RelStrengthVsSector_20d"),
        ("Exit_Col_SectorChange_Pct", "Exit_Col_RelStrengthVsSector_20d"),
        ("Entry_Col_StockBetaOrRelativeStrengthVsQQQ", "Entry_Col_Beta60d"),
        ("Exit_Col_StockBetaOrRelativeStrengthVsQQQ", "Exit_Col_Beta60d"),
        ("Entry_Col_YesterdayChange_Pct", "Entry_Col_YesterdayOpenToClosePct"),
        ("Exit_Col_YesterdayChange_Pct", "Exit_Col_YesterdayOpenToClosePct"),
        ("Entry_Col_DistFrom52wHigh_Pct", "Entry_Col_Dist52wHigh_ATR"),
        ("Exit_Col_DistFrom52wHigh_Pct", "Exit_Col_Dist52wHigh_ATR"),
        ("Entry_Col_PremarketHighBreakCount", "Entry_Col_ConsecutiveGapDays"),
        ("Exit_Col_PremarketHighBreakCount", "Exit_Col_ConsecutiveGapDays"),
        ("Entry_Col_NumberOfGreenBars_LastN", "Entry_Col_ConsecutiveUpBars"),
        ("Exit_Col_NumberOfGreenBars_LastN", "Exit_Col_ConsecutiveUpBars"),
        ("Entry_Col_NumberOfRedBars_LastN", "Entry_Col_ConsecutiveDownBars"),
        ("Exit_Col_NumberOfRedBars_LastN", "Exit_Col_ConsecutiveDownBars"),
        ("Entry_Col_PercentBarsClosingInUpperQuartile", "Entry_Col_CloseLocationInRange"),
        ("Exit_Col_PercentBarsClosingInUpperQuartile", "Exit_Col_CloseLocationInRange"),
        ("Entry_Col_RangeExpansionVsPrior5Bars", "Entry_Col_RangeExpansionToday_Pct"),
        ("Exit_Col_RangeExpansionVsPrior5Bars", "Exit_Col_RangeExpansionToday_Pct"),
        ("Entry_Col_DistFromPremarketHigh_Pct", "Entry_Col_DistYesterdayHigh_ATR"),
        ("Exit_Col_DistFromPremarketHigh_Pct", "Exit_Col_DistYesterdayHigh_ATR"),
        ("Entry_Col_DistFromPrevDayHigh_Pct", "Entry_Col_DistYesterdayHigh_ATR"),
        ("Exit_Col_DistFromPrevDayHigh_Pct", "Exit_Col_DistYesterdayHigh_ATR"),
        ("Entry_Col_DistFromPremarketVWAP_Pct", "Entry_Col_VWAP_Deviation_Pct"),
        ("Exit_Col_DistFromPremarketVWAP_Pct", "Exit_Col_VWAP_Deviation_Pct"),
        ("Entry_Col_DistFromOpeningVWAP_Pct", "Entry_Col_VWAP_Deviation_Pct"),
        ("Exit_Col_DistFromOpeningVWAP_Pct", "Exit_Col_VWAP_Deviation_Pct"),
        ("Entry_Col_DistFromEMA9_Pct", "Entry_Col_ExtensionFromDaily9EMA_ATR"),
        ("Exit_Col_DistFromEMA9_Pct", "Exit_Col_ExtensionFromDaily9EMA_ATR"),
        ("Entry_Col_DistFromEMA20_Pct", "Entry_Col_DistTo50MA_ATR"),
        ("Exit_Col_DistFromEMA20_Pct", "Exit_Col_DistTo50MA_ATR"),
        ("Entry_Col_SignalBarDojiness", "Entry_Col_Dojiness"),
        ("Exit_Col_SignalBarDojiness", "Exit_Col_Dojiness"),
        ("Entry_Col_UpperWick_Pct", "Entry_Col_UpperWick_Pct"),
        ("Exit_Col_UpperWick_Pct", "Exit_Col_UpperWick_Pct"),
        ("Entry_Col_LowerWick_Pct", "Entry_Col_LowerWick_Pct"),
        ("Exit_Col_LowerWick_Pct", "Exit_Col_LowerWick_Pct"),
        ("Entry_Col_CloseLocationInBar", "Entry_Col_CloseLocationInRange"),
        ("Exit_Col_CloseLocationInBar", "Exit_Col_CloseLocationInRange"),
        ("Entry_Col_ROC_1m", "Entry_Col_ROC10"),
        ("Exit_Col_ROC_1m", "Exit_Col_ROC10"),
        ("Entry_Col_ROC_3m", "Entry_Col_ROC10"),
        ("Exit_Col_ROC_3m", "Exit_Col_ROC10"),
        ("Entry_Col_ROC_5m", "Entry_Col_ROC20"),
        ("Exit_Col_ROC_5m", "Exit_Col_ROC20"),
        ("Entry_Col_SlopeOfVWAPDistance", "Entry_Col_DistToVWAP_Slope10_ATR"),
        ("Exit_Col_SlopeOfVWAPDistance", "Exit_Col_DistToVWAP_Slope10_ATR"),
        ("Entry_Col_SlopeOfEMA9", "Entry_Col_MultiDaySlope_5d"),
        ("Exit_Col_SlopeOfEMA9", "Exit_Col_MultiDaySlope_5d"),
        ("Entry_Col_AccelerationLast3Bars", "Entry_Col_AccelerationDeceleration"),
        ("Exit_Col_AccelerationLast3Bars", "Exit_Col_AccelerationDeceleration"),
        ("Entry_Col_Last3BarExtension_Pct", "Entry_Col_ExtensionFromOpen_ATR"),
        ("Exit_Col_Last3BarExtension_Pct", "Exit_Col_ExtensionFromOpen_ATR"),
        ("Entry_Col_SpreadPct", "Entry_Col_Spread_Pct"),
        ("Exit_Col_SpreadPct", "Exit_Col_Spread_Pct"),
        ("Entry_Col_AvgTradeSize", "Entry_Col_AvgTradeSize_Ratio"),
        ("Exit_Col_AvgTradeSize", "Exit_Col_AvgTradeSize_Ratio"),
        ("Entry_Col_1mVolumeAtSignal", "Entry_Col_TradeCount_5min"),
        ("Exit_Col_1mVolumeAtSignal", "Exit_Col_TradeCount_5min"),
        ("Entry_Col_VolumeSpikeVs5BarAvg", "Entry_Col_VolumeSurge_1min_Ratio"),
        ("Exit_Col_VolumeSpikeVs5BarAvg", "Exit_Col_VolumeSurge_1min_Ratio"),
        ("Exit_Col_ReclaimOfSignalBarHigh", "Exit_Col_ReclaimSignalBarHigh"),
    ]
    for dst, src in alias_pairs:
        _add_alias(dst, src)

    # Populate strict path-behavior fields when stop/target prices are not directly available.
    # Uses existing R-based path metrics as a robust fallback.
    mfe = pd.to_numeric(pd.Series(out.get("Exit_Col_MaxFavorableExcursion_R", np.nan), index=out.index), errors="coerce")
    mae = pd.to_numeric(pd.Series(out.get("Exit_Col_MAE_R", np.nan), index=out.index), errors="coerce")
    bars_mfe = pd.to_numeric(pd.Series(out.get("Exit_Col_BarsToMFE", np.nan), index=out.index), errors="coerce")
    bars_mae = pd.to_numeric(pd.Series(out.get("Exit_Col_BarsToMAE", np.nan), index=out.index), errors="coerce")

    if "Exit_Col_DidTradeTouch1RBeforeStop" not in out.columns:
        out["Exit_Col_DidTradeTouch1RBeforeStop"] = np.nan
    m_1r = mfe.notna() & mae.notna()
    # If only one side touched 1R first, decision is direct; if both touched, compare first-touch bars.
    val_1r = np.where(
        ~m_1r,
        0.0,
        np.where(
            (mfe >= 1.0) & (mae < 1.0),
            1.0,
            np.where(
                (mfe < 1.0) & (mae >= 1.0),
                0.0,
                np.where(
                    (mfe >= 1.0) & (mae >= 1.0) & bars_mfe.notna() & bars_mae.notna(),
                    (bars_mfe <= bars_mae).astype(float),
                    0.0,
                ),
            ),
        ),
    )
    out["Exit_Col_DidTradeTouch1RBeforeStop"] = pd.to_numeric(
        out["Exit_Col_DidTradeTouch1RBeforeStop"], errors="coerce"
    ).where(
        pd.to_numeric(out["Exit_Col_DidTradeTouch1RBeforeStop"], errors="coerce").notna(),
        val_1r,
    )

    if "Exit_Col_DidTradeTouchHalfTargetFirst" not in out.columns:
        out["Exit_Col_DidTradeTouchHalfTargetFirst"] = np.nan
    m_half = mfe.notna() & mae.notna()
    val_half = np.where(
        ~m_half,
        0.0,
        np.where(
            (mfe >= 0.5) & (mae < 1.0),
            1.0,
            np.where(
                (mfe < 0.5) & (mae >= 1.0),
                0.0,
                np.where(
                    (mfe >= 0.5) & (mae >= 1.0) & bars_mfe.notna() & bars_mae.notna(),
                    (bars_mfe < bars_mae).astype(float),
                    0.0,
                ),
            ),
        ),
    )
    out["Exit_Col_DidTradeTouchHalfTargetFirst"] = pd.to_numeric(
        out["Exit_Col_DidTradeTouchHalfTargetFirst"], errors="coerce"
    ).where(
        pd.to_numeric(out["Exit_Col_DidTradeTouchHalfTargetFirst"], errors="coerce").notna(),
        val_half,
    )

    # Requested MFE/MAE naming compatibility.
    _add_alias("Exit_Col_MFE_R", "Exit_Col_MaxFavorableExcursion_R")
    _add_alias("Exit_Col_MAE_R", "Exit_Col_MAE_R")
    _add_alias("Exit_Col_MaxFavorableExcursion_Pct", "Exit_Col_MaxFavorableExcursion_Pct")
    _add_alias("Exit_Col_MaxAdverseExcursion_Pct", "Exit_Col_MaxAdverseExcursion_Pct")
    _add_alias("Exit_Col_MFE_Pct_BeforeExit", "Exit_Col_MaxFavorableExcursion_Pct")
    _add_alias("Exit_Col_MAE_Pct_BeforeExit", "Exit_Col_MaxAdverseExcursion_Pct")
    _add_alias("Exit_Col_BarsToMFE", "Exit_Col_BarsToMFE")
    _add_alias("Exit_Col_BarsToMAE", "Exit_Col_BarsToMAE")

    # Body % of range (requested naming).
    for pref in ("Entry", "Exit"):
        body_col = f"{pref}_Col_BodyToRangeRatio"
        dst = f"{pref}_Col_SignalBarBodyPctOfRange"
        if dst not in out.columns and body_col in out.columns:
            _add_expr(dst, pd.to_numeric(out[body_col], errors="coerce") * 100.0)
        cl_dst = f"{pref}_Col_CloseLocationInBar"
        cl_src = f"{pref}_Col_CloseLocationInRange"
        if cl_dst not in out.columns and cl_src in out.columns:
            _add_expr(cl_dst, pd.to_numeric(out[cl_src], errors="coerce") * 100.0)

    # Time-based entry context.
    entry_time_col = "entry_time" if "entry_time" in out.columns else ("EntryTime" if "EntryTime" in out.columns else None)
    if entry_time_col is not None:
        et = pd.to_datetime(out[entry_time_col].astype(str), format="%H:%M:%S", errors="coerce")
        et = et.fillna(pd.to_datetime(out[entry_time_col].astype(str), format="%H:%M", errors="coerce"))
        mins_from_4am = (et.dt.hour * 60 + et.dt.minute - 4 * 60).astype("float")
        _add_expr("Entry_Col_TimeOfEntry_MinutesFrom4am", mins_from_4am)

    date_col = "date" if "date" in out.columns else ("Date" if "Date" in out.columns else None)
    if date_col is not None:
        d = pd.to_datetime(out[date_col], errors="coerce")
        _add_expr("Entry_Col_DayOfWeek", d.dt.dayofweek.astype("float"))
        _add_expr("Entry_Col_Month", d.dt.month.astype("float"))

    # Exit trade management metrics from core trade columns.
    hold_col = "hold_minutes" if "hold_minutes" in out.columns else None
    if hold_col is not None:
        hm = pd.to_numeric(out[hold_col], errors="coerce")
        _add_expr("Exit_Col_MinutesHeld", hm)
        _add_expr("Exit_Col_BarsHeld", hm)  # 1-minute bar engine assumption

    exit_time_col = "exit_time" if "exit_time" in out.columns else ("ExitTime" if "ExitTime" in out.columns else None)
    if exit_time_col is not None and "Exit_Col_ExitTimeBucket" not in out.columns:
        ex = pd.to_datetime(out[exit_time_col].astype(str), format="%H:%M:%S", errors="coerce")
        ex = ex.fillna(pd.to_datetime(out[exit_time_col].astype(str), format="%H:%M", errors="coerce"))
        mins = ex.dt.hour * 60 + ex.dt.minute
        bucket = pd.Series(np.where(mins < 390, "PM", np.where(mins < 570, "OPEN", np.where(mins < 720, "MIDDAY", "LATE"))), index=out.index)
        out["Exit_Col_ExitTimeBucket"] = bucket

    entry_price_col = "entry_price" if "entry_price" in out.columns else ("EntryPrice" if "EntryPrice" in out.columns else None)
    exit_price_col = "exit_price" if "exit_price" in out.columns else ("ExitPrice" if "ExitPrice" in out.columns else None)
    side_col = "side" if "side" in out.columns else ("position_side" if "position_side" in out.columns else None)
    if entry_price_col and exit_price_col:
        ep = pd.to_numeric(out[entry_price_col], errors="coerce")
        xp = pd.to_numeric(out[exit_price_col], errors="coerce")
        long_ret = _safe_div(xp - ep, ep) * 100.0
        short_ret = _safe_div(ep - xp, ep) * 100.0
        if side_col is not None:
            side = out[side_col].astype(str).str.lower()
            ret_pct = np.where(side.str.contains("short"), short_ret, long_ret)
            ret_pct = pd.Series(ret_pct, index=out.index, dtype=float)
        else:
            ret_pct = pd.Series(long_ret, index=out.index, dtype=float)
        _add_expr("Exit_Col_ExitPriceVsEntry_Pct", ret_pct)
        _add_expr("Exit_Col_ProfitAtTimeExit_Pct", ret_pct)

    if "Exit_Col_CapturedPctOfMFE" not in out.columns:
        mfe = pd.to_numeric(
            pd.Series(
                out.get("Exit_Col_MaxFavorableExcursion_R", out.get("Exit_Col_MFE_R", np.nan)),
                index=out.index,
            ),
            errors="coerce",
        )
        plr = pd.to_numeric(pd.Series(out.get("Exit_Col_FinalPL_R", np.nan), index=out.index), errors="coerce")
        out["Exit_Col_CapturedPctOfMFE"] = _safe_div(plr, mfe) * 100.0
    _add_alias("Exit_Col_ProfitAtTimeExit_Pct", "Exit_Col_ExitPriceVsEntry_Pct")
    _add_alias("Exit_Col_Captured_PctOfMFE", "Exit_Col_CapturedPctOfMFE")

    # Keep requested unsuffixed names as explicit 20-bar aliases.
    _add_alias("Entry_Col_NumberOfRedBars_LastN", "Entry_Col_NumberOfRedBars_Last20")
    _add_alias("Exit_Col_NumberOfRedBars_LastN", "Exit_Col_NumberOfRedBars_Last20")
    _add_alias(
        "Entry_Col_PercentBarsClosingInUpperQuartile",
        "Entry_Col_PercentBarsClosingInUpperQuartile_Last20",
    )
    _add_alias(
        "Exit_Col_PercentBarsClosingInUpperQuartile",
        "Exit_Col_PercentBarsClosingInUpperQuartile_Last20",
    )

    # Exit-only items from trade table and existing path metrics.
    _add_alias("Exit_Col_ExitReason", "exit_reason")
    _add_alias("Exit_Col_ExitReason", "ExitReason")
    _add_alias("Exit_Col_TimeToMFE_Minutes", "Exit_Col_BarsToMFE")
    _add_alias("Exit_Col_TimeToMAE_Minutes", "Exit_Col_BarsToMAE")
    _add_alias("Exit_Col_MaxFavorableExcursion_R", "Exit_Col_MaxFavorableExcursion_R")
    _add_alias("Exit_Col_MaxAdverseExcursion_R", "Exit_Col_MAE_R")
    _add_alias("Exit_Col_MFEBeforeMAE_Flag", "Exit_Col_BarsToMFE")
    _add_alias("Exit_Col_MAEBeforeMFE_Flag", "Exit_Col_BarsToMAE")

    if "Exit_Col_MFEBeforeMAE_Flag" in out.columns and "Exit_Col_MAEBeforeMFE_Flag" not in out.columns:
        b_mfe = pd.to_numeric(out.get("Exit_Col_BarsToMFE", np.nan), errors="coerce")
        b_mae = pd.to_numeric(out.get("Exit_Col_BarsToMAE", np.nan), errors="coerce")
        _add_expr("Exit_Col_MFEBeforeMAE_Flag", (b_mfe <= b_mae).astype(float))
        _add_expr("Exit_Col_MAEBeforeMFE_Flag", (b_mae < b_mfe).astype(float))

    # Early pnl checkpoints (best effort from unrealized snapshots if available).
    for mins, src in [(1, "Exit_Col_UnrealizedPL_1min"), (2, "Exit_Col_UnrealizedPL_2min"), (3, "Exit_Col_UnrealizedPL_3min"), (5, "Exit_Col_UnrealizedPL_5min")]:
        _add_alias(f"Exit_Col_PnL_{mins}m", src)

    # External market/fundamental fallback (best-effort):
    # If these columns are still all-null after Phase 2, backfill from yfinance using trade date/ticker.
    date_col_for_ext = "date" if "date" in out.columns else ("Date" if "Date" in out.columns else None)
    ticker_col_for_ext = "ticker" if "ticker" in out.columns else ("Ticker" if "Ticker" in out.columns else None)
    ext_cols = [
        "Entry_Col_MarketCap", "Exit_Col_MarketCap",
        "Entry_Col_FloatShares", "Exit_Col_FloatShares",
        "Entry_Col_ShortInterestPctFloat", "Exit_Col_ShortInterestPctFloat",
        "Entry_Col_QQQPremarketChange_Pct", "Exit_Col_QQQPremarketChange_Pct",
        "Entry_Col_SPYPremarketChange_Pct", "Exit_Col_SPYPremarketChange_Pct",
        "Entry_Col_StockVsSPX_TodayPct", "Exit_Col_StockVsSPX_TodayPct",
    ]
    need_ext_backfill = False
    for c in ext_cols:
        if c not in out.columns:
            need_ext_backfill = True
            break
        if pd.to_numeric(out[c], errors="coerce").notna().sum() == 0:
            need_ext_backfill = True
            break

    # yfinance backfill is intentionally disabled for runtime bottleneck isolation.
    if False and need_ext_backfill and date_col_for_ext and ticker_col_for_ext:
        try:
            import logging
            from functools import lru_cache
            import yfinance as yf

            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
            dnorm = pd.to_datetime(out[date_col_for_ext], errors="coerce")
            if getattr(dnorm.dt, "tz", None) is not None:
                dnorm = dnorm.dt.tz_localize(None)
            dnorm = dnorm.dt.normalize()
            tser = out[ticker_col_for_ext].astype(str).str.upper().str.strip()

            dmin = dnorm.min()
            dmax = dnorm.max()
            if pd.notna(dmin) and pd.notna(dmax):
                start = (pd.Timestamp(dmin) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
                end = (pd.Timestamp(dmax) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

                @lru_cache(maxsize=32)
                def _daily(symbol: str) -> Optional[pd.DataFrame]:
                    try:
                        h = yf.download(
                            tickers=symbol,
                            start=start,
                            end=end,
                            interval="1d",
                            auto_adjust=False,
                            progress=False,
                            group_by="column",
                            threads=False,
                        )
                    except Exception:
                        h = None
                    if h is None or h.empty:
                        try:
                            h = yf.Ticker(symbol).history(
                                start=start, end=end, interval="1d", auto_adjust=False, actions=False
                            )
                        except Exception:
                            return None
                    if h is None or h.empty:
                        return None
                    idx = pd.to_datetime(h.index, errors="coerce")
                    try:
                        idx = idx.tz_localize(None)
                    except Exception:
                        pass
                    out_h = pd.DataFrame(index=idx.normalize())
                    for c in ("Open", "Close"):
                        if isinstance(h.columns, pd.MultiIndex):
                            lvl0 = h.columns.get_level_values(0)
                            if c in lvl0:
                                sub = h[c]
                                s = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
                                out_h[c] = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy()
                        elif c in h.columns:
                            out_h[c] = pd.to_numeric(h[c], errors="coerce")
                    return out_h if not out_h.empty else None

                spy = _daily("SPY")
                spx = _daily("^GSPC")
                qqq = _daily("QQQ")

                spy_gap = None
                spx_ret = None
                qqq_gap = None
                if spy is not None and {"Open", "Close"}.issubset(spy.columns):
                    prev = spy["Close"].shift(1)
                    spy_gap = ((spy["Open"] - prev) / prev.replace(0, np.nan)) * 100.0
                if spx is not None and "Close" in spx.columns:
                    spx_ret = spx["Close"].pct_change(fill_method=None) * 100.0
                if qqq is not None and {"Open", "Close"}.issubset(qqq.columns):
                    prev = qqq["Close"].shift(1)
                    qqq_gap = ((qqq["Open"] - prev) / prev.replace(0, np.nan)) * 100.0

                def _fill_map(col: str, series: Optional[pd.Series]) -> None:
                    if series is None:
                        return
                    vals = dnorm.map(series)
                    if col not in out.columns:
                        out[col] = vals
                    else:
                        cur = pd.to_numeric(out[col], errors="coerce")
                        out[col] = cur.where(cur.notna(), vals)

                _fill_map("Entry_Col_SPYPremarketChange_Pct", spy_gap)
                _fill_map("Exit_Col_SPYPremarketChange_Pct", spy_gap)
                _fill_map("Entry_Col_StockVsSPX_TodayPct", spx_ret)
                _fill_map("Exit_Col_StockVsSPX_TodayPct", spx_ret)
                _fill_map("Entry_Col_QQQPremarketChange_Pct", qqq_gap)
                _fill_map("Exit_Col_QQQPremarketChange_Pct", qqq_gap)
                # If SPY premarket gap is unavailable, keep compatibility by borrowing existing SPX-relative series.
                for dst, src in (
                    ("Entry_Col_SPYPremarketChange_Pct", "Entry_Col_StockVsSPX_TodayPct"),
                    ("Exit_Col_SPYPremarketChange_Pct", "Exit_Col_StockVsSPX_TodayPct"),
                ):
                    if dst in out.columns and src in out.columns:
                        cur = pd.to_numeric(out[dst], errors="coerce")
                        srcv = pd.to_numeric(out[src], errors="coerce")
                        out[dst] = cur.where(cur.notna(), srcv)

                @lru_cache(maxsize=4096)
                def _info(tkr: str) -> dict:
                    try:
                        tk = yf.Ticker(str(tkr))
                        info = getattr(tk, "info", None) or {}
                        fast = getattr(tk, "fast_info", None) or {}
                        out_d: dict = {}
                        if isinstance(info, dict):
                            out_d.update(info)
                        if isinstance(fast, dict):
                            out_d.update(fast)
                        return out_d
                    except Exception:
                        return {}

                def _first_num(d: dict, keys: tuple[str, ...]) -> float:
                    for k in keys:
                        v = d.get(k)
                        if v is None:
                            continue
                        try:
                            return float(v)
                        except Exception:
                            continue
                    return np.nan

                uniq_t = [t for t in pd.unique(tser) if isinstance(t, str) and t]
                mcap_map: dict[str, float] = {}
                f_map: dict[str, float] = {}
                spf_map: dict[str, float] = {}

                def _fetch_one(sym: str) -> tuple[str, float, float, float]:
                    info = _info(str(sym))
                    mcap = _first_num(info, ("marketCap", "market_cap"))
                    flt = _first_num(info, ("floatShares", "float_shares", "sharesFloat", "shares_float"))
                    spf = _first_num(
                        info,
                        ("shortPercentOfFloat", "short_percent_of_float", "shortPercentFloat", "short_percent_float"),
                    )
                    spf_out = spf * (100.0 if np.isfinite(spf) and spf <= 1.0 else 1.0) if np.isfinite(spf) else np.nan
                    return sym, mcap, flt, spf_out

                max_symbols = 1200  # cap for safety; current universe ~1k
                workers = min(24, max(4, (os.cpu_count() or 8)))
                try:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = [ex.submit(_fetch_one, sym) for sym in uniq_t[:max_symbols]]
                        for fut in as_completed(futs):
                            try:
                                sym, mcap, flt, spf = fut.result()
                            except Exception:
                                continue
                            mcap_map[sym] = mcap
                            f_map[sym] = flt
                            spf_map[sym] = spf
                except Exception:
                    # Fallback sequentially if thread pool fails for any reason.
                    for sym in uniq_t[:max_symbols]:
                        try:
                            sym2, mcap, flt, spf = _fetch_one(sym)
                        except Exception:
                            continue
                        mcap_map[sym2] = mcap
                        f_map[sym2] = flt
                        spf_map[sym2] = spf

                def _fill_ticker(col: str, mapping: dict[str, float]) -> None:
                    vals = tser.map(mapping)
                    if col not in out.columns:
                        out[col] = vals
                    else:
                        cur = pd.to_numeric(out[col], errors="coerce")
                        out[col] = cur.where(cur.notna(), vals)

                _fill_ticker("Entry_Col_MarketCap", mcap_map)
                _fill_ticker("Exit_Col_MarketCap", mcap_map)
                # FloatShares enrichment is intentionally disabled for now (data supplier bottleneck).
                # _fill_ticker("Entry_Col_FloatShares", f_map)
                # _fill_ticker("Exit_Col_FloatShares", f_map)
                _fill_ticker("Entry_Col_ShortInterestPctFloat", spf_map)
                _fill_ticker("Exit_Col_ShortInterestPctFloat", spf_map)

                # Causal market-cap fallback to avoid look-ahead:
                # use trade-time prices * float shares when direct marketCap is missing.
                entry_price_col = "entry_price" if "entry_price" in out.columns else ("EntryPrice" if "EntryPrice" in out.columns else None)
                exit_price_col = "exit_price" if "exit_price" in out.columns else ("ExitPrice" if "ExitPrice" in out.columns else None)
                # Float-based market-cap estimate is disabled while FloatShares feed is disabled.
        except Exception:
            # Keep enrichment robust; missing yfinance should not fail export.
            pass

    return out


def _run_phase2_continuous_tracking(
    tr_slice: pd.DataFrame,
    enriched_trades_chunk: pd.DataFrame,
    result: RunResult,
    get_bars_slice: Callable[[str, pd.Timestamp, time, time], Optional[pd.DataFrame]],
    *,
    ticker_col: str,
    entry_time_col: str,
    exit_time_col: str,
    timeline_step_seconds: int,
) -> pd.DataFrame:
    """Concatenate entry–exit bar paths per sub-chunk of trades, then attach_continuous_tracking."""
    _ts = max(1, int(timeline_step_seconds))
    chunk_trades_list: list[pd.DataFrame] = []
    n_trades = len(tr_slice)
    for start in range(0, n_trades, _CONTINUOUS_TRACKING_CHUNK_SIZE):
        end = min(start + _CONTINUOUS_TRACKING_CHUNK_SIZE, n_trades)
        tr_cc = tr_slice.iloc[start:end]
        trades_cc = enriched_trades_chunk.iloc[start:end]
        slices_cc: list[pd.DataFrame] = []
        for _, row in tr_cc.iterrows():
            ticker = str(row.get(ticker_col, row.get("ticker", ""))).strip()
            date_val = row.get("date", row.get("Date"))
            if pd.isna(date_val):
                continue
            date_ts = _calendar_date_key(date_val)
            entry_t = _parse_time_str(row.get(entry_time_col, row.get("entry_time")))
            exit_t = _parse_time_str(row.get(exit_time_col, row.get("exit_time")))
            if entry_t is None or exit_t is None:
                continue
            sl = get_bars_slice(ticker, date_ts, entry_t, exit_t)
            if sl is not None and not sl.empty:
                slices_cc.append(sl)
        el_cc = pd.concat(slices_cc, ignore_index=True) if slices_cc else pd.DataFrame()
        el_cc = _ensure_naive_datetime_column(el_cc)
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
                timeline_step_seconds=_ts,
            )
            chunk_trades_list.append(cr.trades)
        else:
            chunk_trades_list.append(trades_cc)
    return pd.concat(chunk_trades_list, ignore_index=True) if chunk_trades_list else enriched_trades_chunk


def _enrich_trades_with_long(
    tr_slice: pd.DataFrame,
    enriched_long: pd.DataFrame,
    el_index: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    result: RunResult,
    ticker_col: str,
    entry_time_col: str,
    exit_time_col: str,
    *,
    phase2_continuous: bool = True,
    timeline_step_seconds: int = 60,
) -> pd.DataFrame:
    """Apply Phase 2 enrichment for trade rows sharing one enriched_long (entry/exit; continuous if phase2_continuous)."""
    if tr_slice.empty:
        return tr_slice.drop(columns=["_year"], errors="ignore")

    if not isinstance(el_index, dict) or not el_index:
        cols = list(enriched_long.columns) if enriched_long is not None and len(enriched_long.columns) else []
        raise RuntimeError(
            "Phase 2 requires a non-empty (ticker, date) index from enriched long (_build_enriched_long_index). "
            f"Got empty or invalid el_index. enriched_long rows={len(enriched_long)}, "
            f"columns={cols!r}. Need {_EL_TICKER_COL!r} and {_EL_DATE_COL!r} or {_EL_DATETIME_COL!r}."
        )

    get_bars_slice = _bars_slice_from_index(el_index)
    enriched_trades_chunk = _vectorized_entry_exit_elite_from_index(
        tr_slice,
        el_index,
        ticker_col,
        entry_time_col,
        exit_time_col,
    )

    if not phase2_continuous:
        return _restore_orig_idx(enriched_trades_chunk, tr_slice)

    chunk_result_df = _run_phase2_continuous_tracking(
        tr_slice,
        enriched_trades_chunk,
        result,
        get_bars_slice,
        ticker_col=ticker_col,
        entry_time_col=entry_time_col,
        exit_time_col=exit_time_col,
        timeline_step_seconds=timeline_step_seconds,
    )
    return _restore_orig_idx(chunk_result_df, tr_slice)


def _process_single_enrichment_chunk(
    chunk_start: int,
    chunk_dates: list,
    tr_chunk: pd.DataFrame,
    *,
    full_year_cached: bool,
    cache_file_str: Optional[str],
    cache_path_obj: Optional[Path],
    base_cache_key: str,
    year: str,
    tickers: set,
    end_time: time,
    session_start: time,
    session_end: time,
    cleaned_year_data: dict,
    wide_path_by_year: dict,
    load_full_columns: bool,
    result: Optional[RunResult],
    ticker_col: str,
    entry_time_col: str,
    exit_time_col: str,
    inner_workers: int = 1,
    phase2_continuous: bool = True,
    timeline_step_seconds: int = 60,
) -> tuple[int, pd.DataFrame]:
    """One date-chunk: load enriched long once, then optionally split trade rows across inner_workers threads."""
    enriched_long = None
    if full_year_cached and cache_file_str:
        d_min, d_max = min(chunk_dates), max(chunk_dates)
        lo, hi = _bounds_for_parquet_date_column(cache_file_str, d_min, d_max)
        enriched_long = pd.read_parquet(
            cache_file_str,
            filters=[("Date", ">=", lo), ("Date", "<=", hi)],
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
            unchunk = unchunk.copy()
            unchunk["_orig_idx"] = tr_chunk["_orig_idx"].values
        return chunk_start, unchunk

    el_index = _build_enriched_long_index(enriched_long)
    if not el_index:
        cols = list(enriched_long.columns)
        raise RuntimeError(
            f"Phase 2: cannot build (ticker, date) index for year={year!r}, chunk_start={chunk_start}, "
            f"chunk_dates={chunk_dates[:8]!r}{'...' if len(chunk_dates) > 8 else ''}, "
            f"enriched_long rows={len(enriched_long)}, columns={cols!r}. "
            f"Requires non-empty {_EL_TICKER_COL!r} and {_EL_DATE_COL!r} or {_EL_DATETIME_COL!r} on enriched long."
        )

    iw = _safe_worker_int(inner_workers, default=1, cap=64)
    batches = _split_trade_df_for_workers(tr_chunk, iw)
    if len(batches) <= 1:
        if phase2_continuous and result is None:
            raise RuntimeError("phase2_continuous=True requires RunResult for chunk enrichment.")
        chunk_result_df = _enrich_trades_with_long(
            tr_chunk,
            enriched_long,
            el_index,
            result,
            ticker_col,
            entry_time_col,
            exit_time_col,
            phase2_continuous=phase2_continuous,
            timeline_step_seconds=timeline_step_seconds,
        )
    else:
        pool = len(batches)
        tw = max(1, int(timeline_step_seconds))

        def _enrich_batch(batch: pd.DataFrame) -> pd.DataFrame:
            if phase2_continuous and result is None:
                raise RuntimeError("phase2_continuous=True requires RunResult for chunk enrichment.")
            return _enrich_trades_with_long(
                batch,
                enriched_long,
                el_index,
                result,
                ticker_col,
                entry_time_col,
                exit_time_col,
                phase2_continuous=phase2_continuous,
                timeline_step_seconds=tw,
            )

        with ThreadPoolExecutor(max_workers=pool) as ex:
            futs = [ex.submit(_enrich_batch, b) for b in batches]
            parts = [f.result() for f in futs]
        chunk_result_df = pd.concat(parts, ignore_index=True)

    return chunk_start, chunk_result_df


def _process_single_enrichment_chunk_worker(args: tuple) -> tuple[int, pd.DataFrame]:
    return _process_single_enrichment_chunk(**args)


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
    chunk_workers: int = 1,
) -> RunResult:
    """
    Enrich trades after the backtest (Phase 2): Entry_Col_*, Exit_Col_*, and by default Continuous_Col_*.

    For each unique (year, ticker, date) in trades:
    - Load wide data (from DataFrame or Path via pd.read_parquet with filters)
    - Filter to bars from session_start through max exit_time for that (ticker, date)
    - wide_to_long, enrich_long_with_library_columns
    For each trade: vectorized Entry_* / Exit_* (same bar-pick rules as get_row_at_time) using a (ticker,date) index; fails if the index cannot be built.
    - Elite slice columns (MFE/MAE, etc.) from bars between entry and exit
    Continuous_*: ``_run_phase2_continuous_tracking`` when ``get_continuous_columns()`` is non-empty.

    chunk_workers: threads splitting **trade rows within one date-chunk** (after shared enriched_long
    is built once for that chunk). Date-chunks run **one after another** (chunk 1, then chunk 2, …).
    Final trade order is restored with _orig_idx across the full result.

    If cache_dir is set, enriched long DataFrames are saved per year (keyed by year + hash of
    date set) and reused on subsequent runs to skip wide_to_long + librarycolumn for that year.
    Handles: empty trades, missing columns, entry_time/exit_time as "H:MM" or "HH:MM".
    """
    if not getattr(config, "use_library_columns", True):
        return result
    if not has_librarycolumn():
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
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None
    tr["_orig_idx"] = np.arange(len(tr))
    total_ticker_days = int(sum(len(v) for v in by_year.values()))
    pbar = (
        tqdm(total=max(1, total_ticker_days), desc="Phase 2", unit="ticker-day")
        if (tqdm and show_progress)
        else None
    )
    per_year_enriched: list[pd.DataFrame] = []
    col_suffix = "full" if load_full_columns else "restr"
    phase2_continuous = _phase2_should_attach_continuous(tr)
    try:
        _tls = int(getattr(config, "timeline_step_seconds", 60))
    except (TypeError, ValueError):
        _tls = 60
    timeline_step_seconds_phase2 = max(1, _tls)

    try:
        year_keys = sorted(by_year.keys())
        for year in year_keys:
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

            base_cache_key = f"{year}_{_enrichment_cache_dates_tickers_id(dates, tickers)}_{col_suffix}"
            cache_file = cache_path_obj / f"{base_cache_key}.parquet" if cache_path_obj else None
            full_year_cached = cache_file is not None and cache_file.is_file()

            tr_year = tr[tr["_year"] == year]
            if tr_year.empty:
                continue

            chunk_tasks: list[tuple[int, list, pd.DataFrame]] = []
            cache_file_str = str(cache_file) if cache_file is not None else None
            for chunk_start in range(0, len(dates), CHUNK_DAYS):
                chunk_dates = dates[chunk_start : chunk_start + CHUNK_DAYS]
                if not chunk_dates:
                    continue
                chunk_dates_set = set(pd.Timestamp(d).normalize() for d in chunk_dates)
                tr_chunk = tr_year[tr_year["date"].dt.normalize().isin(chunk_dates_set)]
                if tr_chunk.empty:
                    continue
                chunk_tasks.append((chunk_start, chunk_dates, tr_chunk.copy()))

            cw = _safe_worker_int(chunk_workers, default=1, cap=64)
            year_chunk_results: list[pd.DataFrame] = []
            chunk_done_units: dict[int, int] = {}
            for cs, cds, _tc in chunk_tasks:
                cds_set = set(pd.Timestamp(d).normalize() for d in cds)
                chunk_done_units[int(cs)] = sum(
                    1 for (_tt, dd) in ticker_dates if pd.Timestamp(dd).normalize() in cds_set
                )

            proc_workers = _safe_worker_int(
                getattr(config, "phase2_process_workers", os.getenv("BT_PHASE2_PROCESS_WORKERS", 1)),
                default=1,
                cap=16,
            )
            can_process_chunks = (
                proc_workers > 1
                and len(chunk_tasks) > 1
                and not phase2_continuous
                and not isinstance(cleaned_year_data.get(year), pd.DataFrame)
                and not isinstance(cleaned_year_data.get(str(year)), pd.DataFrame)
            )

            if can_process_chunks:
                worker_args = []
                for cs, cds, tc in chunk_tasks:
                    worker_args.append(
                        dict(
                            chunk_start=cs,
                            chunk_dates=cds,
                            tr_chunk=tc,
                            full_year_cached=full_year_cached,
                            cache_file_str=cache_file_str,
                            cache_path_obj=cache_path_obj,
                            base_cache_key=base_cache_key,
                            year=year,
                            tickers=tickers,
                            end_time=end_time,
                            session_start=session_start,
                            session_end=session_end,
                            cleaned_year_data=cleaned_year_data,
                            wide_path_by_year=wide_path_by_year,
                            load_full_columns=load_full_columns,
                            result=None,
                            ticker_col=ticker_col,
                            entry_time_col=entry_time_col,
                            exit_time_col=exit_time_col,
                            inner_workers=cw,
                            phase2_continuous=False,
                            timeline_step_seconds=timeline_step_seconds_phase2,
                        )
                    )
                with ProcessPoolExecutor(max_workers=min(proc_workers, len(worker_args))) as ex:
                    futs = [ex.submit(_process_single_enrichment_chunk_worker, wa) for wa in worker_args]
                    for fut in as_completed(futs):
                        cs_done, cdf = fut.result()
                        year_chunk_results.append((int(cs_done), cdf))
                        if pbar is not None:
                            pbar.update(int(chunk_done_units.get(int(cs_done), 0)))
                year_chunk_results = [cdf for _, cdf in sorted(year_chunk_results, key=lambda x: x[0])]
            else:
                for cs, cds, tc in chunk_tasks:
                    _, cdf = _process_single_enrichment_chunk(
                        cs,
                        cds,
                        tc,
                        full_year_cached=full_year_cached,
                        cache_file_str=cache_file_str,
                        cache_path_obj=cache_path_obj,
                        base_cache_key=base_cache_key,
                        year=year,
                        tickers=tickers,
                        end_time=end_time,
                        session_start=session_start,
                        session_end=session_end,
                        cleaned_year_data=cleaned_year_data,
                        wide_path_by_year=wide_path_by_year,
                        load_full_columns=load_full_columns,
                        result=result,
                        ticker_col=ticker_col,
                        entry_time_col=entry_time_col,
                        exit_time_col=exit_time_col,
                        inner_workers=cw,
                        phase2_continuous=phase2_continuous,
                        timeline_step_seconds=timeline_step_seconds_phase2,
                    )
                    year_chunk_results.append(cdf)
                    if pbar is not None:
                        pbar.update(int(chunk_done_units.get(int(cs), 0)))

            year_enriched = pd.concat(year_chunk_results, ignore_index=True) if year_chunk_results else tr_year.drop(columns=["_year"], errors="ignore")
            if "_year" in year_enriched.columns:
                year_enriched = year_enriched.drop(columns=["_year"], errors="ignore")
            # _orig_idx already set per chunk; ensure it exists for un-enriched fallback
            if "_orig_idx" not in year_enriched.columns and "_orig_idx" in tr_year.columns:
                year_enriched["_orig_idx"] = tr_year["_orig_idx"].values
            per_year_enriched.append(year_enriched)

        # Reassemble in original trade order
        if not per_year_enriched:
            return result
        enriched_trades_with_continuous = pd.concat(per_year_enriched, ignore_index=True)
        if "_orig_idx" in enriched_trades_with_continuous.columns:
            enriched_trades_with_continuous = enriched_trades_with_continuous.sort_values("_orig_idx").drop(columns=["_orig_idx"])
        if "gst" in enriched_trades_with_continuous.columns:
            enriched_trades_with_continuous = enriched_trades_with_continuous.drop(columns=["gst"])
        enriched_trades_with_continuous = _ensure_requested_entry_exit_columns(enriched_trades_with_continuous)
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
    max_workers: int = 1,
    chunk_workers: int = 1,
) -> None:
    """
    Run Phase 2 on raw_results in place: Entry_Col_*, Exit_Col_*, and Continuous_Col_* (when tracking columns are configured).
    Use after engine.run(..., defer_column_phase=True).
    Parallelism: max_workers across accounts; chunk_workers splits trade rows within each date-chunk.
    If cache_dir is set, enriched long DataFrames are cached there per year for reuse.
    If load_full_columns=True, load all wide columns so enrichment can compute full Col_* snapshots.
    Uses more memory and disk; cache key includes 'full' vs 'restr' so they don't mix.

    max_workers: parallelism across (year, account) RunResults. 1 = sequential (default).
    chunk_workers: trade-row parallelism **inside** each date-chunk in enrich_trades_post_backtest.
    Uses threads (shared memory; good for I/O + releasing the GIL in numpy/pandas hot paths).
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
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None
    pbar = tqdm(total=n, desc="Phase 2 accounts", unit="acct") if (tqdm and n and n > 1) else None
    workers = _safe_worker_int(max_workers, default=1, cap=32)
    chunk_w = _safe_worker_int(chunk_workers, default=1, cap=64)
    try:
        if workers <= 1 or n <= 1:
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
                        show_progress=(n == 1),
                        load_full_columns=load_full_columns,
                        chunk_workers=chunk_w,
                    )
                    raw_results[year][acct] = enriched
                    if pbar is not None:
                        pbar.update(1)
        else:
            workers = min(workers, n)
            future_to_key: dict = {}

            def _enrich_one_account(run_res: RunResult) -> RunResult:
                return enrich_trades_post_backtest(
                    run_res,
                    cleaned_year_data,
                    wide_path_by_year,
                    start,
                    end,
                    config,
                    cache_dir=cache_dir,
                    show_progress=False,
                    load_full_columns=load_full_columns,
                    chunk_workers=chunk_w,
                )

            with ThreadPoolExecutor(max_workers=workers) as ex:
                for year, by_acct in raw_results.items():
                    for acct, res in by_acct.items():
                        fut = ex.submit(_enrich_one_account, res)
                        future_to_key[fut] = (year, acct)
                for fut in as_completed(future_to_key):
                    year, acct = future_to_key[fut]
                    raw_results[year][acct] = fut.result()
                    if pbar is not None:
                        pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
