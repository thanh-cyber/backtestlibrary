from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


def _normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    out = series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(out, errors="coerce")


def _require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


_TIME_COL_RE = re.compile(r"^(?:\d{1,2}:\d{2}(?::\d{2})?)$")


def _normalize_time_label(label: str) -> str:
    s = str(label).strip()
    if _TIME_COL_RE.match(s):
        parts = s.split(":")
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if len(parts) == 3 else 0
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return s


def _extract_time_columns(columns: Iterable[object]) -> list[str]:
    out: list[str] = []
    for c in columns:
        cs = str(c).strip()
        if _TIME_COL_RE.match(cs):
            out.append(cs)
    return out


def _extract_volume_time_map(
    columns: Iterable[object],
    *,
    prefixes: tuple[str, ...] = ("Vol", "Volume"),
) -> dict[str, str]:
    vol_map: dict[str, str] = {}
    for c in columns:
        cs = str(c).strip()
        for pref in prefixes:
            pref_re = rf"^{re.escape(pref)}\s+(\d{{1,2}}:\d{{2}}(?::\d{{2}})?)$"
            m = re.match(pref_re, cs, flags=re.IGNORECASE)
            if m:
                vol_map[_normalize_time_label(m.group(1))] = cs
                break
    return vol_map


def _daily_from_long_intraday(
    df: pd.DataFrame,
    *,
    ticker_col: str,
    timestamp_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
) -> pd.DataFrame:
    df = df[[ticker_col, timestamp_col, high_col, low_col, close_col, volume_col]].copy()
    df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[high_col] = _to_numeric(df[high_col])
    df[low_col] = _to_numeric(df[low_col])
    df[close_col] = _to_numeric(df[close_col])
    df[volume_col] = _to_numeric(df[volume_col]).fillna(0.0)

    df = df.dropna(subset=[ticker_col, timestamp_col, high_col, low_col, close_col])
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Date", "DailyHigh", "DailyLow", "DailyClose", "DayVolume", "DayPV"])

    df = df.sort_values([ticker_col, timestamp_col])
    df["Date"] = df[timestamp_col].dt.normalize()
    df["_pv"] = df[close_col] * df[volume_col]

    daily = (
        df.groupby([ticker_col, "Date"], as_index=False)
        .agg(
            DailyHigh=(high_col, "max"),
            DailyLow=(low_col, "min"),
            DailyClose=(close_col, "last"),
            DayVolume=(volume_col, "sum"),
            DayPV=("_pv", "sum"),
        )
        .sort_values([ticker_col, "Date"])
    )
    return daily.rename(columns={ticker_col: "Ticker"})


def _daily_from_wide_intraday(
    wide_df: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    price_time_cols: list[str] | None = None,
    volume_time_prefixes: tuple[str, ...] = ("Vol", "Volume"),
) -> pd.DataFrame:
    _require_columns(wide_df, [ticker_col, date_col], "wide_df")
    ticker_series = _normalize_ticker(wide_df[ticker_col])
    date_series = _normalize_date(wide_df[date_col])

    time_cols = price_time_cols or _extract_time_columns(wide_df.columns)
    if not time_cols:
        raise ValueError("No minute price columns detected in wide_df.")

    # Sort columns by actual time value to preserve session order.
    def _to_dt(col_name: str) -> datetime:
        parts = _normalize_time_label(col_name).split(":")
        return datetime(2000, 1, 1, int(parts[0]), int(parts[1]), int(parts[2]))

    time_cols = sorted(time_cols, key=_to_dt)

    price_matrix = wide_df[time_cols].apply(_to_numeric).to_numpy(dtype=np.float32, na_value=np.nan)
    valid_price = np.isfinite(price_matrix)

    # Daily OHLC approximation from one-price-per-minute series.
    daily_high = np.where(valid_price.any(axis=1), np.nanmax(price_matrix, axis=1), np.nan)
    daily_low = np.where(valid_price.any(axis=1), np.nanmin(price_matrix, axis=1), np.nan)

    # last valid minute price per row
    last_idx = np.where(valid_price.any(axis=1), valid_price.shape[1] - 1 - np.argmax(valid_price[:, ::-1], axis=1), -1)
    daily_close = np.where(
        last_idx >= 0,
        price_matrix[np.arange(len(wide_df)), np.clip(last_idx, 0, price_matrix.shape[1] - 1)],
        np.nan,
    )

    vol_map = _extract_volume_time_map(wide_df.columns, prefixes=volume_time_prefixes)
    vol_cols = [vol_map[_normalize_time_label(c)] for c in time_cols if _normalize_time_label(c) in vol_map]
    if vol_cols:
        vol_matrix = wide_df[vol_cols].apply(_to_numeric).fillna(0.0).to_numpy(dtype=np.float32)
        # align to price columns that had mapped volume
        aligned_price_cols = [c for c in time_cols if _normalize_time_label(c) in vol_map]
        aligned_price_matrix = wide_df[aligned_price_cols].apply(_to_numeric).to_numpy(dtype=np.float32, na_value=np.nan)
        day_volume = vol_matrix.sum(axis=1)
        day_pv = np.nansum(aligned_price_matrix * vol_matrix, axis=1)
        del vol_matrix, aligned_price_matrix
    else:
        # No volume columns: fallback to simple average of valid prices.
        day_volume = valid_price.sum(axis=1).astype(float)
        day_pv = np.nansum(price_matrix, axis=1)
    del valid_price, price_matrix

    out = pd.DataFrame(
        {
            "Ticker": ticker_series,
            "Date": date_series,
            "DailyHigh": daily_high,
            "DailyLow": daily_low,
            "DailyClose": daily_close,
            "DayVolume": day_volume,
            "DayPV": day_pv,
        }
    )
    out = out.dropna(subset=["Ticker", "Date", "DailyHigh", "DailyLow", "DailyClose"])
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _finalize_daily_metrics(daily: pd.DataFrame, *, atr_period: int = 14) -> pd.DataFrame:
    if atr_period <= 0:
        raise ValueError("atr_period must be > 0")
    if daily.empty:
        return pd.DataFrame(columns=["Ticker", "Date", "Col_ATR14", "Col_VWAP"])

    daily = daily.copy().sort_values(["Ticker", "Date"])
    daily["Col_VWAP"] = np.where(
        daily["DayVolume"] > 0,
        daily["DayPV"] / daily["DayVolume"],
        daily["DailyClose"],
    )

    prev_close = daily.groupby("Ticker")["DailyClose"].shift(1)
    tr1 = daily["DailyHigh"] - daily["DailyLow"]
    tr2 = (daily["DailyHigh"] - prev_close).abs()
    tr3 = (daily["DailyLow"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    daily["Col_ATR14"] = (
        tr.groupby(daily["Ticker"])
        .transform(lambda s: s.ewm(alpha=1.0 / float(atr_period), adjust=False, min_periods=1).mean())
        .astype(float)
    )
    return daily[["Ticker", "Date", "Col_ATR14", "Col_VWAP"]]


def build_daily_exit_metrics_from_minute_bars(
    minute_bars: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    timestamp_col: str = "Timestamp",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    atr_period: int = 14,
) -> pd.DataFrame:
    """Build daily Col_ATR14 and Col_VWAP from minute OHLCV bars.

    Returns one row per (Ticker, Date):
      - Col_ATR14: Wilder ATR over daily bars (from minute-derived daily OHLC).
      - Col_VWAP: Session VWAP from minute close*volume / volume.
    """
    _require_columns(
        minute_bars,
        [ticker_col, timestamp_col, high_col, low_col, close_col, volume_col],
        "minute_bars",
    )
    daily = _daily_from_long_intraday(
        minute_bars,
        ticker_col=ticker_col,
        timestamp_col=timestamp_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
    )
    return _finalize_daily_metrics(daily, atr_period=atr_period)


def build_daily_exit_metrics_from_wide_minute_columns(
    wide_df: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    price_time_cols: list[str] | None = None,
    volume_time_prefixes: tuple[str, ...] = ("Vol", "Volume"),
    atr_period: int = 14,
) -> pd.DataFrame:
    """Build daily Col_ATR14 and Col_VWAP from wide minute columns.

    Expected shape: one row per ticker/date with minute price columns like
    '4:00', '4:01', ... and optional volume columns like 'Vol 4:00', 'Vol 4:01'.
    """
    daily = _daily_from_wide_intraday(
        wide_df,
        ticker_col=ticker_col,
        date_col=date_col,
        price_time_cols=price_time_cols,
        volume_time_prefixes=volume_time_prefixes,
    )
    return _finalize_daily_metrics(daily, atr_period=atr_period)


def build_daily_exit_metrics(
    intraday_df: pd.DataFrame,
    *,
    mode: str = "auto",
    atr_period: int = 14,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    timestamp_col: str = "Timestamp",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    wide_ticker_col: str | None = None,
    price_time_cols: list[str] | None = None,
    volume_time_prefixes: tuple[str, ...] = ("Vol", "Volume"),
) -> pd.DataFrame:
    """Build daily Col_ATR14/Col_VWAP from either long OHLCV or wide minute data.

    mode:
      - 'auto': infer from available columns
      - 'long': require long OHLCV minute bars
      - 'wide': require one-price-per-minute columns (optional per-minute volume columns)
    """
    mode_norm = mode.lower().strip()
    if mode_norm not in {"auto", "long", "wide"}:
        raise ValueError("mode must be one of: auto, long, wide")

    long_ready = all(c in intraday_df.columns for c in (ticker_col, timestamp_col, high_col, low_col, close_col, volume_col))
    wide_ready = (
        (wide_ticker_col or ticker_col) in intraday_df.columns
        and date_col in intraday_df.columns
        and bool(price_time_cols or _extract_time_columns(intraday_df.columns))
    )

    if mode_norm == "long":
        return build_daily_exit_metrics_from_minute_bars(
            intraday_df,
            ticker_col=ticker_col,
            timestamp_col=timestamp_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            atr_period=atr_period,
        )
    if mode_norm == "wide":
        return build_daily_exit_metrics_from_wide_minute_columns(
            intraday_df,
            ticker_col=wide_ticker_col or ticker_col,
            date_col=date_col,
            price_time_cols=price_time_cols,
            volume_time_prefixes=volume_time_prefixes,
            atr_period=atr_period,
        )

    if long_ready:
        return build_daily_exit_metrics_from_minute_bars(
            intraday_df,
            ticker_col=ticker_col,
            timestamp_col=timestamp_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            atr_period=atr_period,
        )
    if wide_ready:
        return build_daily_exit_metrics_from_wide_minute_columns(
            intraday_df,
            ticker_col=wide_ticker_col or ticker_col,
            date_col=date_col,
            price_time_cols=price_time_cols,
            volume_time_prefixes=volume_time_prefixes,
            atr_period=atr_period,
        )
    raise ValueError(
        "Could not auto-detect intraday format. Provide long OHLCV minute bars or wide minute columns."
    )


def merge_exit_metrics_into_backtest_data(
    backtest_df: pd.DataFrame,
    daily_exit_metrics: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Merge Col_ATR14 and Col_VWAP into the backtest dataframe used by engine.run()."""
    _require_columns(backtest_df, [ticker_col, date_col], "backtest_df")
    _require_columns(daily_exit_metrics, ["Ticker", "Date", "Col_ATR14", "Col_VWAP"], "daily_exit_metrics")

    left = backtest_df.copy()
    left_keys = pd.MultiIndex.from_arrays(
        [_normalize_ticker(left[ticker_col]), _normalize_date(left[date_col])]
    )

    right = daily_exit_metrics[["Ticker", "Date", "Col_ATR14", "Col_VWAP"]].copy()
    right["Ticker"] = _normalize_ticker(right["Ticker"])
    right["Date"] = _normalize_date(right["Date"])
    right = right.dropna(subset=["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"], keep="last")
    right = right.set_index(["Ticker", "Date"])

    mapped = right.reindex(left_keys)
    for col in ("Col_ATR14", "Col_VWAP"):
        metric_values = mapped[col].to_numpy()
        if col not in left.columns:
            left[col] = metric_values
        elif overwrite:
            left[col] = np.where(pd.notna(metric_values), metric_values, left[col])
        else:
            left[col] = left[col].where(pd.notna(left[col]), metric_values)
    return left


def attach_exit_metrics_from_minute_bars(
    cleaned_year_data: dict[str, pd.DataFrame],
    minute_bars: pd.DataFrame,
    *,
    backtest_ticker_col: str = "Ticker",
    backtest_date_col: str = "Date",
    minute_ticker_col: str = "Ticker",
    minute_timestamp_col: str = "Timestamp",
    minute_high_col: str = "High",
    minute_low_col: str = "Low",
    minute_close_col: str = "Close",
    minute_volume_col: str = "Volume",
    atr_period: int = 14,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute Col_ATR14/Col_VWAP from minute bars and merge into each year dataframe."""
    daily_metrics = build_daily_exit_metrics_from_minute_bars(
        minute_bars,
        ticker_col=minute_ticker_col,
        timestamp_col=minute_timestamp_col,
        high_col=minute_high_col,
        low_col=minute_low_col,
        close_col=minute_close_col,
        volume_col=minute_volume_col,
        atr_period=atr_period,
    )

    out: dict[str, pd.DataFrame] = {}
    for year, df_year in cleaned_year_data.items():
        out[year] = merge_exit_metrics_into_backtest_data(
            df_year,
            daily_metrics,
            ticker_col=backtest_ticker_col,
            date_col=backtest_date_col,
            overwrite=overwrite,
        )
    return out


def attach_exit_metrics_from_intraday_data(
    cleaned_year_data: dict[str, pd.DataFrame],
    intraday_df: pd.DataFrame | None = None,
    *,
    mode: str = "auto",
    backtest_ticker_col: str = "Ticker",
    backtest_date_col: str = "Date",
    minute_ticker_col: str = "Ticker",
    minute_timestamp_col: str = "Timestamp",
    minute_high_col: str = "High",
    minute_low_col: str = "Low",
    minute_close_col: str = "Close",
    minute_volume_col: str = "Volume",
    wide_ticker_col: str = "Ticker",
    wide_date_col: str = "Date",
    wide_price_time_cols: list[str] | None = None,
    wide_volume_time_prefixes: tuple[str, ...] = ("Vol", "Volume"),
    atr_period: int = 14,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute Col_ATR14/Col_VWAP from long or wide intraday data and merge into each year dataframe.

    Memory optimization: if intraday_df is None, metrics are computed per-year
    directly from each year's dataframe to avoid creating one large combined table.
    """
    out: dict[str, pd.DataFrame] = {}
    if intraday_df is None:
        for year, df_year in cleaned_year_data.items():
            daily_metrics = build_daily_exit_metrics(
                df_year,
                mode=mode,
                atr_period=atr_period,
                ticker_col=minute_ticker_col,
                date_col=wide_date_col,
                timestamp_col=minute_timestamp_col,
                high_col=minute_high_col,
                low_col=minute_low_col,
                close_col=minute_close_col,
                volume_col=minute_volume_col,
                wide_ticker_col=wide_ticker_col,
                price_time_cols=wide_price_time_cols,
                volume_time_prefixes=wide_volume_time_prefixes,
            )
            out[year] = merge_exit_metrics_into_backtest_data(
                df_year,
                daily_metrics,
                ticker_col=backtest_ticker_col,
                date_col=backtest_date_col,
                overwrite=overwrite,
            )
        return out

    daily_metrics = build_daily_exit_metrics(
        intraday_df,
        mode=mode,
        atr_period=atr_period,
        ticker_col=minute_ticker_col,
        date_col=wide_date_col,
        timestamp_col=minute_timestamp_col,
        high_col=minute_high_col,
        low_col=minute_low_col,
        close_col=minute_close_col,
        volume_col=minute_volume_col,
        wide_ticker_col=wide_ticker_col,
        price_time_cols=wide_price_time_cols,
        volume_time_prefixes=wide_volume_time_prefixes,
    )
    for year, df_year in cleaned_year_data.items():
        out[year] = merge_exit_metrics_into_backtest_data(
            df_year,
            daily_metrics,
            ticker_col=backtest_ticker_col,
            date_col=backtest_date_col,
            overwrite=overwrite,
        )
    return out
