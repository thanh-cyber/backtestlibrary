"""Unit tests for data module helpers."""
from __future__ import annotations

import pandas as pd

from backtestlibrary import data


class TestStandardMinuteColumns:
    """Unit tests for _standard_minute_columns."""

    def test_default_session(self):
        cols = data._standard_minute_columns()
        assert cols[0] == "9:30"
        assert cols[-1] == "16:00"
        assert len(cols) == 391

    def test_custom_session(self):
        cols = data._standard_minute_columns((10, 0), (11, 0))
        assert cols[0] == "10:00"
        assert cols[-1] == "11:00"
        assert len(cols) == 61


class TestFilterWideToStandardColumns:
    """Unit tests for _filter_wide_to_standard_columns."""

    def test_empty_returns_empty(self):
        wide = pd.DataFrame()
        standard = data._standard_minute_columns()
        out = data._filter_wide_to_standard_columns(wide, standard)
        assert out.empty

    def test_keeps_ticker_date_and_standard_cols(self):
        wide = pd.DataFrame({
            "Ticker": ["AAPL"],
            "Date": [pd.Timestamp("2022-01-03")],
            "9:30": [100.0],
            "10:00": [101.0],
            "extra_col": [999],
        })
        standard = ["9:30", "9:31", "10:00"]
        out = data._filter_wide_to_standard_columns(wide, standard)
        assert "Ticker" in out.columns
        assert "Date" in out.columns
        assert "9:30" in out.columns
        assert "10:00" in out.columns
        assert "extra_col" not in out.columns


class TestPivotLongToWide:
    """Unit tests for _pivot_long_to_wide."""

    def test_empty_returns_empty(self):
        df = pd.DataFrame()
        out = data._pivot_long_to_wide(
            df,
            ticker_col="ticker",
            date_col="date",
            time_col="time",
            price_col="close",
            volume_col="volume",
        )
        assert out.empty
        assert list(out.columns) == ["Ticker", "Date"]

    def test_pivots_correctly(self):
        df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": ["2022-01-03", "2022-01-03", "2022-01-03"],
            "time": ["9:30", "9:31", "9:32"],
            "close": [100.0, 101.0, 102.0],
        })
        out = data._pivot_long_to_wide(
            df,
            ticker_col="ticker",
            date_col="date",
            time_col="time",
            price_col="close",
            volume_col="volume",
        )
        assert "Ticker" in out.columns
        assert "Date" in out.columns
        assert "9:30" in out.columns
        assert "9:31" in out.columns
        assert "9:32" in out.columns
        assert len(out) == 1
        assert out["9:30"].iloc[0] == 100.0
