"""Unit tests for feed module: validate_feed, normalize_feed, PandasDataFeed, resample_wide_intraday."""
from __future__ import annotations

import pytest
import pandas as pd

from backtestlibrary.feed import (
    REQUIRED_FEED_COLUMNS,
    DataFeedConfig,
    PandasDataFeed,
    normalize_feed,
    resample_wide_intraday,
    validate_feed,
)


def _minute_wide_df(rows: list[tuple[str, str, dict]]) -> pd.DataFrame:
    """Build a wide-format feed DataFrame. rows: [(ticker, date_str, {time_col: price, ...})]."""
    out = []
    for ticker, date_str, prices in rows:
        row = {"Ticker": ticker, "Date": pd.Timestamp(date_str)}
        row.update(prices)
        out.append(row)
    return pd.DataFrame(out)


class TestValidateFeed:
    def test_valid_feed_passes(self):
        df = _minute_wide_df([("AAPL", "2022-01-03", {"9:30": 100.0, "9:31": 101.0})])
        validate_feed(df)

    def test_missing_date_raises(self):
        df = pd.DataFrame({"Ticker": ["A"], "9:30": [100.0]})
        with pytest.raises(ValueError, match="Date"):
            validate_feed(df)

    def test_missing_ticker_raises(self):
        df = pd.DataFrame({"Date": [pd.Timestamp("2022-01-03")], "9:30": [100.0]})
        with pytest.raises(ValueError, match="Ticker"):
            validate_feed(df)

    def test_no_time_columns_raises(self):
        df = pd.DataFrame({"Date": [pd.Timestamp("2022-01-03")], "Ticker": ["A"]})
        with pytest.raises(ValueError, match="time-column"):
            validate_feed(df)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_feed(pd.DataFrame())

    def test_not_dataframe_raises(self):
        with pytest.raises(ValueError, match="DataFrame"):
            validate_feed(None)


class TestNormalizeFeed:
    def test_normalizes_date_and_sorts(self):
        df = _minute_wide_df([
            ("B", "2022-01-04", {"9:30": 1.0}),
            ("A", "2022-01-03", {"9:30": 1.0}),
        ])
        out = normalize_feed(df)
        assert out["Date"].iloc[0] <= out["Date"].iloc[1]
        assert list(out["Ticker"].iloc[:2]) == ["A", "B"]

    def test_uppercases_ticker(self):
        df = _minute_wide_df([("aapl", "2022-01-03", {"9:30": 100.0})])
        out = normalize_feed(df)
        assert out["Ticker"].iloc[0] == "AAPL"


class TestPandasDataFeed:
    def test_single_df_splits_by_year(self):
        df = _minute_wide_df([
            ("A", "2022-01-03", {"9:30": 1.0}),
            ("A", "2023-01-03", {"9:30": 2.0}),
        ])
        feed = PandasDataFeed(df)
        out = feed.to_cleaned_year_data()
        assert "2022" in out and "2023" in out
        assert len(out["2022"]) == 1 and len(out["2023"]) == 1

    def test_dict_by_year_returns_same_keys(self):
        df22 = _minute_wide_df([("A", "2022-01-03", {"9:30": 1.0})])
        df23 = _minute_wide_df([("A", "2023-01-03", {"9:30": 1.0})])
        feed = PandasDataFeed({"2022": df22, "2023": df23})
        out = feed.to_cleaned_year_data()
        assert set(out.keys()) == {"2022", "2023"}

    def test_resample_config_applied(self):
        # 1m columns 9:30..9:34 -> 5m should give 9:30 bar (last = 9:34 value)
        row = {"Ticker": "A", "Date": pd.Timestamp("2022-01-03")}
        for m in range(30, 35):
            row[f"9:{m:02d}"] = 100.0 + m  # 9:34 = 134
        df = pd.DataFrame([row])
        feed = PandasDataFeed(df, config=DataFeedConfig(resample_minutes=5))
        out = feed.to_cleaned_year_data()
        assert "2022" in out
        resampled = out["2022"]
        assert "9:30" in resampled.columns
        assert "9:31" not in resampled.columns
        assert resampled["9:30"].iloc[0] == 134.0  # last of 9:30..9:34


class TestResampleWideIntraday:
    def test_5m_reduces_columns(self):
        row = {"Ticker": "A", "Date": pd.Timestamp("2022-01-03")}
        for m in range(30, 60):
            row[f"9:{m:02d}"] = 100.0
        df = pd.DataFrame([row])
        out = resample_wide_intraday(df, rule_minutes=5, session_start=(9, 30), session_end=(10, 0))
        # 9:30 to 10:00 -> bars 9:30, 9:35, 9:40, 9:45, 9:50, 9:55, 10:00 (7 bars)
        assert "9:30" in out.columns and "9:35" in out.columns
        assert "9:31" not in out.columns
        assert len([c for c in out.columns if _is_time_like(c)]) == 7

    def test_price_last_is_close(self):
        row = {"Ticker": "A", "Date": pd.Timestamp("2022-01-03")}
        for m in range(30, 35):
            row[f"9:{m:02d}"] = 100.0 + m  # 9:34 = 134
        df = pd.DataFrame([row])
        out = resample_wide_intraday(df, rule_minutes=5, price_agg="last")
        assert out["9:30"].iloc[0] == 134.0

    def test_volume_sum_aggregated(self):
        row = {"Ticker": "A", "Date": pd.Timestamp("2022-01-03")}
        for m in range(30, 35):
            row[f"9:{m:02d}"] = 100.0
            row[f"Vol 9:{m:02d}"] = 10.0
        df = pd.DataFrame([row])
        out = resample_wide_intraday(df, rule_minutes=5, session_start=(9, 30), session_end=(9, 35))
        assert "Vol 9:30" in out.columns
        assert out["Vol 9:30"].iloc[0] == 50.0  # 5 * 10

    def test_empty_returns_copy(self):
        df = pd.DataFrame(columns=["Date", "Ticker", "9:30"])
        out = resample_wide_intraday(df, rule_minutes=5)
        assert out.empty and out is not df

    def test_rule_one_preserves_values(self):
        df = _minute_wide_df([("A", "2022-01-03", {"9:30": 100.0, "9:31": 101.0})])
        out = resample_wide_intraday(df, rule_minutes=1, session_start=(9, 30), session_end=(9, 32))
        assert "9:30" in out.columns and "9:31" in out.columns
        assert out["9:30"].iloc[0] == 100.0 and out["9:31"].iloc[0] == 101.0


def _is_time_like(c: str) -> bool:
    import re
    return bool(re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", str(c)))
