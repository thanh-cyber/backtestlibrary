"""Tests for Phase 2 trade enrichment helpers (continuous gating, cache ids)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from backtestlibrary.trade_enrichment import (
    _build_enriched_long_index,
    _calendar_date_key,
    _enrichment_cache_dates_tickers_id,
    _ensure_naive_datetime_column,
    _phase2_should_attach_continuous,
    _pick_bar_row_indices,
)


def test_cache_id_differs_when_tickers_differ():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    a = _enrichment_cache_dates_tickers_id(dates, {"AAA", "BBB"})
    b = _enrichment_cache_dates_tickers_id(dates, {"ZZZ"})
    assert a != b


def test_cache_id_stable_for_ticker_order():
    dates = [pd.Timestamp("2024-01-02")]
    a = _enrichment_cache_dates_tickers_id(dates, {"b", "a"})
    b = _enrichment_cache_dates_tickers_id(dates, {"a", "b"})
    assert a == b


def test_phase2_continuous_when_tracking_columns_configured():
    tr = pd.DataFrame({"ticker": ["X"]})
    with patch(
        "backtestlibrary.trade_enrichment.get_continuous_columns",
        return_value=["Col_Test"],
    ):
        assert _phase2_should_attach_continuous(tr) is True


def test_phase2_continuous_false_when_no_tracking_columns():
    tr = pd.DataFrame({"x": [1]})
    with patch("backtestlibrary.trade_enrichment.get_continuous_columns", return_value=[]):
        assert _phase2_should_attach_continuous(tr) is False


def test_phase2_continuous_false_empty_trades():
    tr = pd.DataFrame()
    with patch(
        "backtestlibrary.trade_enrichment.get_continuous_columns",
        return_value=["Col_Test"],
    ):
        assert _phase2_should_attach_continuous(tr) is False


def test_pick_bar_row_indices_exact_nearest_and_cap():
    """Aligned with get_row_at_time: exact bar, else nearest within 5 minutes, else -1."""
    bar_mins = np.array([570.0, 571.0, 580.0])
    targets = np.array([571.0, 572.0, 600.0, np.nan])
    ix = _pick_bar_row_indices(targets, bar_mins)
    assert ix.tolist() == [1, 1, -1, -1]


def test_calendar_date_key_matches_naive_and_tz_aware():
    assert _calendar_date_key("2022-12-02") == _calendar_date_key(
        pd.Timestamp("2022-12-02 00:00:00-05:00")
    )


def test_build_enriched_long_index_key_tz_naive_lookup():
    """Enriched Date tz-aware + trade naive same calendar day must share one index bucket."""
    df = pd.DataFrame(
        {
            "Ticker": ["ZZ"],
            "Date": [pd.Timestamp("2022-12-02 00:00:00-05:00")],
            "datetime": [pd.Timestamp("2022-12-02 10:00:00")],
            "Col_ATR14": [1.5],
        }
    )
    idx = _build_enriched_long_index(df)
    assert ("ZZ", pd.Timestamp("2022-12-02")) in idx
    assert len(idx) == 1


def test_ensure_naive_datetime_column_strips_tz():
    ts = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
    df = pd.DataFrame({"datetime": [ts], "x": [1.0]})
    out = _ensure_naive_datetime_column(df)
    assert out is not df
    got = pd.to_datetime(out["datetime"].iloc[0])
    assert getattr(got, "tzinfo", None) is None
