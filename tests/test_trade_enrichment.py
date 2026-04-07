"""Tests for Phase 2 trade enrichment helpers and side inference."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from backtestlibrary.trade_enrichment import (
    _apply_elite_exit_from_slice,
    _build_enriched_long_index,
    _calendar_date_key,
    _enrichment_cache_dates_tickers_id,
    _ensure_naive_datetime_column,
    _get_enrich_columns_for_year_static,
    _infer_trade_side,
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


def test_pick_bar_row_indices_unsorted_bar_minutes_fallback():
    bar_mins = np.array([580.0, 570.0, 571.0])  # intentionally unsorted
    targets = np.array([572.0, 580.0])
    ix = _pick_bar_row_indices(targets, bar_mins)
    assert ix.tolist() == [2, 0]


def test_pick_bar_row_indices_sorted_fast_path_parity_with_dense_reference():
    rng = np.random.default_rng(42)

    def dense_reference(target_mins: np.ndarray, bar_mins: np.ndarray, max_delta: int = 5) -> np.ndarray:
        out = np.full(len(target_mins), -1, dtype=np.int64)
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
        picked = np.where(has_exact, j_exact, np.where(mn <= float(max_delta), j_min, -1)).astype(np.int64)
        out[np.flatnonzero(finite)] = picked
        return out

    for _ in range(100):
        n_bars = int(rng.integers(1, 80))
        n_targets = int(rng.integers(1, 120))
        bar_mins = np.sort(rng.integers(570, 960, size=n_bars).astype(np.float64))
        # inject some duplicates to mirror realistic intraday feeds
        if n_bars > 4 and rng.random() < 0.5:
            dup_i = int(rng.integers(1, n_bars - 1))
            bar_mins[dup_i] = bar_mins[dup_i - 1]
        targets = rng.integers(560, 970, size=n_targets).astype(np.float64)
        if n_targets > 5:
            nan_idx = rng.choice(n_targets, size=max(1, n_targets // 10), replace=False)
            targets[nan_idx] = np.nan
        got = _pick_bar_row_indices(targets, bar_mins, max_delta=5)
        exp = dense_reference(targets, bar_mins, max_delta=5)
        assert np.array_equal(got, exp)


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


def test_build_enriched_long_index_sorts_each_day_chronologically():
    df = pd.DataFrame(
        {
            "Ticker": ["ZZ", "ZZ", "ZZ"],
            "Date": [pd.Timestamp("2022-12-02")] * 3,
            "datetime": [
                pd.Timestamp("2022-12-02 10:05:00"),
                pd.Timestamp("2022-12-02 09:35:00"),
                pd.Timestamp("2022-12-02 09:55:00"),
            ],
            "Close": [3.0, 1.0, 2.0],
        }
    )
    idx = _build_enriched_long_index(df)
    day = idx[("ZZ", pd.Timestamp("2022-12-02"))]
    assert day["datetime"].tolist() == sorted(df["datetime"].tolist())
    assert day["Close"].tolist() == [1.0, 2.0, 3.0]


def test_ensure_naive_datetime_column_strips_tz():
    ts = pd.Timestamp("2024-01-02 10:00", tz="US/Eastern")
    df = pd.DataFrame({"datetime": [ts], "x": [1.0]})
    out = _ensure_naive_datetime_column(df)
    assert out is not df
    got = pd.to_datetime(out["datetime"].iloc[0])
    assert getattr(got, "tzinfo", None) is None


def test_infer_trade_side_detects_short_from_gross_pnl():
    trade = {
        "shares": 100,
        "entry_price": 10.0,
        "exit_price": 9.0,
        "gross_pnl": 100.0,  # short formula: shares * (entry - exit)
    }
    assert _infer_trade_side(trade) == "short"


def test_apply_elite_exit_short_vwap_deviation_sign():
    # Exit bar vwap=9.5, exit=9.0, atr=1.0 -> short deviation should be +0.5
    slice_df = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-01-02 09:30"), pd.Timestamp("2026-01-02 09:31")],
            "Close": [10.0, 9.0],
            "Col_ATR14": [1.0, 1.0],
            "Col_VWAP": [9.8, 9.5],
        }
    )
    trade = {"shares": 100, "gross_pnl": 100.0}
    _apply_elite_exit_from_slice(
        trade,
        slice_df,
        entry_price=10.0,
        exit_price=9.0,
        net_pnl=80.0,
        side="short",
    )
    assert trade["Exit_Col_ExitVWAPDeviation_ATR"] == 0.5


def test_get_enrich_columns_handles_dataframe_year_map_without_truthiness_error():
    cleaned = {"2026": pd.DataFrame({"Ticker": ["AAPL"], "Date": [pd.Timestamp("2026-01-02")]})}
    out = _get_enrich_columns_for_year_static("2026", cleaned, {})
    assert out is None


def test_infer_trade_side_ambiguous_near_zero_uses_price_direction():
    trade = {
        "shares": 100,
        "entry_price": 10.0,
        "exit_price": 9.99,
        "gross_pnl": 0.0,  # near-flat after costs can look ambiguous
    }
    assert _infer_trade_side(trade) == "short"
