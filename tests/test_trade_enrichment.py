from __future__ import annotations

import pandas as pd

from backtestlibrary.trade_enrichment import (
    _apply_elite_exit_from_slice,
    _get_enrich_columns_for_year_static,
    _infer_trade_side,
)


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
    assert trade["Col_ExitVWAPDeviation_ATR"] == 0.5


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
