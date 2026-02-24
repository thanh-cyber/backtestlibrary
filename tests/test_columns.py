"""Tests for columns module (entry/exit/continuous from librarycolumn)."""
from __future__ import annotations

import pandas as pd

from backtestlibrary.columns import (
    apply_entry_columns,
    apply_exit_columns,
    get_entry_columns,
    get_exit_columns,
    get_continuous_columns,
    has_librarycolumn,
)


class TestColumnsModule:
    """Test column getters and apply_exit_columns."""

    def test_get_exit_columns_returns_list(self):
        """Without librarycolumn, returns default exit columns."""
        cols = get_exit_columns()
        assert isinstance(cols, list)
        assert "Col_ATR14" in cols
        assert "Col_VWAP" in cols

    def test_get_entry_columns_returns_list(self):
        """get_entry_columns returns a list."""
        cols = get_entry_columns()
        assert isinstance(cols, list)

    def test_get_continuous_columns_returns_list(self):
        """get_continuous_columns returns a list."""
        cols = get_continuous_columns()
        assert isinstance(cols, list)

    def test_apply_entry_columns_adds_entry_prefix(self):
        """apply_entry_columns adds Entry_Col_X when column is in get_entry_columns()."""
        trade_dict = {"ticker": "AAPL"}
        row = pd.Series({"Col_ATR14": 1.5, "Col_VWAP": 100.2})
        apply_entry_columns(trade_dict, row)
        entry_cols = get_entry_columns()
        if "Col_ATR14" in entry_cols:
            assert trade_dict.get("Entry_Col_ATR14") == 1.5
        if "Col_VWAP" in entry_cols:
            assert trade_dict.get("Entry_Col_VWAP") == 100.2

    def test_apply_exit_columns_adds_exit_suffix(self):
        """apply_exit_columns adds Col_X_Exit from row to trade_dict."""
        trade_dict = {"ticker": "AAPL", "net_pnl": 100.0}
        row = pd.Series({"Col_ATR14": 1.5, "Col_VWAP": 100.2, "other": 99})
        apply_exit_columns(trade_dict, row)
        assert trade_dict.get("Col_ATR14_Exit") == 1.5
        assert trade_dict.get("Col_VWAP_Exit") == 100.2

    def test_apply_exit_columns_skips_missing(self):
        """Missing or invalid values are skipped."""
        trade_dict = {}
        row = pd.Series({"Col_ATR14": 1.0})
        apply_exit_columns(trade_dict, row)
        assert "Col_ATR14_Exit" in trade_dict
        assert trade_dict["Col_ATR14_Exit"] == 1.0
        # Col_VWAP not in row so no Col_VWAP_Exit
        assert "Col_VWAP_Exit" not in trade_dict or trade_dict.get("Col_VWAP_Exit") is None

    def test_has_librarycolumn_bool(self):
        """has_librarycolumn returns bool."""
        assert isinstance(has_librarycolumn(), bool)
