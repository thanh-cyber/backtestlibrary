"""Tests for columns module (entry/exit/continuous from column_library)."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from backtestlibrary.bt_types import RunResult
from backtestlibrary import columns


def _mock_lib(entry=None, exit_=None, continuous=None):
    """Fake column_library with ENTRY_COLUMNS, EXIT_SNAPSHOT_COLUMNS, CONTINUOUS_TRACKING_COLUMNS."""
    m = type("MockLib", (), {})()
    if entry is not None:
        m.ENTRY_COLUMNS = entry
    if exit_ is not None:
        m.EXIT_SNAPSHOT_COLUMNS = exit_
    if continuous is not None:
        m.CONTINUOUS_TRACKING_COLUMNS = continuous
    return m


class TestColumnsModule:
    """Test column getters and apply_* with mocked column_library."""

    def test_get_entry_columns_returns_list_when_defined(self):
        """When column_library.ENTRY_COLUMNS is defined and non-empty, returns it."""
        with patch.object(columns, "_lib", return_value=_mock_lib(entry=["Col_ATR14", "Col_VWAP"])):
            cols = columns.get_entry_columns()
        assert cols == ["Col_ATR14", "Col_VWAP"]

    def test_get_entry_columns_uses_builtin_when_missing(self):
        """When column_library has no ENTRY_COLUMNS, returns built-in list."""
        with patch.object(columns, "_lib", return_value=object()):
            cols = columns.get_entry_columns()
        assert isinstance(cols, list)
        assert "Col_ATR14" in cols
        assert len(cols) > 2

    def test_get_entry_columns_uses_builtin_when_empty(self):
        """When ENTRY_COLUMNS is empty, returns built-in list."""
        with patch.object(columns, "_lib", return_value=_mock_lib(entry=[])):
            cols = columns.get_entry_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_exit_columns_returns_list_when_defined(self):
        """When column_library.EXIT_SNAPSHOT_COLUMNS is defined and non-empty, returns it."""
        with patch.object(columns, "_lib", return_value=_mock_lib(exit_=["Col_ATR14", "Col_VWAP"])):
            cols = columns.get_exit_columns()
        assert cols == ["Col_ATR14", "Col_VWAP"]

    def test_get_exit_columns_uses_builtin_when_missing(self):
        """When column_library has no EXIT_SNAPSHOT_COLUMNS, returns built-in list."""
        with patch.object(columns, "_lib", return_value=object()):
            cols = columns.get_exit_columns()
        assert isinstance(cols, list)
        assert "Col_ATR14" in cols
        assert len(cols) > 2

    def test_get_exit_columns_uses_builtin_when_empty(self):
        """When EXIT_SNAPSHOT_COLUMNS is empty, returns built-in list."""
        with patch.object(columns, "_lib", return_value=_mock_lib(exit_=[])):
            cols = columns.get_exit_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_continuous_columns_returns_list_when_defined(self):
        """When column_library.CONTINUOUS_TRACKING_COLUMNS is defined and non-empty, returns it."""
        with patch.object(columns, "_lib", return_value=_mock_lib(continuous=["Col_RSI14"])):
            cols = columns.get_continuous_columns()
        assert cols == ["Col_RSI14"]

    def test_get_continuous_columns_uses_builtin_when_missing(self):
        """When column_library has no CONTINUOUS_TRACKING_COLUMNS, returns built-in list (20 from librarycolumn)."""
        with patch.object(columns, "_lib", return_value=object()):
            cols = columns.get_continuous_columns()
        assert isinstance(cols, list)
        assert "Col_RSI14" in cols
        assert len(cols) == 20

    def test_get_continuous_columns_allows_empty_override(self):
        """When column_library sets CONTINUOUS_TRACKING_COLUMNS=[], returns []."""
        with patch.object(columns, "_lib", return_value=_mock_lib(continuous=[])):
            cols = columns.get_continuous_columns()
        assert cols == []

    def test_get_entry_columns_uses_builtin_when_lib_missing(self):
        """When column_library cannot be imported, returns built-in list."""
        with patch.object(columns, "_lib", return_value=None):
            cols = columns.get_entry_columns()
        assert isinstance(cols, list)
        assert "Col_ATR14" in cols

    def test_apply_entry_columns_adds_entry_prefix(self):
        """apply_entry_columns adds Entry_Col_X when column is in get_entry_columns()."""
        with patch.object(columns, "_lib", return_value=_mock_lib(entry=["Col_ATR14", "Col_VWAP"])):
            trade_dict = {"ticker": "AAPL"}
            row = pd.Series({"Col_ATR14": 1.5, "Col_VWAP": 100.2})
            columns.apply_entry_columns(trade_dict, row)
            assert trade_dict.get("Entry_Col_ATR14") == 1.5
            assert trade_dict.get("Entry_Col_VWAP") == 100.2

    def test_apply_exit_columns_adds_exit_suffix(self):
        """apply_exit_columns adds Exit_Col_X from row to trade_dict."""
        with patch.object(columns, "_lib", return_value=_mock_lib(exit_=["Col_ATR14", "Col_VWAP"])):
            trade_dict = {"ticker": "AAPL", "net_pnl": 100.0}
            row = pd.Series({"Col_ATR14": 1.5, "Col_VWAP": 100.2, "other": 99})
            columns.apply_exit_columns(trade_dict, row)
            assert trade_dict.get("Exit_Col_ATR14") == 1.5
            assert trade_dict.get("Exit_Col_VWAP") == 100.2

    def test_apply_exit_columns_skips_missing(self):
        """Missing or invalid values are skipped."""
        with patch.object(columns, "_lib", return_value=_mock_lib(exit_=["Col_ATR14", "Col_VWAP"])):
            trade_dict = {}
            row = pd.Series({"Col_ATR14": 1.0})
            columns.apply_exit_columns(trade_dict, row)
            assert "Exit_Col_ATR14" in trade_dict
            assert trade_dict["Exit_Col_ATR14"] == 1.0
            assert "Exit_Col_VWAP" not in trade_dict or trade_dict.get("Exit_Col_VWAP") is None

    def test_has_librarycolumn_bool(self):
        """has_librarycolumn returns bool."""
        assert isinstance(columns.has_librarycolumn(), bool)

    def test_attach_continuous_tracking_raises_clear_error_when_ticker_missing(self):
        mock_lib = type("MockLib", (), {"add_continuous_tracking": lambda *a, **k: pd.DataFrame()})()
        result = RunResult(
            final_balance=100_000.0,
            total_return_pct=0.0,
            total_trades=1,
            winning_trades=0,
            losing_trades=1,
            total_pnl=-10.0,
            trades=pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-01-02")],
                    "entry_time": ["09:30"],
                    "exit_time": ["09:31"],
                    "net_pnl": [-10.0],
                }
            ),
            daily_equity=[100_000.0, 99_990.0],
        )
        enriched_long = pd.DataFrame(
            {"Ticker": ["AAPL"], "datetime": [pd.Timestamp("2026-01-02 09:30")]}
        )
        with patch.object(columns, "_lib", return_value=mock_lib):
            with pytest.raises(ValueError, match="missing ticker column"):
                columns.attach_continuous_tracking(result, enriched_long)
