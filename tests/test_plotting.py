"""Tests for plotting module. Require matplotlib (pip install backtestlibrary[plot])."""
from __future__ import annotations

import pytest

import pandas as pd

# Use non-interactive backend so tests run without display (e.g. in CI / no Tk).
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

from backtestlibrary.bt_types import RunResult
from backtestlibrary.plotting import plot_equity_drawdown, plot_result, plot_trade_pnl

pytest.importorskip("matplotlib")


def test_plot_result_no_data():
    """plot_result with empty daily_equity does not crash."""
    r = RunResult(
        final_balance=100_000,
        total_return_pct=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=0.0,
        trades=pd.DataFrame(),
        daily_equity=[],
    )
    plot_result(r, show=False)


def test_plot_result_with_equity():
    """plot_result with equity curve produces figure."""
    r = RunResult(
        final_balance=105_000,
        total_return_pct=5.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        total_pnl=5000.0,
        trades=pd.DataFrame(),
        daily_equity=[100_000.0, 102_000.0, 105_000.0],
    )
    plot_result(r, title="Test", show=False)


def test_plot_trade_pnl_empty():
    """plot_trade_pnl with no trades does not crash."""
    r = RunResult(
        final_balance=100_000,
        total_return_pct=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=0.0,
        trades=pd.DataFrame(),
        daily_equity=[100_000.0],
    )
    plot_trade_pnl(r, show=False)


def test_plot_trade_pnl_with_trades():
    """plot_trade_pnl with trades DataFrame."""
    r = RunResult(
        final_balance=101_000,
        total_return_pct=1.0,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        total_pnl=1000.0,
        trades=pd.DataFrame([{"net_pnl": 500.0}, {"net_pnl": -200.0}]),
        daily_equity=[100_000.0, 100_500.0, 101_000.0],
    )
    plot_trade_pnl(r, show=False)


def test_plot_equity_drawdown():
    """plot_equity_drawdown with list."""
    plot_equity_drawdown([100_000.0, 98_000.0, 102_000.0], title="Curve", show=False)
