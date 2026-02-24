"""
Backtrader-style rich matplotlib plotting for backtest results.

Requires optional dependency: pip install backtestlibrary[plot]
Plots equity curve, drawdown, and optional trade P&L.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .bt_types import RunResult


def _ensure_mpl():
    """Lazy import matplotlib; raise helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install backtestlibrary[plot]"
        ) from e


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Return drawdown (negative) from equity curve: equity - running max."""
    if len(equity) == 0:
        return np.array([])
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return np.where(peak > 0, dd / peak * 100.0, 0.0)  # % drawdown


def plot_result(
    result: RunResult,
    *,
    dates: Optional[List[pd.Timestamp]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
    show: bool = True,
    ax_equity: Any = None,
    ax_drawdown: Any = None,
):
    """
    Plot equity curve and drawdown (backtrader-style).

    Uses result.daily_equity for the curve. If dates is provided, x-axis is
    calendar dates; otherwise 0..N-1 (trading days).

    Parameters
    ----------
    result : RunResult
        Single backtest run result (e.g. results[year][account]).
    dates : list of pd.Timestamp, optional
        One per equity point (len = len(result.daily_equity)). If None, uses 0..N-1.
    title : str, optional
        Figure/suptitle.
    figsize : tuple
        Figure size (width, height).
    show : bool
        If True, call plt.show() at the end.
    ax_equity, ax_drawdown : matplotlib axes, optional
        If both provided, draw into these axes and do not create a new figure.
    """
    plt = _ensure_mpl()
    equity = np.array(result.daily_equity, dtype=float)
    if len(equity) == 0:
        if ax_equity is None:
            fig, (ax_equity, ax_drawdown) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])
        ax_equity.set_title("Equity (no data)")
        if show:
            plt.show()
        return

    dd = _drawdown_series(equity)
    n = len(equity)
    if dates is not None and len(dates) == n:
        x = np.arange(n)  # use indices for sharex
        x_label = dates
    else:
        x = np.arange(n)
        x_label = x

    if ax_equity is not None and ax_drawdown is not None:
        ax_equity.plot(x, equity, color="C0", linewidth=1.5, label="Equity")
        ax_equity.set_ylabel("Equity")
        ax_equity.legend(loc="upper left")
        ax_equity.grid(True, alpha=0.3)
        ax_equity.set_title("Portfolio value")

        ax_drawdown.fill_between(x, dd, 0, color="C3", alpha=0.4, label="Drawdown %")
        ax_drawdown.set_ylabel("Drawdown %")
        ax_drawdown.set_xlabel("Trading day" if dates is None else "Date")
        ax_drawdown.legend(loc="lower left")
        ax_drawdown.grid(True, alpha=0.3)
        ax_drawdown.set_title("Drawdown")

        if dates is not None and len(dates) == n:
            try:
                import matplotlib.dates as mdates
                ax_drawdown.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax_drawdown.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax_drawdown.xaxis.get_majorticklabels(), rotation=45, ha="right")
            except Exception:
                pass
        if title:
            ax_equity.figure.suptitle(title)
        if show:
            plt.tight_layout()
            plt.show()
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])
    ax1.plot(x, equity, color="C0", linewidth=1.5, label="Equity")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Portfolio value")

    ax2.fill_between(x, dd, 0, color="C3", alpha=0.4, label="Drawdown %")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Trading day" if dates is None else "Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Drawdown")

    if dates is not None and len(dates) == n:
        try:
            import matplotlib.dates as mdates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        except Exception:
            pass
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def plot_equity_drawdown(
    equity_curve: List[float],
    *,
    dates: Optional[List[pd.Timestamp]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
    show: bool = True,
):
    """
    Plot raw equity curve and its drawdown (no RunResult).

    Parameters
    ----------
    equity_curve : list of float
        [start_bal, end_day1, end_day2, ...].
    dates : list of pd.Timestamp, optional
        One per point; if None, x-axis is 0..N-1.
    title, figsize, show
        Same as plot_result.
    """
    r = RunResult(
        final_balance=equity_curve[-1] if equity_curve else 0.0,
        total_return_pct=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        total_pnl=0.0,
        trades=pd.DataFrame(),
        daily_equity=list(equity_curve),
    )
    plot_result(r, dates=dates, title=title, figsize=figsize, show=show)


def plot_trade_pnl(
    result: RunResult,
    *,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
):
    """
    Bar chart of per-trade net P&L (backtrader Trade observer style).
    """
    plt = _ensure_mpl()
    t = result.trades
    if t is None or t.empty or "net_pnl" not in t.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or "Trade P&L (no trades)")
        if show:
            plt.show()
        return

    net = t["net_pnl"].astype(float)
    colors = np.where(net >= 0, "C2", "C3")  # green / red
    x = np.arange(len(net))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, net, color=colors, alpha=0.8, edgecolor="none")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Net P&L")
    ax.set_xlabel("Trade")
    ax.set_title(title or "Trade P&L")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if show:
        plt.show()
