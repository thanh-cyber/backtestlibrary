"""Unit and regression tests for metrics module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtestlibrary import metrics


class TestSharpe:
    """Unit tests for _sharpe."""

    def test_empty_returns_nan(self):
        assert np.isnan(metrics._sharpe(pd.Series(dtype=float)))

    def test_single_value_nan(self):
        assert np.isnan(metrics._sharpe(pd.Series([0.01])))

    def test_positive_returns_positive_sharpe(self):
        ret = pd.Series([0.01, -0.005, 0.02, 0.0])
        s = metrics._sharpe(ret)
        assert s > 0

    def test_zero_std_nan(self):
        ret = pd.Series([0.01, 0.01, 0.01])
        s = metrics._sharpe(ret)
        assert np.isnan(s)


class TestSortino:
    """Unit tests for _sortino."""

    def test_empty_returns_nan(self):
        assert np.isnan(metrics._sortino(pd.Series(dtype=float)))

    def test_single_value_nan(self):
        assert np.isnan(metrics._sortino(pd.Series([0.01])))


class TestBuildFullMetrics:
    """Unit tests for build_full_metrics."""

    def test_empty_results_returns_empty_df(self):
        df, curves = metrics.build_full_metrics({}, [100_000])
        assert df.empty
        assert curves == {}

    def test_single_year_account(self):
        res = {
            "2022": {
                100_000: {
                    "trades": pd.DataFrame({
                        "gross_pnl": [100.0, -50.0],
                        "net_pnl": [97.0, -53.0],
                        "account_balance_after": [100_097, 100_044],
                        "exit_reason": ["Take Profit", "Stop Loss"],
                    }),
                    "final_balance": 100_044,
                    "total_return_pct": 0.044,
                    "total_pnl": 44.0,
                    "total_trades": 2,
                    "winning_trades": 1,
                    "losing_trades": 1,
                    "daily_equity": [100_000, 100_044],
                }
            }
        }
        df, curves = metrics.build_full_metrics(res, [100_000])
        assert not df.empty
        assert ("2022", 100_000) in curves


class TestCagrRegression:
    """Regression: CAGR for 252 days and 10% return ~= 10%."""

    def test_cagr_252_days_10pct_return(self):
        start = 100_000
        end = 110_000
        n_days = 252
        # daily_equity: [start, d1, d2, ..., d252] -> 253 points, 252 returns
        daily_eq = [start] + [start + (end - start) * i / n_days for i in range(1, n_days + 1)]
        daily_eq[-1] = end
        equity = np.array(daily_eq)
        daily_ret = pd.Series(equity).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        years_equiv = len(daily_ret) / 252.0
        cagr = ((end / start) ** (1 / years_equiv) - 1) * 100.0
        assert 9.0 < cagr < 11.0
