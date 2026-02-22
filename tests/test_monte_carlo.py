"""Regression tests for Monte Carlo bootstrap."""
from __future__ import annotations

import pandas as pd

from backtestlibrary.monte_carlo import run_monte_carlo


def test_run_monte_carlo_returns_expected_structure():
    """Monte Carlo returns mc_df and finals_cache with expected shape."""
    res = {
        "2022": {
            100_000: {
                "trades": pd.DataFrame({
                    "net_pnl": [100.0, -50.0, 75.0],
                }),
            }
        }
    }
    mc_df, finals_cache = run_monte_carlo(res, [100_000], mc_runs=100, seed=42)
    assert not mc_df.empty
    assert "year" in mc_df.columns
    assert "MC Mean Final $" in mc_df.columns
    assert ("2022", 100_000) in finals_cache
    assert len(finals_cache[("2022", 100_000)]) == 100
