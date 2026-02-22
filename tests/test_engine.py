"""Unit and integration tests for ChronologicalBacktestEngine."""
from __future__ import annotations

from datetime import time

import pandas as pd

from backtestlibrary.bt_types import EntryCandidate
from backtestlibrary.engine import BacktestConfig, ChronologicalBacktestEngine


class TestSizePosition:
    """Unit tests for _size_position."""

    def test_requires_stop_price(self):
        """Entries without stop_price return 0 shares."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL", "9:30": 100.0})
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=None)
        assert shares == 0
        assert value == 0.0

    def test_requires_risk_config(self):
        """Without risk_pct or fixed_risk, returns 0 shares."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=None,
            fixed_risk_per_trade=None,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL"})
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=95.0)
        assert shares == 0
        assert value == 0.0

    def test_risk_pct_sizing(self):
        """Risk-based sizing: 5% of 10k, $5/share risk = 100 shares."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
            equity_cap_pct=1.0,
            absolute_cap_value=1e9,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL"})
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=95.0)
        risk_per_share = 5.0
        risk_dollars = 10_000 * 0.05
        expected_shares = int(risk_dollars / risk_per_share)
        assert shares == expected_shares
        assert value == shares * 100.0

    def test_fixed_risk_sizing(self):
        """Fixed $ risk per trade."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            fixed_risk_per_trade=500.0,
            equity_cap_pct=1.0,
            absolute_cap_value=1e9,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL"})
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=95.0)
        expected = int(500.0 / 5.0)
        assert shares == expected
        assert value == shares * 100.0

    def test_zero_entry_price_returns_zero(self):
        """Zero entry price returns 0 shares."""
        cfg = BacktestConfig(session_start=time(9, 30), session_end=time(16, 0), risk_pct_per_trade=0.05)
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({})
        shares, value = engine._size_position(0.0, row, 10_000, stop_price=95.0)
        assert shares == 0
        assert value == 0.0


class TestEngineIntegration:
    """Integration tests: full engine run with synthetic data."""

    def test_full_run_produces_result_structure(self, sample_cleaned_year_data, strategy_one_entry_with_stop):
        """Full run returns dict[year][account] RunResult with expected fields."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        )
        engine = ChronologicalBacktestEngine(cfg)
        results = engine.run(
            sample_cleaned_year_data,
            strategy_one_entry_with_stop,
            starting_accounts=[100_000],
        )
        assert "2022" in results
        assert 100_000 in results["2022"]
        r = results["2022"][100_000]
        assert hasattr(r, "final_balance")
        assert hasattr(r, "total_return_pct")
        assert hasattr(r, "total_trades")
        assert hasattr(r, "trades")
        assert hasattr(r, "daily_equity")
        assert isinstance(r.trades, pd.DataFrame)
        assert isinstance(r.daily_equity, list)
        # daily_equity: [start_bal, end_day1, ...]
        assert len(r.daily_equity) >= 1

    def test_entries_without_stop_price_skipped(self, sample_cleaned_year_data):
        """Entries with stop_price=None are skipped."""

        def find_entries(day_df, timeline, get_price, day_context=None):
            if day_df.empty:
                return []
            return [
                EntryCandidate(
                    ticker="AAPL",
                    row_index=day_df.index[0],
                    entry_time=time(9, 46),
                    entry_price=100.5,
                    side="long",
                    stop_price=None,  # skipped
                    target_price=110.0,
                )
            ]

        def check_exit(pos, row, current_time, get_price, day_context=None):
            return None

        class S:
            pass

        S.find_entries_for_day = staticmethod(find_entries)
        S.check_exit = staticmethod(check_exit)

        cfg = BacktestConfig(session_start=time(9, 30), session_end=time(16, 0), risk_pct_per_trade=0.05)
        engine = ChronologicalBacktestEngine(cfg)
        results = engine.run(sample_cleaned_year_data, S(), [100_000])
        r = results["2022"][100_000]
        assert r.total_trades == 0
        assert r.trades.empty

    def test_daily_equity_length_is_one_plus_trading_days(
        self, sample_cleaned_year_data, strategy_one_entry_with_stop
    ):
        """daily_equity = [start_bal, end_day1, end_day2, ...] -> len = 1 + num_trading_days."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        )
        engine = ChronologicalBacktestEngine(cfg)
        results = engine.run(
            sample_cleaned_year_data,
            strategy_one_entry_with_stop,
            starting_accounts=[100_000],
        )
        r = results["2022"][100_000]
        # 1 day in sample_cleaned_year_data -> daily_equity = [start, end_day1]
        assert len(r.daily_equity) == 2
