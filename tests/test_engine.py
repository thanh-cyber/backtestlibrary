"""Unit and integration tests for ChronologicalBacktestEngine."""
from __future__ import annotations

from datetime import time

import pandas as pd

from backtestlibrary.bt_types import EntryCandidate
from backtestlibrary.engine import (
    BacktestConfig,
    ChronologicalBacktestEngine,
    commission_per_order_us_stock_fixed,
)
from backtestlibrary.sizers import FixedSizeSizer, KellySizer, PercentOfEquitySizer, RiskSizer


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

    def test_fixed_size_sizer(self):
        """FixedSizeSizer returns fixed share count; no stop required."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            sizer=FixedSizeSizer(stake=50),
            equity_cap_pct=1.0,
            absolute_cap_value=1e9,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL"})
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=None)
        assert shares == 50
        assert value == 5000.0

    def test_percent_of_equity_sizer(self):
        """PercentOfEquitySizer: 10% of 10k at 100 = 10 shares."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            sizer=PercentOfEquitySizer(percent=0.10),
            equity_cap_pct=1.0,
            absolute_cap_value=1e9,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL"})
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=None)
        assert shares == 10
        assert value == 1000.0

    def test_kelly_sizer(self):
        """KellySizer with win_rate and payoff_ratio sizes by Kelly fraction."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            sizer=KellySizer(win_rate=0.55, payoff_ratio=1.2, fraction=0.5),
            equity_cap_pct=2.0,
            absolute_cap_value=1e9,
        )
        engine = ChronologicalBacktestEngine(cfg)
        row = pd.Series({"Ticker": "AAPL"})
        # f* = (0.55*1.2 - 0.45)/1.2 = 0.175; half-Kelly = 0.0875
        # risk_dollars = 10_000 * 0.0875 = 875, risk_per_share = 5 -> 175 shares (cap allows 200)
        shares, value = engine._size_position(100.0, row, 10_000, stop_price=95.0)
        assert shares == 175
        assert value == 17500.0

    def test_pick_sizer_via_config(self):
        """Config.sizer selects which sizer is used; default is RiskSizer."""
        row = pd.Series({"Ticker": "AAPL"})
        # Default (no sizer) = RiskSizer with config
        cfg_risk = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
            equity_cap_pct=1.0,
            absolute_cap_value=1e9,
        )
        engine_risk = ChronologicalBacktestEngine(cfg_risk)
        sh1, _ = engine_risk._size_position(100.0, row, 10_000, stop_price=95.0)
        assert sh1 == 100
        # Explicit FixedSize (cap high enough so 200 shares allowed)
        cfg_fixed = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            sizer=FixedSizeSizer(stake=200),
            equity_cap_pct=2.0,
            absolute_cap_value=1e9,
        )
        engine_fixed = ChronologicalBacktestEngine(cfg_fixed)
        sh2, _ = engine_fixed._size_position(100.0, row, 10_000, stop_price=95.0)
        assert sh2 == 200


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
        results, _metrics_df, _curves = engine.run(
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
        # Engine returns (results, metrics_df, equity_curves)
        assert isinstance(_metrics_df, pd.DataFrame)
        assert len(_metrics_df) >= 1
        assert ("2022", 100_000) in _curves
        assert len(_curves[("2022", 100_000)]) == len(r.daily_equity)

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
        results, _, _ = engine.run(sample_cleaned_year_data, S(), [100_000])
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
        results, _metrics_df, _curves = engine.run(
            sample_cleaned_year_data,
            strategy_one_entry_with_stop,
            starting_accounts=[100_000],
        )
        r = results["2022"][100_000]
        # 1 day in sample_cleaned_year_data -> daily_equity = [start, end_day1]
        assert len(r.daily_equity) == 2

    def test_use_library_columns_on_by_default(self, sample_cleaned_year_data, strategy_one_entry_with_stop):
        """Default config has use_library_columns=True so entry/exit columns are always included; run completes and produces trades."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        )
        assert cfg.use_library_columns is True
        engine = ChronologicalBacktestEngine(cfg)
        results, _, _ = engine.run(
            sample_cleaned_year_data,
            strategy_one_entry_with_stop,
            starting_accounts=[100_000],
        )
        r = results["2022"][100_000]
        assert r.total_trades >= 1
        assert isinstance(r.trades, pd.DataFrame)

    def test_use_library_columns_on_captures_entry_and_exit_columns(
        self, sample_cleaned_year_data, strategy_one_entry_with_stop
    ):
        """With use_library_columns=True, engine runs entry/exit column logic (entry at entry time, exit at exit)."""
        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
            use_library_columns=True,
        )
        engine = ChronologicalBacktestEngine(cfg)
        results, _, _ = engine.run(
            sample_cleaned_year_data,
            strategy_one_entry_with_stop,
            starting_accounts=[100_000],
        )
        r = results["2022"][100_000]
        assert r.total_trades >= 1
        assert not r.trades.empty
        # Exit columns from librarycolumn (or defaults) should be present on the trade row
        first = r.trades.iloc[0]
        assert "Exit_Col_ATR14" in first.index or "Exit_Col_VWAP" in first.index

    def test_trade_side_column_is_persisted_for_short_positions(self, sample_cleaned_year_data):
        def find_entries(day_df, timeline, get_price, day_context=None):
            return [
                EntryCandidate(
                    ticker="AAPL",
                    row_index=day_df.index[0],
                    entry_time=time(9, 46),
                    entry_price=100.5,
                    side="short",
                    stop_price=105.5,
                    target_price=95.0,
                )
            ]

        def check_exit(pos, row, current_time, get_price, day_context=None):
            return None

        class S:
            pass

        S.find_entries_for_day = staticmethod(find_entries)
        S.check_exit = staticmethod(check_exit)

        cfg = BacktestConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            risk_pct_per_trade=0.05,
        )
        engine = ChronologicalBacktestEngine(cfg)
        results, _, _ = engine.run(sample_cleaned_year_data, S(), [100_000])
        trades = results["2022"][100_000].trades
        assert not trades.empty
        assert "side" in trades.columns
        assert trades["side"].iloc[0] == "short"


class TestIBKRFixedCommission:
    """IBKR US stock Fixed commission component (per order); see BacktestConfig defaults."""

    _PS = 0.005
    _MN = 1.0
    _CAP = 0.01

    def test_minimum_per_order_when_raw_below_one_dollar(self):
        # 100 sh * 0.005 = 0.50 < 1.00 min
        assert (
            commission_per_order_us_stock_fixed(
                100,
                1_000.0,
                commission_per_share=self._PS,
                commission_min_per_order=self._MN,
                commission_max_pct_per_order=self._CAP,
            )
            == 1.0
        )

    def test_per_share_when_above_minimum_under_cap(self):
        # 500 * 0.005 = 2.5
        assert (
            commission_per_order_us_stock_fixed(
                500,
                10_000.0,
                commission_per_share=self._PS,
                commission_min_per_order=self._MN,
                commission_max_pct_per_order=self._CAP,
            )
            == 2.5
        )

    def test_one_percent_cap_binds(self):
        # raw = 2000 * 0.005 = 10; cap = 1% * 10_000 = 100 -> 10; max(1, 10) = 10
        assert (
            commission_per_order_us_stock_fixed(
                2000,
                10_000.0,
                commission_per_share=self._PS,
                commission_min_per_order=self._MN,
                commission_max_pct_per_order=self._CAP,
            )
            == 10.0
        )

    def test_cap_binds_below_raw(self):
        # raw = 5000 * 0.005 = 25; cap = 1% * 1000 = 10 -> fee 10
        assert (
            commission_per_order_us_stock_fixed(
                5000,
                1_000.0,
                commission_per_share=self._PS,
                commission_min_per_order=self._MN,
                commission_max_pct_per_order=self._CAP,
            )
            == 10.0
        )

    def test_round_trip_is_sum_of_two_orders(self):
        entry_fee = commission_per_order_us_stock_fixed(
            100,
            1_000.0,
            commission_per_share=self._PS,
            commission_min_per_order=self._MN,
            commission_max_pct_per_order=self._CAP,
        )
        exit_fee = commission_per_order_us_stock_fixed(
            100,
            950.0,
            commission_per_share=self._PS,
            commission_min_per_order=self._MN,
            commission_max_pct_per_order=self._CAP,
        )
        assert entry_fee == 1.0
        assert exit_fee == 1.0
        assert entry_fee + exit_fee == 2.0

    def test_backtest_config_default_commission_matches_ib_fixed(self):
        cfg = BacktestConfig(session_start=time(9, 30), session_end=time(16, 0), risk_pct_per_trade=0.05)
        assert cfg.commission_per_share == 0.005
        assert cfg.commission_min_per_order == 1.0
        assert cfg.commission_max_pct_per_order == 0.01
