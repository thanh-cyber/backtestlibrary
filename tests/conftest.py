"""Shared pytest fixtures for backtestlibrary tests."""
from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from backtestlibrary.bt_types import EntryCandidate


@pytest.fixture
def sample_day_df_wide() -> pd.DataFrame:
    """Minimal wide-format day DataFrame (1 ticker, 1 date, session bars)."""
    return pd.DataFrame({
        "Ticker": ["AAPL"],
        "Date": [pd.Timestamp("2022-01-03")],
        "9:30": [99.5],
        "9:31": [99.6],
        "9:45": [100.0],
        "9:46": [100.5],
        "10:00": [101.0],
        "16:00": [102.0],
        "Col_ATR14": [1.5],
        "Exit_Price": [102.0],
    })


@pytest.fixture
def sample_cleaned_year_data(sample_day_df_wide) -> dict[str, pd.DataFrame]:
    """Minimal cleaned_year_data for engine: 1 day, 1 ticker."""
    return {"2022": sample_day_df_wide}


@pytest.fixture
def strategy_one_entry_with_stop():
    """Strategy that returns one entry with stop_price (required by engine)."""

    def find_entries_for_day(day_df, timeline, get_price, day_context=None):
        if day_df.empty:
            return []
        row = day_df.iloc[0]
        entry_time = time(9, 46)
        entry_price = 100.5
        stop = 95.5
        target = 110.0
        return [
            EntryCandidate(
                ticker=str(row.get("Ticker", "AAPL")),
                row_index=day_df.index[0],
                entry_time=entry_time,
                entry_price=entry_price,
                side="long",
                stop_price=stop,
                target_price=target,
            )
        ]

    def check_exit(pos, row, current_time, get_price, day_context=None):
        return None

    class S:
        pass

    S.find_entries_for_day = staticmethod(find_entries_for_day)
    S.check_exit = staticmethod(check_exit)
    return S()
