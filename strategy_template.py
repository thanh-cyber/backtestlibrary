"""Strategy template (example only).

This file is NOT part of the backbone internals. Copy and customize for each
strategy while reusing the same loader/engine/metrics/monte-carlo modules.
"""

from __future__ import annotations

from datetime import time
from typing import Optional

import pandas as pd

from .bt_types import EntryCandidate, ExitSignal, Position


class StrategyTemplate:
    """Fill in your own entry/exit logic."""

    def prepare_day(
        self,
        day_df: pd.DataFrame,
        timeline: list[time],
        get_price,
    ):
        """Optional vectorized precompute hook called once per day.

        Return any context object (arrays, masks, lookup dicts) used by
        find_entries_for_day/check_exit for faster per-tick logic.
        """
        return None

    def find_entries_for_day(
        self,
        day_df: pd.DataFrame,
        timeline: list[time],
        get_price,
        day_context=None,
    ) -> list[EntryCandidate]:
        candidates: list[EntryCandidate] = []
        # TODO: implement your entry scan logic.
        # Example:
        # - iterate rows/tickers
        # - find first qualifying timestamp
        # - append EntryCandidate(side='long' or 'short', stop/target optional)
        return candidates

    def check_exit(
        self,
        position: Position,
        row: pd.Series,
        current_time: time,
        get_price,
        day_context=None,
    ) -> Optional[ExitSignal]:
        # TODO: implement stop/target/indicator/time rules.
        return None

