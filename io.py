"""I/O utilities for backtest outputs."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .bt_types import RunResult


def write_trades_csv(
    result: RunResult,
    path: Union[str, Path],
    *,
    enriched_long_df: Optional[pd.DataFrame] = None,
) -> None:
    """Write backtest trades to CSV with Entry, Exit, and optionally Continuous columns.

    Columns are labelled: Entry_Col_X (entry snapshot), Exit_Col_X (exit snapshot),
    Continuous_Col_X_Entry/Exit/Max/Min/At30min/At60min (continuous, only if enriched_long_df provided).

    Args:
        result: RunResult from ChronologicalBacktestEngine.
        path: Output file path (e.g. "trades.csv" or Path("backtest_results/trades.csv")).
        enriched_long_df: Optional minute-level long DataFrame with Ticker, datetime and Col_*.
                          If provided, runs attach_continuous_tracking so CSV includes Continuous_Col_* columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = result.trades if result.trades is not None else pd.DataFrame()
    if enriched_long_df is not None and not enriched_long_df.empty:
        from .columns import attach_continuous_tracking
        result = attach_continuous_tracking(result, enriched_long_df)
        if result.trades is not None:
            to_write = result.trades
    to_write.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def write_trades_excel(
    result: RunResult,
    path: Union[str, Path],
    *,
    enriched_long_df: Optional[pd.DataFrame] = None,
) -> None:
    """Write backtest trades to Excel with Entry, Exit, and optionally Continuous columns.

    Same column labelling as write_trades_csv: Entry_Col_*, Exit_Col_*, Continuous_Col_*.

    Args:
        result: RunResult from ChronologicalBacktestEngine.
        path: Output file path (e.g. "trades.xlsx").
        enriched_long_df: Optional minute-level long DataFrame; if provided, adds Continuous_Col_* columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = result.trades if result.trades is not None else pd.DataFrame()
    if enriched_long_df is not None and not enriched_long_df.empty:
        from .columns import attach_continuous_tracking
        result = attach_continuous_tracking(result, enriched_long_df)
        if result.trades is not None:
            to_write = result.trades
    to_write.to_excel(path, index=False, engine="openpyxl")
