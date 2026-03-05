"""Reusable backtesting backbone components.

This package intentionally excludes strategy logic. Strategies should be
implemented separately and plugged into the engine via the Strategy protocol.
"""

from .data import GapperDataLoader, LoaderConfig, ParquetDataLoader, ParquetLoaderConfig
from .engine import ChronologicalBacktestEngine, BacktestConfig
from .analyzers import build_full_metrics
from .monte_carlo import run_monte_carlo
from .exit_metrics import (
    attach_exit_metrics_from_intraday_data,
    attach_exit_metrics_from_minute_bars,
    build_daily_exit_metrics,
    build_daily_exit_metrics_from_minute_bars,
    build_daily_exit_metrics_from_wide_minute_columns,
    merge_exit_metrics_into_backtest_data,
)
from .bt_types import (
    Analyzer,
    EntryCandidate,
    ExitSignal,
    Position,
    RunResult,
    Sizer,
    SizerConfig,
    Strategy,
    TradeRecord,
)
from .sizers import FixedSizeSizer, KellySizer, PercentOfEquitySizer, RiskSizer
from .analyzers import DEFAULT_ANALYZERS
from .feed import (
    DataFeedConfig,
    PandasDataFeed,
    REQUIRED_FEED_COLUMNS,
    normalize_feed,
    resample_wide_intraday,
    validate_feed,
)
from .io import write_trades_csv, write_trades_excel
from .trade_enrichment import enrich_results
from .plotting import plot_equity_drawdown, plot_result, plot_trade_pnl
from .librarycolumn_enrichment import (
    enrich_cleaned_year_data,
    get_row_at_time,
    wide_to_long,
    enrich_long_with_library_columns,
)
from .columns import (
    apply_exit_columns,
    attach_continuous_tracking,
    get_entry_columns,
    get_exit_columns,
    get_continuous_columns,
    has_librarycolumn,
)
from .column_definitions import (
    ENTRY_COLUMNS,
    EXIT_SNAPSHOT_COLUMNS,
    CONTINUOUS_TRACKING_COLUMNS,
)

__all__ = [
    "Analyzer",
    "BacktestConfig",
    "FixedSizeSizer",
    "KellySizer",
    "PercentOfEquitySizer",
    "RiskSizer",
    "DEFAULT_ANALYZERS",
    "ChronologicalBacktestEngine",
    "EntryCandidate",
    "ExitSignal",
    "GapperDataLoader",
    "LoaderConfig",
    "ParquetDataLoader",
    "ParquetLoaderConfig",
    "Position",
    "RunResult",
    "Sizer",
    "SizerConfig",
    "Strategy",
    "TradeRecord",
    "build_full_metrics",
    "run_monte_carlo",
    "build_daily_exit_metrics_from_minute_bars",
    "build_daily_exit_metrics_from_wide_minute_columns",
    "build_daily_exit_metrics",
    "merge_exit_metrics_into_backtest_data",
    "attach_exit_metrics_from_minute_bars",
    "attach_exit_metrics_from_intraday_data",
    "write_trades_csv",
    "write_trades_excel",
    "enrich_results",
    "enrich_cleaned_year_data",
    "get_row_at_time",
    "wide_to_long",
    "enrich_long_with_library_columns",
    "DataFeedConfig",
    "PandasDataFeed",
    "REQUIRED_FEED_COLUMNS",
    "normalize_feed",
    "resample_wide_intraday",
    "validate_feed",
    "plot_result",
    "plot_equity_drawdown",
    "plot_trade_pnl",
    "apply_exit_columns",
    "attach_continuous_tracking",
    "get_entry_columns",
    "get_exit_columns",
    "get_continuous_columns",
    "has_librarycolumn",
    "ENTRY_COLUMNS",
    "EXIT_SNAPSHOT_COLUMNS",
    "CONTINUOUS_TRACKING_COLUMNS",
]
