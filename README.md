# Backtest Backbone

Reusable, strategy-agnostic backtesting backbone for consistent research runs.

## Install (editable for development)

```bash
git clone https://github.com/thanh-cyber/backtestlibrary.git
cd backtestlibrary
pip install -e .
```

After install:

```python
from backtestlibrary import ChronologicalBacktestEngine, BacktestConfig
```

## What is included

- Data import + cleaning + cache (`data.py`)
- **Flexible Pandas data feed + resampling** (`feed.py`) — validate/normalize any DataFrame to the engine contract; resample wide intraday bars (e.g. 1m → 5m)
- Chronological execution engine (`engine.py`)
- Modular analyzers + full metrics (`analyzers.py`)
- Monte Carlo stress testing (`monte_carlo.py`)
- Strategy plug-in interface (`bt_types.py`)

## What is not included

- Any specific strategy logic (long, short, PM, open, AH, etc.)

## Usage flow

1. Implement your strategy class using `find_entries_for_day()` and `check_exit()`
2. Load data with `GapperDataLoader`
3. Run engine with `ChronologicalBacktestEngine`
4. Use the returned `(results, metrics_df, equity_curves)` and/or `run_monte_carlo()`

See `run_backbone_template.py` and `strategy_template.py`.

### Engine output

`engine.run(cleaned_year_data, strategy, starting_accounts)` returns a 3-tuple:

```python
results, metrics_df, equity_curves = engine.run(cleaned_year_data, strategy, starting_accounts)
# results       → dict[year][account] = RunResult
# metrics_df    → full metrics table (one row per year/account)
# equity_curves → dict[(year, account)] = list of daily equity values
```

### Per-run modular analyzers

Each `RunResult` has `result.analyzers` populated by the engine (when using `DEFAULT_ANALYZERS` or a custom analyzer list):

```python
result = results["2025"][100_000]
print(result.analyzers["FullMetrics"])   # full metrics dict for this run
print(result.analyzers["SQN"])
print(result.analyzers["DrawdownDuration"])
```

You can still call `build_full_metrics(results, starting_accounts)` yourself if you have results from another source; the engine now computes it automatically and returns it as the second element of the tuple.

### Flexible Pandas data feed

Any DataFrame (or dict of DataFrames by year) that has **Date**, **Ticker**, and at least one **time column** (e.g. `9:30`, `9:31`) can be validated, normalized, and passed to the engine:

```python
from backtestlibrary import PandasDataFeed, DataFeedConfig, validate_feed, normalize_feed

# Single DataFrame or dict[str, DataFrame]
feed = PandasDataFeed(my_df, config=DataFeedConfig(session_start=(9, 30), session_end=(16, 0)))
cleaned_year_data = feed.to_cleaned_year_data()
results, metrics_df, equity_curves = engine.run(cleaned_year_data, strategy, starting_accounts)
```

Use `validate_feed(df)` to check contract; `normalize_feed(df)` to normalize dates and tickers.

### Resampling

Resample wide-format minute bars to a coarser timeframe (e.g. 5m) so the engine runs on fewer bars. Use `timeline_step_seconds = rule_minutes * 60` in `BacktestConfig` when running on resampled data:

```python
from backtestlibrary import resample_wide_intraday, PandasDataFeed, DataFeedConfig

# Option A: resample when building the feed
feed = PandasDataFeed(data, config=DataFeedConfig(resample_minutes=5))
cleaned_year_data = feed.to_cleaned_year_data()

# Option B: resample a DataFrame directly
resampled_df = resample_wide_intraday(wide_df, rule_minutes=5, session_start=(9, 30), session_end=(16, 0))

# Run engine with 5m step
config = BacktestConfig(..., timeline_step_seconds=300)  # 5 * 60
```

## Performance notes

- The engine keeps chronological execution semantics unchanged.
- Under the hood, it precomputes a vectorized per-day price matrix for faster
  repeated lookups.
- Strategies can optionally implement `prepare_day(...)` and receive
  `day_context` in `find_entries_for_day(..., day_context)` and
  `check_exit(..., day_context)` for additional speedups without changing fill
  realism.

