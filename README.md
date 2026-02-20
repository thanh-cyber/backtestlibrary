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
- Chronological execution engine (`engine.py`)
- Full metrics output (`metrics.py`)
- Monte Carlo stress testing (`monte_carlo.py`)
- Strategy plug-in interface (`bt_types.py`)

## What is not included

- Any specific strategy logic (long, short, PM, open, AH, etc.)

## Usage flow

1. Implement your strategy class using `find_entries_for_day()` and `check_exit()`
2. Load data with `GapperDataLoader`
3. Run engine with `ChronologicalBacktestEngine`
4. Generate standardized outputs with:
   - `build_full_metrics()`
   - `run_monte_carlo()`

See `run_backbone_template.py` and `strategy_template.py`.

## Performance notes

- The engine keeps chronological execution semantics unchanged.
- Under the hood, it precomputes a vectorized per-day price matrix for faster
  repeated lookups.
- Strategies can optionally implement `prepare_day(...)` and receive
  `day_context` in `find_entries_for_day(..., day_context)` and
  `check_exit(..., day_context)` for additional speedups without changing fill
  realism.

