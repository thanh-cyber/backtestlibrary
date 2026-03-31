# Changelog

All notable changes to the backtestlibrary repo are documented here.

---

## [Unreleased / Latest] — Robustness: optional deps and sparse intraday data

**Commit:** `f2291b8`

This update hardens the library for environments where optional dependencies are missing and for datasets with sparse or incomplete intraday time columns.

### Summary

- **Optional dependencies:** The engine and post-backtest enrichment no longer raise when `librarycolumn` or `tqdm` are not installed. They degrade gracefully (no enrichment when librarycolumn is missing; no progress bar when tqdm is missing).
- **Exit columns:** Missing or invalid exit snapshot values are no longer written as `NaN` into trade dicts; those keys are skipped so output stays clean and tests pass.
- **Sparse intraday data:** Missing minute columns (e.g. not every 9:30–16:00 slot present) no longer abort the run; those slots remain NaN and downstream logic skips unavailable prices.

### Changes by file

#### `columns.py`

- **`apply_exit_columns(trade_dict, row)`**  
  - **Before:** For every column in the exit list, the function wrote a value into `trade_dict`—using `float('nan')` when the value was missing or invalid.  
  - **After:** Only adds keys when the value is present and valid (numeric, finite). Missing or invalid values are skipped (no key added).  
  - **Why:** Keeps trade dicts free of noisy `NaN` fields and matches test expectations (`test_apply_exit_columns_skips_missing`).

#### `engine.py`

- **Imports**  
  - Now imports `has_librarycolumn` from `columns`.

- **Progress bar (`show_progress=True`)**  
  - **Before:** `from tqdm.auto import tqdm` ran unconditionally when `show_progress` was True, causing `ModuleNotFoundError` if `tqdm` was not installed.  
  - **After:** The import is wrapped in `try/except`; if `tqdm` is unavailable, `pbar` stays `None` and the run continues without a progress bar.

- **Enrichment bootstrap**  
  - **Before:** When `use_library_columns` was True and `enriched_long_by_year` was not provided, the engine called `enrich_cleaned_year_data(in_memory)` whenever in-memory data existed, which could fail in minimal environments without `librarycolumn`.  
  - **After:** Enrichment is only attempted when `has_librarycolumn()` is True. If librarycolumn is not installed, the engine skips building enriched long data (weaker in-loop entry/exit row snapshots). Full **Continuous_*** and rich Phase 2 columns still require librarycolumn via `enrich_results` / `enrich_trades_post_backtest`.

- **`_build_day_price_cache(day_df, timeline)`**  
  - **Before:** For each time in `timeline`, if no matching wide column (e.g. `9:30`, `9:31`) was found in `day_df`, the function raised `ValueError` and aborted the run.  
  - **After:** If no time column is found for a given slot, the loop `continue`s and that slot is left as NaN in the price matrix. Downstream code already treats NaN as “no price” and skips accordingly.  
  - **Why:** Supports sparse or partially filled wide intraday data (e.g. only some minutes present) without failing.

#### `trade_enrichment.py`

- **Imports**  
  - Now imports `has_librarycolumn` from `columns`.

- **`enrich_trades_post_backtest(...)`**  
  - **Before:** If `use_library_columns` was True, the function proceeded to load data and run the full enrichment pipeline, which could later fail when `column_library` was missing.  
  - **After:** If `use_library_columns` is True but `has_librarycolumn()` is False, the function returns the result unchanged (no-op). Avoids running the heavy path only to fail on optional imports.

- **Progress bars (two places)**  
  - **Before:** `from tqdm.auto import tqdm` was used without a try/except; with progress enabled and `tqdm` missing, the enrichment could raise.  
  - **After:** Both progress-bar code paths wrap the tqdm import in `try/except` and only create a progress bar when `tqdm` is not None.

### Behaviour summary

| Scenario | Before | After |
|----------|--------|--------|
| `show_progress=True`, tqdm not installed | `ModuleNotFoundError` | No progress bar; run completes |
| `use_library_columns=True`, librarycolumn not installed | Could fail during enrichment or post-enrichment | Engine and post-enrichment skip column enrichment; run completes |
| Exit row missing a column in the exit list | Key added with `NaN` | Key omitted |
| Wide data missing a minute column (e.g. no `10:15`) | `ValueError`, run aborted | Slot left NaN; run continues |

### Tests

All 65 tests pass after these changes, including:

- `test_apply_exit_columns_adds_exit_suffix`
- `test_apply_exit_columns_skips_missing`
- `test_has_librarycolumn_bool`
- Full engine, feed, metrics, plotting, and Monte Carlo suites.

---

## Earlier changes (summary)

- **e09e420** — Single combined trades CSV; engine exit-only columns; column gap reporting; verify script.
- **d2c757d** — Add `split_entry_exit` option to `write_trades_csv` (trades, entry_columns, exit_columns).
- **ea1d1af** — Fix zero MFE/MAE/unrealized PL: build enriched_long and pass to engine for ATR.
- **c137d9e** — Fix unrealized PL snapshots (nearest bar); rename `Col_DistToInitialStop_R` to `Col_MAE_R`.
- **555112b** — librarycolumn_enrichment: vectorize wide_to_long, parallelize enrich_cleaned_year_data.
- **f6a04f4** — Post-backtest enrichment: load data, run backtest, then compute Entry/Exit/Continuous Col_* for trades only.
- **b8de235** — Docs: Library columns safety section in README; strategy_template note on Col_* at entry.
- **3d38a9a** — Safety: copy entry_column_snapshot per position to avoid same-bar shared dict.
- **068aa72** — Docs and I/O: README, write_trades_csv/Excel, run_backbone_template updates.
- **2d467ad** — Engine: `use_library_columns` flag (default on); entry/exit columns at entry/exit time; columns/io/plotting/sizers; tests.
- **4f7d1c6** — README: document engine 3-tuple and result.analyzers; build_full_metrics use pre-computed FullMetrics when available.
- **40178fa** — Engine returns (results, metrics_df, equity_curves); metrics merged into analyzers.
- **2443128** — Add test suite, parquet cache, metrics fix, risk-based sizing.
- **91af42a** — GapperDataLoader: parquet cache + _meta.json, no pickle; LoaderConfig cache_file→cache_dir.

---

For full project overview, install, and usage, see [README.md](README.md).
