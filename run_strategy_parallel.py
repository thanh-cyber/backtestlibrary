#!/usr/bin/env python3
"""
Strategy-agnostic parallel per-year backtest runner.

Usage:
  From a project that has a strategy in a Python module:
    python -m backtestlibrary.run_strategy_parallel \
      --strategy-module my_strategy \
      --strategy-class MyStrategy \
      --years 2022,2023 \
      --accounts 2000,1000000 \
      --session-start 4:00 --session-end 9:24 \
      --cache-subdir my_strategy \
      --output-slug my_strategy

  Strategy module must define the strategy class with prepare_day, find_entries_for_day, check_exit.
  Optional: SESSION_START, SESSION_END, CACHE_SUBDIR (used if CLI not provided).
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _ensure_project_root_on_path() -> str:
    cwd = os.getcwd()
    project_root = cwd if os.path.isdir(os.path.join(cwd, "alldata")) else os.path.dirname(cwd)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from backtestlibrary import BacktestConfig, ChronologicalBacktestEngine  # noqa: E402
from backtestdata import load_cleaned_year_data  # noqa: E402
from backtestdata.loader import get_parquet_dir  # noqa: E402


def _parse_time(s: str) -> time:
    """Parse '4:00' or '9:24' -> time(4, 0) or time(9, 24)."""
    part = s.strip().split(":")
    hour = int(part[0])
    minute = int(part[1]) if len(part) > 1 else 0
    return time(hour, minute)


@dataclass(frozen=True)
class YearRunConfig:
    year: int
    parquet_dir: str
    cache_dir: str
    starting_accounts: tuple[int, ...]
    results_dir: str
    force: bool
    session_start: time
    session_end: time
    strategy_module: str
    strategy_class: str
    strategy_kwargs: tuple[tuple[str, Any], ...]  # serializable (key, value) pairs
    fixed_risk_per_trade: Optional[float] = None
    risk_pct_per_trade: float = 0.05
    session_name: str = "pm_only"  # for load_cleaned_year_data session
    output_slug: str = "strategy"
    project_root: str = ""  # so workers can sys.path.insert(0, project_root) before importing strategy


def _year_pickle_path(results_dir: str, year: int, output_slug: str, suffix: str = "") -> str:
    return str(Path(results_dir) / f"{output_slug}_year_{year}{suffix}.pkl")


def _run_one_year(cfg: YearRunConfig) -> tuple[int, str]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if getattr(cfg, "project_root", None) and cfg.project_root not in sys.path:
        sys.path.insert(0, cfg.project_root)
    mod = importlib.import_module(cfg.strategy_module)
    StrategyClass = getattr(mod, cfg.strategy_class)
    strategy_kwargs = dict(cfg.strategy_kwargs)
    strategy = StrategyClass(**strategy_kwargs)

    suffix = "_d100" if cfg.fixed_risk_per_trade is not None else ""
    out_path = _year_pickle_path(cfg.results_dir, cfg.year, cfg.output_slug, suffix=suffix)
    if (not cfg.force) and os.path.isfile(out_path):
        return (cfg.year, out_path)

    cleaned_year_data = load_cleaned_year_data(
        years=[cfg.year],
        parquet_dir=cfg.parquet_dir,
        cache_dir=cfg.cache_dir,
        show_progress=False,
        session=cfg.session_name,
    )

    if cfg.fixed_risk_per_trade is not None:
        risk_kw = {"fixed_risk_per_trade": cfg.fixed_risk_per_trade, "risk_pct_per_trade": None}
    else:
        risk_kw = {"risk_pct_per_trade": cfg.risk_pct_per_trade}

    engine = ChronologicalBacktestEngine(
        BacktestConfig(
            session_start=cfg.session_start,
            session_end=cfg.session_end,
            timeline_step_seconds=60,
            use_library_columns=True,
            defer_column_phase=True,
            **risk_kw,
        )
    )
    raw_results, metrics_df, equity_curves = engine.run(
        cleaned_year_data, strategy, list(cfg.starting_accounts), show_progress=False
    )

    with open(out_path, "wb") as f:
        pickle.dump(
            {"year": cfg.year, "raw_results": raw_results, "metrics_df": metrics_df, "equity_curves": equity_curves},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return (cfg.year, out_path)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Strategy-agnostic parallel per-year backtest runner.")
    ap.add_argument("--strategy-module", required=True, help="Dotted module path (e.g. strategies.my_strategy)")
    ap.add_argument("--strategy-class", required=True, help="Strategy class name (e.g. MyStrategy)")
    ap.add_argument("--years", default="2022,2023,2024,2025", help="Comma-separated years")
    ap.add_argument("--workers", type=int, default=10, help="Max worker processes")
    ap.add_argument("--force", action="store_true", help="Re-run even if year pickles exist")
    ap.add_argument("--accounts", default="2000,1000000", help="Comma-separated starting accounts")
    ap.add_argument("--session-start", default="4:00", help="Session start time (e.g. 4:00)")
    ap.add_argument("--session-end", default="9:24", help="Session end time (e.g. 9:24)")
    ap.add_argument("--cache-subdir", default=None, help="Subdir under cache/ (default: last part of strategy-module)")
    ap.add_argument("--output-slug", default=None, help="Prefix for pickle files (default: same as cache-subdir)")
    ap.add_argument("--results-dir", default=None, help="Directory for pickles (default: PROJECT_ROOT/backtest_results)")
    ap.add_argument("--fixed-risk", type=float, default=None, help="Fixed $ risk per trade; default = risk-pct of account")
    ap.add_argument("--risk-pct", type=float, default=0.05, help="Risk per trade as fraction of account (default 0.05)")
    ap.add_argument("--session", default="pm_only", help="Data session: pm_only, rth_only, etc.")
    args = ap.parse_args(argv)

    years = [int(x.strip()) for x in str(args.years).split(",") if x.strip()]
    starting_accounts = tuple(int(x.strip().replace("_", "")) for x in str(args.accounts).split(",") if x.strip())
    fixed_risk = float(args.fixed_risk) if args.fixed_risk is not None else None
    session_start = _parse_time(args.session_start)
    session_end = _parse_time(args.session_end)

    slug = args.cache_subdir or args.strategy_module.split(".")[-1]
    cache_subdir = slug
    output_slug = args.output_slug or slug

    parquet_base = Path(get_parquet_dir()).resolve()
    minute_dir = parquet_base / "us_stocks_sip" / "minute_aggs_v1"
    parquet_dir = str(minute_dir) if minute_dir.exists() else str(parquet_base)
    bd_root = parquet_base.parent
    cache_dir = str(bd_root / "cache" / cache_subdir)
    os.makedirs(cache_dir, exist_ok=True)

    results_dir = args.results_dir or os.path.join(PROJECT_ROOT, "backtest_results")
    os.makedirs(results_dir, exist_ok=True)

    # Strategy kwargs: empty by default; strategy module can expose STRATEGY_KWARGS or we add CLI later
    strategy_kwargs: tuple[tuple[str, Any], ...] = ()

    print("Strategy:", args.strategy_module, args.strategy_class)
    print("Parquet dir:", parquet_dir)
    print("Cache dir:", cache_dir)
    print("Years:", years)
    print("Workers:", args.workers)
    print("Accounts:", starting_accounts)
    print("Session:", session_start, "->", session_end)
    print("Risk:", f"${fixed_risk:.0f}/trade" if fixed_risk is not None else f"{args.risk_pct*100:.0f}% of account")

    cfgs = [
        YearRunConfig(
            year=y,
            parquet_dir=parquet_dir,
            cache_dir=cache_dir,
            starting_accounts=starting_accounts,
            results_dir=results_dir,
            force=bool(args.force),
            session_start=session_start,
            session_end=session_end,
            strategy_module=args.strategy_module,
            strategy_class=args.strategy_class,
            strategy_kwargs=strategy_kwargs,
            fixed_risk_per_trade=fixed_risk,
            risk_pct_per_trade=float(args.risk_pct),
            session_name=args.session,
            output_slug=output_slug,
            project_root=PROJECT_ROOT,
        )
        for y in years
    ]

    out_paths: dict[int, str] = {}
    workers = max(1, min(int(args.workers), len(cfgs)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one_year, c) for c in cfgs]
        for fut in as_completed(futs):
            y, p = fut.result()
            out_paths[y] = p
            print(f"Finished {y} -> {p}")

    combined_raw: dict[str, Any] = {}
    metrics_parts: list[pd.DataFrame] = []
    combined_equity: dict[Any, Any] = {}

    for y in sorted(out_paths):
        with open(out_paths[y], "rb") as f:
            payload = pickle.load(f)
        raw_results = payload.get("raw_results", {})
        metrics_df = payload.get("metrics_df", None)
        equity_curves = payload.get("equity_curves", {})

        for year_key, by_acct in (raw_results or {}).items():
            combined_raw[str(year_key)] = by_acct
        if metrics_df is not None and hasattr(metrics_df, "empty") and not metrics_df.empty:
            metrics_parts.append(metrics_df)
        if equity_curves:
            combined_equity.update(equity_curves)

    combined_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    suffix = "_d100" if fixed_risk is not None else ""
    combined_path = os.path.join(
        results_dir, f"{output_slug}_phase1_cache_" + "_".join(map(str, years)) + suffix + ".pkl"
    )
    with open(combined_path, "wb") as f:
        pickle.dump(
            {"raw_results": combined_raw, "metrics_df": combined_metrics, "equity_curves": combined_equity},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Wrote combined:", combined_path)
    if not combined_metrics.empty and "total_trades" in combined_metrics.columns:
        print("Total trades:", int(combined_metrics["total_trades"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
