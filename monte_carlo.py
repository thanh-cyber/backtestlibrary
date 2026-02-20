from __future__ import annotations

import numpy as np
import pandas as pd


def _res_get(res, key: str, default=None):
    if isinstance(res, dict):
        return res.get(key, default)
    return getattr(res, key, default)


def run_monte_carlo(
    results_by_year_account: dict,
    account_sizes: list[int],
    mc_runs: int = 5000,
    seed: int = 42,
    ruin_level: float = 0.0,
    deep_dd_pct: float = -50.0,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """Monte Carlo bootstrap on trade-level net_pnl.

    Returns:
        mc_df: aggregate monte carlo statistics.
        finals_cache: per (year, account) final balance distribution.
    """

    rng = np.random.default_rng(seed)
    rows = []
    finals_cache: dict[tuple[str, int], np.ndarray] = {}

    for year in sorted(results_by_year_account.keys()):
        for acct in account_sizes:
            res = results_by_year_account.get(year, {}).get(acct)
            if not res:
                continue
            t = _res_get(res, "trades")
            if t is None or t.empty or "net_pnl" not in t.columns:
                continue

            pnl = t["net_pnl"].astype(float).values
            n_trades = len(pnl)
            if n_trades == 0:
                continue

            samples = rng.choice(pnl, size=(mc_runs, n_trades), replace=True)
            equity_paths = acct + np.cumsum(samples, axis=1)
            final_balances = equity_paths[:, -1]
            peaks = np.maximum.accumulate(equity_paths, axis=1)
            path_dd = (equity_paths - peaks) / peaks
            max_dd_per_path = path_dd.min(axis=1) * 100.0

            finals_cache[(year, acct)] = final_balances
            rows.append(
                {
                    "year": year,
                    "starting_account": acct,
                    "Trades": n_trades,
                    "MC Runs": mc_runs,
                    "MC Mean Final $": float(np.mean(final_balances)),
                    "MC Median Final $": float(np.median(final_balances)),
                    "MC P5 Final $": float(np.percentile(final_balances, 5)),
                    "MC P25 Final $": float(np.percentile(final_balances, 25)),
                    "MC P75 Final $": float(np.percentile(final_balances, 75)),
                    "MC P95 Final $": float(np.percentile(final_balances, 95)),
                    "Prob(Final < Start) %": float((final_balances < acct).mean() * 100),
                    f"Prob(Ruin <= {ruin_level:g}) %": float((final_balances <= ruin_level).mean() * 100),
                    f"Prob(Max DD <= {deep_dd_pct:g}%) %": float((max_dd_per_path <= deep_dd_pct).mean() * 100),
                    "MC Mean Max DD %": float(np.mean(max_dd_per_path)),
                    "MC P5 Max DD %": float(np.percentile(max_dd_per_path, 5)),
                    "MC P95 Max DD %": float(np.percentile(max_dd_per_path, 95)),
                }
            )

    return pd.DataFrame(rows), finals_cache

