"""
Modular analyzer system for backtest results.
Each analyzer implements the Analyzer protocol and can be attached to the engine.
Also holds metric helpers and build_full_metrics (replacing the former metrics.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bt_types import Analyzer, RunResult


# ---------- Metric helpers (formerly in metrics.py) ----------
def _res_get(res, key: str, default=None):
    if isinstance(res, dict):
        return res.get(key, default)
    return getattr(res, key, default)


def _profit_factor(gross_pnl_series: pd.Series) -> float:
    wins = gross_pnl_series[gross_pnl_series > 0].sum()
    losses = abs(gross_pnl_series[gross_pnl_series < 0].sum())
    return float(wins / losses) if losses != 0 else float(np.inf)


def _expectancy(net_pnl_series: pd.Series) -> float:
    return float(net_pnl_series.mean()) if len(net_pnl_series) else 0.0


def _sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return np.nan
    excess = returns - risk_free
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(periods_per_year)) if std != 0 else np.nan


def _sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return np.nan
    downside = returns[returns < 0]
    if len(downside) < 2:
        return np.nan
    dstd = downside.std()
    return float((returns.mean() - risk_free) / dstd * np.sqrt(periods_per_year)) if dstd != 0 else np.nan


def _ulcer_index(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(np.sqrt(np.mean(drawdown**2)) * 100)


def _max_consecutive_losses(net_pnl_series: np.ndarray) -> int:
    mx = cur = 0
    for x in net_pnl_series:
        if x < 0:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return mx


# ---------- Analyzers ----------
class FullMetricsAnalyzer(Analyzer):
    """Produces the full metrics row for a single RunResult (same as one row of build_full_metrics)."""

    def analyze(self, result: RunResult) -> dict:
        t = result.trades
        if t is None or t.empty:
            return {}

        start_bal = result.final_balance - result.total_pnl
        final_bal = result.final_balance
        equity = np.array(result.daily_equity)
        daily_ret = pd.Series(equity).pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        gross_pnl = t["gross_pnl"].astype(float)
        net_pnl = t["net_pnl"].astype(float)
        num_trades = result.total_trades
        wins = result.winning_trades
        losses = result.losing_trades
        total_return = result.total_return_pct
        total_pnl = result.total_pnl

        peak_arr = np.maximum.accumulate(equity)
        dd = (equity - peak_arr) / np.maximum(peak_arr, 1e-15)
        max_dd = float(dd.min() * 100.0)
        avg_dd = float(dd.mean() * 100.0)
        ulcer = _ulcer_index(equity)

        avg_win = float(gross_pnl[gross_pnl > 0].mean()) if (gross_pnl > 0).any() else 0.0
        avg_loss = float(gross_pnl[gross_pnl < 0].mean()) if (gross_pnl < 0).any() else 0.0
        pf = _profit_factor(gross_pnl)
        expectancy = _expectancy(net_pnl)
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        stop_ct = int((t["exit_reason"] == "Stop Loss").sum()) if "exit_reason" in t.columns else 0
        take_ct = int((t["exit_reason"] == "Take Profit").sum()) if "exit_reason" in t.columns else 0
        time_ct = int((t["exit_reason"] == "Time Exit").sum()) if "exit_reason" in t.columns else 0
        stop_pct = (stop_ct / num_trades * 100.0) if num_trades else 0.0
        take_pct = (take_ct / num_trades * 100.0) if num_trades else 0.0
        time_pct = (time_ct / num_trades * 100.0) if num_trades else 0.0

        num_periods = len(daily_ret)
        years_equiv = max(num_periods / 252.0, 1e-9)
        cagr = (
            ((final_bal / start_bal) ** (1 / years_equiv) - 1) * 100.0
            if start_bal > 0 and num_periods >= 1
            else total_return
        )
        sharpe = _sharpe(daily_ret)
        sortino = _sortino(daily_ret)
        calmar = (cagr / 100.0) / abs(max_dd / 100.0) if max_dd != 0 else np.nan
        recovery_factor = (
            total_pnl / abs(max_dd * start_bal / 100.0)
            if max_dd != 0 and start_bal != 0
            else np.nan
        )

        skew = float(pd.Series(net_pnl).skew()) if len(net_pnl) > 2 else np.nan
        kurtosis = float(pd.Series(net_pnl).kurt()) if len(net_pnl) > 3 else np.nan
        avg_hold_min = float(t["hold_minutes"].mean()) if "hold_minutes" in t.columns else np.nan
        max_consec_losses = _max_consecutive_losses(net_pnl.values)
        win_rate = (wins / num_trades * 100.0) if num_trades else 0.0

        return {
            "final_balance": final_bal,
            "total_return_pct": total_return,
            "total_pnl": total_pnl,
            "total_trades": num_trades,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": win_rate,
            "avg_trade_pnl": (total_pnl / num_trades) if num_trades else 0.0,
            "stop_loss_exits": stop_ct,
            "take_profit_exits": take_ct,
            "time_exits": time_ct,
            "max_drawdown_pct": max_dd,
            "CAGR_pct": cagr,
            "avg_dd_pct": avg_dd,
            "ulcer_index": ulcer,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "profit_factor": pf,
            "recovery_factor": recovery_factor,
            "expectancy": expectancy,
            "payoff_ratio": payoff_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "stop_pct": stop_pct,
            "take_profit_pct": take_pct,
            "time_exit_pct": time_pct,
            "avg_hold_min": avg_hold_min,
            "max_consec_losses": max_consec_losses,
            "skew": skew,
            "kurtosis": kurtosis,
        }


class SQNAnalyzer(Analyzer):
    """System Quality Number: (expectancy / std_expectancy) * sqrt(n_trades)."""

    def analyze(self, result: RunResult) -> float:
        if result.total_trades < 2:
            return np.nan
        t = result.trades
        if t is None or t.empty:
            return np.nan
        expectancy = result.total_pnl / result.total_trades
        std_expectancy = t["net_pnl"].std()
        return (
            (expectancy / std_expectancy) * np.sqrt(result.total_trades)
            if std_expectancy != 0
            else np.nan
        )


class DrawdownDurationAnalyzer(Analyzer):
    """Max and average drawdown duration in bars (days)."""

    def analyze(self, result: RunResult) -> dict:
        equity = np.array(result.daily_equity)
        if len(equity) < 2:
            return {"max_dd_bars": 0, "avg_dd_bars": 0.0}
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.maximum(peak, 1e-15)
        in_dd = dd < 0
        padded = np.concatenate(([False], in_dd, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) == 0 or len(ends) == 0:
            return {"max_dd_bars": 0, "avg_dd_bars": 0.0}
        durations = ends - starts
        return {
            "max_dd_bars": int(durations.max()),
            "avg_dd_bars": float(durations.mean()),
        }


class LongShortAnalyzer(Analyzer):
    """Long vs short breakdown (when trades have 'side' column)."""

    def analyze(self, result: RunResult) -> dict:
        t = result.trades
        if t is None or t.empty or "side" not in t.columns:
            return {}
        long = t[t["side"] == "long"]
        short = t[t["side"] == "short"]
        return {
            "long_trades": len(long),
            "long_win_rate": (long["net_pnl"] > 0).mean() * 100 if len(long) else 0,
            "short_trades": len(short),
            "short_win_rate": (short["net_pnl"] > 0).mean() * 100 if len(short) else 0,
        }


class StreakAnalyzer(Analyzer):
    """Max win streak and max loss streak."""

    def analyze(self, result: RunResult) -> dict:
        t = result.trades
        if t is None or t.empty or "net_pnl" not in t.columns:
            return {"max_win_streak": 0, "max_loss_streak": 0}
        pnl = t["net_pnl"].values
        max_win_streak = max_loss_streak = cur_win = cur_loss = 0
        for x in pnl:
            if x > 0:
                cur_win += 1
                max_win_streak = max(max_win_streak, cur_win)
                cur_loss = 0
            else:
                cur_loss += 1
                max_loss_streak = max(max_loss_streak, cur_loss)
                cur_win = 0
        return {"max_win_streak": max_win_streak, "max_loss_streak": max_loss_streak}


# ---------- Convenience API (replaces metrics.build_full_metrics) ----------
def build_full_metrics(
    results_by_year_account: dict,
    account_sizes: list[int],
) -> tuple[pd.DataFrame, dict[tuple[str, int], list[float]]]:
    """Build a unified full metrics dataframe + equity curves. Uses FullMetricsAnalyzer."""
    rows = []
    equity_curves: dict[tuple[str, int], list[float]] = {}
    full_analyzer = FullMetricsAnalyzer()

    for year in sorted(results_by_year_account.keys()):
        for acct in account_sizes:
            res = results_by_year_account.get(year, {}).get(acct)
            if not res:
                continue
            t = _res_get(res, "trades")
            if t is None or t.empty:
                continue

            if isinstance(res, RunResult):
                run_result = res
                # Use pre-computed FullMetrics when available (engine already ran analyzers)
                row = None
                if res.analyzers and "FullMetrics" in res.analyzers:
                    row = dict(res.analyzers["FullMetrics"])
            else:
                start_bal = float(acct)
                final_bal = float(_res_get(res, "final_balance", start_bal))
                total_pnl = float(_res_get(res, "total_pnl", final_bal - start_bal))
                daily_eq = _res_get(res, "daily_equity") or []
                run_result = RunResult(
                    final_balance=final_bal,
                    total_return_pct=float(_res_get(res, "total_return_pct", 0.0)),
                    total_trades=int(_res_get(res, "total_trades", len(t))),
                    winning_trades=int(_res_get(res, "winning_trades", 0)),
                    losing_trades=int(_res_get(res, "losing_trades", max(0, len(t) - _res_get(res, "winning_trades", 0)))),
                    total_pnl=total_pnl,
                    trades=t,
                    daily_equity=daily_eq if len(daily_eq) > 1 else [start_bal] + t["account_balance_after"].tolist(),
                )
                row = None
            if row is None:
                row = full_analyzer.analyze(run_result)
            if not row:
                continue
            row["year"] = year
            row["starting_account"] = acct
            rows.append(row)
            equity_curves[(year, acct)] = list(run_result.daily_equity)

    return pd.DataFrame(rows), equity_curves


DEFAULT_ANALYZERS: list[Analyzer] = [
    FullMetricsAnalyzer(),
    SQNAnalyzer(),
    DrawdownDurationAnalyzer(),
    LongShortAnalyzer(),
    StreakAnalyzer(),
]
