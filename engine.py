from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from .bt_types import Position, RunResult, Strategy, TradeRecord
from .analyzers import build_full_metrics
from .sizers import RiskSizer
from .columns import apply_entry_columns, apply_exit_columns
from .trade_enrichment import enrich_trades_post_backtest


@dataclass
class BacktestConfig:
    session_start: time
    session_end: time
    fixed_risk_per_trade: Optional[float] = None  # Fixed $ risk per trade (e.g. 500)
    risk_pct_per_trade: Optional[float] = None   # % of account risk per trade (e.g. 0.05 = 5%); used by RiskSizer
    margin_requirement: float = 1.0
    sizer: Optional[Any] = None  # Pluggable sizer (FixedSizeSizer, PercentOfEquitySizer, RiskSizer, KellySizer). If None, uses RiskSizer(config).
    float_cap_pct: float = 0.05
    equity_cap_pct: float = 0.10
    absolute_cap_value: float = 1_000_000.0
    commission: float = 3.0
    sec_taf_fee_per_share: float = 0.00013
    slippage_pct: float = 0.005
    gst_rate: float = 0.10
    borrow_annual_base: float = 0.30
    borrow_annual_high: float = 0.50
    timeline_step_seconds: int = 60
    float_col: str = "Float_Numeric"
    exit_price_col: str = "Exit_Price"
    use_library_columns: bool = True  # When True, engine runs entry/exit columns from librarycolumn at entry/exit time; default on so columns are always included


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _time_to_seconds(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def _minutes_between(start: time, end: time) -> int:
    return max(1, _time_to_minutes(end) - _time_to_minutes(start))


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        return x if np.isfinite(x) else None
    s = str(value).replace("$", "").replace(",", "").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def _pl_r_for_side(side: str, entry_price: float, current_price: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if side.lower() == "short":
        return (entry_price - current_price) / atr
    return (current_price - entry_price) / atr


def _iter_times(start: time, end: time, step_seconds: int) -> list[time]:
    step_s = max(1, int(step_seconds))
    current = _time_to_seconds(start)
    end_s = _time_to_seconds(end)
    out: list[time] = []
    while current <= end_s:
        hh = current // 3600
        rem = current % 3600
        mm = rem // 60
        ss = rem % 60
        out.append(time(hh, mm, ss))
        current += step_s
    return out


def _time_candidates(t: time) -> list[object]:
    return [
        f"{t.hour}:{t.minute:02d}",
        f"{t.hour:02d}:{t.minute:02d}",
        f"{t.hour}:{t.minute:02d}:00",
        f"{t.hour:02d}:{t.minute:02d}:00",
        f"{t.hour}:{t.minute:02d}:30",
        f"{t.hour:02d}:{t.minute:02d}:30",
        time(t.hour, t.minute, 0),
        time(t.hour, t.minute, 30),
    ]


def _time_candidates_utc_fallback(t: time) -> list[str]:
    """Polygon flat files: cache may use UTC column names (9:30 ET = 14:30 UTC EST, 13:30 EDT)."""
    out = []
    for offset in (5, 4):
        uh = (t.hour + offset) % 24
        out.append(f"{uh}:{t.minute:02d}")
        out.append(f"{uh:02d}:{t.minute:02d}")
    return out


def get_row_price(row: pd.Series, t: time) -> Optional[float]:
    """Robustly read a bar price from wide time-column row."""
    candidates = _time_candidates(t)
    for c in candidates:
        if c in row.index:
            px = _to_float(row[c])
            if px is not None and px > 0:
                return px
    # Fallback for mixed string types
    str_candidates = {str(c) for c in candidates}
    for col in row.index:
        col_str = str(col)
        if col_str in str_candidates or any(col_str.endswith(sc) for sc in str_candidates):
            px = _to_float(row[col])
            if px is not None and px > 0:
                return px
    # Polygon/cache: try UTC column names (9:30 ET -> 14:30 or 13:30 UTC)
    for c in _time_candidates_utc_fallback(t):
        if c in row.index:
            px = _to_float(row[c])
            if px is not None and px > 0:
                return px
    return None


def _to_float_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    out = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(out, errors="coerce")


def _build_day_price_cache(
    day_df: pd.DataFrame,
    timeline: list[time],
) -> tuple[dict[object, pd.Series], dict[time, int], np.ndarray, Callable[[pd.Series, time], Optional[float]]]:
    """Vectorized per-day price matrix for fast repeated lookups.

    Keeps execution logic unchanged while reducing repeated column scans.
    """
    row_lookup = {idx: row for idx, row in day_df.iterrows()}
    n_rows = len(day_df)
    n_times = len(timeline)
    matrix = np.full((n_rows, n_times), np.nan, dtype=float)
    idx_to_pos = {idx: pos for pos, idx in enumerate(day_df.index)}

    for col_pos, t in enumerate(timeline):
        chosen_col = None
        for cand in _time_candidates(t):
            if cand in day_df.columns:
                chosen_col = cand
                break
        if chosen_col is None:
            # Fallback by string-equality for odd spreadsheet column types.
            cand_str = {str(c) for c in _time_candidates(t)}
            for col in day_df.columns:
                col_s = str(col)
                if col_s in cand_str or any(col_s.endswith(cs) for cs in cand_str):
                    chosen_col = col
                    break
        if chosen_col is None:
            # Polygon/cache: try UTC column names (9:30 ET -> 14:30 or 13:30 UTC)
            for c in _time_candidates_utc_fallback(t):
                if c in day_df.columns:
                    chosen_col = c
                    break
        if chosen_col is None:
            continue
        col_values = _to_float_series(day_df[chosen_col]).to_numpy(dtype=float, na_value=np.nan)
        matrix[:, col_pos] = col_values

    time_to_pos = {t: i for i, t in enumerate(timeline)}

    def fast_get(row: pd.Series, t: time) -> Optional[float]:
        row_pos = idx_to_pos.get(row.name)
        t_pos = time_to_pos.get(t)
        if row_pos is None or t_pos is None:
            return get_row_price(row, t)
        px = matrix[row_pos, t_pos]
        if np.isfinite(px) and px > 0:
            return float(px)
        return None

    return row_lookup, time_to_pos, matrix, fast_get


def _supports_positional_args(fn, arg_count: int) -> bool:
    """Check if callable accepts at least arg_count positional-or-keyword args."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= arg_count


class ChronologicalBacktestEngine:
    """Chronological engine with pluggable entry/exit strategy callbacks."""

    def __init__(self, config: BacktestConfig, analyzers=None):
        self.config = config
        self.analyzers = analyzers or []
        self.sizer = config.sizer if config.sizer is not None else RiskSizer()

    def run(
        self,
        cleaned_year_data: dict[str, Union[pd.DataFrame, Path]],
        strategy: Strategy,
        starting_accounts: list[int],
        chunk_days: int = 21,
        show_progress: bool = False,
    ) -> tuple[dict[str, dict[int, RunResult]], pd.DataFrame, dict[tuple[str, int], list[float]]]:
        """Run backtest per year/account; returns (results, metrics_df, equity_curves).
        When a value is a Path (stream_from_cache), the year is read in date-chunks to avoid loading full year in RAM.
        chunk_days: number of trading days per chunk when streaming from Path (default 21 ~ 1 month).
        show_progress: if True, show a tqdm progress bar (0-100% by day or chunk).
        Entry/Exit/Continuous columns are applied in a post-backtest enrichment pass.
        """
        results: dict[str, dict[int, RunResult]] = {}
        iter_items = list(cleaned_year_data.items())
        # Paths for streaming: used by post-enrichment to load wide data for trades
        wide_path_by_year = {
            k: Path(v) for k, v in cleaned_year_data.items()
            if isinstance(v, (Path, str))
        }

        # Precompute total steps so the bar shows 0-100% (per day for in-memory, per chunk for streamed)
        total_steps = 0
        if show_progress:
            for year, data in iter_items:
                if isinstance(data, (Path, str)):
                    path = Path(data)
                    path_str = str(path.resolve())
                    try:
                        dates_df = pd.read_parquet(path_str, columns=["Date"])
                    except Exception:
                        dates_df = pd.read_parquet(path_str)
                    if "Date" not in dates_df.columns:
                        continue
                    dates_df["Date"] = pd.to_datetime(dates_df["Date"], errors="coerce").dt.normalize()
                    n_dates = dates_df["Date"].dropna().nunique()
                    n_chunks = (n_dates + chunk_days - 1) // chunk_days if n_dates else 0
                    total_steps += n_chunks * len(starting_accounts)
                else:
                    df = data
                    if "Date" in df.columns:
                        df = df.copy()
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
                        df = df.dropna(subset=["Date"])
                        n_days = df.groupby("Date").ngroups
                    else:
                        n_days = 0
                    total_steps += n_days * len(starting_accounts)
        pbar = None
        if show_progress and total_steps > 0:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=total_steps, desc="Backtest", unit="day")
            except ImportError:
                pass

        for year, data in iter_items:
            results[year] = {}
            if isinstance(data, (Path, str)):
                path = Path(data)
                for starting_account in starting_accounts:
                    run_result = self._run_single_account_streamed(
                        path, starting_account, strategy, chunk_days, pbar=pbar
                    )
                    run_result = self._enrich_trades(run_result, cleaned_year_data, wide_path_by_year)
                    results[year][starting_account] = run_result
            else:
                df_year = data
                if "Date" in df_year.columns:
                    df_year = df_year.copy()
                    df_year["Date"] = pd.to_datetime(df_year["Date"], errors="coerce").dt.normalize()
                    df_year = df_year.dropna(subset=["Date"]).sort_values("Date")
                daily_groups = df_year.groupby("Date")
                for starting_account in starting_accounts:
                    run_result = self._run_single_account(
                        daily_groups,
                        starting_account,
                        strategy,
                        pbar=pbar,
                        enriched_long_df=None,
                    )
                    run_result = self._enrich_trades(run_result, cleaned_year_data, wide_path_by_year)
                    results[year][starting_account] = run_result

        if pbar is not None:
            pbar.close()
        metrics_df, equity_curves = build_full_metrics(results, starting_accounts)
        return results, metrics_df, equity_curves

    def _enrich_trades(
        self,
        result: RunResult,
        cleaned_year_data: dict,
        wide_path_by_year: dict,
    ) -> RunResult:
        """Post-backtest enrichment: Entry_Col_*, Exit_Col_*, Continuous_Col_*."""
        return enrich_trades_post_backtest(
            result,
            cleaned_year_data,
            wide_path_by_year,
            session_start=self.config.session_start,
            session_end=self.config.session_end,
            config=self.config,
        )

    def _run_single_account_streamed(
        self,
        path: Path,
        starting_account: int,
        strategy: Strategy,
        chunk_days: int,
        pbar: Optional[Any] = None,
    ) -> RunResult:
        """Run backtest for one account by reading parquet in date-chunks; merge results."""
        path = Path(path)
        path_str = str(path.resolve())
        # Get unique dates without loading full file (read only Date column)
        try:
            dates_df = pd.read_parquet(path_str, columns=["Date"])
        except Exception:
            dates_df = pd.read_parquet(path_str)
            if "Date" not in dates_df.columns:
                return RunResult(
                    final_balance=float(starting_account),
                    total_return_pct=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    total_pnl=0.0,
                    trades=pd.DataFrame(),
                    daily_equity=[float(starting_account)],
                )
        dates_df["Date"] = pd.to_datetime(dates_df["Date"], errors="coerce").dt.normalize()
        unique_dates = sorted(dates_df["Date"].dropna().unique().tolist())
        del dates_df
        if not unique_dates:
            return RunResult(
                final_balance=float(starting_account),
                total_return_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                trades=pd.DataFrame(),
                daily_equity=[float(starting_account)],
            )
        all_trades: list[dict] = []
        daily_equity: list[float] = [float(starting_account)]
        balance = float(starting_account)
        for i in range(0, len(unique_dates), chunk_days):
            chunk_dates = unique_dates[i : i + chunk_days]
            d_min, d_max = chunk_dates[0], chunk_dates[-1]
            try:
                df_chunk = pd.read_parquet(path_str, filters=[("Date", ">=", d_min), ("Date", "<=", d_max)])
            except Exception:
                df_chunk = pd.read_parquet(path_str)
                df_chunk = df_chunk[
                    (pd.to_datetime(df_chunk["Date"], errors="coerce").dt.normalize() >= pd.Timestamp(d_min))
                    & (pd.to_datetime(df_chunk["Date"], errors="coerce").dt.normalize() <= pd.Timestamp(d_max))
                ]
            if df_chunk.empty:
                continue
            df_chunk["Date"] = pd.to_datetime(df_chunk["Date"], errors="coerce").dt.normalize()
            df_chunk = df_chunk.dropna(subset=["Date"]).sort_values("Date")
            daily_groups = df_chunk.groupby("Date")
            run = self._run_single_account(daily_groups, int(round(balance)), strategy, pbar=None)
            all_trades.extend(run.trades.to_dict("records") if not run.trades.empty else [])
            balance = run.final_balance
            if run.daily_equity:
                daily_equity.extend(run.daily_equity[1:])
            del df_chunk
            if pbar is not None:
                pbar.update(1)
        total_pnl = balance - float(starting_account)
        total_return_pct = (total_pnl / float(starting_account) * 100.0) if starting_account > 0 else 0.0
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        total_trades = len(all_trades)
        winning_trades = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
        losing_trades = total_trades - winning_trades
        result = RunResult(
            final_balance=balance,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            trades=trades_df,
            daily_equity=daily_equity,
        )
        for analyzer in self.analyzers:
            name = analyzer.__class__.__name__.replace("Analyzer", "")
            result.analyzers[name] = analyzer.analyze(result)
        return result

    def _run_single_account(
        self,
        daily_groups,
        starting_account: int,
        strategy: Strategy,
        pbar: Optional[Any] = None,
        enriched_long_df: Optional[pd.DataFrame] = None,
    ) -> RunResult:
        get_row_at_time = None
        if enriched_long_df is not None:
            from .librarycolumn_enrichment import get_row_at_time as _get_row_at_time
            get_row_at_time = _get_row_at_time

        account_balance = float(starting_account)
        available_balance = float(starting_account)
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        trades_list: list[dict] = []
        daily_equity: list[float] = [float(starting_account)]

        for date, df_day in daily_groups:
            timeline = _iter_times(self.config.session_start, self.config.session_end, self.config.timeline_step_seconds)
            row_lookup, _time_pos, _price_matrix, get_price = _build_day_price_cache(df_day, timeline)
            day_context = None
            prepare_day = getattr(strategy, "prepare_day", None)
            if callable(prepare_day):
                day_context = prepare_day(df_day, timeline, get_price)

            find_entries = strategy.find_entries_for_day
            entries_with_context = _supports_positional_args(find_entries, 4)
            if entries_with_context:
                entries = find_entries(df_day, timeline, get_price, day_context) or []
            else:
                entries = find_entries(df_day, timeline, get_price) or []
            entries.sort(key=lambda x: (x.entry_time.hour, x.entry_time.minute, x.entry_time.second, str(x.ticker)))

            entries_by_time: dict[time, list] = {}
            for e in entries:
                entries_by_time.setdefault(e.entry_time, []).append(e)

            check_exit = strategy.check_exit
            exits_with_context = _supports_positional_args(check_exit, 5)
            open_positions: list[Position] = []
            for current_time in timeline:
                # 1) Enter new positions in strict chronological order.
                for entry in entries_by_time.get(current_time, []):
                    row = row_lookup.get(entry.row_index)
                    if row is None:
                        continue
                    shares, entry_value = self._size_position(
                        entry.entry_price, row, account_balance,
                        stop_price=entry.stop_price, side=entry.side,
                    )
                    if shares <= 0 or entry_value <= 0:
                        continue
                    required_margin = entry_value * self.config.margin_requirement
                    if available_balance < required_margin:
                        continue

                    entry_column_snapshot: Optional[dict] = None
                    if self.config.use_library_columns:
                        row_for_entry = row
                        if get_row_at_time is not None:
                            row_at_entry = get_row_at_time(
                                enriched_long_df, entry.ticker, date, entry.entry_time
                            )
                            if row_at_entry is not None:
                                row_for_entry = row_at_entry
                        entry_column_snapshot = {}
                        apply_entry_columns(entry_column_snapshot, row_for_entry)
                        entry_column_snapshot = entry_column_snapshot.copy()

                    open_positions.append(
                        Position(
                            ticker=entry.ticker,
                            row_index=entry.row_index,
                            side=entry.side,
                            shares=shares,
                            entry_price=entry.entry_price,
                            entry_time=entry.entry_time,
                            stop_price=entry.stop_price,
                            target_price=entry.target_price,
                            metadata=entry.metadata,
                            starting_account=float(starting_account),
                            entry_column_snapshot=entry_column_snapshot,
                        )
                    )
                    available_balance -= required_margin

                # 2) Evaluate exits.
                still_open: list[Position] = []
                for i, pos in enumerate(open_positions):
                    row = row_lookup.get(pos.row_index)
                    if row is None:
                        still_open.append(pos)
                        continue
                    # ---- Update elite exit tracking every bar ----
                    current_price = get_price(row, current_time)
                    atr = _to_float(row.get("Col_ATR14"))
                    if current_price is not None and current_price > 0 and atr is not None and atr > 0:
                        current_pl_r = _pl_r_for_side(pos.side, pos.entry_price, current_price, atr)
                        if current_pl_r > pos.mfe_r:
                            pos.mfe_r = current_pl_r
                            pos.bars_to_mfe = pos.bars_since_entry
                            pos.peak_pl_r = current_pl_r
                        if current_pl_r < pos.mae_r:
                            pos.mae_r = current_pl_r
                            pos.bars_to_mae = pos.bars_since_entry
                        pos.max_dd_from_mfe = min(pos.max_dd_from_mfe, current_pl_r - pos.peak_pl_r)

                        hm = f"{current_time.hour:02d}{current_time.minute:02d}"
                        if hm == "1000":
                            pos.unrealized_pl_1000 = current_pl_r
                        elif hm == "1030":
                            pos.unrealized_pl_1030 = current_pl_r
                        elif hm == "1100":
                            pos.unrealized_pl_1100 = current_pl_r
                        elif hm == "1130":
                            pos.unrealized_pl_1130 = current_pl_r
                        elif hm == "1200":
                            pos.unrealized_pl_1200 = current_pl_r
                        elif hm == "1230":
                            pos.unrealized_pl_1230 = current_pl_r
                        elif hm == "1300":
                            pos.unrealized_pl_1300 = current_pl_r
                        elif hm == "1330":
                            pos.unrealized_pl_1330 = current_pl_r
                        elif hm == "1400":
                            pos.unrealized_pl_1400 = current_pl_r
                        elif hm == "1430":
                            pos.unrealized_pl_1430 = current_pl_r
                        elif hm == "1500":
                            pos.unrealized_pl_1500 = current_pl_r
                        elif hm == "1530":
                            pos.unrealized_pl_1530 = current_pl_r
                        elif hm == "1600":
                            pos.unrealized_pl_1600 = current_pl_r

                    # Increment bar counter once per processed bar while position is open.
                    pos.bars_since_entry += 1

                    if exits_with_context:
                        signal = check_exit(pos, row, current_time, get_price, day_context)
                    else:
                        signal = check_exit(pos, row, current_time, get_price)

                    # Engine-level time exit safeguard.
                    if signal is None and current_time == self.config.session_end:
                        forced = _to_float(row.get(self.config.exit_price_col))
                        if forced is None:
                            forced = get_price(row, current_time)
                        if forced is not None and forced > 0:
                            from .bt_types import ExitSignal

                            signal = ExitSignal(exit_price=forced, reason="Time Exit")

                    if signal is None:
                        still_open.append(pos)
                        continue

                    net_pnl, trade = self._close_trade(
                        pos=pos,
                        row=row,
                        date=date,
                        exit_time=current_time,
                        exit_price=float(signal.exit_price),
                        exit_reason=signal.reason,
                    )
                    total_trades += 1
                    if net_pnl > 0:
                        winning_trades += 1
                    elif net_pnl < 0:
                        losing_trades += 1

                    account_balance += net_pnl
                    available_balance += pos.shares * pos.entry_price * self.config.margin_requirement + net_pnl
                    available_balance = max(0.0, available_balance)
                    trade.account_balance_after = account_balance
                    tdict = asdict(trade)
                    if self.config.use_library_columns:
                        row_for_exit = row
                        if get_row_at_time is not None:
                            row_at_exit = get_row_at_time(
                                enriched_long_df, pos.ticker, date, current_time
                            )
                            if row_at_exit is not None:
                                row_for_exit = row_at_exit
                        if pos.entry_column_snapshot:
                            tdict.update(pos.entry_column_snapshot)
                        apply_exit_columns(tdict, row_for_exit)
                    trades_list.append(tdict)

                    if account_balance <= 0:
                        still_open.extend(open_positions[i + 1 :])
                        break

                open_positions = still_open

            daily_equity.append(account_balance)
            if pbar is not None:
                pbar.update(1)
            if account_balance <= 0:
                break

        final_balance = account_balance
        total_pnl = final_balance - float(starting_account)
        total_return_pct = (total_pnl / float(starting_account) * 100.0) if starting_account > 0 else 0.0
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        result = RunResult(
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            trades=trades_df,
            daily_equity=daily_equity,
        )
        for analyzer in self.analyzers:
            name = analyzer.__class__.__name__.replace("Analyzer", "")
            result.analyzers[name] = analyzer.analyze(result)
        return result

    def _size_position(
        self,
        entry_price: float,
        row: pd.Series,
        account_balance: float,
        stop_price: Optional[float] = None,
        side: str = "long",
    ) -> tuple[int, float]:
        """Use configured sizer to get raw size, then apply float/equity/absolute caps."""
        if entry_price is None or not np.isfinite(entry_price) or entry_price <= 0:
            return 0, 0.0
        shares, _ = self.sizer.size(
            entry_price, row, account_balance, stop_price, side, self.config
        )
        if shares <= 0:
            return 0, 0.0
        caps: list[float] = []

        if self.config.float_col in row.index:
            f = _to_float(row.get(self.config.float_col))
            if f is not None and f > 0:
                caps.append(f * self.config.float_cap_pct)

        if account_balance > 0:
            caps.append((account_balance * self.config.equity_cap_pct) / entry_price)
        caps.append(self.config.absolute_cap_value / entry_price)

        if caps:
            shares = min(shares, *caps)
        shares = int(max(0, shares))
        return shares, float(shares * entry_price)

    def _close_trade(
        self,
        pos: Position,
        row: pd.Series,
        date: pd.Timestamp,
        exit_time: time,
        exit_price: float,
        exit_reason: str,
    ) -> tuple[float, TradeRecord]:
        entry_value = pos.shares * pos.entry_price
        exit_value = pos.shares * exit_price

        entry_slippage = entry_value * self.config.slippage_pct
        exit_slippage = exit_value * self.config.slippage_pct
        slippage_total = entry_slippage + exit_slippage
        hold_minutes = _minutes_between(pos.entry_time, exit_time)

        if pos.side.lower() == "short":
            borrow_annual = self.config.borrow_annual_high if entry_value > 1_000_000 else self.config.borrow_annual_base
            borrow_daily = borrow_annual / 365.0
            borrow_cost = entry_value * borrow_daily * (hold_minutes / (24.0 * 60.0))
        else:
            borrow_cost = 0.0
        sec_taf_fee = pos.shares * self.config.sec_taf_fee_per_share
        commission_total = self.config.commission
        gst = (commission_total + sec_taf_fee) * self.config.gst_rate
        total_costs = commission_total + sec_taf_fee + slippage_total + gst + borrow_cost

        if pos.side.lower() == "long":
            gross_pnl = pos.shares * (exit_price - pos.entry_price)
        else:
            gross_pnl = pos.shares * (pos.entry_price - exit_price)
        net_pnl = gross_pnl - total_costs

        trade = TradeRecord(
            ticker=pos.ticker,
            date=pd.Timestamp(date),
            entry_time=f"{pos.entry_time.hour:02d}:{pos.entry_time.minute:02d}",
            entry_price=pos.entry_price,
            exit_time=f"{exit_time.hour:02d}:{exit_time.minute:02d}",
            exit_price=exit_price,
            exit_reason=exit_reason,
            shares=pos.shares,
            hold_minutes=hold_minutes,
            gross_pnl=gross_pnl,
            commission=commission_total,
            sec_taf_fee=sec_taf_fee,
            slippage=slippage_total,
            gst=gst,
            borrow_cost=borrow_cost,
            net_pnl=net_pnl,
            account_balance_after=0.0,  # set by caller after balance update
        )
        # ---- Save elite exit analytics ----
        trade.Col_MaxFavorableExcursion_R = float(pos.mfe_r)
        trade.Col_DistToInitialStop_R = float(pos.mae_r)
        trade.Col_BarsToMFE = int(pos.bars_to_mfe)
        trade.Col_BarsToMAE = int(pos.bars_to_mae)
        trade.Col_MaxDrawdownFromMFE_R = float(pos.max_dd_from_mfe)
        atr = _to_float(row.get("Col_ATR14"))
        vwap = _to_float(row.get("Col_VWAP"))
        trade.Exit_Col_ATR14 = float(atr) if atr is not None else 0.0
        trade.Exit_Col_VWAP = float(vwap) if vwap is not None else 0.0
        if atr is not None and atr > 0 and pos.shares != 0:
            risk_dollar_unit = (pos.entry_price * abs(pos.shares)) / atr
            trade.Col_FinalPL_R = float(net_pnl / risk_dollar_unit) if risk_dollar_unit > 0 else 0.0
            if vwap is not None:
                if pos.side.lower() == "short":
                    trade.Col_ExitVWAPDeviation_ATR = float((vwap - exit_price) / atr)
                else:
                    trade.Col_ExitVWAPDeviation_ATR = float((exit_price - vwap) / atr)
        trade.Col_HoldMinutes = int(hold_minutes)
        trade.Col_ExitHourNumeric = float(exit_time.hour + exit_time.minute / 60.0)
        trade.Col_BarsSinceEntry = int(pos.bars_since_entry)
        trade.Col_PosSize_PctAccount = (
            float((abs(pos.shares) * pos.entry_price) / pos.starting_account * 100.0)
            if pos.starting_account > 0
            else 0.0
        )
        trade.Col_UnrealizedPL_1000 = float(pos.unrealized_pl_1000)
        trade.Col_UnrealizedPL_1030 = float(pos.unrealized_pl_1030)
        trade.Col_UnrealizedPL_1100 = float(pos.unrealized_pl_1100)
        trade.Col_UnrealizedPL_1130 = float(pos.unrealized_pl_1130)
        trade.Col_UnrealizedPL_1200 = float(pos.unrealized_pl_1200)
        trade.Col_UnrealizedPL_1230 = float(pos.unrealized_pl_1230)
        trade.Col_UnrealizedPL_1300 = float(pos.unrealized_pl_1300)
        trade.Col_UnrealizedPL_1330 = float(pos.unrealized_pl_1330)
        trade.Col_UnrealizedPL_1400 = float(pos.unrealized_pl_1400)
        trade.Col_UnrealizedPL_1430 = float(pos.unrealized_pl_1430)
        trade.Col_UnrealizedPL_1500 = float(pos.unrealized_pl_1500)
        trade.Col_UnrealizedPL_1530 = float(pos.unrealized_pl_1530)
        trade.Col_UnrealizedPL_1600 = float(pos.unrealized_pl_1600)
        return net_pnl, trade

