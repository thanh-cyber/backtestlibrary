from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd

from .bt_types import Position, RunResult, Strategy, TradeRecord


@dataclass
class BacktestConfig:
    session_start: time
    session_end: time
    fixed_position_cost: float = 100.0
    margin_requirement: float = 1.0
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


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


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


def _iter_times(start: time, end: time, step_seconds: int) -> list[time]:
    step_minutes = max(1, step_seconds // 60)
    current = _time_to_minutes(start)
    end_m = _time_to_minutes(end)
    out: list[time] = []
    while current <= end_m:
        out.append(time(current // 60, current % 60))
        current += step_minutes
    return out


def get_row_price(row: pd.Series, t: time) -> Optional[float]:
    """Robustly read a bar price from wide time-column row."""
    candidates = [
        f"{t.hour}:{t.minute:02d}",
        f"{t.hour:02d}:{t.minute:02d}",
        f"{t.hour}:{t.minute:02d}:00",
        f"{t.hour:02d}:{t.minute:02d}:00",
        f"{t.hour}:{t.minute:02d}:30",
        f"{t.hour:02d}:{t.minute:02d}:30",
        time(t.hour, t.minute, 0),
        time(t.hour, t.minute, 30),
    ]
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
    return None


class ChronologicalBacktestEngine:
    """Chronological engine with pluggable entry/exit strategy callbacks."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        cleaned_year_data: dict[str, pd.DataFrame],
        strategy: Strategy,
        starting_accounts: list[int],
    ) -> dict[str, dict[int, RunResult]]:
        results: dict[str, dict[int, RunResult]] = {}

        for year, df_year in cleaned_year_data.items():
            results[year] = {}
            if "Date" in df_year.columns:
                df_year = df_year.copy()
                df_year["Date"] = pd.to_datetime(df_year["Date"], errors="coerce").dt.normalize()
                df_year = df_year.dropna(subset=["Date"]).sort_values("Date")

            daily_groups = df_year.groupby("Date")
            for starting_account in starting_accounts:
                run_result = self._run_single_account(daily_groups, starting_account, strategy)
                results[year][starting_account] = run_result
        return results

    def _run_single_account(self, daily_groups, starting_account: int, strategy: Strategy) -> RunResult:
        account_balance = float(starting_account)
        available_balance = float(starting_account)
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        trades_list: list[dict] = []

        for date, df_day in daily_groups:
            timeline = _iter_times(self.config.session_start, self.config.session_end, self.config.timeline_step_seconds)
            entries = strategy.find_entries_for_day(df_day, timeline, get_row_price) or []
            entries.sort(key=lambda x: (x.entry_time.hour, x.entry_time.minute, x.entry_time.second, str(x.ticker)))

            entries_by_time: dict[time, list] = {}
            for e in entries:
                entries_by_time.setdefault(e.entry_time, []).append(e)

            open_positions: list[Position] = []
            for current_time in timeline:
                # 1) Enter new positions in strict chronological order.
                for entry in entries_by_time.get(current_time, []):
                    row = df_day.loc[entry.row_index]
                    shares, entry_value = self._size_position(entry.entry_price, row, account_balance)
                    if shares <= 0 or entry_value <= 0:
                        continue
                    required_margin = entry_value * self.config.margin_requirement
                    if available_balance < required_margin:
                        continue

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
                        )
                    )
                    available_balance -= required_margin

                # 2) Evaluate exits.
                still_open: list[Position] = []
                for pos in open_positions:
                    row = df_day.loc[pos.row_index]
                    signal = strategy.check_exit(pos, row, current_time, get_row_price)

                    # Engine-level time exit safeguard.
                    if signal is None and current_time == self.config.session_end:
                        forced = _to_float(row.get(self.config.exit_price_col))
                        if forced is None:
                            forced = get_row_price(row, current_time)
                        if forced is not None and forced > 0:
                            from .bt_types import ExitSignal

                            signal = ExitSignal(exit_price=forced, reason="Time Exit")

                    if signal is None:
                        still_open.append(pos)
                        continue

                    net_pnl, trade = self._close_trade(
                        pos=pos,
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
                    trades_list.append(asdict(trade))

                    if account_balance <= 0:
                        break

                open_positions = still_open
                if account_balance <= 0:
                    break

            if account_balance <= 0:
                break

        final_balance = account_balance
        total_pnl = final_balance - float(starting_account)
        total_return_pct = (total_pnl / float(starting_account) * 100.0) if starting_account > 0 else 0.0
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        return RunResult(
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            trades=trades_df,
        )

    def _size_position(self, entry_price: float, row: pd.Series, account_balance: float) -> tuple[int, float]:
        shares = int(self.config.fixed_position_cost / entry_price) if entry_price > 0 else 0
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
        return shares, shares * entry_price

    def _close_trade(
        self,
        pos: Position,
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

        borrow_annual = self.config.borrow_annual_high if entry_value > 1_000_000 else self.config.borrow_annual_base
        borrow_daily = borrow_annual / 365.0
        borrow_cost = entry_value * borrow_daily * (hold_minutes / (24.0 * 60.0))
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
        return net_pnl, trade

