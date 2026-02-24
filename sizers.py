"""Pluggable position sizers (backtrader-style). Engine applies float/equity/absolute caps after sizer returns."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .bt_types import SizerConfig


def _to_float(value) -> Optional[float]:
    """Coerce to float; return None if missing or non-finite."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return x if abs(x) != float("inf") and x == x else None
    try:
        s = str(value).replace("$", "").replace(",", "").strip()
        x = float(s)
    except (ValueError, TypeError):
        return None
    return x if abs(x) != float("inf") and x == x else None


class FixedSizeSizer:
    """Fixed number of shares per trade. Does not require stop_price."""

    def __init__(self, stake: int = 100):
        if stake < 0:
            stake = 0
        self.stake = int(stake)

    def size(
        self,
        entry_price: float,
        row: pd.Series,
        account_balance: float,
        stop_price: Optional[float],
        side: str,
        config: SizerConfig,
    ) -> tuple[int, float]:
        if entry_price is None or entry_price <= 0:
            return 0, 0.0
        shares = self.stake
        return shares, float(shares * entry_price)


class PercentOfEquitySizer:
    """Size by notional = account_balance * percent. Does not require stop_price."""

    def __init__(self, percent: float = 0.10):
        self.percent = max(0.0, min(1.0, float(percent)))

    def size(
        self,
        entry_price: float,
        row: pd.Series,
        account_balance: float,
        stop_price: Optional[float],
        side: str,
        config: SizerConfig,
    ) -> tuple[int, float]:
        if entry_price is None or entry_price <= 0 or account_balance <= 0 or self.percent <= 0:
            return 0, 0.0
        notional = account_balance * self.percent
        shares = int(notional / entry_price)
        if shares <= 0:
            return 0, 0.0
        return shares, float(shares * entry_price)


class RiskSizer:
    """Risk-based sizing: risk_pct_per_trade or fixed_risk_per_trade; requires stop_price."""

    def size(
        self,
        entry_price: float,
        row: pd.Series,
        account_balance: float,
        stop_price: Optional[float],
        side: str,
        config: SizerConfig,
    ) -> tuple[int, float]:
        if stop_price is None or entry_price <= 0:
            return 0, 0.0
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0, 0.0
        risk_dollars: Optional[float] = None
        if config.risk_pct_per_trade is not None and account_balance > 0:
            risk_dollars = account_balance * config.risk_pct_per_trade
        elif config.fixed_risk_per_trade is not None:
            risk_dollars = config.fixed_risk_per_trade
        if risk_dollars is None or risk_dollars <= 0:
            return 0, 0.0
        shares = int(risk_dollars / risk_per_share)
        if shares <= 0:
            return 0, 0.0
        return shares, float(shares * entry_price)


class KellySizer:
    """Kelly criterion: f* = (p*b - q)/b; use fraction of f* as risk_pct. Requires stop_price."""

    def __init__(
        self,
        win_rate: float,
        payoff_ratio: float,
        fraction: float = 1.0,
        max_risk_cap: Optional[float] = 0.25,
    ):
        self.win_rate = max(0.0, min(1.0, float(win_rate)))
        self.payoff_ratio = max(0.0, float(payoff_ratio))
        self.fraction = max(0.0, float(fraction))
        self.max_risk_cap = max_risk_cap  # cap risk_pct (e.g. 0.25); None = no cap

    def size(
        self,
        entry_price: float,
        row: pd.Series,
        account_balance: float,
        stop_price: Optional[float],
        side: str,
        config: SizerConfig,
    ) -> tuple[int, float]:
        if stop_price is None or entry_price <= 0:
            return 0, 0.0
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0, 0.0
        p, q = self.win_rate, 1.0 - self.win_rate
        b = self.payoff_ratio
        if b <= 0:
            return 0, 0.0
        # f* = (p*b - q) / b
        f_star = (p * b - q) / b
        if f_star <= 0:
            return 0, 0.0
        risk_pct = self.fraction * f_star
        if self.max_risk_cap is not None and risk_pct > self.max_risk_cap:
            risk_pct = self.max_risk_cap
        if account_balance <= 0 or risk_pct <= 0:
            return 0, 0.0
        risk_dollars = account_balance * risk_pct
        shares = int(risk_dollars / risk_per_share)
        if shares <= 0:
            return 0, 0.0
        return shares, float(shares * entry_price)
