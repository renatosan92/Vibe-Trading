"""Prop-firm evaluation and funded-account simulator.

Rules differ by firm and change often, so both presets below are
approximations of the published 50K plans as of late 2025. Check your firm's
current rulebook and edit ``PropRules`` to match before trusting a pass rate.

* topstep_50k: $3,000 target, $2,000 maximum loss that trails the END-OF-DAY
  balance high and stops trailing once it reaches the starting balance, $1,000
  daily loss limit (trading stops for the day, account survives), best day
  must be under 50% of total profit (consistency).
* apex_50k: $3,000 target, $2,500 drawdown that trails the INTRADAY peak
  (including open profit) and stops at start + $100, no daily loss limit,
  no consistency rule in the evaluation.

Intrabar order is unknown on 1-minute bars, so the simulator is conservative:
each trade's worst open loss (MAE) is tested against the drawdown floor first,
then its best open profit (MFE) is allowed to raise an intraday trailing floor,
then the closing balance is tested against that raised floor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from .strategy import Trade


@dataclass(frozen=True)
class PropRules:
    name: str
    start: float = 50_000.0
    target: float = 3_000.0
    max_dd: float = 2_000.0
    trailing: str = "eod"  # "eod" or "intraday"
    trail_lock: float = 0.0  # floor stops trailing at start + trail_lock
    daily_loss: float = 1_000.0  # 0 disables
    daily_loss_fails: bool = False  # True: hitting it fails the account
    consistency: float = 0.5  # best day must be < this share of profit (0 disables)
    min_days: int = 2
    # funded-account payout model (generic, Topstep-like)
    payout_win_days: int = 5  # winning days needed between payouts
    payout_win_threshold: float = 150.0  # a "winning day" nets at least this
    payout_frac: float = 0.5  # share of profit you may withdraw
    payout_cap: float = 5_000.0
    payout_min: float = 250.0  # smaller requests are not worth making
    trader_split: float = 0.9  # trader keeps 90% of each payout


PRESETS = {
    "topstep_50k": PropRules(name="topstep_50k"),
    "apex_50k": PropRules(
        name="apex_50k", max_dd=2_500.0, trailing="intraday", trail_lock=100.0,
        daily_loss=0.0, consistency=0.0, min_days=1,
        payout_win_days=8, payout_win_threshold=50.0, payout_frac=0.5,
        payout_cap=2_000.0, payout_min=500.0, trader_split=1.0,
    ),
}


def trades_by_day(trades: list[Trade]) -> dict[date, list[Trade]]:
    out: dict[date, list[Trade]] = defaultdict(list)
    for t in trades:
        out[t.day].append(t)
    return out


class Account:
    """Balance, trailing floor and daily-loss handling shared by both phases."""

    def __init__(self, r: PropRules):
        self.r = r
        self.bal = r.start
        self.peak = r.start
        self.floor = r.start - r.max_dd
        self.lock = r.start + r.trail_lock

    def _raise_floor(self, level: float) -> None:
        if level > self.peak:
            self.peak = level
            self.floor = min(self.lock, max(self.floor, self.peak - self.r.max_dd))

    def trade_day(self, day_trades: list[Trade]) -> tuple[float, bool]:
        """Apply one day. Returns (day pnl, blown)."""
        r = self.r
        day_start = self.bal
        for t in day_trades:
            dll_left = (self.bal - day_start) + r.daily_loss if r.daily_loss else np.inf
            worst = t.mae
            # daily loss limit liquidates the position at the limit
            if worst <= -dll_left:
                self.bal = day_start - r.daily_loss
                if self.bal <= self.floor:
                    return self.bal - day_start, True
                if r.daily_loss_fails:
                    return self.bal - day_start, True
                break
            if self.bal + worst <= self.floor:
                self.bal = self.floor
                return self.bal - day_start, True
            if r.trailing == "intraday":
                self._raise_floor(self.bal + t.mfe)
            self.bal += t.pnl
            if self.bal <= self.floor:
                self.bal = self.floor
                return self.bal - day_start, True
            if r.trailing == "intraday":
                self._raise_floor(self.bal)
        if r.trailing == "eod":
            self._raise_floor(self.bal)
        return self.bal - day_start, False


@dataclass
class EvalResult:
    start_day: date
    outcome: str  # "pass", "fail", "timeout" (ran out of max_days), "open" (data ended)
    days: int  # sessions elapsed (calendar of trading sessions, not only trade days)
    end_day: date


def run_evaluation(days: list[date], tbd: dict[date, list[Trade]], start_idx: int,
                   r: PropRules, max_days: int = 0) -> tuple[EvalResult, int]:
    acct = Account(r)
    best_day = 0.0
    active = 0
    for j in range(start_idx, len(days)):
        d = days[j]
        if max_days and j - start_idx >= max_days:
            return EvalResult(days[start_idx], "timeout", max_days, days[j - 1]), j - 1
        dt = tbd.get(d, [])
        if dt:
            active += 1
        pnl, blown = acct.trade_day(dt)
        best_day = max(best_day, pnl)
        if blown:
            return EvalResult(days[start_idx], "fail", j - start_idx + 1, d), j
        profit = acct.bal - r.start
        consistent = r.consistency == 0 or profit <= 0 or best_day < r.consistency * profit
        if profit >= r.target and active >= r.min_days and consistent:
            return EvalResult(days[start_idx], "pass", j - start_idx + 1, d), j
    return EvalResult(days[start_idx], "open", len(days) - start_idx, days[-1]), len(days) - 1


@dataclass
class FundedResult:
    payouts: int
    withdrawn: float  # trader's share, dollars
    blown: bool
    days: int


def run_funded(days: list[date], tbd: dict[date, list[Trade]], start_idx: int,
               r: PropRules, horizon: int = 120) -> FundedResult:
    """Trade a funded account for up to ``horizon`` sessions, taking payouts."""
    funded = replace(r, trail_lock=0.0) if r.trailing == "eod" else r
    acct = Account(funded)
    win_days = 0
    payouts, withdrawn = 0, 0.0
    end = min(len(days), start_idx + horizon)
    for j in range(start_idx, end):
        pnl, blown = acct.trade_day(tbd.get(days[j], []))
        if blown:
            return FundedResult(payouts, withdrawn, True, j - start_idx + 1)
        if pnl >= r.payout_win_threshold:
            win_days += 1
        profit = acct.bal - r.start
        if win_days >= r.payout_win_days and profit > 0:
            amount = min(r.payout_cap, r.payout_frac * profit)
            if amount >= r.payout_min:
                acct.bal -= amount
                withdrawn += amount * r.trader_split
                payouts += 1
                win_days = 0
                # after the first payout the floor locks at the starting balance
                acct.floor = max(acct.floor, r.start - r.max_dd)
    return FundedResult(payouts, withdrawn, False, end - start_idx)


def pass_rate(trades: list[Trade], days: list[date], r: PropRules, step: int = 1,
              funded_horizon: int = 120, max_days: int = 40) -> dict:
    """Start a fresh evaluation on every ``step``-th session and report outcomes.

    ``max_days`` (default 40 sessions, about two months of evaluation fees)
    counts a slow evaluation as a non-pass; 0 lets it run until the data ends.
    """
    tbd = trades_by_day(trades)
    evals: list[EvalResult] = []
    funded: list[FundedResult] = []
    for i in range(0, len(days), step):
        res, end_j = run_evaluation(days, tbd, i, r, max_days)
        evals.append(res)
        if res.outcome == "pass" and end_j + 1 + funded_horizon <= len(days):
            funded.append(run_funded(days, tbd, end_j + 1, r, funded_horizon))
    closed = [e for e in evals if e.outcome != "open"]
    passed = [e for e in closed if e.outcome == "pass"]
    failed = [e for e in closed if e.outcome == "fail"]
    timed_out = [e for e in closed if e.outcome == "timeout"]
    out = {
        "rules": r.name,
        "attempts": len(closed),
        "pass_rate": len(passed) / len(closed) if closed else float("nan"),
        "fail_rate": len(failed) / len(closed) if closed else float("nan"),
        "timeout_rate": len(timed_out) / len(closed) if closed else float("nan"),
        "max_eval_days": max_days,
        "median_days_to_pass": float(np.median([e.days for e in passed])) if passed else float("nan"),
        "median_days_to_fail": float(np.median([e.days for e in failed])) if failed else float("nan"),
        "open_at_end": len(evals) - len(closed),
    }
    if funded:
        out.update({
            "funded_samples": len(funded),
            "funded_any_payout": float(np.mean([f.payouts > 0 for f in funded])),
            "funded_avg_payouts": float(np.mean([f.payouts for f in funded])),
            "funded_avg_withdrawn": float(np.mean([f.withdrawn for f in funded])),
            "funded_blown": float(np.mean([f.blown for f in funded])),
            "funded_horizon_days": funded_horizon,
        })
    return out
