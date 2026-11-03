"""Opening Range Breakout for MNQ, simulated on 1-minute bars.

Rules (every one is a parameter in ``ORBParams``):

1. Opening range (OR) = high/low of the ``or_minutes`` starting at ``or_start``
   minutes after 09:30 ET (``or_start=-30, or_minutes=30`` = the 09:00-09:30 range).
   Entries are only taken from 09:30 on.
2. Skip the day if it is a news day, or if the OR is too small or too large
   relative to the average daily range of the prior 14 sessions.
3. Breakout = a ``confirm_tf``-minute bar CLOSES above the OR high (long) or
   below the OR low (short), before ``last_entry`` minutes after the open.
   Optional filters: the close must be on the right side of session VWAP, and
   the day's open must be on the right side of the 20-day EMA.
4. Entry = open of the next 1-minute bar, plus slippage.
5. Stop = other side of the range, or its midpoint. Target = ``target_r`` x risk.
   Optional break-even move once price reaches ``breakeven_r`` x risk.
6. Time exit at ``exit_minute`` (minutes after 09:30). Max ``max_trades`` a day;
   a second trade is only taken after a losing first one.

Fill model is deliberately conservative: when one bar touches both stop and
target, the stop is assumed to fill first; stops and market exits pay
slippage, targets fill at the limit price; a gap through the stop fills at the
bar's open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from .data import Session

TICK = 0.25
POINT_VALUE = 2.0  # MNQ: $2 per index point (NQ is $20)


@dataclass(frozen=True)
class ORBParams:
    or_minutes: int = 15
    or_start: int = 0  # minutes after 09:30 the range starts (-30 = 09:00)
    confirm_tf: int = 5  # minutes per confirmation bar (1 = any 1m close)
    last_entry: int = 90  # no new entries after 09:30 + this many minutes (11:00)
    exit_minute: int = 380  # flatten at 09:30 + 380 = 15:50
    stop_mode: str = "opposite"  # "opposite" or "mid"
    target_r: float = 2.0
    breakeven_r: float = 0.0  # 0 disables
    min_or_frac: float = 0.0  # OR width / avg daily range, lower bound
    max_or_frac: float = 1.0  # upper bound
    vwap_filter: bool = True
    trend_filter: bool = False
    news: frozenset = field(default_factory=frozenset)
    max_trades: int = 1
    direction: str = "both"  # "both", "long", "short"
    # sizing and costs
    risk_dollars: float = 400.0
    max_contracts: int = 50  # 50K prop accounts typically cap at 5 NQ / 50 MNQ
    commission_rt: float = 1.50  # $ per contract round turn (MNQ, prop-firm style)
    slippage_ticks: int = 1

    def label(self) -> str:
        d = asdict(self)
        d.pop("news")
        return ", ".join(f"{k}={v}" for k, v in d.items())


@dataclass(frozen=True)
class Trade:
    day: date
    side: int  # +1 long, -1 short
    entry_min: int
    exit_min: int
    entry: float
    exit: float
    stop: float
    target: float
    contracts: int
    pnl: float  # dollars, after costs
    r: float  # pnl / dollars risked
    mae: float  # worst open-trade dollars (<= 0), costs included
    mfe: float  # best open-trade dollars (>= 0), costs included
    reason: str  # "stop", "target", "time", "breakeven"


def _first(mask: np.ndarray) -> int:
    """Index of the first True, or len(mask) if none."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else len(mask)


def _simulate_exit(s: Session, k: int, side: int, entry: float, stop: float, target: float,
                   end: int, p: ORBParams):
    """Walk bars k..end-1 (vectorised) and return exit index, price and reason."""
    slip = p.slippage_ticks * TICK
    hi, lo, op = s.high[k:end], s.low[k:end], s.open[k:end]
    risk = abs(entry - stop)
    fav = (hi - entry) if side > 0 else (entry - lo)

    def hits(stop_px: float, start: int):
        if side > 0:
            st = _first(lo[start:] <= stop_px) + start
            tg = _first(hi[start:] >= target) + start
        else:
            st = _first(hi[start:] >= stop_px) + start
            tg = _first(lo[start:] <= target) + start
        return st, tg

    st, tg = hits(stop, 0)
    reason_stop = "stop"
    if p.breakeven_r > 0:
        be = _first(fav >= p.breakeven_r * risk)
        if be < min(st, tg):
            # from the bar after the trigger bar the stop sits at entry
            st = hits(entry, be + 1)[0] if be + 1 < len(hi) else len(hi)
            stop = entry
            reason_stop = "breakeven"

    n = len(hi)
    if st < n and st <= tg:  # stop first (also when both hit in the same bar)
        gap = op[st] < stop if side > 0 else op[st] > stop
        px = (op[st] if gap else stop) - side * slip
        return k + st, px, reason_stop
    if tg < n:
        return k + tg, target, "target"
    last = end - 1
    return last, s.close[last] - side * slip, "time"


def run_session(s: Session, p: ORBParams) -> list[Trade]:
    if s.day in p.news:
        return []
    or_end = p.or_start + p.or_minutes
    in_or = (s.minute >= p.or_start) & (s.minute < or_end)
    if not in_or.any():
        return []
    or_hi = float(s.high[in_or].max())
    or_lo = float(s.low[in_or].min())
    rth0 = int(np.searchsorted(s.minute, 0, side="left"))
    scan0 = int(np.searchsorted(s.minute, max(or_end, 0), side="left"))
    width = or_hi - or_lo
    if width <= 0:
        return []
    if not np.isnan(s.prior_range_avg):
        frac = width / s.prior_range_avg
        if frac < p.min_or_frac or frac > p.max_or_frac:
            return []

    end = int(np.searchsorted(s.minute, p.exit_minute, side="left"))
    trend = 0
    if p.trend_filter and not np.isnan(s.ema20_prior):
        trend = 1 if s.open[rth0] > s.ema20_prior else -1

    trades: list[Trade] = []
    tf = p.confirm_tf
    i = scan0 + tf - 1  # index of the 1m bar whose close ends the first confirm bar
    slip = p.slippage_ticks * TICK
    while i < end - 1 and len(trades) < p.max_trades:
        if s.minute[i] >= p.last_entry:
            break
        c = s.close[i]
        side = 0
        if c > or_hi:
            side = 1
        elif c < or_lo:
            side = -1
        if side and p.direction != "both" and (side > 0) != (p.direction == "long"):
            side = 0
        if side and p.vwap_filter and (c - s.vwap[i]) * side <= 0:
            side = 0
        if side and trend and side != trend:
            side = 0
        if trades and side == trades[-1].side:  # 2nd trade must be the other way
            side = 0
        if not side:
            i += tf
            continue

        k = i + 1
        entry = s.open[k] + side * slip
        if p.stop_mode == "mid":
            stop = (or_hi + or_lo) / 2.0
        else:
            stop = or_lo if side > 0 else or_hi
        risk_pts = (entry - stop) * side
        if risk_pts <= 0:  # entry gapped past the stop level
            i += tf
            continue
        target = entry + side * p.target_r * risk_pts
        per_ct = risk_pts * POINT_VALUE + p.commission_rt + slip * POINT_VALUE
        contracts = int(min(p.max_contracts, p.risk_dollars // per_ct))
        if contracts < 1:
            i += tf
            continue

        x, px, reason = _simulate_exit(s, k, side, entry, stop, target, end, p)
        cost = p.commission_rt * contracts
        pnl = (px - entry) * side * POINT_VALUE * contracts - cost
        seg_hi, seg_lo = s.high[k:x + 1], s.low[k:x + 1]
        worst = (seg_lo.min() - entry) if side > 0 else (entry - seg_hi.max())
        best = (seg_hi.max() - entry) if side > 0 else (entry - seg_lo.min())
        if reason in ("stop", "breakeven"):
            worst = min(worst, (px - entry) * side)
        if reason == "target":
            best = min(best, (target - entry) * side)
        mae = min(0.0, worst * POINT_VALUE * contracts) - cost
        mfe = max(0.0, best * POINT_VALUE * contracts - cost)
        trades.append(Trade(
            day=s.day, side=side, entry_min=int(s.minute[k]), exit_min=int(s.minute[x]),
            entry=entry, exit=px, stop=stop, target=target, contracts=contracts,
            pnl=pnl, r=pnl / (per_ct * contracts), mae=mae, mfe=mfe, reason=reason,
        ))
        if pnl > 0:
            break  # done for the day after a winner
        # resume scanning from the next confirmation bar after the exit
        i = x + 1
        offset = (i - (scan0 - 1)) % tf
        i += (tf - offset) % tf
    return trades


def run(sessions: list[Session], p: ORBParams) -> list[Trade]:
    out: list[Trade] = []
    for s in sessions:
        out.extend(run_session(s, p))
    return out


def stats(trades: list[Trade], n_days: int) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    # daily equity for drawdown
    by_day: dict[date, float] = {}
    for t in trades:
        by_day[t.day] = by_day.get(t.day, 0.0) + t.pnl
    daily = np.array(list(by_day.values()))
    eq = np.cumsum(daily)
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], eq])) - np.concatenate([[0.0], eq])).max())
    return {
        "trades": len(trades),
        "trade_days": len(by_day),
        "days": n_days,
        "win_rate": float(len(wins) / len(pnl)),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "net": float(pnl.sum()),
        "avg_trade": float(pnl.mean()),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "avg_r": float(np.mean([t.r for t in trades])),
        "max_dd": dd,
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
    }
