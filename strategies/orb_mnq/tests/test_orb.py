"""Fill-model and prop-rule tests on hand-built sessions (no market data needed)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np

from strategies.orb_mnq.data import Session
from strategies.orb_mnq.propfirm import Account, PropRules
from strategies.orb_mnq.strategy import ORBParams, Trade, run_session


def make_session(path: list[tuple[float, float, float, float]], day=date(2024, 3, 4)) -> Session:
    n = 390
    bars = list(path) + [path[-1][3:] * 4] * (n - len(path))  # flat after the scripted path
    o, h, l, c = (np.array(x, dtype=float) for x in zip(*bars))
    return Session(day=day, minute=np.arange(n), open=o, high=h, low=l, close=c,
                   volume=np.ones(n), vwap=np.full(n, 100.0), prior_close=np.nan,
                   prior_range_avg=np.nan, ema20_prior=np.nan)


def opening_range(lo=95.0, hi=105.0, minutes=5):
    return [(100.0, hi, lo, 100.0)] * minutes


P = ORBParams(or_minutes=5, confirm_tf=1, vwap_filter=False, target_r=2.0,
              commission_rt=0.0, slippage_ticks=0, risk_dollars=1_000.0, max_contracts=10)


def test_long_hits_target():
    # breakout close at 106, enter 106 next bar, stop 95 (11 pts), target 128
    path = opening_range() + [(104, 106.5, 104, 106), (106, 110, 105, 109), (109, 129, 108, 128)]
    (t,) = run_session(make_session(path), P)
    assert t.side == 1 and t.reason == "target"
    assert t.entry == 106 and t.stop == 95 and t.target == 128
    assert t.contracts == min(10, 1_000 // (11 * 2))
    assert t.pnl == (128 - 106) * 2 * t.contracts


def test_same_bar_stop_and_target_counts_as_stop():
    path = opening_range() + [(104, 106.5, 104, 106), (106, 130, 90, 100)]
    (t,) = run_session(make_session(path), P)
    assert t.reason == "stop" and t.exit == 95


def test_gap_through_stop_fills_at_open():
    path = opening_range() + [(104, 106.5, 104, 106), (106, 107, 105, 106), (90, 91, 88, 89)]
    (t,) = run_session(make_session(path), P)
    assert t.reason == "stop" and t.exit == 90


def test_second_trade_only_after_loss_and_opposite_side():
    p = ORBParams(**{**P.__dict__, "max_trades": 2})
    path = opening_range() + [(104, 106.5, 104, 106), (106, 106, 94, 94.5),
                              (94.5, 95, 93, 93.5), (93.5, 94, 70, 71)]
    trades = run_session(make_session(path), p)
    assert [t.side for t in trades] == [1, -1]
    assert trades[0].pnl < 0 and trades[1].reason == "target"


def test_news_day_skipped():
    p = ORBParams(**{**P.__dict__, "news": frozenset({date(2024, 3, 4)})})
    path = opening_range() + [(104, 106.5, 104, 106), (106, 130, 105, 128)]
    assert run_session(make_session(path), p) == []


def test_range_filter():
    s = make_session(opening_range() + [(104, 106.5, 104, 106)] * 3)
    s = Session(**{**s.__dict__, "prior_range_avg": 20.0})  # OR = 10 pts = 0.5 of avg range
    assert run_session(s, ORBParams(**{**P.__dict__, "max_or_frac": 0.4})) == []
    assert run_session(s, ORBParams(**{**P.__dict__, "max_or_frac": 0.6})) != []


def _t(pnl, mae=None, mfe=0.0):
    return Trade(day=date(2024, 1, 1), side=1, entry_min=0, exit_min=1, entry=0, exit=0, stop=0,
                 target=0, contracts=1, pnl=pnl, r=0, mae=min(0.0, pnl) if mae is None else mae,
                 mfe=mfe, reason="x")


def test_eod_trailing_floor_locks_at_start():
    a = Account(PropRules(name="t", daily_loss=0))
    a.trade_day([_t(1_500)])
    assert a.floor == 49_500
    a.trade_day([_t(1_500)])
    assert a.floor == 50_000  # stops trailing at the starting balance
    a.trade_day([_t(1_000)])
    assert a.floor == 50_000


def test_intraday_trailing_uses_open_profit():
    a = Account(PropRules(name="t", max_dd=2_500, trailing="intraday", trail_lock=100, daily_loss=0))
    _, blown = a.trade_day([_t(-400, mae=-400, mfe=1_000)])
    # peak 51,000 -> floor 48,500; balance 49,600 survives
    assert not blown and a.floor == 48_500


def test_daily_loss_limit_caps_the_day():
    a = Account(PropRules(name="t", daily_loss=1_000))
    pnl, blown = a.trade_day([_t(-700), _t(-700)])
    assert not blown and pnl == -1_000


def test_lucidflex_floor_locks_at_50100_and_resets_after_payout():
    from strategies.orb_mnq.propfirm import PRESETS, run_funded
    r = PRESETS["lucidflex_50k"]
    a = Account(r)
    a.trade_day([_t(2_200)])
    assert a.floor == 50_100  # peak 52,200 - 2,000 would be 50,200, but it locks at 50,100
    days = [date(2024, 1, d) for d in range(1, 8)]
    tbd = {d: [replace(_t(400), day=d)] for d in days[:5]}
    res = run_funded(days, tbd, 0, r, horizon=7)
    # 5 days of +$400 = $2,000 profit -> request 50% = $1,000, trader keeps 90%
    assert res.payouts == 1 and res.withdrawn == 900
