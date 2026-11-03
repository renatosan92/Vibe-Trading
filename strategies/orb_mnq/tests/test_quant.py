"""Checks for the maths module: each formula on data with a known answer."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from strategies.orb_mnq import quant as Q
from strategies.orb_mnq.propfirm import Account, PRESETS
from strategies.orb_mnq.strategy import Trade


def test_garch_recovers_parameters():
    rng = np.random.default_rng(1)
    w, a, b = 2e-6, 0.08, 0.9
    n = 3000
    r = np.empty(n)
    v = w / (1 - a - b)
    for t in range(n):
        r[t] = rng.standard_normal() * np.sqrt(v)
        v = w + a * r[t] ** 2 + b * v
    fw, fa, fb = Q.garch11_fit(r)
    assert abs(fa - a) < 0.04 and abs(fb - b) < 0.05
    assert 0.9 < fa + fb < 0.999


def test_hurst_separates_trend_from_random_walk():
    rng = np.random.default_rng(2)
    rw = np.cumsum(rng.standard_normal(5000))
    trend = np.cumsum(np.convolve(rng.standard_normal(5100), np.ones(50) / 50, "valid")[:5000])
    assert Q.hurst(trend) > Q.hurst(rw) + 0.2


def test_dickey_fuller_flags_mean_reversion():
    rng = np.random.default_rng(3)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    stat, half = Q.dickey_fuller(x)
    assert stat < -2.86 and 4 < half < 10  # theta ~ 0.1 -> half-life ~ 6.6


def test_gbm_baseline():
    assert abs(Q.gbm_hit_prob(2.0) - 1 / 3) < 1e-12


def _trade(day, r, risk_per_ct=100.0, contracts=5):
    pnl = r * risk_per_ct * contracts
    return Trade(day=day, side=1, entry_min=0, exit_min=1, entry=0, exit=0, stop=0, target=0,
                 contracts=contracts, pnl=pnl, r=r, mae=min(0.0, pnl), mfe=max(0.0, pnl),
                 reason="x", risk_per_ct=risk_per_ct)


def test_kelly_sizing_and_guard():
    d0 = date(2024, 1, 1)
    hist = [_trade(d0 + timedelta(days=i), 2.0 if i % 2 else -1.0) for i in range(40)]
    kin = Q.rolling_kelly_inputs(hist, window=100, min_trades=30)
    p, b, f = kin[hist[-1].day]
    assert abs(p - 0.5) < 0.05 and abs(b - 2.0) < 1e-9 and abs(f - (p - (1 - p) / b)) < 1e-12
    cfg = Q.KellyConfig(fraction=0.5, max_risk=10_000, loss_streak=0)
    sizer = Q.KellySizer(kin, cfg)
    acct = Account(PRESETS["lucidflex_50k"])
    t = sizer(acct, hist[-1])
    # risk = 0.5 * f* * $2,000 buffer, contracts = floor(risk / $100)
    assert t.contracts == int(0.5 * f * 2000 // 100)
    acct.bal = acct.floor + 450  # buffer nearly gone: the guard vetoes the entry
    assert sizer(acct, hist[-1]) is None


def test_negative_kelly_means_no_trade():
    d0 = date(2024, 1, 1)
    hist = [_trade(d0 + timedelta(days=i), 1.0 if i % 3 == 0 else -1.0) for i in range(40)]
    kin = Q.rolling_kelly_inputs(hist)
    sizer = Q.KellySizer(kin, Q.KellyConfig())
    assert sizer(Account(PRESETS["lucidflex_50k"]), hist[-1]) is None


def test_monte_carlo_sure_win_and_sure_loss():
    up = Q.monte_carlo_eval(np.array([1.0]), risk=500, days=10, paths=100)
    down = Q.monte_carlo_eval(np.array([-1.0]), risk=500, days=10, paths=100)
    assert up["p_pass"] == 1.0 and down["p_bust"] == 1.0


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(4)
    x = rng.normal(0.05, 1.0, 1000)
    assert Q.deflated_sharpe(x, 10)["dsr"] > Q.deflated_sharpe(x, 10_000)["dsr"]
