"""The maths from "How to Implement Mathematics into Algorithmic Trading", applied to the ORB.

Each piece answers one question, in the guide's pipeline order:

    2 Diagnose  trend or revert?        hurst(), dickey_fuller(), ar1(), gbm_hit_prob()
    3 Gate      allowed to trade today? garch_gate(), hmm_gate()
    5 Size      how many contracts?     KellySizer (capped Kelly on the drawdown buffer)
    6 Guard     inside prop rules?      KellySizer (buffer floor + loss-streak pause)
    7 Evaluate  is the edge real?       evaluate()
    8 Validate  luck or skill?          t_stat(), deflated_sharpe(), monte_carlo_eval()

Everything that is fitted (GARCH, HMM, Kelly p and b) only ever uses data from
before the day it is applied to, so nothing here peeks at the future.
Implied volatility (Black-76) is not used: there is no NQ options history here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .data import Session
from .strategy import Trade

# ── Stage 1: data ──────────────────────────────────────────────────────────


def daily_log_returns(sessions: list[Session]) -> tuple[list[date], np.ndarray]:
    """RTH close-to-close log returns; the first session gets 0."""
    closes = np.array([s.close[s.minute >= 0][-1] if (s.minute < 0).any() else s.close[-1]
                       for s in sessions])
    r = np.zeros(len(closes))
    r[1:] = np.log(closes[1:] / closes[:-1])
    return [s.day for s in sessions], r


# ── Stage 2: diagnose ──────────────────────────────────────────────────────


def hurst(x: np.ndarray, lags=range(2, 64)) -> float:
    """Std(X[t+k] - X[t]) ~ k^H on a log-price path."""
    lags = np.array(list(lags))
    tau = np.array([np.std(x[k:] - x[:-k]) for k in lags])
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def dickey_fuller(x: np.ndarray) -> tuple[float, float]:
    """Regress dX on X (with constant). Returns (DF t-stat, OU half-life in bars)."""
    dx, xl = np.diff(x), x[:-1]
    A = np.column_stack([np.ones_like(xl), xl])
    coef, res, *_ = np.linalg.lstsq(A, dx, rcond=None)
    resid = dx - A @ coef
    s2 = resid @ resid / (len(dx) - 2)
    se = math.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
    theta = -coef[1]
    half = math.log(2) / theta if theta > 0 else float("inf")
    return float(coef[1] / se), half


def ar1(r: np.ndarray) -> float:
    r = r - r.mean()
    return float((r[1:] @ r[:-1]) / (r[:-1] @ r[:-1]))


def gbm_hit_prob(target_r: float) -> float:
    """Driftless chance a +target_r R target is hit before a -1R stop."""
    return 1.0 / (1.0 + target_r)


def diagnose(sessions: list[Session], start_min: int = 0, end_min: int = 120) -> dict:
    """Per-day Hurst, DF and AR(1) on 1m log prices in the trading window, averaged."""
    hs, dfs, halves, a1 = [], [], [], []
    for s in sessions:
        m = (s.minute >= start_min) & (s.minute < end_min)
        lp = np.log(s.close[m])
        if len(lp) < 80:
            continue
        hs.append(hurst(lp, range(2, 30)))
        t, h = dickey_fuller(lp)
        dfs.append(t)
        halves.append(h)
        a1.append(ar1(np.diff(lp)))
    rng = np.random.default_rng(0)
    rw = [hurst(np.cumsum(rng.standard_normal(end_min - start_min)), range(2, 30)) for _ in range(300)]
    return {
        "hurst_mean": float(np.mean(hs)),
        "hurst_random_walk_ref": float(np.mean(rw)),
        "df_mean": float(np.mean(dfs)),
        "df_share_below_5pct": float(np.mean(np.array(dfs) < -2.86)),
        "ou_half_life_median_min": float(np.median(halves)),
        "ar1_1m_mean": float(np.mean(a1)),
        "days": len(hs),
    }


# ── Stage 3: regime gates ──────────────────────────────────────────────────


def garch11_fit(r: np.ndarray) -> tuple[float, float, float]:
    """Zero-mean GARCH(1,1) by Gaussian maximum likelihood. Returns (omega, alpha, beta)."""
    r = np.asarray(r, float)
    v0 = r.var()

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        v = np.empty_like(r)
        v[0] = v0
        for t in range(1, len(r)):
            v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
        return 0.5 * np.sum(np.log(v) + r ** 2 / v)

    best = None
    for a0, b0 in ((0.05, 0.9), (0.1, 0.85), (0.15, 0.8)):
        res = minimize(nll, [v0 * (1 - a0 - b0), a0, b0], method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(float(x) for x in best.x)


def garch_sigma(r: np.ndarray, w: float, a: float, b: float) -> np.ndarray:
    """sigma[t] = forecast for day t made with returns up to t-1 (the live recursion)."""
    v = np.empty(len(r))
    v[0] = w / (1 - a - b)
    for t in range(1, len(r)):
        v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
    return np.sqrt(v)


def garch_gate(sessions: list[Session], k: float = 0.8, refit_every: int = 21,
               min_fit: int = 250) -> tuple[frozenset, np.ndarray, np.ndarray]:
    """Breakout gate: trade only if sigma_t >= k * sigma_LR.

    Refits every ``refit_every`` sessions on all earlier sessions (no look-ahead),
    the offline-weekly / live-recursion split from the guide. Days before the
    first fit are left open. Returns (blocked days, sigma, sigma_LR).
    """
    days, r = daily_log_returns(sessions)
    sig = np.full(len(r), np.nan)
    lr = np.full(len(r), np.nan)
    params = None
    for t in range(len(r)):
        if t >= min_fit and (params is None or (t - min_fit) % refit_every == 0):
            params = garch11_fit(r[1:t])
        if params is not None:
            w, a, b = params
            # run the recursion over history to get today's forecast
            v = w / (1 - a - b)
            for x in r[max(1, t - 500):t]:
                v = w + a * x * x + b * v
            sig[t] = math.sqrt(v)
            lr[t] = math.sqrt(w / (1 - a - b))
    blocked = frozenset(d for d, s, l in zip(days, sig, lr) if not np.isnan(s) and s < k * l)
    return blocked, sig, lr


def _hmm_em(x: np.ndarray, iters: int = 100):
    """2-state Gaussian HMM by Baum-Welch. Returns (pi, A, mu, sd)."""
    mu = np.array([0.0, 0.0])
    sd = np.array([x.std() * 0.6, x.std() * 1.6])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    n = len(x)
    for _ in range(iters):
        B = norm.pdf(x[:, None], mu, sd) + 1e-300
        al = np.empty((n, 2))
        c = np.empty(n)
        al[0] = pi * B[0]
        c[0] = al[0].sum()
        al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]
            c[t] = al[t].sum()
            al[t] /= c[t]
        be = np.empty((n, 2))
        be[-1] = 1
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        g /= g.sum(1, keepdims=True)
        xi = np.zeros((2, 2))
        for t in range(n - 1):
            m = A * np.outer(al[t], B[t + 1] * be[t + 1]) / c[t + 1]
            xi += m / m.sum()
        pi = g[0]
        A = xi / xi.sum(1, keepdims=True)
        mu = (g * x[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0))
    order = np.argsort(sd)  # state 0 = calm, 1 = volatile
    return pi[order], A[np.ix_(order, order)], mu[order], sd[order]


def hmm_gate(sessions: list[Session], want: str = "volatile", p_min: float = 0.7,
             refit_every: int = 63, min_fit: int = 250) -> tuple[frozenset, np.ndarray]:
    """Trade only when the forward-filtered P(want regime) for today is >= p_min.

    The filter uses returns up to yesterday (prediction step), refit quarterly.
    """
    days, r = daily_log_returns(sessions)
    p_vol = np.full(len(r), np.nan)
    model = None
    alpha = None
    for t in range(len(r)):
        if t >= min_fit and (model is None or (t - min_fit) % refit_every == 0):
            model = _hmm_em(r[1:t])
            pi, A, mu, sd = model
            alpha = pi.copy()
            for x in r[1:t]:  # re-run the filter up to yesterday with the new fit
                alpha = (alpha @ A) * norm.pdf(x, mu, sd)
                alpha /= alpha.sum()
        if model is not None:
            pi, A, mu, sd = model
            pred = alpha @ A  # P(regime today | returns to yesterday)
            p_vol[t] = pred[1]
            alpha = pred * norm.pdf(r[t], mu, sd)  # update after today's close
            alpha /= alpha.sum()
    pw = p_vol if want == "volatile" else 1 - p_vol
    blocked = frozenset(d for d, p in zip(days, pw) if not np.isnan(p) and p < p_min)
    return blocked, p_vol


# ── Stages 5-6: capped Kelly on the drawdown buffer + risk guard ────────────


def rolling_kelly_inputs(trades: list[Trade], window: int = 100, min_trades: int = 30):
    """For each trade day: (p, b, f*) from the ``window`` trades closed before that day."""
    by_day: dict[date, tuple[float, float, float]] = {}
    rs = []
    days = sorted({t.day for t in trades})
    tr_by_day: dict[date, list[Trade]] = {}
    for t in trades:
        tr_by_day.setdefault(t.day, []).append(t)
    for d in days:
        hist = np.array(rs[-window:])
        if len(hist) >= min_trades:
            w, l = hist[hist > 0], hist[hist <= 0]
            p = len(w) / len(hist)
            b = w.mean() / -l.mean() if len(w) and l.mean() < 0 else 0.0
            f = p - (1 - p) / b if b > 0 else -1.0
            by_day[d] = (p, b, f)
        rs.extend(t.r for t in tr_by_day[d])
    return by_day


@dataclass
class KellyConfig:
    fraction: float = 0.5  # c in f_f = c * f*
    max_risk: float = 1_000.0  # hard cap per trade ($)
    min_risk: float = 0.0  # floor (0 = let Kelly go as small as it wants)
    guard_frac: float = 0.2  # block if remaining buffer - risk < guard_frac * max_dd
    loss_streak: int = 3  # after this many losses in a row ...
    pause_days: int = 1  # ... sit out this many trade days
    max_contracts: int = 40
    window: int = 100
    min_trades: int = 30
    fallback_risk: float = 0.0  # risk before enough history (0 = do not trade)


class KellySizer:
    """Stage 5 + 6 for one account: size = c * f* * remaining buffer, then the guard vetoes."""

    def __init__(self, kelly_by_day: dict, cfg: KellyConfig):
        self.k = kelly_by_day
        self.cfg = cfg
        self.streak = 0
        self.pause = 0
        self.last: Trade | None = None
        self.last_day: date | None = None

    def __call__(self, acct, t: Trade) -> Trade | None:
        c = self.cfg
        if self.last is not None:
            self.streak = self.streak + 1 if self.last.pnl <= 0 else 0
            if c.loss_streak and self.streak >= c.loss_streak:
                self.pause, self.streak = c.pause_days, 0
            self.last = None
        if self.pause and t.day != self.last_day:
            self.pause -= 1
            self.last_day = t.day
            return None
        self.last_day = t.day
        remaining = acct.bal - acct.floor
        kin = self.k.get(t.day)
        if kin is None:
            risk = c.fallback_risk
        else:
            fstar = kin[2]
            if fstar <= 0:
                return None  # no edge, no trade
            risk = c.fraction * fstar * remaining
        risk = min(c.max_risk, max(c.min_risk, risk))
        if risk <= 0 or remaining - risk < c.guard_frac * acct.r.max_dd:
            return None
        n = int(min(c.max_contracts, risk // t.risk_per_ct))
        if n < 1:
            return None
        s = n / t.contracts
        out = replace(t, contracts=n, pnl=t.pnl * s, mae=t.mae * s, mfe=t.mfe * s)
        self.last = out
        return out


def kelly_sizer_factory(trades: list[Trade], cfg: KellyConfig, funded_cfg: KellyConfig | None = None):
    kin = rolling_kelly_inputs(trades, cfg.window, cfg.min_trades)

    def make(funded: bool = False):
        return KellySizer(kin, funded_cfg if (funded and funded_cfg) else cfg)

    return make


# ── Stage 7: evaluate ──────────────────────────────────────────────────────


def evaluate(trades: list[Trade], days: list[date]) -> dict:
    r = np.array([t.r for t in trades])
    w, l = r[r > 0], r[r <= 0]
    p = len(w) / len(r)
    W, L = (w.mean() if len(w) else 0.0), (-l.mean() if len(l) else 0.0)
    pnl_by_day = {d: 0.0 for d in days}
    for t in trades:
        pnl_by_day[t.day] += t.r  # daily R, flat days included
    dr = np.array(list(pnl_by_day.values()))
    downside = np.sqrt(np.mean(np.minimum(dr, 0) ** 2))
    hits = np.mean([t.reason == "target" for t in trades])
    return {
        "expectancy_R": float(r.mean()),
        "p": float(p),
        "b": float(W / L) if L else float("inf"),
        "breakeven_p": float(L / (W + L)) if W + L else float("nan"),
        "kelly_f_star": float(p - (1 - p) / (W / L)) if L and W else float("nan"),
        "sharpe_daily_ann": float(dr.mean() / dr.std(ddof=1) * math.sqrt(252)),
        "sortino_ann": float(dr.mean() / downside * math.sqrt(252)) if downside else float("nan"),
        "target_hit_rate": float(hits),
    }


# ── Stage 8: validate ──────────────────────────────────────────────────────


def t_stat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def trades_needed(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float((1.96 * x.std(ddof=1) / x.mean()) ** 2) if x.mean() else float("inf")


def deflated_sharpe(daily: np.ndarray, n_trials: int) -> dict:
    """Bailey & Lopez de Prado (2014). Probability the true Sharpe beats the best-of-N luck level."""
    x = np.asarray(daily, float)
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3 = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    g4 = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
    gamma = 0.5772156649
    sr_var = 1.0 / (T - 1)  # null variance of the per-period Sharpe estimate
    sr0 = math.sqrt(sr_var) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                               + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr)
    return {"sharpe_ann": sr * math.sqrt(252), "luck_benchmark_ann": sr0 * math.sqrt(252),
            "dsr": float(norm.cdf(z)), "trials": n_trials}


def monte_carlo_eval(daily_R: np.ndarray, risk: float, max_dd: float = 2_000.0,
                     target: float = 3_000.0, lock: float = 100.0, days: int = 80,
                     paths: int = 10_000, seed: int = 0) -> dict:
    """Resample daily R results; EOD-trailing drawdown that locks at start + lock."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(daily_R, size=(paths, days)) * risk
    eq = np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(np.maximum(eq, 0), axis=1)
    floor = np.minimum(peak - max_dd, lock)
    bust_mask = eq <= floor
    pass_mask = eq >= target
    first_bust = np.where(bust_mask.any(1), bust_mask.argmax(1), days)
    first_pass = np.where(pass_mask.any(1), pass_mask.argmax(1), days)
    return {"risk": risk, "p_pass": float(np.mean(first_pass < first_bust)),
            "p_bust": float(np.mean(first_bust < first_pass))}
