"""Apply the "How to Implement Mathematics into Algorithmic Trading" pipeline to both ORB versions.

    python -m strategies.orb_mnq.math_study --data nq_1m.parquet --label start

Prints, for the 09:30-10:00 and 09:00-09:30 ranges under LucidFlex 50K rules:
stage 2 diagnosis, stage 3 regime gates (GARCH, HMM) and the stage 4 slope
filter against the baseline, stage 5-6 capped Kelly on the drawdown buffer
with the risk guard, and the stage 7-8 statistics (expectancy, Kelly f*,
Sharpe, Sortino, t-stat, deflated Sharpe, Monte Carlo bust curve).
The comparison window starts once GARCH and the HMM have a year of history.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

import numpy as np

from . import quant as Q
from .data import build_sessions, load_bars
from .propfirm import PRESETS, pass_rate
from .report import RECOMMENDED
from .strategy import run, stats

TRIALS = 6_200  # parameter variants tried across this project (grids + checks)


def row(label, part, p, rules, make_sizer=None, trades=None, **kw):
    tr = trades if trades is not None else run(part, p)
    st = stats(tr, len(part))
    pr = pass_rate(tr, [s.day for s in part], rules, make_sizer=make_sizer, **kw)
    print(f"  {label:34s} trades={st['trades']:4d} win={st['win_rate']:.0%} PF={st['profit_factor']:.2f} "
          f"maxDD=${st['max_dd']:,.0f} | pass={pr['pass_rate']:.0%} fail={pr['fail_rate']:.0%} "
          f"stalled={pr['timeout_rate']:.0%} days={pr['median_days_to_pass']:.0f} | "
          f"funded payout={pr.get('funded_any_payout', float('nan')):.0%} "
          f"${pr.get('funded_avg_withdrawn', float('nan')):,.0f} blown={pr.get('funded_blown', float('nan')):.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="end", choices=["start", "end"])
    ap.add_argument("--start", default="2021-09-01", help="first day of the comparison window")
    args = ap.parse_args()

    rules = PRESETS["lucidflex_50k"]
    sessions = build_sessions(load_bars(args.data, label=args.label))
    start = date.fromisoformat(args.start)
    win = [s for s in sessions if s.day >= start]
    days = [s.day for s in win]
    print(f"{len(sessions)} sessions {sessions[0].day}..{sessions[-1].day}; comparing {start}..{days[-1]}")

    garch_blocked, _, _ = Q.garch_gate(sessions)
    hmm_vol, _ = Q.hmm_gate(sessions, "volatile")
    hmm_calm, _ = Q.hmm_gate(sessions, "calm")

    versions = {"09:30-10:00": RECOMMENDED,
                "09:00-09:30": replace(RECOMMENDED, or_start=-30, or_minutes=30)}
    for name, base in versions.items():
        print(f"\n=== {name} range ===")
        diag = Q.diagnose(sessions, base.or_start, base.or_start + 120)
        print("Stage 2 diagnose:", {k: round(v, 3) for k, v in diag.items()})

        all_trades = run(sessions, base)
        tr = [t for t in all_trades if t.day >= start]
        ev = Q.evaluate(tr, days)
        r = np.array([t.r for t in tr])
        daily = {d: 0.0 for d in days}
        for t in tr:
            daily[t.day] += t.r
        dr = np.array(list(daily.values()))
        print("Stage 7 evaluate:", {k: round(v, 3) for k, v in ev.items()},
              f"(GBM chance of the {base.target_r}R target before the stop: {Q.gbm_hit_prob(base.target_r):.0%})")
        dsr = Q.deflated_sharpe(dr, TRIALS)
        print(f"Stage 8 validate: t-stat={Q.t_stat(r):.2f}, trades needed={Q.trades_needed(r):.0f}, "
              f"Sharpe={dsr['sharpe_ann']:.2f} vs luck level {dsr['luck_benchmark_ann']:.2f} "
              f"after {TRIALS} variants -> deflated Sharpe {dsr['dsr']:.2f}")
        mc = [Q.monte_carlo_eval(dr, risk, days=250) for risk in (150, 250, 400, 500, 750, 1000)]
        print("Stage 8 Monte Carlo (risk: pass/bust):",
              ", ".join(f"${m['risk']}: {m['p_pass']:.0%}/{m['p_bust']:.0%}" for m in mc))

        print("Stages 3-4 gates and filters ($750 fixed risk, 40-session evaluation limit):")
        row("baseline", win, base, rules, funded_risk_scale=0.5)
        row("GARCH gate (sigma >= 0.8 x long-run)", win, replace(base, blocked=garch_blocked), rules, funded_risk_scale=0.5)
        row("HMM gate: P(volatile) >= 0.7", win, replace(base, blocked=hmm_vol), rules, funded_risk_scale=0.5)
        row("HMM gate: P(calm) >= 0.7", win, replace(base, blocked=hmm_calm), rules, funded_risk_scale=0.5)
        row("linear-regression slope (30 bars)", win, replace(base, slope_len=30), rules, funded_risk_scale=0.5)

        print("Stages 5-6 sizing (250-session evaluation limit):")
        row("fixed $750", win, base, rules, trades=tr, max_days=250, funded_risk_scale=0.5)
        row("fixed $250", win, replace(base, risk_dollars=250.0), rules, max_days=250, funded_risk_scale=0.5)
        for c in (1.0, 2.0):
            cfg = Q.KellyConfig(fraction=c, max_risk=1_000.0, min_risk=250.0)
            row(f"Kelly {c}x on buffer + guard", win, base, rules, trades=tr, max_days=250,
                make_sizer=Q.kelly_sizer_factory(all_trades, cfg))
        kf = Q.kelly_sizer_factory(all_trades, Q.KellyConfig(fraction=1.0, max_risk=750.0, min_risk=250.0))
        row("eval $750, funded Kelly + guard", win, base, rules, trades=tr, funded_horizon=250,
            make_sizer=lambda funded=False: kf(True) if funded else None)
        row("eval $750, funded half risk", win, base, rules, trades=tr, funded_horizon=250, funded_risk_scale=0.5)


if __name__ == "__main__":
    main()
