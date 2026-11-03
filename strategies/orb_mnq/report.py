"""Score the recommended MNQ ORB settings: by year, in/out of sample, per prop preset.

    python -m strategies.orb_mnq.report --data path/to/nq_1m.csv

Also prints a zero-edge baseline: the same trades with their average profit
removed. The gap between the two pass rates is what the strategy's edge adds
over pure position-sizing luck.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import date

import pandas as pd

from .data import build_sessions, load_bars
from .news import NEWS_SETS
from .propfirm import PRESETS, pass_rate
from .strategy import ORBParams, run, stats

# Chosen on 2023-2024 only (run_backtest.py grid + the entry-window check).
RECOMMENDED = ORBParams(
    or_minutes=30,
    confirm_tf=1,
    last_entry=60,  # no new entries after 10:30
    exit_minute=380,  # flat by 15:50
    stop_mode="opposite",
    target_r=2.0,
    breakeven_r=1.0,
    vwap_filter=False,  # with a 30-minute range it never changes a signal
    max_or_frac=1.0,
    max_trades=1,
    news=NEWS_SETS["fomc_cpi"],
    risk_dollars=750.0,
    max_contracts=40,  # LucidFlex 50K cap: 4 minis / 40 micros
)
FUNDED_RISK_SCALE = 0.5  # trade the funded account at half the evaluation risk


def zero_edge(trades):
    m = sum(t.pnl for t in trades) / len(trades)
    return [replace(t, pnl=t.pnl - m, mae=t.mae - max(m, 0.0), mfe=max(0.0, t.mfe - m)) for t in trades]


def score(sessions, p, rules, funded_scale=FUNDED_RISK_SCALE):
    days = [s.day for s in sessions]
    trades = run(sessions, p)
    st = stats(trades, len(days))
    if st["trades"]:
        st.update(pass_rate(trades, days, rules, funded_risk_scale=funded_scale))
        st["zero_edge_pass_rate"] = pass_rate(zero_edge(trades), days, rules)["pass_rate"]
    return st, trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="end", choices=["start", "end"])
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--risk", type=float, default=0)
    ap.add_argument("--funded-scale", type=float, default=FUNDED_RISK_SCALE,
                    help="funded-account risk as a share of evaluation risk")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    p = RECOMMENDED if not args.risk else replace(RECOMMENDED, risk_dollars=args.risk)
    sessions = build_sessions(load_bars(args.data, label=args.label))
    split = date.fromisoformat(args.split)
    parts = {
        "in_sample": [s for s in sessions if s.day < split],
        "out_of_sample": [s for s in sessions if s.day >= split],
        "full": sessions,
    }
    for y in sorted({s.day.year for s in sessions}):
        yr = [s for s in sessions if s.day.year == y]
        if len(yr) > 100:
            parts[str(y)] = yr

    os.makedirs(args.out, exist_ok=True)
    report = {"params": {k: v for k, v in asdict(p).items() if k != "news"}, "results": {}}
    rows = []
    for preset, rules in PRESETS.items():
        for name, part in parts.items():
            st, trades = score(part, p, rules, args.funded_scale)
            report["results"].setdefault(preset, {})[name] = st
            rows.append({"preset": preset, "period": name, **{k: st.get(k) for k in (
                "trades", "win_rate", "profit_factor", "net", "max_dd", "pass_rate", "fail_rate",
                "zero_edge_pass_rate", "median_days_to_pass", "funded_any_payout",
                "funded_avg_payouts", "funded_avg_withdrawn", "funded_blown")}})
        if preset == "lucidflex_50k":
            pd.DataFrame([asdict(t) for t in run(sessions, p)]).to_csv(
                os.path.join(args.out, "trades_recommended.csv"), index=False)

    with open(os.path.join(args.out, "report_recommended.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()
