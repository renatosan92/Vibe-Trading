"""Backtest the MNQ ORB, tune it on an in-sample window, and score it out of sample.

    python -m strategies.orb_mnq.run_backtest --data path/to/nq_1m.csv

Writes ``results/summary.json``, ``results/grid_in_sample.csv`` and
``results/trades_<preset>.csv`` next to this file (or ``--out``).

Selection is done ONLY on the in-sample years: the score is the evaluation
pass rate under the chosen prop rules, with a minimum trade count and a
positive profit factor. The out-of-sample year is never used to pick
parameters, so its numbers are the honest ones.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, replace
from datetime import date

import pandas as pd

from .data import build_sessions, load_bars
from .news import NEWS_SETS
from .propfirm import PRESETS, pass_rate
from .strategy import ORBParams, run, stats

GRID = {
    "or_minutes": [5, 15, 30],
    "confirm_tf": [1, 5],
    "stop_mode": ["opposite", "mid"],
    "target_r": [1.0, 1.5, 2.0, 3.0],
    "breakeven_r": [0.0, 1.0],
    "vwap_filter": [False, True],
    "max_or_frac": [0.25, 0.4, 1.0],
    "max_trades": [1, 2],
    "last_entry": [60, 90],
    "news": ["none", "fomc_cpi"],
}
FIXED = {"exit_minute": 380}


def grid_params(risk_dollars: float):
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        kw = dict(zip(keys, combo))
        if kw["breakeven_r"] and kw["breakeven_r"] >= kw["target_r"]:
            continue
        news_key = kw.pop("news")
        yield news_key, ORBParams(news=NEWS_SETS[news_key], risk_dollars=risk_dollars, **FIXED, **kw)


def evaluate(sessions, p: ORBParams, rules) -> dict:
    trades = run(sessions, p)
    days = [s.day for s in sessions]
    out = stats(trades, len(days))
    if out["trades"]:
        out.update(pass_rate(trades, days, rules))
    return out, trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="end", choices=["start", "end"])
    ap.add_argument("--split", default="2025-01-01", help="first out-of-sample day")
    ap.add_argument("--preset", default="topstep_50k", choices=sorted(PRESETS))
    ap.add_argument("--risk", type=float, default=0, help="$ risk per trade (0 = search 300/500/750)")
    ap.add_argument("--min-trades", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    rules = PRESETS[args.preset]
    sessions = build_sessions(load_bars(args.data, label=args.label))
    split = date.fromisoformat(args.split)
    ins = [s for s in sessions if s.day < split]
    oos = [s for s in sessions if s.day >= split]
    print(f"sessions: {len(sessions)}  in-sample {ins[0].day}..{ins[-1].day} ({len(ins)})"
          f"  out-of-sample {oos[0].day}..{oos[-1].day} ({len(oos)})")

    risks = [args.risk] if args.risk else [300.0, 500.0, 750.0]
    rows = []
    for risk in risks:
        for news_key, p in grid_params(risk):
            st, _ = evaluate(ins, p, rules)
            if not st.get("trades"):
                continue
            row = {**{k: v for k, v in asdict(p).items() if k != "news"}, "news": news_key, **st}
            rows.append(row)
    grid = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    grid.to_csv(os.path.join(args.out, f"grid_in_sample_{args.preset}.csv"), index=False)

    ok = grid[(grid.trades >= args.min_trades) & (grid.profit_factor > 1.1)]
    ok = ok.sort_values(["pass_rate", "profit_factor"], ascending=False)
    print(f"\n{len(grid)} combos tried, {len(ok)} meet trades>={args.min_trades} and PF>1.1")
    cols = ["or_minutes", "confirm_tf", "stop_mode", "target_r", "breakeven_r", "vwap_filter",
            "max_or_frac", "max_trades", "news", "risk_dollars", "trades", "win_rate",
            "profit_factor", "net", "max_dd", "pass_rate", "fail_rate", "median_days_to_pass",
            "funded_any_payout"]
    print(ok[cols].head(15).to_string(index=False))

    summary = {"preset": asdict(rules), "split": args.split, "top": []}
    for _, row in ok.head(5).iterrows():
        kw = {k: row[k] for k in ORBParams.__dataclass_fields__ if k in row and k != "news"}
        kw = {k: (v.item() if hasattr(v, "item") else v) for k, v in kw.items()}
        p = ORBParams(news=NEWS_SETS[row["news"]], **kw)
        res = {"params": {**kw, "news": row["news"]}}
        for name, part in [("in_sample", ins), ("out_of_sample", oos), ("full", sessions)]:
            st, trades = evaluate(part, p, rules)
            res[name] = st
        # every preset on the full sample, for comparison
        res["by_preset_full"] = {k: evaluate(sessions, p, v)[0] for k, v in PRESETS.items()}
        summary["top"].append(res)

    best = summary["top"][0]
    bp = ORBParams(news=NEWS_SETS[best["params"]["news"]],
                   **{k: v for k, v in best["params"].items() if k != "news"})
    trades = run(sessions, bp)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(
        os.path.join(args.out, f"trades_{args.preset}.csv"), index=False)
    with open(os.path.join(args.out, f"summary_{args.preset}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nBest in-sample params, scored out of sample:")
    for name in ("in_sample", "out_of_sample", "full"):
        s = best[name]
        print(f"  {name:14s} trades={s['trades']:4d} win={s['win_rate']:.1%} PF={s['profit_factor']:.2f}"
              f" net=${s['net']:,.0f} maxDD=${s['max_dd']:,.0f} pass={s.get('pass_rate', float('nan')):.1%}")


if __name__ == "__main__":
    main()
