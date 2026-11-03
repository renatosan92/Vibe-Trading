# MNQ Opening Range Breakout, sized for prop-firm evaluations

A backtestable ORB for Micro Nasdaq futures (MNQ, $2 per point), plus a
simulator of 50K prop-firm evaluation and funded-account rules, and a
TradingView Pine Script port.

## The rules (recommended settings)

| Rule | Setting |
|---|---|
| Opening range | High and low of 09:30 to 10:00 ET (30 minutes) |
| Entry | First 1-minute bar that **closes** above the range high (long) or below the range low (short), entered at the next bar's open. No new entries after 10:30 ET |
| Stop | Other side of the range |
| Target | 2R (twice the distance to the stop) |
| Break-even | Stop moves to entry once price is 1R in profit |
| Time exit | Flat by 15:50 ET |
| Trades per day | 1 |
| News filter | Skip FOMC and CPI days (`news.py`) |
| Size | Contracts so the stop costs about $750, max 50 MNQ |
| Costs | $1.50 round turn per contract, 1 tick slippage on entries, stops and time exits |

The fill model is conservative: when one bar touches both the stop and the
target, the stop is assumed to fill first, and a gap through the stop fills
at the bar's open.

## Results

Data: NQ 1-minute bars, 2022-12-27 to 2025-12-11 (735 sessions). NQ and MNQ
trade at the same price, so NQ bars are an MNQ tape. Parameters were chosen on
2023-2024 only; 2025 was held out and is the honest number.

| Period | Trades | Win rate | Profit factor | Net (MNQ, $750 risk) | Max drawdown |
|---|---|---|---|---|---|
| 2023 | 171 | 45.0% | 1.20 | $9,157 | $4,514 |
| 2024 | 183 | 48.1% | 1.63 | $24,855 | $4,376 |
| **2025 (out of sample)** | 165 | 47.9% | **1.13** | $4,893 | $4,892 |
| Full | 521 | 47.0% | 1.32 | $39,249 | $4,892 |

Evaluation pass rate: a new 50K evaluation is started on every session in the
period and run until it passes or fails; it counts as a non-pass if it has not
passed within 40 sessions (about two monthly fees).

| Rules | Period | Pass | Fail | Median sessions to pass | Same trades with zero edge |
|---|---|---|---|---|---|
| Topstep-style | 2023-24 | 49% | 49% | 18 | 20% |
| Topstep-style | **2025** | **38%** | 56% | 21 | 33% |
| Apex-style | 2023-24 | 50% | 49% | 17 | 23% |
| Apex-style | **2025** | **38%** | 50% | 21 | 36% |

"Zero edge" replays the same trades with their average profit subtracted, so
it shows how often position sizing and luck alone would pass. The strategy
clearly beat that baseline in 2023-24 and only slightly beat it in 2025.

Risk per trade is the biggest lever. At $500 risk the account fails less but
passes slower (25% pass, 29% fail, 46% still running after 40 sessions; with no
time limit about 47% pass overall). At $1,000 risk failures jump to 68% or more.

Funded phase: of the evaluations that passed in 2023-24, about 60% reached at
least one payout within the next 120 sessions under the generic payout model in
`propfirm.py` (5 winning days of $150+, withdraw up to 50% of profit).

### What this means

* The edge is real but thin, and it was weaker in 2025 (profit factor 1.13).
  Expect to pass about 4 in 10 evaluations, not every one.
* Treat each evaluation as a bet with a fee: a 40% pass rate means about 2.5
  attempts per funded account on average. It pays only if the payouts you
  collect beat those fees, which the funded numbers above suggest but do not
  guarantee.
* Two filters did **not** help on MNQ in this sample: the VWAP filter (with a
  30-minute range it never changed a signal) and the 20-day EMA trend filter
  (it lost money in 2025). The range-size filter also did not improve results.
* 5- and 15-minute ranges and the midpoint stop were worse than the 30-minute
  range with the opposite-side stop.

## Prop rules modelled

Both presets in `propfirm.py` approximate published 50K plans. Firms change
their rules often, so check your firm's rulebook and edit `PropRules`.

* `topstep_50k`: $3,000 target, $2,000 maximum loss trailing the end-of-day
  balance and locking at the start balance, $1,000 daily loss limit (stops the
  day, does not fail the account), best day under 50% of profit.
* `apex_50k`: $3,000 target, $2,500 drawdown trailing the intraday peak
  including open profit and locking at start + $100, no daily loss limit, no
  evaluation consistency rule.

## Running it

You need a 1-minute NQ or MNQ CSV with a US/Eastern timestamp column and
open, high, low, close, volume. The data used here is not committed; it is the
public file `data/Dataset_NQ_1min_2022_2025.csv` from
github.com/chrislernunes/nq-opening-range-breakout. Pass `--label start` if
your timestamps mark the start of each bar (the default assumes the end, which
is what that file uses).

```bash
python -m strategies.orb_mnq.report --data nq_1m.csv              # recommended settings
python -m strategies.orb_mnq.report --data nq_1m.csv --risk 500   # other risk sizes
python -m strategies.orb_mnq.run_backtest --data nq_1m.csv        # full grid search (about 15 min)
python -m strategies.orb_mnq.run_backtest --data nq_1m.csv --preset apex_50k
pytest strategies/orb_mnq/tests
```

Outputs go to `strategies/orb_mnq/results/` (gitignored): the grid, the
trade list and a JSON summary.

## TradingView

`pine/mnq_orb.pine` is the same strategy for a 1-minute `CME_MINI:MNQ1!`
chart. Its defaults are the recommended settings. Update the news-date input
for the months you trade. TradingView's own fill model is more optimistic than
the Python one on bars that touch both stop and target.

## Files

| File | What it does |
|---|---|
| `data.py` | Loads 1-minute bars and cuts them into 09:30-16:00 sessions with VWAP and daily-range context |
| `strategy.py` | ORB rules, fill model, sizing and trade statistics |
| `propfirm.py` | Evaluation and funded-account simulator, pass-rate study |
| `news.py` | FOMC, CPI and jobs-report dates for 2023-2025 |
| `run_backtest.py` | Grid search on the in-sample years, scored out of sample |
| `report.py` | Recommended settings by year and preset, with the zero-edge baseline |
| `pine/mnq_orb.pine` | TradingView strategy |

Backtests are not live results. Slippage around the open can be worse than one
tick, and a strategy that passed in the past can fail the next evaluation.
