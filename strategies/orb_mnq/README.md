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
| Size | Contracts so the stop costs about $750 in the evaluation, max 40 MNQ (LucidFlex cap). Half that risk once funded |
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
| **LucidFlex** | 2023-24 | 50% | 49% | 17 | 20% |
| **LucidFlex** | **2025** | **38%** | 56% | 21 | 33% |
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

Funded phase (LucidFlex rules, funded account traded at half the evaluation
risk, 120 sessions): across the full sample 76% of passed accounts collected at
least one payout, averaging 1.3 payouts and $941 to the trader, but 86% of them
were eventually blown. The LucidFlex rule that moves the loss limit to $50,100
after every payout leaves little room, so most funded accounts end after one or
two payouts. In 2025 only 44 passed accounts had 60 sessions left to test, and
25% of those reached a payout, so the 2025 funded numbers are thin and weak.
Trading the funded account at full evaluation risk blew it more often for about
the same payouts; at a quarter of the risk it survived longer but paid less.

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

## Longer history: Sep 2020 to Nov 2025

A second free NQ 1-minute file (back-adjusted continuous contract, bar-start
timestamps, from github.com/hindsight-finance/Silver-Bullet-AM-Session) covers
2020-09-01 to 2025-11-21. No free source for 2019 to Aug 2020 or for 2026 was
reachable. LucidFlex 50K, $750 risk, 40-session evaluation limit:

| Year | 09:30-10:00 PF | Pass | 09:00-09:30 PF | Pass |
|---|---|---|---|---|
| 2020 (Sep-Dec) | 0.92 | 4% | 0.87 | 14% |
| 2021 | 1.34 | 31% | 1.11 | 28% |
| 2022 | 1.26 | 49% | 1.23 | 45% |
| 2023 | 1.20 | 46% | 1.04 | 30% |
| 2024 | 1.62 | 56% | 1.24 | 32% |
| 2025 | 1.13 | 40% | 1.35 | 46% |
| All | 1.28 | 41% | 1.16 | 35% |

2021-2022 were never used to choose settings; the 09:30 version held up there.

## The maths guide applied (`quant.py`, `math_study.py`)

Each stage of "How to Implement Mathematics into Algorithmic Trading" was
added and tested on 2021-09 to 2025-11 (GARCH and the HMM need a year of
history first). Everything fitted uses only earlier data.

| Stage | What was added | 09:30-10:00 result | 09:00-09:30 result |
|---|---|---|---|
| 2 Diagnose | Hurst, Dickey-Fuller / OU half-life, AR(1) on 1m prices in the trading window | Hurst 0.39 = random-walk reference 0.39; DF significant on only 13% of days | Hurst 0.42 vs 0.39; DF significant on 3% of days |
| 3 Gate | GARCH(1,1), refit monthly: trade only if sigma >= 0.8 x long-run | Pass 44% -> 22%, PF 1.30 -> 1.17. Hurts | Pass 38% -> 31%. Hurts |
| 3 Gate | 2-state HMM, refit quarterly, P(regime) >= 0.7 | "Volatile" and "calm" each won in a different half of the sample. No stable edge | Same |
| 4 Signal | Linear-regression slope must agree with the breakout | No change (PF 1.30) | Slightly worse |
| 5-6 Size + guard | Capped Kelly (rolling 100 trades) on the remaining drawdown buffer; block if buffer - risk < 20% of the limit; sit out a day after 3 losses | Never fails, but 41% of evaluations stall when the buffer gets too small for 1 contract; passes take a median 127 sessions | 67% stall |
| 7 Evaluate | Expectancy, p, b, Kelly f*, Sharpe, Sortino | +0.12R a trade, p 47%, b 1.47, f* 11%, Sharpe 1.5, Sortino 2.7 | +0.09R, p 31%, b 2.69, f* 5%, Sharpe 1.1 |
| 8 Validate | t-stat, trades needed, deflated Sharpe (6,200 variants tried), Monte Carlo | t = 3.1 (passes the t >= 3 bar). Deflated Sharpe 0.27: after this many variants, a 1.5 Sharpe is within what luck produces | t = 2.4, deflated Sharpe 0.08: likely luck |

Sizing comparison, 09:30-10:00, with up to 250 sessions to pass:

| Sizing | Pass | Fail | Stalled | Median sessions to pass | Funded: any payout / avg to you / blown (250 sessions) |
|---|---|---|---|---|---|
| Fixed $750 (current) | 45% | 55% | 0% | 19 | 81% / $908 / 96% |
| Fixed $250 | 68% | 12% | 20% | 176 | 65% / $404 / 1% |
| Kelly 1x on buffer + guard | 59% | 0% | 41% | 127 | 62% / $404 / 0% |
| Eval $750, funded on Kelly + guard | 44% | 54% | 3% | 18 | 84% / $489 / 0% |

The guide's Monte Carlo rule (lower risk until bust < 10-15%) points to about
$150 a trade, but a 30-minute MNQ range usually needs more than $150 for one
micro, so it mostly cannot trade.

**Verdict.** The guide's validation stages are the most useful part: they
show the 09:30 edge is real but thin (t 3.1) and does not survive a
multiple-testing correction, and the 09:00 version is probably noise. None of
the gates or filters improved results, so the default strategy is unchanged.
Kelly sizing with the guard trades speed for survival: it almost never blows
an account but most attempts crawl. If the evaluation has no time limit and
no monthly fee, fixed $250 or Kelly 1x passes more often than $750; if time
costs money, $750 passes about as often and 10x faster. In the funded
account, Kelly + guard kept every account alive for a year but paid about half
as much as half-risk fixed sizing in that year.

## 09:00-09:30 range variant

Set `or_start=-30, or_minutes=30` (Pine: "Range starts at" = 09:00) to use the
half hour before the open as the range, with entries from 09:30. Same exit
rules, LucidFlex 50K, $750 risk:

| Range | Year | Win rate | Profit factor | Net | Max drawdown | Pass | Fail | Median days to pass |
|---|---|---|---|---|---|---|---|---|
| 09:30-10:00 | 2023 | 45% | 1.20 | $9,502 | $4,514 | 45% | 53% | 19 |
| 09:30-10:00 | 2024 | 48% | 1.63 | $24,854 | $4,376 | 56% | 44% | 15 |
| 09:30-10:00 | 2025 | 48% | 1.13 | $4,892 | $4,892 | 38% | 56% | 21 |
| 09:00-09:30 | 2023 | 26% | 1.01 | $460 | $11,531 | 29% | 71% | 10 |
| 09:00-09:30 | 2024 | 32% | 1.24 | $18,736 | $16,650 | 33% | 67% | 6 |
| 09:00-09:30 | 2025 | 34% | 1.36 | $24,912 | $5,560 | 46% | 54% | 8 |

The pre-open range is narrower, so the stop is tighter and the position about
twice as large (8 micros on average instead of 3-4). The opening spike at 09:30
stops it out often, which gives a ~30% win rate and much deeper drawdowns. It
was the better range in 2025 and the worse one in 2023-24. Tuning its other
settings on 2023-24 picked a 1R target and 5-minute confirmation that broke even
in 2025, so there is no stable edge to tune. The 09:30-10:00 range stays the
default.

## Prop rules modelled

All presets in `propfirm.py` approximate published 50K plans; `lucidflex_50k`
is the default. Firms change
their rules often, so check your firm's rulebook and edit `PropRules`.

* `lucidflex_50k` (Lucid Trading help center and rule summaries, checked
  2026-11-03): $3,000 target, $2,000 loss limit trailing the end-of-day balance
  and locking at $50,100 once the balance passes $52,100, no daily loss limit
  (it is an optional add-on; set `daily_loss` if you bought it), 50% consistency
  in the evaluation only, 40 micros max. Funded: payout after 5 days of $150+
  with positive cycle profit, request 50% of profit up to $2,000 (minimum $500),
  90% split, loss limit moves to $50,100 after each payout, live review after 5
  payouts. The funded scaling plan (2 to 4 minis by profit) is not modelled; at
  $375-750 risk the strategy trades 2-6 micros, well under it.
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
python -m strategies.orb_mnq.math_study --data nq_1m.parquet --label start   # maths-guide study
python -m strategies.orb_mnq.run_backtest --data nq_1m.csv --preset topstep_50k
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
| `quant.py` | GARCH, HMM, Hurst, Dickey-Fuller, Kelly sizer with risk guard, Monte Carlo, t-stat, deflated Sharpe |
| `math_study.py` | Applies the maths guide to both range versions and prints the comparison |
| `pine/mnq_orb.pine` | TradingView strategy |

Backtests are not live results. Slippage around the open can be worse than one
tick, and a strategy that passed in the past can fail the next evaluation.
