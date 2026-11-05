# Analysis plan: does "compression toward 50" in political markets survive tight books?

**Status:** written and committed *before* any political-market data were pulled or
examined (see the git timestamp on this file). The frozen data-freeze-v1 sample contains
only 68 political markets, which were not analyzed, so this test uses newly collected data.

## Motivation

Le (2026, arXiv 2602.19520) pools about 429,000 Kalshi and Polymarket contracts. Its
most robust finding is underconfidence in political markets: prices compressed toward
50%. In our frozen (sports-dominated) sample, the aggregate midpoint gap was concentrated
in books with spreads above 50 cents in events listing three or more markets. We test
whether political "compression toward 50" in Kalshi data survives once those books are
set aside.

## Data (to be collected after this commit)

**Universe.** Every Kalshi market whose series has category "Elections" or "Politics"
in the public series list (`GET /series`).

**Selection.** A market is included if all of the following hold:
- It is a binary market that settled YES or NO.
- Its close time is between 2025-08-01 and 2026-11-04.
- Its lifetime is at least 6 hours.

**Entry rule.** This is identical to the frozen retrospective stratum. Use the earliest
daily candle (`period_interval=1440`) in which the closing best bid is above 0 and the
closing best ask is below 100, with ask > bid. Record that candle's closing bid and ask,
and its closing trade price if one exists.

**Derived variables.**
- midpoint = (bid + ask)/2
- spread = ask − bid
- event = exchange event ticker
- event size = number of included markets in the event

## Estimands

**Calibration slope.** The OLS slope of the outcome Y (0/1) on price/100.
- Slope > 1 means prices are compressed toward 50 (underconfidence).
- We also report the mean gap, mean(Y×100 − price), and the intercept.

**Inference.** Event-clustered percentile bootstrap, 2,000 draws, numpy PCG64 seed
20260824.

## Pre-specified comparisons (midpoint is the primary price)

| Label | Sample |
|---|---|
| P0 | All political markets |
| P1 (primary "clean" subset) | Spread ≤ 50¢ **and** event size 1–2 |
| P2 | Spread ≤ 50¢, any event size |
| P3 | Spread ≥ 51¢ |
| P4 | Spread ≤ 4¢ |

**Secondary.** Repeat P0 and P1 using the closing trade price, for markets with a trade
in the entry candle. This is closer to the trade-based measure in Le (2026).

## Decision rules (fixed now)

- **Replication.** Compression is replicated if the P0 slope's 95% CI lies entirely
  above 1.
- **"Survives" in tight books.** If replicated, compression survives if the P1 slope's
  95% CI lies entirely above 1.
- **"Artifact-consistent."** If replicated, the result is artifact-consistent if the P1
  CI includes 1 **and** the P1 point estimate is closer to 1 than the P0 estimate.
- **Otherwise.** Any other pattern is reported as inconclusive.
- **Thin subset.** If P1 has fewer than 200 markets or fewer than 30 events, P1 is
  reported descriptively only and no decision is drawn.

## Not decided in advance (will be labeled exploratory)

- Any split by election type, office, or horizon
- Any cutoff other than those above
- Any comparison with Polymarket
