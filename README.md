# Miscalibration Without Tradable Alpha — replication package

Data and analysis code for **"Miscalibration Without Tradable Alpha: Displayed
Prices, Actionable Quotes, and Fees in a Regulated Prediction Market"**
(L. K. Swan, 2026; preprint in `paper/`).

Prediction-market prices are widely read as probability forecasts. This paper
estimates calibration and economic exploitability on the same venue (Kalshi)
while walking the price object from the displayed price to the order-book
midpoint to the executable quote to the fee-adjusted taker return.
**A displayed probability is a forecast; an actionable quote is an offer.**

## Reproduce every exhibit

Requires Python 3.10+; `numpy` for the robustness script only.

```bash
# Tables 1-5 and the in-text prospective/retrospective estimates
python tools/gate_analysis.py            # prospective stratum
python tools/gate_analysis.py --retro    # retrospective stratum (57,808 markets)

# Verification ledger (94 deterministic checks against the paper) + Section 3.7 / Table 6
python robustness/verify_and_robustness.py
```

`freeze/` contains the frozen console outputs these scripts produced at the
analysis freeze; `logs/gate_analysis*.json` are the frozen machine-readable
artifacts. The crosswalk: Tables 1-5 <- `tools/gate_analysis.py` <- `logs/*.jsonl`;
Section 3.7 / Table 6 <- `robustness/verify_and_robustness.py` (post-freeze,
clearly labeled as such in the paper).

## Data dictionary

`logs/retro_pairs.jsonl` — one settled market per line (retrospective stratum):
`t` ticker · `event` exchange event ticker (cluster key) · `series` series
ticker · `open`/`close` ISO timestamps · `life_h` market lifetime in hours ·
`r` settlement ("yes"/"no") · `b`/`a` best bid/ask (cents) at the earliest
two-sided daily candle close · `p` candle close price when present (else the
analysis uses the book midpoint; see paper §2.3) · `qts` entry-candle UNIX time.

`logs/market_snapshots.jsonl` — prospective snapshots: `d` date · `t` ticker ·
`p` displayed price · `b`/`a` best bid/ask · `v` volume · `oi` open interest ·
`htc` hours to close.

`logs/resolutions.jsonl` — official settlements for snapshotted markets:
`d` date · `t` ticker · `r` outcome · `c` settlement time.

`logs/retro_noquote.jsonl` — settled markets excluded from the retrospective
analysis set for lack of a qualifying two-sided early book (flow accounting).

## Provenance

This is an analysis-grade export of the frozen release **data-freeze-v1**
(tagged 2026-08-26) from the project's development repository. The
pre-specified exploitability rule (n >= 200, >= 30 event clusters, clustered
95% CI bound > 0) was committed to the version-controlled governance record
before the tradability analyses; the full collection pipeline and governance
history are available to editors and reviewers on request. One deviation from
the freeze, by necessity: `signals/__init__.py` is an empty stub here (the
original imports trading modules excluded from this export);
`signals/fees.py` and `tools/gate_analysis.py` are byte-identical to the freeze.

## Licenses

Code: MIT (see `LICENSE`). Data (`logs/`, `freeze/`): CC BY 4.0 — attribute as
"Swan (2026), Miscalibration Without Tradable Alpha replication package."
Market data were collected from Kalshi's public API endpoints.

## Contact

Lawton K. Swan — lkswan@ufl.edu
