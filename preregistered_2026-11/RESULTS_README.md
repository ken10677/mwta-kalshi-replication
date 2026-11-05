# Results: political compression test (pre-registered)

- Plan: `PLAN_political_compression_test.md` (commit 8e46d8e, 2026-11-05 14:14 ET).
- Code: `pull_political.py`, `analyze_political.py` (commit b85429c, 14:15 ET), run unchanged.
- Data: `political_entries.jsonl.gz`. It holds 7,477 settled Elections/Politics markets (Aug 2025 to Nov 4, 2026); 6,922 had a two-sided entry book.
- Output: `RESULTS_political_compression_test.json`.

**Pre-registered verdict:** not replicated. The P0 midpoint slope was 0.979, 95% CI [0.947, 1.010], so the CI is not entirely above 1. The trade-price slope was 0.907 [0.844, 0.968].

**Pre-specified comparison P3 (spread ≥ 51¢, 12.6% of markets):** slope 2.414 [1.991, 2.880]; mean gap −14.82 [−20.07, −9.54].

Everything else in later files is labeled exploratory.

## Added after the pre-registered results (not pre-registered)
- `EXPLORATORY_political_eventsize.py` / `.json`: political markets split by spread x event size, and the share of wide-book midpoints between 40 and 60. Exploratory.
- `DESCRIPTIVE_figure2.py`, `Figure_2.json/.png/.pdf`: reliability buckets for the pre-specified subsets P0, P2, P3 (descriptive figure).
- `DESCRIPTIVE_figure2.py`, `Figure_2.*` (updated Nov 5, 2026): now four panels (P0, P1, P2, P3) with the pre-registered slope and mean gap printed on each. P0/P2/P3 bucket intervals are unchanged from the first version.

## Correction: the realized sample window
The plan's window was markets closing 2025-08-01 to 2026-11-04. Kalshi's `GET /markets` endpoint lists only markets settled after the exchange's historical cutoff (`GET /historical/cutoff` reported `market_settled_ts` = 2026-09-06 on Nov 5, 2026); earlier markets are served from `/historical/markets`. `pull_political.py` queried only `/markets`, so the pre-registered sample (above) contains markets that closed 2026-08-15 to 2026-11-04, not the full window. The "Aug 2025 to Nov 4, 2026" description earlier in this file describes the filter, not the realized close dates. `CHECK_unsettled_political_markets.json` lists the 107 political markets that had closed in the window but were not yet settled when checked later on Nov 5.

## Added after the pre-registered results (not pre-registered), continued
- `WINDOW_COMPLETION_pull_political_historical.py`, `WINDOW_COMPLETION_run.py`, `WINDOW_COMPLETION_historical_entries.jsonl.gz`, `WINDOW_COMPLETION_RESULTS.json`: the same series list, filters, and entry rule applied to `/historical/markets`, then `analyze_political.py` run unchanged on (a) the 13,285 added two-sided markets and (b) the full window (20,207 markets; live and historical de-duplicated by ticker, live kept). Verdict in both: not replicated (P0 slope 0.940 [0.907, 0.971] and 0.953 [0.931, 0.976]); P3 slope 1.708 [1.133, 2.232] and 2.027 [1.640, 2.400].
- `SENSITIVITY_excl_nonpolitical_RESULTS.json`, `SENSITIVITY_excluded_series.json`: the pre-registered analysis without 499 shipping-traffic and post-count markets that Kalshi files under Politics. Verdict unchanged.
- `EXPLORATORY_jdm_checks.py` / `.json`: slope, intercept, crossing point, and mean gap by spread x event size in both studies; wide-book calibration at the bid, ask, and midpoint; decomposition of the political mean gap; gap by time from entry to close; entry-spread distributions. Usage: `python EXPLORATORY_jdm_checks.py logs/retro_pairs.jsonl <political entries jsonl> OUT_JSON`.
- `DESCRIPTIVE_figureS1.py`, `Figure_S1.*`: entry-spread distributions in both studies.
