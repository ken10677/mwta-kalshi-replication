# Results: political compression test (pre-registered)

- Plan: `PLAN_political_compression_test.md` (commit 8e46d8e, 2026-11-05 14:14 ET).
- Code: `pull_political.py`, `analyze_political.py` (commit b85429c, 14:15 ET), run unchanged.
- Data: `political_entries.jsonl.gz`. It holds 7,477 settled Elections/Politics markets (Aug 2025 to Nov 4, 2026); 6,922 had a two-sided entry book.
- Output: `RESULTS_political_compression_test.json`.

**Pre-registered verdict:** not replicated. The P0 midpoint slope was 0.979, 95% CI [0.947, 1.010], so the CI is not entirely above 1. The trade-price slope was 0.907 [0.844, 0.968].

**Pre-specified comparison P3 (spread ≥ 51¢, 12.6% of markets):** slope 2.414 [1.991, 2.880]; mean gap −14.82 [−20.07, −9.54].

Everything else in later files is labeled exploratory.
