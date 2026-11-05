# Post-freeze analyses (November 2026)

These scripts support the revised manuscript **"Who Said Fifty? Empty Order Books,
Calibration, and Taker Returns in a Regulated Prediction Market."** They read the
same frozen data as everything else in this repository (`logs/`, release
`data-freeze-v1`) and use the same filters, fee rule, and event clusters as
`tools/gate_analysis.py`.

**None of this was pre-specified.** The book-width and event-structure analyses
came out of an audit of the frozen results after the planned analyses were done.
The 50-cent spread split was chosen after looking at the spread distribution. The
paper labels all of it post-freeze and exploratory.

## Reproduce (from the repository root)

```bash
pip install numpy pandas matplotlib
python post_freeze_2026-11/width_event_structure.py logs post_freeze_2026-11/out   # ~5 min
python post_freeze_2026-11/fe_coverage.py           logs post_freeze_2026-11/out
python post_freeze_2026-11/prospective_horizon.py   logs post_freeze_2026-11/out
python post_freeze_2026-11/make_figure1.py post_freeze_2026-11/out/width_event_structure.json post_freeze_2026-11/out/Figure_1
```

Inference: event-clustered percentile bootstrap, numpy PCG64, seed 20260824;
10,000 draws (1,000 for the within-series fixed-effects comparison, 2,000 for the
Figure 1 bucket intervals). These runs draw a different random stream from the
frozen exhibits, so intervals for quantities that also appear in the frozen
exhibits can differ in the second decimal; the paper reports the frozen ones.

## Crosswalk (manuscript exhibit <- script, output key)

| Exhibit | Script | Output |
|---|---|---|
| Table 2 (gap by spread x event size) | `width_event_structure.py` | `bins`, `gap`, `composition` |
| Table 3 Panel A (cutoffs) | `width_event_structure.py` | `cutoffs` |
| Table 3 Panel B (dropping series families) | `width_event_structure.py` | `drop_family` |
| Table 3 Panel C (horizon x width) | `width_event_structure.py` | `horizon` |
| Table 3 note, Section 4.4 (within series x bucket) | `width_event_structure.py`, `fe_coverage.py` | `within_series`; `segment_share_in_mixed_cells` |
| Section 4.3 mid-curve gaps by width x event size | `fe_coverage.py` | `midcurve_by_width_eventsize` |
| Table 5 Panel B, Sections 5.1-5.2 (taker returns by spread; gate in narrow books) | `width_event_structure.py` | `taker`, `gate_narrow50` |
| Section 6 (prospective by width; entry horizon) | `width_event_structure.py`, `prospective_horizon.py` | `prospective`; `median_htc_hours`, `share_ge_7d` |
| Figure 1 | `make_figure1.py` | `reliability`, `composition`, `gap` |

Tables 1, 4, and 5 Panel A, the prospective results, the all-markets horizon
split, and the coherence scans are the frozen exhibits produced by
`tools/gate_analysis.py` and `robustness/verify_and_robustness.py`.

## `exploration/`: how the analysis got here

The exploratory runs that led to the final specification, kept so the path is
visible. They use 2,000 draws and earlier cut points (for example a 21-cent split),
so their numbers differ from the paper's.

- `stress_test_book_width.py`: cutoffs, spread bins, series fixed effects, dropped series, horizon, near-empty books (21-cent split)
- `mechanism_probe.py`: two-market vs three-plus-market events within spread bins
- `width_decomposition.py`: spread groups, event size, taker returns, gate check in narrow books
- `figure_data.py`: first version of the Figure 1 data
- `robustness_50c.py`: first pass at the 50-cent split
