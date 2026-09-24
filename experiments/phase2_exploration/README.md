# Phase 2 exploration scripts (2026-09-24)

Small scripts that were run **before** the evaluation protocol was frozen, to compare the
protocol options and to understand why the static meta-features fail. Their numbers are
reported in Appendix A of `docs/metalearner_methodology.md`.

**Exploratory, not official results.** The official Phase 2 results come from
`train_static.py` (in `results/phase2/`). These scripts use earlier, simpler settings
(e.g. 100 trees, smaller grids, 5 pool brackets) and were written as quick experiments.

| script | question | protocol decision / appendix | runtime |
|---|---|---|---|
| `common.py` | shared helpers (loads `data/metafeatures.csv`, models, metrics) | – | – |
| `e1_splitters.py` | plain vs. stratified grouped splitting | P1 / A.1 | ~1 min |
| `e1b.py` | same, 50 repeats, 6 bins, raw vs. log k | P1 / A.1 | ~5 min |
| `e2_diag.py` | feature–k correlations, errors per k bracket | A.6 | < 1 min |
| `e3_nested.py` | tuned (nested CV) vs. default settings | P4 | ~8 min |
| `e4_ttest.py` | plain vs. corrected t-test on 300 simulated pools | P6 / A.5 | ~1.5 min |
| `e5_schemes.py` | 3/5/10 folds × 10/20 repeats, clipping and rounding | P2, P5 / A.2, A.3 | ~1 min |
| `e6_lc_noise.py` | noise of a learning-curve step (not the curve itself) | P8 / A.4 | ~4 min |
| `e7_why.py` | permutation test, size-only features, image-variant check | A.6 | ~3 min |

Run from the repository root, e.g. `python experiments/phase2_exploration/e4_ttest.py`.
Runtimes were measured on a 2-core machine.
