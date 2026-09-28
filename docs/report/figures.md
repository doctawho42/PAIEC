# Figures for the technical report

These are the figures `docs/report/draft.md` needs, each with the data behind it. None has been drawn yet. Every figure should come from a committed file (`results/*.json`, a table in `docs/findings.md`, or a module constant), through one plotting script. The proposal is `tools/report_figures.py`, not yet written, which reads only those sources, so that a reviewer can regenerate every panel without the gated data. Key paths are given as they appear in the files on commit `f7e7d87`. Where a figure needs a number that is not yet committed, it says so.

Conventions for all panels:

- **Budget axis.** Budgets sit on the log2(1 + n) axis, so 0, 1, 3, 7, 15 and 31 are equally spaced, as the competition's ALC defines it.
- **Paired differences.** They are drawn with the cluster-bootstrap SE (the middle of "run / cluster / stratified"). The run SE is too small to be honest (§4.3 of the draft).
- **Parents.** Always name the four parents (matharena, multi_swebench, real_webagents, researchcodebench) when a panel splits by them. Never show anonymous hidden benchmark ids.

---

## Main text

### Fig. 1: The protocol, budget-major, with one shared `labeled` list (§2.2)

A schematic, not a plot. It shows three pairs on two benchmarks, their acquisition and evaluation halves, and the six checkpoints. At each checkpoint the evaluation workers are recreated, and every target receives the same `labeled` list, which holds every pair's first B labels. An inset contrasts it with the legacy pair-major order, where one pair runs through all six budgets before the next starts.

- Data: none. Content from `docs/protocol.md` (Flow, Scoring) and `paiec/official.py`.

### Fig. 2: What the first replica got wrong (§2.3)

Two panels on one ALC axis.

- **Left: the legacy replica.** The predictor ladder: constant 0.5, empirical mean, smoothed Beta(2,2), own labels plus prior, pooled difficulty plus prior, and the assembled predictor in both evaluation orders (0.1898 and 0.1725).
- **Right: the official replica, R1.** Constant, empirical mean, smoothed, pooled anchor, Predictor with run-LOBO prior, Predictor with target-LOBO prior, and the base-rate oracle.

The point is the collapse of the pooled-difficulty lever, from about 0.020 to about 0.0007, and the empirical mean falling behind 0.5.

- Left: `docs/findings.md`, "The predictor ladder" and "Evaluation order is worth 0.017" (`experiments/ladder.py`, `experiments/order_sensitivity.py`). These are legacy numbers and should be labelled as such in the panel.
- Right: `results/official_baselines.json`, `r1.table` (ALC and per-budget means) and `r1.paired_diffs` (`experiments/official_baselines.py`).

### Fig. 3: Formative feedback against the regimes (§5.1, §5.4)

Brier by budget, one panel.

- Formative run 1: its nine pairs as thin lines and their mean as a thick line (legacy Predictor).
- Formative run 2: mean only (shipped hier).
- For each model, the replica's mean on test-like runs and on public R1, with a band of ± one single-run sd.

It shows the budget-0 optimism, the level fix, and the flat B3 to B31 of run 2 against the test-like expectation.

- Run 1 per pair: `docs/findings.md`, "Against the first real formative feedback"; the same rows are `paiec.testlike.FEEDBACK`.
- Run 2 budget means: `docs/findings.md`, "Subject side at budgets 0 and 1". **Missing**: run 2's per-pair rows (TODO(record) into findings.md before drawing them).
- Shipped hier, test-like, mix/whole, no-shift, R1: `results/subject_side.json`, `summary.regimes.<regime>.ship.budgets`.
- Legacy Predictor, test-like: `results/level_calibration.json`, `summary.testlike_all` (Predictor row, `budgets`). Legacy Predictor, R1: `results/level_calibration.json`, `summary.r1`.
- Single-run sds per budget: `results/testlike_check.json`, `summary.check` (Predictor). **Check**: the per-budget sds for the shipped hier are not stored; compute them from `results/level_calibration.json` `raw` rows for the shipped config, or run the plotting script over `data/subject_side_rows`.

### Fig. 4: Where the pairs sit: public, test-like and hidden (§1.2, §3.4, §4.1)

Distributions of the pair-accuracy logit on the evaluated responses. Three series:

- public R1 pairs (a histogram);
- test-like pairs at the default regime (a histogram);
- the nine formative run 1 pairs read through the lower root of p(1-p) = B31 (rug marks).

Mark the level means: public -0.74, test-like realised -1.29, feedback about -1.6. A second rug should show the feedback pairs' upper roots, because the audit reads two of them (bench F) as high-rate pairs.

- Public R1: `experiments/hier_design/levels.json` (from `levels.py` after `runs.py`).
- Test-like realised levels: `results/testlike_check.json`, `summary.check` (regime statistics) and `summary.level_identification`.
- Feedback: `paiec.testlike.FEEDBACK` and `paiec.testlike.rate_from_brier`.

### Fig. 5: Calibrating the level prior (§3.4, §5.3)

A heat map of selection-half ALC over mu0 (x) and attr_scale (y) at sigma_mu 2.5. Overlay contours of the public-guard cost at +0.003 on each weighting, the 40-run screen where only that is available. Mark four points: the rule's argmax (mu0 -3.5, attr_scale 0.25), the recommended configuration (mu0 -3.0, 0.25), the shipped configuration (mu0 -2.5, 0.5) and hier's fitted defaults. A side strip should show that sigma_mu 2.5 wins in every cell (the edge that binds).

- `results/level_calibration.json`: `summary.grid_surface` for the surface, `summary.selection` for the per-configuration ALC and paired differences, `summary.r1` for guard costs, and `summary.shortlist` for the confirmed set (`experiments/level_calibration.py`).

### Fig. 6: What the level fix trades, budget by budget (§5.3)

The paired Brier difference against the legacy Predictor at each budget, with cluster-SE bars, on test-like runs and on both public weightings. Plot it for the shipped configuration, the recommended one and hier's fitted defaults. It shows the large B0 and B1 gains on test-like runs, the B0 cost on public runs, and hier giving that cost back from B7 on.

- Test-like, selection half: `results/level_calibration.json`, `summary.selection["hier G mu0=-2.50 sm=2.50 as=0.50"].diff_budgets` and the same for `mu0=-3.00 ... as=0.25`.
- Public: `summary.r1` for the configurations scored there. **Check**: the shipped configuration's public per-budget differences may exist only for the 40-run screen; if so, use `results/subject_side.json` `summary.regimes.r1b/r1p.ship.budgets` against the Predictor's R1 budgets on matching runs, and label the run sets.

### Fig. 7: Pooling a benchmark's level needs company (§5.2)

hier minus the legacy Predictor, per pair appearance on multi-subject benchmarks, grouped by how many pairs of the target's benchmark the run holds (alone, 2, 3 or more). Show all four settings (benchmark-first or pair-uniform × split scope), with the swe_rebench single-subject pair as a separate marker. Annotate the real run's composition (5 alone, 4 in twos).

- `docs/findings.md`, "Hierarchical model" (the pairs-per-benchmark table).
- Underlying rows: `results/hier_eval.json`, `r1.<weighting>/<scope>.diffs` (`experiments/hier_eval.py --summarise` rebuilds the grouping).

### Fig. 8: The acceptance gate (§4.5)

The nested test-like ALC difference against the honest correlation r (0, 0.1, 0.2, 0.3, 0.5, 0.7), with two lines (transferred slope, per-pair slope) and cluster-SE bands. Add the gate at -0.002 as a horizontal rule and the honest oracle as a point at the right.

A secondary x axis gives the realised within-pair r on test-like runs (-0.002, 0.081, 0.163, 0.247, 0.420, 0.609). Overlay the measured covariates at their within-pair r, with their forced per-pair differences:

- stated_size +0.41, -0.0022;
- position -0.09;
- position_within -0.04;
- format_score -0.01;
- log_length +0.05;
- image_ref -0.02.

- `results/harness_thresholds.json`: `thresholds.honest["transferred nested"]`, `thresholds.honest["per-pair nested"]` (`tl`, `tl_cluster_se`, `tl_draw_sd`), `tables.honest["r=..."].meta.r_within_pair_tl`, `acceptance.oracle_honest` (`experiments/harness.py --stage table`).
- Covariates: `results/itemcov_eval.json`, `harness` (`experiments/itemcov_eval.py`).

---

## Negative results

### Fig. 9: Every idea against the gate (§6)

A forest plot. One row per idea gives its nested test-like ALC difference against the shipped model, with its cluster SE, and a second marker for the worst public weighting. Draw vertical rules at 0 and at the gate (-0.002). Group the rows:

- the item-side layer (itemsig, library default and nested);
- the known-sign cues (six covariates, nested and forced);
- the subject side (E, H, the date forms, T, T1.8, the combined selection);
- the multiple-choice floor correction (adopted), for contrast.

Leave space for the `TODO(step5)` studies and meta-heads (`TODO(record)`).

- itemsig: `results/itemsig_eval.json`, `summary.regimes.<regime>.nested_within`, `.nested_joint`, `.default`, `.in_sample_best`.
- Known-sign cues: `results/itemcov_eval.json`, `harness` (nested and forced lines per covariate) and `harness_train_scale`.
- Subject side: `results/subject_side.json`, `summary.nested`, `summary.gates`, `summary.per_config`, `summary.student_t`.
- MCQ floor: `results/mcq_floor.json`, `regimes`.

### Fig. 10: Item difficulty does not transfer (§6.1, §6.2)

Per benchmark, grouped bars of Pearson r with item difficulty:

- TF-IDF, leave-one-benchmark-out;
- embedding ridge, leave-one-benchmark-out;
- embedding kNN, leave-one-benchmark-out;
- embedding ridge within the benchmark (5-fold);
- item_features group mean within the benchmark;
- embedding with whole groups held out;
- the blind LLM judge.

Draw the gate's honest r = 0.3 as a horizontal rule. It shows the within/across gap and that the within-benchmark signal is mostly group identity.

- Embeddings and TF-IDF+SVD: `results/emb_transfer.json`, `lobo.emb_ridge_centred`, `lobo.emb_knn_centred`, `lobo.tfidf_ridge_centred`, `within.emb_ridge`, `group_only`, `within_groupcv.emb_ridge` (`experiments/emb_transfer.py`).
- TF-IDF text map (legacy data analysis): `docs/findings.md`, "What transfers between benchmarks" (`experiments/transfer.py`).
- LLM judge: `docs/findings.md`, "Language-model difficulty judgement" (`experiments/llm_rating/analysis.py`, with CIs).

### Fig. 11: The item-level gap nobody reaches (§6.3, §6.9)

Per regime (test-like, mix/whole, R1 benchmark-first, R1 pair-uniform), three bars: the pair-rate oracle, the item oracle and the shipped model's B31. Annotate the gap (0.063 test-like) and the share the itemsig layer recovers at B31 (0.5%).

- `results/itemsig_eval.json`, `summary.regimes.<regime>.oracle`.
- Note the `testlike.item_oracle` divergence documented in findings. Use the script's `theta_map` refit, which is what the summary stores.

---

## Appendix figures

### Fig. A1: The level prior adapts under empirical Bayes (§5.3)

The EB level centre by budget (0, 1, 3, 7, 31), one line per regime: public benchmark-first, public pair-uniform, test-like with no shift, level_mean -1.2, default and -2.0. Mark each regime's mean pair logit.

- `results/level_calibration.json`, `summary.eb_adaptation`.

### Fig. A2: Split scope decides what dense runs can read (§2.4)

For dense matharena, real_webagents and researchcodebench, show two panels:

- the share of a pair's evaluation items that carry other subjects' acquired labels at B31, under split scope 'pair' and 'benchmark';
- the Predictor's B31 under each scope against the base-rate oracle.

- `results/official_baselines.json`, `r2.<benchmark>`. The coverage shares and the scope comparison are also in `docs/findings.md`, "One benchmark, every pair (R2)".

### Fig. A3: The empirical mean's ALC follows from base rates (§2.5)

A scatter of per-run empirical-mean ALC against the run's E[p(1-p)] over the 600 R1 runs. Draw the analytic line 0.025 + 1.2118 E[p(1-p)] and the exact expectation. Mark the organisers' leaderboard entry (0.1801) on the y axis, with the inverted E[p(1-p)] of about 0.128.

- `results/official_baselines.json`, `r1.per_run`, `r1.analytic`, `r1.leaderboard`.

### Fig. A4: hier ablations (§5.5)

A forest plot of each ablation and sensitivity minus hier's default on the primary R1 setting, with cluster and stratified SEs, and the Bonferroni bar for sixteen comparisons (2.95 SEs).

- `results/hier_eval.json`, `r1["benchmark/pair"].diffs`. The table is in `docs/findings.md`, "Ablations and sensitivities".

### Fig. A5: Pooling gain against subjects in the pool (legacy)

The gain over no pooling against the number of subjects (3 to 80), on log axes, with the N^0.7 guide. Label it as a legacy-replica measurement.

- `docs/findings.md`, "How fragile the pooling gain is" (`experiments/pool_robustness.py`).

---

## Open questions for the figures

- **Per-pair rows of formative run 2.** Figs. 3 and 4 need them: record them in `docs/findings.md` first.
- **Single-run sds by budget for the shipped hier** (Fig. 3). Compute from stored rows; they are not in a summary block.
- **Public per-budget differences for the shipped configuration** (Fig. 6). Confirm which run sets hold them.
- **Palette and size.** One palette across all figures: the legacy Predictor, hier's defaults, the shipped hier and the smoothed mean should keep the same colour everywhere. Keep panels readable in greyscale, and size them for a two-column NeurIPS page.
