# Figures for the technical report

These are the figures `docs/report/draft.md` needs, each with the data behind it. None has been drawn yet. Every figure should come from a committed file (`results/*.json`, a table in `docs/findings.md`, or a module constant), through one plotting script. The proposal is `tools/report_figures.py`, not yet written, which reads only those sources, so that a reviewer can regenerate every panel without the gated data. Key paths are given as they appear in the files on commit `4d2cc4f` and in the results files the draft's Appendix B lists as not yet committed (`results/ship_confirm.json`, `results/formative_feedback.json`, `results/level_audit.json`, `results/heads_eval.json`). Where a figure needs a number that is not stored, it says so.

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
- Formative run 2: its eight pairs as thin lines and their mean as a thick line (shipped hier, archive of ee5085a).
- For each model, the replica's mean on test-like runs and on public R1, with a band of ± one single-run sd.

It shows the budget-0 optimism, the level fix, and the flat B3 to B31 of run 2 against the test-like expectation. The caption must say that the two runs are different draws with no subject in common, so the gap between them is not a measured effect.

- Runs 1 and 2 per pair, with ECE: `results/formative_feedback.json`, `record.run1.pairs` and `record.run2.pairs` (`brier`, `ece_b`, `n`); budget means in `record.<run>.budgets`. The organisers' tables are `results/formative/run1.txt` and `run2.txt`. Label benchmarks with the relabelled letters of the draft's Appendix A.1, never the anonymous ids.
- Shipped hier, test-like, mix/whole, no-shift, R1: `results/ship_confirm.json`, `regimes.<regime>.shipped.<runs>.budgets` (the same numbers as `results/subject_side.json`, `summary.regimes.<regime>.ship.budgets`).
- Single-run sds of the shipped hier per budget: `results/ship_confirm.json`, `single_run_sd.<regime>` (also `regimes.<regime>.shipped.<runs>.single_run_sd`).
- Legacy Predictor, test-like: `results/ship_confirm.json`, `regimes.tl.vs.legacy['0-299'].budgets[*].other` (paired with the shipped rows), or `results/level_calibration.json`, `summary.testlike_all`. Legacy Predictor, R1: `regimes.r1b.vs.legacy['0-149'].budgets[*].other` and `regimes.r1p.vs.legacy['0-99'].budgets[*].other`.
- Single-run sds of the legacy Predictor per budget: `results/testlike_check.json`, `summary.check` (Predictor).
- Optional overlay: the shipped model's matched estimate on run 1 by budget, `results/formative_feedback.json`, `reading.q2_shipped_on_run1['pool_B_K=15'].estimate_budgets`, labelled as a prediction for a like-sized run.

### Fig. 4: Where the pairs sit: public, test-like and hidden (§1.2, §3.4, §4.1)

Distributions of the pair-accuracy logit on the evaluated responses. Three series:

- public R1 pairs (a histogram);
- test-like pairs at the default regime (a histogram);
- the 17 formative pairs of runs 1 and 2, as rug marks at the root their neighbours favour, with the other root as an open mark.

Mark the level means on one scale (the continuity-corrected pair logit): public R1 realised -0.71, test-like realised -1.29, run 1's lower-root reading -1.51, and the pooled 17-pair reading -0.65 (K = 15) to -0.78 (K = 40) with its SE of about 0.5. Also mark the shipped LEVEL's implied mean on this scale, -1.15.

- Public R1 and test-like realised levels: `results/formative_feedback.json`, `reading.comparison.public_R1_realised` and `.tuned_regime_realised`; the pair-level distributions behind them in `experiments/hier_design/levels.json` (public) and `results/testlike_check.json`, `summary.check` (test-like).
- Feedback pairs: `results/formative_feedback.json`, `reading.per_pair['K=15'|'K=40']` (`run1_pool_A`, `run2_pool_B`: both roots, the neighbour share above 0.5, the chosen root and its logit), `reading.level_distribution` (pooled means, SEs, plain and corrected logits) and `reading.comparison.run1_readings_recomputed` (run 1's three readings).
- The shipped LEVEL on this scale: `reading.comparison.shipped_LEVEL_pair_logit_scale`.

### Fig. 5: Calibrating the level prior (§3.4, §5.3)

A heat map of selection-half ALC over mu0 (x) and attr_scale (y) at sigma_mu 2.5. Overlay contours of the public-guard cost at +0.003 on each weighting, the 40-run screen where only that is available. Mark four points: the rule's argmax (mu0 -3.5, attr_scale 0.25), the recommended (neighbouring) configuration (mu0 -3.0, 0.25), the shipped configuration (mu0 -2.5, 0.5) and hier's fitted defaults. A side strip should show that sigma_mu 2.5 wins in every cell (the edge that binds).

- `results/level_calibration.json`: `summary.grid_surface` for the surface, `summary.selection` for the per-configuration ALC and paired differences, `summary.r1` for guard costs, and `summary.shortlist` for the confirmed set (`experiments/level_calibration.py`).

### Fig. 6: What the level fix trades, budget by budget (§5.3)

The paired Brier difference against the legacy Predictor at each budget, with cluster-SE bars, on test-like runs and on both public weightings. Plot it for the shipped configuration, the recommended one and hier's fitted defaults. It shows the large B0 and B1 gains on test-like runs, the B0 cost on public runs, and hier giving that cost back from B7 on.

- Shipped configuration: `results/ship_confirm.json`, `regimes.tl.vs.legacy['0-299'].budgets` (test-like, 300 runs), `regimes.r1b.vs.legacy['0-149'].budgets` and `regimes.r1p.vs.legacy['0-99'].budgets` (public), each with `diff` and `cluster_se` per budget, on runs paired pair for pair. Label the run sets.
- Recommended configuration and hier's defaults: `results/level_calibration.json`, `summary.selection[...].diff_budgets` (test-like selection half) and `summary.r1` (public).

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
- the meta-learned heads (nested lines are exactly 0; show the forced lines);
- the closed language-model and encoder probes that give a harness line (the 4B judge, the entropy and hidden-state heads, the fine-tuned encoder);
- the multiple-choice floor correction (adopted), for contrast.

The attempt probe (no ALC line: its accuracy was below the floor) and the pairwise comparisons (a q statistic, not an ALC line) go in the caption, not the plot.

- itemsig: `results/itemsig_eval.json`, `summary.regimes.<regime>.nested_within`, `.nested_joint`, `.default`, `.in_sample_best`.
- Known-sign cues: `results/itemcov_eval.json`, `harness` (nested and forced lines per covariate) and `harness_train_scale`.
- Subject side: `results/subject_side.json`, `summary.nested`, `summary.gates`, `summary.per_config`, `summary.student_t`.
- Meta-learned heads: `results/heads_eval.json`, `rows.current.configs` (and `rows.legacy.configs`, the study as it ran): `lopo_*` nested, `force_all_*` forced.
- Closed probes: `results/llm4b_close.json`, `results/hidden_state_probe.json`, `results/finetune_encoder.json` (their harness stages); in-context learning's value map in `results/icl_probe.json`.
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

- **Resolved:** run 2's per-pair rows (`results/formative_feedback.json`), the shipped hier's single-run sds by budget and its public per-budget differences (`results/ship_confirm.json`).
- **Code of the plotted rows.** Figs. 3, 6 and every shipped-model series come from rows of the run-2 archive's library (old multiple-choice floor, solver before the floored-fit fix). Say so in the captions; the corrected floor and the fix move ALC by at most 0.0005 (draft §5.4).
- **Palette and size.** One palette across all figures: the legacy Predictor, hier's defaults, the shipped hier and the smoothed mean should keep the same colour everywhere. Keep panels readable in greyscale, and size them for a two-column NeurIPS page.
