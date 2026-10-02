# Figures for the technical report

These are the figures `docs/report/draft.md` needs, each with the data behind it. Twelve are now drawn by `tools/report_figures.py` into `docs/report/fig/` (SVG with text as paths, and PNG at 200 dpi); the section "Drawn figures" below gives each one's caption, its file and the keys it reads. The rest of this file is the plan they came from, with a status line on every entry. Every figure reads committed results files only (`results/*.json`, enforced by the script's `Inputs.load`, which refuses any other path), never `data/`, scratch or a table in `docs/findings.md`, so a reviewer can regenerate every panel without the gated data:

```bash
python tools/report_figures.py           # all figures, and docs/report/fig/manifest.json
python tools/report_figures.py --check   # rebuild in a temporary directory and compare
pytest tests/test_report_figures.py      # the same checks, plus what each panel plots
```

`docs/report/fig/manifest.json` records, per figure, the results files it read with their sha256, the sha256 of each file written and the function that drew it, and once the script's sha256 and the library versions (Python 3.10.8, matplotlib 3.9.2, numpy 1.26.4). The output is byte-deterministic for a given matplotlib (no dates in the files, a fixed SVG hash salt, the bundled DejaVu Sans, the Agg backend, seeded jitter). `tests/test_report_figures.py` fails when an input or the script changed after drawing, when a figure stops building or reads another file, or, on the same matplotlib, when a fresh build differs from the committed bytes. It is skipped where matplotlib is not installed (it is not in `pyproject.toml`'s dependencies). The build takes about 5 s on one process and under 300 MB.

Plan entries below give key paths as they appear in the files on commit `4d2cc4f` and later. Where a figure needs a number that is not stored in `results/`, it says so.

Conventions for all panels:

- **Budget axis.** Budgets sit on the log2(1 + n) axis, so 0, 1, 3, 7, 15 and 31 are equally spaced, as the competition's ALC defines it.
- **Paired differences.** They are drawn with the cluster-bootstrap SE (the middle of "run / cluster / stratified"). The run SE is too small to be honest (§4.3 of the draft).
- **Parents.** Always name the four parents (matharena, multi_swebench, real_webagents, researchcodebench) when a panel splits by them. Never show anonymous hidden benchmark ids.
- **Style (drawn figures).** One colour per model everywhere, from the Okabe-Ito colour-blind-safe set: shipped hier blue `#0072B2`, legacy Predictor vermillion `#D55E00`, smoothed Beta(2,2) green `#009E73`, the neighbouring configuration orange `#E69F00`, hier's fitted level prior reddish purple `#CC79A7`, the calibrated smoothed mean sky blue `#56B4E9`; formative runs black with hollow markers. Every series also has its own marker and line style, so the panels read in greyscale. Sequential scales use cividis. Width 5.5 in (the NeurIPS text width), 6 to 7 pt type. Small ALC differences are plotted in units of 10^-3 ALC, as the axis labels say; ± bars are cluster SEs unless the caption says otherwise.

---

## Placement in the draft

`docs/report/draft.md` embeds the PNGs (`fig/<name>.png`, relative to the draft) where the text discusses them, with a short caption; the draft's App H.1 points readers here for the full captions and this table's map from draft figures to drawn ones (the captions no longer carry file pointers, which a PDF reader could not follow). Main-text figures are numbered 1 to 7 in order of appearance; appendix figures carry their appendix's letter (A1, B1, D1, E1, F1). The captions below are the full ones; the draft's are condensed from them and quote no other number. The draft names the models as in its §2.1 (LegacyP, hier-ship, hier-rec, hier-fit, Smooth, Smooth-cal, hier-EB, hier-nosubj); the drawn legends keep the older names ("legacy Predictor", "shipped hier", "neighbouring config" and so on), and each draft caption gives both. The pre-trim draft, `docs/report/draft_v1_long.md`, numbered the same figures 1 to 12 in its own order (D10, D3, D1, D5, D2, D6, D7, D4, D11, D12, D9, D8).

| draft figure | section | drawn figure | file | the claim it carries |
|---|---|---|---|---|
| Figure 1 | §5.1 | D1 | `learning_curves` | the formative runs sit within single-run noise of both the tuned and the public regimes |
| Figure 2 | §5.3 | D2 | `level_surface` | a flat plateau; the widest level prior wins; the guard, not the selection half, decides |
| Figure 3 | §5.4 | D6 | `level_fix_budgets` | the tuned gain is at B0 and B1; on public runs hier-ship pays at B0 and recovers from B7 |
| Figure 4 | §5.4 | D4 | `regime_sensitivity` | at the feedback's readings no alternative beats hier-ship by the rule's 0.002 |
| Figure 5 | §6.1 | D3 | `gate_curve` | a transferred slope needs an honest r of about 0.3; no measured covariate reaches the gate |
| Figure 6 | §6.2 | D12 | `ideas_forest` | no idea's nested line reaches -0.002 |
| Figure 7 | §6.3 | D8 | `item_gap` | the item-level headroom is large, and the itemsig layer recovers 0.5% of it |
| Figure A1 | App A.4 | D10 | `empirical_mean` | the replica agrees with the analytic ALC of the empirical mean |
| Figure B1 | App B.3 | D11 | `hier_ablations` | which of hier's components pull their weight on public runs |
| Figure D1 | App D.2 | D5 | `formative_runs` | the per-pair spread of each formative run |
| Figure E1 | App E.3 | D7 | `eb_adaptation` | the EB centre moves the right way, and the date shift moves it further down |
| Figure F1 | App F.2 | D9 | `transfer` | item difficulty from text maps does not transfer between benchmarks |

Two notes for readers of the draft. D9's and D3 (c)'s TF-IDF bars are `experiments/emb_transfer.py`'s TF-IDF+SVD ridge against full-sample Rasch difficulty, not the `experiments/transfer.py` map the draft's §6.3 and App F.2 quote (whose target is the naive solve-rate logit; `results/gate_and_ci.json`, `ci.rows`). And the P1 results files of 2026-10-02 (`results/baselines_p1.json`, `results/pooling_decomposition.json`, `results/gate_and_ci.json`) are not read by any figure; the draft's tables carry those claims (the pass probabilities in §6.1, pooling by company in §2.4 and App A.2, the decomposition in §5.5), so the candidate panels (a gate panel with the pass probabilities, a pooling-by-company panel, plan Fig. 7, and a decomposition bar for P1.9) are not planned.

## Drawn figures

Each entry: the file in `docs/report/fig/` (`.svg` and `.png`), the plan entry it implements, a caption for the report, and the keys it reads. Which library the shipped hier's replica rows come from differs by figure. In D1, D2, D5, D6, D7 (a) and D8 they are formative run 2's (`paiec/hier.py` 70a3a81a: the old multiple-choice floor, before the floored-fit fix); the corrected floor and the fix move ALC by at most 0.0005 (`results/ship_confirm.json`, `code_gap`; draft App E.7 and App H.2). D3's harness rows (library 3f75a549, except the known-sign cues, scored on the legacy rows 0c05d35e) and the P1a runs of D4 and D7 (b, c) (`paiec/hier.py` d9a95612, the archive's code at `4d2cc4f`) have both fixes. Run sets differ between files and are named in each caption (review m6).

### D1. `learning_curves`: Brier by budget, by regime, with the formative runs (review m8 and P2.20 item 1; plan Fig. 3)

**Caption.** Brier score by budget (log2(1+B) axis) for the shipped hier (blue; the band is ±1 single-run sd, the spread of one run's mean over the regime's runs), the legacy Predictor (vermillion, dashed) and the Beta(2,2) smoothed mean (green, dash-dot) in five replica regimes. The three formative runs' pair means are overlaid in black in every panel: run 1 is the legacy Predictor (ALC 0.2113), run 2 the shipped hier (0.1926), run 3 the shipped hier's rebuilt archive (0.1817, recomputed from its table; the headline score was not pasted). Run sets: (a) tuned test-like, seed 2, shipped and legacy on runs 0-299, smoothed on 0-199; (b) test-like with groups merged at random and wholes, seed 3, shipped on 0-199, comparators on 0-99; (c) test-like without the date shift, seed 3, runs 0-99; (d) public R1 benchmark-first, runs 0-149; (e) public R1 pair-uniform, runs 0-99. Each comparator is drawn over the runs it was matched on pair for pair. The formative runs are three draws with different subjects (one pair recurs between runs 1 and 3, under different models), 8 or 9 pairs on 7 benchmarks each, so the gaps between them are not measured effects. Placed against the shipped model's single-run sds, run 2 lies +0.14 to +1.40 sd above the tuned test-like mean by budget (+0.84 for ALC) and -0.79 to +0.24 sd around public benchmark-first (-0.44); run 3 lies -0.06 to +0.89 sd and -1.08 to 0.00 sd (+0.50 and -0.82 for ALC).

- Shipped hier: `results/ship_confirm.json`, `regimes.<regime>.shipped.<widest span>.budgets` (runs and ALC in the panel's corner); band: `single_run_sd.<regime>.B0..B31`.
- Legacy Predictor and smoothed mean: `regimes.<regime>.vs.legacy|smoothed.<widest span>.budgets[*].other`.
- Formative runs: `results/formative_feedback.json`, `record.run1|run2.budgets.brier_mean`; `results/formative_run3.json`, `run3.budgets.brier_mean` (checked against `three_runs.table`). The z values above: `results/formative_run3.json`, `placement.regimes.<regime>.run2_exact_budgets` and `.run3`.

### D2. `level_surface`: the level-calibration surface (plan Fig. 5)

**Caption.** (a) Test-like selection-half ALC (runs 0-99, seed 2) of hier over the level prior's centre mu0 and the attribute scale attr_scale, at sigma_mu 2.5; lower is better. Upper number: ALC. Lower number: the public cost against the legacy Predictor, the worse of the two R1 weightings on 100 runs each, or, marked ˢ, the 40-run benchmark-first screen where only that was scored. The shipped cell, which the level calibration screened only, carries the 100-run numbers the level audit scored on the same runs (marked ᵃ: -0.0015 benchmark-first and -0.0018 pair-uniform, so -0.0015; its screen value was -0.0026). Hatched cells fail the guard (cost above +0.003); 21 of the 30 scored cells have a guard number and 4 fail it. Marked: the selection rule's choice (mu0 -3.5, attr_scale 0.25, ALC 0.1574), the neighbouring configuration the level calibration recommended (-3.0 / 0.25, 0.1578), and the shipped configuration (-2.5 / 0.5, 0.1608), which the audit chose. hier's fitted level prior is not on this grid; it scores 0.1906 on the same runs, and the legacy Predictor 0.2039. Grey cells were not scored. (b) In every scored cell, ALC at sigma_mu 0.9, 1.3 and 1.8 minus ALC at 2.5 in the same cell: all positive (smallest 0.0002 at 1.8), so 2.5, the edge of the grid, is best everywhere; P1a's sigma_mu 3.5 and 5.0 are in D4.

- `results/level_calibration.json`: `summary.grid_surface` (ALC at every sigma_mu), `summary.r1.r1b|r1p.<config>.diff.mean` (guard, 100 runs) or `summary.shortlist.screen.<config>` (screen, 40 runs), `summary.selected.chosen`, `summary.ship.config` (the level calibration's recommendation), `summary.selection.configs.hier|Predictor.ALC`, `config.guard`.
- The shipped cell's guard: `results/level_audit.json`, `mild["r1b 0-99: …"|"r1p 0-99: …"]["vs PRED"]["hier G mu0=-2.50 sm=2.50 as=0.50"]["minus ref (run / cluster / stratified SE)"][0]` (the same seed-0 runs 0-99: the neighbouring configuration's audit values, +0.0008 and +0.0007, equal the level calibration's).

### D3. `gate_curve`: the acceptance gate, with the measured covariates on it (plan Fig. 8, with a strip from Fig. 10)

**Caption.** (a) The harness's nested test-like ALC difference against the shipped hier, in 10^-3, for a synthetic covariate x = r z + sqrt(1 - r^2) e at honest correlation r with item difficulty (bottom axis; the top axis gives the realised within-pair r on test-like runs). Lines: the transferred slope (blue) and the per-pair slope (orange), each the mean of 8 noise draws with a ±1 cluster-SE band; dots: the 8 draws; labels: draws that pass the gate on their own. The gate is -0.002. The transferred slope passes most draws from r = 0.3 and every draw from 0.5; the per-pair slope from 0.5. The honest oracle (-0.0360 ± 0.0030) is off scale. (b) Zoom near the gate, on the within-pair r axis: each covariate scored through the harness, at the absolute value of its within-pair r (the harness fits the slope's sign) and its nested line with ±1 cluster SE: the transferred line, and for the known-sign cues their nested selection over the allowed forms (0 for four cues, which nested selection cannot switch on where a cue varies on one parent; +0.00003 for log_length and image_ref, switched on in some folds; stated_size, at r 0.41, exists on researchcodebench alone). None of the 44 reaches the gate: the best is the 14B's time_log_minutes (-0.0008 ± 0.0004), and some lines are switched on and lose, up to +0.0013 (the 14B's ent_first256). Markers are translucent because points overlap: many covariates sit at exactly 0 (never switched on), and the 14B's rubric_reasoning (|r| 0.094, +0.0011) lies under its lp_first256 (0.093, +0.0011). The per-pair nested lines of the same covariates lie between -0.0002 and +0.0001. The 4B judge covers 61% of test-like evaluated items and the 4B hidden-state probes 51%; the known-sign cues were scored on the legacy harness rows (library 0c05d35e), everything else on the current rows (3f75a549). (c) Signals read off the item text, per benchmark: Pearson r with item difficulty over the benchmark's items and its 95% group-bootstrap CI. TF-IDF and embedding rows are predictions from the other three benchmarks (leave one benchmark out) against full-sample Rasch difficulty; the 4B judge's rating (sign as declared, rated on two benchmarks), the 14B rubric head (the declared primary, out of fold) and the 14B's ent_first1024 (commit D's primary) are against honest difficulty (mean over 5 subject folds). These are pooled correlations, not the within-group ones the draft quotes for entropy. The grey band is the honest r over which the transferred slope goes from passing most draws to passing every draw.

- Gate table: `results/harness_thresholds.json`, `thresholds.honest["transferred nested"|"per-pair nested"]` (`tl`, `tl_cluster_se`, `pass_draws`, `smallest_r_majority_of_draws`, `smallest_r_every_draw`), `tables.honest["r=..."].meta.r_within_pair_tl`, `tables.honest["r=..."].lines.<line>.replicates.tl`, `acceptance.oracle_honest["transferred nested"].regimes.tl`, `meta.gate`.
- Covariates (b): `harness.<covariate>` (`r_within_pair_tl`, `lines["transferred nested"].tl`, `.tl_cluster_se`, `coverage_eval_items`) in `results/llm4b_close.json` (rating, digit_mode, entropy, nll), `results/hidden_state_probe.json` (entropy, surprisal, profile, hidden, hidden_mean, hidden_L9/18/27/36), `results/strong_llm_eval.json` (heads rubric_ridge, rubric_ridge_posfree, rubric_all_ridge, judge_ridge; rubric_sum, the eight rubric features, solve_share, time_log_minutes) and its `entropy.harness` (ent_first1024, ent_first256, lp_first1024, lp_first256, ent_n_tokens, ent_closed; the attempt features, on one parent, are left out), `results/finetune_encoder.json` (frozen_nested, frozen_lobo512, finetuned, finetuned_e3); `results/itemcov_eval.json`, `harness.<cue>.lines["allowed nested"].regimes.tl`.
- Correlations (c): `results/emb_transfer.json`, `lobo.tfidf_ridge_centred|emb_ridge_centred|emb_knn_centred.<parent>.pearson` and `.pearson_ci_groupboot`; `results/llm4b_close.json`, `signs.features.rating` (`declared_sign`, `units.<parent>.pearson`); `results/strong_llm_eval.json`, `heads.rubric_ridge.per_parent.<parent>.pearson` and `entropy.signs.features.ent_first1024.units.<parent>.pearson`.

### D4. `regime_sensitivity`: the shipped level at the feedback's reading, P1a (plan Fig. A6)

**Caption.** Each alternative minus the shipped hier, ALC in 10^-3 (negative: the alternative is better), in seven regimes ("hier, no subject prior" is P1a's `onepl`, hier with its attribute prior and identity link off; it is not the plain 1PL of P1.9, which is not drawn), test-like ones ordered by realised level: TUNED (-1.28, sd 1.62), AUDIT (-0.93, 2.05), MIXTURE (-0.64, 2.13), READING (-0.63, 1.68), FLAT (-0.28, 2.23), then public R1 benchmark-first and pair-uniform; 80 test-like and 60 public runs a regime (seed 11). Thick bar ±1 cluster SE, thin bar ±1.96 cluster SE (its right end is the rule's U95); hollow marker: the mean of the four parents' means. Dashed rule: the gain the rule required, -0.002, in READING and AUDIT; dotted rule: the loss bound, +0.002 on test-like and +0.001 on public regimes. Off-scale values are given at the edge. No alternative crosses -0.002 in READING or AUDIT (the best is the EB level on -3.0 / 0.5 in AUDIT, -0.0005 ± 0.0007); the neighbouring configuration wins only in TUNED (-0.0034 ± 0.0010); the calibrated smoothed mean is level with the shipped model only in TUNED (-0.0015 ± 0.0019) and loses 0.004 to 0.011 elsewhere. The rule's outcome: no candidate, the shipped configuration stays. Every regime redraws the catalogue that chose the level, and READING and AUDIT realised -0.63 and -0.93, milder than targeted.

- `results/regime_sensitivity.json`: `summary.vs_ship.<regime>.<config>.exact` (`D`, `cluster_se`, `U95`), `summary.vs_ship.<regime>.<config>.table.parent_level.mean`, `summary.realised.<regime>` (`mean`, `sd`), `summary.configs.<regime>.ship.runs`, `rule.thresholds`, `rule.outcome`.

### D5. `formative_runs`: the formative runs pair by pair (plan Fig. 3, per-pair part)

**Caption.** Each formative run's pairs (grey, thin) and their mean (black), by budget. Reference lines for the run's model: the legacy Predictor for run 1 and the shipped hier for runs 2 and 3, as the replica's mean on tuned test-like runs (solid) and on public R1 benchmark-first runs (dashed), with ±1 single-run sd bands for the shipped hier. Run 1's B0 spans 0.17 to 0.57 over its 9 pairs; runs 2 and 3 span 0.16 to 0.27 and 0.20 to 0.28. The three runs are different draws (see D1); benchmarks are not labelled.

- `results/formative_feedback.json`, `record.run1|run2.pairs[*].brier` and `.budgets.brier_mean`; `results/formative_run3.json`, `run3.pairs[*].brier`, `run3.budgets.brier_mean` and `three_runs.table` (pairs, benchmarks); reference lines as in D1 from `results/ship_confirm.json`.

### D6. `level_fix_budgets`: what the level fix trades, budget by budget (plan Fig. 6)

**Caption.** Paired Brier difference against the legacy Predictor by budget. (a) The shipped configuration in five regimes, ±1 cluster SE: tuned test-like (runs 0-299), mix/whole (0-99), no date shift (0-99), public benchmark-first (0-149) and pair-uniform (0-99). On tuned test-like runs it gains 0.135 at B0 and 0.076 at B1 (0.140 and 0.077 on mix/whole; 0.040 and 0.023 without the date shift); on public runs it costs 0.005 (benchmark-first) and 0.014 (pair-uniform) at B0 and gains from B3 on. (b) Three configurations on the level calibration's runs, without SEs (the file stores none per budget for them): the shipped configuration, the neighbouring one and hier's fitted level prior, on the test-like selection half (runs 0-99, solid) and public benchmark-first runs 0-99 (dashed). The shipped configuration's public line comes from `results/ship_confirm.json` on the same 100 runs, because the level calibration did not score it there.

- (a) `results/ship_confirm.json`, `regimes.<regime>.vs.legacy.<widest span>.budgets[*]` (`diff`, `cluster_se`).
- (b) `results/level_calibration.json`, `summary.selection.configs.<config>.diff_budgets` and `summary.r1.r1b.<config>.diff_budgets`; `results/ship_confirm.json`, `regimes.r1b.vs.legacy["0-99"].budgets[*].diff`.

### D7. `eb_adaptation`: the empirical-Bayes level centre by budget (plan Fig. A1)

**Caption.** The EB estimate of a new benchmark's level centre (item-level logit) by budget, mean over runs; colour is the regime's mean pair logit on one scale for all panels (dark: low), dashed lines are public regimes. (a) EB on hier's fitted level prior (tau 2) in the level calibration's regimes; it stays near the public centre on public runs and moves down from the first label on test-like ones, ordered by level. (b, c) EB on the shipped prior and on mu0 -3.0 / attr_scale 0.5 in the P1a regimes (fresh-seed runs on the library of `4d2cc4f`, labelled apart from (a)). The EB centre is on the item-level scale and the regimes' levels on the pair-accuracy scale. The two are ordered alike only within regimes that share the date shift: the date shift moves the centre further down, because the inflated attribute standings are taken back (docs/findings.md, "The empirical-Bayes level adapts the right way"). So the date-shifted test-like lines end below the unshifted ones (in (a), "tl no shift" at level -1.33 ends at -2.53, above "tl" at -1.29, which ends at -4.34), and in (b) public R1 pair-uniform (level -0.77) ends above FLAT, MIXTURE and READING (levels -0.28 to -0.64).

- (a) `results/level_calibration.json`, `summary.eb_adaptation.estimates["hierEB-cs,tm=2.0@hier | <regime>"].B*[0]`, `summary.eb_adaptation.pair_logit_mean`.
- (b, c) `results/regime_sensitivity.json`, `summary.eb_traces["eb_ship|eb_adapt | <regime>"].B*.mu0`, `summary.realised.<regime>.mean`.

### D8. `item_gap`: the item-level gap nobody reaches (plan Fig. 11)

**Caption.** Brier on the evaluated responses, per regime: the pair-rate oracle (every response at the pair's own rate), the item oracle (in-sample Rasch difficulty, theta refitted per pair by `theta_map`) and the shipped hier at B31. The gap between the two oracles (± cluster SE) is what an item-difficulty model could add; the itemsig layer's nested line recovers 0.5% of it at B31 on tuned test-like runs and 1.3% to 2.9% elsewhere (shares from unrounded values: findings rounds public benchmark-first to 1.4%). Run sets: test-like 0-299, mix/whole 0-199, public benchmark-first 0-199, pair-uniform 0-99.

- `results/itemsig_eval.json`, `summary.regimes.<regime>.oracle` (`pair_rate_oracle`, `item_oracle`, `base_b31`, `gap.mean`, `gap.cluster_se`) and `.nested_within.budgets[-1].diff`.

### D9. `transfer`: item difficulty does not transfer (plan Fig. 10)

**Caption.** Pearson r between predicted and Rasch item difficulty (full sample, subject ability divided out, standardised within benchmark), per benchmark. Leave-one-benchmark-out predictions (TF-IDF+SVD ridge, embedding ridge, embedding kNN; 95% group-bootstrap CIs) are null to negative; within a benchmark (5-fold ridge on embeddings) they reach 0.27 to 0.66, about what the item_features group mean alone gives, and fall when whole groups are held out. The dashed rule is the gate's bar for a transferred slope, honest r 0.3; these correlations are against full-sample difficulty, which flatters them (review W6). The blind LLM judge's correlations of the plan are only in `docs/findings.md` and are not drawn; the 4B and 14B judges are in D3 (c).

- `results/emb_transfer.json`, `lobo.tfidf_ridge_centred|emb_ridge_centred|emb_knn_centred.<parent>` (`pearson`, `pearson_ci_groupboot`), `within.emb_ridge.<parent>.pearson`, `group_only.<parent>.pearson`, `within_groupcv.emb_ridge.<parent>.pearson`; the bar from `results/harness_thresholds.json`, `thresholds.honest["transferred nested"].smallest_r_majority_of_draws`.

### D10. `empirical_mean`: the empirical mean's ALC follows from base rates (plan Fig. A3)

**Caption.** Each of the 600 public R1 runs: its empirical-mean ALC against its E[p(1-p)] over pairs. Line: the analytic approximation 0.025 + 1.2118 E[p(1-p)]. Black square: the mean over runs (0.2526 at E[p(1-p)] 0.181, ±1.96 SE); hollow diamond: the exact expectation there (0.2496). Vermillion: the organisers' leaderboard entry (0.1801), drawn at the E[p(1-p)] an empirical mean would need to score it (0.128); 2.5% of the replica's runs have a lower E[p(1-p)].

- `results/official_baselines.json`, `r1.per_run[*]` (`q`, `alc["empirical mean"]`), `r1.analytic` (`formula`, `E_q`, `observed`, `exact`), `r1.leaderboard.entries["organisers' entry"]` (`alc`, `q_if_empirical_mean`, `share_runs_q_below`).

### D11. `hier_ablations`: hier's ablations and sensitivities (plan Fig. A4)

**Caption.** Each of hier's sixteen options alone minus hier's default (fitted level prior), ALC in 10^-3, on public R1 benchmark-first runs with split scope pair (runs 0-149, or 0-99 where the label says 100). Thick bar ±1 pair-cluster SE; thin bars ±2.955 SEs, the two-sided Bonferroni bar for sixteen comparisons, pair-cluster (blue) and stratified by benchmark (grey). Removing the attribute prior, the pooled level or the feature groups costs 0.0012 to 0.0039; doubling either width costs 0.005 to 0.006. Three options beat the default by more than two pair-cluster SEs (sigma_mu ×0.5, the Student-t level, the Laplace fit without the line); none clears the Bonferroni bar on the pair-cluster SE, and all three do on the stratified one. These options were measured on hier's defaults, before the level calibration.

- `results/hier_eval.json`, `r1["benchmark/pair"].diffs["<option>"].hier` (`diff`, `cluster_se`, `cluster_se_strat`, `runs`), `config.n_comparisons`.

### D12. `ideas_forest`: every idea against the gate (plan Fig. 9)

**Caption.** Each idea's test-like ALC difference against the shipped model (10^-3, negative is better), ±1 pair-cluster SE; filled blue: the nested line the gate reads; hollow grey: a forced or fixed configuration; vermillion triangle: the same line's worse public R1 weighting (none for the meta-learned heads, which were scored on test-like and mix/whole runs only); a triangle off scale is drawn at the edge with its value, as for the 14B's time_log_minutes, whose two lines gain 0.0037 on public runs. Dashed: the gate, -0.002. No nested line reaches it. The best are the subject side's ordered reasoning effort E (-0.0010 ± 0.0003, on in all four folds, but short of -0.002) and the 14B's time_log_minutes (-0.0008 ± 0.0004); the meta-learned heads' nested lines are exactly 0 (never switched on), and forced they cost +0.0002 to +0.0006. The corrected multiple-choice floor, adopted as a bug fix rather than through the gate, gains 0.0003 for contrast. Footings differ: itemsig, the known-sign cues, the subject side and the multiple-choice floor were scored on the run-2 library's rows (the cues on the legacy harness rows), the probes on the current harness rows; itemsig's public lines are selected within each public regime. Left out, as the plan says: the attempt probe (no ALC line), the 14B's attempts (on one parent: every nested line 0, forced per-pair -0.0001) and the pairwise comparisons (a q statistic).

- itemsig: `results/itemsig_eval.json`, `summary.regimes.<regime>.nested_within|nested_joint` (`ALC.mean`, `ALC.cluster_se`) and `.default|in_sample_best` (`mean`, `cluster_se`).
- Known-sign cues: `results/itemcov_eval.json`, `harness.<cue>.lines["allowed nested"|"per-pair s=0.5 from B7 (forced)"].regimes.<regime>` (`est`, `cluster_se`).
- Subject side: `results/subject_side.json`, `summary.nested.primary.<regime>`, `summary.per_config.<config>.ALC.<regime>`, `summary.student_t.regimes.<regime>.configs.<T|T1.8>.dalc`.
- Meta-learned heads: `results/heads_eval.json`, `rows.current.configs.<lopo_*|force_all_*>.tl` (`alc`, `cluster_se`).
- Probes: `harness.<covariate>.lines["transferred nested"|"transferred from B1 (forced)"]` (`tl`, `tl_cluster_se`, `r1b`, `r1p`) in `results/llm4b_close.json`, `results/hidden_state_probe.json`, `results/finetune_encoder.json`, `results/strong_llm_eval.json` (and its `entropy.harness`).
- Multiple-choice floor: `results/mcq_floor.json`, `regimes.<regime>.compare["fix - ship"].all.per_appearance`.
- The gate: `results/harness_thresholds.json`, `meta.gate.tl`.

---

## Plan

Section numbers in the plan entries below are those of the pre-trim draft, `docs/report/draft_v1_long.md`; the table under "Placement in the draft" gives the current placement.

## Main text

### Fig. 1: The protocol, budget-major, with one shared `labeled` list (§2.2)

**Status:** not drawn. A schematic with no data; it belongs in a drawing tool, not in `tools/report_figures.py`.

A schematic, not a plot. It shows three pairs on two benchmarks, their acquisition and evaluation halves, and the six checkpoints. At each checkpoint the evaluation workers are recreated, and every target receives the same `labeled` list, which holds every pair's first B labels. An inset contrasts it with the legacy pair-major order, where one pair runs through all six budgets before the next starts.

- Data: none. Content from `docs/protocol.md` (Flow, Scoring) and `paiec/official.py`.

### Fig. 2: What the first replica got wrong (§2.3)

**Status:** not drawn. The left panel's legacy ladder exists only as tables in `docs/findings.md`, which the results-only rule excludes; the right panel can come from `results/official_baselines.json` (`r1.table`, `r1.paired_diffs`), but the base-rate oracle is not in that table.

Two panels on one ALC axis.

- **Left: the legacy replica.** The predictor ladder: constant 0.5, empirical mean, smoothed Beta(2,2), own labels plus prior, pooled difficulty plus prior, and the assembled predictor in both evaluation orders (0.1898 and 0.1725).
- **Right: the official replica, R1.** Constant, empirical mean, smoothed, pooled anchor, Predictor with run-LOBO prior, Predictor with target-LOBO prior, and the base-rate oracle.

The point is the collapse of the pooled-difficulty lever, from about 0.020 to about 0.0007, and the empirical mean falling behind 0.5.

- Left: `docs/findings.md`, "The predictor ladder" and "Evaluation order is worth 0.017" (`experiments/ladder.py`, `experiments/order_sensitivity.py`). These are legacy numbers and should be labelled as such in the panel.
- Right: `results/official_baselines.json`, `r1.table` (ALC and per-budget means) and `r1.paired_diffs` (`experiments/official_baselines.py`).

### Fig. 3: Formative feedback against the regimes (§5.1, §5.4)

**Status:** drawn, as D1 (regime means with single-run sd bands and the three runs' means, in every regime) and D5 (each run's pairs). The optional overlay of the matched estimate is not drawn.

Brier by budget, one panel.

- Formative run 1: its nine pairs as thin lines and their mean as a thick line (legacy Predictor).
- Formative run 2: its eight pairs as thin lines and their mean as a thick line (shipped hier, archive of ee5085a).
- Formative run 3: its nine pairs as thin lines and their mean as a thick line (shipped hier, archive of 4d2cc4f, uploaded as a regression and latency check only).
- For each model, the replica's mean on test-like runs and on public R1, with a band of ± one single-run sd.

It shows the budget-0 optimism, the level fix, and the flat B3 to B31 of run 2 against the test-like expectation. The caption must say that the three runs are different draws, sharing one (subject, benchmark) pair between runs 1 and 3 under different models, so the gaps between them are not measured effects, and that run 3's ALC is recomputed from its table (its headline score was not pasted).

- Runs 1 and 2 per pair, with ECE: `results/formative_feedback.json`, `record.run1.pairs` and `record.run2.pairs` (`brier`, `ece_b`, `n`); budget means in `record.<run>.budgets`. Run 3: `results/formative_run3.json`, `run3.pairs` (the same fields) and `run3.budgets`. The organisers' tables are `results/formative/run1.txt`, `run2.txt` and `run3.txt`. Label benchmarks with the relabelled letters of the draft's Appendix A.1, never the anonymous ids.
- Shipped hier, test-like, mix/whole, no-shift, R1: `results/ship_confirm.json`, `regimes.<regime>.shipped.<runs>.budgets` (the same numbers as `results/subject_side.json`, `summary.regimes.<regime>.ship.budgets`).
- Single-run sds of the shipped hier per budget: `results/ship_confirm.json`, `single_run_sd.<regime>` (also `regimes.<regime>.shipped.<runs>.single_run_sd`).
- Legacy Predictor, test-like: `results/ship_confirm.json`, `regimes.tl.vs.legacy['0-299'].budgets[*].other` (paired with the shipped rows), or `results/level_calibration.json`, `summary.testlike_all`. Legacy Predictor, R1: `regimes.r1b.vs.legacy['0-149'].budgets[*].other` and `regimes.r1p.vs.legacy['0-99'].budgets[*].other`.
- Single-run sds of the legacy Predictor per budget: `results/testlike_check.json`, `summary.check` (Predictor).
- Optional overlay: the shipped model's matched estimate on run 1 by budget, `results/formative_feedback.json`, `reading.q2_shipped_on_run1['pool_B_K=15'].estimate_budgets`, labelled as a prediction for a like-sized run.

### Fig. 4: Where the pairs sit: public, test-like and hidden (§1.2, §3.4, §4.1)

**Status:** not drawn. The public pair-level distribution is in `experiments/hier_design/levels.json`, outside `results/`; the level marks alone are in `results/formative_feedback.json`.

Distributions of the pair-accuracy logit on the evaluated responses. Three series:

- public R1 pairs (a histogram);
- test-like pairs at the default regime (a histogram);
- the 17 formative pairs of runs 1 and 2, as rug marks at the root their neighbours favour, with the other root as an open mark.

Mark the level means on one scale (the continuity-corrected pair logit): public R1 realised -0.71, test-like realised -1.29, run 1's lower-root reading -1.51, and the pooled 17-pair reading -0.65 (K = 15) to -0.78 (K = 40) with its SE of about 0.5. Also mark the shipped LEVEL's implied mean on this scale, -1.15.

- Public R1 and test-like realised levels: `results/formative_feedback.json`, `reading.comparison.public_R1_realised` and `.tuned_regime_realised`; the pair-level distributions behind them in `experiments/hier_design/levels.json` (public) and `results/testlike_check.json`, `summary.check` (test-like).
- Feedback pairs: `results/formative_feedback.json`, `reading.per_pair['K=15'|'K=40']` (`run1_pool_A`, `run2_pool_B`: both roots, the neighbour share above 0.5, the chosen root and its logit), `reading.level_distribution` (pooled means, SEs, plain and corrected logits) and `reading.comparison.run1_readings_recomputed` (run 1's three readings).
- The shipped LEVEL on this scale: `reading.comparison.shipped_LEVEL_pair_logit_scale`.

### Fig. 5: Calibrating the level prior (§3.4, §5.3)

**Status:** drawn, as D2. The grid is too sparse for contours, so each cell prints its guard cost and failing cells are hatched.

A heat map of selection-half ALC over mu0 (x) and attr_scale (y) at sigma_mu 2.5. Overlay contours of the public-guard cost at +0.003 on each weighting, the 40-run screen where only that is available. Mark four points: the rule's argmax (mu0 -3.5, attr_scale 0.25), the recommended (neighbouring) configuration (mu0 -3.0, 0.25), the shipped configuration (mu0 -2.5, 0.5) and hier's fitted defaults. A side strip should show that sigma_mu 2.5 wins in every cell (the edge that binds).

- `results/level_calibration.json`: `summary.grid_surface` for the surface, `summary.selection` for the per-configuration ALC and paired differences, `summary.r1` for guard costs, and `summary.shortlist` for the confirmed set (`experiments/level_calibration.py`).

### Fig. 6: What the level fix trades, budget by budget (§5.3)

**Status:** drawn, as D6. The neighbouring configuration and hier's defaults have no per-budget SE in the file, so they are drawn without bars.

The paired Brier difference against the legacy Predictor at each budget, with cluster-SE bars, on test-like runs and on both public weightings. Plot it for the shipped configuration, the recommended one and hier's fitted defaults. It shows the large B0 and B1 gains on test-like runs, the B0 cost on public runs, and hier giving that cost back from B7 on.

- Shipped configuration: `results/ship_confirm.json`, `regimes.tl.vs.legacy['0-299'].budgets` (test-like, 300 runs), `regimes.r1b.vs.legacy['0-149'].budgets` and `regimes.r1p.vs.legacy['0-99'].budgets` (public), each with `diff` and `cluster_se` per budget, on runs paired pair for pair. Label the run sets.
- Recommended configuration and hier's defaults: `results/level_calibration.json`, `summary.selection[...].diff_budgets` (test-like selection half) and `summary.r1` (public).

### Fig. 7: Pooling a benchmark's level needs company (§5.2)

**Status:** not drawn yet. The rows are in `results/hier_eval.json`, but the grouping by company is computed by `experiments/hier_eval.py --summarise`; porting it is the next figure to add.

hier minus the legacy Predictor, per pair appearance on multi-subject benchmarks, grouped by how many pairs of the target's benchmark the run holds (alone, 2, 3 or more). Show all four settings (benchmark-first or pair-uniform × split scope), with the swe_rebench single-subject pair as a separate marker. Annotate the real run's composition (5 alone, 4 in twos).

- `docs/findings.md`, "Hierarchical model" (the pairs-per-benchmark table).
- Underlying rows: `results/hier_eval.json`, `r1.<weighting>/<scope>.diffs` (`experiments/hier_eval.py --summarise` rebuilds the grouping).

### Fig. 8: The acceptance gate (§4.5)

**Status:** drawn, as D3 (a) and (b). The measured covariates sit on the within-pair r axis directly, where both they and the synthetic lines are measured, at |r|; they are drawn with their nested lines, not the forced per-pair ones listed below.

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

**Status:** drawn, as D12, with one row per line rather than per idea where an idea has a nested and a forced line, and with E, H, Dlog, Dclip and H+E standing for the subject side's ten single and combined configurations.

A forest plot. One row per idea gives its nested test-like ALC difference against the shipped model, with its cluster SE, and a second marker for the worst public weighting. Draw vertical rules at 0 and at the gate (-0.002). Group the rows:

- the item-side layer (itemsig, library default and nested);
- the known-sign cues (six covariates, nested and forced);
- the subject side (E, H, the date forms, T, T1.8, the combined selection);
- the meta-learned heads (nested lines are exactly 0; show the forced lines);
- the closed language-model and encoder probes that give a harness line (the 4B judge, the entropy and hidden-state heads, the fine-tuned encoder, the 14B's rubric heads and scales, and the 14B's reasoning entropy on all four parents, commit D);
- the multiple-choice floor correction (adopted), for contrast.

The attempt probe (no ALC line: its accuracy was below the floor), the 14B's attempts (GO for correlation, but on one parent: every nested line is exactly 0, and the forced per-pair line is -0.0001) and the pairwise comparisons (a q statistic, not an ALC line) go in the caption, not the plot.

- itemsig: `results/itemsig_eval.json`, `summary.regimes.<regime>.nested_within`, `.nested_joint`, `.default`, `.in_sample_best`.
- Known-sign cues: `results/itemcov_eval.json`, `harness` (nested and forced lines per covariate) and `harness_train_scale`.
- Subject side: `results/subject_side.json`, `summary.nested`, `summary.gates`, `summary.per_config`, `summary.student_t`.
- Meta-learned heads: `results/heads_eval.json`, `rows.current.configs` (and `rows.legacy.configs`, the study as it ran): `lopo_*` nested, `force_all_*` forced.
- Closed probes: `results/llm4b_close.json`, `results/hidden_state_probe.json`, `results/finetune_encoder.json` (their harness stages); in-context learning's value map in `results/icl_probe.json`; the 14B's `results/strong_llm_eval.json`, `harness.<covariate>.lines` (nested and forced) and `.placebo`, with the attempt calls in `attempts.decision` and `attempts.probe_only.decision`; the entropy job's `entropy.harness.<feature>.lines` and `.placebo`, with its call in `entropy.verdict`.
- MCQ floor: `results/mcq_floor.json`, `regimes`.

### Fig. 10: Item difficulty does not transfer (§6.1, §6.2)

**Status:** drawn, as D9, without the blind LLM judge (its correlations are only in `docs/findings.md`); D3 (c) adds the 4B and 14B judges and the 14B's entropy, with CIs.

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

**Status:** drawn, as D8.

Per regime (test-like, mix/whole, R1 benchmark-first, R1 pair-uniform), three bars: the pair-rate oracle, the item oracle and the shipped model's B31. Annotate the gap (0.063 test-like) and the share the itemsig layer recovers at B31 (0.5%).

- `results/itemsig_eval.json`, `summary.regimes.<regime>.oracle`.
- Note the `testlike.item_oracle` divergence documented in findings. Use the script's `theta_map` refit, which is what the summary stores.

---

## Appendix figures

### Fig. A1: The level prior adapts under empirical Bayes (§5.3)

**Status:** drawn, as D7, with both panels.

The EB level centre by budget (0, 1, 3, 7, 31), one line per regime: public benchmark-first, public pair-uniform, test-like with no shift, level_mean -1.2, default and -2.0. Mark each regime's mean pair logit.

- `results/level_calibration.json`, `summary.eb_adaptation`.
- Optional second panel: the P1a regimes' EB centre by budget for both adaptive configurations, with each regime's realised level. Sources: `results/regime_sensitivity.json`, `summary.eb_traces["eb_ship | <regime>"]` and `["eb_adapt | <regime>"]` (`mu0`, `sigma_mu` by budget), and `summary.realised.<regime>.mean`. These traces come from fresh-seed runs at the current library and should be labelled apart from the level calibration's.

### Fig. A2: Split scope decides what dense runs can read (§2.4)

**Status:** not drawn. `results/official_baselines.json`, `r2`, holds each predictor's Brier on dense runs for one split scope only; the coverage shares and the scope comparison are only in `docs/findings.md`.

For dense matharena, real_webagents and researchcodebench, show two panels:

- the share of a pair's evaluation items that carry other subjects' acquired labels at B31, under split scope 'pair' and 'benchmark';
- the Predictor's B31 under each scope against the base-rate oracle.

- `results/official_baselines.json`, `r2.<benchmark>`. The coverage shares and the scope comparison are also in `docs/findings.md`, "One benchmark, every pair (R2)".

### Fig. A3: The empirical mean's ALC follows from base rates (§2.5)

**Status:** drawn, as D10.

A scatter of per-run empirical-mean ALC against the run's E[p(1-p)] over the 600 R1 runs. Draw the analytic line 0.025 + 1.2118 E[p(1-p)] and the exact expectation. Mark the organisers' leaderboard entry (0.1801) on the y axis, with the inverted E[p(1-p)] of about 0.128.

- `results/official_baselines.json`, `r1.per_run`, `r1.analytic`, `r1.leaderboard`.

### Fig. A4: hier ablations (§5.5)

**Status:** drawn, as D11.

A forest plot of each ablation and sensitivity minus hier's default on the primary R1 setting, with cluster and stratified SEs, and the Bonferroni bar for sixteen comparisons (2.95 SEs).

- `results/hier_eval.json`, `r1["benchmark/pair"].diffs`. The table is in `docs/findings.md`, "Ablations and sensitivities".

### Fig. A5: Pooling gain against subjects in the pool (legacy)

**Status:** not drawn. A legacy measurement whose numbers are only in `docs/findings.md`.

The gain over no pooling against the number of subjects (3 to 80), on log axes, with the N^0.7 guide. Label it as a legacy-replica measurement.

- `docs/findings.md`, "How fragile the pooling gain is" (`experiments/pool_robustness.py`).

### Fig. A6: The shipped level at the feedback's reading, P1a (§5.3)

**Status:** drawn, as D4.

Draw a forest plot with one panel per regime, ordered by realised level: TUNED, AUDIT, MIXTURE, READING, FLAT, then R1 benchmark-first and R1 pair-uniform. Head each panel with the regime's realised mean and sd.

- **Rows:** one per alternative, plotting X minus the shipped configuration with its cluster-SE bar.
- **Rules:** a vertical rule at 0 in every panel. In READING and AUDIT, add the decision rule's bar at -0.002. In TUNED, MIXTURE and FLAT, add the +0.002 loss bound. In the two public panels, add the +0.001 loss bound.
- **Second marker:** the parent-level mean, drawn hollow.
- **The point:** no alternative crosses -0.002 in READING or AUDIT, the neighbouring configuration wins only in TUNED, and the calibrated smoothed mean is level with the shipped model only in TUNED.
- **Caption:** say that every regime redraws the catalogue that chose the level. Say that READING and AUDIT realised -0.63 and -0.93, milder than targeted.

Sources, all in `results/regime_sensitivity.json`:

- `summary.vs_ship.<regime>.<config>.exact`: `D`, `cluster_se`, `U95` and `PL`;
- `summary.vs_ship.<regime>.<config>.table.parent_level`: `mean`, `se` and `range`;
- `summary.realised.<regime>`: `mean` and `sd`;
- `rule.thresholds`: the bars.

The SHIP ALC per regime is `summary.configs.<regime>.ship.ALC`.

---

## Open questions for the figures

- **Resolved:** run 2's per-pair rows (`results/formative_feedback.json`), the shipped hier's single-run sds by budget and its public per-budget differences (`results/ship_confirm.json`), and the palette and size (see "Style" under the conventions).
- **Code of the plotted rows.** Most shipped-model series come from rows of the run-2 archive's library (old multiple-choice floor, solver before the floored-fit fix); the head of "Drawn figures" says which figures, and which use the current library. The corrected floor and the fix move ALC by at most 0.0005 (draft §5.4).
- **matplotlib is not a declared dependency.** `tests/test_report_figures.py` skips without it; adding a `report` extra to `pyproject.toml` (matplotlib 3.9) would make the figures reproducible from a clean install. The committed bytes are tied to matplotlib 3.9.2; on another version the test still checks inputs, script and buildability, not bytes.
- **Not yet drawn:** Fig. 7 and Fig. 2's right panel (their sources are in `results/`; a company grouping for the shipped configuration against the legacy Predictor is now stored in `results/pooling_decomposition.json`, `formative.<scope>.comparisons["ship - legacy"].by_companions`), and Figs. 1, 2's left panel, 4, A2 and A5 (no data, or data outside `results/`).
- **Placed in the draft:** all twelve drawn figures ("Placement in the draft", above).
