# Review of draft v0: "Calibrating a hierarchical response model for a benchmark-level cold start"

Internal review, written as a critical NeurIPS 2026 Evaluations & Datasets reviewer would read the report. The competition page says reports are assessed on criteria adapted from the E&D guidelines: technical soundness, clarity, reproducibility and the quality of the experimental analysis first, novelty second. Date 2026-09-27.

**What was read.**
- `docs/report/draft.md` (754 lines, written against commit `f7e7d87`) and `docs/report/figures.md`.
- `docs/findings.md`, `docs/protocol.md` (for the data counts), and the competition page text.
- Every `results/*.json` file the draft cites.
- The stored per-run rows in `data/subject_side_rows/` (gitignored), used to check numbers the draft could not quote.

Nothing in the draft or the code was changed, and nothing was committed.

**Conventions.**
- Numbers marked **[R]** were computed by the reviewer from stored rows (section 7 says how). They reproduce the recorded summaries exactly where the two overlap. They carry run SEs only and still need a committed script with cluster SEs before the report can quote them.
- "F§" cites a `docs/findings.md` section, as in the draft.

---

## 1 What the report claims

The report describes a competition entry for PAIEC. Each prediction is P(correct) for an (AI system, benchmark item) pair, given 0 to 31 revealed labels and scored by Brier ALC, on benchmarks never seen in training. It claims four contributions:

1. **A replica of the official streaming evaluator.** It shows that an earlier replica measured the wrong information set, and that the conclusion it drove (pooled item difficulty is the main lever, worth about 0.020) shrinks to about 0.0007 under the verified protocol at formative size.
2. **A hierarchical Bayesian item-response predictor (`hier`),** refitted from `labeled` alone at every checkpoint, with exact integration of item residuals and an INLA-style read of the target.
3. **A calibration of the benchmark-level prior to the hidden test.** It uses test-like runs built from public data to reproduce the first formative feedback, and changes three global hyperparameters under a public guard. The headline is -0.043 ALC against the first submission's predictor on those runs.
4. **An acceptance harness with a gate table.** The table maps a covariate's honest correlation with item difficulty to its expected ALC gain. It comes with a catalogue of negative results for item-side signals: TF-IDF, embeddings, LLM judge, a similarity layer, known-sign cues, and subject-side priors.

## 2 Overall assessment

The engineering and the internal discipline are unusually good for a competition report:
- a verified protocol replica;
- paired comparisons with three kinds of SE;
- nested leave-one-parent-out selection;
- results files that record their commands and code digests;
- negative results kept as negative.

The number audit bears this out. Of the roughly 150 numbers checked, all but a handful match their cited sources (section 6).

The weakness is in what the headline claim rests on:
- **The headline is self-referential.** The main gain is measured in a regime the authors built to reproduce the failure of the predictor it is compared against. The regime's size-setting knob (the date shift) is synthetic, and the gain falls from 0.042 to 0.014 without it.
- **The shipped configuration is not the one the report confirms.** Its own held-out and guard numbers are not in the report; it quotes a neighbouring configuration instead.
- **The feedback reading has changed.** The level reading the regime is built on conflicts with the audit that changed what shipped.
- **The platform does not corroborate the gain's size.** The only independent platform observation (formative run 2, ALC 0.1926) sits at about the 80th percentile of the tuned regime and the 33rd percentile of plain public runs [R]. It is also above the organisers' entry on its own leaderboard reading, though on a different draw.

None of this is fatal. Most of it can be fixed with numbers the repository already holds (section 7). But as written, the abstract overstates what is known.

| criterion | now | after the P0 revisions |
|---|---|---|
| technical soundness | fair (2/4) | good (3/4) |
| clarity | fair (2/4): dense, jargon-heavy, the shipped-config story told three times | good with a glossary and trimming |
| reproducibility | good infrastructure, real gaps: unpinned data, scratch studies, undisclosed rater | good (3/4) once pinned and released |
| quality of experimental analysis | good within the regime; missing baselines and regime sensitivity | good to excellent |
| significance to the competition community | moderate to high: the protocol analysis and the negative catalogue are useful to others | |

**Recommendation:** major revision. Worth a poster once the P0 items are done. An oral would need the P1 baselines and regime sensitivities.

## 3 Strengths

- **S1. Protocol care.** The replica is checked decision for decision against the organisers' client. Platform-like argument copies with 16 simulated workers give bit-identical per-pair Brier. The empirical-mean ALC agrees with its analytic expectation to -0.0005 ± 0.0006 once split and stream order are salted (§2.5). Reporting how the first replica misled the research (§2.3) is valuable to other teams.
- **S2. Statistical hygiene.** Comparisons are paired on identical runs and checkpoints. Differences carry run, pair-cluster and stratified SEs, and the report says what none of them covers (between-benchmark variation). It gives Bonferroni bars for the ablation table, leave-one-parent-out ranges, and selection versus confirmation halves.
- **S3. Nested selection is used where it matters.** It is applied in the itemsig, known-sign and subject-side studies, with the selection redone inside every bootstrap resample (§4.2, §6.3).
- **S4. Negative results are first-class.** Each one has its script, its results file and a "what was cut" note in the findings. The item-oracle gap (0.063 Brier on test-like runs) frames why item-side ideas fail.
- **S5. The inference is careful and the reasons are stated.** Examples: why PQL fails with item variance near 9 and one label per item, with a pinned test; why the Laplace mode under-reacts to a pair's first labels. Both are backed by tests.
- **S6. Provenance.** The results files record commands, wall times and code digests (`passes`), and `--summarise` rebuilds summaries from stored rows. The reviewer used these rows to recompute the shipped config's missing numbers in minutes. That is the best evidence that the infrastructure works.
- **S7. The conduct rule is taken seriously.** Feedback sets only global hyperparameters, and nothing is keyed on anonymous IDs (§3.4).

## 4 Major weaknesses

### W1. The headline gain is a property of a self-built regime, and the platform does not corroborate its size

The abstract and §1.4 lead with -0.043 ALC against the legacy Predictor on test-like runs. Four things qualify it.

- **The regime was tuned to that predictor's failure.** Its defaults are the grid point closest to run 1's B0 and B1 *for the legacy Predictor*: in sample, with the synthetic date shift chosen because it reproduces that prior's linear date term (§4.1, `testlike_check.json` notes). Measuring against that same predictor partly measures the tuning.
- **The size is set by a synthetic knob.** Without the date shift the gain is 0.014, not 0.042. §5.3's sensitivity table shows this for the neighbouring config; for the shipped config itself it is -0.0144 on the same 100 runs [R]. The date shift is the one knob that sets the size of the result, and nothing public can check it (§6.6 says so).
- **The better regime-internal comparator is the smoothed mean.** On the selection half, smoothed Beta(2,2) already beats the legacy Predictor by 0.025 (0.1788 against 0.2039). The shipped hier minus smoothed is about -0.018. That, or a level-calibrated smoothed mean (W4), is the number that says what the modelling adds.
- **The platform check is ambiguous.** Formative run 2 of the shipped archive scored ALC 0.1926 [derived]. Against the shipped model's own single-run distributions [R]:

| regime (shipped model) | mean ALC | single-run sd | run 2's z | share of runs scoring ≥ 0.1926 | run 2's per-budget z, B0..B31 |
|---|---|---|---|---|---|
| test-like, primary (300 runs) | 0.1658 | 0.0318 | +0.84 | 0.20 | +0.98 +0.15 +0.86 +0.89 +0.98 +1.40 |
| test-like, no date shift (100) | 0.1585 | 0.0335 | +1.02 | 0.18 | +1.00 +0.45 +0.89 +1.11 +1.20 +1.67 |
| public R1, benchmark-first (150) | 0.2051 | 0.0285 | -0.44 | 0.67 | +0.06 -0.78 -0.49 -0.46 -0.27 +0.23 |
| public R1, pair-uniform (100) | 0.2020 | 0.0261 | -0.36 | 0.68 | -0.24 -0.67 -0.37 -0.32 -0.13 +0.50 |

  One run cannot discriminate the regimes. But run 2's profile is at least as close to plain public runs as to the tuned regime: every budget is within 0.8 sd of public R1, while B31 is 1.4 sd above the test-like mean. That is what one would expect if the hidden levels are more central than the regime assumes (W3).

**What the abstract should do instead:**
- give the gain as a range across regimes, labelled regime-conditional: for the shipped config, 0.014 without the date shift to 0.044 on mix/whole (W2); for the neighbouring config, up to 0.052 at level_mean -2.0;
- report the shipped model against a regime-neutral baseline;
- place run 2 in these distributions;
- drop or qualify "its budget-0 Brier fell from 0.359 to 0.237". The two runs are different draws with different subjects and benchmarks, so the drop is not a measured effect of the change.

### W2. The shipped configuration's own held-out and guard numbers are missing, and the report quotes a neighbour instead

Two things about the shipped config (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5) are not in the report:

- **Its selection-half number is effectively in sample.** The -0.0431 is from the selection half, and the audit that picked this config looked at the selection half.
- **Its held-out numbers belong to another config.** The abstract's "confirms at 0.042 on held-out runs" and §5.3's -0.0418 are for the mu0 -3.0 / attr_scale 0.25 config. The audit's "gives up about 0.002" and "gains about 0.0024 on public runs" are approximate, and `results/` does not hold them. The shipped config appears in `level_calibration.json` only in `selection` and in a 40-run public `screen` (-0.0026).

The stored rows give the missing numbers directly [R] (run SEs):

| shipped config minus legacy Predictor | runs | shipped ALC | legacy ALC | difference |
|---|---|---|---|---|
| test-like, selection half (reproduces `summary.selection`) | 0-99 | 0.1608 | 0.2039 | -0.0431 ± 0.0019 |
| test-like, confirmation half | 100-199 | 0.1677 | 0.2073 | **-0.0396 ± 0.0019** |
| test-like, never used by the level calibration | 200-299 | 0.1687 | 0.2107 | **-0.0419 ± 0.0017** |
| test-like, all | 0-299 | 0.1658 | 0.2073 | -0.0415 ± 0.0011 |
| public R1, benchmark-first, split scope pair | 0-99 | 0.2053 | 0.2068 | **-0.0015 ± 0.0010** |
| public R1, benchmark-first | 0-149 | | | -0.0017 ± 0.0008 |
| public R1, pair-uniform, split scope pair | 0-99 | 0.2020 | 0.2038 | **-0.0018 ± 0.0010** |
| test-like, groups merged at random and wholes (seed 3) | 0-99 | 0.1694 | 0.2129 | -0.0436 (difference of means) |
| test-like, no date shift (seed 3) | 0-99 | 0.1585 | 0.1729 | -0.0144 (difference of means) |

In the last two rows the Predictor's ALC is `level_calibration.json` `summary.sensitivity`. The runs were matched pair for pair through `raw['meta|<regime>|<i>']`: 100 of 100 match in both regimes. The level-mean sensitivities (-1.2, -2.0) were never scored for the shipped config.

What these rows show:
- **The audit's approximate figures are right.** The shipped config beats the legacy Predictor on public runs. Its margin over the -3.0/0.25 config on the same 100 runs is 0.0023 (benchmark-first) and 0.0025 (pair-uniform), which confirms the "about 0.0024".
- **The cost in the tuned regime is 0.0022 on the confirmation half but 0.0030 on the selection half** (0.1608 against 0.1578). "About 0.002" (§3.4 item 2, §5.3) should give both.
- **Runs 200-299 are the cleanest held-out estimate the repository can offer.** They were scored for the shipped model by the itemsig and subject-side studies but never used to choose the level.

Commit a script (for example `experiments/ship_confirm.py` writing `results/ship_confirm.json`) that computes these rows with (parent, subject) cluster and stratified SEs and per-parent ranges. Record the result in findings, and quote it in the abstract, §1.4 and §5.3 in place of the neighbour's numbers.

### W3. The level reading behind the regime conflicts with the audit, and the regime does not realise its own target

- **The regime's reading.** Its default level (mean -1.6, sd 1.5) reads every run-1 pair's B31 through the *lower* root of p(1-p).
- **The audit's reading.** The audit that changed the shipped config (§3.4, F§ "What actually shipped, after the audit") says p6 and p8 are more plausibly high-rate and p9 is ambiguous. The reviewer recomputed the reading [R]:

| reading of run 1's nine pairs | mean pair logit | sd |
|---|---|---|
| all lower roots (the regime's default) | -1.64 | 1.49 |
| p6 and p8 at upper roots (the audit's reading) | -1.08 | 1.99 |
| p6, p8 and p9 at upper roots | -0.98 | 2.05 |

- **The realised level differs from the target.** The regime realises a pair-logit mean of -1.29 (sd 1.70) on the 300 check runs (`testlike_check.json`, `summary.check.default.regime.pair_logit`), not the -1.6 that §4.1 states.
- **The shipped choice was never tested where its rationale applies.** The config was moved to the milder setting "because hidden levels look spread both ways". Yet no regime with that spread was scored: `level_sd` is on findings' list of cut knobs, and the sensitivities move only the mean (-1.2, -2.0).

Score the shipped config, the -3.0/0.25 config, the EB level, a calibrated smoothed mean and the legacy Predictor at `level_mean` about -1.1 and `level_sd` 2.0, and under a two-component mixture. This is the one experiment that tests the audit's reasoning. It is also where the EB level (§5.3), which adapts both ways, should beat any fixed shift. If it does, the reason given for not shipping EB (about 70 lines outside the library, a 0.8 s worst call) needs revisiting.

### W4. Baselines are missing, so calibration cannot be separated from modelling

- **A level-calibrated smoothed mean.** Choose a Beta prior mean and strength under the same guard and selection rule. This is the minimal model that "predicts lower". §5.3 claims "calibration is worth far more than the modelling", but a moved level is only shown for hier and the legacy Predictor. With the calibrated smoothed mean, the report could state how much of 0.042 needs hier at all. The "twenty times" comparison in §5.3 (0.042 in the tuned regime against 0.002 on public runs) compares gains in two different regimes and should go.
- **The organisers' reference predictors.** BLE (an LLM predictor) and the empirical mean with BLE acquisition are in the baseline repository and never appear. If BLE cannot run offline, say why. Findings reads the best leaderboard entry (0.1172) as either per-item signal or a run with lower base-rate variance: it beats the base-rate oracle expected for such runs. The report should say what BLE does on public and test-like runs.
- **A plain 1PL (Rasch) model with a pooled level and no attributes,** to isolate the attribute prior's share of the hier gain at B0 and B1.
- Optionally, **amortized calibration** [Truong et al. 2025] (organisers' co-authored work), or at least a sentence on why it was not run.

### W5. Contribution 1 conflates the information set with run size

The abstract attributes the fall from about 0.020 to about 0.0007 to measuring "the wrong information set". But the two numbers differ in more than that:
- **Different setups.** The legacy ladder scored all 221 pairs in one session, with state carried across pairs. The 0.0007 is at formative size (about two pairs per benchmark).
- **The official protocol still rewards pooling on dense runs.** On R2 under per-pair splits, the legacy Predictor beats smoothed by 0.017 to 0.039, "mostly by reading the target item's own labels from other subjects" (§2.4).

The drop may therefore be mostly run size, or split scope, not the information set. Either decompose it on the official replica (pooled difficulty on and off, via `WARMUP`, on dense R2 and on formative-size runs, under both split scopes) or rephrase the claim to "at formative size under the verified protocol".

### W6. The acceptance harness's claims need tightening

- **"Pre-registered" is not supported.** The abstract, §1.3 and §4.4 use the word, but no record fixes the gate before the studies. The gates live in the scripts' plan blocks, committed in `f7e7d87` together with the results. The harness rows were built at `bd0be67`, after the itemsig study. Say "fixed before each study's runs, in its plan block", or cite commits that show it.
- **The acceptance check rests on an unreleased study.** The harness validates against "the in-sample oracle that an earlier study scored" (-0.0437 / -0.0557). That study is `heads.py`, which exists only in session scratch (§6.4). Release it, or present the reproduction as a self-consistency check without an external target.
- **The r = 0.3 pass is marginal.** The transferred line at honest r = 0.3 scores -0.00238 against a bar of -0.002, with draw sd 0.0005 and worst parent -0.0019 (`harness_thresholds.json`). A covariate with true honest r = 0.3 passes only about three times in four. "The bar is an honest r of about 0.3" should read "about 0.3 to 0.35".
- **The gate uses the thinnest regime.** The test-like regime has the least item structure: the oracle takes 46% of the pair-rate Brier there against 55% on public runs. At r = 0.2 the transferred line gains -0.0019 on mix/whole and -0.0023 to -0.0029 on public runs, but only -0.0009 on test-like runs. So the gate leans toward false negatives. Say this, and report the table's pass row under mix/whole as well.
- **The correlation scales are mixed.** The gate is in honest r: correlation with a difficulty fitted on other subject folds, which is noisier than full-sample Rasch difficulty. §6.1, §6.2 and §6.9 instead compare correlations with *full-sample* Rasch difficulty ("at most 0.23", "0.48 on mathematics") against that bar. The bias runs in the authors' favour, since a covariate's correlation with honest z is lower. Still, state the scale, give z_honest's reliability, and express §6.9's comparison on one scale.
- **The gate cannot see B0 by construction, and §6.9 overstates its reach at B1.** With a transferred slope a centred covariate *does* act at B1: the acceptance oracle's B1 Brier difference is -0.044. Only a per-pair slope is zero there. §6.9's "Only the level and subject priors act there" is therefore wrong for B1.
- **The harness has no findings section.** Contribution 4 has no section in `docs/findings.md`; its numbers exist only in `results/harness_thresholds.json`. That breaks the repository's own rule that findings maps every number to its script.

### W7. Some negative results are underpowered or measured on a different footing, yet the abstract claims them all

- **The LLM judge.** n = 45 per benchmark. The CIs include r = 0.3 on multi_swebench [-0.17, +0.41] and real_webagents [-0.09, +0.47]. "The judge clears that on one of four" is a statement about point estimates. The local Qwen3-4B judge (step 5) could rate several hundred items per benchmark at no API cost. Do that before concluding, or phrase the result as inconclusive outside mathematics.
- **TF-IDF.** The leave-one-benchmark-out correlations have no CIs. The embedding study has group-bootstrap CIs in `emb_transfer.json`; report them.
- **Mixed footing.** §6's introduction says each idea "was measured leave-one-parent-out against the model that shipped at the time". That is false for TF-IDF and the LLM ratings, which are data-level correlations, and for §6.7, which ran on the legacy replica against the legacy Predictor.
- **Pending studies.** Meta-heads is TODO and five step-5 studies are pending, yet the abstract says "every item-side signal we tried falls short". Scope the sentence to the studies completed, or finish them first.

### W8. Reproducibility gaps a reader or the organisers cannot close

- **The data is not pinned.** `paiec/fetch.py` calls `hf_hub_download` without a `revision`, and the competition page says the collections may grow. Record the measurement-db commit SHA used for every result, and pin it in `fetch.py`.
- **The baseline repository is not pinned.** §8.2 clones `aims-foundations/paiec_baseline` at HEAD. The replica mirrors a specific `streaming_ingestion.py`, so pin its commit.
- **Some cited numbers exist only in session scratch:**
  - the meta-heads study (§6.4, and the harness acceptance target);
  - the inventory classification behind §2.1's class shares;
  - the "step2b audit" behind §3.4.

  Release them, or remove the numbers.
- **The rater is undisclosed.** The blind ratings are hard-coded in `experiments/llm_rating/ratings_main.py` with no model, prompt, date or blinding protocol. The competition requires disclosing pretrained models. The ratings also cannot be re-derived without the prompt.
- **Row files are not released.** The stored rows are gitignored: `data/subject_side_rows` (11 MB) and `data/harness_rows` (43 MB). The itemsig rows are not in `data/` at all, because they were written to a scratch `--rows` directory. Regenerating any of them takes CPU hours. Publish them with the code release (a Hugging Face dataset or Zenodo). They are small, and they let a reader check every table in minutes, as this review did. Mind the dataset's CC-BY-SA terms for derived data.
- **Code and numbers are out of step.**
  - The §5.4 numbers, formative run 2 and `dist/paiec.zip` all use the old multiple-choice floor (`mcq_floor.json` provenance: commit `bd0be67`, `library_floor: old`). §5.5 describes the corrected floor as adopted.
  - The working tree has uncommitted changes to `paiec/hier.py` and `tests/test_hier.py` from another lane.
  - Freeze a tag, rebuild the archive, and add a table mapping each result to the commit or archive that produced it.
  - If the final selected submission differs from the archive that produced run 2, re-run §5.4 for it.

### W9. Conduct and disclosure need explicit statements

- **The inventory scan.** `experiments/inventory_scan.py` cloned the repositories of the organisers' 161 inventory benchmarks and inspected their result files. The test pool is drawn from that inventory, and the rules say competition-specific training and curation must use the public pool. §8.6 should state three things: what was read (file listings or contents), that nothing from it entered any model or selection, and that the offline bank was dropped partly for this reason. §6.7 currently gives only the feasibility count.
- **Feedback reads.** The level was set from run 1's aggregates, and the audit then read run 1 per pair. §5.1 says run 2 "will be read per pair ... before refitting the level distribution once". State how many formative submissions informed choices, and pre-commit how run 2 (and any run 3) will be used. This guards against adaptive over-fitting to a handful of noisy draws. It is allowed use, but reviewers will want the accounting.

## 5 Minor weaknesses and clarity

- **m1. Jargon and naming.** Terms include parent, pseudo-benchmark, appearance, R1/R2, r1b/r1p, mix/whole, target-LOBO, strict run-LOBO, guard and "± run / cluster / stratified". Add a notation table in §4. The same model is called "legacy Predictor", "the Predictor", "the first submission's predictor" and, in findings, "the shipped Predictor". Use one name throughout (for example LegacyP), and "hier-ship" for the shipped model.
- **m2. The shipped-config story is told three times** (§3.4, §5.3, A.3), with slightly different numbers. Tell it once in §5.3, with the W2 table.
- **m3. "Few subjects cross benchmarks"** (§2.1: 22 of 287) sits beside theta_s being keyed on canonical name. By name, 117 of 220 pairs link across benchmarks (F§ step-2). Explain subject versus canonical name.
- **m4. "We report both scopes wherever the difference can matter"** (§2.4) is contradicted by §7: most studies, including the level calibration's public guard, ran split scope 'pair' only.
- **m5. Three statements lack their caveats:**
  - §3.3's tau2_res = -0.002 and link weight 0.0018 omit findings' caveat that the estimator is biased downward and the identity conclusion is provisional.
  - §3.2's "for a single pair it is the exact marginal" holds only when the pair's labels load on x through a'x alone. With labelled group effects it is an approximation. Say so.
  - §3.4's "It is also the only use of one noisy run that generalises" is an assertion; cut it or argue it.
- **m6. Mixed run sets.** §6.3's public benchmark-first base is 0.2057 (runs 0-199), while §5.4 gives 0.2051 (runs 0-149). Label the run sets.
- **m7. Incomplete tables.** §5.5 shows 12 rows but speaks of sixteen comparisons; the hard floor and sigma_delta x0.5 are missing. §5.2's baseline table drops B3 and B15. Use one SE format across tables.
- **m8. No figures exist yet.** E&D readers expect at least:
  - learning curves (Brier by budget) for the shipped model in each regime, with the platform runs overlaid and ±1 single-run sd bands (the sds are now available, section 7);
  - the level-calibration surface;
  - the gate curve.

  `figures.md` lists three figures as lacking data. Two of those gaps (run 2 against the regimes' single-run sds, and the shipped config's public differences) can be filled from the stored rows now.
- **m9. Length.** About 8,600 words of prose, plus tables and appendices, is more than a reader needs for the main story. Trim candidates: most of §2.3 (keep the three bullets), §5.5 into the appendix, §6.7 into one paragraph, and A.3 and A.4 merged into the main tables.
- **m10. Literature.** Add:
  - label-shift and prior-adjustment work, which is what the level calibration and the EB level are: Saerens et al. 2002, Lipton et al. 2018;
  - IRT for NLP benchmarks (Lalor et al.; Rodriguez et al. 2021) and explanatory IRT / LLTM (Fischer 1973), for item_features group effects;
  - adaptive-quadrature GLMM inference (for example Pinheiro and Bates), next to Breslow and Clayton.

  Also: check Kwa et al. 2025's title (the reviewer recalls "Measuring AI Ability to Complete Long Tasks"), and check the 2025-26 arXiv IDs and venues (Ge et al., Krsteski and Meyer, Lugoloobi et al., Cencerrado et al.) against the published versions.

## 6 Number audit

Every number in the draft was checked against its cited source where one exists. Status "OK" means it matches to the reported rounding.

| draft location | claim | source checked | status |
|---|---|---|---|
| §1.4, §5.1 | run 1 ALC 0.2113 and budgets | F§ first feedback; ALC formula (0.21131) | OK |
| §1.4, §5.1, §5.4 | run 2 ALC 0.1926 (derived) | ALC formula on the rounded budgets (0.1926) | OK, ±0.0005 from rounding |
| §5.1 | "0.5 at B0 and B1 would give 0.1996" | recomputed (0.19964) | OK |
| §5.1 | B0 excess over B31 0.054; B3-B31 0.03 to 0.045 above the test-like means | recomputed | OK |
| §1.4, §5.3, A.3 | shipped config selection half: ALC 0.1608, -0.0431 ± 0.0019 / 0.0032 / 0.0028; per parent -0.0336 to -0.0483 | `level_calibration.json` `summary.selection.configs['hier G mu0=-2.50 sm=2.50 as=0.50']` | OK |
| §5.3, A.3 | guarded configs, confirmation and public costs | F§ "The public guard decides" | OK |
| §5.3 | sensitivity table | F§ "Sensitivity to the regime" | OK |
| §5.4 | all 36 budget cells and 5 ALCs ± SE | `subject_side.json` `summary.regimes.*.ship` | OK |
| §4.5 | every gate-table cell; honest-oracle row; within-pair r values; draw sd 0.0005 | `harness_thresholds.json` `thresholds.honest`, `acceptance.oracle_honest`, `tables.honest.*.meta` | OK |
| §4.5 | "r = 0 forced per-pair costs +0.0007 to +0.0010" | same file: +0.00065 to +0.00105 across regimes and start budgets | OK (range slightly wider) |
| §4.5 | acceptance -0.0440 / -0.0558 against targets -0.0437 / -0.0557 | `acceptance.check` | OK; the target's source is not in the repository (W6) |
| §4.1 | B3-B31 z +0.09 to +0.50; R1 B0 z +8.5; ALC 0.2073 / 0.2075 / 0.2113 | `testlike_check.json` `summary.check.default.ppc`, `summary.r1` | OK |
| §4.1 | item oracle 46% against 55% | F§ "Against the item oracle" | OK |
| §7 | level-mean distances 0.63 / 0.37 / 0.46 | `testlike_check.json` `summary.level_identification` | OK (B0/B1 only; including B31 they are 0.63 / 0.62 / 0.83) |
| §2.3, §5.2 | official baselines; Predictor minus smoothed 0.0026 ± 0.0009 and 0.0069 ± 0.0016 | F§ R1; `official_baselines.json` (empirical mean 0.2526 ± 0.0014) | OK |
| §5.2, A.2 | hier against the Predictor, all four settings; "by company" table | F§ "Against the Predictor" | OK |
| §5.5 | ablations | F§ "Ablations and sensitivities" | OK; 2 of 16 rows omitted |
| §5.5 | MCQ floor -0.00026 ± 0.00003 / 0.00008; public -0.00027 to -0.00045 | `mcq_floor.json` | OK |
| §5.6 | latencies | F§ "Calibrating", Latency | OK |
| §6.1 | TF-IDF LOBO correlations; embedding ridge -0.16 / -0.03 / -0.23 / -0.02; kNN -0.08 to +0.16; within 0.66 / 0.27 / 0.38 / 0.51 | F§; `emb_transfer.json` (`lobo.emb_ridge_centred`, `lobo.emb_knn_raw`, `within.emb_ridge`) | OK |
| §6.2 | ratings table and control | F§ "Language-model difficulty judgement" | OK (not rerun) |
| §6.3 | itemsig nested and in-sample lines | `itemsig_eval.json` (`nested_within`: -1.7e-5, cluster 8.4e-5) | OK |
| §6.5, A.4 | sign table; stated_size -0.0022 ± 0.0005, placebo +0.0008 | F§ "Item covariates with a known sign" | OK |
| §6.6 | nested selection table | F§ "Subject side", Nested selection | OK |
| §3.3 | Hyper values | `submission/prior.json` `hyper` | OK |
| §2.5, §8.2 | 34 and 243 test functions | AST count at HEAD: 34 and 243 (247 in the working tree, because of uncommitted `test_hier.py` additions) | OK at HEAD |
| §2.1 | 571,921 responses; 221 pairs; 225,843 responses; 22 of 287 subjects; repeat shares | `docs/protocol.md` (not in findings or results) | OK against protocol.md |
| §8.4 | wall times | F§; embedding probe 232 s and 480 MB match `emb_transfer.json` | OK, except the harness "25 + 24 min", which is not in findings (results record 1,453 s for one pass) |
| §8.6 | commit trailers: Claude Opus 5 (2), Claude Opus 5.5 (5) | `git log` | OK |

**Discrepancies to fix:**

1. **§3.4 item 2 and §5.3.** "Gives up about 0.002 in the tuned regime": it is 0.0030 on the selection half (0.1608 against 0.1578) and 0.0022 on the confirmation half [R]. Give both.
2. **§6.9, item 1.** "B0 and B1 ... were the largest errors (0.237 and 0.195)": run 2's B3 is 0.196, above B1.
3. **§6.9, item 1.** "Only the level and subject priors act there": a transferred-slope item covariate acts at B1 (W6).
4. **§7.** "35 of 2,425 stratum appearances": findings says 35 of 2,425 *pair* appearances, all of them strata. Strata are 689 appearances.
5. **§1.2.** "Mixing the training mean into the prior made ALC worse" is a legacy-replica result, left unmarked despite the draft's own convention. Under the official protocol it is contradicted:
   - §5.5's "level centre 0 instead of the mean" costs +0.0011;
   - step 2 found a Gaussian level at the leave-one-out mean beats Beta(2,2) by 0.0028.
6. **§3.2.** "Item variance near 8.7": the shipped sigma_d² + sigma_g² = 2.671² + 1.542² = 9.51, and findings' 0.46 scale factor uses T ≈ 9.5. The 8.7 comes from the `hier.py` docstring, so say which fit it is.
7. **§4.1.** "Tilted toward ... mean -1.6, sd 1.5": the realised check-run pair logit is -1.29 (sd 1.70). Report the realised value.
8. **§4.1.** "Only the grouping of items is new": the date shift also changes subjects' visible release and access dates.
9. **§2.4.** "The predictor's gain" is the legacy Predictor's. Name it; hier's R2 behaviour differs (researchcodebench, benchmark scope: +0.0040 against the Predictor).
10. **§2.1.** matharena's `competition` has 27 levels (item_signal) but 25 competitions (F§ "What transfers"; `emb_transfer.json` `info.groups`). Kangaroo is 336 items (F§ MCQ floor) or 338 (F§ itemcov). Reconcile both.
11. **§5.1.** "They drew different subjects" has no source in findings or results. Record it with the run 2 table, including any recurring IDs (Q1).
12. **§5.1.** "The per-pair matched estimate for the recommended configuration ... 0.178": in findings this estimate refers to the mu0 -3.0 / 0.25 config (it replaces that config's 0.167). Say so, and give the shipped config's own estimate (Q2).
13. **§1.2.** "At most 1,000 ... so it has 5 to 12 pairs": the maximum follows from the cap and the 80-item floor. The minimum of 5 is the replica's `sample_run` choice.
14. **§6 introduction.** "Each idea below was measured leave-one-parent-out against the model that shipped at the time" (W7).
15. **References.** Kwa et al. 2025's title (m10).

## 7 Reviewer's computations from stored rows

These come from files already in the repository or its gitignored data directory. No new runs were made.

**The shipped model's per-pair Brier by budget:**
- source: the `ship` arm of `data/subject_side_rows/{tl,tl_mix-whole,tl_no_shift,r1b,r1p}.jsonl`, the same runs as `experiments/subject_side.py`;
- a pair's ALC is the weighted sum of its six budgets, and a run's ALC is the mean over its pairs.

**The legacy Predictor's per-pair ALC:**
- test-like runs: `results/testlike_check.json` `raw['check|default|<i>']['rows'][*]['pred'][6]`. These are the seed-2 check runs that the level calibration reuses; the pair lists match the subject-side rows run for run (0 mismatches).
- public runs: `results/hier_eval.json` `raw['r1|<weighting>|pair|0|<i>|Predictor']['rows'][*][6]`.

**Consistency checks:**
- The test-like rows reproduce `level_calibration.json`'s selection half exactly (0.1608, 0.2039, -0.0431 ± 0.0019).
- The public rows reproduce the guard's Predictor ALCs exactly (0.2068 and 0.2038 on runs 0-99).
- For the two sensitivity regimes, the Predictor's ALC is taken from `level_calibration.json` `summary.sensitivity`, after checking that the 100 runs match pair for pair.

Results: the W2 table, the W1 run-2 table and the W3 level readings. The per-budget single-run sds of the shipped model, which `figures.md` lists as missing, are:

| regime | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
|---|---|---|---|---|---|---|---|
| test-like (300) | 0.0209 | 0.0385 | 0.0367 | 0.0354 | 0.0336 | 0.0318 | 0.0318 |
| test-like, mix/whole (200) | 0.0171 | 0.0400 | 0.0371 | 0.0357 | 0.0354 | 0.0327 | 0.0329 |
| test-like, no date shift (100) | 0.0423 | 0.0418 | 0.0384 | 0.0327 | 0.0307 | 0.0286 | 0.0335 |
| public R1, benchmark-first (150) | 0.0330 | 0.0373 | 0.0345 | 0.0286 | 0.0264 | 0.0252 | 0.0285 |
| public R1, pair-uniform (100) | 0.0297 | 0.0393 | 0.0297 | 0.0279 | 0.0257 | 0.0244 | 0.0261 |

These are sds of run means over runs that redraw pairs from one catalogue. That is the right spread against which to place a single platform run of similar size, but it does not cover pair sizes that differ from the platform's.

## 8 Questions for the authors

- **Q1. Do any IDs recur between runs 1 and 2?** The page says anonymous subject and benchmark IDs are permanent across submissions, and the split is fixed. If any (subject_id, benchmark_id) pair recurs, its evaluation items are the same up to the 1,000-item cut. That gives a genuinely paired platform comparison of the legacy Predictor and hier. Matching one's own runs on IDs is not identifying them.
- **Q2. What is the per-pair matched estimate on run 1 for the shipped config itself?** And what would the audit's per-pair reading predict for run 2's pairs?
- **Q3. W3's experiment.** Under the audit's level reading (mean about -1.1, sd about 2.0, or a mixture), how do the shipped config, -3.0/0.25, EB, a calibrated smoothed mean and the legacy Predictor rank?
- **Q4. Why 0.002 for the gate?** How big is the summative common subset expected to be, and how large a difference does it resolve?
- **Q5. What does the shipped config lose on a single-subject benchmark** (swe_rebench-like) against the legacy Predictor and against smoothed? Findings says this was not broken out, and single-subject benchmarks are excluded from test-like runs by default.
- **Q6. Is the read along the target's line exact** when labelled items carry group effects? What is the worst measured error of that case in the `hier.py` docstring?
- **Q7. How were the blind ratings produced?** Which model, which prompt, when, and how was the rater kept blind to benchmark identity and difficulty? Could it have seen measurement-db or the benchmarks' leaderboards?
- **Q8. Which archive (hash) produced run 2?** Will the selected final submission be rebuilt with the corrected floor, and re-validated at the same numbers?
- **Q9. How many formative submissions exist or are planned,** and which choices did each inform?
- **Q10. What evidence beyond the synthetic date shift supports attr_scale 0.5,** that is, attribute priors inflated for hidden subjects? Run 2's B0 (0.237) is at the public R1 mean (0.235, z +0.06) and 1 sd above the test-like mean.
- **Q11. What does the model do on image-only items?** About 30% of the inventory is image benchmarks. Is matharena's textless third a usable proxy for B0 and B1 behaviour on such items?
- **Q12. What is the worst-case per-call time on dense multi_swebench and matharena** (about 2,500 labeled entries at B31)? It is still unmeasured, and the per-call timeout is undisclosed.
- **Q13. Can a benchmark's level be predicted from its items' content?** That is the only item-side route to B0, where the headroom is. For example, a zero-shot absolute difficulty rating averaged over the benchmark's labeled items, calibrated on the five public levels. Was this considered, and is five points of calibration the reason not to try it?

## 9 Prioritised revisions

### P0: blocking for submission (soundness, honesty of claims, disclosure)

1. **Report the shipped configuration's own numbers.** Commit `experiments/ship_confirm.py` (reading the stored rows, as in section 7) to produce `results/ship_confirm.json`. It should report the shipped config against the legacy Predictor and against the smoothed mean on runs 0-99, 100-199, 200-299 and 0-299, and on both public weightings. Use run, cluster and stratified SEs, per-parent ranges and per-budget differences. Record it in findings and use it in the abstract, §1.4, §3.4 and §5.3. *Done when:* no sentence about the shipped model quotes the -3.0/0.25 config's numbers.
2. **Rewrite the abstract and headline around what is known.**
   - The gain against the legacy Predictor is regime-conditional: about 0.040 to 0.042 held out [R], 0.044 on mix/whole [R], and 0.014 without the date shift [R]. Across level means only the neighbouring config was scored (0.039 to 0.052); score the shipped one.
   - The platform's run 2 (0.1926) is +0.84 sd in the tuned regime and -0.44 sd in public runs.
   - Two formative runs on different draws do not measure an improvement.
   - Drop "fell from 0.359 to 0.237" as an effect, or label it as two different draws.
   - Scope "every item-side signal we tried" to completed studies.
3. **Record formative run 2 in findings:** per-pair rows, pair and benchmark counts, ECE, the archive hash, and ID overlap with run 1 (Q1). Then do the pooled per-pair reading of runs 1 and 2 that §5.1 promises, and state in advance which global hyperparameter it may change.
4. **Disclose:**
   - the rating model, prompt and protocol (Q7);
   - the inventory-scan statement (W9);
   - the feedback-use accounting (W9).
5. **Pin inputs:** the measurement-db revision SHA (in `fetch.py` and §8.2), the `paiec_baseline` commit, and the Python and library versions (already listed).
6. **Close the scratch provenance.** Move `heads.py`, the inventory classification and the step2b audit into `experiments/` with their outputs, or delete the numbers that depend on them: §2.1's class shares, §4.5's acceptance target and §3.4's audit figures.
7. **Fix the 15 discrepancies** listed in section 6.
8. **Freeze the code.** Tag a commit, rebuild `dist/paiec.zip` there, and add a table mapping each §5 number to the code or archive that produced it (old or corrected MCQ floor). Keep the other lane's uncommitted `hier.py` changes out of the reported numbers, or re-run §5.4 on them.

### P1: needed for a sound, complete analysis

9. **Add baselines:** a level-calibrated smoothed mean, selected with the same rule and guard; BLE, or a reason it cannot run; and a plain 1PL with a pooled level and no attributes (W4). Re-state "calibration is worth far more than the modelling" against them, and drop the "twenty times" comparison.
10. **Run the regime sensitivity at the audit's reading** (W3, Q3): level_mean about -1.1 with level_sd 2.0, and a two-component mixture. Include EB. If EB wins there, revisit shipping it.
11. **Decompose contribution 1** (W5): pooled difficulty on and off, under both split scopes, on dense and formative-size official runs. Or rephrase the claim.
12. **Tighten the gate:**
    - drop "pre-registered" or evidence it;
    - give the pass probability at r = 0.3 from the draw sd;
    - report the gate under mix/whole;
    - put every correlation on one scale, the honest z, and state its reliability;
    - correct the B1 statement;
    - add a harness section to findings (W6).
13. **Power the judge.** Run the local Qwen3-4B judge on several hundred items per benchmark, and give CIs for every transfer correlation, TF-IDF included (W7).
14. **Report between-benchmark variation** for the headline: the parent-level mean ± SE across the four parents (the harness already stores `parent_se`) next to the cluster SEs.
15. **Break out single-subject benchmarks** for the shipped config (Q5).
16. **Check a wider level prior.** sigma_mu 2.5 is on the edge of the scored grid; score 3.5 and 5 at the shipped mu0 and attr_scale.
17. **Release the row files** (W8), so section 7's checks and every table can be reproduced in minutes.

### P2: clarity, presentation, length

18. Add a glossary and notation table, one name per model and one SE format (m1, m7).
19. Tell the shipped-config story once (m2), and trim to the target length (m9).
20. Build figures from `results/` only through `tools/report_figures.py`. Priority: learning curves by regime with the platform runs and single-run sd bands (section 7), the level surface, and the gate curve (m8).
21. Complete the literature and verify the references (m10).
22. Make the small wording fixes: m3 to m6; §2.4's scope sentence; §3.2 and §3.3's caveats.

## 10 A possible abstract, for calibration of tone

> PAIEC asks for the probability that an AI system answers a benchmark item correctly, from visible attributes and 0 to 31 revealed labels, on benchmarks absent from training. We rebuilt the organisers' streaming evaluator and found that an earlier replica, which scored all pairs in one session with the target's own labels only, had made cross-subject pooling look like the main lever; at formative size under the verified protocol it is worth under 0.001 ALC. Our predictor is a hierarchical Bayesian item-response model refitted from the revealed labels at every checkpoint. Its main lever turned out to be the prior on a new benchmark's level: we chose three global hyperparameters on public-data runs built to resemble the first formative feedback, under a public guard. In that regime the shipped model improves on our first submission by 0.040 to 0.042 ALC on held-out runs (0.014 when a synthetic date shift is removed), at no cost on public runs; its one scored formative run (0.1926) is within one run's noise of both the tuned and the public regimes. A harness with a fixed gate rejects every item-side signal we could transfer from five public benchmarks: text features, embeddings, an LLM judge, a similarity layer and known-sign cues. We report each negative result with its script.
