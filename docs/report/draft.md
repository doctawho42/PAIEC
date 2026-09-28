# Calibrating a hierarchical response model for a benchmark-level cold start

Technical report for the Predictive AI Evaluation Competition (PAIEC), NeurIPS 2026. **Draft v0, 2026-09-27.**

Authors: TODO(team). Code: TODO(link to the public release). Report written against commit `f7e7d87` of branch `official-protocol`.

**Provenance convention.** Every number in this draft is taken from `docs/findings.md` (cited as F§ plus the section title), from `docs/protocol.md` (P§), or from a file in `results/`, and each table names the script that produces it. The few numbers that are derived here are marked "derived", with their inputs. Numbers that exist only in session scratch are not quoted; they are marked `TODO(record)` until the study is copied into `experiments/` and written into `docs/findings.md`. Placeholders for studies still running are marked `TODO(step5)`.

---

## Abstract

PAIEC asks for the probability that an AI system (a *subject*) answered a benchmark item correctly. The inputs are the subject's and the item's visible attributes plus 0 to 31 revealed labels, and the score is Brier ALC on benchmarks that never appear in the training data. We make four contributions.

1. **A protocol-faithful replica.** We rebuilt the organisers' streaming evaluator from their published client. The replica we had built before the client was public measured the wrong information set: pair-major scoring, and only the target subject's own labels. Under it, pooling item difficulty across subjects looked like the main lever (about 0.020 ALC). Under the verified protocol, on formative-sized runs, that lever is worth about 0.0007.
2. **A hierarchical Bayesian predictor.** It is refitted at every checkpoint from the shared `labeled` list alone, with a benchmark level, a subject standing, a pair deviation, item-feature groups and an integrated item residual.
3. **A calibration of the model's benchmark-level prior to the hidden test.** We built test-like runs from public data that reproduce the first formative feedback and chose three global hyperparameters on them, under a guard on public runs. Nothing is keyed to an anonymous test benchmark.
4. **One acceptance harness for every other idea.** It uses nested leave-one-parent-out selection and has a pre-registered gate of 0.002 ALC.

On the platform, our first submission scored ALC 0.2113. The shipped model scored 0.1926 on a different formative draw, and its budget-0 Brier fell from 0.359 to 0.237. On the test-like runs used for selection, the shipped model gains 0.043 ALC over the first submission's predictor (cluster SE 0.003); its guarded neighbour configuration confirms at 0.042 on held-out runs. Every item-side signal we tried falls short. That covers TF-IDF, neural embeddings, LLM difficulty judgements, a text-similarity residual layer and known-sign item cues. Each one either does not transfer to a held-out benchmark or stays below the gate. The harness's gate table puts the bar at an honest correlation of about 0.3 with item difficulty, and nothing transferable in the public data reaches it.

---

## 1 Introduction

### 1.1 The task

For each subject-item pair the program returns P(correct) from two dictionaries: the subject (eight string fields, such as normalized_name, provider, release_date, harness and reasoning_effort) and the item (item_content, item_features, interactors and an anonymous benchmark_id). It also receives `labeled`, the outcomes revealed so far [P§ Interfaces]. Items of each subject-benchmark pair with at least 80 distinct items are split 50/50 into acquisition and evaluation pools. Labels are revealed from the acquisition pool, and Brier is measured on the evaluation pool at budgets 0, 1, 3, 7, 15 and 31, then averaged equally over pairs:

    ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31

The k-th acquired label counts toward every budget it survives into, so the first label is worth 0.9 of a budget's weight and labels 16 to 31 are worth 0.1 each [P§ Scoring]. B0 and B1 alone carry 0.3 of the weight.

### 1.2 What makes it hard

**The level of a new benchmark is not in the training data.** The five public benchmarks' levels average -0.56 on the pair-accuracy logit scale, with sd 0.90 (F§ "The step-2 analyses behind the model", `experiments/hier_design/levels.py`). Mixing the training mean into the prior made ALC worse (F§ "What transfers between benchmarks"). The first formative feedback then put the hidden pairs roughly 0.9 logit below the public centre (§5.1).

**Per-item structure is most of the headroom and does not transfer.** On all 221 public pairs, the Brier score falls from 0.25 (constant 0.5) to 0.176 when the pair's true accuracy is known, and to 0.0433 when each item's true probability is known (F§ "How much there is to win", `experiments/ceilings.py`). But item difficulty predicted from item text has a leave-one-benchmark-out correlation between -0.22 and 0.23 (§6.1).

**Runs are small.** A formative run holds at most 1,000 subject-item pairs across both pools, so it has 5 to 12 pairs [P§ Runs]. The one formative run whose per-pair feedback we recorded held 9 pairs over 7 benchmarks, and 5 of those pairs were alone on their benchmark (F§ "Against the first real formative feedback"). Pooling across subjects therefore has little to pool.

**No state survives between budgets.** Evaluation workers are recreated at every checkpoint, so `predict` has to be a pure function of `(input, labeled)` [P§ Flow].

### 1.3 Contributions

1. A replica of the official streaming evaluator (`paiec/official.py`), checked decision for decision against the organisers' client. §2 also shows what our first replica got wrong and which conclusions that reversed.
2. `paiec/hier.py`, a hierarchical Bayesian item-response predictor fitted from `labeled` alone. Its predictive integrates the item residual exactly and reads the target's linear predictor along its exact posterior line (§3).
3. `paiec/testlike.py`, test-like runs built from public data (pseudo-benchmarks, about one pair per benchmark, tilted base rates). They reproduce the first formative feedback at every budget, and we use them to calibrate the level prior under a public guard. The rule we kept: *global hyperparameters only, never per-benchmark facts* (§3.4, §4).
4. `experiments/harness.py`, an acceptance harness. It maps a covariate's honest correlation with item difficulty to an expected ALC gain and applies one gate to every idea. With it we report a catalogue of negative results, each with its script (§6).

### 1.4 Headline numbers

| what | value | source |
|---|---|---|
| first submission (legacy Predictor), formative run 1 | ALC 0.2113, B0 0.359 | F§ "Against the first real formative feedback" |
| shipped hier, formative run 2 (different draw) | ALC 0.1926 (derived), B0 0.237 | budgets in F§ "Subject side at budgets 0 and 1"; ALC by the formula |
| shipped hier minus legacy Predictor, test-like runs 0-99 | -0.0431 ± 0.0019 / 0.0032 / 0.0028 | `results/level_calibration.json`, `experiments/level_calibration.py` |
| shipped hier, test-like runs 0-299 | ALC 0.1658 ± 0.0018 | `results/subject_side.json`, `experiments/subject_side.py` |
| shipped hier, public formative-like runs | 0.2051 (benchmark-first, 150 runs), 0.2020 (pair-uniform, 100 runs) | `results/subject_side.json` |
| live leaderboard, 2026-09-24 | organisers' entry 0.1801, best entry 0.1172 | F§ "Against the live leaderboard" |

(± run SE / cluster-bootstrap SE / stratified SE; see §4.3. Leaderboard entries are single noisy draws: a formative run's ALC has a single-run sd of 0.02 to 0.04.)

---

## 2 Data, protocol and our replica

### 2.1 Data

We use only measurement-db, the organisers' public training release (Hugging Face `aims-foundations/measurement-db`, gated; the dataset card lists CC-BY-SA-4.0). It holds 571,921 responses over 287 subjects and six benchmarks [P§ Data]. Five of the benchmarks are binary. mmdocrag is fraction-valued (only 4.8% of its responses are 0 or 1) and is excluded, as the test excludes non-binary benchmarks. matharena's 0.1% non-binary tail is dropped. Keeping pairs with at least 80 distinct items leaves 221 pairs and 225,843 responses:

| benchmark | pairs | distinct items | mean accuracy | item_features key (levels, share of item variance) |
|---|---|---|---|---|
| multi_swebench | 82 | 2,126 | 0.219 | lang (8, 0.05) |
| matharena | 81 | 1,633 | 0.607 | competition (27, 0.50) |
| researchcodebench | 31 | 212 | 0.354 | paper (20, 0.43) |
| real_webagents | 26 | 233 | 0.330 | website (12, 0.24) |
| swe_rebench | 1 | 6,306 | 0.479 | none |

Sources: pair counts [P§ Data]. Distinct items from `results/harness_thresholds.json` (`meta.oracle_info`). Mean accuracy from the R2 table in F§ "Under the official protocol" and, for swe_rebench, F§ "How much there is to win". Variance shares from F§ "The step-2 analyses behind the model" (`experiments/hier_design/item_signal.py`, recorded output, not rerun).

Properties that shaped the method:

- **Few subjects cross benchmarks.** Only 22 of 287 subjects appear on more than one benchmark. `interactors` is empty everywhere [P§ Data].
- **Repeats are uneven.** 99.8% of matharena items have repeated responses, 99.6% of swe_rebench's, 0.4% of multi_swebench's [P§ Data].
- **Some subject attributes live on one benchmark.** Harness strings appear on only one multi-subject benchmark, multi_swebench (81 of 82 pairs, 12 strings). reasoning_effort appears only on matharena (23 of 81 pairs) (F§ "Subject side at budgets 0 and 1").
- **matharena items are partly textless.** A third of them have no task text ("See image", or fragments of a system prompt). Its 336 Kangaroo items are five-option multiple choice with the options in an image (F§ "What transfers between benchmarks"; F§ "The multiple-choice floor, corrected").
- **Raw solve rates confound difficulty.** Items attempted only by recent subjects look easy, so difficulty targets are Rasch estimates with subject ability divided out (`paiec/rasch.py`; F§ "What transfers between benchmarks").

The hidden benchmarks are drawn from the organisers' inventory of 161 candidate benchmarks, which were assigned at random to the training and test pools before curation. By a classification of their titles and descriptions, it is 39% text QA, 30% images, 15% agents, 6.5% code, 4.6% video, 2.8% math and 2.8% audio. That classification had 95% agreement with 80 hand labels (F§ "Item covariates with a known sign"; its script is in session scratch, `TODO(record)`). The public benchmarks are therefore not a representative sample of the hidden task types.

### 2.2 The protocol, as verified

We read the organisers' streaming client (`tools/streaming_ingestion.py` in the public baseline repository) alongside the competition page [P§]. Six properties of the protocol matter for this work:

- **Scoring is budget-major.** Every pair of a run is evaluated at budget 0 with `labeled` empty. Acquisition then continues until every pair holds up to one label, every pair is evaluated at budget 1, and so on.
- **`labeled` is one shared list per checkpoint.** It holds the first B labels of *every* pair in the run: other subjects' labels on the target's benchmark, and the target subject's labels on other benchmarks.
- **Workers are recreated.** Evaluation workers (up to 16) are recreated from the submission at every checkpoint. Identical inputs are predicted once per checkpoint.
- **The default acquisition policy is a hash rule.** The candidate is taken iff `u < min(1, labels_remaining / items_remaining)`, where u is a sha256 digest of the candidate and its context.
- **Only binary data is used.** Only binary responses count, and only pairs with at least 80 distinct items. Each pair's 50/50 split is fixed across submissions and budgets.
- **The runtime rejects bad outputs.** It rejects non-finite or out-of-range outputs and signals failures by exit code (40, 41, 42).

### 2.3 What our first replica got wrong, and why it mattered

Our first replica (`paiec/evaluator.py`, kept as the *legacy* replica so its numbers stay reproducible) was built before the client was public. It differed from the protocol in five ways [P§ The legacy replica]:

1. **Pair-major scoring.** Each pair was scored at all six budgets before the next pair started, with one predictor instance throughout. A stateful predictor therefore held earlier pairs' full 31-label trajectories while it was being scored at budget 0.
2. **Own labels only.** `labeled` held only the target subject's own labels, never other subjects'.
3. **All pairs in one session.** It scored all 221 pairs in one session instead of 5 to 12.
4. **Leaked identifiers.** It passed private identifiers and raw benchmark names to `predict`.
5. **A process-salted split.** Its first version keyed the split on Python's `hash()`, which is salted per process, so the "fixed" split changed between runs. Both replicas now hash by digest.

The consequences were not cosmetic. On the legacy replica, evaluation order alone moved the assembled predictor from 0.1898 to 0.1725 (F§ "Evaluation order is worth 0.017", `experiments/order_sensitivity.py`). The ladder there made pooled item difficulty look worth about 0.020 ALC (F§ "The predictor ladder", `experiments/ladder.py`), which pointed the research at cross-subject pooling.

Re-measured on the official replica, over 600 formative-sized runs (R1), the picture changes (F§ "Under the official protocol", `experiments/official_baselines.py`):

- **The organisers' empirical-mean baseline is no better than answering 0.5.** It scores 0.2526 ± 0.0014 against 0.2500, because one label sends it to 0 or 1 (0.3746 at B1).
- **The predictor's margin is small.** The predictor we had built beats a Beta(2,2)-smoothed mean by only 0.0026 ± 0.0009 (pair-cluster SE) with a strict prior, or 0.0069 ± 0.0016 with a target-leave-one-benchmark-out prior.
- **Pooled difficulty adds about 0.0007 at formative size.** It needs 64 distinct labeled items on a benchmark, and a formative run has about two pairs per benchmark.

The legacy numbers ranked predictors under a different information set. Nothing in this report is quoted from the legacy replica except where marked "legacy".

### 2.4 Unknowns kept as parameters

The replica treats what the client leaves open as parameters, not assumptions [P§ Still unknown]. The main one is `split_scope`: whether the 50/50 split is drawn per pair or per benchmark item.

- Under per-pair splits, other subjects' acquired labels land on most of a target's evaluation items on dense runs (83% to 99.5% at B31).
- Under per-benchmark splits, that coverage falls to 0% to 5%.

On dense runs (every pair of one benchmark), the predictor's gain over the smoothed mean on matharena is -0.0354 under per-pair splits and -0.0159 under per-benchmark splits (F§ "One benchmark, every pair (R2)"). We report both scopes wherever the difference can matter.

Other open points are listed in `docs/protocol.md`:

- which recorded response is revealed for a repeated item;
- the per-call timeout;
- how a formative run picks and cuts its pairs.

The formative feedback's item counts suggest the platform splits after cutting, with a floor near 88 kept items (`paiec/testlike.py` docstring).

### 2.5 How the replica was verified

- **The default policy matches.** `default_policy` is checked decision for decision against the organisers' `run_streaming` [P§ Interfaces].
- **Argument copies do not change results.** On one run, the platform-like argument copies with 16 simulated workers give per-pair Brier and ECE bit-identical to the fast path for all six predictors (F§ "Under the official protocol").
- **The replica agrees with the analytic formula.** The empirical mean's ALC follows from base rates: about 0.025 + 1.2118 E[p(1-p)]. Once the split and stream order are salted per run, the replica's residual against the exact expectation is -0.0005 ± 0.0006 (F§ "The empirical mean's ALC is a function of base rates").
- **The invariants are pinned by tests.** `tests/test_official.py` (34 test functions) covers them. The repository's test suite has 243 test functions over 10 files, runs on synthetic data and needs no download.

---

## 3 Method

### 3.1 Model

`paiec/hier.py` models the log-odds that subject s answers item i of benchmark b correctly:

    eta = mu_b + theta_s + delta_sb - g_i - e_i
    p   = c_i + (1 - c_i - slip) * E[ sigmoid(eta) ]

| term | meaning | prior |
|---|---|---|
| mu_b | the benchmark's level, shared by every subject's labels on it | N(mu0, sigma_mu^2), Gaussian; Student-t optional |
| theta_s | the subject's standing, keyed on its canonical name | attribute ridge mean (times attr_scale), sd sigma_attr; identity term capped |
| delta_sb | the pair's own deviation | N(0, sigma_delta^2) |
| g_i | item_features group effects, one per key whose visible values are neither all numeric nor constant | N(0, sigma_g^2) divided evenly over a benchmark's keys; the group share of item variance is capped at 0.25 |
| e_i | the item's own residual | N(0, sigma_d^2), integrated exactly per item; sigma_d^2 + sigma_g^2 is the median public item variance |
| c_i | guessing floor: `guess` times the multiple-choice floor read from the item text (`paiec/mcq.py`) | fixed, guess 0.5 |

A component that no label touched (a new benchmark, subject, pair, group or item) stays at its prior, and the predictive integrates over it. At budget 0 the prediction is therefore the level prior plus the attribute prior, integrated. That is where the hidden test's headroom turned out to be (§5.1).

Everything is on the scale of an item-level (Rasch) model, where the public variance components were measured. Everything is also fitted from `labeled` alone: one fit per content fingerprint of `labeled`, so the prediction is a pure function of `(input, labeled)`.

### 3.2 Inference

**Item residuals are integrated, not maximised.** A joint mode over all effects with an item variance near 8.7 and one label per item is the classic penalised-quasi-likelihood failure [Breslow and Clayton 1993]. The item effects absorb each label and the fit lands where the logistic is flat. In a pinned test, a pair with 160 of 200 successes is predicted at 0.70 on a new item that way, and at 0.79 once the item residual is integrated (`tests/test_hier.py`). The vector x therefore holds every component except the e_i. Each item's likelihood is the one-dimensional integral over e_i: adaptive Gauss-Hermite with 20 nodes, or a fixed 201-node trapezoid on floored items, whose integrand can be bimodal. Newton with a line search finds the mode of the joint log posterior of x.

**The target is read along its line.** The Laplace (Gaussian) posterior of the target's linear predictor a'x under-reacts to a pair's first labels, by 0.013 to 0.02 in probability after one to seven labels, because the mode of a skewed posterior sits nearer the prior than its mean. So a'x is read on a grid along the Gaussian conditional mean, with the exact log posterior at each point. This is the Gaussian strategy of INLA [Rue, Martino and Chopin 2009], and for a single pair it is the exact marginal. E[sigmoid] is then taken by a 171-node trapezoid, exact to 1e-9. The docstring of `paiec/hier.py` documents the remaining approximation error case by case. On cases drawn from the model it moves Brier by about 0.001 at B1 and by at most 1e-4 from B7 on.

**Cost.** Without floors the log posterior is concave and the mode unique. A floored item's success term is not concave, so Newton's matrix is shifted to positive definite when needed. Fits that stop unconverged are counted.

### 3.3 Offline prior and empirical-Bayes hyperparameters

`paiec/prior.py` fits one item-level model per public benchmark, `logit p = mu_b + t_s - g_i - e_i`. It uses each subject's first recorded response per item (the one acquisition reveals) and finds the variances by Laplace-EM. That single fit supplies two things:

- **The subject standings t_s**, centred within the benchmark. An attribute ridge predicts them from provider, release date (linear in days since 2023), model size parsed from the name, reasoning-effort dummies and a harness-present flag (`paiec/subjects.py`).
- **The levels, item variance and group share**, which `fit_hyper` pools across benchmarks.

Each hyperparameter the included benchmarks cannot identify falls back to a fixed `REFERENCE` value that no fit went into. For example, mu0 = 0 when fewer than three levels are available.

Every experiment refits the prior and the hyperparameters *without the target's parent benchmark* (§4.2). The shipped bundle is fitted on all five public benchmarks. Its values are listed in F§ "Verdict: ship hier with the level moved down", with the level prior from `LEVEL` in `submission/model.py`, and match `submission/prior.json` as built by `tools/build_submission.py`:

| mu0 | sigma_mu | attr_scale | sigma_theta | sigma_delta | sigma_attr | sigma_d | sigma_g | slip | guess |
|---|---|---|---|---|---|---|---|---|---|
| **-2.5** | **2.5** | **0.5** | 0.1 | 2.382 | 1.018 | 2.671 | 1.542 | 0.01 | 0.5 |

The three bold fields are the level prior set for the hidden test (§3.4). The rest are the empirical-Bayes fit.

The subject's cross-benchmark identity carries almost nothing beyond its attributes. The shared part of a named model's attribute residual across benchmarks, tau2_res, is estimated at -0.002 on the public data and clipped to [0.01, 0.2]. That puts sigma_theta at its floor of 0.1 and the link weight across benchmarks at 0.0018 (`paiec/hier.py` docstring; F§ "The step-2 analyses behind the model").

### 3.4 Calibrating the level prior for the hidden test

**The signal.** The first formative feedback (§5.1) showed the legacy predictor confidently optimistic at budget 0. It predicted near 0.75 on pairs whose rates were low. Two readings point the same way:

- Reading each pair's B31 Brier as irreducible noise, p(1-p) = B31, puts the hidden pairs at a mean of about -1.6 on the pair-accuracy logit scale (lower roots). The public mean is -0.74 (F§ "Against the first real formative feedback").
- Independently, *if* the organisers' leaderboard entry is the empirical-mean baseline, inverting its ALC puts E[p(1-p)] near 0.128, against 0.181 on public runs (F§ "Against the live leaderboard").

**The regime.** We cannot fit to hidden labels, and one run of nine pairs identifies little. So we built test-like runs from public data that reproduce the feedback (§4.1), and chose the level prior on them.

**The candidates.** Grids over (mu0, sigma_mu, attr_scale) for hier, with Gaussian and Student-t levels. A pair-level empirical-Bayes re-estimate of the level at every checkpoint. A level fix for the legacy predictor. All are listed in F§ "Calibrating for the hidden test".

**The selection.** The best mean ALC on the selection half of the test-like runs (runs 0 to 99), among configurations that lose at most 0.003 ALC against the legacy predictor on *both* public weightings (the guard). Configurations were confirmed on runs 100 to 199.

**The rule we kept.** The feedback may set *global* hyperparameters of the level distribution, here three numbers: mu0, sigma_mu and attr_scale. It never sets anything keyed on an anonymous benchmark or subject id. No prediction was shaped to probe hidden labels, and no feedback number was fitted beyond the regime's defaults (F§ "Against the real feedback"). This follows the competition's conduct rule against extracting test data and identifying anonymous ids. It is also the only use of one noisy run that generalises to a summative test on a different subset.

**What shipped.** The rule's argmax was mu0 -3.5 at attr_scale 0.25. The recommendation took mu0 -3.0 on a tie-break by the guard's margin (0.0004 behind on selection, level on confirmation). An audit then moved the shipped choice to the milder guarded configuration, **mu0 -2.5, sigma_mu 2.5, attr_scale 0.5** (F§ "What actually shipped, after the audit"). The audit's reasons:

1. Two of the feedback's pairs are more plausibly high-rate. Their B0 was already close to their B31 under an optimistic prior. So the hidden levels look spread both ways, and the aggressive setting loses about 0.015 per pair on high-rate pairs.
2. The two configurations sit on a flat plateau. The milder one gives up about 0.002 in the tuned regime, gains about 0.0024 on public runs and has no worse measured worst case.
3. The "held-out" half redraws runs from the same catalogue of pairs. It measures redrawing, not generalisation to unseen parents.

### 3.5 Run-time engineering

- **Purity and caching.** `predict` is a pure function of `(input, labeled)`. Fits are cached under a content fingerprint of `labeled`. Items and subjects are keyed on digests of all their visible fields, because the official input carries no ids and text prefixes collide.
- **Failure handling.** Nothing raises. A failed fit falls back to the prediction without labels, then to 0.5. An unusable `prior.json` falls back to the level prior without a subject prior.
- **Packaging.** The archive ships `model.py`, `prior.json` and the run-time modules renamed to `paiec_rt/`, so that no platform-side `paiec` can shadow them. Shipped code imports only the standard library and numpy at module level, and BLAS is held to one thread.
- **The build either passes every check or produces no archive.** `tools/build_submission.py` fails unless all of the following hold:
  - a fresh interpreter loads `model.py` the way the validator does and predicts real items in the official format at every budget, with a fresh predictor per budget;
  - those predictions match the in-repository predictor bit for bit;
  - no fallback fires and no heavy library loads;
  - the organisers' validator prints OK.

---

## 4 Evaluation methodology

### 4.1 Runs

| regime | what it is | used for |
|---|---|---|
| **R1** public formative-like | `official.sample_run`: 5 to 12 pairs, cut to the 1,000-item cap; seed 0. Benchmark-first weighting (a run holds 8.6 pairs over 4.4 benchmarks) and pair-uniform weighting (3.4 benchmarks) | baselines; the public guard |
| **R2** dense | `official.dense_run`: every eligible pair of one benchmark, whole | the most cross-subject evidence a run can carry |
| **test-like** | `testlike.Regime()` at its defaults, seed 2 (seed 3 for sensitivities); see below | primary regime for hidden-test choices |
| sensitivities | level_mean -1.2 and -2.0 instead of -1.6; groups merged at random with no strata ("mix/whole"); no date shift | robustness of every hidden-test choice |

Sources: F§ "Under the official protocol"; F§ "Hierarchical model"; F§ "Calibrating for the hidden test".

**How test-like runs are built** (`paiec/testlike.py`).

- *Pseudo-benchmarks.* The five public benchmarks are too few and too central, so each is cut into pseudo-benchmarks: item_features groups sorted by difficulty and merged, difficulty strata (cross-fitted, so a subject's own labels never choose its items' stratum), whole parents, and random chunks for the single-subject benchmark. Each gets its own anonymous benchmark_id. Every label is a real recorded response; only the grouping of items is new.
- *Run shape.* A run draws 5 to 12 pairs. With some probability the next pair joins a pseudo-benchmark already in the run; otherwise it opens a new one. This reproduces the feedback's about one pair per benchmark.
- *Level tilt.* The draw is tilted toward a target distribution of pair accuracy logits (mean -1.6, sd 1.5).
- *Date shift.* Visible release and access dates are shifted 1.25 years later. This reproduces the feedback's budget-0 optimism for the legacy prior's linear date term.

**How well the regime matches.** The defaults are the grid point closest to the feedback's B0 and B1. That agreement is in sample. The budgets not used in tuning (B3 to B31) sit at z +0.09 to +0.50 of the replica's single-run sd without tuning. On public R1 the feedback's B0 is at z +8.5. ALC alone does not separate the regimes: the legacy predictor scores 0.2073 on test-like runs, 0.2075 on R1 and 0.2113 in the feedback (`experiments/testlike_check.py` docstring; `results/testlike_check.json`).

**How the regime is thin.** Pseudo-benchmarks carry less item structure than public benchmarks. An item-level oracle gains 46% of the pair-rate oracle's Brier on test-like runs against 55% on public runs (F§ "Item signal from the pair's own labels"). Any item-aware against level-only comparison must therefore also hold on mix/whole and on R1.

### 4.2 Leave-one-parent-out

For every target, the subject prior and every empirical-Bayes hyperparameter are fitted without the target's *parent* benchmark (`prior.build(pairs, (parent,))`). The model factory dispatches on the anonymous benchmark_id, so `predict` never sees a name. On public runs the parent is the benchmark itself. A stricter line, which fits without every benchmark of the run (strict run-LOBO), is reported for the main comparison (F§ "Strict run-LOBO").

Hyperparameters chosen by a study (a covariate's slope, a layer's configuration) are selected by **nested** leave-one-parent-out:

1. For each held-out parent, choose the configuration (or "off") on the other parents' pair appearances.
2. Score that choice on the held-out parent.
3. Redo the selection inside every bootstrap resample.

### 4.3 Statistics

All comparisons are **paired**: every candidate scores the same runs, checkpoints and acquisitions. A run's ALC is the mean over its pairs. Differences are written as "± run SE / cluster SE / stratified SE":

- **run SE** is over runs. It overstates precision, because runs redraw the same pairs.
- **cluster SE** is a pair-cluster bootstrap with 2,000 resamples and a ratio estimator. The cluster is (benchmark, subject) on public runs and (parent, subject) on test-like runs, where pseudo-benchmarks of one parent overlap.
- **stratified SE** is the same bootstrap resampled within each parent benchmark.

None of these covers variation *between* benchmarks, and four parents carry every test-like number. We therefore also report per-parent and leave-one-parent-out ranges, and a benchmark-level mean with its SE across benchmarks where it matters. When many options are compared with one default, we state the Bonferroni bar (for example 2.95 SEs two-sided for sixteen comparisons; F§ "Ablations and sensitivities").

### 4.4 Selection discipline

- **Selection and confirmation halves.** Candidates are chosen on runs 0 to 99 and confirmed on runs 100 to 199 (§3.4). The halves redraw from one catalogue of pairs, so they measure redrawing, not new parents.
- **Guards.** Every hidden-test choice must hold on both public weightings.
- **Pre-registered gates.** A component ships only if all four hold (F§ "Subject side at budgets 0 and 1"):
  1. nested selection switches it on in at least 3 of 4 folds;
  2. the nested test-like difference is at most -0.002;
  3. no held-out parent is above +0.002;
  4. neither public weighting loses more than 0.001.
- **Provenance.** Results files record each invocation's command, wall time and digests of the code it ran (`passes`), and `--summarise` rebuilds every summary from stored rows.

### 4.5 The acceptance harness and its gate table

Every later item-side idea reduces to one number per item. `experiments/harness.py` scores any such covariate x the same way, against the model that ships. It adds a centred logit offset:

    q_i = sigmoid( logit p_i + cap(beta (x_i - mean of x over labeled items)) ),  cap(o) = 4 tanh(o/4)

Here p_i is the shipped hier's prediction, fitted leave-one-parent-out. Centring keeps hier's level, so only the ordering of items comes from x, and at B0 the covariate does nothing. The slope beta is either:

- **transferred**: one coefficient per budget, fitted on the other parents; or
- **per-pair**: a MAP from the target pair's own labels, under a zero-mean prior of s per standard deviation of x.

The harness first reproduces the in-sample oracle that an earlier study scored: -0.0437 test-like and -0.0557 mix/whole as target, against -0.0440 and -0.0558 as reproduced. It then builds the **gate table** from degraded oracles, x = r z + sqrt(1 - r^2) noise, where z is an *honest* difficulty fitted on other subject folds only. For each r the table records what a covariate with that honest correlation would score through the gate:

| honest r (within-pair r, test-like) | transferred slope, nested: test-like | mix/whole | R1 bench-first | R1 pair-uniform | passes | per-pair slope, nested: test-like | passes |
|---|---|---|---|---|---|---|---|
| 0 (-0.002) | +0.00005 | +0.00004 | +0.00001 | -0.00001 | no | +0.00001 | no |
| 0.1 (0.081) | +0.00003 | -0.00020 | -0.00038 | -0.00052 | no | +0.00001 | no |
| 0.2 (0.163) | -0.00088 | -0.00191 | -0.00231 | -0.00286 | no | -0.00009 | no |
| 0.3 (0.247) | **-0.00238** | -0.00449 | -0.00526 | -0.00635 | **yes** | -0.00032 | no |
| 0.5 (0.420) | -0.00717 | -0.01238 | -0.01415 | -0.01693 | yes | **-0.00252** | **yes** |
| 0.7 (0.609) | -0.01511 | -0.02390 | -0.02715 | -0.03212 | yes | -0.00767 | yes |
| honest oracle | -0.0361 ± 0.0010 / 0.0030 / 0.0029 | -0.0484 | -0.0540 | -0.0620 | yes | -0.0224 | yes |

Source: `results/harness_thresholds.json` (`thresholds.honest`, `acceptance`), `python experiments/harness.py --stage table`; 300 test-like, 150 mix/whole and 100 + 100 public runs.

So a covariate needs an honest correlation of about 0.3 with item difficulty if its slope can be transferred from other benchmarks, and about 0.5 if the slope must be learned per pair from at most 31 labels. An uninformative covariate forced on with a per-pair slope costs +0.0007 to +0.0010 (the r = 0 forced rows in the results file). One noise draw of a degraded oracle moves its test-like difference by about its cluster SE (draw sd 0.0005 at r = 0.3), so a real covariate, which is one draw, should be read against that spread as well.

---

## 5 Results

### 5.1 Formative feedback

Two formative runs have been scored on the platform. They drew different subjects, and each is one noisy draw.

| run | model | pairs | B0 | B1 | B3 | B7 | B15 | B31 | ALC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | legacy Predictor, commit b68492c | 9 (7 benchmarks) | 0.3589 | 0.2539 | 0.2043 | 0.1712 | 0.1693 | 0.1568 | 0.2113 |
| 2 | shipped hier, LEVEL -2.5 / 2.5 / 0.5 | 8, `TODO(record)` | 0.237 | 0.195 | 0.196 | 0.184 | 0.178 | 0.183 | 0.1926 (derived) |

Sources: run 1 from F§ "Against the first real formative feedback" (per-pair rows in `testlike.FEEDBACK`). Run 2 budgets from F§ "Subject side at budgets 0 and 1"; its ALC is derived by the ALC formula from those rounded budgets. Its pair count and per-pair table are not yet in `docs/findings.md`: `TODO(record)` from the Codabench scoring output.

**Run 1 lost its score at budget 0.**

- It scored 0.3589 where a constant 0.5 scores 0.25, with ECE up to 0.75. Answering 0.5 at B0 and B1 alone would have given about 0.1996.
- The relative attribute standing placed strong 2025-26 subjects near p = 0.75, on pairs whose B31-implied rates were 0.006 to 0.45 (lower roots).
- On the pair-accuracy logit scale those rates average -1.6 with sd 1.5, where public pairs average -0.74 with sd 1.50.

This is the observation behind §3.4.

**Run 2, with the shipped level prior, lost far less at budget 0.** B0 was 0.237 against run 1's 0.359, and close to the test-like mean for the shipped model (0.2165, §5.4). But its curve is flat from B1 on: 0.195 at B1 and 0.183 at B31. From B3 to B31 it sits 0.03 to 0.045 above the test-like means (0.1645 to 0.1386). Its B0 excess over B31 (0.054) is the headroom the subject-side study (§6.6) targeted. One run cannot say whether the gap at B3 to B31 comes from item structure, more central hidden rates than the regime assumes, or draw noise. `TODO(step5)`: read run 2 per pair, pooled with run 1's nine pairs, as F§ "What actually shipped, after the audit" prescribes, and place its budgets against the test-like single-run sds.

**The leaderboard.** On 2026-09-24 the organisers' entry scored 0.1801 and the best entry 0.1172, both single draws. Before run 2, the per-pair matched estimate for the recommended configuration on run 1 was about 0.178, with a range of 0.169 to 0.183 (F§ "What actually shipped, after the audit"). Run 2's 0.1926 on a different draw lies above that range. We report it as it is.

### 5.2 Public formative-like runs (R1)

**Baselines** (600 runs, benchmark-first, split scope 'pair'; F§ "Under the official protocol", `experiments/official_baselines.py`):

| predictor | B0 | B1 | B7 | B31 | ALC |
|---|---|---|---|---|---|
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| empirical mean (official baseline) | 0.2500 | 0.3746 | 0.2131 | 0.1913 | 0.2526 ± 0.0014 |
| smoothed Beta(2,2) | 0.2500 | 0.2349 | 0.2059 | 0.1908 | 0.2158 ± 0.0008 |
| legacy Predictor, target-LOBO prior | 0.2313 | 0.2238 | 0.2034 | 0.1832 | 0.2090 ± 0.0009 |
| base-rate oracle E[p(1-p)] | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 ± 0.0010 |

**hier against the legacy Predictor** (300 runs per setting; F§ "Hierarchical model", `experiments/hier_eval.py`, `results/hier_eval.json`). These are hier's fitted defaults, before the level calibration:

| weighting, split scope | hier minus Predictor | 95%, pair-cluster | 95%, stratified |
|---|---|---|---|
| benchmark-first, pair (primary) | -0.0024 ± 0.0004 / 0.0013 / 0.0008 | [-0.0047, +0.0002] | [-0.0040, -0.0008] |
| benchmark-first, benchmark | -0.0010 ± 0.0003 / 0.0014 / 0.0009 | [-0.0035, +0.0017] | [-0.0027, +0.0007] |
| pair-uniform, pair | -0.0017 ± 0.0004 / 0.0008 / 0.0007 | [-0.0033, -0.0002] | [-0.0031, -0.0003] |
| pair-uniform, benchmark | -0.0015 ± 0.0003 / 0.0008 / 0.0007 | [-0.0030, +0.0000] | [-0.0029, +0.0000] |

The gain comes from pooling a benchmark's level across the subjects a run holds on it. On multi-subject benchmarks it is worth 0.0015 to 0.0033 to a target that shares its benchmark with one other pair, 0.0019 to 0.0044 with two or more, and nothing measurable to a target alone on its benchmark (at most 0.0016 ± 0.0014). On the single-subject swe_rebench pair hier is worse, by 0.006 to 0.009 against the Predictor, mostly through its widened level prior. Weighted like the real run (5 targets alone, 4 in pairs), hier's expected gain on public data would be 0.0006 to 0.0024. On that evidence the first verdict was to keep the Predictor (F§ "Verdict: keep the Predictor"). §5.3 reversed it.

### 5.3 Test-like runs: calibrating the level

**What moving the level buys, unconstrained** (selection half, runs 0 to 99; F§ "What moving the level buys, unconstrained"):

| family | best configuration | ALC | minus Predictor | public cost, bench-first / pair-uniform |
|---|---|---|---|---|
| legacy Predictor (first submission) | | 0.2039 | | |
| hier, fitted defaults | | 0.1906 | -0.0134 ± 0.0009 / 0.0017 / 0.0014 | -0.0019 / -0.0017 |
| smoothed Beta(2,2) | | 0.1788 | -0.0252 ± 0.0013 / 0.0021 / 0.0019 | +0.0070 / +0.0075 |
| hier, Gaussian level | mu0 -4, sigma_mu 2.5, attr_scale 0.5 | 0.1572 | -0.0467 ± 0.0025 / 0.0042 / 0.0036 | +0.0036 / +0.0032 |
| hier, Student-t level | mu0 -4, scale 1.8, attr_scale 0.5 | 0.1570 | -0.0469 ± 0.0025 / 0.0042 / 0.0036 | not scored |
| hier, empirical-Bayes level | on the Gaussian above | 0.1567 | -0.0473 ± 0.0025 / 0.0042 / 0.0036 | +0.0034 / +0.0033 |
| Predictor with a level fix | off -2.5, sc 1, va 2 | 0.1562 | -0.0477 ± 0.0027 / 0.0045 / 0.0038 | +0.0329 / +0.0329 |

Four observations follow from this table and the grid behind it.

- **The legacy predictor loses to answering from Beta(2,2) here.** What it loses is its prior. Every family reaches about 0.156 to 0.157 once its level is moved, and the plateau is flat: 14 of 108 Gaussian configurations lie within 0.001 of the best.
- **The widest level prior always won.** At every shift, sigma_mu 2.5 beat 1.8, 1.3 and 0.9 in every cell of both grids.
- **The public guard decides.** Moving the level costs the Predictor far more on public runs than it costs hier. hier gives back most of its B0 and B1 losses from B7 on. The Predictor's fit (`fitting.fit_ab`) also diverges when the prior sits far from the labels, a library bug reported in F§ "fit_ab diverges when the prior is far from the labels".
- **Calibration is worth far more than the modelling.** Under the guard, the best hier configurations gain about 0.042 on the confirmation half. That is about twenty times the roughly 0.002 that hier's pooled level gained on public runs (F§ "Verdict: ship hier with the level moved down").

**The shipped configuration** (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5):

- on the selection half, ALC 0.1608, and minus the legacy Predictor -0.0431 ± 0.0019 / 0.0032 / 0.0028, ranging from -0.0336 (multi_swebench's appearances left out) to -0.0483 (matharena's left out) (`results/level_calibration.json`, `summary.selection`);
- on all 300 test-like runs, ALC 0.1658 ± 0.0018 (`results/subject_side.json`).

The recommended aggressive configuration (mu0 -3.0, attr_scale 0.25) was confirmed on held-out runs 100 to 199 at -0.0418 ± 0.0024 / 0.0047 / 0.0042 against the Predictor. On public runs it cost +0.0008 ± 0.0012 / 0.0025 / 0.0022 (benchmark-first) and +0.0007 ± 0.0012 / 0.0021 / 0.0019 (pair-uniform). The shipped milder one gains about 0.0024 over it on public runs and gives up about 0.002 in the tuned regime (F§ "What actually shipped, after the audit").

**Sensitivity to the regime** (mu0 -3.0 / attr_scale 0.25 against the Predictor, 100 runs each; F§ "Sensitivity to the regime"):

| level_mean -1.2 | level_mean -2.0 | groups merged at random, no strata | no date shift |
|---|---|---|---|
| -0.0388 ± 0.0021 / 0.0044 / 0.0039 | -0.0524 ± 0.0022 / 0.0044 / 0.0039 | -0.0467 ± 0.0017 / 0.0049 / 0.0038 | -0.0139 ± 0.0014 / 0.0028 / 0.0026 |

The gain holds wherever the hidden level sits within the range one feedback run allows. Without the synthetic date shift, which is what inflates the attribute prior in this regime, it shrinks to 0.014.

**The empirical-Bayes level adapts in the right direction.** At budget 1 it moves most of the way from the public centre toward each regime's level, and it orders the regimes by their levels (F§ "The empirical-Bayes level adapts the right way"). It ties the fixed configuration because the level fixed at B0 decides most of the gain. It was not shipped: it is about 70 lines of experiment code outside the library, with a slower worst call.

### 5.4 The shipped model across regimes

| regime | runs | B0 | B1 | B3 | B7 | B15 | B31 | ALC ± run SE |
|---|---|---|---|---|---|---|---|---|
| test-like (primary) | 300 | 0.2165 | 0.1892 | 0.1645 | 0.1525 | 0.1450 | 0.1386 | 0.1658 ± 0.0018 |
| test-like, mix/whole | 200 | 0.2178 | 0.1931 | 0.1692 | 0.1590 | 0.1524 | 0.1454 | 0.1711 ± 0.0023 |
| test-like, no date shift | 100 | 0.1948 | 0.1763 | 0.1620 | 0.1479 | 0.1413 | 0.1353 | 0.1585 ± 0.0033 |
| public R1, benchmark-first | 150 | 0.2350 | 0.2243 | 0.2128 | 0.1973 | 0.1852 | 0.1771 | 0.2051 ± 0.0023 |
| public R1, pair-uniform | 100 | 0.2442 | 0.2215 | 0.2069 | 0.1929 | 0.1813 | 0.1708 | 0.2020 ± 0.0026 |
| platform, formative run 2 | 1 | 0.237 | 0.195 | 0.196 | 0.184 | 0.178 | 0.183 | 0.1926 (derived) |

Source: `results/subject_side.json` (`summary.regimes.*.ship`), `experiments/subject_side.py`. Its 'ship' arm reproduces `experiments/itemsig_eval.py`'s base exactly.

### 5.5 Ablations of hier

These are ablations of hier's fitted defaults, before the level calibration: public R1, primary setting, each option alone against the default on the same runs (F§ "Ablations and sensitivities", `experiments/hier_eval.py`).

| option | runs | minus default | where it acts |
|---|---|---|---|
| attribute prior off | 150 | +0.0039 ± 0.0003 / 0.0009 / 0.0008 | B0 to B3 |
| feature groups off | 150 | +0.0017 ± 0.0002 / 0.0003 / 0.0003 | B7 to B31 |
| level pooling off (a level per pair) | 150 | +0.0012 ± 0.0003 / 0.0004 / 0.0004 | B1, B3 |
| level centre 0 instead of the mean | 150 | +0.0011 ± 0.0003 / 0.0007 / 0.0006 | per benchmark ±0.004 |
| pair deviation off | 150 | +0.0001 ± 0.0001 / 0.0001 / 0.0001 | |
| linking off, relink 0.1 / 0.3 | 100 | 0.0000 to -0.0001 | |
| text term on | 100 | +0.0002 ± 0.0001 / 0.0001 / 0.0001 | 5.3 ms a call |
| Student-t level (nu 3) | 100 | -0.0005 ± 0.0001 / 0.0002 / 0.0001 | 18.9 ms a call |
| Laplace fit without the line | 150 | -0.0007 ± 0.0001 / 0.0003 / 0.0001 | B1, B3 |
| sigma_mu x0.5 | 100 | -0.0021 ± 0.0003 / 0.0008 / 0.0004 | B1, B3; swe_rebench -0.0068 |
| sigma_mu x2 | 100 | +0.0049 ± 0.0004 / 0.0015 / 0.0005 | B1, B3 |
| sigma_delta x2 | 100 | +0.0060 ± 0.0004 / 0.0013 / 0.0008 | B1, B3 |

**Three components pull their weight on public runs:** the attribute prior, the feature groups and the pooled level.

**Three options beat the default by more than two cluster SEs:** a narrower level prior, the Student-t level and the Laplace fit. None clears a two-sided Bonferroni bar for sixteen comparisons. All three make the model react less to a run's first labels. The deliberately widened level prior costs about 0.002 on public runs. That is the price of the bet on hidden levels far from the public centre, which the first feedback supported.

**The corrected multiple-choice floor.** It now reads "(A, B, C, D, or E)" lists and ignores TikZ point labels. It gains 0.00026 ± 0.00003 / 0.00008 on test-like runs and 0.00027 to 0.00045 on public runs, all on matharena at B0 and B1 (F§ "The multiple-choice floor, corrected", `experiments/mcq_floor.py`). It is adopted in the library. `dist/paiec.zip` still ships the old floor until it is rebuilt (`TODO(verify)`: rebuild and resubmit).

### 5.6 Cost

The shipped configuration takes 0.75 ms per evaluation call on test-like runs and 0.88 ms on public runs, with a slowest single call of 0.45 s (a checkpoint's first call, which fits). This was measured with six processes on a shared machine. On dense real_webagents and researchcodebench it takes 2.0 to 2.4 ms per call, at most 0.16 s. A formative run is about 3,000 evaluation calls, a few seconds against the 8-hour limit (F§ "Calibrating for the hidden test", Latency). Dense multi_swebench and matharena, hier's worst case, were not measured (§7).

---

## 6 What does not transfer

Each idea below was measured leave-one-parent-out against the model that shipped at the time, and each is recorded as negative with its script. They share one reason for failing. The hidden test's headroom lies at budgets 0 and 1 and in per-item structure of benchmarks nobody has seen, and none of these signals reaches either.

| idea | best honest estimate | verdict | script, results |
|---|---|---|---|
| item difficulty from TF-IDF text, across benchmarks | LOBO Pearson 0.14, 0.23, 0.09, -0.22, 0.16 | does not transfer | `experiments/transfer.py` (data-level) |
| item difficulty from Qwen3-Embedding-0.6B | LOBO ridge -0.16 / -0.03 / -0.23 / -0.02; within benchmark 0.66 / 0.27 / 0.38 / 0.51 | does not transfer | `experiments/emb_transfer.py`, `results/emb_transfer.json` |
| blind LLM difficulty rating | 0.482 on matharena; -0.13 to 0.21 elsewhere | mathematics only | `experiments/llm_rating/analysis.py` |
| text-similarity residual layer (itemsig) | nested -0.00002 ± 0.00002 / 0.00008 / 0.00008 | no gain | `experiments/itemsig_eval.py`, `results/itemsig_eval.json` |
| meta-learned heads on frozen embeddings | `TODO(record)` | `TODO(record)` | scratch `heads.py`, not in the repository |
| known-sign item cues (ordinal fields, position, format, length, stated size, image refs) | nested test-like 0 to +0.00003 | none passes the sign rule or the gate | `experiments/itemcov_eval.py`, `results/itemcov_eval.json` |
| subject side: harness identity, ordered effort, date forms, Student-t level | best nested -0.00097 ± 0.00033 (ordered effort) | none passes | `experiments/subject_side.py`, `results/subject_side.json` |
| acquisition policies (legacy replica) | coverage-first -0.0000 ± 0.0013 | none beats random | `experiments/acquisition.py` |
| post-hoc temperature and slip (legacy replica) | loses 0.0007 ± 0.0011 | does not transfer | `experiments/acquisition.py` |
| offline bank of per-item results | 1 of 161 inventory benchmarks usable | closed | `experiments/inventory_scan.py` |

### 6.1 Item difficulty from text: TF-IDF and neural embeddings

Leave-one-benchmark-out, a TF-IDF map from item text to Rasch difficulty correlates at 0.14 on matharena, 0.23 on multi_swebench, 0.09 on real_webagents, -0.22 on researchcodebench and 0.16 on swe_rebench. Within matharena the same model reaches 0.73, but most of that is identifying which competition an item comes from. Competition alone explains 40% of the variance, and removing the competition mean drops the correlation to 0.49 (F§ "What transfers between benchmarks").

Qwen3-Embedding-0.6B embeddings do no better across benchmarks (F§ "Neural embeddings do not carry difficulty to an unseen benchmark"):

- LOBO ridge correlations are -0.16, -0.03, -0.23 and -0.02, and kNN -0.08 to +0.16.
- R² as predicted is at most 0 everywhere, and the carried slope has the wrong sign.
- Within a benchmark (5-fold) they reach 0.66, 0.27, 0.38 and 0.51. But the item_features group mean alone reaches 0.57, 0.12, 0.41 and 0.52, and with whole groups held out the embeddings fall to 0.40, 0.16, -0.02 and -0.09. So within a benchmark the embedding mostly identifies the group, which hier already learns from labels.

*Literature.* Amortized calibration predicts item difficulty from embedded question content and reports generalisation across datasets [Truong et al. 2025]. ADeLe reports that black-box predictors built on embeddings or fine-tuning are weaker than rubric-based demand levels, especially out of distribution [Zhou et al. 2025]. Our result is the out-of-distribution end of that picture on agentic and code benchmarks: the direction of difficulty in embedding space is not shared between benchmarks. For bug-fixing tasks, Agent Psychometrics predicts task-level success on unseen benchmarks from issue statements *together with* repository context, solutions and test cases [Ge et al. 2026]. The competition's items carry the issue text and grouping metadata only.

### 6.2 LLM difficulty judgements

180 items across four benchmarks were rated blind on a 0-100 scale and compared with Rasch difficulty (F§ "Language-model difficulty judgement"). The rating model is `TODO(disclose)`: the ratings are hard-coded in `experiments/llm_rating/ratings_main.py`, and the repository does not record which model produced them.

| benchmark | n | Pearson | 95% CI |
|---|---|---|---|
| matharena | 45 | 0.482 | [+0.22, +0.68] |
| multi_swebench | 45 | 0.133 | [-0.17, +0.41] |
| real_webagents | 45 | 0.210 | [-0.09, +0.47] |
| swe_rebench | 45 | -0.128 | [-0.41, +0.17] |
| pooled within benchmark | 180 | 0.174 | [+0.03, +0.31] |

A control on 60 fresh multi_swebench items with the full issue text gives Pearson +0.252 (interval touching zero) and Spearman +0.109, so truncation was not the explanation.

*Literature.* Human-labelled difficulty of math and coding problems is linearly decodable from model activations [Lugoloobi and Russell 2025], and rubric-based LLM annotation of task demands predicts instance-level performance [Zhou et al. 2025]. We see the judge track difficulty where it is visible in the statement (competition mathematics) and not where it lies in the environment (repository size, files to touch, test harness). The gate table (§4.5) needs an honest r of 0.3 across benchmarks. The judge clears that on one of four.

`TODO(step5)`: close-out of the local Qwen3-4B-Instruct-2507 judge. Its digit-distribution rating, entropy and text NLL are extracted by `experiments/llm_features.py`, and it will be run through the harness's `--stage eval`.

### 6.3 A label-conditional text-similarity layer

`paiec/itemsig.py` carries hier's residuals on a pair's labeled items to unlabeled items with similar text. It was run over a 480-configuration grid, with nested leave-one-parent-out selection redone in every bootstrap resample (F§ "Item signal from the pair's own labels"):

| regime (base ALC) | nested, selected within the regime | in-sample best of 480 |
|---|---|---|
| test-like, primary (0.1658) | -0.00002 ± 0.00002 / 0.00008 / 0.00008 | -0.00006 |
| test-like, mix/whole (0.1711) | -0.00028 ± 0.00005 / 0.00021 / 0.00019 | -0.00037 |
| public R1, benchmark-first (0.2057) | -0.00003 ± 0.00012 / 0.00049 / 0.00046 | -0.00071 |
| public R1, pair-uniform (0.2020) | -0.00076 ± 0.00018 / 0.00031 / 0.00029 | -0.00112 |

The layer cannot act at B0 or B1 by construction: there are no labels at B0, and at B1 centring within the pair zeroes the single residual. At B31 it takes 0.5% of the gap between the pair-rate oracle and the item oracle. That gap is 0.063 of Brier on test-like runs, and the base already matches the pair-rate oracle there. Selected on the public runs of the other parents, the layer picks aggressive settings that lose +0.0019 ± 0.0007 on held-out researchcodebench. Not shipped.

*Literature.* Generic assessors that transfer instance-level performance from reference instances do well in distribution, but out of distribution "no clear winner emerges and the overall performance is worse" [Pacchiardi et al. 2024].

### 6.4 Meta-learned heads on frozen embeddings

`TODO(record)`: an episode-trained study put linear and low-rank heads on frozen Qwen3-Embedding-0.6B features on top of the shipped hier: a meta-learned difficulty direction, a learned-metric few-shot kernel, and an assessor-style subject × item term. It was scored with nested leave-one-parent-out selection. The study, `heads.py`, is in session scratch and not yet in the repository. Before this paragraph quotes a result, copy it into `experiments/` and record its nested result in `docs/findings.md`. What is recorded is its in-sample oracle, the same head trained on the parent's in-sample Rasch difficulty: -0.0437 test-like and -0.0557 mix/whole. The acceptance harness reproduces these as -0.0440 and -0.0558 (`results/harness_thresholds.json`, `acceptance`).

### 6.5 Item covariates with a known sign

`paiec/itemcov.py` reads benchmark-agnostic cues off the item dict, each with its sign declared in advance: ordinal difficulty fields, problem position, answer format, text length, stated amount of work and image references. The plan allowed a transferred slope only if a cue had the declared sign *and* agreed with the other benchmarks' mean on 4 of 5 units (F§ "Item covariates with a known sign"). Three facts decide the result:

- **Most cues cannot be tested across benchmarks.** No public item carries a difficulty-named field, and position, format and stated size each vary on one benchmark only.
- **The one universal cue is unstable.** Text length is present everywhere but changes sign: +0.28 on matharena, +0.36 on real_webagents, -0.43 within paper on researchcodebench.
- **Nothing passes the gate.** Through the harness, nested test-like differences are between 0 and +0.00003.

The one cue with real item signal is researchcodebench's stated size ("Approximately 7 line(s) of code"). It has a within-pair r of +0.41, and a forced per-pair slope from B7 gains -0.0022 ± 0.0005 on its pairs, against a placebo of +0.0008. That sits where the gate table says an r of 0.3 to 0.5 covariate should. But it exists on one public benchmark, so it cannot be selected leave-one-parent-out. If one hidden benchmark in seven carried such a cue, the run-level value would be about 0.0003.

*Literature.* Task length in human time predicts agent success on software tasks [Kwa et al. 2025], and stated lines of code is a crude, benchmark-specific proxy for it. Agent Psychometrics finds repository and patch size informative [Ge et al. 2026], but those are not visible in the competition's item input.

### 6.6 The subject side at budgets 0 and 1

Four changes to the priors that act at B0 and B1 were measured against the shipped model (F§ "Subject side at budgets 0 and 1"): harness identity (H), ordered reasoning effort (E), the form of the release-date trend (D: clip, hinge, both, log) and a Student-t level (T).

| selection | folds on | test-like | public R1 bench-first / pair-uniform | no date shift | worst held-out parent |
|---|---|---|---|---|---|
| E vs ship | 4 | -0.00097 ± 0.00033 | -0.00002 / +0.00001 | -0.00005 | 0 |
| H vs ship | 1 | +0.00008 | +0.00005 / +0.00008 | +0.00005 | +0.0004 |
| Dlog vs ship | 3 | +0.00033 | +0.00048 / +0.00056 | +0.00050 | +0.0044 |
| all ten, selected on test-like alone | E 4, Dlog 3 | -0.00052 ± 0.00118 | +0.00047 / +0.00058 | +0.00043 | +0.0044 |

None passes the gates:

- **H cannot be measured leave-one-parent-out.** Harness strings exist on one multi-subject benchmark. Within multi_swebench the harness is the largest attribute (+0.75 for Agentless, -0.75 for MSWE-agent), as Ge et al. find for the scaffold component [Ge et al. 2026]. That is within-benchmark transfer, not what the hidden test asks for.
- **E is switched on in every fold but gains half the bar,** and all of that gain comes through a lower date slope under the synthetic date shift.
- **The date form is decided by the shift, not by data.** On real dates the hinge fits held-out standings best and gains 0.0004 to 0.0007. Under the shift it loses.
- **The Student-t level ties or loses,** at five to nine times the cost.

*Literature.* Observational scaling laws predict benchmark performance from a few capability dimensions shared across model families [Ruan et al. 2024], which supports putting subject attributes in the prior. What no public run can check is how far hidden subjects lie past the public release dates. That, not the attribute model, sets the B0 error.

### 6.7 Acquisition, calibration and the offline bank (legacy replica)

These were measured on the legacy replica and have not been re-measured under the official protocol.

- **Acquisition.** A-optimal acquisition targeting the evaluation pool was worse than random, because it starved the shared difficulty estimate of distinct items. Coverage-first selection moved ALC by -0.0000 ± 0.0013 (F§ "Acquisition and calibration"). The submission ships no `labeling.py`.
- **Calibration.** Post-hoc temperature and slip fitted on four benchmarks and applied to the fifth lose 0.0007 ± 0.0011. The fitted temperatures range from 0.8 to 1.6, so the constant is itself a property of the benchmark.
- **The offline bank.** Of the organisers' 161 inventory benchmarks, 5 publish per-item outputs for at least nine models, and only one publishes them graded (F§ "The offline bank"). The rules also restrict competition-specific training to the public pool.

*Literature.* Active and anchor-point selection choose informative items for a population of models [Li et al. 2024; Vivek et al. 2024; Maia Polo et al. 2024]. Here the evaluation items are a fixed half, the first label is worth 0.9 of a budget, and the level of a new benchmark, not the ranking of items, dominates the error.

### 6.8 Studies in progress

- `TODO(step5)` **attempt probe**: whether a proxy model's attempt at the item carries transferable difficulty signal.
- `TODO(step5)` **Qwen3-4B judge close-out** (§6.2).
- `TODO(step5)` **entropy and hidden-state features**. Token-level entropy predicts task difficulty across 17 agentic benchmarks [Krsteski and Meyer 2026], and pre-generation activations predict a model's *own* success better than length or TF-IDF [Lugoloobi et al. 2026; Cencerrado et al. 2025]. Our subjects are other systems, so the open question is whether a proxy model's signal transfers to them.
- `TODO(step5)` **in-context and pairwise (anchored) comparisons**: an LLM shown a pair's labeled items and asked about a target item.
- `TODO(step5)` **fine-tuning** an encoder on public item difficulty, read through the harness.

### 6.9 Why these fail, in two numbers

1. **Item-side terms cannot reach the budgets that matter most.** B0 and B1 carry 0.3 of ALC's weight, and on the second formative run they were the largest errors (0.237 and 0.195). A centred item offset is zero at B0, and a per-pair slope carries nothing at B1, where centring zeroes the pair's single label. Only the level and subject priors act there.
2. **Item-level headroom is large but needs a signal no public cue has.** On test-like runs it is 0.063 of Brier (the item oracle gains 46% of the pair-rate oracle's Brier). The gate table puts the bar at an honest cross-benchmark correlation of 0.3. Learned text maps reach at most 0.23 leave-one-benchmark-out, and the zero-shot judge clears 0.3 on mathematics only (0.48).

---

## 7 Limitations

- **Five public benchmarks.** Four parents carry every test-like number, and no standard error here covers variation between benchmarks. Leave-one-parent-out ranges are reported for that reason.
- **The test-like regime is tuned in sample to one feedback run.** Its level is not identified by that run (level_mean -1.2, -1.6 and -2.0 are 0.63, 0.37 and 0.46 run sds from the feedback at B0 and B1), and its budget-0 optimism rests on a synthetic date shift that matches the feedback for the legacy prior only. The second formative run's B3 to B31 sit 0.03 to 0.045 above the regime's means for the shipped model (§5.4). Whether that is noise, more central hidden rates or item structure the regime lacks is open.
- **Selection optimism.** Selection and confirmation halves redraw from one catalogue of pairs. Several hyperparameters (`WIDEN`, `G_CAP`, slip, guess, `MAX_LEVELS`, the REFERENCE value of sigma_delta) were set with all five public benchmarks in view. sigma_mu 2.5 lies on the edge of the grid that was scored.
- **Single-subject benchmarks.** hier loses 0.006 to 0.009 against the legacy Predictor on the one public single-subject benchmark, mostly through the widened level prior. The hidden test may hold such benchmarks.
- **Unmeasured cases.** Dense multi_swebench and matharena were not re-measured for hier, so the case the model was built for, and its worst-case latency, are unmeasured. Most studies ran split scope 'pair' only.
- **Library defects found and reported, not fixed.**
  - `fitting.fit_ab`, the legacy Predictor's fit, diverges when its prior sits far from the labels: 5 of 1,598 test-like pair appearances.
  - `testlike.item_oracle` oscillates on 35 of 2,425 stratum appearances, which understated the item oracle on strata (32.7%, not 14.5%).
  - hier's floored Newton fit stopped unconverged once in a collapsed mode (public pair-uniform run 108).
  - The shipped archive predates the corrected multiple-choice floor.
- **Provenance gaps.** A few design inputs are recorded only as outputs: `item_signal.py` and `predictor_check.py` were not rerun, and the inventory classification and the meta-heads study live in session scratch. The legacy-replica results in §6.7 were not re-measured under the official protocol.
- **Formative feedback is noisy.** A single run's ALC has sd 0.02 to 0.04, and the two runs had different subjects. We read feedback only for global hyperparameters.

---

## 8 Reproducibility

### 8.1 Environment

Development ran on an Apple M1 Pro (8 cores, 16 GB) shared with other jobs, under Python 3.10.8 with numpy 1.26.4, scipy 1.10.1, scikit-learn 1.7.2 and pandas 2.2.2. torch 2.6.0 and transformers 4.57.6 were used only for the offline LLM features.

    pip install -e ".[dev]"

The submission needs numpy alone (`requirements.txt`).

### 8.2 Data

measurement-db is gated. Accept the terms on the dataset page, then:

    export HF_TOKEN=...          # or huggingface-cli login
    python -m paiec.fetch        # the core tables of six benchmarks into data/

The organisers' baseline repository (validator and streaming client) has no license and is not redistributed. Clone it:

    git clone https://github.com/aims-foundations/paiec_baseline third_party/paiec_baseline

`pytest` needs neither: it runs on synthetic data (243 test functions).

### 8.3 Rebuilding the submission

    python tools/build_submission.py        # fit prior.json, bake LEVEL, zip dist/paiec.zip, run every check
    python third_party/paiec_baseline/check_submission_zip.py dist/paiec.zip

`dist/paiec.zip` exists only if every check passed (§3.5). The rollback to the first submission's predictor is `--legacy`.

### 8.4 Re-running the experiments

Wall times are as recorded, on the shared machine above. Every results file records its commands and code digests under `passes`, and `--summarise` rebuilds its summary from stored rows.

| result | command | wall time | output |
|---|---|---|---|
| official baselines (§2.3, §5.2) | `python experiments/official_baselines.py --runs 600 --jobs 6` | 12 min, 6 processes | `results/official_baselines.json` |
| hier vs Predictor, ablations (§5.2, §5.5) | `python experiments/hier_eval.py --jobs 8 --skip 'r2\|researchcodebench\|pair\|hier t3 level'`, then `--summarise` | 2 h 16 min, 8 processes, 4 resumed invocations | `results/hier_eval.json` |
| test-like regime check (§4.1) | `python experiments/testlike_check.py --phase tune --jobs 5`, then `--phase check --resume --validate --jobs 5` | about 6 CPU hours | `results/testlike_check.json` |
| level calibration (§5.3) | `python experiments/level_calibration.py --stage S --resume --jobs 6` for S = grid, grid2, r1base, eb, t3, prob, eb, screen, confirm, final; then `--summarise` | 3 h 11 min, 6 processes | `results/level_calibration.json` (19 MB) |
| itemsig layer (§6.3) | `python experiments/itemsig_eval.py --stage run --jobs 4 --rows DIR`, then `--stage oracle`, `--summarise`, `--stage verify`, `--summarise` | 38 min, 4 processes | `results/itemsig_eval.json` |
| acceptance harness (§4.5) | `python experiments/harness.py --stage collect --jobs 2`, `--stage verify`, `--stage table` | 25 + 24 min, about 1 GB | `results/harness_thresholds.json` |
| known-sign cues (§6.5) | `python experiments/itemcov_eval.py --stage signs`, `--stage harness`, `--stage harness --scale train`, `--stage inventory`, `--stage show` | 11 min, 1 process | `results/itemcov_eval.json` |
| subject side (§6.6) | `python experiments/subject_side.py --stage verify`, `--stage diag`, `--stage run --jobs 2` (plus `--configs` passes), `--summarise` | about 4 h 25 min over four invocations, 2 processes | `results/subject_side.json` |
| multiple-choice floor (§5.5) | `python experiments/mcq_floor.py --stage items`, `--stage run`, `--stage summary` | about 50 min, 2 processes | `results/mcq_floor.json` |
| embedding transfer (§6.1) | `python experiments/llm_features.py --download` then `--resume`; `python experiments/emb_transfer.py` | extraction `TODO(record)`; probe 232 s, 480 MB | `results/emb_transfer.json` |
| LLM rating study (§6.2) | `python experiments/llm_rating/analysis.py` | seconds | printed |
| legacy-replica results (§6.7) | `python experiments/ceilings.py`, `ladder.py --seeds 4`, `transfer.py`, `order_sensitivity.py`, `pool_robustness.py`, `acquisition.py --seeds 4` | minutes each | printed |
| offline bank (§6.7) | `python experiments/inventory_scan.py` (needs network: clones repositories) | `TODO(record)` | `results/inventory.csv`, `results/clone_scan.csv` |

Large per-row files (`data/harness_rows`, `data/itemsig_eval_rows`, `data/subject_side_rows`) are gitignored and regenerated by the scripts above.

### 8.5 Seeds and determinism

- Public formative-like runs use `official.sample_run` with seed 0. Test-like runs use seed 2 (primary), seed 3 (sensitivities) and seed 1 (tuning).
- `official`'s seed 0 is the persistent 50/50 split, and other seeds stand in for other hidden splits.
- Bootstraps use 2,000 resamples.
- The replica's acquisition is the platform's sha256 rule, and every hash is a digest, never Python's salted `hash()`.
- LLM features use no sampling, fixed batches and pinned model revisions.
- Every task in the long experiments is a pure function of its key, which makes interrupted runs resumable bit for bit.

### 8.6 Disclosure

- **Training data.** measurement-db only, the public training pool. No external per-item data was used.
- **Formative feedback.** It was used only as described in §3.4: aggregate budget statistics that set three global level hyperparameters. It was never keyed on anonymous benchmark or subject ids, and never used to probe hidden labels.
- **LLM coding assistant.** The code, the experiments, `docs/` and this draft were developed with Claude Code (Anthropic). Commit trailers record Claude Opus 5 and Claude Opus 5.5 as co-authors.
- **Blind LLM difficulty ratings (§6.2).** `TODO(disclose)`: name the model that produced them.
- **Pretrained models.** Qwen3-Embedding-0.6B (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`) and Qwen3-4B-Instruct-2507 (revision `cdbee75f17c01a7cc42f958dc650907174af0554`), for offline research features only. Neither is part of the submission, which uses no pretrained model.
- **Community code.** The organisers' baseline repository (validator, streaming client and reference predictors), read and run locally, not redistributed.

---

## References

- Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *JASA* 88(421).
- Cencerrado, I. V. M., et al. (2025). No answer needed: predicting LLM answer accuracy from question-only linear probes. arXiv:2509.10625.
- Ge, C., Kryvosheieva, D., Fried, D., et al. (2026). Agent psychometrics: task-level performance prediction in agentic coding benchmarks. COLM 2026, arXiv:2604.00594.
- Krsteski, S. and Meyer, C. (2026). Predicting task difficulty without rollouts. arXiv:2608.05797.
- Kwa, T., West, B., et al. (2025). Measuring AI ability to complete long software tasks. arXiv:2503.14499.
- Li, Y., Ma, J., Ballesteros, M., Benajiba, Y. and Horwood, G. (2024). Active evaluation acquisition for efficient LLM benchmarking. arXiv:2410.05952.
- Lugoloobi, W. and Russell, C. (2025). LLMs encode how difficult problems are. arXiv:2510.18147.
- Lugoloobi, W., Foster, T., Bankes, W. and Russell, C. (2026). LLMs encode their failures: predicting success from pre-generation activations. COLM 2026, arXiv:2602.09924.
- Maia Polo, F., Weber, L., Choshen, L., Sun, Y., Xu, G. and Yurochkin, M. (2024). tinyBenchmarks: evaluating LLMs with fewer examples. arXiv:2402.14992.
- Pacchiardi, L., Cheke, L. and Hernández-Orallo, J. (2024). 100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances. arXiv:2409.03563.
- Rasch, G. (1960). *Probabilistic models for some intelligence and attainment tests.*
- Ruan, Y., Maddison, C. J. and Hashimoto, T. (2024). Observational scaling laws and the predictability of language model performance. arXiv:2405.10938.
- Rue, H., Martino, S. and Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. *JRSS B* 71(2).
- Truong, S., Tu, Y., Liang, P., Li, B. and Koyejo, S. (2025). Reliable and efficient amortized model-based evaluation. arXiv:2503.13335.
- Vivek, R., Ethayarajh, K., Yang, D. and Kiela, D. (2024). Anchor points: benchmarking models with much fewer examples. arXiv:2309.08638.
- Zhou, L., Pacchiardi, L., Martínez-Plumed, F., Collins, K. M., et al. (2025). General scales unlock AI evaluation with explanatory and predictive power. arXiv:2503.06378.

`TODO(verify)`: author lists and venues against the published versions. The arXiv identifiers and titles were checked against local copies of the papers.

---

## Appendix A: supplementary tables

### A.1 Formative run 1, per pair

The legacy Predictor on formative run 1. Letters are anonymous benchmarks, relabelled.

| pair | benchmark | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|
| p1 | A | 0.4224 | 0.2851 | 0.2698 | 0.2067 | 0.2085 | 0.2057 |
| p2 | B | 0.4970 | 0.2861 | 0.1394 | 0.0764 | 0.0650 | 0.0513 |
| p3 | C | 0.3583 | 0.2419 | 0.2429 | 0.2437 | 0.2751 | 0.1923 |
| p4 | D | 0.5663 | 0.3234 | 0.1379 | 0.0518 | 0.0184 | 0.0063 |
| p5 | E | 0.3032 | 0.2477 | 0.2479 | 0.2431 | 0.2437 | 0.2472 |
| p6 | F | 0.1728 | 0.1670 | 0.1675 | 0.1665 | 0.1719 | 0.1678 |
| p7 | A | 0.4771 | 0.2939 | 0.1738 | 0.1335 | 0.1285 | 0.1286 |
| p8 | F | 0.1806 | 0.1747 | 0.1760 | 0.1827 | 0.1745 | 0.1749 |
| p9 | G | 0.2519 | 0.2653 | 0.2834 | 0.2360 | 0.2386 | 0.2366 |

Source: F§ "Against the first real formative feedback".

### A.2 hier gain by how many pairs of the target's benchmark a run holds

hier minus Predictor, multi-subject benchmarks, pair-cluster SEs (F§ "Hierarchical model"):

| weighting, split scope | alone | 2 pairs | 3 or more |
|---|---|---|---|
| benchmark-first, pair | -0.0016 ± 0.0014 | -0.0033 ± 0.0011 | -0.0044 ± 0.0010 |
| benchmark-first, benchmark | -0.0009 ± 0.0014 | -0.0020 ± 0.0011 | -0.0029 ± 0.0011 |
| pair-uniform, pair | +0.0003 ± 0.0014 | -0.0017 ± 0.0011 | -0.0024 ± 0.0008 |
| pair-uniform, benchmark | -0.0001 ± 0.0012 | -0.0015 ± 0.0011 | -0.0019 ± 0.0008 |

### A.3 Guarded configurations, both test-like halves and both public weightings

| configuration | selection half | confirmation half | minus Predictor, confirmation | public bench-first | public pair-uniform |
|---|---|---|---|---|---|
| hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 | 0.1578 | 0.1655 ± 0.0031 | -0.0418 ± 0.0024 / 0.0047 / 0.0042 | +0.0008 | +0.0007 |
| hier G mu0 -3.5, sigma_mu 2.5, attr_scale 0.25 (rule's argmax) | 0.1574 | 0.1655 ± 0.0033 | -0.0418 ± 0.0026 / 0.0051 / 0.0045 | +0.0025 | +0.0023 |
| hier EB-cs, tau 2, on G mu0 -3.0, attr_scale 0.5 | 0.1580 | 0.1658 ± 0.0031 | -0.0416 ± 0.0022 / 0.0043 / 0.0038 | +0.0005 | +0.0006 |
| Predictor fix off -1.5, sc 1, va 3 | 0.1630 | 0.1696 ± 0.0027 | -0.0377 ± 0.0016 / 0.0032 / 0.0029 | +0.0026 | +0.0028 |
| hier, fitted defaults | 0.1906 | 0.1949 ± 0.0024 | -0.0124 ± 0.0009 / 0.0019 / 0.0016 | -0.0019 | -0.0017 |
| legacy Predictor | 0.2039 | 0.2073 ± 0.0022 | | 0.2068 | 0.2038 |

Source: F§ "The public guard decides", `experiments/level_calibration.py`. The shipped configuration (mu0 -2.5, attr_scale 0.5) was chosen after this table by the audit (§3.4). Its selection-half row is ALC 0.1608, minus Predictor -0.0431 ± 0.0019 / 0.0032 / 0.0028 (`results/level_calibration.json`).

### A.4 Known-sign cues: sign check

Spearman correlation with honest difficulty; within item_features groups in brackets (F§ "Item covariates with a known sign"):

| covariate | matharena | multi_swebench | real_webagents | researchcodebench | swe_rebench |
|---|---|---|---|---|---|
| position_within | +0.12 ± 0.05 (+0.15) | absent | absent | absent | absent |
| format_score | +0.19 ± 0.13 (+0.07) | constant | constant | 3 items off | 3 items off |
| log_length | +0.28 ± 0.08 (+0.06) | -0.02 ± 0.05 (+0.00) | +0.36 ± 0.08 (+0.29) | -0.10 ± 0.17 (-0.43) | +0.04 ± 0.01 |
| stated_size | absent | absent | absent | +0.50 ± 0.06 (+0.43) | absent |
| image_ref | -0.04 ± 0.09 | +0.05 ± 0.03 (+0.05) | absent | -0.08 ± 0.15 | +0.05 ± 0.01 |

### A.5 Strict run-LOBO

With everything fitted without every benchmark of the run, over runs 0 to 149: Predictor 0.2111 ± 0.0019, hier 0.2089 ± 0.0020. hier minus Predictor is -0.0022 ± 0.0004 / 0.0010 / 0.0006 (95% [-0.0039, -0.0001], stratified [-0.0033, -0.0010]). Both models lose about 0.004 against their target-LOBO lines (F§ "Strict run-LOBO").

---

## Appendix B: open items before submission

- `TODO(record)` Formative run 2's per-pair table and pair count in `docs/findings.md`.
- `TODO(record)` The meta-heads study (`heads.py` and its outputs) into `experiments/` and `docs/findings.md`.
- `TODO(record)` The inventory classification script behind the 161-benchmark class shares.
- `TODO(record)` Wall times for `experiments/llm_features.py`, `experiments/inventory_scan.py` and `tools/build_submission.py`.
- `TODO(disclose)` The model that produced the blind ratings in `experiments/llm_rating/`.
- `TODO(verify)` Rebuild `dist/paiec.zip` with the corrected multiple-choice floor and state which archive each formative run used.
- `TODO(step5)` The five studies of §6.8, each through `experiments/harness.py --stage eval` and its gate.
- `TODO(team)` Authors, affiliations and the public code link. Include measurement-db terms in the release notes.
- `TODO(figures)` See `docs/report/figures.md`.
