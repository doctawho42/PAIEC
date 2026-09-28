# Calibrating a hierarchical response model for a benchmark-level cold start

Technical report for the Predictive AI Evaluation Competition (PAIEC), NeurIPS 2026. **Draft v1, 2026-09-28.**

Authors: TODO(team). Code: TODO(link to the public release). Report written against branch `official-protocol`; every script, results file and test it cites is committed there (the P0 revision in `00bdf04`, the Kaggle record in the commit after it). The submission archive's code is that of `4d2cc4f`.

**Provenance convention.** Every number in this draft is taken from `docs/findings.md` (cited as F§ plus the section title), from `docs/protocol.md` (P§), or from a file in `results/`, and each table names the script that produces it. The few numbers that are derived here are marked "derived", with their inputs. Numbers that exist only in session scratch are not quoted. Appendix C maps each number of §5 to the script, results file and code that produced it, and to the submission archives.

**Names.** The *legacy Predictor* is the predictor of our first submission (`paiec/predict.py`). *hier* is the hierarchical model of §3. The *shipped configuration* is hier with the level prior mu0 -2.5, sigma_mu 2.5, attr_scale 0.5 (`LEVEL` in `submission/model.py`). The *neighbouring configuration* is hier with mu0 -3.0, sigma_mu 2.5, attr_scale 0.25, which the level calibration recommended and an audit replaced before shipping (§3.4). Numbers for the neighbouring configuration are labelled as such and never stand for the shipped one.

---

## Abstract

PAIEC asks for the probability that an AI system (a *subject*) answered a benchmark item correctly, from the subject's and the item's visible attributes and 0 to 31 revealed labels. The score is Brier ALC on benchmarks that never appear in the training data. We rebuilt the organisers' streaming evaluator from their published client. An earlier replica, which scored every pair in one session and showed each target only its own labels, had made pooling item difficulty across subjects look like the main lever (about 0.020 ALC); at formative size under the verified protocol it is worth about 0.0007. Our predictor is a hierarchical Bayesian item-response model refitted from the revealed labels at every checkpoint. Its main lever turned out to be the prior on a new benchmark's level. We set three global hyperparameters of that prior on test-like runs built from public data to resemble the first formative feedback, under a guard on public runs.

What that change is worth depends on the regime it is measured in. In the tuned test-like regime the shipped model improves on our first submission's predictor by 0.040 ALC on the confirmation half and by 0.042 on runs never used in choosing the level (cluster SE 0.004; across the four parent benchmarks the SE is 0.008 to 0.011), and by 0.044 with item groups merged at random. It gains 0.022 to 0.029 in regimes whose level sits nearer a per-pair reading of the feedback, and 0.014 when the regime's synthetic date shift is removed. On public formative-like runs it is level with that predictor (0.0015 to 0.0018 better, within about one cluster SE). Against a Beta(2,2)-smoothed mean, which has no level prior to tune, it gains 0.017 in the tuned regime and 0.009 on public runs.

Two formative runs have been scored on the platform, on different draws with no subject in common: 0.2113 for the first submission and 0.1926 for the shipped model. Run 2 lies 0.84 single-run sds above the shipped model's mean in the tuned regime and 0.44 below it on public runs. Two runs on different draws do not measure an improvement. A reading of both runs' 17 pairs, under a decision rule fixed and hash-locked before the reading was computed but after both runs' tables had been seen, puts the hidden level near the public centre, 1.0 to 1.2 SEs above the tuned regime's, and left the level prior unchanged.

In the studies we completed, every item-side signal either does not transfer to a held-out benchmark or stays below an acceptance harness's gate of 0.002 ALC. That covers text and neural-embedding difficulty maps, language-model judgements, entropy and hidden-state probes, in-context and pairwise prompting, a text-similarity layer, meta-learned heads, known-sign item cues, a fine-tuned encoder, and a 14B model's demand rubric and reasoning attempts run on a free GPU. The 14B's token entropy while it attempts a problem is the one signal that passed its correlation bar (a within-competition Spearman of 0.37 with difficulty on mathematics); it was measured on one benchmark, and at that coverage it does not pass the gate. The harness's gate table puts the bar at an honest correlation of 0.3 to 0.4 with item difficulty for a slope transferred from other benchmarks. We report each negative result with its script.

---

## 1 Introduction

### 1.1 The task

For each subject-item pair the program returns P(correct) from two dictionaries: the subject (eight string fields, such as normalized_name, provider, release_date, harness and reasoning_effort) and the item (item_content, item_features, interactors and an anonymous benchmark_id). It also receives `labeled`, the outcomes revealed so far [P§ Interfaces]. Items of each subject-benchmark pair with at least 80 distinct items are split 50/50 into acquisition and evaluation pools. Labels are revealed from the acquisition pool, and Brier is measured on the evaluation pool at budgets 0, 1, 3, 7, 15 and 31, then averaged equally over pairs:

    ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31

The k-th acquired label counts toward every budget it survives into, so the first label is worth 0.9 of a budget's weight and labels 16 to 31 are worth 0.1 each [P§ Scoring]. B0 and B1 alone carry 0.3 of the weight. The platform's run score is the unweighted mean of the per-pair ALCs: recomputed that way from the returned tables, both of our formative scores reproduce to the reported precision, and weighting pairs by their item counts does not (F§ "Formative feedback, runs 1 and 2").

### 1.2 What makes it hard

**The level of a new benchmark is not in the training data.** The five public benchmarks' levels average -0.56 on the pair-accuracy logit scale, with sd 0.90 (F§ "The step-2 analyses behind the model", `experiments/hier_design/levels.py`). The other benchmarks predict a new benchmark's level poorly: a leave-one-out level misses by 0.86 logit on average (same section). The training mean is still the better centre: centring hier's level prior at 0 instead costs 0.0011 on public runs (§5.5). On the legacy replica, mixing it into the legacy Predictor's prior cost 0.004 (legacy). Read through the lower roots of p(1-p) = B31, the first formative run put the hidden pairs roughly 0.9 logit below the public centre (§5.1). Read per pair against replica pairs, the same run is spread both ways, and the pooled reading of both formative runs puts the hidden level near the public centre, with an SE of about 0.5 logit (§5.1).

**Per-item structure is most of the headroom and does not transfer.** On all 221 public pairs, the Brier score falls from 0.25 (constant 0.5) to 0.176 when the pair's true accuracy is known, and to 0.0433 when each item's true probability is known (F§ "How much there is to win", `experiments/ceilings.py`). But item difficulty predicted from item text has a leave-one-benchmark-out correlation between -0.22 and 0.23 (§6.1).

**Runs are small.** A formative run holds at most 1,000 subject-item pairs across both pools. With at least 80 items a pair, that allows at most 12 pairs [P§ Runs]. The site's example shows 5 pairs, our two scored runs held 9 and 8, and the replica draws 5 to 12. Both scored runs spread their pairs over 7 benchmarks: 5 of run 1's pairs and 6 of run 2's were alone on their benchmark (F§ "Formative feedback, runs 1 and 2"). Pooling across subjects therefore has little to pool.

**No state survives between budgets.** Evaluation workers are recreated at every checkpoint, so `predict` has to be a pure function of `(input, labeled)` [P§ Flow].

### 1.3 Contributions

1. A replica of the official streaming evaluator (`paiec/official.py`), checked decision for decision against the organisers' client. §2 also shows what our first replica got wrong and which conclusions that reversed.
2. `paiec/hier.py`, a hierarchical Bayesian item-response predictor fitted from `labeled` alone. Its predictive integrates the item residual exactly and reads the target's linear predictor along its exact posterior line (§3).
3. `paiec/testlike.py`, test-like runs built from public data (pseudo-benchmarks, about one pair per benchmark, tilted base rates). They were tuned to the first formative feedback's B0 and B1 and sit within half a single-run sd of its other budgets. We use them to calibrate the level prior under a public guard. The rule we kept: *global hyperparameters only, never per-benchmark facts* (§3.4, §4).
4. `experiments/harness.py`, an acceptance harness. It maps a covariate's honest correlation with item difficulty to an expected ALC gain and applies one gate to every idea. With it we report a catalogue of negative results, each with its script (§6).

### 1.4 Headline numbers

All rows are the shipped configuration unless the row says otherwise. Differences are "shipped minus comparator" in ALC, lower is better.

| what | value | source |
|---|---|---|
| formative run 1: legacy Predictor (commit b68492c), 9 pairs | ALC 0.2113, B0 0.359 | `results/formative_feedback.json` (`record.run1`) |
| formative run 2: shipped model, archive of commit ee5085a, 8 pairs, a different draw with no subject in common | ALC 0.1926, B0 0.237 | `results/formative_feedback.json` (`record.run2`) |
| run 2 in the shipped model's single-run distribution | z +0.84 test-like, -0.44 public benchmark-first | `results/ship_confirm.json` (`run2_placement`) |
| minus legacy Predictor, test-like runs never used to choose the level (200-299) | -0.0419 ± 0.0017 / 0.0037 / 0.0034; parent-level -0.038 ± 0.008 | `results/ship_confirm.json`, `experiments/ship_confirm.py` |
| minus legacy Predictor, test-like confirmation half (100-199) | -0.0396 ± 0.0019 / 0.0037 / 0.0034; parent-level -0.035 ± 0.011 | same |
| minus legacy Predictor, groups merged at random (mix/whole, 0-99) | -0.0436 ± 0.0014 / 0.0039 / 0.0031 | same |
| minus legacy Predictor, levels nearer the per-pair feedback reading (level_mean -0.8 and untilted; 40 runs each, library at `4d2cc4f`) | -0.0286 ± 0.0026 / 0.0039 / 0.0035 and -0.0222 ± 0.0025 / 0.0049 / 0.0043 | `results/level_audit.json`, `experiments/level_audit.py` |
| minus legacy Predictor, no date shift (0-99) | -0.0144 ± 0.0012 / 0.0024 / 0.0022 | `results/ship_confirm.json` |
| minus legacy Predictor, public R1 | -0.0017 ± 0.0008 / 0.0021 / 0.0018 benchmark-first (150 runs); -0.0018 ± 0.0010 / 0.0017 / 0.0016 pair-uniform (100) | same |
| minus smoothed Beta(2,2) | -0.0169 ± 0.0007 / 0.0020 / 0.0019 test-like (0-199); -0.0086 ± 0.0010 / 0.0030 / 0.0021 public benchmark-first | same |
| shipped model, test-like runs 0-299 | ALC 0.1658 ± 0.0018 | same; `results/subject_side.json` |
| live leaderboard, 2026-09-24 | organisers' entry 0.1801, best entry 0.1172 | F§ "Against the live leaderboard" |

(± run SE / cluster-bootstrap SE / stratified SE; the parent-level mean ± SE is taken across the four parent benchmarks; see §4.3. Leaderboard entries and formative runs are single noisy draws: a formative run's ALC has a single-run sd of 0.026 to 0.034 in the replica's regimes.)

---

## 2 Data, protocol and our replica

### 2.1 Data

We use only measurement-db, the organisers' public training release (Hugging Face `aims-foundations/measurement-db`, gated; revision `bc8204d8…`, §8.2; the dataset card lists CC-BY-SA-4.0). It holds 571,921 responses over 287 subjects and six benchmarks [P§ Data]. Five of the benchmarks are binary. mmdocrag is fraction-valued (only 4.8% of its responses are 0 or 1) and is excluded, as the test excludes non-binary benchmarks. matharena's 0.1% non-binary tail is dropped. Keeping pairs with at least 80 distinct items leaves 221 pairs and 225,843 responses:

| benchmark | pairs | distinct items | mean accuracy | item_features key (levels, share of item variance) |
|---|---|---|---|---|
| multi_swebench | 82 | 2,126 | 0.219 | lang (8, 0.05) |
| matharena | 81 | 1,633 | 0.607 | competition (25, 0.50) |
| researchcodebench | 31 | 212 | 0.354 | paper (20, 0.43) |
| real_webagents | 26 | 233 | 0.330 | website (12, 0.24) |
| swe_rebench | 1 | 6,306 | 0.479 | none |

Sources: pair counts [P§ Data]. Distinct items from `results/harness_thresholds.json` (`meta.oracle_info`) and, for swe_rebench, P§ Data. `experiments/data_counts.py` re-counts the pairs, responses and distinct items from `data/` and gets the same numbers (`results/data_counts.json`). Mean accuracy from the R2 table in F§ "Under the official protocol" and, for swe_rebench, F§ "How much there is to win". Variance shares from F§ "The step-2 analyses behind the model" (`experiments/hier_design/item_signal.py`, recorded output, not rerun). matharena's 1,633 items in eligible pairs span 25 competitions. The 0.50 share was estimated on all 1,751 items with a binary response, which span 27 (imc_2025 and miklos_2025 occur only on subjects below the 80-item floor; `results/data_counts.json`, `matharena`).

Properties that shaped the method:

- **Few subjects cross benchmarks.** Only 22 of 287 subjects appear on more than one benchmark. `interactors` is empty everywhere [P§ Data].
- **Repeats are uneven.** 99.8% of matharena items have repeated responses, 99.6% of swe_rebench's, 0.4% of multi_swebench's [P§ Data].
- **Some subject attributes live on one benchmark.** Harness strings appear on only one multi-subject benchmark, multi_swebench (81 of 82 pairs, 12 strings). reasoning_effort appears only on matharena (23 of 81 pairs) (F§ "Subject side at budgets 0 and 1").
- **matharena items are partly textless.** A third of them have no task text ("See image", or fragments of a system prompt). Its 336 Kangaroo items (all in eligible pairs; `results/data_counts.json`) are five-option multiple choice with the options in an image (F§ "What transfers between benchmarks"; F§ "The multiple-choice floor, corrected").
- **Raw solve rates confound difficulty.** Items attempted only by recent subjects look easy, so difficulty targets are Rasch estimates with subject ability divided out (`paiec/rasch.py`; F§ "What transfers between benchmarks").

The hidden benchmarks are drawn from the organisers' inventory of 161 candidate benchmarks, which were assigned at random to the training and test pools before curation. Keyword rules on the titles alone (no description, paper or item is read) class 108 of the 161 (67%) as evaluations of AI systems. Of those 108, 39% are text QA, 30% images, 15% agents, 6.5% code, 4.6% video, 2.8% math and 2.8% audio. On 80 labelled titles from the organisers' full sheet, the rules agree on which titles are evaluations for 82.5%, and on the category for 38 of the 40 that both call evaluations (95%). The 80 labels were assigned by an AI agent reading each title, not by a person (`experiments/inventory_classes.py`, `results/inventory_classes.json`; F§ "What this could do on the hidden test"). The public benchmarks are therefore not a representative sample of the hidden task types.

### 2.2 The protocol, as verified

We read the organisers' streaming client (`tools/streaming_ingestion.py` in the public baseline repository, commit `82d330dd`, §8.2) alongside the competition page [P§]. Six properties of the protocol matter for this work:

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
- **The legacy Predictor's margin is small.** It beats a Beta(2,2)-smoothed mean by only 0.0026 ± 0.0009 (pair-cluster SE) with a strict prior, or 0.0069 ± 0.0016 with a target-leave-one-benchmark-out prior.
- **Pooled difficulty adds about 0.0007 at formative size.** It needs 64 distinct labeled items on a benchmark, and a formative run has about two pairs per benchmark.

The two setups differ in run size and in split handling as well as in the information set, and this comparison does not separate them. What it shows is that at formative size, under the verified protocol, pooled difficulty is not the lever. The legacy numbers ranked predictors under a different information set. Nothing in this report is quoted from the legacy replica except where marked "legacy".

### 2.4 Unknowns kept as parameters

The replica treats what the client leaves open as parameters, not assumptions [P§ Still unknown]. The main one is `split_scope`: whether the 50/50 split is drawn per pair or per benchmark item.

- Under per-pair splits, other subjects' acquired labels land on most of a target's evaluation items on dense runs (83% to 99.5% at B31).
- Under per-benchmark splits, that coverage falls to 0% to 5%.

On dense runs (every pair of one benchmark), the legacy Predictor's gain over the smoothed mean on matharena is -0.0354 under per-pair splits and -0.0159 under per-benchmark splits (F§ "One benchmark, every pair (R2)"). hier depends on the scope differently. On dense researchcodebench it beats the legacy Predictor by 0.0042 under per-pair splits and loses by 0.0040 under per-benchmark splits (F§ "Hierarchical model", R2). Dense matharena and multi_swebench were not run for hier. We report both scopes for the baselines and for the comparison of hier with the legacy Predictor (§5.2). The later studies ran split scope 'pair' only (§7).

Other open points are listed in `docs/protocol.md`:

- which recorded response is revealed for a repeated item;
- the per-call timeout;
- how a formative run picks and cuts its pairs.

The formative feedback's item counts suggest the platform splits after cutting, with a floor near 88 kept items (`paiec/testlike.py` docstring).

### 2.5 How the replica was verified

- **The default policy matches.** `default_policy` is checked decision for decision against the organisers' `run_streaming` [P§ Interfaces].
- **Argument copies do not change results.** On one run, the platform-like argument copies with 16 simulated workers give per-pair Brier and ECE bit-identical to the fast path for all six predictors (F§ "Under the official protocol").
- **The replica agrees with the analytic formula.** The empirical mean's ALC follows from base rates: about 0.025 + 1.2118 E[p(1-p)]. Once the split and stream order are salted per run, the replica's residual against the exact expectation is -0.0005 ± 0.0006 (F§ "The empirical mean's ALC is a function of base rates").
- **The invariants are pinned by tests.** `tests/test_official.py` (34 test functions) covers them. The test suite has 362 test functions over 17 files (542 collected tests); it runs on synthetic data and needs no download.

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

**Item residuals are integrated, not maximised.** A joint mode over all effects is the classic penalised-quasi-likelihood failure [Breslow and Clayton 1993], with an item variance near 9.5 (the shipped sigma_d² + sigma_g² = 2.671² + 1.542²) and one label per item. The item effects absorb each label and the fit lands where the logistic is flat. In a pinned test with item variance 8.7, a pair with 160 of 200 successes is predicted at 0.70 on a new item that way, and at 0.79 once the item residual is integrated (`tests/test_hier.py`). The vector x therefore holds every component except the e_i. Each item's likelihood is the one-dimensional integral over e_i: adaptive Gauss-Hermite with 20 nodes, or a fixed 201-node trapezoid on floored items, whose integrand can be bimodal. Newton with a line search finds the mode of the joint log posterior of x.

**The target is read along its line.** The Laplace (Gaussian) posterior of the target's linear predictor a'x under-reacts to a pair's first labels, by 0.013 to 0.02 in probability after one to seven labels, because the mode of a skewed posterior sits nearer the prior than its mean. So a'x is read on a grid along the Gaussian conditional mean, with the exact log posterior at each point. This is the Gaussian strategy of INLA [Rue, Martino and Chopin 2009]. For one pair alone it is the exact marginal. With other subjects on the benchmark, the line ignores their skew. E[sigmoid] is then taken by a 171-node trapezoid, exact to 1e-9. The docstring of `paiec/hier.py` documents the remaining approximation error case by case. On cases drawn from the model it moves Brier by about 0.001 at B1 and by at most 1e-4 from B7 on.

**Cost.** Without floors the log posterior is concave and the mode unique. A floored item's success term is not concave, so Newton's matrix is shifted to positive definite when needed. Where Newton stops short on a posterior that holds a floored success, trust-region Newton from three starts settles the fit (`Problem._settle`, commit `4d2cc4f`; F§ "Floored fits"). Where Newton converges it changes nothing, bit for bit. Fits still unconverged after it are counted.

### 3.3 Offline prior and empirical-Bayes hyperparameters

`paiec/prior.py` fits one item-level model per public benchmark, `logit p = mu_b + t_s - g_i - e_i`. It uses each subject's first recorded response per item (the one acquisition reveals) and finds the variances by Laplace-EM. That single fit supplies two things:

- **The subject standings t_s**, centred within the benchmark. An attribute ridge predicts them from provider, release date (linear in days since 2023), model size parsed from the name, reasoning-effort dummies and a harness-present flag (`paiec/subjects.py`).
- **The levels, item variance and group share**, which `fit_hyper` pools across benchmarks.

Each hyperparameter the included benchmarks cannot identify falls back to a fixed `REFERENCE` value that no fit went into. For example, mu0 = 0 when fewer than three levels are available.

Every experiment refits the prior and the hyperparameters *without the target's parent benchmark* (§4.2). The shipped bundle is fitted on all five public benchmarks, with the level prior from `LEVEL` in `submission/model.py`, and matches `submission/prior.json` as built by `tools/build_submission.py` (F§ "Verdict: ship hier with the level moved down" lists the fitted fields):

| mu0 | sigma_mu | attr_scale | sigma_theta | sigma_delta | sigma_attr | sigma_d | sigma_g | slip | guess |
|---|---|---|---|---|---|---|---|---|---|
| **-2.5** | **2.5** | **0.5** | 0.1 | 2.382 | 1.018 | 2.671 | 1.542 | 0.01 | 0.5 |

The three bold fields are the level prior set for the hidden test (§3.4). The rest are the empirical-Bayes fit.

The subject's cross-benchmark identity carries almost nothing beyond its attributes. The shared part of a named model's attribute residual across benchmarks, tau2_res, is estimated at -0.002 on the public data and clipped to [0.01, 0.2]. That puts sigma_theta at its floor of 0.1 and the link weight across benchmarks at 0.0018 (`paiec/hier.py` docstring; F§ "The step-2 analyses behind the model").

### 3.4 Calibrating the level prior for the hidden test

**The signal.** The first formative feedback (§5.1) showed the legacy Predictor confidently optimistic at budget 0. It predicted near 0.75 on pairs whose rates were low. Two readings pointed the same way when the level was chosen:

- Reading each pair's B31 Brier as irreducible noise, p(1-p) = B31, and taking the lower roots puts the hidden pairs at a mean of about -1.6 (sd 1.5) on the plain logit of the pair's rate. On the same scale the public R1 pair appearances average -0.74 (sd 1.50; `experiments/hier_design/levels.json`, `r1`; F§ "Against the first real formative feedback"). On the continuity-corrected scale of §5.1's pooled reading the two are -1.51 and -0.71 (`results/formative_feedback.json`).
- Independently, *if* the organisers' leaderboard entry is the empirical-mean baseline, inverting its ALC puts E[p(1-p)] near 0.128, against 0.181 on public runs (F§ "Against the live leaderboard").

The lower roots are one reading. Matched against replica pairs, two of the nine pairs (p6 and p8, one benchmark) read as high-rate and a third is ambiguous (F§ "What actually shipped, after the audit"), and the pooled reading of both formative runs is more central still (§5.1).

**The regime.** We cannot fit to hidden labels, and one run of nine pairs identifies little. So we built test-like runs from public data that resemble the feedback (§4.1), and chose the level prior on them.

**The candidates.** Grids over (mu0, sigma_mu, attr_scale) for hier, with Gaussian and Student-t levels. A pair-level empirical-Bayes re-estimate of the level at every checkpoint. A level fix for the legacy Predictor. All are listed in F§ "Calibrating for the hidden test".

**The selection.** The best mean ALC on the selection half of the test-like runs (runs 0 to 99), among configurations that lose at most 0.003 ALC against the legacy Predictor on *both* public weightings (the guard). Configurations were confirmed on runs 100 to 199.

**The rule we kept.** The feedback may set *global* hyperparameters of the level distribution, here three numbers: mu0, sigma_mu and attr_scale. It never sets anything keyed on an anonymous benchmark or subject id, and no prediction was shaped to probe hidden labels. The level grid's selection used nothing from the feedback beyond the regime's defaults. The audit below then read run 1 per pair and chose between two guarded configurations partly on that reading: a use beyond the regime's defaults, still of global hyperparameters only. §8.6 lists every use of the feedback. This follows the competition's conduct rule against extracting test data and identifying anonymous ids.

**What shipped.** The rule's argmax was mu0 -3.5 at attr_scale 0.25. The recommendation took the neighbouring configuration, mu0 -3.0, on a tie-break by the guard's margin (0.0004 behind on selection, level on confirmation). An audit then moved the shipped choice to the milder guarded configuration, **mu0 -2.5, sigma_mu 2.5, attr_scale 0.5** (F§ "What actually shipped, after the audit", `experiments/level_audit.py`, `results/level_audit.json`). The audit's reasons:

1. Two of the feedback's pairs are more plausibly high-rate. Their B0 was already close to their B31 under an optimistic prior, and 70% and 80% of their nearest replica pairs sit above a rate of 0.5. So the hidden levels look spread both ways. Against the legacy Predictor, the aggressive setting loses 0.014 on test-like pairs at rates 0.5 to 0.7 and 0.070 at 0.7 and above, and 0.015 to 0.017 on public matharena.
2. The two configurations sit on a flat plateau. The milder one gives up 0.0030 ± 0.0009 on the selection half (ALC 0.1608 against 0.1578) and 0.0022 ± 0.0011 on the confirmation half (0.1677 against 0.1655). It gains 0.0023 ± 0.0005 and 0.0024 ± 0.0004 on the two public weightings (cluster SEs), and has no worse measured worst case.
3. The "held-out" half redraws runs from the same catalogue of pairs: it repeats 82.2% of the selection half's pairs and 99.5% of its clusters. It measures redrawing, not generalisation to unseen parents.

**After run 2.** The audit asked for the next feedback to be read per pair, pooled with run 1, before refitting the level distribution once. Its decision rule, which could change mu0 and sigma_mu only, was written and hash-locked before the reading was computed, but after both runs' feedback tables had been seen, so the reading is not blind (§5.1). It found the pooled level 1.0 to 1.2 SEs more central than the tuned regime's, below the rule's bar of 2 SEs, and proposed nothing (§5.1). LEVEL is unchanged since run 2.

### 3.5 Run-time engineering

- **Purity and caching.** `predict` is a pure function of `(input, labeled)`. Fits are cached under a content fingerprint of `labeled`. Items and subjects are keyed on digests of all their visible fields, because the official input carries no ids and text prefixes collide.
- **Failure handling.** Nothing raises. A failed fit falls back to the prediction without labels, then to 0.5. An unusable `prior.json` falls back to the level prior without a subject prior.
- **Packaging.** The archive ships `model.py`, `prior.json` and the run-time modules renamed to `paiec_rt/`, so that no platform-side `paiec` can shadow them. Shipped code imports only the standard library and numpy at module level, and BLAS is held to one thread at run time.
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
| sensitivities | level_mean -1.2 and -2.0 instead of -1.6; groups merged at random with no strata ("mix/whole"); no date shift; and, for the audit, level_mean -0.8 and no level tilt (seed 5) | robustness of every hidden-test choice |

Sources: F§ "Under the official protocol"; F§ "Hierarchical model"; F§ "Calibrating for the hidden test"; F§ "What actually shipped, after the audit".

**How test-like runs are built** (`paiec/testlike.py`).

- *Pseudo-benchmarks.* The five public benchmarks are too few and too central, so each is cut into pseudo-benchmarks: item_features groups sorted by difficulty and merged, difficulty strata (cross-fitted, so a subject's own labels never choose its items' stratum), whole parents, and random chunks for the single-subject benchmark. Each gets its own anonymous benchmark_id. Every label is a real recorded response. What is new is the grouping of items and, under the date shift below, the release and access dates the predictor sees.
- *Run shape.* A run draws 5 to 12 pairs. With some probability the next pair joins a pseudo-benchmark already in the run; otherwise it opens a new one. This reproduces the feedback's about one pair per benchmark.
- *Level tilt.* The draw is tilted toward a target distribution of pair accuracy logits (mean -1.6, sd 1.5). The tilt realises part of that shift. The 300 check runs have a mean pair logit of -1.29, with sd 1.70 (cluster SE 0.16; `results/testlike_check.json`). The sensitivity targets -1.2 and -2.0 realise -1.10 and -1.56, and the audit's -0.8 and untilted regimes -0.77 and -0.59.
- *Date shift.* Visible release and access dates are shifted 1.25 years later. This reproduces the feedback's budget-0 optimism for the legacy prior's linear date term.

**How well the regime matches.** The defaults are the grid point closest to the feedback's B0 and B1. That agreement is in sample. The budgets not used in tuning (B3 to B31) sit at z +0.09 to +0.50 of the replica's single-run sd without tuning. On public R1 the feedback's B0 is at z +8.5. ALC alone does not separate the regimes: the legacy Predictor scores 0.2073 on test-like runs, 0.2075 on R1 and 0.2113 in the feedback (`experiments/testlike_check.py` docstring; `results/testlike_check.json`).

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

None of these covers variation *between* benchmarks, and four parents carry every test-like number. We therefore also report per-parent and leave-one-parent-out ranges and, for the headline, the **parent-level** mean ± SE: the mean and SE of the four multi-subject parents' appearance-weighted means (`results/ship_confirm.json`). On the headline it is about 2.7 times the cluster SE (§5.3). When many options are compared with one default, we state the Bonferroni bar (for example 2.95 SEs two-sided for sixteen comparisons; F§ "Ablations and sensitivities").

### 4.4 Selection discipline

- **Selection and confirmation halves.** Candidates are chosen on runs 0 to 99 and confirmed on runs 100 to 199 (§3.4). The halves redraw from one catalogue of pairs, so they measure redrawing, not new parents. Test-like runs 200 to 299 were scored for the shipped model by later studies and never used to choose the level.
- **Guards.** Every hidden-test choice must hold on both public weightings.
- **Gates fixed in each study's plan.** A component ships only if all four hold (F§ "Subject side at budgets 0 and 1"):
  1. nested selection switches it on in at least 3 of 4 folds;
  2. the nested test-like difference is at most -0.002;
  3. no held-out parent is above +0.002;
  4. neither public weighting loses more than 0.001.

  These gates live in the scripts' plan blocks and were committed together with the results, so the repository does not show that they preceded each study's runs. The one reading of formative feedback made after run 2 had its decision rule written into its results file, with a timestamp and a hash, before the reading was computed, and a test checks that the stored rule is the script's and predates the first reading. It fixes the rule, not what had been seen: both runs' feedback tables, run 2's per-pair Brier included, were known when it was written. The reading code was then revised twice in its descriptive outputs, and the earlier versions, rebuilt and re-run, give the same decision (§5.1).
- **Provenance.** Results files record each invocation's command, wall time and digests of the code it ran (`passes`), and `--summarise` rebuilds every summary from stored rows. Where a script changed after one of its stages had written results (`formative_feedback.py`, `level_audit.py`), `experiments/script_revisions.py` rebuilds each earlier version from a log of the edits and checks it against the recorded digest.

### 4.5 The acceptance harness and its gate table

Every later item-side idea reduces to one number per item. `experiments/harness.py` scores any such covariate x the same way, against the model that ships (F§ "Acceptance harness"). It adds a centred logit offset:

    q_i = sigmoid( logit p_i + cap(beta (x_i - mean of x over labeled items)) ),  cap(o) = 4 tanh(o/4)

Here p_i is the shipped hier's prediction, fitted leave-one-parent-out. Centring keeps hier's level, so only the ordering of items comes from x, and at B0 the covariate does nothing. The slope beta is either:

- **transferred**: one coefficient per budget, fitted on the other parents; or
- **per-pair**: a MAP from the target pair's own labels, under a zero-mean prior of s per standard deviation of x.

The harness first reproduces the in-sample oracle that the meta-heads study scored (§6.4): -0.0437 test-like and -0.0557 mix/whole as target (`results/heads_eval.json`, `experiments/heads_eval.py`), against -0.0439 and -0.0558 as reproduced. It then builds the **gate table** from degraded oracles, x = r z + sqrt(1 - r^2) noise, where z is an *honest* difficulty fitted on other subject folds only. For each r the table records what a covariate with that honest correlation would score through the gate, averaged over 8 noise draws, and how many single draws pass:

| honest r (within-pair r, test-like) | transferred slope, nested: test-like (cluster SE, draw sd) | mix/whole | R1 bench-first | R1 pair-uniform | draws passing | per-pair slope, nested: test-like | draws passing |
|---|---|---|---|---|---|---|---|
| 0 (0.00) | -0.00002 (0.00001, 0.00006) | +0.00001 | -0.00001 | +0.00000 | 0/8 | -0.00000 | 0/8 |
| 0.1 (0.08) | -0.00010 (0.00005, 0.00043) | -0.00027 | -0.00034 | -0.00044 | 0/8 | -0.00000 | 0/8 |
| 0.2 (0.16) | -0.00083 (0.00015, 0.00086) | -0.00161 | -0.00191 | -0.00245 | 1/8 | -0.00010 | 0/8 |
| 0.3 (0.25) | **-0.00255** (0.00032, 0.00068) | -0.00457 | -0.00517 | -0.00639 | **6/8** | -0.00032 | 0/8 |
| 0.4 (0.33) | -0.00462 (0.00052, 0.00081) | -0.00805 | -0.00906 | -0.01109 | 7/8 | -0.00136 | 0/8 |
| 0.5 (0.42) | -0.00738 (0.00077, 0.00090) | -0.01246 | -0.01399 | -0.01699 | 8/8 | **-0.00255** | **8/8** |
| 0.7 (0.61) | -0.01528 (0.00143, 0.00091) | -0.02392 | -0.02693 | -0.03213 | 8/8 | -0.00768 | 8/8 |
| honest oracle | -0.0360 ± 0.0010 / 0.0030 / 0.0029 | -0.0484 | -0.0539 | -0.0619 | passes | -0.0223 | passes |

Source: `results/harness_thresholds.json` (`thresholds.honest`, `acceptance`), `python experiments/harness.py --stage table`; 300 test-like, 150 mix/whole and 100 + 100 public runs, on rows collected with the library of the current archive (corrected multiple-choice floor and floored-fit fix; F§ "Acceptance harness", Provenance). The table built before that re-collection, on the legacy rows, differs from this one by at most 0.0002 in the test-like column and 0.0004 elsewhere; its transferred line at r = 0.3 was -0.00238.

So a covariate needs an honest correlation of about 0.3 to 0.4 with item difficulty if its slope can be transferred from other benchmarks: a single covariate at r = 0.3 passes on 6 of 8 draws, at 0.4 on 7 of 8, and from 0.5 on every draw. It needs about 0.5 if the slope must be learned per pair from at most 31 labels. An uninformative covariate forced on with a per-pair slope (s = 0.5) costs +0.0009 from B1 and +0.0007 from B7. One noise draw of a degraded oracle moves its test-like difference by more than its cluster SE (draw sd 0.0007 against 0.0003 at r = 0.3), so a real covariate, which is one draw, should be read against that spread as well. The gate uses the thinnest regime: at r = 0.2 the transferred line gains 0.0016 on mix/whole and 0.0019 to 0.0025 on public runs, but 0.0008 on test-like runs, so the gate leans toward false negatives.

---

## 5 Results

### 5.1 Formative feedback

Two formative runs have been scored on the platform, each on its own draw of pairs, and each is one noisy draw (F§ "Formative feedback, runs 1 and 2"; `experiments/formative_feedback.py`, `results/formative_feedback.json`; the organisers' tables, byte for byte, in `results/formative/`).

| run | model; archive (sha256) | pairs (benchmarks, subjects) | B0 | B1 | B3 | B7 | B15 | B31 | ALC | mean ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | legacy Predictor, commit b68492c; `8e28d930…` | 9 (7, 9) | 0.3589 | 0.2539 | 0.2043 | 0.1712 | 0.1693 | 0.1568 | 0.2113 | 0.170 |
| 2 | hier, LEVEL -2.5 / 2.5 / 0.5, commit ee5085a (old multiple-choice floor, solver before the floored-fit fix); `2c64eaad…` | 8 (7, 8) | 0.2368 | 0.1947 | 0.1963 | 0.1836 | 0.1785 | 0.1833 | 0.1926 | 0.100 |

Recomputed as the unweighted mean of per-pair ALCs, the budget tables give 0.2112999889 and 0.1926232375, which round to the reported 0.2113 and 0.192623. The per-pair tables are in Appendix A.1. Run 2's archive holds the modules of commit ee5085a byte for byte. That is the library of §5.4's rows, before the corrected floor and the floored-fit fix. The archive now selected (sha256 `4a882cc7…`) differs by those two changes, worth -0.0003 on test-like runs (§5.5, Appendix C).

**No pair recurs.** No subject appears in both runs, so no (subject, benchmark) pair recurs and the platform gives no paired comparison of the two predictors. Both runs drew from the same seven hidden benchmarks with different subjects. The difference of two independent runs' ALC has an sd of about 0.04, against the observed 0.019. Two runs on different draws do not measure the change from the legacy Predictor to hier.

**Run 1 lost its score at budget 0.**

- It scored 0.3589 where a constant 0.5 scores 0.25, with ECE up to 0.75. Answering 0.5 at B0 and B1 alone would have given about 0.1996.
- The relative attribute standing placed strong 2025-26 subjects near p = 0.75. Read through the lower roots of p(1-p) = B31, the pairs' rates were 0.006 to 0.45.
- On the plain logit of the pair's rate those lower roots average -1.6 with sd 1.5, where the public R1 pair appearances average -0.74 with sd 1.50 (`experiments/hier_design/levels.json`, `r1`). Matched against replica pairs, p6, p8 and p9 read as high-rate instead, and the nine pairs average -0.98 with sd 2.05 (plain logit; F§ "Formative feedback, runs 1 and 2").

This is the observation behind §3.4.

**Run 2, with the shipped level prior, lost far less at budget 0, on different pairs.** Its B0 was 0.237 against run 1's 0.359. That is 0.02 above the shipped model's test-like mean (0.2165) and level with its public benchmark-first mean (0.2350, §5.4). From B1 its curve is flat: from B3 to B31 it sits 0.03 to 0.045 above the test-like means (0.1645 to 0.1386). Placed in the shipped model's single-run distributions (`results/ship_confirm.json`, `run2_placement`):

| regime (runs) | mean ALC | single-run sd | run 2's z | share of runs ≥ 0.1926 | run 2's z by budget, B0..B31 |
|---|---|---|---|---|---|
| test-like (300) | 0.1658 | 0.0318 | +0.84 | 0.20 | +0.98 +0.15 +0.86 +0.89 +0.98 +1.40 |
| test-like, mix/whole (200) | 0.1711 | 0.0329 | +0.65 | 0.245 | +1.12 +0.05 +0.72 +0.70 +0.72 +1.15 |
| test-like, no date shift (100) | 0.1585 | 0.0335 | +1.02 | 0.18 | +1.00 +0.45 +0.89 +1.11 +1.20 +1.67 |
| public R1, benchmark-first (150) | 0.2051 | 0.0285 | -0.44 | 0.67 | +0.06 -0.78 -0.49 -0.46 -0.27 +0.23 |
| public R1, pair-uniform (100) | 0.2020 | 0.0261 | -0.36 | 0.68 | -0.24 -0.67 -0.37 -0.32 -0.13 +0.50 |

One run cannot discriminate the regimes. Run 2's profile is at least as close to plain public runs as to the tuned regime: every budget is within 0.8 sd of public R1, while its B31 is 1.4 sd above the test-like mean.

**What the shipped configuration was expected to score.** Before run 2, the audit's per-pair matched estimate on run 1 was about 0.178 (0.169 to 0.183) for the neighbouring configuration; it replaced that configuration's average-shift estimate of 0.167. The shipped configuration's own estimate, by the same estimator, is 0.1812 (K = 15 neighbours) and 0.1830 (K = 40) on the audit's pool, where the neighbouring configuration scores 0.1824 and 0.1839, and 0.1774 and 0.1779 on the shipped model's own replica rows. Across robustness variants on pools that hold test-like runs it ranges from 0.1766 to 0.1824; pools of public runs alone give 0.205, on matches about five times further away (F§ "Formative feedback, runs 1 and 2"; `experiments/level_audit.py` agrees, 0.1765 and 0.1789). Run 2 is a different draw, so these are predictions for a like-sized run, not for its pairs. Its 0.1926 lies 0.015 above the shipped rows' estimate (z +0.48 of a test-like single-run sd). It did better than predicted at B0 (by 0.008) and worse from B3 on (by 0.017 to 0.022 at B3 to B15 and 0.034 at B31). Run 2's pairs sit nearer a rate of 0.5 than run 1's (mean B31 0.183 against 0.157).

**The pooled reading of both runs, under a rule fixed in advance.** Before any matching of run 2's pairs or any pooled reading was computed, we wrote into the results file what the reading could change and when (F§ "Formative feedback, runs 1 and 2"; written 2026-09-28 05:06:59 UTC, sha256 `0de18448…`; the first reading ran at 05:13:48). This fixes the rule, not what had been seen: both runs' feedback tables, run 2's per-pair Brier included, and the audit's scratch outputs for run 1 were known when it was written. It may change one global hyperparameter only, the level distribution of a new benchmark (mu0 and sigma_mu of LEVEL). A candidate is proposed only if, for both K = 15 and K = 40, the 17 pairs' mean (or sd) level differs from the tuned regime's realised -1.29 (sd 1.70) by more than 2 SEs that resample the 7 benchmarks, and only if a guard finds no bias from the reading itself. Even then it would have to pass the level calibration's gate on fresh test-like runs set to the reading before shipping.

| reading (continuity-corrected pair logit) | mean (SE) | sd (SE) |
|---|---|---|
| runs 1 and 2, 17 pairs, K = 15 | -0.65 (0.51) | 1.80 (0.36) |
| runs 1 and 2, 17 pairs, K = 40 | -0.78 (0.51) | 1.75 (0.34) |
| tuned test-like regime, realised | -1.29 (0.16) | 1.70 |
| public R1, realised | -0.71 (0.12) | 1.42 |

The pooled mean is 1.24 (K = 15) and 1.00 (K = 40) SEs above the regime's, and the sd 0.29 and 0.14 SEs above. The bias guard did not trip. **No candidate; LEVEL stays.** The shipped LEVEL implies, on this scale, a mean of -1.15, a between-benchmark sd of 1.15 and a within sd of 1.10; the 17 pairs give a between sd of 1.20 to 1.49 (90% interval from 0 to about 1.7) and a within sd of 1.01 to 1.40, which is consistent. Two components are not supported (BIC by 0.68 and 0.38, bar 2): the only structure is two pairs near a zero rate. Descriptively, run 2's B0 excess over B31 does not sit mainly on low-rate pairs (0.047 on the 4 pairs read below a rate of 0.5, 0.078 on the 3 above, at K = 15), and by the rule fixed in advance that changes nothing.

**The reading was run four times.** The read stage ran at 05:13:48, 05:15:57, 05:19:08 and 05:20:19 under three versions of the script (digests `5099ef81…`, `321c189e…` and `93cb6a84…`, the last being the file in the repository), and the stored reading is the last. The rule's text was byte-identical throughout: the read stage refuses to run when the stored rule and the script's differ. The two edits after the first reading changed descriptive outputs only. The first extended the split of run 2's B0 excess to K = 40 and kept apart q3, whose B31 has no real root (the first version split the pairs 4 and 4 at K = 15, 0.047 below a rate of 0.5 and 0.060 at or above it, q3 counted above), and added explanatory notes. The second added the neighbours' match distance to the robustness variants. The edit between the preregistration and the first reading touched the record stage only. Rebuilt from a log of the edits and re-run, the preregistration's version and both earlier reading versions give the stored decision, z values and bias guard exactly, and the same value in every one of the 1,502 to 1,559 fields they share with the stored reading (`experiments/script_revisions.py`, `results/script_revisions.json`).

What the two runs support is modest. The pooled level is near the public centre, and more central than the regime in which the headline gain was measured, but within the noise of 7 benchmarks. The regimes nearer that reading give the shipped configuration 0.022 to 0.029 over the legacy Predictor, not 0.042 (§5.3).

**The leaderboard.** On 2026-09-24 the organisers' entry scored 0.1801 and the best entry 0.1172, both single draws.

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

| weighting, split scope | hier minus legacy Predictor | 95%, pair-cluster | 95%, stratified |
|---|---|---|---|
| benchmark-first, pair (primary) | -0.0024 ± 0.0004 / 0.0013 / 0.0008 | [-0.0047, +0.0002] | [-0.0040, -0.0008] |
| benchmark-first, benchmark | -0.0010 ± 0.0003 / 0.0014 / 0.0009 | [-0.0035, +0.0017] | [-0.0027, +0.0007] |
| pair-uniform, pair | -0.0017 ± 0.0004 / 0.0008 / 0.0007 | [-0.0033, -0.0002] | [-0.0031, -0.0003] |
| pair-uniform, benchmark | -0.0015 ± 0.0003 / 0.0008 / 0.0007 | [-0.0030, +0.0000] | [-0.0029, +0.0000] |

The gain comes from pooling a benchmark's level across the subjects a run holds on it. On multi-subject benchmarks it is worth 0.0015 to 0.0033 to a target that shares its benchmark with one other pair, 0.0019 to 0.0044 with two or more, and nothing measurable to a target alone on its benchmark (at most 0.0016 ± 0.0014). On the single-subject swe_rebench pair hier's fitted defaults are worse, by 0.006 to 0.009 against the legacy Predictor, mostly through the widened level prior. Weighted like the real run (5 targets alone, 4 in pairs), hier's expected gain on public data would be 0.0006 to 0.0024. On that evidence the first verdict was to keep the legacy Predictor (F§ "Verdict: keep the Predictor"). §5.3 reversed it.

### 5.3 Test-like runs: calibrating the level

**What moving the level buys, unconstrained** (selection half, runs 0 to 99; F§ "What moving the level buys, unconstrained"):

| family | best configuration | ALC | minus legacy Predictor | public cost, bench-first / pair-uniform |
|---|---|---|---|---|
| legacy Predictor (first submission) | | 0.2039 | | |
| hier, fitted defaults | | 0.1906 | -0.0134 ± 0.0009 / 0.0017 / 0.0014 | -0.0019 / -0.0017 |
| smoothed Beta(2,2) | | 0.1788 | -0.0252 ± 0.0013 / 0.0021 / 0.0019 | +0.0070 / +0.0075 |
| hier, Gaussian level | mu0 -4, sigma_mu 2.5, attr_scale 0.5 | 0.1572 | -0.0467 ± 0.0025 / 0.0042 / 0.0036 | +0.0036 / +0.0032 |
| hier, Student-t level | mu0 -4, scale 1.8, attr_scale 0.5 | 0.1570 | -0.0469 ± 0.0025 / 0.0042 / 0.0036 | not scored |
| hier, empirical-Bayes level | on the Gaussian above | 0.1567 | -0.0473 ± 0.0025 / 0.0042 / 0.0036 | +0.0034 / +0.0033 |
| legacy Predictor with a level fix | off -2.5, sc 1, va 2 | 0.1562 | -0.0477 ± 0.0027 / 0.0045 / 0.0038 | +0.0329 / +0.0329 |

Four observations follow from this table and the grid behind it.

- **The legacy Predictor loses to answering from Beta(2,2) here.** What it loses is its prior. Every family reaches about 0.156 to 0.157 once its level is moved, and the plateau is flat: 14 of 108 Gaussian configurations lie within 0.001 of the best.
- **The widest level prior always won.** At every shift, sigma_mu 2.5 beat 1.8, 1.3 and 0.9 in every cell of both grids.
- **The public guard decides.** Moving the level costs the legacy Predictor far more on public runs than it costs hier. hier gives back most of its B0 and B1 losses from B7 on. The legacy Predictor's fit (`fitting.fit_ab`) also diverges when the prior sits far from the labels, a library bug reported in F§ "fit_ab diverges when the prior is far from the labels".
- **In this regime, moving the level is worth more than any modelling choice we measured.** Every family gains most of its lead by its level. How much of the shipped model's gain a level-calibrated smoothed mean would recover was not measured (§7).

**The shipped configuration** (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5), paired with the legacy Predictor and the smoothed mean on identical runs (F§ "Shipped configuration, confirmed"; `experiments/ship_confirm.py`, `results/ship_confirm.json`):

| regime, runs | shipped ALC | minus legacy Predictor | parent-level | parents (range) | minus smoothed Beta(2,2) |
|---|---|---|---|---|---|
| test-like, 0-99 (selection half) | 0.1608 | -0.0431 ± 0.0019 / 0.0032 / 0.0028 | -0.038 ± 0.009 | -0.059 to -0.014 | -0.0180 ± 0.0010 / 0.0021 / 0.0020 |
| test-like, 100-199 (confirmation half) | 0.1677 | -0.0396 ± 0.0019 / 0.0037 / 0.0034 | -0.035 ± 0.011 | -0.058 to -0.007 | -0.0159 ± 0.0009 / 0.0021 / 0.0020 |
| test-like, 200-299 (never used to choose the level) | 0.1687 | -0.0419 ± 0.0017 / 0.0037 / 0.0034 | -0.038 ± 0.008 | -0.057 to -0.021 | not stored |
| test-like, 0-299 | 0.1658 | -0.0415 ± 0.0011 / 0.0034 / 0.0031 | -0.037 ± 0.009 | -0.058 to -0.014 | -0.0169 ± 0.0007 / 0.0020 / 0.0019 (0-199) |
| mix/whole, 0-99 | 0.1694 | -0.0436 ± 0.0014 / 0.0039 / 0.0031 | -0.029 ± 0.012 | -0.065 to -0.010 | -0.0176 ± 0.0008 / 0.0022 / 0.0019 |
| no date shift, 0-99 | 0.1585 | -0.0144 ± 0.0012 / 0.0024 / 0.0022 | -0.012 ± 0.005 | -0.024 to -0.000 | -0.0220 ± 0.0014 / 0.0030 / 0.0028 |
| public R1 benchmark-first, 0-149 | 0.2051 | -0.0017 ± 0.0008 / 0.0021 / 0.0018 | -0.003 ± 0.005 | -0.013 to +0.010 | -0.0086 ± 0.0010 / 0.0030 / 0.0021 |
| public R1 pair-uniform, 0-99 | 0.2020 | -0.0018 ± 0.0010 / 0.0017 / 0.0016 | -0.002 ± 0.005 | -0.015 to +0.009 | -0.0093 ± 0.0011 / 0.0021 / 0.0019 |

The rows are hier with the old multiple-choice floor and the solver before the floored-fit fix, the code of run 2's archive (§5.4, Appendix C). The selection-half row reproduces `results/level_calibration.json`'s `summary.selection` exactly.

- **Held out, the gain is 0.040 to 0.042.** Runs 200 to 299 are the cleanest held-out estimate the repository offers. The shipped model beats the legacy Predictor on 296 of 300 test-like runs. Leaving one parent out of all 300 gives -0.047 to -0.032.
- **Between benchmarks it is less certain.** The parent-level SE (0.009) is about 2.7 times the cluster SE. matharena gains least (-0.014 ± 0.009) and multi_swebench most (-0.058 ± 0.004).
- **Most of it is at B0 and B1.** By budget on runs 0 to 299, the shipped model gains 0.135 (cluster SE 0.011) at B0, 0.076 at B1, 0.037 at B3, 0.016 at B7 and 0.008 at B15 and B31: B0 and B1 carry 0.029 of the 0.042.
- **Against the smoothed mean, which has no level to tune, the gain is 0.016 to 0.018 in the tuned regime and 0.009 on public runs.** It is larger without the date shift (0.022) and positive on every test-like parent.
- **On public runs the shipped model is level with the legacy Predictor.** Its point estimates are 0.0015 to 0.0018 better, within about one cluster SE (0.81 cluster SEs benchmark-first, 1.02 pair-uniform). It loses at B0 (+0.0049 ± 0.0065 benchmark-first, +0.0140 ± 0.0072 pair-uniform, cluster SEs) and gains from B7 on (0.004 to 0.008 a budget). It loses on public matharena (+0.010, +0.009) and on the single-subject swe_rebench pair (+0.010, one subject, no SE).

**Against the neighbouring configuration** (mu0 -3.0, attr_scale 0.25; F§ "The public guard decides", F§ "What actually shipped, after the audit"). The neighbouring configuration was confirmed on runs 100 to 199 at -0.0418 ± 0.0024 / 0.0047 / 0.0042 against the legacy Predictor, and cost +0.0008 and +0.0007 on the two public weightings. Against it, the shipped configuration gives up 0.0030 ± 0.0005 / 0.0009 / 0.0007 on the selection half and 0.0022 ± 0.0005 / 0.0011 / 0.0009 on the confirmation half. It gains 0.0023 ± 0.0002 / 0.0005 / 0.0004 (benchmark-first) and 0.0024 ± 0.0002 / 0.0004 / 0.0004 (pair-uniform) on public runs 0 to 99 (`results/level_audit.json`, `mild`).

**Sensitivity to the regime.** Minus the legacy Predictor, ± run / cluster / stratified SE. The level_mean -1.2 and -2.0 regimes were scored for the neighbouring configuration only.

| configuration | level_mean -1.2 (100 runs) | level_mean -2.0 (100) | mix/whole (100) | no date shift (100) | level_mean -0.8 (40; realised -0.77) | no level tilt (40; realised -0.59) |
|---|---|---|---|---|---|---|
| shipped | not scored | not scored | -0.0436 ± 0.0014 / 0.0039 / 0.0031 | -0.0144 ± 0.0012 / 0.0024 / 0.0022 | -0.0286 ± 0.0026 / 0.0039 / 0.0035 | -0.0222 ± 0.0025 / 0.0049 / 0.0043 |
| neighbouring | -0.0388 ± 0.0021 / 0.0044 / 0.0039 | -0.0524 ± 0.0022 / 0.0044 / 0.0039 | -0.0467 ± 0.0017 / 0.0049 / 0.0038 | -0.0139 ± 0.0014 / 0.0028 / 0.0026 | -0.0283 ± 0.0033 / 0.0049 / 0.0044 | -0.0196 ± 0.0033 / 0.0063 / 0.0055 |
| smoothed Beta(2,2) | -0.0228 | -0.0281 | -0.0260 | +0.0077 | -0.0183 | -0.0119 |
| legacy Predictor with a level fix, off -1.5 | -0.0353 | -0.0446 | -0.0398 | -0.0137 | -0.0275 | -0.0232 |

Sources: the shipped row's mix/whole and no-shift cells from `results/ship_confirm.json`; the first four columns of the other rows from F§ "Sensitivity to the regime" (`results/level_calibration.json`); the last two columns from `results/level_audit.json` (`extra`: seed 5, the date shift kept, the library at `4d2cc4f`). The mix/whole and no-shift runs are the same 100 runs for every row.

The gain shrinks as the hidden level rises toward the public centre: 0.042 at the tuned regime's realised -1.29, 0.029 at -0.77 and 0.022 at -0.59. The level_mean -0.8 and untilted regimes are the nearest scored to the pooled feedback reading (§5.1). Without the synthetic date shift, which is what inflates the attribute prior in this regime, the gain against the legacy Predictor shrinks to 0.014; against the smoothed mean it does not shrink.

**The empirical-Bayes level adapts in the right direction.** At budget 1 it moves most of the way from the public centre toward each regime's level, and it orders the regimes by their levels (F§ "The empirical-Bayes level adapts the right way"). It ties the fixed configuration because the level fixed at B0 decides most of the gain. It was not shipped: it is about 70 lines of experiment code outside the library, with a slower worst call.

### 5.4 The shipped model across regimes

| regime | runs | B0 | B1 | B3 | B7 | B15 | B31 | ALC ± run SE | single-run sd of ALC |
|---|---|---|---|---|---|---|---|---|---|
| test-like (primary) | 300 | 0.2165 | 0.1892 | 0.1645 | 0.1525 | 0.1450 | 0.1386 | 0.1658 ± 0.0018 | 0.0318 |
| test-like, mix/whole | 200 | 0.2178 | 0.1931 | 0.1692 | 0.1590 | 0.1524 | 0.1454 | 0.1711 ± 0.0023 | 0.0329 |
| test-like, no date shift | 100 | 0.1948 | 0.1763 | 0.1620 | 0.1479 | 0.1413 | 0.1353 | 0.1585 ± 0.0033 | 0.0335 |
| public R1, benchmark-first | 150 | 0.2350 | 0.2243 | 0.2128 | 0.1973 | 0.1852 | 0.1771 | 0.2051 ± 0.0023 | 0.0285 |
| public R1, pair-uniform | 100 | 0.2442 | 0.2215 | 0.2069 | 0.1929 | 0.1813 | 0.1708 | 0.2020 ± 0.0026 | 0.0261 |
| platform, formative run 2 | 1 | 0.2368 | 0.1947 | 0.1963 | 0.1836 | 0.1785 | 0.1833 | 0.1926 | |

Source: `results/subject_side.json` (`summary.regimes.*.ship`), `experiments/subject_side.py`; single-run sds from `results/ship_confirm.json` (`single_run_sd`); run 2 from `results/formative_feedback.json`. The 'ship' arm reproduces `experiments/itemsig_eval.py`'s base exactly.

**Which code these numbers describe.** The rows were scored at commit `bd0be67` with `paiec/hier.py` 70a3a81a and the old multiple-choice floor: the library of run 2's archive, before the corrected floor (`f7e7d87`) and the floored-fit fix (`4d2cc4f`). (`paiec/prior.py` and `paiec/subjects.py` are at their `f7e7d87` versions, whose new terms are off by default and leave the prior and predictions unchanged; F§ "Subject side at budgets 0 and 1".) The archive now selected, `4a882cc7…`, adds both changes. The corrected floor moves ALC by -0.00026 (test-like), -0.00027 (benchmark-first) and -0.00045 (pair-uniform) (`results/mcq_floor.json`). The floored-fit fix changes nothing where Newton converges; on runs 0 to 199 it moves ALC by 0, -0.0000002 and -0.000008 (`results/hier_floor.json`). Neither is added to the table.

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

**Three options beat the default by more than two cluster SEs:** a narrower level prior, the Student-t level and the Laplace fit. None clears a two-sided Bonferroni bar for sixteen comparisons. All three make the model react less to a run's first labels. The deliberately widened level prior costs about 0.002 on public runs. That is the price of the bet on hidden levels far from the public centre, which the first feedback's lower-root reading supported and the pooled reading of both runs supports less (§5.1).

**The corrected multiple-choice floor.** It now reads "(A, B, C, D, or E)" lists and ignores TikZ point labels. It gains 0.00026 ± 0.00003 / 0.00008 on test-like runs and 0.00027 to 0.00045 on public runs, all on matharena at B0 and B1 (F§ "The multiple-choice floor, corrected", `experiments/mcq_floor.py`; measured on the legacy harness rows, with the solver before the floored-fit fix). It is adopted in the library. The archive was rebuilt with it and the floored-fit fix on 2026-09-27 (sha256 `4a882cc7…`; validator OK, run check bit-identical, `prior.json` byte-identical to the previous build). Formative run 2 used the earlier archive.

### 5.6 Cost

The shipped configuration takes 0.80 to 1.04 ms per evaluation call on the subject-side runs, with a slowest single call of 0.16 to 0.31 s, measured with two processes on a shared machine (F§ "Subject side at budgets 0 and 1", Latency). The neighbouring configuration, measured with six processes, took 0.75 ms on test-like runs and 0.88 ms on public runs, slowest call 0.45 s, and 2.0 to 2.4 ms per call on dense real_webagents and researchcodebench, at most 0.16 s (F§ "Calibrating for the hidden test", Latency). The floored-fit fix changes the time per call by -0.2% to +0.2% on public runs (F§ "Floored fits", Latency). A formative run is about 3,000 evaluation calls, a few seconds against the 8-hour limit. Dense multi_swebench and matharena, hier's worst case, were not measured (§7).

---

## 6 What does not transfer

Each idea below is recorded as negative with its script. The label-conditional layer, the meta-learned heads, the known-sign cues and the subject-side priors were scored against the shipped hier leave-one-parent-out, and so were the language-model probes that produce an item covariate. The text maps, the embeddings and the blind ratings are data-level correlations with Rasch difficulty, as are the probes' own correlation prongs. §6.7's results come from the legacy replica. They share one reason for failing. The hidden test's headroom lies at budgets 0 and 1 and in per-item structure of benchmarks nobody has seen, and none of these signals reaches either.

| idea | best honest estimate | verdict | script, results |
|---|---|---|---|
| item difficulty from TF-IDF text, across benchmarks | LOBO Pearson 0.14, 0.23, 0.09, -0.22, 0.16 | does not transfer | `experiments/transfer.py` (data-level) |
| item difficulty from Qwen3-Embedding-0.6B | LOBO ridge -0.16 / -0.03 / -0.23 / -0.02; within benchmark 0.66 / 0.27 / 0.38 / 0.51 | does not transfer | `experiments/emb_transfer.py`, `results/emb_transfer.json` |
| blind LLM difficulty rating (rater undocumented, §6.2) | 0.482 on matharena; -0.13 to 0.21 elsewhere | mathematics only; inconclusive elsewhere | `experiments/llm_rating/analysis.py` |
| Qwen3-4B zero-shot judge (rating, digit, entropy, nll) | no benchmark-equal test-like difference ≤ -0.001 | closed | `experiments/llm4b_close.py`, `results/llm4b_close.json` |
| a 4B model's own attempts | graded accuracy 2.3% and 7.8%, below the 10% floor | cannot be read | `experiments/attempt_probe.py`, `results/attempt_probe.json` |
| entropy and hidden-state heads | LOBO r -0.05 (entropy) and +0.12 (hidden state) | replicated null | `experiments/hidden_state_probe.py`, `results/hidden_state_probe.json` |
| in-context learning over the pair's labels | r 0.206 ± 0.040, below 0.3 | closed | `experiments/icl_probe.py`, `results/icl_probe.json` |
| pairwise (anchored) comparisons | pooled q 0.540, below 0.60 | dropped | `experiments/pairwise_probe.py`, `results/pairwise_probe.json` |
| text-similarity residual layer (itemsig) | nested -0.00002 ± 0.00002 / 0.00008 / 0.00008 | no gain | `experiments/itemsig_eval.py`, `results/itemsig_eval.json` |
| meta-learned heads on frozen embeddings | nested 0 (every head off in every fold); forced +0.0002 to +0.0006 | no gain | `experiments/heads_eval.py`, `results/heads_eval.json` |
| fine-tuned encoder | 0 of 4 parents at held-out r ≥ 0.3, best +0.05; transferred line +0.00048 | closed | `experiments/finetune_encoder.py`, `results/finetune_encoder.json` |
| Qwen3-14B demand rubric and direct ratings (Kaggle, two T4s) | LOBO head r +0.19, positive on 4 of 4 parents (-0.04 within competition on matharena); best nested line -0.00084 ± 0.00043 | null for ALC | `experiments/strong_llm_eval.py`, `results/strong_llm_eval.json` |
| Qwen3-14B reasoning attempts (matharena only) | within-competition rho 0.372 [0.209, 0.516] on the 147 probe texts, 0.352 on all 270 attempted; forced per-pair line -0.00011 ± 0.00005 | GO for correlation; NULL for ALC (one parent) | same |
| known-sign item cues (ordinal fields, position, format, length, stated size, image refs) | nested test-like 0 to +0.00003 | none passes the sign rule or the gate | `experiments/itemcov_eval.py`, `results/itemcov_eval.json` |
| subject side: harness identity, ordered effort, date forms, Student-t level | best nested -0.00097 ± 0.00033 (ordered effort) | none passes | `experiments/subject_side.py`, `results/subject_side.json` |
| acquisition policies (legacy replica) | coverage-first -0.0000 ± 0.0013 | none beats random | `experiments/acquisition.py` |
| post-hoc temperature and slip (legacy replica) | loses 0.0007 ± 0.0011 | does not transfer | `experiments/acquisition.py` |
| offline bank of per-item results | 1 of 161 inventory benchmarks judged usable (the judgement is unrecorded, §6.7) | closed | `experiments/inventory_scan.py` |

### 6.1 Item difficulty from text: TF-IDF and neural embeddings

Leave-one-benchmark-out, a TF-IDF map from item text to Rasch difficulty correlates at 0.14 on matharena, 0.23 on multi_swebench, 0.09 on real_webagents, -0.22 on researchcodebench and 0.16 on swe_rebench. Within matharena the same model reaches 0.73, but most of that is identifying which competition an item comes from. Competition alone explains 40% of the variance of this difficulty target, and removing the competition mean drops the correlation to 0.49 (F§ "What transfers between benchmarks"); the Rasch variance component in §2.1 puts the competition's share at 0.50.

Qwen3-Embedding-0.6B embeddings do no better across benchmarks (F§ "Neural embeddings do not carry difficulty to an unseen benchmark"):

- LOBO ridge correlations are -0.16, -0.03, -0.23 and -0.02, and kNN -0.08 to +0.16.
- R² as predicted is at most 0 everywhere, and the carried slope has the wrong sign.
- Within a benchmark (5-fold) they reach 0.66, 0.27, 0.38 and 0.51. But the item_features group mean alone reaches 0.57, 0.12, 0.41 and 0.52, and with whole groups held out the embeddings fall to 0.40, 0.16, -0.02 and -0.09. So within a benchmark the embedding mostly identifies the group, which hier already learns from labels.

*Literature.* Amortized calibration predicts item difficulty from embedded question content and reports generalisation across datasets [Truong et al. 2025]. ADeLe reports that black-box predictors built on embeddings or fine-tuning are weaker than rubric-based demand levels, especially out of distribution [Zhou et al. 2025]. Our result is the out-of-distribution end of that picture on agentic and code benchmarks: the direction of difficulty in embedding space is not shared between benchmarks. For bug-fixing tasks, Agent Psychometrics predicts task-level success on unseen benchmarks from issue statements *together with* repository context, solutions and test cases [Ge et al. 2026]. The competition's items carry the issue text and grouping metadata only.

### 6.2 LLM difficulty judgements

180 items across four benchmarks were rated blind to their difficulty on a 0-100 scale and compared with Rasch difficulty (F§ "Language-model difficulty judgement").

| benchmark | n | Pearson | 95% CI |
|---|---|---|---|
| matharena | 45 | 0.482 | [+0.22, +0.68] |
| multi_swebench | 45 | 0.133 | [-0.17, +0.41] |
| real_webagents | 45 | 0.210 | [-0.09, +0.47] |
| swe_rebench | 45 | -0.128 | [-0.41, +0.17] |
| pooled within benchmark | 180 | 0.174 | [+0.03, +0.31] |

A control on 60 fresh multi_swebench items with the full issue text (capped at 4,000 characters) gives Pearson +0.252 (interval touching zero) and Spearman +0.109, so truncation was not the explanation.

**The rater's prompt and protocol are undocumented.** The ratings are hard-coded in `experiments/llm_rating/ratings_main.py` and `ratings_control.py`, which entered the repository in its first commit (2026-09-23), co-authored by Claude Opus 5 in a Claude Code session. The team confirms that the rater was the Claude model of that coding session (the commit's trailer names Claude Opus 5). Its prompt, the date and the sampling settings are recorded nowhere. The rater was blind to difficulty only: item ids carry the benchmark, and the texts name their domain. Whether it saw measurement-db (which that session held) or the benchmarks' public leaderboards cannot be excluded. The correlations re-derive from the stored ratings; the ratings do not (§8.6).

*Literature.* Human-labelled difficulty of math and coding problems is linearly decodable from model activations [Lugoloobi and Russell 2025], and rubric-based LLM annotation of task demands predicts instance-level performance [Zhou et al. 2025]. We see the judge track difficulty where it is visible in the statement (competition mathematics) and not where it lies in the environment (repository size, files to touch, test harness). The gate table (§4.5) needs an honest r of 0.3 to 0.4 across benchmarks. The judge reaches 0.3 against full-sample difficulty on one benchmark of four. With 45 items per benchmark the intervals on multi_swebench and real_webagents include 0.3, so outside mathematics the result is inconclusive rather than negative.

A fully specified local judge, Qwen3-4B-Instruct-2507, read each item once and rated it as a digit (`experiments/llm_features.py`). It was closed (F§ "The 4B judge, closed out"): no benchmark-equal test-like difference reached -0.001 through the harness, and on text-bearing matharena items every feature's partial correlation was below 0.2 with the declared sign. Its extraction covers matharena and 1,941 of 2,078 multi_swebench items only.

### 6.3 A label-conditional text-similarity layer

`paiec/itemsig.py` carries hier's residuals on a pair's labeled items to unlabeled items with similar text. It was run over a 480-configuration grid, with nested leave-one-parent-out selection redone in every bootstrap resample (F§ "Item signal from the pair's own labels"):

| regime (base ALC) | nested, selected within the regime | in-sample best of 480 |
|---|---|---|
| test-like, primary, runs 0-299 (0.1658) | -0.00002 ± 0.00002 / 0.00008 / 0.00008 | -0.00006 |
| test-like, mix/whole, runs 0-199 (0.1711) | -0.00028 ± 0.00005 / 0.00021 / 0.00019 | -0.00037 |
| public R1, benchmark-first, runs 0-199 (0.2057) | -0.00003 ± 0.00012 / 0.00049 / 0.00046 | -0.00071 |
| public R1, pair-uniform, runs 0-199 (0.2020) | -0.00076 ± 0.00018 / 0.00031 / 0.00029 | -0.00112 |

The layer cannot act at B0 or B1 by construction: there are no labels at B0, and at B1 centring within the pair zeroes the single residual. At B31 it takes 0.5% of the gap between the pair-rate oracle and the item oracle. That gap is 0.063 of Brier on test-like runs, and the base already matches the pair-rate oracle there. Selected on the public runs of the other parents, the layer picks aggressive settings that lose +0.0019 ± 0.0007 on held-out researchcodebench. Not shipped.

*Literature.* Generic assessors that transfer instance-level performance from reference instances do well in distribution, but out of distribution "no clear winner emerges and the overall performance is worse" [Pacchiardi et al. 2024].

### 6.4 Meta-learned heads on frozen embeddings

An episode-trained study put linear and low-rank heads on frozen Qwen3-Embedding-0.6B features on top of the shipped hier: a meta-learned difficulty direction, a learned-metric few-shot kernel, and an assessor-style subject × item term (F§ "Meta-learned heads on frozen embeddings"; `experiments/heads_eval.py`, `results/heads_eval.json`). Its strength was chosen by nested leave-one-parent-out, and a head was used on the held-out parent only if its inner mean was below 0.

- **Nested, every head is off in every fold**, on the rows it ran on and on the rows of the current library, so the nested difference is exactly 0. The smallest inner mean over folds and strengths is +0.000007.
- **Forced on**, the combined head costs +0.0002 to +0.0006 test-like.
- **It learns something where the benchmark has been seen**, with subject folds inside the training benchmarks: the best two of five configurations (the difficulty head at λ 0.001, and all heads at PCA 256) gain 0.0005 and 0.0017 test-like and 0.0024 and 0.0046 mix/whole. Across all five the range is +0.0005 to -0.0017 test-like, where both λ 0.01 variants lose to hier, and -0.0008 to -0.0046 mix/whole. None of it reaches a held-out parent.
- **The same head on in-sample Rasch difficulty** gains -0.0437 test-like (cluster SE 0.0037) and -0.0557 mix/whole. That is the acceptance target the harness reproduces (§4.5).

The study ran in session scratch; the script reproduces all 21 of its result files bit for bit.

### 6.5 Item covariates with a known sign

`paiec/itemcov.py` reads benchmark-agnostic cues off the item dict, each with its sign declared in advance: ordinal difficulty fields, problem position, answer format, text length, stated amount of work and image references. The plan allowed a transferred slope only if a cue had the declared sign *and* agreed with the other benchmarks' mean on 4 of 5 units (F§ "Item covariates with a known sign"; computed on the legacy harness rows, before the corrected floor and the floored-fit fix). Three facts decide the result:

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
- **The offline bank.** Of the organisers' 161 inventory benchmarks, 5 publish per-item outputs for at least nine models, and only one (capability) was judged to publish them graded (F§ "The offline bank"). How that judgement was made for the five (phyblock, mmdocrag, atmossci_bench, capability, engdesign) is not recorded: `results/strong_repos.csv` has no producing script, and the step may have inspected their files (§8.6). The rules also restrict competition-specific training to the public pool. What the scan read, and that nothing from it entered any model or selection, is stated in §8.6.

*Literature.* Active and anchor-point selection choose informative items for a population of models [Li et al. 2024; Vivek et al. 2024; Maia Polo et al. 2024]. Here the evaluation items are a fixed half, the first label is worth 0.9 of a budget, and the level of a new benchmark, not the ranking of items, dominates the error.

### 6.8 Studies closed since v0

Six language-model and encoder studies that v0 listed as in progress are closed, each by the rule its plan fixed (the rows of the §6 table; F§ "The 4B judge, closed out", "Attempting instead of judging", "Entropy profiles and hidden-state probes", "Few-shot prompting", "Fine-tuning an encoder"):

- **The 4B zero-shot judge** (rating, digit, entropy, nll): closed; no benchmark-equal test-like difference ≤ -0.001 (§6.2).
- **A 4B model's own attempts**: below the plan's floor. Graded accuracy was 2.3% answering at once and 7.8% with a short chain of thought, so the attempts are mostly forced guesses and the design cannot show whether they carry difficulty. A post-hoc feature met the plan's GO numbers on 32 extreme items and is discounted.
- **Entropy and hidden-state heads**: a replicated null. Leave-one-benchmark-out r is -0.05 for the entropy head (positive on 2 of 4 parents) and +0.12 for the hidden-state head (none at 0.2); the nested harness lines are +0.00098 and +0.00003, and 0 and -0.000001.
- **In-context learning over the pair's labels**: closed. It learns from the labels (+0.158 ± 0.059 in r over zero-shot) but reaches r 0.206 ± 0.040, below 0.3 on every parent, and mostly duplicates what hier learns from the same labels.
- **Pairwise (anchored) comparisons**: dropped at pooled q 0.540, against a bar of 0.60.
- **A fine-tuned encoder**: closed. 0 of 4 parents reach a held-out r of 0.3 (best +0.05), and the transferred harness line is +0.00048.

The plan's escalation after the 4B's floor ran as well: a stronger model on a free GPU (F§ "Strong model on Kaggle: Qwen3-14B rubric and attempts"; `experiments/strong_llm_eval.py`, `results/strong_llm_eval.json`). Qwen3-14B, 4-bit AWQ under vLLM on two Kaggle T4s, ran for one 10.7-hour session. It rated all 4,326 items of the four parents on a ten-scale demand rubric (eight ADeLe-style demands, a judged solve share and an expert's time), and made four reasoning attempts of up to 4,096 tokens on 270 of the 819 checkable matharena texts, the 147 probe texts first.

- **The attempts pass their correlation rule.** The rule and the primary feature were fixed for the probe texts before any output existed. The primary, the mean token entropy of the raw distribution, reaches a within-competition Spearman of 0.372 [0.209, 0.516] against difficulty, with graded accuracy at 20.5%: GO. On the 2026 contests, which post-date the model and so cannot have been memorised, it is 0.585; on the other contests 0.29, so the pooled value leans on the 2026 ones. On all 270 texts it is 0.352 [0.251, 0.448]. It is the first item-side signal to pass its correlation bar. 97% of the attempts were truncated at 4,096 tokens, so it reads how the model starts to reason, not whether it finishes.
- **Neither use passes the gate.** The rubric's primary head reaches a leave-one-parent-out r of 0.19 (0.11 within a pair), and the best nested harness line of any rubric covariate is -0.00084. The attempts cover one parent, so no transferred slope can be fitted leave one parent out; their forced per-pair lines give -0.0001. A covariate with an honest r ≈ 0.3 on all four parents' items would sit just past the gate at this coverage (-0.0025, passing on 6 of 8 noise draws). Whether the entropy reaches that on the three parents that are not mathematics is unmeasured, and token entropy transfers weakly across agentic benchmarks (below).
- **What was left.** The other 549 attempt texts are matharena too, so they cannot change the verdict. An entropy-only job on every benchmark is the variant with upside; it would need the model at predict time for the hidden benchmarks' items, and waits for the organisers to say whether a model may run there.

*Literature.* Token-entropy profiles carry some difficulty signal within benchmarks but little across them. On 17 agentic benchmarks their Spearman correlation falls from 0.19 under K-fold to 0.14 leave-one-benchmark-out [Krsteski and Meyer 2026]. Pre-generation activations predict a model's *own* success better than length or TF-IDF [Lugoloobi et al. 2026; Cencerrado et al. 2025]. Our subjects are other systems. A 4B proxy's signal did not transfer to them; a 14B's attempt entropy tracks their difficulty on mathematics, measured on one benchmark. Rubric-based demand levels predict instance-level performance [Zhou et al. 2025]. Here nine of the ten scales carry the declared sign on at least three of the four benchmarks, but weakly: within a test-like pair they correlate 0.03 to 0.15 with difficulty.

### 6.9 Why these fail, in two numbers

1. **Item-side terms barely reach the budgets that matter most.** B0 and B1 carry 0.3 of ALC's weight. On the second formative run B0 was the largest error (0.237), and B1 and B3 were level (0.195 and 0.196). A centred item offset is zero at B0. There only the level and subject priors act, and the harness's uncentred B0 term, which never passes the gate (-0.0027 even for the honest oracle). At B1 a per-pair slope carries nothing, because centring zeroes the pair's single label. A slope transferred from other benchmarks does act there: the honest oracle's transferred line gains 0.036 of Brier at B1.
2. **Item-level headroom is large but needs a signal no public cue has.** On test-like runs it is 0.063 of Brier (the item oracle gains 46% of the pair-rate oracle's Brier). The gate table puts the bar at an honest cross-benchmark correlation of 0.3 to 0.4: a transferred slope passes on 6 of 8 noise draws at 0.3 and on 7 of 8 at 0.4. Learned text maps reach at most 0.23 leave-one-benchmark-out, a 14B's demand rubric 0.19, and the zero-shot judge 0.48 on mathematics only. A 14B's attempt entropy reaches 0.37 within competition, also on mathematics only. All are correlations with full-sample or fold-averaged Rasch difficulty, more generous scales than the gate's honest one.

---

## 7 Limitations

- **Five public benchmarks.** Four parents carry every test-like number. The parent-level SE of the headline (0.009) is about 2.7 times its cluster SE, and the per-parent gains range from 0.014 (matharena) to 0.058 (multi_swebench).
- **The test-like regime is tuned in sample to one feedback run.** Its level is not identified by that run (level_mean -1.2, -1.6 and -2.0 are 0.63, 0.37 and 0.46 run sds from the feedback at B0 and B1). It realises a pair logit of -1.29, not its target of -1.6. The pooled reading of both formative runs is more central (-0.65 to -0.78, SE 0.5). In the regimes nearest that reading the shipped configuration's gain over the legacy Predictor is 0.022 to 0.029, not 0.042, and a regime with the wider level spread (sd about 1.8 to 2) that the audit's reasoning calls for was not scored. The budget-0 optimism rests on a synthetic date shift that matches the feedback for the legacy prior only; without it the gain is 0.014.
- **Missing baselines.** No level-calibrated smoothed mean, no plain 1PL model with a pooled level and no attributes, and neither of the organisers' reference predictors other than the empirical mean (BLE, or the empirical mean with BLE acquisition) was scored. So the report cannot say how much of the tuned-regime gain needs hier rather than a moved level.
- **Selection optimism.** Selection and confirmation halves redraw from one catalogue of pairs. Several hyperparameters (`WIDEN`, `G_CAP`, slip, guess, `MAX_LEVELS`, the REFERENCE value of sigma_delta) were set with all five public benchmarks in view. sigma_mu 2.5 lies on the edge of the grid that was scored. The shipped configuration was never scored at level_mean -1.2 or -2.0.
- **Single-subject benchmarks.** On the one public single-subject benchmark the shipped configuration loses about 0.010 against the legacy Predictor (+0.0099 over 130 benchmark-first appearances, +0.0114 over 7 pair-uniform ones) and 0.013 to 0.014 against the smoothed mean, from one subject and with no SE (`results/ship_confirm.json`). hier's fitted defaults lost 0.006 to 0.009 there, mostly through the widened level prior. The hidden test may hold such benchmarks.
- **Unmeasured cases.** Dense multi_swebench and matharena were not re-measured for hier, so the case the model was built for, and its worst-case latency, are unmeasured. The later studies ran split scope 'pair' only.
- **Library defects found.**
  - `fitting.fit_ab`, the legacy Predictor's fit, diverges when its prior sits far from the labels: 5 of 1,598 test-like pair appearances. Reported, not fixed.
  - `testlike.item_oracle` oscillates on 35 of 2,425 test-like pair appearances, all of them on difficulty strata (689 appearances). That understated the item oracle on strata (32.7%, not 14.5%). Reported, not fixed.
  - hier's floored Newton fit could stop unconverged: 11 of 3,030 floored fits on 400 public runs, one of them collapsed (pair-uniform run 108). This is fixed in `4d2cc4f` (`Problem._settle`). No fit is left unconverged, and run 108's pair goes from 0.217 to 0.770 at B31, against 0.669 observed (F§ "Floored fits", `results/hier_floor.json`). The archive was rebuilt on 2026-09-27 with the corrected floor and this fix (sha256 `4a882cc7…`). Formative run 2 used the earlier archive, and the shipped configuration's numbers in §5.3 and §5.4 come from its code, except the two regimes re-measured at `4d2cc4f` (§5.4, Appendix C).
- **Provenance gaps.** A few design inputs are recorded only as outputs: `item_signal.py` and `predictor_check.py` were not rerun. Two checks in F§ "Acceptance harness", Caveats, exist only as scratch and are marked indicative there; this report does not quote them. The legacy-replica results in §6.7 were not re-measured under the official protocol. The blind rater is undocumented (§6.2), and the 80 inventory validation labels were assigned by an AI agent (§2.1). How five inventory repositories were judged to publish raw or graded outputs is unrecorded (§8.6). Nothing records which exact bytes were uploaded for run 1.
- **Formative feedback is noisy.** A single run's ALC has sd 0.02 to 0.04, and the two runs are different draws with no subject in common. We read feedback only for global hyperparameters.

---

## 8 Reproducibility

### 8.1 Environment

Development ran on an Apple M1 Pro (8 cores, 16 GB; macOS 26.5.1, Darwin 25.5.0) shared with other jobs, under Anaconda's Python 3.10.8 with numpy 1.26.4 (OpenBLAS 0.3.23.dev), scipy 1.10.1, scikit-learn 1.7.2, pandas 2.2.2, pyarrow 23.0.1, huggingface_hub 0.36.2 and pytest 9.0.3. torch 2.6.0 (CPU), transformers 4.57.6, tokenizers 0.22.2 and safetensors 0.8.0 were used only for the offline language-model features and, torch alone, for the meta-learned heads of §6.4 (`experiments/heads_eval.py`). None of them is in `pyproject.toml`: re-running those studies needs them installed separately. The Kaggle study of §6.8 ran `kaggle/strong_probe/strong_probe.py` on Kaggle's two T4 GPUs, with vLLM 0.9.2, torch 2.7.0 and transformers 4.53.2 installed by the notebook (`results/strong_llm_eval.json`, `run.notebook`); reading its outputs locally needs nothing beyond this environment. Every command means the `python` of that environment (docs/report/provenance_facts.md §1.3).

    pip install -e ".[dev]"

The submission needs numpy alone (`requirements.txt`).

### 8.2 Data and pinned inputs

measurement-db is gated. Accept the terms on the dataset page, then:

    export HF_TOKEN=...          # or huggingface-cli login
    python -m paiec.fetch        # the core tables of six benchmarks into data/, at the pinned revision

Every result in this report used measurement-db at revision `bc8204d811823da849c6686bf124d4ca9f82e4de`, downloaded on 2026-09-24. For all 24 parquet files the local sha256 equals the etag recorded at that revision, so `data/` is byte for byte the dataset there. `paiec/fetch.py` now downloads that revision by default (`REVISION`); `PAIEC_DATA_REVISION` or `--revision` overrides it, and `tests/test_fetch.py` checks the pin without network. Whether the dataset's main branch has moved since cannot be checked offline.

The organisers' baseline repository (validator and streaming client) has no license and is not redistributed. Clone it at the commit the replica mirrors:

    git clone https://github.com/aims-foundations/paiec_baseline third_party/paiec_baseline
    git -C third_party/paiec_baseline checkout 82d330ddcdb16016a3ae9e048db7e588ba2c6a39

At that commit `tools/streaming_ingestion.py` has sha256 `c6f2610f…` and `check_submission_zip.py` `abe3cae0…`.

`pytest` needs neither: it runs on synthetic data (362 test functions over 17 files, 542 collected tests; to be updated at the tagged commit).

### 8.3 Rebuilding the submission

    python tools/build_submission.py        # fit prior.json, bake LEVEL, zip dist/paiec.zip, run every check
    python third_party/paiec_baseline/check_submission_zip.py dist/paiec.zip

`dist/paiec.zip` exists only if every check passed (§3.5). The rollback to the first submission's predictor is `--legacy`. The archive writer is deterministic: rebuilding commit ee5085a from `git archive` with that tree's own build script reproduces run 2's archive byte for byte. The fit of `prior.json` is not bit-stable across BLAS threading: with OpenBLAS, OMP and vecLib held to one thread it differs in its last bits (at most 2.1e-11 relative) and so does the archive hash. A byte-identical rebuild needs the same BLAS and thread count as the build machine (§8.1); it was shown there, with the default thread settings, and not on other hardware (F§ "Formative feedback, runs 1 and 2", Archives).

| archive (sha256) | code | used for |
|---|---|---|
| `8e28d930d45b16aee351bfbbc75531a6d12079667793232d4bd74a17f9d47b9f` | b68492c, legacy Predictor | formative run 1 (the file downloaded back from the platform has these bytes) |
| `2c64eaada491cbf85bd54ae190cdbb28f9d0a13870df0dc77a3a850f90cf661f` | ee5085a, hier, old floor, solver before the floored-fit fix | formative run 2 (the file downloaded back from the platform has these bytes) |
| `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090` | 4d2cc4f, hier, corrected floor and floored-fit fix | the archive now selected; a planned run 3 as a regression and latency check only |

### 8.4 Re-running the experiments

Wall times are as recorded, on the shared machine above. Every results file records its commands and code digests under `passes`, and `--summarise` rebuilds its summary from stored rows.

| result | command | wall time | output |
|---|---|---|---|
| official baselines (§2.3, §5.2) | `python experiments/official_baselines.py --runs 600 --jobs 6` | 12 min, 6 processes | `results/official_baselines.json` |
| hier vs legacy Predictor, ablations (§5.2, §5.5) | `python experiments/hier_eval.py --jobs 8 --skip 'r2\|researchcodebench\|pair\|hier t3 level'`, then `--summarise` | 2 h 16 min, 8 processes, 4 resumed invocations | `results/hier_eval.json` |
| test-like regime check (§4.1) | `python experiments/testlike_check.py --phase tune --jobs 5`, then `--phase check --resume --validate --jobs 5` | about 6 CPU hours | `results/testlike_check.json` |
| level calibration (§5.3) | `python experiments/level_calibration.py --stage S --resume --jobs 6` for S = grid, grid2, r1base, eb, t3, prob, eb, screen, confirm, final; then `--summarise` | 3 h 11 min, 6 processes | `results/level_calibration.json` (19 MB) |
| level audit (§3.4, §5.3) | `python experiments/level_audit.py`, then `--stage extra --jobs 1` | 5 s; 1,414 s, 1 process, 1.9 GB | `results/level_audit.json` |
| script versions behind stored results (§4.4, §5.1, §5.3) | `python experiments/script_revisions.py --stage replay`, then `--stage reread` | instant; 40 s, 0.3 GB | `results/script_revisions.json` |
| shipped configuration, confirmed (§1.4, §5.1, §5.3) | `python experiments/ship_confirm.py` | about 5 s, 0.56 GB; reads `data/subject_side_rows` | `results/ship_confirm.json` |
| formative feedback (§5.1) | `python experiments/formative_feedback.py --stage record`, `--stage prereg`, `--stage read` | seconds, 0.3 GB | `results/formative_feedback.json` |
| itemsig layer (§6.3) | `python experiments/itemsig_eval.py --stage run --jobs 4 --rows DIR`, then `--stage oracle`, `--summarise`, `--stage verify`, `--summarise` | 38 min, 4 processes | `results/itemsig_eval.json` |
| acceptance harness (§4.5) | `python experiments/harness.py --stage collect --jobs 1`, `--stage verify --legacy data/harness_rows_legacy`, `--stage table --resume`, `--stage show` | 39 min collection, 1 min verification, 44 min table | `results/harness_thresholds.json` |
| meta-learned heads (§6.4) | `python experiments/heads_eval.py --rows legacy`, then `--rows current` | 54 and 52 min, 1 process, 1.07 GB | `results/heads_eval.json` |
| known-sign cues (§6.5) | `python experiments/itemcov_eval.py --stage signs`, `--stage harness --rows data/harness_rows_legacy`, `HARNESS_SCALE=pooled ... --stage harness --scale train --rows data/harness_rows_legacy`, `--stage inventory`, `--stage show` | 11 min, 1 process | `results/itemcov_eval.json` |
| subject side (§6.6) | `python experiments/subject_side.py --stage verify`, `--stage diag`, `--stage run --jobs 2` (plus `--configs` passes), `--summarise` | about 4 h 25 min over four invocations, 2 processes | `results/subject_side.json` |
| multiple-choice floor (§5.5) | `python experiments/mcq_floor.py --stage items`, `--stage run --harness-rows data/harness_rows_legacy`, `--stage summary` | about 50 min, 2 processes | `results/mcq_floor.json` |
| floored fits (§3.2, §7) | `python experiments/hier_floor_replay.py --stage compare --both`, `--stage timing`, `--stage synthetic`, `--stage modes`, `--stage hidden`, `--stage summary` | about 1 hour over the stages (34 min for the replay, on two shards) | `results/hier_floor.json` |
| embedding transfer (§6.1) | `python experiments/llm_features.py --download` then `--resume`; `python experiments/emb_transfer.py` | extraction `TODO(record)`; probe 232 s, 480 MB | `results/emb_transfer.json` |
| LLM rating study (§6.2) | `python experiments/llm_rating/analysis.py` | seconds | printed |
| closed language-model and encoder studies (§6.8) | `experiments/llm4b_close.py`, `attempt_probe.py`, `hidden_state_probe.py`, `icl_probe.py`, `pairwise_probe.py`, `finetune_encoder.py`, each with the stages in its F§ section | see F§ | `results/<script>.json` |
| a 14B on Kaggle (§6.8) | on Kaggle, `kaggle/strong_probe/strong_probe.py` (its README); then `python experiments/strong_llm_eval.py --stage S` for S = check-schema, ingest, signs, harness, reference, attempts (by default both readings: every attempted text and the probe texts), verdict, run, show | one 10.7-hour Kaggle session on two T4s; locally about 1.5 hours, 1 process (signs 14 min, harness 68 min, reference 8 min) | `results/strong_llm_eval.json` |
| inventory classes (§2.1) | `python experiments/inventory_classes.py` (`--out`, `--out-csv` to write elsewhere) | seconds, no network | `results/inventory_classes.json`, `.csv` |
| data counts (§2.1) | `python experiments/data_counts.py` | 2 s, 0.2 GB | `results/data_counts.json` |
| legacy-replica results (§6.7) | `python experiments/ceilings.py`, `ladder.py --seeds 4`, `transfer.py`, `order_sensitivity.py`, `pool_robustness.py`, `acquisition.py --seeds 4` | minutes each | printed |
| offline bank (§6.7) | `python experiments/inventory_scan.py` (needs network: clones repositories; §8.6) | not recorded; it ran before 2026-09-23 19:14 UTC | `results/inventory.csv`, `results/clone_scan.csv` |

Large per-row files (`data/harness_rows`, `data/harness_rows_legacy`, `data/itemsig_eval_rows`, `data/subject_side_rows`, `data/hier_floor`) are gitignored and regenerated by the scripts above, with one exception: `data/harness_rows_legacy` comes from `experiments/harness.py --stage collect` run at a checkout of `bd0be67` (library `0c05d35e…`). At the current commit the collection writes `data/harness_rows`, which differs on 157 of 300 test-like runs (F§ "Acceptance harness", Provenance), and the numbers of `heads_eval.py --rows legacy`, `itemcov_eval.py` and `mcq_floor.py` depend on the legacy rows. `experiments/ship_confirm.py`, `formative_feedback.py`, `level_audit.py` and `script_revisions.py --stage reread` read `data/subject_side_rows`, so they need `experiments/subject_side.py`'s rows first.

`formative_feedback.py --stage record` verifies the two formative archives only when given them (`--archive1`, `--rebuild1`, `--archive2`, `--rebuild2`; the paths used are in `passes` and point into session scratch, which is not durable). Without them it keeps the `archives` already stored, so the hashes in `results/formative_feedback.json` are the durable record, and re-verifying needs the archives themselves.

### 8.5 Seeds and determinism

- Public formative-like runs use `official.sample_run` with seed 0. Test-like runs use seed 2 (primary), seed 3 (sensitivities), seed 1 (tuning) and seed 5 (the audit's two extra regimes).
- `official`'s seed 0 is the persistent 50/50 split, and other seeds stand in for other hidden splits.
- Bootstraps use 2,000 resamples.
- The replica's acquisition is the platform's sha256 rule, and every hash is a digest, never Python's salted `hash()`.
- LLM features use no sampling, fixed batches and pinned model revisions.
- Every task in the long experiments is a pure function of its key, which makes interrupted runs resumable bit for bit.

### 8.6 Conduct and disclosure

- **Training data.** measurement-db only, the public training pool, at the revision of §8.2. No external per-item data was used.
- **Formative feedback, every use.** Two formative submissions have been scored (F§ "Formative feedback, runs 1 and 2", Uses; `results/formative_feedback.json`, `submissions`).
  - *Run 1* (legacy Predictor) set the test-like regime's defaults: its lower-root B31 reading gave the level target, the date shift is the grid point whose legacy-Predictor B0 and B1 are closest to it, and its shape motivated two run-shape settings. Its composition (5 pairs alone, 4 sharing a benchmark) weighted the first verdict of §5.2. The level calibration chose three global hyperparameters on runs of that regime, and estimated each candidate's score on run 1 by adding its paired difference to run 1's budgets, as a sanity check, not a selection criterion. The audit then read run 1 per pair and moved the shipped level between two guarded configurations. So run 1 was read per pair twice, not only through aggregate statistics; every use set global quantities only.
  - *Run 2* (shipped model) motivated the subject-side study, which shipped nothing, and the pooled reading of §5.1, which changed nothing. That reading's decision rule was fixed and hash-locked before it was computed, but both runs' tables had been seen, and the reading code was revised twice afterwards in its descriptive outputs; the earlier versions give the same decision (§5.1). No hyperparameter has changed since run 2: `LEVEL` and `submission/prior.json` are byte-identical to run 2's archive. The two library changes after it (the multiple-choice floor and the floored-fit fix) came from public-data findings.
  - *A third submission* of the archive now selected is planned as a regression and latency check only; its score is not to select or tune anything.
  - The live leaderboard (read once, 2026-09-24) is used for placement only.
  - The organisers' tables are stored verbatim as a record (`results/formative/`), with their anonymous subject and benchmark ids. The docs, the figures and the research code (`paiec/testlike.py`'s `FEEDBACK`) use relabelled letters, and the ids are used only to check the two runs' overlap, to give the same letter to the same benchmark, and to group pairs by benchmark in the reading of §5.1 (its bootstrap resamples the 7 benchmark ids). No model input, prior, hyperparameter or selection is keyed on an id, nothing in `paiec/`, `submission/` or `tools/` reads `results/formative/`, and `prior.json` holds no item- or benchmark-keyed table. No prediction was shaped to probe hidden labels.
- **The inventory scan.** `experiments/inventory_scan.py` (first commit, 2026-09-23) cloned each GitHub repository of the organisers' 161-benchmark inventory with `--filter=blob:none --depth 1`, listed its tracked files, matched regular expressions against the file *paths* (results-like directories, data extensions, model names), stored counts and up to three example paths, and deleted the clone. The script opened no file's contents, but a blobless clone checks out the default branch, so each repository's files were downloaded to a temporary directory and deleted with the clone. The inventory holds candidates for both pools, so these may include hidden-test benchmarks. One step is not recorded: the judgement that of the five repositories with outputs for nine or more models (phyblock, mmdocrag, atmossci_bench, capability, engdesign) only capability publishes graded outcomes. `results/strong_repos.csv`, which lists them, has no producing script, and the judgement needs file names or contents beyond the scan's counts, so an inspection of those repositories' files cannot be excluded. mmdocrag is a public benchmark; the other four may be hidden-test benchmarks. Nothing from the scan, its outputs, that step or the inventory enters `paiec/`, `submission/` or `tools/`, and no per-item data from any inventory repository was kept or used. `TODO(team)`: confirm how the raw-versus-graded judgement was made. The offline bank it assessed was closed on feasibility and because the rules restrict competition-specific training and curation to the public pool (§6.7). The inventory's titles, not the scan, feed two descriptive analyses: the task-type shares of §2.1 and a keyword scan for difficulty cues (§6.5). Neither sets a model, prior or hyperparameter.
- **Blind LLM difficulty ratings (§6.2).** The rater was the Claude model of the Claude Code session that made the repository's first commit (the commit's trailer names Claude Opus 5; confirmed by the team). Its prompt, the date and the protocol are undocumented. The rater was blind to difficulty, not to benchmark identity, and its exposure to measurement-db or public leaderboards cannot be excluded. The fully specified replacement is the local Qwen3-4B judge (§6.2).
- **Inventory validation labels (§2.1).** The 80 labels in `results/inventory_hand_labels.csv` were assigned by an AI agent reading each title, not by a person; we report them as such (confirmed by the team, 2026-09-28), and the agreement figures of §2.1 are agreement with that agent.
- **LLM coding assistant.** The code, the experiments, `docs/` and this draft were developed with Claude Code (Anthropic). Commit trailers record Claude Opus 5 and Claude Opus 5.5 as co-authors.
- **Pretrained models.** Qwen3-Embedding-0.6B (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`) and Qwen3-4B-Instruct-2507 (revision `cdbee75f17c01a7cc42f958dc650907174af0554`), for offline research features only, and Qwen/Qwen3-14B-AWQ (revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`), run on Kaggle for the research features of §6.8. None is part of the submission, which uses no pretrained model. The Kaggle notebook downloaded measurement-db itself with the team's Hugging Face token, into a directory outside its Output. The kit asks for the notebook and its versions to stay private, and its Output holds no reference answer and no correctness flag, which the kit's tests check (`kaggle/strong_probe/README.md`).
- **Community code.** The organisers' baseline repository at commit `82d330dd` (validator, streaming client and reference predictors), read and run locally, not redistributed.

---

## References

- Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *JASA* 88(421).
- Cencerrado, I. V. M., et al. (2025). No answer needed: predicting LLM answer accuracy from question-only linear probes. arXiv:2509.10625. (`TODO(verify)`: the local v3 lists the first author as Iván Vicente Moreno Cencerrado and the venue as an ICLR 2026 workshop.)
- Ge, C., Kryvosheieva, D., Fried, D., et al. (2026). Agent psychometrics: task-level performance prediction in agentic coding benchmarks. COLM 2026, arXiv:2604.00594.
- Krsteski, S. and Meyer, C. (2026). Predicting task difficulty without rollouts. arXiv:2608.05797.
- Kwa, T., West, B., Becker, J., et al. (2025). Measuring AI ability to complete long software tasks. arXiv:2503.14499 (v4, 2026; the March 2025 v1 is titled "Measuring AI ability to complete long tasks", `TODO(verify)`).
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

Titles were checked against local copies of 12 of the 13 arXiv papers (none for Truong et al. 2025). Author lists, venues and page ranges are `TODO(verify)`: Truong et al. (title, authors, venue), Cencerrado et al. (name form, venue), the venues of Maia Polo, Ruan, Vivek, Pacchiardi, Li and Zhou, and the page ranges of Breslow and Clayton and of Rue et al. Rasch (1960) is not yet cited in the text, and its publisher is to be added.

---

## Appendix A: supplementary tables

### A.1 Formative runs, per pair

Letters relabel the anonymous benchmarks; the same letter is the same benchmark in both runs. They are for recording only, and no model input is keyed on them (F§ "Formative feedback, runs 1 and 2"; the organisers' tables are `results/formative/run1.txt` and `run2.txt`).

Run 1, the legacy Predictor:

| pair | benchmark | n | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| p1 | A | 76 | 0.4224 | 0.2851 | 0.2698 | 0.2067 | 0.2085 | 0.2057 | 0.2568 | 0.171 |
| p2 | B | 50 | 0.4970 | 0.2861 | 0.1394 | 0.0764 | 0.0650 | 0.0513 | 0.1682 | 0.288 |
| p3 | C | 56 | 0.3583 | 0.2419 | 0.2429 | 0.2437 | 0.2751 | 0.1923 | 0.2558 | 0.264 |
| p4 | D | 58 | 0.5663 | 0.3234 | 0.1379 | 0.0518 | 0.0184 | 0.0063 | 0.1636 | 0.344 |
| p5 | E | 60 | 0.3032 | 0.2477 | 0.2479 | 0.2431 | 0.2437 | 0.2472 | 0.2515 | 0.073 |
| p6 | F | 44 | 0.1728 | 0.1670 | 0.1675 | 0.1665 | 0.1719 | 0.1678 | 0.1686 | 0.038 |
| p7 | A | 53 | 0.4771 | 0.2939 | 0.1738 | 0.1335 | 0.1285 | 0.1286 | 0.2065 | 0.204 |
| p8 | F | 44 | 0.1806 | 0.1747 | 0.1761 | 0.1827 | 0.1745 | 0.1749 | 0.1771 | 0.042 |
| p9 | G | 44 | 0.2519 | 0.2653 | 0.2834 | 0.2360 | 0.2386 | 0.2366 | 0.2535 | 0.103 |

Run 2, the shipped model (archive of ee5085a):

| pair | benchmark | n | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| q1 | D | 58 | 0.2555 | 0.1939 | 0.1840 | 0.1931 | 0.1868 | 0.1857 | 0.1957 | 0.091 |
| q2 | G | 44 | 0.2706 | 0.1635 | 0.1790 | 0.1500 | 0.1458 | 0.1490 | 0.1696 | 0.136 |
| q3 | C | 113 | 0.2593 | 0.2469 | 0.2408 | 0.2437 | 0.2418 | 0.2524 | 0.2458 | 0.045 |
| q4 | G | 44 | 0.2576 | 0.2406 | 0.2559 | 0.2311 | 0.2315 | 0.2161 | 0.2392 | 0.093 |
| q5 | F | 44 | 0.2256 | 0.2300 | 0.2553 | 0.2329 | 0.2228 | 0.2392 | 0.2347 | 0.104 |
| q6 | E | 60 | 0.2366 | 0.1973 | 0.1989 | 0.1961 | 0.1950 | 0.1983 | 0.2009 | 0.068 |
| q7 | A | 75 | 0.2333 | 0.2126 | 0.2282 | 0.2119 | 0.2008 | 0.2240 | 0.2164 | 0.104 |
| q8 | B | 54 | 0.1562 | 0.0729 | 0.0283 | 0.0098 | 0.0034 | 0.0014 | 0.0386 | 0.160 |

Mean ECE by budget, B0..B31: run 1 0.381, 0.253, 0.195, 0.097, 0.090, 0.047; run 2 0.210, 0.098, 0.118, 0.084, 0.057, 0.076.

### A.2 hier gain by how many pairs of the target's benchmark a run holds

hier minus legacy Predictor, multi-subject benchmarks, pair-cluster SEs (F§ "Hierarchical model"):

| weighting, split scope | alone | 2 pairs | 3 or more |
|---|---|---|---|
| benchmark-first, pair | -0.0016 ± 0.0014 | -0.0033 ± 0.0011 | -0.0044 ± 0.0010 |
| benchmark-first, benchmark | -0.0009 ± 0.0014 | -0.0020 ± 0.0011 | -0.0029 ± 0.0011 |
| pair-uniform, pair | +0.0003 ± 0.0014 | -0.0017 ± 0.0011 | -0.0024 ± 0.0008 |
| pair-uniform, benchmark | -0.0001 ± 0.0012 | -0.0015 ± 0.0011 | -0.0019 ± 0.0008 |

### A.3 Guarded configurations, both test-like halves and both public weightings

| configuration | selection half | confirmation half | minus legacy Predictor, confirmation | public bench-first (runs 0-99) | public pair-uniform (runs 0-99) |
|---|---|---|---|---|---|
| **hier G mu0 -2.5, sigma_mu 2.5, attr_scale 0.5 (shipped)** | 0.1608 | 0.1677 ± 0.0028 | -0.0396 ± 0.0019 / 0.0037 / 0.0034 | -0.0015 | -0.0018 |
| hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 (neighbouring; recommended, not shipped) | 0.1578 | 0.1655 ± 0.0031 | -0.0418 ± 0.0024 / 0.0047 / 0.0042 | +0.0008 | +0.0007 |
| hier G mu0 -3.5, sigma_mu 2.5, attr_scale 0.25 (rule's argmax) | 0.1574 | 0.1655 ± 0.0033 | -0.0418 ± 0.0026 / 0.0051 / 0.0045 | +0.0025 | +0.0023 |
| hier EB-cs, tau 2, on G mu0 -3.0, attr_scale 0.5 | 0.1580 | 0.1658 ± 0.0031 | -0.0416 ± 0.0022 / 0.0043 / 0.0038 | +0.0005 | +0.0006 |
| legacy Predictor fix off -1.5, sc 1, va 3 | 0.1630 | 0.1696 ± 0.0027 | -0.0377 ± 0.0016 / 0.0032 / 0.0029 | +0.0026 | +0.0028 |
| hier, fitted defaults | 0.1906 | 0.1949 ± 0.0024 | -0.0124 ± 0.0009 / 0.0019 / 0.0016 | -0.0019 | -0.0017 |
| legacy Predictor | 0.2039 | 0.2073 ± 0.0022 | | 0.2068 | 0.2038 |

Sources: F§ "The public guard decides", `experiments/level_calibration.py`; the shipped row from `results/level_audit.json` (`mild`) and `results/ship_confirm.json`. The public columns are differences against the legacy Predictor except in its own row, which gives its ALC. The shipped configuration was chosen after the rest of this table, by the audit (§3.4).

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

With everything fitted without every benchmark of the run, over runs 0 to 149: legacy Predictor 0.2111 ± 0.0019, hier 0.2089 ± 0.0020. hier minus legacy Predictor is -0.0022 ± 0.0004 / 0.0010 / 0.0006 (95% [-0.0039, -0.0001], stratified [-0.0033, -0.0010]). Both models lose about 0.004 against their target-LOBO lines (F§ "Strict run-LOBO").

---

## Appendix B: open items before submission

- `TODO(team)` Confirm how the five inventory repositories were judged to publish raw or graded outputs, and whether any of their files was opened (§6.7, §8.6).
- `TODO(team)` Tag the commit the final archive is built from, and update the test counts of §2.5 and §8.2 there. Every script, results file and test this draft cites is committed (the P0 revision in `00bdf04`, the Kaggle record in the commit after it).
- `TODO(record)` Wall times for `experiments/llm_features.py` and `tools/build_submission.py` (the rebuild of ee5085a took 41 s, `results/formative_feedback.json`, `archives.run2.how`). `experiments/inventory_scan.py`'s is not recorded and cannot be recovered.
- `TODO(verify)` The references listed after the bibliography.
- §5.4 describes the run-2 archive's library, not `4a882cc7…`'s; the gap is measured (§5.4, Appendix C) but the subject-side 'ship' arm was not re-scored at `4d2cc4f`.
- P1 experiments not run: the regime at the audit's reading (level_mean about -0.7 to -1.1 with level_sd about 1.8 to 2, and a two-component mixture) for the shipped, neighbouring, EB, calibrated-smoothed-mean and legacy predictors; a level-calibrated smoothed mean and a plain 1PL baseline; BLE; the shipped configuration at level_mean -1.2 and -2.0; sigma_mu 3.5 and 5; dense multi_swebench and matharena for hier.
- Release the row files (`data/subject_side_rows`, `data/harness_rows`, `data/harness_rows_legacy`) with the code, so that §5's tables re-derive in minutes.
- `experiments/harness.py --stage verify --scratch` still expects the heads study's scratch row format (F§ "Meta-learned heads on frozen embeddings", Caveat).
- `TODO(team)` Authors, affiliations and the public code link. Include measurement-db terms in the release notes.
- `TODO(figures)` See `docs/report/figures.md`.

---

## Appendix C: which code produced each number of §5

"Old floor" is the multiple-choice floor before `f7e7d87`; "pre-fix solver" is `paiec/hier.py` before the floored-fit fix of `4d2cc4f`. Where Newton converges the fix changes nothing, bit for bit; its measured effect on ALC is 0 (test-like), -0.0000002 (public benchmark-first) and -0.000008 (public pair-uniform) on runs 0 to 199 (`results/hier_floor.json`). The corrected floor moves ALC by -0.00026, -0.00027 and -0.00045 (`results/mcq_floor.json`).

| number (section) | script | results file | code it ran | relation to the archives |
|---|---|---|---|---|
| formative run 1, all budgets and pairs (§5.1, A.1) | platform; recorded by `experiments/formative_feedback.py --stage record` | `results/formative/run1.txt`, `results/formative_feedback.json` (`record.run1`) | b68492c, legacy Predictor | archive `8e28d930…`, verified byte for byte against the file downloaded from the platform and against b68492c |
| formative run 2, all budgets and pairs (§5.1, §5.4, A.1) | platform; same | `results/formative/run2.txt`, `record.run2` | ee5085a: hier 70a3a81a, old floor, pre-fix solver | archive `2c64eaad…`, verified byte for byte against ee5085a |
| run 2's placement, single-run sds (§5.1, §5.4) | `experiments/ship_confirm.py` | `results/ship_confirm.json` (`run2_placement`, `single_run_sd`) | rows of `experiments/subject_side.py` at bd0be67: hier 70a3a81a, old floor, pre-fix solver | the run-2 archive's library; not re-run for `4a882cc7…` |
| matched estimates on run 1; the pooled reading (§5.1) | `experiments/formative_feedback.py --stage read` (script `93cb6a84…`; earlier versions give the same decision, `experiments/script_revisions.py`); `experiments/level_audit.py` | `results/formative_feedback.json` (`reading`); `results/level_audit.json` (`feedback`, `mild`); `results/script_revisions.json` | rows of `level_calibration.py` (head bba726c, hier 70a3a81a) and `subject_side.py` (same hier), old floor, pre-fix solver | the run-2 archive's library |
| baselines (§5.2) | `experiments/official_baselines.py` | `results/official_baselines.json` | b68492c, inferred: the results file records no commit, and it entered the repository in b68492c unchanged since; no hier | the legacy Predictor is the run-1 archive's predictor |
| hier vs legacy Predictor, hier's defaults; ablations (§5.2, §5.5, A.2, A.5) | `experiments/hier_eval.py` | `results/hier_eval.json` | bba726c library: hier f3f4e7da, old floor | neither archive's hier or LEVEL |
| unconstrained families, guarded configurations, neighbouring configuration's sensitivities (§5.3, A.3) | `experiments/level_calibration.py` | `results/level_calibration.json` | head bba726c with hier 70a3a81a, old floor, pre-fix solver | the run-2 archive's hier.py; other LEVELs |
| the shipped configuration's own numbers (§1.4, §5.3, A.3) | `experiments/ship_confirm.py` | `results/ship_confirm.json` | rows of `subject_side.py` at bd0be67 (hier 70a3a81a, old floor, pre-fix solver; prior.py and subjects.py at f7e7d87) | the run-2 archive's library; `4a882cc7…` differs by the two changes above, not added |
| shipped minus neighbouring (§3.4, §5.3) | `experiments/level_audit.py --stage mild` | `results/level_audit.json` (`mild`) | the same rows as above, and `level_calibration.py`'s | the run-2 archive's library |
| level_mean -0.8 and untilted regimes (§5.3) | `experiments/level_audit.py --stage extra` (script `40ed9e5b…`; the file since differs in the `analyse` stage and the docstring only, `results/script_revisions.json`) | `results/level_audit.json` (`extra`) | the library at 4d2cc4f: corrected floor, floored-fit fix | the current archive's library |
| the shipped model across regimes (§5.4) | `experiments/subject_side.py` | `results/subject_side.json` | bd0be67 working tree: hier 70a3a81a, old floor, pre-fix solver | the run-2 archive's library |
| the corrected floor (§5.5) | `experiments/mcq_floor.py` | `results/mcq_floor.json` | bd0be67 (library 2a58be15), legacy harness rows, pre-fix solver | the first change between the two archives |
| the floored-fit fix (§3.2, §7) | `experiments/hier_floor_replay.py` | `results/hier_floor.json` | hier at f7e7d87 against 4d2cc4f, corrected floor | the second change between the two archives |
| latencies (§5.6) | `experiments/subject_side.py`; `experiments/level_calibration.py`; `experiments/hier_floor_replay.py --stage timing` | the three results files | as above | shipped config on the run-2 library; neighbouring config; the fix |
