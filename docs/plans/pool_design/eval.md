# Pool evaluation: does a pool-fitted hier replace archive-3? (design for amendment 4)

Written 2026-10-07 at repository HEAD a725df9 (`a725df933d57de614a60c687ee2ba00fd94b80f1`),
read only. Pool: measurement-db revision `54074453da7b9911be0f48e4c560be547bc84d21` under
`data/pool/` (`data/pool/manifest.json`). This note designs the experiment and its decision
rule. It reuses `data.md` (the data facts) and `code.md` (the code path and the candidates
C1 to C3) and does not redo them.

**What this note is and is not.** It runs no predictor and computes no ALC or Brier on any
data. It reads no formative feedback file. The only feedback-derived numbers in it are
quoted from committed files that already hold them: the RS study's level targets
(`experiments/regime_sensitivity.py:250-263`), the run-shape defaults of
`paiec/testlike.py:507-537`, and, in section 12 only, the run-level V figures of
`results/hidden_runs.json`. It reads no item text. Three descriptive scripts were written
for it, in `eval_design/`, run with `PYTHONDONTWRITEBYTECODE=1`. They draw run compositions,
compute attribute scores and do level arithmetic. None of them scores anything:

| script | output | what |
|---|---|---|
| `eval_design/feasibility.py` | `eval_design/feasibility.json` | the benchmark split of section 3 and the realisability of every regime of section 4, on all-item pair logits from `pairs.csv` |
| `eval_design/anchor.py` | `eval_design/anchor.json` | mean attribute score of the evaluation set's subjects under archive-3's ridge and under the prototype pool ridge |
| `eval_design/level_held.py` | `eval_design/level_held.json` | LEVEL's accuracy-scale quantiles under both bundles |

**Owner of the numbers.** Every number the study reports will come from one script,
`experiments/pool_eval.py`, writing `results/pool_eval.json`. The figures below marked
*planning* come from the scratch scripts above and from code.md's prototype. They are not
results, and the script's pre-scoring stages re-measure each one.

## 0 In brief

1. **Question.** Does hier with its priors refitted on the expanded public pool (code.md
   §8: C1, C2 or C3) predict better than archive-3 on the hidden test? The answer decides
   whether archive-4 replaces archive-3 by 2026-10-16. archive-3 is the fallback at every
   step.
2. **The difficulty.** Held-out public benchmarks favour a level learned from the pool, by
   construction. The formative runs read low. So the level is evaluated apart from "the
   rest" (standings, ridge, identity table, item variance, group share). Gains must show
   in regimes whose realised level matches the hidden readings, P-READING and P-AUDIT. No
   material loss is allowed in the untilted pool regime P-FLAT or the others.
3. **Runs.** Every run is drawn with the official protocol from held-out pool benchmarks.
   Every prior is fitted without all of the run's benchmarks (strict run-level exclusion,
   one cached build per run). Formative-size runs have 5 to 12 pairs, mostly one per
   benchmark, at most 1,000 subject-item pairs. Dense runs have 7 benchmarks with up to
   8 subjects each.
4. **Comparator.** archive-3 as shipped (`submission/prior.json`, LEVEL), on identical
   runs and checkpoints. On the old RS regimes, the RS study's stored `ship` rows.
5. **No winner's curse.** The 132 evaluation benchmarks are split once, 50 for selection
   (S) and 82 for confirmation (C). S picks one candidate, which is recorded before any C
   task is scored. C tests that candidate alone, against archive-3.
6. **Metric.** D is the paired difference in run ALC (candidate minus archive-3). Its SE
   comes from a cluster bootstrap over benchmarks, widened to the two-way (benchmark,
   subject) SE when that is larger.
7. **Gate on C.** In P-READING and in P-AUDIT: D ≤ -0.002 with U95 < 0, the
   benchmark-equal mean ≤ 0, and no single benchmark carrying the gain. In P-TUNED,
   P-MIXTURE and P-FLAT: D ≤ +0.002. In the dense regimes: D ≤ +0.001. On the old RS
   regimes: a mean loss of at most +0.0005 and RS's own per-regime bounds. Latency and
   every check must also pass. The rule is quoted in full in section 9.
8. **Cost.** About 20 process-hours for the core. Before selection that is about 6 hours
   wall on 3 shard processes, beside the machine's other jobs; after it, about 3.5 hours.
9. **Feedback.** It enters only as the RS study's global level targets and run shape. The
   hidden runs' V figures are read once, after the gate, for description only.

## 1 What is decided, and why it is hard

**The decision.** Archive-4 differs from archive-3 in `prior.json` alone (C1, C3), or in
`prior.json` and the `LEVEL` literal of `submission/model.py` (C2). This holds because no
shipped module changes (code.md §0.2, §7, §9). Archive-4 replaces archive-3 only if three
things hold:

* this evaluation's gate passes;
* the build (`tools/build_submission.py`) and S3's stress gate (P2 plan §3, S3 row) pass
  on archive-4;
* the one regression submission that P2 plan §4 allows is scored with finite outputs by
  2026-10-16.

**Why held-out public benchmarks are not enough.** The competition page says benchmarks
were assigned to pools at random before curation. So the pool's benchmark levels describe
the population the hidden ones come from. Their mean corrected logit is +0.06, sd 1.54
(data.md §5). A level fitted on the pool is calibrated to that population. On unselected
held-out pool benchmarks it therefore wins by construction.

The formative runs read lower:

* the pooled continuity-corrected reading of runs 1 and 2 is a mean of -0.715 with sd
  1.772 (the RS study's READING target);
* the audit's reading of run 1 is -1.10 with sd 2.00 (AUDIT);
* the lower-root reading of run 1, now known to be biased, realised -1.29 with sd 1.70
  in test-like runs (TUNED).

Those readings rest on 17 pairs from 7 benchmark ids, a level-mean SE of about 0.5 (P1a
plan §11). The summative runs likely draw on the same 7 benchmarks. An evaluation that
showed only unselected held-out benchmarks would therefore reward the bet that the
hidden 7 are typical. That is the bet the formative readings speak against.

**What the design does about it.**

1. **It separates the level from the rest by construction of the arms (section 2).** C1
   keeps archive-3's level and changes the rest. C2 is C1 with the pool's level. C3 is C1
   with the identity guards lifted. So C1 minus archive-3 is the rest at archive-3's
   level, C2 minus C1 is the level on the new rest, and C3 minus C1 is the identity term.
2. **It scores every arm in regimes tilted to the readings (P-READING, P-AUDIT, P-TUNED,
   P-MIXTURE) as well as untilted (P-FLAT).** These are the RS study's five regimes,
   rebuilt from the pool's own benchmarks (section 4).
3. **It asks for gains where the hidden readings sit.** The gate requires them in
   P-READING and P-AUDIT. Everywhere else it only bounds losses. A pool level that wins
   P-FLAT and loses P-AUDIT does not pass.

**Where archive-3's level sits on the pool's subjects** (*planning*,
`eval_design/level_held.json`). At the benchmark-weighted mean attribute score of the
evaluation set's pairs, archive-3's level prior has accuracy-scale quantiles of -2.20,
-1.15 and -0.19 at z = -1, 0 and +1. Here the accuracy scale is the logit of
E[sigmoid(mu + attr_scale m + sqrt(V) e)], with V = sigma_theta² + sigma_delta² + T. So
archive-3's median for the average pair (-1.15) sits between the TUNED (-1.29) and AUDIT
(-1.10) targets, 0.43 below READING's -0.715, and 1.2 logits below the pool's centre
(+0.06, data.md §5). C1 (LEVEL literal over the prototype refit) gives -2.33, -1.10 and -0.00: the same
centre, about 16% wider. code.md §4 finds the same: the B0 prior centre moves by at most
0.024. C2's pool level sits near the pool's own centre, at accuracy scale about +0.1. So
the regimes bracket the arms: archive-3, C1 and C3 bet near AUDIT and TUNED, below
READING; C2 bets near FLAT.

## 2 Arms

### 2.1 The comparator A3 (archive-3)

* **On pool runs:** `HierPredictor(*paiec.prior.load("submission/prior.json"))`. The file's
  sha256 is `c7ce3b845d584f95…`. It is the bundle archive-3 carries: zipping HEAD's sources
  with the build's member list reproduces archive-3's sha256 `4a882cc7…` (code.md §7;
  `results/stress_archive.json: archive` lists its prior.json at the same 18,951 bytes). It carries LEVEL (mu0 -2.5, sigma_mu 2.5,
  attr_scale 0.5), fitted on the five legacy benchmarks.
  * No legacy benchmark enters a pool run (section 3), so A3 needs no refit and no
    exclusion. It is exactly the shipped configuration.
  * The in-repo runtime equals archive-3's: `paiec/hier.py` is `d9a9561202b63b9d`, the
    digest P1a pinned; code.md §9 changes no shipped module; and `tools/build_submission.py`
    `check_run` holds the archive bit-identical to the research package on the same
    prior.json.
* **On the old RS regimes (section 4.4):** the RS study's stored `ship` rows,
  `data/regime_sensitivity_rows/<R>/<i>.json`, key `res.ship.b`, full-precision floats.
  These are hier over `level_calibration.bundle(parent, 0.0)` (the five less the target's
  parent) with LEVEL, as P1a scored it (`experiments/regime_sensitivity.py:668-677`).

### 2.2 The candidates (code.md §8, unchanged)

All three use code.md's pool build: the `paiec/pool.py` loader and caps, `benchmark_fit`
per benchmark, `from_standings` on the standings rows, and fit_hyper's arithmetic on the
fit dicts (code.md §9). Each run's bundle is built without all of the run's benchmarks
(section 3.3).

| arm | subject prior and widths | level (mu0, sigma_mu) | attr_scale | identity (id_cap; sigma_theta) | role |
|---|---|---|---|---|---|
| **C1** pool refit, LEVEL kept | pool | -2.5, 2.5 (LEVEL written over) | 0.5 | 0.1; clip [0.01, 0.2] | the rest, at archive-3's level centre |
| **C2** pool EB level (code.md's C2a) | pool | the pool's empirical-Bayes level at the run-time T (fixed-T pass) | 0.5 | as C1 | the bet that hidden levels are typical |
| **C3** identity re-estimated | pool; sigma_delta recomputed | -2.5, 2.5 | 0.5 | none (1.0); sqrt(tau2_res) unclipped | the identity term, level held |

**C2's three choices** (code.md §8) are fixed by amendment 4. If amendment 4 is silent,
the script locks these defaults:

* **Level set.** Every benchmark of code.md's level part, flagged directions included.
  They belong to the population (data.md §7).
* **Widening.** None. The sd is the EB sd net of mean sampling variance, times
  sqrt(1 + 1/n). WIDEN's 1.44 corrected an ML sd fitted on five levels (code.md §3.2).
  At n ≈ 125 it would only add width the data do not ask for.
* **attr_scale.** 0.5, so that only the level moves.

*Planning:* about N(+0.16, 2.31²) on the prototype (code.md §4).

**C2's level is computed per run.** It is the EB summary of the fixed-T levels of the
run's training benchmarks. The fixed-T pass runs once, at the full pool's T, which is
code.md §6's stated cheap choice: T is a median over about 120 item variances, one of
which is the held-out benchmark's. The script records, for 50 runs of each half, T
recomputed without the run's benchmarks, and C2's mu0 and sigma_mu at that T. Both are
descriptive and change nothing.

**A descriptive arm, C23** (code.md's C2a+C3): both changes together, scored on S only. It
is not a candidate. It gives the level-by-identity interaction.

**What is fixed before the lock.** Every build choice of code.md §10, which this
evaluation never tunes:

* the benchmark rules of the subject part and the level part;
* K, M_p and the pair order;
* the group share without items;
* the rule for degenerate and non-converged fits;
* the level mode and C2's three choices;
* the identity mode;
* the provider threshold;
* the revision.

They go into the `CANDIDATES` block of the script, which the lock hashes. They are chosen
from descriptive facts (data.md, code.md) before any ALC exists. They were made looking
at the whole pool, confirmation benchmarks included, but never at an outcome.

**Order of preference** (for ties at selection, section 8): C1, C3, C2. That is the order
of increasing departure from archive-3's behaviour. C1 changes the rest; C3 also lifts
the identity guards, which code.md §5 marks as the riskier change for named subjects; C2
also moves the level, against the readings.

### 2.3 The decomposition (descriptive, every regime)

| contrast | reads as |
|---|---|
| C1 - A3 | the pool rest at archive-3's level centre (including the about 16% wider accuracy-scale level, section 1) |
| C2 - C1 | the pool's level on the pool rest |
| C3 - C1 | the identity guards lifted |
| C23 - C2 - C3 + C1 (S only) | their interaction |

Each contrast has the paired D and its SEs of section 7. None of these enters the rule
except through C1, C2 and C3 against A3.

## 3 Data, the evaluation set and the split

### 3.1 The evaluation set E

E holds the benchmarks a pool run may draw from: **132 benchmarks, 12,668 eligible pairs**
(`eval_design/anchor.json`). They are every benchmark with at least one eligible pair
(data.md §2: ≥ 80 distinct items with a first 0/1 response) that is:

* **binary, or mixed with a 0/1 share of at least 0.95.** The 29 mixed benchmarks below
  0.95 cannot be read without `grading_criterion` (data.md §2). The hidden test admits
  binary item-level benchmarks only (CLAUDE.md).
* **not one of the legacy six.** matharena, multi_swebench, real_webagents and
  researchcodebench are scored in the old RS regimes. swe_rebench has one subject.
  mmdocrag is fractional.

E keeps the nine direction-flagged benchmarks (lower_is_better or unordered). They are
part of the population the hidden benchmarks are drawn from, stored as is (data.md §2,
§7). They fill the bottom of the level distribution, so the tilted regimes lean on them.
Section 7 reports every contrast without them as well (descriptive). E keeps 12
single-pair benchmarks: a hidden benchmark can have a single subject.

The prior's training set is not E. It is code.md's rule (subject part and level part),
legacy five included, less the run's benchmarks.

### 3.2 The split into S and C (fixed now, by rule)

* **The rule.** Within each response kind (binary; mixed at a share ≥ 0.95), sort E's
  benchmarks by level, lowest first, with ties broken by name. The level is `level_cc` in
  `benchmarks.csv`, the mean corrected logit of the pairs' first responses (data.md §5).
  Cut them into consecutive blocks of 8. Within a block, rank the benchmarks by
  sha256 of `("pool-eval-split", name)`. The first round(3 × size / 8) go to S and the
  rest to C.
* **Why benchmarks and not seeds.** Between-benchmark variation dominates every
  comparison here. P1a found that fresh seeds on the same catalogue are not fresh data
  (P1a plan, amendment A5). Splitting by benchmark makes C's outcomes independent of the
  benchmarks that chose the candidate.
* **Why 3 to 5.** Selection only has to rank three candidates. The gate's power rests on
  C's number of benchmarks (section 7).

**Result** (`eval_design/feasibility.json: split`; it uses `feasibility.py`'s
implementation of this rule, which the script must reproduce exactly):

| half | benchmarks | level mean / sd | below -1 / -1.5 / -2 / -3 | single-pair | ≥ 4 pairs | flagged | mixed | benchmark-mean named share |
|---|---|---|---|---|---|---|---|---|
| S | 50 | +0.11 / 1.57 | 9 / 3 / 3 / 3 | 4 | 42 | 6 | 3 | 0.63 |
| C | 82 | +0.10 / 1.29 | 15 / 10 / 4 / 2 | 8 | 65 | 3 | 5 | 0.59 |

The halves share subjects (gpt 4o is on 52 benchmarks). Runs are drawn benchmark-first, so
this weakens their independence only through subject effects. Section 14 discusses it.

### 3.3 Priors per run: strict exclusion

For a pool run with benchmark set B_run, every candidate's bundle is
`pool.bundle(fits, exclude = B_run, ...)` (code.md §9).

* **What it uses.** The cached per-benchmark fits, so a bundle costs one `from_standings`
  (2.2 to 2.4 CPU s, code.md §6) plus fit_hyper's arithmetic. C1, C2, C3 and C23 share the
  standings rows and differ in hyperparameters only.
* **Why not folds.** Strict exclusion costs about 3 s a run (about 1 CPU hour over
  the plan). Folds would cost less, but they fail where it matters, and strict
  exclusion is also stricter than RS's per-target exclusion.
  * Per-target exclusion (`experiments/regime_sensitivity.py:652-700`) lets another of the
    run's benchmarks inform a subject's prior. On the hidden test such a benchmark is
    hidden too.
  * Folds of about 16 benchmarks cannot hold a formative-shaped run at a low level. A run
    needs about 6.6 distinct benchmarks, and a fold has 2 to 3 below -1.5.
* **Cache.** Bundles are cached on disk, keyed by the sorted B_run and the candidate
  block's digest.
* **Check.** `hyper.excluded` must equal B_run, and `included` together with
  `prior.meta['included']` must meet B_run in nothing (`paiec/hier.py:1847-1861` already
  compares the two).

On old RS runs the exclusion is the target's parent, as P1a's Maker did. That makes the
candidate arms `pool.bundle(fits, (parent,))` (code.md §9).

### 3.4 Items: a precondition the author decides

hier reads `item_content` (the multiple-choice floor, `paiec/hier.py:1944-1953`) and
`item_features` (group effects, `:532-542`). `items.parquet` was not fetched (data.md §6,
code.md §2). Two modes, chosen in amendment 4 before the lock:

* **Mode I (preferred): items fetched for E and the training set.** Pool runs then pass
  real item dicts, as the hidden test does. The group share is estimated from features.
  * E holds about 4.09 million items (`benchmarks.csv: n_items`). The disk had 13 GB free
    (98% used) on 2026-10-07.
  * So the item files' sizes are listed first, from repository metadata, without
    downloading. Mode I runs only if E plus the training set fit with at least 5 GB left
    free.
  * Item text never enters a committed file. Rows hold item keys only.
* **Mode B (items-blind).** Every pool run passes `item_content = ""` and
  `item_features = ""`, and interactors from the response rows. The floor and groups are
  then inert on pool runs for every arm.
  * Every pool arm fixes the group share at archive-3's 0.25 (code.md §2, §10.3). So no
    candidate changes the split of T, which this mode could not see. T itself
    (sigma_d² + sigma_g²), which every item uses, is still refitted and tested.
  * The old RS regimes keep their full legacy items in both modes.
  * The result is labelled items-blind in the results file and in the findings.

## 4 Regimes

### 4.1 Run shapes

**Formative-size (P- regimes).** These adapt `paiec/testlike.py`'s Sampler
(`:694-763`) to real benchmarks. The defaults come from formative run 1's shape, the use
App G.6 lists (`docs/report/draft.md:1675`).

* A run draws n ~ U{5, ..., 12} pairs one at a time, from the half's benchmarks.
* With probability `repeat` = 0.25, and if a benchmark already in the run has a free
  pair, the next pair joins one of those benchmarks. Otherwise it opens a benchmark not
  yet in the run. Either way the benchmark is chosen in proportion to the tilt weight
  w_b, and the pair uniformly among its free pairs.
* A free pair is one not yet drawn whose subject_id and eight visible fields are not yet
  in the run (testlike's rule, `:711-740`).
* Each pair keeps a uniform random subset of at most 160 items. Then
  `official.sample_run(cut, rng, cap=1000, n_pairs=len(cut), weighting="pair",
  min_items=80)` makes the proportional cut to the 1,000-item cap, as `Sampler.run`
  does (`paiec/testlike.py:742-763`, `paiec/official.py:114-186`).
* No date shift, no minimum release date, no recency weight. Pool subjects keep their
  real dates.

*Planning, C half* (`feasibility.json`):

* 8.4 to 8.6 pairs per run, on 6.5 to 6.7 benchmarks;
* a share of pairs alone in their benchmark of 0.60 to 0.63. App G.6's §2.4 count for
  runs 1 to 3 is 16 of 26, that is 0.615. It is quoted for comparison only, and nothing is
  tuned to it.

**Dense (D- regimes).** A plausible summative shape (P2 plan §6 asks the organisers), at a
cost this machine can carry.

* 7 distinct benchmarks drawn ∝ w_b among the half's benchmarks with at least 4 eligible
  pairs: 42 in S, 65 in C.
* For each benchmark, min(8, n_b) pairs uniformly without replacement, with distinct
  eight-field dicts.
* Each pair keeps at most 120 items. There is no cap.
* So a run has about 52 pairs and evaluates at most 60 items a pair. S3's dense
  composition had 140 pairs; this one is a third of it.

**Protocol.** Every run is played as `paiec/official.py` plays it:

* a persistent 50/50 split with split scope `'pair'`, as in RS (`:83-100`);
* the platform's sha256 random acquisition (`:310-322`), the first recorded response
  revealed (`:359`);
* every response of an evaluation item scored;
* one shared `labeled` per checkpoint (`:478-496`), and a fresh instance per checkpoint.

The harness is `level_calibration.checkpoints` and `evaluate`
(`experiments/level_calibration.py:508-561`), verified against `official.run_official` on
pool runs (check RP3, section 10). One instance per arm per checkpoint serves the whole
run, because every benchmark of a pool run has the same prior.

### 4.2 Level targets (global, from committed files)

The realised level is RS's (`experiments/regime_sensitivity.py:624-634`; P1a plan §4): the
continuity-corrected logit of each pair appearance's evaluated responses
(`testlike.describe`), with the unweighted mean and sd (ddof 0) over all appearances.

| regime | target | source |
|---|---|---|
| P-READING | mean -0.715, sd 1.772 | RS `TARGETS['READING']`: the pooled reading of runs 1 and 2 |
| P-AUDIT | mean -1.10, sd 2.00 | RS `TARGETS['AUDIT']`: the audit's reading of run 1 |
| P-TUNED | mean -1.290, sd 1.696 | RS `TARGETS['TUNED']`: the tuned regime's realisation |
| P-MIXTURE | share below -3 of 0.118; mean of the rest -0.226 | RS `TARGETS['MIXTURE']`; low component (-4.38, 0.171) as RS `MIX` |
| P-FLAT | none: w_b = 1, the half's own levels | the unselected held-out benchmarks |
| D-READING | as P-READING | |
| D-FLAT | none | |

The script imports these from `experiments/regime_sensitivity.py`, whose constants block
is digest-pinned. Nothing is re-derived from a feedback table.

**Not scored**, with the reason:

* **A blend of the reading and the pool.** Its centre (about -0.4) depends on an assumed
  SE of the reading, and READING and FLAT bracket it.
* **A pool regime with the date shift.** Real dates are the honest choice on real
  benchmarks. The ridge's date term on newer subjects stays untested (section 14).
* **Pair-weighted pool draws.** alignment_faking (4,278 pairs) and sib200 (3,341) would
  fill them, and the formative shape (7 benchmarks for 8 or 9 pairs) says benchmark-first.
* **Split scope `'benchmark'`.**

### 4.3 Tilt and knob search (composition only, no predictor)

* **The weight.** w_b ∝ min(t_b, 20 · mean t), with t_b = target density at the
  benchmark's level over a Gaussian-kernel base density of the half's levels (bandwidth
  0.7, each benchmark once). The target of P-MIXTURE is a two-component mixture, as
  `paiec/testlike.py:543-557, 622-645` builds one. The level is `level_cc`, the mean over
  the benchmark's pairs. So the tilt acts on benchmarks, not subjects, as RS's
  `tilt='benchmark'` did (`testlike.py` docstring).
* **The grid.**
  * Gaussian targets: m_b ∈ {-3.5, -3.375, ..., +0.5} × s_b ∈ {0.6, 0.8, ..., 2.2}.
  * The mixture: low weight ∈ {0.06, 0.10, ..., 0.30} × high mean ∈ {-1.5, -1.375, ...,
    0.0}, high sd 1.151 as RS.
* **The objective.** RS's: ((mean - m*)/0.05)² + ((sd - s*)/0.05)², and for the mixture
  ((share - 0.118)/0.01)² + ((rest - (-0.226))/0.05)² (`KNOB_SCALE`).
* **The draws.** 1,000 a setting on CAL_SEED, per half, on all-item pair logits
  (`pairs.csv: logit_cc`), which makes the search fast.
* **Fixed after the search.** The argmin and its realised mean, sd, share below -3 and
  effective number of benchmarks are written before the lock.

*Planning* (`feasibility.json`, 250 draws a setting, a coarser grid):

| regime | S: knobs, realised mean / sd | S: effective benchmarks | C: knobs, realised mean / sd | C: effective benchmarks |
|---|---|---|---|---|
| P-FLAT | none, +0.06 / 1.84 | 49.6 | none, +0.06 / 1.57 | 80.9 |
| P-READING | N(-0.75, 1.6), -0.70 / 1.76 | 39.4 | N(-0.875, 1.6), -0.72 / 1.80 | 65.1 |
| P-AUDIT | N(-1.25, 1.8), -1.08 / 1.97 | 34.6 | N(-1.125, 1.8), -1.06 / 1.99 | 56.0 |
| P-TUNED | N(-1.625, 1.2), -1.35 / 1.67 | 24.4 | N(-1.5, 1.4), -1.32 / 1.75 | 46.8 |
| P-MIXTURE | (0.10 at -4.38; 0.90 at -1.0), share 0.088, rest -0.48 | 35.0 | (0.10 at -4.38; 0.90 at -0.375), share 0.120, rest -0.27 | 53.2 |
| D-FLAT | none, -0.10 / 1.94 | 41.0 | none, +0.04 / 1.54 | 63.3 |
| D-READING | N(-0.875, 1.6), -0.68 / 1.76 | 34.9 | N(-0.875, 1.4), -0.71 / 1.73 | 53.1 |

Effective benchmarks are exp(entropy) of the benchmarks' appearance shares.

* **Every target is reached in C**, to within 0.04 on the mean (0.05 on P-MIXTURE's
  rest mean) and 0.05 on the sd.
* **In S, P-MIXTURE falls short of the share target** (0.088 against 0.118). S has 3
  benchmarks below -3, and E has 5 in all.
* **A regime that misses its target by more than 0.15** on the mean, or 0.03 on the share,
  is recorded as "short of target" with its realised level before the lock. It keeps its
  name and its place in the rule, as RS kept its milder realisations (P1a amendment A6).
* **The pool supports the tilts far better than the five legacy benchmarks did.** The
  tilted C regimes keep 47 to 65 effective benchmarks. RS's regimes rested on 4 parents
  (P1a plan §11).

### 4.4 The old RS regimes (the guard)

These are P1a's seven regimes on its scoring seed 11 (520 tasks: TUNED, READING, AUDIT,
MIXTURE and FLAT test-like at 80 runs each; R1B and R1P at 60 each), with RS's stored
compositions (`data/regime_sensitivity_rows/_compositions.json`) and A3's stored `ship`
rows. Only C1, C2 and C3 are scored there.

Their realised levels are P1a amendment A6's: READING -0.63 / 1.68, AUDIT -0.93 / 2.05,
and so on. They carry LEVEL's selection optimism: the same catalogue chose and guarded
LEVEL (P1a amendment A5). They are therefore a guard that leans towards A3, which suits
a no-loss bound.

S4's seed (20261006, `experiments/ablations_at_level.py`, 520 tasks with stored `ship`
rows) may be scored as a replicate. It is descriptive and outside the rule.

### 4.5 Realisation checks (before scoring)

For each P- and D- regime, the realised level of the planned scoring runs, on evaluated
responses, must lie within RS's `REALISE_TOL` of the calibration seed's value, which is on
all-item logits; the tolerance absorbs the small difference (`:289`):
mean ±0.30, sd ±0.30, share ±0.04, rest mean ±0.25. *Planning:* the 120-run block sds are
0.05 to 0.12 on the mean and 0.03 to 0.09 on the sd, so the bounds are about 3 block sds.
If a check trips, scoring does not start, the cause is found (a bug), and the knobs are
not re-tuned on a scoring seed.

## 5 What each row holds

The rows go to `data/pool_eval_rows/<half>/<regime>/<i>.json`, gitignored. Every planned
run's composition goes to `data/pool_eval_rows/<half>/_compositions.json` before scoring.

* **Per pair appearance:**
  * the benchmark (real name; caller only, never passed to predict);
  * its kind, binary or mixed, and its direction flag;
  * subject_id;
  * its identity key: `("n", canon_name)` or `("k", digest)`, as hier's
    `identity_key` (`paiec/hier.py:1918-1927`);
  * named or unnamed, and whether the name has a row in each arm's identity table;
  * release year;
  * kept and evaluated items, responses, the evaluated rate and logit;
  * the pair's v = mean of p(1 - p) at its evaluated rate.
* **Per arm and pair:** Brier and ECE at the six budgets, and the mean B0 prediction.
* **Per arm and run:**
  * calls, and the mean and max per call of CPU time (`time.process_time`) and wall;
  * fallbacks (`failures`, `unconverged`);
  * the bundle's key and digest, and its mu0, sigma_mu, T, sigma_theta, sigma_delta and
    id_cap.
* **Per task:** the resident set, the library and script digests, and the load average.

Rows hold no item text and no formative value.

## 6 Seeds and run counts

**Seeds.** None of these seeds has been used. Every run uses
`np.random.default_rng([seed, half, regime_code, i])`.

| seed | use |
|---|---|
| `SPLIT_SALT = "pool-eval-split"` | the split, section 3.2 |
| `CAL_SEED = 4001` | knob search and calibration draws; never scored |
| `SEL_SEED = 4002` | S half, every P- and D- regime |
| `CONF_SEED = 4003` | C half, every P- and D- regime |
| `TIMING_SEED = 4004` | the timing stage (S half); predictions timed, labels stripped, no Brier |
| RS seed 11 | the old RS regimes (stored compositions) |
| `BOOT_SEED = 0` | every bootstrap |

**Planned runs.**

| stage | P- regimes (5 each) | D- regimes (2 each) | RS | arms scored |
|---|---|---|---|---|
| S | 80 a regime (400) | 30 a regime (60) | 520 tasks | P-: A3, C1, C2, C3, C23. D-: A3, C1, C2, C3. RS: C1, C2, C3 |
| C, core | 120 a regime (600) | 50 a regime (100) | | A3 and the carried X* |
| C, extra | the same 600 runs | | | the other candidates; descriptive, after the gate is read |

* **Why 120 runs in C.** In the untilted regime that is about 1,020 pair appearances on
  82 benchmarks, about 12 per benchmark. A benchmark's mean difference is then dominated
  by its own effect, not by the draw. More runs barely shrink the cluster SE (P1a plan
  §5).
* **Task order.** Tasks run in run-index-major order (i, then regime), on 3 shard
  processes by task index, so any stopping point leaves every regime with the same
  prefix.
* **Resumability.** A worker exits with code 75 once its resident set passes 1.45 GB,
  P1a's guard, and resumes from there.

**The one allowed adjustment.** It is made at the timing stage, before the lock. If the
measured unit costs project the core above 30 process-hours:

1. the dense item cap drops from 120 to 80;
2. if the projection is still above 30 process-hours, the dense counts drop to 20 (S) and
   40 (C).

Formative counts never change. The counts written at the lock are final.

## 7 Statistics

The definitions follow `experiments/regime_sensitivity.py:1118-1210` (`RegimeRows`) and
`experiments/level_calibration.py:741-768` (`Boot`), with the cluster changed to the
benchmark.

* **Run ALC.** The mean over the run's pairs of 0.1 B0 + 0.2 (B1 + B3 + B7 + B15)
  + 0.1 B31. A pair appearance weighs w = 1 / (pairs in its run).
* **D(X, R).** The mean over R's planned runs of the run's pair-mean ALC of X minus that
  of A3, on identical runs and checkpoints.
  * With v_k = Σ w (ALC_X - ALC_A3) and n_k = Σ w over cluster k's appearances,
    D = Σ v_k / Σ n_k exactly.
* **SE_b.** The benchmark-cluster bootstrap: clusters are benchmarks, the ratio estimator
  (W·v)/(W·n), 2,000 multinomial resamples, seed 0.
* **SE_2.** The two-way SE. SE_s and SE_bs are the same bootstrap with the identity key
  and with the (benchmark, identity key) cell as cluster. Then
  SE_2 = sqrt(max(0, SE_b² + SE_s² - SE_bs²)). A model's standing recurs across
  benchmarks (gpt 4o is on 52), so subjects are crossed with benchmarks.
* **SE_g = max(SE_b, SE_2)** is the SE the gate reads. **U95 = D + 1.96 SE_g.**
* **PB(X, R).** Each benchmark's appearance-weighted mean difference D_b, averaged with
  each benchmark once. It guards against a few heavily drawn benchmarks carrying D.
* **JK(X, R).** max over benchmarks b of (Σ v - v_b) / (Σ n - n_b): D with one benchmark's
  appearances removed, the most favourable one.

**Reported for every arm pair and regime** (descriptive):

* D with run SE, SE_b, SE_s, SE_bs, SE_2 and the bootstrap's 2.5th and 97.5th
  percentiles;
* PB, JK, and the share of benchmarks with D_b < 0;
* per budget, D_B with SE_b;
* every arm's mean ALC, mean B0 prediction against the mean evaluated rate
  (calibration-in-the-large), ECE at B0 and over the ALC weights, and its fallback
  counts;
* the decomposition of section 2.3;
* D by subject kind: named with an identity row, named without one, unnamed (data.md §7
  asks for this, and it is where C3 acts);
* D without the flagged benchmarks, by response kind, by level tercile of the benchmark,
  and on pairs released on or after 2025-01-01;
* the v decomposition per arm: the pair-rate oracle v and ALC - v, as
  `results/hidden_runs.json` defines them, on the replica's own runs;
* the realised level of the scored runs (mean, sd, share below -3 and -4, SE_b) and their
  effective number of benchmarks;
* latency per arm: mean, p99 and max CPU and wall time per call, and the ratio to A3 in
  the same tasks;
* an importance-reweighted check: P-FLAT's appearances reweighted to each tilted target
  by benchmark level, against the tilted regime's own D.

**Sensitivity of the gate** (*planning*; the S half's measured SEs are recorded at
selection). SE_b ≈ σ_b / sqrt(n_eff), where σ_b is the sd across benchmarks of the
per-benchmark mean difference. In C, n_eff is about 65 (P-READING) and 56 (P-AUDIT).
Condition (a) passes in one regime with probability about Φ((Δ - max(0.002, 1.96 SE)) /
SE), for a true gain Δ:

| σ_b | SE (P-READING) | pass at Δ = 0.002 | 0.004 | 0.006 | false pass at Δ = 0 |
|---|---|---|---|---|---|
| 0.005 | 0.0006 | 0.50 | ≈ 1 | ≈ 1 | < 0.001 |
| 0.010 | 0.0012 | 0.39 | 0.92 | ≈ 1 | 0.025 |
| 0.020 | 0.0025 | 0.12 | 0.36 | 0.67 | 0.025 |

* **Hier-against-hier priors that differ.** RS measured cluster SEs of 0.0003 to 0.0042
  on 4 parents (P1a amendment A7). The pool's benchmark clusters see between-benchmark
  variation, which RS's (parent, subject) clusters did not.
* **In practice.** The rule finds a gain of 0.004 or more when the per-benchmark spread
  is moderate. A gain of 0.002 is a coin flip at best.
* **Joint passing.** (a) needs both P-READING and P-AUDIT. Their runs share C's
  benchmarks, so they are positively correlated. Under no true difference, both pass
  together with probability about 0.01. Only one candidate reaches C, so there is no
  multiplicity at confirmation.

## 8 Selection on S (before any C task is scored)

All of S's planned tasks and all 520 RS tasks are scored for every listed arm. Then the
`select` stage reads them once.

**Eligibility.** Candidate X ∈ {C1, C2, C3} is eligible if all four hold:

* **(E1) The old RS regimes.**
  * The mean over the seven RS regimes of D(X, RS-R) is at most +0.0005.
  * D(X, RS-R) is at most +0.002 in each test-like RS regime (TUNED, READING, AUDIT,
    MIXTURE, FLAT), and at most +0.001 in RS-R1B and RS-R1P. These are P1a's own (c) and
    (d) bounds.
* **(E2) No material loss on S.** D_S ≤ +0.002 in P-TUNED, P-MIXTURE and P-FLAT, and
  D_S ≤ +0.001 in D-FLAT and D-READING.
* **(E3) Latency, on S and RS tasks.** The mean CPU time per evaluation call is at most
  2.0 times A3's in the same tasks, and no formative-size call takes over 2.0 s of CPU.
* **(E4) Completeness.** Every planned S and RS task is scored for every listed arm, and
  every check of section 10 passed.

**Choice.** Among the eligible, take the lowest M(X) = [D_S(X, P-READING) +
D_S(X, P-AUDIT)] / 2. If another eligible candidate is within 0.0005 of that minimum and
comes earlier in the order C1, C3, C2, take it instead.

* X* is carried to C only if M(X*) ≤ -0.001.
* Otherwise the outcome is "no candidate carried: archive-3 stays". C may still be scored
  for the report, marked descriptive.

**Why the old RS guard is a mean plus RS's bounds, and not +0.0005 in each regime.** The
brief's example bound was "no loss larger than 0.0005 on the old RS regimes". Read per
regime, it would fail almost any change to the subject prior by noise alone. A candidate
with no true effect and a per-regime cluster SE of 0.0015 (RS's onepl had 0.0014 to
0.0017; eb_fit 0.0032 to 0.0042, P1a amendment A7) has each point estimate below +0.0005
with probability about 0.63. All seven would be, even with strong correlation, with
probability well under one half. The mean over seven regimes keeps the brief's +0.0005 on
an average that noise moves less. P1a's own bounds stop any single regime from losing
materially. **The literal per-regime +0.0005 is an option** the author can take instead
in amendment 4. It is stricter, and it leans further towards archive-3.

**Why these filters do not bias the confirmation.** (E1) to (E3) use only S and RS data,
and they bound losses rather than estimate a gain. The confirmation estimate on C stays
independent of how X* was chosen.

**The selection record.** It holds:

* every condition for every candidate;
* M(X) with SE_b;
* the S half's per-regime SE_b, and the power they imply for gains of 0.002, 0.004 and
  0.006;
* X*, and the time stamp.

It is written to `results/pool_eval.json: selection` and committed before the first
CONF_SEED task is scored. The commit is the outside time stamp P1a lacked (amendment A2).

## 9 The decision rule (to be quoted verbatim as `RULE_TEXT`)

```
POOL-EVAL DECISION RULE: does a pool-fitted hier replace archive-3?
(amendment 4 to docs/plans/p2_final_plan.md; docs/plans/p2_pool_eval.md;
experiments/pool_eval.py). Fixed before any run of SEL_SEED or CONF_SEED was
scored.

Terms
  A3        archive-3: paiec.hier.HierPredictor over submission/prior.json
            (LEVEL mu0 -2.5, sigma_mu 2.5, attr_scale 0.5; fitted on the five
            legacy benchmarks). On RS tasks: P1a's stored `ship` rows.
  C1,C2,C3  the candidates of CANDIDATES (code.md section 8: pool refit with
            LEVEL kept; with the pool's EB level; with the identity guards
            lifted), each built per run without every benchmark of the run.
  E, S, C   the 132 evaluation benchmarks and their selection (50) and
            confirmation (82) halves (SPLIT).
  P-READING, P-AUDIT, P-TUNED, P-MIXTURE, P-FLAT
            formative-size pool regimes; D-READING, D-FLAT dense pool regimes;
            RS-TUNED, RS-READING, RS-AUDIT, RS-MIXTURE, RS-FLAT, RS-R1B, RS-R1P
            P1a's regimes on its scoring seed 11 (520 tasks).
  D(X,R)    the mean over R's planned runs of the run's pair-mean ALC of X
            minus that of A3, on identical runs and checkpoints.
  SEg(X,R)  max(SEb, SE2): SEb the cluster-bootstrap SE over benchmarks (ratio
            estimator, 2,000 resamples, seed 0); SE2 the two-way SE
            sqrt(max(0, SEb^2 + SEs^2 - SEbs^2)) from the same bootstrap with
            the subject identity key and the (benchmark, key) cell as clusters.
  U95(X,R)  D(X,R) + 1.96 SEg(X,R).
  PB(X,R)   the mean over benchmarks, each once, of the benchmark's
            appearance-weighted mean difference.
  JK(X,R)   the largest D(X,R) with all appearances of one benchmark removed.

Selection (S and RS only; recorded and committed before any CONF_SEED task
is scored). X in {C1, C2, C3} is eligible iff
  (E1) the mean of D(X,RS-R) over the seven RS regimes is <= +0.0005, and
       D(X,RS-R) <= +0.002 in each RS test-like regime and <= +0.001 in
       RS-R1B and RS-R1P;
  (E2) on S: D <= +0.002 in P-TUNED, P-MIXTURE and P-FLAT, and D <= +0.001 in
       D-FLAT and D-READING;
  (E3) on the S and RS tasks: X's mean CPU time per evaluation call is at most
       2.0 times A3's in the same tasks, and no formative-size call of X takes
       more than 2.0 s of CPU;
  (E4) every planned S and RS task is scored for every listed arm and every
       check passed.
  Among the eligible, X* minimises M(X) = [D_S(X,P-READING) +
  D_S(X,P-AUDIT)] / 2; an eligible candidate within 0.0005 of the minimum
  that comes earlier in the order C1, C3, C2 is taken instead. X* is carried
  iff M(X*) <= -0.001; otherwise the outcome is "no candidate carried:
  archive-3 stays".

Gate (C only; read once, on the full planned set). X* replaces A3 iff
  (a) in P-READING and in P-AUDIT: D <= -0.002 and U95 < 0;
  (b) in P-READING and in P-AUDIT: PB <= 0 and JK <= -0.001;
  (c) in P-TUNED, P-MIXTURE and P-FLAT: D <= +0.002;
  (d) in D-READING and D-FLAT: D <= +0.001;
  (e) (E1) holds for X*;
  (f) on the C tasks: X*'s mean CPU time per evaluation call is at most 2.0
      times A3's; no formative-size call takes more than 2.0 s of CPU; dense
      calls have a p99 of at most 1.0 s and a maximum of at most 30 s of CPU;
  (g) every check passed and every planned C task is scored for A3 and X*.

Reading
  1. Pass: X* is built as archive-4 (CANDIDATES' rule on every benchmark of
     the training rule, nothing excluded). It ships only if
     tools/build_submission.py passes every check and the organisers'
     validator, S3's gate (P2 plan section 3) holds on archive-4, and the one
     regression submission of P2 plan section 4 is scored with finite outputs
     no later than 2026-10-16. Otherwise archive-3 stays.
  2. Fail, or no candidate carried: "archive-3 stays", recorded as a negative
     result with every D, SE and condition, for every candidate scored.
  3. Everything else (the decomposition, budgets, subject kinds, flags,
     tercile and recency cuts, v decomposition, reweighting, latency tables,
     the C-extra arms) is descriptive and changes nothing.
  4. Nothing in this rule, CANDIDATES, SPLIT, the regimes and their knobs,
     the seeds, the run counts or the statistics changes once a row of
     SEL_SEED exists. No run is added or dropped after rows are read; each
     stage's rule is applied once, to its full planned set. If a check fails,
     the cause is fixed, every affected task is re-scored in full and the
     deviation is recorded; the rule is not rewritten. If the gate has not
     been read by 2026-10-13 23:59 UTC, the outcome is "not read in time:
     archive-3 stays".
  5. Formative feedback enters only as RS's level targets and testlike's run
     shape. results/hidden_runs.json is read once, after the gate, for the
     descriptive placement of eval.md section 12. Formative run 3 sets nothing.
```

**What the rule adds to the brief's floor.**

* **(b).** Many benchmarks carry the gain, not a few heavily drawn ones. This is the
  analogue of RS's parent-level sign.
* **(d).** The summative shape is unknown, and the dense regimes are its plausible other
  form.
* **(e).** It is stated on C as well, though its data are not C's, so that the gate
  lists every condition the archive must meet.
* **(f).** CPU time rather than wall time. S3 had to disclose a wall-clock deviation
  under a load of 23 to 235 (P2 amendment 1).
* **The deadline clause** in reading 4.

Nothing is loosened against RS's rule, except (E1)'s form, which is explained in
section 8.

## 10 Checks before scoring

Every stage after `lock` re-runs the lock check and refuses to run when it fails. The
lessons of P1a's amendments A1 (tolerances at the stored precision), A2 (commit before
scoring), A3 (one script digest for every row) and A9 (iterate sorted containers; set
`PYTHONHASHSEED=0`) are built in.

1. **`lock`.**
   * sha256 of `RULE_TEXT`;
   * the plan file `docs/plans/p2_pool_eval.md` (this design, condensed), which quotes
     `RULE_TEXT` verbatim;
   * the constants block: CANDIDATES, SPLIT, regimes and knobs, seeds, counts and the
     rule's numbers;
   * the digests of the shipped modules, which must equal HEAD a725df9's
     (`paiec/hier.py` d9a9561202b63b9d, `paiec/prior.py` 3a8a8c7dcd0da53e,
     `submission/prior.json` c7ce3b845d584f95). The one exception is a `prior.py`
     refactor that amendment 4 names (code.md §9); then check RP4 must hold.
   * The plan, the script and its tests are committed together before the first
     SEL_SEED task is scored.
2. **`split` and `regimes`.** The split of section 3.2 and its balance table. The knob
   search on CAL_SEED (section 4.3). The planned compositions of every half and regime,
   with the realisation checks (section 4.5).
3. **`fits`.** `pool.fit_all` at code.md's caps, cached on disk, and the fixed-T pass for
   C2. The report records convergence, the degenerate fits and the rule applied to them.
4. **`reproduce`.**
   * **RP1.** A3 re-scored on P1a's seed-11 runs 0 and 1 of each of the seven regimes (14
     tasks) equals the stored `ship` rows per pair and budget to 1e-9. P1a's review
     re-scored the last run of every regime and found every Brier identical, largest
     difference 0.0 (amendment A9). The rows are full-precision floats, so 1e-9 is at the
     stored precision (the A1 lesson).
   * **RP2.** The archive-3 runtime, as `tools/build_submission.py` `check_run` exercises
     it, and the in-repo HierPredictor give bit-identical predictions over
     `submission/prior.json` on two TIMING_SEED pool runs. Predictions are compared with
     each other, not with labels.
   * **RP3.** On one formative-size and one dense TIMING_SEED run, the harness and
     `official.run_official(..., workers=1)` give identical predictions for A3 and C1, as
     `level_calibration.py --verify` does for the test-like runs.
   * **RP4.** `pool.bundle` on the legacy five with the archive-3 options rebuilds
     `submission/prior.json` byte for byte, as `tools/build_submission.py`'s default path
     must, and `PR.build(pairs, (parent,))` is unchanged for each legacy parent. This
     guards the A3 rows that RS stored.
   * **RP5.** On ten random exclusion sets, every candidate bundle loads through
     `from_json` and `HierPredictor`, and its exclusion lists are right (section 3.3).
5. **`timing`** (before the lock). Six formative-size and two dense TIMING_SEED runs,
   every arm. The evaluation targets carry no label into the stage, so no Brier can be
   computed, and the code asserts this. It records CPU and wall time per call and per run,
   bundle build time and the resident set. It projects the plan's total, and the
   adjustment rule of section 6 is applied. P1a's probe printed ALCs on old runs and had
   to disclose them (P1a plan §12); this stage cannot.

**Library work this needs** (code.md §9 already plans most of it):

* `paiec/pool.py` with `bundle(fits, exclude, ...)`;
* the pool sampler and statistics in `experiments/pool_eval.py`, not in a shipped module;
* `tests/test_pool_eval.py`, covering:
  * the sampler's invariants: half-only benchmarks, distinct pairs and dicts, at least 80
    kept items, the 1,000 cap, reproducibility, and uniform weights without a tilt;
  * the split rule;
  * the ratio-estimator identity D = Σv/Σn, the jackknife, and the two-way SE on
    synthetic clusters;
  * the selection and gate functions on synthetic tables;
  * that no stage before `describe-hidden` opens `results/formative*` or
    `results/hidden_runs.json`.

`pytest` stays green.

## 11 Compute budget and wall time on this machine

**The machine.** An Apple M1 Pro with 8 cores and 16 GB (code.md). On 2026-10-07 two other
jobs used about 5.3 cores and 3 GB, with load averages between 5.8 and 23 that
afternoon. The plan therefore runs 3
shard processes, BLAS pinned to one thread, at most 1.45 GB each.

**Unit costs** (*planning*; the timing stage re-measures them):

| unit | cost | source |
|---|---|---|
| hier arm, formative-size run (about 8.5 pairs, about 500 evaluated inputs, six checkpoints) | 6.0 s | P1a plan §6: 5.2 to 6.4 s per hier config per test-like run at a load near 7 |
| bundles of one run (one `from_standings`, four hyperparameter sets, C2's EB summary) | 3 s | code.md §6: 2.2 CPU s per exclusion |
| hier arm, dense run (about 52 pairs, up to 60 evaluated items each) | 80 s (uncertain by 2x) | S3's dense composition (140 pairs, 1,400 targets) took 210 s wall over six checkpoints at a load of 99 to 235 (`results/stress_archive.json: dense`) |
| hier arm, RS task | 5.3 s | P1a plan §6 |
| per-benchmark fits, about 166 benchmarks at code.md's caps, plus the fixed-T pass | about 900 CPU s | code.md §6 (750 CPU s; 138 CPU s for the prototype's pass) |

**Totals.**

| stage | work | process-seconds |
|---|---|---|
| fits | one time | 900 |
| regimes | 1,000 draws × about 300 settings × 7 regimes × 2 halves, all-item logits; then the planned compositions | 1,800 |
| reproduce | 14 RS tasks; 4 pool runs | 300 |
| timing | 6 formative-size + 2 dense runs, every arm | 1,000 |
| S, P- | 400 runs × (5 arms × 6.0 + 3) | 13,200 |
| S, D- | 60 runs × (4 arms × 80 + 3) | 19,400 |
| RS guard | 520 tasks × 3 arms × 5.3 | 8,300 |
| C core, P- | 600 runs × (2 arms × 6.0 + 3) | 9,000 |
| C core, D- | 100 runs × (2 arms × 80 + 3) | 16,300 |
| summarise | 3 bootstraps × 14 regimes × arm pairs | 900 |
| **core total** | | **about 71,100 (19.8 process-hours)** |
| C extra (descriptive, after the gate) | 600 runs × 3 arms × 6.0 | 10,800 |

**Wall time on 3 shards.**

* **Up to the selection record:** 44,900 s, or 4.2 hours. At a 1.5x slowdown beside the
  other jobs, about 6.2 hours.
* **C core:** 25,300 s, or 2.3 hours. About 3.5 hours at 1.5x.
* **C extra:** about 1.5 hours.

**The dense runs carry half the cost.** If they run twice as slow as planned, the core
grows to about 29.5 process-hours, just under section 6's 30-hour trigger. Anything
slower trips the trigger, and the dense cap and counts drop as section 6 says.

**Memory.** P1a's process peaked at 1.24 to 1.30 GB with instances released after each
checkpoint (P1a plan §6). Three shards come to at most 4.4 GB beside the other jobs. A
pool run loads only its own pairs' cells: about 12 pairs of at most 160 items, or 56 of
at most 120. Fitting stays in the `fits` stage at code.md's caps, with at most 1.5 GB per
worker.

**Disk.**

* The rows take about 50 to 100 MB; P1a's 520 tasks of 11 configs took 15 MB.
* The bundle cache holds one subject prior per run (about 1,160 runs, about 85 KB each)
  and the runs' hyperparameter sets beside them: about 0.1 GB.
* Mode I needs the item files on top, so their sizes are listed first (section 3.4).

## 12 The formative runs: what is used, and the one check after the decision

**Used, as App G.6 allows** (`docs/report/draft.md:1669-1689`; P2 plan amendments 2 and 3):

1. **Global level targets.** RS `TARGETS` for READING, AUDIT, MIXTURE and TUNED. These
   come from the pooled reading of runs 1 and 2 and from run 1's audit and lower-root
   reading, numbers already committed. They are quoted, not recomputed. G.6 lets the
   feedback set global quantities of the level distribution and nothing keyed on an id.
2. **Run shape.** testlike's defaults (n_pairs 5 to 12, at most 160 items a pair,
   repeat 0.25, the 1,000 cap), which run 1's shape set.

**Not used:**

* no per-pair value;
* no anonymous id;
* no formative table during any stage that can affect the decision;
* nothing from run 3 before the gate is read; after it, only its run-level v and ALC - v,
  as below;
* S4's hidden-v weights (`HV_COUNTS`) before the gate is read; after it, only as in the
  last item below.

**After the gate is read and recorded: `describe-hidden`, descriptive only.** It reads
`results/hidden_runs.json` once. It takes `decomposition.run{1,2,3}.v`, `.alc_minus_v` and
`.pairs`: run-level aggregates with no ids. The values are v = 0.1536, 0.1797 and 0.1608,
and ALC - v = 0.0577 (run 1, LegacyP), 0.0130 (run 2, hier) and 0.0208 (run 3,
archive-3). It writes:

* **where each hidden run's v falls** in each P- regime's distribution of run v over C
  runs of 8 or 9 pairs, as a z-score and a percentile, as
  `results/hidden_runs.json: replica.placement_z` does for the RS regimes;
* **where runs 2 and 3's ALC - v fall** in A3's replica distribution of ALC - v in each
  P- regime, and the carried candidate's replica ALC - v beside it. No hidden number
  exists for a candidate;
* **if archive-4 ships,** the regression submission's ALC - v beside run 3's, by P2 plan
  §4's rule ("selects and tunes nothing and is compared with nothing to choose the
  final");
* **D reweighted to the formative runs' v-bin counts**, through S4's `HV_COUNTS`, the
  three counts only.

This placement says which pool regime the hidden runs resemble. It is written after the
decision, so it cannot move it.

**Amendment 4 must add to App G.6 the three new uses:**

* the regimes' targets reused for the pool evaluation;
* the post-decision placement;
* S4's v-bin counts reused.

It must also confirm, in the author's words, that nothing else was read.

## 13 Schedule and fallback

| date (2026) | work |
|---|---|
| Wed 10-07 | data.md, code.md, this design |
| Thu 10-08 | `paiec/pool.py` and tests (code.md §9); decide Mode I or B, list item sizes, fetch if Mode I; `fits`; `experiments/pool_eval.py` stages `split`, `regimes`, `reproduce`, `timing`; amendment 4 committed with the plan, `RULE_TEXT` and constants (the lock) |
| Fri 10-09 | S scoring and the RS guard (about 6 h wall); `select`; selection record committed |
| Sat 10-10 | C core scoring (about 3.5 h); `summarise`; the gate read once; C extra afterwards |
| Sun 10-11 to Mon 10-12 | if it passes: archive-4 built (`tools/build_submission.py --pool`); every check and the validator; S3 on archive-4; an amendment announcing the one regression submission |
| Tue 10-13 to Thu 10-15 | the regression submission and its finiteness read; `describe-hidden` |
| Fri 10-16 | the final archive recorded |

**Hard stops.** If the gate has not been read by 2026-10-13 23:59 UTC, archive-3 stays.
If archive-4 has not been uploaded and scored finite by 2026-10-16, archive-3 stays. The
report work of P2 plan §7 (findings and R1 to R6 from 10-13) continues alongside. This
evaluation's outcome, pass or fail, enters the findings with its script.

## 14 What this cannot settle

* **The hidden reading's uncertainty.** READING and AUDIT come from 17 pairs on 7
  benchmark ids, with a level-mean SE of about 0.5. The tilted regimes bracket the
  reading's centre, not its uncertainty.
* **Whether the summative draws the same 7 benchmarks.** If it draws afresh from the
  population, P-FLAT is the relevant regime, and the rule asks only that a candidate not
  lose there.
* **Items.** In Mode B, the floor and group effects are untested on pool runs. The old RS
  regimes test them on the legacy items only.
* **Subjects.**
  * The hidden subjects' names, dates and records in the public pool are unknown. Pool
    runs carry the pool's own mix: 58% to 63% of pairs named per benchmark, real dates,
    mostly 2024.
  * The pool ridge's date term on newer subjects is untested.
  * The named and unnamed breakdown shows how much C1 and C3 lean on the identity table.
* **Shared subjects across S and C.** A candidate that helps or hurts on gpt 4o helps or
  hurts in both halves. Benchmark-first draws limit this, and SE_2 sees it within C, but
  not the dependence between S and C.
* **One split.** Another split would give other numbers. The design does not resample
  it, because that would reopen selection.
* **Leakage through shared quantities.**
  * C2's level is computed at the full pool's T, which saw each held-out benchmark's item
    variance through a median of about 120. Section 2.2's check records how much it
    moves.
  * Every candidate's build choices were made looking at the whole pool's descriptive
    statistics, confirmation benchmarks included, though at no outcome.
* **Protocol unknowns.**
  * The split scope (`'pair'` only).
  * Which recorded response is revealed (the first).
  * How the platform treats repeats and multi-condition cells (wmt_mqm: data.md §2). Here
    every response of an evaluation item is scored.
  * The dense shape, a guess at a third of S3's.
* **Power.** A true gain under about 0.003 will often be missed. The rule errs towards
  archive-3, which is the fallback anyway.

## 15 What amendment 4 must contain

1. **The decision to unfreeze**, scoped to `prior.json`, the `LEVEL` literal (C2 only)
   and the offline code of code.md §9. archive-3 stays the fallback.
2. **code.md §10's eight choices, and C2's three** (section 2.2). They go into
   `CANDIDATES`.
3. **Mode I or Mode B** (section 3.4), with the item sizes listed.
4. **This design's split, regimes, seeds, counts, statistics and `RULE_TEXT`**, in
   `docs/plans/p2_pool_eval.md`, together with the choice between (E1) as written and the
   literal per-regime +0.0005 (section 8).
5. **The App G.6 additions** of section 12.
6. **The schedule and the hard stops** of section 13.
7. **The commit**, made before any SEL_SEED task is scored.

## Pointers

* **This design's descriptive scripts.** `eval_design/feasibility.py` and `.json`;
  `eval_design/anchor.py` and `.json` (the archive-3 ridge puts E's pairs at a
  benchmark-weighted mean attribute score of -1.03, the prototype pool ridge at -0.02);
  `eval_design/level_held.py` and `.json`.
* **What the arms rest on.**
  * The P1a study: `docs/plans/p1a_regime_sensitivity.md`,
    `experiments/regime_sensitivity.py`, `results/regime_sensitivity.json`.
  * The acceptance harness: `experiments/harness.py` (gate `GATE`, `:237`).
  * The official replica: `paiec/official.py`.
  * Test-like runs: `paiec/testlike.py`.
  * The prior: `paiec/prior.py`.
* **The plan and its amendments.** `docs/plans/p2_final_plan.md`: §3 (S3's gate), §4 (the
  regression submission), §6 (the summative question) and amendments 1 to 3. App G.6 is
  `docs/report/draft.md:1669-1689`.
* **Input digests at writing** (sha256, first 16 hex):

  | file | digest |
  |---|---|
  | `paiec/hier.py` | d9a9561202b63b9d |
  | `paiec/prior.py` | 3a8a8c7dcd0da53e |
  | `paiec/official.py` | c70684dc15b560bc |
  | `paiec/testlike.py` | 00e96c4c1b5ee5fd |
  | `experiments/level_calibration.py` | 0d810dddfde74741 |
  | `experiments/regime_sensitivity.py` | af80ed50bc02a92d |
  | `submission/model.py` | a9561290e62d3f7c |
  | `submission/prior.json` | c7ce3b845d584f95 |
  | `results/hidden_runs.json` | 442f2bd1e980c368 |
  | `results/regime_sensitivity.json` | 74c06d5247bd59b1 |
