## Amendment 4 (2026-10-07): the expanded pool, and a gate for refitting hier's priors on it

Written on 2026-10-07, before any predictor was scored on the expanded pool. The
work behind it counted benchmarks and pairs, fitted per-benchmark item models,
drew run compositions and did level arithmetic. It computed no ALC and no Brier
on any data. Its design notes are `data.md`, `code.md` and `eval.md` with their
scripts, under `data/p2_work/pool_design/` (gitignored). Every number below
marked *planning* comes from them. The `fits`, `regimes` and `timing` stages
re-measure those numbers before the lock.

This amendment is committed before `experiments/pool_eval.py` scores any task of
`SEL_SEED`, `CONF_SEED` or the RS guard. That script quotes section 4.7 verbatim
as `RULE_TEXT`. Its lock hashes three things: the text of this amendment, from
its heading to the next `## Amendment` heading; `RULE_TEXT`; and its constants
block. Only a later dated amendment changes anything here.

In this amendment, "P2 §n" is section n of this plan, and "section 4.n" is a
section of this amendment.

### 4.1 Why the code is unfrozen

**The pool grew.** On 2026-10-06 and 2026-10-07, measurement-db grew from the
six legacy benchmarks to 255 catalogue entries. 252 are public: 249 new and the
legacy six.

* **The fetch.** We fetched revision `54074453da7b9911be0f48e4c560be547bc84d21`
  on 2026-10-07T13:12:36Z with `tools/fetch_pool.py` (a725df9). It went into
  `data/pool/` (manifest `data/pool/manifest.json`, sha256 `a58a23fb78642c14…`).
* **Eligible pairs.** 166 public benchmarks have at least one eligible pair,
  13,499 pairs in all. 137 benchmarks meet the rule of section 4.3 (data.md §2,
  §7).

**Pools are exchangeable.** The competition page says that benchmarks were
"randomly assigned to separate training and test pools before curation". It
also says that the collections may grow while the pool assignments and the
sampling design stay fixed. So the public pool and the hidden test pool are
random draws of benchmarks from one population. The growth adds training
benchmarks and does not change what the test samples. A public benchmark held
out of a fit therefore stands in for a hidden one, and the pool has enough of
them, which five benchmarks never had.

**What archive-3 rests on.** Five benchmarks and 221 pairs fixed both parts of
archive-3's bundle:

* **Subject prior.** The attribute ridge has 17 coefficients and 7 providers.
  The identity table has 106 rows, 21 of them for linked names.
* **Widths.** T = sigma_d² + sigma_g² is 9.51, sigma_delta 2.38, sigma_attr
  1.02 and sigma_theta 0.1 (`submission/prior.json`; code.md §3.2).

The pool offers much more (data.md §4, §7):

* **For the subject prior:** 128 benchmarks and 8,388 eligible pairs, carrying
  583 canonical names. 340 of those names appear on two or more benchmarks.
* **For the item variance and the level:** 137 benchmarks.

**The difficulty is the level.** The pool's benchmark levels have mean +0.06
and sd 1.54 (data.md §5). A level here is the mean continuity-corrected pair
logit, with each benchmark counted once. The median of the shipped level prior
sits at their 15.7th percentile.

The formative readings sit low:

* READING, the pooled reading of runs 1 and 2: mean -0.715, sd 1.772.
* AUDIT, run 1's audit: -1.10, sd 2.00.
* TUNED: -1.29, sd 1.70. This is the realisation of the regime that run 1's
  lower-root reading set (-1.6, sd 1.5 on plain logits). That reading is now
  known to be biased (P2 §5, R2).

These readings rest on 17 pairs from 7 benchmark ids, and the SE of their level
mean is about 0.5. The summative runs likely draw on the same 7 benchmarks.

A level learned from the pool wins on unselected held-out public benchmarks by
construction. Such benchmarks therefore cannot decide alone. This amendment
answers in three ways:

1. It builds the candidates so that the level is separated from the rest of
   the prior.
2. It scores them in pool regimes tilted to the readings and in untilted ones,
   as the RS study did.
3. It requires gains in the regimes of the readings and only bounds losses
   elsewhere.

**Decision of the author (2026-10-07).** Amendment 1's code freeze is lifted
for hier's priors only, under the gate of this amendment. The decision is due by
2026-10-16, and archive-3 is the fallback at every step.

### 4.2 Scope of the unfreezing

* **What may differ in archive-4.** For C1 and C3, archive-4 differs from
  archive-3 in `prior.json` alone. For C2 it also differs in the `LEVEL` literal
  of `submission/model.py` (line 68), and in nothing else in that file.
  model.py's docstring stays as written: archive-4's "fitted on every eligible
  public pair" (lines 3-8) means the pairs of section 4.3's rule, and the
  findings say so.
* **What stays byte-identical** (lock-checked; sha256, first 16 hex, at
  a725df9):

  | file | sha256 |
  |---|---|
  | `paiec/__init__.py` | `e3b0c44298fc1c14` |
  | `paiec/hier.py` | `d9a9561202b63b9d` |
  | `paiec/prior.py` | `3a8a8c7dcd0da53e` |
  | `paiec/predict.py` | `46193d8dd7701ec2` |
  | `paiec/fitting.py` | `ec3af9224f48309a` |
  | `paiec/irt.py` | `3f7d93e5d012699c` |
  | `paiec/subjects.py` | `79eaa16553a90e6b` |
  | `paiec/mcq.py` | `f240425d01fb7278` |

  `fit_hyper` is not refactored. Its arithmetic (`paiec/prior.py:756-807`) is
  copied into the new offline module and checked against it (RP4).
* **New or changed offline code:**
  * `paiec/pool.py`, which is not shipped and never imported by a shipped
    module;
  * `tools/pool_benchmarks.json`, the committed training lists;
  * `tools/build_submission.py`, which gains a `--pool` path;
  * `experiments/pool_eval.py` and its tests.
* **S2 stays closed.** Nothing else ships.
* **archive-3 stays at hand.** `--pool` overwrites `dist/paiec.zip` and
  `submission/prior.json`, so before any build:
  * `dist/paiec.zip` (sha256 `4a882cc7d410e6a9…`) is copied to
    `dist/archive-3.zip`;
  * its `prior.json` (sha256 `c7ce3b845d584f95…`) is copied to
    `data/pool_eval/a3_prior.json`.

  The comparator reads that copy after checking its digest.

### 4.3 Data, eligibility and folds

**Files.** The pool is measurement-db at revision `54074453`, under
`data/pool/<b>/formatted_tables/{benchmarks,subjects,responses}.parquet`.
`items.parquet` is not fetched (Mode B, below). A later revision is a new fit
and needs a new amendment.

**The level part L.** A benchmark is in L if it meets three conditions:

* its catalogue entry has `release` public and `granularity` item;
* its `response_type` is binary, or mixed with a 0/1 share of at least 0.95
  among its non-null responses;
* it has at least one eligible pair.

That is 137 benchmarks at this revision: 128 binary and 9 mixed, the legacy
five included. matharena is mixed at a share of 0.9991. mmdocrag is fractional,
so it is not in L. The other 29 mixed benchmarks cannot be read without
`items.grading_criterion`, and the remaining types are not binary (data.md §2).

**The subject part U.** U is L less the nine direction-flagged benchmarks:

* `lower_is_better`: crow, emoji_attack, harmbench, jailbreakbench,
  llmail_inject and wmt_mqm;
* `unordered`: alignment_faking, safeagentbench and tensortrust.

U has 128 benchmarks and 8,388 eligible pairs; 115 benchmarks have two or more
pairs. In the flagged nine a stronger subject scores lower, or the subjects have
no order. Their standings would enter the ridge and the identity table with the
wrong sign or with none. Their levels and item variances belong to the
population, so they stay in L.

**Pairs.**

* A pair is (subject_id, benchmark). Its responses are the 0/1 values of
  `responses.parquet`, in file order. Nulls and other values are dropped.
* An item's label is its first 0/1 response in file order. That is the label
  acquisition reveals (`paiec/official.py:356-359`) and the label
  `benchmark_fit` fits (`paiec/prior.py:250-260`).
* A pair is eligible with at least 80 distinct items
  (`paiec/official.py:66-80`).
* The subject is the eight visible fields of `subjects.parquet`
  (`paiec/data.py:20-21`). `display_name` is never read.
* Items carry `item_content` "", `item_features` "" and `interactors` "". This
  is Mode B, below; the shipped loader also blanks interactors
  (`paiec/data.py:41`).

**Caps for the fits.** The caps apply to the fits only, never to evaluation
runs.

* **Items.** Each pair keeps its first M_p = 2,000 items in one benchmark-wide
  order: ascending `stable_hash('pool-item', b, item_id)`. On a dense benchmark
  this equals a benchmark cap of 2,000. On a sparse one it keeps every pair:
  sib200's pairs hold 204 of 83,640 items each, and a benchmark-wide cap
  dropped them all (code.md §2).
* **Pairs.** Each benchmark keeps K = 100 pairs, in round-robin order over
  canonical names: ascending (j, `stable_hash('pool-name', c)`).
  * c is the pair's `canon_name` (`paiec/prior.py:84-111`), or its subject_id
    when it has none.
  * j is the pair's rank among the benchmark's pairs with the same c, by
    `stable_hash('pool-pair', b, subject_id)`.
  * `stable_hash` is `paiec.evaluator`'s.
* **Why that order.** The same names come first on every benchmark, which
  keeps links. A name with many pairs on one benchmark takes one slot per round.
* *Planning* (`amend4_order.py`, `amend4_order.json`), on U at K = 100:
  * 3,136 rows;
  * 222 names on two or more benchmarks, against 176 in subject-id order and
    212 in a plain name-hash order;
  * on alignment_faking, which is in L only, all 74 names are kept, against 6.

**Mode B: items-blind.** `items.parquet` is not fetched, for three reasons:

* E (below) holds about 4.09 million items;
* the disk had 14 GB free (97% used) on 2026-10-07;
* the fetch writes item text under `data/`.

What follows from that:

* **Every fit and every pool run sees blank item fields.** hier's
  multiple-choice floor (`paiec/hier.py:1944-1953`) and its group effects
  (`:532-542`) are inert on pool data for every arm, the comparator included.
* **The group share is not refitted.** Every candidate fixes it at 0.25, which
  is G_CAP and archive-3's realised share (code.md §2). So no candidate changes
  how T splits between sigma_d and sigma_g. T itself is refitted.
* **The RS guard keeps full items.** It runs on the legacy benchmarks with their
  items (section 4.5), so the floor and the group effects are exercised there.
* **Labelling.** Results and findings are labelled items-blind.

**The evaluation set E.** E is L less the legacy five: 132 benchmarks and
12,668 eligible pairs. It includes the nine flagged benchmarks and 12
single-pair benchmarks, since a hidden benchmark can have a single subject. The
legacy five enter the evaluation only through the RS guard.

**Folds.**

* **Outer split: selection and confirmation.** E is split once into a
  selection half S of 50 benchmarks and a confirmation half C of 82. The rule:
  1. Within each response kind (binary; mixed), sort E's benchmarks by
     `level_cc` in `benchmarks.csv`, lowest first, ties broken by name.
     `level_cc` is the mean corrected logit of the pairs' first responses.
  2. Cut the sorted list into consecutive blocks of 8.
  3. Within a block, order the benchmarks by the sha256 hex digest of the UTF-8
     bytes of Python's `repr(("pool-eval-split", name))`.
  4. The first `round(3 * size / 8)` (Python's `round`) go to S and the rest
     to C.

  *Planning* (`eval_design/feasibility.py`): S has 50 benchmarks, level mean
  +0.11 and sd 1.57, 6 flagged and 3 mixed. C has 82, level mean +0.10 and sd
  1.29, 3 flagged and 5 mixed. The script recomputes the split and must
  reproduce `feasibility.json`'s halves exactly.
* **Inner exclusion: the run.** For a pool run with benchmark set B_run, every
  candidate is built without B_run: its training set is L ∖ B_run, and U ∖ B_run
  for the subject part. No benchmark of the run informs any target's prior,
  which is the situation on the hidden test. The other benchmarks of both
  halves stay in training. The halves keep selection outcomes apart from
  confirmation outcomes; they do not split the training data, which the hidden
  test shares with every public benchmark.
* **RS guard: the target's parent.** On RS tasks each target's prior is built
  without its legacy parent, as P1a's Maker did
  (`experiments/regime_sensitivity.py:652-700`). The candidates then meet A3's
  stored rows on P1a's convention.
* **No k-fold scheme, for two reasons:**
  * Folds of about 16 benchmarks cannot hold a formative-shaped run at a low
    level. A run needs about 6.6 benchmarks, and a fold has 2 or 3 below -1.5.
  * The per-benchmark fits do not depend on the exclusion set, so strict
    exclusion costs one `from_standings` per run (about 2.2 CPU s; code.md §6).

### 4.4 The candidates and the comparator

**The pool build PB(excl).** excl is B_run on a pool run, a legacy parent on an
RS task, and empty for the archive.

1. **Fits.** For every benchmark of L with at least two pairs after the caps,
   `f_b = paiec.prior.benchmark_fit(ps_b)`. This is the free item variance at
   the defaults, max_iter 500 and tol 1e-4 (`paiec/prior.py:229`).
   * Each fit is kept as returned, whether or not EM or Newton converged. The
     counts are reported.
   * The fits do not depend on excl. They are computed once and cached.
2. **Subject prior.** Take the rows `paiec.prior.standings` would build
   (`paiec/prior.py:355-367`: centred t and t_var) from the multi-pair fits of
   U ∖ excl. Then
   `prior = paiec.prior.from_standings(rows, 2.0, sorted(L ∖ excl), sorted(excl), None)`.
   No design terms are passed, so the provider threshold stays at min_count 8,
   as shipped.
3. **Hyperparameters.** These follow fit_hyper's arithmetic
   (`paiec/prior.py:756-807`) on these fits, with one change: the share is
   fixed.

   | field | value |
   |---|---|
   | T | the median `item_var` of the multi-pair fits of L ∖ excl |
   | share | 0.25 |
   | sigma_g, sigma_d | sqrt(0.25 T), sqrt(0.75 T) |
   | sigma_attr | sqrt(max(total_var - s2_res, 0)) |
   | th2, sigma_theta | th2 from the identity mode below; sigma_theta = sqrt(th2) |
   | sigma_delta | sqrt(max(s2_res - u_mean - th2, 0.1)) × 1.4 (WIDEN[1]) |
   | slip, guess, text_share, nu_mu, shift | 0.01, 0.5, 0.2, 0, 0 (Hyper's defaults) |
   | excluded, included | sorted(excl), sorted(L ∖ excl) |

   total_var, s2_res, u_mean and tau2_res come from from_standings' meta.
4. **The bundle.** It is `paiec.prior.to_json(prior, hyper)`. It holds the same
   keys as archive-3's bundle and no per-benchmark level.

**C1: the rest, at archive-3's level.**

* PB, with archive-3's identity guards: th2 = clip(tau2_res, 0.01, 0.2) and
  id_cap = 0.1.
* LEVEL written over the fit: mu0 -2.5, sigma_mu 2.5, attr_scale 0.5, as
  `tools/build_submission.py:118-120` does now.

**C2: the pool's level, on the same rest.** C2 is C1 with mu0 and sigma_mu
replaced as follows.

* **The fixed-T levels.** T_full is step 3's T with excl empty. For every
  benchmark b of L, single-pair ones included,
  `g_b = benchmark_fit(ps_b, s2e=T_full)`, computed once and cached.
* **Each level.** For b in L ∖ excl:
  * level_b = g_b.mu + mean(g_b.t) minus the mean, over b's capped subjects s,
    of `prior.attribute(s)[0]` under this build's subject prior. That is the
    level at attribute score 0, placed as fit_hyper places it
    (`paiec/prior.py:774-779`).
  * v_b = g_b.mu_var.
  * Levels with |level_b| > 10 are dropped and counted. None is expected at a
    fixed T.
* **The summary (m, s2).** A precision-weighted moment iteration:
  1. start from s2 = max(var(levels, ddof 1) - mean(v), 1e-6);
  2. at most 500 times: w = 1 / (s2 + v), m = Σ w·level / Σ w, and
     s2 = max(Σ w·((level - m)² - v) / Σ w, 1e-6);
  3. stop once s2 moves by less than 1e-10.
* **The level prior.** mu0 = m and sigma_mu = sqrt(s2) × sqrt(1 + 1/n), where
  n is the number of levels used.
  * WIDEN[0] is not applied. It corrected an ML sd fitted on five levels
    (`paiec/prior.py:44-46`), and n is about 130 here.
  * The flagged benchmarks are included, because they belong to the
    population.
  * attr_scale stays 0.5.
* *Planning:* about N(+0.16, 2.31²) on the item scale at T_pool 5.80
  (code.md §4), with an accuracy-scale centre near +0.1.
* **In the archive** (excl empty), model.py's LEVEL becomes
  `{"mu0": round(m, 3), "sigma_mu": round(sqrt(s2 * (1 + 1/n)), 3),
  "attr_scale": 0.5}`. The evaluation uses the unrounded values.

**C3: the identity guards lifted, the level held.** C3 is C1 with two changes:

* th2 = max(tau2_res, 0.01): no ceiling, and the floor keeps theta proper;
* id_cap = 1.0, which is no cap, since the weight min(V / (V + vid), id_cap) of
  `paiec/hier.py:1910` never exceeds 1.

sigma_delta follows from th2 through step 3. *Planning:* sigma_theta 0.635,
sigma_delta 2.05 and link weight 0.088, against C1's 0.447, 2.14 and 0.042
(code.md §5).

**What each contrast reads** (descriptive):

| contrast | what it shows |
|---|---|
| C1 - A3 | the pool's rest at archive-3's level |
| C2 - C1 | the level |
| C3 - C1 | the identity guards |

C1's rest covers the standings, the ridge, the identity table, T, sigma_delta,
sigma_attr and sigma_theta. C1 keeps archive-3's level centre: *planning*, the
B0 prior centre moves by at most 0.024 on every reference set. On the accuracy
scale, at E's mean attribute score, C1's median is -1.10 against archive-3's
-1.15, and its spread is about 16% wider (code.md §4; eval.md §1).

**Not candidates, and why:**

| option | why not |
|---|---|
| mu0 re-matched to archive-3's B0 centre | within that tolerance already |
| a Student-t or mixture level | `_check_bundle` refuses it, and it costs 5 to 10 times the time per call |
| `source_model_label` read in `canon_name` | a change to the shipped `prior.py` |
| run-time EB of the level | measured before and not shipped |
| C2 and C3 together | a fourth arm |

**Order of preference for ties:** C1, C3, C2, by increasing departure from
archive-3.

**The comparator A3.**

* **On pool runs:** `paiec.hier.HierPredictor(*paiec.prior.load("data/pool_eval/a3_prior.json"))`
  with default flags. This is exactly archive-3's runtime:
  * the in-repo `hier.py` has archive-3's digest;
  * `tools/build_submission.py` `check_run` holds the archive bit-identical to
    the research package on the same prior.json, and RP2 checks it again.

  No legacy benchmark enters a pool run, so A3 needs no refit.
* **On RS tasks:** P1a's stored rows,
  `data/regime_sensitivity_rows/<R>/<i>.json`, key `res.ship.b`. They are
  full-precision floats, written under library digests equal to HEAD's.

*Planning* values (prototype: benchmark-wide item cap, subject-id order; the
`fits` stage re-measures them):

| | A3 | C1 | C2 | C3 |
|---|---|---|---|---|
| T | 9.513 | 5.80 | 5.80 | 5.80 |
| sigma_d, sigma_g | 2.671, 1.542 | 2.085, 1.204 | 2.085, 1.204 | 2.085, 1.204 |
| sigma_theta / id_cap | 0.1 / 0.1 | 0.447 / 0.1 | 0.447 / 0.1 | 0.635 / 1.0 |
| sigma_delta, sigma_attr | 2.382, 1.018 | 2.141, 0.208 | 2.141, 0.208 | 2.05, 0.208 |
| mu0, sigma_mu, attr_scale | -2.5, 2.5, 0.5 | -2.5, 2.5, 0.5 | about +0.16, 2.31, 0.5 | -2.5, 2.5, 0.5 |
| identity rows / linked names | 106 / 21 | about 494 / 167 | as C1 | as C1 |

### 4.5 The evaluation

**Regimes.**

| regime | runs | target: realised level of the evaluated responses (mean, sd ddof 0, over pair appearances) | role |
|---|---|---|---|
| P-READING | formative-size, the half's benchmarks | mean -0.715, sd 1.772 | gain |
| P-AUDIT | formative-size | mean -1.10, sd 2.00 | gain |
| P-TUNED | formative-size | mean -1.290, sd 1.696 | loss bound +0.002 |
| P-MIXTURE | formative-size | share below -3 of 0.118; rest mean -0.226 | loss bound +0.002 |
| P-FLAT | formative-size | none (w_b = 1) | loss bound +0.002 |
| D-READING | dense | as P-READING | loss bound +0.001 |
| D-FLAT | dense | none | loss bound +0.001 |
| RS-TUNED, RS-READING, RS-AUDIT, RS-MIXTURE, RS-FLAT, RS-R1B, RS-R1P | P1a's seed-11 compositions (`data/regime_sensitivity_rows/_compositions.json`) | P1a's (amendment A6 of the P1a plan) | guard (E1) |

The targets are imported from `experiments/regime_sensitivity.py`, whose sha256
the lock pins (`af80ed50bc02a92d`): `TARGETS` (:250-264), `MIX` (:234),
`KNOB_SCALE` (:267) and `REALISE_TOL` (:289).

**Formative-size runs** (`paiec/testlike.py` Sampler, :559-763, adapted to real
benchmarks; run-shape defaults at :507-537):

* A run draws n ~ U{5, ..., 12} pairs, one at a time, from the half's
  benchmarks.
* With probability 0.25, and if a benchmark already in the run has a free pair,
  the next pair joins one of those benchmarks. Otherwise it opens a new one.
  Either way the benchmark is chosen in proportion to w_b, and the pair
  uniformly among the benchmark's free pairs.
* A free pair is one not yet drawn whose subject_id and eight visible fields are
  not yet in the run.
* Each pair keeps a uniform random subset of at most 160 items. Then
  `official.sample_run(cut, rng, cap=1000, n_pairs=len(cut), weighting="pair",
  min_items=80)` makes the proportional cut.
* Subjects keep their real dates, with no date shift.

**Dense runs:**

* 7 benchmarks are drawn without replacement, in proportion to w_b, among the
  half's benchmarks with at least 4 eligible pairs (42 in S, 65 in C).
* From each, min(8, n_b) pairs are drawn uniformly without replacement, with
  distinct eight-field dicts.
* Each pair keeps at most 120 items, and there is no cap.

**Tilt and knobs.**

* **The weight.** w_b ∝ min(t_b, 20 · mean t), where t_b is the target density
  at level_b over a Gaussian kernel density (bandwidth 0.7) of the half's
  levels, each benchmark counted once.
* **P-MIXTURE's target** is a two-component mixture with RS's low component
  (-4.38, 0.171) and high sd 1.151. Its weight and high mean come from the
  grid.
* **The knob search.** It runs on `CAL_SEED`, with 1,000 draws a setting, per
  half, on `pairs.csv`'s all-item pair logits. The grids:
  * Gaussian: m ∈ {-3.5, -3.375, ..., +0.5} × s ∈ {0.6, 0.8, ..., 2.2};
  * mixture: low weight ∈ {0.06, 0.10, ..., 0.30} × high mean
    ∈ {-1.5, -1.375, ..., 0.0}.

  The objective is RS's: Σ ((realised - target) / `KNOB_SCALE`)². Each argmin
  is fixed at the lock.
* **Short of target.** A regime that misses its target by more than 0.15 on the
  mean, or by more than 0.03 on the share, is recorded as "short of target". It
  keeps its place in the rule, as RS kept its milder realisations (P1a plan,
  amendment A6). *Planning:*
  * every C regime reaches its target, within 0.04 on the mean and 0.05 on the
    mixture's rest mean;
  * S's P-MIXTURE reaches a share of 0.088, below the 0.118 target.
* **Realisation check.** The planned scoring runs' realised level must lie
  within `REALISE_TOL` of the CAL_SEED value: mean ±0.30, sd ±0.30, share ±0.04
  and rest mean ±0.25. If the check trips, scoring does not start and the cause
  is found. The knobs are never re-tuned on a scoring seed.

**Protocol.** Every run is played as `paiec/official.py` plays it:

* a persistent 50/50 split at seed 0 with split scope 'pair' (:83-100);
* the platform's sha256 acquisition (:310-319);
* the first recorded response revealed;
* every 0/1 response of an evaluation item scored;
* one shared `labeled` per checkpoint, and a fresh instance per checkpoint and
  arm;
* budgets 0, 1, 3, 7, 15 and 31;
* anonymous subject and benchmark ids, as `official._slots` assigns them
  (`paiec/official.py:348-349`).

The harness is `experiments/level_calibration.py` `checkpoints` and `evaluate`
(:508-561), verified against `official.run_official` (RP3). On pool runs the
loader keeps every 0/1 response of the drawn pairs, in file order, with no cap.

**Seeds.** None of 4001 to 4004 has been used as a seed in the repository.

| seed | use |
|---|---|
| `SPLIT_SALT = "pool-eval-split"` | the split |
| `CAL_SEED = 4001` | knobs; never scored |
| `SEL_SEED = 4002` | S |
| `CONF_SEED = 4003` | C |
| `TIMING_SEED = 4004` | the timing stage, labels stripped |
| RS seed 11 | the stored compositions |
| `BOOT_SEED = 0` | every bootstrap |

Each run uses `np.random.default_rng([seed, half_code, regime_code, i])`.

**Run counts.**

| stage | P- regimes | D- regimes | RS | arms |
|---|---|---|---|---|
| S | 80 each (400) | 30 each (60) | 520 tasks | pool runs: A3, C1, C2, C3; RS tasks: C1, C2, C3 |
| C, core | 120 each (600) | 50 each (100) | | A3 and the carried X* |
| C, extra | the same 600 runs | | | the two other candidates; after the gate is committed; descriptive |

* **Task order.** Tasks run in run-index-major order on 3 shard processes, so
  any stopping point leaves every regime with the same prefix.
* **Resumability.** A worker exits with code 75 once its resident set passes
  1.45 GB, and the stage resumes from there.
* **The one allowed adjustment.** It is made at the timing stage, before the
  lock. If the projected core exceeds 30 process-hours:
  1. the dense item cap drops from 120 to 80;
  2. if the projection is still above 30 process-hours, the dense counts drop
     to 20 (S) and 40 (C).

  Formative-size counts never change.

**Metric and statistics.**

* **Pair ALC** is 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31
  (`paiec/evaluator.py:49`, `paiec/official.py:60-61`). Run ALC is its mean over
  the run's pairs.
* **D(X, R)** is the mean over R's planned runs of run ALC(X) minus run
  ALC(A3), on identical runs and checkpoints. With each appearance weighted
  1 / (pairs in its run), D = Σ_k v_k / Σ_k n_k over clusters k.
* **SE_b** is the cluster bootstrap over benchmarks: ratio estimator, 2,000
  multinomial resamples, seed 0. It is `Boot`
  (`experiments/level_calibration.py:741-768`) with the benchmark as the
  cluster.
* **SE_s and SE_bs** are the same bootstrap with two other clusters: the
  identity key (`paiec/hier.py:1918-1927`: ("n", canon_name), or ("k", the
  digest of the eight fields)), and the (benchmark, key) cell.
* **The SE the rule reads.** SE_2 = sqrt(max(0, SE_b² + SE_s² - SE_bs²)),
  SE_g = max(SE_b, SE_2) and U95 = D + 1.96 SE_g.
* **PB** is the mean, with each benchmark once, of the benchmark's
  appearance-weighted mean difference.
* **JK** is the largest D with all appearances of one benchmark removed.
* **On RS regimes,** D is as P1a defined it.

**Descriptive outputs.** None of these changes anything.

* **For every arm pair and regime:**
  * D with run SE, SE_b, SE_s, SE_bs, SE_2 and the bootstrap's 2.5th and 97.5th
    percentiles;
  * PB, JK and the share of benchmarks with D_b < 0;
  * D by budget.
* **For every arm:**
  * mean ALC;
  * the mean B0 prediction against the mean evaluated rate;
  * ECE;
  * fallback counts;
  * latency.
* **Cuts of D:**
  * the three contrasts above;
  * by subject kind: named with an identity row, named without one, unnamed;
  * without the flagged benchmarks;
  * by response kind;
  * by level tercile;
  * on pairs released on or after 2025-01-01.
* **The runs:**
  * per run, v and ALC - v on the replica's own runs;
  * the realised levels and the effective number of benchmarks.
* **C2's T.** For 50 runs a half, T recomputed without the run's benchmarks,
  and C2's level at that T.
* **A reweighting check.** P-FLAT's appearances reweighted to each tilted
  target by benchmark level, against the tilted regime's own D.

**Checks before any scored task.** Each stage after the lock re-runs the lock
check and refuses to run if it fails.

1. **`split`.** It reproduces `feasibility.json`'s halves.
2. **`regimes`.** The knob search; every planned composition, written to
   `data/pool_eval_rows/<half>/_compositions.json`; the realisation checks.
3. **`fits`.** PB's fits and the fixed-T pass, cached under `data/pool_fits/`.
   It reports convergence and degenerate item variances, and writes
   `results/pool_fits.json`.
4. **RP1.** A3 re-scored on P1a's seed-11 runs 0 and 1 of each of the seven RS
   regimes (14 tasks) equals the stored `ship` rows per pair and budget, to
   1e-9.
5. **RP2.** The archive-3 runtime, loaded from the zip in a fresh interpreter
   as `check_run` loads it, and the in-repo A3 give bit-identical predictions on
   two TIMING_SEED pool runs. Predictions are compared with each other, never
   with labels.
6. **RP3.** On one formative-size and one dense TIMING_SEED run, the harness
   and `official.run_official(..., workers=1)` give identical predictions for A3
   and C1.
7. **RP4.** PB's code reproduces the shipped build, with the share set by
   fit_hyper's rule instead of fixed. It runs on the legacy five's pairs as
   `paiec.data.load_pairs()` gives them, with items and no caps.
   * With excl empty and LEVEL written over, it rebuilds archive-3's
     `prior.json` byte for byte.
   * With excl = (parent,), it equals `paiec.prior.build(pairs, (parent,))`
     for each legacy parent.
8. **RP5.** On ten random exclusion sets, every candidate bundle passes
   `from_json` and `HierPredictor`, with `hyper.excluded` equal to excl and
   `included` disjoint from it.
9. **`timing`.** Six formative-size and two dense TIMING_SEED runs, every arm.
   * Labels are stripped from the targets, and the code asserts that no Brier
     can be computed.
   * It records CPU and wall time per call and per run, the bundle time and
     the resident set.
   * It projects the core's cost and applies the adjustment rule.
10. **`lock`.** sha256 of:
    * `RULE_TEXT`;
    * this amendment's text;
    * the constants block;
    * the shipped modules (section 4.2);
    * `a3_prior.json`;
    * the fits cache index;
    * the planned compositions.

    Settings: `PYTHONHASHSEED=0`, one BLAS thread, and every set or dict that
    orders output iterated sorted. The script, its tests and the pre-lock
    results are committed before the first scored task. That commit is the
    outside time stamp P1a lacked (P1a plan, amendment A2).

### 4.6 Selection, then confirmation

* **One candidate reaches C.** S and the RS guard pick at most one candidate,
  X*. C then tests X* alone against A3, so confirmation has no multiplicity.
  S's winner's curse stays in S.
* **The selection record.** It holds:
  * every condition for every candidate;
  * M(X) with SE_b;
  * S's SE_b per regime, and the power they imply for gains of 0.002, 0.004
    and 0.006;
  * X*;
  * the time stamp.

  It is written to `results/pool_eval.json: selection` and committed before
  the first CONF_SEED task is scored.
* **Why (E1) is a mean plus P1a's per-regime bounds.** A literal "+0.0005 in
  each RS regime" would fail by noise alone. Take a candidate with no true
  effect and a per-regime cluster SE of 0.0015 (RS measured 0.0003 to 0.0042
  for hier variants; P1a plan, amendment A7). Each of its point estimates lies
  below +0.0005 with probability about 0.63, and all seven together with
  probability well under one half. The mean keeps +0.0005 on a quantity that
  noise moves less. P1a's bounds stop any single regime from losing materially.
* **The RS guard leans towards A3.** The same catalogue chose and guarded LEVEL
  (P1a plan, amendment A5). That suits a no-loss bound.
* *Planning* power, in one regime of C, for a true gain Δ:

  | sd across benchmarks of the per-benchmark mean difference | SE_b in P-READING | pass at Δ = 0.002 | at 0.004 | at 0.006 | false pass at Δ = 0 |
  |---|---|---|---|---|---|
  | 0.005 | 0.0006 | 0.50 | about 1 | about 1 | < 0.001 |
  | 0.010 | 0.0012 | 0.39 | 0.92 | about 1 | 0.025 |
  | 0.020 | 0.0025 | 0.12 | 0.36 | 0.67 | 0.025 |

  Both feedback regimes must pass, and they share C's benchmarks. Under no true
  difference they pass together with probability about 0.01. A true gain under
  about 0.003 will often be missed, and the rule errs towards archive-3, which
  is the fallback anyway.

### 4.7 The decision rule (verbatim `RULE_TEXT`)

```
POOL-EVAL DECISION RULE: does a pool-fitted hier replace archive-3?
(amendment 4 to docs/plans/p2_final_plan.md; experiments/pool_eval.py).
Fixed before any task of SEL_SEED, CONF_SEED or the RS guard was scored.

Terms
  A3        archive-3: paiec.hier.HierPredictor over archive-3's prior.json
            (sha256 c7ce3b845d584f95...; LEVEL mu0 -2.5, sigma_mu 2.5,
            attr_scale 0.5; fitted on the five legacy benchmarks). On RS
            tasks: P1a's stored `ship` rows.
  C1,C2,C3  the candidates of CANDIDATES (amendment 4, section 4.4): the
            pool build with LEVEL kept; with the pool's empirical-Bayes
            level; with the identity guards lifted. On a pool run each is
            built without every benchmark of the run, on an RS task without
            the target's parent.
  E, S, C   the 132 evaluation benchmarks and their selection (50) and
            confirmation (82) halves (SPLIT).
  P-READING, P-AUDIT, P-TUNED, P-MIXTURE, P-FLAT
            formative-size pool regimes; D-READING, D-FLAT dense pool
            regimes; RS-TUNED, RS-READING, RS-AUDIT, RS-MIXTURE, RS-FLAT,
            RS-R1B, RS-R1P P1a's regimes on its scoring seed 11 (520 tasks).
  D(X,R)    the mean over R's planned runs of the run's pair-mean ALC of X
            minus that of A3, on identical runs and checkpoints.
  SEg(X,R)  max(SEb, SE2): SEb the cluster-bootstrap SE over benchmarks
            (ratio estimator, 2,000 resamples, seed 0); SE2 the two-way SE
            sqrt(max(0, SEb^2 + SEs^2 - SEbs^2)) from the same bootstrap
            with the subject identity key and the (benchmark, key) cell as
            clusters.
  U95(X,R)  D(X,R) + 1.96 SEg(X,R).
  PB(X,R)   the mean over benchmarks, each once, of the benchmark's
            appearance-weighted mean difference.
  JK(X,R)   the largest D(X,R) with all appearances of one benchmark
            removed.

Selection (S and RS only; recorded and committed before any CONF_SEED task
is scored). X in {C1, C2, C3} is eligible iff
  (E1) the mean of D(X,RS-R) over the seven RS regimes is <= +0.0005, and
       D(X,RS-R) <= +0.002 in each RS test-like regime and <= +0.001 in
       RS-R1B and RS-R1P;
  (E2) on S: D <= +0.002 in P-TUNED, P-MIXTURE and P-FLAT, and D <= +0.001
       in D-FLAT and D-READING;
  (E3) on the S pool tasks X's mean CPU time per evaluation call is at most
       2.0 times A3's in the same tasks, and on the S and RS tasks no
       formative-size call of X takes more than 2.0 s of CPU;
  (E4) every planned S and RS task is scored for every listed arm and every
       check passed.
  Among the eligible, X* minimises M(X) = [D_S(X,P-READING) +
  D_S(X,P-AUDIT)] / 2; an eligible candidate within 0.0005 of the minimum
  that comes earlier in the order C1, C3, C2 is taken instead. X* is
  carried iff M(X*) <= -0.001; otherwise the outcome is "no candidate
  carried: archive-3 stays".

Gate (C only; read once, on the full planned set). X* replaces A3 iff
  (a) in P-READING and in P-AUDIT: D <= -0.002 and U95 < 0;
  (b) in P-READING and in P-AUDIT: PB <= 0 and JK <= -0.001;
  (c) in P-TUNED, P-MIXTURE and P-FLAT: D <= +0.002;
  (d) in D-READING and D-FLAT: D <= +0.001;
  (e) (E1) holds for X*;
  (f) on the C tasks: X*'s mean CPU time per evaluation call is at most 2.0
      times A3's; no formative-size call takes more than 2.0 s of CPU;
      dense calls have a p99 of at most 1.0 s and a maximum of at most 30 s
      of CPU;
  (g) every check passed and every planned C task is scored for A3 and X*.

Reading
  1. Pass: X* is built as archive-4 (the pool build on every benchmark of
     the training rule, nothing excluded; for C2, model.py's LEVEL becomes
     the full-pool EB level rounded to 3 decimals). It becomes the final
     only if tools/build_submission.py passes every check and the
     organisers' validator, S3's gate (P2 plan section 3) holds on
     archive-4, the one regression submission of P2 plan section 4 is
     uploaded no later than 2026-10-16 and every pair of it is scored with
     finite outputs, and the final choice is recorded by 2026-10-20 23:59
     UTC. Otherwise archive-3 stays.
  2. Fail, or no candidate carried: "archive-3 stays", recorded as a
     negative result with every D, SE and condition, for every candidate
     scored.
  3. Everything else (the decomposition, budgets, subject kinds, flags,
     tercile and recency cuts, v decomposition, latency tables, the C-extra
     arms, the post-gate placement of the hidden runs) is descriptive and
     changes nothing.
  4. Nothing in this rule, CANDIDATES, SPLIT, the regimes and their knob
     grids, the seeds, the run counts or the statistics changes once a row
     of SEL_SEED, CONF_SEED or the RS guard exists. No run is added or
     dropped after rows are read; each stage's rule is applied once, to its
     full planned set. If a check fails, the cause is fixed, every affected
     task is re-scored in full and the deviation is recorded; the rule is
     not rewritten. If the gate has not been read and committed by
     2026-10-13 23:59 UTC, the outcome is "not read in time: archive-3
     stays".
  5. Formative feedback enters only as the RS study's global level targets
     and testlike's run shape. results/hidden_runs.json is read once, after
     the gate is committed, for the descriptive placement of amendment 4,
     section 4.9. Formative run 3 sets nothing.
```

### 4.8 After the gate

**Pass.**

1. **Build archive-4** with `python tools/build_submission.py --pool
   --candidate X*`. That is PB with excl empty, plus the LEVEL literal for C2.
   * It runs every existing check: the bundle checks, the import checks, the
     bit-identity `check_run` on the legacy session, and the organisers'
     validator.
   * `--pool` adds four checks:
     * `included` equals L in `tools/pool_benchmarks.json`;
     * id_cap and sigma_theta are X*'s;
     * the bundle's keys equal archive-3's, with no per-benchmark level;
     * the archive differs from archive-3 only in `prior.json`, plus model.py's
       LEVEL line for C2.
   * The archive is copied to `dist/archive-4.zip` and its sha256 recorded, by
     2026-10-15.
2. **Run S3 on archive-4:** `python tools/stress_archive.py --archive
   dist/archive-4.zip --no-rebuild --out results/stress_archive4.json`.
   * S3's gate (P2 §3) is read pooled, as amendment 1 read it, with the
     per-part figures beside it.
   * No fix is made. A failure means archive-3 stays.
3. **Amendment 5,** committed before the upload, announces the one regression
   submission of P2 §4, with archive-4's sha256.
4. **The upload** is made no later than 2026-10-16. P2 §4's date is
   unchanged.
5. **The rule for the final.** archive-4 becomes the final if every pair of
   that run is scored with finite outputs, whatever its score (P2 §4).
   * Its table is recorded as run 4, by an `experiments/formative_run4.py`
     modelled on `formative_run3.py`.
   * It is used only through aggregate, id-free quantities such as ALC - v.
   * It selects and tunes nothing, and it is compared with nothing to choose
     the final.
6. **The final choice** is recorded in a dated amendment by 2026-10-20 23:59
   UTC. If run 4's table has not arrived by then, archive-3 stays the final.
   For this path only, this replaces P2 §3's "the final archive is chosen
   on 2026-10-17".

**Fail, or no candidate carried.**

* archive-3 stays. Nothing is built and nothing is submitted.
* The negative result is recorded with every D, SE and condition.
* C extra and `describe-hidden` still run, as descriptive outputs.

**Not read in time.** If the gate is not committed by 2026-10-13 23:59 UTC,
archive-3 stays. Whatever was scored is reported as incomplete and descriptive.

### 4.9 Formative feedback (App G.6)

**Used before and during the decision.** Each of these is a use App G.6
already lists, reused as a global quantity:

1. **RS `TARGETS`.** READING, AUDIT and MIXTURE come from the pooled reading of
   runs 1 and 2 and from run 1's audit. TUNED is the realisation of the regime
   that run 1's lower-root reading set. They are quoted from
   `experiments/regime_sensitivity.py` and not recomputed.
2. **testlike's run shape** (`paiec/testlike.py:507-537`): n_pairs 5 to 12,
   repeat 0.25, at most 160 items a pair and the 1,000 cap.

**Used after the gate is committed, descriptive only.** These are new uses,
which App G.6 must list:

3. **`describe-hidden`.** It reads `results/hidden_runs.json` once and takes
   `decomposition.run{1,2,3}.v`, `.alc_minus_v` and `.pairs`: run-level, id-free
   aggregates. It places each run's v in each P- regime's distribution of run v
   over C runs of 8 or 9 pairs, as a z-score and a percentile. It places runs 2
   and 3's ALC - v in A3's replica distribution of ALC - v, with X*'s replica
   ALC - v beside it.
4. **If archive-4 ships,** run 4's table, as P2 §4 allows.

**Not used:**

* any per-pair value;
* any anonymous id;
* any formative table before the gate is committed;
* run 3 for anything but item 3;
* S4's v-bin weights.

App G.6 gains a paragraph listing items 1 to 4. In the commit, the author
confirms that nothing else from the feedback was read for this evaluation.

### 4.10 Compute budget

**Setting.** The machine is an Apple M1 Pro with 8 cores and 16 GB, shared
with other jobs. The scoring runs on 3 shard processes, with BLAS pinned to one
thread and at most 1.45 GB each.

*Planning* unit costs:

* a hier arm costs 6.0 s on a formative-size run and 80 s on a dense run
  (uncertain by a factor of 2), and 5.3 s on an RS task;
* the bundles of one run cost 3 s;
* the fits, with the fixed-T pass, cost 900 CPU s.

| stage | process-seconds |
|---|---|
| fits | 900 |
| regimes (knob search, compositions) | 1,800 |
| reproduce | 300 |
| timing | 1,000 |
| S, P-: 400 runs × (4 arms × 6.0 + 3) | 10,800 |
| S, D-: 60 runs × (4 × 80 + 3) | 19,400 |
| RS guard: 520 tasks × 3 arms × 5.3 | 8,300 |
| C core, P-: 600 × (2 × 6.0 + 3) | 9,000 |
| C core, D-: 100 × (2 × 80 + 3) | 16,300 |
| summarise | 900 |
| **core total** | **about 68,700 (19.1 process-hours)** |
| C extra, after the gate: 600 × 2 × 6.0 | 7,200 |

**Wall time on 3 shards, at 1.5 times the planned unit costs beside the other
jobs:**

* about 5.9 hours to the selection record;
* about 3.6 hours for the C core;
* about 1.0 hour for C extra.

**Cost risks and resources.**

* **The dense runs carry 52% of the core.** If they run twice as slow as
  planned, the projection reaches about 28.8 process-hours, just under the
  adjustment's 30-hour trigger.
* **Memory:** at most 4.4 GB in all.
* **Disk:** about 0.2 GB for rows, the fits cache and the bundle cache.

### 4.11 Schedule and hard stops

| date (2026, UTC) | work |
|---|---|
| Wed 10-07 | the design notes; this amendment drafted |
| Thu 10-08 | this amendment committed; `paiec/pool.py` and `tests/test_pool.py`; `fits` |
| Fri 10-09 | `experiments/pool_eval.py` and `tests/test_pool_eval.py`; `split`, `regimes`, `reproduce`, `timing`; the lock commit |
| Sat 10-10 | S and RS scoring; `select`; the selection record committed |
| Sun 10-11 | C core; `summarise`; the gate read once and committed; C extra; `describe-hidden` |
| Mon 10-12 to Tue 10-13 | slack. The `--pool` build path and its tests are written meanwhile, since they do not depend on the outcome |
| Wed 10-14 to Thu 10-15 | on a pass: archive-4 built and checked; S3 on archive-4; amendment 5 |
| Fri 10-16 | on a pass: the regression upload |
| by Tue 10-20 | run 4's table read for finiteness; the final recorded |

**Hard stops.**

* The gate must be committed by 2026-10-13 23:59 UTC.
* archive-4 must be built, have passed S3, and be announced by 2026-10-15
  23:59 UTC.
* It must be uploaded by 2026-10-16 23:59 UTC.
* The final must be recorded by 2026-10-20 23:59 UTC.

Missing any of them leaves archive-3 the final.

**Alongside.** The report work of P2 §7 continues, and this evaluation's
outcome enters the findings with its script either way. P2 §5 gains R7, the
expanded pool, ranked after R2.

### 4.12 What the report will say

**In every case.** The main text gets one paragraph and one table; the details
go to an appendix. They state:

* **The pool.** Its growth on 2026-10-06 and 2026-10-07, the revision, the
  random assignment of benchmarks to pools, and what that implies.
* **The process.** The unfreezing and its scope: priors only, no runtime
  change. The rule was fixed and committed before any comparison (the commit
  id). The S/C split, the tilted and untilted regimes, the items-blind
  limitation and any deviation.
* **The confirmation table.** D, SE_g, U95, PB and JK per regime for X*. If no
  candidate was carried, the same for every candidate on S.
* **The decomposition** into the rest, the level and the identity term,
  descriptive.
* **The data facts.** The pool's level distribution against LEVEL, and the link
  coverage.
* **The App G.6 additions** of section 4.9, and the limits of section 4.13.

**Pass.**

* **Methods.** The submitted predictor is hier with its priors refitted on the
  public pool. The subject prior comes from 128 benchmarks, and the item
  variance and level from 137. Under C1 or C3 the level prior stays at LEVEL;
  under C2 it is the pool's empirical-Bayes level.
* **Results.** On the 82 confirmation benchmarks, which played no part in
  selection, X* beat archive-3 by the stated D in the regimes of the formative
  readings, and lost no more than the stated bounds elsewhere.
* **The hidden runs.**
  * Runs 1 to 3 scored earlier configurations.
  * Run 4 was a regression check: it is read for finite outputs, and its
    ALC - v is reported beside run 3's as an aggregate. Nothing was chosen
    from it.
* **The test-like headline** (-0.042 against LegacyP) stays a replica result
  about archive-3's configuration. The pool evaluation is reported as a second
  replica result, on real held-out benchmarks.

**Fail.**

* **The final.** The submitted predictor stays archive-3. Refitting hier's
  priors on the expanded pool did not pass the gate fixed in advance.
* **The record.** The conditions that failed, with their numbers, or "no
  candidate reached M ≤ -0.001 on the selection half".
* **If C2 looks good in P-FLAT,** the text says that held-out public benchmarks
  favour a pool level by construction, and that the rule asked for gains where
  the readings sit.
* **What the pool shows** about the rest (the identity slopes, the flat pool
  ridge) is reported descriptively, with no claim of gain.

**Not read in time.** One sentence saying so, and the incomplete numbers marked
descriptive.

### 4.13 What this cannot settle, and disclosures

**What this cannot settle.**

* **The readings' own uncertainty.** The tilted regimes bracket the readings'
  centre, not their uncertainty. They also match level only: hidden benchmarks
  may differ in kind in ways no tilt reproduces.
* **Whether the summative draws the same 7 benchmarks.** If it draws afresh,
  P-FLAT is the relevant regime, and there the rule only bounds losses.
* **Items.** In Mode B the floor and the group effects are untested on pool
  runs. The RS guard tests them on the legacy items.
* **Subjects.**
  * The hidden subjects' names and dates are unknown. Pool runs carry the
    pool's own mix: 58 to 63% of pairs named per benchmark, mostly 2024 dates.
  * S and C share subjects (gpt 4o is on 52 benchmarks). SE_2 sees this within
    C, but not across the halves.
* **One split.** The design does not resample the split, because that would
  reopen selection.
* **T_full.** C2's fixed-T levels use T_full, which saw each held-out
  benchmark's item variance through a median of about 124. The run-level T is
  recorded beside it.
* **Protocol unknowns:**
  * the split scope (only 'pair' is scored);
  * which response is revealed (the first);
  * repeats and multi-condition cells (wmt_mqm), scored as repeats;
  * the dense shape, which is a guess.
* **Power.** A true gain under about 0.003 will often be missed.

**Disclosures.**

1. **Choices made on the whole pool.** The build choices were made from
   descriptive statistics of the whole pool, confirmation benchmarks included.
   They are the rules, the caps, Mode B and the three candidates. The
   statistics were the level distribution, the link coverage and code.md §5's
   identity slopes. Those slopes regress a name's residual standing on one
   benchmark on its mean residual elsewhere, and they bear directly on C3. They
   used C's benchmarks' standings. No predictor's ALC or Brier was computed on
   any pool data.
2. **The split was seen early.** The split and the regimes' feasibility were
   computed before this amendment (`eval_design/feasibility.py`), from levels
   only.
3. **Where the design notes are.** The notes and their scripts sit under
   `data/p2_work/pool_design/`. The scripts and notes are committed at the lock
   under `docs/plans/pool_design/`. Outputs with per-pair rows derived from the
   gated data are not committed. No note or output holds item text.
