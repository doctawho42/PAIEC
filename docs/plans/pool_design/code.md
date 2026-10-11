# Fitting hier's priors on the expanded pool: the code path

Repository HEAD a725df9, read only. Pool: measurement-db revision `54074453`
under `data/pool/` (`data.md` describes it; this note reuses its facts and does
not redo them). Written 2026-10-07 for amendment 4.

What this note is and is not. It says what has to change, file by file, to fit
hier's priors on the pool, measures what those fits cost and how large the
result is, and describes the fitted quantities. It compares no predictors and
computes no ALC or Brier on any data. It reads no formative feedback file, no
anonymous id and no item text (`items.parquet` was not fetched). Every number
below comes from a script in `code/`, run with `PYTHONDONTWRITEBYTECODE=1` (no
bytecode was written into the repository; `git status` is unchanged):

| script | output | what |
|---|---|---|
| `code/code_timing.py` | `code/code_timing.jsonl` | CPU per EM iteration of `benchmark_fit` at several pair and item caps |
| `code/code_levels.py` | `code/code_levels.json` | levels on the run-time scale, attribute scores, identity weights and slopes, a benchmark jackknife |
| `code/code_variants.py` | `code/code_variants.json` | the exact fixed-item-variance level pass, with and without the flagged benchmarks; B0 prior centres |
| `code/code_harness_slopes.py` | `code/code_harness_slopes.txt` | the identity slopes across different evaluation pipelines |
| `code/code_archive.py` | `code/code_archive.json`, `code/c1_proto_prior.json` | a C1-shaped prior.json: size, worker load time, the organisers' validator |

All of them start from the interrupted attempt's prototype
(`proto_loader.py`, `proto_build.py`, `fits_K100_M2000.pkl`,
`build_K100_M2000.json`). I read those files and reuse them. They are sound
for what they did (section 2 lists the three changes the loader needs). The
prototype covers the 128 binary benchmarks, at most 100 pairs (K) and 2,000
items (M) each. Its item cap removed sib200, so 127 benchmarks were fitted: 113
with two or more pairs and 14 with one. It applies no direction filter.
"Prototype" below means that set.

The machine is an Apple M1 Pro with 8 cores and 16 GB. Other jobs held its load
average between 9 and 95 while these scripts ran, so costs are given in CPU
seconds.

## 0 In brief

1. **Loader.** `paiec/data.py` cannot read the pool. It reads a hard-coded
   benchmark list, `response.parquet` rather than `responses.parquet`, and
   needs `items.parquet`. A new offline loader is needed. The pool's columns
   are the old ones (data.md §6), so it is short. `proto_loader.py` is a sound
   base. It needs three changes: a per-pair item cap (the per-benchmark cap
   drops sib200), a pair order that keeps links, and an explicit selection
   rule.
2. **No shipped module has to change for C1 or C3.** The build can run the
   per-benchmark fits in worker processes and feed them to the existing
   `from_standings` and to fit_hyper's arithmetic. The archive then differs
   from archive-3 in `prior.json` alone. C2 also changes the `LEVEL` literal in
   `submission/model.py`. The offline driver, the build tool and the tests do
   change.
3. **The caps are not optional.** Uncapped, one benchmark (llm_uncertainty_bench,
   100 pairs, 149,793 items) took 111 CPU s per EM iteration and 8.2 GB. At
   K = 100 and M = 2,000 the 127 prototype fits took 565 CPU s, with at most
   1.5 GB per worker.
4. **`LEVEL` written over a pool refit is not the same level prior.** It is
   relative to the bundle's item variance T and attribute centring, and both
   change (T from 9.51 to 5.80; the mean attribute score of the pool's subjects
   from -0.91 to -0.01). The two shifts nearly cancel in the B0 prior centre:
   it moves by between -0.024 and +0.012 across the reference subject sets
   checked.
5. **A level estimated from the pool needs a new estimator.** fit_hyper's free
   item-variance levels give sigma_mu 9.18, because five fits on 2 to 5
   subjects have item variances of 101 to 1,835. An exact pass at the
   run-time T gives levels at attribute score 0 of N(+0.16, 2.30²) after
   empirical Bayes. On the same scale LEVEL is about N(-2.09, 2.10²), and its
   median sits at the 15.7th percentile of the pool's levels.
6. **The identity guard binds everywhere on the pool.** tau2_res is 0.40
   (benchmark-jackknife SE 0.12), above the clip ceiling of 0.2. Every
   prototype identity weight exceeds id_cap = 0.1. A name's residual on other
   benchmarks predicts its residual on a held-out one with slope 0.70 to 0.90.
   The slope stays there when the other benchmarks must use a different
   evaluation pipeline.
7. **The archive has room.** A pool prior.json is 84 KB at 494 table rows
   (about 100 to 110 KB for all 688 names), against 19 KB now. A worker loads
   it in 3 ms more, and the organisers' validator accepts the archive.

## 1 What the shipped build does now

* `tools/build_submission.py:442` loads the five legacy benchmarks with
  `paiec.data.load_pairs()`. `write_bundle` (`:113-126`) filters them with
  `official.eligible` and fits `PR.build(pairs, ())` (`:118`). It writes
  model.py's `LEVEL` over mu0, sigma_mu and attr_scale (`:120`), saves the
  bundle and checks that it reads back (`:121-123`).
* `shipped_level` (`:98-110`) parses `LEVEL` from model.py's source by AST,
  so the build and the fallback cannot disagree.
* `_check_bundle` (`:155-179`) requires five things: an attribute prior
  (`:167`), the level equal to `LEVEL` (`:169-172`), a Gaussian level
  (`:173`), nothing excluded, and `included` equal to `PUBLIC` (`:175-176`).
* `check_run` (`:387-416`) predicts a formative-sized session of the legacy
  pairs in fresh interpreters. It requires the archive to agree bit for bit with
  the research package on the same prior.json, and requires that no heavy
  package loads. `run_validator` (`:419-428`) runs the organisers' validator.
* `PR.build` (`paiec/prior.py:810-816`) is `build_prior` and then `fit_hyper`.
  `build_prior` (`:542-554`) is `standings` (`:355-367`, one row per pair of a
  benchmark with two or more pairs, centred within the benchmark) and then
  `from_standings` (`:557-667`). That function fits the ridge (`:613`), runs
  both identity passes (`:617-618`), and builds the cells and the table
  (`:627-657`). `fit_hyper` (`:672-807`) calls `benchmark_fit` (`:229-341`)
  for every included benchmark, through the in-process cache `_FITS` (`:226`,
  `:261-263`).
* At run time hier gives each benchmark's level the prior N(mu0, sigma_mu²)
  (`paiec/hier.py:1405`). Theta's mean is shift plus attr_scale times the
  attribute score (`:1903`), plus the identity term, whose weight is capped at
  id_cap (`:1910`). The item residual variance is sigma_d² + sigma_g² where no
  key groups the items (`:1955-1960`).

## 2 Loader

**Why `paiec/data.py` cannot be reused.**

* The benchmark list is hard-coded (`:18-19`).
* It reads `response.parquet` and requires `items.parquet` (`:33-35`). The
  pool has `formatted_tables/responses.parquet`, and items were not fetched
  (data.md §6).
* It sets interactors to `""` (`:41`).
* It keeps file order and 0/1 responses (`:37`, `:46-56`), which is right.

`official.eligible` (`paiec/official.py:66-80`) still applies afterwards: it
drops non-0/1 responses and pairs under 80 items. Its `FRACTIONAL` exclusion
becomes part of the type rule below.

**Columns.** Measured against the tables this repository used (data.md §6),
the pool's columns have the same names, order and types:

* responses: subject_id, item_id, response_id, benchmark_id, trial,
  test_condition, interactors, response;
* subjects: subject_id, display_name and the eight visible fields.

The loader should still name the columns it needs and fail if one is missing.
`pyarrow.parquet.read_table(columns=...)` does that, as `proto_loader.py:37-39`
uses it. It must take only `paiec.data.SUBJECT_FIELDS` (`paiec/data.py:20-21`):
`display_name` is not visible at predict time.

**What `proto_loader.py` does.** Each check is against data.md §2.

* It takes the first 0/1 response of each (subject_id, item_id) in file order
  (`:48-51`). This is the replica's rule: acquisition reveals `by[k][0]`, and
  `benchmark_fit` fits `rs[0]` (`paiec/prior.py:250-260`).
* It drops nulls, applies min_items after the caps (`:57-59`), and takes the
  eight fields plus `_sid` (`:63-73`).
* Items carry empty content and features. Interactors come from the first
  response row of each item (`:76-80`).
* Caps are deterministic: pairs by `stable_hash('pool-pair', b, subject_id)`,
  items by `stable_hash('pool-item', b, item_id)` per benchmark (`:52-62`).

**Changes it needs.**

1. **A per-pair item cap in one benchmark-wide order.** Rank the benchmark's
   items once by the same hash, and let each pair keep its first M_p items in
   that order. On a dense benchmark this equals a benchmark cap of M_p. On a
   sparse one it keeps every pair. sib200's pairs hold 204 of 83,640 items
   each, so the 2,000-item cap left about 5 per pair and dropped the benchmark
   (data.md §7). Uncapped at K = 100, sib200 costs 0.21 CPU s per EM iteration
   (`code_timing.jsonl`).
2. **A pair order that keeps links.** Ranking pairs by a hash of the canonical
   name, in one order for every benchmark, keeps 212 linked names at K = 100;
   the subject-id order keeps 176 (data.md §7).
3. **An explicit selection rule from the catalogue.**
   * Take `response_type` and `response_scale.direction` from
     `data/pool/manifest.json`, which the prototype read from `survey.json`.
   * Pin the revision, `54074453da7b9911be0f48e4c560be547bc84d21`.
   * The rule is a parameter. It has a subject part and a level part, because
     data.md §7 argues they differ (the nine direction-flagged benchmarks
     belong in the level part but not in the standings).

**Interactors are a choice that moves a hyperparameter.** With interactors
read from the response rows, four prototype benchmarks had group keys, and
their median share was capped at G_CAP = 0.25 (`paiec/prior.py:49, 761-762`).
With interactors blanked, as `paiec/data.py:41` does, no benchmark has a key.
The share then falls back to REFERENCE's 1/(1 + 2.5²) = 0.138
(`paiec/prior.py:762-763`), and the split of T into sigma_d and sigma_g moves.
Archive-3's realised share is 0.25 exactly: 1.542² over 2.671² + 1.542². So
fixing the share at 0.25 when items are absent keeps that split. Estimating it
needs `items.parquet` (`tools/fetch_pool.py --items`), which writes item text
under `data/`. That is the author's decision, outside this workflow.

**Mixed benchmarks.** A mixed benchmark's binary items are marked only in
`items.grading_criterion`, which was not fetched (data.md §2). Without it the
loader can apply the 0/1-share rule of data.md §7 and nothing finer.

## 3 `paiec/prior.py` on the pool: what holds and what breaks

### 3.1 Per-benchmark fits (`benchmark_fit`, `:229-341`)

* **The model and code are unchanged and run on the pool.** Each benchmark is
  fitted once, independently of any exclusion set, which is what makes
  leave-one-benchmark-out cheap (module docstring, `:15-16`).
* **Convergence.** In the prototype, 9 fits did not converge in EM (500
  iterations, `:229`, `:309`) and 13 did not converge in Newton
  (`build_K100_M2000.json`). The pass at a fixed item variance (3.2) failed to
  converge on one benchmark, tau2_bench.
* **Degenerate item variances.** The free item variance has median 5.80 over
  the 113 multi-pair fits (quartiles 3.02 and 11.27). Five exceed 100:
  revolve 101, agentic_review_perturbation 112, disco 1,027, workarena 1,814
  and finegrain_t2i 1,835. Each has 2 to 5 pairs, on items whose labels the
  few subjects share. The fitted level follows the variance (workarena
  mu = -47.6; `fits_K100_M2000.pkl`). The median T is robust to them. The
  levels are not (3.2).
* **Cost.** Section 6.

### 3.2 fit_hyper's pooling (`:672-807`)

**What has to change in the code, and what does not.** `fit_hyper` takes
pairs and refits through the in-process cache, so a build in one process would
fit all benchmarks serially. Two ways out:

1. **An offline copy of its arithmetic on fit dicts.** The arithmetic is
   `:756-807`, and `proto_build.py:129-150` is a faithful copy for the pool
   case: the share fallback, the single-pair fits at T, levels at attribute
   score 0, the clip, sigma_delta and sigma_attr.
2. **A refactor of `fit_hyper` into a core on fits.** This changes the bytes of
   the shipped `prior.py` but not run-time behaviour, which `check_run`'s
   bit-identity check guards.

Option 1 keeps the archive delta at prior.json. Option 2 avoids duplicated
code.

| field | archive-3 (public five) | prototype pool (K100/M2000) | source |
|---|---|---|---|
| T = sigma_d² + sigma_g² | 9.513 | 5.797 (median of 113) | `:760` |
| group share | 0.25 (at G_CAP) | 0.25 (median of 4 keyed fits, capped) | `:761-765` |
| sigma_d, sigma_g | 2.671, 1.542 | 2.085, 1.204 | `:764` |
| sigma_theta | 0.1 (tau2_res -0.002, clipped to 0.01) | 0.447 (tau2_res 0.403, clipped to 0.2) | `:792-795` |
| sigma_delta | 2.382 | 2.141 | `:797` |
| sigma_attr | 1.018 | 0.208 | `:798` |
| link weight | 0.0018 | 0.042 | `hier.py:362-367` |
| fitted mu0, sigma_mu (before LEVEL) | -1.263, 2.676 | -0.389, 9.185 | `:787-789` |

**The level part is the one that breaks.**

* **On free item variances it is meaningless.** fit_hyper takes each level
  from the free-item-variance fit (`:779`). On the pool that gives a level
  distribution with sd 6.35, skew -5.0 and excess kurtosis 34. Its range runs
  from -48.4 to +18.7, against a median of +0.33 (`code_levels.json:
  levels.free_T_as_fit_hyper`).
* **The level has to sit on the run-time T.** At run time hier puts every
  benchmark at one item variance T. A level is therefore comparable across
  benchmarks only at that T. The exact way is a second `benchmark_fit(ps,
  s2e=T)` per benchmark; the variance argument exists (`:229`, `:266-268`).
  It cost 138 CPU s for the 127 prototype benchmarks (`code_variants.json`).
* **A cheap map tracks it.** Map each free level through the logistic-normal
  map logit E[sigmoid(mu + sqrt(T_b) z)], then back at T. On 25 benchmarks,
  including the five degenerate ones, this gives levels with correlation 0.992
  to the exact pass. The mean absolute difference is 0.21 and the largest 0.93,
  on finegrain_t2i (`code_levels.json: mapped_minus_exact`).
* **WIDEN was set for five levels.** sigma_mu = sd × sqrt(1 + 1/n) × WIDEN[0]
  (`:789`, `WIDEN` at `:46`) widens the ML sd by 1.44. That correction was
  argued for an ML sd fitted on five levels (`:44-45`, `hier.py` Hyper
  docstring). At n = 127 the sd's own SE is about 0.15. Whether to widen is a
  choice to fix in advance (section 8, C2a).

### 3.3 Standings, ridge and identity table (`:355-667`)

**What carries over unchanged.** `standings`, `_ridge`, `_residuals`,
`_offsets`, `_components` and `from_standings` run unchanged on the
prototype's 2,912 rows, in 2.3 CPU s (`code_levels.json`). Their weighting
(pair, cell and name-pair) is data.md §7's table.

**What the refit produces.** (`code_variants.json: subject_part`,
`code_levels.json: attribute_scores`.)

* **Rows and design.** 2,912 rows against archive-3's 220. 33 coefficients and
  24 providers, against 17 and 7.
* **Coefficients.** The intercept goes from -2.34 to -0.47, the date
  coefficient (per 400 days) from 2.41 to 0.29, and the harness flag from 0.34
  to 0.015. defog's provider coefficient is 3.98, which bird_sql's rows set
  under pair weighting.
* **Variances.** total_var is 2.61 and s2_res 2.56, so the attributes explain
  0.04 of the standings' variance leave-one-benchmark-out (sigma_attr 0.21;
  archive-3 1.02). Without the nine flagged benchmarks: 2,624 rows, total_var
  2.34, s2_res 2.31, sigma_attr 0.19, tau2_res 0.42, sigma_delta 2.02.
* **Attribute scores.** Over the prototype's rows, the score has sd 0.41 under
  the pool ridge and 1.83 under archive-3's. The two correlate at 0.50.
  Weighting benchmarks equally, the mean score is -0.007 under the pool ridge
  and -0.909 under archive-3's. On the legacy subjects the pool ridge gives
  +0.06 to +0.27 and archive-3's -0.53 to +0.35.
* **Identity table.** 494 rows, 167 of them linked. By the number of residual
  benchmarks behind a row: 1 for 327 rows, 2 for 79, 3 for 29, 4 for 14 and 5
  or more for 45. The residual means have sd 1.98: 2.25 for rows on one
  benchmark, 1.29 on two or more, 0.91 on five or more. Archive-3 has 106 rows
  and 21 linked names. Uncapped, the pool has 583 names on the ordered set and
  688 on all 166 benchmarks (data.md §4, §7).

**Choices that need no code change.**

* **Provider threshold.** `_ridge` passes `design` to `Spec.from_frame`
  (`:379`), whose `min_count=8` (`paiec/subjects.py:185, 193-196`) is a
  keyword. `build_prior(..., design={"min_count": m})` therefore raises the
  provider threshold. It is recorded in `meta['design']` (`:665-666`).
* **Rule.** A subject-part rule different from the level-part rule is the same
  rows filtered before `from_standings`. `included` stays the full list, which
  keeps `HierPredictor`'s `included` check (`paiec/hier.py:1847-1861`)
  consistent.

**Choices that would change `prior.py`.**

* Row weights of min(1, K/n_b) in `_ridge` and the cell means (data.md §7).
* A cap on the item variance inside `benchmark_fit`.

Both functions are offline, but the file ships whole. The pair cap (section 2)
gets the weighting without such a change.

## 4 LEVEL: how it is written, what "kept" means, and the pool's level distribution

**Mechanism.**

* `LEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}`
  (`submission/model.py:68`) is read by the build (section 1) and written over
  the fitted bundle.
* model.py uses it alone, on `Hyper()`'s public-five widths, when prior.json is
  unusable (`:89`).
* It is a level at attribute score 0 on the bundle's own item scale: the run
  adds attr_scale × m_s to theta (`paiec/hier.py:1903`), and the item residual
  has the bundle's variance T (`:1955-1960`).

**"LEVEL kept" after a refit.** The same three numbers mean a different prior
once T and the ridge are refitted. Two quantities show it:

* **The level's quantiles in accuracy terms** (logit of a benchmark's rate at
  standing 0). At archive-3's T of 9.51, LEVEL's median is -1.15. At the pool's
  T of 5.80 the same mu0 = -2.5 is -1.38. Re-expressing LEVEL at T_pool so that
  its ±1 sd points and median keep their accuracy-scale values gives about
  N(-2.09, 2.10²) (`code_levels.json: LEVEL_mapped_to_T_pool`).
* **The B0 prior centre.** Below is E[sigmoid] over the level, theta, the pair
  deviation and the item residual, at the reference set's mean attribute score.
  It is approximate: no identity term, no floor, no group key, slip 0.01
  (`code_variants.json: b0_centres`).

| reference subjects | mean score, archive-3 / pool ridge | archive-3 | C1, LEVEL literal | C1, LEVEL re-expressed at T_pool | mu0 that gives archive-3's centre under C1 | C2a (pool EB level) |
|---|---|---|---|---|---|---|
| matharena's | +0.35 / +0.19 | 0.316 | 0.293 | 0.316 | -2.19 | 0.518 |
| multi_swebench's | -0.02 / +0.07 | 0.304 | 0.288 | 0.311 | -2.29 | 0.513 |
| real_webagents' | -0.20 / +0.11 | 0.297 | 0.289 | 0.313 | -2.40 | 0.515 |
| researchcodebench's | -0.53 / +0.11 | 0.286 | 0.289 | 0.313 | -2.55 | 0.515 |
| the pool's, by benchmark | -0.91 / -0.01 | 0.273 | 0.285 | 0.308 | -2.66 | 0.509 |

What the table shows:

* **The B0 centre barely moves.** The smaller T lowers it, and the loss of
  archive-3's negative intercept raises it. With LEVEL kept literally, the
  centre is within 0.024 of archive-3's on every set.
* **Re-expressing the level alone overshoots by 0.02 to 0.035.** It corrects
  T but not the attribute centre or the smaller total variance (16.8 against
  21.4).
* **The spread across subjects nearly vanishes.** On the eta scale it drops
  from about 0.91 (0.5 × 1.83) to 0.20 (0.5 × 0.41), because the pool ridge is
  nearly flat. That is a change in "the rest", not in the level.

**The pool's own level distribution, at the run-time scale.** These are exact
levels at T_pool = 5.797 and attribute score 0 under the pool ridge, from the
127 prototype benchmarks (`code_variants.json: exact_levels_T_pool`).

| set | n | mean | EB mean | EB sd (net of mean sampling variance) | 5% / 50% / 95% | excess kurtosis | fit_hyper's formula (×1.44) |
|---|---|---|---|---|---|---|---|
| all binary | 127 | +0.18 | +0.16 | 2.30 (0.19) | -3.51 / +0.33 / +3.61 | 1.05 | 3.41 |
| without the nine flagged | 118 | +0.37 | +0.35 | 2.12 (0.16) | -3.25 / +0.40 / +3.64 | 0.65 | 3.15 |
| the flagged | 9 | -2.29 | -2.44 | 2.96 | | | |

* **Where LEVEL sits in it.** LEVEL re-expressed at T_pool has its median at
  the 15.7th percentile of these levels (13.6th without the flagged), and its
  95th percentile at the 66.9th. This agrees with data.md §5's 15.7% on the
  accuracy scale.
* **Shape.** The distribution is close to Gaussian, with moderate tails. A
  Student-t level (`nu_mu > 0`) is implemented, but `_check_bundle` refuses it
  (`tools/build_submission.py:173`), and findings put it at 5 to 10 times the
  Gaussian's time per call.
* **Should it replace LEVEL?** Code-wise this is C2, below. Statistically the
  pool estimate is the population distribution the hidden benchmarks were
  drawn from (the competition page's random assignment). The team's reading of
  run 1, quoted in the task brief as a logit mean of about -1.6, would sit
  about 2.9 sd of a seven-benchmark mean below the pool's accuracy-scale
  centre: (-1.6 - 0.06) / (1.54 / √7), with data.md §5's +0.06 and 1.54. I
  read no feedback file for this. The arithmetic treats a pair-mean logit as a
  level, and that pair mean also holds the run's subjects.
* **What C2 bets on.** C2 bets that the hidden benchmarks are a typical draw
  and that what the formative runs sample does not shift the level. Held-out
  public benchmarks favour C2 by construction. Only regimes whose realised
  level matches the hidden reading can show what it costs if the bet is wrong.

## 5 The identity cap and the clip

**The shipped guards.** id_cap = 0.1 (`paiec/hier.py:349`, applied at
`:1910`) and the tau2 clip [0.01, 0.2] (`paiec/prior.py:794`) were set when
tau2_res was -0.002 on 21 linked names (`paiec/prior.py:599-602, 701-721`). On
the prototype pool (`code_levels.json: identity`, `jackknife`):

**The variance components.**

* tau2_res is 0.403. Its benchmark jackknife over the 113 multi-pair
  benchmarks gives SE 0.123, with leave-one-out values from 0.350 to 0.439.
  bird_sql moves it most, by -0.053.
* At alpha 2 it is 0.408. Without the flagged benchmarks it is 0.419. The link
  noise subtracted is 0.048.
* s2d_res is 0.641 (SE 0.141). s2_res is 2.56 (SE 0.67), and bird_sql and
  alignment_faking move it most.
* s2d_res (0.64) plus tau2_res (0.40) is well below s2_res (2.56), so the named
  rows vary much less than the unnamed ones. sigma_delta, one width for all
  pairs, is pooled over both.

**What the guards do to the weights.**

* The identity weight min(V / (V + s2d/k), id_cap) has V = sigma_theta² + u.
  Uncapped, it runs over the 1,569 prototype rows with an entry from 0.24 to
  0.93, with median 0.64, at the clipped tau2. At the raw tau2 it runs from
  0.39 to 0.96, with median 0.77.
* So id_cap binds on 100% of rows. archive-3 at the clip's floor capped 91% of
  the two-benchmark rows and 12% of the others (`paiec/prior.py:714-718`).

**What the data say the weight should be.** For each name with cells on two
or more benchmarks, I regressed its residual cell on benchmark b on the mean
of its other cells. The cells are from_standings' alpha-2 residuals, offsets
removed, replicated in `code_levels.py`.

| other cells | cells | slope, any benchmark | slope, outside own suite (HELM, MathArena, IgakuQA) | slope, different harness string only | model weight, raw tau2 / clipped |
|---|---|---|---|---|---|
| 1 | 158 | 0.70 | 0.70 | 0.71 (94 cells) | 0.39 / 0.24 |
| 2 to 4 | 173 | 0.74 | 0.75 | 0.75 (180) | 0.56 to 0.69 / 0.38 to 0.52 |
| 5 or more | 498 | 0.90 | 0.86 | 0.86 (481) | 0.81 to 0.90 / 0.69 to 0.82 |

* **It is not an artefact of shared pipelines.** Restricting the other cells to
  other suites or other harness strings leaves the slopes where they were
  (`code_variants.json: loo_slopes`, `code_harness_slopes.txt`). 36 of the
  names on exactly two benchmarks are on ai2d_test and hallusionbench, both
  VLMEvalKit, and still the slope holds.
* **What the model would give uncapped.** The weight formula at the raw tau2,
  with no cap, sits at or below the measured slope wherever a name has one,
  three or more other cells, so the cap is not what keeps the weight below the
  data. At two other cells (87 cells, correlation 0.33) the formula is the
  higher, 0.56 against 0.42.
* **What lifting the clip changes elsewhere.** sigma_delta drops to 2.05
  (`:797`) and the within-run link weight rises to 0.088.
* **Larger risks once the weight is large.** At id_cap 0.1 a name collision
  costs little. At a weight of 0.4 to 0.9 it does not. data.md §4 lists short
  canonical names ("o1", "r1", "v3"; none collides in the pool). A related
  oddity: rbar is a residual from the full-scale ridge but is added next to
  attr_scale × m. That is immaterial under the nearly flat pool ridge.

## 6 Run time

From `code_timing.jsonl` and the prototype's `build_K100_M2000.json`. RSS is
the process's running maximum.

| fit | pairs | items | cells | CPU per EM iteration | RSS |
|---|---|---|---|---|---|
| matharena, uncapped (shipped scale) | 81 | 1,633 | 19,554 | 0.32 s | 0.34 GB |
| multi_swebench, uncapped | 82 | 2,127 | 57,835 | 0.70 s | 0.52 GB |
| alignment_faking, K100 M2000 | 100 | 1,464 | 47,787 | 0.78 s | 0.69 GB |
| alignment_faking, K200 | 200 | 1,464 | 95,608 | 1.41 s | 0.91 GB |
| alignment_faking, K400 | 400 | 1,464 | 191,057 | 3.38 s | 1.36 GB |
| alignment_faking, K1000 | 1,000 | 1,464 | 477,381 | 14.4 s | |
| sib200, K100, no item cap | 100 | 16,932 | 20,400 | 0.21 s | |
| sib200, K400, no item cap | 400 | 37,740 | 81,600 | 0.86 s | |
| mmbench_v11, all pairs, M2000 | 251 | 2,000 | 482,672 | 4.33 s | |
| llm_uncertainty_bench, K100 M2000 | 100 | 2,000 | 66,649 | 0.69 s | 1.38 GB |
| llm_uncertainty_bench, K100, no item cap | 100 | 149,793 | 4,993,100 | 111 s | 8.2 GB |

* **Iterations.** The prototype's EM ran 21 iterations at the median, 194 at
  the 90th percentile and 500 at most.
* **Why caps are required.**
  * One uncapped dense benchmark takes about 40 CPU minutes at the median
    iteration count, and 8 GB.
  * The pool's eligible cells number 34.4 million uncapped, against 2.42
    million in the prototype.
  * Cost grows faster than linearly in pairs: in `Problem.precision`
    (`paiec/hier.py:954-970`), each item's within-item score covariance is
    either dense over all columns or pairwise over its labels.
  * alignment_faking uncapped (4,278 pairs) is out of reach.
* **The prototype build.**
  * Load: 27 CPU s.
  * Fits: 565 CPU s, 319 s wall as recorded (parallelism not recorded). The
    slowest fit is disco, at 49.5 s and 207 iterations. Each worker peaked at
    1.53 GB or less.
  * The fixed-T level pass (C2 only): 138 CPU s.
  * from_standings: 2.3 CPU s.
  * Bundle, checks and validator: seconds.
* **Scaling to 166 or 250 benchmarks** (estimates from the per-benchmark costs,
  at the same caps). 166 benchmarks with the mixed ones cost about 750 CPU s;
  250 about 1,150 CPU s. At six workers on an idle machine that is 3 to 6
  minutes wall, the longest fit included, and at most 6 × 1.5 GB of memory.
* **Leave one benchmark out.** The fits do not depend on the exclusion set.
  Each exclusion costs one from_standings, 2.2 CPU s at the median (113
  jackknife runs in 68 s wall on 4 workers), plus fit_hyper's arithmetic.
  166 parents cost about 6 CPU minutes. The single-pair fits at T, and the C2
  pass, depend on T, and T is a median over 113 or more fits. Fitting the C2
  pass once at the full pool's T and dropping the held-out benchmark's level
  per exclusion is the cheap, nearly exact choice. It has to be stated in
  advance, because T then saw the held-out benchmark.

## 7 Archive limits and shipped-module constraints

**Size and load** (`code_archive.json`). The C1-shaped prototype bundle has
LEVEL written over the prototype pool fit.

| | archive-3's prior.json | C1-shaped prototype |
|---|---|---|
| prior.json | 18,951 bytes (106 table rows, 5 included) | 84,389 bytes, 25,775 deflated (494 rows, 127 included) |
| archive | 73,374 bytes | 92,807 bytes; every other member is byte-identical |
| json load + from_json + HierPredictor | 0.89 ms | 4.06 ms |
| fresh interpreter, model.py exec (validator-style loader) | 30.2 ms median | 33.0 ms median, 36.5 max |
| numpy import / make() | 77 ms / 0.005 ms | 81 ms / 0.009 ms |
| organisers' validator, full checks | (archive-3 passed) | OK |

* **The other members are archive-3's.** Zipped from HEAD's sources with the
  build's own `members` and `write_zip`, the archive hashes to
  `4a882cc7d410e6a9…`, archive-3's sha256 (`docs/plans/p2_final_plan.md`
  section 1). So a C1 or C3 archive differs from archive-3 in prior.json
  alone.
* **All 688 names would fit.** The table costs about 120 bytes a row, so all
  688 names give roughly 100 to 110 KB.
* **Neither size nor load time is a limit.** The validator sets no size limit
  (`third_party/paiec_baseline/check_submission_zip.py:177-232`). Load time
  is paid once per recreated worker, against an 8-hour run limit
  (`docs/protocol.md:133`).
* **theta_prior costs 56 µs per subject dict** and is memoised
  (`build_K100_M2000.json`; `paiec/hier.py:1882-1892`).

**Constraints every configuration must keep.**

1. **What ships.** `paiec/{__init__,hier,prior,predict,fitting,irt,subjects,mcq}.py`,
   renamed `paiec_rt/` (`tools/build_submission.py:61-64`). Imports inside it
   are relative; an absolute `paiec.` import names a module the archive lacks
   (`:249-251`).
2. **Module-level imports: the standard library and numpy only** (`:69`,
   checked at `:268-269`).
3. **Every local import must resolve to a shipped module, lazy ones included**
   (`:264-267`). The exceptions are the `OFFLINE_ONLY` functions
   `irt.evaluate` and `prior.main` (`:67`). So the pool loader must live in an
   unshipped module and never be imported from a shipped one, unless the
   importing function is added to `OFFLINE_ONLY`. `prior.main` already
   imports `.data` that way (`paiec/prior.py:865`).
4. **scipy, scikit-learn and pandas never load at run time on public inputs**
   (`HEAVY`, `:71`, checked at `:403-404`). The pool fit's pandas use stays
   in offline functions (`paiec/prior.py:374`).
5. **predict stays a pure function of (input, labeled).** hier caches fits by
   a fingerprint of `labeled` (`paiec/hier.py:1989-2011`) and per-subject and
   per-item memos keyed on content digests. prior.json is static. C1 to C3 add
   no state.
6. **Nothing raises out of predict** (`submission/model.py:109-115`,
   `paiec/hier.py:2039-2057`). prior.json must pass `from_json`
   (`paiec/prior.py:833-849`, `SubjectPrior.validate` `:168-197`), which
   refuses non-finite values and negative ones other than mu0 and shift. C2's
   positive mu0 passes.
7. **The archive predicts bit for bit like the research package on the same
   prior.json** (`check_run`). One BLAS thread (`submission/model.py:43-45`).
8. **App G.6.** prior.json holds no item- or benchmark-keyed table, and no
   shipped or tools code reads `results/formative/` or an anonymous id. The
   pool bundle carries only Hyper and SubjectPrior (`to_json`,
   `paiec/prior.py:824-830`). fit_hyper's per-benchmark levels (`rep['levels']`)
   and the fixed-T levels must stay out of it. `included` lists public
   benchmark names, as it does now.
9. **The fallback keeps archive-3's widths.** `Hyper()`'s defaults and `PUBLIC`
   (`paiec/hier.py:244, 338-359`) stay the public-five fit. They matter only in
   model.py's fallback (`:89`). Under C1 to C3 that fallback would predict at
   LEVEL with archive-3's widths. Changing them changes the shipped `hier.py`.

## 8 Candidate configurations

Each configuration is a set of offline build options. Only the last column
reaches the archive.

| | subject prior (ridge, identity table, s2d) | sigma_delta, sigma_attr, sigma_d, sigma_g | sigma_theta | id_cap | mu0, sigma_mu | attr_scale | archive against archive-3 |
|---|---|---|---|---|---|---|---|
| **C0** archive-3 | public five | public five | 0.1 | 0.1 | -2.5, 2.5 (LEVEL) | 0.5 | none (the fallback) |
| **C1** pool refit, LEVEL kept | pool | pool | pool, clip [0.01, 0.2] (0.447 in the prototype) | 0.1 | -2.5, 2.5 (LEVEL) | 0.5 | prior.json only |
| **C2a** C1 with the pool's EB level | pool | pool | as C1 | 0.1 | pool EB at T_pool: about +0.16, 2.30 (all) or +0.35, 2.12 (without flagged) | 0.5 | prior.json, and the LEVEL literal and docstring in model.py |
| **C3** C1 with the identity re-estimated | pool | pool; sigma_delta recomputed (2.05) | sqrt(tau2_res) unclipped (0.635) | none (1.0) | -2.5, 2.5 (LEVEL) | 0.5 | prior.json only |
| **C2a+C3** | pool | as C3 | as C3 | none | as C2a | 0.5 | as C2a |

* **C1.** Run the loader and the parallel fits, `from_standings`, then
  fit_hyper's arithmetic, with LEVEL written over as now. By the table in
  section 4, its B0 centre is within 0.024 of archive-3's at every reference
  set. A variant that re-matches mu0 to archive-3's B0 centre would need a
  reference subject set fixed in advance, and its mu0 would land between
  -2.19 and -2.66. It is not needed at that tolerance. What changes is the
  rest: subject spread at B0 (section 4), T, sigma_theta and the link weight
  (from 0.0018 to 0.042).
* **C2a.** It adds the fixed-T pass (138 CPU s) and the EB summary. The
  numbers come out of a committed script and are written into
  `submission/model.py`'s LEVEL, because the build checks bundle against
  literal (`tools/build_submission.py:169-172`) and LEVEL is also the
  fallback. Three things must be fixed in advance: the level set (all
  benchmarks, or without the flagged ones), the widening (none, against
  `WIDEN[0]`, which gives 3.41), and attr_scale (kept at 0.5, so that only the
  level moves).
* **C2b: the pool's spread with the hidden reading's centre.** mu0 -2.5, which
  App G.6 lets the feedback set as a global quantity, with sigma_mu at the
  pool's 2.30. It is C1 to within 0.2 of sigma_mu. Within the widths
  regime-sensitivity scanned at this mu0, 2.5 and 3.5 differed by under 0.001
  (`docs/findings.md`, "P1.16"). It is listed so that the level is visibly
  split into a centre (the feedback) and a spread (the pool), not as an arm of
  its own.
* **C3.** id_cap and sigma_theta are Hyper fields in prior.json, and the clip
  is in the offline arithmetic, so this is again prior.json only. Lifting the
  cap (rather than setting it at a measured slope) is backed by section 5: the
  uncapped model weight sits at or below the measured slopes wherever a name
  has one, three or more other cells. It is set apart from the level so that
  the identity effect can be read without the level's confound. On named
  subjects it moves theta's prior mean by w × rbar, where rbar has sd 0.9 to
  2.3 depending on k. That is the largest B0 change among C1 to C3 for those
  subjects.
* **Excluded.**
  * Reading `source_model_label` in `canon_name` would name 5,755 more pairs.
    It changes the shipped `prior.py` and run-time linking in every run.
  * A mixture or Student-t level.
  * Run-time EB of the level, which was measured before and not shipped
    (`docs/findings.md`, "The empirical-Bayes level (descriptive)").

## 9 The minimal code changes, file by file

**New: `paiec/pool.py`.** Offline. Not in `PACKAGE`, and never imported by a
shipped module.

* `select(manifest, rule)`: the revision pin, status ok, release public,
  granularity item, the type rule, and the direction rule per part.
* `load_benchmark(b, K, M_p, order)`: `proto_loader.py` with the section 2
  changes and named-column reads.
* `fit_all(names, caps, jobs, cache)`: `benchmark_fit` in worker processes,
  cached on disk keyed by (revision, benchmark, rule, caps, s2e).
* `bundle(fits, exclude, level=..., identity=..., share=..., design=...)`:
  standings rows, `PR.from_standings(rows, 2.0, included, excluded, design)`,
  and fit_hyper's `:756-807` arithmetic on fit dicts. The parameters are the
  share when no fit has keys, the clip bounds, the level mode (written over,
  or EB on the fixed-T pass) and id_cap.
* A `main()` that prints the report `prior.main` prints (`:862-879`).

**Changed: `tools/build_submission.py`** (not shipped).

* A `--pool` path in `main` (`:431-469`) and `write_bundle` (`:113-126`) that
  fits through `paiec.pool`. `check_run` keeps the legacy session
  (`load_pairs`, `:442`) for the bit-identity check, since the pool has no item
  text.
* `_check_bundle` (`:175-176`) compares `included` with the pinned pool list,
  not `PUBLIC`. The list goes in a committed file written by the rule, so that
  the tests need no data.
* New checks on id_cap and sigma_theta against the configuration's values.
* The docstring (`:1-45`) is updated.

**Changed: tests** (not shipped).

* `tests/test_submission.py`: `toy_bundle` (`:130-140`) and
  `test_archive_ships_the_calibrated_hier` (`:235`) expect the pool list.
* New `tests/test_pool.py`, on synthetic `formatted_tables`. It checks:
  * the first 0/1 response in file order, nulls dropped and min_items;
  * deterministic caps, and that the per-pair cap keeps a sparse benchmark;
  * the type and direction rules;
  * a missing column raising.
* A test that the archive differs from archive-3 only in prior.json (C1, C3),
  or in prior.json and model.py (C2).

**Changed for C2 only: `submission/model.py`.** The LEVEL literal (`:68`) and
the docstring (`:3-8`, `:62-67`).

**Unchanged.** `paiec/hier.py`, `prior.py`, `subjects.py`, `predict.py`,
`fitting.py`, `irt.py` and `mcq.py` (all shipped), and `paiec/data.py` and
`official.py`.

**The one optional shipped-file change.** Refactor `fit_hyper` (`:672-807`)
into a core on fit dicts, if the duplicated arithmetic is judged worse than a
byte change in `prior.py`.

**The gate's evaluation harness** (not shipped, but the gate needs it).

* `experiments/level_calibration.py:260-265` (`bundle`) and
  `experiments/regime_sensitivity.py:668-677` (`Maker.build`) fit `PR.build` on
  the legacy pairs. A pool arm swaps in `pool.bundle(fits, (parent,))`.
* Test-like pseudo-benchmarks group by `item_features`
  (`paiec/testlike.py:209`, `PARENT_KEYS`). Pool parents without items give
  strata and chunks only.
* Evaluation inputs built from the pool carry no `item_content` and no
  `item_features`. On held-out pool benchmarks hier would then run without its
  multiple-choice floor (`paiec/hier.py:1944-1953`) and without group effects,
  unlike on the hidden test. Removing that difference needs `items.parquet`.

## 10 Choices amendment 4 has to fix in advance

1. **The benchmark rule**, separately for the subject part and the level part.
   One candidate is data.md §7's "ordered set" for the subject part, and all
   binary and mixed benchmarks at a 0/1 share of 0.95 or more for the level
   part.
2. **The caps**: K, M_p and the pair order. K = 100 with the canonical-name
   order, and M_p of a few hundred, cost about 1 CPU s per EM iteration on
   the largest benchmarks.
3. **The group share without items**: 0.25, archive-3's, or the 0.138 fallback
   that blank interactors produce.
4. **The degenerate and non-converged fits**: the five with item variance over
   100, and the 9 + 13 that did not converge. Keep them in the subject part,
   or drop them by a stated rule. The level part uses the fixed-T pass either
   way.
5. **The level mode**: LEVEL literal (C1, C3), or the pool EB (C2a) with its
   set and widening.
6. **The identity mode**: archive-3's guards (C1, C2a), or no cap and an
   unclipped tau2 (C3).
7. **The provider threshold**: `design={"min_count": ...}` passes through
   without a code change.
8. **The revision**: `54074453`. Collections may grow (the competition
   page), so a later fetch is a new fit.
