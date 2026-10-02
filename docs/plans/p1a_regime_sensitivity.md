# P1a: regime sensitivity of the shipped level (plan and rule, fixed before scoring)

**Status.** Fixed at 2026-10-01T14:56:59Z (UTC), before any run of the scoring seed was
drawn or scored. The decision rule (section 8) is quoted verbatim from
`RULE_TEXT` in `experiments/regime_sensitivity.py`, whose sha256 is
`47e4735cf7ab0192e2c375054911221485e19ec71775bea8390b0e60287dc58e`. The script
also holds this file's sha256 (`PLAN_SHA256`) and the time stamp
(`FIXED_AT_UTC`). The files are new and uncommitted at HEAD 380fecf; the hash
and time stamp are self-reported until the team commits them. Committing both
files before the scoring stage starts gives the lock an outside time stamp.

**Owner of the numbers.** Every number this study reports will come from
`python experiments/regime_sensitivity.py` writing `results/regime_sensitivity.json`.
The numbers below marked *probe* come from scratch probes run to write this
plan (section 12). They are not results; the script's `regimes` and
`reproduce` stages re-measure each one.

## 1 What this answers

The internal review (`docs/report/review_v0.md`) asks:

* **W3 / Q3 / P1.10.** The shipped level (`submission/model.py` LEVEL: mu0 -2.5,
  sigma_mu 2.5, attr_scale 0.5) was chosen in the tuned test-like regime
  (realised pair logit -1.29, sd 1.70) under a public guard ("Calibrating for the
  hidden test", "What actually shipped, after the audit"). The audit moved it to
  the milder setting because the hidden levels "look spread both ways". The
  pooled reading of formative runs 1 and 2 puts them at a mean of about -0.65 to
  -0.96 with an sd of 1.75 to 2.2 ("Formative feedback, runs 1 and 2"). No regime
  with that level and spread has been scored for the shipped config. How do the
  shipped config, -3.0/0.25, EB, a calibrated smoothed mean and the legacy
  Predictor rank there, and under a two-component mixture?
* **P1.9 / W4.** What does a level-calibrated smoothed mean, selected by the same
  rule as LEVEL, get? What does a plain pooled-level model without the subject
  prior get? Together they show how much of the gain needs hier.
* **P1.16.** sigma_mu 2.5 sits on the edge of the scored grid. Do sigma_mu 3.5 and
  5.0, at the shipped mu0 and attr_scale, do better?

The study can produce one action: a list of **candidates** under the rule of
section 8. A candidate is a proposal for the team. It does not ship by itself.

## 2 What is not allowed

* Formative run 3 (`results/formative_run3.json`) is a regression check. It is
  not read for anything here: no level, regime, threshold or config comes from
  it.
* No predictor is scored on the scoring seed (11) before this plan and the rule
  are fixed. No threshold, config, regime knob, seed, run count or statistic
  changes after the first row of seed 11 exists. There is no early stopping and
  no extension: the rule is applied once, to the full planned set.
* Every prior and hyperparameter fitted offline is fitted leaving the target's
  parent benchmark out (`level_calibration.bundle(parent)` and `ppred(parent)`,
  dispatched on the anonymous `benchmark_id`), as in every earlier calibration
  study. The two global settings chosen here are the calibrated smoothed mean's
  (n0, m0), selected on the same old runs and by the same rule as LEVEL
  (section 3), and the regime knobs, set by run composition alone with no
  predictor (section 4).
* All configs are scored on identical runs and checkpoints (paired).

## 3 Configs

All hier configs use `level_calibration.bundle(parent, 0.0)`, which is
`paiec.prior.build(pairs, (parent,))`. Names on the left are the keys in
`CONFIGS`.

| key | definition | why | probe s/run (mean of 10 runs) |
|---|---|---|---|
| `ship` | hier, `replace(hyper, mu0=-2.5, sigma_mu=2.5, attr_scale=0.5)`; level_calibration name `hier G mu0=-2.50 sm=2.50 as=0.50` | the reference: what ships | 5.2 |
| `aggr` | hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 (`level_calibration.SHIP`) | the guarded neighbour the calibration recommended | 5.4 |
| `eb_fit` | hier with the empirical-Bayes hyperparameters `prior.build` fits, nothing overridden (level_calibration `hier`). Held-out-parent fits: mu0 -0.44 to -1.55, sigma_mu 1.11 to 3.26, attr_scale 1; on all five, `Hyper()`'s -1.263 / 2.676 | "EB" in the task's sense: the level before LEVEL overrides it | 5.3 |
| `eb_adapt` | `hierEB-cs,tm=2.0@hier G mu0=-3.00 sm=2.50 as=0.50` (`level_calibration.EBHier`): centre and scale re-estimated from every pair in `labeled` at each checkpoint, tau_mu 2, tau_s 0.3, floor 0.5 | the adaptive "EB level" of review W3 and findings' tables; scored by the audit's extra stage | 6.3 |
| `eb_ship` | the same adaptation on the shipped base, `hierEB-cs,tm=2.0@hier G mu0=-2.50 sm=2.50 as=0.50` | identical to `ship` at B0, so `eb_ship` minus `ship` is the adaptation alone; if EB is to be shipped, this is the form it would take | 6.4 |
| `wide35` | hier G mu0 -2.5, **sigma_mu 3.5**, attr_scale 0.5 | P1.16 | 5.4 |
| `wide50` | hier G mu0 -2.5, **sigma_mu 5.0**, attr_scale 0.5 | P1.16 | 5.2 |
| `legacy` | `paiec.predict.Predictor(*ppred(parent))` (the first submission) | the old baseline; the original guard's comparator | 5.6 (2.8 to 12.7) |
| `smooth` | `baselines.smoothed_mean(4.0, 0.5)`, Beta(2,2) on the pair's own labels | the regime-neutral baseline | 0.1 |
| `smcal` | `baselines.smoothed_mean(n0*, m0*)`, (n0*, m0*) selected below | P1.9 / W4: the minimal model that "predicts lower" | 0.1 |
| `onepl` | hier at `ship`'s level (mu0 -2.5, sigma_mu 2.5) with `Flags(attributes=False, identity=False)`: a pooled benchmark level, item difficulty, group effects and the MCQ floor, no subject prior | P1.9 / W4: the subject prior's share; a cheap hier flag setting | 5.4 |

**`onepl`'s centre.** With the attribute prior off, mu0 is no longer measured
from an attribute score of 0, so `onepl`'s effective centre differs from
`ship`'s by attr_scale times the pool's mean attribute score (-0.53 to +0.35 on
public pools, `Hyper`'s docstring). It also ignores the date shift's attribute
optimism. Both differences are part of what `ship` minus `onepl` measures.

**`smcal`, defined exactly.** The prediction is (k + n0 m0) / (n + n0), with k
successes in n own labels (subject dict and `benchmark_id` matched as in
`baselines.smoothed_mean`).

* Grid: n0 in {0.5, 1, 2, 4, 8, 16, 32} times m0 in {0.05, 0.10, ..., 0.50}, 70
  points. Beta(2,2) is (4, 0.5).
* Selection is LEVEL's rule (`level_calibration.select`) on LEVEL's runs. Take
  test-like seed 2 runs 0-99 (the selection half). The guard is public R1 seed 0
  runs 0-99, benchmark-first and pair-uniform. The comparator is the legacy
  Predictor's per-pair ALC stored in `results/level_calibration.json`, the rows
  LEVEL's guard used. The rule: among the grid points whose mean ALC minus the
  legacy Predictor is at most +0.003 (`GUARD`) on both weightings, take the one
  with the lowest selection-half mean ALC. Ties within 1e-6 go to the larger n0,
  then to the m0 nearer 0.5.
* If no point passes the guard, `smcal` is the unconstrained argmin on the
  selection half (same ties). It is flagged `guard_failed`, with every point's
  guard loss recorded. This gives the smoothed mean its best shot in the tuned
  regime, so hier's margin over it is not flattered. Beta(2,2) itself loses
  +0.0070 and +0.0075 on that guard ("Calibrating for the hidden test"), so this
  branch is likely.
* Check: the grid's (4, 0.5) column must equal `level_calibration.json`'s stored
  smoothed rows on those runs to 1e-5 (their packing).
* The stage reads seeds 2 and 0 only. It writes (n0*, m0*) into the results
  before the scoring stage may start.

**Not scored** (cost, or a different question): the Student-t level; the
legacy Predictor with a level fix; a plainer 1PL with group effects and the
floor off; BLE.

## 4 Regimes

Every test-like regime is `paiec.testlike.Regime` at its defaults (date shift
1.25 years, swe_rebench excluded, benchmark-level tilt) except the level knobs
named. The *realised* level is `testlike_check`'s: the continuity-corrected
logit of each pair appearance's evaluated responses (`testlike.describe`), with
the unweighted mean and sd (ddof 0) over all appearances. Tilt targets and
realised levels differ: the default target -1.6 / 1.5 realises -1.29 / 1.70. So
each knob is set by a search over run composition (no predictor) on the
calibration seed 10, 1,000 draws a setting. It minimises ((mean - m*) / 0.05)^2
+ ((sd - s*) / 0.05)^2, or for the mixture ((share < -3 - 0.118) / 0.01)^2 +
((mean of the rest + 0.226) / 0.05)^2.

| key | knobs | target (realised scale) | probe, seed 10, 1,000 draws: mean / sd (share < -3) | why |
|---|---|---|---|---|
| `TUNED` | `Regime()` | known: -1.290 / 1.696 (`testlike_check.json`, seed 2) | -1.273 / 1.731 (0.136) | LEVEL was chosen here; the reproduction regime |
| `READING` | level_mean -0.85, level_sd 1.75 | -0.715 / 1.772 | -0.729 / 1.793 (0.089) | the preregistered pooled reading of runs 1 and 2 |
| `AUDIT` | level_mean -1.8, level_sd 2.1 | -1.10 / 2.00 | -1.102 / 1.997 (0.159) | the audit's reading of run 1 (review W3) |
| `MIXTURE` | `level_mix ((0.20, -4.38, 0.171), (0.80, -0.75, 1.151))` (new option, section 10) | share < -3 0.118; rest mean -0.226 | -0.705 / 2.062 (0.118); rest -0.227 / 1.676 | the two-component shape of the per-pair reading |
| `FLAT` | level_mean None (no tilt) | none: the catalogue's own levels | -0.338 / 2.287 (0.103) | hidden levels like the public catalogue's |
| `R1B` | `official.sample_run`, weighting 'benchmark', split scope 'pair' | known -0.71 / 1.42 (public R1) | | the guard |
| `R1P` | the same, weighting 'pair' | | | the guard |

**READING: which estimate, and why.** The target is the mean of the K 15 and
K 40 readings in `results/formative_feedback.json`
(`reading.level_distribution`): continuity-corrected means -0.650 and -0.779
(mean -0.715), sds 1.800 and 1.745 (mean 1.772).

* The continuity-corrected logit is the convention of the regimes' realised
  levels. The preregistration fixed it as the reading's level for that reason,
  and the comparator -1.290 / 1.696 is on that scale.
* The plain logits (-0.83 / 2.23 and -0.96 / 2.17) blow up the two pairs near a
  zero rate (-5.06 and -6.57 against -4.21 and -4.55) and are not commensurable.
* Averaging the two K values averages over the one pair whose root flips between
  them (run 2's q1: +1.09 at K 15, -1.09 at K 40). Neither K was preferred in
  the preregistration.

**AUDIT.** The review's "level_mean about -1.1 with level_sd 2.0", taken as
given on the realised scale. It is the audit's run-1 reading (p6 and p8 at
upper roots, -1.08 / 1.99 on plain logits). On the continuity-corrected scale
that reading would be milder. Taking it as given makes AUDIT the lower of the
two feedback-motivated regimes, so READING and AUDIT bracket the feedback
between -0.7 and -1.1.

**MIXTURE, defined from the data.** Both two-component fits of the 17-pair
reading (`reading.level_distribution.K=15/K=40.mixture.two`) put weight 0.117 at
-4.378 (sd 0.171): the two pairs near a zero rate, 2 of 17 = 0.118. They put
the rest at -0.155 or -0.301 (sd 1.165, 1.137). The targets are what the data
show directly:

* the share of pair logits below -3 is 0.118;
* the mean of the rest is -0.226 (the K 15 and K 40 means of the 15 pairs above
  -3, -0.153 and -0.299).

The low component keeps the fit's mean and sd (-4.38, 0.171) and the high
component the fit's sd (1.151, the mean of the two). The weight and the high
mean are searched. The search hit its first grid's edge (w 0.18), so the grid
was extended, still before scoring: w in {0.12, ..., 0.26} times high mean in
{-0.25, ..., -0.75}. The probe's argmin is w 0.20, high mean -0.75.

The mixture is a stress test, not a supported model. A second component
improves BIC by only 0.68 and 0.38, under the bar of 2, and is preferred in 0%
and 0.4% of bootstrap draws.

The catalogue cannot realise the high component's sd of 1.15: pairs within one
pseudo-benchmark already spread with sd 1.04, and only 3 of 37 live
pseudo-benchmarks sit below -3. The mixture's rest sd is 1.68, and that is
recorded. Its overall mean, -0.71, matches READING's, at a different shape (sd
2.06, a separate low mode), so MIXTURE against READING shows whether shape
matters.

**Realisation checks** (on the scoring seed, before any predictor). For each
test-like regime, the realised level of its 80 planned runs must lie within
these bounds of its seed-10 value above:

* mean ±0.30 and sd ±0.30;
* for MIXTURE, also the share below -3 ±0.04 and the rest's mean ±0.25.

On seed 10, disjoint 80-run blocks vary by sd 0.07 to 0.10 (mean), 0.05 to 0.10
(sd), 0.008 to 0.013 (share) and 0.07 to 0.085 (rest mean) (*probe*). So the
bounds are about 3 block sds: a trip means a bug, not noise. If one trips,
scoring does not start. The knobs are not re-tuned on seed 11.

**Not scored:** the no-date-shift regime. Every test-like regime here keeps the
synthetic date shift, so whatever is concluded about attr_scale stays
conditional on it (review Q10).

## 5 Seeds and runs

**Seeds already used**, all `np.random.default_rng([seed, i])`:

* test-like seed 1: tuning (`testlike_check`).
* test-like seed 2: LEVEL's selection half (runs 0-99) and confirmation half
  (100-199); runs 200-299 in `subject_side`, `itemsig_eval` and `harness`.
* test-like seed 3: the sensitivity regimes.
* test-like seed 5: `level_audit --stage extra`, runs 0-39.
* public `sample_run` seed 0: `hier_eval`, LEVEL's guard (runs 0-99, screen
  0-39) and `official_baselines` (runs 0-599).

LEVEL was selected on seed 2 runs 0-99 with the seed 0 guard runs 0-99. It was
confirmed on seed 2 runs 100-199 and seed 3. The audit used seeds 2, 0, 3 and 5.

**Fresh seeds:**

* `CAL_SEED = 10`: regime-knob calibration only. Draws, never scored.
* `SCORE_SEED = 11`: every scored run of every regime, runs 0..N-1. One seed
  for all regimes, as level_calibration's sensitivities shared seed 3.
* `REPRO`: test-like seed 2 runs 0-4, for the reproduction checks only (old runs
  with stored rows).

**Runs per regime: 80 test-like (each of five) and 60 public (each of two).**
This is a cut from the 100 per regime of level_calibration, made for the time
budget (section 6).

* The cluster SEs barely shrink with more runs, because clusters recur across
  runs of one catalogue: shipped minus legacy has cluster SE 0.0032 on 100 runs
  and 0.0034 on 300 (`ship_confirm.json`).
* The cut costs about 12% in run SE on test-like regimes and 29% on public ones.

Tasks run in run-index-major order (i, then regime), split over 2 shards by
task index, so any stopping point leaves every regime with the same prefix.
Stopping early still ends in no reading: the rule waits for the full set.

## 6 Compute

**Timing probe** (*probe*; section 12). All 11 configs were scored in one
process on 6 test-like runs (seed 2, runs 0-5) and 4 public runs (seed 0,
benchmark-first and pair-uniform, runs 0-1). A second process of this study ran
alongside, at a machine load average near 7 on 8 cores. The cost per run, all
configs plus drawing and checkpoints:

* test-like: 42 s (36 to 53);
* public: 62 s (43 to 79).

Per config: the hier family 5.2 to 6.4 s, `legacy` 5.6 s (2.8 to 12.7: its text
fit on researchcodebench), the smoothed configs under 0.3 s. The slowest single
call was 1.14 s (`legacy`); the hier family's was at most 0.46 s.

**Memory.** The probe peaked at 1.24 GB resident (1.30 GB peak footprint) per
process. That is with the model instances released after each checkpoint and no
results JSON loaded. Holding every checkpoint's instances, as level_calibration's
factories do, with two results files loaded, sent the footprint to 3.3 GB on a
9-run probe. So the scoring stage keeps only counters from its instances and
runs `gc.collect()` after each task. Each worker exits cleanly (resumable) once
its resident set passes 1.45 GB. Two workers therefore stay under about 2.9 GB.
They are started as two independent shard processes, with no parent process
holding data.

**Estimate.**

| stage | work | process-seconds |
|---|---|---|
| `smcal` | 300 old runs (checkpoints and the 70-point grid) | about 250 |
| `regimes` | 5 test-like regimes: 1,000 seed-10 draws and 80 seed-11 draws each (`--grid` adds the 106-setting knob search of `KNOB_GRID`, about 16 min) | about 60 |
| `reproduce` | 5 old test-like runs, all configs | about 210 |
| `score` | 5 x 80 x 42 + 2 x 60 x 62 = 16,800 + 7,440 | 24,240 |
| bundles | 2 x 50 | 100 |
| `summarise` | bootstraps, rule | about 300 |

That is about 25,100 process-seconds: 3.5 hours on two processes, so a 5-hour
budget allows a 1.4x slowdown. 100 runs a regime would take 33,400
process-seconds of scoring (4.6 hours on two processes before overhead), which
leaves no margin on a shared machine; hence 80 and 60. If the machine runs
slower than estimated, the scoring takes longer (it is resumable). The run
counts do not change.

## 7 Statistics

The definitions are those of `level_calibration.compare` and
`ship_confirm.Block`.

* **Run ALC.** A run's ALC is the mean over its pairs of 0.1 B0 + 0.2 (B1 + B3
  + B7 + B15) + 0.1 B31.
* **Per config and regime.** Mean ALC ± run SE; mean B0 prediction; ECE (B0,
  ALC-weighted); latency (mean and max per call, ratio to `ship` in the same
  tasks); fallback counts (`failures`, `unconverged`).
* **X minus `ship`, paired.** D ± run SE / cluster SE / stratified SE (`Boot`,
  2,000 resamples, seed 0; stratified by parent), with U95 = D + 1.96 cluster SE
  and the bootstrap's own 2.5 and 97.5 percentiles. Also:
  * per-budget differences, each with its three SEs;
  * per parent: the appearance-weighted mean with its cluster SE (single-subject
    swe_rebench on public runs: point value only);
  * the parent-level mean ± SE over the four multi-subject parents (the only SE
    that sees between-benchmark variation), and its range;
  * the range when one parent is left out (`lopo`);
  * the share of runs X wins.
* **The same against `legacy` and `smooth`,** for continuity with
  `ship_confirm.json` (W1/W4). Also `ship` minus `smcal` and `ship` minus
  `onepl`: the decomposition.
* **Rank tables.** Per regime, configs by mean ALC (review Q3). Two configs
  whose paired |D| is under 1.96 cluster SEs are marked as tied. Descriptive.
* **EB traces.** `eb_adapt`'s and `eb_ship`'s estimated (mu0, sigma_mu) by
  budget and regime, as `level_calibration.eb_adaptation`.
* **Realised levels per regime on the scored runs:** mean, sd, cluster SE,
  leave-one-parent-out range, share below -3.

**Sensitivity of the rule to noise** (to read a pass or a miss).
Hier-versus-hier differences have cluster SEs near 0.001 (MILD minus AGGR: 0.0009
to 0.0011 on 100 test-like runs; 0.0011 to 0.0014 on the audit's 40-run
regimes).

* A config truly 0.004 better passes (a) in one regime with probability about
  0.98, 0.003 better about 0.84, 0.002 better about 0.5. So the rule finds
  improvements of 0.003 or more and is a coin flip at 0.002.
* Under no true difference, (a) passes in one regime with probability about
  0.02. READING and AUDIT are positively correlated (same seed and catalogue).
  Over the roughly eight configs that could plausibly pass, the chance of at
  least one false candidate is several percent. That is acceptable only because
  a candidate goes to the team, not to the archive.

## 8 The decision rule (verbatim `RULE_TEXT`)

```
P1a DECISION RULE: regime sensitivity of the shipped level
(docs/plans/p1a_regime_sensitivity.md; experiments/regime_sensitivity.py).
Fixed before any run of the scoring seed was drawn or scored.

Terms
  SHIP      the shipped configuration: paiec.hier.HierPredictor with
            submission/model.py LEVEL (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5)
            over prior.build fitted without the target's parent benchmark.
  X         any config of CONFIGS other than SHIP.
  regimes   TUNED, READING, AUDIT, MIXTURE, FLAT (test-like; scoring seed 11;
            runs 0-79 each) and R1B, R1P (public R1, split scope 'pair';
            scoring seed 11; runs 0-59 each), as REGIMES defines them.
  D(X,R)    the mean over R's planned runs of the run's pair-mean ALC of X
            minus that of SHIP, both scored on identical runs.
  SEc(X,R)  the cluster-bootstrap SE of D(X,R): Boot of
            experiments/level_calibration.py, 2,000 resamples, bootstrap seed
            0, ratio estimator; the cluster is (parent, subject) on test-like
            runs and (benchmark, subject) on public runs.
  U95(X,R)  D(X,R) + 1.96 SEc(X,R).
  Pq(X,R)   the mean of X minus SHIP over the pair appearances of parent q,
            each weighted 1 / its run's pair count.
  PL(X,R)   the mean of Pq(X,R) over the four multi-subject parents
            matharena, multi_swebench, real_webagents and researchcodebench.
  G(X,R)    D(X,R) with the legacy Predictor in place of SHIP.

X is a CANDIDATE to replace SHIP if and only if all of (a) to (f) hold.
  (a) In READING and in AUDIT: D(X,R) <= -0.002 and U95(X,R) < 0.
  (b) In READING and in AUDIT: PL(X,R) < 0, and Pq(X,R) <= +0.004 for every
      multi-subject parent q.
  (c) In TUNED, in MIXTURE and in FLAT: D(X,R) <= +0.002.
  (d) In R1B and in R1P: D(X,R) <= +0.001 and G(X,R) <= +0.003.
  (e) Pooled over every scored task, X's mean evaluation call takes at most
      2.0 times SHIP's (both timed in the same tasks), and no single call of
      X takes longer than 2.0 s.
  (f) Every reproduction and realisation check of the plan passed, and every
      planned run of every regime was scored for every config.

Reading
  1. No candidate: the outcome is "no candidate: SHIP stays", recorded as a
     negative result for every config.
  2. One or more candidates: all are listed, ordered by the mean of
     D(X,READING) and D(X,AUDIT). The rule does not choose among them, and
     nothing ships automatically: a candidate needs the team's decision, an
     archive that passes tools/build_submission.py, and one formative
     regression check read for errors and latency only.
  3. P1.16: wide35 (sigma_mu 3.5) and wide50 (sigma_mu 5.0) are judged by
     (a) to (f) exactly. "A wider level prior does better" is recorded only
     if one of them is a candidate, and otherwise recorded as no. No sigma_mu
     between or beyond the scored values is chosen from these results.
  4. EB: eb_fit, eb_adapt and eb_ship are judged by (a) to (f) exactly. Only
     if eb_adapt or eb_ship is a candidate is shipping an adaptive level
     revisited (it is experiment code, not library code).
  5. Everything else (ranks by regime, per-budget and per-parent tables, the
     smoothed-mean and no-subject-prior decompositions, EB traces, ECE) is
     descriptive and changes nothing.
  6. Nothing in this rule, CONFIGS, REGIMES, the seeds, the run counts or the
     statistics changes once a row of the scoring seed exists. No run is
     added or dropped after rows are read, and the rule is applied once, to
     the full planned set. If a check fails, the cause is fixed, every
     affected run is re-scored in full and the deviation is recorded; the
     rule is not rewritten. Formative run 3 is not used for anything here.
```

Beyond the task's floor, it adds:

* the parent-level sign in (b), because four parents carry every number and the
  cluster SEs do not see between-benchmark variation;
* FLAT in (c), because the pooled reading is about as central as public R1
  ("Formative feedback, runs 1 and 2"), so a candidate that wins at -0.7 to -1.1
  by losing on central levels is a bet on one level;
* the legacy guard in (d), LEVEL's own original guard;
* the latency gate (e), because a slow worst call was one stated reason not to
  ship EB and the Student-t level;
* the preconditions (f).

Nothing is loosened.

**How P1.16 is read.** As reading 3 says: wide35 and wide50 go through (a) to
(f) like every other config. The finding is recorded as a positive ("a wider
level prior does better") only if one is a candidate. Otherwise it is a negative
result, with D by regime and the per-budget pattern. The probe's ALCs on old runs
(section 12) are not evidence either way.

## 9 The script and its stages

`experiments/regime_sensitivity.py` holds the constants block now (the rule, the
configs, regimes, seeds and run counts, the thresholds, the hashes and the time
stamp) and nothing else. The stages below are to be written against those
constants, unchanged.

1. `lock`: checks `sha256(RULE_TEXT) == RULE_SHA256`, that this file contains
   `RULE_TEXT` verbatim, and `sha256(this file) == PLAN_SHA256`. It records the
   check in the results. Every other stage refuses to run when it fails.
2. `smcal`: section 3. Old seeds only. Writes (n0*, m0*), every grid point's
   selection ALC and guard losses, and the check.
3. `regimes`: realised levels for the fixed knobs on seed 10 (1,000 draws) and
   on seed 11's planned runs (composition only, no predictor), with the
   realisation checks of section 4. `--grid` re-runs the knob search on seed
   10 over the full `KNOB_GRID` (a superset of what the probes drew) and
   records whether its argmin is the knobs fixed here. The knobs stay as fixed
   either way; a different argmin is reported.
4. `reproduce`: on test-like seed 2 runs 0-4.
   * `ship`'s per-pair Brier must equal `data/harness_rows/tl/<i>.pkl`, whose
     library digest is 3f75a549; `hier_floor.json` (`tl_lib_ship`) shows the
     current solver changes no prediction on these runs. Tolerance 1e-9.
   * `legacy`'s must equal `testlike_check.json`'s rows on every pair whose
     parent is not matharena, to 1e-9; matharena pairs differ by the corrected
     multiple-choice floor (f7e7d87) and are reported.
   * `smooth`'s must equal `level_calibration.json`'s to 1e-5.

   *Probe:* exact (0.0) for `ship` against `data/hier_floor/tl_lib_ship` on
   the two of these runs it holds (0 and 3); exact on every non-matharena pair
   for `legacy`, up to 0.011 a budget on matharena pairs; at most 5.3e-6 for
   `smooth`.
5. `score`: the planned tasks, `--shard k/2`, resumable. Rows go to
   `data/regime_sensitivity_rows/<regime>/<i>.json` (gitignored) with:
   * the run's pairs (pseudo-benchmark or benchmark, parent, subject, evaluated
     items, rate, logit);
   * per config and pair, Brier and ECE by budget and the mean B0 prediction;
   * per config, the call count, mean and max call time, and fallback counts;
   * the EB traces;
   * resident memory and the library digests.

   It refuses to start unless `lock`, `smcal`, `regimes` and `reproduce` passed.
   The harness is `level_calibration.checkpoints` and `evaluate`: the
   platform's slots and random acquisition, a fresh instance per checkpoint, one
   worker, deduplicated inputs, verified against `official.run_official` by
   `level_calibration.py --verify`.
6. `summarise`: packs the rows into `results/regime_sensitivity.json`, computes
   section 7, runs the TUNED consistency check and applies the rule.
   * **TUNED consistency check.** `ship` minus `legacy` must lie within 3
     combined SEs of the known -0.0415 (`ship_confirm.json`, test-like runs
     0-299, cluster SE 0.0034). The shipped rows there are old-floor; the
     measured code gap is at most 0.0003, which this ignores.
   * **The rule.** Writes a table of every condition for every config, the
     candidate list and the outcome.

   A failed check marks the outcome "rule not applied: check failed" instead of
   reading the rule.

## 10 Library change needed (not made in this step)

The mixture needs a `paiec/testlike.py` option:

* `Regime.level_mix: tuple = ()` of (weight, mean, sd) triples. When it is
  non-empty, the tilt's target density is the mixture: per component, the
  benchmark tilt uses sd sqrt(max(sd^2 - within^2, min_level_sd^2)) and the pair
  tilt uses sd itself. level_mean and level_sd are ignored. Weights must be
  positive and sum to 1.
* Tests: `level_mix=()` leaves `Sampler.omega` bit-identical for the default and
  two other knob settings; a one-component mixture equals the Gaussian tilt to
  1e-12; bad weights raise `ValueError`.

The probe's out-of-library reimplementation (it recomputes the benchmark tilt
from the untilted sampler) reproduces `Sampler`'s own omega to 7e-18 with one
component, and serves as the reference. `pytest` must stay green.

## 11 What this cannot settle

* Every regime is cut from the same five public benchmarks (four parents carry
  the test-like ones). A ranking that holds in all five test-like regimes still
  rests on those four. The parent-level SE is the honest yardstick.
* READING and AUDIT come from 17 pairs on 7 benchmark ids (level mean SE about
  0.5). They bracket that reading's centre but not its uncertainty.
* The date shift is in every test-like regime.
* MIXTURE's main component is wider than the reading's (1.68 against 1.15).
* The rule detects gains of 0.003 or more and is a coin flip at 0.002
  (section 7).
* The hier rows here are at the current library (corrected floor, floored-fit
  fix); `ship_confirm.json`'s shipped rows are old-floor. Comparisons with
  earlier tables carry that gap (at most 0.0005).

## 12 How this plan was made (provenance)

Scratch probes were run under
`/private/tmp/claude-501/-Users-nikitapolomosnov-PycharmProjects-PAIEC/9df8ecfa-42c0-4508-919c-f794f1770909/scratchpad/p1a/`.
None touched seed 11.

* `timing.py` and `mem_probe.py` scored all 11 configs on old runs (test-like
  seed 2 runs 0-5, public seed 0 runs 0-1 of each weighting) for time and
  memory, and compared `ship`, `legacy` and `smooth` with stored rows. They also
  printed each config's ALC on those runs, and those ALCs were visible while
  this plan was written. They come from 10 runs already used to choose LEVEL,
  far too few to separate configs. Nothing was set from them: the configs and
  the thresholds of (a) to (d) are the task's, the run counts come from the
  timings, and the additions of section 8 apply to every config alike and name
  none.
* `realise.py`, `realise2.py`, `realise3.py`, `realise4.py` and `draw_se.py`
  drew runs on seed 10 only, with no predictor, to set the regime knobs and the
  realisation bounds.

Scratch is not durable. The script's `regimes`, `reproduce` and `smcal` stages
re-measure everything this plan quotes from the probes.

Input digests (sha256, first 16 hex) at the time of fixing:

* `paiec/hier.py` d9a9561202b63b9d
* `paiec/prior.py` 3a8a8c7dcd0da53e
* `paiec/predict.py` 46193d8dd7701ec2
* `paiec/official.py` c70684dc15b560bc
* `paiec/testlike.py` b3ad531e5a218469 (before the `level_mix` option)
* `paiec/baselines.py` 64fe42f5a63d22e1
* `experiments/level_calibration.py` 0d810dddfde74741
* `submission/model.py` a9561290e62d3f7c
* `results/formative_feedback.json` 9154d93b9148d193

## Amendments after review

Appended on 2026-10-01, after the study was scored and summarised and a verifier
reviewed it. Nothing above this heading changed. The lock now checks the text before
this heading against `PLAN_SHA256`, byte for byte (`lock_checks`;
`review.provenance.plan`). The rule, configs, regimes, seeds, run counts and statistics
are as fixed. No row was re-scored, no number of the summary changed, and the outcome
stands: **no candidate: SHIP stays** (`rule.outcome`).

Every number below is in `results/regime_sensitivity.json` under the key named. The
`review` stage of `experiments/regime_sensitivity.py` writes them; its commands are in
the script's header.

**Edits to the script after scoring.** Every edit made after the rows were scored sits
in a block marked "after review". Cutting the blocks out gives back, byte for byte, the
script that scored every row (sha256 `0bef6bf07d23bd87...`; `review.provenance.script`,
checked by `tests/test_regime_sensitivity.py`). The blocks add the `review` stage and let
the lock read only the fixed part of this file. The code that wrote the stages, the rows
and the summary is unchanged.

### A1 D1 and the two readings (review V1, V7)

D1 changed how a locked check is applied. The legacy reproduction is compared at the
six decimals `experiments/testlike_check.py` stores, not at `REPRO_TOL['legacy']` = 1e-9
(`DEVIATIONS`). What `review.provenance.d1` records:

* **Timing.** The first reproduce run failed at 1e-9; its log ends at 15:40:07Z. The run
  with the fix passed at 16:26:39Z. The first scoring-seed row was written at 16:44:02Z.
  So D1 was made, and its check passed, before any run of seed 11 was scored.
* **Size.** The stored differences were 4.7e-7 to 5.0e-7 on the five runs. All are
  within half a unit of the sixth decimal, and every stored value equals this study's
  value rounded to six decimals.
* **Reach.** No row depends on the check: the score stage reads only the reproduce
  stage's ok flag.
* **The two readings.** With D1 the outcome is "no candidate: SHIP stays". Read
  literally, it is "rule not applied: check failed" (`rule.literal_reading`). D1 can
  touch only condition (f). No config meets (a), so none meets (a) to (e), and neither
  reading has a candidate.

**Left to the team.** Accept D1, or record P1a as "not applied (literal)"; the
conclusion is the same either way. Future plans should set reproduction tolerances at
the comparator's stored precision.

### A2 What the lock rests on (V2)

Nothing was committed before scoring, so the lock's time stamp is self-reported. The
evidence for it is local: file mtimes and scratch copies (`review.provenance.evidence`;
every one of its checks holds). All times are UTC on 2026-10-01.

| time | event |
|---|---|
| 14:50:26 | the planner's rule text (`rule.txt`, sha256 = `RULE_SHA256`) last written |
| 14:56:59 | `FIXED_AT_UTC`; the plan file last written (a copy kept with its mtime; sha256 = `PLAN_SHA256`) |
| 15:11:26 | the implementer's copy of the planner's file; its constants block hashes to `CONSTANTS_SHA256` |
| 15:26:54 | seed 11 first drawn, by the regimes stage (run composition only, no predictor) |
| 15:40:07 | the first reproduce run ends; D1 found |
| 16:26:39 | reproduce with D1's fix passes |
| 16:29:09 | the script last written before scoring (a copy kept with its mtime; sha256 = the scored digest) |
| 16:44:02 | the first scoring-seed row |
| 16:56:16 | the dry run of summarise on 19 rows (A4) |
| 19:15:28 | the last of the 520 rows |
| 19:16:01 | summarise; the rule applied |

**Left to the team.** Commit the plan, the script, the tests and the results together.
The findings should say that the lock is evidenced by local mtimes and scratch copies,
not by a commit made before scoring.

### A3 Stage provenance (V8)

smcal and regimes ran under script `dbbf63b6`, reproduce under `316f8e91`, and all 520
rows and the summary under `0bef6bf0` (`review.provenance.stages`, `.rows`). The script
was edited between stages, never during scoring.

`review.rerun` re-ran smcal, regimes (without `--grid`) and reproduce under the final
script, whose stage code is the scored script's, into a scratch directory. Every
deterministic field is identical: 366, 205 and 150 numbers, largest difference 0.0. The
composition file's digest is identical too. Left out of the comparison: provenance, wall
times, call timings and the time estimate built from them, and the knob grid, which only
`--grid` runs.

### A4 The partial-set dry run (V5)

During scoring, a dry run of summarise (scratch `impl/dry_summarise.py`) ran on the 19
rows that existed at 16:56:16Z, into a scratch copy of the results. It printed the
structure, the checks and the outcome, "rule not applied: the planned set is not
complete". The statistics it computed stayed in that scratch file.

The dry run, every row and the final summary carry the single script digest `0bef6bf0`
(`review.provenance.dry_run`, `.rows`). So no code, constant or run count changed after
it.

### A5 Fresh seeds are not fresh data (V3)

The seed-11 runs are new draws, but from the same catalogue and the same 221 public
pairs that chose LEVEL (and aggr, level_calibration's winner) and guarded it. TUNED and
the guards (c) and (d) therefore carry that selection's optimism, toward SHIP or aggr.
They are not independent replication. aggr's lead in TUNED (D -0.0034,
`review.headline.D`) is consistent with this.

### A6 Realised levels of the scored runs (V4)

Section 4 states targets, and the scored runs realised something milder. The findings
quote the realised levels (`review.levels`: seed 11, every pair appearance, unweighted),
not the targets:

| regime | target | seed 10: mean / sd | seed 11 (scored): mean / sd | share below -3 | share below -4 |
|---|---|---|---|---|---|
| TUNED | -1.290 / 1.696 | -1.273 / 1.731 | -1.276 / 1.621 | 0.121 | 0.028 |
| READING | -0.715 / 1.772 | -0.729 / 1.793 | -0.627 / 1.678 | 0.076 | 0.015 |
| AUDIT | -1.10 / 2.00 | -1.102 / 1.997 | -0.930 / 2.054 | 0.136 | 0.060 |
| MIXTURE | share below -3 0.118; rest -0.226 | -0.705 / 2.062 | -0.642 / 2.126 (rest -0.201) | 0.110 | 0.057 |
| FLAT | none | -0.338 / 2.287 | -0.280 / 2.225 | 0.098 | 0.031 |
| R1B | none | | -0.556 / 1.346 | 0.055 | 0.019 |
| R1P | none | | -0.767 / 1.465 | 0.080 | 0.025 |

* **The feedback regimes span -0.63 to -0.93**, not section 4's "-0.7 to -1.1". The
  shifts from seed 10 lie inside the realisation bounds fixed before scoring (±0.30), so
  no check failed.
* **MIXTURE's low mode exists.** 36 of its 627 appearances (0.057) sit below -4, against
  0.015 in READING. 26 of the 36 (0.72) come from two multi_swebench pseudo-benchmarks,
  `q1of3` and `q1of2`, and 8 more from `real_webagents::all`.
* **As a stress test it is weaker than its name.** Its share below -3 (0.110) is under
  TUNED's (0.121) and AUDIT's (0.136), and AUDIT's share below -4 (0.060) is as large.

### A7 How close the configs are (V6)

`review.headline` takes X minus SHIP from `summary.vs_ship[R][X].exact`. D (cluster SE)
in READING and in AUDIT:

| config | READING | AUDIT |
|---|---|---|
| wide35 | +0.0006 (0.0003) | -0.0003 (0.0004) |
| eb_adapt | +0.0003 (0.0004) | -0.0005 (0.0007) |
| eb_ship | +0.0003 (0.0003) | -0.0002 (0.0004) |
| aggr | +0.0015 (0.0011) | +0.0005 (0.0014) |
| wide50 | +0.0023 (0.0007) | +0.0006 (0.0008) |
| onepl | +0.0026 (0.0014) | +0.0016 (0.0017) |
| eb_fit | +0.0173 (0.0032) | +0.0233 (0.0042) |

Only wide35, eb_adapt and eb_ship sit within ±0.0006 of SHIP in both regimes, with
cluster SEs of 0.0003 to 0.0007. Across all the hier configs the cluster SEs run from
0.0003 to 0.0042.

So the run report's "every hier variant sits within ±0.0006 of SHIP" holds only for
those three. Its "hier-versus-hier cluster SEs of 0.0003 to 0.0008" holds only for them
and wide50 (0.0007, 0.0008); aggr's, onepl's and eb_fit's are larger. The findings scope
both sentences that way and take every other number straight from
`summary.vs_ship[R][X].exact`.

### A8 smcal's guard reference (V9)

The guard compares against the legacy Predictor's per-pair ALC stored in
`results/level_calibration.json` (committed in ee5085a, 2026-09-26). Those rows predate
the multiple-choice floor fix f7e7d87 (2026-09-27), so on matharena pairs they are not
what the current legacy Predictor predicts. The plan asked for the stored rows, and the
smcal stage's guard note did not mention this. The note is now attached to the stage as
`smcal.guard_reference_note_after_review`.

`review.smcal_guard` re-scored the current legacy Predictor on the guard's 200 runs
(public R1 seed 0, runs 0-99 of each weighting):

* **Off matharena** the two agree to the stored precision: every pair is within 5.5e-6.
* **On matharena pairs** the current Predictor is better, by 0.0023 (192 pairs,
  benchmark-first) and 0.0018 (334 pairs, pair-uniform) on average.
* **Per run**, current minus stored is -0.00048 (run SE 0.00011, benchmark-first) and
  -0.00067 (0.00010, pair-uniform).

So every guard loss grows against the current Predictor. The best grid point,
(4, 0.4), moves from +0.0023 / +0.0040 to +0.0027 / +0.0047, against a guard of 0.003.
No point passes against either reference, so smcal stays the unconstrained argmin on
the selection half, (2.0, 0.25), and is still flagged `guard_failed`. The stored guard
losses are reproduced exactly (largest difference 0.0). The stale reference cannot
change the selection.

### A9 Reproducing rows (V10)

The EB traces in a row are ordered by iterating a set of parent strings in
`Maker.__call__`. That order depends on `PYTHONHASHSEED`, so rows are not byte for byte
reproducible. `eb_traces` only aggregates them, so no number depends on the order.

`review.rescore` re-scored the last planned run of every regime (seven tasks, every
config, with `PYTHONHASHSEED` unset). Brier, ECE, mean B0 predictions, call counts and
fallback counters are identical, with largest Brier difference 0.0. Of the 14 EB traces,
none is in its stored order and all 14 are equal once sorted.

The first full pytest run (default BLAS threads, while the reproduce stage ran, then
killed) showed one F, at the 149th test (`review.provenance.first_pytest`). That
position is `tests/test_hier.py::test_formative_size_is_fast`, a 5 s wall-clock bound,
and no test file up to it has changed since. The single-thread full run passes.

**Left for a later script version.** Iterate `sorted(set(self.ids.values()))` in
`Maker.__call__`. This version cannot: that is the scoring code the rows record.

### What is left to the team

1. Accept D1, or record P1a as "not applied (literal)" (A1).
2. Commit the plan, the script, the tests and the results together (A2, A3).
3. Write the P1a section of `docs/findings.md` from A1 to A9:
   * both readings;
   * the lock's evidence;
   * the fresh-seeds sentence;
   * the realised levels;
   * the scoped closeness statements;
   * the smcal guard note.
