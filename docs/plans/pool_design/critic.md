# Adversarial review of `amendment4_draft.md`

Written 2026-10-07 against repository HEAD a725df9, read only. I read
`amendment4_draft.md`, `data.md`, `code.md`, `eval.md`, `impl_plan.md`,
`docs/plans/p2_final_plan.md`, App G.6 of `docs/report/draft.md`, and the code
the draft cites. I recomputed descriptive quantities only: catalogue fields, tilt
weights from `eval_design/feasibility.py`'s own functions, standings
correlations from `pairs.csv`, and attribute scores of the two priors. I ran no
predictor and computed no ALC or Brier. I read no formative table, no anonymous
id and no item text. I wrote nothing except this file.

"The draft" means `amendment4_draft.md`, and §4.n is its section n.

## Verdict

The design is careful and most of its plumbing checks out (see "What holds").
It should not be committed as written, for five substantive reasons:

1. The level is not separated from the rest. Archive-3's ridge carries a large
   date term, so C1 changes the effective level for any subject whose date
   differs from the reference sets. The regimes that must show gains have no
   date shift. (R1)
2. The selection/confirmation split puts the three lowest direction-flagged
   benchmarks in S. That makes S's tilted regimes mostly about
   `lower_is_better` benchmarks, while C's are not. (R2)
3. Exclusion works per benchmark, but the pool holds families of sibling
   benchmarks from a single source. The competition's "assigned before
   curation" suggests a hidden benchmark has no such siblings. (R3)
4. C3 was motivated by an identity-transfer statistic computed on C's
   benchmarks. (R4)
5. The lock does not pin the code and data that produce the candidates, so the
   shipped archive is not bound to the configuration that was evaluated. (R5)

The other required changes are cheaper: R6 to R12 fix text, harness, rules
statements and ambiguities.

None of these needs a predictor score to fix. All can be made before the lock.

## Required changes

### R1. The date term moves the level, so C1 is not "the rest at archive-3's level"

**Evidence.**

* **The two ridges' date terms.** The shipped prior's date coefficient is 2.414
  per 400 days (`submission/prior.json`, `coef[1]`). The C1-shaped prototype's
  is 0.285 (`code/c1_proto_prior.json`). Both use the linear date column
  `(days - 400) / 400`, with no clip (`paiec/subjects.py:240-241`). Theta's mean
  adds `attr_scale × score` (`paiec/hier.py:1902-1903`).
* **The effect of a date shift.** I took the dated pairs of `pairs.csv`, 113
  benchmarks with each benchmark counted once, and shifted their
  `release_date` by 456 days. That is RS's `date_shift` of 1.25 years
  (`paiec/testlike.py:517`; RS keeps it in every test-like regime,
  `experiments/regime_sensitivity.py:235-236`).
  * Under archive-3's ridge, the mean score moves from -1.620 to +1.132, and
    theta's prior mean rises by 1.38 at `attr_scale` 0.5.
  * Under the prototype pool ridge, it moves from -0.064 to +0.261, so theta's
    mean rises by 0.16.
* **The B0 centre.** I approximated it as code.md §4 does, as
  E[sigmoid(-2.5 + 0.5 m + sqrt(V) z)] with V = 21.4 for archive-3 and 16.8
  for C1.
  * At real dates: archive-3 0.252 and C1 0.286, a difference of +0.033.
  * At +1.25 years: archive-3 0.348 and C1 0.298, a difference of -0.050.
  * §4.4's "within 0.024 on every reference set" therefore holds only on code.md's
    reference sets at their real dates (undated subjects imputed at the
    median). On dated subjects it fails in both directions, and its sign flips
    with the subjects' dates.
* **What LEVEL was calibrated on.** LEVEL and the RS targets were set on
  date-shifted test-like runs. The draft's pool regimes keep real dates (§4.5,
  "with no date shift"), and the gains of gate (a) are required only there.
  Pool subjects are mostly from 2024 (data.md §3). The formative evidence for
  the shift (App G.6, run 1) points to hidden subjects newer than the legacy
  ones, which are newer than most pool subjects.

**Required.**

1. **Before the lock**, extend code.md §4's B0-centre table and
   `eval_design/anchor.py`:
   * add the evaluation subjects at real dates and at +1.25 years;
   * cover A3, C1, C2 and C3;
   * report dated and undated subjects separately;
   * record the result in `results/pool_fits.json`.
2. **Qualify the claims.** §4.4 ("C1 keeps archive-3's level centre"), the
   contrast table and §4.12 must say that the claim holds at real dates. Under a
   date shift, C1 is also a level change.
3. **Add two date-shifted pool regimes**, P-READING-DS and P-AUDIT-DS:
   * the same planned compositions, with `testlike.shifted(p, 1.25, None)`
     applied to every pair;
   * they enter (E2) and gate condition (c) as loss bounds, D ≤ +0.002;
   * the cost is about 1.1 CPU-hours in S and 0.8 in C at the planning unit
     costs.

   Alternatively, write down why the RS guard alone suffices. It is the only
   place where date-shifted subjects meet the candidates, and (E1) bounds only
   a mean of +0.0005.
4. **Pre-commit what happens if item 1 shows** C1's centre at +1.25 years
   differs from archive-3's by more than the 0.024 tolerance. Either:
   * add a candidate C1' whose mu0 is re-matched to archive-3's B0 centre by a
     fixed rule (§4.4's "Not candidates" table rejects this only on the
     real-date tolerance); or
   * state that C1 bets on hidden subjects being no newer than the pool's.

### R2. The split puts the three lowest flagged benchmarks in S

**Evidence.** The binary lowest block of the split rule (`feasibility.py:58-66`)
holds eight benchmarks:

| benchmark | level | half | flagged |
|---|---|---|---|
| llmail_inject | -5.42 | S | yes |
| bird_sql | -4.27 | C | |
| crow | -3.37 | S | yes |
| monkey_power_laws | -3.25 | C | |
| wmt_mqm | -3.10 | S | yes |
| risebench | | C | |
| agentic_review_perturbation | | C | |
| disco | | C | |

**Tilt weights at feasibility.json's knobs.** I used the same `weights()`
function. "Flagged" is the share of the tilt weight on the nine flagged
benchmarks; Kish n is 1/Σw².

| regime | flagged share, S | flagged share, C | Kish n, S | Kish n, C |
|---|---|---|---|---|
| P-READING | 0.220 | 0.048 | 35.1 | 54.7 |
| P-AUDIT | 0.333 | 0.046 | 22.6 | 42.6 |
| P-TUNED | 0.376 | 0.055 | 14.9 | 32.4 |

* wmt_mqm alone holds 12.5% of S's P-AUDIT weight.
* U leaves the flagged benchmarks out of the subject part because their sign
  is reversed or absent (§4.3). So M(X), which is read on S, is driven by
  benchmarks where the subject prior points the wrong way. C's gain regimes
  are driven by bird_sql, monkey_power_laws, risebench and similar
  benchmarks.
* Selection and confirmation therefore measure different things. The split
  biases selection against the candidates that lean hardest on subject
  standings (C3, then C1).

**Required.**

* **Re-stratify the split** by (response kind, direction flag) before the
  lock. Levels and flags are outcome-blind.
* **One workable rule:** flagged benchmarks in level blocks of 3, with 1 to S
  by the same sha256 rank; the others as now. On my run of it:
  * S has 49 benchmarks and C has 83;
  * the flagged share of P-AUDIT's weight is 0.17 in S and 0.14 in C;
  * Kish n is 24.9 and 45.8.
* **Recompute** `feasibility.json` and the knob search.
* **Lock a balance check.** Per tilted regime, the flagged weight must differ
  by at most 0.10 between S and C. Report the largest single-benchmark weight
  too.
* **Disclose** that the split was revised on levels and flags only.

### R3. Sibling benchmarks from one source leak standings past the benchmark-level exclusion

**Evidence.** Benchmarks of L share a catalogue `source_url` or `paper_url`
(`data/pool/<b>/formatted_tables/benchmarks.parquet`):

* medhelm_* (10; `crfm.stanford.edu/helm/medhelm`, all one paper,
  arXiv 2505.23802);
* VLMEval: ai2d_test, hallusionbench, mathvista_mini and mmbench_v11;
* open-llm-leaderboard-old: mmlu and truthfulqa_mc;
* McGill-NLP: visualwebarena and workarena;
* ScalingIntelligence: kernelbench and monkey_power_laws;
* kmmlu and kormedmcqa;
* bridging_gap and helm_afr (one paper);
* the helm safety suite.

How strongly siblings' standings agree, from `pairs.csv`, over shared
canonical names:

| benchmark pair | shared names | correlation |
|---|---|---|
| medhelm_med_qa and medhelm_medbullets | 13 | 0.967 |
| medhelm_med_mcqa and medhelm_med_qa | 13 | 0.962 |

The competition says benchmarks were assigned to pools "before curation". If
the unit of assignment was the source, a hidden benchmark has no public
siblings, while a held-out public one keeps its siblings in training under
§4.3's B_run exclusion. That overstates identity transfer for the candidates
and not for A3, whose prior has no pool cells. C3 gains the most from it, then
C1. monkey_power_laws, one of the heaviest benchmarks in C's gain regimes, has
a sibling.

**Required.**

* **Pre-register a family rule.** A family is a connected component over the
  normalized `source_url` (host plus the first two path segments) and the
  `paper_url`.
* **Exclude whole families.** Build each run's candidates without the whole
  family of every benchmark of B_run, and each RS task's without the parent's
  family. The cost is unchanged: one `from_standings` per run.
* **Use family exclusion in the gate.** B_run-only exclusion becomes
  descriptive.
* **If the author rejects this,** say why in §4.13, and report D separately on
  appearances whose benchmark has a family member in training.

### R4. C3 was designed on C's identity statistics

**Evidence.** code.md §5's slope table regresses a name's residual cell on a
held-out benchmark on its other cells. It measures out-of-benchmark transfer of
the identity residual, the mechanism C3 changes, and it used every benchmark,
C's included. §4.13's Disclosures, item 1, admits this. It is the one
pre-registration step that saw outcome-like data on the confirmation half.

**Required.**

* **Recompute the slope table before the lock:**
  * held-out cells on S benchmarks only;
  * "other cells" on S plus the legacy five only;
  * no C cell anywhere;
  * pin the result in the lock.
* **Keep C3 only if** its uncapped weight lies at or below the S-only slope for
  one and for three or more other cells, code.md §5's own criterion.
* **If C3 is carried,** the report must say that C's confirmation is not blind
  to the identity statistics.

### R5. Bind the archive to the evaluated configuration, and complete the lock

**Evidence.** §4.5(10) locks:

* `RULE_TEXT`;
* the amendment's text;
* the constants block;
* the shipped modules;
* `a3_prior.json`;
* the fits cache index;
* the compositions.

It does not lock:

* `paiec/pool.py`, which builds every candidate;
* `experiments/pool_eval.py`;
* the harness modules that the stored RS rows also pin (row `lib`, e.g.
  `data/regime_sensitivity_rows/TUNED/0.json`):
  * `paiec/official.py` c70684dc15b560bc;
  * `paiec/testlike.py` 00e96c4c1b5ee5fd;
  * `experiments/level_calibration.py` 0d810dddfde74741;
  * `paiec/data.py`;
  * `paiec/evaluator.py`;
* `data/pool/manifest.json` (a58a23fb…).

The `--pool` build is written after the lock (§4.11; `impl_plan.md` §11). C2's
LEVEL literal is computed at build time.

**Required.**

* **Add the missing digests to the lock.** Have the loader verify each pool
  file's sha256 against the manifest.
* **Write the excl-empty bundles at the `fits` stage.** That means C1, C3, and
  C2 with its LEVEL rounded to 3 decimals. Record their sha256 and the C2
  literal in the lock.
* **Make `--pool` reproduce the bundle exactly.** It must reproduce X*'s locked
  bundle byte for byte from the locked `pool.py` and fits cache, and refuse
  otherwise.

### R6. model.py's text would be false under C2, and imprecise under all three

**Evidence.**

* §4.2 keeps model.py's docstring and comments.
* Under C2, the shipped file would say "level prior moved down for the hidden
  test: mu0 -2.5, sigma_mu 2.5" (`submission/model.py:3-4`) and would call
  LEVEL "the milder of the two guarded configurations" (`:64-67`), beside a
  pool EB literal.
* Under every candidate, the file says the prior is "fitted on every eligible
  public pair" (`:5-6`). The pool build is capped at K = 100 pairs and 2,000
  items a pair, and is items-blind.
* code.md §9 planned these edits (`:3-8`, `:62-67`).

**Required.**

* Allow docstring and comment edits in model.py.
* Check them with an AST comparison that ignores docstrings, comments and the
  LEVEL value; this becomes the fourth `--pool` check.
* Write text that is true for the configuration that ships. Runtime bytes do
  not change.

### R7. The named harness does not measure CPU time per call

**Evidence.** (E3) and (f) read CPU time per call. `level_calibration.evaluate`
times calls with `time.perf_counter` alone (`experiments/level_calibration.py:540-546`).

**Required.**

* Name the harness as a copy that records `time.process_time` and wall time per
  call.
* Point RP3 at that copy.
* Define the "process-hours" of the adjustment rule (§4.5) as CPU seconds.

### R8. The feedback statements understate the new use

**Evidence.**

* §4.9 presents items 1 and 2 as uses that App G.6 already lists. But G.6
  lists the RS targets in a study that "changed nothing"
  (`docs/report/draft.md:1677`).
* The report's limit reads: the feedback "may set global hyperparameters of the
  level distribution, here three numbers" (`draft.md:305`).
* Under the draft, the targets define the regimes whose gains decide whether a
  refitted subject prior, item variance and identity guards ship. That is
  global and id-free, but it selects more than three numbers.
* G.6's "Training data" names "the revision of App G.2" (`draft.md:1671`).

**Required.**

* List items 1 and 2 as reused for a ship decision, which is a new use.
* Amend `draft.md:305` to match.
* Update G.2 and G.6 to name revision 54074453, which this evaluation trains
  on whether or not archive-4 ships.

### R9. Supersede the plan text the draft collides with, and add a lock deadline

**Evidence.** The draft collides with three passages of
`docs/plans/p2_final_plan.md`:

* amendment 1: "No archive-4 and no formative submission follow" (`:160-161`);
* §3: "Anything not ready by 2026-10-17 does not ship" (`:66`). The draft
  replaces only "chosen on 2026-10-17";
* §3's definition of archive-4 as "archive-3 plus the S1 or S3 fix"
  (`:63-64`).

§4.11's hard stops omit the lock, which `impl_plan.md` §8 dates to "no later
than 2026-10-09 23:59 UTC by A4 §4.11".

**Required.**

* Supersede these three passages explicitly.
* Add a hard stop for the lock, for example 2026-10-10 23:59 UTC; if the lock
  misses it, archive-3 stays.

### R10. The tie-break and the carry threshold can conflict

**Evidence.** If the argmin has M = -0.0013 and an earlier candidate has
M = -0.0009, the tie-break takes the earlier candidate. "X* carried iff
M(X*) ≤ -0.001" then fails, although the argmin qualified.

**Required.** Apply the tie-break only among eligible candidates with
M ≤ -0.001, or state the stated outcome as intended.

### R11. Correct the seed claim

**Evidence.** §4.5 says that none of 4001 to 4004 has been used. But
`eval_design/feasibility.py:189` draws its planning runs with
`default_rng([4001, k])`, and that script will be committed under
`docs/plans/pool_design/`. This is harmless: CAL_SEED is never scored, and
`[seed, half, regime, i]` streams differ from the two-element ones.

**Required.** Say so.

### R12. Mode B: list the item sizes, as eval.md required, and add an MCQ cut

**Evidence.**

* eval.md §3.4 required the item files' sizes to be listed from repository
  metadata before choosing a mode. The draft chose Mode B on the disk figure
  alone.
* At least 21 of E's 132 benchmarks (6 in S, 15 in C, all with level above -1)
  describe themselves in the catalogue's `one_line_description` or
  `description` as multiple-choice.
* On those benchmarks the floor (`paiec/hier.py:1944-1953`; `Flags.floor`
  True, `:438`) will act at test time for every arm, but never on pool runs.
* These benchmarks weigh most in the FLAT loss bounds.

**Required.**

* List the sizes without downloading and record the decision.
* Pre-register a descriptive cut by a catalogue MCQ flag (a regex on benchmark
  metadata, not item text), with D reported without those benchmarks in every
  regime.

## Recommended, not blocking

* **Power.** Use Kish n, not exp(entropy), in the power table. In C, P-READING
  has 54.7 against 65 and P-AUDIT 42.6 against 56, so SE_b runs about 9 to 15%
  above plan.
* **C's P-MIXTURE.** bird_sql holds 20.7% of the tilt weight (Kish n 19.3), so
  condition (c) there is close to a bird_sql-only bound. Consider capping any
  benchmark's appearance share, for example at 10%, or say so in the rule.
* **C2's v_b.** `mu_var` is Var(mu), not Var(mu + mean t). The EB s² is
  slightly understated (`paiec/prior.py:335` returns no covariance). Either
  state the approximation or bound it.
* **The timing stage.** Say which half it draws from; eval.md says S.
* **The disk figure.** The draft says 14 GB free and eval.md 13 GB.
* **bird_sql's provider coefficient.** It sets a provider coefficient of 3.98
  (defog) under pair weighting (code.md §3.3). Report the largest attribute
  shifts any single benchmark induces.
* **Wall time.** The machine's load average was 9 to 95 (code.md, preamble),
  against the 1.5x factor the plan assumes. Before the lock, pre-commit which
  planned runs are dropped if the lock slips, for example S's dense regimes to
  20 runs. Never do this after it.

## What holds (checked)

* **Digests.** The shipped modules' digests at HEAD equal §4.2's table and the
  stored RS rows' `lib` digests (`TUNED/0.json`).
* **Exclusion of the run's own benchmark.** It is complete apart from R3 and
  C2's T_full:
  * `from_standings` does not filter rows itself (`paiec/prior.py:557-667`), so
    rows must come from U ∖ excl, as PB says;
  * per-benchmark fits do not depend on excl (`:229-341`);
  * T, the ridge, the Spec medians and the identity cells all come from
    L ∖ excl or U ∖ excl;
  * `HierPredictor` refuses mismatched included and excluded lists
    (`paiec/hier.py:1847-1861`).
* **No legacy parent has a twin that shares its subjects.**
  matharena_platform shares 0 canonical names with matharena.
* **The comparator.** A3 is exactly archive-3's runtime: `dist/paiec.zip` is
  73,374 bytes and `results/stress_archive.json` records sha256 `4a882cc7…`.
  RP1 and RP2 are the right checks.
* **The withheld three** (imagenet_hard, swe_together, visual_memory) are
  reproducibility holds, not test-pool entries
  (`data/p2_work/mdb_new/README.md:212-224`). Only their catalogue rows are
  local, and the release rule excludes them.
* **Platform runtime is no risk.**
  * A C1-shaped prior.json is 84 KB.
  * The validator passes and model.py loads "hier".
  * model.py executes in 33 ms (median).
  * S3 measured archive-3's dense `predict_cpu` at p99 0.048 s and maximum
    0.35 s, well inside (f).
  * 16 workers recreated at each of 6 checkpoints add under 0.2 s CPU a
    worker.
* **Line references.** The code references I checked are right (±2 lines):
  * `prior.py:229, 250-260, 355-367, 756-807`;
  * `hier.py:1910, 1918-1927, 1944-1953`;
  * `official.py:66-80, 83-100, 310-319, 348-359`;
  * `regime_sensitivity.py:234, 250, 267, 289, 652`;
  * `level_calibration.py:508-561, 741-768`;
  * `evaluator.py:49`.
* **The gate under no effect.** Its false-pass rate, about 0.01 for (a) in both
  feedback regimes, follows from its assumptions, once the Kish correction
  above is applied.
* **Feasibility.** The plan is feasible by 2026-10-16 if the lock lands by
  2026-10-10.
  * The binding constraint is the roughly 14 hours of code and tests in
    `impl_plan.md` §§2-6, not compute.
  * Compute is about 21 process-hours with R1's regimes, about 7 to 8 hours of
    wall time at 1.5x, and 2 to 3 times that under the observed loads.
  * The gate deadline of 2026-10-13 leaves about two days of slack.
  * Whether run 4's table arrives by 2026-10-20 is outside the team's control.
    The fallback covers it.
