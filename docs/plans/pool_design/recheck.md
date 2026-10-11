# Adversarial re-check of amendment 4 after the calendar revision

Written 2026-10-11, 18:10 to 19:00 UTC (Moscow evening), against repository HEAD
2bfcd17, read only. Checked: `amendment4_short.md` ("A4"), `p2_pool_eval.md`
("PE"), `impl_plan.md` and `dates_log.md` against `critic.md` (R1 to R12), with
`docs/plans/p2_final_plan.md`, App G.6 and line 305 of `docs/report/draft.md`, the
`a4final/*.json` outputs and the code the files cite.

What I ran, all descriptive:

* sha256 of the design files, the pinned modules and the manifests;
* reads of `split_v2.json`, `power_v2.json`, `b0_dates.json`, `slopes_s.json` and
  `families.json`;
* one recomputation of the dense regimes' Kish numbers with `split_v2.py`'s own
  functions;
* parquet *metadata* (row counts, row-group sizes) of the largest item tables;
* the mean pair rates the RS targets imply, which is arithmetic on design
  constants.

What I did not do: run a predictor, compute an ALC or Brier, read item text or
read a formative table. I could not re-hash `dist/paiec.zip`, because a plugin
hook blocks `dist/`; I used `results/stress_archive.json` instead (see the end).
I wrote nothing except this file.

## Verdict

**Not ready to commit as is.** R1 to R12 are answered, most of them in full, and
the quoted digest of PE is correct. The arithmetic of the calendar holds. What is
left is cheap: text, rule precision and lock completeness. None of it needs a
predictor score, and all of it can be done before commit 1 on 2026-10-12. The one
item of substance is N1: C1m's gains are read at real dates, where its B0 centre
sits +0.045 above archive-3's. A4 claims to separate the level from the date
term, and for C1m the gate does not do that. The author has to choose between a
rule change and a corrected claim.

Any edit to PE changes its sha256. A4 then needs the new digest in full, and
impl_plan step 0.1 its first 16 hex.

## Digests

| file | computed sha256 | quoted | verdict |
|---|---|---|---|
| p2_pool_eval.md | `3e0800974e9f580823a0bc56a97c80e3b7dd769959d4b3de5d1fcceceb14f3c5` | A4 preamble, in full; impl_plan 0.1, first 16 hex | correct |
| amendment4_short.md | `c655e4828a0bae8252ea4f8f4a3cabb08f26870ac657366452d3a20340558e3e` | dates_log | correct |
| impl_plan.md | `1a95760d177e0ae42b5ac10c189e0fe72262faade6be804c7c123c384d8cac00` | dates_log | correct |
| data/pool/manifest.json | `ff5482069780ea78…` | PE 3.1, impl_plan 0.1 | correct |
| pool_manifest_no_items.json | `a58a23fb78642c14…` | PE 3.1 | correct |

PE quotes no digest of A4 or impl_plan, so the chain has no cycle.

PE §2 and §9 item 10 pin these files, and each equals HEAD (first 16 hex):

* the eight shipped modules;
* `submission/model.py` (a9561290e62d3f7c);
* `official.py`, `testlike.py`, `level_calibration.py`, `regime_sensitivity.py`
  (af80ed50bc02a92d), `data.py` and `evaluator.py`;
* `_compositions.json` (c1520f4be78e37dc);
* `submission/prior.json` (c7ce3b845d584f95, archive-3's).

## R1 to R12: are they really fixed?

| | status | residual |
|---|---|---|
| R1 date term | **fixed as the critic asked**: the B0 table at `fits` (real and +1.25 y, dated and undated, every arm), the claims qualified, the twins as loss bounds in (E2) and (c), C1m pre-committed | **N1**: C1m's gain regimes carry a +0.045 centre shift |
| R2 split | fixed: flagged units in blocks of 3, family units, balance check locked (planning: largest difference 0.027) | minor: the 0.10 cap is applied before the dense sampler drops the benchmarks with fewer than 4 pairs (N9) |
| R3 families | fixed: the critic's rule, whole-family exclusion, family clusters, family-whole split; families.json gives 230 families, 8 with two or more members in L (26 benchmarks) | recommended: HELM suites are separate families (below) |
| R4 C3 | fixed: `slopes_s.json` matches PE 4.4 (k = 1: 0.61 against 0.43; k ≥ 3: 0.757 against 0.804); dropped by the stated criterion | none |
| R5 lock | mostly fixed: pool.py, the harness modules, the manifest, the excl-empty bundles and a byte-for-byte `--pool` | **N3**: three files are unpinned; **N2**: A4's own lock sentence is narrower than PE 9.10 |
| R6 model.py | fixed (A4 4.2, AST check) | minor: PE §2 says "docstring (lines 1-8)", but the docstring runs to line 40, and lines 13-14 and 62-64 describe the build writing LEVEL in (N9) |
| R7 CPU time | fixed (`evaluate_timed`, process_time, process-hours defined) | none |
| R8 feedback | partly fixed | **N4**: under C1m the date shift and A3's LEVEL *set* mu0, and the report text says they "set no parameter of it" |
| R9 plan collisions | fixed: §1 decision 1, §3 (definition and dates), §4's date and amendment 1 superseded; the quotes match p2_final_plan.md lines 62-66 and 71-72; lock hard stop added | none |
| R10 tie-break | fixed: it acts within Q only | minor: "a member ... which comes earlier" is ambiguous when two earlier members qualify (N7) |
| R11 seeds | fixed: only `feasibility.py:189` and `split_v2.py:48` use 4001, on two-element streams; 4002 to 4004 appear nowhere in code | none |
| R12 items | fixed: sizes from hub metadata, Mode A, the catalogue-MCQ cut and the floor-share cut | none |

## Required changes

### N1. C1m's gains are read where its level is not archive-3's

**Evidence** (`a4final/b0_dates.json`):

| C1m minus A3, B0 centre | all | dated | undated |
|---|---|---|---|
| real dates (the gain regimes) | +0.045 | +0.063 | +0.024 |
| +1.25 years (the twins) | 0.000 | −0.012 | +0.024 |

* A3's own centre moves from 0.275 to 0.330 under the shift.
* The RS targets imply mean pair rates of about 0.386 (READING), 0.338 (AUDIT)
  and 0.296 (TUNED). These are E over N(m, s²) of the logistic; they are design
  constants, not scores.
* At real dates A3 (0.275), C1 (0.284) and C1m (0.320) all sit below the
  targets' rate on E's subjects; C2 (0.509) sits above. A higher real-date
  centre among the first three therefore moves B0 towards the targets' rate
  whatever the rest does.
* PE 4.3's contrast table says "C1m − A3 reads the pool's rest at archive-3's
  level centre at +1.25 years". The gate never reads gains there: the twins
  carry only the +0.002 loss bound.
* So M(C1m) and gate (a) for C1m measure the rest plus a level shift that exists
  only at real dates. C1m could pass on that shift while its rest costs up to
  +0.002 under the shift LEVEL was calibrated with. That is the dating the
  formative evidence points to (App G.6, run 1).
* A4 §4.1 says the candidates separate the level from the date term. For C1m and
  C2 the rule does not make that separation.

**Required**, before commit 1, one of:

* (a) For every candidate, add "D ≤ 0 in P-READING-DS and P-AUDIT-DS" to (E2) and
  to (c). This is the cheapest rule change: no new runs, power kept in (a).
* (b) Read (a) and (b) in both the real-date regimes and their twins: four gain
  regimes, lower power.
* (c) Keep the rule, and say in A4 §4.1, §4.3 and PE 4.3 that C1m's gains at real
  dates include its +0.045 centre shift (dated +0.063), and that the twins only
  bound it. The author then records accepting it.

### N2. A4's own description of the lock narrows PE's under the precedence clause

**Evidence.** A4's preamble says "The lock hashes this amendment's text ...,
`RULE_TEXT`, that file and the script's constants block." It also says "where the
two differ, this amendment governs". PE 9.10 hashes much more: pool.py, the
harness modules, the manifest, the fits index, the bundles and the compositions.
Read literally, the preamble drops all of that.

**Required.** Reword it to "The lock hashes everything p2_pool_eval §9 item 10
lists, among them ...".

### N3. Lock completeness

**Evidence.**

* `experiments/regime_sensitivity.py`, which `pool_eval.py` imports for `TARGETS`,
  imports `experiments.ship_confirm` and `paiec.baselines` (lines 404-406).
* `level_calibration.py:114` imports `paiec.baselines` too.
* The stored RS rows pin `paiec/baselines.py` (`lib`: 64fe42f5a63d22e1, equal
  at HEAD).
* The split's unit levels (`level_cc`) and the knob search's "all-item pair
  logits" come from `benchmarks.csv` (d88247c984edc5df) and `pairs.csv`
  (2e913b8b24c29fc8). Neither is committed (PE 13: no CSV) or pinned. The two
  agree to 9e-16.

**Required.**

* Add `paiec/baselines.py` and `experiments/ship_confirm.py` to PE 9.10.
* For the CSVs, either pin both digests in the lock, or have the stages derive
  levels and logits from the manifest-verified pool files through `paiec/pool.py`
  and check them against the CSVs.

### N4. The feedback statements under C1m (R8 residual)

**Evidence.**

* PE §11 item 3 uses the 1.25-year shift for "C1m's matching". mu0_m is solved
  against c(A3, Σ_DS), whose LEVEL the feedback set.
* The proposed `draft.md:305` sentence nonetheless says the feedback's quantities
  "set no parameter of it".
* A4 §4.6 lists the three quantities only as deciding whether a prior ships.

**Required.**

* Say that under C1m the date shift, through A3's LEVEL, also sets mu0 by the
  fixed matching rule.
* That is still a global level hyperparameter, so it is inside the
  `draft.md:305` limit. "Here three numbers" then needs a clause for C1m.

### N5. Post-lock fixes need a scope

**Evidence.** Reading 4 says "if a check fails, the cause is fixed, every
affected task is re-scored in full". `impl_plan`'s risk list counts "a bundle
fails at run time" as a check failure. Neither says three things:

* which files a post-lock fix may change;
* whether changing `paiec/pool.py`, a bundle or the constants block forces a
  re-lock, and whether that re-lock can land after the 10-13 stop;
* whether hier's own fallbacks (it catches errors and answers the no-label
  prediction) count as failures.

That is the one route left by which a candidate's code could change after rows
are seen.

**Required.** Add to PE §9, or to Reading 4:

* Fallbacks and other outputs of the shipped runtime are scored as they occur and
  counted. They are never check failures.
* A fix may touch only harness and I/O code (`experiments/pool_eval.py`, the
  loaders), and must leave every locked bundle digest and the constants block
  unchanged.
* A fix that would change `paiec/pool.py`'s bundle arithmetic, `CANDIDATES`, a
  bundle or a constant makes the outcome "archive-3 stays" for that candidate.
* Every fix is committed, with its diff, before the re-score.

### N6. RP3 must not produce a pre-lock comparison

**Evidence.**

* RP3 runs `official.run_official` for A3 and C1 on TIMING_SEED runs drawn from
  S. `run_official` scores pairs.
* RP2 and `timing` say "predictions compared with each other, never with
  labels" and "labels stripped". RP3 says neither.

**Required.**

* State that RP3 runs on label-randomised copies (acquisition still works), or
  that `run_official`'s scores are discarded and never written or printed.
* Add an assert, as `timing` has one.

### N7. RULE_TEXT precision

**Required.**

* (E1) says "each RS test-like regime", which the Terms do not define. Name them:
  RS-TUNED, -READING, -AUDIT, -MIXTURE, -FLAT.
* The tie-break should read "the earliest member of Q, in the order C1, C1m, C2,
  within 0.0005 of that minimum".
* Reading 1's "S3's gate (P2 plan section 3) holds" should add "read on all calls
  pooled, as amendment 1 read it". PE §10 says this, but RULE_TEXT leaves the
  per-part or pooled choice open on 10-16, after the gate.

### N8. Feasibility of C's item reads (memory and time)

**Evidence** (parquet metadata only):

* mind2web (in C; one subject, so one pair) holds 17,153 items with 3.13 GB of
  uncompressed `content`, in 2 row groups. The larger row group is 1.84 GB.
* swe_smith (in C; three subjects) has a 1.54 GB `verifier` column in row groups
  of up to 1.19 GB.
* Neither has 4 pairs, so neither enters a dense run. Both can enter every
  formative-size run of C, and the fits read them: mind2web's fixed-T pass for
  C2, up to M_p = 2,000 items.
* A pyarrow read filtered on item_id decompresses whole column chunks. One drawn
  mind2web pair can therefore push a scoring worker far past the 1.45 GB limit,
  and do it again on every resume. The exit-75 rule turns that into a loop or a
  stall: "not read in time".
* `timing` draws from S only, so it cannot see this.

**Required.**

* `items_of` reads with column projection (`item_id`, `content`, `item_features`
  only; never `verifier` or `grading_criterion`) and bounded batches
  (`ParquetFile.iter_batches`).
* `fits` records peak RSS and read CPU for mind2web and swe_smith.
* `timing` adds one formative-size run with a mind2web pair and one with a
  swe_smith pair, drawn from C. They are labels-stripped, so this is
  outcome-free.

### N9. Small consistency fixes

* **The dense weight cap.** The cap of 0.10 is applied before the dense sampler
  restricts itself to benchmarks with at least 4 pairs. S's D-READING then gives
  one benchmark 0.111.
  * Apply the cap after the restriction, and define the balance check's dense
    weights the same way.
  * PE 5.3's dense Kish figures (S 34.0/30.4 and 23.7/22.4; C 73.0/29.4 and
    42.6/27.5) are right for the restricted weights; I reproduced them. No saved
    output holds them, so `regimes` must write them.
* **The knob search and the targets measure different things.** The knob search
  runs on all-item pair logits, and the targets and the realisation check on
  evaluated responses. Define the CAL_SEED realisation on the same quantity as
  the scoring-run check, or the check can trip on that difference alone.
* **PE §2's docstring range.** Widen "(lines 1-8)" to the module docstring,
  lines 1-40, to agree with A4 4.2.
* **impl_plan §13.** Its heading still says "due by 2026-10-15", although the
  step starts after the gate's commit on 10-15.
* **A4's date.** "committed on 2026-10-12" and the heading date must be the real
  commit date. A4 is not hashed by PE, so fixing it costs nothing.
* **The commit-1 filter.** It must exclude `eval_describe/pool_describe.json`.
  That file holds 13,499 per-pair logits (`pair_logit`), and PE 13 allows only
  outputs "that hold no per-pair row". Alternatively, strip those lists.

## Recommended, not blocking

* **HELM suites.** The family key `crfm.stanford.edu/helm/<suite>` makes MedHELM,
  HELM safety, helm_thaiexam and helm_afr's suites separate families. They share
  one harness and one naming of subjects. Add a descriptive cut for HELM-hosted
  benchmarks beside the sibling cut.
* **The precedence clause.** A4 says "only a later dated amendment changes
  anything here". Add "except that none changes the rule, the candidates, the
  split, the regimes, the seeds, the counts or the statistics once a scored row
  exists", so that Reading 4 cannot be amended around.
* **The `--pool` build environment.** It must pin `PYTHONHASHSEED=0` and one BLAS
  thread, as the lock does. A multithreaded BLAS can change the last bits of the
  ridge, and the byte-for-byte check would then refuse the build (archive-3
  stays).
* **The loss bounds are point estimates.** At a true loss of +0.003 with a
  per-regime SE near 0.0017, D ≤ +0.002 still passes in roughly a quarter of
  draws. This was accepted as in P1a, but the report should say it.

## The calendar against the compute and the code

* **Time.** It is now 2026-10-11 18:08 UTC, 53.9 hours before the lock's hard
  stop.
* **Compute is not the constraint.** PE §12's arithmetic checks out:
  * the core is 67,050 process-seconds;
  * S and RS take 5.0 h of wall at 1.5x on 3 shards, and C with `summarise` 3.6 h;
  * at three times those, 15 h and 11 h, each inside its day;
  * the trigger sits at about 2.6x the dense cost.
* **Unpriced cases still fit.** Twice the formative-size unit cost under Mode A
  has no adjustment of its own. It still fits: S and RS 20.5 h, C 15.1 h at three
  times the 1.5x figure.
* **8 cores and 16 GB.** They allow more than 3 shards if the machine is free.
  That is slack, not a requirement. N8 is the compute risk that is not priced.
* **The code is the constraint.** About 18 h of session time
  (`impl_plan`: pool.py 4 h, tests 2 h, pool_eval.py 7.5 h, tests 2.5 h, plus the
  stages) and the author's commits must fit into 2026-10-12 and 2026-10-13. There
  is no slack for:
  * an RP1 or RP4 failure;
  * a balance-check amendment;
  * another usage-limit pause like that of 2026-10-08 to 10-11.
* **The failure mode is safe.** It is "not locked in time": archive-3 stays and
  nothing is submitted.
* **After the lock.** Scoring, the build and S3 fit the 10-14 to 10-17 window
  (stress_archive.py has `--archive` and `--no-rebuild`).
  `experiments/formative_run4.py` does not exist yet; it is needed only by 10-20.
* **The report.** P2 §7's report work (R1 to R6, 13.10 to 20.10) loses most of
  10-12 to 10-17 to this path.

## Decisions the author must confirm (in commit 1)

1. **Decision 6.** A4 presents the new calendar (lock 10-13, gate 10-15, archive-4
   10-16, upload 10-17, final 10-20) as the author's. It came from the workflow's
   task text (dates_log), so the author must confirm it, together with:
   * the dense counts of 20 and 40;
   * about 18 h of code on 10-12 and 10-13;
   * the slip of the report schedule.
2. **N1.** Make the twins no-loss for every candidate (D ≤ 0), make them gain
   regimes as well, or keep real-date gains with C1m's +0.045 shift disclosed.
3. **C1m's matching set.** C1m is matched on all of E (−2.035). Matched on dated
   subjects only it would be −1.891. With all of E, dated subjects sit −0.012
   below A3 under the shift.
4. **The open choices of impl_plan 0.2.** Mode A; (E1) as a mean plus P1a's
   bounds; C2's choices; family-whole split, exclusion and clusters; C3 dropped;
   the 0.10 cap; the App G.2 and G.6 wording, with N4's corrected sentence.
5. **An outside time stamp.** The design calls the lock commit "the outside time
   stamp", but commits here are local: origin/main is at 6915f98, and 8 commits
   are unpushed. A local commit's date is author-set. Pushing commits 1 to 4 when
   they are made gives a real time stamp. That needs the author's approval.

## What holds (checked)

* **The family split.** PE 5.1's S list equals `split_v2.json`'s family halves
  (44 and 88; pairs 988 and 11,680). Its flagged members are crow, emoji_attack
  and safeagentbench in S, and the other six in C. Every tilted regime's flagged
  share differs by at most 0.027.
* **The planning numbers.** PE 4.2's B0 table, PE 4.4's slope table, PE 7's power
  table and PE 12's budget sums equal the planning outputs.
* **The RS guard.** 5 test-like regimes × 80 tasks and R1B and R1P × 60 make 520
  tasks. The rows carry `res.ship` and library digests equal to HEAD.
* **The bundle checks.** `_check_bundle` already requires model.py's LEVEL to
  equal the bundle's level exactly, so C1m's and C2's literals are enforced.
  `HierPredictor` accepts explicit hyperparameters whose `included` and
  `excluded` lists match the prior's.
* **Archive-3.** Its prior.json digest (c7ce3b845d584f95) is
  `submission/prior.json` at HEAD. The archive's digest (4a882cc7d410e6a9) is the
  one `results/stress_archive.json` records. I could not hash `dist/` itself (a
  hook blocked it), so `dist/paiec.zip` is not re-hashed. Step 1 of impl_plan
  does that.
* **No leakage, apart from what PE discloses.** A3 has no pool data. Candidates
  exclude the run's families, or the RS parent. Per-benchmark fits do not depend
  on excl. Three quantities see every benchmark: T_full, mu0_m and C2's T. They
  are disclosed (PE 14 and 15). No label of C enters selection except through
  training standings.
* **No freedom left after the lock, except N5.** Knobs, compositions and the
  realisation check are fixed before the lock. Selection is committed before C.
  C-extra and `describe-hidden` come after the gate. S3 makes no fix, and the
  upload happens once.
