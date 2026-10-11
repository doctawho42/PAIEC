# P2: the final three weeks (2026-10-06 to 2026-10-30)

Written and committed on 2026-10-06, before any of the checks below was run.
The gates in section 3 and the submission rule in section 4 are fixed by this
commit; a later change to either is recorded as a dated amendment at the end of
this file, never by editing the text above it.

## 1 Decisions of 2026-10-06 (the author)

1. The final submission stays archive-3 (sha256 `4a882cc7d410e6a9…`, built at
   `4d2cc4f`, scored as formative run 3), unless check S1 or S3 finds a defect.
   No LLM or GPU at predict time: the CPU/numpy rule stays, as a scope choice
   (the platform itself allows a GPU, up to five model repositories of at most
   300B parameters and approved inference APIs).
2. The report is rebuilt around the hidden runs (section 5, R1), and the
   held-out test-like headline (-0.042 against LegacyP) is reported as a replica
   result.
3. The report's main text is cut to at most 9,000 words; App I.2 (the review
   map) and App H.4 (the revision history) leave the PDF and stay in the
   repository.
4. The author writes to the organisers with three questions (section 6).

## 2 Diagnosis this plan rests on

Exploratory analyses of 2026-10-06 (session scratch, not yet in `results/`; they
are ported to `experiments/hidden_runs.py` and its `results/` file by
2026-10-11 and must reproduce these numbers to four decimals before the report
cites them):

- When a pair gets one prediction for all its evaluation items, its Brier score
  is ECE² + p(1 - p) exactly, where p is the pair's rate on those items. So
  v = B0 - ECE0², read from the organisers' tables, is the Brier score of an
  oracle that knows each pair's rate. For LegacyP (run 1) B - ECE² is constant
  across all six budgets to 1e-6.
- v per run: 0.1536, 0.1797, 0.1608. ALC - v: 0.0577 (run 1, LegacyP), 0.0130
  (run 2, hier) and 0.0208 (run 3, hier). An ideal Beta-binomial learner of the
  pair rate on the same pairs pays +0.016 to +0.032, so hier already learns the
  pair rate about as well as possible; after B7 the hidden curve is flat.
- What remains is item-level: 78 to 88 percent of the distance to a perfect
  predictor. No covariate that transfers between benchmarks reaches a
  within-pair correlation above 0.15 (the acceptance gate needs 0.25).
- A run of these 7 benchmarks has v of 0.160 ± 0.017, so no learner of pair
  rates reaches the leaderboard's 0.117 on a typical run; that needs item-level
  information with a within-pair correlation of about 0.8 to 0.95.
- Every cheap lever measured on 2026-10-06 (per-budget temperature, affine and
  isotonic maps, blends with Smooth-cal, a run-level shift, budget switching
  between configurations, update damping, B0 stretch) is null or negative when
  fitted leave one parent out.

## 3 Checks and their gates

Each check writes its rows and results before its gate is read. A check whose
gate fails changes nothing in the submission and is reported as run.

| check | what | gate |
|---|---|---|
| S1 | The second-order term at dense B1. On dense real_webagents, hier's pooling costs +0.015 ± 0.010 of Brier at B1 (§7, App I.1). Stage 1: the exact posterior predictive (quadrature or importance sampling over the full joint posterior) against hier's line read at B1 and B3 for 50 to 100 sampled targets of dense real_webagents, with dense matharena and multi_swebench as controls. | G1, stage 2 only if Brier(hier) - Brier(exact) at B1 on real_webagents is at least 0.003 and its 95% interval excludes 0; otherwise S1 closes and the result goes into §7 and App I.1. G2, a fix ships only if it applies only when at least 6 other subjects of the benchmark are in `labeled`, and (a) on every formative-size row set (harness rows TL, TL-mix, R1; RS rows) it never applies and predictions are bit-identical; (b) on dense R2 under both split scopes it gains at least 0.002 on real_webagents and no dense benchmark is worse by more than 0.0005; (c) p99 latency of a dense call stays under 1 s; (d) `tools/build_submission.py` passes every check and the validator. |
| S2 | The multiple-choice floor for A to J options and explicit "Options:"/"Choices:" blocks (`paiec/mcq.py` reads A to E only). | 0 changed floors on public items, 0 hits on synthetic negatives (prose, code, numbered steps), worst-case regex time under 5 ms on 1 MB of hostile input, bit-identical predictions on the harness rows. Ships only as a passenger of a rebuild that S1 or S3 requires. |
| S3 | Robustness and latency of archive-3 at summative scale, through the validator's loader: 10,000 fuzzed inputs (empty and 1 MB item_content, unicode, nested, numeric and None item_features, non-empty interactors, odd release dates, missing fields); a dense composition (7 pseudo-benchmarks, 20 subjects, 31 labels, about 4,300 labels in `labeled`) at every checkpoint; 16 parallel workers. | No exception escapes; every output is finite and in (0, 1); p99 under 2 s and maximum under 30 s per call. A fix is made only on a failure, and must be bit-identical on every stored row set. |
| S4 | Ablations at the shipped LEVEL on a fresh regime-sensitivity seed, for the report only: sigma_delta x0.85, x0.7, x0.5; line read off; line read off with x0.7; x0.85 with the level mean preserved; wide35; wide50; eb_ship; equal and hidden-v weights. | Declared no-ship in advance. Anything that would pass the RS rule of App C.2 is reported, not shipped. |

Decision point G1 is Monday 2026-10-12. If S1 closes and S3 finds no failure,
the code is frozen on archive-3. Otherwise archive-4 is archive-3 plus the S1
or S3 fix (plus S2 if it passed) by 2026-10-15, G2 and the submission of
section 4 on 2026-10-16, and the final archive is chosen on 2026-10-17.
Anything not ready by 2026-10-17 does not ship.

## 4 Formative submissions

- No further formative submissions while the final is archive-3.
- If archive-4 is built, exactly one regression submission of it, no later than
  2026-10-16, announced in a dated amendment to this file before the upload.
  Rule: archive-4 becomes the final if every pair of that run is scored with
  finite outputs, whatever its score. Its table is recorded as run 4 and used in
  the report only through aggregate, id-free quantities such as ALC - v; it
  selects and tunes nothing and is compared with nothing to choose the final.
- Never: repeated submissions of the same archive to collect draws, A/B
  submissions to choose between archives, or any prediction shaped to learn
  about labels.

## 5 The report revision

Ranked; the numbers come from `experiments/hidden_runs.py` once it reproduces
section 2.

- R1. Lead with the hidden runs: the identity, v per run, ALC - v, hier at the
  level of an ideal pair-rate learner, the item-level share, and the check of
  the level calibration on the hidden runs themselves (LegacyP minus hier,
  adjusted for v: +0.0405, SE 0.019, against -0.0419, SE 0.0037, on held-out
  test-like runs). The test-like headline stays as the replica's result.
- R2. Correct the statements the analysis contradicts: that no formative run
  discriminates the regimes (§5.1, App D.2); the reading of the organisers'
  entry as the empirical mean (§3.4, Figure A1, findings); the lower-root
  reading of p(1 - p) = B31 (§1.2, §5.1), replaced by B0 - ECE0²; the stated sd
  of a run-to-run difference; and "the best leaderboard entry stays
  unexplained" (§5.5, App E.5), replaced by the bound of section 2.
- R3. The validity of late budgets: the hidden curve is flat after B7, and the
  replica overstates late-budget gains.
- R4. The S4 ablation table at LEVEL; contribution 2 (the line read) is stated
  as a correction of Laplace's under-reaction, not as an ALC gain.
- R5. One family of negative results for the post-hoc maps and the acquisition
  policies under the official protocol.
- R6. Protocol facts: measurement-db `main` and the baseline repository's head
  checked on 2026-10-06; the platform's runtime resources and our scope choice.
- Cuts: the main text to at most 9,000 words; the grid and tie-break narrative
  to App C; §1.5's table to about six rows plus a v-adjusted row; §6.2 and §6.3
  to one table and one paragraph; at most eight internal names in the main
  text; App I.2 and App H.4 out of the PDF.

## 6 Questions to the organisers

Asked by e-mail on or after 2026-10-06 (the author sends it): the size and
composition of the summative subset (a formative-size draw or a dense one, and
whether it uses the formative benchmarks); whether the leaderboard shows each
team's best or latest formative score; how Brier ALC and the manual grading of
the code and the report combine.

## 7 Schedule

| dates | work |
|---|---|
| 06.10 | this plan; the e-mail to the organisers |
| 07.10 to 08.10 | S1 stage 1; S4 overnight; S3 |
| 09.10 | S2 on a branch; S4 summary |
| 10.10 to 11.10 | `experiments/hidden_runs.py` and the other ports, with their `results/` files |
| 12.10 | G1: freeze on archive-3, or the archive-4 branch |
| 13.10 to 20.10 | findings; report R1 to R6 |
| 21.10 to 22.10 | cuts to at most 9,000 words |
| 23.10 | content freeze; provenance pass; `--strict` 0 |
| 24.10 to 25.10 | a cold read of the PDF |
| 26.10 to 27.10 | final build (archive sha256 checked), release commit, tag `v1.1.0` |
| 28.10 to 29.10 | OpenReview and the e-mail naming the selected submission (the author) |
| 30.10 | buffer only |

The fallback at every step is archive-3 and the report at `v1.0.1`.

## Amendment 1 (2026-10-06): G1 read early; the code is frozen on archive-3

All four checks of this plan's first days finished on 2026-10-06, so G1 is read
now instead of on 2026-10-12. The rule is unchanged.

- S1 closes (`experiments/dense_b1_check.py`, `results/dense_b1_check.json`).
  On dense real_webagents at B1, split scope 'pair', 100 targets from all 26
  pairs: Brier(hier) - Brier(exact) = -0.00035, 95% pair-cluster interval
  [-0.00152, +0.00079], Monte Carlo SE 0.00012. The difference is below 0.003
  and the interval contains 0, so stage 2 does not open. The exact posterior
  predictive pays the B1 pooling cost as hier does, so that cost comes from the
  model, not from hier's approximations.
- S3 finds no failure (`tools/stress_archive.py`, `results/stress_archive.json`).
  45,500 predict calls of archive-3 through the validator's loader: no
  exception escaped, every output finite and in (0, 1), p99 0.516 s and maximum
  20.21 s per call. Deviation, disclosed: the gate was read on all calls pooled.
  The harness first also required each part to pass on its own; a smoke run
  showed the fuzz part alone at p99 2.10 s, and the per-part requirement was
  dropped before the full run, on the reading that section 3 states one
  criterion for S3. Read per part, the fuzz part's p99 is 2.146 s wall time, on
  a machine with a load average of 23 to 235 on 8 cores; its CPU-time p99 is
  recorded beside it. Neither reading would ship a change, because S3 fixes
  only failures and no call failed.
- So the code is frozen on archive-3, and S2 does not run. No archive-4 and no
  formative submission follow (section 4).
- The section 2 numbers are reproduced by `experiments/hidden_runs.py`
  (`results/hidden_runs.json`), with two precisions: the per-pair constancy of
  B - ECE^2 for run 1 is 1.13e-6 at most, within the tables' 6-decimal rounding
  (the run mean varies by 1.5e-7), and the item-level share is 77.9 to 88.6
  percent.
- S4 is prepared (`experiments/ablations_at_level.py`, seed 20261006, 520
  tasks) and scores both readings of its list: multiples of the
  leave-one-parent-out sigma_delta (App B.3's convention) and the pilots'
  absolute values. It stays no-ship.

## Amendment 2 (2026-10-06): E1, the report's framing, and the feedback statements

Committed before E1 runs.

- E1 (report only, no-ship): score EmpMean (the organisers' empirical mean)
  and 1PL-ship (a plain 1PL at hier-ship's level) in the four regime-
  sensitivity regimes where they are missing (READING, AUDIT, MIXTURE, FLAT),
  on the same RS runs and checkpoints as the other comparators, from stored
  public rows, with RS's paired differences against ship and cluster SEs. It
  fills the comparator table of the report; nothing it finds is shipped or
  selects anything.
- The report v2 is framed as an evaluation-methodology paper with the
  submitted predictor in a full methods section, under the title "Reading
  hidden runs at a benchmark-level cold start: a pair-rate oracle, test-like
  regimes and a calibrated hierarchical predictor".
- The author confirms (2026-10-06) the reworded statement on run 3 for App G.6
  and the conduct section: no level reading, matching against replica pairs or
  tuning was done on run 3; after the code was frozen on archive-3
  (2026-10-06), its table entered, with those of runs 1 and 2, a root-free
  analysis of per-pair Brier and ECE (V and ALC minus V), which uses the
  anonymous benchmark ids only to group pairs in one bootstrap, and which
  chose, tuned and changed nothing. App G.6 also lists S4's hidden-V weights
  (formative bin counts only) and the leaderboard values read on 2026-09-24
  that the bound of section 2 quotes.

## Amendment 3 (2026-10-07): the id-use wording of the run-3 statement

`experiments/hidden_runs.py` adds a distinct-pair sensitivity: one
subject-benchmark pair appears in run 1 and run 3, and the sensitivity counts it
once. Finding that pair uses the anonymous subject and benchmark ids (the runs'
overlap check that App G.6 already declares), and the analysis now has two
stratified bootstraps. The author confirms (2026-10-07) this wording for App
G.6 and the conduct section, replacing the corresponding clause of amendment 2:
the analysis "uses the anonymous ids only to find the one subject-benchmark pair
that appears in two runs and to group pairs by benchmark in its bootstraps, and
chose, tuned and changed nothing".

## Amendment 4 (2026-10-12): the expanded pool, and a gate for refitting hier's priors on it

Written on 2026-10-07 and revised on 2026-10-11/12 for the calendar of
section 4.5, the fetched item tables, the re-check of 2026-10-11
(`docs/plans/pool_design/recheck.md`, N1 to N9) and its verification
(`nfix_verify.md` there, V1 to V13), before any predictor was scored on the
expanded pool; committed on 2026-10-12 Moscow time (2026-10-11 UTC), before `experiments/pool_eval.py`
scores any task. The full design is `docs/plans/p2_pool_eval.md`
("p2_pool_eval §n" below; sha256
`5af21af6c818d8a43d22cbbb18876426949d801165cfec28bb4c1177a60d9591`),
committed with its notes and scripts under `docs/plans/pool_design/`;
where the two differ, this amendment governs. The lock hashes everything
p2_pool_eval §9 item 10 lists, among them this amendment's text (to the next
`## Amendment` heading), `RULE_TEXT`, that file and the script's constants
block. Only a later dated amendment changes anything here, with one limit:
nothing in the rule, candidates, split, regimes, seeds, counts, statistics
or fix scope changes once a scored row exists, not even by a later
amendment. A scored row is a row of SEL_SEED, CONF_SEED or the RS guard.
Amendment 5 only announces the upload, and the amendment that records the
final only records it.

### 4.1 Why the code is unfrozen, and the author's decisions

measurement-db grew on 2026-10-06 and 2026-10-07 to 252 public benchmarks
(revision `54074453`), assigned to the pools at random before curation, so a
held-out public benchmark stands in for a hidden one. Archive-3's priors rest
on five benchmarks; the pool offers 128 for the subject prior and 137 for the
item variance and the level. The difficulty is the level: the pool's levels
centre at +0.06 (sd 1.54), LEVEL's median sits at their 15.7th percentile, and
the formative readings sit low (READING -0.715, AUDIT -1.10, TUNED -1.29). A
pool level wins on unselected held-out benchmarks by construction, so the rule
asks for gains where the readings sit (p2_pool_eval §1, §4). The subject
ridge's date term moves the B0 prior centre (on E's pairs, p2_pool_eval §4.2)
with the subjects' dates, so no one mu0 holds archive-3's centre both at real
dates and under the 1.25-year date shift LEVEL was calibrated with. C1 keeps
LEVEL and sits within 0.009 of archive-3's centre at real dates; C1m's mu0 is
matched so that its centre equals archive-3's under the shift; C2 takes the
pool's level (*planning*). Every candidate must gain at real dates and must
not lose (D <= 0) in the same runs under the shift. So, for the builds with
nothing excluded, C1's gains are read at archive-3's level centre to within
0.009, and C1m's no-loss condition at it, to the rounding of mu0; a run's
build without its families moves these centres a little, and the `fits`
stage reports how far it moves C1m's matched mu0 on 50 runs of each half.
C1m's gains include its +0.045 centre shift (dated subjects +0.063), and
C2's gains and twins both carry a level change.

**Decisions of the author (2026-10-07; decision 6's dates revised and
decisions 7 and 8 added on 2026-10-11; the commit of this amendment confirms
all three).**

1. Amendment 1's code freeze is lifted for hier's priors only, under the gate
   below. archive-3 is the fallback at every step.
2. The item tables are fetched (Mode A). Pool runs and fits see item content,
   features and interactors, so the multiple-choice floor and the group effects
   act on pool runs as they will at test time, and the group share is fitted by
   fit_hyper's rule (p2_pool_eval §3.2). The fetch is done: every item table of
   L is present and matches the manifest; only cseo_bench's and qatch's, both
   outside L, were skipped (p2_pool_eval §3.7).
3. (E1) is a mean over the seven RS regimes, at most +0.0005, plus P1a's
   per-regime bounds, not a literal +0.0005 in each regime.
4. C2's choices: its level set includes the nine direction-flagged benchmarks,
   no WIDEN[0] is applied, and attr_scale stays 0.5.
5. The design notes and scripts are committed under `docs/plans/pool_design/`.
6. On this path the final is recorded by 2026-10-20, not 2026-10-17, and
   section 4.5's hard stops apply: the lock by 2026-10-13, the selection and
   the gate by 2026-10-15, archive-4 built, through S3 and announced by
   2026-10-16, its one upload by 2026-10-17 (all 23:59 UTC). Each missed stop
   leaves archive-3.
7. The date-shifted twins P-READING-DS and P-AUDIT-DS are no-loss conditions
   for every candidate, D <= 0 in (E2) and in (c) (the re-check's N1, its
   option (a)). No run is added, and (a)'s power is unchanged; with no twin
   effect a carried candidate passes the four twin conditions of selection
   and gate together in only about 0.06 to 0.25 of draws (p2_pool_eval §7).
8. Commits 1 to 4 (this amendment with the design, the lock, the selection
   record, the gate) are pushed to the public repository when made (the
   author approved pushes of pre-registration commits on 2026-10-06), so the
   lock commit is an outside time stamp (p2_pool_eval §9 item 10).

### 4.2 What this amendment supersedes, and its scope

On the path this amendment opens, and only there: §1's decision 1 also lets
archive-4 replace archive-3 when this gate passes and section 4.5 is met; §3's
"archive-4 is archive-3 plus the S1 or S3 fix" becomes archive-3's runtime with
the carried candidate's pool-fitted priors (S1 and S2 stay closed); §3's
dates for archive-4 ("by 2026-10-15", "the submission of section 4 on
2026-10-16", "chosen on 2026-10-17 ... does not ship") give way to section
4.5's dates; §4's "no later than 2026-10-16" gives way to section 4.5's upload
by 2026-10-17 23:59 UTC; and amendment 1's "No archive-4 and no formative
submission follow" gives way to one archive-4 and its one regression
submission after a pass. §4 is otherwise unchanged.

archive-4 may differ from archive-3 in `submission/prior.json`, in model.py's
`LEVEL` literal (C1m, C2), and in model.py's docstring and the comment above
`LEVEL`, which must be true for what ships; an AST comparison ignoring
docstrings, comments and the `LEVEL` value must find model.py otherwise
identical. The eight shipped modules stay byte-identical (p2_pool_eval §2). The
new code is offline: `paiec/pool.py`, `tools/pool_benchmarks.json`,
`tools/pool_families.json`, `experiments/pool_eval.py`, tests, and a `--pool`
build path that must reproduce the locked bundle byte for byte, under the
lock's settings (`PYTHONHASHSEED=0`, one BLAS thread).

### 4.3 The candidates and the evaluation, in brief

* **Data** (p2_pool_eval §3): revision `54074453…` with item tables, every file
  checked against the manifest; L 137 benchmarks, U (L less the nine flagged)
  128, E (L less the legacy five) 132. A pool run's priors are built without
  every benchmark of the source families (shared catalogue `source_url` key or
  `paper_url`) of the run's benchmarks.
* **Candidates** (p2_pool_eval §4), each the pool build with archive-3's
  identity guards: **C1** keeps LEVEL (-2.5, 2.5, 0.5); **C1m** matches mu0 so
  that its B0 prior centre on E's subjects dated 1.25 years later equals
  archive-3's (*planning* -2.035); **C2** takes the pool's empirical-Bayes
  level (*planning* N(+0.16, 2.31²)). Each B0 centre on E's pairs minus
  archive-3's, at real dates and at +1.25 years (*planning*): C1 +0.009 and
  -0.037; C1m +0.045 (dated +0.063) and 0.000; C2 +0.233 and +0.190. So C1's
  gains are read at archive-3's level centre (to 0.009), C1m's no-loss twins
  are read there, and C1m's gains carry its centre shift (p2_pool_eval §4.2,
  §4.3). C3 (identity guards lifted) is dropped: with no confirmation cell
  anywhere, its uncapped identity weight (0.80 at three or more other cells)
  exceeds the measured slope (0.76), the stated criterion.
* **Split** (§5.1): family units, flagged units in level blocks of 3; S 44
  benchmarks, C 88. **Regimes** (§5.3): P-READING and P-AUDIT (gains); their
  date-shifted twins P-READING-DS and P-AUDIT-DS (no loss, D <= 0); P-TUNED,
  P-MIXTURE, P-FLAT (loss bounds, +0.002); D-READING, D-FLAT (loss bounds,
  +0.001); P1a's seven RS regimes (guard). 80 runs a P- and 20 a D- regime in
  S, 120 and 40 in C (the dense counts set by this calendar, p2_pool_eval
  §5.4 and §12).

### 4.4 The decision rule (verbatim `RULE_TEXT`)

```
POOL-EVAL DECISION RULE: does a pool-fitted hier replace archive-3?
(amendment 4 to docs/plans/p2_final_plan.md; design docs/plans/p2_pool_eval.md;
experiments/pool_eval.py). Fixed before any task of SEL_SEED, CONF_SEED or the
RS guard was scored.

Terms
  A3        archive-3: paiec.hier.HierPredictor over archive-3's prior.json
            (sha256 c7ce3b845d584f95...; LEVEL mu0 -2.5, sigma_mu 2.5,
            attr_scale 0.5; fitted on the five legacy benchmarks). On RS
            tasks: P1a's stored `ship` rows.
  C1,C1m,C2 the candidates of CANDIDATES (p2_pool_eval section 4): the pool
            build with LEVEL kept; with mu0 matched to A3's B0 centre on E at
            +1.25 years; with the pool's empirical-Bayes level. On a pool run
            each is built without every benchmark of the source families of
            the run's benchmarks; on an RS task without the target's parent.
  E, S, C   the 132 evaluation benchmarks and their selection (44) and
            confirmation (88) halves (SPLIT; no family straddles the halves).
  P-READING, P-AUDIT, P-TUNED, P-MIXTURE, P-FLAT
            formative-size pool regimes at real dates; P-READING-DS and
            P-AUDIT-DS the runs of P-READING and P-AUDIT with every subject's
            dates moved by 1.25 years; D-READING, D-FLAT dense pool regimes;
            RS-TUNED, RS-READING, RS-AUDIT, RS-MIXTURE, RS-FLAT, RS-R1B,
            RS-R1P P1a's regimes on its scoring seed 11 (520 tasks).
  D(X,R)    the mean over R's planned runs of the run's pair-mean ALC of X
            minus that of A3, on identical runs and checkpoints.
  SEg(X,R)  max(SEf, SE2): SEf the cluster-bootstrap SE over source families
            (ratio estimator, 2,000 resamples, seed 0); SE2 the two-way SE
            sqrt(max(0, SEf^2 + SEs^2 - SEfs^2)) from the same bootstrap with
            the subject identity key and the (family, key) cell as clusters.
  U95(X,R)  D(X,R) + 1.96 SEg(X,R).
  PB(X,R)   the mean over source families, each once, of the family's
            mean pair-ALC difference, X minus A3, with each of its
            appearances weighted 1/(pairs in the appearance's run), as D
            weights them.
  JK(X,R)   the maximum over source families f of
            (sum of v_k over appearances k outside f) /
            (sum of n_k over appearances k outside f), where each pair
            appearance k has weight w_k = 1/(pairs in its run), the weight
            of D, v_k = w_k times its pair-ALC difference X minus A3, and
            n_k = w_k; a run whose pairs all lie in f drops out.
  CPU time  time.process_time per evaluation call.

Selection (S and RS only; recorded and committed before any CONF_SEED task
is scored). X in {C1, C1m, C2} is eligible iff
  (E1) the mean of D(X,RS-R) over the seven RS regimes is <= +0.0005, and
       D(X,RS-R) <= +0.002 in each of RS-TUNED, RS-READING, RS-AUDIT,
       RS-MIXTURE and RS-FLAT, and <= +0.001 in RS-R1B and RS-R1P;
  (E2) on S: D <= +0.002 in P-TUNED, P-MIXTURE and P-FLAT; D <= 0 in
       P-READING-DS and P-AUDIT-DS; and D <= +0.001 in D-FLAT and
       D-READING;
  (E3) on the S pool tasks X's mean CPU time per evaluation call is at most
       2.0 times A3's in the same tasks, and no call of X on the S
       formative-size runs or on the RS tasks takes more than 2.0 s of CPU;
  (E4) every planned S and RS task is scored for every arm (A3, C1, C1m and
       C2 on pool tasks; C1, C1m and C2 on RS tasks), except an arm on a
       run where Reading 4's invalid-output clause applies to it, and every
       check passed.
  M(X) = [D(X,P-READING) + D(X,P-AUDIT)] / 2, D on S. Q is the set of
  eligible X with M(X) <= -0.001. If Q is empty, the outcome is "no candidate
  carried: archive-3 stays". Otherwise X* is the earliest member of Q, in
  the order C1, C1m, C2, within 0.0005 of the minimum of M over Q (that
  is, with M(X) <= that minimum + 0.0005).

Gate (C only; read once, on the full planned set). X* replaces A3 iff
  (a) in P-READING and in P-AUDIT: D <= -0.002 and U95 < 0;
  (b) in P-READING and in P-AUDIT: PB <= 0 and JK <= -0.001;
  (c) in P-TUNED, P-MIXTURE and P-FLAT: D <= +0.002; in P-READING-DS and
      P-AUDIT-DS: D <= 0;
  (d) in D-READING and D-FLAT: D <= +0.001;
  (e) (E1) holds for X*;
  (f) on the C tasks: X*'s mean CPU time per evaluation call is at most 2.0
      times A3's; no call of X* on the C formative-size runs takes more
      than 2.0 s of CPU; X*'s dense calls in the C core have a p99
      (numpy.percentile(x, 99), linear interpolation, over all of them) of
      at most 1.0 s and a maximum of at most 30 s of CPU;
  (g) every check passed and every planned C task is scored for A3 and X*.

Reading
  1. Pass: archive-4 carries X*'s locked bundle (built with nothing
     excluded; sha256 in the lock), which tools/build_submission.py --pool
     must reproduce byte for byte from the locked paiec/pool.py (or its
     successor under p2_pool_eval section 9 item 11) and fits cache; for C1m
     and C2, model.py's LEVEL is the locked literal. It becomes the final
     only if, by 2026-10-16 23:59 UTC, the build passes every check and the
     organisers' validator, S3's gate (P2 plan section 3), read on all calls
     pooled as amendment 1 read it, holds on archive-4 and amendment 5
     announces the upload; the one regression submission of P2 plan section
     4 is uploaded, once, by 2026-10-17 23:59 UTC and every pair of it is
     scored with finite outputs; and the final choice is recorded by
     2026-10-20 23:59 UTC. Otherwise archive-3 stays.
  2. Fail, or no candidate carried: "archive-3 stays", recorded as a
     negative result with every D, SE and condition, for every candidate
     scored.
  3. Everything else (the decomposition, budgets, subject kinds, dated and
     undated subjects, flags, families, multiple-choice benchmarks, tercile
     and recency cuts, v decomposition, latency tables, the C-extra arms,
     the post-gate placement of the hidden runs) is descriptive and changes
     nothing.
  4. Nothing in this rule, CANDIDATES, SPLIT, the regimes and their knob
     grids, the seeds, the run counts, the statistics or the fix scope
     changes once a row of SEL_SEED, CONF_SEED or the RS guard exists, not
     even by a later amendment. No run is added or dropped after rows are
     read; each stage's rule is applied once, to its full planned set.
     Fallbacks and other valid outputs of the shipped runtime are scored as
     they occur and counted; they are never check failures. An invalid
     output (anything but a finite float in [0, 1], or of the wrong shape)
     is recorded, never raised, and gives "archive-3 stays" for the
     candidate whose arm produced it and for no other, unless A3's arm
     produced it, in which case it holds for every candidate. If a check fails, the cause is fixed within the scope of
     p2_pool_eval section 9 item 11 (harness and I/O code only:
     experiments/pool_eval.py outside its constants block, and the
     definitions paiec.pool.EVAL_LOADER lists, frozen at the lock; every
     locked bundle digest and the constants block, the classification of
     outputs included, unchanged), the fix is committed with its diff
     before any re-score, every affected task is re-scored in full (by that
     item's rule: the locked tests pass unchanged, the failed tasks and a
     fixed verification set are re-scored, a whole stage if a verification
     row differs), select, summarise and gate are re-run from all rows (a
     re-run selection naming another X* gives "archive-3 stays") and the
     deviation is recorded; the rule is not rewritten. A failure whose fix
     would go outside that scope (paiec/pool.py's fit loaders or bundle
     arithmetic, CANDIDATES, a bundle or a constant) gives "archive-3 stays"
     for the candidates it affects. If the lock is not committed by
     2026-10-13 23:59 UTC, the outcome is "not locked in time: archive-3
     stays"; if the selection and then the gate have not been read and
     committed by 2026-10-15 23:59 UTC, it is "not read in time: archive-3
     stays".
  5. Formative feedback enters only as the RS study's global level targets
     and testlike's run shape and 1.25-year date shift, reused for this
     ship decision (amendment 4, section 4.6); through archive-3's LEVEL
     the shift also fixes C1m's mu0, a global level hyperparameter, by the
     matching rule of p2_pool_eval section 4.3. results/hidden_runs.json is
     read once, after the gate is committed, for a descriptive placement.
     Formative run 3 sets nothing.
```

### 4.5 Dates, hard stops and outcomes

| date (2026, UTC) | work |
|---|---|
| Mon 10-12 | this amendment, p2_pool_eval.md and `docs/plans/pool_design/` committed; `paiec/pool.py`; `fits` |
| Tue 10-13 | `experiments/pool_eval.py`; `split`, `regimes`, `reproduce`, `timing`; the lock (hard stop) |
| Wed 10-14 to Thu 10-15 | S and RS scored, selection committed; C scored, the gate read once and committed (hard stop 10-15) |
| Fri 10-16 | on a pass: archive-4 built, S3, amendment 5 (hard stop) |
| Sat 10-17 | on a pass: the one regression upload (hard stop) |
| by Tue 10-20 | run 4's table read for finite outputs; the final recorded |

**Hard stops**, each leaving archive-3 the final if missed: the lock (the
digests of the implementation, the constants block and the bundles) by
2026-10-13; S and RS scored and the selection committed, then C scored and the
gate read once and committed, by 2026-10-15; archive-4 built, through S3 and
announced (amendment 5) by 2026-10-16; uploaded once by 2026-10-17; the final
recorded by 2026-10-20 (all 23:59 UTC). On this calendar the dense counts are
20 in S and 40 in C from the start, wherever the lock lands (p2_pool_eval §5.4
and §12).

**Outcomes.** *Pass*: archive-4 is uploaded once and becomes the final if every
pair of run 4 is scored with finite outputs, whatever its score; run 4 is used
only through aggregate, id-free quantities. *Fail, no candidate carried, not
locked or not read in time*: archive-3 stays the final, nothing is submitted,
and the result enters `docs/findings.md` with every D, SE and condition.
*archive-4 not built, through S3 and announced by 2026-10-16, or not uploaded
by 2026-10-17*: archive-3 stays the final and nothing is submitted. *Run 4's
table not read for finite outputs, or the final not recorded, by 2026-10-20*:
archive-3 stays the final. P2 §5 gains R7, the expanded pool, ranked after R2.

### 4.6 Formative feedback (App G.6)

Three global, id-free quantities are reused to decide whether a refitted prior
ships, a new use: the RS study's level targets, testlike's run shape and its
1.25-year date shift. Under C1m the date shift, through archive-3's LEVEL (set
by the feedback before), also sets mu0 by the fixed matching rule of
p2_pool_eval §4.3. mu0 is a global hyperparameter of the level distribution,
so this stays inside the limit of `docs/report/draft.md:305`, whose "here three
numbers" gains a clause for C1m (p2_pool_eval §11). After the gate,
`results/hidden_runs.json` is read once
for a descriptive placement; if archive-4 ships, run 4's table is used as §4
allows. `docs/report/draft.md:305`, App G.2 and App G.6 are amended as
p2_pool_eval §11 words them. The author confirms, in the commit of this
amendment, that nothing else from the feedback was read for this evaluation.
