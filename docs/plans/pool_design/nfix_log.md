# Amendment 4: the re-check's N1 to N9, fixed (2026-10-11)

Written 2026-10-11, 18:45 to 19:15 UTC, against repository HEAD 2bfcd17. It
covers `amendment4_short.md` ("A4"), `p2_pool_eval.md` ("PE") and
`impl_plan.md`, revised for the required changes N1 to N9 of `recheck.md`.

Rules kept: no git write; no ALC or Brier of any predictor; no item text; no
formative feedback beyond what App G.6 and PE §11 list. Writes stay inside
this folder.

What I ran:

* sha256 of files;
* a static AST scan of the in-repository import closure of the harness modules;
* one new planning script, `a4final/dense_cap.py`. It does composition only,
  like `split_v2.py`, on `benchmarks.csv`, `pairs.csv` and `families.json`.

No predictor was run, and no item table or formative table was read.

**Where the N1 choice came from.** The choice of option (a) comes from the
workflow's task text, not from the author in this session. So does "the
author chose (2026-10-11) the full design with candidates C1, C1m, C2". A4
records it as decision 7, and its heading says the commit of the amendment
confirms it. The author confirms it in commit 1, as for decision 6.

The pre-fix copies of the three files are in `nfix_orig/`, for the diff. They
are not for commit 1 (PE §13 now says so).

## Digests

| file | before | after |
|---|---|---|
| p2_pool_eval.md | `3e0800974e9f580823a0bc56a97c80e3b7dd769959d4b3de5d1fcceceb14f3c5` | `3341c4742b3a243409e26e24e3de77aba2dde049361635644e9913b5bf6e4a39` |
| amendment4_short.md | `c655e4828a0bae8252ea4f8f4a3cabb08f26870ac657366452d3a20340558e3e` | `a45df8833b3e2bbd3c30951a69bf9cc1cf75fd701d1a2afdfd4f521785da874f` |
| impl_plan.md | `1a95760d177e0ae42b5ac10c189e0fe72262faade6be804c7c123c384d8cac00` | `05ecff7df35975961c4a88823ca8dfafa19728e3e1c3029ccdf4f5ec5dbc30a9` |
| a4final/dense_cap.py (new) | | `bc692edb849a701ee1767dfdd01e5b086e352a52ec46c3866bf90cb807535f80` |
| a4final/dense_cap.json (new) | | `a75298c6a0a79cc979b2ded8ae099793b0d402fc1a4ddb8ec1ada609ef43dd6c` |

* A4's preamble quotes PE's new sha256 in full. impl_plan step 0.1 quotes its
  first 16 hex, `3341c4742b3a2434`.
* PE names `nfix_log.md` and `recheck.md` but quotes no digest of A4,
  impl_plan or this log. The chain still has no cycle.
* PE was not touched after its digest was taken. Any later edit to PE needs
  the new digest in both places again.

## What changed, finding by finding

### N1: C1m's gains are read where its level is not archive-3's

The rule change of option (a) is adopted for every candidate.

* **`RULE_TEXT`**:
  * (E2) is now "D <= +0.002 in P-TUNED, P-MIXTURE and P-FLAT; D <= 0 in
    P-READING-DS and P-AUDIT-DS; and D <= +0.001 in D-FLAT and D-READING".
  * (c) is now "in P-TUNED, P-MIXTURE and P-FLAT: D <= +0.002; in P-READING-DS
    and P-AUDIT-DS: D <= 0".
  * The twins no longer appear in any +0.002 list, so the wording is the same
    wherever the twins are named.
* **A4**:
  * §4.1 has a new decision 7.
  * §4.1's paragraph no longer says the candidates "separate the level from
    ... the ridge's date term". It now says what holds:
    * the date term moves the B0 centre on E's pairs;
    * C1 sits within 0.009 of archive-3's centre at real dates;
    * C1m equals archive-3's centre under the shift, to the rounding of mu0;
    * every candidate must gain at real dates and must not lose in the twins;
    * so C1's gains, and C1m's no-loss condition, are read at archive-3's
      centre;
    * C1m's gains include its +0.045 shift (dated subjects +0.063);
    * all of this holds for the builds with nothing excluded, and `fits`
      reports how a run's build moves the centres.
  * §4.3 now gives each candidate's centre difference at both dates (+0.009
    and -0.037; +0.045 and 0.000; +0.233 and +0.190) and what follows from
    it.
  * §4.3's regime list now reads: twins "no loss, D <= 0"; P-TUNED,
    P-MIXTURE, P-FLAT "+0.002"; dense "+0.001".
* **PE**:
  * §1's "So" bullet is rewritten in the same terms. It ends with what is
    exactly true: between them, the rule reads the pool's rest at
    archive-3's centre under each date assumption, as a gain at real dates
    (C1) and as a no-loss condition under the shift (C1m).
  * §4.2's last paragraph:
    * C1m's centre differences are added (dated and undated).
    * It states that the rule does not split C1m's gain at real dates into
      its centre shift and its rest.
    * A new paragraph defines "level centre" (c on E's pairs, for the build
      with nothing excluded) and says where the run-level and per-regime
      departures are reported.
  * §4.3's contrast table is now two columns, real dates and twins, and says
    where the rule reads each contrast. A closing paragraph says what a
    candidate that passes has shown, on point estimates.
  * §5.3:
    * the role column for the twins reads "no loss, D ≤ 0";
    * the twins bullet says they are no-loss conditions for every candidate.
  * §7 has a new bullet on the twins' D ≤ 0:
    * the power of (a) is unchanged;
    * for a nil twin effect, each twin holds with probability about 0.5;
    * both together hold with probability 0.25 to 0.5;
    * for a twin gain of at least one SE, each holds with probability at
      least 0.84.

    The bullet also states the recheck's recommended note: the loss bounds
    are point estimates.
  * §14's Pass text: X* did not lose (D ≤ 0) in the date-shifted runs. Under
    C1m, the gains at real dates include its centre shift.
  * §15: the R1 row is updated, and a new table maps N1 to N9 to sections.
* **impl_plan**:
  * 0.2 lists the twins as no-loss conditions;
  * §5's `RULE` gains the twins' threshold 0;
  * §6 has the tests.

### N2: the lock sentence

A4's preamble now reads: "The lock hashes everything p2_pool_eval §9 item 10
lists, among them this amendment's text ..., `RULE_TEXT`, that file and the
script's constants block."

### N3: lock completeness

* **PE §9 item 10** pins three more modules:
  * `paiec/baselines.py`, 64fe42f5a63d22e1;
  * `experiments/ship_confirm.py`, ec6e73ea1e3e8935;
  * `experiments/subject_side.py`, 888c11b0c1b13c05. `ship_confirm.py`
    imports it (line 112). The recheck did not list it; the AST scan found
    it.

  All three equal HEAD. The static closure of `regime_sensitivity.py`,
  `level_calibration.py`, `testlike.py` and `official.py` contains nothing
  else outside the shipped eight.
* **Import-closure check.** The `lock` stage computes the import closure of
  `experiments/pool_eval.py` and `paiec/pool.py`. It refuses if the closure
  holds an unpinned file.
* **CSVs.** Both options are taken.
  * `paiec/pool.py` derives each pair's n and k, the pair logits and
    `level_cc` from the manifest-verified responses.
  * `split` checks them against `pairs.csv` (n and k exactly, logits to 1e-12)
    and `benchmarks.csv` (`level_cc` to 1e-12).
  * It writes them to `data/pool_eval/levels.json`, which every later stage
    reads.
  * The lock hashes `levels.json` and both CSVs (d88247c984edc5df,
    2e913b8b24c29fc8).
  * PE §5.1 defines `level_cc` by its formula, which is inventory.py's:
    log((k + 0.5) / (n - k + 0.5)), averaged over the eligible pairs.

### N4: the feedback statements under C1m

* **PE §11**:
  * Item 3 now says that under C1m the shift, through A3's LEVEL, sets mu0 by
    the matching rule. mu0 is a global level hyperparameter, inside the
    `draft.md:305` limit.
  * The section's lead sentence says the same.
* **The `draft.md:305` text.**
  * "here three numbers: ...γ" gains a clause for C1m.
  * The new sentence no longer says "they set no parameter of it". It says
    that, beyond what each candidate keeps of archive-3's level prior (C1 all
    three numbers, C1m σ_μ and γ, C2 γ), the feedback set no parameter,
    except μ0 under C1m.
* **Disclosure 3** (PE §15) adds the dependence.
* **A4**: §4.6 says the same, and `RULE_TEXT` Reading 5 adds "through
  archive-3's LEVEL the shift also fixes C1m's mu0 ... by the matching rule".

### N5: the scope of a post-lock fix

PE §9 has a new item 11, and `RULE_TEXT` Reading 4 is rewritten to point to
it.

* **Not failures.** Fallbacks and other finite outputs of the shipped runtime
  are scored and counted. They are never check failures.
* **Non-finite predictions.** This is a precision the recheck did not ask for:
  a non-finite prediction cannot be scored. Its cause lies outside the fix
  scope, so it gives "archive-3 stays" for that arm's candidate.
* **What a fix may touch**:
  * `experiments/pool_eval.py`, outside its constants block;
  * in `paiec/pool.py`, only `load_eval_pairs` and the index cache that it
    alone uses.

  `paiec/pool.py` is checked by an AST comparison with those definitions
  removed. `items_of` is excluded because it feeds the fits; this is a
  narrower reading of the recheck's "the loaders". The fits read every item
  file of L, and `timing` reads the two largest of E, before the lock.
* **Out of scope.** The fit loaders, the bundle arithmetic, `CANDIDATES`, a
  bundle or a constant. A fix there gives "archive-3 stays" for the
  candidates it affects.
* **How a fix is recorded**:
  * it is committed with its diff and a deviation record before any re-score;
  * the lock record is not rewritten; the fix appends successor digests, and
    the lock check and `--pool` accept those only for files in scope;
  * a re-scored task must reproduce its earlier per-run bundle digests.
* **Time.** A fix commit is not a re-lock and may land after 10-13. The 10-15
  stop still binds.
* **Also changed**:
  * PE §10's build step;
  * Reading 1 ("or its successor under ... item 11");
  * impl_plan §5 (`lock`, `score`, the new `record-fix` stage), §6, §11 and
    the risks. "A bundle fails at run time" is now "archive-3 stays".

### N6: RP3

* **PE §9 item 6.** RP3 runs on label-randomised copies:
  * every 0/1 response is replaced by `default_rng([TIMING_SEED, 3, i])`;
  * items, subjects, dates and the acquisition order are unchanged.
* `run_official`'s scores are discarded, never written or printed.
* An assert checks that every response equals its stream draw, and a test
  checks that no RP3 output holds a Brier or an ALC.
* impl_plan §5 and §6 follow.

### N7: RULE_TEXT precision

* (E1) names RS-TUNED, RS-READING, RS-AUDIT, RS-MIXTURE and RS-FLAT.
* The tie-break now reads: "X* is the earliest member of Q, in the order C1,
  C1m, C2, within 0.0005 of the minimum of M over Q (that is, with M(X) <=
  that minimum + 0.0005)".
* Reading 1 now reads: "S3's gate (P2 plan section 3), read on all calls
  pooled as amendment 1 read it".
* PE §7 states the tie-break. impl_plan §6 adds tests: two earlier members
  inside the margin, and the inclusive bound.

### N8: memory and time of C's item reads

* **PE §3.2: reading items.** `items_of` reads `item_id`, `content` and
  `item_features` only, never `verifier` or `grading_criterion`. It reads
  through `ParquetFile(path, buffer_size=READ_BUFFER > 0,
  pre_buffer=False).iter_batches(batch_size=ITEM_BATCH <= 256, columns=...)`.
  The mind2web and swe_smith row-group facts are stated.
* **The mixed benchmarks' scale report** reads `grading_criterion` by a
  separate batched read of the nine, never through `items_of`. It writes
  counts only.
* **PE §9 item 3.** mind2web and swe_smith are each fitted in a fresh worker,
  which records its peak RSS (`ru_maxrss`) and the CPU time of the read and of
  the fit.
* **PE §9 item 9.** `timing` adds two formative-size runs from C, labels
  stripped, on a scoring worker. Their first pair is fixed: mind2web's one
  pair, and a swe_smith pair. If a worker passes 1.45 GB or a call fails, the
  cause is fixed before the lock.
* **PE §5.4.** The exit-75 check runs between tasks, after the rows are
  written, so no task is repeated or killed for its size. This clarification
  closes the loop the recheck describes.
* **PE §12.** The timing row notes the two runs; the figure stays 1,200 s.
* impl_plan §2, §3, §4, §5, §7 and the risks follow.

### N9: small consistency fixes

* **The dense weight cap.** The weights are now built in this order: the tilt
  over the half, then the sampler's restriction (for dense runs, the
  benchmarks with at least 4 pairs), then renormalisation, then the 0.10 cap.
  The balance check, Kish n and the largest share are read on these weights
  (PE §5.1, §5.3).
  * The new `a4final/dense_cap.py` reproduces the recheck's figures for the
    old order. For S's D-READING these are Kish 23.7 / 22.4, with crow at
    0.111.
  * It then computes the new order:

    | | S's D-READING (planning) |
    |---|---|
    | argmin | N(-0.75, 1.6), was N(-0.875, 1.6) |
    | realised, all-item | -0.68 / 1.76 |
    | Kish n | 25.6 / 24.0 |
    | crow | 0.098 |

  * C's D-READING (42.6 / 27.5) and both D-FLATs are unchanged.
  * **The balance check on the dense sampler's weights.** D-READING's
    difference is 0.052 (S 0.166, C 0.114) for the family rule, and 0.030 for
    the critic's benchmark rule. Both pass 0.10. The 0.027 quoted before was
    read on the unrestricted weights; PE §5.1 now says so.
  * PE §5.3's planning table is updated, and `regimes` writes every Kish
    figure (PE §9 item 2).
* **The knob proxy and the realised level.** PE §5.3 now separates two
  quantities.
  * **The realised level** is read on evaluated responses
    (`testlike.describe`'s logit), as the targets are.
  * **The knob search** stays on the all-item pair logits, now from
    `levels.json`, as a cheaper proxy.
  * At the argmin, `regimes` re-draws CAL_SEED's 1,000 runs and measures
    them on evaluated responses. That is the CAL_SEED value, which "short of
    target" and the realisation check use.
  * Both realisations are written. The planning table is marked as proxy
    figures.
* **PE §2.** The docstring range is now the module docstring, lines 1-40,
  naming lines 13-14 and 62-64. impl_plan §11 is changed the same way.
* **impl_plan §13's heading.** It now reads "after the gate's commit on
  2026-10-15; due by 2026-10-20, within P2 §7's '13.10 to 20.10: findings;
  report R1 to R6'". The date is taken from `p2_final_plan.md` §7.
* **A4's date.** The text keeps 2026-10-12. impl_plan step 0.3 now says that
  if commit 1 lands on another day, the heading, the preamble's "committed
  on" and §4.5's first row change before the commit. PE does not hash A4.
* **The commit-1 filter.** PE §13 and impl_plan 0.3 exclude
  `eval_describe/pool_describe.json` (13,499 `pair_logit` values) and
  `nfix_orig/`. They add `recheck.md` and `nfix_log.md` to the folder's list.

## Changes that affect code

For the implementer. Each item names where the design now states it.

**Rule constants (`experiments/pool_eval.py` constants block)**

1. `RULE_TEXT` is the new A4 §4.4 block, byte for byte. It changed in (E1),
   (E2), the tie-break, (c), and Readings 1, 4 and 5.
2. `RULE`:
   * the twins' threshold is 0 in (E2) and in (c), separate from the +0.002
     of P-TUNED, P-MIXTURE and P-FLAT;
   * the twins are no longer in any +0.002 list;
   * the dense +0.001 is unchanged.
3. `select`:
   * X* is the earliest of C1, C1m, C2 in Q with M(X) <= min_Q M + 0.0005,
     inclusive;
   * (E2) checks the twins at D <= 0;
   * `gate` (c) does the same.
4. `REGIMES`: the 0.10 cap is applied after each sampler's restriction.

**Lock (`PINNED`, the `lock` stage)**

5. Add to `PINNED`:
   * `paiec/baselines.py` 64fe42f5a63d22e1;
   * `experiments/ship_confirm.py` ec6e73ea1e3e8935;
   * `experiments/subject_side.py` 888c11b0c1b13c05;
   * `data/p2_work/pool_design/benchmarks.csv`
     d88247c984edc5df4083a90d8659f71f1d51eeb41212b10f0eff72612501a405;
   * `data/p2_work/pool_design/pairs.csv`
     2e913b8b24c29fc86c94ea9894373a376a635bda82f4672f8d3d7ebb54efac6a;
   * `data/pool_eval/levels.json`.
6. `lock` computes the static in-repository import closure of
   `experiments/pool_eval.py` and `paiec/pool.py`: every `import` and `from`
   statement, at any depth. It refuses if the closure holds an unpinned file.
7. The lock check accepts a successor digest only for a file in a fix's scope,
   and only from a committed `results/pool_eval.json: fixes` record. For
   `paiec/pool.py` the check is an AST comparison with `load_eval_pairs` and
   its index-cache helpers removed. The constants block, `RULE_TEXT`, the
   bundles, the fits index, the compositions and `levels.json` have no
   successors.
8. `tools/build_submission.py --pool` accepts a recorded successor
   `paiec/pool.py` under the same AST rule. It must still rebuild the locked
   bundle byte for byte.

**Harness rules (`score`, the new `record-fix`)**

9. hier's fallbacks and every other finite output are scored and counted per
   arm and regime. They are never check failures and are never re-scored.
10. A non-finite prediction is recorded, and gives "archive-3 stays" for that
    arm's candidate (for every candidate if A3's arm produced it).
11. The exit-75 resident-set check runs between tasks, after the task's rows
    are written. It never kills a task, and a resume starts at the next
    unwritten task.
12. A new `record-fix` stage checks a fix's scope and writes the deviation
    record:
    * scope: `experiments/pool_eval.py` outside the constants block, plus
      `paiec.pool.load_eval_pairs` and its index cache;
    * the deviation record: cause, files, old and new sha256, affected tasks;
    * the author commits the record with the diff before any re-score;
    * an out-of-scope fix is refused, and recorded as "archive-3 stays" for
      the candidates it affects.
13. A re-scored task must record the same per-run bundle digest for every
    (excl, candidate) it recorded before the fix.
14. Keep `load_eval_pairs` and its index cache free of anything a fit or a
    bundle uses. The fits cache key hashes the source of `load_fit_pairs`
    and `items_of` only.

**Memory handling (`paiec/pool.py`, `fits`, `timing`)**

15. `items_of`:
    * reads `ITEM_COLUMNS = ("item_id", "content", "item_features")` only,
      through `pq.ParquetFile(path, buffer_size=READ_BUFFER,
      pre_buffer=False).iter_batches(batch_size=ITEM_BATCH,
      columns=ITEM_COLUMNS)`, with `READ_BUFFER` > 0 and `ITEM_BATCH` <= 256;
    * keeps the requested ids and drops each batch;
    * never reads `verifier` or `grading_criterion`;
    * never uses `read_table` with a filter.

    pyarrow's defaults (`buffer_size=0`, `batch_size=65536`) do not meet this.
16. A new `scales_of_mixed`: the descriptive scale report of the nine mixed
    benchmarks of L, by a separate batched read of `item_id` and
    `grading_criterion`. It writes counts only.
17. `fit_all` fits mind2web and swe_smith each in a fresh worker process. Each
    worker records `ru_maxrss` and the `process_time` of the item read and of
    the fit, and these go to `results/pool_fits.json`.
18. `timing` adds two formative-size TIMING_SEED runs from C's benchmarks,
    untilted:
    * the first pair is fixed: mind2web's one pair in one run, a swe_smith
      pair (from the run's stream) in the other;
    * each runs on a scoring worker, every arm, labels stripped;
    * peak RSS is recorded per run;
    * if a worker passes 1.45 GB or a call fails, the lock does not proceed
      until the cause is fixed.

**Weights and regimes (`regimes`)**

19. The order is:
    * the tilt t_b and the clip at 20 times the mean, both over the half;
    * then the restriction to the sampler's benchmarks (dense: at least 4
      eligible pairs);
    * then renormalisation;
    * then the water-filling cap at 0.10.
20. The balance check uses the dense sampler's restricted, capped weights.
21. `regimes` writes, for every regime and half:
    * the knobs;
    * the proxy and the evaluated-response realisations;
    * Kish n by benchmark and by family;
    * the largest weight share and the flagged weight share;
    * c(X, ·) - c(A3, ·) on the planned appearances, for each candidate, with
      the excl-empty builds (descriptive).
22. **The realised level.** The knob objective stays on the all-item pair
    logits from `levels.json`. At each argmin, the same 1,000 CAL_SEED runs
    are drawn again and measured on evaluated responses (`testlike.describe`,
    split seed 0, scope 'pair'). That is the CAL_SEED value. "Short of
    target" and the realisation check use it, and the scoring runs are
    measured the same way.

**Levels (`paiec/pool.py`, `split`)**

23. New `pair_counts(b)` and `levels(names)`. For each eligible pair, n
    (distinct items with a 0/1 response) and k (items whose first 0/1
    response in file order is 1), and the logit log((k + 0.5) / (n - k +
    0.5)). `level_cc` is the mean of the logits over the benchmark's eligible
    pairs.
24. `split` checks these against `pairs.csv` (n and k exactly, `logit_cc` to
    1e-12) and `benchmarks.csv` (`level_cc` to 1e-12); a mismatch is a check
    failure. It writes `data/pool_eval/levels.json` (benchmark, subject_id,
    n, k), and the split and the knob search read levels from it.

**RP3 (`reproduce`)**

25. RP3 runs `evaluate_timed` and `official.run_official(..., workers=1)` on
    label-randomised copies:
    * every 0/1 response is drawn from `default_rng([TIMING_SEED, 3, i])`;
    * the copies are built from the runs' pairs and that stream alone;
    * an assert checks that every response equals its draw;
    * `run_official`'s scores are discarded, never written or printed.

**Build path and model.py (after the gate)**

26. On a pass, the text that must be made true for X* is the module docstring
    of `submission/model.py`, lines 1-40 (lines 13-14 included), and the
    comment above `LEVEL`, lines 62-67 (lines 62-64 included). The AST rule
    is unchanged.

**Tests (`tests/test_pool.py`, `tests/test_pool_eval.py`)**

27. New tests:
    * `items_of` requests exactly `ITEM_COLUMNS`, with bounded batches and a
      positive buffer, on a synthetic table that has `verifier` and
      `grading_criterion` and several row groups;
    * `pair_counts` and the levels;
    * levels against the CSVs, and that a mismatch raises;
    * the cap after the dense restriction;
    * the balance check on dense weights;
    * the realisation on evaluated responses;
    * the twins' D <= 0 binding in (E2) and (c), where +0.0001 fails and 0
      passes;
    * the tie-break: C1 -0.0011, C1m -0.0012, C2 -0.0015 gives C1, and the
      inclusive margin;
    * RP3's copies, and that no RP3 output holds a Brier or an ALC;
    * the import-closure refusal;
    * the successor accepted for a `load_eval_pairs` change and refused for a
      `bundle` change;
    * the resident-set check after the rows are written.

**Estimates**

28. impl_plan's estimates rise:
    * `pool.py` from 4 to 4.5 h;
    * `pool_eval.py` from 7.5 to 8 h;
    * `test_pool_eval.py` from 2.5 to 3 h.

    So the work before the lock rises from about 18 h to about 19.5 h, on
    2026-10-12 and 2026-10-13.

## Not changed, and open

* **The recheck's recommended items are not applied** (they lie outside
  N1 to N9), except the point-estimate note in PE §7. Still open:
  * a descriptive HELM-suite cut;
  * a precedence clause that freezes the rule once a scored row exists;
  * pinning `PYTHONHASHSEED=0` and one BLAS thread for the `--pool` build.
    This one affects code if adopted.
* **`dates_log.md` is left as the record of the calendar revision.** Its
  "after" digests are now superseded by the table above.
* **`survey.json` (for the author).** It holds per-benchmark lists of
  `canon_names`, one entry per eligible pair (948 for afrieval). There are no
  outcomes in them, but they are per-pair in shape. PE §13 allows only
  outputs "that hold no per-pair row". Excluding the file from commit 1 costs
  nothing.
* **For the author to confirm in commit 1**, beside `recheck.md`'s list:
  * decision 7 (option (a)), which came from the task text;
  * the narrower fix scope (`items_of` excluded);
  * the non-finite-output rule;
  * the CAL_SEED value on evaluated responses, with the knob search on the
    proxy;
  * the new D-READING (S) knob, N(-0.75, 1.6) planned;
  * the about 19.5 h estimate.

## Checks run

* **Superseded phrases are gone** from all three files:
  * "separate(s) the level", "pyarrow filter", "lines 1-8", ":1-8)";
  * "due by 2026-10-15", "each RS test-like regime", "minimises M over Q";
  * "twins as loss bounds", "set no parameter of it";
  * the old PE digest.
* **Every twin mention** in `RULE_TEXT`, A4 §4.3, PE §5.3's table and impl_plan
  reads D <= 0. Every +0.002 bound names only P-TUNED, P-MIXTURE, P-FLAT and
  the RS test-like regimes.
* **`RULE_TEXT`'s width.** Its longest line is the 79-character header line,
  which is unchanged; the edited lines are at most 76 characters, and there
  are no tabs.
* **`dense_cap.py`** reproduces PE's cap-first dense figures exactly: S
  34.0 / 30.4 and 23.7 / 22.4; C 73.0 / 29.4 and 42.6 / 27.5.
* **The digests quoted for new lock entries** were computed on disk and equal
  `git show HEAD:<path>`.
