# Amendment 4: implementation plan (revised for the final design)

Written 2026-10-07 against repository HEAD a725df9; revised the same day for
the final design, on 2026-10-11/12 (HEAD 2bfcd17) for A4's calendar and the
finished item fetch (`dates_log.md` lists the changes), and on 2026-10-11 for
the re-check's N1 to N9 (`recheck.md`; `nfix_log.md` lists the changes), and
on 2026-10-11 again for the verification's V1 to V13 (`nfix_verify.md`;
`vfix_log.md` lists the changes; its section "Changes that affect code"
merges the verification's items 29 to 41 with the new ones, 42 to 58, and
says which of nfix_log's items 1 to 28 they amend; together they list what
the code must follow). It implements
`amendment4_short.md` ("A4", to be pasted into
`docs/plans/p2_final_plan.md` as amendment 4) and `p2_pool_eval.md` ("PE", to
be committed as `docs/plans/p2_pool_eval.md`). Section numbers such as PE §4.3
are that file's. It replaces the plan written for `amendment4_draft.md`, which
the critic's R1 to R12 superseded.

**Conventions.**

* **Estimates** are working time for one Claude Code session, plus wall time
  for runs.
* **Commit points** are marked **[commit]**. The author makes every commit, with
  the repository's local address. Commits 1 to 4 are pushed to the public
  repository when made (A4 decision 8, PE §9 item 10), so that the lock commit
  is an outside time stamp.
* **Ownership.** Every result file is written by the script named beside it.
* **Never committed:** files under `data/` and `dist/`, `submission/prior.json`,
  any CSV or pickle, any per-pair JSON, any item text.
* **Item text** is read at run time only, never written: not in a cache, a row,
  a log or a test fixture. Caches and rows hold item ids and sha256 digests.

## 0 Before any code (the author; about 45 min)

1. **The item fetch is done** (checked 2026-10-11 by
   `pool_design/a4final/items_present.py`, recorded in PE §3.1 and §3.7).
   `data/pool/manifest.json` (`fetched_utc` 2026-10-08T04:04:26Z, sha256
   `ff548206…`) holds the four tables for 250 of the 255 entries; the item
   tables of cseo_bench (ordinal, 5.69 GB) and qatch (continuous, 0.71 GB),
   neither in L, were skipped with `--items-skip` (committed at 2bfcd17), and
   the three withheld entries have a catalogue row only. The benchmarks,
   subjects and responses entries equal the pre-items manifest's
   (`data/p2_work/pool_manifest_no_items.json`, a58a23fb…) for all 255; every
   item file of L is present, and all 548 files of L match the manifest in size
   and sha256 on disk. Free disk: 11.24 GB on 2026-10-11 17:53 UTC. PE changed
   again for the re-check's N1 to N9 and for the verification's V1 to V13, so
   its sha256 in A4 was recomputed (now `5af21af6c818d8a4…`).
2. **Confirm the open choices**, which A4 and PE set as defaults (the list is
   in the summary that came with them, and in `recheck.md`'s and
   `nfix_verify.md`'s "Decisions the author must confirm"): Mode A; (E1) as a
   mean plus P1a's bounds; C2's choices; the family-whole split and family
   clusters; C1m and its matching set (all of E, mu0 -2.035 planned; dated
   subjects only would give -1.891); C3 dropped; the twins as no-loss
   conditions, D ≤ 0 in (E2) and (c) for every candidate (the re-check's N1,
   its option (a); A4 decision 7); the 0.10 weight cap, applied after the dense
   restriction; the App G.6 wording with PE §11's C1m clause (the re-check's
   N4); the scope of a post-lock fix (PE §9 item 11: only `pool_eval.py`
   outside its constants block and the definitions `paiec.pool.EVAL_LOADER`
   names, frozen at the lock; `items_of` outside it); invalid outputs (not one
   finite float in [0, 1], or the wrong shape) recorded, never raised, giving
   "archive-3 stays" for their arm's candidate and no other, or for every candidate if A3's
   (the verification's V5, V6); the precedence clause (A4's preamble, PE's:
   nothing in the rule, candidates, split, regimes, seeds, counts, statistics
   or fix scope changes once a scored row exists, not even by a later
   amendment; V7) and the mechanical re-score rule (PE §9 item 11, with
   "archive-3 stays" if a re-run `select` names another X*; V8); the CAL_SEED
   value on evaluated responses with the knob search on the proxy, and S's
   D-READING knob N(-0.75, 1.6) planned; the dates (the calendar of 2026-10-11,
   A4 §4.5) and the dense counts of 20 (S) and 40 (C) it sets (PE §5.4, §12),
   with about 20.5 h of code before the lock; commits 1 to 4 pushed to the
   public repository when made (A4 decision 8; the approval date, 2026-10-06,
   came with the task text of the verification fixes, so the author confirms
   it); and the commit-1 filter (below).
3. **[commit 1]**, on 2026-10-12: A4 pasted at the end of
   `docs/plans/p2_final_plan.md`; `docs/plans/p2_pool_eval.md`;
   `docs/plans/pool_design/` (PE §13's list: the notes, `recheck.md` and
   `nfix_log.md`, `nfix_verify.md` and `vfix_log.md` among them, the scripts,
   their JSON and text outputs without per-pair rows; never
   `eval_describe/pool_describe.json`, which holds 13,499 per-pair logits, nor
   the pre-fix copies in `nfix_orig/` and `vfix_orig/`; the author also decides
   on `survey.json`, whose `canon_names` lists run one entry per pair, and on
   `code/c1_proto_prior.json`, a full prior bundle fitted on the whole pool,
   the kind of file the repository never commits). No code yet. It fixes the
   rule before any comparison. A4's heading date and its preamble's "committed
   on" date must be the commit's real UTC date: if commit 1 lands on another
   day, change both (and §4.5's first row) before committing. PE does not hash
   A4, so no digest changes; the lock hashes A4 as committed. Push commit 1
   when it is made.

## 1 Keep archive-3 at hand (10 min)

1. Copy `dist/paiec.zip` to `dist/archive-3.zip`; check sha256
   `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090` in full
   (the lock records it, PE §9 item 10). If it is not archive-3, rebuild
   archive-3 at a725df9 with `python tools/build_submission.py` first.
2. Extract its `prior.json` to `data/pool_eval/a3_prior.json`; check sha256
   `c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770`.
3. Both later steps that write `dist/paiec.zip` and `submission/prior.json`
   (step 12) would otherwise destroy the comparator.

## 2 `paiec/pool.py`, the offline pool module (4.5 h)

Not in `PACKAGE` (`tools/build_submission.py:61`); no shipped module imports it.
Its base is `pool_design/proto_loader.py`, the arithmetic of
`proto_build.py:129-150` and `pool_design/a4final/families.py`.

* **Constants.** `REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"`,
  `FLAGGED` (the nine), `SHARE01_MIN = 0.95`, `K = 100`, `M_P = 2000`,
  `MIN_ITEMS = 80`, `DATE_SHIFT = 1.25`, `MCQ_RE` (PE §3.7); `ITEM_COLUMNS =
  ("item_id", "content", "item_features")`, `ITEM_BATCH` (at most 256 rows)
  and `READ_BUFFER` (> 0 bytes; PE §3.2, the re-check's N8).
* **`verify(path)`.** Size and sha256 against `data/pool/manifest.json`, once per
  process and file; raises on a mismatch or a missing entry. Every reader calls
  it before opening a pool file.
* **`select(manifest)` -> (L, U).** PE §3.3: the revision pin; release,
  granularity, response_type and direction from each catalogue row; the 0/1
  share among non-null responses (cached); at least one eligible pair.
  Expected 137 and 128.
* **`families(names)`.** PE §3.5, the rule of `a4final/families.py`; must give
  the same table (230 families; 8 with two or more members in E).
* **`excl_for(B_run)`.** Every benchmark of L in a family of B_run.
* **`items_of(b, item_ids)`.** Mode A item dicts (PE §3.2): `content` and
  `item_features` from `items.parquet`, read as
  `pq.ParquetFile(path, buffer_size=READ_BUFFER,
  pre_buffer=False).iter_batches(batch_size=ITEM_BATCH, columns=ITEM_COLUMNS)`,
  keeping the rows whose item_id is requested and dropping each batch after
  use; never `verifier` or `grading_criterion`, never a whole-table or
  filtered `read_table` (the re-check's N8: mind2web's larger row group holds
  1.84 GB of uncompressed `content`, swe_smith's `verifier` row groups up to
  1.19 GB). `interactors` from the item's first 0/1 response row in file
  order, benchmark-wide. Text stays in memory. It feeds the fits, so it is
  outside a post-lock fix's scope (PE §9 item 11).
* **`pair_counts(b)` and `levels(names)`** (the re-check's N3). For every
  eligible pair, n (distinct items with a 0/1 response) and k (items whose
  first 0/1 response in file order is 1); the pair logit
  log((k + 0.5) / (n - k + 0.5)); `level_cc` the mean over the benchmark's
  eligible pairs. From the manifest-verified responses, no item text.
* **`scales_of_mixed(names)`.** The descriptive scale report of the nine mixed
  benchmarks of L (PE §3.2): a separate batched read of `item_id` and
  `grading_criterion` only; counts written, no text; never through
  `items_of`.
* **`load_fit_pairs(b)`.** Named-column reads; the first 0/1 response per
  (subject_id, item_id) in file order; at least 80 distinct items; the per-pair
  item cap M_P in the order `stable_hash('pool-item', b, item_id)`; the
  round-robin pair cap K (PE §3.4); subjects as `paiec.data.SUBJECT_FIELDS` plus
  `_sid`, `display_name` never read; Mode A items. Returns `Pair` objects.
* **`load_eval_pairs(b, subject_ids)`.** The same reading with every 0/1
  response kept, in file order, uncapped, for the drawn pairs only. An index
  cache (npz, ids and offsets only) under `data/pool_eval/index/`, used by
  this function alone. With that cache it is the only part of `pool.py` a
  post-lock fix may change (PE §9 item 11), so keep it and its helpers free
  of anything a fit or a bundle uses.
* **`EVAL_LOADER`** (the verification's V4): the tuple of the definitions a
  post-lock fix may change, exactly `load_eval_pairs` and its index-cache
  helpers (the draft: `INDEX_VERSION`, `_eval_index_path`, `eval_index`,
  `_eval_rows`, `load_eval_pairs`, `build_eval_indexes`, `_eval_index_task`).
  The tuple is not in itself, so `frozen_ast` keeps it and a changed tuple
  fails the AST rule. The lock records it.
* **`check_csvs`**: tolerance 1e-12 for `logit_cc` and `level_cc`, as PE §5.1
  says (the draft has 1e-9; the CSVs agree with the formula to 0 and
  5.8e-15).
* **`fit_all(names, jobs, cache)`.** `benchmark_fit(ps)` for multi-pair
  benchmarks in worker processes (one BLAS thread); T_full; the fixed-T pass
  `benchmark_fit(ps, s2e=T_full)` for every benchmark of L. mind2web and
  swe_smith each run in a fresh worker process, which records its peak
  resident set (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) and the
  `time.process_time` of its item read and of its fit (the re-check's N8).
  Cache under `data/pool_fits/`, keyed by (REVISION, benchmark, K, M_P, s2e,
  sha256 of the source of `load_fit_pairs` and `items_of`); each entry holds
  t, t_var, mu, mu_var, s2_t, item_var, share, keys, convergence flags,
  iterations, CPU time, and no item text. An index file lists every entry's
  sha256.
* **`bundle(fits, excl, cand)`**, PB of PE §4.1: rows from U ∖ excl;
  `PR.from_standings(rows, 2.0, sorted(L ∖ excl), sorted(excl), None)`;
  fit_hyper's arithmetic with fit_hyper's share rule; th2 clipped to
  [0.01, 0.2] and id_cap 0.1; the level by `cand`: "literal" (LEVEL, C1),
  "matched" (mu0_m, C1m), "eb" (C2), or "fit" (fit_hyper's free levels, RP4
  only). Returns (prior, hyper, report).
* **`b0_centre(prior, hyper, pairs_by_benchmark)`** and **`match_mu0(...)`**:
  PE §4.2's functional and C1m's bisection; base `a4final/b0_dates.py`.
* **`eb(levels, var)`.** PE §4.3's iteration (from `code/code_levels.py:68-89`).
* **`CANDIDATES`** = {C1: "literal", C1m: "matched", C2: "eb"}, imported by
  `experiments/pool_eval.py` and `tools/build_submission.py`.
* **`write_lists()`.** `tools/pool_benchmarks.json` (revision, rule text, L, U,
  FLAGGED, E) and `tools/pool_families.json` (family of each benchmark, joining
  keys). Names and catalogue keys only.
* **`main()`.** A report like `prior.main`'s.

## 3 `tests/test_pool.py`, on synthetic `formatted_tables` (2 h)

Tiny parquet tables in `tmp_path` (pyarrow); synthetic item text only.

* **Verification:** a changed byte or a missing manifest entry raises.
* **Responses:** the first 0/1 in file order; nulls and other values dropped;
  min_items.
* **Caps:** the per-pair item cap keeps a sparse benchmark's pairs and equals a
  benchmark-wide cap on a dense one; the round-robin order is deterministic,
  puts the same names first on two benchmarks, and gives a name with several
  pairs one slot per round.
* **Selection:** binary kept; mixed kept at a share of 0.95 or more, dropped
  below; other types dropped; flagged in L and not in U; the revision pinned.
* **Families:** a shared source key or paper joins; arXiv versions and abs/pdf
  collapse; `excl_for` returns whole families.
* **Items:** content, features and interactors as PE §3.2; `display_name` never
  read; a missing column raises; no item text in any cache file written. A
  synthetic `items.parquet` with `verifier` and `grading_criterion` columns and
  several row groups: `items_of` requests exactly `ITEM_COLUMNS` (a spy on
  `ParquetFile.iter_batches`), batches of at most `ITEM_BATCH` rows, a
  positive `READ_BUFFER`, and returns the dicts a whole-table read would.
* **Levels:** `pair_counts` gives n and k on synthetic responses (first 0/1 per
  item in file order; nulls and other values dropped; at least 80 items), and
  the logits and `level_cc` follow the formula.
* **`bundle`:** with `cand="fit"` it equals `PR.build(pairs, excl)` on
  synthetic pairs (the synthetic RP4); exclusion lists consistent;
  `HierPredictor` accepts every candidate; C1m's mu0 matches the target
  centre to 1e-6; `eb` matches a hand-computed case.
* **`CANDIDATES`** equals PE §4.3's table.

`pytest` stays green from the repository root.

## 4 The `fits` stage (45 min of work; 30 to 60 min wall)

`python -m paiec.pool --fit --jobs 4`, with `PYTHONHASHSEED=0` and BLAS
pinned. It writes `results/pool_fits.json` (PE §9 item 3):

* the manifest sha256 and the verification of every file of L;
* per benchmark of L: pairs and items kept, item_var, share and keys, EM and
  Newton convergence, iterations, CPU time, the free and fixed-T levels
  (descriptive; never in a bundle);
* T_full, the group share, degenerate fits (item_var > 100);
* the B0-centre table of PE §4.2 for A3, C1, C1m, C2 (real and +1.25 years;
  all, dated, undated), mu0_m, and its run-level recomputation for 50 runs a
  half (after `regimes` has drawn them; descriptive);
* C2's literal and the size of its `mu_var` omission;
* the R4 slopes on the final fits (record only; C3 stays dropped);
* the leave-one-benchmark-out ridge shifts (largest coefficient and E-score
  change per benchmark of U);
* the mixed benchmarks' declared item scales (descriptive; `scales_of_mixed`);
* for mind2web and swe_smith, each fitted in a fresh worker: peak resident set
  and the CPU time of the item read and of the fit (the re-check's N8);
* the floor-active share per benchmark of E and the catalogue MCQ flag;
* the excl-empty bundles of C1, C1m and C2 in
  `data/pool_eval/bundles/archive_<X>.json`, with their sha256.

Expected: about 1,800 CPU s (the prototype's 900, doubled for the group
effects that items bring; re-measured).

## 5 `experiments/pool_eval.py` (8.5 h)

Pattern: `experiments/regime_sensitivity.py`. It imports
`regime_sensitivity.TARGETS`, `MIX`, `KNOB_SCALE`, `REALISE_TOL` (file digest
checked) and `testlike.describe` and `testlike.shifted`.

* **Constants block.** `RULE_TEXT` (A4 §4.4, byte for byte; the lock checks it
  against the plan file) and `RULE` (the thresholds, each checked to appear in
  the text; the twins' threshold is 0 in (E2) and (c), separate from the
  +0.002 of P-TUNED, P-MIXTURE and P-FLAT; the re-check's N1); `CANDIDATES`
  from `paiec.pool`; `SPLIT` (salt, flagged blocks 3 with 1 to S, other
  blocks 8 with 3 to S, family units); `REGIMES` (targets, grids, the cap
  0.10 applied after each sampler's restriction, the twins, the dense shape);
  seeds 4001 to 4004, 11, 0; counts (S 80/20, C 120/40, RS 520; PE §5.4); the
  adjustment rule (the dense item cap 120 to 80); `PINNED`, every digest of
  PE §9 item 10 (now with `paiec/baselines.py`, `experiments/ship_confirm.py`,
  `experiments/subject_side.py`, `experiments/__init__.py`,
  `benchmarks.csv`, `pairs.csv`, `data/pool_eval/levels.json`, the
  streaming client's digest with its clone's commit, the RS rows digest, and
  `dist/archive-3.zip` and `a3_prior.json` in full; the verification's V1 to
  V3); `classify_output` (V6: valid is one finite float in [0, 1], shape (1,)
  after `numpy.atleast_1d`; anything else is invalid).
* **`evaluate_timed`.** A copy of `level_calibration.checkpoints` and `evaluate`
  (:508-561) that records `time.process_time` and `time.perf_counter` per call
  (R7). RP3 checks it. It does not copy `evaluate`'s `raise
  ValueError("invalid prediction")` (:546-547): it classifies every output
  with `classify_output`, records an invalid one (arm, run, budget, call
  index), never raises on it, and scores the task's other arms on; the
  invalid arm is not scored on that run (PE §5.2, §9 item 11).
* **Stages** (`python experiments/pool_eval.py <stage>`):
  * `split`: `paiec.pool.levels` for every eligible pair of E, checked
    against `pairs.csv` (n and k exactly, `logit_cc` to 1e-12) and
    `benchmarks.csv` (`level_cc` to 1e-12), written to
    `data/pool_eval/levels.json` (benchmark, subject_id, n, k); the
    family-unit split from those levels; reproduces `a4final/split_v2.json`'s
    family halves and `tools/pool_families.json`.
  * `regimes`: the weights (tilt over the half, then the sampler's
    restriction, then the 0.10 cap); the knob search on CAL_SEED (1,000 draws
    a setting) on the all-item proxy from `levels.json`; the planned
    compositions of every half and regime to
    `data/pool_eval_rows/<half>/_compositions.json` (the twins reuse
    READING's and AUDIT's); at each argmin the same 1,000 CAL_SEED runs drawn
    again and their realised level measured on evaluated responses with
    `testlike.describe` (the CAL_SEED value); short of target and the
    realisation checks on that quantity; the balance check on the sampler's
    weights (a failure stops the lock; PE §5.1); for every regime and half,
    the knobs, both realisations, Kish n by benchmark and by family, the
    largest and the flagged weight share, and c(X, ·) - c(A3, ·) on the
    planned appearances (PE §4.2, §9 item 2).
  * `reproduce`: RP1 to RP5 (PE §9). RP1 also computes the digest of the 520
    RS row files (PE §9 item 10: sha256 of the sorted `"<path>\t<sha256>\n"`
    lines, paths relative to `data/regime_sensitivity_rows/`; the planning
    value is `a4final/rs_rows_digest.json`'s). RP3 (V11) on two TIMING_SEED
    runs, run 0 formative-size and run 1 dense, and run 0's twin, all on
    label-randomised copies: run i's 0/1 responses, in the run's pair order
    and then file order, replaced by the successive `rng.integers(0, 2)` of
    `rng = default_rng([TIMING_SEED, 3, i])`; the twin is run 0's copy with
    shifted dates; the copies built from the runs' pairs and that stream
    alone; an assert that every response equals its draw; each arm's
    predictions captured through a recording factory (a wrapper that logs
    every call's prediction by checkpoint and call order), because
    `run_official` returns scores, not predictions; `run_official`'s
    returned dict discarded, never written or printed (the re-check's N6).
  * `timing`: S's benchmarks as before, plus two formative-size runs from C's
    benchmarks whose first pair is fixed (mind2web's one pair; a swe_smith
    pair from the run's stream), on a scoring worker, on label-randomised
    copies (V12): run i's responses replaced as in RP3, from
    `default_rng([TIMING_SEED, 9, i])`, i in the fixed order S formative-size
    0 to 5, S dense 6 and 7, mind2web 8, swe_smith 9; an assert that every
    response equals its draw and that no Brier or ALC is computed; peak
    resident set per run; a run whose worker passes 1.45 GB stops the lock
    until the reader is fixed (the re-check's N8); the projection; the
    adjustment.
  * `lock`: every digest of PE §9 item 10 (with V1 to V3's: the RS rows
    digest equal to RP1's, the streaming client's digest and its clone's
    commit), mu0_m, the C2 literal and the tuple `paiec.pool.EVAL_LOADER`
    (V4), `PYTHONHASHSEED=0` and BLAS checked, the time stamp; the static
    in-repo import closure of `experiments/pool_eval.py` and `paiec/pool.py`,
    refused if it holds an unpinned file (the re-check's N3); the lock check
    accepts a successor digest only for a file in a post-lock fix's scope,
    from a committed `fixes` record (PE §9 item 11); at the first `score`
    the lock commit's time is read and checked against the hard stop,
    2026-10-13 23:59 UTC (a later lock is recorded as "not locked in time",
    PE §10; the old lock-slip rule is gone, PE §5.4).
  * `score --stage S|RS|C|C-extra --shard k/3`: rows to
    `data/pool_eval_rows/<half>/<regime>/<i>.json` (per appearance: benchmark,
    family, kind, flag, catalogue MCQ flag, HELM-hosted flag (PE §6),
    subject_id, identity key, named and row flags, dated, release year, kept
    and evaluated items, responses, evaluated rate and logit, v; per arm: Brier
    and ECE at six budgets, mean B0 prediction; per arm and run: calls, the
    sums and maxima of CPU and wall per call, and for a dense run every call's
    CPU time in call order (V10), fallbacks, invalid outputs, the bundle's
    digest and hyperparameters; per task: resident set, digests, load average).
    Bundles cached under `data/pool_eval/bundles/` keyed by sorted excl and the
    candidate's digest. Each stage re-runs the lock check, exits with 75 above
    1.45 GB, checked between tasks after a task's rows are written (never
    killing a task), resumes. hier's fallbacks and every other valid output are
    counted per arm and regime and scored as they occur; they are never check
    failures. An invalid output is recorded, never raised, and gives "archive-3
    stays" for that arm's candidate and no other, or for every candidate if
    A3's arm produced it; the other arms stay eligible and (E4) does not
    require the failed arm on that run (B3) (the re-check's N5; V5, V6; PE §9 item 11).
  * **The check log** (V10), in every stage before and after the lock: each
    check appends (stage, name, result with the failing values, UTC) to
    `results/pool_eval.json: checks`; `select` reads (E4)'s "every check
    passed" and `gate` (g)'s from it; a check resolved by a recorded fix
    counts once its re-run passes and is logged (PE §9).
  * `record-fix` (after the lock, only after a check failure): checks a fix's
    scope (only `experiments/pool_eval.py` outside the constants block, and
    the definitions the locked `EVAL_LOADER` tuple names; `frozen_ast` of
    `pool.py` equal to the locked one, the tuple included; every other
    locked digest unchanged); runs the locked `tests/test_pool.py` and
    `tests/test_pool_eval.py` unchanged and requires both green; lists the
    tasks to re-score by rule (V8): the failed tasks, plus run 0 and the last
    run scored before the fix of every regime of every stage with rows,
    which must reproduce their stored rows bit for bit outside the timing
    and resource fields, or else every task of that stage; writes the
    deviation record (cause, files, old and new sha256, those tasks, the
    test result) to `results/pool_eval.json: fixes`, which the author
    commits with the diff before any re-score. An out-of-scope fix is
    refused and recorded as "archive-3 stays" for the candidates it affects
    (PE §9 item 11). After the re-score, `select`, `summarise` and `gate`
    are re-run from all rows; a re-run `select` that names another X* than
    the committed selection gives "archive-3 stays".
  * `select`: re-verifies the RS rows digest immediately before it reads a
    row (V2); (E1) to (E4), (E3)'s maximum over every call of X on S's
    formative-size runs and the RS tasks, (E4)'s arms A3, C1, C1m, C2 on pool
    tasks and C1, C1m, C2 on RS tasks, and its checks from the check log
    (V9c, V9d, V10); M(X), D on S; Q, X* (R10: the earliest member of Q, in
    the order C1, C1m, C2, with M within 0.0005 of Q's minimum, inclusive),
    to `results/pool_eval.json: selection`. Refuses unless every S and RS
    task exists.
  * `summarise` and `gate`: PE §6's statistics (family clusters; PB with each
    appearance weighted 1/(pairs in its run), V9a), (a) to (g) read once
    ((f): X*'s calls, the p99 `numpy.percentile(x, 99)`, linear, over all of
    X*'s dense calls in the C core from the rows' per-call lists, V9b; (g)'s
    checks from the check log), to `results/pool_eval.json: summary, gate`;
    on a pass also `tools/pool_build.json` (X*, its locked bundle sha256 and
    literal).
  * `describe-hidden`: refuses before the gate is in a commit (`git log` for
    the gate's results digest); reads `results/hidden_runs.json` once (PE §11
    item 4).
* **Guards in code:** no stage before `describe-hidden` opens
  `results/formative*` or `results/hidden_runs.json` (a test patches `open`);
  every container that orders output is iterated sorted.

## 6 `tests/test_pool_eval.py` (3.5 h)

On synthetic pools:

* **Sampler:** half-only benchmarks; distinct pairs and dicts; at least 80 kept
  items; the 1,000 cap; reproducible; uniform weights without a tilt; the twin
  equals its parent run but for the shifted dates.
* **Dense sampler:** 7 benchmarks, at most 8 pairs each, at most 120 items each.
* **Split:** families kept whole; flagged blocks of 3; a toy catalogue; levels
  from `levels.json` equal to the CSVs' on a toy pool, and a mismatch raises.
* **Weights:** the water-filling cap; the cap applied after the dense
  restriction (a toy half where the cap before it leaves a share above 0.10);
  the balance check read on the dense sampler's weights; the knob objective on
  the proxy, and the CAL_SEED value and the realisation check on evaluated
  responses (`testlike.describe`).
* **Statistics:** D = Σv/Σn; PB and JK by family, PB with the appearance
  weight 1/(pairs in its run), on a toy table where the unweighted mean
  gives another sign (V9a); the two-way SE on synthetic crossed clusters
  against a direct computation; (f)'s p99 equal to
  `numpy.percentile(x, 99)` with linear interpolation on a toy list (V9b).
* **`select` and `gate`** on hand-built tables: every condition binds; the
  twins' D ≤ 0 binds in (E2) and (c) (+0.0001 in P-READING-DS fails, 0
  passes); the tie-break acts only within Q; an argmin of -0.0013 with an
  earlier candidate at -0.0009 carries the argmin (R10); C1 -0.0011, C1m
  -0.0012, C2 -0.0015 gives C1 (two earlier members inside the margin, the
  first wins); C1 at exactly the minimum + 0.0005 is taken (inclusive); the
  -0.001 threshold holds.
* **Invalid outputs** (V5, V6): `classify_output` on NaN, inf, -0.01, 1.01,
  a pair of values and a valid 0.3; `evaluate_timed` with a toy arm that
  returns NaN records it, does not raise and scores the other arms; the
  candidate's outcome is "archive-3 stays", and every candidate's when the
  toy A3 does it.
* **RP3 and `timing`:** the label-randomised copies equal their stream draws
  (`[TIMING_SEED, 3, i]` and `[TIMING_SEED, 9, i]`, pair order then file
  order); the twin reuses its parent's copy; the recording factory gives the
  same predictions as `evaluate_timed` on a toy run; no RP3 or `timing`
  output holds a Brier or an ALC.
* **Lock and fixes:** an unpinned module in the import closure is refused,
  and `experiments/__init__.py` is in the closure and pinned (V1); the RS
  rows digest follows its rule on a toy directory, and a changed byte in one
  row makes `select` refuse (V2); a changed streaming-client digest fails the
  lock check (V3); a successor digest for `pool.py` is accepted when only an
  `EVAL_LOADER` definition changed and refused when `bundle` changed or the
  tuple changed (V4); a changed constants block is refused; `record-fix`
  lists the failed tasks plus the verification set, and every task of the
  stage when a toy verification row differs outside the timing fields, and
  refuses when a locked test fails (V8); a re-run `select` naming another X*
  gives "archive-3 stays".
* **Check log:** a failed check makes (E4) and (g) fail; a check resolved by
  a recorded fix passes once its re-run is logged (V10).
* **Resumability:** the resident-set check runs after a task's rows are
  written, and a resumed stage starts at the next unwritten task.
* **`RULE_TEXT`** equals the plan file's block and contains every number of
  `RULE`.
* **No feedback before the gate.**

## 7 Pre-lock runs (2 h wall, mostly unattended)

1. `split` (seconds); 2. `regimes` (about 30 min on 3 processes); 3.
`reproduce` (about 20 min; RP1 at 1e-9); 4. `timing` (about 25 min, with the
two C runs of mind2web and swe_smith; the adjustment if the core projects
above 30 process-hours); 5. `pytest` fully green with one BLAS thread.

## 8 The lock (the author; 20 min)

* `python experiments/pool_eval.py lock` writes `results/pool_eval.json: lock`.
* **[commit 2]**: `paiec/pool.py`, `tools/pool_benchmarks.json`,
  `tools/pool_families.json`, `experiments/pool_eval.py`, `tests/test_pool.py`,
  `tests/test_pool_eval.py`, `results/pool_fits.json`, `results/pool_eval.json`
  (pre-lock stages, the check log and the lock). Pushed when made (A4
  decision 8): this commit is the outside time stamp.
* **Deadline:** hard stop 2026-10-13 23:59 UTC (A4 §4.5); a lock after it
  means archive-3 stays. The dense counts are 20 (S) and 40 (C) wherever the
  lock lands (PE §5.4, §12); there is no lock target and no lock-slip rule.

## 9 Selection: S and the RS guard (about 5 h wall on 3 shards at PE §12's 1.5x figure, 15 h at three times it; 30 min attended; 2026-10-14)

1. `score --stage S --shard k/3` for k = 1, 2, 3 in the background, then
   `score --stage RS --shard k/3`. Watch the resident set and the load.
2. `select` writes the selection record.
3. **[commit 3]**: `results/pool_eval.json: selection`, before any CONF_SEED
   task, and no later than 2026-10-15 23:59 UTC (in practice by the end of
   2026-10-14, so that C fits). Pushed when made.

## 10 Confirmation and the gate (about 3.6 h wall at the 1.5x figure, 11 h at three times it; 30 min attended; 2026-10-15)

1. `score --stage C --shard k/3`.
2. `summarise`, then `gate`, read once.
3. **[commit 4]**: `results/pool_eval.json: summary, gate`, and
   `tools/pool_build.json` on a pass, by 2026-10-15 23:59 UTC. Pushed when
   made.
4. Descriptive, afterwards: `score --stage C-extra` (about 1.5 h) and
   `describe-hidden`. **[commit 5]**.

## 11 The `--pool` build path (3 h, written during step 9 or 10)

It does not depend on the outcome and builds nothing until the gate passes.

* **`tools/build_submission.py`:** `--pool [--candidate X]` refuses to run
  unless `PYTHONHASHSEED=0` and one BLAS thread are set, checked as the lock
  checks them (the re-check's recommended item; PE §10); reads
  `tools/pool_build.json`; checks `paiec/pool.py` (or a successor recorded
  under PE §9 item 11, by its AST rule) and the fits cache index against the
  lock; rebuilds the excl-empty bundle through
  `paiec.pool.bundle(fits, (), CANDIDATES[X])` and refuses unless it equals the
  locked bundle byte for byte; for C1m and C2 requires model.py's `LEVEL` to be
  the locked literal and prints it; `_check_bundle` (:155-179) compares
  `included` with L of `tools/pool_benchmarks.json`, and id_cap, sigma_theta
  and the level with X*'s locked values; checks the bundle's keys equal
  archive-3's with no per-benchmark level; checks the archive differs from
  `dist/archive-3.zip` only in `prior.json` and `submission/model.py`, and that
  model.py's AST, ignoring docstrings, comments and the `LEVEL` value, equals
  archive-3's (R6); keeps `check_run` on the legacy session; updates its
  docstring (:1-45).
* **`submission/model.py`** (on a pass only): the module docstring (:1-40,
  including :13-14 on the build writing LEVEL in) and the comment above
  `LEVEL` (:62-67, including :62-64) rewritten true for X* (PE §2, §10);
  `LEVEL` for C1m or C2.
* **`tests/test_submission.py`:** `toy_bundle` (:130-140) and
  `test_archive_ships_the_calibrated_hier` (:235) accept a pool bundle when
  `submission/prior.json` is one; a new test of the AST rule on a synthetic
  docstring change and a synthetic code change.

## 12 On a pass: archive-4 (2026-10-16 to 2026-10-17)

1. `python tools/build_submission.py --pool` (10 min); copy `dist/paiec.zip` to
   `dist/archive-4.zip`; record its sha256.
2. `python tools/stress_archive.py --archive dist/archive-4.zip --no-rebuild
   --out results/stress_archive4.json` (about 30 min); S3's gate read pooled,
   the per-part figures beside it.
3. Amendment 5 (the author) announces the regression submission with
   archive-4's sha256. **[commit 6]**: amendment 5, `results/stress_archive4.json`,
   the build changes and model.py's text. Steps 1 to 3 by 2026-10-16 23:59 UTC
   (hard stop: archive-3 stays, nothing is submitted).
4. Upload (the author), once, by 2026-10-17 23:59 UTC (hard stop).
5. When run 4's table arrives, `experiments/formative_run4.py` records it in
   `results/formative_run4.json` and checks every pair for finite outputs.
6. Amendment 6 (the author) records the final by 2026-10-20 23:59 UTC.
   **[commit 7]**.

On a fail, or no candidate carried: steps 11's use and 12 are skipped, and
archive-3 stays. The negative result goes to the findings.

## 13 Findings and report (3 h, after the gate's commit on 2026-10-15; due by 2026-10-20, within P2 §7's "13.10 to 20.10: findings; report R1 to R6")

* `docs/findings.md`: "The expanded pool (amendment 4)": the data facts, the
  families, the candidates and the B0-centre table, the S-only identity slopes
  and C3's removal, the confirmation table and the decomposition, Mode A, the
  disclosures (PE §15), every number with its key in `results/pool_eval.json`
  or `results/pool_fits.json`.
* The report follows PE §14, with the App G.2, G.6 and `draft.md:305` text of
  PE §11, confirmed by the author.
* `docs/protocol.md` gains the pool revision, the item schema and the
  interactors unknown; README and CLAUDE.md drop "fitted on the five public
  benchmarks" if archive-4 ships.

## Totals

| | session time | wall time of runs |
|---|---|---|
| before the lock (steps 0 to 8) | about 20.5 h (2026-10-12 to 2026-10-13; 18 h before the re-check's fixes, 19.5 h before the verification's) | about 3 h |
| scoring and gate (steps 9 and 10; 2026-10-14 to 2026-10-15) | about 1 h attended | about 8.6 h at PE §12's 1.5x figure (26 h at three times it), plus 1.5 h extra |
| build path and archive-4 (steps 11 and 12; 2026-10-14 to 2026-10-17) | about 3.5 h | about 45 min, plus the platform's scoring |
| findings and report (step 13) | about 3 h | |

## Risks, and what happens then

* **The item fetch.** Done: every item file of L (2.28 GB) is present and
  verified (step 0). What remains on disk is the fits cache, the rows and the
  bundles (about 0.2 GB, PE §12), beside 11.24 GB free; a file that later fails
  the manifest check is a check failure (next item).
* **The calendar leaves no slack day.** The work before the lock (about 20.5 h
  of session time, steps 0 to 8, of which about 1.5 h are the re-check's
  fixes and about 1 h the verification's) has 2026-10-12 and 2026-10-13; the
  two scoring
  stages have one day each before 2026-10-15 23:59 UTC. A late lock means
  "not locked in time", and a re-score forced by a check failure after the
  lock may not fit, which means "not read in time"; either way archive-3
  stays.
* **A manifest mismatch.** A check failure: re-fetch the file, re-run `fits`.
* **RP1 fails at 1e-9.** Find the cause before scoring; the rows are full
  floats, so a failure means a real change.
* **The balance check fails.** The lock stops; a dated amendment adopts the
  critic's benchmark-unit split or records the failure, before any scoring
  (PE §5.1).
* **A dense task is slower than planned.** The timing-stage adjustment covers
  it; nothing is cut after the lock; if time runs out, "not read in time".
* **A fit fails to converge.** Kept as returned. A crash is a check failure:
  fix, re-run `fits`, re-lock before scoring.
* **A bundle fails at run time.** Its construction is in `paiec/pool.py`'s
  bundle arithmetic, outside a post-lock fix's scope, so the outcome is
  "archive-3 stays" for the candidates it affects (PE §9 item 11); RP5's ten
  random exclusion sets test this before the lock. A fallback inside hier is
  not a failure: it is scored and counted.
* **A large item file exhausts a worker.** `items_of` reads projected columns
  in bounded batches; `fits` and `timing` measure mind2web and swe_smith
  before the lock, and a timing worker above 1.45 GB stops the lock until the
  reader is fixed. After the lock `items_of` is out of a fix's scope, so a
  failure there means "archive-3 stays".
* **Any other failure after the lock.** Fixed only within PE §9 item 11's
  scope, through `record-fix` and a commit with the diff before any re-score;
  the locked tests must pass unchanged, and the re-scored set is fixed by
  rule (the failed tasks and the verification set, or the whole stage); the
  stops of 2026-10-15 and later still bind.
* **An invalid output.** Recorded, never raised; "archive-3 stays" for its
  arm's candidate and no other, or for every candidate if A3's; never fixed (PE §9 item
  11).
* **The machine is loaded.** 3 shards; CPU time, not wall time, in the latency
  conditions.
