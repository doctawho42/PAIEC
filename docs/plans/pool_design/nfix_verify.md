# Verification of nfix_log.md (N1 to N9), 2026-10-11

Written 2026-10-11 against repository HEAD 2bfcd17, read only, for
`amendment4_short.md` ("A4"), `p2_pool_eval.md` ("PE"), `impl_plan.md` and
`nfix_log.md`, with `recheck.md` and `critic.md` as the reference.

What I ran, all descriptive:

* sha256 of the design files, the pre-fix copies in `nfix_orig/`, every file
  PE §2 and §9 item 10 pins (on disk and at HEAD), the two planning CSVs and
  the official streaming client;
* a static AST scan of the in-repository import closure of the harness
  modules and of the draft `paiec/pool.py`;
* `a4final/dense_cap.py` re-run on copies in the session scratchpad. Its JSON
  output is byte-identical to `a4final/dense_cap.json` (a75298c6…);
* the CSV arithmetic: `logit_cc` against log((k + 0.5) / (n - k + 0.5)) (largest
  difference 0, 13,499 pairs), and `level_cc` against the mean pair logit
  (largest difference 5.8e-15, 166 benchmarks);
* reads of `b0_dates.json`, `split_v2.json`, `dense_cap.json` and
  `results/regime_sensitivity.json`, and the parquet footers (metadata only)
  of mind2web's and swe_smith's item files;
* reads of the code the rule leans on: `official.py`, `level_calibration.py`'s
  harness, `hier.py`'s `predict`, `identity_key` and `item_groups`, and
  `prior.benchmark_fit`. I also read the in-progress `paiec/pool.py`, only to
  compare it with the design.

Not done: no predictor run, no ALC or Brier, no item text read, no formative
table opened, no git write. I wrote only this file (and scratch copies outside
the repository).

## Verdict

**Not ready to commit as is, but close.** These are resolved as the re-check
asked: N1 (option (a)), N2, N4, N7, N8 and N9. N3, N5 and N6 are resolved in
substance, but each leaves a gap. A4 and PE disagree in two places, both in
Reading 4. Three lock-completeness gaps and two post-lock choices remain that
the re-check did not list. Every item below is text, a lock entry or a
harness rule. None needs a predictor score, and all can be done before
commit 1.

After the edits, PE's sha256 changes again. The new value goes in full into
A4's preamble, and its first 16 hex into impl_plan step 0.1. A change to
`RULE_TEXT` touches A4 and the constant in `experiments/pool_eval.py` only.

## Digests

| file | computed sha256 | quoted in | verdict |
|---|---|---|---|
| p2_pool_eval.md | `3341c4742b3a243409e26e24e3de77aba2dde049361635644e9913b5bf6e4a39` | A4 preamble, in full; impl_plan 0.1, `3341c4742b3a2434`; nfix_log | correct |
| amendment4_short.md | `a45df8833b3e2bbd3c30951a69bf9cc1cf75fd701d1a2afdfd4f521785da874f` | nfix_log | correct |
| impl_plan.md | `05ecff7df35975961c4a88823ca8dfafa19728e3e1c3029ccdf4f5ec5dbc30a9` | nfix_log | correct |
| a4final/dense_cap.py / .json | `bc692edb…` / `a75298c6…` | nfix_log | correct; the JSON reproduces byte for byte |
| nfix_orig/ (the three pre-fix files) | `c655e482…`, `1a95760d…`, `3e080097…` | nfix_log "before" | correct |
| benchmarks.csv, pairs.csv | `d88247c984edc5df…a405`, `2e913b8b24c29fc8…fac6a` | PE 9.10; nfix_log item 5, in full | correct |

* PE quotes no digest of A4, impl_plan or nfix_log, so the chain has no cycle.
* Every module PE pins equals HEAD on disk: the eight shipped modules,
  `submission/model.py`, `official.py`, `testlike.py`, `level_calibration.py`,
  `regime_sensitivity.py`, `data.py`, `evaluator.py`, and the three new pins
  (`baselines.py`, `ship_confirm.py`, `subject_side.py`).
* These data files equal their quoted digests on disk:
  `_compositions.json`, `submission/prior.json`, `data/pool/manifest.json` and
  the pre-items manifest.
* model.py's cited lines are right: the docstring spans 1-40, lines 13-14 are
  on baking LEVEL in, the comment above `LEVEL` is 62-67, and `LEVEL` is on
  line 68.
* `RULE_TEXT` is ASCII, has no tabs and is at most 79 columns wide.
* Two small inaccuracies in nfix_log itself, not in the design:
  * the CSVs are untracked, so they cannot "equal `git show HEAD:<path>`";
  * "contains nothing else outside the shipped eight" is wrong (V1).

## N1 to N9

| | status | residual |
|---|---|---|
| N1 | **resolved**. Option (a) is in RULE_TEXT (E2) and (c), A4 decision 7, A4 §4.1 and §4.3, and PE §1, 4.2, 4.3, 5.3, 7 and 14. The claims now say what holds: C1m's gains carry its +0.045 shift; only its no-loss half is read at archive-3's centre. The figures match `b0_dates.json` (C1 +0.0086 / -0.0365; C1m +0.0447 / 0.000, dated +0.0632; C2 +0.2332 / +0.1898). PE §7's probabilities are right (0.5 each; 0.25 to 0.5 together; at least 0.84 at one SE). | Recommended: PE §7 gives the twins' cost per stage. Over selection and gate together, a candidate with no twin effect passes the twins in only about 0.06 to 0.25 of draws. |
| N2 | **resolved**. A4's preamble now hashes "everything p2_pool_eval §9 item 10 lists". | none |
| N3 | **partly resolved**. `baselines.py`, `ship_confirm.py`, `subject_side.py` and both CSVs are pinned. The levels are derived, checked and written to `levels.json`, and the closure check exists. | **V1**: `experiments/__init__.py` is in the closure and is not pinned. **V2, V3** below are lock gaps of the same kind. |
| N4 | **resolved**. PE §11 item 3 and the `draft.md:305` clause and sentence (checked against line 305 as it stands), Disclosure 3, A4 §4.6 and Reading 5 all agree. | none |
| N5 | **resolved in PE, not fully in RULE_TEXT**. PE §9 item 11 covers fallbacks, non-finite outputs, scope, out of scope, record and time. | **V4, V5**: Reading 4 differs from PE. **V6, V8**: two choices are left after the lock. |
| N6 | **resolved in substance**: label-randomised copies, scores discarded, an assert and a test. | **V11**: the stream index `i` and the draw are not defined, so the assert cannot yet be written. `run_official` returns no predictions, so RP3 must log them through its factory. |
| N7 | **resolved**. (E1) names the five RS regimes, the tie-break is unambiguous and inclusive, and Reading 1 reads S3 pooled. PE §7 and the impl_plan tests agree. | none |
| N8 | **resolved**. The `items_of` contract, the separate scale read, fresh-worker fits with `ru_maxrss`, the two C runs at `timing`, and the exit-75 check after the rows are written. The footers confirm the facts: mind2web `content` is 1,839.0 + 1,294.2 MB uncompressed in 2 row groups; swe_smith `verifier` is 1,169.2 + 366.4 MB. The draft `items_of` already follows the contract (`use_threads=False`). | Recommended (V12): define "labels stripped" at `timing`. |
| N9 | **resolved**. The cap comes after the dense restriction, and every figure reproduces: S D-READING N(-0.75, 1.6), -0.684 / 1.755, Kish 25.6 / 24.0, crow 0.098; balance 0.052 by family and 0.030 by benchmark; cap-first 23.7 / 22.4, crow 0.111. The realisation now uses evaluated responses and the knob search the proxy. The twins' evaluated sets equal their parents', because `official.split` ignores dates, so the twins need no realisation of their own. The docstring range, impl_plan §13 heading, A4 date rule and commit-1 filter are fixed. | Commit filter: see the author list. |

## Are A4 and PE consistent?

Checked and consistent:

* the candidates and their planning centres;
* the twins' role (D <= 0, in (E2) and (c)) and the loss bounds;
* the counts (80/20 in S, 120/40 in C, 520 RS), the split (44/88), the seeds
  and the calendar;
* the fix's time rule, the model.py rule and the feedback statements;
* the decision list against impl_plan 0.2.

Two differences remain, both in `RULE_TEXT` Reading 4. A4 governs where the
two differ, so each is a real conflict.

* **V4.** Reading 4 gives the fix scope as `paiec.pool.load_eval_pairs`. PE §9
  item 11 adds "and the index cache it alone uses". Read by A4, a fix to the
  index cache is out of scope and archive-3 stays. Read by PE, it is in scope.
* **V5.** Reading 4 sends a non-finite output to "archive-3 stays" only "for
  the candidate whose arm produced it". PE adds "for every candidate if A3's
  did". A3's case is therefore undefined in the governing text.

## RULE_TEXT: terms and checkability

Every term in the rule is defined, either in its Terms or by a pointer to PE
§3 to §6 (B0 centre, families, formative-size and dense, planned runs, pair
ALC, identity key). Each subject has an identity key: `hier.py:1918-1927`
gives ("n", canon name) or ("k", the subject's digest), so SEs and SEfs have
no undefined cluster. The open points:

* **V9a, PB.** "the family's appearance-weighted mean difference" does not say
  what the weight is. It should be 1/(pairs in the appearance's run), as for D
  (PE §6) and as P1a's Pq stated. An unweighted mean gives another number, and
  PB <= 0 can turn on it.
* **V9b, gate (f).** It does not say whose calls are meant ("no formative-size
  call", "dense calls"). It should say X*'s, as (E3) does. The p99 is not
  defined; name one estimator, for example `numpy.percentile(x, 99)` with
  linear interpolation over all of X*'s dense calls in the C core.
* **V9c, (E3).** "formative-size call" is defined for pool runs only. RS tasks
  are test-like or R1 `sample_run` draws, so write "every call of X on the S
  formative-size runs and on the RS tasks".
* **V9d, (E4).** "every listed arm" is listed nowhere in the rule. Write "A3,
  C1, C1m and C2 on pool tasks; C1, C1m and C2 on RS tasks".
* **V9e.** `D_S` in M(X) is not in Terms. Write "D on S".
* **V10, checkability.** Two conditions cannot be read from the outputs as
  impl_plan lists them:
  * (f)'s p99 needs every dense call's CPU time. "CPU and wall per call" in
    the rows is ambiguous; store the per-call list, or at least each dense
    run's full list.
  * (E4) and (g)'s "every check passed" needs a check log. Each stage should
    write each check's name, result and time to `results/pool_eval.json`.

## Is any choice left after the lock?

The design closes most choices: the knobs, compositions, seeds, counts,
mu0_m, the C2 literal and the bundles are locked; the stages run in
run-index-major order with no early stop; CPU time, not wall time, is used;
S3 has no fix; the upload happens once. Five gaps remain.

* **V6, raise or record.** `level_calibration.evaluate`, which
  `evaluate_timed` copies, raises `ValueError` on any prediction that is not
  finite, lies outside [0, 1] or has the wrong shape (lines 546-547). Under
  PE §9 item 11 such an output is two things at once:
  * a "non-finite prediction", which gives archive-3 stays for that candidate;
  * "an exception raised in the harness", a check failure that may be fixed in
    `pool_eval.py` and then re-scored.

  After rows exist, the author could choose either. `hier.predict` never
  raises and returns a value in [1e-4, 1 - 1e-4] or 0.5 (`hier.py:2039-2071`),
  so the case is unlikely, but the choice must be closed:
  * define an *invalid output* as anything that is not a finite float in
    [0, 1] of the right shape, with the non-finite consequence;
  * `evaluate_timed` records it (arm, run, budget), keeps scoring the other
    arms, and never raises on it;
  * the validation and classification code is outside the fix scope.
* **V7, precedence.** A4's preamble and PE §0 say "Only a later dated
  amendment changes anything here". That allows an amendment written after
  rows exist to change the rule, Reading 4 included. Reading 4 itself says
  nothing changes once a row exists. Adopt the re-check's recommended clause
  in both files: no amendment changes the rule, CANDIDATES, SPLIT, the
  regimes, the seeds, the counts, the statistics or the fix scope once a row
  of SEL_SEED, CONF_SEED or the RS guard exists. Amendment 5 only announces
  the upload.
* **V8, "every affected task".** The rule does not define which tasks a fix
  affects, so the deviation record names them after rows are seen. Make it
  mechanical:
  * the fix commit runs the locked `tests/test_pool.py` and
    `tests/test_pool_eval.py` unchanged and green;
  * the failed tasks are re-scored;
  * so is a fixed verification set, run 0 and the last scored run of every
    regime of each stage already scored, which must reproduce its stored rows
    bit for bit. If any row differs, every task of that stage is re-scored;
  * `select`, `summarise` and `gate` are re-run from all rows after any fix.
* **V4 again.** The design never names the index-cache helpers, so after the
  lock the author could call any definition one. The draft `paiec/pool.py`
  already freezes a list, `EVAL_LOADER = ("INDEX_VERSION", "_eval_index_path",
  "eval_index", "_eval_rows", "load_eval_pairs", "build_eval_indexes",
  "_eval_index_task")`. RULE_TEXT and PE should name "the definitions
  `paiec.pool.EVAL_LOADER` lists, frozen at the lock", and the lock should
  record that tuple.
* **V2, the RS comparator.** A3 on RS is read from
  `data/regime_sensitivity_rows/<R>/<i>.json`, 520 files. They are under the
  gitignored `data/`, and nothing pins them: PE §9 item 10 pins only
  `_compositions.json`, and `results/regime_sensitivity.json` records only
  that file's digest. (E1), and with it the selection, reads them, and RP1
  re-checks only 14 of the 520. They could change after the lock without any
  check noticing.

## Required before commit 1

| | change | where |
|---|---|---|
| V1 | Pin `experiments/__init__.py`. It is empty, `e3b0c44298fc1c14`, tracked, and executed by every `import experiments.*` in `regime_sensitivity.py`, `level_calibration.py`, `ship_confirm.py` and `subject_side.py`. Without it the lock refuses by its own closure rule, or the code needs an exemption that is written nowhere. | PE §9 item 10; `PINNED` |
| V2 | Pin the 520 RS row files: one digest over the sorted (relative path, sha256) list, recorded at `lock` and re-verified by `select` before it reads them. | PE §9 item 10; §4.6 |
| V3 | Record at the lock the sha256 of `third_party/paiec_baseline/tools/streaming_ingestion.py` (`c6f2610f9c8b6cad…`, clone at 82d330d). `official.official_client()` executes it on the first `make_hook` call in every process that builds checkpoints, and `run_official`'s default `deepcopy=True` uses its `AcquisitionHook` in RP3. A static AST closure cannot see it. | PE §9 item 10 |
| V4 | Fix scope = the definitions in `paiec.pool.EVAL_LOADER`, frozen at the lock, in RULE_TEXT Reading 4 and PE §9 item 11 alike. | A4 §4.4; PE §9.11 |
| V5 | Reading 4: a non-finite output of A3's arm gives "archive-3 stays" for every candidate. | A4 §4.4 |
| V6 | Define invalid outputs (not a finite float in [0, 1], or the wrong shape) with the non-finite consequence. The harness records them and never raises on them, and that classification is outside the fix scope. | A4 §4.4 Reading 4; PE §9.11; §5.2 |
| V7 | The precedence clause. | A4 preamble; PE §0 |
| V8 | A mechanical "affected tasks" rule, as above. | PE §9.11; A4 Reading 4 (one clause) |
| V9 | The precision fixes a to e. | A4 §4.4; PE §6 (PB weight) |
| V10 | Rows hold per-call CPU times for dense runs, and each stage writes a check log. | PE §6 and §9; impl_plan §5 |
| V11 | RP3's stream: `i` is the RP3 run's index; the twin uses its parent's randomised copy with shifted dates; the draw is `integers(0, 2)` per 0/1 response, in pair order and then file order (or another rule, as long as one is fixed). | PE §9 item 6 |
| V13 | PE §9 item 10 calls the lock commit "the outside time stamp". That holds only if commits 1 to 4 are pushed when made. Either push them (the author's decision) or write "the local commit time stamp". | PE §9.10 |

## Recommended

* **V12, `timing`'s "labels stripped".** It is not defined. If `labeled` holds
  no labels, hier may take another path or fall back, and the CPU projection,
  hence the 30 process-hour adjustment, would not measure the scoring path.
  The N8 item-read check is unaffected. Use label-randomised copies from a
  separate stream, for example `[TIMING_SEED, 9, i]`, and keep the no-Brier
  assert.
* **The `--pool` build environment.** Pin `PYTHONHASHSEED=0` and one BLAS
  thread, as the lock does. The re-check recommended this; it is still open.
* **`dist/archive-3.zip`.** Add its full digest to the lock list beside
  `a3_prior.json`.
* **Fix-commit tests.** A fix commit runs the locked tests green. This is
  already part of V8, but should also be said in PE §9.11.
* **HELM-suite cut.** The descriptive cut is still open from the re-check.
* **Power over both stages.** State the twins' cost over selection and gate
  together in PE §7 (see N1).

## Changes that affect code (for the orchestrator)

These add to nfix_log's items 1 to 28.

29. `PINNED` += `experiments/__init__.py` (e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855).
30. `lock` writes one digest over the sorted (path, sha256) list of
    `data/regime_sensitivity_rows/{TUNED,READING,AUDIT,MIXTURE,FLAT,R1B,R1P}/<i>.json`
    (520 files). `select` and RP1 re-verify it.
31. `lock` records the sha256 of
    `third_party/paiec_baseline/tools/streaming_ingestion.py`
    (c6f2610f9c8b6cad8a2338ddc07a0cb25d40bca761a6628937a2cee2705a6267) and the
    clone's commit. Every stage's lock check compares it.
32. The fix scope is exactly `paiec.pool.EVAL_LOADER`, as the draft has it. The
    lock records the tuple, `frozen_ast` removes exactly it, and a changed
    tuple is out of scope.
33. `evaluate_timed` does not copy `evaluate`'s `raise ValueError("invalid
    prediction")`. It records an invalid output (not a finite float in [0, 1],
    or the wrong shape) per arm, run and budget, scores the other arms on, and
    the result gives "archive-3 stays" for that arm's candidate, or for all of
    them if A3's arm produced it. The classification code sits in the
    constants block.
34. `record-fix` and `score`:
    * the locked tests must pass unchanged after a fix;
    * re-score the failed tasks plus a fixed verification set (run 0 and the
      last scored run of every regime of each scored stage), which must
      reproduce its stored rows bit for bit; otherwise re-score the whole
      stage;
    * re-run `select`, `summarise` and `gate` from all rows.
35. The rows store every call's CPU time for dense runs. Gate (f)'s p99 is one
    fixed estimator over X*'s dense calls in the C core.
36. Each stage appends a check log (name, result, UTC) to
    `results/pool_eval.json`. `select` and `gate` read (E4) and (g) from it.
37. PB uses the appearance weight 1/(pairs in the run), the same weight as D.
38. RP3:
    * the randomising stream is `default_rng([TIMING_SEED, 3, i])`, with i the
      RP3 run index;
    * draws are `integers(0, 2)` per 0/1 response in pair order and then file
      order;
    * the twin reuses its parent's copy;
    * predictions are captured through a recording factory, since
      `run_official` returns scores, not predictions;
    * the returned dict is discarded.
39. `timing` (recommended): label-randomised copies on stream
    `[TIMING_SEED, 9, i]` rather than labels removed.
40. `--pool` build (recommended): `PYTHONHASHSEED=0` and one BLAS thread,
    checked as at the lock.
41. The draft `paiec/pool.py:532` has `check_csvs(..., tol=1e-9)`. The design
    says 1e-12, both for `logit_cc` and for `level_cc`; the CSVs agree with
    the formula to 0 and 5.8e-15.

## Decisions the author must confirm in commit 1

This is the re-check's list merged with nfix_log's, plus the decisions this
verification adds:

1. **Decision 6, the calendar.** Lock 10-13, gate 10-15, archive-4 10-16,
   upload 10-17, final 10-20. With it:
   * the dense counts of 20 and 40;
   * about 19.5 h of code on 10-12 and 10-13;
   * the slip of the report schedule.
2. **Decision 7 (N1, option (a)).** The twins are no-loss conditions, D <= 0,
   for every candidate. This choice came from the task text.
3. **C1m's matching set.** C1m is matched on all of E (-2.035 planned), not on
   dated subjects only (-1.891).
4. **impl_plan 0.2's open choices:**
   * Mode A;
   * (E1) as a mean plus P1a's bounds;
   * C2's choices;
   * the family-whole split, exclusion and clusters;
   * C3 dropped;
   * the 0.10 cap, applied after the dense restriction.
5. **The narrow post-lock fix scope.** `items_of` is outside it, and the scope
   is `EVAL_LOADER` (V4). Non-finite and invalid outputs give "archive-3
   stays", including A3's case (V5, V6).
6. **The levels.** The CAL_SEED value is read on evaluated responses and the
   knob search runs on the proxy. The new S D-READING knob is N(-0.75, 1.6).
7. **The precedence clause** (V7) and the mechanical re-score rule (V8).
8. **The time stamp.** Either push commits 1 to 4 when they are made, or drop
   "outside" from "outside time stamp" (V13).
9. **The commit-1 filter.** Leave out:
   * `eval_describe/pool_describe.json`;
   * `nfix_orig/`;
   * `survey.json`, whose canon_names come one entry per pair.

   Also decide on `code/c1_proto_prior.json`. It is a full prior bundle with
   494 name rows, fitted on the whole pool, so it is the same kind of file as
   `submission/prior.json`, which the repository never commits.
10. **The report and the feedback.** The wording of App G.2, App G.6 and
    `draft.md:305`, with the C1m clause, and the statement that nothing else
    from the feedback was read.

### The same list in Russian

1. Решение 6: календарь (лок 13.10, гейт 15.10, archive-4 16.10, загрузка 17.10, финал 20.10), dense-прогонов 20 и 40, около 19,5 ч кода 12–13.10, отчёт сдвигается.
2. Решение 7 (N1, вариант (a)): близнецы со сдвигом дат — условие «без потерь», D ≤ 0, для каждого кандидата (выбор пришёл из текста задачи).
3. C1m: mu0 подбирается по всем парам E (план −2,035), а не только по датированным (−1,891).
4. Открытые выборы impl_plan 0.2: режим A; (E1) как среднее плюс границы P1a; выборы C2; разбиение, исключение и кластеры по семействам; C3 исключён; кэп 0,10 после dense-ограничения.
5. Узкая область исправлений после лока: только pool_eval.py вне блока констант и EVAL_LOADER, items_of вне её; нефинитный или недопустимый выход даёт «archive-3 остаётся», в том числе если он у A3.
6. Значение CAL_SEED по оценённым ответам, поиск ручек по прокси; новая ручка D-READING для S: N(−0,75; 1,6).
7. Пункт о старшинстве (после первой оценённой строки поправки не меняют правило) и механическое правило перепрогона.
8. Пушить коммиты 1–4 сразу, ради внешней метки времени, или убрать слово «внешняя».
9. Фильтр коммита 1: без pool_describe.json, nfix_orig/ и survey.json; решить судьбу code/c1_proto_prior.json.
10. Тексты App G.2, G.6 и draft.md:305 с оговоркой о C1m; подтверждение, что больше ничего из обратной связи не читалось.
