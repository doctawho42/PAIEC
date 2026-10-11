# Amendment 4: the verification's V1 to V13, fixed (2026-10-11)

Written 2026-10-11, about 20:00 to 21:00 UTC, against repository HEAD
2bfcd17. It covers `amendment4_short.md` ("A4"), `p2_pool_eval.md` ("PE") and
`impl_plan.md`, revised for the required changes V1 to V13 of
`nfix_verify.md` and for its recommended items.

Rules kept:

* no git write;
* no ALC or Brier of any predictor;
* no item text;
* no formative feedback beyond what App G.6 and PE §11 list;
* writes stay inside this folder.

What I ran:

* sha256 of files;
* two new planning scripts in `a4final/`, both descriptive:
  * `rs_rows_digest.py` hashes the bytes of the 520 RS row files and of three
    other lock entries. It opens no row as JSON and reads no score.
  * `helm_cut.py` reads only the catalogue `source_url` of E's 132
    benchmarks, plus `split_v2.json` and `families.json`.

No predictor was run, and no item table or formative table was read.
`dist/` could not be hashed, because a plugin hook blocks it, as it blocked
the re-check. So archive-3's full sha256 is the one that
`results/stress_archive.json`, `results/formative_run3.json`, README and
`docs/release.md` record. impl_plan step 1 checks it on the file itself.

**Where two inputs came from.** Both came from the workflow's task text, not
from the author in this session. The author confirms both in commit 1, as for
decisions 6 and 7.

* **The precedence clause.** Its wording is the task text's: "nothing in the
  rule, candidates, split, regimes, seeds, counts, statistics or fix scope
  changes once a scored row exists, not even by a later amendment". It is
  adopted verbatim in A4's preamble and PE's.
* **V13's approval.** The task text gives the approval of pushes of
  pre-registration commits as dated 2026-10-06. git shows that such a commit
  was pushed: `origin/main` stands at 6915f98, the commit of 2026-10-06 that
  added amendment 1 to `p2_final_plan.md`. git cannot show the approval
  itself. A4 records it as decision 8.

The pre-fix copies of the three files are in `vfix_orig/`, for the diff. They
are not for commit 1 (PE §13 now says so).

## Digests

| file | before | after |
|---|---|---|
| p2_pool_eval.md | `3341c4742b3a243409e26e24e3de77aba2dde049361635644e9913b5bf6e4a39` | `7806244ca4f5d273b788278e3a05086433403368b6011e8c38503f69fef19302` |
| amendment4_short.md | `a45df8833b3e2bbd3c30951a69bf9cc1cf75fd701d1a2afdfd4f521785da874f` | `246a7b612ca6cbc8c6feac697a99cb88b9b31b51af9f9ff78228f0ea34a1b968` |
| impl_plan.md | `05ecff7df35975961c4a88823ca8dfafa19728e3e1c3029ccdf4f5ec5dbc30a9` | `7929c187de475e6e30df5dffc9112c0308f781b5ed970f2f809326c4ffcc1765` |
| a4final/rs_rows_digest.py (new) | | `f92de139c4fd980d9280b1e12214783391bfbe260147a37af1f5b04fc4318036` |
| a4final/rs_rows_digest.json (new) | | `479444fa6ed597e9e117467e1c41eb2c0a96581e0d3a8e1360c3979f41e2fdf2` |
| a4final/helm_cut.py (new) | | `f9230d981d6d22eb65fdac6703d1480f18ca2cb86095bb6584708afe7f1b9ed0` |
| a4final/helm_cut.json (new) | | `f48e62035349441ea1f91c2e85cae0cafaea0da9ce9ba4dc05682ff873532dd4` |

* The "before" digests are nfix_log's "after" digests, which nfix_verify
  confirmed.
* A4's preamble quotes PE's new sha256 in full. impl_plan step 0.1 quotes its
  first 16 hex, `7806244ca4f5d273`.
* PE quotes no digest of A4, impl_plan, nfix_verify or this log, so the chain
  still has no cycle. PE was not touched after its digest was taken.
* A change to `RULE_TEXT` touches A4 and the constant in
  `experiments/pool_eval.py` only. `RULE_TEXT` changed (item 42 below). It is
  still ASCII, has no tabs and is at most 79 columns wide; the header line
  is the widest.

## What changed, finding by finding

### V1: `experiments/__init__.py`

* PE §9 item 10 pins it: e3b0c44298fc1c14, the empty file's digest. It is
  the same as `paiec/__init__.py`'s.
* PE §9 item 10 says why: it is tracked, and every `import experiments.*` in
  `regime_sensitivity.py`, `level_calibration.py`, `ship_confirm.py` and
  `subject_side.py` executes it.
* impl_plan §5's `PINNED` and §6's closure test follow.

### V2: the RS comparator's 520 row files

PE §9 item 10 pins them as one digest.

* **Rule.** The digest is the sha256 of the UTF-8 bytes of the lines
  `"<path>\t<sha256>\n"`, sorted by path. Each path is relative to
  `data/regime_sensitivity_rows/` in POSIX form.
* **Files.** Exactly `<R>/<i>.json`:
  * i 0 to 79 for TUNED, READING, AUDIT, MIXTURE and FLAT;
  * i 0 to 59 for R1B and R1P.
* **Who checks it.** RP1 computes it, the lock records it, and `select`
  re-verifies it immediately before it reads a row.
* **Planning value** (2026-10-11): 520 files,
  `9a601549b2d99739487e7c3808da616f24f9072506671043de878020f59a9b5c`. PE
  quotes its first 16 hex. Commit 1 is pushed when made, so the value is an
  outside reference for these gitignored files from that day on.
* PE §4.6 and §9 item 4 (RP1) state the same. impl_plan §5 and §6 follow.

### V3: the official streaming client

PE §9 item 10 pins
`third_party/paiec_baseline/tools/streaming_ingestion.py` by name.

* Its sha256 is c6f2610f9c8b6cad (in full in `rs_rows_digest.json`).
* The clone's commit, 82d330ddcdb16016a3ae9e048db7e588ba2c6a39, is recorded
  beside it.
* The file is gitignored and has no license, so only its digest is recorded.
* PE states the two reasons. `official.official_client()` executes the file
  in every process that builds checkpoints, and RP3's `run_official` uses its
  `AcquisitionHook`. No static import closure sees it.

### V4: the fix scope is `paiec.pool.EVAL_LOADER`

* `RULE_TEXT` Reading 4 now names "the definitions paiec.pool.EVAL_LOADER
  lists, frozen at the lock". It used to name `paiec.pool.load_eval_pairs`.
* PE §9 item 11 says the same:
  * it names the draft's seven definitions;
  * the lock records the tuple;
  * a fix that changes the tuple, or any definition the tuple does not name,
    is out of scope;
  * the AST rule is `paiec.pool.frozen_ast` with exactly those definitions
    removed.
* The draft's tuple does not contain its own name, so `frozen_ast` keeps the
  tuple and a changed tuple fails the AST rule.
* PE §9 item 10 records the tuple at the lock.

### V5 and V6: invalid outputs, A3's case included

* **`RULE_TEXT` Reading 4.** "Fallbacks and other valid outputs ... are never
  check failures." An *invalid output* is "anything but a finite float in
  [0, 1], or of the wrong shape". It is recorded and never raised. It gives
  "archive-3 stays" for the candidate whose arm produced it, and for every
  candidate if A3's arm produced it. "The constants block, the classification
  of outputs included" stays unchanged.
* **PE §5.2**, a new bullet:
  * `evaluate_timed` does not copy `evaluate`'s raise (:546-547);
  * the definition, after `numpy.atleast_1d`, is shape (1,);
  * the record is (arm, run, budget, call index);
  * the other arms of the task are scored on, and the invalid arm is not
    scored on that run;
  * `classify_output` sits in the constants block.
* **PE §9 item 11** has two bullets now:
  * "Valid outputs are never failures";
  * "Invalid outputs": never a check failure, never re-scored, never fixed.

  In "Check failures", "a non-finite value raised in the harness" becomes "a
  non-finite value in the harness's own arithmetic", and an arm's invalid
  output is excluded.
* **PE §6** counts invalid outputs beside fallbacks.

### V7: precedence

* **A4's preamble and PE's preamble.** Each now says: "Only a later dated
  amendment changes anything here, with one limit: nothing in the rule,
  candidates, split, regimes, seeds, counts, statistics or fix scope changes
  once a scored row exists, not even by a later amendment." The clause is
  word for word the task text's; a check with whitespace normalised finds it
  once in each file.
* Both files then define a scored row as a row of SEL_SEED, CONF_SEED or the
  RS guard. Both say that amendment 5 only announces the upload, and that the
  amendment recording the final only records it.
* **`RULE_TEXT` Reading 4** adds "or the fix scope" and "not even by a later
  amendment" to its own list, so the rule and both preambles agree.

### V8: what is re-scored after a fix

* **PE §9 item 11**, a new bullet with four steps:
  1. the fix commit runs the locked `tests/test_pool.py` and
     `tests/test_pool_eval.py`, unchanged, and both pass;
  2. the failed tasks are re-scored in full;
  3. so is a fixed verification set:
     * run 0, and the last run scored before the fix, of every regime of
       every stage with rows (S, RS, C);
     * each must reproduce its stored rows bit for bit, in every field but
       the timing and resource fields;
     * if any verification row differs, every task of that stage is
       re-scored;
  4. `select`, `summarise` and `gate` are re-run from all rows.
* The deviation record now holds the tasks of steps 2 and 3 and the test
  result of step 1.
* **`RULE_TEXT` Reading 4** gets the one clause nfix_verify asked for ("by
  that item's rule: ..."), plus the re-run and its outcome (next list).

**Two precisions beyond nfix_verify**, for the author to confirm:

* **The timing and resource fields are not compared.** These are the CPU
  and wall times, the resident set and the load average. They never
  reproduce, so a literal "bit for bit" would always re-score the stage.
* **A re-run `select` that names another X*.** If its X* differs from the
  committed selection record, the outcome is "archive-3 stays". This can only
  happen when a verification row of S or RS differs, so that S or RS is
  re-scored in full after C rows exist. C's rows were then scored for
  another candidate. The case is in `RULE_TEXT` Reading 4 and in PE §9
  item 11, so the two agree.

### V9: precision of the rule

* **(a) PB** (Terms, PE §6). It is the family's mean pair-ALC difference, X
  minus A3, with each appearance weighted 1/(pairs in the appearance's run),
  as D weights them. PE writes it as v_f / n_f and says an unweighted mean is
  not PB.
* **(b) Gate (f).** It now reads:
  * "no call of X* on the C formative-size runs takes more than 2.0 s";
  * "X*'s dense calls in the C core have a p99 (numpy.percentile(x, 99),
    linear interpolation, over all of them) of at most 1.0 s and a maximum of
    at most 30 s".
* **(c) (E3).** "no call of X on the S formative-size runs or on the RS tasks
  takes more than 2.0 s of CPU".
* **(d) (E4).** "every arm (A3, C1, C1m and C2 on pool tasks; C1, C1m and C2
  on RS tasks)".
* **(e) M(X).** "M(X) = [D(X,P-READING) + D(X,P-AUDIT)] / 2, D on S".
* PE §6 gains a "Latency" bullet that states (b) and (c). It also states the
  literal reading of "mean CPU time per evaluation call": the CPU time summed
  over all calls, divided by their number.

### V10: checkability

* **PE §5.2, "Timing in the rows".** Every row holds, per arm, the number of
  calls and the sums and maxima of CPU and wall time. A dense run's row also
  holds every call's CPU time, in call order.
* **PE §9, a new paragraph, "The check log".**
  * Every stage, before and after the lock, appends (stage, name, result with
    the failing values, UTC) to `results/pool_eval.json: checks`.
  * A check missing from the log has not passed.
  * `select` reads (E4) from the log, and `gate` reads (g).
  * A failure that a recorded fix resolved counts as passed once its re-run
    passes and is logged. Without this clause, the fix path of Reading 4
    could never pass (E4) or (g). This is a precision beyond nfix_verify.

### V11: RP3's stream

PE §9 item 6 now fixes the stream.

* **The runs.** RP3's run 0 is formative-size and run 1 is dense. i is the
  RP3 run's index.
* **The draws.** Run i's copy replaces every 0/1 response with the successive
  draws `rng.integers(0, 2)` of
  `rng = numpy.random.default_rng([TIMING_SEED, 3, i])`. The responses are
  taken in the run's pair order, and within a pair in file order.
* **The twin** is run 0's copy with its dates shifted, so it reuses its
  parent's draws.
* **The predictions** are captured through a recording factory, because
  `run_official` returns scores, not predictions. The factory logs every
  call's prediction by checkpoint and call order. `run_official`'s dict is
  discarded.

### V12 (recommended): `timing` on label-randomised copies

PE §9 item 9 and the seed table of §5.4 now say:

* `timing` runs on label-randomised copies, not with labels removed;
* the stream is `[TIMING_SEED, 9, i]`, with the same draw order as RP3;
* i follows the stage's fixed order: S formative-size runs 0 to 5, S dense
  runs 6 and 7, mind2web 8, swe_smith 9;
* the asserts are kept: every response equals its draw, and no Brier or ALC
  is computed.

PE gives the reason: with an empty `labeled`, hier could take another path or
fall back, and the projection would not measure the scoring path. §5.4 also
notes that the three-element streams differ from every composition stream.

### V13: the time stamp

* **PE §9 item 10** now reads: "Commits 1 to 4 are pushed to the public
  repository when made (the author approved pushes of pre-registration
  commits on 2026-10-06), so the lock commit is an outside time stamp, which
  P1a lacked".
* **The same in other places:**
  * A4 decision 8;
  * a "Pushes" paragraph in PE §13;
  * "pushed when made" in PE §14;
  * impl_plan's conventions, step 0.2, step 0.3 and commits 2, 3 and 4.
* A4's decisions heading now says decisions 7 and 8 were added on 2026-10-11
  and that the commit confirms all three.

### The recommended items

* **The `--pool` build environment.**
  * A4 §4.2 says the byte-for-byte rebuild runs "under the lock's settings
    (`PYTHONHASHSEED=0`, one BLAS thread)".
  * PE §10 step 1 and §9 item 10 say the build checks them as the lock does,
    and refuses to run without them.
  * impl_plan §11 follows.
* **`dist/archive-3.zip`'s full sha256 in the lock.** PE §2 and §9 item 10
  give 4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090, with
  `a3_prior.json`'s full digest (c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770)
  beside it. impl_plan step 1 checks both in full.
* **The twins' combined power cost** (PE §7, new bullet; one clause in A4
  decision 7):
  * With no twin effect, a carried candidate passes all four twin conditions
    (selection and gate) in only about 0.06 to 0.25 of draws: 0.25² to 0.5²,
    taking S and C as independent.
  * With a twin gain of one SE or more in each, it passes all four in at least
    about 0.50 to 0.71 of draws: 0.84⁴ to 0.84².
  * PE adds that each twin replays its parent's runs. The joint chance with
    (a) is therefore not a product.
* **The descriptive HELM-hosted cut** (PE §6, a new descriptive cut). A
  benchmark is HELM-hosted when its own catalogue `source_url` is on
  crfm.stanford.edu, with a path under `helm/`.
  * *Planning* (`a4final/helm_cut.py`): 14 benchmarks of E (helm_afr,
    helm_bbq, helm_thaiexam, legalbench and the ten medhelm_*).
  * All 14 are in C. They fall in 5 families (bridging_gap, harmbench,
    helm_thaiexam, legalbench, medhelm_ehr_sql) and 5 suites (lite, medhelm,
    mmlu-winogrande-afr, safety, thaiexam).
  * S has none, so the cut is read in C's regimes only. harmbench and
    bridging_gap are not themselves HELM-hosted; their families are, through
    helm_bbq and helm_afr.
* **The fix-commit tests.** They are step 1 of V8 in PE §9 item 11, and they
  are in `record-fix` (impl_plan §5).

### Other edits

* **Pointers to this round.** PE §15 has a new table that maps V1 to V13 and
  the recommended items to sections. PE's N8 row now says "on
  label-randomised copies since V12".
* **The commit-1 lists.**
  * PE §13 adds `nfix_verify.md` and `vfix_log.md` to the folder's list and
    keeps `vfix_orig/` out.
  * impl_plan step 0.3 does the same, and names `survey.json` and
    `code/c1_proto_prior.json` as the author's decisions.
* **Estimates.** impl_plan's estimates rise to about 20.5 h before the lock
  (the step 0.2 list and the risks).

## Corrections to nfix_log

nfix_verify found two inaccuracies in nfix_log itself. They are corrected
here; nfix_log is left as written, as the record.

* **The CSVs.** `benchmarks.csv` and `pairs.csv` are untracked, so they cannot
  "equal `git show HEAD:<path>`". Their digests were computed on disk.
* **The closure.** "contains nothing else outside the shipped eight" is wrong:
  `experiments/__init__.py` is in the static closure too (V1).

## Changes that affect code

For the implementer. Items 1 to 28 are nfix_log's. They stand, except where
the next list amends them. Items 29 to 41 are nfix_verify's, adopted as
written; where a later item makes one concrete, it says so. Items 42 to 58
are new.

**nfix_log's items amended by this round**

* 1 (`RULE_TEXT`): superseded by 42.
* 5 (`PINNED`): extended by 29, 30, 31 and 46.
* 7 (the AST rule for `paiec/pool.py`): it now removes exactly the
  `EVAL_LOADER` definitions (32, 56).
* 9 and 10: "finite output" becomes "valid output", and "non-finite
  prediction" becomes "invalid output", with A3's case (33, 43, 44).
* 12 (`record-fix`): scope `EVAL_LOADER`; the locked tests; the verification
  set; the re-runs (32, 34, 54, 55).
* 18 (`timing`): "labels stripped" becomes label-randomised copies (39, 53).
* 25 (RP3): made concrete by 38 and 52.
* 27 (tests) and 28 (estimates): extended by 58.

**nfix_verify's items 29 to 41** (adopted)

29. `PINNED` += `experiments/__init__.py`
    (e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855).
30. `lock` writes one digest over the sorted (path, sha256) list of
    `data/regime_sensitivity_rows/{TUNED,READING,AUDIT,MIXTURE,FLAT,R1B,R1P}/<i>.json`
    (520 files). `select` and RP1 re-verify it. Item 45 makes it concrete.
31. `lock` records the sha256 of
    `third_party/paiec_baseline/tools/streaming_ingestion.py`
    (c6f2610f9c8b6cad8a2338ddc07a0cb25d40bca761a6628937a2cee2705a6267) and the
    clone's commit (82d330ddcdb16016a3ae9e048db7e588ba2c6a39). Every stage's
    lock check compares it.
32. The fix scope is exactly `paiec.pool.EVAL_LOADER`, as the draft has it.
    The lock records the tuple, `frozen_ast` removes exactly it, and a changed
    tuple is out of scope. Item 56 adds detail.
33. `evaluate_timed` does not copy `evaluate`'s `raise ValueError("invalid
    prediction")`.
    * It records an invalid output (not a finite float in [0, 1], or the
      wrong shape) per arm, run and budget, and scores the other arms on.
    * The result is "archive-3 stays" for that arm's candidate, or for all of
      them if A3's arm produced it.
    * The classification code sits in the constants block.

    Items 43 and 44 make it concrete.
34. `record-fix` and `score`:
    * the locked tests must pass unchanged after a fix;
    * re-score the failed tasks plus a fixed verification set (run 0 and the
      last scored run of every regime of each scored stage), which must
      reproduce its stored rows bit for bit; otherwise re-score the whole
      stage;
    * re-run `select`, `summarise` and `gate` from all rows.

    Items 54 and 55 make it concrete.
35. The rows store every call's CPU time for dense runs. Gate (f)'s p99 is one
    fixed estimator over X*'s dense calls in the C core. Items 49 and 50 make
    it concrete.
36. Each stage appends a check log (name, result, UTC) to
    `results/pool_eval.json`. `select` and `gate` read (E4) and (g) from it.
    Item 48 makes it concrete.
37. PB uses the appearance weight 1/(pairs in the run), the same weight as D.
38. RP3:
    * the randomising stream is `default_rng([TIMING_SEED, 3, i])`, with i the
      RP3 run index;
    * draws are `integers(0, 2)` per 0/1 response in pair order and then file
      order;
    * the twin reuses its parent's copy;
    * predictions are captured through a recording factory;
    * the returned dict is discarded.

    Item 52 makes it concrete.
39. `timing`: label-randomised copies on stream `[TIMING_SEED, 9, i]`, not
    labels removed. Adopted; item 53 makes it concrete.
40. `--pool` build: `PYTHONHASHSEED=0` and one BLAS thread, checked as at the
    lock. Adopted; item 47 makes it concrete.
41. The draft `paiec/pool.py:532` has `check_csvs(..., tol=1e-9)`. The design
    says 1e-12, both for `logit_cc` and for `level_cc`; the CSVs agree with
    the formula to 0 and 5.8e-15. impl_plan §2 now says so.

**New items**

42. **`RULE_TEXT`** is the new A4 §4.4 block, byte for byte. It changed in:
    * the PB term;
    * (E3), (E4), M(X) and (f);
    * Reading 4: the precedence words; valid and invalid outputs; A3's case;
      the `EVAL_LOADER` scope; "the classification of outputs included"; the
      re-score clause; the re-run of `select`, `summarise` and `gate`, and
      the outcome when X* changes.

    `RULE`'s thresholds are unchanged. The lock still checks that each one
    appears in the text.
43. **`classify_output(v)`** in the constants block returns valid iff
    `numpy.atleast_1d(numpy.asarray(v, float))` has shape (1,), is finite and
    lies in [0, 1]. Anything that cannot be converted is invalid too.
44. **`evaluate_timed`** calls `classify_output` on every output.
    * It records an invalid one as (arm, run, budget, call index) and never
      raises on it.
    * It keeps scoring the task's other arms. The invalid arm gets no Brier on
      that run; the row records why.
    * `score` counts invalid outputs per arm and regime.
    * `select` and `gate` turn any invalid output into "archive-3 stays" for
      its candidate, and for every candidate if A3 produced it.
45. **The RS rows digest:**
    * `hashlib.sha256("".join(f"{path}\t{sha256}\n" for path, sha256 in
      sorted(rows)).encode("utf-8"))`, with `path` relative to
      `data/regime_sensitivity_rows/` in POSIX form;
    * it refuses unless the files are exactly 0..79.json in each of TUNED,
      READING, AUDIT, MIXTURE, FLAT and 0..59.json in R1B and R1P;
    * RP1 computes it and writes it beside its result;
    * `lock` records it and checks that it equals RP1's;
    * `select` recomputes it immediately before it reads a row, and refuses on
      a mismatch.

    The planning value, 9a601549b2d99739487e7c3808da616f24f9072506671043de878020f59a9b5c,
    is in `a4final/rs_rows_digest.json`, and that script implements the rule.
46. **`PINNED`** holds two digests in full:
    * `dist/archive-3.zip`,
      4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090;
    * `data/pool_eval/a3_prior.json`,
      c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770.

    impl_plan step 1 checks both in full.
47. **`tools/build_submission.py --pool`** refuses to run unless
    `PYTHONHASHSEED=0` and one BLAS thread are set, checked by the same code
    the lock uses.
48. **The check log** is `results/pool_eval.json: checks`.
    * Each entry is (stage, check name, result with the failing values, UTC),
      written by every stage before and after the lock.
    * A check missing from the log has not passed.
    * `select` reads (E4) from the entries of the pre-lock stages, the lock,
      S and RS. `gate` reads (g) from all entries up to the gate.
    * A failure resolved by a recorded fix counts once its re-run passes and
      is logged.
49. **Row timing fields.** Each row holds, per arm and run, the number of
    calls and the sums and maxima of CPU and wall time per call. A dense
    run's row also holds every call's CPU time, in call order. Each row also
    counts fallbacks and invalid outputs.
50. **Latency, computed from the rows:**
    * the mean CPU per call is the summed CPU over the summed calls, over the
      tasks named;
    * (E3)'s maximum is over every call of X on S's formative-size runs and on
      the RS tasks;
    * (f)'s maximum is over every call of X* on C's formative-size runs;
    * (f)'s p99 is `numpy.percentile(x, 99)` (default linear interpolation)
      over every dense call of X* in the C core;
    * (f)'s dense maximum is over the same list.
51. **`select`:**
    * (E4)'s arms are A3, C1, C1m and C2 on pool tasks, and C1, C1m and C2 on
      RS tasks;
    * M(X) uses D on S;
    * PB in `summarise` uses the weight of 37.
52. **RP3:**
    * run 0 is formative-size, run 1 dense, and the twin is run 0's;
    * the copy of run i draws `rng.integers(0, 2)` from
      `rng = default_rng([TIMING_SEED, 3, i])`, once per 0/1 response, in
      the run's pair order and then file order;
    * the twin is `testlike.shifted` applied to run 0's copy;
    * the recording factory wraps each arm's factory and logs (checkpoint,
      call index, prediction);
    * predictions are compared exactly between the two harnesses.
53. **`timing`:**
    * the copy of run i draws as in 52, from `default_rng([TIMING_SEED, 9,
      i])`;
    * the fixed order is S formative-size 0 to 5, S dense 6 and 7, mind2web 8,
      swe_smith 9;
    * it asserts that every response equals its draw and that no Brier or ALC
      is computed.
54. **`record-fix`:**
    * it runs the locked `tests/test_pool.py` and `tests/test_pool_eval.py`,
      unchanged, and refuses unless both pass;
    * it lists the failed tasks;
    * it lists the verification set: run 0 and the last run scored before the
      fix, of every regime of every stage with rows;
    * after the re-score, it compares every verification row with its stored
      row in every field except CPU and wall times, resident set and load
      average; any difference re-scores the whole stage;
    * the deviation record holds the cause, the files, the old and new
      sha256, the task lists and the test result.
55. **After any re-score**, `select`, `summarise` and `gate` are re-run from
    all rows. If the re-run `select` names another X* than the committed
    selection record, the outcome is "archive-3 stays".
56. **`EVAL_LOADER`** does not list its own name, so `frozen_ast` keeps the
    tuple, and any change to it fails the AST rule. `lock` records the tuple
    in `results/pool_eval.json: lock`.
57. **The HELM-hosted cut** (descriptive):
    * a per-appearance row flag, true when the benchmark's own catalogue
      `source_url` has host crfm.stanford.edu (without www.) and first path
      segment `helm` (the rule of `a4final/helm_cut.py`);
    * `summarise` reports D by that flag in C's regimes;
    * S has no such benchmark (planning).
58. **Tests and estimates.**
    * New tests in `tests/test_pool_eval.py`, listed in impl_plan §6:
      * `classify_output` and the invalid-output path, A3's case included;
      * PB's weight, on a toy table where the unweighted mean changes sign;
      * the p99 estimator;
      * the label-randomised copies of RP3 and `timing`, and the recording
        factory;
      * `experiments/__init__.py` in the closure;
      * the RS rows digest, and `select`'s refusal on a changed row;
      * a changed client digest;
      * a changed `EVAL_LOADER` tuple;
      * `record-fix`'s task lists and its refusal on a failing locked test;
      * the re-run X* rule;
      * the check log.
    * Estimates: `pool_eval.py` from 8 to 8.5 h, and `test_pool_eval.py` from
      3 to 3.5 h. The work before the lock is about 20.5 h, on 2026-10-12 and
      2026-10-13.

## Not changed, and open

* **For the author to confirm in commit 1**, beside `recheck.md`'s and
  `nfix_verify.md`'s lists:
  * the precedence clause, whose wording came from the task text;
  * decision 8: pushes of commits 1 to 4, with the approval date of
    2026-10-06 that came from the task text;
  * the four precisions beyond nfix_verify:
    * timing and resource fields are left out of the bit-for-bit comparison;
    * a re-run `select` naming another X* gives "archive-3 stays";
    * a check resolved by a recorded fix counts as passed;
    * mean CPU per call is summed CPU over summed calls;
  * the about 20.5 h estimate.
* **The commit-1 filter.** It is still the author's decision for
  `survey.json` (per-pair in shape) and `code/c1_proto_prior.json` (a full
  prior bundle fitted on the whole pool). PE §13 is unchanged on both.
  impl_plan step 0.3 names them.
* **`dates_log.md` and `nfix_log.md`** are left as the record of their
  rounds. Their "after" digests are superseded by the table above.
* **`dist/` was not hashed** (the hook). impl_plan step 1 does it before any
  build.

## Checks run

* **PE's digest.** It is quoted in full in A4's preamble and by its first 16
  hex in impl_plan step 0.1. No old PE digest and no placeholder is left in
  the three files.
* **`RULE_TEXT`** is ASCII, has no tabs, and its widest line is the
  79-character header line, which is unchanged.
* **The precedence clause** (whitespace normalised) appears once in A4 and
  once in PE.
* **Superseded phrases are gone** from all three files:
  * "labels stripped" (except where PE §15's N8 row says what it became);
  * "every listed arm", "D_S(";
  * "appearance-weighted mean difference";
  * "non-finite prediction" as a rule, "paiec.pool.load_eval_pairs;" as the
    scope;
  * "that commit is the outside time stamp".
* **A4, PE and impl_plan agree** on each of these:
  * `EVAL_LOADER`;
  * A3's invalid-output case;
  * the definition of an invalid output;
  * the push of commits 1 to 4;
  * the p99 estimator;
  * PB's weight;
  * the verification set;
  * the re-run X* rule;
  * `PYTHONHASHSEED=0` for `--pool`.
* **The planning values were recomputed** on 2026-10-11:
  * `rs_rows_digest.py` gives `_compositions.json`'s c1520f4be78e37dc, equal
    to PE's pin, and the client's and `experiments/__init__.py`'s digests
    equal to nfix_verify's;
  * the twins' probabilities are Φ(1) = 0.841, 0.841² = 0.708 and
    0.841⁴ = 0.501.
