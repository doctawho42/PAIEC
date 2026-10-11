# Final check of amendment 4 and p2_pool_eval.md, after vfix_log (2026-10-12)

Checked against repository HEAD 2bfcd17. The two files are
`amendment4_short.md` ("A4") and `p2_pool_eval.md` ("PE"), as `vfix_log.md`
left them, with `impl_plan.md`, `nfix_verify.md` and `nfix_log.md` as the
reference. It looks only for blockers to committing the amendment:

* a condition that cannot be evaluated from stored outputs;
* an undefined term in `RULE_TEXT`;
* a freedom left to change the rule, or to choose, after a scored row exists;
* an A4/PE inconsistency that changes a decision;
* a wrong digest.

What I ran, all descriptive:

* sha256 of the design files and of every file PE pins that can be read;
* the RS rows digest, recomputed from the files' bytes only, without opening
  any row as JSON;
* a format check of `RULE_TEXT`, and a count of the precedence clause;
* reads of `regime_sensitivity.py`'s `RULE_TEXT` (P1a's D),
  `hier.identity_key` and the draft `paiec/pool.py`'s `EVAL_LOADER` and
  `frozen_ast`.

I scored no predictor and computed no ALC or Brier. I read no item text and
opened no formative table. I made no git write. This file is the only one I
wrote.

## Verdict

**Not ready to commit: three blockers (B1 to B3).** Each is closed by one or
two sentences of text, and none needs a score. Everything else holds: the
digests, the A4/PE agreement, the V1 to V13 answers and the remaining terms
of `RULE_TEXT`.

## Digests

| item | computed | quoted | verdict |
|---|---|---|---|
| p2_pool_eval.md | `7806244ca4f5d273b788278e3a05086433403368b6011e8c38503f69fef19302` | A4 preamble (in full); impl_plan step 0.1 (`7806244ca4f5d273`); vfix_log | correct |
| amendment4_short.md | `246a7b612ca6cbc8c6feac697a99cb88b9b31b51af9f9ff78228f0ea34a1b968` | vfix_log | correct |
| impl_plan.md | `7929c187de475e6e30df5dffc9112c0308f781b5ed970f2f809326c4ffcc1765` | vfix_log | correct |
| a4final/rs_rows_digest.py, .json; helm_cut.py, .json | `f92de139…`, `479444fa…`, `f9230d98…`, `f48e6203…` | vfix_log | correct |
| the 520 RS row files (rule of PE §9 item 10) | `9a601549b2d99739487e7c3808da616f24f9072506671043de878020f59a9b5c`, exactly 0..79 / 0..59 | PE §9 item 10 (16 hex); vfix_log item 45 | correct |
| the eight shipped modules, model.py, official, testlike, level_calibration, regime_sensitivity, data, evaluator, baselines, ship_confirm, subject_side, `experiments/__init__.py` | all equal to PE's 16-hex prefixes | PE §2, §9 item 10 | correct |
| streaming_ingestion.py; clone commit | `c6f2610f9c8b6cad…6267`; HEAD `82d330ddcdb16016a3ae9e048db7e588ba2c6a39` | PE §9 item 10; vfix_log 31 | correct |
| `_compositions.json`, benchmarks.csv, pairs.csv, manifest | `c1520f4be78e37dc`, `d88247c984edc5df`, `2e913b8b24c29fc8`, `ff548206…` | PE | correct |
| archive-3's prior.json (= `submission/prior.json` on disk) | `c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770` | PE §2, §9 item 10; `RULE_TEXT` (16 hex) | correct |
| `dist/paiec.zip` | not hashed: a plugin hook blocks `dist/` | `4a882cc7…5090` in PE equals `results/stress_archive.json`, `results/formative_run3.json`, README, `docs/release.md` | consistent; impl_plan step 1 checks the file itself |

Also checked:

* `RULE_TEXT` is ASCII, has no tabs and is at most 79 columns wide.
* The precedence clause, whitespace normalised, appears once in A4 and twice
  in PE: in its preamble and in the V7 row of §15.
* PE quotes no digest of A4, so the digest chain has no cycle.

## Blockers

### B1. The check log does not say which entry counts, and a check fixed before the lock fails (E4) and (g) for good

**The text.** PE §9, "The check log":

* every stage, *before and after the lock*, appends each check it runs;
* `select` reads (E4)'s "every check passed" from the entries of the
  pre-lock stages, the lock, S and RS, and `gate` reads (g)'s from all
  entries;
* the only way a failure can count as passed is "a failure that a recorded
  fix resolved (item 11)", and item 11 covers failures *after* the lock.

**Why it blocks.** The design itself expects failures before the lock:

* §9 item 9: "the cause is fixed before the lock";
* §5.3: "if it trips, scoring does not start and the cause is found";
* §5.1: a balance-check failure, after which an amendment may "keep this rule
  with the failure recorded".

Pre-lock stages are also re-run during development as a matter of course,
and every run appends entries. So two things are undefined:

* whether a pre-lock failure, fixed and then re-run with a passing result,
  counts as passed;
* which entry decides when one check has several.

Read literally, any logged pre-lock failure makes (E4) fail for every X, and
(g) fail too: "no candidate carried: archive-3 stays", whatever the scores.
The §5.1 option of keeping the rule with the failure recorded defeats itself
the same way. Which entries count would be decided after rows exist, by
whoever reads the log, and the decision changes the outcome. That is a
remaining freedom, and the log cannot be read mechanically.

**Fix.** One sentence in PE §9, "The check log". For example:

> "Before the lock, a failed check is fixed and re-run; the lock refuses
> unless the latest entry of every pre-lock check passes, and (E4) and (g)
> read a pre-lock check by that entry. After the lock, a check passes only
> if all its entries pass, or its failures were resolved by a recorded fix
> (item 11) and its re-run passed."

Also delete §5.1's "or keeps this rule with the failure recorded", or say
how that recorded failure is read. As written, it can only give archive-3.
PE changes, so its digest goes again into A4's preamble and into impl_plan
step 0.1.

### B2. JK is undefined when a family fills a run, and has two readings otherwise

**The text.** `RULE_TEXT`, A4:166: "JK(X,R) the largest D(X,R) with all
appearances of one source family removed". D (A4:155) is "the mean over R's
planned runs of the run's pair-mean ALC".

**Read literally**, removing a family recomputes each run's pair-mean over
the run's remaining pairs. A run whose pairs all come from the removed family
then has no pair-mean, and D over "R's planned runs" is undefined.

**How likely that is.** A formative-size run of 5 to 12 pairs repeats a
benchmark with probability 0.25 per draw. MedHELM's ten benchmarks are one
family in C. So a single-family run among C's 120 P-READING and 120 P-AUDIT
runs is a plausible event, not an edge case (roughly one chance in four, on
the sampler's shape).

**The other reading.** PE §6 writes D = Σ_k v_k / Σ_k n_k over clusters, and
the earlier design, `eval.md:546`, defined JK as (Σv − v_f) / (Σn − n_f),
with the appearance weights left unchanged. That reading is always defined,
and in general it gives another number.

**Why it blocks.** Gate (b) needs JK <= -0.001, so the choice between the two
can decide the gate, and the governing text does not make it.

**Fix.** In `RULE_TEXT`'s Terms, for example:

> "JK(X,R) the largest, over source families f, of (sum of v_k over k != f)
> / (sum of n_k over k != f): D with one family's appearances removed and
> every other appearance's weight 1/(pairs in its run) unchanged"

PE §6 says the same. `RULE_TEXT` changes, so A4 and the constant change;
`RULE`'s thresholds do not.

### B3. (E4) and Reading 4 disagree on what one candidate's invalid output does to the others

**The text.**

* `RULE_TEXT` (E4) makes *every* X ineligible unless "every planned S and RS
  task is scored for every arm (A3, C1, C1m and C2 on pool tasks; C1, C1m
  and C2 on RS tasks)".
* PE §5.2 says an arm with an invalid output "is not scored on that run".
* Read together, one invalid output of C2 on one S task makes C1 and C1m
  ineligible as well.
* Reading 4 (A4:233-235) gives "archive-3 stays" "for the candidate whose arm
  produced it, and for every candidate if A3's arm produced it". The A3
  clause only has a use if a candidate's invalid output leaves the other
  candidates eligible.
* impl_plan items 44 and 51 and its §6 test ("scores the other arms; the
  candidate's outcome is archive-3 stays, and every candidate's when the toy
  A3 does it") are written for that narrow reading.

**Why it blocks.** Take C2 with an invalid output on one S task, and C1
otherwise carried. Under (E4) as written, no candidate is carried. Under
Reading 4's evident intent, C1 is carried. Which applies would be decided
after the row exists. The case is improbable (`hier.predict` returns values
in [1e-4, 1 - 1e-4] or 0.5), but V5 was required for the same kind of case.

**Fix.** Choose one reading and write it in `RULE_TEXT`.

* *Narrow:* (E4) "... for every arm, an arm with an invalid output excepted
  (Reading 4)", and Reading 4 "... for the candidate whose arm produced it
  and no other, unless A3's arm produced it".
* *Wide:* Reading 4 "... gives 'archive-3 stays' for every candidate if it
  occurs on an S or RS task or in A3's arm, and for X* if it occurs on a C
  task".

PE §5.2 and §9 item 11 follow the choice. impl_plan items 44 and 51 and the
§6 test follow it too.

## Not blockers (checked)

**A4 and PE agree** on every decision-bearing item:

* the candidates and their planning centres;
* the split, 44 and 88;
* the regimes and the twins' D <= 0;
* the counts: 80/20, 120/40, 520;
* the seeds;
* the hard stops: 10-13, 10-15, 10-16, 10-17, 10-20;
* the fix scope (`EVAL_LOADER`);
* the precedence clause;
* the model.py rule;
* the `--pool` settings;
* the feedback uses (A4 §4.6, `RULE_TEXT` Reading 5, PE §11).

**P1a's D**, which (E1) reads "as P1a defined it", is word for word
`RULE_TEXT`'s D (`regime_sensitivity.py`'s Terms).

**Every other condition can be evaluated from the rows** as impl_plan §5
lists them:

* (a) to (f);
* (E1) to (E3);
* M, Q and X*;
* the identity key, which depends on config only (`hier.py:1918-1927`), not
  on the bundle.

**The draft `paiec/pool.py`'s `EVAL_LOADER`** has the seven names PE gives
and does not list itself, so `frozen_ast` keeps it.

**A re-run `gate` after a fix** is mechanical. Every stream is seeded, so a
re-score changes only what the harness bug changed.

## Changes that affect code (continuing vfix_log's 1 to 58)

**From the blockers**, once the author picks the wording:

59. **B1, the check log.** The lock refuses unless the latest entry of every
    pre-lock check passes. (E4) and (g) read pre-lock checks by that entry,
    and post-lock checks by "all entries pass, or failures resolved by a
    `fixes` record and a passing re-run".
    * Tests: a pre-lock failure followed by a passing pre-lock re-run does
      not fail (E4); a post-lock failure with no `fixes` record does.
60. **B2, JK.** `summarise` and `gate` compute JK = max_f (Σ_k v_k − v_f) /
    (Σ_k n_k − n_f), with the weights 1/(pairs in the run) unchanged.
    * Test: a toy table with one run made of one family's pairs only (JK
      stays defined), and one where recomputing the run means gives another
      value.
    * `RULE_TEXT` (constants block) changes with A4.
61. **B3, (E4) and invalid outputs.** `select` implements the chosen
    reading.
    * Test: C2 invalid on one S task, C1 otherwise carried, gives the
      outcome of the chosen reading.

**Not blockers, but the code must respect them before the lock:**

62. **No row field that a fix in scope changes.** Rows should hold nothing
    of this kind outside the excluded fields (CPU and wall times, resident
    set, load average): no UTC write time, no git commit id, and no digest
    of `experiments/pool_eval.py` or of `EVAL_LOADER`'s definitions.
    Otherwise every verification row of V8 differs after any fix, and the
    whole stage is always re-scored. impl_plan §5's "per task: ...
    digests" should name which digests: the bundle's and the pinned
    modules', not the fixable files'.
63. **The check log holds only what PE calls checks.** That means the lock
    check, manifest and digest checks, the split and CSV checks, the
    realisation and balance checks, RP1 to RP5, the timing limits, row I/O
    and task completion. It does not hold descriptive flags, such as short
    of target (P-MIXTURE is short in both halves, *planning*), convergence
    counts or B0-centre offsets. Logged as failures, these would fail (E4)
    by themselves.

## Preconditions already listed (not text blockers)

* **Two inputs came from the workflow's task text, not from the author.**
  vfix_log records both:
  * the precedence clause's wording;
  * decision 8, the pushes of commits 1 to 4, and their approval dated
    2026-10-06.
* **The author confirms them in commit 1**, as vfix_log's "Not changed, and
  open" lists, before anything is pushed. A push to the public repository
  needs the author's own approval in the session; this check gives none.
