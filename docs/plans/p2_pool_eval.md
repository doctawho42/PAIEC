# P2 pool evaluation: does a pool-fitted hier replace archive-3?

Written on 2026-10-07 and revised on 2026-10-11/12, for amendment 4's calendar
(sections 5.4, 10, 12 and 13), the fetched item tables (sections 3.1, 3.7
and 11), the re-check's N1 to N9 (`docs/plans/pool_design/recheck.md`;
`nfix_log.md` lists the edits) and its verification's V1 to V13
(`nfix_verify.md`; `vfix_log.md` lists the edits; section 15 says where each
is answered), before any predictor was scored on the expanded pool. It is the
full design behind amendment 4 of `docs/plans/p2_final_plan.md`, which pins
this file by sha256 and quotes the decision rule (`RULE_TEXT`) verbatim. The
rule, its terms and its thresholds are the amendment's; this file defines what
the rule's terms name (the data, the candidates, the split, the regimes, the
statistics, the checks) and how the work runs. Only a later dated amendment
changes anything here, with one limit: nothing in the rule, candidates,
split, regimes, seeds, counts, statistics or fix scope changes once a scored
row exists, not even by a later amendment. A scored row is a row of
SEL_SEED, CONF_SEED or the RS guard; amendment 5 only announces the upload,
and the amendment that records the final only records it.

The work behind it counted benchmarks and pairs, fitted per-benchmark item
models, drew run compositions and did prior arithmetic. It computed no ALC and
no Brier on any data. Its notes and scripts are committed under
`docs/plans/pool_design/` (section 13). Every number marked *planning* comes
from them, and the `fits`, `regimes` and `timing` stages re-measure each one
before the lock. "The critic" is `docs/plans/pool_design/critic.md`, whose
required changes R1 to R12 this design answers (section 15).

## 1 Why, in brief

* **The pool grew.** On 2026-10-06 and 2026-10-07 measurement-db grew from six
  benchmarks to 255 catalogue entries, 252 of them public. Archive-3's priors
  rest on five benchmarks and 221 pairs.
* **Pools are exchangeable.** The competition page says benchmarks were
  "randomly assigned to separate training and test pools before curation", and
  that the collections may grow while the assignment and the sampling design
  stay fixed. A public benchmark held out of a fit stands in for a hidden one.
* **The difficulty is the level.** The pool's benchmark levels have mean +0.06
  and sd 1.54 (data.md §5); the shipped LEVEL's median sits at their 15.7th
  percentile. The formative readings sit low: READING -0.715 (sd 1.772), AUDIT
  -1.10 (2.00), TUNED -1.29 (1.70), on 17 pairs from 7 benchmark ids. A level
  learned from the pool wins on unselected held-out benchmarks by construction,
  so such benchmarks cannot decide alone.
* **So** gains are required at real dates in pool regimes tilted to the
  readings; in the same runs with every subject's dates moved by the 1.25 years
  LEVEL was calibrated with, no loss is allowed (D ≤ 0); elsewhere losses are
  bounded. The date term of the subject ridge moves the level centre with the
  subjects' dates, so no one mu0 holds archive-3's centre under both dates. C1
  keeps archive-3's mu0 and holds its centre at real dates (to 0.009), C1m
  matches mu0 to hold it under the shift, and C2 changes the level (section
  4). Between them the rule reads the pool's rest of the prior at archive-3's
  level centre (as section 4.2 defines it) under each date assumption: as a
  gain at real dates (C1), and as a no-loss condition under the shift (C1m).

## 2 Scope

* **What may differ in archive-4.** `submission/prior.json`; for C1m and C2 also
  the `LEVEL` literal of `submission/model.py` (line 68); and, for every
  candidate, model.py's module docstring (lines 1-40; lines 13-14 describe
  the build writing LEVEL in) and the comment block above `LEVEL` (lines
  62-67; lines 62-64 describe the same), so that their text is true for the
  configuration that ships. An AST comparison that ignores docstrings,
  comments and the `LEVEL` value must find model.py otherwise identical to
  archive-3's (section 10).
* **What stays byte-identical** (sha256, first 16 hex, at a725df9; the lock
  checks them): `paiec/__init__.py` e3b0c44298fc1c14, `hier.py`
  d9a9561202b63b9d, `prior.py` 3a8a8c7dcd0da53e, `predict.py`
  46193d8dd7701ec2, `fitting.py` ec3af9224f48309a, `irt.py` 3f7d93e5d012699c,
  `subjects.py` 79eaa16553a90e6b, `mcq.py` f240425d01fb7278. `fit_hyper` is not
  refactored; its arithmetic (`paiec/prior.py:756-807`) is copied into the
  offline module and checked against it (RP4).
* **New or changed offline code:** `paiec/pool.py` (not shipped, never imported
  by a shipped module); `tools/pool_benchmarks.json` and
  `tools/pool_families.json` (benchmark names and catalogue keys only);
  `experiments/pool_eval.py` and its tests; `tools/build_submission.py` gains a
  `--pool` path, written after the lock and bound by it (section 10).
* **S1 and S2 stay closed.** Nothing else ships.
* **archive-3 stays at hand.** Before any build, `dist/paiec.zip` (sha256
  `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090`, as
  `results/stress_archive.json` and `results/formative_run3.json` record it)
  is copied to `dist/archive-3.zip`, and its `prior.json` (sha256
  `c7ce3b845d584f95…`) to `data/pool_eval/a3_prior.json`. The lock records
  both digests in full (section 9 item 10). The comparator reads that copy
  after checking its digest.

## 3 Data

### 3.1 Files and their verification

* **Revision** `54074453da7b9911be0f48e4c560be547bc84d21`, fetched by
  `tools/fetch_pool.py --items` into
  `data/pool/<b>/formatted_tables/{benchmarks,subjects,responses,items}.parquet`,
  with the item tables of cseo_bench and qatch skipped (`--items-skip`,
  committed at 2bfcd17; neither is in L, section 3.7). The manifest it
  rewrites, `data/pool/manifest.json` (`fetched_utc`
  2026-10-08T04:04:26Z; `items_skipped` names the two), lists every file's size
  and sha256. It supersedes the pre-items manifest (sha256 `a58a23fb…`, kept as
  `data/p2_work/pool_manifest_no_items.json`), whose benchmarks, subjects and
  responses entries it keeps unchanged for all 255 entries. Its own sha256
  (`ff548206…` on 2026-10-11) is recorded by the `fits` stage and pinned by the
  lock.
* **Verification.** `paiec/pool.py` checks each pool file's size and sha256
  against the manifest before it first reads the file in a process, and raises
  on a mismatch or a missing entry. The `fits` stage checks every file of L. A
  mismatch is a check failure.
* A later revision is a new fit and needs a new amendment.

### 3.2 Items present (Mode A)

The item tables are fetched, so pool runs and fits see items as the hidden test
does.

* **Item dicts.** `item_content` is the item's `content` and `item_features`
  its `item_features` ("" when null). `interactors` is the `interactors` value
  of the item's first 0/1 response row in file order, benchmark-wide (""
  when null), so one item has one dict, as the official input has
  (`docs/protocol.md`: four keys per item). `benchmark_id` is the anonymous id
  `official._slots` assigns.
* **What that switches on, for every arm.** hier's multiple-choice floor
  (`paiec/hier.py:1944-1953`, `Flags.floor` on) and its group effects
  (`:532-542`) act on pool runs exactly as at test time. A3 gets them too.
* **The group share** is fit_hyper's rule (`paiec/prior.py:761-763`): the median
  share of the keyed multi-pair fits of L ∖ excl, capped at G_CAP 0.25, and
  REFERENCE's 0.138 if no fit has keys. No candidate fixes it. T and its split
  into sigma_d and sigma_g are both refitted and both tested.
* **Item text never enters a file.** Not a committed file, a cache, a row or a
  log. Caches and rows hold item ids and sha256 digests of the content; the
  evaluation loader reads the content of a drawn pair's items from
  `items.parquet` at run time, through `items_of` (next item).
* **Reading items** (`paiec.pool.items_of`; the re-check's N8). It reads the
  columns `item_id`, `content` and `item_features` only, never `verifier` or
  `grading_criterion`, in bounded batches:
  `pyarrow.parquet.ParquetFile(path, buffer_size=READ_BUFFER,
  pre_buffer=False).iter_batches(batch_size=ITEM_BATCH, columns=...)`, with
  READ_BUFFER > 0 (a buffered stream, not the whole column chunk) and
  ITEM_BATCH at most 256 rows. It keeps the rows of the requested item ids and
  drops each batch after use. A filtered whole-table read decompresses whole
  column chunks, and two item files of E are large (parquet metadata):
  mind2web, in C with one pair, holds 17,153 items with 3.13 GB of
  uncompressed `content` in 2 row groups, the larger 1.84 GB; swe_smith, in C
  with three pairs, has a 1.54 GB `verifier` column in row groups of up to
  1.19 GB. Neither has 4 pairs, so neither enters a dense run, but both can
  enter any formative-size run of C, and the fits read both.
* **The mixed benchmarks** keep the 0/1-share rule of section 3.3; their
  `grading_criterion` is not read by the rule. The `fits` stage reports,
  descriptively, how the 9 mixed benchmarks of L declare their items' scales,
  from a separate read of `item_id` and `grading_criterion` of those nine
  only, in the same bounded batches (counts written, no text); `items_of`
  never serves it.
* **Sizes and disk** are in section 3.7.

### 3.3 The level part L, the subject part U, the evaluation set E

* **L**: catalogue `release` public and `granularity` item; `response_type`
  binary, or mixed with a 0/1 share of at least 0.95 among non-null responses;
  at least one eligible pair. 137 benchmarks (128 binary, 9 mixed), the legacy
  five included. matharena is mixed at 0.9991; mmdocrag is fractional and out.
* **U** = L less the nine direction-flagged benchmarks: `lower_is_better`
  crow, emoji_attack, harmbench, jailbreakbench, llmail_inject, wmt_mqm;
  `unordered` alignment_faking, safeagentbench, tensortrust. 128 benchmarks,
  8,388 eligible pairs. Their standings would enter the ridge and the identity
  table with the wrong sign or none; their levels and item variances stay in L.
* **E** = L less the legacy five: 132 benchmarks, 12,668 eligible pairs,
  including the nine flagged and 12 single-pair benchmarks. The legacy five
  enter the evaluation only through the RS guard.

### 3.4 Pairs and the caps of the fits

* A pair is (subject_id, benchmark); its responses are the 0/1 values of
  `responses.parquet` in file order, nulls and other values dropped. An item's
  label is its first 0/1 response in file order (`paiec/official.py:356-359`,
  `paiec/prior.py:250-260`). Eligible: at least 80 distinct items
  (`paiec/official.py:66-80`). The subject is the eight visible fields
  (`paiec/data.py:20-21`); `display_name` is never read.
* **Caps, for the fits only, never for evaluation runs.**
  * Items: each pair keeps its first M_p = 2,000 items in one benchmark-wide
    order, ascending `stable_hash('pool-item', b, item_id)` (keeps sib200's
    pairs, which a benchmark-wide cap dropped; code.md §2).
  * Pairs: K = 100 per benchmark, round-robin over canonical names, ascending
    (j, `stable_hash('pool-name', c)`), c the pair's `canon_name` (or its
    subject_id when it has none), j its rank among the benchmark's pairs with
    the same c by `stable_hash('pool-pair', b, subject_id)`. `stable_hash` is
    `paiec.evaluator`'s. *Planning* (`amend4_order.py`): on U, 3,136 rows and
    222 names on two or more benchmarks, against 176 in subject-id order.

### 3.5 Source families

* **Rule.** Two catalogue entries are joined when they share a normalised
  `source_url` key (lower-cased host without `www.`, plus the first two path
  segments) or a normalised `paper_url` (arXiv id without version, OpenReview
  forum id, DOI, else host plus path). A family is a connected component.
  `paiec/pool.py` writes the table to `tools/pool_families.json` (each
  benchmark's family id, the alphabetically first member, and the joining
  keys); `pool_design/a4final/families.py` is its planning twin and must agree.
* **Result** (*planning*, all 255 entries): 230 families, 10 with two or more
  members. The 8 that have two or more members in L, 26 benchmarks, all in E:

  | family | members in E |
  |---|---|
  | VLMEval (huggingface.co/datasets/VLMEval) | ai2d_test, hallusionbench, mathvista_mini, mmbench_v11 |
  | MedHELM (arXiv 2505.23802) | medhelm_ehr_sql, _head_qa, _med_mcqa, _med_qa, _medbullets, _medcalc_bench, _medec, _medhallu, _pubmed_qa, _race_based_med |
  | arXiv 2412.12417 | bridging_gap, helm_afr |
  | HELM safety, arXiv 2402.04249 | harmbench, helm_bbq |
  | ScalingIntelligence | kernelbench, monkey_power_laws |
  | evaluate-llm-on-korean-dataset | kmmlu, kormedmcqa |
  | open-llm-leaderboard-old | mmlu, truthfulqa_mc |
  | McGill-NLP | visualwebarena, workarena |

  The legacy five are singletons. Siblings agree closely: medhelm_med_qa and
  medhelm_medbullets correlate 0.967 over 13 shared names (critic, R3).

### 3.6 Exclusion: whole families

* **Pool runs.** For a run with benchmark set B_run, excl = F(B_run), every
  benchmark of L in the family of a benchmark of B_run. Every candidate trains
  on L ∖ excl (level part) and U ∖ excl (subject part). No source of the run
  informs any target's prior, which is the hidden test's situation if the
  sources were assigned whole.
* **RS tasks.** excl is the family of the target's legacy parent, which is the
  parent alone, as P1a's Maker did (`experiments/regime_sensitivity.py:652-700`).
* **Cost** is unchanged: the per-benchmark fits do not depend on excl, so a run
  costs one `from_standings`.
* **Benchmark-only exclusion is not scored.** The descriptive cut of D by
  whether an appearance's benchmark has family members in E (section 6) shows
  where siblings matter.

### 3.7 Item files and the multiple-choice benchmarks (R12)

Sizes before the fetch come from the hub's file metadata at the revision
(`a4final/item_sizes.py`, no download); the facts after it from the manifest and
the files on disk (`a4final/items_present.py`, which hashes the files of L and
parses none; measured 2026-10-11 17:53 UTC, written 2026-10-11/12); the
multiple-choice facts from catalogue metadata and from the shipped floor's
counts on the items (`a4final/mcq_catalogue.py`, counts only).

| | value |
|---|---|
| item files of the 252 public benchmarks, by the hub's metadata | 11.31 GB in all; L 2.28 GB; E 2.17 GB |
| largest item files | cseo_bench 5.69 GB (not in L), scale_mrt 0.83, qatch 0.71 (not in L); in L: mind2web 0.52, swe_smith 0.25, tensortrust 0.24 |
| item tables fetched | 250 of the 255 catalogue entries: every public benchmark but cseo_bench (ordinal, 5.69 GB) and qatch (continuous, 0.71 GB), skipped because neither is in L (nor in E); the three withheld entries (imagenet_hard, swe_together, visual_memory) have only a catalogue row |
| item tables of L | present for all 137; all 548 files of L (four tables each) match the manifest in size and sha256 on disk; item files: L 2.28 GB, E 2.17 GB, all 250 4.92 GB |
| the other three tables | the benchmarks, subjects and responses entries of all 255 entries equal the pre-items manifest's |
| `data/pool` on disk | 7.42 GB |
| free disk | 11.24 GB (10.47 GiB) on 2026-10-11 17:53 UTC, after the fetch; 7.9 GB on 2026-10-07 20:06, while the fetch was still writing cseo_bench's items, which the manifest's run skipped |
| E's benchmarks the catalogue calls multiple-choice (*planning*) | 21: S 4 (kmmlu, mmlu, mmlupro, tumlu), C 17 |
| E's benchmarks on which the shipped floor is positive for at least half of the items read (at most 20,000 a benchmark; 131 of 132 read; *planning*) | 21: S 3 (egoschema, kmmlu, kormedmcqa), C 18; 43 have any positive item |

Only L's item files are needed (2.28 GB), and all are present. The two flags
overlap only in part: mmlu, mmlupro, tumlu, bbq and perumedqa are flagged by the
catalogue but almost never floored, and dave, agi_elo and egoschema the reverse,
because the floor reads options from the item text.

* **The catalogue flag** (`MCQ_RE`): `\b(multiple|multi)[\s-]*choice\b|\bmcqa?s?\b`,
  case-insensitive, on `name`, `one_line_description`, `description` and
  `benchmark_features` of the catalogue row. Never on item text.
* **Use.** D is reported in every regime with and without the
  catalogue-flagged benchmarks, and by the floor-active share (descriptive). In
  Mode A the floor acts on these benchmarks for every arm on pool runs, as it
  will at test time, so the cut shows where the floor and the candidates'
  smaller T interact.

## 4 The candidates and the comparator

### 4.1 The pool build PB(excl)

excl is F(B_run) on a pool run, the parent's family on an RS task, and empty for
the archive.

1. **Fits.** For every benchmark of L with at least two pairs after the caps,
   `f_b = paiec.prior.benchmark_fit(ps_b)` at its defaults (max_iter 500, tol
   1e-4; `paiec/prior.py:229`), on the Mode A item dicts. Each fit is kept as
   returned, converged or not; the counts are reported. The fits do not depend
   on excl; they are computed once and cached (section 9).
2. **Subject prior.** The rows `paiec.prior.standings` would build
   (`paiec/prior.py:355-367`) from the multi-pair fits of U ∖ excl, then
   `paiec.prior.from_standings(rows, 2.0, sorted(L ∖ excl), sorted(excl), None)`.
   No design terms; the provider threshold stays min_count 8.
3. **Hyperparameters**: fit_hyper's arithmetic (`paiec/prior.py:756-807`) on
   these fits: T the median `item_var` of the multi-pair fits of L ∖ excl; the
   share by fit_hyper's rule (section 3.2); sigma_g, sigma_d = sqrt(share T),
   sqrt((1 - share) T); th2 = clip(tau2_res, 0.01, 0.2), sigma_theta =
   sqrt(th2), id_cap 0.1; sigma_delta = sqrt(max(s2_res - u_mean - th2, 0.1))
   × 1.4; sigma_attr = sqrt(max(total_var - s2_res, 0)); slip, guess,
   text_share, nu_mu, shift at Hyper's defaults (0.01, 0.5, 0.2, 0, 0);
   `excluded` sorted(excl), `included` sorted(L ∖ excl).
4. **The bundle** is `paiec.prior.to_json(prior, hyper)`: archive-3's keys and
   no per-benchmark level.

### 4.2 The B0 centre (the level functional)

For an arm X and a set Σ of pair appearances,

c(X, Σ) = mean over Σ's benchmarks, each once, of the mean over the benchmark's
pairs s of (1 - slip) E_z[sigmoid(mu0 + m_X(s) + sqrt(sigma_mu² + V_X(s) +
sigma_delta² + T) z)],

with (m_X(s), V_X(s)) = (shift + attr_scale × attribute mean, sigma_theta² + u),
X's theta prior without its identity term (`paiec/hier.py:1895-1906`),
T = sigma_d² + sigma_g², z standard normal on 80 Gauss-Hermite points. No floor,
no group key, no text term. Σ_real is E's eligible pairs at their real dates;
Σ_DS is the same pairs with release and access dates moved by 1.25 years
(`paiec.testlike.shift_date`, RS's `date_shift`). Undated subjects do not move.

**Why it matters** (R1). Archive-3's ridge has a date coefficient of 2.414 per
400 days, the pool ridge one of about 0.38 (linear date column, no clip,
`paiec/subjects.py:240-241`). LEVEL was calibrated on test-like runs whose
subjects, released in 2025 or later, were dated 1.25 years later
(`paiec/testlike.py:514-517`). So the same mu0 is a different effective level
for the two ridges, and the difference depends on the subjects' dates. The
draft's claim that C1's centre stays within 0.024 of archive-3's on every
reference set is withdrawn: it held only with undated subjects imputed at the
median date (critic, R1).

*Planning* (`a4final/b0_dates.py`: the prototype fits of U, items-blind; the
`fits` stage recomputes the table on the final fits for every arm and writes it
to `results/pool_fits.json`):

| arm | Σ_real: all | dated | undated | Σ_DS: all | dated | undated |
|---|---|---|---|---|---|---|
| A3 | 0.275 | 0.255 | 0.297 | 0.330 | 0.347 | 0.297 |
| C1 (mu0 -2.5) | 0.284 | 0.282 | 0.285 | 0.294 | 0.299 | 0.285 |
| C1m (mu0 -2.035) | 0.320 | 0.318 | 0.321 | 0.330 | 0.335 | 0.321 |
| C2 (+0.16, 2.31) | 0.509 | 0.506 | 0.510 | 0.520 | 0.526 | 0.510 |

4,092 of E's 12,668 pairs are dated. At real dates C1 sits +0.009 from A3
(dated +0.027, undated -0.013); at +1.25 years it sits -0.037 (dated -0.049).
So C1 holds archive-3's level only at real dates; under the date shift it is
also a level change. C1m holds it under the shift (0.000; dated -0.012,
undated +0.024) and sits +0.045 above it at real dates (dated +0.063,
undated +0.024). Neither does both, so the rule reads every candidate under
both dates: gains at real dates (P-READING, P-AUDIT) and no loss, D ≤ 0, in
their date-shifted twins (section 5.3). For C1 the gains are read at
archive-3's level centre and the twins at a centre 0.037 lower; for C1m the
gains include its +0.045 centre shift, and the twins read the pool's rest at
archive-3's level centre, where it may not lose. The rule does not split
C1m's gain at real dates into its centre shift and its rest; the twins only
require that the rest not lose where the centre is held.

"Level centre" here and below is c on E's pairs for the build with nothing
excluded (mu0_m rounded to 3 decimals). A pool run's build without its
families, and a regime's own pairs, can sit off these figures. The `fits`
stage reports, for 50 runs of each half, the mu0 the matching rule would give
without the run's families (section 4.3), and the `regimes` stage reports
c(X, ·) - c(A3, ·) for every candidate on every regime's planned appearances,
at the regime's dates, with the excl-empty builds (descriptive).

### 4.3 The candidates

| | subject prior and widths | mu0, sigma_mu, attr_scale | identity |
|---|---|---|---|
| **C1** | PB | LEVEL: -2.5, 2.5, 0.5 | archive-3's guards (clip [0.01, 0.2], id_cap 0.1) |
| **C1m** | PB | mu0_m, 2.5, 0.5 | as C1 |
| **C2** | PB | the pool's EB level, 0.5 | as C1 |

* **C1m's mu0_m** solves c(C1m, Σ_DS) = c(A3, Σ_DS) for the excl-empty build,
  by bisection on [-10, 10] to 1e-6, rounded to 3 decimals. It is computed once,
  at the `fits` stage, used for every run and for the archive, and locked.
  *Planning*: -2.035. For 50 runs of each half the `fits` stage also records
  the mu0 the rule would give without the run's families (descriptive).
* **C2's level** (the draft's choices, unchanged). T_full is step 3's T with
  excl empty. For every benchmark b of L, single-pair ones included,
  g_b = `benchmark_fit(ps_b, s2e=T_full)`, computed once and cached. For b in
  L ∖ excl: level_b = g_b.mu + mean(g_b.t) minus the mean over b's capped
  subjects of `prior.attribute(s)[0]` under this build's subject prior;
  v_b = g_b.mu_var; |level_b| > 10 dropped and counted. The summary (m, s2) is
  the precision-weighted moment iteration: start s2 = max(var(levels, ddof 1) -
  mean(v), 1e-6); at most 500 times w = 1/(s2 + v), m = Σw·level/Σw,
  s2 = max(Σw((level - m)² - v)/Σw, 1e-6); stop when s2 moves by under 1e-10.
  mu0 = m, sigma_mu = sqrt(s2) × sqrt(1 + 1/n), n the levels used; no WIDEN[0];
  flagged benchmarks included; attr_scale 0.5. Pool runs use each run's
  unrounded (m, s2); the archive's `LEVEL` literal is the excl-empty value
  rounded to 3 decimals, written into the locked bundle at the `fits` stage.
  `mu_var` is Var(mu), not Var(mu + mean t), so s2 is slightly understated
  (`paiec/prior.py:335` returns no covariance); the `fits` stage reports the
  size of that term's omission as the mean of `s2_t / n_pairs` over the levels
  used. *Planning*: N(+0.16, 2.31²) at T 5.80.
* **Order of preference for ties:** C1, C1m, C2, by increasing departure from
  archive-3: prior.json; prior.json and mu0; prior.json, mu0 and sigma_mu.

**What each contrast reads, and where the rule reads it** (centre
differences on E's pairs, section 4.2, *planning*):

| contrast | in P-READING and P-AUDIT (real dates) | in their twins (+1.25 years) |
|---|---|---|
| C1 - A3 | the pool's rest at archive-3's level centre (+0.009); gains: M, (a), (b) | the rest with the centre 0.037 lower (dated 0.049); no loss: D ≤ 0 in (E2) and (c) |
| C1m - A3 | the rest with the centre 0.045 higher (dated 0.063); gains: M, (a), (b) | the pool's rest at archive-3's level centre (0.000); no loss: D ≤ 0 in (E2) and (c) |
| C2 - A3 | the rest with the pool's level, mu0 and sigma_mu (centre +0.233); gains: M, (a), (b) | the same (centre +0.190); no loss: D ≤ 0 in (E2) and (c) |
| C1m - C1 | a pure shift of mu0 by mu0_m + 2.5 (descriptive) | the same (descriptive) |
| C2 - C1 | the pool's level on the pool's rest (descriptive) | the same (descriptive) |

So a candidate that is eligible on S and passes the gate on C has, on point
estimates: as C1, gained at archive-3's level centre at real dates and not
lost under the shift, where its centre is lower; as C1m, gained at real dates
with its centre 0.045 higher and not lost under the shift, where its centre
is archive-3's; as C2, gained and not lost with the pool's level under both
dates. Only the no-loss half of C1m's reading is at archive-3's centre.

"The rest" is the standings, the ridge (its date term included), the identity
table, T and its split, sigma_delta, sigma_attr and sigma_theta.

### 4.4 C3 is not a candidate (R4)

C3 lifted the identity guards (th2 unclipped, id_cap 1.0). It was motivated by
code.md §5's identity slopes, which used every benchmark, C's included. They
were recomputed with no C cell anywhere (`a4final/slopes_s.py`): standings rows
of U on S plus the legacy five only (prototype fits; matharena, chatgpt_drift,
matharena_platform and moe_cap fitted the same way), from_standings' ridge,
residual cells and offsets on those rows alone, held-out cells on S, other
cells on S or the legacy five. The criterion, code.md §5's own and fixed before
the computation: C3 stays only if the model weight at the raw tau2,
tau2 / (tau2 + s2d / k) with k the bin's mean number of other cells, is at or
below the slope for k = 1 and for k ≥ 3 pooled.

| bin | cells | slope (95% interval over cells) | model weight |
|---|---|---|---|
| k = 1 | 48 | 0.61 (0.43 to 0.80) | 0.43 |
| k = 2 (not in the criterion) | 28 | 0.55 | 0.60 |
| k ≥ 3, mean k 5.4 | 67 | 0.76 (0.61 to 0.90) | 0.80 |

40 benchmarks with rows, 1,001 rows, 74 linked names, tau2_res 1.42, s2d_res
1.87. The weight exceeds the slope at k ≥ 3, so C3 is dropped. The benchmark-unit
split of the critic gives the same verdict (k ≥ 3: slope 0.74, weight 0.82), and
so does the full-pool prototype's tau2 and s2d (weight 0.77 at k = 5.4). The
`fits` stage repeats the computation on the final fits for the record; it does
not reopen the decision. With C3 out, C's confirmation no longer rests on a
candidate designed on C's identity statistics.

### 4.5 Not candidates

| option | why not |
|---|---|
| C3, the identity guards lifted | section 4.4 |
| C1m matched on dated subjects only (mu0 -1.891 planned) | one matching rule, on all of E's pairs, fixed in advance |
| a Student-t or mixture level | `_check_bundle` refuses it; 5 to 10 times the time per call |
| `source_model_label` read in `canon_name` | a change to the shipped `prior.py` |
| run-time EB of the level | measured before and not shipped |
| two changes in one arm | a fourth arm |

### 4.6 The comparator A3

* **Pool runs:** `paiec.hier.HierPredictor(*paiec.prior.load("data/pool_eval/a3_prior.json"))`
  with default flags: archive-3's runtime (the in-repo `hier.py` has its
  digest; `check_run` and RP2 hold the archive bit-identical to it). No legacy
  benchmark enters a pool run, so A3 needs no refit.
* **RS tasks:** P1a's stored rows, `data/regime_sensitivity_rows/<R>/<i>.json`,
  key `res.ship.b`, full-precision floats under library digests equal to
  HEAD's. They sit under the gitignored `data/`, and
  `results/regime_sensitivity.json` records only `_compositions.json`'s
  digest, so the 520 files are pinned as one digest (section 9 item 10): RP1
  computes it, the lock records it, and `select` re-verifies it immediately
  before it reads them. RP1 re-scores 14 of the 520 against their rows.

## 5 The evaluation

### 5.1 The split into S and C (R2, R3)

* **Units.** A unit is a source family's members in E (section 3.5); most
  units are single benchmarks. A unit's level is the mean `level_cc` of its
  members; a benchmark's `level_cc` is the mean over its eligible pairs of the
  pair logit log((k + 0.5) / (n - k + 0.5)), n the pair's distinct items and
  k the items whose first 0/1 response is 1 (section 3.4). `paiec/pool.py`
  derives n, k, the pair logits and `level_cc` from the manifest-verified
  responses; the `split` stage checks them against the planning tables
  (`pairs.csv`: n and k exactly, `logit_cc` to 1e-12; `benchmarks.csv`:
  `level_cc` to 1e-12; a mismatch is a check failure) and writes them to
  `data/pool_eval/levels.json` (benchmark, subject_id, n, k; no item id),
  which the lock hashes and every later stage reads. A unit's kind is mixed
  if all members are mixed, else binary; it is flagged if any member is
  direction-flagged.
* **Rule.**
  1. Flagged units: sort by level, lowest first, ties by unit id; cut into
     consecutive blocks of 3; within a block, order by the sha256 hex digest of
     the UTF-8 bytes of `repr(("pool-eval-split", unit_id))`; the first
     `round(size / 3)` go to S.
  2. Other units, within each kind: blocks of 8 in the same order; the first
     `round(3 * size / 8)` go to S.
  3. Every benchmark goes to its unit's half. No family straddles the halves.
* **Result** (*planning*, `a4final/split_v2.py`, rule `family`; the `split`
  stage must reproduce its halves exactly):

  | half | benchmarks | pairs | level mean / sd | below -1.5 / -3 | ≥ 4 pairs | flagged | mixed | multi-member families |
  |---|---|---|---|---|---|---|---|---|
  | S | 44 | 988 | -0.00 / 1.29 | 3 / 2 | 34 | crow, emoji_attack, safeagentbench | 3 | 2 |
  | C | 88 | 11,680 | +0.15 / 1.45 | 10 / 3 | 73 | the other six (helm_bbq joins as harmbench's sibling) | 5 | 6 |

  S: adaptivestep, alpha_sql, appworld, averimatec, bertaqa, bird_sql,
  causal_axioms, chatgpt_drift, critic_discernment_game, crow, devbench, disco,
  editbench, egoschema, emoji_attack, evouna, faithcot, financebench,
  finegrain_t2i, healthadminbench, kmmlu, kormedmcqa, llm_bp_tag,
  matharena_platform, mj_bench, mmlu, mmlupro, moe_cap, naep_llm_students,
  nester, ood_prediction, openbiorq, pku_saferlhf, prox, safeagentbench,
  software_world_models, sugarcrepe, swepolybench, tau2_bench, truthfulqa_mc,
  tumlu, visual_riddles, vl_rewardbench, workbench_revisited. C is the rest of E.
* **Balance check** (locked; computed at `regimes` on the argmin knobs): in
  every tilted regime, the flagged share of the tilt weight differs by at most
  0.10 between S and C. The weights are those the regime's sampler draws
  with (section 5.3): for the dense regimes, restricted to the benchmarks
  with at least 4 eligible pairs, then capped. *Planning*: largest
  difference 0.052 (D-READING: S 0.166, C 0.114; `a4final/dense_cap.py`),
  0.026 among the formative-size regimes (P-TUNED; `a4final/split_v2.py`);
  the largest single-benchmark weight is 0.10 in S (the cap) and 0.077 in C
  (monkey_power_laws). The figure of 0.027 quoted before was D-READING on
  the unrestricted weights. If the check fails, the lock stops, and a dated
  amendment, before any scoring, adopts the critic's benchmark-unit rule (R2;
  S 49, C 83 planned, which also passes the check, D-READING 0.030 on the
  dense sampler's weights); `RULE_TEXT` names 44 and 88, so no switch is
  automatic.
* **Disclosure.** The split was revised from the draft's on levels, direction
  flags and families only, before any predictor was scored.

### 5.2 Run shapes and protocol

* **Formative-size runs** (`paiec/testlike.py` Sampler, :559-763, on real
  benchmarks; run shape :507-537). n ~ U{5, ..., 12} pairs drawn one at a time
  from the half's benchmarks. With probability 0.25, and if a benchmark already
  in the run has a free pair, the next pair joins one of those; otherwise it
  opens a new one. Either way the benchmark is chosen ∝ w_b and the pair
  uniformly among its free pairs (not yet drawn; subject_id and eight visible
  fields not yet in the run). Each pair keeps a uniform random subset of at
  most 160 items; then `official.sample_run(cut, rng, cap=1000,
  n_pairs=len(cut), weighting="pair", min_items=80)` cuts proportionally.
  Subjects keep their real dates.
* **Dense runs.** 7 benchmarks drawn without replacement ∝ w_b among the half's
  benchmarks with at least 4 eligible pairs (34 in S, 73 in C); min(8, n_b)
  pairs from each, uniformly, with distinct eight-field dicts; at most 120 items
  a pair; no cap.
* **Protocol**, as `paiec/official.py` plays it: a persistent 50/50 split at
  seed 0, scope 'pair' (:83-100); the platform's sha256 acquisition
  (:310-319); the first recorded response revealed; every 0/1 response of an
  evaluation item scored; one shared `labeled` per checkpoint and a fresh
  instance per checkpoint and arm; budgets 0, 1, 3, 7, 15, 31; anonymous ids
  as `official._slots` assigns them (:348-349). Pool runs keep every 0/1
  response of the drawn pairs, in file order, uncapped, with Mode A items.
* **The harness** (R7) is `experiments/pool_eval.py`'s `evaluate_timed`: a
  copy of `level_calibration.checkpoints` and `evaluate` (:508-561) that
  records, per call, `time.process_time` and `time.perf_counter` differences.
  RP3 checks this copy against `official.run_official`. "CPU time" in the rule
  is `time.process_time`; "process-hours" are CPU seconds summed over
  processes, divided by 3,600.
* **Invalid outputs** (the verification's V6). `evaluate` raises
  `ValueError("invalid prediction")` on any value that is not finite, lies
  outside [0, 1] or has the wrong shape (:546-547); `evaluate_timed` does not
  copy that raise. An *invalid output* is anything a call returns that is not
  one finite float in [0, 1] (after `numpy.atleast_1d`, shape (1,)).
  `evaluate_timed` records each one (arm, run, budget, call index), scores
  the other arms of the task on, and never raises on it; the arm that
  produced it is not scored on that run. The consequence is the rule's
  (Reading 4): "archive-3 stays" for that arm's candidate and for no other
  (the other arms stay eligible, and (E4) does not require that arm on that
  run), unless A3's arm produced it, in which case it holds for every
  candidate (the final check's B3). The classification is one function in
  the constants block (`classify_output`), so it lies outside the scope of a
  post-lock fix (section 9 item 11). `hier.predict` never raises and returns
  a value in [1e-4, 1 - 1e-4] or 0.5 (`paiec/hier.py:2039-2071`), so the case
  is not expected.
* **Timing in the rows** (V10). Every row holds, per arm, the number of
  calls and the sums and maxima of their CPU and wall times; a dense run's
  row also holds every call's CPU time, in call order, so that gate (f)'s
  p99, `numpy.percentile(x, 99)` with linear interpolation over all of X*'s
  dense calls in the C core, can be read from the rows alone.

### 5.3 Regimes, tilt and knobs

| regime | runs | target (realised level of evaluated responses: mean, sd ddof 0, over appearances) | role in the rule |
|---|---|---|---|
| P-READING | formative-size | -0.715, 1.772 | gain |
| P-AUDIT | formative-size | -1.10, 2.00 | gain |
| P-READING-DS | P-READING's runs, dates +1.25 years | as P-READING | no loss, D ≤ 0 |
| P-AUDIT-DS | P-AUDIT's runs, dates +1.25 years | as P-AUDIT | no loss, D ≤ 0 |
| P-TUNED | formative-size | -1.290, 1.696 | loss bound +0.002 |
| P-MIXTURE | formative-size | share below -3 0.118, rest mean -0.226 | loss bound +0.002 |
| P-FLAT | formative-size | none (w_b = 1) | loss bound +0.002 |
| D-READING | dense | as P-READING | loss bound +0.001 |
| D-FLAT | dense | none | loss bound +0.001 |
| RS-TUNED, -READING, -AUDIT, -MIXTURE, -FLAT, -R1B, -R1P | P1a's seed-11 compositions | P1a's | guard (E1) |

* **The date-shifted twins (R1).** P-READING-DS and P-AUDIT-DS replay the
  planned compositions of P-READING and P-AUDIT with `testlike.shifted(p, 1.25,
  None)` applied to every pair, as RS shifted every test-like pair
  (`paiec/testlike.py:291-314`; `experiments/regime_sensitivity.py:235-236`).
  Responses, items and subject_ids are untouched. The acquisition hashes the
  visible input, dates included, so the twins also redraw which items are
  acquired, as RS's shift did. Bundles are shared with the real-date runs
  (same B_run). The twins are no-loss conditions, D ≤ 0 in (E2) and (c), for
  every candidate (the re-check's N1): no candidate carries on gains at real
  dates while it loses under the date shift LEVEL was calibrated with
  (section 4.3 says what each candidate's twins read).
* **Targets** are imported from `experiments/regime_sensitivity.py` (sha256
  `af80ed50bc02a92d`, pinned): `TARGETS` (:250-264), `MIX` (:234),
  `KNOB_SCALE` (:267), `REALISE_TOL` (:289).
* **Weights.** t_b is the target density at level_b over a Gaussian-kernel
  density (bandwidth 0.7) of the half's levels, each benchmark once;
  w_b ∝ min(t_b, 20 · mean t), the mean over the half. The weights are then
  restricted to the benchmarks the regime's sampler can draw (every benchmark
  of the half for formative-size runs; for dense runs those with at least 4
  eligible pairs, section 5.2) and renormalised, and only then capped: no
  benchmark's share exceeds 0.10 (water-filling: shares above 0.10 are set to
  0.10 and the rest rescaled, until none exceeds it). The balance check of
  section 5.1, Kish n and the largest share are read on these weights.
  P-MIXTURE's target is RS's low component (-4.38, 0.171) and a high component
  of sd 1.151, weight and mean from the grid.
* **The realised level** of a set of runs is read, as the targets are, on the
  evaluated responses: per appearance, the continuity-corrected logit of all
  its evaluated responses (`testlike.describe`'s `logit`, at the protocol's
  split, seed 0, scope 'pair'); over appearances, the mean, the sd (ddof 0),
  the share below -3 and the mean of the rest.
* **The knob search** runs on `CAL_SEED`, 1,000 draws a setting, per half, on a
  cheaper proxy: each drawn pair's all-item pair logit from
  `data/pool_eval/levels.json` (section 5.1). Grids: Gaussian m ∈ {-3.5,
  -3.375, ..., +0.5} × s ∈ {0.6, 0.8, ..., 2.2}; mixture low weight ∈ {0.06,
  0.10, ..., 0.30} × high mean ∈ {-1.5, -1.375, ..., 0.0}. Objective RS's:
  Σ((proxy realisation - target) / `KNOB_SCALE`)². Each argmin is fixed at
  the lock. At the argmin, `regimes` draws the same 1,000 CAL_SEED runs again
  and measures their realised level on evaluated responses: that is the
  CAL_SEED value of every regime. The twins have no knobs of their own.
* **Short of target.** A regime whose CAL_SEED value misses its target by more
  than 0.15 on the mean or 0.03 on the share is recorded as short of target and
  keeps its place in the rule, as RS kept its milder realisations (P1a plan,
  A6).
* **Realisation check.** The planned scoring runs' realised level, on the same
  quantity, must lie within `REALISE_TOL` of the CAL_SEED value (mean ±0.30, sd
  ±0.30, share ±0.04, rest mean ±0.25); if it trips, scoring does not start and
  the cause is found. Knobs are never re-tuned on a scoring seed. Proxy and
  evaluated-response realisations are both written (descriptive); only the
  latter enters the two tests (the re-check's N9).

*Planning* (formative-size rows `a4final/split_v2.py`, dense rows
`a4final/dense_cap.py`; 250 draws a setting, a coarser grid, the cap on, after
the dense restriction; realised figures on the all-item proxy, not on evaluated
responses; Kish n = 1/Σw² of the weights the sampler draws with, by benchmark
and by family). The `regimes` stage writes every one of these figures for the
final knobs (section 9):

| regime | S: knobs → realised mean / sd | S: Kish n, benchmarks / families | C: knobs → realised | C: Kish n |
|---|---|---|---|---|
| P-FLAT | none → -0.05 / 1.59 | 44.0 / 40.3 | none → +0.11 / 1.70 | 88.0 / 39.1 |
| P-READING | N(-0.75, 1.6) → -0.67 / 1.82 | 33.3 / 31.4 | N(-0.75, 1.8) → -0.71 / 1.83 | 60.0 / 36.2 |
| P-AUDIT | N(-1.25, 1.6) → -1.09 / 1.99 | 25.8 / 24.8 | N(-1.375, 1.8) → -1.09 / 1.92 | 39.1 / 28.8 |
| P-TUNED | N(-1.75, 1.2) → -1.32 / 1.72 | 18.3 / 17.7 | N(-1.5, 1.4) → -1.24 / 1.68 | 33.1 / 25.7 |
| P-MIXTURE | (0.06 at -4.38; 0.94 at -0.5) → share 0.086, rest -0.24 | 30.9 / 29.2 | the same → share 0.036, rest -0.36 | 68.2 / 42.8 |
| D-FLAT | none → -0.28 / 1.61 | 34.0 / 30.4 | none → +0.14 / 1.77 | 73.0 / 29.4 |
| D-READING | N(-0.75, 1.6) → -0.68 / 1.76 | 25.6 / 24.0 | N(-1.0, 1.6) → -0.76 / 1.77 | 42.6 / 27.5 |

* **The cap binds only in S** (crow in P-AUDIT and P-TUNED, bird_sql in
  P-MIXTURE). Without it bird_sql held 21% of S's P-MIXTURE weight. With the
  cap applied before the dense restriction, as `split_v2.py` drew, S's
  D-READING took N(-0.875, 1.6), Kish n 23.7 / 22.4, and crow 0.111 of the
  weight; with the cap after it, crow holds 0.098 at the argmin.
* **P-MIXTURE is short of target in both halves**, on the share below -3: S by
  0.032, C by 0.082. bird_sql (-4.27), which carried the draft's C, is now in S
  and capped; C's three benchmarks below -3 (llmail_inject -5.42,
  monkey_power_laws -3.25, wmt_mqm -3.10) sit away from the low component. It
  keeps its place as a loss bound.
* **Medhelm's ten benchmarks are one family in C**, so C's Kish n by family is
  well below its Kish n by benchmark; the power table of section 7 uses both.

### 5.4 Seeds and run counts

| seed | use |
|---|---|
| `SPLIT_SALT = "pool-eval-split"` | the split |
| `CAL_SEED = 4001` | knobs; never scored |
| `SEL_SEED = 4002` | S |
| `CONF_SEED = 4003` | C |
| `TIMING_SEED = 4004` | the runs of `timing`, RP2 and RP3, never scored; `timing` and RP3 run on label-randomised copies, from the streams `[TIMING_SEED, 9, i]` and `[TIMING_SEED, 3, i]` (section 9 items 6 and 9) |
| RS seed 11 | the stored compositions |
| `BOOT_SEED = 0` | every bootstrap |

Each run uses `np.random.default_rng([seed, half_code, regime_code, i])`; the
twins reuse their parent regime's stream and composition. **Seed history**
(R11): 4002 to 4004 have not been used. 4001 has: the planning scripts
`eval_design/feasibility.py` and `a4final/split_v2.py` draw their planning
runs with two-element streams `default_rng([4001, k])`. That is harmless:
CAL_SEED is never scored, and the four-element streams of this design differ
from them. The three-element label-randomising streams of `timing` and RP3
differ from every composition stream too.

| stage | P- regimes | D- regimes | RS | arms |
|---|---|---|---|---|
| S | 80 compositions each of 5 regimes (400), plus the 2 twins (160 plays) | 20 each (40) | 520 tasks | pool: A3, C1, C1m, C2; RS: C1, C1m, C2 |
| C, core | 120 each (600), plus the 2 twins (240 plays) | 40 each (80) | | A3 and X* |
| C, extra | the same 840 plays | | | the two other candidates; after the gate is committed; descriptive |

* **Task order.** Run-index-major order on 3 shard processes, so any stopping
  point leaves every regime with the same prefix.
* **Resumability.** A worker exits with code 75 once its resident set passes
  1.45 GB; the stage resumes. The check runs between tasks, after a task's rows
  are written, so a resumed stage starts at the next unwritten task and never
  repeats a finished one; no task is killed for its size. The `timing` stage
  measures the two largest item files of E on a scoring worker before the lock
  (section 9 item 9).
* **The dense counts follow the calendar** (revised 2026-10-11, before the
  lock and before any task was scored, from section 12's budget; no outcome
  exists to decide from). The design of 2026-10-07 planned 30 (S) and 50 (C)
  dense runs a regime, cut to 20 and 40 if the lock slipped past its target,
  2026-10-09 23:59 UTC, four days before the gate's hard stop. On amendment
  4's calendar the lock's hard stop, 2026-10-13 23:59 UTC, leaves two days
  before the gate's, 2026-10-15 23:59 UTC: one day for each scoring stage and
  none of slack, less than the slipped case of 2026-10-07 left (three to four
  days). At the 30 and 50 counts, S and the RS guard overrun one day under the
  loads seen on 2026-10-07 before the adjustment below can trip; at 20 and 40
  each stage fits in a day up to it (section 12). So the dense counts are 20
  (S) and 40 (C), wherever the lock lands.
* **The one allowed adjustment**, at the timing stage, before the lock: if the
  projected core exceeds 30 process-hours, the dense item cap drops from 120 to
  80. The 2026-10-07 rule's second step, the dense counts to 20 (S) and 40
  (C), is already the plan. Formative-size counts never change, and nothing
  is cut after the lock.

## 6 Statistics

* **Pair ALC** is 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31
  (`paiec/official.py:60-61`); run ALC its mean over the run's pairs.
* **D(X, R)** is the mean over R's planned runs of run ALC(X) minus run
  ALC(A3), on identical runs and checkpoints. With each appearance weighted
  1 / (pairs in its run), D = Σ_k v_k / Σ_k n_k over clusters k.
* **SE_f**: the cluster bootstrap over source families (ratio estimator, 2,000
  multinomial resamples, seed 0): `Boot`
  (`experiments/level_calibration.py:741-768`) with the family as cluster.
  Siblings share sources and subjects, so they are one cluster.
* **SE_s, SE_fs**: the same bootstrap with the identity key
  (`paiec/hier.py:1918-1927`) and with the (family, key) cell as clusters.
  SE_2 = sqrt(max(0, SE_f² + SE_s² - SE_fs²)); SE_g = max(SE_f, SE_2);
  U95 = D + 1.96 SE_g.
* **PB**: the mean over families, each once, of the family's mean
  difference v_f / n_f, with each appearance weighted 1 / (pairs in its run),
  the weight of D (the verification's V9a); an unweighted mean of the
  family's appearances is a different number and is not PB. **JK** (the final check's B2): the
  maximum over families f of (sum of v_k for appearances k outside f) /
  (sum of n_k for appearances k outside f), with w_k = 1 / (pairs in k's
  run), v_k = w_k d_k and n_k = w_k, so a run whose pairs all lie in f
  drops out.
* **On RS regimes** D is as P1a defined it.
* **Latency** ((E3), (f); the verification's V9b, V9c). X's mean CPU time per
  evaluation call over a set of tasks is its CPU time summed over all its
  calls there, divided by the number of those calls. (E3)'s maximum is over
  every call of X on S's formative-size runs and on the RS tasks; (f)'s over
  every call of X* on C's formative-size runs. (f)'s p99 is
  `numpy.percentile(x, 99)` (linear interpolation, numpy's default) over x,
  the CPU times of all of X*'s dense calls in the C core, read from the
  rows' per-call lists (section 5.2).

**Descriptive outputs** (none changes anything):

* for every arm pair and regime: D with run SE, SE_f, SE_s, SE_fs, SE_2 and the
  bootstrap's 2.5th and 97.5th percentiles; SE over benchmarks as well; PB, JK,
  the share of families with D_f < 0; D by budget;
* for every arm: mean ALC; mean B0 prediction against the mean evaluated rate;
  ECE; fallback and invalid-output counts; latency (mean, p99, max CPU and
  wall per call);
* cuts of D: the contrasts of section 4.3; dated and undated subjects (and
  released on or after 2025-01-01), in every regime and in the twins; subject
  kind (named with an identity row, named without, unnamed); without the
  flagged benchmarks; with and without the catalogue-MCQ benchmarks, and by the
  floor-active share; appearances whose benchmark has family members in E
  against singletons; HELM-hosted benchmarks against the rest (the
  re-check's recommended cut: a benchmark whose own catalogue `source_url`
  is on crfm.stanford.edu with a path under `helm/`; the family key makes
  the HELM suites separate families, though they share one harness and one
  naming of subjects; *planning*, `a4final/helm_cut.py`: 14 benchmarks of E,
  all in C, in 5 families, so the cut is read in C's regimes only); response
  kind; level tercile;
* the runs: per run, v and ALC - v on the replica's own runs; realised levels,
  Kish n and the largest appearance share;
* C1m's and C2's run-level recomputations (section 4.3);
* a reweighting check: P-FLAT's appearances reweighted to each tilted target by
  benchmark level, against the tilted regime's own D.

## 7 Selection, then confirmation

* **One candidate reaches C.** S and the RS guard pick at most one, X*; C tests
  X* alone against A3, so confirmation has no multiplicity.
* **The tie-break and the carry threshold (R10)** act together: only eligible
  candidates with M ≤ -0.001 can be chosen, and the tie-break works among them.
  X* is the earliest member of Q, in the order C1, C1m, C2, with M(X) within
  0.0005 of Q's minimum (inclusive), so with two earlier members inside the
  margin the first of them is taken (the re-check's N7).
* **The selection record** holds every condition for every candidate, M(X) with
  SE_f, S's SE_f per regime and the power they imply for gains of 0.002, 0.004
  and 0.006, X*, and the time stamp. It is written to
  `results/pool_eval.json: selection` and committed before the first CONF_SEED
  task is scored.
* **Why (E1) is a mean plus P1a's per-regime bounds.** A literal "+0.0005 in
  each RS regime" fails by noise alone: with no true effect and a per-regime
  cluster SE of 0.0015 (RS measured 0.0003 to 0.0042 for hier variants; P1a
  plan, A7), each point estimate lies below +0.0005 with probability about
  0.63, all seven with probability well under one half. The guard leans towards
  A3, since the same catalogue chose and guarded LEVEL (P1a plan, A5).
* *Planning* power of (a) in one regime of C (recommended change 1: Kish n, not
  exp(entropy); `a4final/power_v2.py`). SE = σ / sqrt(n), σ the sd across
  clusters of the cluster-mean difference; n the Kish number by family (36.2
  in P-READING, 28.8 in P-AUDIT), a lower bound, and by benchmark (60.0, 39.1),
  an upper one:

  | σ | SE, P-READING (families) | pass at Δ = 0.002 | 0.004 | 0.006 | false pass at Δ = 0 |
  |---|---|---|---|---|---|
  | 0.005 | 0.0008 | 0.50 | 0.99 | 1.00 | 0.008 |
  | 0.010 | 0.0017 | 0.23 | 0.67 | 0.95 | 0.025 |
  | 0.020 | 0.0033 | 0.09 | 0.23 | 0.44 | 0.025 |

  In P-AUDIT the same σ gives SEs 12% larger. Both regimes must pass, and under
  no difference they pass together with probability about 0.01. A true gain
  under about 0.004 will often be missed; the rule errs towards archive-3,
  which is the fallback anyway.
* **The twins' D ≤ 0** (the re-check's N1) leaves (a)'s power as above and adds
  no run. It is a condition on point estimates: where a candidate's true
  effect in a twin is nil, that twin's condition holds with probability about
  one half, and the two twins' together with probability between about 0.25
  and 0.5, as their correlation runs from none to full; where its true effect
  there is a gain of one SE or more, each holds with probability at least
  0.84. So a candidate whose gain at real dates does not carry over to the
  shifted dates passes the twins in at most about half of draws, and one whose
  gain carries over by a SE or more passes each twin in at least 0.84 of them.
* **The twins over selection and gate together** (the verification's
  recommended item). A carried candidate meets both twins twice: in (E2) on S
  and in (c) on C, on disjoint benchmarks, which are taken as independent
  here. With no twin effect it passes all four twin conditions in only about
  0.06 to 0.25 of draws (0.25² to 0.5², as the two twins of one stage run
  from uncorrelated to fully correlated). With a twin gain of one SE or more
  in each, it passes all four in at least about 0.50 to 0.71 of draws (0.84⁴
  to 0.84²). This is the twins' cost on its own. Each twin replays its
  parent's runs, so its noise is correlated with the gain regimes', and the
  chance of passing the whole rule is not this times (a)'s power. Decision
  7, which the author confirms in commit 1, carries this cost; the rule errs
  towards archive-3.
* **The loss bounds** (+0.002, +0.001) are point estimates too: at a true loss
  of +0.003 and a per-regime SE near 0.0017, D ≤ +0.002 still holds in roughly
  a quarter of draws, as P1a accepted.

## 8 The decision rule

`RULE_TEXT` is quoted verbatim in amendment 4 of `docs/plans/p2_final_plan.md`,
and `experiments/pool_eval.py` holds it byte for byte as a constant. The lock
checks that the constant equals the amendment's block and that every number of
`RULE` (the thresholds) appears in it.

## 9 Checks before any scored task

Each stage after the lock re-runs the lock check and refuses to run if it fails.

**The check log** (the verification's V10). Every stage, before and after the
lock, appends each check it runs to `results/pool_eval.json: checks`, one
entry per check: the stage, the check's name, its result (pass or fail, with
the failing values) and the UTC time. A check that is not in the log has not
passed. `select` reads (E4)'s "every check passed" from the entries of the
pre-lock stages, the lock and the S and RS stages; `gate` reads (g)'s from all
entries up to the gate. The lock refuses unless the latest entry of every
pre-lock check passes, and (E4) and (g) read each pre-lock check by its latest
entry. A post-lock check passes only if all its entries pass, or a recorded fix
(item 11) resolved its failure and the re-run check is logged as passed (the
final check's B1).

1. **`split`** derives every eligible pair's n and k, the pair logits and
   `level_cc` through `paiec/pool.py`, checks them against `pairs.csv` and
   `benchmarks.csv` and writes `data/pool_eval/levels.json` (section 5.1);
   then it reproduces `a4final/split_v2.json`'s family-rule halves from those
   levels, and the families of `tools/pool_families.json`.
2. **`regimes`**: the knob search on the all-item proxy; every planned
   composition, written to `data/pool_eval_rows/<half>/_compositions.json`;
   each regime's CAL_SEED value on evaluated responses, the short-of-target
   flags and the realisation checks (section 5.3); the balance check on the
   weights the samplers draw with. For every regime and half it writes the
   knobs, the proxy and the evaluated-response realisations, Kish n of the
   weights by benchmark and by family, the largest weight share and the
   flagged share (the dense figures of section 5.3 had no saved output
   before `a4final/dense_cap.json`), and c(X, ·) - c(A3, ·) on the planned
   appearances for every candidate (section 4.2).
3. **`fits`**: the manifest check of every file of L; PB's fits and C2's
   fixed-T pass, cached under `data/pool_fits/` (keyed by revision, benchmark,
   K, M_p, s2e and the sha256 of the source of `pool.py`'s fit loaders,
   `load_fit_pairs` and `items_of`; each entry holds t,
   t_var, mu, mu_var, item_var, share, convergence flags, iterations, CPU time
   and no item text); the index of that cache; convergence and degenerate item
   variances (item_var > 100); the B0-centre table of section 4.2 for A3, C1,
   C1m and C2; mu0_m; the C2 literal and the size of its `mu_var` omission;
   the R4 slopes on the final fits (for the record); for each benchmark of U,
   the largest change in any coefficient and in any E subject's attribute score
   when its rows leave the excl-empty ridge (bird_sql set defog's provider
   coefficient to 3.98 in the prototype); the group shares; the excl-empty
   bundles of C1, C1m and C2 written to
   `data/pool_eval/bundles/archive_<X>.json` with their sha256 (R5); the
   descriptive scale report of the nine mixed benchmarks (section 3.2). The
   fits of mind2web and swe_smith (the re-check's N8) each run in a fresh
   worker process, which records its peak resident set (`ru_maxrss`) and the
   CPU time (`time.process_time`) of its item read and of its fit. All of it
   goes to `results/pool_fits.json` (no per-benchmark level in any bundle).
4. **RP1.** A3 re-scored on P1a's seed-11 runs 0 and 1 of each of the seven RS
   regimes (14 tasks) equals the stored `ship` rows per pair and budget, to
   1e-9. RP1 also computes the digest of the 520 RS row files (item 10) and
   writes it beside its result.
5. **RP2.** The archive-3 runtime, loaded from the zip in a fresh interpreter
   as `check_run` loads it, and the in-repo A3 give bit-identical predictions
   on two TIMING_SEED pool runs (predictions compared with each other, never
   with labels).
6. **RP3.** On RP3's two TIMING_SEED runs, run 0 formative-size and run 1
   dense, and on run 0's twin, `evaluate_timed` and
   `official.run_official(..., workers=1)` give identical predictions for A3
   and C1. Both run on label-randomised copies of these runs (the re-check's
   N6; the verification's V11):
   * run i's copy replaces every 0/1 response, taken in the run's pair order
     and, within a pair, in file order, by the successive draws
     `rng.integers(0, 2)` of `rng = numpy.random.default_rng([TIMING_SEED,
     3, i])`, i being the RP3 run's index;
   * the twin is run 0's copy with every subject's dates moved by 1.25 years
     (`testlike.shifted`), so it reuses its parent's draws;
   * items, subjects, dates (but the twin's) and the acquisition order are
     the run's, so acquisition and `labeled` work as on real labels.

   `run_official` returns scores, not predictions, so RP3 captures each
   arm's predictions through a recording factory: a wrapper of the arm's
   factory that logs every call's prediction by checkpoint and call order.
   Predictions are compared with each other, never with labels;
   `run_official`'s returned dict is discarded, never written or printed.
   The code builds the copies from the runs' pairs and the randomising
   stream alone and asserts, before either harness runs, that every response
   of the copies equals its stream draw; a test checks that no RP3 output
   holds a Brier or an ALC.
7. **RP4.** PB's code reproduces the shipped build on the legacy five's pairs as
   `paiec.data.load_pairs()` gives them, with items and no caps: with excl empty
   and LEVEL written over, archive-3's `prior.json` byte for byte; with
   excl = (parent,), `paiec.prior.build(pairs, (parent,))` for each legacy
   parent.
8. **RP5.** On ten random exclusion sets, every candidate bundle passes
   `from_json` and `HierPredictor`, with `hyper.excluded` equal to excl and
   `included` disjoint from it.
9. **`timing`.** Six formative-size and two dense TIMING_SEED runs drawn from
   S's benchmarks, untilted, and two formative-size TIMING_SEED runs drawn
   from C's benchmarks, untilted, whose first pair is fixed: mind2web's one
   pair in the first, a swe_smith pair (uniform among its three, from the
   run's stream) in the second, the other pairs drawn by the Sampler as
   usual (the re-check's N8). Every arm, on a scoring worker as `score` runs
   it, on label-randomised copies (the verification's V12): timing run i's
   copy replaces every 0/1 response, in the run's pair order and then file
   order, by the successive draws `rng.integers(0, 2)` of
   `rng = numpy.random.default_rng([TIMING_SEED, 9, i])`, i the run's index
   in the stage's fixed order (the six formative-size S runs 0 to 5, the two
   dense S runs 6 and 7, the mind2web run 8, the swe_smith run 9). Labels
   are randomised, not removed: with an empty `labeled` hier could take
   another path or fall back, and the projection would not measure the
   scoring path. The code asserts that every response equals its draw and
   that no Brier or ALC is computed. Recorded: CPU and wall per call and
   run, item-read CPU, bundle time, peak resident set; the projection and
   the adjustment of section 5.4. If either C run's worker
   passes 1.45 GB of resident set, or any call fails, the cause is fixed
   before the lock; no outcome exists to decide from.
10. **`lock`** (R5). sha256 of: the amendment's text (its heading to the next
    `## Amendment` heading or the end of the file); `RULE_TEXT`; this file;
    the constants block; `paiec/pool.py`, `experiments/pool_eval.py`,
    `tests/test_pool.py`, `tests/test_pool_eval.py`; the eight shipped modules
    of section 2 and `submission/model.py` (a9561290e62d3f7c); the harness
    modules the RS rows pin, `paiec/official.py` c70684dc15b560bc,
    `paiec/testlike.py` 00e96c4c1b5ee5fd, `experiments/level_calibration.py`
    0d810dddfde74741, `experiments/regime_sensitivity.py` af80ed50bc02a92d,
    `paiec/data.py` e7a9558445f58431, `paiec/evaluator.py` e2e41ae82793aa4a;
    the modules those import (the re-check's N3), `paiec/baselines.py`
    64fe42f5a63d22e1, `experiments/ship_confirm.py` ec6e73ea1e3e8935 and
    `experiments/subject_side.py` 888c11b0c1b13c05 (which `ship_confirm.py`
    imports), and the package file `experiments/__init__.py`
    e3b0c44298fc1c14 (empty and tracked; every `import experiments.*` in
    `regime_sensitivity.py`, `level_calibration.py`, `ship_confirm.py` and
    `subject_side.py` executes it; the verification's V1); the official
    streaming client `third_party/paiec_baseline/tools/streaming_ingestion.py`
    c6f2610f9c8b6cad (gitignored and never committed, so its digest only,
    with the clone's commit 82d330ddcdb16016a3ae9e048db7e588ba2c6a39 beside
    it): `official.official_client()` executes it on the first `make_hook`
    call in every process that builds checkpoints, and RP3's `run_official`
    (default `deepcopy=True`) uses its `AcquisitionHook`, so it is pinned by
    name, as no static import closure sees it (V3);
    `data/regime_sensitivity_rows/_compositions.json` c1520f4be78e37dc; the
    520 RS row files A3 is read from on RS tasks (section 4.6),
    `data/regime_sensitivity_rows/<R>/<i>.json` for R in TUNED, READING,
    AUDIT, MIXTURE, FLAT (i 0 to 79) and R1B, R1P (i 0 to 59), as one digest:
    the sha256 of the UTF-8 bytes of the lines `"<path>\t<sha256>\n"`, path
    relative to `data/regime_sensitivity_rows/` in POSIX form, sorted by path
    (*planning*, 2026-10-11, `a4final/rs_rows_digest.py`: 520 files,
    9a601549b2d99739); RP1 computes it, the lock records it and `select`
    re-verifies it immediately before it reads a row (V2);
    `data/pool/manifest.json`; `tools/pool_benchmarks.json`;
    `tools/pool_families.json`; `dist/archive-3.zip`, in full
    4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090 (the
    verification's recommended item), and `data/pool_eval/a3_prior.json`, in
    full c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770;
    `data/pool_eval/levels.json`; the planning tables it is checked against,
    `data/p2_work/pool_design/benchmarks.csv` d88247c984edc5df and
    `pairs.csv` 2e913b8b24c29fc8; the fits cache index; the three archive
    bundles; the planned compositions. The `lock` stage computes the
    in-repository import closure of `experiments/pool_eval.py` and
    `paiec/pool.py` (static, every `import` and `from` statement, at any
    depth) and refuses if it holds a file this list does not pin. Also
    recorded: mu0_m, the C2 literal and the tuple `paiec.pool.EVAL_LOADER`,
    the fix scope of item 11 (V4).
    Settings: `PYTHONHASHSEED=0`, one BLAS thread, every set or dict that
    orders output iterated sorted; `tools/build_submission.py --pool` runs
    under the same settings and checks them as the lock does (section 10).
    The script, its tests and the pre-lock results are committed before the
    first scored task. Commits 1 to 4 are pushed to the public repository
    when made (the author approved pushes of pre-registration commits on
    2026-10-06), so the lock commit is an outside time stamp, which P1a
    lacked (P1a plan, A2; the verification's V13).
11. **Failures after the lock, and what a fix may touch** (the re-check's N5;
    the verification's V4 to V6 and V8).
    * **Valid outputs are never failures.** hier's fallbacks (an error caught
      and the no-label prediction returned, then 0.5) and every other valid
      output of the shipped runtime (one finite float in [0, 1], section 5.2)
      are scored as they occur and counted per arm and regime (section 6).
      They are never check failures and are never re-scored.
    * **Invalid outputs** (section 5.2: anything but one finite float in
      [0, 1], or of the wrong shape) cannot be scored. Their cause lies in the
      shipped runtime or a bundle, outside the scope below. `evaluate_timed`
      records each one and never raises on it; it is never a check failure,
      never re-scored and never fixed. It gives "archive-3 stays" for the
      candidate whose arm produced it and for no other, unless A3's arm
      produced it, in which case for every candidate (B3). The classification (`classify_output`) sits in the
      constants block, so it is outside the scope of a fix.
    * **Check failures** are a failed lock check, a manifest or digest
      mismatch, an exception raised in the harness or in a bundle's
      construction, a non-finite value in the harness's own arithmetic, a
      failed realisation check, a row that cannot be written or read back,
      and a task a worker cannot finish. An arm's invalid output is none of
      these.
    * **Scope of a fix.** Only harness and I/O code: `experiments/pool_eval.py`
      outside its constants block, and in `paiec/pool.py` only the
      definitions that the tuple `paiec.pool.EVAL_LOADER` names, frozen at the
      lock: the evaluation loader `load_eval_pairs` and the index cache under
      `data/pool_eval/index/` that it alone uses (the draft names
      `INDEX_VERSION`, `_eval_index_path`, `eval_index`, `_eval_rows`,
      `load_eval_pairs`, `build_eval_indexes` and `_eval_index_task`). The
      lock records the tuple; a fix that changes the tuple, or any definition
      it does not name, is out of scope. Everything else stays
      byte-identical: the constants block and `RULE_TEXT`; every locked
      bundle and its digest; the fits cache and its index; the planned
      compositions; `data/pool_eval/levels.json`; every other locked file.
      For `paiec/pool.py` the check is an AST comparison with exactly the
      definitions `EVAL_LOADER` names removed (`paiec.pool.frozen_ast`),
      against the locked file. `items_of` feeds the fits too, so it is not in
      scope; the fits read every item file of L and `timing` reads the two
      largest of E before the lock.
    * **Out of scope.** A failure whose fix would change any other part of
      `paiec/pool.py` (its fit loaders `load_fit_pairs` and `items_of`, and
      its bundle arithmetic `fit_all`, `bundle`, `b0_centre`, `match_mu0`,
      `eb`), `CANDIDATES`, a bundle or a constant gives "archive-3 stays" for
      the candidates it affects (all of them, if the cause is common to every
      arm or lies in A3's comparator).
    * **What is re-scored after a fix** (V8), by rule, so that no one names
      the affected tasks after rows are seen:
      1. the fix commit runs the locked `tests/test_pool.py` and
         `tests/test_pool_eval.py`, unchanged, and both pass;
      2. the failed tasks are re-scored in full;
      3. so is a fixed verification set: for every stage with scored rows
         (S, RS, C), run 0 and the last run scored before the fix of every
         regime of that stage. Each must reproduce its stored rows bit for
         bit in every field but the timing and resource fields (CPU and wall
         times, resident set, load average). If any verification row
         differs, every task of that stage is re-scored;
      4. `select`, `summarise` and `gate` are re-run from all rows after any
         fix. If a re-run `select` names another X* than the committed
         selection record, the outcome is "archive-3 stays".
    * **Record.** Every fix is committed, with its diff and a deviation record
      (cause, files, old and new sha256, the tasks of steps 2 and 3, the test
      result of step 1), before any task is re-scored. The lock record is not
      rewritten: the fix commit appends the files' new digests as successors,
      which the lock check accepts for the files in scope only; `--pool`
      accepts a successor `paiec/pool.py` under the same AST rule and must
      still rebuild the locked bundle byte for byte. A re-scored task must
      record, for every (excl, candidate), the bundle digest recorded before
      the fix wherever one was.
    * **Time.** A fix commit is not a re-lock and may land after 2026-10-13;
      the stops of 2026-10-15 and later still bind, so a failure not fixed and
      re-scored in time gives "not read in time".

## 10 After the gate

**Pass.**

1. **Build** with `python tools/build_submission.py --pool --candidate X*`,
   under `PYTHONHASHSEED=0` and one BLAS thread, which it checks as the lock
   does and refuses to run without (the re-check's recommended item: a
   multithreaded BLAS can change the last bits of the ridge, and the
   byte-for-byte check would then refuse the build). It rebuilds X*'s
   excl-empty bundle from the locked `paiec/pool.py` (digest checked; a
   successor recorded under section 9 item 11 only by that item's AST rule)
   and the locked fits cache (index checked), and refuses unless the result
   equals the locked bundle byte for byte. It runs every existing check
   (bundle, imports, the bit-identity `check_run` on the legacy session, the
   organisers' validator) and four more:
   * `included` equals L in `tools/pool_benchmarks.json`;
   * id_cap, sigma_theta and the level equal X*'s locked values;
   * the bundle's keys equal archive-3's, with no per-benchmark level;
   * the archive differs from `dist/archive-3.zip` only in `prior.json` and in
     `submission/model.py`, and model.py's AST, with docstrings, comments and
     the `LEVEL` value ignored, equals archive-3's (R6).
   model.py's text must be true for X*: the docstring names the pool revision,
   the training rule and caps, and X*'s level (LEVEL kept, matched under the
   date shift, or the pool's EB level); the comment above `LEVEL` gives the
   values' source. Runtime bytes do not change. The archive is copied to
   `dist/archive-4.zip` and its sha256 recorded by 2026-10-16.
2. **S3 on archive-4:** `python tools/stress_archive.py --archive
   dist/archive-4.zip --no-rebuild --out results/stress_archive4.json`, S3's
   gate read pooled as amendment 1 read it, the per-part figures beside it. No
   fix is made; a failure means archive-3 stays.
3. **Amendment 5**, committed before the upload, announces the one regression
   submission of P2 §4 with archive-4's sha256. Steps 1 to 3 are done by
   2026-10-16 23:59 UTC, or archive-3 stays and nothing is submitted.
4. **Upload**, once, no later than 2026-10-17 23:59 UTC (amendment 4 moves P2
   §4's 2026-10-16 on this path); if it is missed, archive-3 stays and nothing
   is submitted.
5. **The final.** archive-4 becomes the final if every pair of that run is
   scored with finite outputs, whatever its score (P2 §4). Its table is
   recorded as run 4 by `experiments/formative_run4.py` (modelled on
   `formative_run3.py`) and used only through aggregate, id-free quantities; it
   selects, tunes and is compared with nothing. The final choice is recorded in
   a dated amendment by 2026-10-20 23:59 UTC; if run 4's table has not arrived
   by then, archive-3 stays the final.

**Fail, or no candidate carried.** archive-3 stays; nothing is built or
submitted; the negative result is recorded with every D, SE and condition; C
extra and `describe-hidden` still run, descriptively.

**Not locked or not read in time.** archive-3 stays; whatever was scored is
reported as incomplete and descriptive.

## 11 Formative feedback (App G.6)

**Reused for this ship decision, a new use** (R8). Each is global and id-free.
In the RS study they changed nothing; here they define the regimes whose gains
and losses decide whether a refitted prior ships, and under C1m the date shift
also sets mu0 (item 3; the re-check's N4).

1. RS `TARGETS`: READING and MIXTURE from the pooled reading of runs 1 and 2,
   AUDIT from the audit's reading of run 1, TUNED the realisation of the regime
   run 1's lower-root reading set; quoted from `regime_sensitivity.py`, not
   recomputed.
2. testlike's run shape (`paiec/testlike.py:507-537`): 5 to 12 pairs, repeat
   0.25, at most 160 items a pair, the 1,000 cap.
3. testlike's date shift of 1.25 years, the grid point closest to run 1's B0 and
   B1, used by the twins and by C1m's matching. Under C1m the shift therefore
   also sets a parameter: mu0_m solves c(C1m, Σ_DS) = c(A3, Σ_DS), and A3's
   LEVEL was itself set by the feedback, so the shift and LEVEL fix mu0 by
   that matching rule (section 4.3). mu0 is a global hyperparameter of the
   level distribution, one of the three numbers `docs/report/draft.md:305`
   allows; nothing keyed on a benchmark or a subject is set.

**After the gate is committed, descriptive only.**

4. `describe-hidden` reads `results/hidden_runs.json` once and takes
   `decomposition.run{1,2,3}.v`, `.alc_minus_v` and `.pairs`. It places each
   run's v in each P- regime's distribution of run v over C runs of 8 or 9
   pairs (z-score and percentile), and runs 2 and 3's ALC - v in A3's replica
   distribution of ALC - v, with X*'s beside it.
5. If archive-4 ships, run 4's table, as P2 §4 allows.

**Not used:** any per-pair value; any anonymous id; any formative table before
the gate is committed; run 3 for anything but item 4; S4's v-bin weights.

**Text for the report** (the author confirms the wording in the commit of the
amendment):

* `docs/report/draft.md:305`: "here three numbers: ...γ." gains, before its
  full stop, "; for the expanded-pool candidate C1m (App G.6), μ0 is the same
  hyperparameter, set by a fixed rule that matches archive-3's level centre
  under the feedback's date shift", and after it: "For the expanded-pool
  evaluation (App G.6) the feedback's global level targets, run shape and
  date shift also defined the regimes that decided whether a pool-refitted
  prior ships. Beyond what each candidate keeps of archive-3's level prior
  (C1 all three numbers, C1m σ_μ and γ, C2 γ), they set no parameter of such
  a prior, except that under C1m the date shift, through that level prior,
  set μ0 by the matching rule."
* App G.2 gains: "The expanded-pool evaluation and, if it shipped, archive-4's
  priors used measurement-db at revision
  `54074453da7b9911be0f48e4c560be547bc84d21` (fetched by
  `tools/fetch_pool.py` on 2026-10-07 and 2026-10-08, with the item tables of
  every public benchmark but cseo_bench and qatch, which it does not use;
  manifest sha256 in `results/pool_fits.json`). Everything else used
  `bc8204d8…`."
* App G.6, "Training data": "...at the revisions of App G.2 (`bc8204d8…` up to
  archive-3; `54074453…` for the pool evaluation of amendment 4, which trains on
  it whether or not archive-4 ships)...".
* App G.6 gains a paragraph listing items 1 to 5 above and: "Nothing else from
  the feedback was read for this evaluation (confirmed by the author, date)."

## 12 Compute budget

The machine is an Apple M1 Pro, 8 cores, 16 GB, shared with other jobs (load
averages 3 to 95 seen on 2026-10-07). Three shard processes, one BLAS thread
each, at most 1.45 GB each.

*Planning* unit costs (re-measured at `timing`; Mode A adds the floor and group
effects, which the items-blind planning did not see): a hier arm 6.0 s on a
formative-size run, 80 s on a dense run (uncertain by 2x), 5.3 s on an RS task;
the bundles of one run 3 s; the fits and the fixed-T pass about 1,800 CPU s
(the prototype's 900 doubled for items).

| stage | process-seconds |
|---|---|
| fits | 1,800 |
| regimes | 1,800 |
| reproduce | 300 |
| timing (with the two C runs of section 9 item 9) | 1,200 |
| S, P-: 560 plays × 4 arms × 6.0, plus 400 × 3 for bundles | 14,640 |
| S, D-: 40 × (4 × 80 + 3) | 12,920 |
| RS guard: 520 × 3 × 5.3 | 8,270 |
| C core, P-: 840 plays × 2 × 6.0, plus 600 × 3 | 11,880 |
| C core, D-: 80 × (2 × 80 + 3) | 13,040 |
| summarise | 1,200 |
| **core total** | **about 67,050 (18.6 process-hours)** |
| C extra, after the gate: 840 × 2 × 6.0 | 10,080 |

Wall time on 3 shards at 1.5 times the unit costs: about 5.7 hours to the
selection record (the pre-lock stages included), 3.6 for the C core, 1.4 for C
extra. The dense runs carry 39% of the core; at twice their planned cost the
projection is about 25.8 process-hours, and the adjustment of section 5.4 trips
at about 2.6 times. (At the counts of 2026-10-07, 60 and 100 dense runs: about
76,800, 21.3 process-hours; 6.6 and 4.1 hours of wall; 47% dense; 31.2
process-hours at twice the dense cost, a trip at about 1.9 times.) Memory at
most 4.4 GB; disk about 0.2 GB for rows and caches, beside the item files, with
11.24 GB free after the fetch (section 3.7).

**Wall time against the calendar** (2026-10-11; the basis of section 5.4's
dense counts). The lock's hard stop, 2026-10-13 23:59 UTC, leaves 48 hours to
the gate's, 2026-10-15 23:59 UTC, for two stages in sequence (S and the RS
guard, `select` and commit 3; then the C core, `summarise`, the gate and
commit 4), one day each. On this machine the loads seen on 2026-10-07
stretched wall time to 2 to 3 times the 1.5x figures (critic, feasibility).
The criterion, read on this budget and on no outcome: each stage's wall time,
at 3 times the 1.5x figure, fits in 24 hours for every dense cost up to the
adjustment's trigger, past which the item cap of 80 takes over; the rest of
the day goes to the stage's attended steps. Post-lock wall hours on 3 shards
(S and RS without the pre-lock stages; C with `summarise`):

| dense runs (S, C) | dense cost | S and RS: 1.5x figure / 3 times it | C core: 1.5x figure / 3 times it |
|---|---|---|---|
| 60, 100 (2026-10-07) | planned | 5.9 / 17.6 | 4.1 / 12.2 |
| 60, 100 | 1.875x (the trigger) | 8.2 / 24.7 | 6.1 / 18.2 |
| 40, 80 (this plan) | planned | 5.0 / 14.9 | 3.6 / 10.9 |
| 40, 80 | 2x | 6.8 / 20.3 | 5.4 / 16.3 |
| 40, 80 | 2.577x (the trigger) | 7.8 / 23.4 | 6.5 / 19.5 |

At the 2026-10-07 counts S and RS overrun the day just below the trigger, where
no adjustment acts; at 20 and 40 dense runs a regime both stages fit up to it.
A failed check that forces a re-score after the lock may still not fit; then
the outcome is "not read in time" and archive-3 stays.

## 13 Schedule, hard stops and commits

| date (2026, UTC) | work |
|---|---|
| Wed 10-07 | the design notes; this file and amendment 4, first version |
| Thu 10-08 to Sun 10-11 | the item fetch ends (manifest 2026-10-08 04:04 UTC; `--items-skip` committed at 2bfcd17); this file and amendment 4 revised for the calendar and the item facts, then for the re-check's N1 to N9 (2026-10-11/12) |
| Mon 10-12 | **commit 1**: amendment 4, this file, `docs/plans/pool_design/`; `paiec/pool.py`, `tests/test_pool.py`; `fits` |
| Tue 10-13 | `experiments/pool_eval.py`, `tests/test_pool_eval.py`; `split`, `regimes`, `reproduce`, `timing`; **commit 2**, the lock (hard stop 23:59) |
| Wed 10-14 | S and RS scoring; `select`; **commit 3**, the selection record; the `--pool` build path and its tests, written while stages run (they do not depend on the outcome) |
| Thu 10-15 | C core; `summarise`; the gate, read once; **commit 4** (hard stop 23:59, the selection's too); then C extra, `describe-hidden`, **commit 5** (descriptive, no hard stop) |
| Fri 10-16 | on a pass: archive-4 built and checked; S3; amendment 5 (hard stop 23:59) |
| Sat 10-17 | on a pass: the one regression upload (hard stop 23:59) |
| by Tue 10-20 | run 4's table read for finiteness; the final recorded |

**Hard stops** (missing any leaves archive-3 the final): the lock committed by
2026-10-13 23:59 UTC; the selection and then the gate committed by 2026-10-15
23:59 UTC; archive-4 built, through S3 and announced by 2026-10-16 23:59 UTC;
uploaded, once, by 2026-10-17 23:59 UTC; the final recorded by 2026-10-20
23:59 UTC.

**Pushes.** Commits 1 to 4 are pushed to the public repository when made, so
their time stamps are outside ones (section 9 item 10; amendment 4's decision
8).

**What `docs/plans/pool_design/` holds** (commit 1): `data.md`, `code.md`,
`eval.md`, `critic.md`, `amendment4_draft.md` (superseded, kept as the record),
`impl_plan.md`, `dates_log.md` (the calendar revision of 2026-10-11/12),
`recheck.md` (the re-check of 2026-10-11), `nfix_log.md` (its N1 to N9
answered), `nfix_verify.md` (the verification of those answers, V1 to V13)
and `vfix_log.md` (V1 to V13 answered); the scripts `inventory.py`,
`inventory2.py`, `section7.py`, `survey.py`, `proto_loader.py`,
`proto_build.py`, `time_fits.py`, `amend4_order.py`, `code/*.py`,
`eval_design/*.py`, `eval_describe/*.py`, `a4final/*.py`; and their JSON and
text outputs that hold no per-pair row. No
CSV, pickle or per-pair JSON, and no item text, goes in. In particular
`eval_describe/pool_describe.json` stays out: it holds 13,499 per-pair logits
(`pair_logit`). The pre-fix copies in `nfix_orig/` and `vfix_orig/` stay out
too.

## 14 What the report will say

* **Always.** One paragraph and one table in the main text, an appendix for the
  rest: the pool's growth and revision, the random assignment and what it
  implies; the unfreezing and its scope (priors and the level literal only, no
  runtime change); the rule fixed and committed before any comparison (the
  commit ids, pushed when made); the family-whole split and exclusion, the
  twins, Mode A; the
  confirmation table (D, SE_g, U95, PB, JK per regime for X*, or for every
  candidate on S if none was carried); the decomposition of section 4.3; the
  pool's level distribution against LEVEL, the B0-centre table, the link
  coverage and the S-only identity slopes; the App G.6 additions; the limits
  below.
* **Pass.** The submitted predictor is hier with priors refitted on the public
  pool (subject prior from 128 benchmarks, item variance and level from 137);
  under C1 the level prior stays at LEVEL, under C1m its centre is matched to
  archive-3's under the date shift, under C2 it is the pool's EB level. On the
  confirmation benchmarks, which played no part in selection, X* beat
  archive-3 by the stated D where the readings sit, did not lose (D ≤ 0) in
  the same runs with every subject's dates moved by 1.25 years, and lost no
  more than the bounds elsewhere. Under C1m the gains at real dates include
  its level centre's shift (+0.045 on E's pairs, *planning*; the final figure
  from `results/pool_fits.json`); only its date-shifted runs, which carry no
  gain requirement, are read at archive-3's level centre. Runs 1 to 3 scored
  earlier configurations; run 4 was a regression check, read for finite
  outputs. The
  test-like headline (-0.042 against LegacyP) stays a replica result about
  archive-3's configuration.
* **Fail.** The predictor stays archive-3; refitting on the expanded pool did
  not pass the gate fixed in advance; the failed conditions with their numbers.
  If C2 looks good in P-FLAT, the text says held-out public benchmarks favour a
  pool level by construction and the rule asked for gains where the readings
  sit.

**What this cannot settle.** The readings' own uncertainty (the tilts bracket
their centre only, and match level, not kind). Whether the summative draws the
same 7 benchmarks (if not, P-FLAT is the relevant regime, and it only bounds
losses). The hidden subjects' names and dates (the twins bracket two date
assumptions, nothing finer). Shared subjects across S and C (gpt 4o is on 52
benchmarks; SE_2 sees it within C, not across the halves). One split, not
resampled. T_full and mu0_m saw every benchmark through a median and a mean.
Protocol unknowns: the split scope (only 'pair' is scored), which response is
revealed (the first), how the platform fills `interactors` (here: the item's
first response row), repeats and multi-condition cells (wmt_mqm, scored as
repeats), the dense shape. Power: a true gain under about 0.004 will often be
missed.

## 15 Disclosures and the critic's changes

**Disclosures.**

1. Choices made on the whole pool. The rules, the caps, the candidates and
   Mode A were chosen from descriptive statistics of the whole pool,
   confirmation benchmarks included (levels, link coverage, the B0-centre
   arithmetic). The identity slopes that motivated C3 used C's standings; C3 is
   dropped by a criterion read on S and the legacy five only (section 4.4). No
   predictor's ALC or Brier was computed on any pool data.
2. The split was seen early and revised once, on levels, flags and families
   only (section 5.1).
3. C1m's mu0 is set from E's subject dicts (both halves) and two priors; no
   label of E enters it beyond the standings PB already uses. Through A3's
   LEVEL and the 1.25-year shift it also rests on the formative feedback
   (section 11, item 3).

**The critic's required changes, and the recommended ones adopted.**

| | change | where |
|---|---|---|
| R1 | C1m (B0 centre matched under the shift) beside C1; the twins as conditions of the rule (no-loss, D ≤ 0, since the re-check's N1); the B0-centre table at real and shifted dates, dated and undated, for every arm, at `fits`; claims qualified | 4.2, 4.3, 5.3 |
| R2 | split stratified by direction flag (blocks of 3), plus family units; balance check locked | 5.1 |
| R3 | exclusion of whole source families; families from the catalogue in `tools/pool_families.json` | 3.5, 3.6 |
| R4 | S-only slopes recomputed; C3 dropped by the stated criterion | 4.4 |
| R5 | every listed digest locked; the loader verifies pool files against the manifest; excl-empty bundles at `fits`; `--pool` reproduces the locked bundle byte for byte | 3.1, 9, 10 |
| R6 | model.py's docstring and comments may change; AST check | 2, 10 |
| R7 | `evaluate_timed`, CPU time per call; process-hours defined | 5.2 |
| R8 | the feedback uses listed as reused for a ship decision; report text for `draft.md:305`, G.2, G.6 | 11 |
| R9 | the plan's passages superseded; the lock's hard stop | amendment 4 |
| R10 | the tie-break acts only among candidates with M ≤ -0.001 | 7, `RULE_TEXT` |
| R11 | the seed history corrected | 5.4 |
| R12 | item sizes listed; Mode A; the MCQ cut | 3.2, 3.7, 6 |
| recommended, all adopted | Kish n in the power table; a 0.10 cap on any benchmark's weight; C2's `mu_var` approximation stated and sized at `fits`; the timing stage draws from S; one disk figure, measured after the fetch; the largest attribute shift one benchmark of U induces, reported at `fits`; the runs dropped if the lock slips, fixed now (on the calendar of 2026-10-11 the reduced dense counts are the plan) | 7, 5.3, 4.3, 9, 3.7, 9, 5.4 and 12 |
| beyond the list | the family as the bootstrap cluster, PB and JK unit; the family-whole split | 5.1, 6 |

**The re-check's required changes** (`docs/plans/pool_design/recheck.md`,
2026-10-11; `nfix_log.md` lists every edit):

| | change | where |
|---|---|---|
| N1 | the twins are no-loss conditions for every candidate, D ≤ 0 in (E2) and (c) (its option (a); amendment 4's decision 7); what each contrast reads, and where the rule reads it, stated for real and shifted dates | 1, 4.2, 4.3, 5.3, 7, 14; `RULE_TEXT` |
| N2 | amendment 4's preamble: the lock hashes everything section 9 item 10 lists | amendment 4 |
| N3 | `paiec/baselines.py`, `experiments/ship_confirm.py` and `experiments/subject_side.py` locked; an import-closure check at `lock`; levels and pair logits derived through `paiec/pool.py`, checked against `benchmarks.csv` and `pairs.csv`, both hashed | 5.1, 5.3, 9 |
| N4 | under C1m the date shift, through A3's LEVEL, sets mu0; the report text says so | 11, 15; amendment 4 §4.6; `RULE_TEXT` Reading 5 |
| N5 | fallbacks are never failures; the scope of a post-lock fix, its record and its time | 9 item 11; `RULE_TEXT` Reading 4 |
| N6 | RP3 on label-randomised copies, scores discarded, an assert and a test | 9 item 6 |
| N7 | (E1)'s RS regimes named; the tie-break's earliest member of Q; S3's gate read pooled | 7; `RULE_TEXT` |
| N8 | `items_of`: column projection and bounded batches; peak resident set and read CPU of mind2web and swe_smith at `fits`; two C runs at `timing` (on label-randomised copies since V12); the resident-set check between tasks | 3.2, 5.4, 9 |
| N9 | the dense cap after the restriction, and the balance check on those weights; the realised level on evaluated responses for the CAL_SEED value and both tests; the dense Kish figures written; model.py's docstring range; `pool_describe.json` kept out of commit 1 | 2, 5.1, 5.3, 9, 13 |

**The verification's changes** (`docs/plans/pool_design/nfix_verify.md`,
2026-10-11; `vfix_log.md` lists every edit):

| | change | where |
|---|---|---|
| V1 | `experiments/__init__.py` pinned (empty, tracked, executed by every `import experiments.*`) | 9 item 10 |
| V2 | the 520 RS row files pinned as one digest over the sorted (path, sha256) lines; RP1 computes it, the lock records it, `select` re-verifies it | 4.6; 9 item 10 |
| V3 | the official streaming client's sha256 and its clone's commit recorded at the lock | 9 item 10 |
| V4 | the fix scope is the definitions `paiec.pool.EVAL_LOADER` names, frozen at the lock; the lock records the tuple | 9 items 10 and 11; `RULE_TEXT` Reading 4 |
| V5 | an invalid output of A3's arm gives "archive-3 stays" for every candidate | 9 item 11; `RULE_TEXT` Reading 4 |
| V6 | invalid outputs defined (not one finite float in [0, 1], or the wrong shape), recorded, never raised; the classification in the constants block | 5.2; 9 item 11; `RULE_TEXT` Reading 4 |
| V7 | the precedence clause: nothing in the rule, candidates, split, regimes, seeds, counts, statistics or fix scope changes once a scored row exists, not even by a later amendment | preamble; amendment 4's preamble; `RULE_TEXT` Reading 4 |
| V8 | the re-score rule: the locked tests pass unchanged; the failed tasks and a fixed verification set re-scored, a whole stage if a verification row differs; `select`, `summarise` and `gate` re-run | 9 item 11; `RULE_TEXT` Reading 4 |
| V9 | PB's appearance weight; gate (f)'s calls are X*'s and its p99 one estimator; (E3)'s calls named; (E4)'s arms listed; `D_S` written "D on S" | 6; `RULE_TEXT` |
| V10 | per-call CPU times of dense runs in the rows; the check log that (E4) and (g) read | 5.2, 6, 9 |
| V11 | RP3's stream, index and draw order; the twin reuses its parent's copy; predictions captured by a recording factory | 9 item 6 |
| V12 (recommended) | `timing` on label-randomised copies, stream `[TIMING_SEED, 9, i]`, not on labels removed | 5.4; 9 item 9 |
| V13 | commits 1 to 4 pushed when made, so the lock commit is an outside time stamp | 9 item 10, 13; amendment 4's decision 8 |
| recommended, all adopted | the `--pool` build under `PYTHONHASHSEED=0` and one BLAS thread; `dist/archive-3.zip`'s full sha256 in the lock; the twins' cost over selection and gate together; the descriptive HELM-hosted cut; the locked tests run by every fix commit | 10, 9 item 10, 7, 6, 9 item 11 |
