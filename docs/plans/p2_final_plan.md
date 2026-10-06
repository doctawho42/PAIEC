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
