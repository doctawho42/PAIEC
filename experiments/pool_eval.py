"""Amendment 4: does a pool-fitted hier replace archive-3? (docs/plans/p2_pool_eval.md)

The pool evaluation of docs/plans/p2_pool_eval.md ("PE" below; section numbers
are its), applied through the decision rule that amendment 4 of
docs/plans/p2_final_plan.md quotes verbatim (RULE_TEXT). Everything from the
line opening "constants: the lock pins this block" to the line closing it is
the constants block: the rule, its thresholds, the candidates, the split, the
regimes and their knob grids, the seeds, the run counts, the statistics' fixed
choices, the pinned digests and the classification of outputs. The `lock`
stage hashes it, and a post-lock fix may change this file only outside it
(PE 9 item 11). Amendment 4a also freezes the decision arithmetic outside it
(the definitions DECISION names: the paired differences, the cluster and
two-way SEs, PB, JK, latency, the selection, the gate and their helpers) by
an AST rule, decision_ast, whose digest the lock records; a fix that changes
them is outside the fix scope. Amendment 4b widens that freeze to the code that
decides which rows reach the arithmetic and whether a stage is complete
(load_rows, missing_tasks, row_path), the per-pair Brier (score_preds) and the
imports the frozen code reads; makes gate (g)'s stage list a block constant
(GATE_STAGES); has apply_selection and apply_gate refuse any rule or bounds
but the block's literals; and checks every frozen object (frozen_names: the
block's and the frozen definitions') against the digest the lock recorded, at
the start and the end of every post-lock stage. Under 4b's wording ("the selection
and the gate", and what decides which rows reach them and whether a stage is
complete) the select, summarise and gate stages are frozen whole, with the helpers
they call that decide a row set, a completeness, a check or the decision (DECISION);
UNFROZEN_READS names where the freeze stops. Amendment 4c moves every hard
stop RULE_TEXT names by two days (HARD_STOPS; RULE_TEXT itself is unchanged),
and the lock records its text's digest beside 4a's and 4b's.

Stages, each a subcommand; after the lock every stage re-runs the lock check
and refuses to run when it fails, and every stage appends the checks it runs
to the check log (results/pool_eval.json: checks):

  split        n, k and the pair logits of every eligible pair of E through
               paiec.pool (manifest-verified responses), checked against the
               planning tables benchmarks.csv and pairs.csv to CSV_TOL, written
               to data/pool_eval/levels.json; the family-unit split from those
               levels, checked against a4final/split_v2.json and families.json
  regimes      the benchmark weights (tilt over the half, the sampler's
               restriction, then the 0.10 cap), the knob search on CAL_SEED's
               all-item proxy, each regime's CAL_SEED value on evaluated
               responses, every planned composition
               (data/pool_eval_rows/<half>/_compositions.json), the realisation
               and short-of-target checks, the balance check, Kish n, the
               B0-centre differences on the planned appearances
  fits-extra   descriptive records the fits stage leaves to this script: the R4
               slopes on the final fits and C1m's matched mu0 without each of
               50 runs' families a half
  reproduce    RP1 to RP5 (PE 9 items 4 to 8); RP2 and RP3 on label-randomised
               copies (the re-check's N6), their scores never computed or kept
  timing       the timing runs on label-randomised copies (V12): CPU per call,
               peak resident set, the core's projection and the one adjustment
  lock         every digest of PE 9 item 10, the import closure, the settings;
               --dry writes it where --state points and changes nothing else
  score        --stage S|RS|C|C-extra --shard k/3: resumable, run-index-major,
               one row file per task; exit 75 above RSS_GUARD_GB between tasks
  status       rows scored so far
  select       (E1) to (E4), M(X), Q and X* -> selection
  summarise    PE 6's statistics on the C core -> summary
  gate         (a) to (g), read once -> gate (and tools/pool_build.json on a pass)
  record-fix   after a post-lock check failure: the fix's scope, the locked tests,
               the tasks to re-score by rule, the deviation record
  describe-hidden  after the gate is committed: results/hidden_runs.json, once
  constants-digest  the constants block's sha256 (for CONSTANTS_SHA256)

Run (PYTHONHASHSEED=0 and one BLAS thread, which `lock` checks):
  python experiments/pool_eval.py split
  python experiments/pool_eval.py regimes --jobs 3
  python experiments/pool_eval.py fits-extra
  python experiments/pool_eval.py reproduce
  python experiments/pool_eval.py timing
  python experiments/pool_eval.py lock
  for k in 1 2 3; do (while :; do python experiments/pool_eval.py score --stage S \\
      --shard $k/3; [ $? -eq 75 ] || break; done) & done; wait

Nothing here writes item text: rows, caches and logs hold ids, counts, digests
and scores. Nothing before describe-hidden opens results/formative* or
results/hidden_runs.json (guarded_open).
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import hashlib  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- constants: the lock pins this block, from this line to its closing line -----------

#: where the lock reads amendment 4 and the design, as committed in commit 1, and the
#: drafts they were written as (only a dry lock reads those)
AMENDMENT_FILE = "docs/plans/p2_final_plan.md"
AMENDMENT_HEADING = "## Amendment 4"
#: amendment 4a (committed in d48f66b, before the lock), which governs where it and the
#: design differ: the lock records its text's sha256 beside amendment 4's
AMENDMENT_4A_HEADING = "## Amendment 4a"
#: amendment 4b (committed in fe2a7f4, before the lock), which governs where it, 4a and the
#: design differ: the lock records its text's sha256 too
AMENDMENT_4B_HEADING = "## Amendment 4b"
#: amendment 4c (committed in 72cc96b, before the lock): every hard stop moved by two days;
#: the lock records its text's sha256 too, and where RULE_TEXT names a date, 4c's governs
AMENDMENT_4C_HEADING = "## Amendment 4c"
DESIGN_FILE = "docs/plans/p2_pool_eval.md"
DRAFT_DIR = "data/p2_work/pool_design"
DRAFT_FILES = {"amendment": "amendment4_short.md", "design": "p2_pool_eval.md"}

RULE_TEXT = """POOL-EVAL DECISION RULE: does a pool-fitted hier replace archive-3?
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
"""

#: the thresholds RULE_TEXT states, for the code that applies it. The twins' bound is
#: its own key (the re-check's N1): the rule before N1, when the twins were loss bounds at
#: +0.002, is this with RULE["twin"] = RULE["loss"]. Amendment 4b: apply_selection and
#: apply_gate refuse any rule but this literal (block_literals)
RULE = {
    "rs_mean": 0.0005,          # (E1) the mean of D over the seven RS regimes
    "rs_testlike": 0.002,       # (E1) each of RS-TUNED .. RS-FLAT
    "rs_public": 0.001,         # (E1) RS-R1B and RS-R1P
    "loss": 0.002,              # (E2), (c) P-TUNED, P-MIXTURE, P-FLAT
    "twin": 0.0,                # (E2), (c) P-READING-DS, P-AUDIT-DS: no loss
    "dense": 0.001,             # (E2), (d) D-FLAT, D-READING
    "latency_ratio": 2.0,       # (E3), (f) mean CPU per call, X over A3
    "latency_max_s": 2.0,       # (E3), (f) no formative-size or RS call above
    "carry": -0.001,            # Q: eligible X with M(X) <= -0.001
    "tie_margin": 0.0005,       # X*: the earliest of Q with M <= min over Q + 0.0005
    "gain": -0.002,             # (a) D <= -0.002
    "z95": 1.96,                # U95 = D + 1.96 SEg
    "pb": 0.0,                  # (b) PB <= 0
    "jk": -0.001,               # (b) JK <= -0.001
    "dense_p99_s": 1.0,         # (f) p99 of X*'s dense calls
    "dense_max_s": 30.0,        # (f) the largest of them
}
#: where each threshold is stated: every phrase, formatted with its value, must appear
#: in RULE_TEXT with runs of white space read as one space (the lock checks it)
RULE_PHRASES = {
    "rs_mean": ("over the seven RS regimes is <= {:+g}",),
    "rs_testlike": ("D(X,RS-R) <= {:+g} in each of RS-TUNED, RS-READING, RS-AUDIT, "
                    "RS-MIXTURE and RS-FLAT",),
    "rs_public": ("and <= {:+g} in RS-R1B and RS-R1P",),
    "loss": ("D <= {:+g} in P-TUNED, P-MIXTURE and P-FLAT",
             "in P-TUNED, P-MIXTURE and P-FLAT: D <= {:+g}"),
    "twin": ("D <= {:g} in P-READING-DS and P-AUDIT-DS",
             "in P-READING-DS and P-AUDIT-DS: D <= {:g};"),
    "dense": ("D <= {:+g} in D-FLAT and D-READING", "in D-READING and D-FLAT: D <= {:+g}"),
    "latency_ratio": ("at most {:.1f} times A3's in the same tasks", "at most {:.1f} times A3's;"),
    "latency_max_s": ("takes more than {:.1f} s of CPU",),
    "carry": ("eligible X with M(X) <= {:g}.",),
    "tie_margin": ("within {:g} of the minimum of M over Q", "M(X) <= that minimum + {:g})"),
    "gain": ("D <= {:g} and U95 < 0",),
    "z95": ("D(X,R) + {:g} SEg(X,R)",),
    "pb": ("PB <= {:g} and JK",),
    "jk": ("JK <= {:g};",),
    "dense_p99_s": ("of at most {:.1f} s and a maximum",),
    "dense_max_s": ("a maximum of at most {:g} s of CPU",),
}
#: further facts RULE_TEXT states that the constants below must agree with
RULE_FACTS = ("the 132 evaluation benchmarks", "selection (44)", "confirmation (88) halves",
              "(520 tasks)", "2,000 resamples, seed 0", "in the order C1, C1m, C2",
              "dates moved by 1.25 years")

#: the candidates (paiec.pool.CANDIDATES, which must equal this) and the order of the
#: tie-break and of preference (PE 4.3, RULE_TEXT)
CANDIDATES = {"C1": "literal", "C1m": "matched", "C2": "eb"}
TIE_ORDER = ("C1", "C1m", "C2")
A3 = "A3"
#: the arms scored per stage (PE 5.4; RULE_TEXT (E4), (g)); "X*" is the selection's
STAGE_ARMS = {"S": ("A3", "C1", "C1m", "C2"), "RS": ("C1", "C1m", "C2"), "C": ("A3", "X*"),
              "C-extra": ("others",)}

#: the split (PE 5.1): family units, flagged units in level blocks of 3 with 1 to S,
#: the others within each kind in blocks of 8 with 3 to S, by the sha256 of
#: repr((salt, unit id)) within a block
HALVES = ("S", "C")
HALF_CODE = {"S": 1, "C": 2}
SPLIT = {"salt": "pool-eval-split", "unit": "family", "flagged_block": 3, "flagged_to_s": 1,
         "block": 8, "to_s": 3, "kinds": ("binary", "mixed")}
SPLIT_EXPECTED = {"benchmarks": {"S": 44, "C": 88}, "pairs": {"S": 988, "C": 11680},
                  "S_flagged": ("crow", "emoji_attack", "safeagentbench")}
#: the planning twins the split stage must reproduce, as committed in commit 1
SPLIT_PLANNING = "docs/plans/pool_design/a4final/split_v2.json"
FAMILIES_PLANNING = "docs/plans/pool_design/a4final/families.json"
#: the planning tables the derived levels are checked against (N3), and the tolerance
CSV_TOL = 1e-12

#: the regimes (PE 5.3). shape: formative-size or dense; target: the RS target the knobs
#: aim at (None: untilted, w_b = 1); twin_of: the date-shifted replay of a regime's runs
REGIMES = {
    "P-READING": {"shape": "formative", "target": "READING", "code": 1},
    "P-AUDIT": {"shape": "formative", "target": "AUDIT", "code": 2},
    "P-READING-DS": {"shape": "formative", "twin_of": "P-READING"},
    "P-AUDIT-DS": {"shape": "formative", "twin_of": "P-AUDIT"},
    "P-TUNED": {"shape": "formative", "target": "TUNED", "code": 3},
    "P-MIXTURE": {"shape": "formative", "target": "MIXTURE", "code": 4},
    "P-FLAT": {"shape": "formative", "target": None, "code": 5},
    "D-READING": {"shape": "dense", "target": "READING", "code": 6},
    "D-FLAT": {"shape": "dense", "target": None, "code": 7},
}
GAIN_REGIMES = ("P-READING", "P-AUDIT")
TWINS = {"P-READING-DS": "P-READING", "P-AUDIT-DS": "P-AUDIT"}
DATE_SHIFT = 1.25
#: (E2) and gate (c), (d): every bounded regime with the RULE key of its bound
LOSS_BOUNDS = (("P-TUNED", "loss"), ("P-MIXTURE", "loss"), ("P-FLAT", "loss"),
               ("P-READING-DS", "twin"), ("P-AUDIT-DS", "twin"),
               ("D-FLAT", "dense"), ("D-READING", "dense"))
#: P1a's seven regimes on its seed 11 (the RS guard, (E1)) with the RULE key of each bound
RS_REGIMES = ("TUNED", "READING", "AUDIT", "MIXTURE", "FLAT", "R1B", "R1P")
RS_BOUNDS = (("TUNED", "rs_testlike"), ("READING", "rs_testlike"), ("AUDIT", "rs_testlike"),
             ("MIXTURE", "rs_testlike"), ("FLAT", "rs_testlike"), ("R1B", "rs_public"),
             ("R1P", "rs_public"))
RS_TASKS = {"TUNED": 80, "READING": 80, "AUDIT": 80, "MIXTURE": 80, "FLAT": 80,
            "R1B": 60, "R1P": 60}
RS_SEED = 11

#: the weights: target density over a Gaussian-kernel density of the half's levels,
#: clipped at MAX_TILT times the mean over the half, restricted to what the sampler
#: draws, renormalised, then capped at CAP_SHARE by water-filling (N9)
BANDWIDTH = 0.7
MAX_TILT = 20.0
CAP_SHARE = 0.10
#: the knob grids (PE 5.3): Gaussian (m, s); the mixture's low weight and high mean,
#: its low component and high sd RS's MIX
GAUSS_M = tuple(round(-3.5 + 0.125 * k, 4) for k in range(33))
GAUSS_S = tuple(round(0.6 + 0.2 * k, 4) for k in range(9))
MIX_WEIGHT = tuple(round(0.06 + 0.04 * k, 4) for k in range(7))
MIX_HIGH_MEAN = tuple(round(-1.5 + 0.125 * k, 4) for k in range(13))
MIX_LOW = (-4.38, 0.171)
MIX_HIGH_SD = 1.151
LEVEL_CUT = -3.0
#: short of target: the CAL_SEED value off its target by more than these (PE 5.3)
SHORT_OF_TARGET = {"mean": 0.15, "share_below": 0.03, "rest_mean": 0.15}
#: the balance check: the flagged share of a tilted regime's weights, S against C
BALANCE_MAX = 0.10

#: run shapes (PE 5.2)
FORMATIVE = {"n_pairs": (5, 12), "repeat": 0.25, "max_items": 160, "cap": 1000,
             "min_items": 80}
DENSE = {"benchmarks": 7, "pairs": 8, "min_pairs": 4, "max_items": 120}
SPLIT_SEED = 0
SPLIT_SCOPE = "pair"

#: seeds (PE 5.4); every run is np.random.default_rng([seed, HALF_CODE, regime code, i])
CAL_SEED = 4001
SEL_SEED = 4002
CONF_SEED = 4003
TIMING_SEED = 4004
BOOT_SEED = 0
BOOTS = 2000
CAL_DRAWS = 1000
STAGE_SEED = {"S": SEL_SEED, "C": CONF_SEED, "C-extra": CONF_SEED}
STAGE_HALF = {"S": "S", "C": "C", "C-extra": "C"}
#: the label-randomising streams [TIMING_SEED, code, i] of RP3 and timing (N6, V11, V12)
LABEL_STREAM = {"rp3": 3, "timing": 9}

#: run counts (PE 5.4): per half, per regime of each shape; twins replay their parents
N_RUNS = {"S": {"formative": 80, "dense": 20}, "C": {"formative": 120, "dense": 40}}

#: the timing stage's runs, in their fixed order (PE 9 item 9): (index, half, regime
#: whose untilted weights and stream code they use, first pair)
TIMING_RUNS = ((0, "S", "P-FLAT", None), (1, "S", "P-FLAT", None), (2, "S", "P-FLAT", None),
               (3, "S", "P-FLAT", None), (4, "S", "P-FLAT", None), (5, "S", "P-FLAT", None),
               (6, "S", "D-FLAT", None), (7, "S", "D-FLAT", None),
               (8, "C", "P-FLAT", ("mind2web", "fixed")),
               (9, "C", "P-FLAT", ("swe_smith", "drawn")))
#: RP3's runs (PE 9 item 6): run 0 formative-size and run 1 dense, from S, untilted, and
#: run 0's twin; RP2 reads the same two runs. (index, half, regime, composition stream
#: index: [TIMING_SEED, HALF_CODE, regime code, stream index], apart from timing's 0..9)
RP3_RUNS = ((0, "S", "P-FLAT", 100), (1, "S", "D-FLAT", 101))
REPRO_TOL = 1e-9
RP1_RUNS = (0, 1)
RP5_SETS = 10
#: the one allowed adjustment (PE 5.4), read at timing: above this projection of the
#: core, the dense item cap drops to the second value
ADJUST = {"core_process_hours": 30.0, "dense_max_items": 80}
#: planning unit costs the projection uses where timing does not measure (PE 12)
PLANNING_COSTS = {"rs_task_arm_s": 5.3, "summarise_s": 1200.0, "fits_s": 1800.0,
                  "regimes_s": 1800.0, "reproduce_s": 300.0}
SHARDS = 3
RSS_GUARD_GB = 1.45
RESUME_EXIT = 75
#: hard stops (A4 4.5; RULE_TEXT Readings 1 and 4) as amendment 4c moved them, all at
#: 23:59 UTC: the lock committed; the selection, then the gate, committed; archive-4 built,
#: through S3 and announced (amendment 5); the one regression upload; the final recorded.
#: Where RULE_TEXT names a date, amendment 4c's governs, and RULE_TEXT stays as committed
HARD_STOPS = {"lock": "2026-10-15T23:59:59Z", "gate": "2026-10-17T23:59:59Z",
              "archive_4": "2026-10-18T23:59:59Z", "regression": "2026-10-19T23:59:59Z",
              "final": "2026-10-22T23:59:59Z"}
#: the date RULE_TEXT names for each stop, and the move (amendment 4c's table): RULE_TEXT
#: must name each as "<date> 23:59 UTC", and each HARD_STOPS date is it plus the move
HARD_STOPS_RULE_TEXT = {"lock": "2026-10-13", "gate": "2026-10-15", "archive_4": "2026-10-16",
                        "regression": "2026-10-17", "final": "2026-10-20"}
HARD_STOP_MOVE_DAYS = 2

#: PE 9 item 10: what the lock pins, with the digest the design states (a prefix or in
#: full) or None where the lock records it first. Gitignored files are pinned by digest.
PINNED = {
    "paiec/__init__.py": "e3b0c44298fc1c14", "paiec/hier.py": "d9a9561202b63b9d",
    "paiec/prior.py": "3a8a8c7dcd0da53e", "paiec/predict.py": "46193d8dd7701ec2",
    "paiec/fitting.py": "ec3af9224f48309a", "paiec/irt.py": "3f7d93e5d012699c",
    "paiec/subjects.py": "79eaa16553a90e6b", "paiec/mcq.py": "f240425d01fb7278",
    "submission/model.py": "a9561290e62d3f7c",
    "paiec/official.py": "c70684dc15b560bc", "paiec/testlike.py": "00e96c4c1b5ee5fd",
    "experiments/level_calibration.py": "0d810dddfde74741",
    "experiments/regime_sensitivity.py": "af80ed50bc02a92d",
    "paiec/data.py": "e7a9558445f58431", "paiec/evaluator.py": "e2e41ae82793aa4a",
    "paiec/baselines.py": "64fe42f5a63d22e1", "experiments/ship_confirm.py": "ec6e73ea1e3e8935",
    "experiments/subject_side.py": "888c11b0c1b13c05",
    "experiments/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "third_party/paiec_baseline/tools/streaming_ingestion.py":
        "c6f2610f9c8b6cad8a2338ddc07a0cb25d40bca761a6628937a2cee2705a6267",
    "data/regime_sensitivity_rows/_compositions.json": "c1520f4be78e37dc",
    "data/pool/manifest.json": "ff5482069780ea7844170f2f4c6596b7cda3b3aa568bbbba3d9e4fede9dae915",
    "dist/archive-3.zip": "4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090",
    "data/pool_eval/a3_prior.json":
        "c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770",
    "data/p2_work/pool_design/benchmarks.csv":
        "d88247c984edc5df4083a90d8659f71f1d51eeb41212b10f0eff72612501a405",
    "data/p2_work/pool_design/pairs.csv":
        "2e913b8b24c29fc86c94ea9894373a376a635bda82f4672f8d3d7ebb54efac6a",
    "paiec/pool.py": None, "experiments/pool_eval.py": None,
    "tests/test_pool.py": None, "tests/test_pool_eval.py": None,
    "tools/pool_benchmarks.json": None, "tools/pool_families.json": None,
    "data/pool_eval/levels.json": None, "data/pool_fits/index.json": None,
    "data/pool_eval/bundles/archive_C1.json": None,
    "data/pool_eval/bundles/archive_C1m.json": None,
    "data/pool_eval/bundles/archive_C2.json": None,
    "data/pool_eval_rows/S/_compositions.json": None,
    "data/pool_eval_rows/C/_compositions.json": None,
    # amendment 4b: the legacy five's measurement-db tables (gitignored), which the RS tasks
    # and RP1 read through paiec.data with PAIEC_DATA unset (settings_ok), every *.parquet
    # of data/<b>/, by the sha256 the lock records
    "data/matharena/benchmarks.parquet": None, "data/matharena/items.parquet": None,
    "data/matharena/response.parquet": None, "data/matharena/subjects.parquet": None,
    "data/multi_swebench/benchmarks.parquet": None, "data/multi_swebench/items.parquet": None,
    "data/multi_swebench/response.parquet": None, "data/multi_swebench/subjects.parquet": None,
    "data/real_webagents/benchmarks.parquet": None, "data/real_webagents/items.parquet": None,
    "data/real_webagents/response.parquet": None, "data/real_webagents/subjects.parquet": None,
    "data/researchcodebench/benchmarks.parquet": None,
    "data/researchcodebench/items.parquet": None,
    "data/researchcodebench/response.parquet": None,
    "data/researchcodebench/subjects.parquet": None,
    "data/swe_rebench/benchmarks.parquet": None, "data/swe_rebench/items.parquet": None,
    "data/swe_rebench/response.parquet": None, "data/swe_rebench/subjects.parquet": None,
}
#: the streaming client's clone (gitignored; its digest above, its commit here, V3)
STREAMING_CLIENT_COMMIT = "82d330ddcdb16016a3ae9e048db7e588ba2c6a39"
#: the 520 RS row files A3 is read from, one digest by RS_ROWS_RULE (V2; planning value)
RS_ROWS_DIGEST_PLANNING = "9a601549b2d99739"
RS_ROWS_RULE = "sha256 of the UTF-8 lines '<path>\\t<sha256>\\n', path relative to " \
               "data/regime_sensitivity_rows/ in POSIX form, sorted by path"
#: the scope of a post-lock fix (PE 9 item 11; N5, V4): this file outside the constants
#: block, and in paiec/pool.py exactly the definitions this tuple names (paiec.pool.
#: EVAL_LOADER must equal it); every other locked file has no successor
FIX_SCOPE = ("experiments/pool_eval.py", "paiec/pool.py")
EVAL_LOADER = ("INDEX_VERSION", "_eval_index_path", "eval_index", "_eval_rows",
               "load_eval_pairs", "build_eval_indexes", "_eval_index_task")
#: amendment 4a: a free fit whose item variance is above this is degenerate. It is left
#: out of every pooled hyperparameter (the item variance, sigma_delta, the group share,
#: C2's level distribution and C1m's matching: paiec.pool.degenerate, whose
#: DEGENERATE_ITEM_VAR must equal this); its pairs still serve the subject standings and
#: the identity table, and its benchmark stays in the evaluation sets. The lock records
#: the list and refuses unless it is amendment 4a's (on this pool multi_moe,
#: finegrain_t2i and revolve)
DEGENERATE_ITEM_VAR = 50.0
DEGENERATE_EXPECTED = ("finegrain_t2i", "multi_moe", "revolve")
#: amendment 4a: the decision arithmetic of this file (the paired differences, the cluster
#: and two-way SEs, PB, JK, latency, the selection, the gate and their helpers), frozen by
#: decision_ast below and recorded at the lock: a post-lock fix that changes any of these
#: top-level definitions is outside the fix scope (Reading 4), so it gives "archive-3
#: stays" for the candidates it affects
DECISION = ("WEIGHTS", "W6", "check_status", "checks_passed", "cluster_boot", "Paired", "D_of",
            "pb_jk", "diff_stats", "p99", "latency", "ratio", "apply_selection", "apply_gate",
            "plan_tasks", "pool_runs", "rs_runs", "stored_ship", "invalid_count",
            "SELECT_STAGES", "selection_table", "reselection", "gate_inputs", "gate_table",
            # amendment 4b: the functions that decide which rows reach the arithmetic and
            # whether a stage is complete, the per-pair Brier, and the imports the frozen code
            # (these definitions and the functions of this block) reads
            "load_rows", "missing_tasks", "row_path", "score_preds",
            "np", "math", "os", "hashlib", "defaultdict", "O",
            # score_preds' budgets, bound by the one import statement that binds WEIGHTS
            "BUDGETS",
            # amendment 4b ("the selection and the gate", and what decides which rows reach
            # them and whether a stage is complete): the select, summarise and gate stages
            # whole (lock_verify2 2.2), and the helpers they call that decide the rows, the
            # completeness, a check or the decision: the post-lock guard and its lock-check
            # entry (postlock), the frozen-object check (frozen_check), the check log's entries
            # (log_check) and their times (utcnow, with the import it reads), and the RS rows
            # digest select compares with the lock's (rs_rows_digest, file_sha, sha_bytes)
            "stage_select", "stage_summarise", "stage_gate", "postlock", "frozen_check",
            "log_check", "utcnow", "datetime", "timezone", "rs_rows_digest", "file_sha",
            "sha_bytes")
#: amendment 4b: the stages whose check log gate condition (g) reads (stage_gate's
#: checks_passed): SELECT_STAGES, then the confirmation's own
GATE_STAGES = ("split", "regimes", "fits-extra", "reproduce", "timing", "lock", "score:S",
               "score:RS", "select", "score:C", "summarise", "gate")
#: where amendment 4b's freeze stops: the names of this file outside the block and outside
#: DECISION that a frozen definition reads. None computes a row set, a completeness, a check
#: result or a condition of the selection or the gate:
#:   io           the JSON I/O and the feedback guard every record passes through (freezing
#:                them would freeze describe-hidden's guard; lock_prep2 7.3)
#:   paths        the directories and the build file the workers share (tests move them)
#:   errors       the stop a failed check raises
#:   lock_check   the lock check itself (the lock's machinery with fix_scope and record-fix,
#:                whose behaviour the locked tests pin); postlock, which runs it, logs it and
#:                refuses on it, is frozen
#:   descriptive  the descriptive parts of the selection and summary records (PE 6)
#: tests/test_pool_eval.py requires every frozen definition to read nothing else
UNFROZEN_READS = {
    "io": ("open", "read_json", "load_state", "save_stage", "_locked_update", "write_json",
           "clean"),
    "paths": ("ROWS", "RS_ROWS", "POOL_BUILD"),
    "errors": ("PoolEvalError",),
    "lock_check": ("lock_check",),
    "descriptive": ("power", "floor_active_set", "cuts", "arm_summary", "run_descriptives",
                    "reweighting_check", "PoolView"),
}
#: the row fields a re-scored verification task may differ in (PE 9 item 11, V8)
TIMING_FIELDS = ("calls_cpu", "cpu_sum", "cpu_max", "wall_sum", "wall_max", "arm_cpu",
                 "arm_wall", "task")
#: the checks (E4) and (g) require in the check log, by stage (V10)
REQUIRED_CHECKS = {
    "split": ("levels against the planning tables", "split reproduces the planning halves",
              "families reproduce the planning table", "split counts"),
    "regimes": ("realisation", "balance", "compositions written", "centres"),
    "fits-extra": ("fits record",),
    "reproduce": ("RP1", "RP2", "RP3", "RP4", "RP5", "RS rows digest"),
    "timing": ("labels randomised", "calls valid", "resident set", "projection"),
    "lock": ("lock",),
}


def classify_output(value):
    """(valid, p, reason) for one call's output (PE 5.2, V6): valid iff it is one
    finite float in [0, 1], numpy.atleast_1d of it having shape (1,), as
    level_calibration.evaluate checks a one-config arm; anything else is an invalid
    output, recorded and never raised (RULE_TEXT Reading 4)."""
    try:
        v = np.atleast_1d(np.asarray(value, float))
    except Exception:
        return False, None, f"not a float ({type(value).__name__})"
    if v.shape != (1,):
        return False, None, f"shape {v.shape}"
    p = float(v[0])
    if not math.isfinite(p):
        return False, None, f"non-finite ({p!r})"
    if p < 0.0 or p > 1.0:
        return False, None, f"outside [0, 1] ({p!r})"
    return True, p, ""


def decision_bound(node):
    """The names a top-level statement binds, as amendment 4a's AST rule reads them: a
    def or class its name, and every name its body declares global; any other statement
    every name it stores (assignment, loop or with target, walrus) or imports, at any
    depth outside the defs and classes it holds, whose own rule applies."""
    import ast
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    out, todo = set(), [node]
    while todo:
        n = todo.pop()
        if isinstance(n, defs):
            out.add(n.name)
            out.update(x for m in ast.walk(n) if isinstance(m, ast.Global) for x in m.names)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[0])
        todo.extend(ast.iter_child_nodes(n))
    return out


def decision_ast(source, names=DECISION):
    """Amendment 4a's AST rule: ast.dump of every top-level statement of `source` that
    binds a name of `names` (decision_bound), in file order, docstrings kept, comments and
    layout ignored. The lock records its sha256 for this file with the locked tuple; a
    fix whose file gives another digest changed the decision arithmetic, a later
    redefinition of a name included."""
    import ast
    names = set(names)
    tree = ast.parse(source)
    body = [n for n in tree.body if decision_bound(n) & names]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def _block_source(path=None):
    """(source, first line, last line) of this file (or `path`) and of its constants block:
    the lines opening and closing it, 1-based, found as whole lines."""
    import io
    with io.open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    first = next(i for i, x in enumerate(lines) if x.startswith("# --- constants: the lock pins"))
    last = next(i for i, x in enumerate(lines) if x == "# --- end of the constants block ---")
    return src, first + 1, last + 1


def block_literals(*names):
    """Amendment 4b: {name: the value the constants block's own assignment gives it},
    re-parsed from this file now: ast.literal_eval of the one top-level statement that binds
    the name, which must be an assignment inside the block. apply_selection and apply_gate
    compare their rule, bounds and regimes with these, so a changed or re-bound object is
    refused whatever the module now holds."""
    import ast
    src, first, last = _block_source()
    body = ast.parse(src).body
    out = {}
    for name in names:
        hits = [n for n in body if name in decision_bound(n)]
        if len(hits) != 1 or not isinstance(hits[0], ast.Assign) or \
                not first <= hits[0].lineno <= last:
            raise RuntimeError(f"{name} is not one assignment of the constants block")
        out[name] = ast.literal_eval(hits[0].value)
    return out


def refuse_unless_block(**objects):
    """Amendment 4b: raise unless every object equals the constants block's literal of its
    name (block_literals)."""
    lit = block_literals(*objects)
    bad = sorted(k for k, v in objects.items() if v != lit[k])
    if bad:
        raise PoolEvalError(f"{', '.join(bad)} differ from the constants block's literals "
                            "(amendment 4b): refused")


def object_digest(x):
    """Amendment 4b: the sha256 of a canonical, type-tagged form of `x` as this process holds
    it now, the same in every process that runs this file (under any module name): values by
    type and content (dict order kept, floats by repr, arrays by dtype, shape and bytes); a
    function of this file by its code (bytecode, constants, names, not its file or line
    numbers), defaults, keyword defaults and closure; a class of this file by its bases and
    attributes; a module, or a class or function of another module, by its name."""
    import json
    import re
    import types
    own = globals()["__name__"]

    def code(c):
        return ["code", c.co_name, c.co_argcount, c.co_posonlyargcount, c.co_kwonlyargcount,
                c.co_flags, c.co_code.hex(), [canon(k) for k in c.co_consts], list(c.co_names),
                list(c.co_varnames), list(c.co_freevars), list(c.co_cellvars)]

    def canon(v):
        if v is None or isinstance(v, (bool, str)):
            return [type(v).__name__, v]
        if isinstance(v, int):
            return ["int", str(v)]
        if isinstance(v, (float, complex)):
            return [type(v).__name__, repr(v)]
        if isinstance(v, bytes):
            return ["bytes", v.hex()]
        if v is Ellipsis:
            return ["ellipsis"]
        if isinstance(v, (tuple, list)):
            return [type(v).__name__, [canon(e) for e in v]]
        if isinstance(v, dict):
            return [type(v).__name__, [[canon(k), canon(e)] for k, e in v.items()]]
        if isinstance(v, (set, frozenset)):
            return [type(v).__name__, sorted(json.dumps(canon(e)) for e in v)]
        if isinstance(v, np.ndarray):
            return ["ndarray", str(v.dtype), list(v.shape),
                    hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()]
        if isinstance(v, np.generic):
            return ["numpy", str(v.dtype), repr(v.item())]
        if isinstance(v, types.CodeType):
            return code(v)
        if isinstance(v, types.FunctionType):
            if v.__module__ != own:
                return ["function", v.__module__, v.__qualname__]
            return ["function", v.__qualname__, code(v.__code__), canon(v.__defaults__),
                    canon(v.__kwdefaults__), [canon(c.cell_contents) for c in v.__closure__ or ()]]
        if isinstance(v, (staticmethod, classmethod)):
            return [type(v).__name__, canon(v.__func__)]
        if isinstance(v, property):
            return ["property", canon(v.fget), canon(v.fset), canon(v.fdel)]
        if isinstance(v, types.ModuleType):
            return ["module", v.__name__]
        if isinstance(v, type):
            if v.__module__ != own:
                return ["type", v.__module__, v.__qualname__]
            return ["class", v.__qualname__, [canon(b) for b in v.__bases__],
                    [[k, canon(a)] for k, a in vars(v).items()
                     if k not in ("__dict__", "__weakref__", "__module__")]]
        if isinstance(v, re.Pattern):
            return ["pattern", v.pattern, v.flags]
        if callable(v) and hasattr(v, "__qualname__"):
            return ["callable", type(v).__name__, getattr(v, "__module__", None), v.__qualname__]
        raise TypeError(f"object_digest: no canonical form for a {type(v).__name__}")
    return hashlib.sha256(json.dumps(canon(x), separators=(",", ":")).encode()).hexdigest()


def frozen_names(path=None):
    """Amendment 4b: the names whose objects every post-lock stage checks against the lock,
    sorted: each name a statement of the constants block binds (decision_bound; a
    comprehension's own variables aside) and each name a statement binding a DECISION name
    binds (the frozen definitions, and the imports they read)."""
    import ast
    src, first, last = _block_source(path)
    out = set()
    for node in ast.parse(src).body:
        bound = decision_bound(node)
        if first <= node.lineno <= last:
            comp = {n.id for c in ast.walk(node) if isinstance(c, ast.comprehension)
                    for n in ast.walk(c.target) if isinstance(n, ast.Name)}
            out |= bound - comp
        elif bound & set(DECISION):
            out |= bound
    return sorted(out)


def frozen_digests(names):
    """{name: object_digest of the object this module binds to it, or '<unbound>'}."""
    g = globals()
    return {n: object_digest(g[n]) if n in g else "<unbound>" for n in names}


def frozen_problems(lk):
    """Amendment 4b: [] or why this process's frozen objects are not the ones the lock
    recorded (lk['frozen_objects'], frozen_digests of frozen_names at the lock): a run-time
    change to a constants-block or frozen object, or a re-binding of its name, fails the
    stage that meets it, at its start or at its end."""
    rec = lk.get("frozen_objects")
    if not rec:
        return ["the lock records no frozen objects (amendment 4b)"]
    out = []
    try:
        if frozen_names() != sorted(rec):
            out.append("the frozen names are not the lock's (amendment 4b)")
    except Exception as exc:
        out.append(f"the frozen names cannot be read ({type(exc).__name__}) (amendment 4b)")
    g = globals()
    for n in sorted(rec):
        try:
            now = object_digest(g[n]) if n in g else "<unbound>"
        except Exception as exc:            # an object changed into one with no canonical form
            now = f"<{type(exc).__name__}>"
        if now != rec[n]:
            out.append(f"{n} is not the object the lock recorded (amendment 4b)")
    return out

# --- end of the constants block ---


CONSTANTS_START = "# --- constants: the lock pins this block"
CONSTANTS_END = "# --- end of the constants block ---\n"
#: sha256 of the constants block as written before the lock (the lock records it too;
#: tests/test_pool_eval.py checks it)
CONSTANTS_SHA256 = "619200e75121e73a7ac5e28f9a80f29f12e51f287485f2f7e3d1c86b9a1bdc3f"


# =========================================================================================
# Harness and I/O. Everything below reads the constants above and sets none of them; a
# post-lock fix may change it (PE 9 item 11), the locked tests passing unchanged, except
# the definitions DECISION names (amendments 4a and 4b), which are frozen at the lock.
# =========================================================================================

import argparse  # noqa: E402
import ast  # noqa: E402
import builtins  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paiec import official as O  # noqa: E402
from paiec import pool as PL  # noqa: E402
from paiec import prior as PR  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.data import SUBJECT_FIELDS, official_subject  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS, Pair, Response  # noqa: E402

W6 = np.asarray(WEIGHTS, float)
#: results/pool_eval.json: every stage's record and the check log (--state overrides)
OUT = os.path.join(ROOT, "results", "pool_eval.json")
#: the fits stage's record (paiec.pool --fit); --fits-record overrides
FITS_RECORD = os.path.join(ROOT, "results", "pool_fits.json")
ROWS = os.path.join(ROOT, "data", "pool_eval_rows")
EVAL_DIR = os.path.join(ROOT, "data", "pool_eval")
LEVELS = os.path.join(EVAL_DIR, "levels.json")
STATS_DIR = os.path.join(EVAL_DIR, "stats")
BUNDLES_DIR = os.path.join(EVAL_DIR, "bundles")
A3_PRIOR = os.path.join(EVAL_DIR, "a3_prior.json")
RS_ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")
POOL_BUILD = os.path.join(ROOT, "tools", "pool_build.json")
HIDDEN = os.path.join(ROOT, "results", "hidden_runs.json")
STATS_VERSION = 1
BLAS_VARS = PL.BLAS_VARS


class PoolEvalError(RuntimeError):
    """A check failed: the stage that meets it stops (and logs it)."""


# --- the feedback guard (PE 11; no stage before describe-hidden reads the feedback) -----

_FEEDBACK = (os.path.join(ROOT, "results", "formative"), HIDDEN)
_OPEN = builtins.open


def guarded_open(path, *a, **kw):
    """open(), refusing results/formative* and results/hidden_runs.json."""
    p = os.fsdecode(os.path.abspath(path)) if isinstance(path, (str, bytes, os.PathLike)) else ""
    if p and (p.startswith(_FEEDBACK[0]) or p == _FEEDBACK[1]):
        raise PoolEvalError(f"{p}: the formative feedback is read only by describe-hidden")
    return _OPEN(path, *a, **kw)


open = guarded_open  # noqa: A001  (every open in this module goes through the guard)


# --- io -------------------------------------------------------------------------------------

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean(x):
    """JSON-safe copy: numpy to Python, tuples to lists, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def file_sha(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_json(path, obj, compact=True):
    """Through a temporary file, so a reader never sees half a file; canonical keys."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        if compact:
            json.dump(clean(obj), f, separators=(",", ":"), sort_keys=True)
        else:
            json.dump(clean(obj), f, indent=1, sort_keys=True)
            f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_state(path=None):
    path = path or OUT
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _locked_update(fn, path=None):
    """Read, change and write the state under a file lock (shards finish side by side)."""
    import fcntl
    path = path or OUT
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        state = load_state(path)
        fn(state)
        state["updated_utc"] = utcnow()
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(clean(state), f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return state


def save_stage(name, value, path=None):
    def put(state):
        state[name] = clean(value)
    return _locked_update(put, path)


def log_check(stage, name, ok, values=None, path=None):
    """Append one check to the check log (V10): stage, name, result with its values, UTC.
    A check not in the log has not passed."""
    entry = {"stage": stage, "name": name, "ok": bool(ok), "values": clean(values), "utc": utcnow()}

    def put(state):
        state.setdefault("checks", []).append(entry)
    _locked_update(put, path)
    print(f"  check [{stage}] {name}: {'pass' if ok else 'FAIL'}", flush=True)
    return bool(ok)


def check_status(state, stages):
    """{(stage, name): passed} over the log's entries of `stages`. The latest entry of a
    check decides; a failure logged after the lock counts as resolved only when a later
    entry passes and a recorded fix names the check (PE 9, V10)."""
    lock_utc = (state.get("lock") or {}).get("utc")
    fixed = {tuple(x) for f in state.get("fixes", []) if f.get("in_scope")
             for x in f.get("resolves", [])}
    groups = defaultdict(list)
    for e in state.get("checks", []):
        if e["stage"] in stages:
            groups[(e["stage"], e["name"])].append(e)
    out = {}
    for key, es in groups.items():
        last = es[-1]
        ok = last["ok"]
        if ok and lock_utc and any(not e["ok"] and e["utc"] > lock_utc for e in es):
            ok = key in fixed
        out[key] = bool(ok)
    return out


def checks_passed(state, stages, required=REQUIRED_CHECKS):
    """(passed, problems): every logged check of `stages` passed, and every required one
    of them is in the log."""
    st = check_status(state, stages)
    problems = [f"{s}: {n}" for (s, n), ok in sorted(st.items()) if not ok]
    for s in stages:
        for n in required.get(s, ()):
            if not any(k == (s, n) or (k[0] == s and k[1].startswith(n + " "))
                       for k in st):
                problems.append(f"{s}: {n} (not in the log)")
    return not problems, problems


def rss_gb():
    """The current resident set (psutil), not the peak: a worker that once read a large
    item file falls back after the read (impl_A, N8)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return maxrss_gb()


def maxrss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


def load_avg():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def env_settings():
    return {v: os.environ.get(v) for v in ("PYTHONHASHSEED",) + tuple(BLAS_VARS) + ("PAIEC_DATA",)}


def settings_ok(env=None):
    """PYTHONHASHSEED=0, one BLAS thread, and (amendment 4b) PAIEC_DATA unset or data/, so
    that paiec.data reads the legacy five's pinned tables."""
    env = env or env_settings()
    data = env.get("PAIEC_DATA")
    return env["PYTHONHASHSEED"] == "0" and all(env[v] == "1" for v in BLAS_VARS) and \
        (not data or os.path.abspath(data) == os.path.join(ROOT, "data"))


_LIB = None


def lib_digests():
    global _LIB
    if _LIB is None:
        _LIB = {f: file_sha(os.path.join(ROOT, f), 16) for f in sorted(PINNED)
                if f.endswith(".py") and os.path.exists(os.path.join(ROOT, f))}
    return _LIB


SCRIPT_SHA = file_sha(os.path.abspath(__file__), 16)


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"head": git("rev-parse", "--short", "HEAD"), "script": SCRIPT_SHA,
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "env": env_settings(), "utc": utcnow()}


def constants_block(path=None):
    """The constants block of this file, as text (the lock hashes it)."""
    with open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    a = src.index(CONSTANTS_START)
    b = src.index(CONSTANTS_END, a) + len(CONSTANTS_END)
    return src[a:b]


def constants_sha256(path=None):
    return sha_bytes(constants_block(path).encode("utf-8"))


def rule_problems(text=RULE_TEXT, rule=RULE, phrases=RULE_PHRASES, facts=RULE_FACTS):
    """Every threshold of RULE, formatted, in RULE_TEXT where RULE_PHRASES says, and
    RULE_FACTS in it: [] or the missing phrases."""
    norm = re.sub(r"\s+", " ", text)
    missing = [f"{k}: {p.format(rule[k])}" for k, ps in phrases.items() for p in ps
               if p.format(rule[k]) not in norm]
    missing += [f"fact: {f}" for f in facts if f not in norm]
    missing += [f"no phrase for {k}" for k in rule if k not in phrases]
    return missing


def constants_problems():
    """The constants against what they mirror: paiec.pool, regime_sensitivity, RULE_TEXT."""
    import experiments.regime_sensitivity as RS
    out = rule_problems()
    if CANDIDATES != PL.CANDIDATES or TIE_ORDER != PL.TIE_ORDER:
        out.append("CANDIDATES or TIE_ORDER differ from paiec.pool's")
    if tuple(PL.EVAL_LOADER) != EVAL_LOADER:
        out.append("paiec.pool.EVAL_LOADER differs from the locked tuple")
    if DATE_SHIFT != PL.DATE_SHIFT:
        out.append("DATE_SHIFT differs from paiec.pool's")
    if RS_TASKS != RS.N_RUNS or RS_SEED != RS.SCORE_SEED:
        out.append("RS_TASKS or RS_SEED differ from regime_sensitivity's")
    if MIX_LOW != RS.MIX[0][1:] or MIX_HIGH_SD != RS.MIX[1][2]:
        out.append("the mixture's fixed components differ from regime_sensitivity.MIX")
    if RSS_GUARD_GB != RS.RSS_GUARD_GB or RESUME_EXIT != RS.RESUME_EXIT:
        out.append("the resident-set guard differs from P1a's")
    if DEGENERATE_ITEM_VAR != PL.DEGENERATE_ITEM_VAR:
        out.append("DEGENERATE_ITEM_VAR differs from paiec.pool's")
    if CSV_TOL != PL.CSV_TOL:
        out.append("CSV_TOL differs from paiec.pool's")
    out += decision_binding_problems()
    # amendment 4b
    if GATE_STAGES != SELECT_STAGES + ("score:C", "summarise", "gate"):
        out.append("GATE_STAGES is not SELECT_STAGES and the confirmation's stages")
    out += block_binding_problems()
    out += legacy_data_problems()
    out += hard_stop_problems()
    return out


def hard_stop_problems(text=RULE_TEXT):
    """Amendment 4c: RULE_TEXT names each stop of HARD_STOPS_RULE_TEXT as '<date> 23:59 UTC',
    and HARD_STOPS holds each moved by HARD_STOP_MOVE_DAYS, at 23:59:59Z."""
    from datetime import date, timedelta
    norm = re.sub(r"\s+", " ", text)
    out = []
    if sorted(HARD_STOPS) != sorted(HARD_STOPS_RULE_TEXT):
        out.append("HARD_STOPS and HARD_STOPS_RULE_TEXT name different stops")
    for k, d in sorted(HARD_STOPS_RULE_TEXT.items()):
        if f"{d} 23:59 UTC" not in norm:
            out.append(f"RULE_TEXT does not name {d} 23:59 UTC ({k})")
        want = (date.fromisoformat(d) + timedelta(days=HARD_STOP_MOVE_DAYS)).isoformat()
        if HARD_STOPS.get(k) != f"{want}T23:59:59Z":
            out.append(f"HARD_STOPS[{k!r}] is {HARD_STOPS.get(k)}, not {want}T23:59:59Z (amendment 4c)")
    return out


#: methods that change a dict, list or set in place (block_binding_problems)
MUTATORS = ("update", "pop", "popitem", "clear", "setdefault", "append", "extend", "insert",
            "remove", "sort", "reverse", "add", "discard", "__setitem__", "__delitem__",
            "__setattr__", "__delattr__")


def block_binding_problems(path=None):
    """Amendment 4b's static guard: no statement outside the constants block binds a name
    the block binds (frozen_names' block part), at its top level or anywhere below it (a
    store, augmented store, delete, or a global declaration), stores or deletes into one of
    its items or attributes, or calls a mutating method (MUTATORS) on it. The run-time check
    (frozen_problems) sees what this cannot, such as a change through globals()."""
    src, first, last = _block_source(path)
    tree = ast.parse(src)
    block = set()
    for node in tree.body:
        if first <= node.lineno <= last:
            comp = {n.id for c in ast.walk(node) if isinstance(c, ast.comprehension)
                    for n in ast.walk(c.target) if isinstance(n, ast.Name)}
            block |= decision_bound(node) - comp

    def root(t):
        while isinstance(t, (ast.Subscript, ast.Attribute, ast.Starred)):
            t = t.value
        return t.id if isinstance(t, ast.Name) else None

    out = set()
    for node in tree.body:
        if first <= node.lineno <= last:
            continue
        for m in ast.walk(node):
            targets = []
            if isinstance(m, (ast.Assign, ast.Delete)):
                targets = m.targets
            elif isinstance(m, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                targets = [m.target]
            elif isinstance(m, (ast.For, ast.AsyncFor, ast.comprehension)):
                targets = [m.target]
            elif isinstance(m, ast.withitem) and m.optional_vars is not None:
                targets = [m.optional_vars]
            for t in targets:
                for x in ast.walk(t):
                    if isinstance(x, (ast.Name, ast.Subscript, ast.Attribute)) and \
                            root(x) in block and not isinstance(x.ctx, ast.Load):
                        out.add(f"line {m.lineno}: {root(x)} is bound or changed outside the "
                                "constants block")
            if isinstance(m, (ast.Global, ast.Nonlocal)):
                out.update(f"line {m.lineno}: global {n} outside the constants block"
                           for n in m.names if n in block)
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and \
                    m.name in block:
                out.add(f"line {m.lineno}: {m.name} is defined again outside the constants block")
            if isinstance(m, (ast.Import, ast.ImportFrom)):
                out.update(f"line {m.lineno}: {a.asname or a.name} is imported over a block name"
                           for a in m.names if (a.asname or a.name).split(".")[0] in block)
            if isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute) and \
                    m.func.attr in MUTATORS and root(m.func.value) in block:
                out.add(f"line {m.lineno}: {root(m.func.value)}.{m.func.attr}(...) outside the "
                        "constants block")
    return sorted(out)


def legacy_data_problems():
    """Amendment 4b: PINNED's legacy data entries are every *.parquet of data/<b>/ for the
    legacy five (paiec.pool.LEGACY, paiec.data.BINARY), as on disk now."""
    from paiec.data import BINARY
    out = []
    if sorted(BINARY) != sorted(PL.LEGACY):
        out.append("paiec.data.BINARY is not paiec.pool.LEGACY")
    want = set()
    for b in PL.LEGACY:
        d = os.path.join(ROOT, "data", b)
        want |= {f"data/{b}/{x}" for x in (os.listdir(d) if os.path.isdir(d) else ())
                 if x.endswith(".parquet")}
    pinned = {p for p in PINNED if p.split("/")[:2] in (["data", b] for b in PL.LEGACY)}
    if not want or pinned != want:
        out.append(f"PINNED's legacy data files {sorted(pinned ^ want)} differ from data/<b>/*.parquet")
    return out


def decision_binding_problems(path=None):
    """Every name of DECISION is bound by exactly one top-level statement of this file."""
    with open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    bound = [decision_bound(node) for node in tree.body]
    out = []
    for n in DECISION:
        k = sum(1 for b in bound if n in b)
        if k != 1:
            out.append(f"DECISION names {n}, bound by {k} top-level statements")
    return out


# --- the pool as the stages read it (levels.json, the lists, families, subjects) ---------

def logit_cc(n, k):
    """The pair logit of PE 5.1: log((k + 0.5) / (n - k + 0.5))."""
    return math.log((k + 0.5) / (n - k + 0.5))


def derive_levels(E, root=None):
    """{b: [[subject_id, n, k], ...]} for every eligible pair of E, from paiec.pool's
    manifest-verified responses (N3): n distinct items with a 0/1 response, k those whose
    first 0/1 response in file order is 1; pairs in paiec.pool's order."""
    return {b: [[str(s), int(n), int(k)] for s, n, k, _ in PL.pair_counts(b, root)]
            for b in sorted(E)}


def levels_bytes(levels):
    return (json.dumps({b: levels[b] for b in sorted(levels)}, separators=(",", ":"),
                       sort_keys=True) + "\n").encode()


def write_levels(levels, path=None):
    path = path or LEVELS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = levels_bytes(levels)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return sha_bytes(data)


def read_levels(path=None):
    with open(path or LEVELS) as f:
        d = json.load(f)
    return {b: [(str(s), int(n), int(k)) for s, n, k in v] for b, v in d.items()}


def level_cc(levels):
    """{b: the mean pair logit} (PE 5.1)."""
    return {b: float(np.mean([logit_cc(n, k) for _, n, k in v])) for b, v in levels.items()}


def check_levels_csv(levels, bench_csv=None, pairs_csv=None, tol=CSV_TOL):
    """levels against the planning tables (N3): the same eligible pairs, n and k exactly,
    logit_cc and level_cc within tol. Raises PoolEvalError; returns the largest
    differences."""
    import csv
    bench_csv = bench_csv or os.path.join(ROOT, "data", "p2_work", "pool_design", "benchmarks.csv")
    pairs_csv = pairs_csv or os.path.join(ROOT, "data", "p2_work", "pool_design", "pairs.csv")
    with open(bench_csv) as f:
        B = {r["benchmark"]: r for r in csv.DictReader(f)}
    P = defaultdict(dict)
    with open(pairs_csv) as f:
        for r in csv.DictReader(f):
            P[r["benchmark"]][r["subject_id"]] = r
    worst = {"logit": 0.0, "level": 0.0}
    problems = []
    lev = level_cc(levels)
    for b in sorted(levels):
        mine = {s: (n, k) for s, n, k in levels[b]}
        if b not in B or set(mine) != set(P[b]):
            problems.append(f"{b}: pair sets differ")
            continue
        for s, (n, k) in mine.items():
            q = P[b][s]
            if int(q["n_items"]) != n or int(q["k_first"]) != k:
                problems.append(f"{b}/{s}: n, k {n}, {k}; csv {q['n_items']}, {q['k_first']}")
            worst["logit"] = max(worst["logit"], abs(float(q["logit_cc"]) - logit_cc(n, k)))
        worst["level"] = max(worst["level"], abs(float(B[b]["level_cc"]) - lev[b]))
    if max(worst.values()) > tol:
        problems.append(f"largest differences {worst} above {tol}")
    if problems:
        raise PoolEvalError("levels disagree with the planning tables: " + "; ".join(problems[:10]))
    return {"benchmarks": len(levels), "pairs": sum(len(v) for v in levels.values()),
            "max_abs_diff": worst, "tol": tol}


def helm_hosted(url):
    """HELM-hosted (PE 6, a4final/helm_cut.py): the benchmark's own catalogue source_url
    on crfm.stanford.edu with a first path segment helm."""
    from urllib.parse import urlsplit
    if not url or not str(url).strip():
        return False
    u = urlsplit(str(url).strip())
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    segs = [s for s in u.path.split("/") if s]
    return host == "crfm.stanford.edu" and bool(segs) and segs[0] == "helm"


class PoolView:
    """E's benchmarks as the stages read them: the committed lists, the families,
    levels.json and (lazily) the subjects' eight visible fields."""

    def __init__(self, E, L, U, family_of, flagged, mixed, mcq, levels, helm=(), root=None,
                 subjects=None):
        self.E, self.L, self.U = sorted(E), sorted(L), sorted(U)
        self.family_of = dict(family_of)
        self.flagged, self.mixed, self.mcq = set(flagged), set(mixed), set(mcq)
        self.helm = set(helm)
        self.levels = {b: list(levels[b]) for b in sorted(levels)}
        self.lev = level_cc(self.levels)
        self.root = root
        self._subjects = dict(subjects or {})

    @classmethod
    def from_files(cls, levels_path=None, lists_dir=None, root=None, helm=True):
        bench, fams = PL.read_lists(lists_dir)
        h = [b for b in bench["E"] if helm and helm_hosted(PL.catalogue(b, root)["source_url"])] \
            if helm else []
        return cls(bench["E"], bench["L"], bench["U"], fams["family_of"], bench["flagged"],
                   bench["mixed"], bench["mcq_catalogue"], read_levels(levels_path), h, root)

    def level(self, b):
        return self.lev[b]

    def kind(self, b):
        return "mixed" if b in self.mixed else "binary"

    def subjects(self, b):
        if b not in self._subjects:
            self._subjects[b] = PL.subjects_of(b, self.root)
        return self._subjects[b]

    def subject(self, b, sid):
        return PL._subject(self.subjects(b), sid)


# --- the split (PE 5.1) ---------------------------------------------------------------------

def split_rank(unit):
    return sha_bytes(repr((SPLIT["salt"], unit)).encode("utf-8"))


def split_halves(pool):
    """({benchmark: 'S' | 'C'}, units) by SPLIT: units are source families' members in E;
    flagged units in level blocks of SPLIT['flagged_block'], the others within each kind
    in blocks of SPLIT['block'], each block ordered by split_rank, the first round(to_s *
    size / block) to S."""
    units = defaultdict(list)
    for b in pool.E:
        units[pool.family_of[b]].append(b)
    ulev = {u: float(np.mean([pool.level(b) for b in m])) for u, m in units.items()}
    uflag = {u: any(b in pool.flagged for b in m) for u, m in units.items()}
    ukind = {u: ("mixed" if all(pool.kind(b) == "mixed" for b in m) else "binary")
             for u, m in units.items()}
    half = {}

    def assign(order, block, to_s):
        for i in range(0, len(order), block):
            blk = sorted(order[i:i + block], key=split_rank)
            k = round(to_s * len(blk) / block)
            for j, u in enumerate(blk):
                for b in units[u]:
                    half[b] = "S" if j < k else "C"

    assign(sorted((u for u in units if uflag[u]), key=lambda u: (ulev[u], u)),
           SPLIT["flagged_block"], SPLIT["flagged_to_s"])
    for kind in SPLIT["kinds"]:
        assign(sorted((u for u in units if not uflag[u] and ukind[u] == kind),
                      key=lambda u: (ulev[u], u)), SPLIT["block"], SPLIT["to_s"])
    return half, {u: {"members": sorted(m), "level": ulev[u], "flagged": uflag[u],
                      "kind": ukind[u]} for u, m in sorted(units.items())}


def half_summary(pool, bs):
    L = np.array([pool.level(b) for b in bs])
    npairs = np.array([len(pool.levels[b]) for b in bs])
    fams = defaultdict(int)
    for b in pool.E:
        fams[pool.family_of[b]] += 1
    return {"benchmarks": len(bs), "pairs": int(npairs.sum()), "level_mean": float(L.mean()),
            "level_sd": float(L.std()), "below_-1.5": int((L < -1.5).sum()),
            "below_-3": int((L < -3).sum()), "ge4_pairs": int((npairs >= DENSE["min_pairs"]).sum()),
            "single_pair": int((npairs == 1).sum()),
            "flagged": sorted(b for b in bs if b in pool.flagged),
            "mixed": int(sum(pool.kind(b) == "mixed" for b in bs)),
            "multi_member_families": sorted({pool.family_of[b] for b in bs
                                             if fams[pool.family_of[b]] > 1})}


# --- per-pair item statistics: ids and counts, no text (the samplers and realisations) -----

def _stats_path(b):
    return os.path.join(STATS_DIR, f"{b}.npz")


def build_pair_stats(b, root=None):
    """For every eligible pair of b (paiec.pool's order): its items in order of first 0/1
    response (codes into a vocabulary of ids), and its items sorted by code with the
    number of 0/1 responses and of ones of each, for lookups. From the verified
    responses; ids and counts only; cached under STATS_DIR by the file's sha256."""
    sha = PL.file_sha256(b, "responses.parquet", root)
    R = PL._responses(b, root)
    key = R.s.astype(np.int64) * (len(R.iids) + 1) + R.i
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    n_cell = np.bincount(inv).astype(np.int32)
    k_cell = np.rint(np.bincount(inv, weights=R.y.astype(float))).astype(np.int32)
    del key, inv
    order = np.argsort(first, kind="stable")
    cs, ci = R.s[first[order]], R.i[first[order]]
    cn, ck = n_cell[order], k_cell[order]
    grp = np.argsort(cs, kind="stable")
    counts = np.bincount(cs, minlength=len(R.sids))
    ends = np.cumsum(counts)
    code = {s: c for c, s in enumerate(R.sids)}
    eligible = PL._pair_rows(R)
    segs = []
    for sid, n, _ in eligible:
        c = code[sid]
        idx = grp[ends[c] - counts[c]:ends[c]]
        if len(idx) != n:
            raise PoolEvalError(f"{b}/{sid}: {len(idx)} items, the pair table says {n}")
        segs.append(idx)
    cat = np.concatenate(segs) if segs else np.zeros(0, np.int64)
    items_raw = ci[cat]
    vocab, items = np.unique(items_raw, return_inverse=True)
    items = items.astype(np.int32)
    ns, ks = cn[cat], ck[cat]
    off = np.concatenate([[0], np.cumsum([len(s) for s in segs])]).astype(np.int64)
    sidx = np.concatenate([np.argsort(items[off[j]:off[j + 1]], kind="stable") + off[j]
                           for j in range(len(segs))]) if segs else np.zeros(0, np.int64)
    z = {"version": np.array(STATS_VERSION), "sha256": np.array(sha),
         "sids": np.array([s for s, _, _ in eligible], dtype=np.str_),
         "offsets": off, "item": items, "scode": items[sidx], "sn": ns[sidx], "sk": ks[sidx],
         "iids": np.array([R.iids[c] for c in vocab.tolist()], dtype=np.str_)}
    os.makedirs(STATS_DIR, exist_ok=True)
    tmp = f"{_stats_path(b)}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, _stats_path(b))
    return z


def _stats_task(task):
    b, root, stats_dir, pool_eval_dir, pool_fits_dir = task
    global STATS_DIR
    STATS_DIR = stats_dir
    PL.EVAL_DIR, PL.FITS_DIR = pool_eval_dir, pool_fits_dir
    load_pair_stats(b, root)
    return b


def load_pair_stats(b, root=None, build=True):
    path = _stats_path(b)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) == STATS_VERSION and \
                    str(z["sha256"]) == PL.file_sha256(b, "responses.parquet", root):
                return PairStats(b, {k: z[k] for k in z.files})
    if not build:
        raise PoolEvalError(f"{b}: no pair statistics at {path}")
    return PairStats(b, build_pair_stats(b, root))


def build_stats(names, jobs=1, root=None):
    """load_pair_stats for every name, building the missing ones in up to `jobs` fresh
    processes (a full read of a large responses file stays in its worker)."""
    todo = [b for b in sorted(names) if not os.path.exists(_stats_path(b))]
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(_stats_task, [(b, PL._root(root), STATS_DIR, PL.EVAL_DIR,
                                                         PL.FITS_DIR) for b in todo]):
                pass
    return {b: load_pair_stats(b, root) for b in sorted(names)}


class LazyStats(dict):
    """{benchmark: PairStats}, each loaded from its cache on first use (never built here)."""

    def __init__(self, root=None):
        super().__init__()
        self.root = root

    def __missing__(self, b):
        v = load_pair_stats(b, self.root, build=False)
        self[b] = v
        return v


class PairStats:
    """One benchmark's per-pair item statistics (build_pair_stats)."""

    def __init__(self, b, z):
        self.b = b
        self.sids = [str(x) for x in z["sids"].tolist()]
        self.at = {s: j for j, s in enumerate(self.sids)}
        self.off = z["offsets"]
        self.item = z["item"]
        self.scode, self.sn, self.sk = z["scode"], z["sn"], z["sk"]
        self.iids = [str(x) for x in z["iids"].tolist()]
        self._code = None

    def codes(self, sid):
        """The pair's item codes in order of first 0/1 response."""
        j = self.at[sid]
        return self.item[self.off[j]:self.off[j + 1]]

    def keys(self, codes):
        return [self.iids[int(c)] for c in codes]

    def counts(self, sid, keys):
        """(responses, ones) of these items of the pair, as arrays."""
        if self._code is None:
            self._code = {s: c for c, s in enumerate(self.iids)}
        j = self.at[sid]
        a, z = int(self.off[j]), int(self.off[j + 1])
        sc = self.scode[a:z]
        c = np.array([self._code[k] for k in keys], np.int64)
        pos = np.searchsorted(sc, c)
        if len(c) and (pos.max() >= len(sc) or not np.all(sc[pos] == c)):
            raise PoolEvalError(f"{self.b}/{sid}: an item is not the pair's")
        return self.sn[a:z][pos], self.sk[a:z][pos]


# --- weights (PE 5.3) -------------------------------------------------------------------------

def npdf(x, m, s):
    return np.exp(-0.5 * ((np.asarray(x, float) - m) / s) ** 2) / s


def tilt(levels, comps):
    """t_b / sum over the benchmarks (each once): the target density at level_b over a
    Gaussian-kernel density of these levels (BANDWIDTH), clipped at MAX_TILT times its
    mean; uniform without a target (comps None)."""
    L = np.asarray(levels, float)
    if comps is None:
        return np.full(len(L), 1.0 / len(L))
    f0 = npdf(L[:, None], L[None, :], BANDWIDTH).mean(1)
    t = sum(w * npdf(L, m, s) for w, m, s in comps) / f0
    t = np.minimum(t, MAX_TILT * t.mean())
    return t / t.sum()


def capped(w, cap=CAP_SHARE):
    """Water-filling: shares above `cap` are set to it and the rest rescaled, until none
    exceeds it (split_v2.capped)."""
    w = np.asarray(w, float) / np.sum(w)
    if cap is None or cap * len(w) < 1:
        return w
    fixed = np.zeros(len(w), bool)
    for _ in range(len(w)):
        over = (w > cap + 1e-15) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = cap
        rest = ~fixed
        w[rest] = w[rest] / w[rest].sum() * (1 - cap * fixed.sum())
    return w


def comps_of(regime, knob):
    """The target density's components for a regime's knob: [(1, m, s)] for a Gaussian
    target, the mixture's two components for P-MIXTURE, None untilted."""
    target = REGIMES[regime].get("target")
    if target is None or knob is None:
        return None
    if target == "MIXTURE":
        wl, mh = knob
        return [(wl, MIX_LOW[0], MIX_LOW[1]), (1.0 - wl, mh, MIX_HIGH_SD)]
    m, s = knob
    return [(1.0, m, s)]


def knob_grid(regime):
    target = REGIMES[regime].get("target")
    if target is None:
        return [None]
    if target == "MIXTURE":
        return [(w, mh) for w in MIX_WEIGHT for mh in MIX_HIGH_MEAN]
    return [(m, s) for m in GAUSS_M for s in GAUSS_S]


def kish(w):
    w = np.asarray(w, float)
    return float(1.0 / np.sum(w ** 2))


class Half:
    """One half's benchmarks (sorted by name) and their eligible pairs, as the samplers
    draw them: each pair's subject_id, the key of its eight visible fields, its all-item
    pair logit (the knob search's proxy, from levels.json)."""

    def __init__(self, pool, name, benchmarks):
        self.pool, self.name = pool, name
        self.bs = sorted(benchmarks)
        self.idx = {b: j for j, b in enumerate(self.bs)}
        self.level = np.array([pool.level(b) for b in self.bs])
        self.sids = [[s for s, _, _ in pool.levels[b]] for b in self.bs]
        self.logit = [np.array([logit_cc(n, k) for _, n, k in pool.levels[b]]) for b in self.bs]
        self.n = np.array([len(x) for x in self.sids], np.int64)
        self.family = [pool.family_of[b] for b in self.bs]
        self.flagged = np.array([b in pool.flagged for b in self.bs])
        intern = {}
        self.dkey = []
        self.by_sid, self.by_dict = defaultdict(list), defaultdict(list)
        for bj, b in enumerate(self.bs):
            ks = []
            for j, sid in enumerate(self.sids[bj]):
                d = tuple(sorted(official_subject(pool.subject(b, sid)).items()))
                k = intern.setdefault(d, len(intern))
                ks.append(k)
                self.by_sid[sid].append((bj, j))
                self.by_dict[k].append((bj, j))
            self.dkey.append(ks)

    def weights(self, comps, shape):
        """(indices of the benchmarks the sampler draws, their weights): the tilt over the
        half, restricted (dense: at least DENSE['min_pairs'] pairs), renormalised, then
        capped (N9)."""
        t = tilt(self.level, comps)
        ok = np.arange(len(self.bs)) if shape == "formative" else \
            np.flatnonzero(self.n >= DENSE["min_pairs"])
        w = t[ok] / t[ok].sum()
        return ok, capped(w)

    def weight_stats(self, ok, w):
        fam = defaultdict(float)
        for j, x in zip(ok.tolist(), w.tolist()):
            fam[self.family[j]] += x
        i = int(np.argmax(w))
        return {"benchmarks": int(len(ok)), "kish_benchmarks": kish(w),
                "kish_families": float(1.0 / sum(v * v for v in fam.values())),
                "max_weight": [self.bs[int(ok[i])], float(w[i])],
                "flagged_weight_share": float(np.sum(w[self.flagged[ok]]))}


def _kth_free(blocked, n, k):
    """The k-th (0-based) pair index in range(n) not in `blocked`."""
    j = k
    for x in sorted(blocked):
        if x <= j:
            j += 1
        else:
            break
    if j >= n:
        raise PoolEvalError("no free pair")
    return j


def draw_formative(H, w, rng, first=None):
    """The pairs of one formative-size run, before any item is cut (PE 5.2): n ~
    U{5..12}; with probability `repeat` (drawn only when an open benchmark has a free
    pair) the next pair joins an open benchmark, otherwise it opens a new one; either way
    the benchmark ∝ w among those with a free pair (in name order), the pair uniformly
    among its free pairs (not drawn, its subject_id and eight fields not yet in the run).
    `first` = (benchmark, 'fixed' | 'drawn') fixes the first pair's benchmark (timing).
    Returns [(benchmark index, pair index)] in draw order."""
    lo, hi = FORMATIVE["n_pairs"]
    n = int(rng.integers(lo, hi + 1))
    free = H.n.copy()
    blocked = defaultdict(set)
    is_open = np.zeros(len(H.bs), bool)
    opened, chosen, sids, dicts = [], [], set(), set()

    def take(bj, j):
        chosen.append((bj, j))
        hits = {(bj, j)}
        sid, d = H.sids[bj][j], H.dkey[bj][j]
        if sid not in sids:
            sids.add(sid)
            hits.update(H.by_sid[sid])
        if d not in dicts:
            dicts.add(d)
            hits.update(H.by_dict[d])
        for b2, j2 in hits:
            if j2 not in blocked[b2]:
                blocked[b2].add(j2)
                free[b2] -= 1
        if not is_open[bj]:
            is_open[bj] = True
            opened.append(bj)

    for step in range(n):
        if step == 0 and first is not None:
            bj = H.idx[first[0]]
            j = 0 if first[1] == "fixed" else _kth_free(blocked[bj], H.n[bj],
                                                         int(rng.integers(free[bj])))
            take(bj, j)
            continue
        open_free = [b for b in sorted(opened) if free[b] > 0]
        if open_free and rng.random() < FORMATIVE["repeat"]:
            cand = np.array(open_free)
        else:
            cand = np.flatnonzero(~is_open & (free > 0))
            if not len(cand):
                break
        p = w[cand] / w[cand].sum()
        bj = int(cand[rng.choice(len(cand), p=p)])
        take(bj, _kth_free(blocked[bj], H.n[bj], int(rng.integers(free[bj]))))
    if len(chosen) < lo:
        raise PoolEvalError(f"a draw ended at {len(chosen)} pairs")
    return chosen


def draw_dense(H, ok, w, rng):
    """The pairs of one dense run (PE 5.2): DENSE['benchmarks'] benchmarks without
    replacement ∝ w among `ok`; from each, in a uniform random order, its pairs whose
    eight-field dict is new within the benchmark, up to DENSE['pairs']."""
    chosen = rng.choice(len(ok), size=min(DENSE["benchmarks"], len(ok)), replace=False, p=w)
    picks = []
    for c in chosen.tolist():
        bj = int(ok[c])
        want = min(DENSE["pairs"], int(H.n[bj]))
        seen, taken = set(), []
        for j in rng.permutation(int(H.n[bj])).tolist():
            d = H.dkey[bj][j]
            if d in seen:
                continue
            seen.add(d)
            taken.append(j)
            if len(taken) == want:
                break
        picks += [(bj, j) for j in taken]
    return picks


def _cut(H, stats, picks, rng, max_items):
    """[(benchmark, subject_id, item ids)]: each pair's items in order of first 0/1
    response, a uniform random subset of max_items kept when it has more
    (testlike.Sampler.run's cut)."""
    out = []
    for bj, j in picks:
        b, sid = H.bs[bj], H.sids[bj][j]
        codes = stats[b].codes(sid)
        if len(codes) > max_items:
            keep = np.zeros(len(codes), bool)
            keep[rng.choice(len(codes), max_items, replace=False)] = True
            codes = codes[keep]
        out.append((b, sid, stats[b].keys(codes)))
    return out


def light_pair(b, sid, keys):
    """A pair with one placeholder response per item: what the run shape, the split and
    the stream read (item keys, subject_id, benchmark), and nothing else."""
    return Pair({}, sid, b, [Response(k, None, 0) for k in keys])


def entries_of(H, stats, shape, picks, rng, dense_max_items=None):
    """A run's composition: [[benchmark, subject_id, items (sorted), kept (indices into
    items, None for all)]], in the run's pair order. Formative-size: the 160-item cut,
    then official.sample_run's proportional cut to the 1,000 cap (it orders the pairs);
    dense: the item cap and no run cap."""
    if shape == "dense":
        cut = _cut(H, stats, picks, rng, dense_max_items or DENSE["max_items"])
        return [[b, sid, sorted(keys), None] for b, sid, keys in cut]
    cut = _cut(H, stats, picks, rng, FORMATIVE["max_items"])
    light = [light_pair(b, sid, keys) for b, sid, keys in cut]
    run = O.sample_run(light, rng, cap=FORMATIVE["cap"], n_pairs=len(light), weighting="pair",
                       min_items=FORMATIVE["min_items"])
    out = []
    for p, kept in run:
        items = sorted(p.item_keys)
        kidx = None if len(kept) == len(items) else [i for i, k in enumerate(items) if k in kept]
        out.append([p.benchmark_id, p.subject_id, items, kidx])
    return out


def run_rng(seed, half, regime, i):
    return np.random.default_rng([seed, HALF_CODE[half], REGIMES[TWINS.get(regime, regime)]["code"], i])


def draw_picks(H, regime, knob, rng, first=None):
    shape = REGIMES[regime]["shape"]
    ok, w = H.weights(comps_of(regime, knob), shape)
    if shape == "dense":
        return draw_dense(H, ok, w, rng)
    full = np.zeros(len(H.bs))
    full[ok] = w
    return draw_formative(H, full, rng, first)


def draw_run(H, stats, regime, knob, seed, i, first=None, dense_max_items=None):
    """Run i of `regime` on `seed` in half H: its composition (entries_of)."""
    rng = run_rng(seed, H.name, regime, i)
    picks = draw_picks(H, regime, knob, rng, first)
    return entries_of(H, stats, REGIMES[regime]["shape"], picks, rng, dense_max_items)


def kept_items(entry):
    _, _, items, kidx = entry
    return list(items) if kidx is None else [items[i] for i in kidx]


def describe_entry(stats, entry):
    """testlike.describe of one composition entry on evaluated responses: the persistent
    split (seed 0, scope 'pair') of the pair's items, its evaluation items among the
    kept ones, every 0/1 response of those, their rate and continuity-corrected logit."""
    b, sid, items, _ = entry
    kept = set(kept_items(entry))
    _, ev = O.split(light_pair(b, sid, items), SPLIT_SEED, SPLIT_SCOPE)
    evk = [k for k in ev if k in kept]
    n, k = stats[b].counts(sid, evk)
    N, K = int(n.sum()), int(k.sum())
    return {"benchmark": b, "subject_id": sid, "items": len(kept), "eval_items": len(evk),
            "responses": N, "p": K / N if N else float("nan"), "logit": T.logit_rate(K, N)}


def realised(logits, cut=LEVEL_CUT):
    """regime_sensitivity.realised: unweighted mean and sd (ddof 0) over appearances, the
    share below `cut`, the rest's mean and sd."""
    import experiments.regime_sensitivity as RS
    return RS.realised(logits, cut)


def objective(r, target):
    """regime_sensitivity's knob objective: the summed squared (realised - target) /
    KNOB_SCALE over the target's keys."""
    import experiments.regime_sensitivity as RS
    tg = RS.TARGETS[target]
    return float(sum(((r[k] - v) / RS.KNOB_SCALE[k]) ** 2 for k, v in tg.items() if k != "cut"))


# --- bundles and arms ---------------------------------------------------------------------

def a3_bundle():
    """archive-3's prior.json (PE 4.6), its digest checked: data/pool_eval/a3_prior.json,
    else an unchanged submission/prior.json (before impl_plan step 1)."""
    want = PINNED["data/pool_eval/a3_prior.json"]
    for p in (A3_PRIOR, os.path.join(ROOT, "submission", "prior.json")):
        if os.path.exists(p) and file_sha(p) == want:
            return PR.load(p)
    raise PoolEvalError("no copy of archive-3's prior.json with sha256 " + want[:16])


class FitsContext:
    """What the candidate bundles are built from: the cached fits (index checked), L and U
    of the committed lists, C1m's locked mu0 from the fits record (or the lock)."""

    def __init__(self, fits_record=None, mu0_m=None):
        bench, _ = PL.read_lists()
        self.L, self.U = bench["L"], bench["U"]
        self.free, self.fixed, self.T_full = PL.load_fits()
        self.index_sha = PL.fits_index_sha256()
        if mu0_m is None:
            rec = read_json(fits_record or FITS_RECORD)
            mu0_m = rec["mu0_m"]["locked"]
        self.mu0_m = float(mu0_m)
        self.arith = PL.source_digest("arithmetic")

    def key(self, X, excl):
        k = json.dumps([X, CANDIDATES[X], sorted(excl), repr(self.mu0_m), self.index_sha,
                        self.arith], separators=(",", ":"))
        return sha_bytes(k.encode())[:24]


class BundleCache:
    """Candidate bundles by (candidate, sorted excl), written once under BUNDLES_DIR/runs/
    as paiec.prior.save writes them and always read back through from_json, so a run's
    arm uses exactly the bytes whose digest its row records."""

    def __init__(self, ctx, directory=None, keep=8):
        self.ctx = ctx
        self.dir = directory or os.path.join(BUNDLES_DIR, "runs")
        self.mem, self.keep = {}, keep

    def get(self, X, excl):
        excl = tuple(sorted(set(excl)))
        key = f"{X}_{self.ctx.key(X, excl)}"
        if key in self.mem:
            return self.mem[key]
        path = os.path.join(self.dir, f"{key}.json")
        c0 = time.process_time()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
        else:
            prior, hyper, _ = PL.candidate_bundle(X, excl, self.ctx.free, self.ctx.fixed,
                                                  self.ctx.L, self.ctx.U, mu0_m=self.ctx.mu0_m)
            data = PL.to_bytes(prior, hyper)
            os.makedirs(self.dir, exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        prior, hyper = PR.from_json(json.loads(data))
        if list(hyper.excluded) != list(excl) or set(hyper.included) & set(excl):
            raise PoolEvalError(f"{key}: the bundle's exclusion lists are not {excl}")
        got = {"prior": prior, "hyper": hyper, "sha256": sha_bytes(data), "key": key,
               "cpu_s": time.process_time() - c0, "excl": list(excl), "candidate": X}
        if len(self.mem) >= self.keep:
            self.mem.pop(next(iter(self.mem)))
        self.mem[key] = got
        return got


def hyper_record(hyper):
    h = hyper.to_dict()
    return {k: h[k] for k in ("mu0", "sigma_mu", "attr_scale", "sigma_d", "sigma_g",
                              "sigma_theta", "sigma_delta", "sigma_attr", "id_cap")
            if k in h} | {"T": h["sigma_d"] ** 2 + h["sigma_g"] ** 2}


class ArmFactory:
    """A no-argument model factory, as level_calibration.evaluate and run_official call
    it: every call (one per checkpoint) builds fresh HierPredictor instances, one per
    bundle, dispatched on the anonymous benchmark_id when the run's benchmarks have
    different bundles (RS tasks: one per parent); `harvest` reads their counters."""

    def __init__(self, bundles, ids=None):
        self.bundles, self.ids = bundles, ids
        self.live, self.failures, self.unconverged = [], 0, 0

    def harvest(self):
        for m in self.live:
            self.failures += int(getattr(m, "failures", 0))
            self.unconverged += int(getattr(m, "unconverged", 0))
        self.live = []

    def __call__(self):
        from paiec.hier import HierPredictor
        self.harvest()
        ms = {k: HierPredictor(p, h) for k, (p, h) in self.bundles.items()}
        self.live = list(ms.values())
        if self.ids is None:
            (m,) = ms.values()
            return m.predict
        ids = self.ids
        return lambda inp, labeled=None: ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled)


# --- the harness (R7): level_calibration.checkpoints and evaluate, timed ------------------

def checkpoints(run, scope=SPLIT_SCOPE, seed=SPLIT_SEED):
    """level_calibration.checkpoints (:508-528), unchanged: the run as paiec.official plays
    it with the platform's random policy, (slots, [(budget, labeled, inputs, index)]),
    each checkpoint's inputs deduplicated as official._evaluate does."""
    slots = O._slots(run, seed, scope)
    hook = O.make_hook(None, False)
    rt = O._Runtime(None, False, None)
    cps = []
    for b in BUDGETS:
        if b:
            O._acquire(slots, b, hook, rt)
        labeled = [e for s in slots for e in s.acquired[:b]]
        inputs, where, index = [], {}, []
        for s in slots:
            for inp, _, key in s.targets:
                j = where.setdefault(key, len(inputs))
                if j == len(inputs):
                    inputs.append(inp)
                index.append(j)
        cps.append((b, labeled, inputs, np.array(index)))
    return slots, cps


def predict_timed(cps, factory, arm, run_id=None, keep_calls=False, raw=False):
    """One arm on every checkpoint, a fresh instance per checkpoint (level_calibration.
    evaluate :531-548), each call timed by time.process_time and time.perf_counter. An
    output that classify_output calls invalid, or a raise, is recorded (arm, run, budget,
    call index) and never raised (PE 5.2, V6). No label is read here."""
    out = {"preds": {}, "calls": 0, "cpu_sum": 0.0, "cpu_max": 0.0, "wall_sum": 0.0,
           "wall_max": 0.0, "invalid": []}
    calls_cpu = [] if keep_calls else None
    c_arm, w_arm = time.process_time(), time.perf_counter()
    call = 0
    for b, labeled, inputs, index in cps:
        fn = factory()
        P = np.full(len(inputs), np.nan)
        for j, inp in enumerate(inputs):
            c0, w0 = time.process_time(), time.perf_counter()
            try:
                v, raised = fn(inp, labeled), None
            except Exception as exc:                  # an arm's raise is an invalid output
                v, raised = None, f"raised {type(exc).__name__}"
            dc, dw = time.process_time() - c0, time.perf_counter() - w0
            out["cpu_sum"] += dc
            out["wall_sum"] += dw
            out["cpu_max"] = max(out["cpu_max"], dc)
            out["wall_max"] = max(out["wall_max"], dw)
            if calls_cpu is not None:
                calls_cpu.append(dc)
            ok, p, why = (False, None, raised) if raised else classify_output(v)
            if ok:
                P[j] = p
            else:
                out["invalid"].append({"arm": arm, "run": run_id, "budget": int(b), "call": call,
                                       "why": why})
            call += 1
        out["preds"][b] = P[index]
        if raw:
            out.setdefault("raw", []).append(P.tolist())
    out["calls"] = call
    if hasattr(factory, "harvest"):
        factory.harvest()
    out["failures"] = int(getattr(factory, "failures", 0))
    out["unconverged"] = int(getattr(factory, "unconverged", 0))
    out["arm_cpu"], out["arm_wall"] = time.process_time() - c_arm, time.perf_counter() - w_arm
    if calls_cpu is not None:
        out["calls_cpu"] = calls_cpu
    return out


def score_preds(slots, preds):
    """level_calibration.evaluate's arithmetic (:549-561) for one arm: per pair Brier and
    ECE at the six budgets and the mean B0 prediction, over every evaluated response."""
    out, at = [], 0
    for s in slots:
        y = np.array([t[1] for t in s.targets], float)
        sl = slice(at, at + len(y))
        at += len(y)
        br = [float(np.mean((preds[b][sl] - y) ** 2)) for b in BUDGETS]
        ec = [O.ece(list(preds[b][sl]), y) for b in BUDGETS]
        out.append({"brier": br, "ece": ec, "q0": float(np.mean(preds[0][sl]))})
    return out


def evaluate_timed(slots, cps, arms, run_id=None, keep_calls=False):
    """evaluate_timed (PE 5.2, R7): every arm on the same checkpoints, timed; an arm with
    an invalid output is recorded and not scored on this run, the others are. Returns
    {arm: predict_timed's record without preds, plus 'rows' (score_preds) or None}."""
    res = {}
    for name, fac in arms.items():
        t = predict_timed(cps, fac, name, run_id, keep_calls)
        t["rows"] = None if t["invalid"] else score_preds(slots, t["preds"])
        del t["preds"]
        res[name] = t
    return res


# --- label-randomised copies (N6, V11, V12) and the recording factory ---------------------

def label_stream(kind, i):
    return [TIMING_SEED, LABEL_STREAM[kind], i]


def label_randomised(run, seed_seq):
    """The run with every 0/1 response replaced, in the run's pair order and then file
    order, by the successive draws rng.integers(0, 2) of default_rng(seed_seq); items,
    subjects, dates and kept sets unchanged. Built from the run's pairs and the stream
    alone."""
    rng = np.random.default_rng(seed_seq)
    out = []
    for p, items in run:
        rs = [Response(r.item_key, r.item, int(rng.integers(0, 2))) for r in p.responses]
        out.append((Pair(p.subject, p.subject_id, p.benchmark_id, rs), items))
    return out


def assert_randomised(copy, seed_seq):
    """Every response of the copy equals its stream draw (the assert of PE 9 items 6, 9)."""
    rng = np.random.default_rng(seed_seq)
    n = 0
    for p, _ in copy:
        for r in p.responses:
            if r.label != int(rng.integers(0, 2)):
                raise AssertionError(f"{p.benchmark_id}/{p.subject_id}: a response is not its draw")
            n += 1
    return n


def twin_of(run):
    """Every pair with its subject's dates moved by DATE_SHIFT years (testlike.shifted, as
    RS shifted every test-like pair); responses, items and subject_ids untouched."""
    return [(T.shifted(p, DATE_SHIFT, None), items) for p, items in run]


class Recording:
    """A model factory that logs every call's prediction, by checkpoint and call order
    (RP3: run_official returns scores, not predictions)."""

    def __init__(self, factory):
        self.factory, self.calls = factory, []

    def __call__(self):
        fn, rec = self.factory(), []
        self.calls.append(rec)

        def g(inp, labeled=None):
            v = fn(inp, labeled)
            rec.append(float(v))
            return v
        return g


_SCORE_WORDS = re.compile(r"brier|\balc\b|\bece\b", re.I)


def assert_no_scores(obj):
    """N6's assert: a record of RP2, RP3 or timing holds no score (no key or value names a
    Brier, an ALC or an ECE), so none can be printed or stored."""
    text = json.dumps(clean(obj))
    if _SCORE_WORDS.search(text):
        raise AssertionError("a score is in a record that must hold none")
    return obj


def recorded_official(run, factory):
    """official.run_official(run, ..., workers=1) with a Recording factory: the
    predictions by checkpoint. Its returned dict is discarded, never kept or printed."""
    rec = Recording(factory)
    _ = O.run_official(run, rec, workers=1)
    del _
    return [list(c) for c in rec.calls]


# --- runs from compositions ----------------------------------------------------------------

def load_run(entries, root=None):
    """The pool run of a composition (PE 5.2): every 0/1 response of each drawn pair's
    items, in file order, uncapped, with Mode A items, through paiec.pool.load_eval_pairs;
    [(Pair, frozenset of kept items)] in the composition's order."""
    by_b = {}
    for b, sid, items, _ in entries:
        by_b.setdefault(b, {})[sid] = items
    got = {}
    for b, keep in by_b.items():
        sids = list(keep)
        for p in PL.load_eval_pairs(b, sids, keep=keep, root=root):
            if set(p.item_keys) != set(keep[p.subject_id]):
                raise PoolEvalError(f"{b}/{p.subject_id}: the loaded items are not the composition's")
            got[(b, p.subject_id)] = p
    return [(got[(e[0], e[1])], frozenset(kept_items(e))) for e in entries]


def identity_key(subject):
    """hier's identity key (paiec/hier.py:1918-1927, subject_key 'canon'): ('n', canon
    name) or ('k', the subject digest), as one string."""
    from paiec.predict import subject_key
    name = PR.canon_name(subject)
    return ("n:" + name) if name else ("k:" + hashlib.sha256(subject_key(subject)).hexdigest()[:16])


def appearance_meta(pool, run, slots, priors):
    """Per pair appearance (PE 5.2's rows, no item text): benchmark, family, kind, flags,
    subject, identity key, named and identity-row flags per arm, the dates shown, kept and
    evaluated items, responses, the evaluated rate and logit, v = rate (1 - rate)."""
    out = []
    for (p, items), s in zip(run, slots):
        subject = official_subject(p.subject)
        y = np.array([t[1] for t in s.targets], float)
        k, n = int(y.sum()), len(y)
        rate = k / n if n else float("nan")
        rel = subject.get("release_date") or ""
        out.append({
            "benchmark": p.benchmark_id, "family": pool.family_of.get(p.benchmark_id),
            "kind": pool.kind(p.benchmark_id), "flagged": p.benchmark_id in pool.flagged,
            "mcq": p.benchmark_id in pool.mcq, "helm": p.benchmark_id in pool.helm,
            "subject_id": p.subject_id, "ident": identity_key(subject),
            "named": bool(PR.canon_name(subject)),
            "id_row": {a: bool(pr is not None and pr.identity(subject) is not None)
                       for a, pr in sorted(priors.items())},
            "dated": bool(rel), "release": rel, "year": int(rel[:4]) if rel[:4].isdigit() else None,
            "kept": len(items), "eval": int(s.n_items), "responses": n, "rate": rate,
            "logit": T.logit_rate(k, n), "v": rate * (1 - rate)})
    return out


# --- statistics (PE 6) -------------------------------------------------------------------

def cluster_boot(keys, w, d, boots=BOOTS, seed=BOOT_SEED):
    """level_calibration.Boot's cluster bootstrap with any cluster key (a ratio estimator:
    an appearance brings w d and w, w = 1 / pairs in its run): clusters sorted, `boots`
    multinomial resamples of all of them, seed `seed`. (SE, the bootstrap distribution)."""
    uniq = sorted(set(keys))
    at = {k: j for j, k in enumerate(uniq)}
    idx = np.array([at[k] for k in keys], np.int64)
    K = len(uniq)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(K, np.full(K, 1 / K), size=boots).astype(float)
    w, d = np.asarray(w, float), np.asarray(d, float)
    v = np.bincount(idx, w * d, K)
    n = np.bincount(idx, w, K)
    dist = (W @ v) / (W @ n)
    return float(dist.std(ddof=1)), dist


class Paired:
    """X minus a reference over one regime's runs, flattened to pair appearances: the
    pair-ALC difference d and its six budget differences, the weight w = 1 / (pairs in
    the appearance's run), the run index and the cluster keys (family, identity key,
    benchmark)."""

    def __init__(self, runs):
        """runs: [(meta, Bx, Bref)], meta a list of appearance dicts with 'family',
        'ident' and 'benchmark', Bx and Bref (pairs, 6) Brier arrays."""
        d, db, w, run, fam, ident, bench = [], [], [], [], [], [], []
        self.metas = []
        for r, (meta, bx, bref) in enumerate(runs):
            bx, bref = np.asarray(bx, float), np.asarray(bref, float)
            n = len(meta)
            if bx.shape != (n, 6) or bref.shape != (n, 6):
                raise PoolEvalError(f"run {r}: {bx.shape} and {bref.shape} for {n} pairs")
            db.append(bx - bref)
            d.append((bx - bref) @ W6)
            w.append(np.full(n, 1.0 / n))
            run.append(np.full(n, r))
            fam += [m["family"] for m in meta]
            ident += [m["ident"] for m in meta]
            bench += [m["benchmark"] for m in meta]
            self.metas += meta
        self.d = np.concatenate(d) if d else np.zeros(0)
        self.db = np.concatenate(db) if db else np.zeros((0, 6))
        self.w = np.concatenate(w) if w else np.zeros(0)
        self.run = np.concatenate(run).astype(int) if run else np.zeros(0, int)
        self.fam, self.ident, self.bench = fam, ident, bench
        self.runs = len(runs)

    def subset(self, mask):
        """The appearances where mask holds, weights kept (a descriptive cut)."""
        out = Paired([])
        m = np.asarray(mask, bool)
        out.d, out.db, out.w, out.run = self.d[m], self.db[m], self.w[m], self.run[m]
        out.fam = [x for x, k in zip(self.fam, m) if k]
        out.ident = [x for x, k in zip(self.ident, m) if k]
        out.bench = [x for x, k in zip(self.bench, m) if k]
        out.metas = [x for x, k in zip(self.metas, m) if k]
        out.runs = len(set(out.run.tolist()))
        return out


def D_of(P):
    """D = sum w d / sum w: the mean over runs of the run's pair-mean difference when every
    run is whole (PE 6)."""
    return float(np.sum(P.w * P.d) / np.sum(P.w))


def pb_jk(P):
    """(PB, JK, {family: D_f}): PB the mean over families, each once, of v_f / n_f with
    the appearance weight w (V9a); JK the largest D with one family's appearances
    removed (None with one family)."""
    v, n = defaultdict(float), defaultdict(float)
    for f, w, d in zip(P.fam, P.w.tolist(), P.d.tolist()):
        v[f] += w * d
        n[f] += w
    per = {f: v[f] / n[f] for f in sorted(v)}
    V, N = sum(v.values()), sum(n.values())
    jk = max(((V - v[f]) / (N - n[f]) for f in v if N - n[f] > 0), default=None)
    return float(np.mean(list(per.values()))), jk, per


def diff_stats(P, rule=RULE, boots=BOOTS, seed=BOOT_SEED):
    """PE 6's statistics of one paired table: D; run SE; SEf, SEs, SEfs (the bootstrap with
    the family, the identity key and the (family, key) cell as clusters), SE2, SEg, U95;
    SE over benchmarks; PB, JK, the share of families with D_f < 0; the bootstrap's 2.5th
    and 97.5th percentiles; D by budget."""
    if not len(P.d):
        return {"D": None, "appearances": 0}
    D = D_of(P)
    rv = np.bincount(P.run, P.w * P.d) / np.maximum(np.bincount(P.run, P.w), 1e-300)
    rv = rv[np.bincount(P.run, P.w) > 0]
    sef, dist = cluster_boot(P.fam, P.w, P.d, boots, seed)
    ses, _ = cluster_boot(P.ident, P.w, P.d, boots, seed)
    sefs, _ = cluster_boot(list(zip(P.fam, P.ident)), P.w, P.d, boots, seed)
    seb, _ = cluster_boot(P.bench, P.w, P.d, boots, seed)
    se2 = math.sqrt(max(0.0, sef ** 2 + ses ** 2 - sefs ** 2))
    seg = max(sef, se2)
    pb, jk, per = pb_jk(P)
    return {"D": D, "appearances": int(len(P.d)), "runs": int(len(rv)),
            "run_se": float(rv.std(ddof=1) / math.sqrt(len(rv))) if len(rv) > 1 else None,
            "SEf": sef, "SEs": ses, "SEfs": sefs, "SE2": se2, "SEg": seg,
            "U95": D + rule["z95"] * seg, "SE_benchmarks": seb,
            "pct": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
            "PB": pb, "JK": jk, "families": len(per),
            "share_families_negative": float(np.mean([x < 0 for x in per.values()])),
            "budgets": (np.sum(P.w[:, None] * P.db, 0) / np.sum(P.w)).tolist()}


def p99(x):
    """Gate (f)'s p99: numpy.percentile(x, 99), linear interpolation (numpy's default)."""
    return float(np.percentile(np.asarray(x, float), 99))


def latency(rows, arm):
    """Pooled over rows: X's CPU time summed over its calls, the calls, the largest call."""
    cpu = calls = 0
    mx = 0.0
    for r in rows:
        a = r["arms"].get(arm)
        if a is None:
            continue
        cpu += a["cpu_sum"]
        calls += a["calls"]
        mx = max(mx, a["cpu_max"])
    return {"cpu_s": cpu, "calls": calls, "mean_s": cpu / calls if calls else None, "max_s": mx}


def ratio(lx, la):
    if not lx["calls"] or not la["calls"] or not la["mean_s"]:
        return None
    return lx["mean_s"] / la["mean_s"]


# --- the rule (RULE_TEXT) -------------------------------------------------------------------

def apply_selection(table, rule=RULE, bounds=LOSS_BOUNDS, rs_bounds=RS_BOUNDS, order=TIE_ORDER):
    """RULE_TEXT's selection on `table`: per candidate X
      D     {regime: D(X, R) on S}
      RS    {RS regime: D(X, RS-R)}
      latency  {"ratio": mean CPU per call over A3's on the S pool tasks,
                "max_s": X's largest call on S's formative-size runs and the RS tasks}
      complete, checks_ok   (E4)
      invalid               X's invalid outputs on S and RS tasks
    and table['A3'] = {'invalid': A3's invalid outputs on S}. Returns the record: every
    condition for every candidate, M, Q, X* and the outcome. Amendment 4b: a rule, bounds,
    order or regime list other than the constants block's literals is refused (raises)."""
    refuse_unless_block(RULE=rule, LOSS_BOUNDS=bounds, RS_BOUNDS=rs_bounds, TIE_ORDER=order,
                        RS_REGIMES=RS_REGIMES, GAIN_REGIMES=GAIN_REGIMES, A3=A3)
    a3_bad = int((table.get(A3) or {}).get("invalid", 0)) > 0
    out, M, eligible = {}, {}, {}
    for X in order:
        t = table[X]
        rs = [t["RS"].get(R) for R in RS_REGIMES]
        e1_mean = None if any(x is None for x in rs) else float(np.mean(rs))
        e1 = {"mean": e1_mean, "mean_ok": e1_mean is not None and e1_mean <= rule["rs_mean"],
              "regimes": {R: {"D": t["RS"].get(R), "bound": rule[k],
                              "ok": t["RS"].get(R) is not None and t["RS"][R] <= rule[k]}
                          for R, k in rs_bounds}}
        e1["ok"] = e1["mean_ok"] and all(v["ok"] for v in e1["regimes"].values())
        e2 = {R: {"D": t["D"].get(R), "bound": rule[k],
                  "ok": t["D"].get(R) is not None and t["D"][R] <= rule[k]} for R, k in bounds}
        lat = t["latency"]
        e3 = {"ratio": lat.get("ratio"), "max_s": lat.get("max_s"),
              "ok": lat.get("ratio") is not None and lat["ratio"] <= rule["latency_ratio"]
              and lat.get("max_s") is not None and lat["max_s"] <= rule["latency_max_s"]}
        inv = int(t.get("invalid", 0))
        e4 = {"complete": bool(t["complete"]), "checks_ok": bool(t["checks_ok"]),
              "ok": bool(t["complete"] and t["checks_ok"])}
        ok = {"E1": e1["ok"], "E2": all(v["ok"] for v in e2.values()), "E3": e3["ok"], "E4": e4["ok"]}
        g = [t["D"].get(R) for R in GAIN_REGIMES]
        M[X] = None if any(x is None for x in g) else float(np.mean(g))
        # Reading 4 (narrow, B3): an invalid output gives "archive-3 stays" for the candidate
        # whose arm produced it and no other, unless A3's arm produced it (then for every one)
        stays = inv > 0 or a3_bad
        eligible[X] = all(ok.values()) and not stays
        out[X] = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "holds": ok, "eligible": eligible[X],
                  "M": M[X], "invalid_outputs": inv, "a3_invalid_outputs": a3_bad,
                  "archive_3_stays_for_it": stays}
    Q = [X for X in order if eligible[X] and M[X] is not None and M[X] <= rule["carry"]]
    if not Q:
        star, outcome, mmin = None, "no candidate carried: archive-3 stays", None
    else:
        mmin = min(M[X] for X in Q)
        star = next(X for X in order if X in Q and M[X] <= mmin + rule["tie_margin"])
        outcome = f"{star} carried to C"
    return {"candidates": out, "Q": Q, "M_min_Q": mmin, "X_star": star, "outcome": outcome,
            "rule": dict(rule), "bounds": [list(b) for b in bounds]}


def apply_gate(g, rule=RULE, bounds=LOSS_BOUNDS):
    """RULE_TEXT's gate on X* against A3 on C: g has 'stats' {regime: diff_stats}, 'E1'
    (the selection's (E1) for X*), 'latency' {'ratio', 'max_s' (formative-size calls),
    'dense_p99_s', 'dense_max_s'}, 'checks_ok', 'complete', 'invalid_X', 'invalid_A3'.
    Returns every condition and the outcome. Amendment 4b: a rule, bounds or regime table
    other than the constants block's literals is refused (raises)."""
    refuse_unless_block(RULE=rule, LOSS_BOUNDS=bounds, REGIMES=REGIMES, GAIN_REGIMES=GAIN_REGIMES)
    s = g["stats"]

    def val(R, k):
        return (s.get(R) or {}).get(k)

    a = {R: {"D": val(R, "D"), "U95": val(R, "U95"),
             "ok": val(R, "D") is not None and val(R, "D") <= rule["gain"] and val(R, "U95") < 0}
         for R in GAIN_REGIMES}
    b = {R: {"PB": val(R, "PB"), "JK": val(R, "JK"),
             "ok": val(R, "PB") is not None and val(R, "PB") <= rule["pb"]
             and val(R, "JK") is not None and val(R, "JK") <= rule["jk"]} for R in GAIN_REGIMES}
    c = {R: {"D": val(R, "D"), "bound": rule[k], "ok": val(R, "D") is not None and val(R, "D") <= rule[k]}
         for R, k in bounds if REGIMES[R]["shape"] == "formative"}
    d = {R: {"D": val(R, "D"), "bound": rule[k], "ok": val(R, "D") is not None and val(R, "D") <= rule[k]}
         for R, k in bounds if REGIMES[R]["shape"] == "dense"}
    e = {"ok": bool(g["E1"])}
    lat = g["latency"]
    f = {**lat, "ok": lat.get("ratio") is not None and lat["ratio"] <= rule["latency_ratio"]
         and lat.get("max_s") is not None and lat["max_s"] <= rule["latency_max_s"]
         and lat.get("dense_p99_s") is not None and lat["dense_p99_s"] <= rule["dense_p99_s"]
         and lat.get("dense_max_s") is not None and lat["dense_max_s"] <= rule["dense_max_s"]}
    gg = {"checks_ok": bool(g["checks_ok"]), "complete": bool(g["complete"]),
          "invalid_X": int(g.get("invalid_X", 0)), "invalid_A3": int(g.get("invalid_A3", 0))}
    gg["ok"] = gg["checks_ok"] and gg["complete"] and not gg["invalid_X"] and not gg["invalid_A3"]
    conds = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": gg}
    holds = {k: (all(v["ok"] for v in conds[k].values()) if k in "abcd" else conds[k]["ok"])
             for k in "abcdefg"}
    passed = all(holds.values())
    return {"conditions": conds, "holds": holds, "pass": passed,
            "outcome": "pass: X* replaces A3" if passed else "fail: archive-3 stays",
            "rule": dict(rule)}


def power(se, gains=(0.002, 0.004, 0.006), rule=RULE):
    """(a)'s pass probability in one regime under a normal D with this SE: D <= -0.002 and
    D + 1.96 SE < 0 (descriptive)."""
    from math import erf, sqrt
    if not se:
        return None
    t = min(rule["gain"], -rule["z95"] * se)
    return {f"{x:g}": 0.5 * (1 + erf((t + x) / se / sqrt(2))) for x in gains}


# --- workers -------------------------------------------------------------------------------

def _paths():
    """The module paths a worker must share with its parent (tests move them)."""
    return {"OUT": OUT, "ROWS": ROWS, "EVAL_DIR": EVAL_DIR, "LEVELS": LEVELS,
            "STATS_DIR": STATS_DIR, "BUNDLES_DIR": BUNDLES_DIR, "A3_PRIOR": A3_PRIOR,
            "FITS_RECORD": FITS_RECORD, "PL.EVAL_DIR": PL.EVAL_DIR, "PL.FITS_DIR": PL.FITS_DIR,
            "PL.POOL": PL.POOL, "PL.TOOLS": PL.TOOLS}


def _set_paths(p):
    g = globals()
    for k, v in p.items():
        if k.startswith("PL."):
            setattr(PL, k[3:], v)
        else:
            g[k] = v


def _worker(task):
    fn, paths, args = task
    _set_paths(paths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return globals()[fn](*args)


def run_tasks(fn, arglist, jobs=1, fresh=False):
    """[fn(*args) for args in arglist], in order: in this process, or (jobs > 1, or fresh)
    in up to `jobs` fresh processes, one task each, so each peak resident set is its own."""
    if not fresh and (jobs <= 1 or len(arglist) <= 1):
        return [_worker((fn, _paths(), args)) for args in arglist]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(max(1, min(jobs, len(arglist))), maxtasksperchild=1) as pool:
        return pool.map(_worker, [(fn, _paths(), args) for args in arglist], chunksize=1)


# --- stage guards (postlock is frozen: DECISION) ----------------------------------------------

def prelock(a, stage):
    """A pre-lock stage: refused once a lock exists (its outputs are locked)."""
    state = load_state(a.state)
    if state.get("lock"):
        raise SystemExit(f"{stage} is a pre-lock stage and the lock exists; its outputs are locked")
    return state


def postlock(a, stage):
    """Every stage after the lock re-runs the lock check and refuses when it fails (PE 9)."""
    state = load_state(a.state)
    if not state.get("lock"):
        raise SystemExit(f"{stage}: no lock in {a.state}; run the lock stage first")
    problems = lock_check(state, a)
    log_check(stage, "lock check", not problems, problems, a.state)
    if problems:
        raise SystemExit(f"{stage}: the lock check failed: {problems}")
    return load_state(a.state)


def rel(path):
    return os.path.relpath(path, ROOT)


# --- stage: split -------------------------------------------------------------------------

def stage_split(a):
    prelock(a, "split")
    t0 = time.time()
    bench, fams = PL.read_lists()
    levels = derive_levels(bench["E"], a.root)
    try:
        csv, ok = check_levels_csv(levels), True
    except PoolEvalError as exc:
        csv, ok = {"error": str(exc)}, False
    log_check("split", "levels against the planning tables", ok, csv, a.state)
    sha = write_levels(levels)
    pool = PoolView.from_files(helm=False, root=a.root)
    half, units = split_halves(pool)
    plan = read_json(os.path.join(ROOT, SPLIT_PLANNING))["splits"]["family"]
    differ = sorted(b for b in set(plan) | set(half) if plan.get(b) != half.get(b))
    log_check("split", "split reproduces the planning halves", not differ, {"differ": differ}, a.state)
    try:
        fam, fok = PL.check_families(fams, os.path.join(ROOT, FAMILIES_PLANNING)), True
    except PL.PoolError as exc:
        fam, fok = {"error": str(exc)}, False
    log_check("split", "families reproduce the planning table", fok, fam, a.state)
    halves = {H: sorted(b for b in pool.E if half[b] == H) for H in HALVES}
    summ = {H: half_summary(pool, halves[H]) for H in HALVES}
    cok = all(summ[H]["benchmarks"] == SPLIT_EXPECTED["benchmarks"][H]
              and summ[H]["pairs"] == SPLIT_EXPECTED["pairs"][H] for H in HALVES) \
        and tuple(summ["S"]["flagged"]) == SPLIT_EXPECTED["S_flagged"]
    log_check("split", "split counts", cok, {H: [summ[H]["benchmarks"], summ[H]["pairs"]]
                                             for H in HALVES}, a.state)
    rec = {"levels": {"path": rel(LEVELS), "sha256": sha, "benchmarks": len(levels),
                      "pairs": sum(len(v) for v in levels.values())},
           "csv": csv, "halves": halves, "summary": summ, "units": units, "rule": SPLIT,
           "ok": bool(ok and not differ and fok and cok), "wall_s": round(time.time() - t0, 1),
           "provenance": provenance()}
    save_stage("split", rec, a.state)
    print(f"split: S {len(halves['S'])}, C {len(halves['C'])}; {'ok' if rec['ok'] else 'FAILED'}")


# --- stage: regimes -----------------------------------------------------------------------

def dense_cap(state):
    """The dense item cap: DENSE['max_items'], or ADJUST's when timing tripped the one
    allowed adjustment (PE 5.4)."""
    adj = ((state.get("timing") or {}).get("adjustment") or {}).get("applied")
    return ADJUST["dense_max_items"] if adj else DENSE["max_items"]


def _knob_task(half, regime, benchmarks, root=None):
    """The knob search of one tilted regime in one half (PE 5.3): every grid setting's
    proxy realisation over CAL_DRAWS draws of CAL_SEED (each drawn pair's all-item logit
    from levels.json), and its objective."""
    pool = PoolView.from_files(helm=False, root=root)
    H = Half(pool, half, benchmarks)
    target = REGIMES[regime]["target"]
    rows = []
    for knob in knob_grid(regime):
        L = []
        for i in range(CAL_DRAWS):
            picks = draw_picks(H, regime, knob, run_rng(CAL_SEED, half, regime, i))
            L += [float(H.logit[bj][j]) for bj, j in picks]
        r = realised(L)
        rows.append({"knob": list(knob), "objective": objective(r, target),
                     **{k: r[k] for k in ("mean", "sd", "share_below", "rest_mean")}})
    best = min(range(len(rows)), key=lambda k: rows[k]["objective"])
    return {"half": half, "regime": regime, "settings": rows, "argmin": rows[best]["knob"],
            "argmin_index": best}


def _plan_task(half, regime, knob, benchmarks, dense_max, root=None):
    """At a regime's knobs: the CAL_SEED value (the same CAL_DRAWS runs drawn again with
    their items, realised on evaluated responses), the proxy beside it, and the planned
    compositions of the half's scoring seed with their realisation."""
    pool = PoolView.from_files(helm=False, root=root)
    H = Half(pool, half, benchmarks)
    stats = LazyStats(root)
    knob = None if knob is None else tuple(knob)
    shape = REGIMES[regime]["shape"]
    pl = {(b, s): float(H.logit[H.idx[b]][j]) for b in H.bs for j, s in enumerate(H.sids[H.idx[b]])}

    def measure(seed, n):
        ev, px, runs, sizes = [], [], {}, []
        for i in range(n):
            entries = draw_run(H, stats, regime, knob, seed, i, dense_max_items=dense_max)
            runs[i] = entries
            ev += [describe_entry(stats, e)["logit"] for e in entries]
            px += [pl[(e[0], e[1])] for e in entries]
            sizes.append(len(entries))
        return runs, realised(ev), realised(px), float(np.mean(sizes))

    _, cal, cal_proxy, cal_size = measure(CAL_SEED, CAL_DRAWS)
    seed = STAGE_SEED[half]
    n = N_RUNS[half][shape]
    runs, score, score_proxy, size = measure(seed, n)
    ok, w = H.weights(comps_of(regime, knob), shape)
    return {"half": half, "regime": regime, "knob": None if knob is None else list(knob),
            "comps": comps_of(regime, knob), "cal": cal, "cal_proxy": cal_proxy,
            "cal_pairs_per_run": cal_size, "score": score, "score_proxy": score_proxy,
            "score_pairs_per_run": size, "seed": seed, "runs": runs, "weights": H.weight_stats(ok, w),
            "dense_max_items": dense_max if shape == "dense" else None}


def realisation_checks(cal, score, regime):
    import experiments.regime_sensitivity as RS
    keys = ("mean", "sd") + (("share_below", "rest_mean") if REGIMES[regime].get("target") == "MIXTURE" else ())
    return {k: {"cal": cal[k], "score": score[k], "diff": score[k] - cal[k], "tol": RS.REALISE_TOL[k],
                "ok": abs(score[k] - cal[k]) <= RS.REALISE_TOL[k]} for k in keys}


def short_of_target(cal, regime):
    import experiments.regime_sensitivity as RS
    target = REGIMES[regime].get("target")
    if target is None:
        return None
    tg = RS.TARGETS[target]
    off = {k: cal[k] - v for k, v in tg.items() if k != "cut"}
    short = any(abs(off[k]) > SHORT_OF_TARGET[k] for k in off if k in SHORT_OF_TARGET)
    return {"off": off, "short": bool(short)}


def compositions_path(half, rows=None):
    return os.path.join(rows or ROWS, half, "_compositions.json")


def by_benchmark(pool, entries, shift=False):
    """{benchmark: [subject dict of each appearance]} (twins: dates moved)."""
    out = defaultdict(list)
    for b, sid, _, _ in entries:
        s = dict(pool.subject(b, sid))
        if shift:
            for f in ("release_date", "access_date"):
                s[f] = T.shift_date(s[f], DATE_SHIFT)
        out[b].append(s)
    return out


def centres(pool, comps, arms):
    """c(X, .) - c(A3, .) on every regime's planned appearances at the regime's dates
    (PE 4.2, 9 item 2), with the excl-empty builds."""
    out = {}
    for H, regs in comps.items():
        for R in REGIMES:
            parent = TWINS.get(R, R)
            ents = [e for run in regs[parent]["runs"].values() for e in run]
            by = by_benchmark(pool, ents, shift=R in TWINS)
            c = {X: PL.b0_centre(p, h, by)[0] for X, (p, h) in arms.items()}
            out.setdefault(H, {})[R] = {"A3": c[A3], **{f"{X}-A3": c[X] - c[A3] for X in TIE_ORDER}}
    return out


def archive_arms():
    """A3 and the three archive bundles (excl empty), as the fits stage wrote them."""
    arms = {A3: a3_bundle()}
    for X in TIE_ORDER:
        with open(os.path.join(BUNDLES_DIR, f"archive_{X}.json")) as f:
            arms[X] = PR.from_json(json.load(f))
    return arms


def regimes_centres_only(a, state):
    """Re-run the regimes stage's B0-centre part alone, on the written compositions (the
    fits stage may finish after the rest of the stage)."""
    rec = state["regimes"]
    pool = PoolView.from_files(helm=False, root=a.root)
    comps = {H: read_compositions(H, state)["regimes"] for H in HALVES}
    try:
        cen, cok = centres(pool, comps, archive_arms()), True
    except Exception as exc:
        cen, cok = {"error": f"{type(exc).__name__}: {exc}"}, False
    log_check("regimes", "centres", cok, None if cok else cen, a.state)
    rec["centres"] = cen
    rec["ok"] = bool(rec.get("realisation_ok") and rec.get("balance_ok") and cok)
    save_stage("regimes", rec, a.state)
    print(f"regimes (centres): {'ok' if cok else 'FAILED'}")


def stage_regimes(a):
    state = prelock(a, "regimes")
    if not (state.get("split") or {}).get("ok"):
        raise SystemExit("regimes needs a passed split stage")
    if getattr(a, "centres_only", False):
        return regimes_centres_only(a, state)
    t0 = time.time()
    halves = state["split"]["halves"]
    dense_max = dense_cap(state)
    tilted = [(H, R) for H in HALVES for R in REGIMES if R not in TWINS and REGIMES[R].get("target")]
    print(f"regimes: knob search, {len(tilted)} searches on {a.jobs} processes", flush=True)
    knobs = {}
    for res in run_tasks("_knob_task", [(H, R, halves[H], a.root) for H, R in tilted], a.jobs):
        knobs[(res["half"], res["regime"])] = res
    print(f"regimes: pair statistics of E ({time.time() - t0:.0f}s)", flush=True)
    pool = PoolView.from_files(helm=False, root=a.root)
    build_stats(pool.E, a.jobs, a.root)
    plan = [(H, R) for H in HALVES for R in REGIMES if R not in TWINS]
    print(f"regimes: CAL_SEED values and planned compositions ({time.time() - t0:.0f}s)", flush=True)
    got = run_tasks("_plan_task", [(H, R, (knobs[(H, R)]["argmin"] if (H, R) in knobs else None),
                                    halves[H], dense_max, a.root) for H, R in plan], a.jobs)
    res = {(g["half"], g["regime"]): g for g in got}
    comps, rec_regimes, real_ok = {}, {}, True
    for H, R in plan:
        g = res[(H, R)]
        comps.setdefault(H, {})[R] = {"knob": g["knob"], "comps": g["comps"], "seed": g["seed"],
                                      "dense_max_items": g["dense_max_items"],
                                      "runs": {str(i): e for i, e in sorted(g["runs"].items())}}
        chk = realisation_checks(g["cal"], g["score"], R)
        ok = all(v["ok"] for v in chk.values())
        real_ok &= ok
        log_check("regimes", f"realisation {H} {R}", ok, chk, a.state)
        if (H, R) in knobs:
            # the CAL_SEED runs drawn again with their items are the search's runs at the argmin
            row = knobs[(H, R)]["settings"][knobs[(H, R)]["argmin_index"]]
            same = all(abs(row[k] - g["cal_proxy"][k]) <= 1e-12 for k in ("mean", "sd"))
            real_ok &= same
            log_check("regimes", f"knob search reproduced {H} {R}", same,
                      {"search": [row["mean"], row["sd"]],
                       "redrawn": [g["cal_proxy"]["mean"], g["cal_proxy"]["sd"]]}, a.state)
        rec_regimes.setdefault(H, {})[R] = {
            "knob": g["knob"], "cal": g["cal"], "cal_proxy": g["cal_proxy"], "score": g["score"],
            "score_proxy": g["score_proxy"], "cal_pairs_per_run": g["cal_pairs_per_run"],
            "score_pairs_per_run": g["score_pairs_per_run"], "checks": chk,
            "short_of_target": short_of_target(g["cal"], R), "weights": g["weights"],
            "search": None if (H, R) not in knobs else
            {"settings": len(knobs[(H, R)]["settings"]), "argmin": knobs[(H, R)]["argmin"],
             "argmin_row": knobs[(H, R)]["settings"][knobs[(H, R)]["argmin_index"]]}}
    shas = {}
    for H in HALVES:
        path = compositions_path(H)
        write_json(path, {"half": H, "seed": STAGE_SEED[H], "dense_max_items": dense_max,
                          "regimes": comps[H]})
        shas[H] = file_sha(path)
    log_check("regimes", "compositions written", True, {H: shas[H][:16] for H in HALVES}, a.state)
    bal = {}
    for R in REGIMES:
        if R in TWINS or not REGIMES[R].get("target"):
            continue
        s = rec_regimes["S"][R]["weights"]["flagged_weight_share"]
        c = rec_regimes["C"][R]["weights"]["flagged_weight_share"]
        bal[R] = {"S": s, "C": c, "abs_diff": abs(s - c), "ok": abs(s - c) <= BALANCE_MAX}
    bok = all(v["ok"] for v in bal.values())
    log_check("regimes", "balance", bok, bal, a.state)
    try:
        cen, cok = centres(pool, comps, archive_arms()), True
    except Exception as exc:                       # the fits have not run: the check fails
        cen, cok = {"error": f"{type(exc).__name__}: {exc}"}, False
    log_check("regimes", "centres", cok, None if cok else cen, a.state)
    rec = {"regimes": rec_regimes, "balance": bal, "centres": cen, "dense_max_items": dense_max,
           "compositions": {H: {"path": rel(compositions_path(H)), "sha256": shas[H]} for H in HALVES},
           "knob_search": {f"{H} {R}": v["settings"] for (H, R), v in knobs.items()},
           "ok": bool(real_ok and bok and cok), "realisation_ok": bool(real_ok), "balance_ok": bool(bok),
           "cal_seed": CAL_SEED, "cal_draws": CAL_DRAWS,
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("regimes", rec, a.state)
    print(f"regimes: {'ok' if rec['ok'] else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)


def read_compositions(half, state=None, rows=None):
    """A half's planned compositions, checked against the digest the regimes stage (and
    the lock, once it exists) recorded."""
    path = compositions_path(half, rows)
    if state is not None:
        want = ((state.get("lock") or {}).get("files") or {}).get(rel(compositions_path(half))) \
            or state["regimes"]["compositions"][half]["sha256"]
        if file_sha(path) != want:
            raise PoolEvalError(f"{path} differs from what was recorded")
    return read_json(path)


# --- stage: fits-extra (descriptive records the fits stage leaves to this script) -------

def r4_slopes(free, fixed, S, flagged):
    """The R4 slopes on the final fits (PE 4.4, a4final/slopes_s.py; record only): the
    standings rows of U on S plus the legacy five, from_standings' ridge, residual cells
    and offsets on those rows alone; held-out cells on S; per bin of the number of other
    cells, the slope and the model weight at the raw tau2. C3 stays dropped either way."""
    names = [b for b in sorted(S) + list(PL.LEGACY) if b not in flagged and (b in free or b in fixed)]
    fits = {b: free.get(b) or fixed[b] for b in names}
    rows = PL.standings_rows(fits, names)
    p = PR.from_standings(rows, PL.ALPHA)
    tau, s2d = float(p.meta["tau2_res"]), float(p.meta["s2d_res"])
    _, _, X, _ = PR._ridge(rows, PL.ALPHA)
    resid, _ = PR._residuals(rows, PL.ALPHA, X)
    cells = defaultdict(list)
    for i, (b, s, _, _) in enumerate(rows):
        k = PR.canon_name(s)
        if k and np.isfinite(resid[i]):
            cells[(k, b)].append(resid[i])
    cells = {kb: float(np.mean(v)) for kb, v in cells.items()}
    off = PR._offsets(cells)
    cells = {(k, b): v - off.get(b, 0.0) for (k, b), v in cells.items()}
    by_k = defaultdict(dict)
    for (k, b), v in cells.items():
        by_k[k][b] = v
    pts = defaultdict(list)
    for k in sorted(by_k):
        d = by_k[k]
        for b in sorted(d):
            if b not in set(S):
                continue
            others = [v for bb, v in sorted(d.items()) if bb != b]
            kk = len(others)
            if kk:
                for g in (("1",) if kk == 1 else ("2",) if kk == 2 else ("3+",)):
                    pts[g].append((float(np.mean(others)), d[b], kk))
    out = {}
    for g in ("1", "2", "3+"):
        arr = np.array(pts.get(g, []))
        if len(arr) < 3:
            out[g] = {"cells": int(len(arr))}
            continue
        x, y = arr[:, 0] - arr[:, 0].mean(), arr[:, 1] - arr[:, 1].mean()
        kbar = float(arr[:, 2].mean())
        out[g] = {"cells": int(len(arr)), "slope": float(x @ y / (x @ x)), "k_mean": kbar,
                  "model_w_raw_tau": tau / (tau + s2d / kbar) if tau > 0 else 0.0}
    keep = all("slope" in out[g] and out[g]["model_w_raw_tau"] <= out[g]["slope"] for g in ("1", "3+"))
    return {"rows": len(rows), "benchmarks": len({r[0] for r in rows}), "tau2_res": tau,
            "s2d_res": s2d, "bins": out, "criterion_would_keep_C3": bool(keep),
            "note": "record only; C3 stays dropped (PE 4.4)"}


def mu0_runs(pool, ctx, comps, n=50, regime="P-READING"):
    """C1m's matching rule without each run's families (PE 4.3; descriptive): the first n
    planned runs of `regime` in each half, mu0 solving c(C1m without the run's families,
    E at +1.25 years) = c(A3, E at +1.25 years), E without its degenerate benchmarks as
    the matching reads it (amendment 4a, paiec.pool.match_set)."""
    a3p, a3h = a3_bundle()
    _, ds = PL.subject_sets(pool.E, pool.root)
    ds = PL.match_set(ds, ctx.free)
    target = PL.b0_centre(a3p, a3h, ds)[0]
    out = {}
    for H in HALVES:
        vals = []
        for i in range(n):
            ents = comps[H]["regimes"][regime]["runs"][str(i)]
            excl = PL.excl_for(sorted({e[0] for e in ents}), pool.family_of, ctx.L)
            p, h, _ = PL.candidate_bundle("C1", excl, ctx.free, ctx.fixed, ctx.L, ctx.U)
            vals.append(PL.match_mu0(p, h, ds, target))
        v = np.array(vals)
        out[H] = {"runs": n, "regime": regime, "mean": float(v.mean()), "sd": float(v.std()),
                  "min": float(v.min()), "max": float(v.max()),
                  "minus_locked": float(v.mean() - ctx.mu0_m)}
    return {"locked_mu0_m": ctx.mu0_m, "target_A3_plus_1.25y": target, "halves": out,
            "matching_set": {"benchmarks": len(ds), "left_out": PL.degenerate(ctx.free)}}


def stage_fits(a):
    """The fits stage (impl_plan step 4): paiec.pool.fits_stage, its record to --fits-record
    (results/pool_fits.json by default)."""
    prelock(a, "fits")
    rec = PL.fits_stage(a.jobs, a.root, a.fits_record)
    print(json.dumps({k: rec[k] for k in ("T_full", "mu0_m", "wall_s")}, default=str))


def stage_fits_extra(a):
    state = prelock(a, "fits-extra")
    t0 = time.time()
    try:
        fr = read_json(a.fits_record)
        ok = fr.get("manifest", {}).get("sha256") == PINNED["data/pool/manifest.json"] and \
            "mu0_m" in fr and "bundles" in fr
        for X in TIE_ORDER:
            ok &= file_sha(os.path.join(ROOT, fr["bundles"][X]["path"])) == fr["bundles"][X]["sha256"]
        ok &= PL.fits_index_sha256() == fr["fits_index_sha256"]
    except Exception as exc:
        fr, ok = {"error": str(exc)}, False
    log_check("fits-extra", "fits record", ok, None if ok else fr, a.state)
    if not ok:
        raise SystemExit("fits-extra needs the fits stage's record and its bundles")
    if not (state.get("regimes") or {}).get("compositions"):
        raise SystemExit("fits-extra needs the regimes stage's compositions")
    ctx = FitsContext(a.fits_record)
    pool = PoolView.from_files(helm=False, root=a.root)
    comps = {H: read_compositions(H, state) for H in HALVES}
    rec = {"fits_record": {"path": rel(a.fits_record), "sha256": file_sha(a.fits_record)},
           "r4_slopes": r4_slopes(ctx.free, ctx.fixed, state["split"]["halves"]["S"], pool.flagged),
           "mu0_m_runs": mu0_runs(pool, ctx, comps), "ok": True,
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("fits-extra", rec, a.state)
    print(f"fits-extra: done ({time.time() - t0:.0f}s)")


# --- the TIMING_SEED runs (RP2, RP3, timing; never scored) -----------------------------------

def timing_compositions(pool, halves, dense_max, stats=None, root=None):
    """The compositions of the timing stage's runs (TIMING_RUNS) and of RP3's (RP3_RUNS),
    drawn on TIMING_SEED, untilted: {'timing': {i: entries}, 'rp3': {i: entries}}."""
    out = {"timing": {}, "rp3": {}}
    Hs = {H: Half(pool, H, halves[H]) for H in HALVES}
    stats = stats if stats is not None else LazyStats(root)
    for i, H, R, first in TIMING_RUNS:
        out["timing"][i] = draw_run(Hs[H], stats, R, None, TIMING_SEED, i, first=first,
                                    dense_max_items=dense_max)
    for i, H, R, k in RP3_RUNS:
        out["rp3"][i] = draw_run(Hs[H], stats, R, None, TIMING_SEED, k, dense_max_items=dense_max)
    return out


def candidate_arms(cache, excl, names=TIE_ORDER):
    """{X: (ArmFactory, the bundle record)} for the candidates `names` without excl."""
    out = {}
    for X in names:
        bd = cache.get(X, excl)
        out[X] = (ArmFactory({None: (bd["prior"], bd["hyper"])}), bd)
    return out


def a3_arm():
    return ArmFactory({None: a3_bundle()})


class ItemReadTimer:
    """While active, the CPU time spent in paiec.pool.items_of, through which
    load_eval_pairs reads the kept items' content (timing's item-read CPU, PE 9 item 9),
    and its calls; paiec.pool.items_of is restored on exit."""

    def __enter__(self):
        self.cpu, self.calls, self._orig = 0.0, 0, PL.items_of

        def timed(*args, **kw):
            c = time.process_time()
            try:
                return self._orig(*args, **kw)
            finally:
                self.cpu += time.process_time() - c
                self.calls += 1
        PL.items_of = timed
        return self

    def __exit__(self, *exc):
        PL.items_of = self._orig
        return False


def _timing_task(i, entries, paths_fits_record, label_kind="timing", dense=None):
    """One timing run on a fresh scoring worker, on its label-randomised copy (V12): load
    (the item read's CPU apart), randomise, assert, bundles, checkpoints, every arm's calls
    timed. No Brier, no ALC: predictions are not compared with anything."""
    c0 = time.process_time()
    rss0 = rss_gb()
    with ItemReadTimer() as items:
        run = load_run(entries)
    load_cpu = time.process_time() - c0
    rss_loaded = rss_gb()
    copy = label_randomised(run, label_stream(label_kind, i))
    del run
    n = assert_randomised(copy, label_stream(label_kind, i))
    ctx = FitsContext(paths_fits_record)
    cache = BundleCache(ctx)
    pool = PoolView.from_files(helm=False)
    excl = PL.excl_for(sorted({p.benchmark_id for p, _ in copy}), pool.family_of, ctx.L)
    c1 = time.process_time()
    arms = {A3: a3_arm()}
    bundles = {}
    for X, (fac, bd) in candidate_arms(cache, excl).items():
        arms[X] = fac
        bundles[X] = {"sha256": bd["sha256"], "key": bd["key"], "cpu_s": bd["cpu_s"]}
    bundle_cpu = time.process_time() - c1
    c2 = time.process_time()
    slots, cps = checkpoints(copy)
    cps_cpu = time.process_time() - c2
    if dense is None:
        dense = any(REGIMES[R]["shape"] == "dense" for j, _, R, _ in TIMING_RUNS if j == i)
    out = {}
    for name, fac in arms.items():
        t = predict_timed(cps, fac, name, f"timing {i}", keep_calls=True)
        calls = t.pop("calls_cpu")
        t.pop("preds")
        t["p99_s"] = p99(calls) if calls else None
        if dense:
            t["calls_cpu"] = calls
        t["invalid"] = len(t["invalid"])
        out[name] = t
    return assert_no_scores({"i": i, "pairs": len(copy), "benchmarks": sorted({p.benchmark_id for p, _ in copy}),
            "responses": n, "excl": list(excl), "dense": dense, "load_cpu": load_cpu,
            "item_read_cpu": items.cpu, "item_reads": items.calls,
            "load_cpu_without_item_read": load_cpu - items.cpu,
            "bundle_cpu": bundle_cpu, "bundles": bundles, "checkpoints_cpu": cps_cpu,
            "arms": out, "rss_gb_start": rss0, "rss_gb_loaded": rss_loaded, "rss_gb_end": rss_gb(),
            "maxrss_gb": maxrss_gb(), "task_cpu": time.process_time() - c0,
            "labels": {"stream": label_stream(label_kind, i), "randomised": n}})


def projection(timing_rows, state, fits_rec=None):
    """The core's process-hours at the measured unit costs (PE 12), planning costs where
    timing does not measure: S (P-: 560 plays x 4 arms; D-: 40 x 4), RS (520 x 3), the C
    core (P-: 840 x 2; D-: 80 x 2), each run's bundles and loading, and the fixed stages."""
    f = [r for r in timing_rows if not r["dense"] and r["i"] < 6]
    d = [r for r in timing_rows if r["dense"]]
    arm_f = float(np.mean([r["arms"][x]["arm_cpu"] for r in f for x in r["arms"]])) if f else 0.0
    arm_d = float(np.mean([r["arms"][x]["arm_cpu"] for r in d for x in r["arms"]])) if d else 0.0
    load_f = float(np.mean([r["load_cpu"] + r["checkpoints_cpu"] for r in f])) if f else 0.0
    load_d = float(np.mean([r["load_cpu"] + r["checkpoints_cpu"] for r in d])) if d else 0.0
    bun = float(np.mean([r["bundle_cpu"] for r in timing_rows])) if timing_rows else 0.0
    nS, nC = N_RUNS["S"], N_RUNS["C"]
    pS = 5 * nS["formative"] + len(TWINS) * nS["formative"]
    pC = 5 * nC["formative"] + len(TWINS) * nC["formative"]
    rs_arm = PLANNING_COSTS["rs_task_arm_s"]
    rp1 = ((state.get("reproduce") or {}).get("RP1") or {}).get("a3_cpu_per_task")
    if rp1:
        rs_arm = float(rp1)
    fixed = PLANNING_COSTS["fits_s"]
    if fits_rec and "cpu_s" in fits_rec:
        fixed = fits_rec["cpu_s"]["free_fits"] + fits_rec["cpu_s"]["fixed_T_fits"] + fits_rec["cpu_s"]["loads"]
    parts = {
        "S_P": pS * (4 * arm_f + load_f) + 5 * nS["formative"] * bun,
        "S_D": 2 * nS["dense"] * (4 * arm_d + load_d + bun),
        "RS": sum(RS_TASKS.values()) * 3 * rs_arm,
        "C_P": pC * (2 * arm_f + load_f) + 5 * nC["formative"] * bun / 3,
        "C_D": 2 * nC["dense"] * (2 * arm_d + load_d + bun / 3),
        "fits": fixed, "regimes": float((state.get("regimes") or {}).get("wall_s") or PLANNING_COSTS["regimes_s"]),
        "reproduce": PLANNING_COSTS["reproduce_s"], "timing": float(sum(r["task_cpu"] for r in timing_rows)),
        "summarise": PLANNING_COSTS["summarise_s"]}
    total = sum(parts.values())
    return {"unit_s": {"arm_formative": arm_f, "arm_dense": arm_d, "load_formative": load_f,
                       "load_dense": load_d, "bundles_per_run": bun, "rs_task_arm": rs_arm},
            "parts_s": parts, "core_s": total, "core_process_hours": total / 3600}


def heaviest_tasks(state, k=1, root=None):
    """The planned C tasks with the most evaluated-or-not responses (counts from the pair
    statistics, ids only): [(responses, regime, i)] per shape, largest first."""
    stats = LazyStats(root)
    comps = read_compositions("C", state)["regimes"]
    out = {}
    for R, v in comps.items():
        tot = []
        for i, ents in v["runs"].items():
            n = sum(int(stats[e[0]].counts(e[1], kept_items(e))[0].sum()) for e in ents)
            tot.append((n, R, int(i)))
        shape = REGIMES[R]["shape"]
        out.setdefault(shape, []).extend(tot)
    return {shape: sorted(v, reverse=True)[:k] for shape, v in out.items()}


def _probe_task(regime, i, entries):
    """The memory and CPU of loading one planned task and building its checkpoints, on a
    fresh worker (descriptive): no predictor is called and nothing is scored."""
    rss0, c0 = rss_gb(), time.process_time()
    run = load_run(entries)
    c1, rss1 = time.process_time(), rss_gb()
    slots, cps = checkpoints(run)
    c2, rss2 = time.process_time(), rss_gb()
    out = {"regime": regime, "i": i, "pairs": len(run),
           "responses": sum(len(p.responses) for p, _ in run),
           "targets": sum(len(s.targets) for s in slots),
           "distinct_inputs": [len(c[2]) for c in cps], "load_cpu": c1 - c0, "checkpoints_cpu": c2 - c1,
           "rss_gb": {"start": rss0, "loaded": rss1, "checkpoints": rss2}, "maxrss_gb": maxrss_gb()}
    return assert_no_scores(out)


def probe_heaviest(a, state):
    """The heaviest planned C task of each shape, loaded on a fresh worker (descriptive; the
    timing stage's runs come from S and two fixed C pairs, and a C task with a
    monkey_power_laws pair holds millions of responses)."""
    heavy = heaviest_tasks(state, 1, a.root)
    comps = read_compositions("C", state)["regimes"]
    tasks = [(R, i, comps[R]["runs"][str(i)]) for shape in sorted(heavy) for _, R, i in heavy[shape]]
    got = run_tasks("_probe_task", tasks, a.jobs, fresh=True)
    return {"tasks": got, "responses_by_shape": {s: v[0][0] for s, v in heavy.items()},
            "note": "descriptive: no predictor called, nothing scored; not one of the timing "
                    "stage's checks"}


def stage_timing(a):
    state = prelock(a, "timing")
    if getattr(a, "probe_heaviest", False):
        rec = probe_heaviest(a, state)
        save_stage("timing_probe", rec, a.state)
        for t in rec["tasks"]:
            print(f"  probe C {t['regime']} {t['i']}: {t['pairs']} pairs, {t['responses']} responses, "
                  f"load {t['load_cpu']:.1f} s, checkpoints {t['checkpoints_cpu']:.1f} s, "
                  f"rss {t['rss_gb']}, maxrss {t['maxrss_gb']:.2f} GB")
        return
    t0 = time.time()
    pool = PoolView.from_files(helm=False, root=a.root)
    halves = state["split"]["halves"]
    dense_max = DENSE["max_items"]
    comps = timing_compositions(pool, halves, dense_max, root=a.root)
    write_json(os.path.join(EVAL_DIR, "timing_compositions.json"), comps)
    print(f"timing: {len(TIMING_RUNS)} runs on fresh workers, {a.jobs} at a time", flush=True)
    rows = run_tasks("_timing_task", [(i, comps["timing"][i], a.fits_record) for i, *_ in TIMING_RUNS],
                     a.jobs, fresh=True)
    calls = {}
    for r in rows:
        print(f"  timing run {r['i']}: {r['pairs']} pairs, {r['responses']} responses, "
              f"load {r['load_cpu']:.2f} s (items {r['item_read_cpu']:.2f} s), "
              f"maxrss {r['maxrss_gb']:.2f} GB, A3 {r['arms'][A3]['arm_cpu']:.1f} s", flush=True)
        for x, v in r["arms"].items():
            if "calls_cpu" in v:          # every dense call's CPU time, beside the record
                calls.setdefault(str(r["i"]), {})[x] = v.pop("calls_cpu")
    write_json(os.path.join(EVAL_DIR, "timing_calls.json"), calls)
    labels_ok = all(r["labels"]["randomised"] == r["responses"] for r in rows)
    log_check("timing", "labels randomised", labels_ok,
              {r["i"]: r["labels"] for r in rows}, a.state)
    bad = {r["i"]: {x: v["invalid"] for x, v in r["arms"].items() if v["invalid"]} for r in rows}
    bad = {k: v for k, v in bad.items() if v}
    log_check("timing", "calls valid", not bad, bad, a.state)
    rss = {r["i"]: {"maxrss_gb": r["maxrss_gb"], "rss_gb_end": r["rss_gb_end"]} for r in rows}
    rss_ok = all(v["maxrss_gb"] <= RSS_GUARD_GB for v in rss.values())
    log_check("timing", "resident set", rss_ok, rss, a.state)
    fits_rec = read_json(a.fits_record) if os.path.exists(a.fits_record) else None
    proj = projection(rows, state, fits_rec)
    adjusted = proj["core_process_hours"] > ADJUST["core_process_hours"]
    adj = {"applied": bool(adjusted), "threshold_process_hours": ADJUST["core_process_hours"],
           "dense_max_items": ADJUST["dense_max_items"] if adjusted else DENSE["max_items"]}
    log_check("timing", "projection", True, {"core_process_hours": proj["core_process_hours"],
                                              "adjustment": adj}, a.state)
    rec = {"runs": rows, "projection": proj, "adjustment": adj,
           "compositions": {"path": rel(os.path.join(EVAL_DIR, "timing_compositions.json"))},
           "ok": bool(labels_ok and not bad and rss_ok), "wall_s": round(time.time() - t0, 1),
           "note": "label-randomised copies; no Brier or ALC is computed", "provenance": provenance()}
    save_stage("timing", rec, a.state)
    print(f"timing: projection {proj['core_process_hours']:.1f} process-hours; "
          f"adjustment {'applied' if adjusted else 'not needed'}; {'ok' if rec['ok'] else 'FAILED'}")


# --- stage: reproduce (RP1 to RP5) --------------------------------------------------------

def rs_rows_digest(root=None):
    """The 520 RS row files as one digest (V2): sha256 of the UTF-8 lines
    '<path>\\t<sha256>\\n', path relative to the rows directory in POSIX form, sorted."""
    root = root or RS_ROWS
    rows = []
    for R, n in sorted(RS_TASKS.items()):
        names = sorted(os.listdir(os.path.join(root, R)))
        want = sorted(f"{i}.json" for i in range(n))
        if [x for x in names if x.endswith(".json")] != want:
            raise PoolEvalError(f"{R}: expected the files 0..{n - 1}.json")
        rows += [(f"{R}/{x}", file_sha(os.path.join(root, R, x))) for x in want]
    text = "".join(f"{p}\t{h}\n" for p, h in sorted(rows))
    return {"files": len(rows), "digest": sha_bytes(text.encode("utf-8"))}


_RS_COMPS = {}


def _rs_compositions():
    path = os.path.join(RS_ROWS, "_compositions.json")
    if path not in _RS_COMPS:
        _RS_COMPS[path] = read_json(path)["runs"]
    return _RS_COMPS[path]


def rs_run(R, i):
    """P1a's run i of regime R on its seed 11, checked against its stored composition."""
    import experiments.regime_sensitivity as RS
    key, run = RS.draw(R, RS_SEED, i)
    meta = RS.describe(key, run)
    comps = _rs_compositions()
    if json.loads(json.dumps(meta)) != comps[R][str(i)]:
        raise PoolEvalError(f"RS {R} {i}: the drawn run differs from P1a's composition")
    return key, run, meta


def rp1():
    """RP1: A3 (P1a's ship, prior.build without the target's parent plus LEVEL) re-scored by
    evaluate_timed on RS runs 0 and 1 of each regime against the stored ship rows, per pair
    and budget, to REPRO_TOL. On P1a's own legacy-data runs, whose rows exist."""
    import experiments.regime_sensitivity as RS
    out, worst, cpu = [], 0.0, []
    for R in RS_REGIMES:
        for i in RP1_RUNS:
            key, run, meta = rs_run(R, i)
            slots, cps = checkpoints(run)
            res = evaluate_timed(slots, cps, {A3: RS.Maker(RS.SHIP, run)}, f"RS {R} {i}")[A3]
            stored = read_json(os.path.join(RS_ROWS, R, f"{i}.json"))["res"][RS.SHIP]["b"]
            mine = [x["brier"] for x in res["rows"]] if res["rows"] else None
            d = None if mine is None or len(mine) != len(stored) else \
                max(abs(x - y) for a, b in zip(mine, stored) for x, y in zip(a, b))
            worst = max(worst, d if d is not None else float("inf"))
            cpu.append(res["arm_cpu"])
            out.append({"regime": R, "run": i, "pairs": len(meta), "max_abs_brier": d,
                        "ok": d is not None and d <= REPRO_TOL})
    return {"tasks": out, "max_abs_brier": worst, "tol": REPRO_TOL, "a3_cpu_per_task": float(np.mean(cpu)),
            "ok": all(x["ok"] for x in out)}


#: RP2's fresh interpreter: the archive loaded as tools/build_submission.check_run loads it,
#: the runs built from their compositions through this module, predictions printed as hex
RP2_RUN = r"""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import importlib.util, json, sys, zipfile, tempfile
zip_path, spec_path, repo = sys.argv[1], sys.argv[2], sys.argv[3]
spec = json.load(open(spec_path))
tmp = tempfile.mkdtemp(prefix="paiec-rp2-")
root = os.path.join(tmp, "submission")
with zipfile.ZipFile(zip_path) as z:
    z.extractall(root)
before = list(sys.path)
sys.path.insert(0, root)
s = importlib.util.spec_from_file_location("submission_model", os.path.join(root, "model.py"))
model = importlib.util.module_from_spec(s)
s.loader.exec_module(model)
sys.path[:] = before
sys.path.insert(0, repo)
import experiments.pool_eval as E
out = {}
for name, r in spec["runs"].items():
    run = E.label_randomised(E.load_run(r["entries"]), r["stream"])
    if r["twin"]:
        run = E.twin_of(run)
    _, cps = E.checkpoints(run)
    out[name] = []
    for _, lab, inputs, _ in cps:
        P = model.make()                     # one fresh instance per checkpoint
        out[name].append([float(P.predict(inp, lab)).hex() for inp in inputs])
print(json.dumps({"model": model.MODEL, "runs": out}))
"""


def rp2(comps, archive=None):
    """RP2: archive-3's runtime from the zip in a fresh interpreter and the in-repo A3 give
    bit-identical predictions on RP3's two runs (their label-randomised copies), compared
    with each other only."""
    import tempfile
    archive = archive or os.path.join(ROOT, "dist", "archive-3.zip")
    if not os.path.exists(archive) or file_sha(archive) != PINNED["dist/archive-3.zip"]:
        return {"ok": False, "error": "no archive-3 copy with the pinned sha256"}
    spec = {"runs": {str(i): {"entries": comps[i], "stream": label_stream("rp3", i), "twin": False}
                     for i, _, _, _ in RP3_RUNS}}
    with tempfile.TemporaryDirectory(prefix="paiec-rp2-spec-") as tmp:
        path = os.path.join(tmp, "spec.json")
        with open(path, "w") as f:
            json.dump(spec, f)                    # ids and stream seeds only, no item text
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, "-c", RP2_RUN, archive, path, ROOT], cwd=tmp, env=env,
                           capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-600:]}
    got = json.loads(r.stdout.strip().splitlines()[-1])
    out = {}
    for name, sp in spec["runs"].items():
        run = label_randomised(load_run(sp["entries"]), sp["stream"])
        _, cps = checkpoints(run)
        mine = predict_timed(cps, a3_arm(), A3, raw=True)["raw"]
        theirs = [[float.fromhex(x) for x in c] for c in got["runs"][name]]
        out[name] = {"calls": sum(len(c) for c in mine), "identical": mine == theirs}
    return assert_no_scores({"model": got["model"], "runs": out, "ok": got["model"] == "hier"
                             and all(v["identical"] for v in out.values())})


def rp3(comps, cache, pool):
    """RP3 (N6, V11): evaluate_timed against official.run_official(..., workers=1) on RP3's
    two runs and run 0's twin, on label-randomised copies, for A3 and C1: identical
    predictions by checkpoint and call. run_official's scores are discarded; nothing
    here computes, keeps or prints a Brier or an ALC."""
    out = {}
    copies = {}
    for i, _, _, _ in RP3_RUNS:
        copies[str(i)] = label_randomised(load_run(comps[i]), label_stream("rp3", i))
        assert_randomised(copies[str(i)], label_stream("rp3", i))
    copies["0-DS"] = twin_of(copies["0"])
    for name, run in copies.items():
        excl = PL.excl_for(sorted({p.benchmark_id for p, _ in run}), pool.family_of, cache.ctx.L)
        bd = cache.get("C1", excl)
        arms = {A3: lambda: a3_arm(), "C1": lambda bd=bd: ArmFactory({None: (bd["prior"], bd["hyper"])})}
        res = {}
        for arm, make in arms.items():
            _, cps = checkpoints(run)
            mine = predict_timed(cps, make(), arm, raw=True)["raw"]
            theirs = recorded_official(run, make())
            res[arm] = {"checkpoints": len(theirs), "calls": sum(len(c) for c in theirs),
                        "identical": mine == theirs}
        out[name] = res
    return assert_no_scores({"runs": out, "ok": all(v["identical"] for r in out.values() for v in r.values()),
                             "labels": {k: label_stream("rp3", int(k[0])) for k in copies}})


def rp4():
    """RP4 (PE 9 item 7): paiec.pool.rp4_legacy in a fresh interpreter under the build's
    setting (no BLAS pin; archive-3 was built so), PYTHONHASHSEED=0: the excl-empty
    literal bundle equals archive-3's prior.json byte for byte, and each leave-one-out
    'fit' bundle equals paiec.prior.build."""
    env = {k: v for k, v in os.environ.items() if k not in BLAS_VARS and k != "PYTHONPATH"}
    env["PYTHONHASHSEED"] = "0"
    code = "import json; from paiec import pool; print(json.dumps(pool.rp4_legacy()))"
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr[-600:]}
    got = json.loads(r.stdout.strip().splitlines()[-1])
    got["ok"] = bool(got["literal_equals_archive_3"] and all(got["leave_one_out"].values()))
    return got


def rp5(ctx, pool, cache=None):
    """RP5: on RP5_SETS random exclusion sets (whole families of 1 to 7 benchmarks of E,
    from default_rng([TIMING_SEED, 5])), every candidate bundle passes from_json and
    HierPredictor, hyper.excluded equal to excl and included disjoint from it."""
    from paiec.hier import HierPredictor
    rng = np.random.default_rng([TIMING_SEED, 5])
    out = []
    for _ in range(RP5_SETS):
        k = int(rng.integers(1, 8))
        picked = sorted(pool.E[j] for j in rng.choice(len(pool.E), k, replace=False))
        excl = PL.excl_for(picked, pool.family_of, ctx.L)
        res = {}
        for X in TIE_ORDER:
            try:
                p, h, _ = PL.candidate_bundle(X, excl, ctx.free, ctx.fixed, ctx.L, ctx.U, mu0_m=ctx.mu0_m)
                p2, h2 = PR.from_json(json.loads(PL.to_bytes(p, h)))
                HierPredictor(p2, h2)
                res[X] = list(h2.excluded) == list(excl) and not set(h2.included) & set(excl)
            except Exception as exc:
                res[X] = f"{type(exc).__name__}: {exc}"
        out.append({"benchmarks": picked, "excl": list(excl), "ok": res})
    return {"sets": out, "ok": all(v is True for s in out for v in s["ok"].values())}


def stage_reproduce(a):
    state = prelock(a, "reproduce")
    parts = [p.strip() for p in a.parts.split(",")] if a.parts else ["digest", "rp1", "rp2", "rp3", "rp4", "rp5"]
    t0 = time.time()
    rec = dict(state.get("reproduce") or {})
    pool = PoolView.from_files(helm=False, root=a.root)
    comps = None
    if {"rp2", "rp3"} & set(parts):
        tc = os.path.join(EVAL_DIR, "timing_compositions.json")
        comps = {int(k): v for k, v in read_json(tc)["rp3"].items()} if os.path.exists(tc) else \
            {int(k): v for k, v in timing_compositions(pool, state["split"]["halves"],
                                                      DENSE["max_items"], root=a.root)["rp3"].items()}
    if "digest" in parts:
        try:
            d = rs_rows_digest()
            ok = d["digest"].startswith(RS_ROWS_DIGEST_PLANNING) and d["files"] == sum(RS_TASKS.values())
        except Exception as exc:
            d, ok = {"error": str(exc)}, False
        rec["rs_rows"] = d
        log_check("reproduce", "RS rows digest", ok, d, a.state)
    if "rp1" in parts:
        # amendment 4b: the legacy five's tables RP1 and the RS tasks read, as the lock pins them
        rec["legacy_data"] = {p: file_sha(os.path.join(ROOT, p)) for p in sorted(PINNED)
                              if p.split("/")[:2] in [["data", b] for b in PL.LEGACY]}
        rec["RP1"] = rp1()
        log_check("reproduce", "RP1", rec["RP1"]["ok"], {k: rec["RP1"][k] for k in ("max_abs_brier", "tol")},
                  a.state)
    if "rp2" in parts:
        rec["RP2"] = rp2(comps)
        log_check("reproduce", "RP2", rec["RP2"]["ok"], rec["RP2"], a.state)
    ctx = cache = None
    if {"rp3", "rp5"} & set(parts):
        ctx = FitsContext(a.fits_record)
        cache = BundleCache(ctx)
    if "rp3" in parts:
        rec["RP3"] = rp3(comps, cache, pool)
        log_check("reproduce", "RP3", rec["RP3"]["ok"], rec["RP3"]["runs"], a.state)
    if "rp4" in parts:
        rec["RP4"] = rp4()
        log_check("reproduce", "RP4", rec["RP4"]["ok"],
                  {k: rec["RP4"].get(k) for k in ("literal_equals_archive_3", "leave_one_out", "error")},
                  a.state)
    if "rp5" in parts:
        rec["RP5"] = rp5(ctx, pool, cache)
        log_check("reproduce", "RP5", rec["RP5"]["ok"], None, a.state)
    rec["ok"] = all((rec.get(k) or {}).get("ok", False) for k in ("RP1", "RP2", "RP3", "RP4", "RP5")) \
        and "rs_rows" in rec and "error" not in rec["rs_rows"]
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["provenance"] = provenance()
    save_stage("reproduce", rec, a.state)
    print(f"reproduce: {', '.join(parts)} done ({time.time() - t0:.0f}s)")


# --- stage: lock (PE 9 item 10) -----------------------------------------------------------

def amendment_section(path, heading=AMENDMENT_HEADING):
    """The amendment's text: from its heading line to the next '## Amendment' heading or
    the end of the file; None when the file has no such heading."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    start = next((k for k, x in enumerate(lines) if x.startswith(heading)), None)
    if start is None:
        return None
    end = next((k for k in range(start + 1, len(lines)) if lines[k].startswith("## Amendment")),
               len(lines))
    return "".join(lines[start:end])


def rule_block(text):
    """The fenced block after the line naming 'verbatim `RULE_TEXT`'."""
    k = text.index("verbatim `RULE_TEXT`")
    s = text.index("```\n", k) + 4
    return text[s:text.index("```", s)]


def quoted_design_sha(text):
    m = re.search(r"sha256\s+`([0-9a-f]{64})`", text)
    return m.group(1) if m else None


def lock_sources(dry=False):
    """(amendment path, its text, design path, whether the drafts were read): the
    committed files, or (dry only) the drafts under DRAFT_DIR."""
    plan, design = os.path.join(ROOT, AMENDMENT_FILE), os.path.join(ROOT, DESIGN_FILE)
    text = amendment_section(plan)
    if (text is None or not os.path.exists(design)) and dry:
        plan = os.path.join(ROOT, DRAFT_DIR, DRAFT_FILES["amendment"])
        design = os.path.join(ROOT, DRAFT_DIR, DRAFT_FILES["design"])
        text = amendment_section(plan)
        return plan, text, design, True
    return plan, text, design, False


def import_closure(paths, root=ROOT):
    """The static in-repository import closure of `paths` (every import and from statement
    at any depth, with the packages they pass through): sorted repo-relative paths."""
    def mods(path):
        tree = ast.parse(open(os.path.join(root, path), encoding="utf-8").read())
        out = set()
        pkg = os.path.dirname(path).replace(os.sep, ".")
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                out.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    up = pkg.split(".")[:len(pkg.split(".")) - (n.level - 1)] if pkg else []
                    base = ".".join(up + ([n.module] if n.module else []))
                out.add(base)
                out.update(f"{base}.{a.name}" for a in n.names)
        return out

    def resolve(m):
        parts = [p for p in m.split(".") if p]
        hits = []
        for k in range(1, len(parts) + 1):
            p = os.path.join(*parts[:k])
            if os.path.isfile(os.path.join(root, p + ".py")):
                hits.append(p + ".py")
            elif os.path.isfile(os.path.join(root, p, "__init__.py")):
                hits.append(os.path.join(p, "__init__.py"))
            else:
                break
        return hits

    seen, todo = set(), [os.path.normpath(p) for p in paths]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for m in sorted(mods(f)):
            for r in resolve(m):
                if r.split(os.sep)[0] in ("paiec", "experiments", "tests", "tools", "submission"):
                    todo.append(os.path.normpath(r))
    return sorted(x.replace(os.sep, "/") for x in seen)


def lock_files():
    """{path: sha256 or None (missing)} of every PINNED file."""
    out = {}
    for p in sorted(PINNED):
        full = os.path.join(ROOT, p)
        out[p] = file_sha(full) if os.path.isfile(full) else None
    return out


def client_commit():
    d = os.path.join(ROOT, "third_party", "paiec_baseline")
    try:
        return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


def frozen(source, names=EVAL_LOADER):
    """paiec.pool.frozen_ast with the removed definitions named explicitly: the module's
    AST (docstrings kept, comments ignored) without the top-level definitions and
    assignments `names` names. The lock passes the locked tuple, so a fix that widened
    paiec.pool.EVAL_LOADER would still be seen (V4)."""
    tree = ast.parse(source)

    def named(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None
    tree.body = [n for n in tree.body if named(n) not in names]
    return ast.dump(tree, include_attributes=False)


def pool_frozen_sha(names=EVAL_LOADER, path=None):
    with open(path or os.path.join(ROOT, "paiec", "pool.py"), encoding="utf-8") as f:
        return sha_bytes(frozen(f.read(), tuple(names)).encode())


def decision_sha(names=DECISION, path=None):
    """sha256 of decision_ast (amendment 4a) of this file, or of `path`, with `names`."""
    with open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        return sha_bytes(decision_ast(f.read(), tuple(names)).encode())


def decision_problems(lk, path=None):
    """[] or why this file's decision arithmetic is not the locked one (amendment 4a)."""
    if lk.get("decision_ast") is None:
        return ["the lock records no decision AST"]
    out = []
    if list(DECISION) != list(lk.get("decision") or []):
        out.append("DECISION is not the locked tuple")
    if decision_sha(lk["decision"], path) != lk["decision_ast"]:
        out.append("experiments/pool_eval.py changed its frozen decision arithmetic (amendment 4a)")
    return out


def lock_digest(rec):
    keys = ("files", "constants_sha256", "rule_sha256", "amendment_sha256", "amendment_4a_sha256",
            "design_sha256", "pool_frozen_ast", "eval_loader", "rs_rows", "mu0_m", "c2_literal",
            "bundles", "decision_ast", "degenerate", "amendment_4b_sha256", "frozen_objects",
            "amendment_4c_sha256", "hard_stops")
    return sha_bytes(json.dumps({k: rec.get(k) for k in keys}, sort_keys=True,
                                default=list).encode())


def stage_lock(a):
    state = load_state(a.state)
    if state.get("lock") and not a.dry:
        raise SystemExit("the lock exists; it is never rewritten (PE 9 item 11)")
    t0 = time.time()
    problems = []
    env = env_settings()
    if not settings_ok(env):
        problems.append(f"settings: PYTHONHASHSEED=0 and one BLAS thread required, got {env}")
    ok, why = checks_passed(state, ["split", "regimes", "fits-extra", "reproduce", "timing"])
    if not ok:
        problems += [f"check: {x}" for x in why]
    plan, text, design, drafts = lock_sources(a.dry)
    if text is None:
        problems.append(f"{plan}: no {AMENDMENT_HEADING} section")
        text = ""
    try:
        if rule_block(text) != RULE_TEXT:
            problems.append("RULE_TEXT differs from the amendment's block")
    except ValueError:
        problems.append("the amendment holds no RULE_TEXT block")
    text_4a = amendment_section(plan, AMENDMENT_4A_HEADING)
    if text_4a is None and not drafts:
        problems.append(f"{plan}: no {AMENDMENT_4A_HEADING} section")
    text_4b = amendment_section(plan, AMENDMENT_4B_HEADING)
    if text_4b is None and not drafts:
        problems.append(f"{plan}: no {AMENDMENT_4B_HEADING} section")
    text_4c = amendment_section(plan, AMENDMENT_4C_HEADING)
    if text_4c is None and not drafts:
        problems.append(f"{plan}: no {AMENDMENT_4C_HEADING} section")
    dsha = file_sha(design) if os.path.exists(design) else None
    if quoted_design_sha(text) != dsha:
        problems.append(f"the amendment quotes the design's sha256 {quoted_design_sha(text)}, "
                        f"the file's is {dsha}")
    problems += [f"constants: {x}" for x in constants_problems()]
    files = lock_files()
    for p, want in sorted(PINNED.items()):
        got = files.get(p)
        if got is None:
            problems.append(f"{p}: missing")
        elif want is not None and not got.startswith(want):
            problems.append(f"{p}: sha256 {got[:16]}, the design pins {want[:16]}")
    closure = import_closure(["experiments/pool_eval.py", "paiec/pool.py"])
    unpinned = [p for p in closure if p not in PINNED]
    if unpinned:
        problems.append(f"the import closure holds unpinned files: {unpinned}")
    commit = client_commit()
    if commit != STREAMING_CLIENT_COMMIT:
        problems.append(f"the streaming client's clone is at {commit}, not {STREAMING_CLIENT_COMMIT}")
    if tuple(PL.EVAL_LOADER) != EVAL_LOADER:
        problems.append("paiec.pool.EVAL_LOADER is not the tuple the constants name")
    try:
        rs = rs_rows_digest()
        rp1 = ((state.get("reproduce") or {}).get("rs_rows") or {}).get("digest")
        if rp1 != rs["digest"]:
            problems.append("the RS rows digest differs from RP1's")
    except Exception as exc:
        rs = {"error": str(exc)}
        problems.append(f"RS rows: {exc}")
    try:
        fr = read_json(a.fits_record)
        mu0_m, c2, bundles = fr["mu0_m"]["locked"], fr["C2"]["literal"], fr["bundles"]
        degen = {"threshold": DEGENERATE_ITEM_VAR, "benchmarks": fr["degenerate"],
                 "item_var": (fr.get("degenerate_rule") or {}).get("item_var")}
        if fr["degenerate"] != PL.degenerate(PL.load_fits()[0]):
            problems.append("the fits record's degenerate list is not paiec.pool.degenerate's")
        if tuple(fr["degenerate"]) != DEGENERATE_EXPECTED:
            problems.append(f"the degenerate fits are {fr['degenerate']}, amendment 4a names "
                            f"{list(DEGENERATE_EXPECTED)}")
        if (fr.get("degenerate_rule") or {}).get("threshold") != DEGENERATE_ITEM_VAR:
            problems.append("the fits record was not made with amendment 4a's threshold")
        for X in TIE_ORDER:
            p = f"data/pool_eval/bundles/archive_{X}.json"
            if files.get(p) != fr["bundles"][X]["sha256"]:
                problems.append(f"{p} differs from the fits record's")
        if files.get("data/pool_fits/index.json") != fr["fits_index_sha256"]:
            problems.append("the fits index differs from the fits record's")
        if fr["manifest"]["sha256"] != files.get("data/pool/manifest.json"):
            problems.append("the manifest differs from the fits record's")
    except Exception as exc:
        mu0_m = c2 = bundles = degen = None
        problems.append(f"fits record: {exc}")
    reg = state.get("regimes") or {}
    for H in HALVES:
        p = f"data/pool_eval_rows/{H}/_compositions.json"
        if files.get(p) != ((reg.get("compositions") or {}).get(H) or {}).get("sha256"):
            problems.append(f"{p} differs from the regimes stage's")
    if (state.get("split") or {}).get("levels", {}).get("sha256") != files.get("data/pool_eval/levels.json"):
        problems.append("levels.json differs from the split stage's")
    if reg.get("dense_max_items") != dense_cap(state):
        problems.append("the compositions' dense item cap is not the timing stage's")
    legacy = (state.get("reproduce") or {}).get("legacy_data") or {}
    if not legacy or any(files.get(p) != d for p, d in legacy.items()):
        problems.append("the legacy five's tables are not the ones RP1 read (amendment 4b)")
    # amendment 4b: every frozen object's digest, and the imports among them by name
    names = frozen_names()
    try:
        frozen_objects = frozen_digests(names)
    except TypeError as exc:
        frozen_objects = None
        problems.append(f"frozen objects: {exc}")
    g = globals()
    frozen_imports = {n: (g[n].__name__ if isinstance(g[n], type(os)) else
                          f"{g[n].__module__}.{g[n].__qualname__}")
                      for n in names if n in g and (isinstance(g[n], type(os)) or (
                          isinstance(g[n], type) and g[n].__module__ != __name__))}
    rec = {"utc": utcnow(), "files": files, "constants_sha256": constants_sha256(),
           "rule_sha256": sha_bytes(RULE_TEXT.encode("utf-8")),
           "amendment": {"path": rel(plan), "drafts": drafts},
           "amendment_sha256": sha_bytes(text.encode("utf-8")),
           "amendment_4a_sha256": None if text_4a is None else sha_bytes(text_4a.encode("utf-8")),
           "amendment_4b_sha256": None if text_4b is None else sha_bytes(text_4b.encode("utf-8")),
           "amendment_4c_sha256": None if text_4c is None else sha_bytes(text_4c.encode("utf-8")),
           "hard_stops": dict(HARD_STOPS),
           "frozen_objects": frozen_objects, "frozen_imports": frozen_imports,
           "design": rel(design),
           "design_sha256": dsha, "pool_frozen_ast": pool_frozen_sha(),
           "eval_loader": list(PL.EVAL_LOADER), "fix_scope": list(FIX_SCOPE),
           "decision": list(DECISION), "decision_ast": decision_sha(), "degenerate": degen,
           "pool_digests": {p: PL.source_digest(p) for p in ("module", "loader", "arithmetic")},
           "rs_rows": rs, "streaming_client": {"sha256": files.get(
               "third_party/paiec_baseline/tools/streaming_ingestion.py"), "commit": commit},
           "mu0_m": mu0_m, "c2_literal": c2, "bundles": bundles, "dense_max_items": dense_cap(state),
           "fits_record": {"path": rel(a.fits_record), "sha256": file_sha(a.fits_record)
                           if os.path.exists(a.fits_record) else None},
           "import_closure": closure, "env": env, "versions": {
               "python": platform.python_version(), "numpy": np.__version__},
           "hard_stop": HARD_STOPS["lock"], "problems": problems, "ok": not problems,
           "dry": bool(a.dry), "wall_s": round(time.time() - t0, 1)}
    rec["digest"] = lock_digest(rec)
    save_stage("lock_dry" if a.dry else "lock", rec, a.state)
    # a dry lock logs under its own stage, so that it never stands for the lock's check
    log_check("lock" if not a.dry else "lock-dry", "lock" if not a.dry else "lock (dry)",
              not problems, problems, a.state)
    if problems and not a.dry:
        # a failed lock is not a lock: keep the record for the failure, drop the key
        def drop(state):
            state["lock_failed"] = state.pop("lock")
        _locked_update(drop, a.state)
    print(f"lock{' (dry)' if a.dry else ''}: {'ok' if not problems else 'FAILED'}; "
          f"{len(problems)} problem(s)")
    for p in problems:
        print("  " + p)


def successors(state, committed=None):
    """{path: [sha256]} of the files in a fix's scope that in-scope fix records allow."""
    out = defaultdict(list)
    for f in state.get("fixes", []):
        if not f.get("in_scope"):
            continue
        if committed is not None and f.get("id") not in committed:
            continue
        for p, d in f.get("files", {}).items():
            if p in FIX_SCOPE:
                out[p].append(d["new"])
    return out


def committed_fix_ids(path=None):
    """The fix ids of results/pool_eval.json as committed (git show HEAD:...)."""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{rel(path or OUT)}"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            return set()
        return {f.get("id") for f in json.loads(r.stdout).get("fixes", [])}
    except Exception:
        return set()


def lock_check(state, a=None):
    """The lock check every post-lock stage runs: [] or the problems. A changed locked file
    passes only when it is in FIX_SCOPE, a committed fix record names its digest as a
    successor, and the fix kept the constants block (this file) or the frozen AST and
    EVAL_LOADER (paiec/pool.py)."""
    lk = state["lock"]
    state_path = getattr(a, "state", None) or OUT
    committed = committed_fix_ids(state_path) if os.path.abspath(state_path) == os.path.abspath(OUT) \
        else None
    succ = successors(state, committed)
    problems = []
    now = lock_files()
    for p, sha in sorted(lk["files"].items()):
        if now.get(p) == sha:
            continue
        if p in FIX_SCOPE and now.get(p) in succ.get(p, []):
            if p == "experiments/pool_eval.py" and constants_sha256() != lk["constants_sha256"]:
                problems.append("the constants block changed")
            if p == "experiments/pool_eval.py":
                problems += decision_problems(lk)
            if p == "paiec/pool.py" and (pool_frozen_sha(lk["eval_loader"]) != lk["pool_frozen_ast"]
                                        or list(PL.EVAL_LOADER) != lk["eval_loader"]):
                problems.append("paiec/pool.py changed outside EVAL_LOADER")
            continue
        problems.append(f"{p} changed since the lock")
    if constants_sha256() != lk["constants_sha256"]:
        problems.append("the constants block changed")
    if sha_bytes(RULE_TEXT.encode("utf-8")) != lk["rule_sha256"]:
        problems.append("RULE_TEXT changed")
    if not settings_ok():
        problems.append(f"settings: {env_settings()}")
    if client_commit() != lk["streaming_client"]["commit"]:
        problems.append("the streaming client's clone moved")
    # amendment 4b: every frozen object is the one the lock recorded (the stage's start),
    # and nothing outside the constants block binds or changes a block name
    problems += frozen_problems(lk)
    problems += block_binding_problems()
    return sorted(set(problems))


def frozen_check(a, stage, when="end"):
    """Amendment 4b: frozen_problems logged as a check of `stage` (at its end, before the
    stage writes its record, or at its start where the stage runs no lock check); a
    failure stops the stage."""
    problems = frozen_problems(load_state(a.state).get("lock") or {})
    log_check(stage, f"frozen objects at the {when}", not problems, problems, a.state)
    if problems:
        raise SystemExit(f"{stage}: a frozen object changed: {problems}")


def lock_commit_time(state, path=None):
    """(commit, UTC time) of the first commit of results/pool_eval.json whose lock digest is
    this lock's, or (None, None)."""
    rp = rel(path or OUT)
    r = subprocess.run(["git", "log", "--reverse", "--format=%H %cI", "--", rp], cwd=ROOT,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        c, t = line.split()
        s = subprocess.run(["git", "show", f"{c}:{rp}"], cwd=ROOT, capture_output=True, text=True)
        try:
            if (json.loads(s.stdout).get("lock") or {}).get("digest") == state["lock"]["digest"]:
                return c, datetime.fromisoformat(t).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return None, None


# --- stage: score -----------------------------------------------------------------------------

def plan_tasks(stage):
    """Every planned task of a stage, run-index-major (i, then regime), so that any stopping
    point leaves every regime with the same prefix (PE 5.4)."""
    if stage == "RS":
        top = max(RS_TASKS.values())
        return [(R, i) for i in range(top) for R in RS_REGIMES if i < RS_TASKS[R]]
    half = STAGE_HALF[stage]
    n = {R: N_RUNS[half][REGIMES[R]["shape"]] for R in REGIMES}
    return [(R, i) for i in range(max(n.values())) for R in REGIMES if i < n[R]]


def shard_tasks(tasks, k, n):
    """Shard k of n (1-based) by task index."""
    return [t for j, t in enumerate(tasks) if j % n == k - 1]


def row_path(stage, R, i, rows=None):
    return os.path.join(rows or ROWS, stage, R, f"{i}.json")


def stage_candidates(stage, state):
    if stage == "S" or stage == "RS":
        return list(TIE_ORDER)
    star = (state.get("selection") or {}).get("X_star")
    if not star:
        raise SystemExit(f"{stage}: no candidate carried")
    return [star] if stage == "C" else [X for X in TIE_ORDER if X != star]


class ScoreContext:
    def __init__(self, state, a, stage):
        self.stage = stage
        self.pool = PoolView.from_files(root=a.root)
        self.fits = FitsContext(a.fits_record, mu0_m=state["lock"]["mu0_m"])
        self.cache = BundleCache(self.fits, keep=16)
        self.a3 = a3_bundle()
        self.cands = stage_candidates(stage, state)
        self.lock_digest = state["lock"]["digest"]
        self.comps = {} if stage == "RS" else {STAGE_HALF[stage]: read_compositions(
            STAGE_HALF[stage], state)}
        self.shard = getattr(a, "shard", None)


def arm_row(t, dense):
    rows = t["rows"]
    out = {"b": None if rows is None else [x["brier"] for x in rows],
           "ece": None if rows is None else [x["ece"] for x in rows],
           "q0": None if rows is None else [x["q0"] for x in rows]}
    for k in ("calls", "cpu_sum", "cpu_max", "wall_sum", "wall_max", "arm_cpu", "arm_wall",
              "failures", "unconverged", "invalid"):
        out[k] = t[k]
    if dense:
        out["calls_cpu"] = [round(x, 7) for x in t["calls_cpu"]]
    return out


def score_pool_task(ctx, R, i):
    stage = ctx.stage
    half = STAGE_HALF[stage]
    c0, w0 = time.process_time(), time.perf_counter()
    entries = ctx.comps[half]["regimes"][TWINS.get(R, R)]["runs"][str(i)]
    run = load_run(entries, ctx.pool.root)
    if R in TWINS:
        run = twin_of(run)
    load_cpu = time.process_time() - c0
    excl = PL.excl_for(sorted({e[0] for e in entries}), ctx.pool.family_of, ctx.fits.L)
    c1 = time.process_time()
    arms, bundles, priors = {}, {}, {}
    if stage in ("S", "C"):
        arms[A3] = a3_arm()
        priors[A3] = ctx.a3[0]
    for X in ctx.cands:
        bd = ctx.cache.get(X, excl)
        arms[X] = ArmFactory({None: (bd["prior"], bd["hyper"])})
        bundles[X] = {"sha256": bd["sha256"], "key": bd["key"], "hyper": hyper_record(bd["hyper"])}
        priors[X] = bd["prior"]
    bundle_cpu = time.process_time() - c1
    c2 = time.process_time()
    slots, cps = checkpoints(run)
    cps_cpu = time.process_time() - c2
    dense = REGIMES[R]["shape"] == "dense"
    res = evaluate_timed(slots, cps, arms, f"{stage} {R} {i}", keep_calls=dense)
    meta = appearance_meta(ctx.pool, run, slots, priors)
    row = {"stage": stage, "half": half, "regime": R, "i": i, "seed": STAGE_SEED[stage],
           "twin_of": TWINS.get(R), "excl": list(excl), "meta": meta,
           "arms": {x: {**arm_row(t, dense), "bundle": bundles.get(x)} for x, t in res.items()},
           "task": {"load_cpu": load_cpu, "bundle_cpu": bundle_cpu, "checkpoints_cpu": cps_cpu,
                    "task_cpu": time.process_time() - c0, "task_wall": time.perf_counter() - w0,
                    "rss_gb": rss_gb(), "maxrss_gb": maxrss_gb(), "load_avg": load_avg(),
                    "lib": lib_digests(), "script": SCRIPT_SHA, "lock": ctx.lock_digest,
                    "pid": os.getpid(), "shard": ctx.shard, "utc": utcnow()}}
    return row


def score_rs_task(ctx, R, i):
    c0, w0 = time.process_time(), time.perf_counter()
    key, run, meta = rs_run(R, i)
    load_cpu = time.process_time() - c0
    ids = T.anon_parents(run)
    parents = sorted(set(ids.values()))
    c1 = time.process_time()
    arms, bundles = {}, {}
    for X in ctx.cands:
        bs = {par: ctx.cache.get(X, (par,)) for par in parents}
        arms[X] = ArmFactory({par: (bd["prior"], bd["hyper"]) for par, bd in bs.items()}, ids)
        bundles[X] = {par: {"sha256": bd["sha256"], "key": bd["key"]} for par, bd in bs.items()}
    bundle_cpu = time.process_time() - c1
    slots, cps = checkpoints(run)
    res = evaluate_timed(slots, cps, arms, f"RS {R} {i}")
    return {"stage": "RS", "regime": R, "i": i, "seed": RS_SEED, "key": key, "meta": meta,
            "arms": {x: {**arm_row(t, False), "bundle": bundles.get(x)} for x, t in res.items()},
            "task": {"load_cpu": load_cpu, "bundle_cpu": bundle_cpu,
                     "task_cpu": time.process_time() - c0, "task_wall": time.perf_counter() - w0,
                     "rss_gb": rss_gb(), "maxrss_gb": maxrss_gb(), "load_avg": load_avg(),
                     "lib": lib_digests(), "script": SCRIPT_SHA, "lock": ctx.lock_digest,
                     "pid": os.getpid(), "shard": ctx.shard, "utc": utcnow()}}


def stage_score(a):
    stage = a.stage
    k, n = (int(x) for x in a.shard.split("/"))
    state = postlock(a, f"score:{stage}")
    if not state.get("lock_time") and os.path.abspath(a.state) == os.path.abspath(OUT):
        c, t = lock_commit_time(state, a.state)
        save_stage("lock_time", {"commit": c, "utc": t, "hard_stop": HARD_STOPS["lock"],
                                 "in_time": bool(t and t <= HARD_STOPS["lock"])}, a.state)
        state = load_state(a.state)
    if stage in ("C", "C-extra"):
        if not state.get("selection"):
            raise SystemExit(f"{stage}: the selection record is not there")
        if os.path.abspath(a.state) == os.path.abspath(OUT) and not committed_has(a.state, "selection"):
            raise SystemExit(f"{stage}: the selection record is not committed")
    if stage == "C-extra" and (not state.get("gate") or (
            os.path.abspath(a.state) == os.path.abspath(OUT) and not committed_has(a.state, "gate"))):
        raise SystemExit("C-extra runs after the gate is committed")
    tasks = shard_tasks(plan_tasks(stage), k, n)
    todo = [t for t in tasks if not os.path.exists(row_path(stage, *t, rows=a.rows))]
    print(f"score {stage} shard {k}/{n}: {len(todo)} of {len(tasks)} tasks to do", flush=True)
    a.shard_label = a.shard
    ctx = ScoreContext(state, a, stage)
    t_start = time.time()
    for j, (R, i) in enumerate(todo):
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = score_rs_task(ctx, R, i) if stage == "RS" else score_pool_task(ctx, R, i)
            path = row_path(stage, R, i, rows=a.rows)
            write_json(path, row)
            if read_json(path) != json.loads(json.dumps(clean(row))):
                raise PoolEvalError(f"{path}: the row does not read back")
        except Exception as exc:
            log_check(f"score:{stage}", f"task {R} {i}", False, f"{type(exc).__name__}: {exc}", a.state)
            raise
        del row
        gc.collect()
        rss = rss_gb()
        print(f"  [{k}/{n}] {stage} {R} {i}: {time.time() - t0:.1f}s, rss {rss:.2f} GB, "
              f"{j + 1}/{len(todo)} in {time.time() - t_start:.0f}s", flush=True)
        if rss > RSS_GUARD_GB and j + 1 < len(todo):
            print(f"  [{k}/{n}] resident set {rss:.2f} GB > {RSS_GUARD_GB} GB: stopping (resumable)",
                  flush=True)
            # amendment 4b: a resumable stop ends this process's part of the stage too
            frozen_check(a, f"score:{stage}", f"resumable stop of shard {k}/{n}")
            sys.exit(RESUME_EXIT)
    frozen_check(a, f"score:{stage}", f"end of shard {k}/{n}")
    log_check(f"score:{stage}", f"shard {k}/{n}", True, {"tasks": len(tasks)}, a.state)


def committed_has(path, key):
    """Whether results/pool_eval.json as committed (HEAD) holds `key` equal to the file's."""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{rel(path)}"], cwd=ROOT, capture_output=True, text=True)
        return r.returncode == 0 and json.loads(r.stdout).get(key) == load_state(path).get(key)
    except Exception:
        return False


def stage_status(a):
    for stage in ("S", "RS", "C", "C-extra"):
        tasks = plan_tasks(stage)
        have = defaultdict(int)
        for R, i in tasks:
            have[R] += os.path.exists(row_path(stage, R, i, rows=a.rows))
        tot = defaultdict(int)
        for R, _ in tasks:
            tot[R] += 1
        print(f"{stage}: " + ", ".join(f"{R} {have[R]}/{tot[R]}" for R in tot))


# --- reading rows -----------------------------------------------------------------------------

def load_rows(stage, rows=None):
    """{regime: [row by run index]} of a stage's existing rows."""
    out = defaultdict(list)
    for R, i in plan_tasks(stage):
        p = row_path(stage, R, i, rows=rows)
        if os.path.exists(p):
            out[R].append(read_json(p))
    return dict(out)


def missing_tasks(stage, rows=None):
    return [t for t in plan_tasks(stage) if not os.path.exists(row_path(stage, *t, rows=rows))]


def pool_runs(rows, x, ref, ref_rows=None):
    """[(meta, Bx, Bref)] for X against ref over a regime's rows (ref from ref_rows, by run
    index, when given: C-extra against the C core's A3)."""
    refs = {r["i"]: r for r in (ref_rows or rows)}
    out = []
    for r in sorted(rows, key=lambda r: r["i"]):
        bx = r["arms"].get(x, {}).get("b")
        br = refs.get(r["i"], {}).get("arms", {}).get(ref, {}).get("b")
        if bx is None or br is None:
            raise PoolEvalError(f"{r['regime']} {r['i']}: {x} or {ref} not scored")
        out.append((r["meta"], bx, br))
    return out


def rs_runs(rows, x, stored):
    """RS rows of X against P1a's stored ship rows (meta from P1a's composition)."""
    out = []
    for r in sorted(rows, key=lambda r: r["i"]):
        bx = r["arms"].get(x, {}).get("b")
        br = stored[r["i"]]
        if bx is None:
            raise PoolEvalError(f"RS {r['regime']} {r['i']}: {x} not scored")
        meta = [{"family": m["parent"], "ident": m["subject"], "benchmark": m["bench"]} for m in r["meta"]]
        out.append((meta, bx, br))
    return out


def stored_ship(R):
    import experiments.regime_sensitivity as RS
    return {i: read_json(os.path.join(RS_ROWS, R, f"{i}.json"))["res"][RS.SHIP]["b"]
            for i in range(RS_TASKS[R])}


def invalid_count(rows_by_regime, arm):
    return sum(len(r["arms"].get(arm, {}).get("invalid") or []) for rs in rows_by_regime.values() for r in rs)


# --- stage: select (frozen whole: DECISION) ----------------------------------------------------

SELECT_STAGES = ("split", "regimes", "fits-extra", "reproduce", "timing", "lock", "score:S",
                 "score:RS", "select")


def selection_table(S, RSr, stored, checks_ok):
    """The per-candidate inputs of apply_selection, from S's and the RS guard's rows."""
    table = {A3: {"invalid": invalid_count(S, A3)}}
    pool_rows = [r for rs in S.values() for r in rs]
    form_rows = [r for R, rs in S.items() if REGIMES[R]["shape"] == "formative" for r in rs]
    rs_rows = [r for rs in RSr.values() for r in rs]
    stats = {}
    def scored(r, arm):
        """The arm is scored on this row, or excused by its invalid output (E4, B3)."""
        a = r["arms"].get(arm) or {}
        return a.get("b") is not None or bool(a.get("invalid"))

    for X in TIE_ORDER:
        complete = all(scored(r, X) and scored(r, A3) for r in pool_rows) and \
            all(scored(r, X) for r in rs_rows)
        D, RSD = {}, {}
        for R, rs in S.items():
            try:
                st = diff_stats(Paired(pool_runs(rs, X, A3)))
                D[R] = st["D"]
                stats.setdefault(X, {})[R] = st
            except PoolEvalError:
                D[R] = None
        for R, rs in RSr.items():
            try:
                RSD[R] = D_of(Paired(rs_runs(rs, X, stored[R])))
            except PoolEvalError:
                RSD[R] = None
        lx, la = latency(pool_rows, X), latency(pool_rows, A3)
        mx = max(latency(form_rows, X)["max_s"], latency(rs_rows, X)["max_s"])
        table[X] = {"D": D, "RS": RSD, "latency": {"ratio": ratio(lx, la), "max_s": mx,
                                                   "mean_s": lx["mean_s"], "A3_mean_s": la["mean_s"]},
                    "complete": complete, "checks_ok": checks_ok,
                    "invalid": invalid_count(S, X) + invalid_count(RSr, X)}
    return table, stats


def reselection(prev, rec):
    """A re-run selection (after a recorded fix) that names another X* than the committed
    one gives "archive-3 stays" (PE 9 item 11, V8)."""
    if prev and prev.get("X_star") != rec["X_star"]:
        rec = dict(rec)
        rec["reselected"] = {"before": prev.get("X_star"), "now": rec["X_star"],
                             "outcome": "archive-3 stays (a re-run selection named another X*)"}
        rec["outcome"] = rec["reselected"]["outcome"]
        rec["X_star"] = None
    return rec


def stage_select(a):
    state = postlock(a, "select")
    miss = missing_tasks("S", a.rows) + [("RS",) + t for t in missing_tasks("RS", a.rows)]
    if miss:
        raise SystemExit(f"select: {len(miss)} planned S and RS tasks are not scored, e.g. {miss[:3]}")
    d = rs_rows_digest()
    ok_d = d["digest"] == state["lock"]["rs_rows"]["digest"]
    log_check("select", "RS rows digest", ok_d, d, a.state)
    if not ok_d:
        raise SystemExit("select: the RS rows changed since the lock")
    stored = {R: stored_ship(R) for R in RS_REGIMES}
    S, RSr = load_rows("S", a.rows), load_rows("RS", a.rows)
    state = load_state(a.state)
    checks_ok, problems = checks_passed(state, SELECT_STAGES)
    table, stats = selection_table(S, RSr, stored, checks_ok)
    sel = apply_selection(table)
    se = {X: {R: (stats.get(X, {}).get(R) or {}).get("SEf") for R in REGIMES} for X in TIE_ORDER}
    rec = {**sel, "table": table, "SEf": se, "check_problems": problems,
           "power_of_a_at_S_SEf": {X: {R: power(se[X].get(R)) for R in GAIN_REGIMES} for X in TIE_ORDER},
           "stats": stats, "utc": utcnow(), "rule_text_sha256": sha_bytes(RULE_TEXT.encode())}
    rec = reselection(state.get("selection"), rec)
    frozen_check(a, "select")
    save_stage("selection", rec, a.state)
    print(f"select: {rec['outcome']}")


# --- stages: summarise and gate (stage_summarise and stage_gate frozen whole: DECISION) -----

def cuts(P, floor_active=None):
    """Descriptive cuts of one paired table (PE 6): dated and undated, recent, subject kind,
    flags, multiple-choice, the floor-active share, family members in E, HELM-hosted,
    response kind, level tercile."""
    if not len(P.d):
        return {}
    m = P.metas
    fam_n = defaultdict(set)
    for x in m:
        fam_n[x["family"]].add(x["benchmark"])
    lv = np.array([x["logit"] for x in m])
    q = np.quantile(lv, [1 / 3, 2 / 3]) if len(lv) > 2 else [lv.min(), lv.max()]
    defs = {
        "dated": [x["dated"] for x in m], "undated": [not x["dated"] for x in m],
        "released_2025_or_later": [bool(x["release"]) and x["release"] >= "2025-01-01" for x in m],
        "named_with_row": [x["named"] and any(x["id_row"].values()) for x in m],
        "named_without_row": [x["named"] and not any(x["id_row"].values()) for x in m],
        "unnamed": [not x["named"] for x in m], "unflagged": [not x["flagged"] for x in m],
        "mcq": [x["mcq"] for x in m], "not_mcq": [not x["mcq"] for x in m],
        "family_members": [len(fam_n[x["family"]]) > 1 for x in m],
        "singletons": [len(fam_n[x["family"]]) == 1 for x in m],
        "helm": [x["helm"] for x in m], "not_helm": [not x["helm"] for x in m],
        "mixed": [x["kind"] == "mixed" for x in m], "binary": [x["kind"] == "binary" for x in m],
        "tercile_low": (lv <= q[0]).tolist(), "tercile_mid": ((lv > q[0]) & (lv <= q[1])).tolist(),
        "tercile_high": (lv > q[1]).tolist()}
    if floor_active is not None:
        defs["floor_active"] = [x["benchmark"] in floor_active for x in m]
        defs["floor_inactive"] = [x["benchmark"] not in floor_active for x in m]
    out = {}
    for k, mask in defs.items():
        if any(mask):
            s = P.subset(mask)
            out[k] = {"D": D_of(s), "appearances": int(len(s.d)), "families": len(set(s.fam))}
    return out


def arm_summary(rows, arm):
    alc, q0, rate, ece0, fb, inv = [], [], [], [], 0, 0
    for r in rows:
        a = r["arms"].get(arm)
        if a is None:
            continue
        fb += a["failures"]
        inv += len(a["invalid"])
        if a["b"] is None:
            continue
        B = np.asarray(a["b"], float)
        alc.append(float(np.mean(B @ W6)))
        q0 += a["q0"]
        rate += [x["rate"] for x in r["meta"]]
        ece0 += [e[0] for e in a["ece"]]
    return {"runs": len(alc), "ALC": float(np.mean(alc)) if alc else None,
            "q0": float(np.mean(q0)) if q0 else None, "rate": float(np.mean(rate)) if rate else None,
            "ece0": float(np.mean(ece0)) if ece0 else None, "fallbacks": fb, "invalid": inv}


def floor_active_set(fits_record):
    """E's benchmarks on which the shipped floor is positive for at least half of the items
    read (the fits stage's floor_share; PE 3.7)."""
    try:
        fs = read_json(fits_record).get("floor_share") or {}
    except Exception:
        return None
    return {b for b, v in fs.items() if (v.get("share") or 0) >= 0.5}


def run_descriptives(rs, arms):
    """Per run of a regime: v (the mean of rate (1 - rate) over its appearances) and each
    arm's run ALC - v; the bundles' levels per run (PE 6, the runs; 4.3's recomputations)."""
    v = [float(np.mean([m["v"] for m in r["meta"]])) for r in rs]
    out = {"v": {"mean": float(np.mean(v)), "sd": float(np.std(v))} if v else None}
    for x in arms:
        alc = [float(np.mean(np.asarray(r["arms"][x]["b"], float) @ W6)) for r in rs
               if (r["arms"].get(x) or {}).get("b") is not None]
        if len(alc) == len(v) and v:
            d = np.array(alc) - np.array(v)
            out[f"{x}: ALC - v"] = {"mean": float(d.mean()), "sd": float(d.std())}
        lev = [(r["arms"][x].get("bundle") or {}).get("hyper") for r in rs if r["arms"].get(x)]
        lev = [h for h in lev if h]
        if lev:
            out[f"{x}: bundle level"] = {k: {"mean": float(np.mean([h[k] for h in lev])),
                                             "sd": float(np.std([h[k] for h in lev]))}
                                         for k in ("mu0", "sigma_mu", "T")}
    return out


def reweighting_check(C, star, state, pool):
    """P-FLAT's appearances reweighted to each tilted target by benchmark level (the C
    half's sampler weights at the regime's knobs), against the tilted regime's own D
    (PE 6; descriptive)."""
    flat = C.get("P-FLAT")
    if not flat:
        return None
    P = Paired(pool_runs(flat, star, A3))
    v, n = defaultdict(float), defaultdict(float)
    for b, w, d in zip(P.bench, P.w.tolist(), P.d.tolist()):
        v[b] += w * d
        n[b] += w
    H = Half(pool, "C", state["split"]["halves"]["C"])
    out = {}
    for R in REGIMES:
        if R in TWINS or not REGIMES[R].get("target"):
            continue
        knob = state["regimes"]["regimes"]["C"][R]["knob"]
        ok, w = H.weights(comps_of(R, None if knob is None else tuple(knob)), REGIMES[R]["shape"])
        wb = {H.bs[j]: x for j, x in zip(ok.tolist(), w.tolist()) if H.bs[j] in v}
        tot = sum(wb.values())
        rw = sum(wb[b] * v[b] / n[b] for b in wb) / tot if tot else None
        own = None
        if C.get(R):
            own = D_of(Paired(pool_runs(C[R], star, A3)))
        out[R] = {"reweighted_P_FLAT": rw, "regime_D": own}
    return out


def gate_inputs(C, star, missing):
    """The gate's inputs from the C core's rows (RULE_TEXT (a) to (d), (f), (g)): per regime
    diff_stats of X* against A3 (D None, with the error, when an arm is not scored); the
    latency of (f), pooled over C's tasks (mean CPU per call over A3's, the largest
    formative-size call, the p99 and maximum of X*'s dense calls); whether every planned C
    task is scored for A3 and X* (none of `missing` and every row's two arms); the invalid
    outputs of each arm."""
    stats = {}
    for R, rs in C.items():
        try:
            stats[R] = diff_stats(Paired(pool_runs(rs, star, A3)))
        except PoolEvalError as exc:
            stats[R] = {"D": None, "error": str(exc)}
    complete = not missing and all(r["arms"].get(x, {}).get("b") is not None for rs in C.values()
                                   for r in rs for x in (A3, star))
    rows = [r for rs in C.values() for r in rs]
    form = [r for R, rs in C.items() if REGIMES[R]["shape"] == "formative" for r in rs]
    dense_calls = [c for R, rs in C.items() if REGIMES[R]["shape"] == "dense" for r in rs
                   for c in (r["arms"].get(star, {}).get("calls_cpu") or [])]
    lx, la = latency(rows, star), latency(rows, A3)
    lat = {"ratio": ratio(lx, la), "max_s": latency(form, star)["max_s"],
           "dense_p99_s": p99(dense_calls) if dense_calls else None,
           "dense_max_s": max(dense_calls) if dense_calls else None, "dense_calls": len(dense_calls),
           "mean_s": lx["mean_s"], "A3_mean_s": la["mean_s"]}
    return {"stats": stats, "latency": lat, "complete": complete,
            "invalid_X": invalid_count(C, star), "invalid_A3": invalid_count(C, A3)}


def gate_table(summ, selection, checks_ok):
    """apply_gate's input: the summary's gate inputs (gate_inputs), X*'s (E1) from the
    committed selection and (g)'s every-check-passed."""
    star = summ["X_star"]
    return {"stats": summ["stats"], "E1": selection["candidates"][star]["holds"]["E1"],
            "latency": summ["latency"], "checks_ok": checks_ok, "complete": summ["complete"],
            "invalid_X": summ["invalid_X"], "invalid_A3": summ["invalid_A3"]}


def stage_summarise(a):
    state = postlock(a, "summarise")
    star = (state.get("selection") or {}).get("X_star")
    if not star:
        raise SystemExit("summarise: no candidate carried")
    miss = missing_tasks("C", a.rows)
    C = load_rows("C", a.rows)
    gi = gate_inputs(C, star, miss)
    cut, arms, runs = {}, {}, {}
    fl = floor_active_set(a.fits_record)
    for R, rs in C.items():
        if "error" not in gi["stats"][R]:
            cut[R] = cuts(Paired(pool_runs(rs, star, A3)), fl)
        arms[R] = {x: arm_summary(rs, x) for x in (A3, star)}
        runs[R] = run_descriptives(rs, (A3, star))
    try:
        rw = reweighting_check(C, star, state, PoolView.from_files(helm=False, root=a.root))
    except Exception as exc:
        rw = {"error": f"{type(exc).__name__}: {exc}"}
    stats, lat, complete = gi["stats"], gi["latency"], gi["complete"]
    extra = {}
    E = load_rows("C-extra", a.rows)
    for R, rs in E.items():
        for X in TIE_ORDER:
            if X != star and rs and X in rs[0]["arms"]:
                try:
                    extra.setdefault(X, {})[R] = D_of(Paired(pool_runs(rs, X, A3, C.get(R))))
                except PoolEvalError:
                    pass
    rec = {"X_star": star, "stats": stats, "cuts": cut, "arms": arms, "runs": runs,
           "reweighting": rw, "latency": lat,
           "complete": complete, "missing": len(miss), "invalid_X": gi["invalid_X"],
           "invalid_A3": gi["invalid_A3"], "C_extra": extra, "utc": utcnow()}
    frozen_check(a, "summarise")
    save_stage("summary", rec, a.state)
    print(f"summarise: {len(C)} regimes; complete {complete}")


def stage_gate(a):
    state = postlock(a, "gate")
    summ = state.get("summary")
    if not summ:
        raise SystemExit("gate: run summarise first")
    prev = state.get("gate")
    if prev and not any(f.get("utc", "") > prev.get("utc", "") for f in state.get("fixes", [])):
        raise SystemExit("gate: read once; it is re-run only after a recorded fix")
    star = summ["X_star"]
    checks_ok, problems = checks_passed(load_state(a.state), GATE_STAGES)
    res = apply_gate(gate_table(summ, state["selection"], checks_ok))
    res.update(X_star=star, check_problems=problems, utc=utcnow())
    frozen_check(a, "gate")
    save_stage("gate", res, a.state)
    if res["pass"]:
        lk = state["lock"]
        write_json(POOL_BUILD, {"candidate": star, "bundle_sha256": lk["bundles"][star]["sha256"],
                                "level": lk["bundles"][star]["level"], "lock": lk["digest"]},
                   compact=False)
    print(f"gate: {res['outcome']}")


# --- stage: record-fix (PE 9 item 11; N5, V4, V8) -------------------------------------------

def parse_task(s):
    stage, R, i = s.split(":")
    return stage, R, int(i)


def verification_set(rows=None):
    """Run 0 and the last scored run of every regime of every stage with rows (V8)."""
    out = []
    for stage in ("S", "RS", "C", "C-extra"):
        done = defaultdict(list)
        for R, i in plan_tasks(stage):
            if os.path.exists(row_path(stage, R, i, rows=rows)):
                done[R].append(i)
        for R, ii in sorted(done.items()):
            out += sorted({(stage, R, min(ii)), (stage, R, max(ii))})
    return out


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if k not in TIMING_FIELDS}
    if isinstance(x, list):
        return [strip_timing(v) for v in x]
    return x


def fix_scope(state):
    """(in scope, the changed files with old and new digests, why not)."""
    lk = state["lock"]
    now = lock_files()
    last = {}
    for f in state.get("fixes", []):
        for p, d in f.get("files", {}).items():
            last[p] = d["new"]
    changed = {p: {"old": last.get(p, sha), "new": now.get(p)} for p, sha in lk["files"].items()
               if now.get(p) != last.get(p, sha)}
    why = []
    for p in changed:
        if p not in FIX_SCOPE:
            why.append(f"{p} is outside a fix's scope")
    if constants_sha256() != lk["constants_sha256"]:
        why.append("the constants block changed")
    if "paiec/pool.py" in changed and (pool_frozen_sha(lk["eval_loader"]) != lk["pool_frozen_ast"]
                                      or list(PL.EVAL_LOADER) != lk["eval_loader"]):
        why.append("paiec/pool.py changed outside the EVAL_LOADER definitions")
    if "experiments/pool_eval.py" in changed:
        why += [f"{x}: outside the fix scope" for x in decision_problems(lk)]
    # amendment 4b: the frozen objects as the lock recorded them, and no block name bound
    # or changed outside the block
    why += [f"{x}: outside the fix scope" for x in frozen_problems(lk) + block_binding_problems()]
    for p in ("tests/test_pool.py", "tests/test_pool_eval.py"):
        if now.get(p) != lk["files"].get(p):
            why.append(f"{p} changed: the locked tests must run unchanged")
    return not why, changed, why


def run_locked_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_pool.py",
                        "tests/test_pool_eval.py"], cwd=ROOT, capture_output=True, text=True)
    return {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-1:] if r.stdout else []}


def stage_record_fix(a):
    state = load_state(a.state)
    if not state.get("lock"):
        raise SystemExit("record-fix: no lock")
    if a.verify:
        return verify_fix(a, state)
    # amendment 4b: the frozen objects at the stage's start (fix_scope's why holds them too)
    # and again at its end, before any row moves or the record is written
    start = frozen_problems(state["lock"])
    ok, changed, why = fix_scope(state)
    tests = run_locked_tests() if ok else None
    if ok and tests["returncode"] != 0:
        ok, why = False, why + ["the locked tests do not pass"]
    end = frozen_problems(state["lock"])
    if end:
        ok, why = False, why + [f"{x}: at the end of record-fix" for x in end]
    failed = [parse_task(s) for s in (a.failed or [])]
    ver = [t for t in verification_set(a.rows) if t not in failed]
    fid = f"fix-{len(state.get('fixes', [])) + 1}-{utcnow()}"
    moved = []
    if ok:
        for stage, R, i in failed + ver:
            p = row_path(stage, R, i, rows=a.rows)
            if os.path.exists(p):
                q = os.path.join(a.rows or ROWS, "_superseded", fid, stage, R, f"{i}.json")
                os.makedirs(os.path.dirname(q), exist_ok=True)
                os.replace(p, q)
                moved.append([stage, R, i])
    rec = {"id": fid, "utc": utcnow(), "cause": a.cause, "files": changed, "in_scope": ok,
           "why_not": why, "tests": tests, "resolves": [x.split(":", 1) for x in (a.resolves or [])],
           "rescore": {"failed": [list(t) for t in failed], "verification": [list(t) for t in ver],
                       "moved": moved}, "frozen_problems": {"start": start, "end": end},
           "outcome": None if ok else "archive-3 stays for the candidates the failure affects"}

    def put(s):
        s.setdefault("fixes", []).append(rec)
    _locked_update(put, a.state)
    print(f"record-fix {fid}: {'in scope' if ok else 'OUT OF SCOPE: ' + '; '.join(why)}; "
          f"commit the record with the diff before any re-score")


def verify_fix(a, state):
    """After the re-score: every verification row must equal its superseded copy outside
    the timing and resource fields, or every task of its stage is re-scored (V8)."""
    fix = next(f for f in state.get("fixes", []) if f["id"] == a.verify)
    frozen_check(a, "record-fix", f"start of verify {fix['id']}")      # amendment 4b
    bad = defaultdict(list)
    for stage, R, i in fix["rescore"]["verification"]:
        old = os.path.join(a.rows or ROWS, "_superseded", fix["id"], stage, R, f"{i}.json")
        new = row_path(stage, R, i, rows=a.rows)
        if not os.path.exists(new):
            raise SystemExit(f"{stage} {R} {i} is not re-scored yet")
        if strip_timing(read_json(old)) != strip_timing(read_json(new)):
            bad[stage].append([R, i])
    # amendment 4b: the end check, before any row moves or the record is written
    frozen_check(a, "record-fix", f"end of verify {fix['id']}")
    for stage in bad:
        for R, i in plan_tasks(stage):
            p = row_path(stage, R, i, rows=a.rows)
            if os.path.exists(p):
                q = os.path.join(a.rows or ROWS, "_superseded", fix["id"] + "-stage", stage, R, f"{i}.json")
                os.makedirs(os.path.dirname(q), exist_ok=True)
                os.replace(p, q)

    def put(s):
        for f in s["fixes"]:
            if f["id"] == fix["id"]:
                f["verified"] = {"utc": utcnow(), "differ": dict(bad),
                                 "whole_stage_rescore": sorted(bad)}
    _locked_update(put, a.state)
    print(f"verify {fix['id']}: {'reproduced' if not bad else 'differs: re-score ' + ', '.join(sorted(bad))}")


# --- stage: describe-hidden (PE 11 item 4) -------------------------------------------------

def stage_describe_hidden(a):
    state = load_state(a.state)
    if not state.get("gate"):
        raise SystemExit("describe-hidden runs after the gate")
    if os.path.abspath(a.state) == os.path.abspath(OUT) and not committed_has(a.state, "gate"):
        raise SystemExit("describe-hidden: the gate is not committed")
    if state.get("describe_hidden"):
        raise SystemExit("describe-hidden: results/hidden_runs.json is read once")
    frozen_check(a, "describe-hidden", "start")
    with _OPEN(a.hidden or HIDDEN) as f:
        hidden = json.load(f)["decomposition"]
    C = load_rows("C", a.rows)
    out = {}
    for R in REGIMES:
        if REGIMES[R]["shape"] != "formative" or R not in C:
            continue
        rs = [r for r in C[R] if len(r["meta"]) in (8, 9)]
        v = np.array([np.mean([m["v"] for m in r["meta"]]) for r in rs])
        av = np.array([np.mean(np.asarray(r["arms"][A3]["b"], float) @ W6) - np.mean([m["v"] for m in r["meta"]])
                       for r in rs if r["arms"][A3]["b"] is not None])
        out[R] = {}
        for k in ("run1", "run2", "run3"):
            h = hidden.get(k) or {}
            x = h.get("v")
            e = {"pairs": h.get("pairs"), "v": x}
            if x is not None and len(v) > 1:
                e.update(z=float((x - v.mean()) / v.std(ddof=1)), pct=float(np.mean(v <= x)))
            y = h.get("alc_minus_v")
            if y is not None and k != "run1" and len(av) > 1:
                e.update(alc_minus_v=y, z_A3=float((y - av.mean()) / av.std(ddof=1)),
                         pct_A3=float(np.mean(av <= y)))
            out[R][k] = e
    frozen_check(a, "describe-hidden")
    save_stage("describe_hidden", {"placement": out, "utc": utcnow(),
                                   "note": "descriptive, after the gate is committed"}, a.state)
    print("describe-hidden: done")


# --- the command line ---------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    def add(name, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("--state", default=OUT, help="the state file (results/pool_eval.json)")
        p.add_argument("--rows", default=ROWS)
        p.add_argument("--fits-record", default=FITS_RECORD)
        p.add_argument("--root", default=None, help="the pool (data/pool)")
        p.add_argument("--jobs", type=int, default=1)
        return p

    for name in ("split", "fits", "fits-extra", "status", "select", "summarise", "gate"):
        add(name)
    add("timing").add_argument("--probe-heaviest", action="store_true",
                               help="descriptive: load the heaviest planned C task of each shape")
    add("regimes").add_argument("--centres-only", action="store_true",
                                help="recompute the B0-centre part on the written compositions")
    add("reproduce").add_argument("--parts", default=None, help="digest,rp1,rp2,rp3,rp4,rp5")
    p = add("lock")
    p.add_argument("--dry", action="store_true", help="record under lock_dry; nothing is locked")
    p = add("score")
    p.add_argument("--stage", dest="score_stage", required=True, choices=("S", "RS", "C", "C-extra"))
    p.add_argument("--shard", default="1/1")
    p = add("record-fix")
    p.add_argument("--cause", default="")
    p.add_argument("--failed", nargs="*", help="stage:regime:i")
    p.add_argument("--resolves", nargs="*", help="stage:check name")
    p.add_argument("--verify", default=None, help="a fix id: compare its verification rows")
    p = add("describe-hidden")
    p.add_argument("--hidden", default=None)
    sub.add_parser("constants-digest")
    a = ap.parse_args(argv)
    if a.stage == "constants-digest":
        print(constants_sha256())
        return
    if getattr(a, "jobs", 1) > 4:
        raise SystemExit("at most 4 processes")
    if a.stage == "score":
        a.stage = a.score_stage
        return stage_score(a)
    fn = {"split": stage_split, "regimes": stage_regimes, "fits": stage_fits, "fits-extra": stage_fits_extra,
          "reproduce": stage_reproduce, "timing": stage_timing, "lock": stage_lock,
          "status": stage_status, "select": stage_select, "summarise": stage_summarise,
          "gate": stage_gate, "record-fix": stage_record_fix,
          "describe-hidden": stage_describe_hidden}[a.stage]
    return fn(a)


if __name__ == "__main__":
    try:
        main()
    except (PoolEvalError, PL.PoolError) as exc:
        sys.exit(f"POOL EVAL CHECK FAILED: {exc}")
