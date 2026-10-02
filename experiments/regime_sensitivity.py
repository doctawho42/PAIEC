"""P1a: the shipped level under the feedback's level readings (review W3, Q3,
P1.9, P1.10, P1.16; docs/plans/p1a_regime_sensitivity.md).

The plan and its decision rule were fixed before any run of the scoring seed
was drawn or scored: RULE_TEXT and every constant from "the lock" down to ROWS
below are the planner's, byte for byte (CONSTANTS_SHA256 pins that block, and
tests/test_regime_sensitivity.py checks it). The stages after it were written
against those constants, unchanged.

The question. The shipped level (submission/model.py LEVEL: mu0 -2.5,
sigma_mu 2.5, attr_scale 0.5) was chosen in the tuned test-like regime
(realised pair logit -1.29, sd 1.70) under a public guard. The preregistered
pooled reading of formative runs 1 and 2 puts the hidden levels at about -0.7
(sd 1.8), the audit's reading of run 1 at -1.1 (sd 2.0). This scores the
shipped config, its neighbours (-3.0/0.25; sigma_mu 3.5 and 5.0), the
empirical-Bayes level (fitted, and adaptive), the legacy Predictor, the
smoothed mean (plain and level-calibrated) and hier without its subject prior
in regimes set to those readings, a two-component mixture, the tuned and
untilted regimes and public R1, on fresh seeds, paired, and reads the result
through RULE_TEXT once.

Stages (the plan, section 9), each a subcommand; every stage after `lock`
re-runs the lock check and refuses to run when it fails:
  lock       RULE_TEXT against RULE_SHA256, the plan file against PLAN_SHA256
             (and that it quotes RULE_TEXT verbatim), LEVEL against
             submission/model.py, the constants block against CONSTANTS_SHA256
  smcal      the calibrated smoothed mean's (n0, m0) by LEVEL's selection rule
             on LEVEL's own runs (old seeds 2 and 0 only), with the check of
             its (4, 0.5) column against level_calibration.json's rows
  regimes    realised levels of the fixed knobs on CAL_SEED (CAL_DRAWS draws)
             and on the scoring seed's planned runs (run composition only, no
             predictor), the realisation checks, and every planned run's
             composition (the score stage checks its runs against it); --grid
             re-runs the knob search over KNOB_GRID on CAL_SEED and records
             whether its argmin is the fixed knobs (they stay fixed either way)
  reproduce  ship, legacy and smooth against stored rows on REPRO's runs, and
             every config's time per run there and on two old public runs
  score      every planned (regime, run) task, all CONFIGS on identical
             checkpoints (level_calibration.checkpoints / evaluate, a fresh
             instance per checkpoint, released after it), --shard k/2,
             resumable; one row file per task in ROWS; a worker exits with
             code 75 (resumable) once its resident set passes RSS_GUARD_GB
  summarise  statistics (level_calibration.Boot; ship_confirm.Block's per-
             parent and parent-level tables), the TUNED consistency check, and
             RULE_TEXT applied once, to the full planned set -> OUT
  status     rows scored so far, per regime

Run:
  python experiments/regime_sensitivity.py lock
  python experiments/regime_sensitivity.py smcal        # about 5 min
  python experiments/regime_sensitivity.py regimes --grid
  python experiments/regime_sensitivity.py reproduce    # about 7 min
  for k in 0 1; do (while :; do python experiments/regime_sensitivity.py score \\
      --shard $k/2; [ $? -eq 75 ] || break; done) & done; wait
  python experiments/regime_sensitivity.py summarise

Output: results/regime_sensitivity.json (OUT); rows in
data/regime_sensitivity_rows/ (ROWS, gitignored).

Library change the plan needed first: paiec.testlike.Regime.level_mix (a
Gaussian-mixture tilt target; plan section 10). MIXTURE below uses it.
"""
# >>> after review (2026-10-01): added after the study was scored and reviewed.
# Every edit made to this file after scoring sits between a line opening ">>> after
# review" and the next line opening "<<< after review". Cutting those blocks out gives
# back, byte for byte, the script every scored row records (SCORED_SCRIPT_SHA256;
# tests/test_regime_sensitivity.py checks it). The blocks add one stage, `review`, which
# reads the stages, rows, summary and rule without changing them, re-runs the
# deterministic pre-scoring stages into a scratch directory, measures what the review
# asked for, and writes its parts under "review" in OUT; and they let the lock accept the
# plan's appended "Amendments after review" section while still pinning the fixed text.
#   python experiments/regime_sensitivity.py review --parts rerun --work-dir DIR
#   python experiments/regime_sensitivity.py review --parts smcal_guard,rescore
#   python experiments/regime_sensitivity.py review --evidence-dir SCRATCH/p1a
# <<< after review
import ast
import hashlib
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- the lock ---------------------------------------------------------------------------

#: when this plan and rule were fixed (UTC), and the commit they were written at
#: (the files themselves were new and uncommitted then)
FIXED_AT_UTC = "2026-10-01T14:56:59Z"
HEAD_AT_FIXING = "380fecf"
PLAN_PATH = "docs/plans/p1a_regime_sensitivity.md"
#: sha256 of the plan file as fixed (the plan quotes RULE_TEXT and RULE_SHA256,
#: not its own hash)
PLAN_SHA256 = "0a857944a8a6842e673480b934891ac6f5046125be91bbc710634de818bd4365"
#: sha256 of RULE_TEXT encoded as UTF-8 (the text is ASCII)
RULE_SHA256 = "47e4735cf7ab0192e2c375054911221485e19ec71775bea8390b0e60287dc58e"

RULE_TEXT = """P1a DECISION RULE: regime sensitivity of the shipped level
(docs/plans/p1a_regime_sensitivity.md; experiments/regime_sensitivity.py).
Fixed before any run of the scoring seed was drawn or scored.

Terms
  SHIP      the shipped configuration: paiec.hier.HierPredictor with
            submission/model.py LEVEL (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5)
            over prior.build fitted without the target's parent benchmark.
  X         any config of CONFIGS other than SHIP.
  regimes   TUNED, READING, AUDIT, MIXTURE, FLAT (test-like; scoring seed 11;
            runs 0-79 each) and R1B, R1P (public R1, split scope 'pair';
            scoring seed 11; runs 0-59 each), as REGIMES defines them.
  D(X,R)    the mean over R's planned runs of the run's pair-mean ALC of X
            minus that of SHIP, both scored on identical runs.
  SEc(X,R)  the cluster-bootstrap SE of D(X,R): Boot of
            experiments/level_calibration.py, 2,000 resamples, bootstrap seed
            0, ratio estimator; the cluster is (parent, subject) on test-like
            runs and (benchmark, subject) on public runs.
  U95(X,R)  D(X,R) + 1.96 SEc(X,R).
  Pq(X,R)   the mean of X minus SHIP over the pair appearances of parent q,
            each weighted 1 / its run's pair count.
  PL(X,R)   the mean of Pq(X,R) over the four multi-subject parents
            matharena, multi_swebench, real_webagents and researchcodebench.
  G(X,R)    D(X,R) with the legacy Predictor in place of SHIP.

X is a CANDIDATE to replace SHIP if and only if all of (a) to (f) hold.
  (a) In READING and in AUDIT: D(X,R) <= -0.002 and U95(X,R) < 0.
  (b) In READING and in AUDIT: PL(X,R) < 0, and Pq(X,R) <= +0.004 for every
      multi-subject parent q.
  (c) In TUNED, in MIXTURE and in FLAT: D(X,R) <= +0.002.
  (d) In R1B and in R1P: D(X,R) <= +0.001 and G(X,R) <= +0.003.
  (e) Pooled over every scored task, X's mean evaluation call takes at most
      2.0 times SHIP's (both timed in the same tasks), and no single call of
      X takes longer than 2.0 s.
  (f) Every reproduction and realisation check of the plan passed, and every
      planned run of every regime was scored for every config.

Reading
  1. No candidate: the outcome is "no candidate: SHIP stays", recorded as a
     negative result for every config.
  2. One or more candidates: all are listed, ordered by the mean of
     D(X,READING) and D(X,AUDIT). The rule does not choose among them, and
     nothing ships automatically: a candidate needs the team's decision, an
     archive that passes tools/build_submission.py, and one formative
     regression check read for errors and latency only.
  3. P1.16: wide35 (sigma_mu 3.5) and wide50 (sigma_mu 5.0) are judged by
     (a) to (f) exactly. "A wider level prior does better" is recorded only
     if one of them is a candidate, and otherwise recorded as no. No sigma_mu
     between or beyond the scored values is chosen from these results.
  4. EB: eb_fit, eb_adapt and eb_ship are judged by (a) to (f) exactly. Only
     if eb_adapt or eb_ship is a candidate is shipping an adaptive level
     revisited (it is experiment code, not library code).
  5. Everything else (ranks by regime, per-budget and per-parent tables, the
     smoothed-mean and no-subject-prior decompositions, EB traces, ECE) is
     descriptive and changes nothing.
  6. Nothing in this rule, CONFIGS, REGIMES, the seeds, the run counts or the
     statistics changes once a row of the scoring seed exists. No run is
     added or dropped after rows are read, and the rule is applied once, to
     the full planned set. If a check fails, the cause is fixed, every
     affected run is re-scored in full and the deviation is recorded; the
     rule is not rewritten. Formative run 3 is not used for anything here.
"""

#: the thresholds RULE_TEXT states, for the code that applies it (lock checks
#: that each value appears in the text as written)
RULE = {
    "gain": 0.002,                 # (a) D <= -0.002 in READING and AUDIT
    "z95": 1.96,                   # (a) U95 = D + 1.96 SEc < 0
    "parent_cap": 0.004,           # (b) every multi-subject parent <= +0.004
    "testlike_loss": 0.002,        # (c) D <= +0.002 in TUNED, MIXTURE, FLAT
    "public_loss": 0.001,          # (d) D <= +0.001 on R1B and R1P
    "legacy_guard": 0.003,         # (d) X minus legacy <= +0.003 on R1B and R1P
    "latency_ratio": 2.0,          # (e) mean call <= 2.0 x SHIP's
    "latency_max_s": 2.0,          # (e) no call over 2.0 s
}
FEEDBACK_REGIMES = ("READING", "AUDIT")             # (a), (b)
NO_LOSS_REGIMES = ("TUNED", "MIXTURE", "FLAT")      # (c)
PUBLIC_REGIMES = ("R1B", "R1P")                     # (d)

# --- configs ----------------------------------------------------------------------------

#: the shipped level; lock checks it equals submission/model.py's LEVEL
LEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
SHIP = "ship"
#: key -> how to build it. 'hier' configs: level_calibration.bundle(parent, 0.0)
#: (prior.build without the target's parent) with `override` replacing Hyper
#: fields (None: as fitted) and `flags` passed to HierPredictor; 'ebhier':
#: level_calibration.EBHier through level_calibration.hier_model(lc_name,
#: parent); 'predictor': paiec.predict.Predictor(*level_calibration.ppred(
#: parent)); 'smoothed': paiec.baselines.smoothed_mean(prior_n=n0, prior_p=m0).
#: lc_name is the config's name in results/level_calibration.json where it has
#: one.
CONFIGS = {
    "ship": {"kind": "hier", "lc_name": "hier G mu0=-2.50 sm=2.50 as=0.50",
             "override": dict(LEVEL), "flags": {}},
    "aggr": {"kind": "hier", "lc_name": "hier G mu0=-3.00 sm=2.50 as=0.25",
             "override": {"mu0": -3.0, "sigma_mu": 2.5, "attr_scale": 0.25}, "flags": {}},
    "eb_fit": {"kind": "hier", "lc_name": "hier", "override": None, "flags": {}},
    "eb_adapt": {"kind": "ebhier", "lc_name": "hierEB-cs,tm=2.0@hier G mu0=-3.00 sm=2.50 as=0.50"},
    "eb_ship": {"kind": "ebhier", "lc_name": "hierEB-cs,tm=2.0@hier G mu0=-2.50 sm=2.50 as=0.50"},
    "wide35": {"kind": "hier", "lc_name": "hier G mu0=-2.50 sm=3.50 as=0.50",
               "override": {"mu0": -2.5, "sigma_mu": 3.5, "attr_scale": 0.5}, "flags": {}},
    "wide50": {"kind": "hier", "lc_name": "hier G mu0=-2.50 sm=5.00 as=0.50",
               "override": {"mu0": -2.5, "sigma_mu": 5.0, "attr_scale": 0.5}, "flags": {}},
    "legacy": {"kind": "predictor", "lc_name": "Predictor"},
    "smooth": {"kind": "smoothed", "lc_name": "smoothed", "n0": 4.0, "m0": 0.5},
    "smcal": {"kind": "smoothed", "n0": None, "m0": None},      # set by the smcal stage
    "onepl": {"kind": "hier", "override": dict(LEVEL),
              "flags": {"attributes": False, "identity": False}},
}
P116 = ("wide35", "wide50")                          # read as RULE_TEXT reading 3
EB = ("eb_fit", "eb_adapt", "eb_ship")               # read as RULE_TEXT reading 4

#: smcal: (k + n0 m0) / (n + n0) on the pair's own labels, selected by
#: level_calibration.select's rule on LEVEL's runs: the lowest mean ALC on test-
#: like seed 2 runs 0-99 among grid points that lose at most GUARD against the
#: legacy Predictor's stored rows (results/level_calibration.json) on public R1
#: seed 0 runs 0-99 of both weightings; ties within SMCAL_TIE to the larger n0,
#: then to the m0 nearer 0.5; none passing: the unconstrained argmin, flagged
#: guard_failed
SMCAL_N0 = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SMCAL_M0 = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SMCAL_SELECT = ("testlike", 2, range(0, 100))
SMCAL_GUARD_RUNS = (("r1", 0, "benchmark", range(0, 100)), ("r1", 0, "pair", range(0, 100)))
SMCAL_GUARD = 0.003
SMCAL_TIE = 1e-6
SMCAL_CHECK_TOL = 1e-5             # the (4, 0.5) column against level_calibration's rows

# --- regimes ----------------------------------------------------------------------------

#: the two-component target of MIXTURE: (weight, mean, sd) on the realised pair-
#: logit scale. Low component and high sd from the 17-pair reading's two-component
#: fits (results/formative_feedback.json reading.level_distribution K=15 and K=40
#: mixture.two: -4.378 / 0.171; sds 1.165, 1.137); weight and high mean set on
#: CAL_SEED to realise its share below -3 (2 of 17) and the rest's mean
MIX = ((0.20, -4.38, 0.171), (0.80, -0.75, 1.151))
#: key -> ('testlike', Regime overrides) or ('r1', sample_run weighting). Every
#: test-like regime keeps Regime()'s other defaults (date shift 1.25, swe_rebench
#: excluded, benchmark tilt); public runs use split scope SPLIT_SCOPE
REGIMES = {
    "TUNED": ("testlike", {}),
    "READING": ("testlike", {"level_mean": -0.85, "level_sd": 1.75}),
    "AUDIT": ("testlike", {"level_mean": -1.8, "level_sd": 2.1}),
    "MIXTURE": ("testlike", {"level_mix": MIX}),
    "FLAT": ("testlike", {"level_mean": None}),
    "R1B": ("r1", "benchmark"),
    "R1P": ("r1", "pair"),
}
SPLIT_SCOPE = "pair"
#: what the knobs were set to realise (continuity-corrected pair logit of the
#: evaluated responses, testlike.describe; unweighted mean and sd, ddof 0)
TARGETS = {
    # the mean of the preregistered K 15 and K 40 readings: means -0.650 and
    # -0.779, sds (ddof 1) 1.800 and 1.745
    "READING": {"mean": -0.715, "sd": 1.772},
    # the review's reading of the audit (run 1, p6 and p8 at upper roots,
    # -1.08 / 1.99 on plain logits), taken as given
    "AUDIT": {"mean": -1.10, "sd": 2.00},
    # 2 of 17 pairs below -3; the K 15 / K 40 means of the other 15, -0.153 and
    # -0.299
    "MIXTURE": {"cut": -3.0, "share_below": 0.118, "rest_mean": -0.226},
    # the tuned regime's known realisation (results/testlike_check.json,
    # summary.check.default.regime.pair_logit, seed 2, 300 runs)
    "TUNED": {"mean": -1.290, "sd": 1.696},
}
#: the knob search (CAL_SEED, CAL_DRAWS draws a setting): argmin of the summed
#: squared (realised - target) / KNOB_SCALE over each regime's grid. The probe
#: drew a subset of the MIXTURE grid; --grid draws all of it
KNOB_SCALE = {"mean": 0.05, "sd": 0.05, "share_below": 0.01, "rest_mean": 0.05}
KNOB_GRID = {
    "READING": {"level_mean": (-0.95, -0.9, -0.85, -0.8, -0.75, -0.7),
                "level_sd": (1.6, 1.7, 1.75, 1.8, 1.9)},
    "AUDIT": {"level_mean": (-2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5),
              "level_sd": (2.0, 2.1, 2.2, 2.3)},
    "MIXTURE": {"low": (-4.38, 0.171), "high_sd": 1.151,
                "weight_low": (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26),
                "high_mean": (-0.25, -0.35, -0.45, -0.55, -0.65, -0.75)},
}
#: what the scratch probes realised for the fixed knobs on CAL_SEED (1,000
#: draws; plan section 4): recorded for comparison, re-measured by `regimes`
PROBE_REALISED = {
    "TUNED": {"mean": -1.273, "sd": 1.731, "share_below": 0.136},
    "READING": {"mean": -0.729, "sd": 1.793, "share_below": 0.089, "rest_mean": -0.423},
    "AUDIT": {"mean": -1.102, "sd": 1.997, "share_below": 0.159},
    "MIXTURE": {"mean": -0.705, "sd": 2.062, "share_below": 0.118, "rest_mean": -0.227,
                "rest_sd": 1.676},
    "FLAT": {"mean": -0.338, "sd": 2.287, "share_below": 0.103},
}
#: realisation check: each test-like regime's planned runs on SCORE_SEED against
#: its CAL_SEED realisation (about 3 sds of an 80-run block on CAL_SEED)
REALISE_TOL = {"mean": 0.30, "sd": 0.30, "share_below": 0.04, "rest_mean": 0.25}

# --- seeds and runs -----------------------------------------------------------------------

#: every run is np.random.default_rng([seed, i]). Seeds used before this study,
#: none of which is scored here: test-like 1 (testlike_check tuning), 2 (LEVEL's
#: selection half 0-99 and confirmation half 100-199; 200-299 in subject_side,
#: itemsig_eval, harness), 3 (the sensitivity regimes), 5 (level_audit extra,
#: 0-39); public sample_run 0 (hier_eval, LEVEL's guard 0-99 and screen 0-39,
#: official_baselines 0-599)
USED_SEEDS = {"testlike": (1, 2, 3, 5), "r1": (0,)}
CAL_SEED = 10                      # knob calibration only: drawn, never scored
CAL_DRAWS = 1000
SCORE_SEED = 11                    # every scored run of every regime
N_RUNS = {"TUNED": 80, "READING": 80, "AUDIT": 80, "MIXTURE": 80, "FLAT": 80,
          "R1B": 60, "R1P": 60}
#: reproduction: old runs with stored rows (ship: data/harness_rows/tl/<i>.pkl,
#: library digest 3f75a549, equal to the current solver on these runs per
#: results/hier_floor.json tl_lib_ship; legacy: results/testlike_check.json
#: raw['check|default|<i>'], exact on pairs whose parent is not matharena, the
#: corrected floor of f7e7d87 changing matharena items only; smooth:
#: results/level_calibration.json, packed to 1e-5)
REPRO = {"regime": ("testlike", 2, {}), "runs": (0, 1, 2, 3, 4)}
REPRO_TOL = {"ship": 1e-9, "legacy": 1e-9, "smooth": 1e-5}
REPRO_EXEMPT_PARENTS = {"legacy": ("matharena",)}
JOBS = 2                           # two independent shard processes
RSS_GUARD_GB = 1.45                # a worker exits (resumable) above this resident size

# --- statistics ---------------------------------------------------------------------------

BOOTS = 2000
BOOT_SEED = 0
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
#: the TUNED consistency check: ship minus legacy within CONSISTENCY_K combined
#: SEs of the known value (results/ship_confirm.json, test-like seed 2 runs
#: 0-299: -0.0415, cluster SE 0.0034)
KNOWN_SHIP_MINUS_LEGACY = (-0.0415, 0.0034)
CONSISTENCY_K = 3.0

OUT = os.path.join(ROOT, "results", "regime_sensitivity.json")
ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")


# --- lock -------------------------------------------------------------------------------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def model_level():
    """submission/model.py's LEVEL, read without importing the module (it
    imports the archive's paiec_rt package)."""
    with open(os.path.join(ROOT, "submission", "model.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("submission/model.py defines no LEVEL")


def check_lock():
    """The lock stage's checks: {name: bool}. All must hold before any other
    stage runs."""
    with open(os.path.join(ROOT, PLAN_PATH), "rb") as f:
        plan = f.read()
    out = {
        "rule_sha256": sha256_bytes(RULE_TEXT.encode("utf-8")) == RULE_SHA256,
        "rule_is_ascii": all(ord(c) < 128 for c in RULE_TEXT),
        "plan_sha256": sha256_bytes(plan) == PLAN_SHA256,
        "plan_quotes_rule": RULE_TEXT.encode("utf-8") in plan,
        "plan_quotes_rule_sha256": RULE_SHA256.encode() in plan,
        "level_is_shipped": model_level() == LEVEL,
        "ship_is_level": CONFIGS[SHIP]["override"] == LEVEL,
        "runs_in_rule": all(f"runs 0-{N_RUNS[r] - 1}" in RULE_TEXT for r in ("TUNED", "R1B"))
        and len({N_RUNS[r] for r in REGIMES if REGIMES[r][0] == "testlike"}) == 1
        and len({N_RUNS[r] for r in PUBLIC_REGIMES}) == 1,
        "seed_in_rule": f"scoring seed {SCORE_SEED}" in RULE_TEXT,
        "seeds_fresh": CAL_SEED not in USED_SEEDS["testlike"] + USED_SEEDS["r1"]
        and SCORE_SEED not in USED_SEEDS["testlike"] + USED_SEEDS["r1"] and CAL_SEED != SCORE_SEED,
        "thresholds_in_rule": all(s in RULE_TEXT for s in (
            "<= -0.002", "1.96 SEc", "<= +0.004", "<= +0.002", "<= +0.001", "<= +0.003",
            "2.0 times", "2.0 s", "2,000 resamples", "bootstrap seed")),
    }
    return out


if sha256_bytes(RULE_TEXT.encode("utf-8")) != RULE_SHA256:
    raise RuntimeError("RULE_TEXT differs from the text fixed at " + FIXED_AT_UTC)


# =========================================================================================
# The stages. Written after the lock, against the constants above, which they do not
# change. Everything below reads the constants; nothing below sets one.
# =========================================================================================

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.ship_confirm as SC  # noqa: E402
from paiec import baselines as BL  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.hier import HierPredictor  # noqa: E402
from paiec.predict import Predictor  # noqa: E402

#: sha256 of this file's constants block, from the line opening "the lock" through the
#: ROWS line, as the planner wrote it (computed on the planner's file before any stage
#: code was added; tests/test_regime_sensitivity.py checks it)
CONSTANTS_SHA256 = "c28c19d6cc4d7e298e8f954e3ef5d941a1c8695551058ac456c276adfdcfab81"
CONSTANTS_START = "# --- the lock ---"
CONSTANTS_END = 'ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")\n'

W6 = np.asarray(WEIGHTS, float)
TESTLIKE = tuple(r for r in REGIMES if REGIMES[r][0] == "testlike")
#: a worker that stops at RSS_GUARD_GB exits with this code; the shard loop restarts it
RESUME_EXIT = 75
LC_JSON = os.path.join(ROOT, "results", "level_calibration.json")
TC_JSON = os.path.join(ROOT, "results", "testlike_check.json")
HARNESS_ROWS = os.path.join(ROOT, "data", "harness_rows", "tl")
COMPOSITIONS = "_compositions.json"            # in ROWS: every planned run's pairs
#: files whose digests every row records; the library part must agree across rows
LIB_FILES = ("paiec/hier.py", "paiec/prior.py", "paiec/predict.py", "paiec/official.py",
             "paiec/testlike.py", "paiec/baselines.py", "paiec/subjects.py", "paiec/mcq.py",
             "paiec/fitting.py", "paiec/irt.py", "paiec/data.py", "paiec/evaluator.py",
             "experiments/level_calibration.py", "submission/model.py")
SMCAL_GRID = tuple((n0, m0) for n0 in SMCAL_N0 for m0 in SMCAL_M0)

#: Deviations from the plan, each found when a check failed, with its cause and fix
#: (RULE_TEXT reading 6). The constants block is not edited; the fix lives here.
DEVIATIONS = [
    {"id": "D1 legacy reproduction tolerance",
     "found": "reproduce stage, 2026-10-01, before any row of the scoring seed was scored",
     "check": "REPRO 'legacy': the legacy Predictor on test-like seed 2 runs 0-4 against "
              "results/testlike_check.json raw['check|default|<i>'] rows[*]['pred'], off matharena, "
              "REPRO_TOL['legacy'] = 1e-9",
     "what_happened": "every run failed at 1e-9, with largest differences 4.7e-7 to 5.0e-7",
     "cause": "the tolerance is below the comparator's precision: experiments/testlike_check.py "
              "stores each Brier as round(x, 6), so an exact reproduction differs from it by up to "
              "5e-7. The planner's probe printed those differences rounded to 6 decimals, which "
              "read as 0.0 ('exact')",
     "fix": "the check is applied at the stored precision: every stored value off matharena "
            "must equal this study's value rounded to 6 decimals (testlike_check's own rounding). "
            "The planned-tolerance result is recorded beside it. Nothing scored depends on the "
            "check, so no run is re-scored",
     "rule": "not rewritten; (f) reads the corrected check, and summary.rule.literal_reading gives "
             "the outcome with the check held at 1e-9"},
]
#: testlike_check.py's rounding of the stored Brier (round(x, 6))
LEGACY_STORED_DECIMALS = 6


# --- lock --------------------------------------------------------------------------------

def constants_block(path=None):
    """The planner's constants block of this file, as text."""
    with open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    a = src.index(CONSTANTS_START)
    b = src.index(CONSTANTS_END, a) + len(CONSTANTS_END)
    return src[a:b]


def lock_checks():
    """check_lock() and the constants block's digest."""
    out = check_lock()
    out["constants_sha256"] = sha256_bytes(constants_block().encode("utf-8")) == CONSTANTS_SHA256
    # >>> after review: the plan may carry an appended "Amendments after review" section;
    # the text before it is what was fixed, and PLAN_SHA256 still pins it byte for byte
    fixed, _ = plan_parts()
    out["plan_sha256"] = sha256_bytes(fixed) == PLAN_SHA256
    out["plan_quotes_rule"] = RULE_TEXT.encode("utf-8") in fixed
    out["plan_quotes_rule_sha256"] = RULE_SHA256.encode() in fixed
    # <<< after review
    return out


def require(state, *stages):
    """Refuse to go on unless the lock holds and every named stage passed."""
    res = lock_checks()
    if not all(res.values()):
        raise SystemExit(f"lock check failed: {[k for k, v in res.items() if not v]}")
    for s in stages:
        if not (state.get(s) or {}).get("ok"):
            raise SystemExit(f"stage '{s}' has not passed; run it first")


# --- io ----------------------------------------------------------------------------------

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean(x):
    """JSON-safe copy: numpy scalars and arrays to Python, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def load_state(path=None):
    path = path or OUT
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_stage(name, value, path=None):
    """Write one stage's result into OUT under a file lock, merging with what is
    there (two stages may finish side by side)."""
    import fcntl
    path = path or OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        state = load_state(path)
        state[name] = clean(value)
        state["updated_utc"] = utcnow()
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return state


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, separators=(",", ":"))
    os.replace(tmp, path)


def file_sha(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


_DIGESTS = None
#: this file's digest when the process started (what a row's code was)
SCRIPT_SHA = file_sha(os.path.abspath(__file__), 16)


def lib_digests():
    global _DIGESTS
    if _DIGESTS is None:
        _DIGESTS = {f: file_sha(os.path.join(ROOT, f), 16) for f in LIB_FILES}
    return _DIGESTS


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"head": git("rev-parse", "--short", "HEAD"),
            "dirty": git("status", "--porcelain", "paiec", "experiments", "submission"),
            "lib": lib_digests(),
            "script": SCRIPT_SHA,
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "utc": utcnow()}


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:                                  # peak, not current, without psutil
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


# --- runs --------------------------------------------------------------------------------

def regime_key(R, seed):
    """level_calibration's regime key for regime R drawn on `seed` (registered on use, so
    level_calibration.draw and describe serve every run here)."""
    kind, arg = REGIMES[R]
    key = f"p1a {R} s{seed}"
    LC.REGIMES[key] = (kind, seed, arg)
    return key


def draw(R, seed, i):
    """(level_calibration key, run): run i of regime R on `seed`, as level_calibration.
    draw makes every run, np.random.default_rng([seed, i])."""
    key = regime_key(R, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return key, LC.draw(key, i)


def describe(key, run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LC.describe(key, run)


def realised(logits, cut=-3.0):
    """The realised level of a set of pair appearances: unweighted mean and sd (ddof 0)
    of their continuity-corrected logits, the share below `cut` and the rest's mean and
    sd."""
    L = np.asarray(logits, float)
    hi = L[L >= cut]
    return {"n": int(len(L)), "mean": float(L.mean()), "sd": float(L.std()),
            "share_below": float(np.mean(L < cut)),
            "rest_mean": float(hi.mean()) if len(hi) else None,
            "rest_sd": float(hi.std()) if len(hi) else None}


def plan_tasks():
    """Every planned (regime, run), run-index-major (i, then regime in REGIMES order)."""
    top = max(N_RUNS.values())
    return [(R, i) for i in range(top) for R in REGIMES if i < N_RUNS[R]]


def shard_tasks(k, n):
    return [t for j, t in enumerate(plan_tasks()) if j % n == k]


def row_path(R, i, rows=ROWS):
    return os.path.join(rows, R, f"{i}.json")


# --- predictors --------------------------------------------------------------------------

class Maker:
    """The model factory of one config on one run for level_calibration.evaluate: every
    call (one per checkpoint) builds fresh per-parent instances, dispatched on the
    anonymous benchmark_id; the previous checkpoint's instances are released after
    their counters (fallbacks, EB traces) are read, so at most one checkpoint's
    instances are alive."""

    def __init__(self, name, run, smcal=None):
        self.name, self.cfg = name, CONFIGS[name]
        self.ids = T.anon_parents(run)
        self.smcal = smcal
        self.k = -1
        self.live = []
        self.failures = self.unconverged = 0
        self.trace = []

    def build(self, parent):
        c = self.cfg
        if c["kind"] == "hier":
            prior, hyper = LC.bundle(parent, 0.0)
            if c["override"] is not None:
                hyper = replace(hyper, **c["override"])
            return HierPredictor(prior, hyper, **c["flags"])
        if c["kind"] == "ebhier":
            return LC.hier_model(c["lc_name"], parent)
        if c["kind"] == "predictor":
            return Predictor(*LC.ppred(parent))
        raise ValueError(c["kind"])

    def harvest(self):
        for m in self.live:
            self.failures += int(getattr(m, "failures", 0))
            self.unconverged += int(getattr(m, "unconverged", 0))
            if isinstance(m, LC.EBHier):
                self.trace += [[BUDGETS[self.k], int(n), round(float(mu), 4), round(float(sm), 4)]
                               for n, mu, sm in m.trace]
        self.live = []

    def __call__(self):
        self.harvest()
        self.k += 1
        c = self.cfg
        if c["kind"] == "smoothed":
            n0, m0 = (c["n0"], c["m0"]) if c["n0"] is not None else self.smcal
            return BL.smoothed_mean(prior_n=float(n0), prior_p=float(m0))
        ms = {par: self.build(par) for par in set(self.ids.values())}
        self.live = list(ms.values())
        ids = self.ids
        return lambda inp, labeled=None: ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled)


def score_run(run, names, smcal):
    """Every config of `names` on the run's checkpoints: {name: per pair Brier, ECE and
    mean B0 prediction, call timing and counters}."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
        out = {}
        for n in names:
            t0 = time.perf_counter()
            mk = Maker(n, run, smcal)
            rows, tm = LC.evaluate(slots, cps, [n], mk)
            mk.harvest()
            r = {"b": [x["brier"] for x in rows[n]], "ece": [x["ece"] for x in rows[n]],
                 "q0": [x["q0"] for x in rows[n]], "calls": tm["calls"], "mean_s": tm["mean_s"],
                 "max_s": tm["max_s"], "failures": mk.failures, "unconverged": mk.unconverged,
                 "secs": round(time.perf_counter() - t0, 3)}
            if mk.trace:
                r["eb"] = mk.trace
            out[n] = r
            del mk, rows
    return out


def smgrid_factory():
    """The 70 smoothed means of SMCAL_GRID as one vector-valued predictor, each entry
    computed as baselines.smoothed_mean computes it."""
    def f():
        def g(inp, labeled=None):
            subject, item = inp
            bm = item.get("benchmark_id")
            s = n = 0
            for (osub, oitem), y in (labeled or []):
                if osub != subject or oitem.get("benchmark_id") != bm:
                    continue
                s += int(y)
                n += 1
            return np.array([float((s + n0 * m0) / (n + n0)) for n0, m0 in SMCAL_GRID])
        return g
    return f


def smcal_name(n0, m0):
    return f"smoothed n0={n0:g} m0={m0:.2f}"


def select_smcal(sel_alc, guard):
    """LEVEL's selection rule on the grid: among points whose guard losses are all at
    most SMCAL_GUARD, the lowest selection ALC; ties within SMCAL_TIE to the larger n0,
    then the m0 nearer 0.5. None passing: the unconstrained argmin, flagged.
    sel_alc: {(n0, m0): ALC}; guard: {(n0, m0): [loss per guard set]}."""
    passing = [g for g in sel_alc if all(x <= SMCAL_GUARD for x in guard[g])]
    pool = passing or list(sel_alc)
    best = min(sel_alc[g] for g in pool)
    tied = [g for g in pool if sel_alc[g] <= best + SMCAL_TIE]
    chosen = sorted(tied, key=lambda g: (-g[0], abs(g[1] - 0.5)))[0]
    return chosen, not passing, passing


# --- stage: smcal --------------------------------------------------------------------------

def stage_smcal():
    state = load_state()
    require(state, "lock")
    t0 = time.time()
    with open(LC_JSON) as f:
        lc_raw = json.load(f)["raw"]
    LC.register_chunks(LC.grid_chunks())
    kind, seed, ids = SMCAL_SELECT
    sets = [("tl", seed, ids)] + [({"benchmark": "r1b", "pair": "r1p"}[w], s, rs)
                                  for _, s, w, rs in SMCAL_GUARD_RUNS]
    assert LC.REGIMES["tl"] == ("testlike", 2, {}) and kind == "testlike" and seed == 2
    assert LC.REGIMES["r1b"] == ("r1", 0, "benchmark") and LC.REGIMES["r1p"] == ("r1", 0, "pair")
    names = [smcal_name(*g) for g in SMCAL_GRID]
    j_ref = SMCAL_GRID.index((4.0, 0.5))
    run_alc = {r: [] for r, _, _ in sets}             # per run: (70,) mean ALC
    legacy = {r: [] for r, _, _ in sets}
    check = {"runs": 0, "pairs": 0, "max_abs_alc_4_0.5": 0.0, "meta_mismatch": [], "missing_rows": []}
    for regime, _, rs in sets:
        for i in rs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run = LC.draw(regime, i)
                slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
                got, _ = LC.evaluate(slots, cps, names, smgrid_factory())
            A = np.array([[LC.alc(x["brier"]) for x in got[n]] for n in names])   # (70, pairs)
            run_alc[regime].append(A.mean(1))
            meta = describe(regime, run)
            stored = lc_raw.get(f"meta|{regime}|{i}")
            if stored is None or [SC.meta_key(p) for p in stored] != [SC.meta_key(p) for p in meta]:
                check["meta_mismatch"].append(f"{regime}|{i}")
            sm = LC.rows_of(lc_raw, regime, i, LC.SMOOTHED)
            pr = LC.rows_of(lc_raw, regime, i, LC.PRED)
            if regime != "tl":
                legacy[regime].append(float(np.mean(pr["alc"])) if pr is not None else float("nan"))
            if sm is None or (regime != "tl" and pr is None):
                check["missing_rows"].append(f"{regime}|{i}")
                continue
            d = float(np.max(np.abs(np.asarray(sm["alc"]) - A[j_ref])))
            check["max_abs_alc_4_0.5"] = max(check["max_abs_alc_4_0.5"], d)
            check["runs"] += 1
            check["pairs"] += A.shape[1]
        print(f"  smcal: {regime} {len(rs)} runs, {time.time() - t0:.0f}s", flush=True)
    sel = np.mean(run_alc["tl"], 0)
    guard_sets = [r for r, _, _ in sets[1:]]
    loss = {r: np.mean(np.array(run_alc[r]) - np.array(legacy[r])[:, None], 0) for r in guard_sets}
    sel_alc = {g: float(sel[j]) for j, g in enumerate(SMCAL_GRID)}
    guard = {g: [float(loss[r][j]) for r in guard_sets] for j, g in enumerate(SMCAL_GRID)}
    chosen, failed, passing = select_smcal(sel_alc, guard)
    ok = (not check["meta_mismatch"] and not check["missing_rows"]
          and check["max_abs_alc_4_0.5"] <= SMCAL_CHECK_TOL
          and check["runs"] == sum(len(rs) for _, _, rs in sets))
    res = {"ok": bool(ok), "chosen": list(chosen), "guard_failed": bool(failed),
           "passing": [list(g) for g in passing],
           "selection": {"regime": "test-like seed 2 (level_calibration 'tl')", "runs": [ids.start, ids.stop - 1]},
           "guard_sets": {r: f"public R1 seed 0, weighting {LC.REGIMES[r][2]}, runs 0-99; legacy rows "
                             "of results/level_calibration.json" for r in guard_sets},
           "grid": [{"n0": g[0], "m0": g[1], "selection_ALC": sel_alc[g],
                     "guard_loss": dict(zip(guard_sets, guard[g])),
                     "passes_guard": g in passing} for g in SMCAL_GRID],
           "chosen_selection_ALC": sel_alc[chosen], "chosen_guard_loss": dict(zip(guard_sets, guard[chosen])),
           "beta22": {"selection_ALC": sel_alc[(4.0, 0.5)], "guard_loss": dict(zip(guard_sets, guard[(4.0, 0.5)]))},
           "min_guard_loss": {r: float(min(guard[g][k] for g in SMCAL_GRID)) for k, r in enumerate(guard_sets)},
           "check": check, "tolerance": SMCAL_CHECK_TOL,
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("smcal", res)
    print(f"smcal: chosen n0={chosen[0]} m0={chosen[1]} guard_failed={failed} check ok={ok} "
          f"(max |d| {check['max_abs_alc_4_0.5']:.2g})", flush=True)


# --- stage: regimes -------------------------------------------------------------------------

def realise_setting(overrides, draws=CAL_DRAWS, seed=CAL_SEED):
    """Realised level of a test-like Regime setting over `draws` runs of `seed`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = T.Sampler(LC.catalogue(), T.Regime().with_(**overrides))
        L, sizes = [], []
        for i in range(draws):
            d = T.describe(S.run(np.random.default_rng([seed, i])))
            L += [x["logit"] for x in d]
            sizes.append(len(d))
    out = realised(L)
    out["pairs_per_run"] = float(np.mean(sizes))
    return out


def knob_search():
    """The knob search of the plan over KNOB_GRID on CAL_SEED (CAL_DRAWS draws a
    setting): the argmin of the summed squared (realised - target) / KNOB_SCALE."""
    out = {}
    for R in ("READING", "AUDIT"):
        g, tg = KNOB_GRID[R], TARGETS[R]
        rows = []
        for lm in g["level_mean"]:
            for ls in g["level_sd"]:
                r = realise_setting({"level_mean": lm, "level_sd": ls})
                obj = ((r["mean"] - tg["mean"]) / KNOB_SCALE["mean"]) ** 2 \
                    + ((r["sd"] - tg["sd"]) / KNOB_SCALE["sd"]) ** 2
                rows.append({"level_mean": lm, "level_sd": ls, "objective": obj,
                             "mean": r["mean"], "sd": r["sd"]})
            print(f"  grid {R} level_mean {lm}", flush=True)
        best = min(rows, key=lambda x: x["objective"])
        fixed = REGIMES[R][1]
        out[R] = {"settings": rows, "argmin": {k: best[k] for k in ("level_mean", "level_sd")},
                  "argmin_is_fixed": best["level_mean"] == fixed["level_mean"]
                  and best["level_sd"] == fixed["level_sd"]}
    g, tg = KNOB_GRID["MIXTURE"], TARGETS["MIXTURE"]
    rows = []
    for w in g["weight_low"]:
        for hm in g["high_mean"]:
            mix = ((w, g["low"][0], g["low"][1]), (1.0 - w, hm, g["high_sd"]))
            r = realise_setting({"level_mix": mix})
            obj = ((r["share_below"] - tg["share_below"]) / KNOB_SCALE["share_below"]) ** 2 \
                + ((r["rest_mean"] - tg["rest_mean"]) / KNOB_SCALE["rest_mean"]) ** 2
            rows.append({"weight_low": w, "high_mean": hm, "objective": obj, "mean": r["mean"],
                         "sd": r["sd"], "share_below": r["share_below"], "rest_mean": r["rest_mean"]})
        print(f"  grid MIXTURE weight_low {w}", flush=True)
    best = min(rows, key=lambda x: x["objective"])
    out["MIXTURE"] = {"settings": rows, "argmin": {k: best[k] for k in ("weight_low", "high_mean")},
                      "argmin_is_fixed": best["weight_low"] == MIX[0][0] and best["high_mean"] == MIX[1][1]}
    return out


def stage_regimes(grid=False):
    state = load_state()
    require(state, "lock")
    t0 = time.time()
    cal, score, checks, comps = {}, {}, {}, {}
    for R in REGIMES:
        kind = REGIMES[R][0]
        metas = {}
        for i in range(N_RUNS[R]):
            key, run = draw(R, SCORE_SEED, i)
            metas[i] = describe(key, run)
        comps[R] = metas
        score[R] = realised([p["logit"] for m in metas.values() for p in m])
        score[R]["runs"] = N_RUNS[R]
        score[R]["pairs_per_run"] = float(np.mean([len(m) for m in metas.values()]))
        if kind != "testlike":
            continue
        cal[R] = realise_setting(REGIMES[R][1])
        keys = ("mean", "sd") + (("share_below", "rest_mean") if R == "MIXTURE" else ())
        checks[R] = {k: {"cal": cal[R][k], "score": score[R][k], "diff": score[R][k] - cal[R][k],
                         "tol": REALISE_TOL[k], "ok": abs(score[R][k] - cal[R][k]) <= REALISE_TOL[k]}
                     for k in keys}
        print(f"  regimes: {R} cal {cal[R]['mean']:+.3f}/{cal[R]['sd']:.3f} "
              f"score {score[R]['mean']:+.3f}/{score[R]['sd']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    vs_target = {}
    for R, tg in TARGETS.items():
        vs_target[R] = {k: cal[R][k] - v for k, v in tg.items() if k != "cut"}
    vs_probe = {R: {k: cal[R][k] - v for k, v in PROBE_REALISED[R].items()
                    if cal[R].get(k) is not None} for R in PROBE_REALISED}
    os.makedirs(ROWS, exist_ok=True)
    cpath = os.path.join(ROWS, COMPOSITIONS)
    write_json(cpath, {"seed": SCORE_SEED, "runs": {R: {str(i): m for i, m in v.items()}
                                                    for R, v in comps.items()}})
    ok = all(c["ok"] for v in checks.values() for c in v.values())
    res = {"ok": bool(ok), "cal_seed": CAL_SEED, "cal_draws": CAL_DRAWS, "score_seed": SCORE_SEED,
           "knobs": {R: (REGIMES[R][1] if REGIMES[R][0] == "testlike" else
                         {"weighting": REGIMES[R][1], "split_scope": SPLIT_SCOPE}) for R in REGIMES},
           "cal": cal, "score": score, "checks": checks, "targets": TARGETS,
           "cal_minus_target": vs_target, "probe": PROBE_REALISED, "cal_minus_probe": vs_probe,
           "compositions": {"file": os.path.relpath(cpath, ROOT), "sha256": file_sha(cpath)},
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    prev = (state.get("regimes") or {}).get("grid")
    if grid:
        res["grid"] = knob_search()
        res["grid"]["note"] = ("the knobs stay as fixed whatever the argmin (plan section 9); "
                               "a different argmin is reported, not used")
        res["wall_s"] = round(time.time() - t0, 1)
    elif prev:
        res["grid"] = prev
    save_stage("regimes", res)
    print(f"regimes: realisation checks {'passed' if ok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)


# --- stage: reproduce ----------------------------------------------------------------------

def harness_brier(slot):
    """Per budget Brier of one pair from experiments/harness.py's stored predictions."""
    K, N = np.asarray(slot["ev_K"], float), np.asarray(slot["ev_N"], float)
    P = np.asarray(slot["ev_p"], float)
    return [float(np.sum(K * (1 - p) ** 2 + (N - K) * p ** 2) / np.sum(N)) for p in P]


def stage_reproduce():
    state = load_state()
    require(state, "lock", "smcal")
    smcal = tuple(state["smcal"]["chosen"])
    t0 = time.time()
    with open(TC_JSON) as f:
        tc_raw = json.load(f)["raw"]
    with open(LC_JSON) as f:
        lc_raw = json.load(f)["raw"]
    LC.register_chunks(LC.grid_chunks())
    kind, seed, knobs = REPRO["regime"]
    assert LC.REGIMES["tl"] == (kind, seed, knobs)
    checks, timing, runs = [], defaultdict(list), {}
    for i in REPRO["runs"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = LC.draw("tl", i)
        meta = describe("tl", run)
        got = score_run(run, list(CONFIGS), smcal)
        for n, r in got.items():
            timing[n].append({"run": f"tl|{i}", "secs": r["secs"], "mean_s": r["mean_s"], "max_s": r["max_s"],
                              "calls": r["calls"], "failures": r["failures"]})
        runs[i] = {"pairs": len(meta), "parents": [p["parent"] for p in meta],
                   "ALC": {n: float(np.mean([LC.alc(b) for b in r["b"]])) for n, r in got.items()}}
        # ship against the harness rows (library digest 3f75a549)
        with open(os.path.join(HARNESS_ROWS, f"{i}.pkl"), "rb") as f:
            h = pickle.load(f)
        same = [(s["name"], s["sid"]) for s in h["slots"]] == [(p["bench"], p["subject"]) for p in meta]
        d = max(abs(x - y) for s, b in zip(h["slots"], got[SHIP]["b"]) for x, y in zip(harness_brier(s), b)) \
            if same else None
        checks.append({"check": f"ship vs data/harness_rows/tl/{i}.pkl", "run": i, "pairs_match": same,
                       "max_abs_brier": d, "tol": REPRO_TOL["ship"], "lib_digest": h["lib_digest"][:8],
                       "level": h["level"],
                       "ok": bool(same and d is not None and d <= REPRO_TOL["ship"] and h["level"] == LEVEL)})
        # legacy against testlike_check's rows, exact off matharena: at the planned
        # tolerance, and at the rows' stored precision (DEVIATIONS[0])
        v = tc_raw.get(f"check|default|{i}")
        same = v is not None and [SC.tc_key(r) for r in v["rows"]] == [SC.meta_key(p) for p in meta]
        per = []
        if same:
            for r, b, p in zip(v["rows"], got["legacy"]["b"], meta):
                per.append((p["parent"], max(abs(x - y) for x, y in zip(r["pred"][:6], b)),
                            all(round(y, LEGACY_STORED_DECIMALS) == x for x, y in zip(r["pred"][:6], b))))
        ex = REPRO_EXEMPT_PARENTS["legacy"]
        d_main = max([x for q, x, _ in per if q not in ex], default=0.0)
        d_ex = max([x for q, x, _ in per if q in ex], default=None)
        stored_equal = all(e for q, _, e in per if q not in ex)
        planned_ok = bool(same and d_main <= REPRO_TOL["legacy"])
        checks.append({"check": f"legacy vs testlike_check.json raw['check|default|{i}']", "run": i,
                       "pairs_match": same, "max_abs_brier": d_main, "tol": REPRO_TOL["legacy"],
                       "planned_tolerance_ok": planned_ok,
                       "equal_at_stored_precision": bool(same and stored_equal),
                       "stored_decimals": LEGACY_STORED_DECIMALS, "deviation": DEVIATIONS[0]["id"],
                       "pairs_checked": sum(q not in ex for q, _, _ in per),
                       "exempt": {"parents": list(ex), "pairs": sum(q in ex for q, _, _ in per),
                                  "max_abs_brier": d_ex,
                                  "equal_at_stored_precision": all(e for q, _, e in per if q in ex)},
                       "ok": bool(same and stored_equal)})
        # smooth against level_calibration.json's rows (packed to 1e-5)
        sm = LC.rows_of(lc_raw, "tl", i, LC.SMOOTHED)
        stored = lc_raw.get(f"meta|tl|{i}")
        same = stored is not None and [SC.meta_key(p) for p in stored] == [SC.meta_key(p) for p in meta]
        d = max(abs(x - y) for pb, b in zip(sm["pb"], got["smooth"]["b"]) for x, y in zip(pb, b)) \
            if same and sm is not None and "pb" in sm else None
        checks.append({"check": f"smooth vs level_calibration.json 'tl|{i}|base' smoothed", "run": i,
                       "pairs_match": same, "max_abs_brier": d, "tol": REPRO_TOL["smooth"],
                       "ok": bool(same and d is not None and d <= REPRO_TOL["smooth"])})
        print(f"  reproduce: run {i} {time.time() - t0:.0f}s "
              f"{[c['ok'] for c in checks[-3:]]}", flush=True)
        del got, run
        gc.collect()
    # timing on two old public runs (seed 0, run 0 of each weighting); informational
    public = {}
    for regime in ("r1b", "r1p"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = LC.draw(regime, 0)
        got = score_run(run, list(CONFIGS), smcal)
        info = {}
        for n, r in got.items():
            timing[n].append({"run": f"{regime}|0", "secs": r["secs"], "mean_s": r["mean_s"],
                              "max_s": r["max_s"], "calls": r["calls"], "failures": r["failures"]})
        for n, lcn in (("legacy", LC.PRED), ("smooth", LC.SMOOTHED)):
            st = LC.rows_of(lc_raw, regime, 0, lcn)
            info[n] = None if st is None else float(np.max(np.abs(
                np.asarray(st["alc"]) - np.array([LC.alc(b) for b in got[n]["b"]]))))
        public[regime] = {"secs": {n: r["secs"] for n, r in got.items()},
                          "run_s": float(sum(r["secs"] for r in got.values())),
                          "max_abs_alc_vs_level_calibration": info}
        print(f"  reproduce: public {regime} run 0 {public[regime]['run_s']:.0f}s", flush=True)
        del got, run
        gc.collect()
    tl_s = [sum(timing[n][k]["secs"] for n in CONFIGS) for k in range(len(REPRO["runs"]))]
    pub_s = [public[r]["run_s"] for r in ("r1b", "r1p")]
    est = sum(N_RUNS[R] for R in TESTLIKE) * float(np.mean(tl_s)) \
        + sum(N_RUNS[R] for R in PUBLIC_REGIMES) * float(np.mean(pub_s))
    ok = all(c["ok"] for c in checks)
    res = {"ok": bool(ok), "runs": list(REPRO["runs"]), "checks": checks, "per_run": runs,
           "deviations": DEVIATIONS,
           "planned_tolerances_ok": all(c.get("planned_tolerance_ok", c["ok"]) for c in checks),
           "timing": {n: v for n, v in timing.items()}, "public_timing": public,
           "estimate": {"testlike_run_s": float(np.mean(tl_s)), "public_run_s": float(np.mean(pub_s)),
                        "score_process_s": est, "score_hours_on_two": est / 2 / 3600,
                        "note": "all configs, scoring only (drawing and checkpoints add a few seconds a run)"},
           "smcal": list(smcal), "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("reproduce", res)
    print(f"reproduce: {'passed' if ok else 'FAILED'}; estimate {est / 2 / 3600:.2f} h on two processes",
          flush=True)


# --- stage: score --------------------------------------------------------------------------

def stage_score(shard, rows_dir=ROWS):
    k, n = shard
    state = load_state()
    require(state, "lock", "smcal", "regimes", "reproduce")
    smcal = tuple(state["smcal"]["chosen"])
    cpath = os.path.join(rows_dir, COMPOSITIONS)
    if file_sha(cpath) != state["regimes"]["compositions"]["sha256"]:
        raise SystemExit(f"{cpath} differs from what the regimes stage wrote")
    with open(cpath) as f:
        comps = json.load(f)["runs"]
    todo = [t for t in shard_tasks(k, n) if not os.path.exists(row_path(*t, rows=rows_dir))]
    print(f"score shard {k}/{n}: {len(todo)} of {len(shard_tasks(k, n))} tasks to do; "
          f"smcal {smcal}", flush=True)
    t_start = time.time()
    for j, (R, i) in enumerate(todo):
        t0 = time.time()
        key, run = draw(R, SCORE_SEED, i)
        meta = describe(key, run)
        if json.loads(json.dumps(meta)) != comps[R][str(i)]:
            raise RuntimeError(f"{R} run {i}: the drawn run differs from the regimes stage's")
        res = score_run(run, list(CONFIGS), smcal)
        rss = rss_gb()
        row = {"regime": R, "i": i, "seed": SCORE_SEED, "key": key, "meta": meta, "res": res,
               "smcal": list(smcal), "task_s": round(time.time() - t0, 3), "rss_gb": round(rss, 3),
               "shard": f"{k}/{n}", "pid": os.getpid(), "lib": lib_digests(),
               "script": SCRIPT_SHA, "utc": utcnow()}
        write_json(row_path(R, i, rows=rows_dir), row)
        del res, run, row
        gc.collect()
        rss = rss_gb()
        print(f"  [{k}/{n}] {R} {i}: {time.time() - t0:.1f}s, rss {rss:.2f} GB, "
              f"{j + 1}/{len(todo)} in {time.time() - t_start:.0f}s", flush=True)
        if rss > RSS_GUARD_GB and j + 1 < len(todo):
            print(f"  [{k}/{n}] resident set {rss:.2f} GB > {RSS_GUARD_GB} GB: stopping (resumable)",
                  flush=True)
            sys.exit(RESUME_EXIT)
    print(f"score shard {k}/{n}: done", flush=True)


def stage_status(rows_dir=ROWS):
    for R in REGIMES:
        have = sum(os.path.exists(row_path(R, i, rows=rows_dir)) for i in range(N_RUNS[R]))
        print(f"{R:8s} {have:3d}/{N_RUNS[R]}")


# --- stage: summarise ----------------------------------------------------------------------

def load_rows(rows_dir=ROWS):
    rows = {R: {} for R in REGIMES}
    for R in REGIMES:
        for i in range(N_RUNS[R]):
            p = row_path(R, i, rows=rows_dir)
            if os.path.exists(p):
                with open(p) as f:
                    rows[R][i] = json.load(f)
    return rows


class RegimeRows:
    """One regime's scored runs: metas, per config and run the (pairs, 6) Brier, one
    level_calibration.Boot over its clusters, and the flattened appearance weights."""

    def __init__(self, R, rows):
        self.R = R
        self.ids = sorted(rows)
        self.metas = [rows[i]["meta"] for i in self.ids]
        self.B = {c: [np.asarray(rows[i]["res"][c]["b"], float) for i in self.ids] for c in CONFIGS}
        self.A = {c: [b @ W6 for b in self.B[c]] for c in CONFIGS}
        self.rows = rows
        self.boot = LC.Boot(self.metas, BOOTS, BOOT_SEED)
        self.cl = np.concatenate(self.boot.idx)
        self.w = np.concatenate([np.full(len(m), 1.0 / len(m)) for m in self.metas])
        self.parent = np.array([p["parent"] for m in self.metas for p in m])
        self.K = len(self.boot.keys)

    def runs_of(self, c):
        return {i: (m, b) for i, m, b in zip(self.ids, self.metas, self.B[c])}

    def arm(self, c):
        a = SC.Arm(f"{c}, scored in the same tasks")
        for i, b in zip(self.ids, self.B[c]):
            a.put(i, b @ W6, b, "same task")
        return a

    def diff(self, x, ref):
        """X minus ref, unrounded: the rule's D, SEc, U95, Pq, PL, and the cluster
        bootstrap's 2.5 and 97.5 percentiles."""
        d = [a - b for a, b in zip(self.A[x], self.A[ref])]
        run_d = np.array([float(np.mean(v)) for v in d])
        c, s = self.boot.se(d)
        rse = float(run_d.std(ddof=1) / math.sqrt(len(run_d))) if len(run_d) > 1 else None
        flat = np.concatenate(d)
        v = np.bincount(self.cl, self.w * flat, minlength=self.K)
        n = np.bincount(self.cl, self.w, minlength=self.K)
        dist = (self.boot.W @ v) / (self.boot.W @ n)
        pq = {}
        for q in sorted(set(self.parent.tolist())):
            m = self.parent == q
            pq[q] = float(np.sum(self.w[m] * flat[m]) / np.sum(self.w[m]))
        multi = [pq[q] for q in PARENTS if q in pq]
        D = float(run_d.mean())
        return {"D": D, "run_se": rse, "cluster_se": c, "strat_se": s, "U95": D + RULE["z95"] * c,
                "pct": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                "Pq": pq, "PL": float(np.mean(multi)) if multi else None,
                "parents_present": [q for q in PARENTS if q in pq],
                "share_runs_x_better": float(np.mean(run_d < 0))}

    def table(self, x, ref):
        """ship_confirm.Block's paired report of X minus ref (rounded to 1e-6): per-
        budget rows, per-parent means with cluster SEs and leave-one-out values, the
        parent-level mean and SE."""
        rep, _ = SC.Block(self.runs_of(x), self.ids).paired(self.arm(ref))
        rep["x_ALC"] = rep.pop("shipped_ALC")
        rep["x_ALC_run_se"] = rep.pop("shipped_ALC_run_se")
        rep["ref_ALC"] = rep.pop("other_ALC")
        rep["ref_ALC_run_se"] = rep.pop("other_ALC_run_se")
        rep["share_runs_x_better"] = rep.pop("runs_shipped_better")
        for b in rep.get("budgets", []):
            b["x"], b["ref"] = b.pop("shipped"), b.pop("other")
        return rep

    def config_summary(self, c):
        run_alc = np.array([float(np.mean(a)) for a in self.A[c]])
        res = [self.rows[i]["res"][c] for i in self.ids]
        ece = [np.asarray(r["ece"], float) for r in res]
        return {"runs": len(self.ids), "ALC": float(run_alc.mean()),
                "ALC_run_se": float(run_alc.std(ddof=1) / math.sqrt(len(run_alc))),
                "ALC_cluster_se": self.boot.se(self.A[c])[0],
                "budgets": [float(x) for x in np.mean([b.mean(0) for b in self.B[c]], 0)],
                "q0": float(np.mean([np.mean(r["q0"]) for r in res])),
                "ece0": float(np.mean([e[:, 0].mean() for e in ece])),
                "ece_alc": float(np.mean([np.mean(e @ W6) for e in ece])),
                "failures": int(sum(r["failures"] for r in res)),
                "unconverged": int(sum(r["unconverged"] for r in res))}

    def level(self, cut=-3.0):
        """The realised level of the scored runs: unweighted over appearances, with a
        cluster-bootstrap SE (same resamples) and the leave-one-parent-out range."""
        L = np.array([p["logit"] for m in self.metas for p in m], float)
        out = realised(L, cut)
        v = np.bincount(self.cl, L, minlength=self.K)
        n = np.bincount(self.cl, np.ones_like(L), minlength=self.K)
        out["cluster_se"] = float(((self.boot.W @ v) / (self.boot.W @ n)).std(ddof=1))
        lo = {q: float(L[self.parent != q].mean()) for q in sorted(set(self.parent.tolist()))}
        out["leave_one_parent_out"] = lo
        out["leave_one_parent_out_range"] = [min(lo.values()), max(lo.values())]
        out["per_parent"] = {q: [float(L[self.parent == q].mean()), int((self.parent == q).sum())]
                             for q in sorted(set(self.parent.tolist()))}
        return out


def latency(rows):
    """Per config: pooled mean call (total time over total calls) and the largest
    call, over every scored task and per regime kind, with ratios to SHIP's in the
    same tasks."""
    acc = defaultdict(lambda: [0.0, 0, 0.0, 0.0])         # time, calls, max, task seconds
    for R, rs in rows.items():
        kind = "testlike" if REGIMES[R][0] == "testlike" else "public"
        for r in rs.values():
            for c, v in r["res"].items():
                for k in ("all", kind):
                    a = acc[(k, c)]
                    a[0] += v["mean_s"] * v["calls"]
                    a[1] += v["calls"]
                    a[2] = max(a[2], v["max_s"])
                    a[3] += v["secs"]
    out = {}
    for k in ("all", "testlike", "public"):
        ship = acc[(k, SHIP)]
        sm = ship[0] / ship[1] if ship[1] else None
        out[k] = {c: {"calls": acc[(k, c)][1],
                      "mean_ms": 1e3 * acc[(k, c)][0] / acc[(k, c)][1] if acc[(k, c)][1] else None,
                      "max_s": acc[(k, c)][2], "config_s": acc[(k, c)][3],
                      "ratio_to_ship": (acc[(k, c)][0] / acc[(k, c)][1]) / sm if sm and acc[(k, c)][1] else None}
                  for c in CONFIGS}
    return out


def eb_traces(rows):
    acc = defaultdict(list)
    for R, rs in rows.items():
        for r in rs.values():
            for c, v in r["res"].items():
                for b, _, mu, sm in v.get("eb", ()):
                    acc[(c, R, b)].append((mu, sm))
    out = defaultdict(dict)
    for (c, R, b), xs in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        a = np.array(xs)
        out[f"{c} | {R}"][f"B{b}"] = {"mu0": float(a[:, 0].mean()), "sigma_mu": float(a[:, 1].mean()),
                                      "mu0_sd": float(a[:, 0].std()), "models": len(xs)}
    return dict(out)


def ranks(G):
    """Configs by mean ALC, each with the configs it is tied with (paired |D| under
    1.96 cluster SEs). Descriptive."""
    alc = {c: float(np.mean([np.mean(a) for a in G.A[c]])) for c in CONFIGS}
    order = sorted(CONFIGS, key=alc.get)
    ties = defaultdict(list)
    cs = list(CONFIGS)
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            d = G.diff(cs[a], cs[b])
            if abs(d["D"]) < RULE["z95"] * d["cluster_se"]:
                ties[cs[a]].append(cs[b])
                ties[cs[b]].append(cs[a])
    return [{"rank": k + 1, "config": c, "ALC": alc[c], "tied_with": sorted(ties[c], key=order.index)}
            for k, c in enumerate(order)]


def apply_rule(diffs, glegacy, lat, checks_ok, complete):
    """RULE_TEXT's (a) to (f) for every config but SHIP. diffs[R][X]: X minus SHIP
    (RegimeRows.diff); glegacy[R][X]: X minus legacy; lat: latency()['all']."""
    table = {}
    for X in CONFIGS:
        if X == SHIP:
            continue
        a = {R: {"D": diffs[R][X]["D"], "U95": diffs[R][X]["U95"],
                 "ok": diffs[R][X]["D"] <= -RULE["gain"] and diffs[R][X]["U95"] < 0}
             for R in FEEDBACK_REGIMES}
        b = {}
        for R in FEEDBACK_REGIMES:
            pq = {q: diffs[R][X]["Pq"][q] for q in PARENTS if q in diffs[R][X]["Pq"]}
            pl = diffs[R][X]["PL"]
            b[R] = {"PL": pl, "Pq": pq, "worst_parent": max(pq, key=pq.get) if pq else None,
                    "ok": pl is not None and pl < 0 and all(v <= RULE["parent_cap"] for v in pq.values())}
        c = {R: {"D": diffs[R][X]["D"], "ok": diffs[R][X]["D"] <= RULE["testlike_loss"]}
             for R in NO_LOSS_REGIMES}
        d = {R: {"D": diffs[R][X]["D"], "G": glegacy[R][X],
                 "ok": diffs[R][X]["D"] <= RULE["public_loss"] and glegacy[R][X] <= RULE["legacy_guard"]}
             for R in PUBLIC_REGIMES}
        lx, ls = lat[X], lat[SHIP]
        e = {"mean_ms": lx["mean_ms"], "ship_mean_ms": ls["mean_ms"], "ratio": lx["ratio_to_ship"],
             "max_s": lx["max_s"],
             "ok": lx["ratio_to_ship"] is not None and lx["ratio_to_ship"] <= RULE["latency_ratio"]
             and lx["max_s"] <= RULE["latency_max_s"]}
        f = {"checks_passed": bool(checks_ok), "every_planned_run_scored": bool(complete),
             "ok": bool(checks_ok and complete)}
        conds = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}
        okc = {k: (all(v["ok"] for v in conds[k].values()) if k in "abcd" else conds[k]["ok"])
               for k in "abcdef"}
        why = []
        for R, v in a.items():
            if not v["ok"]:
                why.append(f"(a) {R}: D {v['D']:+.4f}, U95 {v['U95']:+.4f}")
        for R, v in b.items():
            if not v["ok"]:
                why.append(f"(b) {R}: PL {v['PL']:+.4f}, worst parent {v['worst_parent']} "
                           f"{v['Pq'][v['worst_parent']]:+.4f}")
        for R, v in c.items():
            if not v["ok"]:
                why.append(f"(c) {R}: D {v['D']:+.4f}")
        for R, v in d.items():
            if not v["ok"]:
                why.append(f"(d) {R}: D {v['D']:+.4f}, vs legacy {v['G']:+.4f}")
        if not e["ok"]:
            why.append(f"(e) mean call {e['ratio']:.2f} x ship's, max {e['max_s']:.2f} s")
        if not f["ok"]:
            why.append("(f) a check failed or a planned run is missing")
        cand = all(okc.values())
        table[X] = {"conditions": conds, "holds": okc, "candidate": cand,
                    "reading": "candidate" if cand else "not a candidate: " + "; ".join(why)}
    cands = sorted([X for X in table if table[X]["candidate"]],
                   key=lambda X: np.mean([diffs[R][X]["D"] for R in FEEDBACK_REGIMES]))
    if not complete:
        outcome = "rule not applied: the planned set is not complete"
    elif not checks_ok:
        outcome = "rule not applied: check failed"
    elif not cands:
        outcome = "no candidate: SHIP stays"
    else:
        outcome = "candidates: " + ", ".join(cands)
    applied = bool(complete and checks_ok)
    return {"applied": applied, "outcome": outcome, "candidates": cands if applied else [],
            "candidates_order": "by the mean of D(X, READING) and D(X, AUDIT)",
            "p116": None if not applied else (
                "yes: a wider level prior does better (" + ", ".join(x for x in P116 if x in cands) + ")"
                if any(x in cands for x in P116) else "no: neither sigma_mu 3.5 nor 5.0 is a candidate"),
            "eb": None if not applied else (
                "revisit: an adaptive level is a candidate (" + ", ".join(x for x in EB if x in cands) + ")"
                if any(x in cands for x in ("eb_adapt", "eb_ship")) else
                "no: neither adaptive EB config is a candidate" + (
                    " (eb_fit is)" if "eb_fit" in cands else "")),
            "negative": [] if not applied else [X for X in table if not table[X]["candidate"]],
            "table": table}


def stage_summarise(rows_dir=ROWS):
    state = load_state()
    require(state, "lock")
    t0 = time.time()
    rows = load_rows(rows_dir)
    planned = {R: N_RUNS[R] for R in REGIMES}
    have = {R: len(rows[R]) for R in REGIMES}
    complete = all(have[R] == planned[R] for R in REGIMES) and all(
        set(r["res"]) == set(CONFIGS) for rs in rows.values() for r in rs.values())
    checks = {}
    for s in ("lock", "smcal", "regimes", "reproduce"):
        checks[s] = bool((state.get(s) or {}).get("ok"))
    checks["lock_now"] = all(lock_checks().values())
    # every row: one library, the smcal point chosen, the composition the regimes stage drew
    cpath = os.path.join(rows_dir, COMPOSITIONS)
    comps = {}
    if os.path.exists(cpath):
        with open(cpath) as f:
            comps = json.load(f)["runs"]
    libs = {json.dumps(r["lib"], sort_keys=True) for rs in rows.values() for r in rs.values()}
    smc = {tuple(r["smcal"]) for rs in rows.values() for r in rs.values()}
    bad_comp = [f"{R}|{i}" for R, rs in rows.items() for i, r in rs.items()
                if comps.get(R, {}).get(str(i)) != r["meta"]]
    checks["rows_one_library"] = len(libs) <= 1
    checks["rows_smcal"] = smc <= {tuple(state.get("smcal", {}).get("chosen", ()))}
    checks["rows_compositions"] = not bad_comp
    summary = {"planned": planned, "scored": have, "complete": bool(complete),
               "rows": {"library_digests": [json.loads(x) for x in sorted(libs)],
                        "script_digests": sorted({r["script"] for rs in rows.values() for r in rs.values()}),
                        "compositions_mismatch": bad_comp,
                        "task_s": {R: float(np.mean([r["task_s"] for r in rs.values()])) if rs else None
                                   for R, rs in rows.items()},
                        "max_rss_gb": max([r["rss_gb"] for rs in rows.values() for r in rs.values()],
                                          default=None)}}
    if not any(rows.values()):
        summary["note"] = "no scored rows"
        save_stage("summary", summary)
        return
    regs = {R: RegimeRows(R, rows[R]) for R in REGIMES if rows[R]}
    summary["realised"] = {R: G.level() for R, G in regs.items()}
    summary["configs"] = {R: {c: G.config_summary(c) for c in CONFIGS} for R, G in regs.items()}
    diffs = {R: {X: G.diff(X, SHIP) for X in CONFIGS if X != SHIP} for R, G in regs.items()}
    glegacy = {R: {X: (0.0 if X == "legacy" else G.diff(X, "legacy")["D"]) for X in CONFIGS if X != SHIP}
               for R, G in regs.items()}
    # the rule's quantities next to ship_confirm.Block's rounded tables of the same differences
    vs_ship, vs_legacy, vs_smooth = {}, {}, {}
    agree = 0.0
    for R, G in regs.items():
        vs_ship[R], vs_legacy[R], vs_smooth[R] = {}, {}, {}
        for X in CONFIGS:
            if X != SHIP:
                t = G.table(X, SHIP)
                ex = diffs[R][X]
                agree = max(agree, abs(t["diff"]["mean"] - ex["D"]), abs(t["diff"]["cluster_se"] - ex["cluster_se"]),
                            *(abs(t["per_parent"][q]["mean"] - ex["Pq"][q]) for q in ex["Pq"]))
                if t["parent_level"]["mean"] is not None:
                    agree = max(agree, abs(t["parent_level"]["mean"] - ex["PL"]))
                vs_ship[R][X] = {"exact": ex, "table": t}
            if X != "legacy":
                vs_legacy[R][X] = G.table(X, "legacy")
            if X != "smooth":
                vs_smooth[R][X] = G.table(X, "smooth")
    checks["block_tables_agree"] = agree <= 1e-6
    summary["block_agreement_max_abs"] = agree
    summary["vs_ship"] = vs_ship
    summary["vs_legacy"] = vs_legacy
    summary["vs_smooth"] = vs_smooth
    summary["decomposition"] = {
        R: {f"ship minus {X}": {"mean": -diffs[R][X]["D"], "run_se": diffs[R][X]["run_se"],
                                "cluster_se": diffs[R][X]["cluster_se"], "strat_se": diffs[R][X]["strat_se"],
                                "parent_level": -diffs[R][X]["PL"]}
            for X in ("smcal", "onepl", "smooth", "legacy")} for R in regs}
    summary["ranks"] = {R: ranks(G) for R, G in regs.items()}
    summary["eb_traces"] = eb_traces(rows)
    lat = latency(rows)
    summary["latency"] = lat
    # the TUNED consistency check: ship minus legacy against ship_confirm's -0.0415
    if "TUNED" in regs:
        d = regs["TUNED"].diff(SHIP, "legacy")
        want, want_se = KNOWN_SHIP_MINUS_LEGACY
        comb = math.sqrt(d["cluster_se"] ** 2 + want_se ** 2)
        ok = abs(d["D"] - want) <= CONSISTENCY_K * comb
        summary["tuned_consistency"] = {"ship_minus_legacy": d["D"], "run_se": d["run_se"],
                                        "cluster_se": d["cluster_se"], "known": want, "known_se": want_se,
                                        "combined_se": comb, "z": (d["D"] - want) / comb,
                                        "k": CONSISTENCY_K, "ok": bool(ok)}
        checks["tuned_consistency"] = bool(ok)
    else:
        checks["tuned_consistency"] = False
    summary["checks"] = checks
    checks_ok = all(checks.values())
    if all(R in regs for R in REGIMES):
        rule = apply_rule(diffs, glegacy, lat["all"], checks_ok, complete)
        # the same with every check held at the plan's own tolerances (DEVIATIONS)
        literal_ok = checks_ok and bool((state.get("reproduce") or {}).get("planned_tolerances_ok"))
        lit = apply_rule(diffs, glegacy, lat["all"], literal_ok, complete)
        rule["literal_reading"] = {"checks_passed": literal_ok, "outcome": lit["outcome"],
                                   "candidates": lit["candidates"]}
    else:
        rule = {"applied": False, "outcome": "rule not applied: the planned set is not complete"}
    rule["deviations"] = (state.get("reproduce") or {}).get("deviations", [])
    rule["rule_sha256"] = RULE_SHA256
    rule["thresholds"] = RULE
    rule["applied_utc"] = utcnow()
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["provenance"] = provenance()
    save_stage("summary", summary)
    save_stage("rule", rule)
    save_stage("status", "complete" if rule.get("applied") else "partial")
    print(f"summarise: {have} scored; checks {checks}; outcome: {rule['outcome']}", flush=True)


# >>> after review (2026-10-01) ===========================================================
# The review stage. Everything from here to the closing line below was added after the
# study was scored and reviewed (see the header block and the plan's "Amendments after
# review"). It changes no stage above, no row, no summary number and not the rule's
# reading: it reads them, re-runs the deterministic pre-scoring stages into a scratch
# directory, scores what the review asked to be measured, and writes under "review".

#: sha256 of this file as it was when it scored every row (each row records its first 16
#: hex digits); scored_source() of the current file must hash to it
SCORED_SCRIPT_SHA256 = "0bef6bf07d23bd876dc128a2330b10c8a0d37076d5214f7712e9251a2b56c021"
REVIEW_OPEN, REVIEW_CLOSE = "# >>> after review", "# <<< after review"
#: the heading that opens the plan's appended section; the bytes before it are the plan
#: as fixed
PLAN_AMENDMENTS = "\n## Amendments after review"
#: the light parts read OUT and the rows; smcal_guard, rerun and rescore score runs
REVIEW_PARTS = ("provenance", "levels", "headline", "notes", "smcal_guard", "rerun", "rescore")
LIGHT_PARTS = ("provenance", "levels", "headline", "notes")
#: the run report's "within +-0.0006 of SHIP", read at the report's four decimals
NEAR_SHIP = 0.0006
#: a second low cut beside the plan's -3, for MIXTURE's low component (-4.38, sd 0.171)
LOW_CUT = -4.0
#: rescore: the last planned run of every regime (a rule fixed before re-scoring)
RESCORE_TASKS = tuple((R, N_RUNS[R] - 1) for R in REGIMES)
#: scratch evidence the lock and the deviation rest on, relative to --evidence-dir (the
#: study's scratch directory, plan section 12): what each file is
EVIDENCE = {
    "rule.txt": "the planner's rule text, written before the plan was fixed",
    "impl/regime_sensitivity.orig.py": "the implementer's copy of the planner's file, taken before "
                                       "any stage code was written",
    "impl/reproduce_out.txt": "log of the first reproduce run (legacy failed at 1e-9: D1)",
    "impl/reproduce2_out.txt": "log of the reproduce run with D1's fix",
    "impl/dry_summarise.py": "the dry run of summarise on the rows scored so far",
    "impl/dry_out.json": "its output: a scratch copy of OUT",
    "impl/pytest_out.txt": "log of the first full pytest run (default BLAS threads, killed)",
    "review/evidence/regime_sensitivity.scored.py": "this file as it scored the rows, copied with its "
                                                    "mtime (cp -p) before the review's edits",
    "review/evidence/p1a_regime_sensitivity.fixed.md": "the plan as fixed, copied with its mtime "
                                                       "(cp -p) before the amendment",
}
SMCAL_GUARD_NOTE = (
    "The smcal guard's reference is the legacy Predictor's per-pair ALC stored in "
    "results/level_calibration.json (committed in ee5085a, 2026-09-26). It predates the "
    "multiple-choice floor fix f7e7d87 (2026-09-27), so on matharena pairs it is not what the "
    "current legacy Predictor predicts (off matharena the two agree to the stored precision). "
    "The plan asked for the stored rows. The reference can change the selection only if some "
    "grid point passes the guard against the current legacy Predictor: no point passes against "
    "the stored one, so smcal is the unconstrained argmin on the selection half, which does not "
    "involve the legacy Predictor. review.smcal_guard re-scores the current legacy Predictor on "
    "the guard's 200 runs and recomputes every point's guard loss against it.")
#: one note per finding of the verifier's review (V1 to V10, in its order): what it found,
#: what was done here and where the result is, and what is left
REVIEW_NOTES = [
    {"id": "V1", "finding": "the literal reading depends on D1 (legacy reproduction checked at the "
                            "stored 6 decimals, not 1e-9)",
     "done": "review.provenance.d1 records that D1 was fixed and its check passed before the first "
             "scoring-seed row, that every stored legacy difference lies within testlike_check's "
             "rounding, both readings with their candidates, and which configs meet (a) and (a) to "
             "(e), which (f) does not touch. The score stage reads only the reproduce stage's ok flag, "
             "so no row depends on the check",
     "left": "the team accepts D1, or records P1a as 'not applied (literal)'; either way no config "
             "is a candidate"},
    {"id": "V2", "finding": "the lock time stamp is self-reported: nothing was committed before scoring",
     "done": "review.provenance.evidence and .timeline record what the lock rests on: digests and "
             "mtimes of the planner's rule text, the implementer's pre-edit copy, the plan and the "
             "scored script (both copied with their mtimes before the review's edits), the composition "
             "file and the rows' time stamps",
     "left": "commit the plan, script, tests and results together; the findings say the lock is "
             "evidenced by local mtimes and scratch copies, not by a commit made before scoring"},
    {"id": "V3", "finding": "fresh seeds are not fresh data",
     "done": "recorded: the seed-11 runs are new draws from the same catalogue and the same public "
             "pairs that chose LEVEL (and aggr, level_calibration's winner) and guarded it. TUNED and "
             "the guards (c) and (d) therefore carry that selection's optimism, toward SHIP or aggr, "
             "and are not independent replication",
     "left": "the findings carry the sentence"},
    {"id": "V4", "finding": "the scored runs' realised levels sit milder than the plan's bracket",
     "done": "review.levels gives each regime's realised level on the scored runs beside its target, "
             "the share below -4, and the pseudo-benchmarks that carry MIXTURE's low mode",
     "left": "the findings quote the realised levels, not the targets"},
    {"id": "V5", "finding": "a dry run of summarise read a partial set during scoring",
     "done": "review.provenance.dry_run records it (when, how many rows, what it printed) and that "
             "every row and the final summary carry the single scored script digest",
     "left": "none"},
    {"id": "V6", "finding": "two sentences of the run report overstate how close the configs are",
     "done": "review.headline takes D and its SEs from summary.vs_ship[R][X].exact and lists which "
             "hier configs sit within +-0.0006 of SHIP in both READING and AUDIT, and each config's "
             "cluster SE range there",
     "left": "the findings scope both sentences as review.headline does"},
    {"id": "V7", "finding": "D1 changes a locked check's tolerance after the lock",
     "done": "as V1; both readings are recorded in rule and in review.provenance.d1",
     "left": "the team signs off D1; future plans set reproduction tolerances at the comparator's "
             "stored precision"},
    {"id": "V8", "finding": "stage provenance spans three script versions",
     "done": "review.provenance.stages lists each stage's script digest; review.rerun re-runs smcal, "
             "regimes (without --grid) and reproduce under this file, whose stage code is the scored "
             "script's, into a scratch directory and compares every deterministic field",
     "left": "commit the files together (V2)"},
    {"id": "V9", "finding": "smcal's guard reference predates f7e7d87",
     "done": "review.smcal_guard (the note and the re-scored guard); the same note is attached to the "
             "smcal stage as guard_reference_note_after_review",
     "left": "the findings carry the note"},
    {"id": "V10", "finding": "rows are not byte-reproducible: EB trace order follows PYTHONHASHSEED",
     "done": "review.rescore re-scores the last planned run of every regime and compares Brier, ECE, "
             "B0 means, counters and EB traces (as stored and sorted). review.provenance.first_pytest "
             "locates the F of the first, killed, full pytest run",
     "left": "a later script version iterates sorted parents in Maker.__call__ (not this one: it is "
             "the scoring code the rows record)"},
]


def utc_of(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scored_source(path=None):
    """This file with every after-review block cut out, marker lines included: the script
    that scored the rows, as long as nothing outside the blocks was edited."""
    with open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    keep, inside = [], False
    for ln in lines:
        s = ln.lstrip()
        if s.startswith(REVIEW_OPEN):
            if inside:
                raise ValueError("an after-review block opens inside another")
            inside = True
        elif s.startswith(REVIEW_CLOSE):
            if not inside:
                raise ValueError("an after-review block closes without opening")
            inside = False
        elif not inside:
            keep.append(ln)
    if inside:
        raise ValueError("an after-review block is not closed")
    return "".join(keep)


def plan_parts(path=None):
    """The plan file as bytes, split into the text as fixed and the appended amendments
    (empty before any amendment)."""
    with open(path or os.path.join(ROOT, PLAN_PATH), "rb") as f:
        plan = f.read()
    k = plan.find(PLAN_AMENDMENTS.encode("utf-8"))
    return (plan, b"") if k < 0 else (plan[:k], plan[k:])


def locked_update(fn, path=None):
    """fn(state) on OUT under save_stage's file lock; what fn adds must be JSON-clean."""
    import fcntl
    path = path or OUT
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        state = load_state(path)
        fn(state)
        state["updated_utc"] = utcnow()
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)


def save_review(part, value, path=None):
    """One part into OUT['review'][part], leaving the other parts as they are."""
    def put(state):
        rv = state.get("review") or {}
        rv[part] = clean(value)
        rv["updated_utc"] = utcnow()
        state["review"] = rv
    locked_update(put, path)


def compare_json(a, b, skip=(), path=""):
    """(absolute numeric differences, paths that differ otherwise) between two JSON values;
    keys in `skip`, and keys ending in '_after_review', are not compared."""
    nums, bad = [], []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip or k.endswith("_after_review"):
                continue
            if k not in a or k not in b:
                bad.append(f"{path}/{k}")
                continue
            n, x = compare_json(a[k], b[k], skip, f"{path}/{k}")
            nums += n
            bad += x
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return nums, [f"{path} (length {len(a)} against {len(b)})"]
        for j, (x, y) in enumerate(zip(a, b)):
            n, z = compare_json(x, y, skip, f"{path}[{j}]")
            nums += n
            bad += z
    elif isinstance(a, bool) or isinstance(b, bool):
        if not (isinstance(a, bool) and isinstance(b, bool) and a == b):
            bad.append(path)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        nums.append(abs(float(a) - float(b)))
    elif a != b:
        bad.append(path)
    return nums, bad


def first_failure_position(text):
    """1-based position of the first F or E in a pytest -q progress log, or None."""
    import re
    n = 0
    for ln in text.splitlines():
        if ln.startswith("="):
            break
        m = re.match(r"[.FEsxX]+", ln.strip())
        if not m:
            continue
        for ch in m.group(0):
            n += 1
            if ch in "FE":
                return n
    return None


def collected_tests():
    """pytest's collection order for this repository, as `pytest --collect-only -q` lists it."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    return [ln.strip() for ln in p.stdout.splitlines() if "::" in ln]


# --- review parts ------------------------------------------------------------------------

def review_evidence(evidence_dir, state, first_row, last_row):
    """Digests, mtimes and checks of the scratch evidence, and the time line they give."""
    files = {}
    for rel, what in EVIDENCE.items():
        p = os.path.join(evidence_dir, rel)
        files[rel] = {"what": what, "exists": os.path.exists(p)}
        if files[rel]["exists"]:
            files[rel].update(sha256=file_sha(p), mtime_utc=utc_of(os.stat(p).st_mtime))

    def at(rel):
        return files[rel].get("mtime_utc")
    checks, out = {}, {"dir": evidence_dir, "files": files}
    if files["rule.txt"]["exists"]:
        checks["rule_txt_is_rule_text"] = files["rule.txt"]["sha256"] == RULE_SHA256
        checks["rule_txt_before_fixing"] = at("rule.txt") <= FIXED_AT_UTC
    orig = os.path.join(evidence_dir, "impl/regime_sensitivity.orig.py")
    if os.path.exists(orig):
        checks["orig_constants_block_is_pinned"] = \
            sha256_bytes(constants_block(orig).encode("utf-8")) == CONSTANTS_SHA256
    scored = "review/evidence/regime_sensitivity.scored.py"
    if files[scored]["exists"]:
        checks["scored_copy_is_scored_script"] = files[scored]["sha256"] == SCORED_SCRIPT_SHA256
        checks["scored_script_last_written_before_first_row"] = at(scored) < first_row
    plan = "review/evidence/p1a_regime_sensitivity.fixed.md"
    if files[plan]["exists"]:
        checks["plan_copy_is_fixed_plan"] = files[plan]["sha256"] == PLAN_SHA256
        checks["plan_last_written_at_fixing"] = at(plan) == FIXED_AT_UTC
    comp = os.path.join(ROWS, COMPOSITIONS)
    comp_at = utc_of(os.stat(comp).st_mtime) if os.path.exists(comp) else None
    if comp_at:
        checks["seed11_first_drawn_after_fixing"] = comp_at > FIXED_AT_UTC
        checks["compositions_file_is_the_regimes_stages"] = \
            file_sha(comp) == state["regimes"]["compositions"]["sha256"]
        if os.path.exists(orig):
            checks["constants_copied_before_seed11_drawn"] = at("impl/regime_sensitivity.orig.py") < comp_at
    dry = os.path.join(evidence_dir, "impl/dry_out.json")
    if os.path.exists(dry):
        with open(dry) as f:
            d = json.load(f)
        s = d.get("summary") or {}
        out["dry_run"] = {"utc": (s.get("provenance") or {}).get("utc"),
                          "script": (s.get("provenance") or {}).get("script"),
                          "rows_present": s.get("scored"), "rows_total": sum((s.get("scored") or {}).values()),
                          "row_scripts": (s.get("rows") or {}).get("script_digests"),
                          "printed": "structure, the checks and the outcome",
                          "outcome_printed": (d.get("rule") or {}).get("outcome")}
        checks["dry_run_on_scored_script"] = out["dry_run"]["script"] == SCORED_SCRIPT_SHA256[:16] \
            and out["dry_run"]["row_scripts"] == [SCORED_SCRIPT_SHA256[:16]]
    log = os.path.join(evidence_dir, "impl/pytest_out.txt")
    if os.path.exists(log):
        with open(log) as f:
            pos = first_failure_position(f.read())
        tests = collected_tests()
        name = tests[pos - 1] if pos and pos <= len(tests) else None
        before = sorted({t.split("::")[0] for t in tests[:pos]}) if pos else []
        log_at = os.stat(log).st_mtime
        out["first_pytest"] = {
            "log_mtime_utc": utc_of(log_at), "first_failure_position": pos, "test_at_position_now": name,
            "test_files_up_to_it": before,
            "those_files_unchanged_since_log": all(os.stat(os.path.join(ROOT, t)).st_mtime < log_at for t in before),
            "reading": "the failing test is the one at that position now if no test file up to it changed "
                       "after the log; it was run with default BLAS threads while the reproduce stage ran"}
    tl = [(FIXED_AT_UTC, "plan and rule fixed (FIXED_AT_UTC, self-reported)"),
          (at("rule.txt"), "planner's rule text last written (scratch rule.txt)"),
          (at(plan), "plan file last written (copied with its mtime)"),
          (at("impl/regime_sensitivity.orig.py"), "implementer's copy of the planner's file (constants block)"),
          (comp_at, "scoring seed first drawn: the regimes stage's composition file (no predictor)"),
          (state["regimes"]["provenance"]["utc"], "regimes stage written"),
          (state["smcal"]["provenance"]["utc"], "smcal stage written"),
          (at("impl/reproduce_out.txt"), "first reproduce run ends: legacy fails at 1e-9 (D1 found)"),
          (state["reproduce"]["provenance"]["utc"], "reproduce with D1's fix written (passed)"),
          (at(scored), "script last written (copied with its mtime)"),
          (first_row, "first scoring-seed row"),
          ((out.get("dry_run") or {}).get("utc"), "dry run of summarise on a partial set"),
          (last_row, "last scoring-seed row"),
          (state["summary"]["provenance"]["utc"], "summarise and the rule applied")]
    out["timeline"] = [{"utc": u, "event": e} for u, e in sorted((x for x in tl if x[0]), key=lambda x: x[0])]
    out["checks"] = checks
    out["nature"] = ("local file mtimes, scratch copies and self-recorded time stamps; none is an outside "
                     "time stamp such as a commit made before scoring")
    return out


def review_provenance(state, rows, evidence_dir=None):
    """Which code wrote what and when; D1's timing and both readings; the lock's evidence."""
    flat = sorted((r for R in REGIMES for r in rows[R].values()), key=lambda r: r["utc"])
    first, last = flat[0]["utc"], flat[-1]["utc"]
    stages = {}
    for s in ("smcal", "regimes", "reproduce", "summary"):
        p = (state.get(s) or {}).get("provenance") or {}
        stages[s] = {"script": p.get("script"), "utc": p.get("utc")}
    stages["lock"] = {"checked_utc": (state.get("lock") or {}).get("checked_utc")}
    row_scripts = sorted({r["script"] for r in flat})
    fixed, amend = plan_parts()
    script = {"current_sha256": file_sha(os.path.abspath(__file__)), "scored_sha256": SCORED_SCRIPT_SHA256,
              "cutting_review_blocks_gives_scored": sha256_bytes(scored_source().encode("utf-8")) == SCORED_SCRIPT_SHA256,
              "rows_record_scored": row_scripts == [SCORED_SCRIPT_SHA256[:16]],
              "summary_records_scored": stages["summary"]["script"] == SCORED_SCRIPT_SHA256[:16],
              "stage_scripts": {s: v.get("script") for s, v in stages.items() if v.get("script")},
              "note": "smcal and regimes ran under an earlier version of this file, reproduce under another, "
                      "every row and the summary under the scored one; review.rerun re-runs the three under "
                      "this file"}
    rows_info = {"n": len(flat), "first_utc": first, "last_utc": last, "scripts": row_scripts,
                 "seeds": sorted({r["seed"] for r in flat}), "shards": sorted({r["shard"] for r in flat}),
                 "library_digest_sets": len({json.dumps(r["lib"], sort_keys=True) for r in flat})}
    rep, rule = state.get("reproduce") or {}, state.get("rule") or {}
    leg = [c for c in rep.get("checks", []) if c["check"].startswith("legacy")]
    half = 0.5 * 10.0 ** -LEGACY_STORED_DECIMALS
    lit = rule.get("literal_reading") or {}
    holds = {X: v["holds"] for X, v in (rule.get("table") or {}).items()}
    d1 = {"deviation": DEVIATIONS[0]["id"],
          "first_reproduce_failed_at_planned_tolerance": rep.get("planned_tolerances_ok") is False,
          "reproduce_with_fix_utc": stages["reproduce"]["utc"], "first_row_utc": first,
          "fixed_before_first_row": bool(stages["reproduce"]["utc"] < first),
          "legacy_max_abs_brier": [c["max_abs_brier"] for c in leg],
          "rounding_half_unit": half,
          "all_within_rounding": all(c["max_abs_brier"] <= half for c in leg),
          "equal_at_stored_precision": all(c["equal_at_stored_precision"] for c in leg),
          "outcome_with_d1": rule.get("outcome"), "candidates_with_d1": rule.get("candidates"),
          "outcome_literal": lit.get("outcome"), "candidates_literal": lit.get("candidates"),
          "configs_meeting_a": sorted(X for X, h in holds.items() if h["a"]),
          "configs_meeting_a_to_e": sorted(X for X, h in holds.items() if all(h[k] for k in "abcde")),
          "reading": "D1 changes only the reproduce check's comparison; (f) is the only condition it can "
                     "touch, and no config meets (a) to (e), so neither reading has a candidate"}
    out = {"script": script, "stages": stages, "rows": rows_info, "d1": d1,
           "lock_now": lock_checks(),
           "plan": {"fixed_sha256": sha256_bytes(fixed), "fixed_is_pinned": sha256_bytes(fixed) == PLAN_SHA256,
                    "amendments_bytes": len(amend),
                    "amendments_sha256": sha256_bytes(amend) if amend else None}}
    if evidence_dir:
        ev = review_evidence(evidence_dir, state, first, last)
        out["dry_run"] = ev.pop("dry_run", None)
        out["first_pytest"] = ev.pop("first_pytest", None)
        out["evidence"] = ev
    else:
        prev = (state.get("review") or {}).get("provenance") or {}
        for k in ("evidence", "dry_run", "first_pytest"):
            if prev.get(k) is not None:
                out[k] = prev[k]
        out["evidence_from_an_earlier_review_run"] = "evidence" in out
    return out


def review_levels(state, rows):
    """Realised levels of the scored runs beside their targets, with the share below LOW_CUT
    and the pseudo-benchmarks that carry it."""
    from collections import Counter
    out = {}
    for R in REGIMES:
        app = [(p["bench"], p["parent"], float(p["logit"])) for i in sorted(rows[R]) for p in rows[R][i]["meta"]]
        L = np.array([x[2] for x in app])
        r = realised(L)
        low = [x for x in app if x[2] < LOW_CUT]
        by = defaultdict(int)
        for b, _, _ in low:
            by[b] += 1
        top = sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))
        r["share_below_-4"] = float(np.mean(L < LOW_CUT))
        r["below_-4"] = {"n": len(low), "pseudo_benchmarks": len(by), "by_pseudo_benchmark": dict(top),
                         "by_parent": dict(sorted(Counter(x[1] for x in low).items())),
                         "top_two": [b for b, _ in top[:2]],
                         "top_two_share": sum(c for _, c in top[:2]) / len(low) if low else None}
        sr = (state.get("summary") or {}).get("realised", {}).get(R) or {}
        r["equals_summary"] = bool(sr) and abs(sr["mean"] - r["mean"]) < 1e-12 and abs(sr["sd"] - r["sd"]) < 1e-12
        if R in TARGETS:
            r["target"] = TARGETS[R]
            r["minus_target"] = {k: r[k] - v for k, v in TARGETS[R].items() if k != "cut"}
        cal = (state.get("regimes") or {}).get("cal", {}).get(R)
        if cal:
            r["cal_seed"] = {k: cal[k] for k in ("mean", "sd", "share_below", "rest_mean")}
        out[R] = r
    out["feedback_regimes_realised_mean"] = {R: out[R]["mean"] for R in FEEDBACK_REGIMES}
    out["test_like_by_share_below_-3"] = [[R, out[R]["share_below"]]
                                          for R in sorted(TESTLIKE, key=lambda R: out[R]["share_below"])]
    return out


def review_headline(state):
    """X minus SHIP from summary.vs_ship[R][X].exact, and the scope of the 'within
    +-0.0006 of SHIP' and cluster-SE sentences."""
    vs = state["summary"]["vs_ship"]
    keys = ("D", "run_se", "cluster_se", "strat_se", "U95")
    D = {R: {X: {k: vs[R][X]["exact"][k] for k in keys} for X in vs[R]} for R in REGIMES}
    hier = [X for X in CONFIGS if X != SHIP and CONFIGS[X]["kind"] in ("hier", "ebhier")]
    worst = {X: max(abs(D[R][X]["D"]) for R in FEEDBACK_REGIMES) for X in CONFIGS if X != SHIP}
    near = [X for X in hier if round(worst[X], 4) <= NEAR_SHIP]
    se = {X: [min(D[R][X]["cluster_se"] for R in FEEDBACK_REGIMES),
              max(D[R][X]["cluster_se"] for R in FEEDBACK_REGIMES)] for X in hier}
    near_se = [se[X] for X in near]
    return {"source": "summary.vs_ship[R][X].exact: X minus ship, unrounded",
            "D": D, "feedback_max_abs_D": worst,
            "near_ship": {"criterion": "|D| at four decimals at most 0.0006 in READING and in AUDIT",
                          "hier_configs": near, "other_hier_configs": [X for X in hier if X not in near]},
            "hier_cluster_se_in_feedback_regimes": se,
            "near_ship_cluster_se_range": [min(v[0] for v in near_se), max(v[1] for v in near_se)] if near_se else None,
            "hier_cluster_se_range": [min(v[0] for v in se.values()), max(v[1] for v in se.values())]}


def review_smcal_guard(state):
    """The smcal guard against the current legacy Predictor: re-score it on the guard's
    runs, compare with the stored reference, and recompute every grid point's loss."""
    with open(LC_JSON) as f:
        lc_raw = json.load(f)["raw"]
    LC.register_chunks(LC.grid_chunks())
    names = [smcal_name(*g) for g in SMCAL_GRID]
    stored = {(g["n0"], g["m0"]): g for g in state["smcal"]["grid"]}
    per, loss_now, loss_redo = {}, {}, {}
    t0 = time.time()
    for _, seed, w, rs in SMCAL_GUARD_RUNS:
        reg = {"benchmark": "r1b", "pair": "r1p"}[w]
        assert LC.REGIMES[reg] == ("r1", seed, w)
        grid, now_run, old_run, off, on, meta_bad = [], [], [], [], [], []
        for i in rs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run = LC.draw(reg, i)
                slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
                got, _ = LC.evaluate(slots, cps, names, smgrid_factory())
                mk = Maker("legacy", run)
                lg, _ = LC.evaluate(slots, cps, ["legacy"], mk)
                mk.harvest()
            meta = describe(reg, run)
            st_meta = lc_raw.get(f"meta|{reg}|{i}")
            if st_meta is None or [SC.meta_key(p) for p in st_meta] != [SC.meta_key(p) for p in meta]:
                meta_bad.append(i)
            grid.append(np.array([[LC.alc(x["brier"]) for x in got[n]] for n in names]).mean(1))
            now = np.array([LC.alc(x["brier"]) for x in lg["legacy"]])
            old = np.asarray(LC.rows_of(lc_raw, reg, i, LC.PRED)["alc"], float)
            now_run.append(float(now.mean()))
            old_run.append(float(old.mean()))
            for p, a, b in zip(meta, now, old):
                (on if p["parent"] == "matharena" else off).append(float(a - b))
            del got, lg, mk, run, slots, cps
            gc.collect()
        G, now_run, old_run = np.array(grid), np.array(now_run), np.array(old_run)
        loss_now[reg] = (G - now_run[:, None]).mean(0)
        loss_redo[reg] = (G - old_run[:, None]).mean(0)
        d = now_run - old_run
        per[reg] = {"runs": len(rs), "meta_mismatch": meta_bad,
                    "legacy_now_minus_stored": {"mean": float(d.mean()),
                                                "run_se": float(d.std(ddof=1) / math.sqrt(len(d))),
                                                "share_runs_lower": float(np.mean(d < 0)),
                                                "per_run": d.tolist()},
                    "pairs_off_matharena": {"n": len(off), "max_abs": max(map(abs, off), default=0.0)},
                    "pairs_matharena": {"n": len(on), "mean": float(np.mean(on)) if on else None,
                                        "max_abs": max(map(abs, on), default=0.0)}}
        print(f"  review smcal_guard: {reg} {len(rs)} runs, legacy now minus stored "
              f"{d.mean():+.5f} ({time.time() - t0:.0f}s)", flush=True)
    regs = list(per)
    guard_now = {g: [float(loss_now[r][j]) for r in regs] for j, g in enumerate(SMCAL_GRID)}
    redo = max(abs(float(loss_redo[r][j]) - stored[g]["guard_loss"][r])
               for j, g in enumerate(SMCAL_GRID) for r in regs)
    sel = {g: stored[g]["selection_ALC"] for g in SMCAL_GRID}
    chosen_now, failed_now, passing_now = select_smcal(sel, guard_now)
    best = min(SMCAL_GRID, key=lambda g: max(guard_now[g]))
    return {"note": SMCAL_GUARD_NOTE, "guard": SMCAL_GUARD, "per_weighting": per,
            "stored_guard_recomputed_max_abs": redo,
            "grid": [{"n0": g[0], "m0": g[1], "loss_stored": stored[g]["guard_loss"],
                      "loss_now": dict(zip(regs, guard_now[g]))} for g in SMCAL_GRID],
            "min_guard_loss_now": {r: float(min(guard_now[g][k] for g in SMCAL_GRID)) for k, r in enumerate(regs)},
            "min_guard_loss_stored": state["smcal"]["min_guard_loss"],
            "best_point_now": {"n0": best[0], "m0": best[1], "loss_now": dict(zip(regs, guard_now[best])),
                               "loss_stored": stored[best]["guard_loss"]},
            "passing_now": [list(g) for g in passing_now], "guard_failed_now": bool(failed_now),
            "chosen_now": list(chosen_now), "chosen_stored": state["smcal"]["chosen"],
            "selection_unchanged": list(chosen_now) == list(state["smcal"]["chosen"])}


def review_rerun(state, work_dir=None):
    """smcal, regimes (without --grid) and reproduce re-run under this file into a scratch
    directory (OUT and ROWS redirected), compared field by field with the stored stages."""
    import tempfile
    global OUT, ROWS
    work = os.path.abspath(work_dir or tempfile.mkdtemp(prefix="p1a_rerun_"))
    out_tmp, rows_tmp = os.path.join(work, "regime_sensitivity.json"), os.path.join(work, "rows")
    if os.path.commonpath([work, ROOT]) == ROOT:
        raise SystemExit("review rerun: the work directory must lie outside the repository")
    os.makedirs(work, exist_ok=True)
    with open(out_tmp, "w") as f:
        json.dump({"lock": state["lock"]}, f)
    keep = (OUT, ROWS)
    try:
        OUT, ROWS = out_tmp, rows_tmp
        stage_smcal()
        stage_regimes(grid=False)
        stage_reproduce()
        new = load_state(out_tmp)
    finally:
        OUT, ROWS = keep
    skip = {"provenance", "wall_s", "secs", "run_s", "timing", "estimate"}
    res = {"work_dir": work}
    for s, extra in (("smcal", ()), ("regimes", ("grid", "compositions")), ("reproduce", ())):
        nums, bad = compare_json(state[s], new[s], skip | set(extra))
        res[s] = {"stored_script": state[s]["provenance"]["script"], "stored_utc": state[s]["provenance"]["utc"],
                  "numbers_compared": len(nums), "max_abs_diff": max(nums) if nums else None,
                  "other_differences": bad[:50], "identical": not bad and all(x == 0.0 for x in nums),
                  "not_compared": sorted(skip | set(extra))}
    res["regimes"]["compositions_sha256_equal"] = \
        state["regimes"]["compositions"]["sha256"] == new["regimes"]["compositions"]["sha256"]
    res["all_identical"] = all(res[s]["identical"] for s in ("smcal", "regimes", "reproduce")) \
        and res["regimes"]["compositions_sha256_equal"]
    return res


def review_rescore(state, rows):
    """The last planned run of every regime re-scored, every config, against its row."""
    smcal = tuple(state["smcal"]["chosen"])
    tasks = []
    for R, i in RESCORE_TASKS:
        t0 = time.time()
        row = rows[R][i]
        key, run = draw(R, SCORE_SEED, i)
        meta = describe(key, run)
        got = score_run(run, list(CONFIGS), smcal)
        per = {}
        for c in CONFIGS:
            g, s = got[c], row["res"][c]
            a, b = np.asarray(g["b"], float), np.asarray(s["b"], float)
            per[c] = {"max_abs_brier": float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf"),
                      "ece_equal": bool(np.array_equal(np.asarray(g["ece"], float), np.asarray(s["ece"], float))),
                      "q0_equal": bool(np.array_equal(np.asarray(g["q0"], float), np.asarray(s["q0"], float))),
                      "counters_equal": all(g[k] == s[k] for k in ("calls", "failures", "unconverged"))}
            if "eb" in g or "eb" in s:
                ge, se_ = [list(x) for x in g.get("eb", [])], [list(x) for x in s.get("eb", [])]
                per[c].update(eb_entries=len(se_), eb_equal_as_stored=ge == se_,
                              eb_equal_sorted=sorted(ge) == sorted(se_))
        tasks.append({"regime": R, "i": i, "pairs": len(meta),
                      "meta_equal": json.loads(json.dumps(meta)) == row["meta"],
                      "per_config": per, "secs": round(time.time() - t0, 1)})
        print(f"  review rescore: {R} {i}: max |dBrier| "
              f"{max(v['max_abs_brier'] for v in per.values()):.2g} ({time.time() - t0:.0f}s)", flush=True)
        del got, run
        gc.collect()
    vals = [v for t in tasks for v in t["per_config"].values()]
    eb = [v for v in vals if "eb_entries" in v]
    return {"tasks_rule": "the last planned run of every regime", "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "tasks": tasks, "max_abs_brier": max(v["max_abs_brier"] for v in vals),
            "all_numbers_equal": all(v["max_abs_brier"] == 0.0 and v["ece_equal"] and v["q0_equal"]
                                     and v["counters_equal"] for v in vals)
            and all(t["meta_equal"] for t in tasks),
            "eb_traces": {"compared": len(eb), "equal_as_stored": sum(v["eb_equal_as_stored"] for v in eb),
                          "equal_sorted": sum(v["eb_equal_sorted"] for v in eb)}}


def stage_review(parts, evidence_dir=None, work_dir=None):
    state = load_state()
    require(state, "lock", "smcal", "regimes", "reproduce")
    if not (state.get("rule") or {}).get("applied"):
        raise SystemExit("review reads a summarised study; run summarise first")
    rows = load_rows() if set(parts) & {"provenance", "levels", "rescore"} else None
    for part in parts:
        t0 = time.time()
        if part == "provenance":
            v = review_provenance(state, rows, evidence_dir)
        elif part == "levels":
            v = review_levels(state, rows)
        elif part == "headline":
            v = review_headline(state)
        elif part == "notes":
            v = {"notes": REVIEW_NOTES, "smcal_guard_note": SMCAL_GUARD_NOTE}
        elif part == "smcal_guard":
            v = review_smcal_guard(state)
        elif part == "rerun":
            v = review_rerun(state, work_dir)
        else:
            v = review_rescore(state, rows)
        v["ran"] = {"script": SCRIPT_SHA, "utc": utcnow(), "wall_s": round(time.time() - t0, 1),
                    "scored_stage_code": sha256_bytes(scored_source().encode("utf-8")) == SCORED_SCRIPT_SHA256}
        save_review(part, v)
        if part == "smcal_guard":
            note = {"note": SMCAL_GUARD_NOTE, "see": "review.smcal_guard", "added_utc": utcnow(),
                    "script": SCRIPT_SHA}
            locked_update(lambda s: s["smcal"].__setitem__("guard_reference_note_after_review", note))
        print(f"review: {part} written ({time.time() - t0:.0f}s)", flush=True)


def review_main(argv):
    ap = argparse.ArgumentParser(prog="regime_sensitivity.py review",
                                 description="the review stage, added after the study was scored")
    ap.add_argument("--parts", default=",".join(LIGHT_PARTS),
                    help=f"comma-separated, of {', '.join(REVIEW_PARTS)} (default: the light ones)")
    ap.add_argument("--evidence-dir", default=None, help="provenance: the study's scratch directory")
    ap.add_argument("--work-dir", default=None, help="rerun: a scratch directory outside the repository")
    a = ap.parse_args(argv)
    parts = [p for p in a.parts.split(",") if p]
    bad = [p for p in parts if p not in REVIEW_PARTS]
    if bad:
        ap.error(f"unknown parts {bad}")
    stage_review(parts, a.evidence_dir, a.work_dir)


# <<< after review
# --- driver --------------------------------------------------------------------------------

def stage_lock():
    res = lock_checks()
    for k, v in res.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    print(f"fixed at {FIXED_AT_UTC} (HEAD {HEAD_AT_FIXING}); rule {RULE_SHA256[:16]}; "
          f"plan {PLAN_SHA256[:16]}; constants {CONSTANTS_SHA256[:16]}")
    save_stage("lock", {"ok": all(res.values()), "checks": res, "fixed_at_utc": FIXED_AT_UTC,
                        "head_at_fixing": HEAD_AT_FIXING, "rule_sha256": RULE_SHA256,
                        "plan_sha256": PLAN_SHA256, "constants_sha256": CONSTANTS_SHA256,
                        "rule_text": RULE_TEXT, "checked_utc": utcnow()})
    save_stage("plan", {"configs": CONFIGS, "regimes": {R: list(v) for R, v in REGIMES.items()},
                        "n_runs": N_RUNS, "score_seed": SCORE_SEED, "cal_seed": CAL_SEED,
                        "split_scope": SPLIT_SCOPE, "boots": BOOTS, "boot_seed": BOOT_SEED,
                        "parents": PARENTS, "rss_guard_gb": RSS_GUARD_GB, "jobs": JOBS,
                        "runs_cut": "none: the plan fixes 80 test-like and 60 public runs a regime "
                                    "and allows no change"})
    if not all(res.values()):
        raise SystemExit(1)


def main(argv=None):
    # >>> after review: `review` takes its own options (review_main)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["review"]:
        return review_main(argv[1:])
    # <<< after review
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", nargs="?", default="lock",
                    choices=("lock", "smcal", "regimes", "reproduce", "score", "summarise", "status"))
    ap.add_argument("--shard", default="0/1", help="k/n: score every n-th planned task from k")
    ap.add_argument("--grid", action="store_true", help="regimes: also re-run the knob search")
    a = ap.parse_args(argv)
    if a.stage == "lock":
        stage_lock()
    elif a.stage == "smcal":
        stage_smcal()
    elif a.stage == "regimes":
        stage_regimes(grid=a.grid)
    elif a.stage == "reproduce":
        stage_reproduce()
    elif a.stage == "score":
        k, n = (int(x) for x in a.shard.split("/"))
        if not 0 <= k < n:
            ap.error("--shard k/n with 0 <= k < n")
        stage_score((k, n))
    elif a.stage == "summarise":
        stage_summarise()
    else:
        stage_status()


if __name__ == "__main__":
    main()
