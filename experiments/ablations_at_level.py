"""S4: ablations of the shipped hier at its LEVEL on a fresh regime-sensitivity seed,
for the report only (docs/plans/p2_final_plan.md, section 3, check S4).

The check, as the plan fixes it (its S4 row is pinned byte for byte by S4_ROW_SHA256):
sigma_delta x0.85, x0.7, x0.5; line read off; line read off with x0.7; x0.85 with the
level mean preserved; wide35; wide50; eb_ship; equal and hidden-v weights. Declared
no-ship in advance: anything that would pass the RS rule of App C.2 is reported, not
shipped.

Design. P1a's machinery (experiments/regime_sensitivity.py, imported, never edited: its
constants block is digest-pinned) on a seed none of its stages drew: its seven regimes
(TUNED, READING, AUDIT, MIXTURE, FLAT test-like; R1B, R1P public R1), its run counts (80
test-like and 60 public runs a regime, so 520 tasks), its split scope 'pair', its
checkpoints (level_calibration.checkpoints / evaluate, a fresh instance per checkpoint),
its row format, its statistics (RegimeRows: level_calibration.Boot, 2,000 resamples,
seed 0) and its rule (apply_rule, RULE_TEXT's (a) to (f)), all read through the
imported module. Every config is scored on identical checkpoints, paired with ship.

Configs (CONFIGS). Every hier config is HierPredictor over level_calibration.bundle
(parent, 0.0), prior.build without the target's parent, with LEVEL set over it:
  ship              the shipped configuration (LEVEL mu0 -2.5, sigma_mu 2.5, as 0.5)
  legacy            the legacy Predictor: the RS rule's (d) and the TUNED consistency
                    check need it
  sd_x085, sd_x07, sd_x05
                    sigma_delta times 0.85, 0.7, 0.5 of the leave-one-parent-out fit, as
                    App B.3's sigma_delta x0.5 and x2 were scored (experiments/
                    hier_eval.py, ('scale', 'sigma_delta', k))
  noline            Flags.line off (the target's a'x left Gaussian, plain Laplace)
  sd_x07_noline     both
  sd_x085_mp        sd_x085 with mu0 MU0_MP_X, the global mu0 that restores ship's mean
                    budget-0 prediction (the calib stage)
  wide35, wide50    P1a's: sigma_mu 3.5 and 5.0
  eb_ship           P1a's: hierEB-cs,tm=2.0 over LEVEL
  sd085_abs, sd07_abs, sd05_abs, sd07_abs_noline, sd085_abs_mp
                    the scratch pilots' definitions (pilot2's sd085, sd07, sd05, sd07nl,
                    the critic's sd085mp at mu0 -2.489), kept so their seed-11 numbers can
                    be checked on a fresh seed: sigma_delta set to 2.025, 1.667, 1.191,
                    that is 0.85, 0.7, 0.5 times 2.382, the fit on all five public
                    benchmarks (paiec.hier.Hyper's default), for every parent. That value
                    includes the scored parent, so these five are not leave-one-parent-out;
                    the sd_x family is
The leave-one-parent-out sigma_delta is 1.70 (matharena held out), 2.96
(multi_swebench), 2.41 (real_webagents), 2.56 (researchcodebench), 2.38 (swe_rebench), so
the pilots' 2.025 widens matharena's by 19% and narrows multi_swebench's by 32%.

Weights. 'equal': P1a's, each pair appearance 1 / its run's pair count, so D is the mean
over runs of the run's pair-mean difference. 'hidden_v': the same times
HV_SHARE[bin] / share[bin], where bin is the pair's v = p(1 - p) at its evaluated rate
(cuts HV_BINS) and share the regime's appearance share in that bin, so the replica takes
the formative runs' composition in v. HV_SHARE is read from the organisers' formative
tables (results/formative/run1-3.txt): v = B0 - ECE0^2 of each of the 26 appearances,
clipped to [0, 0.25], binned; only the three counts are used and stored (HV_COUNTS pins
them), never an id. Both weightings use the same cluster-bootstrap resamples.

Seeds. SEED is fresh: none of P1a's used seeds, its calibration seed or its scoring seed.
The calib stage draws P1a's scoring seed (11) runs 0-11 of TUNED, READING and AUDIT, as
the critic's calibration did, and uses budget-0 predictions only (no label is read). A
dry run (--dry-run DIR) scores DRY_SEED instead, so no run of SEED is scored before the
full job, and it can only write outside the repository.

Stages, each a subcommand; every stage after `lock` re-runs the lock and refuses to run
when it fails:
  lock       S4's row in the plan, LEVEL against submission/model.py, P1a's lock, this
             file's constants block against CONSTANTS_SHA256
  calib      both mean-preserving mu0 recomputed (budget 0, seed 11 runs 0-11 of TUNED,
             READING, AUDIT) against MU0_MP_ABS and MU0_MP_X, and every config's
             hyperparameters per parent
  regimes    every planned run's composition on SEED (no predictor), its realised level
             against P1a's CAL_SEED realisation (P1a's REALISE_TOL), the compositions file
  reproduce  this script's factory against P1a's stored rows (seed 11, the last planned
             run of every regime; ship, wide35, wide50, eb_ship, legacy; |dBrier| <= 1e-9)
  score      every planned task, every config on identical checkpoints, --shard k/n,
             resumable; one row file per task; exits 75 (resumable) past RSS_GUARD_GB
  summarise  statistics under both weightings, P1a's rule applied once (reported, not
             shipped), the TUNED consistency check, the checks -> OUT
  status     rows scored so far, per regime

Run (from the repository root):
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export PYTHONHASHSEED=0
  python experiments/ablations_at_level.py lock
  python experiments/ablations_at_level.py calib          # about 1 min
  python experiments/ablations_at_level.py regimes        # under 1 min
  python experiments/ablations_at_level.py reproduce      # about 2.5 min
  for k in 0 1 2; do (while :; do python experiments/ablations_at_level.py score \\
      --shard $k/3; [ $? -eq 75 ] || break; done) & done; wait     # 2.5 to 3 h
  python experiments/ablations_at_level.py summarise      # about 1 min
The score stage costs about 7.5 to 9 CPU-hours: 16 configs on 520 tasks, about 23 s of
CPU per 1,000 evaluation calls a config set makes (about 2,740 calls a task); the dry
run of 2026-10-06 measured it on a machine under heavy outside load.
A dry run: the same with `--dry-run DIR --runs 2` after the stage name.

Output: results/ablations_at_level.json (OUT); rows in data/ablations_at_level_rows/
(ROWS, gitignored). tests/test_ablations_at_level.py covers the arithmetic on synthetic
data.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.regime_sensitivity as RS  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.hier import HierPredictor, Hyper  # noqa: E402
from paiec.predict import Predictor  # noqa: E402

# --- the plan ---------------------------------------------------------------------------

#: when this plan was fixed (UTC): before any run of SEED was scored or drawn
FIXED_AT_UTC = "2026-10-06T14:12:18Z"
PLAN_PATH = "docs/plans/p2_final_plan.md"
#: sha256 of the plan's S4 table row (the line opening "| S4 |", newline included)
S4_ROW_SHA256 = "f0a9af0288f3bf85747bd4e4764c5e93a320e9e43215c0f7e0dbb4e2c2859b20"
S4_READING = ("Declared no-ship in advance: a config that P1a's rule (App C.2) lists as a "
              "candidate is reported as one that would pass, and nothing ships.")

#: the scoring seed: np.random.default_rng([SEED, i]) for run i, as P1a draws every run
SEED = 20261006
#: the dry run's seed: never the scoring seed
DRY_SEED = 99
REGIMES = tuple(RS.REGIMES)                       # TUNED READING AUDIT MIXTURE FLAT R1B R1P
N_RUNS = dict(RS.N_RUNS)                          # 80 test-like, 60 public
SPLIT_SCOPE = RS.SPLIT_SCOPE                      # 'pair'
SHIP = RS.SHIP
LEVEL = dict(RS.LEVEL)
#: sigma_delta fitted on all five public benchmarks (paiec.hier.Hyper's default): the
#: pilots' absolute values are SD_FACTORS times it, rounded to three decimals
SD_ALL_PUBLIC = 2.382
SD_FACTORS = (0.85, 0.7, 0.5)
SD_ABS = (2.025, 1.667, 1.191)
#: mean-preserving mu0 (the calib stage reproduces both to MU0_TOL): sd085_abs's is the
#: critic pilot's -2.488993525 at four decimals; sd_x085's, -2.471505 at four decimals,
#: was computed by the same procedure on the same runs (a dry-run calib) before any run of
#: SEED was drawn
MU0_MP_ABS = -2.489
MU0_MP_X = -2.4715
MU0_TOL = 5e-5
#: calib: the critic's runs, budget 0 only, and the mu0 step of its secant
CALIB_SEED = RS.SCORE_SEED
CALIB_REGIMES = ("TUNED", "READING", "AUDIT")
CALIB_RUNS = range(0, 12)
CALIB_STEP = 0.1

CONFIGS = {
    "ship": {"kind": "hier", "set": dict(LEVEL), "scale": {}, "flags": {}},
    "legacy": {"kind": "predictor"},
    "sd_x085": {"kind": "hier", "set": dict(LEVEL), "scale": {"sigma_delta": 0.85}, "flags": {}},
    "sd_x07": {"kind": "hier", "set": dict(LEVEL), "scale": {"sigma_delta": 0.7}, "flags": {}},
    "sd_x05": {"kind": "hier", "set": dict(LEVEL), "scale": {"sigma_delta": 0.5}, "flags": {}},
    "noline": {"kind": "hier", "set": dict(LEVEL), "scale": {}, "flags": {"line": False}},
    "sd_x07_noline": {"kind": "hier", "set": dict(LEVEL), "scale": {"sigma_delta": 0.7},
                      "flags": {"line": False}},
    "sd_x085_mp": {"kind": "hier", "set": dict(LEVEL, mu0="MU0_MP_X"), "scale": {"sigma_delta": 0.85},
                   "flags": {}},
    "wide35": {"kind": "hier", "set": dict(RS.CONFIGS["wide35"]["override"]), "scale": {}, "flags": {}},
    "wide50": {"kind": "hier", "set": dict(RS.CONFIGS["wide50"]["override"]), "scale": {}, "flags": {}},
    "eb_ship": {"kind": "ebhier", "lc_name": RS.CONFIGS["eb_ship"]["lc_name"]},
    "sd085_abs": {"kind": "hier", "set": dict(LEVEL, sigma_delta=2.025), "scale": {}, "flags": {}},
    "sd07_abs": {"kind": "hier", "set": dict(LEVEL, sigma_delta=1.667), "scale": {}, "flags": {}},
    "sd05_abs": {"kind": "hier", "set": dict(LEVEL, sigma_delta=1.191), "scale": {}, "flags": {}},
    "sd07_abs_noline": {"kind": "hier", "set": dict(LEVEL, sigma_delta=1.667), "scale": {},
                        "flags": {"line": False}},
    "sd085_abs_mp": {"kind": "hier", "set": dict(LEVEL, mu0=MU0_MP_ABS, sigma_delta=2.025), "scale": {},
                     "flags": {}},
}
#: the plan's list, in its order (the rest are references and the pilots' definitions)
S4_LIST = ("sd_x085", "sd_x07", "sd_x05", "noline", "sd_x07_noline", "sd_x085_mp",
           "wide35", "wide50", "eb_ship")
REFERENCES = ("ship", "legacy")
PILOT_FAMILY = ("sd085_abs", "sd07_abs", "sd05_abs", "sd07_abs_noline", "sd085_abs_mp")
#: the scratch pilots' names for the pilot family (for the reader; nothing is read)
PILOT_NAMES = {"sd085_abs": "pilot2 sd085", "sd07_abs": "pilot2 sd07", "sd05_abs": "pilot2 sd05",
               "sd07_abs_noline": "pilot2 sd07nl", "sd085_abs_mp": "critic sd085mp",
               "noline": "pilot2 noline"}

#: hidden-v weights: bin cuts on v = p(1 - p), and the formative tables they are read from
HV_BINS = (0.10, 0.20)
FORMATIVE = ("results/formative/run1.txt", "results/formative/run2.txt", "results/formative/run3.txt")
#: appearances per bin in those tables (v below 0.10, 0.10 to 0.20, 0.20 and above)
HV_COUNTS = (5, 12, 9)

#: reproduction against P1a's stored rows: its scoring seed, the last planned run of every
#: regime, the configs it scored that are scored here
REPRO_TASKS = tuple((R, RS.N_RUNS[R] - 1) for R in REGIMES)
REPRO_CONFIGS = ("ship", "wide35", "wide50", "eb_ship", "legacy")
REPRO_TOL = 1e-9
RSS_GUARD_GB = RS.RSS_GUARD_GB
JOBS = 3

OUT = os.path.join(ROOT, "results", "ablations_at_level.json")
ROWS = os.path.join(ROOT, "data", "ablations_at_level_rows")
# --- end of the plan ----------------------------------------------------------------------

#: sha256 of the plan block above, from "# --- the plan" through the line closing it
CONSTANTS_SHA256 = "c9b7acfb3fdf1cd1164a1a4ea4ee83a64082e892b2e2775e310416a84964e567"
CONSTANTS_START = "# --- the plan ---"
CONSTANTS_END = "# --- end of the plan ---"

W6 = np.asarray(WEIGHTS, float)
#: P1a's own configs, as its module defines them (rs_configs rebinds RS.CONFIGS)
P1A_CONFIGS = RS.CONFIGS
COMPOSITIONS = RS.COMPOSITIONS
RESUME_EXIT = RS.RESUME_EXIT
RS_PATH = os.path.join(ROOT, "experiments", "regime_sensitivity.py")
PUBLIC_PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")

#: set by --dry-run: {"dir", "runs", "seed"}; None for the planned job
DRY = None

clean = RS.clean
utcnow = RS.utcnow
write_json = RS.write_json
file_sha = RS.file_sha
load_state = RS.load_state


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


SCRIPT_SHA = file_sha(os.path.abspath(__file__), 16)


# --- lock -----------------------------------------------------------------------------------

def constants_block(path=None):
    with open(path or os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    a = src.index(CONSTANTS_START)
    b = src.index(CONSTANTS_END, a)
    b = src.index("\n", b) + 1
    return src[a:b]


def s4_row(path=None):
    """The plan's S4 table row(s), each line with its newline."""
    with open(path or os.path.join(ROOT, PLAN_PATH), encoding="utf-8") as f:
        return [ln for ln in f.read().splitlines(keepends=True) if ln.startswith("| S4 |")]


def lock_checks():
    rows = s4_row()
    rs = RS.lock_checks()
    return {
        "plan_has_one_s4_row": len(rows) == 1,
        "s4_row_sha256": len(rows) == 1 and sha256_bytes(rows[0].encode("utf-8")) == S4_ROW_SHA256,
        "level_is_shipped": RS.model_level() == LEVEL,
        "ship_is_level": CONFIGS[SHIP]["set"] == LEVEL and not CONFIGS[SHIP]["scale"],
        "p1a_lock_holds": all(rs.values()),
        "constants_sha256": CONSTANTS_SHA256 is not None
        and sha256_bytes(constants_block().encode("utf-8")) == CONSTANTS_SHA256,
        "seed_fresh": SEED not in RS.USED_SEEDS["testlike"] + RS.USED_SEEDS["r1"]
        and SEED not in (RS.CAL_SEED, RS.SCORE_SEED, DRY_SEED),
        "sd_abs_is_factors_of_all_public": SD_ALL_PUBLIC == Hyper().sigma_delta and all(
            round(k * SD_ALL_PUBLIC, 3) == v for k, v in zip(SD_FACTORS, SD_ABS)),
        "mu0_mp_x_fixed": MU0_MP_X is not None,
        "s4_list_is_scored": set(S4_LIST + REFERENCES + PILOT_FAMILY) == set(CONFIGS),
        "regimes_are_p1as": REGIMES == tuple(RS.REGIMES) and N_RUNS == RS.N_RUNS,
    }


def require(state, *stages):
    res = lock_checks()
    if not all(res.values()):
        raise SystemExit(f"lock check failed: {[k for k, v in res.items() if not v]}")
    for s in stages:
        if not (state.get(s) or {}).get("ok"):
            raise SystemExit(f"stage '{s}' has not passed; run it first")


# --- io and provenance ----------------------------------------------------------------------

def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def inputs():
    """The files a stage reads, with their sha256."""
    paths = list(FORMATIVE) + [PLAN_PATH, "submission/model.py", "results/regime_sensitivity.json",
                               "experiments/regime_sensitivity.py", "experiments/level_calibration.py"]
    return {p: file_sha(os.path.join(ROOT, p)) for p in paths if os.path.exists(os.path.join(ROOT, p))}


def provenance():
    return {"script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "script_sha256": file_sha(os.path.abspath(__file__)),
            "script_sha16_at_start": SCRIPT_SHA,
            "regime_sensitivity_sha256": file_sha(RS_PATH),
            "head": git("rev-parse", "--short", "HEAD"),
            "dirty": git("status", "--porcelain", "paiec", "experiments", "submission", "tests", "results"),
            "lib": RS.lib_digests(), "inputs": inputs(),
            "command": "python " + " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), ROOT)] + sys.argv[1:])
            if sys.argv and sys.argv[0].endswith(".py") else None,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "dry_run": DRY,
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "utc": utcnow()}


def save_stage(name, value):
    return RS.save_stage(name, value, path=OUT)


def row_path(R, i, rows=None):
    return os.path.join(rows or ROWS, R, f"{i}.json")


def seed():
    return DRY["seed"] if DRY else SEED


def n_runs():
    return {R: min(DRY["runs"], N_RUNS[R]) for R in REGIMES} if DRY else dict(N_RUNS)


def plan_tasks():
    """Every planned (regime, run), run-index-major, as P1a orders its tasks."""
    nr = n_runs()
    top = max(nr.values())
    return [(R, i) for i in range(top) for R in REGIMES if i < nr[R]]


def shard_tasks(k, n):
    return [t for j, t in enumerate(plan_tasks()) if j % n == k]


@contextlib.contextmanager
def rs_configs():
    """P1a's statistics and rule (RegimeRows, apply_rule, latency, ranks, eb_traces) read
    its module's CONFIGS at call time: inside this block they read this study's."""
    keep = RS.CONFIGS
    RS.CONFIGS = CONFIGS
    try:
        yield
    finally:
        RS.CONFIGS = keep


# --- predictors -------------------------------------------------------------------------------

def mu0_value(v):
    return MU0_MP_X if v == "MU0_MP_X" else v


def config_hyper(cfg, hyper):
    """A hier config's Hyper from prior.build's: `set` replaces fields (LEVEL and any
    fixed value), then `scale` multiplies the fitted value of the fields it names."""
    sets = {k: (mu0_value(v) if k == "mu0" else v) for k, v in (cfg.get("set") or {}).items()}
    if any(v is None for v in sets.values()):
        raise ValueError(f"a config value is not fixed: {sets}")
    h = replace(hyper, **sets) if sets else hyper
    for f, k in sorted((cfg.get("scale") or {}).items()):
        if f in sets:
            raise ValueError(f"{f} is both set and scaled")
        h = replace(h, **{f: getattr(h, f) * float(k)})
    return h


class Maker:
    """One config's model factory on one run, as P1a's Maker: every call (one per
    checkpoint) builds fresh per-parent instances, dispatched on the anonymous
    benchmark_id; the previous checkpoint's are released after their counters (and EB
    traces) are read. Parents are built in sorted order."""

    def __init__(self, name, run, cfg=None):
        self.name, self.cfg = name, cfg if cfg is not None else CONFIGS[name]
        self.ids = T.anon_parents(run)
        self.k = -1
        self.live = []
        self.failures = self.unconverged = 0
        self.trace = []

    def build(self, parent):
        c = self.cfg
        if c["kind"] == "hier":
            prior, hyper = LC.bundle(parent, 0.0)
            return HierPredictor(prior, config_hyper(c, hyper), **c["flags"])
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
        ms = {par: self.build(par) for par in sorted(set(self.ids.values()))}
        self.live = list(ms.values())
        ids = self.ids
        return lambda inp, labeled=None: ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled)


def score_run(run, names, slots=None, cps=None):
    """Every config of `names` on the run's checkpoints, stored as P1a stores a config:
    per pair Brier, ECE and mean B0 prediction, call timing and counters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cps is None:
            slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
        out = {}
        for n in names:
            t0 = time.perf_counter()
            mk = Maker(n, run)
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
            gc.collect()
    return out


def warm():
    """Fit every parent's bundle and legacy prior once, outside any timed call."""
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for par in PUBLIC_PARENTS:
            LC.bundle(par, 0.0)
            LC.ppred(par)
    return round(time.time() - t0, 1)


# --- hidden-v weights -------------------------------------------------------------------------

_B0_ROW = re.compile(r"^\S+\s+\S+\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


def formative_v(text):
    """v = Brier - ECE^2 of every pair of one formative table's budget-0 block (only the
    numbers are read)."""
    parts = re.split(r"^Label budget: (\d+)\s*$", text, flags=re.M)
    if "0" not in parts:
        raise ValueError("no budget-0 block")
    block = parts[parts.index("0") + 1]
    out = []
    for ln in block.splitlines():
        m = _B0_ROW.match(ln)
        if m:
            out.append(float(m.group(2)) - float(m.group(3)) ** 2)
    return out


def v_bins(v):
    return np.digitize(np.clip(np.asarray(v, float), 0.0, 0.25), HV_BINS)


def hidden_shares(paths=FORMATIVE):
    """(counts per bin, shares) of the formative appearances' v."""
    vs = []
    for p in paths:
        with open(os.path.join(ROOT, p)) as f:
            vs += formative_v(f.read())
    cnt = np.bincount(v_bins(vs), minlength=len(HV_BINS) + 1)
    return cnt, cnt / cnt.sum()


def hv_weights(G, share_hidden):
    """Appearance weights of RegimeRows G under hidden-v weighting, and the replica's own
    counts per bin. A bin the regime does not reach gets no weight (recorded)."""
    p = np.array([q["p"] for m in G.metas for q in m], float)
    b = v_bins(p * (1 - p))
    cnt = np.bincount(b, minlength=len(HV_BINS) + 1)
    share = cnt / cnt.sum()
    f = np.where(share[b] > 0, np.asarray(share_hidden)[b] / np.where(share[b] > 0, share[b], 1.0), 0.0)
    return G.w * f, {"replica_counts": cnt.tolist(), "replica_share": share.tolist(),
                     "bins_without_appearances": [int(j) for j in np.flatnonzero(cnt == 0)]}


def weighted_diff(G, x, ref, w):
    """x minus ref under appearance weights w: the ratio estimator sum(w d) / sum(w), its
    cluster-bootstrap SE on G's resamples (pooled and within parent), the 95% bootstrap
    interval, per budget, per parent and the parent-level mean (P1a's PARENTS). With
    w = G.w it is RegimeRows.diff's D and SEs."""
    dA = np.concatenate([a - b for a, b in zip(G.A[x], G.A[ref])])
    dB = np.concatenate([a - b for a, b in zip(G.B[x], G.B[ref])])
    n = np.bincount(G.cl, w, minlength=G.K)
    nW, nWs = G.boot.W @ n, G.boot.Ws @ n

    def ratio(vals):
        v = np.bincount(G.cl, w * vals, minlength=G.K)
        return float(np.sum(w * vals) / np.sum(w)), (G.boot.W @ v) / nW, (G.boot.Ws @ v) / nWs

    D, dist, sdist = ratio(dA)
    se = float(dist.std(ddof=1))
    budgets = []
    for j, b in enumerate(BUDGETS):
        m, dj, _ = ratio(dB[:, j])
        budgets.append({"budget": b, "diff": m, "cluster_se": float(dj.std(ddof=1)), "alc_part": W6[j] * m})
    pq = {}
    for q in sorted(set(G.parent.tolist())):
        s = G.parent == q
        if np.sum(w[s]) > 0:
            pq[q] = float(np.sum(w[s] * dA[s]) / np.sum(w[s]))
    multi = [pq[q] for q in RS.PARENTS if q in pq]
    return {"D": D, "cluster_se": se, "strat_se": float(sdist.std(ddof=1)),
            "U95": D + RS.RULE["z95"] * se,
            "pct": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
            "budgets": budgets, "Pq": pq, "PL": float(np.mean(multi)) if multi else None}


# --- stage: lock --------------------------------------------------------------------------------

def stage_lock():
    res = lock_checks()
    for k, v in res.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    save_stage("lock", {"ok": all(res.values()), "checks": res, "fixed_at_utc": FIXED_AT_UTC,
                        "s4_row": s4_row()[0] if len(s4_row()) == 1 else None,
                        "s4_row_sha256": S4_ROW_SHA256, "constants_sha256": CONSTANTS_SHA256,
                        "s4_reading": S4_READING, "checked_utc": utcnow(), "provenance": provenance()})
    save_stage("plan", {"seed": seed(), "scoring_seed": SEED, "dry_run": DRY, "regimes":
                        {R: list(RS.REGIMES[R]) for R in REGIMES}, "n_runs": n_runs(),
                        "split_scope": SPLIT_SCOPE, "configs": CONFIGS, "s4_list": S4_LIST,
                        "references": REFERENCES, "pilot_family": PILOT_FAMILY, "pilot_names": PILOT_NAMES,
                        "mu0_mp": {"abs": MU0_MP_ABS, "x": MU0_MP_X, "tol": MU0_TOL},
                        "sd_all_public": SD_ALL_PUBLIC, "sd_factors": SD_FACTORS, "sd_abs": SD_ABS,
                        "hv_bins": HV_BINS, "hv_counts": HV_COUNTS, "formative": FORMATIVE,
                        "boots": RS.BOOTS, "boot_seed": RS.BOOT_SEED, "jobs": JOBS,
                        "rss_guard_gb": RSS_GUARD_GB})
    if not all(res.values()):
        raise SystemExit(1)


# --- stage: calib ---------------------------------------------------------------------------------

def mean_b0(cfgs, runs):
    """Mean budget-0 prediction over the evaluated targets of each run, then over runs,
    for every config of `cfgs` ({name: config}): no label is read."""
    acc = {n: [] for n in cfgs}
    for R, i in runs:
        key, run = RS.draw(R, CALIB_SEED, i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
            b, labeled, inp, index = cps[0]
            assert b == 0 and not labeled
            for n, c in cfgs.items():
                fn = Maker(n, run, c)()
                P = np.array([fn(x, []) for x in inp], float)[index]
                acc[n].append(float(P.mean()))
    return {n: float(np.mean(v)) for n, v in acc.items()}, acc


def secant_mu0(m_ship, m_base, m_up, base_mu0=LEVEL["mu0"], step=CALIB_STEP):
    """The mu0 at which the base config's mean B0 prediction equals ship's, on the line
    through (base_mu0, m_base) and (base_mu0 + step, m_up): the critic's calibration."""
    slope = (m_up - m_base) / step
    return base_mu0 + (m_ship - m_base) / slope, slope


def stage_calib():
    lk = lock_checks()
    if MU0_MP_X is None or CONSTANTS_SHA256 is None:
        # preparation only: MU0_MP_X is computed here before it is fixed, and only in a dry run
        if not DRY:
            raise SystemExit("MU0_MP_X and CONSTANTS_SHA256 are not fixed: run calib with --dry-run first")
        lk = {k: v for k, v in lk.items() if k not in ("mu0_mp_x_fixed", "constants_sha256")}
    if not all(lk.values()):
        raise SystemExit(f"lock check failed: {[k for k, v in lk.items() if not v]}")
    t0 = time.time()
    runs = [(R, i) for R in CALIB_REGIMES for i in CALIB_RUNS]
    up = LEVEL["mu0"] + CALIB_STEP
    cfgs = {"ship": CONFIGS["ship"],
            "abs": dict(CONFIGS["sd085_abs"]),
            "abs_up": dict(CONFIGS["sd085_abs"], set=dict(CONFIGS["sd085_abs"]["set"], mu0=up)),
            "x": dict(CONFIGS["sd_x085"]),
            "x_up": dict(CONFIGS["sd_x085"], set=dict(CONFIGS["sd_x085"]["set"], mu0=up))}
    m, per_run = mean_b0(cfgs, runs)
    mu_abs, s_abs = secant_mu0(m["ship"], m["abs"], m["abs_up"])
    mu_x, s_x = secant_mu0(m["ship"], m["x"], m["x_up"])
    # the mean-preserving configs as fixed, scored on the same runs: their mean B0 prediction
    fixed = {}
    if MU0_MP_X is not None:
        fm, _ = mean_b0({"sd_x085_mp": CONFIGS["sd_x085_mp"], "sd085_abs_mp": CONFIGS["sd085_abs_mp"]}, runs)
        fixed = {k: {"mean_b0": v, "minus_ship": v - m["ship"]} for k, v in fm.items()}
    hypers = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for par in PUBLIC_PARENTS:
            _, h = LC.bundle(par, 0.0)
            for n, c in CONFIGS.items():
                if c["kind"] == "hier" and (MU0_MP_X is not None or n != "sd_x085_mp"):
                    hh = config_hyper(c, h)
                    hypers.setdefault(n, {})[par] = {k: round(float(getattr(hh, k)), 6) for k in
                                                     ("mu0", "sigma_mu", "sigma_delta", "sigma_theta",
                                                      "attr_scale")}
            hypers.setdefault("fitted (prior.build without the parent)", {})[par] = {
                k: round(float(getattr(h, k)), 6) for k in ("mu0", "sigma_mu", "sigma_delta", "sigma_theta")}
    checks = {"abs_reproduces_MU0_MP_ABS": abs(mu_abs - MU0_MP_ABS) <= MU0_TOL,
              "x_reproduces_MU0_MP_X": MU0_MP_X is not None and abs(mu_x - MU0_MP_X) <= MU0_TOL}
    res = {"ok": all(checks.values()), "checks": checks,
           "runs": {"seed": CALIB_SEED, "regimes": list(CALIB_REGIMES),
                    "runs": [CALIB_RUNS.start, CALIB_RUNS.stop - 1]},
           "mean_b0": m, "per_run_mean_b0": per_run, "step": CALIB_STEP,
           "abs": {"mu0_mp": mu_abs, "slope_per_mu0": s_abs, "fixed": MU0_MP_ABS},
           "x": {"mu0_mp": mu_x, "slope_per_mu0": s_x, "fixed": MU0_MP_X},
           "fixed_configs_mean_b0": fixed, "hypers_per_parent": hypers,
           "note": "budget-0 predictions only: no label is read; P1a's scoring seed, not SEED",
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("calib", res)
    print(f"calib: abs mu0 {mu_abs:.6f} (fixed {MU0_MP_ABS}), x mu0 {mu_x:.6f} (fixed {MU0_MP_X}); "
          f"{'ok' if res['ok'] else 'NOT ok'} ({time.time() - t0:.0f}s)", flush=True)


# --- stage: regimes -------------------------------------------------------------------------------

def stage_regimes():
    state = load_state(OUT)
    require(state, "lock")
    t0 = time.time()
    rs_state = load_state(RS.OUT)
    cal = (rs_state.get("regimes") or {}).get("cal") or {}
    comps, score, checks = {}, {}, {}
    nr = n_runs()
    for R in REGIMES:
        metas = {}
        for i in range(nr[R]):
            key, run = RS.draw(R, seed(), i)
            metas[i] = RS.describe(key, run)
        comps[R] = metas
        score[R] = RS.realised([p["logit"] for m in metas.values() for p in m])
        score[R]["runs"] = nr[R]
        score[R]["pairs_per_run"] = float(np.mean([len(m) for m in metas.values()]))
        if RS.REGIMES[R][0] == "testlike" and R in cal:
            keys = ("mean", "sd") + (("share_below", "rest_mean") if R == "MIXTURE" else ())
            checks[R] = {k: {"p1a_cal_seed": cal[R][k], "here": score[R][k], "diff": score[R][k] - cal[R][k],
                             "tol": RS.REALISE_TOL[k], "ok": abs(score[R][k] - cal[R][k]) <= RS.REALISE_TOL[k]}
                         for k in keys}
        print(f"  regimes: {R} {nr[R]} runs, level {score[R]['mean']:+.3f}/{score[R]['sd']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(ROWS, exist_ok=True)
    cpath = os.path.join(ROWS, COMPOSITIONS)
    write_json(cpath, {"seed": seed(), "runs": {R: {str(i): m for i, m in v.items()} for R, v in comps.items()}})
    realised_ok = bool(checks) and all(c["ok"] for v in checks.values() for c in v.values())
    res = {"ok": True, "realisation_ok": realised_ok, "seed": seed(), "n_runs": nr, "score": score,
           "checks": checks, "tolerance_note": "P1a's REALISE_TOL (about 3 sds of an 80-run block)",
           "compositions": {"file": os.path.relpath(cpath, ROOT) if not DRY else cpath, "sha256": file_sha(cpath)},
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("regimes", res)
    print(f"regimes: compositions written; realisation {'within' if realised_ok else 'OUTSIDE'} P1a's "
          f"tolerance ({time.time() - t0:.0f}s)", flush=True)


# --- stage: reproduce -----------------------------------------------------------------------------

def stage_reproduce():
    state = load_state(OUT)
    require(state, "lock")
    t0 = time.time()
    w = warm()
    lib = RS.lib_digests()
    checks, timing = [], {}
    for R, i in REPRO_TASKS:
        t1 = time.time()
        with open(RS.row_path(R, i)) as f:
            old = json.load(f)
        key, run = RS.draw(R, RS.SCORE_SEED, i)
        meta = RS.describe(key, run)
        got = score_run(run, list(REPRO_CONFIGS))
        per = {}
        for c in REPRO_CONFIGS:
            g, s = got[c], old["res"][c]
            a, b = np.asarray(g["b"], float), np.asarray(s["b"], float)
            d = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
            per[c] = {"max_abs_brier": d,
                      "ece_equal": bool(np.array_equal(np.asarray(g["ece"], float), np.asarray(s["ece"], float))),
                      "q0_equal": bool(np.array_equal(np.asarray(g["q0"], float), np.asarray(s["q0"], float))),
                      "counters_equal": all(g[k] == s[k] for k in ("calls", "failures", "unconverged")),
                      "secs": g["secs"]}
            if "eb" in g or "eb" in s:
                per[c]["eb_equal_sorted"] = sorted(map(list, g.get("eb", []))) == sorted(map(list, s.get("eb", [])))
            per[c]["ok"] = bool(d <= REPRO_TOL and per[c]["ece_equal"] and per[c]["q0_equal"]
                                and per[c]["counters_equal"] and per[c].get("eb_equal_sorted", True))
            timing.setdefault(c, []).append(g["secs"])
        same = json.loads(json.dumps(meta)) == old["meta"]
        checks.append({"regime": R, "i": i, "seed": RS.SCORE_SEED, "meta_equal": same,
                       "library_equal": old["lib"] == lib, "per_config": per,
                       "ok": bool(same and old["lib"] == lib and all(v["ok"] for v in per.values())),
                       "secs": round(time.time() - t1, 1)})
        print(f"  reproduce: {R} {i}: max |dBrier| {max(v['max_abs_brier'] for v in per.values()):.2g}, "
              f"{'ok' if checks[-1]['ok'] else 'FAILED'} ({time.time() - t1:.0f}s)", flush=True)
        del got, run
        gc.collect()
    ok = all(c["ok"] for c in checks)
    res = {"ok": ok, "tasks": [list(t) for t in REPRO_TASKS], "configs": list(REPRO_CONFIGS), "tol": REPRO_TOL,
           "checks": checks, "warm_s": w, "secs_per_config": {c: float(np.mean(v)) for c, v in timing.items()},
           "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    save_stage("reproduce", res)
    print(f"reproduce: {'passed' if ok else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)


# --- stage: score ---------------------------------------------------------------------------------

def stage_score(shard):
    k, n = shard
    state = load_state(OUT)
    require(state, "lock", "calib", "regimes", "reproduce")
    cpath = os.path.join(ROWS, COMPOSITIONS)
    if file_sha(cpath) != state["regimes"]["compositions"]["sha256"]:
        raise SystemExit(f"{cpath} differs from what the regimes stage wrote")
    if state["regimes"]["seed"] != seed():
        raise SystemExit("the regimes stage drew another seed")
    with open(cpath) as f:
        comps = json.load(f)["runs"]
    todo = [t for t in shard_tasks(k, n) if not os.path.exists(row_path(*t))]
    print(f"score shard {k}/{n}: {len(todo)} of {len(shard_tasks(k, n))} tasks to do; seed {seed()}", flush=True)
    if not todo:
        print(f"score shard {k}/{n}: done", flush=True)
        return
    print(f"  [{k}/{n}] warm-up {warm()}s", flush=True)
    names = list(CONFIGS)
    t_start = time.time()
    for j, (R, i) in enumerate(todo):
        t0 = time.time()
        key, run = RS.draw(R, seed(), i)
        meta = RS.describe(key, run)
        if json.loads(json.dumps(meta)) != comps[R][str(i)]:
            raise RuntimeError(f"{R} run {i}: the drawn run differs from the regimes stage's")
        res = score_run(run, names)
        rss = RS.rss_gb()
        row = {"regime": R, "i": i, "seed": seed(), "key": key, "meta": meta, "res": res,
               "task_s": round(time.time() - t0, 3), "rss_gb": round(rss, 3), "shard": f"{k}/{n}",
               "pid": os.getpid(), "lib": RS.lib_digests(), "script": SCRIPT_SHA, "utc": utcnow()}
        write_json(row_path(R, i), row)
        del res, run, row
        gc.collect()
        rss = RS.rss_gb()
        print(f"  [{k}/{n}] {R} {i}: {time.time() - t0:.1f}s, rss {rss:.2f} GB, "
              f"{j + 1}/{len(todo)} in {time.time() - t_start:.0f}s", flush=True)
        if rss > RSS_GUARD_GB and j + 1 < len(todo):
            print(f"  [{k}/{n}] resident set {rss:.2f} GB > {RSS_GUARD_GB} GB: stopping (resumable)", flush=True)
            sys.exit(RESUME_EXIT)
    print(f"score shard {k}/{n}: done", flush=True)


def stage_status():
    nr = n_runs()
    for R in REGIMES:
        have = sum(os.path.exists(row_path(R, i)) for i in range(nr[R]))
        print(f"{R:8s} {have:3d}/{nr[R]}")


# --- stage: summarise -----------------------------------------------------------------------------

def load_rows():
    nr = n_runs()
    rows = {R: {} for R in REGIMES}
    for R in REGIMES:
        for i in range(nr[R]):
            p = row_path(R, i)
            if os.path.exists(p):
                with open(p) as f:
                    rows[R][i] = json.load(f)
    return rows


def seed11(rs_sum, R, X):
    """P1a's own seed-11 numbers for a config it scored too (X minus ship)."""
    try:
        e = rs_sum["vs_ship"][R][X]["exact"]
    except (KeyError, TypeError):
        return None
    return {"D": e["D"], "cluster_se": e["cluster_se"], "runs": RS.N_RUNS[R], "seed": RS.SCORE_SEED}


def summarise_regime(G, share_hidden, rs_sum):
    w_hv, hv_info = hv_weights(G, share_hidden)
    out = {"runs": len(G.ids), "pairs": int(len(G.w)), "clusters": int(G.K), "realised": G.level(),
           "hidden_v": hv_info, "configs": {c: G.config_summary(c) for c in CONFIGS}}
    vs, agree = {}, 0.0
    for X in CONFIGS:
        if X == SHIP:
            continue
        ex = G.diff(X, SHIP)
        eq = weighted_diff(G, X, SHIP, G.w)
        agree = max(agree, abs(eq["D"] - ex["D"]), abs(eq["cluster_se"] - ex["cluster_se"]),
                    abs(eq["strat_se"] - ex["strat_se"]))
        eq.update({k: ex[k] for k in ("run_se", "share_runs_x_better", "parents_present")})
        vs[X] = {"equal": eq, "hidden_v": weighted_diff(G, X, SHIP, w_hv),
                 "minus_legacy_equal": None if X == "legacy" else G.diff(X, "legacy")["D"],
                 "seed11": seed11(rs_sum, G.R, X) if X in P1A_CONFIGS else None}
    out["vs_ship"] = vs
    out["equal_matches_regime_rows_max_abs"] = agree
    out["ranks"] = RS.ranks(G)
    return out


def stage_summarise():
    state = load_state(OUT)
    require(state, "lock")
    t0 = time.time()
    rows = load_rows()
    nr = n_runs()
    have = {R: len(rows[R]) for R in REGIMES}
    complete = all(have[R] == nr[R] for R in REGIMES) and all(
        set(r["res"]) == set(CONFIGS) for rs in rows.values() for r in rs.values())
    checks = {s: bool((state.get(s) or {}).get("ok")) for s in ("lock", "calib", "regimes", "reproduce")}
    checks["lock_now"] = all(lock_checks().values())
    checks["realisation"] = bool((state.get("regimes") or {}).get("realisation_ok"))
    cpath = os.path.join(ROWS, COMPOSITIONS)
    comps = {}
    if os.path.exists(cpath):
        with open(cpath) as f:
            comps = json.load(f)["runs"]
    checks["rows_compositions"] = all(comps.get(R, {}).get(str(i)) == r["meta"]
                                      for R, rs in rows.items() for i, r in rs.items())
    libs = {json.dumps(r["lib"], sort_keys=True) for rs in rows.values() for r in rs.values()}
    checks["rows_one_library"] = len(libs) <= 1
    scripts = sorted({r["script"] for rs in rows.values() for r in rs.values()})
    checks["rows_one_script"] = len(scripts) <= 1
    checks["rows_seed"] = {r["seed"] for rs in rows.values() for r in rs.values()} <= {seed()}
    cnt, share_hidden = hidden_shares()
    checks["hidden_v_counts"] = tuple(int(x) for x in cnt) == HV_COUNTS
    summary = {"planned": nr, "scored": have, "complete": bool(complete), "seed": seed(), "dry_run": DRY,
               "hidden_v": {"bins": HV_BINS, "counts": cnt.tolist(), "shares": share_hidden.tolist(),
                            "source": list(FORMATIVE),
                            "definition": "v = B0 - ECE0^2 per formative appearance, clipped to [0, 0.25]; "
                                          "a replica appearance's v is p(1 - p) at its evaluated rate"},
               "rows": {"library_digests": [json.loads(x) for x in sorted(libs)], "script_digests": scripts,
                        "script_now": SCRIPT_SHA, "rows_scored_by_this_script": scripts == [SCRIPT_SHA],
                        "task_s": {R: float(np.mean([r["task_s"] for r in rs.values()])) if rs else None
                                   for R, rs in rows.items()},
                        "max_rss_gb": max([r["rss_gb"] for rs in rows.values() for r in rs.values()], default=None)}}
    if not any(rows.values()):
        summary["note"] = "no scored rows"
        save_stage("summary", summary)
        return
    rs_sum = load_state(RS.OUT).get("summary") or {}
    with rs_configs():
        regs = {R: RS.RegimeRows(R, rows[R]) for R in REGIMES if rows[R]}
        summary["regimes"] = {R: summarise_regime(G, share_hidden, rs_sum) for R, G in regs.items()}
        agree = max(summary["regimes"][R]["equal_matches_regime_rows_max_abs"] for R in regs)
        checks["equal_weights_are_p1as_statistics"] = agree <= 1e-12
        lat = RS.latency(rows)
        summary["latency"] = lat
        summary["eb_traces"] = RS.eb_traces(rows)
        if "TUNED" in regs:
            d = regs["TUNED"].diff(SHIP, "legacy")
            want, want_se = RS.KNOWN_SHIP_MINUS_LEGACY
            comb = math.sqrt(d["cluster_se"] ** 2 + want_se ** 2)
            ok = abs(d["D"] - want) <= RS.CONSISTENCY_K * comb
            summary["tuned_consistency"] = {"ship_minus_legacy": d["D"], "run_se": d["run_se"],
                                            "cluster_se": d["cluster_se"], "known": want, "known_se": want_se,
                                            "combined_se": comb, "z": (d["D"] - want) / comb,
                                            "k": RS.CONSISTENCY_K, "ok": bool(ok)}
            checks["tuned_consistency"] = bool(ok)
        else:
            checks["tuned_consistency"] = False
        checks_ok = all(checks.values())
        if all(R in regs for R in REGIMES):
            diffs = {R: {X: G.diff(X, SHIP) for X in CONFIGS if X != SHIP} for R, G in regs.items()}
            glegacy = {R: {X: (0.0 if X == "legacy" else G.diff(X, "legacy")["D"]) for X in CONFIGS if X != SHIP}
                       for R, G in regs.items()}
            rule = RS.apply_rule(diffs, glegacy, lat["all"], checks_ok, complete)
        else:
            rule = {"applied": False, "outcome": "rule not applied: the planned set is not complete"}
    rule["would_pass"] = list(rule.get("candidates") or [])
    rule["s4_reading"] = S4_READING
    rule["shipped"] = []
    rule["rule_sha256"] = RS.RULE_SHA256
    rule["thresholds"] = RS.RULE
    rule["applied_utc"] = utcnow()
    summary["table"] = table(summary)
    summary["checks"] = checks
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["provenance"] = provenance()
    save_stage("summary", summary)
    save_stage("rule", rule)
    save_stage("status", "complete" if rule.get("applied") else "partial")
    print(f"summarise: {have} scored; checks {[k for k, v in checks.items() if not v] or 'all hold'}; "
          f"rule: {rule['outcome']} (reported, not shipped)", flush=True)
    for line in summary["table"]["lines"]:
        print(line)


def table(summary):
    """The report's table: X minus ship, equal and hidden-v weights, per regime (D and
    cluster SE), plus the mean over READING and AUDIT."""
    regs = summary.get("regimes") or {}
    rows, lines = [], []
    hdr = f"{'config':16s} " + " ".join(f"{R:>17s}" for R in REGIMES if R in regs)
    for wt in ("equal", "hidden_v"):
        lines.append(f"-- {wt} weights: X minus ship, D (cluster SE)")
        lines.append(hdr)
        for X in CONFIGS:
            if X == SHIP:
                continue
            cells, rec = [], {"config": X, "weights": wt}
            for R in REGIMES:
                if R not in regs:
                    continue
                v = regs[R]["vs_ship"][X][wt]
                rec[R] = {"D": v["D"], "cluster_se": v["cluster_se"]}
                cells.append(f"{v['D']:+.5f} ({v['cluster_se']:.5f})")
            fb = [rec[R]["D"] for R in RS.FEEDBACK_REGIMES if R in rec]
            rec["feedback_mean"] = float(np.mean(fb)) if fb else None
            rows.append(rec)
            lines.append(f"{X:16s} " + " ".join(f"{c:>17s}" for c in cells))
    return {"rows": rows, "lines": lines}


# --- driver ---------------------------------------------------------------------------------------

def set_dry_run(path, runs):
    """Redirect OUT and ROWS to a directory outside the repository and score DRY_SEED's
    first `runs` runs of every regime."""
    global DRY, OUT, ROWS
    d = os.path.abspath(path)
    if os.path.commonpath([d, ROOT]) == ROOT:
        raise SystemExit("--dry-run: the directory must lie outside the repository")
    if runs < 1:
        raise SystemExit("--runs must be at least 1")
    os.makedirs(d, exist_ok=True)
    DRY = {"dir": d, "runs": int(runs), "seed": DRY_SEED}
    OUT = os.path.join(d, "ablations_at_level.json")
    ROWS = os.path.join(d, "rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("lock", "calib", "regimes", "reproduce", "score", "summarise", "status"))
    ap.add_argument("--shard", default="0/1", help="score: every n-th planned task from k (k/n)")
    ap.add_argument("--dry-run", default=None, metavar="DIR",
                    help="write OUT and rows under DIR (outside the repository) and score DRY_SEED")
    ap.add_argument("--runs", type=int, default=2, help="with --dry-run: runs per regime (default 2)")
    a = ap.parse_args(argv)
    if a.dry_run:
        set_dry_run(a.dry_run, a.runs)
    elif "--runs" in (argv if argv is not None else sys.argv[1:]):
        ap.error("--runs needs --dry-run: the planned job scores every planned run")
    if a.stage == "lock":
        stage_lock()
    elif a.stage == "calib":
        stage_calib()
    elif a.stage == "regimes":
        stage_regimes()
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
