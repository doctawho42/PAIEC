"""P1.11: contribution 1, decomposed (review W5 and P1.11; docs/report/review_v0.md).

The draft's contribution 1 says that our first replica made pooling item
difficulty across subjects look like the main lever (about 0.020 ALC, the
legacy ladder, docs/findings.md "The predictor ladder") and that at formative
size under the verified protocol it is worth about 0.0007 (the legacy
Predictor's pooled difficulty switched off, docs/findings.md "Formative-sized
runs (R1)"; that number has no committed script). The review (W5) points out
that the two numbers differ in run size and split scope as well as in the
information set, and that the official protocol still rewards pooling on dense
runs under per-pair splits. This script measures pooling on and off on the
official replica (paiec.official), on identical runs and checkpoints, crossing

  run size     formative: official.sample_run at its defaults (benchmark-first),
               runs default_rng([SEED, i]) for i in 0..N_FORMATIVE-1; and dense:
               official.dense_run of each multi-subject public benchmark
  split scope  'pair' and 'benchmark' (official.split)

with these predictors, every one scored on the same checkpoints:

  hier in the shipped configuration (paiec.hier.HierPredictor with
  submission/model.py's LEVEL over paiec.prior.build fitted without the
  target's benchmark, as regime_sensitivity's 'ship'):
    ship           as shipped: the whole shared `labeled` list
    ship_nolevel   Flags(pool_mu=False): a level per pair. Other subjects'
                   labels still reach the target's item residuals and group
                   effects (and their own standings)
    ship_own       the fit restricted, per target, to the target pair's own
                   labels: entries with the target's subject (paiec.predict.
                   subject_key of the eight visible fields) and its
                   benchmark_id. Nothing from other subjects reaches it, and
                   nothing from the subject's labels on other benchmarks
                   (the link, weight 0.0018 at the shipped hyperparameters).
                   hier has no switch for item pooling alone, so this is the
                   'off' variant: a wrapper here (OwnOnly), research-only; the
                   library and the shipped default are unchanged
  the legacy Predictor (paiec.predict.Predictor, attribute prior fitted without
  the target's benchmark: level_calibration.ppred, regime_sensitivity's
  'legacy'):
    legacy         pooled item difficulty, a joint IRT on the benchmark's labels,
                   once the benchmark holds BenchmarkFit.WARMUP = 64 distinct
                   labeled items
    legacy_nopool  WARMUP raised past any run (NoPoolPredictor), so z = 0: the
                   switch behind the draft's 0.0007. z is the Predictor's only
                   cross-subject channel, so this is also the Predictor on the
                   pair's own labels (tests/test_pooling_decomposition.py)
  smooth           Beta(2,2) on the pair's own labels (baselines.smoothed_mean,
                   a reference)

What the differences measure (POOLING):
  ship - ship_own          everything hier takes from other subjects' labels
  ship - ship_nolevel      the pooled level's share, item pooling on
  ship_nolevel - ship_own  pooled item difficulty's share (residuals and group
                           effects), with a level per pair; the two shares add
                           up to the first
  legacy - legacy_nopool   the legacy Predictor's pooled item difficulty

Dense runs. swe_rebench has one subject: its dense run is one pair, on which
every on/off pair of variants coincides by construction (ship_own's
restriction keeps every label; a level per pair is the benchmark's level; 31
labels never reach WARMUP), so it is not run. On researchcodebench, matharena
and multi_swebench every pair of the benchmark is in the run and acquires
exactly as in the full dense run, so `labeled` is the full dense run's at every
checkpoint, but each pair's Brier is taken over at most EVAL_CAP of its
evaluation items (every response of each), the first by a digest rank
(eval_subset). A prediction is a pure function of (input, labeled), so these
are the full dense run's predictions on a subset of its targets, the same
subset for every variant; the cap only keeps the study near two hours on one
process of a shared machine (a full dense multi_swebench run is 28,569
distinct targets a checkpoint, about 18 ms a call for hier at B31).
real_webagents is scored whole.

Statistics. Formative, per scope: ship_confirm.Block, as ship_confirm and
regime_sensitivity use it: a difference is X minus ref paired on identical
runs, "± run SE / cluster SE / stratified SE" (level_calibration.Boot, 2,000
resamples, a ratio estimator, cluster (benchmark, subject), stratified within
benchmark), per benchmark the appearance-weighted mean (weight 1 / run size)
with its cluster SE and the mean with it left out, the benchmark-level mean
± SE over the four multi-subject benchmarks (treating them as the sample,
which the cluster SEs do not), and per budget. By companions: the same per
appearance, split by how many pairs of its benchmark the run holds (1, 2, 3 or
more), with cluster SEs; and the same reweighted to the platform's companion
mix ('platform_mix'): formative runs 1-3 as scored (results/formative_feedback.json,
results/formative_run3.json) hold 16 of their 26 pair appearances alone on their
benchmark, 10 beside one other pair and none beside more, against 193 / 220 / 467
of 880 here; the reweighted difference is the sum of share x class mean, ALC and
per budget, with cluster and stratified SEs that resample the classes jointly.
swe_rebench, the one single-subject benchmark, is alone in every run by
construction (87 of the 193 'alone' appearances under scope 'pair'), so both
readings are also given without its appearances ('without_single_subject';
summary 'single_subject_confound').
Dense, per benchmark and scope: mean over the
benchmark's pairs ± the SE over pairs (they share one `labeled` list and one
split, so it is conditional on both, as in docs/findings.md R2), per budget;
across the four benchmarks, the mean of their means ± the SE across them.
Scope effect: (X - ref under 'pair') minus (X - ref under 'benchmark'), on the
same runs and pairs, with the same SEs. Run-size effect: per benchmark, the
formative mean beside the dense one, with the SE of their difference taken as
if the two were independent (they share pairs, so it overstates).

Checks (the score stage stops on a failure; summarise records them):
  - LEVEL equals submission/model.py's
  - reproduction: runs 0-59 under scope 'pair' are regime_sensitivity's R1B
    runs (same seed, same draw); ship, legacy and smooth must equal its stored
    per pair Brier (data/regime_sensitivity_rows/R1B, all scored with the
    library digests this run checks) to REPRO_TOL, pair for pair
  - at B0 nothing is labeled, so ship, ship_nolevel and ship_own coincide, and
    legacy and legacy_nopool coincide wherever the target's benchmark holds
    fewer than WARMUP distinct labeled items (summarise: max |difference|)
  - every variant's fallbacks (failures) are counted; summarise reports them
  - dense real_webagents (scored whole) against results/hier_eval.json's R2
    rows of the legacy Predictor and the smoothed mean (library of bba726c,
    Brier to 6 decimals): recorded, not required
  - recheck: RECHECK's tasks re-scored by the script as it is at summary time
    must give the stored rows bit for bit (the rows were scored before the
    summary code gained the platform mix and the reading table); summarise
    reports it, with whether it ran this script and library

Run (one process, resumable; a row file per task in ROWS):
  python experiments/pooling_decomposition.py score        # about 100 min, one process
  python experiments/pooling_decomposition.py status
  python experiments/pooling_decomposition.py recheck      # about 3 min
  python experiments/pooling_decomposition.py summarise

Output: results/pooling_decomposition.json (OUT); rows in
data/pooling_decomposition_rows/ (ROWS, gitignored).
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.ship_confirm as SC  # noqa: E402
from paiec import baselines as BL  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec import predict as P  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS, stable_hash  # noqa: E402
from paiec.hier import HierPredictor  # noqa: E402

OUT = os.path.join(ROOT, "results", "pooling_decomposition.json")
ROWS = os.path.join(ROOT, "data", "pooling_decomposition_rows")
RS_ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows", "R1B")
W6 = np.asarray(WEIGHTS, float)

# --- the design ---------------------------------------------------------------------------

#: submission/model.py's LEVEL (checked)
LEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
#: formative runs are official.sample_run(pairs, default_rng([SEED, i])); seed 11 is
#: regime_sensitivity's scoring seed, so runs 0-59 are its R1B runs (reproduced below)
SEED = 11
N_FORMATIVE = 100
SCOPES = ("pair", "benchmark")
#: dense runs, in the order they are scored
DENSE = ("real_webagents", "researchcodebench", "matharena", "multi_swebench")
#: benchmarks with one subject: on/off coincide by construction, not run
SINGLE_SUBJECT = ("swe_rebench",)
#: evaluation items scored per pair on a dense run (None: all)
EVAL_CAP = {"researchcodebench": 32, "matharena": 32, "multi_swebench": 32}
CAP_SALT = "p1.11 dense evaluation subset"
#: legacy_nopool's BenchmarkFit.WARMUP: more distinct items than any run holds
NOPOOL_WARMUP = 10 ** 9
VARIANTS = {
    "ship": {"kind": "hier", "flags": {}, "own": False},
    "ship_nolevel": {"kind": "hier", "flags": {"pool_mu": False}, "own": False},
    "ship_own": {"kind": "hier", "flags": {}, "own": True},
    "legacy": {"kind": "predictor", "pool": True},
    "legacy_nopool": {"kind": "predictor", "pool": False},
    "smooth": {"kind": "smoothed", "n0": 4.0, "m0": 0.5},
}
NAMES = tuple(VARIANTS)
#: (x, ref, what x - ref measures); the first four are the pooling decomposition
POOLING = (
    ("ship", "ship_own", "hier: everything taken from other subjects' labels (pooling on minus off)"),
    ("ship", "ship_nolevel", "hier: the pooled level's share, item pooling on"),
    ("ship_nolevel", "ship_own", "hier: pooled item difficulty's share (residuals and groups), level per pair"),
    ("legacy", "legacy_nopool", "legacy Predictor: pooled item difficulty (WARMUP 64 against off)"),
)
REFERENCE = (
    ("ship", "legacy", "hier as shipped against the legacy Predictor"),
    ("ship", "smooth", "hier as shipped against the smoothed mean"),
    ("ship_own", "smooth", "hier on own labels against the smoothed mean"),
    ("legacy", "smooth", "legacy Predictor against the smoothed mean (findings: -0.0069, target-LOBO)"),
    ("legacy_nopool", "smooth", "legacy Predictor without pooled difficulty against the smoothed mean "
                                "(findings: -0.0062)"),
)
COMPARISONS = POOLING + REFERENCE
#: reproduction against regime_sensitivity's stored rows (R1B: seed 11, scope 'pair',
#: benchmark-first): this study's variant -> the stored config
REPRO = {"runs": tuple(range(60)), "scope": "pair", "seed": 11,
         "configs": {"ship": "ship", "legacy": "legacy", "smooth": "smooth"}}
REPRO_TOL = 1e-12
#: the dense cross-check: this study's variant -> hier_eval.json's R2 name
HE_JSON = os.path.join(ROOT, "results", "hier_eval.json")
HE_DENSE = {"legacy": "Predictor", "smooth": "smoothed Beta(2,2)"}
HE_TOL = 5e-7
#: files whose digests every row records (regime_sensitivity's LIB_FILES)
LIB_FILES = ("paiec/hier.py", "paiec/prior.py", "paiec/predict.py", "paiec/official.py",
             "paiec/testlike.py", "paiec/baselines.py", "paiec/subjects.py", "paiec/mcq.py",
             "paiec/fitting.py", "paiec/irt.py", "paiec/data.py", "paiec/evaluator.py",
             "experiments/level_calibration.py", "submission/model.py")
PARENTS = SC.PARENTS
BOOTS = LC.BOOTS
RSS_GUARD_GB = 1.45
#: exit code of a score process that stopped at RSS_GUARD_GB (rerun to resume)
RESUME_EXIT = 75
COMPANIONS = (("alone", 1, 1), ("two", 2, 2), ("three or more", 3, 10 ** 6))
#: the platform's scored formative runs (their per pair feedback tables), whose
#: companion mix reweights the formative classes: (file, keys down to the pair list)
PLATFORM_RUNS = (("results/formative_feedback.json", ("record", "run1", "pairs")),
                 ("results/formative_feedback.json", ("record", "run2", "pairs")),
                 ("results/formative_run3.json", ("run3", "pairs")))


# --- small helpers ------------------------------------------------------------------------

def r6(x):
    return None if x is None else round(float(x), 6)


def file_sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


SCRIPT_SHA = file_sha(os.path.abspath(__file__))


def lib_digests():
    return {f: file_sha(os.path.join(ROOT, f)) for f in LIB_FILES}


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:                                  # peak, not current, without psutil
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


def model_level():
    """submission/model.py's LEVEL, read without importing the module."""
    with open(os.path.join(ROOT, "submission", "model.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("submission/model.py defines no LEVEL")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


# --- the 'off' variants -------------------------------------------------------------------

def own_key(entry):
    """(subject key, benchmark id) of a labeled entry, or None for an entry
    paiec.predict.records would drop as malformed."""
    try:
        (s, i), _ = entry
        if isinstance(s, dict) and isinstance(i, dict):
            return P.subject_key(s), P._text(i.get("benchmark_id"))
    except Exception:
        pass
    return None


class OwnOnly:
    """predict(input, labeled) of `fn` on the target pair's own labels only: the
    entries whose subject has the target's subject_key and whose item has its
    benchmark_id. The grouping of one labeled list is kept while the same list
    object (of the same length) comes back, which it does for every target of a
    checkpoint; the list is held, so its id cannot be reused meanwhile."""

    def __init__(self, fn):
        self.fn = fn
        self._list, self._len, self._groups = None, -1, None

    def groups(self, labeled):
        if labeled is None or not isinstance(labeled, list):
            return self._group(labeled or [])
        if labeled is not self._list or len(labeled) != self._len:
            self._list, self._len, self._groups = labeled, len(labeled), self._group(labeled)
        return self._groups

    @staticmethod
    def _group(labeled):
        g = {}
        for e in labeled:
            k = own_key(e)
            if k is not None:
                g.setdefault(k, []).append(e)
        return g

    def __call__(self, input, labeled=None):
        s, it = input
        k = (P.subject_key(s), P._text(it.get("benchmark_id")))
        return self.fn(input, list(self.groups(labeled).get(k, ())))


class NoPoolFit(P.BenchmarkFit):
    """BenchmarkFit whose warm-up no run reaches: every z is 0."""
    WARMUP = NOPOOL_WARMUP


class NoPoolEvidence(P.Evidence):
    def fit(self, bid):
        if bid not in self._fits:
            self._fits[bid] = NoPoolFit(self.by_benchmark.get(bid, []), self.dim)
        return self._fits[bid]


class NoPoolPredictor(P.Predictor):
    """The legacy Predictor with its pooled item difficulty switched off
    (WARMUP = NOPOOL_WARMUP); everything else as Predictor._evidence_for."""

    def _evidence_for(self, labeled):
        recs = P.records(labeled)
        fp = P.fingerprint(recs)
        if fp in self._evidence:
            self._evidence.move_to_end(fp)
        else:
            self._evidence[fp] = NoPoolEvidence(recs, self.dim)
            while len(self._evidence) > self.keep:
                self._evidence.popitem(last=False)
        return self._evidence[fp]


class Maker:
    """The model factory of one variant on one run for level_calibration.evaluate:
    every call (one per checkpoint) builds fresh per-benchmark instances,
    dispatched on the anonymous benchmark_id, each with a prior fitted without
    that benchmark; the previous checkpoint's are released after their counters
    are read. `bundle(parent)` -> (SubjectPrior, Hyper) and `ppred(parent)` ->
    (coef, spec) default to level_calibration's (fitted on the public pairs)."""

    def __init__(self, name, run, bundle=None, ppred=None):
        self.name, self.cfg = name, VARIANTS[name]
        self.ids = T.anon_parents(run)
        self.bundle = bundle or (lambda par: LC.bundle(par, 0.0))
        self.ppred = ppred or LC.ppred
        self.live = []
        self.failures = self.unconverged = 0

    def build(self, parent):
        c = self.cfg
        if c["kind"] == "hier":
            prior, hyper = self.bundle(parent)
            return HierPredictor(prior, replace(hyper, **LEVEL), **c["flags"])
        if c["kind"] == "predictor":
            return (P.Predictor if c["pool"] else NoPoolPredictor)(*self.ppred(parent))
        raise ValueError(c["kind"])

    def harvest(self):
        for m in self.live:
            self.failures += int(getattr(m, "failures", 0))
            self.unconverged += int(getattr(m, "unconverged", 0))
        self.live = []

    def __call__(self):
        self.harvest()
        c = self.cfg
        if c["kind"] == "smoothed":
            return BL.smoothed_mean(prior_n=c["n0"], prior_p=c["m0"])
        ms = {par: self.build(par) for par in sorted(set(self.ids.values()))}
        self.live = list(ms.values())
        ids = self.ids

        def fn(inp, labeled=None):
            return ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled)
        return OwnOnly(fn) if c.get("own") else fn


# --- runs and checkpoints -----------------------------------------------------------------

def eval_subset(slot, cap):
    """The evaluation inputs of a slot that are scored: all, or the first `cap`
    distinct ones by a digest of (salt, subject, benchmark, input)."""
    keys, seen = [], set()
    for _, _, k in slot.targets:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    if cap is None or len(keys) <= cap:
        return set(keys)
    keys.sort(key=lambda k: stable_hash(0, CAP_SALT, slot.subject_id, slot.benchmark_id, k))
    return set(keys[:cap])


def checkpoints(run, scope="pair", cap=None, seed=0):
    """level_calibration.checkpoints with an optional cap on the evaluation inputs
    scored per pair (acquisition, and so every `labeled`, untouched):
    (slots, [(budget, labeled, inputs, index)]) and each slot's evaluation
    inputs before the cap."""
    slots = O._slots(run, seed, scope)
    before = [len({t[2] for t in s.targets}) for s in slots]
    if cap is not None:
        for s in slots:
            keep = eval_subset(s, cap)
            s.targets = [t for t in s.targets if t[2] in keep]
            s.n_items = len(keep)
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
    return slots, cps, before


def describe(run, slots, before):
    """Per pair: real benchmark (= parent on public runs) and subject id; evaluation
    items scored and before any cap, responses scored, base rate; pairs of its
    benchmark in the run; the share of its scored items that carry another pair's
    acquired label at B31 (cover31); distinct labeled items on its benchmark at
    each budget."""
    keys_acq = [[P.item_key(a[0][1]) for a in s.acquired] for s in slots]
    out = []
    for j, ((pair, _), s) in enumerate(zip([O._entry(e) for e in run], slots)):
        same = [k for k, t in enumerate(slots) if t.benchmark_id == s.benchmark_id]
        ev = {P.item_key(t[0][1]) for t in s.targets}
        others = set()
        for k in same:
            if k != j:
                others.update(keys_acq[k][:31])
        y = [t[1] for t in s.targets]
        out.append({"bench": pair.benchmark_id, "parent": T.parent_of(pair.benchmark_id),
                    "subject": pair.subject_id, "eval": s.n_items, "eval_all": before[j],
                    "resp": len(y), "p": round(float(np.mean(y)), 5) if y else None,
                    "companions": len(same),
                    "cover31": round(len(ev & others) / len(ev), 5) if ev else None,
                    "labeled_items": [len({x for k in same for x in keys_acq[k][:b]})
                                      for b in BUDGETS]})
    return out


def score_run(run, scope, cap=None, names=NAMES, bundle=None, ppred=None):
    """Every variant of `names` on the run's checkpoints -> (meta, {name: per pair
    Brier and ECE by budget, mean B0 prediction, timing and counters})."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slots, cps, before = checkpoints(run, scope, cap)
        meta = describe(run, slots, before)
        res = {}
        for n in names:
            t0 = time.perf_counter()
            mk = Maker(n, run, bundle, ppred)
            rows, tm = LC.evaluate(slots, cps, [n], mk)
            mk.harvest()
            res[n] = {"b": [x["brier"] for x in rows[n]], "ece": [x["ece"] for x in rows[n]],
                      "q0": [x["q0"] for x in rows[n]], "calls": tm["calls"],
                      "mean_s": tm["mean_s"], "max_s": tm["max_s"], "failures": mk.failures,
                      "unconverged": mk.unconverged, "secs": round(time.perf_counter() - t0, 3)}
            del mk, rows
    return meta, res


def tasks():
    """(kind, run, scope) in the order they are scored: formative runs, each under
    both scopes, then the dense benchmarks."""
    out = [("formative", i, sc) for i in range(N_FORMATIVE) for sc in SCOPES]
    return out + [("dense", b, sc) for b in DENSE for sc in SCOPES]


def row_path(t, rows=ROWS):
    kind, a, sc = t
    if kind == "formative":
        return os.path.join(rows, "formative", sc, f"{a}.json")
    return os.path.join(rows, "dense", f"{a}.{sc}.json")


def draw(t):
    kind, a, _ = t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "formative":
            return O.sample_run(LC.pairs(), np.random.default_rng([SEED, a]))
        return O.dense_run(LC.pairs(), a)


# --- reproduction -------------------------------------------------------------------------

def repro_check(t, meta, res, rows_dir=RS_ROWS):
    """{'checked', 'same_lib', 'max_abs': {variant: max |difference|}, 'ok'} against
    regime_sensitivity's stored R1B row of the same run, or None when the task is
    not one of its runs."""
    kind, i, sc = t
    if kind != "formative" or sc != REPRO["scope"] or i not in REPRO["runs"] or SEED != REPRO["seed"]:
        return None
    path = os.path.join(rows_dir, f"{i}.json")
    if not os.path.exists(path):
        return {"checked": False, "why": f"{path} missing", "ok": None}
    with open(path) as f:
        st = json.load(f)
    lib = lib_digests()
    same_lib = all(st["lib"].get(k) == v for k, v in lib.items() if k in st["lib"])
    pairs_ok = [(m["bench"], m["subject"]) for m in meta] == \
        [(m["bench"], m["subject"]) for m in st["meta"]]
    diffs = {}
    for v, c in REPRO["configs"].items():
        a, b = np.asarray(res[v]["b"], float), np.asarray(st["res"][c]["b"], float)
        diffs[v] = float(np.max(np.abs(a - b))) if a.shape == b.shape else None
    ok = pairs_ok and all(d is not None and d <= REPRO_TOL for d in diffs.values())
    return {"checked": True, "same_lib": same_lib, "pairs_match": pairs_ok, "max_abs": diffs,
            "ok": bool(ok)}


# --- stages -------------------------------------------------------------------------------

def check_level():
    got = model_level()
    if got != LEVEL:
        raise SystemExit(f"submission/model.py LEVEL is {got}, this study's {LEVEL}")


def stage_score(only=None, rows=ROWS, limit=None):
    check_level()
    todo = [t for t in tasks() if (only is None or t[0] == only)
            and not os.path.exists(row_path(t, rows))]
    if limit is not None:
        todo = todo[:limit]
    print(f"score: {len(todo)} of {len(tasks())} tasks to do", flush=True)
    t_start = time.time()
    for j, t in enumerate(todo):
        t0 = time.time()
        run = draw(t)
        cap = EVAL_CAP.get(t[1]) if t[0] == "dense" else None
        meta, res = score_run(run, t[2], cap)
        rep = repro_check(t, meta, res)
        if rep is not None and rep["checked"] and not rep["ok"]:
            raise SystemExit(f"{t}: reproduction against {RS_ROWS} failed: {rep}")
        row = {"task": list(t), "seed": SEED if t[0] == "formative" else None, "cap": cap,
               "meta": meta, "res": res, "repro": rep, "task_s": round(time.time() - t0, 3),
               "rss_gb": round(rss_gb(), 3), "pid": os.getpid(), "lib": lib_digests(),
               "script": SCRIPT_SHA, "utc": utcnow()}
        write_json(row_path(t, rows), row)
        alc = " ".join(f"{n} {np.mean(np.asarray(res[n]['b']) @ W6):.4f}" for n in NAMES)
        del res, run, row
        gc.collect()
        rss = rss_gb()
        print(f"  {t}: {time.time() - t0:.1f}s, rss {rss:.2f} GB, {j + 1}/{len(todo)} in "
              f"{time.time() - t_start:.0f}s; {alc}", flush=True)
        if rss > RSS_GUARD_GB and j + 1 < len(todo):
            print(f"  resident set {rss:.2f} GB > {RSS_GUARD_GB} GB: stopping (resumable)", flush=True)
            sys.exit(RESUME_EXIT)
    print("score: done", flush=True)


#: tasks the recheck stage re-scores with the script as it is now
RECHECK = (("formative", 0, "pair"), ("formative", 0, "benchmark"), ("formative", 99, "pair"),
           ("formative", 99, "benchmark"), ("dense", "real_webagents", "benchmark"))


def recheck_path(rows=ROWS):
    return os.path.join(rows, "recheck.json")


def stage_recheck(rows=ROWS, todo=RECHECK):
    """Re-score `todo` with the script and library as they are now and compare
    with the stored rows, pair for pair (rows may predate a revision of the
    summary code): writes recheck.json beside the rows; summarise reports it."""
    check_level()
    out = []
    for t in todo:
        with open(row_path(t, rows)) as f:
            st = json.load(f)
        t0 = time.time()
        cap = EVAL_CAP.get(t[1]) if t[0] == "dense" else None
        meta, res = score_run(draw(t), t[2], cap)
        same_pairs = [(m["bench"], m["subject"]) for m in meta] == \
            [(m["bench"], m["subject"]) for m in st["meta"]]
        mx = {}
        for n in NAMES:
            a, b = np.asarray(res[n]["b"], float), np.asarray(st["res"][n]["b"], float)
            mx[n] = float(np.max(np.abs(a - b))) if a.shape == b.shape else None
        ok = same_pairs and all(v == 0.0 for v in mx.values()) and meta == st["meta"]
        out.append({"task": list(t), "row_script": st["script"], "same_pairs": same_pairs,
                    "meta_equal": meta == st["meta"], "max_abs": mx, "bit_identical": bool(ok),
                    "secs": round(time.time() - t0, 1)})
        print(f"  recheck {t}: bit-identical {ok}, {time.time() - t0:.0f}s", flush=True)
    rec = {"script": SCRIPT_SHA, "lib": lib_digests(), "utc": utcnow(), "tasks": out,
           "all_bit_identical": all(x["bit_identical"] for x in out)}
    write_json(recheck_path(rows), rec)
    return rec


def stage_status(rows=ROWS):
    have = [t for t in tasks() if os.path.exists(row_path(t, rows))]
    by = {}
    for t in have:
        by.setdefault((t[0], t[2]), 0)
        by[(t[0], t[2])] += 1
    print(f"{len(have)} of {len(tasks())} tasks scored: "
          + ", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(by.items())))


def load_rows(rows=ROWS):
    out = {}
    for t in tasks():
        p = row_path(t, rows)
        if os.path.exists(p):
            with open(p) as f:
                out[t] = json.load(f)
    return out


# --- statistics ---------------------------------------------------------------------------

def companion_class(c):
    for label, lo, hi in COMPANIONS:
        if lo <= c <= hi:
            return label
    raise ValueError(c)


def platform_mix(sources=PLATFORM_RUNS, root=ROOT):
    """The platform's scored formative runs' pair appearances by how many pairs of
    their (anonymous) benchmark the run holds: {'counts', 'shares', 'pairs', 'runs'}."""
    counts = {label: 0 for label, _, _ in COMPANIONS}
    runs = []
    for path, keys in sources:
        with open(os.path.join(root, path)) as f:
            obj = json.load(f)
        for k in keys:
            obj = obj[k]
        per = {}
        for p in obj:
            per[p["benchmark"]] = per.get(p["benchmark"], 0) + 1
        for p in obj:
            counts[companion_class(per[p["benchmark"]])] += 1
        runs.append({"source": f"{path}: {'/'.join(keys)}", "pairs": len(obj), "benchmarks": len(per),
                     "counts": {label: sum(companion_class(per[p["benchmark"]]) == label for p in obj)
                                for label, _, _ in COMPANIONS}})
    n = sum(counts.values())
    return {"counts": counts, "shares": {k: r6(v / n) for k, v in counts.items()}, "pairs": n,
            "runs": runs}


class Formative:
    """One scope's formative runs: per variant and run the (pairs, 6) Brier, and
    ship_confirm.Block's paired statistics."""

    def __init__(self, rows, scope):
        self.scope = scope
        self.ids = sorted(i for (k, i, sc) in rows if k == "formative" and sc == scope)
        rr = [rows[("formative", i, scope)] for i in self.ids]
        self.meta = [r["meta"] for r in rr]
        self.bmeta = [[{"parent": m["parent"], "subject": m["subject"]} for m in mm]
                      for mm in self.meta]
        self.B = {n: [np.asarray(r["res"][n]["b"], float) for r in rr] for n in NAMES}
        self.res = {n: [r["res"][n] for r in rr] for n in NAMES}
        self.comp = np.concatenate([[m["companions"] for m in mm] for mm in self.meta]) \
            if self.ids else np.array([])
        #: appearances on a multi-subject benchmark (SINGLE_SUBJECT's are alone in every run)
        self.multi = np.concatenate([[m["parent"] not in SINGLE_SUBJECT for m in mm]
                                     for mm in self.bmeta]) if self.ids else np.array([], bool)

    def runs_of(self, B):
        return {i: (m, b) for i, m, b in zip(self.ids, self.bmeta, B)}

    @staticmethod
    def arm(ids, B, what):
        a = SC.Arm(what)
        for i, b in zip(ids, B):
            a.put(i, b @ W6, b, "same task")
        return a

    def table(self, xB, refB):
        """Block's paired report of x minus ref, keys renamed; and the Block and the
        per run per pair ALC differences."""
        blk = SC.Block(self.runs_of(xB), self.ids)
        rep, d = blk.paired(self.arm(self.ids, refB, "ref"))
        rep["x_ALC"], rep["x_ALC_run_se"] = rep.pop("shipped_ALC"), rep.pop("shipped_ALC_run_se")
        rep["ref_ALC"], rep["ref_ALC_run_se"] = rep.pop("other_ALC"), rep.pop("other_ALC_run_se")
        rep["share_runs_x_better"] = rep.pop("runs_shipped_better")
        for b in rep.get("budgets", []):
            b["x"], b["ref"] = b.pop("shipped"), b.pop("other")
        return rep, blk, d

    def compare(self, x, ref, shares=None):
        rep, blk, d = self.table(self.B[x], self.B[ref])
        rep["by_companions"] = self.by_companions(blk, d)
        db = [a - b for a, b in zip(self.B[x], self.B[ref])]
        if shares is not None:
            rep["platform_mix"] = self.mix(blk, d, db, shares)
        if self.multi.size and not self.multi.all():
            # the single-subject benchmark is alone in every run, so the 'alone' class
            # (and the platform mix, which weights it most) mixes "alone on its benchmark"
            # with "a benchmark of one subject"; the same readings without it
            ws = {"excluded": list(SINGLE_SUBJECT),
                  "by_companions": self.by_companions(blk, d, keep=self.multi)}
            if shares is not None:
                ws["platform_mix"] = self.mix(blk, d, db, shares, keep=self.multi)
            rep["without_single_subject"] = ws
        return rep

    def mix(self, blk, d, db, shares, keep=None):
        """The difference reweighted to a companion mix: the sum over COMPANIONS
        classes of share x the class's appearance-weighted mean (as by_companions),
        for ALC and per budget, with cluster and stratified bootstrap SEs that
        resample the classes jointly (Boot's W and Ws). None when a class with a
        positive share has no appearance here. keep: a mask of the appearances
        to use (default all)."""
        keep = np.ones(len(self.comp), bool) if keep is None else keep
        sels = []
        for label, lo, hi in COMPANIONS:
            s = float(shares.get(label) or 0.0)
            if s > 0:
                sel = (self.comp >= lo) & (self.comp <= hi) & keep
                if not sel.any():
                    return None
                sels.append((s, sel))

        def one(x):
            point = sum(s * np.sum(blk.w[sel] * x[sel]) / np.sum(blk.w[sel]) for s, sel in sels)
            ses = []
            for W in (blk.boot.W, blk.boot.Ws):
                tot, ok = np.zeros(W.shape[0]), np.ones(W.shape[0], bool)
                for s, sel in sels:
                    v = np.bincount(blk.cl[sel], (blk.w * x)[sel], minlength=blk.K)
                    n = np.bincount(blk.cl[sel], blk.w[sel], minlength=blk.K)
                    den = W @ n
                    ok &= den > 0
                    tot += s * np.divide(W @ v, den, out=np.zeros_like(den), where=den > 0)
                ses.append(float(tot[ok].std(ddof=1)) if ok.sum() > 1 else None)
            return {"mean": r6(point), "cluster_se": r6(ses[0]), "strat_se": r6(ses[1])}

        out = one(blk.flat(d))
        out["budgets"] = [dict(budget=b, **one(blk.flat([r[:, k] for r in db])))
                          for k, b in enumerate(BUDGETS)]
        return out

    def by_companions(self, blk, d, keep=None):
        x = blk.flat(d)
        keep = np.ones(len(self.comp), bool) if keep is None else keep
        out = {}
        for label, lo, hi in COMPANIONS:
            sel = (self.comp >= lo) & (self.comp <= hi) & keep
            if not sel.any():
                continue
            k = len(set(blk.cl[sel].tolist()))
            out[label] = {"mean": r6(np.sum(blk.w[sel] * x[sel]) / np.sum(blk.w[sel])),
                          "cluster_se": r6(blk.sel_se(x, sel)) if k > 1 else None,
                          "appearances": int(sel.sum()), "clusters": k}
        return out

    def variant(self, n):
        run_alc = np.array([float(np.mean(b @ W6)) for b in self.B[n]])
        boot = LC.Boot(self.bmeta, BOOTS)
        ece = [np.asarray(r["ece"], float) for r in self.res[n]]
        calls = sum(r["calls"] for r in self.res[n])
        return {"runs": len(self.ids), "ALC": r6(run_alc.mean()),
                "ALC_run_se": r6(run_alc.std(ddof=1) / math.sqrt(len(run_alc))),
                "ALC_cluster_se": r6(boot.se([b @ W6 for b in self.B[n]])[0]),
                "budgets": [r6(x) for x in np.mean([b.mean(0) for b in self.B[n]], 0)],
                "ece_alc": r6(np.mean([np.mean(e @ W6) for e in ece])),
                "q0": r6(np.mean([np.mean(r["q0"]) for r in self.res[n]])),
                "failures": int(sum(r["failures"] for r in self.res[n])),
                "unconverged": int(sum(r["unconverged"] for r in self.res[n])),
                "mean_call_ms": r6(1e3 * sum(r["mean_s"] * r["calls"] for r in self.res[n]) / calls),
                "max_call_s": r6(max(r["max_s"] for r in self.res[n]))}

    def describe(self):
        flat = [m for mm in self.meta for m in mm]
        comp = np.array([m["companions"] for m in flat])
        cov = np.array([m["cover31"] if m["cover31"] is not None else np.nan for m in flat])
        li = np.array([m["labeled_items"] for m in flat], float)
        out = {"runs": len(self.ids), "appearances": len(flat),
               "pairs_per_run": r6(np.mean([len(m) for m in self.meta])),
               "benchmarks_per_run": r6(np.mean([len({x["bench"] for x in m}) for m in self.meta])),
               "eval_items_per_pair": r6(np.mean([m["eval"] for m in flat])),
               "companions": {label: int(((comp >= lo) & (comp <= hi)).sum())
                              for label, lo, hi in COMPANIONS},
               "single_subject_companions": {
                   label: int(((comp >= lo) & (comp <= hi)
                               & np.array([m["parent"] in SINGLE_SUBJECT for m in flat], bool)).sum())
                   for label, lo, hi in COMPANIONS},
               "cover31": {"all": r6(np.nanmean(cov))},
               "warmup_reached": {f"B{b}": r6(np.mean(li[:, k] >= P.BenchmarkFit.WARMUP))
                                  for k, b in enumerate(BUDGETS) if b}}
        for label, lo, hi in COMPANIONS:
            sel = (comp >= lo) & (comp <= hi)
            if sel.any():
                out["cover31"][label] = r6(np.nanmean(cov[sel]))
        return out


def scope_effect(fp, fb, x, ref):
    """(x - ref under 'pair') minus (x - ref under 'benchmark') on the runs both
    scopes scored, through Block (same pairs: checked)."""
    ids = sorted(set(fp.ids) & set(fb.ids))
    if not ids:
        return None
    ip, ib = [fp.ids.index(i) for i in ids], [fb.ids.index(i) for i in ids]
    for a, b in zip(ip, ib):
        if fp.bmeta[a] != fb.bmeta[b]:
            raise RuntimeError("the two scopes' runs hold different pairs")
    dp = [fp.B[x][a] - fp.B[ref][a] for a in ip]
    db = [fb.B[x][b] - fb.B[ref][b] for b in ib]
    blk = SC.Block({i: (fp.bmeta[a], v) for i, a, v in zip(ids, ip, dp)}, ids)
    rep, _ = blk.paired(Formative.arm(ids, db, "benchmark scope"))
    out = {"runs": rep["runs"], "pair_scope": rep["shipped_ALC"], "benchmark_scope": rep["other_ALC"],
           "diff": rep["diff"], "per_parent": rep["per_parent"], "parent_level": rep["parent_level"]}
    if "budgets" in rep:
        out["budgets"] = [{"budget": b["budget"], "pair_scope": b["shipped"],
                           "benchmark_scope": b["other"], "diff": b["diff"], "run_se": b["run_se"],
                           "cluster_se": b["cluster_se"], "strat_se": b["strat_se"]}
                          for b in rep["budgets"]]
    return out


def mean_se(x):
    x = np.asarray(x, float)
    return {"mean": r6(x.mean()), "se": r6(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None,
            "n": int(len(x))}


class Dense:
    """One dense run: per variant the (pairs, 6) Brier, SEs over the pairs."""

    def __init__(self, row):
        self.meta = row["meta"]
        self.res = row["res"]
        self.B = {n: np.asarray(row["res"][n]["b"], float) for n in NAMES}
        self.cap = row.get("cap")

    def variant(self, n):
        r = self.res[n]
        a = self.B[n] @ W6
        return {"ALC": mean_se(a), "budgets": [r6(x) for x in self.B[n].mean(0)],
                "ece_alc": r6(np.mean(np.asarray(r["ece"], float) @ W6)),
                "q0": r6(np.mean(r["q0"])), "failures": r["failures"],
                "unconverged": r["unconverged"],
                "mean_call_ms": r6(1e3 * r["mean_s"]), "max_call_s": r6(r["max_s"]),
                "secs": r["secs"]}

    def diff(self, x, ref):
        d = self.B[x] - self.B[ref]
        a = d @ W6
        out = {"ALC": mean_se(a), "share_pairs_x_better": r6(np.mean(a < 0)),
               "budgets": [dict(budget=b, **mean_se(d[:, k]), alc_part=r6(W6[k] * d[:, k].mean()))
                           for k, b in enumerate(BUDGETS)]}
        return out, d

    def describe(self):
        cov = [m["cover31"] for m in self.meta if m["cover31"] is not None]
        return {"pairs": len(self.meta), "cap": self.cap,
                "eval_items_scored": r6(np.mean([m["eval"] for m in self.meta])),
                "eval_items_all": r6(np.mean([m["eval_all"] for m in self.meta])),
                "responses_scored": int(sum(m["resp"] for m in self.meta)),
                "cover31": r6(np.mean(cov)) if cov else None,
                "labeled_items": self.meta[0]["labeled_items"] if self.meta else None,
                "mean_p": r6(np.mean([m["p"] for m in self.meta if m["p"] is not None]))}


def across(values):
    """Mean ± SE across benchmarks of per benchmark means (None when fewer than two)."""
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": r6(np.mean(v)), "se": r6(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None,
            "range": [r6(min(v)), r6(max(v))], "n": len(v)}


def identities(rows):
    """The checks that hold by construction, measured: at B0 the three hier variants
    coincide; legacy and legacy_nopool coincide wherever the target's benchmark
    holds fewer than WARMUP distinct labeled items; and where they may differ."""
    h0, lg_below, lg_above, n_above = 0.0, 0.0, 0.0, 0
    for row in rows.values():
        li = np.array([m["labeled_items"] for m in row["meta"]], float)
        B = {n: np.asarray(row["res"][n]["b"], float) for n in NAMES}
        for n in ("ship_nolevel", "ship_own"):
            h0 = max(h0, float(np.max(np.abs(B[n][:, 0] - B["ship"][:, 0]))))
        d = np.abs(B["legacy"] - B["legacy_nopool"])
        below = li < P.BenchmarkFit.WARMUP
        if below.any():
            lg_below = max(lg_below, float(d[below].max()))
        if (~below).any():
            lg_above = max(lg_above, float(d[~below].max()))
            n_above += int((~below).sum())
    return {"hier_variants_at_B0_max_abs": h0,
            "legacy_vs_nopool_below_warmup_max_abs": lg_below,
            "legacy_vs_nopool_at_or_above_warmup_max_abs": lg_above,
            "pair_budget_cells_at_or_above_warmup": n_above}


def reading_table(grid):
    """Per comparison, the six cells the claim is read from, side by side: dense
    (mean of the four benchmarks' means ± SE across them), formative (all
    appearances ± run / cluster / stratified SE, and the parent-level mean ± SE
    over the four multi-subject parents, the dense line's like) and formative at
    the platform's companion mix (± cluster / stratified SE), under each scope.
    'ratio_*' are descriptive quotients of two means, no SE."""
    out = {}
    for key, g in grid.items():
        row = {}
        for sc in SCOPES:
            c = g.get(sc, {})
            f = c.get("formative") or {}
            d = c.get("dense")
            fd, pl, mx = f.get("diff"), f.get("parent_level"), f.get("platform_mix")
            row[f"dense {sc}"] = None if d is None else {"mean": d["mean"], "se": d["se"], "n": d["n"]}
            row[f"formative {sc}"] = None if fd is None else dict(
                fd, parent_level_mean=pl["mean"] if pl else None, parent_level_se=pl["se"] if pl else None)
            row[f"platform mix {sc}"] = None if mx is None else {k: mx[k] for k in ("mean", "cluster_se",
                                                                                    "strat_se")}
            if d is not None and pl and pl.get("mean") is not None and d["mean"]:
                row[f"ratio_formative_parent_level_over_dense {sc}"] = r6(pl["mean"] / d["mean"])
        dp, db = row.get("dense pair"), row.get("dense benchmark")
        if dp and db and dp["mean"]:
            row["ratio_dense_benchmark_over_pair"] = r6(db["mean"] / dp["mean"])
        out[key] = row
    return out


def single_subject_reading(form):
    """The single-subject benchmark (SINGLE_SUBJECT) is alone in every run by
    construction, so the formative 'alone' class, and the platform mix that weights
    it most, mix "a pair alone on its benchmark" with "a benchmark of one subject".
    Per scope: how many of the 'alone' appearances are single-subject; and per
    comparison the 'alone' class and the platform mix with and without them (the
    same Block and resamples; the run-size weights are the runs' as scored)."""
    out = {"what": single_subject_reading.__doc__.split("\n\n")[0].replace("\n    ", " "),
           "excluded": list(SINGLE_SUBJECT), "appearances": {}, "comparisons": {}}
    for sc, f in form.items():
        d = f["describe"]
        n, k = d["companions"].get("alone", 0), d["single_subject_companions"].get("alone", 0)
        out["appearances"][sc] = {"alone": n, "alone_single_subject": k,
                                  "share_of_alone": r6(k / n) if n else None,
                                  "single_subject_by_class": d["single_subject_companions"]}
    for x, ref, _ in COMPARISONS:
        key = f"{x} - {ref}"
        row = {}
        for sc, f in form.items():
            c = f["comparisons"][key]
            ws = c.get("without_single_subject")
            if ws is None:
                continue
            pm, pw = c.get("platform_mix"), ws.get("platform_mix")
            row[sc] = {
                "alone": c["by_companions"].get("alone"),
                "alone_without": ws["by_companions"].get("alone"),
                "platform_mix": None if pm is None else {k_: pm[k_] for k_ in ("mean", "cluster_se", "strat_se")},
                "platform_mix_without": None if pw is None else {k_: pw[k_] for k_ in ("mean", "cluster_se",
                                                                                      "strat_se")}}
        out["comparisons"][key] = row
    return out


def dense_crosscheck(rows, path=HE_JSON):
    """Dense runs scored whole against hier_eval.json's stored R2 rows (per pair
    Brier by budget, rounded to 6 decimals there): max |difference| per variant
    and scope, and whether it is within HE_TOL. hier_eval ran an older library
    (bba726c), so a difference is recorded, not an error."""
    try:
        with open(path) as f:
            raw = json.load(f)["raw"]
    except Exception as exc:
        return {"checked": False, "why": repr(exc)}
    out = {}
    for (kind, b, sc), r in sorted(rows.items()):
        if kind != "dense" or r.get("cap") is not None:
            continue
        for v, name in HE_DENSE.items():
            st = raw.get(f"r2|{b}|{sc}|{name}")
            if st is None:
                continue
            a = np.asarray(r["res"][v]["b"], float)
            c = np.asarray([x[:len(BUDGETS)] for x in st["rows"]], float)
            d = float(np.max(np.abs(a - c))) if a.shape == c.shape else None
            out[f"{b} {sc} {v}"] = {"max_abs": d, "within_tol": d is not None and d <= HE_TOL,
                                    "pairs": int(a.shape[0])}
    return {"checked": bool(out), "tol": HE_TOL, "against": os.path.relpath(path, ROOT),
            "cells": out}


def load_recheck(rows_dir=ROWS):
    """recheck.json, with whether it ran the script and library of this summary."""
    p = recheck_path(rows_dir)
    if not os.path.exists(p):
        return {"checked": False}
    with open(p) as f:
        rec = json.load(f)
    return {"checked": True, "all_bit_identical": rec["all_bit_identical"],
            "script_is_this_one": rec["script"] == SCRIPT_SHA, "lib_is_this_one": rec["lib"] == lib_digests(),
            "row_scripts": sorted({x["row_script"] for x in rec["tasks"]}), "utc": rec["utc"],
            "tasks": [{"task": x["task"], "bit_identical": x["bit_identical"]} for x in rec["tasks"]]}


def stage_summarise(rows_dir=ROWS, out_path=OUT):
    t0 = time.time()
    check_level()
    rows = load_rows(rows_dir)
    if not rows:
        raise SystemExit("no rows")
    libs = {json.dumps(r["lib"], sort_keys=True) for r in rows.values()}
    scripts = sorted({r["script"] for r in rows.values()})
    planned = tasks()
    missing = [list(t) for t in planned if t not in rows]
    out = {"what": "P1.11: cross-subject pooling on and off, by split scope and run size "
                   "(review W5; experiments/pooling_decomposition.py)",
           "design": {"variants": VARIANTS, "pooling": [list(c) for c in POOLING],
                      "reference": [list(c) for c in REFERENCE], "level": LEVEL,
                      "formative": {"sampler": "official.sample_run defaults (benchmark-first)",
                                    "seed": SEED, "runs": N_FORMATIVE},
                      "dense": list(DENSE), "single_subject_not_run": list(SINGLE_SUBJECT),
                      "eval_cap": EVAL_CAP, "scopes": list(SCOPES), "boots": BOOTS,
                      "nopool_warmup": NOPOOL_WARMUP, "warmup": P.BenchmarkFit.WARMUP},
           "complete": not missing, "missing": missing}
    try:
        mix = platform_mix()
    except Exception as exc:                       # recorded, not required
        mix = {"error": repr(exc)}
    out["platform_mix"] = dict(mix, what="the platform's scored formative runs 1-3, pair appearances "
                                         "by how many pairs of their benchmark the run holds; "
                                         "formative differences reweighted to these shares are "
                                         "'platform_mix'")
    shares = mix.get("shares")

    # formative
    fm = {sc: Formative(rows, sc) for sc in SCOPES}
    form = {}
    for sc, f in fm.items():
        if not f.ids:
            continue
        form[sc] = {"runs": len(f.ids), "describe": f.describe(),
                    "variants": {n: f.variant(n) for n in NAMES},
                    "comparisons": {f"{x} - {ref}": dict(what=w, **f.compare(x, ref, shares))
                                    for x, ref, w in COMPARISONS}}
    out["formative"] = form
    if all(fm[sc].ids for sc in SCOPES):
        out["formative_scope_effect"] = {f"{x} - {ref}": scope_effect(fm["pair"], fm["benchmark"], x, ref)
                                         for x, ref, _ in COMPARISONS}

    # dense
    dense, dd = {}, {}
    for b in DENSE:
        for sc in SCOPES:
            r = rows.get(("dense", b, sc))
            if r is None:
                continue
            D = dd[(b, sc)] = Dense(r)
            dense.setdefault(b, {})[sc] = {
                "describe": D.describe(), "variants": {n: D.variant(n) for n in NAMES},
                "comparisons": {f"{x} - {ref}": D.diff(x, ref)[0] for x, ref, _ in COMPARISONS}}
        if all((b, sc) in dd for sc in SCOPES):
            P_, B_ = dd[(b, "pair")], dd[(b, "benchmark")]
            if [(m["bench"], m["subject"]) for m in P_.meta] != [(m["bench"], m["subject"]) for m in B_.meta]:
                raise RuntimeError(f"dense {b}: the two scopes hold different pairs")
            eff = {}
            for x, ref, _ in COMPARISONS:
                dp, db = P_.diff(x, ref)[1], B_.diff(x, ref)[1]
                e = dp - db
                eff[f"{x} - {ref}"] = {"ALC": mean_se(e @ W6),
                                       "budgets": [dict(budget=bb, **mean_se(e[:, k]))
                                                   for k, bb in enumerate(BUDGETS)]}
            dense[b]["scope_effect"] = eff
    out["dense"] = dense
    dsum = {}
    for x, ref, _ in COMPARISONS:
        key = f"{x} - {ref}"
        row = {}
        for sc in SCOPES:
            vals = [dense[b][sc]["comparisons"][key]["ALC"]["mean"] if sc in dense.get(b, {}) else None
                    for b in DENSE]
            row[sc] = {"per_benchmark": dict(zip(DENSE, vals)), "across": across(vals),
                       "budgets_across": [across([dense[b][sc]["comparisons"][key]["budgets"][k]["mean"]
                                                  for b in DENSE if sc in dense.get(b, {})])
                                          for k in range(len(BUDGETS))]}
        effs = [dense[b]["scope_effect"][key]["ALC"]["mean"] if "scope_effect" in dense.get(b, {}) else None
                for b in DENSE]
        row["scope_effect"] = {"per_benchmark": dict(zip(DENSE, effs)), "across": across(effs)}
        dsum[key] = row
    out["dense_across_benchmarks"] = dsum

    # the decomposition: run size x split scope, per comparison
    grid = {}
    for x, ref, w in COMPARISONS:
        key = f"{x} - {ref}"
        g = {"what": w}
        for sc in SCOPES:
            cell = {}
            if sc in form:
                c = form[sc]["comparisons"][key]
                cell["formative"] = {"diff": c["diff"], "parent_level": c["parent_level"],
                                     "by_companions": c["by_companions"],
                                     "platform_mix": c.get("platform_mix")}
                per = {}
                for b in DENSE:
                    fpar = c["per_parent"].get(b)
                    dcell = dense.get(b, {}).get(sc, {}).get("comparisons", {}).get(key)
                    if fpar is None or dcell is None:
                        continue
                    fse, dse = fpar["cluster_se"], dcell["ALC"]["se"]
                    per[b] = {"formative": fpar["mean"], "formative_cluster_se": fse,
                              "dense": dcell["ALC"]["mean"], "dense_se": dse,
                              "dense_minus_formative": r6(dcell["ALC"]["mean"] - fpar["mean"]),
                              "se_if_independent": r6(math.hypot(fse, dse))
                              if fse is not None and dse is not None else None}
                cell["per_benchmark"] = per
            cell["dense"] = dsum[key][sc]["across"]
            g[sc] = cell
        grid[key] = g
    out["decomposition"] = grid
    out["reading"] = reading_table(grid)
    out["single_subject_confound"] = single_subject_reading(form)

    # checks
    reps = [r["repro"] for r in rows.values() if r.get("repro")]
    out["checks"] = {
        "level_is_shipped": model_level() == LEVEL,
        "library_digests_agree_across_rows": len(libs) == 1,
        "library_digests": json.loads(next(iter(libs))) if len(libs) == 1 else None,
        "scripts": scripts,
        "reproduction": {"against": os.path.relpath(RS_ROWS, ROOT), "runs": len(reps),
                         "all_ok": bool(reps) and all(x["ok"] for x in reps),
                         "same_lib": bool(reps) and all(x.get("same_lib") for x in reps),
                         "max_abs": {v: max(x["max_abs"][v] for x in reps) for v in REPRO["configs"]}
                         if reps else None, "tol": REPRO_TOL},
        "identities": identities(rows),
        "dense_crosscheck": dense_crosscheck(rows),
        "recheck": load_recheck(rows_dir),
        "failures": {n: int(sum(r["res"][n]["failures"] for r in rows.values())) for n in NAMES},
        "unconverged": {n: int(sum(r["res"][n]["unconverged"] for r in rows.values())) for n in NAMES},
    }
    secs = [r["task_s"] for r in rows.values()]
    out["provenance"] = {
        "head": git("rev-parse", "--short", "HEAD"),
        "dirty": git("status", "--porcelain", "paiec", "experiments", "submission", "tests"),
        "script_now": SCRIPT_SHA, "lib_now": lib_digests(),
        "python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
        "commands": ["python experiments/pooling_decomposition.py score",
                     "python experiments/pooling_decomposition.py recheck",
                     "python experiments/pooling_decomposition.py summarise"],
        "rows_scored_by": scripts, "summary_script_differs_from_rows":
            any(s != SCRIPT_SHA for s in scripts),
        "tasks": len(rows), "task_seconds_total": round(sum(secs), 1),
        "task_seconds_formative": round(sum(r["task_s"] for t, r in rows.items() if t[0] == "formative"), 1),
        "task_seconds_dense": {f"{t[1]} {t[2]}": r["task_s"] for t, r in rows.items() if t[0] == "dense"},
        "rss_gb_max": max(r["rss_gb"] for r in rows.values()),
        "first_row_utc": min(r["utc"] for r in rows.values()),
        "last_row_utc": max(r["utc"] for r in rows.values()),
        "summarised_utc": utcnow(), "summarise_s": None}
    out["provenance"]["summarise_s"] = round(time.time() - t0, 1)
    write_json(out_path, out)
    with open(out_path) as f:              # re-indent for reading
        obj = json.load(f)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    show(out)
    return out


def fmt(d):
    if d is None:
        return "-"
    if "cluster_se" in d:
        return f"{d['mean']:+.4f} ± {d['run_se']:.4f} / {d['cluster_se']:.4f} / {d['strat_se']:.4f}"
    if d.get("se") is None:
        return f"{d['mean']:+.4f}"
    return f"{d['mean']:+.4f} ± {d['se']:.4f}"


def fmt_mix(d):
    if not d:
        return "-"
    return f"{d['mean']:+.4f} ± {d['cluster_se']:.4f} / {d['strat_se']:.4f}"


def fmt_class(d):
    if not d:
        return "-"
    se = d.get("cluster_se")
    return f"{d['mean']:+.4f}" + ("" if se is None else f" ± {se:.4f}") + f" ({d['appearances']})"


def show(out):
    print("\nformative (± run / cluster / stratified SE), formative at the platform's companion mix "
          "(± cluster / stratified SE) and dense (mean across benchmarks ± SE):")
    for key, g in out["decomposition"].items():
        print(f"  {key}: {g['what']}")
        for sc in SCOPES:
            c = g.get(sc, {})
            f = c.get("formative", {})
            print(f"    {sc:9s} formative {fmt(f.get('diff'))}   platform mix {fmt_mix(f.get('platform_mix'))}"
                  f"   dense {fmt(c.get('dense'))}")
    ss = out.get("single_subject_confound") or {}
    if ss.get("appearances"):
        print("\nwithout the single-subject benchmark (alone in every run):",
              json.dumps(ss["appearances"]))
        for key in ("ship - legacy", "ship - ship_own"):
            for sc, v in (ss["comparisons"].get(key) or {}).items():
                print(f"  {key} {sc}: alone {fmt_class(v['alone'])} -> {fmt_class(v['alone_without'])}; "
                      f"platform mix {fmt_mix(v['platform_mix'])} -> {fmt_mix(v['platform_mix_without'])}")
    print("checks:", json.dumps({k: v for k, v in out["checks"].items()
                                 if k not in ("library_digests",)}, default=str)[:2000])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("score", "status", "recheck", "summarise"))
    ap.add_argument("--only", choices=("formative", "dense"))
    ap.add_argument("--limit", type=int, help="score at most this many tasks")
    ap.add_argument("--rows", default=ROWS, help="row directory (default: ROWS)")
    ap.add_argument("--out", default=OUT, help="summary file (default: OUT)")
    a = ap.parse_args(argv)
    if a.stage == "score":
        stage_score(a.only, rows=a.rows, limit=a.limit)
    elif a.stage == "status":
        stage_status(a.rows)
    elif a.stage == "recheck":
        stage_recheck(a.rows)
    else:
        stage_summarise(a.rows, a.out)


if __name__ == "__main__":
    main()
