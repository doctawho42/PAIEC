"""Acquisition policies under the official protocol, against random and against
random (R5).

docs/plans/p2_final_plan.md, R5, and the v2 outline (T4, App F.8) need one row
for acquisition under the official protocol. The report's App F.8 had only the
legacy replica's numbers ("not re-measured under the official protocol"). The
measurement was made on 2026-10-06 by exploratory scripts in a session scratch
directory (acq_official.py, acq_seeded.py and their two summaries). This script
is their port: it recomputes every row from the repository and checks each
number against the value the scratch recorded (`scratch` in the results file).
It selects, tunes and ships nothing; the submission keeps the platform's random
acquisition (no labeling.py).

Setup. hier-ship's configuration ('hier G mu0=-2.50 sm=2.50 as=0.50', LEVEL)
over leave-one-parent-out priors, one model per parent dispatched on the
anonymous benchmark_id (experiments/level_calibration.factory), on test-like
runs (regime 'tl': paiec.testlike at its defaults, seed 2,
level_calibration.draw('tl', i)), scored by paiec.official.run_official:
budget-major, one shared `labeled`, fresh instances per checkpoint, split scope
'pair', the mirror hook (deepcopy=False, as the scratch). Every policy of a run
is scored on the same run, so differences are paired.

Policies (streaming hooks, as submission/labeling.py's acquisition_function
would be). "With slack" means labels remain and the pool is longer than the
allowance; without slack every remaining candidate is taken, so 31 labels fill
whenever the pool allows, as under the platform's rule.
  random     None: the platform's rule (official.default_policy: sha256 of the
             canonical JSON of [context, input] below labels_remaining /
             items_remaining)
  placebo    the same rule on a salted key: another uniformly random subset of
             the same size, so placebo - random is the noise floor of a policy
             comparison ("random against random")
  balanced   with slack, take a candidate whose non-numeric feature group
             (hier.item_groups, numeric values dropped) the pair has not
             labeled yet; otherwise the platform's rule
  distinct   with slack, skip a candidate visibly identical (content, features,
             interactors) to one the pair already labeled; otherwise the
             platform's rule
  uncertain  BLE's rule on hier's own acquisition-time prediction: with slack,
             take iff |p - 0.5| <= 0.15

Batches. Both score test-like runs 0-89 (RUNS), so the two batches are paired
with each other too.
  fixed    run_official seed 0, the persistent split and stream order: a pair
           that recurs across runs keeps one label subset per policy, so the
           run-level SE understates the noise
  salted   seed 1000 + i on run i (SALT), so recurring pairs get other splits
           and stream orders
The scratch scored the fixed batch on the 91 runs that had finished when its
job was stopped (0-93 without 83, 90 and 92) and the salted batch on 0-89
without the distinct arm. The fixed batch here also scores runs 91 and 93, so
that the scratch's set is recomputed and checked (`scratch_subset`); the main
numbers use runs 0-89 in both batches. The salted batch's distinct arm is new.

Summaries, per batch: each policy's mean ALC; policy - random, policy -
placebo, policy - (random + placebo)/2 ("the two-arm mean") and placebo -
random, each with the run-level SE (sd/sqrt(R), the scratch's) and a pair-
cluster bootstrap SE over (parent, subject) (testlike.cluster_key; 2,000
resamples, seed 0, a ratio estimator as in experiments/harness.py); by budget;
per parent; identical runs. For reference, the harness gate's test-like bar
(TL delta <= -0.002, report section 4.4) against each comparison.

Rows. Each (batch, run) is one task, checkpointed as JSON under --rows (default
data/acquisition_official_rows, gitignored) and skipped when present with the
same library digest. The results file embeds every row (per run and policy:
ALC, Brier by budget, per pair Brier by budget and label count, pairs as
(parent, cluster index): no anonymous id is written), so --summarise-only can
rebuild the summary from it alone (--from-results [PATH], default the results
file). Such a rebuild reads no scratch rows, so it would lose the scratch-row
comparison and the rows' digest the results file records: it must be written
elsewhere (--out OTHER; the script refuses to overwrite PATH or the default
results file), and it copies the source's scratch-row record, marked as not
re-run (scratch.rows_in_source).

Run:  python experiments/acquisition_official.py --jobs 3 [--scratch DIR]
      (about 37 CPU-minutes per batch; 182 run tasks of 5 policies)
      python experiments/acquisition_official.py --summarise-only [--rows DIR] [--scratch DIR]
      python experiments/acquisition_official.py --summarise-only --from-results [PATH] --out OTHER
--scratch DIR names the session scratch directory that holds acq_rows/ and
acq_seeded_rows/; with it every stored scratch row is compared run by run and
policy by policy (bit for bit expected). Without it that check is "not run" and
the embedded summary values are still checked. Writes the results file only if
every required check passes.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec import official as O  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.testlike import cluster_key, parent_of  # noqa: E402

OUT = os.path.join(ROOT, "results", "acquisition_official.json")
ROWS = os.path.join(ROOT, "data", "acquisition_official_rows")
SHIP_NAME = "hier G mu0=-2.50 sm=2.50 as=0.50"
REGIME = "tl"
RUNS = tuple(range(90))
SCRATCH_FIXED_RUNS = tuple(i for i in range(94) if i not in (83, 90, 92))
SALT = 1000
BATCHES = ("fixed", "salted")
POLICY_NAMES = ("random", "placebo", "balanced", "distinct", "uncertain")
GATE_TL = -0.002
NBOOT, BOOT_SEED = 2000, 0
W = np.asarray(WEIGHTS, float)
#: the files whose content decides a row (as experiments/harness.py's LIB, plus
#: the data loader and the IRT/MCQ code hier calls)
LIB = ("paiec/hier.py", "paiec/prior.py", "paiec/subjects.py", "paiec/official.py",
       "paiec/testlike.py", "paiec/mcq.py", "paiec/predict.py", "paiec/fitting.py",
       "paiec/irt.py", "paiec/data.py", "paiec/evaluator.py", "paiec/baselines.py",
       "experiments/level_calibration.py", "submission/model.py")


def seed_of(batch, i):
    return 0 if batch == "fixed" else SALT + i


# --- the policies (verbatim from the scratch acq_official.py) -----------------------------------

_NUM = re.compile(r"\s*[-+]?\d+(\.\d+)?\s*")


def _own(input, labeled):
    """The labels this pair (subject and benchmark of `input`) already holds."""
    s, b = input[0], input[1]["benchmark_id"]
    return [e for e in (labeled or []) if e[0][0] == s and e[0][1]["benchmark_id"] == b]


def group_sig(item):
    """The item's non-numeric feature groups, as hier reads them."""
    from paiec.hier import item_groups
    g = item_groups(item)
    return tuple(sorted((k, v) for k, v in g.items() if not _NUM.fullmatch(str(v))))


def _visible(item):
    return (item["item_content"], item["item_features"], item["interactors"])


def balanced(input, *, labeled=None, context=None):
    """With slack, take a candidate whose non-numeric feature group this pair has not labeled yet."""
    if context["labels_remaining"] <= 0:
        return False
    if context["items_remaining"] <= context["labels_remaining"]:
        return True
    seen = {group_sig(e[0][1]) for e in _own(input, labeled)}
    if group_sig(input[1]) not in seen:
        return True
    return bool(O.default_policy(context, input))


def distinct(input, *, labeled=None, context=None):
    """With slack, skip a candidate visibly identical to one this pair already labeled."""
    if context["labels_remaining"] <= 0:
        return False
    if context["items_remaining"] <= context["labels_remaining"]:
        return True
    if _visible(input[1]) in {_visible(e[0][1]) for e in _own(input, labeled)}:
        return False
    return bool(O.default_policy(context, input))


def uncertain(input, prediction=None, labeled=None, context=None):
    """BLE's policy, on hier's own prediction."""
    if context["labels_remaining"] <= 0:
        return False
    if context["items_remaining"] <= context["labels_remaining"]:
        return True
    return bool(abs(prediction - 0.5) <= 0.15)


def placebo(input, *, labeled=None, context=None):
    """The platform's rule on a salted key: another uniformly random subset (noise floor)."""
    key = json.dumps(["placebo", context, input], sort_keys=True, separators=(",", ":"))
    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return bool(u < min(1, context["labels_remaining"] / context["items_remaining"]))


POLICIES = {"random": None, "placebo": placebo, "balanced": balanced, "distinct": distinct,
            "uncertain": uncertain}


# --- helpers --------------------------------------------------------------------------------------

def rel(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_digest():
    h = hashlib.sha256()
    for p in LIB:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def policy_digest():
    src = "".join(inspect.getsource(f) for f in (_own, group_sig, _visible, balanced, distinct,
                                                    uncertain, placebo))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


class Checks:
    """Every check, recorded; a failed required one stops the write."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, got=None, want=None, required=True, note=None):
        self.items.append({"check": name, "ok": bool(ok), "got": got, "want": want,
                           "required": required, **({"note": note} if note else {})})
        return bool(ok)

    def failed(self):
        return [c for c in self.items if c["required"] and not c["ok"]]


# --- scoring one run --------------------------------------------------------------------------------

def row_path(rows_dir, batch, i):
    return os.path.join(rows_dir, f"{batch}_{i}.json")


def score_run(task):
    """All five policies on one (batch, run); the row is written to rows_dir."""
    batch, i, rows_dir = task
    path = row_path(rows_dir, batch, i)
    lib, pol = lib_digest(), policy_digest()
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("lib") == lib and old.get("policy") == pol:
            return path
    warnings.simplefilter("ignore")
    import experiments.level_calibration as LC
    run = LC.draw(REGIME, i)
    seed = seed_of(batch, i)
    row = {"batch": batch, "i": i, "seed": seed, "regime": REGIME, "config": SHIP_NAME,
           "lib": lib, "policy": pol, "pairs": None, "policies": {}}
    for name in POLICY_NAMES:
        made = []
        t = time.perf_counter()
        res = O.run_official(run, LC.factory([SHIP_NAME], run, made), POLICIES[name], seed=seed,
                             deepcopy=False)
        secs = time.perf_counter() - t
        pairs = [{"parent": parent_of(res["names"][r["benchmark_id"]]), "subject": r["subject_id"],
                  "n_items": r["n_items"]} for r in res["rows"]]
        if row["pairs"] is None:
            row["pairs"] = pairs
        elif pairs != row["pairs"]:
            raise RuntimeError(f"run {batch} {i}: policy {name} scored other pairs")
        row["policies"][name] = {
            "ALC": res["brier"]["ALC"], "by_budget": [res["brier"][b] for b in BUDGETS],
            "pair_brier": [[r["brier"][b] for b in BUDGETS] for r in res["rows"]],
            "n_labels": [r["n_labels"] for r in res["rows"]],
            "mode": res["acquisition_mode"], "secs": round(secs, 2),
            "failures": int(sum(getattr(m, "failures", 0) for m in made))}
    os.makedirs(rows_dir, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, path)
    return path


def tasks_for(rows_dir, with_scratch_runs=True):
    fixed = sorted(set(RUNS) | (set(SCRATCH_FIXED_RUNS) if with_scratch_runs else set()))
    out = [("salted", i, rows_dir) for i in RUNS] + [("fixed", i, rows_dir) for i in fixed]
    return sorted(out, key=lambda t: (t[1], t[0]))


def collect(tasks, jobs):
    todo = [t for t in tasks if not _fresh_row(row_path(t[2], t[0], t[1]))]
    print(f"{len(tasks) - len(todo)} rows stored, {len(todo)} to score on {jobs} processes", flush=True)
    t0 = time.time()
    if jobs <= 1:
        for k, t in enumerate(todo):
            score_run(t)
            print(f"  {k + 1}/{len(todo)} {t[0]} {t[1]} ({time.time() - t0:.0f} s)", flush=True)
        return
    with ProcessPoolExecutor(jobs) as ex:
        for k, p in enumerate(ex.map(score_run, todo)):
            print(f"  {k + 1}/{len(todo)} {os.path.basename(p)} ({time.time() - t0:.0f} s)", flush=True)


def _fresh_row(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return False
    return d.get("lib") == lib_digest() and d.get("policy") == policy_digest()


def load_rows(rows_dir):
    rows = {}
    for p in sorted(glob.glob(os.path.join(rows_dir, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        rows[(d["batch"], d["i"])] = d
    return rows


# --- statistics -----------------------------------------------------------------------------------

def _boot_se(d_pair, w_pair, keys, rng):
    """Pair-cluster bootstrap SE of sum(w d) / sum(w) (a ratio estimator: a resampled
    cluster brings its weight with it), clusters = keys."""
    uk = sorted(set(keys))
    ix = {k: j for j, k in enumerate(uk)}
    S = np.zeros((len(uk),) + np.shape(d_pair)[1:])
    Wc = np.zeros(len(uk))
    for t, k in enumerate(keys):
        S[ix[k]] += w_pair[t] * d_pair[t]
        Wc[ix[k]] += w_pair[t]
    bs = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(uk), len(uk))
        bs.append(S[pick].sum(0) / Wc[pick].sum())
    return np.std(np.asarray(bs), axis=0)


def _pair_arrays(rows, runs, arm):
    """Per pair appearance over `runs`: Brier by budget (pairs x 6) of `arm` (a policy
    name, or 'two_arm' = the mean of random and placebo), run weight 1/n, cluster key,
    parent."""
    B, w, keys, par = [], [], [], []
    for i in runs:
        r = rows[i]
        n = len(r["pairs"])
        if arm == "two_arm":
            b = 0.5 * (np.asarray(r["policies"]["random"]["pair_brier"]) +
                       np.asarray(r["policies"]["placebo"]["pair_brier"]))
        else:
            b = np.asarray(r["policies"][arm]["pair_brier"])
        for j, p in enumerate(r["pairs"]):
            B.append(b[j])
            w.append(1.0 / n)
            keys.append(cluster_key(p["parent"], p["subject"]))
            par.append(p["parent"])
    return np.asarray(B, float), np.asarray(w), keys, par


def _run_values(rows, runs, arm):
    """Per run: (ALC, Brier by budget) of `arm`."""
    a, b = [], []
    for i in runs:
        P = rows[i]["policies"]
        if arm == "two_arm":
            a.append(0.5 * (P["random"]["ALC"] + P["placebo"]["ALC"]))
            b.append(0.5 * (np.asarray(P["random"]["by_budget"]) + np.asarray(P["placebo"]["by_budget"])))
        else:
            a.append(P[arm]["ALC"])
            b.append(np.asarray(P[arm]["by_budget"], float))
    return np.asarray(a), np.asarray(b)


def compare(rows, runs, arm, ref, rng):
    """arm - ref on identical runs: run-level and pair-cluster SEs, by budget, per parent."""
    a, ab = _run_values(rows, runs, arm)
    r, rb = _run_values(rows, runs, ref)
    d, db = a - r, ab - rb
    R = len(runs)
    Ba, w, keys, par = _pair_arrays(rows, runs, arm)
    Br, _, _, _ = _pair_arrays(rows, runs, ref)
    dp = np.column_stack([(Ba - Br) @ W, Ba - Br])         # ALC, then the six budgets
    se_c = _boot_se(dp, w, keys, rng)
    est_pair = (w[:, None] * dp).sum(0) / w.sum()
    per_parent = {}
    for p in sorted(set(par)):
        m = np.array([q == p for q in par])
        per_parent[p] = float((w[m] * dp[m, 0]).sum() / w[m].sum())
    return {"runs": R, "D": float(d.mean()),
            "run_se": float(d.std(ddof=1) / np.sqrt(R)) if R > 1 else None,
            "cluster_se": float(se_c[0]), "clusters": len(set(keys)),
            "D_pair_estimator": float(est_pair[0]),
            "by_budget": [float(x) for x in db.mean(0)],
            "by_budget_run_se": [float(x) for x in db.std(0, ddof=1) / np.sqrt(R)] if R > 1 else None,
            "by_budget_cluster_se": [float(x) for x in se_c[1:]],
            "identical_runs": int((np.abs(d) < 1e-12).sum()),
            "per_parent": per_parent,
            "tl_bar": GATE_TL, "meets_tl_bar": bool(d.mean() <= GATE_TL)}


def summarise(rows, rng_seed=BOOT_SEED):
    """Both batches on RUNS, and the fixed batch on the scratch's run set."""
    out = {}
    sets = {"fixed": ("fixed", RUNS), "salted": ("salted", RUNS),
            "fixed_scratch_subset": ("fixed", SCRATCH_FIXED_RUNS)}
    for name, (batch, runs) in sets.items():
        sub = {i: rows[(batch, i)] for i in runs if (batch, i) in rows}
        if len(sub) < len(runs):
            out[name] = None
            continue
        rng = np.random.default_rng(rng_seed)
        s = {"batch": batch, "runs": len(runs), "run_indices": list(runs),
             "seed": "0" if batch == "fixed" else f"{SALT} + i",
             "mean_ALC": {p: float(np.mean([sub[i]["policies"][p]["ALC"] for i in runs]))
                          for p in POLICY_NAMES},
             "mean_by_budget": {p: [float(x) for x in np.mean([sub[i]["policies"][p]["by_budget"]
                                                                for i in runs], 0)]
                                for p in POLICY_NAMES},
             "failures": {p: int(sum(sub[i]["policies"][p]["failures"] for i in runs))
                          for p in POLICY_NAMES},
             "mean_labels_per_pair": {p: float(np.mean([np.mean(sub[i]["policies"][p]["n_labels"])
                                                         for i in runs])) for p in POLICY_NAMES}}
        s["placebo_minus_random"] = compare(sub, runs, "placebo", "random", rng)
        for p in ("balanced", "distinct", "uncertain"):
            s[p] = {"minus_random": compare(sub, runs, p, "random", rng),
                    "minus_placebo": compare(sub, runs, p, "placebo", rng),
                    "minus_two_arm_mean": compare(sub, runs, p, "two_arm", rng)}
        out[name] = s
    if out.get("fixed") and out.get("salted"):
        out["range"] = ranges(out)
    return out


def ranges(s):
    """The quantities the report quotes, over both batches on runs 0-89."""
    two = {f"{b} {p}": s[b][p]["minus_two_arm_mean"]["D"] for b in ("fixed", "salted")
           for p in ("balanced", "distinct", "uncertain")}
    vs_r = {f"{b} {p}": s[b][p]["minus_random"]["D"] for b in ("fixed", "salted")
            for p in ("balanced", "distinct", "uncertain")}
    plc = {b: s[b]["placebo_minus_random"]["D"] for b in ("fixed", "salted")}
    return {"policies_minus_two_arm_mean": [min(two.values()), max(two.values())],
            "policies_minus_two_arm_mean_each": two,
            "policies_minus_random": [min(vs_r.values()), max(vs_r.values())],
            "policies_minus_random_each": vs_r,
            "placebo_minus_random": plc,
            "largest_abs_placebo_minus_random": max(abs(v) for v in plc.values()),
            "any_meets_tl_bar_vs_two_arm_mean": any(v <= GATE_TL for v in two.values()),
            "meets_tl_bar_vs_random": sorted(k for k, v in vs_r.items() if v <= GATE_TL)}


# --- the scratch values ---------------------------------------------------------------------------

#: What the scratch recorded. (d, x): a value stored or printed to d decimals; "full": a
#: full-precision value. Sources: afs = acq_summary_fixed_subsets.json (= acq_summary.json; the
#: fixed batch on its 91 runs), ass = acq_seeded_summary.json (the salted batch on 0-89),
#: pm = the scratch protocol.md section 5.9a table and text, ol = the v2 outline's T4 note.
SCRATCH = {
    ("fixed_scratch_subset", "mean_ALC", "random"): (4, 0.1595, "afs random_ALC"),
    ("fixed_scratch_subset", "placebo_minus_random", "D"): (5, 0.00228, "afs placebo.minus_random[0]"),
    ("fixed_scratch_subset", "placebo_minus_random", "run_se"): (5, 0.00077, "afs placebo.minus_random[1]"),
    ("fixed_scratch_subset", "placebo_minus_random", "identical_runs"): (0, 0, "afs placebo.identical_runs"),
    ("fixed_scratch_subset", "placebo_minus_random", "by_budget"):
        (4, [0.0, 0.0031, 0.004, 0.0027, 0.001, 0.001], "afs placebo.by_budget"),
    ("fixed_scratch_subset", "balanced", "minus_random", "D"): (5, -0.0001, "afs balanced.minus_random[0]"),
    ("fixed_scratch_subset", "balanced", "minus_random", "run_se"): (5, 0.00062, "afs balanced.minus_random[1]"),
    ("fixed_scratch_subset", "balanced", "minus_random", "identical_runs"): (0, 0, "afs"),
    ("fixed_scratch_subset", "balanced", "minus_random", "by_budget"):
        (4, [0.0, 0.0002, -0.0012, -0.0002, 0.0005, 0.0003], "afs balanced.by_budget"),
    ("fixed_scratch_subset", "distinct", "minus_random", "D"): (5, 2e-05, "afs distinct.minus_random[0]"),
    ("fixed_scratch_subset", "distinct", "minus_random", "run_se"): (5, 7e-05, "afs distinct.minus_random[1]"),
    ("fixed_scratch_subset", "distinct", "minus_random", "identical_runs"): (0, 50, "afs distinct.identical_runs"),
    ("fixed_scratch_subset", "distinct", "minus_random", "by_budget"):
        (4, [0.0, 0.0, -0.0, 0.0002, -0.0001, -0.0], "afs distinct.by_budget"),
    ("fixed_scratch_subset", "uncertain", "minus_random", "D"): (5, 0.00029, "afs uncertain.minus_random[0]"),
    ("fixed_scratch_subset", "uncertain", "minus_random", "run_se"): (5, 0.00077, "afs uncertain.minus_random[1]"),
    ("fixed_scratch_subset", "uncertain", "minus_random", "identical_runs"): (0, 0, "afs"),
    ("fixed_scratch_subset", "uncertain", "minus_random", "by_budget"):
        (4, [0.0, -0.0002, 0.0005, 0.0002, 0.0008, 0.0002], "afs uncertain.by_budget"),
    ("fixed_scratch_subset", "balanced", "minus_two_arm_mean", "D"): (4, -0.0012, "pm 5.9a table"),
    ("fixed_scratch_subset", "uncertain", "minus_two_arm_mean", "D"): (4, -0.0008, "pm 5.9a table"),
    ("salted", "mean_ALC", "random"): (4, 0.1623, "ass random_ALC"),
    ("salted", "placebo_minus_random", "D"): (5, -0.00089, "ass placebo.minus_random[0]"),
    ("salted", "placebo_minus_random", "run_se"): (5, 0.00105, "ass placebo.minus_random[1]"),
    ("salted", "placebo_minus_random", "by_budget"):
        (4, [0.0, -0.0027, -0.0029, 0.0007, 0.0003, 0.0004], "ass placebo.by_budget"),
    ("salted", "balanced", "minus_random", "D"): (5, -0.00195, "ass balanced.minus_random[0]"),
    ("salted", "balanced", "minus_random", "run_se"): (5, 0.00081, "ass balanced.minus_random[1]"),
    ("salted", "balanced", "minus_random", "by_budget"):
        (4, [0.0, -0.0059, -0.0031, 0.0002, -0.0006, -0.0006], "ass balanced.by_budget"),
    ("salted", "uncertain", "minus_random", "D"): (5, -0.00237, "ass uncertain.minus_random[0]"),
    ("salted", "uncertain", "minus_random", "run_se"): (5, 0.00108, "ass uncertain.minus_random[1]"),
    ("salted", "uncertain", "minus_random", "by_budget"):
        (4, [0.0, -0.0056, -0.0041, -0.0015, -0.0007, 0.0], "ass uncertain.by_budget"),
    ("salted", "balanced", "minus_placebo", "D"): (5, -0.00106, "ass balanced_minus_placebo[0]"),
    ("salted", "balanced", "minus_placebo", "run_se"): (5, 0.0009, "ass balanced_minus_placebo[1]"),
    ("salted", "uncertain", "minus_placebo", "D"): (5, -0.00148, "ass uncertain_minus_placebo[0]"),
    ("salted", "uncertain", "minus_placebo", "run_se"): (5, 0.00104, "ass uncertain_minus_placebo[1]"),
    ("salted", "balanced", "minus_two_arm_mean", "D"): (4, -0.0015, "pm 5.9a table"),
    ("salted", "balanced", "minus_two_arm_mean", "run_se"): (4, 0.0007, "pm 5.9a table"),
    ("salted", "uncertain", "minus_two_arm_mean", "D"): (4, -0.0019, "pm 5.9a table"),
    ("salted", "uncertain", "minus_two_arm_mean", "run_se"): (4, 0.0009, "pm 5.9a table"),
}

#: scratch statements that are not a single stored number, checked by hand here
SCRATCH_STATEMENTS = {
    "pm: 'The policies' edges sit at B1 and B3 (about -0.004 at B1, SE 0.002) and are gone by B15'":
        "salted batch, against the two-arm mean: see salted.<policy>.minus_two_arm_mean.by_budget[1] "
        "and by_budget_run_se[1]",
    "ol: 'policies -0.0008 to -0.0019 against the mean of two random arms, random against random up "
    "to 0.0023'": "the scratch's sets (fixed on its 91 runs, salted on 0-89, no salted distinct "
                  "arm): scratch_ranges below",
}

#: scratch values the port does not reproduce as stated, and why. Every scratch row and every
#: number the scratch summaries stored reproduces; this is a rounding in protocol.md's table.
EXPLAINED = {
    "fixed_scratch_subset/uncertain/minus_two_arm_mean/D":
        "protocol.md section 5.9a derived '-0.0008' from the summary's 5-dp values (uncertain - random "
        "+0.00029, placebo - random +0.00228: 0.00029 - 0.00228 / 2 = -0.00085) and wrote it to 4 dp; at "
        "full precision it is -0.000856, -0.0009 at 4 dp. The outline's T4 note ('policies -0.0008 to "
        "-0.0019 against the mean of two random arms') therefore reads -0.0009 to -0.0019 on the "
        "scratch's own sets (scratch_ranges).",
}


def _get(res, path):
    x = res
    for part in path:
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


def _same(got, want, how):
    if how == "full":
        return abs(got - want) <= 5e-5
    return abs(round(got, how) - want) < 1e-9


def compare_scratch(summary):
    items = []
    for path, (how, want, src) in SCRATCH.items():
        name = "/".join(str(p) for p in path)
        try:
            got = _get(summary, path)
        except (KeyError, TypeError, IndexError):
            got = None
        if got is None:
            items.append({"path": name, "status": "not run", "want": want, "source": src})
            continue
        if isinstance(want, list):
            got = [float(g) for g in got]
            ok = len(got) == len(want) and all(_same(g, w_, how) for g, w_ in zip(got, want))
            diff = max(abs(g - w_) for g, w_ in zip(got, want)) if len(got) == len(want) else None
        else:
            got = float(got)
            ok = _same(got, want, how)
            diff = got - want
        st = "match" if ok else ("explained" if name in EXPLAINED else "differs")
        it = {"path": name, "status": st, "got": got, "want": want, "how": how, "diff": diff, "source": src}
        if name in EXPLAINED:
            it["explanation"] = EXPLAINED[name]
        items.append(it)
    return items


def scratch_ranges(summary):
    """The outline's T4 note on the scratch's own sets."""
    f, s = summary.get("fixed_scratch_subset"), summary.get("salted")
    if not (f and s):
        return None
    two = {"fixed balanced": f["balanced"]["minus_two_arm_mean"]["D"],
           "fixed distinct": f["distinct"]["minus_two_arm_mean"]["D"],
           "fixed uncertain": f["uncertain"]["minus_two_arm_mean"]["D"],
           "salted balanced": s["balanced"]["minus_two_arm_mean"]["D"],
           "salted uncertain": s["uncertain"]["minus_two_arm_mean"]["D"]}
    plc = {"fixed": f["placebo_minus_random"]["D"], "salted": s["placebo_minus_random"]["D"]}
    b1 = {p: [s[p]["minus_two_arm_mean"]["by_budget"][1], s[p]["minus_two_arm_mean"]["by_budget_run_se"][1]]
          for p in ("balanced", "uncertain")}
    return {"policies_minus_two_arm_mean": [min(two.values()), max(two.values())], "each": two,
            "placebo_minus_random": plc, "largest_abs_placebo_minus_random": max(map(abs, plc.values())),
            "salted_b1_minus_two_arm_mean_with_run_se": b1,
            "outline_statement_holds": bool(round(min(two.values()), 4) == -0.0019 and
                                            round(max(two.values()), 4) == -0.0008 and
                                            round(max(map(abs, plc.values())), 4) == 0.0023)}


def compare_scratch_rows(rows, scratch_dir):
    """Every stored scratch row against this script's row: ALC and Brier by budget per
    policy. acq_rows/tl_<i>.json is the fixed batch, acq_seeded_rows/tl_<i>.json the salted."""
    if not scratch_dir:
        return None
    out = {}
    for batch, sub in (("fixed", "acq_rows"), ("salted", "acq_seeded_rows")):
        files = sorted(glob.glob(os.path.join(scratch_dir, sub, "tl_*.json")))
        worst, n, missing, per = 0.0, 0, [], {}
        for p in files:
            with open(p) as f:
                old = json.load(f)
            i = int(old["i"])
            if (batch, i) not in rows:
                missing.append(i)
                continue
            new = rows[(batch, i)]["policies"]
            for pol, v in old.items():
                if pol not in POLICY_NAMES:
                    continue
                dv = max([abs(new[pol]["ALC"] - v["ALC"])] +
                         [abs(a - b) for a, b in zip(new[pol]["by_budget"], v["by_budget"])])
                per[pol] = max(per.get(pol, 0.0), dv)
                worst = max(worst, dv)
                n += 1
        out[batch] = {"scratch_files": len(files), "compared_cells": n, "missing_runs": sorted(missing),
                      "max_abs_diff": worst, "max_abs_diff_by_policy": per,
                      "files_sha256": _dir_digest(files)}
    return out


def _dir_digest(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f"{os.path.basename(f)} {sha256(f)}\n".encode())
    return {"files": len(files), "sha256": h.hexdigest()}


# --- the results file -----------------------------------------------------------------------------

def embed_rows(rows):
    """Rows for the results file: pairs as (parent, cluster index), no anonymous id."""
    keys = sorted({(p["parent"], p["subject"]) for r in rows.values() for p in r["pairs"]})
    ix = {k: j for j, k in enumerate(keys)}
    out = []
    for (batch, i), r in sorted(rows.items(), key=lambda t: (t[0][0], t[0][1])):
        out.append({"batch": batch, "i": i, "seed": r["seed"],
                    "pairs": [[p["parent"], ix[(p["parent"], p["subject"])], p["n_items"]] for p in r["pairs"]],
                    "policies": {k: {"ALC": v["ALC"], "by_budget": v["by_budget"],
                                     "pair_brier": v["pair_brier"], "n_labels": v["n_labels"],
                                     "secs": v["secs"], "failures": v["failures"]}
                                 for k, v in r["policies"].items()}})
    return {"clusters": len(keys), "rows": out}


def rows_from_results(path):
    """The inverse of embed_rows (cluster index in place of the anonymous subject id)."""
    with open(path) as f:
        res = json.load(f)
    rows = {}
    for r in res["rows"]["rows"]:
        rows[(r["batch"], r["i"])] = {
            "batch": r["batch"], "i": r["i"], "seed": r["seed"],
            # zero-padded, so that clusters sort as (parent, subject) did and the bootstrap
            # draws the same resamples
            "pairs": [{"parent": p, "subject": f"cluster_{c:06d}", "n_items": n} for p, c, n in r["pairs"]],
            "policies": r["policies"]}
    return rows, res.get("provenance", {}).get("lib_digest")


def data_inputs():
    """The measurement-db tables the rows were scored from (paiec.data.load_pairs reads response,
    items and subjects per benchmark), with the pinned revision."""
    from paiec import data as D
    from paiec.fetch import REVISION
    out = {}
    for b in sorted(os.listdir(D.DATA_DIR)) if os.path.isdir(D.DATA_DIR) else []:
        for t in ("response", "items", "subjects"):
            p = os.path.join(D.DATA_DIR, b, f"{t}.parquet")
            if os.path.exists(p):
                out[rel(p)] = sha256(p)
    return {"data_revision": REVISION, "data_sha256": out or None}


def provenance(inputs, secs, rows_dir, from_results):
    me = os.path.abspath(__file__)
    code = {rel(p): sha256(os.path.join(ROOT, p)) for p in (rel(me), *LIB)}
    status = git("status", "--porcelain", "--", rel(me))
    return {"command": " ".join(["python", rel(sys.argv[0])] + sys.argv[1:]),
            "script": rel(me), "script_sha256": code[rel(me)], "code_sha256": code,
            "lib_digest": lib_digest(), "policy_digest": policy_digest(),
            "head": git("rev-parse", "HEAD"), "script_status": status or "clean",
            "when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "inputs": inputs,
            "rows": rel(rows_dir) if not from_results else f"embedded in {rel(from_results)} (--from-results)",
            "python": platform.python_version(), "numpy": np.__version__,
            "seeds": {"testlike draw": "level_calibration.draw('tl', i): default_rng([2, i])",
                      "fixed split": 0, "salted split": f"{SALT} + i", "bootstrap": BOOT_SEED},
            "wall_s": round(secs, 1)}


SCRATCH_SCRIPTS = {  # basename: sha256 of the exploratory script ported (session scratch, 2026-10-06)
    "acq_official.py": "ab12bbbfa1fe8d64d9ed309169a6ebaceadc2344b40f9168289de7c289890ce3",
    "acq_seeded.py": "ae5c4497716ed6892b9cb8eb5dca0f7d939e423962b9fdffb73d757ede5f7fc0",
    "acq_summary.py": "a078cf80b492128e97dc3024f1459f66e82b4e34e5acbb40f3688198c63e7738",
    "acq_seeded_summary.py": "2080c409e767695bf195253a66b8d2847f247c738e1518bf512a685ff7235e9d",
    "acq_summary_fixed_subsets.json": "b5f421438b63da875b31233f29dfd4e5acc4f1ee39de227e971d4061bf36c4f8",
    "acq_seeded_summary.json": "16cb31b3d3135b63250fecc454bf64c532e8ea1de0a67027266ed29d93c8df09",
    "protocol.md": "bbc91aca69f6be9fd36ad0da219cf6b2616f5afc8a1aeb6d64b4af7200390b21",
}


def show(summary):
    for name in ("fixed", "salted", "fixed_scratch_subset"):
        s = summary.get(name)
        if not s:
            print(f"{name}: rows missing")
            continue
        pr = s["placebo_minus_random"]
        print(f"{name} ({s['runs']} runs): random ALC {s['mean_ALC']['random']:.4f}; placebo - random "
              f"{pr['D']:+.5f} (run {pr['run_se']:.5f}, cluster {pr['cluster_se']:.5f})")
        for p in ("balanced", "distinct", "uncertain"):
            a, b = s[p]["minus_random"], s[p]["minus_two_arm_mean"]
            print(f"  {p:<9} - random {a['D']:+.5f} ({a['run_se']:.5f}, {a['cluster_se']:.5f})  "
                  f"- two-arm mean {b['D']:+.5f} ({b['run_se']:.5f}, {b['cluster_se']:.5f})  "
                  f"identical runs {a['identical_runs']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scratch", default=None,
                    help="session scratch directory holding acq_rows/ and acq_seeded_rows/")
    ap.add_argument("--summarise-only", action="store_true", help="score nothing; summarise stored rows")
    ap.add_argument("--from-results", nargs="?", const=OUT, default=None, metavar="PATH",
                    help="with --summarise-only: read the rows embedded in the results file PATH (default "
                         "%(const)s) instead of --rows. The rebuilt file has no scratch-row comparison, so "
                         "--out must name another file: it may not overwrite PATH or the default results file")
    ap.add_argument("--no-scratch-runs", action="store_true",
                    help="do not score fixed runs 91 and 93 (the scratch subset is then not summarised)")
    args = ap.parse_args(argv)
    if args.from_results:
        same = {os.path.abspath(args.from_results), os.path.abspath(OUT)}
        if os.path.abspath(args.out) in same:
            ap.error(f"--from-results rebuilds the summary from embedded rows without the scratch-row "
                     f"comparison and the rows' digest that {rel(args.from_results)} records; it would "
                     f"overwrite that record. Name another file with --out.")
    t0 = time.time()
    checks = Checks()
    if args.from_results:
        rows, lib = rows_from_results(args.from_results)
        checks.add("embedded rows were made with the current library", lib == lib_digest(), lib, lib_digest())
    else:
        if not args.summarise_only:
            collect(tasks_for(args.rows, not args.no_scratch_runs), args.jobs)
        rows = load_rows(args.rows)
        stale = sorted(f"{b} {i}" for (b, i), r in rows.items()
                       if r.get("lib") != lib_digest() or r.get("policy") != policy_digest())
        checks.add("every row was scored with the current library and policies", not stale, stale[:10], [])
    want = {("fixed", i) for i in RUNS} | {("salted", i) for i in RUNS}
    checks.add("every main run is stored (both batches, runs 0-89)", want <= set(rows),
               len(want & set(rows)), len(want))
    fails = sum(v["failures"] for r in rows.values() for v in r["policies"].values())
    checks.add("no hier failure (fallback prediction) in any scored run", fails == 0, fails, 0, required=False)
    modes = {(k, v.get("mode")) for r in rows.values() for k, v in r["policies"].items() if "mode" in v}
    if modes:
        ok = all(m == ("random" if k == "random" else "stream") for k, m in modes)
        checks.add("random is the platform's rule and every policy a streaming hook", ok, sorted(modes))
    summary = summarise(rows)
    items = compare_scratch(summary)
    n = {s: sum(1 for i in items if i["status"] == s) for s in ("match", "explained", "differs", "not run")}
    checks.add("every scratch summary value reproduces to its recorded precision or is explained",
               n["differs"] == 0, n, None)
    per_row = None if args.from_results else compare_scratch_rows(rows, args.scratch)
    if per_row:
        for b, v in per_row.items():
            checks.add(f"{b} batch: every scratch row reproduces bit for bit (ALC and Brier by budget)",
                       v["max_abs_diff"] <= 1e-12 and not v["missing_runs"],
                       {"max_abs_diff": v["max_abs_diff"], "missing": v["missing_runs"]}, 0.0)
    sr = scratch_ranges(summary)
    res = {"about": {
        "what": "acquisition policies under the official protocol (budget-major, one shared labeled, "
                "fresh instances per checkpoint) against the platform's random rule, with random against "
                "random (placebo) as the noise floor; hier-ship over leave-one-parent-out priors on "
                "test-like runs, paired",
        "plan": "docs/plans/p2_final_plan.md R5; the v2 outline's T4 row 7 and App F.8",
        "sign": "D = ALC(policy) - ALC(reference); negative means the policy is better",
        "regime": "tl (paiec.testlike defaults, seed 2), runs 0-89",
        "config": SHIP_NAME, "split_scope": "pair", "budgets": list(BUDGETS), "weights": list(WEIGHTS),
        "references": {"random": "the platform's rule", "placebo": "the rule on a salted key",
                       "two_arm_mean": "(random + placebo) / 2, per run and per pair"},
        "se": "run_se = sd over runs / sqrt(R) (the scratch's); cluster_se = bootstrap over (parent, "
              "subject) clusters, 2,000 resamples, seed 0",
        "tl_bar": "the harness gate's test-like bar, TL delta <= -0.002 (report section 4.4), for "
                  "reference; the gate also needs public runs and per-parent checks, not run here",
        "id_use": "replica pairs only; no formative feedback is read; pairs are written as (parent, "
                  "cluster index)"},
        "summary": summary, "scratch_ranges": sr}
    res["checks"] = checks.items
    res["scratch"] = {"ported_from": SCRATCH_SCRIPTS, "counts": n, "explained": EXPLAINED,
                      "statements": SCRATCH_STATEMENTS, "items": items, "rows": per_row}
    if args.from_results:
        with open(args.from_results) as f:
            src = json.load(f)
        res["scratch"]["rows_in_source"] = {
            "source": rel(args.from_results), "rows": src.get("scratch", {}).get("rows"),
            "checks": [c for c in src.get("checks", []) if "scratch row" in c["check"]],
            "note": "the scratch-row comparison the source file records, copied, not re-run: "
                    "--from-results reads no scratch rows"}
    res["rows"] = embed_rows(rows) if not args.from_results else _reembed(args.from_results)
    inputs = {"rows_digest": _dir_digest(sorted(glob.glob(os.path.join(args.rows, "*.json"))))
              if not args.from_results else None, **data_inputs()}
    res["provenance"] = provenance(inputs, time.time() - t0, args.rows, args.from_results)
    res["provenance"]["scoring"] = {
        "tasks": len(rows), "policy_runs": sum(len(r["policies"]) for r in rows.values()),
        "sum_of_policy_wall_s": round(sum(v["secs"] for r in rows.values() for v in r["policies"].values()), 1),
        "note": "per-policy wall time inside the worker processes, summed (wall_s above is this "
                "invocation's own time; with --summarise-only the rows were scored by an earlier one)"}
    show(summary)
    print(f"scratch comparison: {n}; rows: " + (json.dumps({b: v["max_abs_diff"] for b, v in per_row.items()})
                                                if per_row else "not compared"))
    if sr:
        print(f"scratch ranges: two-arm {sr['policies_minus_two_arm_mean']}, |placebo - random| <= "
              f"{sr['largest_abs_placebo_minus_random']:.5f}")
    for c in checks.items:
        if not c["ok"]:
            print(f"FAILED{' (required)' if c['required'] else ''}: {c['check']}: got {c['got']} want {c['want']}")
    for it in items:
        if it["status"] in ("differs", "explained"):
            print(f"  {it['status']}: {it['path']} got {it.get('got')} want {it['want']}")
    if checks.failed():
        print("not written: a required check failed")
        return 1
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    os.replace(tmp, args.out)
    print(f"wrote {rel(args.out)} ({res['provenance']['wall_s']} s)")
    return 0


def _reembed(path):
    with open(path) as f:
        return json.load(f)["rows"]


if __name__ == "__main__":
    sys.exit(main())
