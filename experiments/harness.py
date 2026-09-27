"""The acceptance harness for per-item covariates: one gate, on the shipped hier.

Every later item-side idea (ordinal difficulty fields, answer format, length,
LLM ratings, attempt signals, a fine-tuned encoder) is one number per item.
This script scores any such number the same way, against the model that ships,
and it holds the one table of thresholds they are all read against.

The model. A covariate is a dict item_key -> float; an item it does not hold
gets 0. It enters the shipped prediction as a centred logit offset

    q_i = sigmoid(logit p_i + cap(beta * (x_i - xbar))),    cap(o) = 4 tanh(o / 4)

p_i is paiec.hier.HierPredictor exactly as it ships: submission/model.py's
LEVEL (read from the file) over prior.build, fitted leaving the target's parent
benchmark out (experiments/level_calibration.py's bundle and factory, the base
of "Calibrating for the hidden test" and of "Item signal from the pair's own
labels"). xbar is the covariate's mean over labeled items, so hier keeps the
benchmark's level and only the ordering of items comes from x. The offset is
added to hier's output, not fitted inside its model: hier's posterior is not
refitted with the covariate in it, which is what makes one replay of hier serve
every covariate. At B0 nothing is labeled and q = p. Where the offset is 0, q is
p itself, bit for bit, so a covariate of zeros reproduces the shipped
predictions (tests/test_harness.py).

Slopes (VARIANTS):
  transferred  one coefficient per budget, beta_B, fitted on the other parents'
               appearances (test-like and mix/whole, FIT_ON): the minimum of the
               mean pair Brier, a 1-D search per budget. The centre is every
               label on the target's benchmark, its own pair's and other
               subjects'.
  per-pair     beta_p from the target pair's own labels at the checkpoint: the
               MAP of y_j ~ Bernoulli(sigmoid(u_j + beta c_j)) with beta ~ N(0,
               (s / sd_x)^2) over its own labeled items, c_j = x_j - their mean,
               and the target centred on the same mean. s is a prior sd per
               standard deviation of x (S_GRID); sd_x is taken over the training
               parents' evaluated items (the plan caps s at 0.5 for a feature
               whose sign is not known; the grid goes wider so that the
               oracle's line shows what a per-pair slope can reach, and the
               nested choice is reported). u_j is hier's prediction for item j as
               if j itself were unlabeled, at the same checkpoint and with the
               same labels (below), so a label is never scored against a
               prediction it already moved. One own label carries no slope
               after centring, so the term is 0 at B1.
  hybrid       the per-pair MAP with its prior centred on the transferred beta_B
               instead of 0 (the plan's "transferred slope plus a zero-mean
               per-pair component").
  Each is on from B1, or only from B7 (STARTS).

u_j. hier reads an item through its digest (its own residual e_i), its
item_features groups and the multiple-choice floor of its text. The collector
calls hier on each labeled item a second time with an invisible separator
(U+2063) appended to item_content: a new digest, the same groups (features are
untouched) and the same floor (checked). That is hier's predictive for an
unlabeled item of the same groups, with the pair's level fitted on every label,
the item's own included (the part the joint model would also leave in).

Selection: nested leave-one-parent-out over the four multi-subject parents.
For each held-out parent q, every configuration of a variant (and 'off') is
scored on each inner parent q2 != q with whatever it fits (the transferred
slope) fitted on the two remaining parents, and the configuration with the
lowest mean over the three inner parents of their test-like ALC difference is
chosen, or 'off' when no mean is below 0. It is then fitted on the three and
scored on q, in every regime. Public benchmarks outside the four (swe_rebench)
get the choice made the same way over all four parents. Forced lines score a
fixed configuration with no selection: the r -> ALC curve without selection
noise.

Regimes (PLAN): test-like runs (testlike.Regime() at its defaults, seed 2;
level_calibration's 'tl'), mix/whole (groups merged at random and wholes, no
strata, seed 3; 'tl mix/whole'), and public R1 (official.sample_run, seed 0,
split scope 'pair', benchmark-first 'r1b' and pair-uniform 'r1p') as the guard.
These are the runs of level_calibration.py and itemsig_eval.py; the test-like
and mix/whole ones are also the scratch rows of the heads study (--stage verify).

Statistics. Paired ALC differences against the shipped hier, weighted 1 / run
size per pair appearance (a run's ALC is the mean over its pairs), with: the SE
over runs; a pair-cluster bootstrap (BOOTS resamples, a ratio estimator;
cluster (parent, subject) on test-like runs, testlike.cluster_key, and
(benchmark, subject) on public ones); the same bootstrap stratified by parent;
and the SE across the four parents' means (the only one that sees variation
between benchmarks, on four of them). The bootstraps hold the selection fixed.
Per budget, B0 is reported and is 0 by construction.

The item oracle and the threshold table (--stage table). The acceptance oracle
is the parent's in-sample Rasch difficulty (testlike.Catalogue.difficulty: every
subject of the parent, the evaluated responses included) with the transferred
slope cross-fitted leave-one-parent-out, forced on from B1: what the heads study
scored (-0.0437 test-like, -0.0557 mix/whole). The harness must reproduce those
numbers (ACCEPT) before anything else is read. The honest oracle fits the
difficulty on the other subject folds only (HONEST_FOLDS, folds by subject_id),
so a target's own responses never enter its covariate; the gap between the two
is the in-sample oracle's leak. The degraded oracles are x = r z + sqrt(1 -
r^2) e, z a difficulty standardised within its parent, e standard normal per
item (N_REPS draws, per-appearance differences averaged over them), for r in
R_GRID, and r = 0 as the uninformative covariate. The gate table ('honest')
degrades the honest difficulty, fold by fold, so r means what a later
covariate's r means when it is measured against honest difficulty, as the plan
prescribes; the 'in_sample' table degrades the in-sample one and is the heads
study's curve. r is the correlation over a parent's items; within a
pseudo-benchmark, whose difficulty range is narrower, the realised correlation
is lower, and the table reports it (r_within_pair_tl).

Gate (GATE, the plan's acceptance rule): nested selection switches the term on
(in at least 3 of the 4 outer folds; the count is reported), test-like ALC
difference <= -0.002 with the mix/whole one of the same sign and no held-out
parent above +0.002, and neither public weighting worse than +0.001. A forced
line has no selection: it reports whether it would pass if switched on. The
plan's further rule, that an uninformative version costs at most +0.001, is the
table's r = 0 row. A covariate that exists on one parent only cannot be switched
on leave-one-parent-out (its slope has nothing to be fitted or selected on); its
forced per-pair lines show what it does there. One noise draw of a degraded
oracle moves its test-like difference by about as much as its cluster SE
(draw_sd; 0.0005 at r = 0.3), so a real covariate, which is one draw, reads the
table with that spread as well as its own SEs.

Stages
  collect  replay hier on every run of PLAN and store rows (one pickle per run,
           resumable) in data/harness_rows/<regime>/: per pair appearance its
           evaluated items (key, successes, responses) with hier's prediction at
           every checkpoint, and its acquired items (key, label) with hier's
           prediction as labeled and as unlabeled
  verify   the stored predictions against the heads study's scratch rows
           (--scratch DIR or HARNESS_SCRATCH_ROWS; per response, float32 as
           stored there), and the unlabeled-item probe against hier on items
           nobody labeled
  table    the oracle acceptance and the threshold table ->
           results/harness_thresholds.json
  eval     one covariate (--cov: JSON {item_key: x} or a CSV/parquet with
           columns item_key, x) -> results/harness_<name>.json, with the gate
           and the table's rows beside it
  show     the result of table (or of eval, --cov RESULT.json) as markdown

Run:  python experiments/harness.py --stage collect --jobs 2     # 25 min: 650 runs, ~3.6 s each
      python experiments/harness.py --stage verify --scratch DIR  # 40 s
      python experiments/harness.py --stage table                 # 24 min: 62 covariates, ~23 s each
      python experiments/harness.py --stage show                  # markdown tables
      python experiments/harness.py --stage eval --cov my_feature.json --name my_feature   # ~20 s
(wall times on a shared M1, beside a running language-model job; collect keeps
three processes and about 1 GB, and resumes where it stopped.) The rows are
43 MB under data/ (gitignored); results/harness_thresholds.json holds the
acceptance, both tables, the thresholds, the verification and the provenance.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUDGETS = (0, 1, 3, 7, 15, 31)
W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
#: harness regime -> experiments/level_calibration.py's REGIMES key
REGIMES = {"tl": "tl", "mix": "tl mix/whole", "r1b": "r1b", "r1p": "r1p"}
#: runs 0..n-1 of each regime
PLAN = {"tl": 300, "mix": 150, "r1b": 100, "r1p": 100}
FIT_ON = ("tl", "mix")          # appearances the transferred slope is fitted on
SELECT_ON = "tl"                # the regime the nested selection reads
GUARDS = ("r1b", "r1p")
CAP = 4.0                       # |offset| < CAP logits (the heads study's 4 tanh(o / 4))
S_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)   # per-pair prior sd, per standard deviation of x
STARTS = (1, 7)                 # the term acts from this budget on
VARIANTS = ("transferred", "per-pair", "hybrid")
R_GRID = (0.1, 0.2, 0.3, 0.5, 0.7)
N_REPS = 5
HONEST_FOLDS = 5
HONEST_SALT = "harness-honest"
BOOTS = 2000
MARK = "⁣"                 # appended to item_content for the unlabeled-item probe
ACCEPT = {"tl": -0.0437, "mix": -0.0557, "tol": 0.003}
GATE = {"tl": -0.002, "parent": 0.002, "guard": 0.001, "folds_on": 3}
#: forced (unselected) configurations reported beside the nested lines
FORCED = (("transferred", None, 1), ("transferred", None, 7), ("per-pair", 0.5, 1),
          ("per-pair", 0.5, 7), ("hybrid", 0.5, 1))

ROWS = os.path.join(ROOT, "data", "harness_rows")
RECOMPUTED = ("hier is replayed on every run (collect): the heads study's scratch rows lack item keys, "
              "the unlabeled-item predictions the per-pair slope needs and public runs; its "
              "test-like and mix/whole rows are used only to check the replay (verify)")
OUT = os.path.join(ROOT, "results", "harness_thresholds.json")
#: the heads study's rows (rethink2/fine-tuning/rows of the session that wrote
#: this script), for --stage verify only; nothing else reads them
SCRATCH = os.environ.get("HARNESS_SCRATCH_ROWS", "")
SCRATCH_DIRS = {"tl": "tl", "mix": "tl_mix-whole"}


# --- core arithmetic (pure numpy; tests/test_harness.py) ------------------------------

def sigmoid(z):
    z = np.asarray(z, float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def logit(p):
    p = np.asarray(p, float)
    return np.log(p) - np.log1p(-p)


def capped(o):
    return CAP * np.tanh(np.asarray(o, float) / CAP)


def shifted(p, off):
    """hier's p moved by a logit offset; exactly p where the offset is 0."""
    p, off = np.asarray(p, float), np.asarray(off, float)
    return np.where(off == 0.0, p, sigmoid(logit(p) + off))


def delta_brier(p, q, K, N, seg, n_seg):
    """Per segment (pair appearance): Brier of q minus Brier of p over every
    response, items given as K successes in N responses."""
    num = N * (q * q - p * p) - 2 * K * (q - p)
    return np.bincount(seg, num, n_seg) / np.maximum(np.bincount(seg, N, n_seg), 1e-300)


def pair_slope(u, y, c, mask, m, s, iters=50):
    """MAP of beta in y ~ Bernoulli(sigmoid(u + beta c)), beta ~ N(m, s^2), for
    each row (episode) of the padded (E, L) arrays; mask marks real labels. The
    log posterior is concave, so Newton converges; steps are capped at 3 prior sd
    as a guard."""
    u, y, c = (np.asarray(a, float) for a in (u, y, c))
    mask = np.asarray(mask, bool)
    m = np.broadcast_to(np.asarray(m, float), u.shape[:1]).copy()
    s = np.broadcast_to(np.asarray(s, float), u.shape[:1])
    c = np.where(mask, c, 0.0)
    u = np.where(mask, u, 0.0)
    b = m.copy()
    for _ in range(iters):
        pr = sigmoid(u + b[:, None] * c)
        g = np.sum(np.where(mask, c * (y - pr), 0.0), 1) - (b - m) / s ** 2
        h = np.sum(np.where(mask, c * c * pr * (1 - pr), 0.0), 1) + 1 / s ** 2
        step = np.clip(g / h, -3 * s, 3 * s)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def golden(f, lo, hi, iters=60):
    """Minimum of f on [lo, hi] by golden-section search."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_scalar(f, scale, span=6.0, n_grid=41):
    """argmin of a 1-D objective over [-span, span] / scale: a grid, then golden
    section inside the best grid cell's neighbours. 0 is kept when nothing beats
    it (a flat objective leaves the covariate off)."""
    grid = np.linspace(-span, span, n_grid) / scale
    vals = np.array([f(b) for b in grid])
    j = int(np.argmin(vals))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, n_grid - 1)]
    b = golden(f, lo, hi)
    return b if f(b) < f(0.0) else 0.0


def weighted_mean(d, w):
    return float(np.sum(w * d) / np.sum(w))


def boot_se(d, w, cluster, strata=None, boots=BOOTS, seed=0):
    """Cluster bootstrap SE of the ratio estimator sum(w d) / sum(w): clusters
    resampled with replacement, as one pool or within each stratum."""
    rng = np.random.default_rng(seed)
    cl, inv = np.unique(cluster, return_inverse=True)
    S = np.bincount(inv, w * d, len(cl))
    Wc = np.bincount(inv, w, len(cl))
    if strata is None:
        groups = [np.arange(len(cl))]
    else:
        st = np.empty(len(cl), object)
        st[inv] = np.asarray(strata, object)
        groups = [np.flatnonzero(st == v) for v in sorted(set(st.tolist()))]
    num = np.zeros(boots)
    den = np.zeros(boots)
    for g in groups:
        pick = g[rng.integers(0, len(g), (boots, len(g)))]
        num += S[pick].sum(1)
        den += Wc[pick].sum(1)
    return float(np.std(num / den, ddof=1)) if boots > 1 else float("nan")


def run_se(d, w, run):
    runs, inv = np.unique(run, return_inverse=True)
    m = np.bincount(inv, w * d, len(runs)) / np.bincount(inv, w, len(runs))
    return float(np.std(m, ddof=1) / math.sqrt(len(m))) if len(m) > 1 else float("nan")


def stats(d, w, run, cluster, parent, boots=BOOTS, seed=0):
    """Mean and the three SEs of one set of appearances."""
    if not len(d):
        return None
    return {"est": weighted_mean(d, w), "run_se": run_se(d, w, run),
            "cluster_se": boot_se(d, w, cluster, None, boots, seed),
            "strat_se": boot_se(d, w, cluster, parent, boots, seed),
            "n_app": int(len(d)), "n_runs": int(len(np.unique(run))),
            "n_clusters": int(len(np.unique(cluster)))}


def select(crit):
    """The configuration with the lowest inner criterion, or None ('off') unless
    it is below 0. Ties break on the configuration order."""
    best = None
    for c, v in crit.items():
        if v < 0 and (best is None or v < crit[best]):
            best = c
    return best


def degrade(z, r, eps):
    """A covariate at correlation r with z (both standardised): r z + sqrt(1-r^2) e."""
    return r * np.asarray(z, float) + math.sqrt(max(0.0, 1 - r * r)) * np.asarray(eps, float)


def standardise_within(values, groups):
    """values standardised to mean 0, sd 1 within each group."""
    v = np.asarray(values, float).copy()
    g = np.asarray(groups, object)
    for k in set(g.tolist()):
        m = g == k
        sd = v[m].std()
        v[m] = (v[m] - v[m].mean()) / (sd if sd > 0 else 1.0)
    return v


# --- the shipped configuration ------------------------------------------------------

def shipped_level():
    """submission/model.py's LEVEL, read without importing the module (it
    imports the archive's paiec_rt, which exists only inside the zip)."""
    with open(os.path.join(ROOT, "submission", "model.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("no LEVEL in submission/model.py")


def _lc():
    import experiments.level_calibration as LC
    return LC


def ship_name():
    LC = _lc()
    lv = shipped_level()
    name = LC.gname(lv["mu0"], lv["sigma_mu"], lv["attr_scale"])
    if LC.parse(name)[1] != {k: float(v) for k, v in lv.items()}:
        raise RuntimeError(f"{name} is not LEVEL {lv}")
    return name


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


LIB = ("paiec/hier.py", "paiec/prior.py", "paiec/subjects.py", "paiec/official.py",
       "paiec/testlike.py", "paiec/mcq.py", "paiec/predict.py", "paiec/fitting.py",
       "experiments/level_calibration.py", "submission/model.py")


def provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    return {"commit": commit, "lib_digest": digest(LIB), "script_digest": digest(["experiments/harness.py"]),
            "level": shipped_level()}


# --- collect ------------------------------------------------------------------------

def unlabeled_twin(inp):
    """The same subject and item with a new digest and nothing else changed."""
    subject, item = inp
    return [subject, dict(item, item_content=str(item.get("item_content", "")) + MARK)]


def row_path(rows, regime, i):
    return os.path.join(rows, regime, f"{i}.pkl")


def collect_run(regime, i, ship):
    """Replay hier on one run: per pair appearance its evaluated items with
    hier's prediction at every checkpoint, and its acquired items with hier's
    prediction as labeled (acq_p) and as if unlabeled (acq_pu)."""
    from paiec import official as O
    from paiec import testlike as T
    from paiec.data import official_subject
    from paiec.mcq import floor_of
    LC = _lc()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = LC.draw(REGIMES[regime], i)
    slots, cps = LC.checkpoints(run)
    made = []
    fac = LC.factory([ship], run, made)
    out, first, at = [], [], 0
    for entry, s in zip(run, slots):
        pair, items = O._entry(entry)
        by = pair.by_item()
        _, ev = O.split(pair, 0, "pair")
        ev = [k for k in ev if items is None or k in items]
        labels = [int(r.label) for k in ev for r in by[k]]
        if labels != [t[1] for t in s.targets]:
            raise RuntimeError(f"{regime} run {i}: evaluation order differs from official._slots")
        N = np.array([len(by[k]) for k in ev], np.int32)
        K = np.array([sum(int(r.label) for r in by[k]) for k in ev], np.int32)
        pos = at + np.concatenate([[0], np.cumsum(N)[:-1]]).astype(np.int64)
        first.append(pos)
        at += len(labels)
        acq = s.acquired[:31]
        for e in acq:
            if floor_of(e[0][1]["item_content"]) != floor_of(unlabeled_twin(e[0])[1]["item_content"]):
                raise RuntimeError("the unlabeled-item probe changed an item's floor")
        out.append(dict(
            name=s.name, parent=T.parent_of(s.name), kind=T.kind_of(s.name), bench=s.benchmark_id,
            sid=str(pair.subject_id), anon_sid=s.subject_id, subject=official_subject(pair.subject),
            ev_keys=np.array(ev, dtype=object), ev_K=K, ev_N=N,
            ev_p=np.full((6, len(ev)), np.nan),
            acq_keys=np.array(s.keys[:31], dtype=object),
            acq_y=np.array([e[1] for e in acq], np.int8),
            acq_p=np.full((6, len(acq)), np.nan), acq_pu=np.full((6, len(acq)), np.nan)))
    calls = 0
    for bi, (b, labeled, inputs, index) in enumerate(cps):
        fn = fac()
        P = np.array([fn(inp, labeled) for inp in inputs], float)
        pe = P[index]
        calls += len(inputs)
        for d, s, pos in zip(out, slots, first):
            d["ev_p"][bi] = pe[pos]
            for j, e in enumerate(s.acquired[:min(b, 31)]):
                d["acq_p"][bi, j] = fn(e[0], labeled)
        for d, s in zip(out, slots):
            for j, e in enumerate(s.acquired[:min(b, 31)]):
                d["acq_pu"][bi, j] = fn(unlabeled_twin(e[0]), labeled)
                calls += 2
    return dict(regime=regime, run=i, slots=out, ship=ship, level=shipped_level(),
                secs=time.perf_counter() - t0, calls=calls,
                hier_failures=int(sum(getattr(m, "failures", 0) for m in made)),
                hier_unconverged=int(sum(getattr(m, "unconverged", 0) for m in made)))


def _collect_task(task):
    regime, i, ship, rows = task
    path = row_path(rows, regime, i)
    r = collect_run(regime, i, ship)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(r, f, protocol=4)
    os.replace(path + ".tmp", path)
    return regime, i, r["secs"], r["hier_failures"]


def stage_collect(args):
    LC = _lc()
    ship = ship_name()
    todo = []
    for regime in args.regimes:
        os.makedirs(os.path.join(args.rows, regime), exist_ok=True)
        todo += [(regime, i, ship, args.rows) for i in range(PLAN[regime] if args.n is None else args.n)
                 if not os.path.exists(row_path(args.rows, regime, i))]
    print(f"collect: {len(todo)} runs to replay with {ship}", flush=True)
    if not todo:
        return
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = LC.pairs()
        LC.catalogue()
        for regime in args.regimes:
            if LC.REGIMES[REGIMES[regime]][0] == "testlike":
                LC.sampler(REGIMES[regime])
        for par in sorted({p.benchmark_id for p in pairs}):
            LC.bundle(par, 0.0)        # built once, shared by the forked workers
    print(f"collect: data, catalogue and leave-one-parent-out priors in {time.time() - t0:.0f}s",
          flush=True)
    done = 0
    if args.jobs <= 1:
        it = map(_collect_task, todo)
        for regime, i, secs, fails in it:
            done += 1
            print(f"{regime} run {i}: {secs:.1f}s, {fails} hier failures ({done}/{len(todo)})", flush=True)
    else:
        with mp.get_context("fork").Pool(args.jobs, maxtasksperchild=50) as pool:
            for regime, i, secs, fails in pool.imap_unordered(_collect_task, todo, chunksize=1):
                done += 1
                print(f"{regime} run {i}: {secs:.1f}s, {fails} hier failures ({done}/{len(todo)})",
                      flush=True)
    print(f"collect: {len(todo)} runs in {time.time() - t0:.0f}s", flush=True)


# --- verify -------------------------------------------------------------------------

def stage_verify(args):
    """Stored rows against the heads study's scratch rows (same runs, same model),
    and the unlabeled-item probe against hier on items nobody labeled."""
    res = {"scratch": args.scratch}
    for regime, sub in SCRATCH_DIRS.items():
        d = os.path.join(args.scratch, sub) if args.scratch else ""
        if not d or not os.path.isdir(d):
            res[regime] = "no scratch rows"
            continue
        n, worst_e, worst_a, resp, labs = 0, 0.0, 0.0, 0, 0
        for i in range(PLAN[regime]):
            mine, theirs = row_path(args.rows, regime, i), os.path.join(d, f"{i}.pkl")
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                continue
            with open(mine, "rb") as f:
                a = pickle.load(f)
            with open(theirs, "rb") as f:
                b = pickle.load(f)
            if len(a["slots"]) != len(b["slots"]):
                raise RuntimeError(f"{regime} run {i}: pair counts differ")
            for x, y in zip(a["slots"], b["slots"]):
                if x["bench"] != y["benchmark_id"] or x["anon_sid"] != y["subject_id"]:
                    raise RuntimeError(f"{regime} run {i}: pairs differ")
                pe = np.repeat(x["ev_p"], x["ev_N"], axis=1).astype(np.float32)
                yy = np.repeat(np.zeros(len(x["ev_N"])), x["ev_N"])
                if pe.shape != y["eval_p"].shape or len(yy) != len(y["eval_y"]):
                    raise RuntimeError(f"{regime} run {i}: evaluation shapes differ")
                worst_e = max(worst_e, float(np.max(np.abs(pe - y["eval_p"]))))
                ap = x["acq_p"].astype(np.float32)
                both = np.isfinite(ap) & np.isfinite(y["acq_p"])
                if (np.isfinite(ap) != np.isfinite(y["acq_p"])).any():
                    raise RuntimeError(f"{regime} run {i}: labeled-item cells differ")
                if both.any():
                    worst_a = max(worst_a, float(np.max(np.abs(ap[both] - y["acq_p"][both]))))
                if not np.array_equal(x["acq_y"], y["acq_y"]):
                    raise RuntimeError(f"{regime} run {i}: acquired labels differ")
                resp += pe.size
                labs += int(both.sum())
            n += 1
        res[regime] = {"runs": n, "responses_x_budgets": resp, "labeled_cells": labs,
                       "max_abs_eval": worst_e, "max_abs_labeled": worst_a}
        print(regime, res[regime], flush=True)
    res["probe"] = probe_check(args)
    print("probe", res["probe"], flush=True)
    state = load_json(args.out) or {}
    state["verify"] = res
    save_json(args.out, state)


def probe_check(args, n_runs=3):
    """hier on an evaluated item that no label on its benchmark touches, called
    as is and through the unlabeled-item probe: the two must agree exactly."""
    LC = _lc()
    ship = ship_name()
    worst, n = 0.0, 0
    for regime in ("tl", "r1b"):
        for i in range(n_runs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run = LC.draw(REGIMES[regime], i)
            slots, cps = LC.checkpoints(run)
            fac = LC.factory([ship], run, [])
            for b, labeled, inputs, _ in cps[2:4]:
                fn = fac()
                seen = {(e[0][1]["benchmark_id"], e[0][1]["item_content"], e[0][1]["item_features"])
                        for e in labeled}
                for inp in inputs[::7]:
                    it = inp[1]
                    if (it["benchmark_id"], it["item_content"], it["item_features"]) in seen:
                        continue
                    worst = max(worst, abs(fn(inp, labeled) - fn(unlabeled_twin(inp), labeled)))
                    n += 1
    return {"items": n, "max_abs": worst}


# --- rows as flat arrays --------------------------------------------------------------

class Rows:
    """One regime's stored runs as flat arrays. Appearances a = 0..A-1; evaluated
    items flattened (ev_*) with their appearance ev_a; acquired items padded to
    31 per appearance (own_*); siblings = appearances on the same benchmark of
    the same run (grp)."""

    def __init__(self, regime, runs, key_index):
        self.regime = regime
        app, ev = [], []
        for r in runs:
            size = len(r["slots"])
            for s in r["slots"]:
                app.append((r["run"], size, s["parent"], s["kind"], s["bench"], s["sid"]))
                ev.append(s)
        A = len(app)
        self.A = A
        self.run = np.array([a[0] for a in app])
        self.w = 1.0 / np.array([a[1] for a in app], float)
        self.parent = np.array([a[2] for a in app], object)
        self.kind = np.array([a[3] for a in app], object)
        self.bench = np.array([a[4] for a in app], object)
        self.sid = np.array([a[5] for a in app], object)
        public = regime in GUARDS
        self.cluster = np.array([f"{(a[4] if public else a[2])}|{a[5]}" for a in app], object)
        self.strata = self.parent
        g = {}
        self.grp = np.array([g.setdefault((a[0], a[4]), len(g)) for a in app])
        self.n_grp = len(g)
        # evaluated items
        n_ev = [len(s["ev_keys"]) for s in ev]
        self.ev_a = np.repeat(np.arange(A), n_ev)
        self.ev_k = np.array([key_index[k] for s in ev for k in s["ev_keys"]], np.int64)
        self.ev_K = np.concatenate([s["ev_K"] for s in ev]).astype(float)
        self.ev_N = np.concatenate([s["ev_N"] for s in ev]).astype(float)
        self.ev_p = np.concatenate([s["ev_p"] for s in ev], 1)            # (6, n)
        # acquired items (own labels)
        L = 31
        self.own_len = np.array([len(s["acq_y"]) for s in ev])
        self.own_k = np.full((A, L), -1, np.int64)
        self.own_y = np.zeros((A, L))
        self.own_pu = np.full((6, A, L), 0.5)
        for a, s in enumerate(ev):
            n = len(s["acq_y"])
            self.own_k[a, :n] = [key_index[k] for k in s["acq_keys"]]
            self.own_y[a, :n] = s["acq_y"]
            self.own_pu[:, a, :n] = np.where(np.isfinite(s["acq_pu"]), s["acq_pu"], 0.5)
        self.nlab = np.minimum(np.array(BUDGETS)[:, None], self.own_len[None, :])   # (6, A)
        self.hier_failures = int(sum(r.get("hier_failures", 0) for r in runs))
        self.subject = [s["subject"] for s in ev]


def load_rows(rows_dir, regimes, plan=PLAN):
    """{regime: Rows} over one shared item-key index, and the index."""
    raw, keys = {}, {}
    for regime in regimes:
        runs = []
        for i in range(plan[regime]):
            p = row_path(rows_dir, regime, i)
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                runs.append(pickle.load(f))
        if not runs:
            raise RuntimeError(f"no rows for {regime} under {rows_dir}; run --stage collect")
        raw[regime] = runs
        for r in runs:
            for s in r["slots"]:
                for k in s["ev_keys"]:
                    keys.setdefault(k, len(keys))
                for k in s["acq_keys"]:
                    keys.setdefault(k, len(keys))
    rows = {reg: Rows(reg, runs, keys) for reg, runs in raw.items()}
    return rows, list(keys)


# --- covariates -----------------------------------------------------------------------

class Covariate:
    """x per item key, possibly per subject fold: X (F, K) over the harness's key
    index and a fold per appearance (0 for a plain dict). Missing -> 0."""

    def __init__(self, X, fold_of=None, name=""):
        self.X = np.atleast_2d(np.asarray(X, float))
        self.fold_of = fold_of          # sid -> fold, or None
        self.name = name

    @classmethod
    def from_dict(cls, d, keys, name=""):
        return cls(np.array([[float(d.get(k, 0.0)) for k in keys]]), None, name)

    @classmethod
    def from_folds(cls, maps, fold_of, keys, name=""):
        return cls(np.array([[float(m.get(k, 0.0)) for k in keys] for m in maps]), fold_of, name)

    def folds(self, rows):
        if self.fold_of is None:
            return np.zeros(rows.A, np.int64)
        return np.array([self.fold_of(s) for s in rows.sid], np.int64)


class Prep:
    """One regime's quantities that depend on the covariate and not on the slope."""

    def __init__(self, rows, cov):
        R = rows
        f = cov.folds(R)
        X = cov.X
        self.x_ev = X[f[R.ev_a], R.ev_k]
        kk = np.where(R.own_k >= 0, R.own_k, 0)
        x_own = np.where(R.own_k >= 0, X[f[:, None], kk], 0.0)          # (A, 31)
        self.x_own = x_own
        cum = np.concatenate([np.zeros((R.A, 1)), np.cumsum(x_own, 1)], 1)
        nl = R.nlab                                                      # (6, A)
        s_own = cum[np.arange(R.A)[None, :], nl]                         # (6, A)
        self.xbar_own = np.where(nl > 0, s_own / np.maximum(nl, 1), 0.0)
        # benchmark centre: every sibling's own labels, x read in the target's fold
        s_b = np.zeros((6, R.A))
        n_b = np.zeros((6, R.A))
        for ff in np.unique(f):
            xo = np.where(R.own_k >= 0, X[ff][kk], 0.0)
            cf = np.concatenate([np.zeros((R.A, 1)), np.cumsum(xo, 1)], 1)
            sf = cf[np.arange(R.A)[None, :], nl]
            m = f == ff
            for bi in range(6):
                gs = np.bincount(R.grp, sf[bi], R.n_grp)
                gn = np.bincount(R.grp, nl[bi], R.n_grp)
                s_b[bi, m] = gs[R.grp[m]]
                n_b[bi, m] = gn[R.grp[m]]
        self.xbar_bench = np.where(n_b > 0, s_b / np.maximum(n_b, 1), 0.0)
        self.n_bench = n_b
        self.c_bench = self.x_ev[None, :] - self.xbar_bench[:, R.ev_a]   # (6, n_ev)
        self.c_bench[n_b[:, R.ev_a] == 0] = 0.0
        self.c_own = self.x_ev[None, :] - self.xbar_own[:, R.ev_a]
        self.c_own[nl[:, R.ev_a] == 0] = 0.0
        self.z_ev = logit(R.ev_p)                                        # (6, n_ev)
        # per-appearance Brier weights for the transferred fit
        ntot = np.bincount(R.ev_a, R.ev_N, R.A)
        self.wN = R.ev_N / ntot[R.ev_a]
        self.wK = R.ev_K / ntot[R.ev_a]


class Engine:
    """Scores slope configurations for one covariate on every regime."""

    def __init__(self, rows, cov):
        self.rows = rows
        self.cov = cov
        self.prep = {reg: Prep(R, cov) for reg, R in rows.items()}
        self._beta = {}
        self._sd = {}
        self._pp = {}
        self._delta = {}

    # the transferred slope
    def beta(self, train):
        """beta_B (6,) fitted on FIT_ON appearances whose parent is in `train`."""
        key = tuple(sorted(train))
        if key in self._beta:
            return self._beta[key]
        parts = []
        for reg in FIT_ON:
            if reg not in self.rows:
                continue
            R, P = self.rows[reg], self.prep[reg]
            m = np.isin(R.parent[R.ev_a], list(train))
            parts.append((P.z_ev[:, m], P.c_bench[:, m], P.wN[m], P.wK[m]))
        z = np.concatenate([p[0] for p in parts], 1)
        c = np.concatenate([p[1] for p in parts], 1)
        wN = np.concatenate([p[2] for p in parts])
        wK = np.concatenate([p[3] for p in parts])
        scale = self.sd(train)
        out = np.zeros(6)
        for bi in range(1, 6):
            zb, cb = z[bi], c[bi]
            act = cb != 0
            zb, cb, nb, kb = zb[act], cb[act], wN[act], wK[act]
            p0 = sigmoid(zb)
            base = float(np.sum(nb * p0 * p0 - 2 * kb * p0))

            def f(b, zb=zb, cb=cb, nb=nb, kb=kb, base=base):
                q = sigmoid(zb + capped(b * cb))
                return float(np.sum(nb * q * q - 2 * kb * q)) - base

            out[bi] = fit_scalar(f, scale) if len(zb) else 0.0
        self._beta[key] = out
        return out

    def sd(self, train):
        """sd of x over the training parents' evaluated items in the FIT_ON
        regimes (one value per item and appearance); 1 when x is constant there."""
        key = tuple(sorted(train))
        if key not in self._sd:
            xs = []
            for reg in FIT_ON:
                if reg in self.rows:
                    R = self.rows[reg]
                    xs.append(self.prep[reg].x_ev[np.isin(R.parent[R.ev_a], list(train))])
            x = np.concatenate(xs) if xs else np.zeros(1)
            sd = float(np.std(x)) if len(x) > 1 else 0.0
            self._sd[key] = sd if sd > 1e-12 else 1.0
        return self._sd[key]

    def pair_betas(self, reg, m6, s_eff):
        """(6, A) per-pair MAP slopes, prior N(m6[B], s_eff^2)."""
        key = (reg, tuple(np.round(m6, 12)), round(s_eff, 12))
        if key in self._pp:
            return self._pp[key]
        R, P = self.rows[reg], self.prep[reg]
        out = np.zeros((6, R.A))
        L = R.own_k.shape[1]
        j = np.arange(L)[None, :]
        for bi in range(1, 6):
            nl = R.nlab[bi]
            use = nl >= 2
            if not use.any():
                out[bi] = m6[bi]
                continue
            mask = (j < nl[:, None]) & use[:, None]
            c = P.x_own - P.xbar_own[bi][:, None]
            u = logit(R.own_pu[bi])
            b = pair_slope(u[use], R.own_y[use], c[use], mask[use], m6[bi], s_eff)
            out[bi] = m6[bi]
            out[bi, use] = b
        self._pp[key] = out
        return out

    def delta(self, reg, config, train):
        """(A, 6) Brier differences of one configuration on one regime, fitted on
        `train` parents. config = (variant, s, start)."""
        key = (reg, tuple(config), tuple(sorted(train)))
        if key not in self._delta:
            self._delta[key] = self._compute_delta(reg, config, train)
        return self._delta[key]

    def _compute_delta(self, reg, config, train):
        variant, s, start = config
        R, P = self.rows[reg], self.prep[reg]
        off = np.zeros_like(P.z_ev)
        if variant == "transferred":
            b = self.beta(train)
            off = b[:, None] * P.c_bench
        else:
            m6 = self.beta(train) if variant == "hybrid" else np.zeros(6)
            bp = self.pair_betas(reg, m6, s / self.sd(train))
            off = bp[:, R.ev_a] * P.c_own
        off[0] = 0.0
        for bi in range(6):
            if BUDGETS[bi] < start:
                off[bi] = 0.0
        off = capped(off)
        out = np.zeros((R.A, 6))
        for bi in range(1, 6):
            if not off[bi].any():
                continue
            p = R.ev_p[bi]
            q = shifted(p, off[bi])
            out[:, bi] = delta_brier(p, q, R.ev_K, R.ev_N, R.ev_a, R.A)
        return out


def configs_of(variant):
    if variant == "transferred":
        return [("transferred", None, st) for st in STARTS]
    return [(variant, s, st) for s in S_GRID for st in STARTS]


def parent_means(rows, dalc, parents):
    """Weighted mean ALC difference per parent, for the parents present."""
    out = {}
    for q in parents:
        m = rows.parent == q
        if m.any():
            out[q] = weighted_mean(dalc[m], rows.w[m])
    return out


def nested(engine, variant, forced=None):
    """Nested leave-one-parent-out (or one forced configuration): per regime the
    (A, 6) differences each appearance gets under its fold's choice, and the
    choices."""
    rows = engine.rows
    cand = configs_of(variant) if forced is None else [forced]
    folds = {q: tuple(p for p in PARENTS if p != q) for q in PARENTS}
    folds["*"] = PARENTS
    out = {reg: np.zeros((R.A, 6)) for reg, R in rows.items()}
    choices = {}
    sel = rows[SELECT_ON]
    for q, train in folds.items():
        if forced is None:
            crit = {}
            for c in cand:
                vals = []
                for q2 in train:
                    inner = tuple(p for p in train if p != q2)
                    d = engine.delta(SELECT_ON, c, inner) @ W6
                    m = sel.parent == q2
                    vals.append(weighted_mean(d[m], sel.w[m]))
                crit[c] = float(np.mean(vals))
            choice = select(crit)
        else:
            crit, choice = None, forced
        choices[q] = {"choice": None if choice is None else list(choice),
                      "inner": None if crit is None else {cname(c): round(v, 7) for c, v in crit.items()}}
        if choice is None:
            continue
        if choice[0] != "per-pair":
            choices[q]["beta"] = [round(float(b), 5) for b in engine.beta(train)]
        choices[q]["sd_x"] = round(engine.sd(train), 6)
        for reg, R in rows.items():
            m = (R.parent == q) if q != "*" else ~np.isin(R.parent, PARENTS)
            if m.any():
                out[reg][m] = engine.delta(reg, choice, train)[m]
    return out, choices


def cname(c):
    v, s, st = c
    return f"{v} from B{st}" if s is None else f"{v} s={s:g} from B{st}"


def summary(rows, dB, boots=BOOTS, seed=0, per=True):
    """Everything reported for one line on one regime."""
    R = rows
    dalc = dB @ W6
    st = stats(dalc, R.w, R.run, R.cluster, R.strata, boots, seed)
    if st is None:
        return None
    pm = parent_means(R, dalc, sorted(set(R.parent.tolist())))
    multi = [pm[q] for q in PARENTS if q in pm]
    st["parent_mean"] = float(np.mean(multi)) if multi else None
    st["parent_se"] = float(np.std(multi, ddof=1) / math.sqrt(len(multi))) if len(multi) > 1 else None
    st["worst_parent"] = max(multi) if multi else None
    st["by_budget"] = [weighted_mean(dB[:, bi], R.w) for bi in range(6)]
    if per:
        st["by_budget_cluster_se"] = [0.0] + [boot_se(dB[:, bi], R.w, R.cluster, None, boots // 4, seed)
                                              for bi in range(1, 6)]
        st["per_parent"] = {}
        for q in sorted(set(R.parent.tolist())):
            m = R.parent == q
            st["per_parent"][q] = {"est": pm[q], "cluster_se": boot_se(dalc[m], R.w[m], R.cluster[m],
                                                                       None, boots // 4, seed),
                                   "n_app": int(m.sum())}
        if R.regime not in GUARDS:
            st["per_kind"] = {k: round(weighted_mean(dalc[R.kind == k], R.w[R.kind == k]), 6)
                              for k in sorted(set(R.kind.tolist()))}
    return st


def gate(line):
    """The plan's acceptance rule on one line's summaries. A forced line has no
    selection, so its `pass` is None and `pass_if_on` says whether the rest of
    the rule holds."""
    tl, mix = line["regimes"].get("tl"), line["regimes"].get("mix")
    on = line.get("folds_on")
    checks = {
        "selection_on": None if on is None else bool(on >= GATE["folds_on"]),
        "tl": tl is not None and tl["est"] <= GATE["tl"],
        "mix_same_sign": mix is not None and tl is not None and np.sign(mix["est"]) == np.sign(tl["est"])
        and tl["est"] != 0,
        "no_parent_worse": tl is not None and tl["worst_parent"] is not None
        and tl["worst_parent"] <= GATE["parent"],
        "guard": all(line["regimes"].get(g) is None or line["regimes"][g]["est"] <= GATE["guard"]
                     for g in GUARDS),
    }
    rest = all(bool(v) for k, v in checks.items() if k != "selection_on")
    checks["pass_if_on"] = rest
    checks["pass"] = None if on is None else bool(rest and checks["selection_on"])
    return checks


def score_covariate(rows, cov, variants=VARIANTS, forced=FORCED, boots=BOOTS, per=True):
    """Nested lines per variant and the forced lines: {line: {...}} plus the raw
    per-appearance differences for averaging over replicates."""
    eng = Engine(rows, cov)
    lines, raw = {}, {}
    for v in variants:
        dB, ch = nested(eng, v)
        raw[f"{v} nested"] = dB
        on = sum(1 for q in PARENTS if ch[q]["choice"] is not None)
        lines[f"{v} nested"] = {"choices": ch, "folds_on": on}
    for c in forced:
        dB, ch = nested(eng, c[0], forced=c)
        raw[cname(c) + " (forced)"] = dB
        lines[cname(c) + " (forced)"] = {"choices": {q: {k: v for k, v in x.items() if k != "inner"}
                                                     for q, x in ch.items()}}
    for name in lines:
        lines[name]["regimes"] = {reg: summary(R, raw[name][reg], boots, 0, per)
                                  for reg, R in rows.items()}
        lines[name]["gate"] = gate(lines[name])
    return lines, raw


def average_lines(rows, reps, boots=BOOTS):
    """Per-appearance differences averaged over replicate covariates, summarised
    again; the replicates' own estimates' spread beside them."""
    names = list(reps[0][1])
    out = {}
    for name in names:
        dB = {reg: np.mean([r[1][name][reg] for r in reps], 0) for reg in rows}
        line = {"regimes": {reg: summary(R, dB[reg], boots, 0, True) for reg, R in rows.items()}}
        line["replicates"] = {reg: [round(r[0][name]["regimes"][reg]["est"], 6) for r in reps]
                              for reg in rows}
        line["draw_sd"] = {reg: float(np.std(v, ddof=1)) if len(v) > 1 else None
                           for reg, v in line["replicates"].items()}
        fo = [r[0][name].get("folds_on") for r in reps]
        if fo[0] is not None:
            line["folds_on"] = float(np.mean(fo))
            line["folds_on_reps"] = fo
        line["choices_rep0"] = reps[0][0][name]["choices"]
        line["gate"] = gate(line)
        out[name] = line
    return out


# --- oracles ------------------------------------------------------------------------

def oracle_maps():
    """In-sample Rasch difficulty per parent (testlike.Catalogue.difficulty) and
    honest fold maps: fold f's difficulties from the parent's subjects outside f."""
    from paiec import data as D
    from paiec import official as O
    from paiec import testlike as T
    ins, honest = {}, [dict() for _ in range(HONEST_FOLDS)]
    info = {}
    for par in PARENTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = O.eligible(D.load_pairs([par]))
        ins[par] = T.rasch(ps)
        info[par] = {"pairs": len(ps), "items": len(ins[par])}
        for f in range(HONEST_FOLDS):
            keep = [p for p in ps if fold_of(p.subject_id) != f]
            hf = T.rasch(keep)
            honest[f].update(hf)
            info[par][f"unseen_fold{f}"] = len(set(ins[par]) - set(hf))
    return ins, honest, info


def fold_of(sid):
    from paiec.evaluator import stable_hash
    return stable_hash(HONEST_SALT, "fold", str(sid)) % HONEST_FOLDS


def base_matrix(maps, keys, parent_of_key):
    """(F, K) difficulties standardised within parent, one row per fold map, and
    where each is present; items outside the four parents are never present."""
    F = len(maps)
    Z = np.zeros((F, len(keys)))
    has = np.zeros((F, len(keys)), bool)
    pk = np.array([parent_of_key.get(k, "") for k in keys], object)
    for f, m in enumerate(maps):
        v = np.array([m.get(k, np.nan) if pk[j] else np.nan for j, k in enumerate(keys)], float)
        has[f] = np.isfinite(v)
        Z[f, has[f]] = standardise_within(v[has[f]], pk[has[f]])
    return Z, has, pk != ""


def realised_r(rows, cov, Z):
    """Mean over test-like pair appearances of the correlation between a
    covariate and the base it was degraded from (the appearance's fold), over
    the pair's evaluated items: lower than r, as a pseudo-benchmark spans less
    difficulty than its parent."""
    R = rows["tl"]
    f = cov.folds(R)
    rs = []
    for a in range(R.A):
        m = R.ev_a == a
        if m.sum() < 5:
            continue
        x, t = cov.X[f[a], R.ev_k[m]], Z[f[a], R.ev_k[m]]
        if x.std() > 0 and t.std() > 0:
            rs.append(np.corrcoef(x, t)[0, 1])
    return float(np.mean(rs)) if rs else None


def degraded_table(rows, keys, Z, has, in_parent, fold_fn, reps, one, save):
    """r -> the lines, each averaged over `reps` noise draws: x = r z + sqrt(1 -
    r^2) e over the four parents' items (e alone where the base lacks the item),
    0 elsewhere; r = 0 is the uninformative covariate."""
    table = {}
    for r in (0.0,) + tuple(R_GRID):
        runs = []
        cov0 = None
        for rep in range(reps):
            eps = np.random.default_rng([rep, 7919]).standard_normal(len(keys))
            X = np.where(in_parent[None, :], degrade(np.where(has, Z, 0.0), r, eps[None, :]), 0.0)
            cov = Covariate(X, fold_fn, f"r={r}")
            cov0 = cov0 or cov
            runs.append(one(f"r={r:g} rep {rep}", cov, boots=200, per=False))
        avg = average_lines(rows, runs)
        table[f"r={r:g}"] = {"meta": {"r": r, "r_within_pair_tl": realised_r(rows, cov0, Z)},
                             "lines": avg}
        save(table)
        print(f"r={r:g}: " + "  ".join(
            f"{ln}: tl {avg[ln]['regimes']['tl']['est']:+.5f}" for ln in avg), flush=True)
    return table


def stage_table(args):
    t0 = time.time()
    rows, keys = load_rows(args.rows, args.regimes)
    print(f"table: rows of {', '.join(f'{r} {len(set(R.run.tolist()))}' for r, R in rows.items())} "
          f"runs loaded in {time.time() - t0:.0f}s", flush=True)
    ins, honest, info = oracle_maps()
    print(f"table: oracle difficulties in {time.time() - t0:.0f}s", flush=True)
    par_of_key = {k: par for par, d in ins.items() for k in d}
    out = {"meta": {"plan": {r: PLAN[r] for r in rows}, "runs_loaded": {r: int(len(set(R.run.tolist())))
                                                                       for r, R in rows.items()},
                    "appearances": {r: R.A for r, R in rows.items()},
                    "hier_failures": {r: R.hier_failures for r, R in rows.items()},
                    "fit_on": FIT_ON, "select_on": SELECT_ON, "s_grid": S_GRID, "starts": STARTS,
                    "cap": CAP, "boots": BOOTS, "n_reps": args.reps, "r_grid": R_GRID,
                    "honest_folds": HONEST_FOLDS, "gate": GATE, "accept": ACCEPT,
                    "oracle_info": info, "recomputed": RECOMPUTED, **provenance()}}
    state = load_json(args.out) or {}
    if "verify" in state:
        out["verify"] = state["verify"]

    def one(name, cov, boots=BOOTS, per=True):
        t = time.time()
        lines, raw = score_covariate(rows, cov, boots=boots, per=per)
        print(f"  {name}: {time.time() - t:.0f}s  " + "  ".join(
            f"{ln}: tl {lines[ln]['regimes']['tl']['est']:+.5f} mix {lines[ln]['regimes']['mix']['est']:+.5f}"
            for ln in ("transferred nested", "per-pair nested", "hybrid nested")), flush=True)
        return lines, raw

    # acceptance: the in-sample oracle on its raw logit scale, as in the heads study
    cov_ins = Covariate.from_dict(ins_all(ins), keys, "oracle")
    lines, _ = one("oracle (in sample)", cov_ins)
    tl_est = lines["transferred from B1 (forced)"]["regimes"]["tl"]["est"]
    mix_est = lines["transferred from B1 (forced)"]["regimes"]["mix"]["est"]
    out["acceptance"] = {
        "check": {"line": "transferred from B1 (forced)", "tl": tl_est, "mix": mix_est,
                  "target": ACCEPT, "pass": abs(tl_est - ACCEPT["tl"]) <= ACCEPT["tol"]
                  and abs(mix_est - ACCEPT["mix"]) <= ACCEPT["tol"]},
        "oracle_in_sample": lines}
    print(f"acceptance: transferred oracle tl {tl_est:+.5f} (target {ACCEPT['tl']}), mix {mix_est:+.5f} "
          f"(target {ACCEPT['mix']}): {'pass' if out['acceptance']['check']['pass'] else 'FAIL'}",
          flush=True)
    save_json(args.out, out)
    cov_hon = Covariate.from_folds(honest, fold_of, keys, "honest oracle")
    lines, _ = one("oracle (honest, subject folds)", cov_hon)
    out["acceptance"]["oracle_honest"] = lines
    save_json(args.out, out)
    # the degraded oracles: the gate table on the honest base, the heads study's on the in-sample one
    out["tables"] = {}
    for base, maps, fn in (("honest", honest, fold_of), ("in_sample", [ins_all(ins)], None)):
        Z, has, in_parent = base_matrix(maps, keys, par_of_key)

        def save(table, base=base):
            out["tables"][base] = table
            save_json(args.out, out)

        table = degraded_table(rows, keys, Z, has, in_parent, fn, args.reps, one, save)
        out["tables"][base] = table
        out.setdefault("thresholds", {})[base] = thresholds(table)
        save_json(args.out, out)
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    save_json(args.out, out)
    print(f"table: done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


def ins_all(ins):
    return {k: v for d in ins.values() for k, v in d.items()}


def thresholds(table):
    """Per line, the tl / mix / guard estimate at each r (with the cluster SE, the
    spread over noise draws and the worst parent) and the smallest r whose line
    passes the gate (a forced line: would pass if selection switched it on): the
    one table every covariate is read against."""
    out = {}
    names = list(next(iter(table.values()))["lines"])
    for name in names:
        row = {}
        first = None
        for key, t in table.items():
            ln = t["lines"][name]
            rg = ln["regimes"]
            row[key] = {reg: round(rg[reg]["est"], 6) for reg in rg if rg[reg] is not None}
            row[key]["tl_cluster_se"] = round(rg["tl"]["cluster_se"], 6)
            row[key]["tl_worst_parent"] = round(rg["tl"]["worst_parent"], 6)
            if ln.get("draw_sd"):
                row[key]["tl_draw_sd"] = round(ln["draw_sd"]["tl"], 6)
            if "folds_on" in ln:
                row[key]["folds_on"] = ln["folds_on"]
            ok = ln["gate"]["pass"] if ln["gate"]["pass"] is not None else ln["gate"]["pass_if_on"]
            row[key]["pass" if ln["gate"]["pass"] is not None else "pass_if_on"] = ok
            if first is None and ok:
                first = t["meta"]["r"]
        row["smallest_r_passing" if "folds_on" in ln else "smallest_r_passing_if_on"] = first
        out[name] = row
    return out


# --- eval one covariate ---------------------------------------------------------------

def read_covariate(path):
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return {str(k): float(v) for k, v in d.items() if v is not None and math.isfinite(float(v))}
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    col = "x" if "x" in df.columns else [c for c in df.columns if c != "item_key"][0]
    df = df[np.isfinite(df[col].astype(float))]
    return dict(zip(df["item_key"].astype(str), df[col].astype(float)))


def stage_eval(args):
    t0 = time.time()
    rows, keys = load_rows(args.rows, args.regimes)
    d = read_covariate(args.cov)
    cov = Covariate.from_dict(d, keys, args.name)
    covered = {reg: float(np.mean(np.isin(np.array(keys, object)[R.ev_k], list(d))))
               for reg, R in rows.items()}
    lines, _ = score_covariate(rows, cov)
    res = {"name": args.name, "cov": os.path.abspath(args.cov), "n_items": len(d),
           "coverage_eval_items": covered, "lines": lines, **provenance()}
    thr = load_json(OUT)
    if thr and "thresholds" in thr:
        res["thresholds"] = thr["thresholds"]
    res["wall_s"] = round(time.time() - t0, 1)
    out = args.out if args.out != OUT else os.path.join(ROOT, "results", f"harness_{args.name}.json")
    save_json(out, res)
    for name, ln in lines.items():
        rg = ln["regimes"]
        print(f"{name:34s} tl {rg['tl']['est']:+.5f} ± {rg['tl']['run_se']:.5f} / "
              f"{rg['tl']['cluster_se']:.5f} / {rg['tl']['strat_se']:.5f}  mix {rg['mix']['est']:+.5f}  "
              + "  ".join(f"{g} {rg[g]['est']:+.5f}" for g in GUARDS if rg.get(g))
              + "  gate " + ({True: "PASS", False: "fail"}[ln["gate"]["pass"]] if ln["gate"]["pass"] is not None
                             else "(forced: pass if on)" if ln["gate"]["pass_if_on"] else "(forced: fail)"),
              flush=True)
    print(f"eval: {time.time() - t0:.0f}s -> {out}", flush=True)


# --- show -----------------------------------------------------------------------------

def _f(x, nd=4):
    return "" if x is None else f"{x:+.{nd}f}"


def line_row(name, ln, nd=4):
    """One markdown row: tl (± run / cluster / stratified SE), mix, worst parent,
    parent SE, the guards, folds on, gate."""
    rg = ln["regimes"]
    tl = rg["tl"]
    cells = [name, f"{_f(tl['est'], nd)} ± {tl['run_se']:.{nd}f} / {tl['cluster_se']:.{nd}f} / "
                   f"{tl['strat_se']:.{nd}f}",
             _f(rg["mix"]["est"], nd) if rg.get("mix") else "",
             _f(tl["worst_parent"], nd), f"{tl['parent_se']:.{nd}f}" if tl.get("parent_se") else "",
             _f(rg["r1b"]["est"], nd) if rg.get("r1b") else "",
             _f(rg["r1p"]["est"], nd) if rg.get("r1p") else "",
             "" if ln.get("folds_on") is None else f"{ln['folds_on']:g}/4",
             ("pass" if ln["gate"]["pass"] else "no") if ln["gate"]["pass"] is not None
             else ("(pass if on)" if ln["gate"]["pass_if_on"] else "(no)")]
    return "| " + " | ".join(cells) + " |"


HEAD = ("| line | test-like ± run / cluster / strat SE | mix/whole | worst parent | parent SE "
        "| R1 benchmark-first | R1 pair-uniform | folds on | gate |\n|" + "---|" * 9)


def stage_show(args):
    """Markdown tables of a --stage table result (or an --stage eval one)."""
    res = load_json(args.out if args.cov is None else args.cov)
    if "lines" in res:                                  # an eval result
        print(HEAD)
        for name, ln in res["lines"].items():
            print(line_row(name, ln))
        return
    acc = res["acceptance"]
    print(f"acceptance: {acc['check']}")
    for key in ("oracle_in_sample", "oracle_honest"):
        if key in acc:
            print(f"\n{key}\n{HEAD}")
            for name, ln in acc[key].items():
                print(line_row(name, ln))
    for base, table in res.get("tables", {}).items():
        print(f"\n{base} degraded oracles (per-appearance differences averaged over draws)")
        for r, t in table.items():
            print(f"\n{r} (within-pair r on test-like: {t['meta']['r_within_pair_tl']:.3f})\n{HEAD}")
            for name, ln in t["lines"].items():
                print(line_row(name, ln, 5))
        print(f"\n{base}: test-like by budget, B0..B31, per line and r")
        for r, t in table.items():
            for name, ln in t["lines"].items():
                bb = ln["regimes"]["tl"]["by_budget"]
                print(f"{r:6s} {name:34s} " + " ".join(f"{v:+.5f}" for v in bb))
        print(f"\n{base}: test-like per held-out parent")
        for r, t in table.items():
            for name, ln in t["lines"].items():
                pp = ln["regimes"]["tl"]["per_parent"]
                print(f"{r:6s} {name:34s} " + " ".join(f"{q[:5]} {v['est']:+.5f} ({v['cluster_se']:.5f})"
                                                      for q, v in pp.items()))


# --- io -----------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return round(v, 7) if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("collect", "verify", "table", "eval", "show"))
    ap.add_argument("--regimes", nargs="+", default=list(PLAN), choices=list(PLAN))
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="collect runs 0..n-1 instead of PLAN's")
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--cov", default=None)
    ap.add_argument("--name", default="covariate")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.stage == "collect":
        stage_collect(args)
    elif args.stage == "verify":
        stage_verify(args)
    elif args.stage == "table":
        stage_table(args)
    elif args.stage == "show":
        stage_show(args)
    else:
        if not args.cov:
            ap.error("--stage eval needs --cov")
        stage_eval(args)


if __name__ == "__main__":
    main()
