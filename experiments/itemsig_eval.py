"""Item signal from the pair's own labels: paiec/itemsig.py on the shipped hier.

What is measured. paiec.itemsig.ItemSig moves the base prediction of a target
by a logit offset read from the residuals of labeled items with similar text on
the target's benchmark: its own pair's (weight 1) and other subjects' (weight
`other`). The base is exactly what ships: paiec.hier.HierPredictor with
submission/model.py's LEVEL (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5) over
prior.build, here fitted leaving each target's parent benchmark out
(experiments/level_calibration.py's bundle and factory: one model per parent in
the run, dispatched on the anonymous benchmark_id). Every layer configuration
is scored against that base on identical checkpoints, so every difference is
exactly paired.

Regimes (level_calibration.REGIMES, same seeds and runs as there):
  tl            primary: testlike.Regime() at its defaults, seed 2
  tl mix/whole  sensitivity: groups merged at random and wholes, no strata,
                seed 3 (difficulty strata shrink within-pair item variance)
  r1b, r1p      secondary: public R1 (official.sample_run, seed 0, split scope
                'pair'), benchmark-first and pair-uniform

Configurations. The grid over the hyperparameters the layer exposes for
tuning: beta (gain) x tau (prior precision) x gamma (sharpening) x other (the
weight of another subject's record against the own pair's) x eb (the per-pair
empirical-Bayes slope off or on, prior sd 1). The rest stays at SigHyper's
defaults (leave-in residuals centred within pair, content twins on, word weight
0.5, max_shift 1.5, head 2,000 and tail 6,000 characters). "off" (the base
itself, difference 0) is a candidate in every selection.

Harness. level_calibration.checkpoints replays paiec.official's _slots and
_acquire (the platform's random policy); every checkpoint gets a fresh base
instance and a fresh ItemSig around it, one worker, deduplicated inputs, no
argument copies. Per target the base is called once and ItemSig.terms once
(built with other > 0, so the base's p_j cover every record on the benchmark);
every configuration's shift then follows from those terms exactly as
itemsig.shift computes it (the slope through itemsig.slope itself). --stage
verify replays runs through official.run_official with deep copies and
make_itemsig for several configurations and checks the stored rows pair for
pair; it also times the layer's calls (latency).

Selection honesty. Configurations are chosen by nested leave-one-parent-out:
for each parent q, the configuration (or "off") with the lowest mean ALC
difference over the pair appearances whose parent is not q, scored on the
appearances whose parent is q. A target's layer reads only records on its own
benchmark_id, so a pair's difference depends on its own benchmark's labels and
the (fixed) base alone; runs mix parents, so the split is by pair appearance,
each weighted 1/run size as in a run's mean. The nested estimate is the mean
over runs of the run means of those out-of-parent differences. Its cluster
bootstrap redoes the selection inside every resample. The in-sample best (the
lowest mean on all appearances, scored on the same) is reported as a secondary,
optimistic number. Selections: within each regime; on the primary regime,
scored on the others (transfer); and jointly, by the worst regime's mean
over tl, tl mix/whole, r1b and r1p.

Statistics. Paired ALC differences against the shipped hier, "± run SE /
cluster SE / stratified SE": over runs; a cluster bootstrap (2,000 resamples, a
ratio estimator) with (parent, subject) as the cluster (testlike.cluster_key;
(benchmark, subject) on public runs), all clusters as one pool or within each
parent. Oracles on the same pairs (testlike.item_oracle, with theta refitted
by theta_map, as the library's undamped Newton can diverge): the pair-rate
oracle (every evaluated response at the pair's own rate) and the item oracle
(the parent's in-sample Rasch difficulty, theta fitted per pair on its
evaluated responses); their gap is the most an item-difficulty model could add.

Run:  python experiments/itemsig_eval.py --stage run --jobs 4 --rows DIR
      python experiments/itemsig_eval.py --stage oracle --jobs 4 --rows DIR
      python experiments/itemsig_eval.py --summarise --rows DIR
      python experiments/itemsig_eval.py --stage verify --jobs 4 --rows DIR
      python experiments/itemsig_eval.py --summarise --rows DIR
(verify replays the configurations the first summary chose, so it follows it;
38 minutes on four processes of a shared machine)
Rows (one .npz per run, per pair and configuration Brier differences by budget
in float32) go to --rows, default data/itemsig_eval_rows (gitignored), and are
resumable; results/itemsig_eval.json holds the summary, per-run summaries,
verification, latency and the passes. docs/findings.md, "Item signal from the
pair's own labels", reads it.
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
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import fields  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
from paiec import itemsig as I  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.predict import HI, LO  # noqa: E402

OUT = os.path.join(ROOT, "results", "itemsig_eval.json")
ROWS = os.path.join(ROOT, "data", "itemsig_eval_rows")
BOOTS = 2000
W6 = np.asarray(WEIGHTS, float)


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


_LEVEL = shipped_level()
SHIP = LC.gname(_LEVEL["mu0"], _LEVEL["sigma_mu"], _LEVEL["attr_scale"])
assert LC.parse(SHIP)[1] == {k: float(v) for k, v in _LEVEL.items()}, "SHIP is not LEVEL"

#: regime -> runs 0..n-1 scored
PLAN = {"tl": 300, "tl mix/whole": 200, "r1b": 200, "r1p": 100}
#: scored first, in every regime, before PLAN's extensions
CORE = {"tl": 200, "tl mix/whole": 100, "r1b": 100, "r1p": 100}
PRIMARY = "tl"
JOINT = ("tl", "tl mix/whole", "r1b", "r1p")

# --- the grid ------------------------------------------------------------------------------

BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)
TAUS = (0.5, 1.0, 2.0, 4.0)
GAMMAS = (1.0, 2.0, 4.0, 8.0)
OTHERS = (0.0, 0.5, 1.0)
EBS = (0.0, 1.0)
#: everything else at SigHyper's defaults; checked below
FIXED = {"word": 0.5, "center": True, "twins": True, "max_shift": 1.5, "resid": "leave_in",
         "head": 2_000, "tail": 6_000}
_DEF = I.SigHyper()
assert all(getattr(_DEF, k) == v for k, v in FIXED.items()), "SigHyper defaults moved"
#: configurations, in the order of every stored array's config axis
CONFIGS = [(b, t, g, o, e) for e in EBS for o in OTHERS for g in GAMMAS for t in TAUS
           for b in BETAS]
C = len(CONFIGS)
#: the terms every target is built with: other > 0, so p_j cover every record
H_ALL = I.SigHyper(other=1.0)


def cname(c):
    b, t, g, o, e = c
    return f"beta {b:g} tau {t:g} gamma {g:g} other {o:g} eb {e:g}"


NAMES = [cname(c) for c in CONFIGS]
OFF = "off"
CAND = [OFF] + NAMES            # the selection's candidates: column 0 is the base itself


def hyper_of(c):
    b, t, g, o, e = c
    return I.SigHyper(beta=b, tau=t, gamma=g, other=o, eb=e, **FIXED)


DEFAULT = CONFIGS.index((_DEF.beta, _DEF.tau, _DEF.gamma, _DEF.other, _DEF.eb))
assert hyper_of(CONFIGS[DEFAULT]) == _DEF

# index arrays: a config's (gamma, other) weight row, tau, beta, eb
_WO = [(g, o) for g in GAMMAS for o in OTHERS]
_CI_WO = np.array([_WO.index((c[2], c[3])) for c in CONFIGS])
_CI_T = np.array([TAUS.index(c[1]) for c in CONFIGS])
_CB = np.array([c[0] for c in CONFIGS])
_CEB = np.array([c[4] > 0 for c in CONFIGS])
_SLOPE_KEYS = [(g, o, t) for g in GAMMAS for o in OTHERS for t in TAUS]
_CI_S = np.array([_SLOPE_KEYS.index((c[2], c[3], c[1])) for c in CONFIGS])
_SLOPE_H = {k: I.SigHyper(gamma=k[0], other=k[1], tau=k[2], eb=1.0, **FIXED) for k in _SLOPE_KEYS}
_MAXS = FIXED["max_shift"]


def grid_shifts(t):
    """Every configuration's logit shift for one target's terms, as
    itemsig.shift(t, hyper_of(c)) gives it: the weights and the Newton step
    in bulk, the per-pair slope through itemsig.slope (cached per pair in the
    terms' _Bench)."""
    if not len(t.y):
        return np.zeros(C)
    sim = np.clip(FIXED["word"] * t.cw + (1 - FIXED["word"]) * t.cc, 0.0, 1.0)
    sim = np.where(t.twin, 1.0, sim)
    W = np.empty((len(_WO), len(sim)))
    for k, (g, o) in enumerate(_WO):
        W[k] = np.where(t.excl, 0.0, sim ** g * np.where(t.own, 1.0, o))
    r, v = I.residuals(t, H_ALL)
    num, den = W @ r, W @ v
    raw = num[_CI_WO] / (np.asarray(TAUS)[_CI_T] + den[_CI_WO])
    slopes = np.ones(C)
    if _CEB.any():
        s = np.array([I.slope(t, _SLOPE_H[k]) for k in _SLOPE_KEYS])
        slopes = np.where(_CEB, s[_CI_S], 1.0)
    d = _CB * slopes * raw
    if not np.all(np.isfinite(d)):
        raise FloatingPointError("non-finite shift")
    return np.clip(d, -_MAXS, _MAXS)


def apply(p0, d):
    """itemsig.ItemSig.predict's output for base p0 and shifts d."""
    z = math.log(p0 / (1 - p0)) + d
    q = np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))
    return np.where(d == 0.0, p0, np.clip(q, LO, HI))


# --- one run -------------------------------------------------------------------------------

def draw(regime, i):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LC.draw(regime, i)


def run_meta(regime, run, slots):
    """Per pair: the run descriptor (level_calibration.describe), the
    pseudo-benchmark kind, pairs of its benchmark in the run, the labels its
    benchmark holds at B31 (own and other subjects') and the two oracles."""
    desc = LC.describe(regime, run)
    orc = T.item_oracle(run, LC.catalogue())
    by_b = {}
    for s in slots:
        by_b.setdefault(s.benchmark_id, []).append(s)
    out = []
    for d, o, s in zip(desc, orc, slots):
        sib = by_b[s.benchmark_id]
        own = len(s.acquired[:31])
        d = dict(d)
        d.update(kind=T.kind_of(d["bench"]), same=len(sib), own=own,
                 other=sum(len(x.acquired[:31]) for x in sib) - own,
                 level=None if o is None else round(o["level"], 6),
                 item=None if o is None else round(o["item"], 6),
                 b_sd=None if o is None else round(o["b_sd"], 4))
        out.append(d)
    return out


def evaluate(slots, cps, fac):
    """Per pair Brier by budget of the base and every configuration's
    difference from it: (base (pairs, 6), diff (pairs, C, 6)), and counters."""
    preds = {}
    docs = {}
    tm = {"targets": 0, "with_terms": 0, "twin": 0, "fails": 0, "pj_calls": 0,
          "base_s": 0.0, "terms_s": 0.0, "grid_s": 0.0}
    abs_shift = np.zeros(C)
    moved = np.zeros(C)
    for b, labeled, inputs, index in cps:
        sig = I.ItemSig(fac())
        sig._docs = docs            # a cache of a pure function of the item, shared across checkpoints
        P = np.empty((len(inputs), 1 + C))
        for j, inp in enumerate(inputs):
            t0 = time.perf_counter()
            p0 = sig.base_prediction(inp, labeled)
            t1 = time.perf_counter()
            tm["base_s"] += t1 - t0
            P[j] = p0
            if not b:
                continue
            tm["targets"] += 1
            try:
                t = sig.terms(inp, labeled, kinds=("leave_in",), hyper=H_ALL)
                t2 = time.perf_counter()
                tm["terms_s"] += t2 - t1
                if t is None or not len(t.y):
                    continue
                d = grid_shifts(t)
            except Exception:
                tm["fails"] += 1
                continue
            tm["with_terms"] += 1
            tm["twin"] += int(bool(np.any(t.twin & ~t.excl)))
            abs_shift += np.abs(d)
            moved += d != 0.0
            P[j, 1:] = apply(p0, d)
            tm["grid_s"] += time.perf_counter() - t2
        st = next(iter(sig._states.values()), None)
        if st is not None:
            tm["pj_calls"] += sum(bn.calls for bn in st.benches.values() if bn is not None)
        preds[b] = P[index]
    base, diff = [], []
    at = 0
    for s in slots:
        y = np.array([x[1] for x in s.targets], float)
        sl = slice(at, at + len(y))
        at += len(y)
        br = np.stack([np.mean((preds[b][sl] - y[:, None]) ** 2, 0) for b in BUDGETS], 1)
        base.append(br[0])
        diff.append(br[1:] - br[0])
    n = max(tm["with_terms"], 1)
    return np.array(base), np.array(diff), abs_shift / n, moved / n, tm


def path_of(rows, regime, i):
    return os.path.join(rows, f"{regime.replace(' ', '_').replace('/', '-')}_{i}.npz")


def run_task(task):
    rows, regime, i = task
    path = path_of(rows, regime, i)
    if os.path.exists(path):
        return regime, i, "skip", 0.0
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = draw(regime, i)
        slots, cps = LC.checkpoints(run)
        meta = run_meta(regime, run, slots)
        made = []
        base, diff, abs_shift, moved, tm = evaluate(slots, cps, LC.factory([SHIP], run, made))
    tm["hier_failures"] = int(sum(getattr(m, "failures", 0) for m in made))
    tm["unconverged"] = int(sum(getattr(m, "unconverged", 0) for m in made))
    tm["task_s"] = round(time.perf_counter() - t0, 3)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, base=base, diff=diff.astype(np.float32),
                        abs_shift=abs_shift.astype(np.float32), moved=moved.astype(np.float32),
                        meta=json.dumps(meta), tm=json.dumps(tm), names=json.dumps(NAMES))
    os.replace(tmp, path)
    return regime, i, "done", tm["task_s"]


def prepare():
    """Everything a worker reads, built once in the parent before forking."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parents = sorted({p.benchmark_id for p in LC.pairs()})
        bundles = {(par, 0.0): LC.bundle(par, 0.0) for par in parents}
        LC.catalogue()
        for regime in PLAN:
            if LC.REGIMES[regime][0] == "testlike":
                LC.sampler(regime)
    return bundles


def stage_run(args, state):
    plan = dict(PLAN)
    if args.only:
        plan = {x.rsplit(":", 1)[0]: int(x.rsplit(":", 1)[1]) for x in args.only.split(",")}
    os.makedirs(args.rows, exist_ok=True)
    todo = [(args.rows, r, i) for r, n in plan.items() for i in range(n)
            if not os.path.exists(path_of(args.rows, r, i))]
    # every regime's first CORE runs before any extension, so a cut run is balanced
    order = list(plan)
    todo.sort(key=lambda t: (t[2] >= CORE.get(t[1], 0), order.index(t[1]), t[2]))
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    bundles = prepare()
    print(f"{len(todo)} runs to score, {C} configurations; prepared in {time.time() - t0:.0f}s",
          flush=True)
    done = 0
    with mp.get_context("fork").Pool(args.jobs, initializer=LC.init,
                                      initargs=(bundles, {})) as pool:
        for regime, i, st, dt in pool.imap_unordered(run_task, todo):
            done += 1
            print(f"  {done}/{len(todo)} {regime} {i} {st} {dt:.1f}s  "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    state["passes"].append({"stage": "run", "command": " ".join(sys.argv), "runs": len(todo),
                            "plan": plan, "wall_s": round(time.time() - t0, 1),
                            "jobs": args.jobs, **provenance()})


# --- the item oracle, fitted safely ----------------------------------------------------------

def theta_map(y, z, prior_sd=5.0, iters=200):
    """The MAP of theta in y ~ Bernoulli(sigmoid(theta - z)), theta ~ N(0,
    prior_sd^2): Newton with step halving on the (concave) log posterior.
    testlike.item_oracle takes full Newton steps from 0, which oscillate
    without converging when the pair's items sit far from theta 0 (one
    matharena stratum pair at rate 0.67 on difficulties averaging 3.6 swings
    between -2050 and +4150 and scores Brier 0.67 where the MAP scores 0.14)."""
    def lp(t):
        s = t - z
        return float(np.sum(y * s - np.logaddexp(0.0, s)) - t * t / (2 * prior_sd ** 2))
    th, cur = 0.0, lp(0.0)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(th - z)))
        step = (np.sum(y - q) - th / prior_sd ** 2) / (np.sum(q * (1 - q)) + 1 / prior_sd ** 2)
        a = 1.0
        new = lp(th + step)
        while new < cur and a > 1e-12:
            a /= 2
            new = lp(th + a * step)
        if new < cur:
            break
        th, cur = th + a * step, new
        if abs(a * step) < 1e-10:
            break
    return th


def item_oracle(run, catalogue, seed=0, scope="pair", prior_sd=5.0):
    """testlike.item_oracle with theta_map in place of its undamped Newton;
    'lib_item' keeps the library's value for comparison."""
    lib = T.item_oracle(run, catalogue, seed, scope, prior_sd)
    out = []
    for (p, items), o in zip(run, lib):
        if o is None:
            out.append(None)
            continue
        b = catalogue.difficulty[T.parent_of(p.benchmark_id)]
        _, ev = (set(k) & items for k in O.split(p, seed, scope))
        by = p.by_item()
        y = np.array([x.label for k in sorted(ev) for x in by[k]], float)
        z = np.array([b[k] for k in sorted(ev) for _ in by[k]])
        q = 1 / (1 + np.exp(-(theta_map(y, z, prior_sd) - z)))
        out.append({"level": round(o["level"], 6), "item": round(float(np.mean((q - y) ** 2)), 6),
                    "lib_item": round(o["item"], 6), "b_sd": round(o["b_sd"], 4)})
    return out


def oracle_task(task):
    regime, i = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return regime, i, item_oracle(draw(regime, i), LC.catalogue())


def oracle_path(rows):
    return os.path.join(rows, "oracle.json")


def stage_oracle(args, state):
    """The item oracle of every stored run, fitted with theta_map, into
    <rows>/oracle.json (cheap: draws the runs again, fits nothing else)."""
    path = oracle_path(args.rows)
    got = load(path) or {}
    todo = [(r, i) for r, n in PLAN.items() for i in range(n)
            if os.path.exists(path_of(args.rows, r, i)) and str(i) not in got.get(r, {})]
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LC.catalogue()
        for regime in PLAN:
            if LC.REGIMES[regime][0] == "testlike":
                LC.sampler(regime)
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for regime, i, o in pool.imap_unordered(oracle_task, todo, chunksize=8):
            got.setdefault(regime, {})[str(i)] = o
    save(path, got)
    print(f"oracle: {len(todo)} runs in {time.time() - t0:.0f}s", flush=True)
    state["passes"].append({"stage": "oracle", "command": " ".join(sys.argv), "runs": len(todo),
                            "wall_s": round(time.time() - t0, 1), "jobs": args.jobs,
                            **provenance()})


# --- reading the rows ----------------------------------------------------------------------

class Reg:
    """One regime's stored runs as flat per-appearance arrays."""

    def __init__(self, rows, regime, n, oracle=None):
        """oracle: {run index (str): item_oracle's rows}, which replace the
        rows' own item oracle (testlike.item_oracle's, kept as lib_item)."""
        runs = []
        self.oracle_fixed = oracle is not None
        for i in range(n):
            p = path_of(rows, regime, i)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            if json.loads(str(z["names"])) != NAMES:
                raise ValueError(f"{p}: stored under another grid")
            meta = json.loads(str(z["meta"]))
            if oracle is not None:
                orc = oracle[str(i)]
                for m, o in zip(meta, orc):
                    m["lib_item"] = m["item"]
                    if o is not None:
                        if abs(o["level"] - m["level"]) > 1e-6 or abs(o["lib_item"] - m["item"]) > 1e-6:
                            raise ValueError(f"{p}: oracle rows do not match the run")
                        m["item"] = o["item"]
            runs.append({"i": i, "base": z["base"], "diff": z["diff"],
                         "abs_shift": z["abs_shift"], "moved": z["moved"],
                         "meta": meta, "tm": json.loads(str(z["tm"]))})
        if not runs:
            raise ValueError(f"no rows for {regime}")
        self.name, self.runs = regime, runs
        self.R = len(runs)
        self.metas = [r["meta"] for r in runs]
        self.boot = LC.Boot(self.metas)
        self.cl = np.concatenate(self.boot.idx)
        self.K = len(self.boot.keys)
        self.cl_parent = np.array([k[0] for k in self.boot.keys])
        self.run = np.concatenate([[j] * len(r["meta"]) for j, r in enumerate(runs)])
        self.w = np.concatenate([[1.0 / len(r["meta"])] * len(r["meta"]) for r in runs])
        self.pairs = [p for m in self.metas for p in m]
        self.parent = np.array([p["parent"] for p in self.pairs])
        self.parents = sorted(set(self.parent.tolist()))
        self.base = np.concatenate([r["base"] for r in runs])                 # (N, 6)
        D = np.concatenate([r["diff"] for r in runs]).astype(np.float64)       # (N, C, 6)
        self.D = np.concatenate([np.zeros((len(D), 1, 6)), D], 1)             # off first
        self.dalc = self.D @ W6                                               # (N, 1 + C)
        self.n_k = np.bincount(self.cl, self.w, minlength=self.K)
        self.S = np.zeros((self.K, 1 + C))
        np.add.at(self.S, self.cl, self.w[:, None] * self.dalc)

    # per-appearance values -> summaries

    def mean(self, x, sel=None):
        """Mean over runs of the run means; with sel, the weighted mean over
        the selected appearances (a subset's per-appearance mean)."""
        x = np.asarray(x, float)
        if sel is None:
            return float(np.sum(self.w * x) / self.R)
        return float(np.sum(self.w[sel] * x[sel]) / np.sum(self.w[sel]))

    def run_se(self, x):
        per = np.bincount(self.run, self.w * np.asarray(x, float), minlength=self.R)
        return float(per.std(ddof=1) / math.sqrt(self.R))

    def boot_se(self, x, sel=None):
        """Cluster and stratified bootstrap SEs of mean(x, sel), selection fixed."""
        x = np.asarray(x, float)
        keep = np.ones(len(x), bool) if sel is None else sel
        v = np.bincount(self.cl[keep], (self.w * x)[keep], minlength=self.K)
        n = np.bincount(self.cl[keep], self.w[keep], minlength=self.K)
        out = []
        for W in (self.boot.W, self.boot.Ws):
            den = W @ n
            ok = den > 0
            out.append(float(((W @ v)[ok] / den[ok]).std(ddof=1)))
        return out

    def stat(self, x, sel=None):
        m = self.mean(x, sel)
        se = self.boot_se(x, sel)
        out = {"mean": m, "cluster_se": se[0], "strat_se": se[1]}
        if sel is None:
            out["run_se"] = self.run_se(x)
        else:
            out["n"] = int(np.sum(sel))
        return out


def r6(x):
    return None if x is None else round(float(x), 6)


def fmt_stat(s):
    return {k: (r6(v) if isinstance(v, float) else v) for k, v in s.items()}


# --- nested selection ----------------------------------------------------------------------

class Union:
    """Cluster bootstrap weights over the union of every regime's clusters.
    Test-like clusters are (parent, subject) and public ones (benchmark,
    subject), the same keys for the same responses, so one resample draws a
    cluster the same number of times in every regime: the selection on some
    regimes and the score on another see one resample of the public pairs.
    W resamples all clusters as one pool, Ws within each parent."""

    def __init__(self, regs, boots=BOOTS, seed=1):
        keys = sorted({k for g in regs for k in g.boot.keys})
        at = {k: j for j, k in enumerate(keys)}
        K = len(keys)
        rng = np.random.default_rng(seed)
        W = rng.multinomial(K, np.full(K, 1 / K), size=boots).astype(float)
        par = np.array([k[0] for k in keys])
        rs = np.random.default_rng([seed, 2])
        Ws = np.zeros_like(W)
        for p in sorted(set(par.tolist())):
            sel = np.flatnonzero(par == p)
            Ws[:, sel] = rs.multinomial(len(sel), np.full(len(sel), 1 / len(sel)), size=boots)
        self.W = {g.name: W[:, [at[k] for k in g.boot.keys]] for g in regs}
        self.Ws = {g.name: Ws[:, [at[k] for k in g.boot.keys]] for g in regs}
        self._memo = {}

    def parts(self, g, q, strat, inside=False):
        """(W m) @ S and (W m) @ n for regime g, m the clusters whose parent
        is not q (inside: is q), memoised."""
        key = (g.name, q, strat, inside)
        if key not in self._memo:
            W = (self.Ws if strat else self.W)[g.name]
            m = (g.cl_parent == q) if inside else (g.cl_parent != q)
            Wm = W * m
            self._memo[key] = (Wm @ g.S, Wm @ g.n_k)
        return self._memo[key]


def crit(regs, q, U=None, strat=False):
    """Worst regime's mean ALC difference per candidate on appearances whose
    parent is not q: (1 + C,) or, with U (a Union), per resample (B, 1 + C)."""
    out = None
    for g in regs:
        m = g.cl_parent != q
        if not m.any():
            continue
        if U is None:
            c = g.S[m].sum(0) / g.n_k[m].sum()
        else:
            num, den = U.parts(g, q, strat)
            # a resample that draws none of g's clusters outside q leaves g out
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(den[:, None] > 0, num / den[:, None], -np.inf)
        out = c if out is None else np.maximum(out, c)
    return out


def nested(sel_regs, score, U=None, strat=False):
    """Nested leave-one-parent-out: per parent q of the scored regime, the
    candidate with the lowest `crit` without q, scored on q. -> (choice per
    parent, per-appearance values) or, with U, the bootstrap estimates (B,)
    with the selection redone in every resample."""
    parents = sorted(set(score.cl_parent.tolist()))
    if U is None:
        choice = {q: int(np.argmin(crit(sel_regs, q))) for q in parents}
        x = score.dalc[np.arange(len(score.dalc)), [choice[p] for p in score.parent]]
        return choice, x
    Ws = (U.Ws if strat else U.W)[score.name]
    num = np.zeros(len(Ws))
    for q in parents:
        c = np.argmin(crit(sel_regs, q, U, strat), 1)
        num += U.parts(score, q, strat, inside=True)[0][np.arange(len(Ws)), c]
    return num / (Ws @ score.n_k)


def nested_report(sel_regs, score, U):
    """The nested estimate on `score` with SEs (selection redone in every
    resample of U), the choice per parent, per budget, per parent and per
    kind; the SEs marked 'fixed' hold the point estimate's choices."""
    choice, x = nested(sel_regs, score)
    out = {"select_on": [g.name for g in sel_regs], "score": score.name,
           "choice": {q: CAND[c] for q, c in choice.items()},
           "ALC": {"mean": r6(score.mean(x)), "run_se_fixed": r6(score.run_se(x))}}
    for strat in (False, True):
        est = nested(sel_regs, score, U, strat)
        out["ALC"]["strat_se" if strat else "cluster_se"] = r6(est.std(ddof=1))
    fx = score.boot_se(x)
    out["ALC"]["cluster_se_fixed"], out["ALC"]["strat_se_fixed"] = r6(fx[0]), r6(fx[1])
    cols = np.array([choice[p] for p in score.parent])
    per_b = score.D[np.arange(len(cols)), cols]                      # (N, 6)
    out["budgets"] = budget_rows(score, per_b)
    out["per_parent"] = {}
    for q in score.parents:
        sel = score.parent == q
        out["per_parent"][q] = {"choice": CAND[choice[q]], **fmt_stat(score.stat(x, sel)),
                                "b31": r6(score.mean(per_b[:, 5], sel))}
    out["per_kind"] = by_field(score, x, "kind")
    out["alone_vs_shared"] = {k: fmt_stat(score.stat(x, sel)) for k, sel in
                              (("alone", same_of(score) == 1), ("shared", same_of(score) > 1))
                              if sel.any()}
    out["_x"] = x
    out["_b"] = per_b
    return out


def same_of(g):
    return np.array([p["same"] for p in g.pairs])


def by_field(g, x, field):
    vals = np.array([str(p[field]) for p in g.pairs])
    return {v: fmt_stat(g.stat(x, vals == v)) for v in sorted(set(vals.tolist()))}


def budget_rows(g, per_b):
    """Per budget: mean Brier difference (± run / cluster SE, selection
    fixed) and its contribution to ALC (times the budget's weight)."""
    out = []
    for k, b in enumerate(BUDGETS):
        s = g.stat(per_b[:, k])
        out.append({"budget": b, "diff": r6(s["mean"]), "run_se": r6(s["run_se"]),
                    "cluster_se": r6(s["cluster_se"]), "alc_part": r6(W6[k] * s["mean"])})
    return out


# --- summary ---------------------------------------------------------------------------------

def config_table(g):
    """Every candidate on regime g: mean ALC difference and run SE, per budget."""
    m = np.array([g.mean(g.dalc[:, c]) for c in range(1 + C)])
    se = np.array([g.run_se(g.dalc[:, c]) for c in range(1 + C)])
    bud = np.array([[g.mean(g.D[:, c, k]) for k in range(6)] for c in range(1 + C)])
    return m, se, bud


def oracle_rows(g, x, per_b, label):
    """Oracle gap against the layer's gain on the pairs that have an oracle."""
    has = np.array([p["level"] is not None for p in g.pairs])
    if not has.any():
        return None
    lev = np.array([p["level"] if p["level"] is not None else np.nan for p in g.pairs])
    itm = np.array([p["item"] if p["item"] is not None else np.nan for p in g.pairs])
    gap = lev - itm
    base_alc = g.base @ W6
    out = {"pairs": int(has.sum()), "of": len(has), "theta_map": g.oracle_fixed,
           "pair_rate_oracle": r6(g.mean(lev, has)), "item_oracle": r6(g.mean(itm, has)),
           "gap": fmt_stat(g.stat(gap, has)),
           "gap_share_of_pair_rate": r6(g.mean(gap, has) / g.mean(lev, has)),
           "base_alc": r6(g.mean(base_alc, has)), "base_b31": r6(g.mean(g.base[:, 5], has)),
           "base_b31_minus_pair_rate_oracle": r6(g.mean(g.base[:, 5] - lev, has))}
    kinds = np.array([p["kind"] for p in g.pairs])

    def plain(it, sel):
        """testlike_check's `structure` figure: plain means over pairs."""
        lv, im = float(np.mean(lev[sel])), float(np.mean(it[sel]))
        return {"pairs": int(sel.sum()), "pair_rate_oracle": r6(lv), "item_oracle": r6(im),
                "gap_share": r6((lv - im) / lv)}
    out["plain"] = {"all": plain(itm, has),
                    **{f"kind {k}": plain(itm, has & (kinds == k))
                       for k in sorted(set(kinds[has].tolist()))}}
    if g.oracle_fixed:
        lib = np.array([p["lib_item"] if p["lib_item"] is not None else np.nan for p in g.pairs])
        bad = has & (np.abs(lib - itm) > 1e-4)
        out["library_oracle"] = {"item_oracle": r6(g.mean(lib, has)),
                                 "gap_share_of_pair_rate": r6(g.mean(lev - lib, has)
                                                              / g.mean(lev, has)),
                                 "plain": {"all": plain(lib, has),
                                           **{f"kind {k}": plain(lib, has & (kinds == k))
                                              for k in sorted(set(kinds[has].tolist()))}},
                                 "pairs_off_by_1e-4": int(bad.sum()),
                                 "appearance_share_off": r6(bad.sum() / has.sum()),
                                 "runs_with_one": int(len({int(g.run[a]) for a in
                                                           np.flatnonzero(bad)}))}
    if x is not None:
        ga, g31 = -g.mean(x, has), -g.mean(per_b[:, 5], has)
        gp = g.mean(gap, has)
        out[label] = {"gain_alc": r6(ga), "gain_b31": r6(g31),
                      "share_of_gap_alc": r6(ga / gp), "share_of_gap_b31": r6(g31 / gp)}
    out["per_parent"] = {}
    for q in g.parents:
        sel = has & (g.parent == q)
        if sel.any():
            out["per_parent"][q] = {"gap": r6(g.mean(gap, sel)),
                                    "pair_rate_oracle": r6(g.mean(lev, sel)),
                                    "item_oracle": r6(g.mean(itm, sel)),
                                    "base_b31": r6(g.mean(g.base[:, 5], sel))}
    return out


def summarise(args, state):
    regs = {}
    orc = load(oracle_path(args.rows)) or {}
    for r, n in PLAN.items():
        try:
            regs[r] = Reg(args.rows, r, n, orc.get(r))
        except (ValueError, KeyError) as e:
            print(f"skipping {r}: {e!r}", flush=True)
    if PRIMARY not in regs:
        raise SystemExit("no primary rows")
    summ = {"grid": {"beta": BETAS, "tau": TAUS, "gamma": GAMMAS, "other": OTHERS, "eb": EBS,
                     "fixed": FIXED, "configs": C, "default": NAMES[DEFAULT], "ship": SHIP},
            "regimes": {}}
    per_run = {}
    joint = [regs[r] for r in JOINT if r in regs]
    U = Union(list(regs.values()))
    for name, g in regs.items():
        m, se, bud = config_table(g)
        best = int(np.argmin(m))
        top = np.argsort(m)[:10]
        dflt = 1 + DEFAULT
        within = nested_report([g], g, U)
        transfer = nested_report([regs[PRIMARY]], g, U) if name != PRIMARY else None
        jnt = nested_report(joint, g, U) if g in joint else None
        base_alc = g.base @ W6
        tms = [r["tm"] for r in g.runs]
        rs = {"runs": g.R, "appearances": len(g.pairs), "parents": g.parents,
              "base": {"ALC": r6(g.mean(base_alc)), "ALC_run_se": r6(g.run_se(base_alc)),
                       "budgets": [r6(g.mean(g.base[:, k])) for k in range(6)]},
              "pairs_per_run": r6(len(g.pairs) / g.R),
              "benchmarks_per_run": r6(np.mean([len({p["bench"] for p in m_}) for m_ in g.metas])),
              "share_alone": r6(np.mean(same_of(g) == 1)),
              "other_labels_b31": {"share_with_any": r6(np.mean([p["other"] > 0 for p in g.pairs])),
                                   "mean": r6(np.mean([p["other"] for p in g.pairs])),
                                   "own_mean": r6(np.mean([p["own"] for p in g.pairs]))},
              "targets": {"with_terms": int(sum(t["with_terms"] for t in tms)),
                          "of": int(sum(t["targets"] for t in tms)),
                          "twin_share": r6(sum(t["twin"] for t in tms)
                                           / max(1, sum(t["with_terms"] for t in tms))),
                          "layer_failures": int(sum(t["fails"] for t in tms)),
                          "hier_failures": int(sum(t["hier_failures"] for t in tms)),
                          "unconverged": int(sum(t["unconverged"] for t in tms))},
              "task_s": r6(np.mean([t["task_s"] for t in tms])),
              "in_sample_best": {"config": CAND[best],
                                 **fmt_stat(g.stat(g.dalc[:, best])),
                                 "budgets": budget_rows(g, g.D[:, best])},
              "default": {"config": CAND[dflt], **fmt_stat(g.stat(g.dalc[:, dflt])),
                          "budgets": budget_rows(g, g.D[:, dflt]),
                          "per_parent": {q: fmt_stat(g.stat(g.dalc[:, dflt], g.parent == q))
                                         for q in g.parents}},
              "top10": [{"config": CAND[c], "mean": r6(m[c]), "run_se": r6(se[c])} for c in top],
              "b0_max_abs": float(np.max(np.abs(g.D[:, :, 0]))),
              # one label per pair, centred within its pair: every residual is 0
              "b1_max_abs": float(np.max(np.abs(g.D[:, :, 1]))),
              "all": {CAND[c]: [r6(m[c]), r6(se[c])] for c in range(1, 1 + C)},
              "budgets_all": {CAND[c]: [r6(x) for x in bud[c]] for c in range(1, 1 + C)},
              "mean_abs_shift": {CAND[1 + c]: r6(np.mean([r["abs_shift"][c] for r in g.runs]))
                                 for c in range(C)},
              "moved_share": {CAND[1 + c]: r6(np.mean([r["moved"][c] for r in g.runs]))
                              for c in range(C)}}
        rs["in_sample_best"]["per_parent"] = {q: fmt_stat(g.stat(g.dalc[:, best], g.parent == q))
                                              for q in g.parents}
        for key, rep in (("nested_within", within), ("nested_from_primary", transfer),
                         ("nested_joint", jnt)):
            if rep is None:
                continue
            rs[key] = {k: v for k, v in rep.items() if not k.startswith("_")}
            rs[key]["oracle"] = oracle_rows(g, rep["_x"], rep["_b"], "layer")
        rs["oracle"] = oracle_rows(g, None, None, "layer")
        # marginal: the best config with each value of each hyperparameter (in sample)
        rs["marginal_best"] = {}
        for f, vals, pos in (("beta", BETAS, 0), ("tau", TAUS, 1), ("gamma", GAMMAS, 2),
                             ("other", OTHERS, 3), ("eb", EBS, 4)):
            rs["marginal_best"][f] = {}
            for v in vals:
                cs = [1 + c for c, cf in enumerate(CONFIGS) if cf[pos] == v]
                cb = cs[int(np.argmin(m[cs]))]
                rs["marginal_best"][f][f"{v:g}"] = [CAND[cb], r6(m[cb]), r6(se[cb])]
        summ["regimes"][name] = rs
        per_run[name] = [{"run": r["i"], "pairs": len(r["meta"]),
                          "base_alc": r6(float(np.mean(r["base"] @ W6))),
                          "nested_within": r6(float(np.mean(within["_x"][g.run == j]))),
                          "default": r6(float(np.mean(g.dalc[g.run == j, dflt]))),
                          "in_sample_best": r6(float(np.mean(g.dalc[g.run == j, best]))),
                          "pair_rate_oracle": r6(np.mean([p["level"] for p in r["meta"]
                                                          if p["level"] is not None]))
                          if any(p["level"] is not None for p in r["meta"]) else None,
                          "item_oracle": r6(np.mean([p["item"] for p in r["meta"]
                                                     if p["item"] is not None]))
                          if any(p["item"] is not None for p in r["meta"]) else None}
                         for j, r in enumerate(g.runs)]
    # what the procedure would ship: the joint worst-regime choice on every parent
    if joint:
        allc = crit(joint, None)
        c = int(np.argmin(allc))
        summ["ship_candidate"] = {"joint_all_parents": CAND[c],
                                  "joint_all_parents_worst_mean": r6(float(np.min(allc))),
                                  "primary_all_parents": CAND[int(np.argmin(
                                      crit([regs[PRIMARY]], None)))],
                                  "by_regime": {}}
        for name, g in regs.items():
            summ["ship_candidate"]["by_regime"][name] = {
                **fmt_stat(g.stat(g.dalc[:, c])), "budgets": budget_rows(g, g.D[:, c]),
                "per_parent": {q: fmt_stat(g.stat(g.dalc[:, c], g.parent == q))
                               for q in g.parents}}
    state["summary"] = summ
    state["per_run"] = per_run
    return summ


# --- verification and latency ------------------------------------------------------------

def verify_configs(state):
    """The library default, a heavy one (every record, slope on, sharpening
    1, tau 0.5, beta 4), and the configurations the summary chose."""
    picks = [NAMES[DEFAULT], cname((4.0, 0.5, 1.0, 1.0, 1.0)), cname((1.0, 1.0, 4.0, 0.5, 0.0))]
    s = state.get("summary", {})
    for rs in s.get("regimes", {}).values():
        for key in ("nested_within", "nested_joint"):
            picks += [v for v in rs.get(key, {}).get("choice", {}).values() if v != OFF]
        picks.append(rs.get("in_sample_best", {}).get("config"))
    if s.get("ship_candidate"):
        picks.append(s["ship_candidate"]["joint_all_parents"])
    return [p for p in dict.fromkeys(picks) if p and p != OFF]


def verify_task(task):
    """One run through official.run_official (deep copies, one worker): the
    base and ItemSig at each config, per pair Brier against the stored row,
    and the timing of each."""
    rows, regime, i, names = task
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = draw(regime, i)
        z = np.load(path_of(rows, regime, i))
        base, diff = z["base"], z["diff"].astype(np.float64)
        out = []
        fac = LC.factory([SHIP], run, [])
        for name in [OFF] + names:
            if name == OFF:
                f, want = fac, base
            else:
                c = NAMES.index(name)
                f, want = I.make_itemsig(fac, hyper_of(CONFIGS[c])), base + diff[:, c]
            t0 = time.perf_counter()
            res = O.run_official(run, f, deepcopy=True, workers=1)
            wall = time.perf_counter() - t0
            got = np.array([[r["brier"][b] for b in BUDGETS] for r in res["rows"]])
            tm = res["timing"]
            out.append({"regime": regime, "run": i, "config": name,
                        "max_abs_diff": float(np.max(np.abs(got - want))),
                        "eval_calls": tm["evaluation_calls"],
                        "eval_mean_ms": round(1e3 * tm["evaluation_mean_s"], 3),
                        "eval_max_s": round(tm["evaluation_max_s"], 4),
                        "setup_max_s": round(tm["setup_max_s"], 4), "wall_s": round(wall, 2)})
    return out


VERIFY_RUNS = [("tl", 0), ("tl", 1), ("tl mix/whole", 0), ("r1b", 0), ("r1p", 0)]


def stage_verify(args, state):
    names = verify_configs(state)
    todo = [(args.rows, r, i, names) for r, i in VERIFY_RUNS
            if os.path.exists(path_of(args.rows, r, i))]
    t0 = time.time()
    bundles = prepare()
    out = []
    with mp.get_context("fork").Pool(args.jobs, initializer=LC.init,
                                      initargs=(bundles, {})) as pool:
        for res in pool.imap_unordered(verify_task, todo):
            out += res
            print(f"  verified {res[0]['regime']} {res[0]['run']}: max diff "
                  f"{max(r['max_abs_diff'] for r in res):.2e}", flush=True)
    state["verify"] = {"configs": names, "rows": out,
                       "max_abs_diff": max(r["max_abs_diff"] for r in out),
                       "load_average": os.getloadavg()}
    state["passes"].append({"stage": "verify", "command": " ".join(sys.argv),
                            "wall_s": round(time.time() - t0, 1), "jobs": args.jobs,
                            **provenance()})


# --- files ---------------------------------------------------------------------------------

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, separators=(",", ":"))
    os.replace(tmp, path)


def provenance():
    files = ["paiec/itemsig.py", "paiec/hier.py", "paiec/prior.py", "paiec/predict.py",
             "paiec/official.py", "paiec/testlike.py", "experiments/level_calibration.py",
             "experiments/itemsig_eval.py", "submission/model.py"]
    dig = {}
    for f in files:
        with open(os.path.join(ROOT, f), "rb") as fh:
            dig[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "paiec"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:
        head, dirty = None, None
    return {"head": head, "paiec_status": dirty, "digests": dig, "load_average": os.getloadavg()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "oracle", "verify"])
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--only", default=None, help="regime:n,... in place of PLAN (run stage)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    state = load(args.out) or {"config": {"plan": PLAN, "boots": BOOTS, "ship": SHIP,
                                          "level": _LEVEL,
                                          "regimes": {k: list(LC.REGIMES[k]) for k in PLAN},
                                          "sighyper_fields": [f.name for f in fields(I.SigHyper)]},
                               "passes": []}
    if args.stage == "run":
        stage_run(args, state)
        save(args.out, state)
    elif args.stage == "oracle":
        stage_oracle(args, state)
        save(args.out, state)
    elif args.stage == "verify":
        stage_verify(args, state)
        save(args.out, state)
    if args.summarise:
        t0 = time.time()
        summarise(args, state)
        state["passes"].append({"stage": "summarise", "command": " ".join(sys.argv),
                                "wall_s": round(time.time() - t0, 1), **provenance()})
        save(args.out, state)
        s = state["summary"]
        for name, rs in s["regimes"].items():
            nw = rs["nested_within"]
            print(f"{name}: runs {rs['runs']} base {rs['base']['ALC']}; nested {nw['ALC']}; "
                  f"in-sample {rs['in_sample_best']['config']} {rs['in_sample_best']['mean']}; "
                  f"default {rs['default']['mean']}", flush=True)
    if not args.stage and not args.summarise:
        ap.error("--stage or --summarise")


if __name__ == "__main__":
    main()
