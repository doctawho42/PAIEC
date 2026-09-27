"""Subject side at budgets 0 and 1: harness identity, ordered reasoning effort,
the release-date form and a Student-t level, on the shipped hier.

The second formative run of the shipped hier paid 0.237 at B0 and 0.195 at B1
against about 0.18 from B7 on, and B0 and B1 carry 0.3 of ALC's weight. What a
model knows about a pair before and just after its first label is its level
prior and its subject prior. This script measures four changes to them, each
behind a flag, against exactly what ships: paiec.hier.HierPredictor with
submission/model.py's LEVEL (read from the file) over prior.build, every
target's prior and hyperparameters fitted without its parent benchmark
(experiments/level_calibration.py's harness, runs and statistics).

Candidates (COMPONENTS; a configuration is 'ship' or components joined by '+'):
  H       one attribute column per canonical harness string seen on at least 8
          training rows (paiec.subjects.Spec harnesses), on top of the present /
          absent flag, which stays and is what an unseen harness string gets.
          The literature's additive theta_LLM + theta_scaffold (Ge et al.)
  E       reasoning_effort as one ordered level (minimal < low < medium < high
          < xhigh) with a has-effort flag, instead of a dummy per level
  Dclip   the release date linear as now, held to the training rows' range
  Dhinge  linear plus a second slope from the training median date on
  Dhc     both
  Dlog    log of days since 2023 (a saturating trend)
  T       a Student-t level (nu 3) at LEVEL's centre and scale 2.5
  T1.8    the same at scale 1.8 (LEVEL's sigma_mu over prior.WIDEN's 1.44)
The design terms are paiec.subjects' optional Spec terms, threaded through
paiec.prior.build(design=...); the Student-t level is Hyper.nu_mu with
Flags.t_mixture on. Nothing in the library changes at the defaults.

Runs (PLAN): test-like runs (paiec.testlike Regime() defaults, seed 2, runs
0..299: the primary regime, the runs of "Item signal from the pair's own
labels"), groups merged at random and wholes (seed 3, 0..199), public R1
benchmark-first (seed 0, 0..149) and pair-uniform (0..99) as the guard, and
test-like runs without the date shift (seed 3, 0..99). The last regime is here
because the test-like regime's date shift (1.25 years, testlike's docstring) is
a synthetic knob that acts on a prior through its date term: any date form that
extrapolates less gains under the shift for that reason alone. The Student-t
configurations cost about nine times as much and score T_PLAN's runs only.

Selection and gates (GATE). Nested leave-one-parent-out over the four
multi-subject parents: for each held-out parent q, the configuration (or 'ship')
with the lowest mean ALC difference on the selection regimes' appearances whose
parent is not q (the worst of the two test-like regimes), scored on q's
appearances; a component is switched on in a fold if the fold's choice holds it.
A target's prediction depends on its own parent's prior only, so mixing choices
across parents is exactly what running them would give. The inner appearances
were scored with priors fitted without their own parent but with q, so the
selection is nested in the configuration, not in every coefficient (as in
experiments/itemsig_eval.py). A component ships only if it is switched on in at
least three of the four folds, the nested test-like difference is at most
-0.002 with no held-out parent above +0.002, and on both public weightings it
costs at most 0.001.

Statistics: paired differences against 'ship', "± run SE / cluster SE /
stratified SE" as in level_calibration (clusters (parent, subject), 2,000
resamples, a ratio estimator); the nested SEs redo the selection in every
resample of one bootstrap over the union of all regimes' clusters.

Run: python experiments/subject_side.py --stage verify
     python experiments/subject_side.py --stage diag
     python experiments/subject_side.py --stage run --jobs 2
     python experiments/subject_side.py --stage run --jobs 2 --configs COMBOS
     python experiments/subject_side.py --stage run --jobs 2 --configs T,T1.8
     python experiments/subject_side.py --summarise
Rows (per pair Brier by budget, per configuration) are appended to
data/subject_side_rows/<regime>.jsonl as tasks finish, so any stage resumes;
the summary goes to results/subject_side.json. docs/findings.md, "Subject side
at budgets 0 and 1", reads it.
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
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec import prior as PR  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.hier import HierPredictor  # noqa: E402
from paiec.subjects import attrs, days_since_2023, design_row  # noqa: E402

OUT = os.path.join(ROOT, "results", "subject_side.json")
ROWS = os.path.join(ROOT, "data", "subject_side_rows")
ITEMSIG = os.path.join(ROOT, "results", "itemsig_eval.json")
BOOTS = 2000
W6 = np.asarray(WEIGHTS, float)
SHIP = "ship"

#: regime -> runs 0..n-1 (level_calibration.REGIMES gives each its seed and knobs)
PLAN = {"tl": 300, "tl mix/whole": 200, "r1b": 150, "r1p": 100, "tl no shift": 100}
#: the Student-t level's runs (about nine times the Gaussian's cost; it does not
#: read dates, so the no-shift regime is left out)
T_PLAN = {"tl": 100, "tl mix/whole": 60, "r1b": 60, "r1p": 40, "tl no shift": 0}
PRIMARY = "tl"
SELECT = ("tl", "tl mix/whole")
GUARD = ("r1b", "r1p")
GATE = {"test_like": -0.002, "parent": 0.002, "guard": 0.001, "folds": 3}
#: the latest release date among the public pairs (matharena's DeepSeek V4 Pro)
LAST_PUBLIC = days_since_2023("2026-04-24")

COMPONENTS = {
    "H": {"design": {"harness_ids": True}},
    "E": {"design": {"effort_order": True}},
    "Dclip": {"design": {"date_form": "clip"}},
    "Dhinge": {"design": {"date_form": "hinge"}},
    "Dhc": {"design": {"date_form": "hinge_clip"}},
    "Dlog": {"design": {"date_form": "log"}},
    "T": {"nu": 3.0},
    "T1.8": {"nu": 3.0, "level": {"sigma_mu": 1.8}},
}
SINGLES = [SHIP, "H", "E", "Dclip", "Dhinge", "Dhc", "Dlog"]
#: combinations, scored after the singles: the ordered effort, the only single
#: that gained on test-like runs (-0.0010 on the first 121), with each date form
#: that did not lose there on average, and with the harness columns
COMBOS = ["E+Dclip", "E+Dlog", "E+Dhc", "H+E"]
TLEVEL = ["T", "T1.8"]


def shipped_level():
    """submission/model.py's LEVEL, read without importing the module (it
    imports the archive's paiec_rt, which exists only inside the zip)."""
    with open(os.path.join(ROOT, "submission", "model.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL"
                                                for t in node.targets):
            return {k: float(v) for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError("no LEVEL in submission/model.py")


LEVEL = shipped_level()


def parts(name):
    return [] if name == SHIP else name.split("+")


def spec_of(name):
    """(design, nu, level overrides) of a configuration."""
    design, nu, level = {}, 0.0, {}
    dates = [c for c in parts(name) if c.startswith("D")]
    if len(dates) > 1 or len(set(parts(name))) != len(parts(name)):
        raise ValueError(f"{name}: one date form and each component once")
    for c in parts(name):
        comp = COMPONENTS[c]
        design.update(comp.get("design", {}))
        nu = max(nu, comp.get("nu", 0.0))
        level.update(comp.get("level", {}))
    return design, nu, level


def dkey(design):
    return json.dumps(design, sort_keys=True)


# --- per-worker state (built in the parent, inherited by fork) ----------------------------

_BUNDLES = {}


def bundle(parent, design, nu):
    """(SubjectPrior, Hyper) on every public pair outside `parent`."""
    key = (parent, dkey(design), float(nu))
    if key not in _BUNDLES:
        _BUNDLES[key] = PR.build(LC.pairs(), (parent,), nu_mu=float(nu), design=design or None)
    return _BUNDLES[key]


def init_worker(path):
    """A spawned worker's priors: every bundle the parent built."""
    warnings.simplefilter("ignore")
    with open(path, "rb") as f:
        _BUNDLES.update(pickle.load(f))


def model(name, parent):
    design, nu, level = spec_of(name)
    prior, hyper = bundle(parent, design, nu)
    return HierPredictor(prior, replace(hyper, **{**LEVEL, **level}))


class Counts:
    """Failures and unconverged fits of a configuration's models, taken from
    each checkpoint's models when the next checkpoint replaces them, so no
    finished model (and its memoised fits and lines) stays alive."""

    def __init__(self):
        self.failures = self.unconverged = 0
        self.live = []

    def harvest(self):
        for m in self.live:
            self.failures += m.failures
            self.unconverged += m.unconverged
        self.live = []
        return self


def factory(name, ids, made):
    """() -> predict: one model per parent of the run (ids: anonymous
    benchmark_id -> parent, testlike.anon_parents), dispatched on the
    benchmark_id (level_calibration.factory's shape); made is a Counts."""
    def f():
        made.harvest()
        ms = {par: model(name, par) for par in set(ids.values())}
        made.live = list(ms.values())
        return lambda input, labeled=None: ms[ids[input[1]["benchmark_id"]]].predict(input, labeled)
    return f


def describe(regime, run):
    """level_calibration.describe plus what the candidates read: the visible
    subject's harness, effort and (shifted) release date."""
    out = LC.describe(regime, run)
    for d, e in zip(out, run):
        s = T._entry(e).subject
        a = attrs(s)
        d.update(harness=a["harness_id"], effort=None if a["effort_rank"] != a["effort_rank"]
                 else a["effort_rank"], days=None if a["days"] != a["days"] else a["days"])
    return out


def prepared(tasks):
    """The tasks with their runs drawn and played in the parent: (regime, i,
    names, slots, checkpoints, parents by anonymous id, meta), so a worker
    never loads the public pairs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for regime, i, names in tasks:
            run = LC.draw(regime, i)
            slots, cps = LC.checkpoints(run)
            yield regime, i, names, slots, cps, T.anon_parents(run), describe(regime, run)


def run_task(task):
    """A prepared task -> one JSONL row."""
    regime, i, names, slots, cps, ids, meta = task
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, timing = {}, {}
        for name in names:
            made = Counts()
            out, tm = LC.evaluate(slots, cps, [name], factory(name, ids, made))
            made.harvest()
            res[name] = {"b": [[round(x, 7) for x in r["brier"]] for r in out[name]],
                         "q0": [round(r["q0"], 6) for r in out[name]]}
            timing[name] = {"calls": tm["calls"], "mean_s": float(f"{tm['mean_s']:.4g}"),
                            "max_s": round(tm["max_s"], 4),
                            "failures": made.failures, "unconverged": made.unconverged}
    return {"regime": regime, "i": i, "meta": meta, "res": res, "timing": timing,
            "task_s": round(time.perf_counter() - t0, 2)}


# --- rows on disk ----------------------------------------------------------------------------

def rows_path(rows, regime):
    return os.path.join(rows, regime.replace(" ", "_").replace("/", "-") + ".jsonl")


def load_rows(rows, regime):
    """{run index: {'meta', 'res': {name: ...}, 'timing': {name: ...}}}, merging
    every line written for the run."""
    out = {}
    p = rows_path(rows, regime)
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            got = out.setdefault(r["i"], {"meta": r["meta"], "res": {}, "timing": {}})
            if got["meta"] != r["meta"]:
                raise ValueError(f"{regime} run {r['i']}: rows disagree on the run")
            got["res"].update(r["res"])
            got["timing"].update(r["timing"])
    return out


def plan_for(name):
    return T_PLAN if spec_of(name)[1] > 0 else PLAN


def stage_run(args, state):
    names = args.configs
    tasks = []
    for regime in PLAN:
        have = load_rows(args.rows, regime)
        for i in range(max(plan_for(n)[regime] for n in names)):
            need = tuple(n for n in names if i < plan_for(n)[regime]
                         and n not in have.get(i, {}).get("res", {}))
            # the Student-t configurations run one to a task: they are the long ones
            slow = tuple(n for n in need if spec_of(n)[1] > 0)
            fast = tuple(n for n in need if n not in slow)
            tasks += [(regime, i, (n,)) for n in slow]
            if fast:
                tasks.append((regime, i, fast))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} tasks over {names}", flush=True)
    if not tasks:
        return
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LC.pairs()
        LC.catalogue()
        for name in names:
            design, nu, _ = spec_of(name)
            for par in sorted({p.benchmark_id for p in LC.pairs()}):
                bundle(par, design, nu)
    print(f"bundles ready ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(args.rows, exist_ok=True)
    # the Gaussian configurations first, so their summary can be read while the
    # Student-t level's long tasks run
    order = sorted(tasks, key=lambda t: (max(spec_of(n)[1] for n in t[2]), t[0], t[1]))
    done, last = 0, time.time()
    files = {}
    # spawned, not forked: a forked worker shares the parent's public pairs, and
    # macOS counted them in every worker's footprint (1.3 to 1.9 GB each);
    # spawned workers read only the fitted priors, from a pickle
    bfile = os.path.join(args.rows, ".bundles.pkl")
    with open(bfile, "wb") as f:
        pickle.dump(_BUNDLES, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with mp.get_context("spawn").Pool(args.jobs, initializer=init_worker,
                                          initargs=(bfile,)) as pool:
            for row in pool.imap_unordered(run_task, prepared(order)):
                f = files.get(row["regime"])
                if f is None:
                    f = files[row["regime"]] = open(rows_path(args.rows, row["regime"]), "a")
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                f.flush()
                done += 1
                if time.time() - last > 60 or done == len(order):
                    last = time.time()
                    print(f"  {done}/{len(order)} tasks, {time.time() - t0:.0f}s", flush=True)
    finally:
        for f in files.values():
            f.close()
    state["passes"].append({"stage": "run", "configs": names, "command": " ".join(sys.argv),
                            "tasks": len(order), "jobs": args.jobs,
                            "wall_s": round(time.time() - t0, 1), **provenance()})


# --- verification ----------------------------------------------------------------------------

def stage_verify(args, state):
    """The harness against official.run_official (deep copies, one worker) pair
    for pair on one test-like and one public run, for 'ship' and two
    candidates; 'ship' against level_calibration's own SHIP-style model."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for regime, i in (("tl", 0), ("r1b", 0)):
            run = LC.draw(regime, i)
            slots, cps = LC.checkpoints(run)
            for name in (SHIP, "E+Dhc", "H"):
                ids = T.anon_parents(run)
                got, _ = LC.evaluate(slots, cps, [name], factory(name, ids, Counts()))
                ref = O.run_official(run, factory(name, ids, Counts()), deepcopy=True)
                d = max(abs(a - b) for r, g in zip(ref["rows"], got[name])
                        for a, b in zip([r["brier"][b] for b in BUDGETS], g["brier"]))
                out.append({"regime": regime, "run": i, "config": name, "max_abs_diff": d})
            lc = LC.gname(LEVEL["mu0"], LEVEL["sigma_mu"], LEVEL["attr_scale"])
            a, _ = LC.evaluate(slots, cps, [SHIP], factory(SHIP, T.anon_parents(run), Counts()))
            b, _ = LC.evaluate(slots, cps, [lc], LC.factory([lc], run, []))
            d = max(abs(x - y) for r, g in zip(a[SHIP], b[lc]) for x, y in zip(r["brier"], g["brier"]))
            out.append({"regime": regime, "run": i, "config": f"ship vs {lc}", "max_abs_diff": d})
    for r in out:
        print(r, flush=True)
    state["verify"] = out
    state["passes"].append({"stage": "verify", "command": " ".join(sys.argv), **provenance()})


# --- offline diagnostic ----------------------------------------------------------------------

def stage_diag(args, state):
    """How each design predicts public standings (prior.standings, the target
    of the attribute ridge): held out by benchmark (the transfer the hidden
    test needs) and held out by model name with every benchmark in training
    (an unseen model on seen benchmarks and harnesses, Ge et al.'s setting);
    and what the full fit extrapolates to future release dates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = PR.standings(LC.pairs(), ())
    bench = np.array([r[0] for r in rows])
    z = np.array([r[2] for r in rows])
    names = np.array([PR.canon_name(r[1]) or f"row {i}" for i, r in enumerate(rows)])
    X_of = {}

    def fitpred(tr, te, design):
        coef, spec, _, _ = PR._ridge([rows[i] for i in tr], 2.0, design or None)
        return np.array([design_row(attrs(rows[i][1]), spec) @ coef for i in te]), coef, spec

    designs = {n: spec_of(n)[0] for n in SINGLES + COMBOS}
    out = {"rows": len(rows), "lobo": {}, "leave_name_out": {}, "dates": {}}
    for n, des in designs.items():
        per = {}
        for b in sorted(set(bench)):
            te, tr = np.flatnonzero(bench == b), np.flatnonzero(bench != b)
            p, _, _ = fitpred(tr, te, des)
            p = p - p.mean()
            per[b] = {"mse": round(float(np.mean((z[te] - p) ** 2)), 4),
                      "var": round(float(np.mean(z[te] ** 2)), 4),
                      "r": round(float(np.corrcoef(p, z[te])[0, 1]), 4)}
        out["lobo"][n] = {"per": per, "mean_mse": round(float(np.mean([v["mse"] for v in per.values()])), 4)}
        err = np.zeros(len(rows))
        for k in sorted(set(names)):
            te, tr = np.flatnonzero(names == k), np.flatnonzero(names != k)
            p, _, _ = fitpred(tr, te, des)
            err[te] = z[te] - p
        inner = {}
        for q in sorted(set(bench)):
            # strictly nested: fitted without both the outer and the inner benchmark
            errs = []
            for q2 in sorted(set(bench) - {q}):
                te, tr = np.flatnonzero(bench == q2), np.flatnonzero((bench != q) & (bench != q2))
                p, _, _ = fitpred(tr, te, des)
                errs.append(float(np.mean((z[te] - (p - p.mean())) ** 2)))
            inner[q] = float(np.mean(errs))
        out["lobo"][n]["inner_without"] = {q: round(v, 4) for q, v in inner.items()}
        out["leave_name_out"][n] = {"mse": round(float(np.mean(err ** 2)), 4),
                                    **{b: round(float(np.mean(err[bench == b] ** 2)), 4)
                                       for b in sorted(set(bench))}}
        _, coef, spec = fitpred(np.arange(len(rows)), [], des)
        X_of[n] = (coef, spec)
    # strictly nested selection of the design on held-out standings: per outer
    # benchmark, the design with the lowest inner LOBO error without it
    out["nested_lobo"] = {}
    for q in sorted(set(bench)):
        best = min(designs, key=lambda n: out["lobo"][n]["inner_without"][q])
        out["nested_lobo"][q] = {"choice": best, "mse": out["lobo"][best]["per"][q]["mse"],
                                 "ship_mse": out["lobo"][SHIP]["per"][q]["mse"]}
    base = {"normalized_name": "x", "provider": "OpenAI", "release_date": ""}
    for n in ("ship", "Dclip", "Dhinge", "Dhc", "Dlog"):
        coef, spec = X_of[n]
        out["dates"][n] = {d: round(float(design_row(attrs({**base, "release_date": d}), spec) @ coef), 3)
                           for d in ("2024-06-01", "2025-01-01", "2025-06-01", "2026-01-01",
                                     "2026-04-01", "2026-09-01", "2027-06-01")}
    # the same leaving each benchmark out (what the runs' priors hold), and the
    # date coefficient: what a design says about a subject released later
    out["lobo_dates"] = {}
    for q in sorted(set(bench)):
        tr = np.flatnonzero(bench != q)
        row = {}
        for n in ("ship", "E", "Dclip", "Dlog", "E+Dclip", "E+Dlog"):
            _, coef, spec = fitpred(tr, [], designs[n])
            row[n] = {"days_coef": round(float(coef[1]), 3), **{
                d: round(float(design_row(attrs({**base, "release_date": d}), spec) @ coef), 3)
                for d in ("2025-06-01", "2026-04-01", "2027-06-01")}}
        out["lobo_dates"][q] = row
    coef, spec = X_of["H"]
    out["harness_coef"] = dict(zip(spec.harnesses, [round(float(c), 3) for c in coef[-len(spec.harnesses):]]))
    coef, spec = X_of["E"]
    out["effort_coef"] = {"has_effort": round(float(coef[-2]), 3), "rank": round(float(coef[-1]), 3)}
    coef, spec = X_of[SHIP]
    out["ship_effort_dummies"] = dict(zip(spec.efforts, [round(float(c), 3) for c in coef[-len(spec.efforts):]]))
    out["ship_harness_flag"] = round(float(coef[8]), 3)
    print(json.dumps(out, indent=1)[:3000], flush=True)
    state["diag"] = out
    state["passes"].append({"stage": "diag", "command": " ".join(sys.argv), **provenance()})


# --- statistics ------------------------------------------------------------------------------

class Rows:
    """One regime's pair appearances over the runs where every configuration
    in `names` was scored: D[N, C, 6], Brier differences against 'ship' (column
    0), with run, cluster and parent per appearance."""

    def __init__(self, rows, regime, names):
        got = load_rows(rows, regime)
        runs = [i for i in sorted(got) if all(n in got[i]["res"] for n in names)]
        if not runs:
            raise ValueError(f"no run of {regime} holds {names}")
        self.name, self.names, self.runs = regime, list(names), runs
        self.R = len(runs)
        self.metas = [got[i]["meta"] for i in runs]
        self.boot = LC.Boot(self.metas, BOOTS)
        self.cl = np.concatenate(self.boot.idx)
        self.K = len(self.boot.keys)
        self.cl_parent = np.array([k[0] for k in self.boot.keys])
        self.run = np.concatenate([[j] * len(m) for j, m in enumerate(self.metas)])
        self.w = np.concatenate([[1.0 / len(m)] * len(m) for m in self.metas])
        self.pairs = [p for m in self.metas for p in m]
        self.parent = np.array([p["parent"] for p in self.pairs])
        self.parents = sorted(set(self.parent.tolist()))
        B = {n: np.concatenate([np.asarray(got[i]["res"][n]["b"], float) for i in runs])
             for n in names}
        self.base = B[SHIP]
        self.D = np.stack([B[n] - self.base for n in names], 1)          # (N, C, 6)
        self.dalc = self.D @ W6
        self.q0 = {n: np.concatenate([got[i]["res"][n]["q0"] for i in runs]) for n in names}
        self.timing = {n: [got[i]["timing"][n] for i in runs] for n in names}
        self.n_k = np.bincount(self.cl, self.w, minlength=self.K)
        self.S = np.zeros((self.K, len(names)))
        np.add.at(self.S, self.cl, self.w[:, None] * self.dalc)

    def col(self, name):
        return self.names.index(name)

    def mean(self, x, sel=None):
        x = np.asarray(x, float)
        if sel is None:
            return float(np.sum(self.w * x) / self.R)
        return float(np.sum(self.w[sel] * x[sel]) / np.sum(self.w[sel]))

    def run_se(self, x):
        per = np.bincount(self.run, self.w * np.asarray(x, float), minlength=self.R)
        return float(per.std(ddof=1) / math.sqrt(self.R))

    def boot_se(self, x, sel=None):
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
        se = self.boot_se(x, sel)
        out = {"mean": r6(self.mean(x, sel)), "cluster_se": r6(se[0]), "strat_se": r6(se[1])}
        if sel is None:
            out["run_se"] = r6(self.run_se(x))
        else:
            out["n"] = int(np.sum(sel))
        return out


def r6(x):
    return None if x is None else round(float(x), 6)


class Union:
    """One cluster bootstrap over the union of the regimes' clusters, so a
    resample draws a (parent, subject) the same number of times everywhere
    (itemsig_eval.Union): W as one pool, Ws within each parent."""

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

    def parts(self, g, q, strat, inside=False):
        W = (self.Ws if strat else self.W)[g.name]
        m = (g.cl_parent == q) if inside else (g.cl_parent != q)
        Wm = W * m
        return Wm @ g.S, Wm @ g.n_k


def crit(regs, q, U=None, strat=False, idx=None):
    """Worst selection regime's mean ALC difference per configuration (the
    columns idx, default all) over the appearances whose parent is not q; per
    resample with U."""
    out = None
    for g in regs:
        m = g.cl_parent != q
        if not m.any():
            continue
        if U is None:
            c = g.S[m].sum(0) / g.n_k[m].sum()
        else:
            num, den = U.parts(g, q, strat)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(den[:, None] > 0, num / den[:, None], -np.inf)
        out = c if out is None else np.maximum(out, c)
    return out if idx is None else out[..., idx]


def nested(sel_regs, score, U=None, strat=False, idx=None):
    """Per parent q of the scored regime, the configuration with the lowest
    crit without q, scored on q: (choices, per-appearance values), or with U
    the bootstrap estimates with the selection redone in every resample. A
    parent no selection regime holds (swe_rebench on public runs) gets the
    choice made on every selection parent."""
    parents = sorted(set(score.cl_parent.tolist()))
    sel_parents = {p for g in sel_regs for p in g.parents}
    q_of = {q: (q if q in sel_parents else None) for q in parents}
    idx = np.arange(len(score.names)) if idx is None else np.asarray(idx)
    if U is None:
        choice = {q: int(idx[np.argmin(crit(sel_regs, q_of[q], idx=idx))]) for q in parents}
        x = score.dalc[np.arange(len(score.dalc)), [choice[p] for p in score.parent]]
        return choice, x
    Ws = (U.Ws if strat else U.W)[score.name]
    num = np.zeros(len(Ws))
    for q in parents:
        c = idx[np.argmin(crit(sel_regs, q_of[q], U, strat, idx), 1)]
        num += U.parts(score, q, strat, inside=True)[0][np.arange(len(Ws)), c]
    return num / (Ws @ score.n_k)


def budget_rows(g, per_b, sel=None):
    out = []
    for k, b in enumerate(BUDGETS):
        s = g.stat(per_b[:, k], sel)
        out.append({"budget": b, "diff": s["mean"], "cluster_se": s["cluster_se"],
                    "run_se": s.get("run_se"), "alc_part": r6(W6[k] * s["mean"])})
    return out


def acts_on(g, x):
    """Where a configuration's difference sits: pairs whose visible subject has
    a harness string, an effort level, no date, or a (shifted) date past the
    latest public release."""
    days = np.array([np.nan if p["days"] is None else p["days"] for p in g.pairs])
    groups = {"harness": np.array([bool(p["harness"]) for p in g.pairs]),
              "effort": np.array([p["effort"] is not None for p in g.pairs]),
              "undated": np.isnan(days),
              "future": np.nan_to_num(days, nan=-1.0) > LAST_PUBLIC}
    return {k: {"share": r6(np.mean(v)), **({"diff": g.stat(x, v)} if v.any() else {})}
            for k, v in groups.items()}


def config_report(g, name):
    c = g.col(name)
    x = g.dalc[:, c]
    out = {"dalc": g.stat(x), "budgets": budget_rows(g, g.D[:, c]),
           "per_parent": {q: g.stat(x, g.parent == q) for q in g.parents},
           "q0": r6(g.mean(g.q0[name])), "acts_on": acts_on(g, x)}
    tm = g.timing[name]
    calls = sum(t["calls"] for t in tm)
    out["latency"] = {"mean_ms": r6(1e3 * sum(t["mean_s"] * t["calls"] for t in tm) / calls),
                      "max_s": max(t["max_s"] for t in tm),
                      "failures": sum(t["failures"] for t in tm),
                      "unconverged": sum(t["unconverged"] for t in tm)}
    return out


def nested_report(sel_regs, regs, U, names):
    """The nested selection on sel_regs, scored on every regime of regs."""
    out = {"select_on": [g.name for g in sel_regs], "candidates": names}
    for g in regs:
        idx = [g.col(n) for n in names]
        choice, x = nested(sel_regs, g, idx=idx)
        rep = {"choice": {q: g.names[c] for q, c in choice.items()},
               "ALC": {"mean": r6(g.mean(x)), "run_se_fixed": r6(g.run_se(x))}}
        for strat in (False, True):
            est = nested(sel_regs, g, U, strat, idx)
            rep["ALC"]["strat_se" if strat else "cluster_se"] = r6(est.std(ddof=1))
        cols = np.array([choice[p] for p in g.parent])
        per_b = g.D[np.arange(len(cols)), cols]
        rep["budgets"] = budget_rows(g, per_b)
        rep["per_parent"] = {q: {"choice": g.names[choice[q]], **g.stat(x, g.parent == q)}
                             for q in g.parents}
        out[g.name] = rep
    folds = out[sel_regs[0].name]["choice"]
    out["folds_on"] = {c: sum(c in parts(v) for v in folds.values())
                       for c in sorted({c for n in names for c in parts(n)})}
    return out


def gate(nest, names):
    """Each component against GATE on a nested report."""
    out = {}
    prim = nest[PRIMARY]
    for c, on in nest["folds_on"].items():
        worst_parent = max(v["mean"] for v in prim["per_parent"].values())
        reasons = []
        if on < GATE["folds"]:
            reasons.append(f"switched on in {on} of {len(prim['per_parent'])} folds")
        if prim["ALC"]["mean"] > GATE["test_like"]:
            reasons.append(f"nested test-like {prim['ALC']['mean']:+.5f} > {GATE['test_like']}")
        if worst_parent > GATE["parent"]:
            reasons.append(f"a held-out parent at {worst_parent:+.5f}")
        for r in GUARD:
            if r in nest and nest[r]["ALC"]["mean"] > GATE["guard"]:
                reasons.append(f"{r} {nest[r]['ALC']['mean']:+.5f}")
        out[c] = {"folds_on": on, "ship": not reasons, "reasons": reasons}
    return out


def present(rows, names):
    """{regime: Rows} for the regimes where some run holds every one of names."""
    out = {}
    for r in PLAN:
        try:
            out[r] = Rows(rows, r, names)
        except ValueError:
            pass
    return out


def summarise(args, state):
    t0 = time.time()
    all_names = [n for n in SINGLES + COMBOS + TLEVEL]
    have = {r: load_rows(args.rows, r) for r in PLAN}
    scored = [n for n in all_names if any(n in v["res"] for h in have.values() for v in h.values())]
    gauss = [n for n in scored if spec_of(n)[1] == 0]
    tlev = [n for n in scored if spec_of(n)[1] > 0]
    s = {"level": LEVEL, "gate": GATE, "plan": PLAN, "t_plan": T_PLAN,
         "configs": {n: dict(zip(("design", "nu", "level"), spec_of(n))) for n in scored},
         "verify": state.get("verify"), "diag": state.get("diag"), "regimes": {}}
    regs = present(args.rows, gauss)
    for r, g in regs.items():
        s["regimes"][r] = {"runs": g.R, "appearances": len(g.pairs), "ship": {
            "ALC": r6(g.mean(g.base @ W6)), "ALC_run_se": r6(g.run_se(g.base @ W6)),
            "budgets": [r6(g.mean(g.base[:, k])) for k in range(6)], "q0": r6(g.mean(g.q0[SHIP]))},
            "configs": {n: config_report(g, n) for n in gauss if n != SHIP}}
        s["regimes"][r]["ship_latency"] = config_report(g, SHIP)["latency"]
    # cross-check: 'ship' is itemsig_eval's base on the same test-like runs
    if os.path.exists(ITEMSIG):
        with open(ITEMSIG) as f:
            ib = json.load(f)["summary"]["regimes"]
        s["itemsig_base"] = {r: {"itemsig": ib[r]["base"]["ALC"], "runs": ib[r]["runs"]}
                             for r in ("tl", "tl mix/whole") if r in ib}
    sel = [regs[r] for r in SELECT if r in regs]
    U = Union(list(regs.values()))
    s["nested"] = {"primary": nested_report([regs[PRIMARY]], list(regs.values()), U, gauss),
                   "joint": nested_report(sel, list(regs.values()), U, gauss)}
    singles = [n for n in gauss if len(parts(n)) <= 1]
    s["nested"]["singles_joint"] = nested_report(
        sel, list(regs.values()), U, singles) if len(singles) < len(gauss) else None
    s["gates"] = {k: gate(v, gauss) for k, v in s["nested"].items() if v}
    # each configuration on its own: nested selection between it and 'ship'
    s["per_config"] = {}
    for n in gauss:
        if n == SHIP:
            continue
        rep = nested_report(sel, list(regs.values()), U, [SHIP, n])
        prim = rep[PRIMARY]
        s["per_config"][n] = {
            "folds_on": sum(v == n for v in prim["choice"].values()), "choice": prim["choice"],
            "ALC": {r: rep[r]["ALC"] for r in regs},
            "per_parent": {q: v["mean"] for q, v in prim["per_parent"].items()},
            "budgets": [b["diff"] for b in prim["budgets"]],
            "gate": gate(rep, [SHIP, n])}
    if tlev:
        names = [SHIP] + tlev
        tregs = present(args.rows, names)
        TU = Union(list(tregs.values()))
        s["student_t"] = {"runs": {r: g.R for r, g in tregs.items()}, "regimes": {
            r: {"ship_ALC": r6(g.mean(g.base @ W6)),
                "configs": {n: config_report(g, n) for n in tlev},
                "ship_latency": config_report(g, SHIP)["latency"]} for r, g in tregs.items()}}
        if all(r in tregs for r in SELECT):
            s["student_t"]["nested_joint"] = nested_report([tregs[r] for r in SELECT],
                                                           list(tregs.values()), TU, names)
            s["gates"]["student_t"] = gate(s["student_t"]["nested_joint"], names)
    s["summary_s"] = round(time.time() - t0, 1)
    state["summary"] = s
    state["passes"].append({"stage": "summarise", "command": " ".join(sys.argv),
                            "wall_s": s["summary_s"], **provenance()})


# --- bookkeeping -----------------------------------------------------------------------------

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"passes": []}


def save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, separators=(",", ":"))
    os.replace(tmp, path)


def provenance():
    files = ["paiec/hier.py", "paiec/prior.py", "paiec/subjects.py", "paiec/official.py",
             "paiec/testlike.py", "experiments/level_calibration.py", "experiments/subject_side.py"]
    dig = {}
    for f in files:
        with open(os.path.join(ROOT, f), "rb") as fh:
            dig[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "paiec"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        head, dirty = None, None
    return {"head": head, "paiec_dirty": dirty, "digests": dig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "verify", "diag"])
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--configs", default="SINGLES",
                    help="SINGLES, COMBOS, TLEVEL or a comma-separated list of configurations")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    named = {"SINGLES": SINGLES, "COMBOS": COMBOS, "TLEVEL": TLEVEL}
    args.configs = named.get(args.configs) or [c.strip() for c in args.configs.split(",") if c.strip()]
    if SHIP not in args.configs and args.stage == "run":
        args.configs = [SHIP] + args.configs
    for n in args.configs:
        spec_of(n)
    state = load(args.out)
    if args.summarise:
        summarise(args, state)
    elif args.stage == "run":
        stage_run(args, state)
    elif args.stage == "verify":
        stage_verify(args, state)
    elif args.stage == "diag":
        stage_diag(args, state)
    else:
        ap.error("--stage or --summarise")
    save(args.out, state)


if __name__ == "__main__":
    main()
