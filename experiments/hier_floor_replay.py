"""The floored-fit fix in paiec/hier.py, replayed against the solver before it
(docs/findings.md, "The multiple-choice floor, corrected" -> "Floored fits").

hier puts a multiple-choice floor c in the likelihood: a floored success has
likelihood c + (1 - c) s, whose log is convex where s is small against c. From
a prior mean well below a pair's successes Newton meets an indefinite matrix.
paiec/hier.py at PRE_FIX (f7e7d87) shifted it just past its most negative
eigenvalue, which left it nearly singular: the step ran to 1e7 or more, twenty
halvings found no ascent, and the fit stopped where it stood (often the prior
mean), every floored success read as a guess. The working tree's
Problem.solve runs the same Newton loop (Problem._newton) and, only where it
stops short on a posterior that holds a floored success, Problem._settle
(trust-region Newton from where Newton stopped and from the modes with every
floor removed and with every floored success dropped).

The PRE-FIX solver is paiec/hier.py at PRE_FIX, read with `git show` and
executed as the module paiec._hier_prefix; nothing is copied to disk. Its
relative imports resolve to the working tree's paiec.mcq, paiec.predict and
paiec.prior, which PRE_FIX and the fix share (the fix touched paiec/hier.py
only). The FIXED solver is the working tree's paiec.hier.

Stages
  compare   replay the shipped hier (experiments/harness.py ship_name(): the
            level_calibration bundle fitted leaving the target's parent out,
            submission/model.py's LEVEL over it, fresh instances per
            checkpoint, split scope 'pair') on level_calibration's checkpoints
            of results/mcq_floor.json's regimes: public R1 pair-uniform (r1p)
            and benchmark-first (r1b), test-like seed 2 (tl). Rows (ROWS): each
            regime under the corrected floor (paiec.mcq as it is, 'lib') at the
            shipped guess 0.5, r1p and r1b also under the hard floor (guess 1),
            runs 0-199; and each under the floor of bd0be67 ('old',
            experiments/mcq_floor.py's old_floor_of, the floor
            data/harness_rows_legacy was collected with), runs 0-99.
            A run none of whose pairs' first 31 labels is floored is skipped:
            without a floored label no posterior holds a floored success, so
            the two solvers are the same computation.
            The fixed solver runs on every other run, instrumented: every solve
            of a posterior that holds a floored success is recorded (where
            Newton stopped and where _settle took it), and on every one where
            Newton converged, the two other starts _settle would have tried are
            run from scratch and compared with Newton's mode (recorded only,
            the prediction is Newton's; --no-probe skips this).
            The pre-fix solver runs on runs < --verify (30) of every row, and
            on every run where the fixed solver's Newton stopped short
            (--both: on every run). On the others its predictions are the fixed
            solver's by construction: _settle never ran there and _newton is
            PRE_FIX's loop, which the verified runs check bit for bit.
            Where experiments/harness.py's stored rows hold the run,
            predictions are also compared with them: the fixed solver under
            'lib' at the shipped guess with data/harness_rows (collected with
            this library), both solvers under 'old' with
            data/harness_rows_legacy (collected with PRE_FIX's solver and the
            old floor). One JSON per run in data/hier_floor/<row>/,
            resumable; --shard k/n splits the runs over processes.
  timing    latency as the platform pays it: per checkpoint fresh instances
            of both solvers, every input predicted, the order alternating,
            --reps passes, on runs --runs (0:40) of r1p and r1b under the
            corrected floor; uninstrumented. The offline bundles (built once
            per parent and cached) are built before the clock starts, else the
            first solver to meet a parent would pay for it
  synthetic the collapse on toy problems: one pair on four-option items
            (tests/test_hier.py's test_a_floored_fit_far_below_its_successes_
            converges) at levels -4, -2.5 and -6, guess 0.5 and 1, against the
            exact posterior predictive; and seven subjects on three benchmarks
            (tests/test_hier.py's _floored_run) over seeds 0-7 at levels
            -1.263 and -2.5, guess 0.5 and 1: how often Newton stops short
  modes     the search for a second mode of a floored posterior: 2,520 one-pair
            problems (levels -6 to 6, sd 1 to 5, guess 0.5 and 1, 0 to 12
            successes of 15 or 31 on three- and five-option items, with and
            without two other subjects on unfloored items), every floored fit
            probed as in compare
  hidden    hidden-test-like synthetic runs through the SHIPPED bundle
            (submission/prior.json, LEVEL baked in, as tools/build_submission.py
            writes it): 7 benchmarks per scenario (four five-option, one
            four-option, two without options), 2025-26 subjects, records near
            31/31; each scenario at B1, B3, B7, B15 and B31, one fit set per
            budget, both solvers. Variants: 'real' (about 1.3 pairs per
            benchmark, real subject attributes), 'blank' (the same, attributes
            blanked), 'crowdreal' (4 pairs per benchmark with rates 1, 0.2,
            0.05 and 0.5, real attributes) and 'crowd' (the same, blanked)
  summary   -> results/hier_floor.json
  show      markdown tables of the summary

Run (compare with --both on two processes 34 minutes, the two timings side
by side about 13, synthetic 1, modes 1, hidden 10, the last three measured
beside other jobs; a fresh process holds about 0.9 GB and its heap fragments
over hundreds of fits, hence --rss-cap and the restart loop, which the
documented run never needed):
  for k in 0 1; do
    (until python experiments/hier_floor_replay.py --stage compare --both --shard $k/2 --rss-cap 1300;
     do [ $? -eq 3 ] || break; done) &
  done; wait
  python experiments/hier_floor_replay.py --stage timing --regimes r1p &
  python experiments/hier_floor_replay.py --stage timing --regimes r1b
  python experiments/hier_floor_replay.py --stage synthetic
  python experiments/hier_floor_replay.py --stage modes
  python experiments/hier_floor_replay.py --stage hidden
  python experiments/hier_floor_replay.py --stage summary
  python experiments/hier_floor_replay.py --stage show
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402
import paiec.hier as HN  # noqa: E402

#: the commit whose paiec/hier.py is the solver before the fix
PRE_FIX = "f7e7d87"
OUT = os.path.join(ROOT, "results", "hier_floor.json")
DATA = os.path.join(ROOT, "data", "hier_floor")
LEGACY_ROWS = os.path.join(ROOT, "data", "harness_rows_legacy")
W6 = H.W6
#: row -> (regime, floor, Hyper.guess or None for the shipped one, runs 0..n-1)
ROWS = {
    "r1p_lib_ship": ("r1p", "lib", None, 200),
    "r1b_lib_ship": ("r1b", "lib", None, 200),
    "tl_lib_ship": ("tl", "lib", None, 200),
    "r1p_lib_g1": ("r1p", "lib", 1.0, 200),
    "r1b_lib_g1": ("r1b", "lib", 1.0, 200),
    "r1p_old_ship": ("r1p", "old", None, 100),
    "r1b_old_ship": ("r1b", "old", None, 100),
    "tl_old_ship": ("tl", "old", None, 100),
}
REGIME_NAMES = {"r1p": "public R1, pair-uniform", "r1b": "public R1, benchmark-first",
                "tl": "test-like (seed 2)"}
HIDDEN_VARIANTS = {"real": 60, "blank": 40, "crowdreal": 25, "crowd": 25}


# --- the two solvers ------------------------------------------------------------------

_PRE = None


def prefix():
    """paiec/hier.py at PRE_FIX as the module paiec._hier_prefix."""
    global _PRE
    if _PRE is None:
        try:
            src = subprocess.run(["git", "show", f"{PRE_FIX}:paiec/hier.py"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"cannot read paiec/hier.py at {PRE_FIX} with git show: {e}") from e
        name = "paiec._hier_prefix"
        mod = types.ModuleType(name)
        mod.__file__ = f"git:{PRE_FIX}:paiec/hier.py"
        mod.__package__ = "paiec"
        sys.modules[name] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        if hasattr(mod.Problem, "_settle") or not hasattr(HN.Problem, "_settle"):
            raise RuntimeError("expected the solver before the fix at PRE_FIX and the fix in the working tree")
        mod.SOURCE_DIGEST = hashlib.sha256(src.encode()).hexdigest()[:16]
        _PRE = mod
    return _PRE


def file_digest(path):
    with open(os.path.join(ROOT, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


class _Rec:
    """What the instrumented fixed solver saw."""
    def __init__(self):
        self.probe = True
        self.depth = 0
        self.budget = None
        self.fits = []
        self.concave_unconverged = 0


REC = _Rec()
_SOLVE, _SETTLE = HN.Problem.solve, HN.Problem._settle


def _settle(self, st, P, tol):
    self._hf_newton = (float(st.lp), st.x.copy(), float(self.decrement))
    return _SETTLE(self, st, P, tol)


def _probe(self, st, tol):
    """The other two starts _settle would try, from scratch, on a floored fit
    where Newton converged: each refined by _trust and compared with Newton's
    mode. Recorded only; the fit keeps Newton's mode."""
    keep = self.converged, self.decrement, self.start
    out = []
    REC.depth += 1
    try:
        for x, name in self._other_starts():
            try:
                s, _, conv, _ = self._trust(self.state(*x), tol)
                out.append(dict(start=name, conv=bool(conv), gain=float(s.lp - st.lp),
                                dx=float(np.max(np.abs(s.x - st.x))) if len(st.x) else 0.0))
            except Exception as e:     # a failed start is no candidate, as in _settle
                out.append(dict(start=name, error=type(e).__name__))
    finally:
        REC.depth -= 1
        self.converged, self.decrement, self.start = keep
    return out


def _solve(self, x0=None, e0=None, max_iter=100, tol=1e-14):
    if REC.depth:
        return _SOLVE(self, x0, e0, max_iter, tol)
    REC.depth += 1
    self._hf_newton = None
    t0 = time.perf_counter()
    try:
        st, P = _SOLVE(self, x0, e0, max_iter, tol)
    finally:
        REC.depth -= 1
    secs = time.perf_counter() - t0
    if not self.nonconcave:
        REC.concave_unconverged += not self.converged
        return st, P
    f = dict(budget=REC.budget, p=int(self.p), n=int(len(self.y)),
             floored_successes=int(np.sum((self.c > 0) & (self.y > 0.5))), cold=x0 is None,
             newton_converged=self._hf_newton is None, converged=bool(self.converged),
             decrement=float(self.decrement), start=self.start, secs=round(secs, 5))
    if self._hf_newton is not None:
        lp0, x_0, dec0 = self._hf_newton
        f.update(newton_decrement=dec0, lp_gain=float(st.lp - lp0),
                 dx=float(np.max(np.abs(st.x - x_0))) if len(st.x) else 0.0)
    elif REC.probe and self.converged:
        f["probe"] = _probe(self, st, tol)
    REC.fits.append(f)
    return st, P


@contextlib.contextmanager
def recording(probe=True):
    """The fixed solver instrumented (Problem.solve and _settle wrapped) for
    the duration; outside it paiec.hier runs as shipped."""
    REC.probe, REC.fits, REC.concave_unconverged, REC.depth = probe, [], 0, 0
    HN.Problem.solve, HN.Problem._settle = _solve, _settle
    try:
        yield REC
    finally:
        HN.Problem.solve, HN.Problem._settle = _SOLVE, _SETTLE


@contextlib.contextmanager
def floor_patch(fn, mods):
    """Both modules read `floor_of` from their own namespace (hier imported the
    name); the memo of floors is per instance."""
    was = [m.floor_of for m in mods]
    try:
        if fn is not None:
            for m in mods:
                m.floor_of = fn
        yield
    finally:
        for m, w in zip(mods, was):
            m.floor_of = w


# --- compare ------------------------------------------------------------------------

def _mf():
    from experiments import mcq_floor as MF
    return MF


def make(mod, par, ship, guess):
    LC = H._lc()
    nu, ov, eb = LC.parse(ship)
    if eb:
        raise RuntimeError(f"{ship}: an empirical-Bayes level is not the shipped hier")
    prior, hyper = LC.bundle(par, nu)
    if ov is not None:
        hyper = replace(hyper, **ov)
    if guess is not None:
        hyper = replace(hyper, guess=float(guess))
    return mod.HierPredictor(prior, hyper)


def replay(mod, cps, pars, ids, ship, guess):
    """Every checkpoint, fresh instances per parent -> (6 x responses, stats)."""
    preds, unconv, fails, secs = [], 0, 0, 0.0
    for b, labeled, inputs, index in cps:
        REC.budget = b
        ms = {par: make(mod, par, ship, guess) for par in pars}
        t0 = time.perf_counter()
        P = np.array([ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled) for inp in inputs], float)
        secs += time.perf_counter() - t0
        if not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(P > 1):
            raise ValueError("invalid prediction")
        preds.append(P[index])
        unconv += sum(m.unconverged for m in ms.values())
        fails += sum(m.failures for m in ms.values())
    return np.vstack(preds), dict(unconverged=int(unconv), failures=int(fails), predict_secs=round(secs, 4))


def floored_labels(slots, floor_fn):
    from paiec.hier import mcq_text
    return int(sum(floor_fn(mcq_text(e[0][1]["item_content"])) > 0
                   for s in slots for e in s.acquired[:31]))


def harness_check(rows_dir, regime, i, ship, n_slots, P):
    """Max |P - stored ev_p| over the stored run's evaluated items, or None."""
    hp = os.path.join(rows_dir, regime, f"{i}.pkl")
    if not os.path.exists(hp):
        return None
    with open(hp, "rb") as f:
        hr = pickle.load(f)
    if hr.get("ship") != ship or len(hr["slots"]) != n_slots:
        return None
    d = 0.0
    for pos, r in zip(_mf()._harness_first_positions(hr["slots"]), hr["slots"]):
        d = max(d, float(np.nanmax(np.abs(P[:, pos] - np.asarray(r["ev_p"])))))
    return dict(max_abs=d, stored_lib_digest=hr.get("lib_digest"))


def run_one(name, i, args):
    from paiec import official as O
    from paiec import testlike as T
    regime, floor, guess, _ = ROWS[name]
    LC = H._lc()
    HB = prefix()
    MF = _mf()
    t0 = time.perf_counter()
    ship = H.ship_name()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = LC.draw(H.REGIMES[regime], i)
        slots, cps = LC.checkpoints(run)
    ids = T.anon_parents(run)
    pars = sorted(set(ids.values()))
    floor_fn = MF.old_floor_of if floor == "old" else None
    nfl = floored_labels(slots, floor_fn or HN.floor_of)
    n_resp = sum(len(s.targets) for s in slots)
    for par in pars:        # the offline bundles (paiec.prior fits) before any solve is recorded
        LC.bundle(par, LC.parse(ship)[0])
    row = dict(row=name, regime=regime, floor=floor, guess=guess, run=i, ship=ship, floored_labels=nfl,
               responses=n_resp, pairs=len(slots))
    if nfl == 0:
        row.update(skipped=True, secs=round(time.perf_counter() - t0, 2))
        return row
    with floor_patch(floor_fn, (HN, HB)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with recording(probe=args.probe) as rec:
            Pn, st_n = replay(HN, cps, pars, ids, ship, guess)
            fits, concave_unconv = list(rec.fits), rec.concave_unconverged
        newton_failures = sum(not f["newton_converged"] for f in fits)
        base_replayed = bool(args.both or i < args.verify or newton_failures)
        if base_replayed:
            Pb, st_b = replay(HB, cps, pars, ids, ship, guess)
        else:
            Pb, st_b = Pn, dict(unconverged=int(concave_unconv), failures=st_n["failures"], predict_secs=None)
    y = np.array([t[1] for s in slots for t in s.targets], float)
    bounds = np.cumsum([0] + [len(s.targets) for s in slots])
    diff = np.abs(Pn - Pb)
    changed = Pn != Pb
    apps = []
    for k, (entry, s) in enumerate(zip(run, slots)):
        pair, _ = O._entry(entry)
        sl = slice(bounds[k], bounds[k + 1])
        labels = [int(e[1]) for e in s.acquired[:31]]
        pool = [int(c[2]) for c in s.candidates]
        apps.append(dict(name=s.name, parent=T.parent_of(s.name), subject=str(pair.subject_id),
                         responses=int(bounds[k + 1] - bounds[k]), rate=float(y[sl].mean()),
                         label_rate=float(np.mean(labels)) if labels else None, labels=len(labels),
                         pool_rate=float(np.mean(pool)) if pool else None, pool=len(pool),
                         brier_base=[float(np.mean((Pb[b, sl] - y[sl]) ** 2)) for b in range(6)],
                         brier_new=[float(np.mean((Pn[b, sl] - y[sl]) ** 2)) for b in range(6)],
                         mean_base=[float(Pb[b, sl].mean()) for b in range(6)],
                         mean_new=[float(Pn[b, sl].mean()) for b in range(6)],
                         changed=[int(changed[b, sl].sum()) for b in range(6)],
                         maxdiff=[float(diff[b, sl].max(initial=0.0)) for b in range(6)]))
    checks = {}
    if floor == "lib" and guess is None:
        checks["fixed_vs_harness_rows"] = harness_check(H.ROWS, regime, i, ship, len(slots), Pn)
    elif floor == "old":
        checks["fixed_vs_legacy_rows"] = harness_check(LEGACY_ROWS, regime, i, ship, len(slots), Pn)
        if base_replayed:
            checks["prefix_vs_legacy_rows"] = harness_check(LEGACY_ROWS, regime, i, ship, len(slots), Pb)
    row.update(skipped=False, base_replayed=base_replayed, newton_failures=int(newton_failures),
               concave_unconverged=int(concave_unconv), unconverged=dict(base=st_b["unconverged"],
                                                                         new=st_n["unconverged"]),
               failures=dict(base=st_b["failures"], new=st_n["failures"]),
               predict_secs=dict(base=st_b["predict_secs"], new=st_n["predict_secs"]),
               changed=int(changed.sum()), maxdiff=float(diff.max()), apps=apps, fits=fits, checks=checks,
               maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                               / (2 ** 20 if sys.platform == "darwin" else 2 ** 10), 1),
               secs=round(time.perf_counter() - t0, 2))
    return row


def parse_runs(spec, n):
    if not spec:
        return 0, n
    a, b = spec.split(":")
    return int(a), min(int(b), n) if b else n


def current_rss_mb():
    """This process's resident memory now, in MB (ps; ru_maxrss is only the peak)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
        return int(out.stdout.split()[0]) / 1024
    except (OSError, ValueError, IndexError):
        return 0.0


def stage_compare(args):
    names = args.rows or list(ROWS)
    jobs = []
    for name in names:
        a, b = parse_runs(args.runs, ROWS[name][3])
        jobs += [(name, i) for i in range(a, b)]
    k, n = (int(v) for v in args.shard.split("/"))
    jobs = jobs[k::n]
    prefix()
    t0 = time.time()
    for name, i in jobs:
        d = os.path.join(args.data, name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{i}.json")
        if os.path.exists(path) and not args.redo:
            continue
        r = run_one(name, i, args)
        with open(path + ".tmp", "w") as f:
            json.dump(r, f)
        os.replace(path + ".tmp", path)
        if r["skipped"]:
            print(f"{name} {i}: no floored label ({time.time() - t0:.0f}s)", flush=True)
            continue
        starts = Counter(f["start"] for f in r["fits"] if not f["newton_converged"])
        print(f"{name} {i}: floored fits {len(r['fits'])}, Newton short {r['newton_failures']}, "
              f"unconverged {r['unconverged']}, base {'replayed' if r['base_replayed'] else 'copied'}, "
              f"changed {r['changed']}/{6 * r['responses']} max {r['maxdiff']:.3g}, rescued by {dict(starts)}, "
              f"checks {r['checks']}, {r['secs']:.1f}s, rss {r['maxrss_mb']} MB ({time.time() - t0:.0f}s)",
              flush=True)
        gc.collect()
        if args.rss_cap and current_rss_mb() > args.rss_cap:
            # the heap fragments over hundreds of fits; a fresh process resumes where this one stopped
            print(f"resident memory {current_rss_mb():.0f} MB above --rss-cap {args.rss_cap}: exiting with 3 "
                  f"to be restarted (resumable)", flush=True)
            sys.exit(3)


# --- timing -------------------------------------------------------------------------

def stage_timing(args):
    from paiec import testlike as T
    LC = H._lc()
    HB = prefix()
    ship = H.ship_name()
    a, b = parse_runs(args.runs or "0:40", 10 ** 6)
    os.makedirs(os.path.join(args.data, "timing"), exist_ok=True)
    for regime in args.regimes:
        path = os.path.join(args.data, "timing", f"{regime}.json")
        out = {"regime": regime, "runs_spec": [a, b], "reps": args.reps, "runs": []}
        for i in range(a, b):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run = LC.draw(H.REGIMES[regime], i)
                slots, cps = LC.checkpoints(run)
                ids = T.anon_parents(run)
                pars = sorted(set(ids.values()))
                for par in pars:    # the offline bundles, built once per parent, outside the timing
                    LC.bundle(par, LC.parse(ship)[0])
                secs = {"base": 0.0, "new": 0.0}
                calls, same, unconv = 0, True, {"base": 0, "new": 0}
                for rep in range(args.reps):
                    for bi, (bud, labeled, inputs, index) in enumerate(cps):
                        order = ("base", "new") if (bi + rep + i) % 2 == 0 else ("new", "base")
                        got = {}
                        for impl in order:
                            mod = HB if impl == "base" else HN
                            t0 = time.perf_counter()
                            ms = {par: make(mod, par, ship, None) for par in pars}
                            got[impl] = [ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled) for inp in inputs]
                            secs[impl] += time.perf_counter() - t0
                            if rep == 0:
                                unconv[impl] += sum(m.unconverged for m in ms.values())
                        if rep == 0:
                            calls += len(inputs)
                        same &= got["base"] == got["new"]
            out["runs"].append(dict(run=i, calls=calls, secs=secs, same=bool(same), unconverged=unconv))
            print(f"{regime} {i}: base {secs['base']:.2f}s new {secs['new']:.2f}s "
                  f"({100 * (secs['new'] / secs['base'] - 1):+.1f}%) same {same} unconverged {unconv}", flush=True)
            with open(path + ".tmp", "w") as f:
                json.dump(out, f)
            os.replace(path + ".tmp", path)


# --- synthetic problems (tests/test_hier.py's constructions) --------------------------

def subject(name, **kw):
    s = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date", "harness",
                       "harness_version", "reasoning_effort", "subject_features_extra"), "")
    s.update(normalized_name=name, **kw)
    return s


def item(j, bid, features="", text=None):
    return {"item_content": text or f"question {j} of {bid}", "item_features": features,
            "interactors": "", "benchmark_id": bid}


def mcq(j, bid="benchmark_7"):
    return item(j, bid, text=f"Question {j}?\nA) one\nB) two\nC) three\nD) four")


def mcq5(j, bid="benchmark_7", features=""):
    return item(j, bid, features, text=f"Question {j}?\nA) one\nB) two\nC) three\nD) four\nE) five")


def mcq3(j, bid="b"):
    return item(j, bid, text=f"Question {j}?\nA) yes\nB) no\nC) maybe")


def floored_run(seed):
    """tests/test_hier.py's _floored_run: seven subjects on three benchmarks
    (five-option items with a group key, four-option items, items without
    options), all successes, all failures and mixed records."""
    rng = np.random.default_rng(seed)
    rates = [1.0, 0.0, 0.2, 0.5, 0.8, 0.95, 0.05]
    lab = []
    for k, rate in enumerate(rates):
        s = subject(f"m{k}", provider=["openai", "anthropic"][k % 2])
        for b, mk in (("b0", lambda j: mcq5(j, "b0", f"competition=c{j % 3}")),
                      ("b1", lambda j: mcq(j, "b1")), ("b2", lambda j: item(j, "b2"))):
            if (k + len(b) + seed) % 3 == 0 and b != "b0":
                continue
            for j in rng.choice(60, 31, replace=False):
                lab.append([[s, mk(int(j))], int(rng.random() < rate)])
    return lab


def one_pair_exact(h, c, k, n):
    """tests/test_hier.py's one_pair_exact without other subjects: the exact
    posterior predictive for a new item of one pair with k successes in n
    labels on distinct new items, floor c, by quadrature over the level and
    the pair's deviation (Gaussian level)."""
    sig = lambda x: 0.5 * (1 + np.tanh(0.5 * np.asarray(x, float)))  # noqa: E731
    VD = h.sigma_theta ** 2 + h.sigma_attr ** 2 + h.sigma_delta ** 2
    S = h.sigma_d ** 2 + h.sigma_g ** 2
    span = 10 * h.sigma_mu
    mu = np.linspace(h.mu0 - span, h.mu0 + span, 4001)
    d = np.linspace(-8 * math.sqrt(VD), 8 * math.sqrt(VD), 801)
    pm = np.exp(-(mu - h.mu0) ** 2 / (2 * h.sigma_mu ** 2))
    pd = np.exp(-d ** 2 / (2 * VD))
    z = np.linspace(-9, 9, 241)
    w = np.exp(-z ** 2 / 2) / np.exp(-z ** 2 / 2).sum()
    Lt = np.linspace(mu[0] + d[0] - 1, mu[-1] + d[-1] + 1, 12001)
    Hf = np.interp(mu[:, None] + d[None, :], Lt, c + (1 - c) * (sig(Lt[:, None] - math.sqrt(S) * z) @ w))
    wgt = pm[:, None] * pd[None, :] * Hf ** k * (1 - Hf) ** (n - k)
    return float(np.sum(wgt * Hf) / np.sum(wgt))


def stage_synthetic(args):
    HB = prefix()
    out = {"one_pair": [], "seven_subjects": []}
    s = subject("solo")
    for mu0 in (-4.0, -2.5, -6.0):
        for g in (0.5, 1.0):
            h = HN.Hyper(mu0=mu0, sigma_mu=2.5, guess=g)
            for n, k in ((7, 7), (15, 15), (31, 20), (31, 25), (31, 31), (31, 10), (31, 0)):
                lab = [[[s, mcq(j)], int(j < k)] for j in range(n)]
                mb, mn = HB.HierPredictor(None, h, slip=False), HN.HierPredictor(None, h, slip=False)
                pb, pn = mb.predict([s, mcq(999)], lab), mn.predict([s, mcq(999)], lab)
                r = dict(mu0=mu0, guess=g, k=k, n=n, c=0.25 * g, base=pb, base_unconverged=mb.unconverged,
                         new=pn, new_unconverged=mn.unconverged, start=mn.fit_for(lab).prob.start,
                         exact=one_pair_exact(h, 0.25 * g, k, n))
                out["one_pair"].append(r)
                print(f"one pair mu0 {mu0} guess {g} {k}/{n}: pre-fix {pb:.4f} (unconverged {mb.unconverged}) "
                      f"fixed {pn:.4f} ({mn.unconverged}, {r['start']}) exact {r['exact']:.4f}", flush=True)
    for g in (0.5, 1.0):
        for mu0 in (-1.263, -2.5):
            h = HN.Hyper(mu0=mu0, sigma_mu=2.5, guess=g)
            for seed in range(8):
                lab = floored_run(seed)
                mb, mn = HB.HierPredictor(None, h), HN.HierPredictor(None, h)
                tgt = [[subject(f"m{k}", provider=["openai", "anthropic"][k % 2]), mcq5(999, "b0", "competition=c0")]
                       for k in range(7)]
                pb = [mb.predict(t, lab) for t in tgt]
                pn = [mn.predict(t, lab) for t in tgt]
                out["seven_subjects"].append(dict(mu0=mu0, guess=g, seed=seed, base_unconverged=mb.unconverged,
                                                  new_unconverged=mn.unconverged,
                                                  start=mn.fit_for(lab).prob.start, base=pb, new=pn))
                print(f"seven subjects mu0 {mu0} guess {g} seed {seed}: pre-fix unconverged {mb.unconverged}, "
                      f"fixed {mn.unconverged} ({mn.fit_for(lab).prob.start}); all-success subject "
                      f"{pb[0]:.3f} -> {pn[0]:.3f}", flush=True)
    _write_part(args, "synthetic", out)


def _write_part(args, name, obj):
    os.makedirs(args.data, exist_ok=True)
    path = os.path.join(args.data, f"{name}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def stage_modes(args):
    grid = list(itertools.product((-6, -3, 0, 3, 6), (1.0, 2.5, 5.0), (0.5, 1.0), (0, 1, 2, 3, 5, 8, 12),
                                  (15, 31), ("mcq5", "mcq3"), ((), (0.9, 0.9), (0.1, 0.1))))
    tot = Counter()
    gains, dxs, found = [], [], []
    t0 = time.time()
    with recording(probe=True) as rec, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (mu0, smu, guess, k, n, mk, others) in grid:
            s = subject("solo")
            mkf = (lambda j: mcq5(j, "b")) if mk == "mcq5" else mcq3
            lab = [[[s, mkf(j)], int(j < k)] for j in range(n)]
            rng = np.random.default_rng(0)
            for q, r in enumerate(others):
                lab += [[[subject(f"o{q}"), item(100 * (q + 1) + j, "b")], int(rng.random() < r)] for j in range(31)]
            rec.fits.clear()
            m = HN.HierPredictor(None, HN.Hyper(mu0=mu0, sigma_mu=smu, guess=guess), slip=False)
            m.fit_for(lab)
            tot["problems"] += 1
            tot["unconverged"] += m.unconverged
            for f in rec.fits:
                tot["floored fits"] += 1
                if not f["newton_converged"]:
                    tot["Newton short (settled)"] += 1
                    continue
                for c in f.get("probe", []):
                    if "error" in c:
                        tot["candidate errors"] += 1
                    elif not c["conv"]:
                        tot["candidates unconverged"] += 1
                    else:
                        tot["candidates"] += 1
                        gains.append(c["gain"])
                        dxs.append(c["dx"])
                        if c["dx"] > 1e-4 or c["gain"] > HN.LP_TOL:
                            found.append(dict(mu0=mu0, sigma_mu=smu, guess=guess, k=k, n=n, kind=mk,
                                              others=list(others), **c))
    out = dict(problems=len(grid), counts=dict(tot), max_gain=max(gains) if gains else None,
               min_gain=min(gains) if gains else None, max_dx=max(dxs) if dxs else None,
               second_modes=found, secs=round(time.time() - t0, 1))
    print(json.dumps({k: v for k, v in out.items() if k != "second_modes"}, indent=1), "second modes", len(found))
    _write_part(args, "modes", out)


# --- hidden-test-like runs through the shipped bundle ----------------------------------

HIDDEN_SUBJECTS = [("gpt-5.1", "openai", "2025-11-12", "high"), ("claude-opus-4.5", "anthropic", "2025-11-24", ""),
                   ("gemini-3-pro", "google", "2025-11-18", "high"), ("grok-4", "xai", "2025-07-09", ""),
                   ("deepseek-v3.2", "deepseek", "2025-12-01", ""), ("qwen3-max", "alibaba", "2025-09-05", ""),
                   ("kimi-k2-thinking", "moonshot", "2025-11-06", ""),
                   ("gpt-5-mini", "openai", "2025-08-07", "medium"),
                   ("claude-sonnet-4.5", "anthropic", "2025-09-29", ""),
                   ("gemini-2.5-flash", "google", "2025-06-17", "")]


def hidden_subject(variant, name, prov, date, effort):
    s = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date", "harness",
                       "harness_version", "reasoning_effort", "subject_features_extra"), "")
    if variant in ("real", "crowdreal"):
        s.update(normalized_name=name, provider=prov, release_date=date, access_date="2026-05-01",
                 reasoning_effort=effort)
    else:
        s.update(normalized_name="anon-" + name)
    return s


def hidden_item(j, bid, kind):
    if kind == "mc5":
        text = f"Problem {j} of {bid}: which is right?\nA) one\nB) two\nC) three\nD) four\nE) five"
    elif kind == "mc4":
        text = f"Problem {j} of {bid}: which is right?\n(A) one\n(B) two\n(C) three\n(D) four"
    else:
        text = f"Task {j} of {bid}: write the function that does thing number {j}."
    return {"item_content": text, "item_features": "", "interactors": "", "benchmark_id": bid}


def hidden_scenario(variant, seed):
    rng = np.random.default_rng(seed)
    kinds = ["mc5", "mc5", "mc5", "mc5", "mc4", "text", "text"]
    rng.shuffle(kinds)
    pairs = []
    subs = rng.permutation(len(HIDDEN_SUBJECTS))
    crowd = variant.startswith("crowd")
    for b in range(7):
        n_pairs = 1 + int(rng.random() < 0.3) if not crowd else 4
        for k in range(n_pairs):
            s = hidden_subject(variant, *HIDDEN_SUBJECTS[subs[(b + 3 * k) % len(HIDDEN_SUBJECTS)]])
            rate = float(rng.choice([1.0, 1.0, 0.97, 0.93, 0.85, 0.7, 0.5]))
            if crowd:
                rate = [1.0, 0.2, 0.05, 0.5][k]
            bid = f"bench_{seed}_{b}"
            ys = (rng.random(31) < rate).astype(int)
            items = [hidden_item(int(j), bid, kinds[b]) for j in rng.choice(500, 32, replace=False)]
            pairs.append((s, items[:31], ys, items[31]))
    return pairs


def stage_hidden(args):
    from paiec.prior import from_json
    HB = prefix()
    path = os.path.join(ROOT, "submission", "prior.json")
    with open(path) as f:
        prior, hyper = from_json(json.load(f))
    out = {"bundle": os.path.relpath(path, ROOT), "bundle_digest": file_digest("submission/prior.json"),
           "variants": {}}
    counts = dict(HIDDEN_VARIANTS)
    if args.scenarios:
        counts = {v: args.scenarios for v in counts}
    for variant in args.variants or list(counts):
        tot = Counter()
        secs = {"base": 0.0, "new": 0.0}
        maxdiff, worst = 0.0, []
        for seed in range(counts[variant]):
            pairs = hidden_scenario(variant, seed)
            for B in (1, 3, 7, 15, 31):
                lab = [[[s, it], int(y)] for s, its, ys, _ in pairs for it, y in zip(its[:B], ys[:B])]
                got = {}
                for impl, mod in (("base", HB), ("new", HN)):
                    m = mod.HierPredictor(prior, hyper)
                    t0 = time.perf_counter()
                    ps = np.array([m.predict([s, tgt], lab) for s, _, _, tgt in pairs])
                    secs[impl] += time.perf_counter() - t0
                    got[impl] = ps
                    tot[f"unconverged_{impl}"] += m.unconverged
                    tot[f"failures_{impl}"] += m.failures
                    tot[f"fit_sets_unconverged_{impl}"] += m.unconverged > 0
                tot["fit_sets"] += 1
                d = np.abs(got["new"] - got["base"])
                tot["fit_sets_changed"] += bool((d > 0).any())
                maxdiff = max(maxdiff, float(d.max()))
                if d.max() > 0.01:
                    j = int(d.argmax())
                    s, _, ys, _ = pairs[j]
                    worst.append(dict(seed=seed, B=B, subject=s["normalized_name"], successes=int(ys[:B].sum()),
                                      base=float(got["base"][j]), new=float(got["new"][j])))
        out["variants"][variant] = dict(scenarios=counts[variant], counts=dict(tot), maxdiff=maxdiff,
                                        secs=secs, changed=worst)
        print(variant, dict(tot), f"max {maxdiff:.3f}, time base {secs['base']:.1f}s new {secs['new']:.1f}s",
              flush=True)
    _write_part(args, "hidden", out)


# --- summary ------------------------------------------------------------------------

def load_rows(data):
    out = {}
    for name in ROWS:
        rows = []
        for f in glob.glob(os.path.join(data, name, "*.json")):
            with open(f) as fh:
                rows.append(json.load(fh))
        out[name] = sorted(rows, key=lambda r: r["run"])
    return out


def run_alc_diff(r):
    """A run's ALC, fixed minus pre-fix: the mean over its pairs of W6 . diff."""
    if r["skipped"]:
        return 0.0
    return float(np.mean([W6 @ (np.array(a["brier_new"]) - np.array(a["brier_base"])) for a in r["apps"]]))


def summarize_row(name, rows):
    regime, floor, guess, n = ROWS[name]
    rep = [r for r in rows if not r["skipped"]]
    d = np.array([run_alc_diff(r) for r in rows])
    fits = [f for r in rep for f in r["fits"]]
    ch = [r for r in rep if r["changed"]]
    checks = {}
    for key in ("fixed_vs_harness_rows", "fixed_vs_legacy_rows", "prefix_vs_legacy_rows"):
        v = [r["checks"][key]["max_abs"] for r in rep if r["checks"].get(key)]
        if v:
            checks[key] = dict(runs=len(v), max_abs=max(v))
    ver = [r for r in rep if r["base_replayed"] and not r["newton_failures"]]
    return dict(regime=regime, floor=floor, guess=0.5 if guess is None else guess, runs_planned=n,
                runs=len(rows), runs_with_floored_label=len(rep), floored_fits=len(fits),
                cold_floored_fits=sum(f["cold"] for f in fits),
                newton_short=sum(not f["newton_converged"] for f in fits),
                unconverged_before=sum(r["unconverged"]["base"] for r in rep),
                unconverged_after=sum(r["unconverged"]["new"] for r in rep),
                failures=dict(base=sum(r["failures"]["base"] for r in rep), new=sum(r["failures"]["new"] for r in rep)),
                runs_base_replayed=sum(r["base_replayed"] for r in rep),
                runs_changed=len(ch), predictions_changed=sum(r["changed"] for r in rep),
                predictions=sum(6 * r["responses"] for r in rep),
                max_abs_change=max((r["maxdiff"] for r in rep), default=0.0),
                alc_diff=float(d.mean()) if len(d) else None,
                alc_diff_run_se=float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None,
                verified=dict(runs=len(ver), predictions=sum(6 * r["responses"] for r in ver),
                              changed=sum(r["changed"] for r in ver)),
                checks=checks)


def stage_summary(args):
    data = load_rows(args.data)
    out = {"provenance": dict(pre_fix=PRE_FIX, pre_fix_hier_digest=prefix().SOURCE_DIGEST,
                              fixed_hier_digest=file_digest("paiec/hier.py"), lib_digest=H.digest(H.LIB),
                              script_digest=file_digest("experiments/hier_floor_replay.py"),
                              ship=H.ship_name(), level=H.shipped_level(), data=os.path.relpath(args.data, ROOT)),
           "replay": {}, "affected": [], "collapsed_pairs": []}
    rescued, probe = [], Counter()
    gains, dxs = [], []
    for name, rows in data.items():
        if not rows:
            continue
        out["replay"][name] = summarize_row(name, rows)
        for r in rows:
            if r["skipped"]:
                continue
            for f in r["fits"]:
                if not f["newton_converged"]:
                    rescued.append(dict(row=name, run=r["run"], budget=f["budget"], start=f["start"],
                                        converged=f["converged"], lp_gain=f["lp_gain"], dx=f["dx"], p=f["p"],
                                        n=f["n"], floored_successes=f["floored_successes"], secs=f["secs"]))
                elif "probe" in f:
                    probe["converged floored fits probed"] += 1
                    for c in f["probe"]:
                        if "error" in c:
                            probe["candidate errors"] += 1
                        elif not c["conv"]:
                            probe["candidates unconverged"] += 1
                        else:
                            probe["candidates"] += 1
                            gains.append(c["gain"])
                            dxs.append(c["dx"])
                            probe["above LP_TOL"] += c["gain"] > HN.LP_TOL
            if not (r["newton_failures"] or r["changed"]):
                continue
            out["affected"].append(dict(row=name, run=r["run"], unconverged=r["unconverged"],
                                        newton_failures=r["newton_failures"], changed=r["changed"],
                                        maxdiff=r["maxdiff"], alc_diff=run_alc_diff(r)))
            for a in r["apps"]:
                delta = np.abs(np.array(a["mean_new"]) - np.array(a["mean_base"]))
                if delta.max() > 0.05:
                    b = int(delta.argmax())
                    out["collapsed_pairs"].append(dict(
                        row=name, run=r["run"], pair=f"{a['name']} {a['subject'][:8]}", budget=H.BUDGETS[b],
                        before=a["mean_base"][b], after=a["mean_new"][b], before_b31=a["mean_base"][5],
                        after_b31=a["mean_new"][5], brier_before_b31=a["brier_base"][5],
                        brier_after_b31=a["brier_new"][5], observed=a["rate"], responses=a["responses"],
                        label_rate=a["label_rate"], labels=a["labels"], pool_rate=a["pool_rate"], pool=a["pool"]))
    out["rescued"] = dict(fits=len(rescued), converged=sum(f["converged"] for f in rescued),
                          starts=dict(Counter(f["start"] for f in rescued)),
                          dx=[min((f["dx"] for f in rescued), default=None), max((f["dx"] for f in rescued), default=None)],
                          lp_gain=[min((f["lp_gain"] for f in rescued), default=None),
                                   max((f["lp_gain"] for f in rescued), default=None)],
                          settle_secs=[min((f["secs"] for f in rescued), default=None),
                                       max((f["secs"] for f in rescued), default=None)],
                          list=rescued)
    out["probe"] = dict(counts=dict(probe), max_gain=max(gains) if gains else None,
                        min_gain=min(gains) if gains else None, max_dx=max(dxs) if dxs else None,
                        lp_tol=HN.LP_TOL)
    tim = {}
    for f in sorted(glob.glob(os.path.join(args.data, "timing", "*.json"))):
        with open(f) as fh:
            t = json.load(fh)
        rs = t["runs"]
        if not rs:
            continue
        tb, tn = sum(r["secs"]["base"] for r in rs), sum(r["secs"]["new"] for r in rs)
        calls = sum(r["calls"] for r in rs) * t["reps"]
        ratio = np.array([r["secs"]["new"] / r["secs"]["base"] for r in rs])
        tim[t["regime"]] = dict(runs=[rs[0]["run"], rs[-1]["run"]], reps=t["reps"], calls=calls,
                                base_ms_per_call=1000 * tb / calls, new_ms_per_call=1000 * tn / calls,
                                change=tn / tb - 1, per_run_ratio_median=float(np.median(ratio)),
                                per_run_ratio_quartiles=[float(v) for v in np.percentile(ratio, [25, 75])],
                                same_predictions_runs=sum(r["same"] for r in rs), n_runs=len(rs),
                                runs_with_unconverged_base=[r["run"] for r in rs if r["unconverged"]["base"]],
                                per_run_change={r["run"]: r["secs"]["new"] / r["secs"]["base"] - 1
                                                for r in rs if r["unconverged"]["base"]})
    out["timing"] = tim
    for part in ("synthetic", "modes", "hidden"):
        p = os.path.join(args.data, f"{part}.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[part] = json.load(fh)
    if "synthetic" in out:
        sy = out["synthetic"]
        cells = {}
        for r in sy["seven_subjects"]:
            key = f"mu0 {r['mu0']} guess {r['guess']}"
            c = cells.setdefault(key, dict(seeds=0, newton_short=0, unconverged_after=0))
            c["seeds"] += 1
            c["newton_short"] += r["base_unconverged"] > 0
            c["unconverged_after"] += r["new_unconverged"] > 0
        sy["seven_subjects_summary"] = cells
    with open(args.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(args.out + ".tmp", args.out)
    print(f"wrote {os.path.relpath(args.out, ROOT)}")


def stage_show(args):
    with open(args.out) as f:
        s = json.load(f)
    print("| regime | floor | guess | runs | with a floored label | floored fits | unconverged before | after "
          "| runs changed | predictions changed | max abs change | ALC difference (run SE) | base replayed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in s["replay"].items():
        print(f"| {REGIME_NAMES[r['regime']]} | {r['floor']} | {r['guess']:g} | {r['runs']} | "
              f"{r['runs_with_floored_label']} | {r['floored_fits']:,} | {r['unconverged_before']} | "
              f"{r['unconverged_after']} | {r['runs_changed']} | {r['predictions_changed']:,} of "
              f"{r['predictions']:,} | {r['max_abs_change']:.3f} | {r['alc_diff']:+.7f} ({r['alc_diff_run_se']:.7f}) "
              f"| {r['runs_base_replayed']} |")
    print("\nverified (both solvers, no Newton failure) and stored-row checks:")
    for name, r in s["replay"].items():
        print(f"  {name}: verified {r['verified']}, checks {r['checks']}, failures {r['failures']}")
    rs = s["rescued"]
    print(f"\nrescued fits: {rs['fits']} ({rs['converged']} converged), starts {rs['starts']}, dx {rs['dx']}, "
          f"lp gain {rs['lp_gain']}, _settle seconds {rs['settle_secs']}")
    print(f"probe of converged floored fits: {s['probe']}")
    print("\n| row | run | pair | budget | before | after | B31 before -> after | B31 Brier before -> after "
          "| observed (responses) | labels | pool |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in s["collapsed_pairs"]:
        print(f"| {c['row']} | {c['run']} | {c['pair']} | B{c['budget']} | {c['before']:.3f} | {c['after']:.3f} | "
              f"{c['before_b31']:.3f} -> {c['after_b31']:.3f} | {c['brier_before_b31']:.3f} -> "
              f"{c['brier_after_b31']:.3f} | {c['observed']:.3f} ({c['responses']}) | "
              f"{c['label_rate']:.3f} ({c['labels']}) | {c['pool_rate']:.3f} ({c['pool']}) |")
    print("\naffected runs:", ", ".join(f"{a['row']} {a['run']}" for a in s["affected"]))
    for reg, t in s.get("timing", {}).items():
        print(f"timing {reg}: runs {t['runs']} x {t['reps']}, pre-fix {t['base_ms_per_call']:.3f} ms/call, fixed "
              f"{t['new_ms_per_call']:.3f} ({100 * t['change']:+.1f}%), per-run median ratio "
              f"{t['per_run_ratio_median']:.3f} (quartiles {t['per_run_ratio_quartiles'][0]:.3f}, "
              f"{t['per_run_ratio_quartiles'][1]:.3f}), same predictions on {t['same_predictions_runs']}/{t['n_runs']}, "
              f"runs with an unconverged pre-fix fit {t['per_run_change']}")
    sy = s.get("synthetic")
    if sy:
        print("\none pair (four-option items, c = guess/4):")
        for r in sy["one_pair"]:
            print(f"  mu0 {r['mu0']} guess {r['guess']} {r['k']}/{r['n']}: pre-fix {r['base']:.4f} "
                  f"({r['base_unconverged']}) fixed {r['new']:.4f} ({r['new_unconverged']}, {r['start']}) "
                  f"exact {r['exact']:.4f}")
        print("seven subjects, seeds with a fit Newton left short:", sy["seven_subjects_summary"])
    if s.get("modes"):
        m = s["modes"]
        print(f"\nmodes: {m['problems']} problems, {m['counts']}, max gain {m['max_gain']}, max dx {m['max_dx']}, "
              f"second modes {len(m['second_modes'])}")
    if s.get("hidden"):
        for v, h in s["hidden"]["variants"].items():
            c = h["counts"]
            ch = h["changed"]
            rng_b = (min((w["base"] for w in ch), default=None), max((w["base"] for w in ch), default=None))
            rng_n = (min((w["new"] for w in ch), default=None), max((w["new"] for w in ch), default=None))
            print(f"hidden {v}: {h['scenarios']} scenarios, {c['fit_sets']} fit sets, collapsed (pre-fix fit "
                  f"unconverged) {c['fit_sets_unconverged_base']}, changed {c['fit_sets_changed']}, fixed unconverged "
                  f"{c['unconverged_new']}, failures {c['failures_base']}/{c['failures_new']}; changed pairs "
                  f"pre-fix {rng_b} fixed {rng_n}, budgets {sorted({w['B'] for w in ch})}; time pre-fix "
                  f"{h['secs']['base']:.1f}s fixed {h['secs']['new']:.1f}s "
                  f"({100 * (h['secs']['new'] / h['secs']['base'] - 1):+.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=("compare", "timing", "synthetic", "modes", "hidden", "summary", "show"))
    ap.add_argument("--rows", nargs="+", choices=list(ROWS), help="compare: rows (default all)")
    ap.add_argument("--runs", help="compare/timing: runs a:b instead of each row's 0..n-1 (timing: 0:40)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--verify", type=int, default=30,
                    help="compare: replay the pre-fix solver on runs below this in every row")
    ap.add_argument("--both", action="store_true", help="compare: replay the pre-fix solver on every run")
    ap.add_argument("--no-probe", dest="probe", action="store_false",
                    help="compare: do not try the other starts on converged floored fits")
    ap.add_argument("--redo", action="store_true", help="compare: redo runs already stored")
    ap.add_argument("--rss-cap", type=float, default=0.0,
                    help="compare: exit with status 3 once resident memory exceeds this many MB after a run, "
                         "for a wrapper to restart (the heap fragments; 0 = never)")
    ap.add_argument("--regimes", nargs="+", default=["r1p", "r1b"], choices=["r1p", "r1b", "tl"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--variants", nargs="+", choices=list(HIDDEN_VARIANTS))
    ap.add_argument("--scenarios", type=int, default=None, help="hidden: scenarios per variant (default per variant)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    {"compare": stage_compare, "timing": stage_timing, "synthetic": stage_synthetic, "modes": stage_modes,
     "hidden": stage_hidden, "summary": stage_summary, "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
