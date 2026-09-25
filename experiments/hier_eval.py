"""paiec.hier under the official protocol, against the shipped Predictor.

Every configuration scores the same runs (paired). Priors and hyperparameters
are fitted leave-one-benchmark-out relative to what is scored:

  target-LOBO (primary)  for every benchmark of a run, the offline prior and
                         the empirical-Bayes hyperparameters (paiec.prior.build)
                         fitted without that benchmark; the model factory maps
                         the anonymous benchmark_id to the fitted objects, so
                         predict never sees a name. The Predictor's attribute
                         prior is left out the same way.
  strict run-LOBO        everything fitted without every benchmark of the run.
                         On public data that leaves no benchmark, or a single
                         multi-subject one, in most runs, and every field
                         fit_hyper cannot identify falls back to
                         paiec.prior.REFERENCE (sigma_delta 2.5, mu0 0, ...), so
                         it mostly measures those fallbacks.

Regimes
  R1  formative-like runs (official.sample_run defaults, <= 1,000 subject-item
      pairs, 5 to 12 pairs) under two pair weightings, 'benchmark' (a benchmark
      first, the default) and 'pair' (pairs uniform), and two split scopes,
      'pair' and 'benchmark'. The main configurations (smoothed Beta(2,2), the
      Predictor, hier) score runs 0..299 of every setting; on the primary
      setting (benchmark-first, split scope 'pair') strict run-LOBO scores runs
      0..149, the ablations 0..149, and the costly ablations (Student-t level,
      text term, linking) and the sensitivities 0..99 (--runs-*).
  R2  dense: every eligible pair of one benchmark, whole; both split scopes;
      real_webagents and researchcodebench by default (matharena and
      multi_swebench take tens of minutes a configuration), with a few
      ablations.

Statistics. Paired ALC differences with the SE over runs and a pair-cluster
bootstrap SE (runs redraw the same 221 pairs): each pair appearance contributes
its ALC difference / (pairs in its run), summed per pair over runs; 2,000
multinomial resamples of the pairs that appear. The same decomposition gives
each budget's weighted contribution and each benchmark's; the benchmark-level
line weights the benchmarks equally, with the SE across them (five). ECE-ALC
and per-call latency (evaluation calls, `jobs` processes side by side) per
config.

Partial results are checkpointed to results/hier_eval.json as tasks finish
(status 'partial', raw per-pair rows); --resume skips tasks already there and
--summarise rebuilds the summary from it.

Run: python experiments/hier_eval.py --jobs 8 --skip 'r2|researchcodebench|pair|hier t3 level'
(about 2 h 15 min on eight processes of a machine shared with other jobs; the
skipped task, the Student-t level on dense researchcodebench under split scope
'pair', alone takes over 35 minutes). results/hier_eval.json lists the resumed
invocations that produced it; tasks are pure functions of their key, so a
resume reproduces a single run.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec import baselines as B  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec import prior as PR  # noqa: E402
from paiec.data import anon_id  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.hier import HierPredictor  # noqa: E402
from paiec.predict import Predictor, fit_prior  # noqa: E402

OUT = os.path.join(ROOT, "results", "hier_eval.json")
BOOTS = 2000
SMOOTHED, PRED, HIER = "smoothed Beta(2,2)", "Predictor", "hier"
PRED_RUN, HIER_RUN = "Predictor run-LOBO", "hier run-LOBO"
MAIN = (SMOOTHED, PRED, HIER)
STRICT = (PRED_RUN, HIER_RUN)
REFS = (SMOOTHED, PRED, HIER)

#: name -> (bundle variant, Flags overrides, Hyper transform or None)
HIER_CONFIGS = {
    HIER: ("default", {}, None),
    "hier -attributes": ("default", {"attributes": False}, None),
    "hier -pool_mu": ("default", {"pool_mu": False}, None),
    "hier -delta": ("default", {"delta": False}, None),
    "hier -groups": ("default", {"groups": False}, None),
    "hier -link": ("default", {"link": False}, None),
    "hier relink 0.1": ("default", {}, ("relink", 0.1)),
    "hier relink 0.3": ("default", {}, ("relink", 0.3)),
    "hier t3 level": ("t3", {}, None),
    "hier -line": ("default", {"line": False}, None),
    "hier guess 1 (hard floor)": ("default", {}, ("set", "guess", 1.0)),
    "hier centre zero": ("zero", {}, None),
    "hier +text": ("default", {"text": True}, None),
}
SENSITIVITY = {
    "hier sigma_mu x0.5": ("default", {}, ("scale", "sigma_mu", 0.5)),
    "hier sigma_mu x2": ("default", {}, ("scale", "sigma_mu", 2.0)),
    "hier sigma_delta x0.5": ("default", {}, ("scale", "sigma_delta", 0.5)),
    "hier sigma_delta x2": ("default", {}, ("scale", "sigma_delta", 2.0)),
}
ABLATIONS = [n for n in HIER_CONFIGS if n != HIER]
SENS = list(SENSITIVITY)
#: scored on fewer runs (--runs-abl2): the Student-t level costs about seven
#: default fits, and linking or the text term can matter only where a subject
#: recurs across benchmarks or a benchmark reaches `warmup` labeled items
COSTLY = ["hier t3 level", "hier +text", "hier -link", "hier relink 0.1", "hier relink 0.3"] + SENS
CHEAP = [n for n in ABLATIONS if n not in COSTLY]
ALL_HIER = {**HIER_CONFIGS, **SENSITIVITY}
#: R2 ablations: the options dense evidence should move most
DENSE_ABL = ["hier -pool_mu", "hier t3 level", "hier centre zero", "hier -line",
             "hier -groups", "hier sigma_mu x2", "hier +text"]
DENSE_FAST = ("real_webagents", "researchcodebench")
DENSE_SLOW = ("matharena", "multi_swebench")

_pairs = None
_bundles, _pred_priors = {}, {}


def pairs():
    global _pairs
    if _pairs is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from paiec.data import load_pairs
            _pairs = O.eligible(load_pairs())
    return _pairs


def draw(weighting, seed, i):
    return O.sample_run(pairs(), np.random.default_rng([seed, i]), weighting=weighting)


# --- fitted objects, built once in the parent ----------------------------------------

def build_bundle(variant, excluded):
    ex = tuple(sorted(excluded))
    if variant == "default":
        return PR.build(pairs(), ex)
    if variant == "t3":
        return PR.build(pairs(), ex, nu_mu=3.0)
    if variant == "zero":
        prior = PR.build_prior(pairs(), ex)
        hyper, _ = PR.fit_hyper(pairs(), ex, prior, centre="zero")
        return prior, hyper
    raise ValueError(variant)


def fallback_report(excluded):
    """What strict run-LOBO leaves and which hyperparameters fall back."""
    ex = tuple(sorted(excluded))
    prior = PR.build_prior(pairs(), ex)
    _, rep = PR.fit_hyper(pairs(), ex, prior)
    left = sorted({p.benchmark_id for p in pairs()} - set(ex))
    multi = [b for b in left if sum(p.benchmark_id == b for p in pairs()) >= 2]
    return {"left": left, "multi_left": multi, "fallback": rep["fallback"],
            "approximate": rep["approximate"]}


def init(bundles, pred_priors):
    _bundles.update(bundles)
    _pred_priors.update(pred_priors)
    pairs()


def pred_prior(excluded):
    key = tuple(sorted(excluded))
    if key not in _pred_priors:
        train = [p for p in pairs() if p.benchmark_id not in key]
        _pred_priors[key] = fit_prior(train) if train else (None, None)
    return _pred_priors[key]


def hier_model(name, excluded):
    variant, flags, tf = ALL_HIER[name]
    prior, hyper = _bundles[(variant, tuple(sorted(excluded)))]
    if tf is not None:
        if tf[0] == "relink":
            hyper = hyper.relink(tf[1])
        elif tf[0] == "set":
            hyper = replace(hyper, **{tf[1]: tf[2]})
        elif tf[0] == "scale":
            hyper = replace(hyper, **{tf[1]: getattr(hyper, tf[1]) * tf[2]})
    return HierPredictor(prior, hyper, **flags)


def factory(name, names, made):
    """A fresh predict per call, as the platform rebuilds workers per checkpoint.
    `made` collects the models so failures can be counted afterwards."""
    if name == SMOOTHED:
        return lambda: B.smoothed_mean(4.0, 0.5)
    if name == PRED_RUN:
        def f():
            m = Predictor(*pred_prior(names))
            made.append(m)
            return m.predict
        return f
    if name == HIER_RUN:
        def f():
            m = hier_model(HIER, names)
            made.append(m)
            return m.predict
        return f
    per = name == PRED

    def f():
        by = {}
        for b in names:
            m = Predictor(*pred_prior([b])) if per else hier_model(name, [b])
            made.append(m)
            by[anon_id("benchmark", b)] = m
        return lambda input, labeled=None: by[input[1]["benchmark_id"]].predict(input, labeled)
    return f


def score(run, name, names, scope):
    made = []
    res = O.run_official(run, factory(name, names, made), deepcopy=False, workers=1,
                         split_scope=scope)
    rows = [[round(r["brier"][b], 6) for b in BUDGETS] + [round(r["ALC"], 6), round(r["ece_alc"], 6)]
            for r in res["rows"]]
    t = res["timing"]
    return {"rows": rows,
            "timing": {k: t[k] for k in ("evaluation_calls", "evaluation_mean_s",
                                         "evaluation_max_s", "wall_s")},
            "failures": int(sum(getattr(m, "failures", 0) for m in made)),
            "unconverged": int(sum(getattr(m, "unconverged", 0) for m in made))}


def task_key(t):
    return "|".join(str(x) for x in t)


def dispatch(task):
    kind, *args = task
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "r1":
            weighting, scope, seed, i, name = args
            run = draw(weighting, seed, i)
            out = score(run, name, O.run_benchmarks(run), scope)
        else:
            bench, scope, name = args
            run = O.dense_run(pairs(), bench)
            out = score(run, name, [bench], scope)
    out["task_s"] = time.perf_counter() - t0
    return task_key(task), out


# --- statistics ------------------------------------------------------------------

def mean_se(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0]


class Cluster:
    """Pair-cluster bootstrap over the pairs that appear in a set of runs.
    runs: [[pair key per pair of the run]]."""

    def __init__(self, runs, boots=BOOTS, seed=0):
        self.keys = sorted({k for r in runs for k in r})
        index = {k: j for j, k in enumerate(self.keys)}
        self.idx = [np.array([index[k] for k in r]) for r in runs]
        self.bench = np.array([k[0] for k in self.keys])
        rng = np.random.default_rng(seed)
        K = len(self.keys)
        self.W = rng.multinomial(K, np.full(K, 1 / K), size=boots).astype(float)

    def per_pair(self, diffs):
        """diffs: per run, per pair of the run -> each pair's summed contribution
        to the mean over runs of the run's mean difference."""
        v = np.zeros(len(self.keys))
        for i, d in zip(self.idx, diffs):
            np.add.at(v, i, np.asarray(d) / len(d) / len(diffs))
        return v

    def stat(self, diffs):
        v = self.per_pair(diffs)
        bs = self.W @ v
        run = [float(np.mean(d)) for d in diffs]
        return {"diff": float(v.sum()), "run_se": mean_se(run)[1],
                "cluster_se": float(bs.std(ddof=1)),
                "cluster_95": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}

    def by_benchmark(self, diffs):
        """Per benchmark: its contribution to the difference, the mean difference
        per pair appearance, and a pair-cluster SE of that mean (resampling its
        pairs)."""
        out = {}
        s = np.zeros(len(self.keys))
        m = np.zeros(len(self.keys))
        for i, d in zip(self.idx, diffs):
            np.add.at(s, i, np.asarray(d))
            np.add.at(m, i, 1.0)
        contrib = self.per_pair(diffs)
        rng = np.random.default_rng(1)
        for b in sorted(set(self.bench)):
            sel = self.bench == b
            k = int(sel.sum())
            W = rng.multinomial(k, np.full(k, 1 / k), size=self.W.shape[0]).astype(float)
            ratio = (W @ s[sel]) / np.maximum(W @ m[sel], 1e-12)
            out[b] = {"pairs": k, "appearances": int(m[sel].sum()),
                      "contribution": float(contrib[sel].sum()),
                      "per_appearance": float(s[sel].sum() / m[sel].sum()),
                      # one pair cannot be resampled: no SE
                      "per_appearance_cluster_se": float(ratio.std(ddof=1)) if k > 1 else None}
        return out

    @staticmethod
    def benchmark_level(by):
        """Each benchmark's mean difference per appearance, weighted equally,
        with the SE across benchmarks: the uncertainty that matters for a test
        made of new benchmarks, and with five of them a coarse one."""
        x = np.array([v["per_appearance"] for v in by.values()])
        return {"benchmarks": len(x), "mean": float(x.mean()),
                "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None,
                "wins": int((x < 0).sum())}


def get_rows(raw, key):
    r = raw.get(key)
    return None if r is None else np.array(r["rows"])


def summarise_setting(raw, weighting, scope, seed, n_runs, names, refs):
    """One R1 setting: per config the table, and paired statistics vs refs
    over the runs both hold."""
    runs = {i: draw(weighting, seed, i) for i in range(n_runs)}
    pk = {i: [(p.benchmark_id, p.subject_id) for p, _ in run] for i, run in runs.items()}
    have = {n: {i: get_rows(raw, task_key(("r1", weighting, scope, seed, i, n)))
                for i in range(n_runs)} for n in names}
    have = {n: {i: r for i, r in d.items() if r is not None} for n, d in have.items()}
    out = {"configs": {}, "diffs": {}}
    for n in names:
        rs = have[n]
        if not rs:
            continue
        ids = sorted(rs)
        run_brier = np.array([rs[i][:, :len(BUDGETS) + 1].mean(0) for i in ids])
        tim = [raw[task_key(("r1", weighting, scope, seed, i, n))] for i in ids]
        calls = sum(t["timing"]["evaluation_calls"] for t in tim)
        out["configs"][n] = {
            "runs": len(ids),
            "brier": [mean_se(run_brier[:, j]) for j in range(len(BUDGETS) + 1)],
            "ece_alc": mean_se([rs[i][:, -1].mean() for i in ids]),
            "latency": {"mean_ms": 1e3 * sum(t["timing"]["evaluation_calls"] * t["timing"]["evaluation_mean_s"]
                                             for t in tim) / max(calls, 1),
                        "max_s": max(t["timing"]["evaluation_max_s"] for t in tim),
                        "mean_run_s": float(np.mean([t["task_s"] for t in tim]))},
            "failures": int(sum(t["failures"] for t in tim)),
            "unconverged": int(sum(t["unconverged"] for t in tim))}
    for ref in refs:
        if ref not in out["configs"]:
            continue
        for n in names:
            if n == ref or n not in out["configs"]:
                continue
            ids = sorted(set(have[n]) & set(have[ref]))
            if len(ids) < 2:
                continue
            cl = Cluster([pk[i] for i in ids])
            A = [have[n][i] for i in ids]
            R = [have[ref][i] for i in ids]
            alc = [a[:, len(BUDGETS)] - r[:, len(BUDGETS)] for a, r in zip(A, R)]
            st = cl.stat(alc)
            st["runs"] = len(ids)
            st["per_budget"] = {}
            for j, (w, b) in enumerate(zip(WEIGHTS, BUDGETS)):
                s = cl.stat([w * (a[:, j] - r[:, j]) for a, r in zip(A, R)])
                st["per_budget"][str(b)] = {"contribution": s["diff"], "cluster_se": s["cluster_se"]}
            st["by_benchmark"] = cl.by_benchmark(alc)
            st["benchmark_level"] = Cluster.benchmark_level(st["by_benchmark"])
            st["ece_alc_diff"] = cl.stat([a[:, -1] - r[:, -1] for a, r in zip(A, R)])["diff"]
            out["diffs"].setdefault(n, {})[ref] = st
    return out


def summarise_dense(raw, bench, scope, names):
    out = {"configs": {}, "diffs": {}}
    rows = {n: get_rows(raw, task_key(("r2", bench, scope, n))) for n in names}
    rows = {n: r for n, r in rows.items() if r is not None}
    for n, r in rows.items():
        t = raw[task_key(("r2", bench, scope, n))]
        out["configs"][n] = {"pairs": len(r),
                             "brier": [mean_se(r[:, j]) for j in range(len(BUDGETS) + 1)],
                             "ece_alc": mean_se(r[:, -1]),
                             "latency": {"mean_ms": 1e3 * t["timing"]["evaluation_mean_s"],
                                         "max_s": t["timing"]["evaluation_max_s"],
                                         "run_s": t["task_s"]},
                             "failures": t["failures"], "unconverged": t["unconverged"]}
    for ref in REFS:
        if ref not in rows:
            continue
        for n, r in rows.items():
            if n == ref:
                continue
            d = r[:, len(BUDGETS)] - rows[ref][:, len(BUDGETS)]
            out["diffs"].setdefault(n, {})[ref] = {
                "diff": mean_se(d),
                "per_budget": {str(b): float(w * (r[:, j] - rows[ref][:, j]).mean())
                               for j, (w, b) in enumerate(zip(WEIGHTS, BUDGETS))}}
    return out


def run_meta(weighting, seed, n_runs, strict):
    runs = [draw(weighting, seed, i) for i in range(n_runs)]
    nb = [len(O.run_benchmarks(r)) for r in runs]
    share = {}
    for r in runs:
        for b in O.run_benchmarks(r):
            share[b] = share.get(b, 0) + 1
    out = {"runs": n_runs, "n_pairs_mean": float(np.mean([len(r) for r in runs])),
           "benchmarks_per_run": float(np.mean(nb)),
           "share_with": {b: v / n_runs for b, v in sorted(share.items())},
           "subjects_per_run": float(np.mean([len({p.subject_id for p, _ in r}) for r in runs]))}
    if strict:
        rep = [strict[tuple(O.run_benchmarks(r))] for r in runs]
        out["strict_lobo"] = {
            "multi_subject_left": {str(k): int(sum(len(x["multi_left"]) == k for x in rep))
                                   for k in range(5)},
            "nothing_left": int(sum(not x["left"] for x in rep)),
            "fallback_share": {f: float(np.mean([f in x["fallback"] for x in rep]))
                               for f in PR.REFERENCE}}
    return out


def strict_breakdown(raw, weighting, scope, seed, n_runs, strict):
    """hier run-LOBO minus Predictor run-LOBO, and each minus smoothed, by the
    number of multi-subject benchmarks strict run-LOBO leaves."""
    out = {}
    for i in range(n_runs):
        run = draw(weighting, seed, i)
        k = len(strict[tuple(O.run_benchmarks(run))]["multi_left"])
        got = {n: get_rows(raw, task_key(("r1", weighting, scope, seed, i, n)))
               for n in (HIER_RUN, PRED_RUN, SMOOTHED, HIER, PRED)}
        if any(v is None for v in got.values()):
            continue
        a = {n: float(v[:, len(BUDGETS)].mean()) for n, v in got.items()}
        g = out.setdefault(str(min(k, 2)), {"runs": 0, "hier-R minus Pred-R": [],
                                            "hier-R minus smoothed": [], "hier-T minus Pred-T": []})
        g["runs"] += 1
        g["hier-R minus Pred-R"].append(a[HIER_RUN] - a[PRED_RUN])
        g["hier-R minus smoothed"].append(a[HIER_RUN] - a[SMOOTHED])
        g["hier-T minus Pred-T"].append(a[HIER] - a[PRED])
    return {k: {kk: (mean_se(v) if isinstance(v, list) else v) for kk, v in g.items()}
            for k, g in sorted(out.items())}


def summarise(raw, cfg, strict):
    seed = cfg["seed"]
    out = {"r1": {}, "r2": {}, "meta": {}}
    for w in cfg["weightings"]:
        has = any(k.startswith(w + "/") for k in cfg["strict_settings"])
        out["meta"][w] = run_meta(w, seed, cfg["runs_main"], strict if has else None)
        for s in cfg["scopes"]:
            key = f"{w}/{s}"
            names = list(MAIN) + (list(STRICT) if key in cfg["strict_settings"] else []) \
                + (ABLATIONS + SENS if key in cfg["abl_settings"] else [])
            refs = REFS + ((PRED_RUN,) if key in cfg["strict_settings"] else ())
            out["r1"][key] = summarise_setting(raw, w, s, seed, cfg["runs_main"], names, refs)
            if key in cfg["strict_settings"]:
                out["r1"][key]["strict_by_left"] = strict_breakdown(
                    raw, w, s, seed, cfg["runs_main"], strict)
    for b in cfg["dense"]:
        for s in cfg["scopes"]:
            names = [SMOOTHED, PRED, HIER] + (DENSE_ABL if b in cfg["dense_abl"] else [])
            got = summarise_dense(raw, b, s, names)
            if got["configs"]:
                out["r2"][f"{b}/{s}"] = got
    return out


# --- printing ---------------------------------------------------------------------

def f4(x):
    return f"{x[0]:.4f}±{x[1]:.4f}"


def show(summary, cfg):
    comps = cfg["n_comparisons"]
    for key, st in summary["r1"].items():
        print(f"\n=== R1 {key} (weighting/split scope)")
        head = "".join(f"{'B' + str(b):>9}" for b in BUDGETS)
        print(f"{'config':<28}{'runs':>5}{head}{'ALC':>16}{'ECE-ALC':>9}{'ms/call':>8}{'max s':>7}")
        for n, c in st["configs"].items():
            print(f"{n:<28}{c['runs']:>5}" + "".join(f"{x[0]:>9.4f}" for x in c["brier"][:-1])
                  + f"{f4(c['brier'][-1]):>16}{c['ece_alc'][0]:>9.4f}{c['latency']['mean_ms']:>8.2f}"
                  f"{c['latency']['max_s']:>7.2f}")
        print(f"  paired ALC differences (row minus ref): diff, run SE, pair-cluster SE")
        for n, d in st["diffs"].items():
            line = f"  {n:<28}"
            for ref, x in d.items():
                line += f" vs {ref}: {x['diff']:+.4f} ({x['run_se']:.4f}, {x['cluster_se']:.4f})"
            print(line)
        if st.get("strict_by_left"):
            print("  strict run-LOBO by multi-subject benchmarks left: " + "; ".join(
                f"{k}: {g['runs']} runs, hier-R - Pred-R {f4(g['hier-R minus Pred-R'])}, "
                f"hier-T - Pred-T {f4(g['hier-T minus Pred-T'])}"
                for k, g in st["strict_by_left"].items()))
    for key, st in summary["r2"].items():
        print(f"\n=== R2 dense {key}")
        for n, c in st["configs"].items():
            print(f"  {n:<28}" + " ".join(f"{x[0]:.4f}" for x in c["brier"][:-1])
                  + f"  ALC {f4(c['brier'][-1])}  ECE {c['ece_alc'][0]:.4f}  "
                  f"{c['latency']['mean_ms']:.2f} ms/call, max {c['latency']['max_s']:.2f} s, "
                  f"{c['latency']['run_s']:.0f} s")
        for n, d in st["diffs"].items():
            print(f"  {n:<28}" + "  ".join(f"vs {r.split()[0]} {f4(x['diff'])}" for r, x in d.items()))
    print(f"\n{comps} hier options compared with the default on the same runs")


# --- driver -------------------------------------------------------------------------

def tasks_for(cfg):
    """Dense main configurations first (few, long), then R1 in proportion: at
    step k the main configurations of run k and each smaller group's share of
    its runs (strict run-LOBO, ablations, costly ablations), so every group
    advances at the same pace and a stop leaves runs 0..j of each; dense
    ablations last."""
    seed, tasks = cfg["seed"], []
    for b in cfg["dense"]:
        for s in cfg["scopes"]:
            for n in (HIER, PRED, SMOOTHED):
                tasks.append(("r2", b, s, n))
    n = cfg["runs_main"]
    groups = [(list(STRICT), cfg["runs_strict"], cfg["strict_settings"]),
              (CHEAP, cfg["runs_abl"], cfg["abl_settings"]),
              (COSTLY, cfg["runs_abl2"], cfg["abl_settings"])]
    for k in range(n):
        for w in cfg["weightings"]:
            for s in cfg["scopes"]:
                tasks += [("r1", w, s, seed, k, m) for m in MAIN]
        for names, runs, settings in groups:
            for i in range(k * runs // n, (k + 1) * runs // n):
                for key in settings:
                    w, s = key.split("/")
                    tasks += [("r1", w, s, seed, i, m) for m in names]
    for b in cfg["dense_abl"]:
        for s in cfg["scopes"]:
            for n in DENSE_ABL:
                tasks.append(("r2", b, s, n))
    return tasks


def needed_bundles(cfg):
    """(variant, excluded) for every hier model the tasks can build, and the
    run sets that strict run-LOBO needs."""
    names = sorted({p.benchmark_id for p in pairs()})
    need = {(v, (b,)) for v in ("default", "t3", "zero") for b in names}
    sets = set()
    for w in {k.split("/")[0] for k in cfg["strict_settings"]}:
        for i in range(cfg["runs_main"]):
            sets.add(tuple(O.run_benchmarks(draw(w, cfg["seed"], i))))
    need |= {("default", s) for s in sets}
    return sorted(need), sorted(sets)


def write(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=None, separators=(",", ":"))
    os.replace(tmp, path)


def main(args):
    t0 = time.time()
    cfg = {"seed": args.seed, "runs_main": args.runs_main, "runs_abl": args.runs_abl,
           "runs_abl2": args.runs_abl2, "runs_strict": args.runs_strict,
           "weightings": args.weightings, "scopes": args.scopes,
           "strict_settings": args.strict_settings, "abl_settings": args.abl_settings,
           "dense": args.dense, "dense_abl": args.dense_abl, "jobs": args.jobs, "boots": BOOTS,
           "budgets": list(BUDGETS), "weights": list(WEIGHTS),
           "n_comparisons": len(ABLATIONS) + len(SENS), "skipped": args.skip}
    raw, old = {}, {}
    if (args.resume or args.summarise) and os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        raw = old.get("raw", {})
        print(f"loaded {len(raw)} finished tasks from {os.path.relpath(args.out, ROOT)}")
    need, sets = needed_bundles(cfg)
    strict = {s: fallback_report(s) for s in sets}
    command = " ".join([os.path.basename(sys.executable)] + sys.argv)
    history = old.get("invocations", []) + [command]
    doc = {"status": "partial", "command": command, "invocations": history, "config": cfg,
           "strict_lobo_sets": {"|".join(k): v for k, v in strict.items()}, "raw": raw}
    if not args.summarise:
        tasks = [t for t in tasks_for(cfg) if task_key(t) not in raw
                 and not any(x in task_key(t) for x in args.skip)]
        t1 = time.time()
        bundles = {k: build_bundle(*k) for k in need}
        names = sorted({p.benchmark_id for p in pairs()})
        pp = {(b,): fit_prior([p for p in pairs() if p.benchmark_id != b]) for b in names}
        for s in sets:
            train = [p for p in pairs() if p.benchmark_id not in s]
            pp[s] = fit_prior(train) if train else (None, None)
        doc["hyper"] = {f"{v}/{'|'.join(ex)}": h.to_dict() for (v, ex), (_, h) in bundles.items()
                        if len(ex) == 1}
        print(f"{len(bundles)} bundles, {len(pp)} Predictor priors in {time.time() - t1:.0f}s; "
              f"{len(tasks)} tasks on {args.jobs} processes", flush=True)
        last = time.time()
        with mp.get_context("spawn").Pool(args.jobs, initializer=init,
                                          initargs=(bundles, pp)) as pool:
            for k, (key, res) in enumerate(pool.imap_unordered(dispatch, tasks, chunksize=1)):
                raw[key] = res
                if time.time() - last > args.every or k + 1 == len(tasks):
                    doc["wall_s"] = time.time() - t0
                    write(args.out, doc)
                    last = time.time()
                    print(f"  {k + 1}/{len(tasks)} tasks [{time.time() - t0:.0f}s], checkpointed",
                          flush=True)
    summary = summarise(raw, cfg, strict)
    show(summary, cfg)
    doc.update(summary)
    doc["status"] = "complete" if not args.summarise or old.get("status") == "complete" else doc["status"]
    doc["wall_s"] = old.get("wall_s") if args.summarise else time.time() - t0
    write(args.out, doc)
    print(f"\nwrote {os.path.relpath(args.out, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-main", type=int, default=300)
    ap.add_argument("--runs-abl", type=int, default=150)
    ap.add_argument("--runs-abl2", type=int, default=100,
                    help="runs for the costly ablations and the sensitivities")
    ap.add_argument("--runs-strict", type=int, default=150)
    ap.add_argument("--weightings", nargs="+", default=list(O.WEIGHTINGS))
    ap.add_argument("--scopes", nargs="+", default=list(O.SPLIT_SCOPES))
    ap.add_argument("--strict-settings", nargs="*", default=["benchmark/pair"],
                    help="weighting/scope settings that also score strict run-LOBO")
    ap.add_argument("--abl-settings", nargs="*", default=["benchmark/pair"],
                    help="weighting/scope settings that score the ablations")
    ap.add_argument("--dense", nargs="*", default=list(DENSE_FAST),
                    help=f"dense benchmarks; {', '.join(DENSE_SLOW)} take tens of minutes")
    ap.add_argument("--dense-abl", nargs="*", default=list(DENSE_FAST))
    ap.add_argument("--every", type=float, default=120.0, help="seconds between checkpoints")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="drop tasks whose key contains any of these strings")
    ap.add_argument("--summarise", action="store_true")
    main(ap.parse_args())
