"""Reference predictors and the run-time Predictor under the official protocol.

Every earlier number in docs/findings.md came from the legacy pair-major
replica. This re-measures on paiec.official, in two regimes:

  R1  formative-like runs from official.sample_run at its defaults (at most
      1,000 subject-item pairs, 5 to 12 pairs spread over benchmarks). Every
      predictor scores the same runs, so differences are paired and their
      standard errors are over runs.
  R2  dense: every eligible pair of one benchmark with all its items, the most
      cross-subject evidence a run could carry. Standard errors are over the
      benchmark's pairs, which share one `labeled` list and are not independent.

The Predictor is the shipped one (paiec.predict.Predictor at its defaults, a
pure function of input and labeled) with the attribute prior refitted without
any benchmark present in the run, the honest stand-in for a hidden test whose
benchmarks were never public. The public data hold five binary benchmarks, so a
run that touches all five gets no prior at all. A second line refits the prior
without the target's benchmark only; the prior then knows a subject's standing
on the run's other benchmarks, as the shipped prior may know a hidden-test
subject's standing on public ones. In a dense run the two coincide.

Runs are drawn from default_rng([seed, i]) so any run can be rebuilt in any
process, and they are spread over `jobs` processes. The predictors inside run
with workers=1 and no argument copies; one run is checked against deepcopy=True
and 16 workers. Latencies are measured with `jobs` processes busy side by side.

Run: python experiments/official_baselines.py --runs 600 --jobs 6
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec import baselines as B  # noqa: E402
from paiec import data as D  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.predict import Predictor, fit_prior  # noqa: E402

OUT = os.path.join(ROOT, "results", "official_baselines.json")
DENSE = ("multi_swebench", "matharena", "researchcodebench", "real_webagents")
BASELINES = {
    "constant 0.5": lambda: B.const(0.5),
    "empirical mean": lambda: B.empirical_mean,
    "smoothed Beta(2,2)": lambda: B.smoothed_mean(4.0, 0.5),
    "pooled anchor": lambda: B.pooled_anchor(),
}
PREDICTOR, PER_TARGET = "Predictor", "Predictor, target-LOBO prior"
REFS = ("empirical mean", "smoothed Beta(2,2)")
#: live leaderboard, formative runs on the hidden test
LEADERBOARD = {"organisers' entry": 0.1801, "best entry": 0.1172}
#: the empirical mean's ALC for pairs with base rate p: B0 is 0.25, budget B
#: costs p(1-p)(1 + 1/B) when the acquisition rate equals the evaluation rate
EMP_A = WEIGHTS[0] * 0.25
EMP_K = sum(w * (1 + 1 / b) for w, b in zip(WEIGHTS, BUDGETS) if b)

_pairs, _priors = None, {}


def pairs():
    global _pairs
    if _pairs is None:
        _pairs = O.eligible(D.load_pairs())
    return _pairs


def prior(excluded):
    """Attribute prior fitted on every pair outside `excluded`; (None, None)
    when nothing is left, which the Predictor reads as no prior."""
    key = frozenset(excluded)
    if key not in _priors:
        train = [p for p in pairs() if p.benchmark_id not in key]
        _priors[key] = fit_prior(train) if train else (None, None)
    return _priors[key]


def per_target(names):
    """One Predictor per benchmark of the run, each with its benchmark left out
    of the prior, dispatched on the anonymous benchmark_id."""
    by_id = {D.anon_id("benchmark", b): Predictor(*prior([b])) for b in names}
    return lambda input, labeled=None: by_id[input[1]["benchmark_id"]].predict(input, labeled)


def predictors(run, dense=False):
    names = O.run_benchmarks(run)
    out = dict(BASELINES)
    out[PREDICTOR] = lambda: Predictor(*prior(names)).predict
    if not dense:
        out[PER_TARGET] = lambda: per_target(names)
    return out


def base_rates(run, seed=0):
    """Per pair: the success rate p over its evaluation responses, the rate pa
    over its acquisition candidates (the first recorded response, the one
    revealed) and their number na, all as official._slots selects them. A
    constant per-pair prediction of p scores p(1-p) at every budget: the
    base-rate oracle."""
    out = []
    for p, items in run:
        acq, ev = (set(k) & items for k in O.split(p, seed))
        by = p.by_item()
        out.append((float(np.mean([r.label for r in p.responses if r.item_key in ev])),
                    float(np.mean([by[k][0].label for k in acq])), len(acq)))
    return out


def emp_expected(p, pa, na):
    """The empirical mean's expected ALC on one pair. The random policy reveals a
    uniformly random subset in random order, so at budget B it predicts the mean
    of B draws without replacement from na candidates of rate pa, and the pair
    scores p(1-p) + (pa - p)^2 + pa(1-pa)/B * (na-B)/(na-1). With pa = p and no
    finite-pool correction this is the EMP_A + EMP_K * p(1-p) rule."""
    out = WEIGHTS[0] * 0.25
    for w, b in zip(WEIGHTS[1:], BUDGETS[1:]):
        n = min(b, na)
        out += w * (p * (1 - p) + (pa - p) ** 2 + pa * (1 - pa) / n * (na - n) / max(na - 1, 1))
    return out


def scores(res, rows=False):
    out = {"brier": [res["brier"][b] for b in BUDGETS] + [res["brier"]["ALC"]],
           "ece": [res["ece"][b] for b in BUDGETS] + [res["ece"]["ALC"]],
           "timing": res["timing"]}
    if rows:
        out["rows"] = [[r["brier"][b] for b in BUDGETS] + [r["ALC"], r["ece_alc"]]
                       for r in res["rows"]]
    return out


def r1(seed, i):
    run = O.sample_run(pairs(), np.random.default_rng([seed, i]))
    names = O.run_benchmarks(run)
    out = {"run": i, "benchmarks": names, "n_pairs": len(run),
           "n_subjects": len({p.subject_id for p, _ in run}),
           "n_items": sum(len(k) for _, k in run),
           "prior_pairs": len(O.training_pairs(pairs(), run)), "p": base_rates(run)}
    for name, factory in predictors(run).items():
        out[name] = scores(O.run_official(run, factory, deepcopy=False, workers=1))
    return out


def r2(bench, name):
    run = O.dense_run(pairs(), bench)
    res = O.run_official(run, predictors(run, dense=True)[name], deepcopy=False, workers=1)
    return {"benchmark": bench, "predictor": name, "p": base_rates(run),
            "n_items": sum(len(k) for _, k in run), **scores(res, rows=True)}


def order(seed, i):
    """The empirical mean's ALC on run i minus its exact expectation, with the
    split and stream order salted by i. With them fixed, as the platform fixes
    them, a pair's first labels are nearly the same items in every run that
    holds it, so averaging over runs does not average over which items come
    first and the residual need not vanish."""
    run = O.sample_run(pairs(), np.random.default_rng([seed, i]))
    res = O.run_official(run, BASELINES["empirical mean"], deepcopy=False, seed=i + 1)
    return res["brier"]["ALC"] - float(np.mean([emp_expected(*x) for x in base_rates(run, i + 1)]))


def check(seed):
    """Run 0 with the platform's copies and 16 workers against the fast path:
    per-pair Brier and ECE at every budget, and the acquired items, must be
    identical for a pure predictor."""
    run = O.sample_run(pairs(), np.random.default_rng([seed, 0]))
    out = {}
    for name, factory in predictors(run).items():
        a = O.run_official(run, factory, deepcopy=False, workers=1)
        b = O.run_official(run, factory, deepcopy=True, workers=16)
        same = a["acquired"] == b["acquired"] and all(
            ra["brier"] == rb["brier"] and ra["ece"] == rb["ece"]
            for ra, rb in zip(a["rows"], b["rows"]))
        out[name] = {"identical": bool(same), "wall_fast_s": a["timing"]["wall_s"],
                     "wall_platform_s": b["timing"]["wall_s"]}
    return out


def dispatch(task):
    kind, *args = task
    t = time.perf_counter()
    res = {"r1": r1, "r2": r2, "check": check, "order": order}[kind](*args)
    return kind, res, time.perf_counter() - t


# --- aggregation ------------------------------------------------------------------

def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def column(rows, name, key, j=-1):
    return np.array([r[name][key][j] for r in rows])


def latency(timings):
    calls = sum(t["evaluation_calls"] for t in timings)
    total = sum(t["evaluation_calls"] * t["evaluation_mean_s"] for t in timings)
    maxes = [t["evaluation_max_s"] for t in timings]
    return {"calls": calls, "mean_s": total / calls, "max_s": max(maxes),
            "median_run_max_s": float(np.median(maxes)),
            "mean_run_wall_s": float(np.mean([t["wall_s"] for t in timings]))}


def summarise_r1(runs, names):
    table, diffs = {}, {}
    for n in names:
        table[n] = {"brier": [mean_se(column(runs, n, "brier", j)) for j in range(len(BUDGETS) + 1)],
                    "ece_alc": mean_se(column(runs, n, "ece")),
                    "sd_run_alc": float(column(runs, n, "brier").std(ddof=1))}
        for ref in REFS:
            d = column(runs, n, "brier") - column(runs, ref, "brier")
            diffs.setdefault(n, {})[ref] = mean_se(d)
    q = np.array([np.mean([p * (1 - p) for p, _, _ in r["p"]]) for r in runs])
    emp = column(runs, "empirical mean", "brier")
    pred = EMP_A + EMP_K * q
    exact = np.array([np.mean([emp_expected(*x) for x in r["p"]]) for r in runs])
    analytic = {"formula": f"{EMP_A:.3f} + {EMP_K:.4f} * mean p(1-p)",
                "E_q": mean_se(q), "observed": mean_se(emp), "predicted": mean_se(pred),
                "residual": mean_se(emp - pred), "corr": float(np.corrcoef(emp, pred)[0, 1]),
                "exact": mean_se(exact), "exact_residual": mean_se(emp - exact),
                "exact_corr": float(np.corrcoef(emp, exact)[0, 1]),
                "pool_gap": mean_se([np.mean([(pa - p) ** 2 for p, pa, _ in r["p"]]) for r in runs])}
    return table, diffs, analytic, q


def leaderboard(runs, names, q):
    """Where the live scores fall among single R1 runs, and what the
    empirical mean's formula says about the hidden test's base rates."""
    ps = np.array([x[0] for r in runs for x in r["p"]])
    out = {"pairs_mean_p": float(ps.mean()), "pairs_sd_p": float(ps.std()),
           "q_percentiles": {k: float(np.percentile(q, k)) for k in (0, 5, 25, 50, 75, 95, 100)},
           "entries": {}, "fits": {}}
    for n in names:
        alc = column(runs, n, "brier")
        slope, icpt = np.polyfit(q, alc, 1)
        out["fits"][n] = {"intercept": float(icpt), "slope": float(slope),
                          "resid_sd": float(np.std(alc - icpt - slope * q, ddof=2))}
    for entry, v in LEADERBOARD.items():
        q_star = (v - EMP_A) / EMP_K
        disc = 1 - 4 * q_star
        out["entries"][entry] = {
            "alc": v, "q_if_empirical_mean": q_star,
            "homogeneous_p": (1 - np.sqrt(disc)) / 2 if disc >= 0 else None,
            "sd_p_needed_at_public_mean": float(np.sqrt(max(
                0.0, out["pairs_mean_p"] * (1 - out["pairs_mean_p"]) - q_star))),
            "share_runs_q_below": float(np.mean(q <= q_star)),
            "share_runs_at_or_below": {n: float(np.mean(column(runs, n, "brier") <= v))
                                       for n in names},
            "fit_at_q": {n: out["fits"][n]["intercept"] + out["fits"][n]["slope"] * q_star
                         for n in names}}
    return out


def summarise_r2(res, names):
    out = {}
    for bench in DENSE:
        got = {n: res[(bench, n)] for n in names if (bench, n) in res}
        if not got:
            continue
        any_ = next(iter(got.values()))
        per = {"n_pairs": len(any_["rows"]), "n_items": any_["n_items"],
               "mean_p": float(np.mean([x[0] for x in any_["p"]])),
               "mean_q": float(np.mean([p * (1 - p) for p, _, _ in any_["p"]])),
               "emp_expected": float(np.mean([emp_expected(*x) for x in any_["p"]])),
               "predictors": {}}
        for n, r in got.items():
            rows = np.array(r["rows"])
            per["predictors"][n] = {
                "brier": [mean_se(rows[:, j]) for j in range(len(BUDGETS) + 1)],
                "run_brier": r["brier"], "ece_alc": mean_se(rows[:, -1]),
                "diff": {ref: mean_se(rows[:, len(BUDGETS)] - np.array(got[ref]["rows"])[:, len(BUDGETS)])
                         for ref in REFS if ref in got},
                "latency": latency([r["timing"]]) if n == PREDICTOR else None}
        out[bench] = per
    return out


# --- printing -------------------------------------------------------------------

def fmt(m_se):
    return f"{m_se[0]:.4f}±{m_se[1]:.4f}"


def print_table(table, names, extra=None):
    head = "".join(f"{'B' + str(b):>14}" for b in BUDGETS)
    print(f"{'predictor':<30}{head}{'ALC':>14}{'ECE-ALC':>14}")
    for n in names:
        t = table[n]
        print(f"{n:<30}" + "".join(f"{fmt(x):>14}" for x in t["brier"]) + f"{fmt(t['ece_alc']):>14}")
    if extra:
        print(extra)


def main(runs=600, seed=0, jobs=6, dense=True, out=OUT):
    t0 = time.time()
    r1_names = list(BASELINES) + [PREDICTOR, PER_TARGET]
    r2_names = list(BASELINES) + [PREDICTOR]
    tasks = []
    if dense:       # the slowest first, so they overlap the runs
        tasks += [("r2", b, PREDICTOR) for b in DENSE]
        tasks += [("r2", b, n) for b in DENSE for n in BASELINES]
    tasks += [("check", seed)] + [("r1", seed, i) for i in range(runs)]
    tasks += [("order", seed, i) for i in range(runs)]
    r1_res, r2_res, checked, task_s, varied = [], {}, None, {}, []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for k, (kind, res, dt) in enumerate(pool.imap_unordered(dispatch, tasks, chunksize=1)):
            if kind == "r1":
                r1_res.append(res)
            elif kind == "order":
                varied.append(res)
            elif kind == "r2":
                r2_res[(res["benchmark"], res["predictor"])] = res
                task_s[f"r2 {res['benchmark']} / {res['predictor']}"] = dt
            else:
                checked = res
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(tasks)} tasks [{time.time() - t0:.0f}s]", flush=True)
    r1_res.sort(key=lambda r: r["run"])

    table, diffs, analytic, q = summarise_r1(r1_res, r1_names)
    analytic["exact_residual_varied_order"] = mean_se(varied)
    board = leaderboard(r1_res, r1_names, q)
    lat = {n: latency([r[n]["timing"] for r in r1_res]) for n in (PREDICTOR, PER_TARGET)}
    n_pairs = np.array([r["n_pairs"] for r in r1_res])
    n_bench = np.array([len(r["benchmarks"]) for r in r1_res])
    prior_pairs = np.array([r["prior_pairs"] for r in r1_res])
    meta = {"runs": len(r1_res), "n_pairs_mean": float(n_pairs.mean()),
            "n_pairs_range": [int(n_pairs.min()), int(n_pairs.max())],
            "n_subjects_mean": float(np.mean([r["n_subjects"] for r in r1_res])),
            "n_items_mean": float(np.mean([r["n_items"] for r in r1_res])),
            "benchmarks_per_run": {int(k): int((n_bench == k).sum()) for k in np.unique(n_bench)},
            "share_with": {b: float(np.mean([b in r["benchmarks"] for r in r1_res]))
                           for b in D.BINARY},
            "share_prior_empty": float(np.mean(prior_pairs == 0)),
            "prior_pairs_when_fitted": float(prior_pairs[prior_pairs > 0].mean())
            if (prior_pairs > 0).any() else 0.0}

    print(f"\nR1: {meta['runs']} formative-like runs (sample_run defaults, seed {seed}); "
          f"{meta['n_pairs_mean']:.1f} pairs ({meta['n_pairs_range'][0]}-{meta['n_pairs_range'][1]}), "
          f"{meta['n_subjects_mean']:.1f} subjects, {meta['n_items_mean']:.0f} items per run")
    print("  benchmarks per run: " + ", ".join(f"{k}: {v}" for k, v in meta["benchmarks_per_run"].items()))
    print("  share of runs holding: " + ", ".join(f"{b} {v:.0%}" for b, v in meta["share_with"].items()))
    print(f"  run-level LOBO prior empty in {meta['share_prior_empty']:.0%} of runs; "
          f"otherwise fitted on {meta['prior_pairs_when_fitted']:.0f} pairs on average")
    print("\nmean over runs ± SE (Brier by budget, ALC, ECE-ALC)")
    print_table(table, r1_names)
    print(f"{'base-rate oracle':<30}" + f"{fmt(analytic['E_q']):>14}" * (len(BUDGETS) + 1))
    print("\nsd of a single run's ALC: " + ", ".join(
        f"{n} {table[n]['sd_run_alc']:.4f}" for n in r1_names))
    print("\npaired ALC difference (row minus reference), mean ± SE over runs")
    print(f"{'predictor':<30}" + "".join(f"{'vs ' + r:>26}" for r in REFS))
    for n in r1_names:
        print(f"{n:<30}" + "".join(f"{'-' if n == r else fmt(diffs[n][r]):>26}" for r in REFS))
    print(f"\nanalytic check, empirical mean: ALC ~ {analytic['formula']}")
    print(f"  E[p(1-p)] {fmt(analytic['E_q'])}; observed {fmt(analytic['observed'])}, "
          f"formula {fmt(analytic['predicted'])}, residual {fmt(analytic['residual'])}, "
          f"per-run correlation {analytic['corr']:.3f}")
    print(f"  exact expectation (acquisition rate, finite pool): {fmt(analytic['exact'])}, "
          f"residual {fmt(analytic['exact_residual'])}, correlation {analytic['exact_corr']:.3f}; "
          f"mean (pa - p)^2 {fmt(analytic['pool_gap'])}")
    print(f"  the same residual with split and stream order salted per run: "
          f"{fmt(analytic['exact_residual_varied_order'])}")
    print("\nPredictor latency per evaluation call (R1, "
          f"{jobs} processes side by side):")
    for n, v in lat.items():
        print(f"  {n}: mean {v['mean_s'] * 1e3:.2f} ms, max {v['max_s']:.3f} s, median per-run "
              f"max {v['median_run_max_s']:.3f} s, {v['mean_run_wall_s']:.1f} s per run")
    print(f"\nleaderboard: public pairs have mean p {board['pairs_mean_p']:.3f}, sd {board['pairs_sd_p']:.3f}; "
          "run-level E[p(1-p)] percentiles " + ", ".join(
              f"{k}%: {v:.3f}" for k, v in board["q_percentiles"].items()))
    for entry, e in board["entries"].items():
        hp = "none" if e["homogeneous_p"] is None else f"{e['homogeneous_p']:.3f}"
        print(f"  {entry} {e['alc']:.4f}: as an empirical mean it needs E[p(1-p)] = "
              f"{e['q_if_empirical_mean']:.4f} (every pair at p = {hp} or 1-p, or sd(p) = "
              f"{e['sd_p_needed_at_public_mean']:.3f} at the public mean); "
              f"{e['share_runs_q_below']:.1%} of R1 runs have E[p(1-p)] that low")
        print("    share of R1 runs at or below it: " + ", ".join(
            f"{n} {v:.1%}" for n, v in e["share_runs_at_or_below"].items()))
        print("    linear fit of ALC on run E[p(1-p)], evaluated there: " + ", ".join(
            f"{n} {v:.4f}" for n, v in e["fit_at_q"].items()))

    r2_sum = summarise_r2(r2_res, r2_names) if dense else {}
    for bench, per in r2_sum.items():
        print(f"\nR2 dense {bench}: {per['n_pairs']} pairs, {per['n_items']} items, "
              f"mean p {per['mean_p']:.3f}, E[p(1-p)] {per['mean_q']:.4f}, empirical mean expected "
              f"{per['emp_expected']:.4f}; mean ± SE over pairs")
        print_table(per["predictors"], [n for n in r2_names if n in per["predictors"]])
        for n, v in per["predictors"].items():
            d = {r: x for r, x in v["diff"].items() if r != n}
            if d:
                print(f"  {n:<28}" + "  ".join(f"vs {r}: {fmt(x)}" for r, x in d.items()))
        lat2 = per["predictors"].get(PREDICTOR, {}).get("latency")
        if lat2:
            print(f"  Predictor latency: mean {lat2['mean_s'] * 1e3:.2f} ms, max {lat2['max_s']:.3f} s, "
                  f"{lat2['mean_run_wall_s']:.0f} s for the run")

    print("\ndeepcopy=True, workers=16 vs deepcopy=False, workers=1 on run 0: " + ", ".join(
        f"{n} {'identical' if v['identical'] else 'DIFFERENT'}" for n, v in checked.items()))
    wall = time.time() - t0
    print(f"\ntotal {wall:.0f}s with {jobs} processes")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"command": " ".join([os.path.basename(sys.executable)] + sys.argv),
                   "config": {"runs": runs, "seed": seed, "jobs": jobs, "budgets": BUDGETS,
                              "weights": WEIGHTS, "leaderboard": LEADERBOARD},
                   "r1": {"meta": meta, "table": table, "paired_diffs": diffs,
                          "analytic": analytic, "latency": lat, "leaderboard": board,
                          "per_run": [{"run": r["run"], "benchmarks": r["benchmarks"],
                                       "n_pairs": r["n_pairs"], "q": float(qq),
                                       "alc": {n: round(r[n]["brier"][-1], 6) for n in r1_names}}
                                      for r, qq in zip(r1_res, q)]},
                   "r2": r2_sum, "check": checked, "task_seconds": task_s, "wall_s": wall},
                  f, indent=1)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-dense", dest="dense", action="store_false")
    ap.add_argument("--out", default=OUT)
    main(**vars(ap.parse_args()))
