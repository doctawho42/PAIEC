"""Prototype of hier's prior build on the expanded pool: run time and what breaks.

Not repository code and not a candidate build: it exists to measure how long
each stage of paiec.prior takes on 100+ benchmarks, and whether any stage
fails or runs out of memory, so that code.md can say what must change. It
fits nothing that is scored and compares no predictors.

Stages, each timed (CPU seconds of the process that ran it, and wall):
  1. benchmark_fit per benchmark in a process pool (one fit per benchmark, as
     paiec.prior caches them), with at most K pairs per benchmark (stable-hash
     order, proto_loader) and no item cap; single-pair benchmarks are fitted
     after the others, at the median item variance, as fit_hyper does;
  2. from_standings on the standings rows (the attribute ridge, both identity
     passes, the components), unchanged library code;
  3. the arithmetic of fit_hyper on the fits (levels at attribute score 0,
     item variance, the clipped sigma_theta, sigma_delta, sigma_attr), copied
     from paiec/prior.py lines 755-807 so that pairs need not be reloaded;
  4. prior.json as to_json would write it: size, json.load + from_json time.

Usage: python proto_build.py K M JOBS [mixed]   (M = item cap per benchmark, 0 = none)
Writes fits_K<K>_M<M>.pkl and build_K<K>_M<M>.json beside this file.
"""
import json
import math
import os
import pickle
import resource
import sys
import time
import concurrent.futures as cf

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from proto_loader import load_benchmark  # noqa: E402

import numpy as np  # noqa: E402

from paiec import prior as PR  # noqa: E402
from paiec.hier import Hyper, HierPredictor  # noqa: E402

KEEP = ("benchmark", "n_pairs", "n_items", "n_cells", "mu", "mu_var", "t", "t_var", "subjects",
        "s2_t", "s2e", "s2_g", "keys", "item_var", "share", "iterations", "converged",
        "newton_converged")


def fit_one(b, K, s2e=None, M=None):
    t0, c0 = time.perf_counter(), time.process_time()
    pairs = load_benchmark(b, K, M)
    t1, c1 = time.perf_counter(), time.process_time()
    if not pairs:
        return b, None
    if len(pairs) < 2 and s2e is None:
        return b, "single"
    f = PR.benchmark_fit(pairs, s2e=s2e if len(pairs) < 2 else None)
    t2, c2 = time.perf_counter(), time.process_time()
    out = {k: f[k] for k in KEEP}
    out.update(load_cpu=c1 - c0, fit_cpu=c2 - c1, fit_wall=t2 - t1,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    return b, out


def main():
    K = int(sys.argv[1])
    M = int(sys.argv[2]) or None
    jobs = int(sys.argv[3])
    types = ("binary", "mixed") if len(sys.argv) > 4 and sys.argv[4] == "mixed" else ("binary",)
    tag = f"K{K}_M{M or 0}"
    survey = json.load(open(os.path.join(HERE, "survey.json")))["benchmarks"]
    names = sorted(b for b, v in survey.items()
                   if v.get("response_type") in types and v.get("n_eligible_pairs", 0) >= 1)
    # largest first, so the pool is not left waiting on one long fit
    names.sort(key=lambda b: -min(survey[b]["n_eligible_pairs"], K) * survey[b]["eligible_cells"]
               / max(survey[b]["n_eligible_pairs"], 1))
    rep = {"K": K, "M": M, "types": types, "benchmarks": len(names)}
    T0 = time.perf_counter()
    fits, singles = {}, []
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fit_one, b, K, None, M): b for b in names}
        for fu in cf.as_completed(futs):
            b, f = fu.result()
            if f == "single":
                singles.append(b)
            elif f is not None:
                fits[b] = f
                print(f"{b}: {f['n_pairs']} pairs {f['n_cells']} cells it {f['iterations']} "
                      f"conv {f['converged']}/{f['newton_converged']} cpu {f['fit_cpu']:.1f}s",
                      flush=True)
    rep["stage1_multi_wall"] = time.perf_counter() - T0
    multi = dict(fits)
    T = float(np.median([f["item_var"] for f in multi.values()]))
    shares = [f["share"] for f in multi.values() if f["share"] is not None]
    t1 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for b, f in ex.map(fit_one, singles, [K] * len(singles), [T] * len(singles), [M] * len(singles)):
            fits[b] = f
    rep["stage1_single_wall"] = time.perf_counter() - t1
    rep["n_multi"], rep["n_single"] = len(multi), len(singles)
    rep["fit_cpu_total"] = sum(f["fit_cpu"] for f in fits.values())
    rep["load_cpu_total"] = sum(f["load_cpu"] for f in fits.values())
    rep["fit_cpu_top"] = sorted(((round(f["fit_cpu"], 1), b) for b, f in fits.items()), reverse=True)[:12]
    rep["maxrss_mb_worker"] = max(f["maxrss_mb"] for f in fits.values())
    rep["not_converged"] = sorted(b for b, f in fits.items() if not f["converged"])
    rep["newton_not_converged"] = sorted(b for b, f in fits.items() if not f["newton_converged"])
    rep["iterations"] = [int(x) for x in np.percentile([f["iterations"] for f in fits.values()],
                                                      [0, 50, 90, 100])]
    with open(os.path.join(HERE, f"fits_{tag}.pkl"), "wb") as fh:
        pickle.dump(fits, fh)

    # stage 2: the subject prior from the standings rows, library code as is
    rows = []
    for b in sorted(multi):
        f = multi[b]
        t = np.array(f["t"])
        for s, z, v in zip(f["subjects"], t - t.mean(), f["t_var"]):
            rows.append((b, s, float(z), float(v)))
    rep["rows"] = len(rows)
    c0, t0 = time.process_time(), time.perf_counter()
    prior = PR.from_standings(rows, 2.0, sorted(fits), ())
    rep["stage2_cpu"], rep["stage2_wall"] = time.process_time() - c0, time.perf_counter() - t0
    rep["maxrss_mb_main_after_stage2"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    rep["prior_meta"] = {k: v for k, v in prior.meta.items() if k not in ("benchmarks", "included")}
    rep["n_coef"], rep["providers"], rep["efforts"] = len(prior.coef), prior.spec.providers, prior.spec.efforts
    rep["table_rows"] = len(prior.table)
    rep["table_rows_resid"] = sum(1 for r in prior.table.values() if r[1] is not None)

    # stage 3: fit_hyper's arithmetic on the fits (paiec/prior.py 755-807)
    c0 = time.process_time()
    ref, est = dict(PR.REFERENCE), {}
    share = min(float(np.median(shares)), PR.G_CAP) if shares else \
        ref["sigma_g"] ** 2 / (ref["sigma_g"] ** 2 + ref["sigma_d"] ** 2)
    est["sigma_g"], est["sigma_d"] = math.sqrt(share * T), math.sqrt((1 - share) * T)
    levels = {}
    for b, f in fits.items():
        shift = float(np.mean(f["t"])) - float(np.mean([prior.attribute(s)[0] for s in f["subjects"]]))
        levels[b] = f["mu"] + shift
    mus = np.array([levels[b] for b in sorted(levels)])
    n = len(mus)
    est["mu0"] = float(np.mean(mus))
    est["sigma_mu"] = float(np.std(mus, ddof=1) * math.sqrt(1 + 1 / n)) * PR.WIDEN[0]
    meta = prior.meta
    th2 = float(np.clip(meta["tau2_res"], 0.01, 0.2))
    est["sigma_theta"] = math.sqrt(th2)
    est["sigma_delta"] = math.sqrt(max(meta["s2_res"] - meta["u_mean"] - th2, 0.1)) * PR.WIDEN[1]
    est["sigma_attr"] = math.sqrt(max(meta["total_var"] - meta["s2_res"], 0.0))
    hyper = Hyper(**{**Hyper().to_dict(), **est, "nu_mu": 0.0, "excluded": (),
                     "included": tuple(sorted(fits))})
    rep["stage3_cpu"] = time.process_time() - c0
    rep["hyper_fitted"] = {k: round(v, 4) for k, v in hyper.to_dict().items()
                           if k not in ("excluded", "included")}
    rep["hyper_unwidened_sigma_mu"] = float(np.std(mus, ddof=1) * math.sqrt(1 + 1 / n))
    rep["link_weight"] = hyper.link_weight
    rep["item_var_median"], rep["share_used"], rep["n_shares"] = T, share, len(shares)
    q = np.percentile(mus, [0, 5, 10, 25, 50, 75, 90, 95, 100])
    z = (mus - mus.mean()) / mus.std()
    rep["levels_at_attr0"] = {"n": n, "quantiles_0_5_10_25_50_75_90_95_100": [round(float(x), 2) for x in q],
                              "mean": float(mus.mean()), "sd": float(mus.std(ddof=1)),
                              "skew": float(np.mean(z ** 3)), "excess_kurtosis": float(np.mean(z ** 4) - 3)}
    rep["levels_fitted_mu_quantiles"] = [round(float(x), 2) for x in
                                         np.percentile([f["mu"] for f in fits.values()], [0, 25, 50, 75, 100])]

    # stage 4: the bundle as prior.json would hold it
    bundle = PR.to_json(prior, hyper)
    text = json.dumps(bundle, indent=1)
    rep["prior_json_bytes_indent1"] = len(text.encode())
    import zlib
    rep["prior_json_deflated_bytes"] = len(zlib.compress(text.encode(), 6))
    c0 = time.perf_counter()
    for _ in range(20):
        p2, h2 = PR.from_json(json.loads(text))
        HierPredictor(p2, h2)
    rep["load_validate_ms"] = (time.perf_counter() - c0) / 20 * 1e3
    P = HierPredictor(p2, h2)
    subs = [s for f in fits.values() for s in f["subjects"]]
    c0 = time.perf_counter()
    for i, s in enumerate(subs):
        P.theta_prior(i, s)
    rep["theta_prior_us_per_subject"] = (time.perf_counter() - c0) / len(subs) * 1e6
    ws = []
    for s in subs:
        _, V = P.theta_prior(("x", id(s)), s)
    for s in subs:
        obs = p2.identity(s, residual=True)
        if obs is not None:
            m, V = h2.shift, h2.sigma_theta ** 2 + p2.attribute(s)[1]
            ws.append(V / (V + obs[1]))
    rep["identity_weight_uncapped"] = ({"n": len(ws), "quantiles_0_50_90_100":
                                        [round(float(x), 4) for x in np.percentile(ws, [0, 50, 90, 100])],
                                        "share_at_cap_0.1": float(np.mean(np.array(ws) >= 0.1))}
                                       if ws else None)
    rep["total_wall"] = time.perf_counter() - T0
    with open(os.path.join(HERE, f"build_{tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("benchmarks", "n_multi", "n_single", "rows", "fit_cpu_total",
                                          "stage2_cpu", "total_wall")}))


if __name__ == "__main__":
    main()
