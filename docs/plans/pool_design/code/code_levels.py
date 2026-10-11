"""Descriptive checks behind code.md: levels on hier's run-time scale, the
attribute score a refitted ridge gives the same subjects, and the identity
table's transfer. Not repository code; scores nothing and compares no
predictors (no ALC, no Brier, no formative feedback), reads no item text.

Inputs: ../fits_K100_M2000.pkl and the prototype's settings (proto_build.py:
128 binary benchmarks, at most 100 pairs and 2,000 items each, proto_loader's
stable-hash order), submission/prior.json (read only).

Stages (all written to code_levels.json):
  A  levels at attribute score 0 as fit_hyper takes them (free item variance per
     benchmark), mapped to the run-time item variance T by the logistic-normal
     map, and refitted exactly at s2e = T for a subset (4 workers);
  B  the empirical-Bayes summary of those levels;
  C  the mean attribute score of the same subject dicts under the pool
     prototype's ridge and under the shipped prior, and the B0 prior predictive
     centre each configuration implies at that score (a property of the prior,
     no label is scored);
  D  the identity table: size, the uncapped identity weights, the empirical
     leave-one-benchmark-out shrinkage slope of a name's residual cells, and a
     benchmark jackknife of tau2_res, s2d_res and s2_res (4 workers).
"""
import concurrent.futures as cf
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PD)))
sys.path.insert(0, PD)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from paiec import prior as PR  # noqa: E402
from paiec.evaluator import stable_hash  # noqa: E402

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / GH_W.sum()
T_SHIP = 2.671083633914518 ** 2 + 1.5421508550685508 ** 2
JOBS = 4


def f_acc(mu, T):
    """logit E[sigmoid(mu + sqrt(T) z)], z standard normal."""
    p = float(np.sum(GH_W / (1 + np.exp(-(mu + math.sqrt(T) * GH_X)))))
    p = min(max(p, 1e-300), 1 - 1e-16)
    return math.log(p / (1 - p))


def f_inv(a, T):
    lo, hi = -400.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f_acc(mid, T) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eb(levels, var=None):
    x = np.asarray(levels, float)
    n = len(x)
    z = (x - x.mean()) / x.std()
    q = np.percentile(x, [0, 5, 10, 25, 50, 75, 90, 95, 100])
    out = {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
           "fit_hyper_sigma_mu": float(x.std(ddof=1) * math.sqrt(1 + 1 / n) * PR.WIDEN[0]),
           "median": float(np.median(x)), "iqr_sd": float((q[5] - q[3]) / 1.349),
           "skew": float(np.mean(z ** 3)), "excess_kurtosis": float(np.mean(z ** 4) - 3),
           "quantiles_0_5_10_25_50_75_90_95_100": [round(float(v), 3) for v in q]}
    if var is not None:
        v = np.asarray(var, float)
        s2 = max(float(x.var(ddof=1) - v.mean()), 1e-6)
        for _ in range(500):          # REML-free precision-weighted moment iteration
            w = 1 / (s2 + v)
            m = float(np.sum(w * x) / w.sum())
            new = max(float(np.sum(w * ((x - m) ** 2 - v)) / w.sum()), 1e-6)
            if abs(new - s2) < 1e-10:
                break
            s2 = new
        out.update(sampling_var_mean=float(v.mean()), eb_mean=m, eb_sd=math.sqrt(s2))
    return out


def refit_fixed(args):
    b, T = args
    from proto_loader import load_benchmark
    c0 = time.process_time()
    ps = load_benchmark(b, 100, 2000)
    f = PR.benchmark_fit(ps, s2e=T)
    return b, {"mu": f["mu"], "mu_var": f["mu_var"], "tbar": float(np.mean(f["t"])),
               "s2_t": f["s2_t"], "iterations": f["iterations"], "converged": f["converged"],
               "cpu": time.process_time() - c0}


_ROWS = None


def _init(rows):
    global _ROWS
    _ROWS = rows


def jack(b):
    rows = [r for r in _ROWS if r[0] != b]
    c0 = time.process_time()
    p = PR.from_standings(rows, 2.0)
    m = p.meta
    return b, {"tau2_res": m["tau2_res"], "s2d_res": m["s2d_res"], "s2_res": m["s2_res"],
               "total_var": m["total_var"], "linked": m["linked"], "cpu": time.process_time() - c0}


def main():
    t_start = time.perf_counter()
    fits = pickle.load(open(os.path.join(PD, "fits_K100_M2000.pkl"), "rb"))
    multi = {b: f for b, f in fits.items() if f["n_pairs"] >= 2}
    T = float(np.median([f["item_var"] for f in multi.values()]))
    rows = []
    for b in sorted(multi):
        t = np.array(multi[b]["t"])
        for s, z, v in zip(multi[b]["subjects"], t - t.mean(), multi[b]["t_var"]):
            rows.append((b, s, float(z), float(v)))
    c0 = time.process_time()
    pool = PR.from_standings(rows, 2.0, sorted(fits), ())
    out = {"T_pool": T, "T_ship": T_SHIP, "n_benchmarks": len(fits), "n_multi": len(multi),
           "rows": len(rows), "from_standings_cpu_s": time.process_time() - c0}
    ship, ship_h = PR.load(os.path.join(ROOT, "submission", "prior.json"))

    # --- A: levels -----------------------------------------------------------------
    lev = {}
    for b, f in sorted(fits.items()):
        Tb = f["item_var"] if f["n_pairs"] >= 2 else T
        tbar = float(np.mean(f["t"]))
        ma_pool = float(np.mean([pool.attribute(s)[0] for s in f["subjects"]]))
        ma_ship = float(np.mean([ship.attribute(s)[0] for s in f["subjects"]]))
        acc = f_acc(f["mu"] + tbar, Tb)
        lev[b] = {"n_pairs": f["n_pairs"], "T_b": Tb, "mu_var": f["mu_var"],
                  "free": f["mu"] + tbar - ma_pool, "acc_logit_at_mean": acc,
                  "mapped_T_pool": f_inv(acc, T) - ma_pool,
                  "mapped_T_ship_ship_prior": f_inv(acc, T_SHIP) - ma_ship,
                  "ma_pool": ma_pool, "ma_ship": ma_ship}
    # exact refits at s2e = T: the four beyond +-10, the fits with T_b > 30, 20 by hash
    extreme = sorted(b for b, v in lev.items() if abs(v["free"]) > 10 or v["T_b"] > 30)
    rest = sorted((b for b in multi if b not in extreme), key=lambda b: stable_hash("code-sub", b))[:20]
    subset = extreme + rest
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=JOBS) as ex:
        exact = dict(ex.map(refit_fixed, [(b, T) for b in subset]))
    out["refit_wall_s"] = time.perf_counter() - t0
    out["refit_cpu_s"] = sum(v["cpu"] for v in exact.values())
    agree = []
    for b, e in exact.items():
        lev[b]["exact_T_pool"] = e["mu"] + e["tbar"] - lev[b]["ma_pool"]
        lev[b]["exact_mu_var"] = e["mu_var"]
        lev[b]["exact_converged"] = e["converged"]
        agree.append((b, round(lev[b]["free"], 3), round(lev[b]["mapped_T_pool"], 3),
                      round(lev[b]["exact_T_pool"], 3), round(lev[b]["T_b"], 2)))
    out["refit_subset"] = agree
    d = np.array([lev[b]["exact_T_pool"] - lev[b]["mapped_T_pool"] for b in exact])
    out["mapped_minus_exact"] = {"mean": float(-d.mean()), "mean_abs": float(np.abs(d).mean()),
                                 "max_abs": float(np.abs(d).max()),
                                 "corr": float(np.corrcoef([lev[b]["exact_T_pool"] for b in exact],
                                                           [lev[b]["mapped_T_pool"] for b in exact])[0, 1])}
    out["mapped_minus_exact_without_extremes"] = float(np.abs([lev[b]["exact_T_pool"] - lev[b]["mapped_T_pool"]
                                                               for b in rest]).mean())

    # --- B: empirical Bayes ------------------------------------------------------------
    names = sorted(lev)
    mv = [lev[b]["mu_var"] for b in names]
    out["levels"] = {
        "free_T_as_fit_hyper": eb([lev[b]["free"] for b in names], mv),
        "mapped_to_T_pool": eb([lev[b]["mapped_T_pool"] for b in names], mv),
        "mapped_to_T_ship_under_shipped_prior": eb([lev[b]["mapped_T_ship_ship_prior"] for b in names], mv),
        "accuracy_logit_at_mean_standing": eb([lev[b]["acc_logit_at_mean"] for b in names]),
    }
    m2 = np.array([lev[b]["mapped_T_pool"] for b in names])
    # LEVEL on the pool levels' own scale
    out["LEVEL_pit_at_T_pool"] = {
        "share_of_pool_levels_below_mu0": float(np.mean(m2 < -2.5)),
        "share_below_LEVEL_q05_q50_q95": [float(np.mean(m2 < -2.5 + 2.5 * z)) for z in (-1.645, 0, 1.645)],
    }

    # --- C: attribute scores and the B0 prior centre --------------------------------------
    bw_pool = np.mean([lev[b]["ma_pool"] for b in names])
    bw_ship = np.mean([lev[b]["ma_ship"] for b in names])
    allp = [(pool.attribute(s)[0], ship.attribute(s)[0]) for f in fits.values() for s in f["subjects"]]
    legacy = {}
    from proto_loader import load_benchmark
    for b in ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"):
        subs = fits[b]["subjects"] if b in fits else [p.subject for p in load_benchmark(b)]
        legacy[b] = {"n": len(subs), "pool": float(np.mean([pool.attribute(s)[0] for s in subs])),
                     "ship": float(np.mean([ship.attribute(s)[0] for s in subs]))}
    empty = {k: "" for k in ("normalized_name", "provider", "release_date", "access_date", "harness",
                             "harness_version", "reasoning_effort", "subject_features_extra")}
    out["attribute_scores"] = {
        "benchmark_weighted_mean": {"pool_prior": float(bw_pool), "shipped_prior": float(bw_ship)},
        "row_mean": {"pool_prior": float(np.mean([a for a, _ in allp])),
                     "shipped_prior": float(np.mean([b for _, b in allp]))},
        "sd_over_rows": {"pool_prior": float(np.std([a for a, _ in allp])),
                         "shipped_prior": float(np.std([b for _, b in allp]))},
        "corr_rows": float(np.corrcoef([a for a, _ in allp], [b for _, b in allp])[0, 1]),
        "empty_subject": {"pool_prior": pool.attribute(empty)[0], "shipped_prior": ship.attribute(empty)[0]},
        "harness_only_subject": {"pool_prior": pool.attribute({**empty, "harness": "HELM"})[0],
                                 "shipped_prior": ship.attribute({**empty, "harness": "HELM"})[0]},
        "legacy_five": legacy,
        "pool_coef_names": ["intercept", "date", "has_date", "log_size", "has_size", "small", "big",
                            "think", "harness"] + [f"prov:{p}" for p in pool.spec.providers]
                           + [f"eff:{e}" for e in pool.spec.efforts],
        "pool_coef": [round(float(c), 4) for c in pool.coef],
        "shipped_coef": [round(float(c), 4) for c in ship.coef],
    }

    # the B0 prior predictive at a subject of attribute score m: hier with no labels
    # (theta ~ N(attr_scale m, sigma_theta^2 + u), delta, level and item residual
    # independent, no group key, no floor, slip 0.01)
    def b0(h, m, u=0.0):
        S = h["sigma_mu"] ** 2 + h["sigma_theta"] ** 2 + u + h["sigma_delta"] ** 2 \
            + h["sigma_d"] ** 2 + h["sigma_g"] ** 2
        p = float(np.sum(GH_W / (1 + np.exp(-(h["mu0"] + h["attr_scale"] * m + math.sqrt(S) * GH_X)))))
        return (1 - h["slip"]) * p
    proto = json.load(open(os.path.join(PD, "build_K100_M2000.json")))["hyper_fitted"]
    level = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
    hs = ship_h.to_dict()
    c1 = {**proto, **level}
    out["b0_prior_centre"] = {
        "shipped_at_its_bw_mean_score": b0(hs, bw_ship),
        "shipped_at_score_0": b0(hs, 0.0),
        "C1_literal_LEVEL_at_pool_bw_mean_score": b0(c1, bw_pool),
        "C1_literal_LEVEL_at_score_0": b0(c1, 0.0),
        "S_total_shipped": hs["sigma_mu"] ** 2 + hs["sigma_theta"] ** 2 + hs["sigma_delta"] ** 2
                           + hs["sigma_d"] ** 2 + hs["sigma_g"] ** 2,
        "S_total_C1": c1["sigma_mu"] ** 2 + c1["sigma_theta"] ** 2 + c1["sigma_delta"] ** 2
                      + c1["sigma_d"] ** 2 + c1["sigma_g"] ** 2,
    }
    # LEVEL re-expressed at T_pool so that its accuracy-scale quantiles (at score 0, item
    # residual only) are unchanged: each quantile mu -> f_inv(f_acc(mu, T_ship), T_pool)
    qs = {z: f_inv(f_acc(-2.5 + 2.5 * z, T_SHIP), T) for z in (-1.645, -1, 0, 1, 1.645)}
    out["LEVEL_mapped_to_T_pool"] = {"quantile_z": {str(k): round(v, 3) for k, v in qs.items()},
                                     "mu0": round(qs[0], 3),
                                     "sigma_mu_from_pm1": round((qs[1] - qs[-1]) / 2, 3)}

    # --- D: identity -------------------------------------------------------------------
    m = pool.meta
    th_clip = float(np.clip(m["tau2_res"], 0.01, 0.2))
    ws_clip, ws_raw, ks = [], [], []
    for f in multi.values():
        for s in f["subjects"]:
            obs = pool.identity(s, residual=True)
            if obs is None:
                continue
            u = pool.attribute(s)[1]
            ws_clip.append((th_clip + u) / (th_clip + u + obs[1]))
            ws_raw.append((m["tau2_res"] + u) / (m["tau2_res"] + u + obs[1]))
    kcount = defaultdict(int)
    for row in pool.table.values():
        kcount[min(int(row[3]), 5)] += 1
    out["identity"] = {
        "table_rows": len(pool.table), "rows_k_res_1_2_3_4_5plus": dict(sorted(kcount.items())),
        "tau2_res": m["tau2_res"], "tau2_res_alpha2": m["tau2_res_alpha"], "s2d_res": m["s2d_res"],
        "s2_res": m["s2_res"], "total_var": m["total_var"], "u_mean": m["u_mean"],
        "link_noise": m["link_noise"],
        "weights_rows_with_entry": len(ws_clip),
        "w_uncapped_clipped_tau_q0_10_50_90_100": [round(float(x), 3) for x in
                                                  np.percentile(ws_clip, [0, 10, 50, 90, 100])],
        "w_uncapped_raw_tau_q0_10_50_90_100": [round(float(x), 3) for x in
                                              np.percentile(ws_raw, [0, 10, 50, 90, 100])],
        "share_at_cap_0.1": float(np.mean(np.array(ws_clip) >= 0.1)),
    }
    # empirical shrinkage: replicate from_standings' residual cells (paiec/prior.py 613-643)
    coef, spec, X, Ai = PR._ridge(rows, 2.0)
    resid, _ = PR._residuals(rows, 2.0, X)
    cells = defaultdict(list)
    for i, (b, s, _, _) in enumerate(rows):
        k = PR.canon_name(s)
        if k and np.isfinite(resid[i]):
            cells[(k, b)].append(resid[i])
    cells = {kb: float(np.mean(v)) for kb, v in cells.items()}
    off = PR._offsets(cells)
    cells = {(k, b): v - off.get(b, 0.0) for (k, b), v in cells.items()}
    by_k = defaultdict(dict)
    for (k, b), v in cells.items():
        by_k[k][b] = v
    pts = defaultdict(list)
    for k, d in by_k.items():
        if len(d) < 2:
            continue
        for b, y in d.items():
            others = [v for bb, v in d.items() if bb != b]
            kk = len(others)
            pts["1" if kk == 1 else "2" if kk == 2 else "3-4" if kk <= 4 else "5-9" if kk <= 9 else "10+"].append(
                (np.mean(others), y))
    slopes = {}
    tau = m["tau2_res"]
    for g, xy in sorted(pts.items()):
        a = np.array(xy)
        x, y = a[:, 0] - a[:, 0].mean(), a[:, 1] - a[:, 1].mean()
        kmid = {"1": 1, "2": 2, "3-4": 3.5, "5-9": 7, "10+": 15}[g]
        slopes[g] = {"cells": len(a), "slope": float(x @ y / (x @ x)),
                     "corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                     "model_w_tau_raw": tau / (tau + m["s2d_res"] / kmid),
                     "model_w_tau_clip": th_clip / (th_clip + m["s2d_res"] / kmid)}
    out["identity"]["loo_shrinkage_by_other_benchmarks"] = slopes
    # benchmark jackknife of the identity components and residual variance
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=JOBS, initializer=_init, initargs=(rows,)) as ex:
        jk = dict(ex.map(jack, sorted(multi)))
    out["jackknife_wall_s"] = time.perf_counter() - t0
    out["from_standings_cpu_s_per_exclusion"] = float(np.median([v["cpu"] for v in jk.values()]))
    n = len(jk)
    for key in ("tau2_res", "s2d_res", "s2_res", "total_var"):
        vals = np.array([v[key] for v in jk.values()], float)
        full = m[key]
        se = math.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2))
        out.setdefault("jackknife", {})[key] = {"full": full, "se": se, "min": float(vals.min()),
                                                "max": float(vals.max()),
                                                "most_influential": sorted(
                                                    ((round(float(v[key] - full), 4), b) for b, v in jk.items()),
                                                    key=lambda t: -abs(t[0]))[:5]}
    out["levels_per_benchmark"] = {b: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in lev[b].items()}
                                   for b in names}
    out["wall_s"] = time.perf_counter() - t_start
    with open(os.path.join(HERE, "code_levels.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("T_pool", "rows", "mapped_minus_exact", "levels", "b0_prior_centre",
                                          "LEVEL_mapped_to_T_pool", "identity", "jackknife")},
                     indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
