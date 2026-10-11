"""Follow-ups to code_levels.py for code.md (descriptive; scores nothing).

  1. from_standings' meta (the inputs of sigma_theta, sigma_delta, sigma_attr)
     and the ridge's date coefficient on the prototype's standings with and
     without the nine direction-flagged benchmarks (data.md section 2);
  2. the leave-one-benchmark-out shrinkage slope of a name's residual cells
     when the other cells exclude the benchmark's own suite (HELM, MathArena,
     IgakuQA siblings), next to the slope over all other cells;
  3. the exact level of every prototype benchmark at s2e = T_pool (the
     fixed-item-variance pass a pool level would need), its EB summary, with
     and without the flagged benchmarks;
  4. the B0 prior predictive centre at attribute score m for several reference
     subject sets, under archive-3's bundle, C1 with LEVEL kept literally and
     C1 with LEVEL re-expressed to keep its accuracy-scale quantiles.
"""
import concurrent.futures as cf
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PD)))
sys.path.insert(0, PD)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from paiec import prior as PR  # noqa: E402
from code_levels import f_acc, f_inv, eb, refit_fixed, GH_W, GH_X, T_SHIP  # noqa: E402

FLAGGED = {"crow", "emoji_attack", "harmbench", "jailbreakbench", "llmail_inject", "wmt_mqm",
           "alignment_faking", "safeagentbench", "tensortrust"}


def family(b):
    if re.match(r"^(med|v)?helm", b):
        return "helm"
    if b.startswith("matharena"):
        return "matharena"
    if b.startswith("igakuqa"):
        return "igakuqa"
    return b


def rows_of(fits, drop=()):
    rows = []
    for b in sorted(fits):
        f = fits[b]
        if f["n_pairs"] < 2 or b in drop:
            continue
        t = np.array(f["t"])
        for s, z, v in zip(f["subjects"], t - t.mean(), f["t_var"]):
            rows.append((b, s, float(z), float(v)))
    return rows


def residual_cells(rows):
    coef, spec, X, Ai = PR._ridge(rows, 2.0)
    resid, _ = PR._residuals(rows, 2.0, X)
    cells = defaultdict(list)
    for i, (b, s, _, _) in enumerate(rows):
        k = PR.canon_name(s)
        if k and np.isfinite(resid[i]):
            cells[(k, b)].append(resid[i])
    cells = {kb: float(np.mean(v)) for kb, v in cells.items()}
    off = PR._offsets(cells)
    return {(k, b): v - off.get(b, 0.0) for (k, b), v in cells.items()}


def slopes(cells, exclude_family):
    by_k = defaultdict(dict)
    for (k, b), v in cells.items():
        by_k[k][b] = v
    xy = []
    for k, d in by_k.items():
        for b, y in d.items():
            others = [v for bb, v in d.items() if bb != b and (not exclude_family or family(bb) != family(b))]
            if others:
                xy.append((np.mean(others), y, len(others)))
    a = np.array(xy)
    out = {}
    for name, sel in (("all", np.ones(len(a), bool)), ("k1", a[:, 2] == 1), ("k2-4", (a[:, 2] >= 2) & (a[:, 2] <= 4)),
                      ("k5+", a[:, 2] >= 5)):
        x, y = a[sel, 0] - a[sel, 0].mean(), a[sel, 1] - a[sel, 1].mean()
        out[name] = {"cells": int(sel.sum()), "slope": round(float(x @ y / (x @ x)), 3)}
    return out


def meta_of(p):
    m = p.meta
    keys = ("n_rows", "total_var", "s2_res", "u_mean", "tau2_res", "s2d_res", "linked")
    out = {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in keys}
    th2 = float(np.clip(m["tau2_res"], 0.01, 0.2))
    out["sigma_delta"] = round(math.sqrt(max(m["s2_res"] - m["u_mean"] - th2, 0.1)) * PR.WIDEN[1], 4)
    out["sigma_attr"] = round(math.sqrt(max(m["total_var"] - m["s2_res"], 0.0)), 4)
    out["coef_intercept_date_hasdate"] = [round(float(c), 3) for c in p.coef[:3]]
    out["providers"] = len(p.spec.providers)
    return out


def b0(h, m):
    S = h["sigma_mu"] ** 2 + h["sigma_theta"] ** 2 + h["sigma_delta"] ** 2 + h["sigma_d"] ** 2 + h["sigma_g"] ** 2
    return (1 - h["slip"]) * float(np.sum(GH_W / (1 + np.exp(-(h["mu0"] + h["attr_scale"] * m
                                                              + math.sqrt(S) * GH_X)))))


def main():
    t0 = time.perf_counter()
    fits = pickle.load(open(os.path.join(PD, "fits_K100_M2000.pkl"), "rb"))
    multi = {b: f for b, f in fits.items() if f["n_pairs"] >= 2}
    T = float(np.median([f["item_var"] for f in multi.values()]))
    out = {"T_pool": T}
    # 1 and 2
    rows_all, rows_ord = rows_of(fits), rows_of(fits, FLAGGED)
    p_all = PR.from_standings(rows_all, 2.0)
    p_ord = PR.from_standings(rows_ord, 2.0)
    out["subject_part"] = {"prototype_all_binary": meta_of(p_all), "without_flagged": meta_of(p_ord)}
    for tag, rows in (("prototype_all_binary", rows_all), ("without_flagged", rows_ord)):
        cells = residual_cells(rows)
        out.setdefault("loo_slopes", {})[tag] = {"other_cells_any_benchmark": slopes(cells, False),
                                                 "other_cells_outside_own_suite": slopes(cells, True)}
    # 3: exact levels at T_pool for every benchmark
    with cf.ProcessPoolExecutor(max_workers=4) as ex:
        exact = dict(ex.map(refit_fixed, [(b, T) for b in sorted(fits)]))
    out["exact_pass_cpu_s"] = sum(v["cpu"] for v in exact.values())
    out["exact_pass_not_converged"] = sorted(b for b, v in exact.items() if not v["converged"])
    lev = {}
    for b, e in exact.items():
        ma = float(np.mean([p_all.attribute(s)[0] for s in fits[b]["subjects"]]))
        lev[b] = (e["mu"] + e["tbar"] - ma, e["mu_var"])
    for tag, keep in (("all_127", sorted(lev)), ("without_flagged", sorted(b for b in lev if b not in FLAGGED)),
                      ("flagged_only", sorted(b for b in lev if b in FLAGGED))):
        out.setdefault("exact_levels_T_pool", {})[tag] = eb([lev[b][0] for b in keep], [lev[b][1] for b in keep])
    out["exact_levels_per_benchmark"] = {b: [round(v[0], 3), round(v[1], 4)] for b, v in sorted(lev.items())}
    # 4: B0 centres
    ship, ship_h = PR.load(os.path.join(ROOT, "submission", "prior.json"))
    proto = json.load(open(os.path.join(PD, "build_K100_M2000.json")))["hyper_fitted"]
    hs = ship_h.to_dict()
    c1 = {**proto, "mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
    qz = {z: f_inv(f_acc(-2.5 + 2.5 * z, T_SHIP), T) for z in (-1, 0, 1)}
    c1m = {**c1, "mu0": qz[0], "sigma_mu": (qz[1] - qz[-1]) / 2}
    refs = {}
    for b in ("matharena", "multi_swebench", "real_webagents", "researchcodebench"):
        from proto_loader import load_benchmark
        subs = fits[b]["subjects"] if b in fits else [p.subject for p in load_benchmark(b)]
        refs[f"legacy:{b}"] = subs
    refs["pool_benchmark_weighted"] = None
    tab = {}
    for name, subs in refs.items():
        if subs is None:
            ms = np.mean([np.mean([ship.attribute(s)[0] for s in f["subjects"]]) for f in fits.values()])
            mp = np.mean([np.mean([p_all.attribute(s)[0] for s in f["subjects"]]) for f in fits.values()])
        else:
            ms = np.mean([ship.attribute(s)[0] for s in subs])
            mp = np.mean([p_all.attribute(s)[0] for s in subs])
        # the mu0 under C1's other hyperparameters that gives archive-3's B0 centre here
        target = b0(hs, ms)
        lo, hi = -15.0, 15.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if b0({**c1, "mu0": mid}, mp) < target else (lo, mid)
        tab[name] = {"mean_score_shipped": round(float(ms), 3), "mean_score_pool": round(float(mp), 3),
                     "b0_archive3": round(target, 4), "b0_C1_literal": round(b0(c1, mp), 4),
                     "b0_C1_LEVEL_mapped_to_T_pool": round(b0(c1m, mp), 4),
                     "mu0_matching_archive3_b0_under_C1": round(0.5 * (lo + hi), 3)}
    out["b0_centres"] = tab
    out["C1_LEVEL_mapped"] = {"mu0": round(c1m["mu0"], 3), "sigma_mu": round(c1m["sigma_mu"], 3)}
    out["wall_s"] = time.perf_counter() - t0
    with open(os.path.join(HERE, "code_variants.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "exact_levels_per_benchmark"}, indent=1, default=str))


if __name__ == "__main__":
    main()
