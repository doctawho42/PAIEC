"""The identity-transfer slopes on the selection half and the legacy five only
(amendment 4 final, R4). Descriptive: no predictor is scored, no Brier or ALC
is computed, no item text or formative feedback is read.

code.md section 5 regressed a name's residual cell on a held-out benchmark on
the mean of its other cells, over every benchmark. Here no cell of the
confirmation half enters anywhere: the standings rows are those of the subject
part U restricted to S plus the legacy five (multi-pair benchmarks only), the
ridge, the residual cells and the benchmark offsets are from_standings' on
those rows alone, held-out cells lie on S benchmarks, and the other cells on S
or the legacy five. The model weight is the identity weight formula at the
raw (unclipped) tau2_res, tau2 / (tau2 + s2d_res / k), with tau2_res and
s2d_res from the same rows, k the mean number of other cells in the bin.

Pre-registered criterion (amendment 4 final, section 4.4): C3 stays a
candidate iff the model weight is at or below the empirical slope in the bin
k = 1 and in the pooled bin k >= 3.

Fits: the prototype's (../fits_K100_M2000.pkl: K = 100 in subject-id order,
a benchmark-wide 2,000-item cap, items blind); benchmarks of S or the legacy
five that it lacks (mixed ones) are fitted here the same way and cached in
fits_extra.pkl. The `fits` stage recomputes all of this on the final fits.

Run:  PYTHONDONTWRITEBYTECODE=1 python slopes_s.py   -> slopes_s.json
"""
import json
import os
import pickle
import sys
from collections import defaultdict

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PD)))
sys.path.insert(0, HERE)
sys.path.insert(0, PD)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from paiec import prior as PR  # noqa: E402
import split_v2 as SV  # noqa: E402

LEGACY5 = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
FLAGGED = {"crow", "emoji_attack", "harmbench", "jailbreakbench", "llmail_inject", "wmt_mqm",
           "alignment_faking", "safeagentbench", "tensortrust"}


def get_fits(need):
    fits = pickle.load(open(os.path.join(PD, "fits_K100_M2000.pkl"), "rb"))
    cache = os.path.join(HERE, "fits_extra.pkl")
    extra = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    missing = [b for b in need if b not in fits and b not in extra]
    if missing:
        from proto_build import fit_one
        for b in missing:
            _, f = fit_one(b, 100, None, 2000)
            extra[b] = f
            print("fitted", b, None if not isinstance(f, dict) else (f["n_pairs"], f["iterations"]), flush=True)
        with open(cache, "wb") as fh:
            pickle.dump(extra, fh)
    fits.update({b: f for b, f in extra.items() if isinstance(f, dict)})
    return fits


def rows_of(fits, names):
    rows = []
    for b in sorted(names):
        f = fits.get(b)
        if not isinstance(f, dict) or f["n_pairs"] < 2:
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


def bins(cells, held):
    by_k = defaultdict(dict)
    for (k, b), v in cells.items():
        by_k[k][b] = v
    pts = defaultdict(list)
    for k, d in by_k.items():
        for b, y in d.items():
            if b not in held:
                continue
            others = [v for bb, v in d.items() if bb != b]
            kk = len(others)
            if not kk:
                continue
            for g in (("1",) if kk == 1 else ("2",) if kk == 2 else ("3+", "3-4") if kk <= 4 else ("3+", "5+")):
                pts[g].append((np.mean(others), y, kk))
    return pts


def main():
    out = {"what": __doc__.split("\n")[0]}
    for rule in ("bench", "family"):
        half = SV.split(rule)
        S = sorted(b for b in SV.E if half[b] == "S")
        names = [b for b in S + LEGACY5 if b not in FLAGGED]
        fits = get_fits(names)
        rows = rows_of(fits, names)
        p = PR.from_standings(rows, 2.0)
        m = p.meta
        tau, s2d = float(m["tau2_res"]), float(m["s2d_res"])
        cells = residual_cells(rows)
        pts = bins(cells, set(S))
        res = {}
        for g in ("1", "2", "3-4", "5+", "3+"):
            if len(pts.get(g, [])) < 3:
                res[g] = {"cells": len(pts.get(g, []))}
                continue
            a = np.array(pts[g])
            x, y = a[:, 0] - a[:, 0].mean(), a[:, 1] - a[:, 1].mean()
            kbar = float(a[:, 2].mean())
            res[g] = {"cells": int(len(a)), "names": None, "slope": round(float(x @ y / (x @ x)), 3),
                      "corr": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3), "k_mean": round(kbar, 2),
                      "model_w_raw_tau": round(tau / (tau + s2d / kbar), 3) if tau > 0 else 0.0,
                      "model_w_clip_tau": round(float(np.clip(tau, 0.01, 0.2)) /
                                                (float(np.clip(tau, 0.01, 0.2)) + s2d / kbar), 3)}
        keep = all(g in res and "slope" in res[g] and res[g]["model_w_raw_tau"] <= res[g]["slope"]
                   for g in ("1", "3+"))
        # bootstrap over names of the two slopes, for the record (seed 0, 2,000 resamples)
        rng = np.random.default_rng(0)
        boot = {}
        for g in ("1", "3+"):
            a = np.array(pts.get(g, []))
            if len(a) < 3:
                continue
            sl = []
            for _ in range(2000):
                i = rng.integers(0, len(a), len(a))
                x, y = a[i, 0] - a[i, 0].mean(), a[i, 1] - a[i, 1].mean()
                if x @ x > 0:
                    sl.append(float(x @ y / (x @ x)))
            boot[g] = [round(float(q), 3) for q in np.percentile(sl, [2.5, 50, 97.5])]
        out[rule] = {"S": len(S), "benchmarks_with_rows": len({r[0] for r in rows}), "rows": len(rows),
                     "linked": m["linked"], "tau2_res": round(tau, 4), "s2d_res": round(s2d, 4),
                     "s2_res": round(float(m["s2_res"]), 4), "bins": res, "slope_boot_cells_95": boot,
                     "criterion_C3_kept": keep}
    with open(os.path.join(HERE, "slopes_s.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
