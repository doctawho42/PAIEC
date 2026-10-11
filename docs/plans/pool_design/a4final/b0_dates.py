"""The B0 prior centre of each arm at real dates and at +1.25 years
(amendment 4 final, R1). A property of the priors: no predictor is scored, no
Brier or ALC is computed, no label is read beyond what the prototype's
standings already hold, no item text and no formative feedback is read.

The functional (amendment 4 final, section 4.4, "the B0 centre"):
  c(X, Sigma) = sum over the benchmarks b of Sigma of (1/|B|) times the mean
  over b's pairs s of (1 - slip) E_z[sigmoid(mu0 + m_X(s)
  + sqrt(sigma_mu^2 + V_X(s) + sigma_delta^2 + T) z)],
with (m_X(s), V_X(s)) X's theta prior without the identity term
(shift + attr_scale * attribute mean; sigma_theta^2 + the ridge's prediction
variance u), T = sigma_d^2 + sigma_g^2, z standard normal (80-point
Gauss-Hermite). No floor, no group key, no text term.

Sigma: the subject dicts (eight visible fields) of the eligible pairs of E
(pairs.csv), each benchmark once; at real dates and with release and access
dates moved by 1.25 years (paiec.testlike.shift_date, RS's date_shift); all
pairs, dated pairs and undated pairs.

Arms (planning, from the prototype fits; the `fits` stage recomputes them on
the final fits): A3 = submission/prior.json; C1 = the pool subject prior on
U's prototype fits (flagged benchmarks out; ../fits_K100_M2000.pkl plus
fits_extra.pkl) with fit_hyper's arithmetic at share 0.25 and LEVEL written
over; C2 = C1 with the prototype's EB level (code_variants.json); C3 = C1 with
th2 unclipped and id_cap 1.0 (dropped by R4, shown for the record); C1m = C1
with mu0 chosen so that c(C1m, E at +1.25 years) = c(A3, E at +1.25 years).

Run:  PYTHONDONTWRITEBYTECODE=1 python b0_dates.py   -> b0_dates.json
"""
import csv
import json
import math
import os
import pickle
import sys

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PD)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from paiec import prior as PR  # noqa: E402
from paiec.testlike import shift_date  # noqa: E402
import split_v2 as SV  # noqa: E402

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]
FLAGGED = {"crow", "emoji_attack", "harmbench", "jailbreakbench", "llmail_inject", "wmt_mqm",
           "alignment_faking", "safeagentbench", "tensortrust"}
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / GH_W.sum()
YEARS = 1.25


def centre_one(h, m, V):
    S = h["sigma_mu"] ** 2 + V + h["sigma_delta"] ** 2 + h["sigma_d"] ** 2 + h["sigma_g"] ** 2
    return (1 - h["slip"]) * float(np.sum(GH_W / (1 + np.exp(-(h["mu0"] + m + math.sqrt(S) * GH_X)))))


def theta_noid(prior, h, s):
    ma, u = prior.attribute(s)
    return h["shift"] + h["attr_scale"] * ma, h["sigma_theta"] ** 2 + u


def centre(prior, h, by, which):
    vals = []
    for b, subs in by.items():
        ss = [s for s in subs if which == "all" or (which == "dated") == bool(s["release_date"])]
        if not ss:
            continue
        vals.append(np.mean([centre_one(h, *theta_noid(prior, h, s)) for s in ss]))
    return float(np.mean(vals)), len(vals)


def mean_score(prior, by, which):
    vals = []
    for b, subs in by.items():
        ss = [s for s in subs if which == "all" or (which == "dated") == bool(s["release_date"])]
        if ss:
            vals.append(np.mean([prior.attribute(s)[0] for s in ss]))
    return float(np.mean(vals))


def shifted(s):
    t = dict(s)
    for f in ("release_date", "access_date"):
        t[f] = shift_date(t[f], YEARS) if t[f] else t[f]
    return t


def pool_bundle():
    fits = pickle.load(open(os.path.join(PD, "fits_K100_M2000.pkl"), "rb"))
    ex = os.path.join(HERE, "fits_extra.pkl")
    if os.path.exists(ex):
        fits.update({b: f for b, f in pickle.load(open(ex, "rb")).items() if isinstance(f, dict)})
    rows = []
    for b in sorted(fits):
        f = fits[b]
        if f["n_pairs"] < 2 or b in FLAGGED:
            continue
        t = np.array(f["t"])
        for s, z, v in zip(f["subjects"], t - t.mean(), f["t_var"]):
            rows.append((b, s, float(z), float(v)))
    prior = PR.from_standings(rows, 2.0, sorted(fits), ())
    m = prior.meta
    T = float(np.median([f["item_var"] for f in fits.values() if f["n_pairs"] >= 2]))
    base = {"shift": 0.0, "slip": 0.01, "sigma_d": math.sqrt(0.75 * T), "sigma_g": math.sqrt(0.25 * T),
            "sigma_attr": math.sqrt(max(m["total_var"] - m["s2_res"], 0.0)), "attr_scale": 0.5}
    hs = {}
    for name, th2, cap in (("C1", float(np.clip(m["tau2_res"], 0.01, 0.2)), 0.1),
                           ("C3", max(float(m["tau2_res"]), 0.01), 1.0)):
        hs[name] = {**base, "sigma_theta": math.sqrt(th2), "id_cap": cap, "mu0": -2.5, "sigma_mu": 2.5,
                    "sigma_delta": math.sqrt(max(m["s2_res"] - m["u_mean"] - th2, 0.1)) * PR.WIDEN[1]}
    cv = json.load(open(os.path.join(PD, "code", "code_variants.json")))["exact_levels_T_pool"]["all_127"]
    n = cv["n"]
    hs["C2"] = {**hs["C1"], "mu0": cv["eb_mean"], "sigma_mu": cv["eb_sd"] * math.sqrt(1 + 1 / n)}
    meta = {"rows": len(rows), "T": T, "tau2_res": m["tau2_res"], "date_coef_per_400d": float(prior.coef[1]),
            "intercept": float(prior.coef[0])}
    return prior, hs, meta


def main():
    a3p, a3h = PR.load(os.path.join(ROOT, "submission", "prior.json"))
    a3 = a3h.to_dict()
    pool, hs, meta = pool_bundle()
    by_real, by_ds = {}, {}
    for r in csv.DictReader(open(os.path.join(PD, "pairs.csv"))):
        if r["benchmark"] in SV.E:
            s = {k: (r[k] or "") for k in FIELDS}
            by_real.setdefault(r["benchmark"], []).append(s)
            by_ds.setdefault(r["benchmark"], []).append(shifted(s))
    arms = {"A3": (a3p, a3), "C1": (pool, hs["C1"]), "C2": (pool, hs["C2"]), "C3": (pool, hs["C3"])}
    # C1m: mu0 so that C1m's centre on E at +1.25 years equals A3's there
    target = centre(a3p, a3, by_ds, "all")[0]
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if centre(pool, {**hs["C1"], "mu0": mid}, by_ds, "all")[0] < target:
            lo = mid
        else:
            hi = mid
    hs["C1m"] = {**hs["C1"], "mu0": 0.5 * (lo + hi)}
    arms["C1m"] = (pool, hs["C1m"])
    # the same rule matched on dated pairs only, for the record
    target_d = centre(a3p, a3, by_ds, "dated")[0]
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if centre(pool, {**hs["C1"], "mu0": mid}, by_ds, "dated")[0] < target_d:
            lo = mid
        else:
            hi = mid
    mu0_dated = 0.5 * (lo + hi)
    tab = {}
    for name, (pr, h) in arms.items():
        for dk, by in (("real", by_real), ("plus_1.25y", by_ds)):
            for which in ("all", "dated", "undated"):
                c, nb = centre(pr, h, by, which)
                tab.setdefault(name, {})[f"{dk}:{which}"] = round(c, 4)
                tab[name][f"{dk}:{which}:benchmarks"] = nb
        tab[name]["mean_score_real_all"] = round(mean_score(pr, by_real, "all"), 3)
        tab[name]["mean_score_ds_all"] = round(mean_score(pr, by_ds, "all"), 3)
        tab[name]["hyper"] = {k: round(v, 4) for k, v in h.items()
                              if k in ("mu0", "sigma_mu", "sigma_theta", "sigma_delta", "sigma_d",
                                       "sigma_g", "id_cap", "attr_scale")}
    diff = {name: {k: round(tab[name][k] - tab["A3"][k], 4)
                   for k in tab["A3"] if not k.endswith("benchmarks") and isinstance(tab["A3"][k], float)}
            for name in ("C1", "C1m", "C2", "C3")}
    out = {"what": __doc__.split("\n")[0], "E_benchmarks": len(by_real),
           "pairs": sum(len(v) for v in by_real.values()),
           "dated_pairs": sum(bool(s["release_date"]) for v in by_real.values() for s in v),
           "a3_date_coef_per_400d": float(a3p.coef[1]), "pool_meta": meta,
           "centres": tab, "minus_A3": diff, "C1m_mu0_matched_all": round(hs["C1m"]["mu0"], 3),
           "C1m_mu0_if_matched_on_dated_only": round(mu0_dated, 3), "tolerance": 0.024}
    with open(os.path.join(HERE, "b0_dates.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "centres"}, indent=1))
    for name in tab:
        print(name, {k: v for k, v in tab[name].items() if ":" in k and not k.endswith("benchmarks")})


if __name__ == "__main__":
    main()
