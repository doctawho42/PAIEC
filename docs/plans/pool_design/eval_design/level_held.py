"""Planning values for candidate C2's level rule (eval.md section 2): the mu0 and
sigma_mu that keep archive-3's level distribution on the accuracy-logit scale
when the rest of the prior is refitted. Descriptive arithmetic on two bundles'
hyperparameters and mean attribute scores; no predictor is run, no Brier or
ALC computed, no label, item text or formative feedback read.

a(z) = logit E[sigmoid(mu0 + sigma_mu z + attr_scale m + sqrt(V) e)], e ~ N(0, 1)
(80-node Gauss-Hermite), V = sigma_theta^2 + sigma_delta^2 + sigma_d^2 + sigma_g^2
(theta's prior variance without the attribute uncertainty u, plus the item
variance hier gives a benchmark without group keys, paiec/hier.py:1955-1959),
m the benchmark-weighted mean attribute score of the reference pairs.
C2 solves a_C2(0) = a_A3(0) and a_C2(1) - a_C2(-1) = a_A3(1) - a_A3(-1).

Inputs: archive-3's hyperparameters (submission/prior.json, via anchor.json),
the prototype's fitted hyperparameters (../build_K100_M2000.json,
hyper_fitted), and the mean scores of anchor.json (E's pairs) and of the code
task's reference set (../code/code_variants.json, b0_centres). The prototype is
not the candidate's rest; these are planning figures only.
Run: PYTHONDONTWRITEBYTECODE=1 python level_held.py -> level_held.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
X, W = np.polynomial.hermite_e.hermegauss(80)
W = W / W.sum()


def acc(c, V):
    p = float(np.sum(W / (1 + np.exp(-(c + math.sqrt(V) * X)))))
    return math.log(p / (1 - p))


def bisect(target, f, lo, hi):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    anchor = json.load(open(os.path.join(HERE, "anchor.json")))
    a3 = anchor["archive3_hyper"]
    proto = json.load(open(os.path.join(D, "build_K100_M2000.json")))["hyper_fitted"]
    VA = a3["sigma_theta"] ** 2 + a3["sigma_delta"] ** 2 + a3["sigma_d"] ** 2 + a3["sigma_g"] ** 2
    VP = proto["sigma_theta"] ** 2 + proto["sigma_delta"] ** 2 + proto["sigma_d"] ** 2 + proto["sigma_g"] ** 2
    refs = {"E_pairs_benchmark_weighted": (anchor["archive3"]["benchmark_weighted_mean"],
                                           anchor["pool_prototype"]["benchmark_weighted_mean"])}
    cv = os.path.join(D, "code", "code_variants.json")
    if os.path.exists(cv):
        b = json.load(open(cv)).get("b0_centres", {}).get("pool_benchmark_weighted")
        if b:
            refs["code_task_pool_benchmark_weighted"] = (b["mean_score_shipped"], b["mean_score_pool"])
    out = {"what": __doc__.split("\n")[0], "V_archive3": VA, "V_prototype": VP, "refs": {}}
    for name, (mA, mP) in refs.items():
        tgt = {z: acc(a3["mu0"] + a3["sigma_mu"] * z + a3["attr_scale"] * mA, VA) for z in (-1, 0, 1)}
        lit = {z: acc(a3["mu0"] + a3["sigma_mu"] * z + a3["attr_scale"] * mP, VP) for z in (-1, 0, 1)}
        mu0 = bisect(tgt[0], lambda m: acc(m + a3["attr_scale"] * mP, VP), -30.0, 30.0)
        sm = bisect(tgt[1] - tgt[-1], lambda s: acc(mu0 + s + a3["attr_scale"] * mP, VP)
                    - acc(mu0 - s + a3["attr_scale"] * mP, VP), 0.01, 20.0)
        out["refs"][name] = {"mean_score_archive3": mA, "mean_score_prototype": mP,
                             "archive3_accuracy_quantiles_z_m1_0_p1": [tgt[z] for z in (-1, 0, 1)],
                             "C1_literal_accuracy_quantiles_z_m1_0_p1": [lit[z] for z in (-1, 0, 1)],
                             "C2_mu0": mu0, "C2_sigma_mu": sm}
    with open(os.path.join(HERE, "level_held.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
