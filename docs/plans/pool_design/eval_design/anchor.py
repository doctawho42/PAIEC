"""How far a refitted attribute ridge moves the effective level of a fixed mu0
(eval.md section 2, candidate C2; data.md section 7, "LEVEL is a level at
attribute score 0"). Descriptive: no predictor is scored, no Brier or ALC is
computed, no label is read beyond what the prototype's standings already hold,
no formative feedback and no item text is read.

hier adds attr_scale times SubjectPrior.attribute(subject)[0] to a subject's
prior mean (paiec/hier.py:1902-1903), so the effective prior centre of a pair
is mu0 + attr_scale * m_s. This compares m_s under archive-3's subject prior
(submission/prior.json) and under the interrupted attempt's prototype pool
prior (from_standings on fits_K100_M2000.pkl, as proto_build.py stage 2 does)
on the subject dicts of the evaluation set E (132 benchmarks: binary, or mixed
with a 0/1 share of at least 0.95, legacy six excluded; pairs.csv), weighting
each benchmark once and its pairs equally (the weighting of benchmark-first
runs), and by pair.

The prototype is not the candidate's rest (binary only, K = 100, M = 2,000, no
direction filter, no items); the figure is for planning only.

Run from anywhere:  PYTHONDONTWRITEBYTECODE=1 python anchor.py  -> anchor.json
"""
import csv
import json
import os
import pickle
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(D)))
sys.path.insert(0, ROOT)
from paiec import prior as PR  # noqa: E402

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]
LEGACY = {"matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench", "mmdocrag"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    warnings.simplefilter("ignore")
    a3_prior, a3_hyper = PR.load(os.path.join(ROOT, "submission", "prior.json"))
    fits = pickle.load(open(os.path.join(D, "fits_K100_M2000.pkl"), "rb"))
    rows = []
    for b in sorted(fits):
        fb = fits[b]
        if fb is None or fb.get("n_pairs", 0) < 2:
            continue
        t = np.array(fb["t"])
        for s, z, v in zip(fb["subjects"], t - t.mean(), fb["t_var"]):
            rows.append((b, s, float(z), float(v)))
    pool_prior = PR.from_standings(rows, 2.0, sorted(fits), ())
    B = {r["benchmark"]: r for r in csv.DictReader(open(os.path.join(D, "benchmarks.csv")))}
    E = {b for b, r in B.items() if f(r["eligible_pairs"]) > 0 and b not in LEGACY
         and (r["response_type"] == "binary"
              or (r["response_type"] == "mixed" and f(r["share01"]) >= 0.95))}
    by = {}
    for r in csv.DictReader(open(os.path.join(D, "pairs.csv"))):
        if r["benchmark"] in E:
            by.setdefault(r["benchmark"], []).append({k: (r[k] or "") for k in FIELDS})
    out = {"what": __doc__.split("\n")[0], "benchmarks": len(by),
           "pairs": sum(len(v) for v in by.values())}
    for name, pr in (("archive3", a3_prior), ("pool_prototype", pool_prior)):
        per_b = {b: [pr.attribute(s)[0] for s in ss] for b, ss in by.items()}
        bm = np.array([np.mean(v) for v in per_b.values()])
        allv = np.concatenate([np.array(v) for v in per_b.values()])
        named = np.array([pr.attribute(s)[0] for ss in by.values() for s in ss if PR.canon_name(s)])
        unnamed = np.array([pr.attribute(s)[0] for ss in by.values() for s in ss if not PR.canon_name(s)])
        out[name] = {"benchmark_weighted_mean": float(bm.mean()), "sd_of_benchmark_means": float(bm.std()),
                     "pair_weighted_mean": float(allv.mean()), "pair_sd": float(allv.std()),
                     "named_pair_mean": float(named.mean()), "unnamed_pair_mean": float(unnamed.mean()),
                     "named_pairs": int(named.size), "unnamed_pairs": int(unnamed.size)}
    a, p = out["archive3"], out["pool_prototype"]
    out["shift_of_effective_level_at_attr_scale_0.5"] = {
        "benchmark_weighted": 0.5 * (p["benchmark_weighted_mean"] - a["benchmark_weighted_mean"]),
        "pair_weighted": 0.5 * (p["pair_weighted_mean"] - a["pair_weighted_mean"]),
        "note": "pool prototype minus archive-3: the effective centre mu0 + 0.5 m moves by this when "
                "LEVEL's mu0 -2.5 is kept over the refitted ridge; C2 subtracts it from mu0"}
    out["archive3_hyper"] = {k: v for k, v in a3_hyper.to_dict().items() if k not in ("excluded", "included")}
    out["pool_prototype_meta"] = {k: pool_prior.meta[k] for k in ("total_var", "s2_res", "tau2_res", "linked")}
    with open(os.path.join(HERE, "anchor.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
