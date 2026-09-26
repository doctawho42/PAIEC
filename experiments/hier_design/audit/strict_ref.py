"""Strict run-LOBO: how much of hier-R's lead rests on REFERENCE sigma_delta = 2.5,
the one fallback chosen with the public estimates in view (prior.py)? Re-score
hier run-LOBO on the runs where sigma_delta falls back, with the pre-review
REFERENCE value 1.0, and compare with the recorded rows."""
import os, sys, json, warnings
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import multiprocessing as mp
from dataclasses import replace
import numpy as np
import hier_eval as H
from paiec import official as O
from paiec.hier import HierPredictor

SD = float(os.environ.get("SD", "1.0"))


def one(i):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = H.draw("benchmark", 0, i)
        names = tuple(sorted(O.run_benchmarks(run)))
        prior, hyper = H._bundles[("default", names)]
        hyper = replace(hyper, sigma_delta=SD)
        res = O.run_official(run, lambda: HierPredictor(prior, hyper).predict, deepcopy=False,
                             workers=1, split_scope="pair")
    return i, [r["ALC"] for r in res["rows"]]


def main(n=150):
    raw = json.load(open(os.path.join(ROOT, "results", "hier_eval.json")))["raw"]
    ids, sets = [], set()
    for i in range(n):
        names = tuple(sorted(O.run_benchmarks(H.draw("benchmark", 0, i))))
        rep = H.fallback_report(names)
        if "sigma_delta" in rep["fallback"]:
            ids.append(i); sets.add(names)
    bundles = {("default", s): H.build_bundle("default", s) for s in sets}
    print(len(ids), "runs with sigma_delta at REFERENCE", flush=True)
    out = {}
    with mp.get_context("spawn").Pool(3, initializer=H.init, initargs=(bundles, {})) as pool:
        for i, alc in pool.imap_unordered(one, ids):
            out[i] = alc
    key = lambda i, n: H.task_key(("r1", "benchmark", "pair", 0, i, n))
    d_new = np.array([np.mean(out[i]) - np.mean(np.array(raw[key(i, H.HIER_RUN)]["rows"])[:, 6]) for i in ids])
    d_rp = np.array([np.mean(np.array(raw[key(i, H.HIER_RUN)]["rows"])[:, 6])
                     - np.mean(np.array(raw[key(i, H.PRED_RUN)]["rows"])[:, 6]) for i in ids])
    d_np = np.array([np.mean(out[i]) - np.mean(np.array(raw[key(i, H.PRED_RUN)]["rows"])[:, 6]) for i in ids])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f"{len(ids)} runs: hier-R(sigma_delta {SD}) - hier-R(2.5) {d_new.mean():+.5f} ± {se(d_new):.5f}")
    print(f"  hier-R(2.5) - Pred-R {d_rp.mean():+.5f} ± {se(d_rp):.5f};  hier-R({SD}) - Pred-R {d_np.mean():+.5f} ± {se(d_np):.5f}")
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"strict_ref_{SD}.json"), "w"))


if __name__ == "__main__":
    main()
