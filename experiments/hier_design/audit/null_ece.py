"""Is hier's ~0.01 worse ECE-ALC a property of its calibration or of the binned ECE
estimator on small pairs? For each pair and budget, alongside the observed ECE,
compute the ECE expected if labels were drawn from the predictions themselves
(perfect calibration): the estimator's floor for that prediction vector."""
import os, sys, json, warnings
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import multiprocessing as mp
import numpy as np
import hier_eval as H
from paiec import official as O
from paiec.evaluator import WEIGHTS
from paiec.predict import fit_prior

_orig = O.ece
_log = []


def ece_logged(p, y, bins=10):
    p = np.asarray(p, float)
    rng = np.random.default_rng(12345)
    sims = [_orig(p, (rng.random(len(p)) < p).astype(float), bins) for _ in range(200)]
    _log.append(float(np.mean(sims)))
    return _orig(p, y, bins)


def one(args):
    i, name = args
    O.ece = ece_logged
    del _log[:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = H.draw("benchmark", 0, i)
        made = []
        res = O.run_official(run, H.factory(name, O.run_benchmarks(run), made), deepcopy=False,
                             workers=1, split_scope="pair")
    nb = 6
    null = np.array(_log).reshape(len(res["rows"]), nb)
    null_alc = null @ np.array(WEIGHTS)
    obs = np.array([r["ece_alc"] for r in res["rows"]])
    return i, name, obs.tolist(), null_alc.tolist()


def main(n=40):
    names = sorted({p.benchmark_id for p in H.pairs()})
    bundles = {("default", (b,)): H.build_bundle("default", (b,)) for b in names}
    pp = {(b,): fit_prior([p for p in H.pairs() if p.benchmark_id != b]) for b in names}
    tasks = [(i, m) for i in range(n) for m in (H.PRED, H.HIER)]
    out = {}
    with mp.get_context("spawn").Pool(4, initializer=H.init, initargs=(bundles, pp)) as pool:
        for i, name, obs, null in pool.imap_unordered(one, tasks):
            out[f"{i}|{name}"] = {"obs": obs, "null": null}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "null_ece.json"), "w"))
    d_obs, d_null = [], []
    for i in range(n):
        a, b = out[f"{i}|{H.HIER}"], out[f"{i}|{H.PRED}"]
        d_obs.append(np.mean(a["obs"]) - np.mean(b["obs"]))
        d_null.append(np.mean(a["null"]) - np.mean(b["null"]))
    d_obs, d_null = np.array(d_obs), np.array(d_null)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f"runs 0..{n-1}: ECE-ALC hier - Predictor observed {d_obs.mean():+.4f} ± {se(d_obs):.4f}; "
          f"under perfect calibration {d_null.mean():+.4f} ± {se(d_null):.4f}; "
          f"excess {np.mean(d_obs - d_null):+.4f} ± {se(d_obs - d_null):.4f}")
    for m in (H.PRED, H.HIER):
        o = np.mean([np.mean(out[f'{i}|{m}']['obs']) for i in range(n)])
        z = np.mean([np.mean(out[f'{i}|{m}']['null']) for i in range(n)])
        print(f"  {m}: observed ECE-ALC {o:.4f}, null {z:.4f}")


if __name__ == "__main__":
    main()
