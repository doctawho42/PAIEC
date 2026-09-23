"""Does any acquisition policy beat the evaluator's random one? No.

Informed A-optimal selection is worse: it concentrates labels on informative
items and starves the shared difficulty estimate of distinct items. Coverage-first
fixes the coverage and still does not move ALC, so coverage is not the binding
constraint either.

Run: python experiments/acquisition.py --seeds 4
"""
import argparse
import time

import numpy as np

from paiec import baselines as B
from paiec import evaluator as E
from paiec import pipeline
from paiec.data import load_pairs
from paiec.trajectories import collect_trajectories

ALC = lambda d: sum(w * d[b] for w, b in zip(E.WEIGHTS, E.BUDGETS))


def main(seeds=4, gamma=0.5):
    pairs = load_pairs()
    zh, by_b, keys_of, emb = pipeline.build(pairs)
    res = {k: [] for k in ["random", "informed", "coverage"]}
    cov = {}
    t0 = time.time()
    for seed in range(seeds):
        base = collect_trajectories(pairs, B.const(0.5), seed=seed)
        res["random"].append(ALC(pipeline.score(pairs, base, zh, by_b, keys_of, emb, seed=seed)))
        inf = pipeline.informed_traj(pairs, base, zh, by_b, keys_of, emb, gamma=gamma, seed=seed)
        res["informed"].append(ALC(pipeline.score(pairs, inf, zh, by_b, keys_of, emb, seed=seed)))
        cvt = pipeline.coverage_traj(pairs, base, zh, by_b, keys_of, emb, seed=seed)
        res["coverage"].append(ALC(pipeline.score(pairs, cvt, zh, by_b, keys_of, emb, seed=seed)))
        if seed == 0:
            for name, t in [("random", base), ("informed", inf), ("coverage", cvt)]:
                cov[name] = {bid: len({k for p in ps for k, _ in t[(p.subject_id, p.benchmark_id)][:31]})
                             for bid, ps in by_b.items()}
        print(f"  seed {seed} [{time.time() - t0:.0f}s]", flush=True)
    print(f"\n{'policy':<12}{'mean ALC':>10}{'sd':>9}   per-seed")
    for k, v in res.items():
        sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
        print(f"{k:<12}{np.mean(v):>10.4f}{sd:>9.4f}   " + " ".join(f"{x:.4f}" for x in v))
    r = np.array(res["random"])
    for k in ["informed", "coverage"]:
        d = r - np.array(res[k])
        print(f"paired gain of {k} over random: {d.mean():+.4f} +- {d.std(ddof=1):.4f}")
    print("\ndistinct items the pool touched at budget 31, seed 0:")
    for bid in cov.get("random", {}):
        print(f"  {bid:<20}" + "  ".join(f"{k}={cov[k][bid]}" for k in cov))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.5)
    main(**vars(ap.parse_args()))
