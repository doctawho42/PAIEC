"""The predictor ladder with a spread over evaluation splits.

Every number in docs/findings.md that is an ALC comes from here. The split seed
matters: a single measurement carries sd around 0.0015, so differences smaller
than about 0.004 are not readable from one seed.

Run: python experiments/ladder.py --seeds 4
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


def main(seeds=4):
    pairs = load_pairs()
    zh, by_b, keys_of, emb = pipeline.build(pairs)
    res = {k: [] for k in ["constant", "official", "smoothed", "pool+prior", "pool only"]}
    t0 = time.time()
    for seed in range(seeds):
        res["constant"].append(E.run_session(pairs, B.const(0.5), seed=seed)[0]["ALC"])
        res["official"].append(E.run_session(pairs, B.empirical_mean, seed=seed)[0]["ALC"])
        res["smoothed"].append(E.run_session(pairs, B.smoothed_mean(4.0, 0.5), seed=seed)[0]["ALC"])
        traj = collect_trajectories(pairs, B.const(0.5), seed=seed)
        res["pool+prior"].append(ALC(pipeline.score(pairs, traj, zh, by_b, keys_of, emb, seed=seed)))
        flat = {k: 0.0 for k in zh}
        res["pool only"].append(ALC(pipeline.score(pairs, traj, flat, by_b, keys_of, emb, seed=seed)))
        print(f"  seed {seed} [{time.time() - t0:.0f}s]", flush=True)
    print(f"\n{'predictor':<16}{'mean ALC':>10}{'sd':>9}   per-seed")
    for k, v in res.items():
        sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
        print(f"{k:<16}{np.mean(v):>10.4f}{sd:>9.4f}   " + " ".join(f"{x:.4f}" for x in v))
    a, b = np.array(res["pool only"]), np.array(res["pool+prior"])
    print(f"\npaired gain of the attribute prior: {(a - b).mean():+.4f} "
          f"+- {(a - b).std(ddof=1):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4)
    main(**vars(ap.parse_args()))
