"""The R1 runs of experiments/official_baselines.py, reduced to what a
count-based predictor sees.

Runs are rebuilt from default_rng([0, i]) exactly as official_baselines.r1
draws them and pushed once through paiec.official.run_official with a constant
predictor (the random policy does not look at predictions, so the acquired
labels are the ones every predictor gets). Per pair: benchmark, subject, the
revealed labels in order, and the evaluation success rate p_hat. A predictor
that is constant within a pair scores p_hat(1-p_hat) + (f - p_hat)^2 on it
exactly, so any count-based predictor can then be scored on all 600 runs in
seconds, bit-compatible with the replica.

Run: python runs.py   (writes runs.pkl)
"""
import multiprocessing as mp
import pickle

from common import *  # noqa: F401,F403

OUT = os.path.join(HERE, "runs.pkl")
_pairs = None


def one(i, seed=0):
    global _pairs
    from paiec import baselines as B, official as O
    if _pairs is None:
        _pairs = eligible_pairs()
    run = O.sample_run(_pairs, np.random.default_rng([seed, i]))
    res = O.run_official(run, lambda: B.const(0.5), deepcopy=False)
    out = []
    for (p, items), row in zip(run, res["rows"]):
        keys = res["acquired"][(row["subject_id"], row["benchmark_id"])]
        by = p.by_item()
        acq, ev = (set(k) & items for k in O.split(p, 0))
        y_eval = [r.label for k in ev for r in by[k]]
        out.append({"bench": p.benchmark_id, "subject": p.subject_id,
                    "labels": [int(by[k][0].label) for k in keys],
                    "p_eval": float(np.mean(y_eval)), "n_eval": len(ev),
                    "pa": float(np.mean([by[k][0].label for k in acq])), "na": len(acq)})
    return i, out


def main(n=600, jobs=6):
    with mp.get_context("spawn").Pool(jobs) as pool:
        got = dict(pool.imap_unordered(one, range(n), chunksize=4))
    runs = [got[i] for i in range(n)]
    with open(OUT, "wb") as f:
        pickle.dump(runs, f)
    print(len(runs), "runs;", np.mean([len(r) for r in runs]), "pairs per run")


if __name__ == "__main__":
    main()
