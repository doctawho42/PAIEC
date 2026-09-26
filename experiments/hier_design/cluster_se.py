"""Pair-cluster bootstrap of the R1 ALC differences for the candidate level
priors (count-based, exact scoring as in score.py).

The 600 runs redraw the same 221 pairs, so run-level SEs are too small. Each
pair appearance contributes (its ALC difference) / (pairs in its run) to the
run's difference; a bootstrap resample reweights every appearance by how often
its pair was drawn (2,000 multinomial draws over the pairs).

Run: python cluster_se.py
"""
import json

from common import *  # noqa: F401,F403
from paiec.evaluator import BUDGETS, WEIGHTS
from prior_sweep import ByBench
from score import Hier, load_runs, smoothed

CANDIDATES = {
    "Beta(2,2)": lambda: smoothed(),
    "normal own LOBO s=1.3": lambda: ByBench(1.3 / np.sqrt(2), 1.3 / np.sqrt(2), None, False),
    "normal hier LOBO 0.9/0.9": lambda: ByBench(0.9, 0.9, None, True),
    "t3 hier LOBO 0.9/0.9": lambda: ByBench(0.9, 0.9, 3, True),
    "normal hier LOBO 1.3/1.3": lambda: ByBench(1.3, 1.3, None, True),
    "t3 hier LOBO 1.3/0.9": lambda: ByBench(1.3, 0.9, 3, True),
    "normal hier in-sample -0.56 0.9/0.9": lambda: Hier(-0.56, 0.9, 0.93, use_others=True),
}


def per_pair(run, predictor):
    """ALC of each pair of the run."""
    by = {}
    for j, p in enumerate(run):
        by.setdefault(p["bench"], []).append(j)
    out = np.zeros(len(run))
    for w, B in zip(WEIGHTS, BUDGETS):
        for j, p in enumerate(run):
            k, n = sum(p["labels"][:B]), min(B, len(p["labels"]))
            others = [(sum(run[o]["labels"][:B]), min(B, len(run[o]["labels"]))) for o in by[p["bench"]] if o != j]
            f = predictor(k, n, others, p)
            pe = p["p_eval"]
            out[j] += w * (pe * (1 - pe) + (f - pe) ** 2)
    return out


def main(boots=2000):
    runs = load_runs()
    keys = sorted({(p["bench"], p["subject"]) for r in runs for p in r})
    index = {k: i for i, k in enumerate(keys)}
    pid = [np.array([index[(p["bench"], p["subject"])] for p in r]) for r in runs]
    alc = {name: [per_pair(r, make()) for r in runs] for name, make in CANDIDATES.items()}
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=boots).astype(float)
    ref = alc["Beta(2,2)"]
    out = {}
    for name in CANDIDATES:
        contrib = [(a - b) / len(a) for a, b in zip(alc[name], ref)]
        point = float(np.mean([c.sum() for c in contrib]))
        run_se = float(np.std([c.sum() for c in contrib], ddof=1) / np.sqrt(len(runs)))
        v = np.zeros(len(keys))          # each pair's summed contribution over all runs
        for i, c in zip(pid, contrib):
            np.add.at(v, i, c / len(runs))
        bs = W @ v
        out[name] = {"diff": point, "run_se": run_se, "cluster_se": float(bs.std(ddof=1)),
                     "cluster_95": np.percentile(bs, [2.5, 97.5]).tolist()}
        print(f"{name:<38} {point:+.4f}  run SE {run_se:.4f}  pair-cluster SE {bs.std(ddof=1):.4f} "
              f"95% [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    with open(os.path.join(HERE, "cluster_se.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
