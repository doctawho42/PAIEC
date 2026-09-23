"""Does the pooling gain survive a benchmark with few subjects?

Inside a benchmark, permute the subjects once and take nested prefixes
S3 < S5 < S10 < ... Score only the pairs in the smallest prefix every time, so
the scored pairs are identical across pool sizes and the only thing that varies
is how many other subjects contribute labels.

Run: python experiments/pool_robustness.py
"""
import numpy as np

from paiec import baselines as B
from paiec import evaluator as E
from paiec.data import load_pairs
from paiec.trajectories import collect_trajectories
from experiments._robust_legacy import run


def main(keep=("matharena", "multi_swebench"), reps=10):
    allp = load_pairs()
    traj = collect_trajectories(allp, B.const(0.5))
    pairs = [p for p in allp if p.benchmark_id in keep]
    acc, ref = run(pairs, traj, sizes=(3, 5, 10, 20, 40, 80), reps=reps)
    rrow = [np.mean(ref[b]) for b in E.BUDGETS]
    ralc = sum(w * v for w, v in zip(E.WEIGHTS, rrow))
    print(f"{'subjects in pool':<18} " + " ".join(f"B{b:<5}" for b in E.BUDGETS) + " |   ALC    gain")
    for N in sorted({N for N, _ in acc}):
        row = [np.mean(acc[(N, b)]) for b in E.BUDGETS]
        a = sum(w * v for w, v in zip(E.WEIGHTS, row))
        print(f"{N:<18} " + " ".join(f"{v:.4f}" for v in row) + f" | {a:.4f}  {ralc - a:+.4f}")
    print(f"{'no pooling':<18} " + " ".join(f"{v:.4f}" for v in rrow) + f" | {ralc:.4f}")


if __name__ == "__main__":
    main()
