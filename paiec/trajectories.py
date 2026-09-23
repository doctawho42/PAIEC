"""Build one acquisition trajectory per subject-benchmark pair.

The evaluator hands out nested prefixes of a single trajectory, so everything
downstream works from these lists. `collect_trajectories` runs the streaming
phase with whichever predictor and policy are passed; with the defaults it
reproduces the evaluator's own random policy.
"""
from paiec import evaluator as E


def collect_trajectories(pairs, predict, acquisition=None, seed=0):
    traj = {}
    for p in pairs:
        t = E.acquire(p, predict, acquisition, seed=seed)
        traj[(p.subject_id, p.benchmark_id)] = list(
            zip(t.acquired_keys, [y for _, y in t.acquired]))
    return traj
