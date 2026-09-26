"""Part 1c: the shipped Predictor with a benchmark-level centre in its ability
prior, on the R1 runs of experiments/official_baselines.py (same seeds), in
the replica, paired against the stock Predictor.

The Predictor's prior on a pair's standing is a ~ N(m_attr(subject), v_a),
with m_attr relative (centred within benchmark), so its level centre is logit
0. The variants add mu0 to m_attr:
  stock            mu0 = 0, v_a = 2          (target-LOBO attribute prior)
  level LOBO       mu0 = mean level of the other four public benchmarks
  level LOBO v3    the same with v_a = 3 (wider: total sd 1.73 vs 1.41)
  v_a 3            mu0 = 0, v_a = 3
  level in-sample  mu0 = -0.56, fitted on all five (optimistic reference)

Run: python predictor_check.py --runs 300 --jobs 6
"""
import argparse
import json
import multiprocessing as mp

from common import *  # noqa: F401,F403

LOO = json.load(open(os.path.join(HERE, "levels.json")))["between"]["loo_mu0"]
MU_ALL = json.load(open(os.path.join(HERE, "levels.json")))["between"]["mu0"]
VARIANTS = {
    "stock": (None, 2.0),
    "level LOBO": ("loo", 2.0),
    "level LOBO v_a=3": ("loo", 3.0),
    "v_a=3": (None, 3.0),
    "level in-sample": ("all", 2.0),
    "level LOBO v_a=1.5": ("loo", 1.5),
}


def factory(names, mode, v_a):
    from paiec import data as D
    from paiec.predict import Predictor
    import experiments.official_baselines as OB

    class Levelled(Predictor):
        mu0 = 0.0

        def prior_mean(self, subject):
            return super().prior_mean(subject) + self.mu0

    def make():
        by = {}
        for b in names:
            p = Levelled(*OB.prior([b]), v_a=v_a)
            p.mu0 = 0.0 if mode is None else (LOO[b] if mode == "loo" else MU_ALL)
            by[D.anon_id("benchmark", b)] = p
        return lambda input, labeled=None: by[input[1]["benchmark_id"]].predict(input, labeled)
    return make


def one(i):
    from paiec import official as O
    import experiments.official_baselines as OB
    run = O.sample_run(OB.pairs(), np.random.default_rng([0, i]))
    names = O.run_benchmarks(run)
    out = {}
    for v, (mode, v_a) in VARIANTS.items():
        res = O.run_official(run, factory(names, mode, v_a), deepcopy=False, workers=1)
        out[v] = [res["brier"][B] for B in (0, 1, 3, 7, 15, 31)] + [res["brier"]["ALC"]]
    return i, out


def main(runs=300, jobs=6):
    sys.path.insert(0, ROOT)
    with mp.get_context("spawn").Pool(jobs) as pool:
        got = dict(pool.imap_unordered(one, range(runs), chunksize=2))
    res = {v: np.array([got[i][v] for i in range(runs)]) for v in VARIANTS}
    summary = {}
    for v, a in res.items():
        d = a[:, -1] - res["stock"][:, -1]
        summary[v] = {"budgets": a.mean(0).tolist(), "alc": float(a[:, -1].mean()),
                      "diff_vs_stock": float(d.mean()), "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                      "diff_by_budget": (a[:, :-1] - res["stock"][:, :-1]).mean(0).tolist()}
        print(f"{v:<22}", " ".join(f"{x:.4f}" for x in a.mean(0)),
              f"  vs stock {d.mean():+.4f}±{d.std(ddof=1) / np.sqrt(len(d)):.4f}")
    with open(os.path.join(HERE, "predictor_check.json"), "w") as f:
        json.dump({"runs": runs, "summary": summary,
                   "per_run_alc": {v: a[:, -1].tolist() for v, a in res.items()}}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=6)
    main(**vars(ap.parse_args()))
