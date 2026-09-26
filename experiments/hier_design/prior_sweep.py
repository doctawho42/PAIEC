"""Part 1b: how much the level prior's centre and width move ALC, on the public
R1 runs and on scenarios where hidden pairs' accuracies are more extreme.

Predictors are count-based and constant within a pair (score.py), so R1 is
scored exactly as the replica would (checked: Beta(2,2) 0.2158, empirical mean
0.2526). The scenarios keep each R1 run's structure (which pairs share a
benchmark), draw a fresh level per benchmark and a fresh deviation per pair
from a logit-normal hierarchy, and 31 Bernoulli labels per pair; the
evaluation rate is the pair's p. Every predictor sees the same draws.

Scenarios (pair logit = m_b + u_sb):
  public    m_b ~ N(-0.56, 0.90^2), u ~ N(0, 0.93^2): levels.py's fit, E[p(1-p)] 0.181
  stretch   both sds x1.94: E[p(1-p)] 0.128 at the same centre
  between   only sigma_mu grows (2.32): E[p(1-p)] 0.128
  within    only sigma_u grows (2.33): E[p(1-p)] 0.128
  hard      centre -1.80 (mean p 0.20), public sds: E[p(1-p)] 0.128
  easy      centre +1.80 (mean p 0.80), public sds: E[p(1-p)] 0.128
  hard_t    heavy tails: centre -0.56, t3 levels and deviations at public scales

Writes prior_sweep.json. Run: python prior_sweep.py
"""
import itertools
import json
import multiprocessing as mp

from scipy import optimize

from common import *  # noqa: F401,F403
from levels import epq
from score import Hier, load_runs, score_runs, simulate, smoothed

M0, SMU, SU = -0.56, 0.90, 0.93


def scenarios():
    tot = np.hypot(SMU, SU)
    lam = optimize.brentq(lambda l: epq(M0, l * tot) - 0.128, 1, 5)
    s_star = lam * tot
    m_hard = optimize.brentq(lambda m: epq(m, tot) - 0.128, -8, 0)
    return {
        "public": dict(M=M0, S_mu=SMU, S_u=SU),
        "stretch": dict(M=M0, S_mu=SMU * lam, S_u=SU * lam),
        "between": dict(M=M0, S_mu=float(np.sqrt(s_star ** 2 - SU ** 2)), S_u=SU),
        "within": dict(M=M0, S_mu=SMU, S_u=float(np.sqrt(s_star ** 2 - SMU ** 2))),
        "hard": dict(M=m_hard, S_mu=SMU, S_u=SU),
        "easy": dict(M=-m_hard, S_mu=SMU, S_u=SU),
        "heavy_t3": dict(M=M0, S_mu=SMU, S_u=SU, df_m=3, df_u=3),
    }


LOO = json.load(open(os.path.join(HERE, "levels.json")))["between"]["loo_mu0"]


class ByBench:
    """A level prior centred, for each target, on the mean level of the other
    public benchmarks: the honest centre on R1, where the target's own
    benchmark must not inform it. In the scenarios the levels are fresh draws,
    so this is just a centre fitted on public data, as it would be at test."""

    def __init__(self, smu, su, df=None, others=True):
        self.by = {b: Hier(m, smu, su, df_m=df, df_u=df, use_others=others) for b, m in LOO.items()}

    def __call__(self, k, n, others, p):
        return self.by[p["bench"]](k, n, others, p)


def predictors():
    out = {"Beta(2,2)": ("beta", 4.0, 0.5)}
    for s in (1.0, 1.3, 1.6, 2.0, 2.5):
        out[f"LN own LOBO s={s:.1f}"] = ("lobo", s / np.sqrt(2), s / np.sqrt(2), None, False)
    for smu, su in itertools.product((0.9, 1.3, 1.8), (0.9, 1.3, 1.8)):
        out[f"LN hier LOBO smu={smu:.1f} su={su:.1f}"] = ("lobo", smu, su, None, True)
    for smu, su in itertools.product((0.9, 1.3), (0.9, 1.3)):
        out[f"t3 hier LOBO smu={smu:.1f} su={su:.1f}"] = ("lobo", smu, su, 3, True)
    for c, n0 in itertools.product((0.35, 0.4, 0.45), (2.0, 3.0, 4.0)):
        out[f"Beta c={c} n0={n0:g}"] = ("beta", n0, c)
    for mu0, s in itertools.product((-1.2, -0.9, -0.6, -0.3, 0.0), (1.0, 1.3, 1.6, 2.0, 2.5, 3.0)):
        out[f"LN own mu0={mu0:+.1f} s={s:.1f}"] = ("hier", mu0, s / np.sqrt(2), s / np.sqrt(2), None, None, False)
    for mu0, smu, su in itertools.product((-0.9, -0.6, -0.3, 0.0), (0.9, 1.3, 1.8, 2.3), (0.9, 1.3, 1.8)):
        out[f"LN hier mu0={mu0:+.1f} smu={smu:.1f} su={su:.1f}"] = ("hier", mu0, smu, su, None, None, True)
    for mu0, smu, su in itertools.product((-0.6, -0.3), (0.9, 1.3), (0.9, 1.3)):
        out[f"t3 hier mu0={mu0:+.1f} smu={smu:.1f} su={su:.1f}"] = ("hier", mu0, smu, su, 3, 3, True)
        out[f"t3 own mu0={mu0:+.1f} smu={smu:.1f} su={su:.1f}"] = ("hier", mu0, smu, su, 3, 3, False)
    return out


def build(spec):
    if spec[0] == "beta":
        return smoothed(spec[1], spec[2])
    if spec[0] == "lobo":
        return ByBench(spec[1], spec[2], spec[3], spec[4])
    _, mu0, smu, su, dfm, dfu, others = spec
    return Hier(mu0, smu, su, df_m=dfm, df_u=dfu, use_others=others)


_data = {}


def evaluate(args):
    name, spec, scen = args
    by_b, alc = score_runs(_data[scen], build(spec))
    return name, scen, by_b.mean(0).tolist(), alc.tolist()


def init(data):
    _data.update(data)


def main(reps=2, jobs=6):
    runs = load_runs()
    data = {"R1 exact": runs}
    for j, (name, sc) in enumerate(scenarios().items()):
        rng = np.random.default_rng(100 + j)
        sims = []
        for _ in range(reps):
            sims += simulate(runs, rng=rng, **sc)
        data[name] = sims
    preds = predictors()
    tasks = [(n, s, sc) for n, s in preds.items() for sc in data]
    res = {}
    with mp.get_context("fork").Pool(jobs, initializer=init, initargs=(data,)) as pool:
        for name, scen, bb, alc in pool.imap_unordered(evaluate, tasks, chunksize=2):
            res.setdefault(name, {})[scen] = {"budgets": bb, "alc": alc}
    ref = "Beta(2,2)"
    table = {}
    for name in preds:
        table[name] = {}
        for scen in data:
            a = np.array(res[name][scen]["alc"])
            d = a - np.array(res[ref][scen]["alc"])
            table[name][scen] = {"alc": float(a.mean()), "diff_vs_beta22": float(d.mean()),
                                 "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                 "budgets": res[name][scen]["budgets"]}
    best = {sc: min(table, key=lambda n: table[n][sc]["alc"]) for sc in data}
    for n in table:
        reg = {sc: table[n][sc]["alc"] - table[best[sc]][sc]["alc"] for sc in data}
        table[n]["regret"] = reg
        table[n]["max_regret"] = max(reg.values())
        table[n]["max_regret_extreme"] = max(v for k, v in reg.items() if k not in ("R1 exact", "public"))
    with open(os.path.join(HERE, "prior_sweep.json"), "w") as f:
        json.dump({"scenarios": scenarios(), "best": best, "table": table}, f, indent=1)
    scen_names = list(data)
    print("best per scenario:", {sc: (b, round(table[b][sc]["alc"], 4)) for sc, b in best.items()})
    print(f"{'predictor':<44}" + "".join(f"{s[:9]:>10}" for s in scen_names) + f"{'maxreg':>9}")
    for n in sorted(table, key=lambda n: table[n]["max_regret"])[:25] + [ref]:
        print(f"{n:<44}" + "".join(f"{table[n][s]['alc']:>10.4f}" for s in scen_names)
              + f"{table[n]['max_regret']:>9.4f}")


if __name__ == "__main__":
    main()
