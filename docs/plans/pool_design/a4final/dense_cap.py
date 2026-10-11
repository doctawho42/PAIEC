"""The dense regimes' weights with the 0.10 cap applied after the dense
sampler's restriction to benchmarks with at least 4 eligible pairs (re-check
N9), beside the order split_v2.py used (cap first, then restriction).

Composition only, as split_v2.py: no predictor runs, no Brier or ALC, no item
text, no formative feedback beyond the RS study's quoted READING target. It
reuses split_v2.py's data, splits and sampler (the family rule of the design and
the critic's benchmark rule, CAL_SEED 4001 on
split_v2's two-element planning stream, 250 draws a setting, its coarse grid)
and reports, per half: the knobs, the realised all-item level, Kish n of the
weights by benchmark and by family, the largest weight share and the flagged
share of the weight, so the balance check can be read on the dense weights as
the sampler uses them. The tilt t_b and its clip at 20 times the mean are
computed over the half, as before; only the order of restriction and cap
changes.

Writes dense_cap.json beside this file. Run:
  PYTHONDONTWRITEBYTECODE=1 python dense_cap.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import split_v2 as V  # noqa: E402  (loads ../benchmarks.csv, ../pairs.csv, families.json)


def tilt(bs, comps):
    """t_b / sum over the half `bs`, clipped at MAX_TILT times the mean (split_v2.weights without the cap)."""
    L = np.array([V.LEV[b] for b in bs])
    if comps is None:
        return np.ones(len(bs)) / len(bs)
    f0 = V.npdf(L[:, None], L[None, :], V.BW).mean(1)
    t = sum(w * V.npdf(L, m, s) for w, m, s in comps) / f0
    t = np.minimum(t, V.MAX_TILT * t.mean())
    return t / t.sum()


def restricted(bs, w):
    ok = [b for b in bs if len(V.P[b]) >= V.DENSE_MIN]
    pos = {b: i for i, b in enumerate(bs)}
    ww = np.array([w[pos[b]] for b in ok], float)
    return ok, ww / ww.sum()


def stats(ok, w):
    fam = {}
    for b, x in zip(ok, w):
        fam[V.FAM_OF[b]] = fam.get(V.FAM_OF[b], 0.0) + float(x)
    i = int(np.argmax(w))
    return {"benchmarks": len(ok),
            "kish_benchmarks": round(float(1 / np.sum(np.square(w))), 1),
            "kish_families": round(float(1 / sum(v * v for v in fam.values())), 1),
            "max_weight": [ok[i], round(float(w[i]), 3)],
            "flagged_weight_share": round(float(sum(x for b, x in zip(ok, w) if V.FLAG[b])), 3)}


def regime(bs, comps, order):
    """Dense weights as the sampler uses them, under `order` 'cap_first' or 'cap_after'."""
    if order == "cap_first":
        ok, w = restricted(bs, V.capped(tilt(bs, comps), V.CAP_SHARE))
    else:
        ok, w = restricted(bs, tilt(bs, comps))
        w = V.capped(w, V.CAP_SHARE)
    return ok, w


def search(bs, order):
    best = None
    grid = [(m, s) for m in np.arange(-3.0, 0.51, 0.125) for s in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)]
    tgt = V.TARGETS["READING"]
    for m, s in grid:
        comps = [(1.0, float(m), float(s))]
        ok, w = regime(bs, comps, order)
        r = V.dense(ok, w, np.random.default_rng([V.SEED, 1]), V.DRAWS)
        loss = ((r["mean"] - tgt[0]) / 0.05) ** 2 + ((r["sd"] - tgt[1]) / 0.05) ** 2
        if best is None or loss < best[0]:
            best = (loss, comps, r, ok, w)
    return best


def main():
    out = {"what": __doc__.split("\n")[0], "seed": V.SEED, "draws": V.DRAWS, "cap": V.CAP_SHARE,
           "dense_min_pairs": V.DENSE_MIN, "results": {}}
    for rule, order in (("family", "cap_first"), ("family", "cap_after"), ("bench", "cap_after")):
        half = V.split(rule)
        res = {}
        for H in ("S", "C"):
            bs = [b for b in V.E if half[b] == H]
            ok, w = regime(bs, None, order)
            res.setdefault("D-FLAT", {})[H] = stats(ok, w)
            loss, comps, r, ok, w = search(bs, order)
            res.setdefault("D-READING", {})[H] = stats(ok, w) | {
                "knobs": comps, "loss": round(loss, 2),
                "realised_all_item": {"mean": round(r["mean"], 3), "sd": round(r["sd"], 3)}}
        for reg in res:
            res[reg]["balance_abs_diff"] = round(abs(res[reg]["S"]["flagged_weight_share"]
                                                     - res[reg]["C"]["flagged_weight_share"]), 3)
        out["results"].setdefault(rule, {})[order] = res
    with open(os.path.join(HERE, "dense_cap.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
