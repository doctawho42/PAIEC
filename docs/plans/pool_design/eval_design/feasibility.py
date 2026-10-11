"""Feasibility of the pool evaluation's regimes (eval.md sections 3-4): run
composition only. No predictor runs, no Brier or ALC is computed, no item text
is read, no formative feedback is read (the level targets are the RS study's,
quoted from docs/plans/p1a_regime_sensitivity.md section 4).

Reads ../benchmarks.csv and ../pairs.csv (inventory.py's outputs). A pair's
level here is its continuity-corrected logit over all its items (first 0/1
response), pairs.csv `logit_cc`; the evaluation's realised level uses the
evaluated responses instead, so these figures are approximations for planning.

Writes feasibility.json beside this file. Run:
  PYTHONDONTWRITEBYTECODE=1 python feasibility.py
"""
import csv
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
LEGACY = {"matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench", "mmdocrag"}
SALT = "pool-eval-split"
BLOCK, TO_S = 8, 3
TARGETS = {"READING": (-0.715, 1.772), "AUDIT": (-1.10, 2.00), "TUNED": (-1.290, 1.696)}
MIX = {"share_below": 0.118, "rest_mean": -0.226, "low": (-4.38, 0.171), "high_sd": 1.151}
REPEAT, NPAIRS = 0.25, (5, 12)
BW, MAX_TILT = 0.7, 20.0
DENSE_B, DENSE_S, DENSE_MIN = 7, 8, 4


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rank(*a):
    return hashlib.sha256(repr(a).encode()).hexdigest()


B = {r["benchmark"]: r for r in csv.DictReader(open(os.path.join(D, "benchmarks.csv")))}
E = sorted(b for b, r in B.items() if f(r["eligible_pairs"]) > 0 and b not in LEGACY
           and (r["response_type"] == "binary"
                or (r["response_type"] == "mixed" and f(r["share01"]) >= 0.95)))
P = {b: [] for b in E}
NAMED = {b: [] for b in E}
for r in csv.DictReader(open(os.path.join(D, "pairs.csv"))):
    if r["benchmark"] in P:
        P[r["benchmark"]].append(f(r["logit_cc"]))
        NAMED[r["benchmark"]].append(bool(r["canon"]))
LEV = {b: float(np.mean(P[b])) for b in E}

# the split: within each response kind (binary; mixed at a 0/1 share >= 0.95),
# blocks of BLOCK by level from the lowest, round(TO_S * size / BLOCK) of each
# block to S by sha256 rank, the rest to C
HALF = {}
for kind in ("binary", "mixed"):
    order = sorted((b for b in E if B[b]["response_type"] == kind), key=lambda b: (LEV[b], b))
    for i in range(0, len(order), BLOCK):
        blk = sorted(order[i:i + BLOCK], key=lambda b: rank(SALT, b))
        k = round(TO_S * len(blk) / BLOCK)
        for j, b in enumerate(blk):
            HALF[b] = "S" if j < k else "C"


def npdf(x, m, s):
    return np.exp(-0.5 * ((np.asarray(x, float) - m) / s) ** 2) / s


def weights(bs, comps):
    """Benchmark weights: target density / KDE base density of the half's
    levels, capped at MAX_TILT x mean; comps = [(w, m, s)] or None (no tilt)."""
    L = np.array([LEV[b] for b in bs])
    if comps is None:
        return np.ones(len(bs)) / len(bs)
    f0 = npdf(L[:, None], L[None, :], BW).mean(1)
    t = sum(w * npdf(L, m, s) for w, m, s in comps) / f0
    t = np.minimum(t, MAX_TILT * t.mean())
    return t / t.sum()


def formative(bs, w, rng, n_runs):
    ell, alone, sizes, nb, used = [], [], [], [], {}
    for _ in range(n_runs):
        n = int(rng.integers(NPAIRS[0], NPAIRS[1] + 1))
        picked, taken = [], set()
        for _ in range(n):
            opened = sorted(set(picked))
            free_open = [b for b in opened if len([j for j in range(len(P[b])) if (b, j) not in taken])]
            if free_open and rng.random() < REPEAT:
                cand = free_open
            else:
                cand = [b for b in bs if b not in opened]
                if not cand:
                    break
            idx = np.array([bs.index(b) for b in cand])
            ww = w[idx] / w[idx].sum()
            b = cand[int(rng.choice(len(cand), p=ww))]
            free = [j for j in range(len(P[b])) if (b, j) not in taken]
            if not free:
                continue
            j = free[int(rng.integers(len(free)))]
            taken.add((b, j))
            picked.append(b)
            ell.append(P[b][j])
            used[b] = used.get(b, 0) + 1
        from collections import Counter
        c = Counter(picked)
        alone.append(sum(1 for b in picked if c[b] == 1) / max(1, len(picked)))
        sizes.append(len(picked))
        nb.append(len(c))
    return summary(ell, used) | {"pairs_per_run": float(np.mean(sizes)),
                                 "benchmarks_per_run": float(np.mean(nb)),
                                 "share_pairs_alone": float(np.mean(alone))}


def dense(bs, w, rng, n_runs):
    ok = [b for b in bs if len(P[b]) >= DENSE_MIN]
    idx = np.array([bs.index(b) for b in ok])
    ww = w[idx] / w[idx].sum()
    ell, used, npairs = [], {}, []
    for _ in range(n_runs):
        chosen = rng.choice(len(ok), size=min(DENSE_B, len(ok)), replace=False, p=ww)
        k = 0
        for c in chosen:
            b = ok[int(c)]
            js = rng.choice(len(P[b]), size=min(DENSE_S, len(P[b])), replace=False)
            for j in js:
                ell.append(P[b][int(j)])
                used[b] = used.get(b, 0) + 1
                k += 1
        npairs.append(k)
    return summary(ell, used) | {"pairs_per_run": float(np.mean(npairs)), "eligible_benchmarks": len(ok)}


def summary(ell, used):
    ell = np.array(ell)
    u = np.array(list(used.values()), float)
    u /= u.sum()
    rest = ell[ell >= -3]
    return {"mean": float(ell.mean()), "sd": float(ell.std()), "share_below_-3": float(np.mean(ell < -3)),
            "rest_mean": float(rest.mean()), "effective_benchmarks": float(np.exp(-np.sum(u * np.log(u)))),
            "benchmarks_used": int(len(used))}


def search(bs, kind, target, seed):
    best = None
    grid = [(m, s) for m in np.arange(-3.0, 0.51, 0.125) for s in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)]
    for m, s in grid:
        comps = [(1.0, float(m), float(s))]
        w = weights(bs, comps)
        r = (formative if kind == "formative" else dense)(bs, w, np.random.default_rng([seed, 1]), 250)
        loss = ((r["mean"] - target[0]) / 0.05) ** 2 + ((r["sd"] - target[1]) / 0.05) ** 2
        if best is None or loss < best[0]:
            best = (loss, comps, r)
    return best


def search_mix(bs, seed):
    best = None
    for wl in (0.10, 0.14, 0.18, 0.22, 0.26):
        for mh in np.arange(-0.25, -1.26, -0.125):
            comps = [(wl, MIX["low"][0], MIX["low"][1]), (1 - wl, float(mh), MIX["high_sd"])]
            w = weights(bs, comps)
            r = formative(bs, w, np.random.default_rng([seed, 2]), 250)
            loss = ((r["share_below_-3"] - MIX["share_below"]) / 0.01) ** 2 + \
                   ((r["rest_mean"] - MIX["rest_mean"]) / 0.05) ** 2
            if best is None or loss < best[0]:
                best = (loss, comps, r)
    return best


def block_sd(bs, comps, kind, seed, block, blocks=8):
    w = weights(bs, comps)
    fn = formative if kind == "formative" else dense
    ms, ss = [], []
    for k in range(blocks):
        r = fn(bs, w, np.random.default_rng([seed, 100 + k]), block)
        ms.append(r["mean"])
        ss.append(r["sd"])
    return {"block_runs": block, "blocks": blocks, "sd_of_mean": float(np.std(ms, ddof=1)),
            "sd_of_sd": float(np.std(ss, ddof=1))}


def main():
    seed = 4001
    out = {"what": __doc__.split("\n")[0], "E": len(E), "split": {}, "halves": {}}
    for H in ("S", "C"):
        bs = [b for b in E if HALF[b] == H]
        L = np.array([LEV[b] for b in bs])
        npairs = np.array([len(P[b]) for b in bs])
        named = np.mean([np.mean(NAMED[b]) for b in bs])
        flagged = sum(B[b]["direction"] in ("lower_is_better", "unordered") for b in bs)
        out["split"][H] = {"benchmarks": len(bs), "level_mean": float(L.mean()), "level_sd": float(L.std()),
                           "below_-1": int((L < -1).sum()), "below_-1.5": int((L < -1.5).sum()),
                           "below_-2": int((L < -2).sum()), "below_-3": int((L < -3).sum()),
                           "single_pair": int((npairs == 1).sum()), "ge4_pairs": int((npairs >= 4).sum()),
                           "pairs": int(npairs.sum()), "named_share_benchmark_mean": float(named),
                           "direction_flagged": int(flagged),
                           "mixed": int(sum(B[b]["response_type"] == "mixed" for b in bs))}
        h = out["halves"][H] = {}
        h["P-FLAT"] = formative(bs, weights(bs, None), np.random.default_rng([seed, 0]), 600)
        h["P-FLAT"]["block_sd_120"] = block_sd(bs, None, "formative", seed, 120)
        for name, tgt in TARGETS.items():
            loss, comps, r = search(bs, "formative", tgt, seed)
            r["knobs_benchmark_tilt"] = comps
            r["block_sd_120"] = block_sd(bs, comps, "formative", seed, 120)
            h["P-" + name] = r
        loss, comps, r = search_mix(bs, seed)
        r["knobs_benchmark_tilt"] = comps
        h["P-MIXTURE"] = r
        h["D-FLAT"] = dense(bs, weights(bs, None), np.random.default_rng([seed, 3]), 300)
        loss, comps, r = search(bs, "dense", TARGETS["READING"], seed)
        r["knobs_benchmark_tilt"] = comps
        r["block_sd_50"] = block_sd(bs, comps, "dense", seed, 50)
        h["D-READING"] = r
    with open(os.path.join(HERE, "feasibility.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
