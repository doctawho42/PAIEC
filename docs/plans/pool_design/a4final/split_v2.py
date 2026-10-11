"""The revised selection/confirmation split and the regimes' feasibility
(amendment 4 final: R2, R3 and the recommended weight cap and Kish n).

Composition only. No predictor runs, no Brier or ALC is computed, no item text
and no formative feedback is read (the level targets are the RS study's,
quoted from experiments/regime_sensitivity.py). Reads ../benchmarks.csv,
../pairs.csv (inventory.py's outputs) and families.json (families.py).

Two split rules are computed, both by levels, direction flags and families
only:
  bench   R2's rule: the flagged benchmarks in level blocks of 3, one of each
          block to S by sha256 rank; the others within each response kind in
          level blocks of 8, round(3 * size / 8) to S, as before.
  family  the same rule on units, where a unit is a source family's members in
          E (families.py), so that no family straddles the halves.
For each half and rule it runs the planning knob search (250 draws a setting,
the coarse grid of eval_design/feasibility.py) with and without a cap of 0.10
on any benchmark's weight share, and reports the realised level, Kish n of the
weights, the flagged share of the weight, the largest single benchmark's
weight and appearance shares, and the balance check of R2.

Writes split_v2.json beside this file. Run:
  PYTHONDONTWRITEBYTECODE=1 python split_v2.py [--jobs 4]
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
LEGACY = {"matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench", "mmdocrag"}
FLAG_DIRS = ("lower_is_better", "unordered")
SALT = "pool-eval-split"
BLOCK, TO_S = 8, 3
FBLOCK, F_TO_S = 3, 1
TARGETS = {"READING": (-0.715, 1.772), "AUDIT": (-1.10, 2.00), "TUNED": (-1.290, 1.696)}
MIX = {"share_below": 0.118, "rest_mean": -0.226, "low": (-4.38, 0.171), "high_sd": 1.151}
REPEAT, NPAIRS = 0.25, (5, 12)
BW, MAX_TILT = 0.7, 20.0
DENSE_B, DENSE_S, DENSE_MIN = 7, 8, 4
CAP_SHARE = 0.10
SEED = 4001          # CAL_SEED; planning draws only, never scored
DRAWS = 250


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rank(*a):
    return hashlib.sha256(repr(a).encode()).hexdigest()


B = {r["benchmark"]: r for r in csv.DictReader(open(os.path.join(D, "benchmarks.csv")))}
E = sorted(b for b, r in B.items() if f(r["eligible_pairs"]) > 0 and b not in LEGACY
           and r["release"] == "public"
           and (r["response_type"] == "binary"
                or (r["response_type"] == "mixed" and f(r["share01"]) >= 0.95)))
P = {b: [] for b in E}
NAMED = {b: [] for b in E}
for r in csv.DictReader(open(os.path.join(D, "pairs.csv"))):
    if r["benchmark"] in P:
        P[r["benchmark"]].append(f(r["logit_cc"]))
        NAMED[r["benchmark"]].append(bool(r["canon"]))
LEV = {b: float(np.mean(P[b])) for b in E}
FLAG = {b: B[b]["direction"] in FLAG_DIRS for b in E}
KIND = {b: B[b]["response_type"] for b in E}
FAM_OF = json.load(open(os.path.join(HERE, "families.json")))["family_of"]


def split(rule):
    """{benchmark: 'S' | 'C'} under `rule` ('bench' or 'family')."""
    if rule == "bench":
        units = {b: [b] for b in E}
    else:
        units = {}
        for b in E:
            units.setdefault(FAM_OF[b], []).append(b)
    ulev = {u: float(np.mean([LEV[b] for b in m])) for u, m in units.items()}
    uflag = {u: any(FLAG[b] for b in m) for u, m in units.items()}
    ukind = {u: ("mixed" if all(KIND[b] == "mixed" for b in m) else "binary") for u, m in units.items()}
    half = {}

    def assign(order, block, to_s):
        for i in range(0, len(order), block):
            blk = sorted(order[i:i + block], key=lambda u: rank(SALT, u))
            k = round(to_s * len(blk) / block)
            for j, u in enumerate(blk):
                for b in units[u]:
                    half[b] = "S" if j < k else "C"

    assign(sorted((u for u in units if uflag[u]), key=lambda u: (ulev[u], u)), FBLOCK, F_TO_S)
    for kind in ("binary", "mixed"):
        assign(sorted((u for u in units if not uflag[u] and ukind[u] == kind), key=lambda u: (ulev[u], u)),
               BLOCK, TO_S)
    return half


def npdf(x, m, s):
    return np.exp(-0.5 * ((np.asarray(x, float) - m) / s) ** 2) / s


def capped(w, cap):
    """Water-filling: no share above `cap`, the excess spread pro rata."""
    w = np.asarray(w, float) / np.sum(w)
    if cap is None or cap * len(w) < 1:
        return w
    fixed = np.zeros(len(w), bool)
    for _ in range(len(w)):
        over = (w > cap + 1e-15) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = cap
        rest = ~fixed
        w[rest] = w[rest] / w[rest].sum() * (1 - cap * fixed.sum())
    return w


def weights(bs, comps, cap=None):
    L = np.array([LEV[b] for b in bs])
    if comps is None:
        return np.ones(len(bs)) / len(bs)
    f0 = npdf(L[:, None], L[None, :], BW).mean(1)
    t = sum(w * npdf(L, m, s) for w, m, s in comps) / f0
    t = np.minimum(t, MAX_TILT * t.mean())
    return capped(t / t.sum(), cap)


def formative(bs, w, rng, n_runs):
    ell, alone, sizes, nb, used = [], [], [], [], {}
    pos = {b: i for i, b in enumerate(bs)}
    for _ in range(n_runs):
        n = int(rng.integers(NPAIRS[0], NPAIRS[1] + 1))
        picked, taken = [], {}
        for _ in range(n):
            opened = sorted(set(picked))
            free_open = [b for b in opened if taken.get(b, 0) < len(P[b])]
            if free_open and rng.random() < REPEAT:
                cand = free_open
            else:
                cand = [b for b in bs if b not in set(opened)]
                if not cand:
                    break
            idx = np.array([pos[b] for b in cand])
            ww = w[idx] / w[idx].sum()
            b = cand[int(rng.choice(len(cand), p=ww))]
            if taken.get(b, 0) >= len(P[b]):
                continue
            free = [j for j in range(len(P[b])) if j not in taken.setdefault(("s", b), set())]
            j = free[int(rng.integers(len(free)))]
            taken[("s", b)].add(j)
            taken[b] = taken.get(b, 0) + 1
            picked.append(b)
            ell.append(P[b][j])
            used[b] = used.get(b, 0) + 1
        c = Counter(picked)
        alone.append(sum(1 for b in picked if c[b] == 1) / max(1, len(picked)))
        sizes.append(len(picked))
        nb.append(len(c))
    return summary(ell, used) | {"pairs_per_run": float(np.mean(sizes)),
                                 "benchmarks_per_run": float(np.mean(nb)),
                                 "share_pairs_alone": float(np.mean(alone))}


def dense(bs, w, rng, n_runs):
    ok = [b for b in bs if len(P[b]) >= DENSE_MIN]
    pos = {b: i for i, b in enumerate(bs)}
    idx = np.array([pos[b] for b in ok])
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
    top = max(used, key=lambda b: used[b])
    return {"mean": float(ell.mean()), "sd": float(ell.std()), "share_below_-3": float(np.mean(ell < -3)),
            "rest_mean": float(rest.mean()), "effective_benchmarks_exp_entropy": float(np.exp(-np.sum(u * np.log(u)))),
            "kish_appearances": float(1 / np.sum(u ** 2)),
            "max_appearance_share": float(u.max()), "max_appearance_benchmark": top,
            "flagged_appearance_share": float(sum(v for b, v in used.items() if FLAG[b]) / sum(used.values())),
            "benchmarks_used": int(len(used))}


def wstats(bs, w):
    w = np.asarray(w, float)
    i = int(np.argmax(w))
    return {"kish_weights": float(1 / np.sum(w ** 2)),
            "flagged_weight_share": float(sum(w[k] for k, b in enumerate(bs) if FLAG[b])),
            "max_weight_share": float(w[i]), "max_weight_benchmark": bs[i]}


def search(bs, kind, target, cap):
    best = None
    grid = [(m, s) for m in np.arange(-3.0, 0.51, 0.125) for s in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)]
    for m, s in grid:
        comps = [(1.0, float(m), float(s))]
        w = weights(bs, comps, cap)
        r = (formative if kind == "formative" else dense)(bs, w, np.random.default_rng([SEED, 1]), DRAWS)
        loss = ((r["mean"] - target[0]) / 0.05) ** 2 + ((r["sd"] - target[1]) / 0.05) ** 2
        if best is None or loss < best[0]:
            best = (loss, comps, r, w)
    return best


def search_mix(bs, cap):
    best = None
    for wl in (0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
        for mh in np.arange(0.0, -1.51, -0.125):
            comps = [(wl, MIX["low"][0], MIX["low"][1]), (1 - wl, float(mh), MIX["high_sd"])]
            w = weights(bs, comps, cap)
            r = formative(bs, w, np.random.default_rng([SEED, 2]), DRAWS)
            loss = ((r["share_below_-3"] - MIX["share_below"]) / 0.01) ** 2 + \
                   ((r["rest_mean"] - MIX["rest_mean"]) / 0.05) ** 2
            if best is None or loss < best[0]:
                best = (loss, comps, r, w)
    return best


def half_summary(bs):
    L = np.array([LEV[b] for b in bs])
    npairs = np.array([len(P[b]) for b in bs])
    return {"benchmarks": len(bs), "level_mean": float(L.mean()), "level_sd": float(L.std()),
            "below_-1": int((L < -1).sum()), "below_-1.5": int((L < -1.5).sum()),
            "below_-2": int((L < -2).sum()), "below_-3": int((L < -3).sum()),
            "single_pair": int((npairs == 1).sum()), "ge4_pairs": int((npairs >= 4).sum()),
            "pairs": int(npairs.sum()),
            "named_share_benchmark_mean": float(np.mean([np.mean(NAMED[b]) for b in bs])),
            "direction_flagged": sorted(b for b in bs if FLAG[b]),
            "mixed": int(sum(KIND[b] == "mixed" for b in bs)),
            "multi_member_families": sorted({FAM_OF[b] for b in bs
                                             if sum(FAM_OF[x] == FAM_OF[b] for x in E) > 1})}


def one(job):
    rule, H, cap = job
    half = split(rule)
    bs = [b for b in E if half[b] == H]
    out = {"half": half_summary(bs)}
    w = weights(bs, None)
    out["P-FLAT"] = formative(bs, w, np.random.default_rng([SEED, 0]), 600) | wstats(bs, w)
    for name, tgt in TARGETS.items():
        loss, comps, r, w = search(bs, "formative", tgt, cap)
        out["P-" + name] = r | wstats(bs, w) | {"knobs": comps, "loss": loss}
    loss, comps, r, w = search_mix(bs, cap)
    out["P-MIXTURE"] = r | wstats(bs, w) | {"knobs": comps, "loss": loss}
    w = weights(bs, None)
    out["D-FLAT"] = dense(bs, w, np.random.default_rng([SEED, 3]), 300) | wstats(bs, w)
    loss, comps, r, w = search(bs, "dense", TARGETS["READING"], cap)
    out["D-READING"] = r | wstats(bs, w) | {"knobs": comps, "loss": loss}
    return job, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(rule, H, cap) for rule in ("bench", "family") for H in ("S", "C") for cap in (None, CAP_SHARE)]
    res = {}
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for (rule, H, cap), out in ex.map(one, jobs):
            res.setdefault(rule, {}).setdefault("cap" if cap else "nocap", {})[H] = out
    out = {"what": __doc__.split("\n")[0], "E": len(E), "seed": SEED, "draws": DRAWS, "cap": CAP_SHARE,
           "splits": {rule: split(rule) for rule in ("bench", "family")}, "results": res, "balance": {}}
    for rule in res:
        for capk in res[rule]:
            S, C = res[rule][capk]["S"], res[rule][capk]["C"]
            out["balance"].setdefault(rule, {})[capk] = {
                r: {"flagged_weight_S": round(S[r]["flagged_weight_share"], 3),
                    "flagged_weight_C": round(C[r]["flagged_weight_share"], 3),
                    "abs_diff": round(abs(S[r]["flagged_weight_share"] - C[r]["flagged_weight_share"]), 3),
                    "ok_0.10": abs(S[r]["flagged_weight_share"] - C[r]["flagged_weight_share"]) <= 0.10,
                    "kish_S": round(S[r]["kish_weights"], 1), "kish_C": round(C[r]["kish_weights"], 1),
                    "max_weight_S": [S[r]["max_weight_benchmark"], round(S[r]["max_weight_share"], 3)],
                    "max_weight_C": [C[r]["max_weight_benchmark"], round(C[r]["max_weight_share"], 3)]}
                for r in ("P-READING", "P-AUDIT", "P-TUNED", "P-MIXTURE", "D-READING")}
    with open(os.path.join(HERE, "split_v2.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out["balance"], indent=1))
    for rule in res:
        for H in ("S", "C"):
            h = res[rule]["nocap"][H]["half"]
            print(rule, H, h["benchmarks"], round(h["level_mean"], 2), round(h["level_sd"], 2),
                  len(h["direction_flagged"]), h["mixed"], h["multi_member_families"])


if __name__ == "__main__":
    main()
