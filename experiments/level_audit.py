"""The audit of the level calibration that changed what shipped (step 2b).

On 2026-09-26 an audit of experiments/level_calibration.py's recommendation
(hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25, level_calibration.SHIP) moved
the shipped level to the milder guarded configuration, mu0 -2.5, sigma_mu 2.5,
attr_scale 0.5 (submission/model.py LEVEL; docs/findings.md "What actually
shipped, after the audit"; docs/report/draft.md sections 3.4 and 5.1). Its
scripts ran in session scratch (step2b/audit). This is them, moved here: the
analyses are the same code, and the paired comparisons of the milder config,
which the audit scored with fresh runs, are read from rows the repository
already stores. Names below: AGGR = level_calibration.SHIP (the recommendation
the audit examined), MILD = the configuration that ships, PRED = the legacy
Predictor.

Stages (default: all but extra):

  feedback  Per-pair matched estimate of AGGR on formative run 1
            (testlike.FEEDBACK): each feedback pair is matched to the K
            replica pair appearances whose Predictor Brier profile over the six
            budgets is nearest (each budget over its sd in the pool; the pool
            is every regime stored in results/level_calibration.json), and the
            neighbours' AGGR minus PRED is added to the feedback's ALC
            (feedback_match). The same for every config the public screen
            scored, on the pool where all are stored: test-like runs 0-99 and
            public benchmark-first runs 0-39 (feedback_match2; MILD included).
            Robustness: pools restricted by regime, and matching on (B0, B31)
            only, K 25 (feedback_match3).
  analyse   AGGR minus PRED by pair rate, parent and kind on test-like runs
            0-199; public benchmark-first and pair-uniform runs by benchmark;
            the width and attr_scale contrasts by kind (selection half);
            nested leave-one-parent-out selection over the guarded
            configurations (select on runs 0-99 without parent q, score on
            q's pairs in runs 100-199) and over the Gaussian grids (selection
            half); the pair-rate shares by regime (analyse.py). And how much
            the confirmation half (runs 100-199) repeats the selection half
            (runs 0-99): pair appearances whose (pseudo-benchmark, subject) or
            (parent, subject) also appears there.
  mild      MILD against AGGR and PRED on the runs where the audit scored it
            with fresh runs (test-like 100-199; public 0-99, both weightings;
            no-shift 0-29) and on the other stored ones (test-like 0-99 and
            0-199, no-shift 0-99, mix/whole 0-99). MILD's rows are the 'ship'
            arm of experiments/subject_side.py (data/subject_side_rows,
            gitignored; the same runs, drawn by level_calibration.draw); AGGR's
            and PRED's are results/level_calibration.json's. The stage first
            checks that the two sources agree where both hold MILD (test-like
            0-99, public benchmark-first 0-39), to the rows' rounding (1e-5).
  extra     NEW RUNS (optional, about 24 minutes on one process): the audit's
            two regimes nearer the feedback's per-pair reading, 'tl lm-0.8'
            (level_mean -0.8) and 'tl untilted' (no level tilt), seed 5, 40 runs
            each, scoring smoothed, the Predictor grid and hier's defaults
            ('base'), AGGR, mu0 -3.5, MILD and the EB config, with the library
            on disk. The library has changed since the audit (the corrected
            multiple-choice floor, f7e7d87; the floored-fit fix, 4d2cc4f), so
            these re-measure the audit's numbers rather than reproduce them.
            The runs' rows are kept packed under raw_extra in the results file.

What the audit also did and this does not: latency stress runs on dense
public data (stress_latency.py; superseded by findings' own latency
measurements), and attr_scale 1.0 and 0.5 at mu0 -3.0 without the date shift
(noshift_as.py; those two configs are stored nowhere, so --stage extra would
have to score them afresh).

    python experiments/level_audit.py                 # feedback, analyse, mild (about 5 seconds)
    python experiments/level_audit.py --stage extra   # the new regimes (about 24 minutes, 1.9 GB)
    python experiments/level_audit.py --stage summarise

Output: results/level_audit.json.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import level_calibration as L  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import WEIGHTS  # noqa: E402

OUT = os.path.join(ROOT, "results", "level_audit.json")
LC_JSON = os.path.join(ROOT, "results", "level_calibration.json")
SS_ROWS = os.path.join(ROOT, "data", "subject_side_rows")
AGGR, PRED = L.SHIP, L.PRED
MILD = L.gname(-2.5, 2.5, 0.5)
R35 = "hier G mu0=-3.50 sm=2.50 as=0.25"
PFIX = "P off=-1.5 sc=1.00 va=3.0"
EB = "hierEB-cs,tm=2.0@hier G mu0=-3.00 sm=2.50 as=0.50"
#: the audit's new regimes (extra_runs.py)
NEW_REGIMES = {"tl lm-0.8": ("testlike", 5, {"level_mean": -0.8}),
               "tl untilted": ("testlike", 5, {"level_mean": None})}
N_NEW = 40
#: subject_side row files -> level_calibration regime keys
SS_FILES = {"tl": "tl", "r1b": "r1b", "r1p": "r1p", "tl_mix-whole": "tl mix/whole", "tl_no_shift": "tl no shift"}


def r4(x):
    return None if x is None else round(float(x), 4)


def load_lc():
    with open(LC_JSON) as f:
        st = json.load(f)
    L.register_chunks(L.grid_chunks())
    return st, st["raw"], L.collect_metas(st["raw"])


def rate_bin(p):
    return "<0.1" if p < .1 else "0.1-0.3" if p < .3 else "0.3-0.5" if p < .5 else "0.5-0.7" if p < .7 else ">=0.7"


# --- feedback: per-pair matched estimates ------------------------------------------------

def stage_feedback(st, raw, metas):
    others = {"mu0 -3.5": R35, "hier default": L.HIER, "smoothed": L.SMOOTHED, "Pfix -1.5": PFIX}
    pool = []
    for (regime, i) in sorted(metas):
        p = L.rows_of(raw, regime, i, PRED)
        s = L.rows_of(raw, regime, i, AGGR)
        if p is None or s is None or "pb" not in p:
            continue
        o = {k: L.rows_of(raw, regime, i, v) for k, v in others.items()}
        for j, mr in enumerate(metas[(regime, i)]):
            pool.append((regime, i, mr, np.array(p["pb"][j]), s["alc"][j] - p["alc"][j],
                         {k: (None if v is None else v["alc"][j] - p["alc"][j]) for k, v in o.items()}))
    X = np.array([x[3] for x in pool])
    sd = X.std(0)
    FB = [(b, n, np.array(v)) for b, n, v in T.FEEDBACK]
    res = {"pool": {"appearances": len(pool),
                    "by_regime": {r: sum(1 for x in pool if x[0] == r) for r in sorted({x[0] for x in pool})}}}
    for K in (15, 40):
        rows, tot_aggr, tot = [], [], {k: [] for k in others}
        for idx, (b, n, v) in enumerate(FB):
            d = np.sqrt((((X - v) / sd) ** 2).sum(1))
            nn = np.argsort(d)[:K]
            ps = np.array([pool[j][2]["p"] for j in nn])
            ds = np.array([pool[j][4] for j in nn])
            low = T.rate_from_brier(v[5])
            row = {"pair": f"p{idx + 1} ({b})", "B0": float(v[0]), "B31": float(v[5]),
                   "lower root": round(low, 3), "upper root": round(1 - low, 3),
                   "nn mean p": round(float(ps.mean()), 3), "nn share p>0.5": round(float((ps > .5).mean()), 2),
                   "nn median dist": round(float(np.median(d[nn])), 2),
                   "nn regimes": dict(sorted(defaultdict(int, {r: sum(1 for j in nn if pool[j][0] == r)
                                                               for r in {pool[j][0] for j in nn}}).items())),
                   "AGGR - PRED (ALC)": r4(ds.mean()), "se": r4(ds.std(ddof=1) / np.sqrt(K))}
            for k in others:
                vals = [pool[j][5][k] for j in nn if pool[j][5][k] is not None]
                row[k + " - PRED"] = r4(np.mean(vals)) if vals else None
                if vals:
                    tot[k].append(np.mean(vals))
            tot_aggr.append(ds.mean())
            rows.append(row)
        res[f"K={K}"] = {"pairs": rows, "estimate AGGR ALC": T.FEEDBACK_ALC + float(np.mean(tot_aggr)),
                         "mean AGGR - PRED": float(np.mean(tot_aggr)),
                         **{f"estimate {k} ALC": T.FEEDBACK_ALC + float(np.mean(v))
                            for k, v in tot.items() if len(v) == len(FB)}}
    gap = X[:, 0] - X[:, 5]
    ps = np.array([x[2]["p"] for x in pool])
    bins = [(-1, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1)]
    res["share p>0.5 by PRED B0-B31 gap"] = {
        f"{a}..{b}": [round(float((ps[(gap > a) & (gap <= b)] > .5).mean()), 3), int(((gap > a) & (gap <= b)).sum())]
        for a, b in bins}
    res["feedback B0-B31 gaps"] = [round(float(v[0] - v[5]), 4) for _, _, v in FB]
    rb = defaultdict(list)
    for x in pool:
        rb[rate_bin(x[2]["p"])].append(x[4])
    res["pool AGGR - PRED by pair rate (per appearance, unweighted)"] = {
        k: [r4(np.mean(v)), len(v)] for k, v in sorted(rb.items())}
    # re-reading the nine rates as neighbour means (K=40), on the accuracy-logit scale
    nnp = [r["nn mean p"] for r in res["K=40"]["pairs"]]
    lo = [T.rate_from_brier(v[5]) for _, _, v in FB]
    lg = lambda q: float(np.log(q / (1 - q)))
    res["feedback level readings"] = {
        "lower roots: mean, sd of logit": [round(np.mean([lg(q) for q in lo]), 2), round(np.std([lg(q) for q in lo], ddof=1), 2)],
        "neighbour means (K=40): mean, sd of logit": [round(np.mean([lg(q) for q in nnp]), 2),
                                                     round(np.std([lg(q) for q in nnp], ddof=1), 2)]}

    # feedback_match2: every screened config on the pool where all are stored
    names = list(dict.fromkeys(L.screen_configs(raw, 100) + [L.SMOOTHED, L.HIER, PFIX, "Pr off=-1.5 sc=1.00 va=3.0"]))
    keys = [("tl", i) for i in range(100)] + [("r1b", i) for i in range(40)]
    pool2, D = [], {n: [] for n in names}
    missing = []
    for reg, i in keys:
        p = L.rows_of(raw, reg, i, PRED)
        rs = {n: L.rows_of(raw, reg, i, n) for n in names}
        miss = [n for n, r in rs.items() if r is None]
        if miss:
            missing.append([reg, i, miss[:3]])
            continue
        for j, mr in enumerate(metas[(reg, i)]):
            pool2.append((reg, mr["p"], np.array(p["pb"][j])))
            for n in names:
                D[n].append(rs[n]["alc"][j] - p["alc"][j])
    X2 = np.array([x[2] for x in pool2]); sd2 = X2.std(0)
    D = {n: np.array(v) for n, v in D.items()}
    m2 = {"pool": "test-like runs 0-99 and public benchmark-first runs 0-39", "appearances": len(pool2),
          "configs": len(names), "runs skipped (a config missing)": missing}
    for K in (15, 40):
        nns = [np.argsort(np.sqrt((((X2 - np.array(v)) / sd2) ** 2).sum(1)))[:K] for _, _, v in T.FEEDBACK]
        est = {n: T.FEEDBACK_ALC + float(np.mean([D[n][nn].mean() for nn in nns])) for n in names}
        m2[f"K={K}"] = [[n, round(v, 4)] for n, v in sorted(est.items(), key=lambda kv: kv[1])]
        m2[f"K={K} AGGR, MILD"] = [r4(est[AGGR]), r4(est[MILD])]
    res["all screened configs"] = m2

    # feedback_match3: robustness of the AGGR estimate (K 25)
    pool3 = [(x[0], x[2]["p"], x[3], x[4]) for x in pool]

    def est(sub, feats, K=25):
        Xs = np.array([x[2][feats] for x in sub]); sds = Xs.std(0)
        Ds = np.array([x[3] for x in sub]); P = np.array([x[1] for x in sub])
        per, hi = [], []
        for _, _, v in T.FEEDBACK:
            v = np.array(v)[feats]
            nn = np.argsort(np.sqrt((((Xs - v) / sds) ** 2).sum(1)))[:K]
            per.append(Ds[nn].mean()); hi.append(float((P[nn] > .5).mean()))
        return round(T.FEEDBACK_ALC + float(np.mean(per)), 4), [round(float(x), 3) for x in per], hi

    rob = {}
    for label, f in [("all regimes", lambda x: True), ("test-like default only", lambda x: x[0] == "tl"),
                     ("test-like, all regimes", lambda x: x[0].startswith("tl")),
                     ("public only", lambda x: x[0].startswith("r1"))]:
        sub = [x for x in pool3 if f(x)]
        for feats, fl in [(slice(0, 6), "6 budgets"), ([0, 5], "B0,B31")]:
            e, per, hi = est(sub, feats)
            rob[f"{label} | {fl}"] = {"n": len(sub), "AGGR ALC est": e, "per pair AGGR-PRED": per, "share p>0.5": hi}
    res["robustness (K=25)"] = rob
    # the range the audit quoted: K 15 and 40 on every regime, and K 25 on the pools that
    # hold test-like runs (the public-only pool is reported but was not part of it)
    ests = [v["AGGR ALC est"] for k, v in rob.items() if not k.startswith("public only")] + \
        [res["K=15"]["estimate AGGR ALC"], res["K=40"]["estimate AGGR ALC"]]
    res["AGGR estimate range (pools holding test-like runs)"] = [round(min(ests), 4), round(max(ests), 4)]
    return res


# --- analyse ---------------------------------------------------------------------------

def stage_analyse(st, raw, metas):
    out = {}

    def per_pair(regime, ids, name):
        rows = []
        for i in ids:
            r = L.rows_of(raw, regime, i, name)
            if r is None:
                return None
            for j, (mr, a) in enumerate(zip(metas[(regime, i)], r["alc"])):
                rows.append((i, j, mr, a))
        return rows

    def diffs(regime, ids, a, b):
        A, B = per_pair(regime, ids, a), per_pair(regime, ids, b)
        if A is None or B is None:
            return None
        return [(x[0], x[2], x[3] - y[3], len(metas[(regime, x[0])])) for x, y in zip(A, B)]

    def wm(rows):
        w = np.array([1 / r[3] for r in rows]); d = np.array([r[2] for r in rows])
        return float((w * d).sum() / w.sum())

    def wmean(rows):
        return [r4(wm(rows)), len(rows)]

    tl = diffs("tl", range(200), AGGR, PRED)
    by = defaultdict(list)
    for r in tl:
        by["kind " + T.kind_of(r[1]["bench"])].append(r)
        by["p " + rate_bin(r[1]["p"])].append(r)
        by["parent " + r[1]["parent"]].append(r)
    out["test-like 0-199, AGGR - PRED by group (ratio-weighted, pairs)"] = {k: wmean(v) for k, v in sorted(by.items())}

    def pshare(regime, ids):
        ps = np.array([mr["p"] for i in ids if (regime, i) in metas for mr in metas[(regime, i)]])
        return {"n": len(ps), "mean_logit": r4(np.mean([T.logit_rate(p * 60, 60) for p in ps])),
                ">=0.7": r4(np.mean(ps >= .7)), "<0.1": r4(np.mean(ps < .1)), ">0.5": r4(np.mean(ps > .5))}
    out["pair-rate shares by regime (runs 0-199)"] = {
        r: pshare(r, range(200)) for r in ["tl", "r1b", "r1p", "tl lm-1.2", "tl lm-2.0", "tl mix/whole", "tl no shift"]}

    for reg in ("r1b", "r1p"):
        d = diffs(reg, range(100), AGGR, PRED)
        byb = defaultdict(list)
        for r in d:
            byb[r[1]["bench"]].append(r)
        out[f"{reg} 0-99, AGGR - PRED by benchmark"] = {k: wmean(v) for k, v in sorted(byb.items())}
        out[f"{reg} 0-99, AGGR - PRED pooled / without swe_rebench"] = [
            wmean(d), wmean([r for r in d if r[1]["bench"] != "swe_rebench"])]
        dh = diffs(reg, range(100), L.HIER, PRED)
        out[f"{reg} 0-99, hier default - PRED on swe_rebench"] = wmean([r for r in dh if r[1]["bench"] == "swe_rebench"])

    def by_kind(a, b, ids=range(100)):
        d = diffs("tl", ids, a, b)
        byk = defaultdict(list)
        for r in d:
            byk[T.kind_of(r[1]["bench"])].append(r)
        return {"all": wmean(d), **{k: wmean(v) for k, v in sorted(byk.items())}}
    out["selection half, by kind"] = {
        "sm 2.5 - sm 1.3 at mu0 -3, as 0.25": by_kind(AGGR, L.gname(-3.0, 1.3, 0.25)),
        "sm 2.5 - sm 1.8 at mu0 -3, as 0.25": by_kind(AGGR, L.gname(-3.0, 1.8, 0.25)),
        "as 0.25 - as 1 at mu0 -3, sm 2.5": by_kind(AGGR, L.gname(-3.0, 2.5, 1.0)),
        "AGGR - PRED": by_kind(AGGR, PRED),
        "MILD - PRED": by_kind(MILD, PRED),
        "sm 2.5 - sm 1.3 at mu0 -1, as 1": by_kind(L.gname(-1.0, 2.5, 1.0), L.gname(-1.0, 1.3, 1.0))}

    sel = st["summary"]["selected"]
    guarded = [r["config"] for r in sel["order"] if r["guard_ok"]]
    parents = sorted({mr["parent"] for i in range(200) for mr in metas[("tl", i)]})
    cache = {}

    def pp(name, ids):
        k = (name, tuple(ids))
        if k not in cache:
            cache[k] = diffs("tl", ids, name, PRED)
        return cache[k]

    nested = {}
    for q in parents:
        score = {n: wm([r for r in pp(n, range(100)) if r[1]["parent"] != q])
                 for n in guarded if pp(n, range(100)) is not None}
        pick = min(score, key=score.get)
        ev = {n: (wm([r for r in pp(n, range(100, 200)) if r[1]["parent"] == q])
                  if pp(n, range(100, 200)) else None) for n in dict.fromkeys([pick, AGGR, R35])}
        oracle = {n: wm([r for r in pp(n, range(100, 200)) if r[1]["parent"] == q])
                  for n in guarded if pp(n, range(100, 200))}
        best = min(oracle, key=oracle.get)
        nested[q] = {"pick_without_q": pick, "pick_on_q_confirmation": r4(ev[pick]),
                     "AGGR_on_q_confirmation": r4(ev[AGGR]),
                     "best_on_q_confirmation": [best, r4(oracle[best])], "n_guarded_scored": len(oracle)}
    out["nested leave-one-parent-out over the guarded configs (vs PRED)"] = {
        "guarded configs": len(guarded), "ordered configs": len(sel["order"]), "by held-out parent": nested}

    nested_g = {}
    for q in parents:
        sc_out, sc_in = {}, {}
        for n in L.GRID_NAMES:
            d = pp(n, range(100))
            if d is None:
                continue
            sc_out[n] = wm([r for r in d if r[1]["parent"] != q])
            sc_in[n] = wm([r for r in d if r[1]["parent"] == q])
        pick = min(sc_out, key=sc_out.get)
        best = min(sc_in, key=sc_in.get)
        nested_g[q] = {"pick": pick, "pick_on_q": r4(sc_in[pick]), "best_on_q": [best, r4(sc_in[best])],
                       "AGGR_on_q": r4(sc_in.get(AGGR)), "MILD_on_q": r4(sc_in.get(MILD)),
                       "regret": r4(sc_in[pick] - sc_in[best])}
    out["nested leave-one-parent-out over the Gaussian grids (selection half, vs PRED)"] = nested_g

    # how much the confirmation half repeats the selection half
    def apps(ids):
        return [mr for i in ids for mr in metas[("tl", i)]]
    s_app, c_app = apps(range(100)), apps(range(100, 200))
    s_pairs = {(m["bench"], m["subject"]) for m in s_app}
    s_clusters = {(m["parent"], m["subject"]) for m in s_app}
    s_bench, c_bench = {m["bench"] for m in s_app}, {m["bench"] for m in c_app}
    out["confirmation half against the selection half"] = {
        "confirmation appearances": len(c_app),
        "share whose (pseudo-benchmark, subject) pair is in the selection half":
            r4(np.mean([(m["bench"], m["subject"]) in s_pairs for m in c_app])),
        "share whose (parent, subject) cluster is in the selection half":
            r4(np.mean([(m["parent"], m["subject"]) in s_clusters for m in c_app])),
        "pseudo-benchmarks: selection, confirmation, both": [len(s_bench), len(c_bench), len(s_bench & c_bench)],
        "distinct pairs: selection, confirmation": [len(s_pairs), len({(m["bench"], m["subject"]) for m in c_app})],
        "configs in summary.selection": len(st["summary"]["selection"].get("configs", st["summary"]["selection"]))}
    return out


# --- mild: the shipped config from the stored subject-side rows -------------------------

def mild_rows():
    """{(regime, i): (meta, rows)} of experiments/subject_side.py's 'ship' arm,
    in level_calibration's row format (per-pair ALC, budget means)."""
    out = {}
    for f, reg in SS_FILES.items():
        path = os.path.join(SS_ROWS, f + ".jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                x = json.loads(line)
                if "ship" not in x["res"]:
                    continue
                b = np.array(x["res"]["ship"]["b"], float)
                key = (x["regime"], int(x["i"]))
                if key in out:
                    raise RuntimeError(f"two ship rows for {key}")
                out[key] = (x["meta"], {"alc": [float(WEIGHTS @ v) for v in b], "b": list(b.mean(0)),
                                        "ece0": [np.nan, np.nan], "ece_alc": np.nan,
                                        "q0": float(np.mean(x["res"]["ship"]["q0"]))})
    return out


def stage_mild(st, raw, metas):
    rows = mild_rows()
    if not rows:
        return {"skipped": f"no stored rows in {os.path.relpath(SS_ROWS, ROOT)} (python experiments/subject_side.py)"}
    raw = dict(raw)
    check = {"runs": 0, "pairs": 0, "max_abs_alc": 0.0, "meta_mismatch": 0, "runs_not_in_level_calibration": 0}
    for (reg, i), (meta, r) in sorted(rows.items()):
        lm = metas.get((reg, i))
        if lm is None:
            check["runs_not_in_level_calibration"] += 1
            continue
        if [(m["bench"], m["subject"]) for m in lm] != [(m["bench"], m["subject"]) for m in meta]:
            check["meta_mismatch"] += 1
            continue
        stored = L.rows_of(raw, reg, i, MILD)
        if stored is not None:
            check["runs"] += 1
            check["pairs"] += len(r["alc"])
            check["max_abs_alc"] = max(check["max_abs_alc"],
                                       float(np.max(np.abs(np.array(stored["alc"]) - np.array(r["alc"])))))
        else:
            raw[L.task_key((reg, i, (MILD,)))] = {"res": {MILD: r}}
    check["runs_checked_where_both_hold_MILD"] = check.pop("runs")
    out = {"source check (subject_side ship arm vs level_calibration's MILD rows)": check}
    if check["meta_mismatch"] or check["max_abs_alc"] > 1e-5:
        out["error"] = "the subject-side rows do not match level_calibration's runs"
        return out

    def cmp(reg, ids, names, ref):
        ids = [i for i in ids if (reg, i) in metas and all(L.rows_of(raw, reg, i, n) is not None for n in names + [ref])]
        c = L.compare(raw, metas, reg, ids, names, ref=ref)
        return {n: {"runs": len(ids), "ALC": [r4(v["ALC"][0]), r4(v["ALC"][1])],
                    "minus ref (run / cluster / stratified SE)": [r4(v["diff"]["mean"]), r4(v["diff"]["run_se"]),
                                                                  r4(v["diff"]["cluster_se"]), r4(v["diff"]["strat_se"])],
                    "budgets minus ref": [r4(x) for x in v["diff_budgets"]],
                    "leave one parent out": {k: r4(x) for k, x in v["diff"]["lopo"].items()}}
                for n, v in c.items()}

    sets = [("tl", range(100, 200), "confirmation half (audit: fresh runs)"),
            ("tl", range(100), "selection half"),
            ("tl", range(200), "test-like 0-199"),
            ("r1b", range(100), "public benchmark-first 0-99 (audit: fresh runs)"),
            ("r1p", range(100), "public pair-uniform 0-99 (audit: fresh runs)"),
            ("tl no shift", range(30), "no date shift 0-29 (audit: fresh runs)"),
            ("tl no shift", range(100), "no date shift 0-99"),
            ("tl mix/whole", range(100), "mix/whole 0-99 (not scored by the audit)")]
    for reg, ids, label in sets:
        key = f"{reg} {ids.start}-{ids.stop - 1}: {label}"
        out[key] = {"vs PRED": cmp(reg, list(ids), [MILD, AGGR], PRED),
                    "MILD minus AGGR": cmp(reg, list(ids), [MILD], AGGR).get(MILD)}
    out["feedback-matched estimate, MILD and AGGR on one pool"] = feedback_match_both(raw, metas)
    return out


def feedback_match_both(raw, metas):
    """stage_feedback's per-pair matched estimate for MILD and AGGR side by side,
    on the pool where PRED, AGGR and MILD are all stored (every level_calibration
    regime except level_mean -1.2 and -2.0, where MILD was never scored)."""
    pool = []
    for (regime, i) in sorted(metas):
        p, a, m = (L.rows_of(raw, regime, i, n) for n in (PRED, AGGR, MILD))
        if p is None or a is None or m is None or "pb" not in p:
            continue
        for j, mr in enumerate(metas[(regime, i)]):
            pool.append((regime, np.array(p["pb"][j]), a["alc"][j] - p["alc"][j], m["alc"][j] - p["alc"][j], mr["p"]))
    X = np.array([x[1] for x in pool]); sd = X.std(0)
    out = {"appearances": len(pool),
           "by_regime": {r: sum(1 for x in pool if x[0] == r) for r in sorted({x[0] for x in pool})}}
    for K in (15, 40):
        per_a, per_m, hi = [], [], []
        for _, _, v in T.FEEDBACK:
            nn = np.argsort(np.sqrt((((X - np.array(v)) / sd) ** 2).sum(1)))[:K]
            per_a.append(np.mean([pool[j][2] for j in nn]))
            per_m.append(np.mean([pool[j][3] for j in nn]))
            hi.append(round(float(np.mean([pool[j][4] > .5 for j in nn])), 2))
        out[f"K={K}"] = {"estimate AGGR ALC": r4(T.FEEDBACK_ALC + np.mean(per_a)),
                         "estimate MILD ALC": r4(T.FEEDBACK_ALC + np.mean(per_m)),
                         "per pair MILD - PRED": [r4(x) for x in per_m],
                         "per pair AGGR - PRED": [r4(x) for x in per_a], "share p>0.5": hi}
    return out


# --- extra: the audit's new regimes, re-run ------------------------------------------

def extra_tasks():
    tasks = []
    for reg in NEW_REGIMES:
        for i in range(N_NEW):
            tasks += [(reg, i, "base")] + [(reg, i, (c,)) for c in (AGGR, R35, MILD, EB)]
    return tasks


def _work(t):
    k, v = L.run_task(t)
    meta = L.describe(t[0], L.draw(t[0], t[1])) if t[2] == "base" else None
    return k, v, meta


def stage_extra(state, jobs, save):
    L.REGIMES.update(NEW_REGIMES)
    raw = state.setdefault("raw_extra", {})
    todo = [t for t in extra_tasks() if L.task_key(t) not in raw]
    print(f"extra: {len(todo)} tasks", flush=True)
    t0 = last = time.time()
    it = map(_work, todo) if jobs <= 1 else None
    pool = None
    if jobs > 1:
        import multiprocessing as mp
        L.pairs()
        pool = mp.get_context("fork").Pool(jobs)
        it = pool.imap_unordered(_work, todo)
    for n, (k, v, meta) in enumerate(it, 1):
        raw[k] = v
        if meta is not None:
            raw["meta|" + k.split("|")[0] + "|" + k.split("|")[1]] = meta
        if time.time() - last > 60 or n == len(todo):
            L.compact_raw(raw)
            save(state)
            last = time.time()
            print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close()
    L.compact_raw(raw)
    state.setdefault("extra_passes", []).append({"tasks": len(todo), "wall_s": round(time.time() - t0), "jobs": jobs})


def summarise_extra(state):
    raw = state.get("raw_extra")
    if not raw:
        return None
    L.REGIMES.update(NEW_REGIMES)
    L.register_chunks(L.grid_chunks())
    metas = L.collect_metas(raw)
    names = [AGGR, R35, MILD, EB, L.HIER, L.SMOOTHED, PFIX]
    out = {}
    for reg in NEW_REGIMES:
        ids = [i for i in range(N_NEW) if (reg, i) in metas]
        c = L.compare(raw, metas, reg, ids, names)
        ps = [mr["p"] for i in ids for mr in metas[(reg, i)]]
        pr = L.compare(raw, metas, reg, ids, [PRED], ref=L.SMOOTHED)
        out[reg] = {"runs": len(ids), "pair mean logit": r4(np.mean([T.logit_rate(p * 60, 60) for p in ps])),
                    "share p>0.5": r4(np.mean(np.array(ps) > .5)),
                    "PRED ALC": [r4(x) for x in pr[PRED]["ALC"]] if pr else None,
                    **{n: {"ALC": [r4(x) for x in v["ALC"]],
                           "minus PRED (run / cluster / stratified SE)": [r4(v["diff"]["mean"]), r4(v["diff"]["run_se"]),
                                                                         r4(v["diff"]["cluster_se"]), r4(v["diff"]["strat_se"])],
                           "budgets minus PRED": [r4(x) for x in v["diff_budgets"]]} for n, v in c.items()}}
        d = L.compare(raw, metas, reg, ids, [MILD], ref=AGGR)
        out[reg]["MILD minus AGGR (run / cluster SE)"] = [r4(d[MILD]["diff"]["mean"]), r4(d[MILD]["diff"]["run_se"]),
                                                          r4(d[MILD]["diff"]["cluster_se"])] if d else None
    return out


# --- io ----------------------------------------------------------------------------------

def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


LIB = ("paiec/hier.py", "paiec/prior.py", "paiec/subjects.py", "paiec/official.py", "paiec/testlike.py",
       "paiec/mcq.py", "paiec/predict.py", "paiec/fitting.py", "experiments/level_calibration.py")


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="default", choices=("default", "feedback", "analyse", "mild", "extra", "summarise"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    state = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            state = json.load(f)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    t0 = time.time()
    state["meta"] = {"script_digest": digest(["experiments/level_audit.py"]), "lib_digest": digest(LIB),
                     "commit": commit, "AGGR": AGGR, "MILD": MILD, "PRED": PRED,
                     "inputs": ["results/level_calibration.json", "data/subject_side_rows (stage mild)"],
                     "provenance": "session scratch step2b/audit (2026-09-26): feedback_match.py, feedback_match2.py, "
                                   "feedback_match3.py, analyse.py, extra_runs.py, extra_summary.py"}
    stages = ("feedback", "analyse", "mild") if a.stage == "default" else (a.stage,)
    if a.stage == "extra":
        stage_extra(state, a.jobs, lambda s: save_json(a.out, s))
        stages = ("summarise",)
    if any(s in ("feedback", "analyse", "mild") for s in stages):
        st, raw, metas = load_lc()
        for s in stages:
            print(f"stage {s}", flush=True)
            state[s] = {"feedback": stage_feedback, "analyse": stage_analyse, "mild": stage_mild}[s](st, raw, metas)
    if "summarise" in stages or a.stage == "default":
        ex = summarise_extra(state)
        if ex is not None:
            state["extra"] = ex
    state.setdefault("passes", []).append({"command": " ".join(sys.argv[1:]) or "(default)",
                                           "wall_s": round(time.time() - t0),
                                           "script_digest": state["meta"]["script_digest"],
                                           "lib_digest": state["meta"]["lib_digest"], "commit": commit})
    save_json(a.out, state)
    show(state)


def show(state):
    fb = state.get("feedback")
    if fb:
        print("feedback-matched AGGR estimate: K=15 %.4f, K=40 %.4f; range %s" % (
            fb["K=15"]["estimate AGGR ALC"], fb["K=40"]["estimate AGGR ALC"],
            fb["AGGR estimate range (pools holding test-like runs)"]))
        print("  all screened configs, AGGR / MILD: K=15 %s, K=40 %s" % (
            fb["all screened configs"]["K=15 AGGR, MILD"], fb["all screened configs"]["K=40 AGGR, MILD"]))
    an = state.get("analyse")
    if an:
        g = an["test-like 0-199, AGGR - PRED by group (ratio-weighted, pairs)"]
        print("AGGR - PRED by rate:", {k: v for k, v in g.items() if k.startswith("p ")})
        print("public r1b by benchmark:", an["r1b 0-99, AGGR - PRED by benchmark"])
        print("overlap:", an["confirmation half against the selection half"])
    mi = state.get("mild")
    if mi:
        for k, v in mi.items():
            if isinstance(v, dict) and "MILD minus AGGR" in v:
                m = v["MILD minus AGGR"]
                print(f"  {k}: MILD - AGGR {m['minus ref (run / cluster / stratified SE)'] if m else None}; "
                      f"MILD - PRED {v['vs PRED'].get(MILD, {}).get('minus ref (run / cluster / stratified SE)')}")
            else:
                print(" ", k, v)
    if state.get("extra"):
        for reg, v in state["extra"].items():
            print(reg, {n: v[n]["minus PRED (run / cluster / stratified SE)"] for n in (AGGR, MILD) if n in v},
                  "MILD-AGGR", v.get("MILD minus AGGR (run / cluster SE)"))


if __name__ == "__main__":
    main()
