"""W3: the level calibration by model, as a paired 2 x 2 factorial on the
regime-sensitivity runs, with its interaction and paired SEs.

The report (v2 outline, section 2.6 "Calibration by model, paired", Table T3,
App E.4) reads the level calibration and the model as one factorial:

                       default level           calibrated level
    smoothed mean      Smooth     (smooth)     Smooth-cal (smcal)
    hier               hier-fit   (eb_fit)     hier-ship  (ship)

with LegacyP (legacy), our first submission, beside it as a reference cell.
Smooth is the Beta(2, 2)-smoothed mean (n0 4, m0 0.5); Smooth-cal the smoothed
mean at the (n0, m0) that LEVEL's selection rule chose (the rows record it);
hier-fit is paiec.hier.HierPredictor with its level prior as the empirical-Bayes
fit gives it; hier-ship is the shipped LEVEL (mu0 -2.5, sigma_mu 2.5,
attr_scale 0.5). Both calibrated levels were chosen by the same selection rule
on the tuned regime.

experiments/regime_sensitivity.py (RS) scored every cell on identical
checkpoints of the same runs (scoring seed 11; test-like regimes TUNED,
READING, AUDIT, MIXTURE, FLAT, runs 0-79; public R1B and R1P, split scope
'pair', runs 0-59), so every contrast below is paired per pair appearance. RS
stored each two-cell difference with ship, smooth or legacy as the reference,
but not the interaction, which needs all four cells at once; the outline's
interaction range (-0.0125 to -0.0155) was derived from stored means and has no
SE. This script reads RS's stored rows only (data/regime_sensitivity_rows/,
gitignored) and results/regime_sensitivity.json for the checks; it scores
nothing and selects, tunes and ships nothing.

Sign. Every contrast is Delta = ALC(A) - ALC(B) (or a signed sum of cells),
negative when the first-named side is better:

    calibration_in_hier      ship - eb_fit
    calibration_in_smooth    smcal - smooth
    model_at_default         eb_fit - smooth
    model_at_calibrated      ship - smcal
    calibration_main         (calibration_in_hier + calibration_in_smooth) / 2
    model_main               (model_at_default + model_at_calibrated) / 2
    interaction              ship - eb_fit - smcal + smooth
                             = calibration_in_hier - calibration_in_smooth
                             = model_at_calibrated - model_at_default
    legacy_minus_ship        legacy - ship    (T3's first row; RS stores it)
    legacy_minus_smooth      legacy - smooth

A negative interaction means calibrating the level helps hier more than it
helps the smoothed mean (equivalently, hier's structure pays more at the
calibrated level). T3 prints "comparator minus hier-ship"; the two-cell
contrasts here that have ship as one side are those numbers with the stated
sign.

Statistics (RS's own, so every two-cell contrast reproduces RS's stored value
bit for bit, which `checks` verifies): D is the mean over a regime's runs of the
run's pair-mean contrast; run_se the SE of that mean over runs; cluster_se the
cluster bootstrap of experiments/level_calibration.py (Boot, 2,000 resamples,
bootstrap seed 0, ratio estimator, clusters (parent, subject) on test-like runs
and (benchmark, subject) on public runs), the SE RS's rule and T3 use; strat_se
the same with clusters resampled within each parent; pct the bootstrap's 2.5
and 97.5 percentiles. Runs reuse the same public pairs, so run_se understates
the uncertainty and cluster_se is the one to quote. Per budget and per parent
tables come from experiments/ship_confirm.py's Block (rounded to 1e-6), as RS's
tables do. The regimes share clusters, so the per-regime SEs are not
independent and no pooled interaction over regimes is computed.

Run:
    python experiments/factorial_rs.py            # about 5 s
    python experiments/factorial_rs.py --rows DIR --rs PATH --out PATH

Output: results/factorial_rs.json (OUT). Exit status 1 if a check fails (the
file is written either way, with the failing check recorded).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.regime_sensitivity as RS  # noqa: E402
import experiments.ship_confirm as SC  # noqa: E402
from paiec.evaluator import BUDGETS  # noqa: E402

OUT = os.path.join(ROOT, "results", "factorial_rs.json")
RS_JSON = RS.OUT
ROWS = RS.ROWS

#: the report's name of each cell -> RS's config key
CELLS = {"Smooth": "smooth", "Smooth-cal": "smcal", "hier-fit": "eb_fit",
         "hier-ship": "ship", "LegacyP": "legacy"}
#: the 2 x 2 layout: (model, level) -> config key
FACTORIAL = {("smoothed mean", "default"): "smooth", ("smoothed mean", "calibrated"): "smcal",
             ("hier", "default"): "eb_fit", ("hier", "calibrated"): "ship"}
#: contrast -> signed cells, accumulated in this order (for a two-cell contrast, A then
#: B, so A + (-1) B is A - B exactly, as RS's diff computes it)
CONTRASTS = {
    "calibration_in_hier": (("ship", 1.0), ("eb_fit", -1.0)),
    "calibration_in_smooth": (("smcal", 1.0), ("smooth", -1.0)),
    "model_at_default": (("eb_fit", 1.0), ("smooth", -1.0)),
    "model_at_calibrated": (("ship", 1.0), ("smcal", -1.0)),
    "calibration_main": (("ship", 0.5), ("eb_fit", -0.5), ("smcal", 0.5), ("smooth", -0.5)),
    "model_main": (("eb_fit", 0.5), ("smooth", -0.5), ("ship", 0.5), ("smcal", -0.5)),
    "interaction": (("ship", 1.0), ("eb_fit", -1.0), ("smcal", -1.0), ("smooth", 1.0)),
    "legacy_minus_ship": (("legacy", 1.0), ("ship", -1.0)),
    "legacy_minus_smooth": (("legacy", 1.0), ("smooth", -1.0)),
}
DEFINITIONS = {
    "calibration_in_hier": "hier-ship - hier-fit (ship - eb_fit)",
    "calibration_in_smooth": "Smooth-cal - Smooth (smcal - smooth)",
    "model_at_default": "hier-fit - Smooth (eb_fit - smooth)",
    "model_at_calibrated": "hier-ship - Smooth-cal (ship - smcal)",
    "calibration_main": "(calibration_in_hier + calibration_in_smooth) / 2",
    "model_main": "(model_at_default + model_at_calibrated) / 2",
    "interaction": "ship - eb_fit - smcal + smooth = calibration_in_hier - calibration_in_smooth "
                   "= model_at_calibrated - model_at_default",
    "legacy_minus_ship": "LegacyP - hier-ship (legacy - ship)",
    "legacy_minus_smooth": "LegacyP - Smooth (legacy - smooth)",
}
TESTLIKE = RS.TESTLIKE
PUBLIC = RS.PUBLIC_REGIMES
Z95 = 1.96
#: the outline's derived interaction range over the test-like regimes, at four decimals
OUTLINE_INTERACTION_RANGE = (-0.0155, -0.0125)
#: two-cell contrasts against RS's stored, unrounded vs_ship[R][X].exact: (contrast,
#: X, sign) with contrast = sign * (X - ship)
RS_EXACT = (("calibration_in_hier", "eb_fit", -1.0), ("model_at_calibrated", "smcal", -1.0),
            ("legacy_minus_ship", "legacy", 1.0))
#: two-cell contrasts against RS's rounded tables: (contrast, table, X, sign) with
#: contrast = sign * (X - ref) and ref the table's reference
RS_TABLES = (("calibration_in_smooth", "vs_smooth", "smcal", 1.0),
             ("model_at_default", "vs_smooth", "eb_fit", 1.0),
             ("legacy_minus_smooth", "vs_smooth", "legacy", 1.0),
             ("legacy_minus_ship", "vs_legacy", "ship", -1.0))
TABLE_TOL = 5e-7                   # the tables are rounded to 6 decimals
EXACT_TOL = 0.0
#: a contrast stored with the opposite sign: its percentiles are the stored ones negated
#: and swapped, equal up to numpy's linear interpolation, which is not symmetric in the
#: last bit (D and every SE are equal exactly)
PCT_FLIP_TOL = 1e-15
DERIVED_TOL = 1e-12
#: per budget values are each rounded to 1e-6, so a difference of two holds to 1e-6
BUDGET_ALGEBRA_TOL = 1.0000001e-6


# --- small helpers --------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def r6(x):
    return None if x is None else round(float(x), 6)


# --- the contrasts ----------------------------------------------------------------------------

class Regime:
    """One regime's stored rows: RS's RegimeRows (cells, bootstrap, weights) and
    ship_confirm's Block over the same runs (per budget and per parent tables)."""

    def __init__(self, R, rows):
        self.R = R
        self.G = RS.RegimeRows(R, rows)
        self.block = SC.Block(self.G.runs_of(RS.SHIP), self.G.ids)

    def combine(self, weights):
        """Per run: the contrast's per pair ALC (n,) and Brier (n, 6)."""
        G = self.G
        A, B = [], []
        for j in range(len(G.ids)):
            a = np.zeros_like(G.A[weights[0][0]][j])
            b = np.zeros_like(G.B[weights[0][0]][j])
            for c, w in weights:
                a = a + w * G.A[c][j]
                b = b + w * G.B[c][j]
            A.append(a)
            B.append(b)
        return A, B

    def stats(self, d):
        """RS's statistics of a per run list of per pair contrasts (as RegimeRows.diff
        computes them, unrounded)."""
        G = self.G
        run_d = np.array([float(np.mean(v)) for v in d])
        c, s = G.boot.se(d)
        rse = float(run_d.std(ddof=1) / math.sqrt(len(run_d))) if len(run_d) > 1 else None
        flat = np.concatenate(d)
        v = np.bincount(G.cl, G.w * flat, minlength=G.K)
        n = np.bincount(G.cl, G.w, minlength=G.K)
        dist = (G.boot.W @ v) / (G.boot.W @ n)
        D = float(run_d.mean())
        return {"D": D, "run_se": rse, "cluster_se": c, "strat_se": s,
                "ci95_cluster": [D - Z95 * c, D + Z95 * c],
                "pct": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                "z_cluster": D / c if c else None,
                "share_runs_negative": float(np.mean(run_d < 0))}

    def contrast(self, weights):
        A, B = self.combine(weights)
        out = self.stats(A)
        rows = []
        for k, bud in enumerate(BUDGETS):
            col = [x[:, k] for x in B]
            r, c, s = self.block.se3(col)
            m = float(np.mean([np.mean(x) for x in col]))
            rows.append({"budget": bud, "diff": r6(m), "run_se": r6(r), "cluster_se": r6(c),
                         "strat_se": r6(s), "alc_part": r6(RS.W6[k] * m)})
        out["budgets"] = rows
        per, level = self.block.per_parent(A)
        out["per_parent"] = per
        out["parent_level"] = level
        return out

    def cell(self, c):
        s = self.G.config_summary(c)
        s["single_run_sd"] = float(np.std([np.mean(a) for a in self.G.A[c]], ddof=1))
        return s


# --- checks ---------------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, **detail):
        self.items.append({"name": name, "ok": bool(ok), **detail})
        return ok

    def failed(self):
        return [c["name"] for c in self.items if not c["ok"]]


def maxabs(pairs):
    m = 0.0
    for a, b in pairs:
        if a is None or b is None:
            if a is not b:
                return math.inf
            continue
        m = max(m, abs(float(a) - float(b)))
    return m


def check_rows(rows, rows_dir, rs_state, checks):
    """Every planned run of every regime is there, scored by RS's scored script with
    one library, one smcal point (RS's chosen one) and the composition RS drew."""
    have = {R: sorted(rows[R]) for R in RS.REGIMES}
    complete = all(have[R] == list(range(RS.N_RUNS[R])) for R in RS.REGIMES)
    cells_present = all(set(CELLS.values()) <= set(r["res"]) for rs in rows.values() for r in rs.values())
    checks.add("rows_complete", complete and cells_present,
               runs={R: len(v) for R, v in have.items()}, planned=dict(RS.N_RUNS))
    scripts = sorted({r["script"] for rs in rows.values() for r in rs.values()})
    checks.add("rows_scored_script", scripts == [RS.SCORED_SCRIPT_SHA256[:16]], script_digests=scripts)
    libs = {json.dumps(r["lib"], sort_keys=True) for rs in rows.values() for r in rs.values()}
    checks.add("rows_one_library", len(libs) == 1, libraries=len(libs))
    smc = sorted({tuple(r["smcal"]) for rs in rows.values() for r in rs.values()})
    chosen = tuple(rs_state.get("smcal", {}).get("chosen", ()))
    checks.add("rows_smcal_is_rs_choice", smc == [chosen], rows=[list(x) for x in smc], chosen=list(chosen))
    cpath = os.path.join(rows_dir, RS.COMPOSITIONS)
    comps = {}
    if os.path.exists(cpath):
        with open(cpath) as f:
            comps = json.load(f)["runs"]
    bad = [f"{R}|{i}" for R, rs in rows.items() for i, r in rs.items()
           if comps.get(R, {}).get(str(i)) != r["meta"]]
    checks.add("rows_compositions", bool(comps) and not bad, mismatches=len(bad))
    return chosen


def check_against_rs(regs, cells, contrasts, summary, checks):
    """Cells and two-cell contrasts against RS's stored summary; the interaction against
    RS's stored unrounded differences."""
    pairs = []
    for R, cs in cells.items():
        for c in CELLS.values():
            want = summary["configs"][R][c]
            got = cs[c]
            pairs += [(got[k], want[k]) for k in ("ALC", "ALC_run_se", "ALC_cluster_se", "q0", "ece0",
                                                   "ece_alc", "runs")]
            pairs += list(zip(got["budgets"], want["budgets"]))
    checks.add("cells_equal_rs_configs", maxabs(pairs) <= EXACT_TOL, max_abs=maxabs(pairs),
               keys="summary.configs.<R>.<cell>.{ALC, ALC_run_se, ALC_cluster_se, q0, ece0, ece_alc, budgets}")
    pairs, flipped = [], []
    for R in contrasts:
        for name, X, sign in RS_EXACT:
            ex = summary["vs_ship"][R][X]["exact"]
            got = contrasts[R][name]
            pairs += [(got["D"], sign * ex["D"]), (got["run_se"], ex["run_se"]),
                      (got["cluster_se"], ex["cluster_se"]), (got["strat_se"], ex["strat_se"])]
            lo, hi = ex["pct"]
            if sign > 0:
                pairs += list(zip(got["pct"], [lo, hi]))
            else:
                flipped += list(zip(got["pct"], [-hi, -lo]))
    # RS's RegimeRows.diff on the same rows, recomputed now, for every two-cell contrast
    for R, reg in regs.items():
        for name, w in CONTRASTS.items():
            if len(w) == 2 and w[0][1] == 1.0 and w[1][1] == -1.0:
                d = reg.G.diff(w[0][0], w[1][0])
                got = contrasts[R][name]
                pairs += [(got[k], d[k]) for k in ("D", "run_se", "cluster_se", "strat_se")]
                pairs += list(zip(got["pct"], d["pct"]))
    m, mf = maxabs(pairs), maxabs(flipped)
    checks.add("pairs_equal_rs_exact", m <= EXACT_TOL and mf <= PCT_FLIP_TOL, max_abs=m,
               max_abs_flipped_percentiles=mf, tol=[EXACT_TOL, PCT_FLIP_TOL],
               keys="summary.vs_ship.<R>.{eb_fit, smcal, legacy}.exact and RegimeRows.diff")
    pairs = []
    for R in contrasts:
        for name, tab, X, sign in RS_TABLES:
            t = summary[tab][R][X]
            got = contrasts[R][name]
            pairs += [(got["D"], sign * t["diff"]["mean"]), (got["run_se"], t["diff"]["run_se"]),
                      (got["cluster_se"], t["diff"]["cluster_se"]), (got["strat_se"], t["diff"]["strat_se"])]
            for gb, tb in zip(got["budgets"], t["budgets"]):
                pairs += [(gb["diff"], sign * tb["diff"]), (gb["cluster_se"], tb["cluster_se"]),
                          (gb["run_se"], tb["run_se"])]
            for q, pq in t["per_parent"].items():
                pairs += [(got["per_parent"][q]["mean"], sign * pq["mean"]),
                          (got["per_parent"][q]["cluster_se"], pq["cluster_se"])]
    m = maxabs(pairs)
    checks.add("pairs_equal_rs_tables", m <= TABLE_TOL, max_abs=m, tol=TABLE_TOL,
               keys="summary.{vs_smooth, vs_legacy}.<R>.<X>.{diff, budgets, per_parent}")
    # the interaction and main effects against RS's stored unrounded D's, and the algebra
    pairs, alg, alg_b = [], [], []
    for R in contrasts:
        ex = {X: summary["vs_ship"][R][X]["exact"]["D"] for X in ("eb_fit", "smcal", "smooth")}
        c = contrasts[R]
        pairs.append((c["interaction"]["D"], -ex["eb_fit"] - ex["smcal"] + ex["smooth"]))
        pairs.append((c["calibration_in_smooth"]["D"], ex["smcal"] - ex["smooth"]))
        pairs.append((c["model_at_default"]["D"], ex["eb_fit"] - ex["smooth"]))
        alg.append((c["interaction"]["D"], c["calibration_in_hier"]["D"] - c["calibration_in_smooth"]["D"]))
        alg.append((c["interaction"]["D"], c["model_at_calibrated"]["D"] - c["model_at_default"]["D"]))
        alg.append((c["calibration_main"]["D"],
                    0.5 * (c["calibration_in_hier"]["D"] + c["calibration_in_smooth"]["D"])))
        alg.append((c["model_main"]["D"], 0.5 * (c["model_at_default"]["D"] + c["model_at_calibrated"]["D"])))
        for b in range(len(BUDGETS)):
            alg_b.append((c["interaction"]["budgets"][b]["diff"],
                          c["calibration_in_hier"]["budgets"][b]["diff"]
                          - c["calibration_in_smooth"]["budgets"][b]["diff"]))
    m = maxabs(pairs)
    checks.add("interaction_equals_rs_stored_differences", m <= DERIVED_TOL, max_abs=m, tol=DERIVED_TOL,
               keys="-vs_ship.<R>.eb_fit.exact.D - vs_ship.<R>.smcal.exact.D + vs_ship.<R>.smooth.exact.D")
    m_d, m_b = maxabs(alg), maxabs(alg_b)
    checks.add("factorial_algebra", m_d <= DERIVED_TOL and m_b <= BUDGET_ALGEBRA_TOL, max_abs_D=m_d,
               max_abs_budget_rounded=m_b, tol=[DERIVED_TOL, BUDGET_ALGEBRA_TOL])


# --- summary over regimes -------------------------------------------------------------------

def regime_summary(contrasts):
    out = {}
    for name in CONTRASTS:
        tl = [contrasts[R][name] for R in TESTLIKE if R in contrasts]
        pub = {R: contrasts[R][name] for R in PUBLIC if R in contrasts}
        D = [c["D"] for c in tl]
        out[name] = {
            "definition": DEFINITIONS[name],
            "by_regime": {R: {"D": contrasts[R][name]["D"], "cluster_se": contrasts[R][name]["cluster_se"],
                              "run_se": contrasts[R][name]["run_se"], "z_cluster": contrasts[R][name]["z_cluster"]}
                          for R in contrasts},
            "testlike": {"D_range": [min(D), max(D)] if D else None,
                         "D_mean_unweighted": float(np.mean(D)) if D else None,
                         "cluster_se_range": [min(c["cluster_se"] for c in tl), max(c["cluster_se"] for c in tl)]
                         if tl else None,
                         "z_range": [min(c["z_cluster"] for c in tl), max(c["z_cluster"] for c in tl)] if tl else None,
                         "all_negative": all(d < 0 for d in D) if D else None,
                         "all_ci95_below_zero": all(c["ci95_cluster"][1] < 0 for c in tl) if tl else None,
                         "all_ci95_above_zero": all(c["ci95_cluster"][0] > 0 for c in tl) if tl else None},
            "public": {R: [c["D"], c["cluster_se"]] for R, c in pub.items()},
        }
    return out


def outline_check(summ):
    """The outline's derived statements, re-read at four decimals (descriptive)."""
    it = summ["interaction"]["testlike"]["D_range"]
    got = (round(it[0], 4), round(it[1], 4))
    return {"interaction_testlike_range_4dp": list(got),
            "outline_interaction_range": list(OUTLINE_INTERACTION_RANGE),
            "matches": got == OUTLINE_INTERACTION_RANGE,
            "calibration_in_hier_testlike_range_4dp":
                [round(x, 4) for x in summ["calibration_in_hier"]["testlike"]["D_range"]],
            "calibration_in_smooth_testlike_range_4dp":
                [round(x, 4) for x in summ["calibration_in_smooth"]["testlike"]["D_range"]],
            "model_at_default_testlike_range_4dp":
                [round(x, 4) for x in summ["model_at_default"]["testlike"]["D_range"]],
            "model_at_calibrated_by_regime_4dp":
                {R: round(v["D"], 4) for R, v in summ["model_at_calibrated"]["by_regime"].items()}}


# --- provenance -----------------------------------------------------------------------------

def rows_digests(rows_dir):
    """Per regime: the number of row files and the sha256 of their sorted listing
    "<regime>/<i>.json <sha256>" (one line each); the same over every regime."""
    out, every = {}, []
    for R in RS.REGIMES:
        lines = []
        for i in range(RS.N_RUNS[R]):
            p = RS.row_path(R, i, rows=rows_dir)
            if os.path.exists(p):
                lines.append(f"{R}/{i}.json {sha256(p)}")
        every += lines
        out[R] = {"files": len(lines), "listing_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
    out["all"] = {"files": len(every), "listing_sha256": hashlib.sha256("\n".join(every).encode()).hexdigest()}
    return out


def provenance(rows_dir, rs_path, out_path, secs):
    me = os.path.abspath(__file__)
    code = {rel(p): sha256(p) for p in (me, RS.__file__, LC.__file__, SC.__file__)}
    inputs = {rel(rs_path): sha256(rs_path)}
    cpath = os.path.join(rows_dir, RS.COMPOSITIONS)
    if os.path.exists(cpath):
        inputs[rel(cpath)] = sha256(cpath)
    return {"command": " ".join(["python", rel(sys.argv[0])] + sys.argv[1:]),
            "script": rel(me), "script_sha256": code[rel(me)], "code_sha256": code,
            "git_head": git("rev-parse", "HEAD"),
            "script_status": git("status", "--porcelain", "--", rel(me)) or "clean",
            "inputs_sha256": inputs, "rows_dir": rel(rows_dir), "rows": rows_digests(rows_dir),
            "out": rel(out_path), "when_utc": utcnow(), "python": platform.python_version(),
            "numpy": np.__version__, "machine": platform.machine(), "wall_s": round(secs, 2)}


# --- main -----------------------------------------------------------------------------------

def compute(rows, rows_dir=ROWS, rs_path=RS_JSON):
    """Everything but provenance: (result dict, Checks)."""
    with open(rs_path) as f:
        rs_state = json.load(f)
    checks = Checks()
    chosen = check_rows(rows, rows_dir, rs_state, checks)
    regs = {R: Regime(R, rows[R]) for R in RS.REGIMES if rows[R]}
    cells = {R: {c: reg.cell(c) for c in CELLS.values()} for R, reg in regs.items()}
    contrasts = {R: {name: reg.contrast(w) for name, w in CONTRASTS.items()} for R, reg in regs.items()}
    check_against_rs(regs, cells, contrasts, rs_state["summary"], checks)
    summ = regime_summary(contrasts)
    res = {
        "about": {
            "what": "the level calibration by model as a paired 2 x 2 factorial on RS's stored runs, with "
                    "the interaction and its paired SEs (report v2, section 6.3 and App E.4; work item W3)",
            "sign": "Delta = ALC(A) - ALC(B), or the signed sum of cells in `definition`; negative when the "
                    "first-named side is better. Table T3 prints comparator minus hier-ship",
            "se": "cluster_se: level_calibration.Boot over (parent, subject) clusters (test-like) or "
                  "(benchmark, subject) clusters (public), 2,000 resamples, seed 0, ratio estimator, as in RS; "
                  "run_se: SE over runs, which understates (runs reuse public pairs); per regime only, the "
                  "regimes share clusters",
            "selects": "nothing: descriptive, report only, scores nothing",
        },
        "design": {
            "cells": CELLS,
            "factorial": {f"{m} | {lv}": c for (m, lv), c in FACTORIAL.items()},
            "smooth": {"n0": RS.CONFIGS["smooth"]["n0"], "m0": RS.CONFIGS["smooth"]["m0"]},
            "smcal": {"n0": chosen[0] if chosen else None, "m0": chosen[1] if chosen else None},
            "ship_level": dict(RS.LEVEL),
            "eb_fit": "HierPredictor over level_calibration.bundle(parent, 0.0) with the level prior as fitted",
            "contrasts": {k: [[c, w] for c, w in v] for k, v in CONTRASTS.items()},
            "regimes": {R: {"runs": len(regs[R].G.ids), "appearances": int(len(regs[R].G.w)),
                            "clusters": regs[R].G.K, "kind": RS.REGIMES[R][0]} for R in regs},
            "scoring_seed": RS.SCORE_SEED, "boots": RS.BOOTS, "boot_seed": RS.BOOT_SEED,
            "testlike": list(TESTLIKE), "public": list(PUBLIC),
        },
        "cells": cells,
        "contrasts": contrasts,
        "summary": summ,
        "outline_check": outline_check(summ),
    }
    res["checks"] = checks.items
    return res, checks


def show(res):
    names = ("calibration_in_hier", "calibration_in_smooth", "model_at_default", "model_at_calibrated",
             "interaction", "legacy_minus_ship")
    print("cell ALC by regime: " + ", ".join(CELLS))
    for R, cs in res["cells"].items():
        print(f"  {R:8s} " + "  ".join(f"{cs[c]['ALC']:.4f}" for c in CELLS.values()))
    print("contrasts, D (cluster SE):")
    print("  " + " " * 8 + "".join(f"{n[:21]:>22s}" for n in names))
    for R, cs in res["contrasts"].items():
        print(f"  {R:8s}" + "".join(f"{cs[n]['D']:+11.4f} ({cs[n]['cluster_se']:.4f})"[:22].rjust(22)
                                    for n in names))
    it = res["summary"]["interaction"]["testlike"]
    print(f"interaction, test-like: {it['D_range'][0]:+.4f} to {it['D_range'][1]:+.4f}, cluster SE "
          f"{it['cluster_se_range'][0]:.4f} to {it['cluster_se_range'][1]:.4f}, z {it['z_range'][0]:+.2f} to "
          f"{it['z_range'][1]:+.2f}; outline range matches: {res['outline_check']['matches']}")
    bad = [c["name"] for c in res["checks"] if not c["ok"]]
    print("checks: " + ("all passed" if not bad else "FAILED " + ", ".join(bad)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default=ROWS, help="RS row directory (default data/regime_sensitivity_rows)")
    ap.add_argument("--rs", default=RS_JSON, help="RS results file (default results/regime_sensitivity.json)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = RS.load_rows(a.rows)
    if not any(rows.values()):
        raise SystemExit(f"no RS rows under {a.rows}")
    res, checks = compute(rows, a.rows, a.rs)
    res["provenance"] = provenance(a.rows, a.rs, a.out, time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(RS.clean(res), f, indent=1)
        f.write("\n")
    os.replace(tmp, a.out)
    show(res)
    print(f"wrote {rel(a.out)} ({res['provenance']['wall_s']} s)")
    return 1 if checks.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
