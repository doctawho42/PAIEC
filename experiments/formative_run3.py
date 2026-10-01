"""Formative run 3: the record of a regression and latency check, beside runs 1 and 2.

Formative run 3 (scored 2026-10-01) ran the archive built at 4d2cc4f (sha256
4a882cc7...): paiec.hier.HierPredictor with LEVEL mu0 -2.5, sigma_mu 2.5,
attr_scale 0.5, the corrected multiple-choice floor (f7e7d87) and the
floored-fit fix (4d2cc4f). The team uploaded it as a regression and latency
check only. That its score selects or tunes nothing was written down before it
was scored (results/formative_feedback.json 'submissions', committed in
00bdf04). Its feedback table is results/formative/run3.txt, the organisers'
output copied verbatim like run1.txt and run2.txt.

This script records run 3 the way the `record` stage of
experiments/formative_feedback.py records runs 1 and 2, and does nothing else:

  - parses the table with formative_feedback.parse, recomputes the run's ALC
    from the budgets with the official weights, pair-averaged, and checks
    every pair against its summary row (to 1e-6);
  - counts pairs, benchmarks and subjects, and gives the per-budget Brier and
    ECE means;
  - hashes the archive and checks its members against 4d2cc4f
    (formative_feedback.verify_archive). By default that is the current
    submission archive under dist/, which was handed to the team for upload
    and not downloaded back; --archive PATH records a downloaded copy instead.
    dist/ is gitignored and rebuilt by tools/build_submission.py, so the
    default read's archive checks are required: re-verifying needs dist/ to
    hold the 4a882cc7... file, and without it nothing is written. A copy given
    with --archive is recorded whatever its bytes (a difference would be the
    finding), but a stored record whose archive matched is not replaced by one
    whose archive does not without --replace-archive-record;
  - gives the ID overlap with runs 1 and 2 (subjects, benchmarks, (subject,
    benchmark) pairs);
  - places run 3 against the shipped model's single-run sds, by budget and in
    ALC, from results/ship_confirm.json (z-scores in the tuned test-like
    regime, its two variants and the two public weightings), and sets the
    three runs side by side.

No level reading, matching or tuning is done on run 3. The preregistered
decision rule of the pooled reading (results/formative_feedback.json) covered
runs 1 and 2 only. experiments/formative_feedback.py and its results file are
only read, never written: that script's code digests are audited by
experiments/script_revisions.py. Importing it runs nothing (its stages run
from main() only).

Run:  python experiments/formative_run3.py [--archive PATH --archive-how TEXT]
                                            [--replace-archive-record]
Writes results/formative_run3.json, and only if every required check passes
(seconds, one process, well under 0.5 GB; no model, nothing from data/).
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import formative_feedback as FF  # noqa: E402  (read-only use)
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402

OUT = os.path.join(ROOT, "results", "formative_run3.json")
FEEDBACK = {"run1": FF.FEEDBACK["run1"], "run2": FF.FEEDBACK["run2"],
            "run3": os.path.join(ROOT, "results", "formative", "run3.txt")}
SHIP_JSON = os.path.join(ROOT, "results", "ship_confirm.json")
FF_JSON = FF.OUT
#: the current submission archive (tools/build_submission.py's default output)
ARCHIVE = os.path.join(ROOT, "dist", "paiec" + ".zip")
W = np.asarray(WEIGHTS, float)
PAIR_TOL = 1e-6          # six budgets at 6 dp, weights summing to 1, plus the row's own rounding
PURPOSE_PHRASE = "planned as a regression and latency check only"

RUN3 = {
    "scored": "2026-10-01",
    "commit": "4d2cc4f",
    "archive_sha256": "4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090",
    "model": "paiec.hier.HierPredictor, LEVEL mu0 -2.5 sigma_mu 2.5 attr_scale 0.5, the corrected "
             "multiple-choice floor (f7e7d87) and the floored-fit fix (4d2cc4f)",
    "level": {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5},
    "purpose": "a regression and latency check only: that the archive selected at 4d2cc4f runs on the "
               "platform with every pair scored and no errors. Its score selects or tunes nothing.",
    "uploaded_by": "the team",
    "team_report": "all 9 pairs scored, no errors",
    "platform_alc": None,
    "platform_alc_note": "the platform's headline score for run 3 was not pasted, so the table's "
                         "pair-mean ALC is recomputed here and cannot be checked against a reported score",
}
#: the meta of runs 1 and 2 as the record stage holds it
RUNS = {"run1": FF.RUNS["run1"], "run2": FF.RUNS["run2"], "run3": RUN3}
#: ship_confirm.json's regimes, grouped as the task describes them
GROUPS = {"tl": "tuned test-like (primary)", "tl mix/whole": "test-like variant",
          "tl no shift": "test-like variant", "r1b": "public", "r1p": "public"}


def r6(x):
    return None if x is None else round(float(x), 6)


def rel(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def sha256_file(path):
    return FF.sha256_file(path) if os.path.exists(path) else None


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


class Checks:
    """Every check, recorded; required ones stop the write if they fail."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, got, want, required=True):
        self.items.append({"check": name, "ok": bool(ok), "got": got, "want": want,
                           "required": required})
        return bool(ok)

    def failed(self, required_only=True):
        return [c for c in self.items if not c["ok"] and (c["required"] or not required_only)]


# --- the record ---------------------------------------------------------------------------

def summarise(pairs):
    """The record stage's per-run summary (formative_feedback.summarise_run)
    without its run meta: counts, ALC, per-budget Brier and ECE."""
    br = np.array([p["brier"] for p in pairs])
    ec = np.array([p["ece_b"] for p in pairs])
    per_pair = br @ W
    alc = FF.run_alc(pairs)
    item_w = float(np.dot([p["alc"] for p in pairs], [p["n"] for p in pairs]) /
                   sum(p["n"] for p in pairs))
    return {
        "counts": {"pairs": len(pairs), "benchmarks": len({p["benchmark"] for p in pairs}),
                   "subjects": len({p["subject"] for p in pairs}),
                   "pairs_per_benchmark": sorted(Counter(p["benchmark"] for p in pairs).values(),
                                                 reverse=True),
                   "subject_item_pairs": int(sum(p["n"] for p in pairs)),
                   "n_min": int(min(p["n"] for p in pairs)), "n_max": int(max(p["n"] for p in pairs))},
        "alc": {"recomputed": alc,
                "recomputed_6dp": round(alc, 6),
                "from_budget_means": float(np.dot(W, br.mean(0))),
                "summary_rows_mean": float(np.mean([p["alc"] for p in pairs])),
                "item_weighted_alternative": item_w},
        "per_pair_alc_max_abs_diff": float(np.max(np.abs(per_pair - [p["alc"] for p in pairs]))),
        "per_pair_ece_alc_max_abs_diff": float(np.max(np.abs(ec @ W - [p["ece"] for p in pairs]))),
        "budgets": {"budgets": list(BUDGETS),
                    "brier_mean": [float(x) for x in br.mean(0)],
                    "ece_mean": [float(x) for x in ec.mean(0)],
                    "brier_min": [float(x) for x in br.min(0)],
                    "brier_max": [float(x) for x in br.max(0)]},
        "ece": {"run_mean_of_pair_ece_alc": float(np.mean([p["ece"] for p in pairs])),
                "max_pair_ece_b0": float(ec[:, 0].max())},
        "b0_minus_b31_mean": float((br[:, 0] - br[:, 5]).mean()),
    }


#: the fields summarise() shares with formative_feedback.summarise_run
SHARED_FIELDS = ("counts", "per_pair_alc_max_abs_diff", "per_pair_ece_alc_max_abs_diff", "budgets",
                 "ece", "b0_minus_b31_mean")


def letters(run1):
    """docs/findings.md's letters for the anonymous benchmark ids: A, B, ... in
    order of first appearance in run 1. For recording only."""
    out = {}
    for p in run1:
        out.setdefault(p["benchmark"], chr(ord("A") + len(out)))
    return out


def pair_rows(pairs, letter, prefix="s"):
    rows = []
    for j, p in enumerate(pairs):
        b = p["brier"]
        rows.append({"label": f"{prefix}{j + 1}", "subject": p["subject"], "benchmark": p["benchmark"],
                     "letter": letter.get(p["benchmark"]), "n": p["n"],
                     "brier": p["brier"], "ece_b": p["ece_b"],
                     "alc_summary_row": p["alc"], "alc_recomputed": float(np.dot(W, b)),
                     "ece_alc_summary_row": p["ece"], "ece_alc_recomputed": float(np.dot(W, p["ece_b"])),
                     "b1_or_b3_above_b0": bool(b[1] > b[0] or b[2] > b[0]),
                     "b31_at_or_above_0.25": bool(b[5] >= 0.25)})
    return rows


def low_budget_profile(pairs):
    """Descriptive: the run-mean steps B0->B1 and B1->B3 (the repository's
    recurring bug signature is Brier spiking at B1 and B3), and the pairs whose
    B1 or B3 exceed their B0, with their B31 (0.25 is the Brier of a constant
    0.5, where any move away from 0.5 costs)."""
    br = np.array([p["brier"] for p in pairs])
    m = br.mean(0)
    up = [{"subject": p["subject"], "benchmark": p["benchmark"], "b0": p["brier"][0],
           "b1": p["brier"][1], "b3": p["brier"][2], "b31": p["brier"][5]}
          for p in pairs if p["brier"][1] > p["brier"][0] or p["brier"][2] > p["brier"][0]]
    return {"mean_b1_minus_b0": float(m[1] - m[0]), "mean_b3_minus_b1": float(m[2] - m[1]),
            "pairs_with_b1_or_b3_above_b0": up}


def regression_check(pairs):
    vals = np.array([p["brier"] + p["ece_b"] + [p["alc"], p["ece"]] for p in pairs], float)
    return {
        "pairs": len(pairs),
        "every_pair_in_every_section": True,   # parse() refuses a table whose sections disagree
        "all_values_finite_in_unit_interval": bool(np.all(np.isfinite(vals)) and np.all(vals >= 0)
                                                   and np.all(vals <= 1)),
        "team_report": RUN3["team_report"],
        "latency": "the table records no timing; that the run finished and every pair was scored at "
                   "every budget is the only latency evidence here",
        "low_budget_profile": low_budget_profile(pairs),
    }


# --- overlap ------------------------------------------------------------------------------

def overlap_two(a_name, a, b_name, b):
    """formative_feedback.overlap for any two named runs (its keys say run1/run2)."""
    sa, sb = {p["subject"] for p in a}, {p["subject"] for p in b}
    ba, bb = {p["benchmark"] for p in a}, {p["benchmark"] for p in b}
    qa = {(p["subject"], p["benchmark"]) for p in a}
    qb = {(p["subject"], p["benchmark"]) for p in b}
    shared = sorted(qa & qb)
    out = {"subjects": {a_name: len(sa), b_name: len(sb), "shared": sorted(sa & sb)},
           "benchmarks": {a_name: len(ba), b_name: len(bb), "shared": len(ba & bb)},
           "pairs": {"shared": [list(x) for x in shared]},
           "paired_comparison": None}
    if shared:
        rows = []
        for s, bm in shared:
            x = next(p for p in a if (p["subject"], p["benchmark"]) == (s, bm))
            y = next(p for p in b if (p["subject"], p["benchmark"]) == (s, bm))
            rows.append({"subject": s, "benchmark": bm, "n": [x["n"], y["n"]],
                         f"{a_name}_brier": x["brier"], f"{b_name}_brier": y["brier"],
                         f"{b_name}_minus_{a_name}_alc": float(np.dot(W, y["brier"]) - np.dot(W, x["brier"])),
                         f"{b_name}_minus_{a_name}_brier": [float(v - u) for u, v in zip(x["brier"],
                                                                                        y["brier"])]})
        out["paired_comparison"] = rows
    return out


def overlap_all(runs):
    names = list(runs)
    subj = {k: {p["subject"] for p in v} for k, v in runs.items()}
    bench = {k: {p["benchmark"] for p in v} for k, v in runs.items()}
    in_runs = defaultdict(list)
    for k in names:
        for s in sorted(subj[k]):
            in_runs[s].append(k)
    n_by = defaultdict(lambda: {k: [] for k in names})
    for k, v in runs.items():
        for p in v:
            n_by[p["benchmark"]][k].append(p["n"])
    return {"pair_appearances": int(sum(len(v) for v in runs.values())),
            "distinct_subjects": len(in_runs),
            "subjects_in_more_than_one_run": {s: r for s, r in sorted(in_runs.items()) if len(r) > 1},
            "benchmark_ids_in_every_run": len(set.intersection(*bench.values())),
            "distinct_benchmark_ids": len(set.union(*bench.values())),
            "n_by_benchmark": {b: {k: sorted(v) for k, v in d.items()} for b, d in sorted(n_by.items())}}


def benchmark_balanced(pairs, keep):
    by = defaultdict(list)
    for p in pairs:
        if p["benchmark"] in keep:
            by[p["benchmark"]].append(float(np.dot(W, p["brier"])))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


# --- archive ------------------------------------------------------------------------------

def level_of(model_py):
    """LEVEL from a model.py's source, without importing it."""
    tree = ast.parse(model_py)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL" for t in node.targets):
            return {k: float(v) for k, v in ast.literal_eval(node.value).items()}
    return None


def archive_record(path, given, how=""):
    """sha256 of the archive and its members against 4d2cc4f, and how it was obtained."""
    base = {"path": rel(path), "commit": RUN3["commit"], "expected_sha256": RUN3["archive_sha256"]}
    if given:
        base["provenance"] = ("a copy given with --archive, recorded as downloaded back from the platform: "
                              + (how or "(no description given)"))
    else:
        base["provenance"] = ("the current submission archive in the working tree (dist/), built at "
                              "4d2cc4f and handed to the team for upload. It was not downloaded back from "
                              "the platform, so that the uploaded file had these bytes is not verified from "
                              "a downloaded copy (runs 1 and 2's archives were; "
                              "results/formative_feedback.json 'archives'). dist/ is gitignored and rebuilt "
                              "by tools/build_submission.py, so re-verifying needs it to hold the 4a882cc7... "
                              "file; this read's archive checks are required.")
    base["downloaded_back"] = bool(given)
    if not os.path.exists(path):
        return {**base, "available": False}
    v = FF.verify_archive(path, RUN3["commit"])
    v.pop("path", None)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        model = z.read("model.py").decode() if "model.py" in names else None
    st = os.stat(path)
    return {**base, "available": True, **v,
            "equals_selected_archive": v["sha256"] == RUN3["archive_sha256"],
            "size_bytes": st.st_size,
            "file_mtime_utc": datetime.datetime.fromtimestamp(
                st.st_mtime, datetime.timezone.utc).isoformat(timespec="seconds"),
            "level_in_model_py": level_of(model) if model is not None else None}


# --- placement ----------------------------------------------------------------------------

def zscores(sc, budgets, alc):
    """A run's distance from each regime's run mean in that regime's single-run
    sds (ship_confirm.json: run2_placement.regimes for the means, single_run_sd
    for the sds). Positive is worse than the regime's mean."""
    out = {}
    for regime, g in sc["run2_placement"]["regimes"].items():
        sd = sc["single_run_sd"][regime]
        sd_b = np.array([sd[f"B{b}"] for b in BUDGETS], float)
        m_b = np.array(g["mean_budgets"], float)
        out[regime] = {"z_ALC": float((alc - g["mean_ALC"]) / g["sd_ALC"]),
                       "z_budgets": [float(x) for x in (np.asarray(budgets, float) - m_b) / sd_b]}
    return out


def placement(sc, sums, checks):
    regs = sc["run2_placement"]["regimes"]
    run2_stored = sc["run2_placement"]["run2"]
    # reading check: ship_confirm's own run-2 z values from its own inputs
    z2s = zscores(sc, run2_stored["budgets"], run2_stored["ALC"])
    for regime, g in regs.items():
        checks.add(f"ship_confirm {regime}: run 2's stored z_ALC recomputed from the stored means and sds",
                   abs(z2s[regime]["z_ALC"] - g["z_ALC"]) < 1e-4, r6(z2s[regime]["z_ALC"]), g["z_ALC"])
        checks.add(f"ship_confirm {regime}: run 2's stored budget z recomputed",
                   max(abs(a - b) for a, b in zip(z2s[regime]["z_budgets"], g["z_budgets"])) < 1e-4,
                   [r6(x) for x in z2s[regime]["z_budgets"]], g["z_budgets"])
        checks.add(f"ship_confirm {regime}: sd_ALC equals single_run_sd ALC",
                   g["sd_ALC"] == sc["single_run_sd"][regime]["ALC"], g["sd_ALC"],
                   sc["single_run_sd"][regime]["ALC"])
    obs = {k: (sums[k]["budgets"]["brier_mean"], sums[k]["alc"]["recomputed"]) for k in ("run2", "run3")}
    z = {k: zscores(sc, *v) for k, v in obs.items()}
    regimes = {}
    for regime, g in regs.items():
        sd = sc["single_run_sd"][regime]
        d = obs["run3"][1] - obs["run2"][1]
        sd_d = math.sqrt(2) * g["sd_ALC"]
        regimes[regime] = {
            "group": GROUPS.get(regime, "other"),
            "label": sc["regimes"][regime]["label"],
            "runs": g["runs"],
            "mean_ALC": g["mean_ALC"], "sd_ALC": g["sd_ALC"],
            "mean_budgets": g["mean_budgets"],
            "sd_budgets": [sd[f"B{b}"] for b in BUDGETS],
            "run3": {"z_ALC": z["run3"][regime]["z_ALC"], "z_budgets": z["run3"][regime]["z_budgets"]},
            "run2_exact_budgets": {"z_ALC": z["run2"][regime]["z_ALC"],
                                   "z_budgets": z["run2"][regime]["z_budgets"]},
            "run2_as_stored_in_ship_confirm": {"z_ALC": g["z_ALC"], "z_budgets": g["z_budgets"],
                                               "share_runs_at_or_above": g["share_runs_at_or_above"]},
            "run3_minus_run2": {"alc": d, "sd_of_difference_of_two_independent_runs": sd_d,
                                "z": d / sd_d},
        }
    gap = sc.get("code_gap", {})
    return {
        "source": "results/ship_confirm.json (run2_placement.regimes: run means; single_run_sd: sds)",
        "what_the_rows_are": "the shipped hier as formative run 2 ran it: paiec/hier.py 70a3a81a, the old "
                             "multiple-choice floor, the solver before the floored-fit fix "
                             "(ship_confirm.json code_gap.rows_code). Run 3's archive adds the corrected "
                             "floor and the floored-fit fix.",
        "code_gap_not_added": {
            "mcq_floor_fix_minus_old_alc": {r: v.get("est") for r, v in
                                            gap.get("mcq_floor_fix_minus_old", {}).items()},
            "floored_fit_fix_alc": {r: v.get("alc_diff") for r, v in gap.get("floored_fit_fix", {}).items()},
            "note": "measured on runs 0-199 (mcq_floor.json, hier_floor.json) and not added to the "
                    "regimes' means"},
        "sign": "z = (run - regime's run mean) / regime's single-run sd; positive is worse (higher Brier)",
        "run2_exact_budgets_note": "run 2 recomputed from its table at 6 dp; ship_confirm.json placed it "
                                   "with budgets rounded to three decimals (kept as stored)",
        "not_computed": "the share of a regime's runs at or above run 3 needs each run's ALC, which "
                        "ship_confirm.json does not store (they are in data/subject_side_rows, gitignored); "
                        "only z-scores are given",
        "regimes": regimes,
        "note": "each run is one draw of 8 or 9 pairs; a z is its distance from a regime's run mean in that "
                "regime's single-run sds, not a test",
    }


# --- the three runs -----------------------------------------------------------------------

def three_runs(runs, sums, place):
    shared = set.intersection(*({p["benchmark"] for p in v} for v in runs.values()))
    rows = []
    for k in ("run1", "run2", "run3"):
        s, meta = sums[k], RUNS[k]
        lb = low_budget_profile(runs[k])
        z = None
        if k in ("run2", "run3"):
            key = "run3" if k == "run3" else "run2_exact_budgets"
            z = {r: g[key]["z_ALC"] for r, g in place["regimes"].items()}
        rows.append({
            "run": int(k[-1]), "scored": meta["scored"], "commit": meta["commit"], "model": meta["model"],
            "pairs": s["counts"]["pairs"], "benchmarks": s["counts"]["benchmarks"],
            "subjects": s["counts"]["subjects"], "pairs_per_benchmark": s["counts"]["pairs_per_benchmark"],
            "subject_item_pairs": s["counts"]["subject_item_pairs"],
            "n_range": [s["counts"]["n_min"], s["counts"]["n_max"]],
            "platform_alc": meta["platform_alc"],
            "alc_recomputed": s["alc"]["recomputed"], "alc_recomputed_6dp": s["alc"]["recomputed_6dp"],
            "brier_mean": s["budgets"]["brier_mean"], "ece_mean": s["budgets"]["ece_mean"],
            "mean_ece_alc": s["ece"]["run_mean_of_pair_ece_alc"],
            "max_pair_ece_b0": s["ece"]["max_pair_ece_b0"],
            "b0_minus_b31_mean": s["b0_minus_b31_mean"],
            "benchmark_balanced_alc": benchmark_balanced(runs[k], shared),
            "z_ALC_vs_shipped_model_regimes": z,
            "low_budget": {"mean_b1_minus_b0": lb["mean_b1_minus_b0"],
                           "mean_b3_minus_b1": lb["mean_b3_minus_b1"],
                           "pairs_with_b1_or_b3_above_b0": len(lb["pairs_with_b1_or_b3_above_b0"])},
        })
    return {"table": rows,
            "notes": ["run 1 is the legacy Predictor: the shipped model's single-run sds do not describe it, "
                      "so it is not placed",
                      "benchmark_balanced_alc: each run's mean over the benchmark ids all three runs share of "
                      "its per-benchmark mean pair ALC; subjects differ, so it is descriptive only",
                      "run 3's platform score was not pasted; its ALC is the table's pair mean"]}


def can_and_cannot(sums, ov, ov_all, place):
    sd = [g["sd_ALC"] for g in place["regimes"].values()]
    sd_d = [math.sqrt(2) * x for x in sd]
    zs = [abs(g[k]["z_ALC"]) for g in place["regimes"].values() for k in ("run3", "run2_exact_budgets")]
    rec = ov["run1_run3"]["paired_comparison"] or []
    cg = place["code_gap_not_added"]
    gaps = [abs((cg["mcq_floor_fix_minus_old_alc"].get(r) or 0.0) + (cg["floored_fit_fix_alc"].get(r) or 0.0))
            for r in cg["mcq_floor_fix_minus_old_alc"]]
    can = [
        f"the scoring rule once more: every pair's summary ALC is 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + "
        f"0.1 B31 of its budget rows (largest difference "
        f"{sums['run3']['per_pair_alc_max_abs_diff']:.1e}); the pair-mean rule itself was pinned on runs 1 "
        f"and 2 and is applied, not re-tested, for run 3, whose headline score was not pasted",
        "that the archive selected at 4d2cc4f runs end to end on the platform: every pair scored at all six "
        "budgets, no errors reported (the regression and latency check it was uploaded for)",
        f"that the formative evaluation drew on the same {ov_all['benchmark_ids_in_every_run']} benchmark "
        f"ids in all three runs, with {ov_all['distinct_subjects']} distinct subjects over "
        f"{ov_all['pair_appearances']} pair appearances",
        f"that runs 2 and 3 (the shipped model's family) each sit within {max(zs):.2f} single-run sds of "
        f"every regime's mean in ship_confirm.json: one draw each, consistent with all of them",
    ]
    cannot = [
        "compare predictors or configurations: the subjects differ from run to run"
        + (f"; the one recurring (subject, benchmark) pair ({rec[0]['subject']}, {rec[0]['benchmark']}, "
           f"n {rec[0]['n'][0]} both times) ran under two different models with different co-sampled "
           f"pairs (the shared labeled list differs), so it is not a paired comparison of models"
           if rec else ""),
        f"tell regimes apart or detect a change of the size measured offline: the difference of two "
        f"independent runs has an sd of {min(sd_d):.3f} to {max(sd_d):.3f} (sqrt 2 times single-run sds "
        f"{min(sd):.3f} to {max(sd):.3f}), against a measured code gap from run 2's archive to run 3's of "
        f"{max(gaps) if gaps else float('nan'):.5f} in ALC at most (the corrected floor and the floored-fit "
        f"fix together, on runs 0-199)",
        "measure latency: the table records no timing",
        "add to the level reading: none is done on run 3 (below)",
    ]
    return {"can": can, "cannot": cannot}


def purpose_record(checks):
    """Where run 3's purpose was written down before it was scored."""
    with open(FF_JSON) as f:
        ff = json.load(f)
    sub = next((s for s in ff.get("submissions", []) if s.get("run") == 3), None)
    log = git("log", "--format=%h %cI", "-S", PURPOSE_PHRASE, "--", "results/formative_feedback.json")
    first = log.splitlines()[-1].split() if log else None
    out = {"source": "results/formative_feedback.json, submissions[run 3] (read only)",
           "entry": sub,
           "first_commit_with_the_statement": ({"commit": first[0], "committed": first[1]} if first else None)}
    checks.add("formative_feedback.json lists run 3 as unscored, at 4d2cc4f, with the selected archive",
               sub is not None and sub.get("scored") is None and sub.get("commit") == RUN3["commit"]
               and sub.get("archive_sha256") == RUN3["archive_sha256"],
               None if sub is None else {k: sub.get(k) for k in ("scored", "commit", "archive_sha256")},
               {"scored": None, "commit": RUN3["commit"], "archive_sha256": RUN3["archive_sha256"]})
    if first:
        before = first[1][:10] < RUN3["scored"]
        out["stated_before_scoring"] = before
        checks.add("the purpose was committed before run 3 was scored", before, first[1], RUN3["scored"])
    else:
        out["stated_before_scoring"] = None
        checks.add("the commit stating the purpose was found in git", False, None, PURPOSE_PHRASE,
                   required=False)
    pre = ff.get("preregistration", {})
    dec = ff.get("reading", {}).get("decision", {})
    return out, {"sha256": pre.get("sha256"), "written_utc": pre.get("written_utc"),
                 "outcome": dec.get("outcome")}


# --- provenance ---------------------------------------------------------------------------

def provenance(argv, t0, archive_path):
    files = {"script": os.path.abspath(__file__),
             "helper": os.path.join(ROOT, "experiments", "formative_feedback.py"),
             "evaluator": os.path.join(ROOT, "paiec", "evaluator.py")}
    inputs = [FEEDBACK["run1"], FEEDBACK["run2"], FEEDBACK["run3"], SHIP_JSON, FF_JSON,
              os.path.join(ROOT, "submission", "prior.json")]
    tracked = ["experiments/formative_run3.py", "tests/test_formative_run3.py",
               "results/formative/run3.txt", "experiments/formative_feedback.py",
               "results/formative_feedback.json", "results/ship_confirm.json"]
    status = git("status", "--porcelain", "--", *tracked)
    return {"command": " ".join(["python", "experiments/formative_run3.py",
                                 *(sys.argv[1:] if argv is None else argv)]),
            "when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "head": git("rev-parse", "HEAD"),
            "git_status": status.splitlines() if status else [],
            "python": sys.version.split()[0], "numpy": np.__version__,
            "script": rel(files["script"]), "script_sha256": sha256_file(files["script"]),
            "code_sha256": {rel(p): sha256_file(p) for p in files.values()},
            "inputs_sha256": {**{rel(p): sha256_file(p) for p in inputs},
                              **({rel(archive_path): sha256_file(archive_path)}
                                 if os.path.exists(archive_path) else {})},
            "wall_s": round(time.time() - t0, 2)}


# --- main ---------------------------------------------------------------------------------

def build(args, argv=None):
    t0 = time.time()
    checks = Checks()
    runs = {}
    for k, path in FEEDBACK.items():
        with open(path) as f:
            runs[k] = FF.parse(f.read())
    sums = {k: summarise(v) for k, v in runs.items()}

    # runs 1 and 2: the record stage's own numbers, and the scores the platform reported
    for k in ("run1", "run2"):
        ref = FF.summarise_run(k, runs[k])
        same = all(ref[f] == sums[k][f] for f in SHARED_FIELDS) and all(
            ref["alc"][f] == sums[k]["alc"][f] for f in sums[k]["alc"])
        checks.add(f"{k}: summarise() equals formative_feedback.summarise_run on every shared field",
                   same, same, True)
        checks.add(f"{k}: recomputed ALC reproduces the platform's", ref["alc"]["reproduces_reported"],
                   sums[k]["alc"]["recomputed_6dp"], RUNS[k]["platform_alc"])
    # run 3: every pair against its summary row, and the pair mean
    s3 = sums["run3"]
    checks.add("run3: per-pair ALC from the budgets equals the summary rows",
               s3["per_pair_alc_max_abs_diff"] <= PAIR_TOL, s3["per_pair_alc_max_abs_diff"], PAIR_TOL)
    checks.add("run3: per-pair ECE-ALC from the budgets equals the summary rows",
               s3["per_pair_ece_alc_max_abs_diff"] <= PAIR_TOL, s3["per_pair_ece_alc_max_abs_diff"], PAIR_TOL)
    checks.add("run3: pair-mean ALC equals the weighted pair-averaged budget means",
               abs(s3["alc"]["recomputed"] - s3["alc"]["from_budget_means"]) < 1e-12,
               s3["alc"]["from_budget_means"], s3["alc"]["recomputed"])
    checks.add("run3: the summary rows' mean is within 1e-6 of the recomputed ALC",
               abs(s3["alc"]["summary_rows_mean"] - s3["alc"]["recomputed"]) <= PAIR_TOL,
               s3["alc"]["summary_rows_mean"], s3["alc"]["recomputed"])
    reg = regression_check(runs["run3"])
    checks.add("run3: every value finite and in [0, 1]", reg["all_values_finite_in_unit_interval"],
               reg["all_values_finite_in_unit_interval"], True)

    # overlap
    ov = {"run1_run3": overlap_two("run1", runs["run1"], "run3", runs["run3"]),
          "run2_run3": overlap_two("run2", runs["run2"], "run3", runs["run3"]),
          "run1_run2": overlap_two("run1", runs["run1"], "run2", runs["run2"])}
    for a, b in (("run1", "run3"), ("run2", "run3"), ("run1", "run2")):
        ref = FF.overlap(runs[a], runs[b])
        mine = ov[f"{a}_{b}"]
        agree = (ref["subjects"]["shared"] == mine["subjects"]["shared"]
                 and ref["pairs"]["shared"] == mine["pairs"]["shared"]
                 and ref["benchmarks"]["shared"] == mine["benchmarks"]["shared"])
        checks.add(f"overlap {a}/{b} agrees with formative_feedback.overlap", agree, agree, True)
    ov_all = overlap_all(runs)

    # archive. The default read is of dist/, which is gitignored and rebuilt: the record then says
    # the file there is the selected archive, so its checks are required (re-verifying needs dist/
    # at 4a882cc7...). A copy given with --archive is recorded whatever its bytes; main() keeps a
    # stored record whose archive matched from being replaced by one whose archive does not.
    given = bool(args.archive)
    path = os.path.abspath(args.archive) if given else ARCHIVE
    arch = archive_record(path, given, args.archive_how)
    need = not given
    if arch["available"]:
        checks.add("archive: sha256 equals the selected archive's (4a882cc7...)",
                   arch["equals_selected_archive"], arch["sha256"], RUN3["archive_sha256"], required=need)
        checks.add("archive: every tracked member byte-identical to 4d2cc4f",
                   arch["all_tracked_members_match"], arch["all_tracked_members_match"], True, required=need)
        checks.add("archive: model.py's LEVEL is mu0 -2.5, sigma_mu 2.5, attr_scale 0.5",
                   arch["level_in_model_py"] == RUN3["level"], arch["level_in_model_py"], RUN3["level"],
                   required=need)
    else:
        checks.add("archive: present", False, rel(path), "a file", required=need)

    # placement and the three runs
    with open(SHIP_JSON) as f:
        sc = json.load(f)
    place = placement(sc, sums, checks)
    side = three_runs(runs, sums, place)
    purpose, prereg = purpose_record(checks)
    letter = letters(runs["run1"])

    out = {
        "about": "Formative run 3 (scored 2026-10-01): the archive built at 4d2cc4f, uploaded by the team as "
                 "a regression and latency check only. The record of its feedback table, set beside runs 1 "
                 "and 2; nothing here selects, tunes or reads a level. experiments/formative_run3.py.",
        "run3": {
            **RUN3,
            "purpose_stated_before_scoring": purpose,
            "feedback": "results/formative/run3.txt (the organisers' table, verbatim)",
            "benchmark_letters": {"map": letter,
                                  "note": "docs/findings.md's letters, A, B, ... in order of first "
                                          "appearance in run 1; for recording only, no model input is "
                                          "keyed on them"},
            "pairs": pair_rows(runs["run3"], letter),
            **s3,
            "alc_how": "per pair 0.1*B0 + 0.2*(B1+B3+B7+B15) + 0.1*B31 from the budget tables, then the "
                       "unweighted mean over pairs (the rule runs 1 and 2 pinned); equal to the weighted "
                       "sum of the pair-averaged budget means",
            "regression_check": reg,
        },
        "archive": arch,
        "overlap": {**ov, "all_three": ov_all,
                    "note": (f"run 3 shares {ov['run1_run3']['benchmarks']['shared']} of its "
                             f"{ov['run1_run3']['benchmarks']['run3']} benchmark ids with run 1 and "
                             f"{ov['run2_run3']['benchmarks']['shared']} with run 2; it shares "
                             f"{len(ov['run1_run3']['subjects']['shared'])} subject and "
                             f"{len(ov['run1_run3']['pairs']['shared'])} (subject, benchmark) pair with "
                             f"run 1 and {len(ov['run2_run3']['subjects']['shared'])} subjects with run 2. "
                             "A recurring pair is listed under paired_comparison, descriptive only: it ran "
                             "under different models with different co-sampled pairs")},
        "placement": place,
        "three_runs": side,
        "what_three_runs_can_and_cannot_measure": can_and_cannot(sums, ov, ov_all, place),
        "no_reading_or_tuning": {
            "statement": "No level reading, matching or tuning is done on run 3, and nothing in the model, "
                         "its hyperparameters or the archive changes because of it. The preregistered "
                         "decision rule of the pooled reading covered runs 1 and 2 only (17 pairs); run 3 "
                         "is not added to it, and it is not re-run with run 3.",
            "pooled_reading_preregistration": prereg,
        },
        "checks": checks.items,
    }
    out["provenance"] = provenance(argv, t0, path)
    return out, checks


def archive_verified(arch):
    """The archive record holds the selected archive's bytes."""
    return bool(isinstance(arch, dict) and arch.get("available") and arch.get("equals_selected_archive"))


def stored_archive_verified(path):
    """Whether the record already at `path` verified the selected archive (False if none is readable)."""
    try:
        with open(path) as f:
            return archive_verified(json.load(f).get("archive"))
    except (OSError, ValueError, AttributeError):
        return False


def save(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", help="a copy of run 3's archive downloaded back from the platform "
                                      "(default: the current submission archive under dist/)")
    ap.add_argument("--archive-how", dest="archive_how", default="",
                    help="how the --archive copy was obtained, recorded verbatim")
    ap.add_argument("--replace-archive-record", dest="replace_archive_record", action="store_true",
                    help="allow replacing a stored record whose archive was the selected archive with one "
                         "whose archive is not (or is missing)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    out, checks = build(args, argv)
    bad = checks.failed()
    for c in checks.failed(required_only=False):
        print(f"{'FAILED' if c['required'] else 'warning'}: {c['check']}: got {c['got']}, want {c['want']}")
    if bad:
        sys.exit(f"{len(bad)} required check(s) failed; nothing written")
    if (not archive_verified(out["archive"]) and stored_archive_verified(args.out)
            and not args.replace_archive_record):
        sys.exit(f"{args.out} verified the selected archive (4a882cc7...) and this run's archive is not it; "
                 "nothing written (pass --replace-archive-record to replace it)")
    save(out, args.out)
    r3 = out["run3"]
    print(f"run 3: {r3['counts']['pairs']} pairs, {r3['counts']['benchmarks']} benchmarks, "
          f"{r3['counts']['subjects']} subjects; ALC {r3['alc']['recomputed']:.10f} "
          f"(per-pair max diff {r3['per_pair_alc_max_abs_diff']:.1e})")
    print("budgets " + " ".join(f"{x:.4f}" for x in r3["budgets"]["brier_mean"]))
    for regime, g in out["placement"]["regimes"].items():
        print(f"  {regime:13s} mean {g['mean_ALC']:.4f} sd {g['sd_ALC']:.4f}  run3 z {g['run3']['z_ALC']:+.2f}  "
              "by budget " + " ".join(f"{x:+.2f}" for x in g["run3"]["z_budgets"]))
    print(f"archive: {out['archive'].get('sha256')} (equals selected: "
          f"{out['archive'].get('equals_selected_archive')})")
    print(f"{len(checks.items)} checks, {len(checks.failed(False))} failed; "
          f"wrote {os.path.relpath(args.out, ROOT) if args.out.startswith(ROOT) else args.out}")


if __name__ == "__main__":
    main()
