"""Formative runs 1 and 2: the record, and the pooled per-pair level reading.

Two formative submissions have been scored on the platform (review_v0 P0.3,
Q1, Q2, Q8, Q9). Their feedback tables are in results/formative/run1.txt and
run2.txt, copied verbatim from the organisers' scoring output. This script

record    parses both tables and recomputes each run's ALC from its budgets
          with the official weights, pair-averaged (docs/protocol.md,
          "Scoring"); counts pairs, benchmarks and subjects; gives the
          per-budget Brier and ECE means; hashes the archive behind each run
          and checks its members against the commit it was built from
          (--archive1 / --archive2; without them the archives recorded by an
          earlier pass are kept); and gives the ID overlap between the runs
          (subjects, benchmarks, (subject, benchmark) pairs).
prereg    writes the preregistration of the reading (PREREGISTRATION below)
          into the results file before any reading is computed. It refuses to
          run once a reading exists, and `read` refuses to run unless the
          stored preregistration is byte-identical to the one here.
read      the pooled per-pair reading of both runs that "What actually
          shipped, after the audit" (docs/findings.md) asks for, with the
          audit's own estimator: the per-pair matched estimate. It was a
          scratch script (step2b audit, feedback_match*.py); this is a port,
          checked against the values that audit recorded (AUDIT_RECORDED).

The per-pair matched estimate. Each feedback pair's Brier profile over the six
budgets is matched to the replica pair appearances whose profile, for the same
model, is closest: Euclidean distance with each budget standardised by its sd
over the pool, the K nearest. The neighbours' evaluation rates say which root
of p(1-p) = B31 the pair is on, and the neighbours' paired difference
(configuration minus the model that produced the feedback) estimates what the
configuration would have scored on that pair. A run's estimate is its feedback
ALC plus the mean of those differences over its pairs.

Pools. A: results/level_calibration.json, the rows the audit used (the legacy
Predictor's per-pair Brier by budget in every regime, every configuration's
per-pair ALC). B: data/subject_side_rows (gitignored; experiments/
subject_side.py), the shipped hier's per-pair Brier by budget (LEVEL mu0 -2.5,
sigma_mu 2.5, attr_scale 0.5, the hier.py of commit ee5085a and the old
multiple-choice floor: the model formative run 2 ran), paired with the
Predictor's rows for the same runs from results/testlike_check.json (test-like),
results/hier_eval.json (public R1) and pool A (the two seed-3 regimes), every
pairing checked on the run's pair list. Run 1 (the Predictor) is matched on
Predictor profiles, run 2 (the shipped hier) on shipped-hier profiles.

One estimator here is new and says so: `reweighted`, the pool's per-pair
Brier reweighted in bins of pair logit to a given level distribution. It is a
cheap proxy for scoring a regime at that level (review W3), not a substitute.

Run:  python experiments/formative_feedback.py --stage record \
          [--archive1 PATH [--rebuild1 PATH] --run1-how TEXT] \
          [--archive2 PATH [--rebuild2 PATH] --run2-how TEXT]
      python experiments/formative_feedback.py --stage prereg
      python experiments/formative_feedback.py --stage read
Everything goes to results/formative_feedback.json; `passes` records each
stage's command, time, commit and the digests of what it read.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402

OUT = os.path.join(ROOT, "results", "formative_feedback.json")
FEEDBACK = {"run1": os.path.join(ROOT, "results", "formative", "run1.txt"),
            "run2": os.path.join(ROOT, "results", "formative", "run2.txt")}
LC_JSON = os.path.join(ROOT, "results", "level_calibration.json")
TC_JSON = os.path.join(ROOT, "results", "testlike_check.json")
HE_JSON = os.path.join(ROOT, "results", "hier_eval.json")
KS = (15, 40)
BOOTS = 2000
W = np.asarray(WEIGHTS, float)

#: What each formative submission was. The leaderboard score is the one
#: reported with each paste; `record` checks the tables reproduce it.
RUNS = {
    "run1": {"scored": "2026-09-25", "commit": "b68492c",
             "model": "legacy Predictor (paiec.predict), the first submission",
             "platform_alc": 0.211300},
    "run2": {"scored": "2026-09-26", "commit": "ee5085a",
             "model": "paiec.hier.HierPredictor, LEVEL mu0 -2.5 sigma_mu 2.5 attr_scale 0.5, "
                      "old multiple-choice floor, before the floored-fit fix",
             "platform_alc": 0.192623},
}
#: Q9: every formative submission so far or planned, and what it informed.
SUBMISSIONS = [
    {"run": 1, "scored": "2026-09-25", "commit": "b68492c", "model": RUNS["run1"]["model"],
     "alc": 0.2113, "informed": [
         "the test-like regime's defaults (paiec/testlike.py: level_mean -1.6 from the lower "
         "roots of the pairs' B31, the 1.25-year date shift reproducing the B0 optimism)",
         "the level calibration (experiments/level_calibration.py): mu0, sigma_mu and attr_scale "
         "chosen on runs tuned to this feedback, under a public guard",
         "the audit's per-pair reading (p6 and p8 read as high-rate) that moved the shipped "
         "LEVEL from mu0 -3.0 / attr_scale 0.25 to mu0 -2.5 / attr_scale 0.5",
         "findings 'Against the first real formative feedback': support for the widened level prior"]},
    {"run": 2, "scored": "2026-09-26", "commit": "ee5085a", "model": RUNS["run2"]["model"],
     "alc": 0.192623, "informed": [
         "the motivation of the subject-side study at B0 and B1 (experiments/subject_side.py; "
         "verdict: ship none, so nothing changed)",
         "this pooled per-pair reading, which may propose one candidate for the level "
         "distribution (mu0, sigma_mu) under the preregistration below; nothing ships from it "
         "alone"]},
    {"run": 3, "scored": None, "commit": "4d2cc4f",
     "archive_sha256": "4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090",
     "model": "the rebuilt archive: hier with the same LEVEL, the corrected multiple-choice floor "
              "(f7e7d87) and the floored-fit fix (4d2cc4f)",
     "alc": None, "informed": [
         "planned as a regression and latency check only: that the rebuilt archive runs on the "
         "platform without failures or timeouts; its score is not to select or tune anything"]},
]

#: The audit's matched estimates as its scratch outputs recorded them
#: (step2b/audit feedback_match_out.json and feedback_match2_out.json), for the
#: port check. 'full' is every regime of pool A, 'screen' the appearances where
#: every screened configuration was scored (test-like runs 0-99, public
#: benchmark-first runs 0-39).
AUDIT_RECORDED = {
    ("neighbour", "full", 15): 0.1788585185185185,
    ("neighbour", "full", 40): 0.17832227777777776,
    ("shipped", "screen", 15): 0.18117133333333332,
    ("shipped", "screen", 40): 0.18299952777777775,
    ("neighbour", "screen", 15): 0.18242459259259258,
    ("neighbour", "screen", 40): 0.18386077777777776,
}

# The preregistration is fixed text: `prereg` stores it, `read` refuses to run
# if the stored copy differs.
PREREGISTRATION = {
    "written_before": "any matching of run 2's pairs, the pooled 17-pair reading, or the shipped "
                      "config's matched estimate on run 1 was computed. Seen before writing it: "
                      "both feedback tables, and the audit's scratch outputs for run 1 (its "
                      "neighbour shares and its estimates for the mu0 -3.0 / attr_scale 0.25 "
                      "config and, on the screen pool, for the shipped config).",
    "may_change": "one global hyperparameter only: the level distribution of a new benchmark, "
                  "mu0 and sigma_mu of LEVEL in submission/model.py. Not attr_scale, not the item "
                  "or subject side, and nothing keyed on an anonymous benchmark or subject id.",
    "estimator": "the audit's per-pair matched estimate, ported unchanged: Euclidean distance "
                 "on the six per-budget Brier values, each standardised by its sd over the pool, "
                 "K in {15, 40}; run 1 matched on the legacy Predictor's profiles (pool A, every "
                 "regime), run 2 on the shipped hier's profiles (pool B, every regime).",
    "level_reading": "a pair's rate is the root of p(1-p) = B31 that the majority of its K "
                     "neighbours' evaluation rates sit on (upper if more than half exceed 0.5); "
                     "its level is the continuity-corrected logit log((np+0.5)/(n-np+0.5)) at the "
                     "pair's n, the convention of the regimes' realised pair logits. Uncertainty: "
                     "2,000 draws, each resampling the 7 benchmark ids as clusters and taking each "
                     "pair's upper root with probability equal to its neighbour share.",
    "comparator": "the tuned test-like regime the shipped LEVEL was chosen in: realised pair "
                  "logit mean -1.290, sd 1.696 (results/testlike_check.json, "
                  "summary.check.default.regime.pair_logit).",
    "decision_rule": [
        "mu0: a candidate is proposed only if, for both K, the 17-pair mean differs from -1.290 "
        "by more than 2 bootstrap SEs, in the same direction. The candidate is mu0 -2.5 + "
        "(mean - (-1.290)) / f, f = (1 + pi T / 8)^-1/2 with T = sigma_d^2 + sigma_g^2 of Hyper() "
        "(about 0.46), the mean over both K, rounded to the nearest 0.5.",
        "sigma_mu: a candidate is proposed only if, for both K, the 17-pair sd differs from 1.696 "
        "by more than 2 bootstrap SEs, in the same direction. The candidate puts the whole "
        "difference in pair variance on the level: sigma_mu^2 + (sd^2 - 1.696^2) / f^2, floored "
        "at 0.5^2, rounded to the nearest of {1.3, 1.8, 2.5, 3.5, 5}.",
        "Reading bias guard: the root reading is also applied to the replica's own test-like pair "
        "appearances (each through its own B31, on its true side), for the Predictor (pool A) and "
        "the shipped hier (pool B), weighted 9:8 as the two runs' pairs. If that reading moves the "
        "mean, or the sd, by more than 2 bootstrap SEs of the 17-pair value, a candidate from that "
        "quantity is reported as confounded and not proposed.",
        "Otherwise the reading proposes nothing and LEVEL stays.",
        "A proposed candidate is not shipped from this reading. It must first be scored with "
        "experiments/level_calibration.py's harness on fresh test-like runs (an unused seed) with "
        "testlike.Regime's level_mean and level_sd set to the reading, and pass: ALC at least "
        "0.002 below the shipped LEVEL (the acceptance harness's test-like bar) with no parent "
        "above +0.002; at most +0.001 against the shipped LEVEL on both public weightings (the "
        "harness guard); at most +0.003 against the legacy Predictor on both (the level "
        "calibration's guard); and at most +0.002 against the shipped LEVEL on the default "
        "test-like regime."],
    "descriptive_only": "everything else the reading reports (the two-component view, the "
                        "between- and within-benchmark split, where run 2's B0 excess sits, the "
                        "predictions for run 2, the neighbour-mean logits) changes nothing.",
}


# --- parsing and the record ------------------------------------------------------------------

_SECTION = re.compile(r"^(ALC summary|Label budget: (\d+))\s*$", re.M)
_ROW = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.M)


def parse(text):
    """The organisers' feedback text -> list of pairs, each {'subject', 'benchmark',
    'n', 'alc', 'ece', 'brier': [6], 'ece_b': [6]} in the table's order. Every
    section must list the same pairs in the same order."""
    heads = list(_SECTION.finditer(text))
    sections = {}
    for h, nxt in zip(heads, heads[1:] + [None]):
        body = text[h.end(): nxt.start() if nxt else len(text)]
        key = "summary" if h.group(2) is None else int(h.group(2))
        if key in sections:
            raise ValueError(f"section {key} appears twice")
        sections[key] = [(s, b, int(n), float(x), float(e)) for s, b, n, x, e in _ROW.findall(body)]
    missing = [k for k in ("summary",) + BUDGETS if k not in sections]
    if missing:
        raise ValueError(f"missing sections: {missing}")
    ids = [r[:3] for r in sections["summary"]]
    if not ids:
        raise ValueError("no rows")
    for B in BUDGETS:
        if [r[:3] for r in sections[B]] != ids:
            raise ValueError(f"budget {B} lists different pairs")
    return [{"subject": s, "benchmark": b, "n": n, "alc": x, "ece": e,
             "brier": [sections[B][j][3] for B in BUDGETS],
             "ece_b": [sections[B][j][4] for B in BUDGETS]}
            for j, (s, b, n, x, e) in enumerate(sections["summary"])]


def run_alc(pairs):
    """The platform's run ALC: per pair 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31,
    averaged equally over pairs. By linearity it is also the weighted sum of the
    pair-averaged budget means."""
    return float(np.mean([np.dot(W, p["brier"]) for p in pairs]))


def summarise_run(name, pairs):
    br = np.array([p["brier"] for p in pairs])
    ec = np.array([p["ece_b"] for p in pairs])
    per_pair = br @ W
    alc = run_alc(pairs)
    item_w = float(np.dot([p["alc"] for p in pairs], [p["n"] for p in pairs]) /
                   sum(p["n"] for p in pairs))
    reported = RUNS[name]["platform_alc"]
    mult = sorted(Counter(p["benchmark"] for p in pairs).values(), reverse=True)
    return {
        **RUNS[name],
        "pairs": pairs,
        "counts": {"pairs": len(pairs), "benchmarks": len({p["benchmark"] for p in pairs}),
                   "subjects": len({p["subject"] for p in pairs}),
                   "pairs_per_benchmark": mult,
                   "subject_item_pairs": int(sum(p["n"] for p in pairs)),
                   "n_min": int(min(p["n"] for p in pairs)), "n_max": int(max(p["n"] for p in pairs))},
        "alc": {"recomputed": alc,
                "recomputed_6dp": round(alc, 6),
                "from_budget_means": float(np.dot(W, br.mean(0))),
                "summary_rows_mean": float(np.mean([p["alc"] for p in pairs])),
                "item_weighted_alternative": item_w,
                "reported": reported,
                "reproduces_reported": round(alc, 6) == reported,
                "how": "per pair 0.1*B0 + 0.2*(B1+B3+B7+B15) + 0.1*B31 from the budget tables, then "
                       "the unweighted mean over pairs; equal to the weighted sum of the pair-averaged "
                       "budget means. Weighting pairs by their subject-item count does not reproduce "
                       "the score."},
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


def overlap(r1, r2):
    s1, s2 = {p["subject"] for p in r1}, {p["subject"] for p in r2}
    b1, b2 = {p["benchmark"] for p in r1}, {p["benchmark"] for p in r2}
    q1 = {(p["subject"], p["benchmark"]) for p in r1}
    q2 = {(p["subject"], p["benchmark"]) for p in r2}
    shared = sorted(q1 & q2)

    def balanced(run):
        by = defaultdict(list)
        for p in run:
            if p["benchmark"] in b1 & b2:
                by[p["benchmark"]].append(np.dot(W, p["brier"]))
        return float(np.mean([np.mean(v) for v in by.values()])) if by else None

    out = {"subjects": {"run1": len(s1), "run2": len(s2), "shared": sorted(s1 & s2)},
           "benchmarks": {"run1": len(b1), "run2": len(b2), "shared": len(b1 & b2)},
           "pairs": {"shared": [list(x) for x in shared]},
           "paired_comparison": None,
           "benchmark_balanced_alc": {
               "run1": balanced(r1), "run2": balanced(r2),
               "note": "each run's mean over the shared benchmarks of its per-benchmark mean pair "
                       "ALC, so the two runs weigh the benchmarks alike. Subjects differ, so this "
                       "is not a paired comparison of the two predictors."}}
    if shared:
        rows = []
        for s, b in shared:
            a = next(p for p in r1 if (p["subject"], p["benchmark"]) == (s, b))
            c = next(p for p in r2 if (p["subject"], p["benchmark"]) == (s, b))
            rows.append({"subject": s, "benchmark": b, "n": [a["n"], c["n"]],
                         "run1_brier": a["brier"], "run2_brier": c["brier"],
                         "run2_minus_run1_alc": float(np.dot(W, c["brier"]) - np.dot(W, a["brier"]))})
        out["paired_comparison"] = rows
    else:
        out["paired_comparison_note"] = ("no (subject, benchmark) pair recurs, because no subject "
                                         "recurs; there is no paired platform comparison")
    return out


# --- archives -----------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_path(arc):
    """The tracked file an archive member is a verbatim copy of
    (tools/build_submission.py members()); None for prior.json, which is built."""
    if arc in ("model.py", "labeling.py"):
        return f"submission/{arc}"
    if arc == "requirements.txt":
        return "requirements.txt"
    if arc.startswith("paiec_rt/"):
        return "paiec/" + arc.split("/", 1)[1]
    return None


def git_show(commit, path):
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def verify_archive(path, commit):
    """sha256 of the archive, and each member against `git show commit:<source>`."""
    out = {"path": path, "sha256": sha256_file(path), "commit": commit, "members": {}}
    with zipfile.ZipFile(path) as z:
        for arc in sorted(z.namelist()):
            data = z.read(arc)
            src = git_path(arc)
            row = {"sha256": hashlib.sha256(data).hexdigest(), "source": src}
            if src is not None:
                ref = git_show(commit, src)
                row["matches_commit"] = ref == data
            out["members"][arc] = row
    tracked = [r for r in out["members"].values() if r["source"] is not None]
    out["all_tracked_members_match"] = bool(tracked) and all(r["matches_commit"] for r in tracked)
    prior = out["members"].get("prior.json", {}).get("sha256")
    current = os.path.join(ROOT, "submission", "prior.json")
    if prior and os.path.exists(current):
        out["prior_json_equals_submission_prior_json"] = prior == sha256_file(current)
    return out


# --- provenance ---------------------------------------------------------------------------

def digest(path, n=16):
    return sha256_file(path)[:n] if os.path.exists(path) else None


def provenance(extra=()):
    files = [os.path.relpath(__file__, ROOT), "results/formative/run1.txt",
             "results/formative/run2.txt", *extra]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        head = None
    return {"when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "head": head, "python": sys.version.split()[0], "numpy": np.__version__,
            "digests": {f: digest(os.path.join(ROOT, f)) for f in files}}


def load(path=OUT):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"passes": []}


def save(state, path=OUT):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def prereg_digest(p=None):
    return hashlib.sha256(json.dumps(p or PREREGISTRATION, sort_keys=True).encode()).hexdigest()


# --- the reading: pools ---------------------------------------------------------------------

def shipped_level():
    """submission/model.py's LEVEL, read without importing it."""
    with open(os.path.join(ROOT, "submission", "model.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL"
                                                for t in node.targets):
            return {k: float(v) for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError("no LEVEL in submission/model.py")


def pool_a():
    """Pool A: level_calibration.json, one row per pair appearance with the
    Predictor's per-budget Brier and each configuration's per-pair ALC."""
    import experiments.level_calibration as LC
    with open(LC_JSON) as f:
        st = json.load(f)
    raw = st["raw"]
    LC.register_chunks(LC.grid_chunks())
    metas = LC.collect_metas(raw)
    lv = shipped_level()
    names = {"shipped": LC.gname(lv["mu0"], lv["sigma_mu"], lv["attr_scale"]),
             "neighbour": LC.SHIP, "smoothed": LC.SMOOTHED, "hier defaults": LC.HIER}
    assert names["neighbour"] == "hier G mu0=-3.00 sm=2.50 as=0.25", names
    rows = []
    for (reg, i) in sorted(metas):
        p = LC.rows_of(raw, reg, i, LC.PRED)
        if p is None or "pb" not in p:
            continue
        cf = {c: LC.rows_of(raw, reg, i, n) for c, n in names.items()}
        for j, m in enumerate(metas[(reg, i)]):
            rows.append({"regime": reg, "run": i, "j": j, "key": (m["bench"], m["subject"]),
                         "eval": m["eval"], "p": m["p"], "logit": m["logit"],
                         "parent": m["parent"], "pred": np.array(p["pb"][j]),
                         "pred_alc": p["alc"][j],
                         "alc": {c: (None if r is None else r["alc"][j]) for c, r in cf.items()}})
    hier_digests = sorted({ps.get("digests", {}).get("paiec/hier.py") for ps in st.get("passes", [])})
    return rows, {"configs": names, "hier_py_digests": hier_digests}


def pool_b(a_rows):
    """Pool B: the shipped hier's per-pair Brier by budget (subject_side rows),
    with the Predictor's for the same pairs where a results file holds them."""
    import experiments.subject_side as SS
    by_a = {(r["regime"], r["run"]): [] for r in a_rows}
    for r in a_rows:
        by_a[(r["regime"], r["run"])].append(r)
    with open(TC_JSON) as f:
        tc = json.load(f)["raw"]
    with open(HE_JSON) as f:
        he = json.load(f)["raw"]
    checks = Counter()
    rows = []
    for reg in SS.PLAN:
        runs = SS.load_rows(SS.ROWS, reg)
        for i in sorted(runs):
            got = runs[i]
            if "ship" not in got["res"]:
                continue
            meta, sb = got["meta"], got["res"]["ship"]["b"]
            pred = [None] * len(meta)
            if reg == "tl":
                tr = tc.get(f"check|default|{i}")
                if tr is not None:
                    ok = len(tr["rows"]) == len(meta) and all(
                        (x["pseudo"], x["subject"], x["eval"]) == (m["bench"], m["subject"], m["eval"])
                        and abs(x["p"] - m["p"]) < 1e-9 for x, m in zip(tr["rows"], meta))
                    checks[f"{reg} testlike_check {'ok' if ok else 'MISMATCH'}"] += 1
                    if ok:
                        pred = [x["pred"][:6] for x in tr["rows"]]
            elif reg in ("r1b", "r1p"):
                w = "benchmark" if reg == "r1b" else "pair"
                hr = he.get(f"r1|{w}|pair|0|{i}|Predictor")
                if hr is not None and len(hr["rows"]) == len(meta):
                    ok = True
                    if (reg, i) in by_a:   # the pair list and the Predictor's rows, both checked
                        ar = by_a[(reg, i)]
                        ok = len(ar) == len(meta) and all(
                            a["key"] == (m["bench"], m["subject"]) and a["eval"] == m["eval"]
                            and float(np.max(np.abs(a["pred"] - np.array(h[:6])))) < 2e-5
                            for a, m, h in zip(ar, meta, hr["rows"]))
                        checks[f"{reg} hier_eval vs pool A {'ok' if ok else 'MISMATCH'}"] += 1
                    else:
                        checks[f"{reg} hier_eval by order only"] += 1
                    if ok:
                        pred = [h[:6] for h in hr["rows"]]
            else:
                ar = by_a.get((reg, i))
                if ar is not None:
                    ok = len(ar) == len(meta) and all(
                        a["key"] == (m["bench"], m["subject"]) and a["eval"] == m["eval"]
                        for a, m in zip(ar, meta))
                    checks[f"{reg} pool A {'ok' if ok else 'MISMATCH'}"] += 1
                    if ok:
                        pred = [list(a["pred"]) for a in ar]
            for j, m in enumerate(meta):
                rows.append({"regime": reg, "run": i, "j": j, "p": m["p"], "logit": m["logit"],
                             "eval": m["eval"], "parent": m["parent"],
                             "ship": np.array(sb[j], float),
                             "pred": None if pred[j] is None else np.array(pred[j], float)})
    return rows, dict(checks)


def ship_consistency(a_rows, b_rows):
    """Pool A's shipped-config per-pair ALC against pool B's ship rows on the
    appearances both hold (the same model, scored by two scripts)."""
    b = {(r["regime"], r["run"], r["j"]): float(np.dot(W, r["ship"])) for r in b_rows}
    d = [abs(r["alc"]["shipped"] - b[(r["regime"], r["run"], r["j"])]) for r in a_rows
         if r["alc"]["shipped"] is not None and (r["regime"], r["run"], r["j"]) in b]
    return {"appearances": len(d), "max_abs_diff": float(max(d)) if d else None,
            "mean_abs_diff": float(np.mean(d)) if d else None}


# --- the reading: estimator --------------------------------------------------------------

def neighbours(X, v, K):
    """The audit's matching: indices of the K rows of X nearest v, each column
    standardised by its sd over X (population sd, as the audit had it)."""
    sd = X.std(0)
    d = np.sqrt((((X - v) / sd) ** 2).sum(1))
    return np.argsort(d)[:K], d


def rate_from_brier(b31):
    from paiec import testlike as T
    return T.rate_from_brier(b31)


def cc_logit(rate, n):
    return math.log((rate * n + 0.5) / (n - rate * n + 0.5))


def plain_logit(rate):
    return math.log(rate / (1 - rate))


def matched(pool, feature, profiles, K, diffs=None, cols=slice(0, 6)):
    """Per feedback pair: its K neighbours' rates and, for every named
    difference, the neighbours' mean. `profiles` are (b31, n, profile) with the
    profile in the pool's feature order."""
    X = np.array([feature(r)[cols] for r in pool])
    P = np.array([r["p"] for r in pool])
    L = np.array([r["logit"] for r in pool])
    D = {k: np.array([f(r) for r in pool], float) for k, f in (diffs or {}).items()}
    out = []
    for b31, n, v in profiles:
        nn, d = neighbours(X, np.asarray(v, float)[cols], K)
        low = rate_from_brier(b31)
        share = float((P[nn] > 0.5).mean())
        upper = share > 0.5
        rate = 1 - low if upper else low
        row = {"b31": b31, "n": n, "lower_root": low, "nn_mean_p": float(P[nn].mean()),
               "nn_share_above_half": share, "root": "upper" if upper else "lower",
               "rate": rate, "logit_cc": cc_logit(rate, n), "logit_plain": plain_logit(rate),
               "logit_cc_lower": cc_logit(low, n), "logit_cc_upper": cc_logit(1 - low, n),
               "nn_mean_logit": float(L[nn].mean()), "nn_median_dist": float(np.median(d[nn])),
               "nn_regimes": dict(sorted(Counter(pool[j]["regime"] for j in nn).items()))}
        for k, arr in D.items():
            x = arr[nn]
            x = x[~np.isnan(x)] if x.ndim == 1 else x[~np.isnan(x).any(1)]
            row[k] = x.mean(0).tolist() if x.ndim > 1 else float(x.mean())
            if x.ndim == 1:
                row[k + "_se"] = float(x.std(ddof=1) / math.sqrt(len(x)))
        out.append(row)
    return out


def estimate(rows, key, base):
    """The run's matched estimate: its feedback (ALC or budget vector) plus
    the mean over its pairs of the neighbours' mean difference."""
    return (np.asarray(base, float) + np.mean([r[key] for r in rows], 0)).tolist()


# --- the reading: the level distribution ---------------------------------------------------

def mixture(x, starts=20, seed=0, floor=0.1):
    """1-D Gaussian mixtures with one and two components by EM (several
    starts, sd floored); BIC for each."""
    x = np.asarray(x, float)
    n = len(x)
    mu, sd = x.mean(), max(x.std(), floor)
    ll1 = float(np.sum(-0.5 * ((x - mu) / sd) ** 2 - np.log(sd * math.sqrt(2 * math.pi))))
    best = None
    rng = np.random.default_rng(seed)
    for s in range(starts):
        if s == 0:
            m = np.array([np.percentile(x, 25), np.percentile(x, 75)])
        else:
            m = rng.choice(x, 2, replace=False) + rng.normal(0, 0.1, 2)
        sg, pi = np.array([sd, sd]), np.array([0.5, 0.5])
        for _ in range(500):
            dens = pi * np.exp(-0.5 * ((x[:, None] - m) / sg) ** 2) / (sg * math.sqrt(2 * math.pi))
            tot = dens.sum(1, keepdims=True) + 1e-300
            r = dens / tot
            nk = r.sum(0) + 1e-12
            m_new = (r * x[:, None]).sum(0) / nk
            sg = np.maximum(np.sqrt((r * (x[:, None] - m_new) ** 2).sum(0) / nk), floor)
            pi = nk / n
            if np.max(np.abs(m_new - m)) < 1e-9:
                m = m_new
                break
            m = m_new
        dens = pi * np.exp(-0.5 * ((x[:, None] - m) / sg) ** 2) / (sg * math.sqrt(2 * math.pi))
        ll = float(np.log(dens.sum(1) + 1e-300).sum())
        if best is None or ll > best[0]:
            o = np.argsort(m)
            best = (ll, m[o], sg[o], pi[o])
    bic1 = -2 * ll1 + 2 * math.log(n)
    bic2 = -2 * best[0] + 5 * math.log(n)
    return {"one": {"mean": float(mu), "sd": float(sd), "bic": bic1},
            "two": {"means": best[1].tolist(), "sds": best[2].tolist(), "weights": best[3].tolist(),
                    "bic": bic2},
            "delta_bic_one_minus_two": bic1 - bic2}


def anova(x, groups):
    """One-way random-effects split of the pair logits by benchmark id: the
    between-benchmark and within-benchmark sd (ANOVA estimator, unbalanced)."""
    x = np.asarray(x, float)
    gs = sorted(set(groups))
    idx = [np.array([i for i, g in enumerate(groups) if g == h]) for h in gs]
    N, k = len(x), len(gs)
    grand = x.mean()
    msb = sum(len(i) * (x[i].mean() - grand) ** 2 for i in idx) / (k - 1)
    msw = sum(((x[i] - x[i].mean()) ** 2).sum() for i in idx) / (N - k)
    n0 = (N - sum(len(i) ** 2 for i in idx) / N) / (k - 1)
    return {"groups": k, "pairs": N, "between_sd": math.sqrt(max(0.0, (msb - msw) / n0)),
            "within_sd": math.sqrt(msw), "n0": n0}


MIX_DRAWS = 500


def level_distribution(pairs, reads, seed=0, boots=BOOTS, full=True):
    """The level distribution of a set of pairs from one K's reads: point
    values and a bootstrap that resamples benchmark ids as clusters (a
    resampled cluster is its own group, duplicates included) and draws each
    pair's root, upper with its neighbour share. With `full`, also the
    one- and two-component mixtures (the share of MIX_DRAWS root draws, on the
    pairs as they are, in which BIC prefers two components by more than 2) and
    the between/within-benchmark split."""
    x = np.array([r["logit_cc"] for r in reads])
    xp = np.array([r["logit_plain"] for r in reads])
    lo = np.array([r["logit_cc_lower"] for r in reads])
    hi = np.array([r["logit_cc_upper"] for r in reads])
    share = np.array([r["nn_share_above_half"] for r in reads])
    bench = [p["benchmark"] for p in pairs]
    cl = sorted(set(bench))
    members = [[i for i, b in enumerate(bench) if b == c] for c in cl]
    rng = np.random.default_rng(seed)
    means, sds, bsd, two = [], [], [], []
    for t in range(boots):
        xs = np.where(rng.random(len(x)) < share, hi, lo)
        vals, grp = [], []
        for k, c in enumerate(rng.choice(len(cl), len(cl))):
            vals += [xs[i] for i in members[c]]
            grp += [k] * len(members[c])
        v = np.array(vals)
        means.append(v.mean())
        sds.append(v.std(ddof=1))
        if full:
            bsd.append(anova(v, grp)["between_sd"])
            if t < MIX_DRAWS:
                two.append(mixture(xs, starts=4, seed=1)["delta_bic_one_minus_two"] > 2)
    q = lambda a: [float(np.percentile(a, 5)), float(np.percentile(a, 95))]  # noqa: E731
    out = {
        "logits_cc": x.tolist(), "logits_plain": xp.tolist(),
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "sd_ddof0": float(x.std()),
        "mean_plain": float(xp.mean()), "sd_plain": float(xp.std(ddof=1)),
        "nn_mean_logit": {"mean": float(np.mean([r["nn_mean_logit"] for r in reads])),
                          "sd": float(np.std([r["nn_mean_logit"] for r in reads], ddof=1))},
        "upper_roots": int(sum(r["root"] == "upper" for r in reads)),
        "bootstrap": {"boots": boots, "clusters": len(cl),
                      "mean_se": float(np.std(means, ddof=1)), "mean_90": q(means),
                      "sd_se": float(np.std(sds, ddof=1)), "sd_90": q(sds),
                      "note": "clusters are the benchmark ids (17 pairs are not 17 independent "
                              "levels); each pair's root drawn upper with its neighbour share"}}
    if full:
        out["bootstrap"].update(between_sd_90=q(bsd), between_sd_se=float(np.std(bsd, ddof=1)),
                                mixture_draws=len(two),
                                share_two_components_preferred=float(np.mean(two)))
        out["mixture"] = mixture(x)
        out["between_within"] = anova(x, bench)
    return out


def level_implied():
    """The shipped LEVEL on the pair-logit scale at attribute score 0: centre
    f mu0, spread f (sigma_mu^2 + sigma_theta^2 + sigma_delta^2)^1/2, with
    f = (1 + pi T / 8)^-1/2 the logistic-normal factor for the item variance
    T = sigma_d^2 + sigma_g^2 (docs/findings.md, 'Calibrating for the hidden
    test'). The hidden subjects' attribute standing moves the centre by
    f attr_scale m, which no feedback shows."""
    from paiec.hier import Hyper
    h = Hyper()
    lv = shipped_level()
    T_ = h.sigma_d ** 2 + h.sigma_g ** 2
    f = (1 + math.pi * T_ / 8) ** -0.5
    within = math.sqrt(h.sigma_theta ** 2 + h.sigma_delta ** 2)
    return {"LEVEL": lv, "T": T_, "f": f, "mean": f * lv["mu0"],
            "between_sd": f * lv["sigma_mu"], "within_sd": f * within,
            "sd": f * math.sqrt(lv["sigma_mu"] ** 2 + within ** 2),
            "hyper": {k: getattr(h, k) for k in ("sigma_theta", "sigma_delta", "sigma_d", "sigma_g")}}


# --- the reading: predictions for run 2 -----------------------------------------------------

def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def reweighted(rows, values, mean, sd, width=0.5, lo=-7.25, hi=7.25):
    """NEW ESTIMATOR (not the audit's). The mean of `values` (per pair
    appearance) with appearances reweighted, in bins of pair logit, to a
    Gaussian level distribution N(mean, sd^2): each bin gets the Gaussian's mass,
    shared equally by the appearances in it. Mass in empty bins is lost and
    reported as coverage."""
    edges = np.arange(lo, hi + 1e-9, width)
    lg = np.array([r["logit"] for r in rows])
    b = np.clip(np.digitize(lg, edges), 0, len(edges))
    cnt = np.bincount(b, minlength=len(edges) + 1)
    cdf = [0.0] + [_ncdf((e - mean) / sd) for e in edges] + [1.0]
    mass = np.diff(cdf)
    w = np.where(cnt[b] > 0, mass[b] / np.maximum(cnt[b], 1), 0.0)
    V = np.asarray(values, float)
    est = (w[:, None] * V).sum(0) / w.sum()
    return {"estimate": est.tolist(), "coverage": float(mass[cnt > 0].sum()),
            "effective_n": float(w.sum() ** 2 / (w ** 2).sum())}


def run_sd(rows, key):
    """Single-run sd of ALC and of each budget by regime: runs' pair means."""
    by = defaultdict(list)
    for r in rows:
        v = r[key]
        if v is not None:
            by[(r["regime"], r["run"])].append(np.append(v, np.dot(W, v)))
    reg = defaultdict(list)
    for (g, _), v in by.items():
        reg[g].append(np.mean(v, 0))
    return {g: {"runs": len(v), "mean": np.mean(v, 0).tolist(), "sd": np.std(v, 0, ddof=1).tolist()}
            for g, v in reg.items()}


# --- the reading ---------------------------------------------------------------------------

def reading(record):
    from paiec import testlike as T
    r1, r2 = record["run1"]["pairs"], record["run2"]["pairs"]
    a_rows, a_info = pool_a()
    b_rows, b_checks = pool_b(a_rows)
    with open(TC_JSON) as f:
        tcs = json.load(f)["summary"]
    regime_pl = tcs["check"]["default"]["regime"]["pair_logit"]
    public_pl = tcs["r1"]["public R1"]["regime"]["pair_logit"]
    out = {"pools": {
        "A": {"source": "results/level_calibration.json", "appearances": len(a_rows),
              "by_regime": dict(Counter(r["regime"] for r in a_rows)), **a_info},
        "B": {"source": "data/subject_side_rows (shipped hier) + Predictor rows from "
                        "results/testlike_check.json, results/hier_eval.json and pool A",
              "appearances": len(b_rows),
              "by_regime": dict(Counter(r["regime"] for r in b_rows)),
              "with_predictor": dict(Counter(r["regime"] for r in b_rows if r["pred"] is not None)),
              "pairing_checks": b_checks},
        "shipped_rows_A_vs_B": ship_consistency(a_rows, b_rows)}}

    # 1. the port check: the audit's own numbers on its own inputs (T.FEEDBACK, 4 dp)
    fb_audit = [(v[5], n, v) for _, n, v in T.FEEDBACK]
    screen = [r for r in a_rows if r["alc"]["shipped"] is not None]
    port = {}
    for (cfg, pool_name, K), rec in AUDIT_RECORDED.items():
        pool = a_rows if pool_name == "full" else screen
        pool = [r for r in pool if r["alc"][cfg] is not None]
        rows = matched(pool, lambda r: r["pred"], fb_audit, K,
                       {"d": lambda r, c=cfg: r["alc"][c] - r["pred_alc"]})
        est = T.FEEDBACK_ALC + float(np.mean([x["d"] for x in rows]))
        port[f"{cfg}|{pool_name}|K={K}"] = {"estimate": est, "audit_recorded": rec,
                                             "abs_diff": abs(est - rec)}
    out["port_check"] = port

    # 2. Q2: the shipped config's matched estimate on run 1 (parsed feedback, 6 dp)
    fb1 = [(p["brier"][5], p["n"], p["brier"]) for p in r1]
    alc1 = record["run1"]["alc"]["recomputed"]
    q2 = {}
    for K in KS:
        diffs = {c: (lambda r, c=c: np.nan if r["alc"][c] is None else r["alc"][c] - r["pred_alc"])
                 for c in ("shipped", "neighbour", "smoothed", "hier defaults")}
        rows = matched(screen, lambda r: r["pred"], fb1, K, diffs)
        q2[f"K={K}"] = {"pool": "A, where the shipped config is scored (test-like runs 0-99, "
                                "public benchmark-first 0-39)",
                        "estimate_alc": {c: alc1 + float(np.mean([x[c] for x in rows]))
                                         for c in diffs},
                        "per_pair_shipped_minus_predictor": [x["shipped"] for x in rows],
                        "per_pair_se": [x["shipped_se"] for x in rows]}
    # robustness, as the audit's third script: pool restricted by regime, and
    # matching on (B0, B31) only, K = 25
    rob = {}
    sel = {"all": lambda r: True, "test-like default only": lambda r: r["regime"] == "tl",
           "public only": lambda r: r["regime"].startswith("r1")}
    for label, f in sel.items():
        sub = [r for r in screen if f(r)]
        for cols, cl in ((slice(0, 6), "6 budgets"), ([0, 5], "B0,B31")):
            rows = matched(sub, lambda r: r["pred"], fb1, 25,
                           {"d": lambda r: r["alc"]["shipped"] - r["pred_alc"]}, cols=cols)
            rob[f"{label} | {cl}"] = {"appearances": len(sub),
                                      "estimate_alc": alc1 + float(np.mean([x["d"] for x in rows])),
                                      "mean_nn_median_dist": float(np.mean([x["nn_median_dist"]
                                                                            for x in rows]))}
    q2["robustness_K=25"] = rob
    # per budget, on pool B (the shipped hier's own rows, three times the runs)
    bp = [r for r in b_rows if r["pred"] is not None]
    for K in KS:
        rows = matched(bp, lambda r: r["pred"], fb1, K,
                       {"db": lambda r: r["ship"] - r["pred"]})
        est_b = estimate(rows, "db", record["run1"]["budgets"]["brier_mean"])
        q2[f"pool_B_K={K}"] = {"pool": "B, every appearance with both rows",
                               "appearances": len(bp),
                               "estimate_budgets": est_b,
                               "estimate_alc": float(np.dot(W, est_b))}
    out["q2_shipped_on_run1"] = q2

    # 3. the per-pair level reading of both runs
    fb2 = [(p["brier"][5], p["n"], p["brier"]) for p in r2]
    reads = {}
    for K in KS:
        a = matched(a_rows, lambda r: r["pred"], fb1, K)
        b = matched(b_rows, lambda r: r["ship"], fb2, K)
        a_b = matched(bp, lambda r: r["pred"], fb1, K)   # run 1 on pool B, as a check
        reads[K] = a + b
        out.setdefault("per_pair", {})[f"K={K}"] = {
            "run1_pool_A": a, "run2_pool_B": b,
            "run1_pool_B_roots_agree": [x["root"] == y["root"] for x, y in zip(a, a_b)]}
    pairs17 = r1 + r2
    n1 = len(r1)
    dist = {f"K={K}": level_distribution(pairs17, reads[K]) for K in KS}
    dist["run1_only"] = {f"K={K}": level_distribution(r1, reads[K][:n1], full=False) for K in KS}
    dist["run2_only"] = {f"K={K}": level_distribution(r2, reads[K][n1:], full=False) for K in KS}
    out["level_distribution"] = dist

    # 4. comparisons
    lr = {}
    for label, up in (("all lower roots", ()), ("p6, p8 upper (the audit)", (5, 7)),
                      ("p6, p8, p9 upper", (5, 7, 8))):
        xs = [(1 - rate_from_brier(v[5])) if j in up else rate_from_brier(v[5])
              for j, (_, n, v) in enumerate(T.FEEDBACK)]
        pl = np.array([plain_logit(x) for x in xs])
        cc = np.array([cc_logit(x, n) for x, (_, n, _) in zip(xs, T.FEEDBACK)])
        lr[label] = {"mean_plain": float(pl.mean()), "sd_plain": float(pl.std(ddof=1)),
                     "mean_cc": float(cc.mean()), "sd_cc": float(cc.std(ddof=1))}
    imp = level_implied()
    out["comparison"] = {"shipped_LEVEL_pair_logit_scale": imp,
                         "tuned_regime_realised": regime_pl,
                         "public_R1_realised": public_pl,
                         "run1_readings_recomputed": lr}

    # 5. the preregistered decision, with the reading-bias guard
    def bias_of(rows, key):
        read, true = [], []
        for r in rows:
            low = rate_from_brier(float(r[key][5]))
            rate = 1 - low if r["p"] > 0.5 else low
            read.append(cc_logit(rate, r["eval"]))
            true.append(r["logit"])
        read, true = np.array(read), np.array(true)
        return {"appearances": len(read), "read_mean": float(read.mean()),
                "true_mean": float(true.mean()), "read_sd": float(read.std()),
                "true_sd": float(true.std()), "mean_shift": float(read.mean() - true.mean()),
                "sd_shift": float(read.std() - true.std())}
    bias = {"predictor_pool_A_tl": bias_of([r for r in a_rows if r["regime"] == "tl"], "pred"),
            "shipped_pool_B_tl": bias_of([r for r in b_rows if r["regime"] == "tl"], "ship")}
    wts = (len(r1) / (len(r1) + len(r2)), len(r2) / (len(r1) + len(r2)))
    bias["weighted_mean_shift"] = (wts[0] * bias["predictor_pool_A_tl"]["mean_shift"] +
                                   wts[1] * bias["shipped_pool_B_tl"]["mean_shift"])
    bias["weighted_sd_shift"] = (wts[0] * bias["predictor_pool_A_tl"]["sd_shift"] +
                                 wts[1] * bias["shipped_pool_B_tl"]["sd_shift"])
    means = {K: dist[f"K={K}"]["mean"] for K in KS}
    ses = {K: dist[f"K={K}"]["bootstrap"]["mean_se"] for K in KS}
    sds = {K: dist[f"K={K}"]["sd"] for K in KS}
    sd_ses = {K: dist[f"K={K}"]["bootstrap"]["sd_se"] for K in KS}
    c_mean, c_sd = -1.290, 1.696
    z_mu = {K: (means[K] - c_mean) / ses[K] for K in KS}
    z_sd = {K: (sds[K] - c_sd) / sd_ses[K] for K in KS}
    lv = imp["LEVEL"]
    mu_prop = None
    if all(abs(z) > 2 for z in z_mu.values()) and len({np.sign(z) for z in z_mu.values()}) == 1:
        m = float(np.mean(list(means.values())))
        mu_prop = round((lv["mu0"] + (m - c_mean) / imp["f"]) * 2) / 2
    sm_prop = None
    if all(abs(z) > 2 for z in z_sd.values()) and len({np.sign(z) for z in z_sd.values()}) == 1:
        s = float(np.mean(list(sds.values())))
        v = max(0.25, lv["sigma_mu"] ** 2 + (s ** 2 - c_sd ** 2) / imp["f"] ** 2)
        sm_prop = min((1.3, 1.8, 2.5, 3.5, 5.0), key=lambda g: abs(g - math.sqrt(v)))
    conf_mu = any(abs(bias["weighted_mean_shift"]) > 2 * ses[K] for K in KS)
    conf_sd = any(abs(bias["weighted_sd_shift"]) > 2 * sd_ses[K] for K in KS)
    confounded = []
    if mu_prop is not None and conf_mu:
        confounded.append({"mu0": mu_prop})
        mu_prop = None
    if sm_prop is not None and conf_sd:
        confounded.append({"sigma_mu": sm_prop})
        sm_prop = None
    dec = {"comparator": {"mean": c_mean, "sd": c_sd},
           "z_mean": {f"K={K}": z_mu[K] for K in KS}, "z_sd": {f"K={K}": z_sd[K] for K in KS},
           "reading_bias": bias, "bias_guard_trips": {"mean": conf_mu, "sd": conf_sd},
           "confounded_candidates": confounded,
           "mu0_candidate": mu_prop, "sigma_mu_candidate": sm_prop,
           "outcome": ("no candidate: LEVEL stays" if mu_prop is None and sm_prop is None else
                       "candidate proposed for scoring under the preregistered gate; nothing ships "
                       "from this reading")}
    out["decision"] = dec

    # 6. predictions for run 2
    obs2 = record["run2"]["budgets"]["brier_mean"] + [record["run2"]["alc"]["recomputed"]]
    sd_ship = run_sd(b_rows, "ship")
    pred = {"observed_run2": obs2, "single_run_sd_shipped": sd_ship}
    p1 = {}
    for K in KS:
        eb = q2[f"pool_B_K={K}"]["estimate_budgets"]
        est = eb + [float(np.dot(W, eb))]
        p1[f"K={K}"] = {"estimate": est,
                        "pool_A_alc": q2[f"K={K}"]["estimate_alc"]["shipped"],
                        "observed_minus_estimate": [o - e for o, e in zip(obs2, est)],
                        "z_test_like": [(o - e) / s for o, e, s in
                                        zip(obs2, est, sd_ship["tl"]["sd"])],
                        "z_public_benchmark_first": [(o - e) / s for o, e, s in
                                                     zip(obs2, est, sd_ship["r1b"]["sd"])]}
    pred["matched_estimate_of_run1"] = p1
    groups = {"test-like": ("tl", "tl mix/whole", "tl no shift"), "public": ("r1b", "r1p")}
    targets = {"audit reading of run 1 (p6, p8 upper; cc)": (
        lr["p6, p8 upper (the audit)"]["mean_cc"], lr["p6, p8 upper (the audit)"]["sd_cc"]),
        "run 1 matched reading, K=15": (dist["run1_only"]["K=15"]["mean"],
                                        dist["run1_only"]["K=15"]["sd"]),
        "17-pair reading, K=15": (dist["K=15"]["mean"], dist["K=15"]["sd"]),
        "17-pair reading, K=40": (dist["K=40"]["mean"], dist["K=40"]["sd"]),
        "tuned regime (reference)": (regime_pl["mean"], regime_pl["sd"])}
    rw = {}
    for g, regs in groups.items():
        sub = [r for r in b_rows if r["regime"] in regs]
        vals = [np.append(r["ship"], np.dot(W, r["ship"])) for r in sub]
        subp = [r for r in sub if r["pred"] is not None]
        valp = [np.append(r["pred"], np.dot(W, r["pred"])) for r in subp]
        for t, (m, s) in targets.items():
            e = reweighted(sub, vals, m, s)
            row = {"target": [m, s], "shipped": e,
                   "observed_minus_shipped": [o - x for o, x in zip(obs2, e["estimate"])]}
            if subp:
                row["predictor"] = reweighted(subp, valp, m, s)
            rw[f"{g} | {t}"] = row
    pred["level_reweighted"] = rw
    pred["run1_observed"] = record["run1"]["budgets"]["brier_mean"] + [alc1]
    out["predictions_run2"] = pred

    # 7. descriptive: where run 2's B0 excess over B31 sits (findings, "Subject
    # side at budgets 0 and 1", asks whether it is on low-rate pairs). A B31 of
    # 0.25 or more reads as 0.5 on either side, so those pairs are kept apart.
    ex = [p["brier"][0] - p["brier"][5] for p in r2]
    b0x = {"b0_minus_b31": ex, "b0_minus_b31_mean": float(np.mean(ex))}
    for K in KS:
        b2 = out["per_pair"][f"K={K}"]["run2_pool_B"]
        side = ["unidentified" if x["lower_root"] >= 0.5 else
                ("below half" if x["rate"] < 0.5 else "above half") for x in b2]
        b0x[f"K={K}"] = {"rate": [x["rate"] for x in b2], "side": side,
                         **{s: {"pairs": side.count(s),
                                "mean_b0_minus_b31": (float(np.mean([e for e, t in zip(ex, side)
                                                                     if t == s]))
                                                      if s in side else None)}
                            for s in ("below half", "unidentified", "above half")}}
    out["run2_b0_excess_by_rate"] = b0x
    out["notes"] = [
        "port_check: the audit's scratch estimator on its own inputs; the screen pool here is "
        "every appearance where the shipped config is scored, the audit's also dropped runs "
        "missing any screened config, hence differences up to 1e-5",
        "level_reweighted is a new estimator (bins of pair logit, width 0.5), a proxy for scoring "
        "a regime at that level; it reweights the level only, not the attribute optimism the "
        "date shift carries, which is why the Predictor's reweighted B0 falls short of run 1's",
        "the 17 pairs sit on 7 benchmark ids; every SE here resamples the benchmarks",
        "the reading-bias guard applies the root reading on the true side; side errors of the "
        "neighbour rule add to it and are not measured"]
    return out


# --- stages ---------------------------------------------------------------------------------

def stage_record(args, state):
    runs = {}
    for name, path in FEEDBACK.items():
        with open(path) as f:
            runs[name] = summarise_run(name, parse(f.read()))
    rec = {**runs, "overlap": overlap(runs["run1"]["pairs"], runs["run2"]["pairs"])}
    old = state.get("archives", {})
    arch = {}
    for name, path, rebuilt in (("run1", args.archive1, args.rebuild1),
                                ("run2", args.archive2, args.rebuild2)):
        if path:
            arch[name] = verify_archive(os.path.abspath(path), RUNS[name]["commit"])
            arch[name]["how"] = getattr(args, f"{name}_how") or ""
            if rebuilt:
                rb = verify_archive(os.path.abspath(rebuilt), RUNS[name]["commit"])
                arch[name]["rebuild"] = rb
                arch[name]["rebuild_equals_archive"] = rb["sha256"] == arch[name]["sha256"]
        else:
            arch[name] = old.get(name, {"sha256": "not recorded"})
    state.update(record=rec, archives=arch, submissions=SUBMISSIONS)
    state["passes"].append({"stage": "record", "command": " ".join(sys.argv), **provenance()})


def stage_prereg(args, state):
    if "reading" in state:
        sys.exit("a reading already exists; the preregistration cannot be (re)written after it")
    if "preregistration" in state and state["preregistration"]["sha256"] != prereg_digest():
        sys.exit("a different preregistration is already stored")
    if "preregistration" not in state:
        state["preregistration"] = {**PREREGISTRATION, "sha256": prereg_digest(),
                                    "written_utc": provenance()["when_utc"]}
    state["passes"].append({"stage": "prereg", "command": " ".join(sys.argv), **provenance()})


def stage_read(args, state):
    pre = state.get("preregistration")
    if pre is None:
        sys.exit("no preregistration stored: run --stage prereg first")
    body = {k: v for k, v in pre.items() if k not in ("sha256", "written_utc")}
    if prereg_digest(body) != pre["sha256"] or pre["sha256"] != prereg_digest():
        sys.exit("the stored preregistration differs from PREREGISTRATION in this script")
    if "record" not in state:
        sys.exit("run --stage record first")
    state["reading"] = reading(state["record"])
    state["passes"].append({"stage": "read", "command": " ".join(sys.argv),
                            **provenance(["results/level_calibration.json",
                                          "results/testlike_check.json",
                                          "results/hier_eval.json",
                                          "data/subject_side_rows/tl.jsonl",
                                          "data/subject_side_rows/r1b.jsonl",
                                          "data/subject_side_rows/r1p.jsonl",
                                          "data/subject_side_rows/tl_mix-whole.jsonl",
                                          "data/subject_side_rows/tl_no_shift.jsonl",
                                          "submission/model.py", "paiec/hier.py"])})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("record", "prereg", "read"), required=True)
    ap.add_argument("--archive1", help="the archive behind formative run 1 (checked against b68492c)")
    ap.add_argument("--archive2", help="the archive behind formative run 2 (checked against ee5085a)")
    ap.add_argument("--rebuild1", help="an archive rebuilt from b68492c, compared with --archive1")
    ap.add_argument("--rebuild2", help="an archive rebuilt from ee5085a, compared with --archive2")
    ap.add_argument("--run1-how", dest="run1_how", default="",
                    help="how the run-1 archive was obtained, recorded verbatim")
    ap.add_argument("--run2-how", dest="run2_how", default="",
                    help="how the run-2 archive was obtained, recorded verbatim")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    state = load(args.out)
    {"record": stage_record, "prereg": stage_prereg, "read": stage_read}[args.stage](args, state)
    save(state, args.out)
    print(f"wrote {os.path.relpath(args.out, ROOT)} ({args.stage})")


if __name__ == "__main__":
    main()
