"""The shipped configuration's own numbers, from stored rows only (review P0.1,
its section 7 and P1.14; docs/report/review_v0.md).

What ships is paiec.hier.HierPredictor with submission/model.py's LEVEL (mu0
-2.5, sigma_mu 2.5, attr_scale 0.5) over prior.build. Until now the report
quoted a neighbouring configuration's held-out numbers for it. This script
reads rows other studies already stored and makes no model call:

  shipped   the 'ship' arm of data/subject_side_rows/<regime>.jsonl
            (experiments/subject_side.py): per pair Brier by budget on every
            run of its PLAN (test-like seed 2 runs 0-299, groups merged at
            random and wholes seed 3 runs 0-199, no date shift seed 3 runs
            0-99, public R1 benchmark-first runs 0-149 and pair-uniform runs
            0-99, split scope 'pair')
  legacy    the first submission's predictor (paiec.predict.Predictor, its
            attribute prior fitted without the target's parent), per pair:
            test-like, results/testlike_check.json raw['check|default|<i>']
            ['rows'][*]['pred'] (seed 2, the same runs); public, results/
            hier_eval.json raw['r1|<weighting>|pair|0|<i>|Predictor']['rows'];
            the two sensitivity regimes, results/level_calibration.json
            raw['<regime>|<i>|base'] 'Predictor' (runs 0-99)
  smoothed  Beta(2,2) on the pair's own labels (baselines.smoothed_mean(4,
            0.5)), per pair: test-like runs 0-199 and the sensitivity regimes'
            runs 0-99 from level_calibration.json's 'base' tasks; public runs
            from hier_eval.json '...|smoothed Beta(2,2)'. No per-pair smoothed
            rows exist for test-like runs 200-299 or mix/whole runs 100-199,
            so there is no smoothed comparison there
  empirical the organisers' empirical mean (baselines.empirical_mean), stored
            per pair by testlike_check.json ('emp') on test-like runs 0-299 and
            on its public R1 runs (benchmark-first, seed 0, the runs of r1b);
            none is stored per pair for pair-uniform runs

Every comparator's run is matched to the shipped rows' run pair for pair
before it is used: testlike_check rows and level_calibration's run metadata
carry (pseudo-benchmark, parent, subject, evaluated items, rate), compared
field by field; hier_eval rows carry no ids, so a hier_eval run is matched
through its Predictor rows, which must equal testlike_check's public R1
Predictor rows exactly (benchmark-first, runs 0-149) or level_calibration's to
its 1e-5 packing (runs 0-99 of both weightings). A comparator run that cannot
be matched is not used, and a stored run that disagrees stops the script.

Statistics (level_calibration's, so the numbers compare with it): a run's ALC
is the mean over its pairs of the weighted Brier sum; a difference is shipped
minus comparator, paired on identical runs, with "± run SE / cluster SE /
stratified SE": over runs, a cluster bootstrap (level_calibration.Boot: 2,000
resamples, a ratio estimator, cluster (parent, subject), seed 0) and the same
bootstrap resampled within each parent benchmark. Per parent: the appearance-
weighted mean difference (weight 1 / run size, experiments/harness.py's
parent_means) with its cluster SE, and the mean with that parent left out
(level_calibration.lopo). The parent-level mean ± SE is the mean and SE of the
four multi-subject parents' means (harness.summary's parent_mean /
parent_se): it treats the four parents as the sample, which the cluster SEs
do not. swe_rebench (one subject; public runs only) is reported per parent and
left out of the parent-level mean; being one cluster, it has no cluster SE
(null), only its point value. Single-run sds are sds over runs of a run's
budget means and ALC: the spread one platform run of similar size is drawn
from, within one regime.

Blocks: runs 0-99, 100-199, 200-299 and all runs of a regime, where the runs
exist; a comparator is scored on a block only if it is stored for every run of
it, plus on every run it is stored for when that is a shorter prefix.

Consistency checks (review section 7; the script stops if any fails):
  - level_calibration.json's selection half for the shipped config (runs
    0-99): ALC 0.1608, the legacy Predictor 0.2039, difference -0.0431 ±
    0.0019, and its cluster and stratified SEs and leave-one-parent-out values
  - the guard's Predictor ALCs 0.2068 (benchmark-first) and 0.2038 (pair-
    uniform) on public runs 0-99, and the smoothed mean's
  - the two sensitivity regimes matched pair for pair (100 of 100 runs), and
    their Predictor and smoothed ALCs equal level_calibration's summary
  - subject_side.json's summary of the shipped arm, level_calibration's own
    rows of the shipped config (test-like runs 0-99, the 40-run public screen)
    pair for pair, mcq_floor.json's shipped arm (test-like runs 0-199, old
    floor), official_baselines.json's per-run smoothed and empirical-mean ALCs
  - the review's [R] numbers (its W1, W2 and section-7 tables) at the
    precision it printed them

Which code the rows are. The shipped rows were scored at bd0be67 with
paiec/hier.py 70a3a81a (subject_side.json's run pass), which is paiec/hier.py
at ee5085a, the commit of the archive that made formative run 2, and with the
old multiple-choice floor (they reproduce mcq_floor.json's old-floor arm). The
archive built at 4d2cc4f differs by the corrected floor (f7e7d87) and the
floored-fit fix (4d2cc4f); 'code_gap' carries what those two changes measured
on runs 0-199 (mcq_floor.json, hier_floor.json), which this script does not
add to anything.

Run: python experiments/ship_confirm.py
(reads about 45 MB of JSON; a few seconds and under 1 GB). Writes
results/ship_confirm.json only if every check passes.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.subject_side as SS  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402

OUT = os.path.join(ROOT, "results", "ship_confirm.json")
RES = os.path.join(ROOT, "results")
W6 = np.asarray(WEIGHTS, float)
BOOTS = LC.BOOTS
SHIP = LC.gname(-2.5, 2.5, 0.5)                 # level_calibration's name for what ships
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
BLOCK = 100
PLAN = dict(SS.PLAN)                            # regime -> runs 0..n-1 of the shipped rows
LABELS = {"tl": "test-like (seed 2, primary)",
          "tl mix/whole": "test-like, groups merged at random and wholes (seed 3)",
          "tl no shift": "test-like, no date shift (seed 3)",
          "r1b": "public R1, benchmark-first (seed 0, split scope pair)",
          "r1p": "public R1, pair-uniform (seed 0, split scope pair)"}
HE_WEIGHTING = {"r1b": "benchmark", "r1p": "pair"}
COMPARATORS = ("legacy", "smoothed", "empirical")
#: formative run 2 (2026-09-26): the shipped hier at ee5085a, 8 pairs. Budgets
#: as docs/findings.md records them ("Subject side at budgets 0 and 1", three
#: decimals); ALC 0.192623 as given to this study (docs/report/draft.md quotes
#: 0.1926, the formula on the rounded budgets). Our own run's feedback, read
#: only to place it among the regimes' single runs.
RUN2 = {"date": "2026-09-26", "commit": "ee5085a", "pairs": 8, "ALC": 0.192623,
        "budgets": [0.237, 0.195, 0.196, 0.184, 0.178, 0.183]}

#: the review's [R] numbers (docs/report/review_v0.md), as printed; each must be
#: reproduced at the printed precision
REVIEW_W2 = [  # (regime, runs, shipped ALC, legacy ALC, difference, run SE)
    ("tl", "0-99", "0.1608", "0.2039", "-0.0431", "0.0019"),
    ("tl", "100-199", "0.1677", "0.2073", "-0.0396", "0.0019"),
    ("tl", "200-299", "0.1687", "0.2107", "-0.0419", "0.0017"),
    ("tl", "0-299", "0.1658", "0.2073", "-0.0415", "0.0011"),
    ("r1b", "0-99", "0.2053", "0.2068", "-0.0015", "0.0010"),
    ("r1b", "0-149", None, None, "-0.0017", "0.0008"),
    ("r1p", "0-99", "0.2020", "0.2038", "-0.0018", "0.0010"),
    ("tl mix/whole", "0-99", "0.1694", "0.2129", "-0.0436", None),
    ("tl no shift", "0-99", "0.1585", "0.1729", "-0.0144", None),
]
REVIEW_SD = {  # regime -> single-run sds B0..B31, ALC (review section 7)
    "tl": "0.0209 0.0385 0.0367 0.0354 0.0336 0.0318 0.0318",
    "tl mix/whole": "0.0171 0.0400 0.0371 0.0357 0.0354 0.0327 0.0329",
    "tl no shift": "0.0423 0.0418 0.0384 0.0327 0.0307 0.0286 0.0335",
    "r1b": "0.0330 0.0373 0.0345 0.0286 0.0264 0.0252 0.0285",
    "r1p": "0.0297 0.0393 0.0297 0.0279 0.0257 0.0244 0.0261",
}
REVIEW_W1 = {  # regime -> (mean ALC, sd, run 2's z, share >= run 2, per-budget z)
    "tl": ("0.1658", "0.0318", "+0.84", "0.20", "+0.98 +0.15 +0.86 +0.89 +0.98 +1.40"),
    "tl no shift": ("0.1585", "0.0335", "+1.02", "0.18", "+1.00 +0.45 +0.89 +1.11 +1.20 +1.67"),
    "r1b": ("0.2051", "0.0285", "-0.44", "0.67", "+0.06 -0.78 -0.49 -0.46 -0.27 +0.23"),
    "r1p": ("0.2020", "0.0261", "-0.36", "0.68", "-0.24 -0.67 -0.37 -0.32 -0.13 +0.50"),
}
#: tolerances for numbers recomputed from rows stored at another rounding:
#: level_calibration packs per pair ALC and Brier in units of 1e-5 (half a unit,
#: 5e-6), the shipped rows hold Brier to 1e-7 (a pair's ALC to 5e-8)
TOL_MEAN = 5e-6
TOL_PAIR = 6e-6


def r6(x):
    return None if x is None else round(float(x), 6)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_label(ids):
    return f"{ids[0]}-{ids[-1]}"


# --- checks ------------------------------------------------------------------------------

class Checks:
    """Every consistency check, recorded; main() stops if any failed."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, got, want, source):
        self.items.append({"check": name, "ok": bool(ok), "got": got, "want": want,
                           "source": source})
        return ok

    def close(self, name, got, want, tol, source):
        return self.add(name, abs(float(got) - float(want)) <= tol, float(got), float(want),
                        f"{source} (tolerance {tol:g})")

    def shown(self, name, got, want, source, signed=False):
        """got printed at want's precision equals want (a string as printed)."""
        if want is None:
            return True
        nd = len(want.split(".")[1]) if "." in want else 0
        txt = f"{got:+.{nd}f}" if signed or want.startswith(("+", "-")) else f"{got:.{nd}f}"
        if txt in ("-0." + "0" * nd, "+0." + "0" * nd) and want.lstrip("+-") == "0." + "0" * nd:
            txt = want
        return self.add(name, txt == want, txt, want, source)

    def failed(self):
        return [c for c in self.items if not c["ok"]]


class Mismatch(RuntimeError):
    pass


# --- loading ------------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def meta_key(p):
    """What identifies a pair appearance across the stored files."""
    return (p["bench"], p["parent"], p["subject"], int(p["eval"]), round(float(p["p"]), 5))


def tc_key(r):
    return (r["pseudo"], r["parent"], r["subject"], int(r["eval"]), round(float(r["p"]), 5))


def load_ship(rows_dir):
    """{regime: {run: (meta, (n, 6) Brier)}} of the 'ship' arm, every run of PLAN."""
    out = {}
    for regime, n in PLAN.items():
        got = SS.load_rows(rows_dir, regime)
        runs = {}
        for i in range(n):
            if i not in got or SS.SHIP not in got[i]["res"]:
                raise Mismatch(f"{regime}: run {i} has no '{SS.SHIP}' row in {rows_dir}")
            b = np.asarray(got[i]["res"][SS.SHIP]["b"], float)
            if b.shape != (len(got[i]["meta"]), len(BUDGETS)):
                raise Mismatch(f"{regime} run {i}: Brier rows {b.shape} against "
                               f"{len(got[i]['meta'])} pairs")
            runs[i] = (got[i]["meta"], b)
        extra = sorted(i for i in got if i >= n and SS.SHIP in got[i]["res"])
        if extra:
            raise Mismatch(f"{regime}: shipped rows beyond PLAN ({extra[:5]})")
        out[regime] = runs
    return out


class Arm:
    """One comparator's per pair rows on one regime: run -> (ALC (n,), Brier (n, 6)),
    and how each run was matched to the shipped rows."""

    def __init__(self, source):
        self.source = source
        self.runs = {}
        self.matched_by = {}

    def put(self, i, alc, b, how):
        self.runs[i] = (np.asarray(alc, float), None if b is None else np.asarray(b, float))
        self.matched_by[i] = how

    def coverage(self):
        ids = sorted(self.runs)
        if not ids:
            return None
        spans, lo, prev = [], ids[0], ids[0]
        for i in ids[1:] + [None]:
            if i is None or i != prev + 1:
                spans.append(f"{lo}-{prev}")
                if i is not None:
                    lo = i
            if i is not None:
                prev = i
        return {"runs": len(ids), "spans": spans,
                "matched_by": sorted(set(self.matched_by.values()))}


def lc_rows(lc_raw, regime, i, name):
    r = LC.rows_of(lc_raw, regime, i, name)
    if r is None:
        return None
    alc = np.asarray(r["alc"], float)
    b = np.asarray(r["pb"], float) if "pb" in r else None
    return alc, b


def lc_meta_matches(lc_raw, regime, i, meta):
    m = lc_raw.get(f"meta|{regime}|{i}")
    if m is None:
        return None
    return [meta_key(p) for p in m] == [meta_key(p) for p in meta]


def build_arms(ship, tc_raw, he_raw, lc_raw):
    """{regime: {comparator: Arm}} with every run matched pair for pair."""
    arms = {r: {c: None for c in COMPARATORS} for r in PLAN}
    # level_calibration's run metadata against the shipped rows, wherever it exists
    lc_match = {}
    for regime in PLAN:
        for i, (meta, _) in ship[regime].items():
            ok = lc_meta_matches(lc_raw, regime, i, meta)
            if ok is False:
                raise Mismatch(f"{regime} run {i}: level_calibration's pairs differ from the shipped rows'")
            if ok:
                lc_match[(regime, i)] = True

    # test-like primary: testlike_check (legacy, empirical), level_calibration (smoothed)
    leg, emp, smo = Arm("testlike_check.json raw['check|default|<i>'] rows[*]['pred']"), \
        Arm("testlike_check.json raw['check|default|<i>'] rows[*]['emp']"), \
        Arm("level_calibration.json raw['tl|<i>|base'] 'smoothed'")
    for i, (meta, _) in ship["tl"].items():
        v = tc_raw.get(f"check|default|{i}")
        if v is not None:
            if [tc_key(r) for r in v["rows"]] != [meta_key(p) for p in meta]:
                raise Mismatch(f"tl run {i}: testlike_check's pairs differ from the shipped rows'")
            pred = np.array([r["pred"] for r in v["rows"]], float)
            e = np.array([r["emp"] for r in v["rows"]], float)
            leg.put(i, pred[:, 6], pred[:, :6], "testlike_check pair ids")
            emp.put(i, e[:, 6], e[:, :6], "testlike_check pair ids")
        if lc_match.get(("tl", i)):
            s = lc_rows(lc_raw, "tl", i, LC.SMOOTHED)
            if s is not None:
                smo.put(i, *s, "level_calibration pair ids")
    arms["tl"].update(legacy=leg, smoothed=smo, empirical=emp)

    # sensitivity regimes: level_calibration's 'base' tasks
    for regime in ("tl mix/whole", "tl no shift"):
        leg = Arm(f"level_calibration.json raw['{regime}|<i>|base'] 'Predictor'")
        smo = Arm(f"level_calibration.json raw['{regime}|<i>|base'] 'smoothed'")
        for i in ship[regime]:
            if not lc_match.get((regime, i)):
                continue
            p, s = lc_rows(lc_raw, regime, i, LC.PRED), lc_rows(lc_raw, regime, i, LC.SMOOTHED)
            if p is not None:
                leg.put(i, *p, "level_calibration pair ids")
            if s is not None:
                smo.put(i, *s, "level_calibration pair ids")
        arms[regime].update(legacy=leg, smoothed=smo)

    # public runs: hier_eval (legacy, smoothed), matched through its Predictor rows;
    # testlike_check's public R1 runs (benchmark-first) carry ids and 'emp'
    for regime, w in HE_WEIGHTING.items():
        leg = Arm(f"hier_eval.json raw['r1|{w}|pair|0|<i>|Predictor'] rows")
        smo = Arm(f"hier_eval.json raw['r1|{w}|pair|0|<i>|smoothed Beta(2,2)'] rows")
        emp = Arm("testlike_check.json raw['r1|public R1|<i>'] rows[*]['emp']")
        for i, (meta, _) in ship[regime].items():
            hp = he_raw.get(f"r1|{w}|pair|0|{i}|Predictor")
            hs = he_raw.get(f"r1|{w}|pair|0|{i}|smoothed Beta(2,2)")
            how = None
            tcr = tc_raw.get(f"r1|public R1|{i}") if regime == "r1b" else None
            if tcr is not None:
                if [tc_key(r) for r in tcr["rows"]] != [meta_key(p) for p in meta]:
                    raise Mismatch(f"{regime} run {i}: testlike_check's public R1 pairs differ")
                e = np.array([r["emp"] for r in tcr["rows"]], float)
                emp.put(i, e[:, 6], e[:, :6], "testlike_check pair ids")
            if hp is not None:
                h = np.array(hp["rows"], float)
                if len(h) != len(meta):
                    raise Mismatch(f"{regime} run {i}: hier_eval holds {len(h)} pairs, the shipped rows {len(meta)}")
                if tcr is not None:
                    t = np.array([r["pred"] for r in tcr["rows"]], float)
                    if np.max(np.abs(h[:, :7] - t)) > 0:
                        raise Mismatch(f"{regime} run {i}: hier_eval's Predictor differs from testlike_check's")
                    how = "hier_eval Predictor = testlike_check Predictor (exact)"
                if lc_match.get((regime, i)):
                    lp = lc_rows(lc_raw, regime, i, LC.PRED)
                    if lp is not None:
                        d = max(np.max(np.abs(lp[0] - h[:, 6])),
                                np.max(np.abs(lp[1] - h[:, :6])) if lp[1] is not None else 0.0)
                        if d > TOL_PAIR:
                            raise Mismatch(f"{regime} run {i}: hier_eval's Predictor differs from "
                                           f"level_calibration's by {d:.2g}")
                        how = how or "hier_eval Predictor = level_calibration Predictor (1e-5)"
                if how:
                    leg.put(i, h[:, 6], h[:, :6], how)
            if hs is not None and how:
                s = np.array(hs["rows"], float)
                if len(s) != len(meta):
                    raise Mismatch(f"{regime} run {i}: hier_eval's smoothed rows hold {len(s)} pairs")
                if lc_match.get((regime, i)):
                    ls = lc_rows(lc_raw, regime, i, LC.SMOOTHED)
                    if ls is not None and max(np.max(np.abs(ls[0] - s[:, 6])),
                                              np.max(np.abs(ls[1] - s[:, :6])) if ls[1] is not None
                                              else 0.0) > TOL_PAIR:
                        raise Mismatch(f"{regime} run {i}: hier_eval's smoothed differs from level_calibration's")
                smo.put(i, s[:, 6], s[:, :6], "the same hier_eval run key as its Predictor rows: " + how)
        arms[regime].update(legacy=leg, smoothed=smo, empirical=emp if emp.runs else None)
    for regime in PLAN:
        for c in COMPARATORS:
            a = arms[regime][c]
            if a is not None and not a.runs:
                arms[regime][c] = None
    return arms


# --- statistics ---------------------------------------------------------------------------

class Block:
    """The shipped rows of one regime on a set of runs, flattened to pair
    appearances, with level_calibration's bootstrap over their clusters."""

    def __init__(self, ship_runs, ids):
        self.ids = list(ids)
        self.metas = [ship_runs[i][0] for i in self.ids]
        self.B = [ship_runs[i][1] for i in self.ids]
        self.alc = [b @ W6 for b in self.B]
        self.R = len(self.ids)
        self.boot = LC.Boot(self.metas, BOOTS)
        self.cl = np.concatenate(self.boot.idx)
        self.w = np.concatenate([np.full(len(m), 1.0 / len(m)) for m in self.metas])
        self.parent = np.array([p["parent"] for m in self.metas for p in m])
        self.K = len(self.boot.keys)

    def flat(self, per_run):
        return np.concatenate([np.asarray(x, float) for x in per_run])

    def se3(self, per_run):
        """(run SE, cluster SE, stratified SE) of the mean over runs of a run's
        pair mean."""
        run_means = np.array([float(np.mean(x)) for x in per_run])
        run_se = float(run_means.std(ddof=1) / math.sqrt(self.R)) if self.R > 1 else None
        c, s = self.boot.se(per_run)
        return run_se, c, s

    def sel_se(self, x, sel):
        """Cluster SE of the appearance-weighted mean of x over sel."""
        v = np.bincount(self.cl[sel], (self.w * x)[sel], minlength=self.K)
        n = np.bincount(self.cl[sel], self.w[sel], minlength=self.K)
        den = self.boot.W @ n
        ok = den > 0
        return float(((self.boot.W @ v)[ok] / den[ok]).std(ddof=1))

    def per_parent(self, per_run):
        x = self.flat(per_run)
        lopo = LC.lopo(self.metas, per_run)
        out = {}
        for q in sorted(set(self.parent.tolist())):
            m = self.parent == q
            k = len(set(self.cl[m].tolist()))
            # one cluster (swe_rebench's single subject) has no between-cluster spread to resample
            out[q] = {"mean": r6(np.sum(self.w[m] * x[m]) / np.sum(self.w[m])),
                      "cluster_se": r6(self.sel_se(x, m)) if k > 1 else None, "clusters": k,
                      "appearances": int(m.sum()),
                      "runs_with": int(sum(any(p["parent"] == q for p in mm) for mm in self.metas)),
                      "leave_out": r6(lopo[q])}
        multi = [out[q]["mean"] for q in PARENTS if q in out]
        lo = [out[q]["leave_out"] for q in out]
        level = {"parents": [q for q in PARENTS if q in out],
                 "mean": r6(np.mean(multi)) if multi else None,
                 "se": r6(np.std(multi, ddof=1) / math.sqrt(len(multi))) if len(multi) > 1 else None,
                 "range": [r6(min(multi)), r6(max(multi))] if multi else None,
                 "leave_one_out_range": [r6(min(lo)), r6(max(lo))]}
        return out, level

    def ship_summary(self):
        run_alc = np.array([float(np.mean(a)) for a in self.alc])
        run_b = np.array([b.mean(0) for b in self.B])
        rse, cse, sse = self.se3(self.alc)
        return {"runs": self.R, "range": rng_label(self.ids), "appearances": int(len(self.w)),
                "clusters": self.K,
                "ALC": r6(run_alc.mean()), "ALC_run_se": r6(rse), "ALC_cluster_se": r6(cse),
                "budgets": [r6(x) for x in run_b.mean(0)],
                "single_run_sd": {"budgets": [r6(x) for x in run_b.std(0, ddof=1)],
                                  "ALC": r6(run_alc.std(ddof=1))},
                "_run_alc": run_alc, "_run_b": run_b}

    def paired(self, arm):
        """Shipped minus the comparator on this block's runs."""
        o_alc = [arm.runs[i][0] for i in self.ids]
        o_b = [arm.runs[i][1] for i in self.ids]
        for i, a, b in zip(self.ids, o_alc, o_b):
            n = len(self.metas[self.ids.index(i)])
            if len(a) != n or (b is not None and b.shape != (n, len(BUDGETS))):
                raise Mismatch(f"run {i}: comparator holds {len(a)} pairs, the shipped rows {n}")
        d = [s - o for s, o in zip(self.alc, o_alc)]
        rse, cse, sse = self.se3(d)
        run_d = np.array([float(np.mean(x)) for x in d])
        per, level = self.per_parent(d)
        out = {"runs": self.R, "range": rng_label(self.ids), "appearances": int(len(self.w)),
               "shipped_ALC": r6(np.mean([np.mean(a) for a in self.alc])),
               "shipped_ALC_run_se": r6(self.se3(self.alc)[0]),
               "other_ALC": r6(np.mean([np.mean(a) for a in o_alc])),
               "other_ALC_run_se": r6(self.se3(o_alc)[0]),
               "diff": {"mean": r6(run_d.mean()), "run_se": r6(rse), "cluster_se": r6(cse),
                        "strat_se": r6(sse)},
               "runs_shipped_better": r6(np.mean(run_d < 0)),
               "per_parent": per, "parent_level": level}
        if all(b is not None for b in o_b):
            db = [s - o for s, o in zip(self.B, o_b)]
            rows = []
            for k, bud in enumerate(BUDGETS):
                col = [x[:, k] for x in db]
                r, c, s = self.se3(col)
                rows.append({"budget": bud, "shipped": r6(np.mean([np.mean(x[:, k]) for x in self.B])),
                             "other": r6(np.mean([np.mean(x[:, k]) for x in o_b])),
                             "diff": r6(np.mean([np.mean(x) for x in col])),
                             "run_se": r6(r), "cluster_se": r6(c), "strat_se": r6(s),
                             "alc_part": r6(W6[k] * np.mean([np.mean(x) for x in col]))})
            out["budgets"] = rows
        return out, d


def blocks_of(regime, arm=None):
    """Standard blocks of a regime's runs; with a comparator, those it covers in
    full plus its whole coverage when that is a shorter prefix."""
    n = PLAN[regime]
    std = [list(range(lo, min(lo + BLOCK, n))) for lo in range(0, n, BLOCK)]
    if n > BLOCK:
        std.append(list(range(n)))
    if arm is None:
        return std
    have = set(arm.runs)
    out = [ids for ids in std if set(ids) <= have]
    pre = [i for i in range(n) if i in have]
    if pre and pre == list(range(len(pre))) and all(pre != ids for ids in out):
        out.append(pre)
    return out


def missing(regime, arm):
    """The 100-run blocks of a regime for which a comparator has no stored rows
    on some run (so it is not compared there)."""
    std = blocks_of(regime)
    parts = std[:-1] if PLAN[regime] > BLOCK else std
    have = set(arm.runs) if arm is not None else set()
    return [rng_label(ids) for ids in parts if not set(ids) <= have]


# --- the study ----------------------------------------------------------------------------

def study(rows_dir):
    t0 = time.time()
    checks = Checks()
    inputs = {os.path.relpath(SS.rows_path(rows_dir, r), ROOT): SS.rows_path(rows_dir, r) for r in PLAN}
    for f in ("testlike_check.json", "hier_eval.json", "level_calibration.json",
              "subject_side.json", "mcq_floor.json", "hier_floor.json", "official_baselines.json"):
        inputs[f"results/{f}"] = os.path.join(RES, f)
    inputs["submission/model.py"] = os.path.join(ROOT, "submission", "model.py")
    digests = {k: sha256(v) for k, v in inputs.items()}

    ship = load_ship(rows_dir)
    tc = load_json(os.path.join(RES, "testlike_check.json"))
    lc = load_json(os.path.join(RES, "level_calibration.json"))
    he = load_json(os.path.join(RES, "hier_eval.json"))
    LC.register_chunks(LC.grid_chunks())
    tc_raw, lc_raw, he_raw = tc["raw"], lc["raw"], he["raw"]
    lcs = lc["summary"]
    arms = build_arms(ship, tc_raw, he_raw, lc_raw)

    out = {"about": {
        "shipped": {"config": SHIP, "level": SS.LEVEL,
                    "rows": "data/subject_side_rows/<regime>.jsonl, arm 'ship'"},
        "comparators": {"legacy": "paiec.predict.Predictor, attribute prior fitted without the "
                                  "target's parent (the first submission's predictor)",
                        "smoothed": "Beta(2,2) on the pair's own labels (baselines.smoothed_mean(4, 0.5))",
                        "empirical": "the organisers' empirical mean (baselines.empirical_mean)"},
        "difference": "shipped minus comparator, ALC (lower is better)",
        "se": "run SE / cluster SE ((parent, subject), 2,000 resamples, seed 0) / stratified by parent",
        "parent_level": "mean and SE of the four multi-subject parents' appearance-weighted means",
        "budgets": list(BUDGETS), "weights": list(WEIGHTS)}}
    out["sources"] = {r: {c: (None if arms[r][c] is None else
                              {"source": arms[r][c].source, **arms[r][c].coverage()})
                          for c in COMPARATORS} for r in PLAN}
    for r in PLAN:
        out["sources"][r]["shipped"] = {"runs": PLAN[r], "file": os.path.relpath(SS.rows_path(rows_dir, r), ROOT)}
    out["not_stored"] = {r: {c: missing(r, arms[r][c]) for c in COMPARATORS if missing(r, arms[r][c])}
                         for r in PLAN}

    # --- per regime ---------------------------------------------------------------------
    regimes, diffs = {}, {}
    for regime in PLAN:
        rs = ship[regime]
        rep = {"label": LABELS[regime], "runs": PLAN[regime],
               "appearances": int(sum(len(m) for m, _ in rs.values())), "shipped": {}, "vs": {}}
        for ids in blocks_of(regime):
            rep["shipped"][rng_label(ids)] = Block(rs, ids).ship_summary()
        for c in COMPARATORS:
            arm = arms[regime][c]
            if arm is None:
                continue
            rep["vs"][c] = {}
            for ids in blocks_of(regime, arm):
                blk = Block(rs, ids)
                res, d = blk.paired(arm)
                rep["vs"][c][rng_label(ids)] = res
                diffs[(regime, c, rng_label(ids))] = (blk, d)
        regimes[regime] = rep
    out["regimes"] = regimes

    # --- consistency checks -----------------------------------------------------------------
    # 1. level_calibration's selection half for the shipped config (runs 0-99)
    sel = lcs["selection"]["configs"]
    got = regimes["tl"]["vs"]["legacy"]["0-99"]
    src = "level_calibration.json summary.selection.configs"
    checks.close("selection: shipped ALC", got["shipped_ALC"], sel[SHIP]["ALC"][0], TOL_MEAN, src + f"['{SHIP}'].ALC[0]")
    checks.close("selection: legacy ALC", got["other_ALC"], sel[LC.PRED]["ALC"][0], TOL_MEAN, src + "['Predictor'].ALC[0]")
    for k in ("mean", "run_se", "cluster_se", "strat_se"):
        checks.close(f"selection: difference {k}", got["diff"][k], sel[SHIP]["diff"][k], TOL_MEAN,
                     src + f"['{SHIP}'].diff.{k}")
    for q, v in sel[SHIP]["diff"]["lopo"].items():
        checks.close(f"selection: leave out {q}", got["per_parent"][q]["leave_out"], v, TOL_MEAN,
                     src + f"['{SHIP}'].diff.lopo.{q}")
    for want, x in (("0.1608", got["shipped_ALC"]), ("0.2039", got["other_ALC"]),
                    ("-0.0431", got["diff"]["mean"]), ("0.0019", got["diff"]["run_se"])):
        checks.shown(f"selection as printed: {want}", x, want, "review section 7")
    sm = regimes["tl"]["vs"]["smoothed"]["0-99"]
    checks.close("selection: smoothed ALC", sm["other_ALC"], sel[LC.SMOOTHED]["ALC"][0], TOL_MEAN,
                 src + "['smoothed'].ALC[0]")
    # 2. the guard's Predictor (and smoothed) ALCs on public runs 0-99
    for regime, want in (("r1b", "0.2068"), ("r1p", "0.2038")):
        g = regimes[regime]["vs"]["legacy"]["0-99"]
        s = regimes[regime]["vs"]["smoothed"]["0-99"]
        base = f"level_calibration.json summary.r1.{regime}"
        checks.close(f"guard {regime}: Predictor ALC", g["other_ALC"], lcs["r1"][regime][LC.PRED]["ALC"][0],
                     TOL_MEAN, base + ".Predictor.ALC[0]")
        checks.shown(f"guard {regime}: Predictor ALC as printed", g["other_ALC"], want, "review section 7")
        checks.close(f"guard {regime}: smoothed ALC", s["other_ALC"], lcs["r1"][regime][LC.SMOOTHED]["ALC"][0],
                     TOL_MEAN, base + ".smoothed.ALC[0]")
    # 3. the sensitivity regimes, matched pair for pair
    for regime in ("tl mix/whole", "tl no shift"):
        n = sum(bool(lc_meta_matches(lc_raw, regime, i, ship[regime][i][0])) for i in range(100))
        checks.add(f"sensitivity {regime}: runs matched pair for pair", n == 100, n, 100,
                   f"level_calibration.json raw['meta|{regime}|<i>'] against the shipped rows")
        base = f"level_calibration.json summary.sensitivity['{regime}']"
        for c, name in (("legacy", LC.PRED), ("smoothed", LC.SMOOTHED)):
            checks.close(f"sensitivity {regime}: {name} ALC", regimes[regime]["vs"][c]["0-99"]["other_ALC"],
                         lcs["sensitivity"][regime][name]["ALC"][0], TOL_MEAN, base + f"['{name}'].ALC[0]")
    # 4. the shipped rows against every other record of them
    ssum = load_json(os.path.join(RES, "subject_side.json"))["summary"]
    checks.add("shipped LEVEL = submission/model.py's", ssum["level"] == SS.LEVEL, ssum["level"], SS.LEVEL,
               "subject_side.json summary.level; submission/model.py LEVEL")
    for regime in PLAN:
        a = regimes[regime]["shipped"][rng_label(list(range(PLAN[regime])))]
        v = ssum["regimes"][regime]["ship"]
        checks.close(f"subject_side summary {regime}: shipped ALC", a["ALC"], v["ALC"], 1e-6,
                     f"subject_side.json summary.regimes['{regime}'].ship.ALC")
        checks.close(f"subject_side summary {regime}: shipped ALC run SE", a["ALC_run_se"], v["ALC_run_se"],
                     1e-6, f"subject_side.json summary.regimes['{regime}'].ship.ALC_run_se")
    for regime, ids in (("tl", range(100)), ("r1b", range(PLAN["r1b"]))):
        worst, n = 0.0, 0
        for i in ids:
            g = lc_rows(lc_raw, regime, i, SHIP)
            if g is None:
                continue
            if not lc_meta_matches(lc_raw, regime, i, ship[regime][i][0]):
                raise Mismatch(f"{regime} run {i}: level_calibration's shipped-config run has other pairs")
            worst = max(worst, float(np.max(np.abs(g[0] - ship[regime][i][1] @ W6))))
            n += 1
        checks.add(f"level_calibration's own rows of {SHIP} on {regime} ({n} runs), per pair ALC",
                   n > 0 and worst <= TOL_PAIR, r6(worst), f"<= {TOL_PAIR:.1e}",
                   f"level_calibration.json raw (grid/screen tasks holding '{SHIP}')")
    mq = load_json(os.path.join(RES, "mcq_floor.json"))
    tl200 = Block(ship["tl"], range(200)).ship_summary()
    checks.close("mcq_floor old-floor arm, test-like runs 0-199", tl200["ALC"],
                 mq["regimes"]["tl"]["alc"]["ship"]["mean"], 1e-6,
                 "mcq_floor.json regimes.tl.alc.ship.mean (provenance.library_floor 'old')")
    ob = {r["run"]: r["alc"] for r in load_json(os.path.join(RES, "official_baselines.json"))["r1"]["per_run"]}
    for c, name in (("smoothed", "smoothed Beta(2,2)"), ("empirical", "empirical mean")):
        arm = arms["r1b"][c]
        d = [abs(float(np.mean(arm.runs[i][0])) - ob[i][name]) for i in sorted(arm.runs) if i in ob]
        checks.add(f"official_baselines per-run {name}, r1b ({len(d)} runs)", bool(d) and max(d) <= 2e-6,
                   r6(max(d)) if d else None, "<= 2e-6", f"official_baselines.json r1.per_run[*].alc['{name}']")
    # 5. the review's [R] numbers as printed
    for regime, runs, s_alc, o_alc, dmean, dse in REVIEW_W2:
        g = regimes[regime]["vs"]["legacy"].get(runs)
        if g is None:
            checks.add(f"review W2 {regime} {runs}", False, None, "block present", "review W2")
            continue
        checks.shown(f"review W2 {regime} {runs}: shipped ALC", g["shipped_ALC"], s_alc, "review W2")
        checks.shown(f"review W2 {regime} {runs}: legacy ALC", g["other_ALC"], o_alc, "review W2")
        checks.shown(f"review W2 {regime} {runs}: difference", g["diff"]["mean"], dmean, "review W2")
        checks.shown(f"review W2 {regime} {runs}: run SE", g["diff"]["run_se"], dse, "review W2")

    # --- the section-7 table and run 2's place ---------------------------------------------
    sd_table, run2 = {}, {}
    for regime in PLAN:
        a = regimes[regime]["shipped"][rng_label(list(range(PLAN[regime])))]
        sds = a["single_run_sd"]["budgets"] + [a["single_run_sd"]["ALC"]]
        sd_table[regime] = {"runs": a["runs"], **{f"B{b}": s for b, s in zip(BUDGETS, sds[:6])}, "ALC": sds[6]}
        for j, want in enumerate(REVIEW_SD[regime].split()):
            checks.shown(f"review section 7 sd {regime} {'ALC' if j == 6 else f'B{BUDGETS[j]}'}",
                         sds[j], want, "review section 7")
        run_alc, run_b = a["_run_alc"], a["_run_b"]
        mean_b, sd_b = run_b.mean(0), run_b.std(0, ddof=1)
        z_b = (np.array(RUN2["budgets"]) - mean_b) / sd_b
        z = (RUN2["ALC"] - run_alc.mean()) / run_alc.std(ddof=1)
        run2[regime] = {"runs": a["runs"], "mean_ALC": r6(run_alc.mean()), "sd_ALC": r6(run_alc.std(ddof=1)),
                        "z_ALC": r6(z), "share_runs_at_or_above": r6(np.mean(run_alc >= RUN2["ALC"])),
                        "z_budgets": [r6(x) for x in z_b],
                        "mean_budgets": [r6(x) for x in mean_b]}
        if regime in REVIEW_W1:
            m, s, zz, share, zb = REVIEW_W1[regime]
            checks.shown(f"review W1 {regime}: mean ALC", run_alc.mean(), m, "review W1")
            checks.shown(f"review W1 {regime}: sd", run_alc.std(ddof=1), s, "review W1")
            checks.shown(f"review W1 {regime}: run 2's z", z, zz, "review W1")
            checks.shown(f"review W1 {regime}: share >= run 2", float(np.mean(run_alc >= RUN2["ALC"])), share,
                         "review W1")
            for b, x, want in zip(BUDGETS, z_b, zb.split()):
                checks.shown(f"review W1 {regime}: run 2's z at B{b}", x, want, "review W1")
    for regime in PLAN:
        for blk in regimes[regime]["shipped"].values():
            blk.pop("_run_alc"), blk.pop("_run_b")
    out["single_run_sd"] = sd_table
    out["run2_placement"] = {"run2": RUN2, "regimes": run2,
                             "note": "run 2 is one draw of 8 pairs; a z is its distance from a regime's run "
                                     "mean in that regime's single-run sds, not a test"}

    # --- the table the report quotes ------------------------------------------------------
    table = []
    for (regime, c, runs), (blk, _) in diffs.items():
        g = regimes[regime]["vs"][c][runs]
        table.append({"regime": regime, "comparator": c, "runs": runs, "n_runs": g["runs"],
                      "shipped_ALC": g["shipped_ALC"], "other_ALC": g["other_ALC"],
                      "diff": g["diff"]["mean"], "run_se": g["diff"]["run_se"],
                      "cluster_se": g["diff"]["cluster_se"], "strat_se": g["diff"]["strat_se"],
                      "parent_mean": g["parent_level"]["mean"], "parent_se": g["parent_level"]["se"],
                      "parent_range": g["parent_level"]["range"],
                      "path": f"regimes['{regime}'].vs.{c}['{runs}']"})
    out["table"] = table

    # --- which code the rows are, and what the current archive changed ----------------------
    sside = load_json(os.path.join(RES, "subject_side.json"))
    run_pass = next(p for p in sside["passes"] if p.get("stage") == "run" and SS.SHIP in p.get("configs", []))
    hf = load_json(os.path.join(RES, "hier_floor.json"))
    gap = {"rows_code": {"head": run_pass.get("head"), "paiec_dirty": run_pass.get("paiec_dirty"),
                         "digests": run_pass.get("digests"),
                         "source": "subject_side.json passes[stage 'run', configs holding 'ship']"},
           "mcq_floor_fix_minus_old": {}, "floored_fit_fix": {}}
    for regime in ("tl", "r1b", "r1p"):
        v = mq["regimes"][regime]
        gap["mcq_floor_fix_minus_old"][regime] = {
            "runs": v["runs"], **v["compare"]["fix - ship"]["all"]["per_appearance"],
            "path": f"mcq_floor.json regimes.{regime}.compare['fix - ship'].all.per_appearance"}
        h = hf["replay"][f"{regime}_lib_ship"]
        gap["floored_fit_fix"][regime] = {
            "runs": h["runs"], "alc_diff": h["alc_diff"], "alc_diff_run_se": h["alc_diff_run_se"],
            "path": f"hier_floor.json replay.{regime}_lib_ship"}
    gap["library_digests"] = git_digests(["paiec/hier.py", "paiec/mcq.py", "paiec/prior.py", "paiec/subjects.py"],
                                          ["ee5085a", "4d2cc4f", "HEAD"])
    gap["note"] = ("the rows are hier.py " + str(run_pass.get("digests", {}).get("paiec/hier.py")) +
                   " with the old multiple-choice floor; the archive built at 4d2cc4f adds the corrected "
                   "floor then the floored-fit fix, measured on runs 0-199 by the two files above "
                   "(corrected floor at the pre-fix solver, then the fix at the corrected floor); "
                   "nothing here adds them in")
    out["code_gap"] = gap

    failed = checks.failed()
    out["checks"] = {"passed": len(checks.items) - len(failed), "failed": len(failed), "items": checks.items}
    out["provenance"] = provenance(digests, time.time() - t0)
    return out, failed


def git_digests(files, revs):
    out = {}
    for rev in revs:
        for f in files:
            try:
                blob = subprocess.run(["git", "show", f"{rev}:{f}"], cwd=ROOT, capture_output=True,
                                      check=True).stdout
                out[f"{rev}:{f}"] = hashlib.sha256(blob).hexdigest()[:16]
            except Exception:
                out[f"{rev}:{f}"] = None
    return out


def provenance(digests, secs):
    me = os.path.abspath(__file__)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", os.path.relpath(me, ROOT)],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head, dirty = None, None
    code = {os.path.relpath(p, ROOT): sha256(p) for p in
            (me, os.path.join(ROOT, "experiments", "level_calibration.py"),
             os.path.join(ROOT, "experiments", "subject_side.py"))}
    return {"command": " ".join([os.path.basename(sys.executable)] + [os.path.relpath(sys.argv[0], ROOT)]
                                + sys.argv[1:]),
            "script": os.path.relpath(me, ROOT), "script_sha256": code[os.path.relpath(me, ROOT)],
            "code_sha256": code, "head": head, "script_status": dirty or "clean",
            "inputs_sha256": digests, "python": platform.python_version(), "numpy": np.__version__,
            "boots": BOOTS, "wall_s": round(secs, 1)}


# --- printing -----------------------------------------------------------------------------

def show(out):
    f4 = "{:+.4f}".format
    print(f"{'regime':<14}{'vs':<10}{'runs':<9}{'shipped':>8}{'other':>8}{'diff':>9}  "
          f"{'run/cluster/strat SE':<22}{'parent mean ± SE':<20}parent range")
    for t in out["table"]:
        pr = t["parent_range"]
        print(f"{t['regime']:<14}{t['comparator']:<10}{t['runs']:<9}{t['shipped_ALC']:>8.4f}"
              f"{t['other_ALC']:>8.4f}{t['diff']:>+9.4f}  "
              f"{t['run_se']:.4f}/{t['cluster_se']:.4f}/{t['strat_se']:.4f}    "
              f"{f4(t['parent_mean'])} ± {t['parent_se']:.4f}   {f4(pr[0])} to {f4(pr[1])}")
    print("\nshipped model, single-run sd (B0 B1 B3 B7 B15 B31 | ALC)")
    for r, v in out["single_run_sd"].items():
        print(f"  {r:<14}{v['runs']:>4}  " + " ".join(f"{v[f'B{b}']:.4f}" for b in BUDGETS) + f" | {v['ALC']:.4f}")
    print(f"\nformative run 2 (ALC {out['run2_placement']['run2']['ALC']}) among each regime's runs")
    for r, v in out["run2_placement"]["regimes"].items():
        print(f"  {r:<14}mean {v['mean_ALC']:.4f} sd {v['sd_ALC']:.4f} z {v['z_ALC']:+.2f} "
              f"share>= {v['share_runs_at_or_above']:.2f}  z by budget "
              + " ".join(f"{x:+.2f}" for x in v["z_budgets"]))
    c = out["checks"]
    print(f"\nchecks: {c['passed']} passed, {c['failed']} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=SS.ROWS, help="experiments/subject_side.py's rows directory")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out, failed = study(args.rows)
    show(out)
    if failed:
        for c in failed:
            print(f"FAILED {c['check']}: got {c['got']}, want {c['want']} ({c['source']})", file=sys.stderr)
        raise SystemExit(f"{len(failed)} consistency check(s) failed; {args.out} not written")
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)
    print(f"wrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
