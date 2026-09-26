"""Test-like runs (paiec/testlike.py) against the first formative feedback.

The feedback (the shipped Predictor, commit b68492c) is one run of 9 pairs on
the hidden test: ALC 0.2113, budget means B0 0.3589, B1 0.2539, B3 0.2043, B7
0.1712, B15 0.1693, B31 0.1568 (testlike.FEEDBACK holds the per-pair rows).
This script tunes the regime's knobs on it and checks the rest against it.

  tune   a grid over level_mean (-1.2, -1.6, -2.0; level_sd 1.5) and
         date_shift (0, 0.75, 1, 1.25, 1.5 years), `--tune-runs` runs each
         (seed 1). A setting's distance to the feedback is the Predictor's B0
         and B1 run means minus the feedback, each over the replica's
         single-run sd, in quadrature; a second distance adds B31, the
         feedback's own evidence on levels, and a third uses only B3, B7 and
         B15, which tuning never looks at. Regime's defaults are set by hand to
         the closest setting, and the summary says whether they match.
  check  Regime() on `--runs` fresh runs (seed 2): regime statistics against
         the targets, the Predictor's per-budget Brier, ALC, B0 ECE and B0
         prediction, the organisers' empirical mean against their leaderboard
         entry (0.1801, a different single run whose identity is not stated),
         pair-level quantiles against the 9 feedback pairs, and the feedback's
         level reading (lower root of p(1-p) = B31) applied like for like to
         the replica's own B31. Sensitivities (`--sens-runs`, seed 3, SENS):
         the level and tilt knobs; item structure (no strata; groups merged
         at random, no strata); what makes the prior optimistic (dates capped
         at 2026-12-31; the equivalent offset on the prior's ability with real
         dates instead of the shift, and a date-blind prior next to the legacy
         one under both, to see whether predictor rankings survive the
         choice); run structure (one pseudo-benchmark per parent; run
         subjects hidden from the prior; split after the cut with an 88-item
         floor). The item structure of four regimes is measured separately
         (testlike.item_oracle, `structure`).
  r1     the same numbers on plain public formative runs, official.sample_run
         at its defaults (seed 0: the runs of experiments/official_baselines.py,
         whose per-run ALCs for both predictors are reproduced and checked).

What the check can and cannot say. B0 and B1 agree with the feedback because
the default was tuned to them: that is in sample, not a prediction (default
Predictor B0 0.3513, feedback z +0.19; B1 0.2656, z -0.35). ALC tells the
regimes nothing: Predictor ALC is 0.2073 here and 0.2075 on public R1, against
0.2113 (both at z +0.16). Only the budget profile separates them: R1 puts the
feedback's B0 at z +8.5 and its B7 and B31 near z -1; here B3 to B31 sit at z
+0.09 to +0.50 without being tuned (distance at B3/B7/B15 0.52 run sds). The
feedback is one run: the replica's single-run sd of ALC is 0.025, of B0 0.041.
The level knob is not identified (see testlike's docstring), and the B0/B1
match rests on date_shift, which is equivalent to an offset on the prior's
ability for the legacy Predictor only: a date-blind prior beats the legacy one
by 0.033 ALC (SE 0.002) under the shift and by 0.002 (SE 0.001) under the
offset, and the order of the three predictors changes at five of seven budget
columns (ranking_agreement). Pseudo-benchmarks carry less item structure than
public benchmarks (item-oracle gain 41% against 55% on R1; 53% with groups
merged at random and no strata), so item-aware against level-only choices must
also hold on that variant and on R1.

The Predictor is the shipped one, built the way submission/model.py builds it
(paiec.predict.Predictor on fit_prior's coefficients) but with its attribute
prior fitted on every eligible public pair outside the target pair's parent
benchmark: one model per parent, dispatched on the anonymous benchmark_id. The
empirical mean is the organisers' baseline (paiec.baselines.empirical_mean).
Both run under paiec.official at its defaults (split scope 'pair', the
platform's random acquisition), fast path (no argument copies, one worker).
Variant is the same Predictor with an offset on its prior mean or its release
date read as the training median (date-blind).

Statistics. The feedback is a single run, so what judges a match is the
replica's single-run sd: the feedback's z-score and percentile among runs.
Means also carry the SE over runs and a cluster bootstrap SE (2,000 resamples,
a ratio estimator) with (parent, subject) as the cluster, because the
catalogue is finite, the tilt concentrates on the pseudo-benchmarks near the
target, and pseudo-benchmarks of one parent overlap. Leave-one-parent-out
ranges sit next to the means: four parents carry every number.

Tasks are pure functions of their key and are checkpointed to --out as they
finish (status 'partial'); --resume skips finished ones, --validate redraws
every stored run and reruns those that no longer draw the same, --summarise
rebuilds the summary from the file.

Run: python experiments/testlike_check.py --phase tune --jobs 5
     (set Regime's defaults to the closest setting), then
     python experiments/testlike_check.py --phase check --resume --validate --jobs 5
(3,400 tasks, about 6 CPU hours in all; the last 814, the sensitivities added after
review and the 14 runs the redraw-below-5 rule changed, took 11 minutes on
five processes with other jobs holding the load average near 100)
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import timedelta  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec import baselines as B  # noqa: E402
from paiec import data as D  # noqa: E402
from paiec import official as O  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS  # noqa: E402
from paiec.predict import Predictor, fit_prior  # noqa: E402
from paiec.subjects import EPOCH  # noqa: E402

OUT = os.path.join(ROOT, "results", "testlike_check.json")
BASELINES = os.path.join(ROOT, "results", "official_baselines.json")
BOOTS = 2000
FB = np.array([r[2] for r in T.FEEDBACK])                  # 9 pairs x 6 budgets
FB_BUDGET = FB.mean(0)
FB_ALC = T.FEEDBACK_ALC
ORGANISERS = 0.1801
GRID_MEAN = (-1.2, -1.6, -2.0)
GRID_SHIFT = (0.0, 0.75, 1.0, 1.25, 1.5)
LABELS = list(map(str, BUDGETS)) + ["ALC"]
#: Keys starting with 'predictor_' configure the predictors, the rest the Regime.
SENS = {
    "no date shift": {"date_shift": 0.0},
    "tilt on the pair's logit": {"tilt": "pair"},
    "level_sd 1.0": {"level_sd": 1.0},
    "level_sd 2.0": {"level_sd": 2.0},
    "one pair per benchmark": {"repeat": 0.0},
    "subjects from 2025-07-01": {"min_release": "2025-07-01"},
    # item structure
    "no strata": {"kinds": ["group", "whole"]},
    "no strata, groups merged at random": {"kinds": ["mix", "whole"]},
    # what makes the prior optimistic
    "dates capped at 2026-12-31": {"date_cap": "2026-12-31"},
    "date-blind predictor added": {"predictor_date_blind": True},
    "ability offset instead of date shift": {"date_shift": 0.0, "predictor_offset_years": 1.25,
                                             "predictor_date_blind": True},
    # run structure
    "one pseudo-benchmark per parent": {"max_per_parent": 1},
    "run subjects hidden from the prior": {"predictor_hide_subjects": True},
    "split after the cut, 88-item floor": {"split_after_cut": True, "min_kept": 88,
                                           "n_pairs": [5, 11]},
}
#: the two variants whose three-predictor rankings are compared
RANKING = ("date-blind predictor added", "ability offset instead of date shift")
#: regimes whose item structure is measured, (kind, name)
STRUCTURE = [("check", "default"), ("sens", "no strata"),
             ("sens", "no strata, groups merged at random"), ("r1", "public R1")]
SEEDS = {"tune": 1, "check": 2, "sens": 3, "r1": 0}

_pairs = _cat = _release = None
_samplers, _priors = {}, {}


# --- per-worker state -----------------------------------------------------------------

def real_release(subject_id):
    """A subject's recorded release date, before any shift."""
    global _release
    if _release is None:
        _release = {p.subject_id: str(p.subject.get("release_date") or "") for p in pairs()}
    return _release.get(subject_id, "")


def pairs():
    global _pairs
    if _pairs is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _pairs = O.eligible(D.load_pairs())
    return _pairs


def catalogue():
    global _cat
    if _cat is None:
        _cat = T.build_catalogue(pairs())
    return _cat


def split_overrides(overrides):
    """(Regime keyword arguments, predictor options) of a task's overrides."""
    reg, opt = {}, {}
    for k, v in overrides.items():
        if k.startswith("predictor_"):
            opt[k[len("predictor_"):]] = v
        else:
            reg[k] = tuple(map(tuple, v)) if k == "kind_weights" else \
                tuple(v) if k in ("n_pairs", "exclude", "kinds") else v
    return reg, opt


def sampler(overrides):
    reg, _ = split_overrides(overrides)
    key = json.dumps(reg, sort_keys=True)
    if key not in _samplers:
        _samplers[key] = T.Sampler(catalogue(), T.Regime().with_(**reg))
    return _samplers[key]


def prior(parent, run=None):
    """The attribute prior on every eligible public pair outside `parent`; with a
    run, also without any of the run's subjects (by id or normalized name)."""
    if run is None:
        key = parent
    else:
        key = (parent, tuple(sorted({p.subject_id for p, _ in run})))
    if key not in _priors:
        if run is None:
            train = [p for p in pairs() if p.benchmark_id != parent]
        else:
            train = T.training_pairs(pairs(), run, parents={parent}, hide_subjects=True)
        if len(_priors) > 64:
            _priors.clear()
        _priors[key] = fit_prior(train) if train else (None, None)
    return _priors[key]


class Variant(Predictor):
    """The shipped Predictor with its attribute prior changed in one of two ways:
    `offset` added to the prior's mean ability (logit), and with `blind` the
    release date read as the training median whatever the subject's, so that
    the prior ignores dates and keeps every other attribute."""

    def __init__(self, coef, spec, offset=0.0, blind=False):
        super().__init__(coef, spec)
        self.offset = float(offset)
        self.median_date = None
        if blind and spec is not None:
            self.median_date = (EPOCH + timedelta(days=round(spec.med_days))).isoformat()

    def prior_mean(self, subject):
        if self.median_date is not None:
            subject = dict(subject, release_date=self.median_date)
        return super().prior_mean(subject) + self.offset


def date_offset(coef, years):
    """What moving a dated subject's release date by `years` adds to the legacy
    prior's mean: the design's date column is (days - 400) / 400
    (paiec.subjects.design_row), with no clipping."""
    if coef is None or not years:
        return 0.0
    return float(coef[1]) * round(years * 365.25) / 400


def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def predictor_factory(run, b0=None, *, offset_years=0.0, blind=False, hide=False):
    """The shipped Predictor, one per parent benchmark, dispatched on the
    anonymous benchmark_id, its prior fitted without the parent (and with
    `hide` without the run's subjects); with an offset or `blind`, the Variant.
    Predictions with no labels are logged in b0 per (benchmark_id, subject) to
    read the prior's B0 prediction back."""
    ids = T.anon_parents(run)
    priors = {par: prior(par, run if hide else None) for par in set(ids.values())}

    def model(par):
        coef, spec = priors[par]
        if not offset_years and not blind:
            return Predictor(coef, spec)
        return Variant(coef, spec, date_offset(coef, offset_years), blind)

    def factory():
        models = {par: model(par) for par in priors}

        def predict(input, labeled=None):
            p = models[ids[input[1]["benchmark_id"]]].predict(input, labeled)
            if b0 is not None and not labeled:
                b0.setdefault((input[1]["benchmark_id"], _canon(input[0])), []).append(p)
            return p
        return predict
    return factory


def regime_of(kind, name, cfg):
    """The Regime a stored run was drawn under (None for public R1)."""
    ov = overrides_of(kind, name, cfg)
    if kind == "r1" or ov is None:
        return None
    return T.Regime().with_(**split_overrides(ov)[0])


def overrides_of(kind, name, cfg):
    if kind == "tune":
        return grid(cfg).get(name)
    if kind == "sens":
        return SENS.get(name)
    return {}


def draw(kind, overrides, i):
    rng = np.random.default_rng([SEEDS[kind], i])
    if kind == "r1":
        return O.sample_run(pairs(), rng)
    return sampler(overrides).run(rng)


def task(t):
    kind, name, overrides, i = t
    t0 = time.perf_counter()
    _, opt = split_overrides(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = draw(kind, overrides, i)
        desc = T.describe(run)
        b0 = {}
        kw = {"offset_years": opt.get("offset_years", 0.0), "hide": opt.get("hide_subjects", False)}
        res = O.run_official(run, predictor_factory(run, b0, **kw), deepcopy=False)
        emp = O.run_official(run, lambda: B.empirical_mean, deepcopy=False)
        blind = O.run_official(run, predictor_factory(run, blind=True, **kw), deepcopy=False) \
            if opt.get("date_blind") else None
    rows = []
    for j, ((p, _), r, e, d) in enumerate(zip(run, res["rows"], emp["rows"], desc)):
        q = b0.get((r["benchmark_id"], _canon(D.official_subject(p.subject))), [np.nan])
        row = {
            "pseudo": d["pseudo"], "parent": d["parent"], "subject": d["subject_id"],
            "release": real_release(d["subject_id"]), "items": d["items"],
            "eval": d["eval_items"],
            "p": round(d["p"], 5), "pa": round(d["pa"], 5), "logit": round(d["logit"], 4),
            "pred": [round(r["brier"][b], 6) for b in BUDGETS] + [round(r["ALC"], 6)],
            "ece0": round(r["ece"][0], 5), "ece_alc": round(r["ece_alc"], 5),
            "q": round(float(np.mean(q)), 5),
            "emp": [round(e["brier"][b], 6) for b in BUDGETS] + [round(e["ALC"], 6)]}
        if blind is not None:
            x = blind["rows"][j]
            row["blind"] = [round(x["brier"][b], 6) for b in BUDGETS] + [round(x["ALC"], 6)]
        rows.append(row)
    return task_key(t), {"rows": rows, "task_s": round(time.perf_counter() - t0, 3),
                         "eval_ms": round(1e3 * res["timing"]["evaluation_mean_s"], 3)}


def task_key(t):
    kind, name, _, i = t
    return f"{kind}|{name}|{i}"


def signature(rows):
    return [(r["pseudo"], r["subject"], r["eval"], r["p"]) for r in rows]


def stale(raw, cfg):
    """Keys of raw whose run no longer draws the same (the draw code or the
    overrides changed since it ran), found by redrawing every run."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, v in raw.items():
            kind, name, i = key.split("|")
            ov = overrides_of(kind, name, cfg)
            if ov is None:
                out.append(key)
                continue
            desc = T.describe(draw(kind, ov, int(i)))
            now = [(d["pseudo"], d["subject_id"], d["eval_items"], round(d["p"], 5)) for d in desc]
            if now != signature(v["rows"]):
                out.append(key)
    return out


# --- statistics ------------------------------------------------------------------

def mean_se(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0]


def cluster(r):
    return T.cluster_key(r["pseudo"], r["subject"])


def cluster_se(runs, values, boots=BOOTS, seed=0):
    """SE of the mean over runs of each run's pair-average of `values`, with
    (parent, subject) resampled as clusters: pseudo-benchmarks of one parent
    overlap, so a subject's pairs on them are not independent. A ratio
    estimator: a resampled cluster brings its weight (appearances / run size)
    with it, so how often a cluster recurs does not count as variance of the
    mean itself."""
    keys = sorted({cluster(r) for run in runs for r in run})
    at = {k: j for j, k in enumerate(keys)}
    num, den = np.zeros(len(keys)), np.zeros(len(keys))
    for run, vals in zip(runs, values):
        for r, x in zip(run, vals):
            num[at[cluster(r)]] += x / len(run)
            den[at[cluster(r)]] += 1 / len(run)
    W = np.random.default_rng(seed).multinomial(len(keys), np.full(len(keys), 1 / len(keys)),
                                                size=boots)
    return float(((W @ num) / (W @ den)).std(ddof=1))


def lopo(runs, values):
    """The mean over runs of each run's pair-average of `values` with one parent's
    pairs left out, per parent (runs left empty are skipped): how much the
    number rests on which parents are in the catalogue."""
    parents = sorted({r["parent"] for run in runs for r in run})
    out = {}
    for q in parents:
        m = [np.mean([x for r, x in zip(run, vals) if r["parent"] != q])
             for run, vals in zip(runs, values) if any(r["parent"] != q for r in run)]
        out[q] = float(np.mean(m))
    return {"min": min(out.values()), "max": max(out.values()), "by_parent": out}


def against(x, target):
    """The target's z-score and percentile among per-run values x."""
    x = np.asarray(x, float)
    sd = float(x.std(ddof=1))
    return {"mean": float(x.mean()), "sd": sd, "target": float(target),
            "z": float((target - x.mean()) / sd) if sd > 0 else None,
            "percentile": float(np.mean(x <= target))}


def other_parents():
    """Where each subject has public records: (subject_id -> benchmarks,
    normalized name -> benchmarks, subject_id -> normalized name); empty names
    are left out of the name map."""
    by_id, by_name, name = defaultdict(set), defaultdict(set), {}
    for p in pairs():
        n = str(p.subject.get("normalized_name") or "").strip().lower()
        by_id[p.subject_id].add(p.benchmark_id)
        name[p.subject_id] = n
        if n:
            by_name[n].add(p.benchmark_id)
    return by_id, by_name, name


def regime_stats(runs, record=None, regime=None):
    """What the runs look like, against the feedback's shape; `regime` gives the
    dates a predictor was shown (release dates shifted and capped)."""
    rows = [r for run in runs for r in run]
    L = np.array([r["logit"] for r in rows])
    ev = np.array([r["eval"] for r in rows])
    n = [len(run) for run in runs]
    nb = [len({r["pseudo"] for r in run}) for run in runs]
    per_bench = [max(Counter(r["pseudo"] for r in run).values()) for run in runs]
    sib = [max(Counter(par for _, par in {(x["pseudo"], x["parent"]) for x in run}).values())
           for run in runs]
    parents = Counter(r["parent"] for r in rows)
    kinds = Counter(T.kind_of(r["pseudo"]) for r in rows)
    pseudo_runs = Counter(p for run in runs for p in {r["pseudo"] for r in run})
    subj_runs = Counter(s for run in runs for s in {r["subject"] for r in run})
    use = np.array(list(Counter(r["pseudo"] for r in rows).values()), float)
    use /= use.sum()
    dated = [r["release"] for r in rows if r["release"]]
    shift = (regime.date_shift, regime.date_cap) if regime is not None else (0.0, None)
    shown = [T.shift_date(d, *shift) for d in dated]
    run_mean = [np.mean([r["logit"] for r in run]) for run in runs]
    run_sd = [np.std([r["logit"] for r in run]) for run in runs]
    in_range = [all(44 <= r["eval"] <= 76 for r in run) for run in runs]
    in_range9 = [ok for ok, run in zip(in_range, runs) if len(run) == 9]
    out = {
        "runs": len(runs),
        "pairs_per_run": {"mean": float(np.mean(n)), "min": int(min(n)), "max": int(max(n)),
                          "share_below_5": float(np.mean(np.array(n) < 5))},
        "benchmarks_per_run": float(np.mean(nb)),
        "benchmarks_per_pair": float(np.sum(nb) / np.sum(n)),
        "runs_with_a_repeated_benchmark": float(np.mean(np.array(per_bench) > 1)),
        "parents_per_run": float(np.mean([len({r["parent"] for r in run}) for run in runs])),
        "runs_with_sibling_pseudos": float(np.mean(np.array(sib) > 1)),
        "distinct_subjects_per_pair": float(np.mean([len({r["subject"] for r in run}) / len(run)
                                                     for run in runs])),
        "eval_items": {"min": int(ev.min()), "median": float(np.median(ev)), "max": int(ev.max()),
                       "share_44_to_76": float(np.mean((ev >= 44) & (ev <= 76))),
                       "share_below_44": float(np.mean(ev < 44)),
                       "share_above_76": float(np.mean(ev > 76)),
                       "runs_all_44_to_76": float(np.mean(in_range)),
                       "runs_of_9_all_44_to_76": float(np.mean(in_range9)) if in_range9 else None},
        "pair_logit": {"mean": float(L.mean()), "sd": float(L.std()),
                       "cluster_se": cluster_se(runs, [[r["logit"] for r in run] for run in runs]),
                       "leave_one_parent_out": lopo(runs, [[r["logit"] for r in run]
                                                           for run in runs]),
                       "run_mean_mean": float(np.mean(run_mean)),
                       "run_sd_mean": float(np.mean(run_sd))},
        "pair_rate": {"mean": float(np.mean([r["p"] for r in rows])),
                      "sd": float(np.std([r["p"] for r in rows])),
                      "E_p1p": float(np.mean([r["p"] * (1 - r["p"]) for r in rows]))},
        "released_2025_or_later": float(np.mean([d >= "2025" for d in dated])) if dated else None,
        "released_2025_07_or_later": float(np.mean([d >= "2025-07" for d in dated])) if dated else None,
        "shown_release_after_2026_09_25": float(np.mean([d > "2026-09-25" for d in shown]))
        if shown else None,
        "shown_release_after_2026": float(np.mean([d >= "2027" for d in shown])) if shown else None,
        "undated": float(np.mean([not r["release"] for r in rows])),
        "parents": {k: v / len(rows) for k, v in parents.most_common()},
        "kinds": {k: v / len(rows) for k, v in kinds.most_common()},
        "distinct_pseudos": len(pseudo_runs),
        "effective_pseudos": float(np.exp(-np.sum(use * np.log(use)))),
        "top_pseudos_share_of_runs": {k: v / len(runs) for k, v in pseudo_runs.most_common(5)},
        "distinct_subjects": len(subj_runs),
        "top_subject_share_of_runs": max(subj_runs.values()) / len(runs),
    }
    if record is not None:
        by_id, by_name, name = record
        out["pairs_whose_subject_has_other_public_parents"] = {
            "by_subject_id": float(np.mean([bool(by_id.get(r["subject"], set()) - {r["parent"]})
                                            for r in rows])),
            "by_normalized_name": float(np.mean(
                [bool(by_name.get(name.get(r["subject"], ""), set()) - {r["parent"]})
                 for r in rows]))}
    return out


def reading(b31):
    """The feedback's reading of pair levels from B31 Brier (lower root of
    p(1-p) = B31): logit mean and sd over pairs."""
    x = np.array([math.log(p / (1 - p)) for p in map(T.rate_from_brier, b31)])
    return [float(x.mean()), float(x.std())]


def ppc(runs, record=None):
    """The Predictor's and the empirical mean's numbers against the feedback."""
    rows = [r for run in runs for r in run]
    P = np.array([np.mean([r["pred"] for r in run], 0) for run in runs])     # runs x 7
    E = np.array([np.mean([r["emp"] for r in run], 0) for run in runs])
    orc = [np.mean([r["p"] * (1 - r["p"]) for r in run]) for run in runs]
    out = {"predictor": {}, "empirical_mean": {}}
    for j, b in enumerate(LABELS):
        target = FB_ALC if b == "ALC" else FB_BUDGET[j]
        st = against(P[:, j], target)
        vals = [[r["pred"][j] for r in run] for run in runs]
        st["run_se"] = mean_se(P[:, j])[1]
        st["cluster_se"] = cluster_se(runs, vals)
        if b in ("0", "1", "31", "ALC"):
            st["leave_one_parent_out"] = lopo(runs, vals)
        out["predictor"][b] = st
        out["empirical_mean"][b] = {"mean": float(E[:, j].mean()), "sd": float(E[:, j].std(ddof=1))}
    out["empirical_mean"]["ALC"] = against(E[:, -1], ORGANISERS)
    evals = [[r["emp"][-1] for r in run] for run in runs]
    out["empirical_mean"]["ALC"]["cluster_se"] = cluster_se(runs, evals)
    out["empirical_mean"]["ALC"]["leave_one_parent_out"] = lopo(runs, evals)
    out["oracle_E_p1p"] = mean_se(orc)
    n = np.array([len(run) for run in runs])
    mid = (n >= 8) & (n <= 10)
    out["predictor"]["ALC_sd_runs_of_8_to_10_pairs"] = float(P[mid, -1].std(ddof=1)) \
        if mid.sum() > 1 else None
    out["predictor_minus_empirical_mean_ALC"] = mean_se(P[:, -1] - E[:, -1])
    ece_max = [max(r["ece0"] for r in run) for run in runs]
    out["b0_ece"] = {"pair_mean": float(np.mean([r["ece0"] for r in rows])),
                     "run_max_mean": float(np.mean(ece_max)),
                     "runs_with_max_at_least_0.7": float(np.mean(np.array(ece_max) >= 0.7))}
    q = np.array([r["q"] for r in rows])
    out["b0_prediction"] = {"mean": float(np.nanmean(q)),
                            "quantiles_10_50_90": [float(x) for x in np.nanquantile(q, [.1, .5, .9])],
                            "share_at_least_0.7": float(np.nanmean(q >= 0.7)),
                            "by_parent": {par: {"mean": float(np.nanmean(x)),
                                                "p90": float(np.nanquantile(x, 0.9)),
                                                "max": float(np.nanmax(x))}
                                          for par in sorted({r["parent"] for r in rows})
                                          for x in [np.array([r["q"] for r in rows
                                                              if r["parent"] == par])]}}
    # the feedback's level reading (lower root of p(1-p) = B31), applied like
    # for like to the replica's own Predictor B31, against realized levels
    out["b31_level_reading"] = {
        "feedback": reading(FB[:, 5]),
        "replica": reading([r["pred"][5] for r in rows]),
        "replica_realized": [float(np.mean([r["logit"] for r in rows])),
                             float(np.std([r["logit"] for r in rows]))],
        "order": "[logit mean, logit sd] over pairs"}
    # pair level: the feedback's 9 rows against every replica pair
    R = np.array([r["pred"][:6] for r in rows])
    qs = [10, 25, 50, 75, 90]
    out["pairs"] = {
        "quantiles": {str(b): {"replica": [float(x) for x in np.percentile(R[:, j], qs)],
                               "feedback": [float(x) for x in np.percentile(FB[:, j], qs)],
                               "feedback_mean_percentile": float(np.mean(
                                   [np.mean(R[:, j] <= f) for f in FB[:, j]]))}
                      for j, b in enumerate(BUDGETS)},
        "share_B0_at_least_0.40": [float(np.mean(R[:, 0] >= 0.40)), float(np.mean(FB[:, 0] >= 0.40))],
        "share_B31_at_most_0.10": [float(np.mean(R[:, 5] <= 0.10)), float(np.mean(FB[:, 5] <= 0.10))],
        "share_flat_B0_minus_B31_under_0.02": [float(np.mean(np.abs(R[:, 0] - R[:, 5]) < 0.02)),
                                               float(np.mean(np.abs(FB[:, 0] - FB[:, 5]) < 0.02))],
        "order": "[replica, feedback]",
    }
    return out


def distance(p, budgets=(0, 1)):
    """How far a setting's Predictor is from the feedback at `budgets`, each in
    the replica's single-run sd, in quadrature."""
    return float(math.sqrt(sum(((p[str(b)]["mean"] - FB_BUDGET[BUDGETS.index(b)])
                                / p[str(b)]["sd"]) ** 2 for b in budgets)))


def ranking(runs):
    """Legacy Predictor, date-blind Predictor and empirical mean on the same runs:
    means, paired differences with (parent, subject) cluster SEs, and order."""
    out = {}
    cols = {"Predictor": "pred", "date-blind Predictor": "blind", "empirical mean": "emp"}
    for j, b in enumerate(LABELS):
        means = {k: float(np.mean([np.mean([r[c][j] for r in run]) for run in runs]))
                 for k, c in cols.items()}
        diffs = {}
        for a, c in (("Predictor", "date-blind Predictor"), ("Predictor", "empirical mean"),
                     ("date-blind Predictor", "empirical mean")):
            vals = [[r[cols[a]][j] - r[cols[c]][j] for r in run] for run in runs]
            diffs[f"{a} - {c}"] = [float(np.mean([np.mean(v) for v in vals])),
                                   cluster_se(runs, vals)]
        out[b] = {"means": means, "differences": diffs,
                  "order": sorted(means, key=means.get)}
    return out


def equivalence(raw, cfg):
    """The legacy Predictor under the date shift against the same Predictor with
    the equivalent offset on its prior and real dates, on the same draws. B0 must
    agree row for row. Later budgets agree only in expectation: the platform's
    random acquisition hashes the visible input, dates included, so a shift also
    redraws which items are acquired. Mean differences (shift - offset) with
    (parent, subject) cluster SEs."""
    runs, vals, dq = [], [], []
    for i in range(cfg["sens_runs"]):
        k0, k1 = (f"sens|{name}|{i}" for name in RANKING)
        if k0 in raw and k1 in raw and signature(raw[k0]["rows"]) == signature(raw[k1]["rows"]):
            runs.append(raw[k0]["rows"])
            vals.append([[x - y for x, y in zip(r0["pred"], r1["pred"])]
                         for r0, r1 in zip(raw[k0]["rows"], raw[k1]["rows"])])
            dq += [abs(r0["q"] - r1["q"]) for r0, r1 in zip(raw[k0]["rows"], raw[k1]["rows"])]
    if len(runs) < 2:
        return None
    out = {"runs": len(runs),
           "b0_max_abs_diff": float(max(abs(v[0]) for run in vals for v in run)),
           "b0_prediction_max_abs_diff": float(max(dq))}
    for j, lab in enumerate(LABELS):
        d = [[v[j] for v in run] for run in vals]
        out[lab] = [float(np.mean([np.mean(x) for x in d])), cluster_se(runs, d)]
    return out


def check_against_baselines(runs, ids):
    """R1 runs are those of official_baselines.py (seed 0): its per-run ALCs for
    the target-LOBO Predictor and the empirical mean must be reproduced."""
    if not os.path.exists(BASELINES):
        return None
    with open(BASELINES) as f:
        per = {r["run"]: r["alc"] for r in json.load(f)["r1"]["per_run"]}
    d_pred, d_emp = [], []
    for i, run in zip(ids, runs):
        if i in per:
            d_pred.append(np.mean([r["pred"][-1] for r in run]) - per[i]["Predictor, target-LOBO prior"])
            d_emp.append(np.mean([r["emp"][-1] for r in run]) - per[i]["empirical mean"])
    if not d_pred:
        return None
    return {"runs": len(d_pred), "max_abs_diff_predictor": float(np.max(np.abs(d_pred))),
            "max_abs_diff_empirical_mean": float(np.max(np.abs(d_emp)))}


# --- item structure ------------------------------------------------------------------

def structure(cfg):
    """How much item structure the pairs carry, per regime (redrawn from their
    seeds): the item-level oracle's Brier gain over the pair-rate oracle
    (testlike.item_oracle) and the sd of item difficulty within a pair, overall,
    by pseudo-benchmark kind and by parent."""
    n = {"check": cfg["runs"], "sens": cfg["sens_runs"], "r1": cfg["r1_runs"]}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind, name in STRUCTURE:
            ov = overrides_of(kind, name, cfg)
            rows = [x for i in range(n[kind])
                    for x in T.item_oracle(draw(kind, ov, i), catalogue()) if x is not None]

            def summ(xs):
                lv = float(np.mean([x["level"] for x in xs]))
                it = float(np.mean([x["item"] for x in xs]))
                return {"pairs": len(xs), "level_oracle": lv, "item_oracle": it, "gain": lv - it,
                        "relative_gain": (lv - it) / lv if lv > 0 else None,
                        "b_sd": float(np.mean([x["b_sd"] for x in xs]))}
            e = {"all": summ(rows)}
            for k in sorted({x["kind"] for x in rows}):
                e[f"kind {k}"] = summ([x for x in rows if x["kind"] == k])
            for par in sorted({x["parent"] for x in rows}):
                e[f"parent {par}"] = summ([x for x in rows if x["parent"] == par])
            out[f"{kind}|{name}"] = e
    return out


# --- tasks, checkpointing, summary ----------------------------------------------------

def grid(cfg):
    out = {}
    for m in cfg["grid_mean"]:
        for d in cfg["grid_shift"]:
            out[f"level_mean {m:+.1f}, date_shift {d:.2f}"] = {"level_mean": m, "date_shift": d,
                                                                "level_sd": 1.5}
    return out


def tasks_for(cfg, phase):
    out = []
    if phase in ("tune", "all"):
        for name, ov in grid(cfg).items():
            out += [("tune", name, ov, i) for i in range(cfg["tune_runs"])]
    if phase in ("check", "all"):
        out += [("check", "default", {}, i) for i in range(cfg["runs"])]
        for name, ov in SENS.items():
            out += [("sens", name, ov, i) for i in range(cfg["sens_runs"])]
        out += [("r1", "public R1", {}, i) for i in range(cfg["r1_runs"])]
    return out


def collect(raw, kind, name, n):
    ids = [i for i in range(n) if f"{kind}|{name}|{i}" in raw]
    return ids, [raw[f"{kind}|{name}|{i}"]["rows"] for i in ids]


def f4(x):
    return None if x is None else round(float(x), 4)


NOTES = [
    "B0 and B1 agree with the feedback because the default was tuned to them "
    "(in sample); B3, B7 and B15 were never used in choosing it "
    "(distance_B3_B7_B15), B31 only as a tie-break among level means.",
    "ALC does not tell the regimes apart: plain public R1 runs match the "
    "feedback's ALC as well as test-like runs do. The budget profile does (B0 "
    "high, B7-B31 low).",
    "level_mean is not identified by one 9-pair run (level_identification): "
    "repeat any downstream choice at level_mean -1.2 and -2.0.",
    "The B0/B1 match rests on date_shift, a change to visible dates. For the "
    "legacy prior (linear, unclipped in days) it equals an additive offset on "
    "the prior's ability (ranking_agreement.legacy_offset_equivalence). For "
    "predictors that read dates otherwise (clipped, nonlinear, ignoring dates) "
    "or read other attributes, B0/B1 results are driven by this knob: rankings "
    "under the shift and under the offset differ (ranking_agreement).",
    "Pseudo-benchmarks carry less item structure than public benchmarks "
    "(structure; catalogue key_share and b_var). A comparison between an item- "
    "or group-aware predictor and a level-only one must hold on 'no strata, "
    "groups merged at random' (kinds mix, whole) and on public R1 before it "
    "drives a choice.",
    "Standard errors cluster on (parent, subject), testlike.cluster_key; "
    "leave_one_parent_out shows how much a number rests on the four parents. "
    "Downstream paired comparisons on test-like runs should cluster the same way.",
    "About a quarter of pairs evaluate fewer than 44 items (the feedback's "
    "minimum): official.py splits a pair before the cut, so the evaluated count "
    "varies; 'split after the cut, 88-item floor' emulates the alternative.",
    "Siblings: most runs hold two or more pseudo-benchmarks of one parent "
    "(runs_with_sibling_pseudos); 'one pseudo-benchmark per parent' removes "
    "them. A run subject's public record (by id or name) reaches a prior "
    "fitted on the other parents; 'run subjects hidden from the prior' and "
    "testlike.training_pairs(hide_subjects=True) withhold it, and should be "
    "used before a name-based subject prior is judged on this regime.",
    "Hooks requested in paiec/official.py (not owned here): training_pairs and "
    "run_benchmarks should map names through testlike.parent_of or refuse "
    "names that are not public benchmarks (on test-like runs they exclude "
    "nothing); sample_run could take split-after-cut and an allocation floor "
    "(88 kept items) as parameters.",
]


def summarise(raw, cfg):
    record = other_parents()
    s = {"feedback": {"budgets": dict(zip(map(str, BUDGETS), FB_BUDGET.tolist())), "ALC": FB_ALC,
                      "pairs": len(T.FEEDBACK), "benchmarks": len({r[0] for r in T.FEEDBACK}),
                      "eval_items": [r[1] for r in T.FEEDBACK],
                      "b31_level_reading": reading(FB[:, 5]),
                      "organisers_entry": ORGANISERS},
         "notes": NOTES}
    tune = {}
    for name, ov in grid(cfg).items():
        _, runs = collect(raw, "tune", name, cfg["tune_runs"])
        if len(runs) < 2:
            continue
        p = ppc(runs)
        st = regime_stats(runs, regime=regime_of("tune", name, cfg))
        tune[name] = {"overrides": ov, "runs": len(runs), "distance": distance(p["predictor"]),
                      "distance_with_B31": distance(p["predictor"], (0, 1, 31)),
                      "distance_B3_B7_B15": distance(p["predictor"], (3, 7, 15)),
                      "predictor": {b: [f4(p["predictor"][b]["mean"]), f4(p["predictor"][b]["sd"])]
                                    for b in LABELS},
                      "empirical_mean_ALC": [f4(p["empirical_mean"]["ALC"]["mean"]),
                                             f4(p["empirical_mean"]["ALC"]["sd"])],
                      "pair_logit": [f4(st["pair_logit"]["mean"]), f4(st["pair_logit"]["sd"])],
                      "b31_level_reading": [f4(x) for x in p["b31_level_reading"]["replica"]],
                      "E_p1p": f4(st["pair_rate"]["E_p1p"]),
                      "b0_prediction": f4(p["b0_prediction"]["mean"])}
    d = T.Regime()
    if tune:
        best = min(tune, key=lambda k: tune[k]["distance"])
        best3 = min(tune, key=lambda k: tune[k]["distance_with_B31"])
        s["tune"] = {"settings": tune, "closest": best, "closest_overrides": tune[best]["overrides"],
                     "closest_with_B31": best3,
                     "closest_with_B31_overrides": tune[best3]["overrides"]}
        same = {k: v for k, v in tune.items() if abs(v["overrides"]["date_shift"] - d.date_shift) < 1e-9}
        s["level_identification"] = {
            "date_shift": d.date_shift,
            "settings": {k: {"distance_B0_B1": f4(v["distance"]),
                             "distance_with_B31": f4(v["distance_with_B31"]),
                             "b31_level_reading": v["b31_level_reading"],
                             "realized_pair_logit": v["pair_logit"]} for k, v in same.items()},
            "feedback_b31_level_reading": [f4(x) for x in reading(FB[:, 5])],
            "reading": "each setting's distance is in single-run sds; under 1 means one run of "
                       "the feedback's size cannot tell it from the feedback"}
    s["default_regime"] = {k: v for k, v in asdict(d).items()}
    if tune:
        for k in ("closest", "closest_with_B31"):
            s[f"default_matches_{k}"] = all(abs(getattr(d, a) - v) < 1e-9
                                            for a, v in s["tune"][f"{k}_overrides"].items())
    for kind, name, n in [("check", "default", cfg["runs"]), ("r1", "public R1", cfg["r1_runs"])] + \
            [("sens", k, cfg["sens_runs"]) for k in SENS]:
        ids, runs = collect(raw, kind, name, n)
        if len(runs) < 2:
            continue
        entry = {"runs": len(runs), "regime": regime_stats(runs, record, regime_of(kind, name, cfg)),
                 "ppc": ppc(runs)}
        entry["distance"] = distance(entry["ppc"]["predictor"])
        entry["distance_with_B31"] = distance(entry["ppc"]["predictor"], (0, 1, 31))
        entry["distance_B3_B7_B15"] = distance(entry["ppc"]["predictor"], (3, 7, 15))
        if kind == "r1":
            entry["official_baselines_check"] = check_against_baselines(runs, ids)
        if kind == "sens":
            entry["overrides"] = SENS[name]
        if all("blind" in r for run in runs for r in run):
            entry["ranking"] = ranking(runs)
        s.setdefault("sensitivity" if kind == "sens" else kind, {})[name] = entry
    sens = s.get("sensitivity", {})
    if all(k in sens and "ranking" in sens[k] for k in RANKING):
        a, b = (sens[k]["ranking"] for k in RANKING)
        s["ranking_agreement"] = {
            "variants": list(RANKING),
            "same_order": {lab: a[lab]["order"] == b[lab]["order"] for lab in LABELS},
            "orders": {lab: [a[lab]["order"], b[lab]["order"]] for lab in LABELS},
            "predictor_minus_date_blind": {
                lab: [a[lab]["differences"]["Predictor - date-blind Predictor"],
                      b[lab]["differences"]["Predictor - date-blind Predictor"]] for lab in LABELS},
            "legacy_offset_equivalence": equivalence(raw, cfg)}
    s["structure"] = structure(cfg)
    times = [v["task_s"] for v in raw.values()]
    s["cost"] = {"tasks": len(raw), "task_s_mean": float(np.mean(times)) if times else None,
                 "task_s_total": float(np.sum(times)) if times else None}
    return s


def catalogue_table():
    cat = catalogue()
    S = T.Sampler(cat, T.Regime(level_mean=None, exclude=(), kinds=T.KINDS))
    out = []
    for j, b in enumerate(cat.pseudos):
        if b.kind == "chunk" and b.tag not in ("c0",):
            continue
        row = {"name": b.name, "kind": b.kind, "groups": list(b.groups), "items": len(b.items),
               "pairs": len(b.pairs), "recent_pairs": len(S.of.get(j, [])),
               "recent_level": f4(S.level[j]) if j in S.of else None}
        diff = cat.difficulty.get(b.parent)
        if diff:
            row["b_var"] = f4(np.var([diff[k] for k in b.items if k in diff]))
            if b.parent in cat.features:
                ks = T.key_share(b.items, diff, cat.features[b.parent])
                row["key_share"] = f4(ks["key_share"])
                row["key_values"] = ks["key_values"]
        out.append(row)
    chunks = sum(b.kind == "chunk" for b in cat.pseudos)
    dates = {}
    for p in pairs():
        d = str(p.subject.get("release_date") or "")
        c = dates.setdefault(p.benchmark_id, Counter())
        c["pairs"] += 1
        c["dated"] += bool(d)
        c["released_2025_01_or_later"] += d >= "2025-01"
        c["released_2025_07_or_later"] += d >= "2025-07"
        c["released_2026_or_later"] += d >= "2026"
    return {"pseudos": out, "chunks": chunks, "info": cat.info,
            "public_release_dates": {b: dict(c) for b, c in sorted(dates.items())}}


def write(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=None, separators=(",", ":"))
    os.replace(tmp, path)


def show(s):
    fb = s["feedback"]
    print("\nfeedback: B0..B31 " + " ".join(f"{x:.4f}" for x in fb["budgets"].values())
          + f"  ALC {fb['ALC']:.4f}  B31 level reading {fb['b31_level_reading'][0]:+.2f}/"
          f"{fb['b31_level_reading'][1]:.2f}")
    if "tune" in s:
        print("\ntuning grid (Predictor B0 / B1 / B31 / ALC run means; distance in run sds):")
        for k, v in sorted(s["tune"]["settings"].items(), key=lambda kv: kv[1]["distance"]):
            p = v["predictor"]
            print(f"  {k:36s} d {v['distance']:.2f} d3 {v['distance_with_B31']:.2f} "
                  f"u {v['distance_B3_B7_B15']:.2f}  B0 {p['0'][0]:.4f} B1 {p['1'][0]:.4f} "
                  f"B31 {p['31'][0]:.4f} ALC {p['ALC'][0]:.4f} (sd {p['ALC'][1]:.4f})  "
                  f"emp {v['empirical_mean_ALC'][0]:.4f}  logit {v['pair_logit'][0]:+.2f}/"
                  f"{v['pair_logit'][1]:.2f} read {v['b31_level_reading'][0]:+.2f}/"
                  f"{v['b31_level_reading'][1]:.2f}  q0 {v['b0_prediction']:.3f}")
        print(f"  closest at B0, B1: {s['tune']['closest']} (default matches: "
              f"{s.get('default_matches_closest')}); with B31: {s['tune']['closest_with_B31']} "
              f"(default matches: {s.get('default_matches_closest_with_B31')})")
    for grp in ("check", "sensitivity", "r1"):
        for name, e in s.get(grp, {}).items():
            p, rs = e["ppc"], e["regime"]
            print(f"\n[{grp}] {name}: {e['runs']} runs, distance {e['distance']:.2f} "
                  f"(with B31 {e['distance_with_B31']:.2f}; B3/B7/B15 {e['distance_B3_B7_B15']:.2f})")
            lo = rs["pair_logit"]["leave_one_parent_out"]
            print("  pairs/run {:.2f} (min {}), benchmarks/pair {:.2f}, sibling runs {:.2f}, eval "
                  "items {}..{} (median {:.0f}, <44 {:.2f}), logit {:+.2f} sd {:.2f} (SE {:.3f}, "
                  "LOPO {:+.2f}..{:+.2f}), E[p(1-p)] {:.4f}, eff. pseudos {:.1f}".format(
                      rs["pairs_per_run"]["mean"], rs["pairs_per_run"]["min"],
                      rs["benchmarks_per_pair"], rs["runs_with_sibling_pseudos"],
                      rs["eval_items"]["min"], rs["eval_items"]["max"], rs["eval_items"]["median"],
                      rs["eval_items"]["share_below_44"],
                      rs["pair_logit"]["mean"], rs["pair_logit"]["sd"], rs["pair_logit"]["cluster_se"],
                      lo["min"], lo["max"], rs["pair_rate"]["E_p1p"], rs["effective_pseudos"]))
            for b in LABELS:
                x = p["predictor"][b]
                extra = ""
                if "leave_one_parent_out" in x:
                    extra = f"  LOPO {x['leave_one_parent_out']['min']:.4f}..{x['leave_one_parent_out']['max']:.4f}"
                print(f"  Predictor {b:>3s}: {x['mean']:.4f} (run sd {x['sd']:.4f}, SE "
                      f"{x['run_se']:.4f}/{x['cluster_se']:.4f})  feedback {x['target']:.4f} "
                      f"z {x['z']:+.2f} pct {x['percentile']:.2f}{extra}")
            em = p["empirical_mean"]["ALC"]
            rd = p["b31_level_reading"]
            print(f"  empirical mean ALC {em['mean']:.4f} (sd {em['sd']:.4f}, SE {em['cluster_se']:.4f}); "
                  f"0.1801 at z {em['z']:+.2f}, pct {em['percentile']:.2f}; oracle E[p(1-p)] "
                  f"{p['oracle_E_p1p'][0]:.4f}; B0 prediction mean {p['b0_prediction']['mean']:.3f}; "
                  f"B0 ECE run max {p['b0_ece']['run_max_mean']:.3f}; B31 reading "
                  f"{rd['replica'][0]:+.2f}/{rd['replica'][1]:.2f} vs feedback "
                  f"{rd['feedback'][0]:+.2f}/{rd['feedback'][1]:.2f}")
            if "ranking" in e:
                for b in ("0", "1", "ALC"):
                    rk = e["ranking"][b]
                    print(f"  ranking {b:>3s}: " + ", ".join(f"{k} {v:.4f}" for k, v in rk["means"].items())
                          + "; " + ", ".join(f"{k} {v[0]:+.4f} ({v[1]:.4f})"
                                             for k, v in rk["differences"].items()))
            if e.get("official_baselines_check"):
                print(f"  vs official_baselines.json: {e['official_baselines_check']}")
    if "ranking_agreement" in s:
        print("\nranking agreement:", json.dumps(s["ranking_agreement"]["same_order"]))
    if "structure" in s:
        print("\nitem structure (item-oracle gain over the pair-rate oracle; within-pair sd of b):")
        for k, e in s["structure"].items():
            print(f"  {k}: " + "; ".join(f"{g} {v['gain']:.4f} ({v['relative_gain']:.0%}, b sd "
                                          f"{v['b_sd']:.2f}, n {v['pairs']})"
                                          for g, v in e.items() if not g.startswith("parent")))


def main(args):
    cfg = {"tune_runs": args.tune_runs, "runs": args.runs, "sens_runs": args.sens_runs,
           "r1_runs": args.r1_runs, "grid_mean": list(args.grid_mean),
           "grid_shift": list(args.grid_shift), "boots": BOOTS, "seeds": SEEDS}
    command = "python experiments/testlike_check.py " + " ".join(sys.argv[1:])
    raw, history = {}, []
    if (args.resume or args.summarise) and os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        raw, history = old.get("raw", {}), old.get("invocations", [])
    if not args.summarise:
        history.append({"command": command, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    if args.validate and raw:
        t0 = time.time()
        bad = stale(raw, cfg)
        for k in bad:
            del raw[k]
        if history:
            history[-1]["stale_dropped"] = bad
        print(f"validated {len(raw) + len(bad)} runs in {time.time() - t0:.0f}s; "
              f"{len(bad)} no longer draw the same and will rerun: {bad}", flush=True)
    todo = [] if args.summarise else [t for t in tasks_for(cfg, args.phase) if task_key(t) not in raw]
    doc = {"status": "partial", "command": command, "invocations": history, "config": cfg}
    doc["catalogue"] = catalogue_table()
    print(f"{len(todo)} tasks to run, {len(raw)} already done", flush=True)
    t0, last = time.time(), time.time()
    if todo:
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for n, (key, out) in enumerate(pool.imap_unordered(task, todo, chunksize=1), 1):
                raw[key] = out
                if time.time() - last > 30 or n == len(todo):
                    doc["raw"] = raw
                    write(args.out, doc)
                    last = time.time()
                    print(f"  {n}/{len(todo)} tasks, {time.time() - t0:.0f}s", flush=True)
    doc["raw"] = raw
    doc["summary"] = summarise(raw, cfg)
    doc["status"] = "complete" if all(task_key(t) in raw for t in tasks_for(cfg, "all")) else "partial"
    doc["wall_s"] = time.time() - t0
    write(args.out, doc)
    show(doc["summary"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--phase", choices=("tune", "check", "all"), default="all")
    ap.add_argument("--tune-runs", type=int, default=100)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--sens-runs", type=int, default=100)
    ap.add_argument("--r1-runs", type=int, default=200)
    ap.add_argument("--grid-mean", type=float, nargs="+", default=GRID_MEAN)
    ap.add_argument("--grid-shift", type=float, nargs="+", default=GRID_SHIFT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="redraw every stored run and rerun those that no longer match")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--out", default=OUT)
    main(ap.parse_args())
