"""The gate's pass probabilities, every quoted transfer correlation on one scale
with a confidence interval, and the single-subject benchmark, from stored
results and stored features only (review P1.12, P1.13 and P1.15;
docs/report/review_v0.md sections 4, 6, 7 and 9).

No language model is run and no predictor is called. What is read:

  results/harness_thresholds.json   the gate table and its noise draws (gate)
  git history and results/*.json    when the gate was fixed (evidence)
  data/strong_llm_eval/targets.json the honest difficulty the harness and the
                                    language-model studies use: per parent the
                                    five subject-fold maps and their mean
  data/<parent>/*.parquet           Rasch refits on halves of the subjects
                                    (split-half reliability) and on the five
                                    folds (checking targets.json); TF-IDF and
                                    embedding maps refitted, because
                                    experiments/transfer.py and emb_transfer.py
                                    store no per-item predictions
  data/features                     Qwen3-Embedding-0.6B and Qwen3-4B features
  data/strong_llm_eval              the 14B's features, entropy table and heads'
                                    out-of-fold predictions
  experiments/llm_rating            the blind ratings
  data/harness_rows/<regime>        test-like pair appearances (within-pair r)
                                    and public ones (swe_rebench's item overlap)
  data/subject_side_rows,           the shipped model and its comparators per
  data/regime_sensitivity_rows      pair appearance (single)

Scales. The gate table's r ("Acceptance harness") is the Pearson correlation
over a parent's items between a covariate and the *honest* difficulty: the
Rasch difficulty fitted on the other four of five subject folds, the fold
being the target subject's. Its secondary axis is the within-pair r: the mean,
over test-like pair appearances with 5+ evaluated items on which x varies, of
the correlation between x (standardised within benchmark as the harness reads
it) and the honest difficulty of the appearance's fold over the pair's
evaluated items (itemcov_eval.realised_r). Most correlations the draft quotes
are against the fold-averaged or full-sample difficulty, often within groups
or by Spearman. This script re-reads each quoted covariate on both of the
harness's scales:

  r_honest   per parent, the mean over the five folds of Pearson(x, z_f) over
             the parent's items carrying x; with group- and item-bootstrap
             intervals (groups: itemcov_eval.GROUP_KEY), and r_vs_mean, the same
             against the fold-averaged difficulty, beside it
  within     the within-pair r, recomputed (checked against every stored
             r_within_pair_tl) with a cluster bootstrap over (parent, subject)

and gives the honest difficulty's reliability two ways: split halves of the
subjects (Rasch refits, Spearman-Brown to four fifths and to all of them), and
the five stored fold maps' mutual correlation under a subject-additive error
model (rel_honest = 1 - (k-1)(1-c) for k folds that each leave one fold out).
kappa = sqrt(rel_honest / rel_all) is what a correlation against the full-sample
difficulty shrinks by on the honest scale.

Stages (each writes its section of results/gate_and_ci.json and keeps the rest)
  gate      P1.12: per r and line, the stored draws' pass count with a Jeffreys
            interval, the normal and t-predictive probability of clearing
            -0.002 from the draws' mean and sd, which condition failed on each
            draw (the worst-parent condition is not stored per draw and is
            inferred where every other condition held), and the same bar read
            on mix/whole with selection kept on test-like runs
  evidence  P1.12: when the gate was committed against when each study's results
            were committed and, where recorded, its data were produced
  scale     P1.12 and P1.13: reliability; every covariate family (blind ratings,
            the 4B judge, the 14B's rubric, heads, attempts and entropy,
            embeddings, TF-IDF) on its quoted scale with intervals and on both
            harness scales; --families picks some
  ci        P1.13: the table of every quoted correlation with its interval, the
            places none can be computed, the 4B-power argument (no model run),
            and what share of its parent-scale r each covariate keeps within a
            pair, beside the degraded oracle's (unit_ratio)
  single    P1.15: swe_rebench, the shipped config against the legacy Predictor,
            the smoothed and empirical means (seed 0 rows, the run-2 archive's
            library) and against P1a's ten alternatives (seed 11 rows, the
            current archive's library)
  show      markdown of what is stored

Run:  python experiments/gate_and_ci.py --stage gate          # seconds
      python experiments/gate_and_ci.py --stage evidence      # seconds
      python experiments/gate_and_ci.py --stage scale         # about 6 minutes, < 1 GB
      python experiments/gate_and_ci.py --stage ci            # seconds
      python experiments/gate_and_ci.py --stage single        # about 30 s, < 1 GB
      python experiments/gate_and_ci.py --stage show
(--stage all runs the five in that order, one process, BLAS on one thread.)
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "gate_and_ci.json")
RES = os.path.join(ROOT, "results")
HT = os.path.join(RES, "harness_thresholds.json")
TARGETS = os.path.join(ROOT, "data", "strong_llm_eval", "targets.json")
P1A_ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
LINES = ("transferred nested", "per-pair nested")
R_ASKED = (0.2, 0.3, 0.4, 0.5)
PROBS = (0.5, 0.8, 0.95)
BOOTS = 2000
SEED = 0
HALF_SPLITS = 20
HALF_SALT = "gate-and-ci-half"
MIN_ITEMS = 20                      # a parent enters a covariate's r_honest with this many items carrying x
HONEST_FOLDS = 5
#: the bar a covariate's correlation is read against, on each scale (harness_thresholds.json,
#: tables.honest: the transferred nested line at r = 0.3 passes on 6 of 8 draws)
FAMILIES = ("reliability", "ratings", "judge4b", "strong14b", "entropy", "embeddings", "tfidf")
P1A_CONFIGS = ("legacy", "smooth", "smcal", "onepl", "eb_fit", "eb_adapt", "eb_ship", "aggr",
               "wide35", "wide50")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def ci4(v):
    return None if v is None else [r4(v[0]), r4(v[1])]


# --- pure statistics (tests/test_gate_and_ci.py) ------------------------------------------------

def jeffreys(k, n, level=0.95):
    """Jeffreys interval for a binomial proportion k / n (Beta(k + 1/2, n - k + 1/2)
    quantiles; the bound is 0 or 1 when k is 0 or n)."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 0.5, n - k + 0.5))
    return [lo, hi]


def normal_pass(mean, sd, bar):
    """P(X <= bar) for X ~ N(mean, sd^2): the chance that one draw clears a bar
    below which lower is better."""
    from scipy.stats import norm
    if not sd > 0:
        return float(mean <= bar)
    return float(norm.cdf((bar - mean) / sd))


def t_predictive_pass(mean, sd, n, bar):
    """The same chance for a new draw when mean and sd are estimated from n draws:
    Student t with n - 1 degrees of freedom, scale sd sqrt(1 + 1/n)."""
    from scipy.stats import t
    if not sd > 0 or n < 2:
        return float(mean <= bar)
    return float(t.cdf((bar - mean) / (sd * math.sqrt(1 + 1 / n)), n - 1))


def spearman_brown(rho, m):
    """Reliability of a measure m times as long as one whose reliability is rho."""
    return m * rho / (1 + (m - 1) * rho)


def rel_from_fold_overlap(c, k=HONEST_FOLDS):
    """(rel_honest, rel_all) from the mean correlation c between k leave-one-fold-out
    estimates of the same quantity, when each subject adds independent error:
    var(e_S) = s / |S|, cov(e_S, e_T) = s |S & T| / (|S| |T|). With |S| = (k-1)/k n
    and |S & T| = (k-2)/k n, 1 - rel_honest = (k - 1)(1 - c); rel_all follows
    from s / V = (1 / rel_honest - 1) (k - 1) / k."""
    rel_h = 1 - (k - 1) * (1 - c)
    if not 0 < rel_h <= 1:
        return rel_h, None
    sv = (1 / rel_h - 1) * (k - 1) / k
    return rel_h, 1 / (1 + sv)


def fisher_ci(r, n, level=0.95):
    from scipy.stats import norm
    if n <= 3 or not abs(r) < 1:
        return None
    z, se = np.arctanh(r), 1 / math.sqrt(n - 3)
    q = norm.ppf(1 - (1 - level) / 2)
    return [float(np.tanh(z - q * se)), float(np.tanh(z + q * se))]


def n_for_power(r_true, r_bar, alpha=0.05, power=0.8):
    """Items needed for a two-sided Fisher-z test at alpha to exclude r_bar with the
    given power when the true correlation is r_true (independent items)."""
    from scipy.stats import norm
    d = abs(np.arctanh(r_bar) - np.arctanh(r_true))
    if d == 0:
        return float("inf")
    return float(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) / d) ** 2 + 3)


def boot_weights(n, groups=None, boots=BOOTS, seed=SEED):
    """(boots, n) resampling counts: items drawn with replacement, or whole groups
    (each item gets its group's count). groups None: the item bootstrap."""
    rng = np.random.default_rng(seed)
    if groups is None:
        pick = rng.integers(0, n, (boots, n))
        W = np.zeros((boots, n))
        np.add.at(W, (np.repeat(np.arange(boots), n), pick.ravel()), 1.0)
        return W
    _, inv = np.unique(np.asarray(groups, object).astype(str), return_inverse=True)
    G = inv.max() + 1
    pick = rng.integers(0, G, (boots, G))
    C = np.zeros((boots, G))
    np.add.at(C, (np.repeat(np.arange(boots), G), pick.ravel()), 1.0)
    return C[:, inv]


def wpearson(W, x, z):
    """Pearson of x and z under each row of weights W ((B, n) or (n,)); NaN where a
    row leaves no spread."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    x = x - x.mean()
    z = z - z.mean()
    W = np.atleast_2d(W)
    N = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, mz = W @ x / N, W @ z / N
        sxx = W @ (x * x) / N - mx * mx
        szz = W @ (z * z) / N - mz * mz
        sxz = W @ (x * z) / N - mx * mz
        r = sxz / np.sqrt(sxx * szz)
    return np.where((sxx > 1e-12) & (szz > 1e-12), r, np.nan)


def wspearman(W, x, z):
    """Spearman on the original ranks under bootstrap weights (ties averaged): the
    resampled items keep the ranks they have in the full sample."""
    from scipy.stats import rankdata
    return wpearson(W, rankdata(x), rankdata(z))


def pct_ci(v, level=0.95):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 50:
        return None
    a = 100 * (1 - level) / 2
    return [float(np.percentile(v, a)), float(np.percentile(v, 100 - a))]


def corr_with_ci(x, z, groups=None, boots=BOOTS, seed=SEED, kind="pearson"):
    """{est, n, groups, ci_group, ci_item, ci_fisher}: the correlation of x and z
    with percentile intervals from a group bootstrap (when groups are given and
    there are at least 5) and an item bootstrap, and the Fisher interval."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(z)
    x, z = x[ok], z[ok]
    f = wpearson if kind == "pearson" else wspearman
    est = float(f(np.ones(len(x)), x, z)[0])
    out = {"est": est, "n": int(len(x)), "statistic": kind}
    out["ci_item"] = pct_ci(f(boot_weights(len(x), None, boots, seed), x, z))
    if groups is not None:
        g = np.asarray([gi if gi else f"item|{i}" for i, gi in enumerate(groups)], object)[ok]
        G = len(set(g.tolist()))
        out["groups"] = G
        out["ci_group"] = pct_ci(f(boot_weights(len(x), g, boots, seed), x, z)) if G >= 5 else None
    out["ci_fisher"] = fisher_ci(est, len(x)) if kind == "pearson" else None
    return out


def fold_r(W, x, Z, has):
    """Mean over folds of the weighted Pearson of x and fold f's difficulty over
    the items fold f has: (B,) or a scalar row."""
    rs = []
    for f in range(Z.shape[0]):
        m = has[f]
        if m.sum() < 5:
            continue
        rs.append(wpearson(np.atleast_2d(W)[:, m], x[m], Z[f, m]))
    return np.nanmean(np.vstack(rs), 0) if rs else np.full(np.atleast_2d(W).shape[0], np.nan)


def cluster_mean_ci(v, clusters, boots=BOOTS, seed=SEED):
    """Mean of v and a percentile interval from resampling clusters (a ratio
    estimator: resampled sum over resampled count)."""
    v = np.asarray(v, float)
    if not len(v):
        return None, None
    cl, inv = np.unique(np.asarray(clusters, object).astype(str), return_inverse=True)
    S = np.bincount(inv, v, len(cl))
    N = np.bincount(inv, None, len(cl)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(cl), (boots, len(cl)))
    est = S[pick].sum(1) / N[pick].sum(1)
    return float(v.mean()), pct_ci(est)


# --- the gate (P1.12) ---------------------------------------------------------------------------

def draw_conditions(tl, mix, r1b, r1p, folds_on, passed, gate):
    """One noise draw against the gate. Every condition but the worst parent's
    is stored per draw; that one is inferred: it held if the draw passed, failed
    if the draw failed while every other condition held, else unknown (None)."""
    known = {"selection_on": folds_on is None or folds_on >= gate["folds_on"],
             "tl": tl <= gate["tl"],
             "mix_same_sign": tl != 0 and np.sign(mix) == np.sign(tl),
             "guard": r1b <= gate["guard"] and r1p <= gate["guard"]}
    if passed:
        parent = True
    elif all(known.values()):
        parent = False
    else:
        parent = None
    mix_reading = {"selection_on": known["selection_on"], "mix": mix <= gate["tl"],
                   "tl_same_sign": mix != 0 and np.sign(mix) == np.sign(tl), "guard": known["guard"]}
    return {**{k: bool(v) for k, v in known.items()}, "worst_parent_inferred": parent,
            "pass": bool(passed), "mix_reading_known": bool(all(mix_reading.values()))}


def gate_cell(ln, meta, gate):
    """What the stored draws of one (line, r) cell say about one real covariate."""
    reps = ln["replicates"]
    n = len(reps["tl"])
    fo = ln.get("folds_on_reps") or [None] * n
    draws = [draw_conditions(reps["tl"][i], reps["mix"][i], reps["r1b"][i], reps["r1p"][i], fo[i],
                             ln["pass_reps"][i], gate) for i in range(n)]
    k = sum(d["pass"] for d in draws)
    tl_ok = sum(d["tl"] for d in draws)
    out = {"r": meta["r"], "r_within_pair_tl": r6(meta.get("r_within_pair_tl")), "draws": n}
    for reg in ("tl", "mix"):
        v = np.asarray(reps[reg], float)
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        avg = ln["regimes"][reg]
        out[reg] = {"mean": r6(mean), "draw_sd": r6(sd), "cluster_se": r6(avg["cluster_se"]),
                    "worst_parent_averaged_line": r6(avg.get("worst_parent")),
                    "draws_values": [r6(x) for x in v],
                    "p_normal": r4(normal_pass(mean, sd, gate["tl"])),
                    "p_t_predictive": r4(t_predictive_pass(mean, sd, n, gate["tl"]))}
    out["tl"].update({
        "pass_count": k, "pass_rate": r4(k / n), "pass_jeffreys95": ci4(jeffreys(k, n)),
        "draws_clearing_bar": tl_ok,
        "failed": {"tl": sum(not d["tl"] for d in draws),
                   "selection_on": sum(not d["selection_on"] for d in draws),
                   "mix_same_sign": sum(not d["mix_same_sign"] for d in draws),
                   "guard": sum(not d["guard"] for d in draws),
                   "worst_parent_inferred": sum(d["worst_parent_inferred"] is False for d in draws),
                   "worst_parent_unknown": sum(d["worst_parent_inferred"] is None for d in draws)},
        "p_full_gate_model": r4(normal_pass(float(np.mean(reps["tl"])), float(np.std(reps["tl"], ddof=1)),
                                            gate["tl"]) * (k / tl_ok) if tl_ok else 0.0),
        "stored_pass_draws": f"{k}/{n}"})
    mk = sum(d["mix_reading_known"] for d in draws)
    out["mix"].update({"pass_count_upper": mk, "pass_rate_upper": r4(mk / n),
                       "note": "selection kept on test-like runs (SELECT_ON); bar -0.002 read on mix/whole; "
                               "its worst-parent condition is not stored per draw, so the count is an upper "
                               "bound"})
    out["per_draw"] = draws
    return out


def interp_r(cells, reg, p_target, gate):
    """Smallest r (0.005 steps, mean and draw sd interpolated linearly between the
    table's rows) at which one draw clears -0.002 on `reg` with probability
    p_target under the normal model."""
    rs = np.array([c["r"] for c in cells])
    mu = np.array([c[reg]["mean"] for c in cells])
    sd = np.array([c[reg]["draw_sd"] for c in cells])
    for r in np.arange(rs.min(), rs.max() + 1e-9, 0.005):
        m, s = np.interp(r, rs, mu), np.interp(r, rs, sd)
        if normal_pass(m, s, gate["tl"]) >= p_target:
            return round(float(r), 3)
    return None


def stage_gate(args):
    ht = load_json(HT)
    gate = ht["meta"]["gate"]
    out = {"source": "results/harness_thresholds.json, tables.honest (8 noise draws a cell)",
           "gate": gate, "lines": {}}
    for line in LINES:
        cells = []
        for key, t in sorted(ht["tables"]["honest"].items(), key=lambda kv: kv[1]["meta"]["r"]):
            cells.append(gate_cell(t["lines"][line], t["meta"], gate))
        out["lines"][line] = {
            "cells": {f"r={c['r']:g}": c for c in cells},
            "r_for_probability": {reg: {str(p): interp_r(cells, reg, p, gate) for p in PROBS}
                                  for reg in ("tl", "mix")},
            "asked": {f"r={r:g}": {k: (c["tl"][k] if k in c["tl"] else None)
                                   for k in ("pass_count", "pass_jeffreys95", "p_normal", "p_t_predictive",
                                             "p_full_gate_model", "mean", "draw_sd")}
                      | {"mix_p_normal": c["mix"]["p_normal"], "mix_pass_count_upper": c["mix"]["pass_count_upper"]}
                      for r in R_ASKED for c in cells if abs(c["r"] - r) < 1e-9}}
    T = out["lines"]["transferred nested"]["cells"]
    P = out["lines"]["per-pair nested"]["cells"]
    checks = [
        ("draft §4.5: transferred, r = 0.3 passes on 6 of 8 draws", T["r=0.3"]["tl"]["pass_count"] == 6),
        ("draft §4.5: transferred, r = 0.4 passes on 7 of 8 draws", T["r=0.4"]["tl"]["pass_count"] == 7),
        ("draft §4.5: transferred, every draw passes from r = 0.5",
         all(T[k]["tl"]["pass_count"] == 8 for k in ("r=0.5", "r=0.7"))),
        ("draft §4.5: per-pair, no draw passes at r = 0.4, every draw at 0.5",
         P["r=0.4"]["tl"]["pass_count"] == 0 and P["r=0.5"]["tl"]["pass_count"] == 8),
        ("draft §4.5: draw sd 0.0007 against cluster SE 0.0003 at r = 0.3",
         round(T["r=0.3"]["tl"]["draw_sd"], 4) == 0.0007 and round(T["r=0.3"]["tl"]["cluster_se"], 4) == 0.0003),
        ("draft §4.5: at r = 0.2, mix/whole 0.0016, public 0.0019 to 0.0025, test-like 0.0008",
         round(-T["r=0.2"]["mix"]["mean"], 4) == 0.0016 and round(-T["r=0.2"]["tl"]["mean"], 4) == 0.0008),
    ]
    out["draft_checks"] = [{"claim": c, "holds": bool(ok)} for c, ok in checks]
    per = ht["meta"].get("wall_s")
    out["full_mix_gate"] = {
        "storable_now": "the bar read on mix/whole with selection on test-like runs (above); the "
                        "mix/whole worst parent of the averaged line only",
        "not_stored": "each draw's worst parent on mix/whole, and any selection made on mix/whole",
        "run_needed": "experiments/harness.py --stage table with SELECT_ON = 'mix' and gate() reading "
                      "the mix/whole estimate and worst parent, with average_lines() keeping each draw's "
                      "worst parent; the rows exist (data/harness_rows/mix, 150 runs), so only the table "
                      "stage reruns: the honest table's 7 rows x 8 draws = 56 covariates at the recorded "
                      "~27 s each, about 25 minutes on one process",
        "table_wall_s_recorded": per}
    # B1: what a centred covariate does at B1, from the stored per-budget lines
    acc = ht["acceptance"]["oracle_honest"]
    b1 = {"honest_oracle_transferred_B1_brier": r6(acc["transferred nested"]["regimes"]["tl"]["by_budget"][1]),
          "honest_oracle_transferred_B1_cluster_se": r6(
              acc["transferred nested"]["regimes"]["tl"]["by_budget_cluster_se"][1]),
          "honest_oracle_per_pair_B1_brier": r6(acc["per-pair nested"]["regimes"]["tl"]["by_budget"][1]),
          "honest_oracle_b0_term_alc": r6(acc["b0 nested"]["regimes"]["tl"]["est"]),
          "transferred_by_budget_at_r": {
              f"r={t['meta']['r']:g}": [r6(v) for v in t["lines"]["transferred nested"]["regimes"]["tl"]["by_budget"]]
              for t in ht["tables"]["honest"].values() if t["meta"]["r"] in (0.2, 0.3, 0.4, 0.5)},
          "per_pair_by_budget_at_r=0.5": [r6(v) for v in
                                          ht["tables"]["honest"]["r=0.5"]["lines"]["per-pair nested"]["regimes"]["tl"]["by_budget"]]}
    b1["draft_sentence_checked"] = (
        "§6.9 item 1: 'At B1 a per-pair slope carries nothing, because centring zeroes the pair's single "
        "label. A slope transferred from other benchmarks does act there: the honest oracle's transferred "
        "line gains 0.036 of Brier at B1.'")
    b1["holds"] = bool(round(-b1["honest_oracle_transferred_B1_brier"], 3) == 0.036
                       and b1["honest_oracle_per_pair_B1_brier"] == 0)
    out["b1"] = b1
    return out


# --- when the gate was fixed (P1.12) ------------------------------------------------------------

def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=False).stdout.strip()


def first_commit(path=None, pickaxe=None):
    args = ["log", "--format=%h %aI", "--reverse"]
    if pickaxe:
        args += ["-S", pickaxe]
    if path:
        args += ["--diff-filter=A", "--", path] if not pickaxe else ["--", path]
    out = git(*args).splitlines()
    if not out:
        return None
    h, t = out[0].split(" ", 1)
    return {"commit": h, "time": t}


def stage_evidence(args):
    from datetime import datetime, timezone
    gate_c = first_commit("experiments/harness.py", 'GATE = {"tl": -0.002')
    gt = datetime.fromisoformat(gate_c["time"]).astimezone(timezone.utc) if gate_c else None
    sle = load_json(os.path.join(RES, "strong_llm_eval.json"))
    tl_main = (sle.get("run") or {}).get("timeline") or {}
    tl_ent = ((sle.get("entropy") or {}).get("run") or {}).get("timeline") or {}
    data_times = {
        "results/strong_llm_eval.json (rubric and attempts)": tl_main.get("rubric_first_write_utc"),
        "results/strong_llm_eval.json (entropy, commit D)": tl_ent.get("entropy_first_write_utc"),
    }
    rule_d = first_commit("experiments/strong_llm_eval.py", 'ENTROPY_PRIMARY = "ent_first1024"')
    studies = ["results/harness_thresholds.json", "results/itemsig_eval.json", "results/itemcov_eval.json",
               "results/subject_side.json", "results/llm4b_close.json",
               "results/attempt_probe.json", "results/hidden_state_probe.json", "results/icl_probe.json",
               "results/finetune_encoder.json", "results/heads_eval.json", "results/strong_llm_eval.json"]
    rows = []
    for s in studies:
        fc = first_commit(s)
        row = {"results": s, "first_commit": fc}
        if fc and gt:
            t = datetime.fromisoformat(fc["time"]).astimezone(timezone.utc)
            row["relation"] = ("committed together with the gate" if fc["commit"] == gate_c["commit"] else
                               "committed before the gate existed in any commit" if t < gt else
                               "committed after the gate was committed")
        rows.append(row)
    for k, v in data_times.items():
        if v and gt:
            t = datetime.strptime(v, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            rows.append({"results": k, "data_first_written_utc": v,
                         "relation": ("data produced after the gate was committed (by "
                                      f"{(t - gt).total_seconds() / 3600:.1f} h)") if t > gt else
                         "data produced before the gate was committed"})
    return {
        "gate_first_commit": gate_c,
        "gate_first_commit_utc": gt.isoformat() if gt else None,
        "entropy_rule_first_commit": rule_d,
        "studies": rows,
        "reading": ("The gate (GATE in experiments/harness.py) first appears in a commit together with "
                    "harness_thresholds.json, itemcov_eval.json and subject_side.json, so for those the "
                    "repository shows only that gate and results were committed at once. itemsig_eval.json "
                    "was committed a day before any commit holds the gate. For the 14B's rubric and attempts "
                    "and for the entropy job (commit D), the gate was in a commit before their data were "
                    "produced on Kaggle; commit D's reading rule, which keeps the feature only if a nested "
                    "harness line passes harness.gate, was also committed before its data. The studies "
                    "committed in 4d2cc4f and later were scored after the gate's commit, but the repository "
                    "does not record when their inputs were produced. None of this is a pre-registration: "
                    "no gate was registered with a third party or committed before every study it judged."),
    }


# --- honest difficulty: targets, groups, reliability --------------------------------------------

class Honest:
    """The honest targets the harness and the language-model studies use
    (data/strong_llm_eval/targets.json: per parent the fold-averaged map 'mean'
    and the five fold maps), item groups, and per-parent arrays."""

    def __init__(self):
        from experiments import itemcov_eval as ICE
        from paiec.hier import parse_features
        t = load_json(TARGETS)
        self.digest = t["digest"]
        self.mean = t["mean"]
        self.folds = t["folds"]
        self.items = {q: sorted(self.mean[q]) for q in PARENTS}
        self.parent_of = {k: q for q in PARENTS for k in self.items[q]}
        self.group_of = {}
        for q in PARENTS:
            it = ICE.load_items(q)
            gk = ICE.GROUP_KEY[q]
            for k in self.items[q]:
                f = (it.get(k) or {}).get("item_features", "")
                self.group_of[k] = parse_features(f).get(gk, "") if f else ""
        self.Z, self.has, self.zm = {}, {}, {}
        for q in PARENTS:
            ks = self.items[q]
            Z = np.array([[self.folds[f].get(k, np.nan) for k in ks] for f in range(len(self.folds))], float)
            self.has[q] = np.isfinite(Z)
            self.Z[q] = np.where(self.has[q], Z, 0.0)
            self.zm[q] = np.array([self.mean[q][k] for k in ks], float)


def fold_overlap_reliability(H_):
    out = {}
    for q in PARENTS:
        Z, has = H_.Z[q], H_.has[q]
        cs = []
        F = Z.shape[0]
        for a in range(F):
            for b in range(a + 1, F):
                m = has[a] & has[b]
                cs.append(float(np.corrcoef(Z[a, m], Z[b, m])[0, 1]))
        c = float(np.mean(cs))
        rel_h, rel_all = rel_from_fold_overlap(c, F)
        r_fm = [float(np.corrcoef(Z[f, has[f]], H_.zm[q][has[f]])[0, 1]) for f in range(F)]
        out[q] = {"items": len(H_.items[q]), "mean_pairwise_fold_r": r4(c), "rel_honest": r4(rel_h),
                  "rel_all": r4(rel_all), "kappa": r4(math.sqrt(rel_h / rel_all)) if rel_all else None,
                  "r_fold_vs_fold_mean": r4(float(np.mean(r_fm)))}
    return out


def split_half_reliability(H_, splits=HALF_SPLITS):
    """Rasch refits (testlike.rasch, the harness's) on random halves of each
    parent's subjects; Spearman-Brown to four fifths (a fold-honest fit) and to all
    subjects. Also refits the five folds and compares them with targets.json."""
    from paiec import data as D
    from paiec import official as O
    from paiec import testlike as T
    from paiec.evaluator import stable_hash
    from experiments import harness as Hm
    out = {}
    for q in PARENTS:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = O.eligible(D.load_pairs([q]))
        # the stored fold maps, refitted
        dmax = 0.0
        for f in range(HONEST_FOLDS):
            hf = T.rasch([p for p in ps if Hm.fold_of(p.subject_id) != f])
            st = H_.folds[f]
            common = [k for k in hf if k in st]
            if common:
                dmax = max(dmax, float(np.max(np.abs(np.array([hf[k] for k in common])
                                                     - np.array([st[k] for k in common])))))
        rhos = []
        for s in range(splits):
            half = [stable_hash(HALF_SALT, "half", s, p.subject_id) % 2 for p in ps]
            a = T.rasch([p for p, h in zip(ps, half) if h == 0])
            b = T.rasch([p for p, h in zip(ps, half) if h == 1])
            common = sorted(set(a) & set(b))
            if len(common) >= 20:
                rhos.append(float(np.corrcoef([a[k] for k in common], [b[k] for k in common])[0, 1]))
        rho = float(np.mean(rhos))
        rel_h, rel_all = spearman_brown(rho, 1.6), spearman_brown(rho, 2.0)
        out[q] = {"subjects": len(ps), "splits": len(rhos), "half_r_mean": r4(rho),
                  "half_r_sd": r4(float(np.std(rhos, ddof=1))), "rel_honest": r4(rel_h), "rel_all": r4(rel_all),
                  "kappa": r4(math.sqrt(rel_h / rel_all)), "ceiling_r_honest": r4(math.sqrt(rel_h)),
                  "targets_json_refit_max_abs_diff": r6(dmax), "wall_s": round(time.time() - t0, 1)}
        log(f"reliability {q}: half r {rho:.3f}, rel honest {rel_h:.3f}, all {rel_all:.3f}, "
            f"refit max diff {dmax:.2g}")
        del ps
    return out


# --- a covariate on the harness's two scales ----------------------------------------------------

class Within:
    """Test-like pair appearances of the harness rows, for the within-pair r."""

    def __init__(self, H_):
        from experiments import harness as Hm
        self.Hm = Hm
        rows, keys = Hm.load_rows(Hm.ROWS, ["tl"])
        self.rows, self.keys = rows, keys
        self.R = rows["tl"]
        self.items_bench = Hm.benchmark_items(rows, keys)
        self.folds = H_.folds
        R = self.R
        self.app = []
        Zk = np.array([[f.get(k, np.nan) for k in keys] for f in self.folds], float)
        for a in range(R.A):
            idx = R.ev_k[R.ev_a == a]
            if len(idx) < 5:
                continue
            f = Hm.fold_of(R.sid[a])
            self.app.append((a, idx, Zk[f, idx]))
        self.n = len(self.app)

    def eval_x(self, raw, name):
        cov, info = self.Hm.eval_covariate(raw, self.items_bench, self.keys, name, standardise=True)
        return cov.X[0], info

    def r(self, X, boots=BOOTS):
        rs, cl, varies = [], [], 0
        R = self.R
        for a, idx, z in self.app:
            x = X[idx]
            if x.std() > 0:
                varies += 1
                ok = np.isfinite(z)
                if ok.sum() >= 5 and x[ok].std() > 0 and z[ok].std() > 0:
                    rs.append(np.corrcoef(x[ok], z[ok])[0, 1])
                    cl.append(R.cluster[a])
        m, ci = cluster_mean_ci(rs, cl, boots)
        return {"r": r6(m), "ci_cluster": ci4(ci), "appearances": len(rs),
                "share_varying": r4(varies / max(self.n, 1)), "clusters": len(set(cl))}


def parent_scale(H_, xmap, boots=BOOTS, seed=SEED):
    """r_honest per parent (fold-averaged Pearson against fold-specific honest
    difficulty, items carrying x) with group and item bootstrap intervals,
    r_vs_mean, r_parent_filled (x at its mean where missing, as the harness reads
    a covariate's absent items), and the mean over parents with a stratified
    group-bootstrap interval and the SE across parents."""
    per, boots_by_parent = {}, {}
    for qi, q in enumerate(PARENTS):
        ks = H_.items[q]
        x = np.array([xmap.get(k, np.nan) for k in ks], float)
        ok = np.isfinite(x)
        if ok.sum() < MIN_ITEMS:
            continue
        Z, has = H_.Z[q][:, ok], H_.has[q][:, ok]
        xo = x[ok]
        g = np.array([H_.group_of[k] or f"item|{k}" for k in ks], object)[ok]
        est = float(fold_r(np.ones(ok.sum()), xo, Z, has)[0])
        Wg = boot_weights(ok.sum(), g, boots, [seed, qi])
        bg = fold_r(Wg, xo, Z, has)
        bi = fold_r(boot_weights(ok.sum(), None, boots, [seed, qi, 1]), xo, Z, has)
        xf = np.where(ok, x, np.nanmean(x))
        filled = float(fold_r(np.ones(len(ks)), xf, H_.Z[q], H_.has[q])[0])
        G = len(set(g.tolist()))
        per[q] = {"items": int(ok.sum()), "coverage": r4(ok.mean()), "groups": G,
                  "r_honest": r4(est), "ci_group": ci4(pct_ci(bg)) if G >= 5 else None,
                  "ci_item": ci4(pct_ci(bi)),
                  "r_vs_mean": r4(float(wpearson(np.ones(ok.sum()), xo, H_.zm[q][ok])[0])),
                  "r_parent_filled": r4(filled)}
        boots_by_parent[q] = bg if G >= 5 else bi
    out = {"parents": per}
    if per:
        vals = np.array([per[q]["r_honest"] for q in per])
        st = np.nanmean(np.vstack([boots_by_parent[q] for q in per]), 0)
        out["mean_over_parents"] = {
            "parents": len(per), "r_honest": r4(vals.mean()),
            "se_across_parents": r4(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else None,
            "ci_stratified_group": ci4(pct_ci(st)),
            "r_vs_mean": r4(float(np.mean([per[q]["r_vs_mean"] for q in per])))}
    return out


def covariate_scales(H_, WI, name, xmap, stored_within=None):
    t0 = time.time()
    out = {"orientation": "+ = harder", "parent_scale": parent_scale(H_, xmap)}
    if WI is not None:
        X, info = WI.eval_x(xmap, name)
        w = WI.r(X)
        if stored_within is not None:
            w["stored"] = r6(stored_within)
            w["matches_stored"] = bool(w["r"] is not None and abs(w["r"] - stored_within) < 1e-6)
        out["within_pair"] = w
    out["wall_s"] = round(time.time() - t0, 1)
    mp = out["parent_scale"].get("mean_over_parents") or {}
    log(f"  {name}: r_honest {mp.get('r_honest')} (vs mean {mp.get('r_vs_mean')}), within "
        f"{(out.get('within_pair') or {}).get('r')}")
    return out


# --- covariate families ---------------------------------------------------------------------------

def family_ratings(H_, WI):
    """The blind ratings (experiments/llm_rating): the quoted Pearsons with
    Fisher, item- and group-bootstrap intervals; the harness r of the rated items.
    No within-pair r: 45 rated items a benchmark leave a test-like pair a handful."""
    from experiments.llm_rating.ratings_control import R as R_CTRL
    from experiments.llm_rating.ratings_main import R as R_MAIN
    from experiments import itemcov_eval as ICE
    from paiec.hier import parse_features
    main = load_json(os.path.join(RES, "rate2_truth.json"))
    ctrl = load_json(os.path.join(RES, "ctrl_truth.json"))
    groups = {}
    for b in sorted({d["bench"] for d in main} | {"multi_swebench"}):
        it = ICE.load_items(b)
        gk = ICE.GROUP_KEY.get(b)
        groups[b] = {k: (parse_features(v.get("item_features", "")).get(gk, "") if gk else "")
                     for k, v in it.items()}
    quoted = {}
    for b in sorted({d["bench"] for d in main}):
        ds = [d for d in main if d["bench"] == b]
        x = np.array([R_MAIN[d["uid"]] for d in ds], float)
        z = np.array([d["zb"] for d in ds], float)
        g = [groups[b].get(d["key"], "") for d in ds]
        c = corr_with_ci(x, z, g if b in ICE.GROUP_KEY else None)
        c["spearman"] = corr_with_ci(x, z, g if b in ICE.GROUP_KEY else None, kind="spearman")
        quoted[b] = c
    # pooled within benchmark, as analysis.py
    import pandas as pd
    df = pd.DataFrame([{"b": d["bench"], "x": R_MAIN[d["uid"]], "z": d["zb"],
                        "g": d["bench"] + "|" + (groups[d["bench"]].get(d["key"], "") or "item|" + d["key"])}
                       for d in main])
    gb = df.groupby("b")
    xx = ((df.x - gb.x.transform("mean")) / gb.x.transform("std")).to_numpy()
    zz = ((df.z - gb.z.transform("mean")) / gb.z.transform("std")).to_numpy()
    quoted["pooled_within_benchmark"] = corr_with_ci(xx, zz, df.g.to_numpy())
    quoted["pooled_within_benchmark"]["note"] = ("analysis.py's interval uses n - 3 twice (ci(r, len(df) - 3)); "
                                                 "ci_fisher here uses n")
    xc = np.array([R_CTRL[d["uid"]] for d in ctrl], float)
    zc = np.array([d["zb"] for d in ctrl], float)
    gc = [groups["multi_swebench"].get(d["key"], "") for d in ctrl]
    quoted["control_multi_swebench_full_text"] = corr_with_ci(xc, zc, gc)
    quoted["control_multi_swebench_full_text"]["spearman"] = corr_with_ci(xc, zc, gc, kind="spearman")
    xmap = {d["key"]: float(R_MAIN[d["uid"]]) for d in main if d["bench"] in PARENTS}
    cov = {"blind_rating": {"orientation": "+ = harder (0-100 difficulty rating)",
                            "parent_scale": parent_scale(H_, xmap),
                            "within_pair": None,
                            "within_pair_not_computed": "45 rated items a benchmark: a test-like pair's "
                                                        "evaluated items hold a handful, the rest read as the "
                                                        "mean, so a within-pair r would describe the zeros"}}
    return {"quoted": quoted, "covariates": cov,
            "target": "zb in results/rate2_truth.json and ctrl_truth.json (Rasch difficulty, the study's own "
                      "fit), as the draft's §6.2 table"}


def family_judge4b(H_, WI):
    """Qwen3-4B judge (experiments/llm4b_close.py): stored correlation blocks with
    group/item intervals; both harness scales, within-pair checked against the
    stored r_within_pair_tl."""
    from experiments import llm4b_close as L4
    st = load_json(os.path.join(RES, "llm4b_close.json"))
    maps = L4.feature_maps(L4.FEAT_DIR)
    cov = {}
    for f, sign in L4.FEATURES.items():
        s = sign or 1
        raw = maps[f]
        stored = (st.get("harness", {}).get(f) or {}).get("r_within_pair_tl")
        # the stored within-pair r was taken on the raw (unoriented) map
        res = covariate_scales(H_, WI, f"4b {f}", raw, stored)
        if s < 0:
            ps = res["parent_scale"]
            for q, v in ps["parents"].items():
                for k in ("r_honest", "r_vs_mean", "r_parent_filled"):
                    v[k] = r4(-v[k])
                for k in ("ci_group", "ci_item"):
                    if v[k]:
                        v[k] = [r4(-v[k][1]), r4(-v[k][0])]
            mp = ps.get("mean_over_parents")
            if mp:
                mp["r_honest"], mp["r_vs_mean"] = r4(-mp["r_honest"]), r4(-mp["r_vs_mean"])
                if mp["ci_stratified_group"]:
                    mp["ci_stratified_group"] = [r4(-mp["ci_stratified_group"][1]), r4(-mp["ci_stratified_group"][0])]
            w = res.get("within_pair")
            if w and w["r"] is not None:
                w["r_oriented"] = r6(-w["r"])
                w["ci_cluster_oriented"] = [r4(-w["ci_cluster"][1]), r4(-w["ci_cluster"][0])] if w["ci_cluster"] else None
        res["declared_sign"] = sign
        res["orientation"] = "declared sign applied to the parent scale (+ = harder); within_pair.r is the raw map's"
        cov[f"4b {f}"] = res
    feats = st["signs"]["features"]
    quoted = {}
    for f in L4.FEATURES:
        u = feats[f]["units"]
        quoted[f] = {unit: {s: u[unit].get(s) for s in ("spearman_within", "partial_spearman", "partial2_spearman",
                                                         "pearson", "spearman")}
                     | {"n": u[unit].get("n"), "groups": u[unit].get("groups")}
                     for unit in ("matharena", "matharena text-bearing", "multi_swebench") if unit in u}
    return {"quoted": quoted, "covariates": cov,
            "source": "results/llm4b_close.json signs.features (stored, 2,000 resamples over groups and items)"}


def family_strong14b(H_, WI):
    """Qwen3-14B rubric, judged solve share, heads and attempts
    (experiments/strong_llm_eval.py): the exact harness maps (x digests checked
    against the stored ones); stored correlation blocks quoted."""
    from experiments import strong_llm_eval as SLE
    state = load_json(SLE.OUT)
    joined = SLE.load_joined(SLE.WORK)
    reg = SLE.stage_registry(joined, state)
    want = ["head rubric_ridge", "head rubric_ridge_posfree", "head rubric_all_ridge", "head judge_ridge",
            "solve_share", "rubric_sum", "rubric_reasoning", "rubric_knowledge", "rubric_work",
            "rubric_interaction", "rubric_volume", "rubric_atypicality", "rubric_precision",
            "rubric_unguessability", "time_log_minutes", "att_cot_tok_entropy"]
    maps = SLE.harness_maps(SLE.WORK, joined, reg, want)
    cov, digests = {}, {}
    for name in want:
        if name not in maps:
            cov[name] = {"missing": "not in the harness maps"}
            continue
        sh = state["harness"].get(name) or {}
        dig = SLE.maps_digest({name: maps[name]})
        digests[name] = {"x_digest": dig, "stored": sh.get("x_digest"), "match": dig == sh.get("x_digest")}
        cov[f"14b {name}"] = covariate_scales(H_, WI, f"14b {name}", maps[name], sh.get("r_within_pair_tl"))
    heads = {h: {"per_parent": {q: {s: v.get(s) for s in ("pearson", "pearson_within", "spearman_within")}
                                | {"n": v.get("n"), "groups": v.get("groups")}
                                for q, v in state["heads"][h]["per_parent"].items()},
                 "random_effects": state["heads"][h]["random_effects"],
                 "random_effects_within_group": state["heads"][h]["random_effects_within_group"],
                 "mean_pearson": state["heads"][h]["mean_pearson"]}
             for h in state["heads"]}
    feats = state["signs"]["features"]
    scales = {}
    for f in ("solve_share", "rubric_sum"):
        scales[f] = {u: {s: v.get(s) for s in ("pearson", "spearman_within")} | {"n": v.get("n")}
                     for u, v in feats[f]["units"].items() if isinstance(v, dict) and v.get("n")}
        scales[f]["random_effects"] = feats[f].get("random_effects")
        scales[f]["random_effects_within"] = feats[f].get("random_effects_within")
    att = {"probe_texts": state["attempts"]["probe_only"]["decision"].get("best"),
           "probe_rule": state["attempts"]["probe_only"]["decision"].get("rule"),
           "all_attempted": state["attempts"]["decision"].get("best"),
           "all_attempted_2026": state["attempts"]["decision"].get("best_2026"),
           "note": "the draft quotes 0.372 [0.209, 0.516] on the 147 probe texts for the primary feature "
                   "(F§ 'The attempts: GO'); the probe-only decision's best is another feature"}
    return {"quoted": {"heads": heads, "solve_share_and_rubric_sum": scales, "attempts": att},
            "covariates": cov, "x_digests": digests,
            "source": "results/strong_llm_eval.json (stored intervals: 2,000 resamples over groups and items; "
                      "random effects across the four parents)"}


def family_entropy(H_, WI):
    from experiments import strong_llm_eval as SLE
    state = load_json(SLE.OUT)
    est = SLE.entropy_state(state)
    joined, _ = SLE.entropy_work_checked(SLE.WORK, est)
    reg = SLE.entropy_stage_registry(joined, est)
    names = ["ent_first1024", "ent_first256"]
    maps = SLE.entropy_harness_maps(joined, reg, names)
    cov, digests = {}, {}
    for name in names:
        sh = est["harness"].get(name) or {}
        dig = SLE.maps_digest({name: maps[name]})
        digests[name] = {"x_digest": dig, "stored": sh.get("x_digest"), "match": dig == sh.get("x_digest")}
        cov[f"entropy {name}"] = covariate_scales(H_, WI, f"entropy {name}", maps[name],
                                                  sh.get("r_within_pair_tl"))
    f = est["signs"]["features"]["ent_first1024"]
    quoted = {u: {s: v.get(s) for s in ("pearson", "spearman_within", "pearson_within", "partial2_spearman")}
              | {"n": v.get("n"), "groups": v.get("groups")}
              for u, v in f["units"].items() if isinstance(v, dict) and v.get("n")}
    quoted["random_effects_within"] = f.get("random_effects_within")
    quoted["random_effects"] = f.get("random_effects")
    quoted["test_retest"] = (est.get("consistency") or {}).get("test_retest")
    return {"quoted": {"ent_first1024": quoted}, "covariates": cov, "x_digests": digests,
            "source": "results/strong_llm_eval.json entropy.signs (stored intervals)"}


def _within_preds(ET, df, feats, learner, splitter="kfold"):
    """experiments/emb_transfer.py's within(), returning its predictions too."""
    from sklearn.model_selection import GroupKFold, KFold
    out, preds = {}, np.full(len(df), np.nan)
    for hb in ET.BENCH:
        idx = np.flatnonzero(df.bench.values == hb)
        z = df.z.values[idx]
        g = df.group.values[idx]
        valid = np.isfinite(z)
        idx, z, g = idx[valid], z[valid], g[valid]
        pred = np.zeros(len(idx))
        if splitter == "group":
            ng = len(np.unique(g))
            if ng < 3:
                out[hb] = None
                continue
            folds = list(GroupKFold(min(5, ng)).split(idx, groups=g))
        else:
            folds = list(KFold(5, shuffle=True, random_state=0).split(idx))
        for tr, te in folds:
            Ftr, Fte = feats(idx[tr], idx[te])
            inner = []
            for itr, ite in KFold(5, shuffle=True, random_state=1).split(tr):
                inner.append((learner(Ftr[itr], z[tr][itr], Ftr[ite], None), z[tr][ite]))
            hp, _ = ET.pick(inner)
            pred[te] = learner(Ftr, z[tr], Fte, None)[hp]
        out[hb] = ET.scores(pred, z)
        preds[idx] = pred
    return out, preds


def _group_only_preds(ET, df):
    import pandas as pd
    from sklearn.model_selection import KFold
    out, preds = {}, np.full(len(df), np.nan)
    for hb in ET.BENCH:
        idx = np.flatnonzero(df.bench.values == hb)
        d = df.iloc[idx].reset_index(drop=True)
        z, g = d.z.values, d.group.values
        pred = np.zeros(len(d))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(d):
            m = pd.Series(z[tr]).groupby(g[tr]).mean()
            pred[te] = pd.Series(g[te]).map(m).fillna(0.0).values
        out[hb] = ET.scores(pred, z)
        preds[idx] = pred
    return out, preds


def family_embeddings(H_, WI):
    """Qwen3-Embedding-0.6B (experiments/emb_transfer.py), refitted to keep its
    predictions: LOBO ridge (benchmark-centred) and kNN (raw), within-benchmark
    5-fold ridge, the group mean alone, and ridge with whole groups held out. Point
    estimates and LOBO intervals are checked against results/emb_transfer.json."""
    from experiments import emb_transfer as ET
    from paiec import llmfeat as F
    stored = load_json(os.path.join(RES, "emb_transfer.json"))
    import pandas as pd
    dfs, Es = [], []
    for b in ET.BENCH:
        d, E, _, _ = ET.load_bench(b)
        dfs.append(d)
        Es.append(E)
    df = pd.concat(dfs, ignore_index=True)
    E = np.vstack(Es)
    del Es
    bench = df.bench.values
    Ec = ET.centre_rows(E, bench)
    ridge = lambda Ftr, ytr, Fte, w: ET.ridge_path(Ftr, ytr, Fte, w)  # noqa: E731
    knn = lambda Ftr, ytr, Fte, w: ET.knn_path(Ftr, ytr, Fte)  # noqa: E731
    emb_raw = lambda tr, te: (E[tr], E[te])  # noqa: E731
    emb_c = lambda tr, te: (Ec[tr], Ec[te])  # noqa: E731
    specs = {"lobo emb_ridge_centred": ("lobo", emb_c, ridge), "lobo emb_knn_raw": ("lobo", emb_raw, knn),
             "within emb_ridge": ("within", emb_raw, ridge),
             "within_groupcv emb_ridge": ("within_groupcv", emb_raw, ridge)}
    preds, quoted, checks = {}, {}, []
    for name, (kind, f, l) in specs.items():
        if kind == "lobo":
            res, p = ET.lobo(df, f, l)
        else:
            res, p = _within_preds(ET, df, f, l, "group" if kind == "within_groupcv" else "kfold")
        preds[name] = p
        sec, key = name.split(" ")
        st = stored[sec][key]
        for b in ET.BENCH:
            if res.get(b) is None:
                continue
            checks.append({"what": f"{name} {b} pearson", "got": r6(res[b]["pearson"]),
                           "stored": r6(st[b]["pearson"]), "ok": abs(res[b]["pearson"] - st[b]["pearson"]) < 1e-6})
            if kind == "lobo":
                checks.append({"what": f"{name} {b} group CI", "got": ci4(res[b]["pearson_ci_groupboot"]),
                               "stored": ci4(st[b]["pearson_ci_groupboot"]),
                               "ok": np.allclose(res[b]["pearson_ci_groupboot"], st[b]["pearson_ci_groupboot"],
                                                 atol=1e-9)})
        quoted[name] = {}
        for b in ET.BENCH:
            m = (bench == b) & np.isfinite(p)
            if not m.any():
                continue
            c = corr_with_ci(p[m], df.z.values[m], df.group.values[m])
            if kind == "lobo":
                c["ci_group_stored_1000"] = ci4(st[b]["pearson_ci_groupboot"])
            quoted[name][b] = c
    gres, gp = _group_only_preds(ET, df)
    preds["group_only"] = gp
    quoted["group_only"] = {}
    for b in ET.BENCH:
        m = bench == b
        checks.append({"what": f"group_only {b} pearson", "got": r6(gres[b]["pearson"]),
                       "stored": r6(stored["group_only"][b]["pearson"]),
                       "ok": abs(gres[b]["pearson"] - stored["group_only"][b]["pearson"]) < 1e-6})
        quoted["group_only"][b] = corr_with_ci(gp[m], df.z.values[m], df.group.values[m])
    # per-item maps keyed by item_id: a key's prediction for every item_id it stands for
    ids_of = {}
    for b in ET.BENCH:
        index = F.load(ET.FEAT, b)[0]
        for k, ids in zip(index["key"], index["item_ids"]):
            ids_of[k] = [str(i) for i in ids]
    cov = {}
    for name in ("lobo emb_ridge_centred", "lobo emb_knn_raw", "within emb_ridge"):
        xmap = {}
        for k, v in zip(df.key, preds[name]):
            if np.isfinite(v):
                for iid in ids_of.get(k, []):
                    xmap[iid] = float(v)
        cov[f"emb {name}"] = covariate_scales(H_, WI, f"emb {name}", xmap)
    return {"quoted": quoted, "covariates": cov, "checks": checks,
            "target": "emb_transfer.py's z: Rasch difficulty (paiec.rasch, every subject), averaged over the "
                      "item_ids of a key, standardised within benchmark"}


def _tfidf_lobo(df, target, min_te=50):
    """experiments/transfer.py's item_transfer(), returning its predictions."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from paiec.items import numeric_features
    preds = np.full(len(df), np.nan)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for b in sorted(df.benchmark_id.unique()):
        tr, te = df[df.benchmark_id != b], df[df.benchmark_id == b]
        if len(te) < min_te:
            continue
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, strip_accents="unicode")
        Xtr, Xte = v.fit_transform(tr.content.fillna("")), v.transform(te.content.fillna(""))
        sc = StandardScaler().fit(numeric_features(tr))
        Xtr = sparse.hstack([Xtr, sc.transform(numeric_features(tr))]).tocsr()
        Xte = sparse.hstack([Xte, sc.transform(numeric_features(te))]).tocsr()
        m = Ridge(alpha=3.0).fit(Xtr, tr[target].values, sample_weight=np.sqrt(tr.n.values))
        preds[[pos[i] for i in te.index]] = m.predict(Xte)
    return preds


def _tfidf_within_matharena(ma, target):
    """transfer.py's within-matharena 5-fold map (text-bearing items)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    d = ma[~ma.degenerate].reset_index(drop=True)
    pred = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(d):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        X1 = v.fit_transform(d.content.iloc[tr].fillna(""))
        X2 = v.transform(d.content.iloc[te].fillna(""))
        m = Ridge(alpha=3.0).fit(X1, d[target].values[tr], sample_weight=np.sqrt(d.n.values[tr]))
        pred[te] = m.predict(X2)
    return d, pred


def family_tfidf(H_, WI):
    """The TF-IDF text map of experiments/transfer.py, refitted. The draft's
    numbers (0.14, 0.23, 0.09, -0.22, 0.16) are its first call: the target is the
    naive solve-rate logit standardised within benchmark (paiec.items.item_table's
    z, higher = easier), every item; the second call (Rasch difficulty,
    text-bearing items) is what 'to Rasch difficulty' would mean."""
    import re
    from paiec.data import load_pairs
    from paiec.items import item_table
    from paiec.rasch import item_difficulty
    from experiments import itemcov_eval as ICE
    from paiec.hier import parse_features
    t0 = time.time()
    pairs = load_pairs()
    df = item_table(pairs)
    D = item_difficulty(pairs)
    del pairs
    df["b"] = [D[bid].get(k, np.nan) for bid, k in zip(df.benchmark_id, df.item_key)]
    g = df.groupby("benchmark_id").b
    df["zb"] = (df.b - g.transform("mean")) / g.transform("std")
    c = df.content.fillna("")
    df["degenerate"] = c.str.contains("See image", case=False) | (c.str.len() < 120)
    df["group"] = [parse_features(f).get(ICE.GROUP_KEY.get(b, ""), "") if ICE.GROUP_KEY.get(b) else ""
                   for b, f in zip(df.benchmark_id, df.features.fillna(""))]
    log(f"tfidf: items and Rasch in {time.time() - t0:.0f}s")
    p_naive = _tfidf_lobo(df, "z")
    sub = df[~df.degenerate]
    p_rasch = _tfidf_lobo(sub, "zb")
    log(f"tfidf: LOBO maps in {time.time() - t0:.0f}s")
    quoted = {"naive_target_all_items": {}, "rasch_target_text_bearing": {}}
    draft = {"matharena": 0.14, "multi_swebench": 0.23, "real_webagents": 0.09, "researchcodebench": -0.22,
             "swe_rebench": 0.16}
    for b in sorted(df.benchmark_id.unique()):
        m = (df.benchmark_id == b).to_numpy() & np.isfinite(p_naive)
        if m.any():
            grp = df.group.to_numpy()[m] if b in ICE.GROUP_KEY else None
            cq = corr_with_ci(p_naive[m], df.z.to_numpy()[m], grp)
            cq["draft"] = draft.get(b)
            cq["reproduces_draft"] = bool(draft.get(b) is not None and round(cq["est"], 2) == draft[b])
            quoted["naive_target_all_items"][b] = cq
        ms = (sub.benchmark_id == b).to_numpy() & np.isfinite(p_rasch)
        if ms.any():
            grp = sub.group.to_numpy()[ms] if b in ICE.GROUP_KEY else None
            quoted["rasch_target_text_bearing"][b] = corr_with_ci(p_rasch[ms], sub.zb.to_numpy()[ms], grp)
    ma = df[df.benchmark_id == "matharena"].copy()
    ma["comp"] = ma.features.map(lambda s: (re.search(r"competition=([^;]+)", s or "") or [None, ""])[1])
    ma["resid"] = ma.z - ma.groupby("comp").z.transform("mean")
    quoted["within_matharena_5fold"] = {}
    for target, lab, want in (("z", "raw", 0.73), ("resid", "within competition", 0.49)):
        d, pred = _tfidf_within_matharena(ma, target)
        cq = corr_with_ci(pred, d[target].to_numpy(), d.comp.to_numpy())
        cq["draft"] = want
        cq["reproduces_draft"] = bool(round(cq["est"], 2) == want)
        quoted["within_matharena_5fold"][lab] = cq
    ss = ((ma.groupby("comp").zb.transform("mean") - ma.zb.mean()) ** 2).sum()
    tot = ((ma.zb - ma.zb.mean()) ** 2).sum()
    ssn = ((ma.groupby("comp").z.transform("mean") - ma.z.mean()) ** 2).sum()
    totn = ((ma.z - ma.z.mean()) ** 2).sum()
    quoted["matharena_competition_share"] = {"naive": r4(ssn / totn), "rasch": r4(ss / tot),
                                             "competitions": int(ma.comp.nunique())}
    cov = {}
    keyed = df.item_key.astype(str).to_numpy()
    cov["tfidf lobo naive"] = covariate_scales(
        H_, WI, "tfidf lobo naive",
        {k: -float(v) for k, v, b in zip(keyed, p_naive, df.benchmark_id) if np.isfinite(v) and b in PARENTS})
    cov["tfidf lobo naive"]["orientation"] = "minus the predicted solve-rate logit (+ = harder)"
    cov["tfidf lobo rasch"] = covariate_scales(
        H_, WI, "tfidf lobo rasch",
        {k: float(v) for k, v, b in zip(sub.item_key.astype(str).to_numpy(), p_rasch, sub.benchmark_id)
         if np.isfinite(v) and b in PARENTS})
    return {"quoted": quoted, "covariates": cov, "wall_s": round(time.time() - t0, 1),
            "targets": {"naive": "paiec.items.item_table z: logit((k + 0.5) / (n - k + 0.5)) standardised within "
                                 "benchmark (higher = easier); the draft's 0.14 / 0.23 / 0.09 / -0.22 / 0.16",
                        "rasch": "paiec.rasch.item_difficulty standardised within benchmark, text-bearing items"}}


def stage_scale(args):
    state = load_json(OUT) if os.path.exists(OUT) else {}
    sec = state.get("scale") or {}
    fams = args.families or list(FAMILIES)
    H_ = Honest()
    log(f"honest targets: digest {H_.digest}, items {[len(H_.items[q]) for q in PARENTS]}")
    WI = None
    if any(f != "reliability" for f in fams):
        WI = Within(H_)
        log(f"within-pair: {WI.n} test-like appearances with 5+ evaluated items")
    sec.setdefault("families", {})
    for fam in fams:
        t0 = time.time()
        log(f"family {fam}")
        if fam == "reliability":
            res = {"fold_overlap": fold_overlap_reliability(H_), "split_half": split_half_reliability(H_)}
            if WI is None:
                WI = Within(H_)
            res["within_pair_implied"] = within_pair_reliability(H_, WI, res["split_half"])
        else:
            res = {"ratings": family_ratings, "judge4b": family_judge4b, "strong14b": family_strong14b,
                   "entropy": family_entropy, "embeddings": family_embeddings, "tfidf": family_tfidf}[fam](H_, WI)
        res["wall_s"] = round(time.time() - t0, 1)
        res["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
        sec["families"][fam] = res
        sec["targets"] = {"file": os.path.relpath(TARGETS, ROOT), "sha256": sha256(TARGETS), "digest": H_.digest}
        state["scale"] = sec
        save(state)
        log(f"family {fam}: {res['wall_s']}s, maxrss {res['maxrss_gb']} GB")
    return None


def within_pair_reliability(H_, WI, split):
    """Approximate reliability of the honest difficulty within a test-like pair:
    the error variance a fold-honest fit carries ((1 - rel_honest) times its
    variance over the parent, from the split halves) against the variance of the
    honest difficulty over a pair's evaluated items, averaged over appearances."""
    out = {}
    var_parent = {q: float(np.mean([np.var(H_.Z[q][f, H_.has[q][f]]) for f in range(HONEST_FOLDS)]))
                  for q in PARENTS}
    by = {q: [] for q in PARENTS}
    R = WI.R
    for a, idx, z in WI.app:
        ok = np.isfinite(z)
        if ok.sum() >= 5:
            by[R.parent[a]].append(float(np.var(z[ok])))
    for q in PARENTS:
        if not by[q]:
            continue
        err = (1 - split[q]["rel_honest"]) * var_parent[q]
        vw = float(np.mean(by[q]))
        rel_w = 1 - err / vw if vw > 0 else None
        out[q] = {"var_parent": r4(var_parent[q]), "var_within_pair_mean": r4(vw),
                  "error_var": r4(err), "rel_within_pair": r4(rel_w),
                  "ceiling_within_pair_r": r4(math.sqrt(rel_w)) if rel_w and rel_w > 0 else None,
                  "appearances": len(by[q])}
    return out


# --- P1.13 table --------------------------------------------------------------------------------

def stage_ci(args):
    state = load_json(OUT)
    fam = state["scale"]["families"]
    gate = state.get("gate") or {}
    rows = []

    def add(where, what, quoted, stat, c, source, scale=None, note=None):
        unit = what.rsplit(", ", 2)[-2] if what.endswith(("spearman_within", "partial_spearman")) \
            else what.rsplit(", ", 1)[-1]
        if scale and unit in scale.get("per_parent", {}):
            scale = {**scale, "this_parent": scale["per_parent"][unit]}
        rows.append({"where": where, "what": what, "quoted": quoted, "statistic": stat,
                     "est": r4(c.get("est")) if c else None, "n": (c or {}).get("n"),
                     "groups": (c or {}).get("groups"),
                     "ci_group": ci4((c or {}).get("ci_group")), "ci_item": ci4((c or {}).get("ci_item")),
                     "ci_fisher": ci4((c or {}).get("ci_fisher")), "source": source,
                     "harness_scale": scale, "note": note})

    def hs(cov):
        if not cov:
            return None
        mp = cov["parent_scale"].get("mean_over_parents") or {}
        w = cov.get("within_pair") or {}
        return {"r_honest_mean": mp.get("r_honest"), "ci_stratified_group": mp.get("ci_stratified_group"),
                "se_across_parents": mp.get("se_across_parents"),
                "per_parent": {q: [v["r_honest"], v["ci_group"] or v["ci_item"]]
                               for q, v in cov["parent_scale"]["parents"].items()},
                "within_pair": w.get("r_oriented", w.get("r")),
                "within_pair_ci": w.get("ci_cluster_oriented", w.get("ci_cluster"))}

    if "tfidf" in fam:
        t = fam["tfidf"]
        for b, c in t["quoted"]["naive_target_all_items"].items():
            add("§1.2, §6 table, §6.1, §6.9", f"TF-IDF LOBO, {b}", c.get("draft"),
                "Pearson, naive solve-rate target, all items", c, "recomputed (transfer.py refitted)",
                hs(t["covariates"]["tfidf lobo naive"]) if b in PARENTS else None,
                "no item groups on swe_rebench; single subject, so no honest folds" if b == "swe_rebench" else None)
        for b, c in t["quoted"]["rasch_target_text_bearing"].items():
            add("§6.1 (what 'to Rasch difficulty' would be)", f"TF-IDF LOBO, Rasch target, text-bearing, {b}", None,
                "Pearson, Rasch target, text-bearing items", c, "recomputed (transfer.py's second call)")
        for lab, c in t["quoted"]["within_matharena_5fold"].items():
            add("§6.1", f"TF-IDF within matharena, {lab}", c.get("draft"), "Pearson, 5-fold, text-bearing", c,
                "recomputed", None, "groups are competitions")
    if "embeddings" in fam:
        e = fam["embeddings"]
        quotedv = {"lobo emb_ridge_centred": {"matharena": -0.16, "multi_swebench": -0.03, "real_webagents": -0.23,
                                              "researchcodebench": -0.02},
                   "within emb_ridge": {"matharena": 0.66, "multi_swebench": 0.27, "real_webagents": 0.38,
                                        "researchcodebench": 0.51},
                   "group_only": {"matharena": 0.57, "multi_swebench": 0.12, "real_webagents": 0.41,
                                  "researchcodebench": 0.52},
                   "within_groupcv emb_ridge": {"matharena": 0.40, "multi_swebench": 0.16, "real_webagents": -0.02,
                                                "researchcodebench": -0.09},
                   "lobo emb_knn_raw": {"matharena": -0.08, "multi_swebench": -0.01, "real_webagents": -0.02,
                                        "researchcodebench": 0.16}}
        for name, per in e["quoted"].items():
            for b, c in per.items():
                cov = e["covariates"].get(f"emb {name}")
                add("§6 table, §6.1", f"embedding {name}, {b}", quotedv.get(name, {}).get(b), "Pearson", c,
                    "recomputed (emb_transfer.py refitted; LOBO intervals also stored, 1,000 resamples)",
                    hs(cov) if cov else None)
    if "ratings" in fam:
        r = fam["ratings"]
        for b, c in r["quoted"].items():
            add("§6 table, §6.2", f"blind rating, {b}", None, "Pearson against the study's Rasch target", c,
                "recomputed from the stored ratings", hs(r["covariates"]["blind_rating"]) if b in PARENTS else None,
                c.get("note"))
    if "judge4b" in fam:
        j = fam["judge4b"]
        for f, units in j["quoted"].items():
            for u, v in units.items():
                for s in ("spearman_within", "partial_spearman"):
                    if v.get(s):
                        add("§6.2, §6.8", f"4B judge {f}, {u}, {s}", None, s, v[s] | {"n": v["n"], "groups": v["groups"]},
                            "stored (llm4b_close.json)", hs(j["covariates"].get(f"4b {f}")))
    if "strong14b" in fam:
        s = fam["strong14b"]
        for h, v in s["quoted"]["heads"].items():
            for q, pp in v["per_parent"].items():
                add("§6 table, §6.8, §6.9", f"14B head {h}, {q}", 0.19 if h == "rubric_ridge" else None,
                    "Pearson, LOBO head", pp["pearson"] | {"n": pp["n"], "groups": pp["groups"]},
                    "stored", hs(s["covariates"].get(f"14b head {h}")))
            re_ = v["random_effects"]
            rows.append({"where": "§6 table, §6.8, §6.9", "what": f"14B head {h}, random-effects mean over parents",
                         "quoted": 0.19 if h == "rubric_ridge" else None, "statistic": "Pearson, random effects",
                         "est": re_["mean"], "ci_random_effects": re_["ci"],
                         "prediction_interval": re_.get("prediction_interval"), "source": "stored"})
        for f, units in s["quoted"]["solve_share_and_rubric_sum"].items():
            for u, v in units.items():
                if isinstance(v, dict) and "pearson" in v:
                    add("§6.8 (the 14B's judged solve share)", f"14B {f}, {u}", None, "Pearson",
                        v["pearson"] | {"n": v["n"]}, "stored", hs(s["covariates"].get(f"14b {f}")))
        a = s["quoted"]["attempts"]
        rows.append({"where": "abstract, §6 table, §6.8, §6.9", "what": "14B attempts, primary, all attempted texts",
                     "quoted": 0.352, "statistic": "Spearman within competition", "est": a["all_attempted"]["rho"],
                     "ci_group": a["all_attempted"]["ci"], "n": a["all_attempted"]["n"],
                     "groups": a["all_attempted"]["groups"], "source": "stored",
                     "harness_scale": hs(s["covariates"].get("14b att_cot_tok_entropy")),
                     "note": "one parent; the 0.372 [0.209, 0.516] on the 147 probe texts is F§'s, not re-read here"})
    if "entropy" in fam:
        e = fam["entropy"]["quoted"]["ent_first1024"]
        quoted = {"matharena": 0.29, "multi_swebench": 0.12, "real_webagents": -0.03, "researchcodebench": 0.33}
        for u, v in e.items():
            if isinstance(v, dict) and "spearman_within" in v and v.get("n"):
                add("abstract, §6 table, §6.8, §6.9", f"14B entropy ent_first1024, {u}", quoted.get(u),
                    "Spearman within group", v["spearman_within"] | {"n": v["n"], "groups": v["groups"]}, "stored",
                    hs(fam["entropy"]["covariates"]["entropy ent_first1024"]) if u in PARENTS else None)
        rows.append({"where": "abstract, §6.9", "what": "14B entropy, random-effects mean within group",
                     "quoted": 0.18, "statistic": "Spearman within group, random effects",
                     "est": e["random_effects_within"]["mean"], "ci_random_effects": e["random_effects_within"]["ci"],
                     "prediction_interval": e["random_effects_within"].get("prediction_interval"), "source": "stored"})
    not_computable = [
        {"what": "TF-IDF on swe_rebench, group interval", "why": "swe_rebench items carry no item_features group"},
        {"what": "anything on swe_rebench on the harness scales",
         "why": "one subject: no subject folds, so no honest difficulty, and no test-like pairs (excluded by default)"},
        {"what": "blind ratings, within-pair r", "why": "45 rated items a benchmark; a pair holds a handful"},
        {"what": "blind ratings on researchcodebench", "why": "the study rated none"},
        {"what": "the blind ratings themselves", "why": "the rater's prompt and settings are unrecorded; "
                                                         "the correlations re-derive from the stored ratings only"},
        {"what": "4B judge on real_webagents and researchcodebench", "why": "the extraction covered matharena "
                                                                           "and 1,941 of 2,078 multi_swebench items"},
        {"what": "14B attempts beyond matharena", "why": "attempts were run on matharena texts only"},
        {"what": "group intervals with fewer than 5 groups",
         "why": "a percentile bootstrap over 1-4 clusters is degenerate; the item interval is given"},
    ]
    state["ci"] = {"rows": rows, "not_computable": not_computable,
                   "power_4b": power_4b(fam, state),
                   "unit_ratio": unit_ratios(fam, gate),
                   "note": ("ci_group: 95% percentile interval, 2,000 resamples of item_features groups (competition, "
                            "language, website, paper); ci_item: of items; ci_fisher: Fisher z with n items; stored "
                            "rows keep their study's resamples. Group intervals over 8 languages (multi_swebench) "
                            "are indicative only. harness_scale: the covariate re-read as the gate table reads r "
                            "(r_honest: Pearson against the fold-specific honest difficulty over a parent's items, "
                            "mean over parents with a stratified group interval) and within test-like pairs")}
    save(state)


#: the covariates the draft names when it quotes what share of its correlation a covariate keeps
#: within a test-like pair (the 14B's primary head, judged solve share, rubric sum and expert-time
#: estimate, and the reasoning entropy's primary)
RATIO_NAMED = ("14b head rubric_ridge", "14b solve_share", "14b rubric_sum", "14b time_log_minutes",
               "entropy ent_first1024")


def unit_ratios(fam, gate):
    """What share of its parent-scale correlation a covariate keeps within a test-like
    pair: within_pair r (oriented where the family orients it) over
    parent_scale.mean_over_parents.r_honest. The ratio is read only where the
    parent-scale stratified group interval excludes 0 ('clear'); elsewhere it is
    stored but flagged, because a quotient over a near-zero r means nothing. The
    degraded oracle's ratio is the gate table's r_within_pair_tl / r per row."""
    cov = {}
    for fname, F in fam.items():
        for name, cv in (F.get("covariates") or {}).items():
            mp = (cv.get("parent_scale") or {}).get("mean_over_parents") or {}
            w = cv.get("within_pair") or {}
            wr, ph, ci = w.get("r_oriented", w.get("r")), mp.get("r_honest"), mp.get("ci_stratified_group")
            if wr is None or not ph:
                continue
            clear = ci is not None and (ci[0] > 0 or ci[1] < 0)
            cov[name] = {"family": fname, "within_pair": r6(wr), "parent_scale": r6(ph),
                         "parent_scale_ci": ci, "parents": mp.get("parents"), "ratio": r4(wr / ph),
                         "clear": bool(clear)}
    clear = {k: v["ratio"] for k, v in cov.items() if v["clear"]}
    named = {k: cov[k]["ratio"] for k in RATIO_NAMED if k in cov}
    oracle = {}
    for k, c in ((gate.get("lines") or {}).get("transferred nested", {}).get("cells") or {}).items():
        if c.get("r") and c.get("r_within_pair_tl") is not None:
            oracle[k] = r4(c["r_within_pair_tl"] / c["r"])
    rng = (lambda d: [min(d.values()), max(d.values())] if d else None)
    return {"what": " ".join(unit_ratios.__doc__.split()), "covariates": cov,
            "named": {"covariates": named, "range": rng(named)},
            "clear": {"covariates": clear, "range": rng(clear),
                      "lowest": min(clear, key=clear.get) if clear else None,
                      "highest": max(clear, key=clear.get) if clear else None},
            "not_clear": sorted(k for k, v in cov.items() if not v["clear"]),
            "oracle": {"by_r": oracle, "at_r_0.3": oracle.get("r=0.3"), "range": rng(oracle)}}


def power_4b(fam, state):
    """Whether powering the 4B judge is still needed, from stored numbers only."""
    out = {"question": "W7/P1.13 asked for the local Qwen3-4B judge on several hundred items a benchmark, because "
                       "the blind ratings (45 a benchmark) leave r = 0.3 inside their intervals outside mathematics"}
    r = (fam.get("ratings") or {}).get("quoted", {})
    for b in ("multi_swebench", "real_webagents"):
        if b in r:
            est = r[b]["est"]
            out[f"blind_{b}"] = {"r": r4(est), "ci_fisher": ci4(r[b]["ci_fisher"]),
                                 "items_to_exclude_0.3_at_80pct_power": round(n_for_power(est, 0.3)),
                                 "items_in_benchmark": {"multi_swebench": 2126, "real_webagents": 233}[b]}
    s = fam.get("strong14b") or {}
    ss = (s.get("quoted") or {}).get("solve_share_and_rubric_sum", {}).get("solve_share", {})
    out["14b_solve_share"] = {u: {"pearson": v["pearson"]["est"], "ci_group": v["pearson"]["ci_group"],
                                  "n": v["n"]} for u, v in ss.items() if isinstance(v, dict) and "pearson" in v
                              and u in PARENTS}
    hd = (s.get("quoted") or {}).get("heads", {}).get("rubric_ridge", {})
    if hd:
        out["14b_primary_head"] = {q: {"pearson": v["pearson"]["est"], "ci_group": v["pearson"]["ci_group"]}
                                   for q, v in hd["per_parent"].items()}
    j = fam.get("judge4b") or {}
    q4 = (j.get("quoted") or {}).get("rating", {})
    if "multi_swebench" in q4:
        out["4b_rating_multi_swebench_spearman_within"] = q4["multi_swebench"]["spearman_within"]
    cov = s.get("covariates") or {}
    out["within_pair"] = {k: (v.get("within_pair") or {}).get("r") for k, v in cov.items()
                          if k in ("14b solve_share", "14b head rubric_ridge", "14b rubric_sum")}
    sle = load_json(os.path.join(RES, "strong_llm_eval.json"))
    best = sle["verdict"]["features"]
    out["14b_best_nested_tl"] = min((v.get("best_nested_tl") for v in best.values()
                                     if v.get("best_nested_tl") is not None), default=None)
    out["reading"] = (
        "Not needed. The question the larger 4B run would answer is whether a judge's rating reaches an honest "
        "r of 0.3 outside mathematics. The 14B has since rated every item of the four parents, including the same "
        "judged solve share, so the item count is no longer what limits the answer: real_webagents has 233 items "
        "and researchcodebench 212 in all, and a judge whose true r is the blind rater's 0.21 cannot be shown "
        "below 0.3 on real_webagents with every item rated. Where items are plentiful (multi_swebench, 2,126), "
        "both local judges sit far below the bar with intervals that exclude it. Where the 14B's judged solve "
        "share and primary head reach 0.3 against fold-averaged difficulty (researchcodebench), they still fail "
        "the gate, which reads them within test-like pairs, where they reach about 0.1. A 4B judge is weaker than "
        "the 14B on every parent both cover; more 4B items could narrow an interval but could not move a "
        "covariate that the stronger judge, at full coverage, does not carry through the gate.")
    return out


# --- P1.15: the single-subject benchmark --------------------------------------------------------

def _summ(d, w=None):
    """Mean (unweighted, and weighted 1 / run size as experiments/ship_confirm.py's
    per-parent means), SE over appearances, range and the share the shipped model loses."""
    d = np.asarray(d, float)
    n = len(d)
    out = {"mean": r6(d.mean()), "se_appearances": r6(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
           "min": r6(d.min()), "max": r6(d.max()), "share_shipped_worse": r4(float(np.mean(d > 0)))}
    if w is not None:
        w = np.asarray(w, float)
        out["mean_weighted"] = r6(float(np.sum(w * d) / np.sum(w)))
    return out


def stage_single(args):
    import experiments.level_calibration as LC
    import experiments.ship_confirm as SC
    import experiments.subject_side as SS
    t0 = time.time()
    ship = SC.load_ship(SS.ROWS)
    tc = load_json(os.path.join(RES, "testlike_check.json"))
    lc = load_json(os.path.join(RES, "level_calibration.json"))
    he = load_json(os.path.join(RES, "hier_eval.json"))
    LC.register_chunks(LC.grid_chunks())
    arms = SC.build_arms(ship, tc["raw"], he["raw"], lc["raw"])
    del tc, lc, he
    out = {"benchmark": "swe_rebench", "about": (
        "the one public benchmark with a single subject; differences are shipped minus comparator per pair "
        "appearance (ALC, lower is better); se_appearances is the SE over its appearances, one a run, which "
        "resample the subject's items through the run's item cap; it is not a cluster SE (one cluster)")}
    sec = {}
    for regime in ("r1b", "r1p"):
        ev, s_alc, s_b, vs = [], [], [], {}
        for i, (meta, B) in sorted(ship[regime].items()):
            w = 1.0 / len(meta)                    # a run's ALC is the mean over its pairs
            for j, m in enumerate(meta):
                swe = m["parent"] == "swe_rebench"
                if swe:
                    ev.append(int(m["eval"]))
                    s_alc.append(float(B[j] @ W6))
                    s_b.append(B[j].tolist())
                for c in SC.COMPARATORS:
                    a = arms[regime][c]
                    if a is None or i not in a.runs:
                        continue
                    alc, bb = a.runs[i]
                    d = float(B[j] @ W6 - alc[j])
                    v = vs.setdefault(c, {"d": [], "db": [], "other": [], "sw": [], "wd": 0.0, "w": 0.0,
                                          "swe_wd": 0.0, "swe_w": 0.0})
                    v["wd"] += w * d
                    v["w"] += w
                    if swe:
                        v["d"].append(d)
                        v["sw"].append(w)
                        v["other"].append(float(alc[j]))
                        if bb is not None:
                            v["db"].append((B[j] - bb[j]).tolist())
                        v["swe_wd"] += w * d
                        v["swe_w"] += w
        res = {"appearances": len(ev), "runs": len(ship[regime]),
               "eval_items": {"min": min(ev), "median": float(np.median(ev)), "max": max(ev)},
               "ship_alc": r6(float(np.mean(s_alc))), "ship_budgets": [r6(x) for x in np.mean(s_b, 0)], "vs": {}}
        for c, v in vs.items():
            if not v["d"]:
                continue
            e = _summ(v["d"], v["sw"])
            e["appearances"] = len(v["d"])
            e["other_alc"] = r6(float(np.mean(v["other"])))
            if v["db"]:
                db = np.array(v["db"])
                e["budgets"] = [r6(x) for x in db.mean(0)]
                e["budgets_se"] = [r6(x) for x in db.std(0, ddof=1) / math.sqrt(len(db))]
            # the regime's ALC difference (mean over runs of run means) and the part swe_rebench carries
            e["regime_diff"] = r6(v["wd"] / v["w"])
            e["swe_rebench_part_of_regime_diff"] = r6(v["swe_wd"] / v["w"])
            e["regime_diff_other_pairs_only"] = r6((v["wd"] - v["swe_wd"]) / (v["w"] - v["swe_w"]))
            res["vs"][c] = e
        sec[regime] = res
    out["seed0_run2_library"] = {"rows": "data/subject_side_rows (arm 'ship'; bd0be67, hier 70a3a81a, old floor)",
                                 "comparators": "experiments/ship_confirm.py's matched arms", **sec}
    # P1a rows: seed 11, the current archive's library
    p1a = {}
    for R in ("R1B", "R1P"):
        d_by, base, b_by, acc, w_by = {}, [], {}, {}, {}
        eval_items, n_runs = [], 0
        for fn in sorted(os.listdir(os.path.join(P1A_ROWS, R)), key=lambda s: int(s.split(".")[0])):
            if not fn.endswith(".json"):
                continue
            row = load_json(os.path.join(P1A_ROWS, R, fn))
            n_runs += 1
            w = 1.0 / len(row["meta"])
            for j, m in enumerate(row["meta"]):
                swe = m["parent"] == "swe_rebench"
                sb = np.array(row["res"]["ship"]["b"][j], float)
                if swe:
                    eval_items.append(int(m["eval"]))
                    base.append(float(sb @ W6))
                for c in P1A_CONFIGS:
                    if c not in row["res"]:
                        continue
                    ob = np.array(row["res"][c]["b"][j], float)
                    d = float(sb @ W6 - ob @ W6)
                    a = acc.setdefault(c, [0.0, 0.0, 0.0, 0.0])
                    a[0] += w * d
                    a[1] += w
                    if swe:
                        a[2] += w * d
                        a[3] += w
                        d_by.setdefault(c, []).append(d)
                        w_by.setdefault(c, []).append(w)
                        b_by.setdefault(c, []).append((sb - ob).tolist())
        e = {"runs": n_runs, "appearances": len(base),
             "eval_items": {"min": min(eval_items), "median": float(np.median(eval_items)), "max": max(eval_items)},
             "ship_alc": r6(float(np.mean(base))), "vs": {}}
        for c, v in d_by.items():
            s = _summ(v, w_by[c])
            db = np.array(b_by[c])
            s["budgets"] = [r6(x) for x in db.mean(0)]
            s["budgets_se"] = [r6(x) for x in db.std(0, ddof=1) / math.sqrt(len(db))]
            a = acc[c]
            s["regime_diff"] = r6(a[0] / a[1])
            s["swe_rebench_part_of_regime_diff"] = r6(a[2] / a[1])
            s["regime_diff_other_pairs_only"] = r6((a[0] - a[2]) / (a[1] - a[3]))
            e["vs"][c] = s
        p1a[R] = e
    out["seed11_current_library"] = {"rows": "data/regime_sensitivity_rows (P1a; library of 4d2cc4f)",
                                     "configs": load_json(os.path.join(RES, "regime_sensitivity.json"))["plan"]["configs"],
                                     **p1a}
    # item overlap between appearances (the harness rows hold the evaluated items)
    from experiments import harness as Hm
    rows, keys = Hm.load_rows(Hm.ROWS, ["r1b", "r1p"])
    ov = {}
    for reg, R in rows.items():
        sets = [set(R.ev_k[R.ev_a == a].tolist()) for a in range(R.A) if R.parent[a] == "swe_rebench"]
        js = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        allk = set().union(*sets) if sets else set()
        ov[reg] = {"appearances": len(sets), "mean_pairwise_jaccard": r4(float(np.mean(js))) if js else None,
                   "distinct_items": len(allk), "items_per_appearance": r4(float(np.mean([len(s) for s in sets])))
                   if sets else None}
    out["item_overlap_seed0_runs0_99"] = ov
    out["measured"] = [
        "one subject on one benchmark (swe_rebench, 6,306 items), in public R1 runs, where the 1,000-item cap "
        "cuts its evaluation half to a different subset in every run",
        "the shipped model against the legacy Predictor, the smoothed and empirical means at the run-2 archive's "
        "library (seed 0), and against ten alternatives at the current archive's library (seed 11)",
        "where in the budgets the difference sits, and how much of a public regime's ALC difference it carries",
    ]
    out["not_measured"] = [
        "variation between subjects or between single-subject benchmarks: one subject on one benchmark, so no "
        "interval generalises to a hidden single-subject benchmark",
        "the test-like regime: swe_rebench is excluded from test-like runs by default, so the tuned-regime gains "
        "say nothing about single-subject benchmarks",
        "a single-subject benchmark at a level far from the public centre: swe_rebench's pair accuracy is about "
        "0.49 (logit near 0)",
        "an item-level interval for the comparators: per-item rows exist for the shipped hier only (harness rows)",
    ]
    out["wall_s"] = round(time.time() - t0, 1)
    out["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
    state = load_json(OUT) if os.path.exists(OUT) else {}
    state["single"] = out
    save(state)


# --- output ---------------------------------------------------------------------------------------

def provenance():
    files = ["experiments/gate_and_ci.py", "experiments/harness.py", "experiments/itemcov_eval.py",
             "experiments/llm4b_close.py", "experiments/strong_llm_eval.py", "experiments/emb_transfer.py",
             "experiments/transfer.py", "experiments/ship_confirm.py", "paiec/testlike.py", "paiec/rasch.py"]
    inputs = ["results/harness_thresholds.json", "results/strong_llm_eval.json", "results/llm4b_close.json",
              "results/emb_transfer.json", "results/rate2_truth.json", "results/ctrl_truth.json",
              "results/ship_confirm.json", "results/regime_sensitivity.json"]
    return {"head": git("rev-parse", "HEAD"), "status_of_script": git("status", "--short", "experiments/gate_and_ci.py"),
            "code_sha256": {f: sha256(os.path.join(ROOT, f))[:16] for f in files if os.path.exists(os.path.join(ROOT, f))},
            "inputs_sha256": {f: sha256(os.path.join(ROOT, f))[:16] for f in inputs if os.path.exists(os.path.join(ROOT, f))},
            "python": platform.python_version(), "numpy": np.__version__, "boots": BOOTS, "seed": SEED}


def save(state):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, default=float)
    os.replace(tmp, OUT)


def stage_show(args):
    st = load_json(OUT)
    g = st.get("gate")
    if g:
        print("## Gate: one covariate's chance to pass (8 stored noise draws a cell)\n")
        print("| line | r (within pair) | test-like mean (draw sd) | draws passing [Jeffreys 95%] | P normal | "
              "P t-pred | failed: bar / parent | mix/whole mean (draw sd) | P normal mix | mix upper count |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for line, L in g["lines"].items():
            for k, c in L["cells"].items():
                t, m = c["tl"], c["mix"]
                print(f"| {line} | {c['r']:g} ({c['r_within_pair_tl']:.3f}) | {t['mean']:+.5f} ({t['draw_sd']:.5f}) "
                      f"| {t['pass_count']}/{c['draws']} {t['pass_jeffreys95']} | {t['p_normal']:.2f} | "
                      f"{t['p_t_predictive']:.2f} | {t['failed']['tl']} / {t['failed']['worst_parent_inferred']} | "
                      f"{m['mean']:+.5f} ({m['draw_sd']:.5f}) | {m['p_normal']:.2f} | {m['pass_count_upper']}/{c['draws']} |")
            print(f"\n{line}: r for P(test-like clears) = {L['r_for_probability']['tl']}; mix/whole "
                  f"{L['r_for_probability']['mix']}\n")
        print("B1:", json.dumps(g["b1"], indent=1)[:900])
    sc = (st.get("scale") or {}).get("families", {})
    if "reliability" in sc:
        print("\n## Reliability of the honest difficulty\n")
        for q, v in sc["reliability"]["split_half"].items():
            fo = sc["reliability"]["fold_overlap"][q]
            wp = sc["reliability"].get("within_pair_implied", {}).get(q, {})
            print(f"{q}: split-half r {v['half_r_mean']} -> rel honest {v['rel_honest']}, all {v['rel_all']}, "
                  f"kappa {v['kappa']}; fold overlap rel honest {fo['rel_honest']}, kappa {fo['kappa']}; "
                  f"within-pair rel {wp.get('rel_within_pair')}")
    for fam, F in sc.items():
        for name, cv in (F.get("covariates") or {}).items():
            if "parent_scale" not in cv:
                continue
            mp = cv["parent_scale"].get("mean_over_parents") or {}
            w = cv.get("within_pair") or {}
            per = {q: v["r_honest"] for q, v in cv["parent_scale"]["parents"].items()}
            print(f"{name}: r_honest {mp.get('r_honest')} {mp.get('ci_stratified_group')} vs mean "
                  f"{mp.get('r_vs_mean')}; per parent {per}; within {w.get('r_oriented', w.get('r'))} "
                  f"{w.get('ci_cluster_oriented', w.get('ci_cluster'))} stored {w.get('stored')}")
    s = st.get("single")
    if s:
        print("\n## swe_rebench\n")
        for k in ("seed0_run2_library", "seed11_current_library"):
            for reg, v in s[k].items():
                if isinstance(v, dict) and "vs" in v:
                    print(f"{k} {reg}: {v['appearances']} appearances, ship {v['ship_alc']}; " + "; ".join(
                        f"{c} {e['mean']:+.4f} ± {e['se_appearances']:.4f}" for c, e in v["vs"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=("gate", "evidence", "scale", "ci", "single", "all", "show"))
    ap.add_argument("--families", nargs="*", choices=FAMILIES, help="scale: the families to (re)compute")
    args = ap.parse_args()
    stages = ("gate", "evidence", "scale", "ci", "single") if args.stage == "all" else (args.stage,)
    for s in stages:
        t0 = time.time()
        if s == "show":
            stage_show(args)
            continue
        log(f"stage {s}")
        if s in ("gate", "evidence"):
            res = {"gate": stage_gate, "evidence": stage_evidence}[s](args)
            state = load_json(OUT) if os.path.exists(OUT) else {}
            state[s] = res
            save(state)
        else:
            {"scale": stage_scale, "ci": stage_ci, "single": stage_single}[s](args)
        state = load_json(OUT)
        state.setdefault("passes", []).append({
            "stage": s, "command": "python experiments/gate_and_ci.py " + " ".join(sys.argv[1:]),
            "wall_s": round(time.time() - t0, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **provenance()})
        save(state)
        log(f"stage {s}: {time.time() - t0:.0f}s -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
