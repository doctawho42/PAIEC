"""The hidden formative runs: the pair-rate oracle, and what it bounds.

docs/plans/p2_final_plan.md (sections 2 and 5, R1 to R3) rests on exploratory
analyses of 2026-10-06 that lived in a session scratch directory. This script
is their port: it recomputes every number those sections cite from files in
the repository, and checks each one against the value the scratch analysis
recorded (`scratch` in the results file). It selects, tunes and ships nothing.

The identity. When a pair's evaluated responses all get one prediction q, the
pair's Brier is (q - o)^2 + o(1 - o) and its ECE is |q - o| (one occupied bin of
paiec.official.ece), where o is the response-weighted success rate. So
v = B - ECE^2 is the Brier of an oracle that knows each pair's rate, read from
the organisers' tables with no root to choose and no model of ours. At budget 0
every predictor we ran gives (nearly) one value per pair, so v = B0 - ECE0^2.

Blocks (keys of results/hidden_runs.json):

  identity        B_k - ECE_k^2 per pair and budget: constant across the six
                  budgets for LegacyP (run 1) up to the tables' 6-dp rounding
  decomposition   v per run, ALC - v, B_k - v by budget and its weighted split
                  (0.1 B0, 0.2 x B1..B15, 0.1 B31), the B31-root reading it
                  replaces, hier's runs pooled in bins of v
  item_level      the item-level share of (ALC - item oracle), with the item
                  oracle borrowed from the public regimes' ratio item oracle /
                  pair-rate oracle (results/itemsig_eval.json)
  ideal           an exact-grid ideal pair-rate learner (logit-normal prior,
                  hypergeometric labels, split noise) on the hidden pairs' own v,
                  both roots, three priors (seed 11)
  empirical_mean, nonmonotone, run_spread
                  one random stream (seed 7), in the scratch's order: the
                  organisers' empirical mean on these pairs, root-agnostic; the
                  ideal learner's share of non-monotone single pairs; the run-
                  level distribution of v and of hier's ALC over bootstrap runs
                  of 9 pairs (pairs iid, and stratified by benchmark: the
                  anonymous benchmark ids are used for that grouping only, as
                  the report's App G.6 allows; no id is written out)
  entry           the organisers' entry (0.1801) against the empirical mean:
                  the plug-in line 0.025 + 1.2118 p(1-p) and a Monte Carlo with
                  p(1-p) = min(B31, 0.25) (seed 0)
  vadjusted       ALC - v per run with pair-level SEs; LegacyP minus hier, and
                  run 3 minus run 2, v-adjusted; against the held-out test-like
                  comparison of results/ship_confirm.json (runs 200-299)
  curve           the shape of hier's hidden curve (runs 2 and 3) against the
                  replica regimes' means (ship_confirm.json), and the
                  resampling test of its flatness after B7 (seed 0;
                  data/subject_side_rows)
  split_floor     the B31 floor of a pure pair-rate learner on these pairs
  replica         the identity's B0 estimator on replica rows with known rates,
                  the noise of a v-adjusted score, and run 1's and hier's
                  placement in each RS regime (data/regime_sensitivity_rows)
  requirement     the within-pair item correlation r an ideal learner needs to
                  reach 0.117 (synthetic populations, seed 3; --no-requirement
                  skips its ~2 minutes)

Inputs. results/formative/run1-3.txt (the organisers' tables, verbatim),
checked field by field against results/formative_feedback.json and
results/formative_run3.json; results/ship_confirm.json; results/
itemsig_eval.json. Two blocks read gitignored rows the scratch analyses read
too: `curve.test` reads data/subject_side_rows/{tl,r1b,r1p}.jsonl (arm 'ship',
experiments/subject_side.py) and `replica` reads data/regime_sensitivity_rows/
<regime>/*.json (experiments/regime_sensitivity.py). Without them those
blocks are null and their scratch checks are not run. The leaderboard values
(organisers' entry 0.1801, best entry 0.1172, read 2026-09-24) are constants
from docs/findings.md, "Against the live leaderboard"; the empirical mean's
line 0.025 + 1.2118 p(1-p) is the report's Figure A1.

Use of the feedback. Only aggregate, id-free quantities leave this script: run
and bin means, SEs, counts, distribution summaries. Nothing is keyed on an
anonymous id; benchmark ids only group pairs inside the stratified bootstrap.

Random streams. Each Monte Carlo block uses the generator, seed and draw order
of the scratch script it ports, so its numbers reproduce bit for bit; the
scratch scripts and their sha256 are listed under `scratch.ported_from`.

Not ported here (cited by the plan, but from elsewhere): the within-pair r of
transferable covariates (<= 0.15, gate 0.25) is results/gate_and_ci.json and
the report's App F.1; the cheap-lever nulls of section 2's last bullet are the
post-hoc family, a separate port.

Run:  python experiments/hidden_runs.py [--no-requirement] [--out PATH]
Writes results/hidden_runs.json only if every required check passes. About
2.5 minutes on one CPU, under 1 GB; nothing is parallel.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.special import expit, log_expit  # noqa: E402
from scipy.stats import binom, norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import formative_feedback as FF  # noqa: E402  (read-only use: its parser)
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402

OUT = os.path.join(ROOT, "results", "hidden_runs.json")
TABLES = {r: os.path.join(ROOT, "results", "formative", f"run{r}.txt") for r in (1, 2, 3)}
FF_JSON = os.path.join(ROOT, "results", "formative_feedback.json")
R3_JSON = os.path.join(ROOT, "results", "formative_run3.json")
SHIP_JSON = os.path.join(ROOT, "results", "ship_confirm.json")
ITEMSIG_JSON = os.path.join(ROOT, "results", "itemsig_eval.json")
SS_ROWS = os.path.join(ROOT, "data", "subject_side_rows")
RS_ROWS = os.path.join(ROOT, "data", "regime_sensitivity_rows")

W = np.asarray(WEIGHTS, float)
KS = [1, 3, 7, 15, 31]
MODELS = {1: "LegacyP (paiec.predict.Predictor; archive of b68492c)",
          2: "hier-ship (archive of ee5085a: old multiple-choice floor)",
          3: "hier-ship (archive-3, 4d2cc4f)"}
HIER = (2, 3)
LEADERBOARD = {"organisers_entry": 0.1801, "best_entry": 0.1172, "read": "2026-09-24",
               "source": "docs/findings.md, 'Against the live leaderboard'"}
TARGET = 0.117           # the scratch's threshold: the best entry, rounded
EMP_LINE = (0.025, 1.2118)   # docs/report/draft.md Figure A1: EmpMean ALC = a + b E[p(1-p)]
RS_REGIMES = ["TUNED", "READING", "AUDIT", "MIXTURE", "FLAT", "R1B", "R1P"]
ITEM_ORACLE_REGIMES = {"tl": "TL", "tl mix/whole": "TL-mix", "r1b": "R1-bf", "r1p": "R1-pu"}
SEEDS = {"ideal": 11, "stream7": 7, "entry": 0, "curve": 0, "requirement": 3}
V_BINS = (0.10, 0.20)
CURVE_EDGES = (0.12, 0.20)
SCAN = [0.10, 0.11, 0.12, 0.125, 0.13, 0.14, 0.15, 0.16]
NBOOT = 20000


# --- small helpers ---------------------------------------------------------------------------

def rel(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_digest(path, pattern="*.json"):
    """sha256 over the sorted 'name sha256' lines of a directory's files."""
    files = sorted(glob.glob(os.path.join(path, pattern)))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{os.path.basename(f)} {sha256(f)}\n".encode())
    return {"files": len(files), "sha256": h.hexdigest()}


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def fl(x):
    return [float(t) for t in np.asarray(x, float).ravel()]


class Checks:
    """Every check, recorded; a failed required one stops the write."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, got=None, want=None, required=True, note=None):
        self.items.append({"check": name, "ok": bool(ok), "got": got, "want": want,
                           "required": required, **({"note": note} if note else {})})
        return bool(ok)

    def failed(self):
        return [c for c in self.items if c["required"] and not c["ok"]]


# --- the tables --------------------------------------------------------------------------------

def load_tables(paths=None):
    """The organisers' tables -> {run: [pair dicts in table order]} (FF.parse)."""
    out = {}
    for r, p in (paths or TABLES).items():
        with open(p) as f:
            out[r] = FF.parse(f.read())
    return out


def check_tables(runs, checks):
    with open(FF_JSON) as f:
        fb = json.load(f)["record"]
    with open(R3_JSON) as f:
        r3 = json.load(f)["run3"]
    stored = {1: fb["run1"]["pairs"], 2: fb["run2"]["pairs"], 3: r3["pairs"]}
    fields = ("subject", "benchmark", "n", "brier", "ece_b")
    for r, P in runs.items():
        same = len(P) == len(stored[r]) and all(a[k] == b[k] for a, b in zip(P, stored[r])
                                                 for k in fields)
        checks.add(f"run {r}: table equals the stored record pair for pair (order, n, Brier, ECE)",
                   same, len(P), len(stored[r]))
    want = {1: FF.RUNS["run1"]["platform_alc"], 2: FF.RUNS["run2"]["platform_alc"],
            3: r3["alc"]["recomputed_6dp"]}
    for r, P in runs.items():
        got = round(float(np.mean([W @ np.array(p["brier"]) for p in P])), 6)
        checks.add(f"run {r}: pair-mean ALC reproduces the recorded score", got == want[r], got, want[r])


def pair_v(p):
    """v = B0 - ECE0^2, written as the scratch wrote it (python floats)."""
    return p["brier"][0] - p["ece_b"][0] ** 2


def arrays(P):
    return (np.array([p["brier"] for p in P]), np.array([p["ece_b"] for p in P]),
            np.array([pair_v(p) for p in P]))


# --- identity and decomposition -----------------------------------------------------------------

def identity(runs):
    """B_k - ECE_k^2 per pair across budgets. A table value is rounded to 6 dp, so
    one estimate is off by at most 5e-7 + 1e-6 ECE_k (+2.5e-13); a pair's range
    over budgets can be at most the sum of its two largest such errors."""
    out = {}
    for r, P in runs.items():
        B, E, _ = arrays(P)
        est = B - E ** 2
        rg = est.max(1) - est.min(1)
        top2 = np.sort(E, 1)[:, -2:].sum(1)
        bound = 1e-6 + 1e-6 * top2 + 5e-13
        m = est.mean(0)
        out[f"run{r}"] = {
            "pairs": len(P),
            "mean_B_minus_ece2_by_budget": fl(m),
            "range_of_run_mean_over_budgets": float(m.max() - m.min()),
            "per_pair_range_max": float(rg.max()),
            "per_pair_range_median": float(np.median(rg)),
            "pairs_within_rounding_bound": int((rg <= bound).sum()),
            "per_pair_range_over_bound_max": float((rg / bound).max()),
            "mean_B_by_budget": fl(B.mean(0)), "mean_ece_by_budget": fl(E.mean(0)),
            "mean_ece2_by_budget": fl((E ** 2).mean(0)),
        }
    return out


def decomposition(runs):
    out = {}
    for r, P in runs.items():
        B, E, v = arrays(P)
        ex = B - v[:, None]
        alc = float(W @ B.mean(0))
        vm = float(v.mean())
        pair_ex = B @ W - v
        b0 = float(0.1 * ex[:, 0].mean())
        b1_15 = float((0.2 * ex[:, 1:5]).sum(1).mean())
        b31 = float(0.1 * ex[:, 5].mean())
        res31 = v - (B[:, 5] - E[:, 5] ** 2)
        out[f"run{r}"] = {
            "model": MODELS[r], "pairs": len(P),
            "v": vm, "alc": alc, "alc_minus_v": alc - vm,
            "excess_by_budget": fl(ex.mean(0)),
            "b0_term": b0, "b1_b15_terms": b1_15, "b31_term": b31,
            "share_b0_b1_b3": float((0.1 * ex[:, 0].mean() + 0.2 * (ex[:, 1].mean() + ex[:, 2].mean()))
                                    / (alc - vm)),
            "b31_mean": float(B[:, 5].mean()), "b31_minus_v": float(B[:, 5].mean() - vm),
            "alc_minus_v_ece2_convention": float(W @ (E ** 2).mean(0)),
            "pair_alc_minus_v_sd": float(pair_ex.std(ddof=1)),
            "pair_alc_minus_v_se": float(pair_ex.std(ddof=1) / np.sqrt(len(P))),
            "pair_v_sd": float(v.std(ddof=1)),
            "pairs_v_ge_0.20": int((v >= 0.2).sum()), "pairs_v_lt_0.10": int((v < 0.1).sum()),
            "mean_ece0_sq": float((E[:, 0] ** 2).mean()),
            "within_pair_resolution_b31": {"mean": float(res31.mean()),
                                           "pairs_ge_0.011": int((res31 >= 0.011).sum())},
        }
    # hier's two runs pooled (17 pairs), in bins of v
    P = [p for r in HIER for p in runs[r]]
    B, E, v = arrays(P)
    ex = B - v[:, None]
    pair_ex = B @ W - v
    bins = np.digitize(v, V_BINS)
    names = [f"v < {V_BINS[0]:.2f}", f"{V_BINS[0]:.2f} <= v < {V_BINS[1]:.2f}", f"v >= {V_BINS[1]:.2f}"]
    tab = {}
    for j, name in enumerate(names):
        sel = bins == j
        tab[name] = {"pairs": int(sel.sum()), "excess_by_budget": fl(ex[sel].mean(0)),
                     "alc_minus_v": float(pair_ex[sel].mean()),
                     "share_of_pooled_alc_minus_v": float(pair_ex[sel].sum() / pair_ex.sum())}
    res31 = v - (B[:, 5] - E[:, 5] ** 2)
    out["hier_pooled"] = {
        "pairs": len(P), "v": float(v.mean()), "alc_minus_v": float(pair_ex.mean()),
        "excess_by_budget": fl(ex.mean(0)), "rms_ece0": float(np.sqrt((E[:, 0] ** 2).mean())),
        "bins": tab,
        "within_pair_resolution_b31": {"mean": float(res31.mean()),
                                       "pairs_ge_0.011": int((res31 >= 0.011).sum())},
    }
    return out


def item_level(dec, itemsig):
    """Share of (ALC - item oracle) that is item-level, (v - io)/(ALC - io), with the
    item oracle io = ratio x v and the ratio item oracle / pair-rate oracle taken
    from the public regimes (the hidden pairs' item oracle is not identifiable)."""
    reg = itemsig["summary"]["regimes"]
    ratios = {name: reg[k]["oracle"]["item_oracle"] / reg[k]["oracle"]["pair_rate_oracle"]
              for k, name in ITEM_ORACLE_REGIMES.items()}
    lo, hi = min(ratios.values()), max(ratios.values())
    out = {"ratios": ratios, "ratio_range": [lo, hi]}
    for r in (1, 2, 3):
        d = dec[f"run{r}"]
        v, alc = d["v"], d["alc"]
        sh = {rt: (v - rt * v) / (alc - rt * v) for rt in (lo, hi)}
        out[f"run{r}"] = {"share_range": [min(sh.values()), max(sh.values())],
                          "v_minus_item_oracle_range": [v - hi * v, v - lo * v]}
    o = reg["tl"]["oracle"]
    out["tl_measured"] = {"pair_rate_oracle": o["pair_rate_oracle"], "item_oracle": o["item_oracle"],
                          "gap": o["gap"]["mean"], "alc": o["base_alc"],
                          "share": o["gap"]["mean"] / (o["base_alc"] - o["item_oracle"])}
    return out


# --- the ideal pair-rate learner (scratch ideal.py, s10_pairideal.py) -----------------------------

GRID = np.linspace(-9, 9, 721)
PI = expit(GRID)
LPI, L1PI = np.log(PI), np.log1p(-PI)


def prior_w(mu, sd):
    w = np.exp(-0.5 * ((GRID - mu) / sd) ** 2)
    return w / w.sum()


def qtable(pw, kmax=31):
    """q[k][s]: posterior mean of the pair rate after s successes in k labels."""
    lp = np.log(pw + 1e-300)
    q = {}
    for k in range(0, kmax + 1):
        s = np.arange(k + 1)[:, None]
        ll = lp[None, :] + s * LPI[None, :] + (k - s) * L1PI[None, :]
        ll -= ll.max(1, keepdims=True)
        w = np.exp(ll)
        q[k] = (w * PI[None, :]).sum(1) / w.sum(1)
    return q


def sim_pair(p_e, n_e, n_a, q, rng, sims=400):
    """Expected Brier at the six budgets of an ideal learner with qtable q on a pair
    whose evaluation pool has rate p_e over n_e items; acquisition successes
    K_a ~ BetaBinomial(n_a, s_e + 1, n_e - s_e + 1) (the split noise), labels the
    first k of a random order of the acquisition pool."""
    v = p_e * (1 - p_e)
    s_e = p_e * n_e
    th = rng.beta(s_e + 1, n_e - s_e + 1, size=sims)
    Ka = rng.binomial(n_a, th)
    out = np.zeros(6)
    out[0] = v + (q[0][0] - p_e) ** 2
    u = rng.random((sims, n_a))
    order = np.argsort(u, axis=1)
    pool = (order < Ka[:, None]).astype(int)
    cs = np.cumsum(pool, axis=1)
    for j, k in enumerate(KS, start=1):
        kk = min(k, n_a)
        s = cs[:, kk - 1]
        out[j] = v + np.mean((q[kk][s] - p_e) ** 2)
    return out


IDEAL_PRIORS = {"team inference (-1.6, 1.5)": (-1.6, 1.5),
                "pooled reading (-0.70, 1.75)": (-0.70, 1.75),
                "R1-like (-0.70, 1.42)": (-0.70, 1.42)}


def ideal(runs):
    """Both roots of each pair's exact v, and their prior-weighted mixture; no root
    is chosen. v is clipped to [1e-4, 0.25] (run 1 has a pair at v = 0)."""
    rng = np.random.default_rng(SEEDS["ideal"])
    out = {}
    for pn, (mu, sd) in IDEAL_PRIORS.items():
        q = qtable(prior_w(mu, sd))
        for runs_sel in [(2, 3), (1,)]:
            lo_c, hi_c, mix_c, obs = [], [], [], []
            for r in runs_sel:
                for p in runs[r]:
                    v = p["brier"][0] - p["ece_b"][0] ** 2
                    v = min(max(v, 1e-4), 0.25)
                    d = np.sqrt(0.25 - v)
                    plo, phi = 0.5 - d, 0.5 + d
                    n = p["n"]
                    clo = sim_pair(plo, n, n, q, rng, sims=600)
                    chi = sim_pair(phi, n, n, q, rng, sims=600)
                    wlo = norm.pdf(np.log(plo / (1 - plo)), mu, sd)
                    whi = norm.pdf(np.log(phi / (1 - phi)), mu, sd)
                    wl = wlo / (wlo + whi)
                    lo_c.append(clo - v)
                    hi_c.append(chi - v)
                    mix_c.append(wl * (clo - v) + (1 - wl) * (chi - v))
                    obs.append(np.array(p["brier"]) - v)
            lo_c, hi_c, mix_c, obs = map(np.array, (lo_c, hi_c, mix_c, obs))
            out[f"{pn} | runs {runs_sel}"] = {
                "pairs": len(obs), "lower": fl(lo_c.mean(0)), "upper": fl(hi_c.mean(0)),
                "mix": fl(mix_c.mean(0)), "observed": fl(obs.mean(0)),
                "alc_minus_v": {"lower": float(W @ lo_c.mean(0)), "upper": float(W @ hi_c.mean(0)),
                                "mix": float(W @ mix_c.mean(0)), "observed": float(W @ obs.mean(0))}}
    hk = [k for k in out if k.endswith("runs (2, 3)")]
    out["summary_hier"] = {
        "lower_roots_team_prior": out["team inference (-1.6, 1.5) | runs (2, 3)"]["alc_minus_v"]["lower"],
        "prior_weighted_range": [min(out[k]["alc_minus_v"]["mix"] for k in hk),
                                 max(out[k]["alc_minus_v"]["mix"] for k in hk)],
        "upper_roots_range": [min(out[k]["alc_minus_v"]["upper"] for k in hk),
                              max(out[k]["alc_minus_v"]["upper"] for k in hk)],
        "observed_hier": out[hk[0]]["alc_minus_v"]["observed"],
        "note": "ALC - v over the 17 hier pairs of runs 2 and 3; per run, hier's is in "
                "decomposition.run2/run3.alc_minus_v",
    }
    return out


# --- one stream, seed 7 (scratch s7_runs.py): empirical mean, non-monotonicity, run spread ---------

def stream7(runs):
    rng = np.random.default_rng(SEEDS["stream7"])

    def emp_curve(p_e, n):
        """The organisers' empirical mean (q_0 = 0.5, then the mean of the first k
        labels), expected Brier on a pair of rate p_e and n items each side."""
        sims = 4000
        s_e = p_e * n
        th = rng.beta(s_e + 1, n - s_e + 1, size=sims)
        Ka = rng.binomial(n, th)
        u = rng.random((sims, n))
        pool = (np.argsort(u, 1) < Ka[:, None]).astype(int)
        cs = np.cumsum(pool, 1)
        out = [p_e * (1 - p_e) + (0.5 - p_e) ** 2]
        for k in KS:
            kk = min(k, n)
            q = cs[:, kk - 1] / kk
            out.append(p_e * (1 - p_e) + np.mean((q - p_e) ** 2))
        return np.array(out)

    def root(v):
        return 0.5 - np.sqrt(max(0.25 - v, 0))

    emp = {}
    for r in (1, 2, 3):
        C = np.array([emp_curve(root(pair_v(p)), p["n"]) for p in runs[r]])
        emp[f"run{r}"] = {"budgets": fl(C.mean(0)), "alc": float(W @ C.mean(0))}
    scan = []
    for v in SCAN:
        C = emp_curve(root(v), 54)
        scan.append([v, float(W @ C)])
    x = LEADERBOARD["organisers_entry"]
    v_at = None
    for (v0, a0), (v1, a1) in zip(scan, scan[1:]):
        if a0 <= x <= a1:
            v_at = v0 + (x - a0) * (v1 - v0) / (a1 - a0)
            break
    empirical = {"root_agnostic": emp, "v_scan_n54": scan, "v_at_organisers_entry": v_at,
                 "note": "the expected ALC depends on the pair's v only (labels hypergeometric "
                         "within the acquisition pool, split noise between pools); the root "
                         "chosen is the lower one and by symmetry does not matter"}

    nonmono = {}
    for mu, sd in [(-0.70, 1.42), (-1.29, 1.70)]:
        q = qtable(prior_w(mu, sd))
        c1 = c3 = 0
        N = 4000
        for _ in range(N):
            L = rng.normal(mu, sd)
            pi = expit(L)
            n = 55
            ye = rng.random(n) < pi
            ya = rng.random(n) < pi
            pe = ye.mean()
            lab = np.cumsum(ya)

            def br(k):
                qq = q[k][lab[k - 1]] if k else q[0][0]
                return pe * (1 - pe) + (qq - pe) ** 2
            b0, b1, b3 = br(0), br(1), br(3)
            c1 += b1 > b0 + 0.01
            c3 += b3 > b1 + 0.01
        nonmono[f"prior=truth logit N({mu},{sd})"] = {"B1>B0+0.01": c1 / N, "B3>B1+0.01": c3 / N}
    nonmono["P(>=8 of 17 with B3>B1+0.01)"] = {str(pr): float(binom.sf(7, 17, pr))
                                                for pr in [0.20, 0.25, 0.30]}

    allp = [(r, p) for r in (1, 2, 3) for p in runs[r]]
    v_all = np.array([pair_v(p) for _, p in allp])
    alc_h = np.array([W @ np.array(p["brier"]) for r, p in allp if r in HIER])
    bench = [p["benchmark"] for _, p in allp]
    bench_h = [p["benchmark"] for r, p in allp if r in HIER]
    iid_v = np.array([rng.choice(v_all, 9).mean() for _ in range(NBOOT)])
    iid_a = np.array([rng.choice(alc_h, 9).mean() for _ in range(NBOOT)])
    ub = sorted(set(bench))
    byb_v = {b: v_all[[i for i, x_ in enumerate(bench) if x_ == b]] for b in ub}
    byb_a = {b: alc_h[[i for i, x_ in enumerate(bench_h) if x_ == b]] for b in ub}

    def strat(byb):
        """A run of 9 pairs: every benchmark once, two at random twice, values
        resampled within benchmark."""
        out = np.empty(NBOOT)
        for t in range(NBOOT):
            dbl = rng.choice(len(ub), 2, replace=False)
            vals = []
            for j, b in enumerate(ub):
                k = 2 if j in dbl else 1
                vals.extend(rng.choice(byb[b], k))
            out[t] = np.mean(vals)
        return out
    st_v = strat(byb_v)
    st_a = strat(byb_a)

    def summ(x):
        return {"mean": float(x.mean()), "sd": float(x.std()), "p5": float(np.percentile(x, 5)),
                "p1": float(np.percentile(x, 1)), "P_le_0.128": float(np.mean(x <= 0.128)),
                "P_le_0.117": float(np.mean(x <= TARGET)), "P_le_0.10": float(np.mean(x <= 0.10))}
    spread = {"v_iid": summ(iid_v), "v_strat": summ(st_v), "alc_hier_iid": summ(iid_a),
              "alc_hier_strat": summ(st_a), "pairs": {"v": len(v_all), "hier": len(alc_h)},
              "benchmarks": len(ub), "boots": NBOOT}
    pi_, ps_ = np.mean(iid_a <= TARGET), np.mean(st_a <= TARGET)
    zmax = {1: 0, 10: 1.539, 50: 2.249, 200: 2.746, 1000: 3.241}
    spread["best_of_n_hier"] = {str(N): {"P_min_le_0.117_iid": float(1 - (1 - pi_) ** N),
                                         "P_min_le_0.117_strat": float(1 - (1 - ps_) ** N),
                                         "expected_min_iid_normal": float(iid_a.mean() - iid_a.std() * zmax[N])}
                                for N in [1, 10, 50, 200, 1000]}
    zq = {10: 1.282, 50: 2.054, 200: 2.576}
    spread["mean_alc_reaching_0.117_once_in_n"] = {str(N): float(TARGET + zq[N] * iid_a.std())
                                                   for N in [10, 50, 200]}
    return empirical, nonmono, spread


# --- the organisers' entry (scratch formative.py), seed 0 ----------------------------------------

def entry(runs, empirical):
    rng = np.random.default_rng(SEEDS["entry"])
    a0, a1 = EMP_LINE
    out = {"leaderboard": LEADERBOARD, "line": {"a": a0, "b": a1, "source": "docs/report/draft.md Figure A1"},
           "level_reading_of_entry": (LEADERBOARD["organisers_entry"] - a0) / a1}
    for r in (1, 2, 3):
        P = runs[r]
        br = np.array([p["brier"] for p in P])
        n = np.array([p["n"] for p in P])
        pq = np.minimum(br[:, 5], 0.25)
        v = np.array([pair_v(p) for p in P])
        emp_mc = []
        for _ in range(4000):
            a = []
            for k in range(len(P)):
                p = 0.5 - np.sqrt(max(0.25 - pq[k], 0))
                if rng.random() < .5:
                    p = 1 - p
                ne = n[k]
                na = n[k]
                pe = rng.binomial(ne, p) / ne
                acq = rng.random(na) < p
                order = rng.permutation(na)
                s = [0.25]
                for bb in BUDGETS[1:]:
                    ph = acq[order[:bb]].mean()
                    s.append(pe * (1 - ph) ** 2 + (1 - pe) * ph ** 2)
                a.append(W @ np.array(s))
            emp_mc.append(np.mean(a))
        emp_mc = np.array(emp_mc)
        out[f"run{r}"] = {
            "plug_in_b31": float(np.mean(a0 + a1 * pq)),
            "plug_in_v": float(np.mean(a0 + a1 * v)),
            "mc_mean": float(emp_mc.mean()), "mc_sd": float(emp_mc.std()),
            "P_mc_le_entry": float((emp_mc <= LEADERBOARD["organisers_entry"]).mean()),
            "root_agnostic_expected": empirical["root_agnostic"][f"run{r}"]["alc"],
            "v": float(v.mean()),
        }
    out["v_an_empirical_mean_needs_for_entry"] = empirical["v_at_organisers_entry"]
    return out


# --- v-adjusted comparisons (scratch s9_vadj.py, hidden part) -------------------------------------

def vadjusted(runs, ship):
    H = {r: np.array([W @ np.array(p["brier"]) - (p["brier"][0] - p["ece_b"][0] ** 2) for p in runs[r]])
         for r in (1, 2, 3)}
    alcs = {r: np.array([W @ np.array(p["brier"]) for p in runs[r]]) for r in (1, 2, 3)}
    out = {}
    for r in (1, 2, 3):
        out[f"run{r}"] = {"alc_minus_v": float(H[r].mean()), "pair_sd": float(H[r].std(ddof=1)),
                          "se": float(H[r].std(ddof=1) / np.sqrt(len(H[r]))), "raw_alc": float(alcs[r].mean())}
    h = np.concatenate([H[2], H[3]])
    ah = np.concatenate([alcs[2], alcs[3]])
    d = H[1].mean() - h.mean()
    se = np.sqrt(H[1].var(ddof=1) / len(H[1]) + h.var(ddof=1) / len(h))
    out["legacy_minus_hier"] = {"v_adjusted": float(d), "se": float(se),
                                "raw": float(alcs[1].mean() - ah.mean()),
                                "note": "run 1 (LegacyP) minus runs 2 and 3 pooled (hier-ship, 17 pairs); "
                                        "different pairs and subjects, so the difference also carries "
                                        "their composition"}
    d23 = H[3].mean() - H[2].mean()
    se23 = np.sqrt(H[3].var(ddof=1) / len(H[3]) + H[2].var(ddof=1) / len(H[2]))
    out["run3_minus_run2"] = {"v_adjusted": float(d23), "se": float(se23),
                              "raw": float(alcs[3].mean() - alcs[2].mean())}
    rows = {t["runs"]: t for t in ship["table"] if t["regime"] == "tl" and t["comparator"] == "legacy"}
    tl = {k: {"shipped_minus_legacy": rows[k]["diff"], "run_se": rows[k]["run_se"],
              "cluster_se": rows[k]["cluster_se"], "strat_se": rows[k]["strat_se"],
              "parent_mean": rows[k]["parent_mean"], "parent_se": rows[k]["parent_se"]}
          for k in ("0-99", "100-199", "200-299", "0-299") if k in rows}
    out["test_like"] = {"source": "results/ship_confirm.json table (regime tl, comparator legacy)",
                        "held_out": "200-299", "blocks": tl}
    ho = tl["200-299"]
    gap = d - (-ho["shipped_minus_legacy"])
    gse = float(np.sqrt(se ** 2 + ho["cluster_se"] ** 2))
    out["agreement"] = {"hidden_legacy_minus_hier": float(d),
                        "test_like_legacy_minus_hier": -ho["shipped_minus_legacy"],
                        "difference": float(gap), "se": gse, "z": float(gap / gse)}
    return out


# --- the curve after B7 -------------------------------------------------------------------------

def curve_shape(runs, ship):
    X = np.array([p["brier"] for r in HIER for p in runs[r]])
    steps = {"B0->B1": X[:, 1] - X[:, 0], "B1->B3": X[:, 2] - X[:, 1], "B3->B7": X[:, 3] - X[:, 2],
             "B7->B15": X[:, 4] - X[:, 3], "B15->B31": X[:, 5] - X[:, 4], "B7->B31": X[:, 5] - X[:, 3],
             "B1->B7": X[:, 3] - X[:, 1]}
    plat = {k: [float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s)))] for k, s in steps.items()}
    reg = {}
    for name, v in ship["run2_placement"]["regimes"].items():
        m = np.array(v["mean_budgets"])
        reg[name] = {"B0->B1": float(m[1] - m[0]), "B1->B3": float(m[2] - m[1]),
                     "B3->B7": float(m[3] - m[2]), "B7->B31": float(m[5] - m[3])}
    per_run = {}
    for r in (1, 2, 3):
        B = np.array([p["brier"] for p in runs[r]])
        drop = (B[:, 0] - B[:, 5]).mean()
        per_run[f"run{r}"] = {"b7_minus_b31": float((B[:, 3] - B[:, 5]).mean()),
                              "share_of_drop_by_B1": float((B[:, 0] - B[:, 1]).mean() / drop),
                              "share_of_drop_by_B3": float((B[:, 0] - B[:, 2]).mean() / drop),
                              "share_of_drop_by_B7": float((B[:, 0] - B[:, 3]).mean() / drop),
                              "b3_minus_b1": float((B[:, 2] - B[:, 1]).mean())}
    return {"hier_pairs": len(X), "platform_steps_mean_se": plat,
            "pairs_B3_gt_B1_plus_0.01": int((X[:, 2] > X[:, 1] + 0.01).sum()),
            "pairs_B1_gt_B0_plus_0.01": int((X[:, 1] > X[:, 0] + 0.01).sum()),
            "replica_mean_steps": reg, "replica_source": "results/ship_confirm.json run2_placement "
                                                         "regimes[*].mean_budgets (report App E.7)",
            "per_run": per_run}


def curve_test(runs, ss_dir):
    """17 replica pairs (hier-ship's stored rows) with the hidden runs' composition
    by p(1-p) bin, both sides binned on min(B31, 0.25), 20,000 times."""
    paths = {reg: os.path.join(ss_dir, f"{reg}.jsonl") for reg in ("tl", "r1b", "r1p")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    def b_of(v):
        return int(v >= CURVE_EDGES[0]) + int(v >= CURVE_EDGES[1])
    plat = np.array([p["brier"] for r in HIER for p in runs[r]])
    pb = np.array([b_of(min(x, 0.25)) for x in plat[:, 5]])
    comp = np.bincount(pb, minlength=3)

    def stat(X):
        return ((X[:, 5] - X[:, 3]).mean(), (X[:, 5] - X[:, 3]).std(ddof=1),
                (X[:, 2] - X[:, 1]).mean(), (X[:, 3] - X[:, 1]).mean())
    ps = stat(plat)
    out = {"platform": {"composition_by_bin": [int(c) for c in comp], "B7->B31_mean": float(ps[0]),
                        "B7->B31_sd": float(ps[1]), "B1->B3_mean": float(ps[2]), "B1->B7_mean": float(ps[3])},
           "bins_on": "min(B31, 0.25)", "edges": list(CURVE_EDGES), "draws": 20000, "replica": {}}
    rng = np.random.default_rng(SEEDS["curve"])
    for reg, path in paths.items():
        rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "ship" in r["res"]:
                    rows += r["res"]["ship"]["b"]
        X = np.array(rows)
        xb = np.array([b_of(min(x, 0.25)) for x in X[:, 5]])
        idx = [np.flatnonzero(xb == k) for k in range(3)]
        S = []
        for _ in range(20000):
            pick = np.concatenate([rng.choice(idx[k], comp[k]) for k in range(3)])
            S.append(stat(X[pick]))
        S = np.array(S)
        out["replica"][reg] = {
            "pairs": len(X),
            "B7->B31_mean_of17": [float(S[:, 0].mean()), float(S[:, 0].std())],
            "P(mean >= platform)": float((S[:, 0] >= ps[0]).mean()),
            "B7->B31_sd_of17": float(S[:, 1].mean()),
            "P(sd <= platform)": float((S[:, 1] <= ps[1]).mean()),
            "P(mean>=plat and sd<=plat)": float(((S[:, 0] >= ps[0]) & (S[:, 1] <= ps[1])).mean()),
            "B1->B3_mean_of17": [float(S[:, 2].mean()), float(S[:, 2].std())],
            "P(B1->B3 >= platform)": float((S[:, 2] >= ps[2]).mean()),
            "B1->B7_mean_of17": [float(S[:, 3].mean()), float(S[:, 3].std())],
            "P(B1->B7 >= platform)": float((S[:, 3] >= ps[3]).mean()),
        }
    return out


def split_floor(runs):
    """A pure pair-rate learner's B31 excess: split noise p(1-p)(1/n_a + 1/n_e)
    with n_a = n_e = n, plus the sampling error of 31 labels from n items
    (finite-population corrected), v = B31 - ECE31^2 as the scratch took it;
    against the runs' mean ECE^2 at B7, B15 and B31."""
    out = {}
    for r in HIER:
        rows = []
        for p in runs[r]:
            n = p["n"]
            v = p["brier"][5] - p["ece_b"][5] ** 2
            split = v * 2 / n
            samp31 = v / 31 * max(n - 31, 0) / max(n - 1, 1)
            rows.append((split, samp31, p["ece_b"][5] ** 2, p["ece_b"][4] ** 2, p["ece_b"][3] ** 2))
        a = np.array(rows).mean(0)
        out[f"run{r}"] = {"split_noise": float(a[0]), "sampling_31": float(a[1]),
                          "floor_pure_rate_learner_B31": float(a[0] + a[1]),
                          "platform_excess_B31": float(a[2]), "platform_excess_B15": float(a[3]),
                          "platform_excess_B7": float(a[4])}
    return out


# --- replica rows (scratch s6_murphy.py validation, s9_vadj.py, the placement z) ------------------

def replica(runs, rs_dir):
    if not all(os.path.isdir(os.path.join(rs_dir, reg)) for reg in RS_REGIMES):
        return None
    obs = {"run1 (LegacyP)": ("legacy", np.array([np.array(p["brier"]) - pair_v(p) for p in runs[1]])),
           "runs 2+3 (hier)": ("ship", np.array([np.array(p["brier"]) - pair_v(p) for r in HIER for p in runs[r]])),
           "run2 (hier)": ("ship", np.array([np.array(p["brier"]) - pair_v(p) for p in runs[2]])),
           "run3 (hier)": ("ship", np.array([np.array(p["brier"]) - pair_v(p) for p in runs[3]]))}
    out = {"validation": {}, "noise": {}, "placement_z": {}}
    for reg in RS_REGIMES:
        files = sorted(glob.glob(os.path.join(rs_dir, reg, "*.json")))
        Bs, E, V = [], [], []
        per = {"ship": {}, "legacy": {}}
        for f in files:
            with open(f) as fh:
                r = json.load(fh)
            if "ship" in r["res"]:
                for m, b, e in zip(r["meta"], r["res"]["ship"]["b"], r["res"]["ship"]["ece"]):
                    Bs.append(b)
                    E.append(e)
                    V.append(m["p"] * (1 - m["p"]))
            for arm in per:
                x = r["res"].get(arm)
                if not isinstance(x, dict) or "b" not in x:
                    continue
                for m, b in zip(r["meta"], x["b"]):
                    per[arm].setdefault(r["i"], []).append((np.array(b), m["p"] * (1 - m["p"])))
        Bs, E, V = np.array(Bs), np.array(E), np.array(V)
        err = (Bs - E ** 2) - V[:, None]
        out["validation"][reg] = {"appearances": len(V), "bias_by_budget": fl(err.mean(0)),
                                  "sd_by_budget": fl(err.std(0)), "b31_minus_v": float(np.mean(Bs[:, 5] - V))}
        out["noise"][reg] = {}
        R = {}
        for arm, byrun in per.items():
            runm = np.array([np.mean([W @ b - v for b, v in L]) for L in byrun.values()])
            alc = np.array([np.mean([W @ b for b, _ in L]) for L in byrun.values()])
            exb = np.array([np.mean([b - v for b, v in L], 0) for L in byrun.values()])
            R[arm] = exb
            out["noise"][reg][arm] = {"runs": len(runm), "alc": float(alc.mean()), "alc_sd": float(alc.std()),
                                      "v": float(np.mean([np.mean([v for _, v in L]) for L in byrun.values()])),
                                      "alc_minus_v": float(runm.mean()), "alc_minus_v_sd": float(runm.std()),
                                      "excess_by_budget": fl(exb.mean(0)),
                                      "sd_ratio_raw_over_vadjusted": float(alc.std() / runm.std())}
        out["placement_z"][reg] = {}
        for name, (arm, ex) in obs.items():
            Rm = R[arm]
            o = ex.mean(0)
            z = (o - Rm.mean(0)) / Rm.std(0)
            za = (W @ o - (Rm @ W).mean()) / (Rm @ W).std()
            out["placement_z"][reg][name] = {"arm": arm, "z_by_budget": fl(z), "z_alc_minus_v": float(za)}
    zl = [out["placement_z"][g]["run1 (LegacyP)"]["z_by_budget"] for g in RS_REGIMES[:5]]
    out["placement_summary"] = {
        "run1_vs_R1B_B0_B1": out["placement_z"]["R1B"]["run1 (LegacyP)"]["z_by_budget"][:2],
        "run1_vs_R1P_B0_B1": out["placement_z"]["R1P"]["run1 (LegacyP)"]["z_by_budget"][:2],
        "run1_vs_test_like_range": [float(np.min(zl)), float(np.max(zl))],
        "hier_pooled_range": [float(min(min(out["placement_z"][g]["runs 2+3 (hier)"]["z_by_budget"]) for g in RS_REGIMES)),
                              float(max(max(out["placement_z"][g]["runs 2+3 (hier)"]["z_by_budget"]) for g in RS_REGIMES))],
        "hier_per_run_range": [float(min(min(out["placement_z"][g][k]["z_by_budget"]) for g in RS_REGIMES
                                         for k in ("run2 (hier)", "run3 (hier)"))),
                               float(max(max(out["placement_z"][g][k]["z_by_budget"]) for g in RS_REGIMES
                                         for k in ("run2 (hier)", "run3 (hier)")))],
        "note": "z = (hidden mean of B_k - v minus the regime's run mean) / the regime's single-run sd "
                "(runs of formative size, 8 to 9 pairs); the pooled hier row has 17 pairs, so its z "
                "against a 9-pair sd understates its precision",
        "test_like": RS_REGIMES[:5],
    }
    out["noise_summary"] = {
        "ship_alc_sd_range": [min(out["noise"][g]["ship"]["alc_sd"] for g in RS_REGIMES),
                              max(out["noise"][g]["ship"]["alc_sd"] for g in RS_REGIMES)],
        "ship_alc_minus_v_sd_range": [min(out["noise"][g]["ship"]["alc_minus_v_sd"] for g in RS_REGIMES),
                                      max(out["noise"][g]["ship"]["alc_minus_v_sd"] for g in RS_REGIMES)],
        "ship_sd_ratio_range": [min(out["noise"][g]["ship"]["sd_ratio_raw_over_vadjusted"] for g in RS_REGIMES),
                                max(out["noise"][g]["ship"]["sd_ratio_raw_over_vadjusted"] for g in RS_REGIMES)],
    }
    out["validation_summary"] = {
        "b0_bias_range": [min(out["validation"][g]["bias_by_budget"][0] for g in RS_REGIMES),
                          max(out["validation"][g]["bias_by_budget"][0] for g in RS_REGIMES)],
        "b0_sd_range": [min(out["validation"][g]["sd_by_budget"][0] for g in RS_REGIMES),
                        max(out["validation"][g]["sd_by_budget"][0] for g in RS_REGIMES)],
        "b31_bias_range": [min(out["validation"][g]["bias_by_budget"][5] for g in RS_REGIMES),
                           max(out["validation"][g]["bias_by_budget"][5] for g in RS_REGIMES)],
    }
    return out


# --- what reaching 0.117 needs (scratch s4_pop.py simulate, s8_req.py) ----------------------------

def _kap(tau2):
    return np.sqrt(1 + np.pi * tau2 / 8)


def simulate(mu, sd, tau, rs, n=55, pairs=3000, seed=0):
    """Synthetic pairs: rate logit ~ N(mu, sd); item logits c + eps, eps ~ N(0,
    tau^2), c = logit x kappa(tau); one response per item in evaluation and
    acquisition pools of n items. An item covariate x = r eps / tau + noise.
    'cal' is the exact grid-posterior learner under the calibrated prior,
    'oracle' knows c from B0."""
    rng = np.random.default_rng(seed)
    L = rng.normal(mu, sd, pairs)
    K = _kap(tau ** 2)
    c = L * K
    eps_e = rng.normal(0, tau, (pairs, n))
    eps_a = rng.normal(0, tau, (pairs, n))
    pe = expit(c[:, None] + eps_e)
    pa = expit(c[:, None] + eps_a)
    ye = (rng.random((pairs, n)) < pe).astype(float)
    ya = (rng.random((pairs, n)) < pa).astype(float)
    eta_e = rng.normal(size=(pairs, n))
    eta_a = rng.normal(size=(pairs, n))
    p_emp = ye.mean(1)
    v_e = p_emp * (1 - p_emp)
    item_or = ((pe - ye) ** 2).mean(1)
    G = np.linspace(-10, 10, 301)
    lp = -0.5 * ((G / K - mu) / sd) ** 2
    out = {}
    for r in rs:
        xe = r * eps_e / tau + np.sqrt(1 - r * r) * eta_e
        xa = r * eps_a / tau + np.sqrt(1 - r * r) * eta_a
        kr = _kap(tau ** 2 * (1 - r * r))
        oe = r * tau * xe
        oa = r * tau * xa
        cal = np.zeros((pairs, 6))
        orc = np.zeros((pairs, 6))
        q_or = expit((c[:, None] + oe) / kr)
        b_or = ((q_or - ye) ** 2).mean(1)
        orc[:] = b_or[:, None]
        for ch in range(0, pairs, 500):
            sl = slice(ch, min(ch + 500, pairs))
            se = expit((G[None, None, :] + oe[sl][:, :, None]) / kr)
            za = (G[None, None, :] + oa[sl][:, :, None]) / kr
            ll1 = log_expit(za)
            ll0 = log_expit(-za)
            inc = ya[sl][:, :, None] * ll1 + (1 - ya[sl][:, :, None]) * ll0
            cum = np.cumsum(inc, axis=1)
            for j, k in enumerate(BUDGETS):
                ll = lp[None, :] + (cum[:, k - 1, :] if k > 0 else 0.0)
                ll = ll - ll.max(1, keepdims=True)
                w = np.exp(ll)
                w /= w.sum(1, keepdims=True)
                q = np.einsum('png,pg->pn', se, w)
                cal[sl, j] = ((q - ye[sl]) ** 2).mean(1)
        out[r] = {"cal": cal, "oracle": orc}
    return {"v": v_e, "item": item_or, "rates": p_emp, "res": out}


REQ_RS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
REQ_POPS = {"hidden-like v~0.165 (-0.70,1.42)": dict(mu=-0.70, sd=1.42),
            "organisers-run-like v~0.125 (-1.64,1.49)": dict(mu=-1.64, sd=1.49)}


def requirement():
    def req(d, target=TARGET):
        xs = np.array(REQ_RS)
        ys = np.array([d[r] for r in REQ_RS])
        if ys.min() > target:
            return None
        i = np.where(ys <= target)[0][0]
        if i == 0:
            return 0.0
        return float(xs[i - 1] + (target - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))
    res = {}
    for tau in [2.0, 2.6, 3.2]:
        for pn, kw in REQ_POPS.items():
            S = simulate(tau=tau, rs=REQ_RS, pairs=2500, seed=SEEDS["requirement"], **kw)
            cal = {r: float(W @ S["res"][r]["cal"].mean(0)) for r in REQ_RS}
            orc = {r: float(W @ S["res"][r]["oracle"].mean(0)) for r in REQ_RS}
            res[f"tau={tau} | {pn}"] = {
                "v": float(S["v"].mean()), "item": float(S["item"].mean()),
                "cal": {str(r): cal[r] for r in REQ_RS}, "known_rate": {str(r): orc[r] for r in REQ_RS},
                "cal_budgets_r1": fl(S["res"][1.0]["cal"].mean(0)),
                "req_r_cal": req(cal), "req_r_known": req(orc)}
    hk = [k for k in res if "hidden-like" in k]
    res["summary"] = {"hidden_like_req_r_known": [res[k]["req_r_known"] for k in hk],
                      "hidden_like_req_r_cal": [res[k]["req_r_cal"] for k in hk],
                      "taus": [2.0, 2.6, 3.2]}
    return res


# --- everything, the main numbers ----------------------------------------------------------------

def compute(runs, ss_dir=SS_ROWS, rs_dir=RS_ROWS, with_rows=True, with_requirement=True):
    with open(SHIP_JSON) as f:
        ship = json.load(f)
    with open(ITEMSIG_JSON) as f:
        itemsig = json.load(f)
    res = {"tables": {f"run{r}": {"model": MODELS[r], "pairs": len(P),
                                  "benchmarks": len({p["benchmark"] for p in P}),
                                  "alc": float(np.mean([W @ np.array(p["brier"]) for p in P]))}
                      for r, P in runs.items()}}
    res["identity"] = identity(runs)
    res["decomposition"] = decomposition(runs)
    res["item_level"] = item_level(res["decomposition"], itemsig)
    res["ideal"] = ideal(runs)
    res["empirical_mean"], res["nonmonotone"], res["run_spread"] = stream7(runs)
    res["entry"] = entry(runs, res["empirical_mean"])
    res["vadjusted"] = vadjusted(runs, ship)
    res["curve"] = {"shape": curve_shape(runs, ship),
                    "test": curve_test(runs, ss_dir) if with_rows else None}
    res["split_floor"] = split_floor(runs)
    res["replica"] = replica(runs, rs_dir) if with_rows else None
    res["requirement"] = requirement() if with_requirement else None
    return res


def main_numbers(res):
    """The headline numbers, flat; tests/test_hidden_runs.py recomputes the ones
    that need no data rows and compares them with the stored file."""
    d, il, va = res["decomposition"], res["item_level"], res["vadjusted"]
    m = {"identity_run1_per_pair_range_max": res["identity"]["run1"]["per_pair_range_max"],
         "identity_run1_range_of_mean": res["identity"]["run1"]["range_of_run_mean_over_budgets"]}
    for r in (1, 2, 3):
        k = f"run{r}"
        m.update({f"v_{k}": d[k]["v"], f"alc_{k}": d[k]["alc"], f"alc_minus_v_{k}": d[k]["alc_minus_v"],
                  f"b0_term_{k}": d[k]["b0_term"], f"b1_b15_terms_{k}": d[k]["b1_b15_terms"],
                  f"b31_term_{k}": d[k]["b31_term"], f"b31_minus_v_{k}": d[k]["b31_minus_v"],
                  f"item_share_lo_{k}": il[k]["share_range"][0], f"item_share_hi_{k}": il[k]["share_range"][1],
                  f"alc_minus_v_se_{k}": va[k]["se"],
                  f"entry_plug_in_{k}": res["entry"][k]["plug_in_b31"],
                  f"entry_mc_mean_{k}": res["entry"][k]["mc_mean"],
                  f"entry_P_le_{k}": res["entry"][k]["P_mc_le_entry"],
                  f"empmean_root_agnostic_{k}": res["empirical_mean"]["root_agnostic"][k]["alc"]})
        for b, x in zip(BUDGETS, d[k]["excess_by_budget"]):
            m[f"excess_B{b}_{k}"] = x
    s = res["ideal"]["summary_hier"]
    m.update({"ideal_lower_team_hier": s["lower_roots_team_prior"],
              "ideal_prior_weighted_lo": s["prior_weighted_range"][0],
              "ideal_prior_weighted_hi": s["prior_weighted_range"][1],
              "hier_pooled_alc_minus_v": s["observed_hier"]})
    rs = res["run_spread"]
    m.update({"v_run_strat_mean": rs["v_strat"]["mean"], "v_run_strat_sd": rs["v_strat"]["sd"],
              "v_run_iid_mean": rs["v_iid"]["mean"], "v_run_iid_sd": rs["v_iid"]["sd"],
              "P_v_le_0.117_strat": rs["v_strat"]["P_le_0.117"], "P_v_le_0.117_iid": rs["v_iid"]["P_le_0.117"],
              "alc_hier_run_sd_iid": rs["alc_hier_iid"]["sd"], "alc_hier_run_sd_strat": rs["alc_hier_strat"]["sd"],
              "legacy_minus_hier_vadj": va["legacy_minus_hier"]["v_adjusted"],
              "legacy_minus_hier_vadj_se": va["legacy_minus_hier"]["se"],
              "legacy_minus_hier_raw": va["legacy_minus_hier"]["raw"],
              "run3_minus_run2_vadj": va["run3_minus_run2"]["v_adjusted"],
              "run3_minus_run2_vadj_se": va["run3_minus_run2"]["se"],
              "held_out_tl_shipped_minus_legacy": va["test_like"]["blocks"]["200-299"]["shipped_minus_legacy"],
              "held_out_tl_cluster_se": va["test_like"]["blocks"]["200-299"]["cluster_se"],
              "curve_hidden_b7_b31_mean": res["curve"]["shape"]["platform_steps_mean_se"]["B7->B31"][0],
              "v_at_organisers_entry": res["empirical_mean"]["v_at_organisers_entry"]})
    for r in HIER:
        sf = res["split_floor"][f"run{r}"]
        m[f"split_floor_B31_run{r}"] = sf["floor_pure_rate_learner_B31"]
        m[f"platform_excess_B31_run{r}"] = sf["platform_excess_B31"]
    ct = res["curve"]["test"]
    if ct:
        for reg in ("tl", "r1b", "r1p"):
            m[f"curve_P_flat_{reg}"] = ct["replica"][reg]["P(mean >= platform)"]
            m[f"curve_replica_b7_b31_{reg}"] = ct["replica"][reg]["B7->B31_mean_of17"][0]
    rp = res["replica"]
    if rp:
        m["z_run1_vs_R1B_B0"] = rp["placement_summary"]["run1_vs_R1B_B0_B1"][0]
        m["z_run1_vs_R1P_B0"] = rp["placement_summary"]["run1_vs_R1P_B0_B1"][0]
        m["replica_b0_bias_lo"], m["replica_b0_bias_hi"] = rp["validation_summary"]["b0_bias_range"]
    rq = res["requirement"]
    if rq:
        for t, x in zip(rq["summary"]["taus"], rq["summary"]["hidden_like_req_r_known"]):
            m[f"req_r_known_hidden_tau{t}"] = x
    return m


# --- the scratch values, and the comparison -------------------------------------------------------

#: What the scratch analysis recorded, by results path. ("full", x): a stored
#: full-precision value (scratch JSON); (d, x): a value printed to d decimals
#: (scratch stdout or its notes); ("pct", x): a percentage printed as an integer.
#: Sources: s6 = s6_murphy.py, s7 = s7_runs.py, s9 = s9_vadj.py, s10 =
#: s10_pairideal.py, fm = formative.py, ct = curve_test.py, cs = curve_shape.py,
#: sf = split_floor.py, s8 = s8_req.py; gap = gap.md, pl = docs/plans/p2_final_plan.md.
SCRATCH = {
    ('identity', 'run1', 'mean_B_minus_ece2_by_budget'): ('full', [0.1536370784011111, 0.1536370877218889, 0.1536369361048889, 0.153637020434, 0.15363698811677776, 0.15363703326555556], 's6_murphy.json'),
    ('identity', 'run1', 'mean_B_by_budget'): ('full', [0.35885511111111107, 0.2538957777777778, 0.20431466666666664, 0.17116022222222219, 0.16932255555555556, 0.15675833333333333], 's6_murphy.json'),
    ('identity', 'run1', 'mean_ece_by_budget'): ('full', [0.3811597777777778, 0.2529681111111111, 0.19483622222222222, 0.09672288888888889, 0.08972611111111112, 0.04651888888888888], 's6_murphy.json'),
    ('identity', 'run2', 'mean_B_minus_ece2_by_budget'): ('full', [0.179666320893875, 0.17951531895162504, 0.177659249122125, 0.175252668413875, 0.17463544312137497, 0.17588514957825002], 's6_murphy.json'),
    ('identity', 'run2', 'mean_B_by_budget'): ('full', [0.23684087499999998, 0.19470175, 0.19629649999999998, 0.18358662500000003, 0.17847899999999997, 0.18326375], 's6_murphy.json'),
    ('identity', 'run2', 'mean_ece_by_budget'): ('full', [0.210268625, 0.097836125, 0.11827237499999999, 0.084306125, 0.05711237500000001, 0.0762345], 's6_murphy.json'),
    ('identity', 'run3', 'mean_B_minus_ece2_by_budget'): ('full', [0.16081912004766669, 0.1610803431878889, 0.16089234739644442, 0.1588928268901111, 0.1602482422582222, 0.15838852467511114], 's6_murphy.json'),
    ('identity', 'run3', 'mean_B_by_budget'): ('full', [0.23509711111111115, 0.1868357777777778, 0.18850977777777778, 0.16641955555555554, 0.16674822222222221, 0.164402], 's6_murphy.json'),
    ('identity', 'run3', 'mean_ece_by_budget'): ('full', [0.25445988888888893, 0.14587788888888886, 0.1353366666666667, 0.07657233333333334, 0.0727551111111111, 0.06584733333333331], 's6_murphy.json'),
    ('identity', 'run1', 'per_pair_range_max'): (6, 0.0, "gap.md section 2 ('per-pair range 0.0', 6 decimals)"),
    ('identity', 'run1', 'range_of_run_mean_over_budgets'): (6, 0.0, "plan section 2 ('constant ... to 1e-6')"),
    ('decomposition', 'run1', 'v'): ('full', 0.1536370784011111, 's7_dec.json'),
    ('decomposition', 'run1', 'alc'): ('full', 0.21129998888888887, 's7_dec.json'),
    ('decomposition', 'run1', 'excess_by_budget'): ('full', [0.20521803271, 0.10025869937666668, 0.05067758826555557, 0.017523143821111115, 0.015685477154444447, 0.0031212549322222278], 's7_dec.json'),
    ('decomposition', 'run1', 'b0_term'): ('full', 0.020521803271000002, 's7_dec.json'),
    ('decomposition', 'run1', 'b1_b15_terms'): ('full', 0.03682898172355556, 's7_dec.json'),
    ('decomposition', 'run1', 'b31_term'): ('full', 0.0003121254932222228, 's7_dec.json'),
    ('decomposition', 'run2', 'v'): ('full', 0.17966632089387502, 's7_dec.json'),
    ('decomposition', 'run2', 'alc'): ('full', 0.19262323750000002, 's7_dec.json'),
    ('decomposition', 'run2', 'excess_by_budget'): ('full', [0.057174554106125, 0.015035429106124995, 0.016630179106124993, 0.003920304106125002, -0.0011873208938750076, 0.003597429106124999], 's7_dec.json'),
    ('decomposition', 'run2', 'b0_term'): ('full', 0.0057174554106125, 's7_dec.json'),
    ('decomposition', 'run2', 'b1_b15_terms'): ('full', 0.006879718284899997, 's7_dec.json'),
    ('decomposition', 'run2', 'b31_term'): ('full', 0.0003597429106125, 's7_dec.json'),
    ('decomposition', 'run3', 'v'): ('full', 0.16081912004766669, 's7_dec.json'),
    ('decomposition', 'run3', 'alc'): ('full', 0.18165257777777777, 's7_dec.json'),
    ('decomposition', 'run3', 'excess_by_budget'): ('full', [0.07427799106344443, 0.026016657730111113, 0.027690657730111104, 0.005600435507888881, 0.005929102174555553, 0.0035828799523333252], 's7_dec.json'),
    ('decomposition', 'run3', 'b0_term'): ('full', 0.007427799106344443, 's7_dec.json'),
    ('decomposition', 'run3', 'b1_b15_terms'): ('full', 0.013047370628533332, 's7_dec.json'),
    ('decomposition', 'run3', 'b31_term'): ('full', 0.00035828799523333253, 's7_dec.json'),
    ('decomposition', 'run1', 'alc_minus_v'): (4, 0.0577, 'plan section 2'),
    ('decomposition', 'run1', 'b31_mean'): (4, 0.1568, 'gap.md section 2 table'),
    ('decomposition', 'run1', 'b31_minus_v'): (4, 0.0031, 'gap.md section 2 table'),
    ('decomposition', 'run1', 'alc_minus_v_ece2_convention'): (4, 0.0577, 'protocol.md section 7'),
    ('decomposition', 'run1', 'pairs_v_ge_0.20'): (0, 3, 's7 stdout'),
    ('decomposition', 'run1', 'pairs_v_lt_0.10'): (0, 2, 's7 stdout'),
    ('decomposition', 'run1', 'pair_v_sd'): (3, 0.082, 's7 stdout'),
    ('decomposition', 'run2', 'alc_minus_v'): (4, 0.013, 'plan section 2'),
    ('decomposition', 'run2', 'b31_mean'): (4, 0.1833, 'gap.md section 2 table'),
    ('decomposition', 'run2', 'b31_minus_v'): (4, 0.0036, 'gap.md section 2 table'),
    ('decomposition', 'run2', 'alc_minus_v_ece2_convention'): (4, 0.0156, 'protocol.md section 7'),
    ('decomposition', 'run2', 'pairs_v_ge_0.20'): (0, 4, 's7 stdout'),
    ('decomposition', 'run2', 'pairs_v_lt_0.10'): (0, 1, 's7 stdout'),
    ('decomposition', 'run2', 'pair_v_sd'): (3, 0.079, 's7 stdout'),
    ('decomposition', 'run3', 'alc_minus_v'): (4, 0.0208, 'plan section 2'),
    ('decomposition', 'run3', 'b31_mean'): (4, 0.1644, 'gap.md section 2 table'),
    ('decomposition', 'run3', 'b31_minus_v'): (4, 0.0036, 'gap.md section 2 table'),
    ('decomposition', 'run3', 'alc_minus_v_ece2_convention'): (4, 0.0215, 'protocol.md section 7'),
    ('decomposition', 'run3', 'pairs_v_ge_0.20'): (0, 2, 's7 stdout'),
    ('decomposition', 'run3', 'pairs_v_lt_0.10'): (0, 2, 's7 stdout'),
    ('decomposition', 'run3', 'pair_v_sd'): (3, 0.058, 's7 stdout'),
    ('decomposition', 'run2', 'share_b0_b1_b3'): ('pct', 93, 'gap.md section 3'),
    ('decomposition', 'run3', 'share_b0_b1_b3'): ('pct', 87, 'gap.md section 3'),
    ('decomposition', 'run2', 'mean_ece0_sq'): (3, 0.057, 'gap.md section 3(a)'),
    ('decomposition', 'run3', 'mean_ece0_sq'): (3, 0.074, 'gap.md section 3(a)'),
    ('decomposition', 'run2', 'within_pair_resolution_b31', 'mean'): (4, 0.0038, 'gap.md section 2'),
    ('decomposition', 'run3', 'within_pair_resolution_b31', 'mean'): (4, 0.0024, 'gap.md section 2'),
    ('decomposition', 'hier_pooled', 'within_pair_resolution_b31', 'pairs_ge_0.011'): (0, 4, 'gap.md section 2'),
    ('decomposition', 'hier_pooled', 'rms_ece0'): (3, 0.257, 'gap.md section 3(a)'),
    ('decomposition', 'hier_pooled', 'alc_minus_v'): (4, 0.0171, 'gap.md section 3(b)'),
    ('decomposition', 'hier_pooled', 'bins', 'v < 0.10', 'pairs'): (0, 3, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v < 0.10', 'excess_by_budget'): (3, [0.129, 0.049, 0.013, 0.007, 0.002, 0.002], 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v < 0.10', 'alc_minus_v'): (3, 0.027, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v < 0.10', 'share_of_pooled_alc_minus_v'): ('pct', 28, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', '0.10 <= v < 0.20', 'pairs'): (0, 8, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', '0.10 <= v < 0.20', 'excess_by_budget'): (3, [0.083, 0.01, 0.017, 0.005, 0.005, 0.005], 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', '0.10 <= v < 0.20', 'alc_minus_v'): (3, 0.016, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', '0.10 <= v < 0.20', 'share_of_pooled_alc_minus_v'): ('pct', 44, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v >= 0.20', 'pairs'): (0, 6, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v >= 0.20', 'excess_by_budget'): (3, [0.013, 0.021, 0.035, 0.004, 0.0, 0.003], 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v >= 0.20', 'alc_minus_v'): (3, 0.013, 'gap.md section 1 table'),
    ('decomposition', 'hier_pooled', 'bins', 'v >= 0.20', 'share_of_pooled_alc_minus_v'): ('pct', 28, 'gap.md section 1 table'),
    ('item_level', 'ratios', 'TL'): (2, 0.54, 'gap.md section 2'),
    ('item_level', 'ratios', 'TL-mix'): (2, 0.46, 'gap.md section 2'),
    ('item_level', 'ratios', 'R1-bf'): (2, 0.45, 'gap.md section 2'),
    ('item_level', 'ratios', 'R1-pu'): (2, 0.44, 'gap.md section 2'),
    ('item_level', 'run1', 'share_range'): ('pct', [55, 60], "gap.md section 3 table (c), plan section 2 ('78 to 88')"),
    ('item_level', 'run1', 'v_minus_item_oracle_range'): (2, [0.07, 0.09], 'gap.md section 3 table (c)'),
    ('item_level', 'run2', 'share_range'): ('pct', [86, 88], "gap.md section 3 table (c), plan section 2 ('78 to 88')"),
    ('item_level', 'run2', 'v_minus_item_oracle_range'): (2, [0.08, 0.1], 'gap.md section 3 table (c)'),
    ('item_level', 'run3', 'share_range'): ('pct', [78, 81], "gap.md section 3 table (c), plan section 2 ('78 to 88')"),
    ('item_level', 'run3', 'v_minus_item_oracle_range'): (2, [0.07, 0.09], 'gap.md section 3 table (c)'),
    ('item_level', 'tl_measured', 'share'): ('pct', 70, 'gap.md section 3 table'),
    ('item_level', 'tl_measured', 'gap'): (3, 0.063, 'gap.md section 3 table'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (2, 3)', 'lower'): ('full', [0.016965065560194864, 0.021597790489924135, 0.020199478635255248, 0.015524899148063353, 0.010965558945781386, 0.007574523562162764], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (2, 3)', 'upper'): ('full', [0.2782892978387481, 0.16626646319405727, 0.08335972365243641, 0.037767302735033315, 0.01808500172424984, 0.009537218506628076], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (2, 3)', 'mix'): ('full', [0.048519779323441606, 0.03883733778583896, 0.02794255702677473, 0.01837446102451656, 0.011814728786912952, 0.007778837289843304], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (2, 3)', 'observed'): ('full', [0.06622931484823529, 0.02084902073058823, 0.02248572661294117, 0.004809785436470586, 0.002580197201176466, 0.0035897266129411715], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (1,)', 'lower'): ('full', [0.01721133331823566, 0.020317749632137908, 0.0193390963203887, 0.015052983253811424, 0.010588083888082876, 0.007331556106507816], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (1,)', 'upper'): ('full', [0.31012346748872044, 0.18423699298020907, 0.08859290938275986, 0.039555506990384015, 0.01811206510830335, 0.009695836823663028], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (1,)', 'mix'): ('full', [0.04875147378440711, 0.0379514227294834, 0.027255111101819097, 0.01801237964553975, 0.011483304429744861, 0.007602686097759403], 's10_pairideal.json'),
    ('ideal', 'team inference (-1.6, 1.5) | runs (1,)', 'observed'): ('full', [0.205206938781, 0.10024760544766669, 0.05066649433655557, 0.017512049892111105, 0.01567438322544444, 0.0031101610032222204], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (2, 3)', 'lower'): ('full', [0.036237140854145984, 0.03721790132262763, 0.028310275009796763, 0.019195273060739523, 0.012222517482160116, 0.007975143827586604], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (2, 3)', 'upper'): ('full', [0.14981983210487554, 0.0875820605531445, 0.04881386481440236, 0.025715433874655996, 0.014242450690362347, 0.008517047705289162], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (2, 3)', 'mix'): ('full', [0.07098012124989903, 0.05263919237940565, 0.03480400692496743, 0.021240832150284994, 0.01285741709415468, 0.00814473287053129], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (2, 3)', 'observed'): ('full', [0.06622931484823529, 0.02084902073058823, 0.02248572661294117, 0.004809785436470586, 0.002580197201176466, 0.0035897266129411715], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (1,)', 'lower'): ('full', [0.04541263278450757, 0.03927668713823382, 0.02832281152053576, 0.0180179534406087, 0.01172112892542776, 0.007597539740136167], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (1,)', 'upper'): ('full', [0.172724777582527, 0.09469026049831769, 0.05001934513707683, 0.025568206542738444, 0.013991252554969762, 0.008313972442279091], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (1,)', 'mix'): ('full', [0.07977160923040731, 0.054099119678167135, 0.034462309375949685, 0.020185066731951862, 0.012343363218839585, 0.007779137377999162], 's10_pairideal.json'),
    ('ideal', 'pooled reading (-0.70, 1.75) | runs (1,)', 'observed'): ('full', [0.205206938781, 0.10024760544766669, 0.05066649433655557, 0.017512049892111105, 0.01567438322544444, 0.0031101610032222204], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (2, 3)', 'lower'): ('full', [0.03308507576001883, 0.030790716476146247, 0.024205918415404183, 0.01725162337064082, 0.011694638207167967, 0.007886524414960506], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (2, 3)', 'upper'): ('full', [0.15859284752900105, 0.09928497828464246, 0.05421835560491751, 0.028259132932694924, 0.014851051167910764, 0.008745909044467834], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (2, 3)', 'mix'): ('full', [0.0636671704109357, 0.04758215941644906, 0.03184818293955684, 0.02006444703539125, 0.012451855223257601, 0.008071121365581831], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (2, 3)', 'observed'): ('full', [0.06622931484823529, 0.02084902073058823, 0.02248572661294117, 0.004809785436470586, 0.002580197201176466, 0.0035897266129411715], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (1,)', 'lower'): ('full', [0.04153983798579181, 0.03466310146812468, 0.026054525205006642, 0.017396732481936968, 0.01141786801288964, 0.007490287414945081], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (1,)', 'upper'): ('full', [0.18221852271124117, 0.1113197009455846, 0.06019876295526115, 0.029702329305177157, 0.015342052359941813, 0.008832694650526088], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (1,)', 'mix'): ('full', [0.07146408433434843, 0.050938783973038, 0.033681630945118765, 0.020268204405412476, 0.012343894783836514, 0.00781982293708459], 's10_pairideal.json'),
    ('ideal', 'R1-like (-0.70, 1.42) | runs (1,)', 'observed'): ('full', [0.205206938781, 0.10024760544766669, 0.05066649433655557, 0.017512049892111105, 0.01567438322544444, 0.0031101610032222204], 's10_pairideal.json'),
    ('ideal', 'summary_hier', 'lower_roots_team_prior'): (4, 0.0161, "gap.md section 3(b), plan ('+0.016')"),
    ('ideal', 'summary_hier', 'prior_weighted_range'): (3, [0.025, 0.032], "gap.md section 3(b), plan ('to +0.032')"),
    ('ideal', 'summary_hier', 'upper_roots_range'): (2, [0.05, 0.09], 'gap.md section 3(b)'),
    ('ideal', 'summary_hier', 'observed_hier'): (4, 0.0171, 'gap.md section 3(b)'),
    ('empirical_mean', 'root_agnostic', 'run1', 'budgets'): (4, [0.25, 0.3139, 0.2104, 0.1797, 0.1672, 0.1618], 's7 stdout'),
    ('empirical_mean', 'root_agnostic', 'run1', 'alc'): (4, 0.2154, 's7 stdout (gap.md: 0.215, 0.246, 0.224)'),
    ('empirical_mean', 'root_agnostic', 'run2', 'budgets'): (4, [0.25, 0.3636, 0.2433, 0.2084, 0.1947, 0.1887], 's7 stdout'),
    ('empirical_mean', 'root_agnostic', 'run2', 'alc'): (4, 0.2459, 's7 stdout (gap.md: 0.215, 0.246, 0.224)'),
    ('empirical_mean', 'root_agnostic', 'run3', 'budgets'): (4, [0.25, 0.3273, 0.2181, 0.1875, 0.175, 0.1694], 's7 stdout'),
    ('empirical_mean', 'root_agnostic', 'run3', 'alc'): (4, 0.2235, 's7 stdout (gap.md: 0.215, 0.246, 0.224)'),
    ('empirical_mean', 'v_scan_n54', 0, '1'): (4, 0.1506, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 1, '1'): (4, 0.1629, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 2, '1'): (4, 0.1754, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 3, '1'): (4, 0.1823, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 4, '1'): (4, 0.1867, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 5, '1'): (4, 0.1977, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 6, '1'): (4, 0.211, 's7 stdout'),
    ('empirical_mean', 'v_scan_n54', 7, '1'): (4, 0.2231, 's7 stdout'),
    ('empirical_mean', 'v_at_organisers_entry'): (3, 0.124, "gap.md section 4 ('v ~ 0.124')"),
    ('nonmonotone', 'prior=truth logit N(-0.7,1.42)', 'B1>B0+0.01'): (2, 0.3, 's7 stdout'),
    ('nonmonotone', 'prior=truth logit N(-0.7,1.42)', 'B3>B1+0.01'): (2, 0.24, 's7 stdout'),
    ('nonmonotone', 'prior=truth logit N(-1.29,1.7)', 'B1>B0+0.01'): (2, 0.26, 's7 stdout'),
    ('nonmonotone', 'prior=truth logit N(-1.29,1.7)', 'B3>B1+0.01'): (2, 0.21, 's7 stdout'),
    ('nonmonotone', 'P(>=8 of 17 with B3>B1+0.01)', '0.2'): (3, 0.011, 's7 stdout'),
    ('nonmonotone', 'P(>=8 of 17 with B3>B1+0.01)', '0.25'): (3, 0.04, 's7 stdout'),
    ('nonmonotone', 'P(>=8 of 17 with B3>B1+0.01)', '0.3'): (3, 0.105, 's7 stdout'),
    ('run_spread', 'v_iid', 'mean'): ('full', 0.1643807068773703, 's7_spread.json'),
    ('run_spread', 'v_iid', 'sd'): ('full', 0.023509160904513784, 's7_spread.json'),
    ('run_spread', 'v_strat', 'mean'): ('full', 0.15988000144731387, 's7_spread.json'),
    ('run_spread', 'v_strat', 'sd'): ('full', 0.016893807108869008, 's7_spread.json'),
    ('run_spread', 'alc_hier_iid', 'mean'): ('full', 0.18691457840055556, 's7_spread.json'),
    ('run_spread', 'alc_hier_iid', 'sd'): ('full', 0.02048121721514321, 's7_spread.json'),
    ('run_spread', 'alc_hier_strat', 'mean'): ('full', 0.18676271643722225, 's7_spread.json'),
    ('run_spread', 'alc_hier_strat', 'sd'): ('full', 0.014268238722781473, 's7_spread.json'),
    ('run_spread', 'v_iid', 'p5'): (4, 0.1239, 's7 stdout'),
    ('run_spread', 'v_iid', 'p1'): (4, 0.1046, 's7 stdout'),
    ('run_spread', 'v_iid', 'P_le_0.128'): (4, 0.0678, 's7 stdout'),
    ('run_spread', 'v_iid', 'P_le_0.117'): (5, 0.02985, 's7 stdout'),
    ('run_spread', 'v_iid', 'P_le_0.10'): (5, 0.00605, 's7 stdout'),
    ('run_spread', 'v_strat', 'p5'): (4, 0.1314, 's7 stdout'),
    ('run_spread', 'v_strat', 'p1'): (4, 0.12, 's7 stdout'),
    ('run_spread', 'v_strat', 'P_le_0.128'): (4, 0.0338, 's7 stdout'),
    ('run_spread', 'v_strat', 'P_le_0.117'): (5, 0.0066, 's7 stdout'),
    ('run_spread', 'v_strat', 'P_le_0.10'): (5, 0.0003, 's7 stdout'),
    ('run_spread', 'alc_hier_iid', 'p5'): (4, 0.1523, 's7 stdout'),
    ('run_spread', 'alc_hier_iid', 'p1'): (4, 0.1366, 's7 stdout'),
    ('run_spread', 'alc_hier_iid', 'P_le_0.128'): (4, 0.003, 's7 stdout'),
    ('run_spread', 'alc_hier_iid', 'P_le_0.117'): (5, 0.0006, 's7 stdout'),
    ('run_spread', 'alc_hier_iid', 'P_le_0.10'): (5, 0.0, 's7 stdout'),
    ('run_spread', 'alc_hier_strat', 'p5'): (4, 0.1628, 's7 stdout'),
    ('run_spread', 'alc_hier_strat', 'p1'): (4, 0.1533, 's7 stdout'),
    ('run_spread', 'alc_hier_strat', 'P_le_0.128'): (4, 0.0, 's7 stdout'),
    ('run_spread', 'alc_hier_strat', 'P_le_0.117'): (5, 0.0, 's7 stdout'),
    ('run_spread', 'alc_hier_strat', 'P_le_0.10'): (5, 0.0, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '1', 'P_min_le_0.117_iid'): (4, 0.0006, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '1', 'P_min_le_0.117_strat'): (4, 0.0, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '1', 'expected_min_iid_normal'): (4, 0.1869, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '10', 'P_min_le_0.117_iid'): (4, 0.006, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '10', 'P_min_le_0.117_strat'): (4, 0.0, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '10', 'expected_min_iid_normal'): (4, 0.1554, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '50', 'P_min_le_0.117_iid'): (4, 0.0296, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '50', 'P_min_le_0.117_strat'): (4, 0.0, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '50', 'expected_min_iid_normal'): (4, 0.1409, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '200', 'P_min_le_0.117_iid'): (4, 0.1131, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '200', 'P_min_le_0.117_strat'): (4, 0.0, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '200', 'expected_min_iid_normal'): (4, 0.1307, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '1000', 'P_min_le_0.117_iid'): (4, 0.4513, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '1000', 'P_min_le_0.117_strat'): (4, 0.0, 's7 stdout'),
    ('run_spread', 'best_of_n_hier', '1000', 'expected_min_iid_normal'): (4, 0.1205, 's7 stdout'),
    ('run_spread', 'mean_alc_reaching_0.117_once_in_n', '10'): (4, 0.1433, 's7 stdout'),
    ('run_spread', 'mean_alc_reaching_0.117_once_in_n', '50'): (4, 0.1591, 's7 stdout'),
    ('run_spread', 'mean_alc_reaching_0.117_once_in_n', '200'): (4, 0.1698, 's7 stdout'),
    ('entry', 'run1', 'plug_in_b31'): (4, 0.215, 'formative.json'),
    ('entry', 'run1', 'mc_mean'): (4, 0.2146, 'formative.json'),
    ('entry', 'run1', 'mc_sd'): (4, 0.0186, 'formative.json'),
    ('entry', 'run1', 'P_mc_le_entry'): ('full', 0.0185, 'formative.json'),
    ('entry', 'run2', 'plug_in_b31'): (4, 0.2467, 'formative.json'),
    ('entry', 'run2', 'mc_mean'): (4, 0.2468, 'formative.json'),
    ('entry', 'run2', 'mc_sd'): (4, 0.0183, 'formative.json'),
    ('entry', 'run2', 'P_mc_le_entry'): ('full', 0.0, 'formative.json'),
    ('entry', 'run3', 'plug_in_b31'): (4, 0.224, 'formative.json'),
    ('entry', 'run3', 'mc_mean'): (4, 0.2238, 'formative.json'),
    ('entry', 'run3', 'mc_sd'): (4, 0.0197, 'formative.json'),
    ('entry', 'run3', 'P_mc_le_entry'): ('full', 0.0045, 'formative.json'),
    ('entry', 'level_reading_of_entry'): (3, 0.128, 'docs/report/draft.md Figure A1'),
    ('vadjusted', 'run1', 'alc_minus_v'): ('full', 0.057662910487777795, 's9_vadj.json hidden'),
    ('vadjusted', 'run1', 'pair_sd'): (4, 0.0569, 's9 stdout'),
    ('vadjusted', 'run1', 'se'): (4, 0.019, 's9 stdout'),
    ('vadjusted', 'run2', 'alc_minus_v'): ('full', 0.012956916606125, 's9_vadj.json hidden'),
    ('vadjusted', 'run2', 'pair_sd'): (4, 0.0131, 's9 stdout'),
    ('vadjusted', 'run2', 'se'): (4, 0.0046, 's9 stdout'),
    ('vadjusted', 'run3', 'alc_minus_v'): ('full', 0.020833457730111117, 's9_vadj.json hidden'),
    ('vadjusted', 'run3', 'pair_sd'): (4, 0.0078, 's9 stdout'),
    ('vadjusted', 'run3', 'se'): (4, 0.0026, 's9 stdout'),
    ('vadjusted', 'legacy_minus_hier', 'v_adjusted'): (4, 0.0405, 's9 stdout, plan section 5 R1'),
    ('vadjusted', 'legacy_minus_hier', 'se'): (4, 0.0191, 's9 stdout (plan: 0.019)'),
    ('vadjusted', 'legacy_minus_hier', 'raw'): (4, 0.0245, 's9 stdout'),
    ('vadjusted', 'run3_minus_run2', 'v_adjusted'): (4, 0.0079, 's9 stdout'),
    ('vadjusted', 'run3_minus_run2', 'se'): (4, 0.0053, 's9 stdout'),
    ('vadjusted', 'run3_minus_run2', 'raw'): (4, -0.011, 's9 stdout'),
    ('vadjusted', 'test_like', 'blocks', '200-299', 'shipped_minus_legacy'): (4, -0.0419, 'plan section 5 R1'),
    ('vadjusted', 'test_like', 'blocks', '200-299', 'cluster_se'): (4, 0.0037, 'plan section 5 R1'),
    ('curve', 'test', 'platform', 'composition_by_bin'): (0, [4, 6, 7], 'curve_test.json'),
    ('curve', 'test', 'platform', 'B7->B31_mean'): (4, -0.0012, 'curve_test.json'),
    ('curve', 'test', 'platform', 'B7->B31_sd'): (4, 0.0114, 'curve_test.json'),
    ('curve', 'test', 'platform', 'B1->B3_mean'): (4, 0.0016, 'curve_test.json'),
    ('curve', 'test', 'platform', 'B1->B7_mean'): (4, -0.016, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'pairs'): (0, 2425, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'B7->B31_mean_of17'): (4, [-0.0145, 0.0065], 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'P(mean >= platform)'): ('full', 0.01105, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'B7->B31_sd_of17'): (4, 0.0258, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'P(sd <= platform)'): ('full', 0.0101, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'P(mean>=plat and sd<=plat)'): ('full', 0.0005, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'B1->B3_mean_of17'): (4, [-0.0196, 0.0128], 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'P(B1->B3 >= platform)'): ('full', 0.0378, 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'B1->B7_mean_of17'): (4, [-0.0302, 0.0139], 'curve_test.json'),
    ('curve', 'test', 'replica', 'tl', 'P(B1->B7 >= platform)'): ('full', 0.15065, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'pairs'): (0, 1307, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'B7->B31_mean_of17'): (4, [-0.0215, 0.0076], 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'P(mean >= platform)'): ('full', 0.00065, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'B7->B31_sd_of17'): (4, 0.0305, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'P(sd <= platform)'): ('full', 0.00115, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'P(mean>=plat and sd<=plat)'): ('full', 0.0, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'B1->B3_mean_of17'): (4, [-0.0123, 0.0132], 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'P(B1->B3 >= platform)'): ('full', 0.1417, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'B1->B7_mean_of17'): (4, [-0.029, 0.0148], 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1b', 'P(B1->B7 >= platform)'): ('full', 0.19405, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'pairs'): (0, 874, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'B7->B31_mean_of17'): (4, [-0.0229, 0.0084], 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'P(mean >= platform)'): ('full', 0.00205, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'B7->B31_sd_of17'): (4, 0.0333, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'P(sd <= platform)'): ('full', 0.0008, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'P(mean>=plat and sd<=plat)'): ('full', 0.0, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'B1->B3_mean_of17'): (4, [-0.0145, 0.0138], 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'P(B1->B3 >= platform)'): ('full', 0.11365, 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'B1->B7_mean_of17'): (4, [-0.0283, 0.0154], 'curve_test.json'),
    ('curve', 'test', 'replica', 'r1p', 'P(B1->B7 >= platform)'): ('full', 0.21655, 'curve_test.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B0->B1'): (4, [-0.0454, 0.0116], 'curve_shape.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B1->B3'): (4, [0.0016, 0.0068], 'curve_shape.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B3->B7'): (4, [-0.0177, 0.0066], 'curve_shape.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B7->B15'): (4, [-0.0022, 0.0017], 'curve_shape.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B15->B31'): (4, [0.001, 0.0026], 'curve_shape.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B7->B31'): (4, [-0.0012, 0.0028], 'curve_shape.json'),
    ('curve', 'shape', 'platform_steps_mean_se', 'B1->B7'): (4, [-0.016, 0.0052], 'curve_shape.json'),
    ('curve', 'shape', 'replica_mean_steps', 'tl', 'B0->B1'): (4, -0.0273, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl', 'B1->B3'): (4, -0.0247, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl', 'B3->B7'): (4, -0.012, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl', 'B7->B31'): (4, -0.0139, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl mix/whole', 'B0->B1'): (4, -0.0247, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl mix/whole', 'B1->B3'): (4, -0.0239, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl mix/whole', 'B3->B7'): (4, -0.0102, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl mix/whole', 'B7->B31'): (4, -0.0136, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl no shift', 'B0->B1'): (4, -0.0185, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl no shift', 'B1->B3'): (4, -0.0143, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl no shift', 'B3->B7'): (4, -0.0141, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'tl no shift', 'B7->B31'): (4, -0.0126, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1b', 'B0->B1'): (4, -0.0107, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1b', 'B1->B3'): (4, -0.0115, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1b', 'B3->B7'): (4, -0.0155, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1b', 'B7->B31'): (4, -0.0202, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1p', 'B0->B1'): (4, -0.0227, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1p', 'B1->B3'): (4, -0.0146, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1p', 'B3->B7'): (4, -0.014, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'replica_mean_steps', 'r1p', 'B7->B31'): (4, -0.0221, "curve_shape.json (from App E.7's 4-dp means)"),
    ('curve', 'shape', 'per_run', 'run2', 'b7_minus_b31'): (4, 0.0003, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run3', 'b7_minus_b31'): (4, 0.002, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run2', 'share_of_drop_by_B1'): ('pct', 79, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run3', 'share_of_drop_by_B1'): ('pct', 68, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run1', 'share_of_drop_by_B1'): ('pct', 52, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run1', 'share_of_drop_by_B3'): ('pct', 76, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run1', 'share_of_drop_by_B7'): ('pct', 93, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run2', 'b3_minus_b1'): (4, 0.0016, 'gap.md section 1'),
    ('curve', 'shape', 'per_run', 'run3', 'b3_minus_b1'): (4, 0.0017, 'gap.md section 1'),
    ('curve', 'shape', 'pairs_B3_gt_B1_plus_0.01'): (0, 8, 'gap.md section 1'),
    ('curve', 'shape', 'pairs_B1_gt_B0_plus_0.01'): (0, 2, 'gap.md section 1'),
    ('split_floor', 'run2', 'split_noise'): (4, 0.0061, 'split_floor.json'),
    ('split_floor', 'run2', 'sampling_31'): (4, 0.0026, 'split_floor.json'),
    ('split_floor', 'run2', 'floor_pure_rate_learner_B31'): (4, 0.0088, 'split_floor.json'),
    ('split_floor', 'run2', 'platform_excess_B31'): (4, 0.0074, 'split_floor.json'),
    ('split_floor', 'run2', 'platform_excess_B15'): (4, 0.0038, 'split_floor.json'),
    ('split_floor', 'run2', 'platform_excess_B7'): (4, 0.0083, 'split_floor.json'),
    ('split_floor', 'run3', 'split_noise'): (4, 0.0061, 'split_floor.json'),
    ('split_floor', 'run3', 'sampling_31'): (4, 0.0021, 'split_floor.json'),
    ('split_floor', 'run3', 'floor_pure_rate_learner_B31'): (4, 0.0082, 'split_floor.json'),
    ('split_floor', 'run3', 'platform_excess_B31'): (4, 0.006, 'split_floor.json'),
    ('split_floor', 'run3', 'platform_excess_B15'): (4, 0.0065, 'split_floor.json'),
    ('split_floor', 'run3', 'platform_excess_B7'): (4, 0.0075, 'split_floor.json'),
    ('replica', 'validation', 'TUNED', 'bias_by_budget', '0'): (4, -0.0001, 's6 stdout'),
    ('replica', 'validation', 'TUNED', 'sd_by_budget', '0'): (3, 0.001, 's6 stdout'),
    ('replica', 'validation', 'TUNED', 'bias_by_budget', '5'): (4, -0.0098, 's6 stdout'),
    ('replica', 'validation', 'TUNED', 'b31_minus_v'): (4, 0.0, 's6 stdout'),
    ('replica', 'validation', 'READING', 'bias_by_budget', '0'): (4, -0.0001, 's6 stdout'),
    ('replica', 'validation', 'READING', 'sd_by_budget', '0'): (3, 0.001, 's6 stdout'),
    ('replica', 'validation', 'READING', 'bias_by_budget', '5'): (4, -0.012, 's6 stdout'),
    ('replica', 'validation', 'READING', 'b31_minus_v'): (4, -0.0008, 's6 stdout'),
    ('replica', 'validation', 'AUDIT', 'bias_by_budget', '0'): (4, -0.0001, 's6 stdout'),
    ('replica', 'validation', 'AUDIT', 'sd_by_budget', '0'): (3, 0.001, 's6 stdout'),
    ('replica', 'validation', 'AUDIT', 'bias_by_budget', '5'): (4, -0.009, 's6 stdout'),
    ('replica', 'validation', 'AUDIT', 'b31_minus_v'): (4, 0.0007, 's6 stdout'),
    ('replica', 'validation', 'MIXTURE', 'bias_by_budget', '0'): (4, -0.0002, 's6 stdout'),
    ('replica', 'validation', 'MIXTURE', 'sd_by_budget', '0'): (3, 0.002, 's6 stdout'),
    ('replica', 'validation', 'MIXTURE', 'bias_by_budget', '5'): (4, -0.0114, 's6 stdout'),
    ('replica', 'validation', 'MIXTURE', 'b31_minus_v'): (4, 0.0, 's6 stdout'),
    ('replica', 'validation', 'FLAT', 'bias_by_budget', '0'): (4, -0.0001, 's6 stdout'),
    ('replica', 'validation', 'FLAT', 'sd_by_budget', '0'): (3, 0.001, 's6 stdout'),
    ('replica', 'validation', 'FLAT', 'bias_by_budget', '5'): (4, -0.0092, 's6 stdout'),
    ('replica', 'validation', 'FLAT', 'b31_minus_v'): (4, 0.0005, 's6 stdout'),
    ('replica', 'validation', 'R1B', 'bias_by_budget', '0'): (4, -0.0001, 's6 stdout'),
    ('replica', 'validation', 'R1B', 'sd_by_budget', '0'): (3, 0.001, 's6 stdout'),
    ('replica', 'validation', 'R1B', 'bias_by_budget', '5'): (4, -0.0201, 's6 stdout'),
    ('replica', 'validation', 'R1B', 'b31_minus_v'): (4, -0.0035, 's6 stdout'),
    ('replica', 'validation', 'R1P', 'bias_by_budget', '0'): (4, -0.0, 's6 stdout'),
    ('replica', 'validation', 'R1P', 'sd_by_budget', '0'): (3, 0.001, 's6 stdout'),
    ('replica', 'validation', 'R1P', 'bias_by_budget', '5'): (4, -0.0183, 's6 stdout'),
    ('replica', 'validation', 'R1P', 'b31_minus_v'): (4, -0.0036, 's6 stdout'),
    ('replica', 'validation_summary', 'b0_sd_range', '1'): (3, 0.001, "gap.md section 2 ('sd 0.001 in all 7 regimes')"),
    ('replica', 'validation_summary', 'b31_bias_range'): (3, [-0.02, -0.009], 'gap.md section 2'),
    ('replica', 'noise', 'TUNED', 'ship', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'ship', 'alc'): ('full', 0.16901302978713426, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'ship', 'alc_sd'): ('full', 0.02685790009939263, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'ship', 'v'): ('full', 0.14111756428596117, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'ship', 'alc_minus_v'): ('full', 0.027895465501173083, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'ship', 'alc_minus_v_sd'): ('full', 0.009685733710133747, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'ship', 'excess_by_budget'): ('full', [0.07427325437210218, 0.051868425514707696, 0.029070572673877977, 0.01493280697594306, 0.006557878357483285, -0.00017796640439542616], 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'alc'): ('full', 0.21237304963585507, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'alc_sd'): ('full', 0.02135502293862037, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'v'): ('full', 0.14111756428596117, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'alc_minus_v'): ('full', 0.0712554853498939, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'alc_minus_v_sd'): ('full', 0.018689769051021567, 's9_vadj.json tab'),
    ('replica', 'noise', 'TUNED', 'legacy', 'excess_by_budget'): ('full', [0.2118915577166311, 0.13181179195633494, 0.06681611876988816, 0.03226935167396826, 0.014990287523688297, 0.008888195934548447], 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'alc'): ('full', 0.1881296720286169, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'alc_sd'): ('full', 0.02598957760837729, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'v'): ('full', 0.16065394619271658, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'alc_minus_v'): ('full', 0.027475725835900366, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'alc_minus_v_sd'): ('full', 0.00954633882395929, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'ship', 'excess_by_budget'): ('full', [0.07399370546747278, 0.05127484583796667, 0.02773523106946877, 0.015344638090274235, 0.006647524748552498, -0.0012409266009935605], 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'alc'): ('full', 0.21296140350152407, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'alc_sd'): ('full', 0.023163025589635466, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'v'): ('full', 0.16065394619271658, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'alc_minus_v'): ('full', 0.0523074573088075, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'alc_minus_v_sd'): ('full', 0.013717389241857022, 's9_vadj.json tab'),
    ('replica', 'noise', 'READING', 'legacy', 'excess_by_budget'): ('full', [0.15193237046399016, 0.09446691490706861, 0.04808191224351162, 0.025657915245386687, 0.013894637645555472, 0.006939442541039965], 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'alc'): ('full', 0.16503027031647055, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'alc_sd'): ('full', 0.029805397719857667, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'v'): ('full', 0.13428282708479924, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'alc_minus_v'): ('full', 0.030747443231671313, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'alc_minus_v_sd'): ('full', 0.00864992540853147, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'ship', 'excess_by_budget'): ('full', [0.09199502311781846, 0.05652664837159471, 0.029644859993174212, 0.014123428648440484, 0.007048326313737145, 0.0007928825450015444], 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'alc'): ('full', 0.19634572293540403, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'alc_sd'): ('full', 0.025365678032847127, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'v'): ('full', 0.13428282708479924, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'alc_minus_v'): ('full', 0.06206289585060478, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'alc_minus_v_sd'): ('full', 0.016802221669245013, 's9_vadj.json tab'),
    ('replica', 'noise', 'AUDIT', 'legacy', 'excess_by_budget'): ('full', [0.1900282823407023, 0.11291602784337984, 0.0585630175743498, 0.026218398830335044, 0.013624495655534558, 0.007956796358146865], 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'alc'): ('full', 0.1830425090568601, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'alc_sd'): ('full', 0.027574128346798274, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'v'): ('full', 0.15403472645850477, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'alc_minus_v'): ('full', 0.02900778259835531, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'alc_minus_v_sd'): ('full', 0.009131953935225926, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'ship', 'excess_by_budget'): ('full', [0.0771182902474386, 0.05343876348210932, 0.029634206198032597, 0.016551022857239632, 0.006797366492124793, 0.00011681767710173094], 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'alc'): ('full', 0.20933049878113233, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'alc_sd'): ('full', 0.026864774100833, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'v'): ('full', 0.15403472645850477, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'alc_minus_v'): ('full', 0.05529577232262752, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'alc_minus_v_sd'): ('full', 0.014294116519694927, 's9_vadj.json tab'),
    ('replica', 'noise', 'MIXTURE', 'legacy', 'excess_by_budget'): ('full', [0.1581064835309702, 0.09801196426194196, 0.05265682201675591, 0.02738454281272269, 0.01483667067410344, 0.009071240164256793], 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'alc'): ('full', 0.17391536483905098, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'alc_sd'): ('full', 0.026472284905277744, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'v'): ('full', 0.14115420194275496, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'alc_minus_v'): ('full', 0.03276116289629605, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'alc_minus_v_sd'): ('full', 0.009369592939737666, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'ship', 'excess_by_budget'): ('full', [0.1004595684212199, 0.06213699988766961, 0.03074824009330938, 0.014329305949511418, 0.006270359439159485, 0.0001822498024407463], 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'runs'): (0, 80, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'alc'): ('full', 0.19396280596297782, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'alc_sd'): ('full', 0.025285214415331363, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'v'): ('full', 0.14115420194275496, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'alc_minus_v'): ('full', 0.05280860402022286, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'alc_minus_v_sd'): ('full', 0.017449408079910973, 's9_vadj.json tab'),
    ('replica', 'noise', 'FLAT', 'legacy', 'excess_by_budget'): ('full', [0.15129410499980384, 0.09563410592145066, 0.052509765719459076, 0.024690973318550295, 0.012191020229412399, 0.006740204824679795], 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'runs'): (0, 60, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'alc'): ('full', 0.21140445502062377, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'alc_sd'): ('full', 0.0229343101488602, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'v'): ('full', 0.1864268218119044, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'alc_minus_v'): ('full', 0.02497763320871937, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'alc_minus_v_sd'): ('full', 0.00924988585593701, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'ship', 'excess_by_budget'): ('full', [0.05669445891619788, 0.04364403335475842, 0.030503464641187875, 0.017854774551199007, 0.006257751530593723, -0.0034381749844823385], 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'runs'): (0, 60, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'alc'): ('full', 0.21164308380273958, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'alc_sd'): ('full', 0.02025962937349248, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'v'): ('full', 0.1864268218119044, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'alc_minus_v'): ('full', 0.025216261990835195, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'alc_minus_v_sd'): ('full', 0.00859018028337754, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1B', 'legacy', 'excess_by_budget'): ('full', [0.04383823087328997, 0.03859352853465902, 0.03115189089544274, 0.021899197070826378, 0.012116524536071267, 0.0008021069610630242], 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'runs'): (0, 60, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'alc'): ('full', 0.1948031152283906, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'alc_sd'): ('full', 0.025570539479104093, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'v'): ('full', 0.17088191630624147, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'alc_minus_v'): ('full', 0.023921198922149146, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'alc_minus_v_sd'): ('full', 0.010932827031151513, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'ship', 'excess_by_budget'): ('full', [0.058969214422054814, 0.0439173430772201, 0.028472721130135208, 0.015618596669414283, 0.0040254614860186205, -0.003825469926139927], 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'runs'): (0, 60, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'alc'): ('full', 0.19875134762241467, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'alc_sd'): ('full', 0.021254815649971986, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'v'): ('full', 0.17088191630624147, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'alc_minus_v'): ('full', 0.027869431316173178, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'alc_minus_v_sd'): ('full', 0.010153754037300481, 's9_vadj.json tab'),
    ('replica', 'noise', 'R1P', 'legacy', 'excess_by_budget'): ('full', [0.057106641660674244, 0.045672025341660996, 0.03214004155129941, 0.020981060816396956, 0.01197800168781129, 4.541270672015416e-05], 's9_vadj.json tab'),
    ('replica', 'noise_summary', 'ship_alc_sd_range'): (3, [0.023, 0.03], 'gap.md section 5.2'),
    ('replica', 'noise_summary', 'ship_alc_minus_v_sd_range', '0'): (4, 0.0086, 'gap.md section 5.2'),
    ('replica', 'noise_summary', 'ship_alc_minus_v_sd_range', '1'): (3, 0.011, 'gap.md section 5.2'),
    ('replica', 'placement_summary', 'run1_vs_R1B_B0_B1'): (1, [8.9, 4.2], 'gap.md section 5.3'),
    ('replica', 'placement_summary', 'run1_vs_R1P_B0_B1'): (1, [7.2, 2.8], 'gap.md section 5.3'),
    ('replica', 'placement_summary', 'run1_vs_test_like_range'): (1, [-1.3, 1.0], 'gap.md section 5.3'),
    ('replica', 'placement_summary', 'hier_per_run_range'): (1, [-2.0, 0.8], 'gap.md section 5.4'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'v'): ('full', 0.1694450247933884, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'item'): ('full', 0.10956156932823084, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal_budgets_r1'): ('full', [0.1864767640887446, 0.16209997894417283, 0.14144690758889752, 0.12819170186882514, 0.119239460390986, 0.11464774096460098], 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.0'): ('full', 0.2001204889943665, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.3'): ('full', 0.19527571317296022, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.5'): ('full', 0.18646267394290966, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.7'): ('full', 0.17270000438577648, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.8'): ('full', 0.16368707763339393, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.9'): ('full', 0.15297956020385323, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.95'): ('full', 0.14688598076700635, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '1.0'): ('full', 0.14030806026391088, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.0'): ('full', 0.17250365051925343, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.3'): ('full', 0.16773598894506442, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.5'): ('full', 0.15878553551450122, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.7'): ('full', 0.14455344815174018, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.8'): ('full', 0.13506890971289673, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.9'): ('full', 0.12361205897809836, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.95'): ('full', 0.11698028922428887, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '1.0'): ('full', 0.10956156932823101, 's8_req.json'),
    ('requirement', 'tau=2.0 | hidden-like v~0.165 (-0.70,1.42)', 'req_r_known'): ('full', 0.9498513912843565, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'v'): ('full', 0.13119788429752066, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'item'): ('full', 0.08702829148783393, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal_budgets_r1'): ('full', [0.14810569273289478, 0.13079728538656926, 0.11376832441161715, 0.10374256532701559, 0.09560708313659931, 0.09144762314464738], 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.0'): ('full', 0.1547231061507956, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.3'): ('full', 0.15144611117674792, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.5'): ('full', 0.14539513741906157, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.7'): ('full', 0.1357949852305203, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.8'): ('full', 0.1294221904225845, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.9'): ('full', 0.12178747450295664, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.95'): ('full', 0.11742909849133024, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '1.0'): ('full', 0.11273838324011448, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.0'): ('full', 0.13354352503078895, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.3'): ('full', 0.1300761857861546, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.5'): ('full', 0.12355280819844042, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.7'): ('full', 0.1130897842800227, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.8'): ('full', 0.10605752645790921, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.9'): ('full', 0.09750267586494585, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.95'): ('full', 0.09252659137129979, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '1.0'): ('full', 0.08702829148783393, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'req_r_cal'): ('full', 0.954573913234437, 's8_req.json'),
    ('requirement', 'tau=2.0 | organisers-run-like v~0.125 (-1.64,1.49)', 'req_r_known'): ('full', 0.6252564889373083, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'v'): ('full', 0.16851596694214876, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'item'): ('full', 0.09188839387033995, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal_budgets_r1'): ('full', [0.17574404335564242, 0.1485961656515381, 0.12557263783124126, 0.11045444616418247, 0.10126564694335412, 0.09696108879794443], 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.0'): ('full', 0.19929279322573834, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.3'): ('full', 0.1934120050741877, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.5'): ('full', 0.1827240703357216, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.7'): ('full', 0.16586975925361558, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.8'): ('full', 0.15465867187500534, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.9'): ('full', 0.1410681473266523, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.95'): ('full', 0.1331529382811896, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '1.0'): ('full', 0.12444829253342188, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.0'): ('full', 0.17149888985645353, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.3'): ('full', 0.16571006727491758, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.5'): ('full', 0.15484551445459097, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.7'): ('full', 0.13736271408237144, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.8'): ('full', 0.12549333791642608, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.9'): ('full', 0.11079536625740116, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.95'): ('full', 0.1020270279872298, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '1.0'): ('full', 0.09188839387034001, 's8_req.json'),
    ('requirement', 'tau=2.6 | hidden-like v~0.165 (-0.70,1.42)', 'req_r_known'): ('full', 0.8577857823750188, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'v'): ('full', 0.1301913388429752, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'item'): ('full', 0.07295756420398701, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal_budgets_r1'): ('full', [0.14013277614412312, 0.12017238316366513, 0.10167643372381781, 0.08951095653296637, 0.08216315815630484, 0.07760740119817441], 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.0'): ('full', 0.15373988685052878, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.3'): ('full', 0.14977011118978087, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.5'): ('full', 0.142388662465573, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.7'): ('full', 0.13050334056786642, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.8'): ('full', 0.12245800257277792, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.9'): ('full', 0.11259723839707915, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.95'): ('full', 0.10682940235778195, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '1.0'): ('full', 0.10047860404958057, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.0'): ('full', 0.1325245954816042, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.3'): ('full', 0.1283314013217847, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.5'): ('full', 0.12037379299280072, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.7'): ('full', 0.10738882645032671, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.8'): ('full', 0.09846345467043517, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.9'): ('full', 0.08730427891495154, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.95'): ('full', 0.08060386799992676, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '1.0'): ('full', 0.07295756420398691, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'req_r_cal'): ('full', 0.8553507058431518, 's8_req.json'),
    ('requirement', 'tau=2.6 | organisers-run-like v~0.125 (-1.64,1.49)', 'req_r_known'): ('full', 0.5519646004749414, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'v'): ('full', 0.16788601652892562, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'item'): ('full', 0.07843218250423994, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal_budgets_r1'): ('full', [0.16845340552165408, 0.13818073369201833, 0.11271845012962577, 0.09768633052546644, 0.08817719335119632, 0.08357594802013127], 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.0'): ('full', 0.19867212035778561, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.3'): ('full', 0.1920597403186746, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.5'): ('full', 0.1800878545016989, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.7'): ('full', 0.16108271886394282, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.8'): ('full', 0.14828732796806268, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.9'): ('full', 0.13250105021307754, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '0.95'): ('full', 0.12310289768884626, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'cal', '1.0'): ('full', 0.11255547689383993, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.0'): ('full', 0.17084476991918035, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.3'): ('full', 0.16431728102253915, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.5'): ('full', 0.15212671100147104, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.7'): ('full', 0.13233269806576906, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.8'): ('full', 0.11869287187102702, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.9'): ('full', 0.10146246712014381, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '0.95'): ('full', 0.09092616073470106, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'known_rate', '1.0'): ('full', 0.07843218250424004, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'req_r_cal'): ('full', 0.9789307585591714, 's8_req.json'),
    ('requirement', 'tau=3.2 | hidden-like v~0.165 (-0.70,1.42)', 'req_r_known'): ('full', 0.809824910647791, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'v'): ('full', 0.12946803305785123, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'item'): ('full', 0.06264905408347148, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal_budgets_r1'): ('full', [0.13474892278455178, 0.11299548614703178, 0.09237075418174964, 0.07918345634927684, 0.07182350828990736, 0.06738889182707604], 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.0'): ('full', 0.15290986615951732, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.3'): ('full', 0.14847409196834685, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.5'): ('full', 0.14022553802472942, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.7'): ('full', 0.12682530275653198, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.8'): ('full', 0.11762872783411539, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.9'): ('full', 0.10614029821397414, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '0.95'): ('full', 0.09925934906629413, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'cal', '1.0'): ('full', 0.09148842245475591, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.0'): ('full', 0.13174752574156157, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.3'): ('full', 0.12704057836449592, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.5'): ('full', 0.11812767098338764, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.7'): ('full', 0.1034470743593341, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.8'): ('full', 0.09320743716464788, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.9'): ('full', 0.08013997271683333, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '0.95'): ('full', 0.07208589936019069, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'known_rate', '1.0'): ('full', 0.0626490540834716, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'req_r_cal'): ('full', 0.8054727047551662, 's8_req.json'),
    ('requirement', 'tau=3.2 | organisers-run-like v~0.125 (-1.64,1.49)', 'req_r_known'): ('full', 0.5153627405243189, 's8_req.json'),
    ('requirement', 'summary', 'hidden_like_req_r_known'): (2, [0.95, 0.86, 0.81], "gap.md section 4, plan ('0.8 to 0.95')"),
}

#: scratch values the port does not reproduce as stated, and why
_DOUBLE = ("gap.md's bin table has no saved script and appears to round twice (to 4 dp, then to 3 dp "
           "half up): {got} prints as {a} at 3 dp but as {b} through {c}. The bin counts, the other "
           "budgets, ALC - v and the shares all match.")
_E7 = ("curve_shape.py differenced the regime means as App E.7 prints them (4 dp); the port differences "
       "the 6-dp means of results/ship_confirm.json. They agree to 1e-4 (largest gap 8e-5).")
EXPLAINED = {
    "identity/run1/per_pair_range_max":
        "gap.md wrote 'per-pair range 0.0' (6 decimals) and the plan 'constant across all six budgets to "
        "1e-6'. The largest per-pair range of B - ECE^2 over the six budgets is 1.13e-6; every one of run "
        "1's 9 pairs lies within the bound the tables' 6-dp rounding allows (1e-6 x (1 + the pair's two "
        "largest ECEs), at most 0.95 of it), and the run mean varies by 1.5e-7 across budgets. The "
        "identity is exact up to the tables' rounding: to 1e-6 for the run mean, to about 2e-6 per pair.",
    "decomposition/run2/alc_minus_v_ece2_convention":
        "protocol.md section 7's 0.0156 sums the 4-dp-rounded budget means of ECE^2 (0.0572, 0.0152, "
        "0.0186, 0.0083, 0.0038, 0.0074); at full precision the weighted sum is 0.015656, i.e. 0.0157.",
    "decomposition/hier_pooled/within_pair_resolution_b31/pairs_ge_0.011":
        "the fourth pair sits at 0.01092, which gap.md counted at 3-dp rounding (0.011). At full precision "
        "3 of the 17 pairs reach 0.011 and 4 reach 0.0109.",
    "decomposition/hier_pooled/bins/0.10 <= v < 0.20/excess_by_budget":
        _DOUBLE.format(got="B15's 0.00448", a="0.004", b="0.005", c="0.0045"),
    "decomposition/hier_pooled/bins/v >= 0.20/excess_by_budget":
        _DOUBLE.format(got="B3's 0.03448", a="0.034", b="0.035", c="0.0345"),
    "item_level/run2/share_range":
        "gap.md gives 86-88% and the plan '78 to 88 percent'. With the public ratios at full precision "
        "(0.4379 to 0.5423, results/itemsig_eval.json) run 2's share is 86.4% to 88.6%: the upper end "
        "rounds to 89%, which gap.md truncated. The plan's range becomes 78 to 89 percent.",
    "empirical_mean/v_at_organisers_entry":
        "gap.md's 'v ~ 0.124' was read off the scan; linear interpolation between the scan's v = 0.120 "
        "(ALC 0.1754) and 0.125 (0.1823) gives 0.1234. The scan is a Monte Carlo on one 54-item pair and "
        "is not smooth at the 0.001 level, so it supports 0.123 to 0.124; the reading (an empirical mean "
        "scores 0.1801 only on a run with v near 0.12, far below these runs' 0.154 to 0.180) is unchanged.",
    "curve/shape/replica_mean_steps/tl mix/whole/B3->B7": _E7,
    "curve/shape/replica_mean_steps/r1b/B1->B3": _E7,
    "curve/shape/replica_mean_steps/r1b/B3->B7": _E7,
    "curve/shape/replica_mean_steps/r1p/B0->B1": _E7,
    "curve/shape/replica_mean_steps/r1p/B7->B31": _E7,
    "replica/validation_summary/b0_sd_range/1":
        "gap.md says the B0 estimator has 'sd 0.001 in all 7 regimes'; s6_murphy.py itself printed 0.002 "
        "for MIXTURE (0.0016), and the port matches s6 regime by regime. The sd is 0.0010 to 0.0016.",
}


def _get(res, path):
    x = res
    for part in path:
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


def _same(got, want, how):
    """One value against the scratch's, at the precision the scratch recorded it."""
    if how == "full":
        return abs(got - want) <= 5e-5
    if how == "pct":
        return round(100 * got) == want
    return abs(round(got, how) - want) < 1e-9


def compare_scratch(res):
    """Every SCRATCH entry: 'match', 'explained' (in EXPLAINED, with the reason),
    'differs' (stops the write) or 'not run' (a block that needs data rows or the
    requirement simulation was skipped)."""
    items = []
    for path, (how, want, src) in SCRATCH.items():
        name = "/".join(str(p) for p in path)
        try:
            got = _get(res, path)
        except (KeyError, TypeError, IndexError):
            got = None
        if got is None:
            items.append({"path": name, "status": "not run", "want": want, "source": src})
            continue
        if isinstance(want, list):
            got = [float(g) for g in got]
            ok = len(got) == len(want) and all(_same(g, w, how) for g, w in zip(got, want))
            diff = max(abs(g - w) for g, w in zip(got, want)) if len(got) == len(want) else None
        else:
            got = float(got)
            ok = _same(got, want, how)
            diff = got - want
        st = "match" if ok else ("explained" if name in EXPLAINED else "differs")
        it = {"path": name, "status": st, "got": got, "want": want, "how": how, "diff": diff, "source": src}
        if name in EXPLAINED:
            it["explanation"] = EXPLAINED[name]
        items.append(it)
    return items


# --- provenance, main -----------------------------------------------------------------------------

SCRATCH_SCRIPTS = {  # basename: sha256 of the exploratory script ported (session scratch, 2026-10-06)
    "common.py": "701bfb95f0e10f5d66422a12287f57ff43516ac0dfe9fbecf72aadfb4a7b01de",
    "ideal.py": "6256ce990dc0d5cf78b030b3b9b7b44951b88b7dfdd3b3a2575f3a4db66634d0",
    "formative.py": "b14a55b508a5960c6da0ad6fb01d85b3a7a3d94eb65cb50ef1ee91e099c80edd",
    "s1_runs.py": "e6fc855043174feeb5cfd528ecb2990f88153e019c413c37e8b49cde2082cc13",
    "s4_pop.py": "af0540c2ae22b9d2c00b4836dcd2b6c08aeb7e64d70f321bafa9a9ad77d87934",
    "s6_murphy.py": "3aa5054cf282131b63c89df8e3bf74d0f8f53902853bbd3a00d0ccd5449d597e",
    "s7_runs.py": "547bb6bf963d196d8f394ccc24c0d7e81e6de3b5ea908fd4a4b323f4e569c094",
    "s8_req.py": "755df294405aaafd1ae7d12fb634fd50f8691785e7ff6186cb6814c387c94890",
    "s9_vadj.py": "a2aed245b56c311fd0ee70a1cb3006aeb078bb5b37404c8ce9511635850b3d06",
    "s10_pairideal.py": "6a1c68d0d59d32d28fd22930d882dd92848604f4efe8ace8d63805f5bcb6b16b",
    "curve_test.py": "8d36339dfd398a32aefae7fb7decaf5b7b963c34673edf87cabc45b7f4fb6ea0",
    "curve_shape.py": "9dc3b8d42eb9e18f08ecc4e05bbf79bf22011f4a04cff7d4d09ff0ec34c3eb10",
    "split_floor.py": "411ee3c0df5a9382f569cf413a37a7e17a3f521ddb0bc8ce8ac7976b016e1a60",
    "ece_excess.py": "d6a925f83af713088bbcd05457aff5008e35cacd67376fb1087350a268794ebf",
}


def inputs_digest(with_rows):
    d = {rel(p): sha256(p) for p in (*TABLES.values(), FF_JSON, R3_JSON, SHIP_JSON, ITEMSIG_JSON)}
    if with_rows:
        for reg in ("tl", "r1b", "r1p"):
            p = os.path.join(SS_ROWS, f"{reg}.jsonl")
            if os.path.exists(p):
                d[rel(p)] = sha256(p)
        for reg in RS_REGIMES:
            p = os.path.join(RS_ROWS, reg)
            if os.path.isdir(p):
                d[rel(p) + "/*.json"] = dir_digest(p)
    return d


def provenance(inputs, secs):
    me = os.path.abspath(__file__)
    code = {rel(p): sha256(p) for p in (me, FF.__file__, os.path.join(ROOT, "paiec", "evaluator.py"))}
    status = git("status", "--porcelain", "--", rel(me))
    return {"command": " ".join(["python", rel(sys.argv[0])] + sys.argv[1:]),
            "script": rel(me), "script_sha256": code[rel(me)], "code_sha256": code,
            "head": git("rev-parse", "HEAD"), "script_status": status or "clean",
            "when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "inputs_sha256": inputs, "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "seeds": SEEDS, "wall_s": round(secs, 1)}


def show(res):
    d = res["decomposition"]
    print("run  model                         v       ALC     ALC-v   B0      B1..B15  B31")
    for r in (1, 2, 3):
        x = d[f"run{r}"]
        print(f"{r:<4} {x['model'][:28]:<29} {x['v']:.4f}  {x['alc']:.4f}  {x['alc_minus_v']:+.4f} "
              f"{x['b0_term']:+.4f} {x['b1_b15_terms']:+.4f}  {x['b31_term']:+.4f}")
    i1 = res["identity"]["run1"]
    print(f"identity, run 1: largest per-pair range {i1['per_pair_range_max']:.2e} "
          f"({i1['pairs_within_rounding_bound']}/{i1['pairs']} within the 6-dp rounding bound); "
          f"run-mean range {i1['range_of_run_mean_over_budgets']:.1e}")
    s = res["ideal"]["summary_hier"]
    print(f"ideal learner on hier's 17 pairs: lower roots {s['lower_roots_team_prior']:+.4f}, prior-weighted "
          f"{s['prior_weighted_range'][0]:+.4f} to {s['prior_weighted_range'][1]:+.4f}; hier {s['observed_hier']:+.4f}")
    il = res["item_level"]
    print("item-level share: " + ", ".join(f"run {r} {il[f'run{r}']['share_range'][0]:.3f}-"
                                          f"{il[f'run{r}']['share_range'][1]:.3f}" for r in (1, 2, 3)))
    rs = res["run_spread"]
    print(f"v of a 9-pair run: strat {rs['v_strat']['mean']:.4f} ± {rs['v_strat']['sd']:.4f}, iid "
          f"{rs['v_iid']['mean']:.4f} ± {rs['v_iid']['sd']:.4f}; P(v<=0.117) {rs['v_strat']['P_le_0.117']:.4f} / "
          f"{rs['v_iid']['P_le_0.117']:.4f}")
    va = res["vadjusted"]
    print(f"LegacyP - hier, v-adjusted {va['legacy_minus_hier']['v_adjusted']:+.4f} (SE "
          f"{va['legacy_minus_hier']['se']:.4f}); held-out test-like shipped - legacy "
          f"{va['test_like']['blocks']['200-299']['shipped_minus_legacy']:+.4f} "
          f"({va['test_like']['blocks']['200-299']['cluster_se']:.4f})")
    e = res["entry"]
    print("organisers' entry: " + "; ".join(
        f"run {r} plug-in {e[f'run{r}']['plug_in_b31']:.4f} MC {e[f'run{r}']['mc_mean']:.4f} "
        f"P(<=0.1801) {e[f'run{r}']['P_mc_le_entry']:.4f}" for r in (1, 2, 3)))
    ct = res["curve"]["test"]
    sh = res["curve"]["shape"]["platform_steps_mean_se"]["B7->B31"]
    print(f"curve after B7: hidden B7->B31 {sh[0]:+.4f} ± {sh[1]:.4f}" + (
        "; P(as flat) " + ", ".join(f"{k} {v['P(mean >= platform)']:.4f}" for k, v in ct["replica"].items())
        if ct else "; rows absent"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-requirement", action="store_true", help="skip the ~2-minute synthetic block")
    ap.add_argument("--no-rows", action="store_true", help="do not read data/*_rows")
    args = ap.parse_args(argv)
    t0 = time.time()
    checks = Checks()
    runs = load_tables()
    check_tables(runs, checks)
    with_rows = not args.no_rows
    res = compute(runs, with_rows=with_rows, with_requirement=not args.no_requirement)
    res["main"] = main_numbers(res)
    items = compare_scratch(res)
    n = {s: sum(1 for i in items if i["status"] == s) for s in ("match", "explained", "differs", "not run")}
    res["scratch"] = {"ported_from": SCRATCH_SCRIPTS, "counts": n, "explained": EXPLAINED, "items": items}
    checks.add("every scratch value reproduces to its recorded precision or is explained",
               n["differs"] == 0, n, None)
    checks.add("run 1 (LegacyP): B - ECE^2 per pair constant across budgets within the 6-dp rounding",
               res["identity"]["run1"]["pairs_within_rounding_bound"] == res["identity"]["run1"]["pairs"],
               res["identity"]["run1"]["pairs_within_rounding_bound"], res["identity"]["run1"]["pairs"])
    res["checks"] = checks.items
    res = {"about": {
        "what": "the hidden formative runs read through Brier = ECE^2 + p(1-p): the pair-rate oracle v, "
                "ALC - v, the ideal pair-rate learner, the item-level share, the run-level spread, the "
                "organisers' entry, v-adjusted comparisons, the curve after B7",
        "plan": "docs/plans/p2_final_plan.md sections 2 and 5 (R1-R3)",
        "v": "per pair B0 - ECE0^2 (exact for one prediction per pair at budget 0)",
        "weights": list(WEIGHTS), "budgets": list(BUDGETS),
        "id_use": "anonymous benchmark ids group pairs in the stratified bootstrap only; none is written here",
        "rows": "curve.test reads data/subject_side_rows/{tl,r1b,r1p}.jsonl; replica reads "
                "data/regime_sensitivity_rows/<regime>/*.json (gitignored); null when not read"}, **res}
    res["provenance"] = provenance(inputs_digest(with_rows), time.time() - t0)
    show(res)
    print(f"scratch comparison: {n}")
    bad = checks.failed()
    for c in checks.items:
        if not c["ok"]:
            print(f"FAILED{' (required)' if c['required'] else ''}: {c['check']}: got {c['got']} want {c['want']}")
    for it in items:
        if it["status"] in ("differs", "explained"):
            print(f"  {it['status']}: {it['path']} got {it.get('got')} want {it['want']}")
    if bad:
        print("not written: a required check failed")
        return 1
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    os.replace(tmp, args.out)
    print(f"wrote {rel(args.out)} ({res['provenance']['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
