"""The acceptance harness for per-item covariates: one gate, on the shipped hier.

Every later item-side idea (ordinal difficulty fields, answer format, length,
LLM ratings, attempt signals, a fine-tuned encoder) is one number per item.
This script scores any such number the same way, against the model that ships,
and it holds the one table of thresholds they are all read against
(docs/findings.md, "Acceptance harness").

The model. A covariate is a dict item_key -> float; an item it does not hold
gets 0. It enters the shipped prediction as a centred logit offset

    q_i = sigmoid(logit p_i + cap(beta * (x_i - xbar))),    cap(o) = 4 tanh(o / 4)

p_i is paiec.hier.HierPredictor exactly as it ships: submission/model.py's
LEVEL (read from the file) over prior.build, fitted leaving the target's parent
benchmark out (experiments/level_calibration.py's bundle and factory, the base
of "Calibrating for the hidden test" and of "Item signal from the pair's own
labels"). xbar is the covariate's mean over labeled items, so hier keeps the
benchmark's level and only the ordering of items comes from x. The offset is
added to hier's output, not fitted inside its model: hier's posterior is not
refitted with the covariate in it, which is what makes one replay of hier serve
every covariate. At B0 nothing is labeled and a centred offset is 0, so the
centred terms are blind at B0 by construction; the uncentred B0 term ('b0'
below) is the one path that acts there. Where the offset is 0, q is p itself,
bit for bit, so a covariate of zeros reproduces the shipped predictions
(tests/test_harness.py).

Input convention (--stage eval). x may be on any scale. By default it is
standardised within each public benchmark over the items that carry it
(standardised(): experiments/itemcov_eval.py's rule; 0, the benchmark's mean,
where an item lacks x or where fewer than MIN_VARY of the benchmark's items are
off x's most common value; clipped at +-CLIP sd), so benchmark-level offsets in
x cannot act and s below means the same on every benchmark. --raw reads x as
given. The uncentred B0 term reads raw x in both cases (see 'b0').

Slopes (VARIANTS):
  transferred  one coefficient per budget, beta_B, fitted on the other parents'
               appearances (test-like and mix/whole, FIT_ON): the minimum of the
               mean pair Brier, a 1-D search per budget. The centre is every
               label on the target's benchmark, its own pair's and other
               subjects'.
  per-pair     beta_p from the target pair's own labels at the checkpoint: the
               MAP of y_j ~ Bernoulli(sigmoid(u_j + beta c_j)) with beta ~ N(0,
               (s / sd_x)^2) over its own labeled items, c_j = x_j - their mean,
               and the target centred on the same mean. s is a prior sd per
               within-benchmark sd of x (S_GRID); sd_x is Engine.sd, the pooled
               within-parent sd over the training parents' evaluated items on
               which x varies (SCALE = 'within'; the legacy 'pooled' sd, taken
               across parents, made s depend on benchmark offsets and on the
               parents where x is absent). The plan caps s at 0.5 for a feature
               whose sign is not known; the grid goes wider so that the
               oracle's line shows what a per-pair slope can reach, and the
               nested choice is reported. u_j is hier's prediction for item j as
               if j itself were unlabeled, at the same checkpoint and with the
               same labels (below), so a label is never scored against a
               prediction it already moved. One own label carries no slope
               after centring, so the term is 0 at B1.
  hybrid       the per-pair MAP with its prior centred on the transferred beta_B
               instead of 0 (the plan's "transferred slope plus a zero-mean
               per-pair component").
  b0           an uncentred transferred offset at B0 only, beta_0 (x0_i - c0):
               x0 is x on an absolute scale (Covariate.X0; --stage eval: the raw
               values, since at B0 a run-time predictor sees one item and no
               benchmark to standardise over), c0 its mean over the training
               parents' evaluated items, beta_0 fitted like beta_B on their B0
               predictions. An item without x0 gets no offset. It adds to any
               centred line (the budgets are disjoint). On the degraded oracles
               x0 is x standardised within parent, so every parent has mean 0
               and the term reads only differences within a parent (a
               pseudo-benchmark's level within its parent included); a real
               absolute covariate would also carry levels between benchmarks.
  Each centred variant is on from B1, or only from B7 (STARTS).

u_j. hier reads an item through its digest (its own residual e_i), its
item_features groups and the multiple-choice floor of its text. The collector
calls hier on each labeled item a second time with an invisible separator
(U+2063) appended to item_content: a new digest, the same groups (features are
untouched) and the same floor (checked). That is hier's predictive for an
unlabeled item of the same groups, with the pair's level fitted on every label,
the item's own included (the part the joint model would also leave in).

Selection: nested leave-one-parent-out over the four multi-subject parents.
For each held-out parent q, every configuration of a variant is scored on each
inner parent q2 != q with whatever it fits (the slopes) fitted on the two
remaining parents. The criterion is the mean over the three inner parents of
their test-like ALC difference, with an SE from the parents' linearised
pair-cluster SEs (lin_se). The configuration with the lowest criterion among
those below -SELECT_K inner SEs is chosen, or 'off' when there is none (the
choice without the margin is recorded too). It is then fitted on the three
and scored on q, in every regime. A fold counts as on (folds_on) only if the
choice moves some prediction of the held-out parent: a covariate constant on
q is chosen on its inner folds and does nothing there (folds_chosen counts
those too). Public benchmarks outside the four (swe_rebench) get the choice
made the same way over all four parents. Forced lines score a fixed
configuration with no selection: the r -> ALC curve without selection noise.

Regimes (PLAN): test-like runs (testlike.Regime() at its defaults, seed 2;
level_calibration's 'tl'), mix/whole (groups merged at random and wholes, no
strata, seed 3; 'tl mix/whole'), and public R1 (official.sample_run, seed 0,
split scope 'pair', benchmark-first 'r1b' and pair-uniform 'r1p') as the guard.
These are the runs of level_calibration.py and itemsig_eval.py; the test-like
and mix/whole ones are also the scratch rows of the heads study (--stage verify).

Statistics. Paired ALC differences against the shipped hier, weighted 1 / run
size per pair appearance (a run's ALC is the mean over its pairs), with: the SE
over runs; a pair-cluster bootstrap (BOOTS resamples, a ratio estimator;
cluster (parent, subject) on test-like runs, testlike.cluster_key, and
(benchmark, subject) on public ones); the same bootstrap stratified by parent;
and the SE across the four parents' means (the only one that sees variation
between benchmarks, on four of them). These hold the selection fixed, so for a
nested line they are lower bounds. With per (eval, the oracles) a nested
line's test-like summary also carries sel_cluster_se: the same cluster
bootstrap with the inner selection redone in every resample
(selection_boot_se; the fitted slopes held fixed). Per budget, B0 is 0 by
construction for the centred terms.

The item oracle and the threshold table (--stage table). The acceptance oracle
is the parent's in-sample Rasch difficulty (testlike.Catalogue.difficulty: every
subject of the parent, the evaluated responses included) with the transferred
slope cross-fitted leave-one-parent-out, forced on from B1: what the heads study
scored (-0.0437 test-like, -0.0557 mix/whole). The harness must reproduce those
numbers (ACCEPT) before anything else is read. The honest oracle fits the
difficulty on the other subject folds only (HONEST_FOLDS, folds by subject_id),
so a target's own responses never enter its covariate; the gap between the two
is the in-sample oracle's leak. A second honest oracle folds by canonical model
name (name_fold_maps), so one model under several harnesses or efforts cannot
leak across folds. The degraded oracles are x = r z + sqrt(1 - r^2) e, z a
difficulty standardised within its parent, e standard normal per item (N_REPS
draws, per-appearance differences averaged over them), for r in R_GRID, and
r = 0 as the uninformative covariate. The gate table ('honest') degrades the
honest difficulty, fold by fold, so r means what a later covariate's r means
when it is measured against honest difficulty, as the plan prescribes; the
'in_sample' table degrades the in-sample one and is the heads study's curve. r
is the correlation over a parent's items; within a pseudo-benchmark, whose
difficulty range is narrower, the realised correlation is lower, and the table
reports it (r_within_pair_tl).

Gate (GATE, the plan's acceptance rule): nested selection switches the term on
(on and acting in at least 3 of the 4 outer folds), test-like ALC difference
<= -0.002 with the mix/whole one of the same sign and no held-out parent above
+0.002, and neither public weighting worse than +0.001. A forced line has no
selection: it reports whether it would pass if switched on. The plan's step-4
rule that an uninformative version costs at most +0.0005 (UNINFORMATIVE_CAP)
is read on the table's r = 0 row. A covariate that exists on one parent only
cannot be switched on leave-one-parent-out (its slope has nothing to be fitted
or selected on); its forced per-pair lines show what it does there. A real
covariate is one noise draw, not the average over draws, so the table reports,
per cell, how many draws pass on their own (pass_reps; thresholds():
pass_draws, and the smallest r at which most or every draw passes).

Provenance. Every stored run carries the digest of the library it was
collected with (LIB); table and eval record the rows' digests beside the
current one, and eval warns when they differ. --stage verify re-collects
RECHECK's runs with the library on disk and compares them with the stored rows
(they must agree exactly, BLAS pinned to one thread as below), and with
--legacy DIR compares every run with an earlier library's rows.

Stages
  collect  replay hier on every run of PLAN and store rows (one pickle per run,
           resumable) in data/harness_rows/<regime>/: per pair appearance its
           evaluated items (key, successes, responses) with hier's prediction at
           every checkpoint, and its acquired items (key, label) with hier's
           prediction as labeled and as unlabeled
  verify   the re-collection check, --legacy, the heads study's scratch rows
           (--scratch DIR or HARNESS_SCRATCH_ROWS; per response, float32 as
           stored there), and the unlabeled-item probe against hier on items
           nobody labeled
  table    the oracle acceptance and the threshold table ->
           results/harness_thresholds.json (--resume keeps what a run with the
           same library, script, rows and grid already wrote)
  eval     one covariate (--cov: JSON {item_key: x} or a CSV/parquet with
           columns item_key, x; --raw) -> results/harness_<name>.json, with the
           gate and the table's rows beside it
  show     the result of table (or of eval, --cov RESULT.json) as markdown

Run:  python experiments/harness.py --stage collect --jobs 1     # 39 min: 650 runs, ~3.6 s each
      python experiments/harness.py --stage verify --legacy data/harness_rows_legacy   # 1 min
      python experiments/harness.py --stage table --resume        # 44 min: 94 covariates, ~27 s each
      python experiments/harness.py --stage show                  # markdown tables
      python experiments/harness.py --stage eval --cov my_feature.json --name my_feature   # ~60 s
(wall times on a shared M1, beside a running language-model job; collect keeps
one process of about 0.2 GB per job, and resumes where it stopped.) The rows
are 43 MB under data/ (gitignored); results/harness_thresholds.json holds the
acceptance, both tables, the thresholds, the verification and the provenance.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUDGETS = (0, 1, 3, 7, 15, 31)
W6 = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
#: harness regime -> experiments/level_calibration.py's REGIMES key
REGIMES = {"tl": "tl", "mix": "tl mix/whole", "r1b": "r1b", "r1p": "r1p"}
#: runs 0..n-1 of each regime
PLAN = {"tl": 300, "mix": 150, "r1b": 100, "r1p": 100}
FIT_ON = ("tl", "mix")          # appearances the transferred slope is fitted on
SELECT_ON = "tl"                # the regime the nested selection reads
GUARDS = ("r1b", "r1p")
CAP = 4.0                       # |offset| < CAP logits (the heads study's 4 tanh(o / 4))
S_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)   # per-pair prior sd, per within-benchmark sd of x
STARTS = (1, 7)                 # the term acts from this budget on
VARIANTS = ("transferred", "per-pair", "hybrid", "b0")
#: Engine.sd: 'within' (pooled within-benchmark sd) or 'pooled' (the pre-audit unit; HARNESS_SCALE=pooled
#: makes it the default again, e.g. for experiments/itemcov_eval.py --scale train on the legacy rows)
SCALE = os.environ.get("HARNESS_SCALE", "within")
SELECT_K = 1.0                  # selection: inner mean below -SELECT_K inner SEs
CLIP = 3.0                      # eval: x standardised within benchmark, clipped at +-CLIP sd
MIN_VARY = 10                   # eval: a benchmark with fewer items off x's mode gets no x
R_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
N_REPS = 8                      # noise draws per degraded-oracle cell (--reps)
HONEST_FOLDS = 5
HONEST_SALT = "harness-honest"
BOOTS = 2000
MARK = "⁣"                 # appended to item_content for the unlabeled-item probe
ACCEPT = {"tl": -0.0437, "mix": -0.0557, "tol": 0.003}
GATE = {"tl": -0.002, "parent": 0.002, "guard": 0.001, "folds_on": 3}
UNINFORMATIVE_CAP = 0.0005      # the plan's step 4: an r = 0 covariate costs at most this
#: forced (unselected) configurations reported beside the nested lines
FORCED = (("transferred", None, 1), ("transferred", None, 7), ("per-pair", 0.5, 1),
          ("per-pair", 0.5, 7), ("hybrid", 0.5, 1), ("b0", None, 0))
#: honest-table rows whose draw 0 --stage table also scores with the selection-aware SE
SEL_CHECK_R = (0.0, 0.2, 0.3, 0.4)
#: runs re-collected by --stage verify and compared with the stored rows
RECHECK = (("tl", 0), ("tl", 17), ("mix", 3), ("r1b", 0), ("r1p", 5))

ROWS = os.path.join(ROOT, "data", "harness_rows")
RECOMPUTED = ("hier is replayed on every run (collect): the heads study's scratch rows lack item keys, "
              "the unlabeled-item predictions the per-pair slope needs and public runs; its "
              "test-like and mix/whole rows are used only to check the replay (verify)")
OUT = os.path.join(ROOT, "results", "harness_thresholds.json")
#: the heads study's rows (rethink2/fine-tuning/rows of the session that wrote
#: this script), for --stage verify only; nothing else reads them
SCRATCH = os.environ.get("HARNESS_SCRATCH_ROWS", "")
SCRATCH_DIRS = {"tl": "tl", "mix": "tl_mix-whole"}


# --- core arithmetic (pure numpy; tests/test_harness.py) ------------------------------

def sigmoid(z):
    z = np.asarray(z, float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def logit(p):
    p = np.asarray(p, float)
    return np.log(p) - np.log1p(-p)


def capped(o):
    return CAP * np.tanh(np.asarray(o, float) / CAP)


def shifted(p, off):
    """hier's p moved by a logit offset; exactly p where the offset is 0."""
    p, off = np.asarray(p, float), np.asarray(off, float)
    return np.where(off == 0.0, p, sigmoid(logit(p) + off))


def delta_brier(p, q, K, N, seg, n_seg):
    """Per segment (pair appearance): Brier of q minus Brier of p over every
    response, items given as K successes in N responses."""
    num = N * (q * q - p * p) - 2 * K * (q - p)
    return np.bincount(seg, num, n_seg) / np.maximum(np.bincount(seg, N, n_seg), 1e-300)


def pair_slope(u, y, c, mask, m, s, iters=50):
    """MAP of beta in y ~ Bernoulli(sigmoid(u + beta c)), beta ~ N(m, s^2), for
    each row (episode) of the padded (E, L) arrays; mask marks real labels. The
    log posterior is concave, so Newton converges; steps are capped at 3 prior sd
    as a guard."""
    u, y, c = (np.asarray(a, float) for a in (u, y, c))
    mask = np.asarray(mask, bool)
    m = np.broadcast_to(np.asarray(m, float), u.shape[:1]).copy()
    s = np.broadcast_to(np.asarray(s, float), u.shape[:1])
    c = np.where(mask, c, 0.0)
    u = np.where(mask, u, 0.0)
    b = m.copy()
    for _ in range(iters):
        pr = sigmoid(u + b[:, None] * c)
        g = np.sum(np.where(mask, c * (y - pr), 0.0), 1) - (b - m) / s ** 2
        h = np.sum(np.where(mask, c * c * pr * (1 - pr), 0.0), 1) + 1 / s ** 2
        step = np.clip(g / h, -3 * s, 3 * s)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def golden(f, lo, hi, iters=60):
    """Minimum of f on [lo, hi] by golden-section search."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_scalar(f, scale, span=6.0, n_grid=41):
    """argmin of a 1-D objective over [-span, span] / scale: a grid, then golden
    section inside the best grid cell's neighbours. 0 is kept when nothing beats
    it (a flat objective leaves the covariate off)."""
    grid = np.linspace(-span, span, n_grid) / scale
    vals = np.array([f(b) for b in grid])
    j = int(np.argmin(vals))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, n_grid - 1)]
    b = golden(f, lo, hi)
    return b if f(b) < f(0.0) else 0.0


def weighted_mean(d, w):
    return float(np.sum(w * d) / np.sum(w))


def boot_se(d, w, cluster, strata=None, boots=BOOTS, seed=0):
    """Cluster bootstrap SE of the ratio estimator sum(w d) / sum(w): clusters
    resampled with replacement, as one pool or within each stratum."""
    rng = np.random.default_rng(seed)
    cl, inv = np.unique(cluster, return_inverse=True)
    S = np.bincount(inv, w * d, len(cl))
    Wc = np.bincount(inv, w, len(cl))
    if strata is None:
        groups = [np.arange(len(cl))]
    else:
        st = np.empty(len(cl), object)
        st[inv] = np.asarray(strata, object)
        groups = [np.flatnonzero(st == v) for v in sorted(set(st.tolist()))]
    num = np.zeros(boots)
    den = np.zeros(boots)
    for g in groups:
        pick = g[rng.integers(0, len(g), (boots, len(g)))]
        num += S[pick].sum(1)
        den += Wc[pick].sum(1)
    return float(np.std(num / den, ddof=1)) if boots > 1 else float("nan")


def lin_se(d, w, cluster):
    """Linearised (sandwich) SE of the ratio estimator sum(w d) / sum(w) with
    independent clusters: sqrt(G / (G - 1) sum_c (S_c - m W_c)^2) / sum(w). The
    analytic counterpart of boot_se, cheap enough for every inner criterion; 0
    with fewer than two clusters."""
    d, w = np.asarray(d, float), np.asarray(w, float)
    cl, inv = np.unique(cluster, return_inverse=True)
    G = len(cl)
    if G < 2 or not np.sum(w) > 0:
        return 0.0
    S = np.bincount(inv, w * d, G)
    W = np.bincount(inv, w, G)
    r = S - (S.sum() / W.sum()) * W
    return float(math.sqrt(G / (G - 1) * np.sum(r * r)) / W.sum())


def run_se(d, w, run):
    runs, inv = np.unique(run, return_inverse=True)
    m = np.bincount(inv, w * d, len(runs)) / np.bincount(inv, w, len(runs))
    return float(np.std(m, ddof=1) / math.sqrt(len(m))) if len(m) > 1 else float("nan")


def stats(d, w, run, cluster, parent, boots=BOOTS, seed=0):
    """Mean and the three SEs of one set of appearances."""
    if not len(d):
        return None
    return {"est": weighted_mean(d, w), "run_se": run_se(d, w, run),
            "cluster_se": boot_se(d, w, cluster, None, boots, seed),
            "strat_se": boot_se(d, w, cluster, parent, boots, seed),
            "n_app": int(len(d)), "n_runs": int(len(np.unique(run))),
            "n_clusters": int(len(np.unique(cluster)))}


def select(crit, se=None, k=0.0):
    """The configuration with the lowest inner criterion among those below
    -k times their inner SE (below 0 when se is None or k = 0), or None ('off').
    Ties break on the configuration order."""
    best = None
    for c, v in crit.items():
        bar = -k * se[c] if se is not None and k else 0.0
        if v < min(bar, 0.0) and (best is None or v < crit[best]):
            best = c
    return best


def degrade(z, r, eps):
    """A covariate at correlation r with z (both standardised): r z + sqrt(1-r^2) e."""
    return r * np.asarray(z, float) + math.sqrt(max(0.0, 1 - r * r)) * np.asarray(eps, float)


def standardise_within(values, groups):
    """values standardised to mean 0, sd 1 within each group."""
    v = np.asarray(values, float).copy()
    g = np.asarray(groups, object)
    for k in set(g.tolist()):
        m = g == k
        sd = v[m].std()
        v[m] = (v[m] - v[m].mean()) / (sd if sd > 0 else 1.0)
    return v


def standardised(xmap, items_bench, clip=CLIP, min_vary=MIN_VARY):
    """The input convention of --stage eval (experiments/itemcov_eval.py's rule):
    x standardised within each benchmark over the benchmark's items that carry
    it, clipped at +-clip sd. An item without x, or on a benchmark where fewer
    than min_vary of the items that carry x are off its most common value,
    gets no entry (0 in the harness: its benchmark's mean). A run-time
    predictor would standardise over the benchmark's visible items instead."""
    out = {}
    for bench, keys in items_bench.items():
        ks = [k for k in keys if k in xmap]
        if not ks:
            continue
        x = np.array([xmap[k] for k in ks], float)
        off = len(x) - np.unique(x, return_counts=True)[1].max()
        if off < min_vary or not x.std() > 0:
            continue
        z = np.clip((x - x.mean()) / x.std(), -clip, clip)
        out.update({k: float(v) for k, v in zip(ks, z)})
    return out


# --- the shipped configuration ------------------------------------------------------

def shipped_level():
    """submission/model.py's LEVEL, read without importing the module (it
    imports the archive's paiec_rt, which exists only inside the zip)."""
    with open(os.path.join(ROOT, "submission", "model.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LEVEL"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("no LEVEL in submission/model.py")


def _lc():
    import experiments.level_calibration as LC
    return LC


def ship_name():
    LC = _lc()
    lv = shipped_level()
    name = LC.gname(lv["mu0"], lv["sigma_mu"], lv["attr_scale"])
    if LC.parse(name)[1] != {k: float(v) for k, v in lv.items()}:
        raise RuntimeError(f"{name} is not LEVEL {lv}")
    return name


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


LIB = ("paiec/hier.py", "paiec/prior.py", "paiec/subjects.py", "paiec/official.py",
       "paiec/testlike.py", "paiec/mcq.py", "paiec/predict.py", "paiec/fitting.py",
       "experiments/level_calibration.py", "submission/model.py")


def provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    return {"commit": commit, "lib_digest": digest(LIB), "script_digest": digest(["experiments/harness.py"]),
            "level": shipped_level()}


# --- collect ------------------------------------------------------------------------

def unlabeled_twin(inp):
    """The same subject and item with a new digest and nothing else changed."""
    subject, item = inp
    return [subject, dict(item, item_content=str(item.get("item_content", "")) + MARK)]


def row_path(rows, regime, i):
    return os.path.join(rows, regime, f"{i}.pkl")


def collect_run(regime, i, ship):
    """Replay hier on one run: per pair appearance its evaluated items with
    hier's prediction at every checkpoint, and its acquired items with hier's
    prediction as labeled (acq_p) and as if unlabeled (acq_pu)."""
    from paiec import official as O
    from paiec import testlike as T
    from paiec.data import official_subject
    from paiec.mcq import floor_of
    LC = _lc()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = LC.draw(REGIMES[regime], i)
    slots, cps = LC.checkpoints(run)
    made = []
    fac = LC.factory([ship], run, made)
    out, first, at = [], [], 0
    for entry, s in zip(run, slots):
        pair, items = O._entry(entry)
        by = pair.by_item()
        _, ev = O.split(pair, 0, "pair")
        ev = [k for k in ev if items is None or k in items]
        labels = [int(r.label) for k in ev for r in by[k]]
        if labels != [t[1] for t in s.targets]:
            raise RuntimeError(f"{regime} run {i}: evaluation order differs from official._slots")
        N = np.array([len(by[k]) for k in ev], np.int32)
        K = np.array([sum(int(r.label) for r in by[k]) for k in ev], np.int32)
        pos = at + np.concatenate([[0], np.cumsum(N)[:-1]]).astype(np.int64)
        first.append(pos)
        at += len(labels)
        acq = s.acquired[:31]
        for e in acq:
            if floor_of(e[0][1]["item_content"]) != floor_of(unlabeled_twin(e[0])[1]["item_content"]):
                raise RuntimeError("the unlabeled-item probe changed an item's floor")
        out.append(dict(
            name=s.name, parent=T.parent_of(s.name), kind=T.kind_of(s.name), bench=s.benchmark_id,
            sid=str(pair.subject_id), anon_sid=s.subject_id, subject=official_subject(pair.subject),
            ev_keys=np.array(ev, dtype=object), ev_K=K, ev_N=N,
            ev_p=np.full((6, len(ev)), np.nan),
            acq_keys=np.array(s.keys[:31], dtype=object),
            acq_y=np.array([e[1] for e in acq], np.int8),
            acq_p=np.full((6, len(acq)), np.nan), acq_pu=np.full((6, len(acq)), np.nan)))
    calls = 0
    for bi, (b, labeled, inputs, index) in enumerate(cps):
        fn = fac()
        P = np.array([fn(inp, labeled) for inp in inputs], float)
        pe = P[index]
        calls += len(inputs)
        for d, s, pos in zip(out, slots, first):
            d["ev_p"][bi] = pe[pos]
            for j, e in enumerate(s.acquired[:min(b, 31)]):
                d["acq_p"][bi, j] = fn(e[0], labeled)
        for d, s in zip(out, slots):
            for j, e in enumerate(s.acquired[:min(b, 31)]):
                d["acq_pu"][bi, j] = fn(unlabeled_twin(e[0]), labeled)
                calls += 2
    return dict(regime=regime, run=i, slots=out, ship=ship, level=shipped_level(), lib_digest=digest(LIB),
                secs=time.perf_counter() - t0, calls=calls,
                hier_failures=int(sum(getattr(m, "failures", 0) for m in made)),
                hier_unconverged=int(sum(getattr(m, "unconverged", 0) for m in made)))


def _collect_task(task):
    regime, i, ship, rows = task
    path = row_path(rows, regime, i)
    r = collect_run(regime, i, ship)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(r, f, protocol=4)
    os.replace(path + ".tmp", path)
    return regime, i, r["secs"], r["hier_failures"]


def stage_collect(args):
    LC = _lc()
    ship = ship_name()
    todo = []
    for regime in args.regimes:
        os.makedirs(os.path.join(args.rows, regime), exist_ok=True)
        todo += [(regime, i, ship, args.rows) for i in range(PLAN[regime] if args.n is None else args.n)
                 if not os.path.exists(row_path(args.rows, regime, i))]
    print(f"collect: {len(todo)} runs to replay with {ship}", flush=True)
    if not todo:
        return
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = LC.pairs()
        LC.catalogue()
        for regime in args.regimes:
            if LC.REGIMES[REGIMES[regime]][0] == "testlike":
                LC.sampler(REGIMES[regime])
        for par in sorted({p.benchmark_id for p in pairs}):
            LC.bundle(par, 0.0)        # built once, shared by the forked workers
    print(f"collect: data, catalogue and leave-one-parent-out priors in {time.time() - t0:.0f}s",
          flush=True)
    done = 0
    if args.jobs <= 1:
        it = map(_collect_task, todo)
        for regime, i, secs, fails in it:
            done += 1
            print(f"{regime} run {i}: {secs:.1f}s, {fails} hier failures ({done}/{len(todo)})", flush=True)
    else:
        with mp.get_context("fork").Pool(args.jobs, maxtasksperchild=50) as pool:
            for regime, i, secs, fails in pool.imap_unordered(_collect_task, todo, chunksize=1):
                done += 1
                print(f"{regime} run {i}: {secs:.1f}s, {fails} hier failures ({done}/{len(todo)})",
                      flush=True)
    print(f"collect: {len(todo)} runs in {time.time() - t0:.0f}s", flush=True)


# --- verify -------------------------------------------------------------------------

def stage_verify(args):
    """The stored rows (--rows): re-collected on RECHECK's runs with the library
    on disk (must agree exactly: the rows are the current library's), compared
    with an earlier library's rows (--legacy: what changed, run by run), with
    the heads study's scratch rows (--scratch; identical when the library is
    the one the heads study ran), and the unlabeled-item probe against hier on
    items nobody labeled."""
    res = {"lib_digest": digest(LIB), "rows": os.path.relpath(args.rows, ROOT)}
    res["recollect"] = recollect_check(args.rows)
    print("recollect", res["recollect"], flush=True)
    if args.legacy:
        res["vs_legacy"] = compare_rows(args.rows, args.legacy)
        print("vs legacy", res["vs_legacy"], flush=True)
    res["scratch"] = args.scratch
    for regime, sub in SCRATCH_DIRS.items():
        d = os.path.join(args.scratch, sub) if args.scratch else ""
        if not d or not os.path.isdir(d):
            res[regime] = "no scratch rows"
            continue
        n, worst_e, worst_a, resp, labs, differ = 0, 0.0, 0.0, 0, 0, 0
        for i in range(PLAN[regime]):
            mine, theirs = row_path(args.rows, regime, i), os.path.join(d, f"{i}.pkl")
            if not (os.path.exists(mine) and os.path.exists(theirs)):
                continue
            with open(mine, "rb") as f:
                a = pickle.load(f)
            with open(theirs, "rb") as f:
                b = pickle.load(f)
            if len(a["slots"]) != len(b["slots"]):
                raise RuntimeError(f"{regime} run {i}: pair counts differ")
            run_worst = 0.0
            for x, y in zip(a["slots"], b["slots"]):
                if x["bench"] != y["benchmark_id"] or x["anon_sid"] != y["subject_id"]:
                    raise RuntimeError(f"{regime} run {i}: pairs differ")
                pe = np.repeat(x["ev_p"], x["ev_N"], axis=1).astype(np.float32)
                yy = np.repeat(np.zeros(len(x["ev_N"])), x["ev_N"])
                if pe.shape != y["eval_p"].shape or len(yy) != len(y["eval_y"]):
                    raise RuntimeError(f"{regime} run {i}: evaluation shapes differ")
                de = float(np.max(np.abs(pe - y["eval_p"])))
                ap = x["acq_p"].astype(np.float32)
                both = np.isfinite(ap) & np.isfinite(y["acq_p"])
                if (np.isfinite(ap) != np.isfinite(y["acq_p"])).any():
                    raise RuntimeError(f"{regime} run {i}: labeled-item cells differ")
                da = float(np.max(np.abs(ap[both] - y["acq_p"][both]))) if both.any() else 0.0
                if not np.array_equal(x["acq_y"], y["acq_y"]):
                    raise RuntimeError(f"{regime} run {i}: acquired labels differ")
                worst_e, worst_a, run_worst = max(worst_e, de), max(worst_a, da), max(run_worst, de, da)
                resp += pe.size
                labs += int(both.sum())
            n += 1
            differ += run_worst > 0
        res[regime] = {"runs": n, "runs_differing": differ, "responses_x_budgets": resp, "labeled_cells": labs,
                       "max_abs_eval": worst_e, "max_abs_labeled": worst_a}
        print(regime, res[regime], flush=True)
    res["probe"] = probe_check(args)
    print("probe", res["probe"], flush=True)
    state = load_json(args.out) or {}
    state["verify"] = res
    save_json(args.out, state)


def _row_diff(a, b):
    """Max |difference| per stored prediction field of two collections of one
    run (inf where their NaN patterns differ), and the number of cells that
    differ; the runs' pairs, items and labels must be the same."""
    worst, cells = {}, 0
    if len(a["slots"]) != len(b["slots"]):
        raise RuntimeError("pair counts differ")
    for x, y in zip(a["slots"], b["slots"]):
        for k in ("ev_K", "ev_N", "acq_y"):
            if not np.array_equal(x[k], y[k]):
                raise RuntimeError(f"{k} differs")
        if list(x["ev_keys"]) != list(y["ev_keys"]) or list(x["acq_keys"]) != list(y["acq_keys"]):
            raise RuntimeError("item keys differ")
        for k in ("ev_p", "acq_p", "acq_pu"):
            u, v = x[k], y[k]
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                worst[k] = float("inf")
                continue
            dd = np.abs(np.where(np.isnan(u), 0.0, u - v))
            worst[k] = max(worst.get(k, 0.0), float(dd.max()) if dd.size else 0.0)
            cells += int(np.sum(dd > 0))
    return worst, cells


def recollect_check(rows_dir, runs=RECHECK):
    """Re-collect RECHECK's runs with the library on disk and compare with the
    stored rows (BLAS pinned to one thread, as collect runs)."""
    LC = _lc()
    ship = ship_name()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LC.pairs()
        LC.catalogue()
    out = {"lib_digest": digest(LIB), "runs": {}}
    for regime, i in runs:
        p = row_path(rows_dir, regime, i)
        if not os.path.exists(p):
            out["runs"][f"{regime} {i}"] = "not stored"
            continue
        with open(p, "rb") as f:
            stored = pickle.load(f)
        worst, cells = _row_diff(collect_run(regime, i, ship), stored)
        out["runs"][f"{regime} {i}"] = {"stored_lib_digest": stored.get("lib_digest"), "max_abs": worst,
                                        "cells_differing": cells}
    out["exact"] = all(isinstance(v, dict) and v["cells_differing"] == 0 and all(math.isfinite(x) for x in
                                                                                  v["max_abs"].values())
                       for v in out["runs"].values())
    return out


def compare_rows(new_dir, old_dir):
    """Per regime: runs in both directories, how many differ, the prediction
    cells that differ and the largest difference (an earlier library's rows
    against these)."""
    out = {}
    for regime in PLAN:
        n, differ, cells, worst, digests = 0, 0, 0, 0.0, set()
        for i in range(PLAN[regime]):
            a, b = row_path(new_dir, regime, i), row_path(old_dir, regime, i)
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            with open(a, "rb") as f:
                x = pickle.load(f)
            with open(b, "rb") as f:
                y = pickle.load(f)
            digests.add(str(y.get("lib_digest")))
            w, c = _row_diff(x, y)
            n += 1
            differ += c > 0
            cells += c
            worst = max(worst, max(w.values()) if w else 0.0)
        out[regime] = {"runs": n, "runs_differing": differ, "cells_differing": cells, "max_abs": worst,
                       "old_lib_digest": sorted(digests)}
    return out


def probe_check(args, n_runs=3):
    """hier on an evaluated item that no label on its benchmark touches, called
    as is and through the unlabeled-item probe: the two must agree exactly."""
    LC = _lc()
    ship = ship_name()
    worst, n = 0.0, 0
    for regime in ("tl", "r1b"):
        for i in range(n_runs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run = LC.draw(REGIMES[regime], i)
            slots, cps = LC.checkpoints(run)
            fac = LC.factory([ship], run, [])
            for b, labeled, inputs, _ in cps[2:4]:
                fn = fac()
                seen = {(e[0][1]["benchmark_id"], e[0][1]["item_content"], e[0][1]["item_features"])
                        for e in labeled}
                for inp in inputs[::7]:
                    it = inp[1]
                    if (it["benchmark_id"], it["item_content"], it["item_features"]) in seen:
                        continue
                    worst = max(worst, abs(fn(inp, labeled) - fn(unlabeled_twin(inp), labeled)))
                    n += 1
    return {"items": n, "max_abs": worst}


# --- rows as flat arrays --------------------------------------------------------------

class Rows:
    """One regime's stored runs as flat arrays. Appearances a = 0..A-1; evaluated
    items flattened (ev_*) with their appearance ev_a; acquired items padded to
    31 per appearance (own_*); siblings = appearances on the same benchmark of
    the same run (grp)."""

    def __init__(self, regime, runs, key_index):
        self.regime = regime
        app, ev = [], []
        for r in runs:
            size = len(r["slots"])
            for s in r["slots"]:
                app.append((r["run"], size, s["parent"], s["kind"], s["bench"], s["sid"]))
                ev.append(s)
        A = len(app)
        self.A = A
        self.run = np.array([a[0] for a in app])
        self.w = 1.0 / np.array([a[1] for a in app], float)
        self.parent = np.array([a[2] for a in app], object)
        self.kind = np.array([a[3] for a in app], object)
        self.bench = np.array([a[4] for a in app], object)
        self.sid = np.array([a[5] for a in app], object)
        public = regime in GUARDS
        self.cluster = np.array([f"{(a[4] if public else a[2])}|{a[5]}" for a in app], object)
        self.strata = self.parent
        g = {}
        self.grp = np.array([g.setdefault((a[0], a[4]), len(g)) for a in app])
        self.n_grp = len(g)
        # evaluated items
        n_ev = [len(s["ev_keys"]) for s in ev]
        self.ev_a = np.repeat(np.arange(A), n_ev)
        self.ev_k = np.array([key_index[k] for s in ev for k in s["ev_keys"]], np.int64)
        self.ev_K = np.concatenate([s["ev_K"] for s in ev]).astype(float)
        self.ev_N = np.concatenate([s["ev_N"] for s in ev]).astype(float)
        self.ev_p = np.concatenate([s["ev_p"] for s in ev], 1)            # (6, n)
        # acquired items (own labels)
        L = 31
        self.own_len = np.array([len(s["acq_y"]) for s in ev])
        self.own_k = np.full((A, L), -1, np.int64)
        self.own_y = np.zeros((A, L))
        self.own_pu = np.full((6, A, L), 0.5)
        for a, s in enumerate(ev):
            n = len(s["acq_y"])
            self.own_k[a, :n] = [key_index[k] for k in s["acq_keys"]]
            self.own_y[a, :n] = s["acq_y"]
            self.own_pu[:, a, :n] = np.where(np.isfinite(s["acq_pu"]), s["acq_pu"], 0.5)
        self.nlab = np.minimum(np.array(BUDGETS)[:, None], self.own_len[None, :])   # (6, A)
        self.hier_failures = int(sum(r.get("hier_failures", 0) for r in runs))
        self.hier_unconverged = int(sum(r.get("hier_unconverged", 0) for r in runs))
        self.lib_digests = sorted({str(r.get("lib_digest")) for r in runs})
        self.subject = [s["subject"] for s in ev]


def load_rows(rows_dir, regimes, plan=PLAN):
    """{regime: Rows} over one shared item-key index, and the index."""
    raw, keys = {}, {}
    for regime in regimes:
        runs = []
        for i in range(plan[regime]):
            p = row_path(rows_dir, regime, i)
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                runs.append(pickle.load(f))
        if not runs:
            raise RuntimeError(f"no rows for {regime} under {rows_dir}; run --stage collect")
        raw[regime] = runs
        for r in runs:
            for s in r["slots"]:
                for k in s["ev_keys"]:
                    keys.setdefault(k, len(keys))
                for k in s["acq_keys"]:
                    keys.setdefault(k, len(keys))
    rows = {reg: Rows(reg, runs, keys) for reg, runs in raw.items()}
    return rows, list(keys)


# --- covariates -----------------------------------------------------------------------

class Covariate:
    """x per item key, possibly per subject fold: X (F, K) over the harness's key
    index and a fold per appearance (0 for a plain dict). Missing -> 0 in X.
    X0 (same shape) is x on the scale the uncentred B0 term reads ('b0'), NaN
    where an item has no x (no B0 offset there); it defaults to X."""

    def __init__(self, X, fold_of=None, name="", X0=None):
        self.X = np.atleast_2d(np.asarray(X, float))
        self.X0 = self.X if X0 is None else np.atleast_2d(np.asarray(X0, float))
        if self.X0.shape != self.X.shape:
            raise ValueError("X0 must have X's shape")
        self.fold_of = fold_of          # sid -> fold, or None
        self.name = name

    @classmethod
    def from_dict(cls, d, keys, name="", x0=None):
        """x0: the dict the B0 term reads (default d); a key it lacks gets NaN."""
        x0 = d if x0 is None else x0
        return cls(np.array([[float(d.get(k, 0.0)) for k in keys]]), None, name,
                   np.array([[float(x0.get(k, np.nan)) for k in keys]]))

    @classmethod
    def from_folds(cls, maps, fold_of, keys, name=""):
        return cls(np.array([[float(m.get(k, 0.0)) for k in keys] for m in maps]), fold_of, name,
                   np.array([[float(m.get(k, np.nan)) for k in keys] for m in maps]))

    def folds(self, rows):
        if self.fold_of is None:
            return np.zeros(rows.A, np.int64)
        return np.array([self.fold_of(s) for s in rows.sid], np.int64)


class Prep:
    """One regime's quantities that depend on the covariate and not on the slope."""

    def __init__(self, rows, cov):
        R = rows
        f = cov.folds(R)
        X = cov.X
        self.x_ev = X[f[R.ev_a], R.ev_k]
        self.x0_ev = cov.X0[f[R.ev_a], R.ev_k]                           # NaN: no B0 offset
        kk = np.where(R.own_k >= 0, R.own_k, 0)
        x_own = np.where(R.own_k >= 0, X[f[:, None], kk], 0.0)          # (A, 31)
        self.x_own = x_own
        cum = np.concatenate([np.zeros((R.A, 1)), np.cumsum(x_own, 1)], 1)
        nl = R.nlab                                                      # (6, A)
        s_own = cum[np.arange(R.A)[None, :], nl]                         # (6, A)
        self.xbar_own = np.where(nl > 0, s_own / np.maximum(nl, 1), 0.0)
        # benchmark centre: every sibling's own labels, x read in the target's fold
        s_b = np.zeros((6, R.A))
        n_b = np.zeros((6, R.A))
        for ff in np.unique(f):
            xo = np.where(R.own_k >= 0, X[ff][kk], 0.0)
            cf = np.concatenate([np.zeros((R.A, 1)), np.cumsum(xo, 1)], 1)
            sf = cf[np.arange(R.A)[None, :], nl]
            m = f == ff
            for bi in range(6):
                gs = np.bincount(R.grp, sf[bi], R.n_grp)
                gn = np.bincount(R.grp, nl[bi], R.n_grp)
                s_b[bi, m] = gs[R.grp[m]]
                n_b[bi, m] = gn[R.grp[m]]
        self.xbar_bench = np.where(n_b > 0, s_b / np.maximum(n_b, 1), 0.0)
        self.n_bench = n_b
        self.c_bench = self.x_ev[None, :] - self.xbar_bench[:, R.ev_a]   # (6, n_ev)
        self.c_bench[n_b[:, R.ev_a] == 0] = 0.0
        self.c_own = self.x_ev[None, :] - self.xbar_own[:, R.ev_a]
        self.c_own[nl[:, R.ev_a] == 0] = 0.0
        self.z_ev = logit(R.ev_p)                                        # (6, n_ev)
        # per-appearance Brier weights for the transferred fit
        ntot = np.bincount(R.ev_a, R.ev_N, R.A)
        self.wN = R.ev_N / ntot[R.ev_a]
        self.wK = R.ev_K / ntot[R.ev_a]


class Engine:
    """Scores slope configurations for one covariate on every regime. scale is
    what the per-pair prior sd s (and the transferred slope's search grid) is
    measured in: 'within' (default) the within-benchmark sd of x, 'pooled' the
    sd pooled across the training parents (the harness before its audit; see
    sd())."""

    def __init__(self, rows, cov, scale=SCALE):
        if scale not in ("within", "pooled"):
            raise ValueError(f"scale {scale!r}")
        self.rows = rows
        self.cov = cov
        self.scale = scale
        self.prep = {reg: Prep(R, cov) for reg, R in rows.items()}
        self._beta = {}
        self._beta0 = {}
        self._sd = {}
        self._x0 = {}
        self._pp = {}
        self._delta = {}

    def _train_x(self, train, attr="x_ev"):
        """x over the training parents' evaluated items in the FIT_ON regimes (one
        value per item and appearance), with each value's parent."""
        xs, ps = [], []
        for reg in FIT_ON:
            if reg in self.rows:
                R = self.rows[reg]
                m = np.isin(R.parent[R.ev_a], list(train))
                xs.append(getattr(self.prep[reg], attr)[m])
                ps.append(R.parent[R.ev_a][m])
        if not xs:
            return np.zeros(0), np.zeros(0, object)
        return np.concatenate(xs), np.concatenate(ps)

    # the transferred slope
    def beta(self, train):
        """beta_B (6,) fitted on FIT_ON appearances whose parent is in `train`."""
        key = tuple(sorted(train))
        if key in self._beta:
            return self._beta[key]
        parts = []
        for reg in FIT_ON:
            if reg not in self.rows:
                continue
            R, P = self.rows[reg], self.prep[reg]
            m = np.isin(R.parent[R.ev_a], list(train))
            parts.append((P.z_ev[:, m], P.c_bench[:, m], P.wN[m], P.wK[m]))
        z = np.concatenate([p[0] for p in parts], 1)
        c = np.concatenate([p[1] for p in parts], 1)
        wN = np.concatenate([p[2] for p in parts])
        wK = np.concatenate([p[3] for p in parts])
        scale = self.sd(train)
        out = np.zeros(6)
        for bi in range(1, 6):
            zb, cb = z[bi], c[bi]
            act = cb != 0
            zb, cb, nb, kb = zb[act], cb[act], wN[act], wK[act]
            p0 = sigmoid(zb)
            base = float(np.sum(nb * p0 * p0 - 2 * kb * p0))

            def f(b, zb=zb, cb=cb, nb=nb, kb=kb, base=base):
                q = sigmoid(zb + capped(b * cb))
                return float(np.sum(nb * q * q - 2 * kb * q)) - base

            out[bi] = fit_scalar(f, scale) if len(zb) else 0.0
        self._beta[key] = out
        return out

    def sd(self, train):
        """The unit of s: the sd of x over the training parents' evaluated items
        in the FIT_ON regimes (one value per item and appearance); 1 when x is
        constant there.

        'within': the pooled within-parent sd, sqrt(sum over parents of
        sum (x - parent mean)^2 / their count), over the parents on which x
        varies. A benchmark-level offset in x, or a parent where x is absent
        (constant), leaves it unchanged, so s means the same thing on every
        benchmark. For x standardised within benchmark (--stage eval's default,
        the degraded oracles) it is about 1.
        'pooled': the sd over all those values together. A covariate with
        benchmark offsets gets a larger unit (raw log_length: 2.36), one absent
        from most items a smaller one (format_score: 0.047, which made s = 0.5
        a prior sd of 10.7 per within-benchmark sd)."""
        key = tuple(sorted(train))
        if key not in self._sd:
            x, par = self._train_x(train)
            if self.scale == "pooled":
                sd = float(np.std(x)) if len(x) > 1 else 0.0
            else:
                ss, n = 0.0, 0
                for q in sorted(set(par.tolist())):
                    xq = x[par == q]
                    if len(xq) > 1 and np.ptp(xq) > 0:
                        ss += float(np.sum((xq - xq.mean()) ** 2))
                        n += len(xq)
                sd = math.sqrt(ss / n) if n else 0.0
            self._sd[key] = sd if sd > 1e-12 else 1.0
        return self._sd[key]

    # the uncentred B0 term
    def x0_ref(self, train):
        """(c0, sd0): the mean and sd of X0 over the training parents' evaluated
        items that carry it (FIT_ON regimes): the fixed reference the B0 term is
        centred on, and its search grid's unit."""
        key = tuple(sorted(train))
        if key not in self._x0:
            x, _ = self._train_x(train, "x0_ev")
            x = x[np.isfinite(x)]
            c0 = float(np.mean(x)) if len(x) else 0.0
            sd = float(np.std(x)) if len(x) > 1 else 0.0
            self._x0[key] = (c0, sd if sd > 1e-12 else 1.0)
        return self._x0[key]

    def beta0(self, train):
        """The B0 slope: minimum of the mean pair Brier at B0 over the FIT_ON
        appearances of the `train` parents, offset beta0 (x0 - c0)."""
        key = tuple(sorted(train))
        if key in self._beta0:
            return self._beta0[key]
        c0, sd0 = self.x0_ref(train)
        zs, cs, ns, ks = [], [], [], []
        for reg in FIT_ON:
            if reg not in self.rows:
                continue
            R, P = self.rows[reg], self.prep[reg]
            m = np.isin(R.parent[R.ev_a], list(train)) & np.isfinite(P.x0_ev)
            zs.append(P.z_ev[0, m])
            cs.append(P.x0_ev[m] - c0)
            ns.append(P.wN[m])
            ks.append(P.wK[m])
        zb, cb = np.concatenate(zs), np.concatenate(cs)
        nb, kb = np.concatenate(ns), np.concatenate(ks)
        act = cb != 0
        zb, cb, nb, kb = zb[act], cb[act], nb[act], kb[act]
        out = 0.0
        if len(zb):
            p0 = sigmoid(zb)
            base = float(np.sum(nb * p0 * p0 - 2 * kb * p0))

            def f(b):
                q = sigmoid(zb + capped(b * cb))
                return float(np.sum(nb * q * q - 2 * kb * q)) - base

            out = fit_scalar(f, sd0)
        self._beta0[key] = out
        return out

    def pair_betas(self, reg, m6, s_eff):
        """(6, A) per-pair MAP slopes, prior N(m6[B], s_eff^2)."""
        key = (reg, tuple(np.round(m6, 12)), round(s_eff, 12))
        if key in self._pp:
            return self._pp[key]
        R, P = self.rows[reg], self.prep[reg]
        out = np.zeros((6, R.A))
        L = R.own_k.shape[1]
        j = np.arange(L)[None, :]
        for bi in range(1, 6):
            nl = R.nlab[bi]
            use = nl >= 2
            if not use.any():
                out[bi] = m6[bi]
                continue
            mask = (j < nl[:, None]) & use[:, None]
            c = P.x_own - P.xbar_own[bi][:, None]
            u = logit(R.own_pu[bi])
            b = pair_slope(u[use], R.own_y[use], c[use], mask[use], m6[bi], s_eff)
            out[bi] = m6[bi]
            out[bi, use] = b
        self._pp[key] = out
        return out

    def delta(self, reg, config, train):
        """(A, 6) Brier differences of one configuration on one regime, fitted on
        `train` parents. config = (variant, s, start)."""
        key = (reg, tuple(config), tuple(sorted(train)))
        if key not in self._delta:
            self._delta[key] = self._compute_delta(reg, config, train)
        return self._delta[key]

    def _compute_delta(self, reg, config, train):
        variant, s, start = config
        R, P = self.rows[reg], self.prep[reg]
        off = np.zeros_like(P.z_ev)
        if variant == "b0":
            c0, _ = self.x0_ref(train)
            ok = np.isfinite(P.x0_ev)
            off[0, ok] = self.beta0(train) * (P.x0_ev[ok] - c0)
        else:
            if variant == "transferred":
                b = self.beta(train)
                off = b[:, None] * P.c_bench
            elif variant in ("per-pair", "hybrid"):
                m6 = self.beta(train) if variant == "hybrid" else np.zeros(6)
                bp = self.pair_betas(reg, m6, s / self.sd(train))
                off = bp[:, R.ev_a] * P.c_own
            else:
                raise ValueError(f"variant {variant!r}")
            off[0] = 0.0
        for bi in range(6):
            if BUDGETS[bi] < start:
                off[bi] = 0.0
        off = capped(off)
        out = np.zeros((R.A, 6))
        for bi in range(6):
            if not off[bi].any():
                continue
            p = R.ev_p[bi]
            q = shifted(p, off[bi])
            out[:, bi] = delta_brier(p, q, R.ev_K, R.ev_N, R.ev_a, R.A)
        return out


def configs_of(variant):
    if variant == "transferred":
        return [("transferred", None, st) for st in STARTS]
    if variant == "b0":
        return [("b0", None, 0)]
    return [(variant, s, st) for s in S_GRID for st in STARTS]


def parent_means(rows, dalc, parents):
    """Weighted mean ALC difference per parent, for the parents present."""
    out = {}
    for q in parents:
        m = rows.parent == q
        if m.any():
            out[q] = weighted_mean(dalc[m], rows.w[m])
    return out


def outer_folds():
    """Held-out parent -> training parents; '*' (appearances outside the four
    parents, swe_rebench) trains on all four."""
    folds = {q: tuple(p for p in PARENTS if p != q) for q in PARENTS}
    folds["*"] = PARENTS
    return folds


def inner_criterion(engine, c, train):
    """The inner selection criterion of configuration c for one outer fold: the
    mean over the inner parents q2 of the SELECT_ON ALC difference on q2 with c
    fitted on train minus q2, and its SE (the parents' linearised cluster SEs
    combined as independent)."""
    sel = engine.rows[SELECT_ON]
    vals, var = [], 0.0
    for q2 in train:
        inner = tuple(p for p in train if p != q2)
        d = engine.delta(SELECT_ON, c, inner) @ W6
        m = sel.parent == q2
        vals.append(weighted_mean(d[m], sel.w[m]))
        var += lin_se(d[m], sel.w[m], sel.cluster[m]) ** 2
    return float(np.mean(vals)), math.sqrt(var) / len(vals)


def nested(engine, variant, forced=None, k=SELECT_K):
    """Nested leave-one-parent-out (or one forced configuration): per regime the
    (A, 6) differences each appearance gets under its fold's choice, and the
    choices. A configuration is chosen only if its inner mean is below -k inner
    SEs; `acts` records whether the choice moves any prediction of the held-out
    parent (a covariate constant there does not)."""
    rows = engine.rows
    cand = configs_of(variant) if forced is None else [forced]
    out = {reg: np.zeros((R.A, 6)) for reg, R in rows.items()}
    choices = {}
    for q, train in outer_folds().items():
        if forced is None:
            crit, se = {}, {}
            for c in cand:
                crit[c], se[c] = inner_criterion(engine, c, train)
            choice = select(crit, se, k)
            loose = select(crit)
        else:
            crit, se, choice, loose = None, None, forced, forced
        choices[q] = {"choice": None if choice is None else list(choice),
                      "choice_no_margin": None if loose is None else list(loose),
                      "inner": None if crit is None else {cname(c): round(v, 7) for c, v in crit.items()},
                      "inner_se": None if se is None else {cname(c): round(v, 7) for c, v in se.items()},
                      "acts": False}
        if choice is None:
            continue
        if choice[0] in ("transferred", "hybrid"):
            choices[q]["beta"] = [round(float(b), 5) for b in engine.beta(train)]
        if choice[0] == "b0":
            choices[q]["beta0"] = round(float(engine.beta0(train)), 5)
            choices[q]["c0"] = round(engine.x0_ref(train)[0], 6)
        choices[q]["sd_x"] = round(engine.sd(train), 6)
        for reg, R in rows.items():
            m = (R.parent == q) if q != "*" else ~np.isin(R.parent, PARENTS)
            if m.any():
                out[reg][m] = engine.delta(reg, choice, train)[m]
                choices[q]["acts"] = bool(choices[q]["acts"] or np.any(out[reg][m] != 0))
    return out, choices


def folds_on(choices):
    """Outer folds (of the four parents) whose choice is on and acts there."""
    return sum(1 for q in PARENTS if q in choices and choices[q]["choice"] is not None
               and choices[q]["acts"])


def selection_boot_se(engine, variant, boots=BOOTS, seed=0, k=SELECT_K):
    """Pair-cluster bootstrap SE of a nested line's SELECT_ON estimate with the
    inner selection redone in every resample: the inner criteria and their
    linearised SEs are recomputed on the resampled clusters, each outer fold
    chooses again, and the estimate is the resample's mean under those choices.
    The fitted slopes are held at their full-sample values. The resamples are
    boot_se's (same seed, same clusters), so where no resample changes a choice
    this equals the fixed-selection cluster SE."""
    R = engine.rows[SELECT_ON]
    cand = configs_of(variant)
    cl, inv = np.unique(R.cluster, return_inverse=True)
    G = len(cl)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, G, (boots, G))
    cnt = np.zeros((boots, G))
    for b in range(boots):
        cnt[b] = np.bincount(pick[b], minlength=G)
    Wc = np.bincount(inv, R.w, G)
    num = np.zeros(boots)
    for q, train in outer_folds().items():
        mq = (R.parent == q) if q != "*" else ~np.isin(R.parent, PARENTS)
        if not mq.any():
            continue
        crit = np.zeros((boots, len(cand)))
        var = np.zeros((boots, len(cand)))
        outer = np.zeros((G, len(cand)))
        for ci, c in enumerate(cand):
            for q2 in train:
                inner = tuple(p for p in train if p != q2)
                d = engine.delta(SELECT_ON, c, inner) @ W6
                m = R.parent == q2
                S = np.bincount(inv[m], R.w[m] * d[m], G)
                W = np.bincount(inv[m], R.w[m], G)
                on = (W > 0).astype(float)
                Sb, Wb, nb = cnt @ S, cnt @ W, cnt @ on
                mb = np.divide(Sb, Wb, out=np.zeros(boots), where=Wb > 0)
                ss = cnt @ (S * S) - 2 * mb * (cnt @ (S * W)) + mb * mb * (cnt @ (W * W))
                vb = np.divide(nb * np.maximum(ss, 0.0), np.maximum(nb - 1, 1) * np.maximum(Wb, 1e-300) ** 2)
                crit[:, ci] += mb / len(train)
                var[:, ci] += np.where(nb > 1, vb, 0.0)
            d = engine.delta(SELECT_ON, c, train) @ W6
            outer[:, ci] = np.bincount(inv[mq], R.w[mq] * d[mq], G)
        se = np.sqrt(var) / len(train)
        ok = crit < np.minimum(-k * se, 0.0)
        masked = np.where(ok, crit, np.inf)
        best = np.argmin(masked, 1)
        chosen = np.isfinite(masked[np.arange(boots), best])
        vals = cnt @ outer                                   # (boots, n_cand)
        num += np.where(chosen, vals[np.arange(boots), best], 0.0)
    est = num / (cnt @ Wc)
    return float(np.std(est, ddof=1))


def cname(c):
    v, s, st = c
    return f"{v} from B{st}" if s is None else f"{v} s={s:g} from B{st}"


def summary(rows, dB, boots=BOOTS, seed=0, per=True):
    """Everything reported for one line on one regime."""
    R = rows
    dalc = dB @ W6
    st = stats(dalc, R.w, R.run, R.cluster, R.strata, boots, seed)
    if st is None:
        return None
    pm = parent_means(R, dalc, sorted(set(R.parent.tolist())))
    multi = [pm[q] for q in PARENTS if q in pm]
    st["parent_mean"] = float(np.mean(multi)) if multi else None
    st["parent_se"] = float(np.std(multi, ddof=1) / math.sqrt(len(multi))) if len(multi) > 1 else None
    st["worst_parent"] = max(multi) if multi else None
    st["by_budget"] = [weighted_mean(dB[:, bi], R.w) for bi in range(6)]
    if per:
        st["by_budget_cluster_se"] = [boot_se(dB[:, bi], R.w, R.cluster, None, boots // 4, seed)
                                      if dB[:, bi].any() else 0.0 for bi in range(6)]
        st["per_parent"] = {}
        for q in sorted(set(R.parent.tolist())):
            m = R.parent == q
            st["per_parent"][q] = {"est": pm[q], "cluster_se": boot_se(dalc[m], R.w[m], R.cluster[m],
                                                                       None, boots // 4, seed),
                                   "n_app": int(m.sum())}
        if R.regime not in GUARDS:
            st["per_kind"] = {k: round(weighted_mean(dalc[R.kind == k], R.w[R.kind == k]), 6)
                              for k in sorted(set(R.kind.tolist()))}
    return st


def gate(line):
    """The plan's acceptance rule on one line's summaries. A forced line has no
    selection, so its `pass` is None and `pass_if_on` says whether the rest of
    the rule holds."""
    tl, mix = line["regimes"].get("tl"), line["regimes"].get("mix")
    on = line.get("folds_on")
    checks = {
        "selection_on": None if on is None else bool(on >= GATE["folds_on"]),
        "tl": tl is not None and tl["est"] <= GATE["tl"],
        "mix_same_sign": mix is not None and tl is not None and np.sign(mix["est"]) == np.sign(tl["est"])
        and tl["est"] != 0,
        "no_parent_worse": tl is not None and tl["worst_parent"] is not None
        and tl["worst_parent"] <= GATE["parent"],
        "guard": all(line["regimes"].get(g) is None or line["regimes"][g]["est"] <= GATE["guard"]
                     for g in GUARDS),
    }
    rest = all(bool(v) for k, v in checks.items() if k != "selection_on")
    checks["pass_if_on"] = rest
    checks["pass"] = None if on is None else bool(rest and checks["selection_on"])
    return checks


def score_covariate(rows, cov, variants=VARIANTS, forced=FORCED, boots=BOOTS, per=True, scale=SCALE):
    """Nested lines per variant and the forced lines: {line: {...}} plus the raw
    per-appearance differences for averaging over replicates. folds_on counts
    the outer folds whose choice is on and acts on the held-out parent;
    folds_chosen those whose choice is on at all; folds_on_no_margin those that
    would be on without the SELECT_K margin. With per, each nested line's
    test-like summary also carries sel_cluster_se (selection_boot_se)."""
    eng = Engine(rows, cov, scale)
    lines, raw = {}, {}
    for v in variants:
        dB, ch = nested(eng, v)
        raw[f"{v} nested"] = dB
        lines[f"{v} nested"] = {
            "choices": ch, "folds_on": folds_on(ch),
            "folds_chosen": sum(1 for q in PARENTS if ch[q]["choice"] is not None),
            "folds_on_no_margin": sum(1 for q in PARENTS if ch[q]["choice_no_margin"] is not None)}
    for c in forced:
        dB, ch = nested(eng, c[0], forced=c)
        raw[cname(c) + " (forced)"] = dB
        lines[cname(c) + " (forced)"] = {"choices": {q: {k: v for k, v in x.items()
                                                         if k not in ("inner", "inner_se", "choice_no_margin")}
                                                     for q, x in ch.items()}}
    for name in lines:
        lines[name]["regimes"] = {reg: summary(R, raw[name][reg], boots, 0, per)
                                  for reg, R in rows.items()}
        lines[name]["gate"] = gate(lines[name])
    if per:
        for v in variants:
            tl = lines[f"{v} nested"]["regimes"].get(SELECT_ON)
            if tl is not None:
                tl["sel_cluster_se"] = selection_boot_se(eng, v, boots)
    return lines, raw


def average_lines(rows, reps, boots=BOOTS):
    """Per-appearance differences averaged over replicate covariates, summarised
    again; the replicates' own estimates' spread beside them."""
    names = list(reps[0][1])
    out = {}
    for name in names:
        dB = {reg: np.mean([r[1][name][reg] for r in reps], 0) for reg in rows}
        line = {"regimes": {reg: summary(R, dB[reg], boots, 0, True) for reg, R in rows.items()}}
        line["replicates"] = {reg: [round(r[0][name]["regimes"][reg]["est"], 6) for r in reps]
                              for reg in rows}
        line["draw_sd"] = {reg: float(np.std(v, ddof=1)) if len(v) > 1 else None
                           for reg, v in line["replicates"].items()}
        fo = [r[0][name].get("folds_on") for r in reps]
        if fo[0] is not None:
            line["folds_on"] = float(np.mean(fo))
            line["folds_on_reps"] = fo
            line["folds_chosen_reps"] = [r[0][name].get("folds_chosen") for r in reps]
        # each draw is what one real covariate would be: does it pass on its own?
        line["pass_reps"] = [bool(r[0][name]["gate"]["pass"] if r[0][name]["gate"]["pass"] is not None
                                  else r[0][name]["gate"]["pass_if_on"]) for r in reps]
        line["choices_rep0"] = reps[0][0][name]["choices"]
        line["gate"] = gate(line)
        out[name] = line
    return out


# --- oracles ------------------------------------------------------------------------

def oracle_maps():
    """In-sample Rasch difficulty per parent (testlike.Catalogue.difficulty) and
    honest fold maps: fold f's difficulties from the parent's subjects outside f."""
    from paiec import data as D
    from paiec import official as O
    from paiec import testlike as T
    ins, honest = {}, [dict() for _ in range(HONEST_FOLDS)]
    info = {}
    for par in PARENTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = O.eligible(D.load_pairs([par]))
        ins[par] = T.rasch(ps)
        info[par] = {"pairs": len(ps), "items": len(ins[par])}
        for f in range(HONEST_FOLDS):
            keep = [p for p in ps if fold_of(p.subject_id) != f]
            hf = T.rasch(keep)
            honest[f].update(hf)
            info[par][f"unseen_fold{f}"] = len(set(ins[par]) - set(hf))
    return ins, honest, info


def fold_of(sid):
    from paiec.evaluator import stable_hash
    return stable_hash(HONEST_SALT, "fold", str(sid)) % HONEST_FOLDS


def name_fold_maps():
    """The honest oracle with folds by canonical model name (paiec.prior.
    canon_name) instead of subject_id, so near-duplicate subjects (one model
    under several harnesses or efforts; 75 of multi_swebench's 82 pairs belong
    to 13 names) share a fold: (maps, fold function)."""
    from paiec import data as D
    from paiec import official as O
    from paiec import testlike as T
    from paiec.evaluator import stable_hash
    from paiec.prior import canon_name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        allp = O.eligible(D.load_pairs(list(PARENTS)))
    name_of = {str(p.subject_id): (canon_name(p.subject) or str(p.subject_id)) for p in allp}

    def fold_name(sid):
        return stable_hash(HONEST_SALT, "fold", name_of.get(str(sid), str(sid))) % HONEST_FOLDS

    maps = [dict() for _ in range(HONEST_FOLDS)]
    for par in PARENTS:
        ps = [p for p in allp if p.benchmark_id == par]
        for f in range(HONEST_FOLDS):
            maps[f].update(T.rasch([p for p in ps if fold_name(p.subject_id) != f]))
    return maps, fold_name


def base_matrix(maps, keys, parent_of_key):
    """(F, K) difficulties standardised within parent, one row per fold map, and
    where each is present; items outside the four parents are never present."""
    F = len(maps)
    Z = np.zeros((F, len(keys)))
    has = np.zeros((F, len(keys)), bool)
    pk = np.array([parent_of_key.get(k, "") for k in keys], object)
    for f, m in enumerate(maps):
        v = np.array([m.get(k, np.nan) if pk[j] else np.nan for j, k in enumerate(keys)], float)
        has[f] = np.isfinite(v)
        Z[f, has[f]] = standardise_within(v[has[f]], pk[has[f]])
    return Z, has, pk != ""


def realised_r(rows, cov, Z):
    """Mean over test-like pair appearances of the correlation between a
    covariate and the base it was degraded from (the appearance's fold), over
    the pair's evaluated items: lower than r, as a pseudo-benchmark spans less
    difficulty than its parent."""
    R = rows["tl"]
    f = cov.folds(R)
    rs = []
    for a in range(R.A):
        m = R.ev_a == a
        if m.sum() < 5:
            continue
        x, t = cov.X[f[a], R.ev_k[m]], Z[f[a], R.ev_k[m]]
        if x.std() > 0 and t.std() > 0:
            rs.append(np.corrcoef(x, t)[0, 1])
    return float(np.mean(rs)) if rs else None


def degraded_table(rows, keys, Z, has, in_parent, fold_fn, reps, one, save, done=None):
    """r -> the lines, each averaged over `reps` noise draws: x = r z + sqrt(1 -
    r^2) e over the four parents' items (e alone where the base lacks the item),
    0 elsewhere; r = 0 is the uninformative covariate. The B0 term reads the
    same x on the four parents' items and nothing elsewhere. Rows already in
    `done` (a resumed table) are kept."""
    table = dict(done or {})
    for r in (0.0,) + tuple(R_GRID):
        if f"r={r:g}" in table:
            print(f"r={r:g}: kept from the resumed table", flush=True)
            continue
        runs = []
        cov0 = None
        for rep in range(reps):
            eps = np.random.default_rng([rep, 7919]).standard_normal(len(keys))
            X = np.where(in_parent[None, :], degrade(np.where(has, Z, 0.0), r, eps[None, :]), 0.0)
            cov = Covariate(X, fold_fn, f"r={r}", np.where(in_parent[None, :], X, np.nan))
            cov0 = cov0 or cov
            runs.append(one(f"r={r:g} rep {rep}", cov, boots=200, per=False))
        avg = average_lines(rows, runs)
        table[f"r={r:g}"] = {"meta": {"r": r, "r_within_pair_tl": realised_r(rows, cov0, Z)},
                             "lines": avg}
        save(table)
        print(f"r={r:g}: " + "  ".join(
            f"{ln}: tl {avg[ln]['regimes']['tl']['est']:+.5f} ({sum(avg[ln]['pass_reps'])}/{reps})"
            for ln in avg), flush=True)
    return {k: table[k] for k in sorted(table, key=lambda k: float(k[2:]))}


def stage_table(args):
    t0 = time.time()
    rows, keys = load_rows(args.rows, args.regimes)
    print(f"table: rows of {', '.join(f'{r} {len(set(R.run.tolist()))}' for r, R in rows.items())} "
          f"runs loaded in {time.time() - t0:.0f}s", flush=True)
    ins, honest, info = oracle_maps()
    print(f"table: oracle difficulties in {time.time() - t0:.0f}s", flush=True)
    par_of_key = {k: par for par, d in ins.items() for k in d}
    prov = provenance()
    reps = {"honest": args.reps, "in_sample": args.reps_in_sample}
    out = {"meta": {"plan": {r: PLAN[r] for r in rows}, "runs_loaded": {r: int(len(set(R.run.tolist())))
                                                                       for r, R in rows.items()},
                    "appearances": {r: R.A for r, R in rows.items()},
                    "hier_failures": {r: R.hier_failures for r, R in rows.items()},
                    "hier_unconverged": {r: R.hier_unconverged for r, R in rows.items()},
                    "rows_lib_digest": rows_digest(rows), "rows_dir": os.path.relpath(args.rows, ROOT),
                    "fit_on": FIT_ON, "select_on": SELECT_ON, "s_grid": S_GRID, "starts": STARTS,
                    "variants": VARIANTS, "scale": SCALE, "select_k": SELECT_K,
                    "cap": CAP, "boots": BOOTS, "n_reps": reps, "r_grid": R_GRID,
                    "honest_folds": HONEST_FOLDS, "gate": GATE, "uninformative_cap": UNINFORMATIVE_CAP,
                    "accept": ACCEPT, "oracle_info": info, "recomputed": RECOMPUTED, **prov}}
    state = load_json(args.out) or {}
    same = bool(args.resume and state.get("meta")
                and all(state["meta"].get(k) == _jsonable(out["meta"][k])
                        for k in ("lib_digest", "rows_lib_digest", "n_reps", "r_grid", "runs_loaded", "variants",
                                  "scale", "select_k", "s_grid", "starts", "boots")))
    if args.resume:
        print(f"table: resume {'from ' + args.out if same else 'impossible (provenance differs): from scratch'}",
              flush=True)
    for k in ("verify",):
        if k in state:
            out[k] = state[k]
    if same:
        for k in ("acceptance", "tables", "thresholds", "selection_se"):
            if k in state:
                out[k] = state[k]
        # the script digests of every pass whose parts are kept, oldest first
        out["meta"]["script_digests"] = (state["meta"].get("script_digests") or [state["meta"]["script_digest"]])
        if out["meta"]["script_digests"][-1] != prov["script_digest"]:
            out["meta"]["script_digests"] = out["meta"]["script_digests"] + [prov["script_digest"]]

    def one(name, cov, boots=BOOTS, per=True):
        t = time.time()
        lines, raw = score_covariate(rows, cov, boots=boots, per=per)
        print(f"  {name}: {time.time() - t:.0f}s  " + "  ".join(
            f"{ln}: tl {lines[ln]['regimes']['tl']['est']:+.5f} mix {lines[ln]['regimes']['mix']['est']:+.5f}"
            for ln in ("transferred nested", "per-pair nested", "b0 nested")), flush=True)
        return lines, raw

    # acceptance: the in-sample oracle on its raw logit scale, as in the heads study
    acc = out.setdefault("acceptance", {})
    if "oracle_in_sample" not in acc:
        cov_ins = Covariate.from_dict(ins_all(ins), keys, "oracle")
        lines, _ = one("oracle (in sample)", cov_ins)
        tl_est = lines["transferred from B1 (forced)"]["regimes"]["tl"]["est"]
        mix_est = lines["transferred from B1 (forced)"]["regimes"]["mix"]["est"]
        acc["check"] = {"line": "transferred from B1 (forced)", "tl": tl_est, "mix": mix_est,
                        "target": ACCEPT, "pass": abs(tl_est - ACCEPT["tl"]) <= ACCEPT["tol"]
                        and abs(mix_est - ACCEPT["mix"]) <= ACCEPT["tol"]}
        acc["oracle_in_sample"] = lines
        save_json(args.out, out)
    ck = acc["check"]
    print(f"acceptance: transferred oracle tl {ck['tl']:+.5f} (target {ACCEPT['tl']}), mix {ck['mix']:+.5f} "
          f"(target {ACCEPT['mix']}): {'pass' if ck['pass'] else 'FAIL'}", flush=True)
    if "oracle_honest" not in acc:
        lines, _ = one("oracle (honest, subject folds)", Covariate.from_folds(honest, fold_of, keys, "honest"))
        acc["oracle_honest"] = lines
        save_json(args.out, out)
    if "oracle_honest_namefold" not in acc:
        nmaps, nfold = name_fold_maps()
        lines, _ = one("oracle (honest, model-name folds)", Covariate.from_folds(nmaps, nfold, keys, "honest name"))
        acc["oracle_honest_namefold"] = lines
        save_json(args.out, out)
    ins_tl = acc["oracle_in_sample"]["transferred nested"]["regimes"]["tl"]["est"]
    hon_tl = acc["oracle_honest"]["transferred nested"]["regimes"]["tl"]["est"]
    acc["leak"] = {"tl_transferred_nested_in_sample": ins_tl, "tl_transferred_nested_honest": hon_tl,
                   "share_of_in_sample_that_is_leak": (ins_tl - hon_tl) / ins_tl if ins_tl else None}
    # the degraded oracles: the gate table on the honest base, the heads study's on the in-sample one
    tables = out.setdefault("tables", {})
    for base, maps, fn in (("honest", honest, fold_of), ("in_sample", [ins_all(ins)], None)):
        Z, has, in_parent = base_matrix(maps, keys, par_of_key)

        def save(table, base=base):
            tables[base] = table
            save_json(args.out, out)

        table = degraded_table(rows, keys, Z, has, in_parent, fn, reps[base], one, save,
                               done=tables.get(base, {}))
        tables[base] = table
        out.setdefault("thresholds", {})[base] = thresholds(table)
        save_json(args.out, out)
    # how much redoing the selection inside the bootstrap widens a nested line's
    # cluster SE where the choice is uncertain: draw 0 of the honest table's
    # middle rows, scored in full (the table's draws are scored without it)
    sel = out.setdefault("selection_se", {})
    Z, has, in_parent = base_matrix(honest, keys, par_of_key)
    for r in SEL_CHECK_R:
        if f"r={r:g}" in sel:
            continue
        eps = np.random.default_rng([0, 7919]).standard_normal(len(keys))
        X = np.where(in_parent[None, :], degrade(np.where(has, Z, 0.0), r, eps[None, :]), 0.0)
        cov = Covariate(X, fold_of, f"r={r}", np.where(in_parent[None, :], X, np.nan))
        lines, _ = score_covariate(rows, cov, variants=("transferred", "per-pair"), forced=())
        sel[f"r={r:g}"] = {ln: {"tl": lines[ln]["regimes"]["tl"]["est"],
                                "cluster_se": lines[ln]["regimes"]["tl"]["cluster_se"],
                                "sel_cluster_se": lines[ln]["regimes"]["tl"]["sel_cluster_se"],
                                "folds_on": lines[ln]["folds_on"], "folds_on_no_margin": lines[ln]["folds_on_no_margin"],
                                "choices": {q: v["choice"] for q, v in lines[ln]["choices"].items()}}
                           for ln in ("transferred nested", "per-pair nested")}
        print(f"selection SE r={r:g}: " + "  ".join(
            f"{ln}: tl {v['tl']:+.5f} cluster {v['cluster_se']:.5f} selection-aware {v['sel_cluster_se']:.5f}"
            for ln, v in sel[f"r={r:g}"].items()), flush=True)
        save_json(args.out, out)
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    if same:
        out["meta"]["wall_s_passes"] = (state["meta"].get("wall_s_passes") or [state["meta"].get("wall_s")]) \
            + [out["meta"]["wall_s"]]
    save_json(args.out, out)
    print(f"table: done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


def rows_digest(rows):
    """The library digests the loaded rows were collected with, per regime."""
    return {reg: R.lib_digests for reg, R in rows.items()}


def ins_all(ins):
    return {k: v for d in ins.values() for k, v in d.items()}


def thresholds(table):
    """Per line, the tl / mix / guard estimate at each r (with the cluster SE, the
    spread over noise draws, the worst parent and the pass rate over draws) and
    the smallest r whose averaged line passes the gate (a forced line: would pass
    if selection switched it on), the smallest r at which a majority of draws
    pass and the smallest at which every draw does: the one table every
    covariate is read against. A real covariate is one draw, so the pass rate is
    the chance that a covariate at that r passes."""
    out = {}
    names = list(next(iter(table.values()))["lines"])
    for name in names:
        row = {}
        first = most = every = None
        forced = True
        for key, t in table.items():
            ln = t["lines"][name]
            rg = ln["regimes"]
            row[key] = {reg: round(rg[reg]["est"], 6) for reg in rg if rg[reg] is not None}
            row[key]["tl_cluster_se"] = round(rg["tl"]["cluster_se"], 6)
            row[key]["tl_worst_parent"] = round(rg["tl"]["worst_parent"], 6)
            row[key]["r_within_pair_tl"] = t["meta"].get("r_within_pair_tl")
            if ln.get("draw_sd"):
                row[key]["tl_draw_sd"] = round(ln["draw_sd"]["tl"], 6)
            if "folds_on" in ln:
                row[key]["folds_on"] = ln["folds_on"]
                forced = False
            ok = ln["gate"]["pass"] if ln["gate"]["pass"] is not None else ln["gate"]["pass_if_on"]
            row[key]["pass" if ln["gate"]["pass"] is not None else "pass_if_on"] = ok
            pr = ln.get("pass_reps")
            if pr:
                row[key]["pass_draws"] = f"{sum(pr)}/{len(pr)}"
                if most is None and 2 * sum(pr) > len(pr):
                    most = t["meta"]["r"]
                if every is None and all(pr):
                    every = t["meta"]["r"]
            if first is None and ok:
                first = t["meta"]["r"]
        sfx = "_if_on" if forced else ""
        row["smallest_r_passing" + sfx] = first
        row["smallest_r_majority_of_draws" + sfx] = most
        row["smallest_r_every_draw" + sfx] = every
        out[name] = row
    return out


# --- eval one covariate ---------------------------------------------------------------

def read_covariate(path):
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return {str(k): float(v) for k, v in d.items() if v is not None and math.isfinite(float(v))}
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    col = "x" if "x" in df.columns else [c for c in df.columns if c != "item_key"][0]
    df = df[np.isfinite(df[col].astype(float))]
    return dict(zip(df["item_key"].astype(str), df[col].astype(float)))


def keys_by_parent(rows, keys):
    """{parent: [item key]} of the items the rows hold (evaluated or acquired)."""
    out = {}
    for R in rows.values():
        par_ev = R.parent[R.ev_a]
        for k, q in zip(R.ev_k.tolist(), par_ev.tolist()):
            out.setdefault(q, set()).add(keys[k])
        for a in range(R.A):
            for k in R.own_k[a][R.own_k[a] >= 0].tolist():
                out.setdefault(R.parent[a], set()).add(keys[k])
    return {q: sorted(v) for q, v in out.items()}


def benchmark_items(rows, keys):
    """{benchmark: [item key]} for the benchmarks the rows hold: every item of
    the public benchmark (str(item_id) of data/<benchmark>/items.parquet, the
    key of the rows), or the items the rows hold where that file is missing."""
    held = keys_by_parent(rows, keys)
    out = {}
    for b, ks in held.items():
        path = os.path.join(os.environ.get("PAIEC_DATA", os.path.join(ROOT, "data")), b, "items.parquet")
        try:
            import pandas as pd
            out[b] = pd.read_parquet(path, columns=["item_id"]).item_id.astype(str).tolist()
        except Exception:
            out[b] = ks
    return out


def eval_covariate(raw, items_bench, keys, name="", standardise=True):
    """The Covariate --stage eval scores, and what was done to x. By default x is
    standardised within benchmark (standardised()), the centred terms read it
    and the uncentred B0 term reads raw x (its absolute scale: at B0 a
    run-time predictor sees one item and no benchmark to standardise over);
    standardise=False reads raw x everywhere."""
    d = standardised(raw, items_bench) if standardise else dict(raw)
    info = {"input": "standardised within benchmark (clip +-%g sd, min_vary %d)" % (CLIP, MIN_VARY)
            if standardise else "raw", "per_benchmark": {}}
    for b, ks in items_bench.items():
        have = [k for k in ks if k in raw]
        info["per_benchmark"][b] = {"items": len(ks), "with_x": len(have),
                                    "standardised_nonzero": int(sum(1 for k in have if d.get(k, 0.0) != 0.0))}
    known = {k for ks in items_bench.values() for k in ks}
    info["keys_outside_benchmarks"] = int(sum(1 for k in raw if k not in known))
    return Covariate.from_dict(d, keys, name, x0=raw), info


def stage_eval(args):
    """One covariate against the shipped hier. Input convention: --cov holds x on
    any scale, one value per item key (str(item_id)); an item without x gets
    none. By default x is standardised within each public benchmark over the
    items that carry it (0 = the benchmark's mean, clipped at +-CLIP sd; a
    benchmark where fewer than MIN_VARY items are off x's mode gets no x), so
    the per-pair prior sd s is per within-benchmark sd whatever x's scale, and
    benchmark-level offsets in x cannot act. --raw skips that (x as given;
    Engine.sd is still within-benchmark)."""
    t0 = time.time()
    rows, keys = load_rows(args.rows, args.regimes)
    raw = read_covariate(args.cov)
    cov, info = eval_covariate(raw, benchmark_items(rows, keys), keys, args.name, not args.raw)
    covered = {reg: float(np.mean(np.isin(np.array(keys, object)[R.ev_k], list(raw))))
               for reg, R in rows.items()}
    lines, _ = score_covariate(rows, cov)
    prov = provenance()
    res = {"name": args.name, "cov": os.path.abspath(args.cov), "n_items": len(raw), "x": info,
           "coverage_eval_items": covered, "scale": SCALE, "select_k": SELECT_K, "gate": GATE,
           "rows_lib_digest": rows_digest(rows), "lines": lines, **prov}
    stale = sorted({d for v in res["rows_lib_digest"].values() for d in v} - {prov["lib_digest"]})
    if stale:
        res["warning"] = (f"rows collected with library {stale}, library now {prov['lib_digest']}: "
                          "re-collect (--stage collect into a new --rows) before trusting the numbers")
        print("WARNING:", res["warning"], flush=True)
    thr = load_json(OUT)
    if thr and "thresholds" in thr:
        res["thresholds"] = thr["thresholds"]
    res["wall_s"] = round(time.time() - t0, 1)
    out = args.out if args.out != OUT else os.path.join(ROOT, "results", f"harness_{args.name}.json")
    save_json(out, res)
    for name, ln in lines.items():
        rg = ln["regimes"]
        sel = rg["tl"].get("sel_cluster_se")
        print(f"{name:34s} tl {rg['tl']['est']:+.5f} ± {rg['tl']['run_se']:.5f} / "
              f"{rg['tl']['cluster_se']:.5f}{'' if sel is None else f' (sel {sel:.5f})'} / "
              f"{rg['tl']['strat_se']:.5f}  mix {rg['mix']['est']:+.5f}  "
              + "  ".join(f"{g} {rg[g]['est']:+.5f}" for g in GUARDS if rg.get(g))
              + ("" if ln.get("folds_on") is None else f"  on {ln['folds_on']}/4")
              + "  gate " + ({True: "PASS", False: "fail"}[ln["gate"]["pass"]] if ln["gate"]["pass"] is not None
                             else "(forced: pass if on)" if ln["gate"]["pass_if_on"] else "(forced: fail)"),
              flush=True)
    print(f"eval: {time.time() - t0:.0f}s -> {out}", flush=True)


# --- show -----------------------------------------------------------------------------

def _f(x, nd=4):
    return "" if x is None else f"{x:+.{nd}f}"


def line_row(name, ln, nd=4):
    """One markdown row: tl (± run / cluster / stratified SE; the selection-aware
    cluster SE in brackets where computed), mix, worst parent, parent SE, the
    guards, folds on, gate (with the pass rate over noise draws where there are
    draws)."""
    rg = ln["regimes"]
    tl = rg["tl"]
    sel = tl.get("sel_cluster_se")
    pr = ln.get("pass_reps")
    gate_s = (("pass" if ln["gate"]["pass"] else "no") if ln["gate"]["pass"] is not None
              else ("(pass if on)" if ln["gate"]["pass_if_on"] else "(no)"))
    cells = [name, f"{_f(tl['est'], nd)} ± {tl['run_se']:.{nd}f} / {tl['cluster_se']:.{nd}f}"
                   + ("" if sel is None else f" ({sel:.{nd}f})") + f" / {tl['strat_se']:.{nd}f}",
             _f(rg["mix"]["est"], nd) if rg.get("mix") else "",
             _f(tl["worst_parent"], nd), f"{tl['parent_se']:.{nd}f}" if tl.get("parent_se") else "",
             _f(rg["r1b"]["est"], nd) if rg.get("r1b") else "",
             _f(rg["r1p"]["est"], nd) if rg.get("r1p") else "",
             "" if ln.get("folds_on") is None else f"{ln['folds_on']:g}/4",
             gate_s + ("" if not pr else f" {sum(pr)}/{len(pr)}")]
    return "| " + " | ".join(cells) + " |"


HEAD = ("| line | test-like ± run / cluster (selection-aware) / strat SE | mix/whole | worst parent | parent SE "
        "| R1 benchmark-first | R1 pair-uniform | folds on | gate (draws passing) |\n|" + "---|" * 9)


def gate_table(thr, lines=("transferred nested", "per-pair nested", "b0 nested")):
    """The thresholds block as one markdown table per line: r (within-pair r),
    test-like (cluster SE, draw sd), mix/whole, worst parent, R1 b / p, folds
    on, gate, draws passing."""
    out = []
    for name in lines:
        if name not in thr:
            continue
        row = thr[name]
        out.append(f"\n{name}\n| r (within pair) | test-like (cluster SE, draw sd) | mix/whole | worst parent "
                   f"| R1 b / p | folds on | gate | draws passing |\n|" + "---|" * 8)
        for key, c in row.items():
            if not key.startswith("r="):
                continue
            rw = c.get("r_within_pair_tl")
            out.append(f"| {key[2:]} ({'' if rw is None else f'{rw:.2f}'}) | {_f(c.get('tl'), 5)} "
                       f"({c.get('tl_cluster_se', 0):.5f}, {c.get('tl_draw_sd') or 0:.5f}) | {_f(c.get('mix'), 5)} "
                       f"| {_f(c.get('tl_worst_parent'), 5)} | {_f(c.get('r1b'), 5)} / {_f(c.get('r1p'), 5)} "
                       f"| {c.get('folds_on', '')} | {'pass' if c.get('pass', c.get('pass_if_on')) else 'no'} "
                       f"| {c.get('pass_draws', '')} |")
        out.append("smallest r: " + ", ".join(f"{k} {v}" for k, v in row.items() if k.startswith("smallest")))
    return "\n".join(out)


def stage_show(args):
    """Markdown tables of a --stage table result (or an --stage eval one)."""
    res = load_json(args.out if args.cov is None else args.cov)
    if "lines" in res:                                  # an eval result
        print(f"x: {res.get('x', {}).get('input', 'raw')}; rows {res.get('rows_lib_digest')}, "
              f"library {res.get('lib_digest')}")
        print(HEAD)
        for name, ln in res["lines"].items():
            print(line_row(name, ln))
        return
    acc = res["acceptance"]
    print(f"acceptance: {acc['check']}\nleak: {acc.get('leak')}")
    for key in ("oracle_in_sample", "oracle_honest", "oracle_honest_namefold"):
        if key in acc:
            print(f"\n{key}\n{HEAD}")
            for name, ln in acc[key].items():
                print(line_row(name, ln))
    for base, thr in res.get("thresholds", {}).items():
        print(f"\n{base} gate table\n" + gate_table(thr))
    for base, table in res.get("tables", {}).items():
        print(f"\n{base} degraded oracles (per-appearance differences averaged over draws)")
        for r, t in table.items():
            print(f"\n{r} (within-pair r on test-like: {t['meta']['r_within_pair_tl']:.3f})\n{HEAD}")
            for name, ln in t["lines"].items():
                print(line_row(name, ln, 5))
        print(f"\n{base}: test-like by budget, B0..B31, per line and r")
        for r, t in table.items():
            for name, ln in t["lines"].items():
                bb = ln["regimes"]["tl"]["by_budget"]
                print(f"{r:6s} {name:34s} " + " ".join(f"{v:+.5f}" for v in bb))
        print(f"\n{base}: test-like per held-out parent")
        for r, t in table.items():
            for name, ln in t["lines"].items():
                pp = ln["regimes"]["tl"]["per_parent"]
                print(f"{r:6s} {name:34s} " + " ".join(f"{q[:5]} {v['est']:+.5f} ({v['cluster_se']:.5f})"
                                                      for q, v in pp.items()))


# --- io -----------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return round(v, 7) if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("collect", "verify", "table", "eval", "show"))
    ap.add_argument("--regimes", nargs="+", default=list(PLAN), choices=list(PLAN))
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="collect runs 0..n-1 instead of PLAN's")
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--reps", type=int, default=N_REPS, help="noise draws per cell of the honest (gate) table")
    ap.add_argument("--reps-in-sample", type=int, default=5, help="noise draws per cell of the in-sample table")
    ap.add_argument("--resume", action="store_true", help="table: keep the parts of --out computed with the "
                    "same library, script, rows and grid")
    ap.add_argument("--legacy", default=None, help="verify: rows of an earlier library to compare against")
    ap.add_argument("--raw", action="store_true", help="eval: read x as given, not standardised within benchmark")
    ap.add_argument("--cov", default=None)
    ap.add_argument("--name", default="covariate")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.stage == "collect":
        stage_collect(args)
    elif args.stage == "verify":
        stage_verify(args)
    elif args.stage == "table":
        stage_table(args)
    elif args.stage == "show":
        stage_show(args)
    else:
        if not args.cov:
            ap.error("--stage eval needs --cov")
        stage_eval(args)


if __name__ == "__main__":
    main()
