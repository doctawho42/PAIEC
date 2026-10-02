"""P1.9 baselines (review W4 and P1.9; docs/report/review_v0.md): the organisers'
reference predictors, a plain 1PL, and how much of the shipped model's gain over
the legacy Predictor is the level calibration and how much the model.

Runs. Exactly the runs P1a scored (experiments/regime_sensitivity.py, scoring
seed 11): TUNED test-like runs 0-79 and public R1 runs 0-59 under both
weightings (R1B benchmark-first, R1P pair-uniform), split scope 'pair'. Every
config P1a already scored is read from its stored rows (data/
regime_sensitivity_rows/, the library this script runs, digests checked); only
the configs it lacks are scored here, on the same checkpoints, and P1a's rows
are checked to be reproduced on every run before a new row is written.

Configs (CONFIGS; the first six from P1a's rows, the rest scored here):
  ship       the shipped hier: LEVEL mu0 -2.5, sigma_mu 2.5, attr_scale 0.5
  legacy     the legacy Predictor (the first submission)
  smooth     the smoothed mean, Beta(2,2)
  smcal      the smoothed mean with P1a's calibrated prior, n0 2 and m0 0.25:
             (k + n0 m0) / (n + n0) on the pair's own labels
  onepl      hier at LEVEL with the attribute prior and the identity table off
  eb_fit     hier at the level prior.build fits (no LEVEL override)
  rasch      the plain 1PL: hier at LEVEL with the attribute prior, the identity
             table, the cross-benchmark link, the item_features group effects,
             the multiple-choice floor and the slip all off. What is left is
             logit p = mu_b + a_sb - d_i: one level per benchmark shared by its
             subjects (prior N(mu0, sigma_mu^2)), one ability per (subject,
             benchmark) pair around it (prior N(0, sigma_theta^2 + sigma_attr^2
             + sigma_delta^2)) and one difficulty per item (prior N(0, sigma_d^2
             + sigma_g^2), integrated out per item), refitted from `labeled` at
             every checkpoint like hier. It differs from onepl by
             RASCH_MINUS_ONEPL and nothing else
  rasch_fit  the same 1PL at its own leave-one-parent-out level: mu0 and
             sigma_mu as paiec.prior.fit_hyper derives them for a model
             without attributes (each training benchmark's fitted level plus
             its mean standing, no attribute score subtracted; their mean, and
             their sd widened as for hier), every other hyperparameter as
             prior.build fits it. The 1PL one would fit from public data, with
             no calibration to the hidden test
  empmean    the organisers' empirical mean (paiec.baselines.empirical_mean, a
             copy of third_party/paiec_baseline/empirical_mean), with the
             platform's random acquisition

BLE and the empirical mean with BLE acquisition cannot run here: the `ble`
stage reads third_party/paiec_baseline (never importing or copying it) and
records why, what they would need, and how many predictions these runs hold.

Statistics: P1a's (RegimeRows: level_calibration.Boot, 2,000 resamples, seed
0, cluster (parent, subject) on test-like runs and (benchmark, subject) on
public runs; ship_confirm.Block's per-budget, per-parent and parent-level
tables). "± a / b / c" is run / cluster / parent-stratified SE.

Decomposition (review W4, P1.9): ship minus legacy = (smcal minus legacy)
+ (ship minus smcal): the level calibration with the minimal model, then the
model at a calibrated level. Each part is a paired difference with its own
SEs; their sum is the total exactly. The split depends on the order, so the
other order is given too: (eb_fit minus legacy) + (ship minus eb_fit), the model
at its fitted level, then the calibration within hier; and the 1PL's own
calibration, rasch minus rasch_fit. A chain at calibrated levels, legacy ->
smcal -> rasch -> onepl -> ship, splits the model part further (descriptive).

Stages:
  ble        what BLE and its acquisition variant need -> OUT['ble']
  score      every planned run: the new configs, and the reproduction checks
             against P1a's rows; resumable, one row file per run in ROWS; exits
             with code 75 (resumable) past RSS_GUARD_GB
  summarise  statistics, decomposition and checks -> OUT
  status     rows scored so far

Run:
  python experiments/baselines_p1.py ble
  while :; do python experiments/baselines_p1.py score; [ $? -eq 75 ] || break; done
  python experiments/baselines_p1.py summarise

Output: results/baselines_p1.json (OUT); rows in data/baselines_p1_rows/ (ROWS,
gitignored). tests/test_baselines_p1.py covers the arithmetic on synthetic data.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.level_calibration as LC  # noqa: E402
import experiments.regime_sensitivity as RS  # noqa: E402
from paiec import baselines as BL  # noqa: E402
from paiec import prior as PR  # noqa: E402
from paiec import testlike as T  # noqa: E402
from paiec.evaluator import BUDGETS, WEIGHTS  # noqa: E402
from paiec.hier import HierPredictor  # noqa: E402

OUT = os.path.join(ROOT, "results", "baselines_p1.json")
ROWS = os.path.join(ROOT, "data", "baselines_p1_rows")
RS_ROWS = RS.ROWS
RS_JSON = RS.OUT
BASELINE_REPO = os.path.join(ROOT, "third_party", "paiec_baseline")

W6 = np.asarray(WEIGHTS, float)

# --- plan ---------------------------------------------------------------------------------

#: the P1a regimes scored here, with P1a's run counts and scoring seed
REGIMES = ("TUNED", "R1B", "R1P")
SEED = RS.SCORE_SEED
N_RUNS = {R: RS.N_RUNS[R] for R in REGIMES}
SPLIT_SCOPE = RS.SPLIT_SCOPE
#: P1a's calibrated smoothed mean (results/regime_sensitivity.json smcal.chosen)
SMCAL = (2.0, 0.25)
SHIP = "ship"
#: configs read from P1a's rows
STORED = ("ship", "legacy", "smooth", "smcal", "onepl", "eb_fit")
#: the plain 1PL's switches: hier's Flags with every component beyond a pooled level,
#: a pair ability and an item difficulty off
RASCH_FLAGS = {"attributes": False, "identity": False, "link": False, "groups": False,
               "floor": False, "slip": False}
#: what the plain 1PL turns off beyond P1a's onepl (attributes and identity off there)
RASCH_MINUS_ONEPL = {
    "link": "one ability per (subject, benchmark) pair; onepl keys theta on the canonical "
            "name, linking a subject's pairs across the run's benchmarks (weight about 0.16 "
            "without an attribute prior)",
    "groups": "no item_features group effects; their variance sigma_g^2 joins each item's "
              "residual, so an item's prior variance is the same",
    "floor": "no multiple-choice floor, in the likelihood or the prediction",
    "slip": "no slip (0.01) at prediction",
}
#: configs scored here: key -> how to build it
CONFIGS = {
    "rasch": {"kind": "hier", "level": "LEVEL", "flags": dict(RASCH_FLAGS)},
    "rasch_fit": {"kind": "hier", "level": "fit", "flags": dict(RASCH_FLAGS)},
    "empmean": {"kind": "empirical"},
}
NEW = tuple(CONFIGS)
ALL = STORED + NEW
#: reproduction of P1a's rows: smooth and smcal on every run, the rest on CHECK_RUNS
CHECK_EVERY = ("smooth", "smcal")
CHECK_SOME = ("ship", "onepl", "legacy")
CHECK_RUNS = {R: (0, N_RUNS[R] - 1) for R in REGIMES}
#: largest |Brier| difference accepted against a stored row (the same code and library
#: compute the same floats; the bound only absorbs a different summation order)
REPRO_TOL = 1e-9
RSS_GUARD_GB = 1.45
RESUME_EXIT = 75

#: the decomposition's paths: (name, [(step label, x, ref), ...]); each path's steps
#: telescope from its first ref to its last x
PATHS = {
    "calibration_first": [("level calibration (smcal - legacy)", "smcal", "legacy"),
                          ("model at a calibrated level (ship - smcal)", "ship", "smcal")],
    "model_first": [("model at its fitted level (eb_fit - legacy)", "eb_fit", "legacy"),
                    ("level calibration within hier (ship - eb_fit)", "ship", "eb_fit")],
    "rasch": [("the 1PL at its fitted level (rasch_fit - legacy)", "rasch_fit", "legacy"),
              ("level calibration within the 1PL (rasch - rasch_fit)", "rasch", "rasch_fit"),
              ("hier's extras at LEVEL (ship - rasch)", "ship", "rasch")],
    "chain": [("calibrated smoothed mean (smcal - legacy)", "smcal", "legacy"),
              ("item model and pooled level (rasch - smcal)", "rasch", "smcal"),
              ("link, groups, floor, slip (onepl - rasch)", "onepl", "rasch"),
              ("attribute prior and identity (ship - onepl)", "ship", "onepl")],
}
#: a share of the total is reported only where the total is this many cluster SEs from 0
SHARE_MIN_Z = 3.0

#: what was read in third_party/paiec_baseline on 2026-10-02 (commit BASELINE_COMMIT);
#: the ble stage re-reads the repository and records whether it still says so
BASELINE_COMMIT = "82d330ddcdb16016a3ae9e048db7e588ba2c6a39"
BLE_FACTS = {
    "predictor": "ble/model.py predict(): one agent run per prediction (ble/src/agent/agent.py "
                 "run_agent): an LLM estimates P(correct), calls retrieval tools over the public "
                 "measurement-db tables, revises, and submits; no parameter is fitted to the "
                 "response tables",
    "model_default": "openai/gpt-5.6-luna (ble/model.py CFG['llm']), Responses API, reasoning_effort "
                     "'medium'; other providers only through litellm, imported lazily",
    "credentials": "OPENAI_API_KEY (environment or ble/submission_config.json api_keys) for the "
                   "predictor, and for query embeddings when semantic retrieval is on",
    "data": "a payload prepared by tools/prepare_data.py from the Hugging Face datasets "
            "aims-foundations/measurement-db and aims-foundations/measurement-db-embed (network "
            "and an HF login); its keyword tools read the tables, the optional semantic tool "
            "the precomputed text-embedding-3-small vectors",
    "per_prediction": "up to max_steps turns (10 by default, 5 in the competition example "
                      "config), each LLM request capped at 120 s, a 240 s deadline per "
                      "prediction (question_timeout), 32,000 output tokens a turn retried at "
                      "64,000 and 128,000; probabilities clipped to [0.02, 0.98]; a failed or "
                      "unfinished run raises (no fallback)",
    "acquisition": "ble/labeling.py: request a candidate's label when BLE's prediction for it is "
                   "within 0.15 of 0.5, or when every remaining candidate is needed to fill the "
                   "budget: a BLE prediction per acquisition candidate on top of the evaluation "
                   "predictions",
    "mock_mode": "tools/smoke_test.py --mock replaces the LLM with scripted replies and the corpus "
                 "with small synthetic tables; it checks the plumbing and its probabilities are "
                 "illustrative (ble/README.md). There is no mode that predicts without an LLM, so "
                 "there is nothing of BLE to score offline",
    "mean_ble_acquisition": "empirical_mean_ble_acquisition/: evaluation predictions are the "
                            "unsmoothed empirical mean (0.5 without labels); only the acquisition "
                            "differs, by BLE's uncertainty policy, which needs a BLE prediction "
                            "for every candidate it does not force. Under the platform's random "
                            "acquisition the predictor is empmean, scored here; under BLE's "
                            "acquisition it is unmeasured",
}
BLE_WHY_NOT = (
    "BLE cannot run in this study. Every prediction is an LLM agent run (default "
    "openai/gpt-5.6-luna over the OpenAI Responses API), which needs network access, an "
    "OPENAI_API_KEY and a payload prepared from measurement-db and measurement-db-embed over "
    "the network. This lane allows no network and no language model, and holds no key. The "
    "only offline mode (tools/smoke_test.py --mock) answers from scripted replies on synthetic "
    "tables: it is a plumbing test, not a predictor. The empirical mean with BLE acquisition "
    "predicts the empirical mean but chooses its labels with BLE predictions, so it cannot "
    "run either; the empirical mean itself, with the platform's random acquisition, is scored "
    "here as empmean.")


# --- io -----------------------------------------------------------------------------------

clean = RS.clean
write_json = RS.write_json
file_sha = RS.file_sha
utcnow = RS.utcnow

#: files whose digests every row records: P1a's library list and the two scripts it ran
LIB_FILES = RS.LIB_FILES
SCRIPT_SHA = file_sha(os.path.abspath(__file__), 16)


def lib_digests():
    return {f: file_sha(os.path.join(ROOT, f), 16) for f in LIB_FILES}


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"head": git("rev-parse", "--short", "HEAD"),
            "dirty": git("status", "--porcelain", "paiec", "experiments", "submission", "tests"),
            "lib": lib_digests(), "script": SCRIPT_SHA,
            "regime_sensitivity_script": file_sha(os.path.join(ROOT, "experiments",
                                                               "regime_sensitivity.py"), 16),
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "utc": utcnow()}


def load_state(path=OUT):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_part(name, value, path=OUT):
    state = load_state(path)
    state[name] = clean(value)
    state["updated_utc"] = utcnow()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def plan_tasks():
    """Every planned (regime, run), run-index-major as P1a ordered them."""
    top = max(N_RUNS.values())
    return [(R, i) for i in range(top) for R in REGIMES if i < N_RUNS[R]]


def row_path(R, i, rows=ROWS):
    return os.path.join(rows, R, f"{i}.json")


def stored_row(R, i, rows=RS_ROWS):
    with open(RS.row_path(R, i, rows=rows)) as f:
        return json.load(f)


def same_meta(a, b):
    """Two run compositions as JSON values (tuples and lists alike)."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


# --- the plain 1PL's levels ---------------------------------------------------------------

def noattr_level(pairs, exclude, hyper):
    """(mu0, sigma_mu, report) of a model without attributes, as paiec.prior.fit_hyper
    derives them on the same pairs and exclusion as `hyper`: each included benchmark's
    fitted level plus its mean standing (fit_hyper subtracts the pool's mean attribute
    score only when its prior has attributes, and this one has none), their mean, and
    their sd widened for a new level. The prior passed is empty but records the
    exclusion, which fit_hyper checks."""
    bare = PR.SubjectPrior(meta={"excluded": list(hyper.excluded), "included": list(hyper.included)})
    h0, rep = PR.fit_hyper(pairs, exclude, prior=bare)
    if tuple(h0.included) != tuple(hyper.included) or tuple(h0.excluded) != tuple(hyper.excluded):
        raise ValueError("the 1PL level was fitted on other benchmarks than the bundle")
    return float(h0.mu0), float(h0.sigma_mu), {
        "levels": {b: float(v) for b, v in rep["levels"].items()},
        "estimated": [k for k in ("mu0", "sigma_mu") if k in rep["estimated"]]}


_LEVELS = {}


def fit_level(parent):
    """noattr_level for one parent, on every public pair outside it (cached)."""
    if parent not in _LEVELS:
        _, hyper = LC.bundle(parent, 0.0)
        mu0, sm, rep = noattr_level(LC.pairs(), (parent,), hyper)
        _LEVELS[parent] = {"mu0": mu0, "sigma_mu": sm, "hier_mu0": float(hyper.mu0),
                           "hier_sigma_mu": float(hyper.sigma_mu), **rep}
    return _LEVELS[parent]


def config_hyper(name, parent, hyper):
    """The Hyper a config of CONFIGS uses for one parent, from prior.build's."""
    c = CONFIGS[name]
    if c["level"] == "LEVEL":
        return replace(hyper, **RS.LEVEL)
    if c["level"] == "fit":
        lv = fit_level(parent)
        return replace(hyper, mu0=lv["mu0"], sigma_mu=lv["sigma_mu"])
    raise ValueError(c["level"])


# --- scoring --------------------------------------------------------------------------------

class Maker:
    """The model factory of one config of CONFIGS on one run, as regime_sensitivity.
    Maker builds P1a's: every call (one per checkpoint) builds fresh per-parent
    instances, dispatched on the anonymous benchmark_id, and the previous checkpoint's
    are released once their counters are read. Parents are built in sorted order."""

    def __init__(self, name, run):
        self.name, self.cfg = name, CONFIGS[name]
        self.ids = T.anon_parents(run)
        self.live = []
        self.failures = self.unconverged = 0

    def build(self, parent):
        prior, hyper = LC.bundle(parent, 0.0)
        return HierPredictor(prior, config_hyper(self.name, parent, hyper), **self.cfg["flags"])

    def harvest(self):
        for m in self.live:
            self.failures += int(getattr(m, "failures", 0))
            self.unconverged += int(getattr(m, "unconverged", 0))
        self.live = []

    def __call__(self):
        self.harvest()
        if self.cfg["kind"] == "empirical":
            return BL.empirical_mean
        ms = {par: self.build(par) for par in sorted(set(self.ids.values()))}
        self.live = list(ms.values())
        ids = self.ids
        return lambda inp, labeled=None: ms[ids[inp[1]["benchmark_id"]]].predict(inp, labeled)


def score_one(name, maker, slots, cps):
    """One config on the run's checkpoints, stored as P1a stores a config."""
    t0 = time.perf_counter()
    rows, tm = LC.evaluate(slots, cps, [name], maker)
    if hasattr(maker, "harvest"):
        maker.harvest()
    return {"b": [x["brier"] for x in rows[name]], "ece": [x["ece"] for x in rows[name]],
            "q0": [x["q0"] for x in rows[name]], "calls": tm["calls"], "mean_s": tm["mean_s"],
            "max_s": tm["max_s"], "failures": int(getattr(maker, "failures", 0)),
            "unconverged": int(getattr(maker, "unconverged", 0)),
            "secs": round(time.perf_counter() - t0, 3)}


def max_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def stage_score(rows_dir=ROWS, limit=None):
    st = load_state(RS_JSON)
    smcal = tuple(st["smcal"]["chosen"])
    if smcal != SMCAL:
        raise SystemExit(f"P1a's smcal is {smcal}, this plan's {SMCAL}")
    cpath = os.path.join(RS_ROWS, RS.COMPOSITIONS)
    if file_sha(cpath) != st["regimes"]["compositions"]["sha256"]:
        raise SystemExit(f"{cpath} differs from what P1a's regimes stage wrote")
    with open(cpath) as f:
        comps = json.load(f)["runs"]
    lib = lib_digests()
    todo = [t for t in plan_tasks() if not os.path.exists(row_path(*t, rows=rows_dir))]
    if limit is not None:
        todo = todo[:limit]
    print(f"score: {len(todo)} of {len(plan_tasks())} runs to do", flush=True)
    t_start = time.time()
    for j, (R, i) in enumerate(todo):
        t0 = time.time()
        old = stored_row(R, i)
        if old["lib"] != lib:
            raise SystemExit(f"{R} {i}: P1a's row was scored with another library: "
                             f"{[k for k in lib if old['lib'].get(k) != lib[k]]}")
        key, run = RS.draw(R, SEED, i)
        meta = RS.describe(key, run)
        if not (same_meta(meta, comps[R][str(i)]) and same_meta(meta, old["meta"])):
            raise RuntimeError(f"{R} run {i}: the drawn run differs from P1a's")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
            res = {}
            for n in NEW:
                res[n] = score_one(n, Maker(n, run), slots, cps)
                gc.collect()
            checks = {}
            names = CHECK_EVERY + (CHECK_SOME if i in CHECK_RUNS[R] else ())
            for n in names:
                got = score_one(n, RS.Maker(n, run, smcal), slots, cps)
                d = max_abs(got["b"], old["res"][n]["b"])
                checks[n] = {"max_abs_brier": d, "ok": bool(d <= REPRO_TOL), "secs": got["secs"]}
                gc.collect()
        levels = {par: {k: fit_level(par)[k] for k in ("mu0", "sigma_mu", "hier_mu0", "hier_sigma_mu")}
                  for par in sorted(set(T.anon_parents(run).values()))}
        rss = RS.rss_gb()
        row = {"regime": R, "i": i, "seed": SEED, "key": key, "meta": meta, "res": res,
               "checks": checks, "rasch_fit_levels": levels, "smcal": list(smcal),
               "task_s": round(time.time() - t0, 3), "rss_gb": round(rss, 3), "pid": os.getpid(),
               "lib": lib, "script": SCRIPT_SHA, "utc": utcnow()}
        if not all(c["ok"] for c in checks.values()):
            bad = {n: c["max_abs_brier"] for n, c in checks.items() if not c["ok"]}
            raise RuntimeError(f"{R} run {i}: P1a's rows not reproduced: {bad}")
        write_json(row_path(R, i, rows=rows_dir), row)
        del res, run, row, slots, cps
        gc.collect()
        rss = RS.rss_gb()
        print(f"  {R} {i}: {time.time() - t0:.1f}s, rss {rss:.2f} GB, checks "
              f"{sorted(checks)}, {j + 1}/{len(todo)} in {time.time() - t_start:.0f}s", flush=True)
        if rss > RSS_GUARD_GB and j + 1 < len(todo):
            print(f"  resident set {rss:.2f} GB > {RSS_GUARD_GB} GB: stopping (resumable)", flush=True)
            sys.exit(RESUME_EXIT)
    print("score: done", flush=True)


def stage_status(rows_dir=ROWS):
    for R in REGIMES:
        have = sum(os.path.exists(row_path(R, i, rows=rows_dir)) for i in range(N_RUNS[R]))
        print(f"{R:6s} {have:3d}/{N_RUNS[R]}")


# --- ble -------------------------------------------------------------------------------------

def literal_assign(path, name):
    """The literal value assigned to `name` at a module's top level, read with ast
    (the module is never imported), or None."""
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def git_head(repo):
    try:
        p = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
        return p.stdout.strip() or None
    except Exception:
        return None


def inspect_baseline(repo=BASELINE_REPO):
    """What the organisers' baseline repository says about BLE, read without importing
    or copying any of it: {present, commit, observed facts, checks against BLE_FACTS}."""
    if not os.path.isdir(repo):
        return {"present": False, "repo": os.path.relpath(repo, ROOT)}

    def read(rel):
        try:
            with open(os.path.join(repo, rel), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    cfg = literal_assign(os.path.join(repo, "ble", "model.py"), "CFG") or {}
    try:
        example = json.loads(read(os.path.join("ble", "submission_config.example.json")) or "{}")
    except ValueError:
        example = {}
    readme = read(os.path.join("ble", "README.md"))
    smoke = read(os.path.join("tools", "smoke_test.py"))
    client = read(os.path.join("ble", "src", "agent", "llm_client.py"))
    bounds = read(os.path.join("ble", "src", "agent", "belief_state.py"))
    mean_acq = read(os.path.join("empirical_mean_ble_acquisition", "labeling.py"))
    mean_model = read(os.path.join("empirical_mean_ble_acquisition", "model.py"))
    ble_lab = read(os.path.join("ble", "labeling.py"))
    observed = {
        "ble_default_llm": cfg.get("llm"),
        "ble_default_max_steps": cfg.get("max_steps"),
        "ble_question_timeout_s": cfg.get("question_timeout"),
        "ble_reasoning_effort": cfg.get("reasoning_effort"),
        "example_config": {k: v for k, v in example.items() if k != "api_keys"},
        "example_config_api_keys": sorted((example.get("api_keys") or {}).keys()),
        "llm_client_requests_openai": "post_json" in client and "litellm" in client,
        "probability_bounds": re.findall(r"^(M(?:IN|AX)_PROBABILITY)\s*=\s*([0-9.]+)", bounds, re.M),
        "mock_flag": "--mock" in smoke,
        "mock_is_scripted": "scripted" in readme.lower() and "illustrative" in readme.lower(),
        "readme_needs_openai_key": "OPENAI_API_KEY" in readme,
        "readme_payload_from_hf": "measurement-db-embed" in readme and "prepare_data.py" in readme,
        "acquisition_window": re.findall(r"abs\(prediction - 0\.5\) <= ([0-9.]+)", ble_lab),
        "mean_acquisition_calls_ble": "from ble.model import predict" in mean_acq,
        "mean_acquisition_predicts_empirical_mean": "from empirical_mean.model import predict" in mean_model,
    }
    checks = {
        "commit_as_read": git_head(repo) == BASELINE_COMMIT,
        "default_llm_is_luna": observed["ble_default_llm"] == "openai/gpt-5.6-luna",
        "needs_openai_key": observed["readme_needs_openai_key"]
        and observed["example_config_api_keys"] == ["OPENAI_API_KEY"],
        "payload_needs_network": observed["readme_payload_from_hf"],
        "offline_mode_is_mock_only": observed["mock_flag"] and observed["mock_is_scripted"],
        "acquisition_window_0.15": observed["acquisition_window"] == ["0.15"],
        "mean_variant_uses_ble_for_acquisition_only": observed["mean_acquisition_calls_ble"]
        and observed["mean_acquisition_predicts_empirical_mean"],
    }
    return {"present": True, "repo": os.path.relpath(repo, ROOT), "commit": git_head(repo),
            "commit_as_read": BASELINE_COMMIT, "observed": observed, "checks": checks,
            "all_checks_hold": all(checks.values())}


def prediction_counts(rows_dir=RS_ROWS):
    """Evaluation predictions on the planned runs, from P1a's stored call counts (every
    config makes the same calls; 'ship' is read): what BLE would have to answer, each an
    agent run, before any acquisition prediction."""
    out, total = {}, 0
    for R in REGIMES:
        n = 0
        for i in range(N_RUNS[R]):
            p = RS.row_path(R, i, rows=rows_dir)
            if not os.path.exists(p):
                return None
            with open(p) as f:
                n += int(json.load(f)["res"][SHIP]["calls"])
        out[R] = {"runs": N_RUNS[R], "predictions": n, "per_run": n / N_RUNS[R]}
        total += n
    out["total"] = total
    return out


def stage_ble():
    t0 = time.time()
    res = {"runnable": False, "why_not": BLE_WHY_NOT, "facts": BLE_FACTS,
           "repository": inspect_baseline(), "evaluation_predictions": prediction_counts(),
           "would_need": [
               "network access to the OpenAI Responses API and an OPENAI_API_KEY with access to "
               "openai/gpt-5.6-luna (or another provider through litellm, not installed here)",
               "a payload from tools/prepare_data.py: measurement-db and measurement-db-embed "
               "downloaded from Hugging Face (network, HF login), restricted to the training "
               "benchmarks of each run's held-out parent to stay leave-one-parent-out (the "
               "tool downloads every eligible public benchmark, so this needs a filtered payload)",
               "one agent run of up to 5 to 10 LLM turns (240 s deadline) for each evaluation "
               "prediction counted in evaluation_predictions, plus one per acquisition "
               "candidate under BLE's own acquisition",
               "a fixed seed is not available: BLE runs are stochastic, so paired comparisons "
               "would need its predictions recorded once and replayed"],
           "wall_s": round(time.time() - t0, 2), "provenance": provenance()}
    save_part("ble", res)
    rep = res["repository"]
    print(f"ble: not runnable here; repository present {rep['present']}"
          + (f", checks {'hold' if rep.get('all_checks_hold') else 'DIFFER'}" if rep["present"] else ""),
          flush=True)


# --- statistics -------------------------------------------------------------------------------

class Rows(RS.RegimeRows):
    """P1a's RegimeRows on runs that hold the configs it scored and the ones scored
    here: its statistics (diff, table, config_summary) for every config of ALL."""

    def __init__(self, R, rows, extra=NEW):
        super().__init__(R, rows)
        for c in extra:
            self.B[c] = [np.asarray(rows[i]["res"][c]["b"], float) for i in self.ids]
            self.A[c] = [b @ W6 for b in self.B[c]]

    def boot_dist(self, x, ref):
        """The cluster bootstrap's resampled values of D(x, ref) (the ratio estimator
        RegimeRows.diff takes its SE from), on its fixed resamples."""
        flat = np.concatenate([a - b for a, b in zip(self.A[x], self.A[ref])])
        v = np.bincount(self.cl, self.w * flat, minlength=self.K)
        n = np.bincount(self.cl, self.w, minlength=self.K)
        return (self.boot.W @ v) / (self.boot.W @ n)


def merge(rs_rows, new_rows):
    """{run: P1a's row with the configs scored here added to its res}; the two must hold
    the same runs with the same compositions, and the new rows every config of NEW."""
    if sorted(rs_rows) != sorted(new_rows):
        raise ValueError(f"runs differ: P1a {sorted(rs_rows)[:5]}..., here {sorted(new_rows)[:5]}...")
    out = {}
    for i, r in rs_rows.items():
        n = new_rows[i]
        if not same_meta(r["meta"], n["meta"]):
            raise ValueError(f"run {i}: compositions differ")
        missing = [c for c in STORED if c not in r["res"]] + [c for c in NEW if c not in n["res"]]
        if missing:
            raise ValueError(f"run {i}: no rows for {missing}")
        out[i] = {**r, "res": {**r["res"], **{c: n["res"][c] for c in NEW}}}
    return out


def part(G, x, ref):
    """One paired difference x minus ref: P1a's exact quantities (D, run, cluster and
    stratified SEs, bootstrap percentiles, per-parent means, parent-level mean) and
    ship_confirm.Block's table (per budget, per parent, parent-level mean and SE)."""
    ex = G.diff(x, ref)
    tb = G.table(x, ref)
    return {"x": x, "ref": ref, "D": ex["D"], "run_se": ex["run_se"], "cluster_se": ex["cluster_se"],
            "strat_se": ex["strat_se"], "pct95": ex["pct"], "PL": ex["PL"],
            "parent_level": tb["parent_level"], "per_parent": tb["per_parent"],
            "budgets": tb.get("budgets"), "x_ALC": tb["x_ALC"], "ref_ALC": tb["ref_ALC"],
            "share_runs_x_better": ex["share_runs_x_better"]}


def path_summary(G, steps):
    """A path's steps, each a paired difference, and its total (last x minus first ref):
    the steps' D sum to the total's exactly; each step's share of the total, with the
    cluster bootstrap's 95% interval on the same resamples, where the total is at least
    SHARE_MIN_Z cluster SEs from 0."""
    first, last = steps[0][2], steps[-1][1]
    total = part(G, last, first)
    tdist = G.boot_dist(last, first)
    z = total["D"] / total["cluster_se"] if total["cluster_se"] else float("inf")
    shareable = abs(z) >= SHARE_MIN_Z
    out = []
    for label, x, ref in steps:
        p = part(G, x, ref)
        p["label"] = label
        if shareable:
            s = G.boot_dist(x, ref) / tdist
            p["share_of_total"] = {"point": p["D"] / total["D"],
                                   "pct95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}
        else:
            p["share_of_total"] = None
        out.append(p)
    gap = abs(sum(p["D"] for p in out) - total["D"])
    return {"total": {"label": f"{last} - {first}", **total, "z_cluster": z},
            "steps": out, "shares_reported": shareable,
            "shares_note": None if shareable else
            f"the total is {z:+.1f} cluster SEs from 0, under {SHARE_MIN_Z:g}: no share is given",
            "sum_of_steps_minus_total": gap}


def budget_view(p, budgets=(0, 1)):
    """The step's differences at the named budgets, with run / cluster / stratified SEs and
    the ALC weight times the difference (alc_part)."""
    if not p.get("budgets"):
        return None
    return {f"B{b['budget']}": {k: b[k] for k in ("x", "ref", "diff", "run_se", "cluster_se", "strat_se",
                                                  "alc_part")}
            for b in p["budgets"] if b["budget"] in budgets}


def load_rows(R, rows_dir=ROWS, rs_rows_dir=RS_ROWS):
    new, old = {}, {}
    for i in range(N_RUNS[R]):
        p = row_path(R, i, rows=rows_dir)
        if os.path.exists(p):
            with open(p) as f:
                new[i] = json.load(f)
            old[i] = stored_row(R, i, rows=rs_rows_dir)
    return old, new


def brief(p, label=None):
    """A difference's headline numbers: D, the three SEs and the parent-level mean and SE."""
    pl = p.get("parent_level") or {}
    return {"label": label or p.get("label") or f"{p['x']} - {p['ref']}", "D": p["D"],
            "run_se": p["run_se"], "cluster_se": p["cluster_se"], "strat_se": p["strat_se"],
            "parent_level_mean": pl.get("mean"), "parent_level_se": pl.get("se"),
            "share_of_total": p.get("share_of_total")}


def headline(paths, vs_ship):
    """Per regime, the numbers the review asks for, read off the paths and vs_ship."""
    cal, mod = paths["calibration_first"], paths["model_first"]
    return {"ship_minus_legacy": brief(cal["total"], "ship - legacy"),
            "calibration_first": [brief(q) for q in cal["steps"]],
            "model_first": [brief(q) for q in mod["steps"]],
            "rasch_path": [brief(q) for q in paths["rasch"]["steps"]],
            "chain": [brief(q) for q in paths["chain"]["steps"]],
            "vs_ship": {c: brief(vs_ship[c], f"{c} - ship") for c in vs_ship}}


def se3(p):
    """'run / cluster / stratified' SEs at four decimals ('-' where undefined)."""
    return " / ".join("-" if p.get(k) is None or not math.isfinite(p[k]) else f"{p[k]:.4f}"
                      for k in ("run_se", "cluster_se", "strat_se"))


def stage_summarise(rows_dir=ROWS, out=OUT):
    t0 = time.time()
    st = load_state(RS_JSON)
    rs_sum = st["summary"]
    checks, summary = {}, {"regimes": {}}
    lib_now = lib_digests()
    all_levels = {}
    have = {}
    for R in REGIMES:
        old, new = load_rows(R, rows_dir)
        have[R] = len(new)
        if not new:
            continue
        rows = merge(old, new)
        G = Rows(R, rows)
        libs = {json.dumps(r["lib"], sort_keys=True) for r in new.values()} | \
               {json.dumps(r["lib"], sort_keys=True) for r in old.values()}
        checks[f"{R}: one library for P1a's rows and these"] = len(libs) == 1
        checks[f"{R}: that library is the current one"] = libs == {json.dumps(lib_now, sort_keys=True)}
        rep = [c for r in new.values() for c in r["checks"].values()]
        checks[f"{R}: P1a's rows reproduced on every run"] = all(c["ok"] for c in rep) and all(
            set(CHECK_EVERY) <= set(r["checks"]) for r in new.values())
        checks[f"{R}: ship, onepl, legacy reproduced on runs {list(CHECK_RUNS[R])}"] = all(
            i in new and set(CHECK_SOME) <= set(new[i]["checks"]) for i in CHECK_RUNS[R])
        scripts = sorted({r["script"] for r in new.values()})
        checks[f"{R}: every row scored by this script"] = scripts == [SCRIPT_SHA]
        repro = {n: max([r["checks"][n]["max_abs_brier"] for r in new.values() if n in r["checks"]],
                        default=None) for n in CHECK_EVERY + CHECK_SOME}
        for r in new.values():
            for par, lv in r["rasch_fit_levels"].items():
                prev = all_levels.setdefault(par, lv)
                if prev != lv:
                    checks[f"rasch_fit level of {par} is one value"] = False
        # P1a's own numbers for its configs, recomputed from the merged rows (on the full
        # planned set only: P1a's summary is over every run)
        complete = len(new) == N_RUNS[R]
        agree = None
        if complete:
            agree = 0.0
            for c in STORED:
                if c == SHIP:
                    continue
                mine, theirs = G.diff(c, SHIP), rs_sum["vs_ship"][R][c]["exact"]
                agree = max(agree, *(abs(mine[k] - theirs[k]) for k in ("D", "run_se", "cluster_se", "strat_se")))
            checks[f"{R}: P1a's vs_ship numbers recomputed"] = agree <= 1e-12
        configs = {c: G.config_summary(c) for c in ALL}
        for c in ALL:
            configs[c]["mean_call_ms"] = 1e3 * float(np.mean([rows[i]["res"][c]["mean_s"] for i in G.ids]))
            configs[c]["max_call_s"] = float(max(rows[i]["res"][c]["max_s"] for i in G.ids))
        vs_ship = {c: part(G, c, SHIP) for c in ALL if c != SHIP}
        vs_legacy = {c: part(G, c, "legacy") for c in NEW}
        paths = {k: path_summary(G, steps) for k, steps in PATHS.items()}
        b01 = {}
        for k, steps in PATHS.items():
            for label, x, ref in steps:
                b01[f"{x} - {ref}"] = budget_view(part(G, x, ref))
        for x, ref in (("ship", "rasch"), ("ship", "rasch_fit"), ("onepl", "rasch"), ("rasch", "smooth")):
            b01[f"{x} - {ref}"] = budget_view(part(G, x, ref))
        alc = {c: configs[c]["ALC"] for c in ALL}
        summary["regimes"][R] = {
            "runs": len(new), "complete": complete, "pairs": int(len(G.w)), "clusters": G.K,
            "realised": G.level(), "configs": configs,
            "rank": sorted(ALL, key=alc.get),
            "vs_ship": vs_ship, "vs_legacy": vs_legacy,
            "headline": headline(paths, vs_ship),
            "decomposition": paths, "budgets_0_1": b01,
            "reproduction_max_abs_brier": repro, "p1a_agreement_max_abs": agree,
            "rows": {"script_digests": scripts,
                     "task_s": float(np.mean([r["task_s"] for r in new.values()])),
                     "max_rss_gb": max(r["rss_gb"] for r in new.values()),
                     "written_utc": [min(r["utc"] for r in new.values()), max(r["utc"] for r in new.values())]}}
        del G, rows
        gc.collect()
    summary["rasch_fit_levels"] = all_levels
    summary["scored"] = have
    complete = all(have[R] == N_RUNS[R] for R in REGIMES)
    summary["complete"] = complete
    checks["every planned run scored"] = complete
    checks["smcal is P1a's"] = tuple(st["smcal"]["chosen"]) == SMCAL
    checks["P1a's lock holds"] = all(RS.lock_checks().values())
    ble = load_state(out).get("ble")
    checks["ble stage run"] = ble is not None
    result = {
        "about": "P1.9 baselines (review W4, P1.9): the organisers' reference predictors, a plain "
                 "1PL, and the split of the shipped model's gain over the legacy Predictor into "
                 "level calibration and model, on P1a's runs (seed 11): TUNED 0-79, R1B and R1P "
                 "0-59, split scope 'pair'. Differences are x minus ref, paired on identical runs; "
                 "SEs are run / cluster (parent or benchmark, subject) / parent-stratified, "
                 "level_calibration.Boot with 2,000 resamples, seed 0.",
        "commands": ["python experiments/baselines_p1.py ble",
                     "python experiments/baselines_p1.py score   (repeat while it exits with 75)",
                     "python experiments/baselines_p1.py summarise"],
        "plan": {"regimes": {R: {"seed": SEED, "runs": [0, N_RUNS[R] - 1],
                                 "knobs": RS.REGIMES[R][1]} for R in REGIMES},
                 "split_scope": SPLIT_SCOPE,
                 "stored_configs": {c: RS.CONFIGS[c] for c in STORED},
                 "smcal": {"n0": SMCAL[0], "m0": SMCAL[1], "source": "results/regime_sensitivity.json smcal.chosen"},
                 "new_configs": CONFIGS, "rasch_minus_onepl": RASCH_MINUS_ONEPL,
                 "paths": {k: [list(s) for s in v] for k, v in PATHS.items()},
                 "share_min_z": SHARE_MIN_Z,
                 "reproduction": {"every_run": list(CHECK_EVERY), "runs": {R: list(v) for R, v in CHECK_RUNS.items()},
                                  "configs": list(CHECK_SOME), "tolerance": REPRO_TOL}},
        "checks": checks, "all_checks_hold": all(checks.values()),
        "summary": summary, "wall_s": round(time.time() - t0, 1), "provenance": provenance()}
    for k, v in result.items():
        save_part(k, v, out)
    print(f"summarise: {have} runs; checks {'hold' if result['all_checks_hold'] else 'FAIL'}: "
          f"{[k for k, v in checks.items() if not v]}", flush=True)
    for R, s in summary["regimes"].items():
        c = s["configs"]
        print(f"  {R}: " + ", ".join(f"{k} {c[k]['ALC']:.4f}" for k in s["rank"]))
        for k, p in s["decomposition"].items():
            print(f"    {k}: total {p['total']['D']:+.4f} ± {se3(p['total'])}; " + "; ".join(
                f"{q['x']}-{q['ref']} {q['D']:+.4f} ± {se3(q)}" for q in p["steps"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("ble", "score", "summarise", "status"))
    ap.add_argument("--rows", default=ROWS, help="row directory (default %(default)s)")
    ap.add_argument("--out", default=OUT, help="output file of summarise (default %(default)s)")
    ap.add_argument("--limit", type=int, default=None, help="score: at most this many runs")
    a = ap.parse_args(argv)
    if a.stage == "ble":
        stage_ble()
    elif a.stage == "score":
        stage_score(a.rows, a.limit)
    elif a.stage == "summarise":
        stage_summarise(a.rows, a.out)
    else:
        stage_status(a.rows)


if __name__ == "__main__":
    main()
