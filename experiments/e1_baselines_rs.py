"""E1 (docs/plans/p2_final_plan.md, amendment 2; report only, no-ship): EmpMean and
1PL-ship in the four regime-sensitivity regimes where the report's comparator table
lacks them (READING, AUDIT, MIXTURE, FLAT), on the same runs and checkpoints as the
other comparators, paired with ship, with P1a's D and cluster SE.

What is scored. experiments/baselines_p1.py scored the organisers' empirical mean
(`empmean`, EmpMean) and the plain 1PL at the shipped LEVEL (`rasch`, 1PL-ship) on
P1a's TUNED, R1B and R1P runs only. This script scores the same two configs, built by
baselines_p1's own factory (baselines_p1.Maker, baselines_p1.CONFIGS, baselines_p1.
score_one, imported, never edited), on P1a's other four test-like regimes: scoring
seed 11, runs 0-79 each, split scope 'pair', each run drawn by regime_sensitivity.draw
and checked against P1a's compositions file and its stored row, its checkpoints made by
level_calibration.checkpoints (the platform's random acquisition). ship and every
other comparator are read from P1a's stored rows (data/regime_sensitivity_rows/),
never re-scored; the library must be the one those rows record.

Checks while scoring (every run; a failed check stops the run, nothing is written):
  - P1a's rows are reproduced as baselines_p1 reproduced them: smooth and smcal on
    every run, ship, onepl and legacy on the first and last run of each regime,
    |dBrier| <= baselines_p1.REPRO_TOL;
  - the control: TUNED runs 0-79 are scored again here, and the two configs' rows
    must equal baselines_p1's stored rows (data/baselines_p1_rows/TUNED/) bit for bit
    (Brier, ECE, mean B0 prediction, call count, fallbacks); the summary then checks
    that this script's statistics on its own TUNED rows give baselines_p1's stored
    numbers (results/baselines_p1.json) exactly.

Statistics: baselines_p1's (its Rows, an extension of P1a's RegimeRows: level_
calibration.Boot, 2,000 resamples, bootstrap seed 0, ratio estimator, cluster (parent,
subject); and its part(): D, run / cluster / parent-stratified SE, bootstrap
percentiles, per-parent and parent-level tables, per budget). D is comparator minus
ship, the mean over runs of the run's pair-mean ALC difference: positive when hier-ship
is better, as in the report's T3.

Pinned before scoring: the plan's E1 bullet (E1_BULLET_SHA256, and that it is in the
plan as committed at AMENDMENT_COMMIT, whose commit time must precede every row), and
this file's text from "# --- plan" to "# --- statistics" (rows record its digest,
SCORING_SHA; the summary refuses rows scored by other scoring code). Nothing here is
shipped or selects anything; P1a's rule is not applied.

Stages:
  lock       the pins above, P1a's lock, P1a's smcal and compositions file
  score      every planned (regime, run) task, --shard k/n, resumable, one row file a
             task under --rows; exits 75 (resumable) past baselines_p1.RSS_GUARD_GB
  summarise  statistics, the control's agreement, the T3 cells -> OUT; exits non-zero,
             writing nothing, if a planned row is missing (unless --allow-missing)
  status     rows scored so far

Run (from the repository root):
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export PYTHONHASHSEED=0
  python experiments/e1_baselines_rs.py lock
  for k in 0 1 2; do (while :; do python experiments/e1_baselines_rs.py score \\
      --shard $k/3; [ $? -eq 75 ] || break; done) & done; wait
  python experiments/e1_baselines_rs.py summarise
400 tasks (320 new, 80 control): about 2.3 s a task once a process has fitted its prior
bundles (30 to 50 s a process); on 2026-10-06 the three shards took 4.5 minutes of wall
time under heavy outside load (load average 55 to 72 on 8 cores), and summarise 3 s. On
2026-10-07 the rows, lost with the session scratch, were scored again in four shards (6.4
minutes of wall time at a load average near 8) and reproduce the first rows' scored values
bit for bit (the results' `about` and `rows` blocks).

Output: results/e1_baselines_rs.json (OUT); rows in data/e1_baselines_rs_rows/ (ROWS,
gitignored) or --rows. summarise exits non-zero and writes nothing when a planned row is
missing, unless --allow-missing (a partial summary, all_checks_hold false). tests/test_e1_baselines_rs.py checks the plan against
baselines_p1's and P1a's, the statistics on synthetic rows, and, where the data and the
stored rows are on disk, that this script reproduces baselines_p1's TUNED numbers for
both configs bit for bit.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.baselines_p1 as BP  # noqa: E402
import experiments.level_calibration as LC  # noqa: E402
import experiments.regime_sensitivity as RS  # noqa: E402

OUT = os.path.join(ROOT, "results", "e1_baselines_rs.json")
ROWS = os.path.join(ROOT, "data", "e1_baselines_rs_rows")
SCRIPT_PATH = os.path.abspath(__file__)
PLAN_MARK = "# --- plan"
STATS_MARK = "# --- statistics"

# --- plan ---------------------------------------------------------------------------------

#: the plan that declares E1, the commit that added the declaration, and the sha256 of the
#: E1 bullet of its amendment 2 (from "- E1 (report only" through "selects anything.\n")
PLAN_PATH = "docs/plans/p2_final_plan.md"
AMENDMENT_COMMIT = "a5ef016"
E1_BULLET_START = "- E1 (report only, no-ship)"
E1_BULLET_SHA256 = "b709c8494cdfa7b39233191f8b3036ec976a669cd27ff65e9e668f6f9dbffab7"

#: the regimes E1 fills, and the control regime re-scored to reproduce baselines_p1
REGIMES = ("READING", "AUDIT", "MIXTURE", "FLAT")
CONTROL = ("TUNED",)
SCORED = CONTROL + REGIMES
SEED = RS.SCORE_SEED
N_RUNS = {R: RS.N_RUNS[R] for R in SCORED}
SPLIT_SCOPE = RS.SPLIT_SCOPE
SMCAL = BP.SMCAL
SHIP = BP.SHIP
#: the two comparators, as baselines_p1 defines and builds them
NEW = ("empmean", "rasch")
CONFIGS = {c: BP.CONFIGS[c] for c in NEW}
LABELS = {"empmean": "EmpMean, the organisers' empirical mean",
          "rasch": "1PL-ship, a plain 1PL at hier-ship's level"}
#: configs read from P1a's rows (baselines_p1's STORED)
STORED = BP.STORED
ALL = STORED + NEW
#: reproduction of P1a's rows, as baselines_p1 checked them
CHECK_EVERY = BP.CHECK_EVERY
CHECK_SOME = BP.CHECK_SOME
CHECK_RUNS = {R: (0, N_RUNS[R] - 1) for R in SCORED}
REPRO_TOL = BP.REPRO_TOL
#: the fields of a scored config that must equal baselines_p1's row on the control
EXACT_FIELDS = ("b", "ece", "q0", "calls", "failures", "unconverged")
#: T3's column order; R1B and R1P are read from baselines_p1's results, not scored here
T3_COLUMNS = ("TUNED", "AUDIT", "MIXTURE", "READING", "FLAT", "R1B", "R1P")
RSS_GUARD_GB = BP.RSS_GUARD_GB
RESUME_EXIT = BP.RESUME_EXIT


# --- io -----------------------------------------------------------------------------------

clean = RS.clean
write_json = RS.write_json
file_sha = RS.file_sha
same_meta = BP.same_meta
SCRIPT_SHA = file_sha(SCRIPT_PATH, 16)


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scoring_block(path=SCRIPT_PATH):
    """This file's text from PLAN_MARK through the line before STATS_MARK: the plan
    and the scoring code every row records the digest of."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    a = src.index("\n" + PLAN_MARK) + 1
    b = src.index("\n" + STATS_MARK, a) + 1
    return src[a:b]


SCORING_SHA = hashlib.sha256(scoring_block().encode("utf-8")).hexdigest()


def git(*a):
    try:
        p = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def e1_bullet(text):
    """The plan's E1 bullet: from E1_BULLET_START through the line before the next
    top-level bullet ("\n- "), newline included; None when absent."""
    if E1_BULLET_START not in text:
        return None
    a = text.index(E1_BULLET_START)
    b = text.find("\n- ", a + 1)
    return text[a:] if b < 0 else text[a:b + 1]


def commit_utc(rev):
    t = git("log", "-1", "--format=%cI", rev)
    if not t:
        return None
    return datetime.fromisoformat(t).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_checks():
    """The pins: the E1 bullet in the working plan and in the plan at AMENDMENT_COMMIT,
    absent from its parent; P1a's lock."""
    with open(os.path.join(ROOT, PLAN_PATH), encoding="utf-8") as f:
        now = e1_bullet(f.read())
    then = e1_bullet(git("show", f"{AMENDMENT_COMMIT}:{PLAN_PATH}") or "")
    before = e1_bullet(git("show", f"{AMENDMENT_COMMIT}^:{PLAN_PATH}") or "")
    sha = (lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest() if s else None)
    out = {"e1_bullet_sha256": sha(now) == E1_BULLET_SHA256,
           "e1_bullet_at_amendment_commit": sha(then) == E1_BULLET_SHA256,
           "e1_bullet_absent_before_it": before is None and git("rev-parse", f"{AMENDMENT_COMMIT}^") is not None,
           "p1a_lock": all(RS.lock_checks().values())}
    return out


def require_lock():
    res = lock_checks()
    if not all(res.values()):
        raise SystemExit(f"lock check failed: {[k for k, v in res.items() if not v]}")
    return res


def inputs():
    """The committed files the stages read, with their sha256."""
    paths = [PLAN_PATH, "results/regime_sensitivity.json", "results/baselines_p1.json",
             "experiments/regime_sensitivity.py", "experiments/baselines_p1.py",
             "experiments/level_calibration.py", "experiments/ship_confirm.py",
             "submission/model.py"]
    out = {p: file_sha(os.path.join(ROOT, p)) for p in paths if os.path.exists(os.path.join(ROOT, p))}
    cpath = os.path.join(BP.RS_ROWS, RS.COMPOSITIONS)
    if os.path.exists(cpath):
        out[os.path.relpath(cpath, ROOT)] = file_sha(cpath)
    return out


def provenance(t0=None):
    cmd = None
    if sys.argv and sys.argv[0].endswith(".py"):
        cmd = "python " + " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), ROOT)] + sys.argv[1:])
    return {"script": os.path.relpath(SCRIPT_PATH, ROOT), "script_sha256": file_sha(SCRIPT_PATH),
            "scoring_block_sha256": SCORING_SHA,
            "head": git("rev-parse", "--short", "HEAD"),
            "dirty": git("status", "--porcelain", "paiec", "experiments", "submission", "tests", "results"),
            "lib": BP.lib_digests(), "inputs": inputs(), "command": cmd,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "threads": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "utc": utcnow(),
            "wall_s": None if t0 is None else round(time.time() - t0, 1)}


def plan_tasks():
    """Every planned (regime, run), run-index-major, regimes in SCORED order."""
    top = max(N_RUNS.values())
    return [(R, i) for i in range(top) for R in SCORED if i < N_RUNS[R]]


def shard_tasks(k, n):
    return [t for j, t in enumerate(plan_tasks()) if j % n == k]


def row_path(R, i, rows=ROWS):
    return os.path.join(rows, R, f"{i}.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --- scoring --------------------------------------------------------------------------------

def p1a_state():
    """P1a's results, checked as baselines_p1's score stage checks them: its smcal is
    baselines_p1's, its compositions file is the one its regimes stage wrote."""
    st = RS.load_state()
    if tuple(st["smcal"]["chosen"]) != SMCAL:
        raise SystemExit(f"P1a's smcal is {tuple(st['smcal']['chosen'])}, baselines_p1's {SMCAL}")
    cpath = os.path.join(BP.RS_ROWS, RS.COMPOSITIONS)
    if file_sha(cpath) != st["regimes"]["compositions"]["sha256"]:
        raise SystemExit(f"{cpath} differs from what P1a's regimes stage wrote")
    return st, load_json(cpath)["runs"]


def exact_diff(got, want, fields=EXACT_FIELDS):
    """The fields of a scored config where `got` is not `want` as JSON values (so
    floats must be equal, not close), and the largest |dBrier|."""
    g, w = json.loads(json.dumps(clean(got))), json.loads(json.dumps(clean(want)))
    bad = [k for k in fields if g.get(k) != w.get(k)]
    return bad, BP.max_abs(g.get("b", []), w.get("b", []))


def score_task(R, i, comps, lib):
    """One planned run: the two configs on its checkpoints, the reproduction checks
    against P1a's stored row and, on the control, against baselines_p1's. -> row."""
    t0 = time.time()
    old = BP.stored_row(R, i)
    if old["lib"] != lib:
        raise SystemExit(f"{R} {i}: P1a's row was scored with another library: "
                         f"{[k for k in lib if old['lib'].get(k) != lib[k]]}")
    key, run = RS.draw(R, SEED, i)
    meta = RS.describe(key, run)
    if not (same_meta(meta, comps[R][str(i)]) and same_meta(meta, old["meta"])):
        raise RuntimeError(f"{R} run {i}: the drawn run differs from P1a's")
    bp = None
    if R in CONTROL:
        bp = load_json(BP.row_path(R, i))
        if not same_meta(meta, bp["meta"]) or bp["lib"] != lib:
            raise RuntimeError(f"{R} run {i}: baselines_p1's row is of another run or library")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slots, cps = LC.checkpoints(run, scope=SPLIT_SCOPE)
        res = {}
        for n in NEW:
            res[n] = BP.score_one(n, BP.Maker(n, run), slots, cps)
            gc.collect()
        checks = {}
        names = CHECK_EVERY + (CHECK_SOME if i in CHECK_RUNS[R] else ())
        for n in names:
            got = BP.score_one(n, RS.Maker(n, run, SMCAL), slots, cps)
            d = BP.max_abs(got["b"], old["res"][n]["b"])
            checks[n] = {"max_abs_brier": d, "ok": bool(d <= REPRO_TOL), "secs": got["secs"]}
            gc.collect()
    if bp is not None:
        for n in NEW:
            bad, d = exact_diff(res[n], bp["res"][n])
            checks[f"baselines_p1:{n}"] = {"fields_differing": bad, "max_abs_brier": d, "ok": not bad}
    return {"regime": R, "i": i, "seed": SEED, "key": key, "meta": meta, "res": res,
            "checks": checks, "smcal": list(SMCAL), "task_s": round(time.time() - t0, 3),
            "pid": os.getpid(), "lib": lib, "script": SCRIPT_SHA, "scoring_sha256": SCORING_SHA,
            "utc": utcnow()}


def stage_score(shard=(0, 1), rows_dir=ROWS, limit=None):
    require_lock()
    _, comps = p1a_state()
    lib = BP.lib_digests()
    k, n = shard
    todo = [t for t in shard_tasks(k, n) if not os.path.exists(row_path(*t, rows=rows_dir))]
    if limit is not None:
        todo = todo[:limit]
    print(f"score {k}/{n}: {len(todo)} of {len(shard_tasks(k, n))} tasks to do", flush=True)
    t_start = time.time()
    for j, (R, i) in enumerate(todo):
        row = score_task(R, i, comps, lib)
        rss = RS.rss_gb()
        row["rss_gb"], row["shard"] = round(rss, 3), f"{k}/{n}"
        if not all(c["ok"] for c in row["checks"].values()):
            bad = {c: v for c, v in row["checks"].items() if not v["ok"]}
            raise RuntimeError(f"{R} run {i}: a reproduction check failed: {bad}")
        write_json(row_path(R, i, rows=rows_dir), row)
        print(f"  [{k}/{n}] {R} {i}: {row['task_s']:.1f}s, rss {rss:.2f} GB, checks "
              f"{sorted(row['checks'])}, {j + 1}/{len(todo)} in {time.time() - t_start:.0f}s", flush=True)
        del row
        gc.collect()
        if RS.rss_gb() > RSS_GUARD_GB and j + 1 < len(todo):
            print(f"  [{k}/{n}] resident set over {RSS_GUARD_GB} GB: stopping (resumable)", flush=True)
            sys.exit(RESUME_EXIT)
    print(f"score {k}/{n}: done", flush=True)


# --- statistics -----------------------------------------------------------------------------

def merge(rs_rows, new_rows, new=NEW):
    """baselines_p1.merge for the configs of `new`: {run: P1a's row with them added to its
    res}; the two must hold the same runs with the same compositions."""
    if sorted(rs_rows) != sorted(new_rows):
        raise ValueError(f"runs differ: P1a {sorted(rs_rows)[:5]}..., here {sorted(new_rows)[:5]}...")
    out = {}
    for i, r in rs_rows.items():
        n = new_rows[i]
        if not same_meta(r["meta"], n["meta"]):
            raise ValueError(f"run {i}: compositions differ")
        missing = [c for c in STORED if c not in r["res"]] + [c for c in new if c not in n["res"]]
        if missing:
            raise ValueError(f"run {i}: no rows for {missing}")
        out[i] = {**r, "res": {**r["res"], **{c: n["res"][c] for c in new}}}
    return out


def cell(p):
    """A T3 cell: D (cluster SE), four decimals, signed."""
    return f"{p['D']:+.4f} ({p['cluster_se']:.4f})"


def regime_summary(R, rs_rows, new_rows, new=NEW):
    """baselines_p1's statistics of the configs of `new` in one regime: the merged rows'
    Rows, each config's summary, its paired difference against ship and against the
    legacy Predictor (baselines_p1.part), and the realised level."""
    rows = merge(rs_rows, new_rows, new)
    G = BP.Rows(R, rows, extra=new)
    configs = {c: G.config_summary(c) for c in STORED + tuple(new)}
    for c in configs:
        configs[c]["mean_call_ms"] = 1e3 * float(np.mean([rows[i]["res"][c]["mean_s"] for i in G.ids]))
        configs[c]["max_call_s"] = float(max(rows[i]["res"][c]["max_s"] for i in G.ids))
    vs_ship = {c: BP.part(G, c, SHIP) for c in new}
    for c in new:
        vs_ship[c]["U95"] = G.diff(c, SHIP)["U95"]
    vs_legacy = {c: BP.part(G, c, "legacy") for c in new}
    alc = {c: configs[c]["ALC"] for c in configs}
    return {"runs": len(G.ids), "pairs": int(len(G.w)), "clusters": G.K, "realised": G.level(),
            "configs": configs, "rank": sorted(configs, key=alc.get),
            "vs_ship": vs_ship, "vs_legacy": vs_legacy,
            "headline": {c: {**BP.brief(vs_ship[c], f"{c} - ship"), "U95": vs_ship[c]["U95"],
                             "cell": cell(vs_ship[c])} for c in new},
            "budgets_0_1": {f"{c} - ship": BP.budget_view(vs_ship[c]) for c in new}}, G


def p1a_agreement(G, R, rs_summary):
    """The largest |difference| between P1a's stored vs_ship numbers for its configs and
    the same numbers recomputed from the merged rows."""
    worst = 0.0
    for c in STORED:
        if c == SHIP:
            continue
        mine, theirs = G.diff(c, SHIP), rs_summary["vs_ship"][R][c]["exact"]
        worst = max(worst, *(abs(mine[k] - theirs[k]) for k in ("D", "run_se", "cluster_se", "strat_se")))
    return worst


#: the parts of a config summary that come from timings, not from scores
TIMING_KEYS = ("mean_call_ms", "max_call_s")


def bp_agreement(summ, bp_regime, new=NEW):
    """Whether this script's statistics equal baselines_p1's stored ones for the configs
    of `new`, as JSON values: vs_ship and vs_legacy parts in full, config summaries
    without the timing keys. -> {name: {'equal', 'keys_differing'}}."""
    def js(x):
        return json.loads(json.dumps(clean(x)))

    out = {}
    for c in new:
        for side in ("vs_ship", "vs_legacy"):
            mine = {k: v for k, v in js(summ[side][c]).items() if k != "U95"}
            theirs = js(bp_regime[side][c])
            bad = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
            out[f"{side}.{c}"] = {"equal": not bad, "keys_differing": bad}
        mine = {k: v for k, v in js(summ["configs"][c]).items() if k not in TIMING_KEYS}
        theirs = {k: v for k, v in js(bp_regime["configs"][c]).items() if k not in TIMING_KEYS}
        bad = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        out[f"configs.{c}"] = {"equal": not bad, "keys_differing": bad}
    return out


def scored_digest(rows, new=NEW):
    """sha256 over the scored values of `new` on every run (EXACT_FIELDS, canonical
    JSON, runs in order): equal for any bit-for-bit re-scoring, whatever the timings."""
    h = hashlib.sha256()
    for i in sorted(rows):
        v = {c: {k: rows[i]["res"][c][k] for k in EXACT_FIELDS} for c in new}
        h.update(json.dumps([i, clean(v)], sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def load_rows(R, rows_dir=ROWS):
    """(P1a's rows, this script's rows) of regime R, on the runs scored here."""
    new, old = {}, {}
    for i in range(N_RUNS[R]):
        p = row_path(R, i, rows=rows_dir)
        if os.path.exists(p):
            new[i] = load_json(p)
            old[i] = BP.stored_row(R, i)
    return old, new


def t3_table(regimes, bp_summary):
    """The comparator rows of T3 for the two configs: every column's D, cluster SE and
    cell, its source, and the realised level the column is labelled with."""
    out = {}
    for c in NEW:
        cols = {}
        for R in T3_COLUMNS:
            if R in regimes:
                p, lv = regimes[R]["vs_ship"][c], regimes[R]["realised"]["mean"]
                src = (f"E1 summary.regimes.{R}.vs_ship.{c} (re-scored control, equal to baselines_p1's)"
                       if R in CONTROL else f"E1 summary.regimes.{R}.vs_ship.{c}")
            elif R in bp_summary["regimes"]:
                p, lv = bp_summary["regimes"][R]["vs_ship"][c], bp_summary["regimes"][R]["realised"]["mean"]
                src = f"BP summary.regimes.{R}.vs_ship.{c}"
            else:
                continue
            cols[R] = {"D": p["D"], "cluster_se": p["cluster_se"], "cell": cell(p),
                       "realised_level": lv, "source": src}
        out[c] = {"label": LABELS[c], "columns": cols}
    return out


def missing_rows(rows_dir=ROWS):
    """{regime: planned runs with no row file under rows_dir}, regimes with none left out."""
    out = {}
    for R in SCORED:
        gone = [i for i in range(N_RUNS[R]) if not os.path.exists(row_path(R, i, rows=rows_dir))]
        if gone:
            out[R] = gone
    return out


def stage_summarise(rows_dir=ROWS, out=OUT, allow_missing=False):
    t0 = time.time()
    gone = missing_rows(rows_dir)
    if gone and not allow_missing:
        raise SystemExit(f"summarise: {sum(map(len, gone.values()))} of {sum(N_RUNS.values())} planned rows "
                         f"are missing under {rows_dir} ({ {R: len(v) for R, v in gone.items()} }); nothing "
                         f"written. Score them (stage score) or pass --allow-missing for a partial summary.")
    lock = lock_checks()
    st, _ = p1a_state()
    bp = load_json(BP.OUT)
    lib_now = BP.lib_digests()
    checks = {f"lock: {k}": v for k, v in lock.items()}
    regimes, rowinfo, have = {}, {}, {}
    amend_utc = commit_utc(AMENDMENT_COMMIT)
    for R in SCORED:
        old, new = load_rows(R, rows_dir)
        have[R] = len(new)
        if not new:
            continue
        complete = len(new) == N_RUNS[R]
        summ, G = regime_summary(R, old, new)
        libs = {json.dumps(r["lib"], sort_keys=True) for r in list(new.values()) + list(old.values())}
        checks[f"{R}: one library for P1a's rows and these"] = len(libs) == 1
        checks[f"{R}: that library is the current one"] = libs == {json.dumps(lib_now, sort_keys=True)}
        checks[f"{R}: P1a's smooth and smcal reproduced on every run"] = all(
            set(CHECK_EVERY) <= set(r["checks"]) for r in new.values()) and all(
            c["ok"] for r in new.values() for c in r["checks"].values())
        checks[f"{R}: P1a's ship, onepl, legacy reproduced on runs {list(CHECK_RUNS[R])}"] = all(
            i in new and set(CHECK_SOME) <= set(new[i]["checks"]) and
            all(new[i]["checks"][c]["ok"] for c in CHECK_SOME) for i in CHECK_RUNS[R])
        scoring = sorted({r.get("scoring_sha256") for r in new.values()})
        checks[f"{R}: every row scored by this file's scoring code"] = scoring == [SCORING_SHA]
        utcs = sorted(r["utc"] for r in new.values())
        checks[f"{R}: every row scored after amendment 2 was committed"] = bool(amend_utc) and utcs[0] > amend_utc
        agree = p1a_agreement(G, R, st["summary"]) if complete else None
        if complete:
            checks[f"{R}: P1a's vs_ship numbers recomputed"] = agree <= 1e-12
        summ["complete"] = complete
        summ["p1a_agreement_max_abs"] = agree
        summ["reproduction_max_abs_brier"] = {
            n: max([r["checks"][n]["max_abs_brier"] for r in new.values() if n in r["checks"]], default=None)
            for n in CHECK_EVERY + CHECK_SOME}
        if R in CONTROL:
            exact = {n: sum(not r["checks"][f"baselines_p1:{n}"]["fields_differing"] for r in new.values())
                     for n in NEW}
            summ["control"] = {
                "rows_equal_to_baselines_p1": {n: f"{exact[n]} of {len(new)}" for n in NEW},
                "max_abs_brier_vs_baselines_p1": {n: max(r["checks"][f"baselines_p1:{n}"]["max_abs_brier"]
                                                         for r in new.values()) for n in NEW}}
            checks[f"{R}: rows equal baselines_p1's bit for bit on every run"] = all(
                exact[n] == len(new) for n in NEW)
            if complete:
                ag = bp_agreement(summ, bp["summary"]["regimes"][R])
                summ["control"]["statistics_equal_to_baselines_p1"] = ag
                checks[f"{R}: statistics equal baselines_p1's stored numbers exactly"] = all(
                    v["equal"] for v in ag.values())
        regimes[R] = summ
        rowinfo[R] = {"runs": len(new), "scored_values_sha256": scored_digest(new),
                      "per_run_sha16": {str(i): hashlib.sha256(json.dumps(
                          clean({c: {k: new[i]["res"][c][k] for k in EXACT_FIELDS} for c in NEW}),
                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
                          for i in sorted(new)},
                      "scoring_sha256": scoring, "script_sha16": sorted({r["script"] for r in new.values()}),
                      "utc": [utcs[0], utcs[-1]],
                      "task_s_total": round(float(sum(r["task_s"] for r in new.values())), 1),
                      "task_s_mean": float(np.mean([r["task_s"] for r in new.values()])),
                      "max_rss_gb": max(r.get("rss_gb", 0.0) for r in new.values())}
        del G
        gc.collect()
    complete = all(have[R] == N_RUNS[R] for R in SCORED)
    checks["every planned run scored"] = complete
    checks["P1a's smcal is baselines_p1's"] = tuple(st["smcal"]["chosen"]) == SMCAL
    checks["baselines_p1's checks hold"] = bool(bp.get("all_checks_hold"))
    rows_dir_rel = os.path.relpath(rows_dir, ROOT)
    result = {
        "about": "E1 (docs/plans/p2_final_plan.md, amendment 2; report only, no-ship): EmpMean (the "
                 "organisers' empirical mean, baselines_p1 'empmean') and 1PL-ship (the plain 1PL at "
                 "LEVEL, baselines_p1 'rasch') on P1a's READING, AUDIT, MIXTURE and FLAT runs (seed 11, "
                 "runs 0-79, split scope 'pair'), paired with ship on P1a's stored rows; TUNED 0-79 "
                 "re-scored as the control against baselines_p1. D is comparator minus ship (positive: "
                 "hier-ship better); SEs run / cluster (parent, subject) / parent-stratified, "
                 "level_calibration.Boot with 2,000 resamples, seed 0. "
                 "Run history (disclosed): a dry run on 6 tasks, with an earlier scoring block, preceded "
                 "the scored run, and its rows were discarded. The scored run (2026-10-06, rows written "
                 "18:55 to 18:59 UTC by script sha16 8a8535cf3391aa85, scoring block "
                 "8b9afc6e589cbad3...) kept its rows in a session scratch directory under /private/tmp, "
                 "which the machine's restart cleared. On 2026-10-07 the 400 tasks were scored again "
                 "into the default rows directory by the same scoring block (rows.regimes.*.script_sha16 "
                 "names the script); every regime's scored_values_sha256 and every run's per_run_sha16 "
                 "equal the first run's, so every statistic here is the first run's bit for bit. Only "
                 "the timings (configs' mean_call_ms and max_call_s, rows.regimes.*.utc, task_s_*, "
                 "max_rss_gb) are the second scoring's.",
        "commands": ["python experiments/e1_baselines_rs.py lock",
                     "python experiments/e1_baselines_rs.py score --shard k/3   (k = 0, 1, 2; repeat "
                     "while it exits with 75)",
                     "python experiments/e1_baselines_rs.py summarise"],
        "plan": {"regimes": {R: {"seed": SEED, "runs": [0, N_RUNS[R] - 1], "knobs": RS.REGIMES[R][1],
                                 "role": "control" if R in CONTROL else "E1"} for R in SCORED},
                 "split_scope": SPLIT_SCOPE, "configs": CONFIGS, "labels": LABELS,
                 "built_by": "experiments/baselines_p1.py Maker / score_one / CONFIGS (imported)",
                 "smcal": list(SMCAL),
                 "reproduction": {"every_run": list(CHECK_EVERY), "configs": list(CHECK_SOME),
                                  "runs": {R: list(v) for R, v in CHECK_RUNS.items()},
                                  "tolerance": REPRO_TOL, "control_fields_exact": list(EXACT_FIELDS)},
                 "e1_bullet_sha256": E1_BULLET_SHA256, "amendment_commit": AMENDMENT_COMMIT,
                 "amendment_commit_utc": amend_utc, "no_ship": True,
                 "pythonhashseed": "the two configs' rows do not depend on it: baselines_p1.Maker "
                                   "builds parents in sorted order and neither config keeps an EB trace"},
        "checks": checks, "all_checks_hold": all(checks.values()),
        "summary": {"regimes": regimes, "scored": have, "complete": complete,
                    "t3": t3_table(regimes, bp["summary"])},
        "rows": {"dir": rows_dir_rel if not rows_dir_rel.startswith("..") else rows_dir,
                 "default_dir": os.path.relpath(ROWS, ROOT),
                 "note": "rows are row files as baselines_p1 writes them, one a task; a directory "
                         "outside the repository is temporary. Re-scoring reproduces them bit for "
                         "bit: each regime's scored_values_sha256 covers EXACT_FIELDS of both configs "
                         "on every run, not the timings",
                 "regimes": rowinfo},
        "wall_s": round(time.time() - t0, 1), "provenance": provenance(t0)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(clean(result), f, indent=1)
        f.write("\n")
    os.replace(tmp, out)
    print(f"summarise: {have}; checks {'hold' if result['all_checks_hold'] else 'FAIL'}: "
          f"{[k for k, v in checks.items() if not v]}", flush=True)
    for c, t in result["summary"]["t3"].items():
        print(f"  {t['label']}: " + "; ".join(f"{R} {v['cell']}" for R, v in t["columns"].items()))
    return result


def stage_status(rows_dir=ROWS):
    for R in SCORED:
        n = sum(os.path.exists(row_path(R, i, rows=rows_dir)) for i in range(N_RUNS[R]))
        print(f"{R:8s} {n:3d}/{N_RUNS[R]}")


def stage_lock():
    res = lock_checks()
    print(json.dumps(res, indent=1))
    print(f"scoring block sha256 {SCORING_SHA}")
    if not all(res.values()):
        raise SystemExit(1)


def parse_shard(s):
    k, n = (int(x) for x in s.split("/"))
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"bad shard {s}")
    return k, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("lock", "score", "summarise", "status"))
    ap.add_argument("--rows", default=ROWS, help="row directory (default %(default)s)")
    ap.add_argument("--out", default=OUT, help="output file of summarise (default %(default)s)")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), help="score: k/n")
    ap.add_argument("--limit", type=int, default=None, help="score: at most this many tasks")
    ap.add_argument("--allow-missing", action="store_true",
                    help="summarise: write a partial summary when planned rows are missing (by default "
                         "summarise exits non-zero and writes nothing)")
    a = ap.parse_args(argv)
    if a.stage == "lock":
        stage_lock()
    elif a.stage == "score":
        stage_score(a.shard, os.path.abspath(a.rows), a.limit)
    elif a.stage == "summarise":
        stage_summarise(os.path.abspath(a.rows), a.out, a.allow_missing)
    else:
        stage_status(os.path.abspath(a.rows))


if __name__ == "__main__":
    main()
