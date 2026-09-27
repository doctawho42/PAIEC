"""The multiple-choice floor the shipped hier reads, corrected: two defects of
paiec.mcq on the public items, and what fixing them does to Brier ALC.

paiec.hier floors a target at c = Hyper.guess * mcq.floor_of(mcq_text(content))
(HierPredictor.floor_item; guess 0.5 as shipped, so c = 0.1 on a five-option
item and 1/6 on a three-option one), in the likelihood of a floored item's
labels as well as in its prediction: p = c + (1 - c - slip) E[sigmoid(eta)].
mcq.n_options as it stood at commit bd0be67 (OLD below, verbatim) gets two
kinds of public item wrong, both on matharena:

  listed    the 336 Math Kangaroo items (competitions kangaroo_2025_*, one
            text shared by all of them) are five-option multiple choice with
            the options in an image; the text names them only as "(A, B, C, D,
            or E)" and ends "See image.", so OLD reads no option and hier puts
            no floor under them
  drawing   20 integer-answer items (aime_2025 12, aime_2025_I 3, aime_2025_II
            3, hmmt_feb_2025 2; answers 588, 293, 336, 20) hold TikZ drawings
            whose labelled points, "\\coordinate (A) at (0,0);", match OLD's
            inline "(A) " pattern: floored at 1/5 (1/3 on the hmmt items)

NEW cuts drawing blocks (\\begin{tikzpicture|asy|picture} ... \\end{...}, AoPS
[asy] ... [/asy]; an unclosed block runs to the end) out of the text before
OLD's patterns, and falls back to a parenthesised "(A, B, ..., or X)" list
(the comma before 'or' / 'and' optional), A first and in order, at least three
letters (paiec.itemcov.n_options is the research version). Otherwise it is
OLD. (The optional comma was added after the replay; the list reads all 50,528
public text views, raw and mcq_text, with and without drawings cut, exactly as
the replay's did, so the stored rows stand.) The script holds both, so it
measures the same thing before and after the library adopts NEW (provenance
records which one paiec.mcq currently is).

paiec.mcq adopted a revision of NEW, not NEW itself, and NEW stays here as it
was replayed: the revision cuts only closed drawing blocks, leaves a non-blank
placeholder where one was cut, reads a list joined by 'or' only, and gives
every whitespace run one reading, so that no text costs more than a linear
pass (NEW's two adjacent '\s*' and its blank cuts were quadratic in the worst
case). On all 25,264 public text views (raw and mcq_text, six benchmarks) the
revision and NEW read the same options, so every number below holds for the
adopted floor; tests/test_mcq.py pins that.

Stages
  items    before/after option counts over every public item of every
           benchmark (mmdocrag included), on the text hier reads (mcq_text)
           and on the raw text (the legacy Predictor's), by benchmark and
           item_features group (testlike.PARENT_KEYS: matharena competition),
           every change classified 'listed' or 'drawing' (anything else is
           'other' and must be explained), with the eligible pairs' responses
           and accuracy on the changed items
  run      replay the shipped hier (experiments/harness.py's ship_name():
           submission/model.py's LEVEL over paiec.prior.build fitted leaving
           the target's parent benchmark out, level_calibration's bundle and
           factory) on identical checkpoints (level_calibration.checkpoints:
           official's own _slots and _acquire, split scope 'pair', fresh
           instances per checkpoint) under each floor variant, by
           monkeypatching paiec.hier.floor_of (hier imported the name; the
           memo of floors is per instance). Regimes (PLAN): test-like runs
           (testlike.Regime() defaults, seed 2; 'tl'), public R1
           (official.sample_run, seed 0, benchmark-first 'r1b' and
           pair-uniform 'r1p'). One JSON row per run in --rows, resumable;
           --shard k/n splits the runs over separate processes
  summary  -> results/mcq_floor.json
  show     markdown tables of the summary

Variants: 'ship' (OLD, guess as fitted: 0.5), 'fix' (NEW), 'drawing only'
(the cut, no list fallback), 'listed only' (the list fallback, no cut) and
'fix g1' (NEW with Hyper.guess 1, the hard floor: c = 1/5 on Kangaroo). A
variant is replayed on a run only where some item of the run (candidate or
target) has a different c under it than under 'ship'; elsewhere the replay
would be the same computation (the floor enters hier only through
floor_item, and the offline bundle does not read it), and 'ship' is copied.
The first --verify runs of each regime replay every variant anyway and record
that the copies are exact, and replay hier with paiec.hier.floor_of
unpatched to show which variant the library is; runs that
experiments/harness.py's collect stage stored (data/harness_rows) are
compared with 'ship' prediction for prediction.

Statistics (experiments/harness.py's stats): paired differences per pair
appearance, weighted 1 / run size (a run's ALC is the mean over its pairs),
with the SE over runs, a pair-cluster bootstrap (2,000 resamples, ratio
estimator; cluster (parent, subject), testlike.cluster_key, which is
(benchmark, subject) on public runs) and the same stratified by parent.
Reported overall, per budget (the budget's contribution to ALC is its weight
times its difference; B0 is where the floor acts before any label), and on
subsets: matharena appearances (test-like: pseudo-benchmarks whose parent is
matharena) and the rest, and matharena appearances holding Kangaroo targets,
TikZ targets or neither. A subset is reported as its per-appearance mean and
as its contribution to the overall difference (the difference zeroed outside
it, so the contributions of a partition add up to the overall number).

Run: python experiments/mcq_floor.py --stage items
     python experiments/mcq_floor.py --stage run --shard 0/2 &   # two processes,
     python experiments/mcq_floor.py --stage run --shard 1/2     # ~700 MB each
     python experiments/mcq_floor.py --stage summary
     python experiments/mcq_floor.py --stage show
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402

OUT = os.path.join(ROOT, "results", "mcq_floor.json")
ROWS = os.path.join(ROOT, "data", "mcq_floor_rows")
#: runs 0..n-1 of each regime (experiments/level_calibration.py's REGIMES keys)
PLAN = {"tl": 200, "r1b": 200, "r1p": 200}
BENCHMARKS = ("matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench",
              "swe_rebench")
W6 = H.W6
BUDGETS = H.BUDGETS
EPS = 1e-12            # a prediction counts as changed above this
MATERIAL = 1e-3        # and as materially changed above this


# --- the floors -------------------------------------------------------------------------

#: paiec.mcq at commit bd0be67, verbatim
OLD_PATS = [re.compile(r"(?m)^\s*\(?([A-E])[\)\.]\s+"),      # A) / (A) / A. at line start
            re.compile(r"\(([A-E])\)\s"),                     # inline (A)
            re.compile(r"(?m)^\s*([A-E])\s*[:：]\s")]


def old_n_options(text):
    """Number of answer options, or 0 when the item is not multiple choice."""
    if not text:
        return 0
    best = set()
    for p in OLD_PATS:
        letters = {m.group(1).upper() for m in p.finditer(text)}
        if len(letters) > len(best):
            best = letters
    if len(best) < 3:
        return 0
    for k in range(len(best), 2, -1):
        if set("ABCDE"[:k]) <= best:
            return k
    return 0


def old_floor_of(text, cap=0.34):
    n = old_n_options(text)
    return min(1.0 / n, cap) if n else 0.0


#: drawing code, whose labelled points are not answer options (NEW as replayed)
DRAWING = re.compile(r"\\begin\{(tikzpicture|asy|picture)\}.*?(?:\\end\{\1\}|\Z)"
                     r"|\[asy\].*?(?:\[/asy\]|\Z)", re.S)
#: options named in the instructions (NEW as replayed)
LISTED = re.compile(r"\(\s*((?:[A-H]\s*,\s*)+[A-H]\s*,?\s*(?:or|and)\s+[A-H])\s*\)")


def listed_options(text):
    """Options named in a parenthesised list, A first and in order, at least
    three; 0 when there is none."""
    for m in LISTED.finditer(text):
        letters = re.findall(r"[A-H]", m.group(1))
        k = len(letters)
        if k >= 3 and "".join(letters) == "ABCDEFGH"[:k]:
            return k
    return 0


def new_n_options(text, drawings=True, listed=True):
    """The proposed mcq.n_options (both switches on): OLD on the text with its
    drawings cut out, else a listed '(A, B, ..., or X)'."""
    if not text:
        return 0
    if drawings:
        text = DRAWING.sub(" ", text)
    k = old_n_options(text)
    if k or not listed:
        return k
    return listed_options(text)


def new_floor_of(text, cap=0.34, drawings=True, listed=True):
    n = new_n_options(text, drawings, listed)
    return min(1.0 / n, cap) if n else 0.0


FLOORS = {"old": old_floor_of, "new": new_floor_of,
          "drawing": partial(new_floor_of, listed=False),
          "listed": partial(new_floor_of, drawings=False)}
#: variant -> (floor family, Hyper.guess override or None for the fitted one)
VARIANTS = {"ship": ("old", None), "fix": ("new", None), "drawing only": ("drawing", None),
            "listed only": ("listed", None), "fix g1": ("new", 1.0)}
COMPARE = (("fix", "ship"), ("drawing only", "ship"), ("listed only", "ship"), ("fix g1", "fix"))


def change_kind(old, new, text):
    """'listed' (no option read before, a listed set now), 'drawing' (options
    read before, none once drawings are cut) or 'other'."""
    if old == 0 and new > 0 and old_n_options(DRAWING.sub(" ", text)) == 0 and listed_options(
            DRAWING.sub(" ", text)) == new:
        return "listed"
    if old > 0 and new == 0 and DRAWING.search(text) and old_n_options(DRAWING.sub(" ", text)) == 0:
        return "drawing"
    return "other"


@contextlib.contextmanager
def floor_patch(fn):
    """paiec.hier reads the floor through its module-level name floor_of."""
    import paiec.hier as HM
    was = HM.floor_of
    HM.floor_of = fn
    try:
        yield
    finally:
        HM.floor_of = was


def library_floor():
    """Which of OLD and NEW paiec.mcq.floor_of is, on the unit texts and every
    public item text ('old', 'new', 'neither' or 'unknown' without data)."""
    from paiec import mcq
    from paiec.hier import mcq_text
    texts = list(UNIT_TEXTS)
    for t in public_texts():
        texts += [t, mcq_text(t)]
    agree = {"old": all(mcq.floor_of(t) == old_floor_of(t) for t in texts),
             "new": all(mcq.floor_of(t) == new_floor_of(t) for t in texts)}
    return "new" if agree["new"] and not agree["old"] else "old" if agree["old"] and not agree["new"] \
        else "neither" if not any(agree.values()) else "unknown"


UNIT_TEXTS = ("Reason step by step, referring to the given multiple choice options (A, B, C, D, or E), "
              "of which exactly one is correct.\n\nSee image.",
              "Find the area.\n\\begin{tikzpicture}\n\\coordinate (A) at (0,0);\n\\coordinate (B) at (1,0);\n"
              "\\coordinate (C) at (1,1);\n\\coordinate (D) at (0,1);\n\\coordinate (E) at (2,2);\n"
              "\\end{tikzpicture}\nAnswer with an integer.",
              "Pick: (A) 1 (B) 2 (C) 3 (D) 4", "Q\nA. one\nB. two\nC. three", "A: x\nB: y\nC: z\nD: w",
              "", "no options")


# --- items ----------------------------------------------------------------------------

def _read_items(bench):
    import pandas as pd
    path = os.path.join(ROOT, "data", bench, "items.parquet")
    if not os.path.exists(path):
        return None
    it = pd.read_parquet(path, columns=["item_id", "content", "item_features"])
    return [(str(i), "" if c is None else str(c), "" if f is None else str(f))
            for i, c, f in zip(it.item_id, it.content, it.item_features)]


def public_texts():
    out = []
    for b in BENCHMARKS:
        rows = _read_items(b)
        out += [c for _, c, _ in rows or ()]
    return out


def group_of(bench, features):
    from paiec.testlike import PARENT_KEYS, feature
    key = PARENT_KEYS.get(bench)
    return feature({"item_features": features}, key) if key else ""


def item_table(before=old_n_options, after=new_n_options, benchmarks=BENCHMARKS):
    """Per benchmark: items, items floored before/after (on mcq_text and on the
    raw text) and every change as (group, before, after, kind) -> count, the
    kind from change_kind. `before`/`after` are n_options functions."""
    from paiec.hier import mcq_text
    out = {}
    for b in benchmarks:
        rows = _read_items(b)
        if rows is None:
            continue
        rec = {"items": len(rows)}
        for view, fn in (("mcq_text", mcq_text), ("raw", lambda t: t or "")):
            fl = [0, 0]
            ch = Counter()
            for _, content, feats in rows:
                t = fn(content)
                x, y = before(t), after(t)
                fl[0] += x > 0
                fl[1] += y > 0
                if x != y:
                    ch[(group_of(b, feats), x, y, change_kind(x, y, t))] += 1
            rec[view] = {"floored_before": fl[0], "floored_after": fl[1], "changed": sum(ch.values()),
                         "changes": [{"group": g, "before": x, "after": y, "kind": k, "items": n}
                                     for (g, x, y, k), n in sorted(ch.items())]}
        out[b] = rec
    return out


def changed_responses():
    """Responses of the eligible pairs on items whose floor NEW changes (the
    text hier reads): count and accuracy per kind and benchmark."""
    from paiec.hier import mcq_text
    LC = H._lc()
    memo, acc = {}, defaultdict(list)
    for p in LC.pairs():
        for r in p.responses:
            c = r.item["item_content"]
            if c not in memo:
                t = mcq_text(c)
                x, y = old_n_options(t), new_n_options(t)
                memo[c] = change_kind(x, y, t) if x != y else None
            if memo[c]:
                acc[(p.benchmark_id, memo[c])].append((p.subject_id, r.item_key, r.label))
    return {f"{b}/{k}": {"responses": len(v), "items": len({x[1] for x in v}),
                         "subjects": len({x[0] for x in v}),
                         "accuracy": float(np.mean([x[2] for x in v]))}
            for (b, k), v in sorted(acc.items())}


def stage_items(args):
    res = H.load_json(args.out) or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res["items"] = {"table": item_table(), "responses": changed_responses(),
                        "distinct_changed_texts": distinct_changed()}
    res["provenance"] = provenance()
    H.save_json(args.out, res)
    show_items(res["items"])


def distinct_changed():
    from paiec.hier import mcq_text
    seen = Counter()
    for b in BENCHMARKS:
        for _, c, _ in _read_items(b) or ():
            t = mcq_text(c)
            x, y = old_n_options(t), new_n_options(t)
            if x != y:
                seen[(b, change_kind(x, y, t), c)] += 1
    out = Counter()
    for (b, k, _), n in seen.items():
        out[f"{b}/{k}"] += 1
    return dict(out)


# --- replay ---------------------------------------------------------------------------

def factory(run, ship, guess, made):
    """level_calibration's factory for the shipped config, or the same with
    Hyper.guess overridden."""
    LC = H._lc()
    if guess is None:
        return LC.factory([ship], run, made)
    from paiec import testlike as T
    from paiec.hier import HierPredictor
    ids = T.anon_parents(run)
    nu, ov, _ = LC.parse(ship)

    def f():
        ms = {}
        for par in set(ids.values()):
            prior, hyper = LC.bundle(par, nu)
            ms[par] = HierPredictor(prior, replace(hyper, **ov, guess=float(guess)))
        made.extend(ms.values())
        return lambda input, labeled=None: ms[ids[input[1]["benchmark_id"]]].predict(input, labeled)
    return f


def replay(cps, fac, floor=None):
    """Predictions per checkpoint and evaluation response, (6, n): a fresh
    instance per checkpoint, deduplicated inputs (level_calibration.evaluate),
    paiec.hier.floor_of patched to `floor` unless None."""
    ctx = floor_patch(floor) if floor is not None else contextlib.nullcontext()
    out = []
    with ctx:
        for b, labeled, inputs, index in cps:
            fn = fac()
            P = np.array([fn(inp, labeled) for inp in inputs], float)
            if not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(P > 1):
                raise ValueError("invalid prediction")
            out.append(P[index])
    return np.vstack(out)


def _harness_first_positions(row_slots):
    pos, at = [], 0
    for d in row_slots:
        N = np.asarray(d["ev_N"], np.int64)
        pos.append(at + np.concatenate([[0], np.cumsum(N)[:-1]]).astype(np.int64))
        at += int(N.sum())
    return pos


def run_one(regime, i, verify, harness_rows):
    """Replay one run under every variant it needs -> the run's row."""
    from paiec import official as O
    from paiec import testlike as T
    from paiec.hier import mcq_text
    LC = H._lc()
    t0 = time.perf_counter()
    ship = H.ship_name()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = LC.draw(H.REGIMES[regime], i)
        slots, cps = LC.checkpoints(run)
        ids = T.anon_parents(run)
        nu = LC.parse(ship)[0]
        guesses = {LC.bundle(par, nu)[1].guess for par in set(ids.values())}
    if len(guesses) != 1:
        raise RuntimeError(f"parents disagree on Hyper.guess: {guesses}")
    g_ship = guesses.pop()
    contents = {e[1][1]["item_content"] for s in slots for e in s.candidates} | \
               {t[0][1]["item_content"] for s in slots for t in s.targets}
    fl = {}
    for c in contents:
        t = mcq_text(c)
        fl[c] = {f: fn(t) for f, fn in FLOORS.items()}

    def c_of(v, c):
        fam, g = VARIANTS[v]
        return (g_ship if g is None else g) * fl[c][fam]

    differs = {v: any(c_of(v, c) != c_of("ship", c) for c in contents) for v in VARIANTS}
    force = i < verify
    preds, made, secs = {}, {}, {}
    for v, (fam, g) in VARIANTS.items():
        if v == "ship" or differs[v] or force:
            made[v] = []
            ts = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds[v] = replay(cps, factory(run, ship, g, made[v]), FLOORS[fam])
            secs[v] = round(time.perf_counter() - ts, 3)
    checks = {}
    if force:
        checks["copies_exact"] = {v: float(np.max(np.abs(preds[v] - preds["ship"])))
                                  for v in VARIANTS if not differs[v]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lib = replay(cps, LC.factory([ship], run, []), None)
        checks["library_vs"] = {v: float(np.max(np.abs(lib - preds[v]))) for v in ("ship", "fix")}
    for v in VARIANTS:
        if v not in preds:
            preds[v] = preds["ship"]
    hp = os.path.join(harness_rows, regime, f"{i}.pkl")
    if harness_rows and os.path.exists(hp):
        with open(hp, "rb") as f:
            hr = pickle.load(f)
        if hr.get("ship") == ship and len(hr["slots"]) == len(slots):
            d = 0.0
            for pos, row in zip(_harness_first_positions(hr["slots"]), hr["slots"]):
                d = max(d, float(np.nanmax(np.abs(preds["ship"][:, pos] - np.asarray(row["ev_p"])))))
            checks["harness_rows_max_abs"] = d
    y_all = np.array([t[1] for s in slots for t in s.targets], float)
    kind_all = np.array([("kang" if fl[c]["new"] > fl[c]["old"] == 0 else
                          "tikz" if fl[c]["old"] > 0 and fl[c]["new"] == 0 else "")
                         for c in (t[0][1]["item_content"] for s in slots for t in s.targets)])
    apps, at = [], 0
    n_run = len(slots)
    for entry, s in zip(run, slots):
        pair, _ = O._entry(entry)
        n = len(s.targets)
        sl = slice(at, at + n)
        at += n
        y, kind = y_all[sl], kind_all[sl]
        acq_changed = sum(1 for e in s.acquired[:31]
                          if fl[e[0][1]["item_content"]]["old"] != fl[e[0][1]["item_content"]]["new"])
        app = {"name": s.name, "parent": T.parent_of(s.name), "kind": T.kind_of(s.name),
               "subject": str(pair.subject_id), "n_run": n_run, "resp": n, "items": s.n_items,
               "kang": int(np.sum(kind == "kang")), "tikz": int(np.sum(kind == "tikz")),
               "acq_changed": acq_changed,
               "kang_y": float(y[kind == "kang"].sum()), "tikz_y": float(y[kind == "tikz"].sum()),
               "brier": {}, "chg": {}, "direct": {}}
        for v in VARIANTS:
            P = preds[v][:, sl]
            app["brier"][v] = [float(np.mean((P[b] - y) ** 2)) for b in range(6)]
            app["direct"][v] = {"kang_p": [float(P[b][kind == "kang"].sum()) for b in range(6)],
                                "tikz_p": [float(P[b][kind == "tikz"].sum()) for b in range(6)]}
            if v == "ship":
                continue
            D = np.abs(P - preds["ship"][:, sl])
            direct = kind != ""
            app["chg"][v] = {
                "n_dir": [int(np.sum(D[b][direct] > EPS)) for b in range(6)],
                "n_ind": [int(np.sum(D[b][~direct] > EPS)) for b in range(6)],
                "mat_dir": [int(np.sum(D[b][direct] > MATERIAL)) for b in range(6)],
                "mat_ind": [int(np.sum(D[b][~direct] > MATERIAL)) for b in range(6)],
                "abs_dir": [float(D[b][direct].sum()) for b in range(6)],
                "abs_ind": [float(D[b][~direct].sum()) for b in range(6)],
                "max_dir": [float(D[b][direct].max(initial=0.0)) for b in range(6)],
                "max_ind": [float(D[b][~direct].max(initial=0.0)) for b in range(6)]}
        apps.append(app)
    return {"regime": regime, "run": i, "ship": ship, "guess": g_ship, "differs": differs,
            "replay_s": secs, "checks": checks, "apps": apps,
            "failures": {v: int(sum(getattr(m, "failures", 0) for m in ms)) for v, ms in made.items()},
            "secs": round(time.perf_counter() - t0, 3)}


def row_path(rows, regime, i):
    return os.path.join(rows, regime, f"{i}.json")


def stage_run(args):
    k, n = (int(x) for x in args.shard.split("/"))
    todo = [(r, i) for i in range(max(PLAN.values())) for r in args.regimes if i < plan_n(args, r)]
    todo = [t for j, t in enumerate(todo) if j % n == k]
    todo = [t for t in todo if not os.path.exists(row_path(args.rows, *t))]
    print(f"run shard {k}/{n}: {len(todo)} runs to replay", flush=True)
    t0 = time.time()
    for regime, i in todo:
        os.makedirs(os.path.join(args.rows, regime), exist_ok=True)
        row = run_one(regime, i, args.verify, args.harness_rows)
        path = row_path(args.rows, regime, i)
        with open(path + ".tmp", "w") as f:
            json.dump(row, f)
        os.replace(path + ".tmp", path)
        print(f"{regime} {i}: {row['secs']:.1f}s replayed {sorted(row['replay_s'])} "
              f"checks {row['checks']} ({time.time() - t0:.0f}s)", flush=True)


def plan_n(args, regime):
    return PLAN[regime] if args.n is None else args.n


# --- summary --------------------------------------------------------------------------

def _alc(v):
    return float(np.dot(W6, v))


SUBSETS = {
    "all": lambda a: True,
    "matharena": lambda a: a["parent"] == "matharena",
    "other": lambda a: a["parent"] != "matharena",
    "matharena kangaroo": lambda a: a["parent"] == "matharena" and a["kang"] > 0,
    "matharena tikz": lambda a: a["parent"] == "matharena" and a["tikz"] > 0,
    "matharena neither": lambda a: a["parent"] == "matharena" and a["kang"] == 0 and a["tikz"] == 0,
}


def load_rows(rows, regime, n):
    out = []
    for i in range(n):
        p = row_path(rows, regime, i)
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def summarise_regime(runs, boots):
    apps = [(r["run"], a) for r in runs for a in r["apps"]]
    run = np.array([i for i, _ in apps])
    w = np.array([1.0 / a["n_run"] for _, a in apps])
    cluster = np.array([f"{a['parent']}|{a['subject']}" for _, a in apps])
    parent = np.array([a["parent"] for _, a in apps])
    A = [a for _, a in apps]
    res = {"runs": len(runs), "appearances": len(A),
           "matharena_appearances": int(sum(a["parent"] == "matharena" for a in A)),
           "runs_with_matharena": len({i for i, a in apps if a["parent"] == "matharena"}),
           "runs_replayed": {v: int(sum(r["differs"][v] for r in runs)) for v in VARIANTS},
           "alc": {}, "compare": {}, "checks": checks_of(runs)}
    for v in VARIANTS:
        per_run = defaultdict(float)
        for (i, a), wi in zip(apps, w):
            per_run[i] += wi * _alc(a["brier"][v])
        x = np.array(list(per_run.values()))
        res["alc"][v] = {"mean": float(x.mean()), "run_se": float(x.std(ddof=1) / math.sqrt(len(x)))}
    for a_name, b_name in COMPARE:
        B = np.array([np.array(a["brier"][a_name]) - np.array(a["brier"][b_name]) for a in A])
        d = B @ W6
        comp = {}
        for sname, fn in SUBSETS.items():
            m = np.array([fn(a) for a in A])
            if not m.any():
                continue
            ent = {"appearances": int(m.sum()),
                   "per_appearance": H.stats(d[m], w[m], run[m], cluster[m], parent[m], boots),
                   "contribution": H.stats(np.where(m, d, 0.0), w, run, cluster, parent, boots),
                   "by_budget": [H.stats(B[m, b], w[m], run[m], cluster[m], parent[m], boots)
                                 for b in range(6)],
                   "budget_contribution": [float(W6[b] * H.weighted_mean(np.where(m, B[:, b], 0.0), w))
                                           for b in range(6)]}
            comp[sname] = ent
        comp["per_parent"] = {q: H.stats(d[parent == q], w[parent == q], run[parent == q],
                                         cluster[parent == q], parent[parent == q], boots)
                              for q in sorted(set(parent))}
        comp["predictions"] = prediction_changes(A, a_name, b_name)
        res["compare"][f"{a_name} - {b_name}"] = comp
    res["direct_calibration"] = calibration(A)
    return res


def prediction_changes(A, a_name, b_name):
    """How many evaluation responses' predictions move, per budget, directly
    (the target item's own floor changed) and indirectly (another item's did),
    relative to 'ship' (so only for comparisons against 'ship')."""
    if b_name != "ship":
        return None
    tot = np.zeros(6)
    tot_dir = np.zeros(6)
    agg = {k: np.zeros(6) for k in ("n_dir", "n_ind", "mat_dir", "mat_ind", "abs_dir", "abs_ind")}
    mx = {"max_dir": np.zeros(6), "max_ind": np.zeros(6)}
    for a in A:
        c = a["chg"][a_name]
        tot += a["resp"]
        tot_dir += a["kang"] + a["tikz"]
        for k in agg:
            agg[k] += np.array(c[k], float)
        for k in mx:
            mx[k] = np.maximum(mx[k], np.array(c[k], float))
    return {"responses": tot.tolist(), "direct_responses": tot_dir.tolist(),
            "changed_direct": agg["n_dir"].tolist(), "changed_indirect": agg["n_ind"].tolist(),
            "material_direct": agg["mat_dir"].tolist(), "material_indirect": agg["mat_ind"].tolist(),
            "mean_abs_direct": (agg["abs_dir"] / np.maximum(agg["n_dir"], 1)).tolist(),
            "mean_abs_indirect": (agg["abs_ind"] / np.maximum(agg["n_ind"], 1)).tolist(),
            "max_abs_direct": mx["max_dir"].tolist(), "max_abs_indirect": mx["max_ind"].tolist()}


def calibration(A):
    """Mean prediction per variant against the observed rate, per budget, on
    the evaluation responses whose own floor changed (Kangaroo, TikZ)."""
    out = {}
    for kind in ("kang", "tikz"):
        n = sum(a[kind] for a in A)
        if not n:
            continue
        out[kind] = {"responses": int(n), "rate": float(sum(a[f"{kind}_y"] for a in A) / n),
                     "mean_pred": {v: [float(sum(a["direct"][v][f"{kind}_p"][b] for a in A) / n)
                                       for b in range(6)] for v in VARIANTS}}
    return out


def checks_of(runs):
    out = {"copies_max_abs": 0.0, "library_vs_ship_max_abs": None, "library_vs_fix_max_abs": None,
           "harness_rows_runs": 0, "harness_rows_max_abs": 0.0,
           "hier_failures": {v: 0 for v in VARIANTS}, "verified_runs": 0}
    for r in runs:
        c = r["checks"]
        if "copies_exact" in c:
            out["verified_runs"] += 1
            out["copies_max_abs"] = max([out["copies_max_abs"]] + list(c["copies_exact"].values()))
            for v in ("ship", "fix"):
                k = f"library_vs_{v}_max_abs"
                out[k] = max(out[k] or 0.0, c["library_vs"][v])
        if "harness_rows_max_abs" in c:
            out["harness_rows_runs"] += 1
            out["harness_rows_max_abs"] = max(out["harness_rows_max_abs"], c["harness_rows_max_abs"])
        for v, n in r["failures"].items():
            out["hier_failures"][v] += n
    return out


def losses(regimes):
    """Every (regime, comparison, subset, measure) whose point estimate is
    above 0 (the variant worse), with its z against the largest SE."""
    out = []
    for rg, res in regimes.items():
        for comp, c in res["compare"].items():
            for sname in SUBSETS:
                if sname not in c:
                    continue
                ms = [("ALC", c[sname]["per_appearance"])] + \
                    [(f"B{BUDGETS[b]}", c[sname]["by_budget"][b]) for b in range(6)]
                for meas, st in ms:
                    if st is None or st["est"] <= 0:
                        continue
                    se = max(st["run_se"], st["cluster_se"], st["strat_se"])
                    out.append({"regime": rg, "comparison": comp, "subset": sname, "measure": meas,
                                "est": st["est"], "max_se": se, "z": st["est"] / se if se > 0 else None})
    return out


def stage_summary(args):
    res = H.load_json(args.out) or {}
    regimes = {}
    for rg in args.regimes:
        runs = load_rows(args.rows, rg, plan_n(args, rg))
        if runs:
            regimes[rg] = summarise_regime(runs, args.boots)
    res["regimes"] = regimes
    res["losses"] = losses(regimes)
    res["plan"] = {rg: plan_n(args, rg) for rg in args.regimes}
    res["provenance"] = provenance()
    H.save_json(args.out, res)
    print(f"wrote {args.out}")


def provenance():
    p = H.provenance()
    p["script_digest"] = H.digest(["experiments/mcq_floor.py"])
    try:
        p["library_floor"] = library_floor()
    except Exception as e:           # no data
        p["library_floor"] = f"unknown ({type(e).__name__})"
    p["variants"] = {k: list(v) for k, v in VARIANTS.items()}
    return p


# --- show -----------------------------------------------------------------------------

def show_items(items):
    print("| benchmark | items | floored before | floored after | changed | changes (group: before->after kind x items) |")
    print("|---|---|---|---|---|---|")
    for b, rec in items["table"].items():
        m = rec["mcq_text"]
        ch = "; ".join(f"{c['group'] or '-'}: {c['before']}->{c['after']} {c['kind']} x{c['items']}"
                       for c in m["changes"])
        raw = rec["raw"]
        extra = "" if raw["changes"] == m["changes"] else \
            f" (raw text: {raw['floored_before']}->{raw['floored_after']}, {raw['changed']} changed)"
        print(f"| {b} | {rec['items']} | {m['floored_before']} | {m['floored_after']} | {m['changed']} | "
              f"{ch or '-'}{extra} |")
    print("\nresponses of eligible pairs on changed items:", items["responses"])
    print("distinct changed texts:", items["distinct_changed_texts"])


def fmt(st, k=5):
    if st is None:
        return "-"
    return f"{st['est']:+.{k}f} ({st['run_se']:.{k}f}, {st['cluster_se']:.{k}f}, {st['strat_se']:.{k}f})"


def stage_show(args):
    res = H.load_json(args.out)
    if "items" in res:
        show_items(res["items"])
    for rg, r in res.get("regimes", {}).items():
        print(f"\n### {rg}: {r['runs']} runs, {r['appearances']} appearances "
              f"({r['matharena_appearances']} matharena, in {r['runs_with_matharena']} runs); "
              f"replayed {r['runs_replayed']}")
        print("ALC " + ", ".join(f"{v} {x['mean']:.5f} ({x['run_se']:.5f})" for v, x in r["alc"].items()))
        print(f"checks {r['checks']}")
        for comp, c in r["compare"].items():
            print(f"\n{comp}: difference (run SE, cluster SE, stratified SE)")
            print("| subset | n | per appearance ALC | contribution to ALC | B0 per appearance |")
            print("|---|---|---|---|---|")
            for s in SUBSETS:
                if s in c:
                    print(f"| {s} | {c[s]['appearances']} | {fmt(c[s]['per_appearance'])} | "
                          f"{fmt(c[s]['contribution'])} | {fmt(c[s]['by_budget'][0])} |")
            print("budget contributions (all): " + " ".join(
                f"B{BUDGETS[b]} {c['all']['budget_contribution'][b]:+.6f}" for b in range(6)))
            if "matharena" in c:
                print("matharena per budget: " + " ".join(
                    f"B{BUDGETS[b]} {c['matharena']['by_budget'][b]['est']:+.5f}" for b in range(6)))
            pc = c.get("predictions")
            if pc:
                print("changed predictions per budget (direct / indirect of all responses): " + " ".join(
                    f"B{BUDGETS[b]} {pc['changed_direct'][b]:.0f}/{pc['changed_indirect'][b]:.0f} of "
                    f"{pc['responses'][b]:.0f}" for b in range(6)))
                print("mean |change| direct / indirect: " + " ".join(
                    f"B{BUDGETS[b]} {pc['mean_abs_direct'][b]:.4f}/{pc['mean_abs_indirect'][b]:.4f}"
                    for b in range(6)))
                print("max |change| direct / indirect: " + " ".join(
                    f"B{BUDGETS[b]} {pc['max_abs_direct'][b]:.4f}/{pc['max_abs_indirect'][b]:.4f}"
                    for b in range(6)))
        for kind, cal in r["direct_calibration"].items():
            print(f"\n{kind} targets: {cal['responses']} responses, rate {cal['rate']:.3f}; mean prediction "
                  + "; ".join(f"{v} " + " ".join(f"{x:.3f}" for x in xs) for v, xs in cal["mean_pred"].items()))
    if res.get("losses") is not None:
        print("\npositive (losing) point estimates:")
        for x in res["losses"]:
            print(f"  {x['regime']} {x['comparison']} {x['subset']} {x['measure']}: {x['est']:+.6f} "
                  f"(max SE {x['max_se']:.6f}, z {x['z'] if x['z'] is None else round(x['z'], 2)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("items", "run", "summary", "show"))
    ap.add_argument("--regimes", nargs="+", default=list(PLAN), choices=list(PLAN))
    ap.add_argument("--n", type=int, default=None, help="runs 0..n-1 of each regime instead of PLAN's")
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--harness-rows", default=H.ROWS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--verify", type=int, default=3,
                    help="replay every variant, and hier unpatched, on runs 0..verify-1")
    ap.add_argument("--boots", type=int, default=H.BOOTS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    {"items": stage_items, "run": stage_run, "summary": stage_summary, "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
