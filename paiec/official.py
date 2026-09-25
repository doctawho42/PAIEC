"""Replica of the official PAIEC streaming protocol.

Checked against the organisers' streaming client
(third_party/paiec_baseline/tools/streaming_ingestion.py) and the competition
page. docs/protocol.md separates what is verified from what is still a
parameter here. paiec/evaluator.py is the older pair-major reconstruction, kept
so earlier numbers stay reproducible; its numbers are not comparable to these.

What a predictor has to survive, and where each piece lives below:

  * Scoring is budget-major (run_official). Every pair is evaluated at budget 0
    before any label exists; acquisition then resumes until each pair holds up
    to one label, every pair is evaluated at budget 1, and so on up to 31.
  * At a checkpoint one shared `labeled` goes to every target: each pair's
    first B labels, concatenated over all pairs of the run, whatever their
    subject or benchmark (_evaluate).
  * Evaluation workers are rebuilt from the submission at every checkpoint, so
    nothing kept in memory survives from one budget to the next. Acquisition-
    time predictions come from one long-lived process instead (_Runtime).
  * predict() sees only the official format: eight subject fields, four item
    fields and an anonymous benchmark_id (_slots).
  * A formative run holds at most 1,000 subject-item pairs, so only 5 to 12
    subject-benchmark pairs (sample_run).
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import numbers
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from paiec import data as D
from paiec.evaluator import BUDGETS, MAX_LABELS, MIN_ITEMS, WEIGHTS, Pair, stable_hash

CAP = 1000
SPLIT_SCOPES = ("pair", "benchmark")
_CLIENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "third_party", "paiec_baseline", "tools", "streaming_ingestion.py")


class SubmissionError(ValueError):
    """A failure the platform reports by exit code: 40 predict raised, 41 a call
    ran over its timeout, 42 the value returned is not a probability."""

    def __init__(self, code, message):
        super().__init__(f"[{code}] {message}")
        self.code = code


def alc(by_budget) -> float:
    return float(sum(w * by_budget[B] for w, B in zip(WEIGHTS, BUDGETS)))


# --- which pairs, which items ------------------------------------------------

def eligible(pairs, min_items=MIN_ITEMS):
    """Pairs the test can contain. Only binary responses are eligible, so
    non-binary ones are dropped rather than rounded and fraction-valued
    benchmarks (mmdocrag) are dropped whole; what remains needs >= min_items
    distinct items."""
    out = []
    for p in pairs:
        if p.benchmark_id in D.FRACTIONAL:
            continue
        kept = [r for r in p.responses if r.label in (0, 1)]
        if len(kept) < len(p.responses):
            p = Pair(p.subject, p.subject_id, p.benchmark_id, kept)
        if len(p.item_keys) >= min_items:
            out.append(p)
    return out


def split(pair, seed=0, scope="pair"):
    """Persistent 50/50 split of a pair's distinct items: (acquisition, evaluation).

    Items are ranked by a digest, never by Python's per-process salted hash(),
    so the split is identical in every process, run and budget, and every
    response of an item lands on the same side. With scope='pair' each subject
    gets its own split of a benchmark, and another subject's acquired labels
    can sit on this subject's evaluation items. With 'benchmark' the rank
    ignores the subject, so subjects that attempted the same items share one
    split and that overlap never happens. Which one the coordinator uses is
    unknown."""
    if scope not in SPLIT_SCOPES:
        raise ValueError(f"split scope must be one of {SPLIT_SCOPES}, got {scope!r}")
    who = pair.subject_id if scope == "pair" else ""
    keys = sorted(pair.item_keys,
                  key=lambda k: stable_hash(seed, "split", who, pair.benchmark_id, k))
    half = len(keys) // 2
    return keys[:half], keys[half:]


def stream(pair, acq_keys, seed=0):
    """The fixed order in which a pair's acquisition candidates arrive."""
    return sorted(acq_keys,
                  key=lambda k: stable_hash(seed, "order", pair.subject_id, pair.benchmark_id, k))


# --- what one run contains ----------------------------------------------------

WEIGHTINGS = ("benchmark", "pair")


def sample_run(pairs, rng, *, cap=CAP, n_pairs=None, concentration=0.0,
               subject_affinity=0.0, alloc="proportional", min_items=MIN_ITEMS,
               weighting="benchmark"):
    """Draw the pairs and items of one formative run: [(pair, frozenset of item keys)].

    A run holds at most `cap` unique subject-item pairs across both pools and
    every pair keeps >= min_items items, so it has at most cap // min_items
    pairs; `n_pairs` defaults to a draw from 5..12. Pairs are drawn one at a
    time. With weighting='benchmark' (the default): first a benchmark,
    weighted 1 + concentration * (pairs already drawn from it), so 0 spreads
    the run over benchmarks and a large value piles it onto one; then a pair
    of that benchmark, weighted 1 + subject_affinity * (subject already
    drawn), which sets how often a subject recurs on several benchmarks. With
    weighting='pair' every remaining pair is drawn directly, weighted by the
    product of the same two factors, so at the defaults pairs are uniform and
    a benchmark enters a run in proportion to its number of pairs (a
    single-pair benchmark rarely). Which one the organisers use is unknown;
    'benchmark' keeps earlier runs reproducible. If the pairs hold more than
    `cap` items, each keeps a uniform random subset of min(n, max(min_items,
    t*w)) items with the largest t that fits, w being the pair's size
    ('proportional') or 1 ('equal'). The site's illustrative feedback (5
    pairs, 4 subjects, 4 benchmarks, 46-181 evaluated items per pair, about
    960 items in all) looks like proportional cuts that fill the cap.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    pool = eligible(pairs, min_items)
    if n_pairs is None:
        n_pairs = int(rng.integers(5, 13))
    if n_pairs * min_items > cap:
        raise ValueError(f"{n_pairs} pairs of >= {min_items} items cannot fit a cap of {cap}")
    left, chosen = list(range(len(pool))), []
    for _ in range(min(n_pairs, len(pool))):
        drawn = Counter(pool[i].benchmark_id for i in chosen)
        subjects = {pool[i].subject_id for i in chosen}
        if weighting == "pair":
            w = np.array([(1.0 + concentration * drawn[pool[i].benchmark_id])
                          * (1.0 + subject_affinity * (pool[i].subject_id in subjects))
                          for i in left])
            pick = left[rng.choice(len(left), p=w / w.sum())]
        else:
            benches = sorted({pool[i].benchmark_id for i in left})
            w = np.array([1.0 + concentration * drawn[b] for b in benches])
            b = benches[rng.choice(len(benches), p=w / w.sum())]
            cand = [i for i in left if pool[i].benchmark_id == b]
            w = np.array([1.0 + subject_affinity * (pool[i].subject_id in subjects) for i in cand])
            pick = cand[rng.choice(len(cand), p=w / w.sum())]
        chosen.append(pick)
        left.remove(pick)
    picked = [pool[i] for i in chosen]
    run = []
    for p, k in zip(picked, _allocate([len(p.item_keys) for p in picked], cap, min_items, alloc)):
        keys = p.item_keys
        if k < len(keys):
            keys = [keys[i] for i in rng.choice(len(keys), k, replace=False)]
        run.append((p, frozenset(keys)))
    return run


def _allocate(sizes, cap, min_items, alloc):
    """Items kept per pair so the run fits the cap, none below min_items."""
    n = np.asarray(sizes)
    if n.sum() <= cap:
        return n.tolist()
    if alloc not in ("proportional", "equal"):
        raise ValueError(f"alloc must be 'proportional' or 'equal', got {alloc!r}")
    w = n.astype(float) if alloc == "proportional" else np.ones(len(n))
    kept = lambda t: np.minimum(n, np.maximum(min_items, np.floor(t * w))).astype(int)
    lo, hi = 0.0, float((n / w).max())
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kept(mid).sum() <= cap else (lo, mid)
    return kept(lo).tolist()


def dense_run(pairs, benchmark, min_items=MIN_ITEMS):
    """Every eligible pair of one benchmark with all its items and no cap: the
    most cross-subject evidence a run could carry, the upper end for pooling."""
    return [(p, frozenset(p.item_keys)) for p in eligible(pairs, min_items)
            if p.benchmark_id == benchmark]


def run_benchmarks(run):
    """Real benchmark names in a run. For the caller only, to fit a prior that
    excludes them; predict() only ever sees anonymous ids."""
    return sorted({_entry(e)[0].benchmark_id for e in run})


def training_pairs(pairs, run):
    """Pairs whose benchmark is not in the run: what a leave-benchmark-out prior
    may be fitted on."""
    held = set(run_benchmarks(run))
    return [p for p in pairs if p.benchmark_id not in held]


def _entry(e):
    return e if isinstance(e, tuple) else (e, None)


# --- acquisition hooks -------------------------------------------------------

class Hook:
    """How the platform calls labeling.acquisition_function.

    Mirrors the organisers' AcquisitionHook. A function accepting prediction,
    labeled and context by name, through **kwargs, or as four positionals gets
    streaming decisions and must return a native bool (np.bool_ is an error). A
    one-argument function is 'legacy_rank': it scores every candidate once,
    with no labels, and the top-scored are revealed. None is the platform's
    random policy. The signature alone picks the mode; a TypeError raised
    inside the function is never retried with another call shape.
    """
    NAMES = ("prediction", "labeled", "context")

    def __init__(self, function, deepcopy=True):
        self.function = function
        self.copy = copy.deepcopy if deepcopy else (lambda x: x)
        self.mode, self.positional, self.keywords = "random", False, ()
        if function is None:
            return
        try:
            sig = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise ValueError("cannot inspect acquisition_function's signature") from exc
        params = sig.parameters
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        named = tuple(n for n in self.NAMES if var_kw or (
            n in params and params[n].kind is not inspect.Parameter.POSITIONAL_ONLY))
        shapes = [(False, self.NAMES), (True, ())]
        if named and named != self.NAMES:
            shapes.append((False, named))
        for positional, names in shapes:
            args = (None,) * 4 if positional else (None,)
            if _binds(sig, *args, **dict.fromkeys(names)):
                self.mode, self.positional, self.keywords = "stream", positional, names
                return
        if not _binds(sig, None):
            raise ValueError("acquisition_function must accept input and optional "
                             "prediction, labeled, context")
        self.mode = "legacy_rank"

    @property
    def needs_prediction(self):
        return self.positional or "prediction" in self.keywords

    def __call__(self, input, prediction=None, labeled=None, context=None):
        c = self.copy
        if self.mode == "legacy_rank":
            value = self.function(c(input))
            if isinstance(value, bool) or not isinstance(value, numbers.Real) \
                    or not math.isfinite(float(value)):
                raise ValueError("a one-argument acquisition_function must return a finite number")
            return float(value)
        values = {"prediction": prediction, "labeled": labeled, "context": context}
        if self.positional:
            out = self.function(c(input), prediction, c(labeled), c(context))
        else:
            out = self.function(c(input), **{k: c(values[k]) for k in self.keywords})
        if type(out) is not bool:
            raise ValueError("a streaming acquisition_function must return a bool, "
                             f"got {type(out).__name__}")
        return out


def _binds(sig, *args, **kwargs):
    try:
        sig.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def official_client():
    """The organisers' streaming client module when it is on disk, else None.
    It is stdlib-only and defines functions without running anything."""
    if not os.path.exists(_CLIENT):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_paiec_official_client", _CLIENT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def make_hook(function, deepcopy=True):
    """The organisers' AcquisitionHook when available (it always deep-copies),
    else the mirror above."""
    client = official_client()
    if client is not None and deepcopy:
        return client.AcquisitionHook(function)
    return Hook(function, deepcopy)


def default_policy(context, input) -> bool:
    """The platform's choice when there is no labeling.py, bit for bit.

    A uniform draw from sha256 of the canonical JSON of [context, input]; take
    the candidate iff it is below labels_remaining / items_remaining. That draws
    a uniformly random subset and takes every remaining candidate once the pool
    is no longer than the allowance, so the 31 labels fill whenever they can."""
    key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return u < min(1, context["labels_remaining"] / context["items_remaining"])


# --- the engine -----------------------------------------------------------------

@dataclass
class _Slot:
    """One subject-benchmark pair of a run, already in official format."""
    subject_id: str        # anonymous, as in context and feedback rows
    benchmark_id: str      # anonymous, as in item dicts
    name: str              # real benchmark name; never passed to predict
    candidates: list       # [(item_key, input, label)] in stream order
    targets: list          # [(input, label, dedupe key)], one per evaluation response
    n_items: int           # distinct evaluated items
    cursor: int = 0
    acquired: list = field(default_factory=list)    # [[subject, item], label]
    keys: list = field(default_factory=list)        # item keys of `acquired`
    ranked: list | None = None                      # legacy_rank: candidate indices, best first
    preds: dict = field(default_factory=dict)       # budget -> prediction per target


def _canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _slots(run, seed, scope):
    slots, seen = [], set()
    for entry in run:
        pair, items = _entry(entry)
        sid = D.anon_id("subject", pair.subject_id)
        bid = D.anon_id("benchmark", pair.benchmark_id)
        if (sid, bid) in seen:
            raise ValueError(f"pair {pair.subject_id}/{pair.benchmark_id} is duplicated in the "
                             "run or its anonymous id collides with another")
        seen.add((sid, bid))
        subject = D.official_subject(pair.subject)
        by = pair.by_item()
        inputs = {k: [subject, D.official_item(rs[0].item, bid)]
                  for k, rs in by.items() if items is None or k in items}
        acq, ev = split(pair, seed, scope)
        cands = [(k, inputs[k], int(by[k][0].label)) for k in stream(pair, acq, seed) if k in inputs]
        ev = [k for k in ev if k in inputs]
        targets = []
        for k in ev:
            key = _canon(inputs[k])
            targets += [(inputs[k], int(r.label), key) for r in by[k]]
        slots.append(_Slot(sid, bid, pair.benchmark_id, cands, targets, len(ev)))
    return slots


class _Runtime:
    """What the platform wraps around the submission: fresh instances, deep
    copies, wall-clock per call, and validation of what comes back."""

    def __init__(self, factory, deepcopy, timeout):
        self.factory, self.deepcopy, self.timeout = factory, deepcopy, timeout
        self.times = {"setup": [], "acquisition": [], "evaluation": [], "hook": []}
        self.reused = 0
        self._main = None

    def load(self):
        t = time.perf_counter()
        fn = self.factory()
        self.times["setup"].append(time.perf_counter() - t)
        return fn

    @property
    def main(self):
        """The long-lived instance behind acquisition-time predictions, loaded on
        first use and kept for the whole run."""
        if self._main is None:
            self._main = self.load()
        return self._main

    def predict(self, fn, input, labeled, kind):
        if self.deepcopy:
            input, labeled = _fresh(input, labeled)
        t = time.perf_counter()
        try:
            value = fn(input, labeled)
        except Exception as exc:
            raise SubmissionError(40, f"predict raised {type(exc).__name__}: {exc}") from exc
        dt = time.perf_counter() - t
        self.times[kind].append(dt)
        if self.timeout is not None and dt > self.timeout:
            raise SubmissionError(41, f"predict took {dt:.3f}s, over the {self.timeout}s timeout")
        return _probability(value)

    def hook(self, hook, *args):
        t = time.perf_counter()
        out = hook(*args)
        self.times["hook"].append(time.perf_counter() - t)
        return out

    def summary(self, wall):
        out = {"wall_s": wall, "reused": self.reused}
        for kind, ts in self.times.items():
            out[f"{kind}_calls"] = len(ts)
            out[f"{kind}_mean_s"] = float(np.mean(ts)) if ts else 0.0
            out[f"{kind}_max_s"] = float(max(ts, default=0.0))
        return out


def _fresh(input, labeled):
    """A deep copy for the one shape these take (string-valued dicts, int
    labels), twenty times faster than copy.deepcopy. As with the platform's
    JSON transport, no two entries share a dict, so a predictor that mutates or
    caches by identity sees what it would see there."""
    return ([dict(input[0]), dict(input[1])],
            [[[dict(s), dict(i)], y] for (s, i), y in labeled])


def _probability(value):
    """float() it as the runtime does; failures, NaN, infinities and values
    outside [0, 1] are code 42."""
    try:
        p = float(value)
    except Exception as exc:
        raise SubmissionError(42, f"prediction {value!r} does not convert to float") from exc
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise SubmissionError(42, f"prediction must be a finite probability in [0, 1], got {p}")
    return p


def _acquire(slots, budget, hook, rt):
    """Resume every pair's stream, in run order, until it holds min(budget, 31)
    labels or runs dry. `labeled` is everything revealed so far, all pairs."""
    goal = min(budget, MAX_LABELS)
    if hook.mode == "legacy_rank":
        for s in slots:
            if s.ranked is None:
                score = [rt.hook(hook, inp) for _, inp, _ in s.candidates]
                s.ranked = sorted(range(len(score)), key=lambda i: (-score[i], i))
            for i in s.ranked[len(s.acquired):goal]:
                key, inp, y = s.candidates[i]
                s.acquired.append([inp, y])
                s.keys.append(key)
        return
    for s in slots:
        while len(s.acquired) < goal and s.cursor < len(s.candidates):
            key, inp, y = s.candidates[s.cursor]
            context = {"subject_id": s.subject_id, "benchmark_id": s.benchmark_id,
                       "labels_acquired": len(s.acquired),
                       "labels_remaining": MAX_LABELS - len(s.acquired),
                       "max_labels": MAX_LABELS,
                       "items_remaining": len(s.candidates) - s.cursor}
            s.cursor += 1
            if hook.mode == "random":
                take = default_policy(context, inp)
            else:
                labeled = [e for t in slots for e in t.acquired]
                prediction = (rt.predict(rt.main, inp, labeled, "acquisition")
                              if hook.needs_prediction else None)
                take = rt.hook(hook, inp, prediction, labeled, context)
            if take:
                s.acquired.append([inp, y])
                s.keys.append(key)


def _evaluate(slots, budget, rt, workers, dedupe):
    """One checkpoint: fresh instances, one shared `labeled`, and each distinct
    input predicted once, its prediction reused for every repeated response."""
    labeled = [e for s in slots for e in s.acquired[:budget]]
    inputs, where, index = [], {}, []
    for s in slots:
        for inp, _, key in s.targets:
            j = where.setdefault(key, len(inputs)) if dedupe else len(inputs)
            if j == len(inputs):
                inputs.append(inp)
            index.append(j)
    shards = [rt.load() for _ in range(max(1, min(workers, len(inputs))))]
    preds = [rt.predict(shards[j % len(shards)], inp, labeled, "evaluation")
             for j, inp in enumerate(inputs)]
    rt.reused += len(index) - len(inputs)
    at = 0
    for s in slots:
        s.preds[budget] = [preds[j] for j in index[at:at + len(s.targets)]]
        at += len(s.targets)


def ece(p, y, bins=10) -> float:
    """Response-weighted mean gap between average prediction and success rate
    over equal-width probability bins, as in the formative feedback."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    gap = np.abs(np.bincount(b, p, bins) - np.bincount(b, y, bins))
    return float(gap.sum() / len(p))


def run_official(run, model_factory, acquisition_function=None, *, seed=0,
                 split_scope="pair", deepcopy=True, workers=1, dedupe=True,
                 call_timeout=None):
    """Score one run the way the platform does.

    `run` is a list of pairs or of (pair, item keys) from sample_run/dense_run.
    `model_factory()` returns a fresh predict callable; it is called once for the
    acquisition-time instance (only if the hook asks for predictions) and once
    per worker at every checkpoint, targets dealt round-robin across workers.
    `seed` salts the split and stream order to emulate other hidden splits; 0 is
    the persistent one. deepcopy=False skips the per-call copies of input and
    labeled, for speed, when the predictor is known not to mutate them.

    Returns a dict: 'brier' and 'ece' by budget plus 'ALC' (equal average over
    pairs), 'rows' (one per pair, anonymous ids, as in formative feedback),
    'timing' (per-call wall-clock by kind: setup, acquisition, evaluation,
    hook), 'acquisition_mode', and for the caller only 'benchmarks' (real
    names), 'names' (anonymous id -> name) and 'acquired' (item keys in order).
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    t0 = time.perf_counter()
    slots = _slots(run, seed, split_scope)
    hook = make_hook(acquisition_function, deepcopy)
    rt = _Runtime(model_factory, deepcopy, call_timeout)
    for B in BUDGETS:
        if B:
            _acquire(slots, B, hook, rt)
        _evaluate(slots, B, rt, workers, dedupe)

    rows = []
    for s in slots:
        y = np.array([t[1] for t in s.targets], float)
        brier = {B: float(np.mean((np.array(s.preds[B]) - y) ** 2)) for B in BUDGETS}
        cal = {B: ece(s.preds[B], y) for B in BUDGETS}
        rows.append({"subject_id": s.subject_id, "benchmark_id": s.benchmark_id,
                     "n_items": s.n_items, "n_responses": len(y), "n_labels": len(s.acquired),
                     "brier": brier, "ALC": alc(brier), "ece": cal, "ece_alc": alc(cal)})
    brier = {B: float(np.mean([r["brier"][B] for r in rows])) for B in BUDGETS}
    cal = {B: float(np.mean([r["ece"][B] for r in rows])) for B in BUDGETS}
    brier["ALC"], cal["ALC"] = alc(brier), alc(cal)
    return {"brier": brier, "ece": cal, "rows": rows,
            "timing": rt.summary(time.perf_counter() - t0),
            "acquisition_mode": hook.mode,
            "benchmarks": sorted({s.name for s in slots}),
            "names": {s.benchmark_id: s.name for s in slots},
            "acquired": {(s.subject_id, s.benchmark_id): list(s.keys) for s in slots}}
