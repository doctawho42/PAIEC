"""Replica of the PAIEC streaming evaluator.

Protocol, as published on aimslab.stanford.edu/competition and as implied by the
baseline repository:

  * For each subject-benchmark pair with at least MIN_ITEMS distinct items, the
    items are split 50/50 into an acquisition pool and an evaluation pool. The
    split is fixed across budgets and submissions. All recorded responses for an
    item stay together.
  * Candidates from the acquisition pool arrive one at a time in a fixed random
    order. `acquisition_function` returns True to reveal one recorded response
    or False to skip the item permanently.
  * One acquisition trajectory yields nested label sets. The submission is
    scored at budgets 0, 1, 3, 7, 15, 31 by truncating that trajectory.
  * Brier is averaged over evaluation responses within a pair, then equally
    across pairs. ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31.

Assumptions marked ASSUMPTION are not pinned down by the published text. Each is
a flag so we can measure how much it matters instead of guessing once.

Complexity per pair: O(n_acq) acquisition calls plus O(sum_B n_eval) = O(6*n_eval)
prediction calls. Laplace-style predictors refit per budget, not per item, if they
cache on the identity of the labelled set.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, field

import numpy as np

def stable_hash(*parts) -> int:
    """Python salts str.__hash__ per process, which silently re-randomises the
    50/50 split and the candidate order between runs. The protocol requires a
    split that is fixed across budgets and submissions, so hash by digest."""
    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
MAX_LABELS = 31
MIN_ITEMS = 80


@dataclass(frozen=True)
class Response:
    """One recorded response. Several responses may share an item_key."""
    item_key: str
    item: dict
    label: int


@dataclass
class Pair:
    subject: dict
    subject_id: str
    benchmark_id: str
    responses: list[Response]

    @property
    def item_keys(self) -> list[str]:
        seen, out = set(), []
        for r in self.responses:
            if r.item_key not in seen:
                seen.add(r.item_key)
                out.append(r.item_key)
        return out

    def by_item(self) -> dict[str, list[Response]]:
        d: dict[str, list[Response]] = {}
        for r in self.responses:
            d.setdefault(r.item_key, []).append(r)
        return d


def split_pair(pair: Pair, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic 50/50 item split, stable across budgets and submissions."""
    keys = pair.item_keys
    h = stable_hash(pair.subject_id, pair.benchmark_id) % (2**31)
    rng = np.random.default_rng((seed * 1_000_003 + h) % (2**32))
    perm = rng.permutation(len(keys))
    half = len(keys) // 2
    acq = [keys[i] for i in perm[:half]]
    ev = [keys[i] for i in perm[half:]]
    return acq, ev


def default_acquisition(input, prediction=None, labeled=None, context=None) -> bool:
    """Evaluator's default deterministic random policy: uniform random subset.

    Taking with probability labels_remaining / items_remaining yields a uniformly
    random subset of the requested size, and fills the budget exactly when the
    pool runs short.
    """
    n = context["items_remaining"]
    k = context["labels_remaining"]
    if k <= 0:
        return False
    if k >= n:
        return True
    seed = stable_hash(context["subject_id"], context["benchmark_id"],
                       context["labels_acquired"], n) % (2**32)
    return np.random.default_rng(seed).random() < k / n


def _wants_prediction(fn) -> bool:
    """The evaluator computes a prediction for acquisition only if asked for it."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if "prediction" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values())


def _call_acquisition(fn, inp, prediction, labeled, context) -> bool:
    params = inspect.signature(fn).parameters
    kwargs = {}
    if "labeled" in params:
        kwargs["labeled"] = labeled
    if "context" in params:
        kwargs["context"] = context
    if "prediction" in params:
        kind = params["prediction"].kind
        if kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs["prediction"] = prediction
            return bool(fn(inp, **kwargs))
        return bool(fn(inp, prediction, **kwargs))
    return bool(fn(inp, **kwargs))


@dataclass
class Trace:
    """Everything the evaluator observed, for auditing and for tests."""
    acquired: list = field(default_factory=list)      # [[subject, item], label]
    acquired_keys: list = field(default_factory=list)
    seen_keys: list = field(default_factory=list)
    n_predict_calls_during_acquisition: int = 0


def acquire(pair: Pair, predict, acquisition_function=None, *, seed: int = 0,
            prior_labeled: list | None = None, max_labels: int = MAX_LABELS) -> Trace:
    """Run the online streaming phase for one pair and return its trajectory."""
    acquisition_function = acquisition_function or default_acquisition
    acq_keys, _ = split_pair(pair, seed)
    by_item = pair.by_item()

    order_rng = np.random.default_rng(
        (seed * 7 + stable_hash(pair.subject_id, pair.benchmark_id, "order")) % (2**32))
    order = [acq_keys[i] for i in order_rng.permutation(len(acq_keys))]

    wants_pred = _wants_prediction(acquisition_function)
    trace = Trace()
    base = list(prior_labeled or [])

    for pos, key in enumerate(order):
        trace.seen_keys.append(key)
        n_left = len(order) - pos
        k_left = max_labels - len(trace.acquired)
        if k_left <= 0:
            break
        cand = by_item[key][0]
        inp = [pair.subject, cand.item]
        labeled = base + trace.acquired
        context = {
            "subject_id": pair.subject_id,
            "benchmark_id": pair.benchmark_id,
            "labels_acquired": len(trace.acquired),
            "labels_remaining": k_left,
            "max_labels": max_labels,
            "items_remaining": n_left,
        }
        prediction = None
        if wants_pred:
            prediction = float(predict(inp, labeled))
            trace.n_predict_calls_during_acquisition += 1
        if _call_acquisition(acquisition_function, inp, prediction, labeled, context):
            trace.acquired.append([[pair.subject, cand.item], int(cand.label)])
            trace.acquired_keys.append(key)
    return trace


def score_pair(pair: Pair, predict, trace: Trace, *, seed: int = 0,
               prior_by_budget: dict[int, list] | None = None) -> dict[int, float]:
    """Brier per budget on the evaluation pool, using nested prefixes."""
    _, ev_keys = split_pair(pair, seed)
    by_item = pair.by_item()
    ev_responses = [r for k in ev_keys for r in by_item[k]]
    out = {}
    for B in BUDGETS:
        labeled = list((prior_by_budget or {}).get(B, [])) + trace.acquired[:B]
        se = 0.0
        for r in ev_responses:
            p = float(predict([pair.subject, r.item], labeled))
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError(f"prediction must be a finite probability in [0,1], got {p}")
            se += (p - r.label) ** 2
        out[B] = se / len(ev_responses)
    return out


def run_session(pairs: list[Pair], predict, acquisition_function=None, *,
                seed: int = 0, cross_pair: bool = True,
                max_labels: int = MAX_LABELS) -> tuple[dict[int, float], list[dict]]:
    """Score a whole submission.

    ASSUMPTION cross_pair: the published text says `labeled` contains entries
    "across sampled subject-benchmark pairs". With cross_pair=True a subject's
    labels from already-processed pairs are visible to later pairs of the same
    subject; with False each pair is isolated. Run both and report the gap.
    """
    eligible = [p for p in pairs if len(p.item_keys) >= MIN_ITEMS]
    per_pair, store = [], {}
    for pair in eligible:
        prior = list(store.get(pair.subject_id, [])) if cross_pair else []
        trace = acquire(pair, predict, acquisition_function, seed=seed,
                        prior_labeled=prior, max_labels=max_labels)
        prior_by_budget = None
        if cross_pair:
            done = store.setdefault(pair.subject_id, [])
            prior_by_budget = {B: list(done) for B in BUDGETS}
            done.extend(trace.acquired)
        row = score_pair(pair, predict, trace, seed=seed, prior_by_budget=prior_by_budget)
        row["_subject_id"] = pair.subject_id
        row["_benchmark_id"] = pair.benchmark_id
        row["_n_labels"] = len(trace.acquired)
        per_pair.append(row)
    briers = {B: float(np.mean([r[B] for r in per_pair])) for B in BUDGETS}
    briers["ALC"] = float(sum(w * briers[B] for w, B in zip(WEIGHTS, BUDGETS)))
    return briers, per_pair


def alc(briers: dict) -> float:
    return float(sum(w * briers[B] for w, B in zip(WEIGHTS, BUDGETS)))
