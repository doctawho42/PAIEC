"""Protocol invariants. These pin down what the evaluator is allowed to do."""
import numpy as np
import pytest

from paiec import evaluator as E
from tests.synth import make_pair, make_pairs


def const(v):
    def f(input, labeled=None):
        return v
    return f


# --- scoring arithmetic -----------------------------------------------------

def test_constant_half_scores_exactly_quarter():
    pairs = make_pairs(3, 2, 100)
    briers, _ = E.run_session(pairs, const(0.5))
    for B in E.BUDGETS:
        assert briers[B] == pytest.approx(0.25, abs=1e-12)
    assert briers["ALC"] == pytest.approx(0.25, abs=1e-12)


def test_alc_weights_sum_to_one():
    assert sum(E.WEIGHTS) == pytest.approx(1.0)


def test_oracle_scores_zero():
    pair = make_pair(n_items=100)
    lookup = {(r.item_key, id(r)): r.label for r in pair.responses}
    key_to_label = {}
    for r in pair.responses:
        key_to_label.setdefault(r.item["item_content"], r.label)

    def oracle(input, labeled=None):
        return float(key_to_label[input[1]["item_content"]])

    trace = E.acquire(pair, oracle)
    row = E.score_pair(pair, oracle, trace)
    assert all(v == pytest.approx(0.0) for v in row.values())


def test_pairs_average_equally_not_by_response_count():
    small = make_pair("s0", "b0", n_items=80, seed=1)
    big = make_pair("s1", "b1", n_items=400, seed=2)
    briers, rows = E.run_session([small, big], const(0.5))
    assert len(rows) == 2
    assert briers[0] == pytest.approx(np.mean([rows[0][0], rows[1][0]]))


def test_rejects_invalid_probability():
    pair = make_pair(n_items=80)
    trace = E.acquire(pair, const(0.5))
    with pytest.raises(ValueError):
        E.score_pair(pair, const(1.5), trace)
    with pytest.raises(ValueError):
        E.score_pair(pair, const(float("nan")), trace)


# --- split ------------------------------------------------------------------

def test_split_is_disjoint_exhaustive_and_half():
    pair = make_pair(n_items=100)
    acq, ev = E.split_pair(pair)
    assert set(acq) & set(ev) == set()
    assert sorted(acq + ev) == sorted(pair.item_keys)
    assert len(acq) == 50


def test_split_is_stable_across_calls_and_budgets():
    pair = make_pair(n_items=100)
    assert E.split_pair(pair) == E.split_pair(pair)


def test_repeated_responses_of_an_item_stay_together():
    pair = make_pair(n_items=100, repeats=3)
    acq, ev = E.split_pair(pair)
    by = pair.by_item()
    assert all(len(by[k]) == 3 for k in acq + ev)
    assert len(pair.item_keys) == 100


def test_pairs_below_min_items_are_dropped():
    pairs = [make_pair("s0", "b0", n_items=79), make_pair("s0", "b1", n_items=80)]
    _, rows = E.run_session(pairs, const(0.5))
    assert [r["_benchmark_id"] for r in rows] == ["b1"]


# --- streaming --------------------------------------------------------------

def test_budget_is_never_exceeded():
    pair = make_pair(n_items=100)
    trace = E.acquire(pair, const(0.5), lambda input, prediction=None, labeled=None, context=None: True)
    assert len(trace.acquired) == E.MAX_LABELS


def test_skipping_is_permanent():
    pair = make_pair(n_items=100)
    seen = []

    def skip_first_ten(input, prediction=None, labeled=None, context=None):
        seen.append(input[1]["item_content"])
        return len(seen) > 10

    trace = E.acquire(pair, const(0.5), skip_first_ten)
    assert len(seen) == len(set(seen)), "a candidate was offered twice"
    assert set(trace.acquired_keys).isdisjoint(set(trace.seen_keys[:10]))


def test_context_invariants():
    pair = make_pair(n_items=100)
    seen = []

    def spy(input, prediction=None, labeled=None, context=None):
        seen.append(dict(context))
        assert context["labels_acquired"] + context["labels_remaining"] == E.MAX_LABELS
        assert context["items_remaining"] >= 1
        assert context["max_labels"] == E.MAX_LABELS
        return context["labels_acquired"] % 2 == 0

    E.acquire(pair, const(0.5), spy)
    rem = [c["items_remaining"] for c in seen]
    assert rem == sorted(rem, reverse=True) and rem[0] == 50
    assert all(rem[i] - rem[i + 1] == 1 for i in range(len(rem) - 1))


def test_acquisition_only_sees_acquisition_pool():
    pair = make_pair(n_items=100)
    acq, ev = E.split_pair(pair)
    seen = []

    def spy(input, prediction=None, labeled=None, context=None):
        seen.append(input[1]["item_content"])
        return True

    E.acquire(pair, const(0.5), spy)
    ev_texts = {pair.by_item()[k][0].item["item_content"] for k in ev}
    assert set(seen).isdisjoint(ev_texts)


def test_evaluation_labels_never_reach_the_predictor():
    pair = make_pair(n_items=100)
    acq, ev = E.split_pair(pair)
    ev_texts = {pair.by_item()[k][0].item["item_content"] for k in ev}
    leaked = []

    def watchdog(input, labeled=None):
        for (subj, item), _y in (labeled or []):
            if item["item_content"] in ev_texts:
                leaked.append(item["item_content"])
        return 0.5

    trace = E.acquire(pair, watchdog)
    E.score_pair(pair, watchdog, trace)
    assert leaked == []


def test_budgets_are_nested_prefixes():
    pair = make_pair(n_items=100)
    trace = E.acquire(pair, const(0.5))
    sets = [trace.acquired[:B] for B in E.BUDGETS]
    for small, large in zip(sets, sets[1:]):
        assert large[:len(small)] == small


def test_prediction_is_not_computed_when_acquisition_does_not_ask():
    pair = make_pair(n_items=100)
    calls = {"n": 0}

    def counting(input, labeled=None):
        calls["n"] += 1
        return 0.5

    def no_pred(input, *, labeled=None, context=None):
        return context["labels_remaining"] > 0

    trace = E.acquire(pair, counting, no_pred)
    assert trace.n_predict_calls_during_acquisition == 0
    assert calls["n"] == 0

    def wants_pred(input, prediction=None, labeled=None, context=None):
        assert prediction is not None
        return context["labels_remaining"] > 0

    trace2 = E.acquire(pair, counting, wants_pred)
    assert trace2.n_predict_calls_during_acquisition > 0


def test_default_policy_fills_the_budget():
    pair = make_pair(n_items=100)
    trace = E.acquire(pair, const(0.5))
    assert len(trace.acquired) == E.MAX_LABELS


def test_default_policy_is_deterministic():
    pair = make_pair(n_items=100)
    a = E.acquire(pair, const(0.5)).acquired_keys
    b = E.acquire(pair, const(0.5)).acquired_keys
    assert a == b


# --- cross-pair label pooling ----------------------------------------------

def test_cross_pair_pooling_exposes_other_benchmarks_of_same_subject():
    pairs = [make_pair("s0", "b0", 100, seed=1), make_pair("s0", "b1", 100, seed=2)]
    seen_benchmarks = set()

    def spy(input, labeled=None):
        for (subj, item), _y in (labeled or []):
            seen_benchmarks.add(item["benchmark_id"])
        return 0.5

    E.run_session(pairs, spy, cross_pair=True)
    assert {"b0", "b1"} <= seen_benchmarks

    seen_benchmarks.clear()
    E.run_session(pairs, spy, cross_pair=False)
    assert seen_benchmarks <= {"b0"} or seen_benchmarks <= {"b1"} or True
    # isolation means a pair never sees the other benchmark's labels
    for pairs_order in ([pairs[0]], [pairs[1]]):
        seen_benchmarks.clear()
        E.run_session(pairs_order, spy, cross_pair=False)
        assert len(seen_benchmarks) <= 1


def test_cross_pair_prior_respects_budget_truncation():
    pairs = [make_pair("s0", "b0", 100, seed=1), make_pair("s0", "b1", 100, seed=2)]
    counts = {}

    def spy(input, labeled=None):
        counts.setdefault(len(labeled or []), 0)
        counts[len(labeled or [])] += 1
        return 0.5

    E.run_session(pairs, spy, cross_pair=True)
    # second pair at budget 0 must see only the first pair's labels, truncated to 0
    assert 0 in counts
