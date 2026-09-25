"""Invariants of the official-protocol replica, paiec/official.py."""
import contextlib
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import pytest

from paiec import baselines as B
from paiec import data as D
from paiec import official as O
from paiec.evaluator import BUDGETS, MAX_LABELS, Pair, Response
from tests.synth import make_pair, make_pairs

CONTEXT_KEYS = {"subject_id", "benchmark_id", "labels_acquired", "labels_remaining",
                "max_labels", "items_remaining"}


def const(v):
    return lambda: (lambda input, labeled: v)


def recorder():
    """A factory whose every instance logs the `labeled` of each call it serves."""
    instances = []

    def factory():
        seen = []
        instances.append(seen)

        def f(input, labeled):
            seen.append(labeled)
            return 0.5
        return f
    return instances, factory


def pair_of(entry):
    """(raw subject id, anonymous benchmark id) of a label or an input."""
    subject, item = entry[0] if isinstance(entry[1], int) else entry
    return subject["normalized_name"], item["benchmark_id"]


def anon(pair):
    return D.anon_id("subject", pair.subject_id), D.anon_id("benchmark", pair.benchmark_id)


# --- scoring arithmetic -----------------------------------------------------

def test_constant_half_scores_exactly_quarter_at_every_budget():
    res = O.run_official(make_pairs(3, 2, 100, repeats=2), const(0.5))
    for B_ in BUDGETS:
        assert res["brier"][B_] == pytest.approx(0.25, abs=1e-12)
        assert all(r["brier"][B_] == pytest.approx(0.25, abs=1e-12) for r in res["rows"])
    assert res["brier"]["ALC"] == pytest.approx(0.25, abs=1e-12)


def test_oracle_scores_zero():
    pairs = make_pairs(2, 2, 100)
    truth = {(p.subject_id, r.item["item_content"]): r.label for p in pairs for r in p.responses}
    oracle = lambda: (lambda input, labeled: float(
        truth[(input[0]["normalized_name"], input[1]["item_content"])]))
    res = O.run_official(pairs, oracle)
    assert all(res["brier"][B_] == 0.0 for B_ in BUDGETS)


def test_pairs_average_equally_not_by_response_count():
    small, big = make_pair("s0", "b0", 80, seed=1), make_pair("s1", "b1", 400, seed=2, repeats=3)
    truth = {r.item["item_content"]: r.label for r in small.responses}

    def factory():
        return lambda input, labeled: float(truth[input[1]["item_content"]]) \
            if input[0]["normalized_name"] == "s0" else 0.5
    res = O.run_official([small, big], factory)
    for B_ in BUDGETS:
        assert res["brier"][B_] == pytest.approx(0.125)


def test_ece_matches_its_definition():
    p = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    y = np.array([0, 1, 0, 1, 1])
    expected = (1 * 0.05 + 2 * abs(0.15 - 0.5) + 2 * abs(0.975 - 1.0)) / 5
    assert O.ece(p, y) == pytest.approx(expected)


def test_invalid_predictions_carry_the_platform_codes():
    run = [make_pair(n_items=80)]
    for bad in (1.5, -0.1, float("nan"), float("inf"), "abc", None):
        with pytest.raises(O.SubmissionError) as e:
            O.run_official(run, const(bad))
        assert e.value.code == 42

    def boom():
        def f(input, labeled):
            raise RuntimeError("model crashed")
        return f
    with pytest.raises(O.SubmissionError) as e:
        O.run_official(run, boom)
    assert e.value.code == 40

    def slow():
        def f(input, labeled):
            time.sleep(0.002)
            return 0.5
        return f
    with pytest.raises(O.SubmissionError) as e:
        O.run_official(run, slow, call_timeout=0.001)
    assert e.value.code == 41


def test_timing_is_recorded_per_call():
    res = O.run_official(make_pairs(2, 1, 100), const(0.5))
    t = res["timing"]
    assert t["evaluation_calls"] == 6 * sum(r["n_items"] for r in res["rows"])
    assert 0 <= t["evaluation_mean_s"] <= t["evaluation_max_s"]


# --- what a checkpoint sees ---------------------------------------------------

def test_labeled_is_empty_at_budget_zero_for_every_call():
    instances, factory = recorder()
    O.run_official(make_pairs(3, 2, 100), factory)
    assert instances[0] and all(lab == [] for lab in instances[0])


def test_every_target_sees_every_pairs_first_B_labels():
    pairs = make_pairs(3, 2, 100)
    instances, factory = recorder()
    res = O.run_official(pairs, factory)
    n_labels = {(p.subject_id, anon(p)[1]): len(res["acquired"][anon(p)]) for p in pairs}
    for B_, calls in zip(BUDGETS, instances):
        assert all(lab == calls[0] for lab in calls), "targets of one checkpoint saw different evidence"
        counts = defaultdict(int)
        for entry in calls[0]:
            counts[pair_of(entry)] += 1
        assert counts == {k: min(B_, n) for k, n in n_labels.items() if min(B_, n)}
        if B_:
            assert len({s for s, _ in counts}) == 3 and len({b for _, b in counts}) == 2


def test_budgets_are_nested_prefixes():
    instances, factory = recorder()
    O.run_official(make_pairs(3, 2, 100), factory)
    per_budget = []
    for calls in instances:
        g = defaultdict(list)
        for entry in calls[0]:
            g[pair_of(entry)].append((entry[0][1]["item_content"], entry[1]))
        per_budget.append(g)
    for small, large in zip(per_budget, per_budget[1:]):
        for k, labels in small.items():
            assert large[k][:len(labels)] == labels


def test_state_cannot_cross_checkpoints():
    """A fresh instance serves each checkpoint (and each worker shard), so a
    predictor that counts its calls restarts at every budget."""
    instances, factory = recorder()
    O.run_official(make_pairs(3, 2, 100), factory, workers=3)
    assert len(instances) == 6 * 3
    for calls in instances:
        assert calls and len({len(lab) for lab in calls}) == 1
    sizes = [len({len(lab) for calls in instances[i:i + 3] for lab in calls})
             for i in range(0, 18, 3)]
    assert sizes == [1] * 6


def test_a_mutating_predictor_cannot_corrupt_the_evidence():
    pairs = make_pairs(2, 2, 100)
    clean = O.run_official(pairs, const(0.5))
    instances, spy = recorder()

    def vandal():
        inner = spy()

        def f(input, labeled):
            inner(input, [[[dict(s), dict(i)], y] for (s, i), y in labeled])
            input[1]["benchmark_id"] = "x"
            for (s, i), _ in labeled:
                s["normalized_name"] = i["item_content"] = "x"
            labeled.clear()
            return 0.5
        return f
    res = O.run_official(pairs, vandal)
    assert res["acquired"] == clean["acquired"]
    assert all(e[0][0]["normalized_name"] != "x" for calls in instances for lab in calls for e in lab)


def test_acquisition_predictions_come_from_one_long_lived_instance():
    instances, factory = recorder()

    def wants_prediction(input, prediction, labeled, context):
        assert 0.0 <= prediction <= 1.0
        return O.default_policy(context, input)
    res = O.run_official(make_pairs(2, 2, 100), factory, wants_prediction)
    assert len(instances) == 1 + 6
    long_lived = [calls for calls in instances if len({len(lab) for lab in calls}) > 1]
    assert len(long_lived) == 1, "acquisition-time calls were spread over several instances"
    assert res["timing"]["acquisition_calls"] == len(long_lived[0]) > 0
    assert len({len(lab) for lab in long_lived[0]}) > 6


def test_acquisition_sees_labels_of_other_pairs():
    seen = []

    def spy(input, *, labeled=None, context=None):
        me = (input[0]["normalized_name"], input[1]["benchmark_id"])
        seen.append(any(pair_of(e) != me for e in labeled))
        return O.default_policy(context, input)
    O.run_official(make_pairs(2, 2, 100), const(0.5), spy)
    assert any(seen)


# --- acquisition ---------------------------------------------------------------

def test_default_policy_is_the_sha256_rule_of_the_official_client(tmp_path):
    client = O.official_client()
    if client is None:
        pytest.skip("organisers' streaming client is not on disk")
    slot = O._slots([make_pair(n_items=120)], 0, "pair")[0]
    events, ours = [], []
    for i, (_, inp, _) in enumerate(slot.candidates):
        acquired = sum(ours)
        if acquired == MAX_LABELS:
            break
        ctx = {"subject_id": slot.subject_id, "benchmark_id": slot.benchmark_id,
               "labels_acquired": acquired, "labels_remaining": MAX_LABELS - acquired,
               "max_labels": MAX_LABELS, "items_remaining": len(slot.candidates) - i}
        events.append({"protocol": "streaming_alc_v1", "type": "acquire", "event_id": i,
                       "labeled": [], "input": inp, "context": ctx})
        ours.append(O.default_policy(ctx, inp))
    events.append({"protocol": "streaming_alc_v1", "type": "finished", "predictions_csv": "p\n"})
    theirs, feed = [], iter(events)

    def exchange(payload):
        if "query" in payload:
            theirs.append(payload["query"])
        return next(feed)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"protocol": "streaming_alc_v1"}))
    client.run_streaming(
        config_path=config, predict=None, acquisition_function=None, model_path=None,
        output_dir=tmp_path / "out", concurrency=1, deadline=time.monotonic() + 60,
        per_call_timeout=1, acquisition_timeout=1,
        call_window=lambda *a: contextlib.nullcontext(), validate_score=float,
        prediction_stream=None, group_inputs=None, log=lambda *a: None, exchange=exchange)
    assert theirs == ours and sum(ours) == MAX_LABELS


def test_default_policy_known_value():
    ctx = {"subject_id": "subject_1", "benchmark_id": "benchmark_2", "labels_acquired": 0,
           "labels_remaining": 31, "max_labels": 31, "items_remaining": 50}
    inp = [{"a": "1"}, {"b": "2"}]
    key = '[{"benchmark_id":"benchmark_2","items_remaining":50,"labels_acquired":0,' \
          '"labels_remaining":31,"max_labels":31,"subject_id":"subject_1"},[{"a":"1"},{"b":"2"}]]'
    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    assert O.default_policy(ctx, inp) == (u < 31 / 50)


@pytest.mark.parametrize("n_items,expected", [(50, 25), (62, 31), (80, 31), (81, 31), (200, 31)])
def test_default_policy_fills_31_whenever_the_pool_allows(n_items, expected):
    res = O.run_official([make_pair(n_items=n_items, seed=n_items)], const(0.5))
    assert res["rows"][0]["n_labels"] == expected


def test_default_policy_is_deterministic():
    pairs = make_pairs(3, 2, 100)
    a = O.run_official(pairs, const(0.5))["acquired"]
    b = O.run_official(pairs, const(0.5))["acquired"]
    assert a == b
    c = O.run_official(pairs, const(0.5), seed=1)["acquired"]
    assert a != c


@pytest.mark.parametrize("deepcopy", [True, False])
def test_hook_must_return_a_native_bool(deepcopy):
    def numpy_bool(input, prediction=None, labeled=None, context=None):
        return np.bool_(True)
    with pytest.raises(ValueError, match="bool"):
        O.run_official([make_pair(n_items=80)], const(0.5), numpy_bool, deepcopy=deepcopy)


def test_hook_signature_dispatch_matches_the_official_client():
    client = O.official_client()
    if client is None:
        pytest.skip("organisers' streaming client is not on disk")

    def full(input, prediction=None, labeled=None, context=None): ...
    def kwonly(input, *, labeled=None, context=None): ...
    def two(input, prediction): ...
    def kw(input, **kwargs): ...
    def star(*args): ...
    def one(input): ...
    def labeled_only(input, labeled): ...
    def bad(): ...
    for fn in (None, full, kwonly, two, kw, star, one, labeled_only):
        ours, theirs = O.Hook(fn), client.AcquisitionHook(fn)
        assert (ours.mode, ours.positional, ours.keywords, ours.needs_prediction) == \
               (theirs.mode, theirs.positional, theirs.keywords, theirs.needs_prediction), fn
    with pytest.raises(ValueError):
        O.Hook(bad)
    with pytest.raises(ValueError):
        client.AcquisitionHook(bad)


def test_legacy_rank_reveals_the_top_scored_candidates():
    pair = make_pair(n_items=100)
    idx = lambda item: int(item["item_content"].split()[1])
    res = O.run_official([pair], const(0.5), lambda input: -idx(input[1]))
    assert res["acquisition_mode"] == "legacy_rank"
    acq, _ = O.split(pair)
    by_idx = sorted(acq, key=lambda k: int(k.split(":")[1]))
    assert res["acquired"][anon(pair)] == by_idx[:MAX_LABELS]


# --- split, sampling and format ---------------------------------------------

@pytest.mark.parametrize("scope", ["pair", "benchmark"])
def test_items_never_cross_pools(scope):
    pairs = make_pairs(3, 2, 100, repeats=3)
    offered, targets = defaultdict(set), defaultdict(list)

    def spy(input, *, context=None):
        offered[pair_of(input)].add(input[1]["item_content"])
        return O.default_policy(context, input)

    def factory():
        def f(input, labeled):
            targets[pair_of(input)].append(input[1]["item_content"])
            return 0.5
        return f
    O.run_official(pairs, factory, spy, split_scope=scope, dedupe=False)
    for k in offered:
        assert offered[k].isdisjoint(targets[k])
        per_item = defaultdict(int)
        for t in targets[k]:
            per_item[t] += 1
        assert set(per_item.values()) == {3 * 6}, "an item's repeats were split"
    by_bench = lambda d: {b: set().union(*(v for (s, bb), v in d.items() if bb == b))
                          for _, b in d}
    cross = [by_bench(offered)[b] & by_bench({k: set(v) for k, v in targets.items()})[b]
             for b in by_bench(offered)]
    if scope == "benchmark":
        assert all(not c for c in cross)
    else:
        assert any(cross), "per-pair splits should overlap across subjects"


def test_split_is_half_stable_and_exhaustive():
    pair = make_pair(n_items=101)
    acq, ev = O.split(pair)
    assert (acq, ev) == O.split(pair)
    assert len(acq) == 50 and sorted(acq + ev) == sorted(pair.item_keys)
    assert O.split(pair, seed=1) != (acq, ev)


def test_no_private_fields_reach_predict_or_the_hook():
    pairs = make_pairs(2, 2, 100, repeats=2)
    raw = {p.benchmark_id for p in pairs}
    bad = []

    def check(input, labeled):
        for subject, item in [input] + [e[0] for e in labeled]:
            if set(subject) != set(D.SUBJECT_FIELDS) or set(item) != set(D.ITEM_FIELDS):
                bad.append((sorted(subject), sorted(item)))
            if item["benchmark_id"] in raw or not item["benchmark_id"].startswith("benchmark_"):
                bad.append(item["benchmark_id"])
            if not all(isinstance(v, str) for v in [*subject.values(), *item.values()]):
                bad.append("non-string field")

    def factory():
        def f(input, labeled):
            check(input, labeled)
            return 0.5
        return f

    def hook(input, prediction, labeled, context):
        check(input, labeled)
        if set(context) != CONTEXT_KEYS or context["benchmark_id"] in raw:
            bad.append(context)
        return O.default_policy(context, input)
    res = O.run_official(pairs, factory, hook)
    assert bad == []
    assert res["benchmarks"] == sorted(raw)


def test_repeated_trials_reuse_one_prediction():
    pairs = make_pairs(2, 2, 100, repeats=3)
    a = O.run_official(pairs, const(0.3))
    b = O.run_official(pairs, const(0.3), dedupe=False)
    n_items = sum(r["n_items"] for r in a["rows"])
    assert a["timing"]["evaluation_calls"] == 6 * n_items
    assert b["timing"]["evaluation_calls"] == 6 * 3 * n_items
    assert a["timing"]["reused"] == 6 * 2 * n_items
    assert a["brier"] == b["brier"]


def test_sample_run_respects_the_cap_and_is_reproducible():
    pairs = make_pairs(8, 4, 300)
    for seed in range(20):
        run = O.sample_run(pairs, np.random.default_rng(seed))
        assert 5 <= len(run) <= 12
        assert sum(len(items) for _, items in run) <= O.CAP
        assert all(len(items) >= 80 and items <= set(p.item_keys) for p, items in run)
        again = O.sample_run(pairs, np.random.default_rng(seed))
        assert [(p.subject_id, p.benchmark_id, items) for p, items in run] == \
               [(p.subject_id, p.benchmark_id, items) for p, items in again]
    with pytest.raises(ValueError):
        O.sample_run(pairs, np.random.default_rng(0), n_pairs=13)


def test_sample_run_keeps_small_runs_whole():
    pairs = make_pairs(8, 4, 100)
    run = O.sample_run(pairs, np.random.default_rng(0), n_pairs=5)
    assert all(len(items) == 100 for _, items in run)


def test_concentration_piles_a_run_onto_one_benchmark():
    pairs = make_pairs(8, 4, 100)
    spread = [len(O.run_benchmarks(O.sample_run(pairs, np.random.default_rng(s), n_pairs=6)))
              for s in range(10)]
    piled = [len(O.run_benchmarks(O.sample_run(pairs, np.random.default_rng(s), n_pairs=6,
                                               concentration=1e9)))
             for s in range(10)]
    assert set(piled) == {1} and np.mean(spread) > 2


def _lopsided():
    """Twenty pairs on b0, one on b1."""
    return [make_pair(f"s{i}", "b0", 100, seed=i) for i in range(20)] + [make_pair("s0", "b1", 100)]


def test_benchmark_weighting_is_the_default_and_unchanged():
    pairs = _lopsided() + make_pairs(4, 3, 150)
    key = lambda run: [(p.subject_id, p.benchmark_id, items) for p, items in run]
    for seed in range(10):
        a = O.sample_run(pairs, np.random.default_rng(seed))
        b = O.sample_run(pairs, np.random.default_rng(seed), weighting="benchmark")
        assert key(a) == key(b)
    # pinned: the draw the benchmark-first rule gave before weighting existed
    run = O.sample_run(make_pairs(8, 4, 300), np.random.default_rng(7))
    assert [(p.subject_id, p.benchmark_id) for p, _ in run] == PINNED_SEED_7


PINNED_SEED_7 = [("s6", "b3"), ("s2", "b0"), ("s0", "b3"), ("s5", "b3"), ("s2", "b1"),
                 ("s1", "b1"), ("s5", "b1"), ("s7", "b2"), ("s4", "b3"), ("s1", "b3"),
                 ("s5", "b0"), ("s0", "b0")]


def test_pair_weighting_draws_pairs_uniformly():
    pairs = _lopsided()
    runs = {w: [O.sample_run(pairs, np.random.default_rng(s), n_pairs=5, weighting=w)
                for s in range(200)] for w in O.WEIGHTINGS}
    share = {w: np.mean(["b1" in O.run_benchmarks(r) for r in rs]) for w, rs in runs.items()}
    # benchmark-first: b1 has half the weight of every draw until taken (~0.97);
    # uniform over 21 pairs: 5/21
    assert share["benchmark"] > 0.9
    assert abs(share["pair"] - 5 / 21) < 0.08
    for rs in runs.values():
        assert all(len(r) == 5 and sum(len(k) for _, k in r) <= O.CAP for r in rs)
    again = O.sample_run(pairs, np.random.default_rng(3), n_pairs=5, weighting="pair")
    assert [(p.subject_id, p.benchmark_id) for p, _ in again] == \
           [(p.subject_id, p.benchmark_id) for p, _ in runs["pair"][3]]
    with pytest.raises(ValueError):
        O.sample_run(pairs, np.random.default_rng(0), weighting="subject")


def test_dense_run_takes_every_pair_of_a_benchmark_whole():
    pairs = make_pairs(5, 2, 120)
    run = O.dense_run(pairs, "b1")
    assert [p.subject_id for p, _ in run] == [f"s{i}" for i in range(5)]
    assert all(items == set(p.item_keys) for p, items in run)


def test_eligibility_drops_fractional_benchmarks_non_binary_and_small_pairs():
    ok = make_pair("s0", "b0", 80)
    small = make_pair("s0", "b1", 79)
    frac = make_pair("s0", "mmdocrag", 100)
    extra = Response(item_key="b0:x", item=dict(ok.responses[0].item), label=2)
    noisy = Pair(ok.subject, "s1", "b0", ok.responses + [extra])
    kept = O.eligible([ok, small, frac, noisy])
    assert [(p.subject_id, p.benchmark_id) for p in kept] == [("s0", "b0"), ("s1", "b0")]
    assert len(kept[1].responses) == len(ok.responses)


def test_leave_benchmark_out_helpers():
    pairs = make_pairs(4, 3, 100)
    run = O.sample_run(pairs, np.random.default_rng(3), n_pairs=4, concentration=1e9)
    names = O.run_benchmarks(run)
    assert len(names) == 1
    assert all(p.benchmark_id not in names for p in O.training_pairs(pairs, run))
    assert len(O.training_pairs(pairs, run)) == 8


# --- baselines in the official format ------------------------------------------

def _entry(subject, bench, y, content="x"):
    return [[dict(subject), {"item_content": content, "item_features": "", "interactors": "",
                             "benchmark_id": bench}], y]


def test_empirical_mean_matches_the_organisers_model():
    import importlib.util
    path = O._CLIENT.replace("tools/streaming_ingestion.py", "empirical_mean/model.py")
    try:
        spec = importlib.util.spec_from_file_location("_organisers_mean", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except (FileNotFoundError, OSError):
        pytest.skip("organisers' baseline is not on disk")
    rng = np.random.default_rng(0)
    subjects = [{"normalized_name": f"m{i}", "harness": h} for i in range(2) for h in "ab"]
    for _ in range(200):
        lab = [_entry(subjects[rng.integers(4)], f"benchmark_{rng.integers(2)}", int(rng.integers(2)))
               for _ in range(rng.integers(0, 12))]
        target = [dict(subjects[rng.integers(4)]), {"item_content": "t", "benchmark_id": "benchmark_0"}]
        assert B.empirical_mean(target, lab) == mod.predict(target, lab)


def test_pooled_anchor_formula():
    me, other = {"normalized_name": "a"}, {"normalized_name": "b"}
    lab = [_entry(me, "b0", 1), _entry(other, "b0", 0), _entry(other, "b0", 0),
           _entry(me, "b1", 1), _entry(me, "b1", 0)]
    target = [dict(me), {"item_content": "t", "benchmark_id": "b0"}]
    ms = (2 + 4 * 0.5) / (3 + 4)
    mb = (1 + 4 * ms) / (3 + 4)
    assert B.pooled_anchor()(target, lab) == pytest.approx((1 + 4 * mb) / (1 + 4))
    for f in (B.empirical_mean, B.smoothed_mean(), B.pooled_anchor(), B.const()):
        assert f(target, []) == 0.5
