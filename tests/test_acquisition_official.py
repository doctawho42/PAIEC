"""experiments/acquisition_official.py: the policies as streaming hooks, the paired
statistics on synthetic rows, the rows' round trip through the results file, and
the stored results.

The policies are checked one decision at a time (slack, no slack, no labels
left, twins, unseen groups, the uncertainty band) and through
paiec.official.run_official on synthetic pairs: each is a streaming hook that
returns native bools and fills 31 labels whenever the pool allows, and the
placebo draws another subset of the same size. The statistics are checked on
hand-made rows: a paired difference is the mean of the run differences, the
pair-weighted estimator equals it, the two-arm mean is the mean of the two
random arms, a zero difference has zero SE. If results/acquisition_official.json
exists, its summary must be what this script computes from the rows embedded in
it, every required check must have passed, and no anonymous id may appear.
--summarise-only --from-results refuses to overwrite the default results file
or the file it reads (it would drop the scratch-row comparison), and rebuilds
into another file with the source's scratch-row record copied as not re-run.
"""
import json
import os

import numpy as np
import pytest

from experiments import acquisition_official as A
from paiec import official as O
from paiec.evaluator import BUDGETS, MAX_LABELS, Pair, Response
from tests.synth import make_pairs

SUBJECT = {"normalized_name": "s0", "provider": "acme", "release_date": "2025-01-01",
           "access_date": "2026-01-01", "harness": "default", "harness_version": "",
           "reasoning_effort": "", "subject_features_extra": ""}


def const(v):
    return lambda: (lambda input, labeled=None: v)


def _item(content, features="", bench="benchmark_1"):
    return {"item_content": content, "item_features": features, "interactors": "", "benchmark_id": bench}


def _ctx(acquired=0, items_remaining=40):
    return {"subject_id": "subject_1", "benchmark_id": "benchmark_1", "labels_acquired": acquired,
            "labels_remaining": MAX_LABELS - acquired, "max_labels": MAX_LABELS,
            "items_remaining": items_remaining}


def _lab(content, features="", subject=SUBJECT, y=1):
    return [[dict(subject), _item(content, features)], y]


# --- one decision at a time ----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["placebo", "balanced", "distinct", "uncertain"])
def test_no_slack_takes_and_no_labels_left_skips(name):
    f = A.POLICIES[name]
    inp = [dict(SUBJECT), _item("x")]
    kw = {"labeled": [], "context": _ctx(acquired=10, items_remaining=MAX_LABELS - 10)}
    if name == "uncertain":
        kw["prediction"] = 0.99
    assert f(inp, **kw) is True         # the placebo's own rule takes every item at ratio 1 too
    kw["context"] = _ctx(acquired=MAX_LABELS, items_remaining=5)
    assert f(inp, **kw) is False


def test_distinct_skips_a_visible_twin_only():
    inp = [dict(SUBJECT), _item("same text", "tier=a")]
    twin = [_lab("same text", "tier=a")]
    other_subject = [_lab("same text", "tier=a", subject=dict(SUBJECT, normalized_name="s9"))]
    ctx = _ctx(acquired=3, items_remaining=40)
    assert A.distinct(inp, labeled=twin, context=ctx) is False
    # a twin held by another subject is not this pair's: the platform's rule decides
    assert A.distinct(inp, labeled=other_subject, context=ctx) is O.default_policy(ctx, inp)
    assert A.distinct(inp, labeled=[_lab("other text", "tier=a")], context=ctx) is O.default_policy(ctx, inp)


def test_balanced_takes_an_unseen_group_and_otherwise_defers_to_the_platform():
    ctx = _ctx(acquired=2, items_remaining=40)
    seen = [_lab("a", "grp=red;idx=1"), _lab("b", "grp=red;idx=2")]
    new_group = [dict(SUBJECT), _item("c", "grp=blue;idx=3")]
    same_group = [dict(SUBJECT), _item("d", "grp=red;idx=4")]
    assert A.balanced(new_group, labeled=seen, context=ctx) is True
    assert A.balanced(same_group, labeled=seen, context=ctx) is O.default_policy(ctx, same_group)
    # numeric values are not groups: idx alone never makes a candidate new
    assert A.group_sig(_item("e", "grp=red;idx=99")) == A.group_sig(_item("f", "grp=red;idx=1"))


def test_uncertain_is_bles_band_on_the_prediction():
    inp = [dict(SUBJECT), _item("x")]
    ctx = _ctx(acquired=1, items_remaining=40)
    for p, want in ((0.5, True), (0.36, True), (0.64, True), (0.3, False), (0.9, False), (0.02, False)):
        assert A.uncertain(inp, prediction=p, labeled=[], context=ctx) is want


def test_placebo_is_the_platform_rule_on_another_key():
    inp = [dict(SUBJECT), _item("x")]
    takes = [A.placebo([dict(SUBJECT), _item(f"x{i}")], labeled=[], context=_ctx(0, 50)) for i in range(400)]
    rule = [O.default_policy(_ctx(0, 50), [dict(SUBJECT), _item(f"x{i}")]) for i in range(400)]
    assert abs(np.mean(takes) - MAX_LABELS / 50) < 0.08          # the same rate ...
    assert takes != rule                                          # ... on other draws
    assert A.placebo(inp, labeled=[], context=_ctx(0, 50)) is A.placebo(inp, labeled=[], context=_ctx(0, 50))


# --- through the official protocol ---------------------------------------------------------------

def test_policies_run_as_streaming_hooks_and_fill_31_labels():
    pairs = make_pairs(2, 2, 100)
    acquired = {}
    for name in A.POLICY_NAMES:
        res = O.run_official(pairs, const(0.5), A.POLICIES[name], deepcopy=False)
        assert res["acquisition_mode"] == ("random" if name == "random" else "stream")
        assert all(r["n_labels"] == MAX_LABELS for r in res["rows"])
        acquired[name] = res["acquired"]
        # the constant predictor scores 0.25 whatever is acquired
        assert res["brier"]["ALC"] == pytest.approx(0.25)
    assert acquired["placebo"] != acquired["random"]
    # uncertain at a constant 0.5 takes every candidate in the band, so the stream's first 31
    for k, keys in acquired["uncertain"].items():
        assert len(keys) == MAX_LABELS


def test_a_small_pool_is_taken_whole_by_every_policy():
    pairs = make_pairs(1, 1, 44)             # 22 acquisition candidates < 31 labels
    for name in A.POLICY_NAMES:
        res = O.run_official(pairs, const(0.5), A.POLICIES[name], deepcopy=False)
        assert [r["n_labels"] for r in res["rows"]] == [22]


def test_distinct_never_acquires_a_visible_twin_while_slack_remains():
    # 100 items over 30 distinct texts: the acquisition half has twins to skip
    responses = []
    rng = np.random.default_rng(3)
    for i in range(100):
        it = {"item_content": f"text {i % 30}", "item_features": "", "interactors": "",
              "benchmark_id": "b0"}
        responses.append(Response(item_key=f"b0:{i}", item=it, label=int(rng.random() < 0.4)))
    pair = Pair(subject=dict(SUBJECT), subject_id="s0", benchmark_id="b0", responses=responses)
    res_r = O.run_official([pair], const(0.5), None, deepcopy=False)
    res_d = O.run_official([pair], const(0.5), A.distinct, deepcopy=False)

    def dups(keys):
        texts = [int(k.split(":")[1]) % 30 for k in keys]
        return len(texts) - len(set(texts))
    (kr,), (kd,) = res_r["acquired"].values(), res_d["acquired"].values()
    assert len(kd) == len(kr) == MAX_LABELS
    assert dups(kd) < dups(kr)


# --- statistics on hand-made rows ----------------------------------------------------------------

def _rows(n_runs=6, seed=0, same=False):
    """Synthetic rows: runs of 2-4 pairs over 3 parents x 4 subjects, five arms."""
    rng = np.random.default_rng(seed)
    rows = {}
    for i in range(n_runs):
        n = int(rng.integers(2, 5))
        pairs = [{"parent": f"p{int(rng.integers(3))}", "subject": f"subject_{int(rng.integers(4))}",
                  "n_items": 40} for _ in range(n)]
        base = rng.uniform(0.1, 0.3, size=(n, 6))
        pol = {}
        for k, name in enumerate(A.POLICY_NAMES):
            b = base if same else base + rng.normal(0, 0.01, size=base.shape) * (k > 0)
            pol[name] = {"pair_brier": b.tolist(), "n_labels": [31] * n, "failures": 0, "secs": 0.0,
                         "by_budget": b.mean(0).tolist(), "ALC": float(np.asarray(A.WEIGHTS) @ b.mean(0))}
        rows[i] = {"pairs": pairs, "policies": pol, "seed": 0}
    return rows


def test_paired_difference_is_the_mean_of_run_differences():
    rows = _rows()
    runs = list(rows)
    c = A.compare(rows, runs, "balanced", "random", np.random.default_rng(0))
    d = [rows[i]["policies"]["balanced"]["ALC"] - rows[i]["policies"]["random"]["ALC"] for i in runs]
    assert c["D"] == pytest.approx(np.mean(d), abs=1e-15)
    assert c["D_pair_estimator"] == pytest.approx(c["D"], abs=1e-12)
    assert c["run_se"] == pytest.approx(np.std(d, ddof=1) / np.sqrt(len(d)))
    assert c["cluster_se"] > 0 and len(c["by_budget"]) == 6
    two = A.compare(rows, runs, "balanced", "two_arm", np.random.default_rng(0))
    vs_p = A.compare(rows, runs, "balanced", "placebo", np.random.default_rng(0))
    assert two["D"] == pytest.approx(0.5 * (c["D"] + vs_p["D"]), abs=1e-15)


def test_identical_arms_give_zero_with_zero_se():
    rows = _rows(same=True)
    c = A.compare(rows, list(rows), "placebo", "random", np.random.default_rng(0))
    assert c["D"] == 0 and c["cluster_se"] == 0 and c["identical_runs"] == len(rows)
    assert not c["meets_tl_bar"]


def test_summary_survives_the_round_trip_through_the_results_file(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS", (0, 1, 2, 3))
    monkeypatch.setattr(A, "SCRATCH_FIXED_RUNS", (0, 1, 2, 4))
    raw = {}
    for batch in A.BATCHES:
        for i, r in _rows(n_runs=5, seed=1 if batch == "fixed" else 2).items():
            raw[(batch, i)] = dict(r, batch=batch, i=i)
    s1 = A.summarise(raw)
    path = tmp_path / "acq.json"
    path.write_text(json.dumps({"rows": A.embed_rows(raw), "provenance": {"lib_digest": "x"}}))
    back, lib = A.rows_from_results(str(path))
    assert lib == "x" and set(back) == set(raw)
    s2 = A.summarise(back)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    assert "subject_" not in json.dumps(A.embed_rows(raw))
    assert s1["range"]["policies_minus_two_arm_mean"][0] <= s1["range"]["policies_minus_two_arm_mean"][1]


def _sha(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_from_results_never_overwrites_a_results_file(tmp_path):
    """--summarise-only --from-results drops the scratch-row comparison, so it refuses to
    write the default results file (the default --out) or the file it reads."""
    before = _sha(STORED) if os.path.exists(STORED) else None
    with pytest.raises(SystemExit) as e:
        A.main(["--summarise-only", "--from-results"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        A.main(["--summarise-only", "--from-results", "--out", STORED])
    assert e.value.code == 2
    src = tmp_path / "acq.json"
    src.write_text("{}")
    with pytest.raises(SystemExit) as e:
        A.main(["--summarise-only", f"--from-results={src}", "--out", str(src)])
    assert e.value.code == 2 and src.read_text() == "{}"
    assert (_sha(STORED) if os.path.exists(STORED) else None) == before


def test_from_results_rebuilds_into_another_file(tmp_path, monkeypatch):
    """On synthetic embedded rows: the rebuild writes --out only, leaves the source as it
    was, carries the source's scratch-row record as not re-run, and its summary is the
    rows' summary."""
    monkeypatch.setattr(A, "RUNS", (0, 1, 2, 3))
    monkeypatch.setattr(A, "SCRATCH_FIXED_RUNS", (0, 1, 2, 4))
    monkeypatch.setattr(A, "SCRATCH", {})
    monkeypatch.setattr(A, "data_inputs", lambda: {})
    raw = {}
    for batch in A.BATCHES:
        for i, r in _rows(n_runs=5, seed=1 if batch == "fixed" else 2).items():
            raw[(batch, i)] = dict(r, batch=batch, i=i)
    record = {"fixed": {"max_abs_diff": 0.0, "missing_runs": []}}
    src = tmp_path / "acq.json"
    src.write_text(json.dumps({"rows": A.embed_rows(raw), "provenance": {"lib_digest": A.lib_digest()},
                               "scratch": {"rows": record},
                               "checks": [{"check": "fixed batch: every scratch row reproduces bit for bit",
                                           "ok": True}]}))
    before = _sha(src)
    dst = tmp_path / "rebuilt.json"
    assert A.main(["--summarise-only", f"--from-results={src}", "--out", str(dst)]) == 0
    assert _sha(src) == before
    res = json.loads(dst.read_text())
    assert res["scratch"]["rows"] is None
    assert res["scratch"]["rows_in_source"]["rows"] == record
    assert [c["ok"] for c in res["scratch"]["rows_in_source"]["checks"]] == [True]
    assert res["provenance"]["rows"].startswith("embedded in ")
    assert res["rows"] == json.loads(src.read_text())["rows"]
    back, _ = A.rows_from_results(str(src))
    assert json.dumps(res["summary"], sort_keys=True) == json.dumps(
        json.loads(json.dumps(A.summarise(back))), sort_keys=True)


def test_scratch_comparison_reads_precision():
    assert A._same(0.00228499, 0.00228, 5) and not A._same(0.002286, 0.00228, 5)
    assert A._same(0.15951, 0.1595, 4)


# --- the stored results --------------------------------------------------------------------------

STORED = os.path.join(A.ROOT, "results", "acquisition_official.json")


@pytest.mark.skipif(not os.path.exists(STORED), reason="results/acquisition_official.json not written yet")
def test_stored_results_are_this_scripts_summary_of_its_rows():
    with open(STORED) as f:
        res = json.load(f)
    assert all(c["ok"] for c in res["checks"] if c["required"])
    assert res["scratch"]["counts"]["differs"] == 0
    if res["scratch"]["rows"]:
        assert all(v["max_abs_diff"] == 0.0 and not v["missing_runs"] for v in res["scratch"]["rows"].values())
    rows, _ = A.rows_from_results(STORED)
    s = A.summarise(rows)
    for name in ("fixed", "salted", "fixed_scratch_subset"):
        if res["summary"].get(name) is None:
            continue
        for p in ("balanced", "distinct", "uncertain"):
            for ref in ("minus_random", "minus_placebo", "minus_two_arm_mean"):
                a, b = s[name][p][ref], res["summary"][name][p][ref]
                assert a["D"] == pytest.approx(b["D"], abs=1e-15)
                assert a["cluster_se"] == pytest.approx(b["cluster_se"], abs=1e-15)
    text = json.dumps(res)
    assert "subject_" not in text and "benchmark_" not in text
    assert len(BUDGETS) == len(res["summary"]["fixed"]["placebo_minus_random"]["by_budget"])
