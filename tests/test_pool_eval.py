"""experiments/pool_eval.py (amendment 4's pool evaluation) on synthetic pools and
hand-built tables; no pool download needed.

What is pinned: RULE_TEXT against the amendment's block and every threshold of RULE in
it, the constants block's digest; the classification of outputs and the harness copy
(level_calibration's checkpoints and arithmetic, invalid outputs recorded, never
raised); the label-randomised copies and the recording factory of RP3 and timing, which
never score; the weights (the tilt, the cap after the dense restriction), the samplers'
invariants and the realised level on evaluated responses; the split and the levels
against the planning tables; every statistic of the rule (D, SEf, SE2, SEg, U95, PB with
the appearance weight, JK, p99); the selection and the gate, each condition binding, the
twins' no-loss bound and the tie-break; the lock's import closure, digests and fix scope;
record-fix's re-score rule; the check log; the resident-set check after a row is
written; and that nothing before describe-hidden opens the formative feedback.
"""
import hashlib
import json
import math
import os
import re

import numpy as np
import pytest

import experiments.level_calibration as LC
import experiments.pool_eval as E
from paiec import official as O
from paiec import pool as PL
from paiec import testlike as T
from paiec.evaluator import Pair, Response
from paiec.hier import Hyper
from tests.test_pool import TEXT, bench, write_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: written here as well as in the script, so that moving either needs both edits
CONSTANTS_SHA256 = "619200e75121e73a7ac5e28f9a80f29f12e51f287485f2f7e3d1c86b9a1bdc3f"


# --- a synthetic pool ------------------------------------------------------------------------

def _specs():
    names = [f"model {j}" for j in range(40)]
    out = []
    for j in range(14):
        kw = {}
        if j in (2, 3):
            kw["source_url"] = "https://example.org/family-a/repo"
        if j in (5, 9):
            kw["direction"] = "lower_is_better"
        if j == 7:
            kw["repeats"] = 2
        if j == 11:
            kw["kind"] = "mixed"
        out.append(bench(f"b{j:02d}", n_subjects=6 + j % 4, n_items=90 + 10 * (j % 5), seed=j,
                         names=[names[(3 * j + s) % 40] for s in range(6 + j % 4)], **kw))
    return out


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """A synthetic pool of 14 benchmarks (one family of two, two direction-flagged, one with
    repeated responses, one mixed), its lists, levels and pair statistics; every cache in
    tmp_path. Returns a dict with the PoolView, the halves and the statistics."""
    root = write_pool(str(tmp_path / "pool"), _specs())
    for k, v in {"POOL": root, "FITS_DIR": str(tmp_path / "fits"), "EVAL_DIR": str(tmp_path / "eval"),
                 "TOOLS": str(tmp_path / "tools")}.items():
        monkeypatch.setattr(PL, k, v)
    for k, v in {"EVAL_DIR": str(tmp_path / "eval"), "LEVELS": str(tmp_path / "eval" / "levels.json"),
                 "STATS_DIR": str(tmp_path / "eval" / "stats"), "ROWS": str(tmp_path / "rows"),
                 "BUNDLES_DIR": str(tmp_path / "eval" / "bundles"), "OUT": str(tmp_path / "state.json"),
                 "A3_PRIOR": str(tmp_path / "eval" / "a3_prior.json")}.items():
        monkeypatch.setattr(E, k, v)
    for c in (PL._RESP, PL._VERIFIED, PL._MANIFESTS):
        c.clear()
    os.makedirs(PL.TOOLS)
    PL.write_lists(root, PL.TOOLS)
    bench_, fams = PL.read_lists()
    levels = E.derive_levels(bench_["E"])
    E.write_levels(levels)
    pool = E.PoolView(bench_["E"], bench_["L"], bench_["U"], fams["family_of"], ("b05", "b09"),
                      ("b11",), ("b04",), E.read_levels(), helm=("b06",))
    half, _ = E.split_halves(pool)
    stats = E.build_stats(pool.E)
    yield {"root": root, "pool": pool, "half": half, "stats": stats, "tmp": tmp_path,
           "levels": levels, "fams": fams}
    for c in (PL._RESP, PL._VERIFIED, PL._MANIFESTS):
        c.clear()


def _half(toy, H="C"):
    return E.Half(toy["pool"], H, [b for b in toy["pool"].E if toy["half"][b] == H])


# --- the constants and the rule --------------------------------------------------------------

def _amendment_text():
    for path in (os.path.join(ROOT, E.AMENDMENT_FILE),
                 os.path.join(ROOT, E.DRAFT_DIR, E.DRAFT_FILES["amendment"])):
        t = E.amendment_section(path)
        if t is not None:
            return t
    return None


def test_rule_text_is_the_amendments_block():
    text = _amendment_text()
    if text is None:
        pytest.skip("neither docs/plans/p2_final_plan.md's amendment 4 nor its draft is here")
    assert E.rule_block(text) == E.RULE_TEXT
    assert all(ord(c) < 128 for c in E.RULE_TEXT)


def test_every_threshold_is_stated_in_the_rule_text():
    assert E.rule_problems() == []
    assert set(E.RULE) == set(E.RULE_PHRASES)
    # the twins' old bound (+0.002, before the re-check's N1) is not what the text says
    old = dict(E.RULE, twin=E.RULE["loss"])
    assert any(p.startswith("twin:") for p in E.rule_problems(rule=old))
    assert E.rule_problems(rule=dict(E.RULE, gain=-0.003))


def test_constants_block_digest():
    block = E.constants_block()
    assert block.startswith(E.CONSTANTS_START) and block.endswith(E.CONSTANTS_END)
    for name in ("RULE_TEXT = ", "RULE = ", "CANDIDATES = ", "SPLIT = ", "REGIMES = ", "N_RUNS = ",
                 "PINNED = ", "FIX_SCOPE = ", "def classify_output", "DEGENERATE_ITEM_VAR = ",
                 "DEGENERATE_EXPECTED = ", "DECISION = ", "def decision_bound", "def decision_ast",
                 # amendment 4b
                 "AMENDMENT_4B_HEADING = ", "GATE_STAGES = ", "def block_literals",
                 "def refuse_unless_block", "def object_digest", "def frozen_names",
                 "def frozen_digests", "def frozen_problems",
                 # amendment 4c
                 "AMENDMENT_4C_HEADING = ", "HARD_STOPS = ", "HARD_STOPS_RULE_TEXT = ",
                 "HARD_STOP_MOVE_DAYS = "):
        assert name in block
    assert E.constants_sha256() == E.CONSTANTS_SHA256 == CONSTANTS_SHA256


def test_constants_mirror_what_they_name():
    assert E.constants_problems() == []
    assert E.CANDIDATES == {"C1": "literal", "C1m": "matched", "C2": "eb"} == PL.CANDIDATES
    assert E.TIE_ORDER == ("C1", "C1m", "C2")
    assert E.EVAL_LOADER == tuple(PL.EVAL_LOADER)
    assert dict(E.LOSS_BOUNDS)["P-READING-DS"] == dict(E.LOSS_BOUNDS)["P-AUDIT-DS"] == "twin"
    assert E.RULE["twin"] == 0.0 and E.RULE["loss"] == 0.002 and E.RULE["dense"] == 0.001
    assert len(E.GAUSS_M) == 33 and E.GAUSS_M[0] == -3.5 and E.GAUSS_M[-1] == 0.5
    assert E.GAUSS_S == (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)
    assert E.MIX_WEIGHT == (0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3)
    # amendment 4a
    assert E.DEGENERATE_ITEM_VAR == PL.DEGENERATE_ITEM_VAR == 50.0
    assert E.DEGENERATE_EXPECTED == ("finegrain_t2i", "multi_moe", "revolve")
    assert E.CSV_TOL == PL.CSV_TOL == 1e-12
    assert E.decision_binding_problems() == []
    assert len(E.MIX_HIGH_MEAN) == 13 and E.MIX_HIGH_MEAN[-1] == 0.0
    assert sum(E.RS_TASKS.values()) == 520
    # amendment 4b
    assert E.GATE_STAGES == E.SELECT_STAGES + ("score:C", "summarise", "gate")
    assert E.block_binding_problems() == []
    for name in ("load_rows", "missing_tasks", "row_path", "score_preds", "np", "math", "os",
                 "hashlib", "defaultdict", "O", "BUDGETS"):
        assert name in E.DECISION
    # amendment 4c: every hard stop RULE_TEXT names, moved by two days
    assert E.hard_stop_problems() == []
    assert E.HARD_STOPS == {"lock": "2026-10-15T23:59:59Z", "gate": "2026-10-17T23:59:59Z",
                            "archive_4": "2026-10-18T23:59:59Z",
                            "regression": "2026-10-19T23:59:59Z", "final": "2026-10-22T23:59:59Z"}
    legacy = sorted(p for p in E.PINNED if p.split("/")[0] == "data" and p.split("/")[1] in PL.LEGACY)
    assert legacy == sorted(f"data/{b}/{t}.parquet" for b in PL.LEGACY
                            for t in ("benchmarks", "items", "response", "subjects"))
    assert all(E.PINNED[p] is None for p in legacy)


@pytest.mark.parametrize("value,ok", [
    (0.3, True), (np.float64(0.7), True), ([0.3], True), (0.0, True), (1.0, True),
    (float("nan"), False), (float("inf"), False), (-0.01, False), (1.01, False),
    ([0.2, 0.3], False), (None, False), (object(), False), (np.array([[0.4]]), False)])
def test_classify_output(value, ok):
    got, p, why = E.classify_output(value)
    assert got is ok
    assert (p is not None) is ok and (why == "") is ok


# --- the harness ---------------------------------------------------------------------------

def _toy_run(toy, regime="P-FLAT", i=0, H="C"):
    ents = E.draw_run(_half(toy, H), toy["stats"], regime, None, E.SEL_SEED, i)
    return ents, E.load_run(ents)


def _arm(c=0.3):
    return lambda: (lambda inp, labeled=None: c + 0.05 * (len(labeled or []) % 5) / 5)


def test_harness_is_level_calibrations(toy):
    _, run = _toy_run(toy)
    s1, c1 = E.checkpoints(run)
    s2, c2 = LC.checkpoints(run)
    assert [(b, lab, inp, list(ix)) for b, lab, inp, ix in c1] == \
        [(b, lab, inp, list(ix)) for b, lab, inp, ix in c2]
    res = E.evaluate_timed(s1, c1, {"x": _arm()}, keep_calls=True)
    ref, _ = LC.evaluate(s2, c2, ["x"], _arm())
    assert res["x"]["rows"] == ref["x"]
    assert res["x"]["calls"] == sum(len(inp) for _, _, inp, _ in c1) == len(res["x"]["calls_cpu"])
    assert res["x"]["invalid"] == [] and res["x"]["cpu_max"] >= 0


def test_invalid_outputs_are_recorded_never_raised(toy):
    _, run = _toy_run(toy)
    slots, cps = E.checkpoints(run)

    def nan():
        return lambda inp, labeled=None: float("nan") if labeled else 0.4

    def boom():
        def f(inp, labeled=None):
            raise RuntimeError("x")
        return f
    res = E.evaluate_timed(slots, cps, {"good": _arm(), "nan": nan, "boom": boom}, run_id="r0")
    assert res["nan"]["rows"] is None and res["boom"]["rows"] is None
    bad = res["nan"]["invalid"]
    assert bad and all(x["arm"] == "nan" and x["run"] == "r0" and x["budget"] > 0 for x in bad)
    assert {x["budget"] for x in bad} == {1, 3, 7, 15, 31}
    assert res["boom"]["invalid"][0]["why"].startswith("raised")
    ref, _ = LC.evaluate(*LC.checkpoints(run), ["good"], _arm())
    assert res["good"]["rows"] == ref["good"]


def _sel_table(**over):
    """A selection table where every candidate meets every condition (C1 the best)."""
    t = {E.A3: {"invalid": 0}}
    for k, X in enumerate(E.TIE_ORDER):
        D = {R: 0.0 for R in E.REGIMES}
        D["P-READING"] = D["P-AUDIT"] = -0.003 + 0.0002 * k
        t[X] = {"D": D, "RS": {R: 0.0 for R in E.RS_REGIMES},
                "latency": {"ratio": 1.0, "max_s": 0.1}, "complete": True, "checks_ok": True,
                "invalid": 0}
    for path, v in over.items():
        keys = path.split(".")
        d = t
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return t


def test_invalid_outputs_give_archive_3_stays():
    sel = E.apply_selection(_sel_table(**{"C1.invalid": 1}))
    assert not sel["candidates"]["C1"]["eligible"] and sel["candidates"]["C1"]["archive_3_stays_for_it"]
    assert sel["candidates"]["C1"]["holds"]["E4"]          # the invalid arm is excused from (E4)
    assert sel["X_star"] == "C1m"                          # and no other candidate is affected
    assert not any(sel["candidates"][X]["archive_3_stays_for_it"] for X in ("C1m", "C2"))
    sel = E.apply_selection(_sel_table(**{"A3.invalid": 1}))
    assert sel["X_star"] is None and sel["outcome"] == "no candidate carried: archive-3 stays"
    assert all(v["archive_3_stays_for_it"] for v in sel["candidates"].values())


# --- label-randomised copies, the recording factory, no scores ------------------------------

def test_label_randomised_copies_follow_their_stream(toy):
    _, run = _toy_run(toy)
    seq = E.label_stream("rp3", 0)
    assert seq == [E.TIMING_SEED, 3, 0] and E.label_stream("timing", 4) == [E.TIMING_SEED, 9, 4]
    copy = E.label_randomised(run, seq)
    rng = np.random.default_rng(seq)
    for (p, items), (q, items2) in zip(run, copy):
        assert items == items2 and p.subject == q.subject and p.subject_id == q.subject_id
        assert [r.item_key for r in p.responses] == [r.item_key for r in q.responses]
        assert [r.item for r in p.responses] == [r.item for r in q.responses]
        assert [r.label for r in q.responses] == [int(rng.integers(0, 2)) for _ in q.responses]
    assert E.assert_randomised(copy, seq) == sum(len(p.responses) for p, _ in run)
    p0, k0 = copy[0]
    bad = [(Pair(p0.subject, p0.subject_id, p0.benchmark_id,
                 [Response(r.item_key, r.item, 1 - r.label) for r in p0.responses]), k0)] + copy[1:]
    with pytest.raises(AssertionError):
        E.assert_randomised(bad, seq)


def test_the_twin_reuses_its_parents_copy(toy):
    _, run = _toy_run(toy)
    copy = E.label_randomised(run, E.label_stream("rp3", 0))
    twin = E.twin_of(copy)
    for (p, _), (q, _) in zip(copy, twin):
        assert [r.label for r in p.responses] == [r.label for r in q.responses]
        for f in ("release_date", "access_date"):
            assert q.subject[f] == T.shift_date(p.subject[f], E.DATE_SHIFT)
        assert {k: v for k, v in p.subject.items() if "date" not in k} == \
            {k: v for k, v in q.subject.items() if "date" not in k}


def test_recording_factory_gives_predict_timeds_predictions(toy):
    _, run = _toy_run(toy)
    copy = E.label_randomised(run, E.label_stream("rp3", 0))
    for r in (copy, E.twin_of(copy)):
        mk = lambda: E.ArmFactory({None: (None, Hyper())})   # noqa: E731
        _, cps = E.checkpoints(r)
        mine = E.predict_timed(cps, mk(), "A3", raw=True)["raw"]
        assert mine == E.recorded_official(r, mk())
        assert len(mine) == len(E.BUDGETS)


class _StubCache:
    def __init__(self, ctx_L):
        self.ctx = type("C", (), {"L": ctx_L})()

    def get(self, X, excl):
        return {"prior": None, "hyper": Hyper(excluded=list(excl)), "sha256": "0", "key": "k"}


def _no_score(*a, **k):
    raise AssertionError("a score was computed")


def test_rp3_never_scores_and_holds_no_brier(toy, monkeypatch):
    monkeypatch.setattr(E, "score_preds", _no_score)
    monkeypatch.setattr(E, "a3_arm", lambda: E.ArmFactory({None: (None, Hyper())}))
    comps = {i: E.draw_run(_half(toy, "S"), toy["stats"], R, None, E.TIMING_SEED, k)
             for i, _, R, k in E.RP3_RUNS}
    out = E.rp3(comps, _StubCache(toy["pool"].L), toy["pool"])
    assert out["ok"] and set(out["runs"]) == {"0", "1", "0-DS"}
    text = json.dumps(out).lower()
    assert "brier" not in text and "alc" not in text


# --- weights and samplers --------------------------------------------------------------------

def test_tilt_and_cap():
    L = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
    assert np.allclose(E.tilt(L, None), 0.2)
    t = E.tilt(L, [(1.0, -1.0, 0.8)])
    assert abs(t.sum() - 1) < 1e-12 and t[1] == t.max()
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.exponential(size=int(rng.integers(10, 40))) ** 3
        c = E.capped(w)
        assert abs(c.sum() - 1) < 1e-12 and c.max() <= E.CAP_SHARE + 1e-12
        free = c < E.CAP_SHARE - 1e-12               # the uncapped: one common rescaling
        assert np.allclose(c[~free], E.CAP_SHARE)
        r = c[free] / w[free]
        assert np.allclose(r, r[0]) if free.any() else True
        if free.any() and (~free).any():
            assert w[~free].min() >= w[free].max()
    assert np.allclose(E.capped(np.ones(5)), 0.2)       # fewer than 10: nothing to cap


def test_the_cap_comes_after_the_dense_restriction():
    """A toy half where capping before the restriction leaves a share above 0.10 after it."""
    class H(E.Half):
        def __init__(self, levels, n):
            self.level, self.n, self.bs = np.array(levels), np.array(n), [f"x{j}" for j in range(len(n))]
            self.family, self.flagged = list(self.bs), np.array([j == 0 for j in range(len(n))])
    levels = [-1.0] * 3 + [3.0 + 0.1 * j for j in range(12)]
    n = [8, 1, 1] + [8] * 12
    h = H(levels, n)
    comps = [(1.0, -1.0, 0.5)]
    t = E.tilt(h.level, comps)
    early = E.capped(t)
    ok_ = np.flatnonzero(h.n >= E.DENSE["min_pairs"])
    early = early[ok_] / early[ok_].sum()
    assert early.max() > E.CAP_SHARE + 0.01
    ok, w = h.weights(comps, "dense")
    assert list(ok) == list(ok_) and w.max() <= E.CAP_SHARE + 1e-12 and abs(w.sum() - 1) < 1e-12
    st = h.weight_stats(ok, w)
    assert st["flagged_weight_share"] == pytest.approx(w[0])      # read on the dense weights
    assert st["kish_benchmarks"] == pytest.approx(1 / np.sum(w ** 2))


def test_formative_runs_keep_the_shape(toy):
    H = _half(toy)
    for i in range(40):
        ents = E.draw_run(H, toy["stats"], "P-READING", (-0.5, 1.2), E.SEL_SEED, i)
        assert ents == E.draw_run(H, toy["stats"], "P-READING", (-0.5, 1.2), E.SEL_SEED, i)
        assert 5 <= len(ents) <= 12
        assert {e[0] for e in ents} <= set(H.bs)
        assert len({e[1] for e in ents}) == len(ents)
        dicts = [tuple(sorted(E.official_subject(toy["pool"].subject(e[0], e[1])).items())) for e in ents]
        assert len(set(dicts)) == len(ents)
        kept = [E.kept_items(e) for e in ents]
        assert all(len(k) >= E.FORMATIVE["min_items"] for k in kept)
        assert all(len(e[2]) <= E.FORMATIVE["max_items"] for e in ents)
        assert sum(len(k) for k in kept) <= E.FORMATIVE["cap"]
    picks = E.draw_picks(H, "P-READING", (-0.5, 1.2), E.run_rng(E.CAL_SEED, "C", "P-READING", 3))
    ents = E.draw_run(H, toy["stats"], "P-READING", (-0.5, 1.2), E.CAL_SEED, 3)
    assert sorted((H.bs[b], H.sids[b][j]) for b, j in picks) == sorted((e[0], e[1]) for e in ents)


def test_untilted_weights_are_uniform(toy):
    H = _half(toy)
    ok, w = H.weights(None, "formative")
    assert np.allclose(w, 1 / len(H.bs))
    ok, w = H.weights(None, "dense")
    assert np.allclose(w, 1 / len(ok)) and all(H.n[ok] >= E.DENSE["min_pairs"])


def test_dense_runs_keep_the_shape(toy):
    H = _half(toy)
    for i in range(20):
        ents = E.draw_run(H, toy["stats"], "D-READING", (-0.5, 1.2), E.SEL_SEED, i)
        bs = [e[0] for e in ents]
        assert len(set(bs)) <= E.DENSE["benchmarks"]
        assert all(bs.count(b) <= E.DENSE["pairs"] for b in set(bs))
        assert all(len(toy["pool"].levels[b]) >= E.DENSE["min_pairs"] for b in bs)
        assert all(len(e[2]) <= E.DENSE["max_items"] and e[3] is None for e in ents)
        for b in set(bs):
            ds = [tuple(sorted(E.official_subject(toy["pool"].subject(b, e[1])).items()))
                  for e in ents if e[0] == b]
            assert len(set(ds)) == len(ds)
    ents = E.draw_run(H, toy["stats"], "D-FLAT", None, E.SEL_SEED, 0, dense_max_items=80)
    assert all(len(e[2]) <= 80 for e in ents)


def test_the_first_pair_can_be_fixed(toy):
    H = _half(toy)
    b = H.bs[0]
    for i in range(5):
        picks = E.draw_formative(H, np.ones(len(H.bs)) / len(H.bs), np.random.default_rng([1, i]),
                                 first=(b, "drawn"))
        assert H.bs[picks[0][0]] == b
        picks = E.draw_formative(H, np.ones(len(H.bs)) / len(H.bs), np.random.default_rng([1, i]),
                                 first=(b, "fixed"))
        assert picks[0] == (0, 0)


def test_the_realised_level_is_testlikes_on_evaluated_responses(toy):
    for regime in ("P-FLAT", "D-FLAT"):
        ents, run = _toy_run(toy, regime, 2)
        mine = [E.describe_entry(toy["stats"], e) for e in ents]
        ref = T.describe(run)
        for m, r in zip(mine, ref):
            assert m["benchmark"] == r["pseudo"] and m["subject_id"] == r["subject_id"]
            assert m["eval_items"] == r["eval_items"] and m["items"] == r["items"]
            assert m["logit"] == pytest.approx(r["logit"], abs=1e-12)
            assert m["p"] == pytest.approx(r["p"], abs=1e-12)


def test_the_knob_objective_is_rss(toy):
    import experiments.regime_sensitivity as RS
    r = {"mean": -0.715 + 0.05, "sd": 1.772, "share_below": 0.0, "rest_mean": 0.0}
    assert E.objective(r, "READING") == pytest.approx(1.0)
    r = {"mean": 0.0, "sd": 0.0, "share_below": 0.118 + 0.01, "rest_mean": -0.226 - 0.1}
    assert E.objective(r, "MIXTURE") == pytest.approx(1.0 + 4.0)
    assert E.knob_grid("P-FLAT") == [None] and len(E.knob_grid("P-MIXTURE")) == 7 * 13
    assert len(E.knob_grid("D-READING")) == 33 * 9
    assert E.comps_of("P-MIXTURE", (0.1, -0.5)) == [(0.1, -4.38, 0.171), (0.9, -0.5, 1.151)]
    assert RS.KNOB_SCALE["mean"] == 0.05


def test_short_of_target_and_realisation_checks():
    cal = {"mean": -0.5, "sd": 1.7, "share_below": 0.05, "rest_mean": -0.2}
    assert E.short_of_target(cal, "P-READING")["short"]          # 0.215 off the mean
    assert not E.short_of_target(dict(cal, mean=-0.8), "P-READING")["short"]
    assert E.short_of_target(cal, "P-MIXTURE")["short"]          # share 0.068 off
    assert E.short_of_target(cal, "P-FLAT") is None
    chk = E.realisation_checks(cal, dict(cal, mean=cal["mean"] + 0.31), "P-READING")
    assert not chk["mean"]["ok"] and chk["sd"]["ok"] and set(chk) == {"mean", "sd"}
    assert set(E.realisation_checks(cal, cal, "P-MIXTURE")) == {"mean", "sd", "share_below", "rest_mean"}


# --- the split and the levels ----------------------------------------------------------------

def test_the_split_keeps_families_whole_and_blocks_flagged_units():
    E_ = [f"u{j:02d}" for j in range(30)]
    fam = {b: b for b in E_}
    fam.update({"u01": "u00", "u02": "u00"})
    flagged = {"u03", "u04", "u05", "u06", "u07", "u08", "u09"}
    levels = {b: [("s", 100, 10 + 2 * j)] for j, b in enumerate(E_)}
    pool = E.PoolView(E_, E_, E_, fam, flagged, {"u20", "u21"}, (), levels)
    half, units = E.split_halves(pool)
    for f in set(fam.values()):
        assert len({half[b] for b in E_ if fam[b] == f}) == 1
    fu = sorted((u for u in units if units[u]["flagged"]), key=lambda u: (units[u]["level"], u))
    for i in range(0, len(fu), 3):
        blk = fu[i:i + 3]
        assert sum(half[units[u]["members"][0]] == "S" for u in blk) == round(len(blk) / 3)
    other = sorted((u for u in units if not units[u]["flagged"] and units[u]["kind"] == "binary"),
                   key=lambda u: (units[u]["level"], u))
    for i in range(0, len(other), 8):
        blk = other[i:i + 8]
        assert sum(half[units[u]["members"][0]] == "S" for u in blk) == round(3 * len(blk) / 8)
    assert E.split_halves(pool)[0] == half


def _write_csvs(tmp, levels):
    lev = E.level_cc(levels)
    with open(tmp / "benchmarks.csv", "w") as f:
        f.write("benchmark,level_cc\n")
        for b in sorted(levels):
            f.write(f"{b},{lev[b]!r}\n")
    with open(tmp / "pairs.csv", "w") as f:
        f.write("benchmark,subject_id,n_items,k_first,logit_cc\n")
        for b in sorted(levels):
            for s, n, k in levels[b]:
                f.write(f"{b},{s},{n},{k},{E.logit_cc(n, k)!r}\n")


def test_levels_from_the_pool_files_against_the_planning_tables(toy, tmp_path):
    levels = toy["levels"]
    for b, v in levels.items():
        assert [(s, n, k) for s, n, k in v] == [(s, n, k) for s, n, k, _ in PL.pair_counts(b)]
    _write_csvs(tmp_path, levels)
    got = E.check_levels_csv(levels, tmp_path / "benchmarks.csv", tmp_path / "pairs.csv")
    assert got["max_abs_diff"]["logit"] == 0.0 and got["max_abs_diff"]["level"] <= 1e-15
    b = sorted(levels)[0]
    s, n, k = levels[b][0]
    bad = {**levels, b: [(s, n, k + 1)] + list(levels[b][1:])}
    with pytest.raises(E.PoolEvalError):
        E.check_levels_csv(bad, tmp_path / "benchmarks.csv", tmp_path / "pairs.csv")
    with open(tmp_path / "pairs.csv") as f:
        rows = f.read().splitlines()
    head, first = rows[0], rows[1].split(",")
    first[-1] = repr(float(first[-1]) + 1e-9)
    with open(tmp_path / "pairs.csv", "w") as f:
        f.write("\n".join([head, ",".join(first)] + rows[2:]) + "\n")
    with pytest.raises(E.PoolEvalError):
        E.check_levels_csv(levels, tmp_path / "benchmarks.csv", tmp_path / "pairs.csv")
    assert E.read_levels() == {b: [tuple(x) for x in v] for b, v in levels.items()}


def test_pair_statistics_hold_ids_and_counts_only(toy):
    for b in toy["pool"].E:
        st = toy["stats"][b]
        for sid, n, _ in toy["levels"][b]:
            keys = st.keys(st.codes(sid))
            assert len(keys) == n == len(set(keys))
            rn, rk = st.counts(sid, keys)
            assert np.all(rn >= 1) and np.all((rk >= 0) & (rk <= rn))
    for name in os.listdir(E.STATS_DIR):
        with np.load(os.path.join(E.STATS_DIR, name)) as z:
            for k in z.files:
                if z[k].dtype.kind == "U":
                    assert not any(TEXT in x for x in z[k].ravel().tolist())


# --- statistics -------------------------------------------------------------------------------

def _paired(rng, runs=30, fams=6, idents=8):
    out = []
    for r in range(runs):
        n = int(rng.integers(3, 9))
        meta = [{"family": f"f{rng.integers(fams)}", "ident": f"s{rng.integers(idents)}",
                 "benchmark": f"b{rng.integers(12)}"} for _ in range(n)]
        bx = rng.uniform(0.1, 0.3, (n, 6))
        out.append((meta, bx, bx + rng.normal(0.001, 0.01, (n, 6))))
    return out


def test_D_is_the_mean_of_run_differences_and_a_ratio():
    runs = _paired(np.random.default_rng(1))
    P = E.Paired(runs)
    run_d = [float(np.mean((bx - br) @ E.W6)) for _, bx, br in runs]
    assert E.D_of(P) == pytest.approx(np.mean(run_d), abs=1e-15)
    st = E.diff_stats(P)
    assert st["D"] == pytest.approx(np.mean(run_d), abs=1e-15)
    assert st["run_se"] == pytest.approx(np.std(run_d, ddof=1) / math.sqrt(len(run_d)))
    assert st["budgets"] == pytest.approx(
        np.mean([np.mean(bx - br, 0) for _, bx, br in runs], 0).tolist(), abs=1e-14)


def test_pb_uses_the_appearance_weight():
    """A toy table where the unweighted family mean has the other sign (V9a)."""
    m = lambda f: {"family": f, "ident": "s", "benchmark": f}   # noqa: E731
    d = lambda x: np.full((1, 6), x)                              # noqa: E731
    runs = [([m("A")], d(0.001), np.zeros((1, 6))),                # weight 1
            ([m("A")] + [m("B")] * 9, np.vstack([d(-0.005)] + [d(0.0)] * 9), np.zeros((10, 6)))]
    P = E.Paired(runs)
    pb, jk, per = E.pb_jk(P)
    assert per["A"] == pytest.approx((1 * 0.001 + 0.1 * -0.005) / 1.1) and per["A"] > 0
    assert per["B"] == 0.0 and pb == pytest.approx(per["A"] / 2) and pb > 0
    unweighted = np.mean([np.mean([0.001, -0.005]), 0.0])
    assert unweighted < 0                                           # another sign
    V, N = 0.001 - 0.0005, 2.0
    assert jk == pytest.approx(max((V - 0.0005) / (N - 1.1), V / (N - 0.9)))
    assert E.D_of(P) == pytest.approx(V / N)


def test_the_cluster_bootstrap_is_level_calibrations():
    rng = np.random.default_rng(2)
    runs = _paired(rng)
    metas = [[{"parent": m["family"], "subject": m["ident"]} for m in meta] for meta, _, _ in runs]
    vals = [(bx - br) @ E.W6 for _, bx, br in runs]
    boot = LC.Boot(metas, E.BOOTS, E.BOOT_SEED)
    P = E.Paired(runs)
    se, _ = E.cluster_boot(list(zip(P.fam, P.ident)), P.w, P.d)
    assert se == pytest.approx(boot.se(vals)[0], rel=1e-10)


def test_se2_seg_and_u95():
    P = E.Paired(_paired(np.random.default_rng(3)))
    st = E.diff_stats(P)
    sef = E.cluster_boot(P.fam, P.w, P.d)[0]
    ses = E.cluster_boot(P.ident, P.w, P.d)[0]
    sefs = E.cluster_boot(list(zip(P.fam, P.ident)), P.w, P.d)[0]
    assert (st["SEf"], st["SEs"], st["SEfs"]) == (sef, ses, sefs)
    assert st["SE2"] == pytest.approx(math.sqrt(max(0.0, sef ** 2 + ses ** 2 - sefs ** 2)))
    assert st["SEg"] == max(st["SEf"], st["SE2"])
    assert st["U95"] == st["D"] + 1.96 * st["SEg"]
    # a direct computation of SEf: the resamples of the family totals
    fams = sorted(set(P.fam))
    v = np.array([sum(w * d for f, w, d in zip(P.fam, P.w, P.d) if f == g) for g in fams])
    n = np.array([sum(w for f, w in zip(P.fam, P.w) if f == g) for g in fams])
    W = np.random.default_rng(0).multinomial(len(fams), np.full(len(fams), 1 / len(fams)), size=2000)
    assert sef == pytest.approx(float(((W @ v) / (W @ n)).std(ddof=1)), rel=1e-12)


def test_p99_is_numpys_linear_percentile():
    x = [0.1, 0.5, 0.2, 3.0, 0.05, 0.7, 0.9]
    assert E.p99(x) == float(np.percentile(x, 99))
    assert E.p99(x) == pytest.approx(0.9 + 0.94 * (3.0 - 0.9))


def test_latency_is_pooled_over_calls():
    rows = [{"arms": {"X": {"cpu_sum": 1.0, "calls": 10, "cpu_max": 0.5},
                      "A3": {"cpu_sum": 1.0, "calls": 20, "cpu_max": 0.1}}},
            {"arms": {"X": {"cpu_sum": 3.0, "calls": 10, "cpu_max": 0.7},
                      "A3": {"cpu_sum": 1.0, "calls": 20, "cpu_max": 0.2}}}]
    lx, la = E.latency(rows, "X"), E.latency(rows, "A3")
    assert lx["mean_s"] == 0.2 and lx["max_s"] == 0.7 and la["mean_s"] == 0.05
    assert E.ratio(lx, la) == pytest.approx(4.0)


# --- the selection and the gate ----------------------------------------------------------------

def test_selection_carries_a_candidate_that_meets_every_condition():
    sel = E.apply_selection(_sel_table())
    assert sel["X_star"] == "C1" and sel["Q"] == ["C1", "C1m", "C2"]
    assert all(v["eligible"] for v in sel["candidates"].values())


@pytest.mark.parametrize("path,value,cond", [
    ("C1.RS.TUNED", 0.004, "E1"), ("C1.RS.TUNED", 0.0021, "E1"), ("C1.RS.R1B", 0.0011, "E1"),
    ("C1.D.P-TUNED", 0.0021, "E2"), ("C1.D.P-MIXTURE", 0.0021, "E2"), ("C1.D.P-FLAT", 0.0021, "E2"),
    ("C1.D.P-READING-DS", 0.0001, "E2"), ("C1.D.P-AUDIT-DS", 0.0001, "E2"),
    ("C1.D.D-FLAT", 0.0011, "E2"), ("C1.D.D-READING", 0.0011, "E2"),
    ("C1.latency", {"ratio": 2.01, "max_s": 0.1}, "E3"), ("C1.latency", {"ratio": 1.0, "max_s": 2.01}, "E3"),
    ("C1.complete", False, "E4"), ("C1.checks_ok", False, "E4")])
def test_each_selection_condition_binds(path, value, cond):
    sel = E.apply_selection(_sel_table(**{path: value}))
    c = sel["candidates"]["C1"]
    assert not c["holds"][cond] and not c["eligible"]
    assert sel["X_star"] == "C1m"


def test_the_rs_mean_binds_on_its_own():
    t = _sel_table()
    t["C1"]["RS"] = {R: 0.0006 for R in E.RS_REGIMES}
    sel = E.apply_selection(t)
    assert not sel["candidates"]["C1"]["E1"]["mean_ok"]
    assert all(v["ok"] for v in sel["candidates"]["C1"]["E1"]["regimes"].values())


def test_the_twins_are_no_loss_conditions_and_the_old_bound_is_refused():
    for R in E.TWINS:
        assert not E.apply_selection(_sel_table(**{f"C1.D.{R}": 0.0001}))["candidates"]["C1"]["eligible"]
        assert E.apply_selection(_sel_table(**{f"C1.D.{R}": 0.0}))["candidates"]["C1"]["eligible"]
    # before N1 the twins were at the +0.002 loss bound; amendment 4b: apply_selection and
    # apply_gate refuse any rule but the constants block's literal
    old = dict(E.RULE, twin=E.RULE["loss"])
    with pytest.raises(E.PoolEvalError, match="RULE"):
        E.apply_selection(_sel_table(), rule=old)
    with pytest.raises(E.PoolEvalError, match="RULE"):
        E.apply_gate(_gate_input(), rule=old)


def _with_M(m):
    t = _sel_table()
    for X, v in m.items():
        t[X]["D"]["P-READING"] = t[X]["D"]["P-AUDIT"] = v
    return t


def test_the_tie_break_acts_within_Q():
    # an argmin of -0.0013 with an earlier candidate at -0.0009 (not in Q) carries the argmin
    sel = E.apply_selection(_with_M({"C1": -0.0009, "C1m": -0.0013, "C2": 0.0}))
    assert sel["Q"] == ["C1m"] and sel["X_star"] == "C1m"
    # two earlier members inside the margin: the first wins
    sel = E.apply_selection(_with_M({"C1": -0.0011, "C1m": -0.0012, "C2": -0.0015}))
    assert sel["X_star"] == "C1"
    # exactly at the minimum + 0.0005 is taken (inclusive)
    m = -0.0016
    sel = E.apply_selection(_with_M({"C1": m + 0.0005, "C1m": 0.0, "C2": m}))
    assert sel["X_star"] == "C1"
    sel = E.apply_selection(_with_M({"C1": m + 0.0005 + 1e-9, "C1m": 0.0, "C2": m}))
    assert sel["X_star"] == "C2"
    # the -0.001 threshold holds, inclusive
    sel = E.apply_selection(_with_M({"C1": -0.001, "C1m": 0.0, "C2": 0.0}))
    assert sel["Q"] == ["C1"]
    sel = E.apply_selection(_with_M({"C1": -0.000999, "C1m": 0.0, "C2": 0.0}))
    assert sel["X_star"] is None and sel["outcome"].startswith("no candidate carried")


def _gate_input(**over):
    st = {R: {"D": 0.0, "U95": 0.001, "PB": 0.0, "JK": 0.0} for R in E.REGIMES}
    for R in E.GAIN_REGIMES:
        st[R] = {"D": -0.004, "U95": -0.001, "PB": -0.003, "JK": -0.002}
    g = {"stats": st, "E1": True, "latency": {"ratio": 1.2, "max_s": 0.5, "dense_p99_s": 0.4,
                                              "dense_max_s": 5.0},
         "checks_ok": True, "complete": True, "invalid_X": 0, "invalid_A3": 0}
    for path, v in over.items():
        keys = path.split(".")
        d = g
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return g


def test_the_gate_passes_when_every_condition_holds():
    res = E.apply_gate(_gate_input())
    assert res["pass"] and all(res["holds"].values())


@pytest.mark.parametrize("path,value,cond", [
    ("stats.P-READING.D", -0.0019, "a"), ("stats.P-AUDIT.U95", 0.0, "a"),
    ("stats.P-READING.PB", 0.0001, "b"), ("stats.P-AUDIT.JK", -0.0009, "b"),
    ("stats.P-TUNED.D", 0.0021, "c"), ("stats.P-FLAT.D", 0.0021, "c"),
    ("stats.P-READING-DS.D", 0.0001, "c"), ("stats.P-AUDIT-DS.D", 0.0001, "c"),
    ("stats.D-READING.D", 0.0011, "d"), ("stats.D-FLAT.D", 0.0011, "d"),
    ("E1", False, "e"), ("latency.ratio", 2.01, "f"), ("latency.max_s", 2.01, "f"),
    ("latency.dense_p99_s", 1.01, "f"), ("latency.dense_max_s", 30.1, "f"),
    ("checks_ok", False, "g"), ("complete", False, "g"), ("invalid_X", 1, "g"), ("invalid_A3", 1, "g")])
def test_each_gate_condition_binds(path, value, cond):
    res = E.apply_gate(_gate_input(**{path: value}))
    assert not res["holds"][cond] and not res["pass"]
    assert sum(not v for v in res["holds"].values()) == 1


def test_gate_boundaries():
    assert E.apply_gate(_gate_input(**{"stats.P-READING.D": -0.002}))["pass"]
    assert E.apply_gate(_gate_input(**{"stats.P-READING-DS.D": 0.0}))["pass"]
    assert E.apply_gate(_gate_input(**{"stats.P-READING.JK": -0.001}))["pass"]
    assert E.apply_gate(_gate_input(**{"latency.dense_p99_s": 1.0}))["pass"]


# --- the lock, the fix scope, the check log ---------------------------------------------------

def test_the_import_closure_refuses_an_unpinned_module(tmp_path):
    for p, src in {"paiec/__init__.py": "", "paiec/a.py": "from paiec import b\n",
                   "paiec/b.py": "import numpy\n", "experiments/__init__.py": "",
                   "experiments/x.py": "def f():\n    import experiments.y\n    from paiec.a import g\n",
                   "experiments/y.py": "from .z import h\n", "experiments/z.py": ""}.items():
        os.makedirs(tmp_path / os.path.dirname(p), exist_ok=True)
        (tmp_path / p).write_text(src)
    got = E.import_closure(["experiments/x.py"], root=str(tmp_path))
    assert got == ["experiments/__init__.py", "experiments/x.py", "experiments/y.py", "experiments/z.py",
                   "paiec/__init__.py", "paiec/a.py", "paiec/b.py"]


def test_the_real_import_closure_is_pinned():
    got = E.import_closure(["experiments/pool_eval.py", "paiec/pool.py"])
    assert "experiments/__init__.py" in got and "experiments/__init__.py" in E.PINNED
    assert "paiec/baselines.py" in got and "experiments/ship_confirm.py" in got
    assert [p for p in got if p not in E.PINNED] == []


def test_the_rs_rows_digest_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RS_TASKS", {"A": 2, "B": 3})
    rows = []
    for R, n in (("A", 2), ("B", 3)):
        os.makedirs(tmp_path / R)
        for i in range(n):
            data = f"{R}{i}".encode()
            (tmp_path / R / f"{i}.json").write_bytes(data)
            rows.append((f"{R}/{i}.json", hashlib.sha256(data).hexdigest()))
    text = "".join(f"{p}\t{h}\n" for p, h in sorted(rows))
    got = E.rs_rows_digest(str(tmp_path))
    assert got == {"files": 5, "digest": hashlib.sha256(text.encode()).hexdigest()}
    (tmp_path / "B" / "2.json").write_bytes(b"B2!")
    assert E.rs_rows_digest(str(tmp_path))["digest"] != got["digest"]
    os.remove(tmp_path / "B" / "2.json")
    with pytest.raises(E.PoolEvalError):
        E.rs_rows_digest(str(tmp_path))


def _lock_state(files):
    return {"lock": {"files": dict(files), "constants_sha256": E.constants_sha256(),
                     "rule_sha256": hashlib.sha256(E.RULE_TEXT.encode()).hexdigest(),
                     "pool_frozen_ast": E.pool_frozen_sha(), "eval_loader": list(E.EVAL_LOADER),
                     "decision": list(E.DECISION), "decision_ast": E.decision_sha(),
                     "frozen_objects": E.frozen_digests(E.frozen_names()),
                     "streaming_client": {"commit": "c0"}, "utc": "2026-10-13T00:00:00Z"}}


@pytest.fixture
def locked(monkeypatch):
    files = {p: f"{hash(p) & 0xffff:064x}" for p in E.PINNED}
    monkeypatch.setattr(E, "lock_files", lambda: dict(files))
    monkeypatch.setattr(E, "client_commit", lambda: "c0")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for v in E.BLAS_VARS:
        monkeypatch.setenv(v, "1")
    return files


def test_the_lock_check_sees_any_change_outside_a_recorded_fix(locked, monkeypatch, tmp_path):
    st = _lock_state(locked)
    a = type("A", (), {"state": str(tmp_path / "s.json")})()
    assert E.lock_check(st, a) == []
    now = dict(locked)
    now["third_party/paiec_baseline/tools/streaming_ingestion.py"] = "f" * 64
    monkeypatch.setattr(E, "lock_files", lambda: dict(now))
    assert E.lock_check(st, a) == ["third_party/paiec_baseline/tools/streaming_ingestion.py changed since the lock"]
    monkeypatch.setattr(E, "client_commit", lambda: "c1")
    assert "the streaming client's clone moved" in E.lock_check(st, a)


def test_a_successor_is_accepted_only_in_scope(locked, monkeypatch, tmp_path):
    a = type("A", (), {"state": str(tmp_path / "s.json")})()
    now = dict(locked)
    now["experiments/pool_eval.py"] = "e" * 64
    monkeypatch.setattr(E, "lock_files", lambda: dict(now))
    st = _lock_state(locked)
    assert E.lock_check(st, a) == ["experiments/pool_eval.py changed since the lock"]
    st["fixes"] = [{"id": "f1", "in_scope": True, "files": {"experiments/pool_eval.py":
                                                            {"old": locked["experiments/pool_eval.py"],
                                                             "new": "e" * 64}}}]
    assert E.lock_check(st, a) == []
    monkeypatch.setattr(E, "constants_sha256", lambda path=None: "0" * 64)
    assert "the constants block changed" in E.lock_check(st, a)
    # a successor of a file outside the scope is never accepted
    now2 = dict(locked)
    now2["paiec/hier.py"] = "d" * 64
    monkeypatch.setattr(E, "lock_files", lambda: dict(now2))
    st["fixes"][0]["files"] = {"paiec/hier.py": {"old": locked["paiec/hier.py"], "new": "d" * 64}}
    assert "paiec/hier.py changed since the lock" in E.lock_check(_lock_state(locked) | {"fixes": st["fixes"]}, a)


def test_the_pool_successor_rule_is_the_frozen_ast():
    with open(os.path.join(ROOT, "paiec", "pool.py")) as f:
        src = f.read()
    assert E.frozen(src) == PL.frozen_ast(src)
    loader = src.replace("    ix = eval_index(b, root)\n", "    ix = eval_index(b, root)  # fixed\n    ix = ix\n")
    assert loader != src and E.frozen(loader) == E.frozen(src)
    arith = src.replace("def round_literal(x) -> float:", "def round_literal(x, d=3) -> float:")
    assert arith != src and E.frozen(arith) != E.frozen(src)
    tup = src.replace('"load_eval_pairs", "build_eval_indexes", "_eval_index_task")',
                      '"load_eval_pairs", "build_eval_indexes", "_eval_index_task", "bundle")')
    assert tup != src and E.frozen(tup) != E.frozen(src)


def test_fix_scope(locked, monkeypatch):
    st = _lock_state(locked)
    ok, changed, why = E.fix_scope(st)
    assert ok and changed == {}
    now = dict(locked, **{"experiments/pool_eval.py": "e" * 64})
    monkeypatch.setattr(E, "lock_files", lambda: dict(now))
    ok, changed, why = E.fix_scope(st)
    assert ok and set(changed) == {"experiments/pool_eval.py"}
    now["paiec/prior.py"] = "1" * 64
    ok, _, why = E.fix_scope(st)
    assert not ok and "paiec/prior.py is outside a fix's scope" in why
    now.pop("paiec/prior.py")
    now["paiec/prior.py"] = locked["paiec/prior.py"]
    now["tests/test_pool_eval.py"] = "2" * 64
    ok, _, why = E.fix_scope(st)
    assert not ok and any("locked tests" in w for w in why)


def _row(stage, R, i, x=0.1, cpu=0.01):
    return {"stage": stage, "regime": R, "i": i, "meta": [{"family": "f", "ident": "s", "benchmark": "b"}],
            "arms": {"A3": {"b": [[x] * 6], "cpu_sum": cpu, "cpu_max": cpu, "calls": 3, "invalid": []}},
            "task": {"utc": "now", "rss_gb": cpu}}


def test_record_fix_rescore_rule(tmp_path, locked, monkeypatch):
    rows = str(tmp_path / "rows")
    for R, i in (("P-READING", 0), ("P-READING", 1), ("P-READING", 2), ("P-AUDIT", 0)):
        E.write_json(E.row_path("S", R, i, rows=rows), _row("S", R, i))
    assert E.verification_set(rows) == [("S", "P-AUDIT", 0), ("S", "P-READING", 0), ("S", "P-READING", 2)]
    state_path = str(tmp_path / "state.json")
    E.save_stage("lock", _lock_state(locked)["lock"], state_path)
    monkeypatch.setattr(E, "run_locked_tests", lambda: {"returncode": 0, "tail": ["ok"]})
    a = type("A", (), {"state": state_path, "rows": rows, "verify": None, "cause": "a reader bug",
                       "failed": ["S:P-READING:1"], "resolves": ["score:S:task P-READING 1"]})()
    E.stage_record_fix(a)
    fix = E.load_state(state_path)["fixes"][0]
    assert fix["in_scope"] and fix["rescore"]["failed"] == [["S", "P-READING", 1]]
    assert sorted(map(tuple, fix["rescore"]["moved"])) == sorted(
        [("S", "P-READING", 1), ("S", "P-AUDIT", 0), ("S", "P-READING", 0), ("S", "P-READING", 2)])
    # the re-score: verification rows equal outside the timing fields -> verified
    for stage, R, i in [tuple(x) for x in fix["rescore"]["moved"]]:
        E.write_json(E.row_path(stage, R, i, rows=rows), _row(stage, R, i, cpu=0.5))
    E.stage_record_fix(type("A", (), {"state": state_path, "rows": rows, "verify": fix["id"]})())
    assert E.load_state(state_path)["fixes"][0]["verified"]["whole_stage_rescore"] == []
    # a verification row that differs in a scored field: the whole stage is re-scored
    E.stage_record_fix(type("A", (), {"state": state_path, "rows": rows, "verify": None, "cause": "x",
                                      "failed": [], "resolves": []})())
    fix2 = E.load_state(state_path)["fixes"][1]
    for stage, R, i in [tuple(x) for x in fix2["rescore"]["moved"]]:
        E.write_json(E.row_path(stage, R, i, rows=rows), _row(stage, R, i, x=0.2 if i == 0 else 0.1))
    E.stage_record_fix(type("A", (), {"state": state_path, "rows": rows, "verify": fix2["id"]})())
    assert E.load_state(state_path)["fixes"][1]["verified"]["whole_stage_rescore"] == ["S"]
    assert not any(os.path.exists(E.row_path("S", R, i, rows=rows)) for R, i in E.plan_tasks("S"))
    # the locked tests failing: out of scope, nothing moved
    monkeypatch.setattr(E, "run_locked_tests", lambda: {"returncode": 1, "tail": ["1 failed"]})
    E.stage_record_fix(type("A", (), {"state": state_path, "rows": rows, "verify": None, "cause": "y",
                                      "failed": [], "resolves": []})())
    fix3 = E.load_state(state_path)["fixes"][2]
    assert not fix3["in_scope"] and fix3["outcome"].startswith("archive-3 stays")


def test_the_check_log(tmp_path):
    p = str(tmp_path / "s.json")
    E.log_check("split", "split counts", False, {"x": 1}, p)
    ok, why = E.checks_passed(E.load_state(p), ["split"])
    assert not ok and "split: split counts" in why
    assert any("not in the log" in w for w in why)
    for n in E.REQUIRED_CHECKS["split"]:
        E.log_check("split", n, True, None, p)
    assert E.checks_passed(E.load_state(p), ["split"])[0]            # pre-lock: the re-run counts
    E.save_stage("lock", {"utc": "2000-01-01T00:00:00Z"}, p)
    E.log_check("score:S", "task P-READING 3", False, "boom", p)
    E.log_check("score:S", "task P-READING 3", True, None, p)
    st = E.load_state(p)
    assert not E.check_status(st, ["score:S"])[("score:S", "task P-READING 3")]   # no fix recorded
    st["fixes"] = [{"in_scope": True, "resolves": [["score:S", "task P-READING 3"]]}]
    assert E.check_status(st, ["score:S"])[("score:S", "task P-READING 3")]
    sel = E.apply_selection(_sel_table(**{"C1.checks_ok": False, "C1m.checks_ok": False,
                                          "C2.checks_ok": False}))
    assert sel["X_star"] is None
    assert not E.apply_gate(_gate_input(checks_ok=False))["pass"]


def test_the_resident_set_check_runs_after_the_rows_are_written(tmp_path, monkeypatch):
    rows = str(tmp_path / "rows")
    state = {"lock": {"mu0_m": -2.0, "digest": "d"}, "regimes": {}}
    monkeypatch.setattr(E, "postlock", lambda a, s: state)
    monkeypatch.setattr(E, "ScoreContext", lambda st, a, stage: object())
    done = []

    def task(ctx, R, i):
        done.append((R, i))
        return _row("S", R, i)
    monkeypatch.setattr(E, "score_pool_task", task)
    monkeypatch.setattr(E, "rss_gb", lambda: 2.0)
    a = type("A", (), {"stage": "S", "shard": "1/3", "state": str(tmp_path / "s.json"), "rows": rows})()
    # amendment 4b: a resumable stop checks the frozen objects too, against the lock's record
    E.save_stage("lock", {"frozen_objects": E.frozen_digests(E.frozen_names())}, a.state)
    with pytest.raises(SystemExit) as ex:
        E.stage_score(a)
    assert ex.value.code == E.RESUME_EXIT
    assert E.check_status(E.load_state(a.state), ["score:S"]) == {
        ("score:S", "frozen objects at the resumable stop of shard 1/3"): True}
    first = E.shard_tasks(E.plan_tasks("S"), 1, 3)
    assert done == [first[0]] and os.path.exists(E.row_path("S", *first[0], rows=rows))
    with pytest.raises(SystemExit):
        E.stage_score(a)
    assert done == first[:2]


def test_tasks_are_run_index_major_and_shards_partition_them():
    for stage in ("S", "C", "RS"):
        tasks = E.plan_tasks(stage)
        assert len(set(tasks)) == len(tasks)
        shards = [E.shard_tasks(tasks, k, 3) for k in (1, 2, 3)]
        assert sorted(sum(shards, [])) == sorted(tasks)
        idx = [i for _, i in tasks]
        assert idx == sorted(idx)
    S = E.plan_tasks("S")
    assert sum(R == "P-READING" for R, _ in S) == 80 and sum(R == "D-FLAT" for R, _ in S) == 20
    assert sum(R == "P-AUDIT-DS" for R, _ in S) == 80
    C = E.plan_tasks("C")
    assert sum(R == "P-FLAT" for R, _ in C) == 120 and sum(R == "D-READING" for R, _ in C) == 40
    assert len(E.plan_tasks("RS")) == 520


def test_no_feedback_is_read_before_the_gate(tmp_path):
    for p in (os.path.join(ROOT, "results", "hidden_runs.json"),
              os.path.join(ROOT, "results", "formative_run3.json"),
              os.path.join(ROOT, "results", "formative_feedback.json")):
        with pytest.raises(E.PoolEvalError):
            E.guarded_open(p)
    with E.guarded_open(os.path.join(ROOT, "results", "pool_fits.json") if os.path.exists(
            os.path.join(ROOT, "results", "pool_fits.json")) else __file__) as f:
        f.read(1)
    with open(os.path.join(ROOT, "experiments", "pool_eval.py")) as f:
        src = f.read()
    tree = __import__("ast").parse(src)
    users = set()
    for n in tree.body:
        if isinstance(n, __import__("ast").FunctionDef):
            names = {x.id for x in __import__("ast").walk(n) if isinstance(x, __import__("ast").Name)}
            if {"HIDDEN", "_OPEN"} & names:
                users.add(n.name)
    assert users == {"stage_describe_hidden", "guarded_open"}
    assert re.search(r"^open = guarded_open", src, re.M)


def test_describe_hidden_refuses_before_the_gate(tmp_path):
    a = type("A", (), {"state": str(tmp_path / "s.json"), "rows": str(tmp_path), "hidden": None})()
    with pytest.raises(SystemExit):
        E.stage_describe_hidden(a)


# --- a pool task end to end, on the synthetic pool with its own fits --------------------------

def test_a_pool_task_and_a_timing_run_end_to_end(toy, monkeypatch):
    import types
    PL.fit_all(toy["pool"].L, 1, log=lambda *a: None, fresh=False)
    rec = toy["tmp"] / "fits_record.json"
    rec.write_text(json.dumps({"mu0_m": {"locked": -2.0}}))
    ctx_f = E.FitsContext(str(rec))
    p0, h0, _ = PL.candidate_bundle("C1", (), ctx_f.free, ctx_f.fixed, ctx_f.L, ctx_f.U)
    monkeypatch.setattr(E, "a3_bundle", lambda: (p0, h0))
    cache = E.BundleCache(ctx_f)
    H = _half(toy, "S")
    comps = {"S": {"regimes": {R: {"runs": {"0": E.draw_run(H, toy["stats"], R, None, E.SEL_SEED, 0)}}
                               for R in ("P-READING", "D-FLAT")}}}
    ctx = types.SimpleNamespace(stage="S", pool=toy["pool"], fits=ctx_f, cache=cache, a3=(p0, h0),
                                cands=list(E.TIE_ORDER), lock_digest="d", comps=comps, shard="1/1")
    rows = {R: E.score_pool_task(ctx, R, 0) for R in ("P-READING", "P-READING-DS", "D-FLAT")}
    for R, row in rows.items():
        n = len(row["meta"])
        assert set(row["arms"]) == {"A3", "C1", "C1m", "C2"}
        for x, a in row["arms"].items():
            assert np.asarray(a["b"]).shape == (n, 6) and a["invalid"] == [] and a["calls"] > 0
            assert (a["bundle"] is None) == (x == "A3")
            assert ("calls_cpu" in a) == (R == "D-FLAT")
        assert TEXT not in json.dumps(E.clean(row))
        assert row["excl"] == list(PL.excl_for(sorted({m["benchmark"] for m in row["meta"]}),
                                               toy["pool"].family_of, ctx_f.L))
    real, twin = rows["P-READING"], rows["P-READING-DS"]
    assert [m["subject_id"] for m in real["meta"]] == [m["subject_id"] for m in twin["meta"]]
    assert [m["release"] for m in twin["meta"]] == [T.shift_date(m["release"], E.DATE_SHIFT)
                                                    for m in real["meta"]]
    assert {x: a["bundle"]["sha256"] for x, a in real["arms"].items() if a["bundle"]} == \
        {x: a["bundle"]["sha256"] for x, a in twin["arms"].items() if a["bundle"]}
    assert real["arms"]["C1m"]["bundle"]["hyper"]["mu0"] == -2.0
    # a selection table on these rows (one run a regime): every arm scored, no invalid output
    S = {"P-READING": [real], "P-READING-DS": [twin], "D-FLAT": [rows["D-FLAT"]]}
    table, _ = E.selection_table(S, {}, {}, True)
    assert all(table[X]["complete"] and table[X]["invalid"] == 0 for X in E.TIE_ORDER)
    assert table["C1"]["D"]["P-READING"] == pytest.approx(
        np.mean((np.asarray(real["arms"]["C1"]["b"]) - np.asarray(real["arms"]["A3"]["b"])) @ E.W6))
    # the fits-extra records run on these fits
    sl = E.r4_slopes(ctx_f.free, ctx_f.fixed, [b for b in toy["pool"].E if toy["half"][b] == "S"],
                     toy["pool"].flagged)
    assert sl["rows"] > 0 and set(sl["bins"]) == {"1", "2", "3+"}
    # a timing run: label-randomised, every arm timed, nothing scored
    monkeypatch.setattr(E, "score_preds", _no_score)
    out = E._timing_task(6, comps["S"]["regimes"]["D-FLAT"]["runs"]["0"], str(rec))
    assert out["labels"]["randomised"] == out["responses"] > 0 and out["dense"]
    assert set(out["arms"]) == {"A3", "C1", "C1m", "C2"} and all(v["invalid"] == 0 for v in out["arms"].values())
    assert "brier" not in json.dumps(E.clean(out)).lower()
    # the item read's CPU apart from the load's (PE 9 item 9)
    assert out["item_reads"] >= 1 and 0 <= out["item_read_cpu"] <= out["load_cpu"]
    assert out["load_cpu_without_item_read"] == pytest.approx(out["load_cpu"] - out["item_read_cpu"])


def test_the_item_read_timer_counts_items_of_and_restores_it(toy):
    ents = E.draw_run(_half(toy, "C"), toy["stats"], "P-FLAT", None, E.SEL_SEED, 0)
    orig = PL.items_of
    with E.ItemReadTimer() as t:
        run = E.load_run(ents)
        assert PL.items_of is not orig
    assert PL.items_of is orig
    assert t.calls == len({e[0] for e in ents}) and t.cpu >= 0
    assert [p.subject_id for p, _ in run] == [p.subject_id for p, _ in E.load_run(ents)]


def test_the_knob_search_and_the_cal_seed_value(toy, monkeypatch):
    monkeypatch.setattr(E, "GAUSS_M", (-1.0, 0.0))
    monkeypatch.setattr(E, "GAUSS_S", (1.0, 1.6))
    monkeypatch.setattr(E, "CAL_DRAWS", 30)
    monkeypatch.setattr(E, "N_RUNS", {"S": {"formative": 3, "dense": 2}, "C": {"formative": 3, "dense": 2}})
    bs = [b for b in toy["pool"].E if toy["half"][b] == "C"]
    k = E._knob_task("C", "P-READING", bs)
    assert len(k["settings"]) == 4
    objs = [r["objective"] for r in k["settings"]]
    assert k["argmin_index"] == int(np.argmin(objs)) and k["argmin"] == k["settings"][k["argmin_index"]]["knob"]
    g = E._plan_task("C", "P-READING", k["argmin"], bs, E.DENSE["max_items"])
    row = k["settings"][k["argmin_index"]]
    assert g["cal_proxy"]["mean"] == pytest.approx(row["mean"], abs=1e-12)
    assert g["cal_proxy"]["sd"] == pytest.approx(row["sd"], abs=1e-12)
    H = E.Half(E.PoolView.from_files(helm=False), "C", bs)
    ev = [E.describe_entry(toy["stats"], e)["logit"]
          for i in range(30) for e in E.draw_run(H, toy["stats"], "P-READING", tuple(k["argmin"]), E.CAL_SEED, i)]
    assert g["cal"] == E.realised(ev)
    assert sorted(g["runs"]) == [0, 1, 2] and g["seed"] == E.CONF_SEED
    assert g["runs"][0] == E.draw_run(H, toy["stats"], "P-READING", tuple(k["argmin"]), E.CONF_SEED, 0)
    assert g["weights"]["max_weight"][1] <= E.CAP_SHARE + 1e-12 or len(bs) < 10


def test_select_refuses_when_an_rs_row_changed(tmp_path, monkeypatch):
    state = {"lock": {"rs_rows": {"digest": "a" * 64}}}
    monkeypatch.setattr(E, "postlock", lambda a, s: state)
    monkeypatch.setattr(E, "missing_tasks", lambda stage, rows=None: [])
    monkeypatch.setattr(E, "rs_rows_digest", lambda root=None: {"files": 520, "digest": "b" * 64})
    a = type("A", (), {"state": str(tmp_path / "s.json"), "rows": str(tmp_path)})()
    with pytest.raises(SystemExit):
        E.stage_select(a)
    assert E.check_status(E.load_state(a.state), ["select"]) == {("select", "RS rows digest"): False}


def test_a_pool_successor_passes_the_lock_check_only_under_the_ast_rule(locked, monkeypatch, tmp_path):
    a = type("A", (), {"state": str(tmp_path / "s.json")})()
    st = _lock_state(locked)
    now = dict(locked, **{"paiec/pool.py": "9" * 64})
    monkeypatch.setattr(E, "lock_files", lambda: dict(now))
    st["fixes"] = [{"id": "f", "in_scope": True,
                    "files": {"paiec/pool.py": {"old": locked["paiec/pool.py"], "new": "9" * 64}}}]
    assert E.lock_check(st, a) == []                                  # the frozen AST is the lock's
    monkeypatch.setattr(E, "pool_frozen_sha", lambda names=E.EVAL_LOADER, path=None: "8" * 64)
    assert E.lock_check(st, a) == ["paiec/pool.py changed outside EVAL_LOADER"]
    monkeypatch.undo()


def test_a_reselection_naming_another_candidate_gives_archive_3_stays():
    rec = E.apply_selection(_with_M({"C1": 0.0, "C1m": -0.002, "C2": 0.0}))
    assert E.reselection({"X_star": "C1m"}, rec)["X_star"] == "C1m"
    got = E.reselection({"X_star": "C1"}, rec)
    assert got["X_star"] is None and got["outcome"].startswith("archive-3 stays")
    assert E.reselection(None, rec)["X_star"] == "C1m"


def test_records_that_must_hold_no_score_are_checked():
    assert E.assert_no_scores({"runs": {"0": {"calls": 3, "identical": True}},
                               "benchmarks": ["medhelm_medcalc_bench"]})
    for bad in ({"brier": [0.2]}, {"ALC": 0.2}, {"x": {"ece": 0.1}}):
        with pytest.raises(AssertionError):
            E.assert_no_scores(bad)


def _c_row(R, i, rng, gain):
    n = int(rng.integers(5, 9))
    meta = [{"family": f"f{rng.integers(6)}", "ident": f"n:m{rng.integers(9)}", "benchmark": f"b{rng.integers(12)}",
             "dated": bool(j % 2), "release": "2025-03-01" if j % 2 else "", "named": True,
             "id_row": {"A3": False, "C1": bool(j % 3)}, "flagged": False, "mcq": j == 0,
             "kind": "binary", "helm": False, "logit": float(rng.normal()), "v": 0.2, "rate": 0.3}
            for j in range(n)]
    base = rng.uniform(0.15, 0.25, (n, 6))
    x = base + (gain + rng.normal(0, 1e-4, (n, 6)) if R in E.GAIN_REGIMES else -1e-5)
    dense = E.REGIMES[R]["shape"] == "dense"

    def arm(b, cpu):
        a = {"b": b.tolist(), "ece": np.full((n, 6), 0.05).tolist(), "q0": [0.3] * n, "calls": 10,
             "cpu_sum": cpu * 10, "cpu_max": cpu, "wall_sum": cpu * 10, "wall_max": cpu, "failures": 0,
             "unconverged": 0, "invalid": [], "bundle": {"hyper": {"mu0": -2.5, "sigma_mu": 2.5, "T": 5.0}}}
        if dense:
            a["calls_cpu"] = [cpu] * 10
        return a
    return {"stage": "C", "regime": R, "i": i, "meta": meta,
            "arms": {"A3": arm(base, 0.01), "C1": arm(x, 0.012)}, "task": {}}


def test_summarise_and_gate_on_hand_built_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "N_RUNS", {"S": {"formative": 2, "dense": 1}, "C": {"formative": 6, "dense": 3}})
    rng = np.random.default_rng(5)
    rows = str(tmp_path / "rows")
    for R, i in E.plan_tasks("C"):
        E.write_json(E.row_path("C", R, i, rows=rows), _c_row(R, i, rng, -0.004))
    path = str(tmp_path / "s.json")
    E.save_stage("selection", {"X_star": "C1", "candidates": {"C1": {"holds": {"E1": True}}}}, path)
    lock = {"digest": "d", "bundles": {"C1": {"sha256": "ab" * 32,
                                              "level": {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}}}}
    monkeypatch.setattr(E, "postlock", lambda a, s: E.load_state(path))
    asked = []
    monkeypatch.setattr(E, "checks_passed", lambda state, stages, required=None: asked.append(stages) or (True, []))
    # amendment 4b: the stages end with the frozen-object check against the lock's record,
    # here of this process's objects as the test leaves them (N_RUNS, checks_passed)
    E.save_stage("lock", dict(lock, frozen_objects=E.frozen_digests(E.frozen_names())), path)
    a = type("A", (), {"state": path, "rows": rows, "fits_record": str(tmp_path / "none.json"), "root": None})()
    E.stage_summarise(a)
    summ = E.load_state(path)["summary"]
    assert summ["complete"] and summ["X_star"] == "C1"
    assert summ["stats"]["P-READING"]["D"] == pytest.approx(-0.004, abs=1e-4)
    assert summ["latency"]["ratio"] == pytest.approx(1.2) and summ["latency"]["dense_p99_s"] == pytest.approx(0.012)
    assert "dated" in summ["cuts"]["P-READING"] and "runs" in summ
    monkeypatch.setattr(E, "POOL_BUILD", str(tmp_path / "pool_build.json"))
    E.stage_gate(a)
    gate = E.load_state(path)["gate"]
    assert set(gate["holds"]) == set("abcdefg")
    assert gate["holds"]["c"] and gate["holds"]["d"] and gate["holds"]["e"] and gate["holds"]["f"]
    assert gate["holds"]["a"] == (gate["conditions"]["a"]["P-READING"]["U95"] < 0
                                  and gate["conditions"]["a"]["P-AUDIT"]["U95"] < 0)
    assert gate["holds"]["g"]
    assert asked[-1] == E.GATE_STAGES                       # (g) reads the block's stage list
    st = E.load_state(path)
    assert E.check_status(st, ["summarise", "gate"]) == {("summarise", "frozen objects at the end"): True,
                                                          ("gate", "frozen objects at the end"): True}
    if gate["pass"]:
        assert E.read_json(E.POOL_BUILD) == {"candidate": "C1", "bundle_sha256": "ab" * 32, "lock": "d",
                                             "level": {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}}
    else:
        assert not os.path.exists(E.POOL_BUILD)
    with pytest.raises(SystemExit):                        # read once
        E.stage_gate(a)


def test_the_lock_reads_the_amendment_to_the_next_heading(tmp_path):
    sha = "ab" * 32
    text = ("# Plan\n\n## Amendment 3 (x)\nold\n\n## Amendment 4 (2026-10-12): pool\n"
            f"The design (sha256\n`{sha}`).\n\n### 4.4 The decision rule (verbatim `RULE_TEXT`)\n\n"
            "```\nRULE\n  line two\n```\n\nafter\n\n## Amendment 5 (y)\nnext\n")
    p = tmp_path / "plan.md"
    p.write_text(text)
    sec = E.amendment_section(str(p))
    assert sec.startswith("## Amendment 4") and "## Amendment 5" not in sec and sec.endswith("after\n\n")
    assert E.rule_block(sec) == "RULE\n  line two\n"
    assert E.quoted_design_sha(sec) == sha
    assert E.amendment_section(str(tmp_path / "missing.md")) is None
    p.write_text(text.replace("## Amendment 5 (y)\nnext\n", ""))
    assert E.amendment_section(str(p)).endswith("after\n\n")


# --- amendment 4a: the decision arithmetic frozen at the lock --------------------------------------

def _source():
    with open(os.path.join(ROOT, "experiments", "pool_eval.py"), encoding="utf-8") as f:
        return f.read()


#: edits inside the frozen definitions (the gate's U95, JK's max, the budget weights twice)
DECISION_EDITS = (
    ('val(R, "D") <= rule["gain"] and val(R, "U95") < 0}', 'val(R, "D") <= rule["gain"]}'),
    ("    jk = max(((V - v[f]) / (N - n[f])", "    jk = min(((V - v[f]) / (N - n[f])"),
    ("W6 = np.asarray(WEIGHTS, float)", "W6 = np.asarray(WEIGHTS, float) / 2"),
    ("from paiec.evaluator import BUDGETS, WEIGHTS, Pair, Response  # noqa: E402",
     "from paiec.evaluator import BUDGETS, Pair, Response  # noqa: E402\nWEIGHTS = (1.0,) * 6"),
    ("    return float(np.percentile(np.asarray(x, float), 99))",
     "    return float(np.percentile(np.asarray(x, float), 98))"),
    # amendment 4b: the rows, the completeness checks, the per-pair Brier, the imports
    # (lock_verify 2.3's first two edits among them)
    ("            out[R].append(read_json(p))", "            if i:\n                out[R].append(read_json(p))"),
    ("    return [t for t in plan_tasks(stage) if not os.path.exists(row_path(stage, *t, rows=rows))]",
     "    return []"),
    ('    return os.path.join(rows or ROWS, stage, R, f"{i}.json")',
     '    return os.path.join(rows or ROWS, stage, R, f"{i + 1}.json")'),
    ("        br = [float(np.mean((preds[b][sl] - y) ** 2)) for b in BUDGETS]",
     "        br = [float(np.mean(np.abs(preds[b][sl] - y))) for b in BUDGETS]"),
    ("import math  # noqa: E402", "import cmath as math  # noqa: E402"),
    ("from collections import defaultdict  # noqa: E402",
     "from collections import OrderedDict as defaultdict  # noqa: E402"),
    # amendment 4b, the stages whole (lock_verify2 2.2's edits 3, 6a to 6e, 6h and 6i, which
    # its harness reading accepted) and the helpers they call that decide a check or a row set
    ('    res = apply_gate(gate_table(summ, state["selection"], checks_ok))',
     '    summ = dict(summ, complete=True)\n    res = apply_gate(gate_table(summ, state["selection"], checks_ok))'),
    ("    checks_ok, problems = checks_passed(load_state(a.state), GATE_STAGES)",
     "    checks_ok, problems = True, []"),
    ("    checks_ok, problems = checks_passed(load_state(a.state), GATE_STAGES)",
     "    checks_ok, problems = checks_passed(load_state(a.state), GATE_STAGES[:6])"),
    ('    miss = missing_tasks("C", a.rows)\n', "    miss = []\n"),
    ('    C = load_rows("C", a.rows)\n    gi = gate_inputs(C, star, miss)',
     '    C = {R: rs[1:] for R, rs in load_rows("C", a.rows).items()}\n    gi = gate_inputs(C, star, miss)'),
    ('    S, RSr = load_rows("S", a.rows), load_rows("RS", a.rows)',
     '    S, RSr = {R: rs[1:] for R, rs in load_rows("S", a.rows).items()}, load_rows("RS", a.rows)'),
    ("    checks_ok, problems = checks_passed(state, SELECT_STAGES)", "    checks_ok, problems = True, []"),
    ('    miss = missing_tasks("S", a.rows) + [("RS",) + t for t in missing_tasks("RS", a.rows)]',
     "    miss = []"),
    ("""    print(f"select: {rec['outcome']}")""", """    print(f"select -> {rec['outcome']}")"""),
    ("    problems = lock_check(state, a)", "    problems = []"),
    ('    entry = {"stage": stage, "name": name, "ok": bool(ok), "values": clean(values), "utc": utcnow()}',
     '    entry = {"stage": stage, "name": name, "ok": True, "values": clean(values), "utc": utcnow()}'),
    ('    problems = frozen_problems(load_state(a.state).get("lock") or {})', "    problems = []"),
    ('    return {"files": len(rows), "digest": sha_bytes(text.encode("utf-8"))}',
     '    return {"files": len(rows), "digest": "9a601549b2d99739487e7c3808da616f24f9072506671043de878020f59a9b5c"}'),
    ('    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")', '    return "2000-01-01T00:00:00Z"'),
    ("    return h.hexdigest()[:n] if n else h.hexdigest()", "    return h.hexdigest()[:n]"),
    ("    return hashlib.sha256(b).hexdigest()", "    return hashlib.sha1(b).hexdigest()"))
#: edits outside them: harness and I/O code a post-lock fix may still change, among them
#: the lock check itself and the descriptive parts of the records (UNFROZEN_READS)
HARNESS_EDITS = (
    ('        print(f"{stage}: " + ", ".join(f"{R} {have[R]}/{tot[R]}" for R in tot))',
     '        print(f"{stage} -> " + ", ".join(f"{R} {have[R]}/{tot[R]}" for R in tot))'),
    ('    """{path: sha256 or None (missing)} of every PINNED file."""',
     '    """Every PINNED file\'s sha256, None for a missing one."""'),
    ('        rate = k / n if n else float("nan")', '        rate = (k / n) if n > 0 else float("nan")'),
    ("    return sorted(set(problems))", "    return sorted(set(problems), key=str)"),
    ('    return {"runs": len(alc), "ALC": float(np.mean(alc)) if alc else None,',
     '    return {"runs": int(len(alc)), "ALC": float(np.mean(alc)) if alc else None,'))


def test_the_decision_ast_sees_the_decision_arithmetic_and_nothing_else():
    src = _source()
    base = E.decision_ast(src)
    for old, new in DECISION_EDITS:
        assert src.count(old) == 1, old
        assert E.decision_ast(src.replace(old, new)) != base, old
    # a later redefinition, a tuple assignment, a global rebinding, a def inside a block
    for tail in ("\napply_gate = lambda g, **k: {'pass': True}\n",
                 "\nratio, _unused = (lambda lx, la: 1.0), 0\n",
                 "\ndef _helper():\n    global latency\n    latency = None\n",
                 "\nif True:\n    def p99(x):\n        return 0.0\n",
                 "\nfrom math import fsum as D_of\n"):
        assert E.decision_ast(src + tail) != base, tail
    for old, new in HARNESS_EDITS:
        assert src.count(old) == 1, old
        assert E.decision_ast(src.replace(old, new)) == base, old
    assert E.decision_ast(src + "\n# a comment\n") == base
    assert E.decision_ast(src.replace("def pb_jk(P):", "def pb_jk(P):  # PB and JK")) == base


def test_the_frozen_definitions_name_their_helpers():
    """Every module-level def, class or assignment a frozen definition reads is frozen too,
    is in the constants block, or is one of three I/O names (no arithmetic)."""
    import ast
    src = _source()
    tree = ast.parse(src)
    a, b = src.index(E.CONSTANTS_START), src.index(E.CONSTANTS_END)
    first, last = src[:a].count("\n") + 1, src[:b].count("\n") + 1
    own = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            for n in E.decision_bound(node):
                own.setdefault(n, []).append(node.lineno)
    block = {n for n, lines in own.items() if all(first <= x <= last for x in lines)}
    # where the freeze stops (UNFROZEN_READS), written here as well as in the script, so
    # that moving either needs both edits: the JSON I/O and the feedback guard, the paths
    # the workers share (ROWS: row_path's default rows directory), the stop a failed check
    # raises, the lock check itself (postlock, which runs, logs and enforces it, is frozen),
    # and the descriptive parts of the selection and summary records
    unfrozen = {"io": ("open", "read_json", "load_state", "save_stage", "_locked_update",
                       "write_json", "clean"),
                "paths": ("ROWS", "RS_ROWS", "POOL_BUILD"), "errors": ("PoolEvalError",),
                "lock_check": ("lock_check",),
                "descriptive": ("power", "floor_active_set", "cuts", "arm_summary",
                                "run_descriptives", "reweighting_check", "PoolView")}
    assert E.UNFROZEN_READS == unfrozen
    names_unfrozen = {n for v in unfrozen.values() for n in v}
    assert not names_unfrozen & (set(E.DECISION) | block) and names_unfrozen <= set(own)
    allowed = set(E.DECISION) | block | names_unfrozen
    read_unfrozen = set()
    imported = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= E.decision_bound(node)
    for node in tree.body:
        names = E.decision_bound(node) & set(E.DECISION)
        in_block = first <= node.lineno <= last and isinstance(node, ast.FunctionDef)
        if not names and not in_block:
            continue
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert not (used & set(own)) - allowed, (sorted(names), sorted((used & set(own)) - allowed))
        read_unfrozen |= used & names_unfrozen
        # amendment 4b: every import the frozen code (and the block's functions) reads is frozen;
        # a module a definition imports itself (`import ast` in its body) is its own binding,
        # part of the frozen code
        local = {(a.asname or a.name).split(".")[0] for m in ast.walk(node)
                 if isinstance(m, (ast.Import, ast.ImportFrom)) for a in m.names}
        bad = (used & imported) - set(E.DECISION) - local
        assert not bad, (node.lineno, sorted(bad))
    # the list is exact: every name it holds is read by a frozen definition
    assert read_unfrozen == names_unfrozen, sorted(names_unfrozen - read_unfrozen)
    # what the amendments name is in the tuple: 4a's arithmetic, 4b's rows and completeness,
    # and "the selection and the gate" whole, with the helpers that decide a check
    for name in ("Paired", "D_of", "cluster_boot", "diff_stats", "pb_jk", "latency", "p99",
                 "apply_selection", "apply_gate", "selection_table", "gate_inputs", "gate_table",
                 "load_rows", "missing_tasks", "row_path", "score_preds",
                 "stage_select", "stage_summarise", "stage_gate", "postlock", "frozen_check",
                 "log_check", "utcnow", "rs_rows_digest", "file_sha", "sha_bytes"):
        assert name in E.DECISION


def _edited(tmp_path, old, new):
    src = _source()
    assert src.count(old) == 1
    p = tmp_path / "pool_eval_fix.py"
    p.write_text(src.replace(old, new), encoding="utf-8")
    return str(p)


def test_a_fix_to_the_decision_arithmetic_is_outside_the_fix_scope(locked, monkeypatch, tmp_path):
    a = type("A", (), {"state": str(tmp_path / "s.json")})()
    st = _lock_state(locked)
    real = E.decision_sha
    now = dict(locked, **{"experiments/pool_eval.py": "e" * 64})
    monkeypatch.setattr(E, "lock_files", lambda: dict(now))
    fix = [{"id": "f1", "in_scope": True, "files": {"experiments/pool_eval.py":
                                                    {"old": locked["experiments/pool_eval.py"],
                                                     "new": "e" * 64}}}]
    for old, new in DECISION_EDITS:
        path = _edited(tmp_path, old, new)
        monkeypatch.setattr(E, "decision_sha", lambda names=E.DECISION, path_=None, p=path: real(names, p))
        ok, changed, why = E.fix_scope(st)
        assert not ok and set(changed) == {"experiments/pool_eval.py"}
        assert any("frozen decision arithmetic" in w and "outside the fix scope" in w for w in why)
        assert any("frozen decision arithmetic" in x for x in E.lock_check(dict(st, fixes=fix), a))
    for old, new in HARNESS_EDITS:
        path = _edited(tmp_path, old, new)
        monkeypatch.setattr(E, "decision_sha", lambda names=E.DECISION, path_=None, p=path: real(names, p))
        ok, changed, why = E.fix_scope(st)
        assert ok and why == []
        assert E.lock_check(dict(st, fixes=fix), a) == []
    # a lock without the decision record refuses every successor of this file
    bare = {"lock": {k: v for k, v in st["lock"].items() if k not in ("decision", "decision_ast")}}
    ok, _, why = E.fix_scope(bare)
    assert not ok and any("no decision AST" in w for w in why)


def test_the_lock_digest_covers_the_decision_and_the_degenerate_list():
    rec = {"files": {}, "decision_ast": "a" * 64,
           "degenerate": {"threshold": 50.0, "benchmarks": ["finegrain_t2i", "multi_moe", "revolve"]}}
    d = E.lock_digest(rec)
    assert d != E.lock_digest(dict(rec, decision_ast="b" * 64))
    assert d != E.lock_digest(dict(rec, amendment_4a_sha256="c" * 64))
    assert d != E.lock_digest(dict(rec, amendment_4c_sha256="e" * 64))            # amendment 4c
    assert d != E.lock_digest(dict(rec, hard_stops=dict(E.HARD_STOPS, lock="2026-10-13T23:59:59Z")))
    assert d != E.lock_digest(dict(rec, degenerate=dict(rec["degenerate"], benchmarks=[])))
    assert d == E.lock_digest(dict(rec, utc="later"))


def test_a_dry_lock_logs_under_its_own_stage_and_records_amendment_4a(tmp_path):
    """A dry lock (even a failed one, here with no fits record) never stands for the lock's
    own check, before or after the lock; it records amendment 4a's text's digest."""
    import types
    state = tmp_path / "s.json"
    a = types.SimpleNamespace(state=str(state), dry=True, fits_record=str(tmp_path / "none.json"))
    E.stage_lock(a)
    st = json.loads(state.read_text())
    assert "lock_dry" in st and "lock" not in st and not st["lock_dry"]["ok"]
    assert [(c["stage"], c["name"], c["ok"]) for c in st["checks"]] == [("lock-dry", "lock (dry)", False)]
    assert E.checks_passed(st, ["lock"])[1] == ["lock: lock (not in the log)"]
    text = E.amendment_section(os.path.join(ROOT, E.AMENDMENT_FILE), E.AMENDMENT_4A_HEADING)
    if text is None:
        pytest.skip("amendment 4a is not in docs/plans/p2_final_plan.md here")
    assert text.startswith("## Amendment 4a") and "item variance above 50" in text
    assert st["lock_dry"]["amendment_4a_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert "## Amendment 4a" not in E.amendment_section(os.path.join(ROOT, E.AMENDMENT_FILE))
    # amendments 4b and 4c too, and 4c's hard stops (a dry lock reads the committed plan here)
    for key, heading in (("amendment_4b_sha256", E.AMENDMENT_4B_HEADING),
                         ("amendment_4c_sha256", E.AMENDMENT_4C_HEADING)):
        t = E.amendment_section(os.path.join(ROOT, E.AMENDMENT_FILE), heading)
        if t is not None:
            assert st["lock_dry"][key] == hashlib.sha256(t.encode()).hexdigest()
    assert st["lock_dry"]["hard_stops"] == E.HARD_STOPS
    assert st["lock_dry"]["hard_stop"] == "2026-10-15T23:59:59Z"


def _own_decision_ast(source, names):
    """This test file's own copy of amendment 4a's AST rule, so that the lock's digest is
    recomputed by code a post-lock fix cannot reach (the locked tests run unchanged)."""
    import ast
    names = set(names)
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def bound(node):
        out, todo = set(), [node]
        while todo:
            n = todo.pop()
            if isinstance(n, defs):
                out.add(n.name)
                out.update(x for m in ast.walk(n) if isinstance(m, ast.Global) for x in m.names)
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, ast.alias):
                out.add((n.asname or n.name).split(".")[0])
            todo.extend(ast.iter_child_nodes(n))
        return out
    body = [n for n in ast.parse(source).body if bound(n) & names]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def test_the_decision_ast_is_recomputed_independently_and_matches_the_lock():
    src = _source()
    assert _own_decision_ast(src, E.DECISION) == E.decision_ast(src)
    for old, new in DECISION_EDITS + HARNESS_EDITS:
        assert _own_decision_ast(src.replace(old, new), E.DECISION) == \
            E.decision_ast(src.replace(old, new))
    state = os.path.join(ROOT, "results", "pool_eval.json")
    lk = (json.load(open(state)).get("lock") or {}) if os.path.exists(state) else {}
    if not lk.get("decision_ast"):
        pytest.skip("no lock with a decision record in results/pool_eval.json")
    assert list(E.DECISION) == lk["decision"]
    assert hashlib.sha256(_own_decision_ast(src, lk["decision"]).encode()).hexdigest() == lk["decision_ast"]
    assert lk["degenerate"]["benchmarks"] == list(E.DEGENERATE_EXPECTED)


# --- amendment 4b: the wider freeze, the frozen objects, the legacy data -------------------------

def test_the_frozen_names_are_the_blocks_and_the_frozen_definitions():
    names = E.frozen_names()
    assert names == sorted(set(names))
    for n in ("RULE", "REGIMES", "LOSS_BOUNDS", "GATE_STAGES", "PINNED", "DECISION", "BOOTS",
              "classify_output", "decision_ast", "object_digest", "frozen_problems",
              "diff_stats", "apply_gate", "load_rows", "score_preds", "W6", "WEIGHTS",
              "BUDGETS", "np", "os", "defaultdict", "O", "SELECT_STAGES",
              # amendment 4b: the selection and the gate whole, and what decides their checks
              "stage_select", "stage_summarise", "stage_gate", "postlock", "frozen_check",
              "log_check", "utcnow", "datetime", "timezone", "rs_rows_digest", "file_sha",
              "sha_bytes", "UNFROZEN_READS"):
        assert n in names, n
    # a comprehension's own variable (GAUSS_M's k) is no module name; the harness outside
    # DECISION is not frozen (UNFROZEN_READS among it)
    for n in ("k", "lock_check", "read_json", "load_state", "save_stage", "ROWS", "PL", "cuts",
              "stage_score", "stage_record_fix", "fix_scope"):
        assert n not in names, n


def test_frozen_problems_see_a_run_time_change_and_a_rebinding(monkeypatch):
    lk = {"frozen_objects": E.frozen_digests(E.frozen_names())}
    assert E.frozen_problems(lk) == []
    assert E.frozen_problems({}) == ["the lock records no frozen objects (amendment 4b)"]
    monkeypatch.setitem(E.RULE, "gain", 1.0)                 # lock_verify 2.4's uncaught case
    bad = E.frozen_problems(lk)
    assert "RULE is not the object the lock recorded (amendment 4b)" in bad
    assert any(x.startswith("apply_gate ") for x in bad)     # its default is that dict
    monkeypatch.undo()
    assert E.frozen_problems(lk) == []
    for name, value in (("diff_stats", lambda P, **k: {"D": 0.0}), ("BOOTS", 200),
                        ("TWINS", {}), ("np", math), ("GATE_STAGES", ("lock",)),
                        ("object_digest", lambda x: "0" * 64)):
        monkeypatch.setattr(E, name, value)
        got = E.frozen_problems(lk)
        assert got and (f"{name} is not the object the lock recorded (amendment 4b)" in got
                        or name == "object_digest"), (name, got)
        monkeypatch.undo()
    monkeypatch.setattr(E.Paired, "subset", lambda self, mask: self)
    assert "Paired is not the object the lock recorded (amendment 4b)" in E.frozen_problems(lk)
    monkeypatch.undo()
    old = E.W6.copy()
    try:
        E.W6[0] = 0.5
        assert "W6 is not the object the lock recorded (amendment 4b)" in E.frozen_problems(lk)
    finally:
        E.W6[:] = old
    monkeypatch.delattr(E, "RS_SEED")
    assert "RS_SEED is not the object the lock recorded (amendment 4b)" in E.frozen_problems(lk)
    monkeypatch.undo()
    # an object changed into one with no canonical form is reported, not raised
    monkeypatch.setattr(E, "BOOTS", object())
    assert E.frozen_problems(lk) == ["BOOTS is not the object the lock recorded (amendment 4b)"]
    monkeypatch.undo()
    assert E.frozen_problems(lk) == []


FROZEN_SCRIPT = r"""
import hashlib, json, runpy, sys
sys.path.insert(0, sys.argv[2])
g = runpy.run_path(sys.argv[1], run_name=sys.argv[3])
print(json.dumps(g["frozen_digests"](g["frozen_names"]()), sort_keys=True))
"""


def test_the_frozen_digests_are_the_same_in_every_process():
    """Another interpreter, another hash seed, and the file run under another module name
    (a stage runs it as __main__) give the same digests."""
    import subprocess
    import sys
    here = json.dumps(E.frozen_digests(E.frozen_names()), sort_keys=True)
    for seed, name in (("0", "__pool_eval_copy__"), ("5", "experiments.pool_eval")):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        env.pop("PYTHONPATH", None)
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", FROZEN_SCRIPT,
                            os.path.join(ROOT, "experiments", "pool_eval.py"), ROOT, name],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert r.stdout.strip().splitlines()[-1] == here


def test_object_digest_is_type_tagged():
    d = E.object_digest
    assert d((1, 2)) != d([1, 2]) and d(1) != d(1.0) and d(True) != d(1) and d("1") != d(1)
    assert d({"a": 1, "b": 2}) != d({"b": 2, "a": 1})                     # order is kept
    assert d(np.array([1.0, 2.0])) != d(np.array([1.0, 2.5])) and d(-0.0) != d(0.0)
    assert d(E.diff_stats) == d(E.diff_stats) and d(E.diff_stats) != d(E.pb_jk)
    with pytest.raises(TypeError):
        d(object())


def test_the_lock_check_and_the_fix_scope_see_a_changed_frozen_object(locked, monkeypatch, tmp_path):
    a = type("A", (), {"state": str(tmp_path / "s.json")})()
    st = _lock_state(locked)
    assert E.lock_check(st, a) == []
    monkeypatch.setitem(E.REGIMES["D-FLAT"], "shape", "formative")
    assert "REGIMES is not the object the lock recorded (amendment 4b)" in E.lock_check(st, a)
    ok, _, why = E.fix_scope(st)
    assert not ok and any(w.startswith("REGIMES is not the object") for w in why)


def test_a_stage_that_changes_a_frozen_object_fails_at_its_end(tmp_path, monkeypatch):
    path = str(tmp_path / "s.json")
    E.save_stage("lock", {"frozen_objects": E.frozen_digests(E.frozen_names())}, path)
    E.save_stage("summary", {"X_star": "C1", "stats": {}, "latency": {}, "complete": True,
                             "invalid_X": 0, "invalid_A3": 0}, path)
    E.save_stage("selection", {"candidates": {"C1": {"holds": {"E1": True}}}}, path)
    monkeypatch.setattr(E, "POOL_BUILD", str(tmp_path / "pool_build.json"))

    def mutate(state, a=None):                     # a harness function changing a block object
        E.TWINS["P-FLAT"] = "P-READING"
        return []
    # the lock check is harness (UNFROZEN_READS); postlock, frozen, runs it and logs its pass
    monkeypatch.setattr(E, "lock_check", mutate)
    a = type("A", (), {"state": path, "rows": str(tmp_path / "rows")})()
    try:
        with pytest.raises(SystemExit):
            E.stage_gate(a)
    finally:
        E.TWINS.pop("P-FLAT", None)
    st = E.load_state(path)
    assert "gate" not in st and not os.path.exists(E.POOL_BUILD)
    assert E.check_status(st, ["gate"]) == {("gate", "lock check"): True,
                                            ("gate", "frozen objects at the end"): False}
    end = [c for c in st["checks"] if c["name"] == "frozen objects at the end"][0]
    assert end["values"] == ["TWINS is not the object the lock recorded (amendment 4b)"]


def test_apply_selection_and_apply_gate_refuse_another_rule(monkeypatch):
    assert E.block_literals("RULE", "LOSS_BOUNDS")["RULE"] == E.RULE
    for kw in ({"rule": dict(E.RULE, gain=-0.003)}, {"bounds": E.LOSS_BOUNDS[:-1]},
               {"bounds": tuple(list(b) for b in E.LOSS_BOUNDS)}):
        with pytest.raises(E.PoolEvalError, match="amendment 4b"):
            E.apply_gate(_gate_input(), **kw)
        with pytest.raises(E.PoolEvalError, match="amendment 4b"):
            E.apply_selection(_sel_table(), **kw)
    with pytest.raises(E.PoolEvalError, match="RS_BOUNDS"):
        E.apply_selection(_sel_table(), rs_bounds=E.RS_BOUNDS[:-1])
    with pytest.raises(E.PoolEvalError, match="TIE_ORDER"):
        E.apply_selection(_sel_table(), order=("C1m", "C1", "C2"))
    # a run-time change of the module's own objects is refused too (lock_verify 2.4)
    monkeypatch.setitem(E.RULE, "gain", 1.0)
    with pytest.raises(E.PoolEvalError, match="RULE"):
        E.apply_gate(_gate_input())
    monkeypatch.undo()
    monkeypatch.setattr(E, "GAIN_REGIMES", ("P-FLAT",))
    with pytest.raises(E.PoolEvalError, match="GAIN_REGIMES"):
        E.apply_selection(_sel_table())
    monkeypatch.undo()
    assert E.apply_gate(_gate_input())["pass"] and E.apply_selection(_sel_table())["X_star"] == "C1"


def test_the_static_guard_sees_a_block_name_bound_outside_the_block(tmp_path):
    src = _source()
    for tail in ('\ndef _h():\n    RULE["gain"] = 1.0\n', "\ndef _h():\n    global BOOTS\n    BOOTS = 1\n",
                 "\nTWINS = {}\n", "\ndef _h():\n    REGIMES.update({})\n",
                 "\ndef _h():\n    LOSS_BOUNDS += ((\"P-FLAT\", \"loss\"),)\n",
                 "\ndef _h():\n    del PINNED[\"dist/archive-3.zip\"]\n",
                 "\nfrom math import pi as BOOTS\n", "\ndef classify_output(v):\n    return True, 0.5, ''\n",
                 "\nfor RS_SEED in range(2):\n    pass\n"):
        p = tmp_path / "pe.py"
        p.write_text(src + tail, encoding="utf-8")
        assert E.block_binding_problems(str(p)), tail
    p.write_text(src + "\ndef _h():\n    x = RULE['gain']\n    return [k for k in REGIMES]\n", encoding="utf-8")
    assert E.block_binding_problems(str(p)) == []


def test_the_lock_records_the_frozen_objects_and_the_legacy_data():
    rec = {"files": {}, "frozen_objects": {"RULE": "a" * 64}, "amendment_4b_sha256": "b" * 64}
    d = E.lock_digest(rec)
    assert d != E.lock_digest(dict(rec, frozen_objects={"RULE": "c" * 64}))
    assert d != E.lock_digest(dict(rec, amendment_4b_sha256="d" * 64))
    if not all(os.path.isdir(os.path.join(ROOT, "data", b)) for b in PL.LEGACY):
        pytest.skip("the legacy five's tables are not under data/")
    assert E.legacy_data_problems() == []


def test_settings_need_paiec_data_unset_or_data(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for v in E.BLAS_VARS:
        monkeypatch.setenv(v, "1")
    monkeypatch.delenv("PAIEC_DATA", raising=False)
    assert E.settings_ok()
    monkeypatch.setenv("PAIEC_DATA", os.path.join(ROOT, "data") + os.sep)
    assert E.settings_ok()
    monkeypatch.setenv("PAIEC_DATA", "/elsewhere/data")
    assert not E.settings_ok()


def test_amendments_4b_and_4c_are_read_to_their_next_heading():
    plan = os.path.join(ROOT, E.AMENDMENT_FILE)
    text = E.amendment_section(plan, E.AMENDMENT_4B_HEADING)
    if text is None:
        pytest.skip("amendment 4b is not in docs/plans/p2_final_plan.md here")
    assert text.startswith("## Amendment 4b") and "s2_res" in text and "score_preds" in text
    a4a = E.amendment_section(plan, E.AMENDMENT_4A_HEADING)
    assert "## Amendment 4b" not in a4a
    c = E.amendment_section(plan, E.AMENDMENT_4C_HEADING)
    if c is None:
        pytest.skip("amendment 4c is not in docs/plans/p2_final_plan.md here")
    assert "## Amendment 4c" not in text and c.startswith("## Amendment 4c")
    # 4c's table: each stop, as RULE_TEXT names it, and where it moved
    norm = re.sub(r"\s+", " ", c)
    for k, was in E.HARD_STOPS_RULE_TEXT.items():
        now = E.HARD_STOPS[k][:10]
        assert re.search(rf"\| {was} \| {now} \|", norm), (k, was, now)


def test_the_hard_stops_are_amendment_4cs(monkeypatch):
    """Amendment 4c: RULE_TEXT keeps its dates; the code applies each moved by two days."""
    assert E.hard_stop_problems() == []
    for k, d in E.HARD_STOPS_RULE_TEXT.items():
        assert f"{d} 23:59 UTC" in re.sub(r"\s+", " ", E.RULE_TEXT)
        assert E.HARD_STOPS[k] > f"{d}T23:59:59Z"
    monkeypatch.setitem(E.HARD_STOPS, "lock", "2026-10-13T23:59:59Z")
    assert E.hard_stop_problems() == ["HARD_STOPS['lock'] is 2026-10-13T23:59:59Z, not "
                                      "2026-10-15T23:59:59Z (amendment 4c)"]
    monkeypatch.undo()
    assert E.hard_stop_problems(E.RULE_TEXT.replace("2026-10-20 23:59 UTC", "2026-10-21 23:59 UTC")) == \
        ["RULE_TEXT does not name 2026-10-20 23:59 UTC (final)"]
    monkeypatch.setattr(E, "HARD_STOPS", {k: v for k, v in E.HARD_STOPS.items() if k != "final"})
    assert "HARD_STOPS and HARD_STOPS_RULE_TEXT name different stops" in E.hard_stop_problems()


def test_the_lock_time_record_uses_4cs_lock_stop(tmp_path, monkeypatch):
    """The first score stage records when the lock was committed against 4c's lock stop."""
    path = str(tmp_path / "s.json")
    E.save_stage("lock", {"frozen_objects": E.frozen_digests(E.frozen_names()), "digest": "d"}, path)
    monkeypatch.setattr(E, "OUT", path)
    monkeypatch.setattr(E, "postlock", lambda a, s: E.load_state(path))
    monkeypatch.setattr(E, "lock_commit_time", lambda state, p=None: ("c1", "2026-10-15T20:00:00Z"))

    def stop(*args):
        raise RuntimeError("stop after the lock time")
    monkeypatch.setattr(E, "ScoreContext", stop)
    a = type("A", (), {"stage": "S", "shard": "1/3", "state": path, "rows": str(tmp_path / "rows")})()
    with pytest.raises(RuntimeError, match="stop after the lock time"):
        E.stage_score(a)
    lt = E.load_state(path)["lock_time"]
    assert lt == {"commit": "c1", "utc": "2026-10-15T20:00:00Z", "hard_stop": "2026-10-15T23:59:59Z",
                  "in_time": True}


def test_record_fix_checks_the_frozen_objects_at_its_start_and_end(tmp_path, locked, monkeypatch):
    """Amendment 4b: a frozen object changed while record-fix runs makes the fix out of scope
    and moves no row; a verification that meets a changed object stops and logs it."""
    rows = str(tmp_path / "rows")
    E.write_json(E.row_path("S", "P-READING", 0, rows=rows), _row("S", "P-READING", 0))
    state_path = str(tmp_path / "state.json")
    E.save_stage("lock", _lock_state(locked)["lock"], state_path)

    def tests_that_mutate():
        E.RULE["gain"] = 1.0
        return {"returncode": 0, "tail": ["ok"]}
    monkeypatch.setattr(E, "run_locked_tests", tests_that_mutate)
    monkeypatch.setitem(E.RULE, "gain", E.RULE["gain"])          # undone at the end
    a = type("A", (), {"state": state_path, "rows": rows, "verify": None, "cause": "c",
                       "failed": [], "resolves": []})()
    E.stage_record_fix(a)
    fix = E.load_state(state_path)["fixes"][0]
    assert not fix["in_scope"] and fix["rescore"]["moved"] == []
    assert fix["frozen_problems"]["start"] == [] and \
        "RULE is not the object the lock recorded (amendment 4b)" in fix["frozen_problems"]["end"]
    assert os.path.exists(E.row_path("S", "P-READING", 0, rows=rows))
    # verify: the start check sees the changed object, logs it and stops
    with pytest.raises(SystemExit):
        E.stage_record_fix(type("A", (), {"state": state_path, "rows": rows, "verify": fix["id"]})())
    st = E.load_state(state_path)
    assert E.check_status(st, ["record-fix"]) == {
        ("record-fix", f"frozen objects at the start of verify {fix['id']}"): False}
    assert "verified" not in st["fixes"][0]


def test_the_lock_records_amendment_4c_and_the_frozen_objects_of_this_file():
    """Once a lock exists: its amendment 4c digest is the committed text's, its hard stops
    are 4c's, and its frozen objects are this process's (the locked tests run unchanged)."""
    state = os.path.join(ROOT, "results", "pool_eval.json")
    lk = (json.load(open(state)).get("lock") or {}) if os.path.exists(state) else {}
    if not lk.get("frozen_objects"):
        pytest.skip("no lock with frozen objects in results/pool_eval.json")
    c = E.amendment_section(os.path.join(ROOT, E.AMENDMENT_FILE), E.AMENDMENT_4C_HEADING)
    assert lk["amendment_4c_sha256"] == hashlib.sha256(c.encode()).hexdigest()
    assert lk["hard_stops"] == E.HARD_STOPS and lk["hard_stop"] == E.HARD_STOPS["lock"]
    assert sorted(lk["frozen_objects"]) == E.frozen_names()
    assert E.frozen_problems(lk) == []


# --- amendment 4b: the select, summarise and gate stages, frozen whole, driven on rows ---------
#
# Synthetic S, RS and C rows run through the real stage functions and the real frozen helpers
# (postlock, checks_passed, rs_rows_digest, stored_ship, frozen_check); only the lock check
# itself (harness, UNFROZEN_READS) is replaced, and the run counts are made small (block
# objects: the lock below records them as the test leaves them). Each case must end in a
# refusal or in "archive-3 stays", so that a post-lock edit to the stages or to what they read
# that loses a row, a task, a check or the summary's completeness fails these locked tests.

_STAGE_RUNS = {"S": {"formative": 3, "dense": 2}, "C": {"formative": 4, "dense": 2}}
_GAIN = {"C1": -0.004, "C1m": 0.0, "C2": 0.0}


def _stage_arm(b, cpu, dense):
    n = len(b)
    a = {"b": np.asarray(b, float).tolist(), "ece": np.full((n, 6), 0.05).tolist(), "q0": [0.3] * n,
         "calls": 10, "cpu_sum": cpu * 10, "cpu_max": cpu, "wall_sum": cpu * 10, "wall_max": cpu,
         "failures": 0, "unconverged": 0, "invalid": [],
         "bundle": {"hyper": {"mu0": -2.5, "sigma_mu": 2.5, "T": 5.0}}}
    if dense:
        a["calls_cpu"] = [cpu] * 10
    return a


def _stage_row(stage, R, i, arms, rng, loss0=None):
    """A pool row of A3 and `arms`: X is A3 plus _GAIN[X] (and noise) in the gain regimes and
    A3 - 1e-5 elsewhere; loss0 = (regime, loss) adds the loss to C1 in that regime's run 0."""
    n = int(rng.integers(5, 9))
    meta = [{"family": f"f{j % 6}", "ident": f"n:m{(j + i) % 9}", "benchmark": f"b{(j + 3 * i) % 12}",
             "dated": bool(j % 2), "release": "2025-03-01" if j % 2 else "", "named": True,
             "id_row": {"A3": False, "C1": bool(j % 3)}, "flagged": False, "mcq": j == 0,
             "kind": "binary", "helm": False, "logit": float(rng.normal()), "v": 0.2, "rate": 0.3}
            for j in range(n)]
    base = rng.uniform(0.15, 0.25, (n, 6))
    dense = E.REGIMES[R]["shape"] == "dense"
    out = {E.A3: _stage_arm(base, 0.01, dense)}
    for X in arms:
        x = base + (_GAIN[X] + rng.normal(0, 1e-4, (n, 6)) if R in E.GAIN_REGIMES else -1e-5)
        if loss0 and X == "C1" and (R, i) == (loss0[0], 0):
            x = x + loss0[1]
        out[X] = _stage_arm(x, 0.012, dense)
    return {"stage": stage, "regime": R, "i": i, "meta": meta, "arms": out, "task": {}}


class _NoPool:
    @staticmethod
    def from_files(**kw):
        raise RuntimeError("no pool in this test")


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A locked state over synthetic rows: every pre-lock check passing, the lock (its RS rows
    digest and frozen objects recorded after the patches below), S and RS rows written. Returns
    a namespace with the stage arguments and helpers to write C rows and read the state."""
    import types
    import experiments.regime_sensitivity as RS
    monkeypatch.setattr(E, "N_RUNS", {k: dict(v) for k, v in _STAGE_RUNS.items()})
    monkeypatch.setattr(E, "RS_TASKS", {R: 2 for R in E.RS_REGIMES})
    monkeypatch.setattr(E, "RS_ROWS", str(tmp_path / "rs_rows"))
    monkeypatch.setattr(E, "POOL_BUILD", str(tmp_path / "pool_build.json"))
    monkeypatch.setattr(E, "PoolView", _NoPool)
    problems = []
    monkeypatch.setattr(E, "lock_check", lambda state, a=None: list(problems))
    rng = np.random.default_rng(11)
    rows = str(tmp_path / "rows")
    for R, i in E.plan_tasks("RS"):
        n = int(rng.integers(5, 9))
        b = rng.uniform(0.15, 0.25, (n, 6))
        E.write_json(os.path.join(E.RS_ROWS, R, f"{i}.json"), {"res": {RS.SHIP: {"b": b.tolist()}}})
        meta = [{"parent": f"q{j % 3}", "subject": f"s{j}", "bench": f"b{j % 4}"} for j in range(n)]
        E.write_json(E.row_path("RS", R, i, rows=rows),
                     {"stage": "RS", "regime": R, "i": i, "meta": meta, "task": {},
                      "arms": {X: _stage_arm(b - 1e-5, 0.012, False) for X in E.TIE_ORDER}})
    path = str(tmp_path / "state.json")
    checks = [{"stage": s, "name": n, "ok": True, "values": None, "utc": "1999-12-31T00:00:00Z"}
              for s, ns in E.REQUIRED_CHECKS.items() for n in ns]
    level = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
    lock = {"utc": "2000-01-01T00:00:00Z", "digest": "d" * 64,
            "rs_rows": E.rs_rows_digest(), "frozen_objects": E.frozen_digests(E.frozen_names()),
            "bundles": {X: {"sha256": f"{k}" * 64, "level": level} for k, X in enumerate(E.TIE_ORDER)}}
    E.write_json(path, {"checks": checks, "lock": lock})
    a = types.SimpleNamespace(state=path, rows=rows, fits_record=str(tmp_path / "none.json"), root=None)
    ns = types.SimpleNamespace(a=a, rows=rows, path=path, rng=rng, problems=problems,
                               state=lambda: E.load_state(path))

    def write_stage(stage, arms, loss0=None, skip=()):
        for R, i in E.plan_tasks(stage):
            if (R, i) not in skip:
                E.write_json(E.row_path(stage, R, i, rows=rows), _stage_row(stage, R, i, arms, rng, loss0))
    ns.write = write_stage
    return ns


def _checks_of(st, stage):
    return {n: ok for (s, n), ok in E.check_status(st, [stage]).items()}


def test_the_frozen_stages_carry_select_summarise_and_gate_on_synthetic_rows(staged):
    """The control: every task scored, every check passing: C1 carried, the gate passes."""
    staged.write("S", E.TIE_ORDER)
    E.stage_select(staged.a)
    sel = staged.state()["selection"]
    assert sel["X_star"] == "C1" and sel["outcome"] == "C1 carried to C" and sel["Q"] == ["C1"]
    for X in E.TIE_ORDER:
        assert sel["table"][X]["complete"] and sel["table"][X]["checks_ok"]
        for R in E.REGIMES:                                   # every planned run reached it
            assert sel["stats"][X][R]["runs"] == _STAGE_RUNS["S"][E.REGIMES[R]["shape"]]
    staged.write("C", ("C1",))
    E.stage_summarise(staged.a)
    summ = staged.state()["summary"]
    assert summ["complete"] and summ["missing"] == 0 and summ["X_star"] == "C1"
    for R in E.REGIMES:
        assert summ["stats"][R]["runs"] == _STAGE_RUNS["C"][E.REGIMES[R]["shape"]]
    E.stage_gate(staged.a)
    st = staged.state()
    assert st["gate"]["pass"] and st["gate"]["outcome"] == "pass: X* replaces A3"
    assert all(st["gate"]["holds"].values())
    assert E.read_json(E.POOL_BUILD)["candidate"] == "C1"
    for stage in ("select", "summarise", "gate"):
        assert _checks_of(st, stage) == {"lock check": True, "frozen objects at the end": True} | (
            {"RS rows digest": True} if stage == "select" else {})


@pytest.mark.parametrize("stage,R,i", [("S", "P-TUNED", 1), ("S", "D-FLAT", 0), ("S", "P-READING", 0),
                                       ("RS", "R1B", 1), ("RS", "TUNED", 0)])
def test_select_refuses_a_missing_S_or_RS_task(staged, stage, R, i):
    staged.write("S", E.TIE_ORDER)
    os.remove(E.row_path(stage, R, i, rows=staged.rows))
    with pytest.raises(SystemExit, match="not scored"):
        E.stage_select(staged.a)
    assert "selection" not in staged.state()


def test_select_refuses_a_changed_rs_row(staged):
    staged.write("S", E.TIE_ORDER)
    p = os.path.join(E.RS_ROWS, "FLAT", "1.json")
    row = E.read_json(p)
    row["res"]["ship"]["b"][0][0] += 0.01
    E.write_json(p, row)
    with pytest.raises(SystemExit, match="RS rows changed"):
        E.stage_select(staged.a)
    st = staged.state()
    assert "selection" not in st and _checks_of(st, "select")["RS rows digest"] is False


def test_a_failing_check_gives_archive_3_stays_at_select_and_at_the_gate(staged, tmp_path):
    staged.write("S", E.TIE_ORDER)
    good = E.load_state(staged.path)
    # a task check failed after the lock, and no recorded fix resolves it: (E4) fails for all
    E.log_check("score:S", "task P-FLAT 2", False, "boom", staged.path)
    E.stage_select(staged.a)
    sel = staged.state()["selection"]
    assert sel["X_star"] is None and sel["outcome"] == "no candidate carried: archive-3 stays"
    assert all(not sel["candidates"][X]["holds"]["E4"] for X in E.TIE_ORDER)
    assert "score:S: task P-FLAT 2" in sel["check_problems"]
    with pytest.raises(SystemExit, match="no candidate carried"):
        E.stage_summarise(staged.a)
    with pytest.raises(SystemExit, match="run summarise first"):
        E.stage_gate(staged.a)
    # the same at the gate: a C task's check failed after the lock gives (g) false
    E.write_json(staged.path, good)
    E.stage_select(staged.a)
    staged.write("C", ("C1",))
    E.stage_summarise(staged.a)
    E.log_check("score:C", "task P-AUDIT 1", False, "boom", staged.path)
    E.stage_gate(staged.a)
    gate = staged.state()["gate"]
    assert not gate["holds"]["g"] and not gate["conditions"]["g"]["checks_ok"]
    assert gate["outcome"] == "fail: archive-3 stays" and not os.path.exists(E.POOL_BUILD)
    assert "score:C: task P-AUDIT 1" in gate["check_problems"]
    assert sum(not v for v in gate["holds"].values()) == 1


def test_a_failing_lock_check_refuses_every_post_lock_stage(staged):
    staged.write("S", E.TIE_ORDER)
    staged.problems.append("paiec/hier.py changed since the lock")
    for stage, fn in (("select", E.stage_select), ("summarise", E.stage_summarise),
                      ("gate", E.stage_gate)):
        with pytest.raises(SystemExit, match="lock check failed"):
            fn(staged.a)
        assert _checks_of(staged.state(), stage) == {"lock check": False}
    assert not {"selection", "summary", "gate"} & set(staged.state())


def test_a_missing_c_task_or_an_incomplete_summary_gives_archive_3_stays(staged):
    staged.write("S", E.TIE_ORDER)
    E.stage_select(staged.a)
    selected = E.load_state(staged.path)
    # (1) a planned C task not scored
    staged.write("C", ("C1",), skip=(("P-FLAT", 3),))
    E.stage_summarise(staged.a)
    summ = staged.state()["summary"]
    assert not summ["complete"] and summ["missing"] == 1
    E.stage_gate(staged.a)
    gate = staged.state()["gate"]
    assert not gate["holds"]["g"] and not gate["conditions"]["g"]["complete"]
    assert gate["outcome"] == "fail: archive-3 stays" and sum(not v for v in gate["holds"].values()) == 1
    assert not os.path.exists(E.POOL_BUILD)
    # (2) every task there, but X* not scored on one (no output and no invalid one)
    E.write_json(staged.path, selected)
    staged.write("C", ("C1",))
    p = E.row_path("C", "D-READING", 1, rows=staged.rows)
    row = E.read_json(p)
    row["arms"]["C1"]["b"] = None
    E.write_json(p, row)
    E.stage_summarise(staged.a)
    summ = staged.state()["summary"]
    assert not summ["complete"] and summ["missing"] == 0 and summ["stats"]["D-READING"]["D"] is None
    E.stage_gate(staged.a)
    gate = staged.state()["gate"]
    assert not gate["holds"]["g"] and gate["outcome"] == "fail: archive-3 stays"
    # (3) a summary that records itself incomplete, read by the gate as it is stored
    E.write_json(staged.path, selected)
    staged.write("C", ("C1",))
    E.stage_summarise(staged.a)
    st = staged.state()
    assert st["summary"]["complete"]
    st["summary"]["complete"] = False
    E.write_json(staged.path, st)
    E.stage_gate(staged.a)
    gate = staged.state()["gate"]
    assert not gate["holds"]["g"] and gate["outcome"] == "fail: archive-3 stays"
    assert sum(not v for v in gate["holds"].values()) == 1 and not os.path.exists(E.POOL_BUILD)


def test_a_first_row_that_decides_the_outcome_is_read(staged):
    """Run 0 carries a loss that alone turns the outcome: the stages read it (a stage that
    dropped every regime's first row would carry C1 and pass the gate)."""
    # at select: C1's loss in P-READING's run 0 lifts M(C1) above -0.001
    staged.write("S", E.TIE_ORDER, loss0=("P-READING", 0.05))
    fresh = E.load_state(staged.path)
    E.stage_select(staged.a)
    sel = staged.state()["selection"]
    assert sel["X_star"] is None and sel["outcome"] == "no candidate carried: archive-3 stays"
    assert sel["candidates"]["C1"]["M"] > E.RULE["carry"]
    assert sel["stats"]["C1"]["P-READING"]["runs"] == _STAGE_RUNS["S"]["formative"]
    # at the gate: the same loss in C's run 0 fails (a)
    E.write_json(staged.path, fresh)
    staged.write("S", E.TIE_ORDER)
    E.stage_select(staged.a)
    assert staged.state()["selection"]["X_star"] == "C1"
    staged.write("C", ("C1",), loss0=("P-READING", 0.05))
    E.stage_summarise(staged.a)
    summ = staged.state()["summary"]
    assert summ["complete"] and summ["stats"]["P-READING"]["runs"] == _STAGE_RUNS["C"]["formative"]
    assert summ["stats"]["P-READING"]["D"] > E.RULE["gain"]
    E.stage_gate(staged.a)
    gate = staged.state()["gate"]
    assert not gate["holds"]["a"] and gate["outcome"] == "fail: archive-3 stays"
    assert not os.path.exists(E.POOL_BUILD)
