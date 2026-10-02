"""experiments/pooling_decomposition.py (P1.11) on synthetic data (no download needed).

The 'off' variants: OwnOnly is the predictor on the target pair's own labels and
nothing else (bit for bit, with and without its per-list grouping, malformed
entries dropped as paiec.predict.records drops them, and the identity on a run of
one pair); NoPoolPredictor is the legacy Predictor below WARMUP, and the Predictor
on the pair's own labels everywhere, and differs from it above WARMUP. The
evaluation cap leaves every `labeled` untouched and scores a fixed subset of
targets; without it the checkpoints are level_calibration's. score_run's
identities by construction; the reproduction check; the statistics against direct
arithmetic and ship_confirm.Block. The library default is untouched."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

import experiments.level_calibration as LC
import experiments.ship_confirm as SC
from experiments import pooling_decomposition as PD
from paiec import hier as H
from paiec import predict as P
from paiec.evaluator import BUDGETS
from paiec.hier import HierPredictor, Hyper
from tests.synth import make_pairs

W6 = PD.W6


def bundle(parent):
    return None, Hyper()


def ppred(parent):
    return None, None


@pytest.fixture(scope="module")
def run():
    # two benchmarks, four subjects: at B31 a benchmark holds 124 labels, past WARMUP
    return make_pairs(n_subjects=4, n_benchmarks=2, n_items=90, seed=3)


@pytest.fixture(scope="module")
def cps(run):
    slots, cps, before = PD.checkpoints(run, "pair")
    return slots, cps


@pytest.fixture(scope="module")
def scored(run):
    return PD.score_run(run, "pair", None, PD.NAMES, bundle, ppred)


def own_list(labeled, inp):
    s, it = inp
    return [e for e in labeled if e[0][0] == s and e[0][1]["benchmark_id"] == it["benchmark_id"]]


# --- the design --------------------------------------------------------------------------

def test_level_is_the_shipped_one():
    assert PD.model_level() == PD.LEVEL == {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}


def test_library_default_unchanged():
    assert H.Flags().pool_mu is True
    assert P.BenchmarkFit.WARMUP == 64
    assert PD.NoPoolFit.WARMUP == PD.NOPOOL_WARMUP > 10 ** 6


def test_tasks_and_paths(tmp_path):
    ts = PD.tasks()
    assert len(ts) == 2 * PD.N_FORMATIVE + 2 * len(PD.DENSE)
    assert len({PD.row_path(t, str(tmp_path)) for t in ts}) == len(ts)
    assert ts[0] == ("formative", 0, "pair") and ts[1] == ("formative", 0, "benchmark")
    assert not set(PD.DENSE) & set(PD.SINGLE_SUBJECT)
    assert [c[:2] for c in PD.POOLING] == [("ship", "ship_own"), ("ship", "ship_nolevel"),
                                          ("ship_nolevel", "ship_own"), ("legacy", "legacy_nopool")]
    assert set(PD.REPRO["configs"]) <= set(PD.NAMES)


# --- checkpoints and the evaluation cap ---------------------------------------------------

@pytest.mark.parametrize("scope", ["pair", "benchmark"])
def test_checkpoints_without_cap_are_level_calibrations(run, scope):
    _, mine, before = PD.checkpoints(run, scope)
    _, theirs = LC.checkpoints(run, scope)
    assert len(mine) == len(theirs) == len(BUDGETS)
    for (b1, l1, i1, x1), (b2, l2, i2, x2) in zip(mine, theirs):
        assert b1 == b2 and l1 == l2 and i1 == i2 and np.array_equal(x1, x2)
    assert all(n > 0 for n in before)


def test_eval_cap_keeps_labeled_and_scores_a_fixed_subset(run):
    s0, full, before0 = PD.checkpoints(run, "pair")
    s1, capped, before1 = PD.checkpoints(run, "pair", cap=10)
    s2, again, _ = PD.checkpoints(run, "pair", cap=10)
    assert before0 == before1
    for (b, l0, i0, _), (_, l1, i1, x1), (_, l2, i2, _) in zip(full, capped, again):
        assert l0 == l1 == l2                       # acquisition untouched
        assert i1 == i2                             # deterministic
        assert all(inp in i0 for inp in i1)         # a subset of the full run's targets
        assert len(x1) == sum(len(s.targets) for s in s1)
    for a, s in zip(s0, s1):
        keys = {t[2] for t in s.targets}
        assert len(keys) == min(10, len({t[2] for t in a.targets})) == s.n_items
        # every response of a kept input is kept
        assert sorted(t[1] for t in a.targets if t[2] in keys) == sorted(t[1] for t in s.targets)


# --- the 'off' variants -------------------------------------------------------------------

def test_own_only_is_the_fit_on_own_labels(cps):
    slots, cp = cps
    h = HierPredictor(None, replace(Hyper(), **PD.LEVEL))
    wrapped = PD.OwnOnly(h.predict)
    b, labeled, inputs, _ = cp[-1]
    assert len({PD.own_key(e) for e in labeled}) > 2
    for inp in inputs[::17]:
        mine = own_list(labeled, inp)
        assert 0 < len(mine) <= 31 < len(labeled)
        want = HierPredictor(None, replace(Hyper(), **PD.LEVEL)).predict(inp, mine)
        assert wrapped(inp, labeled) == want                      # grouped once, then memo
        assert PD.OwnOnly(h.predict)(inp, list(labeled)) == want  # a fresh list, regrouped
    # the full list says more than the own labels
    full = [h.predict(inp, labeled) for inp in inputs[::17]]
    own = [wrapped(inp, labeled) for inp in inputs[::17]]
    assert max(abs(a - c) for a, c in zip(full, own)) > 1e-4


def test_own_only_is_the_identity_on_one_pair(run):
    _, cp, _ = PD.checkpoints(run[:1], "pair")
    h = HierPredictor(None, replace(Hyper(), **PD.LEVEL))
    w = PD.OwnOnly(HierPredictor(None, replace(Hyper(), **PD.LEVEL)).predict)
    for b, labeled, inputs, _ in cp:
        for inp in inputs[::9]:
            assert w(inp, labeled) == h.predict(inp, labeled)


def test_own_only_drops_what_records_drops(cps):
    slots, cp = cps
    _, labeled, inputs, _ = cp[3]
    seen = []
    w = PD.OwnOnly(lambda inp, lab: seen.append(lab) or 0.5)
    junk = [None, "x", [["a", "b"], 1], [[{}, {}]], 7]
    w(inputs[0], list(labeled) + junk)
    assert seen[-1] == own_list(labeled, inputs[0])
    w(inputs[0], None)
    assert seen[-1] == []


def test_nopool_predictor(cps):
    slots, cp = cps
    lo = cp[2]                  # B3: 24 labels, below WARMUP on either benchmark
    hi = cp[-1]                 # B31: 124 labels a benchmark
    for b, labeled, inputs, _ in (lo, hi):
        per_b = {}
        for (s, it), _ in labeled:
            per_b.setdefault(it["benchmark_id"], set()).add(P.item_key(it))
        reached = max(len(v) for v in per_b.values()) >= P.BenchmarkFit.WARMUP
        assert reached == (b == 31)
        pool, nopool = P.Predictor(None, None), PD.NoPoolPredictor(None, None)
        a = np.array([pool.predict(inp, labeled) for inp in inputs])
        c = np.array([nopool.predict(inp, labeled) for inp in inputs])
        own = np.array([P.Predictor(None, None).predict(inp, own_list(labeled, inp))
                        for inp in inputs[::11]])
        assert np.array_equal(c[::11], own)       # no pooling: the pair's own labels
        if reached:
            assert np.max(np.abs(a - c)) > 1e-3   # the switch acts
        else:
            assert np.array_equal(a, c)


# --- scoring ------------------------------------------------------------------------------

def test_score_run_identities(scored):
    meta, res = scored
    assert set(res) == set(PD.NAMES)
    B = {n: np.asarray(res[n]["b"], float) for n in PD.NAMES}
    n = len(meta)
    assert all(B[k].shape == (n, len(BUDGETS)) for k in B)
    for k in ("ship_nolevel", "ship_own"):
        assert np.array_equal(B[k][:, 0], B["ship"][:, 0])
    li = np.array([m["labeled_items"] for m in meta])
    below = li < P.BenchmarkFit.WARMUP
    assert below.any() and (~below).any()
    assert np.array_equal(B["legacy"][below], B["legacy_nopool"][below])
    assert all(res[k]["failures"] == 0 for k in PD.NAMES)
    assert all(m["companions"] == 4 for m in meta)
    assert all(0 <= m["cover31"] <= 1 for m in meta)
    assert all(m["eval"] == m["eval_all"] for m in meta)
    # under per-pair splits other subjects' labels land on the target's items
    assert np.mean([m["cover31"] for m in meta]) > 0.3
    rows = {("formative", 0, "pair"): {"meta": meta, "res": res}}
    ident = PD.identities(rows)
    assert ident["hier_variants_at_B0_max_abs"] == 0
    assert ident["legacy_vs_nopool_below_warmup_max_abs"] == 0
    assert ident["pair_budget_cells_at_or_above_warmup"] > 0


def test_benchmark_scope_hides_other_subjects_labels(run):
    slots, cp, before = PD.checkpoints(run, "benchmark")
    meta = PD.describe(run, slots, before)
    assert max(m["cover31"] for m in meta) == 0


def test_score_run_is_one_pair_identity(run):
    meta, res = PD.score_run(run[:1], "pair", None, ("ship", "ship_own", "ship_nolevel"), bundle, ppred)
    a = np.asarray(res["ship"]["b"])
    assert np.array_equal(a, np.asarray(res["ship_own"]["b"]))
    assert np.allclose(a, np.asarray(res["ship_nolevel"]["b"]), atol=1e-9)


# --- the reproduction check ---------------------------------------------------------------

def test_repro_check(tmp_path, scored):
    meta, res = scored
    st = {"meta": meta, "lib": PD.lib_digests(),
          "res": {c: {"b": res[v]["b"]} for v, c in PD.REPRO["configs"].items()}}
    (tmp_path / "0.json").write_text(json.dumps(st))
    t = ("formative", 0, "pair")
    rep = PD.repro_check(t, meta, res, str(tmp_path))
    assert rep["checked"] and rep["ok"] and rep["same_lib"] and rep["pairs_match"]
    bad = json.loads(json.dumps(st))
    bad["res"]["ship"]["b"][0][3] += 1e-9
    (tmp_path / "0.json").write_text(json.dumps(bad))
    assert PD.repro_check(t, meta, res, str(tmp_path))["ok"] is False
    assert PD.repro_check(("formative", 0, "benchmark"), meta, res, str(tmp_path)) is None
    assert PD.repro_check(("formative", 60, "pair"), meta, res, str(tmp_path)) is None
    assert PD.repro_check(("formative", 1, "pair"), meta, res, str(tmp_path))["checked"] is False


# --- statistics ---------------------------------------------------------------------------

def fake_rows(n_runs=12, seed=0):
    """Rows of both scopes over the same pairs, random Brier, with a known structure:
    ship_own = ship + 0.01 at B31 on pairs with a companion."""
    rng = np.random.default_rng(seed)
    parents = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
    rows = {}
    for i in range(n_runs):
        k = int(rng.integers(3, 7))
        benches = [parents[j] for j in rng.integers(0, 5, k)]
        subs = [f"s{int(x)}" for x in rng.integers(0, 6, k)]
        seen, meta = set(), []
        for b, s in zip(benches, subs):
            if (b, s) in seen:
                continue
            seen.add((b, s))
            meta.append({"bench": b, "parent": b, "subject": s})
        for m in meta:
            m["companions"] = sum(x["bench"] == m["bench"] for x in meta)
            m["cover31"] = 0.5
            m["eval"] = m["eval_all"] = 40
            m["labeled_items"] = [0, 1, 3, 7, 15, 31]
        for sc in PD.SCOPES:
            base = rng.uniform(0.1, 0.3, (len(meta), 6))
            res = {}
            for n in PD.NAMES:
                b = base + rng.normal(0, 0.01, base.shape)
                res[n] = {"b": b.tolist(), "ece": (b / 2).tolist(), "q0": [0.4] * len(meta),
                          "calls": 10, "mean_s": 0.001, "max_s": 0.01, "failures": 0,
                          "unconverged": 0, "secs": 1.0}
            res["ship_own"]["b"] = (np.asarray(res["ship"]["b"])
                                    + np.array([[0, 0, 0, 0, 0, 0.01 * (m["companions"] > 1)]
                                                for m in meta])).tolist()
            rows[("formative", i, sc)] = {"meta": meta, "res": res}
    return rows


def test_formative_statistics():
    rows = fake_rows()
    f = PD.Formative(rows, "pair")
    rep = f.compare("ship", "ship_own")
    per_run = [float(np.mean((x - y) @ W6)) for x, y in zip(f.B["ship"], f.B["ship_own"])]
    assert rep["diff"]["mean"] == pytest.approx(np.mean(per_run), abs=1e-6)
    assert rep["diff"]["run_se"] == pytest.approx(np.std(per_run, ddof=1) / math.sqrt(len(per_run)),
                                                  abs=1e-6)
    # the same as ship_confirm.Block directly
    blk = SC.Block(f.runs_of(f.B["ship"]), f.ids)
    ref, _ = blk.paired(PD.Formative.arm(f.ids, f.B["ship_own"], "x"))
    assert rep["diff"] == ref["diff"] and rep["per_parent"] == ref["per_parent"]
    assert rep["budgets"][5]["diff"] < 0 and rep["budgets"][0]["diff"] == 0
    bc = rep["by_companions"]
    assert bc["alone"]["mean"] == pytest.approx(0, abs=1e-12)
    assert bc["two"]["mean"] == pytest.approx(-0.01 * W6[5], abs=1e-6)
    v = f.variant("ship")
    assert v["runs"] == 12 and v["failures"] == 0
    assert f.describe()["appearances"] == sum(len(rows[("formative", i, "pair")]["meta"]) for i in range(12))


def test_platform_mix_counts_the_real_runs():
    mix = PD.platform_mix()
    assert mix["pairs"] == 26 and [r["pairs"] for r in mix["runs"]] == [9, 8, 9]
    assert all(r["benchmarks"] == 7 for r in mix["runs"])
    assert mix["counts"] == {"alone": 16, "two": 10, "three or more": 0}
    assert sum(mix["shares"].values()) == pytest.approx(1, abs=1e-5)


def test_platform_mix_from_files(tmp_path):
    runs = {"a": [{"benchmark": "x"}, {"benchmark": "x"}, {"benchmark": "y"},
                  {"benchmark": "z"}, {"benchmark": "z"}, {"benchmark": "z"}]}
    (tmp_path / "f.json").write_text(json.dumps({"r": runs}))
    mix = PD.platform_mix((("f.json", ("r", "a")),), str(tmp_path))
    assert mix["counts"] == {"alone": 1, "two": 2, "three or more": 3}
    assert mix["runs"][0]["counts"] == mix["counts"]
    with pytest.raises(ValueError):
        PD.companion_class(0)


def test_platform_mix_reweights_the_classes():
    rows = fake_rows(n_runs=30, seed=6)
    f = PD.Formative(rows, "pair")
    bc = f.compare("ship", "ship_own")["by_companions"]
    shares = {"alone": 0.6, "two": 0.4, "three or more": 0.0}
    rep = f.compare("ship", "ship_own", shares)
    mx = rep["platform_mix"]
    # ship_own = ship + 0.01 at B31 with a companion: the mix is 0.4 x the 'two' class
    assert mx["mean"] == pytest.approx(0.6 * bc["alone"]["mean"] + 0.4 * bc["two"]["mean"], abs=2e-6)
    assert mx["mean"] == pytest.approx(-0.4 * 0.01 * W6[5], abs=1e-6)
    assert mx["budgets"][5]["mean"] == pytest.approx(-0.4 * 0.01, abs=1e-6)
    assert all(b["mean"] == pytest.approx(0, abs=1e-12) for b in mx["budgets"][:5])
    assert mx["cluster_se"] is not None and mx["strat_se"] is not None
    # all weight on one class: its mean, and a cluster SE near by_companions' (same resamples)
    one = f.compare("ship", "ship_own", {"two": 1.0})["platform_mix"]
    assert one["mean"] == pytest.approx(bc["two"]["mean"], abs=1e-6)
    assert one["cluster_se"] == pytest.approx(bc["two"]["cluster_se"], rel=1e-3, abs=1e-6)
    # a class with weight but no appearance: no estimate
    rows1 = {k: v for k, v in rows.items()}
    f1 = PD.Formative(rows1, "pair")
    f1.comp = np.minimum(f1.comp, 2)
    assert f1.compare("ship", "ship_own", {"three or more": 1.0})["platform_mix"] is None
    assert "platform_mix" not in f.compare("ship", "ship_own")


def test_without_single_subject_drops_its_appearances_only():
    rows = fake_rows(n_runs=30, seed=6)
    f = PD.Formative(rows, "pair")
    shares = {"alone": 0.6, "two": 0.4, "three or more": 0.0}
    rep = f.compare("ship", "legacy", shares)
    ws = rep["without_single_subject"]
    assert ws["excluded"] == list(PD.SINGLE_SUBJECT)
    # by hand: the appearance-weighted (1 / run size) mean of the kept appearances of a class
    x = np.concatenate([(a - b) @ W6 for a, b in zip(f.B["ship"], f.B["legacy"])])
    w = np.concatenate([np.full(len(m), 1.0 / len(m)) for m in f.meta])
    par = np.array([m["parent"] for mm in f.meta for m in mm])
    for label, lo, hi in PD.COMPANIONS:
        sel = (f.comp >= lo) & (f.comp <= hi) & ~np.isin(par, PD.SINGLE_SUBJECT)
        if sel.any():
            assert ws["by_companions"][label]["mean"] == pytest.approx(
                np.sum(w[sel] * x[sel]) / np.sum(w[sel]), abs=1e-6)
            assert ws["by_companions"][label]["appearances"] == int(sel.sum())
    assert ws["platform_mix"]["mean"] == pytest.approx(
        0.6 * ws["by_companions"]["alone"]["mean"] + 0.4 * ws["by_companions"]["two"]["mean"], abs=2e-6)
    # an all-true mask is the plain reading
    blk = SC.Block(f.runs_of(f.B["ship"]), f.ids)
    _, d = blk.paired(PD.Formative.arm(f.ids, f.B["legacy"], "x"))
    assert f.by_companions(blk, d, keep=np.ones(len(f.comp), bool)) == rep["by_companions"]
    # the describe counts and the summary reading
    dsc = f.describe()
    assert sum(dsc["single_subject_companions"].values()) == int(np.isin(par, PD.SINGLE_SUBJECT).sum())
    form = {"pair": {"describe": dsc, "comparisons": {f"{a} - {b}": f.compare(a, b, shares)
                                                      for a, b, _ in PD.COMPARISONS}}}
    ss = PD.single_subject_reading(form)
    app = ss["appearances"]["pair"]
    assert app["alone"] == dsc["companions"]["alone"]
    assert app["alone_single_subject"] == dsc["single_subject_companions"]["alone"]
    v = ss["comparisons"]["ship - legacy"]["pair"]
    assert v["alone_without"] == ws["by_companions"].get("alone")
    assert v["platform_mix_without"]["mean"] == ws["platform_mix"]["mean"]
    # no single-subject appearance: no such key
    rows2 = {k: v_ for k, v_ in rows.items()}
    f2 = PD.Formative(rows2, "pair")
    f2.multi = np.ones(len(f2.comp), bool)
    assert "without_single_subject" not in f2.compare("ship", "legacy", shares)


def test_reading_table():
    grid = {"a - b": {"what": "w",
                      "pair": {"formative": {"diff": {"mean": -0.002, "run_se": 1e-4, "cluster_se": 2e-4,
                                                      "strat_se": 1e-4},
                                             "parent_level": {"mean": -0.003, "se": 0.001},
                                             "platform_mix": {"mean": -0.001, "cluster_se": 3e-4,
                                                              "strat_se": 2e-4, "budgets": []}},
                               "dense": {"mean": -0.03, "se": 0.01, "n": 4}},
                      "benchmark": {"formative": {}, "dense": {"mean": -0.015, "se": 0.006, "n": 4}}}}
    t = PD.reading_table(grid)["a - b"]
    assert t["dense pair"] == {"mean": -0.03, "se": 0.01, "n": 4}
    assert t["formative pair"]["parent_level_mean"] == -0.003 and t["formative pair"]["cluster_se"] == 2e-4
    assert t["platform mix pair"] == {"mean": -0.001, "cluster_se": 3e-4, "strat_se": 2e-4}
    assert t["ratio_formative_parent_level_over_dense pair"] == pytest.approx(0.1)
    assert t["ratio_dense_benchmark_over_pair"] == pytest.approx(0.5)
    assert t["formative benchmark"] is None and t["platform mix benchmark"] is None


def test_recheck_against_stored_rows(tmp_path, run, monkeypatch):
    score_run = PD.score_run
    monkeypatch.setattr(PD, "draw", lambda t: run[:2])
    monkeypatch.setattr(PD, "score_run",
                        lambda r, sc, cap=None, names=PD.NAMES: score_run(r, sc, cap, names, bundle, ppred))
    t = ("formative", 0, "pair")
    meta, res = score_run(run[:2], "pair", None, PD.NAMES, bundle, ppred)
    PD.write_json(PD.row_path(t, str(tmp_path)), {"meta": meta, "res": res, "script": "old"})
    rec = PD.stage_recheck(str(tmp_path), (t,))
    assert rec["all_bit_identical"] and rec["tasks"][0]["row_script"] == "old"
    got = PD.load_recheck(str(tmp_path))
    assert got["checked"] and got["all_bit_identical"] and got["script_is_this_one"]
    res["ship"]["b"][0][2] += 1e-12
    PD.write_json(PD.row_path(t, str(tmp_path)), {"meta": meta, "res": res, "script": "old"})
    assert not PD.stage_recheck(str(tmp_path), (t,))["all_bit_identical"]
    assert PD.load_recheck(str(tmp_path / "none"))["checked"] is False


def test_scope_effect_is_the_difference_of_differences():
    rows = fake_rows(seed=1)
    fp, fb = PD.Formative(rows, "pair"), PD.Formative(rows, "benchmark")
    eff = PD.scope_effect(fp, fb, "legacy", "smooth")
    dp = fp.compare("legacy", "smooth")["diff"]["mean"]
    db = fb.compare("legacy", "smooth")["diff"]["mean"]
    assert eff["diff"]["mean"] == pytest.approx(dp - db, abs=2e-6)
    assert eff["pair_scope"] == pytest.approx(dp, abs=1e-6)


def test_dense_statistics():
    rng = np.random.default_rng(4)
    meta = [{"bench": "matharena", "parent": "matharena", "subject": f"s{j}", "eval": 48,
             "eval_all": 100, "resp": 96, "p": 0.6, "companions": 9, "cover31": 0.9,
             "labeled_items": [0, 9, 27, 63, 120, 200]} for j in range(9)]
    res = {n: {"b": rng.uniform(0.1, 0.3, (9, 6)).tolist(), "ece": np.zeros((9, 6)).tolist(),
               "q0": [0.5] * 9, "failures": 0, "unconverged": 0, "mean_s": 0.01, "max_s": 0.1,
               "secs": 3.0} for n in PD.NAMES}
    D = PD.Dense({"meta": meta, "res": res, "cap": 48})
    out, d = D.diff("legacy", "legacy_nopool")
    a = (np.asarray(res["legacy"]["b"]) - np.asarray(res["legacy_nopool"]["b"])) @ W6
    assert out["ALC"]["mean"] == pytest.approx(a.mean(), abs=1e-6)
    assert out["ALC"]["se"] == pytest.approx(a.std(ddof=1) / 3, abs=1e-6)
    assert sum(b["alc_part"] for b in out["budgets"]) == pytest.approx(a.mean(), abs=1e-5)
    assert D.describe()["eval_items_scored"] == 48 and D.describe()["cap"] == 48
    acr = PD.across([0.01, 0.03, None, 0.02])
    assert acr["mean"] == pytest.approx(0.02) and acr["n"] == 3
    assert acr["se"] == pytest.approx(0.01 / math.sqrt(3), abs=1e-6)
    assert PD.across([None]) is None


def test_summarise_on_fake_rows(tmp_path, monkeypatch):
    rows = fake_rows(n_runs=6, seed=2)
    for t, r in rows.items():
        r.update({"task": list(t), "repro": None, "task_s": 1.0, "rss_gb": 0.5,
                  "lib": PD.lib_digests(), "script": PD.SCRIPT_SHA, "utc": "2026-10-02T00:00:00Z"})
        PD.write_json(PD.row_path(t, str(tmp_path)), r)
    monkeypatch.setattr(PD, "N_FORMATIVE", 6)
    out = PD.stage_summarise(str(tmp_path), str(tmp_path / "out.json"))
    assert not out["complete"] and len(out["missing"]) == 2 * len(PD.DENSE)
    assert set(out["formative"]) == set(PD.SCOPES)
    g = out["decomposition"]["ship - ship_own"]
    assert g["pair"]["formative"]["diff"]["mean"] < 0
    assert g["pair"]["dense"] is None
    json.loads((tmp_path / "out.json").read_text())       # strict JSON, no NaN


def test_dense_crosscheck(tmp_path):
    rng = np.random.default_rng(5)
    b = rng.uniform(0.1, 0.3, (4, 6))
    res = {n: {"b": b.tolist()} for n in PD.NAMES}
    rows = {("dense", "real_webagents", "pair"): {"cap": None, "res": res},
            ("dense", "matharena", "pair"): {"cap": 32, "res": res}}
    stored = [list(np.round(x, 6)) + [0.2, 0.1] for x in b]
    raw = {f"r2|real_webagents|pair|{name}": {"rows": stored} for name in PD.HE_DENSE.values()}
    raw["r2|matharena|pair|Predictor"] = {"rows": stored}
    path = tmp_path / "he.json"
    path.write_text(json.dumps({"raw": raw}))
    out = PD.dense_crosscheck(rows, str(path))
    assert out["checked"] and set(out["cells"]) == {"real_webagents pair legacy", "real_webagents pair smooth"}
    assert all(c["within_tol"] and c["pairs"] == 4 for c in out["cells"].values())
    raw["r2|real_webagents|pair|Predictor"]["rows"][0][2] += 1e-3
    path.write_text(json.dumps({"raw": raw}))
    assert not PD.dense_crosscheck(rows, str(path))["cells"]["real_webagents pair legacy"]["within_tol"]
    assert PD.dense_crosscheck(rows, str(tmp_path / "missing.json"))["checked"] is False
