"""experiments/ship_confirm.py's arithmetic and run matching on synthetic rows
(no data needed): its paired statistics are level_calibration.compare's, the
parent-level numbers are the harness's, blocks and gaps are what is stored,
and a comparator run is used only when it matches the shipped rows pair for
pair."""
import math

import numpy as np
import pytest

import experiments.level_calibration as LC
from experiments import ship_confirm as SC

W6 = np.asarray(SC.WEIGHTS, float)


def meta(parent, subject, n=50, p=0.3, tag="g0"):
    return {"bench": f"{parent}::{tag}", "parent": parent, "subject": subject, "eval": n,
            "p": p, "logit": 0.0, "harness": "", "effort": None, "days": 100}


def synthetic(runs=12, seed=0):
    """Shipped rows and a comparator's on runs of 3 to 6 pairs over the four
    multi-subject parents and one single-subject one."""
    rng = np.random.default_rng(seed)
    parents = list(SC.PARENTS) + ["swe_rebench"]
    ship, other = {}, SC.Arm("synthetic")
    for i in range(runs):
        k = int(rng.integers(3, 7))
        ms = []
        for j in range(k):
            q = parents[int(rng.integers(0, len(parents)))]
            s = "only" if q == "swe_rebench" else f"s{int(rng.integers(0, 4))}"
            ms.append(meta(q, s, p=round(float(rng.random()), 5), tag=f"g{j}"))
        b = rng.uniform(0.05, 0.3, (k, 6))
        ob = np.clip(b + rng.normal(0.01, 0.03, (k, 6)), 0, 1)
        ship[i] = (ms, b)
        other.put(i, ob @ W6, ob, "synthetic")
    return ship, other


def test_paired_statistics_are_level_calibrations():
    ship, other = synthetic()
    ids = sorted(ship)
    got, _ = SC.Block(ship, ids).paired(other)
    # level_calibration.compare on the same rows, stored the way compact() writes them
    raw, metas = {}, {}
    for i in ids:
        ms, b = ship[i]
        metas[("tl", i)] = ms
        raw[f"tl|{i}|base"] = {"res": {
            LC.SMOOTHED: {"alc": list(b @ W6), "b": list(b.mean(0)), "ece0": [0, 0], "ece_alc": 0, "q0": 0},
            LC.PRED: {"alc": list(other.runs[i][0]), "b": list(other.runs[i][1].mean(0)),
                      "ece0": [0, 0], "ece_alc": 0, "q0": 0}}, "timing": {}}
    ref = LC.compare(raw, metas, "tl", ids, [LC.SMOOTHED])[LC.SMOOTHED]
    for k in ("mean", "run_se", "cluster_se", "strat_se"):
        assert got["diff"][k] == pytest.approx(ref["diff"][k], abs=1e-6)
    for q, v in ref["diff"]["lopo"].items():
        assert got["per_parent"][q]["leave_out"] == pytest.approx(v, abs=1e-6)
    assert got["shipped_ALC"] == pytest.approx(ref["ALC"][0], abs=1e-6)
    for k, row in enumerate(got["budgets"]):
        assert row["diff"] == pytest.approx(ref["diff_budgets"][k], abs=1e-6)


def test_per_parent_and_parent_level_by_hand():
    ship, other = synthetic(seed=3)
    ids = sorted(ship)
    got, d = SC.Block(ship, ids).paired(other)
    x, w, par = [], [], []
    for i in ids:
        ms, b = ship[i]
        dd = b @ W6 - other.runs[i][0]
        x += list(dd)
        w += [1 / len(ms)] * len(ms)
        par += [m["parent"] for m in ms]
    x, w, par = np.array(x), np.array(w), np.array(par)
    means = {}
    for q in set(par):
        m = par == q
        means[q] = float(np.sum(w[m] * x[m]) / np.sum(w[m]))
        assert got["per_parent"][q]["mean"] == pytest.approx(means[q], abs=1e-6)
    multi = [means[q] for q in SC.PARENTS if q in means]
    lvl = got["parent_level"]
    assert lvl["mean"] == pytest.approx(np.mean(multi), abs=1e-6)
    assert lvl["se"] == pytest.approx(np.std(multi, ddof=1) / math.sqrt(len(multi)), abs=1e-6)
    assert lvl["range"] == [pytest.approx(min(multi), abs=1e-6), pytest.approx(max(multi), abs=1e-6)]
    # one subject is one cluster: no cluster SE for it
    assert got["per_parent"]["swe_rebench"]["clusters"] == 1
    assert got["per_parent"]["swe_rebench"]["cluster_se"] is None
    assert got["diff"]["mean"] == pytest.approx(np.mean([np.mean(v) for v in d]), abs=1e-6)


def test_single_run_sd_is_over_run_means():
    ship, _ = synthetic(seed=5)
    ids = sorted(ship)
    s = SC.Block(ship, ids).ship_summary()
    run_b = np.array([ship[i][1].mean(0) for i in ids])
    run_alc = np.array([np.mean(ship[i][1] @ W6) for i in ids])
    assert s["single_run_sd"]["ALC"] == pytest.approx(run_alc.std(ddof=1), abs=1e-6)
    assert s["single_run_sd"]["budgets"] == [pytest.approx(v, abs=1e-6) for v in run_b.std(0, ddof=1)]


def test_shown_compares_at_the_printed_precision():
    c = SC.Checks()
    assert c.shown("a", -0.043133, "-0.0431", "t")
    assert c.shown("b", 0.0019309, "0.0019", "t")
    assert c.shown("c", 0.843, "+0.84", "t")
    assert not c.shown("d", -0.00149, "-0.0016", "t")
    assert not c.shown("e", 0.16086, "0.1608", "t")
    assert c.shown("f", -0.00004, "0.0000", "t")
    assert [x["ok"] for x in c.items] == [True, True, True, False, False, True]
    assert len(c.failed()) == 2


def test_blocks_cover_only_what_is_stored(monkeypatch):
    monkeypatch.setattr(SC, "PLAN", {"tl": 300, "r1b": 150})
    a = SC.Arm("x")
    for i in range(200):
        a.put(i, [0.1], np.zeros((1, 6)), "x")
    labels = [SC.rng_label(b) for b in SC.blocks_of("tl", a)]
    assert labels == ["0-99", "100-199", "0-199"]
    assert SC.missing("tl", a) == ["200-299"]
    assert [SC.rng_label(b) for b in SC.blocks_of("r1b")] == ["0-99", "100-149", "0-149"]
    assert SC.missing("r1b", None) == ["0-99", "100-149"]


def _world():
    """One run per regime, a testlike_check row set and hier_eval rows that match it."""
    ms = [meta("matharena", "s0", p=0.25), meta("real_webagents", "s1", p=0.5)]
    b = np.full((2, 6), 0.2)
    ship = {r: {0: (ms, b)} for r in ("tl", "tl mix/whole", "tl no shift", "r1b", "r1p")}
    pred = [[0.21] * 6 + [0.21], [0.22] * 6 + [0.22]]
    rows = [{"pseudo": m["bench"], "parent": m["parent"], "subject": m["subject"], "eval": m["eval"],
             "p": m["p"], "pred": pr, "emp": [0.25] * 7} for m, pr in zip(ms, pred)]
    tc = {"check|default|0": {"rows": rows}, "r1|public R1|0": {"rows": rows}}
    he = {"r1|benchmark|pair|0|0|Predictor": {"rows": [r + [0.0] for r in pred]},
          "r1|benchmark|pair|0|0|smoothed Beta(2,2)": {"rows": [[0.23] * 7 + [0.0]] * 2},
          "r1|pair|pair|0|0|Predictor": {"rows": [r + [0.0] for r in pred]}}
    return ship, tc, he


def test_runs_are_used_only_when_matched(monkeypatch):
    monkeypatch.setattr(SC, "PLAN", {r: 1 for r in ("tl", "tl mix/whole", "tl no shift", "r1b", "r1p")})
    ship, tc, he = _world()
    arms = SC.build_arms(ship, tc, he, {})
    assert arms["tl"]["legacy"].runs[0][0].tolist() == [0.21, 0.22]
    assert arms["tl"]["smoothed"] is None                 # level_calibration holds no run
    assert arms["r1b"]["legacy"].matched_by[0].startswith("hier_eval Predictor = testlike_check")
    assert arms["r1b"]["smoothed"].runs[0][0].tolist() == [0.23, 0.23]
    # hier_eval rows carry no ids: without level_calibration's run there is no match
    assert arms["r1p"]["legacy"] is None and arms["r1p"]["smoothed"] is None
    assert arms["tl mix/whole"]["legacy"] is None


def test_a_disagreeing_stored_run_stops(monkeypatch):
    monkeypatch.setattr(SC, "PLAN", {r: 1 for r in ("tl", "tl mix/whole", "tl no shift", "r1b", "r1p")})
    ship, tc, he = _world()
    tc["check|default|0"]["rows"][1] = {**tc["check|default|0"]["rows"][1], "subject": "other"}
    with pytest.raises(SC.Mismatch):
        SC.build_arms(ship, tc, he, {})
    ship, tc, he = _world()
    he["r1|benchmark|pair|0|0|Predictor"]["rows"][0][6] += 1e-6
    with pytest.raises(SC.Mismatch):
        SC.build_arms(ship, tc, he, {})
