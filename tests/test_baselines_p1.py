"""experiments/baselines_p1.py (P1.9) on synthetic data (no download needed).

The plan is P1a's runs and smcal; the plain 1PL differs from P1a's onepl by the
documented switches and nothing else, and at budget 0 it is the closed-form
1PL marginal (no floor, no slip, the group variance folded into the item's);
its labels reach neither another benchmark nor another group; its fitted level
is fit_hyper's with the attribute shift left out. The statistics are P1a's
RegimeRows on merged rows, the decomposition's steps telescope to their total
and shares are given only where the total stands clear of 0. The BLE record is
read from the baseline repository without importing it."""
import copy
import json
import math

import numpy as np
import pytest

from experiments import baselines_p1 as P
from experiments import regime_sensitivity as RS
from paiec import hier as H
from paiec import prior as PR
from paiec.hier import HierPredictor, Hyper
from tests.synth import make_pairs

W6 = np.asarray(P.WEIGHTS, float)
#: E over a standard normal on a grid far finer than hier's
ZF = np.linspace(-12, 12, 48_001)
WF = np.exp(-ZF ** 2 / 2) / np.exp(-ZF ** 2 / 2).sum()
HY = Hyper(mu0=-0.7, sigma_mu=1.1, sigma_theta=0.2, sigma_delta=0.9, sigma_attr=0.6,
           sigma_d=1.3, sigma_g=0.8)
MCQ = "Which is prime?\n(A) 4\n(B) 6\n(C) 7\n(D) 9"


def subject(name, **kw):
    s = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date", "harness",
                       "harness_version", "reasoning_effort", "subject_features_extra"), "")
    s.update(normalized_name=name, **kw)
    return s


def item(j, bid, features="", text=None):
    return {"item_content": text or f"question {j} of {bid}", "item_features": features,
            "interactors": "", "benchmark_id": bid}


def expect_sigmoid(m, v):
    return float(np.sum(WF / (1 + np.exp(-(m + math.sqrt(v) * ZF)))))


# --- the plan --------------------------------------------------------------------------------

def test_the_plan_is_p1as_runs():
    assert P.REGIMES == ("TUNED", "R1B", "R1P")
    assert P.SEED == RS.SCORE_SEED == 11 and P.SPLIT_SCOPE == RS.SPLIT_SCOPE == "pair"
    assert P.N_RUNS == {"TUNED": 80, "R1B": 60, "R1P": 60}
    tasks = P.plan_tasks()
    assert len(tasks) == len(set(tasks)) == 200
    assert tasks[:4] == [("TUNED", 0), ("R1B", 0), ("R1P", 0), ("TUNED", 1)]
    assert P.SMCAL == (2.0, 0.25)
    assert set(P.STORED) <= set(RS.CONFIGS) and not set(P.NEW) & set(RS.CONFIGS)
    assert all(R in RS.REGIMES for R in P.REGIMES)
    for path in P.PATHS.values():          # each path is a chain from its first ref
        for (_, x, _), (_, _, ref) in zip(path, path[1:]):
            assert x == ref
        assert all(c in P.ALL for _, x, ref in path for c in (x, ref))


def test_rasch_differs_from_onepl_by_the_documented_switches():
    onepl, rasch = RS.CONFIGS["onepl"], P.CONFIGS["rasch"]
    assert onepl["override"] == RS.LEVEL and rasch["level"] == "LEVEL"
    flags = H.Flags()
    added = {k: v for k, v in rasch["flags"].items() if onepl["flags"].get(k, getattr(flags, k)) != v}
    assert added == {k: False for k in P.RASCH_MINUS_ONEPL}
    assert {k: rasch["flags"][k] for k in onepl["flags"]} == onepl["flags"]
    assert P.CONFIGS["rasch_fit"]["flags"] == rasch["flags"] and P.CONFIGS["rasch_fit"]["level"] == "fit"
    H.Flags(**rasch["flags"])              # every switch is a Flags field


# --- the plain 1PL ----------------------------------------------------------------------------

def test_the_plain_1pl_at_budget_0_is_the_closed_form():
    """No labels: logit p ~ N(mu0, sigma_mu^2 + sigma_theta^2 + sigma_attr^2 +
    sigma_delta^2 + sigma_d^2 + sigma_g^2), p = E[sigmoid], with no floor and no
    slip, on a free-answer item and on a four-option one alike. onepl's floor and
    slip move the second."""
    rasch = HierPredictor(None, HY, **P.RASCH_FLAGS)
    v = HY.sigma_mu ** 2 + HY.sigma_theta ** 2 + HY.sigma_attr ** 2 + HY.sigma_delta ** 2 \
        + HY.sigma_d ** 2 + HY.sigma_g ** 2
    want = expect_sigmoid(HY.mu0, v)
    s = subject("m", provider="acme", release_date="2025-03-01")
    free = rasch.predict([s, item(0, "b", "grp=a")], [])
    mcq = rasch.predict([s, item(1, "b", "grp=a", text=MCQ)], [])
    assert free == pytest.approx(want, abs=1e-6) and mcq == pytest.approx(want, abs=1e-6)
    onepl = HierPredictor(None, HY, **RS.CONFIGS["onepl"]["flags"])
    assert onepl.predict([s, item(1, "b", "grp=a", text=MCQ)], []) > mcq + 0.05
    assert rasch.failures == 0 and onepl.failures == 0


def test_the_plain_1pl_links_nothing_across_benchmarks_or_groups():
    """A subject's labels on one benchmark leave its prediction on another at the
    budget-0 value (no link), and a group of failing items moves no new item of that
    group more than one of another group (no group effects); with the link or the
    groups on, both move."""
    m = subject("model-x", provider="acme")
    labeled = [[[m, item(j, "b1", f"grp={'a' if j % 2 else 'b'}")], int(j % 2 == 0)] for j in range(40)]
    labeled += [[[subject(f"o{k}"), item(j, "b1", f"grp={'a' if j % 2 else 'b'}")], int(j % 2 == 0)]
                for k in range(3) for j in range(40)]
    rasch = HierPredictor(None, HY, **P.RASCH_FLAGS)
    other = rasch.predict([m, item(99, "b2")], labeled)
    assert other == pytest.approx(rasch.predict([m, item(99, "b2")], []), abs=1e-9)
    linked = HierPredictor(None, HY, **{**P.RASCH_FLAGS, "link": True})
    assert abs(linked.predict([m, item(99, "b2")], labeled) - other) > 1e-4
    ga, gb = item(200, "b1", "grp=a"), item(201, "b1", "grp=b")
    assert rasch.predict([m, ga], labeled) == pytest.approx(rasch.predict([m, gb], labeled), abs=1e-12)
    grouped = HierPredictor(None, HY, **{**P.RASCH_FLAGS, "groups": True})
    assert grouped.predict([m, gb], labeled) - grouped.predict([m, ga], labeled) > 0.05


def test_the_fitted_level_is_fit_hypers_without_the_attribute_shift():
    pairs = make_pairs(6, 4, 100)
    for p in pairs:
        p.subject["provider"] = "openai" if p.subject_id < "s3" else "anthropic"
        p.subject["release_date"] = f"2025-0{1 + int(p.subject_id[1:])}-01"
    # uneven subject sets, so the benchmarks' mean attribute scores differ
    keep = {"b0": {"s0", "s1", "s2", "s3"}, "b1": {"s2", "s3", "s4", "s5"}}
    pairs = [p for p in pairs if p.subject_id in keep.get(p.benchmark_id, {p.subject_id})]
    held = ("b3",)
    prior, hyper = PR.build(pairs, held)
    assert prior is not None and prior.has_attributes
    mu0, sm, rep = P.noattr_level(pairs, held, hyper)
    _, full = PR.fit_hyper(pairs, held, prior)
    shift = {}
    for p in pairs:
        if p.benchmark_id not in held:
            shift.setdefault(p.benchmark_id, []).append(prior.attribute(p.subject)[0])
    assert sorted(rep["levels"]) == ["b0", "b1", "b2"] and rep["estimated"] == ["mu0", "sigma_mu"]
    for b, lv in rep["levels"].items():
        assert lv == pytest.approx(full["levels"][b] + np.mean(shift[b]), abs=1e-9)
    lv = np.array(list(rep["levels"].values()))
    assert mu0 == pytest.approx(lv.mean())
    assert sm == pytest.approx(lv.std(ddof=1) * math.sqrt(1 + 1 / 3) * PR.WIDEN[0])
    assert any(abs(np.mean(v)) > 1e-3 for v in shift.values())      # the shift is real here
    # its Hyper keeps prior.build's other fields and passes HierPredictor's exclusion check
    h = P.replace(hyper, mu0=mu0, sigma_mu=sm)
    HierPredictor(prior, h, **P.RASCH_FLAGS)
    assert (h.sigma_delta, h.sigma_d, h.excluded) == (hyper.sigma_delta, hyper.sigma_d, hyper.excluded)


# --- statistics and the decomposition --------------------------------------------------------

#: synthetic Brier offsets per config: legacy far behind, the others near ship
OFFSET = {"legacy": 0.04, "smooth": 0.02, "smcal": 0.006, "eb_fit": 0.025, "onepl": 0.002,
          "rasch": 0.004, "rasch_fit": 0.012, "empmean": 0.05}


def synthetic_rows(runs=14, seed=0, offset=OFFSET):
    """P1a-like rows (every RS config) and rows of the configs scored here, on the
    same runs."""
    rng = np.random.default_rng(seed)
    parents = list(RS.PARENTS)
    old, new = {}, {}
    for i in range(runs):
        k = int(rng.integers(3, 8))
        meta = [{"bench": f"{q}::g{j}", "parent": q, "subject": f"s{int(rng.integers(0, 5))}",
                 "eval": 50, "p": 0.3, "logit": float(rng.normal(-1, 1.5))}
                for j, q in enumerate(rng.choice(parents, k))]
        base = rng.uniform(0.05, 0.3, (k, 6))

        def res(c):
            b = np.clip(base + offset.get(c, 0.0) + rng.normal(0, 0.01, (k, 6)), 0, 1)
            return {"b": b.tolist(), "ece": (b / 2).tolist(), "q0": list(rng.random(k)), "calls": 10,
                    "mean_s": 0.001, "max_s": 0.01, "failures": 0, "unconverged": 0, "secs": 0.1}
        old[i] = {"meta": meta, "res": {c: res(c) for c in RS.CONFIGS}}
        new[i] = {"meta": json.loads(json.dumps(meta)), "res": {c: res(c) for c in P.NEW}}
    return old, new


def test_merged_rows_give_p1as_statistics_for_every_config():
    old, new = synthetic_rows()
    G = P.Rows("TUNED", P.merge(old, new))
    ref = RS.RegimeRows("TUNED", old)
    for c in ("onepl", "smcal", "legacy"):
        a, b = G.diff(c, "ship"), ref.diff(c, "ship")
        assert all(a[k] == b[k] for k in ("D", "run_se", "cluster_se", "strat_se"))
    d = G.diff("rasch", "ship")
    run_d = [np.mean(new[i]["res"]["rasch"]["b"] @ W6 - np.asarray(old[i]["res"]["ship"]["b"]) @ W6)
             for i in sorted(old)]
    assert d["D"] == pytest.approx(np.mean(run_d), abs=1e-12)
    assert d["run_se"] == pytest.approx(np.std(run_d, ddof=1) / math.sqrt(len(run_d)), abs=1e-12)
    assert np.std(G.boot_dist("rasch", "ship"), ddof=1) == pytest.approx(d["cluster_se"], abs=1e-12)
    s = G.config_summary("empmean")
    assert s["runs"] == len(old) and s["ALC"] == pytest.approx(
        np.mean([np.mean(np.asarray(new[i]["res"]["empmean"]["b"]) @ W6) for i in sorted(old)]))


def test_merge_refuses_rows_that_do_not_pair():
    old, new = synthetic_rows(runs=4)
    bad = copy.deepcopy(new)
    bad[2]["meta"][0]["subject"] = "someone else"
    with pytest.raises(ValueError):
        P.merge(old, bad)
    bad = copy.deepcopy(new)
    del bad[1]["res"]["rasch"]
    with pytest.raises(ValueError):
        P.merge(old, bad)
    with pytest.raises(ValueError):
        P.merge(old, {i: new[i] for i in (0, 1, 2)})


def test_every_path_telescopes_and_shares_need_a_clear_total():
    old, new = synthetic_rows()
    G = P.Rows("TUNED", P.merge(old, new))
    for name, steps in P.PATHS.items():
        s = P.path_summary(G, steps)
        tot = G.diff(steps[-1][1], steps[0][2])["D"]
        assert s["total"]["D"] == pytest.approx(tot, abs=1e-15)
        assert sum(p["D"] for p in s["steps"]) == pytest.approx(tot, abs=1e-12)
        assert s["sum_of_steps_minus_total"] < 1e-12
        for p, (label, x, ref) in zip(s["steps"], steps):
            assert (p["label"], p["x"], p["ref"]) == (label, x, ref)
            assert p["D"] == G.diff(x, ref)["D"]
            assert {"run_se", "cluster_se", "strat_se"} <= set(p) and p["budgets"]
        assert s["shares_reported"]                  # legacy is 0.04 behind: a clear total
        assert sum(p["share_of_total"]["point"] for p in s["steps"]) == pytest.approx(1.0)
        for p in s["steps"]:
            lo, hi = p["share_of_total"]["pct95"]
            assert lo <= p["share_of_total"]["point"] <= hi
    # a total within noise of 0 gives no share
    flat = dict(OFFSET, legacy=0.0, smcal=0.0)
    old, new = synthetic_rows(seed=1, offset=flat)
    G = P.Rows("TUNED", P.merge(old, new))
    s = P.path_summary(G, P.PATHS["calibration_first"])
    if abs(s["total"]["z_cluster"]) < P.SHARE_MIN_Z:
        assert not s["shares_reported"] and all(p["share_of_total"] is None for p in s["steps"])
        assert s["shares_note"]


def test_budget_view_reads_the_tables_budgets():
    old, new = synthetic_rows()
    G = P.Rows("TUNED", P.merge(old, new))
    p = P.part(G, "ship", "rasch")
    v = P.budget_view(p)
    assert sorted(v) == ["B0", "B1"]
    for k, b in ((0, v["B0"]), (1, v["B1"])):
        want = np.mean([np.mean(np.asarray(old[i]["res"]["ship"]["b"])[:, k]
                                - np.asarray(new[i]["res"]["rasch"]["b"])[:, k]) for i in sorted(old)])
        assert b["diff"] == pytest.approx(want, abs=1e-6)
        assert b["cluster_se"] > 0 and b["strat_se"] > 0 and b["run_se"] > 0


# --- BLE --------------------------------------------------------------------------------------

def fake_baseline(root):
    (root / "ble" / "src" / "agent").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "empirical_mean_ble_acquisition").mkdir()
    (root / "ble" / "model.py").write_text(
        'CFG = {"llm": "openai/gpt-5.6-luna", "max_steps": 10, "question_timeout": 240,\n'
        '       "reasoning_effort": "medium", "api_keys": {}}\n'
        'raise SystemExit("imported")\n')
    (root / "ble" / "submission_config.example.json").write_text(
        json.dumps({"llm": "openai/gpt-5.6-luna", "max_steps": 5, "api_keys": {"OPENAI_API_KEY": ""}}))
    (root / "ble" / "README.md").write_text(
        "Mock mode uses scripted LLM replies; its probabilities are illustrative. Needs "
        "OPENAI_API_KEY. Run tools/prepare_data.py (measurement-db, measurement-db-embed).")
    (root / "ble" / "labeling.py").write_text("ok = abs(prediction - 0.5) <= 0.15\n")
    (root / "ble" / "src" / "agent" / "llm_client.py").write_text("post_json; import litellm\n")
    (root / "ble" / "src" / "agent" / "belief_state.py").write_text(
        "MIN_PROBABILITY = 0.02\nMAX_PROBABILITY = 0.98\n")
    (root / "tools" / "smoke_test.py").write_text('parser.add_argument("--mock")\n')
    (root / "empirical_mean_ble_acquisition" / "labeling.py").write_text(
        "    from ble.model import predict\n")
    (root / "empirical_mean_ble_acquisition" / "model.py").write_text(
        "from empirical_mean.model import predict\n")


def test_the_ble_record_is_read_without_importing(tmp_path):
    fake_baseline(tmp_path)
    got = P.inspect_baseline(str(tmp_path))
    o, c = got["observed"], got["checks"]
    assert got["present"] and o["ble_default_llm"] == "openai/gpt-5.6-luna"
    assert o["ble_default_max_steps"] == 10 and o["ble_question_timeout_s"] == 240
    assert o["example_config_api_keys"] == ["OPENAI_API_KEY"] and "api_keys" not in o["example_config"]
    assert o["probability_bounds"] == [["MIN_PROBABILITY", "0.02"], ["MAX_PROBABILITY", "0.98"]] or \
        o["probability_bounds"] == [("MIN_PROBABILITY", "0.02"), ("MAX_PROBABILITY", "0.98")]
    assert all(v for k, v in c.items() if k != "commit_as_read") and not c["commit_as_read"]
    assert P.inspect_baseline(str(tmp_path / "nowhere")) == {
        "present": False, "repo": P.os.path.relpath(str(tmp_path / "nowhere"), P.ROOT)}
    assert "no network" in P.BLE_WHY_NOT and "empmean" in P.BLE_WHY_NOT
