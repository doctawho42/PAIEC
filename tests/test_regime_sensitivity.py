"""experiments/regime_sensitivity.py (P1a) and the library option it needed,
paiec.testlike.Regime.level_mix, on synthetic data (no download needed).

The lock: the decision rule, the plan file and the planner's constants block are
pinned by digest, so a change to any of them after the scoring seed was drawn
fails here. level_mix: empty, it leaves the sampler bit for bit as before the
option existed; one component of weight 1 is the Gaussian tilt; bad mixtures
raise. The stages' arithmetic: smcal's selection rule, the vector smoothed grid
against baselines.smoothed_mean, the task plan and its shards, the paired
statistics against ship_confirm.Block and level_calibration.compare, and the
rule's conditions (a) to (f) and readings.

After review: cutting the script's after-review blocks out gives back, byte for
byte, the script that scored the rows; the plan's appended amendments leave its
fixed text pinned; and the review parts' arithmetic on synthetic inputs."""
import hashlib
import math
from collections import defaultdict

import numpy as np
import pytest

import experiments.level_calibration as LC
from experiments import regime_sensitivity as RS
from experiments import ship_confirm as SC
from paiec import baselines as B
from paiec import testlike as T
from tests.test_testlike import KEYS, benchmark

W6 = np.asarray(RS.WEIGHTS, float)

# --- the lock ------------------------------------------------------------------------------

#: written here as well as in the script, so that editing both is needed to move either
RULE_SHA256 = "47e4735cf7ab0192e2c375054911221485e19ec71775bea8390b0e60287dc58e"
CONSTANTS_SHA256 = "c28c19d6cc4d7e298e8f954e3ef5d941a1c8695551058ac456c276adfdcfab81"
PLAN_SHA256 = "0a857944a8a6842e673480b934891ac6f5046125be91bbc710634de818bd4365"


def test_rule_text_is_the_fixed_one():
    assert RS.RULE_SHA256 == RULE_SHA256
    assert hashlib.sha256(RS.RULE_TEXT.encode("utf-8")).hexdigest() == RULE_SHA256


def test_constants_block_is_the_planners():
    block = RS.constants_block()
    assert block.startswith("# --- the lock ---") and block.endswith(RS.CONSTANTS_END)
    assert "RULE_TEXT = " in block and "N_RUNS = " in block and "CONFIGS = " in block
    assert RS.CONSTANTS_SHA256 == CONSTANTS_SHA256
    assert hashlib.sha256(block.encode("utf-8")).hexdigest() == CONSTANTS_SHA256


def test_lock_checks_pass():
    assert RS.PLAN_SHA256 == PLAN_SHA256
    res = RS.lock_checks()
    assert res and all(res.values()), {k: v for k, v in res.items() if not v}


def test_constants_as_the_rule_reads_them():
    assert RS.CONFIGS[RS.SHIP]["override"] == RS.LEVEL == {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}
    assert set(RS.FEEDBACK_REGIMES + RS.NO_LOSS_REGIMES + RS.PUBLIC_REGIMES) == set(RS.REGIMES)
    assert RS.TESTLIKE == ("TUNED", "READING", "AUDIT", "MIXTURE", "FLAT")
    assert all(RS.N_RUNS[r] == 80 for r in RS.TESTLIKE) and RS.N_RUNS["R1B"] == RS.N_RUNS["R1P"] == 60
    assert len(RS.SMCAL_GRID) == 70 and (4.0, 0.5) in RS.SMCAL_GRID
    assert sum(w for w, _, _ in RS.MIX) == pytest.approx(1.0)


# --- level_mix -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cat():
    pairs = (benchmark("alpha", 16, level=-1.5, seed=1)
             + benchmark("beta", 16, level=0.5, seed=2, repeats=2)
             + benchmark("gamma", 12, n_groups=5, per_group=60, level=-0.5, seed=3)
             + benchmark("solo", 1, n_groups=1, per_group=400, seed=4))
    return T.build_catalogue(pairs, keys=KEYS, min_subjects=3, strata=(2, 3), folds=4)


def reference_omega(S, cat):
    """Sampler.omega as paiec/testlike.py computed it before level_mix existed
    (b3ad531e), recomputed from the sampler's candidate pool."""
    r = S.regime
    cands, of = S.cands, S.of
    live = sorted(of)
    J = len(cat.pseudos)
    kw = dict(r.kind_weights)
    ws = np.array([c[3] for c in cands])
    ell = np.array([c[2] for c in cands])
    jj = np.array([c[0] for c in cands])
    wsum = np.bincount(jj, ws, J)
    level = np.full(J, np.nan)
    level[live] = np.bincount(jj, ws * ell, J)[live] / wsum[live]
    subjects, root = defaultdict(set), defaultdict(float)
    for j, p, *_ in cands:
        subjects[cat.pseudos[j].parent].add(p.subject_id)
    wk = {j: kw.get(cat.pseudos[j].kind, 1.0) for j in live}
    for j in live:
        root[cat.pseudos[j].parent] += wk[j] * math.sqrt(len(of[j]))
    base = np.zeros(J)
    for j in live:
        par = cat.pseudos[j].parent
        base[j] = math.log1p(len(subjects[par])) * (wk[j] * math.sqrt(len(of[j]))) / root[par]
    share = ws / wsum[jj]
    within = float(np.sqrt(np.sum(base[jj] * share * (ell - level[jj]) ** 2) / np.sum(base[jj] * share)))
    if r.level_mean is None:
        omega = base[jj] * share
    elif r.tilt == "benchmark":
        sd_b = math.sqrt(max(r.level_sd ** 2 - within ** 2, r.min_level_sd ** 2))
        L = level[live]
        f0 = (base[live][None, :] * T._npdf(L[:, None], L[None, :], r.bandwidth)).sum(1) / base[live].sum()
        t = np.zeros(J)
        t[live] = T._npdf(L, r.level_mean, sd_b) / f0
        t[live] = np.minimum(t[live], r.max_tilt * np.sum(base[live] * t[live]) / base[live].sum())
        omega = base[jj] * t[jj] * share
    else:
        pi = base[jj] * share
        f0 = (pi[None, :] * T._npdf(ell[:, None], ell[None, :], r.bandwidth)).sum(1) / pi.sum()
        t = T._npdf(ell, r.level_mean, r.level_sd) / f0
        t = np.minimum(t, r.max_tilt * np.sum(pi * t) / pi.sum())
        omega = pi * t
    return omega / omega.sum()


@pytest.mark.parametrize("regime", [
    T.Regime(min_release=None),
    T.Regime(min_release=None, level_mean=None),
    T.Regime(min_release=None, level_mean=-0.5, level_sd=2.0),
    T.Regime(min_release=None, tilt="pair", level_mean=-2.0),
])
def test_empty_level_mix_leaves_the_sampler_bit_for_bit(cat, regime):
    assert regime.level_mix == ()
    S = T.Sampler(cat, regime)
    assert np.array_equal(S.omega, reference_omega(S, cat))
    S2 = T.Sampler(cat, regime.with_(level_mix=()))
    assert np.array_equal(S.omega, S2.omega)


@pytest.mark.parametrize("tilt", ["benchmark", "pair"])
def test_one_component_mixture_is_the_gaussian_tilt(cat, tilt):
    g = T.Sampler(cat, T.Regime(min_release=None, tilt=tilt, level_mean=-1.0, level_sd=1.2))
    m = T.Sampler(cat, T.Regime(min_release=None, tilt=tilt, level_mix=((1.0, -1.0, 1.2),),
                                level_mean=3.0, level_sd=0.4))      # ignored under a mixture
    assert np.max(np.abs(g.omega - m.omega)) <= 1e-12
    for i in range(5):
        a, b = g.run(np.random.default_rng([7, i])), m.run(np.random.default_rng([7, i]))
        assert [(p.subject_id, p.benchmark_id, sorted(k)) for p, k in a] == \
               [(p.subject_id, p.benchmark_id, sorted(k)) for p, k in b]


def test_mixture_weight_moves_the_draws(cat):
    def mean_of(w_low):
        S = T.Sampler(cat, T.Regime(min_release=None, level_mix=((w_low, -2.5, 0.8), (1 - w_low, 0.5, 0.8))))
        assert isinstance(S.stats["benchmark_sd"], tuple) and len(S.stats["benchmark_sd"]) == 2
        return S.stats["one_draw_logit_mean"]
    assert mean_of(0.8) < mean_of(0.5) < mean_of(0.2)


@pytest.mark.parametrize("mix", [
    ((0.5, 0.0, 1.0), (0.6, 1.0, 1.0)),          # weights sum to 1.1
    ((-0.2, 0.0, 1.0), (1.2, 1.0, 1.0)),         # a negative weight
    ((1.0, 0.0),),                               # not a triple
    ((1.0, 0.0, 0.0),),                          # sd 0
    ((1.0, float("nan"), 1.0),),                 # mean not finite
])
def test_bad_mixtures_raise(cat, mix):
    with pytest.raises(ValueError):
        T.Sampler(cat, T.Regime(min_release=None, level_mix=mix))


def test_mixture_runs_keep_the_invariants(cat):
    r = T.Regime(min_release=None, level_mix=((0.3, -2.0, 0.5), (0.7, 0.0, 1.0)))
    S = T.Sampler(cat, r)
    for i in range(10):
        run = S.run(np.random.default_rng([5, i]))
        assert 1 <= len(run) <= r.n_pairs[1] and sum(len(k) for _, k in run) <= r.cap
        assert len({p.subject_id for p, _ in run}) == len(run)


# --- the plan's tasks ------------------------------------------------------------------------------

def test_tasks_are_run_index_major_and_shards_partition_them():
    tasks = RS.plan_tasks()
    assert len(tasks) == sum(RS.N_RUNS.values()) == 520
    assert len(set(tasks)) == len(tasks)
    assert tasks[:7] == [(R, 0) for R in RS.REGIMES]
    idx = [i for _, i in tasks]
    assert idx == sorted(idx)
    s0, s1 = RS.shard_tasks(0, 2), RS.shard_tasks(1, 2)
    assert not set(s0) & set(s1) and sorted(s0 + s1) == sorted(tasks)
    # any prefix of the plan leaves every regime with a prefix of its runs
    for n in (13, 200, 451):
        got = defaultdict(list)
        for R, i in tasks[:n]:
            got[R].append(i)
        assert all(v == list(range(len(v))) for v in got.values())


def test_regime_keys_register_with_level_calibration():
    key = RS.regime_key("MIXTURE", 11)
    assert LC.REGIMES[key] == ("testlike", 11, {"level_mix": RS.MIX})
    key = RS.regime_key("R1P", 11)
    assert LC.REGIMES[key] == ("r1", 11, "pair")


def test_realised_level():
    got = RS.realised([-4.0, -3.5, -1.0, 0.0, 1.0])
    assert got["mean"] == pytest.approx(-1.5) and got["sd"] == pytest.approx(np.std([-4, -3.5, -1, 0, 1]))
    assert got["share_below"] == pytest.approx(0.4) and got["rest_mean"] == pytest.approx(0.0)


# --- smcal ------------------------------------------------------------------------------------

def labeled_lists(seed=0):
    rng = np.random.default_rng(seed)
    subj = [{"normalized_name": f"s{k}", "release_date": "2025-01-01"} for k in range(3)]
    items = [{"benchmark_id": f"b{k % 2}", "item_content": f"q{k}"} for k in range(12)]
    lab = [((subj[int(rng.integers(0, 3))], items[int(rng.integers(0, 12))]), int(rng.random() < 0.4))
           for _ in range(int(rng.integers(0, 40)))]
    return subj, items, lab


def test_smoothed_grid_is_baselines_smoothed_mean():
    g = RS.smgrid_factory()()
    for seed in range(20):
        subj, items, lab = labeled_lists(seed)
        for s in subj:
            for it in items[:4]:
                v = g((s, it), lab)
                assert v.shape == (len(RS.SMCAL_GRID),)
                for j, (n0, m0) in enumerate(RS.SMCAL_GRID):
                    assert v[j] == B.smoothed_mean(n0, m0)((s, it), lab)


def test_smcal_selection_rule():
    grid = [(1.0, 0.2), (4.0, 0.2), (4.0, 0.5), (8.0, 0.3), (8.0, 0.45)]
    sel = {(1.0, 0.2): 0.150, (4.0, 0.2): 0.152, (4.0, 0.5): 0.170, (8.0, 0.3): 0.1600005,
           (8.0, 0.45): 0.160}
    guard = {(1.0, 0.2): [0.010, 0.0], (4.0, 0.2): [0.004, 0.001], (4.0, 0.5): [0.0, 0.0],
             (8.0, 0.3): [0.002, 0.003], (8.0, 0.45): [0.003, -0.001]}
    chosen, failed, passing = RS.select_smcal({g: sel[g] for g in grid}, guard)
    # (1, 0.2) and (4, 0.2) fail the guard; (8, 0.3) and (8, 0.45) tie within 1e-6:
    # the same n0, so the m0 nearer 0.5
    assert not failed and set(passing) == {(4.0, 0.5), (8.0, 0.3), (8.0, 0.45)}
    assert chosen == (8.0, 0.45)
    # ties go to the larger n0 first
    sel2 = {**sel, (4.0, 0.5): 0.1600001}
    assert RS.select_smcal(sel2, guard)[0] == (8.0, 0.45)
    sel3 = {(4.0, 0.5): 0.16, (8.0, 0.3): 0.1600004}
    assert RS.select_smcal(sel3, guard)[0] == (8.0, 0.3)
    # nothing passes: the unconstrained argmin, flagged
    bad = {g: [0.01, 0.01] for g in grid}
    chosen, failed, passing = RS.select_smcal(sel, bad)
    assert failed and passing == [] and chosen == (1.0, 0.2)


# --- statistics ---------------------------------------------------------------------------------

def synthetic_rows(runs=14, seed=0):
    """Rows as the score stage writes them, for every config, on runs of 3 to 7 pairs
    over the four multi-subject parents."""
    rng = np.random.default_rng(seed)
    rows = {}
    shift = {c: rng.normal(0, 0.01) for c in RS.CONFIGS}
    for i in range(runs):
        k = int(rng.integers(3, 8))
        meta = []
        for j in range(k):
            q = RS.PARENTS[int(rng.integers(0, 4))]
            meta.append({"bench": f"{q}::g{j}", "parent": q, "subject": f"s{int(rng.integers(0, 5))}",
                         "eval": 50, "p": round(float(rng.random()), 5), "logit": float(rng.normal(-1, 2))})
        base = rng.uniform(0.05, 0.3, (k, 6))
        res = {}
        for c in RS.CONFIGS:
            b = np.clip(base + shift[c] + rng.normal(0, 0.01, (k, 6)), 0, 1)
            res[c] = {"b": b.tolist(), "ece": rng.uniform(0, 0.2, (k, 6)).tolist(),
                      "q0": rng.uniform(0, 1, k).tolist(), "calls": 100, "mean_s": 0.01,
                      "max_s": 0.05, "failures": 0, "unconverged": 0, "secs": 1.0}
        rows[i] = {"meta": meta, "res": res}
    return rows


def test_paired_difference_is_ship_confirms_and_level_calibrations():
    rows = synthetic_rows()
    G = RS.RegimeRows("READING", rows)
    ex = G.diff("aggr", RS.SHIP)
    t = G.table("aggr", RS.SHIP)
    assert t["diff"]["mean"] == pytest.approx(ex["D"], abs=1e-6)
    for k in ("run_se", "cluster_se", "strat_se"):
        assert t["diff"][k] == pytest.approx(ex[k], abs=1e-6)
    for q in RS.PARENTS:
        assert t["per_parent"][q]["mean"] == pytest.approx(ex["Pq"][q], abs=1e-6)
    assert t["parent_level"]["mean"] == pytest.approx(ex["PL"], abs=1e-6)
    assert ex["U95"] == pytest.approx(ex["D"] + 1.96 * ex["cluster_se"])
    assert ex["pct"][0] < ex["D"] < ex["pct"][1]
    # D is the mean over runs of the run's pair-mean difference
    want = np.mean([np.mean(np.asarray(r["res"]["aggr"]["b"]) @ W6 - np.asarray(r["res"][RS.SHIP]["b"]) @ W6)
                    for r in rows.values()])
    assert ex["D"] == pytest.approx(want, abs=1e-12)
    # Pq weights each appearance by 1 / its run's pair count
    num = den = 0.0
    for r in rows.values():
        d = np.asarray(r["res"]["aggr"]["b"]) @ W6 - np.asarray(r["res"][RS.SHIP]["b"]) @ W6
        for m, x in zip(r["meta"], d):
            if m["parent"] == "matharena":
                num += x / len(r["meta"])
                den += 1 / len(r["meta"])
    assert ex["Pq"]["matharena"] == pytest.approx(num / den, abs=1e-12)
    # level_calibration.compare on the same rows
    raw, metas = {}, {}
    for i, r in rows.items():
        metas[("tl", i)] = r["meta"]
        raw[f"tl|{i}|base"] = {"res": {
            n: {"alc": list(np.asarray(r["res"][c]["b"]) @ W6), "b": list(np.asarray(r["res"][c]["b"]).mean(0)),
                "ece0": [0, 0], "ece_alc": 0, "q0": 0}
            for n, c in ((LC.SMOOTHED, "aggr"), (LC.PRED, RS.SHIP))}, "timing": {}}
    ref = LC.compare(raw, metas, "tl", sorted(rows), [LC.SMOOTHED])[LC.SMOOTHED]["diff"]
    for k in ("mean", "run_se", "cluster_se", "strat_se"):
        assert ref[k] == pytest.approx(ex[k if k != "mean" else "D"], abs=1e-12)


def test_regime_level_and_config_summary():
    rows = synthetic_rows()
    G = RS.RegimeRows("TUNED", rows)
    lv = G.level()
    L = [p["logit"] for r in rows.values() for p in r["meta"]]
    assert lv["mean"] == pytest.approx(np.mean(L)) and lv["cluster_se"] > 0
    lo, hi = lv["leave_one_parent_out_range"]
    assert lo <= hi
    s = G.config_summary(RS.SHIP)
    assert s["ALC"] == pytest.approx(np.mean([np.mean(np.asarray(r["res"][RS.SHIP]["b"]) @ W6)
                                              for r in rows.values()]))
    rk = RS.ranks(G)
    assert [r["config"] for r in rk] == sorted(RS.CONFIGS, key=lambda c: np.mean(
        [np.mean(np.asarray(r["res"][c]["b"]) @ W6) for r in rows.values()]))


def test_latency_is_pooled_over_tasks():
    rows = {"TUNED": synthetic_rows(3), "R1B": synthetic_rows(2, seed=1)}
    rows = {R: rows.get(R, {}) for R in RS.REGIMES}
    rows["TUNED"][0]["res"]["legacy"].update(mean_s=0.04, max_s=1.5)
    lat = RS.latency(rows)["all"]
    calls = 100 * 5
    assert lat[RS.SHIP]["calls"] == calls and lat[RS.SHIP]["ratio_to_ship"] == pytest.approx(1.0)
    assert lat["legacy"]["mean_ms"] == pytest.approx(1e3 * (0.04 * 100 + 0.01 * 400) / calls)
    assert lat["legacy"]["max_s"] == 1.5


# --- the rule -----------------------------------------------------------------------------------

def fake(D, U95=None, Pq=None):
    Pq = Pq if Pq is not None else {q: D for q in RS.PARENTS}
    return {"D": D, "U95": D + 0.002 if U95 is None else U95, "Pq": Pq,
            "PL": float(np.mean([Pq[q] for q in RS.PARENTS]))}


def rule_inputs(good="wide35"):
    """Every config but `good` loses in READING; `good` meets (a) to (e)."""
    diffs, glegacy = {}, {}
    for R in RS.REGIMES:
        diffs[R], glegacy[R] = {}, {}
        for X in RS.CONFIGS:
            if X == RS.SHIP:
                continue
            if X == good:
                D = -0.004 if R in RS.FEEDBACK_REGIMES else 0.001 if R in RS.NO_LOSS_REGIMES else 0.0005
            else:
                D = 0.003
            diffs[R][X] = fake(D)
            glegacy[R][X] = 0.0 if X == "legacy" else -0.001
    lat = {c: {"mean_ms": 5.0, "ratio_to_ship": 1.0, "max_s": 0.5} for c in RS.CONFIGS}
    return diffs, glegacy, lat


def test_rule_finds_a_candidate_that_meets_every_condition():
    diffs, gl, lat = rule_inputs()
    r = RS.apply_rule(diffs, gl, lat, True, True)
    assert r["applied"] and r["candidates"] == ["wide35"]
    assert r["outcome"] == "candidates: wide35"
    assert r["p116"].startswith("yes") and r["eb"].startswith("no")
    assert set(r["negative"]) == set(RS.CONFIGS) - {RS.SHIP, "wide35"}
    assert r["table"]["wide35"]["reading"] == "candidate"
    assert r["table"]["aggr"]["reading"].startswith("not a candidate: (a) READING")


@pytest.mark.parametrize("breaker,cond", [
    (lambda d, g, l: d["READING"]["wide35"].update(D=-0.0019, U95=-0.0001), "a"),
    (lambda d, g, l: d["AUDIT"]["wide35"].update(U95=0.0), "a"),
    (lambda d, g, l: d["AUDIT"]["wide35"].update(PL=0.0), "b"),
    (lambda d, g, l: d["READING"]["wide35"]["Pq"].update(matharena=0.0041), "b"),
    (lambda d, g, l: d["FLAT"]["wide35"].update(D=0.0021), "c"),
    (lambda d, g, l: d["R1P"]["wide35"].update(D=0.0011), "d"),
    (lambda d, g, l: g["R1B"].update(wide35=0.0031), "d"),
    (lambda d, g, l: l["wide35"].update(ratio_to_ship=2.01), "e"),
    (lambda d, g, l: l["wide35"].update(max_s=2.01), "e"),
])
def test_rule_conditions_each_bind(breaker, cond):
    diffs, gl, lat = rule_inputs()
    breaker(diffs, gl, lat)
    r = RS.apply_rule(diffs, gl, lat, True, True)
    assert r["candidates"] == [] and r["outcome"] == "no candidate: SHIP stays"
    assert not r["table"]["wide35"]["holds"][cond]
    assert r["p116"].startswith("no")


def test_rule_boundaries_are_inclusive_where_the_text_says_so():
    diffs, gl, lat = rule_inputs()
    for R in RS.FEEDBACK_REGIMES:
        diffs[R]["wide35"].update(D=-0.002, U95=-1e-9)
        diffs[R]["wide35"]["Pq"]["matharena"] = 0.004
    diffs["TUNED"]["wide35"]["D"] = 0.002
    diffs["R1B"]["wide35"]["D"] = 0.001
    gl["R1P"]["wide35"] = 0.003
    lat["wide35"].update(ratio_to_ship=2.0, max_s=2.0)
    assert RS.apply_rule(diffs, gl, lat, True, True)["candidates"] == ["wide35"]


def test_rule_is_not_applied_without_its_preconditions():
    diffs, gl, lat = rule_inputs()
    r = RS.apply_rule(diffs, gl, lat, False, True)
    assert not r["applied"] and r["outcome"] == "rule not applied: check failed" and r["candidates"] == []
    assert not r["table"]["wide35"]["holds"]["f"]
    r = RS.apply_rule(diffs, gl, lat, True, False)
    assert not r["applied"] and "not complete" in r["outcome"]


def test_rule_orders_candidates_and_reads_eb():
    diffs, gl, lat = rule_inputs(good="eb_ship")
    for R in RS.REGIMES:
        diffs[R]["smcal"] = dict(diffs[R]["eb_ship"])
        diffs[R]["smcal"]["Pq"] = dict(diffs[R]["eb_ship"]["Pq"])
        gl[R]["smcal"] = gl[R]["eb_ship"]
    for R in RS.FEEDBACK_REGIMES:
        diffs[R]["smcal"]["D"] = -0.006
    r = RS.apply_rule(diffs, gl, lat, True, True)
    assert r["candidates"] == ["smcal", "eb_ship"]
    assert r["eb"].startswith("revisit") and r["p116"].startswith("no")


# --- deviations --------------------------------------------------------------------------------

def test_legacy_check_reads_testlike_checks_own_rounding():
    """D1: the legacy reproduction compares at the precision testlike_check stores."""
    import os
    with open(os.path.join(RS.ROOT, "experiments", "testlike_check.py")) as f:
        src = f.read()
    assert f'"pred": [round(r["brier"][b], {RS.LEGACY_STORED_DECIMALS}) for b in BUDGETS]' in src
    ids = [d["id"] for d in RS.DEVIATIONS]
    assert len(ids) == len(set(ids)) and all(d["cause"] and d["fix"] for d in RS.DEVIATIONS)
    # the constants the deviation leaves alone
    assert RS.REPRO_TOL["legacy"] == 1e-9 and RS.REPRO_EXEMPT_PARENTS["legacy"] == ("matharena",)


# --- summarise, end to end on synthetic rows ------------------------------------------------------

def write_rows(rows_dir, n_runs, shift, seed=0):
    """Rows for every planned task of a shrunken plan: every config is SHIP's Brier
    plus a constant shift (by config and regime), so each D is known exactly."""
    import json
    import os
    rng = np.random.default_rng(seed)
    comps = {}
    for R in RS.REGIMES:
        comps[R] = {}
        os.makedirs(os.path.join(rows_dir, R), exist_ok=True)
        for i in range(n_runs[R]):
            k = int(rng.integers(4, 9))
            meta = [{"bench": f"{RS.PARENTS[j % 4]}::g{j}", "parent": RS.PARENTS[j % 4],
                     "subject": f"s{int(rng.integers(0, 6))}", "eval": 50,
                     "p": round(float(rng.random()), 5), "logit": round(float(rng.normal(-1, 2)), 4)}
                    for j in range(k)]
            base = rng.uniform(0.05, 0.3, (k, 6))
            res = {}
            for c in RS.CONFIGS:
                b = base + shift(c, R)
                res[c] = {"b": b.tolist(), "ece": np.full((k, 6), 0.05).tolist(), "q0": [0.4] * k,
                          "calls": 50, "mean_s": 0.01, "max_s": 0.1, "failures": 0, "unconverged": 0,
                          "secs": 0.5}
            res["eb_ship"]["eb"] = [[b, 10 * b, -2.5, 2.5] for b in RS.BUDGETS]
            row = {"regime": R, "i": i, "meta": meta, "res": res, "smcal": [2.0, 0.25], "task_s": 1.0,
                   "rss_gb": 1.0, "lib": {"paiec/hier.py": "x"}, "script": "y"}
            with open(RS.row_path(R, i, rows=rows_dir), "w") as f:
                json.dump(row, f)
            comps[R][str(i)] = meta
    with open(os.path.join(rows_dir, RS.COMPOSITIONS), "w") as f:
        json.dump({"seed": RS.SCORE_SEED, "runs": comps}, f)


def test_summarise_end_to_end(tmp_path, monkeypatch):
    import json
    n_runs = {R: 5 for R in RS.REGIMES}
    monkeypatch.setattr(RS, "N_RUNS", n_runs)
    monkeypatch.setattr(RS, "BOOTS", 200)
    monkeypatch.setattr(SC, "BOOTS", 200)                      # ship_confirm.Block's resamples
    monkeypatch.setattr(RS, "OUT", str(tmp_path / "out.json"))
    monkeypatch.setattr(RS, "lock_checks", lambda: {"shrunken plan": True})   # the lock reads N_RUNS

    def shift(c, R):
        if c == "legacy":
            return 0.0415                                   # the TUNED consistency check holds
        if c == "wide35":
            return -0.004 if R in RS.FEEDBACK_REGIMES else 0.0
        return 0.0 if c == RS.SHIP else 0.003
    rows_dir = str(tmp_path / "rows")
    write_rows(rows_dir, n_runs, shift)
    ok = {"ok": True}
    state = {"lock": ok, "smcal": {"ok": True, "chosen": [2.0, 0.25]}, "regimes": ok,
             "reproduce": {"ok": True, "planned_tolerances_ok": False, "deviations": RS.DEVIATIONS}}
    with open(RS.OUT, "w") as f:
        json.dump(state, f)
    RS.stage_summarise(rows_dir=rows_dir)
    with open(RS.OUT) as f:
        out = json.load(f)
    s, rule = out["summary"], out["rule"]
    assert s["complete"] and all(s["checks"].values()), s["checks"]
    assert s["tuned_consistency"]["ship_minus_legacy"] == pytest.approx(-0.0415, abs=1e-12)
    assert s["vs_ship"]["READING"]["wide35"]["exact"]["D"] == pytest.approx(-0.004, abs=1e-12)
    assert s["vs_ship"]["AUDIT"]["aggr"]["table"]["diff"]["mean"] == pytest.approx(0.003, abs=1e-6)
    assert s["decomposition"]["TUNED"]["ship minus legacy"]["mean"] == pytest.approx(-0.0415, abs=1e-12)
    assert s["eb_traces"]["eb_ship | TUNED"]["B31"]["mu0"] == pytest.approx(-2.5)
    assert s["ranks"]["READING"][0]["config"] == "wide35"
    assert rule["applied"] and rule["candidates"] == ["wide35"]
    assert rule["outcome"] == "candidates: wide35" and rule["p116"].startswith("yes")
    # held at the plan's own tolerances, D1's check fails and the rule is not applied
    assert rule["literal_reading"]["outcome"] == "rule not applied: check failed"
    assert rule["deviations"][0]["id"].startswith("D1")
    assert out["status"] == "complete"


def test_summarise_waits_for_the_full_set(tmp_path, monkeypatch):
    import json
    import os
    n_runs = {R: 3 for R in RS.REGIMES}
    monkeypatch.setattr(RS, "N_RUNS", n_runs)
    monkeypatch.setattr(RS, "BOOTS", 100)
    monkeypatch.setattr(SC, "BOOTS", 100)                      # ship_confirm.Block's resamples
    monkeypatch.setattr(RS, "OUT", str(tmp_path / "out.json"))
    monkeypatch.setattr(RS, "lock_checks", lambda: {"shrunken plan": True})   # the lock reads N_RUNS
    rows_dir = str(tmp_path / "rows")
    write_rows(rows_dir, n_runs, lambda c, R: 0.0 if c != "legacy" else 0.0415)
    os.remove(RS.row_path("FLAT", 2, rows=rows_dir))
    with open(RS.OUT, "w") as f:
        json.dump({"lock": {"ok": True}}, f)
    RS.stage_summarise(rows_dir=rows_dir)
    with open(RS.OUT) as f:
        out = json.load(f)
    assert not out["summary"]["complete"] and out["summary"]["scored"]["FLAT"] == 2
    assert not out["rule"]["applied"] and "not complete" in out["rule"]["outcome"]
    assert out["status"] == "partial"


# --- after review: the review stage ---------------------------------------------------------------

#: the script as it scored every row; cutting the after-review blocks out must give it back
SCORED_SCRIPT_SHA256 = "0bef6bf07d23bd876dc128a2330b10c8a0d37076d5214f7712e9251a2b56c021"


def test_cutting_the_review_blocks_gives_back_the_scored_script():
    assert RS.SCORED_SCRIPT_SHA256 == SCORED_SCRIPT_SHA256
    assert hashlib.sha256(RS.scored_source().encode("utf-8")).hexdigest() == SCORED_SCRIPT_SHA256


def test_review_blocks_must_balance(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("a = 1\n    # >>> after review: why\n    b = 2\n\n    # <<< after review\nc = 3\n")
    assert RS.scored_source(str(p)) == "a = 1\nc = 3\n"
    for bad in ("# >>> after review\nx = 1\n",
                "x = 1\n# <<< after review\n",
                "# >>> after review\n# >>> after review\n# <<< after review\n"):
        p.write_text(bad)
        with pytest.raises(ValueError):
            RS.scored_source(str(p))


def test_plan_amendments_leave_the_fixed_text_pinned(tmp_path, monkeypatch):
    fixed, amend = RS.plan_parts()
    assert hashlib.sha256(fixed).hexdigest() == PLAN_SHA256
    assert amend == b"" or amend.startswith(RS.PLAN_AMENDMENTS.encode("utf-8"))
    p = tmp_path / "plan.md"
    p.write_bytes(fixed + b"\n## Amendments after review\n\nA note.\n")
    f2, a2 = RS.plan_parts(str(p))
    assert f2 == fixed and a2 == b"\n## Amendments after review\n\nA note.\n"
    # the lock still catches an edit to the fixed text, amended or not
    edited = fixed.replace(b"Fixed at", b"Fixed on", 1)
    monkeypatch.setattr(RS, "plan_parts", lambda path=None: (edited, b"\n## Amendments after review\n"))
    assert not RS.lock_checks()["plan_sha256"]


def test_compare_json():
    a = {"x": 1.0, "y": [1, 2, {"z": "s"}], "provenance": {"utc": "t0"}, "k_after_review": 1, "f": True}
    b = {"x": 1.5, "y": [1, 2, {"z": "s"}], "provenance": {"utc": "t1"}, "f": True}
    nums, bad = RS.compare_json(a, b, {"provenance"})
    assert max(nums) == 0.5 and bad == []
    nums, bad = RS.compare_json(a, {**b, "f": 1, "y": [1, 2]}, {"provenance"})
    assert "/f" in bad and any("length" in x for x in bad)
    assert RS.compare_json({"a": 1}, {"b": 1})[1] == ["/a", "/b"]


def test_first_failure_position():
    assert RS.first_failure_position("." * 72 + " [ 11%]\n" + "." * 72 + " [ 23%]\n....F......exit 143\n") == 149
    assert RS.first_failure_position("....s..E\n") == 8
    assert RS.first_failure_position("......\n===== 6 passed =====\n../../x.py:60: F\n") is None


def test_save_review_merges_parts(tmp_path):
    import json
    path = str(tmp_path / "out.json")
    with open(path, "w") as f:
        json.dump({"summary": {"n": 1}, "review": {"old": {"v": 0}}}, f)
    RS.save_review("a", {"x": np.float64(0.5), "t": (1, 2)}, path)
    RS.save_review("b", {"y": np.int64(3)}, path)
    with open(path) as f:
        out = json.load(f)
    assert out["summary"] == {"n": 1} and out["review"]["old"] == {"v": 0}
    assert out["review"]["a"] == {"x": 0.5, "t": [1, 2]} and out["review"]["b"] == {"y": 3}


def level_rows():
    """One run per regime; MIXTURE's appearances below -4 sit on two pseudo-benchmarks."""
    logits = {"MIXTURE": [("multi_swebench::q1of3", -5.0), ("multi_swebench::q1of3", -4.5),
                          ("multi_swebench::q1of2", -4.2), ("matharena::g1", -4.1),
                          ("matharena::g1", -3.5), ("real_webagents::g0", 0.0)]}
    rows = {}
    for R in RS.REGIMES:
        app = logits.get(R, [("matharena::g1", -1.0), ("researchcodebench::g0", 1.0)])
        rows[R] = {0: {"meta": [{"bench": b, "parent": b.split("::")[0], "logit": x} for b, x in app]}}
    return rows


def test_review_levels():
    state = {"summary": {"realised": {"MIXTURE": RS.realised([x for _, x in [(0, -5.0), (0, -4.5), (0, -4.2),
                                                                             (0, -4.1), (0, -3.5), (0, 0.0)]])}},
             "regimes": {"cal": {}}}
    lv = RS.review_levels(state, level_rows())
    m = lv["MIXTURE"]
    assert m["share_below_-4"] == pytest.approx(4 / 6) and m["share_below"] == pytest.approx(5 / 6)
    assert m["below_-4"]["by_pseudo_benchmark"] == {"multi_swebench::q1of3": 2, "matharena::g1": 1,
                                                    "multi_swebench::q1of2": 1}
    assert m["below_-4"]["top_two"] == ["multi_swebench::q1of3", "matharena::g1"]
    assert m["below_-4"]["top_two_share"] == pytest.approx(0.75)
    assert m["below_-4"]["by_parent"] == {"matharena": 1, "multi_swebench": 3}
    assert m["equals_summary"] and not lv["READING"]["equals_summary"]
    assert m["minus_target"]["share_below"] == pytest.approx(5 / 6 - RS.TARGETS["MIXTURE"]["share_below"])
    assert lv["READING"]["mean"] == 0.0 and lv["READING"]["below_-4"]["top_two_share"] is None


def test_review_headline_scopes_the_near_ship_claim():
    D = {"wide35": (0.00061, -0.0003), "eb_adapt": (0.0003, -0.0005), "eb_ship": (0.0003, -0.0002),
         "aggr": (0.0015, 0.0005), "wide50": (0.0023, 0.0006), "onepl": (0.0026, 0.0016),
         "eb_fit": (0.0173, 0.0233), "legacy": (0.02, 0.03), "smooth": (0.01, 0.01), "smcal": (0.008, 0.004)}
    vs = {R: {X: {"exact": {"D": D[X][R == "AUDIT"], "run_se": 0.001, "strat_se": 0.001, "U95": 0.0,
                            "cluster_se": 0.0004 if X in ("wide35", "eb_adapt", "eb_ship") else 0.0012}}
              for X in D} for R in RS.REGIMES}
    h = RS.review_headline({"summary": {"vs_ship": vs}})
    assert h["near_ship"]["hier_configs"] == ["eb_adapt", "eb_ship", "wide35"]
    assert set(h["near_ship"]["other_hier_configs"]) == {"aggr", "eb_fit", "wide50", "onepl"}
    assert h["near_ship_cluster_se_range"] == [0.0004, 0.0004] and h["hier_cluster_se_range"] == [0.0004, 0.0012]
    assert h["feedback_max_abs_D"]["onepl"] == 0.0026


def test_review_provenance_reads_d1_and_both_readings(monkeypatch):
    rows = {R: {} for R in RS.REGIMES}
    for k, R in enumerate(RS.REGIMES):
        rows[R][0] = {"utc": f"2026-10-01T17:0{k}:00Z", "script": RS.SCORED_SCRIPT_SHA256[:16], "seed": 11,
                      "shard": "0/2", "lib": {"paiec/hier.py": "x"}}
    leg = [{"check": f"legacy vs testlike_check.json raw['check|default|{i}']", "max_abs_brier": 4.9e-7,
            "equal_at_stored_precision": True} for i in range(2)]
    holds = {X: {k: k != "a" for k in "abcdef"} for X in RS.CONFIGS if X != RS.SHIP}
    holds["wide35"].update(a=True, c=False)
    state = {"lock": {"checked_utc": "2026-10-01T19:00:00Z"},
             "smcal": {"provenance": {"script": "s1", "utc": "2026-10-01T15:27:57Z"}},
             "regimes": {"provenance": {"script": "s1", "utc": "2026-10-01T15:26:54Z"}},
             "reproduce": {"provenance": {"script": "s2", "utc": "2026-10-01T16:26:39Z"}, "checks": leg,
                           "planned_tolerances_ok": False},
             "summary": {"provenance": {"script": RS.SCORED_SCRIPT_SHA256[:16], "utc": "2026-10-01T19:16:01Z"}},
             "rule": {"outcome": "no candidate: SHIP stays", "candidates": [],
                      "literal_reading": {"outcome": "rule not applied: check failed", "candidates": []},
                      "table": {X: {"holds": h} for X, h in holds.items()}}}
    p = RS.review_provenance(state, rows)
    d1 = p["d1"]
    assert d1["fixed_before_first_row"] and d1["first_row_utc"] == "2026-10-01T17:00:00Z"
    assert d1["all_within_rounding"] and d1["equal_at_stored_precision"]
    assert d1["first_reproduce_failed_at_planned_tolerance"]
    assert d1["configs_meeting_a"] == ["wide35"] and d1["configs_meeting_a_to_e"] == []
    assert p["script"]["rows_record_scored"] and p["script"]["summary_records_scored"]
    assert p["script"]["cutting_review_blocks_gives_scored"]
    assert p["rows"]["n"] == len(RS.REGIMES) and p["rows"]["last_utc"] == "2026-10-01T17:06:00Z"
    assert p["plan"]["fixed_is_pinned"] and all(p["lock_now"].values())
    leg[0]["max_abs_brier"] = 6e-7
    assert not RS.review_provenance(state, rows)["d1"]["all_within_rounding"]


def test_review_notes_and_parts():
    ids = [n["id"] for n in RS.REVIEW_NOTES]
    assert ids == [f"V{k}" for k in range(1, 11)]
    assert all(n["finding"] and n["done"] and n["left"] for n in RS.REVIEW_NOTES)
    assert set(RS.LIGHT_PARTS) < set(RS.REVIEW_PARTS)
    assert RS.RESCORE_TASKS == tuple((R, RS.N_RUNS[R] - 1) for R in RS.REGIMES)
    assert all(rel in RS.EVIDENCE for rel in ("rule.txt", "impl/regime_sensitivity.orig.py",
                                              "impl/dry_out.json"))
    with pytest.raises(SystemExit):
        RS.main(["review", "--parts", "nonesuch"])
