"""experiments/ablations_at_level.py (S4) on synthetic data (no download needed).

The plan: the plan's S4 row and this script's constants block are pinned by digest;
the regimes, run counts and split scope are P1a's; every config the plan lists is
scored, with the references and the pilots' definitions beside them. A config's
hyperparameters: `set` replaces fields, `scale` multiplies the leave-one-parent-out
fit, ship and P1a's shared configs build P1a's Hyper exactly. Hidden-v weights: the
formative tables' counts, and weights that are the equal ones when the replica's v
composition already matches. Under equal weights the weighted difference is P1a's
RegimeRows.diff. summarise end to end on a shrunken dry-run plan."""
import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest

from experiments import ablations_at_level as A
from experiments import regime_sensitivity as RS
from paiec.hier import Hyper

#: written here as well as in the script, so that editing both is needed to move either
S4_ROW_SHA256 = "f0a9af0288f3bf85747bd4e4764c5e93a320e9e43215c0f7e0dbb4e2c2859b20"
HY = Hyper(mu0=-0.7, sigma_mu=1.1, sigma_theta=0.2, sigma_delta=0.9, sigma_attr=0.6,
           sigma_d=1.3, sigma_g=0.8)


# --- the plan --------------------------------------------------------------------------------

def test_constants_block_is_pinned():
    assert A.CONSTANTS_SHA256 is not None
    assert hashlib.sha256(A.constants_block().encode("utf-8")).hexdigest() == A.CONSTANTS_SHA256
    assert A.S4_ROW_SHA256 == S4_ROW_SHA256


def test_lock_checks_pass():
    res = A.lock_checks()
    assert all(res.values()), [k for k, v in res.items() if not v]


def test_the_plan_is_p1as_regimes_on_a_fresh_seed():
    assert A.REGIMES == ("TUNED", "READING", "AUDIT", "MIXTURE", "FLAT", "R1B", "R1P")
    assert A.N_RUNS == {"TUNED": 80, "READING": 80, "AUDIT": 80, "MIXTURE": 80, "FLAT": 80,
                        "R1B": 60, "R1P": 60}
    assert A.SPLIT_SCOPE == "pair" and A.DRY is None
    tasks = A.plan_tasks()
    assert len(tasks) == len(set(tasks)) == 520
    assert tasks[:8] == [(R, 0) for R in A.REGIMES] + [("TUNED", 1)]
    shards = [A.shard_tasks(k, 3) for k in range(3)]
    assert sorted(t for s in shards for t in s) == sorted(tasks)
    used = set(RS.USED_SEEDS["testlike"] + RS.USED_SEEDS["r1"]) | {RS.CAL_SEED, RS.SCORE_SEED}
    assert A.SEED not in used and A.DRY_SEED != A.SEED


def test_every_listed_config_is_scored():
    assert len(A.S4_LIST) == 9 and set(A.S4_LIST) | set(A.REFERENCES) | set(A.PILOT_FAMILY) == set(A.CONFIGS)
    assert not set(A.S4_LIST) & set(A.PILOT_FAMILY)
    assert A.CONFIGS[A.SHIP]["set"] == A.LEVEL == RS.LEVEL


# --- hyperparameters --------------------------------------------------------------------------

def test_ship_and_p1as_configs_build_p1as_hyper():
    for c in ("ship", "wide35", "wide50"):
        assert A.config_hyper(A.CONFIGS[c], HY) == replace(HY, **RS.CONFIGS[c]["override"])
    assert A.CONFIGS["eb_ship"]["lc_name"] == RS.CONFIGS["eb_ship"]["lc_name"]


def test_scale_multiplies_the_fit_and_set_replaces_it():
    h = A.config_hyper(A.CONFIGS["sd_x085"], HY)
    assert h.sigma_delta == pytest.approx(0.85 * 0.9) and h.mu0 == -2.5 and h.sigma_mu == 2.5
    assert A.config_hyper(A.CONFIGS["sd_x05"], HY).sigma_delta == pytest.approx(0.45)
    assert A.config_hyper(A.CONFIGS["sd085_abs"], HY).sigma_delta == 2.025
    mp = A.config_hyper(A.CONFIGS["sd_x085_mp"], HY)
    assert mp.mu0 == A.MU0_MP_X and mp.sigma_delta == pytest.approx(0.765)
    assert A.config_hyper(A.CONFIGS["sd085_abs_mp"], HY).mu0 == A.MU0_MP_ABS == -2.489
    assert A.CONFIGS["sd_x07_noline"]["flags"] == {"line": False}
    with pytest.raises(ValueError):
        A.config_hyper({"set": {"sigma_delta": 1.0}, "scale": {"sigma_delta": 0.5}}, HY)


def test_secant_mu0():
    mu, slope = A.secant_mu0(0.42, 0.41, 0.43)
    assert slope == pytest.approx(0.2) and mu == pytest.approx(-2.45)


# --- hidden-v weights -------------------------------------------------------------------------

TABLE = """Formative feedback

Label budget: 0

subject_id      Benchmark         Number  Brier score  Calibration ECE
--------------  ----------------  ------  -----------  ---------------
subject_1  benchmark_1  50   0.300000     0.300000
subject_2  benchmark_2  44   0.090000     0.100000

Label budget: 1

subject_id      Benchmark         Number  Brier score  Calibration ECE
subject_1  benchmark_1  50   0.900000     0.100000
"""


def test_formative_v_reads_the_budget_0_block_only():
    v = A.formative_v(TABLE)
    assert v == pytest.approx([0.21, 0.08])
    assert A.v_bins(v).tolist() == [2, 0]
    assert A.v_bins([-0.01, 0.10, 0.1999, 0.30]).tolist() == [0, 1, 1, 2]


def test_hidden_shares_of_the_formative_tables():
    cnt, sh = A.hidden_shares()
    assert tuple(cnt.tolist()) == A.HV_COUNTS == (5, 12, 9)
    assert sh.sum() == pytest.approx(1.0)


def synthetic_rows(runs=12, seed=0, p=None):
    """Rows as the score stage writes them, for every config of A.CONFIGS."""
    rng = np.random.default_rng(seed)
    shift = {c: rng.normal(0, 0.01) for c in A.CONFIGS}
    rows = {}
    for i in range(runs):
        k = int(rng.integers(3, 8))
        meta = [{"bench": f"{RS.PARENTS[j % 4]}::g{j}", "parent": RS.PARENTS[j % 4],
                 "subject": f"s{int(rng.integers(0, 5))}", "eval": 50,
                 "p": round(float(rng.random() if p is None else p[(i + j) % len(p)]), 5),
                 "logit": round(float(rng.normal(-1, 2)), 4)} for j in range(k)]
        base = rng.uniform(0.05, 0.3, (k, 6))
        res = {}
        for c in A.CONFIGS:
            b = np.clip(base + shift[c] + rng.normal(0, 0.01, (k, 6)), 0, 1)
            res[c] = {"b": b.tolist(), "ece": rng.uniform(0, 0.2, (k, 6)).tolist(),
                      "q0": rng.uniform(0, 1, k).tolist(), "calls": 100, "mean_s": 0.01,
                      "max_s": 0.05, "failures": 0, "unconverged": 0, "secs": 1.0}
        rows[i] = {"regime": "TUNED", "i": i, "meta": meta, "res": res}
    return rows


def test_equal_weights_are_p1as_regime_rows():
    rows = synthetic_rows()
    with A.rs_configs():
        G = RS.RegimeRows("READING", rows)
        for x in ("sd_x085", "noline", "legacy"):
            ex, got = G.diff(x, A.SHIP), A.weighted_diff(G, x, A.SHIP, G.w)
            for k in ("D", "cluster_se", "strat_se", "U95", "PL"):
                assert got[k] == pytest.approx(ex[k], abs=1e-12)
            for q in ex["Pq"]:
                assert got["Pq"][q] == pytest.approx(ex["Pq"][q], abs=1e-12)
            assert sum(b["alc_part"] for b in got["budgets"]) == pytest.approx(got["D"], abs=1e-12)
    assert RS.CONFIGS is A.P1A_CONFIGS


def test_hidden_v_weights():
    # every v bin's share in the replica equals the formative one: the weights are the equal ones
    p = [0.05] * 5 + [0.15] * 12 + [0.5] * 9          # v 0.0475, 0.1275, 0.25
    rows = synthetic_rows(runs=26, p=p)
    with A.rs_configs():
        G = RS.RegimeRows("TUNED", rows)
        w, info = A.hv_weights(G, np.array([0.2, 0.3, 0.5]))
        b = A.v_bins([q["p"] * (1 - q["p"]) for m in G.metas for q in m])
        share = np.bincount(b, minlength=3) / len(b)
        assert np.allclose(w, G.w * np.array([0.2, 0.3, 0.5])[b] / share[b])
        w_same, _ = A.hv_weights(G, share)
        assert np.allclose(w_same, G.w)
        assert A.weighted_diff(G, "sd_x07", A.SHIP, w_same)["D"] == pytest.approx(
            G.diff("sd_x07", A.SHIP)["D"], abs=1e-12)
        # a bin the replica does not reach gets no weight, and is recorded
        rows2 = synthetic_rows(runs=6, p=[0.5])
        G2 = RS.RegimeRows("TUNED", rows2)
        w2, info2 = A.hv_weights(G2, np.array([0.2, 0.3, 0.5]))
        assert info2["bins_without_appearances"] == [0, 1] and np.allclose(w2, G2.w * 0.5)


# --- dry run and summarise ---------------------------------------------------------------------

def test_dry_run_must_write_outside_the_repository(monkeypatch):
    for k in ("DRY", "OUT", "ROWS"):
        monkeypatch.setattr(A, k, getattr(A, k))
    with pytest.raises(SystemExit):
        A.set_dry_run(os.path.join(A.ROOT, "scratch"), 2)
    assert A.DRY is None


def test_summarise_end_to_end(tmp_path, monkeypatch):
    for k in ("DRY", "OUT", "ROWS"):
        monkeypatch.setattr(A, k, getattr(A, k))
    monkeypatch.setattr(RS, "BOOTS", 200)
    A.set_dry_run(str(tmp_path), 3)
    rng = np.random.default_rng(1)
    comps = {}
    for R in A.REGIMES:
        comps[R] = {}
        os.makedirs(os.path.join(A.ROWS, R), exist_ok=True)
        for i in range(3):
            k = 5
            meta = [{"bench": f"{RS.PARENTS[j % 4]}::g{j}", "parent": RS.PARENTS[j % 4],
                     "subject": f"s{int(rng.integers(0, 6))}", "eval": 50,
                     "p": round(float(rng.random()), 5), "logit": round(float(rng.normal(-1, 2)), 4)}
                    for j in range(k)]
            base = rng.uniform(0.05, 0.3, (k, 6))
            res = {}
            for c in A.CONFIGS:
                d = -0.004 if c == "sd_x085" else (0.0415 if c == "legacy" else 0.001 * (c != A.SHIP))
                res[c] = {"b": (base + d).tolist(), "ece": np.full((k, 6), 0.05).tolist(), "q0": [0.4] * k,
                          "calls": 50, "mean_s": 0.01, "max_s": 0.1, "failures": 0, "unconverged": 0,
                          "secs": 0.5}
            row = {"regime": R, "i": i, "seed": A.DRY_SEED, "meta": meta, "res": res, "task_s": 1.0,
                   "rss_gb": 1.0, "lib": {"paiec/hier.py": "x"}, "script": "y"}
            comps[R][str(i)] = meta
            with open(A.row_path(R, i), "w") as f:
                json.dump(row, f)
    with open(os.path.join(A.ROWS, A.COMPOSITIONS), "w") as f:
        json.dump({"seed": A.DRY_SEED, "runs": comps}, f)
    ok = {"ok": True}
    with open(A.OUT, "w") as f:
        json.dump({"lock": ok, "calib": ok, "regimes": dict(ok, realisation_ok=True), "reproduce": ok}, f)
    A.stage_summarise()
    with open(A.OUT) as f:
        out = json.load(f)
    s, rule = out["summary"], out["rule"]
    assert s["complete"] and all(s["checks"].values()), s["checks"]
    v = s["regimes"]["READING"]["vs_ship"]["sd_x085"]
    assert v["equal"]["D"] == pytest.approx(-0.004, abs=1e-12)
    assert v["hidden_v"]["D"] == pytest.approx(-0.004, abs=1e-12)        # a constant shift
    assert s["regimes"]["READING"]["vs_ship"]["wide35"]["seed11"]["seed"] == RS.SCORE_SEED
    assert s["regimes"]["READING"]["vs_ship"]["sd_x07"]["seed11"] is None
    assert s["tuned_consistency"]["ship_minus_legacy"] == pytest.approx(-0.0415, abs=1e-12)
    assert rule["applied"] and rule["would_pass"] == ["sd_x085"] and rule["shipped"] == []
    assert rule["s4_reading"] == A.S4_READING
    assert len(s["table"]["rows"]) == 2 * (len(A.CONFIGS) - 1)
    assert RS.CONFIGS is A.P1A_CONFIGS
