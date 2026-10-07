"""experiments/posthoc_family.py: the post-hoc levers on synthetic appearances, the
budget switch on synthetic regime rows, the scratch comparison, and the stored
results.

On small synthetic harness-like appearances (four regimes, three parents): every
lever at its identity parameters returns hier's prediction; a leave-one-parent-
out fit never sees the held-out parent (changing that parent's predictions
leaves its own fold's parameters unchanged and moves the others'); the run-level
shift reads other pairs' labels only; a temperature planted in every parent is
recovered and gains; zero deltas have zero SE. On synthetic regime rows the
budget switch picks a configuration that is better at one budget everywhere.
The parser of the probes' printed tables reads every line. If results/
posthoc_family.json exists, every required check must have passed, its family
table must restate the blocks, and no anonymous id may appear.
"""
import json
import os

import numpy as np
import pytest

from experiments import posthoc_family as P

PARENTS = ("pa", "pb", "pc")


def _apps(seed=0, n_runs=6, m=24, shrink=1.0, offset_sd=0.8):
    """Harness-like appearances: item logits z ~ N(mu_pair, 1.2), labels ~ sigmoid(z); hier's
    prediction is sigmoid(z / shrink + o) at B0, with a pair offset o ~ N(0, offset_sd)
    (shrink > 1: under-confident), and sigmoid(z / shrink - 0.3 o) from B1 on: the updates
    remove the offset with a 30% overshoot, so the damping that fits is near 1 / 1.3."""
    rng = np.random.default_rng(seed)
    apps = []
    for reg in P.REGS:
        for run in range(n_runs):
            n = 3
            for j in range(n):
                mu = rng.normal(-0.5, 1.0)
                z = rng.normal(mu, 1.2, m)
                K = (rng.random(m) < P.sig(z)).astype(float)
                o = rng.normal(0, offset_sd) if offset_sd else 0.0
                Pk = np.stack([P.sig(z / shrink + (o if bi == 0 else -0.3 * o)) for bi in range(6)])
                y = (rng.random(31) < P.sig(mu)).astype(float)
                apps.append(dict(reg=reg, run=run, n=n, j=j, parent=PARENTS[(run + j) % 3],
                                 bench=f"b{j}", sid=f"s{int(rng.integers(5))}", K=K, N=np.ones(m), P=Pk, y=y))
    return apps


def _sh():
    return np.array([0.2, 0.45, 0.35])


def test_identity_parameters_return_the_prediction():
    apps = P.fresh(_apps())
    P.pp_features(apps)
    for bi in (0, 2, 5):
        S = P.pp_stack(apps, bi)
        for kind, th in (("tonly", [1.0]), ("temp", [0.0, 1.0]), ("blend", [0.0]), ("blend22", [0.0]),
                         ("runshift", [0.0])):
            assert np.allclose(P.pp_apply(kind, th, S), S["p"], atol=1e-9), kind
    U = P.u_prepare(_apps(), _sh())
    P.u2_features(U)
    for bi in (1, 3):
        S = P.u2_stack(U, bi, "w_raw")
        for kind, th in (("upd", [1.0]), ("updp", [1.0]), ("bba", [0.0, 2.0])):
            assert np.allclose(P.u_apply(kind, th, S), S["p"], atol=1e-9), kind
        for kind, th in (("pat", [1.0, 1.0]), ("spr", [1.0]), ("disp", [0.0])):
            assert np.allclose(P.u2_apply(kind, th, S), S["p"], atol=1e-9), kind
    S0 = P.b0_pack(U, "w_raw")
    c = float(np.average(P.logit(S0["pb"]), weights=S0["w"] * S0["N"]))
    for kind, th in (("bw", [1.0]), ("wi", [1.0]), ("bwa", [1.0, c])):
        assert np.allclose(P.b0_q(kind, th, S0, c), S0["p0"], atol=1e-9), kind


def _perturbed(apps, parent):
    out = []
    for a in apps:
        b = dict(a)
        if a["parent"] == parent:
            b["P"] = np.clip(a["P"] * 0.6 + 0.2, 1e-3, 1 - 1e-3)
        out.append(b)
    return out


def test_a_lopo_fit_never_sees_the_held_out_parent():
    base = _apps(seed=1)
    a1 = P.fresh(base)
    a2 = P.fresh(_perturbed(base, "pa"))
    for apps in (a1, a2):
        P.pp_features(apps)
    r1, r2 = P.pp_evaluate(a1, ["tonly"]), P.pp_evaluate(a2, ["tonly"])
    assert r1["tonly"]["params"]["pa"] == r2["tonly"]["params"]["pa"]
    assert r1["tonly"]["params"]["pb"] != r2["tonly"]["params"]["pb"]
    u1, u2 = P.u_prepare(base, _sh()), P.u_prepare(_perturbed(base, "pa"), _sh())
    _, _, p1 = P.u_run_mode(u1, "upd", "lopo", "w_raw", P.u_stack, P.u_fit, P.u_apply)
    _, _, p2 = P.u_run_mode(u2, "upd", "lopo", "w_raw", P.u_stack, P.u_fit, P.u_apply)
    assert p1["pa"] == p2["pa"] and p1["pc"] != p2["pc"]
    assert all(0.5 < th[0] < 1.0 for v in p1.values() for th in v[1:])      # interior, near 1 / 1.3
    # 'lopo' fits on TL and mix only: changing a public appearance moves no fold
    pub = [dict(a, P=np.clip(a["P"] * 0.5, 1e-3, 1)) if a["reg"] == "r1b" else a for a in base]
    u3 = P.u_prepare(pub, _sh())
    _, _, p3 = P.u_run_mode(u3, "upd", "lopo", "w_raw", P.u_stack, P.u_fit, P.u_apply)
    assert p3 == p1


def test_the_run_shift_reads_other_pairs_only():
    base = _apps(seed=2)
    a = P.fresh(base)
    P.pp_features(a)
    i = 4
    own = P.fresh(base)
    own[i]["y"] = 1.0 - own[i]["y"]
    P.pp_features(own)
    assert np.array_equal(own[i]["R"], a[i]["R"])
    other = P.fresh(base)
    j = next(k for k, b in enumerate(base) if b["run"] == base[i]["run"] and b["reg"] == base[i]["reg"] and k != i)
    other[j]["y"] = 1.0 - other[j]["y"]
    P.pp_features(other)
    assert not np.array_equal(other[i]["R"], a[i]["R"])
    assert a[i]["R"][0] == 0.0


def test_a_planted_temperature_is_recovered_and_gains():
    res = P.block_maps(_apps(seed=3, n_runs=10, m=40, shrink=2.0, offset_sd=0.0))
    b0 = [v[0][0] for v in res["summary"]["tonly"]["params"].values()]
    assert all(1.5 < b < 2.7 for b in b0), b0
    for reg in P.REGS:
        assert res["summary"]["tonly"][reg]["delta"] < 0
        assert res["calibration"][reg][0]["gain_in_sample"] < 0


def test_zero_deltas_have_zero_se():
    U = P.u_prepare(_apps(), _sh())
    s = P.u_summarise(U, np.zeros(len(U)), np.zeros((len(U), 6)), "w_hv", np.random.default_rng(0))
    for reg in P.REGS:
        assert s[reg]["delta"] == 0 and s[reg]["se"] == 0 and s[reg]["brier_by_budget"] == [0.0] * 6


def test_budget_switch_takes_a_config_that_is_better_at_one_budget_everywhere(tmp_path):
    for reg in P.RS_REGIMES:
        d = tmp_path / reg
        d.mkdir()
        for i in range(2):
            res = {}
            for c in P.RS_CONFIGS:
                b = np.full((3, 6), 0.2)
                if c == "aggr":
                    b += 0.01
                    b[:, 5] -= 0.02
                elif c != "ship":
                    b += 0.005
                res[c] = {"b": b.tolist()}
            (d / f"{i}.json").write_text(json.dumps({"res": res}))
    out = P.block_budget_switch(str(tmp_path))
    assert out["robust_switch_all"]["schedule"] == ["ship"] * 5 + ["aggr"]
    for reg in P.RS_REGIMES:
        assert out[reg]["best_per_budget"][5] == "aggr"
        assert out[reg]["in_sample_switch_gain"] == pytest.approx(-0.001)
        assert out["robust_switch_all"]["per_regime"][reg] == pytest.approx(-0.001)


def test_hidden_v_reads_the_budget_zero_section_only(tmp_path):
    paths = {}
    for r in (1, 2, 3):
        p = tmp_path / f"run{r}.txt"
        p.write_text("Formative feedback\n\nLabel budget: 0\nsubject_id benchmark_id n brier ece\n"
                     f"subject_11 benchmark_22 50 0.2{r}0000 0.1{r}0000\n"
                     "Label budget: 1\nsubject_11 benchmark_22 50 0.900000 0.900000\n")
        paths[r] = str(p)
    vs = P.hidden_v(paths)
    assert vs.tolist() == pytest.approx([0.21 - 0.11 ** 2, 0.22 - 0.12 ** 2, 0.23 - 0.13 ** 2])


def test_the_printed_tables_parse_completely():
    items = P.printed_items()
    assert len(items) == 6 * 4 * 2 + (18 + 18 + 12) * 4 * 3 + 2 * len(P.PRINTED_DAMPING) + 7
    assert len({it[0] for it in items}) == len(items)


def test_compare_tree_reads_each_blocks_precision():
    st = [i["status"] for i in P.compare_tree("maps", {"a": 1.0 + 1e-12, "b": [0.50001, "x"]},
                                              {"a": 1.0, "b": [0.5, "x"]})]
    assert st == ["match", "match_4dp", "match"]
    assert P.compare_tree("maps", {"a": 0.5002}, {"a": 0.5})[0]["status"] == "differs"
    mine = {"levers": {"k": {"score_raw": {"tl": {"delta": 0.000914, "by_vbin": {"0": 0.0012}}}}}}
    theirs = {"levers": {"k": {"score_raw": {"tl": {"delta": 0.00091, "by_vbin": {"0": 0.0012}}}}}}
    assert [i["status"] for i in P.compare_tree("update_speed", mine, theirs)] == ["match", "match"]


STORED = os.path.join(P.ROOT, "results", "posthoc_family.json")


@pytest.mark.skipif(not os.path.exists(STORED), reason="results/posthoc_family.json not written yet")
def test_stored_results_pass_and_restate_their_blocks():
    with open(STORED) as f:
        res = json.load(f)
    assert all(c["ok"] for c in res["checks"] if c["required"])
    assert res["scratch"]["printed_counts"]["differs"] == 0
    for v in (res["scratch"]["stored_json"] or {}).values():
        assert v.get("status") == "not run" or v["counts"]["differs"] == 0
    rows = res["family"]["rows"]
    lev = res["update_speed"]["levers"]["upd|lopo|fit_w_raw"]["score_raw"]
    assert rows["upd"]["delta"]["tl"] == lev["tl"]["delta"]
    assert rows["runshift"]["delta"]["tl"] == res["maps"]["summary"]["runshift"]["tl"]["delta"]
    assert rows["damp"]["delta"]["tl"] == res["damping"]["plain"]["tl_equal"]["delta_per_pair"]
    best = res["family"]["best_tl_lopo"]["delta"]
    assert all(r["delta"]["tl"] >= best for r in rows.values())
    assert not res["family"]["any_meets_tl_bar"]
    text = json.dumps(res)
    assert "subject_" not in text and "benchmark_" not in text
