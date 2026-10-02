"""experiments/gate_and_ci.py's arithmetic and tools/export_rows.py, on synthetic
inputs; and, where results/gate_and_ci.json exists, that it agrees with the
stored results it was read from. Needs no data/."""
import importlib.util
import json
import math
import os

import numpy as np
import pytest

from experiments import gate_and_ci as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _export_rows():
    spec = importlib.util.spec_from_file_location("export_rows", os.path.join(ROOT, "tools", "export_rows.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- statistics -------------------------------------------------------------------------------

def test_jeffreys_interval():
    lo, hi = G.jeffreys(6, 8)
    assert lo == pytest.approx(0.4084, abs=1e-3) and hi == pytest.approx(0.9440, abs=1e-3)
    assert G.jeffreys(0, 8)[0] == 0.0 and G.jeffreys(8, 8)[1] == 1.0
    assert G.jeffreys(4, 8)[0] < 0.5 < G.jeffreys(4, 8)[1]


def test_pass_probabilities():
    assert G.normal_pass(-0.002, 0.001, -0.002) == pytest.approx(0.5)
    assert G.t_predictive_pass(-0.002, 0.001, 8, -0.002) == pytest.approx(0.5)
    # a draw sd of 0 is a step; the predictive is wider than the plug-in normal
    assert G.normal_pass(-0.003, 0.0, -0.002) == 1.0
    p, pt = G.normal_pass(-0.00255, 0.00068, -0.002), G.t_predictive_pass(-0.00255, 0.00068, 8, -0.002)
    assert 0.5 < pt < p < 1


def test_spearman_brown_and_fold_overlap():
    assert G.spearman_brown(0.5, 2) == pytest.approx(2 / 3)
    assert G.spearman_brown(0.5, 1) == pytest.approx(0.5)
    rel_h, rel_all = G.rel_from_fold_overlap(0.95, 5)
    assert rel_h == pytest.approx(0.8)
    assert rel_all == pytest.approx(1 / (1 + 0.25 * 4 / 5))


def test_fold_overlap_recovers_reliability():
    """Items with a true value plus independent per-subject error: the k
    leave-one-fold-out means' mutual correlation gives the reliability of one of
    them, as rel_from_fold_overlap says."""
    rng = np.random.default_rng(0)
    n_items, n_subj, sd_e, k = 4000, 40, 4.0, 5
    d = rng.standard_normal(n_items)
    E = rng.standard_normal((n_subj, n_items)) * sd_e
    fold = np.arange(n_subj) % k
    est = np.array([d + E[fold != f].mean(0) for f in range(k)])
    c = np.mean([np.corrcoef(est[a], est[b])[0, 1] for a in range(k) for b in range(a + 1, k)])
    rel_h, rel_all = G.rel_from_fold_overlap(c, k)
    true_h = 1 / (1 + sd_e ** 2 / (n_subj * (k - 1) / k))
    true_all = 1 / (1 + sd_e ** 2 / n_subj)
    assert rel_h == pytest.approx(true_h, abs=0.02)
    assert rel_all == pytest.approx(true_all, abs=0.02)


def test_weighted_pearson_and_bootstrap_weights():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    z = 0.5 * x + rng.standard_normal(200)
    assert G.wpearson(np.ones(200), x, z)[0] == pytest.approx(np.corrcoef(x, z)[0, 1])
    W = G.boot_weights(200, None, boots=50, seed=0)
    assert W.shape == (50, 200) and np.all(W.sum(1) == 200)
    groups = np.repeat(np.arange(20), 10).astype(str)
    Wg = G.boot_weights(200, groups, boots=50, seed=0)
    assert np.all(Wg.sum(1) == 200)
    for g in range(20):             # an item carries its group's count
        assert np.all(Wg[:, groups == str(g)] == Wg[:, [np.flatnonzero(groups == str(g))[0]]])
    # duplicated items under integer weights are the weighted correlation
    w = rng.integers(0, 3, 200).astype(float)
    rep = np.repeat(np.arange(200), w.astype(int))
    assert G.wpearson(w, x, z)[0] == pytest.approx(np.corrcoef(x[rep], z[rep])[0, 1])


def test_corr_with_ci_and_fold_r():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(300)
    z = 0.6 * x + 0.8 * rng.standard_normal(300)
    g = np.repeat(np.arange(30), 10).astype(str)
    c = G.corr_with_ci(x, z, g, boots=400)
    assert c["n"] == 300 and c["groups"] == 30
    for k in ("ci_group", "ci_item", "ci_fisher"):
        assert c[k][0] < c["est"] < c[k][1]
    assert G.corr_with_ci(x, z, g, boots=400) == c          # seeded
    assert G.corr_with_ci(x, z, np.full(300, "a"), boots=200)["ci_group"] is None     # one group
    Z = np.vstack([z, z + 0.1 * rng.standard_normal(300)])
    has = np.ones_like(Z, bool)
    has[1, :10] = False
    r = G.fold_r(np.ones(300), x, Z, has)[0]
    want = np.mean([np.corrcoef(x, Z[0])[0, 1], np.corrcoef(x[10:], Z[1, 10:])[0, 1]])
    assert r == pytest.approx(want)


def test_cluster_mean_ci():
    v = np.array([0.1, 0.2, 0.3, 0.4] * 25)
    cl = np.repeat(np.arange(25), 4)
    m, ci = G.cluster_mean_ci(v, cl, boots=500)
    assert m == pytest.approx(0.25) and ci[0] <= 0.25 <= ci[1]
    m, ci = G.cluster_mean_ci(np.arange(100.0), np.arange(100) // 10, boots=500)
    assert ci[0] < m < ci[1] and ci[1] - ci[0] > 10


def test_n_for_power():
    n = G.n_for_power(0.1, 0.3)
    assert n == pytest.approx((2.8016 / (math.atanh(0.3) - math.atanh(0.1))) ** 2 + 3, rel=1e-3)
    assert G.n_for_power(0.3, 0.3) == float("inf")


# --- the gate's draws -------------------------------------------------------------------------

GATE = {"tl": -0.002, "parent": 0.002, "guard": 0.001, "folds_on": 3}


def test_draw_conditions_infers_the_parent_condition():
    ok = G.draw_conditions(-0.003, -0.004, -0.004, -0.005, 4, True, GATE)
    assert ok["worst_parent_inferred"] is True and ok["pass"]
    parent = G.draw_conditions(-0.0023, -0.004, -0.004, -0.005, 4, False, GATE)
    assert parent["worst_parent_inferred"] is False and parent["tl"]
    bar = G.draw_conditions(-0.0018, -0.004, -0.004, -0.005, 4, False, GATE)
    assert bar["worst_parent_inferred"] is None and not bar["tl"]
    assert bar["mix_reading_known"]                 # mix/whole clears the bar where test-like does not
    off = G.draw_conditions(-0.003, -0.004, -0.004, -0.005, 2, False, GATE)
    assert not off["selection_on"] and off["worst_parent_inferred"] is None
    sign = G.draw_conditions(-0.003, 0.001, -0.004, -0.005, 4, False, GATE)
    assert not sign["mix_same_sign"]


def _line(tl, mix, passes, folds=None):
    n = len(tl)
    reg = lambda v: {"cluster_se": 0.0003, "worst_parent": max(v)}  # noqa: E731
    return {"replicates": {"tl": tl, "mix": mix, "r1b": [-0.005] * n, "r1p": [-0.006] * n},
            "folds_on_reps": folds or [4] * n, "pass_reps": passes,
            "regimes": {"tl": reg(tl), "mix": reg(mix)}}


def test_gate_cell_counts():
    tl = [-0.0018, -0.0031, -0.0025, -0.0025, -0.0020, -0.0023, -0.0039, -0.0023]
    mix = [x * 1.8 for x in tl]
    passes = [False, True, True, True, True, False, True, True]
    c = G.gate_cell(_line(tl, mix, passes), {"r": 0.3, "r_within_pair_tl": 0.247}, GATE)
    t = c["tl"]
    assert t["pass_count"] == 6 and t["draws_clearing_bar"] == 7
    assert t["failed"]["tl"] == 1 and t["failed"]["worst_parent_inferred"] == 1
    assert t["mean"] == pytest.approx(np.mean(tl), abs=1e-6)
    assert t["draw_sd"] == pytest.approx(np.std(tl, ddof=1), abs=1e-6)
    assert t["p_full_gate_model"] == pytest.approx(t["p_normal"] * 6 / 7, abs=1e-3)
    assert c["mix"]["pass_count_upper"] == 8


def test_interp_r():
    cells = [{"r": 0.2, "tl": {"mean": -0.001, "draw_sd": 0.0005}},
             {"r": 0.4, "tl": {"mean": -0.003, "draw_sd": 0.0005}}]
    r = G.interp_r(cells, "tl", 0.5, GATE)
    assert r == pytest.approx(0.3, abs=0.006)
    assert G.interp_r(cells, "tl", 0.999999, GATE) is None


# --- the stored result against its sources -----------------------------------------------------

def _stored():
    p = os.path.join(ROOT, "results", "gate_and_ci.json")
    if not os.path.exists(p):
        pytest.skip("results/gate_and_ci.json not written yet")
    with open(p) as f:
        return json.load(f)


def test_stored_gate_matches_harness_table():
    st = _stored()
    if "gate" not in st:
        pytest.skip("no gate stage")
    with open(os.path.join(ROOT, "results", "harness_thresholds.json")) as f:
        thr = json.load(f)["thresholds"]["honest"]
    for line in G.LINES:
        for key, c in st["gate"]["lines"][line]["cells"].items():
            assert c["tl"]["stored_pass_draws"] == thr[line][key]["pass_draws"]
            assert c["tl"]["mean"] == pytest.approx(thr[line][key]["tl"], abs=2e-6)
    assert all(ch["holds"] for ch in st["gate"]["draft_checks"])
    assert st["gate"]["b1"]["holds"]


def test_stored_scale_reproduces_stored_numbers():
    st = _stored()
    fam = (st.get("scale") or {}).get("families") or {}
    if not fam:
        pytest.skip("no scale stage")
    for name, F in fam.items():
        for cov, v in (F.get("covariates") or {}).items():
            w = v.get("within_pair") or {}
            if "stored" in w:
                assert w["matches_stored"], f"{cov}: within-pair r {w['r']} against stored {w['stored']}"
        for k, d in (F.get("x_digests") or {}).items():
            assert d["match"], f"{name} {k}: x digest differs from the stored one"
        for ch in F.get("checks") or []:
            assert ch["ok"], ch
    if "tfidf" in fam:
        assert all(c["reproduces_draft"] for c in fam["tfidf"]["quoted"]["naive_target_all_items"].values())
    if "reliability" in fam:
        for v in fam["reliability"]["split_half"].values():
            assert v["targets_json_refit_max_abs_diff"] == 0.0
            assert 0 < v["rel_honest"] <= v["rel_all"] <= 1


def test_unit_ratios():
    fam = {"f": {"covariates": {
        "a": {"within_pair": {"r": 0.12}, "parent_scale": {"mean_over_parents": {
            "r_honest": 0.2, "ci_stratified_group": [0.1, 0.3], "parents": 4}}},
        "b": {"within_pair": {"r": -0.06, "r_oriented": 0.06}, "parent_scale": {"mean_over_parents": {
            "r_honest": 0.03, "ci_stratified_group": [-0.05, 0.1], "parents": 4}}},
        "c": {"within_pair": {"r": -0.1}, "parent_scale": {"mean_over_parents": {
            "r_honest": -0.1, "ci_stratified_group": [-0.2, -0.01], "parents": 4}}},
        "d": {"parent_scale": {"mean_over_parents": {"r_honest": 0.3}}}}}}
    gate = {"lines": {"transferred nested": {"cells": {"r=0": {"r": 0.0, "r_within_pair_tl": -0.002},
                                                       "r=0.3": {"r": 0.3, "r_within_pair_tl": 0.246}}}}}
    u = G.unit_ratios(fam, gate)
    assert set(u["covariates"]) == {"a", "b", "c"}            # no within-pair r: left out
    assert u["covariates"]["a"]["ratio"] == pytest.approx(0.6)
    assert u["covariates"]["b"]["ratio"] == pytest.approx(2.0) and not u["covariates"]["b"]["clear"]
    assert u["clear"]["range"] == [pytest.approx(0.6), pytest.approx(1.0)] and u["not_clear"] == ["b"]
    assert u["oracle"]["by_r"] == {"r=0.3": pytest.approx(0.82)}   # r = 0 is skipped
    assert u["named"]["covariates"] == {}


def test_stored_unit_ratio():
    st = _stored()
    u = (st.get("ci") or {}).get("unit_ratio")
    if not u:
        pytest.skip("no unit_ratio in the ci stage")
    fam = st["scale"]["families"]
    for name, v in u["covariates"].items():
        f = fam[v["family"]]["covariates"][name]
        w = f["within_pair"]
        assert v["ratio"] == pytest.approx(w.get("r_oriented", w["r"]) /
                                           f["parent_scale"]["mean_over_parents"]["r_honest"], abs=1e-4)
    assert set(u["named"]["covariates"]) == set(G.RATIO_NAMED)


def test_stored_single_matches_ship_confirm():
    st = _stored()
    if "single" not in st:
        pytest.skip("no single stage")
    with open(os.path.join(ROOT, "results", "ship_confirm.json")) as f:
        sc = json.load(f)
    for reg, span in (("r1b", "0-149"), ("r1p", "0-99")):
        for c, cc in (("legacy", "legacy"), ("smoothed", "smoothed")):
            want = sc["regimes"][reg]["vs"][cc][span]["per_parent"]["swe_rebench"]["mean"]
            got = st["single"]["seed0_run2_library"][reg]["vs"][c]["mean_weighted"]
            assert got == pytest.approx(want, abs=2e-6)


# --- tools/export_rows.py ----------------------------------------------------------------------

def _fake_data(root):
    d = root / "subject_side_rows"
    d.mkdir(parents=True)
    (d / "r1b.jsonl").write_text('{"i": 0}\n{"i": 1}\n')
    (d / "tl.jsonl").write_text('{"i": 0}\n')
    (d / ".DS_Store").write_bytes(b"junk")
    h = root / "hier_floor" / "timing"
    h.mkdir(parents=True)
    (h / "a.json").write_text("{}")
    return root


def test_export_verify_restore(tmp_path):
    ER = _export_rows()
    data = _fake_data(tmp_path / "data")
    out = tmp_path / "rel"
    man = ER.export(str(out), ["subject_side_rows", "hier_floor", "harness_rows"], str(data))
    assert set(man["sets"]) == {"subject_side_rows", "hier_floor"}
    assert "harness_rows" in man["missing"]
    s = man["sets"]["subject_side_rows"]
    assert s["files"] == 2 and [m["path"] for m in s["members"]] == ["subject_side_rows/r1b.jsonl",
                                                                       "subject_side_rows/tl.jsonl"]
    assert ER.verify(str(out)) == []
    sums = (out / "SHA256SUMS").read_text().split()
    assert s["archive_sha256"] in sums
    # deterministic: the same rows give the same archive bytes
    out2 = tmp_path / "rel2"
    man2 = ER.export(str(out2), ["subject_side_rows", "hier_floor"], str(data))
    assert man2["sets"]["subject_side_rows"]["archive_sha256"] == s["archive_sha256"]
    # restore reproduces the bytes, and refuses to overwrite other bytes without --force
    dest = tmp_path / "restored"
    assert ER.restore(str(out), str(dest)) == []
    assert (dest / "subject_side_rows" / "r1b.jsonl").read_text() == '{"i": 0}\n{"i": 1}\n'
    assert (dest / "hier_floor" / "timing" / "a.json").read_text() == "{}"
    (dest / "subject_side_rows" / "tl.jsonl").write_text("changed")
    bad = ER.restore(str(out), str(dest))
    assert bad and "exists with other bytes" in bad[0]
    assert ER.restore(str(out), str(dest), force=True) == []
    assert (dest / "subject_side_rows" / "tl.jsonl").read_text() == '{"i": 0}\n'
    # a tampered archive fails verification, and restore refuses it
    arc = out / s["archive"]
    arc.write_bytes(arc.read_bytes()[:-8] + b"\0" * 8)
    assert ER.verify(str(out))
    assert ER.restore(str(out), str(tmp_path / "again"))


def test_export_never_reads_outside_its_sets(tmp_path):
    ER = _export_rows()
    data = _fake_data(tmp_path / "data")
    (data / "matharena").mkdir()
    (data / "matharena" / "response.parquet").write_bytes(b"raw")
    man = ER.export(str(tmp_path / "rel"), list(ER.SETS), str(data))
    paths = [m["path"] for s in man["sets"].values() for m in s["members"]]
    assert not any(p.startswith("matharena") or p.endswith(".parquet") and "strong_llm_eval" not in p
                   for p in paths)
    assert set(ER.SETS) - set(man["sets"]) == set(man["missing"])
