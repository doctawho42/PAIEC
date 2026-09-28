"""The acceptance harness's arithmetic, experiments/harness.py, on synthetic rows
(no data needed): offsets, Brier differences, the two slopes, centring,
nested selection and the standard errors."""
import math

import numpy as np
import pytest

from experiments import harness as H

sig = lambda x: 1 / (1 + np.exp(-x))


# --- offsets and Brier ------------------------------------------------------------------

def test_zero_offset_returns_the_base_bit_for_bit():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-4, 1 - 1e-4, 1000)
    q = H.shifted(p, np.zeros_like(p))
    assert np.array_equal(q, p) and q.tobytes() == p.tobytes()
    # a round trip through the logit alone would not be exact
    off = np.where(np.arange(1000) % 2 == 0, 0.0, 0.3)
    q = H.shifted(p, off)
    assert np.array_equal(q[::2], p[::2])
    assert np.allclose(q[1::2], sig(np.log(p[1::2] / (1 - p[1::2])) + 0.3))


def test_cap_bounds_the_offset_and_is_the_identity_near_zero():
    o = np.array([-100.0, -1e-3, 0.0, 1e-3, 100.0])
    c = H.capped(o)
    assert np.all(np.abs(c) <= H.CAP) and c[2] == 0.0
    assert np.allclose(c[1:4], o[1:4], atol=1e-9)


def test_delta_brier_matches_every_response_counted():
    rng = np.random.default_rng(1)
    n, A = 60, 5
    seg = np.sort(rng.integers(0, A, n))
    N = rng.integers(1, 4, n).astype(float)
    K = np.array([rng.integers(0, k + 1) for k in N.astype(int)], float)
    p, q = rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)
    d = H.delta_brier(p, q, K, N, seg, A)
    for a in range(A):
        m = seg == a
        ys = np.concatenate([[1.0] * int(k) + [0.0] * int(t - k) for k, t in zip(K[m], N[m])])
        pp = np.repeat(p[m], N[m].astype(int))
        qq = np.repeat(q[m], N[m].astype(int))
        assert d[a] == pytest.approx(np.mean((qq - ys) ** 2) - np.mean((pp - ys) ** 2), abs=1e-12)


# --- slopes ---------------------------------------------------------------------------

def test_pair_slope_is_the_map():
    rng = np.random.default_rng(2)
    E, L = 6, 12
    u = rng.normal(0, 1, (E, L))
    c = rng.normal(0, 1.5, (E, L))
    y = (rng.uniform(size=(E, L)) < sig(u - 0.8 * c)).astype(float)
    mask = np.arange(L)[None, :] < rng.integers(2, L + 1, E)[:, None]
    m, s = 0.2, 0.7
    b = H.pair_slope(u, y, c, mask, m, s)
    grid = np.linspace(-6, 6, 120001)
    for e in range(E):
        uu, yy, cc = u[e][mask[e]], y[e][mask[e]], c[e][mask[e]]
        z = uu[None, :] + grid[:, None] * cc[None, :]
        lp = np.sum(yy * z - np.logaddexp(0, z), 1) - (grid - m) ** 2 / (2 * s * s)
        assert b[e] == pytest.approx(grid[np.argmax(lp)], abs=2e-4)


def test_pair_slope_without_labels_is_the_prior_mean():
    u = np.zeros((3, 4))
    b = H.pair_slope(u, u, np.ones((3, 4)), np.zeros((3, 4), bool), np.array([0.0, 0.5, -1.0]), 0.3)
    assert np.allclose(b, [0.0, 0.5, -1.0])


def test_fit_scalar_finds_the_minimum_and_keeps_zero_when_flat():
    f = lambda b: (b - 0.37) ** 2 - 0.37 ** 2
    assert H.fit_scalar(f, 1.0) == pytest.approx(0.37, abs=1e-6)
    assert H.fit_scalar(f, 4.0) == pytest.approx(0.37, abs=1e-6)
    assert H.fit_scalar(lambda b: 0.0, 1.0) == 0.0
    assert H.fit_scalar(lambda b: b * b, 1.0) == pytest.approx(0.0, abs=1e-9)


# --- statistics -----------------------------------------------------------------------

def test_weighted_mean_is_the_mean_of_run_means():
    rng = np.random.default_rng(3)
    run = np.repeat(np.arange(10), rng.integers(2, 9, 10))
    size = np.bincount(run)
    w = 1.0 / size[run]
    d = rng.normal(0, 1, len(run))
    run_means = [d[run == r].mean() for r in range(10)]
    assert H.weighted_mean(d, w) == pytest.approx(np.mean(run_means))
    assert H.run_se(d, w, run) == pytest.approx(np.std(run_means, ddof=1) / math.sqrt(10))


def test_bootstrap_ses():
    rng = np.random.default_rng(4)
    n = 200
    cl = rng.integers(0, 40, n).astype(str)
    st = np.array(["a", "b"])[rng.integers(0, 2, n)]
    w = np.ones(n)
    assert H.boot_se(np.full(n, 0.3), w, cl, None, 200) == pytest.approx(0.0, abs=1e-12)
    assert H.boot_se(np.full(n, 0.3), w, cl, st, 200) == pytest.approx(0.0, abs=1e-12)
    # two strata at very different means: resampling within them removes that variance
    cl2 = np.arange(n).astype(str)
    st2 = np.where(np.arange(n) < n // 2, "a", "b")
    d = np.where(st2 == "a", -1.0, 1.0) + rng.normal(0, 0.01, n)
    pooled = H.boot_se(d, w, cl2, None, 500)
    strat = H.boot_se(d, w, cl2, st2, 500)
    assert strat < pooled / 10
    assert pooled == pytest.approx(1 / math.sqrt(n), rel=0.2)


def test_select_turns_off_unless_below_zero():
    assert H.select({"a": 0.001, "b": 0.0}) is None
    assert H.select({"a": -0.001, "b": -0.003, "c": 0.002}) == "b"
    assert H.select({}) is None


def test_select_with_a_margin_of_inner_ses():
    crit = {"a": -0.001, "b": -0.003}
    se = {"a": 0.0005, "b": 0.004}
    assert H.select(crit, se, 0.0) == "b"
    assert H.select(crit, se, 1.0) == "a"            # b is within one SE of 0, a is two SEs below
    assert H.select(crit, se, 3.0) is None
    assert H.select({"a": -0.001}, {"a": 0.0}, 1.0) == "a"   # a zero SE is no margin


def test_linearised_cluster_se_matches_the_bootstrap():
    rng = np.random.default_rng(6)
    n = 3000
    cl = rng.integers(0, 150, n)
    eff = rng.normal(0, 1, 150)
    d = eff[cl] + rng.normal(0, 1, n)
    w = rng.uniform(0.2, 1.0, n)
    lin = H.lin_se(d, w, cl.astype(str))
    boot = H.boot_se(d, w, cl.astype(str), None, 4000)
    assert lin == pytest.approx(boot, rel=0.06)
    assert H.lin_se(np.full(n, 0.2), w, cl) == pytest.approx(0.0, abs=1e-12)
    assert H.lin_se(d, w, np.zeros(n)) == 0.0          # one cluster: nothing to estimate


def test_gate():
    def line(tl, mix, worst, guard, on):
        reg = {"tl": {"est": tl, "worst_parent": worst}, "mix": {"est": mix},
               "r1b": {"est": guard}, "r1p": {"est": -0.001}}
        return {"regimes": reg, "folds_on": on}
    assert H.gate(line(-0.003, -0.004, 0.001, 0.0005, 4))["pass"]
    assert not H.gate(line(-0.003, -0.004, 0.001, 0.0005, 2))["pass"]      # selection off
    assert not H.gate(line(-0.0015, -0.004, 0.001, 0.0005, 4))["pass"]     # too small
    assert not H.gate(line(-0.003, +0.001, 0.001, 0.0005, 4))["pass"]      # mix disagrees
    assert not H.gate(line(-0.003, -0.004, 0.0025, 0.0005, 4))["pass"]     # a parent loses
    assert not H.gate(line(-0.003, -0.004, 0.001, 0.0015, 4))["pass"]      # the guard
    g = H.gate(line(-0.003, -0.004, 0.001, 0.0005, None))                  # forced: no selection
    assert g["pass"] is None and g["pass_if_on"]


def test_degraded_covariate_has_its_correlation():
    rng = np.random.default_rng(5)
    z = H.standardise_within(rng.normal(3, 2, 200000), np.array(["p"] * 200000))
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std() == pytest.approx(1)
    for r in (0.0, 0.3, 0.7):
        x = H.degrade(z, r, rng.standard_normal(len(z)))
        assert np.corrcoef(x, z)[0, 1] == pytest.approx(r, abs=0.01)
        assert x.std() == pytest.approx(1, abs=0.01)


# --- end to end on synthetic rows -------------------------------------------------------

def synthetic_runs(regime, n_runs, seed, truth, pairs=4, n_eval=40, repeat=0.5):
    """Runs shaped like the collector's rows. Items have difficulty d; the base
    predicts every item at the pair's level (a level-only 'hier'), so an item
    covariate carrying d can only help. Some pairs share a benchmark."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_runs):
        slots = []
        benches = []
        for j in range(pairs):
            par = H.PARENTS[(i + j) % 4]
            if benches and rng.uniform() < repeat and benches[-1][0] == par:
                bench = benches[-1][1]
            else:
                bench = f"{regime}-b{i}-{j}"
            benches.append((par, bench))
            theta = rng.normal(-0.5, 1)
            keys = [f"{par}:{rng.integers(0, 400)}" for _ in range(n_eval + 31)]
            keys = list(dict.fromkeys(keys))
            for k in keys:
                truth.setdefault(k, rng.normal(0, 1.5))
            ev, acq = keys[:n_eval], keys[n_eval:n_eval + 31]
            N = rng.integers(1, 3, len(ev))
            K = np.array([rng.binomial(n, sig(theta - truth[k])) for k, n in zip(ev, N)])
            rate = (K.sum() + 0.5) / (N.sum() + 1)
            p_ev = np.tile(np.full(len(ev), rate), (6, 1))
            ya = np.array([rng.uniform() < sig(theta - truth[k]) for k in acq], np.int8)
            pu = np.full((6, len(acq)), rate)
            for bi, b in enumerate(H.BUDGETS):
                pu[bi, b:] = np.nan
            slots.append(dict(parent=par, kind="group", bench=bench, sid=f"s{rng.integers(0, 12)}",
                              subject={}, ev_keys=np.array(ev, object), ev_K=K, ev_N=N, ev_p=p_ev,
                              acq_keys=np.array(acq, object), acq_y=ya, acq_pu=pu))
        runs.append(dict(run=i, slots=slots, hier_failures=0))
    return runs


@pytest.fixture(scope="module")
def world():
    truth = {}
    tl = synthetic_runs("tl", 24, 0, truth)
    mix = synthetic_runs("mix", 12, 1, truth)
    keys = {}
    for runs in (tl, mix):
        for r in runs:
            for s in r["slots"]:
                for k in list(s["ev_keys"]) + list(s["acq_keys"]):
                    keys.setdefault(k, len(keys))
    rows = {"tl": H.Rows("tl", tl, keys), "mix": H.Rows("mix", mix, keys)}
    return rows, list(keys), truth


def test_centring_reads_every_label_on_the_benchmark(world):
    rows, keys, truth = world
    R = rows["tl"]
    cov = H.Covariate.from_dict(truth, keys)
    P = H.Prep(R, cov)
    x = cov.X[0]
    for a in range(R.A):
        sib = np.flatnonzero(R.grp == R.grp[a])
        for bi, b in enumerate(H.BUDGETS):
            lab = [x[R.own_k[s, j]] for s in sib for j in range(min(b, R.own_len[s]))]
            own = [x[R.own_k[a, j]] for j in range(min(b, R.own_len[a]))]
            assert P.xbar_bench[bi, a] == pytest.approx(np.mean(lab) if lab else 0.0)
            assert P.xbar_own[bi, a] == pytest.approx(np.mean(own) if own else 0.0)


def test_a_covariate_of_zeros_changes_nothing(world):
    rows, keys, _ = world
    eng = H.Engine(rows, H.Covariate.from_dict({}, keys))
    for v in H.VARIANTS:
        for c in H.configs_of(v):
            for reg in rows:
                assert not eng.delta(reg, c, H.PARENTS[:3]).any()
    for v in H.VARIANTS:
        dB, ch = H.nested(eng, v)
        assert all(not d.any() for d in dB.values())
        assert all(ch[q]["choice"] is None for q in H.PARENTS)


def test_the_oracle_gains_and_selection_switches_it_on(world):
    rows, keys, truth = world
    eng = H.Engine(rows, H.Covariate.from_dict(truth, keys))
    beta = eng.beta(H.PARENTS[:3])
    assert beta[0] == 0.0 and np.all(beta[1:] < 0)       # harder items, lower predictions
    for v in H.VARIANTS:
        dB, ch = H.nested(eng, v)
        assert H.folds_on(ch) >= 3
        d = dB["tl"] @ H.W6
        assert H.weighted_mean(d, rows["tl"].w) < -0.005
        if v == "b0":
            assert np.all(dB["tl"][:, 1:] == 0.0)        # the uncentred term acts at B0 only
            assert H.weighted_mean(dB["tl"][:, 0], rows["tl"].w) < 0
            continue
        assert np.all(dB["tl"][:, 0] == 0.0)             # nothing labeled at B0
        if v == "per-pair":
            assert np.all(dB["tl"][:, 1] == 0.0)         # one own label carries no slope
    # from B7 on: no change before B7
    d7 = eng.delta("tl", ("transferred", None, 7), H.PARENTS[1:])
    assert np.all(d7[:, :3] == 0.0) and (d7[:, 3:] != 0).any()
    lines, _ = H.score_covariate(rows, H.Covariate.from_dict(truth, keys), boots=50)
    tl = lines["transferred nested"]["regimes"]["tl"]
    assert tl["est"] < 0 and tl["by_budget"][0] == 0.0
    assert set(tl["per_parent"]) == set(H.PARENTS)
    assert lines["transferred nested"]["gate"]["tl"]


def test_per_pair_slope_reads_the_unlabeled_prediction(world):
    """The per-pair fit uses acq_pu at the checkpoint and only the own labels
    labeled by then: changing a later cell cannot move an earlier budget."""
    rows, keys, truth = world
    cov = H.Covariate.from_dict(truth, keys)
    b1 = H.Engine(rows, cov).pair_betas("tl", np.zeros(6), 0.5)
    R = rows["tl"]
    saved = R.own_pu.copy()
    try:
        R.own_pu[3, :, 7:] = 0.999                       # cells not labeled at B7
        b2 = H.Engine(rows, cov).pair_betas("tl", np.zeros(6), 0.5)
    finally:
        R.own_pu[:] = saved
    assert np.array_equal(b1[3], b2[3])


def test_a_fold_covariate_reads_every_item_in_the_target_s_fold(world):
    rows, keys, truth = world
    R = rows["tl"]
    fold = lambda sid: int(sid[1:]) % 2
    maps = [{k: v for k, v in truth.items()}, {k: -2 * v + 1 for k, v in truth.items()}]
    cov = H.Covariate.from_folds(maps, fold, keys)
    P = H.Prep(R, cov)
    for a in range(0, R.A, 5):
        x = cov.X[fold(R.sid[a])]
        sib = np.flatnonzero(R.grp == R.grp[a])
        lab = [x[R.own_k[s, j]] for s in sib for j in range(min(7, R.own_len[s]))]
        assert P.xbar_bench[3, a] == pytest.approx(np.mean(lab))
        m = R.ev_a == a
        assert np.allclose(P.x_ev[m], x[R.ev_k[m]])


def test_base_matrix_standardises_within_parent_and_skips_other_items():
    keys = ["a:1", "a:2", "a:3", "b:1", "b:2", "c:1"]
    par = {"a:1": "a", "a:2": "a", "a:3": "a", "b:1": "b", "b:2": "b"}
    maps = [{"a:1": 1.0, "a:2": 2.0, "a:3": 3.0, "b:1": 10.0, "b:2": 30.0, "c:1": 5.0},
            {"a:1": 1.0, "a:3": 3.0, "b:1": 10.0, "b:2": 30.0}]
    Z, has, in_parent = H.base_matrix(maps, keys, par)
    assert list(in_parent) == [True] * 5 + [False]
    assert list(has[0]) == [True] * 5 + [False] and list(has[1]) == [True, False, True, True, True, False]
    assert Z[0, :3] == pytest.approx([-1.2247449, 0, 1.2247449], abs=1e-6)
    assert Z[0, 3:5] == pytest.approx([-1, 1]) and Z[1, [0, 2]] == pytest.approx([-1, 1])


# --- the audit's fixes: scale, standardisation, folds on, B0, selection-aware SE -------

def parent_of_key(k):
    return k.split(":")[0]


def test_within_scale_ignores_benchmark_offsets_and_absent_parents(world):
    """A covariate with a benchmark-level offset and absent on one parent: the
    per-pair prior (Engine.sd, 'within') and so every per-pair line are the
    same whatever the offsets; the legacy pooled sd is not."""
    rows, keys, truth = world
    absent = H.PARENTS[2]
    base = {k: v for k, v in truth.items() if parent_of_key(k) != absent}
    shift = {H.PARENTS[0]: 5.0, H.PARENTS[1]: -3.0, H.PARENTS[3]: 0.7}
    moved = {k: v + shift[parent_of_key(k)] for k, v in base.items()}
    e1 = H.Engine(rows, H.Covariate.from_dict(base, keys))
    e2 = H.Engine(rows, H.Covariate.from_dict(moved, keys))
    for train in (H.PARENTS, H.PARENTS[:3], H.PARENTS[1:], (H.PARENTS[0], H.PARENTS[2])):
        assert e1.sd(train) == pytest.approx(e2.sd(train), rel=1e-12)
    # the absent parent (x constant, 0) does not shrink the unit
    assert e1.sd(H.PARENTS) == pytest.approx(e1.sd(tuple(p for p in H.PARENTS if p != absent)), rel=1e-12)
    for c in [c for c in H.configs_of("per-pair") if c[2] == 1][:3]:
        for reg in rows:
            assert np.allclose(e1.delta(reg, c, H.PARENTS[:3]), e2.delta(reg, c, H.PARENTS[:3]), atol=1e-12)
    p1 = H.Engine(rows, H.Covariate.from_dict(base, keys), scale="pooled")
    p2 = H.Engine(rows, H.Covariate.from_dict(moved, keys), scale="pooled")
    assert p2.sd(H.PARENTS) > 1.5 * p1.sd(H.PARENTS)    # the offsets inflate the legacy unit
    with pytest.raises(ValueError):
        H.Engine(rows, H.Covariate.from_dict(base, keys), scale="other")


def test_standardised_follows_itemcov_eval_rule():
    rng = np.random.default_rng(8)
    items = {"a": [f"a{i}" for i in range(50)], "b": [f"b{i}" for i in range(40)],
             "c": [f"c{i}" for i in range(30)]}
    x = {f"a{i}": 10 + 3 * rng.normal() for i in range(50)}
    x["a0"] = 1e6                                          # an outlier: clipped
    x.update({f"b{i}": float(i < 5) for i in range(40)})   # 5 items off the mode: no x on b
    x.update({f"c{i}": float(i) for i in range(20)})       # c: 20 of its 30 items carry x
    x["zz"] = 4.0                                          # on no benchmark
    z = H.standardised(x, items)
    assert not any(k.startswith("b") for k in z) and "zz" not in z
    assert set(k for k in z if k.startswith("c")) == {f"c{i}" for i in range(20)}
    za = np.array([z[f"a{i}"] for i in range(50)])
    assert za.max() == H.CLIP and np.all(np.abs(za) <= H.CLIP)
    zc = np.array([z[f"c{i}"] for i in range(20)])
    assert zc.mean() == pytest.approx(0, abs=1e-12) and zc.std() == pytest.approx(1)
    # an affine map within a benchmark changes nothing
    y = {k: (-2 * v + 7 if k.startswith("c") else v) for k, v in x.items()}
    zy = H.standardised(y, items)
    assert all(zy[f"c{i}"] == pytest.approx(-z[f"c{i}"]) for i in range(20))


def test_eval_covariate_standardises_and_keeps_raw_x_for_b0(world):
    rows, keys, truth = world
    items_bench = H.keys_by_parent(rows, keys)
    assert set(items_bench) == set(H.PARENTS)
    assert all(parent_of_key(k) == q for q, ks in items_bench.items() for k in ks)
    shift = {q: 10.0 * i for i, q in enumerate(H.PARENTS)}
    raw = {k: 3 * v + shift[parent_of_key(k)] for k, v in truth.items() if k in set(keys)}
    cov, info = H.eval_covariate(raw, items_bench, keys, "t")
    cov2, _ = H.eval_covariate({k: v - shift[parent_of_key(k)] for k, v in raw.items()}, items_bench, keys, "t")
    assert np.allclose(cov.X, cov2.X)                       # benchmark offsets cannot act
    assert info["input"].startswith("standardised")
    assert np.allclose(cov.X0[0], [raw.get(k, np.nan) for k in keys], equal_nan=True)
    craw, info_raw = H.eval_covariate(raw, items_bench, keys, "t", standardise=False)
    assert info_raw["input"] == "raw" and np.allclose(craw.X[0], [raw.get(k, 0.0) for k in keys])


def test_folds_on_counts_only_folds_that_act(world):
    """A covariate that varies on one parent only: folds whose choice lands on
    a parent where x is constant do not count, and nothing moves there."""
    rows, keys, truth = world
    only = H.PARENTS[0]
    cov = H.Covariate.from_dict({k: v for k, v in truth.items() if parent_of_key(k) == only}, keys)
    eng = H.Engine(rows, cov)
    for v in ("transferred", "per-pair"):
        dB, ch = H.nested(eng, v, k=0.0)
        assert H.folds_on(ch) <= 1
        if v == "per-pair":                                  # chosen on the inner folds, acting nowhere
            assert sum(ch[q]["choice"] is not None for q in H.PARENTS if q != only) >= 2
        for q in H.PARENTS:
            if q != only:
                assert not ch[q]["acts"]
                assert not dB["tl"][rows["tl"].parent == q].any()
    lines, _ = H.score_covariate(rows, cov, variants=("transferred",), forced=(), boots=20, per=False)
    ln = lines["transferred nested"]
    assert ln["folds_on"] <= 1 and ln["folds_chosen"] >= ln["folds_on"]
    assert ln["gate"]["selection_on"] is False


def test_b0_term_reads_x0_and_nothing_where_x0_is_missing(world):
    rows, keys, truth = world
    ks = set(keys)
    half = {k: v for i, (k, v) in enumerate(sorted(truth.items())) if k in ks and i % 2 == 0}
    cov = H.Covariate.from_dict(truth, keys, x0=half)
    eng = H.Engine(rows, cov)
    R, P = rows["tl"], eng.prep["tl"]
    assert np.isnan(P.x0_ev).any() and np.isfinite(P.x0_ev).any()
    b0 = eng.beta0(H.PARENTS[:3])
    assert b0 < 0                                            # harder items, lower predictions at B0
    d = eng.delta("tl", ("b0", None, 0), H.PARENTS[:3])
    assert np.all(d[:, 1:] == 0.0) and d[:, 0].any()
    # appearances none of whose items carry x0 are untouched
    has = np.bincount(R.ev_a, np.isfinite(P.x0_ev), R.A) > 0
    assert np.all(d[~has, 0] == 0.0)
    c0, _ = eng.x0_ref(H.PARENTS[:3])
    assert c0 == pytest.approx(np.mean([half[k] for k in np.array(keys, object)[R.ev_k][np.isfinite(P.x0_ev)
                                        & np.isin(R.parent[R.ev_a], H.PARENTS[:3])]] +
                                       [half[k] for k in np.array(keys, object)[rows["mix"].ev_k][
                                           np.isfinite(eng.prep["mix"].x0_ev)
                                           & np.isin(rows["mix"].parent[rows["mix"].ev_a], H.PARENTS[:3])]]))


def test_selection_aware_se_is_the_fixed_one_when_no_resample_changes_the_choice(world):
    rows, keys, truth = world
    eng = H.Engine(rows, H.Covariate.from_dict(truth, keys))
    dB, ch = H.nested(eng, "b0")                             # one configuration, far below 0
    R = rows["tl"]
    fixed = H.boot_se(dB["tl"] @ H.W6, R.w, R.cluster, None, 300, 0)
    assert H.selection_boot_se(eng, "b0", 300, 0) == pytest.approx(fixed, rel=1e-9)
    # a noise covariate: choices flip between resamples, and the SE stays finite
    rng = np.random.default_rng(9)
    noise = H.Engine(rows, H.Covariate.from_dict({k: rng.normal() for k in keys}, keys))
    se = H.selection_boot_se(noise, "per-pair", 300, 0)
    assert np.isfinite(se) and se >= 0


def test_score_covariate_reports_the_selection_aware_se(world):
    rows, keys, truth = world
    lines, _ = H.score_covariate(rows, H.Covariate.from_dict(truth, keys), variants=("transferred", "b0"),
                                 forced=(("b0", None, 0),), boots=40)
    for v in ("transferred", "b0"):
        tl = lines[f"{v} nested"]["regimes"]["tl"]
        assert tl["sel_cluster_se"] >= 0
        assert lines[f"{v} nested"]["folds_on_no_margin"] >= lines[f"{v} nested"]["folds_chosen"]
    assert "sel_cluster_se" not in lines["b0 from B0 (forced)"]["regimes"]["tl"]
    assert lines["b0 from B0 (forced)"]["regimes"]["tl"]["by_budget"][0] < 0


def test_thresholds_report_the_pass_rate_over_draws():
    def cell(est, passes, r):
        reg = {"tl": {"est": est, "cluster_se": 0.0004, "worst_parent": est / 2}, "mix": {"est": 2 * est}}
        return {"meta": {"r": r, "r_within_pair_tl": r * 0.8},
                "lines": {"transferred nested": {"regimes": reg, "folds_on": 4.0, "pass_reps": passes,
                                                 "draw_sd": {"tl": 0.0005},
                                                 "gate": {"pass": est <= -0.002, "pass_if_on": est <= -0.002}}}}
    table = {"r=0.2": cell(-0.001, [False] * 5, 0.2), "r=0.3": cell(-0.0024, [True] * 4 + [False], 0.3),
             "r=0.4": cell(-0.004, [True] * 5, 0.4)}
    thr = H.thresholds(table)["transferred nested"]
    assert thr["r=0.3"]["pass_draws"] == "4/5" and thr["r=0.2"]["pass_draws"] == "0/5"
    assert thr["smallest_r_passing"] == 0.3 and thr["smallest_r_majority_of_draws"] == 0.3
    assert thr["smallest_r_every_draw"] == 0.4
    assert thr["r=0.3"]["r_within_pair_tl"] == pytest.approx(0.24)
