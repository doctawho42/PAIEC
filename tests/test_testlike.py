"""The test-like run sampler, paiec/testlike.py, on synthetic pairs."""
import math
from collections import Counter

import numpy as np
import pytest

from paiec import baselines as B
from paiec import data as D
from paiec import official as O
from paiec import testlike as T
from paiec.evaluator import MIN_ITEMS, Pair, Response

sig = lambda x: 1 / (1 + np.exp(-x))
KEYS = {"alpha": "grp", "beta": "grp"}
DATES = ["2024-06-01", "2025-02", "2025-05-10", "2025-08-01", "2025-11-20", "2026-02-01",
         "2026-04-01", ""]


def benchmark(name, n_subjects, n_groups=6, per_group=50, level=0.0, seed=0, repeats=1):
    """Pairs of one benchmark: groups of items whose difficulty rises with the
    group index, subjects with spread abilities and staggered release dates."""
    rng = np.random.default_rng(seed)
    out = []
    diff = {}
    for g in range(n_groups):
        for i in range(per_group):
            diff[f"{name}:{g}:{i}"] = 1.2 * (g - n_groups / 2) + rng.normal(0, 1.0)
    for s in range(n_subjects):
        theta = level + rng.normal(0, 1.0)
        sid = f"{name}-s{s}"
        subject = {"normalized_name": sid, "provider": "acme",
                   "release_date": DATES[s % len(DATES)], "access_date": "2026-05-01",
                   "harness": "", "harness_version": "", "reasoning_effort": "",
                   "subject_features_extra": ""}
        responses = []
        for key, d in diff.items():
            g = key.split(":")[1]
            item = {"item_content": f"question {key}", "item_features": f"grp=g{g};n=1",
                    "interactors": "", "benchmark_id": name}
            for _ in range(repeats):
                responses.append(Response(key, item, int(rng.random() < sig(theta - d))))
        out.append(Pair(subject, sid, name, responses))
    return out


@pytest.fixture(scope="module")
def pairs():
    return (benchmark("alpha", 16, level=-1.5, seed=1)
            + benchmark("beta", 16, level=0.5, seed=2, repeats=2)
            + benchmark("gamma", 12, n_groups=5, per_group=60, level=-0.5, seed=3)
            + benchmark("solo", 1, n_groups=1, per_group=400, seed=4))


@pytest.fixture(scope="module")
def cat(pairs):
    return T.build_catalogue(pairs, keys=KEYS, min_subjects=3, strata=(2, 3), folds=4)


def by_name(cat):
    return {b.name: b for b in cat.pseudos}


# --- the catalogue -------------------------------------------------------------------

def test_every_pseudo_pair_is_a_real_pair_restricted(pairs, cat):
    source = {(p.subject_id, p.benchmark_id): p for p in pairs}
    for b in cat.pseudos:
        assert T.parent_of(b.name) == b.parent and b.name.startswith(b.parent + T.SEP)
        for p in b.pairs:
            assert p.benchmark_id == b.name
            assert len(p.item_keys) >= MIN_ITEMS
            orig = source[(p.subject_id, b.parent)]
            assert p.subject is orig.subject
            ids = {id(r) for r in orig.responses}
            assert all(id(r) in ids for r in p.responses)          # the same responses, labels untouched
            own = set(p.item_keys)
            assert [r for r in orig.responses if r.item_key in own] == p.responses


def test_kinds_and_names(cat):
    kinds = Counter((b.parent, b.kind) for b in cat.pseudos)
    assert kinds[("alpha", "group")] >= 2 and kinds[("alpha", "stratum")] >= 2
    assert kinds[("alpha", "whole")] == 1
    assert kinds[("alpha", "mix")] >= 2
    assert kinds[("gamma", "group")] == 0 and kinds[("gamma", "mix")] == 0   # no key for gamma
    assert kinds[("gamma", "stratum")] >= 2
    assert set(k for (par, k) in kinds if par == "solo") == {"chunk"}
    names = [b.name for b in cat.pseudos]
    assert len(set(names)) == len(names)
    assert len({D.anon_id("benchmark", n) for n in names}) == len(names)


def test_group_segments_partition_the_groups_hardest_first(pairs, cat):
    segs = [b for b in cat.pseudos if b.parent == "alpha" and b.kind == "group"]
    groups = [g for b in segs for g in b.groups]
    assert sorted(groups) == [f"g{g}" for g in range(6)]        # disjoint and complete
    for a, c in zip(segs, segs[1:]):
        assert not (a.items & c.items)
    # the benchmark's groups get harder with their index, so the first segment
    # (hardest first) holds the highest-index group
    assert "g5" in segs[0].groups and "g0" in segs[-1].groups
    rate = lambda b: np.mean([r.label for p in b.pairs for r in p.responses])
    assert rate(segs[0]) < rate(segs[-1])


def test_mixed_segments_partition_the_groups_and_keep_the_key(cat):
    """'mix' merges the same groups in a seeded random order: every group once,
    and more of the item variance between groups than difficulty-sorted merges."""
    for par in ("alpha", "beta"):
        mix = [b for b in cat.pseudos if b.parent == par and b.kind == "mix"]
        grp = [b for b in cat.pseudos if b.parent == par and b.kind == "group"]
        assert sorted(g for b in mix for g in b.groups) == [f"g{g}" for g in range(6)]
        for a, c in zip(mix, mix[1:]):
            assert not (a.items & c.items)
        share = lambda bs: np.mean([T.key_share(b.items, cat.difficulty[par],
                                                cat.features[par])["key_share"] for b in bs])
        assert share(mix) > share(grp)


def test_catalogue_keeps_parent_difficulty_and_features(pairs, cat):
    assert set(cat.difficulty) == {"alpha", "beta", "gamma"}          # not the one-subject 'solo'
    assert set(cat.features) == {"alpha", "beta"}
    alpha = {k for p in pairs if p.benchmark_id == "alpha" for k in p.item_keys}
    assert set(cat.difficulty["alpha"]) == alpha
    assert set(cat.features["alpha"].values()) == {f"g{g}" for g in range(6)}
    # harder groups (higher index) get higher Rasch difficulty
    b = cat.difficulty["alpha"]
    mean = lambda g: np.mean([b[k] for k, v in cat.features["alpha"].items() if v == g])
    assert mean("g5") > mean("g0")


def test_segments_close_at_target_items_and_min_subjects(cat):
    for b in cat.pseudos:
        if b.kind in ("group", "mix"):
            assert sum(len(p.item_keys) >= 90 for p in b.pairs) >= 3


def test_strata_are_ordered_by_difficulty(cat):
    n = by_name(cat)
    rate = lambda b: np.mean([r.label for p in b.pairs for r in p.responses])
    for par in ("alpha", "beta", "gamma"):
        assert rate(n[f"{par}::q1of2"]) < rate(n[f"{par}::q2of2"])
        assert rate(n[f"{par}::q1of3"]) < rate(n[f"{par}::q2of3"]) < rate(n[f"{par}::q3of3"])


def test_stratum_membership_ignores_the_subjects_own_labels(pairs):
    """Cross-fitting: flipping every label of one subject leaves which of its
    items fall in each stratum unchanged."""
    beta = [p for p in pairs if p.benchmark_id == "beta"]
    target = beta[3]
    flipped = Pair(target.subject, target.subject_id, "beta",
                   [Response(r.item_key, r.item, 1 - r.label) for r in target.responses])
    other = [flipped if p is target else p for p in beta]
    a = T.build_catalogue(beta, keys={}, strata=(2, 3), folds=4, wholes=False)
    b = T.build_catalogue(other, keys={}, strata=(2, 3), folds=4, wholes=False)

    def items_of(c):
        return {x.name: set(p.item_keys) for x in c.pseudos for p in x.pairs
                if p.subject_id == target.subject_id}
    assert items_of(a) == items_of(b)
    assert items_of(a)                                          # the subject is in some stratum


def test_chunks_cut_a_single_subject_benchmark(cat):
    chunks = [b for b in cat.pseudos if b.parent == "solo"]
    assert len(chunks) == 400 // 150
    seen = set()
    for b in chunks:
        assert len(b.pairs) == 1 and len(b.pairs[0].item_keys) == 150
        assert not (b.items & seen)
        seen |= b.items


# --- runs ------------------------------------------------------------------------

def test_runs_are_seeded_and_reproducible(cat):
    s1, s2 = T.Sampler(cat, T.Regime(min_release=None)), T.Sampler(cat, T.Regime(min_release=None))
    for i in range(5):
        a = s1.run(np.random.default_rng([3, i]))
        b = s2.run(np.random.default_rng([3, i]))
        assert [(p.subject_id, p.benchmark_id, sorted(k)) for p, k in a] == \
               [(p.subject_id, p.benchmark_id, sorted(k)) for p, k in b]


@pytest.mark.parametrize("regime", [
    T.Regime(min_release=None),
    T.Regime(min_release=None, level_mean=None),
    T.Regime(min_release=None, tilt="pair", level_mean=-2.0),
    T.Regime(min_release="2025-01-01", recency=2.0, exclude=()),
])
def test_run_invariants(cat, regime):
    S = T.Sampler(cat, regime)
    for i in range(20):
        run = S.run(np.random.default_rng([5, i]))
        assert 1 <= len(run) <= regime.n_pairs[1]
        assert sum(len(k) for _, k in run) <= regime.cap
        subjects = [p.subject_id for p, _ in run]
        dicts = [tuple(sorted(D.official_subject(p.subject).items())) for p, _ in run]
        assert len(set(subjects)) == len(run) and len(set(dicts)) == len(run)
        for p, keys in run:
            assert MIN_ITEMS <= len(keys) <= regime.max_items
            assert keys <= set(p.item_keys)
            assert T.parent_of(p.benchmark_id) not in regime.exclude
            if regime.min_release:
                assert T.years_since(p.subject["release_date"], regime.min_release) is not None
        names = {p.benchmark_id for p, _ in run}
        for a in names:
            for c in names:
                if a < c and T.parent_of(a) == T.parent_of(c):
                    A, C = by_name(cat)[a].items, by_name(cat)[c].items
                    assert len(A & C) <= regime.overlap * min(len(A), len(C))


def test_a_run_scores_under_the_official_protocol(cat):
    S = T.Sampler(cat, T.Regime(min_release=None))
    run = S.run(np.random.default_rng(11))
    res = O.run_official(run, lambda: B.smoothed_mean(4.0, 0.5))
    assert len(res["rows"]) == len(run)
    assert set(res["names"].values()) == {p.benchmark_id for p, _ in run}
    parents = T.anon_parents(run)
    assert set(parents) == {r["benchmark_id"] for r in res["rows"]}
    assert set(parents.values()) == set(T.run_parents(run))
    desc = T.describe(run)
    for d, r in zip(desc, res["rows"]):
        assert d["eval_items"] == r["n_items"]
        assert 0.0 <= d["p"] <= 1.0 and math.isfinite(d["logit"])


def test_level_mean_moves_realized_levels(cat):
    def mean_logit(m):
        S = T.Sampler(cat, T.Regime(min_release=None, level_mean=m, level_sd=0.8))
        return np.mean([d["logit"] for i in range(40)
                        for d in T.describe(S.run(np.random.default_rng([9, i])))])
    lo, mid, hi = mean_logit(-2.5), mean_logit(0.0), mean_logit(2.5)
    assert lo < mid < hi
    assert hi - lo > 1.0


def test_one_draw_distribution_tracks_the_target(cat):
    S = T.Sampler(cat, T.Regime(min_release=None, level_mean=-1.0, level_sd=1.2, tilt="pair",
                                max_tilt=1e9))
    assert S.stats["one_draw_logit_mean"] == pytest.approx(-1.0, abs=0.35)


def test_recency_favours_recent_subjects(cat):
    def mean_years(r):
        S = T.Sampler(cat, T.Regime(min_release="2025-01-01", recency=r, level_mean=None))
        return np.mean([T.years_since(p.subject["release_date"], "2025-01-01")
                        for i in range(40) for p, _ in S.run(np.random.default_rng([2, i]))])
    assert mean_years(3.0) > mean_years(0.0) + 0.1


def test_repeat_controls_pairs_per_pseudo_benchmark(cat):
    def per_bench(rep):
        S = T.Sampler(cat, T.Regime(min_release=None, repeat=rep, n_pairs=(8, 8)))
        runs = [S.run(np.random.default_rng([4, i])) for i in range(20)]
        return np.mean([len(r) / len({p.benchmark_id for p, _ in r}) for r in runs])
    assert per_bench(0.0) == 1.0
    assert per_bench(0.6) > 1.2


def test_exclusion_and_parent_cap(cat):
    S = T.Sampler(cat, T.Regime(min_release=None, exclude=("alpha", "solo"), max_per_parent=1))
    for i in range(10):
        run = S.run(np.random.default_rng([6, i]))
        ps = Counter(T.parent_of(p.benchmark_id) for p, _ in run)
        assert "alpha" not in ps and "solo" not in ps
        assert len({p.benchmark_id for p, _ in run if T.parent_of(p.benchmark_id) == "beta"}) <= 1


def test_regime_validation(cat):
    with pytest.raises(ValueError):
        T.Sampler(cat, T.Regime(tilt="nonsense"))
    with pytest.raises(ValueError):
        T.Sampler(cat, T.Regime(min_release="2030-01-01"))


def test_helpers():
    assert T.years_since("2025-07", "2025-01-01") == pytest.approx(181 / 365.25)
    assert T.years_since("2024-12-31", "2025-01-01") is None
    assert T.years_since("", "2025-01-01") is None
    assert T.feature({"item_features": "a=1;lang=go"}, "lang") == "go"
    assert T.feature({"item_features": ""}, "lang") == ""
    assert T.logit_rate(0, 10) == pytest.approx(math.log(0.5 / 10.5))
    assert T.parent_of("multi_swebench::g3") == "multi_swebench"
    assert T.parent_of("matharena") == "matharena"


def test_date_shift_moves_dates_and_nothing_else(cat):
    assert T.shift_date("2025-02", 1.0) == "2026-02-01"
    assert T.shift_date("2025-06-30", -0.5) == "2024-12-29"          # 183 days
    assert T.shift_date("", 1.0) == "" and T.shift_date("soon", 1.0) == "soon"
    base = T.Sampler(cat, T.Regime(min_release="2025-01-01", exclude=(), date_shift=0.0))
    moved = T.Sampler(cat, T.Regime(min_release="2025-01-01", exclude=(), date_shift=1.5))
    for i in range(5):
        a = base.run(np.random.default_rng([8, i]))
        b = moved.run(np.random.default_rng([8, i]))
        assert [(p.subject_id, p.benchmark_id, k) for p, k in a] == \
               [(p.subject_id, p.benchmark_id, k) for p, k in b]      # the same draw
        for (p, _), (q, _) in zip(a, b):
            assert q.responses == p.responses
            assert q.subject["release_date"] == T.shift_date(p.subject["release_date"], 1.5)
            assert q.subject["access_date"] == T.shift_date(p.subject["access_date"], 1.5)
            assert {k: v for k, v in q.subject.items() if not k.endswith("_date")} == \
                   {k: v for k, v in p.subject.items() if not k.endswith("_date")}
            assert p.subject["release_date"] >= "2025"                # selected on the real date
    # the catalogue's own subjects are never modified
    assert all(p.subject["release_date"] in DATES for b in cat.pseudos for p in b.pairs)


# --- knobs added after review ----------------------------------------------------

def test_kinds_filter_and_default_leaves_out_mix(cat):
    kinds_in = lambda S: {T.kind_of(p.benchmark_id) for i in range(30)
                          for p, _ in S.run(np.random.default_rng([12, i]))}
    assert "mix" not in kinds_in(T.Sampler(cat, T.Regime(min_release=None)))
    only = kinds_in(T.Sampler(cat, T.Regime(min_release=None, kinds=("group", "whole"))))
    assert only <= {"group", "whole"} and "group" in only
    assert kinds_in(T.Sampler(cat, T.Regime(min_release=None, kinds=("mix",)))) == {"mix"}
    with pytest.raises(ValueError):
        T.Sampler(cat, T.Regime(kinds=("nonsense",)))
    with pytest.raises(ValueError):
        T.Sampler(cat, T.Regime(kind_weights=(("stratum", 0.0),)))


def test_kind_weights_down_weight_a_kind(cat):
    def stratum_share(w):
        S = T.Sampler(cat, T.Regime(min_release=None, level_mean=None,
                                    kind_weights=(("stratum", w),)))
        ks = [T.kind_of(p.benchmark_id) for i in range(60)
              for p, _ in S.run(np.random.default_rng([13, i]))]
        return np.mean([k == "stratum" for k in ks])
    assert stratum_share(0.1) < stratum_share(1.0) - 0.1
    # weight 1 is the default draw, bit for bit
    a = T.Sampler(cat, T.Regime(min_release=None))
    b = T.Sampler(cat, T.Regime(min_release=None, kind_weights=(("stratum", 1.0),)))
    assert np.array_equal(a.omega, b.omega)


def test_runs_reach_the_minimum_size_when_they_can(cat):
    S = T.Sampler(cat, T.Regime(min_release=None, n_pairs=(5, 8)))
    sizes = [len(S.run(np.random.default_rng([14, i]))) for i in range(40)]
    assert min(sizes) >= 5


def test_date_cap(cat):
    assert T.shift_date("2025-10-01", 1.25, "2026-12-31") == "2026-12-31"
    assert T.shift_date("2025-02-01", 1.0, "2026-12-31") == "2026-02-01"
    assert T.shift_date("2026-03", 0.0, "2025-12-31") == "2025-12-31"
    S = T.Sampler(cat, T.Regime(min_release="2025-01-01", exclude=(), date_shift=1.5,
                                date_cap="2026-12-31"))
    for i in range(10):
        for p, _ in S.run(np.random.default_rng([15, i])):
            assert p.subject["release_date"] <= "2026-12-31"
            assert p.subject["access_date"] <= "2026-12-31"


def test_split_after_cut_evaluates_half_of_what_a_pair_keeps(cat):
    S = T.Sampler(cat, T.Regime(min_release=None, split_after_cut=True, min_kept=88,
                                n_pairs=(5, 11)))
    for i in range(15):
        run = S.run(np.random.default_rng([16, i]))
        for p, keys in run:
            assert set(p.item_keys) == set(keys) and len(keys) >= 88
        for d, (p, keys) in zip(T.describe(run), run):
            assert d["eval_items"] == len(keys) - len(keys) // 2 >= 44
    with pytest.raises(ValueError):                              # 12 x 88 > 1,000
        T.Sampler(cat, T.Regime(min_kept=88))


def test_training_pairs_leave_out_every_parent_of_the_run(pairs, cat):
    S = T.Sampler(cat, T.Regime(min_release=None, exclude=("gamma",)))      # gamma survives
    for i in range(10):
        run = S.run(np.random.default_rng([17, i]))
        parents = set(T.run_parents(run))
        # official's helper compares names, and pseudo names match no public pair
        assert len(O.training_pairs(pairs, run)) == len(pairs)
        train = T.training_pairs(pairs, run)
        assert "gamma" not in parents and any(p.benchmark_id == "gamma" for p in train)
        assert not any(p.benchmark_id in parents for p in train)
        assert len(train) == sum(p.benchmark_id not in parents for p in pairs)
        only = T.training_pairs(pairs, run, parents={"alpha"})
        assert not any(p.benchmark_id == "alpha" for p in only)
        hidden = T.training_pairs(pairs, run, hide_subjects=True)
        ids = {p.subject_id for p, _ in run}
        names = {p.subject["normalized_name"] for p, _ in run}
        assert not any(p.subject_id in ids or p.subject["normalized_name"] in names
                       for p in hidden)


def test_item_oracle_and_key_share(cat):
    S = T.Sampler(cat, T.Regime(min_release=None, exclude=()))
    rows = [x for i in range(10) for x in T.item_oracle(S.run(np.random.default_rng([18, i])), cat)]
    solo = [x for x in rows if x is None]
    real = [x for x in rows if x is not None]
    assert real
    for x in real:
        assert 0 <= x["item"] <= 1 and 0 <= x["level"] <= 0.25 + 1e-12
        assert x["kind"] in T.KINDS and x["parent"] in cat.difficulty
    # difficulty spreads within a pair here, so knowing it beats knowing the rate
    assert np.mean([x["item"] for x in real]) < np.mean([x["level"] for x in real])
    assert all(x is None or x["parent"] != "solo" for x in rows)
    b = {"a": 0.0, "b": 0.0, "c": 2.0, "d": 2.0}
    assert T.key_share("abcd", b, {"a": "x", "b": "x", "c": "y", "d": "y"})["key_share"] == \
        pytest.approx(1.0)
    assert T.key_share("abcd", b, {"a": "x", "b": "y", "c": "x", "d": "y"})["key_share"] == \
        pytest.approx(0.0)


def test_small_helpers():
    assert T.kind_of("matharena") == "public"
    assert [T.kind_of(f"x::{t}") for t in ("g0", "m3", "q1of2", "c4", "all")] == \
        ["group", "mix", "stratum", "chunk", "whole"]
    assert T.cluster_key("matharena::q1of2", "s1") == T.cluster_key("matharena::g3", "s1") == \
        ("matharena", "s1")
    assert T.rate_from_brier(0.25) == 0.5 and T.rate_from_brier(0.3) == 0.5
    assert T.rate_from_brier(0.09) == pytest.approx(0.1)
    assert T.rate_from_brier(0.0) == pytest.approx(1e-4)
