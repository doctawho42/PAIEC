"""The label-conditional per-item correction (paiec/itemsig.py)."""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from paiec import itemsig as I
from paiec import official as O
from paiec.hier import HierPredictor, Hyper, item_groups, make_hier
from paiec.predict import Predictor, item_key
from tests.synth import make_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CITIES = ["Delhi", "Paris", "Lima", "Oslo", "Cairo", "Quito", "Perth", "Tunis", "Hanoi", "Accra",
          "Dakar", "Sofia", "Minsk", "Riga", "Kyiv", "Baku", "Doha", "Suva", "Apia", "Male"]


def subject(name, **kw):
    s = dict.fromkeys(("normalized_name", "provider", "release_date", "access_date", "harness",
                       "harness_version", "reasoning_effort", "subject_features_extra"), "")
    s.update(normalized_name=name, **kw)
    return s


def item(text, bid="bench_t", features=""):
    return {"item_content": text, "item_features": features, "interactors": "", "benchmark_id": bid}


def words(rng, n, vocab=400):
    return " ".join(f"w{k}" for k in rng.integers(0, vocab, n))


def templated(rng, easy, j, bid="bench_t"):
    """Two task templates of a web-agent-like benchmark; which one an item
    uses decides its difficulty, and no item_features say so."""
    a, b = rng.choice(CITIES, 2, replace=False)
    if easy:
        text = (f"Show me places in {a} for {j % 5 + 1} guests on the dates "
                f"Sept {j % 27 + 1} to {j % 27 + 3}.")
    else:
        text = (f"Book the cheapest round-trip flight from {a} to {b} "
                f"departing on day {j % 28 + 1} "
                f"and returning a week later with one checked bag and an aisle seat.")
    return item(text, bid)


def token_benchmark(seed=0, n_labels=31, n_targets=120):
    """One pair on a benchmark whose outcomes depend on the item's template:
    easy items succeed with probability 0.85, hard ones 0.15."""
    rng = np.random.default_rng(seed)
    s = subject("model 1")
    labeled = []
    for j in range(n_labels):
        easy = j % 2 == 0
        labeled.append([[s, templated(rng, easy, j)], int(rng.random() < (0.85 if easy else 0.15))])
    targets = [([s, templated(rng, j % 2 == 0, 100 + j)], j % 2 == 0) for j in range(n_targets)]
    return s, labeled, targets


def hier():
    return HierPredictor(None, Hyper())


# --- the representation ---------------------------------------------------------------

def test_the_index_is_exact_tfidf_cosine():
    """Postings give the cosine of smooth-idf, sublinear-tf vectors fitted on
    the indexed texts, the query's unseen features counted in its norm at the
    largest idf; the Gram matrix agrees."""
    rng = np.random.default_rng(0)
    texts = [words(rng, int(rng.integers(3, 200)), 150) for _ in range(30)]
    bags = [I.features(t) for t in texts]
    for blk in (0, 1):
        ix = I._Index([b[blk] for b in bags[:20]])
        vocab = sorted(set(np.concatenate([b[blk][0] for b in bags[:20]]).tolist()))
        pos = {v: k for k, v in enumerate(vocab)}
        df = np.zeros(len(vocab))
        for b in bags[:20]:
            df[[pos[i] for i in b[blk][0].tolist()]] += 1
        idf = np.log(21 / (1 + df)) + 1
        X = np.zeros((20, len(vocab)))
        for d, b in enumerate(bags[:20]):
            cols = [pos[i] for i in b[blk][0].tolist()]
            X[d, cols] = b[blk][1] * idf[cols]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        for q in range(15, 30):
            ids, tf = bags[q][blk]
            w = np.array([t * (idf[pos[i]] if i in pos else math.log(21) + 1)
                          for i, t in zip(ids.tolist(), tf)])
            x = np.zeros(len(vocab))
            for i, v in zip(ids.tolist(), w):
                if i in pos:
                    x[pos[i]] = v
            assert np.allclose(ix.query(bags[q][blk]), X @ x / np.linalg.norm(w))
        assert np.allclose(I._gram(ix), X @ X.T)
        assert np.array_equal(I._gram(ix, [7, 2]), I._gram(ix)[[7, 2]])


def test_features_hash_the_same_in_every_process():
    """No salted hash(): another interpreter (another PYTHONHASHSEED) gets the
    same buckets, so workers agree."""
    text = "Prove that $x^2 + 1 > 0$ for all real x. été — ok"
    w, c = I.features(text)
    code = ("from paiec import itemsig as I; w, c = I.features(%r); "
            "print(int(w[0].sum()), float(w[1].sum()), int(c[0].sum()), float(c[1].sum()))" % text)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONHASHSEED="12345"), cwd=ROOT).stdout.split()
    assert out == [str(int(w[0].sum())), repr(float(w[1].sum())), str(int(c[0].sum())),
                   repr(float(c[1].sum()))]


def test_long_texts_are_read_at_head_and_tail():
    text = "A" * 5000 + "middle" + "Z" * 5000
    assert I.excerpt(text, 10, 20) == "A" * 10 + "\n" + "Z" * 20
    assert I.excerpt("short", 10, 20) == "short"
    assert "middle" not in I.text_of(item(text), 100, 100)


# --- purity and the no-effect cases -----------------------------------------------------

def test_pure_function_of_input_and_labeled():
    _, labeled, targets = token_benchmark(3)
    others = [[[subject("model 2"), templated(np.random.default_rng(9), j % 3 == 0, 300 + j)],
               j % 2] for j in range(12)]
    labeled = labeled + others
    rng = np.random.default_rng(4)
    shuffled = [labeled[i] for i in rng.permutation(len(labeled))]
    a, b = I.ItemSig(hier()), I.ItemSig(hier())
    first = [a.predict(t, labeled) for t, _ in targets[:40]]
    a.predict(targets[0][0], labeled[:7])           # another list's state in between
    assert [a.predict(t, shuffled) for t, _ in targets[:40]] == first
    assert [b.predict(t, labeled) for t, _ in reversed(targets[:40])] == first[::-1]
    assert a.failures == b.failures == 0
    assert any(p != hier().predict(t, labeled) for p, (t, _) in zip(first, targets))


def test_no_labels_on_the_benchmark_is_the_base_exactly():
    s, labeled, targets = token_benchmark(1)
    base, sig = hier(), I.ItemSig(hier())
    elsewhere = [[[s, item(e[0][1]["item_content"], "another")], e[1]] for e in labeled]
    for t, _ in targets[:20]:
        for lab in ([], None, elsewhere):
            assert sig.predict(t, lab) == base.predict(t, lab)
            assert type(sig.predict(t, lab)) is float
    assert sig.failures == 0


def test_beta_zero_is_the_base_exactly():
    _, labeled, targets = token_benchmark(2)
    base, sig = hier(), I.ItemSig(hier(), beta=0.0)
    assert [sig.predict(t, labeled) for t, _ in targets] == [base.predict(t, labeled)
                                                              for t, _ in targets]


def test_a_pairs_first_label_moves_nothing_when_centred():
    s, labeled, targets = token_benchmark(5)
    base, sig = hier(), I.ItemSig(hier())
    for t, _ in targets[:10]:
        assert sig.predict(t, labeled[:1]) == base.predict(t, labeled[:1])


# --- what it should pick up --------------------------------------------------------------

def test_recovers_a_text_signal_the_base_cannot_see():
    """Outcomes depend on the item's template, which no item_features name: the
    base predicts both templates alike, the layer separates them in the right
    direction and lowers the Brier against the truth."""
    gains = []
    for seed in range(3):
        _, labeled, targets = token_benchmark(seed)
        base, sig = hier(), I.ItemSig(hier())
        pb = np.array([base.predict(t, labeled) for t, _ in targets])
        ps = np.array([sig.predict(t, labeled) for t, _ in targets])
        easy = np.array([e for _, e in targets])
        truth = np.where(easy, 0.85, 0.15)
        assert abs(pb[easy].mean() - pb[~easy].mean()) < 1e-9
        assert ps[easy].mean() - ps[~easy].mean() > 0.02
        assert ps[easy].min() > ps[~easy].max()
        gains.append(np.mean((pb - truth) ** 2) - np.mean((ps - truth) ** 2))
        assert sig.failures == 0
    assert min(gains) > 0


def test_twins_are_pulled_toward_their_label():
    """A target whose exact content is one labeled item's, under another key,
    moves toward that item's label; the base, which keys items on every
    field, does not read it."""
    rng = np.random.default_rng(7)
    s = subject("model 1")
    labeled = [[[s, item(words(rng, 40), features="comp=a")], j % 2] for j in range(12)]
    win, loss = words(rng, 40), words(rng, 40)
    labeled += [[[s, item(win, features="comp=a")], 1], [[s, item(loss, features="comp=a")], 0]]
    base, sig = hier(), I.ItemSig(hier())
    for text, sign in ((win, 1), (loss, -1)):
        t = [s, item(text, features="comp=a_I")]
        terms = sig.terms(t, labeled)
        assert terms.twin.sum() == 1 and not terms.excl.any()
        assert sign * (sig.predict(t, labeled) - base.predict(t, labeled)) > 0.02
        off = I.ItemSig(hier(), twins=False)
        assert abs(off.predict(t, labeled) - base.predict(t, labeled)) < \
            abs(sig.predict(t, labeled) - base.predict(t, labeled))


def test_a_shared_template_is_not_a_twin():
    """A content two labeled items carry is a template (matharena's Kangaroo
    image prompt), not the target's twin."""
    rng = np.random.default_rng(8)
    s = subject("model 1")
    prompt = "You are given a problem in the form of an image. Look at it carefully."
    labeled = [[[s, item(prompt, features=f"comp=k;idx={j}")], j % 2] for j in range(2)]
    labeled += [[[s, item(words(rng, 30))], j % 2] for j in range(8)]
    terms = I.ItemSig(hier()).terms([s, item(prompt, features="comp=k;idx=9")], labeled)
    assert not terms.twin.any()


def test_labels_on_the_targets_own_item_are_left_to_the_base():
    """Another subject's label on the target's very item is read by the base
    through the item's residual; the layer leaves it out."""
    rng = np.random.default_rng(11)
    s, other = subject("model 1"), subject("model 2")
    target = item(words(rng, 30))
    labeled = [[[other, target], 1]] + [[[other, item(words(rng, 30))], j % 2] for j in range(6)]
    sig = I.ItemSig(hier(), center=False, other=1.0)
    terms = sig.terms([s, target], labeled)
    assert terms.excl.sum() == 1 and terms.excl[np.flatnonzero(terms.y == 1)[0]]
    assert I.weights(terms, sig.hyper)[terms.excl].sum() == 0


def test_the_shift_is_clipped_and_scaled_by_beta():
    _, labeled, targets = token_benchmark(4)
    t = targets[0][0]
    raw = I.raw_shift(I.ItemSig(hier()).terms(t, labeled), I.SigHyper())
    for beta, cap in ((1.0, 10.0), (2.0, 10.0), (50.0, 0.3)):
        sig = I.ItemSig(hier(), beta=beta, max_shift=cap)
        assert sig.shift(t, labeled) == pytest.approx(min(max(beta * raw, -cap), cap))


def test_the_pair_slope_backs_off_where_text_says_nothing():
    """The empirical-Bayes slope stays in [0, 1], near 1 where the template
    decides the outcome, and lower when the same labels are shuffled over
    the items."""
    _, labeled, targets = token_benchmark(0, n_labels=31)
    rng = np.random.default_rng(1)
    ys = [e[1] for e in labeled]
    shuffled = [[e[0], ys[k]] for e, k in zip(labeled, rng.permutation(len(ys)))]
    h = I.SigHyper(eb=0.5)
    b = []
    for lab in (labeled, shuffled):
        sig = I.ItemSig(hier(), h)
        terms = sig.terms(targets[0][0], lab)
        b.append(I.slope(terms, h))
    assert 0 <= b[1] < b[0] <= 1 and b[0] > 0.5
    assert I.slope(terms, I.SigHyper()) == 1.0        # off


# --- which p_j are asked for, and how -------------------------------------------------------

def test_p_j_are_asked_for_the_targets_own_pair_only():
    """With other 0 a target's p_j are its own pair's (one base call per
    record, once per labeled list), and the shift is bit for bit the one
    every p_j would give; with other > 0 every record's are asked for."""
    rng = np.random.default_rng(13)
    a, b = subject("model 1"), subject("model 2")
    labeled = [[[a, item(words(rng, 30))], j % 2] for j in range(31)]
    labeled += [[[b, item(words(rng, 30))], int(j % 3 == 0)] for j in range(20)]
    labeled += [[[a, item(words(rng, 30), "elsewhere")], j % 2] for j in range(10)]
    inner, calls = hier(), []

    def counting(input, lab=None):
        calls.append(1)
        return inner.predict(input, lab)
    sig = I.ItemSig(counting)
    ta, tb = [a, item(words(rng, 30))], [b, item(words(rng, 30))]
    sig.predict(ta, labeled)
    assert len(calls) == 1 + 31
    sig.predict([a, item(words(rng, 30))], labeled)
    assert len(calls) == 1 + 31 + 1
    sig.predict(tb, labeled)
    assert len(calls) == 1 + 31 + 1 + 1 + 20
    sig.predict([subject("model 3"), item(words(rng, 30))], labeled)    # no pair: no p_j
    assert len(calls) == 1 + 31 + 1 + 1 + 20 + 1

    calls.clear()
    wide = I.ItemSig(counting, other=0.5)
    wide.predict(ta, labeled)
    assert len(calls) == 1 + 51
    h0 = I.SigHyper()
    for t in (ta, tb):
        own, every = sig.terms(t, labeled), wide.terms(t, labeled)
        assert not own.every and every.every
        assert I.shift(own, h0) == I.shift(every, h0) != 0.0
        with pytest.raises(ValueError):             # own-pair terms under other > 0
            I.raw_shift(own, I.SigHyper(other=0.5))
    assert sig.failures == wide.failures == 0


def test_as_new_adds_one_unselectable_key_and_keeps_every_group():
    """The item as a new one: another key, the same content and benchmark, and
    hier's groups those of the item plus NEW_KEY alone, whatever the format
    of interactors (empty, k=v, JSON, an opaque value, unbalanced brackets)."""
    cases = [("", ""), ("tool=a;n=2", ""), ('{"tool": "a", "k": [1, 2]}', "comp=x"),
             ("browser", "comp=x"), ("browser", ""), ("[open;x=1", "comp=y"),
             ("tool=a", '{"comp": "x"}')]
    for inter, feats in cases:
        it = {"item_content": "Solve 2x = 4.", "item_features": feats, "interactors": inter,
              "benchmark_id": "b"}
        for prefix in (False, True):
            new = I.as_new(it, prefix)
            if prefix and not item_groups(it):
                assert new is None          # would lose its prefix group
                continue
            assert new is not None and item_key(new) != item_key(it), (inter, feats)
            assert {k: new[k] for k in ("item_content", "benchmark_id")} == \
                {k: it[k] for k in ("item_content", "benchmark_id")}
            got, want = item_groups(new, prefix), item_groups(it, prefix)
            assert len(got) == len(want) + 1
            assert {k: v for k, v in got.items() if I.NEW_KEY not in k} == want
    long = {"item_content": "q", "item_features": "f" * 5000, "interactors": "i" * 5000,
            "benchmark_id": "b"}
    assert I.as_new(long) is None                   # hashed whole by hier
    assert I.as_new({"item_content": "q", "item_features": 3, "interactors": None}) is not None


@pytest.mark.parametrize("fmt", ["kv", "json"])
def test_struct_reads_a_labeled_item_as_a_new_one_in_its_own_groups(fmt):
    """resid 'struct' asks the base for each labeled item as a new one that
    keeps its interactors group: equal to an item under another key with the
    same groups, and unlike a marker appended to the interactors string,
    which moves a k=v or JSON item to an unseen level."""
    rng = np.random.default_rng(12)
    s = subject("model 1")

    def inter(k):
        return f"n={k};tool=t{k}" if fmt == "kv" else json.dumps({"n": k, "tool": f"t{k}"})
    labeled = []
    for j in range(18):
        k = j % 3
        it = item(words(rng, 30))
        it["interactors"] = inter(k)
        labeled.append([[s, it], 1 if k == 0 else (0 if k == 1 else j % 2)])
    base, sig = hier(), I.ItemSig(hier(), resid="struct")
    t = sig.terms([s, item(words(rng, 30))], labeled, kinds=("struct", "leave_in"))
    moved = 0
    for k, r in enumerate(t.bench.recs):
        same = dict(r[5], item_content=r[5]["item_content"] + " ")     # another key, same groups
        ref = base.predict([r[4], same], labeled)
        assert t.p["struct"][k] == pytest.approx(ref, abs=1e-12)
        assert abs(t.p["struct"][k] - t.p["leave_in"][k]) > 1e-3
        marked = dict(r[5], interactors=r[5]["interactors"] + "\x00itemsig:new")
        moved += abs(base.predict([r[4], marked], labeled) - ref) > 0.01
    assert moved >= 12
    assert sig.failures == 0


def test_struct_keeps_the_leave_in_p_j_where_no_new_key_fits():
    rng = np.random.default_rng(14)
    s = subject("model 1")
    labeled = [[[s, item(words(rng, 30))], j % 2] for j in range(8)]
    odd = item(words(rng, 30), features="f" * 5000)
    odd["interactors"] = "i" * 5000
    labeled.append([[s, odd], 1])
    sig = I.ItemSig(hier(), resid="struct")
    t = sig.terms([s, item(words(rng, 30))], labeled, kinds=("struct", "leave_in"))
    k = [r[2] for r in t.bench.recs].index(item_key(odd))
    assert t.p["struct"][k] == t.p["leave_in"][k]
    assert all(t.p["struct"][j] != t.p["leave_in"][j] for j in range(len(t.y)) if j != k)


def test_the_feature_memo_is_bounded_in_bytes():
    rng = np.random.default_rng(15)
    sig = I.ItemSig(hier())
    sig.DOC_BYTES = 400_000
    for k in range(40):
        sig.doc(bytes([k]), item(words(rng, 1500, 3_000)), 2_000, 6_000)
    held = sum(a.nbytes for w, c, _ in sig._docs.values() for a in (*w, *c))
    assert 0 < held == sig._doc_bytes <= sig.DOC_BYTES
    assert len(sig._docs) < 40


# --- robustness, wrapping, speed ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    None, [], [None, None], "x", [subject("a")], [subject("a"), "item"],
    [{"normalized_name": 3}, {"benchmark_id": None, "item_features": 7}],
    [subject("a"), {"item_content": None, "item_features": "{bad json", "benchmark_id": "b"}],
    [subject("a"), {"item_content": 12345, "benchmark_id": "b", "interactors": None}],
    [subject("a"), item("\ud800 lone surrogate", "b")],
])
def test_malformed_inputs_never_raise(bad):
    rng = np.random.default_rng(0)
    labeled = [[[subject("a"), item(words(rng, 10), "b", "k=v;n=3")], j % 2] for j in range(5)]
    weird = [[[subject("a"), {"item_content": None, "benchmark_id": "b"}], 1],
             [[subject("a"), item("", "b")], 0], [[subject("b"), item("\ud800", "b")], 1]]
    bad_labels = [[[subject("a"), item("x", "b")], "1"], [[subject("a"), item("y", "b")], 2]]
    for lab in (labeled, None, [], "junk", [None, 1, [[1, 2], 1]], labeled + weird,
                labeled + bad_labels):
        p = I.make_itemsig(lambda: make_hier())()(bad, lab)
        assert isinstance(p, float) and 1e-4 <= p <= 1 - 1e-4


def test_a_failing_layer_answers_the_base_and_a_failing_base_one_half(monkeypatch, capsys):
    _, labeled, targets = token_benchmark(6)
    t = targets[0][0]
    sig = I.ItemSig(hier())
    monkeypatch.setattr(I, "shift", lambda *a, **k: 1 / 0)
    assert sig.predict(t, labeled) == hier().predict(t, labeled)
    assert sig.failures == 1

    def broken(input, labeled=None):
        raise RuntimeError("base down")
    assert I.ItemSig(broken).predict(t, labeled) == 0.5
    assert I.ItemSig(lambda i, l=None: float("nan")).predict(t, labeled) == 0.5


def test_hyperparameters_are_checked():
    for bad in (dict(tau=0), dict(gamma=-1), dict(other=2), dict(word=-0.1), dict(resid="x"),
                dict(beta=float("nan")), dict(beta=-0.5), dict(beta=True), dict(gamma=True),
                dict(max_shift=-1), dict(eb=-1), dict(head=-1), dict(tail=False),
                dict(head="2000"), dict(tau="1")):
        with pytest.raises(ValueError):
            I.SigHyper(**bad)
    with pytest.raises(ValueError):
        I.make_itemsig(lambda: make_hier(), tau=-1)


@pytest.mark.parametrize("base,resid,eb", [
    ("hier", "leave_in", 0.0), ("hier", "struct", 0.0), ("hier", "leave_in", 0.5),
    ("hier", "struct", 0.5), ("predictor", "struct", 0.0), ("hier_object", "leave_in", 0.5),
])
def test_wraps_any_factory_through_the_official_protocol(base, resid, eb):
    """make_itemsig wraps hier (a predict callable or the object) or the
    Predictor as a model_factory, under either residual and with or without
    the per-pair slope: the result is the same under the platform's copies
    and five workers, B0 is the base's, and the layer moves later budgets."""
    pairs = make_pairs(3, 2, 100)
    fac = {"hier": lambda: make_hier(), "predictor": lambda: Predictor().predict,
           "hier_object": lambda: HierPredictor(None, Hyper())}[base]
    kw = dict(resid=resid, eb=eb)
    fast = O.run_official(pairs, I.make_itemsig(fac, **kw), deepcopy=False, workers=1)
    slow = O.run_official(pairs, I.make_itemsig(fac, **kw), deepcopy=True, workers=5)
    assert [r["brier"] for r in fast["rows"]] == [r["brier"] for r in slow["rows"]]
    plain = O.run_official(pairs, lambda: fac() if base != "hier_object" else fac().predict,
                           deepcopy=False)
    assert [r["brier"][0] for r in fast["rows"]] == [r["brier"][0] for r in plain["rows"]]
    later = [[v for B, v in sorted(r["brier"].items()) if B] for r in fast["rows"]]
    assert later != [[v for B, v in sorted(r["brier"].items()) if B] for r in plain["rows"]]


def test_formative_size_is_fast():
    """12 pairs of 31 labels over five benchmarks (372 labels), 500 targets,
    one thread, hier underneath. Four benchmarks hold texts of 20 to 600
    words (about 100 to 3,600 characters; public medians run from 92 to 1,147
    outside researchcodebench) and one holds 100k-character items like
    researchcodebench's. Timed in CPU seconds, which a loaded machine
    inflates far less than wall time: at most 8 times the base's own on the
    same workload, and under 10 s (30 s when the load average exceeds the
    CPU count). Wall times are printed too."""
    rng = np.random.default_rng(1)
    long_ctx = words(rng, 20_000, 3_000)
    labeled, targets = [], []
    for k in range(12):
        bid = f"benchmark_{k % 5}"
        s = subject(f"model {k}", provider="openai")

        def text(j):
            if bid == "benchmark_4":
                return long_ctx + f"\n# TODO block {j}\n" + words(rng, 300, 3_000)
            return words(rng, int(rng.integers(20, 600)), 3_000)
        for j in range(31):
            labeled.append([[s, item(text(j), bid, f"topic=t{j % 6};idx={j}")],
                            int(rng.random() < 0.4)])
        targets += [[s, item(text(100 + j), bid, f"topic=t{j % 7}")] for j in range(42)]
    targets = targets[:500]
    base = make_hier()
    c0, t0 = time.process_time(), time.perf_counter()
    pb = [base(t, labeled) for t in targets]
    c1, t1 = time.process_time(), time.perf_counter()
    predict = I.make_itemsig(lambda: make_hier())()
    ps = [predict(t, labeled) for t in targets]
    c2, t2 = time.process_time(), time.perf_counter()
    cpu_base, cpu_sig = c1 - c0, c2 - c1
    try:
        load = os.getloadavg()[0]
    except (AttributeError, OSError):
        load = 0.0
    print(f"\nformative size: base alone {cpu_base:.2f} s CPU ({t1 - t0:.2f} s wall), with "
          f"itemsig {cpu_sig:.2f} s CPU ({t2 - t1:.2f} s wall, "
          f"{(t2 - t1) / len(targets) * 1e3:.1f} ms a call), load average {load:.0f}")
    assert cpu_sig < 8 * cpu_base + 1.0
    assert cpu_sig < (30.0 if load > (os.cpu_count() or 1) else 10.0)
    assert all(1e-4 <= p <= 1 - 1e-4 for p in ps)
    assert sum(p != q for p, q in zip(ps, pb)) > 400
