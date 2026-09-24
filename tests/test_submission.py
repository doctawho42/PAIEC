"""The submission archive as the platform runs it.

The archive is staged from the member list tools/build_submission.py zips and
model.py is loaded the way the organisers' validator loads it, with the repo's
own paiec out of sys.modules so it cannot stand in for a module the archive
forgot. What only shows in a fresh process (a worker's first call, its thread
limits, several workers side by side) runs in fresh interpreters. No data
needed: the prior is submission/prior.json when a build has written one and a
small synthetic one otherwise, and the inputs are synthetic, in the official
format.
"""
import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (0, 1, 3, 7, 15, 31)


def _load(name, path, on_path=None):
    """spec_from_file_location with sys.path restored afterwards, as the validator does."""
    before = list(sys.path)
    if on_path:
        sys.path.insert(0, str(on_path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = before


bs = _load("build_submission", ROOT / "tools" / "build_submission.py")


def subject(name, provider="openai", release="2025-03-01", effort="high", **kw):
    s = dict.fromkeys(bs.SUBJECT_FIELDS, "")
    s.update(normalized_name=name, provider=provider, release_date=release,
             access_date="2026-01-15", reasoning_effort=effort, **kw)
    return s


def item(text, bid="bench-7f3a", features="tier=1"):
    return {"item_content": text, "item_features": features, "interactors": "",
            "benchmark_id": bid}


def make_session(seed=0, n_items=160, long_words=60):
    """Four anonymous benchmarks x three subjects, 50/50 split per pair.

    The first benchmark's items run to `long_words` words (14000 gives the
    ~90k characters of researchcodebench), the second's share a 420-character
    boilerplate prefix, the third's are multiple choice.
    """
    rng = np.random.default_rng(seed)
    vocab = np.array([f"w{i}" for i in range(4000)])
    subjects = [subject("acme-large 70b", "openai", "2025-06-01"),
                subject("acme-mini 8b", "anthropic", "2024-05-13", "low"),
                subject("other-model", "someorg", "", "")]
    theta = rng.normal(0, 1, len(subjects))
    boiler = "You are given a competition problem. " * 12
    labeled = {B: [] for B in BUDGETS}
    targets = []
    for b in range(4):
        bid = f"b-{rng.integers(16**12):012x}"
        diff = rng.normal(0, 1.2, n_items)
        items = []
        for j in range(n_items):
            words = " ".join(rng.choice(vocab, long_words if b == 0 else 60))
            mcq = words + "\nA) one\nB) two\nC) three\nD) four"
            text = {1: boiler + words, 2: mcq}.get(b, words)
            items.append(item(text, bid, f"tier={j % 3}"))
        for s, sub in enumerate(subjects):
            y = (rng.random(n_items) < 1 / (1 + np.exp(diff - theta[s]))).astype(int)
            perm = rng.permutation(n_items)
            acq, ev = perm[:n_items // 2], perm[n_items // 2:]
            for B in BUDGETS:
                labeled[B] += [[[sub, items[j]], int(y[j])] for j in acq[:B]]
            targets += [([sub, items[j]], int(y[j])) for j in ev]
    order = rng.permutation(len(targets))
    return labeled, [targets[i] for i in order]


@pytest.fixture(scope="module")
def session():
    return make_session()


@pytest.fixture(scope="module")
def long_session():
    return make_session(long_words=14000)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("archive")
    prior = ROOT / "submission" / "prior.json"
    if not prior.exists():
        from paiec.subjects import Spec
        spec = Spec(["openai", "anthropic"], ["high"], 500.0, 3.0)
        prior = tmp / "prior.json"
        prior.write_text(json.dumps({"coef": [0.1, 0.3, 0.0, 0.2, 0.1, -0.3, 0.3, 0.2, 0.1,
                                              0.2, 0.1, 0.1], "spec": spec.to_dict()}))
    bs.check_prior(str(prior))
    files = bs.members(str(prior))
    zpath = tmp / "paiec.zip"
    bs.write_zip(str(zpath), files)
    root = tmp / "submission"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    return zpath, root.resolve(), files


@pytest.fixture(scope="module")
def model(archive):
    """model.py from the archive, with the archive's paiec_rt in sys.modules for
    the duration of this module, and the repo's paiec out of it while model.py
    loads and put back afterwards."""
    ours = lambda *top: [n for n in sys.modules if n.split(".")[0] in top]
    saved = {n: sys.modules.pop(n) for n in ours("paiec", bs.RUNTIME)}
    try:
        m = _load("submission_model", archive[1] / "model.py", on_path=archive[1])
        m.loaded = {n: sys.modules[n] for n in ours(bs.RUNTIME)}
        m.leaked = ours("paiec")
        yield m
    finally:
        for n in ours("paiec", bs.RUNTIME):
            del sys.modules[n]
        sys.modules.update(saved)


def fresh(model):
    """A worker at the start of a checkpoint: same prior, nothing cached."""
    return type(model.PREDICTOR)(model.PREDICTOR.prior_coef, model.PREDICTOR.prior_spec)


def is_probability(p):
    return type(p) is float and math.isfinite(p) and 0.0 <= p <= 1.0


# --- packaging ---------------------------------------------------------------

def test_organisers_validator_accepts_archive(archive):
    if not os.path.isfile(bs.VALIDATOR):
        pytest.skip("third_party/paiec_baseline not cloned")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, bs.VALIDATOR, str(archive[0])], cwd=archive[0].parent,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("OK"), r.stdout + r.stderr


def test_every_loaded_module_comes_from_the_archive(model, archive):
    assert set(model.loaded) >= {f"{bs.RUNTIME}.{m}" for m in
                                 ("predict", "irt", "fitting", "subjects", "mcq")}
    for name, mod in model.loaded.items():
        assert Path(mod.__file__).resolve().is_relative_to(archive[1]), name
    assert model.leaked == []
    bs.check_imports(archive[2])


@pytest.mark.parametrize("source", [
    "from paiec import evaluator\n",                        # unshipped, module level
    "from paiec.predict import Predictor\n",                # ships as paiec_rt, not paiec
    "def predict(i, l=None):\n    import paiec.data\n",     # unshipped, lazily
    "import scipy\n",                                       # optional, module level
])
def test_import_check_rejects(tmp_path, source):
    (tmp_path / "model.py").write_text(source)
    with pytest.raises(bs.BuildError):
        bs.check_imports({"model.py": str(tmp_path / "model.py")})


def test_build_refuses_missing_or_broken_prior(tmp_path):
    with pytest.raises(bs.BuildError):
        bs.members(str(tmp_path / "prior.json"))
    (tmp_path / "prior.json").write_text(json.dumps(
        {"coef": [0.0] * 9, "spec": {"providers": [], "efforts": [],
                                     "med_days": 400.0, "med_log_size": float("nan")}}))
    with pytest.raises(bs.BuildError):
        bs.check_prior(str(tmp_path / "prior.json"))


# --- the predict() contract ----------------------------------------------------

def test_official_inputs_give_native_floats_without_fallback(model, session):
    labeled, targets = session
    for B in BUDGETS:
        P = fresh(model)
        for inp, _ in targets[:40]:
            assert is_probability(P.predict(inp, labeled[B]))
        assert P.failures == 0, B
    assert is_probability(model.predict(targets[0][0], labeled[31]))


def test_edge_inputs_in_the_official_shape_need_no_fallback(model, session):
    labeled, targets = session
    s, it = targets[0][0]
    smoke = [subject("sample-model", "sample-org", "2026-01-01", "medium"),
             {"item_content": "A sample yes/no evaluation item.",
              "item_features": "tier=near-term", "interactors": ""}]
    elsewhere = [e for e in labeled[31] if e[0][1]["benchmark_id"] != it["benchmark_id"]]
    others = [e for e in labeled[31] if e[0][0] != s]
    P = fresh(model)
    for inp, lab in [(smoke, []), (smoke, None), (smoke, labeled[7]),
                     ([s, it], elsewhere), ([s, it], others), ([s, it], []),
                     ([s, dict(it, item_content="")], labeled[31]),
                     ([s, dict(it, item_content="x " * 75_000)], labeled[31]),
                     ([s, dict(it, item_content=None)], labeled[31]),
                     ([dict(s, release_date="2025-07-11T10:00:00Z"), it], labeled[3])]:
        assert is_probability(P.predict(inp, lab))
    assert is_probability(P.predict(smoke))
    assert P.failures == 0


class Unprintable:
    def __str__(self):
        raise ValueError("no text")


def test_malformed_inputs_never_raise_and_malformed_entries_are_dropped(model, session):
    labeled, targets = session
    s, it = targets[0][0]
    junk = [None, 3, "x", [1, 2], [[{}, {}], 0.5], [[s, it], float("nan")], [[s, it], "1"],
            [[s, it], b"1"], [[s, it], None], [[s, it], [1]], [[None, it], 1], [[s, it, it], 1],
            {"a": 1}, [[dict(s, normalized_name=Unprintable()), it], 1]]
    P = fresh(model)
    for inp, lab in [([s, it], junk), ([s, it], labeled[15] + junk), ([s, it], "labels"),
                     ([s, it], {"a": 1}), ([s, it], 3), (None, None), ([], []), ([s], []),
                     (["a", "b"], []), ([s, None], []), ([None, it], []),
                     ([dict.fromkeys(s, None), it], []),
                     ([{k: 7 for k in s}, dict(it, item_content=12345)], labeled[1])]:
        assert is_probability(P.predict(inp, lab)), (inp, lab)
    for B in (0, 15, 31):
        P = fresh(model)
        clean = P.predict([s, it], labeled[B])
        assert P.predict([s, it], labeled[B] + junk) == clean, B
        assert P.predict([s, it], junk + labeled[B][::-1]) == clean, B
        numeric = [[e[0], (np.int64, float, bool)[k % 3](e[1])] for k, e in enumerate(labeled[B])]
        assert P.predict([s, it], numeric) == clean, B
        assert P.failures == 0, B


def test_prediction_is_a_pure_function_of_input_and_labeled(model, session):
    labeled, targets = session
    probe = [inp for inp, _ in targets[:25]]
    P = fresh(model)
    ref = [P.predict(inp, labeled[31]) for inp in probe]
    assert any(f.w is not None for ev in P._evidence.values() for f in ev._fits.values())

    P = fresh(model)
    for B in (31, 0, 1, 3, 7, 15):         # labeled[31] falls out of the cache
        for inp, _ in targets[100:110]:
            P.predict(inp, labeled[B])
    assert [P.predict(inp, labeled[31]) for inp in reversed(probe)] == ref[::-1]

    shuffled = [labeled[31][i] for i in np.random.default_rng(1).permutation(len(labeled[31]))]
    rebuilt = json.loads(json.dumps(shuffled))           # new objects, same content
    P = fresh(model)
    assert [P.predict(copy.deepcopy(inp), rebuilt) for inp in probe] == ref
    assert P.failures == 0


def test_keys_cover_full_item_and_every_subject_field(model):
    pr = model.loaded[f"{bs.RUNTIME}.predict"]
    boiler = "Let n be a positive integer. " * 20
    a, b = item(boiler + "Find n."), item(boiler + "Find 2n.")
    assert a["item_content"][:400] == b["item_content"][:400]
    assert pr.item_key(a) != pr.item_key(b)
    assert pr.item_key(a) == pr.item_key(json.loads(json.dumps(a)))
    for field, value in [("item_features", "tier=2"), ("interactors", "tool"),
                         ("benchmark_id", "bench-other")]:
        assert pr.item_key(dict(a, **{field: value})) != pr.item_key(a), field
    s = subject("m")
    assert all(pr.subject_key(dict(s, **{f: s[f] + "x"})) != pr.subject_key(s)
               for f in bs.SUBJECT_FIELDS)


def test_speed_500_targets_against_12_pairs_of_31_labels(model, long_session):
    labeled, targets = long_session
    assert len(labeled[31]) == 372
    P = fresh(model)
    t = time.time()
    out = [P.predict(inp, labeled[31]) for inp, _ in targets[:500]]
    elapsed = time.time() - t
    assert all(map(is_probability, out)) and P.failures == 0
    assert any(f.w is not None for ev in P._evidence.values() for f in ev._fits.values()), \
        "the item-difficulty fit never ran, so this timed the cheap path"
    assert elapsed < 60, f"{elapsed:.1f}s"


# --- a fresh interpreter, as each evaluation worker is -------------------------

WORKER = r"""
import importlib.util, json, math, sys, types
root, block = sys.argv[1], set(sys.argv[2].split(",")) - {""}
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in block:
            raise ImportError(f"{name} is not installed here")
sys.meta_path.insert(0, Block())
sys.modules["paiec"] = types.ModuleType("paiec")
sys.modules["paiec"].__path__ = []
before = list(sys.path)
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("submission_model", root + "/model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
sys.path[:] = before
heavy = sorted({n.split(".")[0] for n in sys.modules} & {"scipy", "sklearn", "pandas"})
labeled, targets = json.load(open(sys.argv[3]))
subject, item = targets[0][0]
own = lambda y: [[[subject, dict(item, item_content=f"q{j}")], y] for j in range(20)]
out = dict(heavy_on_import=heavy,
           prior=model.predict([subject, item], []),
           strong=model.predict([subject, item], own(1)),
           weak=model.predict([subject, item], own(0)),
           rest=[model.predict(inp, labeled) for inp, _ in targets[:30]],
           fitted=any(f.w is not None for ev in model.PREDICTOR._evidence.values()
                      for f in ev._fits.values()),
           failures=model.PREDICTOR.failures)
print(json.dumps(out))
"""


@pytest.mark.parametrize("blocked", ["", "scipy,sklearn,pandas"])
def test_worker_imports_only_numpy_and_degrades_without_optional_packages(
        archive, session, tmp_path, blocked):
    """A fresh worker whose process already holds a package called paiec of its
    own, with and without scipy, scikit-learn and pandas."""
    labeled, targets = session
    data = tmp_path / "session.json"
    data.write_text(json.dumps([labeled[31], targets[:40]]))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", WORKER, str(archive[1]), blocked, str(data)],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["heavy_on_import"] == []
    assert out["failures"] == 0, r.stderr
    assert out["fitted"] == (not blocked)
    assert all(map(is_probability, [out["prior"], out["strong"], out["weak"]] + out["rest"]))
    assert out["weak"] < out["prior"] < out["strong"]


COLD = r"""
import importlib.util, json, sys, time
root, data, runtime = sys.argv[1], sys.argv[2], sys.argv[5]
start, flip = int(sys.argv[3]), int(sys.argv[4])
before = list(sys.path)
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("submission_model", root + "/model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
sys.path[:] = before
pr = sys.modules[runtime + ".predict"]
threads, fit_joint = [], pr.fit_joint
def spy(*a, **kw):
    from threadpoolctl import threadpool_info
    threads.append({d["filepath"]: d["num_threads"] for d in threadpool_info()})
    return fit_joint(*a, **kw)
pr.fit_joint = spy
labeled, targets = json.load(open(data))
bids = sorted({t[1]["benchmark_id"] for t in targets})
rank = lambda i: (bids.index(targets[i][1]["benchmark_id"]) - start) % len(bids)
order = sorted(range(len(targets)), key=lambda i: (rank(i), -i if flip else i))
t = time.perf_counter()
preds = {order[0]: model.predict(targets[order[0]], labeled)}
first = time.perf_counter() - t
preds.update({i: model.predict(targets[i], labeled) for i in order[1:]})
print(json.dumps(dict(first=first, preds=[preds[i].hex() for i in range(len(targets))],
                      threads=threads, failures=model.PREDICTOR.failures)))
"""


def test_cold_workers_side_by_side_fit_on_one_thread_and_agree_bit_for_bit(
        archive, long_session, tmp_path):
    """Four fresh workers at once, as at a checkpoint, each fitting the four
    benchmarks in another order, with the environment asking for a thread per
    core. Every fit must run on one thread in every BLAS and OpenMP library, the
    first call (imports and the long-text fit included) must stay quick, and
    the four must answer bit for bit alike: a first fit on more threads than
    the later ones made predictions depend on which benchmark came first."""
    pytest.importorskip("threadpoolctl")
    labeled, targets = long_session
    probe = [inp for inp, _ in targets[:60]]
    assert len({it["benchmark_id"] for _, it in probe}) == 4
    data = tmp_path / "session.json"
    data.write_text(json.dumps([labeled[31], probe]))
    n = str(max(4, os.cpu_count() or 1))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(OPENBLAS_NUM_THREADS=n, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n)
    procs = [subprocess.Popen([sys.executable, "-c", COLD, str(archive[1]), str(data), str(k),
                               str(k % 2), bs.RUNTIME], cwd=tmp_path, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in range(4)]
    outs = []
    for p in procs:
        stdout, stderr = p.communicate(timeout=300)
        assert p.returncode == 0, stderr
        outs.append(json.loads(stdout.strip().splitlines()[-1]))
    for o in outs:
        assert o["failures"] == 0
        assert len(o["threads"]) == 4, "a benchmark was not fitted"
        assert all(t == 1 for fit in o["threads"] for t in fit.values()), o["threads"]
    assert max(o["first"] for o in outs) < 20, [round(o["first"], 2) for o in outs]
    assert all(o["preds"] == outs[0]["preds"] for o in outs[1:])


def test_labeling_reproduces_the_platform_default():
    import hashlib
    lab = _load("submission_labeling", ROOT / "submission" / "labeling.py")
    rng = np.random.default_rng(0)
    for n in range(300):
        items_left = int(rng.integers(1, 90))
        ctx = {"subject_id": f"s{n % 7}", "benchmark_id": f"b{n % 3}",
               "labels_acquired": int(rng.integers(0, 31)), "max_labels": 31,
               "labels_remaining": int(rng.integers(0, 32)), "items_remaining": items_left}
        inp = [subject(f"m{n % 5}"), item(f"question {n}")]
        key = json.dumps([ctx, inp], sort_keys=True, separators=(",", ":"))
        u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
        got = lab.acquisition_function(copy.deepcopy(inp), labeled=[], context=copy.deepcopy(ctx))
        assert type(got) is bool and got == (u < min(1, ctx["labels_remaining"] / items_left))
