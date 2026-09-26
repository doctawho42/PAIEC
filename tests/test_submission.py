"""The submission archive as the platform runs it.

The archive is staged from the member list tools/build_submission.py zips and
model.py is loaded the way the organisers' validator loads it, with the repo's
own paiec out of sys.modules so it cannot stand in for a module the archive
forgot. What only shows in a fresh process (a worker's first call, its thread
limits, several workers side by side, a missing optional package, an unusable
prior.json) runs in fresh interpreters. No data needed: the bundle is
submission/prior.json when a build has written a hier one and a small synthetic
stand-in otherwise, and the inputs are synthetic, in the official format.
"""
import copy
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from paiec import prior as PR
from paiec.hier import PUBLIC, Flags, HierPredictor, Hyper
from paiec.subjects import Spec

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
LEVEL = bs.shipped_level()


def subject(name, provider="openai", release="2025-03-01", effort="high", **kw):
    s = dict.fromkeys(bs.SUBJECT_FIELDS, "")
    s.update(normalized_name=name, provider=provider, release_date=release,
             access_date="2026-01-15", reasoning_effort=effort, **kw)
    return s


def item(text, bid="bench-7f3a", features="tier=1"):
    return {"item_content": text, "item_features": features, "interactors": "",
            "benchmark_id": bid}


def subjects(n):
    """Three hand-made subjects (one with no date or effort), then generated ones."""
    base = [subject("acme-large 70b", "openai", "2025-06-01"),
            subject("acme-mini 8b", "anthropic", "2024-05-13", "low"),
            subject("other-model", "someorg", "", "")]
    provs, efforts = ("openai", "anthropic", "google", "someorg"), ("low", "medium", "high", "")
    more = [subject(f"model-{k} {8 * (k % 9 + 1)}b", provs[k % 4],
                    f"202{4 + k % 3}-{k % 12 + 1:02d}-01", efforts[k % 4])
            for k in range(max(0, n - 3))]
    return (base + more)[:n]


def make_session(seed=0, n_items=160, long_words=60, n_bench=4, n_subjects=3):
    """Anonymous benchmarks x subjects, 50/50 split per pair, the first B
    acquisition labels of every pair shared at budget B.

    The first benchmark's items run to `long_words` words (14000 gives the
    ~90k characters of researchcodebench), the second's share a 420-character
    boilerplate prefix, the third's are multiple choice.
    """
    rng = np.random.default_rng(seed)
    vocab = np.array([f"w{i}" for i in range(4000)])
    subs = subjects(n_subjects)
    theta = rng.normal(0, 1, len(subs))
    boiler = "You are given a competition problem. " * 12
    labeled = {B: [] for B in BUDGETS}
    targets = []
    for b in range(n_bench):
        bid = f"b-{rng.integers(16**12):012x}"
        level = rng.normal(-1, 1)
        diff = rng.normal(0, 1.2, n_items)
        items = []
        for j in range(n_items):
            words = " ".join(rng.choice(vocab, long_words if b == 0 else 60))
            mcq = words + "\nA) one\nB) two\nC) three\nD) four"
            text = {1: boiler + words, 2: mcq}.get(b, words)
            items.append(item(text, bid, f"tier={j % 3}"))
        for s, sub in enumerate(subs):
            y = (rng.random(n_items) < 1 / (1 + np.exp(diff - theta[s] - level))).astype(int)
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
def dense_session():
    """One benchmark, every pair of it: multi_swebench's 82 subjects."""
    return make_session(seed=1, n_items=200, n_bench=1, n_subjects=82)


def toy_bundle(path, level=LEVEL, **hyper):
    """A hier bundle on a synthetic subject prior, marked as fitted on all of
    PUBLIC like the shipped one."""
    spec = Spec(["openai", "anthropic"], ["high"], 500.0, 3.0)
    coef = np.array([0.1, 0.8, 0.0, 0.2, 0.1, -0.3, 0.3, 0.2, 0.1, 0.4, -0.2, 0.3])
    prior = PR.SubjectPrior(coef, spec, 0.01 * np.eye(len(coef)), {"acme large 70b": [0.9, 0.6, 2]},
                            {"raw": 2.0, "resid": 2.2},
                            {"benchmarks": list(PUBLIC[:4]), "included": list(PUBLIC),
                             "excluded": []})
    PR.save(str(path), prior, replace(Hyper(), **level, **hyper))
    return path


def legacy_prior(path):
    spec = Spec(["openai", "anthropic"], ["high"], 500.0, 3.0)
    path.write_text(json.dumps({"coef": [0.1, 0.3, 0.0, 0.2, 0.1, -0.3, 0.3, 0.2, 0.1,
                                         0.2, 0.1, 0.1], "spec": spec.to_dict()}))
    return path


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    built = ROOT / "submission" / "prior.json"
    try:
        bs.check_prior(str(built), "hier", LEVEL)
        return built
    except bs.BuildError:
        return toy_bundle(tmp_path_factory.mktemp("bundle") / "prior.json")


def stage(tmp, prior):
    """Zip the build's members with this prior.json and extract the archive."""
    files = bs.members(str(prior))
    zpath = tmp / "paiec.zip"
    bs.write_zip(str(zpath), files)
    root = tmp / "submission"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    return zpath, root.resolve(), files


@pytest.fixture(scope="module")
def archive(tmp_path_factory, bundle):
    bs.check_prior(str(bundle), "hier", LEVEL)
    return stage(tmp_path_factory.mktemp("archive"), bundle)


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


def is_probability(p):
    return type(p) is float and math.isfinite(p) and 0.0 <= p <= 1.0


def fitted(P):
    """Whether some labeled list was fitted (not only prior predictions)."""
    return any(getattr(f, "prob", None) is not None for f in P._fits.values())


def env():
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


# --- packaging ---------------------------------------------------------------

def test_organisers_validator_accepts_archive(archive):
    if not os.path.isfile(bs.VALIDATOR):
        pytest.skip("third_party/paiec_baseline not cloned")
    r = subprocess.run([sys.executable, bs.VALIDATOR, str(archive[0])], cwd=archive[0].parent,
                       env=env(), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("OK"), r.stdout + r.stderr


def test_every_loaded_module_comes_from_the_archive(model, archive):
    assert set(model.loaded) >= {f"{bs.RUNTIME}.{m}" for m in
                                 ("hier", "prior", "predict", "irt", "fitting", "subjects", "mcq")}
    for name, mod in model.loaded.items():
        assert Path(mod.__file__).resolve().is_relative_to(archive[1]), name
    assert model.leaked == []
    assert type(model.PREDICTOR) is model.loaded[f"{bs.RUNTIME}.hier"].HierPredictor
    bs.check_imports(archive[2])


def test_archive_ships_the_calibrated_hier(model, bundle):
    """The shipped configuration: LEVEL over the fitted hyperparameters, a
    Gaussian level, every Flags switch at its default, an attribute prior,
    and one LEVEL for the build and model.py's fallback."""
    assert model.MODEL == "hier" and model.LEVEL == LEVEL
    P = model.make()
    assert {k: getattr(P.hyper, k) for k in LEVEL} == LEVEL
    assert P.hyper.nu_mu == 0 and P.hyper.excluded == () and set(P.hyper.included) == set(PUBLIC)
    assert asdict(P.cfg) == asdict(Flags()) and not P.cfg.text
    assert P.prior is not None and P.prior.has_attributes
    prior, hyper = PR.load(str(bundle))
    assert P.hyper.to_dict() == hyper.to_dict() and P.prior.to_dict() == prior.to_dict()


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


def test_only_prior_main_may_import_the_offline_modules(tmp_path):
    """prior.py ships whole; its main() reads the public data, anything else
    that did would break on the platform."""
    src = (ROOT / "paiec" / "prior.py").read_text()
    bad = tmp_path / "prior.py"
    bad.write_text(src + "\n\ndef _more():\n    from .data import load_pairs\n")
    ok = {f"{bs.RUNTIME}/{n}": str(ROOT / "paiec" / n) for n in bs.PACKAGE}
    bs.check_imports(ok)
    with pytest.raises(bs.BuildError, match="_more|data"):
        bs.check_imports({**ok, f"{bs.RUNTIME}/prior.py": str(bad)})


def test_build_refuses_missing_broken_or_other_priors(tmp_path):
    with pytest.raises(bs.BuildError):
        bs.members(str(tmp_path / "prior.json"))
    broken = tmp_path / "legacy.json"
    broken.write_text(json.dumps(
        {"coef": [0.0] * 9, "spec": {"providers": [], "efforts": [],
                                     "med_days": 400.0, "med_log_size": float("nan")}}))
    with pytest.raises(bs.BuildError):
        bs.check_prior(str(broken))
    good = toy_bundle(tmp_path / "good.json")
    assert bs.check_prior(str(good), "hier", LEVEL) == "hier"
    assert bs.check_prior(str(legacy_prior(tmp_path / "old.json")), "legacy") == "legacy"
    cases = {"legacy.json when hier is asked": legacy_prior(tmp_path / "l2.json"),
             "another level": toy_bundle(tmp_path / "lv.json", {**LEVEL, "mu0": -1.263}),
             "a Student-t level": toy_bundle(tmp_path / "t.json", nu_mu=3.0),
             "a fit leaving a benchmark out": toy_bundle(tmp_path / "ex.json",
                                                         excluded=("matharena",)),
             "not JSON": tmp_path / "junk.json"}
    cases["not JSON"].write_text("{")
    nan = json.loads(good.read_text())
    nan["hyper"]["sigma_delta"] = float("nan")
    cases["a NaN width"] = tmp_path / "nan.json"
    cases["a NaN width"].write_text(json.dumps(nan))
    for what, path in cases.items():
        with pytest.raises(bs.BuildError):
            bs.check_prior(str(path), "hier", LEVEL)
            pytest.fail(what)


# --- a fresh interpreter, as each evaluation worker is -------------------------

LOADER = r"""
import importlib.util, json, math, sys
root, data = sys.argv[1], json.load(open(sys.argv[2]))
before = list(sys.path)
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("submission_model", root + "/model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
sys.path[:] = before
out = dict(model=model.MODEL, cls=type(model.PREDICTOR).__name__, preds={}, failures=0)
for B, labeled in data["labeled"].items():
    P = model.make()
    out["preds"][B] = [P.predict(inp, labeled) for inp, _ in data["targets"]]
    out["failures"] += P.failures
P = model.make()
out["prior_is_none"] = getattr(P, "prior", 0) is None
out["level"] = {k: getattr(P.hyper, k) for k in model.LEVEL} if hasattr(P, "hyper") else None
out["smoke"] = model.predict([{"normalized_name": "sample-model"}, {"item_content": "x"}])
print(json.dumps(out))
"""


def load_fresh(root, session, tmp_path, n=30):
    labeled, targets = session
    data = tmp_path / "loader.json"
    data.write_text(json.dumps({"labeled": {str(B): labeled[B] for B in BUDGETS},
                                "targets": targets[:n]}))
    r = subprocess.run([sys.executable, "-c", LOADER, str(root), str(data)], cwd=tmp_path,
                       env=env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(map(is_probability, [p for ps in out["preds"].values() for p in ps]))
    assert is_probability(out["smoke"])
    return out, r.stderr


def copy_with_prior(archive, tmp_path, prior=None, text=None):
    root = tmp_path / "submission"
    shutil.copytree(archive[1], root)
    if prior is not None:
        shutil.copy(prior, root / "prior.json")
    if text is not None:
        (root / "prior.json").write_text(text)
    return root


@pytest.mark.parametrize("text", ["{", json.dumps({"kind": PR.KIND, "prior": None,
                                                     "hyper": {"sigma_mu": -1}}), "[]"])
def test_unusable_prior_json_falls_back_to_the_level_prior(archive, session, tmp_path, text):
    out, err = load_fresh(copy_with_prior(archive, tmp_path, text=text), session, tmp_path)
    assert out["model"] == "hier without prior.json" and out["cls"] == "HierPredictor"
    assert out["prior_is_none"] and out["level"] == LEVEL and out["failures"] == 0
    assert "prior.json unusable" in err


def test_a_legacy_prior_json_ships_the_predictor(archive, session, tmp_path):
    root = copy_with_prior(archive, tmp_path, prior=legacy_prior(tmp_path / "legacy.json"))
    out, _ = load_fresh(root, session, tmp_path)
    assert out["model"] == "predictor" and out["cls"] == "Predictor" and out["failures"] == 0


def test_a_fresh_worker_agrees_with_this_process(model, archive, session, tmp_path):
    out, _ = load_fresh(archive[1], session, tmp_path)
    labeled, targets = session
    for B in BUDGETS:
        P = model.make()
        assert [P.predict(inp, labeled[B]) for inp, _ in targets[:30]] == out["preds"][str(B)]


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
heavy = lambda: sorted({n.split(".")[0] for n in sys.modules} & {"scipy", "sklearn", "pandas"})
on_import = heavy()
labeled, targets = json.load(open(sys.argv[3]))
subject, item = targets[0][0]
own = lambda y: [[[subject, dict(item, item_content=f"q{j}")], y] for j in range(20)]
out = dict(heavy_on_import=on_import,
           prior=model.predict([subject, item], []),
           strong=model.predict([subject, item], own(1)),
           weak=model.predict([subject, item], own(0)),
           rest=[model.predict(inp, labeled).hex() for inp, _ in targets[:30]],
           fitted=any(getattr(f, "prob", None) is not None
                      for f in model.PREDICTOR._fits.values()),
           failures=model.PREDICTOR.failures, heavy_after=heavy())
print(json.dumps(out))
"""


def test_worker_needs_only_numpy_and_predicts_alike_without_optional_packages(
        archive, session, tmp_path):
    """A fresh worker whose process already holds a package called paiec of its
    own, with and without scipy, scikit-learn and pandas: hier loads none of
    them, at import or while predicting, so both answer bit for bit alike."""
    labeled, targets = session
    data = tmp_path / "session.json"
    data.write_text(json.dumps([labeled[31], targets[:40]]))
    outs = []
    for blocked in ("", "scipy,sklearn,pandas"):
        r = subprocess.run([sys.executable, "-c", WORKER, str(archive[1]), blocked, str(data)],
                           cwd=tmp_path, env=env(), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout.strip().splitlines()[-1])
        assert out["heavy_on_import"] == [] and out["heavy_after"] == []
        assert out["failures"] == 0, r.stderr
        assert out["fitted"]
        assert all(map(is_probability, [out["prior"], out["strong"], out["weak"]]))
        assert out["weak"] < out["prior"] < out["strong"]
        outs.append(out)
    assert outs[0]["rest"] == outs[1]["rest"]


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
hier = sys.modules[runtime + ".hier"]
threads, solve = [], hier.Problem.solve
def spy(self, *a, **kw):
    from threadpoolctl import threadpool_info
    threads.append({d["filepath"]: d["num_threads"] for d in threadpool_info()})
    return solve(self, *a, **kw)
hier.Problem.solve = spy
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
    """Four fresh workers at once, as at a checkpoint, each serving the four
    benchmarks' targets in another order, with the environment asking for a
    thread per core. Every fit must run on one thread in every BLAS and OpenMP
    library, the first call (imports and the fit of the long texts included)
    must stay quick, and the four must answer bit for bit alike."""
    pytest.importorskip("threadpoolctl")
    labeled, targets = long_session
    probe = [inp for inp, _ in targets[:60]]
    assert len({it["benchmark_id"] for _, it in probe}) == 4
    data = tmp_path / "session.json"
    data.write_text(json.dumps([labeled[31], probe]))
    n = str(max(4, os.cpu_count() or 1))
    e = env()
    e.update(OPENBLAS_NUM_THREADS=n, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n)
    procs = [subprocess.Popen([sys.executable, "-c", COLD, str(archive[1]), str(data), str(k),
                               str(k % 2), bs.RUNTIME], cwd=tmp_path, env=e,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in range(4)]
    outs = []
    for p in procs:
        stdout, stderr = p.communicate(timeout=300)
        assert p.returncode == 0, stderr
        outs.append(json.loads(stdout.strip().splitlines()[-1]))
    for o in outs:
        assert o["failures"] == 0
        assert o["threads"], "nothing was fitted"
        assert all(t == 1 for fit in o["threads"] for t in fit.values()), o["threads"]
    assert max(o["first"] for o in outs) < 20, [round(o["first"], 2) for o in outs]
    assert all(o["preds"] == outs[0]["preds"] for o in outs[1:])


# --- the predict() contract ----------------------------------------------------

def test_official_inputs_give_native_floats_without_fallback(model, session):
    labeled, targets = session
    for B in BUDGETS:
        P = model.make()
        for inp, _ in targets[:40]:
            assert is_probability(P.predict(inp, labeled[B]))
        assert P.failures == 0 and P.unconverged == 0, B
        assert fitted(P) == (B > 0), B
    assert is_probability(model.predict(targets[0][0], labeled[31]))


def test_archive_predicts_bit_for_bit_like_the_research_package(model, bundle, session):
    """The archive's predictor and paiec.hier's own, on the same prior.json,
    at every budget and with the labels in another order."""
    labeled, targets = session
    probe = [inp for inp, _ in targets[:60]]
    for B in BUDGETS:
        ours, theirs = model.make(), HierPredictor(*PR.load(str(bundle)))
        got = [ours.predict(inp, labeled[B]) for inp in probe]
        assert got == [theirs.predict(inp, labeled[B][::-1]) for inp in probe], B
        assert ours.failures == theirs.failures == 0


def test_edge_inputs_in_the_official_shape_need_no_fallback(model, session):
    labeled, targets = session
    s, it = targets[0][0]
    smoke = [subject("sample-model", "sample-org", "2026-01-01", "medium"),
             {"item_content": "A sample yes/no evaluation item.",
              "item_features": "tier=near-term", "interactors": ""}]
    elsewhere = [e for e in labeled[31] if e[0][1]["benchmark_id"] != it["benchmark_id"]]
    others = [e for e in labeled[31] if e[0][0] != s]
    P = model.make()
    for inp, lab in [(smoke, []), (smoke, None), (smoke, labeled[7]),
                     ([s, it], elsewhere), ([s, it], others), ([s, it], []),
                     ([s, dict(it, item_content="")], labeled[31]),
                     ([s, dict(it, item_content="x " * 75_000)], labeled[31]),
                     ([s, dict(it, item_content=None)], labeled[31]),
                     ([s, dict(it, item_features="k=" + "v" * 50_000)], labeled[31]),
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
    P = model.make()
    for inp, lab in [([s, it], junk), ([s, it], labeled[15] + junk), ([s, it], "labels"),
                     ([s, it], {"a": 1}), ([s, it], 3), (None, None), ([], []), ([s], []),
                     (["a", "b"], []), ([s, None], []), ([None, it], []),
                     ([dict.fromkeys(s, None), it], []),
                     ([{k: 7 for k in s}, dict(it, item_content=12345)], labeled[1])]:
        assert is_probability(P.predict(inp, lab)), (inp, lab)
    for B in (0, 15, 31):
        P = model.make()
        clean = P.predict([s, it], labeled[B])
        assert P.predict([s, it], labeled[B] + junk) == clean, B
        assert P.predict([s, it], junk + labeled[B][::-1]) == clean, B
        numeric = [[e[0], (np.int64, float, bool)[k % 3](e[1])] for k, e in enumerate(labeled[B])]
        assert P.predict([s, it], numeric) == clean, B
        assert P.failures == 0, B


def test_a_broken_fit_falls_back_to_the_prior_then_to_one_half(model, session, monkeypatch):
    """HierPredictor never raises: a failed fit answers the prediction without
    labels and counts a failure, a failed prior prediction answers 0.5, and
    model.predict answers 0.5 when there is no predictor at all."""
    labeled, targets = session
    inp = targets[0][0]
    no_labels = model.make().predict(inp, [])
    P = model.make()

    def boom(*a, **kw):
        raise RuntimeError("fit failed")

    P.fit_for = boom
    assert P.predict(inp, labeled[31]) == no_labels and P.failures == 1
    P.combine = boom
    assert P.predict(inp, labeled[31]) == 0.5 and P.failures == 2
    monkeypatch.setattr(model.PREDICTOR, "predict", boom)
    assert model.predict(inp, labeled[31]) == 0.5
    monkeypatch.setattr(model, "PREDICTOR", None)
    assert model.predict(inp, labeled[31]) == 0.5


def test_prediction_is_a_pure_function_of_input_and_labeled(model, session):
    labeled, targets = session
    probe = [inp for inp, _ in targets[:25]]
    P = model.make()
    ref = [P.predict(inp, labeled[31]) for inp in probe]
    assert fitted(P)

    P = model.make()
    for B in (31, 0, 1, 3, 7, 15):         # labeled[31] falls out of the cache
        for inp, _ in targets[100:110]:
            P.predict(inp, labeled[B])
    assert len(P._fits) < 6
    assert [P.predict(inp, labeled[31]) for inp in reversed(probe)] == ref[::-1]

    shuffled = [labeled[31][i] for i in np.random.default_rng(1).permutation(len(labeled[31]))]
    rebuilt = json.loads(json.dumps(shuffled))           # new objects, same content
    P = model.make()
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


# --- latency -------------------------------------------------------------------

def timed(P, targets, labeled):
    times, out = [], []
    for inp, _ in targets:
        t = time.perf_counter()
        out.append(P.predict(inp, labeled))
        times.append(time.perf_counter() - t)
    assert all(map(is_probability, out)) and P.failures == 0
    assert fitted(P), "nothing was fitted, so this timed the prior alone"
    return times


def test_speed_formative_500_targets_against_12_pairs_of_31_labels(model, long_session):
    """A formative checkpoint at its largest: 12 pairs, 372 labels, texts up to
    90k characters. A worker's first call fits; every later one reads the fit."""
    labeled, targets = long_session
    assert len(labeled[31]) == 372
    times = timed(model.make(), targets[:500], labeled[31])
    assert times[0] < 10, f"first call {times[0]:.2f}s"
    assert sum(times) < 30, f"{sum(times):.1f}s for 500 calls"


def test_speed_dense_one_benchmark_every_pair(model, dense_session):
    """Dense multi_swebench's shape: 82 subjects on one benchmark, 2,542 labels
    shared by every target."""
    labeled, targets = dense_session
    assert len(labeled[31]) == 82 * 31
    P = model.make()
    times = timed(P, targets[:200], labeled[31])
    assert P.unconverged == 0
    assert times[0] < 20, f"first call {times[0]:.2f}s"
    assert np.mean(times[1:]) < 0.1, f"{1e3 * np.mean(times[1:]):.1f} ms a call"


# --- acquisition -----------------------------------------------------------------

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
