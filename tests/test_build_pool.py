"""tools/build_submission.py --pool, archive-4's build path (docs/plans/p2_pool_eval.md
section 10), on a synthetic pool, a synthetic lock and a stand-in archive-3. No data is
needed and no real archive-4 is built.

What is pinned: the settings --pool refuses to run without; the lock and the gate's
record it reads; paiec/pool.py as the locked file or a recorded successor by the AST
rule (PE 9 item 11); X*'s bundle rebuilt with nothing excluded, byte for byte the
locked one; its level; the pool bundle checks (L, nothing excluded, archive-3's guards
and keys); model.py as archive-3's with only its docstring, the comment above LEVEL
and LEVEL's value rewritten, under the AST rule (a docstring or comment change passes,
a code change fails); the archive differing from archive-3 in prior.json and model.py
only; the run check and the validator on the built archive; nothing destroyed on a
refusal; and the default path still zipping archive-3's bytes.
"""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from paiec import pool as P
from tests.test_pool import standard_specs, write_pool   # synthetic formatted_tables

ROOT = Path(__file__).resolve().parents[1]
#: archive-3's submission/model.py and prior.json (PE 2, 9 item 10)
A3_MODEL_SHA256 = "a9561290e62d3f7c65935c860b7f4300359cc5df34c6bccc207eb85305dfd82e"
A3_PRIOR_SHA256 = "c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770"
MU0_M = -2.104
BENCH = json.loads((ROOT / "tools" / "pool_benchmarks.json").read_text())
LEVELS = {"C1": dict(P.LEVEL_A3), "C1m": {"mu0": -2.035, "sigma_mu": 2.5, "attr_scale": 0.5},
          "C2": {"mu0": 0.16, "sigma_mu": 2.31, "attr_scale": 0.5}}


def _load(name, path):
    before = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = before


bs = _load("build_submission_pool", ROOT / "tools" / "build_submission.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def base_model():
    """archive-3's model.py: the repository's while it is archive-3's, else the
    copy in dist/archive-3.zip."""
    src = (ROOT / "submission" / "model.py").read_bytes()
    if sha(src) == A3_MODEL_SHA256:
        return src.decode()
    if os.path.isfile(bs.ARCHIVE_3):
        with zipfile.ZipFile(bs.ARCHIVE_3) as z:
            src = z.read("model.py")
        if sha(src) == A3_MODEL_SHA256:
            return src.decode()
    pytest.skip("no copy of archive-3's model.py")


# --- a synthetic pool, its fits, bundles, lists and lock, and a stand-in archive-3 ------

class World:
    pass


@pytest.fixture(scope="module")
def world(tmp_path_factory, base_model):
    tmp = tmp_path_factory.mktemp("pool_build")
    mp = pytest.MonkeyPatch()
    mp.setattr(P, "FITS_DIR", str(tmp / "fits"))
    mp.setattr(P, "EVAL_DIR", str(tmp / "eval"))
    mp.setattr(P, "TOOLS", str(tmp / "tools"))
    for c in (P._RESP, P._VERIFIED, P._MANIFESTS):
        c.clear()
    try:
        w = World()
        w.tmp, w.root = tmp, write_pool(tmp / "pool", standard_specs())
        L, _, _ = P.select(w.root)
        P.fit_all(L, 1, w.root, log=lambda *_: None, fresh=False)
        P.write_lists(w.root, P.TOOLS)
        lists = Path(P.TOOLS) / "pool_benchmarks.json"
        bench = json.loads(lists.read_text())
        bench["U"] = [b for b in bench["L"] if b != "lowbet"]          # a flagged stand-in
        lists.write_bytes(P._json_bytes(bench))
        w.bench, w.L, w.U = bench, bench["L"], bench["U"]
        w.free, w.fixed, _ = P.load_fits(w.root)
        w.bundles = P.archive_bundles(w.free, w.fixed, w.L, w.U, MU0_M)
        # archive-3's stand-in: archive-3's model.py, another fit's hier bundle as prior.json
        a3 = tmp / "a3"
        a3.mkdir()
        (a3 / "model.py").write_text(base_model)
        p, h, _ = P.candidate_bundle("C1", ("alpha",), w.free, w.fixed, w.L, w.U)
        w.a3_prior = P.to_bytes(p, h)
        (a3 / "prior.json").write_bytes(w.a3_prior)
        files = bs.members(str(a3 / "prior.json"))
        files["model.py"] = str(a3 / "model.py")
        w.archive3 = tmp / "dist" / "archive-3.zip"
        bs.write_zip(str(w.archive3), files)
        w.a3_sha = sha(w.archive3.read_bytes())
        src = Path(P.__file__).read_bytes()
        w.lock = {
            "files": {bs.LOCK_POOL_PY: sha(src), bs.LOCK_LISTS: sha(lists.read_bytes()),
                      bs.LOCK_FITS_INDEX: P.fits_index_sha256(), bs.LOCK_ARCHIVE_3: w.a3_sha,
                      **{bs.LOCK_BUNDLE.format(X): v[1] for X, v in w.bundles.items()}},
            "pool_frozen_ast": sha(P.frozen_ast(src.decode()).encode()),
            "eval_loader": list(P.EVAL_LOADER), "mu0_m": MU0_M,
            "c2_literal": w.bundles["C2"][2],
            "bundles": {X: {"sha256": v[1], "level": v[2]} for X, v in w.bundles.items()},
            "env": {"PYTHONHASHSEED": "0", **{v: "1" for v in bs.POOL_BLAS}}}
        yield w
    finally:
        mp.undo()
        for c in (P._RESP, P._VERIFIED, P._MANIFESTS):
            c.clear()


@pytest.fixture
def ready(world, monkeypatch):
    """The lock's settings as this process would show them, and the stand-in's digest."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for v in bs.POOL_BLAS:
        monkeypatch.setenv(v, "1")
    monkeypatch.setattr(bs, "_hash_randomization", lambda: 0)
    monkeypatch.setattr(bs, "ARCHIVE_3_SHA256", world.a3_sha)
    return world


def setup(world, tmp_path, X, lock=None, build=None, fixes=()):
    """The gate's record for X and results/pool_eval.json in tmp_path; their paths."""
    lock = copy.deepcopy(world.lock) if lock is None else lock
    build = {"candidate": X, "bundle_sha256": world.bundles[X][1], "level": world.bundles[X][2]} \
        if build is None else build
    ev, bd = tmp_path / "pool_eval.json", tmp_path / "pool_build.json"
    ev.write_text(json.dumps({"lock": lock, "fixes": list(fixes)}))
    bd.write_text(json.dumps(build))
    return {"eval_path": str(ev), "build_path": str(bd), "archive3": str(world.archive3),
            "prior_out": str(tmp_path / "prior.json")}


def build_pool(world, tmp_path, X, run_checks=False, candidate=None, model=None, pairs=None,
               out=None, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = setup(world, tmp_path, X, **kw)
    if model is None:
        model = tmp_path / "repo_model.py"
        model.write_bytes(zipfile.ZipFile(world.archive3).read("model.py"))
    out = str(tmp_path / "dist" / "paiec.zip") if out is None else out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    got = bs.pool_build(out, candidate, model=str(model), pairs=pairs, run_checks=run_checks,
                        **paths)
    return got, out, paths


# --- what --pool shares with paiec.pool and the design -----------------------------------

def test_the_constants_are_paiec_pools_and_the_designs():
    assert bs.POOL_BLAS == P.BLAS_VARS
    assert bs.LEVEL_KEYS == tuple(P.LEVEL_A3)
    assert tuple(P.CANDIDATES) == ("C1", "C1m", "C2")
    assert bs.ARCHIVE_3_SHA256 == "4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090"
    assert bs.POOL_CHANGED == ("model.py", "prior.json")
    assert bs.POOL_PY == os.path.join(str(ROOT), "paiec", "pool.py")


def test_the_settings_are_the_locks(monkeypatch):
    good = {"PYTHONHASHSEED": "0", **{v: "1" for v in bs.POOL_BLAS}}
    monkeypatch.setattr(bs, "_hash_randomization", lambda: 0)
    assert bs.pool_env_problems(good) == []
    assert len(bs.pool_env_problems({**good, "PYTHONHASHSEED": "1"})) == 1
    assert len(bs.pool_env_problems({k: v for k, v in good.items() if k != "PYTHONHASHSEED"})) == 1
    assert len(bs.pool_env_problems({**good, "OPENBLAS_NUM_THREADS": "4"})) == 1
    assert len(bs.pool_env_problems({})) == 1 + len(bs.POOL_BLAS)
    monkeypatch.setattr(bs, "_hash_randomization", lambda: 1)
    assert len(bs.pool_env_problems(good)) == 1


def test_pool_build_refuses_without_the_settings_and_leaves_the_output(world, tmp_path,
                                                                        monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for v in bs.POOL_BLAS:
        monkeypatch.setenv(v, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(bs, "_hash_randomization", lambda: 0)
    out = tmp_path / "dist" / "paiec.zip"
    out.parent.mkdir()
    out.write_bytes(b"archive-3 still here")
    with pytest.raises(bs.BuildError, match="OMP_NUM_THREADS"):
        build_pool(world, tmp_path, "C1", out=str(out))
    assert out.read_bytes() == b"archive-3 still here"


# --- paiec/pool.py: locked, or a recorded successor by the AST rule --------------------------

POOL_SRC = Path(P.__file__).read_text()


def _edit(src, old, new):
    assert src.count(old) == 1, old
    return src.replace(old, new)


IN_LOADER = ("def load_eval_pairs(b, subject_ids, keep=None, root=None) -> list:\n",
             "def load_eval_pairs(b, subject_ids, keep=None, root=None) -> list:\n    _fix = 1\n")
IN_BUNDLE = ("           literal=False):\n    \"\"\"PB(excl)",
             "           literal=False):\n    _fix = 1\n    \"\"\"PB(excl)")


def test_the_frozen_ast_copy_is_paiec_pools():
    for src in (POOL_SRC, _edit(POOL_SRC, *IN_LOADER), _edit(POOL_SRC, *IN_BUNDLE)):
        assert bs.pool_frozen_ast(src, P.EVAL_LOADER) == P.frozen_ast(src)


def _lock_for(src):
    return {"files": {bs.LOCK_POOL_PY: sha(src.encode())}, "eval_loader": list(P.EVAL_LOADER),
            "pool_frozen_ast": sha(P.frozen_ast(src).encode())}


def _fixes(new):
    rec = {"old": "0" * 64, "new": sha(new.encode())}
    return [{"cause": "toy", "files": {bs.LOCK_POOL_PY: rec}}]


def test_the_locked_pool_py_and_a_recorded_eval_loader_fix_pass():
    lock = _lock_for(POOL_SRC)
    assert bs.check_pool_py(POOL_SRC.encode(), lock, [], P.EVAL_LOADER) == "locked"
    fixed = _edit(POOL_SRC, *IN_LOADER)
    assert bs.check_pool_py(fixed.encode(), lock, _fixes(fixed), P.EVAL_LOADER) == "successor"


@pytest.mark.parametrize("case", ["unrecorded", "bundle", "constant", "tuple", "lock's tuple"])
def test_other_pool_py_changes_are_refused(case):
    lock, names = _lock_for(POOL_SRC), P.EVAL_LOADER
    if case == "unrecorded":
        new, fixes = _edit(POOL_SRC, *IN_LOADER), []
    elif case == "bundle":
        new = _edit(POOL_SRC, *IN_BUNDLE)
        fixes = _fixes(new)
    elif case == "constant":
        new = _edit(POOL_SRC, "\nK = 100\n", "\nK = 101\n")
        fixes = _fixes(new)
    elif case == "tuple":            # the fix widens EVAL_LOADER to take bundle in
        new = _edit(POOL_SRC, '"load_eval_pairs", "build_eval_indexes", "_eval_index_task")',
                    '"load_eval_pairs", "build_eval_indexes", "_eval_index_task", "bundle")')
        fixes, names = _fixes(new), names + ("bundle",)
    else:                            # the module's tuple is the lock's, the lock's record is not
        new, fixes = POOL_SRC, []
        lock["eval_loader"] = list(names) + ["bundle"]
    with pytest.raises(bs.BuildError):
        bs.check_pool_py(new.encode(), lock, fixes, names)


# --- model.py: the AST rule and the patch ---------------------------------------------------------

def _model_edits(base):
    lv = 'LEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}'
    return {
        "module docstring": (base.replace("Competition entry point", "Entry point", 1), True),
        "function docstring": (base.replace("A predictor as a new evaluation worker holds it",
                                            "A fresh predictor", 1), True),
        "comment": (base.replace("#: the level prior shipped", "#: the level shipped", 1)
                    .replace("import json  # noqa: E402", "import json  # noqa: E402  (std)", 1),
                    True),
        "LEVEL value": (base.replace(lv, lv.replace("-2.5", "0.16").replace("2.5,", "2.31,")),
                        True),
        "LEVEL key": (base.replace(lv, lv.replace('"sigma_mu"', '"sigma_theta"')), False),
        "LEVEL extra key": (base.replace(lv, lv[:-1] + ', "slip": 0.01}'), False),
        "thread setting": (base.replace('os.environ.setdefault(_var, "1")',
                                        'os.environ.setdefault(_var, "2")', 1), False),
        "an import": (base.replace("import json  # noqa: E402",
                                   "import json  # noqa: E402\nimport re", 1), False),
        "a message": (base.replace("predicting 0.5", "predicting one half", 1), False),
    }


def test_the_ast_rule_ignores_docstrings_comments_and_levels_value_only(base_model):
    ref = bs.model_rule_dump(base_model)
    for what, (edited, same) in _model_edits(base_model).items():
        assert edited != base_model, what
        assert (bs.model_rule_dump(edited) == ref) is same, what
    with pytest.raises(bs.BuildError):
        bs.model_rule_dump(base_model.replace('LEVEL = {"mu0"', 'LEVEL = dict(**{"mu0"', 1)
                           .replace('"attr_scale": 0.5}', '"attr_scale": 0.5})', 1))


@pytest.mark.parametrize("X", ["C1", "C1m", "C2"])
def test_the_patch_rewrites_only_the_docstring_the_comment_and_level(base_model, X, tmp_path):
    level = LEVELS[X]
    text = bs.pool_model_text(X, level, BENCH, base_model)
    patched = bs.patch_model(base_model, text["doc"], text["comment"], level)
    changed = bs.check_model(patched, base_model, level, text)
    spans = {i + 1 for i in bs._patch_spans(base_model)}
    assert set(changed) <= spans and changed
    path = tmp_path / "model.py"
    path.write_text(patched)
    assert bs.shipped_level(str(path)) == level
    compile(patched, "model.py", "exec")
    doc = " ".join(text["doc"].split())
    for fact in (f"candidate {X}", BENCH["revision"], f"{len(BENCH['L'])} public",
                 f"the {len(BENCH['U'])} of them", f"at most {P.K} pairs", "2,000 items",
                 *(repr(v) for v in level.values())):
        assert fact in doc, fact
    old = [p for p in base_model.split('"""')[1].split("\n\n")]
    new = text["doc"].split("\n\n")
    assert len(old) == len(new) and old[0] == new[0] and old[3:] == new[3:]
    assert "fitted on every\neligible public pair" not in patched
    lines = patched.splitlines()
    k = lines.index(next(x for x in lines if x.startswith("LEVEL = ")))
    assert all(x.startswith("#: ") for x in lines[k - len(text["comment"]):k])
    if X == "C1":
        assert lines[k] == 'LEVEL = {"mu0": -2.5, "sigma_mu": 2.5, "attr_scale": 0.5}'
        assert 68 not in changed


def test_check_model_refuses_anything_else(base_model):
    level = LEVELS["C2"]
    text = bs.pool_model_text("C2", level, BENCH, base_model)
    patched = bs.patch_model(base_model, text["doc"], text["comment"], level)
    cases = {
        "code": (patched.replace('os.environ.setdefault(_var, "1")',
                                 'os.environ.setdefault(_var, "2")', 1), level, text),
        "a comment elsewhere": (patched.replace("# noqa: E402", "# noqa: E402 (std)", 1), level,
                                text),
        "another level": (patched, {**level, "mu0": 0.161}, text),
        "another docstring": (patched.replace("candidate C2", "candidate C1", 1), level, text),
        "a fact not named": (patched, level, {**text, "facts": text["facts"] + ["candidate C9"]}),
    }
    for what, (src, lv, tx) in cases.items():
        with pytest.raises(bs.BuildError):
            bs.check_model(src, base_model, lv, tx)
            pytest.fail(what)


def test_patch_model_refuses_a_docstring_it_cannot_quote(base_model):
    for doc in ('a """ b\n', "a \\ b\n", "no newline"):
        with pytest.raises(bs.BuildError):
            bs.patch_model(base_model, doc, [], LEVELS["C1"])


def test_the_text_needs_archive_3s_docstring(base_model):
    with pytest.raises(bs.BuildError):
        bs.pool_model_text("C1", LEVELS["C1"], BENCH,
                           base_model.replace("bakes LEVEL below into them", "writes LEVEL", 1))


# --- the bundle and the archive against archive-3 ------------------------------------------

def test_compare_archives(tmp_path):
    def z(name, **members):
        files = {}
        for k, v in members.items():
            (tmp_path / f"{name}_{k}").write_text(v)
            files[k.replace("_", ".")] = str(tmp_path / f"{name}_{k}")
        bs.write_zip(str(tmp_path / f"{name}.zip"), files)
        return str(tmp_path / f"{name}.zip")
    old = z("old", model_py="m", prior_json="p", requirements_txt="r")
    same = z("same", model_py="m", prior_json="p", requirements_txt="r")
    new = z("new", model_py="m2", prior_json="p2", requirements_txt="r")
    assert bs.compare_archives(same, old) == []
    assert bs.compare_archives(new, old) == ["model.py", "prior.json"]
    for bad in (z("b1", model_py="m", prior_json="p", requirements_txt="r2"),
                z("b2", model_py="m", prior_json="p", requirements_txt="r", extra_py="x")):
        with pytest.raises(bs.BuildError):
            bs.compare_archives(bad, old)


def test_the_pool_bundle_checks(world):
    d = json.loads(world.bundles["C1"][0])
    a3 = json.loads(world.a3_prior)
    level = dict(P.LEVEL_A3)
    bs.check_pool_bundle(d, "C1", level, world.L, a3, None, P.ID_CAP, P.ID_CLIP)
    assert bs.bundle_keys(d) == bs.bundle_keys(a3)
    real = ROOT / "submission" / "prior.json"
    if real.is_file() and sha(real.read_bytes()) == A3_PRIOR_SHA256:   # archive-3's own keys
        assert bs.bundle_keys(d) == bs.bundle_keys(json.loads(real.read_text()))

    def changed(f):
        e = copy.deepcopy(d)
        f(e)
        return e
    cases = {
        "a benchmark excluded": (changed(lambda e: e["hyper"].update(
            excluded=["alpha"], included=[b for b in world.L if b != "alpha"])), level, world.L),
        "another L": (d, level, world.L[1:]),
        "another level": (d, {**level, "mu0": -2.4}, world.L),
        "identity unguarded": (changed(lambda e: e["hyper"].update(id_cap=1.0)), level, world.L),
        "sigma_theta": (changed(lambda e: e["hyper"].update(sigma_theta=1.0)), level, world.L),
        "a Student-t level": (changed(lambda e: e["hyper"].update(nu_mu=3.0)), level, world.L),
        "a per-benchmark level": (changed(lambda e: e["prior"]["meta"].update(
            levels={b: 0.0 for b in world.L})), level, world.L),
        "no subject prior": (changed(lambda e: e.update(prior=None)), level, world.L),
    }
    for what, (e, lv, L) in cases.items():
        with pytest.raises(bs.BuildError):
            bs.check_pool_bundle(e, what, lv, L, a3, None, P.ID_CAP, P.ID_CLIP)
            pytest.fail(what)
    with pytest.raises(bs.BuildError, match="the lock's"):
        bs.check_pool_bundle(d, "C1", level, world.L, a3, {"sigma_d": 0.5}, P.ID_CAP, P.ID_CLIP)


# --- the build, end to end ---------------------------------------------------------------------

def _check_built(world, X, got, out, paths, base_model):
    locked = world.bundles[X][0]
    assert got["candidate"] == X and got["bundle_sha256"] == world.bundles[X][1]
    assert Path(paths["prior_out"]).read_bytes() == locked
    assert got["differ"] == ["model.py", "prior.json"]
    assert bs.compare_archives(out, paths["archive3"]) == ["model.py", "prior.json"]
    with zipfile.ZipFile(out) as z:
        assert z.read("prior.json") == locked
        shipped = z.read("model.py").decode()
        a3_members = zipfile.ZipFile(paths["archive3"]).namelist()
        assert sorted(z.namelist()) == sorted(a3_members)
    level = world.bundles[X][2]
    text = bs.pool_model_text(X, level, world.bench, base_model)
    assert shipped == bs.patch_model(base_model, text["doc"], text["comment"], level)
    assert bs.model_rule_dump(shipped) == bs.model_rule_dump(base_model)
    assert got["level"] == level and got["sha256"] == sha(Path(out).read_bytes())


def test_pool_build_end_to_end_with_the_run_check_and_the_validator(ready, tmp_path, base_model):
    """C2, whose LEVEL literal changes: the bundle rebuilt byte for byte, model.py
    patched, the archive checked in fresh interpreters and by the organisers'
    validator, on a synthetic legacy-like session."""
    if not os.path.isfile(bs.VALIDATOR):
        pytest.skip("third_party/paiec_baseline not cloned")
    from tests.synth import make_pairs
    pairs = make_pairs(n_subjects=3, n_benchmarks=2, n_items=60, seed=5)
    got, out, paths = build_pool(ready, tmp_path, "C2", run_checks=True, candidate="C2",
                                 pairs=pairs)
    _check_built(ready, "C2", got, out, paths, base_model)
    assert got["pool_py"] == "locked"
    with zipfile.ZipFile(out) as z:
        assert '"mu0": %r' % ready.bundles["C2"][2]["mu0"] in z.read("model.py").decode()


@pytest.mark.parametrize("X", ["C1", "C1m"])
def test_pool_build_of_the_other_candidates(ready, tmp_path, base_model, X):
    got, out, paths = build_pool(ready, tmp_path, X)
    _check_built(ready, X, got, out, paths, base_model)
    if X == "C1m":
        assert got["level"] == {"mu0": MU0_M, "sigma_mu": 2.5, "attr_scale": 0.5}
    else:
        assert got["level"] == P.LEVEL_A3


def test_a_recorded_successor_pool_py_builds_the_same_archive(ready, tmp_path):
    first, out1, _ = build_pool(ready, tmp_path / "a", "C1m")
    lock = copy.deepcopy(ready.lock)
    now = lock["files"][bs.LOCK_POOL_PY]
    lock["files"][bs.LOCK_POOL_PY] = "f" * 64                 # the file as locked before a fix
    fixes = [{"cause": "toy", "files": {bs.LOCK_POOL_PY: {"old": "f" * 64, "new": now}}}]
    got, out2, _ = build_pool(ready, tmp_path / "b", "C1m", lock=lock, fixes=fixes)
    assert got["pool_py"] == "successor" and got["sha256"] == first["sha256"]


def test_the_repository_model_may_already_hold_archive_4s_text(ready, tmp_path):
    first, out, _ = build_pool(ready, tmp_path / "a", "C2")
    mine = tmp_path / "model_a4.py"
    mine.write_bytes(zipfile.ZipFile(out).read("model.py"))
    again, _, _ = build_pool(ready, tmp_path / "b", "C2", model=mine)
    assert again["sha256"] == first["sha256"]


def _refusals(w):
    """{case: (build_pool keywords, what the refusal says)}, all for C1m."""
    X, other = "C1m", lambda: copy.deepcopy(w.lock)

    def lock_with(f):
        lk = other()
        f(lk)
        return {"lock": lk}

    build = {"candidate": X, "bundle_sha256": w.bundles[X][1], "level": w.bundles[X][2]}
    return {
        "another candidate asked": ({"candidate": "C1"}, "--candidate"),
        "an unknown candidate": ({"build": {**build, "candidate": "C3"}}, "not one of"),
        "the record's sha256": ({"build": {**build, "bundle_sha256": "0" * 64}},
                                "not the locked one"),
        "the record's level": ({"build": {**build, "level": {**build["level"], "mu0": -2.105}}},
                               "level"),
        "the record lacks a key": ({"build": {"candidate": X, "level": build["level"]}}, "missing"),
        "the lock's bundle sha256": (lock_with(lambda lk: lk["bundles"][X].update(sha256="0" * 64)),
                                     "not the locked one"),
        "the lock's bundle file": (lock_with(lambda lk: lk["files"].update(
            {bs.LOCK_BUNDLE.format(X): "0" * 64})), "not the locked one"),
        "another mu0_m": (lock_with(lambda lk: lk.update(mu0_m=-2.2)), "not the locked one"),
        "the lock's level": (lock_with(lambda lk: lk["bundles"][X].update(
            level={**build["level"], "sigma_mu": 2.4})), "level"),
        "the lock's settings": (lock_with(lambda lk: lk["env"].update(PYTHONHASHSEED="1")),
                                "settings"),
        "a lock key missing": (lock_with(lambda lk: lk.pop("pool_frozen_ast")), "missing"),
        "archive-3's digest": (lock_with(lambda lk: lk["files"].update(
            {bs.LOCK_ARCHIVE_3: "0" * 64})), "sha256"),
        "pool.py not locked": (lock_with(lambda lk: lk["files"].update(
            {bs.LOCK_POOL_PY: "0" * 64})), "neither the locked file"),
        "the frozen AST": (lock_with(lambda lk: lk.update(pool_frozen_ast="0" * 64)),
                           "outside EVAL_LOADER"),
        "the EVAL_LOADER tuple": (lock_with(lambda lk: lk["eval_loader"].append("bundle")),
                                  "EVAL_LOADER"),
        "the fits index": (lock_with(lambda lk: lk["files"].update(
            {bs.LOCK_FITS_INDEX: "0" * 64})), "fits cache index"),
        "the lists": (lock_with(lambda lk: lk["files"].update({bs.LOCK_LISTS: "0" * 64})),
                      "not the locked file"),
    }


REFUSALS = ("another candidate asked", "an unknown candidate", "the record's sha256",
            "the record's level", "the record lacks a key", "the lock's bundle sha256",
            "the lock's bundle file", "another mu0_m", "the lock's level", "the lock's settings",
            "a lock key missing", "archive-3's digest", "pool.py not locked", "the frozen AST",
            "the EVAL_LOADER tuple", "the fits index", "the lists")


def test_every_refusal_case_is_run(world):
    assert sorted(_refusals(world)) == sorted(REFUSALS)


@pytest.mark.parametrize("case", REFUSALS)
def test_pool_build_refuses_and_touches_nothing(ready, tmp_path, case):
    kw, match = _refusals(ready)[case]
    out = tmp_path / "dist" / "paiec.zip"
    out.parent.mkdir()
    out.write_bytes(b"archive-3 still here")
    with pytest.raises(bs.BuildError, match=match):
        build_pool(ready, tmp_path, "C1m", out=str(out), **kw)
    assert out.read_bytes() == b"archive-3 still here"
    assert not (tmp_path / "prior.json").exists()


def test_pool_build_refuses_without_the_gates_record_or_archive_3(ready, tmp_path):
    paths = setup(ready, tmp_path, "C1")
    os.remove(paths["build_path"])
    model = str(ROOT / "submission" / "model.py")
    with pytest.raises(bs.BuildError, match="gate has not passed"):
        bs.pool_build(str(tmp_path / "out.zip"), model=model, run_checks=False, **paths)
    paths = setup(ready, tmp_path, "C1")
    with pytest.raises(bs.BuildError, match="keep archive-3 there"):
        bs.pool_build(str(tmp_path / "out.zip"), model=model, run_checks=False,
                      **{**paths, "archive3": str(tmp_path / "none.zip")})
    with pytest.raises(bs.BuildError, match="archive-3's copy"):
        bs.pool_build(paths["archive3"], model=model, run_checks=False, **paths)
    assert sha(Path(paths["archive3"]).read_bytes()) == ready.a3_sha


def test_pool_build_refuses_a_changed_fits_entry(ready, tmp_path, monkeypatch):
    fits = tmp_path / "fits"
    shutil.copytree(P.FITS_DIR, fits)
    monkeypatch.setattr(P, "FITS_DIR", str(fits))
    entry = json.loads((fits / "index.json").read_text())["entries"]["alpha/free"]["file"]
    with open(fits / entry, "a") as f:
        f.write(" ")
    with pytest.raises(bs.BuildError, match="pool check failed.*changed"):
        build_pool(ready, tmp_path, "C1")


def test_pool_build_refuses_a_repository_model_with_another_change(ready, tmp_path, base_model):
    mine = tmp_path / "model_changed.py"
    mine.write_text(base_model.replace('os.environ.setdefault(_var, "1")',
                                       'os.environ.setdefault(_var, "2")', 1))
    with pytest.raises(bs.BuildError, match="neither archive-3's model.py"):
        build_pool(ready, tmp_path, "C1", model=mine)


def test_a_failed_run_check_removes_the_archive_and_writes_no_prior(ready, tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise bs.BuildError("run check failed (toy)")
    monkeypatch.setattr(bs, "check_run", boom)
    out = tmp_path / "dist" / "paiec.zip"
    out.parent.mkdir()
    out.write_bytes(b"old")
    with pytest.raises(bs.BuildError, match="toy"):
        build_pool(ready, tmp_path, "C1", run_checks=True, pairs=[], out=str(out))
    assert not out.exists() and not (tmp_path / "prior.json").exists()


# --- the command line and the default path -------------------------------------------------------

def test_the_command_line(monkeypatch):
    with pytest.raises(SystemExit):
        bs.main(["--candidate", "C1"])                  # --candidate needs --pool
    with pytest.raises(bs.BuildError, match="ships hier"):
        bs.main(["--pool", "--legacy"])
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(bs.BuildError, match="lock's settings"):
        bs.main(["--pool", "--candidate", "C1"])


def test_a_pool_prior_json_in_submission_passes_the_pool_checks():
    """impl_plan step 11: once --pool has written submission/prior.json, it is a
    pool bundle (test_submission.py's fixtures then fall back to their toy, as
    for any prior.json that is not archive-3's kind) and must pass the pool
    checks with its own level, against L and archive-3's keys."""
    built = ROOT / "submission" / "prior.json"
    d = json.loads(built.read_text()) if built.is_file() else None
    included = (d.get("hyper") or {}).get("included", []) if isinstance(d, dict) else []
    if sorted(included) != sorted(BENCH["L"]):
        pytest.skip("submission/prior.json is not a pool bundle")
    a3, copy3 = None, ROOT / "data" / "pool_eval" / "a3_prior.json"
    if copy3.is_file() and sha(copy3.read_bytes()) == A3_PRIOR_SHA256:
        a3 = json.loads(copy3.read_text())
    if a3 is None and os.path.isfile(bs.ARCHIVE_3):
        a3 = json.loads(zipfile.ZipFile(bs.ARCHIVE_3).read("prior.json"))
    if a3 is None:
        pytest.skip("no copy of archive-3's prior.json")
    level = {k: d["hyper"][k] for k in bs.LEVEL_KEYS}
    bs.check_pool_bundle(d, str(built), level, BENCH["L"], a3, None, P.ID_CAP, P.ID_CLIP)


def test_the_default_path_still_zips_archive_3(tmp_path):
    """members() and write_zip() of the default path, from the current sources and
    archive-3's prior.json, give archive-3's bytes."""
    prior, model = ROOT / "submission" / "prior.json", ROOT / "submission" / "model.py"
    if not prior.is_file() or sha(prior.read_bytes()) != A3_PRIOR_SHA256 or \
            sha(model.read_bytes()) != A3_MODEL_SHA256:
        pytest.skip("submission/ does not hold archive-3's prior.json and model.py")
    out = tmp_path / "default.zip"
    bs.write_zip(str(out), bs.members())
    assert sha(out.read_bytes()) == bs.ARCHIVE_3_SHA256
