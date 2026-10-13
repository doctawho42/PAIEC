"""Fit the prior, bake it into submission/, zip the archive, check it.

    python tools/build_submission.py               # needs the data, see paiec.data
    python tools/build_submission.py --labeling    # also ship submission/labeling.py
    python tools/build_submission.py --legacy      # the Predictor of commit b68492c instead
    PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        VECLIB_MAXIMUM_THREADS=1 python tools/build_submission.py --pool [--candidate X]
                                                   # archive-4, after amendment 4's gate passed

The shipped model is paiec.hier's HierPredictor with the level prior moved down
for the hidden test (docs/findings.md, "Verdict: ship hier with the level moved
down"). prior.json is its bundle, paiec.prior.to_json(prior, hyper): the subject
prior and empirical-Bayes hyperparameters of paiec.prior.build on every
eligible public pair, with LEVEL from submission/model.py written over mu0,
sigma_mu and attr_scale. model.py reads LEVEL from its own source here, so the
build and the model's fallback cannot disagree. --legacy writes the
Predictor's attribute prior ({"coef", "spec"}, fitted on every public pair as
before), which model.py loads as the shipped Predictor; that is the rollback.

The archive holds model.py, prior.json and requirements.txt at its root and the
run-time modules of the research package under paiec_rt/, a name of their own
so that no paiec the platform has imported can stand in for them. prior.py
ships whole; its main() reads the public data and is offline only. labeling.py
stays out by default: no acquisition policy measured better than the platform's
random one.

Anything that would only surface on the platform fails the build instead, and
dist/paiec.zip exists only after every check passed:
  * prior.json unreadable, not finite, the wrong shape, not the kind asked for,
    or (hier) a level prior other than LEVEL, a fit on other than all five
    public benchmarks, no attribute prior, or a prior and hyperparameters
    HierPredictor refuses to pair;
  * a shipped module importing a local module the archive does not carry (the
    research package's own absolute imports included, since it ships renamed),
    or importing anything but the standard library and numpy at module level;
  * a fresh interpreter that loads model.py the way the validator does and
    predicts real items in the official format at every budget, a fresh
    predictor per budget as the platform's recreated workers have it and one
    shared labeled list per budget, and sees a fallback, the wrong model, a
    module from outside the archive, scipy, scikit-learn or pandas loaded (hier
    needs none of them on the public subjects' ISO release dates; another
    date form sends paiec.subjects to pandas, if installed), or anything but a
    native float in [0, 1];
  * those predictions differing in any bit from the research package's own
    predictor built from the same prior.json, in a fresh interpreter with the
    same thread limits;
  * the organisers' validator, full checks, not printing OK.

--pool builds archive-4 (docs/plans/p2_pool_eval.md section 10), and only after
amendment 4's gate passed: tools/pool_build.json, which the gate writes on a pass
only, names the candidate X* it carried. It refuses to run without the lock's
settings, PYTHONHASHSEED=0 and one BLAS thread, which a multithreaded BLAS would
otherwise break in the last bits. It rebuilds X*'s bundle with nothing excluded
from paiec/pool.py and the fits cache, both checked against the lock in
results/pool_eval.json (pool.py may be a successor that its "fixes" record
names, whose AST outside paiec.pool.EVAL_LOADER is the locked file's), and
refuses unless the bytes are the locked bundle's. prior.json is that bundle. model.py is archive-3's
(dist/archive-3.zip, sha256 checked in full) with its module docstring, the
comment above LEVEL and LEVEL's value rewritten for X*, a patch applied to the
archive's copy and checked by an AST comparison that ignores docstrings, comments
and LEVEL's value; submission/model.py itself must be archive-3's or that text.
On top of the checks above (the bundle, the imports, the run check on the legacy
session, the validator) it requires the bundle fitted on L of
tools/pool_benchmarks.json with nothing excluded, archive-3's identity guards
(id_cap 0.1, sigma_theta from the clipped tau2) and archive-3's keys with no
per-benchmark level, the level the lock holds for X*, and an archive that differs
from archive-3 in prior.json and model.py only. submission/prior.json is written
once every check has passed.
"""
import argparse
import ast
import difflib
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUB = os.path.join(ROOT, "submission")
OUT = os.path.join(ROOT, "dist", "paiec.zip")
PACKAGE = ("__init__.py", "hier.py", "prior.py", "predict.py", "fitting.py", "irt.py",
           "subjects.py", "mcq.py")
#: the name paiec's run-time modules ship under
RUNTIME = "paiec_rt"
VALIDATOR = os.path.join(ROOT, "third_party", "paiec_baseline", "check_submission_zip.py")
#: functions only experiments call, allowed to import modules the archive leaves out
OFFLINE_ONLY = {(f"{RUNTIME}/irt.py", "evaluate"), (f"{RUNTIME}/prior.py", "main")}
#: what shipped code may import at module level besides the standard library and itself
MODULE_LEVEL_OK = {"numpy"}
#: optional packages hier does not load, at import or at any call on public inputs
HEAVY = ("scipy", "sklearn", "pandas")
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra")
BUDGETS = (0, 1, 3, 7, 15, 31)
#: model.MODEL for each kind of prior.json
MODELS = {"hier": "hier", "legacy": "predictor"}


class BuildError(RuntimeError):
    pass


def members(prior=os.path.join(SUB, "prior.json"), labeling=False):
    """Archive name -> source file, refusing to go on when one is missing."""
    m = {"model.py": os.path.join(SUB, "model.py"), "prior.json": prior,
         "requirements.txt": os.path.join(ROOT, "requirements.txt")}
    if labeling:
        m["labeling.py"] = os.path.join(SUB, "labeling.py")
    m.update({f"{RUNTIME}/{n}": os.path.join(ROOT, "paiec", n) for n in PACKAGE})
    missing = sorted(a for a, src in m.items() if not os.path.isfile(src))
    if missing:
        raise BuildError(f"missing from the archive: {', '.join(missing)}")
    return m


# --- prior.json ------------------------------------------------------------------------

def shipped_level(model=os.path.join(SUB, "model.py")):
    """LEVEL as model.py defines it: a dict literal of Hyper fields."""
    from paiec.hier import Hyper
    try:
        with open(model) as f:
            tree = ast.parse(f.read(), model)
        (node,) = [n for n in tree.body if isinstance(n, ast.Assign)
                   and [getattr(t, "id", None) for t in n.targets] == ["LEVEL"]]
        level = {str(k): float(v) for k, v in ast.literal_eval(node.value).items()}
        replace(Hyper(), **level)
    except Exception as exc:
        raise BuildError(f"{model} defines no usable LEVEL: {exc!r}") from None
    return level


def write_bundle(pairs, path, level):
    """hier's bundle on every eligible pair, LEVEL written over the fit."""
    from paiec import prior as PR
    from paiec.official import eligible
    pairs = eligible(pairs)
    prior, hyper = PR.build(pairs, ())
    fitted = {k: round(getattr(hyper, k), 3) for k in level}
    hyper = replace(hyper, **level)
    PR.save(path, prior, hyper)
    if PR.to_json(*PR.load(path)) != PR.to_json(prior, hyper):
        raise BuildError(f"{path} does not read back as the fit it was written from")
    print(f"hier bundle fitted on {len(pairs)} pairs of {', '.join(hyper.included)}; "
          f"level {level} over the fitted {fitted} -> {path}")
    return prior, hyper


def write_prior(pairs, path):
    """The Predictor's attribute prior, as the legacy build wrote it."""
    from paiec.predict import fit_prior
    coef, spec = fit_prior(pairs)
    with open(path, "w") as f:
        json.dump({"coef": coef.tolist(), "spec": spec.to_dict()}, f)
    print(f"prior fitted on {len(pairs)} pairs, {len(coef)} coefficients -> {path}")


def _check_legacy(d, path):
    """A NaN median would silently switch the prior off for every subject that
    lacks that attribute, so every number must be finite."""
    import math
    from paiec.subjects import Spec, attrs, design_row
    try:
        spec = Spec.from_dict(d["spec"])
        coef = [float(c) for c in d["coef"]]
    except Exception as exc:
        raise BuildError(f"{path} is unusable: {exc!r}") from None
    if not all(map(math.isfinite, coef + [spec.med_days, spec.med_log_size])):
        raise BuildError(f"{path} holds a non-finite number")
    if len(coef) != len(design_row(attrs({}), spec)):
        raise BuildError(f"{path}: {len(coef)} coefficients for a spec of "
                         f"{len(design_row(attrs({}), spec))} columns")


def _check_bundle(d, path, level):
    """paiec.prior.from_json refuses a malformed subject prior and a
    non-finite or negative hyperparameter; on top of that the bundle must be
    the shipped configuration, fitted on all five public benchmarks."""
    from paiec import prior as PR
    from paiec.hier import PUBLIC, HierPredictor
    try:
        prior, hyper = PR.from_json(d)
        HierPredictor(prior, hyper)
    except Exception as exc:
        raise BuildError(f"{path} is unusable: {exc!r}") from None
    problems = []
    if prior is None or not prior.has_attributes:
        problems.append("no attribute prior")
    if level is not None:
        got = {k: getattr(hyper, k) for k in level}
        if got != level:
            problems.append(f"level prior {got}, model.py's LEVEL is {level}")
    if hyper.nu_mu != 0:
        problems.append(f"a Student-t level (nu_mu {hyper.nu_mu})")
    if hyper.excluded or sorted(hyper.included) != sorted(PUBLIC):
        problems.append(f"fitted on {list(hyper.included)} without {list(hyper.excluded)}")
    if problems:
        raise BuildError(f"{path}: " + "; ".join(problems))
    return prior, hyper


def check_prior(path, kind=None, level=None):
    """The kind of prior.json ('hier' or 'legacy') after checking it as that
    kind; `kind` is the one the build asked for, `level` hier's LEVEL."""
    from paiec.prior import KIND
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as exc:
        raise BuildError(f"{path} is unusable: {exc!r}") from None
    got = "hier" if isinstance(d, dict) and "kind" in d else "legacy"
    if kind is not None and got != kind:
        raise BuildError(f"{path} is a {got} prior, the build asked for {kind}")
    if got == "hier":
        if d.get("kind") != KIND:
            raise BuildError(f"{path} is of kind {d.get('kind')!r}, not {KIND}")
        _check_bundle(d, path, level)
    else:
        _check_legacy(d, path)
    return got


# --- imports ---------------------------------------------------------------------------

def _imports(tree):
    """(import node, outermost enclosing function or None) for every import."""
    stack = [(tree, None)]
    while stack:
        node, func = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child, func
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.append((child, func or getattr(child, "name", "<lambda>")))
            else:
                stack.append((child, func))


def _is_module(name, root):
    """Whether `name` is a module, paiec_rt read as the paiec it is copied from."""
    parts = name.split(".")
    if parts[0] == RUNTIME:
        parts[0] = "paiec"
    path = os.path.join(root, *parts)
    return os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py"))


def _targets(node, arc, root):
    """Modules an import statement loads, relative imports resolved."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    base = node.module or ""
    if node.level:
        pkg = arc.split("/")[:-1]
        pkg = pkg[:len(pkg) - node.level + 1]
        base = ".".join(pkg + ([base] if base else []))
    return [f"{base}.{a.name}" if _is_module(f"{base}.{a.name}", root) else base
            for a in node.names]


def check_imports(files, root=ROOT):
    """Walk every import of every shipped Python file, wherever it sits.

    A local import must resolve to a shipped module unless it is inside an
    OFFLINE_ONLY function; lazy imports count, since a predict() path could
    reach them. At module level, which every worker runs on import, only the
    standard library and numpy may be imported: scipy, scikit-learn and pandas
    are optional and slow, and belong inside the functions that use them. The
    package ships as paiec_rt, so an absolute `paiec.` import in it names a
    module the archive does not have; its imports of each other are relative.
    """
    local = {a.split("/")[0].removesuffix(".py") for a in files} | {"paiec", RUNTIME}
    shipped = {a.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
               for a in files if a.endswith(".py")}
    problems = []
    for arc, src in sorted(files.items()):
        if not arc.endswith(".py"):
            continue
        with open(src) as f:
            tree = ast.parse(f.read(), arc)
        for node, func in _imports(tree):
            for mod in _targets(node, arc, root):
                top = mod.split(".")[0]
                if top in local:
                    if mod not in shipped and (arc, func) not in OFFLINE_ONLY:
                        problems.append(f"{arc}:{node.lineno} imports {mod}, "
                                        "which the archive does not ship")
                elif func is None and top not in sys.stdlib_module_names | MODULE_LEVEL_OK:
                    problems.append(f"{arc}:{node.lineno} imports {mod} at module level")
    if problems:
        raise BuildError("import check failed:\n  " + "\n  ".join(problems))


def write_zip(out, files):
    """Sorted entries with a fixed timestamp, so the same sources give the same bytes."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc in sorted(files):
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with open(files[arc], "rb") as f:
                z.writestr(info, f.read())


# --- the run check -----------------------------------------------------------------------

def official(pair, item):
    """A replica pair's subject and item as the platform sends them."""
    import hashlib
    subject = {k: pair.subject.get(k, "") for k in SUBJECT_FIELDS}
    anon = "b-" + hashlib.sha256(pair.benchmark_id.encode()).hexdigest()[:12]
    return [subject, {"item_content": item.get("item_content", ""),
                      "item_features": item.get("item_features", ""),
                      "interactors": item.get("interactors", ""), "benchmark_id": anon}]


def session(pairs, per_benchmark=3, targets_per_pair=25, seed=0):
    """A formative-sized run in the official format: a few pairs per benchmark,
    items split 50/50, the first B acquisition labels of every pair shared by
    every target at budget B."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by_b = {}
    for p in pairs:
        by_b.setdefault(p.benchmark_id, []).append(p)
    labels = {B: [] for B in BUDGETS}
    targets = []
    for bid in sorted(by_b):
        for i in rng.permutation(len(by_b[bid]))[:per_benchmark]:
            p = by_b[bid][i]
            items = {}
            for r in p.responses:
                items.setdefault(r.item_key, []).append(r)
            keys = sorted(items)
            keys = [keys[j] for j in rng.permutation(len(keys))]
            acq, ev = keys[:len(keys) // 2], keys[len(keys) // 2:]
            for B in BUDGETS:
                labels[B] += [[official(p, items[k][0].item), items[k][0].label] for k in acq[:B]]
            for k in ev[:targets_per_pair]:
                targets.append([official(p, items[k][0].item), [r.label for r in items[k]]])
    return {"labeled": {str(B): v for B, v in labels.items()}, "targets": targets}


#: argv: archive root, session.json, 'archive' or the repository root, HEAVY
RUN = r"""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import importlib.util, json, math, sys, time
root, data, source = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
if source == "archive":
    before = list(sys.path)
    sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("submission_model", os.path.join(root, "model.py"))
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    sys.path[:] = before
    make, what = model.make, model.MODEL
else:
    sys.path.insert(0, source)
    from paiec import prior as PR
    from paiec.hier import HierPredictor
    from paiec.predict import Predictor
    from paiec.subjects import Spec
    d = json.load(open(os.path.join(root, "prior.json")))
    if "kind" in d:
        prior, hyper = PR.from_json(d)
        make, what = (lambda: HierPredictor(prior, hyper)), "hier"
    else:
        import numpy as np
        coef, spec = np.asarray(d["coef"], float), Spec.from_dict(d["spec"])
        make, what = (lambda: Predictor(coef, spec)), "predictor"
out = {}
for B, labeled in data["labeled"].items():
    P = make()
    t0, se, slowest, preds = time.perf_counter(), 0.0, 0.0, []
    for inp, ys in data["targets"]:
        t = time.perf_counter()
        p = P.predict(inp, labeled)
        slowest = max(slowest, time.perf_counter() - t)
        assert type(p) is float and math.isfinite(p) and 0 <= p <= 1, repr(p)
        se += sum((p - y) ** 2 for y in ys) / len(ys)
        preds.append(p.hex())
    assert P.failures == 0, f"budget {B}: {P.failures} fallbacks"
    out[B] = dict(brier=se / len(data["targets"]), seconds=time.perf_counter() - t0,
                  slowest=slowest, preds=preds, labels=len(labeled),
                  unconverged=getattr(P, "unconverged", 0))
where = os.path.realpath(root if source == "archive" else source)
stray = sorted(n for n, m in sys.modules.items() if n.split(".")[0] in ("paiec", "paiec_rt")
               and not os.path.realpath(getattr(m, "__file__", "") or "").startswith(where))
assert not stray, f"modules from outside {where}: {stray}"
heavy = sorted({n.split(".")[0] for n in sys.modules} & set(sys.argv[4].split(",")))
print(json.dumps(dict(model=what, heavy=heavy, budgets=out)))
"""


def _run(root, data, source):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", RUN, root, data, source, ",".join(HEAVY)],
                       cwd=os.path.dirname(data), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(f"run check ({source}) failed:\n{r.stdout}{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def check_run(zip_path, pairs, kind="hier"):
    """The archive in a fresh interpreter, then the research package on the
    same prior.json and session; returns the archive's per-budget results."""
    with tempfile.TemporaryDirectory(prefix="paiec-run-") as tmp:
        root = os.path.join(tmp, "submission")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        data = os.path.join(tmp, "session.json")
        run = session(pairs)
        n = len(run["targets"])
        with open(data, "w") as f:
            json.dump(run, f)
        got = _run(os.path.realpath(root), data, "archive")
        ref = _run(os.path.realpath(root), data, ROOT)
    if got["model"] != MODELS[kind]:
        raise BuildError(f"the archive loaded {got['model']!r}, not {MODELS[kind]!r}")
    if kind == "hier" and got["heavy"]:
        raise BuildError(f"hier loaded {', '.join(got['heavy'])}, which it never needs")
    for B, v in got["budgets"].items():
        same = v["preds"] == ref["budgets"][B]["preds"]
        print(f"  budget {B:>2}: {v['labels']:>3} labels, {n} targets, Brier {v['brier']:.4f}, "
              f"{1e3 * v['seconds'] / n:.2f} ms a call, slowest {v['slowest']:.3f} s"
              + (f", {v['unconverged']} unconverged fit" if v["unconverged"] else "")
              + ("" if same else "  DIFFERS FROM THE RESEARCH PACKAGE"))
        if not same:
            raise BuildError(f"budget {B}: the archive's predictions differ from "
                             "the research package's on the same prior.json")
    print(f"  identical to paiec's own predictor on the same prior.json at every budget; "
          f"optional packages loaded: {', '.join(got['heavy']) or 'none'}")
    return got["budgets"]


def run_validator(zip_path):
    if not os.path.isfile(VALIDATOR):
        raise BuildError(f"organisers' validator not found at {VALIDATOR}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, VALIDATOR, os.path.abspath(zip_path)],
                           cwd=tmp, env=env, capture_output=True, text=True)
    print("  " + (r.stdout + r.stderr).strip().replace("\n", "\n  "))
    if r.returncode != 0 or not r.stdout.startswith("OK"):
        raise BuildError("the organisers' validator did not print OK")


# --- the pool build: archive-4 (--pool; docs/plans/p2_pool_eval.md "PE" section 10) ------
#
# What --pool reads, and from whom. experiments/pool_eval.py writes results/pool_eval.json,
# whose "lock" holds (PE 9 item 10):
#   files            {path relative to the repository: sha256}, among them the LOCK_* keys
#                    below (the fits cache index, X*'s archive bundle, the lists, pool.py,
#                    dist/archive-3.zip)
#   pool_frozen_ast  sha256 of paiec.pool.frozen_ast(pool.py's source), UTF-8
#   eval_loader      the tuple paiec.pool.EVAL_LOADER, as a list
#   mu0_m            C1m's mu0, rounded; c2_literal  C2's level as model.py would hold it
#   bundles          {X: {"sha256", "level"[, "hyper"]}}, as results/pool_fits.json has them
#   env              the settings it ran under (PYTHONHASHSEED and the BLAS variables)
# and whose "fixes" is a list of deviation records, each with "files": {path: {"old",
# "new"}} (PE 9 item 11). The gate writes tools/pool_build.json on a pass only:
# {"candidate": X*, "bundle_sha256": its locked bundle's sha256, "level": its level}.

POOL_EVAL = os.path.join(ROOT, "results", "pool_eval.json")
POOL_BUILD = os.path.join(ROOT, "tools", "pool_build.json")
POOL_PY = os.path.join(ROOT, "paiec", "pool.py")
#: archive-3 as impl_plan step 1 keeps it: what archive-4 may differ from only in POOL_CHANGED
ARCHIVE_3 = os.path.join(ROOT, "dist", "archive-3.zip")
ARCHIVE_3_SHA256 = "4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090"
LOCK_POOL_PY = "paiec/pool.py"
LOCK_LISTS = "tools/pool_benchmarks.json"
LOCK_FITS_INDEX = "data/pool_fits/index.json"
LOCK_ARCHIVE_3 = "dist/archive-3.zip"
LOCK_BUNDLE = "data/pool_eval/bundles/archive_{}.json"
LOCK_KEYS = ("files", "pool_frozen_ast", "eval_loader", "mu0_m", "c2_literal", "bundles", "env")
BUILD_KEYS = ("candidate", "bundle_sha256", "level")
#: the archive members archive-4 may change (PE 2)
POOL_CHANGED = ("model.py", "prior.json")
LEVEL_KEYS = ("mu0", "sigma_mu", "attr_scale")
#: the lock's settings (PE 9 item 10), checked as paiec.pool's fits stage checks them
POOL_HASHSEED = "0"
POOL_BLAS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
#: archive-3's model.py text the pool build rewrites (PE 2): the paragraph naming the level
#: and the fit, and the sentence on the build baking LEVEL in (docstring lines 3-8, 13-14)
MODEL_LEVEL_PARAGRAPH = "The model is paiec.hier's hierarchical predictor"
MODEL_BUILD_SENTENCE = ("tools/build_submission.py fits both, bakes LEVEL below into them, "
                        "and checks the archive before it zips it.")
POOL_BUILD_SENTENCE = ("tools/build_submission.py --pool rebuilds the bundle from the locked "
                       "pool fits, refuses unless it is the locked bundle byte for byte, writes "
                       "its level below as LEVEL, and checks the archive before it zips it.")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path) -> str:
    with open(path, "rb") as f:
        return _sha256(f.read())


def _read_json(path, what):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise BuildError(f"{what} ({path}) is unusable: {exc!r}") from None


def _hash_randomization() -> int:
    return sys.flags.hash_randomization


def pool_env_problems(environ=None) -> list:
    """What keeps this process from the lock's settings: PYTHONHASHSEED=0 (hash
    randomisation off) and every BLAS variable at 1."""
    env = os.environ if environ is None else environ
    out = []
    if env.get("PYTHONHASHSEED") != POOL_HASHSEED or _hash_randomization():
        out.append(f"PYTHONHASHSEED={env.get('PYTHONHASHSEED')!r} (hash randomisation "
                   f"{_hash_randomization()}), not {POOL_HASHSEED!r}")
    out += [f"{v}={env.get(v)!r}, not '1'" for v in POOL_BLAS if env.get(v) != "1"]
    return out


def _level_of(d, what) -> dict:
    """A level dict of exactly LEVEL_KEYS, as floats."""
    try:
        if sorted(d) != sorted(LEVEL_KEYS):
            raise ValueError(f"keys {sorted(d)}")
        return {k: float(d[k]) for k in LEVEL_KEYS}
    except Exception as exc:
        raise BuildError(f"{what}: not a level of {', '.join(LEVEL_KEYS)} ({exc!r})") from None


def pool_state(eval_path=POOL_EVAL, build_path=POOL_BUILD, candidates=("C1", "C1m", "C2")):
    """(build, lock, fixes): the gate's pass record and the lock and fixes of
    results/pool_eval.json, every key --pool reads present."""
    if not os.path.isfile(build_path):
        raise BuildError(f"no {build_path}: the gate has not passed, so archive-3 stays and "
                         "nothing is built")
    build = _read_json(build_path, "the gate's pass record")
    rec = _read_json(eval_path, "the pool evaluation's record")
    lock = rec.get("lock") if isinstance(rec, dict) else None
    if not isinstance(build, dict) or not isinstance(lock, dict):
        raise BuildError(f"{build_path} or the lock of {eval_path} is not a record")
    missing = [f"{build_path}: {k}" for k in BUILD_KEYS if k not in build] + \
        [f"lock: {k}" for k in LOCK_KEYS if k not in lock]
    if missing:
        raise BuildError("missing: " + ", ".join(missing))
    X = build["candidate"]
    if X not in candidates:
        raise BuildError(f"{build_path} names {X!r}, not one of {', '.join(candidates)}")
    build = {**build, "level": _level_of(build["level"], build_path)}
    entry = lock["bundles"].get(X) if isinstance(lock["bundles"], dict) else None
    if not isinstance(entry, dict) or "sha256" not in entry or "level" not in entry:
        raise BuildError(f"the lock records no bundle sha256 and level for {X}")
    fixes = rec.get("fixes") or []
    if not isinstance(fixes, list):
        raise BuildError(f"the fixes of {eval_path} are not a list")
    return build, lock, fixes


def pool_frozen_ast(source, names) -> str:
    """paiec.pool.frozen_ast with the removed names given (the locked EVAL_LOADER),
    so that the check does not rest on the file it checks: ast.dump of the
    module without the top-level definitions and assignments named."""
    tree = ast.parse(source)

    def named(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None
    tree.body = [n for n in tree.body if named(n) not in names]
    return ast.dump(tree, include_attributes=False)


def pool_successors(fixes, path=LOCK_POOL_PY) -> list:
    """The successor digests the fixes record for `path`."""
    out = []
    for e in fixes:
        f = (e.get("files") or {}).get(path) if isinstance(e, dict) else None
        if isinstance(f, dict) and f.get("new"):
            out.append(f["new"])
    return out


def check_pool_py(source: bytes, lock, fixes, eval_loader) -> str:
    """'locked' or 'successor' (PE 9 item 11): paiec/pool.py must be the locked
    file or a successor the fixes record, its EVAL_LOADER the locked tuple, and
    its AST without the definitions that tuple names the locked one."""
    names = tuple(lock["eval_loader"])
    if tuple(eval_loader) != names:
        raise BuildError(f"paiec.pool.EVAL_LOADER is {tuple(eval_loader)}, the lock's {names}")
    sha = _sha256(source)
    if sha == lock["files"].get(LOCK_POOL_PY):
        how = "locked"
    elif sha in pool_successors(fixes):
        how = "successor"
    else:
        raise BuildError(f"paiec/pool.py (sha256 {sha[:16]}) is neither the locked file nor a "
                         "successor recorded in the fixes")
    frozen = _sha256(pool_frozen_ast(source.decode("utf-8"), names).encode("utf-8"))
    if frozen != lock["pool_frozen_ast"]:
        raise BuildError(f"paiec/pool.py ({how}) differs from the locked file outside "
                         f"EVAL_LOADER (frozen AST {frozen[:16]}, locked "
                         f"{str(lock['pool_frozen_ast'])[:16]})")
    return how


# --- model.py: the patch of archive-3's text and the AST rule (PE 2, R6) ----------------

def _docstring_node(tree):
    n = tree.body[0] if tree.body else None
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and \
            isinstance(n.value.value, str):
        return n
    return None


def _level_assign(tree):
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and [getattr(t, "id", None) for t in n.targets] == ["LEVEL"]]
    if len(nodes) != 1:
        raise BuildError(f"model.py assigns LEVEL {len(nodes)} times at module level, not once")
    return nodes[0]


def model_rule_dump(source) -> str:
    """What the AST rule compares (PE 2, 10; the critic's R6): ast.dump of model.py
    with every docstring dropped and LEVEL's values blanked. Comments are not in
    an AST; LEVEL's keys stay, so the level's fields cannot change."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise BuildError(f"model.py does not parse: {exc}") from None
    for node in list(ast.walk(tree)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    lv = _level_assign(tree)
    if not isinstance(lv.value, ast.Dict):
        raise BuildError("model.py's LEVEL is not a dict literal")
    lv.value = ast.Dict(keys=lv.value.keys, values=[ast.Constant(None) for _ in lv.value.values])
    return ast.dump(tree, include_attributes=False)


def _wrap(text, width=79) -> str:
    return textwrap.fill(" ".join(text.split()), width, break_long_words=False,
                         break_on_hyphens=False)


def pool_model_text(X, level, bench, base) -> dict:
    """The text model.py ships for candidate X (PE 2, 10): archive-3's module
    docstring (`base` is archive-3's model.py) with its level paragraph and the
    sentence on the build baking LEVEL in rewritten, and the comment above LEVEL,
    as {"doc", "comment" (lines without '#: '), "facts" (what the docstring must
    name)}. `bench` is tools/pool_benchmarks.json. The author confirms the
    wording when the gate has passed."""
    from paiec import pool as P
    v = {k: repr(float(level[k])) for k in LEVEL_KEYS}
    lv = f"mu0 {v['mu0']}, sigma_mu {v['sigma_mu']} and attr_scale {v['attr_scale']}"
    lead = {
        "C1": (f"The model is paiec.hier's hierarchical predictor with archive-3's level prior, "
               f"{lv} (docs/findings.md, \"Calibrating for the hidden test\" and \"What "
               f"actually shipped, after the audit\"), candidate C1 of "
               f"docs/plans/p2_pool_eval.md."),
        "C1m": (f"The model is paiec.hier's hierarchical predictor with the level prior {lv}, "
                f"candidate C1m of docs/plans/p2_pool_eval.md: archive-3's sigma_mu and "
                f"attr_scale, and the mu0, rounded to 3 decimals, at which its B0 centre (that "
                f"plan's section 4.2: the mean prediction without labels over the pool's "
                f"evaluation pairs, each benchmark once, without floor, group or text term) "
                f"equals archive-3's with every dated subject's release and access dates moved "
                f"{P.DATE_SHIFT} years later."),
        "C2": (f"The model is paiec.hier's hierarchical predictor with the pool's "
               f"empirical-Bayes level prior, {lv}, mu0 and sigma_mu rounded to 3 decimals, "
               f"candidate C2 of docs/plans/p2_pool_eval.md."),
    }[X]
    tail = (f"That level prior sits on top of the empirical-Bayes hyperparameters and the "
            f"subject prior of the pool build (docs/plans/p2_pool_eval.md, section 4.1), "
            f"fitted with nothing excluded on the expanded measurement-db pool at revision "
            f"{bench['revision']}: the item models of its {len(bench['L'])} public item-level "
            f"benchmarks whose responses are binary, or mixed with a 0/1 share of at least "
            f"{P.SHARE01_MIN}, and that have a pair with {P.MIN_ITEMS} or more items, at most "
            f"{P.K} pairs a benchmark and {P.M_P:,} items a pair; the subject prior on the "
            f"{len(bench['U'])} of them whose score is neither lower-is-better nor unordered; "
            f"and archive-3's identity guards. The level is Gaussian and every Flags switch "
            f"stays at its default.")
    source = {
        "C1": ("archive-3's, the milder of the two guarded configurations in docs/findings.md "
               "\"Calibrating for the hidden test\"."),
        "C1m": (f"archive-3's sigma_mu and attr_scale, and mu0 matched to archive-3's B0 centre "
                f"under the {P.DATE_SHIFT}-year date shift (results/pool_fits.json \"mu0_m\")."),
        "C2": ("the pool's empirical-Bayes level, rounded to 3 decimals (results/pool_fits.json "
               "\"C2\")."),
    }[X]
    comment = (f"the level prior shipped, on the item-level scale: {source} prior.json holds the "
               f"same level, which tools/build_submission.py --pool checks against the locked "
               f"bundle; here it serves only when prior.json is unusable.")
    doc = ast.get_docstring(ast.parse(base), clean=False) or ""
    paras = doc.split("\n\n")
    flat = [" ".join(p.split()) for p in paras]
    i = [k for k, p in enumerate(flat) if p.startswith(MODEL_LEVEL_PARAGRAPH)]
    j = [k for k, p in enumerate(flat) if MODEL_BUILD_SENTENCE in p]
    if len(i) != 1 or len(j) != 1 or i == j:
        raise BuildError("archive-3's model.py docstring is not the text the pool build rewrites")
    for k, text in ((i[0], f"{lead} {tail}"),
                    (j[0], flat[j[0]].replace(MODEL_BUILD_SENTENCE, POOL_BUILD_SENTENCE))):
        paras[k] = _wrap(text) + ("\n" if paras[k].endswith("\n") else "")
    return {"doc": "\n\n".join(paras), "comment": textwrap.wrap(comment, 76, break_long_words=False,
                                                                break_on_hyphens=False),
            "facts": [f"candidate {X}", bench["revision"], *v.values()]}


def patch_model(source, doc, comment, level) -> str:
    """`source` (archive-3's model.py) with its module docstring replaced by `doc`,
    the comment block directly above LEVEL by `comment` (one '#: ' line each), and
    LEVEL's literal by `level`; nothing else is touched."""
    if '"""' in doc or "\\" in doc or not doc.endswith("\n"):
        raise BuildError("the docstring must end with a newline and hold no triple quote "
                         "or backslash")
    tree = ast.parse(source)
    d, lv = _docstring_node(tree), _level_assign(tree)
    if d is None or d.lineno != 1:
        raise BuildError("model.py does not open with its docstring")
    lines = source.splitlines(keepends=True)
    c0 = lv.lineno - 1
    while c0 > d.end_lineno and lines[c0 - 1].lstrip().startswith("#"):
        c0 -= 1
    literal = "LEVEL = {" + ", ".join(f'"{k}": {float(level[k])!r}' for k in LEVEL_KEYS) + "}\n"
    return "".join([f'"""{doc}"""\n'] + lines[d.end_lineno:c0]
                   + [f"#: {c}\n" for c in comment] + [literal] + lines[lv.end_lineno:])


def _patch_spans(base):
    """0-based line indices of `base` the patch may change: its docstring, and
    the comment block above LEVEL through LEVEL."""
    tree = ast.parse(base)
    d, lv = _docstring_node(tree), _level_assign(tree)
    lines = base.splitlines()
    c0 = lv.lineno - 1
    while c0 > d.end_lineno and lines[c0 - 1].lstrip().startswith("#"):
        c0 -= 1
    return set(range(d.lineno - 1, d.end_lineno)) | set(range(c0, lv.end_lineno))


def check_model(patched, base, level, text) -> list:
    """The model.py rule (PE 2, 10): `patched` has `base`'s AST once docstrings,
    comments and LEVEL's value are ignored; LEVEL is `level`; the docstring and
    the comment are the text asked for and the docstring names its facts; no
    line of `base` outside the docstring, the comment above LEVEL and LEVEL
    changed. Returns the 1-based lines of `base` that changed."""
    if model_rule_dump(patched) != model_rule_dump(base):
        raise BuildError("model.py differs from archive-3's outside its docstrings, comments "
                         "and LEVEL's value")
    tree = ast.parse(patched)
    if ast.get_docstring(tree, clean=False) != text["doc"]:
        raise BuildError("model.py's docstring is not the text asked for")
    got = {str(k): float(x) for k, x in ast.literal_eval(_level_assign(tree).value).items()}
    if got != {k: float(level[k]) for k in LEVEL_KEYS}:
        raise BuildError(f"model.py's LEVEL is {got}, not {level}")
    flat = " ".join(text["doc"].split())
    absent = [f for f in text["facts"] if f not in flat]
    if absent:
        raise BuildError(f"model.py's docstring does not name {absent}")
    allowed = _patch_spans(base)
    a, b = base.splitlines(), patched.splitlines()
    changed = set()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:                          # base lines replaced or deleted
            span = set(range(i1, i2))
        elif i1 in allowed or i1 - 1 in allowed:
            span = {i1} if i1 in allowed else {i1 - 1}   # lines inserted inside or after a span
        else:
            span = {i1}
        if not span <= allowed:
            raise BuildError(f"the patch changed model.py's lines "
                             f"{sorted(x + 1 for x in span - allowed)}, outside its docstring, "
                             "the comment above LEVEL and LEVEL")
        changed |= span
    return sorted(x + 1 for x in changed)


# --- the bundle and the archive against archive-3 -----------------------------------------

def bundle_keys(d) -> dict:
    """The key sets of a hier bundle that archive-4's must share with archive-3's
    (the subject table's keys are subject names and are not compared)."""
    sub = lambda x: sorted(x) if isinstance(x, dict) else None  # noqa: E731
    p = d.get("prior") if isinstance(d, dict) else None
    p = p if isinstance(p, dict) else {}
    return {"bundle": sub(d), "hyper": sub(d.get("hyper") if isinstance(d, dict) else None),
            "prior": sub(p), "spec": sub(p.get("spec")), "meta": sub(p.get("meta")),
            "s2d": sub(p.get("s2d"))}


def check_pool_bundle(d, what, level, L, a3, locked_hyper=None, id_cap=0.1, id_clip=(0.01, 0.2)):
    """_check_bundle for a pool bundle (PE 10): usable by HierPredictor, an
    attribute prior, the level `level` (model.py's LEVEL), a Gaussian level,
    fitted on L of tools/pool_benchmarks.json with nothing excluded, archive-3's
    identity guards (id_cap, sigma_theta the root of tau2_res clipped to
    id_clip), any hyperparameter the lock records equal, and archive-3's keys
    (`a3`, its bundle), so no per-benchmark level."""
    import numpy as np
    from paiec import prior as PR
    from paiec.hier import HierPredictor
    try:
        prior, hyper = PR.from_json(d)
        HierPredictor(prior, hyper)
    except Exception as exc:
        raise BuildError(f"{what} is unusable: {exc!r}") from None
    problems = []
    if prior is None or not prior.has_attributes:
        problems.append("no attribute prior")
    got = {k: getattr(hyper, k) for k in level}
    if got != level:
        problems.append(f"level prior {got}, model.py's LEVEL is {level}")
    if hyper.nu_mu != 0:
        problems.append(f"a Student-t level (nu_mu {hyper.nu_mu})")
    if hyper.excluded or sorted(hyper.included) != sorted(L):
        problems.append(f"fitted on {len(hyper.included)} benchmarks without "
                        f"{list(hyper.excluded)}, not on the {len(L)} of L")
    if hyper.id_cap != id_cap:
        problems.append(f"id_cap {hyper.id_cap}, not {id_cap}")
    tau2 = (prior.meta or {}).get("tau2_res") if prior is not None else None
    want = None if tau2 is None else math.sqrt(float(np.clip(tau2, id_clip[0], id_clip[1])))
    if want is None or hyper.sigma_theta != want:
        problems.append(f"sigma_theta {hyper.sigma_theta}, not the root of tau2_res {tau2} "
                        f"clipped to {list(id_clip)}")
    for k, x in sorted((locked_hyper or {}).items()):
        if k not in ("included", "excluded") and getattr(hyper, k, None) != x:
            problems.append(f"{k} {getattr(hyper, k, None)}, the lock's {x}")
    mine, theirs = bundle_keys(d), bundle_keys(a3)
    if mine != theirs:
        problems.append("keys " + ", ".join(f"{k} {mine[k]} against archive-3's {theirs[k]}"
                                            for k in mine if mine[k] != theirs[k]))
    if problems:
        raise BuildError(f"{what}: " + "; ".join(problems))
    return prior, hyper


def compare_archives(new, old, allowed=POOL_CHANGED) -> list:
    """The members `new` changes against `old` (PE 10): the same names, every
    member outside `allowed` byte for byte the same."""
    with zipfile.ZipFile(new) as a, zipfile.ZipFile(old) as b:
        na, nb = sorted(a.namelist()), sorted(b.namelist())
        if na != nb:
            raise BuildError(f"the archive's members {na} are not archive-3's {nb}")
        differ = [n for n in na if a.read(n) != b.read(n)]
    bad = [n for n in differ if n not in allowed]
    if bad:
        raise BuildError(f"the archive differs from archive-3 in {bad}, outside {list(allowed)}")
    return differ


def pool_build(out=OUT, candidate=None, *, eval_path=POOL_EVAL, build_path=POOL_BUILD,
               archive3=ARCHIVE_3, model=os.path.join(SUB, "model.py"),
               prior_out=os.path.join(SUB, "prior.json"), pairs=None, run_checks=True) -> dict:
    """archive-4 at `out` (PE 10). Every check that touches nothing comes first;
    then `out` is removed, the archive written and checked (removed again on a
    failure), and `prior_out` written last. `pairs` is the run check's session
    (the legacy five of paiec.data.load_pairs by default)."""
    problems = pool_env_problems()
    if problems:
        raise BuildError("--pool runs under the lock's settings, PYTHONHASHSEED=0 and one BLAS "
                         "thread: " + "; ".join(problems))
    from paiec import pool as P
    try:
        return _pool_build(P, out, candidate, eval_path, build_path, archive3, model, prior_out,
                           pairs, run_checks)
    except P.PoolError as exc:
        raise BuildError(f"pool check failed: {exc}") from None


def _pool_build(P, out, candidate, eval_path, build_path, archive3, model, prior_out, pairs,
                run_checks):
    build, lock, fixes = pool_state(eval_path, build_path, tuple(P.CANDIDATES))
    X = build["candidate"]
    if candidate is not None and candidate != X:
        raise BuildError(f"--candidate {candidate}, but the gate carried {X} ({build_path})")
    print(f"pool build of {X} ({P.CANDIDATES[X]} level), the candidate the gate carried:")
    want = {"PYTHONHASHSEED": POOL_HASHSEED, **{v: "1" for v in POOL_BLAS}}
    env = lock["env"] if isinstance(lock["env"], dict) else {}
    if {k: env.get(k) for k in want} != want:
        raise BuildError(f"the lock records the settings {env}, not {want}")
    files = lock["files"]

    # archive-3, kept apart from the output
    if not os.path.isfile(archive3):
        raise BuildError(f"no {archive3}: keep archive-3 there first (impl_plan step 1)")
    a3sha = _sha256_file(archive3)
    if not a3sha == ARCHIVE_3_SHA256 == files.get(LOCK_ARCHIVE_3):
        raise BuildError(f"{archive3} has sha256 {a3sha}; archive-3's is {ARCHIVE_3_SHA256}, "
                         f"the lock's {files.get(LOCK_ARCHIVE_3)}")
    if os.path.realpath(out) == os.path.realpath(archive3) or \
            (os.path.exists(out) and os.path.samefile(out, archive3)):
        raise BuildError(f"--out {out} is archive-3's copy")
    with zipfile.ZipFile(archive3) as z:
        a3_model, a3_prior = z.read("model.py"), z.read("prior.json")
    print(f"  archive-3: {archive3}, sha256 {a3sha}")

    # paiec/pool.py: the locked file or a recorded successor, by the AST rule
    if os.path.realpath(P.__file__) != os.path.realpath(POOL_PY):
        raise BuildError(f"paiec.pool was imported from {P.__file__}, not {POOL_PY}")
    with open(P.__file__, "rb") as f:
        how = check_pool_py(f.read(), lock, fixes, P.EVAL_LOADER)
    print(f"  paiec/pool.py: the {how} file; outside EVAL_LOADER its AST is the locked one")

    # the lists and the fits cache, against the lock
    lists = os.path.join(P.TOOLS, "pool_benchmarks.json")
    if not os.path.isfile(lists) or _sha256_file(lists) != files.get(LOCK_LISTS):
        raise BuildError(f"{lists} is missing or not the locked file")
    bench, _ = P.read_lists()
    if bench.get("revision") != P.REVISION:
        raise BuildError(f"{lists} is at revision {bench.get('revision')}, not {P.REVISION}")
    L, U = list(bench["L"]), list(bench["U"])
    free, fixed, T_full = P.load_fits()
    if P.fits_index_sha256() != files.get(LOCK_FITS_INDEX):
        raise BuildError("the fits cache index is not the locked one")
    print(f"  fits cache: index sha256 {P.fits_index_sha256()[:16]} as locked, "
          f"{len(free)} free and {len(fixed)} fixed-T fits, every entry's sha256 checked")

    # X*'s bundle, rebuilt with nothing excluded, byte for byte the locked one
    prior, hyper, _ = P.candidate_bundle(X, (), free, fixed, L, U, mu0_m=lock["mu0_m"],
                                         literal=True)
    data = P.to_bytes(prior, hyper)
    sha = _sha256(data)
    key = LOCK_BUNDLE.format(X)
    locked = {build_path: build["bundle_sha256"],
              f"the lock's {X} bundle": lock["bundles"][X]["sha256"],
              f"the lock's {key}": files.get(key)}
    off = {k: v for k, v in locked.items() if v != sha}
    if off:
        raise BuildError(f"the rebuilt {X} bundle has sha256 {sha}, not the locked one: {off}")
    stored = os.path.join(P.EVAL_DIR, "bundles", f"archive_{X}.json")
    if os.path.exists(stored) and _sha256_file(stored) != sha:
        raise BuildError(f"{stored} is not the locked bundle")
    print(f"  the {X} bundle, rebuilt from them with nothing excluded, is the locked one byte "
          f"for byte: sha256 {sha}")

    # its level: the locked one for X*, a 3-decimal literal
    level = {k: getattr(hyper, k) for k in LEVEL_KEYS}
    expected = {"C1": P.LEVEL_A3,
                "C1m": {"mu0": lock["mu0_m"], "sigma_mu": P.MATCHED_SIGMA_MU,
                        "attr_scale": P.ATTR_SCALE},
                "C2": lock["c2_literal"]}[X]
    for what, v in ((f"{X}'s level by the lock", expected), (build_path, build["level"]),
                    (f"the lock's {X} bundle", lock["bundles"][X]["level"])):
        if _level_of(v, what) != level:
            raise BuildError(f"the bundle's level {level} is not {what}'s {v}")
    if any(P.round_literal(x) != x for x in level.values()):
        raise BuildError(f"the level {level} is not a {P.LITERAL_DECIMALS}-decimal literal")
    d = json.loads(data)
    check_pool_bundle(d, f"the {X} bundle", level, L, json.loads(a3_prior),
                      lock["bundles"][X].get("hyper"), P.ID_CAP, P.ID_CLIP)
    print(f"  bundle check passed: fitted on L ({len(L)}) with nothing excluded, id_cap "
          f"{hyper.id_cap}, sigma_theta {hyper.sigma_theta:.6f}, archive-3's keys, no "
          "per-benchmark level")

    # model.py: archive-3's with the patch; the repository's must be one of the two
    base = a3_model.decode("utf-8")
    text = pool_model_text(X, level, bench, base)
    patched = patch_model(base, text["doc"], text["comment"], level)
    changed = check_model(patched, base, level, text)
    with open(model, "rb") as f:
        repo = f.read()
    if repo not in (a3_model, patched.encode("utf-8")):
        raise BuildError(f"{model} is neither archive-3's model.py nor the text archive-4 ships")
    print(f"  model.py: archive-3's with {len(changed)} of its lines in the docstring, the "
          "comment above LEVEL and LEVEL rewritten; with docstrings, comments and LEVEL's "
          f"value ignored its AST is archive-3's; LEVEL = {level}")

    if pairs is None and run_checks:
        from paiec.data import load_pairs
        pairs = load_pairs()
    if os.path.exists(out):
        os.remove(out)
    with tempfile.TemporaryDirectory(prefix="paiec-pool-") as tmp:
        staged = {n: os.path.join(tmp, n) for n in POOL_CHANGED}
        with open(staged["prior.json"], "wb") as f:
            f.write(data)
        with open(staged["model.py"], "wb") as f:
            f.write(patched.encode("utf-8"))
        if shipped_level(staged["model.py"]) != level:
            raise BuildError("the staged model.py's LEVEL is not the bundle's")
        members_ = members(staged["prior.json"])
        members_["model.py"] = staged["model.py"]
        check_imports(members_)
        print(f"import check passed: {', '.join(sorted(members_))}")
        write_zip(out, members_)
    try:
        differ = compare_archives(out, archive3)
        print(f"  the archive differs from archive-3 in {', '.join(differ)} only")
        if run_checks:
            print("run check (fresh interpreters, official format, a fresh predictor and one "
                  "labeled list per budget):")
            check_run(out, pairs, "hier")
            print("organisers' validator:")
            run_validator(out)
    except BaseException:
        os.remove(out)
        raise
    with open(prior_out, "wb") as f:
        f.write(data)
    osha = _sha256_file(out)
    with zipfile.ZipFile(out) as z:
        for i in z.infolist():
            print(f"  {i.filename:<24} {i.file_size:>8} bytes ({i.compress_size} compressed)")
    print(f"wrote {out}, {os.path.getsize(out)} bytes, hier with {X}'s pool bundle, sha256 {osha}; "
          f"{prior_out} is its prior.json. Keep it as dist/archive-4.zip (impl_plan step 12); "
          f"{model} is {'archive-3' if repo == a3_model else 'archive-4'}'s.")
    return {"candidate": X, "sha256": osha, "bundle_sha256": sha, "level": level,
            "differ": differ, "pool_py": how, "model_lines": changed}


def pool_main(args):
    if args.legacy or args.labeling:
        raise BuildError("--pool ships hier and no labeling.py, as archive-3 does")
    pool_build(args.out, args.candidate)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labeling", action="store_true", help="ship submission/labeling.py")
    ap.add_argument("--legacy", action="store_true",
                    help="ship the Predictor of commit b68492c instead of hier")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pool", action="store_true",
                    help="archive-4: the locked pool bundle of the candidate amendment 4's gate "
                         "carried (docs/plans/p2_pool_eval.md section 10); needs "
                         "PYTHONHASHSEED=0 and every BLAS variable at 1")
    ap.add_argument("--candidate", help="with --pool: the candidate the gate carried (C1, C1m "
                                        "or C2), which must be tools/pool_build.json's")
    args = ap.parse_args(argv)
    if args.pool:
        return pool_main(args)
    if args.candidate is not None:
        ap.error("--candidate needs --pool")
    from paiec.data import load_pairs

    if os.path.exists(args.out):
        os.remove(args.out)
    pairs = load_pairs()
    prior = os.path.join(SUB, "prior.json")
    kind = "legacy" if args.legacy else "hier"
    level = None
    if args.legacy:
        write_prior(pairs, prior)
    else:
        level = shipped_level()
        write_bundle(pairs, prior, level)
    check_prior(prior, kind, level)
    files = members(prior, labeling=args.labeling)
    check_imports(files)
    print(f"import check passed: {', '.join(sorted(files))}")
    write_zip(args.out, files)
    try:
        print("run check (fresh interpreters, official format, a fresh predictor and one "
              "labeled list per budget):")
        check_run(args.out, pairs, kind)
        print("organisers' validator:")
        run_validator(args.out)
    except BaseException:
        os.remove(args.out)
        raise
    with zipfile.ZipFile(args.out) as z:
        for i in z.infolist():
            print(f"  {i.filename:<24} {i.file_size:>8} bytes ({i.compress_size} compressed)")
    print(f"wrote {args.out}, {os.path.getsize(args.out)} bytes, {MODELS[kind]}"
          + ("" if args.labeling else "  (labeling.py left out)"))


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        sys.exit(f"BUILD FAILED: {exc}")
