"""Fit the prior, bake it into submission/, zip the archive, check it.

    python tools/build_submission.py               # needs the data, see paiec.data
    python tools/build_submission.py --labeling    # also ship submission/labeling.py
    python tools/build_submission.py --legacy      # the Predictor of commit b68492c instead

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
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labeling", action="store_true", help="ship submission/labeling.py")
    ap.add_argument("--legacy", action="store_true",
                    help="ship the Predictor of commit b68492c instead of hier")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
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
