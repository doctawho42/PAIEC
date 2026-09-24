"""Fit the attribute prior, bake it into submission/, zip the archive, check it.

    python tools/build_submission.py               # needs the data, see paiec.data
    python tools/build_submission.py --labeling    # also ship submission/labeling.py

The archive holds model.py, prior.json and requirements.txt at its root and the
run-time modules of the research package under paiec_rt/, a name of their own
so that no paiec the platform has imported can stand in for them. labeling.py
stays out by default: no acquisition policy measured better than the platform's
random one.

Anything that would only surface on the platform fails the build instead, and
dist/paiec.zip exists only after every check passed:
  * prior.json missing, not finite, or the wrong length for its spec;
  * a shipped module importing a local module the archive does not carry (the
    research package's own absolute imports included, since it ships renamed),
    or importing anything but the standard library and numpy at module level;
  * a fresh interpreter that loads model.py the way the validator does and
    predicts real items in the official format at every budget, with one shared
    labeled list per budget, and sees a fallback, a module from outside the
    archive, or anything but a native float in [0, 1];
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUB = os.path.join(ROOT, "submission")
OUT = os.path.join(ROOT, "dist", "paiec.zip")
PACKAGE = ("__init__.py", "predict.py", "fitting.py", "irt.py", "subjects.py", "mcq.py")
#: the name paiec's run-time modules ship under
RUNTIME = "paiec_rt"
VALIDATOR = os.path.join(ROOT, "third_party", "paiec_baseline", "check_submission_zip.py")
#: functions only experiments call, allowed to import modules the archive leaves out
OFFLINE_ONLY = {(f"{RUNTIME}/irt.py", "evaluate")}
#: what shipped code may import at module level besides the standard library and itself
MODULE_LEVEL_OK = {"numpy"}
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra")
BUDGETS = (0, 1, 3, 7, 15, 31)


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


def write_prior(pairs, path):
    from paiec.predict import fit_prior
    coef, spec = fit_prior(pairs)
    with open(path, "w") as f:
        json.dump({"coef": coef.tolist(), "spec": spec.to_dict()}, f)
    print(f"prior fitted on {len(pairs)} pairs, {len(coef)} coefficients -> {path}")


def check_prior(path):
    """A NaN median would silently switch the prior off for every subject that
    lacks that attribute, so every number must be finite."""
    import math
    from paiec.subjects import Spec, attrs, design_row
    try:
        with open(path) as f:
            d = json.load(f)
        spec = Spec.from_dict(d["spec"])
        coef = [float(c) for c in d["coef"]]
    except Exception as exc:
        raise BuildError(f"{path} is unusable: {exc!r}") from None
    if not all(map(math.isfinite, coef + [spec.med_days, spec.med_log_size])):
        raise BuildError(f"{path} holds a non-finite number")
    if len(coef) != len(design_row(attrs({}), spec)):
        raise BuildError(f"{path}: {len(coef)} coefficients for a spec of "
                         f"{len(design_row(attrs({}), spec))} columns")


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


RUN = r"""
import importlib.util, json, math, os, sys, time
root, data = sys.argv[1], json.load(open(sys.argv[2]))
before = list(sys.path)
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("submission_model", os.path.join(root, "model.py"))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
sys.path[:] = before
out = {}
for B, labeled in data["labeled"].items():
    P = type(model.PREDICTOR)(model.PREDICTOR.prior_coef, model.PREDICTOR.prior_spec)
    t0, se, slowest = time.time(), 0.0, 0.0
    for inp, ys in data["targets"]:
        t = time.time()
        p = P.predict(inp, labeled)
        slowest = max(slowest, time.time() - t)
        assert type(p) is float and math.isfinite(p) and 0 <= p <= 1, repr(p)
        se += sum((p - y) ** 2 for y in ys) / len(ys)
    assert P.failures == 0, f"budget {B}: {P.failures} fallbacks"
    out[B] = dict(brier=se / len(data["targets"]), seconds=time.time() - t0, slowest=slowest)
stray = sorted(n for n, m in sys.modules.items() if n.split(".")[0] in ("paiec", "paiec_rt")
               and not os.path.abspath(getattr(m, "__file__", "") or "").startswith(root))
assert not stray, f"modules from outside the archive: {stray}"
print(json.dumps(out))
"""


def check_run(zip_path, pairs):
    with tempfile.TemporaryDirectory(prefix="paiec-run-") as tmp:
        root = os.path.join(tmp, "submission")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        data = os.path.join(tmp, "session.json")
        with open(data, "w") as f:
            json.dump(session(pairs), f)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, "-c", RUN, os.path.realpath(root), data],
                           cwd=tmp, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(f"run check failed:\n{r.stdout}{r.stderr}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    for B, v in res.items():
        print(f"  budget {B:>2}: Brier {v['brier']:.4f}  {v['seconds']:.1f}s  "
              f"slowest call {v['slowest']:.2f}s")


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
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    from paiec.data import load_pairs

    if os.path.exists(args.out):
        os.remove(args.out)
    pairs = load_pairs()
    prior = os.path.join(SUB, "prior.json")
    write_prior(pairs, prior)
    check_prior(prior)
    files = members(prior, labeling=args.labeling)
    check_imports(files)
    print(f"import check passed: {', '.join(sorted(files))}")
    write_zip(args.out, files)
    try:
        print("run check (fresh interpreter, official format, one labeled list per budget):")
        check_run(args.out, pairs)
        print("organisers' validator:")
        run_validator(args.out)
    except BaseException:
        os.remove(args.out)
        raise
    print(f"wrote {args.out}" + ("" if args.labeling else "  (labeling.py left out)"))


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        sys.exit(f"BUILD FAILED: {exc}")
