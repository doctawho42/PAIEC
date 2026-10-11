"""The archive with a pool-sized prior.json: size, load time per worker, validator.

Not repository code. Builds a C1-shaped bundle from the K100/M2000 prototype
(the pool's subject prior and empirical-Bayes hyperparameters with
submission/model.py's LEVEL written over mu0, sigma_mu and attr_scale), writes
it and an archive with the repository's own member list beside this file, and
then (1) loads model.py from each archive in fresh interpreters the way the
organisers' validator does, timing the load, (2) runs the organisers' validator
on the pool archive. Nothing is written inside the repository outside this
directory (TMPDIR is pointed here). No prediction is scored against a label.
"""
import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PD)))
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP
tempfile.tempdir = TMP
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from paiec import prior as PR  # noqa: E402
from paiec.hier import HierPredictor, Hyper  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("bs", os.path.join(ROOT, "tools", "build_submission.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

LOAD = r"""
import os, sys, time, importlib.util
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
t0 = time.perf_counter()
import numpy
t1 = time.perf_counter()
root = sys.argv[1]
before = list(sys.path)
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("submission_model", os.path.join(root, "model.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path[:] = before
t2 = time.perf_counter()
P = m.make()
t3 = time.perf_counter()
print(m.MODEL, round(1e3 * (t1 - t0), 2), round(1e3 * (t2 - t1), 2), round(1e3 * (t3 - t2), 3),
      len(P.prior.table) if P.prior is not None else 0, len(P.hyper.included))
"""


def load_times(zip_path, n=7):
    out = []
    with tempfile.TemporaryDirectory(dir=TMP) as tmp:
        root = os.path.join(tmp, "sub")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        for _ in range(n):
            r = subprocess.run([sys.executable, "-c", LOAD, root], capture_output=True, text=True,
                               env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, cwd=tmp)
            if r.returncode:
                raise RuntimeError(r.stderr[-2000:])
            out.append(r.stdout.split())
    return {"model": out[0][0], "numpy_import_ms_median": statistics.median(float(o[1]) for o in out),
            "model_py_exec_ms_median": statistics.median(float(o[2]) for o in out),
            "model_py_exec_ms_max": max(float(o[2]) for o in out),
            "make_ms_median": statistics.median(float(o[3]) for o in out),
            "table_rows": int(out[0][4]), "included": int(out[0][5])}


def main():
    fits = pickle.load(open(os.path.join(PD, "fits_K100_M2000.pkl"), "rb"))
    multi = {b: f for b, f in fits.items() if f["n_pairs"] >= 2}
    rows = []
    for b in sorted(multi):
        t = np.array(multi[b]["t"])
        for s, z, v in zip(multi[b]["subjects"], t - t.mean(), multi[b]["t_var"]):
            rows.append((b, s, float(z), float(v)))
    prior = PR.from_standings(rows, 2.0, sorted(fits), ())
    fitted = json.load(open(os.path.join(PD, "build_K100_M2000.json")))["hyper_fitted"]
    level = bs.shipped_level()
    hyper = Hyper(**{**fitted, **level, "excluded": (), "included": tuple(sorted(fits))})
    path = os.path.join(HERE, "c1_proto_prior.json")
    PR.save(path, prior, hyper)
    p2, h2 = PR.load(path)
    HierPredictor(p2, h2)
    rep = {"prior_json_bytes": os.path.getsize(path), "level_written": level}
    t0 = time.perf_counter()
    for _ in range(50):
        with open(path) as f:
            p2, h2 = PR.from_json(json.load(f))
        HierPredictor(p2, h2)
    rep["json_load_from_json_construct_ms"] = (time.perf_counter() - t0) / 50 * 1e3
    shipped = os.path.join(ROOT, "submission", "prior.json")
    t0 = time.perf_counter()
    for _ in range(50):
        with open(shipped) as f:
            p3, h3 = PR.from_json(json.load(f))
        HierPredictor(p3, h3)
    rep["shipped_json_load_from_json_construct_ms"] = (time.perf_counter() - t0) / 50 * 1e3
    rep["shipped_prior_json_bytes"] = os.path.getsize(shipped)

    zips = {}
    for tag, pj in (("shipped", shipped), ("c1_proto", path)):
        files = bs.members(pj)
        bs.check_imports(files)
        z = os.path.join(HERE, f"archive_{tag}.zip")
        bs.write_zip(z, files)
        zips[tag] = z
        with zipfile.ZipFile(z) as zz:
            rep[f"archive_{tag}"] = {"bytes": os.path.getsize(z),
                                     "members": {i.filename: [i.file_size, i.compress_size]
                                                 for i in zz.infolist()}}
        rep[f"load_{tag}"] = load_times(z)
    # the organisers' validator on the pool archive (full checks: it calls predict)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with tempfile.TemporaryDirectory(dir=TMP) as tmp:
        r = subprocess.run([sys.executable, bs.VALIDATOR, os.path.abspath(zips["c1_proto"])],
                           cwd=tmp, env=env, capture_output=True, text=True)
    rep["validator_c1_proto"] = {"returncode": r.returncode, "stdout_head": r.stdout.strip()[:300],
                                 "stderr_tail": r.stderr.strip()[-300:]}
    with open(os.path.join(HERE, "code_archive.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1)[:6000])


if __name__ == "__main__":
    main()
