"""Fit the attribute prior, bake it into submission/, and zip the archive.

The evaluator unpacks model.py at the archive root, so the package travels with
it. Run from the repository root:

    python tools/build_submission.py
    python -c "import zipfile;print(zipfile.ZipFile('dist/paiec.zip').namelist())"
"""
import json
import os
import shutil
import zipfile

from paiec.data import load_pairs
from paiec.predict import fit_prior

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(include_labeling=False):
    pairs = load_pairs()
    coef, spec = fit_prior(pairs)
    sub = os.path.join(ROOT, "submission")
    json.dump({"coef": coef.tolist(), "spec": spec.to_dict()},
              open(os.path.join(sub, "prior.json"), "w"))
    print(f"prior fitted on {len(pairs)} pairs, {len(coef)} coefficients")

    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", "paiec.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(sub, "model.py"), "model.py")
        z.write(os.path.join(sub, "prior.json"), "prior.json")
        if include_labeling:
            z.write(os.path.join(sub, "labeling.py"), "labeling.py")
        z.write(os.path.join(ROOT, "requirements.txt"), "requirements.txt")
        for name in ["__init__.py", "predict.py", "fitting.py", "irt.py",
                     "subjects.py", "mcq.py"]:
            z.write(os.path.join(ROOT, "paiec", name), f"paiec/{name}")
    print(f"wrote {out}")
    print("labeling.py is omitted by default: no acquisition policy measured better "
          "than the evaluator's own random one.")


if __name__ == "__main__":
    main()
