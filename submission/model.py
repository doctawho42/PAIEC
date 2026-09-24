"""Competition entry point: predict(input, labeled) -> float.

The archive carries the run-time half of the research package beside this file,
as paiec_rt/ (predict, fitting, irt, subjects, mcq), and the attribute-prior
coefficients in prior.json, both put there by tools/build_submission.py. The
package has its own name in the archive so that a module called paiec already
imported by the platform's process cannot stand in for it.

The package is imported here, at module level, with this file's directory first
on sys.path. The validator restores sys.path as soon as this module is loaded,
so an import deferred to the first predict() call can fail after every check
has passed, and a predictor that swallows errors then answers the fallback for
every target. Importing now turns a packaging mistake into a failed check.
Nothing heavier than numpy is imported here; scipy and scikit-learn load on the
first call that needs them, since every evaluation worker re-imports this file.

BLAS and OpenMP are held to one thread before numpy loads, unless the platform
says otherwise: up to 16 workers run side by side, and a thread per core in
each made the difficulty fit several times slower. paiec_rt.predict limits the
libraries loaded later as well, at every fit.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if not sys.path or sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from paiec_rt.predict import Predictor  # noqa: E402
from paiec_rt.subjects import Spec  # noqa: E402


def _prior(path):
    """(coef, spec) from prior.json. Unusable means no attribute prior, not no
    predictions: the build refuses to package a missing or broken file."""
    try:
        with open(path) as f:
            d = json.load(f)
        coef, spec = np.asarray(d["coef"], float), Spec.from_dict(d["spec"])
        if coef.ndim != 1 or not np.all(np.isfinite(coef)):
            raise ValueError("coefficients are not a finite vector")
        return coef, spec
    except Exception as exc:
        print(f"prior.json unusable ({exc!r}); predicting without the attribute prior",
              file=sys.stderr)
        return None, None


PREDICTOR = Predictor(*_prior(os.path.join(_HERE, "prior.json")))


def predict(input, labeled=None):
    return PREDICTOR.predict(input, labeled)
