"""Competition entry point: predict(input, labeled) -> float.

Self-contained by design. The evaluator imports this file from the archive root,
so it must not depend on the research package; the attribute-prior coefficients
are baked in by tools/build_submission.py rather than fitted at run time.

Status: this reproduces the configuration that scores 0.181 in the replica when
labels pool across subjects and 0.204 when they do not. It has NOT been run
against the real evaluator. Two behaviours depend on open questions with the
organisers, both flagged in docs/findings.md: whether `labeled` carries other
subjects' labels, and whether state may persist between calls.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_PRIOR = None
_P = None


def _load():
    global _PRIOR, _P
    if _P is not None:
        return _P
    from paiec.predict import Predictor
    from paiec.subjects import Spec
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prior.json")
    coef, spec = None, None
    if os.path.exists(path):
        d = json.load(open(path))
        import numpy as np
        coef, spec = np.array(d["coef"]), Spec.from_dict(d["spec"])
    _P = Predictor(coef, spec)
    return _P


def predict(input, labeled=None):
    return _load().predict(input, labeled)
