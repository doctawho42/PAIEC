"""Acquisition policy for the submission.

Deliberately absent logic. Three policies were measured against the evaluator's
default random one in the replica: A-optimal selection targeting the evaluation
pool, the same with ALC-weight-aware thresholds, and coverage-first selection.
The paired gain of the best of them over random was -0.0000 +- 0.0013 across
four splits, and the A-optimal one was consistently worse because it starves the
shared difficulty estimate of distinct items.

Shipping no labeling.py lets the evaluator use its own random policy, which is
what the measurements say to do. This file documents that as a decision rather
than an omission; delete it before packaging, or leave it, since it only defers.
"""


def acquisition_function(input, *, labeled=None, context=None) -> bool:
    if context is None or context["labels_remaining"] <= 0:
        return False
    if context["items_remaining"] <= context["labels_remaining"]:
        return True
    import numpy as np
    seed = abs(hash((context["subject_id"], context["benchmark_id"],
                     context["labels_acquired"], context["items_remaining"]))) % (2 ** 32)
    return bool(np.random.default_rng(seed).random()
                < context["labels_remaining"] / context["items_remaining"])
