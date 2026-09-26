"""Shared loader for the step-2 variance-component analysis (read-only on the
repo; see README.md). One flat frame of every eligible binary response, in the order
load_pairs gives them (the first recorded response of an item is the one the
platform reveals)."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "flat.parquet")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(p / (1 - p))


def eligible_pairs():
    from paiec import data as D, official as O
    return O.eligible(D.load_pairs())


def flat():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    rows = []
    for p in eligible_pairs():
        seen = {}
        for r in p.responses:
            t = seen.get(r.item_key, 0)
            seen[r.item_key] = t + 1
            rows.append((p.benchmark_id, p.subject_id, r.item_key, int(r.label), t))
    df = pd.DataFrame(rows, columns=["bench", "subject", "item", "y", "rep"])
    df.to_parquet(CACHE)
    return df
