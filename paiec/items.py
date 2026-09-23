"""Predict item difficulty from item text, transferring across benchmarks.

Target: the logit of how often subjects get the item right, standardised inside
its own benchmark. Standardising is what makes the target transferable: the
absolute difficulty level of a held-out benchmark is unknowable from text alone
and has to come from the acquired labels, but the ordering inside the benchmark
is exactly what text can supply.
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict


def item_table(pairs):
    agg = defaultdict(lambda: [0, 0])
    meta = {}
    for p in pairs:
        for r in p.responses:
            k = (p.benchmark_id, r.item_key)
            agg[k][0] += 1
            agg[k][1] += r.label
            if k not in meta:
                meta[k] = r.item
    rows = []
    for (bid, key), (n, s) in agg.items():
        rows.append(dict(benchmark_id=bid, item_key=key, n=n, k=s,
                         content=meta[(bid, key)]["item_content"],
                         features=meta[(bid, key)]["item_features"]))
    df = pd.DataFrame(rows)
    df["logit"] = np.log((df.k + 0.5) / (df.n - df.k + 0.5))
    g = df.groupby("benchmark_id").logit
    df["z"] = (df.logit - g.transform("mean")) / g.transform("std").replace(0, 1)
    return df


_CODE = re.compile(r"```|def |class |import |#include|function ")
_MATH = re.compile(r"\$|\\\(|\\frac|\\boxed|\\begin")


def numeric_features(df):
    c = df.content.fillna("")
    return np.column_stack([
        np.log1p(c.str.len()),
        np.log1p(c.str.count(r"\s+")),
        np.log1p(c.str.count("\n")),
        c.str.count(r"\d") / (c.str.len() + 1),
        c.str.contains(_CODE).astype(float),
        c.str.contains(_MATH).astype(float),
        c.str.count(r"\?").clip(0, 10),
        np.log1p(df.features.fillna("").str.count(";")),
    ])
