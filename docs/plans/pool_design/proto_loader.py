"""Prototype pool loader, for timing the prior fit only (not repository code).

load_benchmark(b, max_pairs=None, max_items=None) -> list[Pair] reads
data/pool/<b>/formatted_tables/{subjects,responses}.parquet, keeps 0/1
responses, takes each (subject_id, item_id)'s first 0/1 response in file order
(the replica's rule; paiec.prior.benchmark_fit reads only that one), keeps
subjects with at least 80 distinct items, and builds one Response per cell.
Items carry no content or features (items.parquet was not fetched);
interactors come from the cell's response row. Caps, when given, are
deterministic: pairs ranked by stable_hash('pool-pair', b, subject_id), items
by stable_hash('pool-item', b, item_id) so every pair keeps the same item
subset (items stay shared across subjects).
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from paiec.evaluator import Pair, Response, stable_hash  # noqa: E402

POOL = os.path.join(ROOT, "data", "pool")
FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]


def _clean(v):
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def load_benchmark(b, max_pairs=None, max_items=None, min_items=80):
    import pyarrow.compute as pc
    d = os.path.join(POOL, b, "formatted_tables")
    t = pq.read_table(os.path.join(d, "responses.parquet"),
                      columns=["subject_id", "item_id", "interactors", "response"],
                      read_dictionary=["subject_id", "item_id", "interactors"])
    y = t.column("response").to_numpy(zero_copy_only=False)
    sid, iid, inter = (t.column(c).combine_chunks() for c in ("subject_id", "item_id", "interactors"))
    s_names, i_names = sid.dictionary.to_pylist(), iid.dictionary.to_pylist()
    in_names = inter.dictionary.to_pylist() + [""]
    s = sid.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    i = iid.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    ic = pc.fill_null(inter.indices, len(in_names) - 1).to_numpy(zero_copy_only=False)
    del t
    idx = np.flatnonzero((y == 0.0) | (y == 1.0))
    key = s[idx] * (len(i_names) + 1) + i[idx]
    _, first = np.unique(key, return_index=True)
    rows = idx[np.sort(first)]                  # each cell's first 0/1 response, file order
    if max_items is not None:
        items = np.unique(i[rows])
        if len(items) > max_items:
            rank = sorted(items, key=lambda k: stable_hash("pool-item", b, i_names[k]))
            rows = rows[np.isin(i[rows], np.array(rank[:max_items]))]
    counts = np.bincount(s[rows], minlength=len(s_names))
    order = list(dict.fromkeys(s[rows].tolist()))          # first appearance
    keep = [k for k in order if counts[k] >= min_items]
    if max_pairs is not None and len(keep) > max_pairs:
        chosen = set(sorted(keep, key=lambda k: stable_hash("pool-pair", b, s_names[k]))[:max_pairs])
        keep = [k for k in keep if k in chosen]
    subj = pd.read_parquet(os.path.join(d, "subjects.parquet")).drop_duplicates("subject_id")
    subj = subj.set_index("subject_id")
    item_dicts, pairs = {}, []
    by_s = {}
    for r in rows[np.isin(s[rows], np.array(keep))]:
        by_s.setdefault(s[r], []).append(r)
    for k in keep:
        name = s_names[k]
        row = subj.loc[name] if name in subj.index else None
        subject = {f: (_clean(row[f]) if row is not None else "") for f in FIELDS}
        subject["_sid"] = str(name)
        resp = []
        for r in by_s[k]:
            it = item_dicts.get(i[r])
            if it is None:
                it = item_dicts[i[r]] = {"item_content": "", "item_features": "",
                                         "interactors": in_names[ic[r]], "benchmark_id": b,
                                         "_key": str(i_names[i[r]])}
            resp.append(Response(item_key=it["_key"], item=it, label=int(y[r])))
        pairs.append(Pair(subject=subject, subject_id=str(name), benchmark_id=b, responses=resp))
    return pairs
