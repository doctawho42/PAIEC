"""Build Pair objects from measurement-db.

Only benchmarks whose responses are genuinely binary are used. mmdocrag is a
fraction-valued benchmark (4.8% of its responses are 0 or 1) and matharena has a
0.1% non-binary tail; both are handled explicitly rather than silently coerced.
"""
import os

import pandas as pd

from paiec.evaluator import Pair, Response

#: where the measurement-db tables live; override with PAIEC_DATA
DATA_DIR = os.environ.get(
    "PAIEC_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

BINARY = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
FRACTIONAL = ["mmdocrag"]
SUBJECT_FIELDS = ["normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra"]


def _clean(v):
    return "" if pd.isna(v) else str(v)


def load_pairs(benchmarks=None, min_items=80, drop_nonbinary=True):
    benchmarks = benchmarks or BINARY
    pairs = []
    for b in benchmarks:
        resp = pd.read_parquet(os.path.join(DATA_DIR, b, "response.parquet"))
        items = pd.read_parquet(os.path.join(DATA_DIR, b, "items.parquet")).set_index("item_id")
        subs = pd.read_parquet(os.path.join(DATA_DIR, b, "subjects.parquet")).set_index("subject_id")
        if drop_nonbinary:
            resp = resp[resp.response.isin([0.0, 1.0])]
        item_dicts = {
            iid: {"item_content": _clean(row.content),
                  "item_features": _clean(row.item_features),
                  "interactors": "",
                  "benchmark_id": _clean(row.benchmark_id),
                  "_key": str(iid)}
            for iid, row in items.iterrows()
        }
        for (sid, bid), grp in resp.groupby(["subject_id", "benchmark_id"], sort=False):
            if grp.item_id.nunique() < min_items:
                continue
            srow = subs.loc[sid] if sid in subs.index else None
            subject = {f: (_clean(srow[f]) if srow is not None and f in srow else "")
                       for f in SUBJECT_FIELDS}
            subject["_sid"] = str(sid)
            responses = [Response(item_key=str(iid), item=item_dicts[iid], label=int(y))
                         for iid, y in zip(grp.item_id.values, grp.response.values)]
            pairs.append(Pair(subject=subject, subject_id=str(sid),
                              benchmark_id=str(bid), responses=responses))
    return pairs
