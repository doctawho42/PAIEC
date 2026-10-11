"""Descriptive survey of the expanded measurement-db pool (revision 54074453).

Reads data/pool/<benchmark>/formatted_tables/{benchmarks,subjects,responses}.parquet
for every public benchmark whose catalogue says binary or mixed, item level.
Writes survey.json beside this file: per benchmark, the counts that decide
what hier's prior fit sees (eligible pairs = (subject_id, benchmark_id) with at
least 80 distinct items carrying a 0/1 response; the cell of an item is the
first 0/1 response in file order, the replica's rule), and the subject fields
that feed paiec.subjects.attrs and paiec.prior.canon_name.

Descriptive only: no predictor is fitted or scored. No item text is read
(items.parquet was not downloaded). Run with PYTHONDONTWRITEBYTECODE=1 so that
nothing is written inside the repository except this directory.
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from paiec.prior import canon_name  # noqa: E402
from paiec.subjects import ISO, attrs  # noqa: E402

POOL = os.path.join(ROOT, "data", "pool")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey.json")
FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]
MIN_ITEMS = 80


def direction(scale):
    try:
        return json.loads(scale).get("direction")
    except Exception:
        return None


def clean(v):
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def one(b, rec):
    t0 = time.time()
    d = os.path.join(POOL, b, "formatted_tables")
    out = {"benchmark": b, "response_type": rec.get("response_type"),
           "direction": direction(rec.get("response_scale")),
           "catalogue": {k: rec.get(k) for k in ("n_subjects", "n_items", "n_responses")}}
    t = pq.read_table(os.path.join(d, "responses.parquet"),
                      columns=["subject_id", "item_id", "benchmark_id", "trial",
                               "test_condition", "interactors", "response"],
                      read_dictionary=["subject_id", "item_id", "benchmark_id",
                                       "test_condition", "interactors"])
    n = t.num_rows
    resp = t.column("response").to_numpy(zero_copy_only=False)
    binary = (resp == 0.0) | (resp == 1.0)
    sid = t.column("subject_id").combine_chunks()
    iid = t.column("item_id").combine_chunks()
    if sid.null_count or iid.null_count:
        raise ValueError(f"null ids: {sid.null_count} subject, {iid.null_count} item")
    s_codes = sid.indices.to_numpy(zero_copy_only=False)
    i_codes = iid.indices.to_numpy(zero_copy_only=False)
    s_names = sid.dictionary.to_pylist()
    trial = t.column("trial").to_numpy(zero_copy_only=False)
    bids = t.column("benchmark_id").combine_chunks().dictionary.to_pylist()
    tc = t.column("test_condition").combine_chunks()
    head = t.column("interactors").slice(0, 200000).to_pylist()
    inter_nonempty = float(np.mean([bool(clean(v)) for v in head])) if n else 0.0
    out.update(n_rows=n, frac_binary=float(binary.mean()) if n else None,
               benchmark_ids=bids[:5], n_benchmark_ids=len(bids),
               n_test_conditions=len(tc.dictionary), max_trial=int(trial.max()) if n else None,
               interactors_nonempty_frac_first200k=inter_nonempty)
    # first 0/1 response per (subject, item), file order
    idx = np.flatnonzero(binary)
    key = s_codes[idx].astype(np.int64) * (len(iid.dictionary) + 1) + i_codes[idx]
    _, first = np.unique(key, return_index=True)
    rows = idx[first]
    out["n_cells"] = int(len(rows))
    out["cells_with_repeats"] = int(len(rows) - 0)  # placeholder overwritten below
    _, counts = np.unique(key, return_counts=True)
    out["cells_with_repeats"] = int((counts > 1).sum())
    # does the first row in file order have the smallest trial of its cell?
    tmin = pd.Series(trial[idx]).groupby(key).min().to_numpy()
    ukey = np.unique(key)
    out["first_row_not_min_trial"] = int((trial[rows] != tmin[np.searchsorted(ukey, key[first])]).sum())
    ps = s_codes[rows]
    y = resp[rows]
    per = pd.DataFrame({"s": ps, "y": y}).groupby("s").y.agg(["size", "mean"])
    elig = per[per["size"] >= MIN_ITEMS]
    out["n_subject_ids"] = int(len(per))
    out["n_eligible_pairs"] = int(len(elig))
    out["eligible_cells"] = int(elig["size"].sum())
    out["items_per_pair"] = ([int(x) for x in np.percentile(elig["size"], [0, 50, 100])]
                             if len(elig) else None)
    out["pair_rate"] = ([round(float(x), 4) for x in np.percentile(elig["mean"], [10, 50, 90])]
                        if len(elig) else None)
    out["pooled_rate"] = (float((elig["size"] * elig["mean"]).sum() / elig["size"].sum())
                          if len(elig) else None)
    # subjects of eligible pairs
    subj = pd.read_parquet(os.path.join(d, "subjects.parquet"))
    out["subject_columns_ok"] = all(f in subj.columns for f in FIELDS + ["subject_id"])
    subj = subj.drop_duplicates("subject_id").set_index("subject_id")
    names, iso, dated, has_name, providers, missing = [], 0, 0, 0, {}, 0
    for s in elig.index:
        sname = s_names[s]
        if sname not in subj.index:
            missing += 1
            sd = {f: "" for f in FIELDS}
        else:
            r = subj.loc[sname]
            sd = {f: clean(r[f]) for f in FIELDS}
        k = canon_name(sd)
        names.append(k)
        rd = sd["release_date"].strip()
        if rd:
            dated += 1
            iso += bool(ISO.fullmatch(rd))
        has_name += bool(sd["normalized_name"].strip())
        a = attrs(sd)
        providers[a["provider"]] = providers.get(a["provider"], 0) + 1
    out["eligible_subjects_missing_from_table"] = missing
    out["dated"], out["dated_iso"], out["with_normalized_name"] = dated, iso, has_name
    out["providers"] = providers
    out["canon_names"] = names
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main():
    m = json.load(open(os.path.join(POOL, "manifest.json")))
    todo = {b: r for b, r in m["benchmarks"].items()
            if r.get("status") == "ok" and r.get("granularity") == "item"
            and r.get("response_type") in ("binary", "mixed")
            and "responses.parquet" in r.get("files", {})}
    res = {}
    with cf.ProcessPoolExecutor(max_workers=int(os.environ.get("JOBS", "3"))) as ex:
        futs = {ex.submit(one, b, r): b for b, r in sorted(todo.items())}
        for f in cf.as_completed(futs):
            b = futs[f]
            try:
                res[b] = f.result()
            except Exception as exc:
                res[b] = {"benchmark": b, "error": repr(exc)}
            print(b, res[b].get("n_eligible_pairs"), res[b].get("seconds"), flush=True)
    json.dump({"revision": m["revision"], "benchmarks": dict(sorted(res.items()))},
              open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
