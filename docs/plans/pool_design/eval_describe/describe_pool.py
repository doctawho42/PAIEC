"""Descriptive statistics of the expanded pool (no predictor): per benchmark,
eligible pairs (>= 80 distinct items with a 0/1 response), subjects, items,
responses per item, pair rates. Writes JSON next to this script."""
import json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = "/Users/nikitapolomosnov/PycharmProjects/PAIEC/data/pool"
m = json.load(open(os.path.join(ROOT, "manifest.json")))
B = m["benchmarks"]
names = sorted(b for b, r in B.items() if r.get("status") == "ok"
               and r.get("granularity") == "item" and r.get("response_type") in ("binary", "mixed"))
out = {}
t0 = time.time()
for b in names:
    r = B[b]
    path = os.path.join(ROOT, b, "formatted_tables", "responses.parquet")
    try:
        t = pq.read_table(path, columns=["subject_id", "item_id", "response"],
                          read_dictionary=["subject_id", "item_id"])
    except Exception as e:
        out[b] = {"error": repr(e)[:200]}
        continue
    n_all = t.num_rows
    resp = t.column("response").to_numpy(zero_copy_only=False)
    keep = (resp == 0.0) | (resp == 1.0)
    s = t.column("subject_id").combine_chunks().indices.to_numpy(zero_copy_only=False)[keep]
    i = t.column("item_id").combine_chunks().indices.to_numpy(zero_copy_only=False)[keep]
    y = resp[keep]
    del t
    rec = {"response_type": r["response_type"], "direction": json.loads(r["response_scale"]).get("direction"),
           "n_responses": int(n_all), "binary_share": float(keep.mean()) if n_all else 0.0}
    if len(y) == 0:
        rec["eligible_pairs"] = 0
        out[b] = rec
        continue
    ni = int(i.max()) + 1
    key = s.astype(np.int64) * ni + i
    uk, inv = np.unique(key, return_inverse=True)
    pair_of_u = uk // ni
    items_per_subject = np.bincount(pair_of_u)
    resp_per_cell = np.bincount(inv)
    succ = np.bincount(s, weights=y, minlength=items_per_subject.size)
    tot = np.bincount(s, minlength=items_per_subject.size)
    elig = np.flatnonzero(items_per_subject >= 80)
    rate = succ[elig] / np.maximum(tot[elig], 1)
    rec.update({
        "subjects_with_binary": int((items_per_subject > 0).sum()),
        "eligible_pairs": int(len(elig)),
        "items": int(np.unique(i).size),
        "items_per_eligible_pair": [int(x) for x in np.percentile(items_per_subject[elig], [0, 50, 100])] if len(elig) else None,
        "responses_per_cell_mean": float(resp_per_cell.mean()),
        "responses_per_cell_max": int(resp_per_cell.max()),
        "pair_rate_mean": float(rate.mean()) if len(elig) else None,
        "pair_rate_sd": float(rate.std()) if len(elig) else None,
        "pair_logit": [float(np.log((succ[e] + 0.5) / (tot[e] - succ[e] + 0.5))) for e in elig],
    })
    out[b] = rec
print(f"{len(names)} benchmarks in {time.time()-t0:.0f}s", file=sys.stderr)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_describe.json"), "w"))
