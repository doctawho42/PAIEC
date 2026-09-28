"""Counts of the public data that the report quotes (docs/report/draft.md section 2.1).

Reads data/<benchmark>/response.parquet and, for matharena, items.parquet, and
counts:

* per binary benchmark: the eligible pairs (a subject's binary responses on at
  least 80 distinct items, paiec.data.load_pairs' rule), their responses and
  their distinct items, with the totals over the five benchmarks;
* for matharena: the items with at least one binary response and the
  competitions they span (the `competition` key of item_features); the items
  and competitions inside eligible pairs; the competitions that occur only on
  subjects below the 80-item floor; and the Kangaroo items (competition name
  starting with "kangaroo").

It fits nothing and runs no model; it needs data/ (python -m paiec.fetch).

    python experiments/data_counts.py      # a few seconds, well under 1 GB

Output: results/data_counts.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paiec.data import BINARY, DATA_DIR  # noqa: E402

OUT = os.path.join(ROOT, "results", "data_counts.json")
MIN_ITEMS = 80


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def features(s):
    """item_features 'k=v;k=v' as a dict (the format of measurement-db)."""
    out = {}
    for part in (s or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def binary(resp):
    return resp[resp.response.isin([0.0, 1.0])]


def eligible_mask(resp, min_items=MIN_ITEMS):
    """Rows of binary responses that belong to an eligible (subject, benchmark) pair."""
    n = resp.groupby(["subject_id", "benchmark_id"]).item_id.transform("nunique")
    return n >= min_items


def pair_counts(resp):
    b = binary(resp)
    m = eligible_mask(b)
    e = b[m]
    return {"responses": int(len(resp)), "binary_responses": int(len(b)),
            "eligible_pairs": int(e.groupby(["subject_id", "benchmark_id"]).ngroups),
            "responses_in_eligible_pairs": int(len(e)),
            "distinct_items_in_eligible_pairs": int(e.item_id.nunique())}


def matharena_counts(resp, items):
    comp = items.set_index("item_id").item_features.map(lambda s: features(s).get("competition"))
    b = binary(resp)
    e = b[eligible_mask(b)]
    all_items = pd.Index(b.item_id.unique())
    el_items = pd.Index(e.item_id.unique())
    c_all = set(comp.reindex(all_items).dropna())
    c_el = set(comp.reindex(el_items).dropna())
    only_below = sorted(c_all - c_el)
    below_items = all_items.difference(el_items)
    kang = comp[comp.str.startswith("kangaroo", na=False)].index
    return {
        "items_with_a_binary_response": int(len(all_items)),
        "competitions_of_those_items": int(len(c_all)),
        "items_in_eligible_pairs": int(len(el_items)),
        "competitions_in_eligible_pairs": int(len(c_el)),
        "competitions_only_below_the_floor": only_below,
        "items_only_below_the_floor": int(len(below_items)),
        "items_of_those_competitions": {c: int((comp.reindex(all_items) == c).sum()) for c in only_below},
        "items_without_a_competition_key": int(comp.reindex(all_items).isna().sum()),
        "kangaroo_items": {"in_items_table": int(len(kang)),
                           "with_a_binary_response": int(len(all_items.intersection(kang))),
                           "in_eligible_pairs": int(len(el_items.intersection(kang))),
                           "competitions": sorted(set(comp.loc[kang]))},
        "items_table_rows": int(len(items)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    out, digests = {"benchmarks": {}}, {}
    for bench in BINARY:
        path = os.path.join(DATA_DIR, bench, "response.parquet")
        digests[os.path.relpath(path, ROOT)] = sha16(path)
        resp = pd.read_parquet(path, columns=["subject_id", "item_id", "benchmark_id", "response"])
        out["benchmarks"][bench] = pair_counts(resp)
        if bench == "matharena":
            ipath = os.path.join(DATA_DIR, bench, "items.parquet")
            digests[os.path.relpath(ipath, ROOT)] = sha16(ipath)
            items = pd.read_parquet(ipath, columns=["item_id", "item_features"])
            out["matharena"] = matharena_counts(resp, items)
    tot = out["benchmarks"].values()
    out["total"] = {k: int(sum(x[k] for x in tot)) for k in
                    ("eligible_pairs", "responses_in_eligible_pairs", "distinct_items_in_eligible_pairs")}
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    out["meta"] = {"when_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   "command": " ".join(["python", "experiments/data_counts.py"] + sys.argv[1:]),
                   "commit": head, "script_digest": sha16(os.path.abspath(__file__)),
                   "min_items": MIN_ITEMS, "data_digests": digests}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    m = out["matharena"]
    print(f"pairs {out['total']['eligible_pairs']}, responses {out['total']['responses_in_eligible_pairs']}")
    print(f"matharena: {m['items_with_a_binary_response']} items over {m['competitions_of_those_items']} "
          f"competitions; {m['items_in_eligible_pairs']} over {m['competitions_in_eligible_pairs']} in eligible "
          f"pairs; only below the floor {m['competitions_only_below_the_floor']}; "
          f"Kangaroo {m['kangaroo_items']}")


if __name__ == "__main__":
    main()
