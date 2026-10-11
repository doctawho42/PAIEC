"""Multiple-choice benchmarks of E by a catalogue flag (amendment 4 final, R12).

The flag is a regular expression on the catalogue's benchmark metadata
(name, one_line_description, description, benchmark_features), never on item
text. Descriptive: it scores nothing. Also lists the item files' sizes from
data/pool/manifest.json once the --items fetch has rewritten it, and, when the
item tables are present, the share of each E benchmark's items on which the
shipped floor (paiec.mcq.floor_of(mcq_text(item_content))) is positive: counts
only, no text is written.

Run:  PYTHONDONTWRITEBYTECODE=1 python mcq_catalogue.py   -> mcq_catalogue.json
"""
import json
import os
import re
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import split_v2 as SV  # noqa: E402

MCQ_RE = re.compile(r"\b(multiple|multi)[\s-]*choice\b|\bmcqa?s?\b", re.I)
POOL = os.path.join(ROOT, "data", "pool")


def flag(b):
    row = pq.read_table(os.path.join(POOL, b, "formatted_tables", "benchmarks.parquet"),
                        columns=["name", "one_line_description", "description", "benchmark_features"]).to_pylist()[0]
    text = " ".join(str(row.get(k) or "") for k in ("name", "one_line_description", "description",
                                                     "benchmark_features"))
    return bool(MCQ_RE.search(text))


def floor_share(b, cap=20000):
    from paiec.mcq import floor_of
    from paiec.hier import mcq_text
    p = os.path.join(POOL, b, "formatted_tables", "items.parquet")
    if not os.path.exists(p):
        return None
    pf = pq.ParquetFile(p)
    n = pos = 0
    for batch in pf.iter_batches(columns=["content"], batch_size=2000):
        for c in batch.column(0).to_pylist():
            n += 1
            try:
                pos += float(floor_of(mcq_text(c or ""))) > 0
            except Exception:
                pass
            if n >= cap:
                break
        if n >= cap:
            break
    return {"items_read": n, "floor_positive": pos, "share": round(pos / n, 4) if n else None,
            "items_total": pf.metadata.num_rows}


def main():
    half = SV.split("family")
    out = {"what": __doc__.split("\n")[0], "regex": MCQ_RE.pattern, "E": {}}
    for b in SV.E:
        out["E"][b] = {"half": half[b], "catalogue_mcq": flag(b), "level_cc": round(SV.LEV[b], 3),
                       "floor": floor_share(b)}
    man = json.load(open(os.path.join(POOL, "manifest.json")))
    sizes = {b: rec["files"]["items.parquet"]["bytes"] for b, rec in man["benchmarks"].items()
             if "items.parquet" in rec.get("files", {})}
    out["manifest_tables"] = man.get("tables")
    out["item_bytes"] = {"benchmarks_with_items_in_manifest": len(sizes),
                         "total_GB": round(sum(sizes.values()) / 1e9, 3),
                         "E_GB": round(sum(v for b, v in sizes.items() if b in SV.E) / 1e9, 3),
                         "largest_10": sorted(((round(v / 1e9, 3), b) for b, v in sizes.items()), reverse=True)[:10]}
    on_disk = {}
    for b in sorted(os.listdir(POOL)):
        p = os.path.join(POOL, b, "formatted_tables", "items.parquet")
        if os.path.exists(p):
            on_disk[b] = os.path.getsize(p)
    out["item_bytes_on_disk"] = {"files": len(on_disk), "total_GB": round(sum(on_disk.values()) / 1e9, 3),
                                 "E_GB": round(sum(v for b, v in on_disk.items() if b in SV.E) / 1e9, 3),
                                 "E_missing": sorted(b for b in SV.E if b not in on_disk),
                                 "largest_10": sorted(((round(v / 1e9, 3), b) for b, v in on_disk.items()),
                                                      reverse=True)[:10]}
    mc = sorted(b for b, v in out["E"].items() if v["catalogue_mcq"])
    out["catalogue_mcq"] = {"n": len(mc), "S": sorted(b for b in mc if half[b] == "S"),
                            "C": sorted(b for b in mc if half[b] == "C")}
    fl = {b: v["floor"]["share"] for b, v in out["E"].items() if v["floor"] and v["floor"]["share"] is not None}
    out["floor_summary"] = {"benchmarks_read": len(fl),
                            "share_ge_0.5": sorted(b for b, s in fl.items() if s >= 0.5),
                            "share_gt_0": sum(1 for s in fl.values() if s > 0)}
    with open(os.path.join(HERE, "mcq_catalogue.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "E"}, indent=1))


if __name__ == "__main__":
    main()
