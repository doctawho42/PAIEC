"""Item tables after the fetch: what is present for L, what was skipped, free disk.

Planning twin of the `fits` stage's manifest check (p2_pool_eval section 3.7).
Reads `benchmarks.csv` (this folder's inventory; L by the rule of
p2_pool_eval section 3.3), `data/pool/manifest.json`, the pre-items manifest
`data/p2_work/pool_manifest_no_items.json`, and the pool files of L, which it
hashes but never parses. Writes counts, sizes, digests and names only: no item
text, no per-pair row.

    python3 -I data/p2_work/pool_design/a4final/items_present.py <repo root>
"""
import csv
import datetime
import hashlib
import json
import os
import sys

LEGACY = ("matharena", "multi_swebench", "real_webagents", "researchcodebench",
          "swe_rebench")
TABLES_OLD = ("benchmarks.parquet", "subjects.parquet", "responses.parquet")


def in_L(r):
    if r["release"] != "public" or r["granularity"] != "item":
        return False
    if float(r["eligible_pairs"] or 0) < 1:
        return False
    if r["response_type"] == "binary":
        return True
    return r["response_type"] == "mixed" and float(r["share01"] or 0) >= 0.95


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(root):
    here = os.path.join(root, "data", "p2_work", "pool_design")
    rows = list(csv.DictReader(open(os.path.join(here, "benchmarks.csv"))))
    L = sorted(r["benchmark"] for r in rows if in_L(r))
    E = [b for b in L if b not in LEGACY]
    man_path = os.path.join(root, "data", "pool", "manifest.json")
    man = json.load(open(man_path))
    pre = json.load(open(os.path.join(root, "data", "p2_work",
                                      "pool_manifest_no_items.json")))
    B = man["benchmarks"]
    with_items = sorted(b for b, e in B.items()
                        if "items.parquet" in e.get("files", {}))
    without = sorted(set(B) - set(with_items))
    public = sorted(b for b, e in B.items() if e.get("release") == "public")

    changed = []
    for b in sorted(B):
        for f in TABLES_OLD:
            new = B[b].get("files", {}).get(f)
            old = pre["benchmarks"].get(b, {}).get("files", {}).get(f)
            if new != old:
                changed.append([b, f])

    bad, n_files, l_bytes = [], 0, 0
    for b in L:
        for f, e in sorted(B[b]["files"].items()):
            p = os.path.join(root, "data", "pool", b, "formatted_tables", f)
            if not os.path.exists(p):
                bad.append([b, f, "missing"])
                continue
            if os.path.getsize(p) != e["bytes"]:
                bad.append([b, f, "size"])
                continue
            if sha256(p) != e["sha256"]:
                bad.append([b, f, "sha256"])
                continue
            n_files += 1
            l_bytes += e["bytes"]

    def item_bytes(names):
        return sum(B[b]["files"]["items.parquet"]["bytes"] for b in names
                   if "items.parquet" in B[b].get("files", {}))

    sizes = json.load(open(os.path.join(here, "a4final", "item_sizes.json")))
    skipped = []
    for b in man.get("items_skipped", []):
        skipped.append({
            "benchmark": b,
            "response_type": B[b].get("response_type"),
            "in_L": b in L,
            "hub_item_bytes": sizes["bytes"].get(b),
            "item_file_on_disk": os.path.exists(os.path.join(
                root, "data", "pool", b, "formatted_tables", "items.parquet")),
        })
    pool_disk = 0
    for dp, _, fn in os.walk(os.path.join(root, "data", "pool")):
        pool_disk += sum(os.path.getsize(os.path.join(dp, f)) for f in fn)
    st = os.statvfs(root)
    free = st.f_bavail * st.f_frsize

    out = {
        "what": "Item tables after the fetch (p2_pool_eval section 3.7)",
        "measured_utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": sha256(man_path),
        "manifest_fetched_utc": man.get("fetched_utc"),
        "revision": man.get("revision"),
        "entries": len(B),
        "public": len(public),
        "with_items": len(with_items),
        "without_items": {b: B[b].get("release") for b in without},
        "items_skipped": skipped,
        "L": len(L),
        "E": len(E),
        "L_missing_items": [b for b in L if b not in with_items],
        "old_tables_changed_vs_pre_items_manifest": changed,
        "L_files_verified": n_files,
        "L_files_listed": sum(len(B[b]["files"]) for b in L),
        "L_files_bad": bad,
        "L_bytes_all_tables": l_bytes,
        "item_bytes": {"all_with_items": item_bytes(with_items),
                       "L": item_bytes(L), "E": item_bytes(E)},
        "data_pool_bytes_on_disk": pool_disk,
        "free_disk_bytes": free,
    }
    path = os.path.join(here, "a4final", "items_present.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
