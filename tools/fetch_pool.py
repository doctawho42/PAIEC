"""Download the expanded measurement-db pool (2026-10-07) for prior fitting.

measurement-db grew on 2026-10-06/07 from six benchmarks to 249 plus the six
legacy ones, each with validated tables under `<benchmark>/formatted_tables/`.
This tool fetches, at one pinned revision, the catalogue record of every
benchmark and, for those released as public, the subject and response tables
(and, with --items, the item table). Traces, assets and raw inputs are never
downloaded. Files land under data/pool/ (gitignored), in the repository's own
layout, and a manifest with every file's size and sha256 goes to
data/pool/manifest.json.

Run:  python tools/fetch_pool.py [--items] [--only B1,B2] [--jobs 4]
The Hugging Face token comes from the local login; it is never printed.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time

REPO = "aims-foundations/measurement-db"
#: the revision the expanded pool was first read at (2026-10-07)
POOL_REVISION = "54074453"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "pool")
CHECKPOINT = "tabular_checkpoint.json"
LEGACY = ["matharena", "mmdocrag", "multi_swebench", "real_webagents",
          "researchcodebench", "swe_rebench"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def full_revision(api, rev):
    """The full commit id of a short revision, so the manifest pins it exactly."""
    info = api.dataset_info(REPO, revision=rev)
    return info.sha


def get(path, rev, out):
    from huggingface_hub import hf_hub_download
    for attempt in range(4):
        try:
            return hf_hub_download(REPO, path, repo_type="dataset", revision=rev, local_dir=out)
        except Exception as e:  # network hiccups: retry, then report
            err = str(e).splitlines()[0][:160]
            if "404" in err or "EntryNotFound" in type(e).__name__:
                return None
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{path}: {err}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rev", default=POOL_REVISION)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--items", action="store_true", help="also fetch items.parquet")
    ap.add_argument("--only", default=None, help="comma-separated benchmarks")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--items-skip", default="",
                    help="comma-separated benchmarks whose item table is not fetched (e.g. not in the training set and too large)")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi
    import pandas as pd
    api = HfApi()
    rev = full_revision(api, a.rev)
    os.makedirs(a.out, exist_ok=True)
    ck = get(CHECKPOINT, rev, a.out)
    with open(ck) as f:
        names = [b["benchmark"] for b in json.load(f)["benchmarks"]] + LEGACY
    if a.only:
        names = [n for n in names if n in set(a.only.split(","))]
    t0 = time.time()
    manifest = {"repo": REPO, "revision": rev, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tables": ["benchmarks", "subjects", "responses"] + (["items"] if a.items else []),
                "items_skipped": sorted(x for x in a.items_skip.split(",") if x),
                "benchmarks": {}}

    def one(b):
        rec = {"files": {}}
        cat = get(f"{b}/formatted_tables/benchmarks.parquet", rev, a.out)
        if cat is None:
            rec["status"] = "no catalogue"
            return b, rec
        row = pd.read_parquet(cat).iloc[0].to_dict()
        rec["release"] = str(row.get("release"))
        for k in ("response_type", "response_scale", "granularity", "n_subjects", "n_items", "n_responses"):
            v = row.get(k)
            rec[k] = v.item() if hasattr(v, "item") else v
        rec["files"]["benchmarks.parquet"] = {"bytes": os.path.getsize(cat), "sha256": sha256(cat)}
        if rec["release"] != "public":
            rec["status"] = "not public"
            return b, rec
        skip = {x for x in a.items_skip.split(",") if x}
        for t in ["subjects", "responses"] + (["items"] if a.items and b not in skip else []):
            p = get(f"{b}/formatted_tables/{t}.parquet", rev, a.out)
            if p is not None:
                rec["files"][f"{t}.parquet"] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
        rec["status"] = "ok"
        return b, rec

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, (b, rec) in enumerate(ex.map(one, names), 1):
            manifest["benchmarks"][b] = rec
            if i % 20 == 0 or i == len(names):
                print(f"{i}/{len(names)} done ({time.time() - t0:.0f}s)", flush=True)
    manifest["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True, default=str)
    st = {}
    for rec in manifest["benchmarks"].values():
        st[rec.get("status")] = st.get(rec.get("status"), 0) + 1
    total = sum(v["bytes"] for rec in manifest["benchmarks"].values() for v in rec["files"].values())
    print(f"revision {rev[:8]}: {st}; {total / 1e9:.2f} GB in {manifest['wall_s']} s")


if __name__ == "__main__":
    sys.exit(main())
