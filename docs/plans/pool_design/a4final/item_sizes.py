"""Item-file sizes of every public benchmark at revision 54074453, from the
Hugging Face hub's file metadata, without downloading (amendment 4 final,
R12). Uses the local login; the token is never printed. Writes
item_sizes.json beside this file: per benchmark the size in bytes of
<b>/formatted_tables/items.parquet, and totals for L and E.

Run:  PYTHONDONTWRITEBYTECODE=1 python item_sizes.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import split_v2 as SV  # noqa: E402

REPO = "aims-foundations/measurement-db"
REV = "54074453da7b9911be0f48e4c560be547bc84d21"
LEGACY5 = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    man = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                                      "pool", "manifest.json")))
    public = sorted(b for b, r in man["benchmarks"].items() if r.get("status") == "ok")
    paths = [f"{b}/formatted_tables/items.parquet" for b in public]
    sizes = {}
    for i in range(0, len(paths), 50):
        for info in api.get_paths_info(REPO, paths[i:i + 50], repo_type="dataset", revision=REV):
            sizes[info.path.split("/")[0]] = int(info.size)
    L = set(SV.E) | set(LEGACY5)
    out = {"what": __doc__.split("\n")[0], "revision": REV, "public": len(public),
           "with_items": len(sizes), "total_GB": round(sum(sizes.values()) / 1e9, 3),
           "L_GB": round(sum(v for b, v in sizes.items() if b in L) / 1e9, 3),
           "E_GB": round(sum(v for b, v in sizes.items() if b in SV.E) / 1e9, 3),
           "largest_10": [[b, round(v / 1e9, 3)] for b, v in sorted(sizes.items(), key=lambda x: -x[1])[:10]],
           "largest_10_in_L": [[b, round(v / 1e9, 3)] for b, v in
                               sorted(((b, v) for b, v in sizes.items() if b in L), key=lambda x: -x[1])[:10]],
           "missing": sorted(set(public) - set(sizes)), "bytes": dict(sorted(sizes.items()))}
    with open(os.path.join(HERE, "item_sizes.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "bytes"}, indent=1))


if __name__ == "__main__":
    main()
