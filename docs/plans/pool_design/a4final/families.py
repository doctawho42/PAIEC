"""Source families of the pool's benchmarks (amendment 4 final, R3).

A family is a connected component of the graph that joins two catalogue
entries when they share a normalised source_url key (host plus the first two
path segments) or a normalised paper_url. Reads only the catalogue rows
(data/pool/<b>/formatted_tables/benchmarks.parquet, columns source_url and
paper_url) of revision 54074453; no responses, no items, no predictor.

Writes families.json beside this file: every benchmark's family id (the
alphabetically first member), the keys that joined it, and the multi-member
families. Run:  PYTHONDONTWRITEBYTECODE=1 python families.py
"""
import json
import os
import re
from urllib.parse import urlsplit, parse_qs

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
POOL = os.path.join(ROOT, "data", "pool")


def src_key(url):
    """host (lower-cased, without www.) plus the first two path segments."""
    if not url or not str(url).strip():
        return None
    u = urlsplit(str(url).strip())
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    segs = [s for s in u.path.split("/") if s][:2]
    return "src:" + "/".join([host] + segs)


def paper_key(url):
    """arXiv ids without version; openreview forum ids; DOIs; else host plus path."""
    if not url or not str(url).strip():
        return None
    s = str(url).strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", s, re.I)
    if m:
        return "paper:arxiv:" + m.group(1)
    u = urlsplit(s)
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    if host == "openreview.net":
        q = parse_qs(u.query).get("id")
        if q:
            return "paper:openreview:" + q[0]
    if host in ("doi.org", "dx.doi.org"):
        return "paper:doi:" + u.path.strip("/").lower()
    return "paper:" + host + "/" + u.path.strip("/").lower()


def main():
    names = sorted(b for b in os.listdir(POOL)
                   if os.path.exists(os.path.join(POOL, b, "formatted_tables", "benchmarks.parquet")))
    keys = {}
    for b in names:
        row = pq.read_table(os.path.join(POOL, b, "formatted_tables", "benchmarks.parquet"),
                            columns=["source_url", "paper_url"]).to_pylist()[0]
        keys[b] = sorted(k for k in (src_key(row["source_url"]), paper_key(row["paper_url"])) if k)
    parent = {b: b for b in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for b in names:
        for k in keys[b]:
            if k in owner:
                ra, rb = find(owner[k]), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                owner[k] = b
    comp = {}
    for b in names:
        comp.setdefault(find(b), []).append(b)
    fam = {b: min(comp[find(b)]) for b in names}
    multi = {min(v): sorted(v) for v in comp.values() if len(v) > 1}
    shared = {}
    for f, members in multi.items():
        cnt = {}
        for b in members:
            for k in keys[b]:
                cnt[k] = cnt.get(k, 0) + 1
        shared[f] = sorted(k for k, c in cnt.items() if c > 1)
    out = {"what": __doc__.split("\n")[0], "rule": "connected components over src_key and paper_key",
           "benchmarks": len(names), "families": len(comp), "multi_member_families": len(multi),
           "family_of": fam, "multi": multi, "joining_keys": shared}
    with open(os.path.join(HERE, "families.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    for f in sorted(multi):
        print(f, len(multi[f]), multi[f], shared[f])
    print(len(names), "benchmarks;", len(comp), "families;", len(multi), "with two or more members")


if __name__ == "__main__":
    main()
