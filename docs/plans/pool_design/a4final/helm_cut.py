"""The descriptive HELM-hosted cut (amendment 4; the re-check's recommended item).

The family key crfm.stanford.edu/helm/<suite> makes the HELM suites separate
families, though they share one harness and one naming of subjects. The cut
of D by "HELM-hosted against the rest" is descriptive and changes nothing.

A benchmark is HELM-hosted when its own catalogue `source_url` has the host
crfm.stanford.edu (lower-cased, without www.) and a path whose first segment
is `helm`. Reads only the catalogue rows
(data/pool/<b>/formatted_tables/benchmarks.parquet, column source_url) of E's
benchmarks, and split_v2.json's family-rule halves and families.json; no
responses, no items, no predictor.

Writes helm_cut.json beside this file.
Run:  PYTHONDONTWRITEBYTECODE=1 python helm_cut.py
"""
import json
import os
from urllib.parse import urlsplit

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
POOL = os.path.join(ROOT, "data", "pool")


def helm_hosted(url):
    if not url or not str(url).strip():
        return False
    u = urlsplit(str(url).strip())
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    segs = [s for s in u.path.split("/") if s]
    return host == "crfm.stanford.edu" and bool(segs) and segs[0] == "helm"


def main():
    halves = json.load(open(os.path.join(HERE, "split_v2.json")))["splits"]["family"]
    family_of = json.load(open(os.path.join(HERE, "families.json")))["family_of"]
    helm = {}
    for b in sorted(halves):
        row = pq.read_table(os.path.join(POOL, b, "formatted_tables", "benchmarks.parquet"),
                            columns=["source_url"]).to_pylist()[0]
        if helm_hosted(row["source_url"]):
            helm[b] = [s for s in urlsplit(row["source_url"]).path.split("/") if s][1]
    out = {
        "what": __doc__.split("\n")[0],
        "rule": "own catalogue source_url: host crfm.stanford.edu, first path segment helm",
        "E": len(halves),
        "helm_hosted": {b: {"half": halves[b], "suite": helm[b], "family": family_of[b]}
                        for b in sorted(helm)},
        "count": {"S": sum(halves[b] == "S" for b in helm),
                  "C": sum(halves[b] == "C" for b in helm)},
        "families": sorted({family_of[b] for b in helm}),
    }
    with open(os.path.join(HERE, "helm_cut.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(out["count"]), len(out["families"]), "families")


if __name__ == "__main__":
    main()
