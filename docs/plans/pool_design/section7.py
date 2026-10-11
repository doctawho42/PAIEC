"""Descriptive counts behind data.md section 7 ('What this means for prior fitting').

Reads pairs.csv and benchmarks.csv (written by inventory.py) and nothing else
from the pool; fits nothing, scores nothing, reads no item text. Writes
section7.json beside this file.

For each benchmark set it counts what paiec.prior would see:
  * standings rows: one per pair on a benchmark with >= 2 pairs
    (paiec/prior.py:355-367), under per-benchmark caps K of the pairs, taken
    in proto_loader's stable-hash order (stable_hash('pool-pair', b, sid));
  * how concentrated the rows are (largest benchmark's share, the top two's,
    Kish's effective number of benchmarks (sum n)^2 / sum n^2);
  * canonical-name coverage of the rows, pair- and benchmark-weighted;
  * the identity cells (name, benchmark) and the cross-products _components
    averages (paiec/prior.py:514-539): sum over names of C(n_k, 2), n_k the
    number of benchmarks with >= 2 pairs on which name k has a row, and the
    share the most-linked names carry;
  * the size of the weight vectors _residuals keeps (one float64 n-vector per
    row of a linked name, paiec/prior.py:459-461), as rows x linked rows x 8
    bytes, per pass;
  * what a cap does to the names linked on >= 2 benchmarks, in proto_loader's
    subject-id hash order and in one name-keyed order shared by every
    benchmark, and which benchmarks the lost links were on.
Run with PYTHONDONTWRITEBYTECODE=1 from the repository root's interpreter.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from paiec.evaluator import stable_hash  # noqa: E402

FLAGGED = {"lower_is_better", "unordered"}


def load():
    bench = {r["benchmark"]: r for r in csv.DictReader(open(os.path.join(HERE, "benchmarks.csv")))}
    pairs = list(csv.DictReader(open(os.path.join(HERE, "pairs.csv"))))
    return bench, pairs


def sets(bench, pairs):
    elig = sorted({p["benchmark"] for p in pairs})

    def share(b):
        try:
            return float(bench[b]["share01"])
        except (KeyError, ValueError):
            return float("nan")
    typ = {b: bench[b]["response_type"] for b in elig}
    dirn = {b: bench[b]["direction"] for b in elig}
    out = {
        "all": elig,
        "binary": [b for b in elig if typ[b] == "binary"],
        "binary+mixed95": [b for b in elig if typ[b] == "binary"
                           or (typ[b] == "mixed" and share(b) >= 0.95)],
    }
    out["binary+mixed95, direction ordered"] = [b for b in out["binary+mixed95"]
                                                if dirn[b] not in FLAGGED]
    out["binary+mixed95, direction ordered, no sib200"] = [
        b for b in out["binary+mixed95, direction ordered"] if b != "sib200"]
    return out


def describe(pairs_by_b, K):
    rows = {}
    for b, ps in pairs_by_b.items():
        if len(ps) < 2:
            continue
        ps = sorted(ps, key=lambda p: stable_hash("pool-pair", b, p["subject_id"]))
        rows[b] = ps if K is None else ps[:K]
    n = {b: len(v) for b, v in rows.items()}
    N = sum(n.values())
    ranked = sorted(n.items(), key=lambda kv: -kv[1])
    named = [p for v in rows.values() for p in v if p["canon"]]
    cover_b = [sum(1 for p in v if p["canon"]) / len(v) for v in rows.values()]
    cells = defaultdict(set)
    for b, v in rows.items():
        for p in v:
            if p["canon"]:
                cells[p["canon"]].add(b)
    nk = {k: len(bs) for k, bs in cells.items()}
    xp = {k: comb(m, 2) for k, m in nk.items() if m >= 2}
    X = sum(xp.values())
    top = sorted(xp.items(), key=lambda kv: -kv[1])
    half, acc = 0, 0
    for _, c in top:
        acc += c
        half += 1
        if acc >= X / 2:
            break
    linked_rows = sum(1 for v in rows.values() for p in v if p["canon"] and nk.get(p["canon"], 0) >= 2)
    return {
        "K": K, "benchmarks_with_rows": len(rows), "rows": N,
        "largest": [ranked[0][0], ranked[0][1], round(ranked[0][1] / N, 4)],
        "top2_share": round(sum(c for _, c in ranked[:2]) / N, 4),
        "kish_effective_benchmarks": round(N ** 2 / sum(c * c for c in n.values()), 1),
        "rows_with_canon": len(named), "rows_with_canon_share": round(len(named) / N, 4),
        "canon_share_mean_over_benchmarks": round(sum(cover_b) / len(cover_b), 4),
        "benchmarks_with_any_canon_row": sum(1 for c in cover_b if c > 0),
        "names": len(nk), "names_ge2": sum(1 for m in nk.values() if m >= 2),
        "names_ge3": sum(1 for m in nk.values() if m >= 3),
        "names_ge10": sum(1 for m in nk.values() if m >= 10),
        "identity_cells": sum(nk.values()),
        "cross_products": X,
        "top_names_cross_products": [[k, nk[k], c, round(c / X, 4)] for k, c in top[:5]],
        "top10_cross_product_share": round(sum(c for _, c in top[:10]) / X, 4) if X else None,
        "names_carrying_half_the_cross_products": half,
        "linked_rows": linked_rows,
        "residual_weight_vectors_mb_per_pass": round(linked_rows * N * 8 / 2 ** 20, 1),
    }


def name_cells(by, K, order):
    """{name: benchmarks with a row of it} under cap K and pair order `order`."""
    c = defaultdict(set)
    for b, v in by.items():
        if len(v) < 2:
            continue
        v = sorted(v, key=lambda p: order(b, p))
        for p in (v if K is None else v[:K]):
            if p["canon"]:
                c[p["canon"]].add(b)
    return c


def by_subject(b, p):
    return stable_hash("pool-pair", b, p["subject_id"])          # proto_loader's order


def by_name(b, p):
    return stable_hash("pool-name", p["canon"] or p["subject_id"])  # one order for every benchmark


def links(by):
    """What a pair cap does to the names linked on >= 2 benchmarks."""
    c0 = name_cells(by, None, by_subject)
    linked0 = {k for k, bs in c0.items() if len(bs) >= 2}
    out = {"linked_uncapped": len(linked0), "caps": []}
    for K in (100, 50, 20):
        cs, cn = name_cells(by, K, by_subject), name_cells(by, K, by_name)
        lost = [k for k in linked0 if len(cs.get(k, ())) < 2]
        where = Counter(b for k in lost for b in c0[k] if b not in cs.get(k, set()))
        out["caps"].append({
            "K": K, "linked_subject_hash_order": sum(1 for bs in cs.values() if len(bs) >= 2),
            "linked_name_hash_order": sum(1 for bs in cn.values() if len(bs) >= 2),
            "names_name_hash_order": len(cn),
            "lost": len(lost), "lost_cells_by_benchmark": where.most_common(6)})
    big = sorted(((len(v), b, sum(1 for p in v if p["canon"]),
                   len({p["canon"] for p in v if p["canon"]})) for b, v in by.items() if len(v) > 100),
                 reverse=True)
    out["benchmarks_over_100_pairs"] = [{"benchmark": b, "pairs": n, "named_pairs": m, "names": k}
                                        for n, b, m, k in big]
    return out


def spread(by):
    """Within-benchmark sample variance (ddof 1) of the pairs' corrected logits
    (accuracy scale, pairs.csv logit_cc), pooled by n - 1 as inventory.py:445-447
    pools its sd (each pair weighing about one, as a row mean over standings
    does), and with each benchmark weighing one."""
    import statistics as st
    var = {b: st.variance([float(p["logit_cc"]) for p in v]) for b, v in by.items() if len(v) >= 2}
    n = {b: len(by[b]) for b in var}
    top = sorted(var, key=lambda b: -n[b])[:3]
    return {"pair_weighted": round(sum(var[b] * (n[b] - 1) for b in var) / sum(n[b] - 1 for b in var), 3),
            "benchmark_weighted": round(sum(var.values()) / len(var), 3),
            "benchmark_median": round(st.median(var.values()), 3),
            "largest": {b: [n[b], round(var[b], 3)] for b in top}}


def shared(by):
    """(name, benchmark) cells holding two or more pairs: one identity cell, several
    ridge rows (paiec/prior.py:627-656)."""
    g = Counter((b, p["canon"]) for b, v in by.items() for p in v if p["canon"])
    multi = {k: n for k, n in g.items() if n >= 2}
    return {"groups": len(multi), "pairs": sum(multi.values()),
            "benchmarks": len({b for b, _ in multi})}


def main():
    bench, pairs = load()
    out = {}
    for name, bs in sets(bench, pairs).items():
        keep = set(bs)
        by = defaultdict(list)
        for p in pairs:
            if p["benchmark"] in keep:
                by[p["benchmark"]].append(p)
        sizes = Counter(len(v) for v in by.values())
        out[name] = {
            "benchmarks": len(by), "pairs": sum(len(v) for v in by.values()),
            "single_pair_benchmarks": sizes.get(1, 0),
            "by_cap": [describe(by, K) for K in (None, 100, 50, 20)],
            "links": links(by),
            "within_benchmark_logit_variance": spread(by),
            "shared_name_within_benchmark": shared(by),
        }
    with open(os.path.join(HERE, "section7.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for name, v in out.items():
        print(name, v["benchmarks"], v["pairs"], v["single_pair_benchmarks"])
        for d in v["by_cap"]:
            print("  ", json.dumps(d))
        print("   links", json.dumps(v["links"]["caps"]))
        print("   spread", json.dumps(v["within_benchmark_logit_variance"]))
        print("   shared", json.dumps(v["shared_name_within_benchmark"]))


if __name__ == "__main__":
    main()
