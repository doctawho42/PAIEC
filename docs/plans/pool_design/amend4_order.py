"""Pair-cap orders for amendment 4 (descriptive; scores nothing, reads no
feedback, no item text). Reads pairs.csv and benchmarks.csv of this directory.

For the subject part (binary + mixed at a 0/1 share >= 0.95, less the nine
direction-flagged benchmarks; benchmarks with >= 2 eligible pairs) and the level
part (the same with the flagged nine), at K = 100 pairs a benchmark, under three
orders:
  sid   stable_hash('pool-pair', b, subject_id)                  (proto_loader)
  name  stable_hash('pool-name', canon or subject_id), ties by sid  (section7.py)
  rr    round robin over names: (rank of the pair within its name on b by sid,
        stable_hash('pool-name', canon or subject_id))
it counts the rows kept, the named rows, the names on >= 2 benchmarks (links),
and on the benchmarks holding several pairs of one name, the distinct names kept.

    PYTHONDONTWRITEBYTECODE=1 python amend4_order.py   # writes amend4_order.json
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from paiec.evaluator import stable_hash  # noqa: E402

FLAGGED = {"crow", "emoji_attack", "harmbench", "jailbreakbench", "llmail_inject", "wmt_mqm",
           "alignment_faking", "safeagentbench", "tensortrust"}
K = 100


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    bench = {r["benchmark"]: r for r in csv.DictReader(open(os.path.join(HERE, "benchmarks.csv")))}
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "pairs.csv"))):
        by[r["benchmark"]].append({"sid": r["subject_id"], "canon": r["canon"]})
    level = sorted(b for b in by if bench[b]["response_type"] == "binary"
                   or (bench[b]["response_type"] == "mixed" and f(bench[b]["share01"]) >= 0.95))
    subject = [b for b in level if bench[b]["direction"] not in FLAGGED and b not in FLAGGED]

    def key_sid(b, p):
        return stable_hash("pool-pair", b, p["sid"])

    def key_name(b, p):
        return (stable_hash("pool-name", p["canon"] or p["sid"]), key_sid(b, p))

    def ordered_rr(b, ps):
        groups = defaultdict(list)
        for p in ps:
            groups[p["canon"] or ("\x00" + p["sid"])].append(p)
        out = []
        for g in groups.values():
            g.sort(key=lambda p: key_sid(b, p))
            for j, p in enumerate(g):
                out.append(((j, stable_hash("pool-name", p["canon"] or p["sid"])), p))
        return [p for _, p in sorted(out, key=lambda t: t[0])]

    res = {"K": K, "parts": {}}
    for part, names in (("subject", subject), ("level", level)):
        multi = [b for b in names if len(by[b]) >= 2]
        part_out = {"benchmarks": len(names), "multi_pair": len(multi),
                    "pairs_uncapped": sum(len(by[b]) for b in names)}
        for order in ("sid", "name", "rr"):
            kept = {}
            for b in multi:
                ps = by[b]
                if order == "sid":
                    ps = sorted(ps, key=lambda p: key_sid(b, p))
                elif order == "name":
                    ps = sorted(ps, key=lambda p: key_name(b, p))
                else:
                    ps = ordered_rr(b, ps)
                kept[b] = ps[:K]
            cells = defaultdict(set)
            for b, ps in kept.items():
                for p in ps:
                    if p["canon"]:
                        cells[p["canon"]].add(b)
            dup = {}
            for b, ps in kept.items():
                c = Counter(p["canon"] for p in by[b] if p["canon"])
                if len(by[b]) > K and any(v > 1 for v in c.values()):
                    dup[b] = {"pairs": len(by[b]), "names_uncapped": len(c),
                              "names_kept": len({p["canon"] for p in ps if p["canon"]}),
                              "named_rows_kept": sum(1 for p in ps if p["canon"])}
            part_out[order] = {
                "rows": sum(len(v) for v in kept.values()),
                "named_rows": sum(1 for v in kept.values() for p in v if p["canon"]),
                "names": len(cells),
                "linked_names": sum(1 for v in cells.values() if len(v) >= 2),
                "capped_benchmarks_with_repeated_names": dup,
            }
        cells_u = defaultdict(set)
        for b in multi:
            for p in by[b]:
                if p["canon"]:
                    cells_u[p["canon"]].add(b)
        part_out["uncapped_linked_names"] = sum(1 for v in cells_u.values() if len(v) >= 2)
        res["parts"][part] = part_out
    with open(os.path.join(HERE, "amend4_order.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    for part, v in res["parts"].items():
        print(part, {k: v[k] for k in ("benchmarks", "multi_pair", "pairs_uncapped",
                                       "uncapped_linked_names")})
        for order in ("sid", "name", "rr"):
            o = v[order]
            print("  ", order, {k: o[k] for k in ("rows", "named_rows", "names", "linked_names")})
            for b, d in sorted(o["capped_benchmarks_with_repeated_names"].items()):
                print("      ", b, d)


if __name__ == "__main__":
    main()
