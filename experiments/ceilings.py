"""How much there is to win at all: constant, pair mean, and per-item oracle.

Run: python experiments/ceilings.py
"""
import numpy as np
import pandas as pd

from paiec.ceilings import pair_floors
from paiec.data import load_pairs


def main():
    pairs = load_pairs()
    rows = []
    for p in pairs:
        f = pair_floors(p)
        f["bench"] = p.benchmark_id
        rows.append(f)
    df = pd.DataFrame(rows)
    agg = df.groupby("bench").agg(
        pairs=("mu", "size"), mean_acc=("mu", "mean"), const=("const_half", "mean"),
        pair_mean=("pair_mean", "mean"), item_loo=("item_loo", "mean"),
        unbiased_pq=("unbiased_pq", "mean"), frac_rep=("frac_rep", "mean")).round(4)
    print(agg.to_string())
    print("\nequally weighted over all pairs:")
    for c in ["const_half", "pair_mean", "item_loo", "unbiased_pq"]:
        print(f"  {c:<12} {df[c].mean():.4f}")
    print("\nitem_loo falls back to the pair mean where an item has no repeats, so it "
          "\nunderstates the gap on the benchmarks with few repeats. unbiased_pq is the "
          "\nhonest estimate, computed only on items that do have repeats.")


if __name__ == "__main__":
    main()
