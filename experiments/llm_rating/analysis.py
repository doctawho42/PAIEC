"""Correlate blind LLM difficulty ratings with Rasch difficulty.

The ratings in ratings_main.py and ratings_control.py were produced by reading
the item texts with the true difficulty withheld. The main sample is 180 items
across four benchmarks, truncated to 450-700 characters; the control is 60 fresh
multi_swebench items with full text, to test whether truncation explained the
null on software-engineering tasks. It did not.

Run: python experiments/llm_rating/analysis.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from experiments.llm_rating.ratings_control import R as R_CTRL
from experiments.llm_rating.ratings_main import R as R_MAIN

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(max(n - 3, 1))
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def load(name, ratings):
    df = pd.DataFrame(json.load(open(os.path.join(HERE, "results", name))))
    missing = [u for u in df.uid if u not in ratings]
    if missing:
        raise SystemExit(f"{name}: unrated items {missing[:5]}")
    df["rating"] = df.uid.map(ratings)
    return df


def main():
    df = load("rate2_truth.json", R_MAIN)
    print("main sample, truncated text")
    print(f"  {'benchmark':<20}{'n':>4}{'pearson':>10}{'spearman':>10}   95% CI")
    for b, d in df.groupby("bench"):
        r, s = pearsonr(d.rating, d.zb)[0], spearmanr(d.rating, d.zb)[0]
        lo, hi = ci(r, len(d))
        print(f"  {b:<20}{len(d):>4}{r:>10.3f}{s:>10.3f}   [{lo:+.2f}, {hi:+.2f}]")
    g = df.groupby("bench")
    rr = (df.rating - g.rating.transform("mean")) / g.rating.transform("std")
    zz = (df.zb - g.zb.transform("mean")) / g.zb.transform("std")
    r = pearsonr(rr, zz)[0]
    lo, hi = ci(r, len(df) - 3)
    print(f"  {'pooled within-bench':<20}{len(df):>4}{r:>10.3f}"
          f"{spearmanr(rr, zz)[0]:>10.3f}   [{lo:+.2f}, {hi:+.2f}]")

    c = load("ctrl_truth.json", R_CTRL)
    r, s = pearsonr(c.rating, c.zb)[0], spearmanr(c.rating, c.zb)[0]
    lo, hi = ci(r, len(c))
    print(f"\ncontrol, multi_swebench with full issue text")
    print(f"  n={len(c)}  pearson={r:+.3f}  spearman={s:+.3f}  CI [{lo:+.2f}, {hi:+.2f}]")
    print("  The gap between Pearson and Spearman is the point: the correlation rests on a "
          "\n  few extreme items, not on a stable ordering.")


if __name__ == "__main__":
    main()
