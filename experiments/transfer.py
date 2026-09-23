"""What transfers between benchmarks and what does not.

Item difficulty from text: leave-one-benchmark-out, around 0.08.
Subject standing from attributes: leave-one-benchmark-out, 0.38 to 0.64.

That asymmetry is the central fact of the whole problem. Run:
    python experiments/transfer.py
"""
import re

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from paiec.data import load_pairs
from paiec.items import item_table, numeric_features
from paiec.rasch import item_difficulty
from paiec.subjects import Spec, design_matrix, subject_frame


def item_transfer(df, target, label):
    print(f"\nitem difficulty from text, target = {label}")
    print(f"  {'held out':<20}{'n':>6}{'pearson':>10}")
    for b in sorted(df.benchmark_id.unique()):
        tr, te = df[df.benchmark_id != b], df[df.benchmark_id == b]
        if len(te) < 50:
            continue
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                            strip_accents="unicode")
        Xtr, Xte = v.fit_transform(tr.content.fillna("")), v.transform(te.content.fillna(""))
        sc = StandardScaler().fit(numeric_features(tr))
        Xtr = sparse.hstack([Xtr, sc.transform(numeric_features(tr))]).tocsr()
        Xte = sparse.hstack([Xte, sc.transform(numeric_features(te))]).tocsr()
        m = Ridge(alpha=3.0).fit(Xtr, tr[target].values, sample_weight=np.sqrt(tr.n.values))
        print(f"  {b:<20}{len(te):>6}{pearsonr(m.predict(Xte), te[target].values)[0]:>10.3f}")


def main():
    pairs = load_pairs()
    df = item_table(pairs)
    D = item_difficulty(pairs)
    df["b"] = [D[bid].get(k, np.nan) for bid, k in zip(df.benchmark_id, df.item_key)]
    g = df.groupby("benchmark_id").b
    df["zb"] = (df.b - g.transform("mean")) / g.transform("std")
    c = df.content.fillna("")
    df["degenerate"] = c.str.contains("See image", case=False) | (c.str.len() < 120)
    print("items whose text carries no task content:")
    print(df.groupby("benchmark_id").degenerate.agg(["size", "sum", "mean"]).round(3).to_string())

    item_transfer(df, "z", "naive share of subjects who solved it")
    item_transfer(df[~df.degenerate], "zb",
                  "Rasch difficulty, text-bearing items only")

    ma = df[df.benchmark_id == "matharena"].copy()
    ma["comp"] = ma.features.map(
        lambda s: (re.search(r"competition=([^;]+)", s or "") or [None, ""])[1])
    for col, lab in [("z", "naive"), ("zb", "Rasch")]:
        ss = ((ma.groupby("comp")[col].transform("mean") - ma[col].mean()) ** 2).sum()
        tot = ((ma[col] - ma[col].mean()) ** 2).sum()
        print(f"\nmatharena: competition alone explains {ss / tot:.3f} of {lab} difficulty "
              f"({ma.comp.nunique()} competitions inside one benchmark_id)")
    ma["resid"] = ma.z - ma.groupby("comp").z.transform("mean")
    for target, lab in [("z", "raw"), ("resid", "within competition")]:
        d = ma[~ma.degenerate].reset_index(drop=True)
        pred = np.zeros(len(d))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(d):
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
            X1 = v.fit_transform(d.content.iloc[tr].fillna(""))
            X2 = v.transform(d.content.iloc[te].fillna(""))
            m = Ridge(alpha=3.0).fit(X1, d[target].values[tr],
                                     sample_weight=np.sqrt(d.n.values[tr]))
            pred[te] = m.predict(X2)
        print(f"  within-benchmark 5-fold, {lab:<20} r={pearsonr(pred, d[target].values)[0]:.3f}")

    sf = subject_frame(pairs)
    spec = Spec.from_frame(sf)
    print("\nsubject standing from attributes, leave-one-benchmark-out")
    print(f"  {'held out':<20}{'n':>6}{'pearson':>10}{'spearman':>10}")
    for b in sorted(sf.benchmark_id.unique()):
        tr, te = sf[sf.benchmark_id != b], sf[sf.benchmark_id == b]
        if len(te) < 10:
            continue
        m = Ridge(alpha=2.0, fit_intercept=False).fit(design_matrix(tr, spec), tr.z.values)
        p = m.predict(design_matrix(te, spec))
        print(f"  {b:<20}{len(te):>6}{pearsonr(p, te.z.values)[0]:>10.3f}"
              f"{spearmanr(p, te.z.values)[0]:>10.3f}")


if __name__ == "__main__":
    main()
