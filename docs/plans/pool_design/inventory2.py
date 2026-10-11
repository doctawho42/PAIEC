"""Second pass over inventory.py's outputs (pairs.csv, benchmarks.csv) plus two
old-vs-new checks on the raw tables. Descriptive only: no predictor is run, no
ALC or Brier computed, no item text read. Writes inventory2.json beside this file.
"""
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
POOL = os.path.join(ROOT, "data", "pool")
SUBJECT_FIELDS = ["normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra"]
LEGACY6 = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench", "mmdocrag"]


def q(x, qs=(0, .05, .1, .25, .5, .75, .9, .95, 1)):
    x = np.asarray(x, float)
    return {k: float(np.quantile(x, k)) for k in qs}


def summ(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "quantiles": q(x)}


def main():
    P = pd.read_csv(os.path.join(HERE, "pairs.csv"), keep_default_na=False, low_memory=False)
    B = pd.read_csv(os.path.join(HERE, "benchmarks.csv"), keep_default_na=False, low_memory=False)
    for c in ("n_items", "k_first"):
        P[c] = P[c].astype(int)
    P["logit_cc"] = P.logit_cc.astype(float)
    out = {}
    E = B[B.eligible_pairs.astype(int) > 0].copy()
    E["eligible_pairs"] = E.eligible_pairs.astype(int)
    E["share01"] = E.share01.astype(float)

    # concentration of pairs
    cnt = P.groupby("benchmark").size().sort_values(ascending=False)
    out["pair_concentration"] = {"top": [(b, int(n), float(n / len(P))) for b, n in cnt.head(10).items()],
                                 "top2_share": float(cnt.head(2).sum() / len(P)),
                                 "top10_share": float(cnt.head(10).sum() / len(P))}

    # stricter eligible readings
    mixed_low = E[(E.response_type == "mixed") & (E.share01 < 0.95)]
    out["mixed_low_share01"] = {"threshold": 0.95, "benchmarks": int(len(mixed_low)),
                                "pairs": int(mixed_low.eligible_pairs.sum()),
                                "names": sorted(mixed_low.benchmark)}
    flagged_dir = E[E.direction.isin(["lower_is_better", "unordered"])]
    out["direction_flagged"] = {r.benchmark: {"direction": r.direction, "pairs": int(r.eligible_pairs),
                                              "level_cc": float(r.level_cc)} for r in flagged_dir.itertuples()}
    lev = E.set_index("benchmark").level_cc.astype(float)
    sets = {
        "binary_only": E.response_type == "binary",
        "binary_plus_mixed_ge095": (E.response_type == "binary") | ((E.response_type == "mixed") & (E.share01 >= 0.95)),
        "binary_higher_is_better_only": (E.response_type == "binary") & (E.direction == "higher_is_better"),
        "binary_mixed_ge095_not_flagged": ((E.response_type == "binary") | ((E.response_type == "mixed") & (E.share01 >= 0.95)))
        & ~E.direction.isin(["lower_is_better", "unordered"]),
    }
    out["level_sets"] = {}
    for name, m in sets.items():
        names = E.benchmark[m]
        out["level_sets"][name] = {**summ(lev[names]), "pairs": int(E.eligible_pairs[m].sum()),
                                   "share_below_LEVEL_median_m1148": float(np.mean(lev[names] < -1.148306))}

    # levels among recent subjects only (release_date known and >= 2025-01-01)
    rd = P.release_date.str.strip()
    P["year"] = [int(x[:4]) if re.match(r"\d{4}", x) else None for x in rd]
    yrs = Counter(int(y) for y in P.year.dropna())
    out["release_year_pairs"] = dict(sorted(yrs.items()))
    out["release_year_distinct_names"] = dict(sorted(Counter(
        int(y) for y in P.drop_duplicates("canon").loc[lambda d: d.canon != ""].year.dropna()).items()))
    recent = P[(rd >= "2025-01-01") & (rd != "")]
    g = recent.groupby("benchmark")
    Lr = pd.DataFrame({"pairs": g.size(), "level": g.logit_cc.mean()})
    out["recent_subjects"] = {
        "pairs": int(len(recent)), "benchmarks": int(len(Lr)),
        "levels_all": summ(Lr.level), "levels_ge3_pairs": summ(Lr.level[Lr.pairs >= 3]) if (Lr.pairs >= 3).sum() > 1 else None,
        "same_benchmarks_all_pairs_level": summ(lev[Lr.index]),
        "paired_diff_recent_minus_all": summ(Lr.level - lev[Lr.index]),
        "paired_diff_ge3": summ((Lr.level - lev[Lr.index])[Lr.pairs >= 3]) if (Lr.pairs >= 3).sum() > 1 else None,
    }
    dated = P[rd != ""]
    gd = dated.groupby("benchmark")
    Ld = pd.DataFrame({"pairs": gd.size(), "level": gd.logit_cc.mean()})
    out["dated_subjects"] = {"pairs": int(len(dated)), "benchmarks": int(len(Ld)), "levels": summ(Ld.level)}

    # subject identities: canonical names plus, without one, the full visible dict
    nocanon = P[P.canon == ""]
    out["distinct_subjects"] = {
        "canonical_names": int(P.loc[P.canon != "", "canon"].nunique()),
        "full_dicts_without_canon": int(nocanon[SUBJECT_FIELDS].astype(str).agg("\x1f".join, axis=1).nunique()),
        "subject_ids_without_canon": int(nocanon.subject_id.nunique()),
    }
    out["distinct_subjects"]["total_identity_keys"] = (out["distinct_subjects"]["canonical_names"]
                                                       + out["distinct_subjects"]["full_dicts_without_canon"])
    # visible model labels a no-canon subject still carries in subject_features_extra
    label_keys = ("source_model_label", "model_identifier", "source_model_identifier", "source_model_name",
                  "initialised_from", "model", "model_name", "base_model")
    def keys_of(s):
        return {p.split("=", 1)[0].strip() for p in s.split(";") if "=" in p}
    kc = Counter()
    any_label = 0
    for s in nocanon.subject_features_extra:
        ks = keys_of(s)
        hit = ks & set(label_keys)
        any_label += bool(hit)
        for k in hit:
            kc[k] += 1
    out["nocanon_extra_labels"] = {"pairs": int(len(nocanon)), "with_any_label_key": any_label,
                                   "by_key": kc.most_common()}
    out["nocanon_by_benchmark_top"] = [(b, int(n)) for b, n in
                                       nocanon.groupby("benchmark").size().sort_values(ascending=False).head(12).items()]
    # benchmark-weighted coverage of the eight fields (each eligible benchmark counts once)
    cov = {}
    for f in SUBJECT_FIELDS:
        share = (P[f].str.strip() != "").groupby(P.benchmark).mean()
        cov[f] = {"mean_over_benchmarks": float(share.mean()), "pairs_share": float((P[f].str.strip() != "").mean())}
    prov = P.provider.str.strip()
    cov["provider_informative"] = {
        "pairs_share": float(((prov != "") & ~prov.str.lower().isin(["unknown", "(agent scaffold)"])).mean()),
        "mean_over_benchmarks": float(((prov != "") & ~prov.str.lower().isin(["unknown", "(agent scaffold)"]))
                                      .groupby(P.benchmark).mean().mean())}
    cov["canon_nonempty"] = {"pairs_share": float((P.canon != "").mean()),
                             "mean_over_benchmarks": float((P.canon != "").groupby(P.benchmark).mean().mean())}
    out["coverage_benchmark_weighted"] = cov
    # harness: present almost everywhere now; is it a per-benchmark constant?
    h = P[P.harness.str.strip() != ""].groupby("benchmark").harness.nunique()
    out["harness_per_benchmark"] = {"benchmarks_with_harness": int(len(h)),
                                    "benchmarks_single_harness_value": int((h == 1).sum())}

    # within-benchmark pairs sharing a canonical name
    c = P[P.canon != ""]
    w = c.groupby(["benchmark", "canon"]).size()
    w = w[w > 1]
    out["within_benchmark_canon_groups"] = {
        "by_benchmark_top": [(b, int(n)) for b, n in w.groupby(level=0).sum().sort_values(ascending=False).head(10).items()],
        "largest_groups": [(b, k, int(n)) for (b, k), n in w.sort_values(ascending=False).head(8).items()]}
    # linking excluding the two training-checkpoint benchmarks
    nb = c.groupby("canon").benchmark.nunique()
    out["canon_k_benchmarks"] = {k: int((nb >= k).sum()) for k in (1, 2, 3, 4, 5, 10, 20)}
    out["canon_pairs_per_name"] = q(c.groupby("canon").size())
    # names on legacy-five benchmarks that now appear elsewhere
    leg = set(c.loc[c.benchmark.isin(LEGACY6[:5]), "canon"])
    nb_new = c[~c.benchmark.isin(LEGACY6)].groupby("canon").benchmark.nunique()
    out["legacy5_names"] = {"names": len(leg), "seen_on_new_benchmarks": int(sum(1 for k in leg if k in nb_new.index)),
                            "seen_on_ge3_new": int(sum(1 for k in leg if nb_new.get(k, 0) >= 3))}
    # duplicate canon names produced by dropping provider words
    out["short_canon_names"] = sorted(k for k in nb.index if len(k) <= 3 or re.fullmatch(r"[\d.]+[bm]?", k))

    # --- raw-table checks --------------------------------------------------------------------
    # matharena: rows reordered; does the first binary response per (subject, item) change?
    def first_cells(path):
        r = pd.read_parquet(path, columns=["subject_id", "item_id", "trial", "response"])
        r = r[r.response.isin([0.0, 1.0])]
        return r.drop_duplicates(["subject_id", "item_id"]).set_index(["subject_id", "item_id"])
    o = first_cells(os.path.join(ROOT, "data", "matharena", "response.parquet"))
    n = first_cells(os.path.join(POOL, "matharena", "formatted_tables", "responses.parquet"))
    j = o.join(n, lsuffix="_old", rsuffix="_new", how="outer")
    out["matharena_first_cells"] = {"cells": int(len(j)), "only_one_side": int(j.isna().any(axis=1).sum()),
                                    "trial_differs": int((j.trial_old != j.trial_new).sum()),
                                    "response_differs": int((j.response_old != j.response_new).sum())}
    # multi_swebench: which rows changed
    o = pd.read_parquet(os.path.join(ROOT, "data", "multi_swebench", "response.parquet"))
    n = pd.read_parquet(os.path.join(POOL, "multi_swebench", "formatted_tables", "responses.parquet"))
    same_cols = ["subject_id", "trial", "response"]
    diff = {c: int((o[c].astype(str).values != n[c].astype(str).values).sum()) for c in o.columns}
    out["multi_swebench_rowwise_diff"] = diff
    o_items = o.groupby("item_id").size()
    n_items = n.groupby("item_id").size()
    out["multi_swebench_items"] = {"old": int(len(o_items)), "new": int(len(n_items)),
                                   "shared_ids": int(len(set(o_items.index) & set(n_items.index)))}
    # map old -> new item ids through row position (row order is identical)
    m = pd.DataFrame({"old": o.item_id.values, "new": n.item_id.values})
    per_old = m.groupby("old").new.nunique()
    per_new = m.groupby("new").old.nunique()
    out["multi_swebench_item_map"] = {"old_ids_mapping_to_gt1_new": int((per_old > 1).sum()),
                                      "new_ids_from_gt1_old": int((per_new > 1).sum())}
    split = per_old[per_old > 1].index
    sub = o[o.item_id.isin(split)]
    out["multi_swebench_split_items"] = {"rows": int(len(sub)), "subjects": int(sub.subject_id.nunique())}
    cells_old = o.groupby(["subject_id", "item_id"]).size()
    cells_new = n.groupby(["subject_id", "item_id"]).size()
    out["multi_swebench_repeat_cells"] = {"old_gt1": int((cells_old > 1).sum()), "new_gt1": int((cells_new > 1).sum())}

    with open(os.path.join(HERE, "inventory2.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps(out, default=lambda x: x.item() if hasattr(x, "item") else str(x))[:20000])


if __name__ == "__main__":
    main()
