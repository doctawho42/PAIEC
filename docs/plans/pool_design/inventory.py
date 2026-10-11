"""Descriptive inventory of the expanded measurement-db pool (data/pool, revision 54074453).

Read-only on the repository; writes only next to this file:
  inventory.json        everything data.md quotes
  benchmarks.csv        one row per public benchmark (types, eligibility, level)
  pairs.csv             one row per eligible subject-benchmark pair (no item text)

Eligibility as our code applies it (paiec/data.py load_pairs lines 29-57,
paiec/official.py eligible lines 66-80, docs/protocol.md "Flow"):
  * responses equal to 0.0 or 1.0 only (load_pairs line 37: resp.response.isin([0.0, 1.0]));
    nulls and every other value dropped;
  * benchmarks whose response_type is 'binary', or 'mixed' with binary
    responses kept as matharena is (matharena's catalogue type is 'mixed');
    every other response_type is dropped whole, as mmdocrag ('fraction') is
    (official.eligible line 73, D.FRACTIONAL);
  * a pair is (subject_id, benchmark_id) (load_pairs line 46) with at least 80
    distinct items left (official.eligible line 78, evaluator.MIN_ITEMS);
  * a pair's rate counts each item once, on its first binary response in file
    order (official._slots line 359 reveals by[k][0]; prior.benchmark_fit line
    259 fits rs[0]).
No predictor is run and no ALC or Brier is computed.
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from paiec.prior import canon_name, PROVIDERS  # noqa: E402
from paiec.subjects import ISO  # noqa: E402

POOL = os.path.join(ROOT, "data", "pool")
OLD = os.path.join(ROOT, "data")
OUT = os.path.dirname(os.path.abspath(__file__))
LEGACY5 = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
LEGACY6 = LEGACY5 + ["mmdocrag"]
SUBJECT_FIELDS = ["normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra"]
MIN_ITEMS = 80
KEEP_TYPES = ("binary", "mixed")
#: shipped LEVEL (submission/model.py line 68) and item variance T = sigma_d^2 + sigma_g^2
#: of submission/prior.json's hyper (hier.item_var line 1959, no group keys)
LEVEL = {"mu0": -2.5, "sigma_mu": 2.5}
GX, GW = np.polynomial.hermite_e.hermegauss(80)
GW = GW / GW.sum()


def clean(v):
    """paiec.data._clean: '' for missing, else str()."""
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def logit(p):
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def read_responses(path):
    cols = ["subject_id", "item_id", "benchmark_id", "trial", "test_condition", "interactors", "response"]
    t = pq.read_table(path, columns=cols,
                      read_dictionary=["subject_id", "item_id", "benchmark_id", "test_condition", "interactors"])
    return t.to_pandas()


def response_profile(df):
    r = df.response
    nn = r.notna()
    vals = r[nn]
    is01 = vals.isin([0.0, 1.0])
    distinct = np.unique(vals.values)
    tc = df.test_condition.astype(object)
    it = df.interactors.astype(object)
    return {
        "rows": int(len(df)), "null": int((~nn).sum()), "binary01": int(is01.sum()),
        "share01_of_nonnull": float(is01.mean()) if len(vals) else None,
        "n_distinct_values": int(len(distinct)),
        "values_sample": [float(x) for x in distinct[:8]],
        "benchmark_ids": sorted(map(str, df.benchmark_id.astype(object).unique()))[:3],
        "test_condition_nonnull": int(tc.notna().sum()),
        "test_condition_distinct": int(tc.dropna().nunique()),
        "interactors_nonnull": int(it.notna().sum()),
        "interactors_distinct": int(it.dropna().nunique()),
        "max_trial": int(df.trial.max()) if len(df) else 0,
    }


def pair_table(df, bench):
    """Eligible pairs of one benchmark from its responses (any type): binary
    responses only, >= MIN_ITEMS distinct items, first binary response per item
    in file order. Also the all-responses rate and the lowest-trial-first rate."""
    d = df[df.response.isin([0.0, 1.0])]
    if not len(d):
        return pd.DataFrame(), {}
    d = d[["subject_id", "item_id", "trial", "test_condition", "response"]].copy()
    subj_cats = d.subject_id.cat.categories
    d["subject_id"] = d.subject_id.cat.codes.astype(np.int64)
    d["item_id"] = d.item_id.cat.codes.astype(np.int64)
    d["_pos"] = np.arange(len(d))
    n_items = d.groupby("subject_id", sort=False).item_id.nunique()
    keep = n_items[n_items >= MIN_ITEMS].index
    d = d[d.subject_id.isin(keep)]
    if not len(d):
        return pd.DataFrame(), {}
    first = d.drop_duplicates(["subject_id", "item_id"], keep="first")
    lowt = d.sort_values(["trial", "_pos"], kind="mergesort").drop_duplicates(["subject_id", "item_id"], keep="first")
    g_first = first.groupby("subject_id").agg(n_items=("item_id", "size"), k_first=("response", "sum"))
    g_all = d.groupby("subject_id").agg(n_resp=("response", "size"), k_all=("response", "sum"))
    g_low = lowt.groupby("subject_id").agg(k_lowtrial=("response", "sum"))
    cells = d.groupby(["subject_id", "item_id"], sort=False).size()
    rep = (cells > 1).groupby(level=0).sum().rename("items_repeated")
    # cells whose first-in-file response is not at the lowest trial
    m = first.merge(lowt[["subject_id", "item_id", "_pos"]], on=["subject_id", "item_id"], suffixes=("", "_low"))
    disagree = int((m._pos != m._pos_low).sum())
    tcs = d.dropna(subset=["test_condition"]).groupby(["subject_id", "item_id"]).test_condition.nunique()
    multi_cond = int((tcs > 1).sum())
    t = g_first.join(g_all).join(g_low).join(rep).reset_index()
    t["subject_id"] = [str(subj_cats[c]) for c in t.subject_id]
    t.insert(0, "benchmark", bench)
    for c in ("k_first", "k_all", "k_lowtrial", "n_items", "n_resp", "items_repeated"):
        t[c] = t[c].astype(int)
    info = {"cells_first_not_lowest_trial": disagree, "cells_multi_condition": multi_cond,
            "cells": int(len(first))}
    return t, info


def subject_rows(bench):
    s = pd.read_parquet(os.path.join(POOL, bench, "formatted_tables", "subjects.parquet"))
    return s


def main():
    t0 = time.time()
    man = json.load(open(os.path.join(POOL, "manifest.json")))
    out = {"revision": man["revision"], "fetched_utc": man["fetched_utc"], "repo": man["repo"]}

    # --- integrity and catalogue --------------------------------------------------------
    bad = []
    for b, rec in man["benchmarks"].items():
        for fn, meta in rec["files"].items():
            p = os.path.join(POOL, b, "formatted_tables", fn)
            if not os.path.exists(p) or os.path.getsize(p) != meta["bytes"] or sha256(p) != meta["sha256"]:
                bad.append(f"{b}/{fn}")
    out["integrity_failures"] = bad
    cat = {}
    for b, rec in sorted(man["benchmarks"].items()):
        row = pd.read_parquet(os.path.join(POOL, b, "formatted_tables", "benchmarks.parquet")).iloc[0].to_dict()
        scale = json.loads(row["response_scale"]) if row.get("response_scale") else {}
        cat[b] = {"release": rec["release"], "status": rec["status"], "response_type": row["response_type"],
                  "granularity": row["granularity"], "direction": scale.get("direction"),
                  "scale_kind": scale.get("kind"),
                  "n_subjects": int(row["n_subjects"]), "n_items": int(row["n_items"]),
                  "n_responses": int(row["n_responses"]), "max_trial": int(row["max_trial"]),
                  "domain": list(row["domain"]) if row.get("domain") is not None else [],
                  "modality": list(row["modality"]) if row.get("modality") is not None else [],
                  "multi_single_turn": row.get("multi_single_turn"),
                  "benchmark_id": row["benchmark_id"]}
    out["catalogue_counts"] = {
        "entries": len(cat),
        "release": dict(Counter(v["release"] for v in cat.values())),
        "public_by_type": dict(Counter(v["response_type"] for v in cat.values() if v["release"] == "public")),
        "public_granularity": dict(Counter(v["granularity"] for v in cat.values() if v["release"] == "public")),
        "withheld": sorted(b for b, v in cat.items() if v["release"] != "public"),
    }
    print("catalogue", out["catalogue_counts"], f"{time.time() - t0:.0f}s", flush=True)

    # --- responses, per public benchmark ---------------------------------------------------
    prof, pairs, pinfo, literal = {}, [], {}, {}
    public = sorted(b for b, v in cat.items() if v["release"] == "public")
    for i, b in enumerate(public):
        df = read_responses(os.path.join(POOL, b, "formatted_tables", "responses.parquet"))
        prof[b] = response_profile(df)
        t, info = pair_table(df, b)
        if cat[b]["response_type"] in KEEP_TYPES:
            pinfo[b] = info
            if len(t):
                pairs.append(t)
        elif len(t):
            # what load_pairs + official.eligible would admit if handed this benchmark
            # (only names in data.FRACTIONAL are dropped whole there)
            literal[b] = {"type": cat[b]["response_type"], "pairs": int(len(t)),
                          "share01_of_nonnull": prof[b]["share01_of_nonnull"]}
        del df
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(public)} {time.time() - t0:.0f}s", flush=True)
    P = pd.concat(pairs, ignore_index=True)

    # --- subjects ---------------------------------------------------------------------------
    srows, subj_info = [], {}
    for b in sorted(P.benchmark.unique()):
        s = subject_rows(b)
        subj_info[b] = {"rows": int(len(s)), "unique_ids": int(s.subject_id.nunique()),
                        "columns": list(s.columns)}
        s = s.drop_duplicates("subject_id").set_index("subject_id")
        for sid in P.loc[P.benchmark == b, "subject_id"]:
            if sid in s.index:
                r = s.loc[sid]
                d = {f: clean(r.get(f)) for f in SUBJECT_FIELDS}
                d["display_name"] = clean(r.get("display_name"))
                d["in_table"] = True
            else:
                d = {f: "" for f in SUBJECT_FIELDS}
                d["display_name"] = ""
                d["in_table"] = False
            d["benchmark"], d["subject_id"] = b, sid
            srows.append(d)
    S = pd.DataFrame(srows)
    P = P.merge(S, on=["benchmark", "subject_id"], how="left")
    P["canon"] = [canon_name({f: r[f] for f in SUBJECT_FIELDS}) for r in P[SUBJECT_FIELDS].to_dict("records")]
    P["name_key"] = P.normalized_name.str.strip().str.lower()
    P["triple"] = [json.dumps([a.strip().lower(), b.strip().lower(), c.strip()]) if a.strip() else ""
                   for a, b, c in zip(P.normalized_name, P.provider, P.release_date)]
    P["full"] = [json.dumps([r[f] for f in SUBJECT_FIELDS]) for r in P[SUBJECT_FIELDS].to_dict("records")]
    P["rate_first"] = P.k_first / P.n_items
    P["logit_cc"] = np.log((P.k_first + 0.5) / (P.n_items - P.k_first + 0.5))
    P["logit_clip"] = [logit(min(max(p, 0.005), 0.995)) for p in P.rate_first]
    P["rate_all"] = P.k_all / P.n_resp
    P["logit_cc_all"] = np.log((P.rate_all * P.n_items + 0.5) / ((1 - P.rate_all) * P.n_items + 0.5))
    P["logit_cc_lowtrial"] = np.log((P.k_lowtrial + 0.5) / (P.n_items - P.k_lowtrial + 0.5))
    P["type"] = [cat[b]["response_type"] for b in P.benchmark]
    P["direction"] = [cat[b]["direction"] for b in P.benchmark]

    elig = sorted(P.benchmark.unique())
    out["eligible"] = {
        "benchmarks": len(elig),
        "by_type": dict(Counter(cat[b]["response_type"] for b in elig)),
        "pairs": int(len(P)),
        "pairs_by_type": dict(P.groupby("type").size()),
        "subject_ids": int(P.subject_id.nunique()),
        "subject_rows_benchmark_x_id": int(len(P)),
        "full_dicts": int(P.full.nunique()),
        "items_in_pairs": int(P.n_items.sum()),
        "pairs_not_in_subject_table": int((~P.in_table).sum()),
        "pairs_per_benchmark_quantiles": {q: float(np.quantile(P.groupby("benchmark").size(), q))
                                          for q in (0, .1, .25, .5, .75, .9, 1)},
        "benchmarks_with_pairs_ge": {k: int((P.groupby("benchmark").size() >= k).sum())
                                     for k in (1, 2, 3, 5, 10, 20, 50)},
        "pairs_n_items_quantiles": {q: float(np.quantile(P.n_items, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        "direction": dict(Counter(str(cat[b]["direction"]) for b in elig)),
        "direction_pairs": dict(Counter(map(str, P.direction))),
    }
    kept_types = [b for b in public if cat[b]["response_type"] in KEEP_TYPES]
    out["kept_type_benchmarks"] = len(kept_types)
    out["kept_type_no_eligible_pair"] = sorted(set(kept_types) - set(elig))
    out["literal_code_other_types"] = literal
    out["mixed_profiles"] = {b: {k: prof[b][k] for k in ("rows", "null", "binary01", "share01_of_nonnull",
                                                        "n_distinct_values", "values_sample")}
                             for b in public if cat[b]["response_type"] == "mixed"}
    out["mixed_eligible_share01"] = {b: prof[b]["share01_of_nonnull"] for b in elig if cat[b]["response_type"] == "mixed"}
    out["binary_profile_issues"] = {b: {k: prof[b][k] for k in ("rows", "null", "binary01", "n_distinct_values")}
                                    for b in public if cat[b]["response_type"] == "binary"
                                    and (prof[b]["null"] or prof[b]["binary01"] != prof[b]["rows"] - prof[b]["null"])}
    out["benchmark_id_mismatch"] = {b: prof[b]["benchmark_ids"] for b in public
                                    if prof[b]["benchmark_ids"] != [cat[b]["benchmark_id"]]}
    out["test_condition_eligible"] = {b: {"nonnull": prof[b]["test_condition_nonnull"],
                                          "distinct": prof[b]["test_condition_distinct"],
                                          "rows": prof[b]["rows"],
                                          "cells_multi_condition": pinfo[b]["cells_multi_condition"]}
                                      for b in elig if prof[b]["test_condition_nonnull"]}
    out["interactors_eligible"] = {b: {"nonnull": prof[b]["interactors_nonnull"],
                                       "distinct": prof[b]["interactors_distinct"], "rows": prof[b]["rows"]}
                                   for b in elig if prof[b]["interactors_nonnull"]}
    out["repeats_eligible"] = {
        "benchmarks_with_repeated_cells": int((P.groupby("benchmark").items_repeated.sum() > 0).sum()),
        "benchmarks_max_trial_gt1": sum(1 for b in elig if cat[b]["max_trial"] > 1),
        "cells_first_not_lowest_trial": {b: pinfo[b]["cells_first_not_lowest_trial"] for b in elig
                                         if pinfo[b]["cells_first_not_lowest_trial"]},
        "pairs_with_repeats": int((P.items_repeated > 0).sum()),
    }
    out["subject_tables"] = {
        "unique_id_violations": {b: v for b, v in subj_info.items() if v["rows"] != v["unique_ids"]},
        "column_sets": dict(Counter(",".join(v["columns"]) for v in subj_info.values())),
    }

    # --- subject fields: coverage -------------------------------------------------------------
    cov = {}
    fields = SUBJECT_FIELDS + ["display_name"]
    for f in fields:
        nonempty = P[f].str.strip() != ""
        per_b = nonempty.groupby(P.benchmark).mean()
        cov[f] = {"pairs_share": float(nonempty.mean()),
                  "benchmarks_any": int((per_b > 0).sum()), "benchmarks_all": int((per_b == 1).sum()),
                  "distinct_values": int(P.loc[nonempty, f].nunique())}
    out["coverage"] = cov
    rd = P.release_date.str.strip()
    out["release_date_forms"] = {
        "empty": int((rd == "").sum()),
        "iso_full": int(rd.str.fullmatch(r"\d{4}-\d{2}-\d{2}").sum()),
        "iso_month": int(rd.str.fullmatch(r"\d{4}-\d{2}").sum()),
        "iso_regex_any": int(sum(bool(ISO.fullmatch(x)) for x in rd if x)),
        "other_examples": sorted(set(x for x in rd if x and not ISO.fullmatch(x)))[:10],
        "year_quantiles": {q: float(np.quantile([int(x[:4]) for x in rd if re.match(r"\d{4}", x)], q))
                           for q in (0, .1, .25, .5, .75, .9, 1)} if (rd != "").any() else None,
    }
    ad = P.access_date.str.strip()
    out["access_date_forms"] = {"empty": int((ad == "").sum()),
                                "iso_regex_any": int(sum(bool(ISO.fullmatch(x)) for x in ad if x)),
                                "other_examples": sorted(set(x for x in ad if x and not ISO.fullmatch(x)))[:10]}
    prov = P.provider.str.strip()
    out["providers"] = {
        "top": Counter(x for x in prov if x).most_common(30),
        "distinct": int(prov[prov != ""].nunique()),
        "distinct_lower": int(prov[prov != ""].str.lower().nunique()),
        "lower_in_PROVIDERS_share_of_nonempty": float(np.mean([x.lower() in PROVIDERS for x in prov if x])),
        "lower_not_in_PROVIDERS_top": Counter(x.lower() for x in prov if x and x.lower() not in PROVIDERS).most_common(25),
    }
    out["reasoning_effort_values"] = Counter(x for x in P.reasoning_effort.str.strip() if x).most_common(20)
    out["harness_top"] = Counter(x for x in P.harness.str.strip() if x).most_common(15)
    sfe = P.subject_features_extra.str.strip()
    keys = Counter()
    for x in sfe:
        if x:
            for part in x.split(";"):
                if "=" in part:
                    keys[part.split("=", 1)[0].strip()] += 1
    out["subject_features_extra_keys"] = keys.most_common(25)
    out["canon"] = {
        "empty_pairs": int((P.canon == "").sum()),
        "empty_benchmarks_any": int((P.canon == "").groupby(P.benchmark).any().sum()),
        "empty_but_display_name": int(((P.canon == "") & (P.display_name.str.strip() != "")).sum()),
        "from_source_model_name": int(((P.normalized_name.str.strip() == "") & (P.canon != "")).sum()),
        "normalized_empty_pairs": int((P.normalized_name.str.strip() == "").sum()),
    }

    # --- linking across benchmarks ------------------------------------------------------------
    link = {}
    for key in ("canon", "name_key", "triple", "full", "subject_id"):
        k = P[key].astype(str)
        ok = k != ""
        nb = P[ok].groupby(k[ok]).benchmark.nunique()
        npairs = P[ok].groupby(k[ok]).size()
        link[key] = {
            "pairs_with_key": int(ok.sum()), "distinct": int(len(nb)),
            "in_ge_k_benchmarks": {kk: int((nb >= kk).sum()) for kk in (1, 2, 3, 5, 10, 20, 30, 50)},
            "pairs_on_keys_in_ge2_benchmarks": int(npairs[nb >= 2].sum()),
            "max_benchmarks": int(nb.max()) if len(nb) else 0,
            "benchmarks_quantiles": {q: float(np.quantile(nb, q)) for q in (.5, .75, .9, .95, .99)} if len(nb) else None,
        }
    out["link"] = link
    c = P[P.canon != ""]
    nb = c.groupby("canon").benchmark.nunique()
    out["canon_top"] = [(k, int(v)) for k, v in nb.sort_values(ascending=False).head(25).items()]
    # consistency of a canonical name's other fields across its pairs
    multi = c[c.canon.isin(nb[nb >= 2].index)]
    def n_distinct_nonempty(s):
        return s[s.str.strip() != ""].str.strip().str.lower().nunique()
    g = multi.groupby("canon")
    out["canon_consistency"] = {
        "linked_names": int(len(nb[nb >= 2])),
        "names_gt1_normalized_name": int((g.normalized_name.apply(n_distinct_nonempty) > 1).sum()),
        "names_gt1_provider": int((g.provider.apply(n_distinct_nonempty) > 1).sum()),
        "names_gt1_release_date": int((g.release_date.apply(n_distinct_nonempty) > 1).sum()),
        "examples_gt1_release_date": [k for k, v in g.release_date.apply(n_distinct_nonempty).items() if v > 1][:12],
        "examples_gt1_normalized_name": {k: sorted(set(multi.loc[multi.canon == k, "normalized_name"]))[:4]
                                         for k, v in g.normalized_name.apply(n_distinct_nonempty).items() if v > 1}
        if True else {},
    }
    nn = P[P.name_key != ""].groupby("name_key")
    out["normalized_name_consistency"] = {
        "names": int(P.loc[P.name_key != "", "name_key"].nunique()),
        "gt1_release_date": int((nn.release_date.apply(n_distinct_nonempty) > 1).sum()),
        "gt1_provider": int((nn.provider.apply(n_distinct_nonempty) > 1).sum()),
    }
    # several pairs of one benchmark under one key
    within = {}
    for key in ("canon", "full"):
        k = P[key].astype(str)
        ok = k != ""
        cnt = P[ok].groupby(["benchmark", k[ok]]).size()
        within[key] = {"benchmarks_with_dup": int((cnt > 1).groupby(level=0).any().sum()),
                       "groups_gt1": int((cnt > 1).sum()), "pairs_in_groups_gt1": int(cnt[cnt > 1].sum())}
    out["within_benchmark_shared_key"] = within
    sid_b = P.groupby("subject_id").benchmark.nunique()
    out["subject_id_cross_benchmark"] = {"ids_in_ge2": int((sid_b >= 2).sum()), "max": int(sid_b.max())}

    # --- levels -------------------------------------------------------------------------------
    def level_rows(Q):
        g = Q.groupby("benchmark")
        L = pd.DataFrame({"pairs": g.size(), "level_cc": g.logit_cc.mean(), "level_clip": g.logit_clip.mean(),
                          "level_cc_all": g.logit_cc_all.mean(), "level_cc_lowtrial": g.logit_cc_lowtrial.mean(),
                          "sd_pairs_cc": g.logit_cc.std(), "mean_rate": g.rate_first.mean(),
                          "items_median": g.n_items.median()})
        L["logit_mean_rate"] = [logit(min(max(p, 0.005), 0.995)) for p in L.mean_rate]
        # sampling variance of the level: between-pair var / n for >= 2 pairs,
        # the binomial delta-method variance of the single pair otherwise
        se2 = []
        for b, row in L.iterrows():
            q = Q[Q.benchmark == b]
            if len(q) >= 2:
                se2.append(float(q.logit_cc.var(ddof=1) / len(q)))
            else:
                k, n = float(q.k_first.iloc[0]), float(q.n_items.iloc[0])
                se2.append(1 / (k + 0.5) + 1 / (n - k + 0.5))
        L["se2"] = se2
        L["type"] = [cat[b]["response_type"] for b in L.index]
        L["direction"] = [str(cat[b]["direction"]) for b in L.index]
        return L

    L = level_rows(P)

    def summ(x, se2=None):
        x = np.asarray(x, float)
        d = {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
             "quantiles": {q: float(np.quantile(x, q)) for q in (0, .05, .1, .25, .5, .75, .9, .95, 1)},
             "share_below_0": float(np.mean(x < 0)), "share_below_m1": float(np.mean(x < -1)),
             "share_below_m2": float(np.mean(x < -2)), "share_above_1": float(np.mean(x > 1)),
             "share_above_2": float(np.mean(x > 2))}
        if se2 is not None and len(x) > 1:
            d["sd_net_of_sampling"] = float(math.sqrt(max(x.var(ddof=1) - float(np.mean(se2)), 0.0)))
        return d

    lv = {}
    lv["all"] = {m: summ(L[m], L.se2 if m == "level_cc" else None)
                 for m in ("level_cc", "level_clip", "level_cc_all", "logit_mean_rate")}
    lv["by_type"] = {t: summ(L.loc[L.type == t, "level_cc"], L.loc[L.type == t, "se2"]) for t in KEEP_TYPES}
    lv["by_pairs"] = {f"ge{k}": summ(L.loc[L.pairs >= k, "level_cc"], L.loc[L.pairs >= k, "se2"])
                      for k in (2, 3, 5, 10)}
    lv["single_pair"] = summ(L.loc[L.pairs == 1, "level_cc"], L.loc[L.pairs == 1, "se2"])
    lv["higher_is_better_or_unset"] = summ(L.loc[L.direction != "lower_is_better", "level_cc"],
                                           L.loc[L.direction != "lower_is_better", "se2"])
    lv["lower_is_better"] = summ(L.loc[L.direction == "lower_is_better", "level_cc"]) \
        if (L.direction == "lower_is_better").sum() > 1 else None
    lv["new_only"] = summ(L.loc[~L.index.isin(LEGACY6), "level_cc"], L.loc[~L.index.isin(LEGACY6), "se2"])
    lv["pair_weighted_mean_cc"] = float(P.logit_cc.mean())
    lv["pair_level_cc"] = summ(P.logit_cc)
    lv["within_benchmark_sd_pooled"] = float(math.sqrt(
        sum((L.pairs[b] - 1) * L.sd_pairs_cc[b] ** 2 for b in L.index if L.pairs[b] >= 2) /
        max(sum(L.pairs[b] - 1 for b in L.index if L.pairs[b] >= 2), 1)))
    lv["legacy5_new_tables"] = {b: {k: (float(L.loc[b, k]) if b in L.index else None)
                                    for k in ("pairs", "level_cc", "level_cc_all", "level_cc_lowtrial", "level_clip")}
                                for b in LEGACY5}
    leg = L.loc[[b for b in LEGACY5 if b in L.index]]
    lv["legacy5_summary_cc"] = summ(leg.level_cc)
    lv["legacy5_summary_cc_all"] = summ(leg.level_cc_all)
    lv["rank_of_legacy5_cc"] = {b: int((L.level_cc < L.loc[b, "level_cc"]).sum()) + 1 for b in leg.index}

    # LEVEL on the accuracy-logit scale: f(mu) = logit E[sigmoid(mu + sqrt(T) z)]
    hyper = json.load(open(os.path.join(ROOT, "submission", "prior.json")))["hyper"]
    T = hyper["sigma_d"] ** 2 + hyper["sigma_g"] ** 2

    def f(mu):
        p = float(np.sum(GW * sig(mu + math.sqrt(T) * GX)))
        return logit(p)

    def finv(y):
        lo, hi = -40.0, 40.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid) < y:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    from statistics import NormalDist
    nd = NormalDist(LEVEL["mu0"], LEVEL["sigma_mu"])
    qs = (.05, .1, .25, .5, .75, .9, .95)
    implied = {q: f(nd.inv_cdf(q)) for q in qs}
    # empirical CDF position of each LEVEL quantile, and LEVEL's CDF at the empirical quantiles
    item_scale = np.array([finv(y) for y in L.level_cc])
    lv["LEVEL"] = {
        "hyper_T": T, "c_probit_approx": (1 + math.pi * T / 8) ** -0.5,
        "f_mu0": f(LEVEL["mu0"]), "p_at_mu0_accuracy": float(sig(f(LEVEL["mu0"]))),
        "implied_accuracy_logit_quantiles": implied,
        "empirical_share_below_implied_quantile": {q: float(np.mean(L.level_cc < implied[q])) for q in qs},
        "pit_of_empirical_levels": {
            "mean": float(np.mean([nd.cdf(x) for x in item_scale])),
            "share_in_central_50": float(np.mean([(.25 <= nd.cdf(x) <= .75) for x in item_scale])),
            "share_in_central_90": float(np.mean([(.05 <= nd.cdf(x) <= .95) for x in item_scale])),
        },
        "empirical_on_item_scale": summ(item_scale),
        "legacy5_on_item_scale": {b: finv(float(L.loc[b, "level_cc"])) for b in leg.index},
    }
    out["levels"] = lv

    # --- old tables (bc8204d8) against the new ones, legacy six ------------------------------------
    cmp = {}
    for b in LEGACY6:
        o_s = pd.read_parquet(os.path.join(OLD, b, "subjects.parquet"))
        n_s = pd.read_parquet(os.path.join(POOL, b, "formatted_tables", "subjects.parquet"))
        o_s = o_s.sort_values("subject_id").reset_index(drop=True)
        n_s = n_s.sort_values("subject_id").reset_index(drop=True)
        same_subjects = list(o_s.columns) == list(n_s.columns) and o_s.astype(str).equals(n_s.astype(str))
        o_r = read_responses(os.path.join(OLD, b, "response.parquet"))
        n_r = read_responses(os.path.join(POOL, b, "formatted_tables", "responses.parquet"))
        key = ["subject_id", "trial", "test_condition", "interactors", "response"]
        def ms(df):
            x = df[key].astype(object).fillna("<NA>").astype(str)
            return x.sort_values(key).reset_index(drop=True)
        same_multiset = ms(o_r).equals(ms(n_r))
        o_items = set(o_r.item_id.astype(str))
        n_items = set(n_r.item_id.astype(str))
        same_order = bool((o_r.subject_id.astype(str).values == n_r.subject_id.astype(str).values).all()
                          and (o_r.response.fillna(-9).values == n_r.response.fillna(-9).values).all()) \
            if len(o_r) == len(n_r) else False
        ot, _ = pair_table(o_r, b)
        nt, _ = pair_table(n_r, b)
        m = ot.merge(nt, on="subject_id", how="outer", suffixes=("_old", "_new"), indicator=True)
        both = m[m._merge == "both"]
        cmp[b] = {
            "subjects_identical": bool(same_subjects), "subjects_old_new": [len(o_s), len(n_s)],
            "rows_old_new": [len(o_r), len(n_r)],
            "response_multiset_identical_ignoring_item_id": bool(same_multiset),
            "row_order_identical": same_order,
            "item_ids_old_new_shared": [len(o_items), len(n_items), len(o_items & n_items)],
            "eligible_pairs_old_new": [int(len(ot)), int(len(nt))],
            "pairs_only_old": int((m._merge == "left_only").sum()), "pairs_only_new": int((m._merge == "right_only").sum()),
            "n_items_differ": int((both.n_items_old != both.n_items_new).sum()) if len(both) else 0,
            "k_first_differ": int((both.k_first_old != both.k_first_new).sum()) if len(both) else 0,
            "k_first_abs_diff_sum": int((both.k_first_old - both.k_first_new).abs().sum()) if len(both) else 0,
            "k_all_differ": int((both.k_all_old != both.k_all_new).sum()) if len(both) else 0,
            "old_level_cc": float(np.mean(np.log((ot.k_first + .5) / (ot.n_items - ot.k_first + .5)))) if len(ot) else None,
            "new_level_cc": float(np.mean(np.log((nt.k_first + .5) / (nt.n_items - nt.k_first + .5)))) if len(nt) else None,
        }
        print("legacy", b, cmp[b], flush=True)
    out["old_vs_new_legacy6"] = cmp
    lj = json.load(open(os.path.join(ROOT, "experiments", "hier_design", "levels.json")))
    out["levels_json_mean_logit"] = {b: lj["benchmarks"][b]["mean_logit"] for b in LEGACY5}
    out["levels_json_between"] = {k: lj["between"][k] for k in ("mu0", "sd_levels")}

    # --- write ----------------------------------------------------------------------------------
    B = pd.DataFrame([{**{"benchmark": b}, **{k: v for k, v in cat[b].items() if k not in ("domain", "modality")},
                       "domain": ",".join(cat[b]["domain"]), "modality": ",".join(cat[b]["modality"]),
                       "rows": prof[b]["rows"] if b in prof else None,
                       "null": prof[b]["null"] if b in prof else None,
                       "share01": prof[b]["share01_of_nonnull"] if b in prof else None,
                       "test_condition_nonnull": prof[b]["test_condition_nonnull"] if b in prof else None,
                       "interactors_nonnull": prof[b]["interactors_nonnull"] if b in prof else None,
                       "eligible_pairs": int(L.pairs[b]) if b in L.index else 0,
                       "level_cc": float(L.level_cc[b]) if b in L.index else None,
                       "sd_pairs_cc": float(L.sd_pairs_cc[b]) if b in L.index and L.pairs[b] > 1 else None,
                       "mean_rate_first": float(L.mean_rate[b]) if b in L.index else None,
                       "linked_canon_pairs": int(P[(P.benchmark == b) & P.canon.isin(nb[nb >= 2].index)].shape[0])
                       if b in L.index else 0}
                      for b in sorted(cat)])
    B.to_csv(os.path.join(OUT, "benchmarks.csv"), index=False)
    cols = ["benchmark", "subject_id", "n_items", "k_first", "n_resp", "k_all", "k_lowtrial", "items_repeated",
            "rate_first", "logit_cc", "type", "direction", "canon"] + SUBJECT_FIELDS + ["display_name", "in_table"]
    P[cols].to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    # eligible domains
    dom = Counter(d for b in elig for d in cat[b]["domain"])
    out["eligible_domains"] = dom.most_common()
    out["eligible_modality"] = Counter(m for b in elig for m in cat[b]["modality"]).most_common()
    out["eligible_turns"] = dict(Counter(str(cat[b]["multi_single_turn"]) for b in elig))
    out["wall_s"] = round(time.time() - t0, 1)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    with open(os.path.join(OUT, "inventory.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=conv)
    print("done", out["wall_s"], "s")


if __name__ == "__main__":
    main()
