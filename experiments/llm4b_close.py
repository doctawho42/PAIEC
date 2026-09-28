"""The 4B judge, closed out: Qwen3-4B-Instruct-2507's zero-shot item features
against honest difficulty and through the acceptance harness.

The features are the ones experiments/llm_features.py extracted into
data/features (paiec.llmfeat): one forward pass of the rating prompt per item,
no generation. Read per item (FEATURES), each with the sign declared before
this script looked at it:

  rating      the expected digit, 0..9, higher = the judge expects more strong
              systems to succeed (llmfeat 'rating'); declared -1 against
              difficulty
  digit_mode  the most likely digit (the "digit rating" a sampled answer would
              give); declared -1
  entropy     the entropy of the ten-digit distribution, nats; declared +1 (the
              judge's uncertainty tracks difficulty: Zotos, van Rijn & Nissim,
              arXiv:2412.11831)
  nll         the mean negative log-likelihood of the task's tokens under the
              4B, nats per token; no declared sign
  digit_mass  the share of the next-token distribution on the digits: a check
              that the judge answered, reported in the sign stage only

Coverage is what the extraction left before it stopped (MPS out of memory):
matharena complete, multi_swebench 18 of 22 rating shards (1,941 of 2,078
unique items; the 137 missing are the longest prompts, llm_features plans
shards by length), and nothing for real_webagents, researchcodebench and
swe_rebench. The extraction is not resumed here (see docs/findings.md, "The 4B
judge, closed out").

--stage signs. Per benchmark with features, the correlation of each feature
with the honest difficulty (experiments/harness.py oracle_maps: Rasch b fitted
without each of the five subject folds, averaged over the folds, higher =
harder; the target of itemcov_eval.py and attempt_probe.py):
  spearman, pearson                over the benchmark's rated items
  spearman_within, pearson_within  within item_features groups (competition,
                                   lang): ranks over the benchmark, demeaned
                                   within group; hier learns group levels from
                                   labels, so this is what a covariate can add
  partial_spearman, partial_pearson
                                   within group and net of log length
                                   (log(1 + characters of item_content), linear,
                                   Frisch-Waugh): the prompt's length ordered
                                   the attempt probe's items as well as any
                                   attempt feature
  partial2_spearman, partial2_pearson
                                   the same, net of the problem's position
                                   too (itemcov.position: log(1 + problem_idx),
                                   matharena only)
with 95% percentile intervals from a bootstrap over groups (primary; 27
competitions, 8 languages) and over items. Subsets: matharena's text-bearing
items (item_content that is not an image placeholder, itemcov.image_ref, and
holds at least MIN_TEXT characters: that removes Kangaroo's 336 image-only
problems, which share one instruction, and 202 items whose content is only a
system prompt or a date and an id),
the same split by contest year (the 2026 contests post-date the 4B), and the
attempt probe's 160 items (data/attempt_probe/targets.parquet, where present)
as a check against its reported -0.31. The sign check is the plan's rule,
leave one benchmark out: a transferred slope needs the declared sign and
agreement with the other units' mean on at least 4 of 5 units; with two units
rated it cannot pass, and the table says where the signs stand.

--stage harness. Each feature (not digit_mass) through experiments/harness.py
on its stored rows of the shipped hier (data/harness_rows; nothing of hier is
recomputed), by harness.eval_covariate (x standardised within benchmark; the B0
term reads raw x) and harness.score_covariate: nested transferred, per-pair,
hybrid and B0 lines, the harness's forced lines, and the per-pair slope forced
from B7 with s = 0.1 and 0.25. A covariate absent from two of the four parents
cannot be switched on leave-one-parent-out for a transferred slope (with a
rated parent held out, its inner folds fit the slope on the parents where x is
constant), so the transferred slope's reading is its forced line, fitted on the
other rated parent. Placebo: x permuted within each benchmark (N_PLACEBO
draws), every line, per-appearance differences averaged over draws.

--stage reference. What the harness gives at this coverage for a covariate of
known quality: the honest difficulty degraded to r (harness.degrade, per
subject fold) on exactly the rated items and 0 elsewhere, N_REF draws per r.

--stage verdict. The plan's kill rule: kill the 4B judge features if the
benchmark-equal test-like ALC difference (the mean of the four parents' means,
harness summary 'parent_mean') is not <= KILL_ALC under either slope variant
(the nested transferred and per-pair lines, and the transferred slope's forced
lines, its only reading here), and the partial r on text-bearing matharena
items is below KILL_R for every feature (Spearman and Pearson, oriented by the
declared sign; stage_verdict says how nll is oriented, and the absolute
reading is recorded beside it).

--stage show. Markdown tables of results/llm4b_close.json.

Run:  python experiments/llm4b_close.py --stage signs       # ~2 min
      python experiments/llm4b_close.py --stage harness     # ~6 min, one process, 0.8 GB, resumable
      python experiments/llm4b_close.py --stage reference   # ~2.5 min
      python experiments/llm4b_close.py --stage verdict
      python experiments/llm4b_close.py --stage show
No language model is loaded. Needs data/<benchmark>/, data/features/ and the
harness rows (python experiments/harness.py --stage collect).
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402
from experiments import itemcov_eval as ICE  # noqa: E402
from paiec import itemcov as IC  # noqa: E402
from paiec import llmfeat as F  # noqa: E402

OUT = os.path.join(ROOT, "results", "llm4b_close.json")
FEAT_DIR = os.path.join(ROOT, "data", "features")
BENCHES = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
GROUP_KEY = ICE.GROUP_KEY
#: name -> declared sign against difficulty (0: none declared)
FEATURES = {"rating": -1, "digit_mode": -1, "entropy": 1, "nll": 0}
DIAG = ("digit_mass",)
MIN_ITEMS = 20
N_AGREE = 4                 # the plan's sign rule: 4 of 5 units
MIN_TEXT = 70               # characters: shorter item_content is a system prompt (58) or a date and id (46)
YEAR_2026 = ("aime_2026", "aime_2026_I", "hmmt_feb_2026", "arxivmath_0126", "arxivmath_0226")
BOOT = 2000
PP_EXTRA = (("per-pair", 0.1, 7), ("per-pair", 0.25, 7))
FORCED_ALL = tuple(H.FORCED) + PP_EXTRA
N_PLACEBO = 3
PLACEBO_BOOTS = 200
REF_R = (0.3, 0.5)
N_REF = 4
KILL_ALC = -0.001
KILL_R = 0.2
STATS = ("spearman", "pearson", "spearman_within", "pearson_within", "partial_spearman", "partial_pearson",
         "partial2_spearman", "partial2_pearson")
NESTED = ("transferred nested", "per-pair nested", "hybrid nested", "b0 nested")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --- features ---------------------------------------------------------------------------

def load_features(bench, feat_dir=FEAT_DIR):
    """{feature: {str(item_id): x}} for one benchmark's rated items (every item_id
    a unique item stands for), and the rated/unique counts."""
    maps = {f: {} for f in tuple(FEATURES) + DIAG}
    try:
        index, _, probs, sc = F.load(feat_dir, bench)
    except FileNotFoundError:
        return maps, {"unique": 0, "rated": 0}
    has = index["has_llm"].to_numpy()
    vals = {"rating": sc["rating"].to_numpy(), "entropy": sc["entropy"].to_numpy(),
            "nll": sc["nll"].to_numpy(), "digit_mass": sc["digit_mass"].to_numpy(),
            "digit_mode": np.where(np.isfinite(probs).all(1), np.nanargmax(np.nan_to_num(probs, nan=-1), 1),
                                   np.nan).astype(float)}
    for i in np.flatnonzero(has):
        for f, v in vals.items():
            x = float(v[i])
            if math.isfinite(x):
                for iid in index["item_ids"].iloc[i]:
                    maps[f][str(iid)] = x
    info = {"unique": int(len(index)), "rated": int(has.sum())}
    if has.any() and (~has).any():
        tok = index["llm_tokens"].to_numpy()
        info["prompt_tokens_rated_max"] = int(tok[has].max())
        info["prompt_tokens_unrated_min"] = int(tok[~has].min())
    return maps, info


def features_digest(feat_dir=FEAT_DIR):
    """sha256 over the rating shards on disk (names and bytes) and the manifest's
    config hash: what the features were read from."""
    h = hashlib.sha256()
    with open(os.path.join(feat_dir, "manifest.json")) as f:
        h.update(json.load(f).get("config_hash", "").encode())
    for b in BENCHES:
        d = os.path.join(feat_dir, b, "llm")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".npz") and not fn.startswith("."):
                h.update(f"{b}/{fn}".encode())
                with open(os.path.join(d, fn), "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:16]


def honest_targets():
    """(mean honest difficulty per parent {item_id: b}, the five fold maps, info):
    itemcov_eval.difficulties()' target for the four parents."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ins, honest, info = H.oracle_maps()
    out = {}
    for par in H.PARENTS:
        keys = set().union(*(set(m) & set(ins[par]) for m in honest))
        out[par] = {k: float(np.mean([m[k] for m in honest if k in m])) for k in keys}
    return out, honest, info


def text_bearing(items):
    """{item_id: bool}: item_content is not an image placeholder (itemcov.image_ref)
    and holds at least MIN_TEXT characters."""
    return {k: bool(IC.image_ref(v) == 0 and len((v.get("item_content") or "").strip()) >= MIN_TEXT)
            for k, v in items.items()}


# --- correlations -------------------------------------------------------------------------

def ranks(x):
    """Average ranks (1..n)."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, r) / cnt)[inv]


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def _all_stats(x, d, C, inv, G):
    """STATS on one sample; inv the group of each item (0..G-1), C (n, k) the
    controls: the partial_* statistics are net of the group means and C's first
    column (log length), the partial2_* ones net of every column (NaN when C has
    one: position exists on matharena only). Spearman forms work on ranks, the
    controls' included."""
    n_g = np.maximum(np.bincount(inv, minlength=G), 1)

    def dm(v):
        return v - (np.bincount(inv, v, G) / n_g)[inv]

    def part(a, b, Cm):
        a, b = dm(a), dm(b)
        Z = np.column_stack([dm(c) for c in Cm.T])
        if np.any(Z):
            coef = np.linalg.lstsq(Z, np.column_stack([a, b]), rcond=None)[0]
            a, b = a - Z @ coef[:, 0], b - Z @ coef[:, 1]
        return pearson(a, b)

    rx, rd = ranks(x), ranks(d)
    rC = np.column_stack([ranks(c) for c in C.T])
    two = [part(rx, rd, rC), part(x, d, C)] if C.shape[1] > 1 else [float("nan")] * 2
    return np.array([pearson(rx, rd), pearson(x, d), pearson(dm(rx), dm(rd)), pearson(dm(x), dm(d)),
                     part(rx, rd, rC[:, :1]), part(x, d, C[:, :1])] + two)


def corr_block(x, d, g, C, boots=BOOT, seed=0):
    """STATS with 95% percentile intervals from a bootstrap over groups (each
    drawn group a group of its own) and one over items. C: the controls, log
    length first (a vector or (n, k))."""
    x, d = (np.asarray(v, float) for v in (x, d))
    L = np.asarray(C, float).reshape(len(x), -1)
    ug, inv = np.unique(np.asarray(g, object).astype(str), return_inverse=True)
    G = len(ug)
    est = _all_stats(x, d, L, inv, G)
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(inv == j) for j in range(G)]
    bg, bi = [], []
    for _ in range(boots):
        if G > 1:
            pick = rng.integers(0, G, G)
            ix = np.concatenate([members[j] for j in pick])
            inv_b = np.repeat(np.arange(G), [len(members[j]) for j in pick])
            bg.append(_all_stats(x[ix], d[ix], L[ix], inv_b, G))
        ix = rng.integers(0, len(x), len(x))
        bi.append(_all_stats(x[ix], d[ix], L[ix], inv[ix], G))
    bg, bi = (np.array(v, float) if v else None for v in (bg, bi))

    def ci(b, j):
        if b is None:
            return None
        v = b[:, j][np.isfinite(b[:, j])]
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) > 50 \
            else None

    out = {"n": int(len(x)), "groups": int(G)}
    for j, s in enumerate(STATS):
        if math.isfinite(est[j]):
            out[s] = {"est": round(float(est[j]), 4), "ci_group": ci(bg, j), "ci_item": ci(bi, j)}
    return out


def _vectors(fmap, target, keys):
    ks = [k for k in keys if k in fmap and k in target]
    return ks, np.array([fmap[k] for k in ks]), np.array([target[k] for k in ks])


def stage_signs(args):
    t0 = time.time()
    feats, cover = {}, {}
    for b in BENCHES:
        feats[b], cover[b] = load_features(b, args.features)
    rated = [b for b in BENCHES if cover[b]["rated"] >= MIN_ITEMS]
    target, _, tinfo = honest_targets()
    log(f"signs: features and honest targets in {time.time() - t0:.0f}s; rated benchmarks {rated}")
    items = {b: ICE.load_items(b) for b in rated}
    res = {"targets": {b: f"honest Rasch difficulty, mean over {H.HONEST_FOLDS} subject folds "
                          f"({len(target.get(b, {}))} items)" for b in target},
           "coverage": {}, "features": {}}
    for b in BENCHES:
        tgt = target.get(b, {})
        with_d = len(tgt)
        rated_d = sum(1 for k in feats[b]["rating"] if k in tgt)
        res["coverage"][b] = {**cover[b], "items_with_difficulty": with_d, "rated_items_with_difficulty": rated_d}
    # multi_swebench's missing shards are its longest prompts: does length relate to difficulty?
    for b in rated:
        tgt = target.get(b, {})
        ks = [k for k in items[b] if k in tgt]
        L = np.array([IC.log_length(items[b][k]) for k in ks])
        dd = np.array([tgt[k] for k in ks])
        r = np.array([k in feats[b]["rating"] for k in ks])
        res["coverage"][b]["log_length_spearman_all"] = round(pearson(ranks(L), ranks(dd)), 4)
        if (~r).any():
            res["coverage"][b]["log_length_spearman_rated"] = round(pearson(ranks(L[r]), ranks(dd[r])), 4)
            res["coverage"][b]["difficulty_mean_rated_unrated"] = [round(float(dd[r].mean()), 3),
                                                                   round(float(dd[~r].mean()), 3)]
    text = text_bearing(items["matharena"]) if "matharena" in items else {}
    comp = {k: IC.features(v).get("competition", "") for k, v in items.get("matharena", {}).items()}
    res["matharena_text_bearing"] = {"items": int(sum(text.values())), "of": len(text)}
    for fi, f in enumerate(tuple(FEATURES) + DIAG):
        per = {}
        for bi, b in enumerate(rated):
            tgt = target.get(b, {})
            ks, x, dd = _vectors(feats[b][f], tgt, list(items[b]))
            if len(ks) < MIN_ITEMS or np.std(x) == 0:
                per[b] = {"n": len(ks), "applies": False}
                continue
            g = [IC.features(items[b][k]).get(GROUP_KEY.get(b, ""), "") for k in ks]
            L = [IC.log_length(items[b][k]) for k in ks]
            pos = [IC.position(items[b][k]) for k in ks]
            if all(v is not None for v in pos):         # matharena's problem_idx
                L = np.column_stack([L, pos])
            per[b] = {"applies": True, **corr_block(x, dd, g, L, args.boots, seed=100 * fi + bi)}
            if b == "matharena":
                m = np.array([text[k] for k in ks])
                per["matharena text-bearing"] = corr_block(x[m], dd[m], np.array(g, object)[m],
                                                           np.array(L)[m], args.boots, seed=100 * fi + 7)
                if np.ndim(L) == 2:
                    # does the judge read the problem's position? (within competition, Spearman)
                    _, gi = np.unique(np.array(g, object)[m].astype(str), return_inverse=True)
                    per["matharena text-bearing"]["feature_vs_position_within"] = round(
                        _all_stats(x[m], np.asarray(L)[m, 1], np.asarray(L)[m, :1], gi, gi.max() + 1)[2], 4)
                    per["matharena text-bearing"]["difficulty_vs_position_within"] = round(
                        _all_stats(dd[m], np.asarray(L)[m, 1], np.asarray(L)[m, :1], gi, gi.max() + 1)[2], 4)
                for yr, sel in (("2025", lambda c: c not in YEAR_2026), ("2026", lambda c: c in YEAR_2026)):
                    my = m & np.array([sel(comp[k]) for k in ks])
                    per[f"matharena text-bearing {yr}"] = corr_block(
                        x[my], dd[my], np.array(g, object)[my], np.array(L)[my], args.boots,
                        seed=100 * fi + (8 if yr == "2025" else 9))
        entry = {"declared_sign": FEATURES.get(f), "units": per}
        if f in FEATURES:
            entry["sign_rule"] = sign_rule(per, FEATURES[f])
            entry["sign_rule_within"] = sign_rule(per, FEATURES[f], "spearman_within")
        res["features"][f] = entry
        msg = "  ".join(f"{u[:18]} {v['spearman']['est']:+.3f}/{v['spearman_within']['est']:+.3f}/"
                        f"{v['partial_spearman']['est']:+.3f}" for u, v in per.items() if v.get("spearman"))
        log(f"{f:10s} (spearman / within / partial) {msg}")
    res["probe_items_check"] = probe_check(feats.get("matharena", {}).get("rating", {}), args.boots)
    res["wall_s"] = round(time.time() - t0, 1)
    state = H.load_json(args.out) or {}
    state["signs"] = res
    state["meta"] = meta(args, tinfo)
    H.save_json(args.out, state)
    log(f"signs: {res['wall_s']}s -> {args.out}")


def sign_rule(per, declared, field="spearman"):
    """The plan's rule over the benchmarks (units) where the feature applies: the
    declared sign on at least N_AGREE units and agreement with the other units'
    mean (leave one unit out) on at least N_AGREE."""
    app = {u: v[field]["est"] for u, v in per.items() if u in BENCHES and v.get("applies")
           and math.isfinite(v[field]["est"])}
    lobo = {}
    for u, r in app.items():
        others = [app[w] for w in app if w != u]
        lobo[u] = bool(others) and bool(np.sign(r) == np.sign(np.mean(others)))
    dec = {u: (None if not declared else bool(np.sign(r) == declared)) for u, r in app.items()}
    n_dec = sum(1 for v in dec.values() if v)
    return {"units": sorted(app), "signs": {u: int(np.sign(r)) for u, r in app.items()},
            "declared_ok": dec, "lobo_agree": lobo, "declared_sign_units": n_dec,
            "lobo_agree_units": sum(lobo.values()),
            "transferred_allowed": bool(declared and n_dec >= N_AGREE and sum(lobo.values()) >= N_AGREE)}


def probe_check(rating_map, boots):
    """The attempt probe's 160 items: within-competition Spearman of -rating with
    b_honest as attempt_probe.within_rho computes it (group-percentile ranks),
    against its stored -0.31, and this script's statistic on the same items."""
    import pandas as pd
    path = os.path.join(ROOT, "data", "attempt_probe", "targets.parquet")
    if not os.path.exists(path):
        return None
    t = pd.read_parquet(path)
    t = t[t.probe]
    t = t.assign(r=[rating_map.get(str(k), np.nan) for k in t.index]).dropna(subset=["r", "b_honest"])
    xr = t.groupby("comp")["r"].transform(lambda s: s.rank(pct=True) - s.rank(pct=True).mean())
    yr = t.groupby("comp")["b_honest"].transform(lambda s: s.rank(pct=True) - s.rank(pct=True).mean())
    stored = None
    try:
        with open(os.path.join(ROOT, "results", "attempt_probe.json")) as f:
            ref = json.load(f)["references_vs_honest"]
        stored = next(v["rho"] for k, v in ref.items() if k.startswith("rating_4b"))
    except Exception:
        pass
    return {"n": int(len(t)), "within_rho_percentile_ranks_neg_rating": round(pearson(-xr.to_numpy(),
                                                                                     yr.to_numpy()), 4),
            "stored_attempt_probe": stored}


# --- harness ------------------------------------------------------------------------------

def compact(ln):
    """The numbers read off one harness line."""
    rg = ln["regimes"]
    tl = rg["tl"]
    pp = tl.get("per_parent") or {}
    covered = [q for q in ("matharena", "multi_swebench") if q in pp]
    se_eq = (math.sqrt(sum(pp[q]["cluster_se"] ** 2 for q in H.PARENTS if q in pp)) / 4
             if all(q in pp and "cluster_se" in pp[q] for q in H.PARENTS) else None)
    out = {"tl": tl["est"], "tl_cluster_se": tl["cluster_se"], "tl_sel_se": tl.get("sel_cluster_se"),
           "tl_benchmark_equal": tl.get("parent_mean"), "tl_benchmark_equal_se": se_eq,
           "tl_covered_mean": float(np.mean([pp[q]["est"] for q in covered])) if covered else None,
           "tl_per_parent": {q: [v["est"], v.get("cluster_se")] for q, v in pp.items()},
           "tl_worst_parent": tl.get("worst_parent"), "tl_by_budget": tl.get("by_budget"),
           "mix": rg["mix"]["est"] if rg.get("mix") else None,
           "mix_benchmark_equal": rg["mix"].get("parent_mean") if rg.get("mix") else None,
           "r1b": rg["r1b"]["est"] if rg.get("r1b") else None, "r1p": rg["r1p"]["est"] if rg.get("r1p") else None,
           "folds_on": ln.get("folds_on"), "folds_chosen": ln.get("folds_chosen"),
           "gate_pass": ln["gate"]["pass"], "gate_pass_if_on": ln["gate"]["pass_if_on"]}
    ch = ln.get("choices")
    if ch:
        out["choices"] = {q: (None if c.get("choice") is None else H.cname(tuple(c["choice"])))
                          for q, c in ch.items()}
    return out


def feature_maps(feat_dir):
    maps = {f: {} for f in FEATURES}
    for b in BENCHES:
        m, _ = load_features(b, feat_dir)
        for f in FEATURES:
            maps[f].update(m[f])
    return maps


def stage_harness(args):
    t0 = time.time()
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    items_bench = H.benchmark_items(rows, keys)
    _, honest, tinfo = honest_targets()
    maps = feature_maps(args.features)
    fdig = features_digest(args.features)
    prov = H.provenance()
    log(f"harness: rows, targets and features in {time.time() - t0:.0f}s")
    state = H.load_json(args.out) or {}
    res = state.get("harness", {})
    same = res.get("_meta", {}).get("features_digest") == fdig and \
        res.get("_meta", {}).get("lib_digest") == prov["lib_digest"]
    if not same:
        res = {}
    res["_meta"] = {"features_digest": fdig, "lib_digest": prov["lib_digest"],
                    "rows_lib_digest": H.rows_digest(rows), "forced": [H.cname(c) for c in FORCED_ALL],
                    "n_placebo": args.placebo, "placebo_boots": PLACEBO_BOOTS, "boots": H.BOOTS,
                    "input": "x standardised within benchmark (harness.eval_covariate), B0 term on raw x"}
    for f in args.feats:
        if f in res and not args.redo:
            log(f"{f}: kept")
            continue
        t1 = time.time()
        raw = maps[f]
        cov, info = H.eval_covariate(raw, items_bench, keys, f, standardise=True)
        cov.keys = keys
        covered = {reg: float(np.mean(np.isin(np.array(keys, object)[R.ev_k], list(raw)))) for reg, R in rows.items()}
        lines, _ = H.score_covariate(rows, cov, forced=FORCED_ALL)
        entry = {"x": info, "coverage_eval_items": covered, **ICE.realised_r(rows, cov, honest),
                 "lines": {name: compact(ln) for name, ln in lines.items()}}
        plac = {}
        for s in range(args.placebo):
            pm = ICE.permuted(raw, items_bench, s)
            cp, _ = H.eval_covariate(pm, items_bench, keys, f + " placebo", standardise=True)
            pl, _ = H.score_covariate(rows, cp, forced=FORCED_ALL, boots=PLACEBO_BOOTS, per=True)
            for name, ln in pl.items():
                plac.setdefault(name, []).append(compact(ln))
        entry["placebo"] = {name: {k: float(np.mean([d[k] for d in ds])) if ds[0][k] is not None else None
                                   for k in ("tl", "tl_benchmark_equal", "tl_covered_mean", "mix", "r1b", "r1p")}
                            | {"tl_draws": [d["tl"] for d in ds],
                               "tl_benchmark_equal_draws": [d["tl_benchmark_equal"] for d in ds],
                               "folds_on": [d["folds_on"] for d in ds]}
                            for name, ds in plac.items()}
        entry["wall_s"] = round(time.time() - t1, 1)
        res[f] = entry
        state["harness"] = res
        state.setdefault("meta", meta(args, tinfo))
        H.save_json(args.out, state)
        a = entry["lines"]
        log(f"{f}: {entry['wall_s']}s  r_within_pair {entry.get('r_within_pair_tl')}  " + "  ".join(
            f"{n.split()[0]} {a[n]['tl']:+.5f} (eq {a[n]['tl_benchmark_equal']:+.5f}, on {a[n]['folds_on']})"
            for n in NESTED) + "  forced: " + "  ".join(
            f"{n} {a[n]['tl_covered_mean']:+.5f}" for n in a if n.endswith("(forced)")))
    state["harness"] = res
    H.save_json(args.out, state)
    log(f"harness: {time.time() - t0:.0f}s -> {args.out}")


def stage_reference(args):
    """The honest difficulty degraded to r on exactly the rated items (0 elsewhere),
    N_REF draws per r, lines averaged over draws (harness.average_lines)."""
    t0 = time.time()
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    target, honest, _ = honest_targets()
    rated = set(feature_maps(args.features)["rating"])
    par_of_key = {k: par for par, d in target.items() for k in d}
    Z, has, _ = H.base_matrix(honest, keys, par_of_key)
    on = np.array([k in rated and k in par_of_key for k in keys])
    out = {"rated_keys_in_rows": int(on.sum()), "r": {}}
    for r in REF_R:
        runs = []
        for rep in range(N_REF):
            eps = np.random.default_rng([rep, 7919]).standard_normal(len(keys))
            X = np.where(on[None, :], H.degrade(np.where(has, Z, 0.0), r, eps[None, :]), 0.0)
            cov = H.Covariate(X, H.fold_of, f"ref r={r}", np.where(on[None, :], X, np.nan))
            runs.append(H.score_covariate(rows, cov, boots=PLACEBO_BOOTS, per=False))
        avg = H.average_lines(rows, runs, boots=PLACEBO_BOOTS)
        out["r"][f"r={r:g}"] = {
            name: {"tl": ln["regimes"]["tl"]["est"], "tl_benchmark_equal": ln["regimes"]["tl"]["parent_mean"],
                   "tl_draws": ln["replicates"]["tl"], "mix": ln["regimes"]["mix"]["est"],
                   "folds_on": ln.get("folds_on")}
            for name, ln in avg.items()}
        log(f"reference r={r:g}: " + "  ".join(f"{n} {v['tl']:+.5f} (eq {v['tl_benchmark_equal']:+.5f})"
                                               for n, v in out["r"][f"r={r:g}"].items()))
    out["wall_s"] = round(time.time() - t0, 1)
    state = H.load_json(args.out) or {}
    state["reference"] = out
    H.save_json(args.out, state)


# --- verdict ------------------------------------------------------------------------------

def stage_verdict(args):
    """The kill rule on the stored stages. ALC prong: no benchmark-equal
    test-like difference <= KILL_ALC on the nested transferred and per-pair
    lines, nor on the transferred slope's forced lines (with two rated parents
    its nested line cannot act on them, so the forced lines are its only
    reading); the forced per-pair lines, which no selection would switch on
    blind, are reported beside it. r prong: the partial r (within competition,
    net of log length) on text-bearing matharena items, oriented by the declared
    sign (nll, with none declared: by multi_swebench's sign, the other rated
    unit), below KILL_R for every feature, Spearman and Pearson; its absolute
    value is reported too."""
    state = H.load_json(args.out)
    if not state or "signs" not in state or "harness" not in state:
        raise SystemExit("run --stage signs and --stage harness first")
    hz = {f: v for f, v in state["harness"].items() if not f.startswith("_")}
    alc = {}
    for f, v in hz.items():
        ln = v["lines"]
        alc[f] = {name: ln[name]["tl_benchmark_equal"] for name in ln
                  if name in ("transferred nested", "per-pair nested", "hybrid nested", "b0 nested")
                  or name.endswith("(forced)")}
    rule_lines = ("transferred nested", "per-pair nested", "transferred from B1 (forced)",
                  "transferred from B7 (forced)")
    prong_a = all(v[k] is None or v[k] > KILL_ALC for v in alc.values() for k in rule_lines)
    prong_a_forced = all(x is None or x > KILL_ALC for v in alc.values() for x in v.values())
    feats = state["signs"]["features"]
    pr = {}
    for f in FEATURES:
        u = feats[f]["units"].get("matharena text-bearing")
        if not u:
            continue
        o = FEATURES[f] or int(np.sign(feats[f]["units"]["multi_swebench"]["partial_spearman"]["est"]))
        pr[f] = {"orientation": o, **{k: u[k]["est"] for k in ("partial_spearman", "partial_pearson",
                                                                "partial2_spearman", "partial2_pearson") if k in u},
                 **{k + "_ci": u[k]["ci_group"] for k in ("partial_spearman", "partial_pearson") if k in u}}
        pr[f]["oriented"] = [o * pr[f]["partial_spearman"], o * pr[f]["partial_pearson"]]
    prong_b = all(max(v["oriented"]) < KILL_R for v in pr.values())
    prong_b_abs = all(abs(v["partial_spearman"]) < KILL_R and abs(v["partial_pearson"]) < KILL_R
                      for v in pr.values())
    kill = bool(prong_a and prong_b)
    state["verdict"] = {
        "rule": (f"kill if no benchmark-equal test-like ALC difference is <= {KILL_ALC} under either slope variant "
                 f"and the oriented partial r on text-bearing matharena items is below {KILL_R} for every feature"),
        "rule_lines": rule_lines, "benchmark_equal_tl": alc, "partial_r_matharena_text": pr,
        "prong_alc_fails": prong_a, "prong_alc_fails_even_forced_per_pair": prong_a_forced,
        "prong_partial_r_below_oriented": prong_b, "prong_partial_r_below_absolute": prong_b_abs,
        "call": "KILL" if kill else "KEEP (the kill rule does not hold)"}
    H.save_json(args.out, state)
    print(json.dumps(state["verdict"], indent=1))


# --- show ---------------------------------------------------------------------------------

def _c(v, nd=2):
    if v is None or v.get("est") is None:
        return ""
    ci = v.get("ci_group") or v.get("ci_item")
    return f"{v['est']:+.{nd}f}" + ("" if not ci else f" [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]")


def _g(x, nd=4):
    return "" if x is None else f"{x:+.{nd}f}"


def stage_show(args):
    s = H.load_json(args.out)
    sg = s.get("signs")
    if sg:
        print("coverage:", json.dumps(sg["coverage"]))
        for unit in ("matharena", "matharena text-bearing", "matharena text-bearing 2025",
                     "matharena text-bearing 2026", "multi_swebench"):
            print(f"\n{unit}\n| feature | declared | n | Spearman | Pearson | Spearman within group | "
                  "Pearson within group | partial Spearman | partial Pearson | partial Spearman, position too | "
                  "partial Pearson, position too |\n|" + "---|" * 11)
            for f, e in sg["features"].items():
                v = e["units"].get(unit)
                if not v or not v.get("spearman"):
                    continue
                print(f"| {f} | {e['declared_sign'] if e['declared_sign'] is not None else ''} | {v['n']} | "
                      + " | ".join(_c(v.get(k)) for k in STATS) + " |")
        print("\nitem-bootstrap intervals, multi_swebench:")
        for f, e in sg["features"].items():
            v = e["units"].get("multi_swebench")
            if v and v.get("spearman"):
                print(f"  {f}: " + ", ".join(f"{k} {v[k]['est']:+.3f} {v[k]['ci_item']}" for k in STATS if k in v))
        print("\nsign rule:")
        for f, e in sg["features"].items():
            if "sign_rule" in e:
                print(f"  {f}: {e['sign_rule']}\n      within: {e['sign_rule_within']}")
        print("probe check:", sg.get("probe_items_check"))
    hz = s.get("harness")
    if hz:
        print("\n| feature, line | test-like ± cluster SE (sel) | benchmark-equal | matharena | multi_swebench "
              "| mix/whole | R1 b / p | folds on | placebo test-like / benchmark-equal |\n|" + "---|" * 9)
        for f, v in hz.items():
            if f.startswith("_"):
                continue
            for name, ln in v["lines"].items():
                pp = ln["tl_per_parent"]
                pl = v["placebo"].get(name, {})
                sel = ln.get("tl_sel_se")
                print(f"| {f}, {name} | {_g(ln['tl'], 5)} ± {ln['tl_cluster_se']:.5f}"
                      + ("" if sel is None else f" ({sel:.5f})")
                      + f" | {_g(ln['tl_benchmark_equal'], 5)} | "
                      + " | ".join(f"{_g(pp[q][0], 5)} ± {pp[q][1]:.5f}" if q in pp else "" for q in
                                   ("matharena", "multi_swebench"))
                      + f" | {_g(ln['mix'], 5)} | {_g(ln['r1b'], 5)} / {_g(ln['r1p'], 5)} | "
                      + ("" if ln["folds_on"] is None else f"{ln['folds_on']}/4")
                      + f" | {_g(pl.get('tl'), 5)} / {_g(pl.get('tl_benchmark_equal'), 5)} |")
            print(f"  {f}: within-pair r {v.get('r_within_pair_tl')}, varies on "
                  f"{v.get('share_tl_appearances_varying')} of test-like appearances; coverage "
                  f"{v['coverage_eval_items']}")
    if s.get("reference"):
        print("\nreference (honest difficulty degraded to r on the rated items):")
        for r, t in s["reference"]["r"].items():
            print(f"  {r}: " + "  ".join(f"{n} {v['tl']:+.5f} (eq {v['tl_benchmark_equal']:+.5f})"
                                         for n, v in t.items()))
    if s.get("verdict"):
        print("\nverdict:", json.dumps(s["verdict"], indent=1))


# --- io -----------------------------------------------------------------------------------

def meta(args, tinfo):
    with open(os.path.join(args.features, "manifest.json")) as f:
        man = json.load(f)
    return {**H.provenance(), "script_digest_llm4b": H.digest(["experiments/llm4b_close.py"]),
            "features_digest": features_digest(args.features),
            "features_manifest": {"config_hash": man.get("config_hash"), "progress": man.get("progress"),
                                  "llm_repo": F.LLM_REPO},
            "llm_shards_on_disk": {b: len([f for f in os.listdir(os.path.join(args.features, b, "llm"))
                                           if f.endswith(".npz") and not f.startswith(".")])
                                   for b in BENCHES if os.path.isdir(os.path.join(args.features, b, "llm"))},
            "features": FEATURES, "diag": DIAG, "boot": args.boots, "min_text": MIN_TEXT, "year_2026": YEAR_2026,
            "kill": {"alc": KILL_ALC, "r": KILL_R}, "target_info": tinfo,
            "extraction": ("not resumed: the remaining rating work (multi_swebench's 4 longest shards, "
                           "real_webagents, researchcodebench, 0.67 M prompt tokens; swe_rebench 3.8 M) needs "
                           "the fp16 4B (8 GB) resident beside other LLM jobs of the same workflow; at the "
                           "manifest's contended rate (79 tokens/s) that is 2.3 h for the four parents (about 45 "
                           "minutes at the ~260 tokens/s of the first, uncontended shards), and the extraction last "
                           "died of MPS OOM on exactly these long shards")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("signs", "harness", "reference", "verdict", "show"))
    ap.add_argument("--features", default=FEAT_DIR)
    ap.add_argument("--rows", default=H.ROWS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boots", type=int, default=BOOT)
    ap.add_argument("--placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--feats", nargs="+", default=list(FEATURES), choices=list(FEATURES))
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    {"signs": stage_signs, "harness": stage_harness, "reference": stage_reference, "verdict": stage_verdict,
     "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
