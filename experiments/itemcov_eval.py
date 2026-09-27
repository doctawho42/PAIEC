"""Item covariates with a known sign (paiec/itemcov.py): sign check and gate.

Candidates, each a pure function of the item dict predict() receives, higher =
harder by declaration (paiec/itemcov.py says why for each):

  ordinal_difficulty  a difficulty-named or work-count item_features key
  position            log(1 + problem index), from a position-named key
                      (matharena problem_idx); plus position_within, the index's
                      percentile within the item's group value (competition),
                      which reads the other items' features and is a diagnostic
  format_score        mcq -1 < short = code = free 0 < proof +1
  log_length          log(1 + characters of item_content)
  stated_size         log(1 + the first stated amount of work, 'Approximately 7
                      line(s) of code')
  image_ref           the text points at an image it does not hold

--stage signs. Per unit, the Spearman correlation of each covariate with item
difficulty, over the unit's items and within its item_features groups (ranks
over the unit, demeaned within group, Pearson; hier learns group levels from
labels, so the within-group part is what a covariate can add on top), with a
bootstrap over groups (items where there are none). Units: the four
multi-subject parents, difficulty = honest Rasch difficulty (the mean over the
harness's subject folds of difficulties fitted without the fold's subjects,
experiments/harness.py oracle_maps); swe_rebench, one subject, -logit of the
item's smoothed pass rate over its trials; and mmdocrag as a directional sixth
unit (fraction responses, -logit of the mean), not counted. A unit applies when
the covariate is present on at least MIN_ITEMS (20) of its items and at least
MIN_VARY (10) of them are off its most common value (three multiple-choice
items among 212 code items do not make a format covariate). The
sign rule (the plan's): a covariate may carry a TRANSFERRED slope only if on at
least 4 of the 5 units (a) its correlation has the declared sign and (b) that
sign equals the sign of the mean correlation of the other applicable units
(leave one unit out). A covariate that applies on fewer than 4 units cannot pass,
whatever its correlations.

--stage harness. Each covariate through the acceptance harness on its stored
rows of the shipped hier (experiments/harness.py; data/harness_rows). x is the
covariate standardised within each public benchmark (standardised(): 0 where
the item lacks the cue or the benchmark barely varies in it, e.g. position
outside matharena), and the per-pair prior sd is s per within-benchmark sd
(BenchScaleEngine; the harness's own scale, an sd pooled over the training
parents, is wrong for covariates absent from some parents). Lines:
  allowed nested   nested leave-one-parent-out selection over the configurations
                   the sign rule allows: a covariate that fails it gets only the
                   zero-centred per-pair slope from B7 with prior sd s in
                   PP_S (0.1, 0.25, 0.5 per sd of x; the plan caps s at 0.5 for
                   a feature whose sign is not known); one that passes may also
                   use the transferred slope (from B1 or B7) and the hybrid
  harness lines    the harness's own nested lines per variant (transferred,
                   per-pair, hybrid; its S_GRID reaches s = 2) and its forced
                   lines, for comparison only; a transferred line of a covariate
                   that failed the sign rule is reported but may not ship
  forced per-pair  each allowed per-pair configuration (s in PP_S, from B7)
                   switched on everywhere without selection
  placebo          the allowed nested line and the forced per-pair lines with x
                   permuted within each benchmark (N_PLACEBO draws, differences
                   averaged over them): what the same configuration gains or
                   costs from noise with the same support
The gate is experiments/harness.py GATE: the selection switches the term on in
at least 3 of the 4 outer folds, test-like ALC difference <= -0.002, the
mix/whole difference of the same sign, no held-out parent above +0.002, and
neither public R1 weighting above +0.001. A covariate present on one parent
only cannot be switched on leave-one-parent-out (with that parent held out its
inner folds see no covariate), so its forced per-pair line on that parent is
the number to read. Each line also carries its realised correlation with
honest difficulty within test-like pairs (r_within_pair_tl, the harness
table's column) so it can be read against results/harness_thresholds.json.

--stage inventory. How often such cues could exist in hidden benchmarks. The
organisers' inventory (results/inventory.csv, 161 benchmarks) holds titles and
links but no items, so this is a keyword scan of titles and nothing more.

--stage show. Markdown tables of results/itemcov_eval.json.

Run:  python experiments/itemcov_eval.py --stage signs       # ~1 min
      python experiments/itemcov_eval.py --stage harness     # ~15 min, one process, < 1.5 GB
      python experiments/itemcov_eval.py --stage harness --scale train   # ~5 min, the harness's own scale
      python experiments/itemcov_eval.py --stage inventory   # seconds
      python experiments/itemcov_eval.py --stage show
Needs the data (data/<benchmark>/) and the harness rows (python
experiments/harness.py --stage collect). Each stage writes its part of
results/itemcov_eval.json and keeps the others.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import harness as H  # noqa: E402
from paiec import itemcov as IC  # noqa: E402

OUT = os.path.join(ROOT, "results", "itemcov_eval.json")
UNITS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
DIRECTIONAL = ("mmdocrag",)
GROUP_KEY = {"matharena": "competition", "multi_swebench": "lang",
             "real_webagents": "website", "researchcodebench": "paper"}
COVS = ("ordinal_difficulty", "position", "position_within", "format_score", "log_length",
        "stated_size", "image_ref")
DECLARED_SIGN = {c: 1 for c in COVS}         # higher = harder, every one
MIN_ITEMS = 20
MIN_VARY = 10                                 # items off the covariate's most common value
N_AGREE = 4
BOOT = 1000
PP_S = (0.1, 0.25, 0.5)
CLIP = 3.0
PP_FORCED = tuple(("per-pair", s, 7) for s in PP_S)
N_PLACEBO = 3


# --- items and covariates ----------------------------------------------------------------

def load_items(bench):
    """{item key: official item dict} of one benchmark (the key is str(item_id),
    as in paiec.data.load_pairs and the harness rows)."""
    import pandas as pd
    from paiec.data import _clean, anon_id
    it = pd.read_parquet(os.path.join(ROOT, "data", bench, "items.parquet"),
                         columns=["item_id", "content", "item_features"])
    bid = anon_id("benchmark", bench)
    return {str(r.item_id): {"item_content": _clean(r.content), "item_features": _clean(r.item_features),
                             "interactors": "", "benchmark_id": bid}
            for r in it.itertuples(index=False)}


def covariate_maps(items_by_bench):
    """{covariate: {item key: x}} over every benchmark's items; an item without
    the cue is left out (the harness reads it as 0). position_within is the
    rank of the item's position among the items of its bench and group value,
    scaled to [0, 1]."""
    maps = {c: {} for c in COVS}
    groups = {}
    for bench, items in items_by_bench.items():
        for k, item in items.items():
            cv = IC.covariates(item)
            for c, x in cv.items():
                if x is not None and math.isfinite(x):
                    maps[c][k] = x
            if cv.get("position") is not None:
                g = IC.features(item).get(GROUP_KEY.get(bench, ""), "")
                groups.setdefault((bench, g), []).append((cv["position"], k))
    for members in groups.values():
        xs = np.array([m[0] for m in members])
        if len(members) < 2 or xs.std() == 0:
            continue
        r = _rank(xs)
        for (x, k), rk in zip(members, (r - 1) / (len(r) - 1)):
            maps["position_within"][k] = float(rk)
    return maps


def _rank(x):
    """Average ranks, 1..n."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.bincount(inv, r)
    return (s / cnt)[inv]


# --- difficulty targets ---------------------------------------------------------------------

def difficulties():
    """{unit: {item key: difficulty}}, higher = harder, and how each was made."""
    import pandas as pd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ins, honest, info = H.oracle_maps()
    out, how = {}, {}
    for par in H.PARENTS:
        keys = set().union(*(set(m) & set(ins[par]) for m in honest))
        d = {}
        for k in keys:
            v = [m[k] for m in honest if k in m]
            if v:
                d[k] = float(np.mean(v))
        out[par] = d
        how[par] = f"honest Rasch difficulty, mean over {H.HONEST_FOLDS} subject folds ({len(d)} items)"
    r = pd.read_parquet(os.path.join(ROOT, "data", "swe_rebench", "response.parquet"),
                        columns=["item_id", "response"])
    r = r[r.response.isin([0.0, 1.0])]
    g = r.groupby(r.item_id.astype(str)).response.agg(["sum", "count"])
    out["swe_rebench"] = {k: float(-math.log((s + 0.5) / (n - s + 0.5))) for k, s, n in
                          zip(g.index, g["sum"], g["count"])}
    how["swe_rebench"] = f"-logit of (successes + 0.5) / (trials + 1), one subject ({len(g)} items)"
    r = pd.read_parquet(os.path.join(ROOT, "data", "mmdocrag", "response.parquet"),
                        columns=["item_id", "response"])
    g = r.groupby(r.item_id.astype(str)).response.mean().clip(0.01, 0.99)
    out["mmdocrag"] = {k: float(-math.log(v / (1 - v))) for k, v in g.items()}
    how["mmdocrag"] = f"-logit of the mean fractional response over subjects ({len(g)} items), directional"
    return out, how


# --- signs --------------------------------------------------------------------------------

def unit_corr(x, d, grp, boot=BOOT, seed=0):
    """Spearman over the unit and within groups (ranks over the unit, demeaned
    within group, Pearson), each with a bootstrap SE over groups."""
    x, d = np.asarray(x, float), np.asarray(d, float)
    grp = np.asarray(grp, object)

    def both(ix):
        xx, dd, gg = x[ix], d[ix], grp[ix]
        rx, rd = _rank(xx), _rank(dd)
        raw = _pearson(rx, rd)
        _, inv = np.unique(gg, return_inverse=True)
        mx = np.bincount(inv, rx) / np.bincount(inv)
        md = np.bincount(inv, rd) / np.bincount(inv)
        return raw, _pearson(rx - mx[inv], rd - md[inv])

    raw, within = both(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(uniq))]
    bs = []
    for _ in range(boot):
        pick = rng.integers(0, len(members), len(members)) if len(members) > 1 else None
        if pick is None:
            ix = rng.integers(0, len(x), len(x))
        else:
            ix = np.concatenate([members[j] for j in pick])
        bs.append(both(ix))
    bs = np.array(bs, float)
    return {"rho": raw, "rho_se": float(np.nanstd(bs[:, 0], ddof=1)),
            "rho_within": within, "rho_within_se": float(np.nanstd(bs[:, 1], ddof=1)),
            "boot_unit": "group" if len(members) > 1 else "item"}


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def sign_rule(per_unit, declared, field="rho"):
    """The plan's rule on one covariate: over the applicable units, how many have
    the declared sign, and how many agree with the mean of the others."""
    app = {u: v[field] for u, v in per_unit.items() if u in UNITS and v.get("applies")
           and v.get(field) is not None and math.isfinite(v[field])}
    declared_ok = {u: bool(np.sign(r) == declared) for u, r in app.items()}
    lobo = {}
    for u, r in app.items():
        others = [app[w] for w in app if w != u]
        lobo[u] = bool(others) and bool(np.sign(r) == np.sign(np.mean(others)))
    n_dec, n_lobo = sum(declared_ok.values()), sum(lobo.values())
    return {"applicable": sorted(app), "n_applicable": len(app), "declared_sign_units": n_dec,
            "lobo_agree_units": n_lobo, "declared": declared,
            "transferred_allowed": bool(n_dec >= N_AGREE and n_lobo >= N_AGREE)}


def stage_signs(args):
    t0 = time.time()
    items = {b: load_items(b) for b in UNITS + DIRECTIONAL}
    maps = covariate_maps(items)
    diff, how = difficulties()
    res = {"targets": how, "min_items": MIN_ITEMS, "min_vary": MIN_VARY, "n_agree": N_AGREE, "boot": BOOT,
           "covariates": {}}
    for c in COVS:
        per = {}
        for u in UNITS + DIRECTIONAL:
            keys = [k for k in items[u] if k in diff[u]]
            have = [k for k in keys if k in maps[c]]
            x = np.array([maps[c][k] for k in have], float)
            entry = {"items": len(keys), "with_cue": len(have), "distinct": int(len(np.unique(x))) if len(x) else 0}
            entry["off_mode"] = int(len(x) - np.unique(x, return_counts=True)[1].max()) if len(x) else 0
            if len(have) >= MIN_ITEMS and entry["distinct"] >= 2 and entry["off_mode"] >= MIN_VARY:
                d = np.array([diff[u][k] for k in have])
                gk = GROUP_KEY.get(u)
                grp = [IC.features(items[u][k]).get(gk, "") if gk else "" for k in have]
                entry.update(unit_corr(x, d, grp))
                entry["applies"] = True
                if len(entry) and entry["distinct"] <= 6:
                    vals, cnt = np.unique(x, return_counts=True)
                    entry["values"] = {f"{v:g}": int(n) for v, n in zip(vals, cnt)}
            else:
                entry["applies"] = False
            per[u] = entry
        rule = sign_rule(per, DECLARED_SIGN[c])
        rule_within = sign_rule(per, DECLARED_SIGN[c], "rho_within")
        res["covariates"][c] = {"units": per, "rule": rule, "rule_within_groups": rule_within}
        print(f"{c:20s} " + "  ".join(
            f"{u[:6]} {per[u]['rho']:+.3f}/{per[u]['rho_within']:+.3f}" if per[u]["applies"] else f"{u[:6]} --"
            for u in UNITS + DIRECTIONAL) + f"  -> declared {rule['declared_sign_units']}/"
            f"{rule['n_applicable']}, lobo {rule['lobo_agree_units']}, transferred "
            f"{'allowed' if rule['transferred_allowed'] else 'not allowed'}", flush=True)
    fmt = {}
    for u in UNITS + DIRECTIONAL:
        cnt = {}
        for item in items[u].values():
            f = IC.answer_format(item)
            cnt[f] = cnt.get(f, 0) + 1
        fmt[u] = cnt
    res["answer_format_counts"] = fmt
    res["wall_s"] = round(time.time() - t0, 1)
    state = H.load_json(args.out) or {}
    state["signs"] = res
    state.update(H.provenance())
    state["itemcov_digest"] = H.digest(["paiec/itemcov.py", "experiments/itemcov_eval.py"])
    H.save_json(args.out, state)
    print(f"signs: {res['wall_s']}s -> {args.out}", flush=True)


# --- harness --------------------------------------------------------------------------------

def nested_over(engine, cands):
    """experiments/harness.py nested() over an explicit list of configurations
    (variant, s, start) instead of one variant's grid."""
    rows = engine.rows
    folds = {q: tuple(p for p in H.PARENTS if p != q) for q in H.PARENTS}
    folds["*"] = H.PARENTS
    out = {reg: np.zeros((R.A, 6)) for reg, R in rows.items()}
    choices = {}
    sel = rows[H.SELECT_ON]
    for q, train in folds.items():
        crit = {}
        for c in cands:
            vals = []
            for q2 in train:
                inner = tuple(p for p in train if p != q2)
                d = engine.delta(H.SELECT_ON, c, inner) @ H.W6
                m = sel.parent == q2
                vals.append(H.weighted_mean(d[m], sel.w[m]))
            crit[c] = float(np.mean(vals))
        choice = H.select(crit)
        choices[q] = {"choice": None if choice is None else list(choice),
                      "inner": {H.cname(c): round(v, 7) for c, v in crit.items()}}
        if choice is None:
            continue
        if choice[0] != "per-pair":
            choices[q]["beta"] = [round(float(b), 5) for b in engine.beta(train)]
        choices[q]["sd_x"] = round(engine.sd(train), 6)
        for reg, R in rows.items():
            m = (R.parent == q) if q != "*" else ~np.isin(R.parent, H.PARENTS)
            if m.any():
                out[reg][m] = engine.delta(reg, choice, train)[m]
    return out, choices


def allowed_cands(transferred_allowed):
    pp = [("per-pair", s, 7) for s in PP_S]
    if not transferred_allowed:
        return pp
    return (H.configs_of("transferred") + [("per-pair", s, st) for s in PP_S for st in H.STARTS]
            + [("hybrid", s, st) for s in PP_S for st in H.STARTS])


def line_of(rows, dB, choices=None, boots=H.BOOTS, per=True):
    line = {"regimes": {reg: H.summary(R, dB[reg], boots, 0, per) for reg, R in rows.items()}}
    if choices is not None:
        line["choices"] = choices
        line["folds_on"] = sum(1 for q in H.PARENTS if choices[q]["choice"] is not None)
    line["gate"] = H.gate(line)
    return line


def realised_r(rows, cov, honest):
    """Mean, over test-like appearances with 5+ evaluated items on which x
    varies, of corr(x, honest difficulty of the subject's fold) over the pair's
    evaluated items (the harness table's r_within_pair_tl), and the share of
    appearances on which x varies at all."""
    R = rows["tl"]
    keys = cov.keys
    rs, varies = [], 0
    for a in range(R.A):
        m = R.ev_a == a
        if m.sum() < 5:
            continue
        x = cov.X[0, R.ev_k[m]]
        f = H.fold_of(R.sid[a])
        z = np.array([honest[f].get(keys[k], np.nan) for k in R.ev_k[m]])
        ok = np.isfinite(z)
        if x.std() > 0:
            varies += 1
            if ok.sum() >= 5 and x[ok].std() > 0 and z[ok].std() > 0:
                rs.append(np.corrcoef(x[ok], z[ok])[0, 1])
    n = int(sum(1 for a in range(R.A) if (R.ev_a == a).sum() >= 5))
    return {"r_within_pair_tl": float(np.mean(rs)) if rs else None,
            "share_tl_appearances_varying": varies / max(n, 1)}


class Cov(H.Covariate):
    def __init__(self, X, keys, name):
        super().__init__(X, None, name)
        self.keys = keys


class BenchScaleEngine(H.Engine):
    """The harness's Engine with x already standardised within each benchmark
    (standardised()), so the per-pair prior sd is s per sd of x on the target's
    own benchmark. The harness takes the sd over the training parents' items
    pooled, which for a covariate absent from some parents (format_score:
    0.047) or with benchmark-level offsets (log_length: 2.36) turns s = 0.5
    into a prior sd of 10.7 or 0.21 per within-benchmark sd."""

    def sd(self, train):
        return 1.0


def standardised(xmap, items_bench):
    """x standardised within each benchmark over the items that carry the cue,
    clipped at +-CLIP sd, the rest at the benchmark's mean (0); a benchmark
    where fewer than MIN_VARY items are off the most common value gets 0 on
    every item. A runtime would standardise over the benchmark's visible items
    (the labeled ones and the target), not over all of them."""
    out = {}
    for bench, keys in items_bench.items():
        ks = [k for k in keys if k in xmap]
        if not ks:
            continue
        x = np.array([xmap[k] for k in ks], float)
        off = len(x) - np.unique(x, return_counts=True)[1].max()
        if off < MIN_VARY or x.std() == 0:
            continue
        z = np.clip((x - x.mean()) / x.std(), -CLIP, CLIP)
        out.update({k: float(v) for k, v in zip(ks, z)})
    return out


def permuted(xmap, items_bench, seed):
    """x shuffled among the items of each benchmark that carry it."""
    rng = np.random.default_rng([seed, 4241])
    out = {}
    for bench, keys in items_bench.items():
        ks = [k for k in keys if k in xmap]
        xs = np.array([xmap[k] for k in ks])
        for k, v in zip(ks, rng.permutation(xs)):
            out[k] = float(v)
    return out


def stage_harness(args):
    t0 = time.time()
    state = H.load_json(args.out) or {}
    if "signs" not in state:
        raise SystemExit("run --stage signs first")
    items = {b: load_items(b) for b in UNITS}
    maps = covariate_maps(items)
    items_bench = {b: list(v) for b, v in items.items()}
    del items
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, honest, _ = H.oracle_maps()
    print(f"harness: rows and maps in {time.time() - t0:.0f}s", flush=True)
    key = "harness" if args.scale == "bench" else "harness_train_scale"
    Eng = BenchScaleEngine if args.scale == "bench" else H.Engine
    res = state.get(key, {})
    todo = [c for c in args.covs if c not in res or args.redo]
    for c in todo:
        t1 = time.time()
        rule = state["signs"]["covariates"][c]["rule"]
        xmap = standardised(maps[c], items_bench) if args.scale == "bench" else maps[c]
        X = np.array([[float(xmap.get(k, 0.0)) for k in keys]])
        cov = Cov(X, keys, c)
        entry = {"n_items_with_cue": len(maps[c]), "n_items_nonzero_standardised": int(sum(v != 0 for v in xmap.values())),
                 "transferred_allowed": rule["transferred_allowed"],
                 "coverage_eval_items": {reg: float(np.mean(np.isin(np.array(keys, object)[R.ev_k], list(xmap))))
                                         for reg, R in rows.items()}}
        if not np.any(X != X.flat[0]):
            entry["skipped"] = "the covariate is constant on every stored item: q = p, difference 0"
            res[c] = entry
            print(f"{c}: constant, skipped", flush=True)
            continue
        entry.update(realised_r(rows, cov, honest))
        eng = Eng(rows, cov)
        lines = {}
        cands = allowed_cands(rule["transferred_allowed"])
        dB, ch = nested_over(eng, cands)
        lines["allowed nested"] = line_of(rows, dB, ch)
        for v in H.VARIANTS:
            dB, ch = H.nested(eng, v)
            lines[f"{v} nested (harness grid)"] = line_of(rows, dB, ch, boots=args.boots_ref, per=False)
        for fc in H.FORCED:
            dB, ch = H.nested(eng, fc[0], forced=fc)
            lines[H.cname(fc) + " (forced)"] = line_of(rows, dB, None, boots=args.boots_ref, per=True)
        for fc in PP_FORCED:
            if H.cname(fc) + " (forced)" not in lines:
                dB, ch = H.nested(eng, fc[0], forced=fc)
                lines[H.cname(fc) + " (forced)"] = line_of(rows, dB, None, boots=args.boots_ref, per=True)
        entry["lines"] = lines
        # placebo: x permuted within each benchmark, the allowed nested line and the
        # forced per-pair lines, each draw's per-appearance differences averaged
        plac = {"allowed nested": [], **{H.cname(fc) + " (forced)": [] for fc in PP_FORCED}}
        raw = {name: [] for name in plac}
        for s in range(args.placebo):
            pm = permuted(xmap, items_bench, s)
            Xp = np.array([[float(pm.get(k, 0.0)) for k in keys]])
            e2 = Eng(rows, Cov(Xp, keys, c + " placebo"))
            dB, ch = nested_over(e2, cands)
            raw["allowed nested"].append(dB)
            plac["allowed nested"].append({"folds_on": sum(1 for q in H.PARENTS if ch[q]["choice"] is not None)})
            for fc in PP_FORCED:
                dB, _ = H.nested(e2, fc[0], forced=fc)
                raw[H.cname(fc) + " (forced)"].append(dB)
        entry["placebo"] = {}
        for name, draws in raw.items():
            if not draws:
                continue
            avg = {reg: np.mean([d[reg] for d in draws], 0) for reg in rows}
            ln = line_of(rows, avg, None, boots=200, per=True)
            entry["placebo"][name] = {
                "draws": len(draws),
                "est": {reg: (None if v is None else round(v["est"], 7)) for reg, v in ln["regimes"].items()},
                "tl_per_parent": {q: round(v["est"], 7) for q, v in ln["regimes"]["tl"]["per_parent"].items()},
                "tl_draw_ests": [round(H.weighted_mean(d["tl"] @ H.W6, rows["tl"].w), 7) for d in draws]}
            if name == "allowed nested":
                entry["placebo"][name]["folds_on"] = [p["folds_on"] for p in plac[name]]
        entry["wall_s"] = round(time.time() - t1, 1)
        res[c] = entry
        state[key] = res
        H.save_json(args.out, state)
        a = lines["allowed nested"]
        rg = a["regimes"]
        fl = lines['per-pair s=0.5 from B7 (forced)']['regimes']['tl']
        print(f"{c}: {entry['wall_s']}s  allowed nested tl {rg['tl']['est']:+.5f} (cl {rg['tl']['cluster_se']:.5f}) "
              f"mix {rg['mix']['est']:+.5f} r1b {rg['r1b']['est']:+.5f} r1p {rg['r1p']['est']:+.5f} folds_on "
              f"{a['folds_on']}/4 gate {a['gate']['pass']}; forced per-pair s=0.5 B7 tl {fl['est']:+.5f} "
              + " ".join(f"{q[:5]} {v['est']:+.5f}" for q, v in fl['per_parent'].items())
              + f"; placebo of that line tl {entry['placebo']['per-pair s=0.5 from B7 (forced)']['est']['tl']:+.5f}",
              flush=True)
    state[key] = res
    state[key + "_meta"] = {"pp_s": PP_S, "n_placebo": args.placebo, "boots": H.BOOTS, "clip": CLIP,
                             "scale": ("x standardised within benchmark (standardised), per-pair prior sd s per "
                                       "within-benchmark sd (BenchScaleEngine)") if args.scale == "bench" else
                                      ("raw x, per-pair prior sd s per sd of x pooled over the training parents' "
                                       "evaluated items (experiments/harness.py Engine as it is)"),
                             "boots_ref": args.boots_ref, "gate": H.GATE, "rows": os.path.relpath(args.rows, ROOT),
                             "runs": {r: H.PLAN[r] for r in H.PLAN},
                             "thresholds": "results/harness_thresholds.json (honest table)",
                             "rows_note": "hier's predictions are read from the stored rows, never recomputed; the "
                                          "library digest at the top level is of the files now on disk, the one "
                                          "below is the harness table's, written beside the rows",
                             "rows_provenance": {k: (H.load_json(H.OUT) or {}).get("meta", {}).get(k)
                                                 for k in ("commit", "lib_digest", "script_digest", "level")}}
    state.update(H.provenance())
    state["itemcov_digest"] = H.digest(["paiec/itemcov.py", "experiments/itemcov_eval.py"])
    H.save_json(args.out, state)
    print(f"harness: {time.time() - t0:.0f}s -> {args.out}", flush=True)


# --- inventory --------------------------------------------------------------------------------

#: title keywords, per cue, in the organisers' inventory (titles only; no items)
TITLE_CUES = {
    "per-item difficulty levels or tiers named": r"\b(?:difficulty[- ]levels?|difficulty[- ]tiers?|tiers?\b|"
                                                 r"progressive|graded difficulty|curricul\w*|easy[- ]to[- ]hard|"
                                                 r"levels? of (?:difficulty|complexity))",
    "multi-step, long-horizon or planning (a step count could be recorded)": r"\b(?:multi-?step|long-?horizon|"
                                                                             r"multi-?hop|multi-?turn|stepwise|"
                                                                             r"planning)",
    "agents": r"\bagents?\b|\bagentic\b",
    "proofs or theorem proving": r"\b(?:proofs?|proving|theorem)",
    "multiple choice": r"\b(?:multiple[- ]choice|mcq)",
    "math or olympiad": r"\b(?:math\w*|olympiad|arithmetic|geometry|inequalit\w*)",
    "code or software engineering": r"\b(?:code|coding|program\w*|software|repositor\w*|swe|bug\w*|compil\w*)",
    "images, video, charts, documents, 3D": r"\b(?:image\w*|visual\w*|vision|video\w*|chart\w*|diagram\w*|"
                                            r"multimodal|mllm\w*|vlm\w*|documents?|figures?|scenes?|3d|spatial)",
}


def stage_inventory(args):
    import pandas as pd
    inv = pd.read_csv(os.path.join(ROOT, "results", "inventory.csv"))
    title = inv["title"].fillna("").astype(str)
    out = {"n": int(len(inv)), "source": "results/inventory.csv (titles only)",
           "caveat": "a title says what a benchmark is about, not which fields its items carry; "
                     "no item of a hidden benchmark is visible", "cues": {}}
    for name, pat in TITLE_CUES.items():
        m = title.str.contains(pat, flags=re.I, regex=True)
        out["cues"][name] = {"share": round(float(m.mean()), 3), "n": int(m.sum()),
                             "examples": title[m].head(6).tolist()}
        print(f"{name:60s} {m.mean():.3f} ({int(m.sum())})", flush=True)
    state = H.load_json(args.out) or {}
    state["inventory"] = out
    H.save_json(args.out, state)


# --- show -------------------------------------------------------------------------------------

def _g(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def stage_show(args):
    st = H.load_json(args.out)
    sg = st.get("signs")
    if sg:
        print("Spearman with difficulty over the unit ± bootstrap SE (within groups)\n")
        print("| covariate | " + " | ".join(UNITS + DIRECTIONAL) + " | declared sign / applicable | LOBO agree | transferred |")
        print("|---|" + "---|" * (len(UNITS + DIRECTIONAL) + 3))
        for c, v in sg["covariates"].items():
            cells = []
            for u in UNITS + DIRECTIONAL:
                e = v["units"][u]
                cells.append(f"{_g(e['rho'])} ± {_g(e['rho_se'], 2, False)} ({_g(e['rho_within'])})" if e["applies"]
                             else f"-- ({e['with_cue']}/{e['items']}, {e.get('off_mode', 0)} off mode)")
            r = v["rule"]
            print(f"| {c} | " + " | ".join(cells) + f" | {r['declared_sign_units']}/{r['n_applicable']} | "
                  f"{r['lobo_agree_units']} | {'yes' if r['transferred_allowed'] else 'no'} |")
        print("\nwithin-group rule:", {c: (v["rule_within_groups"]["declared_sign_units"],
                                         v["rule_within_groups"]["n_applicable"],
                                         v["rule_within_groups"]["lobo_agree_units"])
                                     for c, v in sg["covariates"].items()})
        print("answer formats:", sg["answer_format_counts"])
    hs = st.get("harness", {})
    if hs:
        print("\n" + H.HEAD.replace("| line |", "| covariate, line |"))
        for c, e in hs.items():
            if "lines" not in e:
                print(f"| {c}: {e.get('skipped', '')} |")
                continue
            for name, ln in e["lines"].items():
                if name == "allowed nested" or name.startswith("per-pair s=") or name == "transferred nested (harness grid)":
                    print(H.line_row(f"{c}, {name}", ln, 5))
        print("\n| covariate | r within pair (tl) | tl appearances where x varies | line | test-like per held-out "
              "parent (cluster SE) | placebo, test-like per parent |\n|---|---|---|---|---|---|")
        for c, e in hs.items():
            if "lines" not in e:
                continue
            for name in ("allowed nested",) + tuple(H.cname(fc) + " (forced)" for fc in PP_FORCED):
                pp = e["lines"][name]["regimes"]["tl"]["per_parent"]
                pl = e["placebo"].get(name, {}).get("tl_per_parent", {})
                print(f"| {c} | {_g(e.get('r_within_pair_tl'), 3)} | {e['share_tl_appearances_varying']:.2f} | {name} | "
                      + ", ".join(f"{q[:5]} {v['est']:+.5f} ({v['cluster_se']:.5f})" for q, v in pp.items()) + " | "
                      + ", ".join(f"{q[:5]} {v:+.5f}" for q, v in pl.items()) + " |")
        print("\nby budget, test-like, B0..B31:")
        for c, e in hs.items():
            if "lines" in e:
                for name in ("allowed nested", "per-pair s=0.5 from B7 (forced)"):
                    bb = e["lines"][name]["regimes"]["tl"]["by_budget"]
                    print(f"{c:18s} {name:34s} " + " ".join(f"{v:+.5f}" for v in bb))
        print("\nchoices of the allowed nested line:")
        for c, e in hs.items():
            if "lines" in e:
                ch = e["lines"]["allowed nested"]["choices"]
                print(f"{c:18s} " + "  ".join(f"{q[:5]}: {H.cname(tuple(v['choice'])) if v['choice'] else 'off'}"
                                             for q, v in ch.items()))
    inv = st.get("inventory")
    if inv:
        print(f"\ninventory ({inv['n']} titles):")
        for k, v in inv["cues"].items():
            print(f"  {k}: {v['share']:.3f} ({v['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("signs", "harness", "inventory", "show"))
    ap.add_argument("--rows", default=H.ROWS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--covs", nargs="+", default=list(COVS), choices=list(COVS))
    ap.add_argument("--placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--boots-ref", type=int, default=500)
    ap.add_argument("--redo", action="store_true", help="rescore covariates already in the results")
    ap.add_argument("--scale", default="bench", choices=("bench", "train"),
                    help="bench: x standardised within benchmark (the primary result); train: raw x at the "
                         "harness's own per-pair scale, into 'harness_train_scale' (the artefact it shows)")
    args = ap.parse_args()
    {"signs": stage_signs, "harness": stage_harness, "inventory": stage_inventory,
     "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
