"""A 200-comparison slice: how well does Qwen3-4B-Instruct-2507 order two items
of one benchmark by difficulty, and do its errors stick to items?

Anchored comparisons (the rethink's "anchored-comparisons" study) would place a
new benchmark's items by comparing them with each other or with anchors. Their
value rests on two numbers this slice measures: the comparator's accuracy q at
ordering item pairs by honest difficulty, and whether its errors are
item-sticky (an item it misjudges once it misjudges again, so averaging k
comparisons does not wash the error out). The plan's rule (step 10, declared
before the run): DROP comparisons if the pooled q < Q_DROP, or q < the best
absolute signal on the same pairs + MARGIN (an absolute score costs one call per
item, k comparisons cost k).

Pairs (--stage plan, CPU). Per multi-subject parent a pool of POOL items from
the hidden-state probe's stratified sample (data/features/probe/sample.json,
experiments/hidden_state_probe.py; matharena's items there are text-bearing),
restricted to items with an honest difficulty (llm4b_close.honest_targets: Rasch
b without each of the five subject folds of experiments/harness.py, averaged
over the folds; higher = harder) and the probe's out-of-fold heads: a
systematic sample over honest difficulty (a seeded offset), so the pool spans
the benchmark's range. The pool is put in a seeded random order and compared
along a circulant graph (offsets 1, 2 and POOL/2): 50 comparisons per parent,
every item in exactly 5, 200 in all. A unique item stands for every item_id
with the same text (data/features/<b>/index.parquet); its first item_id is used.

Prompt (--stage run, the language model): the anchored-comparisons probe's
(rethink2/anchored-comparisons/probe_compare.py): system SYSTEM, one user turn
USER with the two tasks (each llmfeat.item_text, cut to ITEM_TOKENS by
llmfeat.head_tail with llmfeat.MARKER), "which task do FEWER of them solve
correctly, that is, which task is harder? Answer with one letter", and the
assistant header. One forward pass per prompt, nothing generated: the
next-token logits of "A" and "B" and their share of the full distribution. Both
presentation orders per pair (400 prompts). Batches are right-padded with no
attention mask (causal: no real token sees a pad) and read at each row's last
real token. Qwen3-4B-Instruct-2507 from the local Hugging Face cache in
weight-only int8 (experiments/attempt_probe.load_model). Raw rows go to
data/pairwise_probe/raw.jsonl (appended per batch, resumable). The run refuses to
start while another language-model process is alive.

--stage analyse (CPU). h = the order-averaged logit difference, oriented so that
h > 0 says item i is harder; q = the share of comparisons whose sign of h agrees
with the sign of the honest difficulty gap (ties 1/2), pooled and per parent,
with a Wilson interval and an item-block bootstrap (items resampled per parent,
a comparison weighted by the product of its items' multiplicities); the same for
each presentation order alone (what one prompt gives), by |gap| tercile, and
within item_features groups. Position consistency: both orders pick the same
item; share_first: the share of prompts answered with the first task (A);
mean_bias: the order-averaged logit difference toward A. Absolute signals on the same comparisons, oriented by their declared sign
(+ = harder): log length (itemcov.log_length, +), the hidden-state probe's
out-of-fold heads (data/features/probe/oof.json: fitted on the other three
parents, predictions of difficulty, +), and the 4B judge's rating (-), modal
digit (-) and digit entropy (+) where rated (matharena and multi_swebench;
llm4b_close.load_features). The best absolute signal is the maximum over the
signals present on all 200 comparisons (a maximum over eleven, so it flatters
the absolute side); the rated subset is reported separately. Equivalent r: under
a bivariate normal, concordance q = 1/2 + arcsin(r)/pi (Greiner), so r = sin(pi (q
- 1/2)): the correlation with difficulty an absolute score of the same q would
have. The comparator as an absolute score: least-squares Bradley-Terry scores
from its 5 comparisons per item (h_ij = t_i - t_j), Pearson with honest b over
each parent's 20 items.
Stickiness: (a) the residual of h on the gap (per parent, a h = a gap + c fit),
correlated between comparisons that share an item, oriented by the shared
item's side: under h_ij = a gap + e_i - e_j + eta, that correlation is var(e) /
(2 var(e) + var(eta)), so twice it is the persistent (item) share of the
comparator's error variance; (b) per item, the number of wrong comparisons out
of its 5, whose variance across items is compared with PERMS permutations of the
wrong/right labels within parent and |gap| tercile (independent errors): a
p-value and the observed / expected variance ratio.

Run:  python experiments/pairwise_probe.py --stage plan        # seconds, CPU
      python experiments/pairwise_probe.py --stage run         # 13 min, 285,190 tokens (361/s), the only LLM job
      python experiments/pairwise_probe.py --stage analyse     # seconds, CPU
      python experiments/pairwise_probe.py --stage show
Needs data/<benchmark>/, data/features/ (llm_features.py; the probe's sample and
out-of-fold heads from hidden_state_probe.py) and, for --stage run only,
Qwen/Qwen3-4B-Instruct-2507 in the local Hugging Face cache.
"""
import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.55")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.icl_probe import hh, load_items, log, other_llm_processes, read_json, write_json  # noqa: E402

OUT = os.path.join(ROOT, "results", "pairwise_probe.json")
DATA = os.path.join(ROOT, "data", "pairwise_probe")
PAIRS = os.path.join(DATA, "pairs.json")
RAW = os.path.join(DATA, "raw.jsonl")
PROBE = os.path.join(ROOT, "data", "features", "probe")
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
GROUP_KEY = {"matharena": "competition", "multi_swebench": "lang", "real_webagents": "website",
             "researchcodebench": "paper"}
POOL = 20
OFFSETS = (1, 2, POOL // 2)           # circulant graph: 50 edges on 20 nodes, degree 5
ITEM_TOKENS = 512
SEED = 11
Q_DROP, MARGIN = 0.60, 0.03
BUDGET, MAX_BATCH = 6144, 8           # padded tokens per forward, rows per forward
BOOTS, PERMS = 2000, 5000
HEADS = ("entropy", "hidden", "surprisal", "profile", "hidden_mean", "hidden_L9", "hidden_L18",
         "hidden_L27", "hidden_L36")
JUDGE = {"rating": -1, "digit_mode": -1, "entropy": 1}     # the 4B judge's declared signs

SYSTEM = ("You are an expert evaluator of AI systems. You judge how hard tasks are for the strongest "
          "current AI models.")
USER = (
    "Below are two tasks from the same AI evaluation benchmark. Each could be a question, a math problem, "
    "a programming or software-engineering task, a task for an agent on a website or computer, or the text "
    "part of a multimodal task. Very long tasks are shortened, with the middle omitted.\n\n"
    "<task A>\n{A}\n</task A>\n\n<task B>\n{B}\n</task B>\n\n"
    "Imagine many strong AI systems of 2025-2026 each attempt both tasks once. Which task do FEWER of them "
    "solve correctly, that is, which task is harder? Answer with one letter, A or B, and nothing else.")


# --- plan ---------------------------------------------------------------------------------

def stage_plan(args):
    import pandas as pd
    from experiments import llm4b_close as L4
    from paiec import itemcov as IC
    items = load_items(PARENTS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        honest, _, _ = L4.honest_targets()
    oof = read_json(os.path.join(PROBE, "oof.json"))
    sample = read_json(os.path.join(PROBE, "sample.json"))["sample"]
    pairs, pools = [], {}
    for b in PARENTS:
        idx = pd.read_parquet(os.path.join(ROOT, "data", "features", b, "index.parquet"), columns=["key", "item_ids"])
        rep = {k: str(ids[0]) for k, ids in zip(idx.key, idx.item_ids)}
        cand = []
        for k in sample[b]:
            iid = rep.get(k)
            if iid and iid in honest[b] and all(iid in oof[h] for h in HEADS) and iid in items[b]:
                cand.append((honest[b][iid], iid))
        cand.sort()
        n = len(cand)
        off = (hh(SEED, "offset", b) % 10000) / 10000
        pool = [cand[int((off + k) * n / POOL)][1] for k in range(POOL)]
        pool = sorted(pool, key=lambda i: hh(SEED, "order", b, i))
        pools[b] = pool
        seen = set()
        for u in range(POOL):
            for o in OFFSETS:
                v = (u + o) % POOL
                e = (min(u, v), max(u, v))
                if e in seen:
                    continue
                seen.add(e)
                i, j = pool[u], pool[v]
                gi = IC.features(items[b][i]).get(GROUP_KEY[b], "")
                gj = IC.features(items[b][j]).get(GROUP_KEY[b], "")
                pairs.append(dict(bench=b, i=i, j=j, b_i=honest[b][i], b_j=honest[b][j], g_i=gi, g_j=gj))
    write_json(PAIRS, dict(seed=SEED, pool=POOL, offsets=list(OFFSETS), pools=pools, pairs=pairs))
    log(f"{len(pairs)} comparisons:", {b: sum(p["bench"] == b for p in pairs) for b in PARENTS},
        "within group:", sum(p["g_i"] == p["g_j"] for p in pairs))


# --- run ----------------------------------------------------------------------------------

def stage_run(args):
    plan = read_json(PAIRS)
    if plan is None:
        raise SystemExit("run --stage plan first")
    others = other_llm_processes()
    if others and not args.force:
        raise SystemExit("another language-model process is alive:\n  " + "\n  ".join(others))
    import torch
    from paiec import llmfeat as F
    from experiments.attempt_probe import load_model
    pairs = plan["pairs"]
    done = set()
    if os.path.exists(RAW):
        with open(RAW) as f:
            done = {(r["k"], r["order"]) for r in map(json.loads, f)}
    items = load_items(PARENTS)
    t0 = time.time()
    model, tok, rev = load_model("int8")
    log(f"model loaded in {time.time() - t0:.0f}s (revision {rev})")
    ida, idb = F.encode(tok, "A"), F.encode(tok, "B")
    assert len(ida) == len(idb) == 1, "A/B are not single tokens"
    ida, idb = ida[0], idb[0]
    marker = F.encode(tok, F.MARKER)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.replace("{A}", "\u0000A\u0000").replace("{B}", "\u0000B\u0000")}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre, rest = rendered.split("\u0000A\u0000")
    mid, post = rest.split("\u0000B\u0000")
    pre, mid, post = F.encode(tok, pre), F.encode(tok, mid), F.encode(tok, post)
    cut = {}

    def ids_of(b, i):
        if (b, i) not in cut:
            cut[(b, i)], _ = F.head_tail(F.encode(tok, F.item_text(items[b][i])), ITEM_TOKENS, marker)
        return cut[(b, i)]

    jobs = []
    for k, p in enumerate(pairs):
        for order in (0, 1):
            if (k, order) in done:
                continue
            a, c = (p["i"], p["j"]) if order == 0 else (p["j"], p["i"])
            jobs.append((k, order, pre + ids_of(p["bench"], a) + mid + ids_of(p["bench"], c) + post))
    jobs.sort(key=lambda x: len(x[2]))
    batches, cur = [], []
    for job in jobs:
        if cur and (len(cur) + 1) * len(job[2]) > BUDGET or len(cur) == MAX_BATCH:
            batches.append(cur)
            cur = []
        cur.append(job)
    if cur:
        batches.append(cur)
    log(f"{len(jobs)} prompts in {len(batches)} batches, {sum(len(j[2]) for j in jobs):,} tokens")
    t0, ntok = time.time(), 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    with open(RAW, "a") as fh:
        for n, chunk in enumerate(batches):
            L = max(len(x[2]) for x in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            for r, x in enumerate(chunk):
                ids[r, :len(x[2])] = torch.tensor(x[2])
            with torch.inference_mode():
                hs = model.model(input_ids=ids.to("mps"), use_cache=False).last_hidden_state
                last = torch.tensor([len(x[2]) - 1 for x in chunk], device="mps")
                h = hs[torch.arange(len(chunk), device="mps"), last]
                lp = torch.log_softmax(model.lm_head(h).float(), -1).cpu().numpy()
            del hs
            torch.mps.empty_cache()
            for (k, order, x), row in zip(chunk, lp):
                fh.write(json.dumps({"k": k, "order": order, "lA": float(row[ida]), "lB": float(row[idb]),
                                     "mass": float(np.exp(row[ida]) + np.exp(row[idb])), "tokens": len(x)}) + "\n")
            fh.flush()
            ntok += sum(len(x[2]) for x in chunk)
            log(f"batch {n + 1}/{len(batches)}: {ntok:,} tokens, {time.time() - t0:.0f}s "
                f"({ntok / max(time.time() - t0, 1e-9):.0f}/s)")
    info = dict(rendered=rendered, model_revision=rev, precision="int8", item_tokens=ITEM_TOKENS,
                secs=round(time.time() - t0, 1), tokens=ntok)
    write_json(os.path.join(DATA, "run.json"), info)


# --- analysis -----------------------------------------------------------------------------

def conc(x, g, w=None):
    """Share of comparisons whose sign of x agrees with the sign of g (ties 1/2)."""
    x, g = np.asarray(x, float), np.asarray(g, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    s = np.sign(x) * np.sign(g)
    return float(np.sum(w * ((s > 0) + 0.5 * (s == 0))) / np.sum(w)) if np.sum(w) > 0 else float("nan")


def wilson(q, n, z=1.96):
    if n == 0:
        return None
    den = 1 + z * z / n
    c = (q + z * z / (2 * n)) / den
    h = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def r_equiv(q):
    return float(math.sin(math.pi * (q - 0.5)))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def block_boot(P, x_fn, boots=BOOTS, seed=0):
    """Item-block bootstrap of q for a signed score per comparison: resample each
    parent's pool items, weight a comparison by the product of its items'
    multiplicities. x_fn(sel) -> score vector over P."""
    rng = np.random.default_rng(seed)
    pools = {}
    for p in P:
        pools.setdefault(p["bench"], set()).update((p["i"], p["j"]))
    pools = {b: sorted(v) for b, v in pools.items()}
    x = x_fn()
    g = np.array([p["b_i"] - p["b_j"] for p in P])
    out = []
    for _ in range(boots):
        mult = {}
        for b, items in pools.items():
            for it in rng.choice(len(items), len(items)):
                mult[(b, items[it])] = mult.get((b, items[it]), 0) + 1
        w = np.array([mult.get((p["bench"], p["i"]), 0) * mult.get((p["bench"], p["j"]), 0) for p in P], float)
        if w.sum() > 0:
            out.append(conc(x, g, w))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def sticky(P, h, g, perms=PERMS, seed=0):
    """(a) residual correlation between comparisons sharing an item; (b) the
    variance across items of their wrong-comparison counts against permutations
    within parent and |gap| tercile."""
    bench = np.array([p["bench"] for p in P], object)
    res = np.zeros_like(h)
    for b in sorted(set(bench)):
        m = bench == b
        A = np.column_stack([g[m], np.ones(m.sum())])
        res[m] = h[m] - A @ np.linalg.lstsq(A, h[m], rcond=None)[0]
    inc = {}
    for n, p in enumerate(P):
        inc.setdefault((p["bench"], p["i"]), []).append((n, 1.0))
        inc.setdefault((p["bench"], p["j"]), []).append((n, -1.0))
    xs, ys = [], []
    for lst in inc.values():
        for u in range(len(lst)):
            for v in range(u + 1, len(lst)):
                xs.append(lst[u][1] * res[lst[u][0]])
                ys.append(lst[v][1] * res[lst[v][0]])
    rho = pearson(xs, ys)
    wrong = (np.sign(h) * np.sign(g) < 0).astype(float)
    t = np.quantile(np.abs(g), [1 / 3, 2 / 3])
    terc = np.digitize(np.abs(g), t)
    strata = [np.flatnonzero((bench == b) & (terc == k)) for b in sorted(set(bench)) for k in range(3)]
    keys = list(inc)
    memb = [np.array([n for n, _ in inc[k]]) for k in keys]

    def var_counts(wv):
        return float(np.var([wv[m].sum() for m in memb]))

    obs = var_counts(wrong)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(perms):
        wv = wrong.copy()
        for s in strata:
            wv[s] = wv[rng.permutation(s)]
        null.append(var_counts(wv))
    null = np.array(null)
    counts = np.array([wrong[m].sum() for m in memb])
    return dict(shared_item_pairs=len(xs), resid_corr=round(rho, 4), persistent_share=round(min(1.0, 2 * rho), 4)
                if rho == rho else None,
                wrong_count_var=round(obs, 4), wrong_count_var_null=round(float(null.mean()), 4),
                var_ratio=round(obs / float(null.mean()), 3) if null.mean() > 0 else None,
                p_perm=round(float((null >= obs).mean()), 4),
                wrong_count_hist={int(c): int((counts == c).sum()) for c in np.unique(counts)},
                items=len(keys))


def bt_scores(P):
    """Least-squares Bradley-Terry scores per parent from h (t_i - t_j), with
    their Pearson against honest b over the pool."""
    out = {}
    for b in PARENTS:
        Pb = [p for p in P if p["bench"] == b]
        its = sorted({p["i"] for p in Pb} | {p["j"] for p in Pb})
        ix = {k: n for n, k in enumerate(its)}
        A = np.zeros((len(Pb) + 1, len(its)))
        y = np.zeros(len(Pb) + 1)
        for n, p in enumerate(Pb):
            A[n, ix[p["i"]]], A[n, ix[p["j"]]], y[n] = 1, -1, p["h"]
        A[-1] = 1
        t = np.linalg.lstsq(A, y, rcond=None)[0]
        bb = np.array([next(p["b_i"] if p["i"] == k else p["b_j"] for p in Pb if k in (p["i"], p["j"])) for k in its])
        out[b] = dict(items=len(its), pearson=round(pearson(t, bb), 4))
    return out


def stage_analyse(args):
    from experiments import llm4b_close as L4
    from paiec import itemcov as IC
    plan = read_json(PAIRS)
    rows = {}
    with open(RAW) as f:
        for r in map(json.loads, f):
            rows.setdefault(r["k"], {})[r["order"]] = r
    P = []
    for k, p in enumerate(plan["pairs"]):
        if k in rows and len(rows[k]) == 2:
            s0 = rows[k][0]["lA"] - rows[k][0]["lB"]          # > 0: i (shown as A) harder
            s1 = rows[k][1]["lA"] - rows[k][1]["lB"]          # > 0: j (shown as A) harder
            P.append(dict(p, k=k, s0=s0, s1=s1, h=(s0 - s1) / 2, bias=(s0 + s1) / 2,
                          mass=(rows[k][0]["mass"] + rows[k][1]["mass"]) / 2))
    if not P:
        raise SystemExit("no complete comparisons yet")
    g = np.array([p["b_i"] - p["b_j"] for p in P])
    h = np.array([p["h"] for p in P])
    bench = np.array([p["bench"] for p in P], object)
    items = load_items(PARENTS)
    oof = read_json(os.path.join(PROBE, "oof.json"))
    judge = {b: L4.load_features(b)[0] for b in PARENTS}
    # absolute signals, + = harder
    sig = {"log_length": lambda b, i: IC.log_length(items[b][i])}
    for hd in HEADS:
        sig[f"head_{hd}"] = (lambda hd_: lambda b, i: oof[hd_].get(i))(hd)
    for f, sgn in JUDGE.items():
        sig[f"judge_{f}"] = (lambda f_, s_: lambda b, i: (None if judge[b][f_].get(i) is None
                                                          else s_ * judge[b][f_][i]))(f, sgn)

    def signal_diff(name):
        v = []
        for p in P:
            a, c = sig[name](p["bench"], p["i"]), sig[name](p["bench"], p["j"])
            v.append(np.nan if a is None or c is None else a - c)
        return np.array(v, float)

    t = np.quantile(np.abs(g), [1 / 3, 2 / 3])
    terc = np.digitize(np.abs(g), t)
    same = np.array([p["g_i"] == p["g_j"] and p["g_i"] != "" for p in P])

    def block(mask):
        n = int(mask.sum())
        q = conc(h[mask], g[mask])
        return dict(n=n, q=round(q, 4), wilson=wilson(q, n), r_equiv=round(r_equiv(q), 4),
                    q_order0=round(conc(np.array([p["s0"] for p in P])[mask], g[mask]), 4),
                    q_order1=round(conc(-np.array([p["s1"] for p in P])[mask], g[mask]), 4),
                    position_consistency=round(float(np.mean([np.sign(p["s0"]) == -np.sign(p["s1"])
                                                              for p, m in zip(P, mask) if m])), 4),
                    mean_bias=round(float(np.mean([p["bias"] for p, m in zip(P, mask) if m])), 4),
                    share_first=round(float(np.mean([(p["s0"] > 0) + (p["s1"] > 0) for p, m in zip(P, mask)
                                                     if m]) / 2), 4),
                    mean_ab_mass=round(float(np.mean([p["mass"] for p, m in zip(P, mask) if m])), 4))

    out = {"pooled": block(np.ones(len(P), bool))}
    out["pooled"]["block_boot_ci"] = block_boot(P, lambda: h)
    out["pooled"]["q_by_gap_tercile"] = [round(conc(h[terc == k], g[terc == k]), 4) for k in range(3)]
    out["pooled"]["gap_tercile_edges"] = [round(float(x), 3) for x in t]
    out["pooled"]["within_group"] = dict(n=int(same.sum()), q=round(conc(h[same], g[same]), 4) if same.any() else None)
    out["pooled"]["across_group"] = dict(n=int((~same).sum()), q=round(conc(h[~same], g[~same]), 4))
    for b in PARENTS:
        out[b] = block(bench == b)
    absq = {}
    for name in sig:
        d = signal_diff(name)
        ok = np.isfinite(d)
        if ok.sum() < 20:
            continue
        e = dict(n=int(ok.sum()), q=round(conc(d[ok], g[ok]), 4), q_compare_same=round(conc(h[ok], g[ok]), 4),
                 per_parent={b: round(conc(d[ok & (bench == b)], g[ok & (bench == b)]), 4)
                             for b in PARENTS if (ok & (bench == b)).sum() >= 10})
        e["r_equiv"] = round(r_equiv(e["q"]), 4)
        absq[name] = e
    full = {k: v for k, v in absq.items() if v["n"] == len(P)}
    best = max(full, key=lambda k: full[k]["q"])
    q = out["pooled"]["q"]
    stick = sticky(P, h, g)
    bt = bt_scores(P)
    drop = q < Q_DROP or q < full[best]["q"] + MARGIN
    verdict = dict(rule=f"DROP if pooled q < {Q_DROP} or q < best absolute q on the same pairs + {MARGIN}",
                   q=q, best_absolute=best, best_absolute_q=full[best]["q"], call="DROP" if drop else "KEEP")
    import hashlib
    with open(os.path.abspath(__file__), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    res = dict(meta=dict(script="experiments/pairwise_probe.py", script_digest=digest, pool=POOL, offsets=list(OFFSETS), seed=SEED,
                         item_tokens=ITEM_TOKENS, n_comparisons=len(P), run=read_json(os.path.join(DATA, "run.json")),
                         target="honest difficulty: Rasch b without each of five subject folds, averaged "
                                "(llm4b_close.honest_targets)"),
               verdict=verdict, compare=out, absolute=absq, stickiness=stick, bradley_terry=bt)
    write_json(OUT, res)
    show(res)


def show(res):
    c = res["compare"]
    print("\n| slice | n | q [Wilson] | r equiv | q order A=i / A=j | position consistency | A/B mass |")
    print("|---|---|---|---|---|---|---|")
    for k in ("pooled",) + PARENTS:
        v = c[k]
        print(f"| {k} | {v['n']} | {v['q']:.3f} {v['wilson']} | {v['r_equiv']:+.3f} | {v['q_order0']:.3f} / "
              f"{v['q_order1']:.3f} | {v['position_consistency']:.3f} | {v['mean_ab_mass']:.3f} |")
    print("pooled block-bootstrap CI", c["pooled"]["block_boot_ci"], "by gap tercile", c["pooled"]["q_by_gap_tercile"],
          "within group", c["pooled"]["within_group"], "across", c["pooled"]["across_group"])
    print("\n| absolute signal | n | q | comparator q on the same | r equiv | per parent |")
    print("|---|---|---|---|---|---|")
    for k, v in sorted(res["absolute"].items(), key=lambda kv: -kv[1]["q"]):
        print(f"| {k} | {v['n']} | {v['q']:.3f} | {v['q_compare_same']:.3f} | {v['r_equiv']:+.3f} | "
              + ", ".join(f"{p[:5]} {x:.2f}" for p, x in v["per_parent"].items()) + " |")
    print("stickiness", json.dumps(res["stickiness"]))
    print("Bradley-Terry", json.dumps(res["bradley_terry"]))
    print("verdict", json.dumps(res["verdict"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["plan", "run", "analyse", "show"])
    ap.add_argument("--force", action="store_true", help="run beside another language-model process")
    args = ap.parse_args()
    if args.stage == "plan":
        stage_plan(args)
    elif args.stage == "run":
        stage_run(args)
    elif args.stage == "analyse":
        stage_analyse(args)
    else:
        res = read_json(OUT)
        if res is None:
            raise SystemExit("run --stage analyse first")
        show(res)


if __name__ == "__main__":
    main()
