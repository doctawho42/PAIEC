"""What an item dict alone says about the item's difficulty inside its benchmark.

Read-only analysis for step 2. Every number in the report comes from this script:

    python item_signal.py > item_signal_out.txt   # ~10 min; writes item_signal.json next to it

Sections
  1. item_features parsing and inventory (keys, cardinality, coverage)
  2. variance components: a Rasch model with subject ability, a benchmark level,
     one random effect per feature key (sigma_g) and an item residual (sigma_d),
     fitted by Laplace-EM on every public binary response
  3. the same signal as a predictor would get it: group effects estimated from
     k labels per group (k = 1..5), and at the official budgets
  4. cheap text statistics, within benchmark and transferred across benchmarks
  5. twins (identical item_content)
  6. how often, in formative-like official runs, a target's group already has labels
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # tiny solves: BLAS threads cost 100x the work
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from paiec import official as O  # noqa: E402
from paiec.data import load_pairs  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "item_signal.json")
BENCH = ["matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
PI8 = np.pi / 8
sig = lambda x: 1 / (1 + np.exp(-x))


# --- 1. parsing ----------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d+(\.\d+)?")


def _split_top(s, seps=";\n"):
    """Split on separators outside brackets and quotes: matharena has values
    like image_detail=["high"], and a JSON list may one day hold a ';'."""
    out, depth, quote, cur = [], 0, None, []
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth = max(0, depth - 1)
        elif ch in seps and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [t.strip() for t in out if t.strip()]


def parse_features(s):
    """Benchmark-agnostic: 'k=v;k=v' (the only format in the public data), a JSON
    object, 'k: v' lines, bare flags, or one opaque value under the key '_'."""
    s = (s or "").strip()
    if not s:
        return {}
    if s[0] == "{":
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {str(k): v if isinstance(v, str) else json.dumps(v, sort_keys=True)
                        for k, v in obj.items()}
        except ValueError:
            pass
    toks = _split_top(s)
    if not any("=" in t or ":" in t for t in toks):
        return {"_": s} if len(toks) == 1 else {t: "1" for t in toks}
    out = {}
    for t in toks:
        if "=" in t:
            k, v = t.split("=", 1)
        elif ":" in t:
            k, v = t.split(":", 1)
        else:
            k, v = t, "1"
        out[k.strip()] = v.strip()
    return out


def is_num(v):
    return bool(_NUM.fullmatch(v.strip()))


# --- data ------------------------------------------------------------------------

def benchmark_tables(pairs):
    """Per benchmark: cells (subject, item, n, k), item dicts, subject ids."""
    out = {}
    by_b = defaultdict(list)
    for p in pairs:
        by_b[p.benchmark_id].append(p)
    for b, ps in by_b.items():
        items, iidx, cells = [], {}, defaultdict(lambda: [0, 0])
        subj = [p.subject_id for p in ps]
        elig = []
        for s, p in enumerate(ps):
            if len(p.item_keys) >= 80:
                elig.append(s)
            for r in p.responses:
                if r.item_key not in iidx:
                    iidx[r.item_key] = len(items)
                    items.append(r.item)
                c = cells[(s, iidx[r.item_key])]
                c[0] += 1
                c[1] += r.label
        si, ji = (np.array(x) for x in zip(*cells.keys()))
        n, k = (np.array(x, float) for x in zip(*cells.values()))
        out[b] = dict(si=si, ji=ji, n=n, k=k, items=items, subjects=subj,
                      keys=list(iidx.keys()), eligible=elig)
    return out


def inventory(items):
    feats = [parse_features(it["item_features"]) for it in items]
    raw = [it["item_features"] for it in items]
    ok = sum(";".join(f"{k}={v}" for k, v in f.items()) == r for f, r in zip(feats, raw) if r)
    keys = Counter(k for f in feats for k in f)
    N = len(items)
    rows = {}
    for key, c in keys.items():
        vals = [f.get(key, "<missing>") for f in feats]
        cnt = Counter(vals)
        sizes = sorted(cnt.values())
        rows[key] = dict(coverage=c / N, cardinality=len(cnt), numeric=np.mean([is_num(v) for v in vals if v != "<missing>"]),
                         size_min=sizes[0], size_median=float(np.median(sizes)), size_max=sizes[-1],
                         examples=[v for v, _ in cnt.most_common(4)],
                         selected=bool(2 <= len(cnt) <= N / 3))
    return feats, dict(n_items=N, nonempty=sum(bool(r) for r in raw), roundtrip_ok=ok,
                       interactors_nonempty=sum(bool(it["interactors"]) for it in items),
                       keys=rows)


def codes(values):
    lv = {}
    return np.array([lv.setdefault(v, len(lv)) for v in values]), len(lv)


# --- 2. variance components ----------------------------------------------------------

def fit_glmm(t, groups, iters=40, lam_t=0.1, s_init=0.5):
    """logit P(y=1) = theta_s - mu - sum_k g_k[level_k(i)] - d_i, Gaussian priors
    g_k ~ N(0, s2_k), d ~ N(0, s2_d), variances by Laplace-EM. `groups` is a list
    of (codes per item, n_levels). Returns the MAP and the variances."""
    si, ji, n, k = t["si"], t["ji"], t["n"], t["k"]
    n_s, n_i = int(si.max()) + 1, len(t["items"])
    L = [nl for _, nl in groups]
    off = np.cumsum([n_s + 1] + L)
    dim = off[-1] + n_i
    s2 = [s_init] * len(groups)
    s2d = s_init
    x = np.zeros(dim)
    gc = [c for c, _ in groups]

    def unpack(p):
        th, mu = p[:n_s], p[n_s]
        gs = [p[off[j]:off[j + 1]] for j in range(len(groups))]
        d = p[off[-1]:]
        return th, mu, gs, d

    def obj(p):
        th, mu, gs, d = unpack(p)
        bi = mu + d + sum(g[c] for g, c in zip(gs, gc)) if gs else mu + d
        eta = th[si] - bi[ji]
        nll = np.sum(n * np.logaddexp(0, eta) - k * eta)
        r = k - n * sig(eta)                   # d nll / d eta = -r
        gb = np.bincount(ji, weights=r, minlength=n_i)   # d nll / d b_i
        grad = np.zeros(dim)
        grad[:n_s] = -np.bincount(si, weights=r, minlength=n_s) + lam_t * th
        grad[n_s] = gb.sum()
        pen = 0.5 * lam_t * th @ th + 0.5 * d @ d / s2d
        for j, (g, c) in enumerate(zip(gs, gc)):
            grad[off[j]:off[j + 1]] = np.bincount(c, weights=gb, minlength=L[j]) + g / s2[j]
            pen += 0.5 * g @ g / s2[j]
        grad[off[-1]:] = gb + d / s2d
        return nll + pen, grad

    hist = []
    for it in range(iters):
        x = minimize(obj, x, jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x
        th, mu, gs, d = unpack(x)
        bi = mu + d + sum(g[c] for g, c in zip(gs, gc)) if gs else mu + d
        eta = th[si] - bi[ji]
        info = np.bincount(ji, weights=n * sig(eta) * (1 - sig(eta)), minlength=n_i)
        vd = 1 / (info + 1 / s2d)
        new = []
        for j, (g, c) in enumerate(zip(gs, gc)):
            im = info / (1 + info * s2d)
            vg = 1 / (np.bincount(c, weights=im, minlength=L[j]) + 1 / s2[j])
            new.append(max(np.mean(g ** 2 + vg), 1e-4))
        s2d_new = max(np.mean(d ** 2 + vd), 1e-4)
        done = abs(s2d_new - s2d) < 1e-4 and all(abs(a - b) < 1e-4 for a, b in zip(new, s2))
        s2, s2d = new, s2d_new
        hist.append((s2d, list(s2)))
        if done:
            break
    th, mu, gs, d = unpack(x)
    return dict(theta=th, mu=float(mu), g=gs, d=d, s2=s2, s2d=s2d, vd=vd, info=info, iters=it + 1)


def anova(y, c, nl):
    """Descriptive between-group share of y: eta^2 and the small-group-corrected
    epsilon^2 (the adjusted R^2 of a one-way ANOVA)."""
    N = len(y)
    m = np.bincount(c, weights=y, minlength=nl) / np.maximum(np.bincount(c, minlength=nl), 1)
    sst = np.sum((y - y.mean()) ** 2)
    ssw = np.sum((y - m[c]) ** 2)
    eta2 = 1 - ssw / sst
    G = len(np.unique(c))
    eps2 = 1 - (ssw / max(N - G, 1)) / (sst / (N - 1))
    return float(eta2), float(eps2)


# --- 3. run-time use ----------------------------------------------------------------

def map_ag(obs, n_a, prior_a, s2g, c, n_g, fix_a=None):
    """Joint MAP of subject levels a and group effects g from single binary
    labels with logit (a_s - sum g)/c. obs: list of (subject slot, [group
    indices], y). Newton on a small dense system. Returns mean, covariance."""
    dim = n_a + n_g
    x = np.zeros(dim)
    ma, va = prior_a
    if fix_a is not None:
        x[:n_a] = fix_a
    else:
        x[:n_a] = ma
    prec = np.concatenate([np.full(n_a, 1 / va if fix_a is None else 1e8), 1 / s2g])
    mean0 = np.concatenate([np.full(n_a, ma) if fix_a is None else fix_a, np.zeros(n_g)])
    A = np.zeros((len(obs), dim))
    y = np.zeros(len(obs))
    for r, (s, gi, yy) in enumerate(obs):
        A[r, s] = 1
        for g in gi:
            A[r, n_a + g] -= 1
        y[r] = yy
    A /= c
    for _ in range(20):
        p = sig(A @ x)
        grad = A.T @ (y - p) - prec * (x - mean0)
        H = (A.T * (p * (1 - p))) @ A + np.diag(prec)
        step = np.linalg.solve(H, grad)
        x += step
        if np.max(np.abs(step)) < 1e-6:
            break
    p = sig(A @ x)
    H = (A.T * (p * (1 - p))) @ A + np.diag(prec)
    return x, np.linalg.inv(H)


def brier_cells(p, n, k):
    return float(np.sum(k * (1 - p) ** 2 + (n - k) * p ** 2)), float(np.sum(n))


def per_item_cells(t):
    by = defaultdict(list)
    for r, j in enumerate(t["ji"]):
        by[j].append(r)
    return {j: np.array(v) for j, v in by.items()}


def one_label(t, rows, rng):
    """One response drawn from an item's recorded responses: a random cell
    weighted by its responses, then a random response of that cell."""
    n, k = t["n"][rows], t["k"][rows]
    r = rows[rng.choice(len(rows), p=n / n.sum())]
    return int(t["si"][r]), int(rng.random() < t["k"][r] / t["n"][r])


def sim_k(t, base, g_codes, nl, s2g, s2d_tot, ks=(1, 2, 3, 4, 5), reps=40, seed=0):
    """Group effect of a target item estimated from k other items of its group.
    'oracle': the k items' difficulties are known (EB, no-group fit).
    'labels': each of the k items carries one binary label from a random
    subject whose level theta - mu is known. Evaluated on every response of the
    held-out items. Also 'all': the leave-one-out group mean of every other item."""
    rng = np.random.default_rng(seed)
    d = base["d"]
    a = base["theta"] - base["mu"]
    s2d = max(s2d_tot - s2g, 1e-3)
    cells = per_item_cells(t)
    members = defaultdict(list)
    for j, c in enumerate(g_codes):
        members[c].append(j)
    c_d = np.sqrt(1 + PI8 * s2d)
    c0 = np.sqrt(1 + PI8 * s2d_tot)
    out = {}
    # leave-one-out, all other items, shrunk
    sse = sst = 0.0
    for c, js in members.items():
        js = np.array(js)
        if len(js) < 2:
            continue
        tot = d[js].sum()
        for j in js:
            m = (tot - d[j]) / (len(js) - 1)
            gh = s2g / (s2g + s2d / (len(js) - 1)) * m
            sse += (d[j] - gh) ** 2
            sst += d[j] ** 2
    out["all"] = dict(r2=float(1 - sse / sst))
    for kk in ks:
        acc = dict(sse_o=0.0, sse_l=0.0, sst=0.0, b0=0.0, bo=0.0, bl=0.0, nr=0.0, groups=0)
        for _ in range(reps):
            for c, js in members.items():
                if len(js) < kk + 1:
                    continue
                js = np.array(js)
                pick = rng.choice(len(js), kk, replace=False)
                lab, rest = js[pick], np.delete(js, pick)
                acc["groups"] += 1
                # oracle difficulties
                w = s2g / (s2g + s2d / kk)
                go, vo = w * d[lab].mean(), s2g * (1 - w)
                # binary labels
                obs = []
                for j in lab:
                    s, y = one_label(t, cells[j], rng)
                    obs.append((0, [0], y, a[s]))
                A = np.array([o[3] for o in obs])
                yv = np.array([o[2] for o in obs], float)
                g = 0.0
                for _ in range(25):
                    p = sig((A - g) / c_d)
                    grad = np.sum(yv - p) / c_d + g / s2g   # d(-log posterior)/dg
                    h = np.sum(p * (1 - p)) / c_d ** 2 + 1 / s2g
                    g -= grad / h
                    if abs(grad / h) < 1e-9:
                        break
                p = sig((A - g) / c_d)
                vl = 1 / (np.sum(p * (1 - p)) / c_d ** 2 + 1 / s2g)
                acc["sse_o"] += np.sum((d[rest] - go) ** 2)
                acc["sse_l"] += np.sum((d[rest] - g) ** 2)
                acc["sst"] += np.sum(d[rest] ** 2)
                rows = np.concatenate([cells[j] for j in rest])
                lat = a[t["si"][rows]]
                nn, kk_ = t["n"][rows], t["k"][rows]
                for key, p in (("b0", sig(lat / c0)),
                               ("bo", sig((lat - go) / np.sqrt(1 + PI8 * (s2d + vo)))),
                               ("bl", sig((lat - g) / np.sqrt(1 + PI8 * (s2d + vl))))):
                    acc[key] += brier_cells(p, nn, kk_)[0]
                acc["nr"] += nn.sum()
        if acc["groups"] == 0:
            continue
        out[kk] = dict(r2_oracle=float(1 - acc["sse_o"] / acc["sst"]),
                       r2_labels=float(1 - acc["sse_l"] / acc["sst"]),
                       brier_base=acc["b0"] / acc["nr"],
                       gain_oracle=(acc["b0"] - acc["bo"]) / acc["nr"],
                       gain_labels=(acc["b0"] - acc["bl"]) / acc["nr"],
                       groups_per_rep=acc["groups"] / reps)
    return out


def sim_budget(t, base, key_codes, s2gs, s2d_tot, m_pairs=1, draws=1000, cut=100, seed=0,
               level="known", scope="benchmark"):
    """A formative-like pair: the target subject's items cut to `cut`, split
    50/50; labels are the first B acquisition items of the target and of
    m_pairs-1 companion subjects on the same benchmark. Brier on the target's
    evaluation responses with and without group effects (keys additive).
    level='known': theta_s - mu known. 'unknown': subject levels estimated from
    the same labels with a N(mean, var) prior over this benchmark's subjects,
    identical for both models, so the difference is what groups add.
    scope='benchmark': one 50/50 split of the benchmark's items for every
    subject, so no label ever sits on a target's evaluation item and the gain
    is what the item dict carries. 'pair': each subject split on its own, as
    official.split(scope='pair'), so a companion's label can be the target
    item's own."""
    rng = np.random.default_rng(seed)
    a_true = base["theta"] - base["mu"]
    elig = t["eligible"]
    prior_a = (float(np.mean(a_true[elig])), float(np.var(a_true[elig])))
    s2g_tot = float(sum(s2gs))
    s2d = max(s2d_tot - s2g_tot, 1e-3)
    c_d, c0 = np.sqrt(1 + PI8 * s2d), np.sqrt(1 + PI8 * s2d_tot)
    by_s = defaultdict(list)
    for r, (s, j) in enumerate(zip(t["si"], t["ji"])):
        by_s[s].append(r)
    offs = np.cumsum([0] + [int(c.max()) + 1 for c in key_codes])
    s2vec = np.concatenate([np.full(int(c.max()) + 1, s2) for c, s2 in zip(key_codes, s2gs)])
    res = {B: [0.0, 0.0, 0.0] for B in BUDGETS}
    gains = {B: [] for B in BUDGETS}
    for _ in range(draws):
        subs = list(rng.choice(elig, m_pairs, replace=False))
        acq, ev = {}, None
        side = rng.random(len(t["items"])) < 0.5          # benchmark scope: True = acquisition
        for q, s in enumerate(subs):
            rows = np.array(by_s[s])
            rows = rows[rng.permutation(len(rows))][:cut]
            if scope == "benchmark":
                on = side[t["ji"][rows]]
                acq[q] = rows[on]
                if q == 0:
                    ev = rows[~on]
            else:
                h = len(rows) // 2
                acq[q] = rows[:h]
                if q == 0:
                    ev = rows[h:]
        evj = t["ji"][ev]
        ev_groups = [[int(offs[kx] + key_codes[kx][j]) for kx in range(len(key_codes))] for j in evj]
        nn, kk_ = t["n"][ev], t["k"][ev]
        labels = {q: [(int(t["ji"][r]), int(rng.random() < t["k"][r] / t["n"][r])) for r in acq[q][:31]]
                  for q in range(m_pairs)}
        fix = np.array([a_true[s] for s in subs]) if level == "known" else None
        for B in BUDGETS:
            raw = [(q, [int(offs[kx] + key_codes[kx][j]) for kx in range(len(key_codes))], y)
                   for q in range(m_pairs) for j, y in labels[q][:B]]
            # only groups holding labels get a free parameter; the rest stay at the prior
            present = sorted({g for _, gi, _ in raw for g in gi})
            loc = {g: i for i, g in enumerate(present)}
            obs = [(q, [loc[g] for g in gi], y) for q, gi, y in raw]
            if obs:
                xg, Sg = map_ag(obs, m_pairs, prior_a, s2vec[present], c_d, len(present), fix_a=fix)
                x0, S0 = map_ag([(q, [], y) for q, _, y in obs], m_pairs, prior_a,
                                np.zeros(0), c0, 0, fix_a=fix)
            else:
                a0 = fix if fix is not None else np.full(m_pairs, prior_a[0])
                va0 = np.zeros(m_pairs) if fix is not None else np.full(m_pairs, prior_a[1])
                xg, Sg = a0.copy(), np.diag(va0)
                x0, S0 = a0.copy(), np.diag(va0)
            # target subject is slot 0; groups without labels add their prior variance
            U = np.zeros((len(ev), m_pairs + len(present)))
            U[:, 0] = 1
            extra = np.zeros(len(ev))
            for e, gi in enumerate(ev_groups):
                for g in gi:
                    if g in loc:
                        U[e, m_pairs + loc[g]] -= 1
                    else:
                        extra[e] += s2vec[g]
            mg = U @ xg
            vg = np.einsum("ij,jk,ik->i", U, Sg, U) + extra
            pg = sig(mg / np.sqrt(1 + PI8 * (s2d + vg)))
            p0 = np.full(len(ev), sig(x0[0] / np.sqrt(1 + PI8 * (s2d_tot + S0[0, 0]))))
            b0, bg = brier_cells(p0, nn, kk_)[0] / nn.sum(), brier_cells(pg, nn, kk_)[0] / nn.sum()
            res[B][0] += b0
            res[B][1] += bg
            res[B][2] += 1
            gains[B].append(b0 - bg)
    out = {B: dict(base=v[0] / v[2], group=v[1] / v[2], gain=(v[0] - v[1]) / v[2],
                   se=float(np.std(gains[B]) / np.sqrt(len(gains[B])))) for B, v in res.items()}
    per_draw = sum(w * np.array(gains[B]) for w, B in zip(WEIGHTS, BUDGETS))
    out["ALC_gain"] = float(per_draw.mean())
    out["ALC_se"] = float(per_draw.std() / np.sqrt(len(per_draw)))
    return out


# --- 4. text -------------------------------------------------------------------------

_MC_OPT = re.compile(r"(?m)^\s*\(?([A-E])[\)\.:]\s|\(([A-E])\)")
_MC_WORD = re.compile(r"multiple[- ]choice|answer choices", re.I)
TEXT_FEATS = ["log_chars", "log_lines", "code_fence", "digit_share", "mc_pattern",
              "math_marks", "questions", "url"]


def text_stats(c):
    return [np.log1p(len(c)), np.log1p(c.count("\n")), float("```" in c),
            sum(ch.isdigit() for ch in c) / (len(c) + 1),
            float(len({a or b for a, b in _MC_OPT.findall(c)}) >= 3 or bool(_MC_WORD.search(c))),
            np.log1p(c.count("$") + c.count("\\(")), min(c.count("?"), 10), float("http" in c)]


# --- 6. coverage in official runs ------------------------------------------------------

def coverage(pairs, runs=150, seed=0, main_keys=None, scope="pair"):
    """Capture every checkpoint's shared `labeled` and, per evaluated input,
    how many labels share its benchmark and its value of each key, and whether
    an identical item_content is labeled."""
    rng = np.random.default_rng(seed)
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))
    names = {}
    state = {"calls": 0}

    def factory():
        B = BUDGETS[state["calls"] % 6]
        state["calls"] += 1
        cache = {}

        def predict(inp, labeled):
            s, it = inp
            bid = it["benchmark_id"]
            if "idx" not in cache:
                idx = defaultdict(Counter)
                for (ls, li), y in labeled:
                    f = parse_features(li["item_features"])
                    for kk, v in f.items():
                        idx[li["benchmark_id"]][(kk, v)] += 1
                    idx[li["benchmark_id"]][("__content", li["item_content"])] += 1
                    idx[li["benchmark_id"]][("__feat", li["item_features"])] += 1
                    idx[li["benchmark_id"]][("__featcontent", li["item_features"] + "\x1f" + li["item_content"])] += 1
                    idx[li["benchmark_id"]][("__all", "")] += 1
                cache["idx"] = idx
            idx = cache["idx"][bid]
            f = parse_features(it["item_features"])
            bname = names.get(bid, bid)
            for kk in main_keys.get(bname, []):
                n_same = idx[(kk, f.get(kk))] if kk in f else 0
                st = stats[(bname, kk)][B]
                st[0] += 1
                st[1] += n_same >= 1
                st[2] += n_same >= 3
                st[3] += n_same
            if it["item_features"]:
                # same full feature string, different content: another version of the problem
                sib = idx[("__feat", it["item_features"])] - \
                    idx[("__featcontent", it["item_features"] + "\x1f" + it["item_content"])]
                st = stats[(bname, "__sibling")][B]
                st[0] += 1
                st[1] += sib >= 1
                st[2] += sib >= 3
                st[3] += sib
            st = stats[(bname, "__same_content")][B]
            st[0] += 1
            st[1] += idx[("__content", it["item_content"])] >= 1
            st[3] += idx[("__all", "")]
            return 0.5
        return predict

    for r in range(runs):
        run = O.sample_run(pairs, rng)
        for p, _ in run:
            from paiec.data import anon_id
            names[anon_id("benchmark", p.benchmark_id)] = p.benchmark_id
        O.run_official(run, factory, deepcopy=False, split_scope=scope)
    out = {}
    for (b, kk), by in stats.items():
        out[f"{b}:{kk}"] = {B: dict(n=v[0], any=v[1] / v[0], ge3=v[2] / v[0], mean=v[3] / v[0])
                            for B, v in sorted(by.items())}
    return out


# --- main --------------------------------------------------------------------------------

def main():
    t0 = time.time()
    allp = O.eligible(load_pairs(min_items=1), min_items=1)
    tabs = benchmark_tables(allp)
    R = {"inventory": {}, "glmm": {}, "anova": {}, "sim_k": {}, "sim_budget": {},
         "ordinal": {}, "text": {}, "twins": {}, "content_keys": {}}
    feats, base, glm = {}, {}, {}
    for b in BENCH:
        t = tabs[b]
        feats[b], R["inventory"][b] = inventory(t["items"])
        print(f"[{time.time()-t0:5.0f}s] {b}: {json.dumps(R['inventory'][b], default=str)[:1500]}")
    # variance components
    for b in BENCH:
        t = tabs[b]
        base[b] = fit_glmm(t, [])
        R["glmm"][b] = {"none": dict(s2d=base[b]["s2d"], mu=base[b]["mu"], iters=base[b]["iters"],
                                     theta_sd=float(np.std(base[b]["theta"])),
                                     reliability=float(np.var(base[b]["d"]) / base[b]["s2d"]),
                                     mean_info=float(np.mean(base[b]["info"])))}
        sel = [k for k, v in R["inventory"][b]["keys"].items() if v["selected"]]
        glm[b] = {}
        R["anova"][b] = {}
        variants = [(k, [k]) for k in sel]
        if len(sel) > 1:
            variants.append(("+".join(sel), sel))
            variants.append(("full_string", ["__full"]))
            variants.append((sel[0] + "+full_string", [sel[0], "__full"]))
        for name, ks in variants:
            groups = []
            for k in ks:
                vals = ([t["items"][j]["item_features"] for j in range(len(t["items"]))] if k == "__full"
                        else [feats[b][j].get(k, "<missing>") for j in range(len(t["items"]))])
                groups.append(codes(vals))
            fit = fit_glmm(t, groups)
            glm[b][name] = (fit, groups)
            R["glmm"][b][name] = dict(s2g=fit["s2"], s2d=fit["s2d"], iters=fit["iters"],
                                      share=[s / (sum(fit["s2"]) + fit["s2d"]) for s in fit["s2"]],
                                      levels=[nl for _, nl in groups])
            if len(ks) == 1:
                R["anova"][b][name] = dict(zip(["eta2", "eps2"], anova(base[b]["d"], *groups[0])))
            print(f"[{time.time()-t0:5.0f}s] {b} {name}: {R['glmm'][b][name]} {R['anova'][b].get(name)}")
        print(f"[{time.time()-t0:5.0f}s] {b} none: {R['glmm'][b]['none']}")
    # LOBO prior for group variance: the selected single keys of the other benchmarks
    # (as a share of item variance: the grouped fits' totals run 5-15% above the
    # no-group total, so shares are rescaled onto it and both models agree at B0)
    share = lambda fit: [x / (sum(fit["s2"]) + fit["s2d"]) for x in fit["s2"]]
    single = {b: {k: share(glm[b][k][0])[0] for k in glm[b] if "+" not in k and k != "full_string"} for b in BENCH}
    R["share_single_keys"] = single
    # ordinal: problem_idx within competition in matharena
    if "problem_idx" in R["inventory"]["matharena"]["keys"]:
        t, f, d = tabs["matharena"], feats["matharena"], base["matharena"]["d"]
        comp, nl = codes([x.get("competition", "") for x in f])
        idx = np.array([float(x.get("problem_idx", "nan")) for x in f])
        pct = np.zeros(len(idx))
        for c in range(nl):
            m = comp == c
            pct[m] = (np.argsort(np.argsort(idx[m])) + 0.5) / m.sum() if m.sum() > 1 else 0.5
        gm = np.bincount(comp, weights=d) / np.bincount(comp)
        resid = d - gm[comp]
        per = [spearmanr(idx[comp == c], d[comp == c])[0] for c in range(nl) if (comp == c).sum() >= 5]
        slope = np.polyfit(pct - 0.5, resid, 1)
        R["ordinal"]["matharena.problem_idx"] = dict(
            spearman_global=float(spearmanr(idx, d)[0]),
            spearman_within_pooled=float(spearmanr(pct, resid)[0]),
            per_competition_median=float(np.median(per)), per_competition_positive=float(np.mean(np.array(per) > 0)),
            n_competitions=len(per), slope_logit_per_full_range=float(slope[0]),
            within_var_explained=float(1 - np.var(resid - np.polyval(slope, pct - 0.5)) / np.var(resid)))
        print(R["ordinal"])
    # run-time: k labels per group
    for b in BENCH:
        R["sim_k"][b] = {}
        for name, (fit, groups) in glm[b].items():
            if len(groups) != 1:
                continue
            others = [v for bb, kv in single.items() if bb != b for v in kv.values()]
            best = [max(kv.values()) for bb, kv in single.items() if bb != b and kv]
            tot = base[b]["s2d"]
            for prior, s2g in (("own", share(fit)[0] * tot), ("lobo_all_keys", float(np.median(others)) * tot),
                               ("lobo_best_key", float(np.median(best)) * tot)):
                R["sim_k"][b][f"{name}|{prior}"] = sim_k(tabs[b], base[b], groups[0][0], groups[0][1],
                                                         s2g, base[b]["s2d"])
                print(f"[{time.time()-t0:5.0f}s] sim_k {b} {name} {prior}: {R['sim_k'][b][f'{name}|{prior}']}")
    # run-time: official budgets
    for b in BENCH:
        R["sim_budget"][b] = {}
        for name, (fit, groups) in glm[b].items():
            if name == "full_string" or (b == "matharena" and name in ("problem_idx", "image_detail")):
                continue
            for scope, level, m in [(sc, lv, mm) for sc in ("benchmark", "pair")
                                    for lv in ("known", "unknown") for mm in (1, 2)]:
                    out = sim_budget(tabs[b], base[b], [g[0] for g in groups],
                                     [x * base[b]["s2d"] for x in share(fit)], base[b]["s2d"],
                                     m_pairs=m, level=level, scope=scope)
                    R["sim_budget"][b][f"{name}|{scope}|{level}|m{m}"] = out
                    print(f"[{time.time()-t0:5.0f}s] budget {b} {name} {scope} {level} m{m}: "
                          + " ".join(f"B{B}:{out[B]['gain']:+.4f}" for B in BUDGETS)
                          + f" ALC {out['ALC_gain']:+.4f} +- {out['ALC_se']:.4f} (base B31 {out[31]['base']:.4f})")
    # the prior an unseen benchmark would get: sweep sigma_g^2, as a share of the
    # item variance and in absolute logits^2 (level unknown, benchmark scope)
    R["sweep"] = {}
    main_key = {"matharena": "competition", "multi_swebench": "lang",
                "real_webagents": "website", "researchcodebench": "paper"}
    for b, key in main_key.items():
        fit, groups = glm[b][key]
        tot = base[b]["s2d"]
        R["sweep"][b] = {}
        grid = [("share", v, v * tot) for v in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)] + \
               [("abs", v, v) for v in (0.25, 0.5, 1.0, 2.0, 4.0)]
        for kind, v, s2g in grid:
            for m in (1, 2):
                out = sim_budget(tabs[b], base[b], [groups[0][0]], [min(s2g, 0.95 * tot)], tot,
                                 m_pairs=m, level="unknown", scope="benchmark")
                R["sweep"][b][f"{kind}={v}|m{m}"] = out
                print(f"[{time.time()-t0:5.0f}s] sweep {b} {key} {kind}={v} m{m}: "
                      + " ".join(f"B{B}:{out[B]['gain']:+.4f}" for B in BUDGETS)
                      + f" ALC {out['ALC_gain']:+.4f} +- {out['ALC_se']:.4f}")
    # every selected key of matharena at one generic share each
    sel = [k for k in glm["matharena"] if "+" not in k and k != "full_string"]
    codes_m = [glm["matharena"][k][1][0][0] for k in sel]
    for v in (0.05, 0.1, 0.2):
        for m in (1, 2):
            out = sim_budget(tabs["matharena"], base["matharena"], codes_m,
                             [v * base["matharena"]["s2d"]] * len(sel), base["matharena"]["s2d"],
                             m_pairs=m, level="unknown", scope="benchmark")
            R["sweep"]["matharena"][f"all_keys_share={v}|m{m}"] = out
            print(f"[{time.time()-t0:5.0f}s] sweep matharena all keys share={v} m{m}: ALC {out['ALC_gain']:+.4f} +- {out['ALC_se']:.4f}")
    # matharena: a full feature string is one problem (raw_item_id, which predict never sees)
    import pandas as pd
    raw = pd.read_parquet(os.path.join(REPO, "data", "matharena", "items.parquet")).set_index("item_id")
    t = tabs["matharena"]
    rid = [raw.loc[k, "raw_item_id"] for k in t["keys"]]
    by_f, by_r = defaultdict(set), defaultdict(list)
    for j, it in enumerate(t["items"]):
        by_f[it["item_features"]].add(rid[j])
        by_r[rid[j]].append(j)
    multi = [js for js in by_r.values() if len(js) > 1]
    d, vd = base["matharena"]["d"], base["matharena"]["vd"]
    subj = defaultdict(set)
    for s_, j in zip(t["si"], t["ji"]):
        subj[j].add(s_)
    R["problem_identity"] = dict(
        items=len(rid), problems=len(by_r), feature_strings=len(by_f),
        strings_with_several_problems=sum(len(v) > 1 for v in by_f.values()),
        problems_with_several_items=len(multi), items_in_them=sum(map(len, multi)),
        within_problem_var=float(np.mean([np.var(d[js], ddof=1) for js in multi])),
        mean_posterior_var=float(np.mean([np.mean(vd[js]) for js in multi])),
        var_d=float(np.var(d)),
        subject_overlap_between_versions=float(np.mean([len(subj[a] & subj[c]) > 0 for js in multi
                                                         for x, a in enumerate(js) for c in js[x + 1:]
                                                         if t["items"][a]["item_content"] != t["items"][c]["item_content"]])))
    print("problem identity", R["problem_identity"])
    # text statistics
    X, Y, BB, G = [], [], [], []
    for b in BENCH:
        t = tabs[b]
        Xb = np.array([text_stats(it["item_content"]) for it in t["items"]])
        d = base[b]["d"]
        rows = {}
        sel = [k for k in glm[b] if "+" not in k and k != "full_string"]
        gcode = glm[b][sel[0]][1][0][0] if sel else None
        for fi, fname in enumerate(TEXT_FEATS):
            x = Xb[:, fi]
            if np.std(x) == 0:
                rows[fname] = dict(constant=True, mean=float(x.mean()))
                continue
            r = dict(spearman=float(spearmanr(x, d)[0]), share=float(np.mean(x > 0)) if fname in ("code_fence", "mc_pattern", "url") else None)
            if gcode is not None:
                nl = int(gcode.max()) + 1
                cnt = np.bincount(gcode, minlength=nl)
                xr = x - (np.bincount(gcode, weights=x, minlength=nl) / cnt)[gcode]
                dr = d - (np.bincount(gcode, weights=d, minlength=nl) / cnt)[gcode]
                r["spearman_within_group"] = float(spearmanr(xr, dr)[0]) if np.std(xr) > 0 else None
            rows[fname] = r
        R["text"][b] = rows
        sd = np.std(Xb, 0)
        sd[sd == 0] = 1
        X.append((Xb - Xb.mean(0)) / sd)
        Y.append((d - d.mean()) / d.std())
        BB += [b] * len(d)
        print(f"[{time.time()-t0:5.0f}s] text {b}: {rows}")
    # cross-benchmark transfer of a ridge on standardized text stats
    Xall, Yall, BB = np.vstack(X), np.concatenate(Y), np.array(BB)
    R["text_transfer"] = {}
    for b in BENCH:
        tr, te = BB != b, BB == b
        lam = 1.0
        w = np.linalg.solve(Xall[tr].T @ Xall[tr] + lam * np.eye(Xall.shape[1]), Xall[tr].T @ Yall[tr])
        pred = Xall[te] @ w
        # within-benchmark 5-fold CV for comparison
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, te.sum())
        Xb, Yb = Xall[te], Yall[te]
        pcv = np.zeros(te.sum())
        for f_ in range(5):
            a_, b_ = fold != f_, fold == f_
            wf = np.linalg.solve(Xb[a_].T @ Xb[a_] + lam * np.eye(Xb.shape[1]), Xb[a_].T @ Yb[a_])
            pcv[b_] = Xb[b_] @ wf
        R["text_transfer"][b] = dict(spearman_lobo=float(spearmanr(pred, Yall[te])[0]),
                                     r2_lobo_best_scale=float(np.corrcoef(pred, Yall[te])[0, 1] ** 2 * np.sign(np.corrcoef(pred, Yall[te])[0, 1])),
                                     r2_cv_within=float(1 - np.mean((Yb - pcv) ** 2) / np.var(Yb)),
                                     weights=dict(zip(TEXT_FEATS, np.round(w, 3).tolist())))
    print(R["text_transfer"])
    # content-derived keys (feature-free): first line, first 200 chars
    for b in BENCH:
        t, d = tabs[b], base[b]["d"]
        R["content_keys"][b] = {}
        for name, fn in (("first_line", lambda c: c.strip().split("\n", 1)[0][:200]),
                         ("prefix200", lambda c: c[:200]),
                         ("prefix2000", lambda c: c[:2000])):
            c_, nl = codes([fn(it["item_content"]) for it in t["items"]])
            eta2, eps2 = anova(d, c_, nl)
            R["content_keys"][b][name] = dict(levels=nl, eta2=eta2, eps2=eps2,
                                              selected=bool(2 <= nl <= len(d) / 3))
        print(b, R["content_keys"][b])
    # twins: identical item_content, and identical full dict (content + features),
    # which is what the platform itself cannot tell apart
    for b, unit in [(b, u) for b in BENCH for u in ("content", "full_dict")]:
        t, d = tabs[b], base[b]["d"]
        by = defaultdict(list)
        for j, it in enumerate(t["items"]):
            by[it["item_content"] if unit == "content" else
               (it["item_content"], it["item_features"])].append(j)
        tw = [js for js in by.values() if len(js) > 1]
        pairs_ = [(a, c) for js in tw for x, a in enumerate(js) for c in js[x + 1:]]
        full = defaultdict(list)
        for j, it in enumerate(t["items"]):
            full[(it["item_content"], it["item_features"])].append(j)
        rng = np.random.default_rng(0)
        rnd = rng.integers(0, len(d), (5000, 2))
        r = dict(items=len(d), distinct_content=len(by), twin_groups=len(tw),
                 items_in_twins=sum(map(len, tw)), largest=max(map(len, tw)) if tw else 0,
                 distinct_full_dict=len(full),
                 twins_differ_in_features=sum(len({t["items"][j]["item_features"] for j in js}) > 1 for js in tw))
        if pairs_:
            a_, c_ = np.array(pairs_).T
            r.update(corr=float(np.corrcoef(np.concatenate([d[a_], d[c_]]), np.concatenate([d[c_], d[a_]]))[0, 1]),
                     mean_abs_diff=float(np.mean(np.abs(d[a_] - d[c_]))),
                     mean_abs_diff_random=float(np.mean(np.abs(d[rnd[:, 0]] - d[rnd[:, 1]]))),
                     n_pairs=len(pairs_))
            # are the twins' difficulties closer than their measurement noise allows?
            r["noise_sd_pair_diff"] = float(np.sqrt(np.mean(base[b]["vd"][a_] + base[b]["vd"][c_])))
            # item-weighted, so one 336-item group does not stand for all twins
            js = np.concatenate([np.array(g) for g in tw])
            cz, nz = codes([gi for gi, g in enumerate(tw) for _ in g])
            r["eta2_items"], r["eps2_items"] = anova(d[js], cz, nz)
            r["groups_le_4"] = sum(len(g) <= 4 for g in tw)
        R["twins"][f"{b}|{unit}"] = r
        print(b, unit, r)
    # cross-benchmark twins
    seen = defaultdict(set)
    for b in BENCH:
        for it in tabs[b]["items"]:
            seen[it["item_content"]].add(b)
    R["twins_cross_benchmark"] = sum(len(v) > 1 for v in seen.values())
    # coverage in official runs
    main_keys = {b: [k for k in glm[b] if "+" not in k and k != "full_string"] for b in BENCH}
    for scope in ("pair", "benchmark"):
        R[f"coverage_{scope}"] = cov = coverage(O.eligible(load_pairs()), runs=150,
                                               main_keys=main_keys, scope=scope)
        for k, v in sorted(cov.items()):
            print(scope, k, {B: (round(x["any"], 3), round(x["mean"], 2)) for B, x in v.items()})
    R["elapsed_s"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", OUT, f"{R['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
