"""Meta-learned heads on frozen Qwen3-Embedding-0.6B features, on top of the
shipped hier, scored by nested leave-one-parent-out on test-like runs (a null).

This is the "meta-heads" study (docs/report/draft.md section 6.4), moved here
from the session scratch where it first ran (rethink2/fine-tuning/heads.py,
2026-09-26). The model, the training loop, the selection and the statistics
are that script's, line for line; only the data loading changed. It read its
own scratch rows; this reads the rows of experiments/harness.py, which hold the
same predictions: data/harness_rows_legacy reproduces the scratch rows exactly
(0 of 450 runs differ, every evaluated and labeled prediction; the subject
dicts and every item's text key match too), and data/harness_rows are the
re-collected rows of the current library (157 of 300 test-like and 74 of 150
mix/whole runs differ, all through matharena: the corrected multiple-choice
floor and the floored-fit fix).

An episode is one pair appearance of a test-like run at one budget B in
{1, 3, 7, 15, 31}: hier's prediction z_i (logit) for each evaluated item, the
labels revealed so far on the same benchmark_id (the pair's own first B, and
up to MAX_OTHER of the other subjects' on that pseudo-benchmark in the run)
with hier's prediction for them, and the subject's visible attributes. A head
outputs a logit offset added to z_i, capped at 4 tanh(o / 4). B0 has no labels,
so every head is inactive there.

Heads (all linear or low-rank; X = PCA(64) of the embedding fitted on the
training parents only, whitened, plus log text length):
  diff    u_i = X_i w + z_i * X_i v; offset = exp(a_B) * (u_i - mean_L u_j)
          (a meta-learned difficulty direction, centred on the labeled items,
          so a benchmark's level from text never enters)
  kern    few-shot kernel: offset = softplus(beta_B) * sum_j K_ij r_j /
          (softplus(tau) + sum_j K_ij p_j (1 - p_j)), K_ij = exp(-|Q(X_i -
          X_j)|^2 / 2) * (own_j + sigmoid(omega) * other_j), r_j = y_j - p_j;
          Q (65 x 16) learned (a meta-learned metric, matching-network style)
  ass     diff plus an assessor-style subject x item term f_s' U V X_i (rank
          2), f_s = provider one-hot, reasoning flag, release year
  all     diff + kern + ass
  oracle  offset = -exp(a_B) * (d_i - mean_L d_j), d the parent's in-sample
          Rasch difficulty (testlike.rasch on every pair of the parent,
          evaluated responses included; averaged over item ids sharing a text):
          what a perfect difficulty covariate would give through this form.
          oracle_r<r> degrades it to correlation r (standardised within
          parent, noise from default_rng(123) in table order).
Loss: sum_B w_B * Brier over the pair's evaluated responses (ALC weights), mean
over episodes, plus an L2 penalty lambda on w, v, U, V, Q. lambda is chosen by
nested leave-one-parent-out over the training parents (grid plus 'off': a head
is used on the held-out parent only if its inner mean is below 0). --force
skips the selection (lams[0] on every fold). 'indist' is the in-distribution
control: train on one half of the subjects of every parent (sha256 of the
subject id), score the other half, so the benchmarks are seen.

Statistics: per pair appearance, ALC difference against hier (B0 contributes
0), weighted 1 / run size; run SE over runs; a (parent, subject) cluster
bootstrap (1,000 resamples, seed 0). Runs: test-like (300, testlike.Regime()
defaults, seed 2) and mix/whole (150, seed 3), the runs of
experiments/level_calibration.py and experiments/harness.py.

Inputs: the harness rows (python experiments/harness.py --stage collect; the
legacy rows are the collection at bd0be67), the embeddings of
experiments/llm_features.py in data/features/ (Qwen3-Embedding-0.6B, pinned
revision; no model is loaded here), and the public pairs for the oracle's
difficulty. Needs torch (CPU). Everything is seeded; torch runs on --threads
threads (2, as the scratch runs did).

    python experiments/heads_eval.py --rows legacy            # the study as it ran (~26 min, 1 process, 2 threads)
    python experiments/heads_eval.py --rows current           # the same on the re-collected rows
    python experiments/heads_eval.py --show

Every configuration of the scratch queue is in CONFIGS; --configs picks some
(regular expression), --resume keeps configurations already in the results
file. Output: results/heads_eval.json (per rows set and configuration, the
test-like and mix/whole summaries, per parent and per kind, the nested choices
and the wall time; provenance under meta).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUD = [0, 1, 3, 7, 15, 31]
WB = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
PARENTS = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
REGIMES = {"tl": 300, "mix": 150}           # harness row directories and run counts
PROV = ["OpenAI", "Anthropic", "Google", "Meta", "DeepSeek", "Alibaba", "xAI", "Mistral"]
MAX_OTHER = 62
FEAT = os.path.join(ROOT, "data", "features")
ROWS = {"current": os.path.join(ROOT, "data", "harness_rows"),
        "legacy": os.path.join(ROOT, "data", "harness_rows_legacy")}
OUT = os.path.join(ROOT, "results", "heads_eval.json")

#: the scratch study's queue (queue_a.sh, queue_b.sh), in its order
CONFIGS = {
    "base": dict(stage="base", variant="diff"),
    "lopo_diff": dict(stage="lopo", variant="diff", lams=[0.01, 0.1, 1.0]),
    "lopo_kern": dict(stage="lopo", variant="kern", lams=[0.01, 0.1, 1.0]),
    "lopo_ass": dict(stage="lopo", variant="ass", lams=[0.01, 0.1, 1.0]),
    "lopo_all": dict(stage="lopo", variant="all", lams=[0.01, 0.1, 1.0]),
    "force_all_0.01": dict(stage="lopo", variant="all", lams=[0.01], force=True),
    "force_all_0.1": dict(stage="lopo", variant="all", lams=[0.1], force=True),
    "force_all_1.0": dict(stage="lopo", variant="all", lams=[1.0], force=True),
    "lopo_oracle": dict(stage="lopo", variant="oracle"),
    "lopo_oracle_r0.1": dict(stage="lopo", variant="oracle_r0.1"),
    "lopo_oracle_r0.2": dict(stage="lopo", variant="oracle_r0.2"),
    "lopo_oracle_r0.3": dict(stage="lopo", variant="oracle_r0.3"),
    "lopo_oracle_r0.5": dict(stage="lopo", variant="oracle_r0.5"),
    "lopo_oracle_r0.7": dict(stage="lopo", variant="oracle_r0.7"),
    "indist_diff_0.001": dict(stage="indist", variant="diff", lams=[0.001]),
    "indist_diff_0.01": dict(stage="indist", variant="diff", lams=[0.01]),
    "indist_all_0.001": dict(stage="indist", variant="all", lams=[0.001]),
    "indist_all_0.01": dict(stage="indist", variant="all", lams=[0.01]),
    "indist_all_d256": dict(stage="indist", variant="all", lams=[0.001], dim=256),
    "lopo_diff_d16": dict(stage="lopo", variant="diff", lams=[0.01, 0.1, 1.0], dim=16),
    "lopo_diff_d256": dict(stage="lopo", variant="diff", lams=[0.01, 0.1, 1.0], dim=256),
}
DEFAULTS = dict(lams=[1e-3, 1e-2, 1e-1], epochs=3, dim=64, force=False, train_regimes=["tl", "mix"])
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ data

def load_table():
    """The embedding table (one row per distinct item text of the four parents,
    first occurrence in PARENTS order, float16 as stored), its parents, the log
    text length, the in-sample Rasch difficulty per text, and item id -> text key."""
    import pandas as pd
    from paiec import data as D
    from paiec import llmfeat
    from paiec import official as O
    from paiec import testlike as T
    keys, embs, parent, nch, item_tk = [], [], [], {}, {}
    for b in PARENTS:
        index, emb, _, _ = llmfeat.load(FEAT, b)
        ok = index["has_emb"].to_numpy()
        keys += list(index["text_key"].to_numpy()[ok])
        embs.append(emb[ok])
        parent += [b] * int(ok.sum())
        ix = pd.read_parquet(os.path.join(FEAT, b, "index.parquet"), columns=["text_key", "n_chars", "item_ids"])
        nch.update(dict(zip(ix.text_key, ix.n_chars)))
        for t, ids in zip(ix.text_key, ix.item_ids):
            for i in ids:
                if item_tk.setdefault(str(i), t) != t:
                    raise RuntimeError(f"item {i} has two text keys")
    E = np.concatenate(embs).astype(np.float16)
    keys = np.array(keys)
    _, first = np.unique(keys, return_index=True)
    first = np.sort(first)
    tk, E, par = keys[first], E[first].astype(np.float32), np.array(parent)[first]
    loglen = np.log1p(np.array([nch.get(k, 0) for k in tk], float))
    acc = {}
    for b in PARENTS:
        ps = O.eligible(D.load_pairs([b]))
        diff = T.rasch(ps)
        for p in ps:
            for r in p.responses:
                acc.setdefault(llmfeat.text_key(r.item), {})[r.item_key] = diff[r.item_key]
        log(b, len(ps), "pairs")
    d = {k: float(np.mean(list(v.values()))) for k, v in acc.items()}
    diffv = np.array([d.get(k, np.nan) for k in tk], float)
    return tk, E, par, loglen, diffv, item_tk


def subj_feats(s):
    f = np.zeros(len(PROV) + 3, np.float32)
    pv = s.get("provider", "")
    f[PROV.index(pv) if pv in PROV else len(PROV)] = 1.0
    f[len(PROV) + 1] = 1.0 if s.get("reasoning_effort", "") not in ("", "none") else 0.0
    rd = s.get("release_date", "")
    try:
        f[len(PROV) + 2] = (int(rd[:4]) + (int(rd[5:7]) - 1) / 12 - 2025.0) if rd else 0.0
    except Exception:
        f[len(PROV) + 2] = 0.0
    return f


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build(regime, rows_dir, pos, item_tk, max_runs=None):
    """Episodes of one regime from the harness rows. Items are keyed by their
    text's table row, as the scratch study keyed them: items sharing a text
    are one row (K and N summed, hier's prediction taken at the first one), and
    predictions go through float32, as its rows stored them."""
    ep = {k: [] for k in ["run", "slot", "bi", "parent", "cluster", "rsize", "fs",
                          "t_row", "t_k", "t_n", "t_z", "l_row", "l_y", "l_p", "l_own",
                          "base_b0", "kind"]}
    n = REGIMES[regime] if max_runs is None else min(max_runs, REGIMES[regime])
    lib = set()

    def rows_of(keys):
        out = []
        for k in keys:
            t = item_tk.get(str(k))
            if t is None or t not in pos:
                raise RuntimeError(f"{regime}: item {k} has no embedding")
            out.append(pos[t])
        return np.array(out, np.int64)

    for run in range(n):
        with open(os.path.join(rows_dir, regime, f"{run}.pkl"), "rb") as f:
            r = pickle.load(f)
        lib.add(r.get("lib_digest"))
        slots = r["slots"]
        acq = []
        for s in slots:
            acq.append((rows_of(s["acq_keys"]), s["acq_y"].astype(float),
                        s["acq_p"].astype(np.float32).astype(float)))
        by = {}
        for si, s in enumerate(slots):
            by.setdefault(s["bench"], []).append(si)
        for si, s in enumerate(slots):
            it = rows_of(s["ev_keys"])
            u, inv = np.unique(it, return_inverse=True)
            K = np.bincount(inv, s["ev_K"].astype(float), len(u))
            N = np.bincount(inv, s["ev_N"].astype(float), len(u))
            first = np.zeros(len(u), int)
            first[inv[::-1]] = np.arange(len(it))[::-1]
            evp = s["ev_p"].astype(np.float32).astype(float)
            fs = subj_feats(s["subject"])
            p0, Ki, Ni = evp[0], s["ev_K"].astype(float), s["ev_N"].astype(float)
            b0 = float(np.sum(Ni * p0 ** 2 - 2 * p0 * Ki + Ki) / Ni.sum())
            a_row, a_y, a_p = acq[si]
            for bi in range(1, 6):
                b = BUD[bi]
                zt = logit(evp[bi][first])
                k = min(b, len(a_row))
                rows, ys, ps = list(a_row[:k]), list(a_y[:k]), list(a_p[bi, :k])
                own = [1.0] * k
                oth = 0
                for sj in by[s["bench"]]:
                    if sj == si:
                        continue
                    t_row, t_y, t_p = acq[sj]
                    kk = min(b, len(t_row), MAX_OTHER - oth)
                    rows += list(t_row[:kk]); ys += list(t_y[:kk]); ps += list(t_p[bi, :kk])
                    own += [0.0] * kk
                    oth += kk
                ep["run"].append(run); ep["slot"].append(si); ep["bi"].append(bi)
                ep["parent"].append(s["parent"]); ep["cluster"].append((s["parent"], s["anon_sid"]))
                ep["rsize"].append(len(slots)); ep["fs"].append(fs); ep["kind"].append(s["kind"])
                ep["t_row"].append(u.astype(np.int64)); ep["t_k"].append(K); ep["t_n"].append(N)
                ep["t_z"].append(zt)
                ep["l_row"].append(np.array(rows, np.int64)); ep["l_y"].append(np.array(ys, float))
                ep["l_p"].append(np.array(ps, float)); ep["l_own"].append(np.array(own, float))
                ep["base_b0"].append(b0)
    for k in ["run", "slot", "bi", "rsize"]:
        ep[k] = np.array(ep[k])
    ep["parent"] = np.array(ep["parent"])
    ep["kind"] = np.array(ep["kind"])
    ep["fs"] = np.stack(ep["fs"])
    ep["regime"] = regime
    log(regime, n, "runs", len(ep["run"]), "episodes")
    return ep, sorted(str(x) for x in lib)


def batch(ep, idx, X, D):
    """Padded tensors for episodes idx."""
    import torch
    nt = max(len(ep["t_row"][i]) for i in idx)
    nl = max(max(len(ep["l_row"][i]) for i in idx), 1)
    B = len(idx)
    Xt = np.zeros((B, nt, X.shape[1]), np.float32); zt = np.zeros((B, nt), np.float32)
    kt = np.zeros((B, nt), np.float32); nn = np.zeros((B, nt), np.float32)
    Xl = np.zeros((B, nl, X.shape[1]), np.float32); yl = np.zeros((B, nl), np.float32)
    pl = np.full((B, nl), 0.5, np.float32); ol = np.zeros((B, nl), np.float32)
    ml = np.zeros((B, nl), np.float32)
    dt = np.zeros((B, nt), np.float32); dl = np.zeros((B, nl), np.float32)
    for a, i in enumerate(idx):
        n = len(ep["t_row"][i])
        Xt[a, :n] = X[ep["t_row"][i]]; zt[a, :n] = ep["t_z"][i]
        kt[a, :n] = ep["t_k"][i]; nn[a, :n] = ep["t_n"][i]
        dt[a, :n] = D[ep["t_row"][i]]
        m = len(ep["l_row"][i])
        if m:
            Xl[a, :m] = X[ep["l_row"][i]]; yl[a, :m] = ep["l_y"][i]
            pl[a, :m] = ep["l_p"][i]; ol[a, :m] = ep["l_own"][i]; ml[a, :m] = 1.0
            dl[a, :m] = D[ep["l_row"][i]]
    T = torch.from_numpy
    return dict(Xt=T(Xt), zt=T(zt), kt=T(kt), nt=T(nn), Xl=T(Xl), yl=T(yl), pl=T(pl),
                ol=T(ol), ml=T(ml), dt=T(dt), dl=T(dl),
                fs=T(ep["fs"][idx].astype(np.float32)),
                bi=torch.from_numpy(ep["bi"][idx] - 1), wb=T(WB[ep["bi"][idx]].astype(np.float32)))


# ------------------------------------------------------------------ model

def make_head(dim, F, variant, rank=2, q=16, seed=0):
    import torch

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            P = torch.nn.Parameter
            self.variant = variant
            self.diff = variant in ("diff", "ass", "all")
            self.ass = variant in ("ass", "all")
            self.kern = variant in ("kern", "all")
            self.oracle = variant == "oracle"
            self.w = P(torch.zeros(dim)); self.v = P(torch.zeros(dim))
            self.a = P(torch.zeros(5))
            self.U = P(0.01 * torch.randn(F, rank, generator=g)); self.V = P(torch.zeros(rank, dim))
            self.Q = P(0.3 * torch.randn(dim, q, generator=g) / np.sqrt(dim))
            self.lb = P(torch.full((5,), -1.0)); self.ltau = P(torch.tensor(0.0))
            self.lom = P(torch.tensor(-1.0))
            self.ao = P(torch.full((5,), -1.0))

        def penalty(self):
            return (self.w ** 2).sum() + (self.v ** 2).sum() + (self.U ** 2).sum() + \
                (self.V ** 2).sum() + (self.Q ** 2).sum()

        def forward(self, b):
            off = torch.zeros_like(b["zt"])
            nL = b["ml"].sum(1, keepdim=True)
            has = (nL > 0).float()
            zl = torch.logit(b["pl"].clamp(1e-6, 1 - 1e-6))
            if self.diff:
                ut = b["Xt"] @ self.w + b["zt"] * (b["Xt"] @ self.v)
                ul = b["Xl"] @ self.w + zl * (b["Xl"] @ self.v)
                if self.ass:
                    wv = (b["fs"] @ self.U) @ self.V
                    ut = ut + (b["Xt"] * wv[:, None, :]).sum(-1)
                    ul = ul + (b["Xl"] * wv[:, None, :]).sum(-1)
                ubar = (ul * b["ml"]).sum(1, keepdim=True) / nL.clamp(min=1)
                off = off + torch.exp(self.a[b["bi"]])[:, None] * (ut - ubar) * has
            if self.kern:
                A = b["Xt"] @ self.Q; Bq = b["Xl"] @ self.Q
                d2 = torch.cdist(A, Bq) ** 2
                wgt = b["ml"] * (b["ol"] + torch.sigmoid(self.lom) * (1 - b["ol"]))
                K = torch.exp(-d2 / 2) * wgt[:, None, :]
                r = (b["yl"] - b["pl"]) * b["ml"]
                vv = b["pl"] * (1 - b["pl"])
                num = (K * r[:, None, :]).sum(-1)
                den = torch.nn.functional.softplus(self.ltau) + (K * vv[:, None, :]).sum(-1)
                off = off + torch.nn.functional.softplus(self.lb[b["bi"]])[:, None] * num / den
            if self.oracle:
                dbar = (b["dl"] * b["ml"]).sum(1, keepdim=True) / nL.clamp(min=1)
                off = off - torch.exp(self.ao[b["bi"]])[:, None] * (b["dt"] - dbar) * has
            return 4.0 * torch.tanh(off / 4.0)

    return Head()


def brier_parts(q, b):
    """Brier over responses per episode: sum_i [n q^2 - 2 q k + k] / sum n."""
    num = (b["nt"] * q ** 2 - 2 * q * b["kt"] + b["kt"]).sum(1)
    return num / b["nt"].sum(1)


def fit(ep, tr, X, D, variant, lam, epochs=4, bs=64, lr=0.02, seed=0):
    import torch
    torch.manual_seed(seed)
    m = make_head(X.shape[1], ep["fs"].shape[1], variant, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(tr)
    for e in range(epochs):
        perm = tr[rng.permutation(n)]
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            b = batch(ep, idx, X, D)
            q = torch.sigmoid(b["zt"] + m(b))
            loss = (b["wb"] * brier_parts(q, b)).mean() + lam * m.penalty()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def score(ep, te, X, D, m, bs=256):
    """Per episode (Brier_head - Brier_base)."""
    import torch
    out = np.zeros(len(te))
    with torch.no_grad():
        for s in range(0, len(te), bs):
            idx = te[s:s + bs]
            b = batch(ep, idx, X, D)
            q0 = torch.sigmoid(b["zt"])
            q1 = torch.sigmoid(b["zt"] + m(b)) if m is not None else q0
            out[s:s + bs] = (brier_parts(q1, b) - brier_parts(q0, b)).numpy()
    return out


def base_brier(ep, te, X, D, bs=256):
    import torch
    out = np.zeros(len(te))
    with torch.no_grad():
        for s in range(0, len(te), bs):
            idx = te[s:s + bs]
            b = batch(ep, idx, X, D)
            out[s:s + bs] = brier_parts(torch.sigmoid(b["zt"]), b).numpy()
    return out


# ------------------------------------------------------------------ features

def features(E, loglen, train_rows, dim=64):
    mu = E[train_rows].mean(0)
    U, S, Vt = np.linalg.svd(E[train_rows] - mu, full_matrices=False)
    P = Vt[:dim].T / (S[:dim] / np.sqrt(len(train_rows)))       # whitened
    X = (E - mu) @ P
    ll = (loglen - loglen[train_rows].mean()) / loglen[train_rows].std()
    return np.concatenate([X, ll[:, None]], 1).astype(np.float32)


# ------------------------------------------------------------------ evaluation

def appearance_delta(ep, idx, d):
    """Episode deltas -> per appearance ALC delta (B0 contributes 0) and per budget."""
    key = {}
    for j, i in enumerate(idx):
        a = (ep["regime"], ep["run"][i], ep["slot"][i])
        rec = key.setdefault(a, {"by": np.zeros(6), "parent": ep["parent"][i],
                                 "cluster": ep["cluster"][i], "w": 1.0 / ep["rsize"][i],
                                 "run": ep["run"][i], "kind": ep["kind"][i]})
        rec["by"][ep["bi"][i]] = d[j]
    for rec in key.values():
        rec["alc"] = float(WB @ rec["by"])
    return list(key.values())


def summarise(apps, boots=1000, seed=0):
    w = np.array([a["w"] for a in apps])
    alc = np.array([a["alc"] for a in apps])
    by = np.stack([a["by"] for a in apps])
    est = float((w * alc).sum() / w.sum())
    est_b = (w[:, None] * by).sum(0) / w.sum()
    runs = {}
    for a in apps:
        runs.setdefault(a["run"], []).append(a)
    rm = np.array([np.sum([x["w"] * x["alc"] for x in v]) / np.sum([x["w"] for x in v]) for v in runs.values()])
    run_se = float(rm.std(ddof=1) / np.sqrt(len(rm))) if len(rm) > 1 else float("nan")
    cl = {}
    for j, a in enumerate(apps):
        cl.setdefault(a["cluster"], []).append(j)
    cls = list(cl.values())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        pick = rng.integers(0, len(cls), len(cls))
        jj = np.concatenate([cls[p] for p in pick])
        bs.append((w[jj] * alc[jj]).sum() / w[jj].sum())
    return dict(alc=est, run_se=run_se, cluster_se=float(np.std(bs)), by_budget=[float(x) for x in est_b],
                n_app=len(apps), n_clusters=len(cls))


def per_parent_mean(ep, idx, d):
    apps = appearance_delta(ep, idx, d)
    out = {}
    for p in PARENTS:
        a = [x for x in apps if x["parent"] == p]
        if a:
            w = np.array([x["w"] for x in a]); v = np.array([x["alc"] for x in a])
            out[p] = float((w * v).sum() / w.sum())
    return out


# ------------------------------------------------------------------ one configuration

class Study:
    """The merged episode store of both regimes for one rows set, and the tables."""

    def __init__(self, rows_dir, max_runs=None):
        self.tk, self.E, self.par, self.loglen, self.diffv, item_tk = load_table()
        pos = {k: i for i, k in enumerate(self.tk)}
        eps, self.lib = {}, {}
        for r in REGIMES:
            eps[r], self.lib[r] = build(r, rows_dir, pos, item_tk, max_runs)
        self.eps = eps
        M = {k: [] for k in eps["tl"] if k != "regime"}
        src = []
        for r in REGIMES:
            for k in M:
                M[k] += list(eps[r][k])
            src += [r] * len(eps[r]["run"])
        for k in ["run", "slot", "bi", "rsize"]:
            M[k] = np.array(M[k])
        M["parent"] = np.array(M["parent"]); M["kind"] = np.array(M["kind"]); M["fs"] = np.stack(M["fs"])
        M["regime"] = None
        self.M, self.src = M, np.array(src)
        self.info = {"table_rows": int(len(self.tk)), "episodes": {r: int(len(eps[r]["run"])) for r in REGIMES},
                     "runs": {r: int(len(set(eps[r]["run"].tolist()))) for r in REGIMES},
                     "rows_lib_digest": self.lib,
                     "oracle_texts": int(np.isfinite(self.diffv).sum())}

    def idx_of(self, parents, regimes, extra=None):
        m = np.isin(self.M["parent"], parents) & np.isin(self.src, regimes)
        if extra is not None:
            m &= extra
        return np.flatnonzero(m)

    def apps_for(self, idx, d):
        out = []
        for r in REGIMES:
            sel = self.src[idx] == r
            self.M["regime"] = r
            out += [dict(x, regime=r) for x in appearance_delta(self.M, idx[sel], d[sel])]
        return out

    def run(self, cfg):
        a = dict(DEFAULTS, **cfg)
        M, E, loglen, par, diffv = self.M, self.E, self.loglen, self.par, self.diffv
        Dv = np.nan_to_num(diffv - np.nanmean(diffv)).astype(np.float32)
        variant = a["variant"]
        if variant.startswith("oracle_r"):
            rr = float(variant[len("oracle_r"):])
            g = np.random.default_rng(123)
            Dn = np.zeros_like(Dv)
            for p in PARENTS:
                mm = (par == p) & np.isfinite(diffv)
                zz = (diffv[mm] - diffv[mm].mean()) / diffv[mm].std()
                Dn[mm] = rr * zz + np.sqrt(1 - rr ** 2) * g.standard_normal(mm.sum())
            Dv = Dn.astype(np.float32)
            variant = "oracle"
        res = {k: a[k] for k in ("stage", "variant", "dim", "lams", "epochs", "train_regimes", "force")}
        t0 = time.time()
        if a["stage"] == "base":
            X = features(E, loglen, np.flatnonzero(np.isin(par, PARENTS)), a["dim"])
            for r in REGIMES:
                ii = self.idx_of(PARENTS, [r])
                bb = base_brier(M, ii, X, Dv)
                by = [float(np.mean(self.eps[r]["base_b0"]))] + \
                     [float(np.mean(bb[M["bi"][ii] == k])) for k in range(1, 6)]
                res[r] = {"base_brier_by_budget_episode_mean": by}
                log(r, [round(x, 5) for x in by])
        elif a["stage"] == "lopo":
            allapps, chosen = [], {}
            for q in PARENTS:
                trp = [p for p in PARENTS if p != q]
                X = features(E, loglen, np.flatnonzero(np.isin(par, trp)), a["dim"])
                inner = {}
                if variant == "oracle":
                    lam_star, inner = 0.0, None
                elif a["force"]:
                    lam_star, inner = a["lams"][0], None
                else:
                    for lam in a["lams"]:
                        vals = []
                        for q2 in trp:
                            tr2 = [p for p in trp if p != q2]
                            X2 = features(E, loglen, np.flatnonzero(np.isin(par, tr2)), a["dim"])
                            m = fit(M, self.idx_of(tr2, a["train_regimes"]), X2, Dv, variant, lam, a["epochs"])
                            te2 = self.idx_of([q2], ["tl"])
                            d = score(M, te2, X2, Dv, m)
                            M["regime"] = "tl"
                            vals.append(np.mean(list(per_parent_mean(M, te2, d).values())))
                        inner[lam] = float(np.mean(vals))
                        log(variant, "outer", q, "lam", lam, "inner mean dALC", round(inner[lam], 6),
                            [round(v, 6) for v in vals])
                    best = min(inner, key=inner.get)
                    lam_star = best if inner[best] < 0 else None
                chosen[q] = {"lam": lam_star, "inner": inner}
                te = np.concatenate([self.idx_of([q], ["tl"]), self.idx_of([q], ["mix"])])
                if lam_star is None:
                    d = np.zeros(len(te))
                else:
                    m = fit(M, self.idx_of(trp, a["train_regimes"]), X, Dv, variant, lam_star, a["epochs"])
                    d = score(M, te, X, Dv, m)
                    if variant == "oracle":
                        import torch
                        chosen[q]["scale"] = torch.exp(m.ao).tolist()
                ap_q = self.apps_for(te, d)
                allapps += ap_q
                log(cfg["variant"], "outer", q, "chosen", lam_star, "held-out tl dALC",
                    round(summarise([x for x in ap_q if x["regime"] == "tl"], boots=200)["alc"], 6))
            res["chosen"] = chosen
            self._summaries(res, allapps, per_kind=True)
        elif a["stage"] == "indist":
            h = np.array([int(hashlib.sha256(str(c[1]).encode()).hexdigest(), 16) % 2 for c in M["cluster"]])
            X = features(E, loglen, np.flatnonzero(np.isin(par, PARENTS)), a["dim"])
            lam = a["lams"][0]
            allapps = []
            for fold in (0, 1):
                m = fit(M, self.idx_of(PARENTS, a["train_regimes"], h == fold), X, Dv, variant, lam, a["epochs"])
                te = self.idx_of(PARENTS, list(REGIMES), h != fold)
                d = score(M, te, X, Dv, m)
                allapps += self.apps_for(te, d)
            self._summaries(res, allapps, per_kind=False)
        res["secs"] = round(time.time() - t0, 1)
        return res

    @staticmethod
    def _summaries(res, allapps, per_kind):
        for r in REGIMES:
            A = [x for x in allapps if x["regime"] == r]
            res[r] = summarise(A)
            res[r]["per_parent"] = {p: summarise([x for x in A if x["parent"] == p], boots=500 if per_kind else 300)
                                    for p in PARENTS}
            if per_kind:
                res[r]["per_kind"] = {k: summarise([x for x in A if x["kind"] == k], boots=300)
                                      for k in sorted({x["kind"] for x in A})}
            log(r, json.dumps({k: (round(res[r][k], 6) if isinstance(res[r][k], float) else res[r][k])
                               for k in ["alc", "run_se", "cluster_se"]}))


# ------------------------------------------------------------------ io

def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(os.path.join(ROOT, p), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def show(state):
    for label, blk in state.get("rows", {}).items():
        print(f"== rows: {label} ({blk.get('source')}; library {blk.get('info', {}).get('rows_lib_digest')})")
        for k, v in blk.get("configs", {}).items():
            t, m = v.get("tl", {}), v.get("mix", {})
            if "alc" in t:
                ch = v.get("chosen")
                on = None if ch is None else sum(c["lam"] is not None for c in ch.values())
                print(f"  {k:20s} tl {t['alc']:+.5f} (run {t['run_se']:.5f} / cl {t['cluster_se']:.5f})  "
                      f"mix {m['alc']:+.5f} (cl {m['cluster_se']:.5f})"
                      + ("" if on is None else f"  folds on {on}/4") + f"  {v['secs']:.0f}s")
            elif "base_brier_by_budget_episode_mean" in t:
                print(f"  {k:20s} tl base Brier {[round(x, 4) for x in t['base_brier_by_budget_episode_mean']]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="current", help="'current', 'legacy' or a rows directory")
    ap.add_argument("--configs", default=None, help="regular expression on CONFIGS names")
    ap.add_argument("--resume", action="store_true", help="keep configurations already in the results file")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-runs", type=int, default=None, help="first n runs of each regime (smoke test)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    state = load_json(a.out) or {"meta": {}, "rows": {}}
    if a.show:
        show(state)
        return
    import torch
    torch.set_num_threads(a.threads)
    label = a.rows if a.rows in ROWS else os.path.basename(os.path.normpath(a.rows))
    rows_dir = ROWS.get(a.rows, a.rows)
    if a.max_runs is not None:
        label += f"_first{a.max_runs}"
    names = [n for n in CONFIGS if a.configs is None or re.search(a.configs, n)]
    blk = state["rows"].setdefault(label, {"configs": {}})
    todo = [n for n in names if not (a.resume and n in blk["configs"])]
    log(f"rows {label}: {len(todo)} configurations to run")
    if not todo:
        return
    study = Study(rows_dir, a.max_runs)
    blk["source"] = os.path.relpath(rows_dir, ROOT)
    blk["info"] = study.info
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    state["meta"] = {"script": "experiments/heads_eval.py", "script_digest": digest(["experiments/heads_eval.py"]),
                     "commit": commit, "torch": torch.__version__, "numpy": np.__version__,
                     "threads": a.threads, "max_other": MAX_OTHER,
                     "provenance": "session scratch rethink2/fine-tuning/heads.py (2026-09-26), moved "
                                   "unchanged except for the data loading"}
    passes = blk.setdefault("passes", [])
    t0 = time.time()

    def save():
        # re-read first, so that runs on other rows sets writing the same file keep their blocks
        cur = load_json(a.out) or {"meta": {}, "rows": {}}
        cur["meta"] = state["meta"]
        cur.setdefault("rows", {})[label] = blk
        save_json(a.out, cur)
        return cur

    for n in todo:
        log("config", n)
        blk["configs"][n] = study.run(CONFIGS[n])
        save()
    passes.append({"command": " ".join(sys.argv[1:]), "configs": todo, "wall_s": round(time.time() - t0),
                   "script_digest": state["meta"]["script_digest"], "commit": commit})
    show(save())


if __name__ == "__main__":
    main()
