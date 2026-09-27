"""Does item difficulty predicted from Qwen3-Embedding-0.6B embeddings transfer to
an unseen benchmark?  Read-only probe; one process, small memory, no model loaded.

Target: Rasch difficulty per item (subject ability divided out, paiec.rasch.rasch
on the pairs exactly as paiec.data.load_pairs builds them: binary responses,
(subject, benchmark) pairs with >= 80 distinct items), averaged over item_ids that
share a feature key, standardised within benchmark.

Features: the precomputed embeddings (paiec.llmfeat.load), joined on `key`
(predict.item_key digest with the public benchmark name; checked against the
index's item_ids). Contrast: TF-IDF (hashed uni+bigrams, df>=5 on the training
rows, sublinear tf) + truncated SVD(200) fit on training rows only, on the same
text the embedding read (item_text: content, newline, item_features; head and
tail 4,000 characters each), and log text length alone.

Learners: ridge (alpha nested) and kNN mean of the k most cosine-similar
training items (k nested). Nested = chosen on the training data only: leave one
training benchmark out for LOBO, inner 5-fold for within-benchmark CV.

Run:  python experiments/emb_transfer.py   (writes results/emb_transfer.json)
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import json
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")
REPO = ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, GroupKFold

from paiec import llmfeat
from paiec.rasch import rasch
from paiec.hier import parse_features

DATA = os.path.join(REPO, "data")
FEAT = os.path.join(DATA, "features")
BENCH = ["matharena", "multi_swebench", "real_webagents", "researchcodebench"]
GROUP = {"matharena": "competition", "multi_swebench": "lang",
         "real_webagents": "website", "researchcodebench": "paper"}
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
KS = [5, 10, 20, 50, 100, 200]
HEAD = 4000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def clean(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


# --------------------------------------------------------------------------- data

def load_bench(b):
    items = pd.read_parquet(os.path.join(DATA, b, "items.parquet"),
                            columns=["item_id", "benchmark_id", "content", "item_features"])
    resp = pd.read_parquet(os.path.join(DATA, b, "response.parquet"),
                           columns=["subject_id", "item_id", "benchmark_id", "response"])
    resp = resp[resp.response.isin([0.0, 1.0])]
    nun = resp.groupby(["subject_id", "benchmark_id"]).item_id.transform("nunique")
    resp = resp[nun >= 80].reset_index(drop=True)          # load_pairs(min_items=80)
    pair = pd.factorize(resp.subject_id.astype(str) + "|" + resp.benchmark_id.astype(str))[0]
    iid_codes, iid_uni = pd.factorize(resp.item_id.astype(str))
    y = resp.response.values.astype(float)
    th, bb = rasch(pair, iid_codes, y, pair.max() + 1, len(iid_uni))
    n_iid = np.bincount(iid_codes, minlength=len(iid_uni))
    b_of = dict(zip(iid_uni, bb))
    n_of = dict(zip(iid_uni, n_iid))

    index, emb, _, _ = llmfeat.load(FEAT, b)
    pos = {k: i for i, k in enumerate(index["key"])}
    ids_of = {k: set(map(str, v)) for k, v in zip(index["key"], index["item_ids"])}

    items = items.set_index(items.item_id.astype(str))
    rows, miss, bad = {}, 0, 0
    for iid, r in items.iterrows():
        if iid not in b_of:
            continue
        it = {"item_content": clean(r.content), "item_features": clean(r.item_features),
              "interactors": "", "benchmark_id": clean(r.benchmark_id)}
        k = llmfeat.key_for(it, b)
        if k not in pos:
            miss += 1
            continue
        if iid not in ids_of[k]:
            bad += 1
        d = rows.setdefault(k, {"key": k, "bsum": 0.0, "n": 0, "n_ids": 0, "content": it["item_content"],
                                "features": it["item_features"], "text": llmfeat.item_text(it)})
        d["bsum"] += b_of[iid] * n_of[iid]
        d["n"] += n_of[iid]
        d["n_ids"] += 1
    df = pd.DataFrame(list(rows.values()))
    df["b"] = df.bsum / df.n
    df["z"] = (df.b - df.b.mean()) / df.b.std()
    df["bench"] = b
    df["group"] = [parse_features(f).get(GROUP[b], "") for f in df.features]
    c = df.content
    df["degenerate"] = c.str.contains("See image", case=False) | (
        (c.str.len() < 120) & (b == "matharena"))
    df["loglen"] = np.log1p(df.text.str.len())
    E = emb[[pos[k] for k in df.key]].astype(np.float32)
    assert np.isfinite(E).all(), f"{b}: missing embeddings"
    # responses mapped to key rows, for the Brier ceiling
    krow = {k: i for i, k in enumerate(df.key)}
    iid2row = {}
    for iid, r in items.iterrows():
        if iid in b_of:
            it = {"item_content": clean(r.content), "item_features": clean(r.item_features),
                  "interactors": "", "benchmark_id": clean(r.benchmark_id)}
            k = llmfeat.key_for(it, b)
            if k in krow:
                iid2row[iid] = krow[k]
    keep = np.array([u in iid2row for u in iid_uni])
    rrow = np.array([iid2row.get(u, -1) for u in iid_uni])[iid_codes]
    ok = rrow >= 0
    R = dict(pair=pair[ok], row=rrow[ok], y=y[ok], th=th[pair[ok]], bi=bb[iid_codes[ok]])
    info = dict(items=len(df), item_ids=int(df.n_ids.sum()), missing_key=miss,
                key_idset_mismatch=bad, pairs=int(pair.max() + 1), responses=int(ok.sum()),
                groups=int(df.group.nunique()), b_sd=float(df.b.std()),
                degenerate=int(df.degenerate.sum()))
    log(b, info)
    return df.drop(columns=["bsum"]), E, R, info


# ----------------------------------------------------------------------- features

def centre_rows(E, labels):
    """Subtract each label's mean row, then L2-normalise (cosine geometry kept)."""
    E = E.copy()
    for g in np.unique(labels):
        m = labels == g
        E[m] -= E[m].mean(0)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(n, 1e-8)


def head_tail_chars(s, n=HEAD):
    return s if len(s) <= 2 * n else s[:n] + "\n" + s[-n:]


class Tfidf:
    """Hashed counts computed once; df filter, idf and SVD fit per training set."""

    def __init__(self, texts):
        hv = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 18, alternate_sign=False,
                               norm=None, strip_accents="unicode", dtype=np.float32)
        self.H = hv.transform([head_tail_chars(t) for t in texts]).tocsr()

    def fit_transform(self, tr, te, n_comp=200, min_df=5):
        Htr = self.H[tr]
        df = np.bincount(Htr.indices, minlength=Htr.shape[1])
        cols = np.flatnonzero(df >= min_df)
        Htr, Hte = Htr[:, cols], self.H[te][:, cols]
        tt = TfidfTransformer(sublinear_tf=True).fit(Htr)
        Xtr = tt.transform(Htr).astype(np.float64)
        Xte = tt.transform(Hte).astype(np.float64)
        if n_comp is None:                       # sparse tf-idf, no SVD
            return Xtr, Xte
        k = min(n_comp, min(Xtr.shape) // 2)
        svd = TruncatedSVD(k, random_state=0, n_iter=5).fit(Xtr)
        Ftr, Fte = svd.transform(Xtr), svd.transform(Xte)
        nrm = lambda F: F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
        return nrm(Ftr).astype(np.float32), nrm(Fte).astype(np.float32)


# ----------------------------------------------------------------------- learners

def ridge_path(Ftr, ytr, Fte, w=None, alphas=ALPHAS):
    """Predictions on Fte for every alpha (intercept unpenalised)."""
    Ftr = Ftr.astype(np.float64); Fte = Fte.astype(np.float64)
    w = np.ones(len(ytr)) if w is None else w / w.mean()
    mx = (w[:, None] * Ftr).sum(0) / w.sum()
    my = (w * ytr).sum() / w.sum()
    sw = np.sqrt(w)[:, None]
    U, s, Vt = np.linalg.svd(sw * (Ftr - mx), full_matrices=False)
    Uy = U.T @ (np.sqrt(w) * (ytr - my))
    P = (Fte - mx) @ Vt.T
    return {a: P @ (s / (s ** 2 + a) * Uy) + my for a in alphas}


def sparse_ridge_path(Xtr, ytr, Xte, w=None, alphas=(0.3, 1, 3, 10, 30)):
    from sklearn.linear_model import Ridge
    return {a: Ridge(alpha=a).fit(Xtr, ytr, sample_weight=w).predict(Xte) for a in alphas}


def knn_path(Ftr, ytr, Fte, ks=KS):
    S = Fte @ Ftr.T
    kmax = min(max(ks), Ftr.shape[0])
    top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
    order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
    top = np.take_along_axis(top, order, 1)
    Y = ytr[top]
    cs = np.cumsum(Y, 1)
    return {k: cs[:, min(k, kmax) - 1] / min(k, kmax) for k in ks}


def r_(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pearsonr(a, b)[0])


def scores(pred, z):
    r = r_(pred, z)
    return dict(n=len(z), pearson=r, spearman=float(spearmanr(pred, z)[0]),
                r2_best_linear=r * r,
                r2_as_predicted=float(1 - np.mean((z - pred) ** 2) / np.var(z)))


def pick(path_by_fold):
    """path_by_fold: list of (dict hp -> pred, truth). Highest mean Pearson."""
    hps = list(path_by_fold[0][0].keys())
    m = {h: np.mean([r_(p[h], t) for p, t in path_by_fold]) for h in hps}
    return max(hps, key=lambda h: m[h]), m


# -------------------------------------------------------------------- experiments

def lobo(df, feats, learner, weights=None):
    """feats(tr_idx, te_idx) -> (Ftr, Fte). Returns per held-out results and preds."""
    out, preds = {}, np.full(len(df), np.nan)
    bench = df.bench.values
    z = df.z.values
    for hb in BENCH:
        tr, te = np.flatnonzero(bench != hb), np.flatnonzero(bench == hb)
        Ftr_all, Fte = feats(tr, te)
        # nested: leave one training benchmark out, on outer-training features
        inner = []
        for ib in [x for x in BENCH if x != hb]:
            itr = np.flatnonzero(bench[tr] != ib); ite = np.flatnonzero(bench[tr] == ib)
            w = None if weights is None else weights(tr[itr])
            p = learner(Ftr_all[itr], z[tr][itr], Ftr_all[ite], w)
            inner.append((p, z[tr][ite]))
        hp, _ = pick(inner)
        w = None if weights is None else weights(tr)
        path = learner(Ftr_all, z[tr], Fte, w)
        oracle = max(path, key=lambda h: r_(path[h], z[te]))
        res = scores(path[hp], z[te])
        res.update(hp=hp, oracle_hp=oracle, oracle_pearson=r_(path[oracle], z[te]))
        # within-group view of the transferred prediction
        g = df.group.values[te]
        pc = path[hp] - pd.Series(path[hp]).groupby(g).transform("mean").values
        zc = z[te] - pd.Series(z[te]).groupby(g).transform("mean").values
        big = pd.Series(g).map(pd.Series(g).value_counts()).values >= 3
        res["pearson_within_group"] = r_(pc[big], zc[big])
        # text-bearing items only
        nd = ~df.degenerate.values[te]
        res["pearson_text_bearing"] = r_(path[hp][nd], z[te][nd])
        # cluster bootstrap over groups for the Pearson
        rng = np.random.default_rng(0)
        ug = np.unique(g); idx_of = {u: np.flatnonzero(g == u) for u in ug}
        bs = []
        for _ in range(1000):
            ii = np.concatenate([idx_of[u] for u in rng.choice(ug, len(ug))])
            bs.append(r_(path[hp][ii], z[te][ii]))
        res["pearson_ci_groupboot"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[hb] = res
        preds[te] = path[hp]
    return out, preds


def within(df, feats, learner, sub_target=None, splitter="kfold"):
    """5-fold CV inside each benchmark (nested hp by inner 5-fold)."""
    out = {}
    for hb in BENCH:
        idx = np.flatnonzero(df.bench.values == hb)
        z = df.z.values[idx] if sub_target is None else sub_target[idx]
        g = df.group.values[idx]
        valid = np.isfinite(z)
        idx, z, g = idx[valid], z[valid], g[valid]
        pred = np.zeros(len(idx))
        if splitter == "group":
            ng = len(np.unique(g))
            if ng < 3:
                out[hb] = None
                continue
            folds = list(GroupKFold(min(5, ng)).split(idx, groups=g))
        else:
            folds = list(KFold(5, shuffle=True, random_state=0).split(idx))
        hps = []
        for tr, te in folds:
            Ftr, Fte = feats(idx[tr], idx[te])
            inner = []
            for itr, ite in KFold(5, shuffle=True, random_state=1).split(tr):
                inner.append((learner(Ftr[itr], z[tr][itr], Ftr[ite], None), z[tr][ite]))
            hp, _ = pick(inner)
            hps.append(hp)
            pred[te] = learner(Ftr, z[tr], Fte, None)[hp]
        res = scores(pred, z)
        res["hps"] = hps
        out[hb] = res
    return out


def group_only(df, splitter="kfold"):
    """Predict z by the training-fold mean of the item's group (unseen group -> 0)."""
    out = {}
    for hb in BENCH:
        d = df[df.bench == hb].reset_index(drop=True)
        z, g = d.z.values, d.group.values
        eta2 = float(((d.groupby("group").z.transform("mean") - z.mean()) ** 2).sum()
                     / ((z - z.mean()) ** 2).sum())
        pred = np.zeros(len(d))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(d):
            m = pd.Series(z[tr]).groupby(g[tr]).mean()
            pred[te] = pd.Series(g[te]).map(m).fillna(0.0).values
        res = scores(pred, z)
        res.update(eta2_in_sample=eta2, n_groups=int(d.group.nunique()))
        out[hb] = res
    return out


# ------------------------------------------------------------------ Brier ceiling

def fit_th(pair, y, n_pairs, offset, lam=1.0, free_slope_x=None):
    """Logistic y ~ pair effect (+ s * x) + offset, ridge lam on the pair effects."""
    has_s = free_slope_x is not None

    def obj(p):
        th = p[:n_pairs]
        eta = th[pair] + offset + (p[-1] * free_slope_x if has_s else 0.0)
        pr = 1 / (1 + np.exp(-eta))
        nll = -np.sum(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12)) + 0.5 * lam * th @ th
        r = y - pr
        g = np.bincount(pair, weights=-r, minlength=n_pairs) + lam * th
        if has_s:
            g = np.append(g, -np.sum(r * free_slope_x))
        return nll, g

    p0 = np.zeros(n_pairs + (1 if has_s else 0))
    res = minimize(obj, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000))
    th = res.x[:n_pairs]
    s = res.x[-1] if has_s else 0.0
    eta = th[pair] + offset + (s * free_slope_x if has_s else 0.0)
    return 1 / (1 + np.exp(-eta)), s


def pair_brier(pair, y, p):
    b = (y - p) ** 2
    return float(np.mean(np.bincount(pair, weights=b) / np.bincount(pair)))


def brier_ceiling(df, R, preds_by_model):
    out = {}
    slopes = {m: {} for m in preds_by_model}
    std = {}
    for m, preds in preds_by_model.items():
        for hb in BENCH:
            te = np.flatnonzero(df.bench.values == hb)
            x = preds[te]
            std[(m, hb)] = (x - x.mean()) / x.std()
    for hb in BENCH:
        r = R[hb]
        n_pairs = r["pair"].max() + 1
        y = r["y"]
        base, _ = fit_th(r["pair"], y, n_pairs, 0.0)
        oracle = 1 / (1 + np.exp(-(r["th"] - r["bi"])))
        res = dict(pair_only=pair_brier(r["pair"], y, base),
                   item_oracle_rasch_in_sample=pair_brier(r["pair"], y, oracle))
        for m in preds_by_model:
            x = std[(m, hb)][r["row"]]
            p, s = fit_th(r["pair"], y, n_pairs, 0.0, free_slope_x=x)
            res[f"{m}_oracle_slope"] = pair_brier(r["pair"], y, p)
            res[f"{m}_slope"] = float(s)
            slopes[m][hb] = float(s)
        out[hb] = res
    # fixed slope transferred from the other three benchmarks (honest LOBO)
    for hb in BENCH:
        r = R[hb]
        n_pairs = r["pair"].max() + 1
        for m in preds_by_model:
            s_fix = float(np.mean([slopes[m][o] for o in BENCH if o != hb]))
            x = std[(m, hb)][r["row"]]
            p, _ = fit_th(r["pair"], r["y"], n_pairs, s_fix * x)
            out[hb][f"{m}_transferred_slope"] = pair_brier(r["pair"], r["y"], p)
            out[hb][f"{m}_transferred_slope_value"] = s_fix
    return out


# --------------------------------------------------------------------------- main

def main():
    dfs, Es, R, info = [], [], {}, {}
    for b in BENCH:
        d, E, r, i = load_bench(b)
        dfs.append(d); Es.append(E); R[b] = r; info[b] = i
    df = pd.concat(dfs, ignore_index=True)
    E = np.vstack(Es); del Es
    bench = df.bench.values
    Ec = centre_rows(E, bench)                      # benchmark mean embedding removed
    gl = (df.bench + "|" + df.group).values
    Eg = centre_rows(E, gl)                         # group mean removed
    tf = Tfidf(df.text.tolist())
    log("hashed tfidf", tf.H.shape, tf.H.nnz, "maxrss MB",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)

    ridge = lambda Ftr, ytr, Fte, w: ridge_path(Ftr, ytr, Fte, w)
    knn = lambda Ftr, ytr, Fte, w: knn_path(Ftr, ytr, Fte)
    emb_raw = lambda tr, te: (E[tr], E[te])
    emb_c = lambda tr, te: (Ec[tr], Ec[te])
    emb_g = lambda tr, te: (Eg[tr], Eg[te])
    L = df.loglen.values[:, None].astype(np.float32)
    loglen = lambda tr, te: (L[tr], L[te])
    loglen_c = df.loglen - df.groupby("bench").loglen.transform("mean")
    Lc = loglen_c.values[:, None].astype(np.float32)
    loglen_cf = lambda tr, te: (Lc[tr], Lc[te])

    def tfidf_c(tr, te):
        Ftr, Fte = tf.fit_transform(tr, te)
        F = np.empty((len(df), Ftr.shape[1]), np.float32); F[tr] = Ftr; F[te] = Fte
        # centre per benchmark on the SVD output (unsupervised; test uses its own items)
        Fc = centre_rows(F, bench)
        return Fc[tr], Fc[te]

    def tfidf_raw(tr, te):
        return tf.fit_transform(tr, te)

    tfc = Tfidf(df.content.tolist())            # content only, as experiments/transfer.py
    sridge = lambda Ftr, ytr, Fte, w: sparse_ridge_path(Ftr, ytr, Fte, w)
    tfidf_content_sparse = lambda tr, te: tfc.fit_transform(tr, te, n_comp=None)
    nb = df.bench.value_counts()
    balanced = lambda idx: 1.0 / nb[bench[idx]].values

    results = {"info": info, "lobo": {}, "within": {}, "within_group_resid": {},
               "within_groupcv": {}, "group_only": {}, "brier": {}}
    lobo_specs = {
        "emb_ridge_raw": (emb_raw, ridge, None),
        "emb_ridge_centred": (emb_c, ridge, None),
        "emb_ridge_centred_balanced": (emb_c, ridge, balanced),
        "emb_knn_raw": (emb_raw, knn, None),
        "emb_knn_centred": (emb_c, knn, None),
        "tfidf_ridge_raw": (tfidf_raw, ridge, None),
        "tfidf_ridge_centred": (tfidf_c, ridge, None),
        "tfidf_knn_centred": (tfidf_c, knn, None),
        "tfidf_content_sparse_ridge": (tfidf_content_sparse, sridge, None),
        "loglen_ridge_raw": (loglen, ridge, None),
        "loglen_ridge_centred": (loglen_cf, ridge, None),
    }
    lobo_preds = {}
    for name, (f, l, w) in lobo_specs.items():
        results["lobo"][name], lobo_preds[name] = lobo(df, f, l, w)
        log("LOBO", name, {b: round(v["pearson"], 3) for b, v in results["lobo"][name].items()})

    within_specs = {
        "emb_ridge": (emb_raw, ridge),
        "emb_knn": (emb_raw, knn),
        "tfidf_ridge": (tfidf_raw, ridge),
        "tfidf_knn": (tfidf_raw, knn),
        "loglen_ridge": (loglen, ridge),
    }
    for name, (f, l) in within_specs.items():
        results["within"][name] = within(df, f, l)
        log("within", name, {b: round(v["pearson"], 3) for b, v in results["within"][name].items()})

    # within-group: target = z less its group mean (groups >= 3 items), features
    # centred within group (embedding) so the group's identity is gone
    gsize = df.groupby(gl).z.transform("size").values
    zres = (df.z - df.groupby(gl).z.transform("mean")).values
    zres[gsize < 3] = np.nan

    def tfidf_g(tr, te):
        Ftr, Fte = tf.fit_transform(tr, te)
        F = np.empty((len(df), Ftr.shape[1]), np.float32); F[tr] = Ftr; F[te] = Fte
        Fc = centre_rows(F, gl)
        return Fc[tr], Fc[te]

    Lg = (df.loglen - df.groupby(gl).loglen.transform("mean")).values[:, None].astype(np.float32)
    for name, (f, l) in {"emb_ridge": (emb_g, ridge), "emb_knn": (emb_g, knn),
                         "tfidf_ridge": (tfidf_g, ridge),
                         "loglen_ridge": (lambda tr, te: (Lg[tr], Lg[te]), ridge)}.items():
        results["within_group_resid"][name] = within(df, f, l, sub_target=zres)
        log("within-group resid", name,
            {b: (round(v["pearson"], 3) if v else None)
             for b, v in results["within_group_resid"][name].items()})
    # leave groups out inside a benchmark (transfer across groups, raw target)
    for name, (f, l) in {"emb_ridge": (emb_raw, ridge), "emb_knn": (emb_raw, knn),
                         "tfidf_ridge": (tfidf_raw, ridge)}.items():
        results["within_groupcv"][name] = within(df, f, l, splitter="group")
        log("within group-CV", name,
            {b: (round(v["pearson"], 3) if v else None)
             for b, v in results["within_groupcv"][name].items()})
    results["group_only"] = group_only(df)
    log("group only", {b: round(v["pearson"], 3) for b, v in results["group_only"].items()})

    # Brier ceiling of a covariate, per held-out benchmark
    bm = {"emb_ridge_centred": lobo_preds["emb_ridge_centred"],
          "emb_knn_centred": lobo_preds["emb_knn_centred"],
          "tfidf_ridge_centred": lobo_preds["tfidf_ridge_centred"]}
    results["brier"] = brier_ceiling(df, R, bm)
    log("brier", json.dumps(results["brier"], indent=1))
    results["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    results["seconds"] = time.time() - T0
    with open(os.path.join(ROOT, "results", "emb_transfer.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    log("done; maxrss MB", results["maxrss_MB"])


if __name__ == "__main__":
    main()
