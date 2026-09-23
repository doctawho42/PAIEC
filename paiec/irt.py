"""Joint IRT on pooled acquired labels, with the item text as a prior on difficulty.

  P(y = 1) = sigmoid(theta_s - b_j),   b_j = w . x_j + eps_j

theta_s is the subject's ability, x_j a low-dimensional text embedding of the item,
w the text-to-difficulty map learned for THIS benchmark at run time, and eps_j a
shrunk per-item residual. Items nobody labelled fall back on w . x_j; items with
labels get a residual on top. Subject ability is estimated at the same time, which
is what the pooled logistic regression could not do: there, a label from a strong
model and a label from a weak one counted the same.

MAP by L-BFGS. Parameters: n_subjects + dim + n_items.
"""
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from paiec import evaluator as E

sig = lambda x: 1 / (1 + np.exp(-x))


def text_embeddings(texts, dim=128, seed=0):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        strip_accents="unicode", max_features=200000)
    X = v.fit_transform(texts)
    d = min(dim, max(2, min(X.shape) - 1))
    Z = TruncatedSVD(d, random_state=seed).fit_transform(X)
    return StandardScaler().fit_transform(Z)


def fit_joint(si, ji, y, X, n_s, n_i, lam_t=1.0, lam_w=1.0, lam_e=5.0):
    d = X.shape[1]

    def obj(p):
        th, w, eps = p[:n_s], p[n_s:n_s + d], p[n_s + d:]
        b = X @ w + eps
        z = th[si] - b[ji]
        pr = sig(z)
        nll = -np.sum(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))
        nll += 0.5 * (lam_t * th @ th + lam_w * w @ w + lam_e * eps @ eps)
        r = y - pr
        gth = np.bincount(si, weights=-r, minlength=n_s) + lam_t * th
        gb = np.bincount(ji, weights=r, minlength=n_i)
        gw = X.T @ gb + lam_w * w
        geps = gb + lam_e * eps
        return nll, np.concatenate([gth, gw, geps])

    p0 = np.zeros(n_s + d + n_i)
    res = minimize(obj, p0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=400, maxfun=600))
    p = res.x
    return p[:n_s], p[n_s:n_s + d], p[n_s + d:]


def evaluate(pairs, traj, dim=128, lam_e=5.0, lam_t=1.0, lam_w=1.0, seed=0):
    by_b = defaultdict(list)
    for p in pairs:
        by_b[p.benchmark_id].append(p)
    texts, keys_of = {}, {}
    emb = {}
    for bid, ps in by_b.items():
        seen, order = set(), []
        for p in ps:
            for r in p.responses:
                if r.item_key not in seen:
                    seen.add(r.item_key); order.append(r.item_key)
                    texts[(bid, r.item_key)] = r.item["item_content"] or ""
        keys_of[bid] = order
        emb[bid] = text_embeddings([texts[(bid, k)] for k in order], dim=dim)
    out = {}
    for B in E.BUDGETS:
        se_all = []
        for bid, ps in by_b.items():
            kidx = {k: i for i, k in enumerate(keys_of[bid])}
            sidx = {(p.subject_id, p.benchmark_id): i for i, p in enumerate(ps)}
            si, ji, y = [], [], []
            for p in ps:
                for k, lab in traj[(p.subject_id, p.benchmark_id)][:B]:
                    si.append(sidx[(p.subject_id, p.benchmark_id)])
                    ji.append(kidx[k]); y.append(lab)
            X = emb[bid]
            if len(y) and len(set(y)) > 1:
                th, w, eps = fit_joint(np.array(si), np.array(ji), np.array(y, float),
                                       X, len(ps), len(X), lam_t, lam_w, lam_e)
                b = X @ w + eps
            else:
                th, b = np.zeros(len(ps)), np.zeros(len(X))
            for p in ps:
                t = th[sidx[(p.subject_id, p.benchmark_id)]]
                _, ev = E.split_pair(p, seed); evs = set(ev)
                se = n = 0.0
                for r in p.responses:
                    if r.item_key not in evs:
                        continue
                    pr = np.clip(sig(t - b[kidx[r.item_key]]), 1e-4, 1 - 1e-4)
                    se += (pr - r.label) ** 2; n += 1
                se_all.append(se / n)
        out[B] = float(np.mean(se_all))
    return out


class TextEmbedder:
    """Fit the text representation once, then only transform.

    The run-time predictor meets item texts one call at a time, and refitting
    TF-IDF plus SVD every time the corpus grows costs more than everything else
    in the pipeline put together. Fitting once on a warm-up corpus and
    transforming afterwards is both faster and the more honest choice: the
    representation stops depending on which items happen to have arrived.
    """

    def __init__(self, dim=64, seed=0):
        self.dim, self.seed = dim, seed
        self.vec = self.svd = self.scaler = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   strip_accents="unicode", max_features=200000)
        X = self.vec.fit_transform(texts)
        d = min(self.dim, max(2, min(X.shape) - 1))
        self.svd = TruncatedSVD(d, random_state=self.seed)
        Z = self.svd.fit_transform(X)
        self.scaler = StandardScaler().fit(Z)
        return self.scaler.transform(Z)

    def transform(self, texts):
        if self.vec is None:
            raise RuntimeError("fit first")
        return self.scaler.transform(self.svd.transform(self.vec.transform(texts)))

    @property
    def fitted(self):
        return self.vec is not None
