"""Runtime predictor: the configuration that scored best in the replica.

    p = c + (1 - c - slip) * sigmoid(kappa * (a + b*z))

    c      guessing floor read off the item text, 0 when the item is not
           multiple choice
    z      item difficulty from a joint IRT fitted at run time on whatever
           labels are visible, with the item text as a prior on difficulty
    a      the subject's standing; its prior comes from the subject's
           attributes, which is the only thing measured to transfer between
           benchmarks. The benchmark's level does not transfer and is left
           entirely to the labels.
    b      how steeply this subject responds to difficulty, fitted per pair
    kappa  Laplace shrinkage over the posterior of (a, b)

Two things this depends on, both open questions with the organisers:

  * whether `labeled` carries labels from other subjects. With them the joint
    IRT has something to fit and the replica scores 0.181; without them z stays
    near zero and it scores 0.204.
  * whether state may persist between calls. The difficulty model needs the
    item texts it has seen, which arrive one call at a time.

`stateful=False` disables the accumulation and uses only what a single call
receives, which is the conservative reading of the rules.
"""
from __future__ import annotations

import numpy as np

from paiec.fitting import fit_ab, sig
from paiec.irt import TextEmbedder, fit_joint
from paiec.mcq import floor_of
from paiec.subjects import Spec, attrs, design_row


class BenchmarkState:
    """Item texts and labels seen so far for one benchmark."""

    def __init__(self, dim=64):
        self.dim = dim
        self.keys: dict[str, int] = {}
        self.texts: list[str] = []
        self.subjects: dict[str, int] = {}
        self.obs: list[tuple[int, int, int]] = []       # subject idx, item idx, label
        self._seen: set = set()
        self._fit_at = -1
        self._z = None
        self._embedder = TextEmbedder(dim)
        self._emb = None
        self._w = None          # text-to-difficulty map from the last joint fit
        self._Sw = None         # its posterior covariance
        self._bmean = 0.0

    def item(self, key, text):
        if key not in self.keys:
            self.keys[key] = len(self.texts)
            self.texts.append(text or "")
        return self.keys[key]

    WARMUP = 64

    def _embeddings(self):
        """Rows for every text seen. Fitted once at WARMUP, extended after."""
        n = len(self.texts)
        if not self._embedder.fitted:
            if n < self.WARMUP:
                return None
            self._emb = self._embedder.fit(self.texts)
            return self._emb
        if len(self._emb) < n:
            self._emb = np.vstack([self._emb,
                                   self._embedder.transform(self.texts[len(self._emb):])])
        return self._emb

    def observe(self, subject_key, item_idx, label):
        if subject_key not in self.subjects:
            self.subjects[subject_key] = len(self.subjects)
        rec = (self.subjects[subject_key], item_idx, int(label))
        if rec not in self._seen:
            self._seen.add(rec)
            self.obs.append(rec)

    def difficulty(self, lam_t=1.0, lam_w=64.0, lam_e=1.0):
        """Standardised, variance-shrunk difficulty for every item seen.

        Items that turn up after the last joint fit take their difficulty from
        the text map alone, with no residual, instead of forcing a refit.
        Refitting on every newly seen evaluation item cost more than the whole
        rest of the pipeline, and an unlabelled item's difficulty is w . x anyway.
        """
        n_i, n_s = len(self.texts), max(len(self.subjects), 1)
        if self._z is not None and len(self.obs) < 1.15 * self._fit_at + 6:
            return self._z if len(self._z) >= n_i else self._extend(n_i, lam_e)
        if len(self.obs) < 8 or len({o[2] for o in self.obs}) < 2 or n_i < 4:
            self._z, self._fit_at = np.zeros(n_i), len(self.obs)
            return self._z
        X = self._embeddings()
        if X is None:
            self._z, self._fit_at = np.zeros(n_i), len(self.obs)
            return self._z
        si = np.array([o[0] for o in self.obs])
        ji = np.array([o[1] for o in self.obs])
        y = np.array([o[2] for o in self.obs], float)
        th, w, eps = fit_joint(si, ji, y, X, n_s, n_i, lam_t, lam_w, lam_e)
        b = X @ w + eps
        s = np.clip(sig(th[si] - b[ji]), 1e-6, 1 - 1e-6)
        s = s * (1 - s)
        Xo, d = X[ji], X.shape[1]
        Sw = np.linalg.inv(lam_w * np.eye(d) + Xo.T @ (Xo * s[:, None]) + 1e-9 * np.eye(d))
        veps = 1.0 / (lam_e + np.bincount(ji, weights=s, minlength=n_i))
        vb = np.einsum("ij,jk,ik->i", X, Sw, X) + veps
        self._w, self._Sw, self._bmean = w, Sw, float(b.mean())
        self._z = -(b - b.mean()) / np.sqrt(1 + (np.pi / 8) * vb)
        self._fit_at = len(self.obs)
        return self._z

    def _extend(self, n_i, lam_e):
        """Difficulty for items seen since the last fit, from the text map only."""
        X = self._embeddings() if self._w is not None else None
        if X is None or len(X) < n_i:
            self._z = np.concatenate([self._z, np.zeros(n_i - len(self._z))])
            return self._z
        Xn = X[len(self._z):n_i]
        bn = Xn @ self._w
        vbn = np.einsum("ij,jk,ik->i", Xn, self._Sw, Xn) + 1.0 / lam_e
        self._z = np.concatenate([self._z, -(bn - self._bmean) / np.sqrt(1 + (np.pi / 8) * vbn)])
        return self._z


class Predictor:
    def __init__(self, prior_coef=None, prior_spec=None, v_a=2.0, v_b=0.25,
                 slip=0.02, dim=64, stateful=True):
        self.prior_coef = prior_coef
        self.prior_spec = prior_spec
        self.v_a, self.v_b, self.slip, self.dim = v_a, v_b, slip, dim
        self.stateful = stateful
        self.states: dict[str, BenchmarkState] = {}
        self._ab: dict = {}

    def prior_mean(self, subject):
        if self.prior_coef is None:
            return 0.0
        return float(np.dot(self.prior_coef, design_row(attrs(subject), self.prior_spec)))

    def _state(self, bid):
        if bid not in self.states:
            self.states[bid] = BenchmarkState(self.dim)
        return self.states[bid]

    def predict(self, input, labeled=None):
        subject, item = input
        bid = item.get("benchmark_id") or ""
        sid = _subject_key(subject)
        st = self._state(bid) if self.stateful else BenchmarkState(self.dim)
        tgt = st.item(_item_key(item), item.get("item_content", ""))
        own = []
        for (osub, oitem), y in (labeled or []):
            if (oitem.get("benchmark_id") or "") != bid:
                continue
            idx = st.item(_item_key(oitem), oitem.get("item_content", ""))
            st.observe(_subject_key(osub), idx, y)
            if _subject_key(osub) == sid:
                own.append((idx, int(y)))
        z = st.difficulty()
        key = (sid, bid, tuple(sorted(own)), st._fit_at)
        if key not in self._ab:
            Z = np.array([z[i] for i, _ in own]) if own else np.zeros(0)
            Y = np.array([v for _, v in own], float)
            self._ab[key] = fit_ab(Z, Y, self.prior_mean(subject), self.v_a, 1.0, self.v_b)
        a, b, va, vb = self._ab[key]
        zt = float(z[tgt])
        k = 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * (va + vb * zt * zt))
        c = floor_of(item.get("item_content", ""))
        return float(np.clip(c + (1 - c - self.slip) * sig(k * (a + b * zt)), 1e-4, 1 - 1e-4))


def _subject_key(s):
    return s.get("_sid") or "\x1f".join(str(s.get(k, "")) for k in sorted(s))


def _item_key(i):
    return i.get("_key") or (i.get("item_content", "") or "")[:400]


def fit_prior(pairs, alpha=2.0):
    """Fit the attribute model on every pair available; returns (coef, spec)."""
    from sklearn.linear_model import Ridge
    from paiec.subjects import Spec, design_matrix, subject_frame
    df = subject_frame(pairs)
    spec = Spec.from_frame(df)
    m = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix(df, spec), df.z.values)
    return m.coef_.copy(), spec


def make(pairs=None, **kw):
    """A predictor ready to hand to the evaluator as `predict(input, labeled)`."""
    coef, spec = (None, None) if pairs is None else fit_prior(pairs)
    return Predictor(coef, spec, **kw).predict
