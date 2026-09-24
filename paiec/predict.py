"""Runtime predictor: the configuration that scored best in the replica.

    p = c + (1 - c - slip) * sigmoid(kappa * (a + b*z))

    c      guessing floor read off the item text, 0 when the item is not
           multiple choice
    z      item difficulty from a joint IRT fitted on the visible labels, with
           the item text as a prior on difficulty. Only a benchmark with
           BenchmarkFit.WARMUP (64) distinct labeled items gets one; below that
           z is 0, so in a run shaped like the site's example (5 pairs over 4
           benchmarks) it never switches on
    a      the subject's standing; its prior comes from the subject's
           attributes, which is the only thing measured to transfer between
           benchmarks. The benchmark's level does not transfer and is left
           entirely to the labels.
    b      how steeply this subject responds to difficulty, fitted per pair
    kappa  Laplace shrinkage over the posterior of (a, b)

The official evaluator settles what this used to leave open. At a checkpoint
every target receives the same `labeled` list, holding every label revealed so
far for every sampled subject and benchmark, and the evaluation workers are
recreated at every checkpoint. So the predictor is a pure function of
(input, labeled): everything is fitted from `labeled` alone and cached under a
fingerprint of its content, which lets a worker serving many targets fit once
and keeps the answer independent of which targets came first. The previous
version accumulated texts and labels across calls and fitted its embedding on
whichever 64 texts arrived first.

Items and subjects are keyed on everything visible about them, because the
official input carries no ids. A 400-character prefix is not enough: every
researchcodebench item shares one, and most matharena items share one with
another item.

Imports within the package are relative, because the archive ships these
modules as paiec_rt (tools/build_submission.py): a module called paiec that the
platform's process had already imported would otherwise stand in for them.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import numbers
import sys
import threading
import traceback
from collections import OrderedDict

import numpy as np

from .fitting import fit_ab, sig
from .irt import TextEmbedder, fit_joint
from .mcq import floor_of
from .subjects import Spec, attrs, design_matrix, design_row, subject_frame

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")
LO, HI = 1e-4, 1 - 1e-4


def _text(v) -> str:
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


_keys: dict[tuple, bytes] = {}
_key_chars = 0
KEY_BUDGET = 1 << 27        # characters the key memo may keep alive


def _key(d: dict, fields) -> bytes:
    """blake2b over the fields, computed once per distinct content.

    Every labeled entry carries its item's full text, up to 150k characters,
    and the whole list is re-read on every call. The memo is keyed on the field
    values themselves: Python caches a string's hash on the object, so looking
    up an entry seen before costs a tuple, and a copy that shares no objects
    with the original still finds it by equality.
    """
    global _key_chars
    raw = tuple([d.get(f) for f in fields])
    try:
        return _keys[raw]
    except (KeyError, TypeError):
        pass
    vals = [_text(v) for v in raw]
    h = hashlib.blake2b(digest_size=16)
    for v in vals:
        h.update(hashlib.blake2b(v.encode("utf-8", "surrogatepass"), digest_size=16).digest())
    k = h.digest()
    if all(v is None or isinstance(v, str) for v in raw):     # 1 == True, "1" != "True"
        n = sum(map(len, vals))
        if _key_chars + n > KEY_BUDGET:
            _keys.clear()
            _key_chars = 0
        _keys[raw] = k
        _key_chars += n
    return k


def item_key(item: dict) -> bytes:
    """The full text, features, interactors and benchmark, so only items the
    evaluator itself could not tell apart share a key."""
    return _key(item, ITEM_FIELDS)


def subject_key(subject: dict) -> bytes:
    return _key(subject, SUBJECT_FIELDS)


def records(labeled) -> list[tuple]:
    """Well-formed entries as (benchmark, subject key, item key, label, subject, item).

    Sorted, so everything built from them is a function of the labels and not of
    the order they were listed in. An entry counts only as [[subject dict, item
    dict], label] with a numeric label of 0 or 1 ("1" is not one), and anything
    else is dropped on its own, so one malformed entry cannot cost the others.
    """
    if labeled is None or isinstance(labeled, (str, bytes, dict)):
        return []
    try:
        entries = list(labeled)
    except Exception:
        return []
    out = []
    for entry in entries:
        try:
            (s, i), y = entry
            if isinstance(s, dict) and isinstance(i, dict) and isinstance(y, numbers.Real) \
                    and y in (0, 1):
                bid = _text(i.get("benchmark_id"))
                out.append((bid, subject_key(s), item_key(i), int(y), s, i))
        except Exception:
            continue
    out.sort(key=lambda r: r[:4])
    return out


def fingerprint(recs) -> bytes:
    """Content of the sorted records; the item key already covers the benchmark."""
    h = hashlib.blake2b(digest_size=16)
    for _, sk, ik, y, _, _ in recs:
        h.update(sk + ik + (b"1" if y else b"0"))
    return h.digest()


class BenchmarkFit:
    """Item difficulty for one benchmark, fitted on its labeled entries only.

    The text embedding is fitted on the labeled items' texts once there are
    WARMUP of them. A target among the labeled items takes its fitted
    difficulty; any other takes the text map's, with no residual. Anything that
    goes wrong here, from a missing scipy or scikit-learn to texts TF-IDF cannot
    use, leaves every difficulty at zero, which reduces the model to the
    attribute prior plus the subject's own labels.
    """

    WARMUP = 64
    TEXT_CAP = 20_000

    def __init__(self, recs, dim=64, lam_t=1.0, lam_w=64.0, lam_e=1.0):
        self.index, texts, subjects = {}, [], {}
        for r in recs:
            if r[2] not in self.index:
                self.index[r[2]] = len(texts)
                texts.append(self.excerpt(_text(r[5].get("item_content"))))
        obs = sorted({(subjects.setdefault(r[1], len(subjects)), self.index[r[2]], r[3])
                      for r in recs})
        self.lam_e = lam_e
        self.z = np.zeros(len(texts))
        self.w = None
        self._z = {}
        if len(obs) < 8 or len({o[2] for o in obs}) < 2 or len(texts) < max(4, self.WARMUP):
            return
        try:
            with _one_thread():
                self._fit(texts, obs, len(subjects), dim, lam_t, lam_w, lam_e)
        except Exception:
            self.z, self.w = np.zeros(len(texts)), None

    def _fit(self, texts, obs, n_s, dim, lam_t, lam_w, lam_e):
        embedder = TextEmbedder(dim)
        X = embedder.fit(texts)
        si, ji, y = (np.array(c) for c in zip(*obs))
        n_i = len(texts)
        th, w, eps = fit_joint(si, ji, y.astype(float), X, n_s, n_i, lam_t, lam_w, lam_e)
        b = X @ w + eps
        s = np.clip(sig(th[si] - b[ji]), 1e-6, 1 - 1e-6)
        s = s * (1 - s)
        Xo, d = X[ji], X.shape[1]
        Sw = np.linalg.inv(lam_w * np.eye(d) + Xo.T @ (Xo * s[:, None]) + 1e-9 * np.eye(d))
        veps = 1.0 / (lam_e + np.bincount(ji, weights=s, minlength=n_i))
        vb = np.einsum("ij,jk,ik->i", X, Sw, X) + veps
        z = -(b - b.mean()) / np.sqrt(1 + (np.pi / 8) * vb)
        if np.all(np.isfinite(z)) and np.all(np.isfinite(w)) and np.all(np.isfinite(Sw)):
            self.embedder, self.w, self.Sw, self.bmean, self.z = embedder, w, Sw, float(b.mean()), z

    @classmethod
    def excerpt(cls, text):
        """What the embedding reads of an item: all of it up to TEXT_CAP
        characters, else the first and the last TEXT_CAP / 2.

        TF-IDF and SVD over researchcodebench's 100k-character items were most
        of a worker's first call, and items from one paper differ only in their
        last few thousand characters. Under paiec.official the excerpt scored
        as the full text did, within noise, on 12-pair researchcodebench runs
        and on runs mixing benchmarks, in 2.5x less time on the former; a cap
        of 8k lost ALC.
        """
        if not cls.TEXT_CAP or len(text) <= cls.TEXT_CAP:
            return text
        half = cls.TEXT_CAP // 2
        return text[:half] + "\n" + text[-half:]

    def z_of(self, key, text) -> float:
        j = self.index.get(key)
        if j is not None:
            return float(self.z[j])
        if self.w is None:
            return 0.0
        if key not in self._z:
            try:
                x = self.embedder.transform([self.excerpt(text)])[0]
                v = float(x @ self.Sw @ x) + 1.0 / self.lam_e
                z = -(float(x @ self.w) - self.bmean) / math.sqrt(1 + (math.pi / 8) * v)
            except Exception:
                z = 0.0
            self._z[key] = z if math.isfinite(z) else 0.0
        return self._z[key]


def _one_thread():
    """BLAS threads contending over the fit's small SVD made it 10-20x slower in
    local tests, and the platform runs up to 16 workers side by side.

    threadpoolctl limits only the libraries already loaded, and scipy's BLAS and
    scikit-learn's OpenMP load with the modules the fit imports, so they are
    imported first. Imported after the limit, they ran the first fit of every
    worker, which is most fits since workers are recreated at every checkpoint,
    on a thread per core: 4x slower with 8 workers on 8 cores, and not the fit
    the same labels got later in the process (one budget's Brier on 12
    matharena pairs moved by 0.002), so a prediction depended on which
    benchmark happened to be fitted first.
    """
    try:
        import scipy.linalg, scipy.optimize  # noqa: F401,E401
        import sklearn.decomposition, sklearn.preprocessing  # noqa: F401,E401
        import sklearn.feature_extraction.text  # noqa: F401
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except Exception:
        return contextlib.nullcontext()


class Evidence:
    """One `labeled` list, with its fits built lazily per benchmark and pair."""

    def __init__(self, recs, dim):
        self.dim = dim
        self.by_benchmark: dict[str, list] = {}
        for r in recs:
            self.by_benchmark.setdefault(r[0], []).append(r)
        self._fits: dict[str, BenchmarkFit] = {}
        self.ab: dict[tuple, tuple] = {}

    def fit(self, bid) -> BenchmarkFit:
        if bid not in self._fits:
            self._fits[bid] = BenchmarkFit(self.by_benchmark.get(bid, []), self.dim)
        return self._fits[bid]


class Predictor:
    def __init__(self, prior_coef=None, prior_spec=None, v_a=2.0, v_b=0.25,
                 slip=0.02, dim=64, keep=4):
        self.prior_coef = prior_coef
        self.prior_spec = prior_spec
        self.v_a, self.v_b, self.slip, self.dim = v_a, v_b, slip, dim
        self.keep = keep                    # labeled lists whose fits stay cached
        self._evidence: OrderedDict[bytes, Evidence] = OrderedDict()
        self._prior: dict[bytes, float] = {}
        self._lock = threading.Lock()
        self.failures = 0

    def prior_mean(self, subject):
        if self.prior_coef is None:
            return 0.0
        m = float(np.dot(self.prior_coef, design_row(attrs(subject), self.prior_spec)))
        return m if math.isfinite(m) else 0.0

    def predict(self, input, labeled=None):
        """P(correct) for input = [subject, item], a native float in [1e-4, 1 - 1e-4].

        Never raises. When the model fails, the answer falls back to the
        attribute prior (the model's own answer with no labels and zero
        difficulty), and from there to 0.5; `failures` counts the fallbacks and
        the first one's traceback goes to stderr.
        """
        with self._lock:
            try:
                p = _probability(self._predict(input, labeled))
                if p is None:
                    raise FloatingPointError("non-finite probability")
                return p
            except Exception:
                self.failures += 1
                if self.failures == 1:
                    traceback.print_exc(file=sys.stderr)
                return self._fallback(input)

    def _predict(self, input, labeled):
        subject, item = input
        text = _text(item.get("item_content"))
        bid = _text(item.get("benchmark_id"))
        ev = self._evidence_for(labeled)
        z = ev.fit(bid).z_of(item_key(item), text)
        a, b, va, vb = self._ab(ev, bid, subject)
        k = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * (va + vb * z * z))
        c = floor_of(text)
        return c + (1 - c - self.slip) * sig(k * (a + b * z))

    def _evidence_for(self, labeled) -> Evidence:
        recs = records(labeled)
        fp = fingerprint(recs)
        if fp in self._evidence:
            self._evidence.move_to_end(fp)
        else:
            self._evidence[fp] = Evidence(recs, self.dim)
            while len(self._evidence) > self.keep:
                self._evidence.popitem(last=False)
        return self._evidence[fp]

    def _ab(self, ev, bid, subject):
        sid = subject_key(subject)
        if (bid, sid) not in ev.ab:
            fit = ev.fit(bid)
            own = [(fit.z[fit.index[r[2]]], r[3]) for r in ev.by_benchmark.get(bid, ())
                   if r[1] == sid]
            Z = np.array([z for z, _ in own])
            Y = np.array([y for _, y in own], float)
            m = self._prior_of(sid, subject)
            ab = fit_ab(Z, Y, m, self.v_a, 1.0, self.v_b)
            ev.ab[(bid, sid)] = ab if all(map(math.isfinite, ab)) else (m, 1.0, self.v_a, self.v_b)
        return ev.ab[(bid, sid)]

    def _prior_of(self, sid, subject):
        if sid not in self._prior:
            if len(self._prior) > 100_000:
                self._prior.clear()
            self._prior[sid] = self.prior_mean(subject)
        return self._prior[sid]

    def _fallback(self, input):
        try:
            subject, item = input
            c = floor_of(_text(item.get("item_content")))
            k = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * self.v_a)
            p = _probability(c + (1 - c - self.slip) * sig(k * self.prior_mean(subject)))
        except Exception:
            p = None
        return 0.5 if p is None else p


def _probability(v):
    """A native float in [LO, HI], or None when v is not a finite number."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return min(max(v, LO), HI) if math.isfinite(v) else None


def fit_prior(pairs, alpha=2.0):
    """Fit the attribute model on every pair available; returns (coef, spec)."""
    from sklearn.linear_model import Ridge
    df = subject_frame(pairs)
    spec = Spec.from_frame(df)
    m = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix(df, spec), df.z.values)
    return m.coef_.copy(), spec


def make(pairs=None, **kw):
    """A predictor ready to hand to the evaluator as `predict(input, labeled)`."""
    coef, spec = (None, None) if pairs is None else fit_prior(pairs)
    return Predictor(coef, spec, **kw).predict
