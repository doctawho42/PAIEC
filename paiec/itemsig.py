"""Label-conditional per-item correction on top of a base predictor.

    logit p = logit p_base + beta_eff * clip(sum_j w_ij r_j / (tau + sum_j w_ij v_j))

A formative run holds about one pair per benchmark, so the only item-level
evidence a target has at run time is its own pair's acquired labels (up to 31),
now and then another subject's on the same benchmark, and the item texts. The
base model (paiec.hier) reads the benchmark's level, the pair's standing and
item_features groups from those labels, and an item's own residual only where
the target's item itself carries labels. What it leaves is the residual r_j =
y_j - p_j of every labeled item j, and this layer carries those residuals to
unlabeled items with similar text, on the logit scale.

    j        a labeled record on the target's benchmark_id: its own pair's
             (weight 1) or another subject's (weight Hyper.other). Records on
             the target's own item (same item key) are left out: the base
             already reads them through that item's residual.
    p_j      what the base predicts for record j's labeled cell itself
             (resid 'leave_in', the default), or for j's subject on j's item
             taken as a NEW item (resid 'struct'): as_new() adds a group key
             no labeled item carries to the item's interactors (else its
             item_features), checked to leave every group hier reads
             (hier.item_groups) as it was, so the key is new while text,
             groups, benchmark and floor are unchanged. Where neither field
             can take the key that way, the record keeps its leave-in p_j.
    r_j      y_j - p_j, centred within its pair (Hyper.center).
    v_j      p_j (1 - p_j).
    w_ij     sim(i, j) ** gamma times the own/other weight; sim 1 for a
             content twin (Hyper.twins): a labeled item with the target's
             exact item_content under another key, and the only labeled item
             of the benchmark with that content.

Residual scale. For a labeled cell the base reads the label itself through
the item's integrated residual, so the leave-in prediction is pulled toward
y_j. Against true leave-one-out residuals (j's label dropped and the base
refitted; 31 public and 30 test-like formative runs, B3 to B31, on the
target's benchmark; review scratch scripts resid.py and resid_summary.py, not
in the repository) the leave-in residuals correlate at 0.99 and are 0.48
times as large, nearly uniformly: the median ratio runs from 0.45 to 0.64 over
p bins, alike for y = 0 and 1, as a logistic-normal item effect predicts. The
struct residuals are 0.90 times the LOO ones (correlation 0.99). The kind
therefore only rescales beta: beta on leave-in residuals is about twice beta
on LOO ones (leave-in beta 1 and 2 score within 5e-5 ALC of LOO beta 0.5 and
1). At formative size struct and leave-in score within 1e-4 of each other, and
the selection rule below picked the leave-in residual. Leave-in p_j also
understate v_j by about a quarter (mean 0.11 to 0.14 against 0.15 to 0.18),
second-order beside tau = 1.

Why the Newton form. With LOO residuals, sum w r / (tau + sum w v) is one
Newton step from 0 for an offset shared by the target and its neighbours in
proportion to w, under a N(0, 1/tau) prior: beta = 1 is that Bayesian step,
and residuals on items the base puts near 0 or 1 count on the logit scale (a
success at p = 0.1 moves the offset more than one at 0.5). On the default
leave-in residuals that step is beta about 2, so the default beta 0.5 is about
a quarter of it. The shift is clipped to +-max_shift.

Why centre. Under a Gaussian prior on the pair's standing the base's residuals
over a pair's labels do not sum to zero: they add up to the prior's pull on the
standing (sum r = (m - m0) / V at the mode). Carried to similar items unchanged,
that would undo the base's shrinkage for exactly the items that look like
labeled ones, an over-reaction to the pair's level that has nothing to do with
items. Centring within each pair keeps only the relative, item-level part, so
a pair's first label never moves anything on its own (one label, residual 0),
and the level stays the base's business.

Per-pair empirical-Bayes slope (Hyper.eb > 0). With the pair's own records as
targets, leave-one-item-out: u_j is the shift the layer would give j from every
other record, and the slope of the pair's residuals on it, b, has the one-step
MAP (1/eb^2 + sum u_j r_j) / (1/eb^2 + sum v_j u_j^2) under b ~ N(1, eb^2).
beta_eff = beta * clip(b, 0, 1): where the text says nothing about this pair's
residuals, the layer backs off. Off (eb = 0) by default.

Why twins are guarded. Identical item_content is not the same item in
matharena: 336 of its 1,633 items share one generic Kangaroo image prompt and
183 a bare system prompt, their difficulties as spread (Rasch sd 2.5 and 1.75)
as the benchmark's. Its real twins, one problem under two competition aliases
(aime_2025 and aime_2025_I), come in content groups of two, with difficulties
correlated at 0.92 (151 such pairs). A content that two or more labeled items
carry is therefore a template, and those items keep their cosine.

Similarity. Hashed TF-IDF, fitted on the labeled items of the target's
benchmark only (distinct items; smooth idf, sublinear tf), in two blocks whose
cosines are averaged with weight Hyper.word: word 1-2-grams (\\w+ tokens,
lower-cased, crc32) and byte 3-5-grams (lower-cased, whitespace collapsed,
polynomial hash), both mixed by splitmix64 into 2^22 buckets. The target is
transformed with that idf; its features no labeled item holds keep the largest
idf in its norm (sklearn's transform would drop them and inflate the cosine).
The text is item_features (first FEATURE_CHARS) plus item_content cut to its
first Hyper.head and last Hyper.tail characters (researchcodebench items run to
150k, their item-specific block within the last few thousand).

Purity and cost. Everything is a function of (input, labeled): records are
sorted (paiec.predict.records), the benchmark's index is built once per
content fingerprint of `labeled` and benchmark (Hyper.head and tail), lazily
when a target of that benchmark first arrives, and kept for `keep`
fingerprints; item features are memoised per item key, within DOC_BYTES. The
p_j are asked only where a weight can be nonzero: with Hyper.other 0 (the
default) that is the target's own pair's records, one base call each when that
pair's first target on the benchmark arrives, at most 31 at a checkpoint;
every other record keeps a placeholder 0.5 that its zero weight never reads,
so the shift is bit for bit what every p_j would give. With Hyper.other > 0
it is every record on the benchmark, which on a dense run (every pair of one
benchmark, 2,500 records at B31) makes the first call take 30 to 40 s. The p_j
are cached per record, residual kind and fingerprint. With no record on the
target's benchmark (B0 included) the answer is the base's own float,
unchanged; so is it with beta 0 or a zero shift. records(labeled) runs here
and again inside hier (about 2.4 ms a call at 372 labels); sharing it would
need a hier entry point that takes the records.

Measured (step 3a; scratch scripts exp_itemsig.py, analyze2.py,
analyze_dense.py and diag_sim2.py, not in the repository). Paired against the
shipped hier config (submission/model.py LEVEL, priors fitted without the
target's parent) on identical checkpoints: 888 configurations (gamma 2, 4, 8;
other 0, 0.5, 1; twins; center; resid; tau 0.5 to 4; beta 0.5 to 2; the slope
on 24 of them), selected by their worst regime on the first half of each of
test-like (testlike.Regime(), seed 2, 100 runs), groups mixed at random with
wholes (kinds mix/whole, seed 3, 50), public R1 benchmark-first and
pair-uniform (seed 0, 50 each), confirmed on the other half. The defaults are
that choice. On the confirmation halves they give -0.00002, -0.00006, -0.00012
and -0.00013 ALC in that order (cluster SEs 0.00003 to 0.00005). The public
gain is on matharena and researchcodebench, from B7 on. The best mean over the
four regimes (uncentred, other subjects at full weight) gives -0.0006 to
-0.0008 on public runs, +0.0001 on test-like ones (+0.001 on researchcodebench).
There it reads the group level that the base shrinks too far on whole public
benchmarks; that is not item-level signal. Twins change nothing measurable
(0.6% of test-like targets have one). The slope halves the loss of the most
aggressive configs and changes nothing at mild ones. On dense runs (every pair
of one benchmark), other subjects' records on similar items do carry signal:
real_webagents gains up to 0.002 (scope pair) and 0.005 (scope benchmark),
best of 888 and so optimistic; researchcodebench goes either way (-0.002 to
+0.008); the defaults stay within 0.0001 of zero. The ceiling at formative
size is low even without label noise: smoothed by this kernel over 31 to 62
labeled items (gamma 1 to 8, group means from the labeled items), public
Rasch difficulties net of feature groups correlate with a target's at 0.14 to
0.29 on matharena (identical contents at weight 1, templates included), 0.04
to 0.10 on researchcodebench, 0.07 to 0.14 on real_webagents and -0.04 to
+0.05 on multi_swebench.
With only a pair's own labels, text similarity carries almost nothing here.
Latency with the shipped hier underneath (scratch scripts dense_fix.py,
firstcall.py and densecalls.py; load average about 7 on 8 cores). Formative
runs through official.run_official with deep copies and two workers
(sample_run seeds 0 to 3): 3.0 to 5.4 ms a call against hier's 0.9 to 1.5 ms;
the slowest call 0.17 to 0.60 s against 0.05 to 0.20 s. A pair's first target on a benchmark makes
its 31 base calls (0.03 to 0.17 s in all); the first target of a benchmark
also featurises its labeled items, up to 0.5 s for 118 long (8k-character
excerpt) items. Dense runs at B31 (every pair of one benchmark, 806 to 2,542
records; hier 17 to 35 ms a call there): the first call 0.28 to 2.05 s, a
later pair's first call median 0.25 to 0.46 s (max 1.1 s), a pair's later
calls 19 to 50 ms.

Never raises: a failure in the layer returns the base's prediction, a failure
of the base 0.5 uncorrected (`failures` counts the former). A record whose p_j
the base fails on takes p_j = 0.5.

Imports: the standard library, numpy and paiec.hier (item_groups, for
as_new), relative within the package, so packaging it means adding itemsig.py
to the modules tools/build_submission.py ships as paiec_rt (hier ships
already) and a factory in submission/model.py; no scipy or scikit-learn is
involved.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
import sys
import threading
import traceback
import zlib
from collections import OrderedDict
from dataclasses import dataclass, fields, replace

import numpy as np

from .hier import item_groups
from .predict import HI, LO, _probability, _text, fingerprint, item_key, records, subject_key

RESIDS = ("struct", "leave_in")
#: a group key no real item carries: as_new() adds NEW_KEY=new to a labeled
#: item to ask the base for the same item as a new one
NEW_KEY = "\x00itemsig"
DIM = 1 << 22                   # hash buckets per block (ids fit in uint32)
NGRAMS = (3, 4, 5)              # byte n-grams
FEATURE_CHARS = 4_000           # item_features read, at most
_WORD = re.compile(r"\w+")
_WS = re.compile(r"\s+")
_U = np.uint64
_I = np.uint32


@dataclass(frozen=True)
class SigHyper:
    """The defaults are the configuration the step-3a grid chose by its worst
    regime (module docstring, "Measured"); the layer is then nearly inert.

    beta, tau      gain and prior precision of the one-step logit offset
    gamma          sharpening: w = sim ** gamma (0 < sim <= 1)
    twins          sim 1 for a content twin (module docstring)
    other          weight of another subject's record against the own pair's
    word           weight of the word block's cosine; the byte block gets 1 - word
    center         centre residuals within each pair
    max_shift      largest |logit shift|
    eb             prior sd of the per-pair slope; 0 switches the slope off
    resid          'struct' (the labeled item as a new one) or 'leave_in'
    head, tail     characters of item_content read from its start and end
    """
    beta: float = 0.5
    tau: float = 1.0
    gamma: float = 2.0
    other: float = 0.0
    word: float = 0.5
    center: bool = True
    twins: bool = True
    max_shift: float = 1.5
    eb: float = 0.0
    resid: str = "leave_in"
    head: int = 2_000
    tail: int = 6_000

    def __post_init__(self):
        for f in ("beta", "tau", "gamma", "other", "word", "max_shift", "eb"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
                raise ValueError(f"{f} must be a finite number, got {v!r}")
        if self.beta < 0 or self.tau <= 0 or self.gamma <= 0 or self.max_shift < 0 \
                or self.eb < 0 or not 0 <= self.other <= 1 or not 0 <= self.word <= 1:
            raise ValueError("need beta >= 0, tau > 0, gamma > 0, max_shift >= 0, eb >= 0 "
                             "and other, word in [0, 1]")
        if self.resid not in RESIDS:
            raise ValueError(f"resid must be one of {RESIDS}, got {self.resid!r}")
        for f in ("head", "tail"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) \
                    or v < 0:
                raise ValueError(f"{f} must be a number >= 0, got {v!r}")
        object.__setattr__(self, "head", int(self.head))
        object.__setattr__(self, "tail", int(self.tail))
        object.__setattr__(self, "center", bool(self.center))
        object.__setattr__(self, "twins", bool(self.twins))


# --- text features --------------------------------------------------------------------

def excerpt(text, head, tail):
    """The first `head` and last `tail` characters of a longer text."""
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n" + (text[len(text) - tail:] if tail else "")


def _mix(h):
    """splitmix64's finaliser, elementwise on uint64 (wrapping)."""
    h = h ^ (h >> _U(30))
    h = h * _U(0xBF58476D1CE4E5B9)
    h = h ^ (h >> _U(27))
    h = h * _U(0x94D049BB133111EB)
    return h ^ (h >> _U(31))


def _bag(h):
    """Sorted distinct buckets (uint32) and their sublinear tf, 1 + log(count)."""
    if not len(h):
        return np.zeros(0, _I), np.zeros(0)
    ids, cnt = np.unique((h % _U(DIM)).astype(_I), return_counts=True)
    return ids, 1.0 + np.log(cnt)


def text_of(item, head, tail):
    feats = _text(item.get("item_features"))[:FEATURE_CHARS]
    return feats + "\n" + excerpt(_text(item.get("item_content")), head, tail)


def features(text):
    """((word ids, tf), (byte n-gram ids, tf)) of one text, deterministic in
    every process (no salted hash())."""
    low = text.lower()
    with np.errstate(over="ignore"):
        toks = _WORD.findall(low)
        h = np.fromiter((zlib.crc32(t.encode("utf-8", "surrogatepass")) for t in toks),
                        _U, len(toks))
        parts = [_mix(h + _U(1 << 40))]
        if len(h) > 1:
            parts.append(_mix(h[:-1] * _U(0x9E3779B97F4A7C15) + h[1:] + _U(2 << 40)))
        word = _bag(np.concatenate(parts))
        b = np.frombuffer(_WS.sub(" ", low).strip().encode("utf-8", "surrogatepass"),
                          np.uint8).astype(_U)
        grams = []
        for n in NGRAMS:
            L = len(b) - n + 1
            if L <= 0:
                continue
            g = np.zeros(L, _U)
            for k in range(n):
                g = g * _U(1099511628211) + b[k:k + L]
            grams.append(_mix(g + _U(n << 48)))
        char = _bag(np.concatenate(grams)) if grams else _bag(np.zeros(0, _U))
    return word, char


class _Index:
    """One block's hashed TF-IDF over a benchmark's labeled items, as postings."""

    def __init__(self, bags):
        self.n = n = len(bags)
        lens = np.array([len(b[0]) for b in bags], int)
        ids = np.concatenate([b[0] for b in bags]) if n else np.zeros(0, _I)
        tf = np.concatenate([b[1] for b in bags]) if n else np.zeros(0)
        doc = np.repeat(np.arange(n), lens)
        self.V, inv, df = np.unique(ids, return_inverse=True, return_counts=True)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self.idf_oov = math.log(1.0 + n) + 1.0
        w = tf * self.idf[inv]
        norm = np.sqrt(np.bincount(doc, w * w, minlength=n))
        w = w / np.where(norm > 0, norm, 1.0)[doc]
        order = np.argsort(inv, kind="stable")
        self.doc, self.w = doc[order], w[order]
        self.ptr = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
        self.feat = np.repeat(np.arange(len(self.V)), df)     # each posting's feature

    def query(self, bag):
        """Cosine of a text (its bag) with every indexed item."""
        ids, tf = bag
        out = np.zeros(self.n)
        if not len(ids) or not len(self.V):
            return out
        pos = np.minimum(np.searchsorted(self.V, ids), len(self.V) - 1)
        hit = self.V[pos] == ids
        w = tf * np.where(hit, self.idf[pos], self.idf_oov)
        norm = math.sqrt(float(w @ w))
        if not norm > 0:
            return out
        p, q = pos[hit], w[hit] / norm
        start = self.ptr[p]
        ln = self.ptr[p + 1] - start
        tot = int(ln.sum())
        if not tot:
            return out
        if 4 * tot > len(self.w):
            # the target touches much of the index: one pass over every posting
            v = np.zeros(len(self.V))
            v[p] = q
            return np.bincount(self.doc, self.w * v[self.feat], minlength=self.n)
        rep = np.repeat(np.arange(len(p)), ln)
        off = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(start, ln)
        return np.bincount(self.doc[off], self.w[off] * q[rep], minlength=self.n)


# --- a labeled item as a new one ------------------------------------------------------------

def _with_key(s):
    """Ways to add NEW_KEY=new to an item_features or interactors string."""
    s = _text(s)
    st = s.strip()
    if not st:
        yield NEW_KEY + "=new"
        return
    if st[0] == "{":
        try:
            obj = json.loads(st)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            yield json.dumps({**obj, NEW_KEY: "new"})
    yield s + "\n" + NEW_KEY + "=new"
    yield s + ";" + NEW_KEY + "=new"


def as_new(item, prefix=False):
    """The item under a new key that the base reads as it reads the item,
    or None.

    NEW_KEY=new goes into its interactors, else into its item_features, in
    the first form (JSON key, new line, ';' token) after which
    hier.item_groups(item, prefix) is the item's own groups plus NEW_KEY
    alone: a key no labeled item carries, so no fit selects it, and every
    group the base does read keeps its level. item_content and benchmark_id
    are untouched (floor, text term, level). prefix is the base's
    Flags.prefix_groups. None when neither field can take the key that way:
    a string past hier's FEATURE_CHARS is hashed whole, and under prefix
    groups an item without features loses its prefix group.
    """
    try:
        want = item_groups(item, prefix)
        for field, gk in (("interactors", "interactors." + NEW_KEY),
                          ("item_features", NEW_KEY)):
            for s in _with_key(item.get(field)):
                new = dict(item)
                new[field] = s
                got = item_groups(new, prefix)
                if got.pop(gk, None) == "new" and got == want:
                    return new
    except Exception:
        pass
    return None


# --- per labeled list and benchmark ------------------------------------------------------

class Terms:
    """What one target reads from its benchmark's records: per record the two
    cosines, twin, own and excluded flags, the label, the pair index and the
    base's p_j by residual kind; `bench` for the per-pair slope. `every`: the
    p_j cover every pair's records, not only the target's own pair's (built
    with Hyper.other > 0), which a hyper with other > 0 needs."""
    __slots__ = ("cw", "cc", "twin", "own", "excl", "y", "pair", "p", "bench", "sk", "every")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Bench:
    """A benchmark's labeled records under one labeled list: their items'
    index, and the base's p_j, filled lazily and only where a weight can be
    nonzero."""

    def __init__(self, model, recs, head, tail):
        self.recs = recs
        self.head, self.tail = head, tail
        docs, where = [], {}
        for r in recs:
            if r[2] not in where:
                where[r[2]] = len(docs)
                docs.append(r)
        self.where = where
        self.rec_doc = np.array([where[r[2]] for r in recs], int)
        feats = [model.doc(r[2], r[5], head, tail) for r in docs]
        self.word = _Index([f[0] for f in feats])
        self.char = _Index([f[1] for f in feats])
        self.digest = [f[2] for f in feats]
        self.carriers = {}
        for g in self.digest:
            self.carriers[g] = self.carriers.get(g, 0) + 1
        codes = {}
        self.code = np.array([codes.setdefault(g, len(codes)) for g in self.digest], int)
        pairs = {}
        self.pair = np.array([pairs.setdefault(r[1], len(pairs)) for r in recs], int)
        self.pairs = pairs
        self.y = np.array([r[3] for r in recs], float)
        self.p = {}             # (kind, pair or None) -> p_j of every record
        self.pj = {}            # (kind, subject key, item key) -> the base's p_j
        self.slopes = {}
        self._rows = {}
        self.calls = 0          # base calls made for p_j

    def base_p(self, model, labeled, kind, pi):
        """p_j of every record for one residual kind: the base's for the
        records of pair pi (of every pair when pi is None), 0.5 for the rest,
        whose weights are zero wherever this array is read."""
        key = (kind, pi)
        got = self.p.get(key)
        if got is None:
            got = np.full(len(self.recs), 0.5)
            rows = range(len(self.recs)) if pi is None else np.flatnonzero(self.pair == pi)
            for k in rows:
                got[k] = self.p_of(model, labeled, kind, self.recs[k])
            self.p[key] = got
        return got

    def p_of(self, model, labeled, kind, r):
        """The base's p_j of one record, in [LO, HI]; its leave-in p_j when
        the item cannot be asked for as a new one."""
        key = (kind, r[1], r[2])
        got = self.pj.get(key)
        if got is None:
            item = r[5]
            if kind == "struct":
                new = as_new(item, model.prefix)
                item = item if new is None else new
            self.calls += 1
            got = min(max(model.base_prediction([r[4], item], labeled), LO), HI)
            self.pj[key] = got
        return got

    def terms(self, model, subject, item, labeled, kinds, other):
        f = model.doc(item_key(item), item, self.head, self.tail)
        cw, cc = self.word.query(f[0]), self.char.query(f[1])
        ik, sk = item_key(item), subject_key(subject)
        excl = np.zeros(len(self.digest), bool)
        twin = np.zeros(len(self.digest), bool)
        j = self.where.get(ik)
        if j is not None:
            excl[j] = True
        elif self.carriers.get(f[2]) == 1:
            twin[self.digest.index(f[2])] = True
        d = self.rec_doc
        pi = self.pairs.get(sk, -1)
        need = None if other > 0 else pi       # whose weights can be nonzero
        return Terms(cw=cw[d], cc=cc[d], twin=twin[d], own=self.pair == pi, excl=excl[d],
                     y=self.y, pair=self.pair, bench=self, sk=sk, every=need is None,
                     p={k: self.base_p(model, labeled, k, need) for k in kinds})

    def rows(self, pi, docs):
        """(word cosines, byte cosines, twins) of labeled items `docs` (pair
        pi's) with every labeled item, for the per-pair slope only."""
        got = self._rows.get(pi)
        if got is None:
            # twins as a target sees them: another labeled item with the same
            # content, when those two are its only carriers
            two = np.array([self.carriers[g] == 2 for g in self.digest], bool)
            same = (self.code[docs][:, None] == self.code[None, :]) & two[docs][:, None]
            got = self._rows[pi] = (_gram(self.word, docs), _gram(self.char, docs), same)
        return got


def _gram(ix, rows=None):
    """Cosines between indexed items `rows` (all by default) and every
    indexed item: each item's own postings, queried against the index."""
    rows = range(ix.n) if rows is None else rows
    M = np.zeros((len(rows), ix.n))
    if not len(ix.doc):
        return M
    by = np.argsort(ix.doc, kind="stable")
    cuts = np.concatenate([[0], np.cumsum(np.bincount(ix.doc, minlength=ix.n))])
    for i, d in enumerate(rows):
        own = by[cuts[d]:cuts[d + 1]]            # d's postings, by position
        if not len(own):
            continue
        feat = np.searchsorted(ix.ptr, own, side="right") - 1   # postings sort by feature
        start = ix.ptr[feat]
        ln = ix.ptr[feat + 1] - start
        rep = np.repeat(np.arange(len(feat)), ln)
        off = np.arange(int(ln.sum())) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(start, ln)
        M[i] = np.bincount(ix.doc[off], ix.w[off] * ix.w[own][rep], minlength=ix.n)
    return M


class _State:
    """One labeled list: its records by benchmark and their _Bench, lazily."""

    def __init__(self, recs):
        self.by = {}
        for r in recs:
            self.by.setdefault(r[0], []).append(r)
        self.benches = {}

    def bench(self, model, bid, head, tail):
        key = (bid, head, tail)
        if key not in self.benches:
            rs = self.by.get(bid)
            self.benches[key] = _Bench(model, rs, head, tail) if rs else None
        return self.benches[key]


# --- the correction ---------------------------------------------------------------------

def _centre(r, pair):
    n = np.bincount(pair)
    return r - (np.bincount(pair, r) / np.maximum(n, 1))[pair]


def residuals(t, h):
    """Centred (Hyper.center) residuals and variances of the records, by
    h.resid. Other pairs' entries are placeholders unless t.every."""
    if h.other > 0 and not t.every:
        raise ValueError("these terms hold only the own pair's p_j; build them with other > 0")
    p = t.p[h.resid]
    r = t.y - p
    if h.center and len(r):
        r = _centre(r, t.pair)
    return r, p * (1 - p)


def weights(t, h):
    sim = np.clip(h.word * t.cw + (1 - h.word) * t.cc, 0.0, 1.0)
    if h.twins:
        sim = np.where(t.twin, 1.0, sim)
    w = sim ** h.gamma * np.where(t.own, 1.0, h.other)
    return np.where(t.excl, 0.0, w)


def raw_shift(t, h):
    """sum w r / (tau + sum w v), before beta and the clip."""
    w = weights(t, h)
    r, v = residuals(t, h)
    return float(w @ r) / (h.tau + float(w @ v))


def slope(t, h):
    """The per-pair empirical-Bayes slope of the target's pair, clipped to
    [0, 1]; 1 when off or when the pair has fewer than two labeled items."""
    if not h.eb > 0:
        return 1.0
    bench, sk = t.bench, t.sk
    key = (sk, h.gamma, h.other, h.word, h.tau, h.center, h.twins, h.resid, h.eb)
    got = bench.slopes.get(key)
    if got is not None:
        return got
    pi = bench.pairs.get(sk, -1)
    mine = np.flatnonzero(bench.pair == pi)
    d = bench.rec_doc
    docs = sorted(set(d[mine].tolist()))
    b = 1.0
    if len(docs) >= 2:
        Sw, Sc, same = bench.rows(pi, docs)
        row = {x: k for k, x in enumerate(docs)}
        r, v = residuals(t, h)
        own = bench.pair == pi
        num, den = 1.0 / h.eb ** 2, 1.0 / h.eb ** 2
        for j in mine:
            k = row[d[j]]
            sim = np.clip(h.word * Sw[k, d] + (1 - h.word) * Sc[k, d], 0.0, 1.0)
            if h.twins:
                sim = np.where(same[k, d], 1.0, sim)
            w = sim ** h.gamma * np.where(own, 1.0, h.other)
            w[d == d[j]] = 0.0                       # leave j's item out
            u = float(w @ r) / (h.tau + float(w @ v))
            num += u * r[j]
            den += v[j] * u * u
        b = min(max(num / den, 0.0), 1.0) if math.isfinite(num / den) else 1.0
    bench.slopes[key] = b
    return b


def shift(t, h, b=None):
    """The logit shift for one target's terms under h (b: the pair's slope,
    computed when None)."""
    if not len(t.y) or h.beta == 0 or h.max_shift == 0:
        return 0.0
    if b is None:
        b = slope(t, h)
    d = h.beta * b * raw_shift(t, h)
    if not math.isfinite(d):
        raise FloatingPointError("non-finite shift")
    return min(max(d, -h.max_shift), h.max_shift)


def _logit(p):
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


# --- the predictor ----------------------------------------------------------------------

class ItemSig:
    """predict(input, labeled): the base's prediction with the item-level shift.

    `base` is a predict callable, or an object with .predict (HierPredictor,
    Predictor). The base must itself be a pure function of (input, labeled);
    it is called once per target, and once per labeled record of the target's
    pair on its benchmark per labeled list (the p_j, cached; every pair's
    records with Hyper.other > 0).
    """

    #: bytes of memoised text features (bucket ids and tf) kept at most
    DOC_BYTES = 1 << 28

    def __init__(self, base, hyper=None, keep=4, **kw):
        self.base = base if callable(base) else base.predict
        hyper = SigHyper() if hyper is None else hyper
        self.hyper = replace(hyper, **kw) if kw else hyper
        self.keep = keep
        # the base's Flags.prefix_groups, for as_new (hier's; False otherwise)
        owner = getattr(self.base, "__self__", None)
        self.prefix = bool(getattr(getattr(owner, "cfg", None), "prefix_groups", False))
        self._states: OrderedDict[bytes, _State] = OrderedDict()
        self._docs: dict = {}
        self._doc_bytes = 0
        self._lock = threading.RLock()
        self.failures = 0

    def doc(self, ik, item, head, tail):
        """(word bag, byte bag, content digest) of an item, memoised on its key."""
        key = (ik, head, tail)
        got = self._docs.get(key)
        if got is None:
            w, c = features(text_of(item, head, tail))
            dig = hashlib.blake2b(_text(item.get("item_content")).encode("utf-8", "surrogatepass"),
                                  digest_size=16).digest()
            got = (w, c, dig)
            size = w[0].nbytes + w[1].nbytes + c[0].nbytes + c[1].nbytes
            if self._doc_bytes + size > self.DOC_BYTES:
                self._docs.clear()
                self._doc_bytes = 0
            self._docs[key] = got
            self._doc_bytes += size
        return got

    def base_prediction(self, input, labeled):
        """The base's answer as a probability in [LO, HI]; 0.5 if it fails."""
        p = self._base(input, labeled)
        return 0.5 if p is None else p

    def _base(self, input, labeled):
        try:
            return _probability(self.base(input, labeled))
        except Exception:
            return None

    def _state(self, recs):
        fp = fingerprint(recs)
        st = self._states.get(fp)
        if st is None:
            st = self._states[fp] = _State(recs)
            while len(self._states) > self.keep:
                self._states.popitem(last=False)
        else:
            self._states.move_to_end(fp)
        return st

    def terms(self, input, labeled, kinds=None, hyper=None):
        """Terms of one target, or None when no record sits on its benchmark.
        With hyper.other 0 they hold the p_j of the target's own pair only,
        and residuals() refuses to read them under other > 0."""
        h = self.hyper if hyper is None else hyper
        subject, item = input
        if not isinstance(subject, dict) or not isinstance(item, dict):
            raise TypeError("input must be [subject dict, item dict]")
        recs = records(labeled)
        if not recs:
            return None
        bench = self._state(recs).bench(self, _text(item.get("benchmark_id")), h.head, h.tail)
        if bench is None:
            return None
        return bench.terms(self, subject, item, labeled, kinds or (h.resid,), h.other)

    def shift(self, input, labeled=None):
        t = self.terms(input, labeled)
        return 0.0 if t is None else shift(t, self.hyper)

    def predict(self, input, labeled=None):
        """P(correct), a native float in [1e-4, 1 - 1e-4]. Never raises."""
        with self._lock:
            p0 = self._base(input, labeled)
            if p0 is None:              # nothing to correct
                return 0.5
            try:
                d = self.shift(input, labeled)
                if d == 0.0:
                    return p0
                p = _probability(_sigmoid(_logit(p0) + d))
                if p is None:
                    raise FloatingPointError("non-finite probability")
                return p
            except Exception:
                self.failures += 1
                if self.failures == 1:
                    try:
                        traceback.print_exc(file=sys.stderr)
                    except Exception:
                        pass
                return p0


def make_itemsig(base_factory, hyper=None, **kw):
    """A factory of fresh ItemSig predict callables around fresh base
    predictors, as paiec.official.run_official's model_factory takes it:
    base_factory() returns a predict callable (or an object with .predict).
    Hyperparameters as SigHyper fields; validated here, once."""
    h = SigHyper() if hyper is None else hyper
    h = replace(h, **kw) if kw else h

    def factory():
        return ItemSig(base_factory(), h).predict
    return factory


HYPER_FIELDS = tuple(f.name for f in fields(SigHyper))
