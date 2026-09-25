"""Hierarchical Bayesian run-time predictor.

    eta = mu_b + theta_s + delta_sb - g_i - e_i
    p   = c + (1 - c - slip) * E[sigmoid(eta)],   eta ~ posterior

    mu_b      the benchmark's level. It is shared by every subject's labels on
              the benchmark, which is what the Predictor leaves unused: on dense
              multi_swebench the other 81 subjects' labels say the level is low
              long before the target's own do. Gaussian by default; Student-t
              tails are an option (Hyper.nu_mu).
    theta_s   the subject's standing, keyed on its canonical name so a model
              seen on two benchmarks of a run is one subject. Its prior comes
              from paiec.prior: attributes, plus the same name's standing on
              public benchmarks, precision-weighted and capped because it
              barely transfers. Both are relative, centred within benchmark.
    delta_sb  the pair's own deviation (full subject dict + benchmark): nearly
              all of a subject's standing that is left after its attributes.
    g_i       group effects u_{b,key,value}, one per item_features key whose
              visible values are neither all numeric nor constant; the keys
              share a fixed group variance, so a useless key costs little.
    e_i       the item's own residual, integrated out exactly per item.
    c         Hyper.guess times the multiple-choice floor (paiec.mcq), in the
              likelihood of a floored item's labels as well as in the
              prediction: a subject that does not know guesses with
              probability guess, so a run of failures can go below 1/n.

Components the target touches that no label touched (a new benchmark, subject,
pair, group or item) stay at their prior mean and variance, and E[sigmoid] is
taken over the target's posterior by quadrature (a 171-node trapezoid, exact
to 1e-9; the probit approximation is off by more than 0.01 at the variances
this model reaches, tests/test_hier.py).

Everything is on the scale of an item-level (Rasch) model, where the public
variance components were measured, and everything is fitted from `labeled`
alone: one fit per content fingerprint, as the Predictor does, so a worker that
serves many targets fits once and the answer is a pure function of (input,
labeled).

Why the item residual is integrated instead of joining the MAP. A joint mode
over (x, e) with an item variance of 8.7 and one label per item is the classic
PQL failure: the item effects absorb each label, the mode sits where the
logistic is flat, and a pair with 160 successes in 200 labels is predicted at
0.70 on a new item (0.79 with e integrated; tests/test_hier.py). So the vector x
holds every component except the e_i, each item's likelihood is the 1-D integral
over its e_i, and Newton with a line search finds the mode of the joint log
posterior of x. Its gradient and Hessian are exact posterior moments under each
item's integrand. The integral is adaptive Gauss-Hermite, 20 nodes at the
item's mode: good to 1e-7 in log L on most items, up to 1e-2 on one whose
labels all sit far out on one side of a wide prior, which moves no prediction
by more than 3e-4 (test_gauss_hermite_error_does_not_reach_predictions). An
item that holds a floored success can have two modes (the item is hard and the
success a guess, or it is not), which 20 nodes at one of them missed by up to
0.04, so it gets a fixed 201-node trapezoid spanning the prior and the mode,
good to 1e-5 (test_item_likelihood_is_the_integral). A group level that only
one labeled item carries is identified only together with that item's
residual, so it is folded into it rather than joining the mode (the same PQL
failure, one level at a time): the item's residual variance grows by the
level's, and a target that shares the level reads it back from that item.

The target. Its eta is a'x plus components no label touched plus, on an item
that carries labels of its own, minus that item's residual. The Laplace
(Gaussian) posterior of a'x under-reacts to a pair's first labels, because the
mode of a skewed logistic posterior sits nearer the prior than its mean: 0.013
to 0.02 too close to the prior after one to seven labels at the default widths.
So a'x is read along its line (_Fit._line): the exact log posterior on a grid
of x(s) = x_hat + Sigma a (s - s_hat) / a'Sigma a, the Gaussian's conditional
mean, INLA's Gaussian strategy for a linear combination. For one pair it is the
exact marginal (test_laplace_error_at_low_budgets). With other subjects on the
benchmark it moves their standings along that conditional mean and ignores
their skew, so a bias toward 0.5 is left at low budgets: for a new subject
after six others' 31 failures 0.011 of the Gaussian's 0.014; with 26 subjects
drawn from the model a mean |line - exact| of 0.014 at B1 (+0.012 toward 0.5),
0.007 at B3 and at most 0.003 from B7 on, which moves Brier by 0.001 at B1 and
by 1e-4 at most from B7 on; for a target with labels of its own after others'
extreme records 0.028 ('1/1 after six 0/31'); and more on floored
benchmarks: 0.026 for a new subject after three 1/7, 0.016 for 3/7 after
twelve 5/15 (final review, final_verify/d3_exact_dense.py; the point cases
rerun at the current defaults in step-2 final_fix/v2_after.txt).
The line integrates the items it moves most, as far as LINE_MAX labels go, and
the others to second order at the mode (_Fit._line), which on dense B31
checkpoints is as close to the exact predictive as no cap at all.

The target's own labeled item is read node by node at every grid point, not as
a Gaussian with its mean and variance, which put targets up to 0.019 toward 0.5
(test_a_target_on_a_labeled_item_reads_its_residual_node_by_node). But the
line reads that residual at the Gaussian conditional mean of the other
subjects who labeled the item, and the part of its shift the line does not
carry (J'x given a'x, sd 0.3 to 0.5 at the median on real runs) is added as
Gaussian variance. Where those subjects' standings are skewed the residual's
posterior mean is off: six subjects at 31/31 who all failed the item, and a
target at 31/31, give 0.365 against the exact 0.295, where the Laplace fit
says 0.311 (and the mirror case -0.071). For such labeled-item targets the
line gives no reliable gain over Laplace: on cases drawn from the model its
mean |p - exact| is 0.004 to 0.012 against Laplace's 0.005 to 0.014, and its
Brier is the same within noise (final_verify/d4_star_sim.py). A 2-D grid over
(a'x, J'x) along the Gaussian conditional path takes the corner only from
+0.070 to +0.061 (step-2 final_fix/plane.py): the other standings' skews are
independent, and one direction cannot carry them. What would is a refit of
the rest at each node of the item's residual (INLA's Laplace strategy), one
per labeled item; it is not done.

A Student-t level (Hyper.nu_mu > 0) is its Gaussian scale mixture: the level
is N(mu0, sigma_mu^2 / lambda) with lambda ~ Gamma(nu/2, nu/2). The fit
replaces the t by a Gaussian at LAM values of lambda for each level labels
touched, refits the whole model at each, weights them by prior weight times
Laplace evidence and predicts with the mixture, each node through its line. A
Laplace fit on the t density itself missed the exact posterior predictive by
up to 0.065 at three labels (the t's posterior is far from Gaussian, and the
line cannot mend that); the mixture is within 0.002 for one pair and 0.004 for
a new subject after six others' failures (test_student_t_level). The levels of
different benchmarks are mixed one at a time: at each node of one level the
whole model is refitted, the other t levels with it, but on their t density
(Laplace), not as mixtures of their own. They reach the mixed level only
through theta, so this is accurate while the link weight is small, which it is
only with an attribute prior (0.002 at the defaults). Without one sigma_attr^2
joins theta and the weight is 0.16 at the defaults and 0.15 under
prior.REFERENCE (0.52 under the earlier REFERENCE, whose sigma_delta was 1.0):
against exact quadrature one subject on two benchmarks was off by up to 0.010
at the defaults without attributes, 0.010 under REFERENCE and 0.025 at weight
0.3, where a Gaussian level stays within 0.006 (final_verify/v3b_link_t.py,
rerun in step-2 final_fix/v3b_after.txt).
Mixing linked t levels jointly would cost LAM^k refits for k levels, and is
not done. It costs LAM refits per benchmark and a line per node per target: 5
to 10 times the Gaussian level's time (5.8 against 0.6 ms a call on one
formative run).

Without floors the log posterior of x is concave (a marginal of a log-concave
density) and its mode unique. The floor's success term log(c + (1 - c) s) is
not concave, so away from the mode Newton's matrix may be indefinite; it is
then shifted to positive definite, and the line search keeps every step an
ascent. Putting the floor in the likelihood is what keeps guessing from being
counted twice: fitted with a plain logistic, labels on a four-option benchmark
already hold the guesses, and adding the floor again at prediction put one
pair 0.06 to 0.15 above the exact posterior predictive (tests/test_hier.py).
A fit whose Newton decrement is still above 1e-8 when it stops (after at most
CLOSE_MAX steps near the mode) is counted in HierPredictor.unconverged.

Linking a subject across the benchmarks of a run goes through theta alone, so
its weight is Hyper.link_weight = sigma_theta^2 / (sigma_theta^2 + sigma_delta^2):
how far a subject's standing on one benchmark, however well measured, moves its
prediction on another. The two step-2 analyses that seem to disagree on it
measure different things. levels-variance's 86% transferable is the raw share on
real_webagents and researchcodebench, 12 subjects and no attributes, where
subject-transfer also finds a raw correlation of 0.56. Over the four
multi-subject benchmarks, with benchmark offsets removed, the raw shared share
is 0.30 to 0.36, and once the attribute mean is taken out it is -0.06
[-0.21, 0.07] on the accuracy scale: what transfers is what release date,
provider and size already say. So with an attribute prior the default weight is
small (the public tau2 of attribute residuals is negative and clipped to 0.01);
without one, the variance the attributes would have explained (sigma_attr^2)
goes into theta, which is shared, and the weight is (sigma_theta^2 +
sigma_attr^2) / (that + sigma_delta^2), 0.16 at the defaults. Flags.link=False turns linking off, and
Hyper.relink(w) sets the weight at a fixed total, so the evaluation can decide.

Size. The fit is dense in its columns (levels, standings, pair deviations,
group levels), and each Newton step costs their cube. A key keeps at most
MAX_LEVELS levels of its own (the rarest others share nothing, like a folded
level), and a labeled list past MAX_COLUMNS columns is fitted benchmark by
benchmark, which gives up only the link across benchmarks.

Imports: numpy and the package's own run-time modules only, relative, so the
module could ship as paiec_rt like predict.py. scipy and scikit-learn are
needed only by the optional text term, through predict's lazy imports.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sys
import threading
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from .mcq import floor_of
from .predict import (SUBJECT_FIELDS, BenchmarkFit, _one_thread, _probability, _text,
                      fingerprint, item_key, records, subject_key)
from .prior import canon_name

GH = 20
_T, _W = np.polynomial.hermite.hermgauss(GH)
_LW = np.log(_W) + _T ** 2
GRID = 201                   # trapezoid nodes over an item with a floored success
_GU = np.linspace(0.0, 1.0, GRID)
_SQ2 = math.sqrt(2.0)
_LOG2PI = math.log(2 * math.pi)
#: E over a standard normal by the trapezoid rule: exact to 3e-10 for a sigmoid
#: at any variance up to 100, where 20-node Gauss-Hermite is off by 0.01
_ZN = np.linspace(-8.5, 8.5, 171)
_ZW = np.exp(-_ZN ** 2 / 2)
_ZW = _ZW / _ZW.sum()
#: the same by Gauss-Hermite: 8 nodes are exact to 2e-12 up to variance 0.1,
#: 20 to 2e-10 up to 1 (the components of a target on a labeled item mostly
#: have less), as (max variance, nodes, weights)
_GH_TIERS = [(v, _SQ2 * t, w / w.sum()) for v, (t, w) in
             ((0.1, np.polynomial.hermite.hermgauss(8)), (1.0, (_T, _W)))]
MISSING = "<missing>"
MAX_KEYS = 8                 # group keys per benchmark
MAX_LEVELS = 64              # group levels per key with a column of their own
MAX_COLUMNS = 1_500          # past this the fit is split by benchmark
FEATURE_CHARS = 4_000        # a longer features string is one opaque value
UNIQUE = (32, 0.8)           # a key with more levels than 0.8 of >= 32 items names items
TN = 64                      # quadrature nodes over an untouched Student-t level
LAM = 16                     # scale-mixture nodes over a touched Student-t level
NEAR = 1e-5                  # Newton decrement below which steps skip the line search
CLOSE_MAX = 50               # most such steps in one solve
LINE_H = 0.75                # grid step along a target's a'x, in its Laplace sds
LINE_MAX = 600               # most labels a line integrates at every grid point
LINE_FRAC = 0.02             # items whose labels move by this share of the most moved
LINE_BYTES = 64 << 20        # memory for a fit's memoised lines
#: the public binary benchmarks Hyper's defaults were fitted on
PUBLIC = ("matharena", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")


@dataclass(frozen=True)
class Hyper:
    """Hyperparameters on the item-level (Rasch) scale.

    The defaults are paiec.prior.fit_hyper on all five public benchmarks, the
    row 'excluding nothing' of `python -m paiec.prior`, rounded to three
    decimals: the shipped model may use every public benchmark, since the
    hidden ones are none of them. An experiment must refit them without the
    benchmarks it scores (fit_hyper's `exclude`, recorded in `excluded`;
    HierPredictor refuses a prior and hyperparameters fitted on different
    pairs or exclusions), and fit_hyper never falls back to these defaults: a
    field the included benchmarks cannot identify takes paiec.prior.REFERENCE,
    which no fit went into (its sigma_delta was chosen with the public range in
    view, see there). slip, guess, text_share, id_cap and paiec.prior's WIDEN and
    G_CAP, like MAX_LEVELS and UNIQUE here, were set looking at all five public
    benchmarks under every exclusion: a leave-one-benchmark-out score carries
    that much optimism, as the Predictor's does (docs/findings.md).

    mu0, sigma_mu   level of a new benchmark at attribute score 0 (the public
                    Rasch levels less their pool's mean attribute score, which
                    is what the run-time eta adds theta's prior mean to): their
                    mean, and their sd s taken to s sqrt(1 + 1/n), the spread
                    of a new level about the mean of n, then widened 1.44x
                    (prior.WIDEN), the factor by which the Gaussian that stayed
                    robust in levels-variance's width sweep (pair scale 1.3)
                    exceeds the public fit (0.9). fit_hyper estimates both only
                    from three levels or more (prior.MIN_LEVELS), else mu0 = 0
                    and sigma_mu = 2.5 (prior.REFERENCE). With Flags.attributes
                    off the centre is off by a pool's mean attribute score
                    (-0.53 to +0.35 on public pools)
    nu_mu           0 keeps the level Gaussian. nu_mu > 0 gives it Student-t
                    tails with nu_mu degrees of freedom and scale sigma_mu, which
                    should then be the unwidened sd (fit_hyper(nu_mu=3) does
                    this, with the 1 + 1/n term). Under a flat prior on the
                    levels' mean and log sd a new level is exactly t with
                    n - 1 degrees of freedom and scale s sqrt(1 + 1/n), so
                    nu_mu = n - 1 is the textbook choice; the Gaussian
                    understates that spread most when few levels are left
                    (three levels: t2). t3 at the public scale is what levels-variance
                    recommends, but in its own sweep the widened Gaussian had
                    lower or equal regret in five of six scenarios and lost
                    0.0006 on public runs (pair-cluster SE 0.0016): the
                    evaluation decides
    sigma_theta     standing shared across benchmarks after attributes: the
                    identity tau2 of attribute residuals, clipped to [0.01, 0.2]
                    (public: negative, so 0.1)
    sigma_delta     pair deviation around the attribute prior: the LOBO
                    attribute residual variance less theta's and the
                    coefficients' share, widened 1.4x likewise
    sigma_attr      the spread the attributes explain; it goes back into
                    theta's prior variance when there is no attribute prior
    sigma_d, g      the median item variance of the multi-subject benchmarks,
                    split by their median group share capped at 0.25 (the
                    robust range of the item-features sweep is 0.2 to 0.3);
                    sigma_g is divided evenly over a benchmark's keys and joins
                    the residual where a benchmark has no keys
    slip            0.01: Rasch predictions above 0.98 came true 99.0% of the
                    time on matharena (levels-variance); not estimated
    guess           the probability that a subject who does not know an item
                    guesses among its options, so the floor is guess / n. 0.5,
                    not estimated: the hard floor (1) scored as no floor on
                    public runs (+0.0002 ALC, step-2 review), holds 31 failures
                    at 1/n, and mcq.floor_of flags code items on
                    researchcodebench where no floor applies
    text_share      share of the item residual the optional text term takes
    id_cap          largest weight the identity table may get
    excluded        the benchmarks these were fitted without
    included        the benchmarks they were fitted on, as fit_hyper saw
                    them; when not given (the defaults, a Hyper built by hand
                    or read from an older dict), PUBLIC less `excluded`.
                    HierPredictor compares it with the subject prior's
                    meta['included'], which catches a prior built on pairs
                    filtered beforehand (excluded empty) meeting the defaults
    """
    mu0: float = -1.263
    sigma_mu: float = 2.676
    nu_mu: float = 0.0
    sigma_theta: float = 0.1
    sigma_delta: float = 2.371
    sigma_attr: float = 1.032
    sigma_d: float = 2.671
    sigma_g: float = 1.542
    slip: float = 0.01
    guess: float = 0.5
    text_share: float = 0.2
    id_cap: float = 0.1
    excluded: tuple = ()
    included: tuple = None

    def __post_init__(self):
        object.__setattr__(self, "excluded", tuple(sorted({str(b) for b in self.excluded})))
        inc = [b for b in PUBLIC if b not in self.excluded] if self.included is None \
            else self.included
        object.__setattr__(self, "included", tuple(sorted({str(b) for b in inc})))

    @property
    def link_weight(self):
        """How much a subject's standing on one benchmark of the run moves its
        prediction on another, with an attribute prior (without one,
        sigma_attr^2 joins sigma_theta^2)."""
        th = self.sigma_theta ** 2
        return th / (th + self.sigma_delta ** 2)

    def relink(self, w):
        """The same hyperparameters with link weight w in [0, 1], the pair's
        total variance around its prior mean unchanged. Without an attribute
        prior sigma_attr^2 still links on top; Flags.link=False is off."""
        tot = self.sigma_theta ** 2 + self.sigma_delta ** 2
        w = min(max(float(w), 0.0), 1.0)
        return replace(self, sigma_theta=math.sqrt(w * tot),
                       sigma_delta=math.sqrt((1 - w) * tot))

    def to_dict(self):
        d = asdict(self)
        d["excluded"], d["included"] = list(self.excluded), list(self.included)
        return d

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        out = {}
        for k, v in dict(d).items():
            if k in ("excluded", "included"):
                if isinstance(v, (str, bytes)):
                    raise TypeError(f"{k} must be a list of benchmark names")
                out[k] = tuple(v)
            elif k in names:
                out[k] = float(v)
        return cls(**out)


@dataclass(frozen=True)
class Flags:
    """Switches for ablations; the defaults are the model as proposed.

    identity, attributes  the two halves of theta's prior
    pool_mu               one level per benchmark shared by all its subjects;
                          off gives every pair a level of its own (Gaussian
                          only)
    link                  one theta per identity key across the run's
                          benchmarks (weight Hyper.link_weight); off gives
                          every pair a theta of its own, so nothing a subject
                          does on one benchmark reaches another
    delta                 off folds the pair deviation into theta (each
                          subject dict keeps its own attribute offset)
    groups                item_features group effects
    prefix_groups         items without features grouped on their first 2,000
                          characters (never measured at run time, so off)
    text                  the Predictor's text embedding as a difficulty term,
                          once a benchmark has `warmup` labeled items; off, as
                          nothing measured says it pays
    floor, slip           the MCQ floor (likelihood and prediction) and the
                          slip at prediction
    t_mixture             a Student-t level as its scale mixture; off leaves
                          the Laplace fit on the t density (to compare)
    line                  the posterior of the target's a'x on a grid along the
                          line of its Gaussian conditional mean (_Fit._line);
                          off leaves it Gaussian (Laplace)
    subject_key           'canon' (normalized_name or source_model_name,
                          normalised; links a model across benchmarks), 'name'
                          or 'full' (all eight fields: links only identical
                          dicts, which on public data happens only between
                          real_webagents and researchcodebench)
    """
    identity: bool = True
    attributes: bool = True
    pool_mu: bool = True
    link: bool = True
    delta: bool = True
    groups: bool = True
    prefix_groups: bool = False
    text: bool = False
    floor: bool = True
    slip: bool = True
    t_mixture: bool = True
    line: bool = True
    subject_key: str = "canon"
    warmup: int = 64
    dim: int = 64


class _Memo(dict):
    """A dict that forgets everything once it holds `limit` entries or `chars`
    units of stored size (characters of text, bytes of arrays), so a worker's
    memory stays bounded however many distinct items, subjects or target
    directions a run holds."""

    def __init__(self, limit, chars=float("inf")):
        super().__init__()
        self.limit, self.chars, self.used = limit, chars, 0

    def put(self, key, value, size=0):
        if len(self) >= self.limit or self.used + size > self.chars:
            self.clear()
            self.used = 0
        self[key] = value
        self.used += size
        return value


# --- features -------------------------------------------------------------------

_NUM = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")


def _split_top(s, seps=";\n"):
    """Split on separators outside brackets and quotes (image_detail=["high"])."""
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


def parse_features(s) -> dict:
    """key -> value from an item_features string, whatever its format.

    'k=v;k=v' is the only public format. A JSON object gives its keys; lines or
    ';' outside brackets and quotes separate tokens, each split on its first
    '=' (or ':'); a token without one is a flag '1'; a string with no separator
    is one opaque value under '_'. Anything unreadable gives no keys: the
    hidden format is unverified, and no keys only means no group effects.
    """
    try:
        s = _text(s).strip()
        if not s:
            return {}
        if len(s) > FEATURE_CHARS:
            return {"_": hashlib.blake2b(s.encode("utf-8", "surrogatepass"),
                                         digest_size=16).hexdigest()}
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
            sep = "=" if "=" in t else (":" if ":" in t else None)
            k, v = t.split(sep, 1) if sep else (t, "1")
            out[k.strip()] = v.strip()
        return out
    except Exception:
        return {}


def item_groups(item, prefix=False) -> dict:
    """Features and interactors of one item, interactors' keys prefixed."""
    out = parse_features(item.get("item_features"))
    for k, v in parse_features(item.get("interactors")).items():
        out["interactors." + k] = v
    if prefix and not out:
        text = _text(item.get("item_content"))[:2000]
        if text:
            out["_prefix"] = hashlib.blake2b(text.encode("utf-8", "surrogatepass"),
                                             digest_size=8).hexdigest()
    return out


def select_keys(feature_dicts) -> list:
    """Keys whose values over these items are neither all numeric nor constant.

    Numeric keys (problem_idx) cost ALC as categories and explain 2% as ordinals;
    a constant key is the level again. A key missing on some items gets its own
    '<missing>' level. Past MAX_KEYS, the keys with fewest levels win.

    A key with about a level per item (more levels than UNIQUE[1] of at least
    UNIQUE[0] items) is dropped too: it names items rather than grouping them.
    Below that size such a key costs nothing, because a level only one labeled
    item carries is folded into that item's residual (_Fit), and a real grouping
    key can look like that in a small sample (matharena's 27 competitions over
    16 random items often do). The cut bounds the number of group columns.
    """
    keys = sorted(set().union(*feature_dicts)) if feature_dicts else []
    n, out = len(feature_dicts), []
    for k in keys:
        vals = [f.get(k, MISSING) for f in feature_dicts]
        present = [v for v in vals if v != MISSING]
        if present and all(_NUM.fullmatch(v.strip()) for v in present):
            continue
        levels = len(set(vals))
        if levels >= 2 and not (n >= UNIQUE[0] and levels > UNIQUE[1] * n):
            out.append((levels, k))
    return [k for _, k in sorted(out)[:MAX_KEYS]]


def mcq_text(text) -> str:
    """The text the MCQ floor is read from: BenchmarkFit.excerpt (at most the
    first and last 10k characters), every line stripped, blank lines dropped.

    mcq's patterns anchor '^\\s*' at every line start, and '\\s*' also runs
    over newlines, so a run of n blank lines costs n^2 (20k newlines: 13 s, the
    step-2 review), and a 150k-character item 3 ms in every recreated worker.
    On stripped non-blank lines they match the same options as before: '\\s*'
    skipped exactly the whitespace that is gone. The trailing newline is kept
    only where the text had trailing whitespace, which '\\s+' after an option
    letter may need. Of the 24 public items floor_of floors, the excerpt drops
    three: researchcodebench code files whose '(A)'..'(D)' sit mid-file, the
    false positives the review found; the stripping changes none of 10,510.
    """
    t = BenchmarkFit.excerpt(_text(text))
    lines = (ln.strip() for ln in t.split("\n"))
    out = "\n".join(ln for ln in lines if ln)
    return out + "\n" if out and t[len(t.rstrip()):] else out


# --- the collapsed posterior ------------------------------------------------------

def _sig(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def _logsig(z):
    return -np.logaddexp(0.0, -z)


def _floored(p, c):
    """Score and curvature (minus the second derivative) in z of a success's
    log(c + (1 - c) p), p = sigmoid(z). The curvature turns negative where p
    is small against c: the term is not concave."""
    P = c + (1 - c) * p
    r = (1 - c) * p * (1 - p) / P
    return r, r * (p - c * (1 - p) / P)


def _mv(A, x):
    """A @ x without BLAS: once LAPACK has woken OpenBLAS's threads, a small
    product through it costs 10x more (0.25 ms for 2,676 x 20), and a target's
    prediction runs outside the fit's one-thread limit, which costs 0.7 ms
    to enter."""
    return np.einsum("...j,j->...", A, x)


def expect_sig(m, v, s2t=0.0, nu=0.0):
    """E[sigmoid(m + sqrt(v) Z + sqrt(s2t) T)], Z standard normal and T standard
    Student-t with nu degrees of freedom (only when s2t > 0)."""
    return expect_mix(np.ones(1), np.array([m], float), np.array([v], float), s2t, nu)


def expect_mix(w, m, v, s2t=0.0, nu=0.0):
    """sum_k w_k E[sigmoid(m_k + sqrt(v_k) Z + sqrt(s2t) T)], as expect_sig."""
    w, m, v = (np.asarray(a, float).ravel() for a in (w, m, v))
    sd = np.sqrt(np.maximum(v, 0.0))
    if s2t > 0:
        z = m[:, None] + sd[:, None] * _ZN[None, :]
        T, wt = t_nodes(float(nu))
        s = _mv(_sig(z[:, None, :] + math.sqrt(s2t) * T[None, :, None]), _ZW)
        return float(_mv(_mv(s, wt), w))
    out, lo = 0.0, -1.0
    for hi, z, wz in _GH_TIERS + [(math.inf, _ZN, _ZW)]:
        k = (v > lo) & (v <= hi) if lo >= 0 else v <= hi
        if k.all():
            return out + float(_mv(_mv(_sig(m[:, None] + sd[:, None] * z), wz), w))
        if k.any():
            out += float(_mv(_mv(_sig(m[k, None] + sd[k, None] * z), wz), w[k]))
        lo = hi
    return out


def _floor_array(floor, n):
    """Per-label floors from an array, or from the old tuple (fl, c_fl, f0, c_f0)."""
    if floor is None:
        return np.zeros(n)
    if isinstance(floor, tuple) and len(floor) == 4:
        c = np.zeros(n)
        c[np.asarray(floor[0], np.int64)] = floor[1]
        c[np.asarray(floor[2], np.int64)] = floor[3]
        return c
    return np.asarray(floor, float).reshape(n)


class _Part:
    """The items one quadrature rule integrates, their labels grouped by item:
    `row` is each label's item among `items`. K nodes per item: adaptive
    Gauss-Hermite, or (grid) the fixed trapezoid of an item that holds a
    floored success."""

    def __init__(self, items, labels, row, grid):
        self.items, self.labels, self.row, self.grid = items, labels, row, grid
        self.K = GRID if grid else GH


def _parts(item, n_items, c, y):
    n = len(item)
    hard = np.zeros(n_items, bool)
    if n:
        hard[item[(c > 0) & (y > 0.5)]] = True
    order = np.argsort(item, kind="stable")
    out = []
    for grid in (False, True):
        lab = order[hard[item[order]] == grid]
        if len(lab):
            items = np.unique(item[lab])
            out.append(_Part(items, lab, np.searchsorted(items, item[lab]), grid))
    return out


class _Items:
    """Each item's residual e_i integrated against its labels, given eta.

    The integrand's mode by safeguarded Newton (a bracket on the sign of the
    derivative always holds a maximum). Then per _Part: Gauss-Hermite nodes
    scaled by the curvature at the mode, or for an item with a floored success
    a trapezoid over [-8.5, 8.5] prior sds widened to cover the mode's +-8 sds.
    Q are the normalised node weights, i.e. the item's posterior over e_i;
    every moment below is under Q. `floor` holds each label's floor; a
    success's likelihood is c + (1 - c) sigmoid.
    """

    def __init__(self, eta, y, item, n_items, s2e, e0=None, floor=None, parts=None):
        n = len(y)
        s2e = np.asarray(s2e, float)
        c = _floor_array(floor, n)
        s1 = np.flatnonzero((c > 0) & (y > 0.5))
        cnt = np.bincount(item, minlength=n_items).astype(float)
        hi = cnt * s2e + 1.0
        lo = -hi
        e = np.zeros(n_items) if e0 is None else np.clip(e0, lo, hi)
        for _ in range(200):
            p = _sig(eta - e[item])
            r, w = y - p, p * (1 - p)
            if len(s1):
                r[s1], w[s1] = _floored(p[s1], c[s1])
            d1 = np.bincount(item, -r, n_items) - e / s2e
            d2 = np.bincount(item, np.maximum(w, 0.0), n_items) + 1 / s2e
            lo = np.where(d1 > 0, e, lo)
            hi = np.where(d1 < 0, e, hi)
            new = e + d1 / d2
            out = (new < lo) | (new > hi)
            new = np.where(out, 0.5 * (lo + hi), new)
            step = float(np.max(np.abs(new - e))) if n_items else 0.0
            e = new
            if step < 1e-12:
                break
        p = _sig(eta - e[item])
        w = p * (1 - p)
        if len(s1):
            w[s1] = _floored(p[s1], c[s1])[1]
        h = np.maximum(np.bincount(item, w, n_items) + 1 / s2e, 0.25 / s2e)
        sd = 1 / np.sqrt(h)
        self.mode = e
        self.parts = _parts(item, n_items, c, y) if parts is None else parts
        self.logL, self.ebar, self.evar = np.zeros(n_items), np.zeros(n_items), s2e.copy()
        self.g, self.Ew, self.var_r, self.ce = (np.zeros(n) for _ in range(4))
        self.Q, self.E, self.dev = [], [], []
        self.where = np.full((n_items, 2), -1, np.int64)
        for pi, part in enumerate(self.parts):
            ii, lab, row, K = part.items, part.labels, part.row, part.K
            v = s2e[ii]
            if part.grid:
                sp = np.sqrt(v)
                a = np.minimum(-8.5 * sp, e[ii] - 8 * sd[ii])
                b = np.maximum(8.5 * sp, e[ii] + 8 * sd[ii])
                E = a[:, None] + (b - a)[:, None] * _GU[None, :]
                lw = np.log((b - a) / (K - 1))[:, None]
            else:
                E = e[ii][:, None] + _SQ2 * sd[ii][:, None] * _T[None, :]
                lw = _LW[None, :] + np.log(_SQ2 * sd[ii])[:, None]
            lw = lw - E ** 2 / (2 * v[:, None]) - 0.5 * (_LOG2PI + np.log(v))[:, None]
            yl, cl = y[lab], c[lab]
            Z = eta[lab][:, None] - E[row]
            P = _sig(Z)
            ll = _logsig(np.where(yl > 0.5, 1.0, -1.0)[:, None] * Z)
            R = yl[:, None] - P
            Wn = P * (1 - P)
            f1 = np.flatnonzero((cl > 0) & (yl > 0.5))
            f0 = np.flatnonzero((cl > 0) & (yl <= 0.5))
            if len(f1):
                cc = cl[f1][:, None]
                ll[f1] = np.logaddexp(np.log(cc), np.log1p(-cc) + ll[f1])
                R[f1], Wn[f1] = _floored(P[f1], cc)
            if len(f0):
                ll[f0] += np.log1p(-cl[f0])[:, None]
            F = np.bincount((row[:, None] * K + np.arange(K)).ravel(), ll.ravel(),
                            len(ii) * K).reshape(len(ii), K)
            L = lw + F
            mx = L.max(1)
            Qu = np.exp(L - mx[:, None])
            tot = Qu.sum(1)
            Q = Qu / tot[:, None]
            q = Q[row]
            g = (q * R).sum(1)
            dev = R - g[:, None]
            eb = (Q * E).sum(1)
            self.logL[ii] = mx + np.log(tot)
            self.g[lab] = g
            self.Ew[lab] = (q * Wn).sum(1)
            self.var_r[lab] = (q * dev ** 2).sum(1)
            self.ebar[ii] = eb
            self.evar[ii] = (Q * (E - eb[:, None]) ** 2).sum(1)
            self.ce[lab] = (q * (E[row] - eb[row][:, None]) * dev).sum(1)    # dE[e]/d eta
            self.Q.append(Q)
            self.E.append(E)
            self.dev.append(dev)
            self.where[ii, 0] = pi
            self.where[ii, 1] = np.arange(len(ii))

    def curvature(self, delta):
        """Minus the second derivative of each item's log L as every label's
        eta moves by delta_o t, at t = 0: sum_o delta_o^2 E[w_o] less the
        variance, under the item's residual posterior, of sum_o delta_o R_o
        (the item's block of Problem.precision, along delta)."""
        out = np.zeros(len(self.logL))
        for pi, part in enumerate(self.parts):
            lab, row, K, n = part.labels, part.row, part.K, len(part.items)
            dl = delta[lab]
            S = np.bincount((row[:, None] * K + np.arange(K)).ravel(),
                            (dl[:, None] * self.dev[pi]).ravel(), n * K).reshape(n, K)
            out[part.items] = np.bincount(row, dl * dl * self.Ew[lab], n) - (self.Q[pi] * S * S).sum(1)
        return out

    def nodes(self, j):
        """Item j's posterior over its residual: (weights, nodes); for an
        array of items in one part, one row each."""
        pi, r = self.where[j].T
        pi = int(np.ravel(pi)[0])
        return self.Q[pi][r], self.E[pi][r]


class _Nodes:
    """The part of an _Items a target reads: each item's residual posterior."""
    __slots__ = ("Q", "E", "where")
    nodes = _Items.nodes

    def __init__(self, it):
        self.Q, self.E, self.where = it.Q, it.E, it.where

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.Q + self.E) + self.where.nbytes


class _State:
    def __init__(self, x, it, lp):
        self.x, self.it, self.lp = x, it, lp


def _within_pairs(item, labels):
    """Ordered pairs (o, o') of distinct labels on the same item."""
    lab = labels[np.argsort(item[labels], kind="stable")]
    if not len(lab):
        return lab, lab
    it = item[lab]
    starts = np.flatnonzero(np.r_[True, it[1:] != it[:-1]])
    sizes = np.diff(np.r_[starts, len(lab)])
    rep = np.repeat(sizes, sizes)
    po = np.repeat(lab, rep)
    offs = np.arange(len(po)) - np.repeat(np.cumsum(rep) - rep, rep)
    qo = lab[np.repeat(np.repeat(starts, sizes), rep) + offs]
    keep = po != qo
    return po[keep], qo[keep]


class Problem:
    """log p(x | labels) up to a constant, every item residual integrated out.

    x ~ N(m, diag(V)); label o has eta_o = sum_s vals[o, s] * x[cols[o, s]] + off[o]
    - e_{item[o]} with e_i ~ N(0, s2e[i]), and P(y_o = 1) = c_o + (1 - c_o)
    sigmoid(eta_o) with floor c_o (default 0). Unused slots carry value 0. A
    coordinate with nu[k] > 0 has a Student-t prior instead, with nu[k] degrees
    of freedom and scale sqrt(V[k]). Shared by the run-time fit and the offline
    empirical Bayes (paiec.prior), and small enough to check against finite
    differences. After solve(), `converged` says whether the Newton decrement
    got below 1e-8 and `decrement` holds its last value.
    """

    def __init__(self, cols, vals, y, item, s2e, m, V, nu=None, floor=None, off=None):
        self.cols = np.asarray(cols, np.int64).reshape(len(y), -1)
        self.vals = np.asarray(vals, float).reshape(len(y), -1)
        self.y = np.asarray(y, float)
        self.item = np.asarray(item, np.int64)
        self.s2e = np.asarray(s2e, float)
        self.m = np.asarray(m, float)
        self.V = np.asarray(V, float).copy()
        self.p, self.n_items = len(self.m), len(self.s2e)
        n = len(self.y)
        self.off = np.zeros(n) if off is None else np.asarray(off, float).reshape(n)
        c = np.zeros(n) if floor is None else np.clip(np.asarray(floor, float).reshape(n), 0, 0.99)
        self.c = c
        self.floors = c if np.any(c > 0) else None
        self.nu = np.zeros(self.p) if nu is None else np.asarray(nu, float).reshape(self.p).copy()
        self.t = np.flatnonzero(self.nu > 0)
        cnt = np.bincount(self.item, minlength=self.n_items)
        self.single = cnt[self.item] == 1
        self.multi = np.flatnonzero(cnt >= 2)
        S = self.cols.shape[1]
        self.pidx = (self.cols[:, :, None] * self.p + self.cols[:, None, :]).reshape(n, S * S)
        self.pval = (self.vals[:, :, None] * self.vals[:, None, :]).reshape(n, S * S)
        self.parts = _parts(self.item, self.n_items, c, self.y)
        self.plans = [self.plan(part, cnt) for part in self.parts]
        self.converged, self.decrement = None, math.inf

    def plan(self, part, cnt, dense=None):
        """How precision() subtracts the within-item score covariance of this
        part's multi-label items: as K dense rows per item over all p columns,
        or pair by pair over the columns the labels touch, whichever costs less
        (rough ns per element of each). None when no item has two labels."""
        multi = np.flatnonzero(cnt[part.items] >= 2)
        if not len(multi):
            return None
        mo = np.flatnonzero(cnt[part.items[part.row]] >= 2)
        k = cnt[part.items[multi]].astype(float)
        S = self.cols.shape[1]
        if dense is None:
            dense = len(multi) * part.K * self.p * (0.1 * self.p + 3) <= 20 * (k * (k - 1)).sum() * S * S
        plan = SimpleNamespace(multi=multi, mo=mo, dense=bool(dense))
        if plan.dense:
            idx = np.full(len(part.items), -1)
            idx[multi] = np.arange(len(multi))
            plan.mrow = idx[part.row[mo]]
        else:
            plan.po, plan.qo = _within_pairs(part.row, mo)
        return plan

    @property
    def wfull(self):
        """Labels whose own score variance the dense rows already subtract."""
        out = np.zeros(len(self.y), bool)
        for part, plan in zip(self.parts, self.plans):
            if plan is not None and plan.dense:
                out[part.labels[plan.mo]] = True
        return out

    def retune(self, k, V, nu=0.0):
        """Coordinate k's prior: variance (or squared t scale) V, nu df."""
        self.V[k], self.nu[k] = V, nu
        self.t = np.flatnonzero(self.nu > 0)

    def eta(self, x):
        if not len(self.y):
            return np.zeros(0)
        return np.einsum("os,os->o", self.vals, x[self.cols]) + self.off

    def prior(self, x):
        """log prior (up to a constant), its gradient, and two curvatures of
        minus the log prior: the exact one and the one Newton uses. They differ
        only on Student-t coordinates, where the second is (nu+1) / (nu V + r^2):
        the precision of the Gaussian that the t's scale mixture holds at its
        current weight, positive where the exact curvature turns negative
        (r^2 > nu V), which keeps Newton an ascent method and lets labels that
        contradict the prior widen the posterior rather than break it."""
        r = x - self.m
        lp = -0.5 * r * r / self.V
        g = -r / self.V
        w = 1.0 / self.V
        c = w.copy()
        t = self.t
        if len(t):
            nu, V, rt = self.nu[t], self.V[t], r[t]
            q = nu * V + rt * rt
            lp[t] = -0.5 * (nu + 1) * np.log1p(rt * rt / (nu * V))
            g[t] = -(nu + 1) * rt / q
            w[t] = (nu + 1) / q
            c[t] = (nu + 1) * (nu * V - rt * rt) / (q * q)
        return float(lp.sum()), g, w, c

    def state(self, x, e0=None) -> _State:
        x = np.asarray(x, float)
        it = _Items(self.eta(x), self.y, self.item, self.n_items, self.s2e, e0, self.c, self.parts)
        return _State(x, it, float(it.logL.sum()) + self.prior(x)[0])

    def grad(self, st):
        g = np.bincount(self.cols.ravel(), (self.vals * st.it.g[:, None]).ravel(), self.p)
        return g + self.prior(st.x)[1]

    def precision(self, st, exact=False, laplace=False):
        """Minus the Hessian: A'diag(E w)A minus the covariance of the label
        scores within each item, plus the prior's curvature: Newton's (see
        prior()), the exact one (exact=True, for checking), or for the Laplace
        covariance (laplace=True) the exact one wherever it is positive, which
        keeps a t level's posterior variance from being understated."""
        it = st.it
        W = np.where(self.wfull, it.Ew, it.Ew - it.var_r)
        P = np.bincount(self.pidx.ravel(), (self.pval * W[:, None]).ravel(),
                        self.p * self.p).reshape(self.p, self.p)
        for pi, plan in enumerate(self.plans):
            if plan is not None:
                P -= self._dense_cov(it, pi) if plan.dense else self._pair_cov(it, pi)
        _, _, w, c = self.prior(st.x)
        d = c if exact else (np.where(c > 0, c, w) if laplace else w)
        P[np.diag_indices(self.p)] += d
        return 0.5 * (P + P.T)

    def _dense_cov(self, it, pi):
        part, plan = self.parts[pi], self.plans[pi]
        K, gl = part.K, part.labels[plan.mo]
        rows = plan.mrow[:, None] * K + np.arange(K)
        idx = rows[:, :, None] * self.p + self.cols[gl][:, None, :]
        w = it.dev[pi][plan.mo][:, :, None] * self.vals[gl][:, None, :]
        Z = np.bincount(idx.ravel(), w.ravel(), len(plan.multi) * K * self.p)
        Z = Z.reshape(len(plan.multi) * K, self.p) * np.sqrt(it.Q[pi][plan.multi].ravel())[:, None]
        return Z.T @ Z

    def _pair_cov(self, it, pi):
        """Off-diagonal within-item score covariances, pair by pair: each pair
        touches only its two labels' columns, so the cost does not grow with
        the number of group levels the way dense rows over all p columns do."""
        part, plan = self.parts[pi], self.plans[pi]
        S = self.cols.shape[1]
        Q, dev = it.Q[pi], it.dev[pi]
        out = np.zeros(self.p * self.p)
        step = max(1, 2_000_000 // (S * S + part.K))
        for a in range(0, len(plan.po), step):
            po, qo = plan.po[a:a + step], plan.qo[a:a + step]
            C = np.einsum("pk,pk,pk->p", Q[part.row[po]], dev[po], dev[qo])
            gp, gq = part.labels[po], part.labels[qo]
            idx = self.cols[gp][:, :, None] * self.p + self.cols[gq][:, None, :]
            val = self.vals[gp][:, :, None] * self.vals[gq][:, None, :] * C[:, None, None]
            out += np.bincount(idx.ravel(), val.ravel(), self.p * self.p)
        return out.reshape(self.p, self.p)

    def solve(self, x0=None, e0=None, max_iter=100, tol=1e-14):
        """Newton with backtracking to the posterior mode: (state, precision).

        Close to the mode (Newton decrement below NEAR = 1e-5, a step shorter
        than 0.01 at any curvature the model reaches) the full step is taken
        without a line search: there the quadrature's noise in log L, which
        its adaptive nodes make ~1e-6 on one-sided items, is larger than the
        ascent Armijo would ask for (the step-2 review saw it give up at 1e-7
        on 24 of 40 floored fits), while the gradient, an exact posterior
        moment, keeps converging quadratically (linearly along a Student-t
        coordinate, whose curvature is the mixture's, not the exact one). The
        fit is `converged` once the decrement is below 1e-8. Close steps stop
        after six once it is, and otherwise go on up to CLOSE_MAX: a linearly
        converging t fit stopped at six with the decrement still near 1e-7 in
        24 of 1,440 checkpoints (final review), counted as unconverged although
        iterating on moved no prediction by more than 1e-5.
        """
        st = self.state(self.m.copy() if x0 is None else x0, e0)
        close = 0
        for _ in range(max_iter):
            g, P = self.grad(st), self.precision(st)
            step = _spd_solve(P, g)
            dec = float(g @ step)
            if not math.isfinite(dec):
                raise FloatingPointError("non-finite Newton step")
            self.decrement = dec
            self.converged = dec < 1e-8
            if dec < tol or (close >= 6 and self.converged) or close >= CLOSE_MAX:
                return st, P
            if dec < NEAR:
                st, close = self.state(st.x + step, st.it.mode), close + 1
                continue
            t = 1.0
            while True:
                new = self.state(st.x + t * step, st.it.mode)
                if new.lp >= st.lp + 1e-4 * t * dec:
                    break
                t *= 0.5
                if t < 1e-6:        # no ascent left above the quadrature's noise
                    return st, P
            st = new
        self.converged = False
        return st, self.precision(st)


def _cholesky(P):
    """Cholesky factor of P, shifted up to positive definite when it is not:
    round-off, or a Newton step far from the mode of a floored likelihood,
    where the shifted matrix still gives an ascent direction."""
    try:
        return np.linalg.cholesky(P), P
    except np.linalg.LinAlgError:
        pass
    scale = max(1.0, float(np.max(np.abs(np.diag(P))))) if len(P) else 1.0
    shift = max(0.0, -float(np.linalg.eigvalsh(P)[0])) + 1e-8 * scale
    for _ in range(12):
        Ps = P + shift * np.eye(len(P))
        try:
            return np.linalg.cholesky(Ps), Ps
        except np.linalg.LinAlgError:
            shift *= 10
    raise np.linalg.LinAlgError("precision could not be made positive definite")


def _spd_solve(P, g):
    _, Ps = _cholesky(P)
    return np.linalg.solve(Ps, g)


def _inverse(P):
    L, _ = _cholesky(P)
    Li = np.linalg.inv(L)
    return Li.T @ Li


@lru_cache(maxsize=8)
def t_nodes(nu):
    """Nodes and weights for E[f(T)], T standard Student-t with nu degrees of
    freedom: T = sqrt(nu) tan(u), whose density in u is proportional to
    cos(u)^(nu - 1) on (-pi/2, pi/2), by the midpoint rule. A bounded f goes
    flat at both ends, so TN = 64 nodes are exact to 1e-8 or better for the
    sigmoids expect_sig() takes (tests/test_hier.py)."""
    u = (np.arange(TN) + 0.5) / TN * math.pi - math.pi / 2
    w = np.cos(u) ** (nu - 1)
    return math.sqrt(nu) * np.tan(u), w / w.sum()


@lru_cache(maxsize=8)
def lam_nodes(nu):
    """(lambda_k, log weight_k) for E[f(lambda)], lambda ~ Gamma(nu/2, rate nu/2):
    the trapezoid rule in u = log lambda, where the density is exp(a u - a e^u)
    (a = nu / 2) up to a constant, over the range where a (e^u - u - 1) <= 14.
    In u the integrand is smooth and has no endpoint singularity, which
    generalized Gauss-Laguerre in lambda has when the labels pin the level
    (the evidence then goes as sqrt(lambda)): on a Gaussian-evidence check
    LAM = 16 nodes put the level's posterior mean within 1e-4 of exact, where
    16 Laguerre nodes were 8e-2 off (step-2 fix scratch lam_quad.py)."""
    a = nu / 2

    def edge(sign):
        lo, hi = 0.0, 1.0
        while a * (math.exp(sign * hi) - sign * hi - 1) < 14:
            hi *= 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if a * (math.exp(sign * mid) - sign * mid - 1) < 14 else (lo, mid)
        return sign * hi
    u = np.linspace(edge(-1), edge(1), LAM)
    lw = a * u - a * np.exp(u)
    lw = lw - lw.max()
    return np.exp(u), lw - np.log(np.exp(lw).sum())


# --- one labeled list, fitted ------------------------------------------------------

class _TooBig(Exception):
    """The labeled list needs more than MAX_COLUMNS columns."""


class _Post:
    """One Gaussian approximation to the posterior of x: its mode, covariance,
    the item residuals' posteriors given x there, and the prior (V, nu) it
    was fitted under."""

    def __init__(self, x, cov, it, V=None, nu=None):
        self.x, self.cov, self.it = x, cov, it
        self.V, self.nu = V, nu
        self.eta = None
        if it is not None:
            it.dev = None           # only the fit needed the per-node scores


def _prior_curvature(x, m, V, nu):
    """Minus the second derivative of the log prior per coordinate, as the
    Laplace covariance takes it (Problem.precision(laplace=True)): 1 / V, and
    on a Student-t coordinate its exact curvature where positive, else the
    scale mixture's."""
    out = 1.0 / np.asarray(V, float)
    t = np.flatnonzero(nu > 0)
    if len(t):
        r2 = (x[t] - m[t]) ** 2
        q = nu[t] * V[t] + r2
        c = (nu[t] + 1) * (nu[t] * V[t] - r2) / (q * q)
        out[t] = np.where(c > 0, c, (nu[t] + 1) / q)
    return out


def _log_prior(X, m, V, nu):
    """log prior of each row of X, as Problem.prior, up to the same constant."""
    r = X - m
    lp = -0.5 * r * r / V
    t = np.flatnonzero(nu > 0)
    if len(t):
        lp[..., t] = -0.5 * (nu[t] + 1) * np.log1p(r[..., t] ** 2 / (nu[t] * V[t]))
    return lp.sum(-1)


class _Fit:
    """Every component some label touches, fitted jointly; the rest stay at
    their prior, so one fit serves every target of a checkpoint. `check`
    raises _TooBig past MAX_COLUMNS; `delta` overrides Flags.delta."""

    def __init__(self, recs, model, check=True, delta=None):
        self.model = model
        h, cfg = model.hyper, model.cfg
        self.delta = cfg.delta if delta is None else delta
        self.col, m, V, nu = {}, [], [], []

        def add(key, mean, var, df=0.0):
            if key not in self.col:
                self.col[key] = len(m)
                m.append(mean)
                V.append(max(var, 1e-8))        # a zero variance pins it to its mean
                nu.append(df)
            return self.col[key]

        items, subjects = {}, {}
        for r in recs:
            items.setdefault(r[2], (r[0], r[5]))
            subjects.setdefault(r[1], r[4])
        by_b = {}
        for ik, (b, _) in items.items():
            by_b.setdefault(b, []).append(ik)
        self.keys = {}
        if cfg.groups:
            for b, iks in by_b.items():
                self.keys[b] = select_keys([model.groups_of(ik, items[ik][1]) for ik in iks])
        carriers = {}
        for ik, (b, it) in items.items():
            ks = self.keys.get(b) or ()
            if ks:
                g = model.groups_of(ik, it)
                for kk in ks:
                    carriers.setdefault(("g", b, kk, g.get(kk, MISSING)), []).append(ik)
        # a level only one labeled item carries is folded into that item; past
        # MAX_LEVELS levels of a key, the rarest are dropped: each carrier's
        # residual takes the level's variance, so they share nothing
        self.folded = {lev: iks[0] for lev, iks in carriers.items() if len(iks) == 1}
        per_key = {}
        for lev, iks in carriers.items():
            if len(iks) >= 2:
                per_key.setdefault(lev[:3], []).append((-len(iks), lev[3], lev))
        self.dropped = set()
        for levs in per_key.values():
            if len(levs) > MAX_LEVELS:
                self.dropped.update(lev for _, _, lev in sorted(levs)[MAX_LEVELS:])
        extra_var = {}
        for lev, iks in carriers.items():
            if lev in self.folded or lev in self.dropped:
                for ik in iks:
                    extra_var[ik] = extra_var.get(ik, 0.0) + h.sigma_g ** 2 / len(self.keys[lev[1]])
        self.text = {}
        if cfg.text:
            for b, iks in by_b.items():
                if len(iks) >= cfg.warmup:
                    try:        # no scikit-learn, or texts TF-IDF cannot use: no text term
                        self.text[b] = model.embed(b, [items[ik][1] for ik in iks],
                                                   self.keys.get(b))
                    except Exception:
                        pass
        self.s2e_b = {b: model.item_var(b, self.keys.get(b), b in self.text) for b in by_b}

        # theta: one per theta key (the identity key, or the pair when linking
        # is off), its prior the mean over the subject dicts sharing it; each
        # dict's own offset from that mean goes to its delta, or with delta off
        # into a fixed offset of its labels
        tp = {sk: model.theta_prior(sk, s) for sk, s in subjects.items()}
        tk = {(b, sk): model.theta_key(b, sk, subjects[sk]) for b, sk in {r[:2] for r in recs}}
        shared = {}
        for (b, sk), key in sorted(tk.items()):
            shared.setdefault(key, {})[sk] = tp[sk]
        acc = {key: (sum(v[0] for v in d.values()) / len(d), sum(v[1] for v in d.values()) / len(d))
               for key, d in shared.items()}
        self.theta_mean = {}
        dv = h.sigma_delta ** 2
        th_extra = 0.0 if self.delta else dv

        rows_c, rows_v, ys, its, floors, offs = [], [], [], [], [], []
        item_idx = {}
        for bid, sk, ik, y, s, it in recs:
            lev = ("mu", bid) if cfg.pool_mu else ("mu", bid, sk)
            c = [add(lev, h.mu0, h.sigma_mu ** 2, h.nu_mu)]
            v = [1.0]
            key = tk[(bid, sk)]
            mean, var = acc[key]
            ct = add(key, mean, var + th_extra)
            self.theta_mean[ct] = mean
            c.append(ct)
            v.append(1.0)
            if self.delta:
                c.append(add(("de", bid, sk), tp[sk][0] - mean, dv))
                v.append(1.0)
            offs.append(0.0 if self.delta else tp[sk][0] - mean)
            keys = self.keys.get(bid) or []
            if keys:
                g = model.groups_of(ik, it)
                for kk in keys:
                    gl = ("g", bid, kk, g.get(kk, MISSING))
                    if gl not in self.folded and gl not in self.dropped:
                        c.append(add(gl, 0.0, h.sigma_g ** 2 / len(keys)))
                        v.append(-1.0)
            tx = self.text.get(bid)
            if tx is not None:
                _, X, index, tvar = tx
                for j, xj in enumerate(X[index[ik]]):
                    c.append(add(("tx", bid, j), 0.0, tvar))
                    v.append(-float(xj))
            rows_c.append(c)
            rows_v.append(v)
            ys.append(y)
            floors.append(model.floor_item(ik, it))
            its.append(item_idx.setdefault(ik, len(item_idx)))
        if check and len(m) > MAX_COLUMNS:
            raise _TooBig(len(m))
        S = max((len(c) for c in rows_c), default=1)
        cols = np.zeros((len(recs), S), np.int64)
        vals = np.zeros((len(recs), S))
        for o, (c, v) in enumerate(zip(rows_c, rows_v)):
            cols[o, :len(c)] = c
            vals[o, :len(v)] = v
        self.item_idx = item_idx
        s2e = np.zeros(len(item_idx))
        for ik, j in item_idx.items():
            s2e[j] = self.s2e_b[items[ik][0]] + extra_var.get(ik, 0.0)
        self.post = _Post(np.array(m, float), np.diag(V) if V else np.zeros((0, 0)), None)
        self.mix, self.converged, self.prob = {}, True, None
        self.lines = _Memo(4096, chars=LINE_BYTES)      # _line, per posterior and direction
        if recs:
            prob = Problem(cols, vals, np.array(ys, float), np.array(its), s2e,
                           np.array(m, float), np.array(V, float), np.array(nu, float),
                           np.array(floors, float), np.array(offs, float))
            with model.threads():
                st, P = prob.solve()
                self.converged = bool(prob.converged)
                if len(prob.t):
                    P = prob.precision(st, laplace=True)
                cov = _inverse(P)
                if not (np.all(np.isfinite(st.x)) and np.all(np.isfinite(cov))):
                    raise FloatingPointError("non-finite fit")
                if len(prob.t) and cfg.t_mixture:
                    self.mix = self._scale_mixture(prob, st)
            self.post = _Post(st.x, cov, st.it, prob.V.copy(), prob.nu.copy())
            # what only the fit needed: labels x slots^2, ~100 MB each dense with text
            prob.pidx = prob.pval = prob.plans = None
            self.prob = prob
            order = np.argsort(prob.item, kind="stable")
            self._order = order
            self._starts = np.searchsorted(prob.item[order], np.arange(len(item_idx) + 1))
            self._count = np.diff(self._starts)

    def _scale_mixture(self, prob, st):
        """{level column: [(weight, _Post)]}: each Student-t level labels
        touched as a mixture over LAM Gaussian levels N(mu0, V / lambda_k),
        the model refitted at each (warm, from the neighbouring node) and
        weighted by prior weight times Laplace evidence, lp at the mode less
        half the log determinants of the level's prior variance and of the
        posterior precision. Other t levels keep their Laplace fit."""
        lam, lw = lam_nodes(float(self.model.hyper.nu_mu))
        out = {}
        for c in [int(k) for k in prob.t]:
            V0, nu0 = float(prob.V[c]), float(prob.nu[c])
            r = float(st.x[c] - prob.m[c])
            k0 = int(np.argmin(np.abs(np.log(lam) - math.log((nu0 + 1) / (nu0 + r * r / V0)))))
            posts, logz = [None] * LAM, np.full(LAM, -np.inf)
            try:
                for seq in (range(k0, LAM), range(k0 - 1, -1, -1)):
                    x0, e0 = st.x, st.it.mode
                    for k in seq:
                        prob.retune(c, V0 / lam[k], 0.0)
                        s, P = prob.solve(x0, e0)
                        self.converged &= bool(prob.converged)
                        if len(prob.t):
                            P = prob.precision(s, laplace=True)
                        L, _ = _cholesky(P)
                        Li = np.linalg.inv(L)
                        logz[k] = lw[k] + s.lp - 0.5 * math.log(prob.V[c]) - np.log(np.diag(L)).sum()
                        x0, e0 = s.x, s.it.mode
                        posts[k] = _Post(s.x, Li.T @ Li, s.it, prob.V.copy(), prob.nu.copy())
            finally:
                prob.retune(c, V0, nu0)
            if not np.all(np.isfinite(logz)):
                raise FloatingPointError("non-finite scale mixture")
            w = np.exp(logz - logz.max())
            w /= w.sum()
            # the fewest nodes that hold all but 1e-6 of the weight
            order = np.argsort(-w, kind="stable")
            n = int(np.searchsorted(np.cumsum(w[order]), 1 - 1e-6)) + 1
            keep = sorted(order[:n])
            tot = float(w[keep].sum())
            out[c] = [(float(w[k]) / tot, posts[k]) for k in keep]
        return out

    def _J(self, post, j):
        """How labeled item j's residual moves with x: sum over its labels of
        dE[e]/d eta_o times eta_o's coefficients."""
        prob = self.prob
        o = self._order[self._starts[j]:self._starts[j + 1]]
        out = {}
        for c, v, w in zip(prob.cols[o].ravel(), prob.vals[o].ravel(),
                           np.repeat(post.it.ce[o], prob.cols.shape[1])):
            if v:
                out[int(c)] = out.get(int(c), 0.0) + float(v * w)
        return out

    def _target(self, subject, item):
        """What the target's eta is made of, apart from x: its coefficients on
        x, the prior mean and variance of what no label touched, the squared
        scale of an untouched Student-t level, its own labeled item (j) or the
        carriers of folded levels it shares, and its level's column."""
        model, h = self.model, self.model.hyper
        cfg = model.cfg
        bid = _text(item.get("benchmark_id"))
        sk, ik = subject_key(subject), item_key(item)
        a, m, v, s2t = {}, 0.0, 0.0, 0.0

        def touch(c, coef):
            a[c] = a.get(c, 0.0) + coef

        level = self.col.get(("mu", bid) if cfg.pool_mu else ("mu", bid, sk))
        if level is None:
            m += h.mu0
            if h.nu_mu > 0:
                s2t = h.sigma_mu ** 2
            else:
                v += h.sigma_mu ** 2
        else:
            touch(level, 1.0)
        mt, vt = model.theta_prior(sk, subject)
        c = self.col.get(model.theta_key(bid, sk, subject))
        dmean = 0.0
        if c is None:
            m, v = m + mt, v + vt + (0.0 if self.delta else h.sigma_delta ** 2)
        else:
            touch(c, 1.0)
            dmean = mt - self.theta_mean[c]
        if self.delta:
            c = self.col.get(("de", bid, sk))
            if c is None:
                m, v = m + dmean, v + h.sigma_delta ** 2
            else:
                touch(c, 1.0)
        else:
            m += dmean
        j = self.item_idx.get(ik) if self.prob is not None else None
        fold = {}               # labeled item -> variance of the levels it shares
        keys = self.keys.get(bid) or []
        if keys:
            g = model.groups_of(ik, item)
            su = h.sigma_g ** 2 / len(keys)
            for kk in keys:
                lev = ("g", bid, kk, g.get(kk, MISSING))
                c = self.col.get(lev)
                if c is not None:
                    touch(c, -1.0)
                elif lev in self.folded:
                    if j is None:       # else it is the target's own, in its residual
                        jj = self.item_idx[self.folded[lev]]
                        fold[jj] = fold.get(jj, 0.0) + su
                elif j is None:         # unseen, or dropped (then in a labeled j's residual)
                    v += su
        tx = self.text.get(bid)
        if tx is not None:
            emb, X, index, var = tx
            try:
                xt = X[index[ik]] if ik in index else model.transform(emb, item)
                if not np.all(np.isfinite(xt)):
                    raise FloatingPointError("non-finite embedding")
                for jx, xj in enumerate(xt):
                    touch(self.col[("tx", bid, jx)], -float(xj))
            except Exception:
                v += var * X.shape[1]           # the term's prior variance
        if j is None:
            v += self.s2e_b.get(bid, model.item_var(bid, None, False))
        return SimpleNamespace(a=a, m=m, v=v, s2t=s2t, j=j, fold=fold, level=level)

    def _labels(self, items):
        """Every label of these labeled items."""
        o, st = self._order, self._starts
        return np.concatenate([o[st[i]:st[i + 1]] for i in items]) if len(items) else o[:0]

    def _line(self, post, a):
        """The posterior of s = a'x beyond its Gaussian approximation, or None.

        The joint log posterior evaluated along x(s) = x_hat + Sigma a (s - s_hat)
        / a'Sigma a, the mean of x given s under the Gaussian, whose own
        conditional density is then the same at every s (INLA's Gaussian
        conditional-mean strategy). For one pair the logistic likelihood sees x
        through a'x alone, so this is the exact marginal; the Laplace fit, whose
        mode sits nearer the prior than the mean of a skewed posterior, was up
        to 0.02 too close to the prior after a few labels. The items whose
        labels the line moves by at least LINE_FRAC of the most moved are
        integrated at every grid point, the most moved first as far as
        LINE_MAX labels go (None, the Gaussian, only when the most moved item
        alone has more). The rest keep their log likelihood to second order
        at the mode, the curvature being what is left of the Gaussian's
        1 / sd^2 along the line once the prior's and the integrated items' own
        are taken out (it counts the within-item score covariance too).
        Returns the grid in blocks (z, weights, the items along the line), the
        hit items' positions, sd and direction.
        """
        key = tuple(sorted(a.items()))
        got = self.lines.get((id(post), key), False)
        if got is not False:
            return got
        prob, line = self.prob, None
        cs = [c for c, _ in key]
        av = np.array([v for _, v in key])
        Sa = _mv(post.cov[:, cs], av)
        s2 = float(_mv(Sa[cs], av))
        if s2 > 1e-10 and len(prob.y):
            d = Sa / s2
            delta = np.einsum("os,os->o", prob.vals, d[prob.cols])
            big = float(np.max(np.abs(delta)))
            hit = self._hit(np.abs(delta), big) if big > 1e-9 else None
            if hit is not None and len(hit):
                line = self._integrate(post, hit, self._labels(hit), d, delta, math.sqrt(s2))
        size = 64 if line is None else sum(b[2].nbytes + b[1].nbytes for b in line.blocks)
        return self.lines.put((id(post), key), line, size)

    def _hit(self, move, big):
        """The labeled items a line integrates, sorted: those with a label
        moving by LINE_FRAC * big or more, the most moved first while their
        labels fit in LINE_MAX. Before the final review a line past LINE_MAX
        labels fell back to the Gaussian whole, up to 0.05 off on dense B31
        checkpoints (final_verify/d1_linemax.py). On dense B31 checkpoints
        simulated from the model the mean |p - HMC| is now 0.0040 to 0.0046,
        the same as with no cap, against 0.0051 to 0.0058 for that fallback;
        requiring the items moved by 20% of the most to fit, else the
        Gaussian, gave 0.0052 where they did not (step-2 final_fix/d2_after.py,
        d2b.py)."""
        mv = np.maximum.reduceat(move[self._order], self._starts[:-1])
        cand = np.flatnonzero(mv >= LINE_FRAC * big)
        order = cand[np.argsort(-mv[cand], kind="stable")]
        k = int(np.searchsorted(np.cumsum(self._count[order]), LINE_MAX, side="right"))
        return np.sort(order[:k])

    def _integrate(self, post, hit, labs, d, delta, sd):
        prob = self.prob
        if post.eta is None:
            post.eta = prob.eta(post.x)
        li = np.searchsorted(hit, prob.item[labs])
        y, c, s2e, e0 = prob.y[labs], prob.c[labs], prob.s2e[hit], post.it.mode[hit]
        # the labels off the line's items, to second order at the mode
        lin = float(_mv(post.it.g, delta) - _mv(post.it.g[labs], delta[labs]))
        it0 = _Items(post.eta[labs], y, li, len(hit), s2e, e0, c)
        curv = 1.0 / sd ** 2 - float(it0.curvature(delta[labs]).sum()) \
            - float(_mv(_prior_curvature(post.x, prob.m, post.V, post.nu), d * d))
        curv = max(curv, 0.0)

        def block(z):
            t, G = sd * z, len(z)
            eta = post.eta[labs][None, :] + delta[labs][None, :] * t[:, None]
            item = (np.arange(G)[:, None] * len(hit) + li[None, :]).ravel()
            it = _Items(eta.ravel(), np.tile(y, G), item, G * len(hit), np.tile(s2e, G),
                        np.tile(e0, G), np.tile(c, G))
            X = post.x[None, :] + t[:, None] * d[None, :]
            ell = it.logL.reshape(G, len(hit)).sum(1) + lin * t - 0.5 * curv * t * t + \
                _log_prior(X, prob.m, post.V, post.nu)
            return z, ell, _Nodes(it)

        blocks = [block(LINE_H * np.arange(-8, 9))]
        for _ in range(4):          # a skewed posterior's long side, until negligible
            ends = [(b[0][0], b[1][0]) for b in blocks] + [(b[0][-1], b[1][-1]) for b in blocks]
            top = max(float(b[1].max()) for b in blocks)
            lo, hi = min(ends), max(ends)
            new = []
            if lo[1] > top - 16:
                new.append(block(lo[0] - LINE_H * np.arange(11, 0, -1)))
            if hi[1] > top - 16:
                new.append(block(hi[0] + LINE_H * np.arange(1, 12)))
            if not new:
                break
            blocks += new
        top = max(float(b[1].max()) for b in blocks)
        if not math.isfinite(top):
            return None
        tot = sum(float(np.exp(b[1] - top).sum()) for b in blocks)
        return SimpleNamespace(blocks=[(z, np.exp(ell - top) / tot, it) for z, ell, it in blocks],
                               index={int(j): k for k, j in enumerate(hit)}, n=len(hit),
                               sd=sd, d=d)

    def _components(self, post, tg):
        """(weights, means, variances) of the target's eta under one posterior
        approximation: over a'x on its line (_line) or Gaussian, and over its
        own labeled item's residual (or the likeliest folded carrier's) node
        by node."""
        m = tg.m + sum(coef * float(post.x[c]) for c, coef in tg.a.items())
        v, J, it = tg.v, {}, post.it
        main, Jm = None, {}         # (k, item) read node by node, and its dE/dx
        if tg.j is not None:
            main, J = (1.0, tg.j), self._J(post, tg.j)
            Jm = J
        elif tg.fold:
            # a shared folded level u given its carrier's residual s = u + e:
            # E[u | s] = k s, Var = su (1 - k), k = su / Var(s)
            ks = {jj: su / float(self.prob.s2e[jj]) for jj, su in tg.fold.items()}
            top = max(sorted(ks), key=lambda jj: ks[jj] ** 2 * float(it.evar[jj]))
            for jj, su in sorted(tg.fold.items()):
                k = ks[jj]
                v += su * (1 - k)
                Jj = {cc: k * w for cc, w in self._J(post, jj).items()}
                if jj == top:
                    main, Jm = (k, jj), Jj
                else:
                    m -= k * float(it.ebar[jj])
                    v += k * k * float(it.evar[jj])
                for cc, w in Jj.items():
                    J[cc] = J.get(cc, 0.0) + w
        line = self._line(post, tg.a) if (self.model.cfg.line and tg.a) else None
        cs = sorted(set(tg.a) | set(J))
        av = np.array([tg.a.get(c, 0.0) for c in cs])
        jv = np.array([J.get(c, 0.0) for c in cs])
        C = post.cov[np.ix_(cs, cs)] if cs else np.zeros((0, 0))
        if line is None:
            vec = av - jv
            v = max(v + float(_mv(_mv(C, vec), vec)), 0.0)
            if main is None:
                return np.ones(1), np.array([m]), np.array([v])
            k, j = main
            Q, E = it.nodes(j)
            keep = Q > 1e-12 * Q.max()
            q = Q[keep] / Q[keep].sum()
            return q, m - k * E[keep], np.full(len(q), v)
        # along the line: a'x on the grid, the residuals' posteriors at each
        # point; the part of J'x the line does not carry stays Gaussian
        if cs:
            CJ = _mv(C, jv)
            v += max(float(_mv(CJ, jv)) - float(_mv(CJ, av)) ** 2 / line.sd ** 2, 0.0)
        jd = sum(w * float(line.d[c]) for c, w in J.items())
        jmd = sum(w * float(line.d[c]) for c, w in Jm.items())
        ws, ms = [], []
        for z, wz, lit in line.blocks:
            ok = np.flatnonzero(wz > 1e-12)
            t = line.sd * z[ok]
            if main is None:
                ws.append(wz[ok])
                ms.append(m + (1 - jd) * t)
                continue
            # a'x moves by t; the other carriers' residuals by their J'd t; the
            # main item's is read at each point (off the line's items, shifted)
            k, j = main
            pos = line.index.get(j)
            if pos is None:
                Q, E = it.nodes(j)
                Q, E = np.tile(Q, (len(ok), 1)), E[None, :] + (jmd / k) * t[:, None]
            else:
                Q, E = lit.nodes(ok * line.n + pos)
            W = wz[ok][:, None] * Q / Q.sum(1)[:, None]
            M = (m + (1 - (jd - jmd)) * t)[:, None] - k * E
            keep = W > 1e-14 * W.max()
            ws.append(W[keep])
            ms.append(M[keep])
        w = np.concatenate(ws)
        return w / w.sum(), np.concatenate(ms), np.full(len(w), max(v, 0.0))

    def mixture(self, subject, item):
        """(weights, means, variances, s2t): the target's eta as a mixture of
        Gaussians, plus sqrt(s2t) times a standard t for an untouched t level."""
        tg = self._target(subject, item)
        mix = self.mix.get(tg.level)
        if mix is None:
            w, m, v = self._components(self.post, tg)
        else:
            parts = [(wk, self._components(p, tg)) for wk, p in mix]
            w = np.concatenate([wk * c[0] for wk, c in parts])
            m = np.concatenate([c[1] for _, c in parts])
            v = np.concatenate([c[2] for _, c in parts])
        return w, m, v, tg.s2t

    def moments(self, subject, item):
        """(m, v, s2t): the mixture's mean and variance, for inspection."""
        w, m, v, s2t = self.mixture(subject, item)
        mean = float(w @ m)
        return mean, float(w @ (v + (m - mean) ** 2)), s2t


class _Split:
    """A labeled list past MAX_COLUMNS, fitted benchmark by benchmark: the
    benchmarks then share no standing, which at the default link weight they
    barely did. A single benchmark still past the budget drops delta."""

    def __init__(self, recs, model):
        by = {}
        for r in recs:
            by.setdefault(r[0], []).append(r)
        self.fits = {}
        for b, rs in by.items():
            try:
                self.fits[b] = _Fit(rs, model)
            except _TooBig:
                self.fits[b] = _Fit(rs, model, check=False, delta=False)
        self.empty = _Fit([], model)
        self.converged = all(f.converged for f in self.fits.values())

    def _fit(self, item):
        return self.fits.get(_text(item.get("benchmark_id")), self.empty)

    def mixture(self, subject, item):
        return self._fit(item).mixture(subject, item)

    def moments(self, subject, item):
        return self._fit(item).moments(subject, item)


# --- the predictor -------------------------------------------------------------------

class HierPredictor:
    """predict(input, labeled) with the offline prior and hyperparameters.

    The two must come from one set of pairs and one exclusion set
    (paiec.prior.build), since a leave-one-benchmark-out experiment would
    otherwise leak the scored benchmarks through the hyperparameters. The
    prior's meta['included'] (every benchmark its pairs held) must equal
    Hyper.included and its meta['excluded'] Hyper.excluded; with hyper=None
    (the defaults, fitted on all of PUBLIC) the prior must have been fitted on
    exactly PUBLIC with nothing excluded, and one that does not record what it
    was fitted on is refused. Anything else raises ValueError. A prior of an
    older build, without meta['included'], passes with explicit
    hyperparameters on the exclusion sets alone. prior=None cannot be checked:
    hyper=None then means the defaults, which an experiment scoring public
    benchmarks must not use.
    """

    def __init__(self, prior=None, hyper=None, keep=4, **flags):
        if prior is not None:
            held = sorted(prior.meta.get("excluded", ()))
            inc = prior.meta.get("included")
            inc = None if inc is None else sorted(inc)
            if hyper is None:
                if held or inc != sorted(PUBLIC):
                    raise ValueError(
                        f"the prior was fitted on {inc if inc is not None else 'unrecorded'} "
                        f"benchmarks without {held}, the default hyperparameters on {list(PUBLIC)}; "
                        "pass hyperparameters fitted on the same pairs (paiec.prior.build)")
            elif held != list(hyper.excluded) or (inc is not None and inc != list(hyper.included)):
                raise ValueError(f"the prior was fitted on {inc} without {held}, the "
                                 f"hyperparameters on {list(hyper.included)} without "
                                 f"{list(hyper.excluded)}")
        hyper = Hyper() if hyper is None else hyper
        self.prior = prior
        self.hyper = hyper
        self.cfg = Flags(**flags)
        if self.cfg.subject_key not in ("canon", "name", "full"):
            raise ValueError("subject_key must be canon, name or full, "
                             f"got {self.cfg.subject_key!r}")
        if hyper.nu_mu > 0 and not self.cfg.pool_mu:
            raise ValueError("a Student-t level (nu_mu > 0) needs pool_mu")
        self.keep = keep
        self._fits: OrderedDict[bytes, object] = OrderedDict()
        self._theta, self._ident = _Memo(100_000), _Memo(100_000)
        self._groups = _Memo(50_000, chars=1 << 24)
        self._floors = _Memo(200_000)
        self._lock = threading.Lock()
        self._empty = None
        self.failures = 0
        self.unconverged = 0

    # per-subject and per-item pieces, independent of labeled, memoised
    def theta_prior(self, sk, subject):
        """(mean, variance) of theta for one subject dict: attributes, then the
        identity table precision-weighted against them. The dict is read as the
        text its key hashes, so two dicts sharing a key share a prior whichever
        came first (release_date 2024 and '2024'); a prior that fails on it
        gives the no-attribute prior rather than a failed fit."""
        got = self._theta.get(sk)
        if got is None:
            got = self._theta.put(sk, self._theta_prior(subject))
        return got

    def _theta_prior(self, subject):
        h, cfg, pr = self.hyper, self.cfg, self.prior
        base = (0.0, h.sigma_theta ** 2 + h.sigma_attr ** 2)
        try:
            subject = {f: _text(subject.get(f)) for f in SUBJECT_FIELDS}
            m, V = 0.0, h.sigma_theta ** 2
            attr = cfg.attributes and pr is not None and pr.has_attributes
            if attr:
                ma, u = pr.attribute(subject)
                m, V = m + ma, V + u
            else:
                V += h.sigma_attr ** 2
            if cfg.identity and pr is not None:
                obs = pr.identity(subject, residual=attr)
                if obs is not None:
                    rbar, vid = obs
                    w = min(V / (V + vid), h.id_cap)
                    m, V = m + w * rbar, (1 - w) ** 2 * V + w * w * vid
            if math.isfinite(m) and math.isfinite(V) and V > 0:
                return m, V
        except Exception:
            pass
        return base

    def identity_key(self, sk, subject):
        got = self._ident.get(sk)
        if got is None:
            kind, key = self.cfg.subject_key, None
            if kind == "canon":
                key = canon_name(subject)
            elif kind == "name":
                key = _text(subject.get("normalized_name")).strip().lower()
            got = self._ident.put(sk, ("n", key) if key else ("k", sk))
        return got

    def theta_key(self, bid, sk, subject):
        """The column theta_s lives in: one per identity key, shared by the
        subject's pairs on every benchmark of the run, or one per pair when
        linking is off."""
        if self.cfg.link:
            return ("th", self.identity_key(sk, subject))
        return ("th", "pair", bid, sk)

    def groups_of(self, ik, item):
        got = self._groups.get(ik)
        if got is None:
            got = item_groups(item, self.cfg.prefix_groups)
            self._groups.put(ik, got, 1 + sum(len(k) + len(v) for k, v in got.items()))
        return got

    def floor_item(self, ik, item):
        """Hyper.guess times the item's MCQ floor (0 with Flags.floor off),
        the floor memoised: reading it off a 100k-character text takes a
        millisecond."""
        if not self.cfg.floor:
            return 0.0
        got = self._floors.get(ik)
        if got is None:
            got = self._floors.put(ik, float(floor_of(mcq_text(item.get("item_content")))))
        return self.hyper.guess * got

    def item_var(self, bid, keys, text):
        """Prior variance of an item residual on a benchmark: the whole item
        variance where no key carries group effects, less the text share."""
        h = self.hyper
        s2 = h.sigma_d ** 2 + (0.0 if keys else h.sigma_g ** 2)
        return s2 * (1 - h.text_share) if text else s2

    def embed(self, bid, items, keys=None):
        """Text term for one benchmark: (embedder, X, index, prior var per dim),
        the variance the share of the residual it takes."""
        texts, index = [], {}
        for it in items:
            index[item_key(it)] = len(texts)
            texts.append(BenchmarkFit.excerpt(_text(it.get("item_content"))))
        from .irt import TextEmbedder
        emb = TextEmbedder(self.cfg.dim)
        with _one_thread():
            X = emb.fit(texts)
        var = self.hyper.text_share * self.item_var(bid, keys, False) / X.shape[1]
        return emb, X, index, var

    def transform(self, emb, item):
        with _one_thread():
            return emb.transform([BenchmarkFit.excerpt(_text(item.get("item_content")))])[0]

    def threads(self):
        if self.cfg.text:
            return _one_thread()
        try:
            from threadpoolctl import threadpool_limits
            return threadpool_limits(1)
        except Exception:
            return contextlib.nullcontext()

    def fit_for(self, labeled):
        """The fit for this labeled list, from the cache when its content was
        seen. A fit that failed is cached as failed, so the targets that follow
        fall back at once instead of refitting one by one."""
        recs = records(labeled)
        fp = fingerprint(recs)
        if fp in self._fits:
            self._fits.move_to_end(fp)
        else:
            try:
                try:
                    fit = _Fit(recs, self)
                except _TooBig:
                    fit = _Split(recs, self)
            except Exception:
                self._keep(fp, None)
                raise
            self.unconverged += not fit.converged
            self._keep(fp, fit)
        fit = self._fits[fp]
        if fit is None:
            raise RuntimeError("the fit of this labeled list failed")
        return fit

    def _keep(self, fp, fit):
        self._fits[fp] = fit
        while len(self._fits) > self.keep:
            self._fits.popitem(last=False)

    def mixture(self, input, labeled=None):
        """((weights, means, variances), s2t, floor, slip) for one target:
        what predict() combines."""
        subject, item = input
        if not isinstance(subject, dict) or not isinstance(item, dict):
            raise TypeError("input must be [subject dict, item dict]")
        w, m, v, s2t = self.fit_for(labeled).mixture(subject, item)
        c = self.floor_item(item_key(item), item)
        return (w, m, v), s2t, c, (self.hyper.slip if self.cfg.slip else 0.0)

    def components(self, input, labeled=None):
        """(m, v, s2t, floor, slip): the target's eta by its mean and variance."""
        (w, m, v), s2t, c, slip = self.mixture(input, labeled)
        mean = float(w @ m)
        return mean, float(w @ (v + (m - mean) ** 2)), s2t, c, slip

    def combine(self, mix, s2t, c, slip):
        """c + (1 - c - slip) E[sigmoid(eta)], by quadrature over each Gaussian
        component and a Student-t level no label touched."""
        return c + (1 - c - slip) * expect_mix(*mix, s2t, self.hyper.nu_mu)

    def predict(self, input, labeled=None):
        """P(correct), a native float in [1e-4, 1 - 1e-4]. Never raises: a
        failure falls back to the prior-only prediction (the answer with no
        labels), then to 0.5; `failures` counts them and the first one's
        traceback goes to stderr, if stderr takes it."""
        with self._lock:
            try:
                p = _probability(self.combine(*self.mixture(input, labeled)))
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
                return self._fallback(input)

    def _fallback(self, input):
        try:
            if self._empty is None:
                self._empty = _Fit([], self)
            subject, item = input
            w, m, v, s2t = self._empty.mixture(subject, item)
            c = self.floor_item(item_key(item), item)
            p = _probability(self.combine((w, m, v), s2t, c,
                                          self.hyper.slip if self.cfg.slip else 0.0))
        except Exception:
            p = None
        return 0.5 if p is None else p


def make_hier(prior=None, hyper=None, **flags):
    """predict(input, labeled) for the evaluator; flags as in Flags."""
    return HierPredictor(prior, hyper, **flags).predict
