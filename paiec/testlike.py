"""Test-like runs: formative runs shaped like the hidden ones, from public data.

The first formative feedback (the shipped Predictor, 9 pairs) does not look like
a public run from official.sample_run. Its 9 pairs sit on 7 benchmarks and 9
subjects, with 44 to 76 evaluated items each, and their base rates are far from
0.5: B31 Brier puts them at logit mean about -1.6 and sd 1.4 (FEEDBACK,
rate_from_brier), where public R1 pairs average 0.38 with sd 0.235 in accuracy.
Its B0 Brier (0.359) says the attribute prior put most of those subjects near
p = 0.75 on pairs whose rates were low. This module draws runs that reproduce
those statistics from public pairs only, so that predictors and level priors
can be chosen for that distribution rather than for the public one.
experiments/testlike_check.py tunes the defaults on the feedback and checks the
rest; the numbers below are its check phase (300 default runs, seed 2) and its
sensitivities (100 runs each, seed 3), in results/testlike_check.json.

Pseudo-benchmarks. The five public benchmarks are too few and too central, so
each is cut into pseudo-benchmarks (Catalogue):

  group    item_features groups (PARENT_KEYS: matharena competition,
           researchcodebench paper, real_webagents website, multi_swebench lang),
           sorted by Rasch difficulty and merged, hardest first, into contiguous
           segments; a segment closes once `min_subjects` subjects hold
           `target_items` items in it, and a tail that never does joins the last
           segment. Merging neighbours in difficulty spreads the levels out.
  mix      the same groups merged in a seeded random order: levels nearer the
           parent's, item structure nearer the parent's too. Built, but drawn
           only when Regime.kinds names it.
  stratum  bands of equal item count by Rasch difficulty (halves and thirds by
           default; q1 is the hardest). Cross-fitted: a subject's band for an
           item comes from a Rasch fit on the other subject folds only, so which
           of its own items land in a stratum never depends on its own labels.
  whole    the parent itself.
  chunk    a benchmark with fewer than `min_subjects` subjects (swe_rebench:
           one subject, 6,306 items, no features) is cut into random chunks of
           `chunk_items` items instead: it has no difficulty to rank by.

A pseudo-benchmark gets its own name, parent + SEP + tag, hence its own
anonymous benchmark_id in paiec.official, and keeps its parent (parent_of) so
anything fitted offline for a run can leave the parent out. Every label is a
real recorded response; only the grouping of items is new. Pairs are
(subject, pseudo-benchmark) with at least 80 items (MIN_ITEMS, not relaxed).

Item structure is thinner than in public benchmarks. Strata cut item-difficulty
variance by construction, and merging groups of similar difficulty removes the
feature key's share of it. On the evaluated responses, an item-level oracle
(item_oracle: the parent's Rasch difficulty, theta fitted per pair) gains 41% of
the pair-rate oracle's Brier on default runs against 55% on public R1 runs; 14%
on strata (28% of default pairs, and the two most drawn pseudo-benchmarks,
real_webagents::q1of2 in 40% of runs and researchcodebench::q1of2 in 36%), 50%
on groups, 53% on wholes. The sd of item difficulty within a pair is 1.92
against 2.37. The key's share of item-difficulty variance (key_share) falls from
0.41 in matharena to 0.00-0.12 in its seven groups, from 0.38 to 0.15/0.26 in
researchcodebench, from 0.22 to 0.11/0.04 in real_webagents; each
multi_swebench group holds one lang. Strata hold item-difficulty variance of
0.66 to 4.1 against 4.0 to 8.2 in their parents. Without strata
(kinds=("group", "whole")) the gain is 51%; with groups merged at random and no
strata (kinds=("mix", "whole")) 53%, key shares back near the parent's (0.06 to
0.48 in matharena), and the feedback's B0/B1 still matched (distance 0.49 run
sds, 0.39 by default). The hidden benchmarks' item structure is unknown, and
the feedback cannot show it: the shipped Predictor's IRT needs 64 labeled items
of a benchmark, so its B31 is about p(1-p) whatever the structure. So the
default regime favours level-only predictors over item-difficulty and
group-effect ones. A comparison between an item- or group-aware predictor and
a level-only one must hold on kinds=("mix", "whole") and on public R1 before it
drives a choice.

Runs (Sampler). A run draws 5 to 12 pairs one at a time. With probability
`repeat` the next pair joins a pseudo-benchmark already in the run (the
feedback's 9 pairs held 7 benchmarks), otherwise it opens a new one. Subjects
are distinct within a run, by subject_id and by the eight visible fields.
Pseudo-benchmarks of one parent in a run share no more than `overlap` of their
items (so a stratum and the group it contains do not both appear). A draw stops
short of its size when every pseudo-benchmark left is open, clashes or has no
free subject; it does not fall back to a second pair on an open one. A draw that
ends below 5 pairs is redrawn (`tries`), so every run has 5 or more: over 1,500
draws 10.9% stop short, at 8.17 pairs a run against 8.5 drawn. Each chosen pair
keeps a uniform random subset of at most `max_items` items, and then
official.sample_run's own proportional cut fits the run to the 1,000-item cap.
official splits a pair 50/50 before that cut, so the evaluated count varies:
25% of default pairs evaluate fewer than 44 items (the feedback's minimum) and
15% more than 76; only 14% of 9-pair runs hold every pair in 44..76, where all
nine feedback pairs are, three of them at exactly 44. That suggests the
platform splits after the cut with a floor near 88 kept items;
split_after_cut=True, min_kept=88 emulates it here (evaluated 44 to 80,
n_pairs at most 11 to fit the cap) and official.py should take it as a
parameter.

A new pseudo-benchmark is drawn with weight base x tilt. The base gives each
parent log(1 + its eligible subjects), split over its pseudo-benchmarks in
proportion to the square root of theirs (times `kind_weights`), so a thin
pseudo-benchmark is not drawn as often as a populated one. swe_rebench is
excluded by default (`exclude`): its one subject would otherwise sit in most
runs.

Knobs (Regime):

  level_mean, level_sd  target distribution of pair accuracy logits. A draw is
           tilted by target density / base density (a kernel estimate), so its
           levels follow the target where the catalogue covers it. With
           tilt='benchmark' (default) the tilt acts on a pseudo-benchmark's
           level, the subject-weighted mean logit of its pairs, at sd
           sqrt(level_sd^2 - within^2); subjects are then drawn by recency
           alone. So low pair rates come from hard benchmarks, as a new hard
           benchmark would give them, not from weak subjects on moderate ones.
           tilt='pair' tilts the pair's own logit instead; level_mean=None
           switches the tilt off. Distinct pseudo-benchmarks and the clash rule
           pull realized levels toward the catalogue's middle: the default
           -1.6 / 1.5 gives pairs at -1.29 / 1.70 on their evaluated responses
           (SE 0.16 clustered by parent and subject; leaving one parent out
           moves the mean over -1.62 to -0.83). The feedback's -1.6 comes from
           the lower root of p(1-p) = B31, which reads every rate above 0.5 as
           one below it; read the same way, the replica's own Predictor B31
           gives -1.69 / 1.31 against the feedback's -1.64 / 1.40. One 9-pair
           run does not identify level_mean: at date_shift 1.25, level_mean
           -1.2, -1.6 and -2.0 are 0.63, 0.37 and 0.46 run sds from the
           feedback at B0/B1, 0.63, 0.62 and 0.83 with B31 added, and the
           budgets not used in tuning (B3, B7, B15) lean to -1.2 (0.32, 0.76,
           1.12). Repeat any downstream choice at -1.2 and -2.0.
  level_mix  a Gaussian-mixture target in place of level_mean / level_sd:
           (weight, mean, sd) triples, weights positive and summing to 1. Each
           component is tilted as one Gaussian target would be (with
           tilt='benchmark' at sd sqrt(max(sd^2 - within^2, min_level_sd^2)),
           with tilt='pair' at sd), the target density is their weighted sum,
           and level_mean and level_sd are ignored. Empty (the default) leaves
           the sampler exactly as without it; one component of weight 1 is the
           Gaussian tilt. experiments/regime_sensitivity.py's MIXTURE regime
           uses it for the two-mode reading of the formative feedback.
  min_release, recency  subjects released on or after min_release (undated
           ones excluded), weighted exp(recency * years since min_release);
           recency is 0 by default and untested against the feedback.
  date_shift  years added to the release and access dates predictors see
           (labels, items and subject_id untouched); date_cap caps the shifted
           dates. Without it the shipped Predictor's B0 is 0.2347, 8.2 single-
           run sds below the feedback, and its B0 prediction averages 0.498:
           the attribute prior puts public 2025-26 subjects at a 90th
           percentile of 0.53 to 0.56 on the parents other than matharena
           (0.67 there; maxima 0.54 to 0.69, 0.73), and a prediction of 0.5
           scores 0.25 at B0 whatever the pair's rate. The feedback's 0.359
           needs subjects placed near 0.75 on low-rate pairs; the default 1.25
           years gives 0.3513 with a mean B0 prediction of 0.682. This is a
           synthetic change to visible inputs, and it matches the feedback
           only for the legacy prior, which is linear and unclipped in days
           since 2023 (paiec.subjects): for it the shift is an additive offset
           on the prior's ability, which reproduces B0 row for row with real
           dates (and the later budgets within their SEs; the platform's
           random acquisition hashes the visible input, dates included, so a
           shift also redraws which items are acquired). A predictor that
           reads dates otherwise (clipped, nonlinear, ignoring dates) or reads
           other attributes (name, size, provider, reasoning_effort) gets a
           different optimism from it, so its B0/B1 results here are driven
           by this knob: a date-blind prior beats the legacy one by 0.033 ALC
           under the shift but by 0.002 under the equivalent offset, and the
           three-way order with the empirical mean flips at B0, B3 to B15 and
           ALC. Other sources of the feedback's optimism are not
           distinguished. The shifted dates run to 2027-07: 19% fall after
           2026-09-25 and 16% after 2026-12-31, outside the stated 2025-26
           pool; capping at 2026-12-31 moves B0 by -0.002 and ALC by -0.001.
  kinds, kind_weights  which pseudo-benchmark kinds are drawn (default: all but
           'mix') and a positive weight per kind on the base weight.
  max_per_parent  at most this many pseudo-benchmarks of one parent a run. By
           default 86% of runs hold two or more of one parent (siblings; 100%
           with repeat=0), 3.75 parents a run; with 1, runs have 5.8 pairs on
           average (repeats fill them) and the Predictor stays within 0.7 run
           sds of the feedback's B0/B1.
  split_after_cut, min_kept, tries  above.

The defaults (level_mean -1.6, level_sd 1.5, date_shift 1.25) are the grid
point of experiments/testlike_check.py closest to the feedback's B0 and B1,
and also closest once B31 is added. That agreement is in sample. ALC does not
tell this regime from public R1 (Predictor 0.2073 here, 0.2075 on R1, 0.2113 in
the feedback); the budget profile does (R1: B0 z +8.5, B7 and B31 near -1).
Everything is seeded: the same catalogue, regime and rng give the same run.

Selection caveats. Group segmentation and the benchmark-level tilt use every
subject's labels, the target's included (its weight in a level is 1/n of the
pairs); tilt='pair' selects on the target pair's own realized rate by design.
None of it alters a label. Stratum membership is cross-fitted, as above. The
catalogue is finite and the tilt concentrates on the pseudo-benchmarks near
the target, so pairs recur across runs (effective pseudo-benchmarks 26, the
most drawn subject in 34% of runs), and pseudo-benchmarks of one parent
overlap: cluster standard errors by (parent, subject) (cluster_key), and
report leave-one-parent-out ranges, since four parents carry every number.
Public records: 15% of default pairs have a subject with public pairs on
another parent by subject_id, 54% by normalized name, so a name-based subject
prior fires more often here than it may on unseen hidden subjects;
training_pairs(..., hide_subjects=True) withholds them (the legacy prior barely
moves: ALC 0.2047 against 0.2055). official.training_pairs and
official.run_benchmarks compare names, and no pseudo-benchmark name matches a
public pair, so on these runs they exclude nothing: fit priors through
training_pairs here (official.py should map names through parent_of or refuse
names that are not public benchmarks).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np

from paiec import data as D
from paiec import official as O
from paiec.evaluator import MIN_ITEMS, Pair, stable_hash

SEP = "::"
#: the informative item_features key of each public benchmark (step-2
#: variance components: 0.50, 0.43, 0.24 and 0.05 of the item variance)
PARENT_KEYS = {"matharena": "competition", "researchcodebench": "paper",
               "real_webagents": "website", "multi_swebench": "lang"}
RECENT = "2025-01-01"
#: pseudo-benchmark kinds; 'mix' is built but off by default (Regime.kinds)
KINDS = ("group", "mix", "stratum", "whole", "chunk")
#: The feedback this regime is matched to (first formative submission, the
#: shipped Predictor at commit b68492c): per-pair Brier at B0..B31.
FEEDBACK = [
    ("A", 76, [0.4224, 0.2851, 0.2698, 0.2067, 0.2085, 0.2057]),
    ("B", 50, [0.4970, 0.2861, 0.1394, 0.0764, 0.0650, 0.0513]),
    ("C", 56, [0.3583, 0.2419, 0.2429, 0.2437, 0.2751, 0.1923]),
    ("D", 58, [0.5663, 0.3234, 0.1379, 0.0518, 0.0184, 0.0063]),
    ("E", 60, [0.3032, 0.2477, 0.2479, 0.2431, 0.2437, 0.2472]),
    ("F", 44, [0.1728, 0.1670, 0.1675, 0.1665, 0.1719, 0.1678]),
    ("A", 53, [0.4771, 0.2939, 0.1738, 0.1335, 0.1285, 0.1286]),
    ("F", 44, [0.1806, 0.1747, 0.1760, 0.1827, 0.1745, 0.1749]),
    ("G", 44, [0.2519, 0.2653, 0.2834, 0.2360, 0.2386, 0.2366]),
]
FEEDBACK_ALC = 0.2113


def parent_of(name: str) -> str:
    """The public benchmark a pseudo-benchmark was cut from (a public name is its
    own parent)."""
    return name.split(SEP, 1)[0]


def kind_of(name: str) -> str:
    """The kind of a pseudo-benchmark from its name; 'public' for a public name."""
    if SEP not in name:
        return "public"
    tag = name.split(SEP, 1)[1]
    return {"g": "group", "m": "mix", "q": "stratum", "c": "chunk"}.get(tag[:1], "whole")


def cluster_key(benchmark: str, subject_id: str) -> tuple:
    """The unit to resample for standard errors on test-like runs: (parent,
    subject). Pseudo-benchmarks of one parent overlap (a stratum shares items
    with groups, 'all' holds them all), so the same subject's responses sit in
    several pseudo-benchmarks and are not independent between them."""
    return parent_of(benchmark), subject_id


def logit_rate(k, n) -> float:
    """Continuity-corrected logit of k successes in n: finite at 0 and n."""
    return math.log((k + 0.5) / (n - k + 0.5))


def rate_from_brier(brier, eps=1e-4) -> float:
    """The lower root p of p(1-p) = brier: the reading of a pair's rate from a
    late-budget Brier used on the feedback. Every rate above 0.5 reads as one
    below it, and a Brier of 0.25 or more reads as 0.5."""
    return float(min(max((1 - math.sqrt(max(0.0, 1 - 4 * brier))) / 2, eps), 0.5))


def feature(item, key) -> str:
    """Value of `key` in the public 'k=v;k=v' item_features format, '' if absent."""
    for kv in str(item.get("item_features") or "").split(";"):
        k, _, v = kv.partition("=")
        if k.strip() == key:
            return v.strip()
    return ""


def _iso(text):
    """A date from 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD' (the public forms), else None."""
    parts = str(text or "").strip()[:10].split("-")
    try:
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1,
                    int(parts[2]) if len(parts) > 2 else 1)
    except (ValueError, IndexError):
        return None


def years_since(release, start) -> float | None:
    """Years from `start` to a release date, None when undated or earlier."""
    r, s = _iso(release), _iso(start)
    if r is None or s is None or r < s:
        return None
    return (r - s).days / 365.25


def shift_date(text, years, cap=None) -> str:
    """An ISO date moved by `years` (as YYYY-MM-DD), no later than `cap` when one
    is given; '' and unreadable dates stay."""
    d = _iso(text)
    if d is None or (not years and cap is None):
        return text
    d = d + timedelta(days=round(years * 365.25))
    c = _iso(cap) if cap is not None else None
    if c is not None and d > c:
        d = c
    return d.isoformat()


def shifted(pair, years, cap=None) -> Pair:
    """The pair with its subject's release and access dates moved by `years` (and
    capped at `cap`): the same responses and subject_id, a subject that looks
    newer to anything reading its dates."""
    if not years and cap is None:
        return pair
    subject = dict(pair.subject)
    for f in ("release_date", "access_date"):
        if f in subject:
            subject[f] = shift_date(subject[f], years, cap)
    return Pair(subject, pair.subject_id, pair.benchmark_id, pair.responses)


# --- item difficulty ---------------------------------------------------------------

def rasch(pairs, sd_theta=2.0, sd_b=3.0, iters=300, tol=1e-3):
    """Item difficulties of a joint Rasch model, logit p = theta_s - b_i, with
    Gaussian ridges, fitted by alternating Newton on every recorded response.
    Returns {item key: b}. Items everyone fails sit at the ridge's bound, tied."""
    si, ii, y, keys = [], [], [], {}
    for j, p in enumerate(pairs):
        for r in p.responses:
            si.append(j)
            ii.append(keys.setdefault(r.item_key, len(keys)))
            y.append(r.label)
    if not keys:
        return {}
    code = np.array(si, np.int64) * len(keys) + np.array(ii, np.int64)
    cells, inv = np.unique(code, return_inverse=True)
    K = np.bincount(inv, np.array(y, float))
    N = np.bincount(inv).astype(float)
    si, ii = cells // len(keys), cells % len(keys)
    th, b = np.zeros(len(pairs)), np.zeros(len(keys))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[si] - b[ii])))
        th += ((np.bincount(si, K - N * p, len(th)) - th / sd_theta ** 2)
               / (np.bincount(si, N * p * (1 - p), len(th)) + 1 / sd_theta ** 2))
        p = 1 / (1 + np.exp(-(th[si] - b[ii])))
        step = ((-np.bincount(ii, K - N * p, len(b)) - b / sd_b ** 2)
                / (np.bincount(ii, N * p * (1 - p), len(b)) + 1 / sd_b ** 2))
        b += step
        if np.abs(step).max() < tol:
            break
    return dict(zip(keys, b.tolist()))


def fold_of(subject_id, folds, salt=0) -> int:
    return stable_hash(salt, "fold", subject_id) % folds


# --- the catalogue ---------------------------------------------------------------

@dataclass
class Pseudo:
    name: str                   # parent + SEP + tag; the pairs' benchmark_id
    parent: str
    kind: str                   # one of KINDS
    tag: str
    groups: tuple = ()          # feature values merged, for kinds 'group' and 'mix'
    items: frozenset = frozenset()          # every item any subject has in it
    pairs: list = field(default_factory=list)       # eligible Pairs


@dataclass
class Catalogue:
    pseudos: list
    info: dict = field(default_factory=dict)
    #: parent -> {item key: Rasch difficulty}, fitted on every subject of the
    #: parent (in sample; for describing item structure, never for drawing)
    difficulty: dict = field(default_factory=dict)
    #: parent -> {item key: value of its PARENT_KEYS feature}
    features: dict = field(default_factory=dict)


def _restrict(pair, keys, name):
    """The pair's responses on `keys`, as a pair of pseudo-benchmark `name`."""
    return Pair(pair.subject, pair.subject_id, name,
                [r for r in pair.responses if r.item_key in keys])


def _pseudo(parent, kind, tag, members, groups=()):
    """members: [(pair, item keys)]; keeps pairs with >= MIN_ITEMS items."""
    name = f"{parent}{SEP}{tag}"
    pairs = [_restrict(p, ks, name) for p, ks in members if len(ks) >= MIN_ITEMS]
    items = frozenset().union(*[ks for _, ks in members]) if members else frozenset()
    return Pseudo(name, parent, kind, tag, tuple(groups), items, pairs)


def _segments(ps, group, order, target_items, min_subjects):
    """Contiguous segments of the groups taken in `order`; a segment closes once
    `min_subjects` subjects hold `target_items` of its items, and a tail that
    never does joins the last segment."""
    members = defaultdict(set)
    for k, g in group.items():
        members[g].add(k)
    own = [set(p.item_keys) for p in ps]

    def closes(seg):
        ks = set().union(*[members[g] for g in seg])
        return sum(len(o & ks) >= target_items for o in own) >= min_subjects

    segs, cur = [], []
    for g in order:
        cur.append(g)
        if closes(cur):
            segs.append(cur)
            cur = []
    if cur:
        if segs:
            segs[-1] = segs[-1] + cur
        else:
            segs.append(cur)
    return [(seg, set().union(*[members[g] for g in seg])) for seg in segs]


def _strata(ps, Ks, folds, salt):
    """{K: {band: [(pair, keys)]}}, band 0 the hardest, cross-fitted over subject
    folds: a subject's items are banded by quantiles of a fit on the other folds."""
    fold = {p.subject_id: fold_of(p.subject_id, folds, salt) for p in ps}
    out = {K: defaultdict(list) for K in Ks}
    for f in range(folds):
        mine = [p for p in ps if fold[p.subject_id] == f]
        b = rasch([p for p in ps if fold[p.subject_id] != f]) if mine else {}
        if not b:
            continue
        vals = np.array(sorted(b.values()))
        for K in Ks:
            cuts = np.quantile(vals, np.linspace(0, 1, K + 1)[1:-1])
            for p in mine:
                keys = [k for k in p.item_keys if k in b]
                bands = K - 1 - np.searchsorted(cuts, [b[k] for k in keys], side="right")
                for q in range(K):
                    out[K][q].append((p, {k for k, x in zip(keys, bands) if x == q}))
    return out


def build_catalogue(pairs, *, target_items=90, min_subjects=3, strata=(2, 3), folds=5,
                    chunk_items=150, wholes=True, mixed=True, keys=None, seed=0):
    """Pseudo-benchmarks from eligible public pairs (see the module docstring).
    With `mixed`, each keyed parent's groups are also merged in a seeded random
    order (kind 'mix'), which keeps the key's share of item variance; Regime
    leaves them out unless its `kinds` names them. Pseudo-benchmarks with no
    eligible pair are dropped."""
    keys = PARENT_KEYS if keys is None else keys
    pairs = O.eligible(pairs)
    by = defaultdict(list)
    for p in pairs:
        by[p.benchmark_id].append(p)
    out, info, difficulty, features = [], {}, {}, {}
    for parent in sorted(by):
        ps = by[parent]
        made = []
        if len(ps) < min_subjects:
            for p in ps:
                ks = list(p.item_keys)
                ks = sorted(ks, key=lambda k: stable_hash(seed, "chunk", parent, k))
                for j in range(len(ks) // chunk_items):
                    made.append(_pseudo(parent, "chunk", f"c{j}",
                                        [(p, set(ks[j * chunk_items:(j + 1) * chunk_items]))]))
        else:
            b = rasch(ps)
            difficulty[parent] = b
            if parent in keys:
                group = {}
                for p in ps:
                    for r in p.responses:
                        group.setdefault(r.item_key, feature(r.item, keys[parent]))
                features[parent] = group
                members = defaultdict(list)
                for k, g in group.items():
                    members[g].append(b[k])
                hard = sorted(members, key=lambda g: (-float(np.mean(members[g])), g))
                for j, (seg, ks) in enumerate(_segments(ps, group, hard, target_items,
                                                        min_subjects)):
                    made.append(_pseudo(parent, "group", f"g{j}",
                                        [(p, ks & set(p.item_keys)) for p in ps], groups=seg))
                if mixed:
                    shuffled = sorted(members, key=lambda g: stable_hash(seed, "mix", parent, g))
                    for j, (seg, ks) in enumerate(_segments(ps, group, shuffled, target_items,
                                                            min_subjects)):
                        made.append(_pseudo(parent, "mix", f"m{j}",
                                            [(p, ks & set(p.item_keys)) for p in ps], groups=seg))
            for K, bands in _strata(ps, strata, folds, seed).items():
                for q, members_ in sorted(bands.items()):
                    made.append(_pseudo(parent, "stratum", f"q{q + 1}of{K}", members_))
            if wholes:
                made.append(_pseudo(parent, "whole", "all", [(p, set(p.item_keys)) for p in ps]))
        made = [m for m in made if m.pairs]
        info[parent] = {"pairs": len(ps), "pseudos": len(made)}
        out += made
    names = [b.name for b in out]
    anon = {D.anon_id("benchmark", n) for n in names}
    if len(set(names)) != len(names) or len(anon) != len(names):
        raise ValueError("pseudo-benchmark names or their anonymous ids collide")
    return Catalogue(out, info, difficulty, features)


# --- the regime ------------------------------------------------------------------

TILTS = ("benchmark", "pair")


@dataclass(frozen=True)
class Regime:
    """How test-like runs are drawn (knobs in the module docstring). The level
    and date_shift defaults are the tuning grid point of
    experiments/testlike_check.py closest to the first formative feedback."""
    level_mean: float | None = -1.6
    level_sd: float = 1.5
    tilt: str = "benchmark"
    min_release: str | None = RECENT
    recency: float = 0.0
    exclude: tuple = ("swe_rebench",)
    date_shift: float = 1.25
    date_cap: str | None = None
    n_pairs: tuple = (5, 12)
    repeat: float = 0.25
    max_items: int = 160
    cap: int = O.CAP
    overlap: float = 0.1
    max_per_parent: int | None = None
    kinds: tuple = ("group", "stratum", "whole", "chunk")
    kind_weights: tuple = ()
    split_after_cut: bool = False
    min_kept: int = MIN_ITEMS
    tries: int = 20
    bandwidth: float = 0.7
    min_level_sd: float = 0.3
    max_tilt: float = 20.0
    level_mix: tuple = ()

    def with_(self, **kw):
        return replace(self, **kw)


def _npdf(x, m, s):
    return np.exp(-0.5 * ((np.asarray(x, float) - m) / s) ** 2) / s


def _mixture(level_mix):
    """Regime.level_mix checked: a tuple of (weight, mean, sd) float triples,
    weights positive and summing to 1, sds positive; () when empty."""
    out = []
    for c in level_mix:
        if len(c) != 3:
            raise ValueError(f"level_mix takes (weight, mean, sd) triples, got {c!r}")
        w, m, s = (float(x) for x in c)
        if not (w > 0) or not (s > 0) or not math.isfinite(m) or not math.isfinite(s):
            raise ValueError(f"level_mix needs a positive weight and sd and a finite mean, got {c!r}")
        out.append((w, m, s))
    if out and abs(sum(w for w, _, _ in out) - 1.0) > 1e-9:
        raise ValueError(f"level_mix weights must sum to 1, got {sum(w for w, _, _ in out)!r}")
    return tuple(out)


class Sampler:
    """Draws test-like runs from a catalogue under a regime; `stats` describes the
    candidate pool and the weights it resolved to."""

    def __init__(self, catalogue, regime=Regime()):
        r = regime
        if r.tilt not in TILTS:
            raise ValueError(f"tilt must be one of {TILTS}, got {r.tilt!r}")
        kw = dict(r.kind_weights)
        if not set(r.kinds) <= set(KINDS) or not set(kw) <= set(KINDS):
            raise ValueError(f"kinds must be among {KINDS}, got {r.kinds!r} / {tuple(kw)!r}")
        if any(not (w > 0) for w in kw.values()):
            raise ValueError("kind weights must be positive; leave a kind out of `kinds` to drop it")
        mix = _mixture(r.level_mix)
        if r.n_pairs[1] * r.min_kept > r.cap:
            raise ValueError(f"{r.n_pairs[1]} pairs of >= {r.min_kept} items cannot fit a cap "
                             f"of {r.cap}")
        self.regime = regime
        self.cat = catalogue
        cands = []                    # (pseudo index, pair, logit, subject weight)
        for j, b in enumerate(catalogue.pseudos):
            if b.parent in r.exclude or b.kind not in r.kinds:
                continue
            for p in b.pairs:
                if len(p.item_keys) < r.min_kept:
                    continue
                if r.min_release:
                    y = years_since(p.subject.get("release_date"), r.min_release)
                    if y is None:
                        continue
                else:
                    y = years_since(p.subject.get("release_date"), RECENT) or 0.0
                k = sum(x.label for x in p.responses)
                cands.append((j, p, logit_rate(k, len(p.responses)), math.exp(r.recency * y)))
        if not cands:
            raise ValueError("no pair satisfies the regime")
        self.cands = cands
        self.canon = [_canon_subject(c[1]) for c in cands]
        self.of = defaultdict(list)
        for c, (j, *_rest) in enumerate(cands):
            self.of[j].append(c)
        live = sorted(self.of)
        J = len(catalogue.pseudos)
        ws = np.array([c[3] for c in cands])
        ell = np.array([c[2] for c in cands])
        jj = np.array([c[0] for c in cands])
        wsum = np.bincount(jj, ws, J)
        level = np.full(J, np.nan)
        level[live] = np.bincount(jj, ws * ell, J)[live] / wsum[live]
        subjects, root = defaultdict(set), defaultdict(float)
        for j, p, *_ in cands:
            subjects[catalogue.pseudos[j].parent].add(p.subject_id)
        wk = {j: kw.get(catalogue.pseudos[j].kind, 1.0) for j in live}
        for j in live:
            root[catalogue.pseudos[j].parent] += wk[j] * math.sqrt(len(self.of[j]))
        base = np.zeros(J)
        for j in live:
            par = catalogue.pseudos[j].parent
            base[j] = math.log1p(len(subjects[par])) * (wk[j] * math.sqrt(len(self.of[j]))) \
                / root[par]
        share = ws / wsum[jj]               # subject share within its pseudo-benchmark
        within = float(np.sqrt(np.sum(base[jj] * share * (ell - level[jj]) ** 2)
                               / np.sum(base[jj] * share)))
        if mix:
            # a Gaussian-mixture target: each component tilted as the one-
            # Gaussian branches below tilt theirs, the densities summed by weight
            if r.tilt == "benchmark":
                sd_b = tuple(math.sqrt(max(s ** 2 - within ** 2, r.min_level_sd ** 2))
                             for _, _, s in mix)
                L = level[live]
                f0 = (base[live][None, :] * _npdf(L[:, None], L[None, :], r.bandwidth)).sum(1) \
                    / base[live].sum()
                dens = sum(w * _npdf(L, m, s) for (w, m, _), s in zip(mix, sd_b))
                t = np.zeros(J)
                t[live] = dens / f0
                t[live] = np.minimum(t[live], r.max_tilt * np.sum(base[live] * t[live])
                                     / base[live].sum())
                omega = base[jj] * t[jj] * share
            else:
                sd_b = tuple(s for _, _, s in mix)
                pi = base[jj] * share
                f0 = (pi[None, :] * _npdf(ell[:, None], ell[None, :], r.bandwidth)).sum(1) / pi.sum()
                t = sum(w * _npdf(ell, m, s) for w, m, s in mix) / f0
                t = np.minimum(t, r.max_tilt * np.sum(pi * t) / pi.sum())
                omega = pi * t
        elif r.level_mean is None:
            omega = base[jj] * share
            sd_b = None
        elif r.tilt == "benchmark":
            sd_b = math.sqrt(max(r.level_sd ** 2 - within ** 2, r.min_level_sd ** 2))
            L = level[live]
            f0 = (base[live][None, :] * _npdf(L[:, None], L[None, :], r.bandwidth)).sum(1) \
                / base[live].sum()
            t = np.zeros(J)
            t[live] = _npdf(L, r.level_mean, sd_b) / f0
            t[live] = np.minimum(t[live], r.max_tilt * np.sum(base[live] * t[live]) / base[live].sum())
            omega = base[jj] * t[jj] * share
        else:
            sd_b = r.level_sd
            pi = base[jj] * share
            f0 = (pi[None, :] * _npdf(ell[:, None], ell[None, :], r.bandwidth)).sum(1) / pi.sum()
            t = _npdf(ell, r.level_mean, r.level_sd) / f0
            t = np.minimum(t, r.max_tilt * np.sum(pi * t) / pi.sum())
            omega = pi * t
        self.omega = omega / omega.sum()
        self.level = level
        # items shared by pseudo-benchmarks of one parent, as a share of the smaller
        self.clash = {}
        by = defaultdict(list)
        for j in live:
            by[catalogue.pseudos[j].parent].append(j)
        for js in by.values():
            for a in js:
                for c in js:
                    if a < c:
                        A, C = catalogue.pseudos[a].items, catalogue.pseudos[c].items
                        if len(A & C) > r.overlap * max(1, min(len(A), len(C))):
                            self.clash.setdefault(a, set()).add(c)
                            self.clash.setdefault(c, set()).add(a)
        m = float(np.sum(self.omega * ell))
        use = np.bincount(jj, self.omega, J)[live]
        use = use[use > 0]
        self.stats = {
            "candidates": len(cands), "pseudos": len(live),
            "subjects": len({c[1].subject_id for c in cands}),
            "within_sd": within, "benchmark_sd": sd_b,
            "one_draw_logit_mean": m,
            "one_draw_logit_sd": float(np.sqrt(np.sum(self.omega * (ell - m) ** 2))),
            "effective_pseudos": float(np.exp(-np.sum(use * np.log(use)))),
        }

    def _pick(self, rng, idx):
        w = self.omega[idx]
        return idx[int(rng.choice(len(idx), p=w / w.sum()))]

    def draw(self, rng):
        """The chosen pairs, before any cut: [(pseudo index, Pair)]. A draw that
        ends below n_pairs[0] (every pseudo-benchmark left is open, clashes or
        has no free subject) is redrawn, up to `tries` times, keeping the
        longest; a draw that ends short of its own size but at n_pairs[0] or
        more is kept."""
        r = self.regime
        n = int(rng.integers(r.n_pairs[0], r.n_pairs[1] + 1))
        best = []
        for _ in range(max(1, r.tries)):
            chosen = self._draw(rng, n)
            if len(chosen) >= r.n_pairs[0]:
                return chosen
            if len(chosen) > len(best):
                best = chosen
        return best

    def _draw(self, rng, n):
        r = self.regime
        chosen, sids, dicts, blocked = [], set(), set(), set()
        per_parent = Counter()
        for _ in range(n):
            def free(c):
                return self.cands[c][1].subject_id not in sids and self.canon[c] not in dicts
            open_js = {j for j, _ in chosen}
            idx = None
            if open_js and rng.random() < r.repeat:
                pool = [c for j in sorted(open_js) for c in self.of[j] if free(c)]
                if pool:
                    idx = np.array(pool)
            if idx is None:
                pool = [c for j in sorted(self.of) if j not in open_js and j not in blocked
                        and (r.max_per_parent is None
                             or per_parent[self.cat.pseudos[j].parent] < r.max_per_parent)
                        for c in self.of[j] if free(c)]
                if not pool:
                    break
                idx = np.array(pool)
            c = self._pick(rng, idx)
            j, p = self.cands[c][0], self.cands[c][1]
            if j not in open_js:
                per_parent[self.cat.pseudos[j].parent] += 1
                blocked |= self.clash.get(j, set())
            chosen.append((j, p))
            sids.add(p.subject_id)
            dicts.add(self.canon[c])
        return chosen

    def run(self, rng):
        """One test-like run, [(Pair, frozenset of item keys)], ready for
        official.run_official. Pairs carry their pseudo-benchmark as benchmark_id.
        With split_after_cut each pair holds only its kept items, so the
        official 50/50 split falls on them and a pair evaluates half of what it
        keeps."""
        r = self.regime
        cut = []
        for _, p in self.draw(rng):
            p = shifted(p, r.date_shift, r.date_cap)
            keys = p.item_keys
            if len(keys) > r.max_items:
                keep = set(keys[i] for i in rng.choice(len(keys), r.max_items, replace=False))
                p = Pair(p.subject, p.subject_id, p.benchmark_id,
                         [x for x in p.responses if x.item_key in keep])
            cut.append(p)
        run = O.sample_run(cut, rng, cap=r.cap, n_pairs=len(cut), weighting="pair",
                           min_items=r.min_kept)
        if r.split_after_cut:
            run = [(Pair(p.subject, p.subject_id, p.benchmark_id,
                         [x for x in p.responses if x.item_key in keys]), keys) for p, keys in run]
        return run


def _canon_subject(pair):
    return tuple(sorted(D.official_subject(pair.subject).items()))


# --- helpers for experiments -------------------------------------------------------

def anon_parents(run) -> dict:
    """Anonymous benchmark_id -> parent benchmark, for the caller: a model factory
    can dispatch on the id to a prior fitted without the parent."""
    return {D.anon_id("benchmark", _entry(e).benchmark_id): parent_of(_entry(e).benchmark_id)
            for e in run}


def run_parents(run) -> list:
    return sorted({parent_of(_entry(e).benchmark_id) for e in run})


def training_pairs(pairs, run, *, parents=None, hide_subjects=False) -> list:
    """Public pairs a prior used on this run may be fitted on. official's
    training_pairs and run_benchmarks compare benchmark names, and a
    pseudo-benchmark's name matches no public pair, so on a test-like run they
    exclude nothing. This leaves out every pair of a parent of the run (or of
    `parents` when given), and with hide_subjects every pair of a run subject
    too, matched by subject_id or normalized_name, so a prior that reads a
    subject's public record by name sees none of it."""
    held = set(run_parents(run)) if parents is None else set(parents)
    out = [p for p in pairs if parent_of(p.benchmark_id) not in held]
    if hide_subjects:
        ids = {_entry(e).subject_id for e in run}
        names = {_name(_entry(e)) for e in run} - {""}
        out = [p for p in out if p.subject_id not in ids and _name(p) not in names]
    return out


def _name(pair) -> str:
    return str(pair.subject.get("normalized_name") or "").strip().lower()


def _entry(e):
    return e[0] if isinstance(e, tuple) else e


def describe(run, seed=0, scope="pair") -> list:
    """Per pair of a run: pseudo-benchmark, parent, subject, release date, kept and
    evaluated items, and the evaluation / acquisition rates as official._slots
    selects them (first recorded response revealed, every response scored)."""
    out = []
    for p, items in run:
        acq, ev = (set(k) & items for k in O.split(p, seed, scope))
        by = p.by_item()
        ys = [x.label for k in ev for x in by[k]]
        ya = [by[k][0].label for k in acq]
        out.append({"pseudo": p.benchmark_id, "parent": parent_of(p.benchmark_id),
                    "subject_id": p.subject_id,
                    "release_date": str(p.subject.get("release_date") or ""),
                    "items": len(items), "eval_items": len(ev),
                    "p": float(np.mean(ys)), "pa": float(np.mean(ya)),
                    "logit": logit_rate(sum(ys), len(ys))})
    return out


def item_oracle(run, catalogue, seed=0, scope="pair", prior_sd=5.0) -> list:
    """How much item structure a pair's evaluated responses carry, per pair:
    the Brier of the pair-rate oracle (every response predicted at the pair's
    own rate) and of an item-level oracle, sigmoid(theta - b_i) with b_i the
    parent's in-sample Rasch difficulty (Catalogue.difficulty) and theta fitted
    on the pair's own evaluated responses (a N(0, prior_sd^2) ridge), and the
    sd of b_i over them. None for a pair whose parent has no difficulty
    (a single-subject benchmark). Both oracles see the answers; only their gap
    is read, as the most an item-difficulty model could gain on that pair."""
    out = []
    for p, items in run:
        par = parent_of(p.benchmark_id)
        b = catalogue.difficulty.get(par)
        if not b:
            out.append(None)
            continue
        _, ev = (set(k) & items for k in O.split(p, seed, scope))
        by = p.by_item()
        y = np.array([x.label for k in sorted(ev) for x in by[k]], float)
        z = np.array([b[k] for k in sorted(ev) for _ in by[k]])
        rate = float(y.mean())
        th = 0.0
        for _ in range(100):
            q = 1 / (1 + np.exp(-(th - z)))
            step = (np.sum(y - q) - th / prior_sd ** 2) / (np.sum(q * (1 - q)) + 1 / prior_sd ** 2)
            th += step
            if abs(step) < 1e-8:
                break
        q = 1 / (1 + np.exp(-(th - z)))
        out.append({"pseudo": p.benchmark_id, "parent": par, "kind": kind_of(p.benchmark_id),
                    "level": float(np.mean((rate - y) ** 2)), "item": float(np.mean((q - y) ** 2)),
                    "b_sd": float(np.std([b[k] for k in ev]))})
    return out


def key_share(items, difficulty, groups) -> dict:
    """Variance of item difficulty over `items` and the share of it that lies
    between the values of their feature (`groups`: item -> value)."""
    ks = [k for k in items if k in difficulty and k in groups]
    if len(ks) < 2:
        return {"b_var": None, "key_share": None, "key_values": len({groups.get(k) for k in ks})}
    x = np.array([difficulty[k] for k in ks])
    g = np.array([groups[k] for k in ks])
    tot = float(x.var())
    between = sum(np.mean(g == v) * (x[g == v].mean() - x.mean()) ** 2 for v in set(g.tolist()))
    return {"b_var": tot, "key_share": float(between / tot) if tot > 0 else 0.0,
            "key_values": len(set(g.tolist()))}
