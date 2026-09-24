# The protocol

Two sources, read side by side: the organisers' streaming client
(`third_party/paiec_baseline/tools/streaming_ingestion.py`, from the public
baseline repository, read-only and not committed) and the competition page.
`paiec/official.py` replicates what they fix and turns what they leave open into
parameters; `tests/test_official.py` pins every point below that is marked
verified. `paiec/evaluator.py` is the earlier, pair-major reconstruction; see the
last section for why its numbers are not comparable.

    import numpy as np
    from paiec import baselines as B, official as O
    from paiec.data import load_pairs

    run = O.sample_run(O.eligible(load_pairs()), np.random.default_rng(0))
    res = O.run_official(run, lambda: B.empirical_mean)   # the factory returns a fresh predict
    res["brier"]["ALC"], res["rows"], res["timing"]

## Interfaces (verified)

    predict(input, labeled) -> float                  # model.py, required
    acquisition_function(input, prediction=None,      # labeling.py, optional
                         labeled=None, context=None) -> bool

`input = [subject, item]`. The subject has exactly eight keys: normalized_name,
provider, release_date, access_date, harness, harness_version,
reasoning_effort, subject_features_extra. The item has exactly four:
item_content, item_features, interactors and benchmark_id, a stable anonymous
id shared by target items and acquired examples of the same benchmark. Every
value is a string; missing values are `""`. `labeled` is a list of
`[[subject, item], label]` with label 0 or 1.

The runtime applies `float()` to what predict returns and rejects conversion
failures, NaN, infinities and values outside [0, 1]. Failures surface as exit
codes: 40 predict raised, 41 a call timed out, 42 invalid probability
(`official.SubmissionError.code`).

The signature of `acquisition_function` alone decides how it is called, and a
TypeError raised inside it is never retried with another shape:

* accepts prediction, labeled and context by name, via `**kwargs`, or as four
  positionals: streaming mode. It must return a native `bool`; `np.bool_` is an
  error. A prediction is computed for it only if it takes `prediction`.
* accepts a single argument: `legacy_rank`. It scores every acquisition
  candidate once, without labels, and the top-scored candidates of each pair are
  revealed at each budget.
* no labeling.py: the platform's random policy. With
  `key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))`
  and `u = int.from_bytes(sha256(key).digest()[:8], "big") / 2**64`, the
  candidate is taken iff `u < min(1, labels_remaining / items_remaining)`. That
  is a uniformly random subset, and every remaining candidate is taken once the
  pool is no longer than the allowance, so 31 labels fill whenever the pool has
  31 items. The replica's `default_policy` is checked decision for decision
  against the organisers' `run_streaming`.

`context` holds subject_id and benchmark_id (anonymous), labels_acquired,
labels_remaining (out of 31, not the distance to the next budget), max_labels
(31) and items_remaining (counting the current candidate). All counts are for
the current pair.

## Flow (verified)

Only subject-benchmark pairs with at least 80 distinct items take part, and only
binary responses: non-binary responses are dropped, and benchmarks whose
responses are fractions (mmdocrag in the public data) are dropped whole
(`official.eligible`). A pair's items split 50/50 into acquisition and evaluation
pools that are fixed across submissions and budgets; all responses of an item
stay on one side. Acquisition candidates arrive in a fixed order and a skipped
candidate is gone for good.

Scoring is budget-major, over six checkpoints 0, 1, 3, 7, 15, 31. Every pair of
the run is evaluated at budget 0 with `labeled` empty. Acquisition then
continues, each pair's stream resuming where it stopped, until every pair holds
up to one label; every pair is evaluated at budget 1; acquisition continues to
three; and so on. No pair gets labels beyond budget B until all pairs have been
evaluated at B.

At a checkpoint one shared `labeled` goes to every target: the first B labels
of every pair in the run, whatever its subject or benchmark. A target on
benchmark X therefore sees other subjects' labels on X, and its own subject's
labels on other benchmarks, from budget 1 on.

Evaluation workers (up to 16 at once) are recreated from the submission at every
checkpoint, so nothing a predictor holds in memory survives from one budget to
the next; within a checkpoint a worker may serve many calls. Identical
`[subject, item]` inputs are predicted once per checkpoint and the prediction is
reused for all their responses. This includes two subjects whose eight visible
fields coincide. Acquisition-time predictions run sequentially in one
long-lived main process, with deep-copied arguments.

The replica emulates this with `model_factory`: one instance, created on first
use, serves acquisition-time predictions for the whole run, and a fresh
instance per worker shard serves each checkpoint, targets dealt round-robin.
Arguments reach predict as deep copies with no shared objects, as JSON transport
would deliver them (`deepcopy=False` skips the copies for speed).

## Scoring (verified)

Brier is the mean squared error over all evaluation responses of a pair, every
trial counted, then averaged equally over pairs whatever their size.

    ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31

The k-th acquired label counts toward every budget it survives into: 0.9 for the
first, 0.7 for the second and third, 0.5 for the fourth to seventh, 0.3 for the
eighth to fifteenth, 0.1 for the sixteenth to thirty-first.

ECE, a diagnostic only, is the response-weighted mean gap between average
prediction and success rate over ten equal-width bins, per pair and budget; the
summary uses the ALC weights. Formative feedback reports, per pair, anonymous
subject and benchmark ids, the number of distinct evaluated subject-item pairs
(fixed across budgets), Brier and ECE; `run_official(...)["rows"]` has the same
shape.

## Runs

Each formative run draws a new sample of the hidden test data, capped at 1,000
unique subject-item pairs across both pools. With at least 80 items per pair
that allows at most 12 pairs. The illustrative feedback on the site shows 5
pairs over 4 subjects and 4 benchmarks with 46 to 181 evaluated items each,
about 960 items in all, so the cap binds and large pairs are cut. The summative
evaluation uses one common test subset for all teams; its size is not stated.

`official.sample_run` draws 5 to 12 pairs (or `n_pairs`), a benchmark at a time
weighted `1 + concentration * (pairs already drawn from it)` and then a pair
weighted `1 + subject_affinity * (subject already drawn)`, and cuts each pair to
a uniform random subset of `min(n, max(80, t*w))` items with the largest t
under the cap (`alloc`: w = size or 1). `official.dense_run` takes every eligible
pair of one benchmark whole, the most cross-subject evidence a run could carry.
`run_benchmarks` and `training_pairs` give the caller the run's real benchmark
names, to fit a prior that leaves them out; predict only sees anonymous ids.

Runtime limits: 8 hours per run including setup; a per-call timeout of
undisclosed length; no extra package installs by default. numpy is certainly
available, torch, sentence-transformers, scipy and scikit-learn very probably.
`run_official(...)["timing"]` records wall-clock per call (setup, acquisition,
evaluation, hook) and `call_timeout` turns an overrun into code 41.

## Still unknown, and how the replica handles it

| unknown | replica |
|---|---|
| whether the 50/50 split is drawn per pair or per benchmark item | `split_scope='pair'` (default) ranks items by a digest salted with the subject; `'benchmark'` ignores the subject, so subjects sharing a benchmark's items share pools and no subject's acquired label is another's evaluation item. When subjects attempted different item subsets, items near the median can still fall on different sides. |
| how a formative run picks pairs and items | `sample_run` knobs above; items cut uniformly, not stratified by pool, after the persistent split |
| hidden test benchmarks | disjoint from the public ones by design; leave-benchmark-out via `training_pairs` is the honest proxy |
| which recorded response is revealed for an acquired item with repeats | the first recorded one |
| order of pairs in `labeled` | run order; during acquisition, grouped by pair rather than chronological |
| legacy_rank direction and ties | higher score first, ties in stream order |
| per-call timeout | unset; watch `timing` and keep calls well under a second on CPU |
| whether workers copy `labeled` per call | copied per call |
| anonymous ids | ours are digests of our names, so default-policy decisions follow the platform's rule but not its actual draws |
| run-level ECE | mean of per-pair ECE |

`seed` salts the split and the stream order; 0 is the persistent split, other
seeds stand in for other hidden splits when averaging out split noise.

## The legacy replica

`paiec/evaluator.py` was reconstructed before the client was available. It is
pair-major: one pair is acquired and scored at all six budgets before the next,
with a single predictor instance throughout, so a stateful predictor sees whole
31-label trajectories of earlier pairs while still being scored at budget 0.
Its `labeled` holds only the target subject's own labels (from earlier pairs
when `cross_pair=True`), never other subjects'. It scores all 221 public pairs
in one session instead of 5 to 12, passes private keys (`_sid`, `_key`) and raw
benchmark names to predict, does not deduplicate inputs, and draws its default
policy and split from numpy generators rather than the sha256 rule.

Every number in `README.md` and `docs/findings.md` so far was measured on it.
They rank predictors under a different information set, cross-subject pooling
most of all, and are not comparable with `paiec.official`; they stay
reproducible because the module is unchanged.

The legacy replica's first version keyed its split on Python's `hash()`, which
is salted per process, so the split silently changed between runs. Both
replicas hash by digest (`evaluator.stable_hash`) for that reason. Residual split
noise in the legacy setting is sd 0.0012 to 0.0017 of ALC per measurement; a
single formative run of 5 to 12 pairs is far noisier than that.

## Data

measurement-db holds 571,921 responses over 287 subjects and six benchmarks.
Five are binary; mmdocrag is fraction-valued, with only 4.8% of its responses
equal to 0 or 1, and matharena has a 0.1% non-binary tail. After dropping the
fractional ones, 221 pairs and 225,843 responses remain: 82 multi_swebench, 81
matharena, 31 researchcodebench, 26 real_webagents and a single swe_rebench pair
with 6,306 items.

Repeats are wildly uneven: 99.8% of matharena items have them, 99.6% of
swe_rebench, 0.4% of multi_swebench. Only 22 of 287 subjects appear on more than
one benchmark. `interactors` is empty everywhere. Twenty-four of the 221 pairs
share all eight visible subject fields with a pair on another benchmark; no two
pairs of one benchmark do, so matching subjects on the full dict never merges two
subjects' labels within a benchmark. Some multi_swebench items share every
visible field with another item (1,342 distinct inputs among the 1,380 items of
its first pair); the platform predicts such inputs once and scores that one
prediction against each item's own outcome.
