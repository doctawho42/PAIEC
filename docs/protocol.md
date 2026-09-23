# The protocol, as reconstructed

Taken from the competition page and the baseline repository. Everything below is
enforced by `paiec/evaluator.py` and pinned by the twenty tests in
`tests/test_evaluator.py`.

## Interfaces

    predict(input, labeled) -> float          # input = [subject, item]
    acquisition_function(input, prediction=None, labeled=None, context=None) -> bool

The evaluator inspects the acquisition signature: if it has no `prediction`
parameter, no prediction is computed for the acquisition decision. `context`
carries subject_id, benchmark_id, labels_acquired, labels_remaining,
max_labels (31) and items_remaining, counting the current candidate.

## Flow

Items of a subject-benchmark pair split 50/50 into acquisition and evaluation
pools when there are at least 80 distinct items. All recorded responses of an
item stay on the same side. The split is fixed across budgets and submissions.

Candidates from the acquisition pool arrive one at a time in a fixed random
order. Returning True reveals one recorded response, False skips the item
permanently.

One trajectory yields nested label sets. The submission is scored at budgets
0, 1, 3, 7, 15 and 31 by truncating that trajectory, so the ordering of what was
taken matters as much as the set.

Brier is averaged over evaluation responses inside a pair, then equally across
pairs regardless of how many responses each has.

    ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31

The marginal weight of the k-th acquired label is the sum of the weights of the
budgets it survives into: 0.9 for the first, 0.7 for the second and third, 0.5
for the fourth through seventh, 0.3 for the eighth through fifteenth, and 0.1
for the sixteenth through thirty-first.

## Reconstructed rather than read

Two things are not pinned down by the published text and are flags in the code,
so their cost can be measured instead of assumed.

The exact form of the evaluator's default random policy. The replica takes a
candidate with probability labels_remaining / items_remaining, which draws a
uniformly random subset and fills the budget exactly when the pool runs short.

Whose labels appear in `labeled`. The published text says entries appear
"across sampled subject-benchmark pairs". `run_session(..., cross_pair=True)`
makes a subject's labels from already-processed pairs visible to its later
pairs; `cross_pair=False` isolates each pair. Whether other *subjects* are
included is the single most consequential unknown in the whole problem.

## A defect worth naming

The first version of the replica keyed the split on `hash((subject_id,
benchmark_id))`. Python salts string hashing per process, so the split that the
protocol requires to be fixed silently changed between runs, and every measured
difference carried that noise. `paiec.evaluator.stable_hash` replaces it with a
blake2b digest; `tests/test_evaluator.py` checks the split is stable.

Residual split noise is sd 0.0012 to 0.0017 of ALC per measurement. Differences
below roughly 0.004 have to be measured paired, across seeds, or not claimed.

## Data

measurement-db holds 571,921 responses over 287 subjects and six benchmarks.
Five are binary; mmdocrag is fraction-valued, with only 4.8% of its responses
equal to 0 or 1, and matharena has a 0.1% non-binary tail. After dropping the
fractional ones, 221 pairs and 225,843 responses remain.

Repeats are wildly uneven: 99.8% of matharena items have them, 99.6% of
swe_rebench, 0.4% of multi_swebench. Only 22 of 287 subjects appear on more than
one benchmark. `interactors` is empty everywhere. Twenty-four of the 221 pairs do
not have a unique set of visible subject attributes, so the official baseline,
which matches subjects on their complete visible attributes, pools labels from
genuinely different subjects.
