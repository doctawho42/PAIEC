# PAIEC

Work on the Predictive AI Evaluation Competition (Stanford AIMS Lab, NeurIPS 2026):
predict whether a given AI system answered a given benchmark item correctly, with a
label budget of 0 to 31, scored by Brier ALC.

This repository holds a replica of the competition's streaming evaluator, the
predictors measured against it, and the experiments behind every number quoted in
`docs/`. Nothing here is a guess: each figure comes from a run, and the ones that
came out negative are written down as negative.

## The task in one page

`predict(input, labeled) -> float` where `input = [subject, item]`. The subject
carries normalized_name, provider, release_date, access_date, harness,
harness_version, reasoning_effort, subject_features_extra. The item carries
item_content, item_features, interactors and an anonymous benchmark_id.

Scoring is streaming. Candidates arrive one at a time in a fixed random order and
`acquisition_function` reveals one or skips it permanently. Items split 50/50 into
acquisition and evaluation pools when there are at least 80 of them. One trajectory
gives nested prefixes at budgets 0, 1, 3, 7, 15, 31. Brier is averaged over
responses inside a pair, then equally across pairs, and

    ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31

so the first acquired label counts toward every budget and is worth 0.9, while
labels 16 to 31 are worth 0.1 each.

## Where things stand

Measured on 221 subject-benchmark pairs from measurement-db, averaged over four
evaluation splits.

| predictor | ALC | sd |
|---|---|---|
| always 0.5 | 0.2500 | 0 |
| empirical mean (the official baseline) | 0.2453 | |
| smoothed mean, Beta(2,2) | 0.2089 | 0.0011 |
| own labels plus the attribute prior, no pooling | 0.204 | |
| pooled difficulty plus the attribute prior | 0.1813 | 0.0017 |
| the assembled run-time predictor, conservative order | 0.1898 | |
| the same, if the evaluator interleaves budgets per pair | 0.1725 | |
| oracle: the true per-item probability | 0.0433 | |

The official baseline is worse than a constant at budget 1 (0.3734), because one
label makes it predict 0 or 1.

The two run-time figures are the same predictor scored under two readings of the
evaluation order, and the gap between them is 0.0173. `paiec/evaluator.py` scores a
pair at all six budgets before moving on, so a predictor that keeps state sees
earlier pairs' full 31-label trajectories while it is still being scored at budget
0. Whether the real evaluator does that or sweeps budget by budget is not
documented. `experiments/order_sensitivity.py` measures both; quote 0.1898 until
the organisers say which it is.

## What transfers between benchmarks, and what does not

Item difficulty from text does not transfer. Training on four benchmarks and testing
on the fifth gives a Pearson correlation around 0.08, and the sign flips on one of
them. Judging difficulty with a language model instead gives 0.48 on competition
mathematics and nothing distinguishable from zero on software-engineering and
web-agent tasks; the full issue text instead of a truncated one does not change that.
The difficulty of a bug report is not in the bug report, it is in the repository
size, the number of files to touch and the test harness.

Subject ability does transfer. Predicting a subject's standing from its attributes,
again holding out a whole benchmark, gives 0.38 to 0.64.

The benchmark's own level does not transfer either. True levels differ by 0.81 in
logit units, and mixing the training mean into the prior makes ALC worse, so the
prior carries relative standing only and the level is left to the labels.

## Things that did not work

An offline bank of per-item results is closed: of the 161 benchmarks in the
organisers' inventory, five publish per-item outputs for at least nine models, and
only one of those five publishes them already graded rather than as raw responses.

No acquisition policy beat the evaluator's random one. A-optimal selection targeting
the evaluation pool is worse, because concentrating labels on informative items
starves the shared difficulty estimate of distinct items. Coverage-first selection
raises distinct-item coverage by about 20% and moves ALC by -0.0000 +- 0.0013.

Post-hoc calibration does not transfer: temperature and slip fitted on four
benchmarks and applied to the fifth lose 0.0007 +- 0.0011, and the fitted
temperatures themselves range from 0.8 to 1.6 across folds.

## Three bugs worth remembering

All three were the same mistake, mishandling the second-order term, and all three
announced themselves the same way: a spike at budgets 1 and 3.

A joint IRT scored by its MAP point estimate, with no integration over the
posterior, gave 0.2405, worse than a constant, dropping to 0.311 at budget 1. A
Laplace approximation fixed it.

Standardising the difficulty scale after shrinking it cancelled the shrinkage,
costing 0.02.

A Hessian written as `np.sum(p * (1 - p))` where `p` is a scalar lost the factor
`n`, so Newton overshot and predicted 0.0001 where the truth was 0.082.

There was also a reproducibility defect: the split used `hash()` on strings, which
Python salts per process, so the supposedly fixed 50/50 split changed between runs.
`paiec.evaluator.stable_hash` fixes it. Split noise is sd 0.0012 to 0.0017 per
measurement, so any difference below about 0.004 needs paired seeds to be readable.

## Layout

    paiec/            the library
      evaluator.py    replica of the streaming evaluator, covered by tests/
      data.py         measurement-db to Pair objects
      fetch.py        download the gated core tables
      items.py        per-item table and cheap text features
      rasch.py        item difficulty with subject ability divided out
      mcq.py          guessing floor read off the item text
      subjects.py     subject attributes and the ability prior
      irt.py          joint IRT, text embeddings
      fitting.py      the per-pair two-parameter fit with Laplace covariance
      trajectories.py acquisition trajectories
      pipeline.py     pooled scoring, acquisition policies, calibration inputs
      acquisition.py  the A-optimal streaming policy
      predict.py      the run-time predictor, assembled
      ceilings.py     analytic per-pair ceilings
    experiments/      one script per reported result
    submission/       competition entry point
    tools/            build the submission archive
    results/          inventory scan and the blind rating samples
    docs/             the write-ups
    tests/            protocol invariants, 20 of them

## Running it

    pip install -e ".[dev]"
    python -m paiec.fetch            # needs accepted terms and HF credentials
    pytest
    python experiments/ceilings.py
    python experiments/ladder.py --seeds 4

measurement-db is gated, so it is not vendored here. Accept the terms on the dataset
page with your own account first.

## Performance

`paiec/predict.py` scores all 221 pairs, 225,843 responses, in about two minutes.
It was three orders of magnitude slower before one fix: every newly seen evaluation
item invalidated the difficulty cache and triggered a full joint IRT refit, so
matharena alone never finished. Items that arrive after the last fit now take their
difficulty from the text map with no residual, which is what an unlabelled item's
difficulty is anyway.

The remaining cost is text embedding, and it is dominated by researchcodebench,
whose items have a median length of 96,000 characters: 58 seconds for its 212 items
against 12 seconds for matharena's 1,633.

## Open questions with the organisers

Whether `labeled` carries labels from other subjects. The difference between 0.204
and 0.181 rests on it. The plan is to find out from a first submission with a
predictor that counts distinct subjects in `labeled`, rather than to ask and have
the answer published.

Whether state may persist between calls. If it may, the pool can be assembled
independently of what `labeled` contains.

How many subjects per benchmark the hidden test has. The pooling gain scales with
it: +0.002 at three subjects, +0.007 at twenty, +0.017 at eighty. The public
benchmarks sit at the top of that curve and the test ones almost certainly do not.

How Brier is computed on fraction-valued benchmarks such as mmdocrag, and whether
they are in the test at all.

Whether the evaluator scores a pair at every budget before moving to the next pair,
or sweeps budget by budget. Worth 0.0173 to a predictor that keeps state.
