# Findings

Every number below names the script that produces it. Where a result is negative
it is written as negative, because the negatives here were more expensive to get
than the positives and are the reason the remaining plan is what it is.

## How much there is to win

`python experiments/ceilings.py`

| ceiling | Brier |
|---|---|
| always 0.5 | 0.2500 |
| the pair's true accuracy | 0.176 |
| the true per-item probability | 0.0433 |

Knowing a system's average accuracy is worth 0.074. Knowing which items it gets
right is worth another 0.133, so the per-item structure is the whole game. On
swe_rebench the contrast is total: mean accuracy 0.479, so the average is worth
0.0004, while the per-item ceiling is 0.0505.

The per-item ceiling is estimated as k(n-k)/(n(n-1)) averaged over items with
repeats, which is unbiased and needs no fitting. It is only computable where
repeats exist, which is most of matharena and swe_rebench and very little of
multi_swebench.

## The predictor ladder

`python experiments/ladder.py --seeds 4`

| predictor | ALC | sd over splits |
|---|---|---|
| always 0.5 | 0.2500 | 0 |
| empirical mean, the official baseline | 0.2453 | |
| smoothed mean Beta(2,2) | 0.2089 | 0.0011 |
| own labels plus attribute prior, no pooling | 0.204 | |
| pooled difficulty plus attribute prior | 0.1813 | 0.0017 |

The official baseline collapses at budget 1 to 0.3734: one label makes it predict
0 or 1. Adding a Beta(2,2) prior to it, a one-line change, is worth 0.036.

The two working levers compose almost additively. The attribute prior is worth
+0.0042 +- 0.0010 paired; pooled difficulty is worth about 0.020.

## What transfers between benchmarks

`python experiments/transfer.py`

Item difficulty from text, leave-one-benchmark-out: 0.14 on matharena, 0.23 on
multi_swebench, 0.09 on real_webagents, -0.22 on researchcodebench, 0.16 on
swe_rebench. Correcting the target for subject mix and dropping items whose text
carries no task content does not rescue it.

Within a benchmark the same model reaches 0.73 on matharena, but most of that is
source identification rather than difficulty: one benchmark_id contains 25
different competitions, competition alone explains 40% of the variance, and
removing the competition mean drops the correlation to 0.49.

Subject standing from attributes, leave-one-benchmark-out: 0.48, 0.38, 0.64, 0.60.
Provider, release date, size parsed from the name, reasoning effort and harness do
not depend on the benchmark, which is exactly what the item text does not manage.

The benchmark's level does not transfer. Levels differ by 0.81 in logit units and
mixing the training mean into the prior costs 0.004, so only relative standing
goes into the prior.

Two data problems surfaced here and are worth knowing. A third of matharena items
have no task text at all: they are Math Kangaroo items whose content is "See
image", or fragments of a system prompt. And the naive difficulty target, the share
of subjects who solved an item, is confounded by which subjects attempted it:
matharena's 2026 contests were only ever run against 2026-era models, so their
items look easy for reasons that have nothing to do with the items.
`paiec/rasch.py` divides subject ability out.

## Language-model difficulty judgement

`python experiments/llm_rating/analysis.py`

180 items across four benchmarks, rated blind on a 0 to 100 scale against Rasch
difficulty:

| benchmark | n | Pearson | 95% CI |
|---|---|---|---|
| matharena | 45 | 0.482 | [+0.22, +0.68] |
| multi_swebench | 45 | 0.133 | [-0.17, +0.41] |
| real_webagents | 45 | 0.210 | [-0.09, +0.47] |
| swe_rebench | 45 | -0.128 | [-0.41, +0.17] |
| pooled within benchmark | 180 | 0.174 | [+0.03, +0.31] |

It works on mathematics and nowhere else. A control on 60 fresh multi_swebench
items with the full issue text rather than a 450-character truncation gives
Pearson +0.252 with the interval touching zero, and Spearman only +0.109, so even
that rests on a handful of extreme items. Truncation was not the explanation.

The feature is still safe to ship: the per-pair slope collapses to zero where the
signal is absent and the Laplace shrinkage keeps it from doing damage at small
budgets. It is just not a lever.

## The offline bank

`python experiments/inventory_scan.py`

The organisers' inventory holds 161 benchmarks, not 100. Of the 132 repositories
that could be cloned, 25 publish any data file under a results-like directory, 15
publish one whose path names a model, and 5 cover at least nine distinct models:
phyblock (19), mmdocrag (62), atmossci_bench (26), capability (11), engdesign (9).

Of those five, only capability publishes graded outcomes. The rest publish raw
model responses that would have to be run through each benchmark's own grader,
which is precisely the expensive part the organisers' curation pipeline exists to
do. So the usable count is one in 161.

Nine models is the floor for usefulness: the pooling gain at ten subjects is
+0.004.

## How fragile the pooling gain is

`python experiments/pool_robustness.py`

Nested subject subsets inside a benchmark, scoring the same three pairs at every
pool size:

| subjects in the pool | gain over no pooling |
|---|---|
| 3 | +0.0020 |
| 5 | +0.0024 |
| 10 | +0.0044 |
| 20 | +0.0072 |
| 40 | +0.0116 |
| 80 | +0.0169 |

The gain grows roughly as N to the 0.7. The public benchmarks have 81 and 82
subjects, the top of that curve. The hidden test is built from freshly curated
NeurIPS 2025 benchmarks, and running eighty systems on each of 161 of them is
expensive, so there is no reason to expect the top of the curve there.

An earlier version of this measurement mixed benchmarks across rows, because only
two benchmarks have enough subjects to fill the N=40 and N=80 rows. The shape
survived the correction; the magnitudes in the first pass were inflated.

## Acquisition and calibration

`python experiments/acquisition.py --seeds 4`

No policy beat the evaluator's random one. A-optimal selection targeting the
evaluation pool is worse, and the reason is instructive: it concentrates labels on
informative items and cuts the distinct items the shared difficulty estimate sees
from 1362 to 1107 on multi_swebench. Coverage-first selection reverses that, taking
coverage to 1658, and moves ALC by -0.0000 +- 0.0013. Coverage is not the binding
constraint either.

Post-hoc calibration does not transfer. Temperature and slip fitted on four
benchmarks and applied to the fifth lose 0.0007 +- 0.0011, and the chosen
temperatures range from 0.8 to 1.6 across folds, so the calibration constant is
itself a property of the benchmark. The shrinkage parameters are already at their
optimum: v_a of 2.0 beats 1.0 by 0.0015 and 3.0 by 0.0001; v_b of 0.25 beats 0.15
by 0.0008 and ties 0.40.

The guessing floor for multiple-choice items is implemented and unmeasurable here:
the public benchmarks contain 24 multiple-choice items out of 10,510. It stays in
as insurance for a test pool that should contain many more.

## What is left

Four routes to per-item difficulty have been tried. Pooling other subjects' labels
works but scales with a subject count we cannot see. Text does not carry the
signal outside mathematics. The offline bank covers one benchmark in 161. No
acquisition policy helps.

The gap from 0.181 to the 0.043 ceiling is almost entirely per-item structure on a
benchmark nobody has seen before. Further hyperparameter work will return nulls of
the same size as the ones above. The next thing that actually resolves something is
a submission: it settles whether labels pool across subjects, which is the
difference between 0.204 and 0.181, and puts a first point on the leaderboard.
