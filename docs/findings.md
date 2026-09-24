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
| assembled run-time predictor, conservative order | 0.1898 | |
| the same, if budgets interleave per pair | 0.1725 | |

The official baseline collapses at budget 1 to 0.3734: one label makes it predict
0 or 1. Adding a Beta(2,2) prior to it, a one-line change, is worth 0.036.

The two working levers compose almost additively. The attribute prior is worth
+0.0042 +- 0.0010 paired; pooled difficulty is worth about 0.020.

### Evaluation order is worth 0.017

`python experiments/order_sensitivity.py`

The replica scores a pair at all six budgets before moving to the next, so a
stateful predictor is holding earlier pairs' full trajectories while being scored at
budget 0. Under that order the run-time predictor reaches 0.1725; forced to sweep
budget by budget, so that state cannot run ahead of the budget being scored, it
reaches 0.1898. B0 moves from 0.2099 to 0.2334.

Nothing in the published rules settles which order the evaluator uses. Until it is
settled, 0.1898 is the number to plan with.

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

## Under the official protocol

`python experiments/official_baselines.py --runs 600 --jobs 6` (12 minutes on six
processes; every number below is also in `results/official_baselines.json`)

Everything above was measured on the legacy pair-major replica. This section
re-measures on `paiec/official.py` (see `docs/protocol.md`), and the two sets of
numbers are not comparable. R1 is 600 formative-like runs from
`official.sample_run` at its defaults, the same runs for every predictor, with
standard errors over runs. A run holds 8.6 pairs (5 to 12), 8.5 subjects and
1,000 items; 291 of the 600 touch all five benchmarks. The single swe_rebench
pair is in 86% of runs, because the sampler draws a benchmark before a pair. R2
takes one benchmark at a time, dense: every eligible pair, whole, with standard
errors over its pairs, which share one `labeled` list.

The Predictor is the shipped one, with its attribute prior refitted on the
benchmarks absent from the run. In 48% of R1 runs no benchmark is absent, so it
runs with no prior; otherwise the prior sees 54 pairs on average. The
target-LOBO line leaves out only the target's own benchmark. That is closer to
the shipped case, where the prior may have seen a hidden-test subject on public
benchmarks. On one run, the platform's argument copies with 16 workers give
per-pair Brier and ECE bit-identical to the fast path, for all six predictors.

### Formative-sized runs (R1)

| predictor | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE-ALC |
|---|---|---|---|---|---|---|---|---|
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2132 |
| empirical mean, the official baseline | 0.2500 | 0.3746 | 0.2572 | 0.2131 | 0.1974 | 0.1913 | 0.2526 ± 0.0014 | 0.1953 |
| smoothed Beta(2,2) | 0.2500 | 0.2349 | 0.2220 | 0.2059 | 0.1957 | 0.1908 | 0.2158 ± 0.0008 | 0.1473 |
| pooled anchor | 0.2500 | 0.2411 | 0.2278 | 0.2069 | 0.1960 | 0.1910 | 0.2184 ± 0.0011 | 0.1481 |
| Predictor, run-LOBO prior | 0.2452 | 0.2313 | 0.2198 | 0.2047 | 0.1948 | 0.1833 | 0.2130 ± 0.0009 | 0.1463 |
| Predictor, target-LOBO prior | 0.2313 | 0.2238 | 0.2162 | 0.2034 | 0.1944 | 0.1832 | 0.2090 ± 0.0009 | 0.1370 |
| base-rate oracle, E[p(1-p)] | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 ± 0.0010 | |

Standard errors on the budgets are at most 0.0032 (the empirical mean at B1), and
at most 0.0016 for every other predictor. Paired ALC differences over the same runs:

| predictor | vs empirical mean | vs smoothed |
|---|---|---|
| constant 0.5 | -0.0026 ± 0.0014 | +0.0342 ± 0.0008 |
| smoothed Beta(2,2) | -0.0368 ± 0.0007 | |
| pooled anchor | -0.0341 ± 0.0005 | +0.0026 ± 0.0004 |
| Predictor, run-LOBO prior | -0.0396 ± 0.0006 | -0.0028 ± 0.0002 |
| Predictor, target-LOBO prior | -0.0436 ± 0.0007 | -0.0068 ± 0.0002 |

The ± above are over runs, and they overstate precision. The 600 runs redraw
the same 221 pairs, so runs are not independent draws of pairs. A pair-cluster
bootstrap over 150 runs (2,000 resamples of pairs) gives much wider intervals:

| difference | over runs | pair-cluster bootstrap | 95% interval |
|---|---|---|---|
| Predictor, run-LOBO, minus smoothed | -0.0028 ± 0.0002 | -0.0026 ± 0.0009 | [-0.0044, -0.0010] |
| Predictor, target-LOBO, minus smoothed | -0.0068 ± 0.0002 | -0.0069 ± 0.0016 | [-0.0101, -0.0040] |
| pooled anchor minus smoothed | +0.0026 ± 0.0004 | +0.0027 ± 0.0027 | [-0.0024, +0.0078] |

Neither interval covers variation between benchmarks, of which there are only
five. The Predictor's edge is also uneven: per pair appearance, run-LOBO minus
smoothed is -0.0061 on multi_swebench, -0.0045 on real_webagents, -0.0028 on
researchcodebench, +0.0003 on matharena and +0.0032 on swe_rebench.

At formative size the official baseline is no better than answering 0.5. One
label sends it to 0 or 1 (0.3746 at B1), and public base rates are close enough
to 0.5 that the later budgets do not pay that back. The Predictor gains only
0.0026 ± 0.0009 over the smoothed mean, or 0.0069 ± 0.0016 with the
target-LOBO prior. About 80% of the target-LOBO gain is the attribute prior at
budgets 0 to 3. Pooled difficulty adds about 0.0007, all of it at B15 and B31:
switching it off (WARMUP=1e9) leaves -0.0062. It needs 64 distinct labeled items
on a benchmark. A run has about two pairs per benchmark, at 31 labels a pair, so
it takes at least three pairs of one benchmark at budget 31, or five at budget
15.

The pooled anchor does not differ significantly from the smoothed mean once
pairs are resampled. Most of its deficit comes from swe_rebench, which has a
single subject: there the anchor has no other subjects, rebuilds both of its
shrinkage targets from the subject's own labels, and so counts them three times
(one label gives 0.744 or 0.256, against the smoothed mean's 0.6 or 0.4). On
multi-subject benchmarks the result is mixed: a gain on multi_swebench, losses on
matharena and researchcodebench. A variant that leaves the target subject out of
m_b is the obvious fix.

The Predictor's hyperparameters (v_a, v_b, slip, WARMUP, TEXT_CAP) were chosen
on these same five benchmarks, while the smoothed and anchor priors are untuned.
No labels leak, but the Predictor's margin is optimistic by a small amount.

### One benchmark, every pair (R2)

| benchmark | pairs | mean p | empirical mean | smoothed | pooled anchor | Predictor | Predictor minus smoothed |
|---|---|---|---|---|---|---|---|
| multi_swebench | 82 | 0.219 | 0.2037 ± 0.0125 | 0.1877 ± 0.0069 | 0.1683 ± 0.0087 | 0.1705 ± 0.0069 | -0.0172 ± 0.0016 |
| matharena | 81 | 0.607 | 0.2746 ± 0.0087 | 0.2269 ± 0.0045 | 0.2267 ± 0.0050 | 0.1915 ± 0.0039 | -0.0354 ± 0.0020 |
| researchcodebench | 31 | 0.354 | 0.2649 ± 0.0142 | 0.2261 ± 0.0078 | 0.2245 ± 0.0101 | 0.1870 ± 0.0075 | -0.0390 ± 0.0020 |
| real_webagents | 26 | 0.330 | 0.2586 ± 0.0234 | 0.2139 ± 0.0122 | 0.2099 ± 0.0129 | 0.1891 ± 0.0131 | -0.0248 ± 0.0035 |

These R2 numbers assume `split_scope='pair'`: each subject's items are split
independently, so other subjects' acquired labels land on most of a pair's
evaluation items (by B31 they cover 83% of them on matharena, 95% on
real_webagents and 99.5% on researchcodebench). Which scope the coordinator uses
is unknown (`docs/protocol.md`). Under `split_scope='benchmark'` that coverage
falls to 5%, 1% and 0%, and most of the item-level gain goes with it:

| benchmark | Predictor minus smoothed, pair scope | benchmark scope | Predictor B31, pair scope | benchmark scope |
|---|---|---|---|---|
| matharena | -0.0354 | -0.0159 | 0.1325 | 0.1822 |
| real_webagents | -0.0248 | -0.0113 | 0.1306 | 0.1730 |
| researchcodebench | -0.0391 | -0.0214 | 0.1250 | 0.1787 |

So "pooled difficulty works dense" holds only under per-pair splits, where it
works mostly by reading the target item's own labels from other subjects. With
per-pair splits, the Predictor's B31 is 0.122 to 0.133, below the base-rate
oracle on every benchmark (E[p(1-p)] 0.143 to 0.206). Under benchmark splits it
is not: real_webagents at 0.173 is above its oracle. One gap stands out
regardless of scope. On multi_swebench the pooled
anchor reaches 0.1702 at B1 against the Predictor's 0.2101, and 0.1683 against
0.1705 in ALC, with ECE 0.120 against 0.169. The benchmark's level (mean p 0.22)
is in the other 81 subjects' labels, and the Predictor does not read it. It fits
ability on the subject's own labels, around an attribute prior that is relative
by design, and it centres difficulties. The level was left out because it does
not transfer *between* benchmarks. Other subjects' labels on the same benchmark,
in the same run, are a different source, and one the Predictor could use.
Elsewhere the level is nearer 0.5 and the pooled anchor roughly ties the smoothed
mean.

### The empirical mean's ALC is a function of base rates

With labels revealed at the evaluation rate p, budget B costs p(1-p)(1 + 1/B),
so ALC is about 0.025 + 1.2118 E[p(1-p)]. On R1, E[p(1-p)] = 0.1809 gives 0.2443,
against 0.2526 observed: a residual of +0.0083 ± 0.0008, with a per-run
correlation of 0.85. Most of the residual comes from a formative pair's two
halves holding only about 40 items each, so their rates differ (mean (pa - p)^2
is 0.0086). The exact expectation, using the acquisition rate and the finite-pool
correction, gives 0.2496. The remaining +0.0030 ± 0.0007 comes from the fixed
stream order. A pair's first labels are nearly the same items in every run that
holds it, so averaging over runs does not average over which items come first.
With the split and stream order salted per run, the residual is -0.0005 ±
0.0006. So the replica does what the formula says, and R1's standard errors are
conditional on one split and stream order, just as the platform's are on its own.

### Latency

A Predictor evaluation call takes 0.76 ms on average on R1, with six processes
busy, and a run takes 2.4 s. The slowest call was 1.4 s: a worker's first call,
which does the fitting. Dense runs average 0.65 to 3.2 ms per call, at most
1.2 s, and dense multi_swebench takes 383 s. That run is slow because every call
re-keys the whole shared `labeled` list (2,542 entries at B31) in
`predict.records`, so a call's cost grows with the evidence. None of this comes
near a plausible timeout.

### Against the live leaderboard

Read on 2026-09-24 from the public Codabench API
(`/api/phases/29785/get_leaderboard/`): the organisers' entry scores 0.1801 and
the best entry 0.1172. Which baseline the organisers' entry is, is not stated.
The baseline repository has three: the empirical mean, BLE (an LLM predictor),
and the empirical mean with BLE acquisition. Everything below assumes it is the
first, with random acquisition. Either of the other two would break the
inversion from ALC to base rates. On public data the
empirical mean averages 0.2526 per formative run, with sd 0.036 over runs, and
2.3% of R1 runs score 0.1801 or lower. If the organisers' entry is the
empirical-mean baseline, the formula puts its run at E[p(1-p)] of about 0.128
(0.120 by a linear fit over R1 runs, which includes the split noise). Public runs
average 0.181, and their 5th percentile is 0.137. So the hidden pairs'
accuracies sit further from 0.5: about 15% or 85% if all pairs were alike, or, at
the public mean accuracy of 0.38, a spread of sd 0.33 across pairs instead of
0.235. Pairs larger than R1's 80-item cuts (the site's example evaluates 46 to
181 items a pair) shrink only the split-noise term. On R1 that term is worth
about 0.005 (0.2496 against 0.2443), not the gap of 0.07.

At E[p(1-p)] = 0.128, linear fits over R1 runs put the Predictor at 0.172
(target-LOBO 0.166), the pooled anchor at 0.169 and the smoothed mean at 0.179.
The base-rate oracle scores 0.128 there. That value sits at the edge of the
public range (2.5% of runs), so the fits are close to extrapolation. An entry at
0.1172 on such a run beats knowing every pair's exact base rate from budget 0,
so it must use per-item signal. Or it was scored on a run with lower
E[p(1-p)]: the formative sample is redrawn for every submission, a single run's
ALC has sd 0.019 to 0.036 here, and the best of many entries is a minimum over
noisy runs. These data cannot tell the two apart. What they do support is
conditional on the organisers' entry being the empirical mean: the hidden test's
base rates are then more extreme than the public ones. Under such rates the
Predictor would land about 0.01 below the organisers' entry and about 0.05 above
the best.
