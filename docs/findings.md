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

## Hierarchical model

`python experiments/hier_eval.py --jobs 8 --skip 'r2|researchcodebench|pair|hier t3 level'`
(2 h 16 min elapsed on eight processes of a machine shared with other jobs, in
four resumed invocations, one of them run into a separate `--out` alongside
another and merged; the session's logs give the elapsed time, and the results
file records only the last pass's 749 s because earlier passes' times were not
kept. Every number below is in `results/hier_eval.json`, regenerated from its
stored rows by `python experiments/hier_eval.py --summarise` against the
library of commit bba726c, the code that produced the rows; `passes` in the file
records each invocation and the digests of the code it ran)

`paiec/hier.py` is a hierarchical Bayesian predictor, eta = mu_b + theta_s +
delta_sb - g_i - e_i, fitted per checkpoint from `labeled` alone: a benchmark
level shared by every subject on it, which the Predictor does not read, an
attribute prior on the subject, item_features group effects and an integrated
item residual. Its prior and hyperparameters come from `paiec/prior.py`. This
section measures it under the official protocol against the shipped Predictor
and the smoothed mean, on identical runs.

**Leave-one-benchmark-out.** The primary line is target-LOBO: for every
benchmark of a run, the subject prior and the empirical-Bayes hyperparameters
are fitted without that benchmark (`prior.build(pairs, (b,))`), and the model
factory dispatches on the anonymous `benchmark_id`, so predict never sees a
name. The Predictor gets its attribute prior the same way (its "target-LOBO
prior" line above). Strict run-LOBO, everything fitted without every benchmark
of the run, is reported for the main configurations on the primary setting.

**Runs.** R1 draws formative-like runs under two pair weightings,
benchmark-first (`sample_run`'s default and the runs above) and pair-uniform
(`sample_run(weighting='pair')`, new: every remaining pair equally likely, so a
run holds 3.4 benchmarks instead of 4.4, matharena and multi_swebench are in
97-99% of runs and swe_rebench in 5% instead of 88%), and scores each under
split scope 'pair' and 'benchmark'. The smoothed mean, the Predictor and hier
score runs 0 to 299 of all four settings. On the primary setting
(benchmark-first, scope 'pair') strict run-LOBO and seven ablations score runs
0 to 149, and the five costly ones (Student-t level, text term, linking off and
two link weights) and four sensitivities runs 0 to 99. R2 is dense
real_webagents and researchcodebench under both scopes.

**Standard errors** are written "± run SE / pair-cluster SE / stratified SE".
The first is over runs. The other two come from a pair-cluster bootstrap
(2,000 resamples of the pairs that appear; a pair's appearances contribute its
ALC difference over the run's size), which resamples all pairs as one pool or,
stratified, each benchmark's pairs among themselves. The unstratified bootstrap
draws the single swe_rebench pair 0 to k times, while a benchmark-first run
holds it in 88% of cases and never more than once, so where that pair's
difference is large the stratified SE is the smaller and the more faithful
one. Neither covers
variation between benchmarks, so a benchmark-level line weights the benchmarks
equally and takes the SE across them. All of this is computed from the stored
per-pair rows (`hier_eval.py --summarise`).

What was cut to stay near two hours, with other jobs holding the machine at a
load average of 100 to 200: ablations under pair weighting and under split
scope 'benchmark'; strict run-LOBO outside the primary setting and beyond 150
runs; 100 rather than 150 runs for the costly ablations and the
sensitivities; dense matharena and multi_swebench (tens of minutes per
configuration; the dense multi_swebench gap that motivated the model is
therefore not re-measured here); and the Student-t level on dense
researchcodebench under scope 'pair' (its benchmark-scope twin took 35 minutes).
An identity-prior ablation was started and dropped for time.

### Against the Predictor (R1, 300 runs per setting)

| weighting, split scope | Predictor - smoothed | hier - smoothed | hier - Predictor | 95%, pair-cluster | 95%, stratified |
|---|---|---|---|---|---|
| benchmark-first, pair (primary) | -0.0067 ± 0.0003 / 0.0013 / 0.0011 | -0.0091 ± 0.0005 / 0.0021 / 0.0015 | -0.0024 ± 0.0004 / 0.0013 / 0.0008 | [-0.0047, +0.0002] | [-0.0040, -0.0008] |
| benchmark-first, benchmark | -0.0056 ± 0.0003 / 0.0012 / 0.0011 | -0.0067 ± 0.0005 / 0.0021 / 0.0015 | -0.0010 ± 0.0003 / 0.0014 / 0.0009 | [-0.0035, +0.0017] | [-0.0027, +0.0007] |
| pair-uniform, pair | -0.0072 ± 0.0003 / 0.0010 / 0.0009 | -0.0090 ± 0.0005 / 0.0013 / 0.0013 | -0.0017 ± 0.0004 / 0.0008 / 0.0007 | [-0.0033, -0.0002] | [-0.0031, -0.0003] |
| pair-uniform, benchmark | -0.0059 ± 0.0003 / 0.0010 / 0.0009 | -0.0074 ± 0.0005 / 0.0013 / 0.0013 | -0.0015 ± 0.0003 / 0.0008 / 0.0007 | [-0.0030, +0.0000] | [-0.0029, +0.0000] |

ALC on the primary setting: smoothed 0.2149 ± 0.0011, Predictor 0.2082 ±
0.0013, hier 0.2058 ± 0.0015. Pooled over pairs, hier is ahead of the
Predictor on every setting, by 0.0010 to 0.0024: 0.75 to 2.23 pair-cluster SEs,
or 1.14 to 2.86 stratified. Both 95% intervals exclude zero on the
pair-uniform, pair-scope setting, and both end at zero on the pair-uniform,
benchmark-scope one. The stratified interval also excludes zero on the primary
setting. Per pair appearance on the primary setting, the gain is on
multi_swebench (-0.0056 ± 0.0012) and real_webagents (-0.0079 ± 0.0028),
where the level shared by the run's other subjects is informative, then
researchcodebench -0.0021 ± 0.0013 and matharena +0.0010 ± 0.0014. The single
swe_rebench pair goes the other way (+0.0081; one pair, no SE).

**The single-subject pair.** hier minus Predictor with and without it:

| weighting, split scope | pooled | pooled without swe_rebench | benchmark level, 5 benchmarks | benchmark level without swe_rebench | swe_rebench (appearances) |
|---|---|---|---|---|---|
| benchmark-first, pair (primary) | -0.0024 ± 0.0013 | -0.0037 ± 0.0010 / 0.0009 [-0.0058, -0.0018] | -0.0013 ± 0.0028 (3 of 5) | -0.0036 ± 0.0019 (3 of 4) | +0.0081 (264) |
| benchmark-first, benchmark | -0.0010 ± 0.0014 | -0.0022 ± 0.0011 / 0.0010 [-0.0044, -0.0002] | -0.0001 ± 0.0026 (2 of 5) | -0.0023 ± 0.0017 (2 of 4) | +0.0088 (264) |
| pair-uniform, pair | -0.0017 ± 0.0008 | -0.0018 ± 0.0008 / 0.0007 [-0.0033, -0.0003] | -0.0000 ± 0.0027 (2 of 5) | -0.0022 ± 0.0021 (2 of 4) | +0.0087 (16) |
| pair-uniform, benchmark | -0.0015 ± 0.0008 | -0.0015 ± 0.0008 / 0.0007 [-0.0030, -0.0001] | -0.0002 ± 0.0021 (2 of 5) | -0.0017 ± 0.0018 (2 of 4) | +0.0059 (16) |

("± pair-cluster SE", then "/ stratified SE" and the pair-cluster 95%
interval without swe_rebench; at benchmark level, ± the SE across benchmarks and
the number where hier is ahead.) That one pair carries a fifth of the
five-benchmark mean. On benchmark-first runs, redrawing it is also more than
half of the unstratified bootstrap's variance (SE 0.0013 against 0.0008
stratified on the primary setting). Without it, hier leads on the
multi-subject benchmarks by 0.0015 to 0.0037 pooled (every pair-cluster
interval excludes zero) and by 0.0017 to 0.0036 at benchmark level (ahead on
2 or 3 of 4, SEs 0.0017 to 0.0021). On swe_rebench hier is 0.006 to 0.009
worse than the Predictor and 0.008 to 0.012 worse than the smoothed mean.
With no other subject on the benchmark, the pair's own first labels move its
level, and most of the loss
goes with the level prior's width: halving sigma_mu gains 0.0068 of it (see
the sensitivities).

**How many pairs of its benchmark a run holds.** Pooling a level needs other
subjects on the benchmark in the same run. hier minus Predictor per pair
appearance on the multi-subject benchmarks, by the number of pairs of the
target's benchmark in its run (appearances in brackets; pair-cluster SEs):

| weighting, split scope | alone | 2 pairs | 3 or more | as in the real run (5 alone, 4 in twos) |
|---|---|---|---|---|
| benchmark-first, pair (primary) | -0.0016 ± 0.0014 (349) | -0.0033 ± 0.0011 (656) | -0.0044 ± 0.0010 (1316) | -0.0024 ± 0.0012 |
| benchmark-first, benchmark | -0.0009 ± 0.0014 (349) | -0.0020 ± 0.0011 (656) | -0.0029 ± 0.0011 (1316) | -0.0014 ± 0.0012 |
| pair-uniform, pair | +0.0003 ± 0.0014 (302) | -0.0017 ± 0.0011 (552) | -0.0024 ± 0.0008 (1715) | -0.0006 ± 0.0011 |
| pair-uniform, benchmark | -0.0001 ± 0.0012 (302) | -0.0015 ± 0.0011 (552) | -0.0019 ± 0.0008 (1715) | -0.0007 ± 0.0011 |

A target alone on its benchmark gains nothing measurable, at most 0.0016 ±
0.0014 and on pair-uniform runs nothing at all. With one other pair it gains
0.0015 to 0.0033, and with two or more 0.0019 to 0.0044. The Predictor's own
margin over the smoothed mean depends on it much less (0.0070, 0.0064 and
0.0091 in the three classes on the primary setting). R1 runs hold about two
pairs per benchmark (8.6 pairs over 4.4 benchmarks, 2.5 per benchmark on
pair-uniform runs). The
one real formative run with feedback (below) held 9 pairs over 7 benchmarks: 5
targets alone on their benchmark and 4 sharing it with one other pair. Weighted
that way, hier's expected gain is 0.0006 to 0.0024 (SE 0.0011 to 0.0012). That
is two SEs on the primary setting and 0.5 to 1.2 SEs on the other three.

**By budget**, hier gains at B7 to B31 (-0.0007, -0.0011 and -0.0004 of ALC
on the primary setting, pair-cluster SEs 0.0001 to 0.0002), where the pooled
level and the group effects have labels to read. It ties or loses at B0 to B3
(B1 +0.0003 and +0.0007, SE 0.0006, benchmark-first; B0 and B1 +0.0004 ±
0.0003 each, pair-uniform).

**Calibration** is not a reason either way. hier's ECE-ALC is 0.010 to 0.014
above the Predictor's on every setting (primary 0.1483 against 0.1375), close
to the smoothed mean's. That gap comes mostly from the estimator. Binned ECE on
pairs of 40 to 90 evaluation items has a floor that rises with the spread of
the predictions, and hier's group effects spread them: with them off, hier's
ECE-ALC is within 0.0005 of the Predictor's. On runs 0 to 39 of the primary
setting, the difference is +0.0108 ± 0.0022 as observed and +0.0095 ± 0.0006
when every label is drawn from the model's own prediction (perfect
calibration, 200 draws per pair and budget). That leaves +0.0014 ± 0.0022 of
excess (`experiments/hier_design/audit/null_ece.py`). ECE is a diagnostic only
(`docs/protocol.md`). No run of any configuration fell back; two strict
run-LOBO fits did not converge.

### Ablations and sensitivities (primary setting)

Each option alone, against hier's default on the same runs:

| option | runs | minus default | where |
|---|---|---|---|
| attribute prior off | 150 | +0.0039 ± 0.0003 / 0.0009 / 0.0008 | B0 +0.0016, B1 +0.0014, B3 +0.0006 |
| level pooling off (a level per pair) | 150 | +0.0012 ± 0.0003 / 0.0004 / 0.0004 | B1 +0.0004, B3 +0.0006 |
| pair deviation off | 150 | +0.0001 ± 0.0001 / 0.0001 / 0.0001 |  |
| feature groups off | 150 | +0.0017 ± 0.0002 / 0.0003 / 0.0003 | B7 to B31 |
| linking off | 100 | +0.0000 ± 0.0000 / 0.0000 / 0.0000 |  |
| relink 0.1 | 100 | -0.0000 ± 0.0000 / 0.0000 / 0.0000 |  |
| relink 0.3 | 100 | -0.0001 ± 0.0000 / 0.0000 / 0.0000 |  |
| Student-t level (nu 3) | 100 | -0.0005 ± 0.0001 / 0.0002 / 0.0001 | B1 -0.0002, B3 -0.0002; 18.9 ms a call |
| line off (Laplace) | 150 | -0.0007 ± 0.0001 / 0.0003 / 0.0001 | B1 -0.0004, B3 -0.0003; 1.6 ms a call |
| hard floor (guess 1; default 0.5) | 150 | +0.0001 ± 0.0000 / 0.0000 / 0.0000 |  |
| level centre 0 instead of the mean | 150 | +0.0011 ± 0.0003 / 0.0007 / 0.0006 | matharena -0.0039, multi_swebench and real_webagents +0.004 |
| text term on | 100 | +0.0002 ± 0.0001 / 0.0001 / 0.0001 | 2.3 pair-cluster SEs; 5.3 ms a call |
| sigma_mu x0.5 | 100 | -0.0021 ± 0.0003 / 0.0008 / 0.0004 | B1 -0.0011, B3 -0.0008; swe_rebench -0.0068 |
| sigma_mu x2 | 100 | +0.0049 ± 0.0004 / 0.0015 / 0.0005 | B1 +0.0032, B3 +0.0013 |
| sigma_delta x0.5 | 100 | -0.0004 ± 0.0003 / 0.0007 / 0.0006 |  |
| sigma_delta x2 | 100 | +0.0060 ± 0.0004 / 0.0013 / 0.0008 | B1 +0.0029, B3 +0.0020 |

The attribute prior, the pooled level and the feature groups each pull their
weight. Four options do nothing measurable at formative size: the pair
deviation, linking at any weight up to 0.3, the floor's guess and the text
term, which costs 0.0002 (2.3 SEs). Linking cannot matter much here: a subject
rarely appears on two benchmarks of one run.

Sixteen options were compared with the default on the same runs: twelve
ablations and four width sensitivities. Three beat it by more than two
pair-cluster SEs: sigma_mu x0.5 (-0.0021, 2.6 SEs), the Student-t level
(-0.0005, 2.8) and the Laplace fit without the line (-0.0007, 2.2). None clears
a two-sided Bonferroni bar for sixteen comparisons (2.95 SEs); the Student-t
level clears the one-sided one (2.73). Stratified by benchmark, all three stand
at 5.0 to 5.2 SEs. Most of their unstratified SE is the swe_rebench pair being
redrawn, and each gains on it (0.0015 to 0.0068). All three act at B1 and B3,
and all three move the model less on a run's first labels. For sigma_mu x0.5
that is the width itself. The Student-t level's fitted scale is about 0.7 of
the Gaussian's. And the Laplace mode of a skewed posterior sits nearer the
prior than its mean does. So on public runs the default over-reacts at low
budgets. Doubling either width loses 0.005 to 0.006, nearly all of it at B1
and B3. The Student-t level costs seven times the default's time a call; the
Laplace fit is faster.

The width is deliberate. `prior.WIDEN` multiplies the public ML sds of the
level and the pair deviation by 1.44 and 1.4, and halving sigma_mu undoes that
and a little more (0.72 of the ML sd). On public runs the widening costs about
0.002 ALC: sigma_mu x0.5 minus the default is -0.0021 ± 0.0008. Against the
Predictor it is -0.0040 ± 0.0011, where the default is -0.0019 on the same 100
runs. The cost falls mostly on targets alone on their benchmark (-0.0028,
against -0.0011 with two or more other pairs) and on swe_rebench (-0.0068).
Keeping the widening is a bet that hidden benchmarks' levels sit further from
the public centre than the public levels sit from each other. Public runs
cannot settle that bet. The first real formative feedback supports it (next
section).

### Against the first real formative feedback

Our first Codabench submission, the shipped Predictor of commit b68492c, was
scored on one formative run: ALC 0.2113 over 9 pairs, 9 distinct subjects and
7 distinct benchmarks (two of them hold two pairs each), with 76, 50, 56, 58,
60, 44, 53, 44 and 44 evaluated subject-item pairs. The site says the platform
picks the subjects, from a hidden pool of 2025-26 models. Brier by budget
(letters are the anonymous benchmarks):

| pair | benchmark | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|
| p1 | A | 0.4224 | 0.2851 | 0.2698 | 0.2067 | 0.2085 | 0.2057 |
| p2 | B | 0.4970 | 0.2861 | 0.1394 | 0.0764 | 0.0650 | 0.0513 |
| p3 | C | 0.3583 | 0.2419 | 0.2429 | 0.2437 | 0.2751 | 0.1923 |
| p4 | D | 0.5663 | 0.3234 | 0.1379 | 0.0518 | 0.0184 | 0.0063 |
| p5 | E | 0.3032 | 0.2477 | 0.2479 | 0.2431 | 0.2437 | 0.2472 |
| p6 | F | 0.1728 | 0.1670 | 0.1675 | 0.1665 | 0.1719 | 0.1678 |
| p7 | A | 0.4771 | 0.2939 | 0.1738 | 0.1335 | 0.1285 | 0.1286 |
| p8 | F | 0.1806 | 0.1747 | 0.1760 | 0.1827 | 0.1745 | 0.1749 |
| p9 | G | 0.2519 | 0.2653 | 0.2834 | 0.2360 | 0.2386 | 0.2366 |
| mean | | 0.3589 | 0.2539 | 0.2043 | 0.1712 | 0.1693 | 0.1568 |

At B0 the Predictor was confidently optimistic, with ECE up to 0.75. Its
relative attribute standing puts a pair's level at p = 0.5 and strong 2025-26
models near 0.75, and these pairs' rates are low. It scored 0.3589 where
answering 0.5 scores 0.25, and answering 0.5 at B0 and B1 alone would have
given about 0.1996.

Assume, for a bound, that B31 is irreducible noise alone. Then p(1 - p) = B31
puts the pairs' rates at 0.29, 0.05, 0.26, 0.006, 0.45, 0.21, 0.15, 0.23 and
0.38. These are the lower roots, which the large B0 errors against predictions
near 0.75 point to. On the accuracy-logit scale that is a mean of -1.6 and an
sd of 1.5. By the logit of each pair's evaluation rate, the public R1 pairs
have a mean of -0.74 and an sd of 1.50 (`experiments/hier_design/levels.py`).
B31 includes model error, which pulls these implied rates toward 0.5, so the
true ones are if anything more extreme.

Nine pairs from one run are thin evidence, but they point the way the
leaderboard reading did ("Against the live leaderboard"). The hidden pairs are
about as spread out as the public ones, around a centre some 0.9 logit lower.
That is the case the widened level prior was built for: a new benchmark's level
far from the public centre. A prior narrowed to fit the public runs would pull
it back toward that centre. The 0.002 the widening costs on public runs is the
price of that bet, and this run supports paying it. It cannot say how hier
would have scored there. hier centres a new benchmark on the other benchmarks'
mean level (mu0 -0.4 to -1.5 on its item-level scale, target-LOBO), not on
p = 0.5, and integrates over a wide level, but nothing here measures that on
hidden pairs. Nothing was fitted to these numbers, and no prediction was shaped
to probe hidden labels.

### One benchmark, every pair (R2)

| benchmark, split scope | smoothed | Predictor | hier | hier - Predictor | ECE-ALC Predictor / hier |
|---|---|---|---|---|---|
| real_webagents, pair | 0.2139 | 0.1891 | 0.1853 | -0.0038 ± 0.0023 | 0.1697 / 0.1657 |
| real_webagents, benchmark | 0.2072 | 0.1959 | 0.1858 | -0.0101 ± 0.0033 | 0.1230 / 0.1305 |
| researchcodebench, pair | 0.2261 | 0.1870 | 0.1828 | -0.0042 ± 0.0019 | 0.1245 / 0.1190 |
| researchcodebench, benchmark | 0.2299 | 0.2085 | 0.2125 | +0.0040 ± 0.0014 | 0.1157 / 0.1174 |

SEs are over the benchmark's pairs, which share one `labeled` list and one
split. They are conditional on both, so they understate the uncertainty.
Counted by benchmark, hier leads on both under scope 'pair' (real_webagents by
1.7 SEs) and on real_webagents under scope 'benchmark'. It gains most at B0 on
real_webagents (0.2062 against 0.2253) and holds its lead through B7.

It loses dense researchcodebench under benchmark scope from B7 on (0.1951
against 0.1787 at B31). When no other subject's label sits on a target's item,
the Predictor's text prior still carries item difficulty from the labeled
items to the unlabeled ones. hier, whose text term is off, has only the paper
groups. Switching the text term on takes back two thirds of that (0.2098,
+0.0013 ± 0.0011 against the Predictor). It helps on all four dense cases
(-0.0010 to -0.0027 against hier's default), but not at formative size
(above).

Dense ablations against hier's default: level pooling off costs +0.0012 to
+0.0057, feature groups off +0.0003 to +0.0103, centre 0 +0.0003 to +0.0025,
sigma_mu x2 +0.0003 to +0.0012. The line off gains 0.0001 to 0.0004 and the
Student-t level 0.0003 to 0.0006 (three cases).

### Strict run-LOBO

Leaving out every benchmark of a benchmark-first run leaves nothing at all in
154 of 300 runs and no multi-subject benchmark in 170. fit_hyper then falls
back to `prior.REFERENCE`: for mu0 and sigma_mu in 99% of runs (mu0 = 0, a new
benchmark at p = 0.5 before attributes), for sigma_theta and sigma_attr in 92%,
and for sigma_delta and the item widths in 57%.

The line is strict about data, not about every constant. REFERENCE
sigma_delta = 2.5 lies inside the 1.6 to 2.9 that target-LOBO estimates, and
it was set knowing the public range: it is the one REFERENCE value chosen with
the public data in view (`paiec/prior.py`). It does not flatter hier. On the 93
of runs 0 to 149 where sigma_delta falls back, the earlier REFERENCE value of
1.0 scores 0.0012 ± 0.0003 better for hier run-LOBO, and hier-R minus
Predictor-R becomes -0.0028 ± 0.0005 instead of -0.0017 ± 0.0005
(`experiments/hier_design/audit/strict_ref.py`, SEs over runs).

Over runs 0 to 149: Predictor 0.2111 ± 0.0019, hier 0.2089 ± 0.0020, hier
minus Predictor -0.0022 ± 0.0004 / 0.0010 / 0.0006 (95% [-0.0039, -0.0001],
stratified [-0.0033, -0.0010]). By how many multi-subject benchmarks are left
to fit on (± run SE / pair-cluster SE):

| multi-subject benchmarks left | runs | hier-R minus Predictor-R | hier minus Predictor, target-LOBO, same runs |
|---|---|---|---|
| none | 93 | -0.0017 ± 0.0005 / 0.0012 | -0.0020 ± 0.0007 / 0.0013 |
| one | 49 | -0.0029 ± 0.0006 / 0.0009 | -0.0022 ± 0.0009 / 0.0016 |
| two or more | 8 | -0.0039 ± 0.0012 / 0.0018 | -0.0024 ± 0.0025 / 0.0024 |

The groups differ by less than their SEs, so these data do not show hier's
strict lead growing with what is left to fit on. Both models lose about 0.004
against their target-LOBO lines (hier +0.0041 ± 0.0004 / 0.0009, Predictor
+0.0043 ± 0.0004 / 0.0009).

### Latency

With eight processes on a machine shared with other jobs, an R1 evaluation
call takes 2.46 to 2.57 ms for hier against 1.94 to 2.17 ms for the Predictor,
and a run 8.0 to 8.5 s against 6.3 to 7.3 s. The slowest single calls, which
include a fit, reached 11.7 s for hier and 15.4 s for the Predictor, inflated by
the load. The Student-t level takes 18.9 ms a call, the text term 5.3 ms,
the Laplace fit without the line 1.6 ms. Dense, hier takes 5.9 to 6.7 ms a
call and at most 1.0 s, the Predictor 2.3 to 9.8 ms and at most 8.1 s; the
Student-t level 56 to 110 ms a call, 10 to 35 minutes a run.

### Verdict: keep the Predictor

The shape of the runs decides it. hier's gain over the Predictor comes from
pooling a benchmark's level across the subjects a run holds on it. It is worth
0.0015 to 0.0033 to a target that shares its benchmark with one other pair,
0.0019 to 0.0044 with two or more, and nothing measurable to a target alone on
its benchmark. The one real formative run held 9 pairs over 7 benchmarks, 5 of
them alone. Weighted like that, hier's expected gain is 0.0006 to 0.0024 ALC
(SE 0.0011 to 0.0012): two SEs on the primary setting, 0.5 to 1.2 SEs on the
other three. Pooled over R1 runs, which hold about two pairs per benchmark, it
is 0.0010 to 0.0024, or 0.0015 to 0.0037 without swe_rebench, with stratified
intervals that exclude zero on two settings. The model does what it was
built for, on runs richer than the real one.

Against that small gain:

* It loses on a benchmark with a single subject: 0.006 to 0.009 against the
  Predictor on swe_rebench, mostly through the widened level prior. The hidden
  test may hold such benchmarks, so this needs fixing before any ship.
* It is not packaged. The submission ships the Predictor's modules, and hier
  would bring `hier.py` (1,864 lines), the run-time half of `prior.py`, a new
  prior.json and a new failure surface.
* Its hyperparameters, like the Predictor's, were chosen with these five
  benchmarks in view.

Calibration is not a reason either way (above).

On public runs alone, the options that make hier react less at B1 and B3 would
be picked over its default: sigma_mu x0.5 by 0.002, about five SEs
stratified. The default keeps the widened level prior as a bet on hidden levels
far from the public centre, and the real feedback supports that bet. If hier is
shipped, it should ship with that bet made explicitly. The level prior's width
is the setting to revisit as more formative feedback arrives, read as feedback
and never probed with predictions shaped for it.

What would change the verdict:

* hidden runs that hold two or more pairs per benchmark (summative runs larger
  than the formative ones would);
* the single-subject loss fixed;
* a re-measurement of dense multi_swebench and matharena, the case the model
  was built for, which this evaluation cut.

The real feedback's larger lesson lies elsewhere. There the Predictor lost
about 0.012 of ALC at B0 and B1 alone, to over-confident attribute standings on
low-rate pairs. That is several times what hier would gain on such a run.

### The step-2 analyses behind the model

Three read-only analyses on the public data set the model's structure and the
ranges of its hyperparameters. Their scripts are now in
`experiments/hier_design/`, together with the final-review checks that the
docstrings of `paiec/hier.py` and `paiec/prior.py` cite. They were copied from
the scratch directory of the session that built the model, with paths made
relative. The README there says what each script produces and whether it was
rerun. Rerun against the library of commit bba726c, these reproduce their
recorded outputs exactly: `identity_prior.py`, `levels.py` (after `runs.py`),
`cluster_se.py` and `prior_sweep.py`; of the review checks,
`final_verify/v2_bench.py` and `final_verify/v3b_link_t.py`; and the audit's
`audit/null_ece.py` and `audit/strict_ref.py`. `fix3/lam_quad.py`, whose
original output was not kept, gives the figure hier.py cites
(`lam_quad_out.txt`). `item_signal.py` (10 minutes) and
`predictor_check.py` (300 replica runs of six Predictor variants) were not
rerun, so their numbers below are the recorded outputs, provenance until rerun.
So are the review checks the README marks that way.

**Identity priors do not transfer beyond attributes** (`identity_prior.py`,
output `identity_prior_out.txt`). This runs leave-one-benchmark-out over the
four multi-subject benchmarks. The target is a pair's standing within its
benchmark on the accuracy-logit scale (variance 1.27). The attribute ridge cuts
the held-out MSE from 1.25 (predicting 0) to 1.05 (mean Pearson 0.51). The same
model's standing on the other benchmarks is keyed on the canonical name, which
links 117 of 220 pairs. Raw, it is worse than nothing (MSE 1.35, Pearson 0.19).
After empirical-Bayes shrinkage it is no better (1.22), and on top of the
attributes it changes nothing (1.06). With benchmark offsets removed, 30% of a
named model's variance is shared across benchmarks (tau2 0.28). Once the
attribute prediction is taken out, the shared part is -0.06 [-0.21, 0.07]: what
transfers is what release date, provider and size already say. Hence
sigma_theta at its clip floor (0.1), the identity weight capped at 0.1, and a
link weight of about 0.002 with an attribute prior.

The final review of the model found that last estimate biased downward. The
attribute residual of a name on one benchmark comes from a ridge that was
fitted leave-one-benchmark-out but still saw the same name's standings on the
other benchmarks, which pulls its residuals on two benchmarks apart.
`paiec/prior.py` uses the same estimator for tau2_res. Until the estimator is
corrected and this is re-measured, the identity conclusion, and the small link
weight it sets, are provisional.

**item_features groups** (`item_signal.py`, output `item_signal_out.txt`).
Rasch variance components per feature key: matharena's `competition` holds
0.50 of the item variance (27 levels), researchcodebench's `paper` 0.43 (20),
real_webagents' `website` 0.24 (12) and multi_swebench's `lang` 0.05 (8).
These were estimated from the labels a formative run holds (subject levels
unknown, one or two pairs of the benchmark). Group effects gain 0.0070 ALC per
pair on matharena, 0.0019 to 0.0038 on researchcodebench, 0.0012 to 0.0017 on
real_webagents and nothing on multi_swebench. With one shared prior share swept
from 0.02 to 0.5, the informative keys gain more as the share grows, while
multi_swebench's useless key loses (0.0015 at 0.5). So the share is the median
of the fitted ones, capped at 0.25 (`prior.G_CAP`). Cheap text statistics do
not transfer between benchmarks (leave-one-out Spearman -0.23 to 0.31).

**Level prior: centre, width, Student-t tails** (`levels.py`, `prior_sweep.py`,
`cluster_se.py`, `predictor_check.py`; outputs `levels.json`,
`prior_sweep_out.txt`, `cluster_se.json`, `predictor_check_out.txt`). On the
pair-accuracy logit scale, the five public levels average -0.56 with sd 0.90
(95% interval 0.54 to 2.60). A benchmark's leave-one-out level misses by 0.86
on average, and pairs spread around their level with sd 0.93. The R1 runs are
scored exactly for count-based predictors. On them, a Gaussian level prior at
the public fit beats Beta(2,2) by 0.0028 (pair-cluster SE 0.0013) when fitted
leave-one-benchmark-out, and by 0.0055 (0.0014) when fitted on all five: the
in-sample centre is optimistic. Student-t tails (3 df) at the same scale gain
less (0.0020, SE 0.0016). Across six synthetic scenarios with more extreme base
rates, the Gaussian widened to 2.3 / 1.3 had the smallest worst-case regret
(0.0073, against 0.0091 for the best Student-t and 0.0271 for Beta(2,2)). A
Student-t prior was best only where the truth itself had Student-t tails.
Added to the Predictor, a leave-one-out level gained 0.0006 (run SE 0.0003) and
an in-sample one 0.0031. Hence a widened Gaussian level by default, the
Student-t level as an option, and mu0 refitted without the scored benchmark.

The sweep scored the R1 runs of seed 0 (runs 0 to 599, `runs.py`), the same
runs this evaluation scores on its primary setting. So hier's level widths were
not chosen on held-out runs. The widened prior chosen there scored worse on
those runs (0.2161 against the best 0.2102, 'R1 exact'), so this bias runs
against hier. The evaluation above agrees at formative size: the Student-t
level is within 0.0005 of the Gaussian, centre 0 loses 0.0011 to the mean, and
a narrower level prior would score about 0.002 better on public runs.

## Calibrating for the hidden test

`python experiments/level_calibration.py --stage S --resume --jobs 6` for S =
grid, grid2, r1base, eb, t3, prob, eb, screen, confirm, final (in that order;
each stage's candidates depend on the stored results of the earlier ones), then
`python experiments/level_calibration.py --summarise` (3 h 11 min elapsed on six
processes of a machine shared with other jobs, plus about 13 minutes lost to two
aborted launches of the confirm stage; `passes` in the results file records
every stage's command, wall time and the digests of the code it ran, and the
library files are the same in every pass. Every number below is in
`results/level_calibration.json`, whose summary `--summarise` rebuilds from the
stored rows)

The first formative feedback ("Against the first real formative feedback") put
the hidden pairs far below the public ones and the shipped Predictor's B0 at
0.359, where answering 0.5 scores 0.25. This section chooses what to ship for
that: on test-like runs (`paiec/testlike.py`), with public formative runs as a
guard, and with the one feedback run as a sanity check only.

**Runs.** The primary regime is `testlike.Regime()` at its defaults, seed 2: the
runs of `experiments/testlike_check.py`'s check phase, whose stored Predictor
rows the harness reproduces to 5e-7 (their rounding). Configurations are
selected on runs 0 to 99 and confirmed on runs 100 to 199. The guard is public
R1 (`official.sample_run`, seed 0, split scope 'pair', the runs of
`experiments/hier_eval.py`), benchmark-first and pair-uniform, runs 0 to 99 of
each: a configuration may lose at most 0.003 ALC against the shipped Predictor
on either weighting. Four sensitivity regimes (seed 3, runs 0 to 99) move one
knob each: the level centre, which one feedback run does not identify
(`level_mean` -1.2 and -2.0 instead of -1.6), item structure (groups merged at
random, no strata), and no date shift (the subjects' own dates, so no attribute
prior is inflated).

**Leave one parent out.** Every target's attribute prior and empirical-Bayes
hyperparameters are fitted without its parent benchmark (`prior.build(pairs,
(parent,))`, `predict.fit_prior`); the factory maps the anonymous `benchmark_id`
to them, so predict never sees a name. On public runs the parent is the
benchmark itself.

**Harness.** All candidates of a run are scored on the same checkpoints by a
harness that replays `official._slots` and `official._acquire` (the platform's
random policy) and evaluates every checkpoint with a fresh instance, one worker
and deduplicated inputs, without argument copies. On two test-like runs and one
public run it gives per-pair Brier identical to `official.run_official` for the
smoothed mean, the Predictor, hier and a re-centred hier (`--verify`).

**Candidates.**

* smoothed Beta(2,2), the shipped Predictor, and hier at its target-LOBO
  defaults.
* The Predictor with a level fix, prior mean `sc * m + off` (m its attribute
  mean, accuracy-logit scale) and prior variance `va`. `va` is a constructor
  argument; the mean is not, so a subclass overrides `prior_mean` (as
  `testlike_check.Variant` does). The shift acts inside the model, so labels
  update from the moved prior; the output is not wrapped. Grid: off in {-2.5,
  -2, ..., 0} x sc in {0.5, 1} x va in {2, 3, 4.5}, then off -3.5 and -3 and sc
  0 and 0.25. All variants of a run share one Evidence and IRT fit per
  checkpoint (`PredictorGrid`); off 0, sc 1, va 2 is the shipped Predictor call
  for call. Each variant was also scored with `robust_fit_ab` in place of
  `fitting.fit_ab` ("Pr" rows, below).
* hier with the level prior set for the test: mu0 and sigma_mu replace the
  fitted ones, attr_scale multiplies the attribute standing's mean, and the rest
  stays as fitted. The first grid, mu0 in {-2.5, ..., 0} x sigma_mu in {0.9,
  1.3, 1.8, 2.5} x attr_scale in {0.5, 0.75, 1}, put its best on three of its
  edges (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5). hier's level is on the
  item-level scale, where -2.5 is about -1.15 on the accuracy-logit scale (a
  factor (1 + pi T / 8)^-1/2 = 0.46 for the item variance T), and a predictive
  sd near 4.6 pulls its B0 prediction toward 0.5. So the grid was extended to
  mu0 in {-5, -4, -3.5, -3} x sigma_mu in {1.3, 1.8, 2.5} x attr_scale in {0.25,
  0.5, 1}. The Student-t level (nu 3) was scored at the Gaussian grids' two best
  (mu0, attr_scale), at scales 1.3 and 1.8.
* Pair-level empirical Bayes (`EBHier`, in the script). At every checkpoint the
  level prior's centre and scale are re-estimated from every pair in `labeled`,
  on all benchmarks. Each pair is eta_p = mu + m_p + u_p, with u_p ~ N(0, S^2)
  and m_p the subject's prior mean standing, and each label is
  Bernoulli(sigmoid(0.46 eta_p)). The MAP of (mu, log S) under mu ~ N(mu0,
  tau^2) and log S ~ N(log S0, 0.3^2), S0 the base's implied pair spread, gives
  mu0 := mu ('c') and sigma_mu := (S^2 - sigma_theta^2 - sigma_delta^2)^1/2,
  floored at 0.5 ('s'). B0 is the base's. tau is 1, or 2 where the name says
  'tm=2.0'. The base is either the fitted defaults or a re-centred Gaussian
  config.

**Selection.** The best mean ALC on the selection half, among configurations
that lose at most 0.003 against the shipped Predictor on both public weightings.
Every Predictor variant is scored on all the public runs. hier configurations
are screened first on benchmark-first runs 0 to 39: twenty Gaussian configs
along the shift frontier at sigma_mu 2.5, which is the best width at every shift
on the selection half, plus every EB config. The nine best that pass the screen
go to confirmation. So do three configs whatever they lose on the screen, for
reference: the two best Gaussian configs and the best EB config on the defaults.
The best Student-t config is scored on test-like runs only, so it cannot be
chosen.

**Statistics.** Paired ALC differences against the shipped Predictor, written "±
run SE / cluster SE / stratified SE". The first is over runs. The second is a
cluster bootstrap (2,000 resamples, a ratio estimator) with (parent, subject) as
the cluster on test-like runs, because pseudo-benchmarks of one parent overlap
(`testlike.cluster_key`), and (benchmark, subject) on public runs. The third is
the same bootstrap stratified by parent benchmark. None covers variation between
benchmarks, and four parents carry every test-like number.

### What moving the level buys, unconstrained

Selection half (runs 0 to 99), best of each family:

| family | best configuration | ALC | minus Predictor | public, benchmark-first / pair-uniform |
|---|---|---|---|---|
| Predictor (shipped) | | 0.2039 | | |
| hier, fitted defaults | | 0.1906 | -0.0134 ± 0.0009 / 0.0017 / 0.0014 | -0.0019 / -0.0017 |
| smoothed Beta(2,2) | | 0.1788 | -0.0252 ± 0.0013 / 0.0021 / 0.0019 | +0.0070 / +0.0075 |
| hier, Student-t level | mu0 -4, scale 1.8, attr_scale 0.5 | 0.1570 | -0.0469 ± 0.0025 / 0.0042 / 0.0036 | not scored |
| hier, Gaussian level | mu0 -4, sigma_mu 2.5, attr_scale 0.5 | 0.1572 | -0.0467 ± 0.0025 / 0.0042 / 0.0036 | +0.0036 / +0.0032 |
| hier, EB centre and scale | on the Gaussian above | 0.1567 | -0.0473 ± 0.0025 / 0.0042 / 0.0036 | +0.0034 / +0.0033 |
| Predictor fix, fit_ab | off -2.5, sc 1, va 2 | 0.1562 | -0.0477 ± 0.0027 / 0.0045 / 0.0038 | +0.0329 / +0.0329 |
| Predictor fix, robust fit | off -2.5, sc 1, va 2 | 0.1560 | -0.0480 ± 0.0027 / 0.0045 / 0.0038 | +0.0171 / +0.0192 |

The shipped Predictor is the worst candidate here, worse than answering from
Beta(2,2). What it loses is its prior: on all 200 runs its B0 is 0.351 and its
B1 0.264, where a moved prior scores 0.19 to 0.20 and about 0.18. Every family,
once its level is moved, reaches 0.156 to 0.157. That plateau is flat: 14 of the
108 Gaussian configurations lie within 0.001 of the best. On all 200 runs,
hier's best gains more at B7 to B31 (-0.016, -0.008 and -0.007 against the
Predictor), where the pooled level and group effects read the labels, and the
Predictor fix's more at B0 and B1 (-0.162 and -0.087, against hier's -0.151 and
-0.081). At every shift the widest level prior scored best: sigma_mu 2.5 beat
1.8, 1.3 and 0.9 in every (mu0, attr_scale) cell of both grids, so the edge of
the asked-for range binds. The shift trades off against attr_scale: at sigma_mu
2.5 the best mu0 is -3.5 at attr_scale 0.25, -4 at 0.5 and below -5 at 1
(0.1574, 0.1572 and 0.1574). The Student-t level matches the Gaussian (-0.0469
against -0.0467 at the same mu0 and attr_scale) at nine times its cost per call.
EB adds little on a well-shifted base (-0.0473 against -0.0467) but much on the
fitted defaults: -0.0275 against -0.0134, all of it from B1 on.

### The public guard decides

Moving the level costs on public runs, whose pairs sit near 0.38. It costs the
Predictor much more than hier. With a fixed offset its B0 and B1 losses stay
until its own labels pull the pair back. `fitting.fit_ab` adds divergences to
that when the prior sits far from the labels (below). hier gives most of its B0
and B1 losses back from B7 on, where it is ahead of the Predictor on public runs
anyway (-0.004 to -0.007 a budget at its defaults). Configurations that pass the
guard, best of each family, on both test-like halves and both public weightings:

| configuration | selection half | confirmation half | minus Predictor, confirmation | public, benchmark-first | public, pair-uniform |
|---|---|---|---|---|---|
| hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 (shipped, below) | 0.1578 (-0.0461) | 0.1655 ± 0.0031 | -0.0418 ± 0.0024 / 0.0047 / 0.0042 | +0.0008 ± 0.0012 / 0.0025 / 0.0022 | +0.0007 ± 0.0012 / 0.0021 / 0.0019 |
| hier G mu0 -3.5, sigma_mu 2.5, attr_scale 0.25 (the rule's argmax) | 0.1574 (-0.0465) | 0.1655 ± 0.0033 | -0.0418 ± 0.0026 / 0.0051 / 0.0045 | +0.0025 ± 0.0013 / 0.0029 / 0.0025 | +0.0023 ± 0.0013 / 0.0024 / 0.0021 |
| hier EB-cs, tau 2, on G mu0 -3.0, sigma_mu 2.5, attr_scale 0.5 | 0.1580 (-0.0459) | 0.1658 ± 0.0031 | -0.0416 ± 0.0022 / 0.0043 / 0.0038 | +0.0005 ± 0.0011 / 0.0022 / 0.0019 | +0.0006 ± 0.0011 / 0.0018 / 0.0016 |
| Predictor fix off -1.5, sc 1, va 3 | 0.1630 (-0.0410) | 0.1696 ± 0.0027 | -0.0377 ± 0.0016 / 0.0032 / 0.0029 | +0.0026 ± 0.0012 / 0.0025 / 0.0022 | +0.0028 ± 0.0011 / 0.0023 / 0.0019 |
| the same, robust fit | 0.1630 (-0.0410) | 0.1696 ± 0.0027 | -0.0377 ± 0.0016 / 0.0032 / 0.0029 | +0.0019 ± 0.0011 / 0.0024 / 0.0021 | +0.0024 ± 0.0011 / 0.0022 / 0.0018 |
| hier EB-cs, tau 2, on the fitted defaults | 0.1764 (-0.0275) | 0.1824 ± 0.0026 | -0.0249 ± 0.0013 / 0.0022 / 0.0022 | -0.0010 ± 0.0008 / 0.0015 / 0.0011 | -0.0007 ± 0.0007 / 0.0009 / 0.0009 |
| hier, fitted defaults | 0.1906 (-0.0134) | 0.1949 ± 0.0024 | -0.0124 ± 0.0009 / 0.0019 / 0.0016 | -0.0019 ± 0.0007 / 0.0014 / 0.0010 | -0.0017 ± 0.0007 / 0.0009 / 0.0008 |
| Predictor (shipped) | 0.2039 | 0.2073 ± 0.0022 | | 0.2068 ± 0.0024 | 0.2038 ± 0.0023 |

(ALC ± SE over runs; in brackets the difference against the Predictor on the
selection half.) The largest shift the guard allows is about mu0 -3.5 at
attr_scale 0.25 for hier and off -1.5 for the Predictor. The best Gaussian
config of the first table, mu0 -4 at attr_scale 0.5, passed the 40-run screen
(+0.0019) but not the full public runs (+0.0036 / +0.0032), and neither did its
EB variants (+0.0030 to +0.0038 on one weighting or both). Under the guard hier
leads the Predictor fix on the test-like runs: the shipped config minus the
Predictor fix is -0.0040 ± 0.0009 / 0.0016 / 0.0015 on the confirmation half and
-0.0046 ± 0.0006 / 0.0014 / 0.0012 on all 200 runs, and -0.0019 and -0.0021
(cluster SE 0.0010, 0.0009) on the two public weightings.

**Which one to ship.** The selection rule's argmax is mu0 -3.5. The next config,
mu0 -3.0 at the same attr_scale and width, is 0.0004 behind it on the selection
half (paired, ± 0.0002 / 0.0004 / 0.0003: one cluster SE), level with it on the
confirmation half (+0.0000 ± 0.0002 / 0.0004 / 0.0004) and on all 200 runs
(+0.0002, cluster SE 0.0004). It is 0.0017 cheaper on both public weightings
(paired cluster SE 0.0003), so it passes the guard with a margin of 0.0022
instead of 0.0005 to 0.0007. This recommendation ships mu0 -3.0, a tie-break on
the guard's margin that the rule as stated does not make. `SHIP` in the script
names it, and the summary's `ship` block pairs it with the alternatives. The EB
variant on mu0 -3.0 at attr_scale 0.5 ties it as well: the shipped config minus
it is -0.0002 (cluster SE 0.0006) on the confirmation half and +0.0003 and
+0.0001 (0.0005) on public runs. It adapts where a fixed prior cannot (below),
but it is about 70 lines of experiment code not in the library, with a slower
worst call. The fixed configuration changes three Hyper fields and nothing else.

**Optimism.** The shipped config gains 0.0461 on the selection half and 0.0418
on the confirmation half; the rule's argmax 0.0465 and 0.0418. Most of that
0.004 is the halves differing, not the selection. Every configuration that moves
the level loses about a tenth of its gain between the halves, chosen or not:
0.0033 for the guarded Predictor fix, 0.0039 for hier G mu0 -3.0 at attr_scale
0.5, 0.0045 to 0.0048 for the unconstrained bests of every family. The
Predictor's own ALC moves from 0.2039 to 0.2073. Leaving one parent out of the
200 runs, the shipped config's gain ranges from 0.032 (multi_swebench out) to
0.052 (matharena out).

**By budget**, the shipped config against the Predictor on all 200 test-like
runs: -0.149 at B0, -0.080, -0.038, -0.016, -0.008 and -0.007 at B31, so Brier
0.202, 0.183, 0.161, 0.150, 0.144 and 0.137. Its mean B0 prediction is 0.33
against the Predictor's 0.68. Mean B0 ECE is 0.22 against 0.42, with a largest
pair of 0.70 against 0.77. On public runs it gives back +0.017 and +0.028 at B0
(benchmark-first, pair-uniform) and +0.007 and +0.005 at B1, and takes -0.004 to
-0.008 at each budget from B7 on (B3 +0.002 and -0.001).

### Against the real feedback

The regime is a fair posterior predictive of the feedback run. The replica's
Predictor averages 0.351, 0.264, 0.199, 0.166, 0.151 and 0.145 by budget (ALC
0.2056), and the feedback's own numbers sit at z +0.18, -0.28, +0.19, +0.17,
+0.57 and +0.37 of its single-run sd. What a candidate would have scored on that
run is estimated as the feedback plus the candidate's mean paired difference
against the Predictor on all 200 runs. The 'near' column repeats it on the 50
runs whose Predictor budget profile is closest to the feedback's.

| candidate | B0 | B1 | B3 | B7 | B15 | B31 | ALC | near |
|---|---|---|---|---|---|---|---|---|
| Predictor (the feedback) | 0.359 | 0.254 | 0.204 | 0.171 | 0.169 | 0.157 | 0.2113 | |
| hier, fitted defaults | 0.345 | 0.231 | 0.187 | 0.162 | 0.164 | 0.150 | 0.1984 | 0.1995 |
| smoothed Beta(2,2) | 0.258 | 0.207 | 0.188 | 0.166 | 0.167 | 0.156 | 0.1869 | 0.1872 |
| Predictor fix off -1.5, sc 1, va 3 | 0.218 | 0.179 | 0.171 | 0.159 | 0.165 | 0.154 | 0.1719 | 0.1731 |
| hier EB-cs, tau 2, on G mu0 -3.0, attr_scale 0.5 | 0.216 | 0.173 | 0.166 | 0.155 | 0.162 | 0.149 | 0.1676 | 0.1692 |
| hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 (shipped) | 0.210 | 0.174 | 0.167 | 0.155 | 0.162 | 0.149 | 0.1673 | 0.1685 |

So the shipped config would have scored about 0.167 on the run that gave the
Predictor 0.2113: below the organisers' 0.1801 and well above the best entry's
0.1172. That is an estimate from one run, not a measurement. Nothing was fitted
to the feedback here beyond what `testlike.Regime`'s defaults already were, and
no prediction was shaped to probe hidden labels.

### Sensitivity to the regime

Minus the Predictor on 100 runs of each regime (± run / cluster / stratified SE;
the Predictor's ALC in the header):

| configuration | level_mean -1.2 (0.2117) | level_mean -2.0 (0.2005) | groups merged at random, no strata (0.2129) | no date shift (0.1729) |
|---|---|---|---|---|
| hier G mu0 -3.0, attr_scale 0.25 (shipped) | -0.0388 ± 0.0021 / 0.0044 / 0.0039 | -0.0524 ± 0.0022 / 0.0044 / 0.0039 | -0.0467 ± 0.0017 / 0.0049 / 0.0038 | -0.0139 ± 0.0014 / 0.0028 / 0.0026 |
| hier G mu0 -3.5, attr_scale 0.25 | -0.0387 | -0.0533 | -0.0471 | -0.0135 |
| hier EB-cs, tau 2, on G mu0 -3.0, attr_scale 0.5 | -0.0380 | -0.0518 | -0.0452 | -0.0144 |
| Predictor fix off -1.5, sc 1, va 3 | -0.0353 | -0.0446 | -0.0398 | -0.0137 |
| hier EB-cs, tau 2, on the fitted defaults | -0.0224 | -0.0310 | -0.0246 | -0.0108 |
| smoothed Beta(2,2) | -0.0228 | -0.0281 | -0.0260 | +0.0077 |
| hier, fitted defaults | -0.0115 | -0.0143 | -0.0080 | -0.0078 |

The shipped config's gain holds wherever the hidden level sits within the range
one feedback run allows, and it grows as that level falls. It holds with item
structure closer to the public benchmarks' (groups merged at random). Without
the date shift the attribute priors are not inflated, and the Predictor's own
ALC falls to 0.1729. Even there every moved prior still gains 0.013 to 0.014,
nearly twice what hier's defaults gain, because the pairs are still low. The
date shift is what reproduces the feedback's B0 optimism in this regime. For the
shipped Predictor it is an offset on its prior's ability. hier's prior reads
dates through the same linear design (`paiec.subjects.design_row`), so the shift
is an offset for it too, and attr_scale 0.25 takes most of it back. If the
hidden subjects' optimism has another source, the no-shift column bounds what is
lost.

### The empirical-Bayes level adapts the right way

The EB centre on the fitted defaults (tau 2), mean over each run's per-parent
models, on the item-level scale:

| regime (mean pair logit, accuracy scale) | B0 | B1 | B3 | B7 | B31 |
|---|---|---|---|---|---|
| public, benchmark-first (-0.68) | -1.20 | -1.30 | -1.24 | -1.20 | -1.15 |
| public, pair-uniform (-0.58) | -1.12 | -1.20 | -1.10 | -1.08 | -1.03 |
| test-like, no date shift (-1.33) | -1.12 | -2.04 | -2.32 | -2.53 | -2.53 |
| test-like, level_mean -1.2 (-1.10) | -1.14 | -2.90 | -3.55 | -3.87 | -4.07 |
| test-like, default (-1.29) | -1.13 | -3.10 | -3.85 | -4.18 | -4.34 |
| test-like, level_mean -2.0 (-1.56) | -1.12 | -3.30 | -4.07 | -4.52 | -4.69 |

It stays at the public centre on public runs. On test-like runs it moves most of
the way from its first labels (one per pair) and orders the regimes by their
levels. Where dates are shifted it moves further than the pair logits alone
would say, because it centres the level at attribute score 0 and the inflated
attribute standings have to be taken back. On the shifted base (mu0 -3.0) it
moves back toward the public centre on public runs (-3.0 at B0, -2.3 at B1, -1.6
at B31) and further down on test-like ones (-3.6 at B1, -4.0 at B31). The
estimated sigma_mu stays between 1.7 and 2.7. What EB adds is at B1 and B3. At
B0 it is its base, and the level fixed at B0 decides most of the test-like gain.
That is why it ties the fixed config instead of beating it. It is also why EB on
the fitted defaults recovers only 0.013 of the 0.031 that the shipped config
adds over hier's defaults on all 200 runs.

### fit_ab diverges when the prior is far from the labels

`fitting.fit_ab` (the shipped Predictor's (a, b) fit) takes full Newton steps
from the prior mean. When that mean sits far from the pair's labels, the first
step overshoots into the flat tail of the logistic. The clipped weights vanish
there and it stops. One test-like pair at 27 of 31, prior mean -1.88, va 3, came
back at a = -13.9 with the prior's variance, a prediction of 0.000 on a pair at
0.83. On 3,000 random small problems (prior mean N(0, 1.5^2), up to 31 labels)
it missed the MAP on 173. On those its log posterior was at least 20 nats below
the maximum (246 at the median), and its prediction at z = 0 was off by 0.69 at
the median and 0.89 at most. `robust_fit_ab` in the script finds the same MAP by
Newton with step halving, and agrees with fit_ab to 4e-16 wherever fit_ab
converges (summary `fit_ab`). The shipped Predictor diverges on 5 of 1,598
test-like pair appearances, costing 0.023 to 0.080 of the pair's ALC (0.0001
overall). It diverges on 8 of 3,206 in the sensitivity regimes, none of them
with the subjects' own dates, and on none of 1,748 public ones. A moved prior
diverges far more often: off -2.5 costs +0.033 on public runs with fit_ab and
+0.017 to +0.019 with the robust fit. Any Predictor level fix needs the fit
fixed first. This is a library bug, reported here and not fixed
(`paiec/fitting.py` is not this experiment's to change).

### Latency

Evaluation calls of the shipped config, six processes side by side on a shared
machine: 0.75 ms on average on test-like runs and 0.88 ms on public ones, the
slowest single call 0.45 s (a checkpoint's first call, which fits). Dense runs
(every pair of one benchmark, whole): real_webagents 2.0 ms a call and at most
0.07 s, researchcodebench 2.4 ms and at most 0.16 s. The Predictor's robust
variants, scored together, take 3.4 ms and at most 1.4 s on the latter (the
Predictor's text fit). The EB variant takes 0.93 ms and at most 0.8 s; the
Student-t level 7.4 to 8.1 ms a call and up to 8.5 s for one call, which an
undisclosed per-call timeout makes a risk for no gain. Dense multi_swebench and
matharena were not run (the Predictor took 383 s on the former, "Latency"
above); hier's cost grows with the labels a benchmark holds, so they are the
worst case left unmeasured. A formative run holds at most 1,000 items, about
3,000 evaluation calls over six checkpoints, a few seconds at these rates
against 8 hours.

### What actually shipped, after the audit

The archive ships the milder guarded configuration, `mu0 -2.5, sigma_mu 2.5,
attr_scale 0.5` (`submission/model.py` LEVEL), not the `mu0 -3.0, attr_scale
0.25` argued for below. The audit of this section (scratch provenance:
step2b audit) changed four readings:

* The regime's level_mean of -1.6 reads every feedback pair's B31 through the
  lower root of p(1-p). For p6 and p8 (bench F), B0 is already close to B31
  under an optimistic prior, so they are more likely high-rate pairs (about
  0.78), and p9 is ambiguous. The hidden levels look spread both ways, not
  uniformly low, and the aggressive config loses on high-rate pairs (about
  0.015 per pair on a matharena-like benchmark).
* The per-pair matched estimate on the feedback run is about 0.178 (0.169 to
  0.183), not 0.167; quote the expected gain as about 0.02 to 0.04, conditional
  on the hidden levels.
* The "held-out" half redraws runs from the same catalogue of pairs, so it
  measures redrawing, not generalisation to unseen pairs or parents; nested
  leave-one-parent-out selection does not carry the chosen level to matharena.
* The two configs sit on a flat plateau: the milder one gives up about 0.002 in
  the tuned regime, gains about 0.0024 on public runs, beats the Predictor there,
  and has no worse measured worst case. With a main effect near 0.03 and
  stakes of at most 0.003 either way, robustness decides.

attr_scale below 1 is a bet that the attribute prior is inflated for hidden
subjects (the regime reproduces the feedback's B0 optimism with a synthetic date
shift). The next feedback should be read per pair, pooled with the first run's
nine pairs, before refitting the level distribution once.

### Verdict: ship hier with the level moved down

Ship `paiec.hier.HierPredictor(prior, replace(hyper, mu0=-3.0, sigma_mu=2.5,
attr_scale=0.25))`, where `prior, hyper = prior.build(pairs, ())` is fitted on
every eligible public pair (`Hyper()`'s defaults are that fit, rounded). The
level is Gaussian (nu_mu 0), all Flags stay at their defaults, and the rest of
the Hyper stays as fitted (sigma_theta 0.1, sigma_delta 2.382, sigma_attr 1.018,
sigma_d 2.671, sigma_g 1.542, slip 0.01, guess 0.5). The evaluation fitted those
fields leaving each parent out, so the shipped fit on all five is close to what
was measured but not identical to it. Against the shipped Predictor this gives:

* test-like: -0.0418 ± 0.0024 / 0.0047 / 0.0042 on the held-out half (ALC 0.1655
  ± 0.0031 against 0.2073), -0.0440 ± 0.0017 / 0.0041 / 0.0036 on all 200 runs
  (0.1617 against 0.2056), -0.039 to -0.052 across the level sensitivities and
  -0.014 without the date shift;
* public: +0.0008 ± 0.0012 / 0.0025 / 0.0022 benchmark-first (0.2076 against
  0.2068) and +0.0007 ± 0.0012 / 0.0021 / 0.0019 pair-uniform (0.2045 against
  0.2038);
* on the feedback run, an estimated ALC of about 0.167 against 0.2113.

This reverses "Verdict: keep the Predictor" above. That verdict weighed a gain
of about 0.002 from pooling levels against a loss on single-subject benchmarks.
The gain here is twenty times larger, and it comes from the level prior itself.
The earlier caveats still hold, and two become blocking:

* hier is not packaged. `tools/build_submission.py` ships the Predictor's
  modules. Shipping this needs `hier.py`, the run-time half of `prior.py` and a
  prior.json carrying the three changed fields, all put through the validator,
  the purity tests and a latency check.
* Single-subject benchmarks are excluded from test-like runs by default
  (swe_rebench), where step 2 found hier losing 0.006 to 0.009 to the Predictor.
  The public guard holds it (in most benchmark-first runs), and the shipped
  config passes with it in, but its own difference on that benchmark was not
  broken out here.
* The regime was tuned to the feedback's B0 and B1 on the shipped Predictor, so
  the size of the gain rests on that tuning. The level knob is not identified,
  and the sensitivities bound it. The gain comes mostly from B0 and B1 (0.031 of
  0.044), which the next formative feedback will show directly.
* sigma_mu 2.5 and attr_scale 0.25 sit on the edges of what was scored: no wider
  level prior and no smaller attr_scale were tried.
* hier's hyperparameters, like the Predictor's, were set with the public
  benchmarks in view, and the public guard runs are the ones hier_eval.py
  already used.

If hier cannot be packaged in time, the fallback is the Predictor with prior
mean m - 1.5, v_a 3 and a robust (a, b) fit: -0.0377 on the held-out half and
+0.0019 / +0.0024 public. That is about 0.004 less than hier on test-like runs
and 0.002 worse on public ones. It needs a prior-mean offset in `Predictor` and
the `fit_ab` fix, both library changes.

**What was cut** to stay near three hours on a shared machine: 100 runs per
half, per public weighting and per sensitivity regime (`hier_eval.py` used 300
public runs); the public guard under split scope 'pair' only; the hier grid's
public cost measured on 40 runs and only along the frontier at sigma_mu 2.5; the
Student-t level on test-like runs only (four configs, so it could not be
chosen); EB with tau 1 and 2 only, the log-scale prior (0.3) and floor (0.5) not
varied; sensitivities for the chosen, the rule's argmax, three near-ties and the
references only; the regime knobs `tilt='pair'`, `level_sd`, one
pseudo-benchmark per parent, split after the cut, and the ability offset in
place of the date shift (which applies to the legacy prior only); dense
multi_swebench and matharena. The results file is 19 MB: rows are packed
(`compact_raw`) and the summary reads them back exactly.

## Item signal from the pair's own labels

`python experiments/itemsig_eval.py --stage run --jobs 4 --rows DIR`, then
`--stage oracle`, `--summarise`, `--stage verify` and `--summarise` again, all
with the same `--rows` (38 minutes on four processes of a machine shared with
other jobs, load average 13 to 21: 31 minutes for the 800 runs, 10 s for the
oracle, 6.5 minutes for verification and latency. Every number below is in
`results/itemsig_eval.json`, whose summary `--summarise` rebuilds from the rows;
the rows, per pair and configuration Brier differences by budget, are 33 MB and
live outside the repository, default `data/itemsig_eval_rows`. `passes` records
each stage's command, wall time and the digests of the code it ran; the library
files are the same in every pass. The script's oracle stage and parts of its
summary were added after the run stage, whose row-writing code is unchanged;
the verify stage reproduces stored rows with the final script)

A formative run holds about one pair per benchmark, so a target rarely sees
another subject's label on its own items. What it does see is its own pair's
acquired labels (up to 31) and the item texts. `paiec/itemsig.py` carries the
base model's residuals on those labeled items to unlabeled items with similar
text: a logit offset beta * sum w r / (tau + sum w v), with w a hashed TF-IDF
cosine to the power gamma, other subjects' records weighted `other`, residuals
centred within each pair, and an optional per-pair empirical-Bayes slope. This
section decides whether to ship it on top of the shipped model.

**Base.** Exactly what ships: `paiec.hier.HierPredictor` with
`submission/model.py`'s LEVEL (mu0 -2.5, sigma_mu 2.5, attr_scale 0.5; the
script reads LEVEL from model.py) over `prior.build`, fitted leaving each
target's parent benchmark out (`level_calibration.bundle` and `factory`).

**Runs.** Primary: test-like runs (`testlike.Regime()` defaults, seed 2, runs 0
to 299; runs 0 to 199 are those of "Calibrating for the hidden test"). As a
sensitivity, groups merged at random and wholes, no strata (`kinds=("mix",
"whole")`, seed 3, runs 0 to 199), because difficulty strata shrink the item
variance within a pair. Secondary: public R1 (`official.sample_run`, seed 0,
split scope 'pair'), benchmark-first runs 0 to 199 and pair-uniform runs 0 to
99. Test-like runs hold 8.1 pairs over 6.2 pseudo-benchmarks; 60% of pair
appearances are alone on theirs, and 40% have another subject's labels on it at
B31 (19 such labels per appearance on average, zeros included). Public
benchmark-first runs hold 8.6 pairs over 4.4 benchmarks, 24% alone;
pair-uniform runs 3.3 benchmarks, 10% alone.

**Configurations.** beta {0.25, 0.5, 1, 2, 4} x tau {0.5, 1, 2, 4} x gamma {1,
2, 4, 8} x other {0, 0.5, 1} x slope {off, on (prior sd 1)}: 480
configurations, plus "off" (the base itself). Everything else stays at
`SigHyper`'s defaults (leave-in residuals centred within pair, twins on, word
weight 0.5, max_shift 1.5). The library default is beta 0.5, tau 1, gamma 2,
other 0, slope off.

**Harness.** `level_calibration.checkpoints` replays the platform's acquisition;
every checkpoint gets a fresh base and a fresh ItemSig around it. Per target the
base is called once and `ItemSig.terms` once, and every configuration's shift
follows from those terms as `itemsig.shift` computes it (the slope through
`itemsig.slope` itself). Replayed through `official.run_official` with deep
copies and `make_itemsig`, 13 configurations on five runs (two test-like, one
of each other regime) match the stored rows pair for pair to 1.7e-9, the
float32 storage of the differences, and the base to 6e-15.

**Selection.** Nested leave-one-parent-out. For each parent q, the configuration
(or "off") with the lowest mean ALC difference over the pair appearances whose
parent is not q, scored on the appearances whose parent is q. A target's layer
reads only records on its own benchmark_id, and the base is fixed, so a pair's
difference depends on nothing from another parent; runs mix parents, so the
split is by pair appearance, each weighted 1/run size as in a run's mean. The
cluster bootstrap redoes the selection in every resample. Three selections:
within each regime; on the primary regime, scored on the others; and jointly,
by the worst regime's mean over all four. The in-sample best (lowest mean on all
appearances of a regime, scored on the same) is the optimistic secondary number.

**Statistics.** Paired ALC differences against the shipped hier, "± run SE /
cluster SE / stratified SE". Clusters are (parent, subject), resampled over the
union of all regimes' clusters (a test-like and a public cluster with the same
key hold the same responses), 2,000 resamples, a ratio estimator, as one pool or
within each parent. The run SE of a nested estimate holds its choices fixed.
Four parents carry every test-like number, and no SE covers variation between
benchmarks.

### Nested leave-one-parent-out

| regime (base ALC) | nested, selected within the regime | nested, selected on the primary regime (and jointly) | in-sample best of 480 | library default |
|---|---|---|---|---|
| test-like, primary (0.1658) | **-0.00002 ± 0.00002 / 0.00008 / 0.00008** | (same) | -0.00006 ± 0.00001 / 0.00003 / 0.00003 | -0.00002 ± 0.00001 / 0.00002 / 0.00002 |
| test-like, mix and wholes (0.1711) | -0.00028 ± 0.00005 / 0.00021 / 0.00019 | -0.00028 ± 0.00003 / 0.00011 / 0.00010 | -0.00037 ± 0.00007 / 0.00014 / 0.00013 | -0.00009 ± 0.00001 / 0.00003 / 0.00003 |
| public R1, benchmark-first (0.2057) | -0.00003 ± 0.00012 / 0.00049 / 0.00046 | -0.00042 ± 0.00004 / 0.00012 / 0.00011 | -0.00071 ± 0.00009 / 0.00017 / 0.00014 | -0.00013 ± 0.00002 / 0.00003 / 0.00003 |
| public R1, pair-uniform (0.2020) | -0.00076 ± 0.00018 / 0.00031 / 0.00029 | -0.00052 ± 0.00007 / 0.00016 / 0.00015 | -0.00112 ± 0.00020 / 0.00030 / 0.00029 | -0.00014 ± 0.00003 / 0.00004 / 0.00004 |

On the primary regime the layer is worth nothing measurable: -0.00002 ALC,
a quarter of a cluster SE. Even the best of the 480 configurations, chosen and
scored on the same runs, gains 0.00006. The library default's line agrees,
within about one cluster SE, with the itemsig docstring's own measurement on
confirmation halves (-0.00002, -0.00006, -0.00012 and -0.00013 in the table's
order). The joint selection chooses exactly what the primary regime chooses,
because the primary regime is always the worst one. On test-like runs its choices are gentle: gamma 1 and tau 0.5
throughout, beta 0.5 to 1, the slope on, other 0 except for researchcodebench
held out (0.5). Per held-out parent they score +0.00005 ± 0.00015 on matharena
(476 appearances), -0.00007 ± 0.00002 on multi_swebench (1,006), -0.00006 ±
0.00009 on real_webagents (443) and +0.00007 ± 0.00014 on researchcodebench
(500).

Scored on the other regimes, those choices gain 0.0003 to 0.0005, 2.5 to 3.5
cluster SEs. Selected within the public regimes instead, the rule picks
aggressive configurations (beta 2 to 4, other 0.5 to 1) that fail on the held-out
parent. With researchcodebench held out, benchmark-first selection takes beta 4,
tau 0.5, gamma 1, other 0.5, slope on, which loses +0.0019 ± 0.0007 on
researchcodebench and cancels the -0.0015 ± 0.0003 it gains on matharena. So
the in-sample bests on public runs (-0.0007 and -0.0011) are not what a
selection would carry to a new benchmark.

### Where it acts

B0 and B1 differences are exactly 0 for every configuration, pair and regime.
At B0 there are no labels. At B1 each pair holds one, and centring within the
pair makes its residual 0, other subjects' included. By budget on the primary
regime, nested (± run / cluster SE; the ALC contribution is the budget's weight
times the difference):

| budget | 0 | 1 | 3 | 7 | 15 | 31 |
|---|---|---|---|---|---|---|
| Brier difference | 0 | 0 | +0.00012 ± 0.00001 / 0.00008 | -0.00002 ± 0.00001 / 0.00007 | -0.00002 ± 0.00001 / 0.00009 | -0.00032 ± 0.00002 / 0.00007 |
| ALC contribution | 0 | 0 | +0.000023 | -0.000003 | -0.000005 | -0.000032 |

The layer costs a little at B3, where a pair's residuals rest on three labels,
and gains 0.0003 of Brier at B31. The primary regime's choices, scored on the
other regimes, gain at every budget from B3 on and most at B31 (-0.0007 to
-0.0015). By kind of pseudo-benchmark on the primary regime: groups +0.00007 ± 0.00006 (1,417
appearances), strata -0.00011 ± 0.00007 (689), wholes -0.00017 ± 0.00012 (319).
Targets alone on their pseudo-benchmark and those sharing it gain the same
(-0.00002 ± 0.00005 each). The public gains are on matharena and
researchcodebench, whose competitions and papers the base's group effects
shrink hard. The best configurations smooth broadly (gamma 1 is best in every
regime) and read other subjects' residuals on public runs (other 0.5 to 1):
they read what similar items share at group level, not what one item has of
its own. That is the itemsig docstring's own reading of its public gain.

### Against the item oracle

Per pair, on its evaluated responses: the pair-rate oracle (every response at
the pair's own rate) and the item oracle (the parent's in-sample Rasch
difficulty, theta fitted per pair; below). Their gap is the most an
item-difficulty model could add to a pair's Brier. Means over runs of run means,
pairs with an oracle (public runs leave swe_rebench out):

| regime | pair-rate oracle | item oracle | gap (± cluster SE), share of the pair-rate oracle | base B31 minus pair-rate oracle | nested gain, ALC (share of the gap) | nested gain, B31 (share of the gap) |
|---|---|---|---|---|---|---|
| test-like, primary | 0.1384 | 0.0750 | 0.0633 ± 0.0041, 46% | +0.0003 | 0.00002 (0.03%) | 0.0003 (0.5%) |
| test-like, mix and wholes | 0.1465 | 0.0676 | 0.0790 ± 0.0068, 54% | -0.0012 | 0.0003 (0.4%) | 0.0010 (1.3%) |
| public, benchmark-first | 0.1706 | 0.0768 | 0.0938 ± 0.0045, 55% | -0.0025 | 0.00002 (0.02%); from the primary 0.0005 (0.5%) | 0.0013 (1.4%); from the primary 0.0012 (1.3%) |
| public, pair-uniform | 0.1739 | 0.0761 | 0.0977 ± 0.0041, 56% | -0.0038 | 0.0008 (0.8%) | 0.0029 (2.9%) |

At B31 the shipped hier already scores what the pair's exact rate would on
test-like runs (+0.0003), and slightly better on the others, where its feature
groups read some item structure. What is left at the item level is 0.063 of
Brier on test-like runs, and the layer takes half a percent of it at B31.
Text similarity over a pair's own 31 labels carries almost none of the item
signal the oracle shows is there. That agrees with the ceiling the itemsig
docstring measured: smoothed by this kernel over 31 to 62 labeled items, public
difficulties net of feature groups correlate with a target's at 0.14 to 0.29 on
matharena and at 0.14 or less elsewhere.

**The library's item oracle diverges.** `testlike.item_oracle` fits theta by
full Newton steps from 0. On a pair whose items sit far from 0 on the parent's
scale these oscillate without converging: one matharena stratum pair at rate
0.67 on difficulties averaging 3.6 swings between -2050 and +4150 and scores
Brier 0.669, where the MAP (theta 4.56) scores 0.144. The script refits theta by
Newton with step halving (`theta_map`) and keeps the library's value beside it.
On the primary regime 35 of 2,425 pair appearances (31 of 300 runs), all of
them strata, are off by more than 1e-4, and there are none in the other
regimes. By plain means over pairs, `experiments/testlike_check.py`'s
definition, the item oracle gains 46.0% of the pair-rate oracle's Brier on
test-like runs, not 41.1%, and 32.7% on strata, not 14.5%; groups (50.3%) and
wholes (52.7%) are unchanged. The testlike docstring's figures and the
`structure` block of `results/testlike_check.json` carry the divergent fits.
This is a library bug, reported here and not fixed (`paiec/testlike.py` is not
this experiment's to change); strata thin the item structure less than the
docstring says.

### The grid, in sample

The best configuration at each value of one hyperparameter, primary regime
(mean ALC difference ± run SE):

| hyperparameter | values: best mean |
|---|---|
| gamma (sharpening) | 1: -0.00006; 2: -0.00002; 4: -0.00001; 8: -0.00001 |
| other (weight of other subjects' records) | 0: -0.00006; 0.5: -0.00006; 1: -0.00005 |
| beta | 0.25: -0.00004; 0.5: -0.00006; 1: -0.00006; 2: -0.00005; 4: -0.00004 |
| tau | 0.5: -0.00006; 1: -0.00006; 2: -0.00005; 4: -0.00005 |
| per-pair slope | off: -0.00006; on: -0.00006 |

(run SEs 0.000005 to 0.00002.) Sharper similarity only loses. The slope changes
nothing at the gentle settings that are best here. On public runs the best
settings are stronger (beta 2 to 4, other 0.5 to 1), which the nested selection
shows does not carry to a held-out parent. The one configuration the joint rule
would ship, chosen in sample on all four regimes, is beta 0.5, tau 0.5, gamma 1,
other 0, slope off. It gains -0.00006 ± 0.00001 / 0.00003 / 0.00003 on the
primary regime, -0.00020 ± 0.00002 / 0.00005 / 0.00005 with mixed groups,
-0.00025 ± 0.00003 / 0.00005 / 0.00004 and -0.00032 ± 0.00004 / 0.00007 /
0.00007 on public runs, and it is not significantly worse than the base on any
parent in any regime (worst: matharena on the primary regime, +0.000002 ±
0.00009).

### Latency

Through `official.run_official` with deep copies and one worker, four such
processes side by side on a machine at load average 15 to 17 (8 cores), on the
five verification runs: the shipped hier takes 0.8 to 1.5 ms a call, its slowest
call 0.07 to 0.13 s. With the layer, 2.6 to 10.8 ms a call over the 13
configurations (the library default 3.2 to 5.5 ms), and the slowest single call
0.2 to 4.6 s (the default 0.4 to 1.0 s): a pair's first target on a benchmark
makes the base calls for its labeled records (every record on the benchmark
when other > 0) and the first target of a benchmark featurises its labeled
items. Run by run that is 1.8 to 12 times the base's mean call (the default 2.8
to 6) and 2 to 52 times its slowest call (the default 3.4 to 15). A formative run
then takes 8 to 29 s instead of 3 to 5 s, against 8 hours; the slowest single
calls are what an undisclosed per-call timeout would see.

### Verdict: do not ship the layer

* On the runs that stand in for the hidden test it does nothing: -0.00002 ±
  0.00008 nested, -0.00006 for the best of 480 configurations chosen in sample.
  That is about a thousandth of a formative run's sd (0.02 to 0.04), and at B31
  0.5% of the item-level gap the oracle shows.
* Where it gains, with mixed groups and on public runs (0.0003 to 0.0005 nested,
  selected on the primary regime), it reads group structure on whole
  benchmarks, not per-item signal, and a selection made on public parents
  loses on a held-out one (researchcodebench +0.0019).
* It costs two to twelve times the base's mean call time (the default three to
  six), slowest calls 3 to 15 times the base's at the default and up to 4.6 s,
  and 760 more lines in the archive with a new failure surface, for an expected
  gain on the hidden test between 0 and about 0.0003.

If it is ever shipped, the configuration is beta 0.5, tau 0.5, gamma 1, other
0, slope off: the joint choice, never measurably worse than the base on any
parent or regime. The library default (beta 0.5, tau 1, gamma 2) is 0.00004
behind it on the primary regime and 0.0001 to 0.0002 behind on the others.

The per-item gap stays where "What transfers between benchmarks" left it. The
base already reaches the pair-rate oracle at B31 on test-like runs, and 0.063
of Brier separates it from the item oracle. Neither text similarity over the
pair's own labels (here) nor text-to-difficulty maps across benchmarks (leave
one benchmark out, correlations -0.22 to 0.23) reach that gap.

**What was cut.** The run took 38 minutes, well inside the budget; what was
left out is scope. Only the five hyperparameters above were tuned: centring,
the residual kind, twins, the word weight, max_shift and the excerpt lengths
stayed at their defaults, and the slope's prior sd at 1. Not run: dense
runs, whose itemsig docstring figures (real_webagents up to -0.002 and -0.005,
researchcodebench -0.002 to +0.008) are not re-measured here; split scope
'benchmark'; the level and date-shift sensitivity regimes (level_mean -1.2 and
-2.0, no date shift). ECE was not recorded. Latency comes from five runs on a
loaded machine. The stored differences are float32 (1.7e-9).
