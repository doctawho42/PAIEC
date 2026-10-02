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

(Added 2026-10-02.) These five are Pearson correlations with the naive target,
the solve-rate logit over all items, not with Rasch difficulty. Against Rasch
difficulty on text-bearing items they are 0.10, 0.14, -0.02, -0.16 and 0.11.
Both sets, with group-bootstrap intervals, are in "Intervals for every transfer
correlation" (`results/gate_and_ci.json`, `ci.rows`).

Within a benchmark the same model reaches 0.73 on matharena, but most of that is
source identification rather than difficulty: one benchmark_id contains 25
different competitions, competition alone explains 40% of the variance, and
removing the competition mean drops the correlation to 0.49. (Added
2026-10-02: 0.73 and 0.49 do not reproduce at HEAD. Refitted on text-bearing
items, they are 0.80 [0.70, 0.86] and 0.58 [0.48, 0.66]; same section.)

Subject standing from attributes, leave-one-benchmark-out: 0.48, 0.38, 0.64, 0.60.
(Added 2026-10-02: these do not reproduce exactly at HEAD either. `transfer.py`
prints them and writes no results file, so they are left as first recorded and
the report does not quote them.)
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

(Group-bootstrap intervals, and why a larger 4B judge run is not needed, are
in "Intervals for every transfer correlation".)

It works on mathematics and nowhere else. A control on 60 fresh multi_swebench
items with the full issue text rather than a 450-character truncation gives
Pearson +0.252 with the interval touching zero, and Spearman only +0.109, so even
that rests on a handful of extreme items. Truncation was not the explanation.

The feature is still safe to ship: the per-pair slope collapses to zero where the
signal is absent and the Laplace shrinkage keeps it from doing damage at small
budgets. It is just not a lever.

**How the ratings were made, and what is not recorded.** The 180 main and 60
control ratings are hard-coded in `experiments/llm_rating/ratings_main.py` and
`ratings_control.py`; the texts and Rasch targets are in
`results/rate2_truth.json` and `results/ctrl_truth.json`. All of them entered
the repository in its first commit (165b4c0, 2026-09-23), co-authored by Claude
Opus 5 in a Claude Code web session, and have not changed since. The rating
model, its version, the prompt, the date, the sampling settings and how the
rater was kept from the targets are recorded nowhere. The team confirmed on
2026-09-28 that the rater was the Claude model of that coding session (the
commit's trailer names Claude Opus 5); the prompt, date and settings remain
unrecorded. "Blind" means the
difficulty was withheld: the item ids carry the benchmark and the texts name
their domain. The main texts are 63 to 700 characters (median 450) and the
control texts 274 to 4,000 (median 1,030), so the control's "full text" was
capped at 4,000 characters. Whether the rater saw measurement-db (which that
session held in `data/`) or the benchmarks' public leaderboards cannot be
excluded. The correlations re-derive from the stored ratings; the ratings do
not re-derive. The fully specified substitute is the local 4B judge ("The 4B
judge, closed out").

## The offline bank

`python experiments/inventory_scan.py`

The organisers' inventory holds 161 benchmarks, not 100. Of the 132 repositories
that could be cloned, 25 publish any data file under a results-like directory, 15
publish one whose path names a model, and 5 cover at least nine distinct models:
phyblock (19), mmdocrag (62), atmossci_bench (26), capability (11), engdesign (9).

Of those five, only capability was judged to publish graded outcomes (how is
not recorded; see the end of this section). The rest publish raw
model responses that would have to be run through each benchmark's own grader,
which is precisely the expensive part the organisers' curation pipeline exists to
do. So the usable count is one in 161.

Nine models is the floor for usefulness: the pooling gain at ten subjects is
+0.004.

**What the scan read.** For each GitHub `code_url` in the inventory it ran
`git clone --filter=blob:none --depth 1`, listed the tracked files with `git
ls-files`, matched regular expressions against the file paths (a results-like
directory, a data-file extension, a model name), stored counts and up to three
example paths in `results/clone_scan.csv`, and deleted the clone. The script
opened no file's contents, but a blobless clone still checks out the default
branch, so each repository's files were downloaded to a temporary directory
and deleted with the clone. The inventory lists candidates for both pools, so
these may include hidden-test benchmarks. Nothing from the scan, `clone_scan.csv`,
`strong_repos.csv` or the inventory enters `paiec/`, `submission/` or `tools/`,
and no per-item data from any inventory repository was kept or used. The bank
was closed on feasibility and because the rules restrict competition-specific
training and curation to the public pool. (`results/strong_repos.csv` has no
producing script, and how "graded outcomes" was judged for the five is not
recorded. That judgement needs file names or contents beyond the scan's
counts, so an inspection of the five repositories' files (phyblock, mmdocrag,
atmossci_bench, capability, engdesign) cannot be excluded; mmdocrag is a public
benchmark, and the other four may be hidden-test benchmarks. Nothing from them
was kept or used. To be confirmed by the team.)

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
15. ("Pooling under the verified protocol" re-measures this switch under both
split scopes, on dense and formative-size runs and at the platform's run
composition: on dense runs pooling is still worth 0.009 to 0.030.)

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
section measures it under the official protocol against the legacy Predictor
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

Our first Codabench submission, the legacy Predictor of commit b68492c, was
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
| p8 | F | 0.1806 | 0.1747 | 0.1761 | 0.1827 | 0.1745 | 0.1749 |
| p9 | G | 0.2519 | 0.2653 | 0.2834 | 0.2360 | 0.2386 | 0.2366 |
| mean | | 0.3589 | 0.2539 | 0.2043 | 0.1712 | 0.1693 | 0.1568 |

(p8's B3 is 0.176050 in the organisers' table, `results/formative/run1.txt`;
`paiec/testlike.py`'s `FEEDBACK` carries it as 0.1760.)

At B0 the Predictor was confidently optimistic, with ECE up to 0.75. Its
relative attribute standing puts a pair's level at p = 0.5 and strong 2025-26
models near 0.75, and these pairs' rates are low. It scored 0.3589 where
answering 0.5 scores 0.25, and answering 0.5 at B0 and B1 alone would have
given about 0.1996.

Assume, for a bound, that B31 is irreducible noise alone. Then p(1 - p) = B31
puts the pairs' rates at 0.29, 0.05, 0.26, 0.006, 0.45, 0.21, 0.15, 0.23 and
0.38. These are the lower roots, which the large B0 errors against predictions
near 0.75 point to. On the plain logit of the rate that is a mean of -1.6 and
an sd of 1.5. By the plain logit of each pair appearance's evaluation rate
(clipped at 0.005, each run weighted equally), the public R1 pairs have a mean
of -0.74 and an sd of 1.50 (`experiments/hier_design/levels.py`, stored in
`experiments/hier_design/levels.json`, `r1`). On the continuity-corrected
pair logit of "Formative feedback, runs 1 and 2" the same two readings are
-1.51 and -0.71.
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

(Later uses of this run are listed in "Formative feedback, runs 1 and 2",
which also records the organisers' table byte for byte in
`results/formative/run1.txt`, with ECE, and reads it together with the second
run. The lower-root reading above is one of three readings of these nine
pairs; the audit's reads p6 and p8 as high-rate pairs, "What actually shipped,
after the audit".)

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
(matharena's 27 levels are those of its 1,751 items with a binary response;
the 1,633 items of its 81 eligible pairs span 25, because imc_2025 and
miklos_2025 occur only on subjects below the 80-item floor:
`experiments/data_counts.py`, `results/data_counts.json`.)
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
the hidden pairs far below the public ones and the legacy Predictor's B0 at
0.359, where answering 0.5 scores 0.25. This section chooses what to ship for
that: on test-like runs (`paiec/testlike.py`), with public formative runs as a
guard, and with the one feedback run as a sanity check only.

**Naming in this section.** "The legacy Predictor" is the predictor of the
first submission, which was shipped when this section was written. "The
recommended config" is `mu0 -3.0, sigma_mu 2.5, attr_scale 0.25`, which this
section argues for. It is not what ships: the audit below moved the archive to
`mu0 -2.5, sigma_mu 2.5, attr_scale 0.5`, whose own numbers are in "Shipped
configuration, confirmed". Every number this section gives for the recommended
config belongs to that config only.

**Runs.** The primary regime is `testlike.Regime()` at its defaults, seed 2: the
runs of `experiments/testlike_check.py`'s check phase, whose stored Predictor
rows the harness reproduces to 5e-7 (their rounding). Configurations are
selected on runs 0 to 99 and confirmed on runs 100 to 199. The guard is public
R1 (`official.sample_run`, seed 0, split scope 'pair', the runs of
`experiments/hier_eval.py`), benchmark-first and pair-uniform, runs 0 to 99 of
each: a configuration may lose at most 0.003 ALC against the legacy Predictor
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

* smoothed Beta(2,2), the legacy Predictor, and hier at its target-LOBO
  defaults.
* The Predictor with a level fix, prior mean `sc * m + off` (m its attribute
  mean, accuracy-logit scale) and prior variance `va`. `va` is a constructor
  argument; the mean is not, so a subclass overrides `prior_mean` (as
  `testlike_check.Variant` does). The shift acts inside the model, so labels
  update from the moved prior; the output is not wrapped. Grid: off in {-2.5,
  -2, ..., 0} x sc in {0.5, 1} x va in {2, 3, 4.5}, then off -3.5 and -3 and sc
  0 and 0.25. All variants of a run share one Evidence and IRT fit per
  checkpoint (`PredictorGrid`); off 0, sc 1, va 2 is the legacy Predictor call
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
that lose at most 0.003 against the legacy Predictor on both public weightings.
Every Predictor variant is scored on all the public runs. hier configurations
are screened first on benchmark-first runs 0 to 39: twenty Gaussian configs
along the shift frontier at sigma_mu 2.5, which is the best width at every shift
on the selection half, plus every EB config. The nine best that pass the screen
go to confirmation. So do three configs whatever they lose on the screen, for
reference: the two best Gaussian configs and the best EB config on the defaults.
The best Student-t config is scored on test-like runs only, so it cannot be
chosen.

**Statistics.** Paired ALC differences against the legacy Predictor, written "±
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

The legacy Predictor is the worst candidate here, worse than answering from
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
| hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 (recommended, below; not what ships) | 0.1578 (-0.0461) | 0.1655 ± 0.0031 | -0.0418 ± 0.0024 / 0.0047 / 0.0042 | +0.0008 ± 0.0012 / 0.0025 / 0.0022 | +0.0007 ± 0.0012 / 0.0021 / 0.0019 |
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
leads the Predictor fix on the test-like runs: the recommended config minus the
Predictor fix is -0.0040 ± 0.0009 / 0.0016 / 0.0015 on the confirmation half and
-0.0046 ± 0.0006 / 0.0014 / 0.0012 on all 200 runs, and -0.0019 and -0.0021
(cluster SE 0.0010, 0.0009) on the two public weightings.

**Which one to ship.** The selection rule's argmax is mu0 -3.5. The next config,
mu0 -3.0 at the same attr_scale and width, is 0.0004 behind it on the selection
half (paired, ± 0.0002 / 0.0004 / 0.0003: one cluster SE), level with it on the
confirmation half (+0.0000 ± 0.0002 / 0.0004 / 0.0004) and on all 200 runs
(+0.0002, cluster SE 0.0004). It is 0.0017 cheaper on both public weightings
(paired cluster SE 0.0003), so it passes the guard with a margin of 0.0022
instead of 0.0005 to 0.0007. This recommendation takes mu0 -3.0, a tie-break on
the guard's margin that the rule as stated does not make. `SHIP` in the script
names it, and the summary's `ship` block pairs it with the alternatives. The EB
variant on mu0 -3.0 at attr_scale 0.5 ties it as well: the recommended config minus
it is -0.0002 (cluster SE 0.0006) on the confirmation half and +0.0003 and
+0.0001 (0.0005) on public runs. It adapts where a fixed prior cannot (below),
but it is about 70 lines of experiment code not in the library, with a slower
worst call. The fixed configuration changes three Hyper fields and nothing else.

**Optimism.** The recommended config gains 0.0461 on the selection half and 0.0418
on the confirmation half; the rule's argmax 0.0465 and 0.0418. Most of that
0.004 is the halves differing, not the selection. Every configuration that moves
the level loses about a tenth of its gain between the halves, chosen or not:
0.0033 for the guarded Predictor fix, 0.0039 for hier G mu0 -3.0 at attr_scale
0.5, 0.0045 to 0.0048 for the unconstrained bests of every family. The
Predictor's own ALC moves from 0.2039 to 0.2073. Leaving one parent out of the
200 runs, the recommended config's gain ranges from 0.032 (multi_swebench out) to
0.052 (matharena out).

**By budget**, the recommended config against the Predictor on all 200 test-like
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
| hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25 (recommended) | 0.210 | 0.174 | 0.167 | 0.155 | 0.162 | 0.149 | 0.1673 | 0.1685 |

So the recommended config would have scored about 0.167 on the run that gave the
Predictor 0.2113: below the organisers' 0.1801 and well above the best entry's
0.1172. That is an estimate from one run, not a measurement. Nothing was fitted
to the feedback here beyond what `testlike.Regime`'s defaults already were, and
no prediction was shaped to probe hidden labels. (The audit below then read the
run per pair and chose between two guarded configurations partly on that
reading. That is a use of the feedback beyond the regime's defaults, though
still of one global setting; "Formative feedback, runs 1 and 2" lists every
use. The per-pair estimate that replaced this 0.167 is 0.178 for the
recommended config and 0.177 to 0.183 for the shipped one.)

### Sensitivity to the regime

Minus the Predictor on 100 runs of each regime (± run / cluster / stratified SE;
the Predictor's ALC in the header):

| configuration | level_mean -1.2 (0.2117) | level_mean -2.0 (0.2005) | groups merged at random, no strata (0.2129) | no date shift (0.1729) |
|---|---|---|---|---|
| hier G mu0 -3.0, attr_scale 0.25 (recommended) | -0.0388 ± 0.0021 / 0.0044 / 0.0039 | -0.0524 ± 0.0022 / 0.0044 / 0.0039 | -0.0467 ± 0.0017 / 0.0049 / 0.0038 | -0.0139 ± 0.0014 / 0.0028 / 0.0026 |
| hier G mu0 -3.5, attr_scale 0.25 | -0.0387 | -0.0533 | -0.0471 | -0.0135 |
| hier EB-cs, tau 2, on G mu0 -3.0, attr_scale 0.5 | -0.0380 | -0.0518 | -0.0452 | -0.0144 |
| Predictor fix off -1.5, sc 1, va 3 | -0.0353 | -0.0446 | -0.0398 | -0.0137 |
| hier EB-cs, tau 2, on the fitted defaults | -0.0224 | -0.0310 | -0.0246 | -0.0108 |
| smoothed Beta(2,2) | -0.0228 | -0.0281 | -0.0260 | +0.0077 |
| hier, fitted defaults | -0.0115 | -0.0143 | -0.0080 | -0.0078 |

The recommended config's gain holds wherever the hidden level sits within the range
one feedback run allows, and it grows as that level falls. It holds with item
structure closer to the public benchmarks' (groups merged at random). Without
the date shift the attribute priors are not inflated, and the Predictor's own
ALC falls to 0.1729. Even there every moved prior still gains 0.013 to 0.014,
nearly twice what hier's defaults gain, because the pairs are still low. The
date shift is what reproduces the feedback's B0 optimism in this regime. For the
legacy Predictor it is an offset on its prior's ability. hier's prior reads
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
the way from its first labels (one per pair) and orders the date-shifted
regimes by their levels; the unshifted regime (-1.33) ends above them all.
Where dates are shifted it moves further than the pair logits alone
would say, because it centres the level at attribute score 0 and the inflated
attribute standings have to be taken back. On the shifted base (mu0 -3.0) it
moves back toward the public centre on public runs (-3.0 at B0, -2.3 at B1, -1.6
at B31) and further down on test-like ones (-3.6 at B1, -4.0 at B31). The
estimated sigma_mu stays between 1.7 and 2.7. What EB adds is at B1 and B3. At
B0 it is its base, and the level fixed at B0 decides most of the test-like gain.
That is why it ties the fixed config instead of beating it. It is also why EB on
the fitted defaults recovers only 0.013 of the 0.031 that the recommended config
adds over hier's defaults on all 200 runs.

### fit_ab diverges when the prior is far from the labels

`fitting.fit_ab` (the legacy Predictor's (a, b) fit) takes full Newton steps
from the prior mean. When that mean sits far from the pair's labels, the first
step overshoots into the flat tail of the logistic. The clipped weights vanish
there and it stops. One test-like pair at 27 of 31, prior mean -1.88, va 3, came
back at a = -13.9 with the prior's variance, a prediction of 0.000 on a pair at
0.83. On 3,000 random small problems (prior mean N(0, 1.5^2), up to 31 labels)
it missed the MAP on 173. On those its log posterior was at least 20 nats below
the maximum (246 at the median), and its prediction at z = 0 was off by 0.69 at
the median and 0.89 at most. `robust_fit_ab` in the script finds the same MAP by
Newton with step halving, and agrees with fit_ab to 4e-16 wherever fit_ab
converges (summary `fit_ab`). The legacy Predictor diverges on 5 of 1,598
test-like pair appearances, costing 0.023 to 0.080 of the pair's ALC (0.0001
overall). It diverges on 8 of 3,206 in the sensitivity regimes, none of them
with the subjects' own dates, and on none of 1,748 public ones. A moved prior
diverges far more often: off -2.5 costs +0.033 on public runs with fit_ab and
+0.017 to +0.019 with the robust fit. Any Predictor level fix needs the fit
fixed first. This is a library bug, reported here and not fixed
(`paiec/fitting.py` is not this experiment's to change).

### Latency

Evaluation calls of the recommended config, six processes side by side on a shared
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

`python experiments/level_audit.py` (the default stages `feedback`, `analyse`
and `mild`, about 5 seconds; `--stage extra --jobs 1`, 1,414 s on one process
plus a first pass that was stopped and resumed, peak 1.9 GB). Every number
below is in `results/level_audit.json`. The script is the audit's own code,
moved from session scratch (step 2b, 2026-09-26) with only its data loading
changed, and it reproduces every number the audit recorded. Where the audit
scored the shipped config on fresh runs, `--stage mild` reads the same runs
from the shipped arm of `data/subject_side_rows` instead, after checking that
the two sources agree on the 140 runs (1,147 pairs) where both hold it (at
most 5.5e-6, the rows' rounding).

The archive ships the milder guarded configuration, `mu0 -2.5, sigma_mu 2.5,
attr_scale 0.5` (`submission/model.py` LEVEL; MILD below), not the recommended
`mu0 -3.0, attr_scale 0.25` argued for above (AGGR below). PRED is the legacy
Predictor. The audit of this section changed four readings:

* The regime's level_mean of -1.6 reads every feedback pair's B31 through the
  lower root of p(1-p). For p6 and p8 (bench F), B0 is already close to B31
  under an optimistic prior, and 70% and 80% of their 40 nearest replica pairs
  sit above a rate of 0.5 (p9: 55%). So they are more likely high-rate pairs,
  and p9 is ambiguous. (The "about 0.78" first written here is the upper root
  of p(1-p) = B31, 0.787 and 0.774, not an estimate; the neighbours' mean rates
  are 0.62 and 0.66.) Reading each pair's level as its neighbours' mean logit
  moves the nine pairs' mean from -1.64 (sd 1.49, lower roots) to -1.02 (sd
  1.52); taking the upper roots for p6 and p8 gives -1.08 (sd 1.99; plain logit,
  `results/formative_feedback.json`). The hidden levels look spread both ways,
  not uniformly low. AGGR loses on high rates: against PRED on test-like
  runs 0 to 199, +0.014 on pairs at rates 0.5 to 0.7 and +0.070 at 0.7 and
  above (against -0.084 below 0.1); on public matharena, +0.0172 and +0.0150
  (benchmark-first, pair-uniform, runs 0 to 99). The "about 0.015 per pair on a
  matharena-like benchmark" first written here is the latter.
* The per-pair matched estimate on the feedback run is 0.1783 (K 40) and 0.1789
  (K 15) for AGGR, 0.1686 to 0.1830 across the pools that hold test-like runs,
  not 0.167. The shipped MILD's own estimate, on the pool where MILD, AGGR and
  PRED are all stored (4,955 appearances), is 0.1765 (K 15) and 0.1789 (K 40),
  against AGGR's 0.1770 and 0.1797 there. "Formative feedback, runs 1 and 2"
  gives the same estimator on the shipped model's own rows (0.1774, 0.1779) and
  on the audit's screen pool (0.1812, 0.1830). Quote the expected gain as about
  0.02 to 0.04, conditional on the hidden levels.
* The "held-out" half redraws runs from the same catalogue of pairs, so it
  measures redrawing, not generalisation to unseen pairs or parents: it repeats
  82.2% of the selection half's pairs, 99.5% of its (parent, subject) clusters
  and all 35 pseudo-benchmarks. Nested leave-one-parent-out selection does not
  carry the chosen level to matharena: with matharena held out it picks mu0
  -3.5, which costs +0.0073 there on the confirmation half (AGGR +0.0026; the
  best guarded config in hindsight -0.0130).
* The two configs sit on a flat plateau (MILD minus AGGR, ± run / cluster /
  stratified SE):

  | regime, runs | MILD minus AGGR | MILD minus PRED |
  |---|---|---|
  | test-like 0-99 (selection half) | +0.0030 ± 0.0005 / 0.0009 / 0.0007 | -0.0431 ± 0.0019 / 0.0032 / 0.0028 |
  | test-like 100-199 (confirmation half) | +0.0022 ± 0.0005 / 0.0011 / 0.0009 | -0.0396 ± 0.0019 / 0.0037 / 0.0034 |
  | public R1 benchmark-first 0-99 | -0.0023 ± 0.0002 / 0.0005 / 0.0004 | -0.0015 ± 0.0010 / 0.0021 / 0.0018 |
  | public R1 pair-uniform 0-99 | -0.0024 ± 0.0002 / 0.0004 / 0.0004 | -0.0018 ± 0.0010 / 0.0017 / 0.0016 |
  | no date shift 0-29 | -0.0005 ± 0.0005 / 0.0005 / 0.0004 | -0.0141 ± 0.0022 / 0.0026 / 0.0024 |
  | mix/whole 0-99 (not scored by the audit) | +0.0031 ± 0.0004 / 0.0010 / 0.0008 | -0.0436 ± 0.0014 / 0.0039 / 0.0031 |

  MILD gives up 0.0030 and 0.0022 in the tuned regime, gains 0.0023 and 0.0024
  on public runs, and beats PRED there by point estimates within about one
  cluster SE (0.7 and 1.02 cluster SEs). Its least favourable measured regime mean against PRED is -0.0015 (public
  benchmark-first); AGGR's is +0.0008 there. With a main effect near 0.03 and
  stakes of at most 0.003 either way, robustness decides.

**Two regimes nearer the per-pair reading, re-measured** (`--stage extra`:
seed 5, 40 runs each, the date shift kept, the library at 4d2cc4f with the
corrected floor and the floored-fit fix, so these are new measurements, not
the audit's). The extra rows were scored under script digest 40ed9e5b. The
file has changed twice since: the `analyse` stage's nested selections now
choose on unrounded means, as the audit's code did, and the docstring gives
the recorded run times. Neither change touches the extra stage, and the
library digest (8b96ccb4) is the same (`experiments/script_revisions.py`,
which rebuilds 40ed9e5b from the committed file and a log of the edits;
`results/script_revisions.json`).
Minus PRED, ± run / cluster / stratified SE:

| regime (realised pair logit; PRED ALC) | AGGR | mu0 -3.5 | MILD (shipped) | EB on mu0 -3.0, attr_scale 0.5 | Predictor fix off -1.5 | smoothed | MILD minus AGGR (run / cluster SE) |
|---|---|---|---|---|---|---|---|
| level_mean -0.8 (-0.77; 0.2153) | -0.0283 ± 0.0033 / 0.0049 / 0.0044 | -0.0273 | -0.0286 ± 0.0026 / 0.0039 / 0.0035 | -0.0286 | -0.0275 | -0.0183 | -0.0003 ± 0.0008 / 0.0011 |
| no level tilt (-0.59; 0.1958) | -0.0196 ± 0.0033 / 0.0063 / 0.0055 | -0.0177 | -0.0222 ± 0.0025 / 0.0049 / 0.0043 | -0.0213 | -0.0232 | -0.0119 | -0.0026 ± 0.0008 / 0.0014 |

So "no worse measured worst case" holds on every regime measured, and the
shipped config's gain shrinks as the hidden level rises toward the public
centre: 0.029 at a realised level of -0.77 and 0.022 at -0.59, against 0.042
at the tuned regime's -1.29. Without a level tilt the Predictor with a moved
prior does as well as any hier config. The level_mean -2.0 regime, where AGGR
should do best, has never been scored for MILD, and nor has level_mean -1.2.
(Regimes set to the feedback's level readings were scored later, on fresh
seeds and under a rule fixed in advance. AGGR scores worse than MILD in every
one of them but the tuned regime, by +0.0005 to +0.0033: "Regime sensitivity
at the feedback's reading".)

attr_scale below 1 is a bet that the attribute prior is inflated for hidden
subjects (the regime reproduces the feedback's B0 optimism with a synthetic date
shift). The audit asked for the next feedback to be read per pair, pooled with
the first run's nine pairs, before refitting the level distribution once. That
reading was made after run 2, under a decision rule written and hash-locked
before it was computed (though after both feedback tables had been seen), and
left LEVEL unchanged ("Formative feedback, runs 1 and 2").

### Verdict: ship hier with the level moved down

(The level in this verdict was superseded before the archive was built: it
ships mu0 -2.5, sigma_mu 2.5, attr_scale 0.5, "What actually shipped, after the
audit". The numbers below are the recommended config's. The shipped config's
own are in "Shipped configuration, confirmed".)

Ship `paiec.hier.HierPredictor(prior, replace(hyper, mu0=-3.0, sigma_mu=2.5,
attr_scale=0.25))`, where `prior, hyper = prior.build(pairs, ())` is fitted on
every eligible public pair (`Hyper()`'s defaults are that fit, rounded). The
level is Gaussian (nu_mu 0), all Flags stay at their defaults, and the rest of
the Hyper stays as fitted (sigma_theta 0.1, sigma_delta 2.382, sigma_attr 1.018,
sigma_d 2.671, sigma_g 1.542, slip 0.01, guess 0.5). The evaluation fitted those
fields leaving each parent out, so the shipped fit on all five is close to what
was measured but not identical to it. Against the legacy Predictor this gives:

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
  The public guard holds it (in most benchmark-first runs), and the recommended
  config passes with it in, but its own difference on that benchmark was not
  broken out here.
* The regime was tuned to the feedback's B0 and B1 on the legacy Predictor, so
  the size of the gain rests on that tuning. The level knob is not identified,
  and the sensitivities bound it. The gain comes mostly from B0 and B1 (0.031 of
  0.044), which the next formative feedback will show directly.
* sigma_mu 2.5 and attr_scale 0.25 sit on the edges of what was scored: no wider
  level prior and no smaller attr_scale were tried. (sigma_mu 3.5 and 5, at
  the shipped mu0 and attr_scale, were scored later and do not do better:
  "Regime sensitivity at the feedback's reading".)
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
| public, benchmark-first | 0.1706 | 0.0768 | 0.0938 ± 0.0045, 55% | -0.0025 | 0.00002 (0.02%); from the primary 0.0005 (0.5%) | 0.0012 (1.3%; first recorded as 0.0013, 1.4%, from rounded inputs: `nested_within` B31 is -0.001248); from the primary 0.0012 (1.3%) |
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

## Neural embeddings do not carry difficulty to an unseen benchmark

`python experiments/emb_transfer.py` (results in `results/emb_transfer.json`;
needs the Qwen3-Embedding-0.6B features from `experiments/llm_features.py`)

Target: Rasch difficulty per item, subject ability divided out, standardised
within benchmark, on the four multi-subject benchmarks (swe_rebench had no
embeddings yet). Hyperparameters nested throughout.

| Pearson r (matharena / multi_swebench / real_webagents / researchcodebench) | |
|---|---|
| leave-one-benchmark-out, embedding ridge (benchmark-centred) | -0.16 / -0.03 / -0.23 / -0.02 |
| leave-one-benchmark-out, embedding kNN | -0.08 / -0.01 / -0.02 / +0.16 |
| leave-one-benchmark-out, TF-IDF+SVD ridge, same items | -0.18 / -0.05 / -0.06 / -0.25 |
| within benchmark, 5-fold, embedding ridge | 0.66 / 0.27 / 0.38 / 0.51 |
| within benchmark, item_features group mean alone | 0.57 / 0.12 / 0.41 / 0.52 |
| within benchmark, whole groups left out | 0.40 / 0.16 / -0.02 / -0.09 |

Transfer to an unseen benchmark is null to negative, no better than TF-IDF,
and R² as predicted is at most 0 everywhere. With the slope carried over from
the other three benchmarks the covariate moves pair Brier by -0.0005 on
average, and the carried slope has the wrong sign. Inside a benchmark the
embedding mostly identifies the item_features group, which hier already
learns from labels, and it already fails across groups on two benchmarks.
Hidden runs give about one pair and at most 31 labels per benchmark, so the
within-benchmark signal cannot be learned there either. The embeddings are kept
for analysis only.

## Acceptance harness

`python experiments/harness.py --stage collect --jobs 1`, then `--stage verify
--legacy data/harness_rows_legacy`, `--stage table --resume` and `--stage show`
(39 minutes for the collection on one process beside a language-model job, 1
minute for verification, 44 minutes for the table, almost all of it in the
first of three passes. Every number
below is in `results/harness_thresholds.json`. The rows are 43 MB in
`data/harness_rows`, gitignored. Each run is stamped with the digest of the
library it was collected with, here 3f75a549673aae6a, which holds the corrected
multiple-choice floor and the floored-fit fix. `tests/test_harness.py` pins the
arithmetic on synthetic rows.)

Every item-side idea is one number per item: difficulty fields, format,
length, language-model ratings, attempt signals, an encoder. The harness scores
any such number the same way, against the hier that ships, and holds the one
table of thresholds they are all read against. A covariate enters as a logit
offset on hier's prediction, capped at ±4 logits. The offset is added to
hier's output; hier is not refitted with the covariate inside it, which is what
lets one replay of hier serve every covariate. The offset has four forms:

* **transferred**: one slope per budget, fitted on the other parents' test-like
  and mix/whole pairs, with x centred on every label on the target's benchmark;
* **per-pair**: a MAP slope from the pair's own labels, with prior sd s per
  within-benchmark sd of x (s from 0.1 to 2). One own label carries no slope
  after centring, so it acts from B3;
* **hybrid**: the per-pair slope with its prior centred on the transferred one;
* **B0 term** (added in the audit): an uncentred transferred offset at B0 only,
  beta_0 (x - c0), with c0 the training parents' mean of x.

The centred forms act from B1 or from B7. Nested leave-one-parent-out selection
over the four multi-subject parents picks a configuration, or "off", for each
held-out parent. The runs are 300 test-like, 150 mix/whole, and 100 + 100 public
R1 (benchmark-first, pair-uniform) as the guard. Differences are paired ALC
against the shipped hier per pair appearance, weighted 1 / run size.

The gate is the plan's rule. Nested selection must be on, and must act, in at
least 3 of the 4 folds. The test-like difference must be at most -0.002, the
mix/whole difference of the same sign, no held-out parent above +0.002, and
neither public weighting above +0.001.

"The gate, tightened (P1.12)" adds the pass probability at each r, the gate
read on mix/whole, when the gate was first committed, the honest difficulty's
reliability, and every quoted correlation on the gate's within-pair scale.

### Acceptance and the honest oracle

The item oracle is each item's Rasch difficulty. The in-sample oracle is fitted
on every subject of its parent, evaluated responses included; that is what the
heads study scored. The honest oracle is fitted on the other four of five
subject folds, so a target's own responses never enter its covariate.
Differences are written "± run SE / cluster SE (selection-aware cluster SE) /
stratified SE".

| oracle, line | test-like | mix/whole | R1 benchmark-first | R1 pair-uniform | folds on |
|---|---|---|---|---|---|
| in-sample, transferred from B1 (forced) | -0.0439 ± 0.0012 / 0.0036 / 0.0036 | -0.0558 | -0.0609 | -0.0688 | |
| in-sample, per-pair nested | -0.0293 ± 0.0008 / 0.0025 (0.0025) / 0.0025 | -0.0387 | -0.0413 | -0.0484 | 4/4 |
| in-sample, B0 term nested | -0.0035 ± 0.0002 / 0.0010 (0.0010) / 0.0008 | -0.0023 | -0.0070 | -0.0082 | 4/4 |
| honest, transferred nested | -0.0360 ± 0.0010 / 0.0030 (0.0030) / 0.0029 | -0.0484 | -0.0539 | -0.0619 | 4/4 |
| honest, transferred from B7 (forced) | -0.0206 ± 0.0005 / 0.0016 / 0.0016 | -0.0281 | -0.0293 | -0.0338 | |
| honest, per-pair nested | -0.0223 ± 0.0007 / 0.0020 (0.0020) / 0.0020 | -0.0316 | -0.0358 | -0.0428 | 4/4 |
| honest, hybrid nested | -0.0359 ± 0.0010 / 0.0031 (0.0031) / 0.0030 | -0.0480 | -0.0545 | -0.0648 | 4/4 |
| honest, B0 term nested | -0.0027 ± 0.0002 / 0.0009 (0.0009) / 0.0007 | -0.0015 | -0.0062 | -0.0073 | 4/4 |
| honest by model name, transferred nested | -0.0362 ± 0.0010 / 0.0030 (0.0030) / 0.0029 | -0.0485 | -0.0534 | -0.0611 | 4/4 |
| honest by model name, per-pair nested | -0.0223 ± 0.0007 / 0.0020 (0.0020) / 0.0020 | -0.0323 | -0.0354 | -0.0418 | 4/4 |

* **Acceptance passes.** The in-sample oracle, transferred and forced from B1,
  gives -0.0439 test-like and -0.0558 mix/whole, against the heads study's
  -0.0437 and -0.0557 (tolerance 0.003). The target now has a committed
  source: `experiments/heads_eval.py` reproduces it as -0.04371 and -0.05566
  (`results/heads_eval.json`, `rows.legacy.configs.lopo_oracle`; "Meta-learned
  heads on frozen embeddings"). The legacy rows, with the old floor and
  solver, gave -0.0440 and -0.0558.
* **About 18% of the in-sample oracle is leak.** The honest oracle gives
  -0.0360 against -0.0439 (17.9%).
* **Near-duplicate subjects do not leak.** Folds by canonical model name put
  one model under several harnesses or efforts into one fold (75 of
  multi_swebench's 82 pairs belong to 13 names). They give -0.0362, against
  -0.0360 with folds by subject_id.
* **Per-pair keeps 62% of the honest oracle's transferred gain** (-0.0223).
  Starting the transferred slope at B7 keeps 57% (-0.0206). The hybrid adds
  nothing (-0.0359).
* **The B0 term gains -0.0027 on the honest oracle, and fails the gate.** That
  is 0.027 of Brier at B0, which carries a tenth of the weight. With
  multi_swebench held out it loses +0.0038. The oracle's raw difficulties sit
  higher on multi_swebench than on the parents its reference c0 came from, so
  an uncentred term also moves a benchmark's level, here the wrong way.

### The gate table

The degraded oracles are x = r z + sqrt(1 - r²) e, where z is the honest
difficulty standardised within its parent and e is standard normal per item.
r is the correlation over a parent's items. The within-pair r in brackets is
the mean correlation over a test-like pair's evaluated items; it is lower
because a pseudo-benchmark spans less difficulty. Each cell averages 8 noise
draws, and test-like differences are given with the pair-cluster SE and the sd
over draws. "Draws passing" counts the draws that pass the gate on their own. A
real covariate is one draw, so that count, not the averaged line, is the
chance that a covariate at this r passes. With 8 draws it moves in steps of
1/8, so a pass rate is indicative.

| r (within pair) | transferred nested: test-like (cluster SE, draw sd) | mix/whole | worst parent | R1 b / p | on | draws passing | per-pair nested: test-like (cluster SE, draw sd) | worst parent | draws passing | B0 term nested: test-like (cluster SE) | draws passing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (0.00) | -0.00002 (0.00001, 0.00006) | +0.00001 | +0.00003 | -0.00001 / +0.00000 | 0.6 | 0/8 | -0.00000 (0.00000, 0.00000) | +0.00000 | 0/8 | +0.00000 (0.00000) | 0/8 |
| 0.1 (0.08) | -0.00010 (0.00005, 0.00043) | -0.00027 | +0.00002 | -0.00034 / -0.00044 | 1.9 | 0/8 | -0.00000 (0.00000, 0.00001) | +0.00000 | 0/8 | -0.00000 (0.00001) | 0/8 |
| 0.2 (0.16) | -0.00083 (0.00015, 0.00086) | -0.00161 | -0.00071 | -0.00191 / -0.00245 | 3.5 | 1/8 | -0.00010 (0.00002, 0.00007) | -0.00005 | 0/8 | -0.00006 (0.00004) | 0/8 |
| 0.3 (0.25) | -0.00255 (0.00032, 0.00068) | -0.00457 | -0.00211 | -0.00517 / -0.00639 | 4 | 6/8 | -0.00032 (0.00007, 0.00009) | -0.00010 | 0/8 | -0.00020 (0.00010) | 0/8 |
| 0.4 (0.33) | -0.00462 (0.00052, 0.00081) | -0.00805 | -0.00390 | -0.00906 / -0.01109 | 4 | 7/8 | -0.00136 (0.00017, 0.00015) | -0.00104 | 0/8 | -0.00037 (0.00017) | 0/8 |
| 0.5 (0.42) | -0.00738 (0.00077, 0.00090) | -0.01246 | -0.00637 | -0.01399 / -0.01699 | 4 | 8/8 | -0.00255 (0.00026, 0.00011) | -0.00227 | 8/8 | -0.00056 (0.00026) | 0/8 |
| 0.7 (0.61) | -0.01528 (0.00143, 0.00091) | -0.02392 | -0.01402 | -0.02693 / -0.03213 | 4 | 8/8 | -0.00768 (0.00077, 0.00036) | -0.00672 | 8/8 | -0.00101 (0.00050) | 0/8 |

Fixed configurations with no selection, test-like, with the draws that would
pass if the term were switched on:

| r | transferred from B1 | transferred from B7 | per-pair s=0.5 from B1 | per-pair s=0.5 from B7 | hybrid s=0.5 from B1 | B0 term |
|---|---|---|---|---|---|---|
| 0 | +0.00009 (0/8) | +0.00004 (0/8) | +0.00091 (0/8) | +0.00073 (0/8) | +0.00097 (0/8) | +0.00001 (0/8) |
| 0.1 | -0.00023 (0/8) | -0.00014 (0/8) | +0.00074 (0/8) | +0.00059 (0/8) | +0.00062 (0/8) | -0.00002 (0/8) |
| 0.2 | -0.00110 (1/8) | -0.00064 (0/8) | +0.00031 (0/8) | +0.00022 (0/8) | -0.00032 (0/8) | -0.00010 (0/8) |
| 0.3 | -0.00255 (6/8) | -0.00148 (1/8) | -0.00039 (0/8) | -0.00038 (0/8) | -0.00187 (2/8) | -0.00022 (0/8) |
| 0.4 | -0.00462 (7/8) | -0.00269 (8/8) | -0.00136 (0/8) | -0.00121 (0/8) | -0.00405 (7/8) | -0.00037 (0/8) |
| 0.5 | -0.00738 (8/8) | -0.00431 (8/8) | -0.00259 (8/8) | -0.00227 (8/8) | -0.00692 (8/8) | -0.00056 (0/8) |
| 0.7 | -0.01528 (8/8) | -0.00898 (8/8) | -0.00583 (8/8) | -0.00506 (8/8) | -0.01496 (8/8) | -0.00101 (0/8) |

* **Transferred slope: r ≈ 0.3-0.4 against honest difficulty** (within-pair
  0.25-0.33). It passes on 1 of 8 draws at r = 0.2, 6 of 8 at r = 0.3, 7 of 8
  at r = 0.4 and every draw from r = 0.5. At r = 0.3 the draw sd (0.0007) is
  twice the cluster SE (0.0003), so a single covariate near the line reads as
  much noise from its draw as from its runs.
* **Per-pair slope: r ≈ 0.5** (within-pair 0.42). It passes on no draw at
  r = 0.4 and on every draw at r = 0.5. It keeps 13% of the transferred gain at
  r = 0.3, 29% at 0.4, 35% at 0.5 and 50% at 0.7. The plan assumed 25-30%.
* **The B0 term never passes.** It gives -0.0002 at r = 0.3 and -0.0010 at
  r = 0.7. Spreading items within a pair at B0 is worth little: B0 carries a
  tenth of the weight.
* **An uninformative covariate stays off.** At r = 0 the nested lines are
  within ±0.00002, with the term on in 0.4 to 0.6 of 4 folds on average. Forced
  on, the per-pair slope with s = 0.5 costs +0.0009 from B1 and +0.0007 from
  B7. That is above the plan's cap of +0.0005 for an uninformative covariate,
  so s = 0.5 must not be switched on blind.
* **Starting at B7 keeps about 58% of the transferred gain** (-0.0015 against
  -0.0026 at r = 0.3), because the term already acts at B1 and B3. At r = 0.3
  the transferred slope gains 0.0026 to 0.0030 of Brier at every budget from B1
  to B31; the per-pair slope grows from B3 to 0.0012 at B31.
* **The public guard never binds for difficulty-like covariates.** Public R1
  gains are 1.8 to 2.5 times the test-like ones.
* **The hybrid adds nothing.** Its nested line is within 0.0002 of the
  transferred one at every r, and never better.

The in-sample table (5 draws per cell; the heads study's curve) sits 16 to 21%
above the honest one. Its transferred line gives -0.0030 at r = 0.3 (5 of 5
draws pass) and -0.0089 at r = 0.5; the heads study's single forced draws gave
-0.0024 and -0.0080 (-0.00239 and -0.00799, `results/heads_eval.json`,
`lopo_oracle_r0.3` and `lopo_oracle_r0.5`). Its per-pair line passes from
r = 0.5, as on the honest base.

### What changed in the audit

* **Input convention.** `--stage eval` now standardises x within each public
  benchmark by default, over the items that carry it, with
  `experiments/itemcov_eval.py`'s rule. x is 0 (the benchmark's mean) where an
  item lacks it, or where fewer than 10 items of the benchmark are off x's most
  common value, and it is clipped at ±3 sd. Benchmark-level offsets in x
  therefore cannot act. `--raw` reads x as given. The B0 term reads raw x in
  both cases, because at B0 a run-time predictor sees one item and has no
  benchmark to standardise over.
* **The per-pair unit.** `Engine.sd`, the unit of s, is now the pooled
  within-benchmark sd of x, taken over the training parents on which x varies.
  Before, it was the sd pooled across parents. That gave format_score (absent
  on three parents) a unit of 0.047, so s = 0.5 meant a prior sd of 10.7 per
  within-benchmark sd, and gave log_length (with benchmark offsets) a unit of
  2.36. A test pins that a covariate with benchmark offsets, absent on one
  parent, gets the same per-pair prior and the same per-pair lines whatever the
  offsets. On the degraded oracles, which are standardised within parent, the
  two units agree to within 3%. `HARNESS_SCALE=pooled` restores the old unit.
* **Selection.** A configuration is switched on only if its inner mean is below
  minus one inner SE, the linearised pair-cluster SE of the three inner
  parents' means. A fold counts as on only if its choice moves some prediction
  on the held-out parent. A covariate that is constant there used to count.
* **Standard errors.** The run, cluster and stratified SEs hold the selection
  fixed, so for a nested line they are lower bounds. The eval and oracle lines
  now also carry a cluster bootstrap that redoes the inner selection in every
  resample, holding the fitted slopes fixed (`sel_cluster_se`). Where no
  resample changes a choice, as on the oracles, it equals the fixed SE. Where
  choices move, it is wider. It covers the test-like regime only, so the
  mix/whole and public-guard SEs of a nested line remain lower bounds. The
  table's draws are scored without it; draw 0 of four honest rows is scored
  with it (`selection_se` in the results file):

  | r | transferred nested: test-like | cluster SE | selection-aware | per-pair nested: test-like | cluster SE | selection-aware |
  |---|---|---|---|---|---|---|
  | 0 | 0 (0 folds on) | 0 | 0.00005 | 0 (0 folds on) | 0 | 0.00001 |
  | 0.2 | -0.00025 (3 folds on) | 0.00017 | 0.00042 | +0.00001 | 0.00004 | 0.00007 |
  | 0.3 | -0.00178 | 0.00047 | 0.00066 | -0.00019 | 0.00017 | 0.00015 |
  | 0.4 | -0.00362 | 0.00069 | 0.00075 | -0.00126 | 0.00025 | 0.00038 |

  Draw 0 at r = 0.3 is one of the two draws that fail the gate.
* **Pass rates.** Each cell of the table reports how many noise draws pass
  (above). `thresholds` in the results file also gives, per line, the smallest
  r at which most draws pass and the smallest at which every draw does.
* **The cap for an uninformative covariate** in the harness docstring is now
  the plan's +0.0005, not +0.001.

### Caveats

* **The centred forms are blind at B0 by construction.** A centred offset is 0
  when nothing is labeled. The B0 term is the one path that acts there. On the
  degraded oracles, which are standardised within parent, every parent has
  mean 0, so the term reads only differences within a parent. That includes a
  pseudo-benchmark's level within its parent: the honest oracle's B0 term gains
  -0.0042 per appearance on difficulty strata against -0.0010 on item_features
  groups. A real covariate on an absolute scale would also carry levels between
  benchmarks. Those can help or hurt (multi_swebench above), and nothing public
  calibrates them. A scratch check on the real rows shows both sides
  (indicative: no script in `experiments/` reproduces it yet). Log
  length with 5 added on multi_swebench gives the same centred lines, bit for
  bit, as log length itself, so benchmark offsets cannot act there. Its B0
  term, which reads raw x, costs +0.0032 test-like. Nested selection switched
  it on with real_webagents held out; its slope, fitted where multi_swebench's
  offset dominates x, then shifts real_webagents' level (+0.0168 there). The
  gate fails it. On log length itself the B0 term gives +0.00001.
* **The offset is not refitted inside hier.** A covariate cannot move hier's
  posterior for the pair's level or for other items.
* **x is standardised over the whole public benchmark.** A run-time predictor
  sees only the benchmark's visible items.
* **u_j leaves label j in.** The per-pair slope's base u_j is hier's prediction
  for item j as if unlabeled, with the pair's level fitted on every label,
  j's included. The audit recomputed u_j with j removed from `labeled` on 80
  test-like runs of the legacy rows. Leaving j out did not raise the per-pair
  gain: the honest oracle nested gave -0.02100 leave-in against -0.02036
  leave-out; r = 0.5, s = 0.5 gave -0.00230 against -0.00235; r = 0.3, s = 0.5
  from B1 gave -0.00025 against -0.00004. On that evidence the low per-pair
  retention at r ≤ 0.3 is not an artefact of leaving y_j in. **This check is
  not reproducible from the repository**: its scripts were session scratch,
  never moved to `experiments/`, and they read the legacy rows through the
  harness's interface as it was before its audit. It was not re-run on the new
  rows, and no results file holds its numbers. Read them as indicative.
* **Four parents.** The SE across the four parents' means is the only one that
  sees variation between benchmarks.

### Provenance

* **Re-collection.** The rows were re-collected with the current library
  (3f75a549673aae6a, the working tree on f7e7d87): 650 runs, 0 hier failures
  and 0 unconverged fits. `--stage verify` re-collects five runs (test-like 0
  and 17, mix/whole 3, benchmark-first 0, pair-uniform 5) and matches the
  stored rows exactly, with BLAS pinned to one thread.
* **Against the legacy rows.** The legacy rows (library 0c05d35ecc363e5b, from
  bd0be67, before the floor and solver fixes) are kept in
  `data/harness_rows_legacy`. The new rows differ on 157 of 300 test-like runs,
  74 of 150 mix/whole, 81 of 100 benchmark-first and 97 of 100 pair-uniform
  runs, by up to 0.25. Every run that differs holds a matharena pair; the change
  reaches that run's other pairs through hier's joint fit.
* **Against the heads study.** Its rows, from session scratch, were computed
  with the old library and are byte-identical to `data/harness_rows_legacy`
  (0 of 450 runs differ; checked by `experiments/heads_eval.py`). The new rows
  differ from them on the same 157 test-like and 74 mix/whole runs, by up to
  0.15. The unlabeled-item probe still matches hier
  exactly on 784 items nobody labeled.
* **The table** ran in three passes. The first (script digest 4a5526b8f6a7b4c1)
  wrote the acceptance and both tables; the second (9b45355fb2edd4df, with
  `--resume`) kept them and added `selection_se`. The code the first pass ran
  is unchanged in the second. A last `--resume` with the final script
  (77fd651cc5e2a00f, which differs only in docstrings) kept every part and
  recomputed nothing; `meta.script_digests` lists all three.
* **Other consumers of the rows.** `experiments/itemcov_eval.py` and
  `experiments/mcq_floor.py` read `data/harness_rows` by default. Their stored
  results (`results/itemcov_eval.json`, `results/mcq_floor.json`) were computed
  on the legacy rows. `mcq_floor.py`'s check against the stored rows replays the
  old floor, so it needs `--harness-rows data/harness_rows_legacy`.
  `itemcov_eval.py --scale train` reproduces its `harness_train_scale` numbers
  only on the legacy rows with `HARNESS_SCALE=pooled`. The commands in "Item
  covariates with a known sign" and "The multiple-choice floor, corrected" now
  carry these flags.

## Item covariates with a known sign

`python experiments/itemcov_eval.py --stage signs`, then `--stage harness
--rows data/harness_rows_legacy`, `HARNESS_SCALE=pooled python
experiments/itemcov_eval.py --stage harness --scale train --rows
data/harness_rows_legacy`, `--stage inventory` and `--stage show` (11 minutes
on one process, under 0.9 GB, beside other jobs. Every number below is in
`results/itemcov_eval.json`. The harness stage reads the stored rows of
`experiments/harness.py` and never recomputes hier. This section's numbers
were computed on the rows as they stood at bd0be67 (library 0c05d35ecc363e5b,
with the old multiple-choice floor and the solver before the floored-fit fix),
which are now kept in `data/harness_rows_legacy`, 650 runs. `data/harness_rows`
holds the re-collected rows, on which this section was not re-run ("Acceptance
harness", Provenance). `HARNESS_SCALE=pooled` restores the per-pair unit the
harness had before its audit, which is what `--scale train` measures. The
rows' provenance in the results file was copied from
`results/harness_thresholds.json` when that file still described the legacy
rows; a re-run now would copy the re-collected rows' provenance. Both harness
commands were re-run this way into a scratch copy of the results file: every
number this section quotes reproduces exactly. Only the harness's own nested
lines on its grid, kept in the results file for reference and not quoted here,
differ: image_ref's per-pair and hybrid lines on the within-benchmark scale,
and format_score's, log_length's and image_ref's on the train scale. The
harness's nested selection now needs a margin of one inner SE ("Acceptance
harness", What changed in the audit); this section's own nested selection
does not use it.)

`paiec/itemcov.py` reads covariates off the item dict that predict() receives.
It uses no labels and no model, and each covariate's sign is declared in
advance (higher = harder):

* **ordinal_difficulty**: an item_features key whose name says difficulty
  (difficulty, level, tier, stars, rating, elo, hardness, complexity, grade),
  with a number or an easy < medium < hard word as its value, or a work count
  (n_steps, num_files, ...). hier.select_keys drops all-numeric keys and treats
  words as unordered groups, so this is the reading hier does not do.
* **position**: log(1 + index) from a position-named key, which here means
  matharena's problem_idx. **position_within** is its percentile within the
  competition. It reads the other items' features, so it is a diagnostic
  only.
* **format_score**: proof +1, multiple choice -1, and short answer, code and
  free text 0, parsed from item_content.
* **log_length**: log(1 + characters).
* **stated_size**: log(1 + the first amount of work the text states). In
  researchcodebench that is the "Approximately 7 line(s) of code" of each TODO
  block.
* **image_ref**: the text points at an image it does not hold ("See image", a
  markdown image).

Every function is benchmark-agnostic and never raises (`tests/test_itemcov.py`).

### The sign check

This is the Spearman correlation with item difficulty on each unit:

* on the four multi-subject parents, honest Rasch difficulty: the mean over
  the harness's five subject folds, each fitted without that fold's subjects;
* on swe_rebench (one subject, 10.6 trials an item), -logit of the smoothed pass
  rate;
* on mmdocrag (fractional responses), -logit of the mean response. This is a
  directional sixth unit and is not counted.

The ± is a bootstrap SE over item_features groups (over items on swe_rebench
and mmdocrag). multi_swebench's resamples its 8 languages and is indicative.
The number in brackets is the correlation within groups, which
is what a covariate can add to hier's group effects. A unit counts when at
least 20 items carry the cue and at least 10 of them differ from its most
common value.

| covariate | matharena | multi_swebench | real_webagents | researchcodebench | swe_rebench | mmdocrag | declared sign / units |
|---|---|---|---|---|---|---|---|
| ordinal_difficulty | absent | absent | absent | absent | absent | absent | 0 / 0 |
| position | -0.03 ± 0.12 (+0.16) | absent | absent | absent | absent | absent | 0 / 1 |
| position_within | +0.12 ± 0.05 (+0.15) | absent | absent | absent | absent | absent | 1 / 1 |
| format_score | +0.19 ± 0.13 (+0.07) | constant | constant | 3 items off | 3 items off | 3 items off | 1 / 1 |
| log_length | +0.28 ± 0.08 (+0.06) | -0.02 ± 0.05 (+0.00) | +0.36 ± 0.08 (+0.29) | -0.10 ± 0.17 (-0.43) | +0.04 ± 0.01 | +0.05 ± 0.02 | 3 / 5 |
| stated_size | absent | absent | absent | +0.50 ± 0.06 (+0.43) | absent | absent | 1 / 1 |
| image_ref | -0.04 ± 0.09 | +0.05 ± 0.03 (+0.05) | absent | -0.08 ± 0.15 | +0.05 ± 0.01 | absent | 2 / 4 |

The plan allows a transferred slope only if two things hold on at least 4 of the
5 units: the declared sign, and agreement with the mean of the other units when
each unit is left out. No candidate passes, and most could not:

* No public item carries a difficulty-named field.
* position, format and stated size each vary on one benchmark only.
* image references vary on four benchmarks, with no consistent sign.
* log_length is the only covariate present everywhere, and its sign is not
  stable. It is positive on matharena (+0.28) and real_webagents (+0.36),
  negative on researchcodebench (-0.10 overall, -0.43 within paper; its
  prompts are whole papers), and null on the SWE sets. Within groups
  it has the declared sign on 4 of 5 units, but it agrees with the other
  units' mean on none, because researchcodebench's -0.43 outweighs the rest.

Formats parse cleanly on matharena: 231 proofs (IMO, USAMO, IMC, Putnam,
Miklós), 338 multiple choice (the 336 Kangaroo items and two prompt variants
of apex_2025::7), 944 short answers, and 242 items whose
content is only a system prompt. On matharena, "proof +0.93 against integer
-0.48" was mostly a difference between competitions: within competition the
format's correlation is +0.07. Every candidate therefore gets only the
zero-centred per-pair slope from B7, with prior sd at most 0.5.

### Through the harness

The harness scores each covariate as an offset on the shipped hier, using
test-like runs (300), mix/whole runs (150) and public R1 runs (100 per weighting)
(`experiments/harness.py`). x is standardised within each benchmark: it is 0
where an item lacks the cue, or where fewer than 10 of the benchmark's items
differ from the most common value, and it is clipped at ±3 sd. The per-pair
prior sd is s per within-benchmark sd (`BenchScaleEngine`, see the side findings
for why).

* "Allowed nested" is nested leave-one-parent-out selection over s in {0.1,
  0.25, 0.5}, from B7.
* The forced lines switch one configuration on everywhere, without selection.
* The placebo permutes x within each benchmark (three draws, differences
  averaged). It shows what the same configuration gains or costs from noise
  with the same support.

Differences are paired ALC against the shipped hier, written "± run SE / cluster
SE / stratified SE".

| covariate, allowed nested | test-like | mix/whole | worst parent | R1 benchmark-first | R1 pair-uniform | folds on | gate |
|---|---|---|---|---|---|---|---|
| position | 0 | 0 | 0 | 0 | 0 | (3/4) | no |
| position_within | 0 | 0 | 0 | 0 | 0 | 0/4 | no |
| format_score | 0 | 0 | 0 | 0 | 0 | (3/4) | no |
| log_length | +0.00003 ± 0.00001 / 0.00002 / 0.00002 | -0.00010 | +0.00004 | -0.00012 | -0.00020 | 3/4 | no |
| stated_size | 0 | 0 | 0 | 0 | 0 | (3/4) | no |
| image_ref | +0.00003 ± 0.00001 / 0.00001 / 0.00001 | +0.00004 | +0.00013 | +0.00001 | +0.00005 | 2/4 | no |

No covariate passes the gate (test-like ≤ -0.002). Nested selection cannot
switch on a covariate that varies on one parent only. With that parent held
out, its inner folds see a constant and choose off. On the other folds, the
choice is made on that parent's inner gains and applied to parents where x is
constant, which gives the "folds on" in brackets and a difference of exactly 0.

log_length and image_ref are switched on in some folds and do nothing. ordinal
difficulty is constant on every stored item, so q = p.

Switched on regardless, the forced per-pair line with s = 0.5 from B7 does this
on the parents that carry the cue (test-like, per held-out parent ± cluster SE,
placebo in brackets):

| covariate | parent | test-like | mix/whole | R1 bench-first / pair-uniform | within-pair r (test-like) |
|---|---|---|---|---|---|
| stated_size | researchcodebench | **-0.0022 ± 0.0005** (+0.0008) | -0.0062 ± 0.0008 | -0.0064 / -0.0060 | +0.41 |
| position | matharena | -0.0004 ± 0.0003 (+0.0007) | -0.0011 ± 0.0005 | -0.0013 / -0.0013 | -0.09 |
| position_within | matharena | +0.0011 ± 0.0003 (+0.0009) | +0.0009 ± 0.0003 | -0.0008 / -0.0005 | -0.04 |
| format_score | matharena | +0.0002 ± 0.0001 (+0.0010) | +0.0002 | -0.0012 / -0.0007 | -0.01 |
| log_length | four parents | +0.0002, +0.0004 ± 0.0001, -0.0005 ± 0.0007, +0.0002 (+0.0002 to +0.0010) | -0.0003 overall | -0.0004 / -0.0005 overall | +0.05 |
| image_ref | matharena, multi_swebench, researchcodebench | +0.0005 ± 0.0002, +0.0001, +0.0003 (+0.0004 to +0.0007) | +0.0004 overall | +0.0003 / +0.0004 overall | -0.02 |

(The within-pair r is the harness table's `r_within_pair_tl`: the mean over
test-like pairs on which x varies of its correlation with honest difficulty
over the pair's evaluated items.)

Only stated_size carries item signal that the per-pair slope can use. It reads
on the threshold table where the table says it should:

* Its within-pair r of 0.41 lies between the table's r = 0.3 and r = 0.5 rows
  (within-pair 0.25 and 0.42).
* Those rows' forced per-pair lines give -0.0004 and -0.0023 per appearance,
  on all four parents.
* It gains -0.0022 per appearance, on the one parent that has the cue.

All of the gain comes from B7 on: -0.0007, -0.0009 and -0.0014 of Brier at B7,
B15 and B31 over all test-like appearances. Its sign is what one would
declare: more code to write is harder. But it exists on one public benchmark,
so neither the nested rule nor the sign rule can be met. At s = 0.25 it gains
-0.0011 ± 0.0002 against a placebo of +0.0002. Pooled over all test-like
appearances, the best forced line is -0.0005, a quarter of the gate.

matharena's problem_idx correlates +0.16 with difficulty within competition in
sample. It does not survive the pseudo-benchmarks, where the within-pair r is
-0.09, and it gains -0.0004 against a placebo of +0.0007. Its percentile
within the competition loses more than its placebo.

### What this could do on the hidden test

Nothing tested here acts at B0: a centred offset is 0 without labels. This
section assumed an uncentred offset would only shift a benchmark's level,
which is the level prior's job. The harness's later B0 term shows it also
spreads items within a pair, but it is worth little ("Acceptance harness"):
-0.0027 on the honest oracle, which still fails the gate, and at most -0.0010
at r = 0.7 on the degraded oracles, where it never passes. The per-pair slope
acts only from B7. hier's formative run scored 0.237 and 0.195 at B0 and B1,
against 0.196, 0.184, 0.178 and 0.183 at B3 to B31. So B0 and B1, which carry
0.3 of the weight and most of the headroom, are out of this section's reach.

A hidden benchmark that states sizes as researchcodebench does would gain about
0.002 on its own pairs if the slope were switched on. That is about 0.0003 of
a run's ALC if one hidden benchmark in seven did. A benchmark whose stated
sizes carry no signal would cost about +0.0008 on its pairs (the placebo).

How often hidden benchmarks carry such cues is speculative. The organisers'
inventory (`results/inventory.csv`, 161 titles) holds no items, and a title
says what a benchmark is about, not which fields its items carry. The title scan
(`--stage inventory`) finds:

* 1 title naming per-item levels (PhyBlock, "progressive");
* 5 multi-step or planning benchmarks;
* 9 agent benchmarks;
* 6 code, 2 math and 1 proof benchmark;
* no multiple-choice benchmark;
* 60 (37%) on images, video, charts, documents or 3D.

The step-2 classification of the same 161 titles (`python
experiments/inventory_classes.py`, seconds, no network; `results/inventory_classes.json`
and `.csv`; moved from session scratch and reproducing it exactly) sorts titles
only, by keyword rules: no description, paper, repository or item is read. It
classes 108 of the 161 (67%) as evaluations of AI systems. Of those 108 it gives
text QA 39%, images 30%, agents 15%, code 6.5%, video 4.6%, math 2.8% and audio
2.8% (of all 161: 26%, 20%, 9.9%, 4.3%, 3.1%, 1.9% and 1.9%, and 33% not an
evaluation). It was checked against 80 labels on a random sample of the
organisers' full 1,261-row sheet (`results/inventory_hand_labels.csv`; 14 of the
80 fall among the 161). Those labels were assigned by an AI agent reading each
title, not by a person. The rules agree with them on evaluation or not for 66
of 80 (82.5%), and on the category for 38 of the 40 titles both call
evaluations (95%). Read by class:

* **Ordinal fields.** Public item_features hold grouping metadata (competition,
  lang, website, paper) plus one index, and nothing difficulty-named on any of
  the five benchmarks. Some source datasets do carry levels: GAIA's Level 1-3
  and Online-Mind2Web's easy/medium/hard among agents, LiveCodeBench's
  difficulty in code, MATH's levels in math. Whether the organisers pass such
  levels into item_features is unknown. A guess is 0 to 1 of 7 hidden
  benchmarks.

  Nothing public could calibrate a slope for one. The harness's gate table
  ("Acceptance harness", re-collected rows) puts an honest-r 0.3 covariate at
  -0.00255 with a transferred slope (6 of 8 draws pass; the table before the
  audit, on the legacy rows, gave -0.0024), but a transferred slope needs the
  cue on public parents. The
  per-pair slope needs r of about 0.5.
* **Position.** Contest-style math only (2.8%).
* **Format.** It varies within a benchmark only in mixed sets like matharena. A
  QA or multiple-choice benchmark usually has a single format, which centring
  removes.
* **Stated size.** Researchcodebench-style code prompts, which are rare.
* **Image references.** Constant in image benchmarks (30%), where every item
  holds an image.
* **Length.** Present everywhere, with an unknown sign.

### Side findings

* **The shipped multiple-choice floor misfired on matharena.**
  `paiec/mcq.py`'s floor, which hier applies, missed all 336 Kangaroo items.
  Those are five-option multiple choice with the options in the image; the
  text says "(A, B, C, D, or E)". The floor also fired on 20 AIME and HMMT
  integer-answer items whose TikZ drawings label points "(A)" to "(E)".
  `itemcov.n_options` cuts drawings out and reads such a list. The floor was
  not changed in this section; it was corrected since, in f7e7d87 ("The
  multiple-choice floor, corrected").
* **The harness's per-pair scale is wrong for some covariates.** The harness's
  per-pair prior sd is s over the sd of x pooled across the training parents'
  evaluated items. That is right for its degraded oracles, which are
  standardised on all four parents. It is wrong for a covariate that is absent
  from a parent or that carries benchmark-level offsets:
  * format_score, with matharena held out, has a pooled sd of 0.047. That turns
    s = 0.5 into a prior sd of 10.7 per unit, and its forced per-pair line on
    matharena costs +0.0023 instead of +0.0002.
  * log_length's pooled sd of 2.36 shrinks s = 0.5 to 0.21 per unit.

  `--scale train` reproduces these numbers (`harness_train_scale`) on the
  legacy rows with `HARNESS_SCALE=pooled` (header above). Both fixes are now
  the harness's defaults ("Acceptance harness", "What changed in the audit"):
  `harness.py --stage eval` standardises x within benchmark, and its per-pair
  unit is the pooled within-benchmark sd of x.

### Verdict: ship none of them

* No candidate passes the sign rule. The one present everywhere, log length,
  changes sign between benchmarks.
* No candidate passes the gate. Nested test-like differences are between 0 and
  +0.00003, where the gate needs -0.002.
* The one cue with real item signal is researchcodebench's stated size (r +0.50,
  -0.0022 on its pairs from B7 when forced). It exists on one public benchmark,
  so it cannot be selected leave-one-parent-out, and its expected value on the
  hidden test is at most about 0.0003.
* These covariates cannot reach B0 or B1, where the hidden test's headroom
  is.

`paiec/itemcov.py` stays research-only.

**What was cut.**

* Difficulty-named fields could not be evaluated: there are none in the public
  data, and there is no fixed a-priori slope, because the plan allows a
  non-zero slope only through the sign rule.
* Counts of numbers, lines or file paths in the text were not tried. Earlier
  findings show cheap text statistics do not transfer.
* x is standardised over the whole parent, not over a run's visible items.
* position_within reads the parent's item list.
* The transferred and hybrid lines are reported in the results file for
  reference only.
* There are three placebo draws per line.

## Subject side at budgets 0 and 1

`python experiments/subject_side.py --stage verify`, then `--stage diag`, then
`--stage run --jobs 2` for the single candidates (1 h 53 min, plus about 15
minutes in two stopped starts whose finished rows were kept). The combinations
ran as `--stage run --jobs 2 --configs E+Dclip,E+Dlog,E+Dhc,H+E` (about 70
minutes; it was started with `,T,T1.8` appended and stopped once the Gaussian
tasks were done, so it left no `passes` record). The Student-t levels ran as
`--stage run --jobs 2 --configs T,T1.8` (67 minutes). `--summarise` comes last.
Everything ran on two spawned worker processes of a machine shared with other
jobs. Every number below is in `results/subject_side.json`, and `--summarise`
rebuilds its summary from the rows. The rows hold per-pair Brier by budget for
every configuration and run; they live outside the repository, by default in
`data/subject_side_rows`.

The second formative run of the shipped hier scored 0.237, 0.195, 0.196, 0.184,
0.178 and 0.183 by budget. B0 and B1 carry 0.3 of ALC's weight. At those two
budgets the model has little beyond its level prior and its subject prior. This
section measures four changes to those priors against exactly what ships. Each
change sits behind a flag that is off by default.

**Candidates.** Three of them are terms of the attribute ridge behind theta's
prior. They are `paiec.subjects`' optional `Spec` terms, reached through
`prior.build(design=...)`. The fourth changes the level prior.

* H, harness identity. It adds a column for each canonical harness string with
  at least 8 training rows (lower case; runs of space, '-', '_', '/' and '.'
  become one space). The present/absent flag stays on top of these columns, and
  an unseen harness string gets the flag alone. This is Ge et al.'s additive
  theta_LLM + theta_scaffold.
* E, ordered reasoning effort. A has-effort flag and a rank (minimal -2, low
  -1, medium 0, high 1, xhigh 2, with max read as xhigh) replace the per-level
  dummies. Today the only dummy is 'high', the one level with 8 rows.
* D, the form of the release date:
  * Dclip is linear, with the date held to the training rows' range.
  * Dhinge is linear plus a second slope from the training median date on.
  * Dhc combines the two.
  * Dlog uses the log of days since 2023, a saturating trend.
* T, a Student-t level (3 df, as a scale mixture) at LEVEL's centre. T uses
  scale 2.5. T1.8 uses 1.8, which is LEVEL's width without the 1.44 widening.

Everything else is `submission/model.py`'s LEVEL (mu0 -2.5, sigma_mu 2.5,
attr_scale 0.5) over `prior.build`, fitted without each target's parent
benchmark. The harness and the runs are those of
`experiments/level_calibration.py`. At their defaults the new terms change
nothing:

* The default design gives the earlier prior, hyperparameters and hier
  predictions bit for bit (`tests/test_hier.py`).
* It rebuilds the shipped prior.json's subject prior exactly.
* The legacy Predictor's path (`subject_frame`, `design`, `design_row`,
  `fit_prior`) gives identical coefficients with each benchmark left out.
* `paiec/hier.py` is unchanged. The Student-t level is its existing
  `Hyper.nu_mu`.

**Runs.**

* Test-like runs (seed 2, runs 0 to 299). This is the primary regime and uses
  the runs of "Item signal from the pair's own labels".
* Groups merged at random, and wholes (seed 3, runs 0 to 199).
* Public R1 as the guard: benchmark-first runs 0 to 149 and pair-uniform runs 0
  to 99.
* Test-like runs without the date shift (seed 3, runs 0 to 99). This regime is
  here because the test-like date shift of 1.25 years is a synthetic knob. It
  acts on a prior through the prior's date term (`paiec/testlike.py`,
  `date_shift`).

The Student-t level costs five to nine times as much. It scores test-like runs
0 to 99, mix/whole 0 to 59 and R1 0 to 59 and 0 to 39.

Two checks tie the harness to earlier results:

* On the same runs, 'ship' reproduces itemsig_eval's base exactly (ALC 0.165755
  and 0.171062).
* On one test-like run and one public run, the harness matches
  `official.run_official` (with deep copies) pair for pair for 'ship', H and
  E+Dhc (`--stage verify`, difference 0).

**Gates**, as the plan sets them. A component ships only if all of these hold:

* Nested leave-one-parent-out selection switches it on, in at least three of
  the four folds.
* The nested test-like difference is at most -0.002.
* No held-out parent is above +0.002.
* Neither public weighting loses more than 0.001.

Nested selection works like this. For each held-out parent, it takes the
configuration (or 'ship') with the lowest mean difference on the other parents'
appearances, taking the worse of the two shifted test-like regimes. It then
scores that choice on the held-out parent. A target's prediction depends only
on its own parent's prior, so a mix of choices across parents is exactly what
running them would give. The inner appearances, though, were scored with priors
fitted without their own parent but with the held-out one. The selection is
therefore nested in the configuration, not in every coefficient. The standings
check below is strictly nested.

### What the public data can identify

Harness strings exist on one multi-subject benchmark, multi_swebench: 81 of 82
pairs, 12 strings, six of them on 12 pairs each. swe_rebench's single pair also
has one (OpenHands), but that pair has no standing. reasoning_effort exists
only on matharena: 23 of 81 pairs (high 12, xhigh 4, low 3, medium 3, max 1).

So when a parent is left out, a harness or effort column is either untrained
(its benchmark is the one held out) or has no target to act on (it is not).
Under leave-one-parent-out, H and E act only through the other coefficients
they move. Their direct effect cannot be measured here; only its
within-benchmark analogue can.

That analogue is on the standings, the attribute ridge's target. Leave out one
model name at a time, with every benchmark in training: the harnesses are seen
and the model is new, which is Ge et al.'s setting. There, the harness columns
cut multi_swebench's MSE from 1.30 to 1.04. Within that benchmark the harness is
the largest attribute: on the full fit its standing is +0.75 for Agentless,
-0.75 for MSWE-agent and -0.35 for OpenHands.

Whether any of this carries to another benchmark, the public data cannot say.
The one cross-benchmark case is swe_rebench's pair on public runs, which H
scores with multi_swebench's OpenHands coefficient. It costs +0.0005 on that
pair.

Release dates are on every benchmark, so the date form is the only candidate
identified with a benchmark left out. Held-out standings (`--stage diag`; ridge
at alpha 2; MSE with the prediction centred within the benchmark):

| design | matharena | multi_swebench | real_webagents | researchcodebench | mean |
|---|---|---|---|---|---|
| ship (linear) | 5.42 | 1.18 | 2.06 | 1.70 | 2.59 |
| H | 5.54 | 1.18 | 2.05 | 1.71 | 2.62 |
| E | 5.42 | 1.11 | 1.91 | 1.82 | 2.57 |
| Dclip | 5.43 | 1.18 | 2.06 | 1.80 | 2.61 |
| Dhinge | 4.94 | 1.07 | 1.85 | 1.69 | 2.39 |
| Dhc | 4.95 | 1.07 | 1.85 | 1.80 | 2.42 |
| Dlog | 6.21 | 1.29 | 2.04 | 1.87 | 2.85 |

(The standings' variances are 7.21, 1.70, 2.78 and 2.60.) Within the public
range the trend is convex, not saturating. The hinge, whose second slope is
steeper, fits every held-out benchmark better. The log form fits worst.

A strictly nested choice takes, for each fold, the design with the lowest error
on the other benchmarks, each fitted without both. The hinge wins three folds of
four: matharena, multi_swebench and real_webagents. researchcodebench's fold
picks E+Dhc, which loses there.

E's gain on multi_swebench and real_webagents is indirect. The effort rank
explains matharena's high-effort 2025-26 models, so the date slope falls: from
2.54 to 2.22 per 400 days with multi_swebench held out, and from 2.57 to 2.27
and 2.63 to 2.27 for the other two.

Beyond the data, the forms part ways. Standings on the full fit, for an OpenAI
subject without a size (item-level scale, before attr_scale 0.5; the latest
public release is 2026-04-24):

| design | 2025-06 | 2026-04 | 2026-09 | 2027-06 |
|---|---|---|---|---|
| ship | +1.30 | +3.13 | +4.05 | +5.70 |
| Dclip | +1.30 | +3.13 | +3.27 | +3.27 |
| Dhinge | +1.09 | +3.40 | +4.56 | +6.63 |
| Dhc | +1.09 | +3.40 | +3.57 | +3.57 |
| Dlog | +1.29 | +2.40 | +2.86 | +3.56 |

### On the runs

Differences are against 'ship'. The test-like column gives ± run SE / cluster
SE; the other columns are means only. The last two columns are the test-like
Brier differences at B0 and B1.

| configuration | test-like | mix/whole | no date shift | R1 benchmark-first | R1 pair-uniform | B0 | B1 |
|---|---|---|---|---|---|---|---|
| H | +0.00007 ± 0.00001 / 0.00003 | +0.00007 | +0.00005 | +0.00009 | +0.00008 | +0.0003 | +0.0002 |
| E | -0.00097 ± 0.00004 / 0.00013 | -0.00096 | -0.00005 | -0.00002 | +0.00001 | -0.0047 | -0.0017 |
| Dclip | +0.00012 ± 0.00011 / 0.00031 | -0.00026 | +0.00001 | +0.00004 | +0.00002 | -0.0001 | +0.0005 |
| Dhinge | +0.00055 ± 0.00011 / 0.00030 | +0.00062 | -0.00066 | -0.00037 | -0.00045 | +0.0030 | +0.0007 |
| Dhc | +0.00052 ± 0.00006 / 0.00019 | +0.00015 | -0.00065 | -0.00034 | -0.00043 | +0.0022 | +0.0010 |
| Dlog | -0.00099 ± 0.00015 / 0.00046 | -0.00132 | +0.00078 | +0.00063 | +0.00076 | -0.0058 | -0.0013 |
| E+Dclip | -0.00069 ± 0.00013 / 0.00038 | -0.00104 | -0.00005 | +0.00001 | +0.00003 | -0.0040 | -0.0010 |
| E+Dlog | -0.00167 ± 0.00018 / 0.00054 | -0.00195 | +0.00068 | +0.00060 | +0.00077 | -0.0091 | -0.0025 |
| E+Dhc | -0.00045 ± 0.00008 / 0.00024 | -0.00080 | -0.00064 | -0.00034 | -0.00042 | -0.0025 | -0.0008 |
| H+E | -0.00090 ± 0.00004 / 0.00015 | -0.00089 | -0.00001 | +0.00007 | +0.00009 | -0.0044 | -0.0016 |

'ship' itself scores 0.1658, 0.1711, 0.1585, 0.2051 and 0.2020 in these
regimes. The combinations were chosen after the singles' first 121 test-like
runs had been seen. Each of them is E with a date form that did not lose on
average there, or E with H.

**Where the differences sit.** Almost all of every difference is at B0 and B1:
63 to 89% of each ALC difference above 0.0002 (one exception is Dclip on
mix/whole, at 48%). Nothing moves by more than 0.0005 of Brier from B7 on.

**Per held-out parent on test-like runs.** Matharena is where the date forms
split: Dlog and E+Dlog +0.0044 ± 0.0014, Dclip and E+Dclip +0.0041, Dhc +0.0023,
Dhinge -0.0025. E and H cannot act on matharena (see above).

**The date shift drives the date results.** For every date form the two
shifted regimes and the unshifted one disagree in sign. E's gain is gone
without the shift (-0.00005). The shift moves test-like subjects 1.25 years
later, most of them past every public release. A design that extrapolates less
then lowers their B0 prediction on pairs tilted low, and it scores. E
extrapolates less through its lower slope, Dlog through its curvature, and Dclip
by construction.

On the subjects' own dates, and on public runs, the same designs change nothing
(E, Dclip) or lose (Dlog +0.0006 to +0.0008). The hinge fits held-out standings
best and gains 0.0004 to 0.0007 there, but it loses under the shift because it
extrapolates more steeply.

So the test-like regime cannot choose between date forms. Whatever it prefers,
it prefers because of a synthetic knob, and that knob was tuned for the legacy
prior's linear date term: testlike's docstring notes that a predictor reading
dates otherwise "gets a different optimism from it".

### Nested selection

For each configuration against 'ship' on its own, and for the choice among all
ten:

| selection | folds on | test-like | mix/whole | R1 benchmark-first | R1 pair-uniform | no date shift | worst held-out parent |
|---|---|---|---|---|---|---|---|
| E vs ship | 4 | -0.00097 ± 0.00033 | -0.00096 ± 0.00049 | -0.00002 | +0.00001 | -0.00005 | 0 (matharena) |
| H vs ship | 1 | +0.00008 | +0.00007 | +0.00005 | +0.00008 | +0.00005 | +0.0004 (matharena) |
| Dclip vs ship | 1 | +0.00080 | +0.00060 | 0 | +0.00001 | +0.00001 | +0.0041 (matharena) |
| Dhinge vs ship | 1 | +0.00068 | +0.00109 | -0.00005 | -0.00005 | -0.00019 | +0.0017 (multi_swebench) |
| Dhc vs ship | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dlog vs ship | 3 | +0.00033 | +0.00084 | +0.00048 | +0.00056 | +0.00050 | +0.0044 (matharena) |
| E+Dlog vs ship | 3 | +0.00008 | +0.00088 | +0.00048 | +0.00061 | +0.00048 | +0.0044 (matharena) |
| E+Dhc vs ship | 3 | +0.00011 | +0.00023 | -0.00029 | -0.00036 | -0.00046 | +0.0023 (matharena) |
| all ten, selected on both shifted regimes | E 3, Dlog 3, Dhinge 1 | +0.00076 ± 0.00102 | +0.00197 ± 0.00112 | +0.00043 ± 0.00014 | +0.00056 ± 0.00013 | +0.00029 | +0.0044 (matharena) |
| all ten, selected on test-like alone | E 4, Dlog 3 | -0.00052 ± 0.00118 | -0.00005 | +0.00047 | +0.00058 | +0.00043 | +0.0044 (matharena) |

(± is the cluster SE, with the selection redone in every resample. E+Dclip,
at +0.00028 with matharena +0.0041, and H+E, at -0.00031 with matharena
+0.0004, are in the results file. Each is switched on in three folds.) No
component passes:

* **E** is switched on in every fold, never loses on a held-out parent and costs
  nothing on public runs. But it gains half the -0.002 the gate asks for, and
  that gain rests entirely on the synthetic date shift.
* **Dlog** is switched on in three folds. One of them is matharena held out,
  where it then costs +0.0044. It also costs 0.0005 on public runs.
* **H** acts only indirectly (+0.00005 to +0.00009 everywhere).
* **Dclip, Dhinge and Dhc** are switched on in at most one fold.
* **Combined.** Selected on both shifted regimes, the choice loses on test-like
  runs as well (+0.0008 and +0.0020). Selected on the primary regime alone, it
  gains 0.0005 there and loses 0.0005 on public runs.

### The Student-t level

Differences against 'ship' on the Student-t runs (± run SE / cluster SE):

| configuration | test-like | mix/whole | R1 benchmark-first | R1 pair-uniform | ms a call | slowest call |
|---|---|---|---|---|---|---|
| T (scale 2.5) | +0.00005 ± 0.00013 / 0.00017 | +0.00039 ± 0.00011 / 0.00020 | +0.00113 ± 0.00021 / 0.00029 | +0.00057 ± 0.00025 / 0.00028 | 4.4 to 6.7 | 1.06 s |
| T1.8 | -0.00000 ± 0.00002 / 0.00003 | -0.00003 ± 0.00003 / 0.00005 | +0.00021 ± 0.00008 / 0.00010 | +0.00043 ± 0.00013 / 0.00014 | 4.5 to 6.6 | 0.92 s |
| ship (Gaussian) | | | | | 0.77 to 1.22 | 0.31 s |

**T.** At the shipped scale, a t level carries its extra tail mass into every
B0 prediction. On test-like runs that costs +0.0019 at B0; on public runs it
gains 0.002 at B0 and then gives back 0.001 to 0.003 at each of B1 and B3.

**T1.8.** At scale 1.8 the t is level with the Gaussian on test-like runs
(0.00000 and -0.00003) and loses 0.0002 to 0.0004 on public runs.

**Nested.** Selection switches T1.8 on in one fold (multi_swebench held out),
for a nested +0.00004. So the earlier tie holds at the shipped level, with the
t mixture as it is now. The cost is five to nine times the Gaussian's per call,
with single calls up to a second.

### Latency

On the same runs, the Gaussian candidates take 0.78 to 1.04 ms a call, as
'ship' does (0.80 to 1.04 ms; slowest call 0.16 to 0.31 s). Their slowest
single call is 0.43 s or less,
except one call of E's at 1.53 s on a public run. That is likely load on the
shared machine, since the other 49 slowest-call figures are 0.07 to 0.43 s. The
new terms add at most eight columns to a ridge that is fitted offline, plus as
many multiplications to each subject's prior mean at run time.

### Verdict: ship none of them

The shipped hier stays as it is, and every flag stays off. By the gates:

* **Harness identity (H)** cannot be measured leaving a parent out. The public
  data hold harness strings on one multi-subject benchmark. Its only
  cross-benchmark test, one swe_rebench pair, loses 0.0005. Within
  multi_swebench the harness is the largest attribute, as Ge et al. find, but
  that is within-benchmark transfer, not what the hidden test asks for.
* **Ordered effort (E)** is the only candidate with no measured downside.
  Nested selection switches it on in all four folds, but it gains -0.00097 ±
  0.00033, half the bar. All of the gain comes through a lower date slope under
  the synthetic date shift; without the shift it is -0.00005.
* **The date form (D)** is decided by the date shift, not by data. On real
  dates the hinge gains 0.0004 to 0.0007 and fits held-out standings best. It
  loses under the shift, and the forms that win under the shift lose on real
  dates and on matharena.
* **The Student-t level (T)** ties or loses, at five to nine times the cost.

**What this says about the headroom at B0 and B1.** The subject side moves B0
and B1 only by what it says about a subject past the public dates. How far the
hidden subjects lie past 2026-04 is exactly what the test-like regime's date
shift assumes and no public run can check. Taken at face value, E+Dlog would
cut test-like B0 by 0.009 and B1 by 0.0025. That is a bet on the shift
hypothesis, and it fails the parent and guard gates.

The next formative feedback, read per pair together with the first run's nine
pairs (as "What actually shipped, after the audit" already asks), is where to
look. (That reading is now in "Formative feedback, runs 1 and 2". Run 2's B0
excess sits on both sides of a rate of 0.5, and by the preregistration that
observation changes nothing.) The question is whether hier's B0 excess over B31 (0.054 on the second
run) sits on low-rate pairs, where an optimistic subject prior would put it. If
it does, damping the date term is worth testing again. That would be a global
change to one hyperparameter, attr_scale or a date clip, not a per-benchmark
fact. If it does not, the hinge's better fit on real dates is the better bet.
The test-like regime alone cannot decide this.

**What was cut.**

* Only one seed per regime was run.
* The Student-t level ran on 100/60/60/40 runs and not without the date shift.
* There was no strict nested harness pass, with priors refitted without both the
  outer and the inner parent; only the standings check is strictly nested.
* attr_scale and LEVEL were not re-tuned together with the date forms.
* The harness threshold (8 rows) and the effort ranks were not varied, and
  effort is read from the field only, not from names.
* Only split scope 'pair' was run, with no dense runs and no level-mean
  sensitivities.
* The combinations were chosen after part of the singles was seen.

## The multiple-choice floor, corrected

`python experiments/mcq_floor.py --stage items`, then `--stage run
--harness-rows data/harness_rows_legacy` (sharded, about 50 minutes on two
processes), `--stage summary`; every number below is in
`results/mcq_floor.json`. `tests/test_mcq.py` pins the adopted floor. The run
stage checks its replay of the shipped floor, prediction for prediction,
against `experiments/harness.py`'s stored rows; the rows it was checked
against (old floor, solver before the floored-fit fix) are now
`data/harness_rows_legacy` ("Acceptance harness", Provenance). The results
were computed with hier's solver before the floored-fit fix ("Floored fits"
below). A re-run now replays the fixed solver, which moves the corrected-floor
arm by that fix's ALC difference: -0.000008 on public R1 pair-uniform (its
-0.0004514 becomes -0.000460, which prints as -0.00046, assuming the old-floor
arm is unchanged, as it is on the runs 0 to 99 the replay covered; runs 100 to
199 of that arm were not replayed), -0.0000002 on
benchmark-first and 0 on test-like; the hard-floor arm moves by -0.000058 and
-0.000019 on the two R1 weightings (`results/hier_floor.json`).

hier floors a target at c = Hyper.guess × `mcq.floor_of(mcq_text(content))`, in
the likelihood of a floored item's labels as well as in its prediction; the
shipped guess is 0.5, so c = 0.1 on a five-option item. On the public items the
floor as it stood at bd0be67 got two things wrong, both on matharena:

* the 336 Kangaroo items are five-option multiple choice with the options in an
  image, named in the text only as "(A, B, C, D, or E)": no floor;
* 20 integer-answer AIME and HMMT items hold TikZ drawings whose
  `\coordinate (A)` labels read as options (A)..(E): a false floor.

`paiec/mcq.py` now cuts closed drawing blocks (TikZ, Asymptote, picture) out
before reading options and falls back to an "(A, B, …, or X)" list. Over all
12,632 public items of the six benchmarks, raw and as hier reads them, only
those 356 items change. The replay used a first version of the fix that read
exactly the same options on every public text but was quadratic in the worst
case; the adopted revision is linear (tests/test_mcq.py compares it with the
old floor on adversarial texts).

Corrected minus shipped, 200 runs per regime, same runs for both arms (± run SE
/ pair-cluster SE):

| regime | ALC | matharena appearances | elsewhere | B0 |
|---|---|---|---|---|
| test-like (seed 2) | -0.00026 ± 0.00003 / 0.00008 | -0.0014 per appearance | 0 (within 3e-7) | -0.0009 |
| public R1, benchmark-first | -0.00027 ± 0.00004 / 0.00011 | -0.0012 | 0 | -0.0011 |
| public R1, pair-uniform | -0.00045 ± 0.00005 / 0.00013 | -0.0012 | 0 | -0.0020 |

The gain sits at B0 and B1 and almost all of it is the list reading: at low
budgets the shipped level, moved down for the hidden test, under-predicts the
public Kangaroo items (0.33 to 0.42 against observed 0.72 to 0.80 at B0), and
the floor lifts them. That is specific to public matharena, so the hidden-test
value is probably smaller; it is still the correct reading of a five-option
item. The drawing cut alone is neutral (+0.00001 test-like). A hard floor
(guess 1) would gain a further 0.0002 to 0.0003 from the same under-prediction
and is not adopted.

One fit went wrong under the fix: r1p run 108, a matharena pair made only of
Kangaroo items, where hier's floored Newton fit stopped unconverged at B31 in a
collapsed mode. It predicted 0.22 at B31 against 0.67 observed, the success
rate over the pair's 148 evaluation responses, which is what its Brier scores.
(This section used to say 0.79. That is the success rate of the pair's
acquisition pool, 34 of its 43 candidate items; its first 31 labels hold 25
successes, 0.81.) It is counted in the r1p number above. These runs did not
record how many other fits stopped unconverged without collapsing; the replay
in "Floored fits" below counts them. The floored fit is fixed since, which
matters more now that the fix makes more pairs floored.

Verdict: adopted in `paiec/mcq.py`. `dist/paiec.zip` was rebuilt with it on
2026-09-27 (sha256 4a882cc7d410e6a9…; validator OK, archive bit-identical to
the in-repo predictor, `submission/prior.json` byte-identical to the previous
build).

### Floored fits

`python experiments/hier_floor_replay.py --stage compare --both` on two
shards (34 minutes; between runs each process stayed under the 1.3 GB of the
restart loop in the script's docstring, which never had to restart one),
`--stage timing` for r1p and r1b side by side (about 13 minutes, beside an
unrelated job), `--stage synthetic` (1 minute), `--stage modes` (45 seconds),
`--stage hidden` (10 minutes beside another job), then `--stage summary` and
`--stage show`. Every number below is in
`results/hier_floor.json`; the per-run replays are in `data/hier_floor/`
(gitignored). The solver before the fix is `paiec/hier.py` at f7e7d87, read
with `git show` into a temporary module; its relative imports resolve to the
working tree's `paiec`, which the fix did not touch. The fixed solver is the
working tree's. The replays use this section's checkpoints: the shipped hier,
fresh instances per checkpoint, runs 0 to 199 of each regime. The synthetic
cases are also pinned in `tests/test_hier.py`
(`test_a_floored_fit_far_below_its_successes_converges` and the three tests
after it). The numbers first came from a scratch replay; the script replays
every run that one did and reproduces its counts exactly. Where it differs,
it says so below.

A success on a floored item has likelihood c + (1 - c) s, and its log is
convex where s is small against c. From a prior mean well below a pair's
successes, Newton meets an indefinite matrix. `_cholesky` shifted it just past
its most negative eigenvalue (by 1e-8 of its largest diagonal element), which
left it nearly singular. The Newton step ran to 1e7 or more, twenty halvings
found no ascent, and the fit stopped where it stood, with every floored success
read as a guess. When the first step failed, that was the prior mean. The fit
was counted in `HierPredictor.unconverged` and predicted from there. Run 108
above was this case, and so were these synthetic ones (`--stage synthetic`):

* one pair on four-option items with its prior eta at -4 stayed at the prior
  mean for every record tried, 7/7, 15/15, 20/31, 25/31 and 31/31. It
  predicted 0.24 each time, against an exact posterior predictive of 0.61 to
  0.95. At -6 it stayed at 0.17 for those records and for 10/31 too;
* at the shipped level (-2.5) the one-pair case collapsed only under the hard
  floor, at 15/15 and 31/31 (0.43 against 0.92 and 0.96);
* seven subjects on three benchmarks, one of them all successes on
  five-option items, stopped short in 4 of 8 draws at the default level
  (-1.263) and in 7 of 8 at -2.5 (guess 0.5), and in 8 of 8 at both under the
  hard floor. Stopping short is not always a collapse. At -2.5, six of the
  seven put the all-success subject at 0.29, against 0.94 to 0.96 at the mode;
  at -1.263 the four stopped within 0.07 of the mode's prediction, one of them
  above it.

The fix is `Problem._settle`. It runs only where Newton on a posterior that
holds a floored success stops unconverged, and it starts from three points:

* where Newton stopped;
* the mode with every floor removed (each success is knowledge);
* the mode with every floored success dropped (each is a guess).

The last two problems are log-concave, so each has one mode. From each start,
`_settle` runs trust-region Newton on the floored posterior. Its multiplier
makes the matrix definite by as much as the radius needs, not by 1e-8. It keeps
the converged candidate with the highest log posterior. Where Newton
converges, nothing runs, so those fits are what they were bit for bit. The
one-pair cases above now match the exact posterior predictive to 1e-4, and
none of the seven-subject fits is left unconverged.

Shipped hier, corrected floor, 200 runs per row. The hard-floor rows (guess 1)
show where the collapse is more frequent. "Floored fits" counts the fits that
hold a floored success. The ALC difference is fixed minus pre-fix, with its
run SE. Both solvers ran on every run that holds a floored label. (The scratch
replay ran these rows with a multi-start variant of the fix and checked the
adopted code on the affected runs and a sample; the script replays the adopted
code throughout, with the same numbers.)

| regime | guess | runs with a floored label | floored fits | unconverged before | after | runs changed | predictions changed | max abs change | ALC difference |
|---|---|---|---|---|---|---|---|---|---|
| public R1, pair-uniform | 0.5 | 121 | 1,693 | 6 | 0 | 4 | 1,484 of 921,330 | 0.617 | -0.000008 (0.000008) |
| public R1, benchmark-first | 0.5 | 75 | 1,337 | 5 | 0 | 3 | 1,619 of 1,011,846 | 0.018 | -0.0000002 (0.0000001) |
| test-like (seed 2) | 0.5 | 94 | 1,471 | 0 | 0 | 0 | 0 of 559,272 | 0 | 0 |
| public R1, pair-uniform | 1 | 121 | 1,693 | 25 | 0 | 10 | 5,353 of 921,330 | 0.536 | -0.000058 (0.000032) |
| public R1, benchmark-first | 1 | 75 | 1,337 | 16 | 0 | 5 | 7,298 of 1,011,846 | 0.461 | -0.000019 (0.000013) |

Predictions changed only on the 22 run-rows where the pre-fix solver had an
unconverged fit. On the other 619 replays (the table's, and the old-floor
ones next), all 5,773,398 predictions are identical bit for bit. Under the old
floor (runs 0 to 99 of each regime), both solvers reproduce
`data/harness_rows_legacy` exactly on the 155 runs of r1p, r1b and test-like
that hold a floored label (81, 63 and 11); no fit there stopped short. Under
the corrected floor, the fixed solver reproduces the re-collected
`data/harness_rows` exactly on the 199 stored runs that hold one (66, 39 and
94).

The 52 rescued fits had stopped 1.6 to 5.9 logits from the mode, 4.5 to 103
nats of log posterior below it. Every one was rescued by continuing from where
Newton stopped, and its whole solve took 0.03 to 0.17 s. The largest changes
are the collapsed pairs (mean prediction; observed is the success rate over
the pair's evaluation responses):

* run 108's pair goes from 0.217 to 0.770 at B31 (0.669 observed over its 148
  evaluation responses), and its B31 Brier from 0.422 to 0.224;
* under the hard floor, r1p run 5 goes from 0.26 to 0.74 (0.75 observed) and
  r1b run 51 from 0.30 to 0.64 (0.79 observed). Like run 108's, these are
  matharena subject 8ae58e36 on Kangaroo items, and it also collapses at B15
  in r1p runs 37 and 108 (0.23 and 0.27, against 0.70 and 0.69 after the fix)
  and at B31 in r1b run 151 (0.37 to 0.76, 0.71 observed);
* in r1p run 187, under the hard floor, two researchcodebench pairs go from
  0.24 and 0.21 to 0.45 and 0.47 (0.42 and 0.47 observed).

Averaged over a regime this is invisible, because few public fits collapse.

**How often it would bite on the hidden test.** The synthetic cases above use
toy hyperparameters. `--stage hidden` runs hidden-test-like synthetic runs
through the shipped bundle (`submission/prior.json`: the level moved down, the
real attribute prior): seven benchmarks per scenario, four of them five-option
multiple choice and one four-option, 2025-26 subjects, records near 31/31,
each scenario fitted at B1, B3, B7, B15 and B31. A fit set collapses when the
solver before the fix leaves a fit unconverged:

| scenario | fit sets | collapsed before the fix | after |
|---|---|---|---|
| about one pair per benchmark (1.3 on average), real subject attributes | 300 | 0 | 0 |
| the same, attributes blanked | 200 | 1 (a 29/31 pair at B31: 0.26, 0.89 after) | 0 |
| four pairs per benchmark (records at rates 1, 0.2, 0.05, 0.5), real attributes | 125 | 10 | 0 |
| the same, attributes blanked | 125 | 35 | 0 |

In the crowded scenarios every collapse is an all-success pair at B7 to B31,
predicted 0.15 to 0.39 before the fix and 0.75 to 0.97 after. A collapse needs
several pairs on one floored benchmark: formative R1-like runs, or a hidden
benchmark drawn with more than one pair. With about one pair per benchmark and
real 2025-26 attributes it is rare, so the scratch replay's toy settings
overstated the risk for the hidden test's usual shape. The fix is still worth
shipping: it changes nothing where Newton converges and removes the failure
where it does not. Its time cost there is small: in the four scenarios in
turn the fixed solver took 5% less, 3% less, 2.5% more and 11% more time than
the solver before it, the most where the most fits are rescued (this stage ran
beside another job, so these are rough).

Converged fits keep Newton's mode because no other mode turned up. The
replay ran both other starts to convergence on every converged floored fit
(8,686; 7,479 on the five corrected-floor rows, the scratch replay's count)
and kept Newton's result regardless. Their log posteriors were within 2e-10
of Newton's, none above `LP_TOL` = 1e-6, and their modes within 3e-7.
`--stage modes` ran the same probe on 2,520 synthetic one-pair problems: levels
-6 to 6, sd 1 to 5, guess 0.5 and 1, 0 to 12 successes of 15 or 31 on three-
and five-option items, with and without two other subjects. It found no second
mode either: 2,057 converged floored fits, their 4,114 other starts within
4e-10 in log posterior and 3e-7 in the mode; the other 103 floored fits
stopped short and were settled, all converged. This is evidence, not proof,
that a floored posterior has one mode. The scratch replay measured that trying
both other starts on every converged floored fit costs 6 to 7% of prediction
time on runs with a floored label; that variant is not in the script and the
figure is not reproduced. The fix pays for the other starts only when Newton
stops short.

**Latency.** `--stage timing` ran runs 0 to 39 of r1p and r1b with fresh
instances per checkpoint, the offline bundles built before the clock starts,
the two solvers alternating within each checkpoint, and two passes. The fixed
solver took 1.422 ms a call against 1.426 before the fix on r1p (-0.2%), and
1.224 against 1.221 on r1b (+0.2%). The per-run ratio has a median of 0.99 on
both, with quartiles 0.93 to 1.05 and 0.96 to 1.04. An unrelated job used
about five cores throughout, so the absolute times are inflated (the scratch
timing, on an idler machine, gave about 0.9 ms a call) and the per-run ratios
are noisy; the comparison stays paired. The r1p runs include the two with a
rescued fit, 21 and 37, which the fix made 2% and 4% slower. (The scratch
timing gave 0.912 against 0.905 ms on r1p and 0.854 against 0.857 on r1b, over
runs 1 to 39: it charged the offline bundles to whichever solver met a parent
first, and dropped run 0 for that.)

Verdict: adopted in `paiec/hier.py`. `dist/paiec.zip` was rebuilt with it on
2026-09-27 together with the corrected floor (sha256 4a882cc7d410e6a9…;
validator OK, bit-identical run check). An official-protocol smoke of the
unzipped archive (3 test-like and 3 public R1 runs, 16 workers) had 0 failures
and 0 unconverged fits; it never reached `_settle`, whose evidence is the
replay above and `tests/test_hier.py`.

## The 4B judge, closed out

`python experiments/llm4b_close.py --stage signs`, then `--stage harness`,
`--stage reference`, `--stage verdict` and `--stage show` (2, 6 and 2.5
minutes and a second, on one process, at most 0.8 GB; no language model is
loaded). Every number below is in `results/llm4b_close.json`. The harness
stage reads the stored rows of `experiments/harness.py` (library
3f75a549673aae6a) and never recomputes hier.

`experiments/llm_features.py` showed Qwen3-4B-Instruct-2507 each public item
once and asked what share of strong 2025-26 systems would solve it, as one
digit (`paiec/llmfeat.py`). It read one forward pass and generated nothing. The
features are these, with the sign declared before this section looked at them
(+ = harder):

* **rating**: the expected digit (declared -).
* **digit**: the most likely digit, the rating a sampled answer would give
  (declared -).
* **entropy**: the entropy of the ten-digit distribution. Declared +: the
  judge's uncertainty should track difficulty (Zotos et al., arXiv:2412.11831).
* **nll**: the mean negative log-likelihood of the task text under the 4B. No
  sign declared.

**Coverage.** The extraction died of MPS out-of-memory on multi_swebench's 19th
of 22 rating shards. What exists:

* matharena: complete, 1,555 unique items standing for 1,755.
* multi_swebench: 1,941 of 2,078 unique items. Shards are planned by length, so
  the 137 missing are the longest prompts (1,439 tokens and more). They are a
  little easier (mean honest difficulty 0.10 against 0.19), and length does not
  order multi_swebench's difficulty (Spearman -0.02).
* real_webagents, researchcodebench, swe_rebench: nothing.

The manifest's progress field still says 10 of 22 shards; 18 are on disk.

**The extraction was not resumed.** 0.67 M prompt tokens remain for the four
parents, plus 3.8 M for swe_rebench. They need the fp16 model (8 GB) resident
while other language-model jobs of the same workflow wait for the one slot.
That is 2.3 hours at the extraction's contended rate (79 tokens a second), or
about 45 minutes uncontended, and the run died on exactly these long shards.
Why more coverage would not change the verdict is at the end.

### Against honest difficulty

The target is the fold-averaged (≈ in-sample) difficulty: Rasch b fitted
without each of the harness's five subject folds, averaged over the folds.
This section and the four after it ("Attempting instead of judging", "Entropy
profiles and hidden-state probes", the pairwise part of "Few-shot prompting",
"Fine-tuning an encoder") correlate against it, and so does "Strong model on
Kaggle: Qwen3-14B rubric and attempts"; the hidden-state heads, the encoder and
the 14B's rubric heads train on it. No label enters any label-free feature. But the
average of five fits, each on four fifths of the subjects, is essentially the
in-sample b: they correlate 0.999 [0.997, 1.0] on the attempt probe's items
(`results/attempt_probe.json`, references). The gate table's r ("Acceptance
harness") is instead against each fold's own honest difficulty, which is
noisier. An r against this target is therefore slightly optimistic when read
on the gate table: the bias favours the feature. No verdict below changes
with it; every feature failed anyway. (In-context learning's r_diff is
against the fold-specific honest difficulty and is not affected.)
Correlations are within item_features groups (competition, language): ranks
over the benchmark, demeaned within group. That is what an item covariate can
add to hier. Two partial forms are also computed:

* **partial**: within group and net of log length. Length ordered the attempt
  probe's items as well as any attempt feature.
* **partial with position**: net of the problem's index too (matharena only).

Intervals are 95% bootstrap over groups: 27 competitions, 8 languages. The
item bootstrap's are similar or narrower. A bootstrap over 8 or fewer clusters
gives intervals that are too narrow and unstable, so multi_swebench's are
indicative only; the same holds wherever this document resamples its 8
languages or 5 contests. Text-bearing matharena items exclude
Kangaroo's 336 image-only problems and 202 items whose content is only a system
prompt or a date and an id. That leaves 1,095 of the 1,633 items with a
difficulty.

| feature | matharena, within competition (1,633) | matharena text-bearing, within (1,095) | text-bearing, partial | text-bearing, partial with position | multi_swebench, within language (1,980) | multi_swebench, partial |
|---|---|---|---|---|---|---|
| rating | +0.18 [+0.12, +0.25] | +0.24 [+0.13, +0.33] | +0.22 [+0.12, +0.30] | +0.17 [+0.09, +0.25] | -0.11 [-0.21, -0.06] | -0.12 [-0.19, -0.07] |
| digit | +0.17 [+0.10, +0.24] | +0.17 [+0.08, +0.23] | +0.16 [+0.09, +0.22] | +0.14 [+0.08, +0.20] | -0.09 [-0.16, -0.04] | -0.09 [-0.15, -0.05] |
| entropy | -0.09 [-0.18, -0.02] | -0.19 [-0.27, -0.09] | -0.16 [-0.25, -0.07] | -0.14 [-0.21, -0.05] | +0.11 [+0.04, +0.21] | +0.11 [+0.05, +0.19] |
| nll | -0.01 [-0.08, +0.08] | -0.20 [-0.29, -0.07] | -0.11 [-0.18, +0.00] | -0.09 [-0.16, +0.01] | +0.05 [-0.04, +0.16] | +0.06 [+0.01, +0.14] |

(Spearman. Pearson tells the same story, smaller: the rating's partial Pearson
on text-bearing items is +0.12.)

* **Every feature changes sign between the two benchmarks.** On
  multi_swebench each goes the declared way, weakly (|rho| ≤ 0.12; nll, with
  no declared sign, rises with difficulty). On matharena each goes the other
  way within competition (nll on the text-bearing items only): the judge gives
  the harder problems of a competition higher ratings and more confident
  digits.
* **Part of that is the problem's position.** Within competition the rating
  rises with problem_idx (+0.23 on text-bearing items), and so does
  difficulty (+0.25). Net of position, the rating is still +0.17, wrong-signed.
* **Across competitions the rating orders nothing.** Over all of matharena it
  gives +0.05 [-0.11, +0.20].
* **It reproduces the attempt probe's reference.** On the probe's 160 items the
  script gives -0.31 for the negated rating, as `results/attempt_probe.json`
  does.
* **The 2026 contests show the same pattern.** They post-date the 4B, and the
  rating gives +0.27 within competition on their 150 text-bearing items. There
  are only five competitions, too few for an interval.
* **The sign rule cannot be met.** It needs the declared sign, and agreement
  with the other units' mean, on 4 of 5 units. Two units are rated, and they
  disagree for every feature within groups (0 of 2 leave-one-out agreements).

### Through the harness

Each feature was scored as an offset on the shipped hier, with x standardised
within benchmark and the B0 term reading raw x (`harness.eval_covariate`). It
covers 61% of the test-like evaluated items. Differences are benchmark-equal
test-like ALC: the mean of the four parents' means, the rule's measure.
real_webagents and researchcodebench contribute exactly 0 to it.

With two of the four parents rated, the transferred slope cannot be switched on
leave-one-parent-out. With a rated parent held out, its inner folds fit the
slope on the parents where x is constant. So its forced lines, fitted on the
other rated parent, are its reading. The placebo permutes x within each
benchmark (three draws).

| feature | transferred nested | per-pair nested | transferred from B1, forced (matharena / multi_swebench ± cluster SE) | per-pair s = 0.5 from B7, forced | B0 term, forced | placebo: transferred B1 / per-pair B7 |
|---|---|---|---|---|---|---|
| rating | 0 (0/4 on) | 0 (0/4) | +0.00008 (+0.00041 ± 0.00053 / -0.00011 ± 0.00007) | +0.00022 | -0.00013 | +0.00003 / +0.00031 |
| digit | 0 (0/4) | 0 (0/4) | +0.00013 (+0.00047 ± 0.00045 / +0.00005 ± 0.00003) | +0.00022 | -0.00002 | +0.00002 / +0.00032 |
| entropy | 0 (0/4) | +0.00003 (1/4) | +0.00049 (+0.00146 ± 0.00039 / +0.00051 ± 0.00011) | +0.00023 | -0.00017 | +0.00001 / +0.00037 |
| nll | 0 (0/4) | +0.00004 (1/4) | +0.00031 (+0.00047 ± 0.00022 / +0.00079 ± 0.00020) | +0.00025 | +0.00085 | +0.00002 / +0.00045 |

* **Nothing gains.** No line of any feature reaches -0.001. The best of all are
  the B0 terms of entropy and the rating, forced, at -0.00017 and -0.00013.
* **A slope carried between the two benchmarks costs.** Fitted on one and
  applied to the other, it costs up to +0.0015 on matharena (entropy) and
  +0.0008 on multi_swebench (nll): the sign flip at work. The hybrid prior
  shows the same: nested selection switched it on for entropy with matharena
  held out (s = 0.25 from B7, centred on multi_swebench's slope), and it cost
  +0.0010 ± 0.0003 there.
* **The per-pair slope pays the noise cost.** Forced at s = 0.5 from B7 it
  costs +0.0002, against +0.0003 to +0.0005 for the permuted feature. Nested
  selection has it act only for entropy and nll with matharena held out, at
  s = 0.25, where it loses +0.00014 and +0.00015.
* **The raw rating and entropy carry a sliver of level at B0.** Their
  uncentred B0 terms, forced, give -0.00018 and -0.00021 (± 0.00004)
  test-like, about -0.002 of Brier at B0, against a placebo of 0. That is a
  sixth of the bar, and nested selection leaves them off. nll's B0 term costs
  +0.0035 on multi_swebench.
* **Within test-like pairs the features barely order difficulty.** The mean
  correlation with honest difficulty over a pair's evaluated items is -0.09
  (rating), -0.06 (digit), +0.05 (entropy) and +0.04 (nll). The two benchmarks
  cancel. The harness table's r = 0.3 row is 0.25 within pair.

**The bar is reachable at this coverage.** The reference stage degrades the
honest difficulty to r on exactly the rated items, with 0 elsewhere (4 draws):

| r | transferred from B1, forced | per-pair nested | hybrid nested |
|---|---|---|---|
| 0.3 | -0.0012 | -0.0002 | -0.0008 |
| 0.5 | -0.0035 | -0.0012 | -0.0028 |

A covariate of honest r ≈ 0.3 with a consistent sign would have cleared -0.001
with a transferred slope, and one of r ≈ 0.5 with a per-pair slope. The judge's
features fail on quality, not on coverage.

### Verdict: KILL

The plan's rule has two prongs, and both hold:

* **ALC.** No benchmark-equal test-like difference is ≤ -0.001, under the
  nested transferred and per-pair lines or the transferred slope's forced lines.
  None is reached even with the per-pair slope forced.
* **Partial r.** On text-bearing matharena items, with the declared sign, every
  feature's partial r is below 0.2 (rating -0.22 Spearman and -0.12 Pearson;
  nll is oriented by multi_swebench's sign). In absolute value the rating's
  partial Spearman is 0.22 [0.12, 0.30], but wrong-signed. Only a per-pair
  slope could use a wrong-signed feature, and that needs r ≈ 0.5.

The 4B zero-shot judge features (rating, digit, entropy, nll) are closed.
`paiec/llmfeat.py` stays research-only. Its embeddings were closed in "Neural
embeddings do not carry difficulty to an unseen benchmark".

**Why finishing the extraction would not change this.**

* **Transferred slope.** It needs 4 of 5 units with the declared sign.
  matharena already fails for every feature, so all four other units would
  have to agree.
* **Per-pair slope.** It needs r ≈ 0.5 within a benchmark. The best |rho| here
  is 0.24. The blind ratings of a far stronger reader ("Language-model
  difficulty judgement") reached 0.21 on real_webagents and -0.13 on
  swe_rebench.
* **Cost.** It would take 2.3 hours of the one language-model slot for the four
  parents, and about 13 more for swe_rebench.

**Against the literature** (numbers as the step-2 review summarised them; not
re-read here):

* **ENEM** (arXiv:2602.06631). Small open judges land between -0.21 and +0.10
  against official item difficulty whatever the prompt (Llama-3.2-3B). The 4B's
  within-group correlations here, oriented, span -0.24 to +0.11: the same band,
  with a documented prompt search behind it (`experiments/llm_features.py`,
  "Prompt").
* **Ballon et al.** (arXiv:2512.14220). Language-model difficulty judgements
  track human difficulty at r ≈ 0.80 but model performance only at ≈ 0.23. The
  target here is model performance, and a 4B stays well below that.
* **Krsteski & Meyer** (arXiv:2608.05797). On 17 agentic benchmarks including
  MathArena, leave-one-benchmark-out Spearman is 0.225 for all features, 0.295
  ± 0.230 for a linear head, 0.017 for embeddings and 0.137 for a token-entropy
  profile, against 0.40-0.48 within benchmark. The ± 0.230 across held-out
  benchmarks is what a sign flip looks like, and here both rated benchmarks
  flip. Their pooled correlations also carry level differences between
  benchmarks, which hier learns from labels; per-pair ALC rewards only order
  within a benchmark.
* **Li et al.** (arXiv:2512.18880). A model's rating of its own chance predicts
  its errors at AUROC ≈ 0.55.

A small zero-shot judge is a dead end for this target. What remains open on the
language-model side is the attempt route ("Attempting instead of judging"),
which needs a model that can solve the items. A 14B that solves a fifth of
them took it later ("Strong model on Kaggle: Qwen3-14B rubric and attempts").

## Attempting instead of judging

`python experiments/attempt_probe.py --stage target`, then `--stage gen --design
D1 --ipb 2` (8 minutes), `--stage gen --design D2 --per-comp 2` (about 1.7
hours on the shared M1), then `--stage analyse`. Every number below is in
`results/attempt_probe.json`. The attempts are in `data/attempt_probe/`
(gitignored).

The 4B judge reads a problem and says how likely it is to be solved. In this
probe the same model, Qwen3-4B-Instruct-2507, is a subject: it attempts each
problem k times, and the item is described by what the attempts look like. No
feature uses a label or the reference answer. Each is oriented so that + means
harder:

* **Agreement.** The share of the k answers equal to the modal one
  (`top_share`, declared the primary feature before the run), and the entropy
  of the answers.
* **Confidence.**
  * The mean token log-prob of the answer inside `\boxed{}`, over all attempts
    or over the modal answer's attempts only.
  * The mean log-prob and the mean next-token entropy over everything
    generated.
  * The entropy of the first generated token.
* **Failure.** No parseable answer or a refusal phrase, the token cap reached,
  and the attempt's length.

Graded accuracy against matharena's reference answer is computed only to
diagnose FLOOR. The platform input carries no reference answer.

There are two designs. Both sample with the model card's settings: temperature
0.7, top-p 0.8, top-k 20.

* **D1, answer only.** k = 8. The assistant turn is prefilled with "The final
  answer is $\boxed{", and generation stops at the closing brace (at most 48
  tokens).
* **D2, short chain of thought.** k = 4. The model is asked to keep its reasoning
  under about 350 words and is capped at 512 new tokens. An attempt that ends
  without `\boxed{}` gets one forced continuation, "**Final Answer** $\boxed{",
  of at most 32 tokens.

The items are the 160 matharena probe items of the attempt-signal design (the
selection is re-derived in the script and matches the design's list exactly):

* text only, with a short checkable answer;
* at least 10 subjects each;
* 10 per competition over 16 competitions, spread over difficulty.

Kangaroo's multiple choice is not attempted, because its options are in the
image. The planned option-logit readout for multiple choice therefore has no
text items on matharena and was not run.

The target is the fold-averaged (≈ in-sample) difficulty: Rasch b fitted
without each of the five subject folds of `experiments/harness.py`, then
averaged over the folds. It is labelled "honest b" in this section's tables;
"The 4B judge, closed out" explains why it is essentially the in-sample b and
why that slightly favours a feature. b over the strong tier (subjects at or
above the median ability) is reported beside it. The statistic is the within-competition Spearman
correlation (ranks within competition, demeaned, pooled), with a 95% bootstrap
interval over competitions. The 2026 contests (AIME 2026 and 2026 I, HMMT
February 2026, arXiv-math January and February 2026) came after the 4B's
release.

### Running a 4B generator on the 16 GB M1

Beside the desktop and the other jobs about 8 GB of memory is free. The fp16
model (8.05 GB) plus its cache therefore did not stay resident: it decoded at
about 13 tokens a second and pushed the machine into swap. The model runs in
weight-only int8 instead: every linear layer and the tied embedding, rounded to
nearest with one absmax scale per row, through torch's MPS int8 matmul, 4.0 GB
in all. On 16 D1 items run both ways, int8 against fp16:

| feature | agreement, int8 against fp16 |
|---|---|
| first-token entropy | correlation 0.99, mean absolute difference 0.05 nats |
| mean answer log-prob | correlation 0.98 |
| top_share | correlation 0.95 |
| modal answer | the same on 15 of 16 items |

Three more changes were needed:

* **Padded rows.** torch 2.6's MPS attention returns NaN on the fully masked
  query of a left-padded row, and the NaN reaches every other row. Those rows
  are now unmasked (`_unmask_padded_rows`).
* **Shared prefill.** Each prompt is prefilled once and its cache is repeated k
  times. This matches the unshared path to 0.01 in entropy and log-prob, and
  cut D1 from about 23 minutes to 8.
* **Static cache for D2.** D2 uses a static cache: a growing one fragments the
  capped MPS pool until it runs out.

Decoding is bound by kernel launches. On an idle machine it takes 0.4-0.5 s a
step at 8-16 rows; during this run it took 0.8-1.6 s a step at 8-16 rows.

**What was cut.** At that speed the planned D2 (160 items, k = 4, up to 1,024
tokens) would take about 8 hours. D2 ran instead on 32 items with a 512-token
cap: the easiest and the hardest probe item of each competition, run
competition by competition. On these two-item groups a within-competition
correlation is the share of competitions ordered correctly, rescaled to
[-1, 1]. It measures the extremes, so it overstates what the full range would
give.

### D1: answering at once carries nothing

160 items × 8 attempts:

| feature | honest b | strong-tier b | 2025 contests (110) | 2026 contests (50) |
|---|---|---|---|---|
| top_share (primary) | -0.01 [-0.13, +0.12] | +0.02 [-0.10, +0.14] | +0.08 [-0.07, +0.23] | -0.23 |
| answer entropy | -0.03 [-0.15, +0.10] | +0.02 [-0.09, +0.13] | +0.06 [-0.09, +0.22] | -0.26 |
| answer log-prob | +0.01 [-0.10, +0.14] | +0.07 [-0.04, +0.18] | +0.05 [-0.12, +0.22] | -0.06 |
| modal answer's log-prob | -0.03 [-0.16, +0.11] | +0.03 [-0.10, +0.17] | +0.00 [-0.18, +0.19] | -0.09 |
| token log-prob | +0.07 [-0.09, +0.22] | +0.15 [+0.03, +0.28] | +0.13 [-0.08, +0.33] | -0.07 |
| token entropy | +0.04 [-0.11, +0.20] | +0.12 [-0.00, +0.24] | +0.15 [-0.03, +0.33] | -0.22 |
| first-token entropy | -0.01 [-0.14, +0.12] | +0.04 [-0.10, +0.19] | -0.02 [-0.16, +0.13] | +0.01 |
| answer length | -0.06 [-0.20, +0.09] | -0.12 [-0.29, +0.05] | -0.01 [-0.21, +0.17] | -0.15 |

The 2026 intervals come from a bootstrap over five competitions and are too
narrow to quote.

Graded accuracy is 2.3% (2.4% on the 2025 contests, 2.0% on the 2026 ones). The
answers are guesses:

* on 45% of the items the modal answer is a default: 0, 1, 2, 3, 2024 or 2025;
* the mean agreement is still 0.82;
* top_share is stable: 15 problems appear under two competition names (AIME
  2025 and its I/II halves), and top_share correlates 0.83 between the two
  copies.

What repeats is the guess, not anything about the item.

On the same 160 items, the prompt's length in characters reaches +0.27 [+0.13,
+0.38], more than any attempt feature. Over all matharena items, within their
item_features groups, log length was +0.06 ("Item covariates with a known
sign"): the text-only, short-answer probe items are a kinder subset.

References on the same items:

| reference (+ = harder) | rho | note |
|---|---|---|
| the 4B judge's digit rating, negated | -0.31 [-0.49, -0.13] | wrong sign: the judge rates the harder problems as likelier to be solved |
| weak tier's mean success, negated | +0.89 [+0.76, +0.97] | those responses are in the target: a ceiling, not an honest number |
| Qwen3-4B-2507-Think's success, negated (70 items) | +0.74 [+0.65, +0.82] | the same 4B family with full thinking, as a matharena subject; also in the target; the interval resamples 8 competitions and is indicative |
| strong-tier b | +0.89 [+0.81, +0.95] | the two targets agree |

### D2: a short chain of thought is at the floor

32 items × 4 attempts:

* **Truncation.** Every attempt hit the 512-token cap, even on the easiest AIME
  problems, whatever the prompt asked. 99% of the answers are forced ones.
* **Accuracy.** Graded accuracy is 7.8%: 16% on the easier item of each
  competition and 0 on the harder one. It is 11.4% on the 2025 contests and 0 on
  the 2026 ones.

| feature | honest b (= strong-tier b) | 2025 contests (11) | 2026 contests (5) |
|---|---|---|---|
| top_share (primary) | -0.40 [-0.80, +0.07] | -0.50 | -0.20 |
| answer entropy | -0.40 [-0.80, +0.07] | -0.50 | -0.20 |
| answer log-prob | +0.00 [-0.50, +0.50] | +0.09 | -0.20 |
| modal answer's log-prob | +0.00 [-0.50, +0.50] | +0.09 | -0.20 |
| token log-prob | +0.62 [+0.25, +1.00] | +0.46 | +1.00 |
| token entropy | +0.62 [+0.25, +1.00] | +0.46 | +1.00 |
| graded success (diagnostic, not label-free) | +0.50 [+0.25, +0.71] | +0.60 | n/a |

The first token's entropy is not a feature here: a chain of thought always
starts the same way ("We are given …"), and the entropy is below 1e-4 on 24 of
the 32 items and at most 0.025 on the rest.

The primary feature has the wrong sign. On the harder items the forced answers
collapse onto a default (27 of 64 are 0 and 9 are 1), so the attempts agree
most where the model is most lost.

What orders the pairs is the reasoning's own uncertainty. Mean token entropy
and log-prob put the harder item above the easier one in 13 of the 16
competitions, including all 5 of the 2026 ones.

* **One attempt suffices.** A single attempt gives the same ordering: +0.62,
  +0.62, +0.62 and +0.75 for the four attempts taken alone.
* **Answering at once shows none of it.** D1's token entropy on the same 32
  items is -0.25.
* **Prompt length explains part of it.** The prompt's length orders 11 of the 16
  pairs (+0.38 [-0.12, +0.75]), and the token entropy follows the prompt's
  length within competition (+0.50 [+0.12, +0.88]).
* **It was found, not declared.** It is the best of six features after
  agreement, the declared one, failed. 13 of 16 has a one-sided sign-test p of
  0.011 uncorrected, and it is measured on extremes.

The judge's digit rating on the same 32 items is -0.62: wrong-signed again.

### Verdict: FLOOR

By the plan's rule the call is FLOOR:

* graded accuracy is 2.3% for D1 and 7.8% for D2, both under 10%;
* the hard half of D2 is at 0%.

The designs that fit on this machine cannot show whether the 4B's attempts
carry difficulty. Answering at once is a clean null (160 items, every |rho| ≤
0.07 against honest b). A 512-token chain of thought never reaches an answer by
itself.

**The GO numbers were met, by a post hoc feature, and do not count.** The
plan's GO clause reads "the best label-free feature": rho ≥ 0.35 with the
right sign, a lower bound above 0.15, and rho ≥ 0.25 on the 2026 contests. D2's
token log-prob and token entropy give +0.625 [+0.25, +1.00], and +1.00 on the
2026 contests (`results/attempt_probe.json`, decision), so on paper they meet
all three. They are discounted for four reasons:

* **Extreme groups.** The 32 items are the easiest and the hardest of each
  competition. On two-item groups the statistic is a 13-of-16 sign test,
  rescaled, and an extreme-group design inflates rho over what the full range
  would give.
* **Prompt length.** The token entropy follows the prompt's length within
  competition at +0.50 [+0.12, +0.88], and length alone orders 11 of the 16
  pairs.
* **Selection.** It is the best of six features, picked after the declared one
  (agreement) failed. With a Bonferroni correction for six, the sign test's
  one-sided p of 0.011 becomes 0.064.
* **The floor.** At 7.8% graded accuracy (0% on the hard half) the design is
  below the plan's 10%, where the attempts are mostly forced guesses.

FLOOR takes precedence over GO because it says the design cannot measure what
GO asks: under 10% accuracy a feature of the attempts reads the model's
confusion, not the item. The plan does not state that precedence; it is
stated here. Both calls route to the plan's step 8, so nothing changes in
practice.

Nothing goes to the harness and no covariate file is written. The
`--stage covariate` export exists for a later GO.

For the next step:

* **Solving is where the signal is.** The Think variant of the same 4B, solving
  with full reasoning as a matharena subject, reaches +0.74 against the
  difficulty its responses help fit.
* **The lead is the token entropy of a truncated attempt.** It is label-free,
  needs k = 1, and does not depend on parsing an answer, so it would apply to
  any text task. That is the "token-entropy profile" of the plan's step 7.
* **What a cheap test needs.** One attempt per item on all 528 pool items, with
  the per-token entropies stored so that the shortest useful prefix can be
  found. Prompt length has to be controlled for, and it should be scored
  against the other benchmarks' honest difficulty before it enters the harness.
  With the per-step speed above, that is about 5-6 hours on this machine.

The FLOOR escalation to a stronger proxy on a GPU is reported here, not taken.
It was taken later, on Kaggle ("Strong model on Kaggle: Qwen3-14B rubric and
attempts").

**Deployment cost, had it passed.** A hidden run holds about 1,000
subject-item pairs over 7 benchmarks. Only short-answer mathematics would be
attempted: about 140 items a run, or about 280 if two such benchmarks are
drawn.

* **D1** costs, for 1,000 items, one prefill of about 300 tokens each (0.3 M in
  all) and about 0.03 M generated tokens. That is about a minute on one
  data-centre GPU with transformers.
* **A truncated attempt** (k = 1, 512 tokens) is about 0.5 M generated tokens
  per 1,000 items, several minutes batched. D2 as run is about 2 M.
* **The multiplier.** Workers are recreated at every checkpoint and keep no
  state, so all of it repeats 6 times unless a disk cache survives, and every
  worker loads the model.

Here, D1 took 2.9 s an item and D2 about 3 minutes an item.

## Entropy profiles and hidden-state probes

`python experiments/hidden_state_probe.py --stage sample`, then `--stage
extract --resume` (36 minutes for 608,366 prompt tokens at 283 tokens a second,
the only language-model job on the machine, at most 3.2 GB of MPS memory),
`--stage heads` (3 minutes), `--stage harness` and `--stage reference` side by
side (15 and 5 minutes, one process each, about 0.5 GB), then `--stage
verdict` and `--stage show`. Every number below is in `results/hidden_state_probe.json`. The
features are in `data/features/probe/` (42 MB, gitignored). The harness stage
reads the stored rows of `experiments/harness.py` (library 3f75a549673aae6a) and
never recomputes hier.

The 4B judge's digit and its attempts carried nothing that transfers ("The 4B
judge, closed out", "Attempting instead of judging"). This probe reads the same
model's internal state instead: Qwen3-4B-Instruct-2507, one forward pass per
item, nothing generated.

* **The prompt** is Agent Psychometrics' instructed form (Ge et al.,
  arXiv:2604.00594): one user turn holding the task text and then "How
  difficult is the above task for an AI agent?", with the assistant header
  appended. The task text is item_content, cut to 1,536 tokens by
  `llmfeat.head_tail` (head, marker, tail), then a newline and item_features.
* **(a) The token-entropy profile** of item_content, the family that
  transferred best for Krsteski & Meyer (arXiv:2608.05797). At every content
  token the model's full next-token distribution gives an entropy and a
  surprisal. The profile statistics are:
  * the mean and sd;
  * the nine deciles;
  * the slope over relative position;
  * the mean absolute step between neighbouring tokens.

  The surprisal gets seven of these statistics.
* **(b) Hidden states.** At the last prompt token (where the answer would
  start) the state is read after layers 9, 18 and 27 and after the final norm
  (36). The mean over the content span is read after layer 18 and after the
  final norm.

**Items.** The sample is stratified: quotas per item_features group, and a
systematic sample over honest difficulty within each group.

| parent | unique items sampled | of | item rows | groups | median content tokens |
|---|---|---|---|---|---|
| matharena, text-bearing | 400 | 974 | 439 | 19 competitions | 134 |
| multi_swebench | 400 | 2,078 | 403 | 8 languages | 314 (17 truncated) |
| real_webagents | 233 | 233 | 233 | 12 websites | 23 |
| researchcodebench | 212 | 212 | 212 | 20 papers | 1,526 (206 truncated) |

The 1,287 item rows carry x on 51% of the test-like evaluated items (47%
mix/whole, 42% and 39% on public R1).

**Checks.**

* **Against the rating extraction.** The mean surprisal of item_content
  correlates 0.87 with the rating extraction's nll on matharena (400 items) and
  0.99 on multi_swebench (374 items). That is the same 4B under another prompt,
  over the whole task text.
* **Against a single-item forward pass.** Six items were run through the
  streamed model one at a time, against the batched shard. Entropy agrees
  within 0.015 nats and surprisal within 0.12 nats at the worst token. The last
  state agrees within 0.17 on values up to 54. That is fp16 batching noise.

**Heads.** The target is the fold-averaged (≈ in-sample) difficulty: Rasch b
fitted without each of the five subject folds, averaged over the folds, then
standardised within benchmark. The heads train on it and are scored against
it. It is essentially the in-sample b, so the r below is slightly optimistic
against the gate table's fold-specific r ("The 4B judge, closed out", Against
honest difficulty); the bias favours the heads. Heads are linear and fitted leave one benchmark
out over the four parents:

* every benchmark weighs the same;
* features are standardised within benchmark (a benchmark's items are visible
  at run time, and no label is used).

Hyperparameters are nested. For held-out parent q, each configuration is fitted
on two of the other three parents and scored on the third. The inner mean
within-benchmark Pearson picks the configuration, which is then refitted on all
three.

Two heads were declared primary before any result was seen:

* **entropy**: ridge on the 13 entropy statistics;
* **hidden**: the last-token state standardised per dimension, PCA to k ≤ 32 on
  the training parents, then ridge. The layer is part of the nested choice.

The secondary heads are surprisal, profile (both profiles), hidden_mean (the
span means), and the hidden head at each layer fixed.

### Leave one benchmark out

Pearson over the held-out benchmark, with 95% intervals from a bootstrap over
item_features groups (multi_swebench's resample 8 languages and are
indicative). The last column is a DerSimonian-Laird random-effects mean over
the four parents (Fisher z), with its 95% prediction interval for a new
benchmark; with four units that interval is itself indicative.

| head | matharena | multi_swebench | real_webagents | researchcodebench | mean | positive | random effects [PI] |
|---|---|---|---|---|---|---|---|
| entropy (primary) | -0.05 [-0.16, +0.05] | +0.05 [-0.06, +0.11] | -0.35 [-0.52, -0.14] | +0.15 [-0.12, +0.38] | -0.05 | 2/4 | -0.05 [-0.75, +0.70] |
| hidden (primary) | +0.19 [-0.02, +0.37] | -0.04 [-0.14, +0.13] | +0.19 [+0.07, +0.31] | +0.14 [-0.15, +0.35] | +0.12 | 3/4 | +0.12 [-0.40, +0.58] |
| surprisal | -0.15 [-0.40, +0.09] | -0.13 [-0.25, -0.03] | -0.37 [-0.49, -0.18] | -0.07 [-0.30, +0.21] | -0.18 | 0/4 | -0.18 [-0.60, +0.32] |
| profile | -0.14 [-0.27, -0.03] | -0.01 [-0.13, +0.07] | -0.37 [-0.53, -0.16] | -0.07 [-0.30, +0.20] | -0.15 | 0/4 | -0.15 [-0.69, +0.50] |
| hidden_mean | -0.21 [-0.43, -0.04] | -0.10 [-0.20, -0.02] | -0.05 [-0.33, +0.23] | +0.11 [-0.18, +0.38] | -0.06 | 1/4 | -0.07 [-0.57, +0.47] |
| layer 9 | +0.37 [+0.15, +0.52] | -0.04 [-0.14, +0.13] | +0.00 [-0.17, +0.18] | +0.14 [-0.15, +0.35] | +0.12 | 3/4 | +0.12 [-0.71, +0.81] |
| layer 18 | +0.12 [-0.12, +0.31] | -0.07 [-0.17, +0.01] | +0.19 [+0.07, +0.31] | +0.04 [-0.19, +0.28] | +0.07 | 3/4 | +0.07 [-0.42, +0.53] |
| layer 27 | +0.19 [-0.02, +0.37] | -0.01 [-0.12, +0.09] | +0.07 [-0.11, +0.27] | +0.33 [+0.17, +0.55] | +0.14 | 3/4 | +0.14 [-0.48, +0.67] |
| layer 36 | -0.05 [-0.25, +0.16] | -0.05 [-0.17, +0.04] | -0.08 [-0.24, +0.06] | +0.08 [-0.12, +0.34] | -0.03 | 1/4 | -0.03 [-0.17, +0.11] |

The same predictions within item_features groups (Pearson, demeaned within
group). This is what a covariate can add to hier, which learns the groups'
levels from labels:

| head | matharena | multi_swebench | real_webagents | researchcodebench | mean |
|---|---|---|---|---|---|
| entropy | +0.06 | +0.05 | -0.26 | +0.01 | -0.03 |
| hidden | +0.28 | -0.05 | +0.17 | -0.02 | +0.10 |
| surprisal | +0.02 | -0.13 | -0.23 | +0.16 | -0.04 |
| hidden_mean | -0.07 | -0.09 | +0.04 | +0.18 | +0.01 |
| layer 27 | +0.28 | +0.01 | +0.04 | +0.46 | +0.20 |

* **Neither primary head reaches the bar.**
  * The entropy head averages -0.05 and is positive on 2 of 4 parents; its
    Spearman mean is -0.07.
  * The hidden head averages +0.12 and is positive on 3 of 4, but no parent
    reaches 0.2. multi_swebench is -0.04.
* **The nested layer choice does not settle.** It picked layer 27, 9, 18 and 9
  for the four held-out parents. The inner criteria were 0.14-0.27, from
  training on two benchmarks at a time.
* **The best secondary head is post hoc and still short.** Layer 27 fixed gives
  +0.14 over the benchmark and +0.20 within group. That rests on
  researchcodebench (+0.46) and matharena (+0.28); multi_swebench gives +0.01
  and real_webagents +0.04. It is one of nine heads, picked after the fact.
* **The surprisal heads transfer backwards.** The surprisal and profile heads
  are negative on all four parents (random-effects mean -0.18 [-0.29, -0.07] for
  surprisal). Fitted on three benchmarks, they order the fourth the wrong way:
  the heterogeneity between units that sank the 4B judge.
* **Inside a benchmark the states carry what the embeddings carried.** A 5-fold
  fit within each benchmark (items at random) reaches:
  * over the benchmark: 0.69, 0.19, 0.40 and 0.60 for layer 27, and 0.59, 0.14,
    0.43 and 0.31 for the hidden head's usual configuration;
  * within group: 0.39, 0.17, 0.26 and 0.25 for layer 27.

  Much of that is group identity. As with the embeddings ("Neural embeddings do
  not carry difficulty to an unseen benchmark"), a hidden run holds about one
  pair per benchmark, so this signal cannot be learned there.
* **One statistic keeps its sign, too weakly to matter.** The entropy profile's
  slope (entropy rising towards the end of the task) has the same sign within
  group on all four parents: +0.11, +0.09, +0.08, +0.14 Spearman. It is the only
  one of the 20 statistics that does, it was found rather than declared, and a
  transferred slope needs r ≈ 0.3.
  * The location statistics (mean, deciles) mostly go negative on matharena,
    real_webagents and researchcodebench and positive on multi_swebench.
  * Within group and net of log length, the entropy head's partial Spearman is
    -0.03, +0.04, -0.10 and +0.10, and the hidden head's +0.25, -0.04, +0.10
    and +0.17.

### Through the harness

Each head's out-of-fold predictions go through the harness as a covariate. On
parent q, x is the head fitted without q. The recipe is `llm4b_close.py`'s:

* x standardised within benchmark (the B0 term reads raw x);
* the nested lines and the forced ones;
* a placebo: x permuted within benchmark, three draws.

Test-like ALC differences against the shipped hier, ± pair-cluster SE, with the
folds nested selection switched on:

| head | transferred nested | per-pair nested | transferred from B1, forced | per-pair s = 0.5 from B7, forced | B0 term, forced | within-pair r |
|---|---|---|---|---|---|---|
| entropy | +0.00098 ± 0.00029 (1/4) | +0.00003 (1/4) | +0.00111 ± 0.00028 | +0.00062 | +0.00004 | -0.02 |
| hidden | 0 (0/4) | -0.00000 (3/4) | +0.00008 ± 0.00015 | +0.00050 | -0.00003 | +0.01 |
| surprisal | 0 (0/4) | +0.00001 (3/4) | +0.00022 ± 0.00031 | +0.00050 | -0.00003 | -0.10 |
| profile | +0.00051 ± 0.00020 (1/4) | -0.00000 (3/4) | +0.00021 ± 0.00027 | +0.00035 | +0.00002 | -0.07 |
| hidden_mean | 0 (0/4) | 0 (0/4) | +0.00019 ± 0.00006 | +0.00105 | +0.00124 | -0.06 |
| layer 9 | 0 (0/4) | +0.00000 (3/4) | -0.00003 ± 0.00009 | +0.00048 | +0.00005 | -0.01 |
| layer 18 | -0.00021 ± 0.00019 (3/4) | -0.00001 (3/4) | -0.00044 ± 0.00022 | +0.00052 | -0.00001 | +0.02 |
| layer 27 | 0 (0/4) | +0.00011 ± 0.00010 (3/4) | +0.00124 ± 0.00053 | +0.00024 | -0.00003 | +0.07 |
| layer 36 | +0.00029 ± 0.00008 (1/4) | -0.00004 (4/4) | +0.00088 ± 0.00022 | +0.00028 | +0.00003 | +0.01 |

The placebo's per-pair s = 0.5 from B7 costs +0.0007 to +0.0011 for every head.
Its nested lines stay within ±0.0001.

For comparison, the reference stage degrades the honest difficulty to r on
exactly these items, with 0 elsewhere (four draws):

| r | transferred nested | per-pair nested | transferred from B1, forced |
|---|---|---|---|
| 0.2 | -0.00018 | 0 | -0.00049 |
| 0.3 | -0.00116 | -0.00009 | -0.00126 |
| 0.5 | -0.00381 | -0.00085 | -0.00381 |

* **Nothing gains.** No line of any head reaches -0.001. The best is layer 18's
  transferred slope, forced, at -0.00044 ± 0.00022, and its nested line gives
  -0.00021 ± 0.00019 (selection-aware SE 0.00032). Layer 18 is a secondary head
  whose LOBO mean is +0.07.
* **Within a test-like pair the heads order nothing.** The mean correlation
  with honest difficulty over a pair's evaluated items is between -0.10 and
  +0.07. The reference just clears -0.001 at this coverage with r = 0.3 over
  the whole parent, which the harness table puts at about 0.25 within a pair.
* **A slope carried between parents costs.** Layer 27's forced transferred
  slope costs +0.0012 test-like and +0.0050 on real_webagents: it is fitted
  where researchcodebench's and matharena's relation dominates and applied
  where there is none. The entropy head's costs +0.0046 on researchcodebench.
  Where nested selection switches the transferred slope on in a single fold,
  it loses there: +0.0046 (entropy), +0.0024 (profile) and +0.0014 (layer 36),
  all on researchcodebench. Layer 18's, on in 3 of 4 folds, gains -0.0002
  (-0.0018 on real_webagents, +0.0003 on multi_swebench).
* **The per-pair slope pays its noise cost.** Forced at s = 0.5 from B7, every
  head costs +0.0002 to +0.0011. That is a little less than its placebo,
  except hidden_mean's (+0.00105 against +0.00089).

### Verdict: replicated null

The keep rule has two prongs, and each primary head must pass both:

* **r prong:** a mean leave-one-benchmark-out within-benchmark Pearson r of at
  least 0.2, positive on at least 3 of the 4 parents;
* **ALC prong:** a nested harness test-like difference of at most -0.001, from
  the transferred or the per-pair slope.

Both prongs fail for both primary heads:

* **entropy:** r = -0.05 with 2 of 4 positive; the nested lines give +0.00098
  and +0.00003.
* **hidden:** r = +0.12 with 3 of 4 positive, none at 0.2; the nested lines
  give 0 and -0.000001.

No secondary head passes either. The token-entropy profile and the instructed
last-token state of Qwen3-4B-Instruct-2507 are closed as item covariates, and
nothing ships. `experiments/hidden_state_probe.py` stays research-only.

**Against the literature** (numbers as the step-2 review summarised them; not
re-read here):

* **Krsteski & Meyer's entropy profile** reached 0.137 leave-one-benchmark-out
  Spearman on 17 agentic benchmarks, against 0.40-0.48 within benchmark. Here
  the entropy head gives -0.07 over four held-out benchmarks. Their pooled
  number also carries level differences between benchmarks, which hier learns
  from labels.
* **Agent Psychometrics' instructed embedding** reached a new-benchmark AUC of
  0.70-0.74. That AUC pools responses, subject ability included, so the plan
  forbids comparing it with a within-benchmark r (its rule 5).
* **Probes find difficulty inside a dataset** (Lugoloobi & Russell,
  arXiv:2510.18147). The within-benchmark fits here (0.59-0.69 on matharena)
  agree. They are largely group identity, and they do not carry to an unseen
  benchmark.

**What this does not test:**

* the entropy of a generated attempt, the attempt probe's lead (a reasoning
  trace, not the reading of the task; measured later with a 14B, "Strong model
  on Kaggle: Qwen3-14B rubric and attempts");
* a larger reader;
* a fine-tuned head (LoRA);
* swe_rebench as a fifth unit;
* the two large parents beyond their 400-item samples.

**Caveats.**

* **Few units.** Four units cannot tell a transferable r of 0.2 from 0. The
  random-effects prediction intervals span about ±0.5.
* **Two training benchmarks per inner fold** make the nested choice noisy (the
  layer above).
* **Standardisation over the sample.** x is standardised within benchmark over
  the sampled items; a run-time predictor sees only the run's items.
* **A second-order leak.** The transferred slope for held-out q is fitted on
  the other parents' out-of-fold x, whose heads saw q's items. That favours the
  covariate, and the result is null anyway.
* **researchcodebench's truncation.** The head of its content is the paper's
  LaTeX preamble, so its entropy profile mostly reads LaTeX.
* **fp16.** The states and profiles carry fp16 batching noise (Checks, above).

## Few-shot prompting

`python experiments/icl_probe.py --stage select` (seconds), then `--stage run
--passes core` (41 minutes), `--stage run --passes perm,half` (51 minutes)
and `--stage analyse`; `python experiments/pairwise_probe.py --stage plan`, then
`--stage run` (13 minutes) and `--stage analyse`. The runs used the one
language-model slot in turn, as the only such job on the machine; the analyses
take seconds on one process. Every number below is in `results/icl_probe.json`
and `results/pairwise_probe.json`. The raw scores are in `data/icl_probe/` and
`data/pairwise_probe/` (gitignored).

Both probes read Qwen3-4B-Instruct-2507 in the attempt probe's weight-only int8
(4.0 GB; "Running a 4B generator on the 16 GB M1"): one forward pass per prompt,
nothing generated. The in-context probe reads the stored rows of
`experiments/harness.py` (library 3f75a549673aae6a) and never recomputes hier.

The protocol allows few-shot evidence only through the pair's own revealed
labels, at most 31. Two probes test what a small model makes of them. They are
the plan's step 10, and both kill rules were declared before the runs.

* **In-context learning.** Shown the items this subject solved and failed so
  far, does the 4B order the subject's remaining items better than without
  them?
* **Pairwise comparisons.** Asked which of two items of one benchmark is
  harder, how often is the 4B right, and do its errors stick to items?

### In-context learning over the pair's labels

**Appearances.** Test-like pair appearances from the harness rows (runs 0 to
299), at budget 15: the pair's first 15 acquired labels in the platform's
order, and its evaluated items. An appearance is eligible when:

* 2 to 13 of its 15 labels are successes (a prompt with one class teaches
  nothing about items);
* it has at least 20 evaluated items, subsampled to 45 in a hash order, with at
  least 4 solved and 4 failed among them (a within-pair correlation with the
  outcome needs both);
* at least 80% of its items carry text (no image placeholder; on matharena also
  at least 70 characters, the rule of "The 4B judge, closed out").

1,236 of the 2,425 appearances are eligible. Four per parent were drawn in a
seeded hash order, with distinct subjects: 16 appearances, 714 evaluated items.
The rule keeps pairs whose rate is away from 0 and 1, where the order of items
matters most, so it is kind to the method. Label rates run from 0.13 to 0.87,
and evaluated rates from 0.11 to 0.83.

**Prompts.** One user turn, under the system line "You are an expert at
predicting which tasks a particular AI system can and cannot solve":

* **icl**: "One AI system attempted each of them once", then the 15 labeled
  items as examples in acquisition order, each cut to 192 tokens and marked
  SOLVED or FAILED, then "judge a new task from the same benchmark", the
  target, and "Did this system solve this task? Answer Yes or No."
* **zs**: the zero-shot control on the same template, with the target alone.
* **perm**: the 15 examples with their outcomes permuted (seeded). The base
  rate is kept and the link between item and outcome is broken. This separates
  learning from the labels from seeing the benchmark's items and its rate.
* **half**: each half of the examples (7 or 8) on its own, the two scores
  averaged: a dose between 0 and 15 examples.

The target is cut to 384 tokens. The score is log p(Yes) - log p(No) of the
first assistant token; the two tokens hold all of the next-token probability
(mass 1.000 in every variant). The prompt's prefix is run once and its cache
reused for each target. For a per-pair slope, the 15 labeled items are also
scored cross-fitted: each half by the prompt whose examples are the other half.

**Measures.** Per appearance, over its evaluated items:

* **r_diff**: Pearson with honest easiness. Easiness is minus the Rasch b fitted
  on the parent's subjects outside the pair's subject fold (the harness's
  honest oracle), so the subject's own responses never enter it. This is the
  harness value map's index, and the kill rule reads it.
* **Outcome**: Spearman with the pair's outcome K/N (matharena has four
  responses per item), and the AUC of solved against failed items.
* **Net of hier**: r_diff net of a linear fit on hier's B15 logit for the same
  items. That is what the score adds to the model that ships.

Means over the 16 appearances, ± the SE over appearances. With 4 per parent,
the mean is also benchmark-equal.

| score | r_diff | r_diff net of hier | Spearman with outcome | AUC | r_diff: matharena / multi_swebench / real_webagents / researchcodebench |
|---|---|---|---|---|---|
| icl, 15 examples | +0.206 ± 0.040 | +0.127 ± 0.050 | +0.209 ± 0.032 | 0.607 | +0.26 / +0.18 / +0.21 / +0.17 |
| half, 7 or 8 examples (two prompts averaged) | +0.204 ± 0.029 | +0.130 ± 0.033 | +0.192 ± 0.033 | 0.594 | +0.24 / +0.16 / +0.21 / +0.21 |
| perm, 15 examples, outcomes permuted | +0.110 ± 0.040 | +0.071 ± 0.042 | +0.123 ± 0.032 | 0.560 | +0.15 / +0.11 / +0.08 / +0.10 |
| zs, no examples | +0.048 ± 0.043 | +0.058 ± 0.041 | +0.041 ± 0.043 | 0.528 | +0.12 / +0.16 / -0.02 / -0.07 |
| hier's B15 prediction | +0.150 ± 0.042 | | +0.198 ± 0.038 | 0.597 | +0.11 / +0.08 / +0.19 / +0.21 |
| honest easiness itself | 1 | | +0.617 ± 0.039 | 0.904 | |

Paired differences on the same items:

| difference | r_diff | r_diff net of hier | Spearman with outcome |
|---|---|---|---|
| icl - zs | +0.158 ± 0.059 | +0.069 ± 0.057 | +0.169 ± 0.052 |
| icl - perm | +0.096 ± 0.035 | +0.056 ± 0.046 | +0.086 ± 0.032 |
| perm - zs | +0.062 ± 0.059 | +0.013 ± 0.063 | +0.083 ± 0.044 |
| icl - half | +0.002 ± 0.028 | -0.003 ± 0.032 | +0.017 ± 0.027 |
| half - zs | +0.156 ± 0.052 | +0.072 ± 0.044 | +0.152 ± 0.052 |

* **The labels help.** icl - zs is +0.158 ± 0.059 in r_diff. By parent it is
  +0.15, +0.02, +0.23 and +0.24, and it is positive on every parent in the
  outcome correlation. Zero-shot, the 4B orders nothing: +0.05 ± 0.04, negative
  on real_webagents and researchcodebench, as the zero-shot judge was ("The 4B
  judge, closed out").
* **Most of the gain is the mapping, not the context.** With the outcomes
  permuted, r_diff is +0.110. icl - perm is +0.096 ± 0.035, positive on every
  parent (+0.12, +0.07, +0.13, +0.07). So about 60% of the gain over zero-shot
  comes from which items were solved. The rest comes from seeing the
  benchmark's items (perm - zs +0.062 ± 0.059).
* **Seven examples already give all of it.** With half the examples (7 or 8
  per prompt) r_diff is +0.204 ± 0.029; icl - half is +0.002 ± 0.028. The
  score's order saturates by about 7 examples.
* **It stays below 0.3.** r_diff is below 0.3 on every parent, and above it on
  4 of the 16 appearances (the best +0.52, on a real_webagents stratum).
* **Much of it is what hier already reads from the same labels.** Net of hier's
  B15 logit, r_diff falls from +0.206 to +0.127. hier's own within-pair order
  gives +0.150 on these items, from item_features groups and floors. Net of
  hier, the mapping's part (icl - perm) is +0.056 ± 0.046 in r_diff and +0.016
  ± 0.033 in the Pearson correlation with the outcome. On researchcodebench,
  where three of the four pairs are wholes over many papers, the net r_diff is
  +0.01: the prompt learns which papers the subject solves, as hier's group
  effects do.
* **It does not read the subject's level.** Under icl the mean P(Yes) is 0.61
  to 1.00 while the evaluated rates run from 0.11 to 0.83. Across the 16
  appearances that mean correlates +0.57 with the evaluated rate (+0.48 with
  the outcomes permuted). The share of solved labels itself correlates +0.84,
  and hier's mean B15 prediction +0.83. Zero-shot, the 4B says No to every
  researchcodebench task (mean P(Yes) at most 0.002), and its mean correlates
  +0.14 with the rate.

### What in-context scores would be worth

**Through the value map.** The harness's honest gate table maps a covariate's
mean within-pair r to test-like ALC ("The gate table"). Read by linear
interpolation at each score's r_diff (`implied_alc` in the results file), and,
in the rows marked "net", at its r_diff net of hier: the in-context scores read
the same 15 labels hier reads, so what they can add to hier is the net r, not
the raw one. (The net rows are the same interpolation in
`results/harness_thresholds.json`'s honest table, at the net r_diff of
`results/icl_probe.json`; they are not stored.)

| score | r_diff | transferred nested | per-pair nested | transferred from B1, forced | transferred from B7, forced | per-pair s = 0.5 from B7, forced |
|---|---|---|---|---|---|---|
| icl | 0.206 | -0.0017 | -0.0002 | -0.0018 | -0.0011 | -0.0001 |
| icl, net of hier | 0.127 | -0.0005 | -0.0001 | -0.0007 | -0.0004 | +0.0004 |
| half | 0.204 | -0.0017 | -0.0002 | -0.0018 | -0.0010 | -0.0001 |
| half, net of hier | 0.130 | -0.0005 | -0.0001 | -0.0007 | -0.0004 | +0.0004 |
| perm | 0.110 | -0.0004 | 0 | -0.0005 | -0.0003 | +0.0005 |
| zs | 0.048 | -0.0001 | 0 | -0.0001 | -0.0001 | +0.0007 |

Both nested lines are short of the gate's -0.002. At the net r the
transferred line is -0.0005, a third of the raw reading, and the direct B15
check below agrees with the lower one. The table's draws pass the
gate on 1 of 8 at within-pair r 0.16 and on 6 of 8 at 0.25, so a single
covariate at 0.21 would pass roughly half the time with a transferred slope.
With a per-pair slope no draw passes up to within-pair r 0.33. The reading is an
upper bound twice over:

* The table's covariates act at every budget from B1. In-context scores at B1
  and B3 see 1 and 3 examples. From 7 examples on they reach their full r
  (the half row), so the transferred slope switched on at B7, -0.0011, is the
  safer reading.
* The selection keeps the mid-rate pairs with text.

One thing points the other way. r_diff counts only what the score shares with
difficulty, and an in-context score could also carry the subject's own pattern.
The next check counts that too.

**A direct check at B15.** The score enters as a logit offset on hier's B15
prediction for the same evaluated items, centred over them and capped as in the
harness. The slope is either transferred (fitted on the other parents'
appearances, on their B15 Brier) or per pair (the harness's MAP from the 15
cross-fitted labels, prior sd 0.5 per within-pair sd of the score). Brier
differences at B15 against hier, over the 16 appearances (± SE):

| x | transferred slope | per-pair slope |
|---|---|---|
| icl | -0.0020 ± 0.0015 | +0.0034 ± 0.0039 |
| half | -0.0033 ± 0.0021 | (no cross-fitted scores) |
| perm | -0.0013 ± 0.0014 | (no cross-fitted scores) |
| zs | +0.0007 ± 0.0006 | +0.0014 ± 0.0019 |
| honest easiness | -0.0765 ± 0.0125 | -0.0502 ± 0.0081 |

* **A transferred slope** gains -0.0020 ± 0.0015 at B15, not distinguishable
  from 0. It gains on multi_swebench (-0.0064) and researchcodebench (-0.0033)
  and loses on real_webagents (+0.0014). At B15's weight of 0.2 that is -0.0004
  of ALC. The half-prompt score gives -0.0033 ± 0.0021, and loses on matharena
  (+0.0038).
* **A per-pair slope** costs +0.0034 ± 0.0039: 15 labels do not fit a slope.
* **For scale**, honest easiness through the same path gains -0.077
  (transferred) and -0.050 (per pair). The in-context score takes 3% of that.

**Cost.** The core pass read 568,168 prompt tokens in 41 minutes (231 a
second), 2.6 minutes per appearance; the two controls read 624,729 tokens in 51
minutes. A formative run evaluates about 1,000 subject-item pairs at each of
five labeled checkpoints: about 1.5 M prompt tokens, or about 1.8 hours here.
On a data-centre GPU that is minutes. Workers are recreated at every
checkpoint, so each would load the model again.

### Verdict: KILL

The rule: kill in-context learning if r_diff(icl) - r_diff(zs) < 0.1 or
r_diff(icl) < 0.3.

* **The gain prong passes.** icl - zs = +0.158 ± 0.059.
* **The level prong fails.** r_diff(icl) = +0.206 ± 0.040, and it is below 0.3
  on every parent.

In-context learning over the pair's own labels with a 4B reader is closed. It
does learn from the labels, which the zero-shot judge could not, and most of
what it learns is the mapping itself. But what it learns mostly duplicates what
hier learns from the same labels, and 7 examples already give all of it. The
value map puts it at about -0.0017 with the nested transferred slope and
-0.0011 from B7 at its raw r, and at -0.0005 and -0.0004 at its r net of hier,
which is what it could add; the direct B15 check (about -0.0004 of ALC) does
not separate it from 0.

### Pairwise comparisons

**Pairs.** For each parent, a pool of 20 items from the hidden-state probe's
stratified sample, taken systematically over honest difficulty (the average over
the five folds). The pool is put in a seeded order and compared along a
circulant graph: 50 comparisons per parent, every item in exactly 5, 200 in all.
25 of them fall inside an item_features group.

**Prompt.** The anchored-comparisons study's prompt: the two tasks, each cut to
512 tokens, then "Which task do FEWER of them solve correctly, that is, which
task is harder? Answer with one letter, A or B." Both presentation orders are
run (400 prompts). The score h is the logit difference between the two letters,
averaged over the orders and oriented so that h > 0 says the first item of the
pair is harder. q is the share of comparisons where the sign of h matches the
sign of the honest difficulty gap; "honest" here is the fold-averaged (≈
in-sample) difficulty of "The 4B judge, closed out", which slightly favours
the comparator. The Wilson intervals below treat the comparisons as
independent, but every item sits in five of them, so they are too narrow; the
item-block bootstrap (pooled [0.429, 0.659]) is the interval to quote, and the
per-parent Wilson intervals are indicative. The equivalent r is sin(pi (q - 1/2)): the
correlation with difficulty that an absolute score of the same q would have
(Greiner's relation, bivariate normal).

| slice | n | q [95% Wilson] | equivalent r | share answered "A" | same item from both orders |
|---|---|---|---|---|---|
| pooled | 200 | 0.540 [0.471, 0.608] | +0.13 | 0.83 | 0.29 |
| matharena | 50 | 0.560 [0.423, 0.688] | +0.19 | 0.87 | 0.26 |
| multi_swebench | 50 | 0.450 [0.321, 0.587] | -0.16 | 0.90 | 0.20 |
| real_webagents | 50 | 0.560 [0.423, 0.688] | +0.19 | 0.55 | 0.70 |
| researchcodebench | 50 | 0.590 [0.452, 0.715] | +0.28 | 1.00 | 0.00 |

The absolute signals on the same comparisons, oriented by their declared sign:

| signal (one call per item) | comparisons | q | the comparator's q on them |
|---|---|---|---|
| hidden-state probe, layer-9 head (best on all 200) | 200 | 0.542 | 0.540 |
| hidden-state probe, entropy head | 200 | 0.532 | 0.540 |
| hidden-state probe, hidden head | 200 | 0.527 | 0.540 |
| log length | 200 | 0.480 | 0.540 |
| 4B judge, digit rating | 90 | 0.600 | 0.511 |
| 4B judge, digit entropy | 90 | 0.556 | 0.511 |

The heads are the probe's out-of-fold predictions, fitted on the other three
parents. The judge rated only matharena and multi_swebench.

* **The 4B does not order items by difficulty.** Pooled q is 0.540 (item-block
  bootstrap [0.429, 0.659]), and 0.45 on multi_swebench. By |gap| tercile q is
  0.50, 0.49 and 0.63: only gaps above 3.5 logits come out at all. Within an
  item_features group q is 0.42 (25 comparisons).
* **It answers by position.** 83% of the prompts are answered with the first
  task, and all of them on researchcodebench. Only 29% of pairs get the same
  item from both orders (none on researchcodebench, 70% on real_webagents).
  Either order alone gives 0.51 or 0.53. Averaging the orders keeps only the
  logit's magnitude.
* **An absolute signal does as well on the same pairs, at a fraction of the
  calls.** The best on all 200 comparisons, the layer-9 head, gives 0.542. The
  4B judge's own digit rating, one call per item, gives 0.600 on its 90 rated
  comparisons, where the comparator gives 0.511.
* **Its errors stick to items.** The residual of h on the gap correlates +0.42
  between comparisons that share an item. Under h = a gap + e_i - e_j + noise,
  that makes 84% of the comparator's error variance a per-item term. The 4B
  compares its own impression of each item, and that impression is not
  difficulty: Bradley-Terry scores fitted from its five comparisons per item
  correlate +0.36, -0.07, +0.23 and +0.17 with honest b over each pool. More
  comparisons per item would not wash that out. The binary errors show the same
  more weakly: the variance of the wrong-answer count per item is 1.17 times
  its permutation null (p = 0.12).

### Verdict: DROP

The rule: drop comparisons if pooled q < 0.60, or if q < the best absolute
signal on the same pairs + 0.03. Both prongs hold: q = 0.540, against 0.60 and
against 0.542 + 0.03. Anchored comparisons with the 4B are dropped. Its errors
are mostly per item, so more comparisons per item would not rescue them.

**Against the literature** (numbers as the step-2 review summarised them; Min
et al. and Wei et al. are cited from memory and not re-read):

* **Small models and demonstration labels.** Min et al. 2022 ("Rethinking the
  Role of Demonstrations", arXiv:2202.12837) and Wei et al. 2023 ("Larger
  language models do in-context learning differently", arXiv:2303.03846)
  report that small models lean on the format and label space of the
  demonstrations more than on the mapping. The permuted control here shows the 4B does read
  the mapping (icl - perm +0.10). What it reads mostly duplicates hier.
* **Generalized Correctness Models** (Xiao et al., arXiv:2509.24988). In-context
  learning with 5 retrieved correctness examples helped Qwen3-32B (+4.6%
  accuracy) but not Qwen3-8B. The 4B here gains +0.16 in r over zero-shot, and
  stays below what would ship.
* **Kolesnikova et al.** (arXiv:2605.18562). With GPT-4o, DeepSeek-V3.2 and
  Qwen3-235B, pairwise judgements beat absolute ones by 0.12-0.15 Spearman
  against human difficulty. The 4B's comparisons are at the absolute signals'
  level here and collapse into a position bias.
* **Ballon et al.** (arXiv:2512.14220). o3 and Gemini 2.5 Pro comparisons track
  human difficulty labels at 0.80-0.82, but the performance of language models
  only at 0.22-0.24. The target here is model performance.
* **Krsteski & Meyer** (arXiv:2608.05797) report a within-benchmark pairwise
  accuracy of about 0.60 leave-one-benchmark-out for their best linear head.
  The 4B comparator's 0.54 is below it.

**What this does not test:**

* a larger reader (the plan's step 8, on a GPU);
* in-context learning at other budgets, or with retrieved rather than
  acquisition-order examples;
* comparisons against labeled anchors from the pair's own items;
* a correction for the comparator's position bias beyond averaging the orders.

**Caveats.**

* **Sixteen appearances and 200 comparisons.** The SEs are over appearances or
  comparisons. Four parents cannot show variation between benchmarks.
* **The selection favours in-context learning** (mid-rate pairs with text), so
  the value-map reading is an upper bound.
* **int8.** The attempt probe measured int8 against fp16 at 0.98-0.99
  correlation on log-probabilities. The scores here were not re-run in fp16.
* **Pools of 20 items.** Bradley-Terry correlations over 20 items have wide
  intervals.

## Fine-tuning an encoder

`python experiments/finetune_encoder.py --stage data` (25 seconds), then
`--stage lora-cache --resume` (12 minutes: 1.45M tokens at 2,023 tokens a
second), `--stage lora --resume` (2 h 58 min for the 12 runs on the M1's MPS,
one process), `--stage eval` (4 minutes, 1 GB), `--stage harness` (4 minutes,
0.7 GB), `--stage verdict` and `--stage show`. The two model stages were the
only language-model job on the machine. Every number below is in
`results/finetune_encoder.json`. The cached states (2.8 GB) and the runs are in
`data/finetune/` (gitignored). The harness stage reads the stored rows of
`experiments/harness.py` (library 3f75a549673aae6a) and never recomputes hier.

The frozen embedding carried nothing to an unseen benchmark ("Neural embeddings
do not carry difficulty to an unseen benchmark"). Every frozen-feature head
since was switched off by nested selection ("Meta-learned heads on frozen
embeddings": 0 of 4 folds for every head, on both row sets). What was left untested is training
the encoder itself, with a loss that asks only for the order of items inside a
benchmark. This is that test, run once. The plan's gate was declared before any
result: GO only if held-out r >= 0.3 on at least 3 of the 4 parents AND a
nested test-like ΔALC <= -0.002. The literature prior was 10-15%: ADeLe's
LoRA-tuned 8B scored 0.692 AUROC out of distribution against a rubric's 0.747.

**The model.** Qwen3-Embedding-0.6B in float32. It reads llmfeat's item text
(item_content, a newline, item_features), cut to 512 tokens with the
end-of-text token: head and tail around the marker, where the frozen features
used 2,048. The pooled embedding is the end-of-text token's final state,
normalised, as in llmfeat.

* **Frozen lower layers.** Layers 0-21 stay frozen. The residual stream
  entering layer 22 is computed once per item and cached in float16.
* **Adapters on the top six layers.** Low-rank adapters, written in plain torch
  (no peft), sit on every linear map of attention and MLP (q, k, v, o, gate, up,
  down): rank 16, alpha 32, B starting at zero. That is 2.2M trained
  parameters, plus a linear head on the pooled embedding. A step runs the six
  layers only, with gradient checkpointing per layer.
* **No adapter dropout.** On MPS, checkpointing does not replay dropout masks.
  With dropout 0.05, gradients with and without checkpointing differed by 0.55,
  against a largest gradient of 0.44. With no dropout they agreed exactly.

**The loss**, per batch of 16 items of one benchmark:

* RankNet over every pair in the batch, each pair weighted by its target gap
  (clipped at 2);
* plus 0.1 × the MSE between the batch-centred scores and the batch-centred
  standardised targets.

A benchmark's level never enters, since the harness standardises x within
benchmark.

**Targets.**

* **Parents:** the fold-averaged (≈ in-sample) difficulty (Rasch b fitted
  without each of five subject folds, then averaged), standardised within
  benchmark. It is essentially the in-sample b, so the held-out r below is
  slightly optimistic against the gate table's fold-specific r ("The 4B judge,
  closed out", Against honest difficulty); the bias favours the encoder.
* **swe_rebench, training only:** minus the logit of the item's smoothed
  success rate over the one subject's roughly 10 trials, on 1,500 items drawn
  at random.
* **Strong-tier difficulty** is reported but never trained on. It is Rasch b
  over the pairs at or above the parent's median ability.
* **In-distribution check (idv):** 15% of each benchmark's items (at most 200)
  are held out of every training run. They show whether training learns
  anything at all.

**Protocol.** Four leave-one-benchmark-out folds over the four parents. For
held-out parent q, each of the other three parents v in turn is the inner
early-stopping benchmark, and the remaining two plus swe_rebench are trained on.
That makes 12 runs, 49-72 steps an epoch.

* **Epoch 0** is the frozen embedding with a ridge head, its alpha chosen on v.
  The ridge also initialises the head: linear probe, then fine-tune.
* **Epochs 1-3** each draw up to 384 items per training benchmark, in
  single-benchmark batches in random order. The optimiser is AdamW: adapters at
  2e-4, the head at 1e-3, weight decay 0.01, 20 warm-up steps then linear
  decay, gradients clipped at 1.
* **Early stopping** keeps the epoch (0-3) with the best Pearson on v. It never
  sees q.
* **The primary ("finetuned")** is q's prediction: the mean of the three inner
  models' predictions, each standardised within q.

**Baselines**, on the same items and targets:

* **frozen_nested:** epoch 0 of the same 12 runs. It uses the same data, the
  same nested choice and the same ensembling, and differs only in the
  training.
* **frozen_lobo512:** a ridge on the 512-token frozen embedding, fitted on all
  three other parents plus swe_rebench, with alpha chosen by inner
  leave-one-benchmark-out.
* **frozen_lobo2048:** the same on the stored 2,048-token embeddings, without
  swe_rebench, which has none stored.
* The published `results/emb_transfer.json` line (in-sample target, 2,048
  tokens) is quoted as it stands.

**The configuration was picked on the in-distribution check only.** Three
settings were piloted for one epoch on one run (held-out researchcodebench,
inner real_webagents), reading only the idv items. The mean change in idv r
over the three training benchmarks was:

| setting | mean change in idv r |
|---|---|
| ridge head, adapters at 2e-4 (the declared setting) | +0.004 |
| zero head | -0.055 |
| ridge head, adapters at 1e-3 | -0.027 |

The declared setting was kept. q's predictions from the pilots were not read.

**Checks.**

* **The frozen pass reproduces the stored embeddings.** Cosine against
  data/features on items short enough that both read the same tokens: minimum
  0.99999 (1,505 matharena, 1,454 multi_swebench, 233 real_webagents items).
* **The adapted model at step 0 is the frozen one.** Through the cached float16
  states, scores correlate 0.99999995 with the frozen embedding's (64 items,
  largest difference 0.0004 on an sd of 0.58).
* **Two MPS out-of-memory crashes** stopped the runs, both in evaluating
  multi_swebench as the inner benchmark: in the eighth run, then in epoch 2 of
  the eleventh. The first lost that run's first epoch and it was redone. The
  second resumed from its epoch-1 checkpoint and redid epoch 2 with the same
  batches. The script now frees memory between runs and before each
  evaluation. The last pass ran with the MPS cap at 0.7 of the recommended
  working set instead of 0.6. Nothing else changed.

### Held-out r

Pearson against honest difficulty over the held-out parent's item_ids, with 95%
intervals from a bootstrap over item_features groups (multi_swebench's
resample 8 languages and are indicative). The last column is a
DerSimonian-Laird random-effects mean over the four parents, with its
prediction interval for a new benchmark; with four units it is indicative.

| predictor | matharena | multi_swebench | real_webagents | researchcodebench | >= 0.3 | random effects [PI] |
|---|---|---|---|---|---|---|
| finetuned (primary) | -0.08 [-0.22, +0.06] | +0.05 [-0.01, +0.10] | -0.09 [-0.32, +0.15] | -0.11 [-0.41, +0.10] | 0/4 | -0.05 [-0.43, +0.35] |
| frozen_nested | -0.07 [-0.20, +0.06] | +0.05 [-0.02, +0.10] | -0.12 [-0.33, +0.14] | -0.09 [-0.38, +0.14] | 0/4 | -0.05 [-0.40, +0.32] |
| finetuned, epoch 3 forced | -0.07 [-0.22, +0.08] | +0.06 [+0.01, +0.11] | -0.09 [-0.30, +0.16] | -0.13 [-0.42, +0.09] | 0/4 | -0.05 [-0.44, +0.36] |
| frozen_lobo512 | -0.03 [-0.15, +0.07] | +0.04 [-0.02, +0.08] | -0.21 [-0.35, -0.01] | -0.01 [-0.30, +0.17] | 0/4 | -0.04 [-0.37, +0.30] |
| frozen_lobo2048 | -0.13 [-0.25, -0.01] | -0.03 [-0.11, +0.01] | -0.13 [-0.32, +0.07] | +0.06 [-0.29, +0.34] | 0/4 | -0.07 [-0.36, +0.24] |
| emb_transfer, published (in-sample target) | -0.16 [-0.29, -0.04] | -0.03 [-0.16, +0.04] | -0.23 [-0.44, -0.01] | -0.02 [-0.32, +0.26] | 0/4 | |

Fine-tuned minus frozen_nested, paired over the same items (group bootstrap):
-0.01 [-0.02, +0.01], +0.00 [-0.01, +0.02], +0.03 [-0.01, +0.06] and -0.02
[-0.11, +0.04]. Against strong-tier difficulty the primary gives -0.11, +0.02,
-0.11 and -0.11 (frozen_nested -0.10, +0.02, -0.13, -0.09). Within
item_features group, Spearman is -0.04, +0.05, +0.01 and -0.03. On
text-bearing matharena items it is -0.07 [-0.19, +0.07].

* **The r prong fails outright.** No parent comes near 0.3. The primary is
  positive only on multi_swebench (+0.05), and its random-effects mean is -0.05.
* **Training changes nothing out of distribution.** The paired difference from
  the frozen control is within ±0.03 on every parent, and every interval spans
  0. The fixed epochs agree: forced to epoch 1, 2 or 3, the mean over parents
  is -0.052, -0.058 and -0.058, against -0.056 for the frozen control.
* **The frozen baselines replicate the old null** on the honest target and on
  both input lengths. Truncating to 512 tokens loses nothing that transferred.

### What the training did learn

The inner and in-distribution curves, over the 12 runs:

| epoch | 0 (frozen ridge) | 1 | 2 | 3 |
|---|---|---|---|---|
| mean r on the inner benchmark v | (baseline) | +0.012 | +0.002 | +0.003 |
| runs whose v improved | | 10/12 | 6/12 | 6/12 |
| runs whose early stopping picked this epoch | 2 | 6 | 2 | 2 |
| idv r, multi_swebench | 0.118 | 0.144 | 0.164 | 0.176 |
| idv r, swe_rebench | 0.125 | 0.127 | 0.149 | 0.161 |
| idv r, matharena | 0.595 | 0.578 | 0.609 | 0.621 |
| idv r, researchcodebench | 0.620 | 0.577 | 0.623 | 0.627 |
| idv r, real_webagents | 0.259 | 0.256 | 0.236 | 0.233 |

The inner row gives the change from epoch 0. Each idv row averages the runs
that trained on that benchmark.

* **It learns inside the benchmarks it trains on.** By epoch 3, idv r rises
  +0.058 on multi_swebench, +0.037 on swe_rebench and +0.026 on matharena: the
  three training sets that fill a whole epoch. It falls 0.026 on real_webagents
  (233 items).
* **None of that reaches a benchmark it did not train on.** The inner benchmark
  gains +0.012 after the first epoch and about nothing after that. The held-out
  parent's paired difference is 0.
* **Early stopping mostly stopped early.** It kept epoch 0 or 1 in 8 of 12
  runs. The training batches' pairwise accuracy stayed between 0.57 and 0.78,
  and rose by at most 0.04 from epoch 1 to epoch 3 in any run. The adapters
  fit their training benchmarks only a little more tightly as training went
  on.

### Through the harness

The out-of-fold predictions as covariates on the stored rows, by
hidden_state_probe's recipe. The primary's placebo (x permuted within
benchmark, three draws) is in the last column. Test-like differences ± the
pair-cluster SE, with the folds where nested selection switched the term on:

| covariate, line | test-like | multi_swebench held out | mix/whole | folds on | gate | placebo |
|---|---|---|---|---|---|---|
| finetuned, transferred nested | +0.00048 ± 0.00010 | +0.00119 | +0.00042 | 1/4 | fail | +0.00002 |
| finetuned, per-pair nested | 0 | 0 | 0 | 0/4 | fail | -0.00000 |
| finetuned, B0 term nested | +0.00020 ± 0.00004 | +0.00049 | +0.00024 | 1/4 | fail | +0.00000 |
| finetuned, transferred from B1 (forced) | +0.00026 ± 0.00013 | +0.00119 | +0.00023 | | | +0.00007 |
| finetuned, per-pair s = 0.5 from B7 (forced) | +0.00061 ± 0.00011 | +0.00035 | +0.00043 | | | +0.00056 |
| frozen_nested, transferred nested | +0.00047 ± 0.00009 | +0.00116 | +0.00041 | 1/4 | fail | |
| finetuned epoch 3, transferred nested | +0.00048 ± 0.00010 | +0.00118 | +0.00041 | 1/4 | fail | |
| frozen_lobo512, transferred nested | +0.00042 ± 0.00008 | +0.00104 | +0.00043 | 1/4 | fail | |

The mean correlation with honest difficulty over a test-like pair's evaluated
items is -0.045 for the primary (-0.055 frozen_nested, -0.060 frozen_lobo512).

* **Nothing gains.** Every nested line is 0 or a loss. The one fold where the
  transferred slope switches on is multi_swebench held out, and there it costs
  +0.0012. On the other three parents x orders items the wrong way (r -0.08 to
  -0.11). Fitted on each of them alone, the transferred slope is +0.08 to +0.19
  logits per unit of x from B1 to B31. Fitted on multi_swebench alone it is
  -0.08 to -0.12. The slope carried to multi_swebench therefore points the
  wrong way there. The frozen control's slopes are the same to within 0.01.
* **The per-pair slope stays off** in every fold. Forced at s = 0.5 from B7 it
  costs +0.0006, the same as its placebo (+0.00056): the noise cost of a slope
  learned from 31 labels on an uninformative x.
* **The fine-tuned lines match the frozen ones** to within 0.00001.

### Verdict: KILL

The gate: held-out r >= 0.3 on at least 3 of 4 parents AND a nested test-like
ΔALC <= -0.002 on the transferred or per-pair line.

* **r prong:** 0 of 4 parents reach 0.3. The best is multi_swebench's +0.05.
* **ALC prong:** the transferred line is +0.00048 and the per-pair line 0. The
  harness gate also fails on every nested line.

Fine-tuning Qwen3-Embedding-0.6B to rank items by difficulty does not transfer
to an unseen benchmark. It moves held-out r by nothing measurable from the
frozen embedding it started from (paired difference -0.02 to +0.03, every
interval across 0). It does learn a little inside the benchmarks it trains on.
The encoder is closed as an item covariate, nothing ships, and
`experiments/finetune_encoder.py` stays research-only. Per the plan, this does
not escalate to a 4B GCM-style LoRA classifier.

**Against the literature** (numbers as the step-2 review summarised them):

* **ADeLe** (Zhou et al., arXiv:2503.06378). A LoRA-tuned LLaMA-8B scored 0.692
  AUROC out of distribution, below a rubric's 0.747. That comparison pools
  subject ability, so it cannot be set against a within-benchmark r (the plan's
  rule 5). The direction agrees: fine-tuning a text model on success labels is
  weak out of distribution.
* **Krsteski & Meyer** (arXiv:2608.05797). Extra capacity helped within
  benchmark (K-fold), not leave-one-benchmark-out, and embeddings gave 0.02
  LOBO Spearman. Here fine-tuning raised in-distribution r by up to 0.06 and
  LOBO r by 0.
* **Kumar et al. 2022**, fine-tuning distorts pretrained features (cited from
  memory, not re-read). Linear-probe-then-fine-tune is their remedy for
  out-of-distribution loss, and it is the recipe here. With frozen features
  that already transfer at 0, there was nothing to keep.

**What this does not test:**

* adapters on all 28 layers, or full fine-tuning (six layers were what fit the
  time on the M1);
* more than 3 epochs, or more than about 1,100 items an epoch;
* other losses (pointwise regression alone, listwise), or other targets for
  training (strong-tier, recent subjects);
* a larger encoder, or a decoder LLM (the plan rules out a 4B classifier);
* more training benchmarks: two parents plus swe_rebench per run is all the
  nested design leaves.

**Caveats.**

* **Four units.** The random-effects prediction interval for a new benchmark
  spans about ±0.4. That cannot tell a transferable r of 0.2 from 0, and it can
  tell neither from 0.3 on any single parent.
* **Two training parents per run.** Early stopping on a third parent leaves two
  parents and swe_rebench to train on, which favours the null. The frozen
  baseline fitted on all three parents (frozen_lobo512) is no better.
* **Standardisation over the whole benchmark.** x is standardised within
  benchmark over all its items, where a run-time predictor sees only the run's
  items.
* **A second-order leak.** The harness's transferred slope for held-out q is
  fitted on the other parents' out-of-fold x, and the encoders that produced
  it were trained on q's items. Like the fold-averaged target, that favours
  the covariate, and the result is null anyway.
* **The pilot** used one run's idv items to choose between three settings. It
  never read an outer fold, and the chosen setting was the one declared first.
* **MPS nondeterminism.** A resumed epoch reuses the same batches but not
  bit-identical arithmetic.

## Strong model on Kaggle: Qwen3-14B rubric and attempts

On Kaggle (one commit of `kaggle/strong_probe/strong_probe.py`, VERSION k1.2),
then locally `python experiments/strong_llm_eval.py --stage check-schema`,
`--stage ingest` (4.5 s), `--stage signs` (14 minutes), `--stage harness` (68
minutes for 36 covariates), `--stage reference` (8 minutes), `--stage attempts`
(its default `--attempt-units both` writes both readings: every attempted text
in 22 s, the probe texts alone in 19 s; `all` or `probe` writes one),
`--stage verdict`, `--stage run` and `--stage show`, one process at a time. No
language model is loaded locally. Every number below is in
`results/strong_llm_eval.json` unless another section of this file is named,
or it is marked as from the Kaggle log or derived; the results section that
holds it is named in brackets where it helps. The harness stage reads the
stored rows of `experiments/harness.py` (library 3f75a549673aae6a) and never
recomputes hier. The export is in `data/features/kaggle/` and the full Kaggle
Output in `data/features/kaggle_raw/` (both gitignored). A second commit
(commit D, 2026-09-30) ran a third job, the reasoning entropy on all four
parents, read under its own rule in the `entropy` section of the same file:
"Commit D: reasoning entropy on all four parents", at the end of this section.

Which script wrote what: signs, heads, harness and reference ran with the
script of commit 4d2cc4f (digest afe6db1c3f9b84d2, unchanged at 00bdf04). The
working tree's script (feb47227acf072d6) changes only the attempts, verdict,
run and show stages, and adds `run` to the sections ingest drops when the
features change. Ingest, attempts, verdict and run were re-run with it. Ingest
refreshed `meta` and dropped nothing, because the features came out
byte-identical (6e7ce7bbd1a8d2ab), and every number attempts and verdict had
held came out the same. `attempts` (both readings) and `verdict` record the
script digest they ran with.

Every local language-model item signal was null, and the attempt probe could
not be read at all: the 4B solved 2-8% of the problems ("Attempting instead of
judging", FLOOR). The plan's step 8 escalates to a stronger open model on a
free GPU, for two uses:

* **Attempts.** Chain-of-thought attempts on matharena, read with the attempt
  probe's rule, which was fixed before any output existed: FLOOR if graded
  accuracy is under 10%; GO if the best label-free feature reaches a
  within-competition rho ≥ 0.35 against honest difficulty, with a 95% lower
  bound above 0.15, and rho ≥ 0.25 on the 2026 contests; KILL if every feature
  is below 0.15. The primary feature was fixed by the export's columns and
  declared log-prob semantics before any output existed: `tok_entropy`, the
  mean next-token entropy of the raw model distribution over the full
  vocabulary, which is the 4B's D2 lead. The rule was meant for the 147 probe
  texts, the texts of the attempt probe's 160 items, which the notebook runs
  first.
* **A demand rubric.** An ADeLe/DeLeAn-style rubric (Zhou et al.,
  arXiv:2503.06378) with Agent Psychometrics' task scales (Ge et al.,
  arXiv:2604.00594): eight 0-5 demand scales (reasoning, knowledge, work,
  interaction, volume, atypicality, precision, unguessability), a judged share
  of strong 2025-26 systems that solve the task (`solve_share`, declared -) and
  an expert's time (`time_log_minutes`, declared +). Each is read off the
  distribution of the answer digit, as its expected value, thinking off and
  greedy. The
  declared primary is `rubric_ridge`, a ridge over the eight levels fitted
  leave one parent out. The rubric prompt drops matharena's `problem_idx`, and
  the sign prong reads matharena net of position.

### The run

* **Model and engine** (meta.manifest). Qwen/Qwen3-14B-AWQ at revision
  31c69efc29464b6bb0aee1398b5a7b50a99340c3: 4-bit AWQ, fp16, vLLM 0.9.2 on its V0
  engine, tensor parallel over two T4s (run.notebook.engine). Qwen3-14B is not a
  matharena subject, so its attempts are not in the target.
* **One commit** (run.sessions). `ARGS` were empty, so the commit ran the rubric
  and then the attempts, probe texts first, until the session deadline. The
  first cell ran at 2026-09-27 18:11:34 UTC and the script ran 38,368 s
  (10.66 h). Status: rubric done, attempts stopped at the deadline. The
  notebook starts a shard only if 1.25 times its expected duration, plus a
  minute, fits before 11 hours less 5 minutes (`strong_probe.py`, `Clock`).
* **Prefix caching fails on the T4.** The first engine start failed inside
  vLLM's Triton `prefix_prefill` kernel on compute capability 7.5
  ("Unsupported conversion from f16 to f16"). The launch cell recognised the
  error and restarted the script with `--no-prefix-caching`, which is not part
  of the job's configuration hash. Everything below ran without prefix caching.
  The error text and the retry are from the Kaggle session log, which the
  manifests do not record. A copy of that log, as pasted from the Logs tab up to
  04:29 UTC (it ends before the session's last shards), is kept beside the
  Output copy as `session.log` (gitignored); `--stage run` records its path,
  sha256 and that it shows the `prefix_prefill` error and the
  `--no-prefix-caching` retry (run.logs). From the first
  cell to the rubric's start took 591 s (run.derived.before_rubric_s): the
  installs, the downloads, the plan and both engine starts.
* **The rubric** took 6,389 s (1.77 h). Its rows were written from 18:24 to
  20:08 UTC, and it prefilled 999 prompt tokens a second. The plan said 0.61 h
  (0.75 h in its slow case), assuming prefix caching: with the 1,073-token
  instruction block shared, 2.05 M of the 6.42 M prompt tokens need a prefill.
  Without caching all 6.42 M do, and the rubric took 2.9 times the plan. 4,078
  texts (4,326 item_ids) on the four parents; every answer parsed (parse rate
  1.00).
* **The attempts.** k = 4 attempts per text, thinking on, at most 4,096 new
  tokens. The sampling is the model card's: temperature 0.6, top-p 0.95, top-k
  20, presence penalty 1.5. After every attempt a greedy continuation forces
  "**Final Answer** $\boxed{" (the answer log-prob). A shard of 5 texts × 4
  attempts took 592 s (median of 53 intervals between shard writes; 545 to
  605). That is 138 generated tokens a second, with the forced readouts' time
  included: 4.40 M attempt tokens in 8.85 h. The plan assumed 300 (150 in its
  slow case).
* **How far it got** (run.timeline, run.progress). 270 of 819 texts in 8.85 h:
  all 147 probe texts (196 item_ids, done 4.92 h after the rubric, against the
  plan's 2.28 h and 4.28 h slow) and 123 others (140 item_ids). That is 336
  items and 1,344 attempt rows: 1,080 attempts, each text's copied to each of
  its item_ids.
* **Truncation** (run.export.summary). 96.9% of the attempts hit the 4,096-token
  cap and had their answer forced; 6.2% closed their reasoning. The mean length
  is 4,076 tokens. So the attempt features describe the first 4,096 tokens of
  reasoning, not finished solutions. Graded accuracy against matharena's
  reference answer is 21.0%, 20.0% on the 2025 contests and 23.1% on the 2026
  ones (a diagnostic; no feature reads it).
* **The recorder check.** The notebook's raw recorder matched the engine's
  greedy log-probs to 0 (maximum gap) over 13,657 forced-readout tokens, and all
  1,080 attempts carry raw full-vocabulary statistics (`logprobs: raw`). The
  answer log-prob is the forced greedy readout for every attempt, one
  distribution for all (ingest.attempt_semantics).

### Coverage

Every row's content hash matched the local text: 4,326 of 4,326, none
mismatched, missing or unknown (ingest). The rubric covers every item, and
every response, of the four parents: matharena 1,755 items (1,555 texts),
multi_swebench 2,126 (2,078), real_webagents 233, researchcodebench 212.
swe_rebench has no rubric, by design: it has one subject. The attempts cover
336 matharena items, 41% of matharena's responses, and 6.7% of the test-like
evaluated items (4.9% mix/whole; 6.6% and 12.3% on public R1).

### The attempts: GO

The target is the fold-averaged (≈ in-sample) difficulty, as in "The 4B judge,
closed out". The statistic is the attempt probe's: within-competition Spearman
(ranks within each competition, demeaned, pooled), with a 95% bootstrap over
the 16 competitions. The probe texts are the ones the notebook flags in
`_detail/attempt_units.parquet`, a table the export's manifest does not hash.
The stage records its sha256 (638e595695951277…, in full in
attempts.probe_only.source_sha256) and checks the flag against the notebook's
`PROBE_IDS`: a text is a probe text if and only if one of its item_ids is
listed. All 270 texts agree, and all 160 listed items are under the 147
flagged texts (attempts.probe_only.probe_ids_check).

| reading | items (competitions) | graded accuracy | tok_entropy, the primary | tok_entropy, 2026 contests | best feature | call |
|---|---|---|---|---|---|---|
| the 147 probe texts, as the rule was fixed (attempts.probe_only) | 196 (16) | 20.5% | +0.372 [+0.209, +0.516] | +0.585 [+0.370, +0.755] (58 items) | lp_first1024 +0.379 [+0.186, +0.557]; 2026 +0.508 | GO |
| all 270 attempted texts (attempts) | 336 (16) | 21.0% | +0.352 [+0.251, +0.448] | +0.488 [+0.337, +0.603] (107 items) | tok_entropy | GO |

* **The primary passes by itself** on the probe texts: 0.372 ≥ 0.35, lower
  bound 0.209 > 0.15, and 0.585 on the 2026 contests. Those contests post-date
  the model (Qwen3 was released in April 2025), so their ordering cannot come
  from memorised problems or solutions, and they are the strongest part of the
  evidence. On the other contests, labelled 2025 by the rule, the primary gives
  0.286 [0.090, 0.471] (138 items), so the pooled 0.372 leans on the 2026 ones.
  Over all 270 texts the split is 0.290 [0.184, 0.407] (229 items) and 0.488.
  The call's best feature on the probe texts is `lp_first1024`, the mean
  log-prob over the first 1,024 tokens, at 0.379, 0.007 above the primary. On
  all 270 texts the primary is the best feature, at 0.352 [0.251, 0.448].
* **It is not the floor.** Graded accuracy is 20.5% on the probe texts, against
  the 4B's 2.3% and 7.8% ("Attempting instead of judging"). The truncated attempts are still mostly forced
  answers, but the model gets a fifth of the problems right.
* **The reasoning's own uncertainty carries it.** On the probe texts
  `tok_lp` gives +0.36, the entropy over the first 1,024 tokens +0.36, and the
  truncation and forced-answer rates +0.36. Agreement between the four answers
  is weaker (`top_share` +0.24), and the first 256 tokens give +0.21. On all
  270 texts every feature over the whole attempt, its reasoning or its first
  tokens, and every answer and rate feature, is a little lower: `tok_lp` +0.34,
  `ent_first1024` +0.30, truncation +0.28, `top_share` +0.14. The two features
  over the answer span, `tok_lp_answer` and `tok_entropy_answer`, exist only
  where an attempt closed its reasoning (16 items on the probe texts, 31 on
  all). They are wrong-signed on both, -0.23 and -0.37 on the probe texts and
  -0.09 and -0.14 on all, and too sparse to read.
* **It is not truncation in disguise.** On the 2026 probe items the
  truncation rate does not vary, so it orders nothing there, and over all texts
  it gives +0.11 on the 2026 contests; the entropy gives +0.585 and +0.488.
* **Length does not explain it.** In the rule's statistic the prompt's length
  orders difficulty at +0.258 [+0.115, +0.389] on the probe texts, and the
  entropy follows it at +0.272 [+0.151, +0.386]. The adjustment for length is
  read in the sign stage's statistic (`llm4b_close.corr_block`: ranks over the
  benchmark, demeaned within competition), which puts the entropy higher than
  the rule does. In that statistic the entropy gives +0.47 [+0.30, +0.58] on
  the probe texts and +0.42 [+0.26, +0.53] net of log length; on all 336 items
  +0.43 [+0.32, +0.51] and +0.40 [+0.29, +0.47] (attempts, `designs.cot`:
  `spearman_within_corr_block` and `partial_on_log_length`). Netting out
  length takes 0.03 to 0.05 off and leaves the rest. Net of competition, log
  length and position the sign stage gives +0.37 [+0.27, +0.45] on all 336
  items (signs).
* **It holds against the strong tier.** Against b over the subjects at or above
  the median ability it gives +0.28 on the probe texts and +0.30 on all texts.
* **Beside the 4B's D2.** The quantity is the same (attempts.reference_4b:
  comparable), but D2 capped attempts at 512 tokens. D2 met the GO numbers
  post hoc on 32 extreme items (+0.625) and was discounted under its floor.
  This is a declared primary, on 196 items across each competition's range.

GO is a correlation call. It says the 14B's attempts order matharena's problems
by difficulty well enough to be worth an ALC test. It says nothing about ALC,
and on one parent that test cannot be run leave one parent out (below).

### The rubric

Within-group Spearman, oriented (+ = harder), with 95% bootstrap intervals over
groups (signs). Matharena's column gives the value net of competition, log
length and position in brackets, which is the sign prong's statistic there.
"Text-bearing" is matharena without image-only and prompt-only items (1,095
items, 19 competitions), within competition and, in brackets, net of position
and log length. The prong counts the parents with the declared sign. The last
column is a DerSimonian-Laird mean over the four parents with its prediction
interval.

| feature | matharena | multi_swebench | real_webagents | researchcodebench | matharena text-bearing | prong | random effects [PI] |
|---|---|---|---|---|---|---|---|
| rubric_reasoning | +0.07 [-0.04, +0.20] (+0.03) | -0.00 [-0.08, +0.06] | +0.18 [+0.07, +0.30] | +0.28 [+0.16, +0.39] | +0.42 (+0.32) | 3/4 | +0.12 [-0.31, +0.50] |
| rubric_knowledge | +0.04 [-0.07, +0.15] (-0.01) | +0.03 [-0.05, +0.09] | +0.19 [+0.08, +0.30] | +0.11 [-0.08, +0.30] | +0.31 (+0.21) | 3/4 | +0.06 [-0.13, +0.25] |
| rubric_work | +0.07 [-0.04, +0.19] (+0.03) | +0.11 [+0.02, +0.17] | +0.22 [+0.12, +0.32] | +0.32 [+0.18, +0.47] | +0.35 (+0.26) | 4/4 | +0.16 [-0.20, +0.48] |
| rubric_interaction | +0.06 [-0.00, +0.14] (+0.02) | +0.09 [+0.00, +0.15] | +0.12 [-0.03, +0.28] | +0.28 [+0.14, +0.42] | +0.20 (+0.10) | 4/4 | +0.12 [-0.16, +0.38] |
| rubric_volume | +0.05 [-0.04, +0.16] (-0.01) | +0.03 [-0.07, +0.11] | +0.18 [+0.07, +0.29] | +0.20 [-0.02, +0.44] | +0.32 (+0.20) | 3/4 | +0.09 [-0.19, +0.34] |
| rubric_atypicality | +0.05 [-0.08, +0.17] (-0.00) | +0.08 [+0.01, +0.13] | +0.19 [+0.06, +0.34] | +0.27 [+0.14, +0.43] | +0.38 (+0.28) | 3/4 | +0.12 [-0.20, +0.42] |
| rubric_precision | +0.04 [-0.05, +0.15] (-0.01) | -0.04 [-0.12, +0.04] | +0.21 [+0.11, +0.33] | +0.24 [+0.11, +0.42] | +0.31 (+0.20) | 2/4 | +0.10 [-0.37, +0.53] |
| rubric_unguessability | +0.02 [-0.09, +0.13] (-0.02) | +0.02 [-0.07, +0.09] | +0.14 [+0.01, +0.27] | +0.28 [+0.13, +0.44] | +0.31 (+0.23) | 3/4 | +0.09 [-0.26, +0.43] |
| solve_share (declared -) | +0.09 [-0.01, +0.21] (+0.05) | +0.11 [-0.01, +0.21] | +0.14 [+0.01, +0.30] | +0.28 [+0.15, +0.47] | +0.42 (+0.34) | 4/4 | +0.13 [-0.10, +0.35] |
| time_log_minutes | +0.10 [-0.01, +0.24] (+0.07) | +0.10 [-0.01, +0.19] | +0.17 [+0.07, +0.29] | +0.29 [+0.17, +0.45] | +0.45 (+0.36) | 4/4 | +0.14 [-0.10, +0.36] |
| rubric_sum (unit-weight mean) | +0.07 [-0.02, +0.19] (+0.02) | +0.06 [-0.04, +0.14] | +0.20 [+0.11, +0.30] | +0.28 [+0.10, +0.44] | +0.36 (+0.25) | 4/4 | +0.13 [-0.19, +0.42] |

multi_swebench's intervals resample 8 languages and are indicative.

* **Right-signed and weak.** Nine of the ten features have the declared sign on
  at least 3 of the 4 parents. The random-effects means are +0.06 to +0.16,
  and every prediction interval for a new benchmark crosses 0.
* **Most of the signal is on the two small parents.** real_webagents gives
  +0.12 to +0.22 and researchcodebench +0.11 to +0.32. multi_swebench, the
  largest parent, gives -0.04 to +0.11.
* **matharena splits.** Over all its items, net of position and length, every
  feature is between -0.02 and +0.07. On the text-bearing items the same net
  statistic is +0.10 to +0.36, and `time_log_minutes` reaches +0.36. The 4B
  judge's rating was wrong-signed there, -0.17 net of position once oriented
  ("The 4B judge, closed out"). The 14B reads the difficulty of a mathematics
  problem it can see. The prong was fixed in advance on all of matharena's
  items, so this does not count toward it.

**Heads** (heads). Ridge fitted leave one parent out, the penalty chosen by
nested leave-one-parent-out, the features and the target standardised within
benchmark. Pearson over the held-out benchmark / within group; matharena's
prong statistic (net of competition, log length and position) in brackets.

| head | matharena | multi_swebench | real_webagents | researchcodebench | mean | positive | prong | random effects [PI] | penalties chosen |
|---|---|---|---|---|---|---|---|---|---|
| rubric_ridge (primary) | +0.23 / -0.04 (-0.04) | +0.01 / +0.04 | +0.19 / +0.20 | +0.33 / +0.29 | +0.191 | 4/4 | 3/4 | +0.19 [-0.51, +0.73] | 100, 100, 0.1, 1 |
| rubric_ridge_posfree | +0.23 / -0.05 (-0.04) | +0.01 / +0.04 | +0.19 / +0.20 | +0.33 / +0.29 | +0.190 | 4/4 | 3/4 | +0.19 [-0.50, +0.73] | 100, 100, 0.1, 1 |
| rubric_all_ridge (levels and their readout diagnostics) | +0.27 / -0.00 (+0.02) | +0.02 / +0.05 | +0.25 / +0.18 | +0.31 / +0.29 | +0.211 | 4/4 | 4/4 | +0.21 [-0.52, +0.76] | 100 in every fold |
| judge_ridge (levels, solve_share, time) | +0.25 / -0.02 (-0.01) | +0.03 / +0.06 | +0.16 / +0.19 | +0.33 / +0.29 | +0.192 | 4/4 | 3/4 | +0.19 [-0.49, +0.72] | 100, 100, 0.1, 1 |

The heads are positive on every parent, but matharena's +0.23 is competition
level: within competition it is -0.04, and hier learns a competition's level
from labels. multi_swebench gives +0.01. The mean of +0.19 is short of the
0.3 the gate table needs.

### Through the harness

Every oriented rubric feature, `rubric_sum`, each head's out-of-fold
predictions and every attempt feature went through the harness on the shipped
hier (`llm4b_close.py`'s recipe: x standardised within benchmark, the B0 term
on raw x, three placebo draws permuted within benchmark). Test-like ALC
differences ± pair-cluster SE, with the folds nested selection switched on;
public R1 benchmark-first / pair-uniform; within-pair r is the mean
correlation with honest difficulty over a test-like pair's evaluated items.

| covariate | transferred nested | per-pair nested | transferred from B7, forced | per-pair s = 0.5 from B7, forced (placebo) | transferred nested, R1 | within-pair r |
|---|---|---|---|---|---|---|
| head rubric_ridge (primary) | +0.00017 ± 0.00036 (4/4) | -0.00015 ± 0.00007 (4/4) | -0.00029 | -0.00004 (+0.00074) | -0.00212 / -0.00215 | 0.108 |
| head rubric_all_ridge | +0.00005 ± 0.00036 (4/4) | -0.00005 ± 0.00006 (4/4) | -0.00039 | -0.00001 (+0.00077) | -0.00227 / -0.00253 | 0.121 |
| head judge_ridge | -0.00039 ± 0.00016 (3/4) | -0.00016 ± 0.00006 (4/4) | -0.00027 | -0.00004 (+0.00075) | -0.00172 / -0.00158 | 0.110 |
| rubric_sum | +0.00027 ± 0.00042 (4/4) | -0.00019 ± 0.00007 (4/4) | -0.00036 | -0.00005 (+0.00065) | -0.00233 / -0.00237 | 0.129 |
| time_log_minutes | -0.00084 ± 0.00043 (4/4) | -0.00016 ± 0.00006 (4/4) | -0.00058 | +0.00005 (+0.00068) | -0.00370 / -0.00389 | 0.145 |
| rubric_knowledge | -0.00062 ± 0.00043 (4/4) | -0.00004 ± 0.00007 (4/4) | -0.00052 | -0.00006 (+0.00065) | -0.00319 / -0.00279 | 0.118 |
| rubric_work | -0.00045 ± 0.00041 (4/4) | -0.00015 ± 0.00009 (4/4) | -0.00063 | -0.00012 (+0.00073) | -0.00326 / -0.00314 | 0.148 |
| solve_share | +0.00020 ± 0.00019 (1/4) | -0.00003 ± 0.00005 (4/4) | -0.00031 | +0.00008 (+0.00064) | -0.00020 / -0.00009 | 0.127 |
| rubric_reasoning | +0.00111 ± 0.00030 (2/4) | +0.00004 ± 0.00004 (3/4) | -0.00006 | +0.00015 (+0.00063) | +0.00018 / +0.00019 | 0.094 |
| att_cot_tok_entropy (attempt primary) | 0 (0/4) | 0 (0/4) | 0 | -0.00011 ± 0.00005 (+0.00003) | 0 / 0 | 0.315 |

* **Nothing passes the gate** (-0.002, nested, on in at least 3 of 4 folds, no
  parent above +0.002, the public guard). The best nested line of any
  covariate is `time_log_minutes`' transferred slope at -0.00084 ± 0.00043, on
  in every fold, with its worst parent at +0.00102. Then come
  `rubric_knowledge` (-0.00062), `rubric_work` (-0.00045) and the judge head
  (-0.00039 ± 0.00016). The primary head gives +0.00017 transferred and
  -0.00015 per-pair.
* **Within a pair the rubric orders little.** Within-pair r is 0.03 to 0.15 for
  the rubric features and 0.108 to 0.121 for the heads. The gate table puts the
  bar at about 0.25.
* **Public runs read more of it.** On public R1 the transferred nested lines
  of the heads and of the strongest levels gain -0.0016 to -0.0039 (the
  primary head -0.00212 and -0.00215). The gate is read on test-like runs, as
  fixed before; public runs are its guard, not its measure.
* **A level term at B0 costs.** `rubric_knowledge`'s uncentred B0 term, nested,
  costs +0.00306 ± 0.00109, and +0.0169 on its worst parent.
* **The attempts cannot be switched on leave one parent out.** They exist on
  matharena only, so with matharena held out the transferred slope is fitted
  on parents where x is constant, and every nested line is exactly 0. Their
  reading is the forced per-pair lines: -0.00012 ± 0.00005 from B1 and -0.00011
  ± 0.00005 from B7, against placebos of +0.00004 and +0.00003. Their
  within-pair r of 0.315 is on the 6.7% of test-like evaluated items they
  cover.

### What this coverage could give

The reference stage degrades honest difficulty to r on exactly the items the
primary head covers, all four parents' (4,169 keys in the rows), with 0
elsewhere (four draws; reference):

| r | transferred nested | per-pair nested | transferred from B1, forced | per-pair s = 0.5 from B7, forced |
|---|---|---|---|---|
| 0.2 | -0.00096 | -0.00007 | -0.00105 | +0.00026 |
| 0.3 | -0.00246 | -0.00029 | -0.00246 | -0.00033 |
| 0.5 | -0.00722 | -0.00249 | -0.00722 | -0.00219 |

This is the ceiling. A covariate of honest r ≈ 0.3 on all four parents, about
0.25 within a pair (the gate table, "Acceptance harness"), gives -0.0025
through a transferred slope, just past the -0.002 gate; the gate table passes
it on 6 of 8 noise draws. The rubric's heads reach 0.19 across benchmarks and
0.11 within a pair. The attempt entropy reaches the strength (0.35 to 0.37 by
the attempt rule, 0.315 within a pair), but on one parent. No transferred slope
can be fitted for it leave one parent out, and a per-pair slope needs r ≈ 0.5
(-0.0025 at 0.5).

### Verdict: GO for correlation, NULL for ALC

* **The attempts: GO** on the 147 probe texts the rule was fixed for, and on
  all 270 texts attempted. The 14B's mean token entropy is the first item-side
  signal in this repository to pass its correlation bar.
* **The harness: NULL** (verdict.call "NULL: no declared primary passes"; keep
  and exploratory_pass empty). The primary head passes the sign prong
  (positive on 3 of 4 parents; matharena, read net of position, is the one
  that fails) and fails every nested gate line. No
  other covariate passes the gate. The attempt primary cannot pass it on one
  parent, and its forced lines give -0.0001.
* **The entropy job, on all four parents: NULL** (commit D, below;
  entropy.verdict.call "NULL: the declared primary ent_first1024 fails the
  gate", under a rule committed in 78e303e before the output existed). The
  14B's reasoning entropy orders difficulty within group on matharena (+0.29)
  and researchcodebench (+0.33), but on multi_swebench only weakly (+0.12) and
  on real_webagents not at all (-0.03). Within a test-like pair it reaches r
  0.12, far below the 0.25 (an honest r of about 0.3 on all four parents) at
  which the reference reaches the gate, and no nested line comes near it.

Nothing ships. `experiments/strong_llm_eval.py` and the Kaggle kit stay
research-only, and the Kaggle line is closed: no further commit is planned.

**What was not done, and why.**

* **The other 549 attempt texts.** At the measured 592 s a shard, they would take
  about 18.1 h (run.derived), two more 11-hour commits. They cannot change the
  verdict: they are matharena problems, on the one parent the attempts already
  cover, so the transferred slope still could not be fitted leave one parent
  out. The call is GO already.
* **An entropy-only job on all four benchmarks: it has now run** (commit D,
  below). One truncated reasoning sample per item, the token entropy alone, on
  every parent's items. It was the one variant with upside, because it is
  what a transferred slope needs: the feature on every parent, so that the
  slope can be fitted leave one parent out. It was first held until the
  organisers said whether a model can run at predict time: the hidden
  benchmarks' items are not public, so the feature would have to be computed
  inside `predict`, where workers are recreated at each of the six
  checkpoints. It was then run as a measurement for this report, not for the
  submission, and is NULL, so that question no longer decides anything. The
  cost at predict time would have been prohibitive anyway: at the 229.6
  generated tokens a second commit D measured on T4s (138 for the attempts),
  one 1,024-token window for each item of a run of up to 1,000 subject-item
  pairs is about 1.2 hours (derived: 1,000 × 1,024 / 229.6 s), before the
  checkpoints' repeats.

**Caveats built into the script.**

* **A second-order leak for the heads.** The transferred slope for a held-out
  parent is fitted on the other parents' out-of-fold x, and their heads saw
  the held-out parent's items. It favours the covariate, and the result is
  null anyway.
* **Standardisation over every covered item.** x is standardised within
  benchmark over all its covered items; a run-time predictor sees only the
  run's items.
* **The target.** The honest target leaves each subject fold out, but the
  average of the five fits is essentially the in-sample b, which slightly
  favours a feature ("The 4B judge, closed out"). The features themselves use
  no label.

**Other caveats.**

* **Truncation.** 97% of the attempts stop at 4,096 tokens, so the attempts are
  reasoning prefixes with forced answers, and the features read how the model
  starts, not whether it finishes.
* **Repeated texts.** The 270 texts stand for 336 item_ids (run.timeline). A
  repeated text carries the same features under each of its item_ids, and each
  counts in the correlations.
* **Two statistics.** The attempt rule ranks within each competition
  (`attempt_probe.within_rho`); the sign table ranks over the benchmark and
  demeans within group (`llm4b_close.corr_block`). They give 0.35 and 0.43 for
  the same feature on the same items, and 0.37 and 0.47 on the probe texts.
* **One session, one seed**, sampled at temperature 0.6.

### Commit D: reasoning entropy on all four parents

On Kaggle, a second commit of `kaggle/strong_probe/strong_probe.py` with
`ARGS = ["--jobs", "entropy", "--no-prefix-caching"]` and the first commit's
Output attached (the job: `kaggle/strong_probe/README.md`, "Коммит D"). Then
locally `python experiments/strong_llm_eval.py` with commit D's stages, one
process at a time: `--stage check-schema` and `--stage ingest` on
`data/features/kaggle_d` (ingest 14 s), `--stage signs --job entropy` (12
minutes), `--stage harness --job entropy` (six covariates, 6 to 8 minutes
each, 40 minutes), `--stage reference --job entropy` (23 minutes),
`--stage consistency` (4 s), `--stage verdict --job entropy` and
`--stage run`. No language model is loaded locally. Every number below is in
the `entropy` section of `results/strong_llm_eval.json`, its subsection named
in brackets, unless it is marked as from the Kaggle log or derived. The export
is in `data/features/kaggle_d/` and the Output in
`data/features/kaggle_raw/run-260930162825627805/` (both gitignored).

**The rule, fixed before the output.** Commit 78e303e (2026-09-28) added the
job and its reading rule, `ENTROPY_RULE`, before any output existed; its
sha256 `72cde00805ff3c87…` is pinned by a test (rule.digest). The export's
`notebook_digest` (`cd3362a7…`, meta.manifest) is the sha256 of
`strong_probe.py` at 78e303e, so the committed kit is what ran, and the local
stages ran `strong_llm_eval.py` at 78e303e (digest 2bfcbad12a949b0b,
verdict.script_digest; unchanged since). Ingest recorded the rule at its first
run (rule_first, 2026-10-01 06:26:30 UTC), and the verdict (07:41:01 UTC)
found it unchanged and read the complete export (verdict.rule_unchanged,
verdict.final). The rule:

* **The job** (version e1.0, config e21faa7f3d0bf929; meta.kind.config). One
  sample for each of the rubric's 4,078 units, thinking on, no system prompt:
  the task text cut as the rubric cuts it (head and tail to 3,072 tokens),
  then "Think through how you would solve this task." The attempts' sampling
  (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5), at most 1,024
  new tokens, a fixed seed per unit, the raw recorder on, no forced readout.
* **The primary** is `ent_first1024`, the mean raw full-vocabulary next-token
  entropy over the first 1,024 generated tokens, declared + (harder, higher).
  `ent_first256`, `lp_first256`, `lp_first1024` (declared -), `ent_n_tokens`
  (+) and `ent_closed` (-) are exploratory: read through the same lines and
  reported, and only the primary can be kept.
* **Kept only if** a nested harness line passes the gate (-0.002 test-like,
  acting in at least 3 of 4 folds, mix/whole of the same sign, no held-out
  parent above +0.002, neither public weighting above +0.001) and the
  declared sign holds on at least 3 of the 4 parents (within group; on
  matharena net of competition, log length and position), on that version and
  config only, and only on the complete export. Even kept, it would enter the
  submission only if the organisers allowed a model at predict time.

**The run.**

* **Time** (run.sessions, run.timeline). The notebook's first cell ran at
  2026-09-30 16:22:33 UTC, and the export was written at 21:22:38: about 5.0
  hours of session (derived). The script ran 17,643 s (4.90 h), the generation
  17,440 s (4.84 h). Its 89 shards were written from 16:34:45 to 21:22:29, a
  shard of 45.8 units every 194 s (median of 88 intervals; 76 to 218 s).
* **Against the plan** (run.plan.entropy, run.derived). The plan assumed 180
  generated tokens a second (135 in its slow case) and 1,000 prompt tokens a
  second without caching: 7.11 h, 9.30 h slow, in one commit (9.7 GPU-hours
  of quota with the set-up, `kaggle/strong_probe/README.md`). The job
  generated 4.00 M tokens (the plan counted every unit to the cap, 4.18 M)
  from 1.94 M prompt tokens, at 229.6 generated tokens a second with prefill
  and checks included: 1.28 times the assumption and 1.7 times the slow case.
  It took 0.68 of the planned hours, where the first commit's rubric took 2.9
  times its plan and its attempts ran below even the slow case. Up to 48
  sequences (the kit's `ENTROPY_SHARD` and `--max-num-seqs`, 45.8 units a
  shard on average) of about 1.5 thousand tokens decode together here,
  against the attempts' 20 of about 4.2 thousand.
* **Prefix caching** was off from the start. The session log, saved beside the
  Output as `session.log` (run.logs: path, sha256), shows
  `--no-prefix-caching` and no `prefix_prefill` error.
* **Coverage** (ingest, verdict.completeness). 4,078 of 4,078 units, 4,326
  item_ids: matharena 1,555 units (1,755 items), multi_swebench 2,078 (2,126),
  real_webagents 233, researchcodebench 212. Every content hash matched, and
  every response of the four parents is covered. swe_rebench has none, by
  design (one subject). The entropy covers every evaluated item of test-like
  and mix/whole runs, and 69% (benchmark-first) and 97% (pair-uniform) on
  public R1, the rest being swe_rebench (harness.ent_first1024.coverage_eval_items).
* **The generations** (run.timeline, ingest.per_benchmark_stats). 91.1% of
  the units reached the 1,024-token cap, and 24.1% closed their reasoning
  within it (over items: real_webagents 95%, researchcodebench 27%,
  multi_swebench 22%, matharena 14%); the mean length is 982 tokens. No text
  is degenerate. The task text was cut for 5.5% of the units, and for 97% of
  researchcodebench's items, whose tasks average 30,786 tokens.
* **The recorder check** (meta.kind.token_stats). Each shard's greedy check
  request matched the engine's raw log-probs to 0 (maximum gap) over 2,848
  tokens on all 89 shards, and all 4,078 units carry raw full-vocabulary
  statistics (`logprobs: raw`).
* **The first commit's tables are unchanged.** The export carries the rubric
  and the attempts again; ingest found their features byte-identical
  (6e7ce7bbd1a8d2ab, ingest.main_features) and rewrote nothing, so every number
  earlier in this section stands.

**The signs** (signs). Within-group Spearman, oriented (+ = harder), with 95%
bootstrap intervals over groups. Matharena's column gives, in brackets, the
value net of competition, log length and position, the prong's statistic
there; "text-bearing" is as in the rubric's table. The last column is the
DerSimonian-Laird mean over the four parents with its prediction interval.

| feature (declared sign) | matharena | multi_swebench | real_webagents | researchcodebench | matharena text-bearing | prong | random effects [PI] |
|---|---|---|---|---|---|---|---|
| ent_first1024, the primary (+) | +0.29 [+0.16, +0.40] (+0.28) | +0.12 [+0.06, +0.16] | -0.03 [-0.19, +0.15] | +0.33 [+0.17, +0.45] | +0.34 (+0.27) | 3/4 | +0.18 [-0.42, +0.67] |
| ent_first256 (+) | +0.26 [+0.13, +0.36] (+0.27) | +0.16 [+0.13, +0.21] | -0.25 [-0.38, -0.09] | +0.29 [+0.14, +0.43] | +0.27 (+0.20) | 3/4 | +0.12 [-0.54, +0.69] |
| lp_first1024 (-) | +0.29 [+0.18, +0.40] (+0.28) | +0.12 [+0.06, +0.16] | +0.02 [-0.15, +0.19] | +0.31 [+0.14, +0.46] | +0.35 (+0.28) | 4/4 | +0.19 [-0.40, +0.67] |
| lp_first256 (-) | +0.26 [+0.15, +0.35] (+0.26) | +0.14 [+0.11, +0.19] | -0.23 [-0.39, -0.04] | +0.25 [+0.11, +0.40] | +0.28 (+0.21) | 3/4 | +0.11 [-0.53, +0.68] |
| ent_n_tokens (+) | -0.13 [-0.22, -0.04] (-0.18) | +0.05 [+0.02, +0.08] | +0.15 [+0.00, +0.32] | +0.17 [-0.01, +0.37] | | 3/4 | +0.05 [-0.53, +0.60] |
| ent_closed (-) | -0.14 [-0.22, -0.04] (-0.19) | +0.03 [-0.04, +0.11] | +0.12 [-0.00, +0.23] | +0.12 [-0.06, +0.32] | | 3/4 | +0.02 [-0.52, +0.55] |

multi_swebench's intervals resample 8 languages and are indicative.

* **The primary passes the sign prong, and it is weak off mathematics and
  research code.** It has the declared sign on 3 of 4 parents. Its
  random-effects mean is +0.18 (95% CI +0.05 to +0.31), and its prediction
  interval for a new benchmark crosses 0.
* **Where it orders difficulty.** On matharena it gives +0.29 within
  competition, +0.28 net of length and position, +0.34 on the text-bearing
  items and +0.40 on the text-bearing 2026 contests (150 items, 5
  competitions), which post-date the model. Within paper on researchcodebench
  it gives +0.33. On multi_swebench, within language, +0.12; on real_webagents,
  within website, -0.03. Beside the rubric ("The rubric", above): both are
  strongest on researchcodebench, the rubric's levels were near 0 within
  competition on matharena where the entropy is not, and they gave +0.12 to
  +0.22 on real_webagents where the entropy gives nothing.
* **The first 256 tokens are wrong-signed on real_webagents** (-0.25 entropy,
  -0.23 log-prob, both intervals below 0), where 95% of the reasoning closes
  within 1,024 tokens. Exploratory; nothing is kept.
* **Length does not carry it on matharena**: net of log length and position
  the primary keeps +0.28 of its +0.29. The two diagnostics, the tokens
  generated and whether the reasoning closed, are wrong-signed there.

**Consistency and test-retest** (consistency; reported, not gating).

* **Two prompts, one quantity.** On matharena, this job's `ent_first1024` (one
  sample, the generic prompt, 1,024 tokens) against the attempts'
  `ent_first1024` (four samples, the `\boxed{}` prompt, the first 1,024 of
  4,096 tokens) gives Spearman 0.866 over the 336 items both cover (0.865 over
  270 prompts), and against the attempts' primary `tok_entropy` 0.770 (0.776).
  On mathematics the generic prompt keeps most of what passed the attempt rule.
* **One sample is a stable reading.** 131 matharena task texts recur under
  other metadata, so as other units with other seeds. One pair per text gives
  Spearman 0.845 (Pearson 0.847; mean absolute difference 0.039 against an sd
  of 0.099 over units), and ICC(1) over all 788 units of those texts is 0.813.
  Sampling noise would shrink a correlation by about a tenth (derived:
  √0.813 = 0.90). The test-retest exists on matharena only.

**Through the harness** (harness). The rubric's recipe: x standardised within
benchmark, the B0 term on raw x, three placebo draws permuted within
benchmark. Test-like ALC differences ± pair-cluster SE, with the folds nested
selection switched on; public R1 benchmark-first / pair-uniform; within-pair r
as above.

| covariate | transferred nested | per-pair nested | transferred from B1, forced | transferred from B7, forced | per-pair s = 0.5 from B7, forced (placebo) | transferred nested, R1 | within-pair r |
|---|---|---|---|---|---|---|---|
| ent_first1024 (primary) | +0.00110 ± 0.00045 (1/4) | +0.00001 ± 0.00006 (3/4) | +0.00003 ± 0.00055 | +0.00015 | +0.00007 (+0.00075) | +0.00066 / +0.00037 | 0.123 |
| ent_first256 | +0.00135 ± 0.00046 (1/4) | -0.00010 ± 0.00011 (4/4) | +0.00089 ± 0.00051 | +0.00074 | -0.00013 (+0.00077) | +0.00145 / +0.00072 | 0.093 |
| lp_first1024 | +0.00085 ± 0.00033 (1/4) | +0.00008 ± 0.00006 (3/4) | -0.00015 ± 0.00045 | -0.00003 | +0.00018 (+0.00076) | +0.00056 / +0.00029 | 0.122 |
| lp_first256 | +0.00113 ± 0.00037 (1/4) | -0.00002 ± 0.00004 (4/4) | +0.00067 ± 0.00042 | +0.00055 | +0.00012 (+0.00073) | +0.00148 / +0.00073 | 0.093 |
| ent_n_tokens | -0.00001 ± 0.00001 (2/4) | 0 (0/4) | -0.00015 ± 0.00013 | -0.00008 | +0.00086 (+0.00077) | -0.00009 / -0.00017 | 0.065 |
| ent_closed | +0.00002 ± 0.00003 (1/4) | +0.00001 ± 0.00001 (3/4) | -0.00001 ± 0.00012 | -0.00002 | +0.00026 (+0.00074) | -0.00008 / -0.00003 | 0.059 |

* **Nothing passes the gate.** The primary's transferred slope is switched on
  in one fold, the one that holds out real_webagents, and costs +0.0058 there,
  which makes its line +0.00110. Its per-pair slope (s = 0.25 from B1, on in 3
  folds) is level at +0.00001, with mix/whole of the other sign (-0.00027).
  No exploratory feature passes either (verdict.exploratory_pass is empty).
* **Why the transferred slope is off.** Forced on in every fold, the slope from
  B1 gains on three parents (matharena -0.00095, multi_swebench -0.00079,
  researchcodebench -0.00268) and on mix/whole and public runs (-0.0018,
  -0.0021, -0.0029), but costs +0.0058 on real_webagents, where the entropy
  orders nothing, and nets +0.00003 ± 0.00055 on test-like runs (placebo
  +0.00006). Nested selection, which reads the training parents' lines, keeps
  it off in every fold whose training parents include real_webagents, and
  switches it on in the one fold that holds real_webagents out, which then
  pays the +0.0058.
* **Within a pair it orders little.** Within-pair r is 0.123 for the primary
  and 0.06 to 0.12 for the rest. The gate table ("Acceptance harness") puts
  0.123 between its rows for an honest r of 0.1 (0.08 within a pair) and 0.2
  (0.16), where a transferred slope gains -0.0001 to -0.0008 and passes on at
  most 1 of 8 noise draws.
* **The reference** (reference; not gating) degrades honest difficulty to r
  on the items the entropy covers, the same 4,169 keys as the rubric's, and
  so gives the rubric's numbers: -0.00096, -0.00246 and -0.00722 transferred
  nested at r = 0.2, 0.3 and 0.5. Reaching the gate needs an honest r of about
  0.3 on all four parents, about 0.25 within a pair; the entropy has half of
  that within a pair, and nothing on one parent.

### Verdict on commit D: NULL, and the Kaggle line is closed

entropy.verdict.call: "NULL: the declared primary ent_first1024 fails the
gate", on the complete export (final), with the rule unchanged since 78e303e
and the provenance check passing (six harness lines read on the table ingest
recorded, features digest 76e7ffe4185a1d4c). keep and exploratory_pass are
empty.

* The 14B's reasoning entropy carries difficulty where it can be read off the
  statement: matharena (+0.29 within competition, +0.40 on the 2026 contests'
  text-bearing items) and researchcodebench (+0.33 within paper). On the two
  agentic parents it is weak (multi_swebench +0.12) or absent (real_webagents
  -0.03).
* Within a test-like pair it reaches r 0.12, against the about 0.25 (honest r
  0.3) the reference needs. No nested line comes near the gate: the
  transferred slope is on in one fold and costs (+0.00110), and the per-pair
  slope is level (+0.00001).
* So no language-model item signal in this repository passes the gate, from
  a 4B or a 14B, as a rating, a rubric, attempts or reasoning entropy.

Nothing ships, and the Kaggle line is closed. The 549 attempt texts left
(commit C) are matharena problems and cannot change any call, and no further
commit is planned.

**Caveats.**

* **One sample per unit**, at temperature 0.6, in one session. On matharena a
  single sample is reliable (ICC 0.81); elsewhere the test-retest is not
  measured.
* **A generic, cut prompt.** One instruction for every benchmark, no answer
  format, and the task text cut to 3,072 tokens; for researchcodebench that
  is the head and tail of a 30,786-token task on average.
* **91% of the windows are full**: the features read the first 1,024 tokens
  of reasoning, the same window as the attempts' `ent_first1024`.
* **Standardisation and the target** are those of the rubric (above): x is
  standardised over every covered item of a benchmark, and the target is the
  fold-averaged honest difficulty, which slightly favours a feature.

## Meta-learned heads on frozen embeddings

`python experiments/heads_eval.py --rows legacy` (3,223 s on one process with
two torch threads, at most 1.07 GB, beside an unrelated five-core job), then
`--rows current` (3,134 s). Every number below is in `results/heads_eval.json`.
The study ran in session scratch on 2026-09-26 (`rethink2/fine-tuning/heads.py`
and its queue scripts). The script is that code with only its data loading
changed: it reads `experiments/harness.py`'s stored rows. The committed legacy
rows (`data/harness_rows_legacy`, the old floor and the solver before the
floored-fit fix) equal the study's own scratch rows exactly: 0 of 300
test-like and 0 of 150 mix/whole runs differ, over 2,135,316 evaluated
response-budget cells. On them the script reproduces all 21 of the study's
result files bit for bit. The current rows (`data/harness_rows`, library
3f75a549673aae6a, with the corrected floor and the floored-fit fix) were run
as well. The legacy pass ran script digest 60cfb012, which differs from the
committed file (6ad7190a) only in how the results file is saved.

**What it is.** Heads trained on episodes sit on frozen Qwen3-Embedding-0.6B
features (PCA 64, whitened on the training parents, plus log length) and add a
capped logit offset to the shipped hier's prediction:

* `diff`, a meta-learned difficulty direction, centred on labeled items;
* `kern`, a learned-metric few-shot kernel on hier's residuals;
* `ass`, `diff` plus a rank-2 subject × item term from provider, reasoning flag
  and release year;
* `all`, the three together.

λ is chosen from {0.01, 0.1, 1} or "off" by nested leave-one-parent-out, and a
head is used on the held-out parent only if its inner mean is below 0.
Training uses test-like and mix/whole episodes of the three training parents.
Scoring uses the held-out parent's appearances on 300 test-like runs (seed 2)
and 150 mix/whole runs (seed 3). The controls are the same form with the
parent's in-sample Rasch difficulty (`oracle`), that oracle degraded to
correlation r, the heads forced on without selection, the heads within
benchmarks already seen (subject folds), and PCA 16 and 256.

ALC difference against hier, ± run SE / (parent, subject) cluster SE,
test-like (mix/whole in brackets):

| configuration | legacy rows (the study as it ran) | current rows |
|---|---|---|
| nested `diff`, `kern`, `ass`, `all`; PCA 16 and 256 | off in 4 of 4 folds for every head: exactly 0 | the same |
| smallest inner mean over folds and λ | +0.000007 (`ass`, matharena held out) | +0.000007 |
| `all` forced, λ 0.01 / 0.1 / 1 | +0.00064 ± 0.00011 / 0.00023; +0.00054; +0.00020 | +0.00064; +0.00053; +0.00019 |
| oracle (in-sample Rasch difficulty) | -0.04371 ± 0.00116 / 0.00374 [-0.05566, cluster SE 0.00620] | -0.04360 ± 0.00116 / 0.00373 [-0.05561] |
| oracle per held-out parent | matharena -0.0488, multi_swebench -0.0342, real_webagents -0.0508, researchcodebench -0.0509 | -0.0482, -0.0343, -0.0507, -0.0509 |
| degraded oracle, r 0.1 / 0.2 / 0.3 / 0.5 / 0.7 | -0.00009 / -0.00086 / -0.00239 / -0.00799 / -0.01765 | -0.00009 / -0.00087 / -0.00239 / -0.00798 / -0.01763 |
| benchmarks seen: `diff` λ 0.001; `all` PCA 256 | -0.00051 ± 0.00027 / 0.00084; -0.00169 ± 0.00035 / 0.00080 [-0.00243; -0.00457] | -0.00048; -0.00139 [-0.00239; -0.00426] |

**Verdict: no gain.** Nested selection switches every head off in every fold,
on both row sets. Forced on, the heads cost +0.0002 to +0.0006. They do learn
something where the benchmark has been seen (-0.0005 to -0.0017 test-like,
-0.0024 to -0.0046 mix/whole), and none of it reaches a held-out parent. The
oracle row is the acceptance target of "Acceptance harness", and the degraded
oracles at r = 0.3 and 0.5 are the forced draws that section compares with.
Nothing ships; `experiments/heads_eval.py` stays research-only.

**Caveat.** `experiments/harness.py --stage verify --scratch` still expects the
study's scratch row format. `data/harness_rows_legacy` holds the same
predictions in another format, so it cannot stand in without a code change.

## Shipped configuration, confirmed

`python experiments/ship_confirm.py` (about 5 seconds on one process, about
560 MB; it runs no model). Every number below is in `results/ship_confirm.json`;
`tests/test_ship_confirm.py` covers the script.

The archive ships `mu0 -2.5, sigma_mu 2.5, attr_scale 0.5` ("What actually
shipped, after the audit"), but "Calibrating for the hidden test" confirmed and
guarded the recommended `mu0 -3.0, attr_scale 0.25`. This section gives the
shipped configuration's own numbers. It reads the shipped arm ('ship') of
`experiments/subject_side.py`'s stored rows (`data/subject_side_rows/`,
gitignored) and pairs it, pair by pair on identical runs, with comparators
already stored:

* **the legacy Predictor** (the first submission): `results/testlike_check.json`
  on test-like runs, `results/hier_eval.json` on public runs, and
  `results/level_calibration.json` on the two sensitivity regimes;
* **smoothed Beta(2,2)**: `results/level_calibration.json` on test-like runs 0
  to 199 and the sensitivity runs, `results/hier_eval.json` on public runs;
* **the organisers' empirical mean**: `results/testlike_check.json`.

Each comparator run is checked before use: field by field where the stored
rows carry ids, prediction by prediction where they do not. A run that
disagrees stops the script, and a run that cannot be matched is left out.
Where rows do not exist, nothing is filled in (`not_stored` in the file):
smoothed on test-like runs 200 to 299 and mix/whole 100 to 199, the legacy
Predictor on mix/whole 100 to 199, and the empirical mean on pair-uniform,
mix/whole and no-shift runs.

150 consistency checks pass. Among them: the selection half of
`level_calibration.json` (0.1608 against 0.2039, -0.0431 with the same run,
cluster and stratified SEs and leave-one-parent-out values); the guard's
Predictor ALCs (0.2068, 0.2038); both sensitivity regimes matched to
`level_calibration.json` on 100 of 100 runs; four other records of the same
rows (`subject_side.json`'s summary, `level_calibration.json`'s own rows of the
shipped config, `mcq_floor.json`'s old-floor arm, `official_baselines.json`'s
per-run smoothed and empirical means); and every number an internal review had
recomputed from these rows.

**Statistics** are those of `experiments/level_calibration.py`: ± run SE /
cluster SE ((parent, subject) on test-like runs, (benchmark, subject) on
public ones; 2,000 resamples) / the same stratified by parent. New here is the
**parent-level** mean ± SE: the mean and SE of the four multi-subject parents'
appearance-weighted means. It is the only SE here that sees variation between
benchmarks.

**Which code the rows are.** `paiec/hier.py` 70a3a81a with the old
multiple-choice floor: the library of formative run 2's archive (ee5085a),
before the corrected floor (f7e7d87) and the floored-fit fix (4d2cc4f).
(`paiec/prior.py` and `paiec/subjects.py` are at their f7e7d87 versions, whose
new terms are off by default and leave the prior and predictions unchanged
there, "Subject side at budgets 0 and 1".) The gap to the archive now selected
was measured on runs 0 to 199 and is not added to any number below: the
corrected floor moves ALC by -0.00026 (test-like), -0.00027 (benchmark-first)
and -0.00045 (pair-uniform) ("The multiple-choice floor, corrected"), the
floored-fit fix by 0, -0.0000002 and -0.000008 ("Floored fits").

### Against the legacy Predictor

| regime, runs | shipped ALC | legacy ALC | shipped minus legacy | parent-level | parents (range) | runs won |
|---|---|---|---|---|---|---|
| test-like, 0-99 (selection half) | 0.1608 | 0.2039 | -0.0431 ± 0.0019 / 0.0032 / 0.0028 | -0.0378 ± 0.0095 | -0.059 to -0.014 | 99% |
| test-like, 100-199 (confirmation half) | 0.1677 | 0.2073 | -0.0396 ± 0.0019 / 0.0037 / 0.0034 | -0.0350 ± 0.0107 | -0.058 to -0.007 | 97% |
| test-like, 200-299 (never used to choose the level) | 0.1687 | 0.2107 | -0.0419 ± 0.0017 / 0.0037 / 0.0034 | -0.0378 ± 0.0075 | -0.057 to -0.021 | 100% |
| test-like, 0-299 | 0.1658 | 0.2073 | -0.0415 ± 0.0011 / 0.0034 / 0.0031 | -0.0369 ± 0.0091 | -0.058 to -0.014 | 98.7% |
| mix/whole (seed 3), 0-99 | 0.1694 | 0.2129 | -0.0436 ± 0.0014 / 0.0039 / 0.0031 | -0.0293 ± 0.0123 | -0.065 to -0.010 | 100% |
| no date shift (seed 3), 0-99 | 0.1585 | 0.1729 | -0.0144 ± 0.0012 / 0.0024 / 0.0022 | -0.0119 ± 0.0049 | -0.024 to -0.000 | 87% |
| public R1 benchmark-first, 0-99 | 0.2053 | 0.2068 | -0.0015 ± 0.0010 / 0.0021 / 0.0018 | -0.0025 ± 0.0050 | -0.013 to +0.011 | 62% |
| public R1 benchmark-first, 0-149 | 0.2051 | 0.2068 | -0.0017 ± 0.0008 / 0.0021 / 0.0018 | -0.0029 ± 0.0049 | -0.013 to +0.010 | 61% |
| public R1 pair-uniform, 0-99 | 0.2020 | 0.2038 | -0.0018 ± 0.0010 / 0.0017 / 0.0016 | -0.0020 ± 0.0053 | -0.015 to +0.009 | 64% |

Leaving one parent out of test-like runs 0 to 299 gives -0.047 to -0.032.

**Where the test-like gain sits.** By budget on runs 0 to 299 (cluster SE in
brackets): -0.135 (0.011) at B0, -0.076 (0.007) at B1, -0.037 (0.003) at B3,
-0.016 (0.001) at B7, -0.008 (0.001) at B15 and -0.008 (0.001) at B31. B0 and
B1 carry 0.029 of the 0.042. By parent: matharena -0.014 ± 0.009, real_webagents
-0.033 ± 0.007, researchcodebench -0.042 ± 0.005 and multi_swebench -0.058 ±
0.004 (cluster SEs). The parent-level SE (0.009) is about 2.7 times the
cluster SE (0.0034): with four parents, the between-benchmark uncertainty of
the headline is larger than the cluster SEs say.

**Public runs.** The shipped config is level with the legacy Predictor: its
point estimates are 0.0015 to 0.0018 better, within about one cluster SE
(0.81 cluster SEs benchmark-first over runs 0 to 149, 1.02 pair-uniform). It loses at
B0 (+0.0049, cluster SE 0.0065, benchmark-first; +0.0140, 0.0072,
pair-uniform) and B1 (+0.0019, +0.0015) and gains from B7 on (-0.004 to
-0.008 a budget). By parent it loses on matharena (+0.0101 ± 0.0031
benchmark-first, +0.0094 ± 0.0023 pair-uniform) and on the single-subject
swe_rebench pair (+0.0099 over 130 benchmark-first appearances, +0.0114 over
7 pair-uniform ones; one subject, so no cluster SE; "A single-subject
benchmark" gives an SE over appearances and the budgets), and gains most on
multi_swebench (-0.0134, -0.0154).

### Against the smoothed mean and the empirical mean

| regime, runs | minus smoothed Beta(2,2) | parent-level | minus empirical mean |
|---|---|---|---|
| test-like, 0-99 | -0.0180 ± 0.0010 / 0.0021 / 0.0020 | -0.0159 ± 0.0034 | -0.0270 ± 0.0014 / 0.0028 / 0.0027 |
| test-like, 100-199 | -0.0159 ± 0.0009 / 0.0021 / 0.0020 | -0.0139 ± 0.0044 | -0.0304 ± 0.0014 / 0.0030 / 0.0029 |
| test-like, 200-299 | not stored | | -0.0335 ± 0.0016 / 0.0035 / 0.0033 |
| test-like, 0-199 | -0.0169 ± 0.0007 / 0.0020 / 0.0019 | -0.0148 ± 0.0039 | |
| test-like, 0-299 | | | -0.0303 ± 0.0009 / 0.0030 / 0.0028 |
| mix/whole, 0-99 | -0.0176 ± 0.0008 / 0.0022 / 0.0019 | -0.0119 ± 0.0050 | not stored |
| no date shift, 0-99 | -0.0220 ± 0.0014 / 0.0030 / 0.0028 | -0.0189 ± 0.0060 | not stored |
| public R1 benchmark-first, 0-149 | -0.0086 ± 0.0010 / 0.0030 / 0.0021 | -0.0108 ± 0.0062 | -0.0447 ± 0.0012 / 0.0028 / 0.0024 |
| public R1 pair-uniform, 0-99 | -0.0093 ± 0.0011 / 0.0021 / 0.0019 | -0.0101 ± 0.0062 | not stored |

The smoothed mean has no level prior to tune, so this is the comparison the
test-like regime's construction does not favour. The shipped config beats it
on every test-like parent (-0.006 to -0.024 on runs 0 to 199), by more without
the date shift than with it, and on public runs by 0.009. On public runs it
loses to it on matharena (+0.0060 benchmark-first, +0.0043 pair-uniform) and
on the swe_rebench pair (+0.0127, +0.0141). By budget on public
benchmark-first runs it gains at every budget (-0.015 at B0 to -0.012 at B31).

### Formative run 2 against these regimes

Run 2 (the shipped model's archive at ee5085a, 8 pairs, ALC 0.192623; "Formative
feedback, runs 1 and 2") is one draw. Its distance from each regime's mean, in
the shipped model's single-run sds over runs that redraw one catalogue of
pairs:

| regime (runs) | mean ALC | single-run sd | run 2's z | share of runs ≥ run 2 | z by budget, B0..B31 |
|---|---|---|---|---|---|
| test-like (300) | 0.1658 | 0.0318 | +0.84 | 0.20 | +0.98 +0.15 +0.86 +0.89 +0.98 +1.40 |
| test-like, mix/whole (200) | 0.1711 | 0.0329 | +0.65 | 0.245 | +1.12 +0.05 +0.72 +0.70 +0.72 +1.15 |
| test-like, no date shift (100) | 0.1585 | 0.0335 | +1.02 | 0.18 | +1.00 +0.45 +0.89 +1.11 +1.20 +1.67 |
| public R1 benchmark-first (150) | 0.2051 | 0.0285 | -0.44 | 0.67 | +0.06 -0.78 -0.49 -0.46 -0.27 +0.23 |
| public R1 pair-uniform (100) | 0.2020 | 0.0261 | -0.36 | 0.68 | -0.24 -0.67 -0.37 -0.32 -0.13 +0.50 |

(The budget z values use run 2's budgets rounded to three decimals. "Formative
run 3" places run 3 against the same regime means and sds from its exact
budget means, and places run 2 again from its exact budgets.) One run cannot
tell these regimes apart. Run 2's profile is at least as close to the public
runs as to the tuned regime: every budget is within 0.8 sd of public R1, while
its B31 is 1.4 sd above the test-like mean.

Single-run sds of the shipped model by budget (B0..B31 | ALC): test-like
0.021, 0.039, 0.037, 0.035, 0.034, 0.032 | 0.032; mix/whole 0.017, 0.040,
0.037, 0.036, 0.035, 0.033 | 0.033; no date shift 0.042, 0.042, 0.038, 0.033,
0.031, 0.029 | 0.033; benchmark-first 0.033, 0.037, 0.035, 0.029, 0.026,
0.025 | 0.029; pair-uniform 0.030, 0.039, 0.030, 0.028, 0.026, 0.024 | 0.026.

### What this changes

* No number of the recommended config stands for the shipped one. The
  shipped config's held-out gains over the legacy Predictor are -0.0396 and
  -0.0419 (confirmation half; runs never used to choose the level), against
  the recommended config's -0.0418 on the confirmation half.
* The gain is conditional on the regime: 0.040 to 0.044 in the tuned regime
  and its mix/whole variant, 0.014 without the date shift, and 0.022 to 0.029
  in the two regimes nearer the per-pair feedback reading ("What actually
  shipped, after the audit"). The level_mean -1.2 and -2.0 sensitivities were
  never scored for the shipped config. (Regimes at the feedback's readings
  were scored later: the gain there is 0.025 to 0.031, "Regime sensitivity at
  the feedback's reading".)
* Against the smoothed mean the gain is 0.017 in the tuned regime and 0.009 on
  public runs.
* The single-subject case (open since "Verdict: ship hier with the level moved
  down"): the shipped config loses about 0.010 to 0.011 against the legacy
  Predictor on the one public single-subject pair and 0.013 to 0.014 against
  the smoothed mean, from one subject.

**What was not done.** No model was run, so nothing here is new data: the rows
were scored by `experiments/subject_side.py` and the comparators by the
scripts named above. The code gap to the current archive is recorded, not
added.

## Formative feedback, runs 1 and 2

`python experiments/formative_feedback.py --stage record`, `--stage prereg`,
then `--stage read` (seconds each on one process, at most 0.3 GB; no model is
run). Every number below is in `results/formative_feedback.json`, whose
`passes` log every stage with its time and code digests.
`tests/test_formative_feedback.py` (11 tests) covers it, including that the
stored preregistration is the script's text and was written before the first
reading. `record` verifies the two archives only when given them (`--archive1
PATH --rebuild1 PATH --run1-how TEXT --archive2 PATH --rebuild2 PATH
--run2-how TEXT`; the paths used are in `passes` and point into session
scratch, which is not durable). Without them it keeps the `archives` already
stored, so the hashes in the results file are the lasting record, and
re-verifying needs the archives themselves. The inputs are `results/formative/run1.txt` and `run2.txt`: the
organisers' per-pair tables as Codabench returned them, byte for byte, with
only the surrounding chat text removed.

**Two submissions had been scored** when this section was written; a third,
scored on 2026-10-01, is recorded in "Formative run 3".

| run | scored | model | pairs | benchmarks | subjects | evaluated subject-item pairs | ALC |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-25 | legacy Predictor, commit b68492c | 9 (2, 2, 1, 1, 1, 1, 1 per benchmark) | 7 | 9 | 485 (44 to 76 a pair) | 0.2113 |
| 2 | 2026-09-26 | hier, LEVEL mu0 -2.5 / sigma_mu 2.5 / attr_scale 0.5, commit ee5085a (old multiple-choice floor, solver before the floored-fit fix) | 8 (2, 1, 1, 1, 1, 1, 1) | 7 | 8 | 492 (44 to 113) | 0.192623 |

**The score.** The platform's ALC is the unweighted mean over pairs of each
pair's 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31. That gives 0.2112999889
and 0.1926232375, which round to the reported 0.2113 and 0.192623. Weighting
pairs by their evaluated items would give 0.2147 and 0.1988, so the platform
does not. Per-pair ALC and ECE recomputed from the budget tables match the
tables' own summary rows to within 6e-7.

**Run 2, per pair.** Letters relabel the anonymous benchmark ids, the same
letter in both runs (run 1's table is in "Against the first real formative
feedback"). They are for recording only; no model input is keyed on them.

| pair | benchmark | n | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE (ALC-weighted) | ECE at B0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | D | 58 | 0.2555 | 0.1939 | 0.1840 | 0.1931 | 0.1868 | 0.1857 | 0.1957 | 0.091 | 0.269 |
| q2 | G | 44 | 0.2706 | 0.1635 | 0.1790 | 0.1500 | 0.1458 | 0.1490 | 0.1696 | 0.136 | 0.357 |
| q3 | C | 113 | 0.2593 | 0.2469 | 0.2408 | 0.2437 | 0.2418 | 0.2524 | 0.2458 | 0.045 | 0.120 |
| q4 | G | 44 | 0.2576 | 0.2406 | 0.2559 | 0.2311 | 0.2315 | 0.2161 | 0.2392 | 0.093 | 0.147 |
| q5 | F | 44 | 0.2256 | 0.2300 | 0.2553 | 0.2329 | 0.2228 | 0.2392 | 0.2347 | 0.104 | 0.043 |
| q6 | E | 60 | 0.2366 | 0.1973 | 0.1989 | 0.1961 | 0.1950 | 0.1983 | 0.2009 | 0.068 | 0.207 |
| q7 | A | 75 | 0.2333 | 0.2126 | 0.2282 | 0.2119 | 0.2008 | 0.2240 | 0.2164 | 0.104 | 0.144 |
| q8 | B | 54 | 0.1562 | 0.0729 | 0.0283 | 0.0098 | 0.0034 | 0.0014 | 0.0386 | 0.160 | 0.395 |
| mean | | | 0.2368 | 0.1947 | 0.1963 | 0.1836 | 0.1785 | 0.1833 | 0.1926 | 0.100 | 0.210 |

Run 1's per-pair ALC is 0.2568, 0.1682, 0.2558, 0.1636, 0.2515, 0.1686,
0.2065, 0.1771 and 0.2535 (p1 to p9), and its ECE-ALC 0.171, 0.288, 0.264,
0.344, 0.073, 0.038, 0.204, 0.042 and 0.103.

| budget | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|
| run 1, mean ECE | 0.381 | 0.253 | 0.195 | 0.097 | 0.090 | 0.047 |
| run 2, mean ECE | 0.210 | 0.098 | 0.118 | 0.084 | 0.057 | 0.076 |

The largest pair ECE at B0 is 0.753 in run 1 and 0.395 in run 2, and the mean
ECE-ALC 0.170 and 0.100. The mean B0 excess over B31 is 0.202 and 0.054.

**Overlap.** No subject appears in both runs (9 and 8 distinct subjects), so
no (subject, benchmark) pair recurs and there is no paired platform comparison
of the two predictors. All 7 benchmark ids are shared: both runs drew from the
same 7 hidden benchmarks, with different subjects. Weighting the 7 benchmarks
equally gives 0.2139 for run 1 and 0.1909 for run 2; the subjects differ, so
that is descriptive only.

**What two runs can and cannot measure.** They pin the scoring rule, and they
give a rough reading of the global level of the hidden benchmarks (7
benchmarks, SE about 0.5 logit, below). They cannot measure the gain from the
legacy Predictor to hier: the subjects differ, and the difference of two
independent runs has an sd of about 0.04 (single-run sds 0.029 to 0.032),
against the observed 0.019.

**Archives.**

* **Run 2** was the archive with sha256
  `2c64eaada491cbf85bd54ae190cdbb28f9d0a13870df0dc77a3a850f90cf661f`: the file
  downloaded from the platform's submission page on 2026-09-28 has exactly
  these bytes, as does the copy kept in session scratch (`record` stage,
  `archives.run2`). Every tracked member is byte-identical to ee5085a, and its
  prior.json (sha256 `c7ce3b84…`) is byte-identical to today's
  `submission/prior.json`. Rebuilding ee5085a from `git archive` with that
  tree's own `tools/build_submission.py`, default BLAS threading, gives the
  same bytes (validator OK, run check bit-identical). With OpenBLAS, OMP and
  vecLib held to one thread, prior.json differs in its last bits (worst
  relative difference 2.1e-11) and the archive hash becomes `660020de…`: a
  byte-identical rebuild needs the same BLAS and thread count as the build
  machine (an 8-core M1 Pro with OpenBLAS 0.3.23.dev, default threading); it
  was not tried elsewhere.
* **Run 1** was `8e28d930d45b16aee351bfbbc75531a6d12079667793232d4bd74a17f9d47b9f`:
  the file downloaded back from the platform's submission page on 2026-09-28
  has these bytes. It is the deterministic build of b68492c, byte-identical to
  a copy written on 2026-09-24 before the upload, with every tracked member
  equal to b68492c (`record` stage, `archives.run1`).
* **The archive now selected** is `4a882cc7…`, built at 4d2cc4f with the
  corrected floor and the floored-fit fix. Its run was planned as a
  regression and latency check only, its score not to select or tune
  anything. It ran as formative run 3, scored 2026-10-01 ("Formative run 3").

**Uses of the feedback, all of them.** Run 1 set the test-like regime's
defaults (its lower-root B31 reading gave level_mean -1.6 and sd 1.5; the
1.25-year date shift is the grid point whose legacy-Predictor B0 and B1 are
closest to it; its shape motivated `repeat` 0.25, set by hand, and its
evaluated counts the off-by-default `split_after_cut`). Its composition (5
pairs alone on their benchmark, 4 sharing one) weighted the first verdict of
"Hierarchical model". The level calibration chose mu0, sigma_mu and attr_scale
on runs of that regime and added each candidate's paired difference to run 1's
budgets as a sanity estimate. The audit then read run 1 per pair and moved the
shipped level from the recommended config to the milder one ("What actually
shipped, after the audit"): the only per-pair reading behind a shipped choice,
and a choice of global hyperparameters only. Run 2 motivated the subject-side
study, which shipped nothing, and this reading, which changed nothing. This
reading and the audit's reading of run 1 later set the level targets of the
READING, AUDIT and MIXTURE regimes of "Regime sensitivity at the feedback's
reading", which also changed nothing. Run 3
is a regression and latency check and is read for nothing else ("Formative
run 3"). The
leaderboard (organisers' entry 0.1801, best 0.1172, read 2026-09-24) is used
for placement only. Nothing is keyed on an anonymous benchmark or subject id,
and no prediction was shaped to probe hidden labels.

### Q2: the shipped config's own matched estimate on run 1

The estimator is the audit's per-pair matched estimate, ported unchanged from
session scratch: each run-1 pair is matched to its K nearest replica pair
appearances by the Euclidean distance of their six per-budget Brier values
under the legacy Predictor, each standardised by its sd over the pool, and a
candidate's estimate is run 1's per-pair Brier plus its mean paired difference
over those neighbours. On the audit's own inputs the port reproduces the
audit's recorded values to within 7.6e-6.

| pool, K | shipped (mu0 -2.5, attr_scale 0.5) | recommended (mu0 -3.0, attr_scale 0.25) | smoothed | hier defaults |
|---|---|---|---|---|
| where the shipped config is scored in `level_calibration.json` (test-like 0-99, public benchmark-first 0-39), K 15 | 0.1812 | 0.1824 | 0.1908 | 0.1988 |
| the same, K 40 | 0.1830 | 0.1839 | 0.1925 | 0.1983 |
| the shipped model's own rows (6,215 appearances with both rows), K 15 | 0.1774 | | | |
| the same, K 40 | 0.1779 | | | |

On its own rows the K 15 estimate by budget is 0.245, 0.190, 0.176, 0.162,
0.162 and 0.150. Across the robustness variants (K 25; six budgets or B0 and
B31 only; all pools or test-like only) it ranges from 0.1766 to 0.1824.
Pools of public runs alone give 0.205, but their matches lie about five times
further away (median distance 2.74 against 0.56). `experiments/level_audit.py`
gives 0.1765 and 0.1789 on the pool where the shipped config, the recommended
config and the Predictor are all stored (4,955 appearances), which agrees.
So the 0.178 that replaced 0.167 in "What actually shipped, after the audit"
belongs to the recommended config; the shipped config's own estimate is 0.177
to 0.183.

**As a prediction for run 2.** Run 2's pairs are not run 1's, so this is a
prediction for a like-sized draw. Observed minus the K 15 estimate on the
shipped rows, by budget: -0.008, +0.004, +0.021, +0.022, +0.017, +0.034, and
+0.015 in ALC. In the shipped model's single-run test-like sds that is -0.38,
+0.11, +0.56, +0.61, +0.50, +1.06 and +0.48 (public benchmark-first: +0.53 in
ALC, +1.34 at B31). B0 came out better than predicted; the excess is in the
late budgets. Run 2's pairs sit nearer a rate of 0.5 than run 1's (mean B31
0.183 against 0.157). A new proxy that reweights the shipped model's test-like
rows to a target level distribution (bins of pair logit, width 0.5; it moves
the level only, not the attribute optimism the date shift carries) predicts
0.177 at the audit's reading of run 1, 0.184 and 0.183 at the pooled 17-pair
readings below, and 0.168 at the tuned regime's level.

### The pooled reading, under a rule fixed in advance

**The preregistration** was written into the results file at 05:06:59 UTC on
2026-09-28 (sha256 `0de18448…`), before any matching of run 2's pairs, the
pooled reading or the shipped config's estimate on run 1 was computed; the
first reading ran at 05:13:48. It fixes the rule, not what had been seen:
both feedback tables, run 2's per-pair Brier included (the `record` stage had
parsed it at 05:04:42), and the audit's scratch outputs for run 1 were known
when it was written. It fixed:

* **What may change:** one global hyperparameter only, the level distribution
  of a new benchmark (mu0 and sigma_mu of LEVEL). Not attr_scale, not the item
  or subject side, nothing keyed on an id.
* **The reading:** a pair's rate is the root of p(1-p) = B31 on the side where
  most of its K neighbours' rates sit (upper if more than half exceed 0.5).
  Run 1 is matched on the legacy Predictor's profiles, run 2 on the shipped
  hier's. Its level is the continuity-corrected logit at the pair's n, the
  convention of the regimes' realised pair logits. Uncertainty comes from
  2,000 draws that resample the 7 benchmark ids as clusters and take each
  pair's upper root with its neighbour share.
* **The comparator:** the tuned regime's realised pair logit, mean -1.290 and
  sd 1.696 (`results/testlike_check.json`).
* **The rule:** propose mu0 (or sigma_mu) only if, for both K = 15 and 40, the
  17-pair mean (or sd) differs from the comparator by more than 2 bootstrap SEs
  in the same direction, and only if a guard finds no bias from the root
  reading itself (applied to the replica's own pairs, weighted 9:8). A proposed
  candidate would still have to be scored on fresh test-like runs set to the
  reading and pass: at least 0.002 better than the shipped LEVEL with no parent
  above +0.002, at most +0.001 against it on both public weightings, at most
  +0.003 against the legacy Predictor on both, and at most +0.002 against it
  on the default test-like regime. Nothing ships from the reading alone.
  Everything else it reports is descriptive and changes nothing.

**The reading** (continuity-corrected pair logit; SEs resample the 7
benchmarks):

| reading | pairs | mean (SE) | sd (SE) | plain logit, mean / sd |
|---|---|---|---|---|
| both runs, K 15 | 17 | -0.65 (0.51; 90% -1.70 to -0.02) | 1.80 (0.36) | -0.83 / 2.23 |
| both runs, K 40 | 17 | -0.78 (0.51) | 1.75 (0.34) | -0.96 / 2.17 |
| run 1 alone (p6, p8 and p9 read as high-rate at both K) | 9 | -0.86 (0.62) | 1.81 (0.41) | -0.98 / 2.05 |
| run 2 alone, K 15 / K 40 | 8 | -0.41 / -0.68 (0.67 / 0.65) | 1.88 / 1.79 | -0.66 / -0.94 |
| tuned test-like regime, realised | | -1.29 (cluster SE 0.16) | 1.70 | |
| public R1, realised | | -0.71 (0.12) | 1.42 | |

For run 1 the three readings are, on the plain logit: all lower roots -1.64 /
1.49 (the regime's default), p6 and p8 upper -1.08 / 1.99 (the audit's), p6,
p8 and p9 upper -0.98 / 2.05 (this reading). Run 2's q1 reads as high-rate at
K 15 and low-rate at K 40; run 2's q3 (B31 above 0.25) has no real root and
reads as 0.5.

**Against the shipped LEVEL.** On the pair-logit scale at attribute score 0,
LEVEL implies a mean of -1.15, a between-benchmark sd of 1.15 and a within sd
of 1.10 (factor 0.46 for the item variance 9.51). The 17 pairs split into a
between-benchmark sd of 1.20 (K 15) or 1.49 (K 40), with a 90% interval from 0
to about 1.7, and a within sd of 1.40 or 1.01. That is consistent with LEVEL.

**Two components are not supported.** A two-component normal fit beats one by
0.68 and 0.38 in BIC (the rule's bar is 2) and is preferred in 0% and 0.4% of
bootstrap draws. The only structure is two pairs near a zero rate (a
component of weight 0.12 at -4.4).

**Decision: no candidate; LEVEL stays.** The pooled mean is +1.24 (K 15) and
+1.00 (K 40) SEs above the regime's, and the sd +0.29 and +0.14 SEs. The bias
guard did not trip (the reading shifts the replica's mean by +0.036 and its sd
by -0.131). The pooled level is more central than the tuned regime's, about
as central as public R1's, but within 1.0 to 1.2 SEs of the regime.

**Run 2's B0 excess by side** (descriptive): at K 15, the 4 pairs read below a
rate of 0.5 have a mean B0 minus B31 of 0.047, the 3 read above it 0.078, and
the unidentified q3 0.007 (K 40: 5 below, 0.052; 2 above, 0.082). The largest
single excess, 0.155, is the pair near a zero rate (q8). The excess does not
sit mainly on low-rate pairs, where an optimistic subject prior would put it.

**The read stage ran four times.** It ran at 05:13:48, 05:15:57, 05:19:08 and
05:20:19 under three script versions (digests 5099ef81, 321c189e and
93cb6a84, the committed file); the stored reading is the last. The rule's
text was byte-identical throughout, since `read` refuses to run when the
stored rule and the script's differ. What changed:

* between the preregistration (d3fe28fd) and the first reading, the `record`
  stage learned to compare a rebuilt archive with the uploaded one; the read
  stage was untouched;
* after the first reading (05:15:31), the descriptive split of run 2's B0
  excess was extended from K 15 to both K and now keeps q3, whose B31 has no
  real root, apart; the first version split the pairs 4 and 4 at K 15 (0.047
  below a rate of 0.5, 0.060 at or above it, q3 counted above). Four
  explanatory notes were added;
* after the second reading (05:18:45), each robustness variant of the shipped
  config's estimate on run 1 also reports its neighbours' match distance.

`python experiments/script_revisions.py --stage replay` rebuilds every
version from the committed file and a log of these edits
(`experiments/script_edits.json`, recovered from the session log of the lane
that made them) and checks each against the digest in `passes`; `--stage
reread` re-runs the read stage under each earlier version. The
preregistration's version and both earlier reading versions give the stored
decision exactly (z_mean, z_sd, the bias guard, the outcome), and the same
value in every field they share with the stored reading: 1,502 fields for
d3fe28fd and 5099ef81, 1,559 for 321c189e, none differing
(`results/script_revisions.json`). The rest are the descriptive fields the
edits added or reshaped. `tests/test_script_revisions.py` checks the replay
and the stored re-reads.

**What this cannot settle.** The 17 pairs sit on 7 benchmark ids, so the level
mean has an SE of about 0.5 logit and the between-benchmark sd is barely
identified. The reading-bias guard reads roots on the true side; how often the
neighbour rule picks the wrong side is not measured (q1 flips between K 15
and K 40, and run 1's p9 has neighbour shares of 0.73 and 0.55). The real test
of the audit's reasoning is to score the shipped config, the recommended one,
the EB level, a calibrated smoothed mean and the legacy Predictor in a regime
with level_mean near -0.7 to -1.1 and level_sd near 1.8; the two regimes of
"What actually shipped, after the audit" come close, and that experiment was
not run. (It has since been run, under a rule fixed before scoring, and found
no candidate; LEVEL stays: "Regime sensitivity at the feedback's reading".) The archive paths and rebuild trees in the results file point into
session scratch, which is not durable; the hashes are the lasting record.

## Formative run 3

`python experiments/formative_run3.py` (0.3 s on one process; no model is
run and nothing in `data/` is read). Every number below is in
`results/formative_run3.json`, its section named in brackets;
`tests/test_formative_run3.py` (14 tests) covers the script. It records run 3
the way the `record` stage recorded runs 1 and 2, with the parsing, summary,
overlap and archive helpers imported from `experiments/formative_feedback.py`,
and changes neither that script nor `results/formative_feedback.json` (a test
checks both hashes). It writes the file only if every required check passes:
32 checks ran, all of them required, and all passed (checks). The input is
`results/formative/run3.txt`, the organisers' per-pair table as returned, byte
for byte, like `run1.txt` and `run2.txt`.

**What was uploaded, and why** (run3, archive). The archive selected at
4d2cc4f, sha256
`4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090`: hier with
LEVEL mu0 -2.5, sigma_mu 2.5, attr_scale 0.5, the corrected multiple-choice
floor and the floored-fit fix. The team uploaded it as a regression and
latency check only; its score selects or tunes nothing. That purpose was
stated before the score existed: `results/formative_feedback.json` lists run 3
as unscored, with this archive and that purpose (`submissions`), a statement
first committed in 00bdf04 on 2026-09-28, three days before the score
(run3.purpose_stated_before_scoring). The file in `dist/` has these bytes,
every tracked member equals 4d2cc4f's, its `prior.json` equals
`submission/prior.json`, and model.py's LEVEL is -2.5 / 2.5 / 0.5. It is the
file handed to the team for upload; unlike runs 1 and 2's archives, it was
not downloaded back from the platform (archive.downloaded_back). `dist/` is
gitignored and rebuilt by `tools/build_submission.py`, so re-verifying needs
it to hold the `4a882cc7…` file: read from `dist/`, the archive checks are
required and nothing is written without it. A copy given with `--archive` is
recorded whatever its bytes, but a stored record whose archive matched is
replaced by one whose archive does not only with `--replace-archive-record`.

**The run** (run3.counts, run3.alc). Scored 2026-10-01: 9 pairs on 7
benchmarks (2, 2, 1, 1, 1, 1, 1 a benchmark), 9 subjects, 485 evaluated
subject-item pairs (44 to 72 a pair). Every pair is in every budget's table,
every value is finite and in [0, 1], and the team reports no errors
(run3.regression_check). The platform's headline score was not pasted, so the
pair-mean rule pinned on runs 1 and 2 is applied here, not tested: it gives
0.1816525778 (0.181653 at six decimals). Weighting pairs by their items would
give 0.1789. Each pair's summary ALC and ECE-ALC match its budget rows to
within 5.0e-7.

| pair | benchmark | n | B0 | B1 | B3 | B7 | B15 | B31 | ALC | ECE (ALC-weighted) | ECE at B0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | G | 44 | 0.2768 | 0.1941 | 0.2119 | 0.1891 | 0.1935 | 0.2067 | 0.2061 | 0.109 | 0.295 |
| s2 | E | 60 | 0.2543 | 0.2962 | 0.3617 | 0.2578 | 0.2635 | 0.2514 | 0.2864 | 0.163 | 0.069 |
| s3 | D | 58 | 0.2071 | 0.1307 | 0.0989 | 0.0927 | 0.0928 | 0.0947 | 0.1132 | 0.095 | 0.338 |
| s4 | C | 56 | 0.2024 | 0.1696 | 0.1873 | 0.1755 | 0.1821 | 0.1865 | 0.1818 | 0.120 | 0.191 |
| s5 | A | 72 | 0.2122 | 0.1377 | 0.1307 | 0.1470 | 0.1303 | 0.1178 | 0.1421 | 0.093 | 0.288 |
| s6 | A | 53 | 0.2254 | 0.1413 | 0.1304 | 0.1233 | 0.1326 | 0.1215 | 0.1402 | 0.095 | 0.312 |
| s7 | G | 44 | 0.2621 | 0.1619 | 0.1923 | 0.1453 | 0.1487 | 0.1508 | 0.1709 | 0.125 | 0.341 |
| s8 | B | 54 | 0.2150 | 0.1365 | 0.1044 | 0.1090 | 0.1023 | 0.1001 | 0.1219 | 0.124 | 0.341 |
| s9 | F | 44 | 0.2606 | 0.3136 | 0.2790 | 0.2581 | 0.2548 | 0.2500 | 0.2722 | 0.139 | 0.116 |
| mean | | | 0.2351 | 0.1868 | 0.1885 | 0.1664 | 0.1667 | 0.1644 | 0.1817 | 0.118 | 0.254 |

Letters are those of runs 1 and 2 (run3.benchmark_letters), for recording
only. Mean ECE by budget, B0..B31: 0.254, 0.146, 0.135, 0.077, 0.073 and
0.066. The largest pair ECE at B0 is 0.341, and the mean B0 excess over B31
0.071 (runs 1 and 2: 0.202 and 0.054).

**The low budgets** (run3.regression_check.low_budget_profile). The run's
mean Brier falls by 0.048 from B0 to B1 and rises by 0.0017 from B1 to B3:
the means show no B1 or B3 spike. Two pairs have B1 or B3 above their B0: s2
(E; B0 0.254, B1 0.296, B3 0.362) and s9 (F; 0.261, 0.314, 0.279), both with a
B31 of at least 0.25, so both near a rate of 0.5. Runs 1 and 2 had one such
pair each (three_runs).

**Overlap** (overlap). Run 3's 7 benchmark ids are those of runs 1 and 2. It
shares no subject with run 2 and one with run 1: subject_431933 on benchmark
A, 53 evaluated items both times (run 1's p7, run 3's s6). Over the three runs
that makes 25 distinct subjects in 26 pair appearances. On that pair run 3's
ALC is 0.0663 lower than run 1's, its B0 0.252 lower (0.225 against 0.477),
and by budget the differences are -0.252, -0.153, -0.043, -0.010, +0.004 and
-0.007. That is descriptive only: the two runs ran different models, and their
other pairs, so the shared `labeled` list, differ. Each benchmark's item
counts across the three runs are in overlap.all_three.n_by_benchmark.

**Against the shipped model's regimes** (placement). As run 2 in "Shipped
configuration, confirmed": the distance from each regime's mean in the shipped
model's single-run sds, positive worse. The regimes' rows are run 2's library
(hier 70a3a81a, the old floor, the solver before the floored-fit fix). The gap
to run 3's archive, at most 0.00046 in ALC on runs 0 to 199
(placement.code_gap_not_added), is not added. Run 2 is placed again from its
exact budgets, after a check that the stored z values of `ship_confirm.json`
(budgets rounded to three decimals) are reproduced.

| regime (runs) | mean ALC | single-run sd | run 3's z | run 3's z by budget, B0..B31 | run 2's z, exact budgets | run 3 minus run 2, in sds of the difference |
|---|---|---|---|---|---|---|
| test-like (300) | 0.1658 | 0.0318 | +0.50 | +0.89 -0.06 +0.65 +0.39 +0.65 +0.81 | +0.84 | -0.24 |
| test-like, mix/whole (200) | 0.1711 | 0.0329 | +0.32 | +1.01 -0.16 +0.52 +0.21 +0.40 +0.58 | +0.65 | -0.24 |
| test-like, no date shift (100) | 0.1585 | 0.0335 | +0.69 | +0.95 +0.25 +0.69 +0.57 +0.83 +1.02 | +1.02 | -0.23 |
| public R1 benchmark-first (150) | 0.2051 | 0.0285 | -0.82 | +0.00 -1.00 -0.70 -1.08 -0.70 -0.50 | -0.44 | -0.27 |
| public R1 pair-uniform (100) | 0.2020 | 0.0261 | -0.78 | -0.30 -0.88 -0.62 -0.95 -0.57 -0.26 | -0.36 | -0.30 |

Run 3 lies 0.50 sd above the tuned regime's mean and 0.82 below public
benchmark-first. Its B0 is 0.89 sd above the tuned regime's and level with
public R1 (+0.00 and -0.30); from B1 on it sits 0.3 to 1.1 sd below public R1.
Runs 2 and 3, one draw each, both lie within 1.02 single-run sds of every
regime's mean ALC. Run 3 minus run 2 is -0.0110 in ALC, 0.23 to 0.30 sds of
the difference of two independent runs (sd 0.037 to 0.047). The share of a
regime's runs at or above run 3 is not computed: `ship_confirm.json` stores no
per-run ALC.

**Three runs side by side** (three_runs).

| run | model; commit | pairs (benchmarks, subjects) | items | B0 | B1 | B3 | B7 | B15 | B31 | ALC | benchmark-balanced ALC | mean ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | legacy Predictor; b68492c | 9 (7, 9) | 485 | 0.3589 | 0.2539 | 0.2043 | 0.1712 | 0.1693 | 0.1568 | 0.2113 | 0.2139 | 0.170 |
| 2 | hier, old floor; ee5085a | 8 (7, 8) | 492 | 0.2368 | 0.1947 | 0.1963 | 0.1836 | 0.1785 | 0.1833 | 0.192623 | 0.1909 | 0.100 |
| 3 | hier, corrected floor and floored-fit fix; 4d2cc4f | 9 (7, 9) | 485 | 0.2351 | 0.1868 | 0.1885 | 0.1664 | 0.1667 | 0.1644 | 0.181653 (recomputed) | 0.1865 | 0.118 |

The benchmark-balanced ALC weights the 7 shared benchmark ids equally; the
subjects differ, so it is descriptive only. Run 1 is not placed: the shipped
model's single-run sds do not describe the legacy Predictor.

**What three runs can and cannot measure**
(what_three_runs_can_and_cannot_measure). They confirm the per-pair scoring
rule once more. They show that the archive selected at 4d2cc4f runs end to end
on the platform, every pair at all six budgets with no errors reported, which
is the regression check it was uploaded for, and that the formative evaluation
drew on the same 7 benchmark ids each time. They cannot compare predictors or
configurations: the subjects differ between runs, and the one recurring pair
ran under two models with different co-sampled pairs. They cannot detect a
change of the size measured offline: the difference of two independent runs
has an sd of 0.037 to 0.047, against a code gap from run 2's archive to run
3's of at most 0.00046. And they do not measure latency: the table records no
timing, so the run finishing with every pair scored is the only latency
evidence.

**No reading, no tuning** (no_reading_or_tuning). No level reading, matching
or tuning is done on run 3, and nothing in the model, its hyperparameters or
the archive changes because of it. The pooled reading's preregistered rule
("The pooled reading, under a rule fixed in advance", sha256 `0de18448…`,
outcome "no candidate: LEVEL stays") covered runs 1 and 2 only, and run 3 is
not added to it.

**Open.** The headline score was not pasted (run3.platform_alc is null); once
it is, the record should check that it equals the recomputed 0.181653. The
archive record is of the file in `dist/`; a copy downloaded back from the
platform would close it as for runs 1 and 2 (`--archive PATH --archive-how
TEXT`). `results/formative_feedback.json` still lists run 3 as unscored: it is
left as it is, because `experiments/script_revisions.py` audits that file.

## Regime sensitivity at the feedback's reading (P1.10, P1.16)

`python experiments/regime_sensitivity.py S` for S = `lock`, `smcal`,
`regimes --grid` and `reproduce`, then `score --shard 0/2` and
`score --shard 1/2` side by side, then `summarise`. The `review` stage was
added after scoring; its commands are in the script's header. A scoring task
(one run, every config) took 31 to 40 s on average (`summary.rows.task_s`).
The 520 rows were written between 16:44:02 and 19:15:28 UTC on 2026-10-01,
on two processes of a machine shared with other jobs, at most 1.0 GB resident
each (`summary.rows.max_rss_gb`). Every number below is in
`results/regime_sensitivity.json`, under the key in brackets. The rows are in
`data/regime_sensitivity_rows/` (gitignored, 520 files).
`tests/test_regime_sensitivity.py` (51 tests) covers the script. The plan,
with the rule quoted verbatim and amendments appended after review, is
`docs/plans/p1a_regime_sensitivity.md`. `paiec/testlike.py` gained one
option for the study, `Regime.level_mix`, a Gaussian-mixture level target.
When the option is empty the sampler is unchanged.

**Outcome.** No config is a candidate, so the shipped configuration stays
(`rule.outcome`: "no candidate: SHIP stays").

* A wider level prior does not do better (`rule.p116`).
* Neither adaptive empirical-Bayes level is a candidate (`rule.eb`).
* Every config is recorded as a negative result (`rule.negative`).

That outcome assumes deviation D1, which changed how one reproduction check
was compared before any scoring-seed row existed ("Provenance and
deviations", below). Without D1, the rule read literally gives "rule not
applied: check failed" (`rule.literal_reading`). Neither reading has a
candidate, because no config meets condition (a). The team accepted D1 on
2026-10-02, so the recorded outcome is "no candidate: SHIP stays"; the
conclusion does not depend on it. Nothing in `submission/` changed.

**What it asks.** The audit moved the archive to the milder level because the
hidden levels "look spread both ways" ("What actually shipped, after the
audit"). The pooled reading of runs 1 and 2 then put them near the public
centre ("Formative feedback, runs 1 and 2"). No regime with that level and
spread had been scored for the shipped configuration. The internal review
(`docs/report/review_v0.md`) asked three questions:

* **W3, Q3 and P1.10.** How do the shipped configuration, the recommended
  -3.0/0.25, the empirical-Bayes level, a calibrated smoothed mean and the
  legacy Predictor rank in regimes at the feedback's level reading, and under
  a two-component mixture?
* **P1.16.** Does sigma_mu 3.5 or 5, at the shipped mu0 and attr_scale, do
  better?
* **W4 and P1.9** (descriptive here). How much of the gain do a
  level-calibrated smoothed mean and hier without its subject prior recover?

### The plan and the rule, and when they were fixed

The plan and its decision rule were fixed at 14:56:59 UTC on 2026-10-01, at
HEAD 380fecf, before any run of the scoring seed was drawn
(`lock.fixed_at_utc`). Three hashes pin them:

* the rule's text, sha256 `47e4735c…`;
* the plan file as fixed, `0a857944…`;
* the script's constants block, `c28c19d6…`, which holds the rule, the
  configs, the regimes, the seeds, the run counts and the thresholds.

The `lock` stage checks all three, and 12 of 12 lock checks pass
(`lock.checks`).

Nothing was committed before scoring, so the time stamp is self-reported. The
evidence for it is local, file mtimes and copies in session scratch, not a
commit (`review.provenance.evidence`):

| UTC, 2026-10-01 | event |
|---|---|
| 14:50:26 | the planner's rule text last written (its sha256 is the rule's) |
| 14:56:59 | the plan fixed (`FIXED_AT_UTC`, and the plan file's mtime) |
| 15:26:54 | seed 11 first drawn, by the regimes stage (run composition only, no predictor) |
| 15:40:07 | the first reproduce run ends; D1 is found |
| 16:26:39 | reproduce passes with D1's fix |
| 16:44:02 | the first scoring-seed row |
| 16:56:16 | a dry run of summarise on the 19 rows scored so far |
| 19:15:28 | the last of the 520 rows |
| 19:16:01 | summarise runs, and the rule is applied once |

Scratch is not durable. The hashes and times copied into the results file are
the lasting record.

**The rule** is quoted verbatim in section 8 of the plan. Its terms:

* D(X, R) is a config's mean run ALC minus SHIP's, on the same runs of
  regime R.
* U95 is D plus 1.96 cluster SEs.
* Pq is the difference over parent q's pair appearances, each weighted by
  1 / its run's pair count.
* PL is the mean of Pq over the four multi-subject parents.

A config is a candidate to replace SHIP only if all six conditions hold:

* (a) in READING and in AUDIT, D ≤ -0.002 and U95 < 0;
* (b) in READING and in AUDIT, PL < 0, and no multi-subject parent above
  +0.004;
* (c) in TUNED, MIXTURE and FLAT, D ≤ +0.002;
* (d) on both public weightings, D ≤ +0.001, and X minus the legacy
  Predictor ≤ +0.003;
* (e) X's mean evaluation call takes at most 2.0 times SHIP's, and no single
  call of X takes over 2.0 s;
* (f) every reproduction and realisation check passed, and every planned run
  was scored.

A candidate would go to the team, not into the archive. Beyond the task's
floor, the plan added the parent-level sign in (b), FLAT in (c), the legacy
guard in (d), and conditions (e) and (f), the same for every config. (b)'s
per-parent cap of +0.004 is looser than the +0.002 of the pooled reading's
gate. No config reached (a), so the cap decided nothing.

The plan's power estimate (section 7) assumed hier-versus-hier cluster SEs
near 0.001. A config truly 0.004 better then passes (a) in one regime with
probability about 0.98, one 0.003 better about 0.84, and one 0.002 better
about 0.5.

Formative run 3 is used for nothing here.

**Seeds.** The regime knobs were set on seed 10 by run composition alone,
with no predictor. Every scored run uses seed 11: runs 0 to 79 of each
test-like regime and 0 to 59 of each public weighting, 520 tasks in all.
Neither seed had been used before.

Fresh seeds are not fresh data, though. Every test-like regime redraws the
catalogue cut from the same four parents. The public runs redraw the same
221 pairs that chose LEVEL and the recommended config, and guarded them. So
TUNED and the guards of (c) and (d) carry that selection's optimism, toward
SHIP or the recommended config, and do not replicate it independently.

### Configs

Every hier config uses `prior.build` fitted without the target's parent
benchmark. So do the legacy Predictor's attribute prior and every
empirical-Bayes hyperparameter (`plan.configs`).

| key | what it is |
|---|---|
| `ship` (SHIP) | hier, LEVEL mu0 -2.5, sigma_mu 2.5, attr_scale 0.5; the reference |
| `aggr` | hier, mu0 -3.0, sigma_mu 2.5, attr_scale 0.25; the config "Calibrating for the hidden test" recommended |
| `eb_fit` | hier with the hyperparameters `prior.build` fits with the target's parent held out, nothing overridden (attr_scale 1) |
| `eb_adapt` | the pair-level empirical-Bayes level of "Calibrating for the hidden test", on mu0 -3.0, sigma_mu 2.5, attr_scale 0.5; centre and scale are re-estimated from `labeled` at every checkpoint, with tau 2 |
| `eb_ship` | the same adaptation on SHIP's level; identical to SHIP at B0, so its difference from SHIP is the adaptation alone |
| `wide35`, `wide50` | SHIP with sigma_mu 3.5 and 5.0 (P1.16) |
| `legacy` | the legacy Predictor, the first submission |
| `smooth` | the smoothed mean, Beta(2,2) |
| `smcal` | a smoothed mean with a calibrated prior, (k + n0 m0) / (n + n0) on the pair's own labels (below) |
| `onepl` | hier at SHIP's level with the attribute prior and identity link off: a pooled level, item difficulty, group effects and the multiple-choice floor, and no subject prior |

**smcal's prior** was chosen by LEVEL's own rule, on LEVEL's own runs
(`smcal`).

* **The grid.** n0 in {0.5, 1, 2, 4, 8, 16, 32} times m0 in {0.05, 0.10, ...,
  0.50}, 70 points.
* **The rule.** Take the best ALC on test-like seed 2 runs 0 to 99, among the
  points that lose at most 0.003 against the legacy Predictor's stored rows
  on public seed 0 runs 0 to 99, on both weightings.
* **No point passes the guard.** The smallest losses are +0.0023
  (benchmark-first) and +0.0040 (pair-uniform), both at (4, 0.4).
* **So smcal is the unguarded best.** As the plan fixed, smcal is the best
  point on the selection half regardless of the guard: n0 2 and m0 0.25, ALC
  0.1588. It is flagged `guard_failed`, and it loses +0.0077 and +0.0096 on
  the guard. Beta(2,2) scores 0.1788 there, with guard losses of +0.0070 and
  +0.0075, as in "Calibrating for the hidden test".
* **The guard's reference is stale, and the choice stands.** The stored
  legacy rows predate the corrected multiple-choice floor. Against the
  current legacy Predictor every guard loss grows, the best point's to
  +0.0027 and +0.0047. No point passes and the choice is unchanged
  (`review.smcal_guard`).

smcal's level is calibrated to the tuned regime only, and was chosen in
sample on its catalogue. TUNED is therefore the smoothed mean's best case.

**onepl's centre.** With the attribute prior off, mu0 is no longer measured
from an attribute score of 0, and the date shift's attribute optimism is
gone. onepl's mean B0 prediction is 0.31, against SHIP's 0.42, on test-like
runs (`summary.configs.<regime>.<config>.q0`). onepl minus SHIP therefore
mixes the subject prior with a lower centre. It is also not the plain 1PL the
review asks for, which would drop the group effects and the floor as well.

### Regimes and what they realised

Every test-like regime is `testlike.Regime()` at its defaults, except for the
level knobs named: the 1.25-year date shift, swe_rebench excluded, and a
benchmark-level tilt. The realised level is the continuity-corrected logit of
each pair appearance's evaluated responses, as in `testlike_check` and the
pooled reading. Its mean and sd are taken over every appearance
(`summary.realised`, `review.levels`).

| regime | knobs | target | seed 10 (1,000 draws), mean / sd | seed 11, scored: mean (cluster SE) / sd | share below -3 / -4 |
|---|---|---|---|---|---|
| TUNED | the defaults (level_mean -1.6, level_sd 1.5) | -1.290 / 1.696, realised (`testlike_check.json`) | -1.273 / 1.731 | -1.276 (0.152) / 1.621 | 0.121 / 0.028 |
| READING | level_mean -0.85, level_sd 1.75 | -0.715 / 1.772, the mean of the pooled reading's K 15 and K 40 values | -0.729 / 1.793 | -0.627 (0.169) / 1.678 | 0.076 / 0.015 |
| AUDIT | level_mean -1.8, level_sd 2.1 | -1.10 / 2.00, the review's statement of the audit's reading | -1.102 / 1.997 | -0.930 (0.219) / 2.054 | 0.136 / 0.060 |
| MIXTURE | `level_mix` (0.20, -4.38, 0.171) + (0.80, -0.75, 1.151) | share below -3 of 0.118 and a rest mean of -0.226, from the reading's two-component fits | -0.705 / 2.062 (share 0.118, rest -0.227) | -0.642 (0.188) / 2.126 (share 0.110, rest -0.201) | 0.110 / 0.057 |
| FLAT | no level tilt | none | -0.338 / 2.287 | -0.280 (0.216) / 2.225 | 0.098 / 0.031 |
| R1B | public R1, benchmark-first, split scope 'pair' | none | | -0.556 (0.116) / 1.346 | 0.055 / 0.019 |
| R1P | public R1, pair-uniform, split scope 'pair' | none | | -0.767 (0.115) / 1.465 | 0.080 / 0.025 |

Each knob was set by a search over run composition on seed 10. `regimes
--grid` re-ran that search over the full knob grid. Its argmin is the fixed
knob for READING, AUDIT and MIXTURE (`regimes.grid`).

The scored runs passed the realisation checks, whose bounds were fixed before
scoring: ±0.30 for the mean and the sd, ±0.04 for the share and ±0.25 for the
rest mean. The largest shift from seed 10 is AUDIT's mean, +0.173
(`regimes.checks`).

* **The feedback regimes realised milder levels than they targeted.** READING
  and AUDIT span -0.63 to -0.93 on the scored runs, not the planned -0.7 to
  -1.1. Together they bracket the pooled reading's means (-0.65 and -0.78),
  but they fall short of the -1.10 the review gave for the audit's reading.
* **MIXTURE's low mode exists, but it is narrow.** 36 of its 627 appearances
  (0.057) sit below -4, against 0.015 in READING. 26 of the 36 come from two
  multi_swebench pseudo-benchmarks (`q1of3` and `q1of2`), and 8 more from
  `real_webagents::all`. Its share below -3 (0.110) is lower than TUNED's
  (0.121) and AUDIT's (0.136). As a stress test it is weaker than its name.
  Its main component is also wider than the reading's (rest sd 1.80 against
  1.15), because the catalogue cannot realise a tight high mode.
* **FLAT is the most central test-like regime** (-0.28), more central than
  public R1 (-0.56 and -0.77). All five test-like regimes keep the synthetic
  date shift.
* **The parents set the level.** In every test-like regime matharena sits
  high (+0.09 in TUNED to +1.68 in FLAT) and multi_swebench low (-1.81 to
  -2.34). Leaving one parent out moves READING's mean anywhere from -1.11
  (matharena out) to -0.06 (multi_swebench out).

### Checks

* **smcal.** The grid's (4, 0.5) column reproduces `level_calibration.json`'s
  smoothed rows to 5.5e-6, on 300 runs and 2,539 pairs (`smcal.check`).
* **Reproduction**, on test-like seed 2 runs 0 to 4 (`reproduce.checks`):
  * SHIP's per-pair Brier matches `data/harness_rows` to at most 1.4e-16
    (tolerance 1e-9).
  * The smoothed mean matches `level_calibration.json` to at most 4.9e-6
    (tolerance 1e-5).
  * The legacy Predictor, off matharena, matches `testlike_check.json` to
    4.7e-7 to 5.0e-7. That fails the planned 1e-9 and passes at the
    comparator's six stored decimals (D1, below).
* **After scoring** (`summary.checks`):
  * Every planned task was scored, 400 test-like and 120 public.
  * All 520 rows carry one set of library digests and one script digest, and
    their run compositions match the regimes stage's.
  * The TUNED consistency check compared SHIP minus the legacy Predictor,
    -0.0434 ± 0.0019 / 0.0037 (run / cluster SE), with the -0.0415 ± 0.0034
    of "Shipped configuration, confirmed". That is z -0.37, against a bound
    of 3 (`summary.tuned_consistency`). The two numbers come from different
    libraries (below) and different seeds.

**Which code these numbers describe.** Every hier config ran on
`paiec/hier.py` d9a95612, the code of the archive now selected (built at
4d2cc4f, sha256 `4a882cc7…`), which has the corrected multiple-choice floor
and the floored-fit fix. The rows record the digests of the library they ran
on (`summary.rows.library_digests`). These equal the files at 4d2cc4f for
`paiec/hier.py`, `prior.py`, `subjects.py`, `mcq.py`, `predict.py`,
`fitting.py`, `official.py` and `submission/model.py`. Only
`paiec/testlike.py`, which does not ship, differs, by `level_mix`. "Shipped
configuration, confirmed" used rows of run 2's library. The gap between the
two libraries was measured there at under 0.0005.

### Configs against SHIP

X minus SHIP in ALC (lower is better), with the cluster SE in brackets. SHIP's
own ALC ± run SE is in the first row (`summary.vs_ship.<regime>.<config>.exact`,
`summary.configs`).

| config | TUNED | READING | AUDIT | MIXTURE | FLAT | R1B | R1P |
|---|---|---|---|---|---|---|---|
| SHIP, ALC | 0.1690 ± 0.0030 | 0.1881 ± 0.0029 | 0.1650 ± 0.0034 | 0.1830 ± 0.0031 | 0.1739 ± 0.0030 | 0.2114 ± 0.0030 | 0.1948 ± 0.0033 |
| aggr | -0.0034 (0.0010) | +0.0015 (0.0011) | +0.0005 (0.0014) | +0.0017 (0.0012) | +0.0033 (0.0014) | +0.0025 (0.0004) | +0.0020 (0.0005) |
| eb_adapt | -0.0027 (0.0006) | +0.0003 (0.0004) | -0.0005 (0.0007) | -0.0001 (0.0004) | +0.0003 (0.0004) | +0.0017 (0.0002) | +0.0020 (0.0002) |
| eb_ship | -0.0014 (0.0004) | +0.0003 (0.0003) | -0.0002 (0.0004) | -0.0001 (0.0003) | -0.0001 (0.0004) | +0.0006 (0.0003) | +0.0014 (0.0003) |
| wide35 | +0.0009 (0.0003) | +0.0006 (0.0003) | -0.0003 (0.0004) | +0.0005 (0.0004) | -0.0010 (0.0005) | +0.0007 (0.0003) | +0.0008 (0.0003) |
| wide50 | +0.0031 (0.0007) | +0.0023 (0.0007) | +0.0006 (0.0008) | +0.0022 (0.0008) | -0.0007 (0.0009) | +0.0026 (0.0007) | +0.0025 (0.0007) |
| onepl | -0.0030 (0.0012) | +0.0026 (0.0014) | +0.0016 (0.0017) | +0.0031 (0.0014) | +0.0049 (0.0017) | +0.0025 (0.0005) | +0.0027 (0.0004) |
| smcal | -0.0015 (0.0019) | +0.0078 (0.0021) | +0.0043 (0.0024) | +0.0071 (0.0021) | +0.0103 (0.0024) | +0.0108 (0.0014) | +0.0100 (0.0018) |
| eb_fit | +0.0309 (0.0035) | +0.0173 (0.0032) | +0.0233 (0.0042) | +0.0164 (0.0028) | +0.0140 (0.0038) | -0.0010 (0.0010) | +0.0009 (0.0011) |
| smooth | +0.0170 (0.0022) | +0.0096 (0.0018) | +0.0142 (0.0024) | +0.0110 (0.0020) | +0.0103 (0.0018) | +0.0075 (0.0029) | +0.0127 (0.0021) |
| legacy | +0.0434 (0.0037) | +0.0248 (0.0037) | +0.0313 (0.0049) | +0.0263 (0.0037) | +0.0200 (0.0044) | +0.0002 (0.0020) | +0.0039 (0.0019) |

**The two regimes the rule reads first**, ± run / cluster / stratified SE,
with U95, the parent-level mean ± SE and the range of the four parents' Pq
(`summary.vs_ship.<regime>.<config>.table.parent_level`):

| config | READING: X minus SHIP | U95 | parent-level | parents | AUDIT: X minus SHIP | U95 | parent-level | parents |
|---|---|---|---|---|---|---|---|---|
| aggr | +0.0015 ± 0.0006 / 0.0011 / 0.0009 | +0.0037 | +0.0016 ± 0.0033 | -0.0058 to +0.0103 | +0.0005 ± 0.0006 / 0.0014 / 0.0011 | +0.0033 | +0.0015 ± 0.0046 | -0.0080 to +0.0142 |
| eb_adapt | +0.0003 ± 0.0003 / 0.0004 / 0.0004 | +0.0012 | +0.0005 ± 0.0010 | -0.0025 to +0.0020 | -0.0005 ± 0.0003 / 0.0007 / 0.0006 | +0.0008 | +0.0002 ± 0.0020 | -0.0051 to +0.0041 |
| eb_ship | +0.0003 ± 0.0003 / 0.0003 / 0.0003 | +0.0010 | +0.0005 ± 0.0005 | -0.0004 to +0.0016 | -0.0002 ± 0.0003 / 0.0004 / 0.0004 | +0.0006 | +0.0002 ± 0.0009 | -0.0023 to +0.0021 |
| wide35 | +0.0006 ± 0.0002 / 0.0003 / 0.0003 | +0.0012 | +0.0007 ± 0.0006 | -0.0006 to +0.0017 | -0.0003 ± 0.0002 / 0.0004 / 0.0003 | +0.0004 | -0.0000 ± 0.0010 | -0.0026 to +0.0020 |
| wide50 | +0.0023 ± 0.0005 / 0.0007 / 0.0007 | +0.0037 | +0.0027 ± 0.0011 | +0.0004 to +0.0047 | +0.0006 ± 0.0004 / 0.0008 / 0.0007 | +0.0021 | +0.0012 ± 0.0021 | -0.0036 to +0.0053 |
| onepl | +0.0026 ± 0.0007 / 0.0014 / 0.0012 | +0.0054 | +0.0028 ± 0.0040 | -0.0064 to +0.0131 | +0.0016 ± 0.0007 / 0.0017 / 0.0014 | +0.0049 | +0.0028 ± 0.0055 | -0.0088 to +0.0176 |
| smcal | +0.0078 ± 0.0010 / 0.0021 / 0.0016 | +0.0119 | +0.0083 ± 0.0066 | -0.0083 to +0.0238 | +0.0043 ± 0.0010 / 0.0024 / 0.0019 | +0.0090 | +0.0062 ± 0.0081 | -0.0118 to +0.0274 |
| eb_fit | +0.0173 ± 0.0017 / 0.0032 / 0.0024 | +0.0237 | +0.0156 ± 0.0118 | -0.0067 to +0.0488 | +0.0233 ± 0.0017 / 0.0042 / 0.0030 | +0.0315 | +0.0175 ± 0.0153 | -0.0136 to +0.0595 |
| smooth | +0.0096 ± 0.0009 / 0.0018 / 0.0016 | +0.0132 | +0.0091 ± 0.0035 | +0.0022 to +0.0189 | +0.0142 ± 0.0010 / 0.0024 / 0.0021 | +0.0189 | +0.0118 ± 0.0062 | -0.0003 to +0.0290 |
| legacy | +0.0248 ± 0.0020 / 0.0037 / 0.0032 | +0.0321 | +0.0239 ± 0.0100 | +0.0011 to +0.0498 | +0.0313 ± 0.0020 / 0.0049 / 0.0040 | +0.0408 | +0.0265 ± 0.0153 | -0.0099 to +0.0648 |

How close the configs come to SHIP in these two regimes (`review.headline`):

* **Three configs sit within ±0.0006 of SHIP in both READING and AUDIT:**
  wide35, eb_adapt and eb_ship. Their cluster SEs are 0.0003 to 0.0007,
  smaller than the 0.001 the plan assumed. For these three, then, missing (a)
  most likely means any gain is under 0.002, not that the test lacked power.
  The closest call is eb_adapt in AUDIT, -0.0005 with a cluster SE of 0.0007.
* **aggr, wide50 and onepl lose 0.0005 to 0.0026,** with cluster SEs of
  0.0007 to 0.0017.
* **Across every hier config** the cluster SEs run from 0.0003 to 0.0042.

### Per parent

Every config with a lower centre than SHIP trades the same two parents:

| config, against SHIP | multi_swebench | matharena |
|---|---|---|
| aggr | gains 0.0055 to 0.0085 | loses 0.0054 in TUNED and 0.0103 to 0.0159 elsewhere |
| smcal | gains 0.0070 to 0.0130 | loses 0.0172 to 0.0319 |
| onepl | gains 0.0059 to 0.0093 | loses 0.0077 to 0.0196 |
| eb_adapt | gains 0.0021 to 0.0058 | loses 0.0011 to 0.0041 |

(Each range runs over the five test-like regimes; source:
`summary.vs_ship.<regime>.<config>.table.per_parent`.)

multi_swebench is the lowest parent in every regime and matharena the
highest. The ranking therefore turns on how a regime weights these two
parents. The parent-level SE, which sees that, is two to four times the
cluster SE.

SHIP against the legacy Predictor shows the same split
(`summary.vs_legacy.<regime>.ship.per_parent`):

* On matharena, SHIP's difference moves from -0.0241 in TUNED to +0.0099 in
  AUDIT and +0.0160 in FLAT, where SHIP loses on that parent.
* On multi_swebench it stays between -0.050 and -0.065.

### Against the legacy Predictor and the smoothed means

SHIP minus each comparator on the same runs, ± run / cluster / stratified SE
(`summary.vs_legacy.<regime>.ship`, `summary.vs_smooth.<regime>.ship`,
`summary.decomposition.<regime>`):

| regime (realised level) | SHIP ALC | legacy ALC | SHIP minus legacy | parent-level | parents (range) | runs won | SHIP minus smoothed Beta(2,2) | SHIP minus smcal |
|---|---|---|---|---|---|---|---|---|
| TUNED (-1.28) | 0.1690 | 0.2124 | -0.0434 ± 0.0019 / 0.0037 / 0.0034 | -0.039 ± 0.008 | -0.060 to -0.024 | 100% | -0.0170 ± 0.0010 / 0.0022 / 0.0021 | +0.0015 ± 0.0010 / 0.0019 / 0.0015 |
| READING (-0.63) | 0.1881 | 0.2130 | -0.0248 ± 0.0020 / 0.0037 / 0.0032 | -0.024 ± 0.010 | -0.050 to -0.001 | 90% | -0.0096 ± 0.0009 / 0.0018 / 0.0016 | -0.0078 ± 0.0010 / 0.0021 / 0.0016 |
| AUDIT (-0.93) | 0.1650 | 0.1963 | -0.0313 ± 0.0020 / 0.0049 / 0.0040 | -0.026 ± 0.015 | -0.065 to +0.010 | 97.5% | -0.0142 ± 0.0010 / 0.0024 / 0.0021 | -0.0043 ± 0.0010 / 0.0024 / 0.0019 |
| MIXTURE (-0.64) | 0.1830 | 0.2093 | -0.0263 ± 0.0019 / 0.0037 / 0.0032 | -0.027 ± 0.010 | -0.054 to -0.004 | 93.8% | -0.0110 ± 0.0009 / 0.0020 / 0.0018 | -0.0071 ± 0.0010 / 0.0021 / 0.0018 |
| FLAT (-0.28) | 0.1739 | 0.1940 | -0.0200 ± 0.0020 / 0.0044 / 0.0036 | -0.022 ± 0.015 | -0.059 to +0.016 | 83.8% | -0.0103 ± 0.0009 / 0.0018 / 0.0016 | -0.0103 ± 0.0011 / 0.0024 / 0.0019 |
| R1B (-0.56) | 0.2114 | 0.2116 | -0.0002 ± 0.0010 / 0.0020 / 0.0017 | -0.002 ± 0.005 | -0.010 to +0.012 | 51.7% | -0.0075 ± 0.0012 / 0.0029 / 0.0021 | -0.0108 ± 0.0010 / 0.0014 / 0.0012 |
| R1P (-0.77) | 0.1948 | 0.1988 | -0.0039 ± 0.0010 / 0.0019 / 0.0017 | -0.004 ± 0.005 | -0.015 to +0.009 | 71.7% | -0.0127 ± 0.0013 / 0.0021 / 0.0020 | -0.0100 ± 0.0010 / 0.0018 / 0.0016 |

* **The gain over the legacy Predictor follows the level.** It is 0.043 in
  TUNED, 0.031 in AUDIT, 0.025 and 0.026 in READING and MIXTURE, and 0.020 in
  FLAT. That agrees with the audit's two extra regimes, 0.029 at a realised
  -0.77 and 0.022 at -0.59 ("What actually shipped, after the audit").
* **On public runs SHIP is level with the legacy Predictor or slightly
  ahead:** -0.0002 benchmark-first and -0.0039 pair-uniform (0.1 and 2.1
  cluster SEs). Seed 0 gave -0.0017 and -0.0018 ("Shipped configuration,
  confirmed").
* **Against the smoothed mean,** which has no level to tune, SHIP gains 0.010
  to 0.017 in every test-like regime, and 0.008 and 0.013 on public runs.
* **Against smcal,** see "What a calibrated level alone recovers", below.

### The rule, config by config

The numbers that decide each condition are given in the order the condition
names its regimes (`rule.table.<config>`). Under "fails", "both" means the
condition fails in both of its regimes. (f) holds for every config under D1.

| config | (a) READING, AUDIT: D | (b) | (c) | (d) R1B, R1P: D; minus legacy | (e) time ratio; slowest call | holds | candidate |
|---|---|---|---|---|---|---|---|
| aggr | +0.0015, +0.0005 | fails both: PL +0.0016, +0.0015; matharena +0.0103, +0.0142 | fails FLAT +0.0033 | +0.0025, +0.0020; +0.0022, -0.0019 | 0.98; 0.64 s | e, f | no |
| eb_fit | +0.0173, +0.0233 | fails both: PL +0.0156, +0.0175; multi_swebench +0.0488, +0.0595 | fails TUNED +0.0309, MIXTURE +0.0164, FLAT +0.0140 | -0.0010, +0.0009; -0.0013, -0.0030 | 0.97; 0.45 s | d, e, f | no |
| eb_adapt | +0.0003, -0.0005 (U95 +0.0012, +0.0008) | fails both: PL +0.0005, +0.0002; matharena +0.0041 in AUDIT | holds | +0.0017, +0.0020; +0.0014, -0.0019 | 1.19; 0.59 s | c, e, f | no |
| eb_ship | +0.0003, -0.0002 (U95 +0.0010, +0.0006) | fails both: PL +0.0005, +0.0002 | holds | +0.0006, +0.0014; +0.0004, -0.0026 | 1.18; 0.70 s | c, e, f | no |
| wide35 | +0.0006, -0.0003 (U95 +0.0012, +0.0004) | fails READING: PL +0.0007 | holds | +0.0007, +0.0008; +0.0005, -0.0031 | 0.95; 0.56 s | c, d, e, f | no |
| wide50 | +0.0023, +0.0006 | fails both: PL +0.0027, +0.0012; real_webagents +0.0047, researchcodebench +0.0053 | fails TUNED +0.0031, MIXTURE +0.0022 | +0.0026, +0.0025; +0.0023, -0.0014 | 0.95; 0.42 s | e, f | no |
| onepl | +0.0026, +0.0016 | fails both: PL +0.0028, +0.0028; matharena +0.0131, +0.0176 | fails MIXTURE +0.0031, FLAT +0.0049 | +0.0025, +0.0027; +0.0023, -0.0013 | 0.93; 0.45 s | e, f | no |
| smcal | +0.0078, +0.0043 | fails both: PL +0.0083, +0.0062; matharena +0.0238, +0.0274 | fails MIXTURE +0.0071, FLAT +0.0103 | +0.0108, +0.0100; +0.0106, +0.0060 | 0.01; 0.06 s | e, f | no |
| smooth | +0.0096, +0.0142 | fails both: PL +0.0091, +0.0118 | fails all three | +0.0075, +0.0127; +0.0073, +0.0088 | 0.01; 0.08 s | e, f | no |
| legacy | +0.0248, +0.0313 | fails both: PL +0.0239, +0.0265 | fails all three | +0.0002, +0.0039; 0, 0 | 0.82; 1.86 s | e, f | no |

No config meets (a) in either regime. wide35 comes nearest: it meets (c) to
(f) and misses (a) by D +0.0006 and -0.0003, against the -0.002 required.

### Ranks (Q3, descriptive)

Configs by ALC in each regime. A config is counted as tied with SHIP when its
paired difference from SHIP is under 1.96 cluster SEs (`summary.ranks`).

| regime | order, best first | SHIP's place; tied with SHIP |
|---|---|---|
| TUNED | aggr, onepl, eb_adapt, smcal, eb_ship, SHIP, wide35, wide50, smooth, eb_fit, legacy | 6th; smcal |
| READING | SHIP, eb_adapt, eb_ship, wide35, aggr, wide50, onepl, smcal, smooth, eb_fit, legacy | 1st; eb_adapt, eb_ship, wide35, aggr, onepl |
| AUDIT | eb_adapt, wide35, eb_ship, SHIP, aggr, wide50, onepl, smcal, smooth, eb_fit, legacy | 4th; every config except smooth, eb_fit and legacy |
| MIXTURE | eb_adapt, eb_ship, SHIP, wide35, aggr, wide50, onepl, smcal, smooth, eb_fit, legacy | 3rd; eb_adapt, eb_ship, wide35, aggr |
| FLAT | wide35, wide50, eb_ship, SHIP, eb_adapt, aggr, onepl, smooth, smcal, eb_fit, legacy | 4th; wide50, eb_ship, eb_adapt |
| R1B | eb_fit, SHIP, legacy, eb_ship, wide35, eb_adapt, aggr, onepl, wide50, smooth, smcal | 2nd; eb_fit, legacy |
| R1P | SHIP, wide35, eb_fit, eb_ship, aggr, eb_adapt, wide50, onepl, legacy, smcal, smooth | 1st; eb_fit |

Two regimes have configs ahead of SHIP by more than 1.96 cluster SEs:

* TUNED: aggr, onepl, eb_adapt and eb_ship.
* FLAT: wide35, at -0.0010 (cluster SE 0.0005).

In every other regime, SHIP is first or tied with the first.

### What a calibrated level alone recovers (W4, descriptive)

**In TUNED, the calibrated smoothed mean matches SHIP.**

* SHIP minus smcal is +0.0015 ± 0.0010 / 0.0019 / 0.0015, with a parent-level
  mean of -0.0010.
* smcal minus the legacy Predictor is -0.0448 ± 0.0028 / 0.0052 / 0.0045,
  against SHIP's -0.0434.

smcal is a smoothed mean with a prior mean of 0.25 and a strength of two
labels. It has no item model and no subject prior, yet in that regime it
takes the whole of the gain over the legacy Predictor. Its prior was chosen
on the same catalogue, so this is its best case, and it fails the public
guard.

**Away from the level it was tuned for, smcal falls behind.** SHIP beats it
by 0.0043 (AUDIT) to 0.0103 (FLAT) in the other test-like regimes, and by
0.0108 and 0.0100 on public runs.

**By budget** (`summary.vs_ship.<regime>.smcal.table.budgets`), smcal predicts
0.25 at B0 everywhere:

* In TUNED it is ahead of SHIP at B0 and B1 (-0.0138 and -0.0079).
* In every regime it is behind from B7 on. At B31 it is behind by +0.0069 in
  TUNED and +0.0089 in READING.

In READING, smcal is barely better than plain Beta(2,2) (-0.0018 ±
0.0017 / 0.0036 / 0.0031), and in FLAT it is level with it (+0.0000)
(`summary.vs_smooth`).

So the tuned regime's headline does not show what hier adds: a moved level
alone produces it. What hier adds over a calibrated level is what it keeps
when the hidden level is not the one it was tuned for, 0.004 to 0.010 in
these regimes. It also stays level with the legacy Predictor on public runs,
where smcal loses 0.006 to 0.011.

**The subject prior (onepl).** In every regime but TUNED, SHIP beats onepl,
by 0.0016 to 0.0049 on test-like runs and by 0.0025 and 0.0027 on public runs
(cluster SEs 0.0005 and 0.0004). In TUNED, onepl is ahead by 0.0030 ±
0.0007 / 0.0012 / 0.0010 (`summary.decomposition`).

* **onepl's TUNED gain is at B0 and B1** (-0.0132 and -0.0055), where its
  lower centre helps on low pairs.
* **At B0 it loses everywhere else,** by +0.0023 to +0.0207.

Because onepl also moves the centre, this does not isolate the subject prior.
It bounds what the attribute prior at attr_scale 0.5 and SHIP's centre are
worth together outside the tuned regime: 0.002 to 0.005.

### The empirical-Bayes level (descriptive)

**eb_ship** differs from SHIP only by re-estimating the level from `labeled`
at each checkpoint.

* In TUNED it is ahead: -0.0014 ± 0.0003 / 0.0004 / 0.0004.
* In READING, AUDIT, MIXTURE and FLAT it is level: -0.0002 to +0.0003.
* On public runs it is behind: +0.0006 and +0.0014.

**eb_adapt**, on a lower base (mu0 -3.0), is ahead in TUNED by 0.0027, level
in the feedback regimes, and costs +0.0017 and +0.0020 on public runs.

**Where eb_ship's centre moves.** The centre is averaged over each run's
per-parent models (`summary.eb_traces`). It starts at -2.5 at B0. By B31 it
has moved:

* down, to -3.7 in TUNED and -3.2 in AUDIT;
* slightly down, to -2.8 in READING and MIXTURE;
* up, to -2.3 in FLAT and to -1.4 and -1.7 on public runs (benchmark-first,
  pair-uniform).

It orders the regimes as before ("The empirical-Bayes level adapts the right
way"). Where the levels spread both ways, though, it gains nothing over the
fixed level. The review expected EB to beat any fixed shift there, and it
does not.

So the decision not to ship EB now rests on this measurement, not only on its
code size and slowest call.

**eb_fit**, the fitted level with no LEVEL override, loses 0.014 to 0.031 in
the test-like regimes and is level on public runs (-0.0010 and +0.0009), as in
"Calibrating for the hidden test".

### P1.16: a wider level prior

**wide35 stays within 0.001 of SHIP everywhere.** It differs by -0.0010 to
+0.0009 in the test-like regimes and by +0.0007 and +0.0008 on public runs.
With cluster SEs of 0.0003 to 0.0005 some of those differences are real, but
all are small. It gains
only in FLAT and AUDIT (-0.0010 and -0.0003), whose realised sds are 2.23 and
2.05; MIXTURE, at 2.13, is not among them.

**wide50 is worse in five of the seven regimes.** It costs 0.0022 to 0.0031
in TUNED, READING, MIXTURE and on both public weightings. It is level in
AUDIT and FLAT (+0.0006 and -0.0007).

**By budget**, both trade B0 against B1 on public runs. wide35 gains at B0
(-0.0032 and -0.0020) and loses from B1 on (+0.0026 and +0.0029 at B1).

Both fail (a). At the shipped mu0 and attr_scale, the grid's edge at sigma_mu
2.5 does not bind. The answer to P1.16 is no, and no other width is chosen
from these results.

### Latency

Pooled over every task (`summary.latency.all`, `rule.table.<config>.conditions.e`):

| config | mean time per call | ratio to SHIP | slowest single call |
|---|---|---|---|
| SHIP | 1.34 ms | 1 | 0.56 s |
| other hier configs | 1.25 to 1.59 ms | 0.93 to 1.19 (the two EB configs slowest) | 0.42 to 0.70 s |
| legacy Predictor | 1.10 ms | | 1.86 s |
| smoothed means | 0.01 ms | | |

Every config passes (e).

No fallback fired: the failure and unconverged counts are 0 for every config
in every regime (`summary.configs`).

The times were taken with two processes on a machine shared with other jobs.
The ratios compare configs timed in the same tasks.

### Provenance and deviations

**D1, the legacy reproduction check.**

* **What happened.** The check was planned at a tolerance of 1e-9 against
  `testlike_check.json`, which stores each Brier rounded to six decimals. The
  stored differences were 4.7e-7 to 5.0e-7, all within half a unit of the
  sixth decimal (`review.provenance.d1`).
* **The fix.** The check now compares at the stored precision. It passed at
  16:26:39, before the first scoring-seed row (16:44:02), and no row depends
  on it.
* **What it can change.** D1 touches only (f).
  * Read literally, (f) fails, and the outcome is "rule not applied: check
    failed".
  * With D1, the outcome is "no candidate: SHIP stays".
  * Neither reading has a candidate.
* **Decided.** The team accepted D1 on 2026-10-02. Future plans should set
  tolerances at the comparator's stored precision.

**The lock** rests on local mtimes and scratch copies (above), not on a
commit made before scoring.

**Script versions.**

* smcal and regimes ran under script digest dbbf63b6, reproduce under
  316f8e91, and all 520 rows and the summary under 0bef6bf0
  (`review.provenance.stages`).
* Re-run under the final script, the three earlier stages give identical
  numbers (366, 205 and 150 numbers compared, largest difference 0.0) and the
  same composition digest (`review.rerun`).
* Every edit made after scoring sits in a block marked "after review".
  Cutting those blocks out gives back the scored script byte for byte
  (`review.provenance.script`, checked by a test). The blocks add the
  `review` stage, and let the lock accept the amendments appended to the
  plan while still pinning the fixed text.

**A dry run on a partial set.** At 16:56:16, 19 rows into scoring, a dry run
of summarise ran into a scratch copy of the results. It printed the
structure, the checks and the outcome "rule not applied: the planned set is
not complete". Every row, the dry run and the summary carry one script
digest, so no code, constant or run count changed after it
(`review.provenance.dry_run`).

**Re-scoring.** The last planned run of each regime was re-scored with every
config, and each matches its row exactly (largest Brier difference 0.0). The
EB traces inside a row are ordered by iterating a set, an order that depends
on PYTHONHASHSEED: 0 of 14 match as stored, and 14 of 14 once sorted. The
summary uses only their aggregates (`review.rescore`).

**smcal's guard reference** predates the corrected floor (above;
`smcal.guard_reference_note_after_review`).

**pytest.** A first full run, with default BLAS threads and while the
reproduce stage ran, showed one failure. It sat at the position of
`tests/test_hier.py::test_formative_size_is_fast`, a 5 s wall-clock bound
(`review.provenance.first_pytest`). Single-threaded full runs pass.

### What this means for the shipped configuration

* **It stays, and nothing scored here is a better bet.** In the two regimes
  set to the feedback's readings, no config beats SHIP by the 0.002 the rule
  required. The three closest, wide35, eb_adapt and eb_ship, sit within
  0.0006 of it, with cluster SEs of 0.0003 to 0.0007. SHIP is first, or tied
  with the first, in READING, AUDIT, MIXTURE and on both public weightings.
* **The audit's move is supported where its reasoning applies.**
  * aggr, the config the calibration recommended, beats SHIP only in TUNED
    (-0.0034 ± 0.0006 / 0.0010 / 0.0008). That is the regime and catalogue
    it was selected on.
  * It loses in the four other test-like regimes (+0.0005 to +0.0033) and on
    public runs (+0.0025 and +0.0020), and its losses sit on matharena, the
    high-level parent.
  * Across the test-like regimes its difference rises with the realised
    level, from -0.0034 at -1.28 to +0.0033 at -0.28.
* **P1.16: no.** sigma_mu 3.5 stays within 0.001 of 2.5, and 5.0 is worse in
  five of the seven regimes.
* **EB: no.** It gains only in the tuned regime and costs on public runs.
* **The gain over the legacy Predictor depends on where the level sits.** It
  is 0.043 in TUNED, 0.020 to 0.031 at the other test-like regimes' realised
  levels (-0.28 to -0.93), and level on public runs.
* **In TUNED, a calibrated smoothed mean takes the same gain.** SHIP's margin
  over it, 0.004 to 0.011, appears only away from the level it was tuned for.
  The case for hier over a moved level is robustness to where the hidden
  level sits, not the tuned-regime headline.

### What this cannot settle

* **Four parents carry every test-like number.** The ranking turns on
  matharena against multi_swebench, and the parent-level SEs are two to four
  times the cluster SEs.
* **Fresh seeds, same data** (above). TUNED and the guards do not replicate
  the original selection independently.
* **The date shift is in every test-like regime.** attr_scale's role
  therefore stays conditional on it (review Q10). The regime without the
  shift was not scored here.
* **The feedback regimes realised milder levels than targeted,** -0.63 and
  -0.93. level_mean -2.0 was not scored for SHIP, and MIXTURE's low mode
  rests on two pseudo-benchmarks.
* **The rule's power is limited.** It detects gains of about 0.003 or more
  and is a coin flip at 0.002 (plan, section 7).
* **Two baselines are missing.** onepl is not a plain 1PL, and BLE was not
  run. (Since then the plain 1PL has been scored on these runs, and BLE was
  found unable to run offline: "Baselines: a plain 1PL, the organisers'
  empirical mean and BLE".)
* **No formative run was read.** Run 3 is not used. Runs 1 and 2 enter only
  through the stored readings that set the targets of READING, AUDIT and
  MIXTURE ("Formative feedback, runs 1 and 2", "What actually shipped, after
  the audit").

## Baselines: a plain 1PL, the organisers' empirical mean and BLE (P1.9)

`python experiments/baselines_p1.py ble`, then `score` (repeated while it exits
with 75) and `summarise`. One process, at most 0.67 GB resident; the final
pass took 21 minutes on a machine shared with other jobs. Every number below
is in `results/baselines_p1.json`, under the key in brackets. The rows are in
`data/baselines_p1_rows/` (gitignored, 200 files).
`tests/test_baselines_p1.py` covers the script.

**What it asks.** The internal review (`docs/report/review_v0.md`, W4 and
P1.9) asked for the baselines that separate the level calibration from the
modelling: a level-calibrated smoothed mean (scored by P1a, "Regime
sensitivity at the feedback's reading"), a plain 1PL with a pooled level and
no attributes, and the organisers' reference predictors, BLE and the
empirical mean with BLE acquisition. It also asked to restate "calibration is
worth far more than the modelling" against them.

### Runs, configs and checks

The runs are P1a's, seed 11: TUNED runs 0 to 79, and public R1 runs 0 to 59
under both weightings (R1B benchmark-first, R1P pair-uniform), split scope
'pair' (`plan.regimes`). Six configs are read from P1a's stored rows: SHIP,
`legacy`, `smooth` (Beta(2,2)), `smcal` (n0 2, m0 0.25), `onepl` and
`eb_fit`. Three are new, scored on the same checkpoints (`plan.new_configs`):

| key | what it is |
|---|---|
| `rasch` | the plain 1PL at the shipped level, mu0 -2.5, sigma_mu 2.5 |
| `rasch_fit` | the same 1PL at its own leave-one-parent-out level (below) |
| `empmean` | the organisers' empirical mean, as in "Under the official protocol" |

**The plain 1PL** is hier with the attributes, the identity link, the group
effects, the multiple-choice floor and slip switched off. What remains is
logit p = mu_b + a_sb - d_i: one pooled level per benchmark, one ability per
(subject, benchmark) pair with prior variance sigma_theta² + sigma_attr² +
sigma_delta², and one difficulty per item with variance sigma_d² + sigma_g²,
integrated out. Inference is hier's, refitted from `labeled` at every
checkpoint. It differs from `onepl` in four ways: one ability per pair (onepl
keys ability on the canonical name), no group effects, no floor and no slip.
Tests check that at B0 it equals the closed-form 1PL marginal, and that labels
reach neither another benchmark nor another group.

**rasch_fit's level** is `fit_hyper`'s derivation without the attribute
shift: each training benchmark's fitted level plus its mean standing, then
their mean, and their sd widened as for hier (`summary.rasch_fit_levels`):

| held-out parent | mu0 | sigma_mu |
|---|---|---|
| matharena | -1.78 | 2.72 |
| multi_swebench | -0.58 | 1.67 |
| real_webagents | -1.36 | 3.39 |
| researchcodebench | -1.18 | 3.37 |
| swe_rebench | -1.56 | 3.21 |

**Checks** (`checks`; `summary.regimes.<R>.reproduction_max_abs_brier` and
`.p1a_agreement_max_abs`):

* Before writing each new row, the script re-scores smooth and smcal on every
  run, and SHIP, onepl and legacy on the first and last run of each regime.
  All match P1a's stored Brier exactly (largest difference 0.0). The run
  compositions match, and the library digests equal those of P1a's rows.
* P1a's differences against SHIP, recomputed from the merged rows, agree with
  P1a's to 0.0.
* A first full pass is bit-identical to the final pass. The final pass was
  rerun with the final script, so every row carries one script digest
  (`provenance.script`, 9825bb14142d4b63).

### BLE cannot run here

The `ble` stage reads `third_party/paiec_baseline` at commit 82d330dd with
`ast` and text reads only. It never imports or copies that code, and all seven
of its checks hold (`ble`).

* **Every BLE prediction is a language-model agent run.** By default it calls
  openai/gpt-5.6-luna over the Responses API, for up to 10 turns (5 in the
  example config), with a 240 s deadline a prediction.
* **It needs credentials, network and a prepared payload.** It needs
  `OPENAI_API_KEY`, network access, and a payload prepared over the network
  from measurement-db and measurement-db-embed. To stay leave-one-parent-out,
  that payload would have to be filtered per held-out parent.
* **There is no offline predictor.** The only offline mode,
  `smoke_test.py --mock`, replaces the model with scripted replies on
  synthetic tables. It tests plumbing, not prediction.
* **The empirical mean with BLE acquisition cannot run either.** It predicts
  the empirical mean but chooses its labels from BLE's predictions. Under
  random acquisition it is `empmean`, which is scored.
* **The cost.** These runs alone hold 569,838 evaluation predictions (213,492
  TUNED, 178,668 R1B, 177,678 R1P), before any acquisition call
  (`ble.evaluation_predictions`).

BLE and the empirical mean with BLE acquisition remain unmeasured. The best
leaderboard entry (0.1172, "Against the live leaderboard") stays unexplained.

### On identical runs

ALC (`summary.regimes.<R>.configs`):

| config | TUNED (80 runs) | R1B (60) | R1P (60) |
|---|---|---|---|
| SHIP | 0.1690 | 0.2114 | 0.1948 |
| onepl | 0.1660 | 0.2140 | 0.1975 |
| rasch | 0.1673 | 0.2162 | 0.1998 |
| smcal | 0.1675 | 0.2222 | 0.2048 |
| rasch_fit | 0.1743 | 0.2173 | 0.2027 |
| smooth | 0.1860 | 0.2189 | 0.2075 |
| eb_fit | 0.1999 | 0.2104 | 0.1957 |
| empmean | 0.2029 | 0.2578 | 0.2397 |
| legacy | 0.2124 | 0.2116 | 0.1988 |

The new configs minus SHIP, ± run / cluster / stratified SE, and for rasch the
parent-level mean ± SE (`summary.regimes.<R>.vs_ship`, `.headline`):

| config | TUNED | R1B | R1P |
|---|---|---|---|
| rasch | -0.0017 ± 0.0007 / 0.0013 / 0.0010; parent-level -0.0002 ± 0.0041 | +0.0048 ± 0.0005 / 0.0007 / 0.0006; +0.0053 ± 0.0023 | +0.0050 ± 0.0004 / 0.0007 / 0.0006; +0.0049 ± 0.0020 |
| rasch_fit | +0.0053 ± 0.0003 / 0.0006 / 0.0006 | +0.0059 ± 0.0007 / 0.0011 / 0.0009 | +0.0078 ± 0.0005 / 0.0009 / 0.0009 |
| empmean | +0.0339 ± 0.0016 / 0.0030 / 0.0028 | +0.0464 ± 0.0019 / 0.0027 / 0.0026 | +0.0449 ± 0.0020 / 0.0030 / 0.0029 |

* **The plain 1PL at the shipped level is level with SHIP in TUNED and behind
  it on public runs.** In TUNED it is 0.0017 ahead (1.4 cluster SEs; the
  parent-level mean is -0.0002 ± 0.0041). On public runs it trails by 0.0048
  and 0.0050, 6.5 to 7.5 cluster SEs.
* **At its own fitted level the 1PL trails SHIP everywhere,** by 0.005 in
  TUNED and by 0.006 and 0.008 on public runs.
* **The organisers' empirical mean trails SHIP by 0.034 in TUNED and by 0.045
  to 0.046 on public runs.** In TUNED it is ahead of the legacy Predictor by
  0.0095: -0.0095 ± 0.0025 / 0.0049 / 0.0043, 1.9 cluster SEs, with a
  parent-level mean of -0.0029 ± 0.0120 (`summary.regimes.TUNED.vs_legacy.empmean`). The legacy
  Predictor's mean B0 prediction there is 0.68, against SHIP's 0.42
  (`summary.regimes.TUNED.configs.<config>.q0`): its attribute prior turns
  the synthetic date shift into optimism, which costs it more than the
  empirical mean loses by jumping to 0 or 1 at B1.
* **Cost.** rasch takes 0.73 to 0.81 ms a call (slowest 0.13 to 0.18 s) and
  rasch_fit 0.72 to 0.81 ms (slowest 0.15 s), on one process of a shared
  machine (`summary.regimes.<R>.configs.rasch|rasch_fit`, `mean_call_ms` and
  `max_call_s`). SHIP's 1.2 to 1.5 ms a call in the same file is read from
  P1a's rows, so it was timed in P1a's tasks, not in these. Read across the
  two, SHIP takes about 1.7 to 1.9 times as long (derived, no SE).

### How much of SHIP's gain is the level calibration

In TUNED, SHIP minus legacy is -0.0434 ± 0.0019 / 0.0037 / 0.0034, with a
parent-level mean of -0.0393 ± 0.0076. The share that is "level calibration"
depends on the order in which the steps are taken
(`summary.regimes.TUNED.decomposition`; each step ± run / cluster / stratified
SE, and its share of the total with a 95% bootstrap interval):

| path | step | difference | share |
|---|---|---|---|
| calibration first | calibrated level (smcal - legacy) | -0.0448 ± 0.0028 / 0.0052 / 0.0045 | 1.03 [0.94, 1.10] |
| | the model at a calibrated level (SHIP - smcal) | +0.0015 ± 0.0010 / 0.0019 / 0.0015 | -0.03 [-0.10, 0.06] |
| model first | the model at its fitted level (eb_fit - legacy) | -0.0124 ± 0.0008 / 0.0024 / 0.0022 | 0.29 [0.19, 0.39] |
| | calibration within hier (SHIP - eb_fit) | -0.0309 ± 0.0018 / 0.0035 / 0.0026 | 0.71 [0.61, 0.81] |
| the 1PL path | the 1PL at its fitted level (rasch_fit - legacy) | -0.0380 ± 0.0019 / 0.0038 / 0.0035 | 0.88 [0.84, 0.91] |
| | calibration within the 1PL (rasch - rasch_fit) | -0.0070 ± 0.0007 / 0.0013 / 0.0011 | 0.16 [0.12, 0.20] |
| | hier's extras at LEVEL (SHIP - rasch) | +0.0017 ± 0.0007 / 0.0013 / 0.0010 | -0.04 [-0.08, 0.02] |

The chain between the calibrated predictors (`.decomposition.chain`): rasch -
smcal -0.0003 ± 0.0004 / 0.0009 / 0.0008; onepl - rasch -0.0013 ± 0.0001 /
0.0002 / 0.0002; SHIP - onepl +0.0030 ± 0.0007 / 0.0012 / 0.0010.

* **No order-free share exists.** Calibrating first gives the calibration the
  whole gain, modelling first gives it 0.71, and on the 1PL path a 1PL at its
  own public level already takes 0.88 before any calibration.
* **What holds in every order: at a calibrated level, in TUNED, hier adds
  nothing.** SHIP - smcal is +0.0015 (cluster SE 0.0019) and SHIP - rasch
  +0.0017 (0.0013). smcal, rasch, onepl and SHIP lie within 0.003 of one
  another.
* **Most of the legacy Predictor's TUNED deficit is its own prior.** A plain
  1PL at its fitted public level, with no calibration at all, beats the legacy
  Predictor by 0.0380 (cluster SE 0.0038). hier at its fitted level beats it
  by only 0.0124. The legacy Predictor's B0 centre (0.68) is the date shift's
  attribute optimism. This sharpens review W1: the regime's headline is
  largely set by the date shift.

**On public runs the total is within noise,** so no share is given: SHIP
minus legacy is -0.0002 ± 0.0010 / 0.0020 / 0.0017 on R1B and -0.0039 ±
0.0010 / 0.0019 / 0.0017 on R1P (z -2.1 on the cluster SE). The steps offset
each other, because there the calibration costs and the model gains:

| step | R1B | R1P |
|---|---|---|
| smcal - legacy | +0.0106 ± 0.0015 / 0.0028 / 0.0025 | +0.0060 ± 0.0015 / 0.0031 / 0.0027 |
| SHIP - smcal | -0.0108 ± 0.0010 / 0.0014 / 0.0012 | -0.0100 ± 0.0010 / 0.0018 / 0.0016 |
| eb_fit - legacy | -0.0013 ± 0.0006 / 0.0012 / 0.0009 | -0.0030 ± 0.0008 / 0.0010 / 0.0009 |
| SHIP - eb_fit | +0.0010 ± 0.0006 / 0.0010 / 0.0010 | -0.0009 ± 0.0005 / 0.0011 / 0.0010 |
| rasch_fit - legacy | +0.0057 | +0.0039 |
| rasch - rasch_fit | -0.0011 | -0.0029 |
| rasch - smcal | -0.0060 | -0.0050 |
| onepl - rasch | -0.0023 | -0.0023 |
| SHIP - onepl | -0.0025 ± 0.0003 / 0.0005 / 0.0005 | -0.0027 ± 0.0002 / 0.0004 / 0.0004 |

So on public runs, at the same level, hier's modelling adds 0.010 over a
calibrated smoothed mean and 0.005 over a plain 1PL: about half from the
groups, the floor, slip and ability keyed on the model's name (onepl - rasch),
half from the subject prior (SHIP - onepl).

### At budgets 0 and 1

Brier differences, cluster SE in brackets (`summary.regimes.<R>.budgets_0_1`).
In TUNED rasch's B0 centre is lower than SHIP's (mean B0 prediction 0.31
against 0.42), so SHIP - rasch there mixes the attribute prior with the centre:
+0.0114 (0.0050) at B0 and +0.0049 (0.0024) at B1, in rasch's favour. Where the
mean B0 predictions match, SHIP beats the 1PL at B0 and B1:

| comparison | mean B0 prediction | B0 | B1 |
|---|---|---|---|
| TUNED, SHIP - rasch_fit | 0.42 and 0.42 | -0.0151 (0.0026) | -0.0093 (0.0014) |
| R1B, SHIP - rasch | 0.30 and 0.31 | -0.0097 (0.0021) | -0.0046 (0.0011) |
| R1P, SHIP - rasch | 0.31 and 0.31 | -0.0113 (0.0020) | -0.0044 (0.0009) |

About 0.0013 to 0.0017 of these B0 gaps separates onepl from rasch at B0 (the
groups, the floor, slip, and ability keyed on the model's name); the rest is
mostly the attribute prior. This is the
attribute prior's share at B0 and B1 that W4 asked for, with the centre held
level.

### What this changes

* **"Calibration is worth far more than the modelling" cannot be stated as a
  share.** The split depends on the order of the steps. What can be stated:
  in the tuned regime every predictor whose level is set for that regime
  (smcal, rasch, onepl, SHIP) lands within 0.003 of the others, so the model
  adds nothing measurable there at a calibrated level. On public runs, where
  the calibration costs, hier's modelling adds 0.010 over a calibrated
  smoothed mean and 0.005 over a plain 1PL at the same level.
* **Nothing ships differently.** The 1PL was not scored under P1a's rule, and
  it trails SHIP on public runs by 0.005.

### Caveats

* **rasch borrows hier's calibrated mu0** (-2.5), which was chosen with the
  attribute prior on. No 1PL-specific level calibration was run. On public
  runs rasch_fit costs +0.0039 to +0.0057 against the legacy Predictor.
* **TUNED favours the configs calibrated to it.** Its seed-11 runs redraw the
  catalogue that chose LEVEL and smcal (P1a's caveat), which favours SHIP,
  smcal and rasch there.
* **BLE is unmeasured** (above).
* **The rows are in the row export** ("Row files for release"), not yet hosted.

## Pooling under the verified protocol: run size, not the information set (P1.11)

`python experiments/pooling_decomposition.py score`, `recheck`, then
`summarise`. 208 tasks in 5,065 s (about 84 minutes) on one process, under 1 GB
resident (`provenance`). Every number below is in
`results/pooling_decomposition.json`, under the key in brackets. The rows are
in `data/pooling_decomposition_rows/` (gitignored).
`tests/test_pooling_decomposition.py` covers the script.

**What it asks.** The report's first contribution said that pooled item
difficulty, worth about 0.020 ALC on the legacy replica ("The predictor
ladder"), was worth about 0.0007 at formative size under the verified protocol
("Under the official protocol"), and put the drop down to the legacy replica's
wrong information set. The review (W5, P1.11) noted that the two numbers also
differ in run size and split scope. It asked for pooling on and off, under
both split scopes, on dense and on formative-size runs of the official replica.

**Variants** (`design.variants`):

| key | what it is |
|---|---|
| `ship` | hier as shipped (LEVEL); every prior fitted with the target's parent left out |
| `ship_nolevel` | the same with a level per pair (`pool_mu` off), to separate the pooled level's share from the items' |
| `ship_own` | the same with pooling off: `labeled` restricted to the target pair's own entries (same subject key and benchmark_id) |
| `legacy`, `legacy_nopool` | the legacy Predictor with its pooled difficulty on and off (WARMUP 1e9), the switch behind the 0.0007 |
| `smooth` | the smoothed mean, Beta(2,2) |

hier has no switch for item pooling alone. `ship_own` is therefore a
research-only wrapper in the script (`OwnOnly`), and the library and the
shipped defaults are unchanged. It also drops the subject's labels on other
benchmarks, but on pairs alone on their benchmark that difference is 0.000000
± 0.000006, so it does not matter.

**Runs** (`design`):

* **Formative size.** 100 runs of `official.sample_run` (seed 11,
  benchmark-first), each scored under both split scopes, with the same seeds
  for on and off. Runs 0 to 59 reproduce P1a's stored runs bit for bit. They
  hold 880 pair appearances, 8.8 a run over 4.34 benchmarks: 193 alone on
  their benchmark, 220 with one companion and 467 with two or more
  (`formative.pair.describe`).
* **The platform's composition.** In formative runs 1 to 3, 16 of 26 pair
  appearances were alone on their benchmark, 10 had one companion and none had
  more (`platform_mix`). The "platform mix" rows reweight the formative
  differences to those shares.
* **Dense.** `official.dense_run` of the four multi-subject benchmarks, under
  both scopes. On researchcodebench, matharena and multi_swebench, each pair's
  Brier is scored on at most 32 evaluation items, a fixed digest-ranked subset
  (`design.eval_cap`). The `labeled` list is the full dense run's, so the
  predictions are unchanged; fewer targets are scored. real_webagents is
  scored whole.
* **Not run: swe_rebench.** It has one subject, so pooling on and off coincide.

### Pooling on minus off

ALC differences, negative where pooling helps (`reading`). Formative rows give
± run / cluster / stratified SE and then the parent-level mean ± SE. Platform
mix rows give ± cluster / stratified SE. Dense rows give the mean ± SE across
the four benchmarks.

| what is pooled | formative, scope 'pair' | formative, 'benchmark' | platform mix, 'pair'; 'benchmark' | dense, 'pair' | dense, 'benchmark' |
|---|---|---|---|---|---|
| hier, everything taken from other subjects (ship - ship_own) | -0.0033 ± 0.0004 / 0.0006 / 0.0005; -0.0037 ± 0.0022 | -0.0020 ± 0.0004 / 0.0005 / 0.0004; -0.0022 ± 0.0013 | -0.0010 ± 0.0003 / 0.0003; -0.0004 ± 0.0003 / 0.0002 | -0.0300 ± 0.0099 | -0.0142 ± 0.0059 |
| hier, the pooled level (ship - ship_nolevel) | -0.0015 ± 0.0003 / 0.0004 / 0.0004 | -0.0012 ± 0.0003 / 0.0003 / 0.0003 | -0.0007 ± 0.0003 / 0.0002; -0.0003 ± 0.0002 / 0.0002 | -0.0043 ± 0.0014 | -0.0029 ± 0.0009 |
| hier, pooled item difficulty (ship_nolevel - ship_own) | -0.0018 ± 0.0002 / 0.0003 / 0.0002 | -0.0008 ± 0.0002 / 0.0002 / 0.0002 | -0.0004 ± 0.0001 / 0.0001; -0.0002 ± 0.0001 / 0.0001 | -0.0258 ± 0.0088 | -0.0114 ± 0.0055 |
| legacy Predictor, pooled difficulty (legacy - legacy_nopool) | -0.0010 ± 0.0001 / 0.0002 / 0.0001; -0.0011 ± 0.0005 | -0.0004 ± 0.0001 / 0.0001 / 0.0001 | exactly 0; exactly 0 | -0.0237 ± 0.0065 | -0.0086 ± 0.0038 |

hier's total by how many pairs of its benchmark the run holds (formative,
scope 'pair'): alone 0.0000 (193 appearances; also 0.0000 on the 106 left
without swe_rebench, which is alone in every run), two -0.0027 ± 0.0007
(220), three or more -0.0053 ± 0.0009 (467).

By budget, hier's total (Brier):

| budget | formative, 'pair' | dense, 'pair' (mean of 4) | dense, 'benchmark' (mean of 4) |
|---|---|---|---|
| B0 | 0 | 0 | 0 |
| B1 | -0.0034 | -0.020 | -0.019 |
| B3 | -0.0037 | -0.025 | -0.016 |
| B7 | -0.0032 | -0.034 | -0.017 |
| B15 | -0.0034 | -0.044 | -0.014 |
| B31 | -0.0059 | -0.055 | -0.011 |

At formative size the level's share sits at B1 to B7 and the items' at B7 to
B31. The legacy switch acts only at B15 (-0.0012) and B31 (-0.0077), where a
benchmark can reach 64 distinct labeled items.

Ratios, descriptive and without SE: formative parent-level over dense, hier
0.12 ('pair') and 0.15 ('benchmark'), legacy 0.047 and 0.057; dense
'benchmark' over 'pair', hier 0.47 and legacy 0.36.

### What this says about the first replica

* **Pooling across subjects is still a large lever on dense runs.** Under the
  verified protocol it is worth 0.009 to 0.030 ALC there, depending on split
  scope and predictor. That brackets the legacy ladder's 0.020. Most of hier's
  dense value is pooled item difficulty (0.026 of 0.030 under 'pair').
* **At formative size the same switches are worth 0.002 to 0.003 for hier and
  0.0004 to 0.001 for the legacy Predictor.** At the platform's composition
  they are worth 0.0004 to 0.001 for hier and exactly 0 for the legacy
  Predictor, whose pooled difficulty needs 64 distinct labeled items on a
  benchmark, which two pairs never reach.
* **The value grows with company,** from nothing for a pair alone on its
  benchmark to 0.005 with two or more companions.
* **Per-benchmark splits halve what is left on dense runs** (ratios 0.47 and
  0.36).

So run size, not the information set, removed the lever, and per-benchmark
splits halve it again. The 0.0007 of "Under the official protocol" was
measured under split scope 'pair' only (600 runs, seed 0). Its like-for-like
value here is the 'pair' scope's 0.0010 ± 0.0002 (cluster SE; 100 runs,
seed 11): same scope, other seeds and run counts.

### Dense runs per benchmark, and hier's cost there

| benchmark (items scored) | pairs | hier, everything pooled: 'pair' / 'benchmark' | SHIP - legacy: 'pair' | 'benchmark' | SHIP's mean call; slowest |
|---|---|---|---|---|---|
| real_webagents (all) | 26 | -0.0126 / -0.0083 | -0.0035 ± 0.0031 | -0.0081 ± 0.0043 | 2.0 to 2.1 ms; 0.08 s |
| researchcodebench (at most 32) | 31 | -0.0415 / -0.0130 | -0.0054 ± 0.0024 | +0.0068 ± 0.0020 | 3.6 to 3.8 ms; 0.16 s |
| matharena (at most 32) | 81 | -0.0521 / -0.0312 | -0.0006 ± 0.0021 | -0.0007 ± 0.0026 | 7.5 to 9.4 ms; 0.39 s |
| multi_swebench (at most 32) | 82 | -0.0139 / -0.0045 | -0.0207 ± 0.0029 | -0.0169 ± 0.0037 | 3.1 to 4.6 ms; 0.21 s |

(`dense.<benchmark>.<scope>.comparisons` and `.variants.ship`; SEs are over
the benchmark's pairs, conditional on one shared `labeled` list and one
split.) These are the first dense matharena and multi_swebench runs of the
shipped hier, and so the first measurement of its worst-case call time (review
Q12): at most 9.4 ms a call on average, and 0.39 s for the slowest single
call, on one process of a shared machine. Each call sees the full dense
`labeled` list, about 2,500 entries at B31. The legacy Predictor's slowest
call was 1.44 s (dense multi_swebench).

### Not explained

* **Low-budget costs, not investigated.** On dense real_webagents with
  per-pair splits, hier's pooling costs +0.015 ± 0.010 of Brier at B1 (the
  item share +0.0058 ± 0.0025) before it gains from B3 on. The legacy
  Predictor's pooled difficulty costs +0.0043 ± 0.0016 and +0.0093 ± 0.0019
  at B1 on dense multi_swebench ('pair', 'benchmark'). A cost at B1 is the
  signature the repository's notes say to check first for a mishandled
  second-order term.
* **At the platform's composition SHIP is not distinguishable from the legacy
  Predictor on these public runs:** +0.0024 ± 0.0023 / 0.0017 ('pair', z
  about 1.0 on the cluster SE) and +0.0045 ± 0.0024 / 0.0015 ('benchmark'),
  against -0.0010 ± 0.0008 / 0.0019 / 0.0015 and +0.0004 ± 0.0007 / 0.0019 /
  0.0015 unweighted (`reading["ship - legacy"]`). By company (scope 'pair',
  cluster SE), SHIP minus legacy is +0.0050 ± 0.0030 for the 'alone' class,
  -0.0019 ± 0.0025 with one companion and -0.0034 ± 0.0019 with more
  (`formative.pair.comparisons["ship - legacy"].by_companions`).
* **That 'alone' class is 45% swe_rebench.** The single-subject benchmark is
  alone in every run by construction: 87 of the 193 'alone' appearances
  (`single_subject_confound.appearances`). Without them SHIP minus legacy is
  +0.0019 ± 0.0033 ('pair') and +0.0030 ± 0.0028 ('benchmark') for the
  'alone' class (106 appearances), and +0.0004 ± 0.0026 / 0.0024 and
  +0.0017 ± 0.0022 / 0.0022 at the platform's composition
  (`single_subject_confound.comparisons["ship - legacy"]`). So most of the
  'alone' class's cost, and of the platform-mix cost, is the single-subject
  benchmark's (+0.010 there; "A single-subject benchmark", P1.15), not pairs
  alone on a multi-subject benchmark, where the residual is within about one
  cluster SE (0.6 and 1.1 for the 'alone' class, 0.2 and 0.8 for the mix).

### Checks and provenance

* No fallback fired and no fit is unconverged (`checks.failures`,
  `.unconverged`).
* The three hier variants are identical at B0, and the two legacy variants
  below WARMUP (`checks.identities`).
* Dense real_webagents matches `results/hier_eval.json`'s stored rows to
  within 5e-7 (`checks.dense_crosscheck`).
* All 208 tasks' rows carry the script digest b4504d05; the summary code
  changed after them, twice (the platform mix and reading table, then the
  single-subject readings; the script is now 4143835a). The `recheck` stage
  re-scored five stored tasks with the current script, bit-identical
  (`checks.recheck`, rerun after the second change), and every number the
  summary held before the second change is unchanged.

### Caveats

* **Dense SEs are conditional.** They are over pairs, given one `labeled`
  list and one split, and the across-benchmark SE rests on four benchmarks.
  Three dense benchmarks are scored on at most 32 items a pair.
* **The platform's composition rests on 26 pair appearances** from three
  runs. The replica's five-benchmark catalogue puts many more pairs together
  on a benchmark than the platform does.

## The gate, tightened (P1.12)

`python experiments/gate_and_ci.py --stage S` for S = `gate`, `evidence`,
`scale` (by family: `reliability`; `ratings judge4b`; `strong14b entropy`;
`embeddings`; `tfidf`), `ci` and `single`. Each stage ran as one process under
0.7 GB; the longest, the TF-IDF refit, took 206 s (`passes`). The script
replays no predictor. It reads stored results, rows and features, and refits
only the text maps (TF-IDF, embedding ridge and kNN) and the Rasch
difficulties. Every number below is in `results/gate_and_ci.json`, under the
key in brackets. `tests/test_gate_and_ci.py` covers the script.

**What it asks.** The review (W6, P1.12) asked to drop "pre-registered" or
evidence it, to give the pass probability at r = 0.3, to report the gate under
mix/whole, to put every correlation on one scale with the honest difficulty's
reliability, and to correct the B1 statement.

**Reproduction.** All six claims about the gate table that the draft's §4.5
makes hold against `results/harness_thresholds.json` (`gate.draft_checks`).
The within-pair r of every 4B, 14B and entropy feature equals its stored
`r_within_pair_tl`. All 28 embedding point estimates and leave-one-out group
intervals of `results/emb_transfer.json` reproduce exactly. The five-fold
Rasch refits match `data/strong_llm_eval/targets.json` exactly.

### Pass probability

For the transferred line, a covariate is one noise draw (`gate.lines["transferred
nested"].asked`). "Draws passing" is the stored count with its Jeffreys 95%
interval. P(clear -0.002) is the chance that one draw's test-like line clears
the bar, from the 8 draws' mean and sd, under a normal model and under a t
predictive. P(full gate) also models the other conditions.

| honest r | draws passing | Jeffreys 95% | P(clear -0.002), normal | t predictive | P(full gate), model |
|---|---|---|---|---|---|
| 0.2 | 1/8 | [0.01, 0.45] | 0.09 | 0.12 | 0.09 |
| 0.3 | 6/8 | [0.41, 0.94] | 0.79 | 0.76 | 0.68 |
| 0.4 | 7/8 | [0.55, 0.99] | 1.00 | 0.99 | 0.87 |
| 0.5 | 8/8 | [0.74, 1.00] | 1.00 | 1.00 | 1.00 |

* **At r = 0.3 a covariate passes about two times in three.** The mean is
  -0.00255 with a draw sd of 0.00068. Of the two failing draws, one misses
  the bar and the other fails the worst-parent condition. That second failure
  is inferred: the other four conditions hold, and a draw's worst parent is
  not stored.
* **The r a covariate needs** (`gate.lines.<line>.r_for_probability`). A
  per-draw chance of 0.5, 0.8 or 0.95 of clearing the bar needs an honest r
  of 0.27, 0.305 or 0.335 on test-like runs, which is a within-pair r of
  about 0.22 to 0.28 (interpolated in the gate table). The per-pair line needs
  0.455 to 0.475.
* **Read on mix/whole** (bar read on mix/whole, selection kept on test-like
  runs), the same chances need 0.215, 0.24 and 0.255, and the per-pair line
  0.365 to 0.39. The pass counts there are 4/8 at r = 0.2 and 8/8 from 0.3.
  They are upper bounds, because a draw's mix/whole worst parent is not
  stored.
* **The full gate under mix/whole was not run.** It needs `harness.py --stage
  table` with SELECT_ON = 'mix', the gate reading mix/whole's estimate and
  worst parent, and each draw's worst parent kept. The rows exist, so only
  the table stage reruns, about 25 minutes on one process
  (`gate.full_mix_gate`). It was deferred because it needs that code change
  in `experiments/harness.py`'s `gate()` and `average_lines()`, outside this
  study's scripts, and because it would only move the degraded oracles'
  pass counts. The stored mix/whole nested lines of the 44 measured
  covariates the draft's Figure 2 (b) draws (transferred and per-pair,
  selection kept on test-like runs) are all above -0.002, the best being the 14B's
  time_log_minutes at -0.0014 (`results/strong_llm_eval.json`,
  `harness.time_log_minutes.lines["transferred nested"].mix`). A full
  mix/whole gate could pass a covariate only where selection on mix/whole
  switched on one that test-like selection did not.

### What the gate can see at B0 and B1

A centred covariate is zero at B0. At B1 a per-pair slope is zero too,
because the pair's single label is its own centre. A slope transferred from
other benchmarks acts at B1 about as much as at later budgets (`gate.b1`):

* the honest oracle's transferred line gains 0.0360 of Brier at B1 (cluster
  SE 0.0040), and its per-pair line exactly 0;
* at the gate's r = 0.3 the transferred line gains 0.0026 of Brier at B1,
  against 0.0027 to 0.0030 at B3 to B31;
* the uncentred B0 term gains -0.0027 ALC even for the honest oracle, and
  fails the gate.

### When the gate was fixed

"Pre-registered" overstates what the repository shows (`evidence`):

* **The gate first appears in commit f7e7d87** (2026-09-27 04:45 UTC),
  together with `harness_thresholds.json`, `itemcov_eval.json` and
  `subject_side.json`. For those three the repository shows only that gate and
  results were committed at once.
* **`itemsig_eval.json` was committed a day earlier** (bd0be67), before any
  commit held the gate.
* **The 14B's data came after it.** The rubric and attempt data were produced
  13.6 hours after f7e7d87, and the entropy job's (commit D) 83.8 hours after.
  The entropy job's own reading rule was committed in 78e303e, before its
  data.
* **The other probes were committed after the gate** (`llm4b_close.json`,
  `attempt_probe.json`, `hidden_state_probe.json`, `icl_probe.json`,
  `finetune_encoder.json` in 4d2cc4f, `heads_eval.json` in 00bdf04). The
  repository does not record when their inputs were produced.

### One correlation scale

**The honest difficulty is reliable** (`scale.families.reliability`).
Split-half Rasch refits give 0.87 to 0.95 and the fold-overlap formula 0.84
to 0.95. A correlation against full-sample difficulty is therefore only 0.5%
to 1.6% higher than against the honest one (shrink factor 0.984 to 0.995).
The implied within-pair reliability is 0.80 to 0.92.

**Every quoted correlation on the gate's within-pair scale.** The bar is
0.247 at an honest r of 0.3 and 0.33 at 0.4 (`scale.families.<family>.covariates.<covariate>.within_pair`):

| covariate | within-pair r | 95% cluster interval |
|---|---|---|
| TF-IDF, leave one benchmark out | 0.065 | [0.03, 0.10] |
| embedding ridge, leave one benchmark out | -0.106 | [-0.13, -0.08] |
| embedding kNN, leave one benchmark out | -0.040 | [-0.06, -0.02] |
| embedding ridge fitted within the same benchmark (not transferable) | 0.251 | [0.23, 0.27] |
| 4B judge, four features (absolute value; the harness fits the sign) | 0.04 to 0.09 | |
| 14B primary head (rubric ridge) | 0.108 | [0.08, 0.14] |
| 14B judged solve share | 0.127 | [0.10, 0.16] |
| 14B rubric sum | 0.129 | [0.10, 0.16] |
| 14B rubric scales and time, the highest two (work; time_log_minutes) | 0.148; 0.145 | [0.12, 0.17]; [0.12, 0.17] |
| 14B attempts' entropy (matharena only; x varies on 20% of appearances) | 0.315 | [0.27, 0.35] |
| 14B reasoning entropy, ent_first1024 (commit D) | 0.123 | [0.09, 0.15] |

On the scale the gate's honest r is defined on (Pearson with the honest
difficulty over a parent's items, mean over the four parents with a
stratified group interval; `.parent_scale.mean_over_parents`): TF-IDF 0.03
[-0.05, 0.13], embedding ridge -0.10, the 14B primary head 0.19 [0.11, 0.26],
the judged solve share 0.21, and the reasoning entropy 0.19 [0.12, 0.25]
(0.32, 0.13, -0.01 and 0.32 by parent). The blind ratings reach 0.30 [0.18,
0.44] on the rated items only, and 0.02 to 0.08 as the harness would read them,
with unrated items at the mean.

**What made the draft's numbers generous is the unit, not the target.** A
degraded oracle keeps 0.82 of its r within a pair (0.81 to 0.87 over the
gate table's rows; `ci.unit_ratio.oracle`). The 14B's primary head, judged
solve share, rubric sum and expert-time estimate and the reasoning entropy's
primary keep only 0.57 to 0.65 (`ci.unit_ratio.named`). Over every covariate
whose parent-scale interval excludes 0 the share runs from 0.30
(rubric_interaction) to 1.02 (the transferred embedding ridge, negative on
both scales), with ent_first256 at 1.01 (`ci.unit_ratio.clear`). For TF-IDF,
the 4B judge and the embedding kNN the parent-scale r is near 0, so the
quotient means nothing (`ci.unit_ratio.not_clear`). The correlations the
draft quoted were taken over a parent's whole range, within groups
(Spearman), or over text-bearing subsets, while the gate reads a covariate
within test-like pairs.

## Intervals for every transfer correlation (P1.13)

The `ci` stage of `experiments/gate_and_ci.py` stores a 105-row table
(`ci.rows`). Each row has a group-bootstrap, an item-bootstrap and a Fisher
interval (2,000 resamples), recomputed where the study stored none and taken
from its results file where it did. Groups are item_features groups
(competition, language, website, paper). Group intervals over multi_swebench's
8 languages are indicative only.

**TF-IDF, leave one benchmark out** ("What transfers between benchmarks").
The quoted values are Pearson correlations with transfer.py's naive target,
the solve-rate logit over all items, not with Rasch difficulty:

| benchmark | naive target, all items | group 95% | Rasch target, text-bearing items | group 95% |
|---|---|---|---|---|
| matharena | 0.14 | [0.06, 0.23] | 0.10 | [0.01, 0.20] |
| multi_swebench | 0.23 | [0.16, 0.27] | 0.14 | [0.09, 0.23] |
| real_webagents | 0.09 | [-0.13, 0.31] | -0.02 | [-0.40, 0.28] |
| researchcodebench | -0.22 | [-0.47, 0.07] | -0.16 | [-0.44, 0.10] |
| swe_rebench | 0.16 | item [0.13, 0.18] (no groups) | 0.11 | item [0.09, 0.14] |

**TF-IDF within matharena** (5-fold, text-bearing items, groups are
competitions): 0.80 [0.70, 0.86], and 0.58 [0.48, 0.66] with the competition
mean removed. "What transfers between benchmarks" quotes 0.73 and 0.49; those
do not reproduce at HEAD.

**Embeddings** (Qwen3-Embedding-0.6B; "Neural embeddings do not carry
difficulty to an unseen benchmark"), Pearson with group 95% intervals, for
matharena, multi_swebench, real_webagents and researchcodebench:

| map | matharena | multi_swebench | real_webagents | researchcodebench |
|---|---|---|---|---|
| ridge, leave one benchmark out | -0.16 [-0.29, -0.03] | -0.03 [-0.16, 0.04] | -0.23 [-0.43, -0.01] | -0.02 [-0.31, 0.26] |
| kNN, leave one benchmark out | -0.08 [-0.21, 0.06] | -0.01 [-0.06, 0.02] | -0.02 [-0.22, 0.20] | +0.16 [-0.11, 0.38] |
| ridge, 5-fold within the benchmark | 0.66 [0.56, 0.73] | 0.27 [0.20, 0.34] | 0.38 [0.19, 0.52] | 0.51 [0.30, 0.66] |
| item_features group mean alone | 0.57 [0.39, 0.69] | 0.12 [-0.06, 0.23] | 0.41 [0.19, 0.55] | 0.52 [0.30, 0.67] |
| ridge, whole groups held out | 0.40 [0.29, 0.51] | 0.16 [0.09, 0.25] | -0.02 [-0.19, 0.15] | -0.09 [-0.30, 0.11] |

**Blind ratings** ("Language-model difficulty judgement"; n = 45 a
benchmark): the Fisher intervals reproduce. Group-bootstrap intervals are
matharena [0.25, 0.64], multi_swebench [-0.17, 0.48] and real_webagents
[0.12, 0.41]. The pooled within-benchmark value is 0.17 with a Fisher interval
of [0.03, 0.31] at n = 180. `experiments/llm_rating/analysis.py`'s pooled
interval subtracts 3 from n twice (`ci(r, len(df) - 3)`); the difference does
not show at two decimals.

**The 4B judge, the 14B's heads and features, and the reasoning entropy:** the
stored intervals are collected in `ci.rows`. For the entropy (Spearman within
group, group 95%): matharena +0.29 [0.16, 0.40], multi_swebench +0.12 [0.06,
0.16], real_webagents -0.03 [-0.19, 0.15], researchcodebench +0.33 [0.17,
0.45]. For the 14B primary head (Pearson, leave one parent out): +0.23 [0.04,
0.40], +0.01 [-0.08, 0.11], +0.19 [0.11, 0.28], +0.33 [0.12, 0.51]. The 14B's
attempt entropy on all attempted texts: 0.35 [0.25, 0.45].

**The 4B probe heads and the fine-tuned encoder** are not in `ci.rows`; their
own results files store the intervals. Leave one benchmark out, the entropy
head's random-effects mean over the four parents is -0.05 [-0.24, 0.14] and
the hidden-state head's +0.12 [-0.004, 0.24]
(`results/hidden_state_probe.json`, `heads.entropy|hidden.random_effects`).
The fine-tuned encoder's held-out Pearson with honest difficulty is -0.08
[-0.22, 0.06], +0.05 [-0.01, 0.10], -0.09 [-0.32, 0.15] and -0.11 [-0.41,
0.10] (matharena, multi_swebench, real_webagents, researchcodebench; group
intervals), random effects -0.05 [-0.14, 0.05]
(`results/finetune_encoder.json`,
`eval.per_cov.finetuned.per_parent.<parent>.honest.pearson` and
`.random_effects`). The known-sign cues' sign checks carry SEs, not
intervals ("Item covariates with a known sign").

**No interval is computable** (`ci.not_computable`) for swe_rebench on the
harness scales (one subject: no subject folds and no test-like pairs), for the
blind ratings within a pair (too sparse) or on researchcodebench (none rated),
for the 4B judge on real_webagents and researchcodebench (not extracted), and
for the 14B's attempts outside matharena.

**Powering the 4B judge is not needed** (`ci.power_4b`). The review (W7)
asked to run the local 4B judge on several hundred items a benchmark, because
the blind ratings' intervals include r = 0.3 outside mathematics.

* **The item count no longer limits the answer.** The 14B has since rated
  every item of the four parents, the same judged solve share included.
  Excluding r = 0.3 at 80% power would take 258 items on multi_swebench, but
  847 on real_webagents, which has 233 items in all.
* **Where items are plentiful, both local judges sit well below the bar.** On
  multi_swebench the 14B's solve share is 0.07 [-0.04, 0.19] (Pearson with
  fold-averaged difficulty) and the 4B's rating 0.11 [0.06, 0.21] (Spearman
  within group, sign as declared; stored for the raw rating as -0.11 [-0.21,
  -0.06]). The draft's Figure 2 (c) draws the rating's Pearson over all rated
  items, 0.11 [0.06, 0.18] as declared: another statistic from the same
  entry of `results/llm4b_close.json`.
* **Where items are few, the 14B still fails the gate.** Its solve share is
  0.34 [0.13, 0.52] on researchcodebench (the one parent where it reaches
  0.3) and 0.18 [-0.03, 0.39] on real_webagents against fold-averaged
  difficulty, but only 0.13 within a test-like pair, and its best nested line
  is -0.00084.

More 4B items could narrow an interval. They could not move a covariate that
the stronger judge, at full coverage, does not carry through the gate.

## A single-subject benchmark (P1.15)

`python experiments/gate_and_ci.py --stage single` (4 s, 0.38 GB). It reads
the stored rows of `experiments/ship_confirm.py`'s matched arms and of P1a.
Every number below is in `results/gate_and_ci.json`, `single`.

swe_rebench is the one public benchmark with a single subject (6,306 items).
In public R1 runs the 1,000-item cap cuts its evaluation half to a different
subset in every run: on runs 0 to 99, 86 appearances have a mean pairwise
Jaccard overlap of 0.026. The SE below is therefore over appearances, which
resample this one subject's items. It is not a cluster SE.

Shipped minus comparator, ALC per appearance, ± SE over appearances:

| library, runs | appearances | minus legacy Predictor | minus smoothed Beta(2,2) |
|---|---|---|---|
| run 2's archive (seed 0), benchmark-first, runs 0-149 | 130 | +0.0101 ± 0.0012 (weighted +0.0099) | +0.0130 ± 0.0014 (weighted +0.0127) |
| run 2's archive (seed 0), pair-uniform, runs 0-99 | 7 | +0.0121 ± 0.0065 (weighted +0.0114) | +0.0151 ± 0.0077 |
| current archive (seed 11, P1a), benchmark-first, runs 0-59 | 51 | +0.0094 ± 0.0016 | +0.0117 ± 0.0018 |
| current archive (seed 11, P1a), pair-uniform, runs 0-59 | 9 | +0.0024 ± 0.0042 | +0.0043 ± 0.0045 |

("Weighted" weights each appearance by 1 / its run's pair count, as
`results/ship_confirm.json` does.)

* **The loss sits at B0 to B3.** Against the legacy Predictor, benchmark-first
  at seed 0, it is +0.024 at B0, +0.018 at B1 and +0.013 at B3, and under
  0.006 from B7 on.
* **It is a small part of the public regimes' difference.** swe_rebench
  carries +0.0010 of benchmark-first's -0.0017 against the legacy Predictor
  (seed 0).
* **Against P1a's alternatives** (seed 11, benchmark-first;
  `single.seed11_current_library.R1B.vs`): SHIP does better than sigma_mu 5
  (by 0.0073 ± 0.0010), the calibrated smoothed mean (0.0062 ± 0.0006), the
  neighbouring configuration (0.0032 ± 0.0004), sigma_mu 3.5 (0.0025 ±
  0.0004), the adaptive EB level on mu0 -3.0 (0.0015 ± 0.0008) and hier
  without its subject prior (0.0009 ± 0.0001). The two wider level priors
  gain at B0 (by 0.005 and 0.012 of Brier) and lose from B1 on. The adaptive
  EB level on SHIP's own prior is level (+0.0004 ± 0.0008). hier with its
  fitted EB hyperparameters does better than SHIP, by 0.0022 ± 0.0010, as do
  the legacy Predictor and the smoothed mean.

**What it cannot say** (`single.not_measured`): variation between subjects or
between single-subject benchmarks is not measured, so no interval generalises
to a hidden single-subject benchmark. swe_rebench is excluded from test-like
runs by default, so the tuned-regime gains say nothing about such
benchmarks. Its pair accuracy is about 0.49, near the public centre, so a
single-subject benchmark far from it is not covered either.

## Row files for release (P1.17)

`python tools/export_rows.py` writes eight deterministic tar.xz archives (sorted
members, mtime 0, owner 0, mode 0644), a `MANIFEST.json` with every member's
sha256, and `SHA256SUMS` to `data/release_rows/`, which is gitignored with the
rest of `data/`. `verify` checks the archives and members against the
manifest, and `restore` unpacks them into `data/`, checking every member.

| set | read by |
|---|---|
| `subject_side_rows` | `ship_confirm.py`, `subject_side.py --summarise`, `formative_feedback.py`, `level_audit.py`, `script_revisions.py --stage reread` |
| `regime_sensitivity_rows` | `regime_sensitivity.py summarise` and `review` |
| `harness_rows` | `harness.py --stage table` and `eval`, `llm4b_close.py`, `strong_llm_eval.py`, `hidden_state_probe.py`, `finetune_encoder.py`, `gate_and_ci.py` |
| `harness_rows_legacy` | `heads_eval.py --rows legacy`, `itemcov_eval.py`, `mcq_floor.py` |
| `hier_floor` | `hier_floor_replay.py --stage summary` |
| `strong_llm_eval` | the 14B's derived tables (features, entropy, honest targets, the heads' out-of-fold predictions), `gate_and_ci.py` |
| `baselines_p1_rows` | `baselines_p1.py summarise` (with `regime_sensitivity_rows`) |
| `pooling_decomposition_rows` | `pooling_decomposition.py summarise` and `recheck` |

The export never copies measurement-db itself, `data/features` (item
features, embeddings, the Kaggle exports) or `third_party/`. The manifest
states that the rows derive from measurement-db, whose gated CC-BY-SA terms
apply to them. Re-run on rows restored from the archives,
`experiments/ship_confirm.py` passes every check with identical numbers, and
`baselines_p1.py summarise` and `pooling_decomposition.py summarise` give
summaries identical to `results/baselines_p1.json` and
`results/pooling_decomposition.json` except for timestamps and wall times
(and `baselines_p1.json`'s `ble` section, which the `ble` stage writes from
`third_party/`, not from the rows). Re-exported with the two new sets, the
six older archives are byte-identical to the first export.

Still open:

* **Hosting, TODO(team).** The archives exist locally only. They need a home
  (a Hugging Face dataset or Zenodo) under measurement-db's CC-BY-SA terms.
  That needs network access and the team's sign-off on publishing derived
  rows of a gated dataset, so it is left to the team.
* **The itemsig rows cannot be exported.** They were written to a scratch
  `--rows` directory and are not in `data/`.
