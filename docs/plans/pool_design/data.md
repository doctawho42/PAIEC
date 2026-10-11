# The expanded public pool: data inventory (descriptive)

measurement-db at revision `54074453da7b9911be0f48e4c560be547bc84d21`, fetched
2026-10-07T13:12:36Z by `tools/fetch_pool.py` into `data/pool/` (manifest
`data/pool/manifest.json`). Repository HEAD a725df9, read only.

What this note is and is not. It counts benchmarks, subjects, pairs and
fields, and describes benchmark levels. It runs no predictor and computes no ALC
or Brier on any data. It uses no formative feedback. It reads no item text:
`items.parquet` was not downloaded. Every number below comes from two scripts in
this directory, run with `PYTHONDONTWRITEBYTECODE=1` so that nothing outside the
directory is written:

* `inventory.py` (111 s) writes `inventory.json`, `benchmarks.csv` (one row per
  catalogue entry) and `pairs.csv` (one row per eligible pair: counts, rate,
  logit, the eight visible fields, display_name, canonical name; no item text).
  Its stdout is in `inventory_run.txt`.
* `inventory2.py` reads those outputs plus the raw legacy tables and writes
  `inventory2.json`.

`inventory.log` is the output of a first run that stopped while writing its JSON
(a tuple used as a dict key); the rerun's output is `inventory_run.txt`. A
plugin hook blocks reading `*.log` here, so I have not read it. The other files in this directory (`survey.*`,
`proto_*.py`, `time_fits.py`, `eval_describe/`) are not mine; I did not read or
change them.

## 1. Catalogue and file integrity

* 255 catalogue entries: 252 `release == public` and 3 withheld
  (`imagenet_hard`, `swe_together`, `visual_memory`). All 252 public entries have
  `granularity == item`.
* Every file listed in the manifest is present, and its size and sha256 match
  (`inventory.json: integrity_failures = []`).
* Public `response_type`: binary 143, mixed 45, continuous_bounded 14,
  continuous 14, ordinal 14, fraction 8, continuous_unbounded 5, likert_5 4,
  nominal 2, win_rate 2, likert_10 1.
* In every public benchmark, the responses' `benchmark_id` equals the catalogue's
  `benchmark_id`, which equals the folder name.

## 2. Eligibility as our code applies it

The rule, with its source:

* Only responses equal to 0.0 or 1.0 count. `paiec/data.py:36-37` keeps
  `resp.response.isin([0.0, 1.0])`, which also drops the nulls the new schema
  allows (`parquet_schemas.yaml:490-502`). `official.eligible` filters again
  (`paiec/official.py:75`).
* A benchmark whose responses are not binary in kind is dropped whole, as
  mmdocrag ('fraction') is (`paiec/official.py:73`, `paiec/data.py:19`). matharena
  is kept after dropping its non-binary tail (`paiec/data.py:3-5`,
  `docs/protocol.md:63-66`), and its catalogue type is `mixed`. The rule is
  therefore extended by type: `binary` and `mixed` benchmarks are kept with
  their 0/1 responses, and every other `response_type` is dropped whole.
* A pair is `(subject_id, benchmark_id)` (`paiec/data.py:46`). It needs at least
  80 distinct items left (`paiec/data.py:47`, `paiec/official.py:78`,
  `paiec/evaluator.py:51`).
* A pair's rate counts each item once, on its first binary response in file
  order. Acquisition reveals `by[k][0]` (`paiec/official.py:359`), the prior fits
  `rs[0]` (`paiec/prior.py:259`), and `Pair.by_item` keeps file order
  (`paiec/evaluator.py:78-82`). Which recorded response the platform reveals is
  listed as unknown (`docs/protocol.md:146`).

### Results

| | benchmarks | pairs | distinct subject_ids |
|---|---|---|---|
| binary or mixed, public | 188 (143 + 45) | | |
| with at least one eligible pair | **166** (128 binary, 38 mixed) | **13,499** (12,622 + 877) | **13,157** |

* There are 13,100 distinct visible eight-field subject dicts and 34,387,842
  distinct (pair, item) cells. Every eligible pair's subject_id is in its
  benchmark's subjects table, and subject_id is unique in every table.
* The legacy five come out exactly as before: matharena 81, multi_swebench 82,
  real_webagents 26, researchcodebench 31 and swe_rebench 1, for 221 pairs
  (`docs/protocol.md:185-187`).
* Pairs per benchmark: median 13, quartiles 5.25 and 28.75, 10th to 90th
  percentile 2 to 83, maximum 4,278. Benchmarks with at least k pairs:

  | k | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
  |---|---|---|---|---|---|---|---|
  | benchmarks | 166 | 152 | 145 | 130 | 102 | 57 | 25 |

  A benchmark needs at least 2 pairs to give relative standings
  (`paiec/prior.py:361`). Fourteen have only one.
* Items per pair: minimum 80, 10th percentile 204, median 488, 90th percentile
  3,088, maximum 433,218.
* A few benchmarks hold most of the pairs. `alignment_faking` has 4,278 (31.7%;
  its subjects are RL-training checkpoints) and `sib200` has 3,341 (24.7%;
  fine-tuned classifiers). The top ten hold 76.5% of all pairs
  (`inventory2.json: pair_concentration`). Any pair-weighted statistic is
  dominated by them, so the statistics below weight benchmarks equally unless
  they say otherwise.
* Of the 188 binary or mixed benchmarks, 22 have no eligible pair. In four of
  them, all binary, every response is null: chartmuseum, ineqmath,
  mmmu_dev_val, naturalreasoning. The full list is in
  `inventory.json: kept_type_no_eligible_pair`.

### Where the extended rule is uncertain (data facts, no decision taken)

* **Mixed benchmarks.** matharena's share of 0/1 among its non-null responses is
  0.9991. In 29 of the 38 eligible mixed benchmarks that share is below 0.95
  (610 pairs; for example rclicks 0.058, fetv 0.093, agc_bench 0.30, lawbench
  0.62). A mixed benchmark declares its scale per item in
  `items.grading_criterion` (`parquet_schemas.yaml:162`), which was not
  downloaded. Without it, a 0/1 response on a binary item cannot be told apart
  from the endpoint of a non-binary scale. vhelm is one sign of this: its pairs'
  logits have an sd of 5.35, because many of its pairs are all-0 or all-1.
* **Other types under the literal code.** `official.eligible` drops only the
  names in `D.FRACTIONAL`. Handed the pool as it is, the code would also admit
  1,600 pairs from 45 benchmarks of other types whose 0/1 values reach 80 items
  (`inventory.json: literal_code_other_types`; for example rewardbench 167,
  livebench 170, infibench 121). The extended rule above excludes them.
* **Direction.** Six binary benchmarks are `lower_is_better`, where 1 is the bad
  outcome: crow, emoji_attack, harmbench, jailbreakbench, llmail_inject and
  wmt_mqm (214 pairs). Three are `unordered`: alignment_faking (4,278 pairs),
  safeagentbench (6) and tensortrust (3). researchcodebench's scale has no
  direction. The schema states there is "no automatic rescaling, grade
  reversal" (`parquet_schemas.yaml:315-322`), so the
  responses are as stored.
* **Nulls.** 44 binary benchmarks hold null responses (for example mj_bench,
  106,650 of 209,509). The 0/1 filter drops them.
* **test_condition and interactors.** 102 eligible benchmarks fill
  `test_condition`. In 25 of them, a subject-item cell holds more than one
  condition: 1,409,289 cells, of which wmt_mqm holds 1,089,810. Our loader
  ignores the column, so the conditions become repeats of one item. Eight
  eligible benchmarks fill `interactors` (468,946 rows). `paiec/data.py:41` sets
  the item's interactors to `""`, while `hier.item_groups` would read them
  (`paiec/hier.py:532-542`).
* **Repeats.** 89 eligible benchmarks have repeated cells, and 78 have
  `max_trial > 1`. In four, the first response in file order is not the
  lowest-trial one: bigfinancebench 43 cells, indeterminacy 4,411,
  lambda_fp_course 91, tau2_bench 372.

## 3. Subject tables: fields and coverage

All 166 eligible subject tables have the same ten columns: `subject_id`,
`display_name`, then the eight visible fields `normalized_name`, `provider`,
`release_date`, `access_date`, `harness`, `reasoning_effort`, `harness_version`
and `subject_features_extra`. predict() sees only the eight
(`paiec/data.py:20-21, 68-71`), so `display_name` is not visible. The schema
says the following:

* `subject_id` is a sha256 of the canonical identity plus every metadata column
  (`parquet_schemas.yaml:40-48`).
* `normalized_name` comes from a model registry and is null when no mapping
  exists (lines 53-60).
* `provider` is filled together with it (lines 61-67).
* `release_date` is one date per normalized_name (lines 68-77).

Coverage of the eight visible fields over the 13,499 eligible pairs (a pair is
one subject row), and with each eligible benchmark counted once:

| field | pairs non-empty | mean over benchmarks | benchmarks with any / all | distinct values |
|---|---|---|---|---|
| normalized_name | 49.0% | 58.4% | 113 / 79 | 666 |
| provider | 49.0% | 58.4% | 113 / 79 | 81 |
| provider, informative (not "Unknown" or "(agent scaffold)") | 33.5% | 57.7% | | |
| release_date | 33.8% | 57.8% | 113 / 71 | 331 |
| access_date | 0.7% | 1.2% | 2 / 2 | 12 |
| harness | 93.8% | 86.1% | 143 / 142 | 148 |
| harness_version | 3.3% | 3.0% | 5 / 5 | 5 |
| reasoning_effort | 0.24% | 0.4% | 3 / 0 | 6 |
| subject_features_extra | 96.7% | 88.6% | 147 / 147 | 12,668 |
| (display_name, not visible) | 100% | 100% | 166 / 166 | 3,672 |

What the values look like:

* **provider.** "Unknown" fills 1,899 pairs and "(agent scaffold)" fills 198.
  `attrs` reads both as an ordinary provider string, and "Unknown" lower-cases to
  the same `unknown` that an empty field gets (`paiec/subjects.py:120`). Of the
  non-empty provider strings, 61.2% fall in `prior.PROVIDERS` once lower-cased
  (`paiec/prior.py:41-43`). The largest outside it are opengvlab, llava,
  microsoft and hugging face.
* **release_date.** All 4,565 non-empty values match the ISO pattern
  `subjects.ISO` (4,537 YYYY-MM-DD and 28 YYYY-MM). By pair, the years are
  2019 1, 2020 1, 2022 68, 2023 367, 2024 3,314, 2025 636 and 2026 178.
* **harness.** In 134 of the 143 benchmarks with a harness, every pair carries
  the same value. It names the evaluation pipeline (for example "HELM",
  "VLMEvalKit", "Anthropic alignment-faking RL evaluation"). The `harness` flag
  in `attrs` (`paiec/subjects.py:128`) is therefore on for 93.8% of pairs. In the
  legacy data it marked agent scaffolds.
* **reasoning_effort.** 32 pairs: high 17, medium 6, xhigh 4, low 3, extended 1,
  max 1. 'extended' is not in `EFFORT_RANK` (`paiec/subjects.py:51-52`).
* **subject_features_extra.** The most common keys are source_model_label
  (5,377), model_identifier (5,128), and experiment_name, initialised_from,
  intervention and training_step (4,278 each, all alignment_faking).
  `source_model_name`, the key `canon_name` falls back on, gives a name to only
  51 pairs.

## 4. Matching subjects across benchmarks

How hier keys a subject:

* `Flags.subject_key = "canon"` (`paiec/hier.py:442`, documented at 424-428).
* `identity_key` (`paiec/hier.py:1918-1927`) returns `("n", canon_name(subject))`.
  When that is empty, it returns `("k", sk)`, where `sk` is the digest of all
  eight fields (`paiec/predict.py:109-110`). A subject without a canonical name
  therefore links only to identical eight-field dicts.
* `theta_key` (`paiec/hier.py:1929-1935`) shares one theta per identity key
  across the run's benchmarks.
* The prior's identity table is keyed on `canon_name` as well
  (`paiec/prior.py:630`, built at 656).
* `canon_name` (`paiec/prior.py:84-111`) takes normalized_name, or else
  `source_model_name=` from subject_features_extra. It lower-cases the name,
  drops parentheticals, turns `-_/` into spaces and drops the words in
  `PROVIDERS`.

Counts over the 13,499 eligible pairs, by key:

| key | pairs with key | distinct | in ≥2 / ≥3 / ≥5 / ≥10 / ≥20 / ≥30 / ≥50 benchmarks | pairs on keys in ≥2 | max benchmarks |
|---|---|---|---|---|---|
| canon_name (hier) | 6,669 (49.4%) | **688** | **366 / 117 / 68 / 31 / 12 / 4 / 1** | 3,860 | 52 ("gpt 4o") |
| normalized_name, lower-cased | 6,619 | 666 | 362 / 116 / 66 / 31 / 11 / 4 / 1 | 3,829 | 52 |
| (normalized_name, provider, release_date) | 6,619 | 667 | 363 / 115 / 66 / 31 / 11 / 4 / 1 | 3,829 | 52 |
| all eight fields | 13,499 | 13,100 | 233 / 50 / 18 / 0 / 0 / 0 / 0 | 575 | 8 |
| subject_id | 13,499 | 13,157 | 233 / 50 / 18 / 0 / 0 / 0 / 0 | 575 | 8 |

* **Distinct subjects.** There are 688 canonical names (canonical names in at
  least 4 benchmarks: 92). The 6,830 pairs with an empty canonical name hold
  6,623 distinct eight-field dicts. hier would therefore see 7,311 identity keys
  in all. The 6,830 are mostly sib200 (3,341), afrieval (948), bridging_gap
  (414), mathvista_mini (263), mmbench_v11 (251), truthfulqa_mc and mmlu (150
  each).
* **Under the stricter readings.** With binary benchmarks only (128
  benchmarks): 630 names, of which 333 / 100 / 52 / 21 / 7 are in ≥2 / ≥3 / ≥5 /
  ≥10 / ≥20 benchmarks. With binary plus mixed at a 0/1 share of at least 0.95
  (137 benchmarks): 659 names, 342 / 107 / 57 / 24 / 9.
* **Legacy names.** 107 canonical names appear on the legacy five. 74 of them
  also appear on new benchmarks, and 57 on at least three new ones.
* **Consistency.** Two normalized names collapse into one canonical name once:
  "Nex N1" and "Nex N1 (DeepSeek V3.1)". One normalized name has two release
  dates ("qwen3 next 80b a3b"). None has two providers. Dropping provider words
  leaves short names: "01", "7b", "aki", "cua", "cwm", "m1", "m2", "m3", "o1",
  "o3", "ola", "r1", "v1", "v3", "v4", "vl2". None of them collides in this pool,
  but each is only a few characters long.
* **Pairs sharing a canonical name on one benchmark.** 49 benchmarks have them:
  326 groups, 5,176 pairs. alignment_faking alone accounts for 4,278 of those
  pairs ("claude 3.5 sonnet" 1,689 checkpoints, "claude sonnet" 475, "claude 3
  opus" 50). The others include bird_sql 120, multi_swebench 75, hallusionbench
  and ai2d_test 69 each, and vhelm 59. In a run, hier gives such pairs one shared
  theta. In the prior table their standings are averaged per (name, benchmark)
  cell (`paiec/prior.py:627-656`).
* **Identical visible dicts on one benchmark.** 11 groups, 68 pairs, in 8
  benchmarks. predict() cannot tell those subjects apart, and the platform
  predicts identical inputs once (`docs/protocol.md:86-89`).
* **Model labels canon_name does not read.** 5,755 of the 6,830 pairs without a
  canonical name carry a visible label in subject_features_extra:
  source_model_label 5,280, source_model_identifier 414, model_identifier 61.
  Normalized like canon_name, only 163 of them (96 names, 14 benchmarks, mostly
  mathvista_mini and mmbench_v11 with 60 each) match an existing canonical name.
  Taken as a key of their own, they give 593 names, 264 of them on two or more
  benchmarks. Recorded as a data fact only; using them would be a code change.

## 5. Benchmark levels

**Definition.** A benchmark's level is the mean, over its eligible pairs, of the
continuity-corrected logit `log((k + 0.5) / (n - k + 0.5))`. Here n is the
pair's distinct items, and k counts those whose first binary response is 1. This
is `experiments/hier_design/levels.py:60-65`'s corrected logit, except that it
uses first responses instead of all of them. Each benchmark counts once. The
sampling variance of a level is the pairs' variance over n for two or more
pairs, and the binomial delta-method variance for one pair.

| set | benchmarks | mean | sd | sd net of sampling | 5% | 10% | 25% | 50% | 75% | 90% | 95% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all eligible | 166 | **+0.06** | **1.54** | 1.51 | -2.38 | -1.61 | -0.77 | +0.15 | +0.93 | +1.73 | +2.35 |
| binary only | 128 | +0.09 | 1.43 | 1.41 | -2.14 | -1.59 | -0.75 | +0.17 | +0.94 | +1.65 | +2.11 |
| binary + mixed with 0/1 share ≥ 0.95 | 137 | +0.08 | 1.40 | | -1.96 | -1.51 | -0.74 | +0.16 | +0.90 | +1.55 | +2.10 |
| the same, without lower_is_better and unordered | 128 | +0.20 | 1.27 | | -1.63 | -1.27 | -0.53 | +0.22 | +0.94 | +1.65 | +2.11 |
| mixed only | 38 | -0.05 | 1.88 | 1.84 | -2.90 | -1.86 | -0.95 | -0.23 | +0.86 | +1.90 | +2.76 |
| ≥ 5 eligible pairs | 130 | -0.02 | 1.49 | 1.46 | -2.59 | -1.58 | -0.82 | +0.11 | +0.90 | +1.50 | +2.04 |
| the 161 new benchmarks (legacy six excluded) | 161 | +0.08 | 1.55 | 1.53 | -2.41 | -1.61 | -0.75 | +0.16 | +0.94 | +1.78 | +2.42 |
| legacy five | 5 | -0.53 | 0.89 | | | | | -0.74 | | | |

Variants and the cut by release date:

* The level's form barely matters. With a plain logit clipped at 0.005: mean
  +0.05, sd 1.47. With all responses in place of first ones: +0.05, 1.54. As the
  logit of the benchmark's mean rate: +0.04, 1.28.
* The six lower_is_better benchmarks average -2.03 (sd 2.37) and make up the
  bottom of the distribution. The minimum is llmail_inject at -5.42.
* By pair rather than by benchmark, the mean is +0.01 and the sd 2.10.
* Within a benchmark, the median sd of pair logits is 0.67 (152 benchmarks with
  two or more pairs). The pooled sd is 1.74, raised by alignment_faking's 2.57.
  The legacy pooled value was 0.93 (`levels.json: sigma_u_pooled`).
* **Subjects released in 2025 or later.** 814 pairs on 59 benchmarks have a
  release date of 2025-01-01 or later. Their levels: mean +0.26, sd 1.34, median
  +0.50. Against all pairs of the same benchmarks, recent minus all is +0.18 (sd
  0.58), or +0.06 (sd 0.46) on the 47 benchmarks with at least 3 recent pairs.
  Only 33.8% of pairs carry a date, and dated pairs lean toward models of 2024.

### Against the five legacy public benchmarks

| benchmark | level here (first response) | levels.json `mean_logit` (all responses) | rank among the 166, ascending |
|---|---|---|---|
| multi_swebench | -1.614 | -1.618 | 17 |
| real_webagents | -0.917 | -0.943 | 38 |
| researchcodebench | -0.741 | -0.738 | 44 |
| swe_rebench | -0.099 | -0.082 | 72 |
| matharena | +0.735 | +0.477 | 115 |

* Computed on all responses, the levels reproduce `levels.json`'s `mean_logit`
  to the third decimal (`inventory.json: legacy5_new_tables.level_cc_all`).
  Their mean, -0.581, is that file's average. `levels.json`'s mu0 of -0.558 and
  sd of 0.905 are a logit-normal-binomial fit, not this mean.
* The legacy five sit on the low side of the pool: four of them are below its
  median, and their mean is 0.59 below the pool's.
* Their sd (0.89) is 0.58 of the pool's (1.54).
* matharena's first-response level (+0.73) differs from its all-response level
  (+0.48) because 99.8% of its items repeat (`docs/protocol.md:189-190`).

### Against the shipped LEVEL (mu0 -2.5, sigma_mu 2.5)

LEVEL (`submission/model.py:68`) is on hier's item-level scale. To place it on
the accuracy-logit scale of the levels above, map a level mu to
`f(mu) = logit E[sigmoid(mu + sqrt(T) z)]` with z standard normal and
T = sigma_d² + sigma_g² = 9.513. T is the item variance of
`submission/prior.json`'s hyper for a benchmark without group keys
(`paiec/hier.py:1955-1959`). The probit approximation of f is a factor of
0.4595, the 0.46 of `docs/findings.md:989-991`. f is monotone, so LEVEL's
quantiles map one to one:

| LEVEL quantile | 5% | 10% | 25% | 50% | 75% | 90% | 95% |
|---|---|---|---|---|---|---|---|
| on the accuracy-logit scale | -3.40 | -2.84 | -1.99 | **-1.15** (p = 0.24) | -0.37 | +0.32 | +0.73 |
| share of the 166 pool levels below it | 2.4% | 4.2% | 6.6% | **15.7%** | 32.5% | 57.2% | 68.7% |

* 84% of eligible benchmarks lie above LEVEL's median, and 31% above its 95th
  percentile.
* The share below LEVEL's median under the other readings: binary only 14.1%;
  binary plus mixed at a 0/1 share of at least 0.95, 13.9%; the same without
  flagged directions, 10.9%.
* Inverting f puts the 166 levels on the item scale at a mean of +0.14 and an sd
  of 3.05, against N(-2.5, 2.5²). Their probability integral transform under
  LEVEL has mean 0.76. 25.9% of them fall in LEVEL's central 50%, and 66.3% in
  its central 90%.
* Caveats on this comparison:
  * It assumes every benchmark has the shipped T, while real item variances
    differ by benchmark.
  * A mean pair logit includes the subjects' mean standing as well as the
    level.
  * hier's own fitted public levels come from `prior.benchmark_fit`, a different
    estimator.
  * The legacy five inverted this way read +1.62, -3.45, -2.01, -1.63 and -0.22.
* LEVEL was set for the hidden test on evidence recorded in `docs/findings.md`
  ("Calibrating for the hidden test"). This note neither revisits that evidence
  nor scores anything. The competition page says hidden and public benchmarks
  were assigned to their pools at random before curation, which makes the pool's
  levels a description of the population the hidden ones come from. Which
  benchmarks and subjects the test actually samples is a separate matter.

## 6. Schema: the new formatted_tables against the tables our loaders read

**Paths.** The loaders read `data/<b>/{response,items,subjects}.parquet`
(`paiec/data.py:33-35`; fetched as `{response,items,subjects,benchmarks}` by
`paiec/fetch.py:22-24`). The pool is laid out as
`data/pool/<b>/formatted_tables/{benchmarks,subjects,responses}.parquet`
(`tools/fetch_pool.py:85, 98-99`). The response file is now plural,
`responses.parquet`, and `items.parquet` is fetched only with `--items`
(`tools/fetch_pool.py:64`).

**Columns.** For the legacy six, the bc8204d8 tables and the new
formatted_tables have identical column names, order and types:

* responses (8 columns): subject_id, item_id, response_id, benchmark_id, trial,
  test_condition, interactors, response.
* subjects (10 columns), as in section 3.
* benchmarks (25 columns).

All 166 eligible subject tables share the same ten columns (schema version 3,
`parquet_schemas.yaml:1-31`). So the old tables were already on this schema.

**Content of the legacy six against the bc8204d8 tables:**

* Subjects tables, row counts and benchmark catalogue rows are identical, except
  multi_swebench's `n_items` (2,126 to 2,127) and `coverage`.
* matharena, real_webagents, researchcodebench, swe_rebench and mmdocrag keep
  every item_id and response_id, and their responses are identical as
  multisets.
  * matharena's row order changed (README line 290). The first binary response
    of every one of its 20,317 subject-item cells is the same row value and the
    same trial as before.
* multi_swebench changed (the 2026-10-02 "full-content and grading-protocol
  corrections", README line 290):
  * Only 339 of its 2,126 old item ids and 9,670 of its 57,975 response ids
    survive.
  * Row order is unchanged, and subject_id, response, test_condition and
    interactors agree row for row.
  * One old item became two new ones: 61 rows, 34 subjects. Repeated cells fall
    from 167 to 140, and 27 trial values are renumbered.
  * 27 pairs gain one item. k (first-response successes) is unchanged for all 82
    pairs, and the level moves from -1.6133 to -1.6143.
* README line 251 says item and response ids changed in the migration. Measured
  against the tables this repository used, that holds for multi_swebench only.
* The eligible pairs and their first-response counts of the five legacy binary
  benchmarks are therefore unchanged, apart from those 27 multi_swebench pairs'
  item counts.

**What the new data needs that the loaders do not provide** (facts, not
changes):

* `load_pairs` hard-codes the benchmark list (`paiec/data.py:18-19`), so the
  type-based rule of section 2 has to come from each benchmark's
  `benchmarks.parquet` `response_type`.
* `load_pairs` requires `items.parquet` (`paiec/data.py:34`). The item dict it
  builds feeds item_features to hier's group effects (`paiec/hier.py:532-542`,
  `paiec/prior.py:269-270`) and item_content to the MCQ floor
  (`paiec/hier.py:1944-1953`). Without items, a fit has neither.
* `response` is nullable, and the 0/1 filter drops nulls. `test_condition`
  (102 eligible benchmarks) and per-response `interactors` (8) are ignored by
  the loader, and `interactors` is hard-coded to `""` (`paiec/data.py:41`).
* A `mixed` benchmark's scale lives per item in `grading_criterion`
  (`parquet_schemas.yaml:162`).
* `response_scale.direction` can be `lower_is_better` or `unordered`.
* `display_name` is present in every subject table and is not one of the eight
  visible fields.

## 7. What this means for prior fitting

This section reads sections 1 to 6 for a refit of hier's prior (`prior.build`,
`paiec/prior.py:810-816`) on the pool. It decides nothing: the choices belong to
amendment 4. New counts come from `section7.py`, which reads only `pairs.csv`
and `benchmarks.csv` and writes `section7.json`. Figures marked "prototype"
come from `build_K100_M2000.json`, the output of the interrupted attempt's
`proto_build.py`: the library's `benchmark_fit` and `from_standings` on the 128
binary benchmarks, with at most 100 pairs (in proto_loader's stable-hash order)
and 2,000 items per benchmark. It has no direction filter, and it scores
nothing. Its item cap dropped sib200: each sib200 pair has 204 of the
benchmark's 83,640 items, so about 5 per pair survive a 2,000-item cap, below
the 80 a pair needs. "The ordered set" below is binary plus mixed at a
0/1 share of at least 0.95 (137 benchmarks) less the nine direction-flagged
ones: 128 benchmarks and 8,388 pairs, 115 benchmarks with two or more pairs and
13 with one.

### Which benchmarks

* **Subject part** (standings, attribute ridge, identity table;
  `paiec/prior.py:355-367, 370-391, 557-667`): the ordered set.
  * The 29 mixed benchmarks below a 0/1 share of 0.95 (610 pairs) cannot be
    read without `grading_criterion` (section 2).
  * In the six lower_is_better benchmarks a stronger subject scores lower, and
    alignment_faking's checkpoints have no order. Their standings would enter
    the ridge and the identity cells with the opposite sign or none.
  * Dropping the nine removes 4,501 pairs, 95% of them alignment_faking's.
* **Level part.** The flagged benchmarks belong to the population the hidden
  ones are drawn from, are stored as is, and fill its bottom (section 5:
  lower_is_better mean -2.03). A level summary of the pool would keep them:
  +0.08, sd 1.40 with them (137 benchmarks), +0.20, sd 1.27 without.
* **Single-pair benchmarks** (13 in the ordered set) give no standing
  (`paiec/prior.py:361`) but do give a level, fitted at the median item
  variance (`paiec/prior.py:769-771`).
* **hier's item-scale level is fragile on small benchmarks.**
  * In the prototype, 4 of the 127 fitted levels lie beyond ±10: workarena
    -47.6, disco -38.1, agentic_review_perturbation -14.2 and finegrain_t2i
    +18.9. Each rests on 3 to 5 pairs, with an item variance of 112 to 1,835
    (median over benchmarks 5.80).
  * 9 fits did not converge in EM and 13 in Newton.
  * fit_hyper's unweighted mean and sd (`paiec/prior.py:787-789`) then give
    mu0 -0.39 and sigma_mu 9.19, against a median level of +0.33.
  * A level taken from the pool would therefore need a robust summary, or the
    accuracy-scale one of section 5. A level fixed in advance is written over
    the fit, as LEVEL is now (`tools/build_submission.py:118`), so the rest of
    the refit does not depend on it.
* **LEVEL is a level at attribute score 0** (`paiec/prior.py:774-779`). hier
  adds attr_scale times the attribute mean to theta (`paiec/hier.py:1902-1903`).
  A refitted ridge moves the subjects' scores (the prototype's sigma_attr is
  0.21, the shipped 1.02). The same mu0 is then the same effective level only if
  the mean score of the subjects it meets has not moved. That can be checked on
  the pool's subject dicts under both priors without scoring anything.
* **Item variance and group shares.** Without `items.parquet` a fit sees no
  `item_features` (`paiec/prior.py:269`, `paiec/hier.py:532-542`). Only the
  per-response interactors then give group keys. In the prototype, 4 benchmarks
  had any, and their median share reached G_CAP, 0.25 (`paiec/prior.py:49,
  761-762`). The prototype's median item variance, 5.80, is below the shipped
  T of 9.51 (section 5).

### Which subjects

* Every eligible pair on a kept benchmark is one standings row. The data give no
  reason to drop pairs by subject, only to weight them (below).
* **Unnamed pairs.**
  * 6,334 of the ordered set's 8,375 rows (75.6%) have no canonical name, 4,289
    of them on sib200 and afrieval.
  * They enter the ridge with provider `unknown` and no date
    (`paiec/subjects.py:120-121`) and link nothing (`paiec/prior.py:630-633`).
  * A hidden subject without a name gets that same attribute cell and nothing
    else, so these rows are the only ones that speak for it.
* **Several pairs of one name on one benchmark** are averaged into one identity
  cell (`paiec/prior.py:633, 639-656`) but stay separate ridge rows. In the
  ordered set there are 707 such pairs, in 214 groups on 38 benchmarks (on all
  166: 5,176 pairs, as in section 4).
* **Fields.** Only the eight visible fields may enter, so `display_name` may
  not. `source_model_label` would name 5,755 more pairs, but reading it is a
  change to `canon_name` (section 4).
* **Some attributes mean something else in the pool** (section 3).
  * `harness` is on for 93.8% of pairs and constant within 134 of 143
    benchmarks. With standings centred within benchmarks
    (`paiec/prior.py:365`), its coefficient rests on the nine where it varies,
    and it no longer marks an agent scaffold.
  * Provider "Unknown" merges with an empty provider.
  * 'extended' effort has no rank.
  * The prototype's ridge has 33 coefficients and 24 providers; the shipped
    prior has 17 coefficients.

### Weighting by benchmark

What each part of `paiec.prior` counts once:

| quantity | counts once each | where |
|---|---|---|
| mu0, sigma_mu | benchmark (mean, sd) | `paiec/prior.py:774-789` |
| sigma_d, sigma_g | benchmark with ≥ 2 pairs (medians) | `paiec/prior.py:756-764` |
| ridge coefficients; total_var, s2_res, u_mean, hence sigma_delta and sigma_attr | standings row, i.e. pair | `paiec/prior.py:385, 620-625, 796-798` |
| identity table, benchmark offsets | (name, benchmark) cell | `paiec/prior.py:491-511, 627-656` |
| tau2_res, hence sigma_theta | pair of benchmarks of one name | `paiec/prior.py:525-538, 794` |

* **The pair-weighted parts are where the concentration matters.**
  * Uncapped, the rows' effective number of benchmarks (Kish, (Σn)²/Σn²) is
    5.9 of 152 on all 166 benchmarks (alignment_faking 31.7%, top two 56.5%).
    On the ordered set it is 5.5 of 115 (sib200 39.9%, top two 51.2%).
  * Capping each benchmark at K pairs, in proto_loader's order, gives 50.4 at
    K = 100 (3,136 rows; the largest, afrieval, holds 3.2%), 69.1 at K = 50
    and 91.6 at K = 20.
* **On the ordered set the weighting barely moves the variances.** The
  within-benchmark variance of pair logits is 0.79 pooled by pair and 0.79
  averaged by benchmark (median 0.41). On all 166 the two are 3.03 and 1.46,
  because of alignment_faking (6.59 over 4,278 pairs). On the ordered set the
  weighting matters mainly for the ridge's coefficients.
* **A cap costs links.**
  * On the ordered set, 340 names have cells on two or more benchmarks
    uncapped, 176 at K = 100, 138 at K = 50 and 102 at K = 20.
  * Of the 164 lost at K = 100, 112 lose their cell on ai2d_test, 107 on
    hallusionbench and 8 on ikp. ai2d_test and hallusionbench have 254 and 253
    pairs, all named, with 216 and 215 names.
  * Ranking pairs by a hash of the canonical name, one order for every
    benchmark, keeps 212 at K = 100.
  * Weights of min(1, K/n_b) on the rows of `_ridge` and on from_standings' row
    means would keep all 340. That is a change to `paiec/prior.py`, not a
    choice of data.
* **tau2 is weighted by name, whatever is done to pairs.** On the ordered set
  "gpt 4o" (43 benchmarks) carries 903 of the 5,708 cross-products (15.8%).
  Eight names carry half, and the top ten 57%. K = 100 leaves 4,957
  cross-products.
* **Size is not a constraint at these counts.**
  * `_residuals` keeps one n-vector per row of a linked name
    (`paiec/prior.py:459-461`): 112 MB per pass uncapped on the ordered set,
    396 MB on all 166.
  * The prototype's from_standings took 2.4 s CPU with a peak of 224 MB, and
    its 127 benchmark fits 565 s CPU in all.

### Canonical-name coverage

* **Linking now rests on data.**
  * In the ordered set, 583 names have rows. 340 of them are on two or more
    benchmarks, 106 on three or more and 23 on ten or more.
  * The shipped prior has 106 table entries and 21 linked names
    (`submission/prior.json`, meta).
  * In the prototype (167 linked names, flagged benchmarks included), tau2_res
    came out 0.40, against -0.002 on the public five
    (`paiec/prior.py:599-602`). That is above the clip's ceiling of 0.2
    (`paiec/prior.py:794`), which the docstring calls within noise of zero
    (`paiec/prior.py:701-721`). sigma_theta would therefore sit at the ceiling,
    0.447.
  * Every identity weight in the prototype was above id_cap, 0.1
    (`paiec/hier.py:1910`), which is not estimated (`paiec/prior.py:731`). On
    the public fit, at the clip's floor, 91% of the rows whose name has two
    residual benchmarks and 12% of the others were (`paiec/prior.py:714-718`).
  * Whether the clip and id_cap stay is a choice for amendment 4.
* **Coverage is uneven.**
  * Uncapped, 24.4% of the ordered set's rows carry a name (52.4% at K = 100),
    or 65.4% averaged over benchmarks. 31 of its 115 multi-pair benchmarks have
    no named row.
  * hier links on `canon_name` both at run time and in the prior
    (`paiec/hier.py:1918-1927`, `paiec/prior.py:630`). An unnamed hidden
    subject gets the attribute prior alone.
  * The public data cannot say how many hidden subjects carry a name. The
    attribute part alone carries the unnamed pairs: 50.6% of all eligible
    pairs, 75.6% of the ordered set's rows. Named and unnamed subjects are
    therefore worth describing separately in any evaluation.

## Pointers

* `inventory.json` holds everything above, including the catalogue counts,
  per-benchmark response profiles, the test_condition, interactors and repeats
  tables, coverage, the link tables, the level summaries and the old/new
  comparison.
* `inventory2.json` holds the concentration figures, the stricter level sets,
  the recent-subject cut, the label keys, benchmark-weighted coverage and the
  raw-table checks.
* `benchmarks.csv` and `pairs.csv` hold the per-benchmark and per-pair rows.
* `section7.json` (from `section7.py`) holds section 7's counts, for all 166
  benchmarks, the binary ones, the 137, the ordered set and the ordered set
  without sib200, each uncapped and at K = 100, 50 and 20.
