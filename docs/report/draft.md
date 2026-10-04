# Calibrating a hierarchical response model for a benchmark-level cold start

Technical report for the Predictive AI Evaluation Competition (PAIEC), NeurIPS 2026 [PAIEC organisers 2026].

Author: Nikita L. Polomoshnov

Affiliation: Moscow State University

Email: nikitapol@fbb.msu.ru

Code: https://github.com/doctawho42/PAIEC (MIT License)

Keywords: predictive evaluation, item response theory, Brier score, cold start, PAIEC

Every number traces, through the script that produced it, to `docs/findings.md` (cited as F§ and a section title), `docs/protocol.md` (P§) or a committed `results/` file, except some wall times in the notes under App G.4's table. Appendix H gives the convention and maps each number to its code.

---

## Abstract

PAIEC asks for the probability that an AI system answers a benchmark item correctly, given visible attributes and 0 to 31 revealed labels. Scoring uses the area under the Brier learning curve (ALC, lower is better) on unseen benchmarks. We rebuilt the organisers' evaluator. An earlier replica made pooling, the use of other subjects' labels, look like the main lever. Under the verified protocol pooling still gains 0.009 to 0.030 ALC on runs holding a whole benchmark. At the platform's one or two pairs per benchmark it gains 0.0004 to 0.001: runs are too small, whatever labels the predictor sees. Our predictor, a hierarchical Bayesian item-response model refitted from the labels at every checkpoint, gains mainly through its prior on a new benchmark's level (mean log-odds of success). We set the prior's three global hyperparameters on test-like runs from public data that resemble the first formative feedback, our first submission's scores on hidden benchmarks. On held-out test-like runs it beats that submission by 0.042 ALC. The gain depends on how runs are built (the regime): 0.014 without the synthetic date shift, 0.025 to 0.031 at the feedback's levels, within noise on public runs. A smoothed mean with a calibrated prior ties it in the tuned regime and trails by 0.004 to 0.011 elsewhere, public runs included. Its two formative runs, each within 1.02 single-run sds (the spread of one run's ALC) of every regime's mean, cannot separate the regimes. Except for a blind rating that tracks difficulty on mathematics only, every completed item-side signal fails to transfer or misses an acceptance gate of 0.002 ALC.

---

## 1 Introduction

### 1.1 The task

For each subject $s$ (an AI system) and item $i$, a submission's `predict` function reads two dictionaries and returns the probability $\hat p_{si}$ that $s$ answers $i$ correctly. The subject dictionary has eight string fields, such as normalized_name, provider, release_date, harness and reasoning_effort. The item dictionary holds item_content, item_features, interactors and an anonymous benchmark_id. The function also receives `labeled`, the outcomes revealed so far [P§ Interfaces].

A subject and a benchmark form a pair $(s, b)$. The items of each pair with at least 80 distinct items are split 50/50 into an acquisition pool, from which labels are revealed, and an evaluation pool $\mathcal E_{sb}$. Budget $k$ is the point at which $k$ labels per pair have been revealed, and the platform scores predictions at the budgets $k \in \mathcal K = \lbrace 0, 1, 3, 7, 15, 31\rbrace$. At each budget it measures each pair's Brier score over the recorded responses on the evaluation pool:

$$
B_k = \frac{1}{\lvert \mathcal R_{sb} \rvert} \sum_{(i, y) \in \mathcal R_{sb}} \big(\hat p^{(k)}_{si} - y\big)^2
$$

Here $\mathcal R_{sb}$ is the list of recorded responses of $s$ on the items of $\mathcal E_{sb}$, each an item $i$ with its binary outcome $y$, and every repeated trial counts. The prediction at budget $k$, $\hat p^{(k)}_{si}$, is made once per item and serves all of that item's responses. The Brier score is a strictly proper scoring rule [Brier 1950; Gneiting and Raftery 2007]. Each pair's scores are summarised as the area under its learning curve (ALC; lower is better):

$$
\mathrm{ALC} = 0.1\,B_0 + 0.2\,B_1 + 0.2\,B_3 + 0.2\,B_7 + 0.2\,B_{15} + 0.1\,B_{31}
$$

The $j$-th acquired label enters every $B_k$ with $k \ge j$. The first label therefore carries 0.9 of the total weight, and labels 16 to 31 carry 0.1 each [P§ Scoring]. The weights of $B_0$ and $B_1$ alone sum to 0.3. The platform's run score is the unweighted mean of the per-pair ALCs, so every pair counts equally (App D.1). During the competition the organisers also score each formative submission on a run over hidden benchmarks, a formative run, and we call their scored tables the formative feedback.

### 1.2 What makes it hard

**The level of a new benchmark is not in the training data.** A benchmark's *level* is the mean over its pairs of $\operatorname{logit} p$, where $p$ is a pair's true success rate. The five public benchmarks' levels average -0.56, with sd 0.90. A leave-one-out prediction, which sets each benchmark's level to the mean of the other four, misses by 0.86 logit on average (F§ "The step-2 analyses behind the model", `experiments/hier_design/levels.py`). The first formative run, which scored our first submission on hidden benchmarks, sat about 0.9 logit below the average of the public runs. That reading treats a pair's Brier score at the last budget as the expected score of predicting its true success rate $p$ for every item, and so solves $p(1-p) = B_{31}$ at the lower root. Matched pair by pair against their nearest public pairs in our replica of the official evaluator, the same run's pairs read as spread both ways. Later readings of the formative feedback are more central (§5.1, §5.4).

**Most of the possible gain is per item, and it does not transfer.** On all 221 public pairs, a constant prediction of 0.5 scores a Brier of 0.25. Knowing each pair's true success rate $p$ lowers it to 0.176, and knowing each item's true probability lowers it to 0.0433 (F§ "How much there is to win", `experiments/ceilings.py`). With one benchmark left out, item difficulty predicted from text does not transfer. A TF-IDF map from item text to difficulty correlates -0.16 to 0.14 with the items' Rasch difficulty on the held-out benchmarks, and every 95% interval lies below 0.3 (§6.3).

**Runs are small.** A formative run holds at most 1,000 subject-item entries across both pools, so at most 12 pairs of at least 80 items [P§ Runs]. Our three scored runs held 9, 8 and 9 pairs over the same 7 benchmarks, and in them 5, 6 and 5 pairs were the only pair from their benchmark (App D.1). Pooling across subjects therefore has little to draw on: at that composition it is worth 0.0004 to 0.001 ALC to hier, our model (§2.4).

**No state survives between budgets.** The platform recreates evaluation workers for every budget, so a submission's predictions have to come from a pure function of `(input, labeled)`, the two dictionaries and the revealed labels [P§ Flow].

### 1.3 Contributions

1. **A verified replica of the official evaluator, and what the first one got wrong** (`paiec/official.py`). It reproduces the organisers' streaming evaluation client decision for decision. Our first replica, the legacy replica, made cross-subject pooling look like the main lever. On the platform that lever disappears because formative runs are small. The change in which labels the predictor sees does not explain the loss (§2.3, §2.4).
2. **A hierarchical item-response predictor fitted from `labeled` alone** (`paiec/hier.py`). It integrates each item's own residual exactly, and it reads the target item's log-odds from the exact log posterior along one line in parameter space. Reading along this line reduces the under-reaction of the Laplace (Gaussian) approximation to a pair's first labels. This reading is exact for one pair whose labelled items share the target item's item_features group effects (§3).
3. **A calibration of the new-benchmark level, and an account of what it buys** (`paiec/testlike.py`). We set global hyperparameters only, on test-like runs built from public data, under a guard that limits losses on public runs. We confirm the shipped configuration, hier-ship, on held-out runs. We also test it, under a decision rule fixed in advance, at the levels that later readings of the formative feedback give. We then split its gain over our first submission into calibration and modelling (§3.4, §5).
4. **An acceptance harness and a catalogue of negative results** (`experiments/harness.py`). A covariate's honest $r$ is its correlation with an item difficulty fitted on other subjects, and the harness maps honest $r$ to the covariate's chance of passing the acceptance gate. An item-side idea is a signal about individual items, from a text-based difficulty map to a 14B model's reasoning entropy. Every item-side idea we completed either does not transfer to a held-out benchmark or stays below the gate. The exception is a blind rating, which tracks difficulty on mathematics only. Each idea has its script, and every text map, judge, probe and encoder has 95% intervals (§6).

### 1.4 Related work

**Item response models.** Our model, hier (§3), is an item-response model in the Rasch tradition [Rasch 1960], fitted hierarchically [Patz and Junker 1999; Fox 2010]. Its item-feature effects are those of explanatory item response theory (IRT) [Fischer 1973; De Boeck and Wilson 2004]. IRT has been used to build and compare ML and NLP test sets [Lalor et al. 2016; Martínez-Plumed et al. 2019; Rodriguez et al. 2021]. Other work chooses small informative item sets with IRT [Maia Polo et al. 2024], anchor points [Vivek et al. 2024] or a learned acquisition policy [Li et al. 2025]. In PAIEC the evaluation items are fixed, and a new benchmark's level dominates the error.

**Predicting performance.** Performance has been predicted from other evaluation records [Ye et al. 2023; Zhang et al. 2024], from shared capability dimensions [Ruan et al. 2024] and from rubric-based demand levels [Zhou et al. 2026]. Instance-level assessors predict it too, but lose their advantage out of distribution [Pacchiardi et al. 2024]. Setting a new benchmark's level resembles re-estimating class priors under label shift [Saerens et al. 2002; Lipton et al. 2018].

**Difficulty of unseen items.** Difficulty is estimated from question text [Benedetto et al. 2023] or embedded question content [Truong et al. 2025]. Work on agentic coding predicts success from issue text with repository state, tests and solutions [Ge et al. 2026], or from task length in human time [Kwa et al. 2025]. Model internals encode difficulty [Lugoloobi and Russell 2025] and predict a model's own success [Lugoloobi et al. 2026; Moreno Cencerrado et al. 2026], and token entropy transfers weakly between agentic benchmarks [Krsteski and Meyer 2026]. §6 tests such signals on benchmarks held out entirely.

### 1.5 Headline numbers

Our submission is hier-ship, and LegacyP is our first submission's predictor. In the table a negative $\Delta$ favours hier-ship, where $\Delta = \mathrm{ALC}(\text{hier-ship}) - \mathrm{ALC}(\text{comparator})$. A number in parentheses directly after a $\Delta$ value is its pair-cluster bootstrap SE (§4.3); a ± marks the SE that its row names. Test-like (TL) runs are built from public data to resemble the first formative feedback. Public runs (R1-bf, R1-pu) redraw public pairs at formative size (§4.1); §2.1 defines the names.

| what | value | where |
|---|---|---|
| formative runs, single draws: run 1 LegacyP; run 2 hier-ship; run 3 hier-ship's rebuilt archive (recomputed from its table) | ALC 0.2113; 0.1926; 0.1817 | §5.1 |
| $\Delta$ LegacyP, held-out TL runs 200 to 299 (seed 2); mean ± SE over the four multi-subject benchmarks (parent-level) | -0.0419 (0.0037); -0.038 ± 0.008 | §5.4 |
| $\Delta$ LegacyP, TL with item groups merged at random (TL-mix, runs 0 to 99, seed 3) | -0.0436 (0.0039) | §5.4 |
| $\Delta$ LegacyP, TL without the synthetic date shift (TL-noshift, runs 0 to 99, seed 3) | -0.0144 (0.0024) | §5.4 |
| $\Delta$ LegacyP, test-like runs set to two later readings of the feedback's level (READING / AUDIT, seed 11; realised level -0.63 / -0.93) | -0.0248 (0.0037) / -0.0313 (0.0049) | §5.4 |
| $\Delta$ LegacyP, public runs (R1-bf runs 0 to 149 / R1-pu runs 0 to 99, seed 0) | -0.0017 (0.0021) / -0.0018 (0.0017) | §5.4 |
| $\Delta$ Smooth-cal, a smoothed label mean with a calibrated prior: TUNED (TL redrawn) / R1-bf / R1-pu, seed 11 | +0.0015 (0.0019) / -0.0108 (0.0014) / -0.0100 (0.0018) | §5.4, §5.5 |
| $\Delta$ 1PL-ship, a plain Rasch model at hier-ship's level: same runs | +0.0017 (0.0013) / -0.0048 (0.0007) / -0.0050 (0.0007) | §5.5 |
| $\Delta$ Smooth, the $\mathrm{Beta}(2, 2)$-smoothed label mean: TL runs 0 to 199 (seed 2) / R1-bf runs 0 to 149 (seed 0) | -0.0169 (0.0020) / -0.0086 (0.0030) | §5.4 |
| $\Delta$ EmpMean, the organisers' empirical-mean baseline: TUNED / R1-bf / R1-pu (seed 11) | -0.0339 (0.0030) / -0.0464 (0.0027) / -0.0449 (0.0030) | §5.5 |
| ten alternative configurations at the two later readings (READING, AUDIT) | none gains the 0.002 that a rule fixed in advance required | §5.4 |
| $\Delta$ LegacyP, the single-subject public benchmark (swe_rebench) | +0.0101 ± 0.0012 (130 R1-bf appearances, run-2 library, seed 0); +0.0094 ± 0.0016 (51, current library, seed 11); SE over appearances | §5.6 |

Sources: `experiments/<name>.py`, which writes `results/<name>.json`, for formative_feedback, formative_run3, ship_confirm, regime_sensitivity, baselines_p1 and gate_and_ci (`single`); full SE triples in §5.4 and the appendices.

---

## 2 Data, protocol and our replica

### 2.1 Names and notation

We use one name per model (Table 1) and one per term or run set (Table 2) throughout. Figure legends keep older names, which each caption explains. App H.1 glosses the internal labels.

*Table 1: models.*

| name | what it is |
|---|---|
| **LegacyP** | Our first submission's predictor (`paiec/predict.py`): a per-pair logistic model with a multiple-choice floor, a subject ability around a relative attribute prior, and item difficulty from a joint item response theory (IRT) fit once a benchmark has 64 labelled items. Scored in formative run 1. |
| **hier** | The hierarchical item-response model of §3 (`paiec/hier.py`). |
| **hier-fit** | hier with every hyperparameter from the empirical-Bayes fit, including its level prior (no LEVEL). |
| **hier-ship** | hier with the level prior LEVEL: centre $\mu_0 = -2.5$, sd $\sigma_\mu = 2.5$ and attribute scale $\gamma = 0.5$, the weight of the subject-attribute prediction in a subject's prior (`LEVEL` in `submission/model.py`). Our submission, scored in formative runs 2 and 3. |
| **hier-rec** | hier with $\mu_0 = -3.0$, $\sigma_\mu = 2.5$, $\gamma = 0.25$: the level calibration's recommendation, which an audit replaced before shipping (§5.4). Its numbers never stand for hier-ship's. |
| **hier-argmax** | hier with $\mu_0 = -3.5$, $\sigma_\mu = 2.5$, $\gamma = 0.25$: the selection rule's choice. |
| **hier-ship σ3.5, hier-ship σ5** | hier-ship with $\sigma_\mu = 3.5$ or $\sigma_\mu = 5.0$. |
| **hier-EB** | hier whose new-benchmark level centre is re-estimated by empirical Bayes at every checkpoint, on a stated base prior: hier-EB on hier-ship, or on $\mu_0 = -3.0$, $\gamma = 0.5$ (App E.3). |
| **hier-nosubj** | hier-ship without its subject prior (attribute prior and identity link off). |
| **1PL-ship, 1PL-fit** | A plain one-parameter logistic model refitted from `labeled` (one pooled level per benchmark, one ability per pair, one difficulty per item), at hier-ship's level or at its own leave-one-parent-out public level. |
| **Smooth** | The $\mathrm{Beta}(2, 2)$-smoothed mean of the pair's labels. |
| **Smooth-cal** | A smoothed mean whose Beta prior (mean $m_0 = 0.25$, strength $n_0 = 2$) was chosen by LEVEL's selection rule on the tuned regime, regardless of the guard. |
| **EmpMean** | The organisers' empirical-mean baseline. |
| **BLE** | The organisers' language-model predictor (we did not run it; §5.5). |
| **LegacyP+fix** | LegacyP with a level fix (offset, scale and variance; values in the tables). |
| **oracles** | Base-rate oracle: every pair at its true rate $p$, so its Brier is $\mathbb E[p(1-p)]$. Pair-rate oracle: every response at its pair's rate. Item oracle: the in-sample Rasch probability per item. |

*Table 2: terms and run sets.*

| term | meaning |
|---|---|
| subject $s$, item $i$, pair $(s, b)$ | an evaluated AI system (eight visible string fields); one item of a benchmark $b$; a subject on one benchmark. ALC averages pairs equally. |
| appearance | one pair in one run. Replica runs redraw pairs, so a pair has many appearances. |
| budget $B_k$ | the checkpoint after $k$ labels per pair, $k \in \mathcal K = \lbrace 0, 1, 3, 7, 15, 31\rbrace$; $B_k$ also names the Brier score there. $\mathrm{ALC} = 0.1\,B_0 + 0.2\,(B_1 + B_3 + B_7 + B_{15}) + 0.1\,B_{31}$. |
| level; LEVEL | a benchmark's mean pair-accuracy logit; hier-ship's prior on the level of a new benchmark. |
| R1 (R1-bf, R1-pu); R2 | public formative-like runs, drawn with benchmark-first or pair-uniform weighting; dense runs: every eligible pair of one benchmark. |
| parent; pseudo-benchmark | a public benchmark (the four multi-subject parents carry every test-like number); a group of one parent's items given its own anonymous id in a test-like run. |
| TL; TL-mix; TL-noshift; TL-1.2, TL-2.0 | test-like runs at the defaults; with item groups merged at random and whole parents; without the date shift; with the target level mean (level_mean) at -1.2 or -2.0. |
| LA-0.8, LA-flat | the level audit's two extra regimes: target level mean -0.8, and no level tilt. |
| RS regimes | TUNED, READING, AUDIT, MIXTURE and FLAT: the regime-sensitivity (RS) study's test-like regimes (§4.1, App C.1). TUNED is TL redrawn at seed 11. |
| selection, confirmation, held-out runs | TL runs 0 to 99, 100 to 199 and 200 to 299; the held-out runs were never used to choose the level. |
| date shift; level tilt | test-like runs shift visible release and access dates 1.25 years later; they tilt the draw of pairs toward a target level distribution. |
| split scope | whether the 50/50 acquisition / evaluation split is drawn per pair ('pair') or per benchmark item ('benchmark'). |
| LOPO; strict run-LOBO; nested LOPO | fitted without the target's parent (leave one parent out); without every benchmark of the run (leave one benchmark out, run-level); selection redone inside each held-out parent and bootstrap resample. |
| guard; gate; rule | the guard: a configuration may lose at most 0.003 ALC to LegacyP, on both R1 weightings. The gate: the four acceptance conditions of §4.4. The rule: the RS study's decision rule (App C.2). |
| honest $r$; within-pair $r$ | correlation with item difficulty fitted on other subject folds ($\tilde d_i$); the same correlation within a test-like pair, the gate's scale. |
| single-run sd | the spread of one run's ALC over a regime's runs: the noise of one formative run. |
| run-2 library; current library | hier as in formative run 2's archive (old multiple-choice floor, solver before the floored-fit fix); hier with both fixes, as in archive-3 (`4d2cc4f`). |
| lower-root reading; pooled reading | run 1's pair rates read from $p(1-p) = B_{31}$ at the lower root; the reading of runs 1 and 2's 17 pairs against matched replica pairs (§5.4). |

### 2.2 Data

We use only measurement-db, the organisers' public training release [Truong et al. 2026]. It is gated, is licensed CC-BY-SA-4.0 and holds 571,921 responses over 287 subjects and six benchmarks [P§ Data]. Our copy is revision `bc8204d8…` (App G.2).

We exclude the fraction-valued mmdocrag (only 4.8% of its responses are 0 or 1), as the test excludes non-binary benchmarks, and drop matharena's 0.1% non-binary tail. Keeping pairs with at least 80 distinct items leaves 221 pairs and 225,843 responses:

| benchmark | pairs | distinct items | mean accuracy | item_features key (levels, share of item variance) |
|---|---|---|---|---|
| multi_swebench | 82 | 2,126 | 0.219 | lang (8, 0.05) |
| matharena | 81 | 1,633 | 0.607 | competition (25, 0.50) |
| researchcodebench | 31 | 212 | 0.354 | paper (20, 0.43) |
| real_webagents | 26 | 233 | 0.330 | website (12, 0.24) |
| swe_rebench | 1 | 6,306 | 0.479 | none |

Sources: App A.1, with the script that recounts the table.

Few subject ids cross benchmarks, but more of the underlying model names do. Only 22 of 287 subjects appear on more than one benchmark [P§ Data], and the `interactors` field is empty everywhere.

In hier, a subject's standing $\theta_s$ (the model's term for how strong the subject is) is keyed on the coarser canonical name: `normalized_name`, else the source model name. Keyed this way, 117 of the 220 pairs of the four multi-subject benchmarks share a canonical name with a pair on another benchmark (F§ "The step-2 analyses behind the model"). On current evidence the link adds almost nothing on top of the attributes, and we treat that reading as provisional (§3.3).

Four other properties of the data shaped the method:

- Repeated responses (one subject answering one item more than once) are uneven across benchmarks: 99.8% of matharena items have them, as do 99.6% of swe_rebench's and 0.4% of multi_swebench's [P§ Data].
- Some subject attributes exist on a single benchmark: harness strings on multi_swebench (81 of 82 pairs, 12 strings) and reasoning_effort on matharena (23 of 81 pairs; F§ "Subject side at budgets 0 and 1").
- Part of matharena is textless: a third of its items have no task text, and its 336 Kangaroo items are five-option multiple choice with the options in an image (App A.1).
- Raw solve rates confound difficulty, because items attempted only by recent subjects look easy. When we predict item difficulty, the target is therefore a Rasch estimate (`paiec/rasch.py`).

The public benchmarks are not a representative sample of the hidden task types. The hidden benchmarks come from the organisers' inventory of 161 candidates, assigned at random to the training and test pools. Our keyword rules on the titles class 67% of them as AI evaluations. Of those, 39% are text QA, 30% images, 15% agents, 6.5% code, 4.6% video, 2.8% math and 2.8% audio (App A.1).

### 2.3 The protocol, as verified

We read the organisers' streaming client alongside the competition page [P§]. The client is `tools/streaming_ingestion.py` in the baseline repository [PAIEC baseline repository 2026], at commit `82d330dd` (App G.2). Six properties of the protocol matter here:

- Scoring is budget-major: all pairs are evaluated at one budget before any pair moves to the next. Every pair of a run is evaluated at budget 0 with `labeled` empty. Acquisition then continues until every pair holds up to one label, every pair is evaluated at budget 1, and so on.
- Each checkpoint has one shared `labeled` list. At checkpoint $B_k$ it holds the first $k$ labels of *every* pair in the run, other subjects' and other benchmarks' included.
- The platform recreates its evaluation workers (up to 16) from the submission at every checkpoint, and identical inputs are predicted once per checkpoint.
- The default acquisition policy is a hash rule. A candidate is taken if and only if $u < \min(1,\, n_{\mathrm{lab}} / n_{\mathrm{items}})$, where $n_{\mathrm{lab}}$ is `labels_remaining` and $n_{\mathrm{items}}$ is `items_remaining`. Here $u \in [0, 1)$ is the first 8 bytes of a sha256 digest of the candidate and its context, read as an integer and divided by $2^{64}$.
- The platform uses only binary data, from pairs with at least 80 distinct items. Each pair's 50/50 split into acquisition and evaluation items is fixed across submissions and budgets.
- The runtime rejects bad outputs (non-finite or out of range) and signals failures by exit codes 40, 41 and 42.

Our replica (`paiec/official.py`) matches the client decision for decision and agrees with the analytic ALC of the empirical mean (App A.4).

### 2.4 What our first replica got wrong, and why it mattered

We built our first replica (`paiec/evaluator.py`) before the client was public and keep it as the *legacy* replica so that its numbers stay reproducible. It scored all 221 pairs in one session, pair-major (each pair through all six budgets before the next), showed each target its own labels and no others, and leaked identifiers to `predict`. App A.2 lists these four differences and a fifth, a split that depended on the process. Every number we quote from the legacy replica is marked "legacy".

On the legacy replica, evaluation order alone moved the assembled run-time predictor from 0.1898 to 0.1725 (legacy; F§ "Evaluation order is worth 0.017"). Its predictor ladder (predictors of growing complexity) made pooled item difficulty look worth about 0.020 ALC (legacy; F§ "The predictor ladder"). Both results pointed our research at pooling, the use of other subjects' labels.

At formative size the gain from pooling nearly vanishes. Over 600 official R1 runs (seed 0; §5.2), pooled difficulty adds about 0.0007 to LegacyP (F§ "Under the official protocol"). LegacyP needs 64 distinct labelled items on a benchmark to pool difficulty, and a formative run has about two pairs per benchmark.

To find what removed the gain, we switched pooling on and off on the official replica, under both split scopes, on dense and on formative-size runs, and reweighted the results to the platform's composition. Of the 26 appearances in formative runs 1 to 3, 16 were alone on their benchmark and 10 had one companion. App A.2 gives the method.

| pooling on minus off, ALC | dense, 'pair' | dense, 'benchmark' | formative, 'pair' | formative, 'benchmark' | platform mix, 'pair' / 'benchmark' |
|---|---|---|---|---|---|
| hier-ship, everything from other subjects | -0.0300 (0.0099) | -0.0142 (0.0059) | -0.0033 (0.0006) | -0.0020 (0.0005) | -0.0010 / -0.0004 (0.0003) |
| of which the pooled level | -0.0043 (0.0014) | -0.0029 (0.0009) | -0.0015 | -0.0012 | -0.0007 / -0.0003 |
| of which pooled item difficulty | -0.0258 (0.0088) | -0.0114 (0.0055) | -0.0018 | -0.0008 | -0.0004 / -0.0002 |
| LegacyP, pooled difficulty | -0.0237 (0.0065) | -0.0086 (0.0038) | -0.0010 (0.0002) | -0.0004 (0.0001) | 0 / 0 |

Dense: the four multi-subject benchmarks, mean (SE across them). Formative: 100 runs (seed 11, R1-bf), cluster SE. App A.2 has the full table. Source: `experiments/pooling_decomposition.py`, `results/pooling_decomposition.json`; F§ "Pooling under the verified protocol".

On dense runs pooling is still worth 0.009 to 0.030 ALC, which brackets the legacy ladder's 0.020. Most of it comes through item difficulty, and per-benchmark splits halve it (ratios 0.47 for hier-ship and 0.36 for LegacyP).

At formative size pooling is worth 0.002 to 0.003 to hier-ship and 0.0004 to 0.001 to LegacyP. At the platform's composition it is worth 0.0004 to 0.001 to hier-ship and exactly 0 to LegacyP.

The gain from pooling that the legacy replica showed is real on dense runs, but the platform's runs are too small to collect it. Run size removed the gain, while the change in which labels a target sees (its own pair's labels or every pair's) did not remove it.

### 2.5 Unknowns kept as parameters

The replica treats what the client leaves open as parameters [P§ Still unknown]. The main one is `split_scope`: whether the 50/50 split is drawn per pair or per benchmark item. On dense runs at $B_{31}$, per-pair splits let other subjects' labels land on 83% to 99.5% of a target's evaluation items, and per-benchmark splits let them land on 0% to 5%.

We report both scopes in three places: the dense R2 baselines (App A.3), hier-fit against LegacyP on public runs (App E.1), and the pooling decomposition (§2.4). Everything else ran with split scope 'pair': the R1 baselines, the level calibration and its guard, hier-ship's confirmation, the RS and baseline studies, the harness, and every official-protocol study of §6. App A.3 lists the other open points.

---

## 3 Method

### 3.1 Model

Our predictor, hier (`paiec/hier.py`), models the log-odds $\eta_{si}$ that subject $s$ answers item $i$ of benchmark $b$ correctly, and predicts the probability $\hat p_{si}$:

$$
\eta_{si} = \mu_b + \theta_s + \delta_{sb} - g_i - e_i, \qquad
\hat p_{si} = c_i + (1 - c_i - \varepsilon)\,\mathbb{E}\big[\operatorname{logit}^{-1}(\eta_{si}) \mid \mathcal L\big]
$$

Here $\varepsilon$ is the slip, and the expectation is over the posterior given the revealed labels $\mathcal L$.

| term | meaning | prior |
|---|---|---|
| $\mu_b$ | the benchmark's level, shared by every subject's labels on it | $\mu_b \sim \mathcal N(\mu_0, \sigma_\mu^2)$, Gaussian; Student-t optional |
| $\theta_s$ | the subject's standing, keyed on its canonical name | Gaussian, mean $\gamma\,\hat\theta^{\mathrm{attr}}_s$ plus a capped identity term, variance from $\sigma_\theta$ and the attribute ridge ($\sigma_{\mathrm{attr}}$ in the ridge's place without an attribute prior); display below |
| $\delta_{sb}$ | the pair's own deviation | $\delta_{sb} \sim \mathcal N(0, \sigma_\delta^2)$ |
| $g_i$ | item_features group effects, one per key whose visible values are neither all numeric nor constant | $\mathcal N(0, \sigma_g^2)$, divided evenly over a benchmark's keys; group share of item variance $\sigma_g^2 / (\sigma_g^2 + \sigma_d^2)$, capped at 0.25 |
| $e_i$ | the item's own residual | $e_i \sim \mathcal N(0, \sigma_d^2)$, or $\mathcal N(0, \sigma_d^2 + \sigma_g^2)$ on a benchmark without item_features keys, integrated exactly per item; $\sigma_d^2 + \sigma_g^2$ is the median public item variance |
| $c_i$ | guessing floor: the guess weight $\omega$ times the multiple-choice floor $f_i$ read from the item text (`paiec/mcq.py`) | $c_i = \omega f_i$, fixed, $\omega = 0.5$ |

In the standing's prior, $\hat\theta^{\mathrm{attr}}_s$ is an attribute ridge's prediction of the standing (§3.3), $u_s$ is the variance of that prediction from the ridge's coefficients, and $\gamma$ is the attribute scale. Its identity term carries the same name's standings on the public benchmarks into the prior, with a capped weight $w_s$:

$$
\theta_s \sim \mathcal N\big(\gamma\,\hat\theta^{\mathrm{attr}}_s + w_s\,\bar r_s,\ (1 - w_s)^2\,(\sigma_\theta^2 + u_s) + w_s^2\,v_s\big),
\qquad
w_s = \min\Big(\frac{\sigma_\theta^2 + u_s}{\sigma_\theta^2 + u_s + v_s},\ w_{\max}\Big)
$$

Here $\bar r_s$ is the mean attribute residual of the name's standings on the public benchmarks, and $v_s$ is its noise: the name's spread across benchmarks divided by the number of benchmarks behind $\bar r_s$. The cap is $w_{\max}$ (`id_cap`). A name with no public standing has $w_s = 0$. Without an attribute prior, the mean drops its attribute term, $\sigma_{\mathrm{attr}}^2$ takes the place of $u_s$, and $\bar r_s$ averages the raw standings (`paiec/hier.py`, `HierPredictor.theta_prior`).

As in a four-parameter logistic model [Barton and Lord 1981], the floor $c_i$ is the lower asymptote and the slip $\varepsilon$ is the gap between the upper asymptote and certainty. The group effects $g_i$ are item covariates, as in explanatory item-response models [Fischer 1973; De Boeck and Wilson 2004].

A component that no label touched (a new benchmark, subject, pair, group or item) stays at its prior, and the predictive distribution integrates over it. At budget 0 the prediction is therefore the level prior plus the attribute prior, integrated. That prediction is where the hidden test's headroom turned out to be (§5.1).

All components are on the scale of an item-level (Rasch) model [Rasch 1960], on which we measured the public variance components, and hier fits them from `labeled` alone.

### 3.2 Inference

**We integrate the item residuals instead of maximising over them.** A joint mode over all effects is penalised quasi-likelihood [Breslow and Clayton 1993], which is biased with binary responses and large random effects [Rodríguez and Goldman 1995; Breslow and Lin 1995]. With the shipped item variance near 9.5 ($\sigma_d^2 + \sigma_g^2 = 2.671^2 + 1.542^2$) and one label per item, the item effects absorb each label. A regression test at item variance 8.7 (`tests/test_hier.py`) pins the size of the effect. There, a pair with 160 of 200 successes is predicted at 0.70 on a new item under the joint mode, and at 0.79 once the item residual is integrated.

We therefore keep every component except the $e_i$ in one vector $\mathbf x$. Adaptive Gauss-Hermite quadrature with 20 nodes integrates each item's likelihood over its $e_i$ [Liu and Pierce 1994; Pinheiro and Bates 1995]. Newton's method finds the mode of the joint log posterior of $\mathbf x$ (App B.1).

**The target is read along its line.** Here the target is the subject and item being predicted. Its loading vector $\mathbf a$ picks out the components of $\mathbf x$ that enter its $\eta_{si}$, so $\mathbf a^\top\mathbf x$ is the part of the target's linear predictor that the labels inform. The Laplace (Gaussian) posterior of $\mathbf a^\top\mathbf x$ [Tierney and Kadane 1986] under-reacts to a pair's first labels, by 0.013 to 0.02 in probability after one to seven labels. It under-reacts because the mode of a skewed posterior sits nearer the prior than its mean.

We therefore read $\mathbf a^\top\mathbf x$ on a grid along the Gaussian conditional mean, with the exact log posterior at each point. With $\hat{\mathbf x}$ the posterior mode and $\boldsymbol\Sigma$ the Laplace covariance, the Gaussian conditional mean of $\mathbf x$ given $\mathbf a^\top\mathbf x = \zeta$ is

$$
\mathbf x(\zeta) = \hat{\mathbf x} + \boldsymbol\Sigma\mathbf a\,
\frac{\zeta - \mathbf a^{\top}\hat{\mathbf x}}{\mathbf a^{\top}\boldsymbol\Sigma\mathbf a}
$$

and at grid point $\zeta$ the log posterior is evaluated at $\mathbf x(\zeta)$, with the item residuals integrated. This is the Laplace strategy of the integrated nested Laplace approximation (INLA) with its conditional-mean simplification, which replaces the conditional mode by the Gaussian conditional mean [Rue, Martino and Chopin 2009, §3.2.2]. Unlike that strategy, our read leaves out the conditional Gaussian's normalising term.

Only when every label loads on $\mathbf x$ through $\mathbf a^\top\mathbf x$ alone does the read give the exact marginal, as for one pair whose labelled items share the target item's group effects. Otherwise the components off the line move along their Gaussian conditional mean with their skew ignored, and the read is an approximation. On cases drawn from the model, its error moves Brier by about 0.001 at $B_1$ and by at most $10^{-4}$ from $B_7$ on (App B.1).

### 3.3 Offline prior and empirical-Bayes hyperparameters

The offline prior (`paiec/prior.py`) fits one item-level model per public benchmark,

$$
\operatorname{logit} \Pr(y_{si} = 1) = \mu_b + t_s - g_i - e_i
$$

where $y_{si}$ is the outcome of subject $s$'s first recorded response to item $i$ and $t_s$ is the subject's standing on that benchmark. Each fit uses every subject's first recorded response per item (the one acquisition reveals) and estimates the variances by Laplace-EM. These fits supply the standings $t_s$. An attribute ridge predicts them from provider, release date (linear in days since 2023), model size parsed from the name, reasoning-effort dummies and a flag for whether the subject's harness field is filled (`paiec/subjects.py`). The same fits supply the levels, the item variance and the group share, which `fit_hyper` pools across benchmarks; App B.1 gives the fallbacks.

Every experiment refits the prior and the hyperparameters *without the target's parent benchmark* (§4.2). The shipped bundle, `submission/prior.json`, is fitted on all five public benchmarks with the level prior from `LEVEL` (F§ "Verdict: ship hier with the level moved down"). By their code names, the columns of the table below are `mu0`, `sigma_mu`, `attr_scale`, `sigma_theta`, `sigma_delta`, `sigma_attr`, `sigma_d`, `sigma_g`, `slip` and `guess`. In it, $\sigma_\theta$ is the sd of the identity link: the part of a subject's standing, beyond its attributes, that its benchmarks share.

| $\mu_0$ | $\sigma_\mu$ | $\gamma$ | $\sigma_\theta$ | $\sigma_\delta$ | $\sigma_{\mathrm{attr}}$ | $\sigma_d$ | $\sigma_g$ | $\varepsilon$ | $\omega$ |
|---|---|---|---|---|---|---|---|---|---|
| **-2.5** | **2.5** | **0.5** | 0.1 | 2.382 | 1.018 | 2.671 | 1.542 | 0.01 | 0.5 |

The three bold fields are LEVEL, the level prior we set for the hidden test (§3.4); the rest are the empirical-Bayes fit [Morris 1983].

On current evidence, a subject's identity across benchmarks carries almost nothing beyond its attributes. We estimate the shared part of a named model's attribute residual across benchmarks, $\tau^2_{\mathrm{res}}$ (`tau2_res`), at -0.002 and clip it to [0.01, 0.2]. That puts $\sigma_\theta$ at its floor of 0.1. It also puts the link weight, how far a subject's standing on one benchmark moves its prediction on another, at 0.0018 (`paiec/hier.py` docstring). The estimator is biased downward, because the ridge that predicts a name's standing on one benchmark saw the same name's standings on the others. Until the estimator is corrected and re-measured, the identity conclusion and the small link weight are provisional (F§ "The step-2 analyses behind the model").

### 3.4 Calibrating the level prior for the hidden test

**The signal.** The first formative feedback showed LegacyP, our first submission's predictor (§2.1), confidently optimistic at budget 0: it predicted near 0.75 on pairs whose rates were low (§5.1). When we chose the level, two readings pointed the same way. First, through the lower roots of $p(1-p) = B_{31}$, the feedback's pairs sit about 0.9 logit below the public centre (the mean logit of the pair rates on public runs). Second, *if* the organisers' leaderboard entry is the empirical-mean baseline, inverting its ALC puts $\mathbb E[p(1-p)]$ near 0.128, against 0.181 on public runs (F§ "Against the live leaderboard"). Later readings are more central (§5.4).

**The regime.** One run of nine pairs identifies little, so we built test-like (TL) runs from public data that resemble the feedback (§4.1) and chose the level prior on them.

**The candidates.** We scored grids over $(\mu_0, \sigma_\mu, \gamma)$ for hier, with Gaussian and Student-t levels. We also scored an empirical-Bayes re-estimate of the level at every checkpoint, and a level fix for LegacyP (F§ "Calibrating for the hidden test").

**The selection.** Our selection rule considers only configurations that lose at most 0.003 ALC against LegacyP on *both* public weightings, benchmark-first and pair-uniform (the guard). Among them it takes the one with the best mean ALC on the selection half (TL runs 0 to 99), and it confirms that choice on runs 100 to 199 (§5.4).

**Limits on the feedback.** The feedback may set *global* hyperparameters of the level distribution, here three numbers: the level prior's centre $\mu_0$ and sd $\sigma_\mu$, and the attribute scale $\gamma$. It never sets anything keyed on an anonymous benchmark or subject id, and we shaped no prediction to probe hidden labels. Both limits follow the competition's conduct rule against extracting test data and identifying anonymous ids. Selection on the level grid used nothing from the feedback beyond the regime's defaults. Two later uses read the feedback per pair, still for global hyperparameters only (§5.4; App G.6 lists every use).

### 3.5 Run-time engineering

The entry point `predict` is a pure function of `(input, labeled)`. Fits are cached under a content fingerprint of `labeled`, and no call raises an exception: a failed fit falls back to the prediction without labels, and then to 0.5. Only if every check passes does the build write the numpy-only archive; the checks include the organisers' validator and a bit-for-bit match with the in-repository predictor (App B.2).

---

## 4 Evaluation methodology

### 4.1 Runs

| regime | what it is | used for |
|---|---|---|
| R1 (R1-bf, R1-pu) | public formative-like runs (`official.sample_run`), seed 0: 5 to 12 pairs, cut to the 1,000-entry cap. Two weightings: benchmark-first (bf; a run holds 8.6 pairs over 4.4 benchmarks) and pair-uniform (pu; 3.4 benchmarks) | baselines; the public guard |
| R2 | dense runs (`official.dense_run`): every eligible pair of one benchmark, whole | the most cross-subject evidence a run can carry |
| TL | test-like runs (`testlike.Regime()` at its defaults), seed 2; see below | the primary regime for hidden-test choices |
| TL-1.2, TL-2.0, TL-mix, TL-noshift | sensitivities, seed 3: level_mean (the target level mean) -1.2 and -2.0 instead of -1.6; item groups merged at random, with no strata, and whole parent benchmarks; no date shift | robustness of every hidden-test choice |
| LA-0.8, LA-flat | the level audit's two extra regimes, seed 5: target level mean -0.8, and no level tilt | the audit (§5.4, App E.2) |
| RS regimes | the test-like regimes of the regime-sensitivity (RS) study, seed 11, 80 runs each: TUNED (the defaults); READING (level_mean -0.85, level_sd 1.75); AUDIT (-1.8, 2.1); MIXTURE (a two-component level target, `Regime.level_mix`); FLAT (no level tilt). R1 at seed 11, 60 runs per weighting. The RS study set the knobs on seed 10, from run composition alone | hier-ship at the feedback's readings, under a rule fixed in advance (§5.4); the baseline study (§5.5) |

Sources: F§ "Under the official protocol", "Hierarchical model", "Calibrating for the hidden test", "What actually shipped, after the audit", "Regime sensitivity at the feedback's reading". Realised levels and the run sets of later studies: App C.1.

We build test-like runs (`paiec/testlike.py`) from real recorded responses only, in four steps (App C.1 gives the full construction):

- We cut the public benchmarks into *pseudo-benchmarks* and give each its own anonymous benchmark_id. A pseudo-benchmark's *parent* is the public benchmark it was cut from. The pseudo-benchmarks are merged item_features groups, cross-fitted difficulty strata, whole parents, and random chunks of swe_rebench, the single-subject benchmark.
- Each run draws 5 to 12 pairs, about one pair per pseudo-benchmark, as in the formative feedback.
- The *level tilt* pulls the draw of pairs toward a target distribution of pair accuracy logits, with mean -1.6 (level_mean) and sd 1.5 (level_sd). The 300 TL runs that check the regime realise a mean pair logit of -1.29, with sd 1.70 (cluster SE 0.16; `results/testlike_check.json`).
- Visible release and access dates move 1.25 years later. For LegacyP's prior, whose date term is linear, this *date shift* reproduces the optimism the feedback showed at budget 0, when LegacyP predicted high success on pairs whose rates were low.

At the budgets not used in tuning, $B_3$ to $B_{31}$, the feedback's $z$ is +0.09 to +0.50 relative to LegacyP's mean on TL runs, in units of the replica's single-run sd at each budget. By contrast, against LegacyP's mean on public R1 the feedback's $B_0$ is at $z = +8.5$. Agreement at $B_0$ and $B_1$ is in sample: we chose the regime's defaults from a grid of knob settings as the point closest to the feedback at those two budgets. ALC alone does not separate the regimes: LegacyP scores 0.2073 on TL runs, 0.2075 on R1 and 0.2113 in the feedback (`results/testlike_check.json`).

Pseudo-benchmarks carry less item structure than public benchmarks. On TL runs the item oracle (an in-sample Rasch probability per item) has a Brier 46% lower than the pair-rate oracle's, against 55% lower on public runs (F§ "Item signal from the pair's own labels"). For that reason, item-aware comparisons must also hold on TL-mix, which drops the difficulty strata, and on R1.

Base ALCs, the absolute scores behind a table's differences, differ between tables because the run sets and library versions differ (App H.2 gives hier-ship's on each), so every table names its runs and seed.

### 4.2 Leave-one-parent-out

For every target, we fit the subject prior and every empirical-Bayes hyperparameter without the target's parent benchmark (`prior.build(pairs, (parent,))`). On public runs the parent is the benchmark itself. The experiments' model factory selects each target's fit by its anonymous benchmark_id, so `predict` never sees a benchmark name. A stricter variant, strict run-LOBO, fits without every benchmark of the run (App E.1).

When a study chooses a hyperparameter, such as a covariate's slope or a layer's configuration, we select it by nested leave-one-parent-out:

1. For each held-out parent, choose the configuration, or "off", on the appearances of the other parents' pairs (an appearance is one pair in one run).
2. Score that choice on the held-out parent.
3. Redo the selection inside every bootstrap resample.

### 4.3 Statistics

All comparisons are paired on the same runs, checkpoints and acquisitions. A run's ALC is the mean over its pairs. A difference $\Delta = \mathrm{ALC}(A) - \mathrm{ALC}(B)$ is negative when $A$ is better. The full format is "$\Delta$ ± a / b / c (run / cluster / stratified SE)":

- The *run SE* is the standard error across runs. It overstates precision, because runs redraw the same pairs.
- The *cluster SE* comes from a pair-cluster bootstrap (2,000 resamples, ratio estimator). Its clusters are (benchmark, subject) on public runs and (parent, subject) on test-like runs.
- The *stratified SE* is the same bootstrap, resampled within each parent.

**In the main text, "±" and an SE in parentheses after a difference mean the cluster SE, unless marked otherwise.** The full triple appears only in the confirmation table of §5.4 and in the appendices.

None of these SEs covers variation between benchmarks, so we also report per-parent and leave-one-parent-out ranges, and the parent-level mean ± SE over the four multi-subject parents. Other spreads are named where they occur. A *single-run sd* is the spread of one run's ALC, or of one run's $B_k$, over a regime's runs. It describes the noise of one run and is not an SE. Two numbers in square brackets give a 95% interval. Many options compared against one default face a Bonferroni bar: 2.95 SEs, two-sided, for sixteen comparisons (F§ "Ablations and sensitivities").

### 4.4 Selection discipline

We choose candidates on the selection runs, TL runs 0 to 99, and confirm them on the confirmation runs, 100 to 199 (§3.4). The halves redraw runs from one catalogue of pairs, so they measure the noise of redrawing; they do not test new parents. The held-out runs, 200 to 299, were never used to choose the level.

Every hidden-test choice must pass the guard, a cap on its loss to LegacyP, on both public weightings.

A component ships only if all four conditions of the gate hold (F§ "Subject side at budgets 0 and 1"):

1. Nested selection switches it on in at least 3 of the 4 folds, one per held-out parent.
2. The nested test-like difference satisfies $\Delta \le -0.002$.
3. No held-out parent has $\Delta > +0.002$.
4. Neither public weighting loses more than 0.001 ALC.

We describe the gate as fixed in the studies' plans rather than pre-registered. App C.3 records, study by study, when each study fixed its gate or decision rule.

---

## 5 Results

### 5.1 Formative feedback

The platform has scored three of our formative runs, each one noisy draw of its own pairs. Besides Brier by budget, it reports expected calibration error (ECE) [Naeini et al. 2015]. The last column weights ECE over budgets as ALC weights Brier, then averages it over pairs.

| run | model; archive (sha256) | pairs (benchmarks, subjects) | $B_0$ | $B_1$ | $B_3$ | $B_7$ | $B_{15}$ | $B_{31}$ | ALC | mean ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LegacyP, commit b68492c; archive-1 `8e28d930…` | 9 (7, 9) | 0.3589 | 0.2539 | 0.2043 | 0.1712 | 0.1693 | 0.1568 | 0.2113 | 0.170 |
| 2 | hier-ship, commit ee5085a (run-2 library: old multiple-choice floor, solver before the floored-fit fix); archive-2 `2c64eaad…` | 8 (7, 8) | 0.2368 | 0.1947 | 0.1963 | 0.1836 | 0.1785 | 0.1833 | 0.1926 | 0.100 |
| 3 | hier-ship, commit 4d2cc4f (current library: corrected floor, floored-fit fix); archive-3 `4a882cc7…` | 9 (7, 9) | 0.2351 | 0.1868 | 0.1885 | 0.1664 | 0.1667 | 0.1644 | 0.1817 (recomputed) | 0.118 |

Sources: F§ "Formative feedback, runs 1 and 2", "Formative run 3"; the organisers' tables, byte for byte, in `results/formative/`; per pair: App D.1.

We uploaded run 3, the rebuilt archive, as a regression and latency check only, and stated that purpose before it was scored. We recomputed its ALC from its table (App D.1).

Formative runs on different draws do not measure an improvement. The difference between two independent runs' ALC has an sd of about 0.04 (0.037 to 0.047 for hier-ship), against observed gaps of 0.019 and 0.011 between consecutive runs.

Run 1's loss came at budget 0. It scored 0.3589 there, where a constant 0.5 scores 0.25, with ECE as high as 0.75. Answering 0.5 at $B_0$ and $B_1$ alone would have given an ALC of about 0.1996.

LegacyP's attribute prior put strong 2025-26 subjects near a predicted probability of 0.75. The pairs' rates were 0.006 to 0.45, each read as the lower root of $p(1-p) = B_{31}$, the smaller of its two solutions $p$. Their mean on the plain logit is -1.6 (sd 1.5), against -0.74 (sd 1.50) for the public R1 appearances (`experiments/hier_design/levels.json`). That run motivated the level calibration (§3.4, §5.4).

Runs 2 and 3, with hier-ship's level prior, lost far less at budget 0, on different pairs. Their $B_0$ was 0.237 and 0.235, against run 1's 0.359 (App D.2 places them budget by budget).

Neither run discriminates the regimes. Placed in hier-ship's single-run distributions, both lie within 1.02 single-run sds (0.026 to 0.034 ALC) of every regime's mean (Figure 1). In the table, a run's $z$ is its signed distance from a regime's mean ALC in single-run sds, positive when the run scored worse than the mean.

| regime (runs) | mean ALC | single-run sd | run 2's $z$ | run 3's $z$ |
|---|---|---|---|---|
| TL (300) | 0.1658 | 0.0318 | +0.84 | +0.50 |
| TL-mix (200) | 0.1711 | 0.0329 | +0.65 | +0.32 |
| TL-noshift (100) | 0.1585 | 0.0335 | +1.02 | +0.69 |
| R1-bf (150) | 0.2051 | 0.0285 | -0.44 | -0.82 |
| R1-pu (100) | 0.2020 | 0.0261 | -0.36 | -0.78 |

Source: `results/ship_confirm.json` (`run2_placement`); run 3 from `results/formative_run3.json` (`placement`). Run-2 library; seeds TL 2, TL-mix and TL-noshift 3, R1 0.

Run 2's profile is at least as close to plain public runs as to the TL regime. Every budget lies within 0.8 sd of public R1, while its $B_{31}$ is 1.4 sd above the TL mean.

![Figure 1: Brier by budget, by regime, with the formative runs](fig/learning_curves.png)

*Figure 1.* Brier by budget for hier-ship ("shipped hier" in the legend; the band is ±1 single-run sd), LegacyP ("legacy Predictor") and Smooth in five replica regimes: (a) TL, (b) TL-mix, (c) TL-noshift, (d) R1-bf, (e) R1-pu. Every panel overlays the pair means of the three formative runs. These are three draws with different subjects, so the gaps between them are not measured effects. The hier-ship curves use the run-2 library.

For comparison, the live leaderboard of 2026-09-24 had the organisers' entry at 0.1801 and the best entry at 0.1172, both single draws (F§ "Against the live leaderboard").

### 5.2 Public formative-like runs: baselines

We scored the baselines on 600 public formative-like runs with benchmark-first weighting (R1-bf; seed 0, split scope 'pair'; F§ "Under the official protocol", `experiments/official_baselines.py`).

| predictor | $B_0$ | $B_1$ | $B_3$ | $B_7$ | $B_{15}$ | $B_{31}$ | ALC ± run SE |
|---|---|---|---|---|---|---|---|
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| EmpMean | 0.2500 | 0.3746 | 0.2572 | 0.2131 | 0.1974 | 0.1913 | 0.2526 ± 0.0014 |
| Smooth | 0.2500 | 0.2349 | 0.2220 | 0.2059 | 0.1957 | 0.1908 | 0.2158 ± 0.0008 |
| LegacyP, LOPO prior | 0.2313 | 0.2238 | 0.2162 | 0.2034 | 0.1944 | 0.1832 | 0.2090 ± 0.0009 |
| base-rate oracle, $\mathbb E[p(1-p)]$ | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 | 0.1809 ± 0.0010 |

EmpMean is no better than answering 0.5, because one label sends it to 0 or 1 (0.3746 at $B_1$). LegacyP beats Smooth by only 0.0026 ± 0.0009 with a strict run-LOBO prior, or by 0.0069 ± 0.0016 with a LOPO prior.

Against LegacyP, hier-fit gains 0.0024 (0.0013). Most of that gain comes from pooling a benchmark's level across the subjects a run holds on it, and a target alone on its benchmark does not get it (App E.1). In hier-fit, the attribute prior, the feature groups and the pooled level each contribute (App B.3). Our first verdict, on that evidence, was to keep LegacyP (F§ "Verdict: keep the Predictor"). The test-like runs of §5.3 reversed it.

### 5.3 Test-like runs: moving the level

With its level moved and no constraint on the public cost, every family reaches the same plateau on the selection half (TL runs 0 to 99, seed 2; F§ "What moving the level buys, unconstrained"). In the table, each family whose level was moved appears with its best configuration, and LegacyP, hier-fit and Smooth appear unmoved. For hier, a configuration sets the level prior's centre $\mu_0$, its sd $\sigma_\mu$ and the attribute scale $\gamma$.

| family | best configuration | ALC | $\Delta = \mathrm{ALC}(\text{family}) - \mathrm{ALC}(\text{LegacyP})$ (cluster SE) | public cost, R1-bf / R1-pu (SE not shown) |
|---|---|---|---|---|
| LegacyP (first submission) | | 0.2039 | | |
| hier-fit | | 0.1906 | -0.0134 (0.0017) | -0.0019 / -0.0017 |
| Smooth | | 0.1788 | -0.0252 (0.0021) | +0.0070 / +0.0075 |
| hier, Gaussian level | $\mu_0 = -4$, $\sigma_\mu = 2.5$, $\gamma = 0.5$ | 0.1572 | -0.0467 (0.0042) | +0.0036 / +0.0032 |
| hier, Student-t level | $\mu_0 = -4$, scale 1.8, $\gamma = 0.5$ | 0.1570 | -0.0469 (0.0042) | not scored |
| hier-EB | on the Gaussian above | 0.1567 | -0.0473 (0.0042) | +0.0034 / +0.0033 |
| LegacyP+fix | offset -2.5, scale 1, variance 2 | 0.1562 | -0.0477 (0.0045) | +0.0329 / +0.0329 |
| Smooth-cal | mean $m_0 = 0.25$, strength $n_0 = 2$ | 0.1588 | -0.0451 (derived, no SE) | +0.0077 / +0.0096 |

Source: `experiments/level_calibration.py`; run and stratified SEs: the same F§ section. The Smooth-cal row was added by the RS study on the same runs (App E.4).

The plateau lies at about 0.156 to 0.157 and is flat, with 14 of 108 Gaussian configurations within 0.001 of the best. Here LegacyP loses to Smooth, the $\mathrm{Beta}(2, 2)$-smoothed mean, and the loss comes from its prior. Smooth-cal, a smoothed mean with a calibrated prior and no model at all, reaches 0.159 (§5.5 asks how much of the gain to call calibration).

The widest level prior always won. At every shift of the level centre, $\sigma_\mu = 2.5$ beat 1.8, 1.3 and 0.9 in every cell of both grids (Figure 2; §5.4 scores wider priors).

The public guard, a cap on the loss to LegacyP on both public weightings, decides which configuration is chosen. Moving the level costs LegacyP far more on public runs than it costs hier, which gives back most of its $B_0$ and $B_1$ losses from $B_7$ on.

![Figure 2: the level-calibration surface](fig/level_surface.png)

*Figure 2.* (a) Selection-half ALC of hier over $\mu_0$ and $\gamma$ at $\sigma_\mu = 2.5$, with each cell's public guard cost. The cost comes from 100 runs per weighting where scored (hier-ship's cell from the level audit on the same runs), otherwise from the level calibration's 40-run screen; hatched cells fail the guard. Marked, with their coordinates in that order and their ALC: hier-argmax (-3.5, 0.25; 0.1574), hier-rec (-3.0, 0.25; 0.1578; "neighbouring config" in the legend) and hier-ship (-2.5, 0.5; 0.1608). (b) In every scored cell, a level prior sd $\sigma_\mu$ of 0.9, 1.3 or 1.8 loses to 2.5, the edge of the grid; Figure 4 shows the wider priors.

### 5.4 How hier-ship was chosen and tested

**(a) Chosen.** The selection rule picked hier-argmax ($\mu_0 = -3.5$, $\gamma = 0.25$). The level calibration broke the tie by the guard's margin and recommended hier-rec ($\mu_0 = -3.0$), which was 0.0004 behind on the selection half and tied on the confirmation half (F§ "The public guard decides").

| configuration; for hier, $(\mu_0, \sigma_\mu, \gamma)$ | selection half, ALC | confirmation half, ALC ± run SE | $\Delta$ LegacyP, confirmation (cluster SE) | R1-bf runs 0 to 99, $\Delta$ LegacyP (SE not shown) | R1-pu runs 0 to 99, $\Delta$ LegacyP (SE not shown) |
|---|---|---|---|---|---|
| hier-ship (-2.5, 2.5, 0.5) | 0.1608 | 0.1677 ± 0.0028 | -0.0396 (0.0037) | -0.0015 | -0.0018 |
| hier-rec (-3.0, 2.5, 0.25; recommended, not shipped) | 0.1578 | 0.1655 ± 0.0031 | -0.0418 (0.0047) | +0.0008 | +0.0007 |
| hier-argmax (-3.5, 2.5, 0.25) | 0.1574 | 0.1655 ± 0.0033 | -0.0418 (0.0051) | +0.0025 | +0.0023 |
| hier-EB (EB-cs, tau 2) on $\mu_0 = -3.0$, $\gamma = 0.5$ | 0.1580 | 0.1658 ± 0.0031 | -0.0416 (0.0043) | +0.0005 | +0.0006 |
| LegacyP+fix (offset -1.5, scale 1, variance 3) | 0.1630 | 0.1696 ± 0.0027 | -0.0377 (0.0032) | +0.0026 | +0.0028 |
| hier-fit | 0.1906 | 0.1949 ± 0.0024 | -0.0124 (0.0019) | -0.0019 | -0.0017 |
| LegacyP | 0.2039 | 0.2073 ± 0.0022 | | ¹ | ¹ |

¹ LegacyP's own ALCs: 0.2068 and 0.2038. In the hier-EB row, EB-cs re-estimates both the centre and the sd of the level prior from the revealed labels at every checkpoint, and tau is the sd of the hyperprior on the centre. Paired, hier-ship minus hier-rec is +0.0030 (0.0009) and +0.0022 (0.0011) on the two TL halves, and -0.0023 (0.0005) and -0.0024 (0.0004) on R1-bf and R1-pu runs 0 to 99 (other SEs: App E.2). Sources: `experiments/level_calibration.py`; hier-ship's row from `results/level_audit.json` (`mild`) and `results/ship_confirm.json`.

**(b) Audited.** The level audit then moved the choice to hier-ship, the milder guarded configuration, for three reasons (F§ "What actually shipped, after the audit"; `results/level_audit.json`):

1. The hidden levels look spread both ways. When we match run 1's nine pairs against replica pairs, two of them, both on one benchmark, are more plausibly high-rate, with 70% and 80% of their nearest replica pairs at a rate above 0.5. A third pair is ambiguous. Read that way, the nine pairs average -0.98 (sd 2.05) on the plain logit, instead of the -1.6 of the lower-root reading. Against LegacyP, hier-rec also loses 0.014 on TL pairs at rates 0.5 to 0.7, 0.070 at rates of 0.7 and above, and 0.015 to 0.017 on public matharena.
2. Both configurations sit on a flat plateau (the table's note), and hier-ship's measured worst case is no worse.
3. The confirmation half repeats 82.2% of the selection half's pairs and 99.5% of its clusters, so it measures redrawing and does not test unseen parents.

**(c) Confirmed.** We then scored hier-ship against LegacyP and Smooth on identical runs, including TL runs 200 to 299, which no choice of the level used. We report each difference with the full SE triple, the parent-level mean ± SE over the four multi-subject parents, and the range over parents (F§ "Shipped configuration, confirmed"; `experiments/ship_confirm.py`, `results/ship_confirm.json`).

| regime, runs | hier-ship ALC | $\Delta$ LegacyP (run / cluster / stratified SE) | parent-level | parents (range) | $\Delta$ Smooth (run / cluster / stratified SE) |
|---|---|---|---|---|---|
| TL, 0 to 99 (selection half) | 0.1608 | -0.0431 ± 0.0019 / 0.0032 / 0.0028 | -0.038 ± 0.009 | -0.059 to -0.014 | -0.0180 ± 0.0010 / 0.0021 / 0.0020 |
| TL, 100 to 199 (confirmation half) | 0.1677 | -0.0396 ± 0.0019 / 0.0037 / 0.0034 | -0.035 ± 0.011 | -0.058 to -0.007 | -0.0159 ± 0.0009 / 0.0021 / 0.0020 |
| TL, 200 to 299 (never used to choose the level) | 0.1687 | -0.0419 ± 0.0017 / 0.0037 / 0.0034 | -0.038 ± 0.008 | -0.057 to -0.021 | not stored |
| TL, 0 to 299 | 0.1658 | -0.0415 ± 0.0011 / 0.0034 / 0.0031 | -0.037 ± 0.009 | -0.058 to -0.014 | -0.0169 ± 0.0007 / 0.0020 / 0.0019 (0 to 199) |
| TL-mix, 0 to 99 | 0.1694 | -0.0436 ± 0.0014 / 0.0039 / 0.0031 | -0.029 ± 0.012 | -0.065 to -0.010 | -0.0176 ± 0.0008 / 0.0022 / 0.0019 |
| TL-noshift, 0 to 99 | 0.1585 | -0.0144 ± 0.0012 / 0.0024 / 0.0022 | -0.012 ± 0.005 | -0.024 to -0.000 | -0.0220 ± 0.0014 / 0.0030 / 0.0028 |
| R1-bf, 0 to 149 | 0.2051 | -0.0017 ± 0.0008 / 0.0021 / 0.0018 | -0.003 ± 0.005 | -0.013 to +0.010 | -0.0086 ± 0.0010 / 0.0030 / 0.0021 |
| R1-pu, 0 to 99 | 0.2020 | -0.0018 ± 0.0010 / 0.0017 / 0.0016 | -0.002 ± 0.005 | -0.015 to +0.009 | -0.0093 ± 0.0011 / 0.0021 / 0.0019 |

Seeds: TL 2, TL-mix and TL-noshift 3, R1 0; run-2 library (App H.2).

On the confirmation half and the held-out runs, the gain over LegacyP is 0.040 to 0.042. Across all 300 TL runs the gain is 0.042, and the first two budgets, $B_0$ and $B_1$, carry 0.029 of it, with Brier gains of 0.135 and 0.076 there (Figure 3). Of the 300 runs, hier-ship beats LegacyP in 296, and leaving out one parent at a time over them gives differences of -0.047 to -0.032.

The gain is less certain between benchmarks. The parent-level SE (0.009) is about 2.7 times the cluster SE. Among the parents, matharena gains least (-0.014 ± 0.009) and multi_swebench most (-0.058 ± 0.004), with per-parent cluster SEs.

Against LegacyP on public runs, hier-ship is within noise: 0.0015 to 0.0018 better, or 0.81 and 1.02 cluster SEs. It loses at $B_0$ and on public matharena (+0.010 and +0.009 on the two weightings) and gains from $B_7$ on. Against Smooth, which has no level to tune, hier-ship gains 0.016 to 0.018 on TL runs and 0.009 on public runs. Without the date shift, which moves the test-like runs' visible release and access dates later, its gain over Smooth is 0.022 (TL-noshift).

![Figure 3: what the level fix trades, budget by budget](fig/level_fix_budgets.png)

*Figure 3.* Paired Brier difference against LegacyP ("legacy Predictor") by budget. (a) hier-ship ("shipped hier") in five regimes, with ±1 cluster SE. On TL runs it gains 0.135 at $B_0$ and 0.076 at $B_1$ (0.040 and 0.023 on TL-noshift). On public runs it costs 0.005 and 0.014 at $B_0$ (cluster SEs about 0.007), is on par at $B_1$ and $B_3$, and gains 0.004 to 0.008 per budget from $B_7$ on. (b) hier-ship, hier-rec ("neighbouring config") and hier-fit on the level calibration's runs, without SEs (none is stored per budget for hier-rec and hier-fit).

**(d) After run 2: the pooled reading.** We read each pair of runs 1 and 2 against matched replica pairs and pooled the pairs into one reading, under a decision rule. The rule was hash-locked before the reading was computed, but after both runs' tables had been seen (App C.3). It could change only $\mu_0$ and $\sigma_\mu$, at a bar of 2 SEs, with SEs that resample the 7 benchmarks. The 17 pairs' level is -0.65 with 15 nearest neighbours per pair and -0.78 with 40 (SE 0.51). These are 1.24 and 1.00 SEs above the TL regime's realised -1.29, so the rule gives no candidate, and LEVEL stays.

The hidden level looks near the public centre (the mean logit of the pair rates on public runs) and more central than the TL regime in which the headline gain was measured. Its distance from that regime is within the noise of 7 benchmarks. We do not read run 3 (App D.4 gives the rule, the table and the checks).

**(e) Tested at the feedback's readings.** The regime-sensitivity (RS) study scored ten alternatives against hier-ship on identical runs. The runs cover both public weightings and the five RS regimes, test-like regimes that differ in their level target (§4.1). TUNED keeps the TL regime's defaults, and READING and AUDIT are set to two later readings of the feedback's level. The study's seed, 11, is fresh, but its runs redraw the catalogue of test-like pairs and the public pairs that chose the level. TUNED and the public guard are therefore not an independent replication (F§ "Regime sensitivity at the feedback's reading"; `results/regime_sensitivity.json`).

To replace hier-ship, an alternative had to gain at least 0.002 in READING and AUDIT, with the upper end of its 95% interval, $U_{95}$, below zero. It also had to lose at most 0.002 in the other test-like regimes and at most 0.001 on public runs. App C.2 gives all five conditions, including the parent-level and time-per-call ones.

| alternative minus hier-ship, ALC (cluster SE) | TUNED (-1.28) | READING (-0.63) | AUDIT (-0.93) | MIXTURE (-0.64) | FLAT (-0.28) | R1-bf | R1-pu |
|---|---|---|---|---|---|---|---|
| hier-ship, ALC | 0.1690 | 0.1881 | 0.1650 | 0.1830 | 0.1739 | 0.2114 | 0.1948 |
| hier-rec | -0.0034 (0.0010) | +0.0015 (0.0011) | +0.0005 (0.0014) | +0.0017 (0.0012) | +0.0033 (0.0014) | +0.0025 (0.0004) | +0.0020 (0.0005) |
| hier-ship σ3.5 | +0.0009 (0.0003) | +0.0006 (0.0003) | -0.0003 (0.0004) | +0.0005 (0.0004) | -0.0010 (0.0005) | +0.0007 (0.0003) | +0.0008 (0.0003) |
| hier-ship σ5 | +0.0031 (0.0007) | +0.0023 (0.0007) | +0.0006 (0.0008) | +0.0022 (0.0008) | -0.0007 (0.0009) | +0.0026 (0.0007) | +0.0025 (0.0007) |
| hier-EB on hier-ship | -0.0014 (0.0004) | +0.0003 (0.0003) | -0.0002 (0.0004) | -0.0001 (0.0003) | -0.0001 (0.0004) | +0.0006 (0.0003) | +0.0014 (0.0003) |
| hier-EB on $\mu_0 = -3.0$, $\gamma = 0.5$ | -0.0027 (0.0006) | +0.0003 (0.0004) | -0.0005 (0.0007) | -0.0001 (0.0004) | +0.0003 (0.0004) | +0.0017 (0.0002) | +0.0020 (0.0002) |
| hier-fit | +0.0309 (0.0035) | +0.0173 (0.0032) | +0.0233 (0.0042) | +0.0164 (0.0028) | +0.0140 (0.0038) | -0.0010 (0.0010) | +0.0009 (0.0011) |
| hier-nosubj | -0.0030 (0.0012) | +0.0026 (0.0014) | +0.0016 (0.0017) | +0.0031 (0.0014) | +0.0049 (0.0017) | +0.0025 (0.0005) | +0.0027 (0.0004) |
| Smooth-cal | -0.0015 (0.0019) | +0.0078 (0.0021) | +0.0043 (0.0024) | +0.0071 (0.0021) | +0.0103 (0.0024) | +0.0108 (0.0014) | +0.0100 (0.0018) |
| Smooth | +0.0170 (0.0022) | +0.0096 (0.0018) | +0.0142 (0.0024) | +0.0110 (0.0020) | +0.0103 (0.0018) | +0.0075 (0.0029) | +0.0127 (0.0021) |
| LegacyP | +0.0434 (0.0037) | +0.0248 (0.0037) | +0.0313 (0.0049) | +0.0263 (0.0037) | +0.0200 (0.0044) | +0.0002 (0.0020) | +0.0039 (0.0019) |

Source: `experiments/regime_sensitivity.py`. Realised mean pair logit in brackets; 80 runs per test-like regime and 60 per public weighting, seed 11, library of `4d2cc4f`. Positive: hier-ship is better. Run and stratified SEs, $U_{95}$, per-parent and per-budget differences: F§ "Regime sensitivity at the feedback's reading"; results in full: App E.2.

![Figure 4: hier-ship at the feedback's readings](fig/regime_sensitivity.png)

*Figure 4.* Each alternative minus hier-ship ("shipped hier") in the seven RS regimes, with the test-like ones ordered by realised level. Bars show ±1 and ±1.96 cluster SEs, and the right end of the thin bar is the rule's $U_{95}$; a hollow marker shows the parent-level mean. The dashed line marks the gain the rule required in READING and AUDIT (-0.002), the dotted lines its loss bounds. The legend's "hier, no subject prior" is hier-nosubj; it is not the plain 1PL of §5.5. No alternative crosses -0.002 in READING or AUDIT.

No configuration passes the rule, so LEVEL stays (Figure 4). We compared one reproduction check at its comparator's stored precision rather than at the planned precision, and we accepted this deviation (App C.3). With or without it, no alternative is a candidate.

The closest alternatives are hier-ship σ3.5 and both hier-EB, within 0.0006 of hier-ship in READING and AUDIT (cluster SEs 0.0003 to 0.0007). Only in TUNED, the regime it was chosen in, does hier-rec beat hier-ship, with a difference of -0.0034 (0.0010). Elsewhere it loses by +0.0005 to +0.0033, and the higher the level, the more it loses.

The gain over LegacyP shrinks as the hidden level rises. It is 0.043 in TUNED, 0.031 in AUDIT, 0.026 in MIXTURE, 0.025 in READING and 0.020 in FLAT, the same pattern as in the level audit's LA-0.8 and LA-flat (App E.2).

Each of these comparisons rests on four parents. Every alternative with a lower level centre than hier-ship gains on multi_swebench, the lowest-level parent, and loses on matharena, the highest. The parent-level SE is two to four times the cluster SE. The rule detects gains of about 0.003 or more, and at 0.002 it detects a gain about half the time.

### 5.5 Calibration against modelling

The baseline study rescored the RS study's TUNED and public runs with EmpMean and a plain 1PL (F§ "Baselines: a plain 1PL, the organisers' empirical mean and BLE"; `results/baselines_p1.json`; App E.4). The plain model is hier with the attributes, identity link, groups, floor and slip switched off. It ran at hier-ship's level (1PL-ship) and at its own leave-one-parent-out public level (1PL-fit).

| ALC; TUNED and R1, seed 11 (the RS runs) | TUNED | R1-bf | R1-pu |
|---|---|---|---|
| hier-ship | 0.1690 | 0.2114 | 0.1948 |
| 1PL-ship | 0.1673 | 0.2162 | 0.1998 |
| Smooth-cal | 0.1675 | 0.2222 | 0.2048 |
| 1PL-fit | 0.1743 | 0.2173 | 0.2027 |
| hier-fit | 0.1999 | 0.2104 | 0.1957 |
| EmpMean | 0.2029 | 0.2578 | 0.2397 |
| LegacyP | 0.2124 | 0.2116 | 0.1988 |

We split hier-ship's TUNED gain over LegacyP, -0.0434 (0.0037), into the level calibration's share and the model's share for three orders of the steps, with 95% bootstrap intervals.

| order of the steps | the level calibration's share | the model's share |
|---|---|---|
| calibrate first: LegacyP $\to$ Smooth-cal $\to$ hier-ship | 1.03 [0.94, 1.10] | -0.03 [-0.10, 0.06] |
| model first: LegacyP $\to$ hier-fit $\to$ hier-ship | 0.71 [0.61, 0.81] | 0.29 [0.19, 0.39] |
| through the 1PL: LegacyP $\to$ 1PL-fit $\to$ 1PL-ship $\to$ hier-ship | 0.16 [0.12, 0.20] | 0.88 [0.84, 0.91] for 1PL-fit; -0.04 [-0.08, 0.02] for hier's extras |

An earlier version of our findings claimed that calibration is worth far more than the modelling. The split depends on the order of the steps, so it does not support that claim, and we withdraw it. Calibrating first gives the calibration the whole gain, and modelling first gives it 0.71. On the path through the 1PL, 1PL-fit takes 0.88 before any calibration.

In every order, the model adds nothing in TUNED once the level is calibrated. There, hier-ship minus Smooth-cal is +0.0015 (0.0019), and hier-ship minus 1PL-ship is +0.0017 (0.0013). Smooth-cal, with no item model and no subject prior, beats LegacyP by 0.045 in TUNED. It trails hier-ship by 0.004 to 0.010 in the four other test-like regimes and by 0.010 to 0.011 on public runs, where it also fails the guard (App E.4). 1PL-ship borrows hier's calibrated $\mu_0$, and TUNED redraws the catalogue of test-like pairs that chose LEVEL and Smooth-cal, so TUNED favours those configurations.

Most of LegacyP's deficit in TUNED is its own prior. 1PL-fit, with no calibration at all, beats LegacyP there by 0.038 (cluster SE 0.004), hier-fit beats it by 0.012, and even EmpMean beats it by 0.0095 (1.9 cluster SEs). LegacyP's mean prediction at $B_0$ in TUNED is 0.68, against 0.42 for hier-ship. Its attribute prior reads the synthetic date shift as higher success rates, and that optimism accounts for much of the headline gain in TUNED.

The level calibration costs ALC on public runs, and the model makes up for it. The difference hier-ship minus LegacyP is within noise there (-0.0002 and -0.0039). At the same level, hier adds 0.010 over Smooth-cal. Over 1PL-ship it adds 0.005, with differences of -0.0048 (0.0007) under benchmark-first weighting and -0.0050 (0.0007) under pair-uniform weighting. About half of that gain is the subject prior (hier-ship minus hier-nosubj), and half is the groups, floor, slip and name-keyed standing (hier-nosubj minus 1PL). At $B_0$ the attribute prior improves Brier once the level centre is matched (App E.4).

We could not run BLE, the organisers' language-model predictor. Each of its predictions is an agent run against a paid API, and these runs alone would need 569,838 of them (App E.5). The best leaderboard entry (0.1172) stays unexplained.

### 5.6 Where it loses, and what it costs

**The single-subject benchmark.** On swe_rebench, the public benchmark with one subject, hier-ship loses to LegacyP. Each appearance of that subject sees a different subset of the benchmark's 6,306 items, and nothing measures variation between subjects or between single-subject benchmarks (F§ "A single-subject benchmark"; `results/gate_and_ci.json`, `single`). Against LegacyP the loss is +0.0101 ± 0.0012 over 130 R1-bf appearances (run-2 library, seed 0) and +0.0094 ± 0.0016 over 51 (current library, seed 11), with SEs over appearances. Against Smooth it is +0.0130 ± 0.0014 and +0.0117 ± 0.0018. The loss sits at $B_0$ to $B_3$. Test-like runs exclude such benchmarks, and swe_rebench's rate (about 0.49) sits near the public centre (App E.6).

**The platform's composition.** Public runs reweighted to the platform's composition, in which most pairs are alone on their benchmark, put hier-ship within noise of LegacyP under per-pair splits (+0.0024 ± 0.0023). Under per-benchmark splits they put it behind (+0.0045 ± 0.0024). The cost is mostly the single-subject benchmark's. Of the replica's 193 pairs that are alone on their benchmark, 87 are swe_rebench. Without it the two differences become +0.0004 ± 0.0026 and +0.0017 ± 0.0022 (F§ "Pooling under the verified protocol"; App A.2).

**Cost.** Per evaluation call, hier-ship takes 0.80 to 1.04 ms in the subject-side study (run-2 library) and 1.34 ms in the RS study (current library; slowest call 0.56 s). The two figures come from separate measurements. The floored-fit fix itself changes the time per call by -0.2% to +0.2%, so it does not account for the gap between them. A call on dense matharena takes 7.5 to 9.4 ms. A formative run is about 3,000 calls, which take a few seconds against the 8-hour limit (App B.5).

---

## 6 What does not transfer

Every item-side idea reduces to a covariate, one number per item, and one harness scores each covariate against one gate (§6.1). We report what we tried in §6.2 and §6.3, with the detail in App F. Every idea we completed either does not transfer to a held-out benchmark or stays below the gate, except a blind rating that tracks difficulty on mathematics only. §6.4 gives two reasons for these failures.

### 6.1 The acceptance harness and its gate

The harness (`experiments/harness.py`) scores a covariate $x_i$ against hier-ship, fitted leave-one-parent-out (without the target's parent benchmark). It adds a centred offset to hier-ship's logit (F§ "Acceptance harness"):

$$
q_i = \operatorname{logit}^{-1}\!\Big(\operatorname{logit}\hat p_i + \operatorname{cap}\big(\beta\,(x_i - \bar x_{\mathcal L})\big)\Big),
\qquad \operatorname{cap}(o) = 4\tanh(o/4)
$$

Here $\hat p_i$ is hier-ship's prediction and $\bar x_{\mathcal L}$ is the mean of $x$ over the labelled items. Centring keeps hier-ship's level, so $x$ only orders items and does nothing at $B_0$. The slope $\beta$ is either *transferred* or *per-pair*. A transferred slope is one coefficient per budget fitted on the other parents. A per-pair slope is a MAP estimate from the target pair's own labels under a zero-mean prior with sd $\sigma_\beta$ per standard deviation of $x$.

We call each configuration that the harness scores a *line*. A *nested* line redoes the choice of configuration inside each held-out parent, and a *forced* line scores one fixed configuration without selection. The difference $\Delta$ is a line's ALC minus hier-ship's, so it is negative when the covariate helps.

The gate table below scores degraded oracles, $x_i = r\,\tilde d_i + \sqrt{1 - r^2}\,\xi_i$, with 8 noise draws for each $r$. Here $\tilde d_i$ is an *honest* difficulty: we split the subjects into folds and fit it only on the folds that do not hold the target subject. It is standardised within its parent benchmark, and the noise $\xi_i$ is standard normal. The covariate's correlation with $\tilde d_i$ is then $r$, which we call the honest $r$. The within-pair $r$ in parentheses is the same correlation measured within a test-like pair, and the honest oracle uses $\tilde d_i$ itself.

A real covariate is a single draw, so we read its chance of passing from the draws rather than from the averaged line. The transferred-slope column gives each line's cluster SE and its *draw sd*, the sd of $\Delta$ across the noise draws:

| honest $r$ (within-pair $r$) | transferred slope: nested TL $\Delta$ (cluster SE, draw sd) | draws passing (Jeffreys 95%) | chance one draw has $\Delta \le -0.002$, normal / t | modelled chance of passing the full gate | per-pair slope: nested TL $\Delta$ | draws passing |
|---|---|---|---|---|---|---|
| 0 (0.00) | -0.00002 (0.00001, 0.00006) | 0/8 | | | -0.00000 | 0/8 |
| 0.1 (0.08) | -0.00010 (0.00005, 0.00043) | 0/8 | | | -0.00000 | 0/8 |
| 0.2 (0.16) | -0.00083 (0.00015, 0.00086) | 1/8 [0.01, 0.45] | 0.09 / 0.12 | 0.09 | -0.00010 | 0/8 |
| 0.3 (0.25) | **-0.00255** (0.00032, 0.00068) | **6/8** [0.41, 0.94] | 0.79 / 0.76 | 0.68 | -0.00032 | 0/8 |
| 0.4 (0.33) | -0.00462 (0.00052, 0.00081) | 7/8 [0.55, 0.99] | 1.00 / 0.99 | 0.87 | -0.00136 | 0/8 |
| 0.5 (0.42) | -0.00738 (0.00077, 0.00090) | 8/8 [0.74, 1.00] | 1.00 / 1.00 | 1.00 | **-0.00255** | **8/8** |
| 0.7 (0.61) | -0.01528 (0.00143, 0.00091) | 8/8 | | | -0.00768 | 8/8 |
| honest oracle | -0.0360 (0.0030) | passes | | | -0.0223 | passes |

Source: `results/harness_thresholds.json`, 300 TL runs, current library; pass probabilities from `results/gate_and_ci.json` (F§ "The gate, tightened"). App C.4 adds the TL-mix and public columns.

- A transferred slope needs an honest $r$ of about 0.3 to 0.35. At 0.3 it passes about two times in three (Figure 5).
- A per-pair slope needs an honest $r$ of about 0.46 (0.455 to 0.475), because it must be learned from at most 31 labels.
- The gate uses the thinnest regime, TL runs, so it leans toward false negatives. At honest $r = 0.2$ the transferred line gains 0.0008 on TL runs, against 0.0016 on TL-mix and 0.0019 to 0.0025 on public runs (App C.4 reads the gate on TL-mix).

Correlations against full-sample difficulty sit on the same scale as the honest $r$. The honest difficulty is reliable (0.84 to 0.95), so a correlation against full-sample difficulty is only 0.5% to 1.6% higher than one against the honest difficulty. What changes them is the unit over which they are taken. The gate reads a covariate within a test-like pair, where its bar is a within-pair $r$ of 0.25. Most correlations in §6.3 are instead taken over a benchmark's whole range, within item_features groups, or over text-bearing subsets. A degraded oracle keeps 0.82 of its $r$ within a pair, and the main covariates of the 14B language model described below keep 0.57 to 0.65 (App F.1).

![Figure 5: the acceptance gate, with the measured covariates on it](fig/gate_curve.png)

*Figure 5.* (a) The nested TL ALC difference against hier-ship ("shipped hier") for a degraded oracle at honest $r$ (top axis: within-pair $r$), for the transferred and the per-pair slope, with the 8 noise draws and the number that pass. The gate is -0.002. (b) The 44 measured covariates at their within-pair $\lvert r\rvert$ and their nested lines, drawn translucent because several overlap. None reaches the gate. (c) Per-benchmark correlations with difficulty, with 95% group-bootstrap intervals, for the text maps and for the language-model studies described below: the 4B judge, and the 14B model's rubric head and reasoning entropy.

### 6.2 What was tried

We record each idea as negative, with its script (Figure 6). The footing column says how we scored each idea. A "harness" idea was scored against hier-ship leave-one-parent-out ("nested" when selection is redone inside each held-out parent). A "data-level" idea was scored as a correlation with difficulty measured on the data ("LOBO" when fitted with the target benchmark left out). A "correlation rule" idea was scored against the bar that the study's own plan set, and a "feasibility" idea by whether usable data exist. The text maps and the blind ratings are data-level correlations with difficulty, and the last three rows come from the legacy replica.

| idea | footing | main number | within-pair $r$ | verdict |
|---|---|---|---|---|
| item difficulty from TF-IDF text | data-level, LOBO | Pearson $r$ with Rasch difficulty -0.16 to 0.14, every 95% interval below 0.3 | 0.07 | does not transfer |
| item difficulty from Qwen3-Embedding-0.6B | data-level, LOBO | ridge -0.23 to -0.02 | -0.11 (ridge); -0.04 (kNN) | does not transfer |
| blind LLM difficulty rating | data-level | 0.482 [0.25, 0.64] on matharena; -0.13 to 0.21 elsewhere | | mathematics only; inconclusive on two benchmarks, below 0.3 on swe_rebench |
| Qwen3-4B zero-shot judge | harness | no TL $\Delta \le -0.001$, parents weighted equally | 0.04 to 0.09 | closed |
| a 4B model's own attempts | correlation rule | graded accuracy 2.3% and 7.8%, below the 10% floor | | cannot be read |
| entropy and hidden-state heads | LOBO $r$ | -0.05 [-0.24, 0.14] and +0.12 [-0.004, 0.24] | | replicated null |
| in-context learning over the pair's labels | correlation rule | $r$ of 0.206 ± 0.040, below 0.3 | | closed |
| pairwise (anchored) comparisons | correlation rule | pooled q 0.540, below 0.60 | | dropped |
| text-similarity residual layer (itemsig) | harness, nested | -0.00002 (0.00008) | | no gain |
| meta-learned heads on frozen embeddings | harness, nested | 0 (every head off in every fold); forced +0.0002 to +0.0006 | | no gain |
| fine-tuned encoder | LOBO $r$; harness | 0 of 4 parents at $r \ge 0.3$; transferred line +0.00048 | | closed |
| Qwen3-14B demand rubric | harness, nested | best line -0.00084 (0.00043) | 0.11 (head); 0.15 (best scales) | null for ALC |
| Qwen3-14B reasoning attempts (matharena) | correlation rule; forced per-pair | Spearman $\rho$ 0.372 [0.209, 0.516] | 0.32 (one parent) | passes its correlation bar; null for ALC |
| Qwen3-14B reasoning entropy, four parents | harness, nested; rule fixed before the data | transferred +0.00110 (0.00045) | 0.12 [0.09, 0.15] | null |
| known-sign item cues | harness, nested | 0 to +0.00003; text length +0.28 / +0.36 / -0.43 within paper; stated size +0.50 on one benchmark | +0.41 (stated size, one benchmark) | none passes |
| subject side (harness identity, effort, date forms, Student-t level) | harness, nested | best -0.00097 (0.00033) | | none passes |
| acquisition policies | legacy replica | coverage-first -0.0000 ± 0.0013 | | none beats random |
| post-hoc temperature and slip | legacy replica | loses 0.0007 ± 0.0011 | | does not transfer |
| offline bank of per-item results | feasibility | 1 of 161 inventory benchmarks judged usable | | closed |

Brackets: 95% intervals. A parenthesis after a nested line holds its cluster SE; other ± values are as each study records them. "Within paper" means within researchcodebench's paper groups. In the pairwise row, q is the share of comparisons in which the model's answer matches the sign of the honest difficulty gap. App F.1 gives the scripts and the intervals.

![Figure 6: every idea against the gate](fig/ideas_forest.png)

*Figure 6.* Each idea's TL ALC difference against hier-ship, with ±1 pair-cluster SE. Filled markers are the nested line the gate reads, hollow markers a forced or fixed configuration, and triangles the same line's worse public weighting. A triangle off scale sits at the edge with its value, as for the 14B model's estimate of an expert's time, part of its demand rubric (-0.0037). The dashed line is the gate, -0.002. No nested line reaches it. The best are the subject side's ordered reasoning effort (-0.0010 ± 0.0003) and the 14B model's estimate of an expert's time (-0.0008 ± 0.0004). The corrected multiple-choice floor, adopted as a bug fix, is drawn for contrast. Footings differ by idea (§6.2's table).

### 6.3 Study by study

**Text maps.** With one benchmark left out, the difficulty predicted from TF-IDF text correlates -0.16 to 0.14 with Rasch difficulty, and that predicted by a ridge on embeddings -0.23 to -0.02. Within a benchmark, embeddings reach 0.27 to 0.66. Most of that comes from identifying the item_features group, which hier already learns from labels, and with whole groups held out they fall to -0.09 to 0.40 (App F.2).

**Language-model judges.** A blind rater with an undocumented prompt, run on 180 items, correlates 0.482 [0.25, 0.64] with Rasch difficulty on matharena and -0.13 to 0.21 elsewhere. With 45 items per benchmark, the intervals on multi_swebench and real_webagents still include 0.3 (inconclusive rather than negative), while swe_rebench's interval, [-0.41, 0.17], excludes it. We closed a fully specified 4B judge through the harness, and the 14B described below rated every item (App F.3).

**A label-conditional layer.** The itemsig layer (`paiec/itemsig.py`) carries hier's residuals on a pair's labelled items over to unlabelled items with similar text. With nested selection over 480 configurations, its $\Delta$ on TL runs is -0.00002. It cannot act at $B_0$ or $B_1$, and at $B_{31}$ it takes 0.5% of the 0.063 Brier gap between the pair-rate and item oracles (Figure 7; App F.4).

![Figure 7: Brier of the pair-rate oracle, the item oracle and hier-ship by regime](fig/item_gap.png)

*Figure 7.* Brier on the evaluated responses per regime for the pair-rate oracle, the item oracle (in-sample Rasch difficulty) and hier-ship ("shipped hier") at $B_{31}$. The gap between the two oracles (± cluster SE) is what an item-difficulty model could add. The itemsig layer's nested line recovers 0.5% of it at $B_{31}$ on TL runs and 1.3% to 2.9% elsewhere. Run sets: TL runs 0 to 299, TL-mix 0 to 199, R1-bf 0 to 199 and R1-pu 0 to 99.

**Probes, heads and encoders.** Under nested selection, meta-learned heads on frozen embeddings are off in every fold, and forcing them on costs +0.0002 to +0.0006. Three probes did not reach their bars: a 4B model's own attempts fell below their accuracy floor, and in-context learning and pairwise comparisons stayed below theirs. Entropy and hidden-state heads are a replicated null. A fine-tuned encoder reaches a LOBO $r$ of 0.3 on 0 of 4 parents (App F.5, F.9).

**Known-sign cues.** We declared the signs of benchmark-agnostic cues in advance (`paiec/itemcov.py`). Most of these cues vary on one benchmark only. Text length, the one cue present everywhere, changes sign. The stated size of researchcodebench items exists on that benchmark alone, where its within-pair $r$ is +0.41. Nested TL differences are 0 to +0.00003 (App F.6).

**The subject side.** We tried four changes to the priors that act at $B_0$ and $B_1$. Ordered reasoning effort is switched on in all four folds, and its $\Delta$ is -0.00097, half the bar. All of that gain comes through a lower date slope under the test-like date shift. Harness identity cannot be measured leave-one-parent-out, the date form is decided by the shift, and a Student-t level ties or loses (App F.7).

**Qwen3-14B on free GPUs.** We ran Qwen3-14B on two Kaggle T4 GPUs. It rated all 4,326 items of the four parents on a demand rubric, attempted 270 matharena texts, and reasoned once over every item. The token entropy of its attempts on mathematics reaches a Spearman $\rho$ of 0.372 [0.209, 0.516] with difficulty, measured within matharena's competitions. It is the one signal past its correlation bar. We scored its reasoning entropy under a rule committed before the data existed. Within group, the entropy orders difficulty on mathematics and research code (+0.29 and +0.33). On the two agentic benchmarks it orders difficulty weakly or not at all (+0.12 and -0.03). Its within-pair $r$ is 0.12, and it fails the gate (App F.9).

**Legacy replica.** No acquisition policy beat the platform's random one, so the submission ships no `labeling.py`. Post-hoc temperature and slip do not transfer. We closed an offline bank of per-item results on feasibility and on the rules' public-pool restriction (App F.8).

### 6.4 Why item-side signals fail

First, item-side terms barely reach the budgets that matter most, $B_0$ and $B_1$. Together they carry 0.3 of ALC's weight. On the second formative run $B_0$ was the largest error (0.237), and $B_1$ and $B_3$ were nearly equal (0.195 and 0.196). On the third run the same three budgets were 0.235, 0.187 and 0.189.

A centred covariate is zero at $B_0$, where only the level and subject priors act. The harness's uncentred $B_0$ term never passes the gate: even for the honest oracle, whose difference of -0.0027 clears the bar, it fails another of the gate's conditions. A per-pair slope is zero at $B_1$ too. A transferred slope does act at $B_1$: at the gate's honest $r$ of 0.3 it gains 0.0026 of Brier there, against 0.0027 to 0.0030 at $B_3$ to $B_{31}$ (`results/gate_and_ci.json`, `gate.b1`).

Second, the item-level headroom is large, and reaching it needs a signal that no public cue has. On TL runs the headroom is 0.063 of Brier. A transferred slope needs an honest $r$ of about 0.3 to 0.35, or 0.25 within a test-like pair. Within a pair, every covariate that can be carried to a new benchmark reaches at most 0.15 (§6.1, §6.2). The larger correlations of §6.3 (0.48 and 0.37 on mathematics, 0.29 and 0.33 within group) are taken over a benchmark's whole range, within groups, or on one parent. They are larger because of the unit over which they are taken, not because of the difficulty, full-sample or honest, that they are correlated with.

---

## 7 Limitations

The evidence rests on five public benchmarks, and four of them, the multi-subject parents, carry every test-like number. The headline's parent-level SE (0.009) is about 2.7 times its cluster SE, and per-parent gains range from 0.014 on matharena to 0.058 on multi_swebench (§5.4).

We built the test-like regime ourselves and tuned it in sample to one feedback run. It realises a level of -1.29 rather than its target level mean of -1.6. The size of its gain comes from a synthetic date shift that matches the feedback for LegacyP's prior only. Without the shift (TL-noshift) the gain is 0.014 (§5.4), and most of the tuned headline is LegacyP's own optimism under that shift (§5.5).

Regimes set to the pooled reading (-0.65 to -0.78) and to the level audit's reading share the tuned regime's catalogue and date shift, and they realised milder levels than targeted. In them hier-ship's gain over LegacyP is 0.025 to 0.031 (§5.4, App C.1).

Over a calibrated level, what hier-ship adds in our measurements is robustness to where the hidden level sits (§5.5). That comparison lacks BLE (the organisers' language-model predictor), which we did not run, and 1PL-ship borrows hier's calibrated level prior centre $\mu_0$ (§5.5, App E.5).

Our estimates are open to selection optimism. The selection and confirmation halves and the RS runs all redraw one catalogue of pairs, and they reuse the public pairs that chose the level. We set several hyperparameters with all five public benchmarks in view (App C.2), and we never scored hier-ship with the test-like regime's target level mean set to -2.0.

On the one public single-subject benchmark, swe_rebench, hier-ship loses about 0.010 ALC to LegacyP. Nothing measures variation between single-subject benchmarks, and the hidden test may hold some (§5.6).

We did not investigate two costs at $B_1$ on dense runs, and they remain unexplained: hier's pooling on real_webagents (+0.015 ± 0.010 of Brier) and LegacyP's pooled difficulty on multi_swebench (+0.004 to +0.009). When a cost appears at $B_1$, the first thing we check for is a mishandled second-order (posterior-curvature) term. We also did not run the full gate under TL-mix (App C.4, App I.1).

LegacyP's fit can diverge when its prior sits far from the labels, and the test-like item oracle oscillates on some strata. We report both defects in our code but have not fixed them. A third defect was that hier's fit on items with a multiple-choice guessing floor could stop unconverged, and we fixed it in archive-3 (App B.6).

Some design inputs are recorded only as outputs. Three records are missing: the blind language-model rater's prompt, the judgement of which inventory benchmarks publish graded per-item outputs, and run 3's headline score, which we did not paste into our records. The RS study's lock (the hashes of its plan and rule) rests on local file times and on the accepted precision deviation (App H.3, C.3). That deviation is one reproduction check compared at its comparator's stored precision rather than at the planned one.

Formative feedback is noisy: a single run's ALC has an sd of 0.02 to 0.04, and the three runs are different draws (one pair recurs, in runs 1 and 3, under different models). We used the feedback only to set global hyperparameters, and used run 3 only for its regression check.

---

## Code, data and conduct

Our only training data is measurement-db at a pinned revision, and we run the organisers' streaming client at a pinned commit (App G.2). App G gives every command that rebuilds the submission or reruns an experiment, with wall times. Every number traces to a committed script and results file, except some wall times in the notes under App G.4's table (App H). We have packed the per-row files behind the tables into archives, except the rows of itemsig, the text-similarity residual layer. We do not host the archives (App G.4).

The formative feedback, the organisers' scored tables of each formative submission, set global quantities only: the test-like regimes' settings (level targets, date shift, run shape) and LEVEL's three hyperparameters. No input, prior or selection is keyed on an anonymous id.

Run 1 set the test-like regime's defaults, and the level audit read it per pair. Run 2 joined it in the pooled reading, which read both runs' pairs against matched replica pairs and changed nothing. We wrote the reading's decision rule before computing it, but after we had seen both runs' tables. Run 3 was a regression check only (§5.4, App G.6).

The inventory scan cloned the repositories of the organisers' benchmark inventory to list and match their file paths. These repositories may include hidden-test benchmarks. Our records do not show whether some of their files were opened. Nothing from the scan, its outputs or the inventory enters `paiec/`, `submission/` or `tools/`, and no per-item data from it was kept or used (App G.6).

We used Qwen3-Embedding-0.6B, Qwen3-4B-Instruct-2507 and Qwen3-14B-AWQ only for research features computed offline, and the submission uses no pretrained model (App G.6). We developed the code, the experiments and this report with Claude Code. The blind difficulty rater of §6.3 was the Claude model of the Claude Code session that made our repository's first commit, and its prompt is undocumented.

---

## References

Author-year; the venue's year where a venue exists, with the arXiv id of every arXiv-first work. Checked on 2026-10-02 against arXiv, Crossref, Semantic Scholar, the venues and the dataset card (`docs/report/references_check.md`); the one preprint without a venue is marked.

- Barton, M. A. and Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series* 1981(1). doi:10.1002/j.2333-8504.1981.tb01255.x
- Benedetto, L., Cremonesi, P., Caines, A., Buttery, P., Cappelli, A., Giussani, A. and Turrin, R. (2023). A survey on recent approaches to question difficulty estimation from text. *ACM Computing Surveys* 55(9): 1-37. doi:10.1145/3556538
- Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association* 88(421): 9-25. doi:10.1080/01621459.1993.10594284
- Breslow, N. E. and Lin, X. (1995). Bias correction in generalised linear mixed models with a single component of dispersion. *Biometrika* 82(1): 81-91. doi:10.1093/biomet/82.1.81
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1): 1-3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- De Boeck, P. and Wilson, M. (eds.) (2004). *Explanatory item response models: a generalized linear and nonlinear approach.* New York: Springer. doi:10.1007/978-1-4757-3990-9
- Fischer, G. H. (1973). The linear logistic test model as an instrument in educational research. *Acta Psychologica* 37(6): 359-374. doi:10.1016/0001-6918(73)90003-6
- Fox, J.-P. (2010). *Bayesian item response modeling: theory and applications.* New York: Springer. doi:10.1007/978-1-4419-0742-4
- Ge, C., Kryvosheieva, D., Fried, D., Girit, U. and Hariharan, K. (2026). Agent psychometrics: task-level performance prediction in agentic coding benchmarks. In *Conference on Language Modeling (COLM 2026)*. arXiv:2604.00594
- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477): 359-378. doi:10.1198/016214506000001437
- Krsteski, S. and Meyer, C. (2026). Predicting task difficulty without rollouts. *COLM 2026 Workshop on Agent Behavior*. arXiv:2608.05797
- Kwa, T., West, B., Becker, J., et al. (2025). Measuring AI ability to complete long software tasks. In *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2503.14499
- Lalor, J. P., Wu, H. and Yu, H. (2016). Building an evaluation scale using item response theory. In *Proceedings of EMNLP 2016*, 648-657. doi:10.18653/v1/D16-1062. arXiv:1605.08889
- Li, Y., Ma, J., Ballesteros, M., Benajiba, Y. and Horwood, G. (2025). Active evaluation acquisition for efficient LLM benchmarking. In *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267: 35581-35602. arXiv:2410.05952
- Lipton, Z. C., Wang, Y.-X. and Smola, A. (2018). Detecting and correcting for label shift with black box predictors. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, PMLR 80: 3122-3130. arXiv:1802.03916
- Liu, Q. and Pierce, D. A. (1994). A note on Gauss-Hermite quadrature. *Biometrika* 81(3): 624-629. doi:10.1093/biomet/81.3.624
- Lugoloobi, W. and Russell, C. (2025). LLMs encode how difficult problems are. arXiv:2510.18147 (preprint; no venue found)
- Lugoloobi, W., Foster, T., Bankes, W. and Russell, C. (2026). LLMs encode their failures: predicting success from pre-generation activations. In *Conference on Language Modeling (COLM 2026)*. arXiv:2602.09924
- Maia Polo, F., Weber, L., Choshen, L., Sun, Y., Xu, G. and Yurochkin, M. (2024). tinyBenchmarks: evaluating LLMs with fewer examples. In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235: 34303-34326. arXiv:2402.14992
- Martínez-Plumed, F., Prudêncio, R. B. C., Martínez-Usó, A. and Hernández-Orallo, J. (2019). Item response theory in AI: analysing machine learning classifiers at the instance level. *Artificial Intelligence* 271: 18-42. doi:10.1016/j.artint.2018.09.004
- Moreno Cencerrado, I. V., Padrés Masdemont, A., Gonzalvez Hawthorne, A., Africa, D. D. and Pacchiardi, L. (2026). No answer needed: predicting LLM answer accuracy from question-only linear probes. *ICLR 2026 Workshop on Principled Design for Trustworthy AI*. arXiv:2509.10625
- Morris, C. N. (1983). Parametric empirical Bayes inference: theory and applications. *Journal of the American Statistical Association* 78(381): 47-55. doi:10.1080/01621459.1983.10477920
- Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In *Proceedings of the AAAI Conference on Artificial Intelligence* 29(1). doi:10.1609/aaai.v29i1.9602
- Pacchiardi, L., Cheke, L. G. and Hernández-Orallo, J. (2024). 100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances. *KDD 2024 Workshop on Evaluation and Trustworthiness of Generative AI Models*. arXiv:2409.03563
- PAIEC baseline repository (2026). https://github.com/aims-foundations/paiec_baseline, commit `82d330dd` (App G.2).
- PAIEC organisers (2026). Predictive AI Evaluation Competition at NeurIPS 2026. https://aimslab.stanford.edu/competition (read 2026-10-02).
- Patz, R. J. and Junker, B. W. (1999). A straightforward approach to Markov chain Monte Carlo methods for item response models. *Journal of Educational and Behavioral Statistics* 24(2): 146-178. doi:10.3102/10769986024002146
- Pinheiro, J. C. and Bates, D. M. (1995). Approximations to the log-likelihood function in the nonlinear mixed-effects model. *Journal of Computational and Graphical Statistics* 4(1): 12-35. doi:10.1080/10618600.1995.10474663
- Rasch, G. (1960). *Probabilistic models for some intelligence and attainment tests.* Copenhagen: Danish Institute for Educational Research. Expanded edition 1980, Chicago: University of Chicago Press.
- Rodriguez, P., Barrow, J., Hoyle, A. M., Lalor, J. P., Jia, R. and Boyd-Graber, J. (2021). Evaluation examples are not equally informative: how should that change NLP leaderboards? In *Proceedings of ACL-IJCNLP 2021 (Volume 1: Long Papers)*, 4486-4503. doi:10.18653/v1/2021.acl-long.346
- Rodríguez, G. and Goldman, N. (1995). An assessment of estimation procedures for multilevel models with binary responses. *Journal of the Royal Statistical Society, Series A* 158(1): 73-89. doi:10.2307/2983404
- Ruan, Y., Maddison, C. J. and Hashimoto, T. (2024). Observational scaling laws and the predictability of language model performance. In *Advances in Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2405.10938
- Rue, H., Martino, S. and Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. *Journal of the Royal Statistical Society, Series B* 71(2): 319-392. doi:10.1111/j.1467-9868.2008.00700.x
- Saerens, M., Latinne, P. and Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation* 14(1): 21-41. doi:10.1162/089976602753284446
- Tierney, L. and Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal densities. *Journal of the American Statistical Association* 81(393): 82-86. doi:10.1080/01621459.1986.10478240
- Truong, N., Truong, S. T. and Koyejo, S. (2026). The AI Measurement Data Bank (measurement-db). AIMS Lab, Stanford University. https://aimslab.stanford.edu/measurement-db; Hugging Face dataset `aims-foundations/measurement-db`, revision `bc8204d811823da849c6686bf124d4ca9f82e4de`, CC-BY-SA-4.0.
- Truong, S. T., Tu, Y., Liang, P., Li, B. and Koyejo, S. (2025). Reliable and efficient amortized model-based evaluation. In *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267: 60238-60265. arXiv:2503.13335
- Vivek, R., Ethayarajh, K., Yang, D. and Kiela, D. (2024). Anchor points: benchmarking models with much fewer examples. In *Proceedings of EACL 2024 (Volume 1: Long Papers)*, 1576-1601. doi:10.18653/v1/2024.eacl-long.95. arXiv:2309.08638
- Ye, Q., Fu, H. Y., Ren, X. and Jia, R. (2023). How predictable are large language model capabilities? A case study on BIG-bench. In *Findings of EMNLP 2023*, 7493-7517. doi:10.18653/v1/2023.findings-emnlp.503. arXiv:2305.14947
- Zhang, Q., Lyu, F., Liu, X. and Ma, C. (2024). Collaborative performance prediction for large language models. In *Proceedings of EMNLP 2024*, 2576-2596. doi:10.18653/v1/2024.emnlp-main.150. arXiv:2407.01300
- Zhou, L., Pacchiardi, L., Martínez-Plumed, F., Collins, K. M., et al. (2026). General scales unlock AI evaluation with explanatory and predictive power. *Nature* 652(8108): 58-67. doi:10.1038/s41586-026-10303-2. arXiv:2503.06378

---

## Appendix A: data, protocol and the legacy replica

### A.1 Data sources and the inventory classification

The counts and shares in the data table of §2.2 come from these sources. Pair counts come from [P§ Data], and distinct-item counts from `results/harness_thresholds.json` (`meta.oracle_info`), except swe_rebench's, which come from P§ Data. The script `experiments/data_counts.py` recounts the pairs, responses and distinct items from `data/` and gets the same numbers (`results/data_counts.json`). Mean accuracies come from the R2 table in F§ "Under the official protocol" and, for swe_rebench, from F§ "How much there is to win". Variance shares come from F§ "The step-2 analyses behind the model" (`experiments/hier_design/item_signal.py`); we quote its recorded output and did not rerun it.

For matharena, the table's row and the same section's note on textless items rest on further counts. In eligible pairs, matharena's 1,633 items span 25 competitions. The competition's 0.50 share of matharena's item variance was estimated on all 1,751 items with a binary response, which span 27 competitions; imc_2025 and miklos_2025 occur only on subjects below the 80-item floor (`results/data_counts.json`, `matharena`). The textless items read "See image" or fragments of a system prompt, and all 336 Kangaroo items are in eligible pairs (`results/data_counts.json`; see F§ "What transfers between benchmarks" and F§ "The multiple-choice floor, corrected"). F§ "What transfers between benchmarks" also records that raw solve rates confound difficulty and that Rasch targets fix it.

The hidden benchmarks come from the organisers' inventory of 161 candidate benchmarks, which they assigned at random to the training and test pools before curation. Keyword rules read only the titles (no description, paper or item) and class 108 of the 161 (67%) as evaluations of AI systems. Of those 108, 39% are text QA, 30% images, 15% agents, 6.5% code, 4.6% video, 2.8% math and 2.8% audio.

We checked the rules on 80 labelled titles from the organisers' full sheet. They agree with the labels on which titles are evaluations for 82.5% of them, and on the category for 38 of the 40 that both call evaluations (95%). An AI agent assigned the 80 labels by reading each title; no person labelled them (`experiments/inventory_classes.py`, `results/inventory_classes.json`; F§ "What this could do on the hidden test"). The public benchmarks are therefore not a representative sample of the hidden task types.

### A.2 The legacy replica, and the pooling decomposition in full

We built our first replica (`paiec/evaluator.py`) before the organisers' streaming client was public, and we keep it as the *legacy* replica so that its numbers stay reproducible. It differed from the protocol in five ways [P§ The legacy replica]:

1. The replica scored pair-major: each pair went through all six budgets before the next pair started, with one predictor instance throughout. A stateful predictor therefore held earlier pairs' full 31-label trajectories while the replica scored it at budget 0.
2. `labeled` held only the target subject's own labels, never other subjects'.
3. One session held all 221 pairs instead of 5 to 12 pairs.
4. The replica passed private identifiers and raw benchmark names to `predict`.
5. The legacy replica's first version keyed the split on Python's `hash()`, which Python salts per process, so the "fixed" split changed between runs. Both replicas hash by digest in the current code.

These differences changed the results: on the legacy replica, evaluation order alone moved the assembled run-time predictor from 0.1898 to 0.1725 (F§ "Evaluation order is worth 0.017", `experiments/order_sensitivity.py`). The legacy predictor ladder (predictors of growing complexity) made pooled item difficulty look worth about 0.020 ALC (F§ "The predictor ladder", `experiments/ladder.py`), and that pointed our research at cross-subject pooling.

On the legacy replica, mixing the training mean into LegacyP's level prior cost 0.004 (legacy). Under the official protocol the training mean is the better centre for hier: centring its level prior at 0 instead costs 0.0011 on public runs (App B.3).

We re-measured the baselines and the pooling gain on the official replica over the 600 formative-sized runs of R1 (F§ "Under the official protocol", `experiments/official_baselines.py`):

- The organisers' empirical-mean baseline is no better than answering 0.5: it scores 0.2526 ± 0.0014 against 0.2500, because one label sends it to 0 or 1 (0.3746 at $B_1$).
- LegacyP beats Smooth by only 0.0026 ± 0.0009 (cluster SE) with a strict prior, or 0.0069 ± 0.0016 with a LOPO prior.
- Pooled difficulty adds about 0.0007 to LegacyP at formative size (600 runs, seed 0). LegacyP pools difficulty only once a benchmark has 64 distinct labelled items (its WARMUP threshold), and a formative run has about two pairs per benchmark.

Run size, and not the information set, made the pooling lever vanish. The legacy ladder and the official R1 runs differ in run size and in split scope as well as in the information set. To separate the three, we switched pooling on and off on the official replica, under both split scopes, on dense runs and on formative-size runs. We then reweighted the formative runs to the platform's own composition: of the 26 appearances in formative runs 1 to 3, 16 were alone on their benchmark and 10 had one companion.

For hier-ship, pooling off restricts `labeled` to the target pair's own entries, and a level per pair separates the pooled level's share from the items' share. For LegacyP, pooling off sets the WARMUP threshold out of reach; that threshold is the switch behind the 0.0007 (F§ "Pooling under the verified protocol"; `experiments/pooling_decomposition.py`, `results/pooling_decomposition.json`).

| pooling on minus off, ALC | dense, scope 'pair' | dense, 'benchmark' | formative, 'pair' | formative, 'benchmark' | platform mix, 'pair' / 'benchmark' |
|---|---|---|---|---|---|
| hier-ship, everything from other subjects | -0.0300 ± 0.0099 | -0.0142 ± 0.0059 | -0.0033 ± 0.0004 / 0.0006 / 0.0005 | -0.0020 ± 0.0004 / 0.0005 / 0.0004 | -0.0010 / -0.0004 (cluster SE 0.0003) |
| of which the pooled level | -0.0043 ± 0.0014 | -0.0029 ± 0.0009 | -0.0015 | -0.0012 | -0.0007 / -0.0003 |
| of which pooled item difficulty | -0.0258 ± 0.0088 | -0.0114 ± 0.0055 | -0.0018 | -0.0008 | -0.0004 / -0.0002 |
| LegacyP, pooled difficulty | -0.0237 ± 0.0065 | -0.0086 ± 0.0038 | -0.0010 ± 0.0001 / 0.0002 / 0.0001 | -0.0004 ± 0.0001 / 0.0001 / 0.0001 | 0 / 0 |

Dense: the four multi-subject benchmarks, mean ± SE across them; researchcodebench, matharena and multi_swebench are scored on at most 32 evaluation items per pair, with the full dense `labeled` list, and real_webagents is scored whole. Formative: 100 runs (seed 11, R1-bf), ± run / cluster / stratified SE. Entries are negative where pooling helps. An earlier version of the pooling study's script scored its rows, and the script's summary code changed afterwards; five tasks re-scored by the current script are bit-identical.

- Under the verified protocol, pooling is still a large lever on dense runs: 0.009 to 0.030 ALC by scope and predictor, which brackets the legacy ladder's 0.020. Most of hier-ship's dense value is pooled item difficulty.
- At formative size pooling is worth 0.002 to 0.003 to hier-ship and 0.0004 to 0.001 to LegacyP. At the platform's composition it is worth 0.0004 to 0.001 to hier-ship and exactly 0 to LegacyP: two pairs never reach the 64 distinct items of the WARMUP threshold. In the replica's formative runs (scope 'pair'), its value to hier-ship grows with company, from nothing for a pair alone on its benchmark to 0.0027 with one companion and 0.0053 with more.
- Per-benchmark splits halve it on dense runs (ratios 0.47 for hier-ship and 0.36 for LegacyP).

The first replica therefore did not invent the lever, but the platform's runs are too small to use it, and per-benchmark splits would halve what is left. The legacy numbers still ranked predictors under a different information set. We quote the legacy replica only where a number is marked "legacy".

On public runs reweighted to the platform's composition (most pairs alone on their benchmark), hier-ship is not distinguishable from LegacyP under per-pair splits, +0.0024 ± 0.0023 ($z$ about 1.0), and is behind it by +0.0045 ± 0.0024 under per-benchmark splits (cluster SEs). We also broke this comparison down by class of pair. In the replica, 87 of the 193 pairs alone on their benchmark are swe_rebench, which is alone in every run. Without swe_rebench the two differences become +0.0004 ± 0.0026 and +0.0017 ± 0.0022, and those of the 'alone' class (pairs alone on their benchmark) go from +0.0050 and +0.0075 to +0.0019 ± 0.0033 and +0.0030 ± 0.0028. So that cost belongs mostly to the single-subject benchmark and not to pairs alone on a multi-subject benchmark (F§ "Pooling under the verified protocol", "Not explained"; `results/pooling_decomposition.json`, `single_subject_confound`).

### A.3 Split scope on dense runs, and other open points

The replica keeps what the organisers' client leaves open as parameters and assumes no value for them [P§ Still unknown]. The main one is `split_scope`, which decides whether the 50/50 split is drawn per pair or per benchmark item.

- Under per-pair splits, other subjects' acquired labels land on most of a target's evaluation items on dense runs (83% to 99.5% at $B_{31}$).
- Under per-benchmark splits, that coverage falls to 0% to 5%.

On dense runs (every pair of one benchmark), the scope changes the comparisons between predictors. Signed values here and in the next paragraph are differences in ALC, negative where the first-named predictor is better. LegacyP minus Smooth on matharena is -0.0354 under per-pair splits and -0.0159 under per-benchmark splits (F§ "One benchmark, every pair (R2)"). For hier the scope can flip the sign: on dense researchcodebench, hier-fit beats LegacyP by 0.0042 under per-pair splits and loses by 0.0040 under per-benchmark splits (F§ "Hierarchical model", R2).

The pooling study later ran hier-ship on all four dense multi-subject benchmarks under both scopes (§2.4; F§ "Pooling under the verified protocol", subsection "Dense runs per benchmark, and hier's cost there"). Against LegacyP, hier-ship gains most on multi_swebench, by 0.021 and 0.017 under per-pair and per-benchmark splits. On matharena the two are tied (-0.0006 and -0.0007, hier-ship minus LegacyP). On researchcodebench hier-ship again gains under per-pair splits, by 0.005, and loses under per-benchmark splits (+0.007). The SEs are over pairs, 0.002 to 0.004. §2.5 lists which studies ran which scope.

The file `docs/protocol.md` lists the other open points:

- which recorded response is revealed for a repeated item;
- the per-call timeout;
- how a formative run picks and cuts its pairs.

The item counts in the formative feedback suggest that the platform splits after cutting, with a floor near 88 kept items (`paiec/testlike.py` docstring).

### A.4 How the replica was verified

- We check `default_policy` decision for decision against the organisers' `run_streaming`, and the two match [P§ Interfaces].
- The platform hands each call its own copy of the arguments, and these copies do not change results. On one run, the replica with platform-like argument copies and 16 simulated workers gives per-pair Brier and ECE (expected calibration error) bit-identical to its fast path, one worker without copies. This holds for all six predictors in the baseline table of F§ "Under the official protocol": the baselines and LegacyP with its strict and its LOPO prior.
- The empirical mean's ALC follows from base rates, $\mathrm{ALC} \approx 0.025 + 1.2118\,\mathbb E[p(1-p)]$, where $p$ is a pair's true success rate, and the replica agrees with this formula. Once the split and stream order are salted per run, the replica's residual against the exact expectation is -0.0005 ± 0.0006 (F§ "The empirical mean's ALC is a function of base rates"; Figure A1).
- `tests/test_official.py` (34 test functions) pins the replica's protocol invariants. At the release commit, tagged `v1.0.0` (App H.4), the test suite has 528 test functions over 24 files (729 collected tests). The suite runs on synthetic data and needs no download.

![Figure A1: the empirical mean's ALC against base rates](fig/empirical_mean.png)

*Figure A1.* The empirical mean's ALC against the run's $\mathbb E[p(1-p)]$ for each of the 600 public R1 runs, with the analytic line $0.025 + 1.2118\,\mathbb E[p(1-p)]$, the mean over runs (0.2526) and the exact expectation at that mean (0.2496). The organisers' leaderboard entry (0.1801) is drawn at the $\mathbb E[p(1-p)]$ an empirical mean would need to score it (0.128). Of the replica's runs, 2.5% lie below it.

---

## Appendix B: model, inference and run-time details

### B.1 Inference details, fallbacks and floored fits

Each item's likelihood is the one-dimensional integral over its residual $e_i$. We compute it by adaptive Gauss-Hermite quadrature with 20 nodes, or by a fixed 201-node trapezoid on floored items (items with a multiple-choice floor $c_i$), whose integrand can be bimodal. Newton's method with a line search finds the mode of the joint log posterior of $\mathbf x$, all components except the item residuals. After we read the target along its line (§3.2), a 171-node trapezoid computes $\mathbb{E}\big[\operatorname{logit}^{-1}(\eta_{si}) \mid \mathcal L\big]$, exact to $10^{-9}$.

The read gives the exact marginal for one pair whose labelled items share the target item's group effects. Labelled items in other item_features groups, or other subjects on the benchmark, add components off the line, which move along their Gaussian conditional mean (their skew ignored), so in those cases the read is an approximation.

The test that checks the read against the exact posterior predictive for one pair, `test_laplace_error_at_low_budgets`, has no groups, no floor and no slip. The docstring of `paiec/hier.py` documents the remaining approximation error case by case, for other subjects on the benchmark and for benchmarks with floored items: on cases drawn from the model, the error moves Brier by about 0.001 at $B_1$ and by at most $10^{-4}$ from $B_7$ on. We have not measured the case of a single pair whose labelled items lie in other groups separately.

**Convergence.** Without floors the log posterior is concave and its mode is unique. A floored item's success term is not concave, so the negative Hessian that Newton's method inverts is shifted to positive definite when needed. Where Newton's method stops short on a posterior that holds a floored success, trust-region Newton from three starts settles the fit (`Problem._settle`, commit `4d2cc4f`; F§ "Floored fits"). Where Newton's method converges, this settling step changes nothing, bit for bit. We count the fits that are still unconverged after it.

**Fallbacks of the offline prior.** When the included benchmarks cannot identify a hyperparameter, it falls back to a fixed `REFERENCE` value that no fit went into. For example, $\mu_0 = 0$ when fewer than three levels are available.

**The identity link, in full.** The variance that a named model's attribute residuals share across benchmarks, $\tau^2_{\mathrm{res}}$, is estimated at -0.002 on the public data and clipped to [0.01, 0.2]. That puts the identity-link sd $\sigma_\theta$ at its floor of 0.1 and the link weight across benchmarks at 0.0018 (`paiec/hier.py` docstring). The estimator is biased downward, because the ridge that predicts a name's standing on one benchmark was fitted without that benchmark but saw the same name's standings on the others, and that pulls the name's residuals on two benchmarks apart. Until we correct the estimator and measure again, two things are provisional: the conclusion that the identity link adds almost nothing beyond the attributes, and the small link weight that the estimate sets (F§ "The step-2 analyses behind the model").

### B.2 Run-time engineering and the build checks

- `predict` is a pure function of `(input, labeled)`, and its fits are cached under a content fingerprint of `labeled`. Items and subjects are keyed on digests of all their visible fields, because the official input carries no ids and text prefixes collide.
- The predictor never raises an exception. A failed fit falls back to the prediction without labels and then to 0.5, and an unusable `prior.json` falls back to the level prior without a subject prior.
- The archive ships `model.py`, `prior.json` and the run-time modules renamed to `paiec_rt/`, so that no platform-side `paiec` can shadow them. Shipped code imports only the standard library and numpy at module level, and BLAS is held to one thread at run time.
- `tools/build_submission.py` writes the archive only if all of the following hold, and fails otherwise:
  - a fresh interpreter loads `model.py` the way the validator does and predicts real items in the official format at every budget, with a fresh predictor per budget;
  - those predictions match the in-repository predictor bit for bit;
  - no fallback fires and no heavy library loads;
  - the organisers' validator prints OK.

### B.3 Ablations of hier-fit

We scored each option alone against hier-fit's default on the same public R1-bf runs, with split scope 'pair' (the primary setting), before the level calibration. The table lists all sixteen comparisons: twelve ablations and four width sensitivities.

| option | runs | minus default (run / cluster / stratified SE) | where it acts |
|---|---|---|---|
| attribute prior off | 150 | +0.0039 ± 0.0003 / 0.0009 / 0.0008 | $B_0$ to $B_3$ |
| feature groups off | 150 | +0.0017 ± 0.0002 / 0.0003 / 0.0003 | $B_7$ to $B_{31}$ |
| level pooling off (a level per pair) | 150 | +0.0012 ± 0.0003 / 0.0004 / 0.0004 | $B_1$, $B_3$ |
| level centre 0 instead of the training mean | 150 | +0.0011 ± 0.0003 / 0.0007 / 0.0006 | ±0.004 per benchmark |
| pair deviation off | 150 | +0.0001 ± 0.0001 / 0.0001 / 0.0001 | |
| identity link off, or relinked at link weight 0.1 / 0.3 (three options) | 100 | 0.0000 to -0.0001 ± 0.0000 / 0.0000 / 0.0000 | |
| hard floor ($\omega = 1$; default $\omega = 0.5$) | 150 | +0.0001 ± 0.0000 / 0.0000 / 0.0000 | |
| text term on | 100 | +0.0002 ± 0.0001 / 0.0001 / 0.0001 | 5.3 ms per call |
| Student-t level ($\nu = 3$) | 100 | -0.0005 ± 0.0001 / 0.0002 / 0.0001 | 18.9 ms per call |
| Laplace fit without the line | 150 | -0.0007 ± 0.0001 / 0.0003 / 0.0001 | $B_1$, $B_3$ |
| $\sigma_\mu \times 0.5$ | 100 | -0.0021 ± 0.0003 / 0.0008 / 0.0004 | $B_1$, $B_3$; swe_rebench -0.0068 |
| $\sigma_\mu \times 2$ | 100 | +0.0049 ± 0.0004 / 0.0015 / 0.0005 | $B_1$, $B_3$ |
| $\sigma_\delta \times 0.5$ | 100 | -0.0004 ± 0.0003 / 0.0007 / 0.0006 | |
| $\sigma_\delta \times 2$ | 100 | +0.0060 ± 0.0004 / 0.0013 / 0.0008 | $B_1$, $B_3$ |

Source: F§ "Ablations and sensitivities", `experiments/hier_eval.py`. The last column names the budgets or benchmarks where an option acts; for the text term and the Student-t level it gives their cost, the time per call.

On public runs the attribute prior, the feature groups and the pooled level each help. A narrower level prior, the Student-t level and the Laplace fit without the line beat the default by more than two cluster SEs. None of the three clears a two-sided Bonferroni bar for sixteen comparisons on the cluster SE, and all three clear it on the stratified SE. Each of them makes the model react less to a run's first labels. The deliberately widened level prior costs about 0.002 on public runs, and §5.4 explains why we kept it.

![Figure B1: hier's ablations and sensitivities](fig/hier_ablations.png)

*Figure B1.* Each of hier-fit's sixteen options alone minus hier-fit, on R1-bf runs, with ±1 cluster SE (the legend's "pair-cluster SE"). The figure also marks the two-sided Bonferroni bar for sixteen comparisons (±2.955 SEs), on the cluster SE and on the stratified SE. A narrower level prior, the Student-t level and the Laplace fit without the line beat hier-fit by more than two cluster SEs. None of them clears the Bonferroni bar on the cluster SE, and all three clear it on the stratified one. In the figure, "line off (Laplace)" is the Laplace fit without the line, and "relink" sets the link weight of the identity link.

### B.4 The corrected multiple-choice floor

The corrected floor reads "(A, B, C, D, or E)" lists and ignores TikZ point labels. It lowers ALC by 0.00026 ± 0.00003 / 0.00008 (run / cluster SE) on test-like runs and by 0.00027 to 0.00045 on public runs, all of it on matharena at $B_0$ and $B_1$ (F§ "The multiple-choice floor, corrected", `experiments/mcq_floor.py`). We measured it on the harness's legacy rows, with hier's solver as it was before the floored-fit fix.

We adopted the corrected floor in the library. On 2026-09-27 we rebuilt the archive with it and with the floored-fit fix (sha256 `4a882cc7…`). The validator printed OK, the run check was bit-identical, and `prior.json` was byte-identical to the previous build. Formative run 2 used the earlier archive, archive-2, and run 3 used the rebuilt one (§5.1).

### B.5 Cost in full

A formative run makes about 3,000 evaluation calls, which take a few seconds against the 8-hour limit. The pooling study measured hier's worst case, dense multi_swebench and matharena, with about 2,500 labelled entries at $B_{31}$. The table gives the mean time per evaluation call and the slowest single call, by study.

| study | model | runs | processes | mean time per call | slowest single call | LegacyP's slowest call |
|---|---|---|---|---|---|---|
| subject-side study | hier-ship | subject-side runs | two | 0.80 to 1.04 ms | 0.16 to 0.31 s | |
| level calibration | hier-rec | test-like; public | six | 0.75 ms; 0.88 ms | 0.45 s | |
| level calibration | hier-rec | dense real_webagents and researchcodebench | six | 2.0 to 2.4 ms | at most 0.16 s | |
| regime-sensitivity (RS) study | hier-ship, current library | 520 runs | two | 1.34 ms | 0.56 s | 1.86 s |
| RS study | both hier-EB variants | the same tasks | two | 1.18 to 1.19 times hier-ship's | | |
| pooling study | hier-ship | dense multi_swebench; dense matharena | one | 3.1 to 4.6 ms; 7.5 to 9.4 ms | 0.39 s | 1.44 s |

Sources, one per study in the table's order: F§ "Subject side at budgets 0 and 1", Latency; F§ "Calibrating for the hidden test", Latency; F§ "Regime sensitivity at the feedback's reading", Latency; F§ "Pooling under the verified protocol". The subject-side, RS and pooling times were taken on a shared machine. No fallback fired in the RS study.

The floored-fit fix changes the time per call by -0.2% to +0.2% on public runs (F§ "Floored fits", Latency).

In the baseline study the 1PL takes 0.72 to 0.81 ms per call, with a slowest call of 0.13 to 0.18 s, on one process of a shared machine. hier-ship's 1.2 to 1.5 ms per call on the same runs was timed in the RS study's tasks rather than in the 1PL's, so the ratio of about 1.7 to 1.9 compares times across tasks (derived; F§ "Baselines: a plain 1PL, the organisers' empirical mean and BLE", Cost; `results/baselines_p1.json`, `summary.regimes.<R>.configs`).

### B.6 Library defects

`fitting.fit_ab`, LegacyP's fit, diverges when its prior sits far from the labels, which happens on 5 of 1,598 test-like pair appearances. We reported this library bug in F§ "fit_ab diverges when the prior is far from the labels" and have not fixed it.

`testlike.item_oracle`, which we also reported and left unfixed, oscillates on 35 of 2,425 test-like pair appearances, all of them on difficulty strata (689 appearances). On strata the oscillation understated the item oracle: it gained 14.5% of the pair-rate oracle's Brier instead of 32.7%.

The floored Newton fit of hier could stop unconverged. This happened in 11 of 3,030 floored fits on 400 public runs, and one of them collapsed (pair-uniform run 108). Commit `4d2cc4f` fixes it (`Problem._settle`). With the fix no fit is left unconverged, and the mean prediction for run 108's pair at $B_{31}$ goes from 0.217 to 0.770, against an observed success rate of 0.669 (F§ "Floored fits", `results/hier_floor.json`). The fix and the corrected floor went into archive-3, the archive rebuilt on 2026-09-27 (sha256 `4a882cc7…`). The numbers for hier-ship in §5.4 and App E.7 come from the earlier archive's code, except the regimes re-measured at `4d2cc4f` (App H.2).

---

## Appendix C: regimes, rules and when they were fixed

Regimes, run sets and predictors carry the names of the main text's glossary.

### C.1 Regime catalogue and realised levels

A tilted regime draws its pairs toward a target level distribution. Every tilted regime realises a milder level than its target, that is, a mean pair logit closer to zero. The table gives the realised mean pair logit of the scored runs, with its sd in parentheses.

| regime (seed) | knobs | target | realised mean (sd) |
|---|---|---|---|
| TL (300 check runs) | level_mean -1.6, level_sd 1.5, date shift | -1.6 (sd 1.5) | -1.29 (1.70), cluster SE 0.16 |
| TL-1.2 (3) | level_mean -1.2 | -1.2 | -1.10 |
| TL-2.0 (3) | level_mean -2.0 | -2.0 | -1.56 |
| LA-0.8 (5) | level_mean -0.8 | -0.8 | -0.77 |
| LA-flat (5) | no level tilt | none | -0.59 |
| TUNED (11) | the defaults | -1.6 (sd 1.5) | -1.28 (1.62) |
| READING (11) | level_mean -0.85, level_sd 1.75 | -0.72 | -0.63 (1.68) |
| AUDIT (11) | level_mean -1.8, level_sd 2.1 | -1.10 | -0.93 (2.05) |
| MIXTURE (11) | a two-component level target (`Regime.level_mix`) | | -0.64 (2.13) |
| FLAT (11) | no level tilt | none | -0.28 (2.23) |
| public R1 | | | -0.71 (1.42) |

Sources: `results/testlike_check.json`; `results/level_audit.json` (`extra`); `results/regime_sensitivity.json` (`summary.realised`); public R1 on the continuity-corrected scale of App D.4. We set the RS regimes' knobs on seed 10 from run composition alone, with no predictor. READING and AUDIT came out milder than their targets, but within the realisation bounds fixed before scoring, which limit how far a regime's scored runs may drift from the level its knobs realised on that seed.

**How test-like runs are built, in full.** The construction is in `paiec/testlike.py` and in the docstring of `experiments/testlike_check.py`.

*Pseudo-benchmarks.* The five public benchmarks are too few and too central, so we cut them into pseudo-benchmarks. Pseudo-benchmarks are of four kinds: item_features groups sorted by difficulty and merged, difficulty strata, whole parents (a *parent* is the public benchmark that a pseudo-benchmark is cut from), and random chunks for the single-subject benchmark. The strata are cross-fitted, so that a subject's own labels never choose its items' stratum. Each pseudo-benchmark gets its own anonymous benchmark_id, and every label is a real recorded response. The construction changes only the grouping of items and, under the date shift below, the release and access dates that the predictor sees.

*Run shape.* A run draws 5 to 12 pairs; the competition site's example run shows 5 pairs. With some probability the next pair joins a pseudo-benchmark already in the run; otherwise it opens a new one. This reproduces the feedback's composition of about one pair per benchmark.

*Date shift.* The visible release and access dates move 1.25 years later. This reproduces the optimism at $B_0$ that the feedback showed for the legacy prior's linear date term.

The test-like regime was tuned to one feedback run, and that run does not identify the regime's level: level_mean -1.2, -1.6 and -2.0 lie 0.63, 0.37 and 0.46 run sds from the feedback at $B_0$ and $B_1$. The regimes set to the feedback's level readings have level sds of 1.7 to 2.1, and MIXTURE's low mode rests on two pseudo-benchmarks.

**Run sets of the later studies.** The baseline study rescored the RS study's TUNED and public runs (seed 11). The pooling decomposition drew two run sets, each under both split scopes. The first is R1 at seed 11, 100 runs drawn benchmark-first; its runs 0 to 59 are the RS study's public R1-bf runs. The second is R2, the dense runs of the four multi-subject benchmarks.

### C.2 The decision rules

**The level selection** (§3.4) took the configuration with the best mean ALC on the selection half of the TL runs (runs 0 to 99), among the configurations that lose at most 0.003 ALC against LegacyP on *both* public weightings (the guard). We confirmed configurations on runs 100 to 199.

**The gate** has the four conditions of §4.4 (F§ "Subject side at budgets 0 and 1").

**The rule** of the RS study, fixed before scoring, asked a replacement to meet five conditions:

1. gain at least 0.002 in READING and in AUDIT, with the upper end of its 95% interval below zero;
2. have a negative parent-level mean there, with no parent losing more than 0.004;
3. lose at most 0.002 in TUNED, MIXTURE and FLAT;
4. lose at most 0.001 on public runs, and at most 0.003 against LegacyP there;
5. take at most twice hier-ship's time per call.

**The reading rule** for runs 1 and 2 is in App D.4.

**With all five public benchmarks in view**, we set `WIDEN`, `G_CAP`, the slip $\varepsilon$, the guess weight $\omega$, `MAX_LEVELS` and the REFERENCE value of $\sigma_\delta$. The value $\sigma_\mu = 2.5$ lay on the edge of the grid we scored. We scored $\sigma_\mu = 3.5$ and 5 later, at hier-ship's $\mu_0$ and $\gamma$, and they do not do better (§5.4).

### C.3 When each rule was fixed

| study | rule | where it was fixed | evidence | caveat |
|---|---|---|---|---|
| the harness and its gate | the four gate conditions | the harness's plan block; first in commit `f7e7d87` | gate and results committed at once | shows only that gate and results were committed together |
| itemsig | none | `experiments/itemsig_eval.py` never held a gate | results committed a day before the gate | read as a plain null |
| the 14B rubric and attempts | rubric and reading rule | plan block | data produced 13.6 hours after `f7e7d87` | |
| the entropy job | reading rule and primary feature | commit `78e303e` | committed before any output existed; data 83.8 hours after `f7e7d87` | |
| the pooled reading of runs 1 and 2 | the reading rule | written into the results file with a timestamp and a hash | a test checks that the stored rule is the script's and predates the first reading | both runs' tables had been seen; four read runs (below) |
| the RS study | plan, rule and constants block, hashed | the script's lock stage | local file times and copies in its results file | nothing committed before scoring; the precision deviation |

**What the repository shows.** The record of when the gates in the scripts' plan blocks were fixed is uneven (F§ "The gate, tightened"; `results/gate_and_ci.json`, `evidence`). The harness's gate first appears in commit `f7e7d87` (2026-09-27 04:45 UTC), together with the results of the harness, the known-sign cues and the subject side. For those three studies the commit shows only that gate and results were committed at once. The itemsig results were committed a day earlier, before any commit held the gate. The 14B's rubric and attempt data were produced 13.6 hours after that commit, and its entropy data 83.8 hours after, under a reading rule committed before them (`78e303e`). The other probes were committed after the gate, but the repository does not record when their inputs were produced. We therefore describe the gate as fixed in the plans of the studies committed from `f7e7d87` on, and we do not call it pre-registered. The script `experiments/itemsig_eval.py` has never contained a gate, so itemsig predates the gate and we read it as a plain null (App F.4).

For the one reading of formative feedback made after run 2, we wrote the decision rule into its results file, with a timestamp and a hash, before the reading was computed. A test checks that the stored rule is the script's and predates the first reading. This fixes the rule but not what had been seen: both runs' feedback tables, including run 2's per-pair Brier, were known when the rule was written. We then revised the reading code twice in its descriptive outputs. The earlier versions, rebuilt and re-run, give the same decision (App D.4).

The RS study (§5.4) hashed its plan, its rule and the constants block of its script before any scoring-seed run was drawn, and a lock stage checks those hashes. Nothing was committed before scoring, so the evidence that the lock came first is local file times and copies, recorded in its results file. A dry run of the rule on the first 19 rows printed only "the planned set is not complete", and no code or constant changed after it. One reproduction check departed from the plan, and we accepted the departure on 2026-10-02 (the precision deviation, below).

**The reading was run four times.** The read stage ran at 05:13:48, 05:15:57, 05:19:08 and 05:20:19 under three versions of the script, with digests `5099ef81…`, `321c189e…` and `93cb6a84…`; the last digest is the file in the repository. The stored reading comes from the last of the four runs. The rule's text was byte-identical throughout: the read stage refuses to run when the stored rule and the script's differ. The two edits after the first reading changed descriptive outputs only. The first edit extended the split of run 2's $B_0$ excess to 40 neighbours, kept apart the pair q3 (its $B_{31}$ has no real root) and added explanatory notes. The version before that edit had split the pairs 4 and 4 at 15 neighbours (0.047 below a pair rate of 0.5, 0.060 at or above it) and counted q3 above. The second edit added the neighbours' match distance to the robustness variants. The edit between the hash-locked rule and the first reading touched the record stage only. We rebuilt the version that locked the rule and both earlier reading versions from a log of the edits, and re-ran them. They give the stored decision, $z$ values and bias guard (the rule's check for bias from the reading itself) exactly, and the same value in every one of the 1,502 to 1,559 fields they share with the stored reading (`experiments/script_revisions.py`, `results/script_revisions.json`).

**The precision deviation.** Before scoring began, we compared one reproduction check at its comparator's stored precision instead of the planned $10^{-9}$, and we accepted this on 2026-10-02. The RS outcome (no candidate; LEVEL stays) assumes that deviation. Read literally, with the check held to $10^{-9}$, the rule was not applied; under either reading there is no candidate.

### C.4 The gate in full

The harness first reproduces the in-sample oracle that the study of the meta-learned heads scored (App F.5). The targets are -0.0437 on test-like runs and -0.0557 on TL-mix, and the harness reproduces -0.0439 and -0.0558 (targets: `results/heads_eval.json`, `experiments/heads_eval.py`). It then builds the *gate table* from degraded oracles, covariates $x_i$ for each item $i$ built as

$$
x_i = r\,\tilde d_i + \sqrt{1 - r^2}\,\xi_i
$$

Here $\tilde d_i$ is an *honest* difficulty, fitted on other subject folds only and standardised within its parent benchmark, $\xi_i$ is standard normal noise, and $r$ sets the honest correlation (§6.1). For each $r$ the table records what a covariate with that honest correlation would score through the gate, averaged over 8 noise draws, and how many single draws pass. In the table, *nested* means that selection is redone inside each held-out parent.

| honest $r$ (within-pair $r$, test-like) | transferred slope, nested: test-like (cluster SE, draw sd) | TL-mix | R1-bf | R1-pu | draws passing | per-pair slope, nested: test-like | draws passing |
|---|---|---|---|---|---|---|---|
| 0 (0.00) | -0.00002 (0.00001, 0.00006) | +0.00001 | -0.00001 | +0.00000 | 0/8 | -0.00000 | 0/8 |
| 0.1 (0.08) | -0.00010 (0.00005, 0.00043) | -0.00027 | -0.00034 | -0.00044 | 0/8 | -0.00000 | 0/8 |
| 0.2 (0.16) | -0.00083 (0.00015, 0.00086) | -0.00161 | -0.00191 | -0.00245 | 1/8 | -0.00010 | 0/8 |
| 0.3 (0.25) | **-0.00255** (0.00032, 0.00068) | -0.00457 | -0.00517 | -0.00639 | **6/8** | -0.00032 | 0/8 |
| 0.4 (0.33) | -0.00462 (0.00052, 0.00081) | -0.00805 | -0.00906 | -0.01109 | 7/8 | -0.00136 | 0/8 |
| 0.5 (0.42) | -0.00738 (0.00077, 0.00090) | -0.01246 | -0.01399 | -0.01699 | 8/8 | **-0.00255** | **8/8** |
| 0.7 (0.61) | -0.01528 (0.00143, 0.00091) | -0.02392 | -0.02693 | -0.03213 | 8/8 | -0.00768 | 8/8 |
| honest oracle | -0.0360 ± 0.0010 / 0.0030 / 0.0029 | -0.0484 | -0.0539 | -0.0619 | passes | -0.0223 | passes |

Source: `results/harness_thresholds.json` (`thresholds.honest`, `acceptance`), `python experiments/harness.py --stage table`; 300 test-like, 150 TL-mix and 100 + 100 public runs, on rows collected with the current library (corrected multiple-choice floor and floored-fit fix; F§ "Acceptance harness", Provenance). The table built before that re-collection, on the legacy rows, differs from this one by at most 0.0002 in the test-like column and 0.0004 elsewhere; its transferred line at $r = 0.3$ was -0.00238.

**What a covariate needs.** One noise draw of a degraded oracle moves its test-like difference by more than its cluster SE (draw sd 0.0007 against 0.0003 at $r = 0.3$). A real covariate is one draw, so we read its chance of passing from the draws rather than from the averaged line (F§ "The gate, tightened"; `experiments/gate_and_ci.py`, `results/gate_and_ci.json`, `gate`):

| honest $r$ (within-pair $r$) | transferred slope: draws passing (Jeffreys 95%) | one draw clears the bar, $\Pr(\Delta \le -0.002)$: normal / Student-t predictive | $\Pr$ of passing the full gate, modelled |
|---|---|---|---|
| 0.2 (0.16) | 1/8 [0.01, 0.45] | 0.09 / 0.12 | 0.09 |
| 0.3 (0.25) | 6/8 [0.41, 0.94] | 0.79 / 0.76 | 0.68 |
| 0.4 (0.33) | 7/8 [0.55, 0.99] | 1.00 / 0.99 | 0.87 |
| 0.5 (0.42) | 8/8 [0.74, 1.00] | 1.00 / 1.00 | 1.00 |

A transferred slope needs an honest $r$ of about 0.3 to 0.35. At $r = 0.3$ a covariate passes about two times in three. The averaged line there is -0.00255, with a draw sd of 0.00068. Of the two failing draws, one misses the bar and one fails the worst-parent condition (inferred: the other four conditions hold, and a draw's worst parent is not stored). A per-draw chance of 0.5, 0.8 or 0.95 of clearing the bar needs an honest $r$ of 0.27, 0.305 or 0.335.

A per-pair slope needs an honest $r$ of about 0.46 (0.455 to 0.475), because it must be learned from at most 31 labels. Forcing a per-pair slope ($\sigma_\beta = 0.5$) on for an uninformative covariate costs +0.0009 ALC on test-like runs when the term acts from $B_1$, and +0.0007 when it acts from $B_7$.

The gate uses the thinnest regime, the one in which a covariate gains least, so it leans toward false negatives. At $r = 0.2$ the transferred line gains 0.0016 on TL-mix and 0.0019 to 0.0025 on public runs, but 0.0008 on test-like runs. Read on TL-mix, with selection kept on test-like runs, the same per-draw chances of clearing the bar need an honest $r$ of 0.215, 0.24 and 0.255 (0.365 to 0.39 for a per-pair slope), and the pass counts are 4/8 at $r = 0.2$ and 8/8 from $r = 0.3$. These counts are upper bounds, because a draw's worst parent on TL-mix is not stored.

We deferred the full gate under TL-mix. It needs a change to `gate()` and `average_lines()` in `experiments/harness.py`, then a re-run of the table stage with selection on TL-mix (about 25 minutes on one process). The re-run would move the degraded oracles' pass counts, but the stored TL-mix nested lines of the 44 measured covariates in Figure 5 (b) are all above -0.002 (the best is -0.0014, the 14B's time_log_minutes), so the full gate under TL-mix could pass a covariate only where selection on TL-mix switched on one that test-like selection did not (App I.1).

---

## Appendix D: formative runs

### D.1 Scores, archives and the recurring pair

Formative runs on different draws do not measure an improvement, and this holds for all three runs: they measure neither the change from LegacyP to hier nor the library changes between runs 2 and 3. The difference between two independent runs' ALCs has an sd of about 0.04 (0.037 to 0.047 for hier-ship). The observed differences are 0.019 between runs 1 and 2 and 0.011 between runs 2 and 3. Measured offline, the code gap between runs 2 and 3 is at most 0.00046.

The platform's run score is the unweighted mean of the per-pair ALCs. Computed that way from the returned budget tables, runs 1 and 2 score 0.2112999889 and 0.1926232375. These round to the reported 0.2113 and 0.192623, so both reported scores reproduce to their precision, while weighting pairs by their item counts does not reproduce them (F§ "Formative feedback, runs 1 and 2"). Run 3's headline score was not pasted, so its 0.1816525778 applies the rule rather than testing it. Each pair's summary ALC matches its budgets to within $5 \times 10^{-7}$.

Run 2's archive holds the modules of commit ee5085a byte for byte. That is the library of App E.7's rows, before the corrected multiple-choice floor and the floored-fit fix. Run 3 used archive-3 (sha256 `4a882cc7…`), which differs from it by those two changes, worth -0.0003 on test-like runs (App B.4, App H.2). We uploaded it as a regression and latency check only, and stated that purpose before it was scored (`results/formative_feedback.json`, `submissions`, first committed in `00bdf04` on 2026-09-28). The platform scored all 9 pairs at every budget and reported no errors. Its table records no timing. The file in `dist/` has the archive's bytes, and every tracked member matches `4d2cc4f`. Unlike the archives of runs 1 and 2, it was not downloaded back from the platform (`results/formative_run3.json`, `archive`).

All three runs spread their pairs over the same 7 hidden benchmarks, and in no run did a benchmark hold more than two of the run's pairs. Runs 1 and 2 share no subject, and run 3 shares none with run 2. Run 3 shares one (subject, benchmark) pair with run 1 (run 1's p7 and run 3's s6, on benchmark A with 53 items both times), so the three runs hold 25 distinct subjects over 26 appearances.

We treat that recurring pair as descriptive only. On it, run 3's ALC is 0.066 lower and its $B_0$ is 0.252 lower. But the pair ran under two models, and the other pairs sampled into each run differed, so the pair saw a different shared `labeled` list in each run.

The per-pair tables below give each pair's Brier at every budget, its ALC and its ECE-ALC, which combines the pair's expected calibration error (ECE) over the budgets with the ALC weights. Letters relabel the anonymous benchmarks, and the same letter is the same benchmark in all three runs. The letters are for recording only, and no model input is keyed on them (F§ "Formative feedback, runs 1 and 2" and "Formative run 3"; the organisers' tables are `results/formative/run1.txt`, `run2.txt` and `run3.txt`).

Run 1, LegacyP:

| pair | benchmark | n | $B_0$ | $B_1$ | $B_3$ | $B_7$ | $B_{15}$ | $B_{31}$ | ALC | ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| p1 | A | 76 | 0.4224 | 0.2851 | 0.2698 | 0.2067 | 0.2085 | 0.2057 | 0.2568 | 0.171 |
| p2 | B | 50 | 0.4970 | 0.2861 | 0.1394 | 0.0764 | 0.0650 | 0.0513 | 0.1682 | 0.288 |
| p3 | C | 56 | 0.3583 | 0.2419 | 0.2429 | 0.2437 | 0.2751 | 0.1923 | 0.2558 | 0.264 |
| p4 | D | 58 | 0.5663 | 0.3234 | 0.1379 | 0.0518 | 0.0184 | 0.0063 | 0.1636 | 0.344 |
| p5 | E | 60 | 0.3032 | 0.2477 | 0.2479 | 0.2431 | 0.2437 | 0.2472 | 0.2515 | 0.073 |
| p6 | F | 44 | 0.1728 | 0.1670 | 0.1675 | 0.1665 | 0.1719 | 0.1678 | 0.1686 | 0.038 |
| p7 | A | 53 | 0.4771 | 0.2939 | 0.1738 | 0.1335 | 0.1285 | 0.1286 | 0.2065 | 0.204 |
| p8 | F | 44 | 0.1806 | 0.1747 | 0.1761 | 0.1827 | 0.1745 | 0.1749 | 0.1771 | 0.042 |
| p9 | G | 44 | 0.2519 | 0.2653 | 0.2834 | 0.2360 | 0.2386 | 0.2366 | 0.2535 | 0.103 |

Run 2, hier-ship (archive of ee5085a):

| pair | benchmark | n | $B_0$ | $B_1$ | $B_3$ | $B_7$ | $B_{15}$ | $B_{31}$ | ALC | ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| q1 | D | 58 | 0.2555 | 0.1939 | 0.1840 | 0.1931 | 0.1868 | 0.1857 | 0.1957 | 0.091 |
| q2 | G | 44 | 0.2706 | 0.1635 | 0.1790 | 0.1500 | 0.1458 | 0.1490 | 0.1696 | 0.136 |
| q3 | C | 113 | 0.2593 | 0.2469 | 0.2408 | 0.2437 | 0.2418 | 0.2524 | 0.2458 | 0.045 |
| q4 | G | 44 | 0.2576 | 0.2406 | 0.2559 | 0.2311 | 0.2315 | 0.2161 | 0.2392 | 0.093 |
| q5 | F | 44 | 0.2256 | 0.2300 | 0.2553 | 0.2329 | 0.2228 | 0.2392 | 0.2347 | 0.104 |
| q6 | E | 60 | 0.2366 | 0.1973 | 0.1989 | 0.1961 | 0.1950 | 0.1983 | 0.2009 | 0.068 |
| q7 | A | 75 | 0.2333 | 0.2126 | 0.2282 | 0.2119 | 0.2008 | 0.2240 | 0.2164 | 0.104 |
| q8 | B | 54 | 0.1562 | 0.0729 | 0.0283 | 0.0098 | 0.0034 | 0.0014 | 0.0386 | 0.160 |

Run 3, hier-ship (archive of 4d2cc4f; `results/formative_run3.json`, `run3.pairs`). s6 is run 1's p7: the same subject on the same benchmark, with 53 items both times.

| pair | benchmark | n | $B_0$ | $B_1$ | $B_3$ | $B_7$ | $B_{15}$ | $B_{31}$ | ALC | ECE-ALC |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | G | 44 | 0.2768 | 0.1941 | 0.2119 | 0.1891 | 0.1935 | 0.2067 | 0.2061 | 0.109 |
| s2 | E | 60 | 0.2543 | 0.2962 | 0.3617 | 0.2578 | 0.2635 | 0.2514 | 0.2864 | 0.163 |
| s3 | D | 58 | 0.2071 | 0.1307 | 0.0989 | 0.0927 | 0.0928 | 0.0947 | 0.1132 | 0.095 |
| s4 | C | 56 | 0.2024 | 0.1696 | 0.1873 | 0.1755 | 0.1821 | 0.1865 | 0.1818 | 0.120 |
| s5 | A | 72 | 0.2122 | 0.1377 | 0.1307 | 0.1470 | 0.1303 | 0.1178 | 0.1421 | 0.093 |
| s6 | A | 53 | 0.2254 | 0.1413 | 0.1304 | 0.1233 | 0.1326 | 0.1215 | 0.1402 | 0.095 |
| s7 | G | 44 | 0.2621 | 0.1619 | 0.1923 | 0.1453 | 0.1487 | 0.1508 | 0.1709 | 0.125 |
| s8 | B | 54 | 0.2150 | 0.1365 | 0.1044 | 0.1090 | 0.1023 | 0.1001 | 0.1219 | 0.124 |
| s9 | F | 44 | 0.2606 | 0.3136 | 0.2790 | 0.2581 | 0.2548 | 0.2500 | 0.2722 | 0.139 |

Mean ECE by budget, per run:

| run | mean ECE by budget, $B_0$ to $B_{31}$ |
|---|---|
| 1 | 0.381 0.253 0.195 0.097 0.090 0.047 |
| 2 | 0.210 0.098 0.118 0.084 0.057 0.076 |
| 3 | 0.254 0.146 0.135 0.077 0.073 0.066 |

Source: the organisers' tables cited above.

### D.2 Placement by budget

With hier-ship's level prior, run 2 lost far less at budget 0 than run 1, though on different pairs: its $B_0$ was 0.237 against run 1's 0.359. That is 0.02 above hier-ship's mean on TL runs (0.2165) and on par with its mean on R1-bf runs (0.2350, App E.7). From $B_1$ on, its curve is flat. From $B_3$, where the TL mean is 0.1645, to $B_{31}$, where it is 0.1386, the curve sits 0.03 to 0.045 above the TL means. On a third draw with the same level prior, run 3 had a $B_0$ of 0.235 and fell to 0.164 at $B_{31}$.

The table places both runs in hier-ship's single-run distributions, against the same regime means and sds. TL is the tuned test-like regime, TL-mix and TL-noshift are its sensitivities (item groups merged at random; no date shift), and R1-bf and R1-pu are the public runs, with benchmark-first and pair-uniform weighting.

| regime (runs) | mean ALC | single-run sd | run 2's $z$ | share of runs at or above run 2's 0.1926 | run 2's $z$ by budget, $B_0$ to $B_{31}$ | run 3's $z$ | run 3's $z$ by budget, $B_0$ to $B_{31}$ |
|---|---|---|---|---|---|---|---|
| TL (300) | 0.1658 | 0.0318 | +0.84 | 0.20 | +0.98 +0.15 +0.86 +0.89 +0.98 +1.40 | +0.50 | +0.89 -0.06 +0.65 +0.39 +0.65 +0.81 |
| TL-mix (200) | 0.1711 | 0.0329 | +0.65 | 0.245 | +1.12 +0.05 +0.72 +0.70 +0.72 +1.15 | +0.32 | +1.01 -0.16 +0.52 +0.21 +0.40 +0.58 |
| TL-noshift (100) | 0.1585 | 0.0335 | +1.02 | 0.18 | +1.00 +0.45 +0.89 +1.11 +1.20 +1.67 | +0.69 | +0.95 +0.25 +0.69 +0.57 +0.83 +1.02 |
| R1-bf (150) | 0.2051 | 0.0285 | -0.44 | 0.67 | +0.06 -0.78 -0.49 -0.46 -0.27 +0.23 | -0.82 | +0.00 -1.00 -0.70 -1.08 -0.70 -0.50 |
| R1-pu (100) | 0.2020 | 0.0261 | -0.36 | 0.68 | -0.24 -0.67 -0.37 -0.32 -0.13 +0.50 | -0.78 | -0.30 -0.88 -0.62 -0.95 -0.57 -0.26 |

Sources: `results/ship_confirm.json` (`run2_placement`); run 3 from `results/formative_run3.json` (`placement`). The regimes' rows are the run-2 archive's library. Run 3's archive differs from it by at most 0.00046, and that difference is not added to the rows. The table gives no share of runs at or above run 3, because `results/ship_confirm.json` stores no per-run ALC.

Neither one run nor two can discriminate the regimes. Run 2's profile is at least as close to the public runs as to TL: at every budget it is within 0.8 sd of the public runs' means, while its $B_{31}$ is 1.4 sd above the TL mean. In run 3, $B_0$ is again about 0.9 sd above the TL mean and on par with the public runs. From $B_1$ on, it lies within 0.81 sd of the TL means (0.06 sd below at $B_1$, above after) and 0.3 to 1.1 sd below the public runs' means. Both runs lie within 1.02 single-run sds of every regime's mean ALC.

The mean Brier of run 3 falls by 0.048 from $B_0$ to $B_1$ and rises by 0.0017 from $B_1$ to $B_3$. Two of its pairs, both near a rate of 0.5 ($B_{31}$ at least 0.25), have $B_1$ or $B_3$ above their $B_0$. Each earlier run has one such pair (App D.1).

![Figure D1: the formative runs pair by pair](fig/formative_runs.png)

*Figure D1.* Each formative run's pairs (grey) and their mean (black) by budget, against the replica's mean for the run's model on tuned test-like runs (solid) and on public R1 runs (dashed). Over its 9 pairs, run 1's $B_0$ spans 0.17 to 0.57. For runs 2 and 3 it spans 0.16 to 0.27 and 0.20 to 0.28.

### D.3 What hier-ship was expected to score

Before run 2, the level audit's per-pair matched estimate of hier-rec's ALC on run 1 was about 0.178 (0.169 to 0.183). This estimate replaced the average-shift estimate of 0.167 for hier-rec, the level calibration's recommended configuration. By the same estimator, hier-ship's estimate on the audit's pool is 0.1812 with 15 neighbours among the replica pairs and 0.1830 with 40, where hier-rec scores 0.1824 and 0.1839. On hier-ship's own replica rows the estimate is 0.1774 and 0.1779. Across robustness variants on pools that hold test-like runs, it ranges from 0.1766 to 0.1824. Pools of public runs alone give 0.205, on matches about five times further away (F§ "Formative feedback, runs 1 and 2"; `experiments/level_audit.py` agrees, with 0.1765 and 0.1789).

Run 2 is a different draw, so these estimates predict a like-sized run and make no prediction for its particular pairs. Its 0.1926 lies 0.015 above the estimate on hier-ship's own replica rows ($z = +0.48$ in units of the test-like single-run sd). Run 2 did better than predicted at $B_0$ (by 0.008) and worse from $B_3$ on: by 0.017 to 0.022 at $B_3$ to $B_{15}$, and by 0.034 at $B_{31}$. Its pairs sit nearer a rate of 0.5 than run 1's (mean $B_{31}$ 0.183 against 0.157).

### D.4 The pooled reading of runs 1 and 2

The reading found the pooled level 1.0 to 1.2 SEs more central than that of the tuned test-like regime (TL). That is below the rule's bar of 2 SEs, and the reading proposed nothing. LEVEL has not changed since run 2.

After run 2, the level audit asked for the next feedback to be read per pair, pooled with run 1, before the level distribution was refitted once. Its decision rule could change only $\mu_0$ and $\sigma_\mu$. Before any matching of run 2's pairs and before any pooled reading was computed, we wrote into the results file what the reading could change and when, and hash-locked that text (F§ "Formative feedback, runs 1 and 2"). The rule was written at 2026-09-28 05:06:59 UTC (sha256 `0de18448…`), and the first reading ran at 05:13:48.

The hash fixes the rule's text, but the reading is not blind. The feedback tables of runs 1 and 2 had been seen when we wrote the rule, run 2's per-pair Brier included, and so had the audit's scratch outputs for run 1.

The rule may change one global hyperparameter only: the level distribution of a new benchmark ($\mu_0$ and $\sigma_\mu$ of LEVEL). It proposes a candidate only when two conditions hold. First, with both 15 and 40 neighbours per pair, the 17 pairs' mean (or sd) level must differ from TL's realised -1.29 (sd 1.70) by more than 2 SEs, computed by resampling the 7 benchmarks. Second, a guard must find no bias from the reading itself. Even then, a candidate would have to pass the level calibration's gate on fresh test-like runs set to the reading before it could ship.

The reading works on the continuity-corrected logit of each pair's rate. On the plain logit, the lower-root reading of run 1 averages -1.6 (sd 1.5), and the public R1 appearances average -0.74 (sd 1.50). Both values come from `experiments/hier_design/levels.json` (`r1`; F§ "Against the first real formative feedback"). On the continuity-corrected scale the two are -1.51 and -0.71 (`results/formative_feedback.json`). Read per pair against replica pairs, run 1's levels are spread both ways.

| reading (continuity-corrected pair logit) | mean (SE) | sd (SE) |
|---|---|---|
| runs 1 and 2, 17 pairs, 15 neighbours | -0.65 (0.51) | 1.80 (0.36) |
| runs 1 and 2, 17 pairs, 40 neighbours | -0.78 (0.51) | 1.75 (0.34) |
| tuned test-like regime, realised | -1.29 (0.16) | 1.70 |
| public R1, realised | -0.71 (0.12) | 1.42 |

The pooled mean is 1.24 (15 neighbours) and 1.00 (40 neighbours) SEs above TL's, and the pooled sd is 0.29 and 0.14 SEs above. Both differences are below the bar and the bias guard did not trip, so the rule gives no candidate.

On this scale the shipped LEVEL implies a mean of -1.15, a between-benchmark sd of 1.15 and a within-benchmark sd of 1.10. The 17 pairs are consistent with it: they give a between sd of 1.20 to 1.49 (90% interval from 0 to about 1.7) and a within sd of 1.01 to 1.40. A two-component mixture of levels is not supported (BIC differences of 0.68 and 0.38, against a bar of 2). The only structure in the pairs is two pairs near a zero rate.

Run 2's $B_0$ excess over $B_{31}$ does not sit mainly on low-rate pairs. With 15 neighbours it is 0.047 on the 4 pairs read below a rate of 0.5 and 0.078 on the 3 above, and the eighth pair, which has no real lower root, is kept apart. This split is descriptive, and under the reading rule it changes nothing.

Run 3 is not added to the reading. The rule covered runs 1 and 2 only, and we do no level reading, matching or tuning on run 3 (`results/formative_run3.json`, `no_reading_or_tuning`).

The reading of runs 1 and 2 supports a modest conclusion: it puts the hidden level near the public centre, with an SE of about 0.5 logit, and more central than the regime in which the headline gain was measured, but within the noise of 7 benchmarks. The level audit's two regimes nearer that reading give hier-ship a margin of 0.022 to 0.029 over LegacyP, rather than 0.042. Regimes set to this reading and to the audit's, scored later under a rule fixed in advance, give it 0.025 to 0.031. No scored alternative beats hier-ship there (§5.4).

---

## Appendix E: calibration details

Names of models and run sets, and terms such as parent, LOPO, the guard and the rule, follow the glossary of §2.1. App H.1 explains the internal labels, such as seeds, archives and libraries.

### E.1 hier-fit against LegacyP on public runs

Before the level calibration, we compared hier-fit with LegacyP on 300 runs per setting (F§ "Hierarchical model", `experiments/hier_eval.py`, `results/hier_eval.json`):

| weighting, split scope | hier-fit minus LegacyP (run / cluster / stratified SE) | 95%, pair-cluster | 95%, stratified |
|---|---|---|---|
| benchmark-first, pair (primary) | -0.0024 ± 0.0004 / 0.0013 / 0.0008 | [-0.0047, +0.0002] | [-0.0040, -0.0008] |
| benchmark-first, benchmark | -0.0010 ± 0.0003 / 0.0014 / 0.0009 | [-0.0035, +0.0017] | [-0.0027, +0.0007] |
| pair-uniform, pair | -0.0017 ± 0.0004 / 0.0008 / 0.0007 | [-0.0033, -0.0002] | [-0.0031, -0.0003] |
| pair-uniform, benchmark | -0.0015 ± 0.0003 / 0.0008 / 0.0007 | [-0.0030, +0.0000] | [-0.0029, +0.0000] |

The gain comes from pooling a benchmark's level across the subjects a run holds on it. On multi-subject benchmarks, a target that shares its benchmark with one other pair gains 0.0015 to 0.0033, a target that shares it with two or more gains 0.0019 to 0.0044, and a target alone on its benchmark gains nothing measurable (at most 0.0016 ± 0.0014). On the single-subject swe_rebench pair, hier-fit is worse than LegacyP by 0.006 to 0.009, mostly through the widened level prior.

Weighted like our first formative run (5 targets alone on their benchmark, 4 sharing it with one other pair), hier's expected gain on public data would be 0.0006 to 0.0024. On that evidence our first verdict was to keep LegacyP (F§ "Verdict: keep the Predictor"), and the test-like runs of §5.3 reversed it.

**By company on the benchmark.** The columns count the pairs of the target's benchmark in its run. Each cell gives hier-fit minus LegacyP on the multi-subject benchmarks, with its pair-cluster SE (F§ "Hierarchical model").

| weighting, split scope | alone | 2 pairs | 3 or more |
|---|---|---|---|
| benchmark-first, pair | -0.0016 ± 0.0014 | -0.0033 ± 0.0011 | -0.0044 ± 0.0010 |
| benchmark-first, benchmark | -0.0009 ± 0.0014 | -0.0020 ± 0.0011 | -0.0029 ± 0.0011 |
| pair-uniform, pair | +0.0003 ± 0.0014 | -0.0017 ± 0.0011 | -0.0024 ± 0.0008 |
| pair-uniform, benchmark | -0.0001 ± 0.0012 | -0.0015 ± 0.0011 | -0.0019 ± 0.0008 |

**Strict run-LOBO.** Here we fit everything without any of the run's benchmarks. Over runs 0 to 149, LegacyP scores 0.2111 ± 0.0019 and hier-fit 0.2089 ± 0.0020. Their difference, hier-fit minus LegacyP, is -0.0022 ± 0.0004 / 0.0010 / 0.0006 (run / cluster / stratified SE), with 95% intervals of [-0.0039, -0.0001] (pair-cluster) and [-0.0033, -0.0010] (stratified). Both models score about 0.004 worse than with LOPO fits, which leave out only the target's parent (F§ "Strict run-LOBO").

### E.2 Sensitivity to the regime, and the RS study in full

**Against hier-rec.** The configuration hier-rec ($\mu_0 = -3.0$, $\gamma = 0.25$; F§ "The public guard decides", F§ "What actually shipped, after the audit") was confirmed on runs 100 to 199 at -0.0418 ± 0.0024 / 0.0047 / 0.0042 (run / cluster / stratified SE) against LegacyP, and it cost +0.0008 and +0.0007 on the two public weightings. Against hier-rec, hier-ship gives up 0.0030 ± 0.0005 / 0.0009 / 0.0007 on the selection half and 0.0022 ± 0.0005 / 0.0011 / 0.0009 on the confirmation half. On public runs 0 to 99 (`results/level_audit.json`, `mild`) it gains 0.0023 ± 0.0002 / 0.0005 / 0.0004 (benchmark-first) and 0.0024 ± 0.0002 / 0.0004 / 0.0004 (pair-uniform).

**Sensitivity to the regime.** Each cell is the configuration minus LegacyP, ± run / cluster / stratified SE. Of the two hier configurations, only hier-rec was scored on TL-1.2 and TL-2.0.

| configuration | TL-1.2 (100 runs) | TL-2.0 (100) | TL-mix (100) | TL-noshift (100) | LA-0.8 (40; realised -0.77) | LA-flat (40; realised -0.59) |
|---|---|---|---|---|---|---|
| hier-ship | not scored | not scored | -0.0436 ± 0.0014 / 0.0039 / 0.0031 | -0.0144 ± 0.0012 / 0.0024 / 0.0022 | -0.0286 ± 0.0026 / 0.0039 / 0.0035 | -0.0222 ± 0.0025 / 0.0049 / 0.0043 |
| hier-rec | -0.0388 ± 0.0021 / 0.0044 / 0.0039 | -0.0524 ± 0.0022 / 0.0044 / 0.0039 | -0.0467 ± 0.0017 / 0.0049 / 0.0038 | -0.0139 ± 0.0014 / 0.0028 / 0.0026 | -0.0283 ± 0.0033 / 0.0049 / 0.0044 | -0.0196 ± 0.0033 / 0.0063 / 0.0055 |
| Smooth | -0.0228 | -0.0281 | -0.0260 | +0.0077 | -0.0183 | -0.0119 |
| LegacyP+fix, offset -1.5 | -0.0353 | -0.0446 | -0.0398 | -0.0137 | -0.0275 | -0.0232 |

Sources: hier-ship's TL-mix and TL-noshift cells from `results/ship_confirm.json`; the first four columns of the other rows from F§ "Sensitivity to the regime" (`results/level_calibration.json`); the last two columns from `results/level_audit.json` (`extra`: seed 5, the date shift kept, the library at `4d2cc4f`). The TL-mix and TL-noshift runs are the same 100 runs for every row.

The gain of hier-ship over LegacyP shrinks as the hidden level rises toward the public centre: 0.042 at the tuned regime's realised level of -1.29, 0.029 at -0.77 and 0.022 at -0.59. The RS regimes below show the same pattern on fresh seeds and the current library: 0.043 at -1.28, 0.031 at -0.93, 0.025 and 0.026 at -0.63 and -0.64, and 0.020 at -0.28. The synthetic date shift is what inflates the attribute prior in this regime. Without it, the gain against LegacyP shrinks to 0.014, while the gain against Smooth does not shrink.

**The RS study in full, at the feedback's reading.** The level audit moved the level because the hidden levels "look spread both ways", and the pooled reading of runs 1 and 2 put them near the public centre (both in §5.4). No regime at that reading had been scored. The RS study scores ten alternatives against hier-ship on identical runs of seven regimes (F§ "Regime sensitivity at the feedback's reading"; `experiments/regime_sensitivity.py`, `results/regime_sensitivity.json`).

The test-like regimes share the catalogue and the date shift of §4.1 and differ only in their level target (§4.1, RS regimes). Every prior is fitted with the target's parent left out, and every hier row runs on the library of the current archive (`4d2cc4f`). Scoring seed 11 had not been used before, but its runs redraw the same catalogue and the same public pairs that chose the level, so TUNED and the public guard are not an independent replication. §5.4 tabulates each alternative minus hier-ship, with cluster SEs.

**Closest alternatives.** None of the configurations we scored is a better bet than hier-ship at the feedback's reading. Three sit within 0.0006 of it in READING and AUDIT: hier-ship σ3.5 and both hier-EB configurations, with cluster SEs of 0.0003 to 0.0007. In READING, AUDIT and MIXTURE and on both public weightings, hier-ship is first or tied with the first (within 1.96 cluster SEs). Configurations ahead of it by more than that margin appear only in TUNED (hier-rec, hier-nosubj and both hier-EB configurations) and in FLAT (hier-ship σ3.5, by 0.0010 ± 0.0005).

**Where hier-rec wins.** It beats hier-ship only in TUNED, the regime and catalogue it was selected on, by 0.0034 (cluster SE 0.0010). In every other regime it loses, by +0.0005 to +0.0033, and the higher the level, the more it loses. Its losses sit on matharena, the highest-level parent. That pattern is the level audit's argument (§5.4).

**Wider level priors.** Neither wider prior does better than hier-ship: hier-ship σ3.5 stays within 0.001 of hier-ship everywhere (-0.0010 to +0.0009), and hier-ship σ5 is worse by 0.0022 to 0.0031 in five of the seven regimes.

**The empirical-Bayes level.** Contrary to the expectation of our internal review, the adaptive empirical-Bayes level does not beat the fixed one where levels spread both ways. On hier-ship, hier-EB gains only in TUNED (-0.0014). It ties hier-ship in the four other test-like regimes and costs +0.0006 and +0.0014 on public runs, which fails the public guard.

**Against LegacyP.** The gain of hier-ship over LegacyP is 0.043 in TUNED, 0.031 in AUDIT, 0.026 in MIXTURE, 0.025 in READING and 0.020 in FLAT, with parent-level SEs of 0.008 to 0.015. On public runs hier-ship is tied with LegacyP or slightly ahead (-0.0002 and -0.0039).

Four parents carry these numbers. Every alternative with a lower centre than hier-ship trades the same two parents: it gains on multi_swebench, the lowest-level parent, and loses on matharena, the highest-level one. The parent-level SE is two to four times the cluster SE. The RS study's decision rule detects gains of about 0.003 or more, and at 0.002 it detects a gain about half the time.

### E.3 The empirical-Bayes level

The empirical-Bayes level adapts in the right direction. At budget 1 it moves most of the way from the public centre toward each regime's level. Among the date-shifted test-like regimes it orders the regimes by their levels, and the date shift itself moves it further down (F§ "The empirical-Bayes level adapts the right way"). It ties the fixed configuration because the level fixed at $B_0$ decides most of the gain.

We did not ship it, because it is about 70 lines of experiment code outside the library and its worst-case call is slower. In the RS study above, at the feedback's reading, it also gains nothing over the fixed level, so the decision rests on measurement as well.

![Figure E1: the empirical-Bayes level centre by budget](fig/eb_adaptation.png)

*Figure E1.* The empirical-Bayes (EB) estimate of a new benchmark's level centre by budget, coloured by each regime's mean pair logit. Dashed lines are public regimes. (a) EB on hier-fit's level prior in the level calibration's regimes. (b, c) hier-EB on hier-ship and on $\mu_0 = -3.0$, $\gamma = 0.5$ in the RS regimes (fresh seeds, library of `4d2cc4f`). The centre is on the item-level scale, and the regimes' levels are on the pair-accuracy scale. The two are ordered alike within regimes that share the date shift. The date shift moves the centre further down because the centre has to offset the inflated attribute standings (F§ "The empirical-Bayes level adapts the right way"), so a date-shifted regime ends below an unshifted one of the same or a lower level.

### E.4 The baseline study: provenance and details

**Smooth-cal in §5.3.** The RS study added Smooth-cal's row to §5.3's table (`results/regime_sensitivity.json`, `smcal`) and scored it on the same runs as the rest of the table. Its difference from LegacyP is derived from the two ALCs (0.1588 for Smooth-cal and 0.2039 for LegacyP). Its public cost is measured against LegacyP's stored rows, which predate the corrected multiple-choice floor. Against the current LegacyP the cost is larger (F§ "Regime sensitivity at the feedback's reading").

**Row provenance of §5.5.** The baseline study scored a plain 1PL and EmpMean on the RS study's TUNED and public runs (F§ "Baselines: a plain 1PL, the organisers' empirical mean and BLE"; `experiments/baselines_p1.py`, `results/baselines_p1.json`). The 1PL is hier with the attributes, the identity link, the groups, the multiple-choice floor and slip switched off. It has one pooled level per benchmark, one ability per (subject, benchmark) pair and one difficulty per item, refitted from `labeled` at every checkpoint. We scored it at the shipped level and at its own leave-one-parent-out public level.

Every other row is the RS study's. We re-scored Smooth and Smooth-cal on every run, and hier-ship, hier-nosubj and LegacyP on the first and last run of each regime. The re-scored values match the RS study's bit for bit. For hier-fit we read the RS study's rows as stored.

**Smooth-cal against hier-ship.** A calibrated level alone recovers the tuned-regime gain. Smooth-cal takes the prior that LEVEL's own selection rule picks on the tuned regime, mean $m_0 = 0.25$ and strength $n_0 = 2$. No grid point passed the public guard, so this is the best point regardless of the guard. In TUNED, Smooth-cal is within noise of hier-ship (+0.0015 ± 0.0019 cluster SE, in Smooth-cal's favour), and it beats LegacyP there by 0.045 with no item model and no subject prior.

Elsewhere Smooth-cal trails hier-ship: by 0.004 to 0.010 in the four other test-like regimes (hier-ship minus Smooth-cal is -0.0043 to -0.0103), and by 0.010 to 0.011 on public runs, where it also fails the guard. The tuned-regime headline therefore measures a moved level. What hier adds over a calibrated level is the gain it keeps when the hidden level is not the one it was tuned for, and its parity with LegacyP on public runs.

**The subject prior.** Without its attribute prior, hier-ship becomes hier-nosubj, which keeps the pooled level, item difficulty, groups and floor. In TUNED it is ahead of hier-ship by 0.0030, and elsewhere it is behind by 0.0016 to 0.0049. Turning the prior off also lowers the $B_0$ centre: the mean $B_0$ prediction is 0.31, against 0.42 for hier-ship. This comparison therefore measures the prior and hier-ship's centre together. The plain 1PL is a different model, 1PL-ship (§5.5).

**LegacyP's deficit.** Most of LegacyP's tuned-regime deficit is its own prior. In TUNED, 1PL-fit, with no calibration at all, beats LegacyP by 0.038 (cluster SE 0.004), and hier-fit beats it by 0.012. LegacyP's mean $B_0$ prediction in TUNED is 0.68, against 0.42 for hier-ship: its attribute prior turns the synthetic date shift into optimism. Even EmpMean is ahead of LegacyP in that regime, by 0.0095 (-0.0095 ± 0.0025 / 0.0049 / 0.0043, 1.9 cluster SEs; parent-level -0.003 ± 0.012). The tuned regime's headline is therefore set largely by the date shift (§7).

**Public runs.** On public runs the calibration costs ALC, and the model makes up for it. There hier-ship minus LegacyP is within noise (-0.0002 and -0.0039), so we do not split it into the calibration's share and the model's share. At the same level, hier adds 0.010 over Smooth-cal. It adds 0.005 over 1PL-ship, with differences of -0.0048 ± 0.0005 / 0.0007 / 0.0006 on benchmark-first runs and -0.0050 ± 0.0004 / 0.0007 / 0.0006 on pair-uniform runs (run / cluster / stratified SE). About half of that gain is the subject prior (hier-ship minus hier-nosubj), and half is the groups, the floor, slip and the standing keyed on the model's name (hier-nosubj minus 1PL-ship).

**The attribute prior at $B_0$.** Where the mean $B_0$ predictions match, the attribute prior pays: hier-ship beats 1PL-ship by 0.010 to 0.015 of Brier at $B_0$ and by 0.004 to 0.009 at $B_1$ (cluster SEs 0.0009 to 0.0026). Of the $B_0$ gap, about 0.0015 separates hier-nosubj from 1PL-ship, and the rest is mostly the attribute prior.

**EmpMean.** In TUNED, EmpMean trails hier-ship by 0.0339 ± 0.0016 / 0.0030 / 0.0028 (run / cluster / stratified SE), and on public runs it trails by 0.045 to 0.046 (cluster SEs 0.003).

TUNED redraws the catalogue that chose LEVEL and Smooth-cal, so TUNED favours hier-ship and Smooth-cal. It also favours 1PL-ship, which borrows hier's calibrated $\mu_0$, chosen with the attribute prior on.

### E.5 BLE

We could not run BLE, and we defer it together with the empirical mean with BLE acquisition. Each BLE prediction is a language-model agent run against a paid API, by default openai/gpt-5.6-luna, with up to 10 turns and 240 s per prediction. Running BLE needs network access, an `OPENAI_API_KEY` for the paid model, and a payload prepared over the network and filtered per held-out parent. On the baseline study's evaluation runs alone it would take about 570,000 agent runs (569,838 exactly). BLE's only offline mode replaces the model with scripted replies.

The empirical mean with BLE acquisition chooses its labels from BLE's predictions, so it cannot run either. Under random acquisition it is EmpMean, scored above. The best leaderboard entry (0.1172) stays unexplained.

We did not run amortized calibration [Truong et al. 2025] either. Its core is a difficulty map from embedded item content, trained across datasets. App F.2 measures such a map leave one benchmark out, and the map does not transfer here.

### E.6 The single-subject benchmark

The benchmark swe_rebench has one subject. Its numbers carry an SE over its appearances, and each appearance sees a different subset of its 6,306 items through the run's item cap (mean pairwise overlap 0.026). That SE is not a cluster SE, and nothing measures variation between subjects or between single-subject benchmarks (F§ "A single-subject benchmark"; `experiments/gate_and_ci.py --stage single`, `results/gate_and_ci.json`, `single`).

On swe_rebench hier-ship loses to both LegacyP and Smooth. Against LegacyP it loses +0.0101 ± 0.0012 over 130 R1-bf appearances (run-2 library, seed 0) and +0.0094 ± 0.0016 over 51 (current library, the RS study's seed 11). Against Smooth the two losses are +0.0130 ± 0.0014 and +0.0117 ± 0.0018. Pair-uniform runs hold 7 and 9 appearances, with losses of +0.0121 ± 0.0065 and +0.0024 ± 0.0042 against LegacyP.

The loss sits at $B_0$ to $B_3$ (+0.024, +0.018 and +0.013 against LegacyP) and stays under 0.006 from $B_7$ on. Of hier-ship's -0.0017 against LegacyP on R1-bf, swe_rebench carries +0.0010.

Among the RS study's alternatives, hier-ship is near the best on swe_rebench. It does better than both wider level priors, hier-rec, Smooth-cal, hier-EB on $\mu_0 = -3.0$ and hier-nosubj, by 0.0009 to 0.0073. The wider priors gain at $B_0$ and lose from $B_1$ on. Besides LegacyP and Smooth, only hier-fit does better (by 0.0022 ± 0.0010).

Test-like runs exclude single-subject benchmarks, so the tuned-regime gains say nothing about them, and swe_rebench's success rate (about 0.49) sits near the public centre.

### E.7 hier-ship across regimes by budget

| regime | runs (seed) | $B_0$ | $B_1$ | $B_3$ | $B_7$ | $B_{15}$ | $B_{31}$ | ALC ± run SE | single-run sd of ALC |
|---|---|---|---|---|---|---|---|---|---|
| TL | 300 (2) | 0.2165 | 0.1892 | 0.1645 | 0.1525 | 0.1450 | 0.1386 | 0.1658 ± 0.0018 | 0.0318 |
| TL-mix | 200 (3) | 0.2178 | 0.1931 | 0.1692 | 0.1590 | 0.1524 | 0.1454 | 0.1711 ± 0.0023 | 0.0329 |
| TL-noshift | 100 (3) | 0.1948 | 0.1763 | 0.1620 | 0.1479 | 0.1413 | 0.1353 | 0.1585 ± 0.0033 | 0.0335 |
| R1-bf | 150 (0) | 0.2350 | 0.2243 | 0.2128 | 0.1973 | 0.1852 | 0.1771 | 0.2051 ± 0.0023 | 0.0285 |
| R1-pu | 100 (0) | 0.2442 | 0.2215 | 0.2069 | 0.1929 | 0.1813 | 0.1708 | 0.2020 ± 0.0026 | 0.0261 |
| platform, formative run 2 | 1 | 0.2368 | 0.1947 | 0.1963 | 0.1836 | 0.1785 | 0.1833 | 0.1926 | |
| platform, formative run 3 (archive `4a882cc7…`) | 1 | 0.2351 | 0.1868 | 0.1885 | 0.1664 | 0.1667 | 0.1644 | 0.1817 (recomputed) | |

Source: `results/subject_side.json` (`summary.regimes.*.ship`), `experiments/subject_side.py`; single-run sds from `results/ship_confirm.json` (`single_run_sd`); run 2 from `results/formative_feedback.json`, run 3 from `results/formative_run3.json`. The 'ship' arm reproduces `experiments/itemsig_eval.py`'s base exactly.

The comparisons with LegacyP and Smooth below come from `results/ship_confirm.json`.

Most of the gain over LegacyP is at $B_0$ and $B_1$. On TL runs 0 to 299, hier-ship gains 0.135 of Brier (cluster SE 0.011) at $B_0$, 0.076 at $B_1$, 0.037 at $B_3$, 0.016 at $B_7$ and 0.008 at $B_{15}$ and $B_{31}$. Weighted into ALC, the gains at $B_0$ and $B_1$ carry 0.029 of the 0.042 ALC gain.

Over Smooth, which has no level to tune, hier-ship gains 0.016 to 0.018 in the tuned regime and 0.009 on public runs. The gain is larger without the date shift (0.022) and positive on every test-like parent.

On public runs hier-ship is within noise of LegacyP. Its point estimates are 0.0015 to 0.0018 better, within about one cluster SE (0.81 cluster SEs on R1-bf and 1.02 on R1-pu). It loses at $B_0$ (+0.0049 ± 0.0065 on R1-bf and +0.0140 ± 0.0072 on R1-pu, cluster SEs) and gains from $B_7$ on (0.004 to 0.008 of Brier per budget). It also loses on public matharena (+0.010 and +0.009 on the two weightings) and on the single-subject swe_rebench pair (+0.010; App E.6).

---

## Appendix F: the studies of §6

### F.1 Summary table and the within-pair scale

Each row of the table below is a negative result, with the script that produced it. All these ideas fail for the same reason: the hidden test's headroom lies at budgets 0 and 1 and in the per-item structure of benchmarks nobody has seen, and none of these signals reaches either.

We scored the label-conditional layer, the meta-learned heads, the known-sign cues, the subject-side priors and the language-model probes that produce an item covariate against hier-ship, leave one parent out. The TF-IDF map is a data-level correlation with a naive solve-rate target. The embeddings and the blind ratings are data-level correlations with Rasch difficulty, as are the probes' own correlation prongs. App F.8's results come from the legacy replica.

Every transfer correlation of a text map, judge, probe head or encoder, and of the 14B, carries a 95% interval (F§ "Intervals for every transfer correlation"; the probe heads' and the encoder's intervals come from their own results files), and a second table below restates them on the gate's within-pair scale. The exceptions are the cases where no interval is computable: swe_rebench on the harness scales, the blind ratings' within-pair $r$, the blind ratings on researchcodebench, the 4B judge on real_webagents and researchcodebench, and the 14B's attempts outside matharena (F§ "Intervals for every transfer correlation", `ci.not_computable`). The column "best honest estimate" holds a correlation, an accuracy, an ALC difference or a count, as each cell says. A signed ALC difference is the variant's ALC minus its comparator's, so it is negative when the variant helps; the comparator is hier-ship except in the legacy-replica studies. An unsigned "gains" or "loses" gives the size of an improvement or a loss. A *line* is one configuration scored by the harness, and its value is such a difference.

| idea | best honest estimate | verdict | script, results |
|---|---|---|---|
| item difficulty from TF-IDF text, across benchmarks | LOBO Pearson $r$ with Rasch difficulty on text-bearing items 0.10, 0.14, -0.02, -0.16, 0.11, every 95% interval below 0.3 (0.14, 0.23, 0.09, -0.22, 0.16 with the naive solve-rate target); within a test-like pair 0.07 | does not transfer | `experiments/transfer.py` (data-level); intervals `results/gate_and_ci.json` |
| item difficulty from Qwen3-Embedding-0.6B | LOBO ridge -0.16 / -0.03 / -0.23 / -0.02 (upper 95% limits 0.26 at most); within benchmark 0.66 / 0.27 / 0.38 / 0.51 | does not transfer | `experiments/emb_transfer.py`, `results/emb_transfer.json` |
| blind LLM difficulty rating (rater undocumented, App F.3) | 0.482 [0.25, 0.64] on matharena; -0.13 to 0.21 elsewhere | mathematics only; inconclusive on multi_swebench and real_webagents, interval below 0.3 on swe_rebench | `experiments/llm_rating/analysis.py`; group intervals `results/gate_and_ci.json` |
| Qwen3-4B zero-shot judge (rating, digit, entropy, and nll, the negative log-likelihood of the task text) | no benchmark-equal test-like ALC difference (the mean of the parents' means) reached -0.001 | closed | `experiments/llm4b_close.py`, `results/llm4b_close.json` |
| a 4B model's own attempts | graded accuracy 2.3% and 7.8%, below the 10% floor | cannot be read | `experiments/attempt_probe.py`, `results/attempt_probe.json` |
| entropy and hidden-state heads | LOBO $r$, random-effects mean over the four parents, -0.05 [-0.24, 0.14] (entropy) and +0.12 [-0.004, 0.24] (hidden state) | replicated null | `experiments/hidden_state_probe.py`, `results/hidden_state_probe.json` |
| in-context learning over the pair's labels | $r$ of 0.206 ± 0.040, below 0.3 | closed | `experiments/icl_probe.py`, `results/icl_probe.json` |
| pairwise (anchored) comparisons | pooled q 0.540, below 0.60 | dropped | `experiments/pairwise_probe.py`, `results/pairwise_probe.json` |
| text-similarity residual layer (itemsig) | nested ALC difference -0.00002 ± 0.00002 / 0.00008 / 0.00008 | no gain | `experiments/itemsig_eval.py`, `results/itemsig_eval.json` |
| meta-learned heads on frozen embeddings | nested ALC difference 0 (every head off in every fold); forced +0.0002 to +0.0006 | no gain | `experiments/heads_eval.py`, `results/heads_eval.json` |
| fine-tuned encoder | 0 of 4 parents at held-out $r \ge 0.3$, best +0.05 [-0.01, 0.10] (multi_swebench, group interval), random effects -0.05 [-0.14, 0.05]; transferred line +0.00048 | closed | `experiments/finetune_encoder.py`, `results/finetune_encoder.json` |
| Qwen3-14B demand rubric and direct ratings (Kaggle, two T4s) | LOBO head $r$ +0.19 (random effects), per parent +0.23 [0.04, 0.40], +0.01 [-0.08, 0.11], +0.19 [0.11, 0.28], +0.33 [0.12, 0.51] (-0.04 within competition on matharena); within-pair $r$ 0.11; best nested line -0.00084 ± 0.00043 | null for ALC | `experiments/strong_llm_eval.py`, `results/strong_llm_eval.json` |
| Qwen3-14B reasoning attempts (matharena only) | within-competition Spearman $\rho$ 0.372 [0.209, 0.516] on the 147 probe texts, 0.352 [0.25, 0.45] on all 270 attempted; forced per-pair line -0.00011 ± 0.00005 | GO for correlation; NULL for ALC (one parent) | same |
| Qwen3-14B reasoning entropy, all four parents (Kaggle, the entropy job) | within-group Spearman $\rho$ +0.29 [0.16, 0.40] / +0.12 [0.06, 0.16] / -0.03 [-0.19, 0.15] / +0.33 [0.17, 0.45] (matharena, multi_swebench, real_webagents, researchcodebench), random effects +0.18; within-pair $r$ 0.12 [0.09, 0.15]; transferred nested +0.00110 ± 0.00045 (on in 1 of 4 folds), per-pair nested +0.00001 ± 0.00006 | NULL, under a rule fixed before the data | same (`entropy`) |
| known-sign item cues (ordinal fields, position, format, length, stated size, image refs) | nested test-like ALC difference 0 to +0.00003 | none passes the sign rule or the gate | `experiments/itemcov_eval.py`, `results/itemcov_eval.json` |
| subject side: harness identity, ordered effort, date forms, Student-t level | best nested line -0.00097 ± 0.00033 (ordered effort) | none passes | `experiments/subject_side.py`, `results/subject_side.json` |
| acquisition policies (legacy replica) | coverage-first, ALC difference -0.0000 ± 0.0013 | none beats random | `experiments/acquisition.py` |
| post-hoc temperature and slip (legacy replica) | loses 0.0007 ± 0.0011 | does not transfer | `experiments/acquisition.py` |
| offline bank of per-item results | 1 of 161 inventory benchmarks judged usable (the judgement is unrecorded, App F.8) | closed | `experiments/inventory_scan.py` |

SEs: the itemsig triple is run / cluster / stratified SE; the ± of a rubric, attempt or entropy line is the cluster SE, and the ± of the in-context $r$ is the SE over pair appearances.

**One correlation scale.** The gate's $r$ is a correlation with an *honest* difficulty, fitted on other subject folds. That difficulty is reliable (0.84 to 0.95 by split-half and fold-overlap estimates), so a correlation against full-sample difficulty is only 0.5% to 1.6% higher than one against the honest difficulty. The gate's $r$ and most correlations in §6.3 differ instead in the unit over which they are taken. The gate reads a covariate within a test-like pair, where its bar is a within-pair $r$ of 0.25 (0.33 at an honest $r$ of 0.4), while those correlations are taken over a benchmark's whole range, within groups, or over text-bearing subsets.

The share of its correlation that a covariate keeps within a pair varies. The share is the within-pair $r$ divided by the *parent-scale* $r$, the Pearson $r$ with honest difficulty over a parent's items. A degraded oracle keeps 0.82 of its $r$ within a pair. The 14B's primary head, judged solve share, rubric sum and expert-time estimate, and its reasoning entropy, keep 0.57 to 0.65. Every covariate whose parent-scale correlation is clear of zero keeps 0.30 to 1.02. For TF-IDF, the 4B judge and the embedding kNN that correlation is near zero, so the share means nothing (`results/gate_and_ci.json`, `ci.unit_ratio`). The table gives the covariates on the gate's within-pair scale (`results/gate_and_ci.json`, `scale`):

| covariate | within-pair $r$ [95% cluster interval] |
|---|---|
| TF-IDF, leave one benchmark out | 0.07 [0.03, 0.10] |
| embedding ridge; kNN, leave one benchmark out | -0.11 [-0.13, -0.08]; -0.04 [-0.06, -0.02] |
| embedding ridge fitted within the same benchmark (cannot transfer) | 0.25 [0.23, 0.27] |
| 4B judge, four features (absolute value; the harness fits the sign) | 0.04 to 0.09 |
| 14B rubric: primary head; judged solve share; the two highest scales | 0.11 [0.08, 0.14]; 0.13 [0.10, 0.16]; 0.15 [0.12, 0.17] |
| 14B reasoning entropy, all four parents | 0.12 [0.09, 0.15] |
| 14B attempts' entropy (matharena only; it varies on 20% of appearances) | 0.32 [0.27, 0.35] |

No covariate that can be carried to a new benchmark reaches 0.25 within a pair. The attempts' entropy reaches it on one parent, where no slope can be transferred leave one parent out.

A slope transferred from other benchmarks acts at $B_1$ about as much as at later budgets: the honest oracle's transferred line gains 0.036 at $B_1$ (cluster SE 0.004; `results/gate_and_ci.json`, `gate.b1`).

### F.2 Item difficulty from text: TF-IDF and neural embeddings

With one benchmark left out, a TF-IDF map from item text correlates with the naive difficulty target (the solve-rate logit over all items) at 0.14 [0.06, 0.23] on matharena, 0.23 [0.16, 0.27] on multi_swebench, 0.09 [-0.13, 0.31] on real_webagents, -0.22 [-0.47, 0.07] on researchcodebench and 0.16 [0.13, 0.18] on swe_rebench. The brackets are 95% group-bootstrap intervals, except on swe_rebench, which has no groups and gets an item interval. Against Rasch difficulty on the items that carry text, which is the target the rest of this section uses, the correlations are 0.10 [0.01, 0.20], 0.14 [0.09, 0.23], -0.02 [-0.40, 0.28], -0.16 [-0.44, 0.10] and 0.11 [0.09, 0.14] (F§ "Intervals for every transfer correlation"; `results/gate_and_ci.json`, `ci.rows`). Within a test-like pair, where the gate reads it, the correlation is 0.07 [0.03, 0.10].

Within matharena (5-fold, text-bearing items) the same model reaches 0.80 [0.70, 0.86], but much of that comes from identifying which competition an item belongs to: removing the competition mean drops it to 0.58 [0.48, 0.66]. (The findings document first recorded 0.73 and 0.49 here; those values do not reproduce with the current code, and we quote the refitted ones.) Competition alone explains 40% of the variance of this difficulty target (F§ "What transfers between benchmarks"). On the Rasch scale, the variance component of §2.2 puts the competition's share at 0.50.

Qwen3-Embedding-0.6B embeddings do no better across benchmarks (F§ "Neural embeddings do not carry difficulty to an unseen benchmark"; 95% group-bootstrap intervals from `results/emb_transfer.json` and `results/gate_and_ci.json`):

- LOBO ridge correlations are -0.16 [-0.29, -0.03], -0.03 [-0.16, 0.04], -0.23 [-0.43, -0.01] and -0.02 [-0.31, 0.26] (matharena, multi_swebench, real_webagents, researchcodebench), and kNN correlations run from -0.08 [-0.21, 0.06] to +0.16 [-0.11, 0.38].
- $R^2$ as predicted is at most 0 everywhere, and the carried slope has the wrong sign.
- Within a benchmark (5-fold) the embeddings reach 0.66 [0.56, 0.73], 0.27 [0.20, 0.34], 0.38 [0.19, 0.52] and 0.51 [0.30, 0.66]. But the item_features group mean alone reaches 0.57 [0.39, 0.69], 0.12 [-0.06, 0.23], 0.41 [0.19, 0.55] and 0.52 [0.30, 0.67], and with whole groups held out the embeddings fall to 0.40 [0.29, 0.51], 0.16 [0.09, 0.25], -0.02 [-0.19, 0.15] and -0.09 [-0.30, 0.11]. So within a benchmark the embedding mostly identifies the group, which hier already learns from labels.

![Figure F1: item difficulty does not transfer](fig/transfer.png)

*Figure F1.* Pearson $r$ between predicted and full-sample Rasch item difficulty, per benchmark, with 95% group-bootstrap intervals. The figure compares leave-one-benchmark-out maps (TF-IDF+SVD ridge, embedding ridge, embedding kNN) with within-benchmark fits (embedding ridge, the item_features group mean, embeddings with whole groups held out). The TF-IDF bars show `experiments/emb_transfer.py`'s TF-IDF+SVD ridge, which differs from the `transfer.py` map quoted above. The dashed rule is the gate's honest $r$ of 0.3; within a test-like pair the bar is 0.25 (§6.1).

*Literature.* Amortized calibration fits one map from embedded question content to difficulty across 22 datasets and matches per-question calibration on held-out questions [Truong et al. 2025]. It does not test a held-out dataset. ADeLe reports that black-box predictors built on embeddings or fine-tuning are weaker than rubric-based demand levels, especially out of distribution [Zhou et al. 2026]. Our result is the out-of-distribution end of that picture on agentic and code benchmarks: the direction of difficulty in embedding space is not shared between benchmarks. For agentic coding tasks, Agent Psychometrics predicts task-level success on unseen benchmarks from issue statements *together with* repository context, solutions and test cases [Ge et al. 2026]. The competition's items carry the issue text and grouping metadata only.

### F.3 Language-model difficulty judgements

A language model rated 180 items across four benchmarks on a scale of 0 to 100, blind to their difficulty, and we compared the ratings with Rasch difficulty (F§ "Language-model difficulty judgement").

| benchmark | n | Pearson $r$ | 95% CI (Fisher) | 95% CI (group bootstrap) |
|---|---|---|---|---|
| matharena | 45 | 0.482 | [+0.22, +0.68] | [+0.25, +0.64] |
| multi_swebench | 45 | 0.133 | [-0.17, +0.41] | [-0.17, +0.48] |
| real_webagents | 45 | 0.210 | [-0.09, +0.47] | [+0.12, +0.41] |
| swe_rebench | 45 | -0.128 | [-0.41, +0.17] | (no groups) |
| pooled within benchmark | 180 | 0.174 | [+0.03, +0.31] | [+0.03, +0.33] |

Group intervals resample item_features groups (`experiments/gate_and_ci.py --stage ci`, `results/gate_and_ci.json`), and the Fisher intervals reproduce. A control on 60 fresh multi_swebench items with the full issue text (capped at 4,000 characters) gives Pearson $r$ +0.252 (interval touching zero) and Spearman $\rho$ +0.109, so truncation was not the explanation.

**The rater's prompt and protocol are undocumented.** The ratings are hard-coded in `experiments/llm_rating/ratings_main.py` and `ratings_control.py`. Both entered the repository in its first commit (2026-09-23), co-authored by Claude Opus 5 in a Claude Code session. The team confirms that the rater was the Claude model of that coding session (the commit's trailer names Claude Opus 5). Its prompt, the date and the sampling settings are recorded nowhere. The rater was blind to difficulty only: item ids carry the benchmark, and the texts name their domain. We cannot exclude that it saw measurement-db (which that session held) or the benchmarks' public leaderboards. The correlations re-derive from the stored ratings; the ratings themselves do not (App G.6).

*Literature.* Human-labelled difficulty of math and coding problems is linearly decodable from model activations [Lugoloobi and Russell 2025], and rubric-based LLM annotation of task demands predicts instance-level performance [Zhou et al. 2026]. We see the judge track difficulty where it is visible in the statement (competition mathematics) and not where it lies in the environment (repository size, files to touch, test harness).

The gate table (§6.1) needs an honest $r$ of 0.3 to 0.35 across benchmarks, and 0.25 within a test-like pair. The judge reaches 0.3 against full-sample difficulty on one benchmark of four. Full-sample difficulty does not flatter the correlations of §6.3: the honest difficulty is reliable (0.84 to 0.95), and a correlation against full-sample difficulty is at most 1.6% higher than one against the honest difficulty. With 45 items per benchmark the intervals on multi_swebench and real_webagents include 0.3, so on those two the result is inconclusive rather than negative. On swe_rebench the interval, [-0.41, +0.17], excludes 0.3.

**A larger 4B run is not needed to settle the question.** The review asked for the local 4B judge on several hundred items per benchmark. After that request, the 14B of App F.9 rated every item of the four parents, with the same judged solve share, so the item count does not limit the answer (F§ "Intervals for every transfer correlation"; `results/gate_and_ci.json`, `ci.power_4b`). Excluding $r = 0.3$ at 80% power would take 258 items on multi_swebench, but 847 on real_webagents, which has 233 items in all.

Where items are plentiful, both local judges sit below the bar, with intervals that exclude it. On multi_swebench the 14B's solve share reaches 0.07 [-0.04, 0.19] (Pearson $r$ with fold-averaged difficulty), and the 4B's rating reaches 0.11 [0.06, 0.21] (Spearman $\rho$ within language, sign as declared; the stored raw rating's is -0.11 [-0.21, -0.06]). Figure 5 (c) draws a different statistic from the same file: the 4B rating's Pearson $r$ over all rated items in the same orientation, 0.11 [0.06, 0.18] (`results/llm4b_close.json`, `signs.features.rating.units.multi_swebench`). Where items are few, the 14B's solve share is 0.34 [0.13, 0.52] on researchcodebench and 0.18 [-0.03, 0.39] on real_webagents against fold-averaged difficulty, and it still fails the gate: within a test-like pair it reaches 0.13, and its best nested line is -0.00084. More 4B items could narrow an interval, but they could not change the verdict on a covariate that the 14B, rating every item, does not carry through the gate.

A fully specified local judge, Qwen3-4B-Instruct-2507, read each item once and rated it as a digit (`experiments/llm_features.py`). We closed it (F§ "The 4B judge, closed out"): no benchmark-equal test-like difference reached -0.001 through the harness, and on text-bearing matharena items every feature's partial correlation was below 0.2 with the declared sign. Its extraction covers matharena and 1,941 of 2,078 multi_swebench items only.

### F.4 A label-conditional text-similarity layer

The layer in `paiec/itemsig.py` carries hier's residuals on a pair's labelled items to unlabelled items with similar text. We ran it over a 480-configuration grid and redid the nested leave-one-parent-out selection in every bootstrap resample (F§ "Item signal from the pair's own labels").

Base ALCs are in parentheses, and the nested column gives ± run / cluster / stratified SE. Here R1-bf is runs 0 to 199 (0.2057), while App E.7 uses runs 0 to 149 (0.2051).

| regime (base ALC) | nested, selected within the regime | in-sample best of 480 |
|---|---|---|
| TL, runs 0-299 (0.1658) | -0.00002 ± 0.00002 / 0.00008 / 0.00008 | -0.00006 |
| TL-mix, runs 0-199 (0.1711) | -0.00028 ± 0.00005 / 0.00021 / 0.00019 | -0.00037 |
| R1-bf, runs 0-199 (0.2057) | -0.00003 ± 0.00012 / 0.00049 / 0.00046 | -0.00071 |
| R1-pu, runs 0-99 (0.2020) | -0.00076 ± 0.00018 / 0.00031 / 0.00029 | -0.00112 |

The layer cannot act at $B_0$ or $B_1$ by construction: there are no labels at $B_0$, and at $B_1$ centring within the pair zeroes the single residual. At $B_{31}$ it takes 0.5% of the gap between the pair-rate oracle and the item oracle. That gap is 0.063 of Brier on test-like runs, and the base already matches the pair-rate oracle there. Selected on the public runs of the other parents, the layer picks aggressive settings that lose on held-out researchcodebench (+0.0019 ± 0.0007). The submission does not include it.

*Literature.* Generic assessors that transfer instance-level performance from reference instances do well in distribution, but out of distribution "no clear winner emerges and the overall performance is worse" [Pacchiardi et al. 2024].

### F.5 Meta-learned heads on frozen embeddings

In an episode-trained study we put linear and low-rank heads on frozen Qwen3-Embedding-0.6B features, on top of hier-ship: a meta-learned difficulty direction, a learned-metric few-shot kernel, and an assessor-style subject-by-item term (F§ "Meta-learned heads on frozen embeddings"; `experiments/heads_eval.py`, `results/heads_eval.json`). Nested leave-one-parent-out chose the regularisation strength $\lambda$, and a head was used on the held-out parent only if its inner mean was below 0.

Nested, every head is off in every fold, both on the rows the study ran on and on the rows of the current library, so the nested difference is exactly 0. The smallest inner mean over folds and values of $\lambda$ is +0.000007. Forced on, the combined head costs +0.0002 to +0.0006 test-like.

The heads learn something where the benchmark has been seen, with subject folds inside the training benchmarks. There the best two of five configurations (the difficulty head at $\lambda = 0.001$, and all heads at PCA 256) gain 0.0005 and 0.0017 test-like and 0.0024 and 0.0046 TL-mix. Across all five the range is +0.0005 to -0.0017 test-like, where both $\lambda = 0.01$ variants lose to hier, and -0.0008 to -0.0046 TL-mix. None of it reaches a held-out parent.

The same head on in-sample Rasch difficulty gives -0.0437 test-like (cluster SE 0.0037) and -0.0557 TL-mix, which is the acceptance target the harness reproduces (App C.4).

The study first ran from a session's scratch files, and the script reproduces all 21 of its result files bit for bit.

### F.6 Item covariates with a known sign

The module `paiec/itemcov.py` reads benchmark-agnostic cues off the item dict, each with its sign declared in advance: ordinal difficulty fields, problem position, answer format, text length, stated amount of work and image references. The plan allowed a transferred slope only if a cue had the declared sign *and* agreed with the other benchmarks' mean on 4 of 5 units, the benchmarks of the sign-check table below (F§ "Item covariates with a known sign"; computed on the legacy harness rows, which predate the corrected multiple-choice floor and the solver's floored-fit fix). Three facts decide the result:

- Most cues cannot be tested across benchmarks. No public item carries a difficulty-named field, and position, format and stated size each vary on one benchmark only.
- The one universal cue is unstable. Text length is present everywhere but changes sign: +0.28 on matharena, +0.36 on real_webagents, -0.43 within paper on researchcodebench, whose item_features groups are papers.
- Nothing passes the gate: through the harness, nested test-like differences are between 0 and +0.00003.

The one cue with real item signal is researchcodebench's stated size ("Approximately 7 line(s) of code"). It has a within-pair $r$ of +0.41, and a forced per-pair slope from $B_7$ gives -0.0022 ± 0.0005 (cluster SE) on its pairs, against a placebo of +0.0008. That sits where the gate table says a covariate with an $r$ of 0.3 to 0.5 should. But the cue exists on one public benchmark, so it cannot be selected leave-one-parent-out. If one hidden benchmark in seven carried such a cue, the run-level value would be about 0.0003.

*Literature.* Task length in human time predicts agent success on software tasks [Kwa et al. 2025], and stated lines of code is a crude, benchmark-specific proxy for it. Agent Psychometrics finds that repository state, tests and the solution patch add predictive power beyond the issue text [Ge et al. 2026], but none of these is visible in the competition's item input.

**Sign check.** The table gives each covariate's Spearman $\rho$ with honest difficulty, ± a bootstrap SE over item_features groups (over items on swe_rebench), with the $\rho$ within item_features groups in parentheses (F§ "Item covariates with a known sign"):

| covariate | matharena | multi_swebench | real_webagents | researchcodebench | swe_rebench |
|---|---|---|---|---|---|
| position_within | +0.12 ± 0.05 (+0.15) | absent | absent | absent | absent |
| format_score | +0.19 ± 0.13 (+0.07) | constant | constant | 3 items off | 3 items off |
| log_length | +0.28 ± 0.08 (+0.06) | -0.02 ± 0.05 (+0.00) | +0.36 ± 0.08 (+0.29) | -0.10 ± 0.17 (-0.43) | +0.04 ± 0.01 |
| stated_size | absent | absent | absent | +0.50 ± 0.06 (+0.43) | absent |
| image_ref | -0.04 ± 0.09 | +0.05 ± 0.03 (+0.05) | absent | -0.08 ± 0.15 | +0.05 ± 0.01 |

### F.7 The subject side at budgets 0 and 1

Against hier-ship we measured four changes to the priors that act at $B_0$ and $B_1$ (F§ "Subject side at budgets 0 and 1"): harness identity (H), ordered reasoning effort (E), the form of the release-date trend (D: clip, hinge, both, log) and a Student-t level (T).

| selection | folds on | TL | R1-bf / R1-pu | TL-noshift | worst held-out parent |
|---|---|---|---|---|---|
| E vs hier-ship | 4 | -0.00097 ± 0.00033 | -0.00002 / +0.00001 | -0.00005 | 0 |
| H vs hier-ship | 1 | +0.00008 | +0.00005 / +0.00008 | +0.00005 | +0.0004 |
| Dlog vs hier-ship | 3 | +0.00033 | +0.00048 / +0.00056 | +0.00050 | +0.0044 |
| all ten, selected on test-like alone | E 4, Dlog 3 | -0.00052 ± 0.00118 | +0.00047 / +0.00058 | +0.00043 | +0.0044 |

None passes the gates:

- H cannot be measured leave-one-parent-out: harness strings exist on one multi-subject benchmark. Within multi_swebench the harness is the largest attribute (+0.75 for Agentless, -0.75 for MSWE-agent), and Ge et al. likewise find a sizeable additive scaffold component, with Agentless above SWE-agent [Ge et al. 2026]. That is within-benchmark transfer, which is not what the hidden test asks for.
- E is switched on in every fold but gains half the bar, and all of that gain comes through a lower date slope under the synthetic date shift.
- The date form is decided by the shift rather than by the data. On real dates the hinge fits held-out standings best and gains 0.0004 to 0.0007; under the shift it loses.
- The Student-t level ties or loses, at five to nine times the cost.

*Literature.* Observational scaling laws predict benchmark performance from a few capability dimensions shared across model families [Ruan et al. 2024], which supports putting subject attributes in the prior. No public run can check how far hidden subjects lie past the public release dates. That distance sets the $B_0$ error; the attribute model does not.

### F.8 Acquisition, calibration and the offline bank (legacy replica)

We measured these on the legacy replica and have not re-measured them under the official protocol.

A-optimal acquisition targeting the evaluation pool was worse than random, because it starved the shared difficulty estimate of distinct items. Coverage-first selection moved ALC by -0.0000 ± 0.0013 (F§ "Acquisition and calibration"). The submission ships no `labeling.py`.

Post-hoc temperature and slip, fitted on four benchmarks and applied to the fifth, lose 0.0007 ± 0.0011. The fitted temperatures range from 0.8 to 1.6, so the constant is itself a property of the benchmark.

Of the organisers' 161 inventory benchmarks, 5 publish per-item outputs for at least nine models, and only one (capability) was judged to publish them graded (F§ "The offline bank"). The record does not say how that judgement was made for the five (phyblock, mmdocrag, atmossci_bench, capability, engdesign): `results/strong_repos.csv` has no producing script, and the step may have inspected their files (App G.6). The rules also restrict competition-specific training to the public pool. App G.6 states what the scan read and that nothing from it entered any model or selection.

*Literature.* Active and anchor-point selection choose informative items for a population of models [Li et al. 2025; Vivek et al. 2024; Maia Polo et al. 2024]. Here the evaluation items are a fixed half, and the first label is worth 0.9 of a budget's weight in the ALC. The level of a new benchmark dominates the error; the ranking of items matters less.

### F.9 Closed probes, and Qwen3-14B on free GPUs

Six language-model and encoder studies are closed, each by the rule its plan fixed; an earlier version of this report listed them as in progress. The rows of App F.1's table summarise them, and the list below gives each verdict with its evidence (F§ "The 4B judge, closed out", "Attempting instead of judging", "Entropy profiles and hidden-state probes", "Few-shot prompting", "Fine-tuning an encoder"):

- The 4B zero-shot judge (rating, digit, entropy, nll) is closed: no benchmark-equal test-like difference reached -0.001 (App F.3).
- A 4B model's own attempts fall below the plan's floor. Graded accuracy was 2.3% answering at once and 7.8% with a short chain of thought, so the attempts are mostly forced guesses and the design cannot show whether they carry difficulty. A post-hoc feature met the plan's thresholds for GO on 32 extreme items, and we discount it.
- The entropy and hidden-state heads give a replicated null. Leave-one-benchmark-out $r$ is -0.05 [-0.24, 0.14] for the entropy head (random-effects mean over the parents; positive on 2 of 4) and +0.12 [-0.004, 0.24] for the hidden-state head (none at 0.2; `results/hidden_state_probe.json`, `heads.<head>.random_effects`). The nested harness lines, for the transferred and then the per-pair slope, are +0.00098 and +0.00003 for the entropy head, and 0 and -0.000001 for the hidden-state head.
- In-context learning over the pair's labels is closed. It learns from the labels (+0.158 ± 0.059 in $r$ over zero-shot) but reaches an $r$ of 0.206 ± 0.040 (both ± are SEs over pair appearances), below 0.3 on every parent, and mostly duplicates what hier learns from the same labels.
- Pairwise (anchored) comparisons were dropped at pooled q 0.540, against a bar of 0.60; q is the share of comparisons in which the model's answer matches the sign of the honest difficulty gap.
- The fine-tuned encoder is closed: 0 of 4 parents reach a held-out $r$ of 0.3. Its correlations are -0.08 [-0.22, 0.06], +0.05 [-0.01, 0.10], -0.09 [-0.32, 0.15] and -0.11 [-0.41, 0.10] (matharena, multi_swebench, real_webagents, researchcodebench; group intervals), and random effects give -0.05 [-0.14, 0.05] (`results/finetune_encoder.json`, `eval.per_cov.finetuned`). The transferred harness line is +0.00048.

The escalation that the plan set after the 4B's floor also ran: a stronger model on a free GPU (F§ "Strong model on Kaggle: Qwen3-14B rubric and attempts"; `experiments/strong_llm_eval.py`, `results/strong_llm_eval.json`). We ran Qwen3-14B (4-bit AWQ under vLLM, on two Kaggle T4s) in two sessions. In the first session, whose script ran 10.7 hours, it rated all 4,326 items of the four parents on a ten-scale demand rubric (eight ADeLe-style demands, a judged solve share and an expert's time). It also made four reasoning attempts of up to 4,096 tokens on 270 of the 819 checkable matharena texts, starting with the 147 probe texts. In the second session (the entropy job, whose script ran 4.9 hours by the same measure), it reasoned once over every item of the four parents, to record its token entropy.

The attempts pass their correlation rule, which we fixed together with the primary feature before any output existed for the probe texts. The primary feature, the mean token entropy of the raw distribution, reaches a within-competition Spearman $\rho$ of 0.372 [0.209, 0.516] against difficulty, with graded accuracy at 20.5%: GO. On the 2026 contests, which post-date the model and so cannot have been memorised, it is 0.585. On the other contests it is 0.29, so the pooled value leans on the 2026 ones. On all 270 texts it is 0.352 [0.251, 0.448]. It is the first item-side signal to pass its correlation bar. Of the attempts, 97% were truncated at 4,096 tokens, so the feature reads how the model starts to reason, and it does not see whether the model finishes.

Neither the rubric nor the attempts pass the gate. The rubric's primary head reaches a leave-one-parent-out $r$ of 0.19 (0.11 within a pair), and the best nested harness line of any rubric covariate is -0.00084. The attempts cover one parent, so no transferred slope can be fitted leave one parent out, and their forced per-pair lines give -0.0001. A covariate with an honest $r \approx 0.3$ on all four parents' items would sit just past the gate at this coverage (-0.0025, passing on 6 of 8 noise draws). The remaining question was whether the entropy reached that on the three parents that are not mathematics.

The entropy job was the variant with a chance to pass, because a slope transferred leave one parent out needs the feature on every parent. We held it while it was unclear whether a model may run at predict time, then ran it as a measurement for this report (F§ "Commit D: reasoning entropy on all four parents"). Its reading rule and primary feature were committed (`78e303e`) before any output existed. The primary feature is the mean raw entropy over the first 1,024 reasoning tokens after a generic prompt. The job covered all 4,078 units (distinct items, standing for 4,326 items) in 4.8 hours of generation, within the script's run time given above, at 1.28 times the planned throughput. The recorder check, which compares the recorded log-probabilities with the engine's own, passed on all 89 shards.

On all four parents the entropy is NULL: within a test-like pair its $r$ is 0.12, half of the 0.25 at which the reference (honest difficulty degraded on the items the entropy covers) reaches the gate. Within group it orders difficulty on mathematics (+0.29; +0.40 on the 2026 contests' text-bearing items) and on research code (+0.33), weakly on multi_swebench (+0.12) and not on real_webagents (-0.03). On mathematics it reads nearly what the attempts read (Spearman $\rho$ 0.87 with the attempts' entropy over the same window of reasoning tokens), and one sample is stable (test-retest intraclass correlation, ICC, 0.81). Nested, the transferred slope is switched on only in the fold that holds out real_webagents, where it costs (+0.00110 overall), and the per-pair slope is neutral (+0.00001). Forced on, the transferred slope gains on the other three parents and on public runs but loses 0.0058 on real_webagents.

Nothing from the Kaggle study enters the submission, and the line is closed. The other 549 attempt texts are also matharena texts, so they cannot change any call, and no further commit is planned.

*Literature.* Token-entropy profiles carry some difficulty signal within benchmarks but little across them: on 17 agentic benchmarks their Spearman $\rho$ falls from 0.19 under K-fold to 0.14 leave-one-benchmark-out [Krsteski and Meyer 2026]. Pre-generation activations predict a model's *own* success better than length or TF-IDF [Lugoloobi et al. 2026] and better than embedding assessors and verbalised confidence, though not on mathematical reasoning [Moreno Cencerrado et al. 2026]. Our subjects are other systems. A 4B proxy's signal did not transfer to them. A 14B's reasoning entropy tracks their difficulty on mathematics and research code and barely on the two agentic benchmarks, in line with the weak cross-benchmark transfer Krsteski and Meyer report for agentic tasks. Rubric-based demand levels predict instance-level performance [Zhou et al. 2026]. Here nine of the ten scales carry the declared sign on at least three of the four benchmarks, but weakly: within a test-like pair they correlate 0.03 to 0.15 with difficulty.

---

## Appendix G: reproducibility

### G.1 Environment

Development ran on an Apple M1 Pro (8 cores, 16 GB; macOS 26.5.1, Darwin 25.5.0) that other jobs shared. The environment was Anaconda's Python 3.10.8 with numpy 1.26.4 (OpenBLAS 0.3.23.dev), scipy 1.10.1, scikit-learn 1.7.2, pandas 2.2.2, pyarrow 23.0.1, huggingface_hub 0.36.2 and pytest 9.0.3. In every command, `python` is the interpreter of this Anaconda environment (`docs/report/provenance_facts.md` §1.3).

We used torch 2.6.0, transformers 4.57.6, tokenizers 0.22.2 and safetensors 0.8.0 only for the offline language-model features, and torch alone for the meta-learned heads of App F.5 (`experiments/heads_eval.py`). None of these libraries is in `pyproject.toml`, and re-running those studies needs them installed separately. The feature extraction ran on the M1 Pro's GPU through torch's MPS (Metal Performance Shaders) backend (`experiments/llm_features.py`, whose `--device` defaults to `mps`).

The Kaggle study of App F.9 ran `kaggle/strong_probe/strong_probe.py` on Kaggle's two T4 GPUs in two sessions. The notebook installed vLLM 0.9.2, torch 2.7.0 and transformers 4.53.2 (`results/strong_llm_eval.json`, `run.notebook` and `entropy.run.notebook`). Reading its outputs locally needs nothing beyond the Anaconda environment.

`tools/report_figures.py` draws the figures with matplotlib 3.9.2, which the `report` extra of `pyproject.toml` pins. The committed bytes of the figures are tied to that version.

    pip install -e ".[dev]"            # add ",report" to redraw the figures

The submission needs numpy alone (`requirements.txt`).

### G.2 Data and pinned inputs

Access to measurement-db is gated. Accept the terms on the dataset page, then run:

    export HF_TOKEN=...          # or huggingface-cli login
    python -m paiec.fetch        # the core tables of six benchmarks into data/, at the pinned revision

Every result in this report used measurement-db at revision `bc8204d811823da849c6686bf124d4ca9f82e4de`, downloaded on 2026-09-24. For all 24 parquet files, the local sha256 equals the etag (the Hugging Face hub's per-file hash) recorded at that revision, so `data/` is byte for byte the pinned dataset. `paiec/fetch.py` downloads the pinned revision by default (`REVISION`), and `PAIEC_DATA_REVISION` or `--revision` overrides it. `tests/test_fetch.py` checks the pin without network. Whether the dataset's main branch has moved since cannot be checked offline.

The organisers' baseline repository (validator and streaming client) has no license, and we do not redistribute it. Clone it at the commit the replica mirrors:

    git clone https://github.com/aims-foundations/paiec_baseline third_party/paiec_baseline
    git -C third_party/paiec_baseline checkout 82d330ddcdb16016a3ae9e048db7e588ba2c6a39

At that commit, `tools/streaming_ingestion.py` has sha256 `c6f2610f…` and `check_submission_zip.py` has `abe3cae0…`.

`pytest` needs neither the data nor the baseline repository; it runs on synthetic data. At the release commit the suite has 528 test functions over 24 files and 729 collected tests. All 729 passed on 2026-10-03 with the data and the baseline repository present.

### G.3 Rebuilding the submission

    python tools/build_submission.py        # fit prior.json, bake LEVEL, zip dist/paiec.zip, run every check
    python third_party/paiec_baseline/check_submission_zip.py dist/paiec.zip

`dist/paiec.zip` exists only if every check passed (§3.5). The flag `--legacy` builds the rollback to LegacyP.

The archive writer is deterministic: rebuilding commit ee5085a from `git archive` with that tree's own build script reproduces run 2's archive byte for byte.

The fit of `prior.json` is not bit-stable across BLAS threading. With OpenBLAS, OMP and vecLib held to one thread, the fit differs in its last bits (at most $2.1 \times 10^{-11}$ relative), and so does the archive hash. A byte-identical rebuild needs the same BLAS and thread count as the build machine (App G.1). We have shown such a rebuild on that machine, with the default thread settings, and on no other hardware (F§ "Formative feedback, runs 1 and 2", Archives).

| archive (sha256) | code | used for |
|---|---|---|
| `8e28d930d45b16aee351bfbbc75531a6d12079667793232d4bd74a17f9d47b9f` | b68492c, LegacyP | formative run 1 (the file downloaded back from the platform has these bytes) |
| `2c64eaada491cbf85bd54ae190cdbb28f9d0a13870df0dc77a3a850f90cf661f` | ee5085a, hier, old floor, solver before the floored-fit fix | formative run 2 (the file downloaded back from the platform has these bytes) |
| `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090` | 4d2cc4f, hier, corrected floor and floored-fit fix | archive-3; formative run 3, a regression and latency check only (the file in `dist/` has these bytes and every tracked member matches 4d2cc4f; it was not downloaded back from the platform, `results/formative_run3.json`, `archive`) |

### G.4 Re-running the experiments

Wall times are as recorded on the shared machine above. Every results file records its commands and code digests under `passes`, and `--summarise` rebuilds a file's summary from its stored rows.

| result | command | wall time | output |
|---|---|---|---|
| official baselines (§2.4, §5.2) | `python experiments/official_baselines.py --runs 600 --jobs 6` | 12 min, 6 processes | `results/official_baselines.json` |
| hier-fit vs LegacyP, ablations (App E.1, App B.3) | `python experiments/hier_eval.py --jobs 8 --skip 'r2\|researchcodebench\|pair\|hier t3 level'`, then `--summarise` | 2 h 16 min, 8 processes, 4 resumed invocations | `results/hier_eval.json` |
| test-like regime check (§4.1) | `python experiments/testlike_check.py --phase tune --jobs 5`, then `--phase check --resume --validate --jobs 5` | about 6 CPU hours | `results/testlike_check.json` |
| level calibration (§5.3, §5.4) | `python experiments/level_calibration.py --stage S --resume --jobs 6` for S = grid, grid2, r1base, eb, t3, prob, eb, screen, confirm, final; then `--summarise` | 3 h 11 min, 6 processes | `results/level_calibration.json` (19 MB) |
| level audit (§5.4, App E.2) | `python experiments/level_audit.py`, then `--stage extra --jobs 1` | 5 s; 1,414 s, 1 process, 1.9 GB | `results/level_audit.json` |
| regime sensitivity, the RS study (§5.4, App C.3, App E.2, §7) | `python experiments/regime_sensitivity.py lock`, `smcal`, `regimes --grid`, `reproduce`, then `score --shard 0/2` and `score --shard 1/2` side by side, then `summarise`; `review` after scoring (commands in the script's header) | scoring 31 to 40 s a run (all eleven configs), 520 runs on two processes, rows written over 2 h 31 min; at most 1.0 GB a process | `results/regime_sensitivity.json`; rows in `data/regime_sensitivity_rows` |
| script versions behind stored results (App C.3, App D.4) | `python experiments/script_revisions.py --stage replay`, then `--stage reread` | instant; 40 s, 0.3 GB | `results/script_revisions.json` |
| hier-ship, confirmed (§1.5, §5.1, §5.4) | `python experiments/ship_confirm.py` | about 5 s, 0.56 GB; reads `data/subject_side_rows` | `results/ship_confirm.json` |
| formative feedback (§5.1, App D) | `python experiments/formative_feedback.py --stage record`, `--stage prereg`, `--stage read` | seconds, 0.3 GB | `results/formative_feedback.json` |
| formative run 3 (§5.1, App D.1, App E.7) | `python experiments/formative_run3.py` (`--archive PATH --archive-how TEXT` to record a copy downloaded back from the platform) | 0.3 s, 1 process; reads no data; needs `dist/` to hold the `4a882cc7…` archive unless given `--archive` | `results/formative_run3.json` |
| itemsig layer (App F.4) | `python experiments/itemsig_eval.py --stage run --jobs 4 --rows DIR`, then `--stage oracle`, `--summarise`, `--stage verify`, `--summarise` | 38 min, 4 processes | `results/itemsig_eval.json` |
| acceptance harness (§6.1, App C.4) | `python experiments/harness.py --stage collect --jobs 1`, `--stage verify --legacy data/harness_rows_legacy`, `--stage table --resume`, `--stage show` | 39 min collection, 1 min verification, 44 min table | `results/harness_thresholds.json` |
| meta-learned heads (App F.5) | `python experiments/heads_eval.py --rows legacy`, then `--rows current` | 54 and 52 min, 1 process, 1.07 GB | `results/heads_eval.json` |
| known-sign cues (App F.6) | `python experiments/itemcov_eval.py --stage signs`, `--stage harness --rows data/harness_rows_legacy`, `HARNESS_SCALE=pooled ... --stage harness --scale train --rows data/harness_rows_legacy`, `--stage inventory`, `--stage show` | 11 min, 1 process | `results/itemcov_eval.json` |
| subject side (App F.7) | `python experiments/subject_side.py --stage verify`, `--stage diag`, `--stage run --jobs 2` (plus `--configs` passes), `--summarise` | about 4 h 25 min over four invocations, 2 processes | `results/subject_side.json` |
| multiple-choice floor (App B.4) | `python experiments/mcq_floor.py --stage items`, `--stage run --harness-rows data/harness_rows_legacy`, `--stage summary` | about 50 min, 2 processes | `results/mcq_floor.json` |
| floored fits (§3.2, §7) | `python experiments/hier_floor_replay.py --stage compare --both`, `--stage timing`, `--stage synthetic`, `--stage modes`, `--stage hidden`, `--stage summary` | about 1 hour over the stages (34 min for the replay, on two shards) | `results/hier_floor.json` |
| embedding transfer (App F.2) | `python experiments/llm_features.py --download` then `--resume`; `python experiments/emb_transfer.py` | see the notes below the table: model download 41 s and 390 s, by a one-off script; embeddings 20 min, 1 process on the GPU (MPS), in one `--resume` run that an out-of-memory error stopped; probe 232 s, 480 MB | `results/emb_transfer.json` |
| LLM rating study (App F.3) | `python experiments/llm_rating/analysis.py` | seconds | printed |
| closed language-model and encoder studies (App F.9) | `experiments/llm4b_close.py`, `attempt_probe.py`, `hidden_state_probe.py`, `icl_probe.py`, `pairwise_probe.py`, `finetune_encoder.py`, each with the stages in its F§ section | see F§ | `results/<script>.json` |
| a 14B on Kaggle: rubric and attempts (App F.9) | on Kaggle, `kaggle/strong_probe/strong_probe.py` (its README); then `python experiments/strong_llm_eval.py --stage S` for S = check-schema, ingest, signs, harness, reference, attempts (by default both readings: every attempted text and the probe texts), verdict, run, show | one Kaggle session on two T4s (the script ran 10.7 h); locally about 1.5 hours, 1 process (signs 14 min, harness 68 min, reference 8 min) | `results/strong_llm_eval.json` |
| a 14B on Kaggle: the entropy job (App F.9) | on Kaggle, the same notebook with `ARGS = ["--jobs", "entropy", "--no-prefix-caching"]` (README, "Коммит D", that is "Commit D", the entropy job); then `python experiments/strong_llm_eval.py` with `--stage check-schema` and `--stage ingest` on `--kaggle data/features/kaggle_d`, `--stage signs`, `harness`, `reference` with `--job entropy`, `--stage consistency`, `--stage verdict --job entropy`, `--stage run` | one Kaggle session on two T4s (the script ran 4.9 h, 4.84 h of it generation, about 5.0 h from the first cell to the export; planned 7.1 h, 9.3 h slow); locally about 1.2 hours, 1 process (signs 12 min, harness 40 min, reference 23 min) | `results/strong_llm_eval.json` (`entropy`) |
| inventory classes (§2.2) | `python experiments/inventory_classes.py` (`--out`, `--out-csv` to write elsewhere) | seconds, no network | `results/inventory_classes.json`, `.csv` |
| data counts (§2.2) | `python experiments/data_counts.py` | 2 s, 0.2 GB | `results/data_counts.json` |
| baselines: plain 1PL, empirical mean, BLE (§5.5, App B.5, App E.4) | `python experiments/baselines_p1.py ble`, then `score` (repeat while it exits with 75), then `summarise` | final pass 21 min, 1 process, at most 0.67 GB; reads the RS study's rows | `results/baselines_p1.json`; rows in `data/baselines_p1_rows` |
| pooling decomposition (§2.4, §5.6, App A.2, App A.3, App B.5) | `python experiments/pooling_decomposition.py score`, `recheck`, `summarise` | 208 tasks in 84 min, 1 process, under 1 GB | `results/pooling_decomposition.json`; rows in `data/pooling_decomposition_rows` |
| gate, intervals, single-subject breakout (§5.6, §6.1, App C.3, App C.4, App E.6, App F) | `python experiments/gate_and_ci.py --stage S` for S = gate, evidence, scale (by `--families`), ci, single | under 4 min a stage, 1 process, under 0.7 GB; reads stored results, rows and features | `results/gate_and_ci.json` |
| row export (App G.4) | `python tools/export_rows.py`, then `verify`; `restore` to unpack | minutes | `data/release_rows` (gitignored): eight tar.xz archives, `MANIFEST.json`, `SHA256SUMS` |
| figures (Figures 1 to 7, A1, B1, D1, E1, F1) | `python tools/report_figures.py`, `--check` to rebuild and compare | about 5 s, under 300 MB; reads `results/*.json` only | `docs/report/fig/*.svg`, `*.png`, `manifest.json` |
| legacy-replica results (App F.8) | `python experiments/ceilings.py`, `ladder.py --seeds 4`, `transfer.py`, `order_sensitivity.py`, `pool_robustness.py`, `acquisition.py --seeds 4` | minutes each | printed |
| offline bank (App F.8) | `mkdir -p data/scan && cp results/inventory.csv data/scan/ && cd data/scan && python ../../experiments/inventory_scan.py` (the script reads `inventory.csv` in the working directory and appends to `clone_scan.csv` there. It skips the slugs already in that file and needs network to clone the others. App G.6 describes the scan.) | not recorded; it ran before 2026-09-23 19:14 UTC | `data/scan/clone_scan.csv` (gitignored); the scan's committed output is `results/clone_scan.csv` |

The extraction ran once, on 2026-09-26, as one `experiments/llm_features.py --resume` process on the M1 Pro's GPU through MPS, while other jobs shared the machine. Its log, `data/features/extract.out`, and the times of its output files under `data/features/`, all gitignored, give its stages. The log first reports 21 s of tokenisation. The embeddings of the four multi-subject benchmarks (4,078 unique items, 1,663,952 tokens in 18 shards) then took 20 min, and App F.2 uses only these. The 4B judge then rated items for 6 h 48 min, with shard rates from 17 to 302 tokens per second, until an MPS out-of-memory error ended the process after 28 of its 39 rating shards. By then it had rated matharena and 1,941 of 2,078 multi_swebench items (App F.3). The script runs swe_rebench's embeddings and ratings last. We did not resume the run. So swe_rebench has no embeddings or ratings, real_webagents and researchcodebench have no ratings, and the wall time of a complete run is unknown.

A one-off script downloaded the two models before `experiments/llm_features.py` was written, with the same `snapshot_download` call that `--download` makes. Its timer printed 41 s for Qwen3-Embedding-0.6B and 390 s for Qwen3-4B-Instruct-2507 in the session that ran it, and the file times in the local Hugging Face cache span 7 min 10 s for both.

Two rebuilds of archive-3 by `tools/build_submission.py` (App G.3) took 80 s and 102.8 s, with a peak resident memory of at most 1.0 GB in the largest process (`docs/release.md`, section 6). The script's rebuilds of archive-2 at ee5085a and of archive-1 at b68492c, the second with BLAS held to one thread, took 41 s and 12 s (`results/formative_feedback.json`, `archives.run2.how` and `archives.run1.how`). The wall time of `experiments/inventory_scan.py` is not recorded: the scan ran before the repository's first commit (2026-09-23 19:14 UTC), and no log of that run is kept.

The rating study's input files, `results/rate2_truth.json` and `results/ctrl_truth.json`, hold no item text. For each rated item they keep its measurement-db item id, its Rasch difficulty and the sha256 digest and length of the excerpt the rater read. `docs/release.md`, section 9, rebuilds the excerpts from measurement-db.

The scripts above regenerate the large per-row files, which are gitignored: `data/harness_rows`, `data/harness_rows_legacy`, `data/itemsig_eval_rows`, `data/subject_side_rows`, `data/hier_floor`, `data/regime_sensitivity_rows`, `data/baselines_p1_rows` and `data/pooling_decomposition_rows`. The one exception is `data/harness_rows_legacy`, which comes from `experiments/harness.py --stage collect` run at a checkout of `bd0be67` (library `0c05d35e…`). At the current commit the collection writes `data/harness_rows`, which differs on 157 of 300 test-like runs (F§ "Acceptance harness", Provenance). The numbers of `heads_eval.py --rows legacy`, `itemcov_eval.py` and `mcq_floor.py` depend on the legacy rows. `experiments/ship_confirm.py`, `formative_feedback.py`, `level_audit.py` and `script_revisions.py --stage reread` read `data/subject_side_rows`, so they need `experiments/subject_side.py`'s rows first.

`tools/export_rows.py` packs eight row sets into deterministic tar.xz archives with a manifest of member sha256s, and `restore` unpacks and checks them. The eight are subject side, the RS study, the current and legacy harness rows, the floored-fit replay, the 14B's derived tables, and the rows of the baseline study and the pooling decomposition. When re-run on restored rows, `experiments/ship_confirm.py` passes every check with identical numbers, and the summaries of `baselines_p1.py` and `pooling_decomposition.py` are identical except for their timestamps (F§ "Row files for release"). The export never copies measurement-db, the language-model features under `data/features/` or `third_party/`, and the manifest carries measurement-db's terms. We do not host the archives, and we do not release the language-model features. A reader without the archives regenerates the rows from measurement-db with the scripts above and can pack them with `tools/export_rows.py`. The 14B's derived tables also need the Kaggle job of App F.9, and the legacy harness rows need a checkout of `bd0be67`. The itemsig rows cannot be packed: they were written to a scratch `--rows` directory instead of `data/itemsig_eval_rows`.

`formative_feedback.py --stage record` verifies the two formative archives only when it is given them (`--archive1`, `--rebuild1`, `--archive2`, `--rebuild2`). The paths we used are in the results file's command record (`passes`) and point into the scratch directory of the working session, which is not durable. Without the archives the script keeps the stored `archives` entry, so the hashes in `results/formative_feedback.json` are the durable record, and re-verifying needs the archives themselves.

`formative_run3.py` checks run 3's archive in `dist/` unless it is given a downloaded copy (`--archive`), and it records which file it read (`results/formative_run3.json`, `archive`). `dist/` is gitignored and `tools/build_submission.py` rebuilds it, so re-verifying needs it to hold the `4a882cc7…` file. When the script reads from `dist/`, the archive checks are required, and it writes nothing if that file is missing. Without `--replace-archive-record`, it never replaces a stored record whose archive matched with one whose archive does not.

### G.5 Seeds and determinism

- Public formative-like runs use `official.sample_run` with seed 0. Test-like runs use seed 2 (primary), seed 3 (sensitivities), seed 1 (tuning) and seed 5 (the level audit's two extra regimes).
- The RS study uses two seeds that nothing had used before. Seed 10 set its regime knobs from run composition alone, with no predictor. Seed 11 drew every scored run, test-like and public. The baseline study rescored the RS study's seed-11 runs. The pooling decomposition drew its formative-size runs at seed 11 too, so its runs 0 to 59 are the RS study's R1-bf runs.
- In `official`, seed 0 is the persistent 50/50 split, and other seeds stand in for other hidden splits.
- Bootstraps use 2,000 resamples.
- The replica's acquisition is the platform's sha256 rule. Every hash is a digest, never Python's salted `hash()`.
- The LLM features use no sampling, fixed batches and pinned model revisions.
- Every task in the long experiments is a pure function of its key, so an interrupted run resumes bit for bit.

### G.6 Conduct and disclosure

**Training data.** We trained only on measurement-db, the public training pool, at the revision of App G.2, and used no external per-item data.

**Formative feedback, every use.** The organisers have scored three formative submissions (F§ "Formative feedback, runs 1 and 2", Uses, and "Formative run 3"; `results/formative_feedback.json`, `submissions`; `results/formative_run3.json`).

Run 1 (LegacyP) set the test-like regime's defaults: its lower-root $B_{31}$ reading gave the level target, the date shift is the grid point whose LegacyP $B_0$ and $B_1$ come closest to the run's own, and the run's shape motivated two run-shape settings. Its composition (5 pairs alone, 4 sharing a benchmark) weighted the first verdict of §5.2. The level calibration chose three global hyperparameters on runs of that regime. It also estimated each candidate's score on run 1 by adding the candidate's paired difference to run 1's budgets, as a sanity check rather than as a selection criterion. The level audit then read run 1 per pair and moved the shipped level between two guarded configurations (§5.4). So run 1 was read twice per pair, as well as through its aggregate statistics, and every use set global quantities only.

Run 2 (hier-ship) motivated the subject-side study, which shipped nothing, and the pooled reading of runs 1 and 2, which changed nothing (§5.4, App D.4). We fixed and hash-locked that reading's decision rule before the reading was computed, but after both runs' tables had been seen. Afterwards we revised the descriptive outputs of the reading code twice, and the earlier versions give the same decision (App C.3). The RS study, which also changed nothing, then took the level targets of three of its regimes, READING, AUDIT and MIXTURE, from that reading and from the level audit's reading of run 1 (§5.4). No hyperparameter has changed since run 2: `LEVEL` and `submission/prior.json` are byte-identical to those in run 2's archive. The two library changes made after that run (the multiple-choice floor and the floored-fit fix) came from public-data findings.

Run 3 (archive-3, built at `4d2cc4f`) was uploaded as a regression and latency check only, a purpose stated in `results/formative_feedback.json` (`submissions`, first committed in `00bdf04`) three days before the run was scored. Its table is recorded (`experiments/formative_run3.py`). No level reading, no matching against replica pairs and no tuning is done on it, the RS study does not use it, and nothing has changed because of it. One descriptive use came later: the pooling study (§2.4) counts, in runs 1 to 3, how many pairs of a run share a benchmark (16 of 26 alone, 10 with one companion), and it reweights its replica differences to those shares. This use reads no Brier, level or label, and it sets nothing.

The live leaderboard was read once, on 2026-09-24, and is used for placement only. Runs 1 to 3 are our only formative submissions. We used their feedback only as this section lists, through the committed scripts. We gave the tables to the Claude Code sessions that wrote and ran those scripts, and used no number or id from them anywhere else, for example in a notebook or a hand calculation. We made no attempt to identify an anonymous benchmark or subject, and we made or shaped no submission to probe hidden labels (each statement of this paragraph confirmed by the author, 2026-10-04).

The organisers' tables are stored verbatim as a record (`results/formative/`), with their anonymous subject and benchmark ids. The docs, the figures and the research code (`paiec/testlike.py`'s `FEEDBACK`) use relabelled letters. The ids are used only to check the runs' overlap, to give the same letter to the same benchmark, to group pairs by benchmark in the pooled reading (App D.4; the reading's bootstrap resamples the 7 benchmark ids), and to count how many pairs of a run share a benchmark (§2.4). No model input, prior, hyperparameter or selection is keyed on an id. Nothing in `paiec/`, `submission/` or `tools/` reads `results/formative/`, and `prior.json` holds no item- or benchmark-keyed table. `tools/report_figures.py` reads each run's mean Brier by budget from `results/formative_feedback.json` and `results/formative_run3.json` to draw Figures 1 and D1. For Figure D1 alone it also reads each pair's Brier by budget and each run's counts of pairs and benchmarks. A search on 2026-10-04 of `paiec/`, `submission/` and `tools/`, at each of the repository's 22 commits up to and including the release commit, found none of the tables' 32 anonymous ids. The only per-pair feedback values in those directories are run 1's Brier rows in `FEEDBACK`, which only scripts in `experiments/` read and no shipped module imports.

**The inventory scan.** `experiments/inventory_scan.py` (first commit, 2026-09-23) cloned each GitHub repository of the organisers' 161-benchmark inventory with `--filter=blob:none --depth 1`. For each repository it listed the tracked files, matched regular expressions against the file *paths* (results-like directories, data extensions, model names), stored counts and up to three example paths, and deleted the clone. The script opened no file's contents, but a blobless clone checks out the default branch, so each repository's files were downloaded to a temporary directory and deleted with the clone. The inventory holds candidates for both pools, so the scanned repositories may include hidden-test benchmarks.

Two steps that followed the scan are not recorded. The first wrote `results/strong_repos.csv`, which no script in the repository produces. It lists the 15 repositories whose matched paths name a model, each with a model count, a model list, a file count and a sample path. Its rows follow the scan: in all 15 the sample is the scan's first example path, and in 14 the file count equals the scan's count of matched paths that name a model (mmdocrag has 346 against the scan's 354). The model lists, however, need each repository's full list of matched paths, which `results/clone_scan.csv` does not keep. So that step listed the 15 repositories' files again, or kept the lists from a run of the scan that we did not commit.

The second step is the judgement that, of the five repositories with outputs for nine or more models (phyblock, mmdocrag, atmossci_bench, capability, engdesign), only capability publishes graded outcomes. The recorded paths do not settle it: atmossci_bench's include `instance_acc.jsonl`, and mmdocrag's include files whose names end in `_llm-judge.jsonl`, names that suggest graded outcomes. The judgement was made in the Claude Code session that made the repository's first commit (2026-09-23), the commit in which `results/strong_repos.csv` entered the repository. No record of how the judgement was made is kept with the repository. So we cannot reproduce the judgement, and we cannot exclude an inspection of those repositories' files. Of the five, mmdocrag is a public benchmark, and the other four may be hidden-test benchmarks.

Nothing from the scan, its outputs, these steps or the inventory enters `paiec/`, `submission/` or `tools/`, and no per-item data from any inventory repository was kept or used. A search on 2026-10-04 of those directories, at each of the repository's 22 commits up to and including the release commit, found no reference to the inventory, the scan or its outputs. It found none to the five repositories either, apart from mmdocrag's name as a public benchmark of measurement-db. Nothing seen in the inventory repositories influenced code, priors, hyperparameters or selections, and no copy of a cloned repository, or of a per-item file from one, was kept anywhere, on any machine (confirmed by the author, 2026-10-04).

The scan assessed an offline bank, which we closed both on feasibility grounds and because the rules restrict competition-specific training and curation to the public pool (App F.8). The inventory's titles, rather than the scan, feed two descriptive analyses, neither of which sets a model, prior or hyperparameter: the task-type shares of §2.2 and a keyword scan for difficulty cues (App F.6).

**Blind LLM difficulty ratings (App F.3).** The rater was the Claude model of the Claude Code session that made the repository's first commit, as the team confirmed; the commit's trailer names Claude Opus 5. Its prompt, the date and the protocol are undocumented. The rater was blind to difficulty, but benchmark identity was visible to it, and we cannot exclude its exposure to measurement-db or public leaderboards. The fully specified replacement is the local Qwen3-4B judge (App F.3).

**Inventory validation labels (App A.1).** The 80 labels in `results/inventory_hand_labels.csv` come from an AI agent that read each title; no person assigned them. We report them as agent labels (confirmed by the team, 2026-09-28), and the agreement figures of App A.1 are agreement with that agent.

**LLM coding assistant.** We developed the code, the experiments, `docs/` and this draft with Claude Code (Anthropic). Commit trailers record Claude Opus 5 and Claude Opus 5.5 as co-authors.

**Pretrained models.** Qwen3-Embedding-0.6B (revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`) and Qwen3-4B-Instruct-2507 (revision `cdbee75f17c01a7cc42f958dc650907174af0554`) produced offline research features only. Qwen/Qwen3-14B-AWQ (revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`) ran on Kaggle for the research features of App F.9. None of them is part of the submission, which uses no pretrained model. The Kaggle notebook downloaded measurement-db itself with the team's Hugging Face token, into a directory outside the notebook's Output directory (the files Kaggle saves with each notebook version). The Kaggle kit, whose README is `kaggle/strong_probe/README.md`, asks that the notebook and its versions stay private. The notebook's Output directory holds no reference answer and no correctness flag, and the kit's tests check this.

**Community code.** We read the organisers' baseline repository at commit `82d330dd` (validator, streaming client and reference predictors) and ran it locally; we do not redistribute it.

---

## Appendix H: provenance

### H.1 Convention and internal labels

Every number in this report comes from `docs/findings.md` (cited as F§ plus the section title), from `docs/protocol.md` (P§) or from a file in `results/`. Each table names the script that produces it, its results file, or the appendix table that names them. App G.4 and App H.2 map the results files to their scripts, and App H.2 also maps each number of §1.5, §2.4, §5 and their appendices to the script, results file and code that produced it, and to the submission archives. We mark the few numbers derived in this report as "derived", with their inputs, and quote none that exists only in the scratch files of our working sessions. The wall times in the notes under App G.4's table are an exception to this convention. Their build times come from `docs/release.md` and `results/formative_feedback.json`. The extraction's stages come from its gitignored log and the times of its output files, and the download's span from the file times of the local Hugging Face cache. The two download times exist only in the record of the session that ran the download.

The script `tools/report_figures.py` draws the figures from `results/` only (App G.4). The file `docs/report/figures.md` gives each drawing an id from D1 to D12 and maps Figures 1 to 7, A1, B1, D1, E1 and F1 to those drawings, with their full captions and the results keys each panel reads. The SVG and PNG files are in `docs/report/fig/`.

Results files record, in their `passes` field, each invocation's command, wall time and digests of the code it ran, and the `--summarise` option rebuilds every summary from the stored rows. Some scripts changed after one of their stages had written results (`formative_feedback.py`, `level_audit.py`). For these, `experiments/script_revisions.py` rebuilds each earlier version from a log of the edits and checks it against the recorded digest.

The report uses these internal labels:

- R1 uses seed 0. TL uses seeds 2 (primary), 3 (sensitivities), 1 (tuning) and 5 (the level audit's two extra regimes). The regime-sensitivity (RS) study uses seeds 10 (setting the regimes' knobs) and 11 (scored runs).
- Archive-1 is `8e28d930…` (LegacyP, run 1), archive-2 is `2c64eaad…` (hier-ship, run 2) and archive-3 is `4a882cc7…` (hier-ship, run 3), the archive selected at the time of writing, which may not be the final selection.
- The "run-2 library" is hier 70a3a81a with the old multiple-choice floor and the solver before the floored-fit fix. The "current library" is that of `4d2cc4f`.
- The harness collected its "legacy rows" at `bd0be67` (library `0c05d35e…`) and its "current rows" at the head of the report's branch at the time of writing.
- In the labels above and in App H.2, a seven-character hexadecimal id names a commit of this repository's history before its one rewrite, on 2026-10-03; a tag of the same name resolves it (see the paragraph after this list). An eight-character id after "hier", "library" or "script", and each archive id above, with or without a trailing "…", is the start of a sha256 of that file's contents (for "library", of the library's files concatenated). Elsewhere, `82d330dd` is a commit of the organisers' baseline repository and `bc8204d8…` the dataset revision.
- F§ and P§ cite sections of `docs/findings.md` and `docs/protocol.md` by title.
- "The precision deviation" is the review's deviation D1: the RS study's reproduction check compared at stored precision (App C.3).
- "The entropy job" is the 14B's second Kaggle session, called "commit D" in its README.
- "The 4B", "the 14B" and "the blind rater" are the judges of §6.3.
- App I.2 maps the items of the internal review (`docs/report/review_v0.md`) to sections.

We rewrote the repository's history once before publishing it, on 2026-10-03 (`docs/release.md`, section 9). Otherwise the history would have published the e-mail address of the development machine's git configuration, the measurement-db item text quoted in two results files and a third-party access token. The rewrite set the author and committer address of every commit to the address of the title block. It replaced the item excerpts in `results/rate2_truth.json` and `results/ctrl_truth.json` by their sha256 digests and lengths, and the access token in one link of `results/inventory.csv` by a marker. No other file changed, but every commit id did. Each seven-character commit id in this report resolves in the published repository through a lightweight tag of the same name, and `docs/commit-map.txt` lists the earlier and the rewritten ids. The digests recorded for the three files are of their earlier bytes, and `docs/release.md`, section 9, shows how to re-check them.

### H.2 Which code produced each number

"Old floor" is the multiple-choice floor before `f7e7d87`, and "pre-fix solver" is `paiec/hier.py` before the floored-fit fix of `4d2cc4f`. Where Newton converges, the fix changes nothing, bit for bit. On runs 0 to 199 its measured effect on ALC is 0 (test-like), -0.0000002 (public benchmark-first) and -0.000008 (public pair-uniform) (`results/hier_floor.json`). In the same three regimes the corrected floor moves ALC by -0.00026, -0.00027 and -0.00045 (`results/mcq_floor.json`).

| number (section) | script | results file | code it ran | relation to the archives |
|---|---|---|---|---|
| formative run 1, all budgets and pairs (§1.5, §5.1, App D.1) | platform; recorded by `experiments/formative_feedback.py --stage record` | `results/formative/run1.txt`, `results/formative_feedback.json` (`record.run1`) | b68492c, LegacyP | archive `8e28d930…`, verified byte for byte against the file downloaded from the platform and against b68492c |
| formative run 2, all budgets and pairs (§1.5, §5.1, App D.1, App E.7) | platform; same | `results/formative/run2.txt`, `record.run2` | ee5085a: hier 70a3a81a, old floor, pre-fix solver | archive `2c64eaad…`, verified byte for byte against ee5085a |
| formative run 3, all budgets and pairs (§1.5, §5.1, App D.1, App E.7) | platform; recorded by `experiments/formative_run3.py` | `results/formative/run3.txt`, `results/formative_run3.json` (`run3`, `three_runs`, `overlap`) | 4d2cc4f: hier with the corrected floor and the floored-fit fix | archive `4a882cc7…`, checked member by member against 4d2cc4f in `dist/`; not downloaded back from the platform |
| run 2's placement, single-run sds (§5.1, App D.2, App E.7) | `experiments/ship_confirm.py` | `results/ship_confirm.json` (`run2_placement`, `single_run_sd`) | rows of `experiments/subject_side.py` at bd0be67: hier 70a3a81a, old floor, pre-fix solver | the run-2 archive's library; not re-run for `4a882cc7…` |
| run 3's placement (§5.1, App D.2) | `experiments/formative_run3.py` | `results/formative_run3.json` (`placement`) | the regime means and sds of the row above | the run-2 archive's library; run 3's archive differs by at most 0.00046, not added |
| matched estimates on run 1; the pooled reading (§5.4, App D.3, App D.4) | `experiments/formative_feedback.py --stage read` (script `93cb6a84…`; earlier versions give the same decision, `experiments/script_revisions.py`); `experiments/level_audit.py` | `results/formative_feedback.json` (`reading`); `results/level_audit.json` (`feedback`, `mild`); `results/script_revisions.json` | rows of `level_calibration.py` (head bba726c, hier 70a3a81a) and `subject_side.py` (same hier), old floor, pre-fix solver | the run-2 archive's library |
| baselines (§5.2) | `experiments/official_baselines.py` | `results/official_baselines.json` | b68492c, inferred: the results file records no commit, and it entered the repository in b68492c unchanged since; no hier | LegacyP is the run-1 archive's predictor |
| hier-fit vs LegacyP; ablations (§5.2, App B.3, App E.1) | `experiments/hier_eval.py` | `results/hier_eval.json` | bba726c library: hier f3f4e7da, old floor | neither archive's hier or LEVEL |
| unconstrained families, guarded configurations, hier-rec's sensitivities (§5.3, §5.4(a), App E.2) | `experiments/level_calibration.py` | `results/level_calibration.json` | head bba726c with hier 70a3a81a, old floor, pre-fix solver | the run-2 archive's hier.py; other LEVELs |
| hier-ship's own numbers (§1.5, §5.4) | `experiments/ship_confirm.py` | `results/ship_confirm.json` | rows of `subject_side.py` at bd0be67 (hier 70a3a81a, old floor, pre-fix solver; prior.py and subjects.py at f7e7d87) | the run-2 archive's library; `4a882cc7…` differs by the two changes above, not added |
| hier-ship minus hier-rec (§5.4, App E.2) | `experiments/level_audit.py --stage mild` | `results/level_audit.json` (`mild`) | the same rows as above, and `level_calibration.py`'s | the run-2 archive's library |
| LA-0.8 and LA-flat (App E.2) | `experiments/level_audit.py --stage extra` (script `40ed9e5b…`; the file since differs in the `analyse` stage and the docstring only, `results/script_revisions.json`) | `results/level_audit.json` (`extra`) | the library at 4d2cc4f: corrected floor, floored-fit fix | the current archive's library |
| the RS study: hier-ship against ten alternatives in seven regimes; the Smooth-cal row of §5.3 (§1.5, §5.3, §5.4, §5.5, App E.2, App E.4, App B.5, §7) | `experiments/regime_sensitivity.py` (rows and summary under script `0bef6bf0…`; earlier stages under `dbbf63b6…` and `316f8e91…`, re-run identically under the final script, `review.rerun`) | `results/regime_sensitivity.json` | the library at 4d2cc4f (hier d9a95612, corrected floor, floored-fit fix), with `paiec/testlike.py`'s `level_mix` added | the current archive's library |
| hier-ship across regimes (App E.7) | `experiments/subject_side.py` | `results/subject_side.json` | bd0be67 working tree: hier 70a3a81a, old floor, pre-fix solver | the run-2 archive's library |
| the corrected floor (App B.4) | `experiments/mcq_floor.py` | `results/mcq_floor.json` | bd0be67 (library 2a58be15), legacy harness rows, pre-fix solver | the first change between the two archives |
| the floored-fit fix (§3.2, §7) | `experiments/hier_floor_replay.py` | `results/hier_floor.json` | hier at f7e7d87 against 4d2cc4f, corrected floor | the second change between the two archives |
| latencies (§5.6, App B.5) | `experiments/subject_side.py`; `experiments/level_calibration.py`; `experiments/hier_floor_replay.py --stage timing`; `experiments/pooling_decomposition.py` (dense); `experiments/baselines_p1.py` (the 1PL) | the five results files | as above, and the library at 4d2cc4f for the last two | hier-ship on the run-2 library; hier-rec; the fix; hier-ship and the 1PL on the current archive's library |
| plain 1PL, EmpMean, BLE; the decomposition of the tuned-regime gain (§1.5, §5.5, App E.4) | `experiments/baselines_p1.py` (script `9825bb14…`) | `results/baselines_p1.json` | the library at 4d2cc4f (hier d9a95612); the RS study's rows for the other configs (the two smoothed means re-scored bit for bit on every run, hier-ship, hier-nosubj and LegacyP on the first and last run of each regime, hier-fit as stored) | the current archive's library |
| pooling on and off; the platform-composition reweighting (§1.2, §2.4, §5.6, App A.2, App A.3, App B.5) | `experiments/pooling_decomposition.py` (rows scored under script `b4504d05…`, summary under `4143835a…`; the `recheck` stage re-scored five tasks with the current script, bit-identical) | `results/pooling_decomposition.json` | the library at 4d2cc4f (hier d9a95612), with a research-only wrapper for pooling off | the current archive's library |
| the single-subject benchmark (§1.5, §5.6, §7, App E.6) | `experiments/gate_and_ci.py --stage single` | `results/gate_and_ci.json` (`single`) | seed 0: the rows of `subject_side.py` and `ship_confirm.py`'s arms (hier 70a3a81a, old floor, pre-fix solver); seed 11: the RS study's rows (hier d9a95612) | the run-2 archive's library and the current archive's, labelled apart |

**The confirmation table of §5.4(c).** Its rows come from hier with the old multiple-choice floor and the solver before the floored-fit fix, which is the code of run 2's archive. The selection-half row reproduces `results/level_calibration.json`'s `summary.selection` exactly.

**The code behind App E.7's rows.** We scored them at commit `bd0be67` with `paiec/hier.py` 70a3a81a and the old multiple-choice floor. That is the library of run 2's archive, before the corrected floor (`f7e7d87`) and the floored-fit fix (`4d2cc4f`). The files `paiec/prior.py` and `paiec/subjects.py` are at their `f7e7d87` versions, whose new terms are off by default and leave the prior and the predictions unchanged (F§ "Subject side at budgets 0 and 1"). Formative run 3 used archive-3, `4a882cc7…`, which adds both changes. The corrected floor moves ALC by -0.00026 (test-like), -0.00027 (benchmark-first) and -0.00045 (pair-uniform) (`results/mcq_floor.json`). The floored-fit fix changes nothing where Newton converges, and on runs 0 to 199 it moves ALC by 0, -0.0000002 and -0.000008 (`results/hier_floor.json`). We add neither shift to those rows.

**Base ALCs across tables.** Base ALCs differ between the report's tables because the run sets differ. On TL runs 0 to 299 (seed 2) hier-ship scores 0.1658, and on TUNED (seed 11, current library) it scores 0.1690. On R1-bf it scores 0.2051 on runs 0 to 149 and 0.2057 on runs 0 to 199 (seed 0, run-2 library), and 0.2114 on 60 runs at seed 11 (current library).

### H.3 Provenance gaps

The record behind this report has these gaps:

- A few design inputs exist only as recorded outputs: we did not rerun `item_signal.py` or `predictor_check.py`.
- Two checks in the Caveats of F§ "Acceptance harness" exist only as scratch files and are marked indicative there. We do not quote them.
- We did not re-measure the legacy-replica results of App F.8 under the official protocol.
- The blind rater's prompt and date are undocumented (App F.3).
- An AI agent assigned the 80 inventory validation labels (App A.1).
- How five inventory repositories were judged to publish raw or graded outputs is not recorded (App G.6).
- The extraction and download times in the notes under App G.4's table rest on records that are not committed: a gitignored log, file times and a session record (App H.1).
- We did not paste the platform's headline score for run 3 into the record, and we recorded its archive from the file in `dist/` rather than from a copy downloaded back from the platform (§5.1, App D.1).
- The RS study has two gaps of its own (App C.3). The study hashed its plan, rule and constants block to lock them. The evidence for that lock is local file times and copies, and no commit made before scoring records it. Its outcome also assumes the precision deviation, which we accepted on 2026-10-02. If the rule is read literally, without that deviation, it was not applied. Under either reading there is no candidate.
- An earlier version of the pooling study's script scored its rows, and the script's summary code changed afterwards. Five of its scoring tasks, each one run under one split scope, are bit-identical when re-scored by the version of the script that wrote the summary (F§ "Pooling under the verified protocol").
- Two correlations (TF-IDF within matharena, 0.73 and 0.49) were first recorded in the findings document from `experiments/transfer.py`, which prints its output and writes no results file. The code at the head of the report's branch, at the time of writing, does not reproduce them, and we quote the refitted values from `results/gate_and_ci.json` (App F.2).

### H.4 Commits and revision history

Draft v1 is dated 2026-09-28. We revised it on 2026-10-01 (the 14B's entropy job, formative run 3 and the RS study) and on 2026-10-02 for the remaining P1 items of the internal review (its second priority level: what a sound, complete analysis needs). Those items were the missing baselines, the pooling decomposition, the gate's tightening and intervals, the single-subject breakout, the row export and the figures. On 2026-10-02 we also restructured the report for the review's P2 items, which concern clarity, presentation and length. We cut the main text to sections 1 to 7, moved detail to these appendices, and used one name per model and one SE format. We kept the pre-trim text as `docs/report/draft_v1_long.md`. A verification pass the same day restored caveats lost in the trim, labelled the headline table in plain words, completed the references and recorded the submission requirements with their sources (`docs/report/report_requirements.md`). The pass changed no reported result.

We wrote the report against branch `official-protocol`. Every script, results file and test it cites is committed there, and so are the report's own files. We committed the P0 revision (the answers to the review's items blocking submission) in `00bdf04`, and the Kaggle record in the commit after it. The entropy job's kit and rule are in `78e303e`, the entropy job's output and formative run 3 in `380fecf`, the RS study in `7fb7450` and the P1 records in `dd3e372`. The submission archive's code is that of `4d2cc4f`. We tagged the release commit `v1.0.0` and pushed it on 2026-10-03 (`docs/release.md`, sections 6 and 7).

On 2026-10-04 we revised the report once more, in a commit after the release commit. The author decided that the row archives are not hosted and that run 3's headline score stays unrecorded, and left open how the raw-versus-graded judgement of App G.6 was made. The author also confirmed the statements that App G.6 marks as confirmed. We added the wall-time notes of App G.4, corrected the device in App G.1 and the archive count and offline-bank command in App G.4, and restructured App I. This revision changed no reported result.

---

## Appendix I: open items and the review map

### I.1 Open items and known gaps

- We did not run BLE, the empirical mean with BLE acquisition or amortized calibration (App E.5).
- We did not run three experiments related to the P1 items of the review. They are hier-ship at a target level mean (level_mean) of -2.0, hier-ship at the feedback's level reading without the synthetic date shift, and a level calibration for the 1PL itself (1PL-ship borrows hier's $\mu_0$, §5.5).
- We did not run the full gate under TL-mix, with selection on TL-mix and each draw's worst parent kept (App C.4). It needs a change to `experiments/harness.py`, a script that the P1 revision did not change.
- F§ "Acceptance harness" dates from draft v1, before the gate was tightened. F§ "The gate, tightened" holds the tightened gate (§6.1, App C.4).
- App E.7 describes the run-2 library, not the current library that `4a882cc7…` holds. The gap is measured (App E.7, App H.2), but we did not re-score the 'ship' arm of the subject-side study at `4d2cc4f`. Formative run 3 shows that `4a882cc7…` runs end to end on the platform. It does not show what that archive scores across the regimes.
- Two pooling costs from P1.11 are not explained (§7): hier's cost at $B_1$ on dense real_webagents (+0.015 ± 0.010 of Brier) and LegacyP's cost at $B_1$ on dense multi_swebench (+0.004 to +0.009). We have not checked whether the second-order (posterior-curvature) term is mishandled there, although a cost at this budget is the usual signature of such a mishandling. At the platform's composition, hier-ship's cost against LegacyP is mostly swe_rebench's (§5.6, App A.2). What remains for pairs alone on a multi-subject benchmark is within about one cluster SE. The open question is therefore the single-subject one (P1.15), rather than how many pairs of its benchmark a run holds.
- We do not host the row archives of App G.4, and we do not release the language-model features under `data/features/` (App G.4).
- Six entries of the figure plan are not drawn: its Figs. 1 (a schematic), 2, 4, 7, A2 and A5 (`docs/report/figures.md`, "Open questions"). The plan's figure numbers are its own and do not match the report's.
- In `experiments/regime_sensitivity.py`, `Maker.__call__` does not iterate over sorted parents, so the order of the empirical-Bayes (EB) traces inside its rows depends on `PYTHONHASHSEED`. No number depends on that order. We left the scored version as it is, because the rows record its digest.
- The pooled interval of `experiments/llm_rating/analysis.py` subtracts 3 from $n$ twice (`ci(r, len(df) - 3)`). The difference does not show at two decimals (App F.3).
- `experiments/harness.py --stage verify --scratch` expects the scratch row format of the meta-learned heads study. `data/harness_rows_legacy` holds the same predictions in another format, so it cannot stand in for those rows without a code change (F§ "Meta-learned heads on frozen embeddings", Caveat).
- `paiec/hier.py` calls the target's line read "INLA's Gaussian strategy" (in the docstring, line 67, and in a comment near line 1609), and it calls a refit at each node "INLA's Laplace strategy" (line 99). §3.2 uses the corrected term. We left the comments as they are, because changing them would change the bytes of the submission archive, which ships `paiec/hier.py`.

### I.2 The review map

The table maps each item of the internal review (`docs/report/review_v0.md`) to the place where this version answers it.

| review item | where it is answered |
|---|---|
| W1, the headline rests on a self-built regime | §5.4(c), §5.5, §7 |
| W2, hier-ship's own held-out and guard numbers | §5.4(a), (c) |
| W3, the level reading and the realised level | §4.1, §5.4(d), (e); App C.1, D.4 |
| W4, missing baselines | §5.5; App E.4, E.5 |
| W5, information set against run size | §2.4; App A.2 |
| W6, the harness's claims | §6.1; App C.3, C.4 |
| W7, underpowered negative results | §6.2, §6.3; App F.3 |
| W8, reproducibility gaps | App G, H |
| W9, conduct and disclosure | Code, data and conduct; App G.6 |
| Q3, Q5, Q12 | §5.4(e); §5.6 and App E.6; App B.5 |
| D1, the precision deviation | App C.3 |
| m1, names and notation; m7, one SE format | §2.1; §4.3 |
| m2, the shipped configuration told once; m9, length | §5.4; this version (App H.4) |
| m3, subject ids against canonical names | §2.2 |
| m4, split scope | §2.5 |
| m5, the caveats of §3.2 and §3.3 | §3.2, §3.3 |
| m6, mixed run sets | §4.1 and every table |
| m8, figures | Figures 1 to 7; Figures A1, B1, D1, E1, F1 |
| m10, literature | §1.4; References |
| P1.9, baselines | §5.5; App E.4, E.5 |
| P1.10, the regime sensitivity at the audit's reading | §5.4 |
| P1.11, the decomposition of contribution 1 | §2.4; App A.2 |
| P1.12, the gate | §6.1, §6.4; App C.3, C.4 |
| P1.13, intervals | §6.2; App F.1, F.3, F.6, F.9 |
| P1.14, between-benchmark variation | §1.5, §5.4 |
| P1.15, single-subject benchmarks | §5.6, §7; App E.6 |
| P1.16, a wider level prior | §5.4; App E.2 |
| P1.17, the row files | App G.4 |
| P2.18 to P2.22 | §2.1, §4.3, §5.4, the figures, §1.4 and References, §2.2, §2.5, §3.2, §3.3 |
