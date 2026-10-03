# Style guide for the rewrite of the technical report

This guide governs the rewrite of `docs/report/draft.md`: every formula in TeX, a main text that reads as plainly as the material allows, and the humanizer rules applied to technical prose. It is written for the people and agents who rewrite one part each. The record of the text before the rewrite is `docs/report/draft_v2_pre_humanize.md` (byte-identical to the draft at the start). The working files sit in the rewrite's scratch directory (`/private/tmp/claude-501/-Users-nikitapolomosnov-PycharmProjects-PAIEC/9df8ecfa-42c0-4508-919c-f794f1770909/scratchpad/humanize/`, not in the repository): `original.md`, the parts to rewrite, `parts/NN_<slug>.md`, listed in `parts/manifest.json`, and `numbers.py`, which checks a rewritten part against its original. The copy of this guide in `docs/report/style_guide.md` is the same text.

Work on one part at a time:

1. Read the part, this guide and the glossary of §2.1 (part `02_data_protocol_replica`).
2. Rewrite the part.
3. Run `python numbers.py parts/NN_x.md rewritten/NN_x.md`. Every number, reference, code span and heading label must match. Fix anything the lint reports as an error.
4. Hand the part back with a short note that lists each deliberate difference `numbers.py` still reports (for example, a code-font formula turned into TeX shows up as a missing code span) and anything you could not resolve.

---

## 1 Hard rules

These override everything else in this guide.

1. **Numbers.** Never change a reported number, its precision ("0.0419" stays "0.0419", "5.0" stays "5.0"), its sign ("+0.0101" keeps its plus), its unit or its qualifiers: the run set, seed, library, SE type and regime it belongs to. A number may move into TeX math ($\mu_0 = -2.5$) and may move within its part, but it stays attached to the same qualifiers.
2. **Claims.** Keep each claim's strength and scope. Hedges and verdict words carry results here: "about", "within noise", "level with", "not distinguishable", "provisional", "inconclusive", "more plausibly", "at least as close", "looks", "suggests", "closed", "null", "does not transfer", "no gain", "none passes", "GO for correlation; null for ALC". Do not add, drop, strengthen or soften any of them. Do not add interpretation, motivation or causal links the draft does not state ("so", "because", "therefore" only where the draft has them).
3. **Citations and references.** Keep every author-year citation ("[Brier 1950; Gneiting and Raftery 2007]"), every F§ "title" and P§ citation, every script and `results/` path, and every reference target ("§5.4", "App D.1", "Appendix D", "Figure 3", "Table 1") exactly. The PDF filter links these forms; write "§5.4" and "App D.1", never "Section 5.4" or "Appendix D.1, below".
4. **No content dropped.** Reword, split, merge and reorder sentences, and move material within the part when that reads better. Every fact, caveat, number, source and cross-reference must survive. When a sentence is pure signposting, rewrite it as a cross-reference that says what is where; do not delete it.
5. **Provenance.** Each table keeps its "Source:" (or "Sources:") note with its script and results file. F§ citations and `results/` citations stay where the numbers they support are.
6. **Markdown that both pandoc and GitHub render.** Inline math `$...$`, display math `$$` on lines of their own, no raw LaTeX outside math, no HTML. Details in section 6.
7. **Forbidden characters.** No em dash (U+2014), no en dash (U+2013), no Unicode minus sign (U+2212), no curly quotes, and no `--` in prose (pandoc's smart punctuation turns it into a dash). Use a period, comma, colon, parentheses or "to". `numbers.py` reports each one as an error.
8. **Text that stays fixed.** Headings (numbers and words); the title block lines ("Author:", "Affiliation:", "Email:", "Code:", "Keywords:"); image lines `![...](fig/<name>.png)` and the "*Figure N.*" label that opens each caption; legend names quoted in captions ("shipped hier", "legacy Predictor", "neighbouring config", "hier, no subject prior"); "*Table N: ...*" labels; every `TODO(team)` and `TODO(record)` marker; the indented shell commands of App G; the two conduct statements named in App I (section 10); the references list (section 10).
9. **Repository.** No git commit or push, no network. Do not edit anything outside the rewrite's own files.

---

## 2 Notation

One symbol per quantity, written exactly as below. Inline math uses `$...$`. Define each symbol in words at its first use in the main text (first uses are listed in section 10); appendices may rely on the main text's definitions.

### 2.1 Indices, data and scoring

| quantity | write | replaces in the draft | note |
|---|---|---|---|
| subject, item, benchmark | $s$, $i$, $b$ | s, i, b | a pair is $(s, b)$ |
| recorded outcome of subject $s$ on item $i$ | $y_{si}$; $y$ for one response in the Brier sum | (none) | $y_{si} \in \lbrace 0, 1\rbrace$; the offline prior's model (F7) uses $\Pr(y_{si} = 1)$ for the first recorded response; Brier counts every response, repeated trials included (F4) |
| the revealed labels | `labeled` in prose; $\mathcal L$ inside a formula | `labeled` | the API argument stays in code font |
| a pair's evaluated items; their recorded responses | $\mathcal E_{sb}$; $\mathcal R_{sb}$ | "the evaluation pool" | formulas only; $\mathcal R_{sb}$ lists every response $(i, y)$ on the items of $\mathcal E_{sb}$, each repeated trial counted |
| budget index | $k$, $k \in \mathcal K = \lbrace 0, 1, 3, 7, 15, 31\rbrace$ | k in {0, 1, 3, 7, 15, 31} | |
| budget checkpoint, and the Brier score there | $B_0, B_1, B_3, B_7, B_{15}, B_{31}$; generic $B_k$ | B0, B1, B3, B7, B15, B31, Bk | always math, in prose, tables and captions; "B0..B31" becomes "$B_0$ to $B_{31}$"; "budget 0" in words stays when it means the number of labels |
| ALC weight of budget $k$ | $w_k$ | 0.1, 0.2 | $w_0 = w_{31} = 0.1$, the others 0.2 |
| area under the learning curve | "ALC" in prose; $\mathrm{ALC}$ in formulas | ALC | |
| Brier score | "Brier" in prose; $B_k$ in formulas | Brier | |
| paired difference in ALC | $\Delta$ | Δ | $\Delta = \mathrm{ALC}(A) - \mathrm{ALC}(B)$, negative when $A$ is better |
| a pair's true success rate | $p$ (or $p_{sb}$ when the pair must be named) | p in p(1-p) | $\mathbb E[p(1-p)]$ is the base-rate oracle's Brier |
| a benchmark's level (mean pair-accuracy logit) | "level" in words | level | in a formula, $\operatorname{logit} p$ averaged over pairs; the model's own level parameter is $\mu_b$ below |

### 2.2 The model (hier)

| quantity | write | replaces | note |
|---|---|---|---|
| linear predictor (log-odds) | $\eta_{si}$ | eta | |
| predicted probability | $\hat p_{si}$; $\hat p_i$ when the pair is fixed | p, P(correct), p_i | the model's output; never $p$, which is the true rate |
| inverse logit | $\operatorname{logit}^{-1}$ | sigmoid | not $\sigma(\cdot)$, which would clash with the standard deviations |
| expectation, normal law, probability | $\mathbb E[\cdot]$, $\mathcal N(\cdot, \cdot)$, $\Pr(\cdot)$ | E[...], N(...), P(...) | |
| the benchmark's level parameter | $\mu_b$ | mu_b | on the item-level (Rasch) scale |
| subject standing | $\theta_s$ | theta_s | "standing" for hier; "ability" stays only for LegacyP's and the 1PL's per-pair term |
| pair deviation | $\delta_{sb}$ | delta_sb | |
| group effects of item $i$ | $g_i$ | g_i | one effect per item_features key, summed |
| item residual | $e_i$ | e_i | |
| multiple-choice floor of item $i$ | $c_i$ | c_i | $c_i = \omega f_i$ |
| floor read from the item text | $f_i$ | "the multiple-choice floor read from the item text" | |
| guess weight | $\omega$ | guess | $\omega = 0.5$; "hard floor" is $\omega = 1$ |
| slip | $\varepsilon$ | slip | $\varepsilon = 0.01$; "slip" stays as the word in prose |
| attribute ridge prediction of a standing | $\hat\theta^{\mathrm{attr}}_s$ | "attribute ridge mean" | the prior centres $\theta_s$ at $\gamma\,\hat\theta^{\mathrm{attr}}_s$ |
| all components except the item residuals | $\mathbf x$ | x | |
| the target's loading vector; the part of its linear predictor that the labels inform | $\mathbf a$; $\mathbf a^\top\mathbf x$ | a'x | $\eta_{si}$ also holds the components no label touched and, on a labelled item, minus that item's residual (`paiec/hier.py` docstring) |
| posterior mode; Laplace covariance | $\hat{\mathbf x}$; $\boldsymbol\Sigma$ | (words) | §3.2 display only |
| a point on the target's line | $\zeta$ | (words) | §3.2 display only |
| offline standing estimate (prior.py) | $t_s$ | t_s | |
| Rasch item difficulty | $d_i$ | "Rasch difficulty" | |
| honest difficulty (fitted on other subject folds) | $\tilde d_i$ | z (in the harness formula) | $z$ is reserved for z-scores |

### 2.3 Hyperparameters

| quantity | write | code name (give it once, at the first use) | shipped value |
|---|---|---|---|
| level prior centre | $\mu_0$ | `mu0` | $-2.5$ (LEVEL) |
| level prior sd | $\sigma_\mu$ | `sigma_mu` | 2.5 (LEVEL) |
| attribute scale | $\gamma$ | `attr_scale` | 0.5 (LEVEL) |
| identity-link sd | $\sigma_\theta$ | `sigma_theta` | 0.1, its floor |
| pair deviation sd | $\sigma_\delta$ | `sigma_delta` | 2.382 |
| attribute prior sd | $\sigma_{\mathrm{attr}}$ | `sigma_attr` | 1.018 |
| item residual sd | $\sigma_d$ | `sigma_d` | 2.671 |
| group effect sd | $\sigma_g$ | `sigma_g` | 1.542 |
| shared residual variance across benchmarks | $\tau^2_{\mathrm{res}}$ | `tau2_res` | estimated at -0.002, clipped to [0.01, 0.2] |
| Student-t degrees of freedom | $\nu$ | `nu` | 3 where scored |
| slip, guess weight | $\varepsilon$, $\omega$ | `slip`, `guess` | 0.01, 0.5 |

LEVEL stays the name of hier-ship's level prior, in plain text: "LEVEL: $\mu_0 = -2.5$, $\sigma_\mu = 2.5$, $\gamma = 0.5$". Code font, `LEVEL`, only when the constant in `submission/model.py` is meant. A configuration triple is written with its symbols: "$\mu_0 = -3.0$, $\sigma_\mu = 2.5$, $\gamma = 0.25$"; in a table whose header names the order, "$(\mu_0, \sigma_\mu, \gamma) = (-3.0, 2.5, 0.25)$" or the bare triple under that header. "sigma_mu x0.5" becomes "$\sigma_\mu \times 0.5$".

Knobs that are configuration labels, not model quantities, stay in words or code: the test-like regime's target level mean and sd (`level_mean`, `level_sd`: write "target level mean -1.6" after naming the knob once), LegacyP+fix's "offset -2.5, scale 1, variance 2" (the draft's "off -2.5, sc 1, va 2" spelled out), the Student-t level's "scale 1.8", hier-EB's "EB-cs, tau 2", Smooth-cal's Beta prior "mean $m_0 = 0.25$, strength $n_0 = 2$" (the draft's "n0 2, m0 0.25").

### 2.4 Statistics and the harness

| quantity | write | replaces | note |
|---|---|---|---|
| run, cluster and stratified SE | in words: "run SE", "cluster SE", "stratified SE" | same | $\mathrm{SE}_{\mathrm{run}}$, $\mathrm{SE}_{\mathrm{cl}}$, $\mathrm{SE}_{\mathrm{str}}$ only inside a formula |
| single-run sd | in words | same | never "SE" |
| standardised placement of a run | $z$ | z | "run 2's $z$ is $+0.84$", or "$z = +0.84$" |
| upper end of a 95% interval | $U_{95}$ | U95 | |
| honest correlation (with $\tilde d_i$) | honest $r$; $r$ inside a formula | honest r | the gate's scale; keep the word "honest" in prose |
| within-pair correlation | within-pair $r$; $r_{\mathrm{pair}}$ inside a formula | within-pair r | the term of Table 2; keep the words "within-pair" |
| other correlations | Pearson $r$, Spearman $\rho$, LOBO $r$ | r, rho | qualify in words |
| covariate | $x_i$ | x, x_i | |
| mean of $x$ over the labelled items | $\bar x_{\mathcal L}$ | "mean of x over labeled items" | |
| harness slope | $\beta$ | beta | |
| per-pair slope prior sd (per sd of $x$) | $\sigma_\beta$ | s | "(s = 0.5)" becomes "($\sigma_\beta = 0.5$)" |
| harness prediction | $q_i$ | q_i | the pairwise probe's "pooled q 0.540" is that study's statistic name and stays as plain text |
| offset cap | $\operatorname{cap}(o) = 4\tanh(o/4)$ | cap(o) = 4 tanh(o/4) | |
| noise of a degraded oracle | $\xi_i$ | noise | |
| the gate's bar | $\Delta \le -0.002$; "the gate, -0.002" in prose | -0.002 | |
| meta-head regularisation | $\lambda$ | λ | "$\lambda = 0.001$" |
| number of neighbours | in words | K | "15 neighbours", "with 15 nearest neighbours per pair"; no symbol, since $K$ would clash with the budget set $\mathcal K$ |
| explained variance | $R^2$ | R² | |
| acquisition hash rule | $u$, $n_{\mathrm{lab}}$, $n_{\mathrm{items}}$ | u, `labels_remaining`, `items_remaining` | see formula F12 |

Unicode symbols in the draft become TeX: Δ to $\Delta$, ≤ to $\le$, ≥ to $\ge$, ≈ to $\approx$, × to $\times$, → to $\to$ ("LegacyP $\to$ Smooth-cal $\to$ hier-ship"), ² to `^2`, λ to $\lambda$, "1e-9" to $10^{-9}$, "2.1e-11" to $2.1 \times 10^{-11}$, "5e-7" to $5 \times 10^{-7}$, "sqrt(1 - r^2)" to $\sqrt{1 - r^2}$, "iff" to "if and only if". Two Unicode characters stay: "±" in SE formats (section 5), and "σ" inside the model names "hier-ship σ3.5" and "hier-ship σ5".

---

## 3 Display formulas

Final TeX for the formulas the main text states. Copy them as written. Displays go on their own lines with a blank line before and after; the surrounding sentence carries the punctuation, so displays end without a period.

**F1. The response model (§3.1).** Replaces the indented `eta = ...` / `p = ...` block.

```
$$
\eta_{si} = \mu_b + \theta_s + \delta_{sb} - g_i - e_i, \qquad
\hat p_{si} = c_i + (1 - c_i - \varepsilon)\,\mathbb{E}\big[\operatorname{logit}^{-1}(\eta_{si}) \mid \mathcal L\big]
$$
```

The expectation is over the posterior given `labeled`; components no label touched stay at their prior inside it.

**F2. The priors (§3.1 term table).** Inline in the table's prior column:

- $\mu_b \sim \mathcal N(\mu_0, \sigma_\mu^2)$, Gaussian; Student-t optional
- $\theta_s$: Gaussian, mean $\gamma\,\hat\theta^{\mathrm{attr}}_s$ plus a capped identity term, variance from $\sigma_\theta$ and the attribute ridge ($\sigma_{\mathrm{attr}}$ in the ridge's place without an attribute prior); the display after the table (F2a) gives it in full
- $\delta_{sb} \sim \mathcal N(0, \sigma_\delta^2)$
- group effects $\mathcal N(0, \sigma_g^2)$, divided evenly over a benchmark's keys; group share $\sigma_g^2 / (\sigma_g^2 + \sigma_d^2) \le 0.25$
- $e_i \sim \mathcal N(0, \sigma_d^2)$, or $\mathcal N(0, \sigma_d^2 + \sigma_g^2)$ on a benchmark without item_features keys, integrated exactly per item; $\sigma_d^2 + \sigma_g^2$ is the median public item variance

**F2a. The standing's prior (§3.1, after the term table).** From `HierPredictor.theta_prior` in `paiec/hier.py`, with the attribute prior on (as shipped):

```
$$
\theta_s \sim \mathcal N\big(\gamma\,\hat\theta^{\mathrm{attr}}_s + w_s\,\bar r_s,\ (1 - w_s)^2\,(\sigma_\theta^2 + u_s) + w_s^2\,v_s\big),
\qquad
w_s = \min\Big(\frac{\sigma_\theta^2 + u_s}{\sigma_\theta^2 + u_s + v_s},\ w_{\max}\Big)
$$
```

$u_s$ is the ridge's predictive variance ($x^\top C x$), $\bar r_s$ and $v_s$ the identity table's mean residual standing and its noise ($s^2_d / k$), $w_{\max}$ the cap `id_cap`; $w_s = 0$ without a public record of the name. Without an attribute prior the attribute term drops, $\sigma_{\mathrm{attr}}^2$ replaces $u_s$ and $\bar r_s$ is the raw mean.
- $c_i = \omega f_i$, fixed, $\omega = 0.5$

**F3. ALC (§1.1).** Keep the draft's expanded form in §1.1 (it holds four separate 0.2 weights) and its factored form in §2.1's Table 2:

```
$$
\mathrm{ALC} = 0.1\,B_0 + 0.2\,B_1 + 0.2\,B_3 + 0.2\,B_7 + 0.2\,B_{15} + 0.1\,B_{31}
$$
```

Table 2 cell: `$\mathrm{ALC} = 0.1\,B_0 + 0.2\,(B_1 + B_3 + B_7 + B_{15}) + 0.1\,B_{31}$`. Where a general form helps: $\mathrm{ALC} = \sum_{k \in \mathcal K} w_k B_k$. The label arithmetic of §1.1 ("the first label is worth 0.9 of a budget's weight, labels 16 to 31 are worth 0.1 each") may be written as: the $j$-th label enters every $B_k$ with $k \ge j$, so it carries $\sum_{k \in \mathcal K,\, k \ge j} w_k$.

**F4. Brier (§1.1).** The draft defines it in words; the main text may display it once:

```
$$
B_k = \frac{1}{\lvert \mathcal R_{sb} \rvert} \sum_{(i, y) \in \mathcal R_{sb}} \big(\hat p^{(k)}_{si} - y\big)^2
$$
```

where $\mathcal R_{sb}$ is the list of recorded responses of $s$ on the items of $\mathcal E_{sb}$, every repeated trial counted, and $\hat p^{(k)}_{si}$ is the prediction at checkpoint $k$, made once per item and used for all its responses [P§ Scoring]. Pairs are then averaged equally; the platform's run score is the unweighted mean of the per-pair ALCs.

**F5. The paired difference (§1.5, §4.3).** $\Delta = \mathrm{ALC}(\text{hier-ship}) - \mathrm{ALC}(\text{comparator})$; in §4.3, $\Delta = A - B$, negative when $A$ is better.

**F6. The line read (§3.2).** Optional display; the text's words stay.

```
$$
\mathbf x(\zeta) = \hat{\mathbf x} + \boldsymbol\Sigma\mathbf a\,
\frac{\zeta - \mathbf a^{\top}\hat{\mathbf x}}{\mathbf a^{\top}\boldsymbol\Sigma\mathbf a}
$$
```

Then $\mathbf a^\top\mathbf x$ is read on a grid of $\zeta$, with the exact log posterior (item residuals integrated) evaluated at $\mathbf x(\zeta)$, without the conditional Gaussian's normalising term. Source: the docstring of `paiec/hier.py`.

**F7. The offline prior's per-benchmark model (§3.3).** $\operatorname{logit} \Pr(y_{si} = 1) = \mu_b + t_s - g_i - e_i$, with $y_{si}$ the first recorded response (replaces the code-font formula; not $p_{si}$, which would read as a true rate or, unhatted, clash with $\hat p_{si}$).

**F8. The harness offset (§6.1).** Replaces the indented `q_i = ...` block.

```
$$
q_i = \operatorname{logit}^{-1}\!\Big(\operatorname{logit}\hat p_i + \operatorname{cap}\big(\beta\,(x_i - \bar x_{\mathcal L})\big)\Big),
\qquad \operatorname{cap}(o) = 4\tanh(o/4)
$$
```

**F9. The degraded oracle (§6.1, App C.4).** $x_i = r\,\tilde d_i + \sqrt{1 - r^2}\,\xi_i$, where $\tilde d_i$ is the honest difficulty, standardised within its parent benchmark, and $\xi_i$ standard normal noise (`experiments/harness.py`, `degrade` and `base_matrix`); only then is $r$ the correlation.

**F10. The shipped item variance (§3.2).** $\sigma_d^2 + \sigma_g^2 = 2.671^2 + 1.542^2$, near 9.5.

**F11. The empirical mean's ALC (App A.4, Figure A1).** $\mathrm{ALC} \approx 0.025 + 1.2118\,\mathbb E[p(1-p)]$.

**F12. The default acquisition rule (§2.3).** The candidate is taken if and only if $u < \min(1,\, n_{\mathrm{lab}} / n_{\mathrm{items}})$, where $n_{\mathrm{lab}}$ is `labels_remaining`, $n_{\mathrm{items}}$ is `items_remaining` and $u \in [0, 1)$ the first 8 bytes of a sha256 digest of the candidate and its context, read as an integer and divided by $2^{64}$ (P§; `paiec/official.py`). Keep the code names once, so the reader can find them in the client.

**F13. The lower-root reading (§1.2, §3.4, §5.1).** "$p(1-p) = B_{31}$" in math. The explicit root $p = \tfrac12\big(1 - \sqrt{1 - 4B_{31}}\big)$, real only for $B_{31} \le 0.25$, may be added once where the reading is defined (§5.1); it is algebra, not a result.

---

## 4 Names

Names are plain text: never math, never code font, never italic or bold (except in the name column of §2.1's tables, which may keep the draft's bold). One name per thing; the glossary of §2.1 is the list, and figure legends keep older names that each caption glosses.

- **Models.** LegacyP, hier, hier-fit, hier-ship, hier-rec, hier-argmax, hier-ship σ3.5, hier-ship σ5, hier-EB, hier-nosubj, 1PL, 1PL-ship, 1PL-fit, Smooth, Smooth-cal, EmpMean, BLE, LegacyP+fix, and the base-rate, pair-rate, item and honest oracles.
- **Run sets and regimes.** R1, R1-bf, R1-pu, R2, TL, TL-mix, TL-noshift, TL-1.2, TL-2.0, LA-0.8, LA-flat, the RS regimes TUNED, READING, AUDIT, MIXTURE and FLAT; selection, confirmation and held-out runs; dense runs; public runs; test-like runs.
- **Replicas, libraries, archives.** the official replica (or "the replica"), the legacy replica, the run-2 library, the current library, archive-1, archive-2, archive-3.
- **Procedures and studies.** the level calibration, the level audit, the RS study, the baseline study, the pooling decomposition (or "the pooling study"), the harness, the gate, the guard, the rule, the reading rule, the subject-side study, itemsig, the meta-learned heads, the 4B judge, the 14B, the entropy job; LOPO, strict run-LOBO, nested LOPO, LOBO (data-level transfer only).
- **Data.** benchmark names as in the data, lowercase (matharena, multi_swebench, researchcodebench, real_webagents, swe_rebench, mmdocrag); field names as the draft writes them (item_features, benchmark_id, normalized_name); measurement-db.

Inside a formula a name goes in `\text{...}`: $\Delta = \mathrm{ALC}(\text{hier-ship}) - \mathrm{ALC}(\text{LegacyP})$. A name keeps its case at the start of a sentence ("hier-ship beats LegacyP"), never "Hier-ship"; prefer an opening that avoids it when the sentence allows ("On public runs hier-ship ..."). Do not cycle synonyms: "test-like runs" (or TL) everywhere, never "synthetic runs"; "standing" for hier's $\theta_s$, never "skill"; "level" for a benchmark's mean pair logit, never "difficulty" (difficulty is per item).

---

## 5 Numbers, ranges, SEs and units

- **Minus signs.** In prose, tables and captions write a plain ASCII hyphen-minus: "-0.042". The PDF filter sets a hyphen before a digit at the start of a word as a true minus, and GitHub shows the hyphen. Use math, $-0.042$, only when the number is part of a formula with symbols ($\mu_0 = -2.5$, $\Delta \le -0.002$, $z = -0.44$). Never wrap a bare number in math to get a minus, and never type U+2212.
- **Plus signs.** Keep every "+" the draft writes ("+0.0101", "z +0.50"); in a Δ column the sign is the result.
- **Precision.** Keep the digits as written in each place. The draft rounds the same quantity differently in different places ("-0.0419" in a table, "0.042" in prose); keep each occurrence as it is.
- **Ranges.** "0 to 31", "0.009 to 0.030", "-0.16 to 0.14", "B3 to B31" ($B_3$ to $B_{31}$). No dash between numbers in prose. Run-set labels inside tables may keep the draft's ASCII hyphen ("runs 0-99"); in prose write "runs 0 to 99". Abbreviated year spans stay as written ("2025-26"). Avoid false ranges: "from X to Y" only for two ends of one scale.
- **Thousands.** Comma grouping in text ("571,921", "1,000"). Avoid numbers of 1,000 or more inside math; if unavoidable, write `1{,}000`.
- **Percentages.** "4.8%", no space, never "percent"; "95% interval".
- **Scientific notation.** In math: $10^{-9}$, $2.1 \times 10^{-11}$.
- **SEs, the §4.3 convention.** In the main text a difference with an SE in parentheses means the cluster SE: "-0.0419 (0.0037)". The full triple is "-0.0415 ± 0.0011 / 0.0034 / 0.0031 (run / cluster / stratified SE)"; the parent-level summary is "mean ± SE" ("-0.038 ± 0.008"); a 95% interval is "[0.25, 0.64]"; a single-run sd is named as such. These formats stay plain text with the Unicode "±"; they are never put in math. Every SE keeps its type when a sentence is restructured.
- **Units.** A number and its unit with a space: "0.042 ALC", "0.0026 of Brier", "0.9 logit", "1.34 ms", "0.56 s", "3 h 11 min", "16 GB", "1.25 years". Write "per call", "per weighting", "per benchmark" for the draft's "a call", "a weighting", "a benchmark" in the main text.
- **Counts and ratios.** "6/8", "0 of 4 parents", "2.7 times" stay as written.
- **Dates and times.** ISO dates ("2026-09-24") and clock times with UTC as in the draft.

---

## 6 Markdown and TeX mechanics

- **Inline math** is `$...$` with no space after the opening `$` or before the closing one, and the closing `$` must not be followed by a digit (`$B_0$1` breaks). Write `$B_0$ and $B_1$`, not `$B_0, B_1$`, when the two are separate items of a sentence.
- **Display math** is `$$` on a line of its own, the formula, `$$` on a line of its own, with a blank line before and after. Never indent it: four spaces turn it into a code block. One display may hold two formulas separated by `\qquad`. Do not use `\begin{equation}`, `\label`, `\tag` or numbered equations; `aligned` inside `$$` is allowed but rarely needed.
- **Commands to use:** Greek letters, `\mathcal`, `\mathbb`, `\mathbf`, `\boldsymbol`, `\hat`, `\tilde`, `\bar`, `\operatorname{logit}`, `\operatorname{cap}`, `\mathrm{...}`, `\text{...}`, `\tfrac`, `\frac`, `\sqrt`, `\sum`, `\min`, `\Pr`, `\tanh`, `\le`, `\ge`, `\approx`, `\times`, `\to`, `\in`, `\mid`, `\lvert ... \rvert`, `\lbrace ... \rbrace`, `\top`, `\big`, `\Big`, `\,`, `\!`, `\qquad`. GitHub and pandoc both render these.
- **Avoid inside math:** `|` (it splits table cells; use `\lvert`, `\rvert` or `\mid`), `*` (GitHub may read emphasis; use `\ast` or `\cdot`), `\{`, `\}` and `\\` (GitHub drops the backslash; use `\lbrace`, `\rbrace`), and Unicode symbols (use the commands). `numbers.py` warns about each.
- **No math** in headings, the title or the abstract. The abstract is also pasted into OpenReview's plain-text field; keep its numbers plain.
- **Tables.** Keep the rows, columns and order. Header cells may hold math ("$B_0$", "$\Delta$ LegacyP"). Do not merge or split tables. Bold stays only where the text refers to it (§3.3's three LEVEL fields; the bold cells of the gate tables in §6.1 and App C.4).
- **Table notes.** The paragraph right after a table that starts "Source:", "Sources:", one word and a colon ("Dense:", "Seeds:"), "¹" or an asterisk is set small by the PDF filter as the table's note. Keep that opening. Such notes may stay as compact fragments; they are the one place where the subjectless-fragment rule does not apply.
- **Figures.** Keep each image line and its caption in the same subsection (the build checks placement against `docs/report/figures.md`). The caption paragraph opens with `*Figure N.*`; the rest may be humanized like prose, keeping panel letters, legend names in quotes and every number.
- **Code font** for files, scripts, functions, constants, API arguments and commit-like identifiers already in code font. Do not move a code identifier into math unless it is a formula (then note it for `numbers.py`'s verbatim report).
- **Lists.** Numbered lists stay numbered where order matters (procedures, the gate's four conditions, the rule's five conditions, the legacy replica's five differences). Bullets without bold heads (section 8).

---

## 7 Voice and readability

- Plain, precise and neutral. First person plural where the authors act ("we built", "we chose", "we scored"); the organisers, the platform or a script as the subject where they act. No jokes, no opinions beyond what the evidence supports.
- Open each paragraph with its point; the evidence and the caveat follow. Keep paragraphs short (two to five sentences in the main text).
- One idea per sentence where the draft packs three. Split semicolon chains into sentences. Keep at most one parenthesis per sentence where possible; move provenance into a final parenthesis or into the table note.
- Vary sentence length: a long sentence that carries a qualified result can sit next to an eight-word one. Avoid runs of very short sentences.
- Define a term before using it, or point to §2.1 at the first use in a section. Expand an abbreviation the first time a part uses it, except the glossary's names.
- Keep the caveats next to the numbers they qualify. A caveat moved to another paragraph changes the claim.
- Prefer verbs to noun stacks ("we refit the prior without the target's parent" rather than "prior refitting without target-parent inclusion").
- Keep cross-references short and specific ("details: App B.1" may become "(App B.1)").

---

## 8 Humanizer rules, adapted to this report

Rewrite each instance; never delete the content it carries. The draft is already free of most vocabulary tells. Its tells are structural, so the first five rules matter most here.

1. **Mechanical bold and inline-header lists.** About 37 bullets in the main text open with a bold sentence ("- **Runs are small.** ..."), and most paragraphs open with a bold run-in. Rules: (a) no bold inside list items, except the numbered contributions of §1.3, whose bold titles stay; a bullet that opens with a bold sentence becomes a paragraph (when it holds more than two sentences) or a plain bullet whose first sentence states the point; (b) a bold run-in label may open a paragraph only in sequences a reader scans: §1.2's four difficulties, §1.4's three strands, §3.2's two steps, §3.4's procedure, §5.4's (a) to (e), §5.6's three topics, §6.3's study labels, and the catalogue subsections of the appendices; the label is a short noun phrase or the point as one short sentence, ending with a period; (c) the convention sentence of §4.3 ("In the main text, ...") may stay bold; (d) elsewhere, drop the bold and keep the sentence. Use italics, not bold, for a term at its definition ("either *transferred* ... or *per-pair* ...").
2. **Compressed fragments and missing actors.** "Sources and the count script: App A.1.", "Other open points: App A.3.", "(method: App A.2)", "Which configuration shipped, and why, is told once, in §5.4". Outside table notes, write a sentence with a subject and a verb: "App A.1 gives the sources and the script that recounts them." Prefer active voice where the actor matters ("we switched pooling on and off"); keep the passive where the actor is the procedure itself ("each pair's split is fixed across submissions").
3. **Tacked-on contrasts and aphoristic closers.** The draft often ends a paragraph with an "X, not Y" line or a semicolon epigram ("The first replica did not invent the lever; the platform's runs are too small to use it."; "the unit, not the target, flatters them"). Keep the content of both halves, because the negated half is often a finding or a disclaimer ("not pre-registered", "redrawing, not new parents"), but write it as a plain clause: "The halves redraw from one catalogue of pairs, so they measure redrawing; they do not test new parents."
4. **-ing tails that add a second claim.** "..., reducing the Gaussian approximation's under-reaction ...", "..., bracketing the legacy ladder's 0.020, ...". Make the second claim its own clause or sentence with a subject.
5. **Diff-anchored writing.** "we no longer make it", "now reads", "§5.3 reversed it", "is gone", "has since". Describe the state and keep the history as dated or located facts: "The first verdict (F§ "Verdict: keep the Predictor") was to keep LegacyP; the test-like runs of §5.3 reversed it." Retractions of an earlier claim stay explicit ("we withdraw the earlier claim that ..."), because the retraction is content.
6. **Synonym cycling.** One name per thing (section 4).
7. **Signposting.** "§6.1 gives the harness ..., §6.2 and §6.3 what was tried, §6.4 why it fails." A roadmap sentence at the start of a long section is acceptable when it says what each subsection establishes; no "this section describes", "we now turn to", "let us look at".
8. **Headings restated by their first sentence.** Start with the content.
9. **Inflated significance and AI vocabulary.** None of: pivotal, crucial, key (adjective; "main" or "primary"), underscore, highlight (verb), showcase, testament, landscape, delve, enhance, foster, garner, interplay, intricate, tapestry, valuable, vibrant, additionally, notably, robust (as praise; "robustness to where the hidden level sits" is a measured property and stays), comprehensive, seamless, leverage. No "serves as", "stands as", "boasts": write "is", "has".
10. **Negative parallelism and forced triplets.** No "not only ... but also". List as many items as the facts have.
11. **Filler and stacked hedges.** No "in order to", "it is important to note", "it should be noted". Collapse two hedges into one only when they say the same thing; a hedge that carries a measured caveat stays.
12. **Generic endings.** No closing sentence that only restates or praises. End a paragraph on its last fact or caveat.
13. **Staccato drama and rhetorical openers.** No runs of fragments for effect, no "The answer?" openers. The draft's one-line run-ins ("Runs are small.") are fine as labels under rule 1(b).
14. **Dashes and quotes.** No em or en dashes, no `--`, straight quotes only.
15. **Title case.** Headings are already sentence case and stay unchanged.

---

## 9 Five examples from the draft

### Example 1: bold inline-header bullet (§2.2)

Before:

> - **Few subject ids cross benchmarks; more model names do.** Only 22 of 287 subjects appear on more than one benchmark [P§ Data], but hier keys a subject's standing theta_s on the coarser canonical name (`normalized_name`, else the source model name), which links 117 of the 220 pairs of the four multi-subject benchmarks (F§ "The step-2 analyses behind the model"). On current evidence that link adds almost nothing on top of the attributes, a provisional reading (§3.3). `interactors` is empty everywhere.

After:

> Few subject ids cross benchmarks, but more model names do. Only 22 of 287 subjects appear on more than one benchmark [P§ Data]. hier keys a subject's standing $\theta_s$ on the coarser canonical name (`normalized_name`, else the source model name), which links 117 of the 220 pairs of the four multi-subject benchmarks (F§ "The step-2 analyses behind the model"). On current evidence the link adds almost nothing on top of the attributes, and we treat that reading as provisional (§3.3). The `interactors` field is empty everywhere.

The bold head became the topic sentence, the code-style name became $\theta_s$, the appended fragment "a provisional reading" became a clause with an actor, and every number, citation and reference is unchanged.

### Example 2: an -ing tail carrying a second claim (§1.3, contribution 2)

Before:

> 2. **A hierarchical item-response predictor fitted from `labeled` alone** (`paiec/hier.py`). It integrates every item residual exactly and reads the target on the exact log posterior along a line, reducing the Gaussian approximation's under-reaction to a pair's first labels (exact for one pair whose labelled items share the target item's group effects; §3).

After:

> 2. **A hierarchical item-response predictor fitted from `labeled` alone** (`paiec/hier.py`). It integrates every item residual exactly and reads the target on the exact log posterior along a line. The line read reduces the Gaussian approximation's under-reaction to a pair's first labels, and it is exact for one pair whose labelled items share the target item's group effects (§3).

The contribution's bold title stays (rule 1(a)); the participle and the stacked parenthesis became a sentence.

### Example 3: an appended participle and an aphoristic closer (§2.4)

Before:

> On dense runs pooling is still worth 0.009 to 0.030 ALC, bracketing the legacy ladder's 0.020, mostly through item difficulty, and per-benchmark splits halve it (ratios 0.47 for hier, 0.36 for LegacyP). At formative size it is worth 0.002 to 0.003 to hier and 0.0004 to 0.001 to LegacyP, and at the platform's composition 0.0004 to 0.001 and exactly 0. The first replica did not invent the lever; the platform's runs are too small to use it. Nothing in this report is quoted from the legacy replica except where marked "legacy".

After:

> On dense runs pooling is still worth 0.009 to 0.030 ALC, which brackets the legacy ladder's 0.020. Most of it comes through item difficulty, and per-benchmark splits halve it (ratios 0.47 for hier and 0.36 for LegacyP). At formative size pooling is worth 0.002 to 0.003 to hier and 0.0004 to 0.001 to LegacyP; at the platform's composition it is worth 0.0004 to 0.001 to hier and exactly 0 to LegacyP. So the first replica did not invent the lever, but the platform's runs are too small to use it. We quote the legacy replica only where a number is marked "legacy".

The pairing of "0.0004 to 0.001 and exactly 0" with hier and LegacyP is made explicit from the table above it (hier-ship's platform-mix row, -0.0010 / -0.0004; LegacyP's, 0 / 0); nothing new is claimed.

### Example 4: diff-anchored retraction with a bold head (§5.5)

Before:

> - **"Calibration is worth far more than the modelling" is not an order-free statement, and we no longer make it.** Calibrating first gives the calibration the whole gain, modelling first 0.71, and on the 1PL path 1PL-fit takes 0.88 before any calibration.

After:

> - The split depends on the order of the steps, so we withdraw the earlier claim that calibration is worth far more than the modelling. Calibrating first gives the calibration the whole gain, modelling first gives it 0.71, and on the path through the 1PL, 1PL-fit takes 0.88 before any calibration.

The retraction stays explicit ("withdraw the earlier claim"), as rule 5 asks; only the bold and the "no longer" framing go.

### Example 5: a code-style formula and its definitions (§6.1)

Before:

>     q_i = sigmoid( logit p_i + cap(beta (x_i - mean of x over labeled items)) ),  cap(o) = 4 tanh(o/4)
>
> Here p_i is hier-ship's prediction; centring keeps its level, so x only orders items, and does nothing at B0. The slope beta is either **transferred**, one coefficient per budget fitted on the other parents, or **per-pair**, a MAP from the target pair's own labels under a zero-mean prior of s per standard deviation of x.

After:

> $$
> q_i = \operatorname{logit}^{-1}\!\Big(\operatorname{logit}\hat p_i + \operatorname{cap}\big(\beta\,(x_i - \bar x_{\mathcal L})\big)\Big),
> \qquad \operatorname{cap}(o) = 4\tanh(o/4)
> $$
>
> Here $\hat p_i$ is hier-ship's prediction and $\bar x_{\mathcal L}$ is the mean of $x$ over the labelled items. Centring keeps hier-ship's level, so $x$ only orders items and does nothing at $B_0$. The slope $\beta$ is either *transferred*, one coefficient per budget fitted on the other parents, or *per-pair*, a MAP estimate from the target pair's own labels under a zero-mean prior with sd $\sigma_\beta$ per standard deviation of $x$.

The indented code line became display math, the prose names became the symbols of section 2, and the defined terms are in italics instead of bold. `numbers.py` reports the same numbers on both sides (4, 4, and $B_0$'s 0).

---

## 10 Notes per part

- **00_front (title block, provenance line, abstract).** The title block lines stay byte for byte. Humanize the provenance paragraph and the abstract lightly, keeping every number; no TeX in either. App I quotes the abstract's length ("1,544" characters) and the 5,000-character limit: leave App I's number as it is; the coordinator recounts it after the rewrite and updates it in one place.
- **01_introduction.** First uses: ALC, $B_k$ and the Brier score (§1.1; F3 and optionally F4); the lower-root reading $p(1-p) = B_{31}$ (§1.2); $\Delta$ (§1.5, F5). The §1.5 headline table keeps its rows and numbers; convert "Δ" to $\Delta$ and the budgets to math.
- **02_data_protocol_replica.** §2.1's Table 1 and Table 2 are the glossary: keep every row; add symbols next to terms where they belong (for example "budget $B_k$", "level; LEVEL" with $\mu_0$, $\sigma_\mu$, $\gamma$ given as LEVEL's values). First uses: $\theta_s$ (§2.2), the hash rule F12 (§2.3). The pooling table keeps its "pooling on minus off" header without U+2212.
- **03_method.** F1, F2, F6 (optional), F7, F10; the §3.3 hyperparameter table's header becomes the symbols of section 2.3 with the bold LEVEL fields kept; the code names appear once, in that table's note or the sentence before it.
- **04_evaluation_methodology.** The regime table and the SE convention of §4.3. The gate's four conditions stay a numbered list.
- **05_results.** The largest part. Every table keeps its rows; header cells "Δ LegacyP" become "$\Delta$ LegacyP"; the header that puts a Unicode minus between "alternative" and "hier-ship" becomes "alternative minus hier-ship"; the step paths use $\to$. The footnote marker "¹" stays. §5.4's (a) to (e) keep their run-in labels.
- **06_what_does_not_transfer.** F8 and F9; the gate table keeps its bold; the ideas table keeps its verdict column word for word.
- **07_limitations.** Bulleted limitations: drop the bold heads (rule 1(a)), keep each bullet's facts and cross-references.
- **08_code_data_conduct.** The italic inline heads go (rule 1). Two statements keep their wording, because App I asks the team to confirm them as worded: "It set global quantities only: ..." (formative feedback) and "nothing from the scan, its outputs or the inventory enters `paiec/`, `submission/` or `tools/`, and no per-item data from it was kept or used". The rest may be edited.
- **09_references.** Not humanized. The only change: each of the 26 en dashes (U+2013) becomes an ASCII hyphen. Page ranges keep their numbers and become "9-25" in form, and Gauss-Hermite in Liu and Pierce's title gets the hyphen the body already uses. Everything else stays byte for byte; `numbers.py` should then report no difference and no lint error for the part.
- **Appendices (10 to 18).** Same rules as the main text, with a lighter hand: the appendices are reference material, so long tables and dense provenance stay; convert every formula and symbol to TeX and fix the structural tells. App G's indented shell blocks stay byte for byte. App H's tables of scripts, results files and archives stay as they are apart from symbols. App I is the team's checklist: keep every `TODO(...)` marker and item; humanize its prose lightly.

---

## 11 Checklist before handing back a part

- `python numbers.py parts/NN_x.md <rewritten>` reports no missing or extra numbers, refs, code spans or headings, or each difference is a deliberate, listed conversion (a code formula turned into TeX).
- No lint errors: no em or en dash, no U+2212, no curly quotes, no `--`, no unpaired `$`, no math in a heading, no "|" inside math in a table row.
- Every table still has its "Source:" note; every figure keeps its image line and "*Figure N.*" caption in the same subsection.
- Every symbol is the one in section 2 and is defined at its first main-text use.
- Every name is the glossary's, in plain text.
- Each paragraph opens with its point; no bold heads in bullets; no tacked-on "X, not Y" fragments; no -ing tails carrying a second claim.
- Read the part once more against the original, paragraph by paragraph, for dropped caveats and changed hedges; `numbers.py` cannot see words.
