# References check for `docs/report/draft.md`

Read 2026-10-02, against the draft at HEAD `dd3e372` (branch `official-protocol`). This closes review item m10 / P2.21 on the verification side. It does not edit the draft. Line numbers below are the draft's lines at `dd3e372`. A second check of the restructured draft, its sources and the two items it closed are in §6.

**How each reference was checked.** Existence, title, authors and year came from the arXiv API (all 13 arXiv ids) and Crossref (every DOI listed below). Venues came from the paper's own header or arXiv comment, PMLR, the ACL Anthology, the NeurIPS proceedings and Crossref. For each claim the draft attributes, I read the abstract. Where the abstract was not enough, I read the relevant section of the full text: Ge et al. v2, Krsteski and Meyer v1 and v2, Moreno Cencerrado et al. v3, Truong et al. v1, Zhou et al. (arXiv v2 and the Nature full text via Europe PMC), and Rue et al. (the February 2007 preprint copy, `www2.stat.duke.edu/~scs/Courses/Stat376/Papers/INLAOrig.pdf`, because the JRSS B version is paywalled). For Breslow and Clayton I read the abstract via OpenAlex. Scholar Sidekick was unavailable (no subscription), and DBLP did not respond. Scratch copies are in the session scratchpad, not in the repository.

**Verdict in one line.** Every reference exists, and no title is fabricated. Five in-text claims need rewording: one terminology error (INLA), one misattribution (Moreno Cencerrado et al.), and three overstatements (Ge et al. twice, Truong et al.). One more needs a supporting citation (Breslow and Clayton). Most entries lack venues, and three in-text year labels change if the list cites versions of record. Rasch (1960) is listed but not cited.

---

## 1 Claims that need rewording (most serious first)

### 1.1 "This is the Gaussian strategy of INLA" (line 211). Wrong term

INLA's *Gaussian* strategy (Rue et al. 2009, §3.2.1) uses the marginal of the Gaussian approximation at the mode, with its mean and variance. That is exactly what the draft says under-reacts. The draft's line instead evaluates the exact log joint posterior on a grid along the Gaussian conditional mean. This is the numerator of INLA's *Laplace* strategy (§3.2.2), with the paper's first simplification: the modal configuration is replaced by the conditional mean of the Gaussian approximation, x*₋ᵢ(xᵢ) ≈ E_G(x₋ᵢ | xᵢ), eq. (12) in the 2007 preprint. As described in the draft and in `paiec/hier.py` ("the exact log posterior on a grid"), the line leaves out the denominator, the Gaussian approximation of the conditional rebuilt at each point.

Proposed text, which changes no number: *"This is INLA's Laplace strategy with its conditional-mean simplification, the conditional mode replaced by the Gaussian conditional mean [Rue, Martino and Chopin 2009, §3.2.2], without the conditional Gaussian's normalising term."*

The same mislabel is in the docstring of `paiec/hier.py` (lines 66–67). That docstring then calls a refit at each node "INLA's Laplace strategy" (line 99), which is the full version without the simplification. This lane edits neither file. Whoever owns the draft should keep the code and the text consistent.

### 1.2 "Pre-generation activations predict a model's *own* success better than length or TF-IDF [Lugoloobi et al. 2026; Cencerrado et al. 2025]" (line 865). Half misattributed

Lugoloobi et al. (2026) report probes "substantially outperforming surface features such as question length and TF-IDF" (abstract). Moreno Cencerrado et al. compare against embedding-based assessors and verbalised confidence (§4.2), not against length or TF-IDF. They also report that the probe generalises poorly on mathematical reasoning (abstract; GSM8K near chance), which is relevant to §6.8.

Proposed text: *"Pre-generation activations predict a model's own success better than length or TF-IDF [Lugoloobi et al. 2026] and better than embedding assessors and verbalised confidence, though not on mathematical reasoning [Moreno Cencerrado et al. 2026]."*

### 1.3 Ge et al. (2026): three attributions, two overstated

- **Line 751: "For bug-fixing tasks, Agent Psychometrics predicts task-level success on unseen benchmarks from issue statements together with repository context, solutions and test cases."** The four benchmarks are SWE-bench Verified, SWE-bench Pro, GSO and Terminal-Bench 2.0. The held-out-benchmark test uses SWE-bench Pro and GSO, and GSO is software optimisation, not bug fixing (the paper says so in Appendix F.2). Replace "For bug-fixing tasks" with "For agentic coding tasks". The rest is supported by the abstract and §5.
- **Line 815: "Agent Psychometrics finds repository and patch size informative".** The paper finds that repository state, test patches and solution patches add predictive power beyond the problem statement (§5.1 Table 3; §6). It does not single out *size*. The judge rubrics include a codebase-scale and a solution-complexity feature (Appendix F), but no result is reported for them alone. Proposed: *"Agent Psychometrics finds that repository state, tests and the solution patch add predictive power beyond the issue text [Ge et al. 2026], but none of these is visible in the competition's item input."*
- **Line 830: "as Ge et al. find for the scaffold component".** Ge et al. find that agent ability decomposes additively into LLM and scaffold abilities, and they rank 72 scaffolds. In their Table 22, Agentless 1.5 sits at +0.715 and SWE-agent at -0.706 (SWE-agent 1.0 at -0.823). This matches the draft's direction and size. They do not claim that the scaffold is the largest component. Proposed: *"...as Ge et al. find a sizeable additive scaffold component, with Agentless above SWE-agent [Ge et al. 2026]."*

### 1.4 "Amortized calibration ... reports generalisation across datasets [Truong et al. 2025]" (line 751). Overstated

Truong et al. fit one difficulty regressor on Llama-3-8B embeddings across 22 datasets, with a dataset description prepended to each question. They argue that this "enables the generalization" across datasets (§4.1). The reported test (§5.2, Fig. 4) is on held-out responses and questions of datasets that are in training. No held-out-dataset evaluation appears in arXiv v1, the only arXiv version.

Proposed: *"Amortized calibration fits one map from embedded question content to difficulty across 22 datasets and matches per-question calibration on held-out questions [Truong et al. 2025]; it does not test a held-out dataset."* This sharpens the contrast with §6.1 and changes no number.

Line 1129 ("Its core is a difficulty map from embedded item content, trained across datasets") is accurate.

### 1.5 "the classic penalised-quasi-likelihood failure [Breslow and Clayton 1993]" (line 209). Supported in kind, needs a stronger source

Breslow and Clayton introduce PQL. They state the binary-data problem mildly: "PQL tends to underestimate somewhat the variance components" and fixed effects for clustered binary data (abstract). The *failure* in the draft's regime (large item variance, one binary label per item) is documented by Rodríguez and Goldman (1995), who report "very substantial downward bias when the random effects are sufficiently large", and by Breslow and Lin (1995), who find first-order estimators "seriously biased" for binary matched pairs.

Proposed: *"...is penalised quasi-likelihood [Breslow and Clayton 1993], whose bias with binary responses and large random effects is well documented [Rodríguez and Goldman 1995; Breslow and Lin 1995]..."* The phrase "classic ... failure" can stay once these two are cited.

---

## 2 Claims that check out

| line | claim | source says | verdict |
|---|---|---|---|
| 751 | ADeLe: black-box predictors on embeddings or fine-tuning are weaker than demand levels, especially out of distribution | arXiv v2 abstract and the Nature text: superior to "black-box assessor baselines based on embeddings or fine-tuning, especially in out-of-distribution settings" | supported; "ADeLe" is the paper's name (annotated-demand-levels) |
| 769 | human-labelled difficulty of maths and coding problems is linearly decodable from activations | abstract: maths and coding subsets of Easy2HardBench, AMC ρ ≈ 0.88 | supported |
| 769, 865 | rubric-based demand annotation predicts instance-level performance | abstract | supported |
| 792 | quote "no clear winner emerges and the overall performance is worse" | verbatim in the abstract; in-distribution parity with LLM-specific assessors also in the abstract | supported |
| 815 | task length in human time predicts agent success on software tasks | 50%-time-horizon framework; success falls with human task time | supported |
| 835 | observational scaling laws: performance from a low-dimensional capability space shared across families | abstract | supported |
| 845 | active and anchor-point selection choose informative items for a population of models | Li et al.: learned acquisition policy over prompts; Vivek et al.: anchor points correlated across models; Maia Polo et al.: IRT-based curated subsets | supported |
| 865 | 17 agentic benchmarks; entropy Spearman 0.19 under K-fold, 0.14 leave-one-benchmark-out | v2 §5.1: ρ = 0.193 (KF) and 0.137 (LOBO); identical in v1 | supported; for context, their length baseline gets 0.101 LOBO and the full feature set 0.225 |
| 865 | weak cross-benchmark transfer for agentic tasks | §6: "transferring these predictions to unseen benchmarks remains an open problem" | supported |

---

## 3 Bibliographic corrections

| entry in the draft | issue | correction |
|---|---|---|
| Breslow and Clayton (1993), *JASA* 88(421) | pages missing | 9–25; doi:10.1080/01621459.1993.10594284 |
| Cencerrado, I. V. M., et al. (2025) | name form and venue (`TODO(verify)`) | First author's surnames are **Moreno Cencerrado** (given names Iván Vicente; paper header and e-mail). Five authors: Moreno Cencerrado, I. V., Padrés Masdemont, A., Gonzalvez Hawthorne, A., Africa, D. D. and Pacchiardi, L. Venue: ICLR 2026 Workshop on Principled Design for Trustworthy AI (poster), v3 of 2026-03-03. In-text label becomes [Moreno Cencerrado et al. 2026] under a version-of-record style, or [Moreno Cencerrado et al. 2025] if arXiv years are kept |
| Ge, C., Kryvosheieva, D., Fried, D., et al. (2026), COLM 2026 | correct | All five authors: Ge, C., Kryvosheieva, D., Fried, D., Girit, U. and Hariharan, K. COLM 2026 is confirmed by the paper's header ("Published as a conference paper at COLM 2026") |
| Krsteski and Meyer (2026) | venue missing; v2 exists | COLM 2026 Workshop on Agent Behavior (v2 header, 2026-09-27; v1 said "Under review"). The numbers cited are the same in v1 and v2 |
| Kwa, T., West, B., Becker, J., et al. (2025) | title `TODO(verify)`; venue missing | The draft's title is right for the version of record. v1 and v2 (March 2025) are "Measuring AI Ability to Complete Long Tasks"; v3 and v4 and the NeurIPS 2025 proceedings use "Measuring AI Ability to Complete Long Software Tasks". Venue: NeurIPS 2025 (arXiv journal-ref; proceedings PDF). 26 authors in v4. The review's recollection (m10) is the v1 title |
| Li, Y., Ma, J., Ballesteros, M., Benajiba, Y. and Horwood, G. (2024) | venue missing; year | ICML 2025, PMLR 267: 35581–35602. In-text label [Li et al. 2024] becomes [Li et al. 2025] under a version-of-record style |
| Lugoloobi and Russell (2025) | none | arXiv preprint only; no venue found |
| Lugoloobi, Foster, Bankes and Russell (2026), COLM 2026 | correct | arXiv comment "Accepted at COLM 2026" (v4) |
| Maia Polo et al. (2024) | venue missing | ICML 2024, PMLR 235: 34303–34326 |
| Pacchiardi, Cheke and Hernández-Orallo (2024) | venue missing | KDD 2024 Workshop on Evaluation and Trustworthiness of Generative AI Models (arXiv comment). Middle initial: Cheke, L. G. |
| Rasch (1960) | publisher missing; not cited | Copenhagen: Danish Institute for Educational Research (Danmarks Paedagogiske Institut), Studies in Mathematical Psychology I; expanded edition 1980, University of Chicago Press. Cite it at line 105 ("Rasch estimates") or line 205 ("item-level (Rasch) model") |
| Ruan, Maddison and Hashimoto (2024) | venue missing | NeurIPS 2024 (spotlight) |
| Rue, Martino and Chopin (2009), *JRSS B* 71(2) | pages missing | 319–392; doi:10.1111/j.1467-9868.2008.00700.x |
| Truong, S., Tu, Y., Liang, P., Li, B. and Koyejo, S. (2025) | title, authors and venue `TODO(verify)` | All correct. Venue: ICML 2025, PMLR 267: 60238–60265. PMLR gives the first author as Sang T. Truong (Truong, S. T.). Not a conflict, but worth knowing: Sang Truong and Sanmi Koyejo are on PAIEC's organiser list |
| Vivek, Ethayarajh, Yang and Kiela (2024) | venue missing | EACL 2024 (Volume 1: Long Papers), 1576–1601; doi:10.18653/v1/2024.eacl-long.95 |
| Zhou, L., Pacchiardi, L., Martínez-Plumed, F., Collins, K. M., et al. (2025) | version of record exists | *Nature* 652(8108): 58–67 (2026); doi:10.1038/s41586-026-10303-2; open access (PMC13043289). 26 authors. In-text label [Zhou et al. 2025] (lines 751, 769, 865 twice) becomes [Zhou et al. 2026] under a version-of-record style |

**One style choice decides three labels.** The draft already uses venue years: Vivek et al. 2024 for an arXiv 2023 preprint. Applied throughout, three in-text labels change: Li et al. 2024 → 2025, Zhou et al. 2025 → 2026, and Cencerrado et al. 2025 → Moreno Cencerrado et al. 2026. These are citation labels, not results, so the "no number changes" rule is not affected. If the team prefers arXiv years, keep the labels and add the venue to each entry.

**Housekeeping in the draft.** Once this list is applied, the paragraph after the References ("Titles were checked against local copies of 12 of the 13 arXiv papers ... `TODO(verify)`") and the Appendix B item "`TODO(verify)` The references listed after the bibliography" can be closed.

---

## 4 Corrected References (one style; new entries marked +)

Style: author-year; venue year where a venue exists; arXiv id for every arXiv-first work; DOI where one exists. Titles are in sentence case, as in the draft. Entries marked + are proposed additions from §5 below. Add only those the text actually cites.

- + Barton, M. A. and Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series* 1981(1). doi:10.1002/j.2333-8504.1981.tb01255.x
- + Benedetto, L., Cremonesi, P., Caines, A., Buttery, P., Cappelli, A., Giussani, A. and Turrin, R. (2023). A survey on recent approaches to question difficulty estimation from text. *ACM Computing Surveys* 55(9): 1–37. doi:10.1145/3556538
- + Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1): 1–3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- Breslow, N. E. and Clayton, D. G. (1993). Approximate inference in generalized linear mixed models. *Journal of the American Statistical Association* 88(421): 9–25. doi:10.1080/01621459.1993.10594284
- + Breslow, N. E. and Lin, X. (1995). Bias correction in generalised linear mixed models with a single component of dispersion. *Biometrika* 82(1): 81–91. doi:10.1093/biomet/82.1.81
- + De Boeck, P. and Wilson, M. (eds.) (2004). *Explanatory item response models: a generalized linear and nonlinear approach.* New York: Springer. doi:10.1007/978-1-4757-3990-9
- + Fischer, G. H. (1973). The linear logistic test model as an instrument in educational research. *Acta Psychologica* 37(6): 359–374. doi:10.1016/0001-6918(73)90003-6
- + Fox, J.-P. (2010). *Bayesian item response modeling: theory and applications.* New York: Springer. doi:10.1007/978-1-4419-0742-4
- Ge, C., Kryvosheieva, D., Fried, D., Girit, U. and Hariharan, K. (2026). Agent psychometrics: task-level performance prediction in agentic coding benchmarks. In *Conference on Language Modeling (COLM 2026)*. arXiv:2604.00594
- + Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477): 359–378. doi:10.1198/016214506000001437
- Krsteski, S. and Meyer, C. (2026). Predicting task difficulty without rollouts. *COLM 2026 Workshop on Agent Behavior*. arXiv:2608.05797
- Kwa, T., West, B., Becker, J., et al. (2025). Measuring AI ability to complete long software tasks. In *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2503.14499
- + Lalor, J. P., Wu, H. and Yu, H. (2016). Building an evaluation scale using item response theory. In *Proceedings of EMNLP 2016*, 648–657. doi:10.18653/v1/D16-1062
- Li, Y., Ma, J., Ballesteros, M., Benajiba, Y. and Horwood, G. (2025). Active evaluation acquisition for efficient LLM benchmarking. In *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267: 35581–35602. arXiv:2410.05952
- + Lipton, Z. C., Wang, Y.-X. and Smola, A. (2018). Detecting and correcting for label shift with black box predictors. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*. arXiv:1802.03916
- + Liu, Q. and Pierce, D. A. (1994). A note on Gauss–Hermite quadrature. *Biometrika* 81(3): 624–629. doi:10.1093/biomet/81.3.624
- Lugoloobi, W. and Russell, C. (2025). LLMs encode how difficult problems are. arXiv:2510.18147
- Lugoloobi, W., Foster, T., Bankes, W. and Russell, C. (2026). LLMs encode their failures: predicting success from pre-generation activations. In *Conference on Language Modeling (COLM 2026)*. arXiv:2602.09924
- Maia Polo, F., Weber, L., Choshen, L., Sun, Y., Xu, G. and Yurochkin, M. (2024). tinyBenchmarks: evaluating LLMs with fewer examples. In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235: 34303–34326. arXiv:2402.14992
- + Martínez-Plumed, F., Prudêncio, R. B. C., Martínez-Usó, A. and Hernández-Orallo, J. (2019). Item response theory in AI: analysing machine learning classifiers at the instance level. *Artificial Intelligence* 271: 18–42. doi:10.1016/j.artint.2018.09.004
- Moreno Cencerrado, I. V., Padrés Masdemont, A., Gonzalvez Hawthorne, A., Africa, D. D. and Pacchiardi, L. (2026). No answer needed: predicting LLM answer accuracy from question-only linear probes. *ICLR 2026 Workshop on Principled Design for Trustworthy AI*. arXiv:2509.10625
- + Morris, C. N. (1983). Parametric empirical Bayes inference: theory and applications. *Journal of the American Statistical Association* 78(381): 47–55. doi:10.1080/01621459.1983.10477920
- + Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In *Proceedings of the AAAI Conference on Artificial Intelligence* 29(1). doi:10.1609/aaai.v29i1.9602
- Pacchiardi, L., Cheke, L. G. and Hernández-Orallo, J. (2024). 100 instances is all you need: predicting the success of a new LLM on unseen data by testing on a few instances. *KDD 2024 Workshop on Evaluation and Trustworthiness of Generative AI Models*. arXiv:2409.03563
- + Patz, R. J. and Junker, B. W. (1999). A straightforward approach to Markov chain Monte Carlo methods for item response models. *Journal of Educational and Behavioral Statistics* 24(2): 146–178. doi:10.3102/10769986024002146
- + Pinheiro, J. C. and Bates, D. M. (1995). Approximations to the log-likelihood function in the nonlinear mixed-effects model. *Journal of Computational and Graphical Statistics* 4(1): 12–35. doi:10.1080/10618600.1995.10474663
- Rasch, G. (1960). *Probabilistic models for some intelligence and attainment tests.* Copenhagen: Danish Institute for Educational Research. Expanded edition 1980, Chicago: University of Chicago Press.
- + Rodriguez, P., Barrow, J., Hoyle, A. M., Lalor, J. P., Jia, R. and Boyd-Graber, J. (2021). Evaluation examples are not equally informative: how should that change NLP leaderboards? In *Proceedings of ACL-IJCNLP 2021 (Volume 1: Long Papers)*, 4486–4503. doi:10.18653/v1/2021.acl-long.346
- + Rodríguez, G. and Goldman, N. (1995). An assessment of estimation procedures for multilevel models with binary responses. *Journal of the Royal Statistical Society, Series A* 158(1): 73–. doi:10.2307/2983404 (end page not given by Crossref or OpenAlex; take it from the publisher's page. *Closed in §6 below: 73–89.*)
- Ruan, Y., Maddison, C. J. and Hashimoto, T. (2024). Observational scaling laws and the predictability of language model performance. In *Advances in Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2405.10938
- Rue, H., Martino, S. and Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. *Journal of the Royal Statistical Society, Series B* 71(2): 319–392. doi:10.1111/j.1467-9868.2008.00700.x
- + Saerens, M., Latinne, P. and Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation* 14(1): 21–41. doi:10.1162/089976602753284446
- + Tierney, L. and Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal densities. *Journal of the American Statistical Association* 81(393): 82–86. doi:10.1080/01621459.1986.10478240
- Truong, S. T., Tu, Y., Liang, P., Li, B. and Koyejo, S. (2025). Reliable and efficient amortized model-based evaluation. In *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267: 60238–60265. arXiv:2503.13335
- Vivek, R., Ethayarajh, K., Yang, D. and Kiela, D. (2024). Anchor points: benchmarking models with much fewer examples. In *Proceedings of EACL 2024 (Volume 1: Long Papers)*, 1576–1601. doi:10.18653/v1/2024.eacl-long.95. arXiv:2309.08638
- + Ye, Q., Fu, H. Y., Ren, X. and Jia, R. (2023). How predictable are large language model capabilities? A case study on BIG-bench. In *Findings of EMNLP 2023*. arXiv:2305.14947
- + Zhang, Q., Lyu, F., Liu, X. and Ma, C. (2024). Collaborative performance prediction for large language models. In *Proceedings of EMNLP 2024*. arXiv:2407.01300
- Zhou, L., Pacchiardi, L., Martínez-Plumed, F., Collins, K. M., et al. (2026). General scales unlock AI evaluation with explanatory and predictive power. *Nature* 652(8108): 58–67. doi:10.1038/s41586-026-10303-2. arXiv:2503.06378

**Competition resources.** The draft cites these in §8.2 but not in the References. A reader outside the competition needs them.

- + PAIEC organisers (2026). Predictive AI Evaluation Competition at NeurIPS 2026. https://aimslab.stanford.edu/competition (organisers: S. Truong, W. Salaudeen, N. Truong, S. Wang, A. Wang, L. Guerdan, Z. Xiao, X. Xie, N. Haber, S. Koyejo; read 2026-10-02)
- + measurement-db (2026). Hugging Face dataset `aims-foundations/measurement-db`, revision `bc8204d811823da849c6686bf124d4ca9f82e4de`, CC-BY-SA-4.0. This is the draft's pinned revision, and it is still the live one (Hub API, last modified 2026-09-18).
- + PAIEC baseline repository. https://github.com/aims-foundations/paiec_baseline, commit `82d330dd` (as in §8.2).

The dataset card's preferred citation could not be read without the gated token, and this lane did not use one. Check the card for a BibTeX entry before submission. *Superseded (§6 below): the card is public, and the draft now cites its preferred entry.*

---

## 5 Core references the draft is missing (all verified above)

Each item says where it would attach. The list is short on purpose: the draft cites about 16 works, and an outside reader mainly needs the IRT, inference and scoring lineage.

- **IRT and Rasch.** Rasch (1960) is already listed; cite it at line 105 or 205. The model's floor and ceiling (`p = c + (1 - c - slip) σ(η)`, §3.1) is the four-parameter logistic form: cite Barton and Lord (1981) for the upper asymptote. Lord and Novick (1968), *Statistical Theories of Mental Test Scores* (Addison-Wesley), is the usual source for the three-parameter guessing floor (Birnbaum's chapters). It is only catalogue-verified here: Open Library lists the 1968 Addison-Wesley book under Lord. Confirm the co-author line before citing it.
- **Explanatory IRT (item_features group effects g_i, §3.1 table).** Fischer (1973) for the LLTM, and De Boeck and Wilson (2004) for item covariates in a GLMM frame. Review m10 asks for this.
- **Hierarchical Bayesian IRT (the abstract's "hierarchical Bayesian item-response model").** Patz and Junker (1999); Fox (2010).
- **IRT for ML and NLP evaluation (§1 or §6 literature).** Lalor et al. (2016); Martínez-Plumed et al. (2019); Rodriguez et al. (2021). Review m10 asks for Lalor and Rodriguez. Vania et al. (2021), "Comparing test sets with item response theory", ACL-IJCNLP 2021, 1141–1158, doi:10.18653/v1/2021.acl-long.92, is an optional fourth.
- **Approximate GLMM inference (§3.2).** Liu and Pierce (1994) for adaptive Gauss–Hermite, which §3.2 uses with 20 nodes. Pinheiro and Bates (1995) compare Laplace and adaptive quadrature; review m10 asks for this. Tierney and Kadane (1986) for Laplace approximations to posterior marginals, next to Rue et al. Pinheiro and Chao (2006), *JCGS* 15(1): 58–81, doi:10.1198/106186006X96962, is optional.
- **Empirical Bayes (§3.3 "empirical-Bayes hyperparameters", the EB level of §5.3).** Morris (1983).
- **Level calibration as prior or label shift (§3.4, review m10).** Saerens et al. (2002) and Lipton et al. (2018). Both re-estimate a base rate under shift, which is what the level prior and the EB level do for a new benchmark.
- **Predicting model performance on benchmarks (§1 or §6.6, next to Ruan et al.).** Ye et al. (2023) predict BIG-bench performance from other evaluation records. Zhang et al. (2024) do collaborative, matrix-factorisation-style performance prediction across models and tasks.
- **Difficulty of a new item from its text (§6.1).** Benedetto et al. (2023) survey question-difficulty estimation from text. With Truong et al., Ge et al. and Krsteski and Meyer, this covers the cold-start side.
- **Brier and calibration (§1.1 scoring; ECE in §5.1 tables).** Brier (1950) and Gneiting and Raftery (2007) for the score. Naeini et al. (2015) for ECE, which the platform reports as a diagnostic.

---

## 6 Second check, 2026-10-02 (verification pass on the restructured draft)

A verifier re-checked the References of the restructured draft (41 entries, each cited at least once; every bracketed in-text citation resolves to an entry), and the draft was corrected the same day. Line numbers in §1 to §5 above are those of `dd3e372` and no longer match the draft.

**Sources consulted, by the first check (§1 to §5) and this one.**

- arXiv API, one query per id: `https://export.arxiv.org/api/query?id_list=<id>` (titles, authors, versions, comments, journal-ref of the 16 arXiv ids; v2 headers of Ge et al. and Krsteski and Meyer for their COLM 2026 venues).
- Crossref: `https://api.crossref.org/works/<doi>` for the 23 DOIs (title, authors, year, venue, volume, pages).
- Semantic Scholar: `https://api.semanticscholar.org/graph/v1/paper/DOI:10.2307/2983404?fields=title,journal,year` (Rodríguez and Goldman's pages).
- PMLR index pages: https://proceedings.mlr.press/v80/ (Lipton et al.), https://proceedings.mlr.press/v235/ (Maia Polo et al.), https://proceedings.mlr.press/v267/ (Li et al., Truong et al.).
- The ACL Anthology pages of Lalor et al., Rodriguez et al., Vivek et al., Ye et al. and Zhang et al., and the NeurIPS proceedings pages of Kwa et al. and Ruan et al.
- Europe PMC, PMC13043289 (Zhou et al., *Nature* full text).
- Hugging Face Hub: https://huggingface.co/api/datasets/aims-foundations/measurement-db (revision and licence) and the dataset card, https://huggingface.co/datasets/aims-foundations/measurement-db/resolve/main/README.md, read 2026-10-02 without a token (HTTP 200). Only the card was read, no data file.

**The two open items are closed.**

- **measurement-db's preferred citation.** The dataset card is public, not gated (only the data files are). Its "Citation" section gives a `@misc` entry: *The AI Measurement Data Bank*, by Nhi Truong, Sang T. Truong and Sanmi Koyejo, 2026, AIMS Lab, Stanford University, https://aimslab.stanford.edu/measurement-db. Main is revision `bc8204d811823da849c6686bf124d4ca9f82e4de` (last modified 2026-09-18), the draft's pinned revision; the licence is CC-BY-SA-4.0. The draft's entry is now "Truong, N., Truong, S. T. and Koyejo, S. (2026). The AI Measurement Data Bank (measurement-db)...", cited as [Truong et al. 2026] in §2.2. Two of its authors are among the competition's organisers.
- **Rodríguez and Goldman (1995), end page.** Semantic Scholar gives *JRSS A* 158, pages 73–89; Crossref and OpenAlex give the first page only. The entry now reads 158(1): 73–89.

**Entries completed.**

| entry | added |
|---|---|
| Lalor, Wu and Yu (2016) | arXiv:1605.08889 (v1 of 2016-05-28, before EMNLP 2016; the style gives every arXiv-first work its id) |
| Lipton, Wang and Smola (2018) | PMLR 80: 3122–3130 |
| Ye, Fu, Ren and Jia (2023) | 7493–7517; doi:10.18653/v1/2023.findings-emnlp.503 |
| Zhang, Lyu, Liu and Ma (2024) | 2576–2596; doi:10.18653/v1/2024.emnlp-main.150 |

Not changed: Naeini et al. (2015). Crossref gives the first author's surname as "Pakdaman Naeini"; the conventional "Naeini, M. P." and the label [Naeini et al. 2015] are kept.

**One attribution corrected (§1.4 of the restructured draft).** §1.4 credited IRT with choosing informative items for three works. Only Maia Polo et al. (tinyBenchmarks) is IRT-based. Vivek et al. select anchor points from correlations of model confidence across models; the v2 full text has no "item response" or "IRT". Li et al. learn an RL acquisition policy, with IRT only as a baseline and an alternative predictor. §1.4 now reads: small informative item sets are chosen with IRT [Maia Polo et al. 2024], anchor points [Vivek et al. 2024] or a learned acquisition policy [Li et al. 2025]. App F.8's wording ("active and anchor-point selection") was already accurate.

**Attributed claims re-checked and supported** (no change): Krsteski and Meyer (17 benchmarks; entropy ρ 0.193 under K-fold and 0.137 leave one benchmark out); the Pacchiardi et al. quote (verbatim in the abstract); Truong et al. (22 datasets, one global model, held-out questions within datasets); Ge et al. (Table 22: Agentless 1.5 at +0.715, SWE-agent at -0.706; additive LLM and scaffold abilities, §5.2); Moreno Cencerrado et al. (better than the baselines except on GSM8K); Lugoloobi et al. 2026 (better than question length and TF-IDF); Rue et al. §3.2.2, eq. (13), the conditional-mean simplification.

**Open.** Nothing in the reference list.
