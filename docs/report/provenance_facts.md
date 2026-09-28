# Provenance facts for the report (review P0.4, P0.5, P0.6)

These are facts for the writer, not report prose. They were gathered on 2026-09-28 at HEAD `4d2cc4f` (branch `official-protocol`), from the repository, `results/` and session scratch only; no network was used. Each fact names its source. "Scratch" means the session scratchpad `…/9df8ecfa-42c0-4508-919c-f794f1770909/scratchpad/`.

---

## 1 Pinned inputs (P0.5)

### 1.1 measurement-db

- **Revision.** `bc8204d811823da849c6686bf124d4ca9f82e4de`.
  - Source: `data/.cache/huggingface/download/<benchmark>/<table>.parquet.metadata`, written by `huggingface_hub` next to each file it downloads (line 1 revision, line 2 etag, line 3 timestamp).
  - All 24 parquet files and the dataset card (`README.md.metadata`) name this revision.
- **When.** 2026-09-24 08:06:46 to 08:07:26 UTC (the metadata timestamps).
- **Integrity.** For all 24 files, the local file's sha256 equals the recorded etag, which is the LFS sha256. So `data/` is byte for byte the dataset at that revision.

  | benchmark | response | items | subjects | benchmarks |
  |---|---|---|---|---|
  | matharena | `33aa835e…` 1,748,755 B | `fb420494…` 310,264 B | `aeec4d5a…` 13,588 B | `1a57bc6b…` 16,804 B |
  | mmdocrag | `bd4030df…` 6,439,197 B | `ee84a597…` 388,955 B | `e9652040…` 9,839 B | `7fdaa5b0…` 16,854 B |
  | multi_swebench | `33d31a4f…` 1,180,533 B | `eaf624e7…` 6,170,506 B | `06eca4d3…` 9,456 B | `89efc2f5…` 16,919 B |
  | real_webagents | `b42381f5…` 94,888 B | `5908db1f…` 48,588 B | `e563ff23…` 8,365 B | `49454f9a…` 16,595 B |
  | researchcodebench | `246db2bc…` 137,884 B | `3aee135a…` 8,928,605 B | `c8dbb5ac…` 8,321 B | `d1c8acda…` 16,391 B |
  | swe_rebench | `2674db69…` 1,505,550 B | `78fd183a…` 39,763,649 B | `4f2f5f34…` 6,747 B | `599823ff…` 17,988 B |

- **The pin in code.** `paiec/fetch.py` now has `REVISION` set to that SHA and passes `revision=` to every `hf_hub_download`.
  - Override with `PAIEC_DATA_REVISION`, or `python -m paiec.fetch --revision <sha|branch|tag>`. Behaviour is otherwise unchanged.
  - The download message now names the revision.
- **The pin's test.** `tests/test_fetch.py`, 5 tests, no network. It covers:
  - every download carries the pinned revision;
  - the override order;
  - failures are collected;
  - a local download, when present, was made at the pinned revision. This test reads the `.metadata` files and skips if there is no `data/`.
- **Not knowable offline.** The date of that commit on the Hub, and whether `main` has moved since. With network, `huggingface_hub.HfApi().list_repo_commits("aims-foundations/measurement-db", repo_type="dataset")` answers both (needs the gated access).
- **License.** The dataset card at the same revision says `license: cc-by-sa-4.0` (`data/README.md`, lines 2 and 231–237).
- **Other local files.** `data/README.md` (the card) was also fetched at this revision. `fetch.py` does not fetch it.

### 1.2 The organisers' baseline repository

- **Clone.** `third_party/paiec_baseline`, HEAD `82d330ddcdb16016a3ae9e048db7e588ba2c6a39`, dated 2026-09-15 20:57:37 -0700, "Add empirical mean baseline with BLE acquisition".
  - Cloned 2026-09-24 10:29:28 +0300 (the reflog's only entry). Never pulled since.
  - The working tree is clean. Origin: `https://github.com/aims-foundations/paiec_baseline`.
- **File hashes** (sha256):
  - `tools/streaming_ingestion.py` `c6f2610f9c8b6cad8a2338ddc07a0cb25d40bca761a6628937a2cee2705a6267`;
  - `check_submission_zip.py` `abe3cae05f4b18450b36edca60faef8a39bdf21821f30ff195c3cc45e177c646`.
- **To pin in §8.2:** `git clone https://github.com/aims-foundations/paiec_baseline third_party/paiec_baseline && git -C third_party/paiec_baseline checkout 82d330ddcdb16016a3ae9e048db7e588ba2c6a39`.
- `docs/protocol.md` names the file it mirrors but not the commit.

### 1.3 Environment actually installed

- **Machine.** Apple M1 Pro (8 cores, 16 GiB), macOS 26.5.1 (build 25F80), Darwin 25.5.0, arm64.
- **Interpreter.** `python` resolves to Anaconda's Python 3.10.8 (conda-forge build, Clang 14.0.6), `~/anaconda3/bin/python`. Note that `python3` resolves to a different interpreter (`/opt/local/bin/python3`) that lacks pandas; every command in the repository means `python`.
- **Libraries.**

  | library | version |
  |---|---|
  | numpy | 1.26.4 (OpenBLAS 0.3.23.dev) |
  | scipy | 1.10.1 |
  | scikit-learn | 1.7.2 |
  | pandas | 2.2.2 |
  | pyarrow | 23.0.1 |
  | torch | 2.6.0 |
  | transformers | 4.57.6 |
  | tokenizers | 0.22.2 |
  | safetensors | 0.8.0 |
  | huggingface_hub | 0.36.2 |
  | pytest | 9.0.3 |

  accelerate is not installed. pandas warns at import that bottleneck 1.3.5 is older than it wants (1.3.6); harmless.
- **Language models used offline.** Revisions confirmed in the local Hugging Face cache (`refs/main` and the only snapshot):
  - Qwen3-Embedding-0.6B `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`;
  - Qwen3-4B-Instruct-2507 `cdbee75f17c01a7cc42f958dc650907174af0554`.

  These match draft §8.6.

### 1.4 Which archive produced which formative run

- **Run 1** (2026-09-25, ALC 0.2113). The legacy Predictor, commit `b68492c` (as given to this lane).
  - Candidate archive: sha256 `8e28d930d45b16aee351bfbbc75531a6d12079667793232d4bd74a17f9d47b9f`, built 2026-09-24 13:26 +0300 (`scratch/mine/staged.zip`). Identical copies are `scratch/verify_sub/paiec.zip` and `verify_sub/r2/paiec2.zip`.
  - Every code member (`model.py`, 6 `paiec_rt` modules, `requirements.txt`) is byte-identical to `b68492c`, and to `bba726c`, which did not touch them. `prior.json` is the legacy `{coef, spec}` bundle (541 bytes).
  - A variant with `labeling.py` (`verify_sub/r2/paiec_lab.zip`, `0084a247…`) also exists. The draft says the submission ships no `labeling.py`, which points to `8e28d930…`.
  - The file the team downloaded back from the platform's submission page on 2026-09-28 has sha256 `8e28d930…`, so this is the uploaded file (`results/formative_feedback.json`, `archives.run1`).
- **Run 2** (2026-09-26, ALC 0.192623). The archive is `scratch/step5/wrapup/old/paiec.zip`, sha256 `2c64eaada491cbf85bd54ae190cdbb28f9d0a13870df0dc77a3a850f90cf661f`. The file the team downloaded back from the platform's submission page on 2026-09-28 has the same sha256, so this is the uploaded file (`results/formative_feedback.json`, `archives.run2`).
  - File time 2026-09-26 09:20 +0300, two minutes before commit `ee5085a` (09:22:17 +0300).
  - Every code member is byte-identical to `ee5085a`: `model.py` = `submission/model.py`, the eight `paiec_rt/*.py` = `paiec/*.py`, and `requirements.txt`.
  - `model.py` differs at `bd0be67`, which changed only its docstring.
  - Its `prior.json` (sha256 `c7ce3b845d584f953d34e282d94065c379dc3d3b88e4cb0dee95a23c23282770`) is byte-identical to today's `submission/prior.json`.
  - Its `paiec_rt` modules predate the corrected multiple-choice floor (`f7e7d87` changed `mcq.py`, `prior.py` and `subjects.py`) and the floored-fit fix (`4d2cc4f` changed `hier.py`).
- **Current archive.** sha256 `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090` (as given).
  - A scratch copy (`scratch/step5/wrapup/verify/rezip.zip`, 2026-09-27 23:55) has that hash, and its code members are byte-identical to `4d2cc4f`.
  - This lane could not read `dist/` itself (a local hook blocks it).

---

## 2 Scratch provenance (P0.6)

All three studies the review names are now in `experiments/`, and all three reproduce their documented numbers.

| study | scratch origin | now | output | re-run |
|---|---|---|---|---|
| meta-learned heads | `rethink2/fine-tuning/heads.py` (+ `collect.py`, `dtab.py`, `queue_a.sh`, `queue_b.sh`) | `experiments/heads_eval.py` | `results/heads_eval.json` | yes, all 21 configurations, on both row sets |
| inventory classification | `rethink2/methodology-critic/c_inventory_meta.py` (C1), `hand_labels.csv`; `rethink2/attempt-signals/a6_inventory_formats.py`, `a6b_…`, `a6c_…` | `experiments/inventory_classes.py` | `results/inventory_classes.json`, `results/inventory_classes.csv`; input `results/inventory_hand_labels.csv` (new) | yes |
| step-2b audit | `step2b/audit/feedback_match.py`, `feedback_match2.py`, `feedback_match3.py`, `analyse.py`, `extra_runs.py`, `extra_summary.py`, `noshift_as.py` | `experiments/level_audit.py` | `results/level_audit.json` | yes (see 2.3 for what re-measures rather than reproduces) |

### 2.1 Meta-learned heads (draft §6.4, §4.5; findings "Acceptance harness")

**What it is.** Heads trained on episodes, sitting on frozen Qwen3-Embedding-0.6B features (PCA 64, whitened, fitted on the training parents, plus log length). They add a capped logit offset to the shipped hier's prediction. There are four heads:
- `diff`, a meta-learned difficulty direction, centred on labeled items;
- `kern`, a learned-metric few-shot kernel on hier's residuals;
- `ass`, `diff` plus a rank-2 subject × item term from provider, reasoning flag and release year;
- `all`, all three together.

Selection and data:
- λ is chosen from {0.01, 0.1, 1} or "off" by nested leave-one-parent-out; a head is used on the held-out parent only if its inner mean is below 0.
- Training uses test-like and mix/whole episodes of the three training parents.
- Scoring is on the held-out parent's test-like and mix/whole appearances: 300 runs (seed 2) and 150 runs (seed 3).

Controls:
- `oracle`: the same form with the parent's in-sample Rasch difficulty.
- `oracle_r<r>`: that oracle degraded to correlation r.
- `force_all_<λ>`: no selection.
- `indist_*`: subject folds within seen benchmarks.
- PCA 16 and 256.

**How it was made reproducible.** The model, training, selection and statistics code is unchanged; only data loading changed. The script reads `experiments/harness.py`'s rows. Checks run by this lane:
- The committed `data/harness_rows_legacy` equals the study's scratch rows exactly. 0 of 300 test-like and 0 of 150 mix/whole runs differ, over 2,135,316 evaluated response × budget cells and every labeled cell.
- The subject dicts and every item's text key match (3,623 slots).
- The rebuilt embedding table (4,078 texts) and in-sample difficulty table (4,035 texts) equal the scratch `emb_table.npz` and `dtab.npz` bit for bit.
- `data/harness_rows` (library `3f75a549673aae6a`, the current one) differs from the scratch rows on 157 of 300 test-like and 74 of 150 mix/whole runs (max 0.168), as findings says.

**Reproduction.** `python experiments/heads_eval.py --rows legacy` reproduces all 21 of the study's result files bit for bit: every float of the test-like and mix/whole summaries, per parent, per kind, and the nested choices. It took 3,223 s on 1 process with 2 torch threads, max RSS 1.07 GB. That is slower than the scratch runs' recorded 1,561 s because an unrelated 5-core job shared the machine. `--rows current` took 3,134 s.
- The legacy pass ran script digest `60cfb012…`. That version differs from the committed file (`6ad7190a…`, which the current pass ran) only in how the results file is saved: a re-read-and-merge was added so two row sets can share one file.
- The current-rows block was written to a side file beside the legacy run and merged into `results/heads_eval.json`.

**Results** (ALC difference against hier, ± run SE / (parent, subject) cluster SE, test-like; mix/whole cluster SE in brackets):

| configuration | legacy rows (the study as it ran) | current rows (library at 4d2cc4f) |
|---|---|---|
| nested `diff`, `kern`, `ass`, `all`; PCA 16 and 256 | switched off in 0 of 4 folds for every head, so exactly 0 | 0 of 4 folds for every head, so exactly 0 |
| smallest inner mean over folds and λ (off unless below 0) | +0.000007 (`ass`, matharena held out); `diff` +0.000009 | the same, +0.000007 and +0.000009 |
| forced `all`, λ 0.01 / 0.1 / 1 | +0.00064 ± 0.00011 / 0.00023; +0.00054; +0.00020 | +0.00064; +0.00053; +0.00019 |
| oracle (in-sample Rasch difficulty) | **-0.04371 ± 0.00116 / 0.00374**, mix/whole **-0.05566** [0.00620] | -0.04360 ± 0.00116 / 0.00373, mix/whole -0.05561 [0.00619] |
| oracle per held-out parent (test-like) | matharena -0.0488, multi_swebench -0.0342, real_webagents -0.0508, researchcodebench -0.0509 | -0.0482, -0.0343, -0.0507, -0.0509 |
| degraded oracle r 0.1 / 0.2 / 0.3 / 0.5 / 0.7 | -0.00009 / -0.00086 / **-0.00239** / **-0.00799** / -0.01765 | -0.00009 / -0.00087 / -0.00239 / -0.00798 / -0.01763 |
| in-distribution control (benchmarks seen), `diff` λ 0.001 / `all` PCA 256 | -0.00051 ± 0.00027 / 0.00084; -0.00169 ± 0.00035 / 0.00080 (mix/whole -0.00243, -0.00457) | -0.00048; -0.00139 (mix/whole -0.00239, -0.00426) |
| hier's own Brier, B0..B31 (test-like) | 0.2162, 0.1885, 0.1644, 0.1522, 0.1448, 0.1386 | 0.2153, 0.1880, 0.1642, 0.1521, 0.1448, 0.1385 |

**Numbers in the docs that depend on it, and their status:**

- **Findings "Acceptance harness".** "the heads study's -0.0437 and -0.0557", and `experiments/harness.py`'s `ACCEPT` target: reproduced, -0.04371 / -0.05566, `results/heads_eval.json` `rows.legacy.configs.lopo_oracle`. Cite that file instead of "the heads study".
- **Findings "The gate table".** "the heads study's single forced draws gave -0.0024 and -0.0080": reproduced, -0.00239 (r 0.3) and -0.00799 (r 0.5), `rows.legacy.configs.lopo_oracle_r0.3` and `lopo_oracle_r0.5`.
- **Findings "Provenance".** "Its scratch rows were computed with the old library": can now say the scratch rows equal `data/harness_rows_legacy` exactly (0 of 450 runs differ).
- **Findings "Fine-tuning an encoder".** "Every frozen-feature head since was switched off by nested selection": supported, in 0 of 4 folds for all six nested lines on both row sets.
- **Draft §4.5.** "the in-sample oracle that an earlier study scored: -0.0437 test-like and -0.0557 mix/whole": now has a committed source (above).
- **Draft §6 table, row "meta-learned heads on frozen embeddings".** The TODOs can be filled:
  - best honest estimate: nested 0 (every head off in every fold); forced +0.0002 to +0.0006;
  - verdict: no gain;
  - script: `experiments/heads_eval.py`, `results/heads_eval.json`.
- **Draft §6.4** (all TODO) and **Appendix B's** `TODO(record)` for heads: the result above. The in-distribution control shows the heads do learn something when the benchmark has been seen: -0.0005 to -0.0017 test-like, -0.0024 to -0.0046 mix/whole.
- **Draft §7** "the meta-heads study live[s] in session scratch": no longer true.
- **Not changed by this lane:**
  - `experiments/harness.py` still names the heads study's scratch rows for `--stage verify --scratch`; its format is the scratch one, so `data/harness_rows_legacy` cannot stand in without a code change.
  - The study's own scratch summary files remain in scratch.

### 2.2 Inventory classification (draft §2.1; findings "What this could do on the hidden test")

**What it is.** Keyword rules on the benchmark **title only**. Each title becomes "not an AI-system evaluation" or one of text_qa, math, code, agent, mm_image, mm_video, audio. No description, paper, repository or item is read.

**Inputs:**
- **The 161-row inventory, `results/inventory.csv`**, committed 2026-09-23 at `165b4c0`, sha256 `58336933…`.
  - Its 161 slugs are, in order, the first 161 rows of the organisers' inventory sheet (all NeurIPS 2025 Datasets and Benchmarks papers). The organisers link that sheet from the competition page: Google Sheet `1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI`.
  - The 161 rows differ from the later download only in apostrophes (3 titles) and author typos (2 rows).
- **The 1,261-row sheet**, downloaded 2026-09-24 (`scratch/web/sheet.csv`, identical to `scratch/inv.csv`, sha256 `882809d0…`).
  - Copied to `data/inventory_sheet.csv` (gitignored) and optional for the script.
- **80 validation labels.** `scratch/rethink2/methodology-critic/hand_labels.csv`, now `results/inventory_hand_labels.csv` with slug and title added.
  - The sample is `sheet.sample(80, random_state=11)` over the 1,261 rows (verified). Only 14 of the 80 fall among the 161.
  - **Who labelled:** the file has no author. It was written in session scratch on 2026-09-26 by the "methodology-critic" agent lane of the Claude session. So "hand labels" means labels assigned by an AI agent reading each title, not by a person. The team confirmed on 2026-09-28 that they are disclosed as such, not relabelled.

**Reproduction.** `python experiments/inventory_classes.py` takes seconds with no network. It reproduces every number of `c_inventory_meta.json`'s C1 part and of the three a6 JSON files exactly.
- Not moved: C2 (a random-effects meta-analysis) and C3 (units needed for a CI). No documented number uses them.

**The numbers, and what they are:**
- **Of the 161 titles, 108 (67.1%) are classified as AI-system evaluations.** The shares below are shares of those 108, not of 161:
  - text_qa 38.9% (42), mm_image 29.6% (32), agent 14.8% (16), code 6.5% (7), mm_video 4.6% (5), math 2.8% (3), audio 2.8% (3).
  - As shares of all 161: text_qa 26.1%, mm_image 19.9%, agent 9.9%, code 4.3%, mm_video 3.1%, math 1.9%, audio 1.9%, not an AI-system evaluation 32.9%.
- **Validation:**
  - The rules agree with the 80 labels on whether a title is an AI-system evaluation for 66 of 80 (82.5%).
  - On the category, they agree for 38 of the 40 titles both call an evaluation (95.0%), and on static QA against code or agent for 39 of 40 (97.5%).
  - The labels themselves put 43 of 80 as evaluations. Their static-QA share is 26 of 43 (60%, Clopper-Pearson 95% [44%, 75%]).
- **The 1,261-row sheet** (for context; no doc number uses it):
  - 849 AI-system evaluations: text_qa 41.0%, mm_image 23.2%, agent 20.3%, code 6.5%, mm_video 3.6%, math 3.3%, audio 2.1%.
  - a6 classes: multimodal 32.8%, other 32.0%, agentic 16.5%, qa_reason 12.1%, code 4.2%, math 2.4%.

**Doc sentences to correct:**
- **Draft §2.1.** "By a classification of their titles and descriptions, it is 39% text QA, 30% images, 15% agents, 6.5% code, 4.6% video, 2.8% math and 2.8% audio. That classification had 95% agreement with 80 hand labels (…; its script is in session scratch, `TODO(record)`)."
  - It is titles only.
  - The shares are of the 108 titles (67%) classified as AI-system evaluations.
  - The 95% is category agreement on 38 of 40 titles that both call evaluations; agreement on evaluation-or-not is 82.5% of 80.
  - The 80 labels come from the 1,261-row sheet (14 among the 161) and were made by an AI agent.
  - Script: `experiments/inventory_classes.py`.
- **Findings "What this could do on the hidden test".** "(scratch provenance: rethink2 methodology-critic `c_inventory_meta.json`, 95% category agreement with 80 hand labels)": the same corrections, and cite `experiments/inventory_classes.py`, `results/inventory_classes.json`.
- **Draft §7 and Appendix B.** The inventory classification is no longer scratch-only.

### 2.3 The step-2b audit (draft §3.4, §5.1, §5.3; findings "What actually shipped, after the audit")

**What it is.** The 2026-09-26 audit of `level_calibration.py`'s recommendation moved the shipped level from AGGR to MILD:
- AGGR = `hier G mu0 -3.0, sigma_mu 2.5, attr_scale 0.25`, `level_calibration.SHIP`;
- MILD = `mu0 -2.5, sigma_mu 2.5, attr_scale 0.5`, the shipped `LEVEL`;
- PRED below means the legacy Predictor.

The audit's record is in `scratch/step2b_results.json` (`audit`), with outputs in `scratch/step2b/audit/`.

**How it was made reproducible.** `experiments/level_audit.py` has three default stages, about 5 s together:
- `feedback`, `analyse`: the audit's own code, reading `results/level_calibration.json`.
- `mild`: the audit scored MILD with fresh runs on test-like 100–199, public 0–99 and no-shift 0–29. This stage reads MILD's rows from the `ship` arm of `data/subject_side_rows` instead: the same runs, drawn by `level_calibration.draw`.
  - It first checks the two sources agree where both hold MILD (test-like 0–99, public benchmark-first 0–39): 140 runs, 1,147 pairs, max difference 5.5e-6, the rows' 1e-5 rounding.

Against the scratch outputs, every feedback-matching number, group mean, public per-benchmark mean, nested pick, regret and pair-rate share is identical at the precision stored.

**Reproduced numbers** (all `results/level_audit.json`):

| claim (in the docs unless marked) | audit's number | reproduced | where |
|---|---|---|---|
| per-pair matched estimate on run 1, "about 0.178 (0.169 to 0.183)" | 0.1783 (K 40), 0.1789 (K 15); 0.171–0.183 by pool; 0.169–0.177 on (B0, B31) | the same: 0.1783, 0.1789; test-like pools 0.1792 and 0.1830, (B0, B31) 0.1686, 0.1707 and 0.1771; range **0.1686–0.1830** | `feedback` `K=40`, `K=15`, `robustness (K=25)` |
| … **for AGGR, not the shipped config** | — | MILD on the pool where MILD, AGGR and PRED are all stored (test-like 0–199, public 0–99 both, mix/whole and no-shift 0–99; 4,955 appearances): **0.1765 (K 15), 0.1789 (K 40)**, against AGGR 0.1770 and 0.1797 on the same pool. On the screen pool (test-like 0–99 + public benchmark-first 0–39): MILD 0.1812 / 0.1830, AGGR 0.1824 / 0.1839 | `mild` `feedback-matched estimate, MILD and AGGR on one pool`; `feedback` `all screened configs` |
| p6 and p8 "more likely high-rate pairs (about 0.78)", p9 ambiguous | neighbours p > 0.5: p6 70%, p8 80%, p9 55% | the same (K 40). "About 0.78" is the **upper root** of p(1-p) = B31 (p6 0.787, p8 0.774), not an estimate. The neighbours' mean p is 0.62 and 0.66 | `feedback` `K=40` `pairs` |
| re-reading the nine rates puts the logit mean near -1.0 | -1.0 | -1.02 (sd 1.52), against the lower roots' -1.64 (sd 1.49) | `feedback` `feedback level readings` |
| AGGR "loses about 0.015 per pair on a matharena-like benchmark" (findings); draft §3.4 says "on high-rate pairs" | public matharena +0.017 / +0.015 | AGGR − PRED on public matharena +0.0172 (benchmark-first, 192 appearances), +0.0150 (pair-uniform, 334) | `analyse` `r1b/r1p 0-99, AGGR - PRED by benchmark` |
| by pair rate (test-like 0–199) | -0.084, -0.065, -0.026, +0.014, +0.070 | the same. So on high-rate pairs it is +0.014 (0.5–0.7) to +0.070 (≥ 0.7), not 0.015: draft §3.4's wording mixes the two | `analyse` `test-like 0-199 … by group` |
| nested leave-one-parent-out does not carry the level to matharena | pick mu0 -3.5, +0.0073 on matharena (AGGR +0.0026, best -0.0130); grid regret 0.0375 on matharena, 0.001–0.007 elsewhere | the same | `analyse` `nested …` |
| the confirmation half repeats the selection half | 82.2% of pairs, 99.5% of clusters, all 35 pseudo-benchmarks, 324 / 340 distinct pairs | the same | `analyse` `confirmation half against the selection half` |
| MILD "gives up about 0.002 in the tuned regime" | +0.0022 ± 0.0011 confirmation, +0.0030 ± 0.0009 selection | MILD − AGGR +0.0022 ± 0.0005 / **0.0011** / 0.0009 (confirmation), +0.0030 ± 0.0005 / 0.0009 / 0.0007 (selection), +0.0026 (0–199) | `mild` |
| MILD "gains about 0.0024 on public runs" | -0.0023 ± 0.0005, -0.0024 ± 0.0004 | MILD − AGGR -0.0023 ± 0.0002 / **0.0005** / 0.0004 (benchmark-first), -0.0024 ± 0.0002 / 0.0004 / 0.0004 (pair-uniform) | `mild` |
| MILD "beats the Predictor there" | -0.0015 ± 0.0021, -0.0018 | MILD − PRED -0.0015 ± 0.0010 / 0.0021 / 0.0018 (benchmark-first), -0.0018 ± 0.0010 / 0.0017 / 0.0016 (pair-uniform): point estimates, within about one cluster SE (0.7 and 1.02 cluster SEs) | `mild` |
| (audit record only) no-shift check, MILD minus AGGR | -0.0005 ± 0.0005 (30 runs) | -0.0005 ± 0.0005 / 0.0005 (0–29); -0.0005 (0–99) | `mild` |
| (new; not scored by the audit, not in the docs) mix/whole | — | MILD − AGGR +0.0031 ± 0.0010 (cluster); MILD − PRED -0.0436 ± 0.0039 | `mild` |
| (audit record and review W2, not in the docs) MILD − PRED, confirmation half | -0.0396 ± 0.0037 | -0.0396 ± 0.0019 / 0.0037 / 0.0034 | `mild` |
| (audit record; the docs quote only hier's default, +0.006 to +0.009) single-subject swe_rebench | AGGR − PRED +0.012 / +0.015, hier default +0.008 | the same | `analyse` |

**What re-measures rather than reproduces.** `--stage extra` scores the audit's two regimes nearer the per-pair reading afresh: `tl lm-0.8` and `tl untilted`, seed 5, 40 runs each, for base, AGGR, mu0 -3.5, MILD and EB.
- It uses the library on disk, which has changed since the audit: the corrected multiple-choice floor (`f7e7d87`) and the floored-fit fix (`4d2cc4f`).
- So its numbers are new measurements, not the audit's (audit: AGGR − PRED -0.0279 at level_mean -0.8 and -0.0193 untilted; MILD -0.0283 and -0.0220).
- The draft does not quote these numbers. They back "has no worse measured worst case".
- Result: see 2.3.1.

**Not moved.**
- `stress_latency.py`, the dense-latency stress test: its numbers (0.46 s first call, 39 ms later calls at B31) are not quoted in the docs; findings' own latency section stands.
- `noshift_as.py`'s attr_scale 1.0 and 0.5 at mu0 -3.0: stored nowhere, not quoted.
- `repro.py`, `runstats.py`: checks and descriptive numbers, not quoted.

**Doc sentences to correct:**
- **Findings "What actually shipped, after the audit".** "(scratch provenance: step2b audit)": cite `experiments/level_audit.py`, `results/level_audit.json`.
- **Draft §3.4 item 1.** "the aggressive setting loses about 0.015 per pair on high-rate pairs": 0.015 is the loss on public matharena pairs. On high-rate test-like pairs AGGR loses +0.014 (rates 0.5–0.7) to +0.070 (≥ 0.7).
- **Draft §3.4 item 2 and §5.3.** "gives up about 0.002 … gains about 0.0024": give +0.0022 (confirmation) and +0.0030 (selection) and -0.0023 / -0.0024, with the SEs above.
- **Draft §5.1.** "the per-pair matched estimate for the recommended configuration on run 1 was about 0.178 (0.169 to 0.183)": that is AGGR's. The shipped MILD's is 0.1765 to 0.1789 on the common pool (AGGR's there is 0.1770 to 0.1797).

#### 2.3.1 The re-measured regimes (`--stage extra`)

- **Cost.** 400 tasks: 40 runs × 2 regimes × (base + 4 configs). They ran on one process in 1,414 s plus a first 13-task pass that was stopped and resumed; peak RSS 1.9 GB.
- **Stored.** The rows are packed under `raw_extra` in `results/level_audit.json`, and the summary is under `extra`.
- **Results.** Minus PRED, ± run / cluster / stratified SE; the audit's (older library) numbers in brackets:

| regime (realised pair logit, share p > 0.5; PRED ALC) | AGGR | mu0 -3.5 | MILD (shipped) | EB | MILD − AGGR (run / cluster SE) |
|---|---|---|---|---|---|
| `tl lm-0.8` (-0.77, 0.30; 0.2153) | -0.0283 ± 0.0033 / 0.0049 / 0.0044 [-0.0279 ± 0.0049] | -0.0273 | **-0.0286** ± 0.0026 / 0.0039 / 0.0035 [-0.0283] | -0.0286 | -0.0003 ± 0.0008 / 0.0011 [about -0.0004] |
| `tl untilted` (-0.59, 0.36; 0.1958) | -0.0196 ± 0.0033 / 0.0063 / 0.0055 [-0.0193 ± 0.0063] | -0.0177 | **-0.0222** ± 0.0025 / 0.0049 / 0.0043 [-0.0220] | -0.0213 | -0.0026 ± 0.0008 / 0.0014 [-0.0027] |

- **Also in the file:**
  - hier's defaults: -0.0095 and -0.0057;
  - smoothed Beta(2,2): -0.0183 and -0.0119;
  - the Predictor level fix (off -1.5, sc 1, va 3): -0.0275 and **-0.0232**, which beats MILD in the untilted regime.
- **Measured worst cases against PRED, as regime means (not per parent):**
  - for MILD, over test-like halves, public both, no-shift, mix/whole and these two regimes, the least favourable is -0.0015 (public benchmark-first);
  - for AGGR, over the same regimes plus level_mean -1.2 and -2.0, it is +0.0008 (public benchmark-first).
- So "no worse measured worst case" holds on every regime measured. The level_mean -2.0 regime, where AGGR should do best, has never been scored for MILD.
- **Script version.** The extra rows were scored under script digest `40ed9e5b…`. The committed file differs from it in two edits, neither touching the extra stage: the `analyse` stage's nested selections choose on unrounded means (07:10 UTC, digest `fdd56ac8…`), and the docstring gives the recorded run times (`8ef6d561…`). The library digest (`8b96ccb4…`) is unchanged. `experiments/script_revisions.py --stage replay` rebuilds `40ed9e5b…` from the committed file and the edit log (`results/script_revisions.json`). A verifier's spot re-score of 4 of the 400 extra tasks with the committed script and library matched bit for bit (scratch `report_p0/verify-repro/extra_spot.py`; not stored).

### 2.4 Other scratch-only statements found in `docs/findings.md` (not moved by this lane)

- **"Acceptance harness", Caveats.** "A scratch check on the real rows shows both sides … +0.0032 test-like … +0.0168 there … +0.00001". Marked indicative. No script.
- **"Acceptance harness", Caveats.** The u_j leave-out check (-0.02100 against -0.02036; -0.00230 against -0.00235; -0.00025 against -0.00004). Already marked "not reproducible from the repository".
- **"Floored fits" (under "The multiple-choice floor, corrected").** Scratch replay counts and timings:
  - "6 to 7% of prediction time";
  - "about 0.9 ms a call";
  - "0.912 against 0.905 ms on r1p and 0.854 against 0.857 on r1b".

  These are marked as scratch in the text. The section's main numbers come from `experiments/hier_floor_replay.py`.

---

## 3 Disclosure facts (P0.4)

### 3.1 Q7: the blind LLM difficulty ratings

**What the repository holds:**
- Code: `experiments/llm_rating/ratings_main.py` (180 ratings, 0–100, keyed by uid), `ratings_control.py` (60 ratings) and `analysis.py`.
- Data: `results/rate2_truth.json` (180 items: uid, bench, key, Rasch `zb`, n, text) and `results/ctrl_truth.json` (60 items: uid, key, zb, n, text).
- All six files entered in the repository's first commit, `165b4c0`, 2026-09-23 19:14:52 UTC, and have not changed since.
  - Author Nikita L. Polomoshnov.
  - Trailers `Co-Authored-By: Claude Opus 5` and `Claude-Session: https://claude.ai/code/session_01SfDQKV6PCMssu2UQX8KxXL`.

**What the repository says about the protocol.** Only `analysis.py`'s docstring: the ratings "were produced by reading the item texts with the true difficulty withheld"; the main sample is "truncated to 450-700 characters"; the control is "60 fresh multi_swebench items with full text".

Measured from the truth files:
- main texts are 63 to 700 characters (median 450);
- control texts are 274 to 4,000 characters (median 1,030), so the "full text" was capped at 4,000 characters.

**Not recorded anywhere in the repository, `results/` or scratch:**
- the rating model and its version;
- the prompt;
- the date and time of rating;
- sampling settings;
- whether all items were rated in one context;
- how the rater was kept from `zb`.

This session's scratch starts on 2026-09-24, after the ratings existed, and the other session scratchpads on this machine are empty.

**Indirect evidence only:**
- The commit that added the ratings is co-authored by Claude Opus 5 in a Claude Code web session.
- This session's 2026-09-26 notes (`scratch/rethink2/all.json`) call it "the blind frontier-LLM rating".
- The most likely rater is therefore Claude Opus 5 inside that coding session. **The team confirmed on 2026-09-28 that the rater was the Claude model of that session; the prompt and date remain undocumented.**

**Blindness:**
- **Benchmark identity was not hidden.** The uids carry the benchmark (`ma`, `mu`, `re`, `sw` prefixes; `re` is real_webagents), and the texts name their domain. "Blind" can only mean the difficulty was withheld.
- The truth files store `zb` next to each text, produced by the same code base. Nothing records that the rater's context excluded them.

**Could the rater have seen measurement-db or leaderboards?**
- **measurement-db:** if the rater was the coding-session model, that session had measurement-db in `data/` and computed `zb` itself. So exposure depended on context handling that is not recorded.
- **Leaderboards:** separately, any frontier model's training data can include the public benchmarks and their public leaderboards (MathArena's competitions and results, Multi-SWE-bench, SWE-rebench, REAL). This cannot be excluded.

**Reproducibility:**
- The correlations re-derive from the hard-coded ratings (`python experiments/llm_rating/analysis.py`).
- The ratings themselves cannot be re-derived.
- A fully specified substitute exists: the local Qwen3-4B-Instruct-2507 judge at revision `cdbee75f…`, run through `experiments/llm_features.py` and `experiments/llm4b_close.py`, with `results/llm4b_close.json`.

### 3.2 W9: the inventory scan

**Script.** `experiments/inventory_scan.py`, in the first commit `165b4c0` (2026-09-23), unchanged since. The copy in `scratch/mcq/adopt/experiments/` is identical.
- It reads `inventory.csv` and appends to `clone_scan.csv` in the working directory; `results/clone_scan.csv` has 161 rows.
- It needs the network.
- No wall time is recorded. It ran before 2026-09-23 19:14 UTC.

**What it read.** For each inventory row whose `code_url` is a GitHub repository:
1. `git clone --filter=blob:none --depth 1 --quiet https://github.com/<owner>/<repo> <tmpdir>`;
2. `git ls-files`;
3. regular expressions on the **file paths**: a results-like directory, a data-file extension, and a model name in the path;
4. it stored counts and up to three example paths;
5. it deleted the clone (8 threads, 180 s timeout per clone).

The script opened no file's contents. But a blobless clone still checks out the default branch, and git fetches the blobs that checkout needs. So each repository's current files were downloaded to a temporary directory and deleted with the clone. The raw-versus-graded judgement below came from a step that is not recorded, so for the five repositories it covers, an inspection of their files cannot be excluded.

**What it found** (findings "The offline bank"; `results/clone_scan.csv`):
- 132 of 161 repositories could be cloned;
- 25 publish a data file under a results-like directory;
- 15 have a model name in such a path;
- 5 cover at least nine models;
- findings says only capability publishes graded outcomes, so "one in 161" was usable.

`results/strong_repos.csv` (15 repositories: model counts, model lists, a sample path) has no producing script in the repository. How "graded outcomes" was judged for the five (phyblock, mmdocrag, atmossci_bench, capability, engdesign) is not recorded; it needs file names or contents, for example mmdocrag's `*_llm-judge.jsonl`. No Claude Code session log of that step (before 2026-09-23 19:14 UTC) was found on this machine. mmdocrag is a public benchmark; the other four may be hidden-test benchmarks. `TODO(team)`: confirm how the judgement was made.

**What it was used for:**
- It decided whether an "offline bank" of per-item results from inventory repositories was feasible. The bank was closed: 1 of 161 usable, and the rules restrict competition-specific training and curation to the public pool (draft §6.7).
- Nothing from the scan, `clone_scan.csv`, `strong_repos.csv` or the inventory enters `paiec/`, `submission/` or `tools/` (no reference by grep).
- No per-item data from any inventory repository was kept or used.
- The inventory's titles (not the scan) feed two descriptive analyses: the task-type shares (`experiments/inventory_classes.py`, section 2.2) and `experiments/itemcov_eval.py --stage inventory`, a keyword scan for difficulty-like cues. Neither sets a model, prior or hyperparameter.

**Context for the statement.**
- The inventory lists candidates from both pools, because the test pool is drawn from it. So the scan listed files of repositories that may be hidden-test benchmarks.
- The competition page says: "Participants must use benchmarks in the public training pool for competition-specific training and curation during development."

### 3.3 W9: every use of formative feedback and of the leaderboard

Two formative runs were scored. Their per-pair tables are in `scratch/formative/run1_feedback.txt` and `run2_feedback.txt` (pasted from Codabench). Another lane is recording them in `results/formative/` (untracked at the time of writing).

**Formative run 1** (2026-09-25; the legacy Predictor at `b68492c`; ALC 0.2113; 9 pairs on 7 benchmarks). It is stored in code as `paiec/testlike.py` `FEEDBACK`: per-pair budget Brier and evaluated-item counts, with benchmarks relabelled A–G and no subject or benchmark ids. Its uses:

1. **Tuning the test-like regime** (`experiments/testlike_check.py --phase tune`, `testlike.Regime` defaults):
   - `level_mean` -1.6 and `level_sd` 1.5 come from reading each pair's B31 through the lower root of p(1-p);
   - `date_shift` 1.25 years is the grid point (level_mean {-1.2, -1.6, -2.0} × date_shift {0, 0.75, 1, 1.25, 1.5}) whose legacy-Predictor B0 and B1 are closest to run 1's;
   - `repeat` 0.25 (the chance that a pair joins a pseudo-benchmark already in the run) is justified in the docstring by run 1's shape (9 pairs on 7 benchmarks). The tune grid did not include it, so it was set by hand;
   - run 1's evaluated counts (44–76) motivated the `split_after_cut` / `min_kept` 88 option, which is off by default and only used in a sensitivity.
2. **The first verdict.** `experiments/hier_eval.py` `FEEDBACK_MIX = {"1": 5, "2": 4}` weights hier's expected public gain by run 1's composition: 5 targets alone on their benchmark, 4 sharing it. That fed "Verdict: keep the Predictor", later reversed. Only the composition is used, no label.
3. **Choosing the level prior** (`experiments/level_calibration.py`): mu0, sigma_mu and attr_scale were selected on runs of the regime tuned in item 1, under the public guard.
   - Its summary adds each candidate's paired difference to run 1's budgets (`feedback_plus_diff`, the 0.167 estimate).
   - The "near" column uses the 50 runs whose Predictor profile is closest to run 1's.
4. **The audit that changed what shipped** (step 2b, now `experiments/level_audit.py`).
   - It read run 1 **per pair**: neighbour matching on each pair's budget profile, and the note that p6 and p8 share one anonymous benchmark (F).
   - It moved the shipped level from AGGR to MILD. This is the only per-pair reading behind a shipped choice.
5. **Diagnostics only:**
   - `testlike_check.py`'s check phase (run 1's z-scores in the replica's single-run spread; level identification distances);
   - `ORGANISERS` 0.1801 as a reference point.

**The live leaderboard** (read 2026-09-24 from the public Codabench API, `/api/phases/29785/get_leaderboard/`: organisers' entry 0.1801, best entry 0.1172).
- If the organisers' entry is the empirical mean, its ALC inverts to E[p(1-p)] ≈ 0.128. Draft §3.4 cites this as the second reading behind lowering the level.
- It is also constant `LEADERBOARD` in `experiments/official_baselines.py` and `ORGANISERS` in `testlike_check.py`, used for placement only.

**Formative run 2** (2026-09-26; hier with the shipped LEVEL; archive `2c64eaad…` = `ee5085a`; ALC 0.192623; 8 pairs):
1. Its B0 and B1 motivated the subject-side study (`experiments/subject_side.py` docstring). Nothing passed that study's gates, so nothing changed.
2. The draft places it against the replica's regimes (§5.1, §5.4). Another lane's `experiments/ship_confirm.py` records its budgets. Both are descriptive.
3. **No global hyperparameter changed after run 2.**
   - `LEVEL` is unchanged, and `submission/prior.json` is byte-identical to the run-2 archive's.
   - The two library changes after it came from public-data findings: the multiple-choice floor (TikZ labels and the Kangaroo option lists on public matharena items) and the floored-fit fix (a non-converged fit on public pair-uniform run 108).
   - No record ties either to run 2.

**The conduct rule as applied.**
- Feedback set only the regime's global knobs and, through the regime, three global level hyperparameters.
- Nothing is keyed on an anonymous benchmark or subject id. `FEEDBACK` stores relabelled letters, and `prior.json` holds no item- or benchmark-keyed table (audit leakage check). The organisers' tables themselves are stored verbatim, ids included, in `results/formative/`, and `experiments/formative_feedback.py` uses the benchmark ids only to group pairs (its bootstrap resamples them); nothing in `paiec/`, `submission/` or `tools/` reads them.
- No prediction was shaped to probe hidden labels.
- The pooled per-pair reading of runs 1 and 2 (draft §5.1) was made on 2026-09-28 under a decision rule hash-locked before it was computed, after both tables had been seen, and changed nothing. Its read stage ran four times under three script versions, which `experiments/script_revisions.py` rebuilds and re-runs: the decision is the same under each (`results/script_revisions.json`).

**Doc sentences the accounting contradicts:**
- **Draft §8.6.** "It was used only as described in §3.4: aggregate budget statistics that set three global level hyperparameters."
  - Run 1 was read per pair twice: each pair's B31 root set the regime's level target, and the audit matched each pair to replica pairs.
  - Its composition (5 alone, 4 shared) weighted the first verdict.
  - Every use still set only global quantities, but they were not aggregate statistics alone.
- **Draft §3.4 ("The rule we kept") and findings "Against the real feedback".** "no feedback number was fitted beyond the regime's defaults".
  - That holds for the level grid's selection.
  - The audit's per-pair reading then chose between two guarded configurations (AGGR, MILD), partly on how p6, p8 and p9 read. That is a discrete choice informed by run 1 beyond the regime's defaults. It is a global hyperparameter, so it stays within the stated rule, but the sentence should say so.
- **Count for Q9.** Two scored formative submissions so far. Run 1 informed items 1 to 4 above; run 2 informed no choice. The pooled per-pair reading of runs 1 and 2 (§5.1) proposed no change.
