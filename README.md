# PAIEC: a hierarchical response model for a benchmark-level cold start

Code, results and report source for an entry to the
[Predictive AI Evaluation Competition (PAIEC)](https://aimslab.stanford.edu/competition)
at NeurIPS 2026, by Nikita L. Polomoshnov (Moscow State University, nikitapol@fbb.msu.ru).
The technical report is *Calibrating a hierarchical response model for a
benchmark-level cold start*; its source is [`docs/report/draft.md`](docs/report/draft.md).

**The task.** For an AI system (the *subject*: eight string fields such as
normalized_name, provider, release_date, harness) and a benchmark *item*
(item_content, item_features, interactors and an anonymous benchmark_id), return
the probability that the subject answers the item correctly, given `labeled`,
the 0 to 31 outcomes revealed so far. Each (subject, benchmark) pair is scored by
Brier on its evaluation items at budgets 0, 1, 3, 7, 15 and 31, combined as

    ALC = 0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31

and averaged equally over pairs. The hidden test's benchmarks never appear in the
public training data, so every test benchmark is a cold start: its level is not
in the training data. Evaluation workers are recreated at every budget, so
`predict` must be a pure function of `(input, labeled)`.

**What is here.**

- A replica of the organisers' streaming evaluator, checked decision for
  decision against their client (`paiec/official.py`; protocol and open points
  in [`docs/protocol.md`](docs/protocol.md)).
- The submitted predictor, *hier-ship*: a hierarchical item-response model
  refitted from `labeled` at every budget (`paiec/hier.py`), with an offline
  attribute prior fitted on the public data (`paiec/prior.py`) and a prior on a
  new benchmark's level (`LEVEL` in `submission/model.py`: mu0 -2.5, sigma_mu 2.5,
  attr_scale 0.5) calibrated on test-like runs built from public data
  (`paiec/testlike.py`).
- One script per reported result under `experiments/`, negative results
  included; most write a `results/<name>.json` that records the commands and
  code digests behind it.

## Headline results

From the report's §1.5 table, which has every row and its sources. Δ is
hier-ship minus the comparator in ALC (negative favours hier-ship), with the
pair-cluster bootstrap SE in parentheses. *LegacyP* is our first submission's
predictor (`paiec/predict.py`). Test-like runs are drawn from public data to
resemble the first formative feedback; public runs redraw public pairs at
formative size, benchmark-first (R1-bf) or pair-uniform (R1-pu).

| what | value | report | in `results/` |
|---|---|---|---|
| formative runs on the platform, single draws: run 1 LegacyP, run 2 hier-ship, run 3 hier-ship's rebuilt archive (recomputed from the platform's table) | ALC 0.2113, 0.1926, 0.1817 | §5.1 | `formative_feedback.json`, `formative_run3.json` |
| Δ LegacyP, held-out test-like runs 200-299 | -0.0419 (0.0037) | §5.4 | `ship_confirm.json` |
| Δ LegacyP, test-like runs without the synthetic date shift | -0.0144 (0.0024) | §5.4 | `ship_confirm.json` |
| Δ LegacyP, test-like runs at two later readings of the feedback's level | -0.0248 (0.0037), -0.0313 (0.0049) | §5.4 | `regime_sensitivity.json` |
| Δ LegacyP, public runs R1-bf, R1-pu | -0.0017 (0.0021), -0.0018 (0.0017) | §5.4 | `ship_confirm.json` |
| Δ a smoothed label mean with a calibrated prior (Smooth-cal): tuned test-like, R1-bf, R1-pu | +0.0015 (0.0019), -0.0108 (0.0014), -0.0100 (0.0018) | §5.4, §5.5 | `regime_sensitivity.json`, `baselines_p1.json` |
| Δ the organisers' empirical-mean baseline, same runs | -0.0339 (0.0030), -0.0464 (0.0027), -0.0449 (0.0030) | §5.5 | `baselines_p1.json` |
| Δ LegacyP on the single-subject public benchmark (swe_rebench) | +0.0101 ± 0.0012 (SE over 130 appearances) | §5.6 | `gate_and_ci.json` (`single`) |

In short: the gain over our first submission comes mainly from the level prior and
is regime-conditional, within noise on public runs. A calibrated smoothed mean
matches hier-ship in the regime it was tuned on and trails it by 0.004 to 0.011
elsewhere. Except a blind rating that tracks difficulty on mathematics only,
every item-side signal we completed (text-based difficulty maps, language-model
judges, a 14B model's reasoning entropy) fails to transfer to a held-out
benchmark or stays below an acceptance gate of 0.002 ALC (report §6). A single
formative run's ALC has an sd of 0.02 to 0.04. Limitations are in report §7.

[`docs/findings.md`](docs/findings.md) is the lab notebook behind the report:
every number with the script that produced it. Its section "Under the official
protocol" and everything after it supersede the earlier numbers, which came
from a legacy replica (`paiec/evaluator.py`) with the wrong `labeled` semantics.

## Repository map

| path | what |
|---|---|
| `paiec/` | the library: `official.py` (evaluator replica), `hier.py` and `prior.py` (the submitted model and its offline prior), `predict.py` (LegacyP, and helpers hier reuses), `testlike.py` (test-like runs), `data.py` and `fetch.py` (data), `evaluator.py` (legacy replica) and research modules |
| `submission/` | `model.py`, the competition entry point `predict(input, labeled)`; `labeling.py`, not shipped by default |
| `tools/` | `build_submission.py` (build and check the archive), `report_figures.py` (the report's figures), `build_report_pdf.py` and `report_pdf/` (the report's PDF), `export_rows.py` (pack per-row files) |
| `experiments/` | one script per result; `_*_legacy.py` reproduce older numbers |
| `results/` | the committed results files; `results/formative/` holds the organisers' formative-feedback tables verbatim |
| `tests/` | protocol invariants and unit tests, on synthetic data |
| `docs/` | `protocol.md`, `findings.md`, `report/` (the report source, figures, provenance and requirements), `release.md` (release checklist) |
| `kaggle/strong_probe/` | the Kaggle kit for the Qwen3-14B probe (README in Russian) |
| `CLAUDE.md` | working notes for the coding assistant the repository was developed with; also a compact architecture overview |

## Install

Python 3.10 or later. Development used Python 3.10.8 with numpy 1.26.4
(OpenBLAS) on an Apple M1 Pro; report App G.1 lists every version.

    git clone https://github.com/doctawho42/PAIEC.git paiec
    cd paiec
    pip install -e ".[dev]"            # ".[dev,report]" to redraw the figures (matplotlib 3.9.2)

The submission itself needs numpy alone (`requirements.txt`). The offline
language-model features (`experiments/llm_features.py`) need torch and
transformers, and the meta-learned heads (`experiments/heads_eval.py`) need
torch; neither is in `pyproject.toml` (report App G.1).

## Data

The build and the experiments need measurement-db
([Hugging Face](https://huggingface.co/datasets/aims-foundations/measurement-db),
by N. Truong, S. T. Truong and S. Koyejo, AIMS Lab, Stanford; cite it as its
dataset card asks); the tests and the figures do not. It is gated and is not
included here. Accept its terms on
the dataset page with your own account, put your token in the environment (or
run `huggingface-cli login`), then fetch the core tables at the pinned revision:

    export HF_TOKEN=...                # your own Hugging Face token
    python -m paiec.fetch              # into data/, at revision bc8204d811823da849c6686bf124d4ca9f82e4de

Every recorded result used that revision. `--revision` or `PAIEC_DATA_REVISION`
fetches another one. `PAIEC_DATA` points the code at another data directory; the
fetch does not read it, so fetch into that directory with
`python -m paiec.fetch --out "$PAIEC_DATA"`.

## Tests

    python -m pytest -q

The tests need neither the data nor the organisers' code: they run on
synthetic data, and the few that need either are skipped when it is absent.
The suite takes 4 to 17 minutes on the development machine, depending on its
load.

## Rebuild and validate the submission archive

The build runs the organisers' validator, which lives in their baseline
repository. That repository has no license, so it is not included: clone it
yourself at the commit the replica mirrors.

    git clone https://github.com/aims-foundations/paiec_baseline third_party/paiec_baseline
    git -C third_party/paiec_baseline checkout 82d330ddcdb16016a3ae9e048db7e588ba2c6a39
    python tools/build_submission.py    # fits submission/prior.json, writes dist/paiec.zip, runs every check
    shasum -a 256 dist/paiec.zip        # or sha256sum

`dist/paiec.zip` exists only if every check passed, including the validator and
a bit-for-bit match with the in-repository predictor (report §3.5, App B.2).
The validator also runs on its own:
`python third_party/paiec_baseline/check_submission_zip.py dist/paiec.zip`.

The archive behind formative run 3 has sha256
`4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090`; its code
(the shipped `paiec` modules, `submission/` and the build script) is unchanged
since commit `4d2cc4f`. Expect that hash only with the development machine's
BLAS and its default thread settings. The fit of `prior.json` is not
bit-stable across BLAS threading (with OpenBLAS, OMP and vecLib held to one
thread it differs in its last bits, at most 2.1e-11 relative), and the hash
moves with it. A byte-identical rebuild was shown on that machine only, not on
other hardware (report App G.3). There a rebuild takes about 1.5 minutes and
1 GB of memory; one from a `git archive` export of `3e6b770` on 2026-10-03
reproduced `4a882cc7…` byte for byte.

## Rerun the experiments

The report's App G.4 lists the command, wall time and output of every result;
these are the main ones. Wall times are as recorded on the shared development
machine.

| result | command | wall time | output |
|---|---|---|---|
| official baselines (§2.4, §5.2) | `python experiments/official_baselines.py --runs 600 --jobs 6` | 12 min, 6 processes | `results/official_baselines.json` |
| hier-fit vs LegacyP, ablations (App E.1, B.3) | `python experiments/hier_eval.py --jobs 8 --skip 'r2\|researchcodebench\|pair\|hier t3 level'`, then `--summarise` | 2 h 16 min, 8 processes | `results/hier_eval.json` |
| level calibration (§5.3, §5.4) | `python experiments/level_calibration.py --stage S --resume --jobs 6` for S = grid, grid2, r1base, eb, t3, prob, eb, screen, confirm, final; then `--summarise` | 3 h 11 min, 6 processes | `results/level_calibration.json` |
| subject side (App F.7; its rows feed the next row) | `python experiments/subject_side.py --stage verify`, `--stage diag`, `--stage run --jobs 2` (plus `--configs` passes), `--summarise` | about 4 h 25 min, 2 processes | `results/subject_side.json`, rows in `data/subject_side_rows` |
| hier-ship confirmed (§1.5, §5.4) | `python experiments/ship_confirm.py` | about 5 s | `results/ship_confirm.json` |
| regime sensitivity (§5.4) | `python experiments/regime_sensitivity.py` with `lock`, `smcal`, `regimes --grid`, `reproduce`, then `score --shard 0/2` and `score --shard 1/2` side by side, then `summarise`; `review` after scoring (commands in the script's header) | rows over 2 h 31 min, 2 processes | `results/regime_sensitivity.json` |
| baselines: plain 1PL, empirical mean (§5.5) | `python experiments/baselines_p1.py ble`, then `score` (repeat while it exits with 75), then `summarise` | final pass 21 min; reads the regime-sensitivity rows | `results/baselines_p1.json` |
| pooling decomposition (§2.4) | `python experiments/pooling_decomposition.py score`, `recheck`, `summarise` | 84 min, 1 process | `results/pooling_decomposition.json` |
| gate, intervals, single-subject benchmark (§5.6, §6) | `python experiments/gate_and_ci.py --stage S` for S = gate, evidence, scale, ci, single | under 4 min a stage | `results/gate_and_ci.json` |
| formative run 3 (§5.1) | `python experiments/formative_run3.py` | 0.3 s; needs the `4a882cc7…` archive in `dist/` | `results/formative_run3.json` |

Per-row files (`data/*_rows`, gitignored) are written by these scripts and read
by the later ones; App G.4 says which script needs which. Some numbers were
computed with earlier library versions: App H.2 maps each number to the code
that produced it, and the gap between the libraries is measured there.

## Figures

    python tools/report_figures.py           # every figure into docs/report/fig/
    python tools/report_figures.py --check   # rebuild in a temporary directory and compare

About 5 seconds. The figures read the committed `results/*.json` only, so they
need no data; their bytes are tied to matplotlib 3.9.2 (the `report` extra).

## The Kaggle kit

[`kaggle/strong_probe/`](kaggle/strong_probe/README.md) ran Qwen3-14B-AWQ on
Kaggle's two T4 GPUs for the item-side features of report App F.9. Its README
is in Russian. The notebook needs your own Hugging Face token as a Kaggle
Secret and should stay private. Its outputs are not in this repository;
`experiments/strong_llm_eval.py` reads them.

## The report

The source is [`docs/report/draft.md`](docs/report/draft.md), with figures in
`docs/report/fig/` and their captions in `docs/report/figures.md`. Build the
PDF with

    python tools/build_report_pdf.py            # dist/report/paiec_report.pdf
    python tools/build_report_pdf.py --strict   # exit 2 on layout problems, placeholders or TODO markers

It needs pandoc 3.1 or later and XeLaTeX (TeX Live). The build checks with
kpsewhich that the packages `tools/report_pdf/template.tex` loads are installed
and names any that are missing. pdftotext and pdftoppm (poppler) are optional:
the first for the check that every number and word of the draft is in the PDF,
the second for `--render`. The PDF was built and checked on macOS (pandoc 3.6.2,
TeX Live 2026) with Times New Roman, Menlo and the committed 200-dpi PNG figures
(vector figures need rsvg-convert or cairosvg). Elsewhere the fonts fall back to
STIXGeneral or TeX Gyre Termes (which lacks Cyrillic), and page breaks may
differ. `--strict` fails until the title block's placeholders and the draft's
`TODO(...)` markers are gone. The competition's requirements for the report are
recorded in `docs/report/report_requirements.md`.

## Disclosures

The report's section "Code, data and conduct" and its App G.6 ("Conduct and
disclosure") are the full statement; in short:

- **Training data:** measurement-db only, the public training pool, at the
  revision above.
- **Formative feedback:** it set global quantities only. Run 1 set the
  test-like regime's defaults, and an audit that read it per pair chose
  between two configurations of the level prior. Run 2 motivated a
  subject-side study and a pooled reading of runs 1 and 2; neither changed the
  shipped model. Run 3 was a regression and latency check. No input, prior or
  selection is keyed on an anonymous id.
- **Pretrained models:** Qwen3-Embedding-0.6B, Qwen3-4B-Instruct-2507 and
  Qwen3-14B-AWQ, for research features only. The submission uses no pretrained
  model.
- **The inventory scan** (`experiments/inventory_scan.py`) cloned the
  repositories of the organisers' benchmark inventory, which may include
  hidden-test benchmarks, matched their file paths and deleted the clones;
  App G.6 records what is not confirmed (whether some files were opened).
  Nothing from it enters `paiec/`, `submission/` or `tools/`.
- **Language models in the work:** the blind difficulty ratings of report §6.3
  were made by the Claude model of the coding session. The code, the
  experiments, the docs and the report were developed with Claude Code
  (Anthropic); the commits' `Co-Authored-By` trailers record it.

## License and third-party material

The code and documentation written for this repository are released under the
MIT License ([`LICENSE`](LICENSE)), copyright 2026 Nikita L. Polomoshnov. The
license does not cover:

- **measurement-db.** The dataset is not included. It is gated and under its
  own terms: per its dataset card, the curation is licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and each
  benchmark's content keeps its upstream licence. Accept the terms yourself
  before fetching it.
  <!-- TODO(team): docs/release.md section 3.1 decides whether the two files below are published; if they are removed, drop the rest of this bullet. -->
  Two committed results files quote it: the blind-rating samples
  `results/rate2_truth.json` (180 items, 45 each from matharena,
  multi_swebench, real_webagents and swe_rebench) and `results/ctrl_truth.json`
  (60 multi_swebench items) hold the text of 239 items from measurement-db,
  some in full and the rest cut to their first 700 or 4,000 characters. These
  excerpts are not covered by the MIT License. The upstream licences recorded
  in the dataset's `benchmarks` tables are CC-BY-NC-SA-4.0 AND CC-BY-SA-4.0
  (matharena), Apache-2.0 (multi_swebench) and CC-BY-4.0 (swe_rebench); for
  real_webagents the dataset records the licence as unknown, so no licence is
  known to cover its 45 excerpts.
- **The organisers' baseline repository** (`third_party/paiec_baseline`). It
  has no license and is not included; clone it yourself as shown above.
  `paiec/official.py` and `paiec/baselines.py` re-implement its streaming client
  and its empirical-mean baseline so as to match them decision for decision,
  and for that repeat a few of its lines: the policy adapter's call signature,
  the sha256 acquisition rule and the empirical mean's error messages. Those
  lines remain the organisers'.
- **The organisers' formative-feedback tables** in `results/formative/`,
  stored verbatim as a record.
- **Rows of the organisers' benchmark inventory.** `results/inventory.csv`
  holds 161 of them (titles, authors, code and dataset links, venue), and
  `results/inventory_hand_labels.csv` and `results/inventory_classes.csv`
  repeat titles from the organisers' sheet. They were inputs to the analyses
  of report §2.2 and App A.1, F.6 and F.8.

## Citation

[`CITATION.cff`](CITATION.cff) gives the citation for the report and this code.
