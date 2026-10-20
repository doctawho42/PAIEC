# PAIEC: a hierarchical response model for a benchmark-level cold start

Code, results and report source for an entry to the
[Predictive AI Evaluation Competition (PAIEC)](https://aims.stanford.edu/competition)
at NeurIPS 2026, by Nikita L. Polomoshnov (Moscow State University, nikitapol@fbb.msu.ru).
The technical report, version 2, is *Reading hidden runs at a benchmark-level
cold start: a pair-rate oracle, test-like regimes and a calibrated hierarchical
predictor*; its source is [`docs/report/draft.md`](docs/report/draft.md), and
[`docs/report/revision_history.md`](docs/report/revision_history.md) records its
revisions; the draft at tag `v1.0.1` is report v1's last revision. Since `v1.2.0`
the report is set in the NeurIPS 2026 template, with at most 8 pages of main
text ("The report" below).

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
- A reading of the hidden formative runs through a pair-rate oracle. For one
  prediction per pair, Brier equals ECE² + p(1 - p), so the organisers'
  per-pair Brier and ECE tables give V, the score of an oracle that knows each
  pair's success rate (estimated at budget 0 where predictions spread within a
  pair). The reading splits each run's score into V, a rate-learning cost by
  budget and an item-level remainder (`experiments/hidden_runs.py`).
- Test-like runs built from public data (`paiec/testlike.py`), the regimes
  they define, checked against the hidden runs, and an acceptance harness for
  item-side signals (`experiments/harness.py`).
- The submitted predictor, *hier-ship*: a hierarchical item-response model
  refitted from `labeled` at every budget (`paiec/hier.py`), with an offline
  attribute prior fitted on the public data (`paiec/prior.py`) and a prior on a
  new benchmark's level (`LEVEL` in `submission/model.py`: mu0 -2.5, sigma_mu 2.5,
  attr_scale 0.5) calibrated on the test-like runs. Its archive, archive-3, is
  the final submission.
- The pool evaluation: hier's priors refitted on the expanded public pool and
  scored against archive-3 under a rule committed and pushed before any scoring
  (`paiec/pool.py`, `experiments/pool_eval.py`). No candidate was carried.
- One script per reported result under `experiments/`, negative results
  included; most write a `results/<name>.json` that records the commands and
  code digests behind it.

## Headline results

From the report's main text (§3 to §8), but for two numbers that only its
appendices state: C2's +0.0019 (Table C.6a) and the 51 appearances (App E.6).
The results keys behind each number are in
[`docs/report/sources_main.md`](docs/report/sources_main.md), Table C.6a's also
at the table. *LegacyP* is
our first submission's predictor (`paiec/predict.py`). *V* is the pair-rate
oracle's score, exact for run 1 and estimated at budget 0 for runs 2 and 3.
Test-like (TL) runs are drawn from public data to resemble the first formative
feedback; the regime-sensitivity (RS) study redraws them under five level
targets (tuned, audit, mixture, reading, flat) beside public runs at formative
size, benchmark-first (R1-bf) or pair-uniform (R1-pu). A number in parentheses
is an SE: the pair SE on the hidden runs, the pair-cluster bootstrap SE on the
replica's runs.

| what | value | report | in `results/` |
|---|---|---|---|
| formative runs on the platform, single draws: run 1 LegacyP, run 2 hier-ship, run 3 archive-3 (hier-ship, the final submission; its ALC recomputed from the platform's table) | ALC 0.2113, 0.1926, 0.1817; V 0.1536, 0.1797, 0.1608 | §6.1, Table 1 | `hidden_runs.json`, `formative_feedback.json`, `formative_run3.json` |
| ALC minus V, the cost of not knowing pair rates: run 1; runs 2, 3; runs 2 and 3 pooled | 0.0577 (0.0190); 0.0130 (0.0046), 0.0208 (0.0026); 0.0171 (0.0027) | §6.1, Table 1 | `hidden_runs.json` |
| the same cost for the ideal pair-rate learner on runs 2 and 3's pairs: at the lower roots; weighted by the prior | 0.0161; 0.0250 to 0.0322 | §6.1 | `hidden_runs.json` |
| estimated item-level share of what remains, runs 2 and 3 | 78% to 89% | §6.1 | `hidden_runs.json` |
| LegacyP minus hier-ship: hidden runs, adjusted for V; held-out TL runs 200 to 299; TL runs without the date shift | +0.0405 (0.019); +0.0419 (0.0037); +0.0144 (0.0024) | §6.3 | `hidden_runs.json`, `ship_confirm.json` |
| LegacyP minus hier-ship on the RS runs: tuned to flat; R1-bf; R1-pu | +0.0434 to +0.0200; +0.0002 (0.0020); +0.0039 (0.0019) | §6.3, Table 2 | `regime_sensitivity.json` |
| Smooth-cal (a smoothed label mean at a calibrated prior) minus hier-ship: tuned; the other four test-like regimes; R1-bf; R1-pu | -0.0015 (0.0019); +0.0043 to +0.0103; +0.0108 (0.0014); +0.0100 (0.0018) | §6.2, Table 2 | `regime_sensitivity.json` |
| the organisers' empirical-mean baseline minus hier-ship, all seven RS regimes | +0.0252 to +0.0464 | §6.3, Table 2 | `baselines_p1.json`, `e1_baselines_rs.json` |
| the change in hier-ship's Brier from B7 to B31 on the hidden runs; 17 replica pairs with the same rate mix on TL, R1-bf and R1-pu | -0.0012 (0.0028); -0.0145, -0.0215, -0.0229 | §6.2 | `hidden_runs.json` |
| V of runs of 9 pairs bootstrapped from the 26 hidden appearances, stratified by benchmark; the chance that V is at most 0.117, the best leaderboard score | 0.160 (sd 0.017); 0.0066 | §3.3 | `hidden_runs.json` |
| the best within-pair correlation of an item covariate that transfers between benchmarks; what the acceptance gate needs | 0.15; 0.25 | §4.3, §6.4 | `gate_and_ci.json`, `harness_thresholds.json` |
| hier-ship's loss to LegacyP on swe_rebench, the single-subject public benchmark | 0.0094 ± 0.0016 (SE over 51 appearances) | §8, App E.6 | `gate_and_ci.json` (`single`) |
| the pool refit: candidates C1, C1m and C2 minus archive-3, mean of the two gain regimes (at most -0.001 to carry) | -0.0007, -0.0017, +0.0019; none eligible, none carried | §7, Table C.6a | `pool_eval.json` |

In short: on the hidden runs hier-ship pays for not knowing pair rates about
what an ideal learner with a population prior pays, and an estimated 78% to 89%
of what remains is item level. Adjusted for V, the hidden runs agree with the
held-out test-like gain over LegacyP, a check partly in sample on LegacyP's side
(report §3.4); on public runs the gain is within noise on R1-bf and about two
cluster SEs on R1-pu. hier's structure pays only once its level is calibrated:
Smooth-cal ties hier-ship in the tuned regime and trails it elsewhere. Test-like
runs overstate late-budget gains, since after seven labels the hidden curves are
flat. No item-side signal, post-hoc map or acquisition policy we completed
passes (report §6.4), and a refit of the priors on the expanded public pool was
not carried, so archive-3 stays the final submission (report §7). On replica
runs one run's ALC for hier-ship has an sd of 0.023 to 0.030, and its ALC minus
V one of 0.009 to 0.011 (report §1). Limitations are in report §8.

[`docs/findings.md`](docs/findings.md) is the lab notebook behind the report:
every number with the script that produced it. Its section "Under the official
protocol" and everything after it supersede the earlier numbers, which came
from a legacy replica (`paiec/evaluator.py`) with the wrong `labeled` semantics.

## Repository map

| path | what |
|---|---|
| `paiec/` | the library: `official.py` (evaluator replica), `hier.py` and `prior.py` (the submitted model and its offline prior), `predict.py` (LegacyP, and helpers hier reuses), `testlike.py` (test-like runs), `pool.py` (the pool evaluation), `data.py` and `fetch.py` (data), `evaluator.py` (legacy replica) and research modules |
| `submission/` | `model.py`, the competition entry point `predict(input, labeled)`; `labeling.py`, not shipped by default |
| `tools/` | `build_submission.py` (build and check the archive; `--pool` for a pool-refitted archive, never built), `fetch_pool.py` (the expanded public pool), `report_figures.py` (the report's figures), `build_report_pdf.py` and `report_pdf/` (the report's PDF), `export_rows.py` (pack per-row files) |
| `experiments/` | one script per result; `_*_legacy.py` reproduce older numbers |
| `results/` | the committed results files; `results/formative/` holds the organisers' formative-feedback tables verbatim |
| `tests/` | protocol invariants and unit tests, on synthetic data |
| `docs/` | `protocol.md`, `findings.md`, `plans/` (the plans and their amendments), `report/` (the report source, figures, source keys, provenance, requirements and revision history), `release.md` (release checklist) |
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

Every recorded result used that revision, except the pool evaluation of report
§7, which used revision 54074453 of the expanded public release, fetched with
`tools/fetch_pool.py` into `data/pool/` (report App G.2). Its refitted priors
were not carried, so the submission depends on `bc8204d8…` alone.
`--revision` or `PAIEC_DATA_REVISION` fetches another revision. `PAIEC_DATA`
points the code at another data directory; the fetch does not read it, so
fetch into that directory with `python -m paiec.fetch --out "$PAIEC_DATA"`.

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
a bit-for-bit match with the in-repository predictor (report §5.4, App B.2).
The validator also runs on its own:
`python third_party/paiec_baseline/check_submission_zip.py dist/paiec.zip`.

The final submission is archive-3, the archive behind formative run 3, sha256
`4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090`. Its
members other than `prior.json`, which the build fits, are the eight shipped
`paiec` modules (as `paiec_rt/`), `submission/model.py` and `requirements.txt`;
they, and `submission/` as a whole, are unchanged since commit `4d2cc4f`, which
archive-3 was built from. `tools/build_submission.py` has changed since then
only by the lines that `831ef7d` added for its `--pool` path (none changed or
removed). That path builds a pool-refitted archive only after the pool
evaluation carries a candidate, and none was carried (report §7). The default
path, the command above, rebuilds archive-3 (`docs/release.md`, section 10).
Expect that hash only with the development machine's BLAS and its default
thread settings. The fit of `prior.json` is not
bit-stable across BLAS threading (with OpenBLAS, OMP and vecLib held to one
thread it differs in its last bits, at most 2.1e-11 relative), and the hash
moves with it. A byte-identical rebuild was shown on that machine only, not on
other hardware (report App G.3). There a rebuild takes under 2 minutes and
about 1 GB of memory; one from a `git archive` export of `3e6b770` on 2026-10-03
reproduced `4a882cc7…` byte for byte.

The history of this repository was rewritten once before publication, to
replace the e-mail address in its commits and to remove quoted measurement-db
item text and an access token (`docs/release.md`, section 9). Every commit id
changed. Seven-character commit ids written before the rewrite, here, in the
report, the scripts and the results files, are kept as they were, and each
resolves through a lightweight tag of the same name. Seven results files also
record the full 40-character old id of the commit they ran at (their `head`
fields). Those resolve by their first seven characters, and
`docs/commit-map.txt` lists every old full id with its new one.

## Rerun the experiments

The report's App G.4 lists the command, wall time and output of every result;
these are the main ones. Wall times are as recorded on the shared development
machine.

| result | command | wall time | output |
|---|---|---|---|
| official baselines (App E.1) | `python experiments/official_baselines.py --runs 600 --jobs 6` | 12 min, 6 processes | `results/official_baselines.json` |
| hier-fit vs LegacyP, ablations (App E.1, B.3) | `python experiments/hier_eval.py --jobs 8 --skip 'r2\|researchcodebench\|pair\|hier t3 level'`, then `--summarise` | 2 h 16 min, 8 processes | `results/hier_eval.json` |
| level calibration (App C.5) | `python experiments/level_calibration.py --stage S --resume --jobs 6` for S = grid, grid2, r1base, eb, t3, prob, eb, screen, confirm, final; then `--summarise` | 3 h 11 min, 6 processes | `results/level_calibration.json` |
| subject side (App F.7; its rows feed the next row) | `python experiments/subject_side.py --stage verify`, `--stage diag`, `--stage run --jobs 2` (plus `--configs` passes), `--summarise` | about 4 h 25 min, 2 processes | `results/subject_side.json`, rows in `data/subject_side_rows` |
| hier-ship confirmed (§6.3, App E.2) | `python experiments/ship_confirm.py` | about 5 s | `results/ship_confirm.json` |
| regime sensitivity (§6.2, §6.3, Table 2, App E.2) | `python experiments/regime_sensitivity.py` with `lock`, `smcal`, `regimes --grid`, `reproduce`, then `score --shard 0/2` and `score --shard 1/2` side by side, then `summarise`; `review` after scoring (commands in the script's header) | rows over 2 h 31 min, 2 processes | `results/regime_sensitivity.json` |
| baselines: plain 1PL, empirical mean (§6.3, Table 2, App E.4) | `python experiments/baselines_p1.py ble`, then `score` (repeat while it exits with 75), then `summarise` | final pass 21 min; reads the regime-sensitivity rows | `results/baselines_p1.json` |
| pooling decomposition (§2, App A.2) | `python experiments/pooling_decomposition.py score`, `recheck`, `summarise` | 84 min, 1 process | `results/pooling_decomposition.json` |
| gate, intervals, single-subject benchmark (§4.3, §6.4, App E.6) | `python experiments/gate_and_ci.py --stage S` for S = gate, evidence, scale, ci, single | under 4 min a stage | `results/gate_and_ci.json` |
| formative run 3 (App D.1) | `python experiments/formative_run3.py` | 0.3 s; needs the `4a882cc7…` archive in `dist/` | `results/formative_run3.json` |
| the hidden runs through the pair-rate oracle (§3, §6.1, §6.2, App D.5 to D.10) | `python experiments/hidden_runs.py`; `--no-rows` rebuilds Table 1 from committed files | 200.7 s with every block, 1 process | `results/hidden_runs.json` |
| the pool evaluation (§7, App C.6) | the pool from `tools/fetch_pool.py` (network); with BLAS held to one thread and `PYTHONHASHSEED=0`, `python -m paiec.pool --fit --jobs 4`, then the stages of `experiments/pool_eval.py` listed in App G.4 | App G.4 | `results/pool_fits.json`, `results/pool_eval.json` |

Per-row files (`data/*_rows`, gitignored) are written by these scripts and read
by the later ones; App G.4 says which script needs which. Some numbers were
computed with earlier library versions: App H.2 maps each number to the code
that produced it, and the gap between the libraries is measured there.

## Figures

    python tools/report_figures.py           # every figure into docs/report/fig/
    python tools/report_figures.py --check   # rebuild in a temporary directory and compare

About 5 seconds. The figures read the committed `results/*.json` only, so they
need no data; their bytes are tied to matplotlib 3.9.2 (the `report` extra).
Each figure is drawn at the width it is printed at, so its type sizes are print
sizes. `docs/report/fig/manifest.json` records each figure's size and smallest
type and glyph, and `--check` also fails if a main-text figure (Figures 1 and 2)
has type under 6 pt, sub- and superscripts included (`docs/report/figures.md`,
"Type size").

## The Kaggle kit

[`kaggle/strong_probe/`](kaggle/strong_probe/README.md) ran Qwen3-14B-AWQ on
Kaggle's two T4 GPUs for the item-side features of report App F.9. Its README
is in Russian. The notebook needs your own Hugging Face token as a Kaggle
Secret and should stay private. Its outputs are not in this repository;
`experiments/strong_llm_eval.py` reads them.

## The report

The source is [`docs/report/draft.md`](docs/report/draft.md), with figures in
`docs/report/fig/` and their captions in `docs/report/figures.md`. The PDF uses
the NeurIPS 2026 template, as the competition asks (about 4 to 8 pages of main
text, references and appendices excluded): the official style file
`neurips_2026.sty`, revision 2026-01-29, with the options `sglblindworkshop`
and `final`. The style file carries no license, so it is not in this
repository: `tools/report_pdf/neurips_2026.sty` is listed in `.gitignore`.
Download the official template,
https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
(sha256 `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`),
and copy `neurips_2026.sty` from it to `tools/report_pdf/`. The build checks
the copy against the pinned sha256
(`c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`,
`STYLE_SHA256` in `tools/build_report_pdf.py`); when the file is missing or
differs, it stops and names the download and both pins. Then build the PDF
with

    python tools/build_report_pdf.py            # dist/report/paiec_report.pdf
    python tools/build_report_pdf.py --strict   # exit 2 on a main text over 8 pages, layout problems, placeholders or TODO markers

`--strict` holds the main text to 8 pages. The main text runs from the title
to the heading "Acknowledgments and Disclosure of Funding"; that section, the
references and the appendices do not count, as the NeurIPS instructions say.
`dist/report/build.json` records the main text's pages
(`pages.main_text_pages`), its length to its last line
(`pages.main_text_measured`), the heading it ends before
(`pages.main_text_ends_before`) and the acknowledgments' page
(`pages.acknowledgments`), and the build prints each main-text section's share
of the pages. The main text fills its 8 pages, so an edit before the
acknowledgments that adds a line needs a cut elsewhere or a move to an
appendix; rebuild with `--strict` after every such edit.

The build needs pandoc 3.1 or later and XeLaTeX (TeX Live). It checks with
kpsewhich that the packages `tools/report_pdf/template.tex` loads are installed
and names any that are missing. pdftotext and pdftoppm (poppler) are optional:
the first for the check that every number and word of the draft is in the PDF,
the second for `--render`. The PDF was built and checked on macOS (pandoc 3.6.2,
TeX Live 2026) with Times New Roman, Menlo and the committed 200-dpi PNG figures
(vector figures need rsvg-convert or cairosvg). Elsewhere the fonts fall back to
STIXGeneral or TeX Gyre Termes (which lacks Cyrillic), and page breaks may
differ. `--strict` fails until the title block's placeholders and the draft's
`TODO(...)` markers are gone. The competition's requirements for the report are
recorded in `docs/report/report_requirements.md`. The PDF goes to OpenReview;
the selected Codabench submission ID, that of archive-3, goes into the
OpenReview form's required field "Codabench submission ID" (digits only), and
no e-mail to the organisers is asked for (`docs/report/report_requirements.md`,
item 7).

## Disclosures

The report's section "Code, data and conduct" and its App G.6 ("Conduct and
disclosure") are the full statement; in short:

- **Training data:** measurement-db only, the public training pool: revision
  `bc8204d8…` for the submission and every result up to it, and revision
  54074453 for the pool evaluation, whose refitted priors were not carried.
- **Formative feedback:** it set global quantities only. Run 1 set the
  test-like regime's defaults, and an audit that read it per pair chose
  between two configurations of the level prior. Run 2 motivated a
  subject-side study and a pooled reading of runs 1 and 2; neither changed the
  shipped model. Run 3 was a regression and latency check. After the code was
  frozen on archive-3 (2026-10-06), the three tables entered a root-free
  analysis of per-pair Brier and ECE (V and ALC minus V), which uses the
  anonymous ids only to find the one subject-benchmark pair that appears in
  two runs and to group pairs by benchmark in its bootstraps, and which chose,
  tuned and changed nothing. S4 (report App B.3) and the post-hoc family
  (App F.8) used only the counts of formative appearances in three bins of V,
  as weights in descriptive fits; nothing passed or was selected. The pool
  evaluation reused the regimes' global settings (level targets, run shape,
  date shift). No input, prior or selection is keyed on an anonymous id.
- **Pretrained models:** Qwen3-Embedding-0.6B, Qwen3-4B-Instruct-2507 and
  Qwen3-14B-AWQ, for research features only. The submission uses no pretrained
  model.
- **The inventory scan** (`experiments/inventory_scan.py`) cloned the
  repositories of the organisers' benchmark inventory, which may include
  hidden-test benchmarks, matched their file paths and deleted the clones;
  App G.6 records what is not confirmed (whether some files were opened).
  Nothing from it enters `paiec/`, `submission/` or `tools/`.
- **Language models in the work:** the blind difficulty ratings of report
  App F.3 were made by the Claude model of the coding session. The code, the
  experiments, the docs and the report were developed with Claude Code
  (Anthropic); the commits' `Co-Authored-By` trailers record it.
- **Funding and competing interests:** the report's unnumbered section
  "Acknowledgments and Disclosure of Funding" states that the work received no
  specific funding and that the author declares no competing interests.

## License and third-party material

The code and documentation written for this repository are released under the
MIT License ([`LICENSE`](LICENSE)), copyright 2026 Nikita L. Polomoshnov. The
license does not cover:

- **measurement-db.** The dataset is not included. It is gated and under its
  own terms: per its dataset card, the curation is licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and each
  benchmark's content keeps its upstream licence. Accept the terms yourself
  before fetching it. No item text from it is in this repository, apart from a
  40-character example in one code comment (`docs/release.md`, section 9). The
  blind-rating files `results/rate2_truth.json` (180 items, 45 each from
  matharena, multi_swebench, real_webagents and swe_rebench) and
  `results/ctrl_truth.json` (60 multi_swebench items) keep, for each rated
  item, its measurement-db item id, its Rasch target and the sha256 digest and
  length of the excerpt that was rated. `docs/release.md` (section 9) rebuilds
  the excerpts from the dataset and checks them against the digests.
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
  holds 161 of them (titles, authors, code and dataset links, venue; one code
  link is stored without its access token), and
  `results/inventory_hand_labels.csv` and `results/inventory_classes.csv`
  repeat titles from the organisers' sheet. They were inputs to the analyses
  of report §8 and App A.1, F.6 and F.8.

## Citation

[`CITATION.cff`](CITATION.cff) gives the citation for the report and this code.
