# Public release checklist

For the person publishing this repository. Written 2026-10-02 against `3e6b770`
(branch `official-protocol`), before the release files were committed; revised
2026-10-03 after the release review, again on 2026-10-03 after the history
rewrite, and on 2026-10-04 after publication. It is the record of what was
decided and done, and the procedure for what is left. The history rewrite is
done (section 9). On 2026-10-03 the release commit was tagged `v1.0.0` and
pushed to https://github.com/doctawho42/PAIEC, with the branches `main` and
`official-protocol` and the 21 lightweight tags named by the old commit ids
(section 7). The commit after it resolves the report's open markers (section
7, step 0); it is to be tagged `v1.0.1` when it is pushed (section 6). Every
command of section 7 is for you to run. Seven-character commit ids
written before the rewrite, in this file and elsewhere, are kept as they were
and resolve through those tags; full old ids are mapped in
`docs/commit-map.txt` (section 9.3).

## 1 Decisions

Taken by the team on 2026-10-02:

- **Author:** Nikita L. Polomoshnov, sole author; affiliation Moscow State
  University; contact e-mail nikitapol@fbb.msu.ru (decided 2026-10-02/03: in the
  report's title block, README.md and CITATION.cff; no other address goes into
  any file).
- **License:** MIT for this repository's own code and documentation (`LICENSE`,
  "Copyright (c) 2026 Nikita L. Polomoshnov"). It does not cover measurement-db
  (section 3.1), the organisers' baseline repository or the few of its lines
  that `paiec/official.py` and `paiec/baselines.py` repeat, the organisers'
  formative tables, or the rows of the organisers' benchmark inventory (README,
  "License and third-party material").
- **Host:** the public GitHub repository https://github.com/doctawho42/PAIEC
  (decided 2026-10-02), filled in README.md, CITATION.cff and the report's
  title block.

Taken by the owner on 2026-10-03, before anything was pushed:

- Rewrite the history once, to settle 3.1 (item text in two results files), 3.2
  (the e-mail address in commit metadata) and the access token of 3.4. Section 9
  records the rewrite, and sections 3.1, 3.2 and 3.4 record the outcome.

Since 2026-10-03 the repository's local git config sets `user.email` to
nikitapol@fbb.msu.ru, so new commits and annotated tags carry that address. The
global config still holds the old address, so a commit or tag made from another
clone needs `git -c user.email=nikitapol@fbb.msu.ru ...`.

Item 3.3 and the rest of 3.4 stand as written in section 3 (the formative
tables are included unless the team decides otherwise).

Taken by the author on 2026-10-04:

- The keywords stay as they are (3.5).
- The row archives of `tools/export_rows.py` are not hosted, and the LLM
  features under `data/features/` are not released (section 2).

The competition needs a public code link in the report and on the OpenReview
form, and the organisers will "review and rerun the released method"
(`docs/report/report_requirements.md`). The README is written for that reader.

## 2 What is in the repository, and what is not

Tracked: `paiec/`, `submission/model.py`, `submission/labeling.py`, `tools/`
(with `tools/report_pdf/`), `experiments/`, `tests/`, `docs/`, `results/`,
`kaggle/strong_probe/` (the kit only), `pyproject.toml`, `requirements.txt`,
`CLAUDE.md`, `LICENSE`, `README.md`, `CITATION.cff`, `.gitignore`.

Excluded, by `.gitignore`:

| path | why it stays out |
|---|---|
| `data/` | measurement-db is gated and under its own terms (curation CC BY-SA 4.0, benchmark content under the upstream licences, per its dataset card: the README on the dataset page at revision `bc8204d8…`, which the original download saved as `data/README.md`; `python -m paiec.fetch` fetches only the 24 parquet tables). The dataset is not redistributed: `data/` stays out of the repository, and no committed file holds its item text, apart from a 40-character example in one code comment (section 3.1; the example: section 9.6). `data/` also holds what is derived from it locally: per-row files (`data/*_rows`, `data/hier_floor`), LLM features and Kaggle exports (`data/features/`), the row-export archives (`data/release_rows/`) and the organisers' full inventory sheet. Fetch with `python -m paiec.fetch`; regenerate the rest with the scripts (report App G.4). |
| `third_party/` | the organisers' baseline repository (validator, streaming client) has no license, so it must never be committed or bundled. Users clone it themselves at `82d330ddcdb16016a3ae9e048db7e588ba2c6a39` (README). |
| `dist/` | build output: the submission archive (`tools/build_submission.py`, written from the gated data and only if every check passes) and the report PDF with its build directory under `dist/report/` (`tools/build_report_pdf.py`). The PDF goes to OpenReview, not into the repository. |
| `submission/prior.json` | fitted from the gated data by the build; rebuilt, not committed. |
| Kaggle outputs | the kit's Output is downloaded to `./kaggle_out/` (now ignored) and copied under `data/features/` (ignored with `data/`). The kit asks for the notebook and its Output to stay private (`kaggle/strong_probe/README.md`). The committed notebook has no outputs (checked: none of its 9 cells has outputs or an execution count), and `kernel-metadata.json` holds the placeholder id `KAGGLE_USERNAME/...` with `is_private: true`. |
| credentials | `.env`, `.env.*`, and now `kaggle.json`. The Kaggle kit reads the Hugging Face token from a Kaggle Secret, never from a file. |
| local tool state | `.DS_Store`, `.ruff_cache/`, `.remember/` and `.claude/settings.local.json` (added now; on the development machine some were ignored before only by a global ignore file). |

Not part of this release, and said so in the report (App G.4, App I.1): the
per-row archives of `tools/export_rows.py` are not hosted (decided by the author
on 2026-10-04; the experiment scripts regenerate the rows from measurement-db,
and the 14B's derived tables also need the Kaggle job of report App F.9), the
LLM features under `data/features/` are not released, and the itemsig rows
cannot be exported.

## 3 Decisions before pushing

Pushing is irreversible in practice (clones, forks, caches), so these come
first. Items 1 to 4 are third-party material or personal metadata that the
history held; none is a credential of ours. Item 5 is the keywords. On
2026-10-03 one rewrite of the history (section 9) settled items 1 and 2 and the
access token of item 4. Item 3 stands as written below (the tables are
included unless the team decides otherwise), and the author confirmed item 5
on 2026-10-04.

1. **Item text from measurement-db in two committed files: removed from every
   commit.** `results/rate2_truth.json` (180 records, 45 each of matharena,
   multi_swebench, real_webagents and swe_rebench) and `results/ctrl_truth.json`
   (60 multi_swebench records) are the samples of the blind difficulty rating
   (report §6.3, App F.3; read by `experiments/llm_rating/analysis.py` and
   `experiments/gate_and_ci.py`). Before the rewrite each record quoted its
   item's text, `content.strip()` cut to at most 700 characters for matharena
   and 450 for multi_swebench, swe_rebench and real_webagents in
   `rate2_truth.json` (no real_webagents text reached that cap; the longest was
   241), and to at most 4,000 in `ctrl_truth.json`. That made 240 records of 239
   distinct items (one item is in both files) and 144,568 characters (63,730 in
   `rate2_truth.json`, 80,838 in `ctrl_truth.json`). Earlier versions of this
   item and of the README gave 700 as the cap for all of `rate2_truth.json`,
   which was wrong for three benchmarks. The excerpts were a partial copy of the
   gated dataset's content and of the upstream benchmarks' text, which this
   repository otherwise does not redistribute (section 2). The upstream licences
   the dataset records (`benchmarks` table, `license`) are:

   | benchmark | records | licence recorded |
   |---|---|---|
   | matharena | 45 | CC-BY-NC-SA-4.0 AND CC-BY-SA-4.0 (non-commercial, share-alike) |
   | multi_swebench | 45 + 60 | Apache-2.0 |
   | swe_rebench | 45 | CC-BY-4.0 |
   | real_webagents | 45 | unknown: nothing known grants republishing these |

   The quoted issue texts also contained e-mail addresses and home-directory
   paths of issue authors' machines; none remain. Each file had one version, in
   every commit since `165b4c0`.

   The rewrite of 2026-10-03 (section 9) replaced each record's `text`, in
   every commit, by `redacted sha256:<hex> chars:<n>`: the sha256 of the
   excerpt's UTF-8 bytes and its length in characters. `uid`, `bench` (in
   `rate2_truth.json` only), `key` (the measurement-db item id), `zb` and `n`
   are unchanged. Neither script reads `text`, and both give identical output
   on the rewritten files, so the rating study still runs from the repository.
   The README's license section now says that the repository holds no item
   text and what the two files keep. The digests recorded for the two files in
   `results/gate_and_ci.json` are of their bytes before the rewrite; section 9
   shows how to re-check them from measurement-db.
2. **The author e-mail in commit metadata: replaced in every commit.** Before
   the rewrite all 21 commits carried, as author and committer, the one address
   configured in the development machine's global git config (19 under the name
   `doctawho42`, 2 under `Nikita L. Polomoshnov`). It was not a GitHub no-reply
   address, and the team decided that no address other than
   nikitapol@fbb.msu.ru goes into any file (section 1). The rewrite of
   2026-10-03 (section 9) set the author and committer e-mail of every commit
   to nikitapol@fbb.msu.ru. It kept names, dates, time zones and messages,
   apart from two commit ids that messages cite (section 9).
   New commits and annotated tags take the address from the repository's local
   config (section 1), and section 4's e-mail check finds any other address. The
   old address occurs in no object of the rewritten repository, and no file in
   the working tree or the report PDF contains it. The commit messages carry the
   coding assistant's `Co-Authored-By` trailers (with Anthropic's no-reply
   address), which the report discloses (App G.6). If nikitapol@fbb.msu.ru is a
   private address of the GitHub account and the account's "Block command line
   pushes that expose my email" setting is on, the push is refused (error
   GH007); check the account's e-mail settings before pushing.
3. **The organisers' formative tables** (`results/formative/run1.txt` to
   `run3.txt`) are stored verbatim, with anonymous subject and benchmark ids and
   per-pair Brier and ECE, no labels. The report relies on them (App D, App G.6).
   Included as they are unless the team decides otherwise.
4. **Rows of the organisers' benchmark inventory.** `results/inventory.csv` is a
   copy of 161 rows of the organisers' inventory sheet (slug, authors, code and
   dataset links, OpenReview id and link, title, venue, curation status; all
   NeurIPS 2025 Datasets and Benchmarks papers; the curation status marks only
   matharena and mmdocrag, both public). `results/inventory_hand_labels.csv`
   holds 80 titles sampled from the organisers' full 1,261-row sheet, with an AI
   agent's labels, and `results/inventory_classes.csv` the 161 titles with their
   keyword classes. `results/clone_scan.csv`, the scan's output, holds per
   repository path counts and up to three example paths.
   `results/strong_repos.csv`, which no script in the repository writes, holds
   for the 15 repositories with model-named paths a model list derived from
   those paths (directory or file names) and one sample path. Neither holds file
   contents. The inventory lists candidates for both pools, so these files name
   possible hidden-test benchmarks; the report discloses the scan (App G.6).
   One row of `inventory.csv` (neurorenderedfake, line 45) had as its code link
   a Harvard Dataverse preview URL whose query string carried an access token
   (`previewurl.xhtml?token=…`): a reviewer link to a dataset that may be
   unpublished, copied from the inventory. It was no credential of ours, but
   publishing it would have republished that access. The rewrite of 2026-10-03
   (section 9) replaced the token by `REDACTED` in every commit and kept the
   rest of the file; exactly one line changed.
   Nothing reads that link (`experiments/inventory_scan.py` clones only
   github.com links). The sha256 recorded for the file in
   `results/inventory_classes.json` (`inputs.inventory_161`) is of its bytes
   before the rewrite; section 9 shows how to re-check it with the organisers'
   sheet. The inventory's public status is not recorded in this repository.
5. **Keywords.** The OpenReview form requires them. The list in the draft's
   title block (`Keywords: predictive evaluation, item response theory, Brier
   score, cold start, PAIEC`) was chosen during the release work, and the
   author confirmed it on 2026-10-04. That line is the one source: the PDF
   build writes it into the PDF's metadata (it is not printed), and
   `tests/test_build_report_pdf.py` checks that CITATION.cff's `keywords`
   repeat it. A change goes there and into CITATION.cff.

## 4 History check

First run on 2026-10-02 over everything reachable from any ref (branches
`master` at `6e19f40` and `official-protocol` at `3e6b770`; no tags, no stash,
no remote, no merge commits): 19 commits, 239 paths ever tracked, 322 distinct
blobs. Values were never printed; matches were inspected masked. The table
records that run. The rewrite of section 9 settled its rows on generic
assignments, item text and e-mail addresses; the results on the rewritten
history follow the commands.

The rewritten history, as checked on 2026-10-03 before the release commit: 21
commits on two branches (`master`, `official-protocol`), 248 paths ever
tracked, 341 distinct blobs, and the 21 lightweight tags of section 9; no
stash, no remote, no merge commits.

| check | result on 2026-10-02, before the rewrite |
|---|---|
| paths ever tracked: `.env*`, `kaggle.json`, `third_party/`, `data/`, `dist/`, `*.parquet`, archives (`.zip`, `.tar`, `.gz`, `.xz`, `.bz2`, `.7z`), `submission/prior.json`, key and certificate files, names containing credential, secret or token, `.DS_Store`, caches, `.pyc`, pickles, arrays and tensors, local tool state | none. Every path ever tracked lies under `docs/`, `experiments/`, `kaggle/`, `paiec/`, `results/`, `submission/` (`model.py`, `labeling.py`), `tests/`, `tools/`, or is one of the five root files (seven since the release files were committed: `.gitignore`, `CITATION.cff`, `CLAUDE.md`, `LICENSE`, `README.md`, `pyproject.toml`, `requirements.txt`) |
| secret-shaped strings in every blob and every commit message: Hugging Face, GitHub, OpenAI, Anthropic, AWS, Slack and Google keys, private-key blocks, JWTs, `kaggle.json` content, bearer headers | none |
| generic assignments (`token=`, `password=`, `api_key=`, `secret=` followed by 16 or more characters) | one: the Dataverse preview link in `results/inventory.csv` (section 3.4); none after the rewrite |
| largest blobs | `results/level_calibration.json` 19.5 MB, `results/testlike_check.json` 10.0 MB, `results/hier_eval.json` 5.7 MB; 8 blobs over 1 MB, none over 50 MB (GitHub warns at 50 MB and refuses 100 MB); about 21 MB of objects in all |
| measurement-db item text (48-character windows of every item's content and grading criterion, at several offsets, against every blob) | `results/rate2_truth.json` and `results/ctrl_truth.json` (section 3.1), removed by the rewrite. Elsewhere only instruction lines shared by hundreds of matharena items ("Please reason step by step, ...", the proof and Kangaroo templates) in the Kaggle kit, `results/strong_llm_eval.json` and three tests: boilerplate, not item content. A 40-character example in a code comment, shorter than the windows, is listed in section 9 |
| e-mail addresses in tracked content | the organisers' public contact in `docs/report/report_requirements.md`; the addresses inside quoted item text (section 3.1), removed with it. Commit metadata: section 3.2 |
| notebook outputs | `kaggle/strong_probe/paiec_strong_probe.ipynb` has none |

Re-run it on the final commit, just before pushing (prints paths or counts only):

    # paths that must never have been committed
    git rev-list --all | while read c; do git ls-tree -r --name-only "$c"; done | sort -u \
      | grep -E '(^|/)\.env($|\.)|kaggle\.json|^third_party/|^data/|^dist/|\.parquet$|\.zip$|^submission/prior\.json$' \
      || echo "no forbidden paths"
    # key-shaped strings anywhere in history: a count, never the match (expect 0;
    # "github[_]pat_" is written so that this file does not match itself)
    git log --all -p | grep -c -E 'hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github[_]pat_|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-|AIza[0-9A-Za-z_-]{35}|BEGIN [A-Z ]*PRIVATE KEY'
    # generic assignments (expect 0; 1 before the rewrite of section 9, the
    # Dataverse link of section 3.4)
    git log --all -p | grep -c -i -E '(token|password|passwd|api_?key|secret)[[:space:]]*[=:][[:space:]]*["'\'']?[A-Za-z0-9_-]{16,}'
    # blobs over 10 MB (expect the two results files above)
    git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
      | awk '$1=="blob" && $2>10000000 {print $2, $3}'
    # commit e-mail: author or committer addresses other than the decided one (expect 0)
    git log --all --format='%ae%n%ce' | sort -u | grep -c -v -x 'nikitapol@fbb.msu.ru'
    # item text in the rating files, in every commit: strings longer than 16
    # characters other than the redaction marker (expect 0)
    python - <<'EOF'
    import json, re, subprocess
    red = re.compile(r"redacted sha256:[0-9a-f]{64} chars:[0-9]+")
    n = 0
    for c in subprocess.run(["git", "rev-list", "--all"], capture_output=True, text=True).stdout.split():
        for f in ("results/rate2_truth.json", "results/ctrl_truth.json"):
            recs = json.loads(subprocess.run(["git", "show", f"{c}:{f}"], capture_output=True).stdout)
            n += sum(isinstance(v, str) and len(v) > 16 and not red.fullmatch(v) for r in recs for v in r.values())
    print(n)
    EOF

The first four were run as written on 2026-10-02 and gave: no forbidden paths,
0, 1, and the two files; the last two did not exist then. On the rewritten
history, on 2026-10-03, the six gave: no forbidden paths, 0, 0, the same two
files (`results/level_calibration.json` 19,471,738 bytes,
`results/testlike_check.json` 10,045,422 bytes), 0 and 0.

Tests: on 2026-10-03, after the rewrite, `python -m pytest -q` with the data
and the baseline repository present gave 729 passed, 1 warning, in 5 min 10 s.
The suite has 528 test functions over 24 test files and 729 collected tests;
the release commit changes no test. Without the 21 old-id tags one test fails
wherever `dist/` holds archive-3 (section 9).

A clean clone without `data/` or the baseline repository (2026-10-03, at
`ec62bf3`, the rewritten head before the release commit, with the 21 tags)
gave 719 passed, 10 skipped, 2 warnings, in 5 min 2 s. The 10 skipped tests
need measurement-db (5), the organisers' baseline repository or streaming
client (4), or archive-3 in `dist/` (1).

Earlier runs, before the last tests were added: on 2026-10-03, `3e6b770` plus
the then uncommitted release work, with the data and the baseline repository,
gave 717 passed, none skipped, 1 warning, in 3 min 58 s (717 collected tests
over 24 test files). On 2026-10-02, a `git archive` export of `3e6b770` without
`data/`, `third_party/` or git history gave 687 passed, 12 skipped (16 min
37 s), and the release review's clean clone (the release files copied in, no
`data/` or `third_party/`) gave 704 passed, 10 skipped in 9 min 59 s. `e880f0f`
then added 9 test functions to `tests/test_build_report_pdf.py`.

## 5 Absolute local paths in committed files

Some committed files name paths on the development machine, mostly the coding
session's scratch directory under `/private/tmp/claude-501/...` and two archive
copies under the user's `Downloads`. They reveal the local user name and the
scratch layout, nothing else.

| file | lines | what |
|---|---|---|
| `results/formative_feedback.json` | 7 | the formative archives' paths in `passes` commands and `archives` records |
| `results/regime_sensitivity.json` | 2 | scratch work directories of the review rerun |
| `results/harness_thresholds.json` | 1 | a scratch rows directory |
| `results/itemsig_eval.json` | 1 | the scratch `--rows` directory |
| `docs/plans/p1a_regime_sensitivity.md` | 1 | a scratch directory |
| `docs/report/edit_plan.md` | 2 | the word counter in scratch |
| `docs/report/style_guide.md` | 1 | the scratch directory of the report's rewrite (line 3) |
| `tests/test_kaggle_probe.py` | 1 | a scratch CSV, compared only if it exists |
| `docs/report/provenance_facts.md` | 1 | the interpreter under the home directory (`~/anaconda3`) |

They are left as they are, and the one rewrite of section 9 did not touch
them. The results files' sha256 digests are recorded elsewhere: in
`docs/report/fig/manifest.json`, which `tests/test_report_figures.py` and
`python tools/report_figures.py --check` verify, and in
`results/formative_run3.json`, `results/gate_and_ci.json` and
`docs/plans/p1a_regime_sensitivity.md`. Editing a results file breaks those
checks and the provenance chain of report App H, for paths that reveal only the
local user name and the scratch layout. The docs and the test are records of
how the work was done and are left as they are too. (The home-directory paths
inside the quoted item text went with it, section 3.1.)

## 6 The tag

- **Name:** `v1.0.0` (CITATION.cff's `version` is `1.0.0`). Annotated: it
  records the tagger's address from `user.email`, which the repository's local
  config sets to nikitapol@fbb.msu.ru (section 1).
- **Commit:** the release commit, that is, the head of `official-protocol` once
  the follow-up edits after the history rewrite are committed: this file,
  README.md, CITATION.cff, CLAUDE.md, `docs/findings.md`, the report's draft and
  provenance notes, and the new `docs/commit-map.txt` (section 9). The release
  files (LICENSE, README.md, CITATION.cff, .gitignore, CLAUDE.md, this file),
  the report's PDF tooling and the other release work are committed already,
  and the URL is filled in.
- **Why there, and not at `4d2cc4f`:** the submission archive's code
  (`paiec/{__init__,hier,prior,predict,fitting,irt,subjects,mcq}.py`,
  `submission/`, `tools/build_submission.py`, `requirements.txt`) is unchanged
  since `4d2cc4f`, the commit archive-3 (`4a882cc7…`, formative run 3) was built
  from: `git diff --stat 4d2cc4f 3e6b770` over those paths is empty, and so is
  the diff against the working tree. So a build at the tag gives archive-3's
  bytes on the development machine with its default BLAS threading (report
  App G.3), while the tag also carries what `4d2cc4f` lacks: the license, the
  README, the later results, the figures and the report the organisers will
  rerun from. Checked on 2026-10-03: a `git archive` export of `3e6b770`, with
  `data/` and the baseline repository linked in, rebuilt sha256
  `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090` byte for
  byte (all 11 members equal to the archive in `dist/` in bytes and zip
  metadata, the run check identical at every budget, the validator OK) in 80 s,
  with a peak resident memory of 1.0 GB for its largest process, on the M1 Pro
  at the default BLAS threading. A clean worktree of `3e6b770`, rebuilt during
  the release review, gave the same bytes in 102.8 s with a peak of 878 MiB.
  Old ids such as `4d2cc4f` and `3e6b770` name commits of the history before
  the rewrite. They resolve through the tags of section 9 to the rewritten
  commits (`a1fbe79`, `b8e6a1f`), whose archive code consists of the same git
  objects as before. After the rewrite, `python tools/build_submission.py`
  rebuilt the same sha256 byte for byte (section 9).
- **One release tag.** An alternative, `v1.0-submission` at `3e6b770`, was
  proposed during the release work and is dropped: the archive code is the same
  at both commits, and only the release head carries the license, README and
  CITATION.cff. At the release commit the report's App I said that the
  release commit is to be tagged `v1.0.0` and that its archive code equals that
  of `4d2cc4f`. Since the commit after it, App A.4 and App H.4 say that the
  release commit is tagged `v1.0.0`, and App H.4 that the archive code is that
  of `4d2cc4f`. The 21 lightweight tags of section 9 only name old commit ids.
- **Check before tagging** (prints the line only if nothing changed; `4d2cc4f`
  resolves through its tag):

      git diff --quiet 4d2cc4f HEAD -- paiec/__init__.py paiec/hier.py paiec/prior.py \
        paiec/predict.py paiec/fitting.py paiec/irt.py paiec/subjects.py paiec/mcq.py \
        submission tools/build_submission.py requirements.txt \
        && echo "archive code unchanged since 4d2cc4f"

- The test counts quoted in report App A.4 and App G.2 are those of section 4:
  528 test functions over 24 test files, 729 collected tests. The release
  commit changes no test file.
- If the report or results change after the tag is pushed, tag the new commit
  `v1.0.1` rather than moving `v1.0.0`. The report changed after the push: the
  commit after the release commit resolves the draft's markers (section 7,
  step 0). Tag it `v1.0.1` when it is pushed, as the release commit was tagged
  (section 7, step 4, with `v1.0.1` and its own message).

## 7 Publish

Before: section 3 settled or standing as written there (the keywords of 3.5
belong to the report and the OpenReview form, not to the push), the rewrite of
section 9 done and section 4's checks run on it, and `python -m pytest -q`
green. After step 0, `git status` on
`official-protocol` shows nothing but ignored files; re-run section 4's
commands then, on the final commit. The local default branch is `master` (at
`f00f593`, old id `6e19f40`), an ancestor of `official-protocol` (19 commits
behind before the release commit), so `main` is a fast-forward. Push only from
this repository, whose history is the rewritten one: never push from a copy
that holds the old history, and never use `git push --mirror`.

    # 0. The release commit itself: the follow-up edits after the rewrite of
    #    section 9, docs/commit-map.txt included. The repository's local config
    #    supplies the address of section 3.2; check it first, since the global
    #    config still holds the old one:
    git config --local user.email            # must print nikitapol@fbb.msu.ru
    git commit ...
    #    The release commit leaves 8 body TODO(...) markers in the draft (10
    #    before it): App G.4's table, App G.6, and App I with the review map.
    #    They do not block the tag, but they would print in the OpenReview PDF
    #    (step 2: --strict exits 2 while one is left). Resolve them (they are
    #    the report's text) in a later commit and build the PDF from it; if
    #    v1.0.0 is already pushed by then, tag that commit v1.0.1 (section 6).
    #    Done on 2026-10-04 in the commit after the release commit: the draft
    #    has 0 markers, and --strict exits 0. v1.0.0 was pushed on 2026-10-03,
    #    so that commit is the one to tag v1.0.1.

    # 1. Create the empty public repository: no README, license or .gitignore
    #    (they come from here). With the GitHub CLI, or on github.com.
    gh repo create doctawho42/PAIEC --public \
      --description "PAIEC (NeurIPS 2026): a hierarchical response model for a benchmark-level cold start"

    # 2. The URL (https://github.com/doctawho42/PAIEC) and the affiliation are
    #    already filled in (section 1). Before tagging:
    python -m pytest -q
    python tools/build_report_pdf.py --strict   # exit 2 while a body TODO(...) marker is
                                                # left; the PDF for OpenReview needs exit 0

    # 3. Remote, branches, and the 21 old-id tags of section 9, named from
    #    docs/commit-map.txt (or push them with "git push origin --tags" after
    #    step 4, which also pushes v1.0.0)
    git remote add origin https://github.com/doctawho42/PAIEC.git
    git merge-base --is-ancestor master official-protocol && echo "fast-forward OK"
    git branch -m master main
    git branch -f main official-protocol     # fast-forward; run while official-protocol is checked out
    git push -u origin main                  # main becomes the default branch
    git push origin official-protocol        # the report names this branch
    git push origin $(grep -v '^#' docs/commit-map.txt | cut -c1-7 | sed 's|^|refs/tags/|')

    # 4. Tag (section 6); the annotated tag records the local user.email
    git tag -a v1.0.0 -m "PAIEC 2026 release: report source, results, submission code (archive-3, unchanged since 4d2cc4f)" main
    git push origin v1.0.0

Optionally, a GitHub release from the tag (`gh release create v1.0.0 --verify-tag
--title "v1.0.0" --notes-file NOTES.md`, or on github.com), with these notes:

> Code, results and report source for *Calibrating a hierarchical response model
> for a benchmark-level cold start* (Nikita L. Polomoshnov), an entry to the
> Predictive AI Evaluation Competition (PAIEC), NeurIPS 2026. The submission's
> code is unchanged since commit 4d2cc4f; `python tools/build_submission.py`
> rebuilds the archive (sha256 4a882cc7… on the development machine, report
> App G.3). The code and docs are MIT-licensed. measurement-db is not included:
> it is gated, so accept its terms and run `python -m paiec.fetch` (revision
> bc8204d8…). Its curation is CC BY-SA 4.0, and the benchmarks' content keeps
> its upstream licences. No item text from it is included, apart from a
> 40-character example in one code comment. The history was rewritten once
> before publication; seven-character commit ids written before then, such as
> 4d2cc4f, resolve through tags of the same name, and `docs/commit-map.txt`
> maps every old full id to its new one. The organisers'
> baseline repository has no license and is not included; clone it at commit
> 82d330dd.

Do not attach the submission archive or the data to the release; the report PDF
may be attached once it is final.

After pushing, check on GitHub that the default branch is `main`, that the
license is detected as MIT, that "Cite this repository" appears (it reads
CITATION.cff), and that the tags are `v1.0.0` and the 21 old-id tags. Check
that secret scanning and push protection are on (the repository's code
security settings); both are free for public repositories.

## 8 After publishing

The repository URL https://github.com/doctawho42/PAIEC is already in
`README.md` (the clone command), `CITATION.cff` (`repository-code`, twice) and
the report's title block (`docs/report/draft.md`, "Code:"). The placeholder
`"OWNER" + "/REPO"` survives only, by design, in `tools/build_report_pdf.py`
(`PLACEHOLDERS`, what `--strict` flags) and in a synthetic title block of
`tests/test_build_report_pdf.py`; do not replace either.

Also, after publishing:

- `docs/report/draft.md`: the "Code:" line may point at a tag,
  `https://github.com/doctawho42/PAIEC/tree/v1.0.1` once that tag exists
  (section 6), or `.../tree/v1.0.0`, whose draft still holds the markers.
  Rebuild the PDF with
  `python tools/build_report_pdf.py --strict` from the later commit that
  resolves the draft's TODO(...) markers (section 7, step 0); at the tagged
  commit, which still holds them, `--strict` exits 2. In that commit, App A.4
  and App H.4 say that the release commit is tagged `v1.0.0`; at the release
  commit, App A.4 and App I said "to be tagged".
- The OpenReview form's code link and keywords (the draft's "Keywords:" line,
  section 3.5), and the e-mail to the organisers naming the selected Codabench
  submission (`docs/report/report_requirements.md`, items 4 and 7).

## 9 The history rewrite

### 9.1 What was done

On 2026-10-03, before anything was pushed, the owner decided to rewrite the
history once (section 1). It had to change three things: the e-mail address of
every commit (section 3.2), the measurement-db item text in two results files
(section 3.1) and the access token in `results/inventory.csv` (section 3.4).

The tool was `git filter-branch` of git 2.50.1, which is built in;
`git filter-repo` was not installed, and nothing was downloaded. It ran once,
on a fresh bare clone of the repository (`git clone --bare --no-local`), over
both branches, `master` and `official-protocol`. Before the run the history had
21 commits, no tags, no stash, no remote and no merge commits. The run was
first rehearsed on a copy, and the real run reproduced the rehearsal's commit
map byte for byte. It used three filters:

- `--env-filter` set `GIT_AUTHOR_EMAIL` and `GIT_COMMITTER_EMAIL` to
  nikitapol@fbb.msu.ru. Names, dates and time zones are unchanged: 19 commits
  under `doctawho42`, 2 under `Nikita L. Polomoshnov`.
- `--index-filter` put three precomputed blobs at the three paths of section
  9.2, in every commit.
- `--msg-filter` mapped the two commit ids that commit messages cite to the new
  ids: the message of old `380fecf` cites `78e303e`, and that of old `035d167`
  cites `b68492c`. They now cite `6918fbc` and `1cfd2a8`, which the tags
  `78e303e` and `b68492c` also name. No other message changed, and the
  `Co-Authored-By` and `Claude-Session` trailers were kept.

No other path or blob changed.

### 9.2 The three files

Each file had one version, unchanged in all 21 commits, and now has one new
version in all 21 commits.

| file | before: blob, sha256, bytes | after: blob, sha256, bytes |
|---|---|---|
| `results/rate2_truth.json` (180 records) | `4c5959eb…`, `4da6f6b566df9aa8…`, 86,672 | `655aefd2…`, `04ab55851615fe39…`, 37,112 |
| `results/ctrl_truth.json` (60 records) | `d523219f…`, `c3975331698d1b7b…`, 91,749 | `2b4d4845…`, `9561597b19332969…`, 10,862 |
| `results/inventory.csv` | `5519a29a…`, `583369336731135b…`, 69,096 | `7dc801c1…`, `e70303baa18cb5a7…`, 69,068 |

The full sha256 of the new files: `results/rate2_truth.json`
`04ab55851615fe39429e6f60df1111a72e612fd8ef6c5d282696ba346ef813f3`,
`results/ctrl_truth.json`
`9561597b19332969ea75fa01e2c5245809a353a1ad31b13af607aaf5203d498f` and
`results/inventory.csv`
`e70303baa18cb5a7ea08bff2df59139f96b321f24776a98a76575be6a4da4fcd`.

In the two truth files, each record's `text` value is now the string
`redacted sha256:<64 hex> chars:<n>`, where the hex is the sha256 of the
excerpt's UTF-8 bytes and n its length in characters. Only that JSON string
changed. Every other byte is unchanged: the keys `uid`, `bench` (in
`rate2_truth.json` only), `key`, `zb` and `n`, their values, and the
serialisation. `key` is the measurement-db item id, and the excerpt was
`content.strip()[:cap]` of that item, with the caps of section 3.1. Only `experiments/llm_rating/analysis.py` (which reads `uid`,
`bench` and `zb`) and `experiments/gate_and_ci.py` (`bench`, `uid`, `zb` and
`key`) read the truth files. Neither reads `text`, both give identical output
on the rewritten files, and no test reads the files.

In `results/inventory.csv`, line 45 (neurorenderedfake), column `code_url`,
`previewurl.xhtml?token=<value>` became `previewurl.xhtml?token=REDACTED`.
Exactly one line changed. Nothing reads that URL:
`experiments/inventory_scan.py` clones only github.com links.

### 9.3 Commit ids: tags named by the old ids

Every commit id changed. `docs/commit-map.txt` lists the 21 pairs, the old and
the new full id, oldest first. The old tips were `6e19f40` (`master`, now
`f00f593`) and `e880f0f` (`official-protocol`, now `ec62bf3`, before the
release commit). Each old 7-character id is also a lightweight tag on its
rewritten commit. Git resolves a ref before an abbreviated object id, so every
old 7-character id written in the docs, results files, scripts, tests and dated
records resolves in the published repository: `git show 4d2cc4f` shows the
rewritten commit `a1fbe79`. Seven results files
(`results/formative_feedback.json`, `formative_run3.json`, `gate_and_ci.json`,
`itemsig_eval.json`, `level_calibration.json`, `ship_confirm.json`,
`subject_side.json`) record full 40-character old ids in their `head` fields.
Those do not resolve as written; their first seven characters do, and
`docs/commit-map.txt` maps each to its new id. Lightweight tags carry no
identity. The tags are pushed with the branches (section 7, step 3).

The ids were kept and tagged, not edited, because about 500 citations of 16 old
ids sit in 56 files, many of them pinned by recorded digests: results files
whose digests `docs/report/fig/manifest.json` and `results/gate_and_ci.json`
record, lines 86 to 329 of `experiments/regime_sensitivity.py`, the first
30,865 bytes of `docs/plans/p1a_regime_sensitivity.md`,
`experiments/script_edits.json`, and `submission/model.py` inside archive-3. Code also passes old ids to git:
without the tags,
`tests/test_formative_run3.py::test_main_writes_a_complete_record` fails
wherever `dist/` holds archive-3, and `experiments/hier_floor_replay.py`,
`experiments/formative_feedback.py`, `experiments/ship_confirm.py` and section
6's diff command break. So the text keeps the old ids everywhere, and nothing
was substituted.

### 9.4 Checks

Run on 2026-10-03; all passed.

- For each of the 21 old and new commit pairs: parents mapped; names,
  timestamps and zones identical; e-mail changed; the tree differs in exactly
  the three paths; and the subtrees of the archive code are the same objects.
  Old and new histories have 341 blobs each and differ only in the three
  replaced blobs.
- Section 4's history checks on the rewritten history: no forbidden paths, 0
  key-shaped strings, 0 generic assignments (1 before, the Dataverse link), and
  the same two blobs over 10 MB. 21 commits, 248 paths ever tracked, 341 blobs.
- The e-mail check: 0. The only author and committer address is
  nikitapol@fbb.msu.ru, and the old address occurs 0 times in all objects of
  the rewritten repository.
- The item-text check (every string longer than 16 characters in the two truth
  files, in every commit, other than the redaction marker): 0.
- `python tools/build_submission.py` on the rewritten tree rebuilt archive-3,
  sha256 `4a882cc7d410e6a9085e4b1044b3e45aa50c370b74901bb55fa30cb1054a5090`,
  byte for byte.
- `python -m pytest -q` with `data/` and the baseline repository present: 729
  passed, 1 warning, in 5 min 10 s (section 4).
- The report PDF built at `ec62bf3`, the rewritten head before the release
  commit, was byte-identical to the build before the rewrite (sha256
  `6b3fcab59101f555…`). The release commit changes the draft, so a build at
  `v1.0.0` gives other bytes.

### 9.5 Recorded digests of the old bytes

Three records hold digests of the three files as they were before the rewrite.
They are left as they are, as records of the bytes that were read, and no test
reads them:

- `results/gate_and_ci.json`, `passes[*].inputs_sha256`: `4da6f6b566df9aa8`
  (`results/rate2_truth.json`) and `c3975331698d1b7b`
  (`results/ctrl_truth.json`), in 13 passes each;
- `results/inventory_classes.json`, `inputs.inventory_161.sha256`
  (`583369336731135b…`);
- `docs/report/provenance_facts.md` line 170, the same inventory digest.

The two blocks below re-check them from the rewritten files. Both ran on
2026-10-03, with measurement-db at revision `bc8204d8…` in `data/` and the
organisers' full inventory sheet (1,261 rows) as `data/inventory_sheet.csv`;
`results/inventory_classes.json` records that sheet's sha256, `882809d0…`
(`inputs.sheet`). They printed the recorded digests:

    rate2_truth.json 4da6f6b566df9aa8 matches the recorded digest
    ctrl_truth.json c3975331698d1b7b matches the recorded digest
    583369336731135b with the token restored (recorded: 583369336731135b)

The first block rebuilds each excerpt of the truth files from measurement-db
(the item's `content`, stripped and cut to the recorded length), checks it
against its digest, puts it back and hashes each file as it was:

    python - <<'EOF'
    import hashlib, json, re
    import pandas as pd
    RED = re.compile(r"redacted sha256:([0-9a-f]{64}) chars:([0-9]+)")
    RECORDED = {"rate2_truth.json": "4da6f6b566df9aa8", "ctrl_truth.json": "c3975331698d1b7b"}
    items = {}
    for name, default_bench in (("rate2_truth.json", None), ("ctrl_truth.json", "multi_swebench")):
        recs = json.load(open(f"results/{name}"))
        for r in recs:
            b = r.get("bench", default_bench)
            if b not in items:
                t = pd.read_parquet(f"data/{b}/items.parquet", columns=["item_id", "content"])
                items[b] = {str(i): ("" if pd.isna(c) else str(c)) for i, c in zip(t.item_id, t.content)}
            digest, n = RED.fullmatch(r["text"]).groups()
            text = items[b][r["key"]].strip()[:int(n)]
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, (name, r["uid"])
            r["text"] = text
        got = hashlib.sha256(json.dumps(recs).encode()).hexdigest()[:16]
        print(name, got, "matches the recorded digest" if got == RECORDED[name] else "DIFFERS")
    EOF

The second block puts the inventory's token back from the organisers' sheet
and hashes the file as it was. It prints a digest, never the token:

    python - <<'EOF'
    import csv, hashlib
    row = next(r for r in csv.DictReader(open("data/inventory_sheet.csv", encoding="utf-8", newline=""))
               if r["benchmark_name_slug"] == "neurorenderedfake")
    token = row["code_url"].split("token=", 1)[1].encode()
    raw = open("results/inventory.csv", "rb").read()
    assert raw.count(b"previewurl.xhtml?token=REDACTED") == 1
    old = raw.replace(b"previewurl.xhtml?token=REDACTED", b"previewurl.xhtml?token=" + token, 1)
    print(hashlib.sha256(old).hexdigest()[:16], "with the token restored (recorded: 583369336731135b)")
    EOF

### 9.6 Kept on purpose

- `experiments/icl_probe.py` line 233, a code comment, quotes as an example the
  40-character text of one real_webagents item, in every commit from `4d2cc4f`
  on (14 before the release commit). Removing it would change the script's
  bytes, so the digest that `results/icl_probe.json` records for it would no
  longer match; removing it now would also take a second rewrite of every
  commit from `4d2cc4f` on.
- The `Claude-Session` trailers of two commit messages (old `165b4c0` and
  `6e19f40`) and the same session link in `docs/report/provenance_facts.md`
  line 300. It is a link to the coding session, not a credential, and the
  report discloses the assistant (App G.6).
- Derived data, unchanged by the rewrite: per-(subject, pseudo-benchmark)
  accuracy aggregates keyed by measurement-db subject ids in
  `results/testlike_check.json` (27,159 records, 207 subjects),
  `results/level_calibration.json` (6,552 records, 209 subjects) and
  `results/level_audit.json` (657 records, 95 subjects); item ids with Rasch
  targets in the truth files; and the 160 `PROBE_IDS` in the Kaggle kit. No
  file holds per-response labels (subject by item correctness), and none holds
  item text beyond the comment above (the instruction lines shared by many
  matharena items, section 4, are boilerplate, not item content).
- The absolute local paths of section 5.

### 9.7 The backup

The history before the rewrite is kept in a private off-line backup, a git
bundle and a copy of `.git`, outside the repository. It holds the old address,
the item text and the token, so it is never pushed or published. The working
repository was then switched to the rewritten history, and its objects from
before the rewrite were pruned.
