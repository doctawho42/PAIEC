# Public release checklist

For the person publishing this repository. Nothing here has been pushed, no
remote has been added and no tag created: every command below is for you to run.
Written 2026-10-02 against `3e6b770` (branch `official-protocol`), before the
release files were committed; revised 2026-10-03 after the release review.

## 1 Decisions

Taken by the team on 2026-10-02:

- **Author:** Nikita L. Polomoshnov, sole author; affiliation Moscow State
  University; contact e-mail nikitapol@fbb.msu.ru (decided 2026-10-02/03: in the
  report's title block, README.md and CITATION.cff; no other address goes into
  any file).
- **License:** MIT for this repository's own code and documentation (`LICENSE`,
  "Copyright (c) 2026 Nikita L. Polomoshnov"). It does not cover measurement-db
  or the excerpts of it in two results files (section 3.1), the organisers'
  baseline repository or the few of its lines that `paiec/official.py` and
  `paiec/baselines.py` repeat, the organisers' formative tables, or the rows of
  the organisers' benchmark inventory (README, "License and third-party
  material").
- **Host:** the public GitHub repository https://github.com/doctawho42/PAIEC
  (decided 2026-10-02), filled in README.md, CITATION.cff and the report's
  title block.

Not decided yet: the questions of section 3 (the keywords are 3.5). 3.1 and 3.2
must be settled before anything is pushed, and 3.2 before the release commit is
made.

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
| `data/` | measurement-db is gated and under its own terms (curation CC BY-SA 4.0, benchmark content under the upstream licences, per its dataset card: the README on the dataset page at revision `bc8204d8…`, which the original download saved as `data/README.md`; `python -m paiec.fetch` fetches only the 24 parquet tables). The dataset is not redistributed: `data/` stays out of the repository, and the excerpts two results files quote are section 3.1's decision. `data/` also holds what is derived from it locally: per-row files (`data/*_rows`, `data/hier_floor`), LLM features and Kaggle exports (`data/features/`), the row-export archives (`data/release_rows/`) and the organisers' full inventory sheet. Fetch with `python -m paiec.fetch`; regenerate the rest with the scripts (report App G.4). |
| `third_party/` | the organisers' baseline repository (validator, streaming client) has no license, so it must never be committed or bundled. Users clone it themselves at `82d330ddcdb16016a3ae9e048db7e588ba2c6a39` (README). |
| `dist/` | build output: the submission archive (`tools/build_submission.py`, written from the gated data and only if every check passes) and the report PDF with its build directory under `dist/report/` (`tools/build_report_pdf.py`). The PDF goes to OpenReview, not into the repository. |
| `submission/prior.json` | fitted from the gated data by the build; rebuilt, not committed. |
| Kaggle outputs | the kit's Output is downloaded to `./kaggle_out/` (now ignored) and copied under `data/features/` (ignored with `data/`). The kit asks for the notebook and its Output to stay private (`kaggle/strong_probe/README.md`). The committed notebook has no outputs (checked: none of its 9 cells has outputs or an execution count), and `kernel-metadata.json` holds the placeholder id `KAGGLE_USERNAME/...` with `is_private: true`. |
| credentials | `.env`, `.env.*`, and now `kaggle.json`. The Kaggle kit reads the Hugging Face token from a Kaggle Secret, never from a file. |
| local tool state | `.DS_Store`, `.ruff_cache/`, `.remember/` and `.claude/settings.local.json` (added now; on the development machine some were ignored before only by a global ignore file). |

Not part of this release, and said so in the report (App G.4, App I): the
per-row archives of `tools/export_rows.py` are not hosted (hosting them is a
separate team decision, under measurement-db's CC BY-SA terms), and the itemsig
rows cannot be exported.

## 3 Decide before pushing

Pushing is irreversible in practice (clones, forks, caches), so settle these
first. Items 1 to 4 are third-party material or personal metadata already in
the history; none is a credential of ours. Item 5 is the keywords.

1. **Item text from measurement-db in two committed files.** `results/rate2_truth.json`
   (180 items, 45 each of matharena, multi_swebench, real_webagents and
   swe_rebench, text cut to at most 700 characters) and `results/ctrl_truth.json`
   (60 multi_swebench items, text up to 4,000 characters) are the samples of the
   blind difficulty rating (report §6.3, App F.3; read by
   `experiments/llm_rating/analysis.py`). Together they quote the text of 239
   distinct items, about 144,000 characters: each excerpt is an item's full text
   or its first 700 or 4,000 characters (checked against `data/` on 2026-10-03).
   That is a partial copy of the gated dataset's content and of the upstream
   benchmarks' text, which this repository otherwise does not redistribute
   (section 2). The upstream licences the dataset records (`benchmarks` table,
   `license`) are:

   | benchmark | excerpts | licence recorded |
   |---|---|---|
   | matharena | 45 | CC-BY-NC-SA-4.0 AND CC-BY-SA-4.0 (non-commercial, share-alike) |
   | multi_swebench | 45 + 60 | Apache-2.0 |
   | swe_rebench | 45 | CC-BY-4.0 |
   | real_webagents | 45 | unknown: nothing known grants republishing these |

   The README's license section now states these licences, links CC BY-SA 4.0,
   says the excerpts are shortened and are not under the MIT License, and
   carries a `TODO(team)` comment pointing here; it no longer reads as if
   keeping them were settled. The quoted issue texts also contain e-mail
   addresses (in `ctrl_truth.json`: one at a company domain, the others git,
   GitHub and CI-runner addresses; in `rate2_truth.json`: one at example.com)
   and home-directory paths of issue authors' machines (both files). Both files
   are in every commit since `165b4c0`, and their digests are recorded in
   `results/gate_and_ci.json`. Options:
   - keep them: this needs the team's judgement that quoting them is compatible
     with measurement-db's terms and each upstream licence, the real_webagents
     excerpts above all, or the organisers' answer (contact in
     `docs/report/report_requirements.md`);
   - delete them, or only the real_webagents rows, in a new commit: the old
     bytes remain in the pushed history, the rating study's analysis can no
     longer run from the repository, and editing a results file breaks the
     digests recorded for it (section 5);
   - rewrite history to remove them (for example with `git filter-repo`): this
     changes every commit hash, and the report cites about twenty of them
     (`b68492c`, `ee5085a`, `4d2cc4f`, ...) as provenance. Not recommended.

   If they are removed, drop the rest of the README's measurement-db bullet
   (after its `TODO(team)` comment) and the release notes' sentence on them
   (section 7).
2. **The author e-mail in commit metadata.** All 19 commits carry, as author and
   committer, the one e-mail address configured in local git (17 under the name
   `doctawho42`, 2 under `Nikita L. Polomoshnov`); it is not a GitHub no-reply
   address, and the team decided that no e-mail address goes anywhere. Pushing
   publishes it, and the current `user.email` (from the global git config) would
   stamp it on the release commits still to be made too, so decide this
   **before the release commit** (section 7, step 0). If GitHub's "Block command
   line pushes that expose my email" setting is on, the push is refused (error
   GH007). For new commits, commit with the no-reply address shown in GitHub's
   e-mail settings (`git -c user.email=<that address> commit ...`, or set it in
   the repository's config; changing the config is the user's decision). For
   the 19 existing commits the choice is a history rewrite (for example
   `git filter-repo --mailmap`, with the cost of section 3.1's last option) or a
   recorded decision that the team accepts publishing the address. No file in
   the working tree, the history's blobs or the report PDF contains it. The
   commit messages also carry the coding assistant's `Co-Authored-By` trailers
   (with Anthropic's no-reply address), which the report discloses (App G.6).
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
   keyword classes. `results/clone_scan.csv` and `results/strong_repos.csv` are
   the scan's outputs: per repository, path counts and up to three example
   paths, no file contents. The inventory lists candidates for both pools, so
   these files name possible hidden-test benchmarks; the report discloses the
   scan (App G.6, with a `TODO(team)` there).
   One row of `inventory.csv` (neurorenderedfake) has as its code link a Harvard
   Dataverse preview URL whose query string carries an access token
   (`previewurl.xhtml?token=…`): a reviewer link to a dataset that may be
   unpublished, copied from the inventory. It is no credential of ours, but
   publishing it republishes that access. Options: keep it (the file's sha256 is
   recorded in `results/inventory_classes.json`, `inputs.inventory_161`, and
   editing the file breaks that record); or blank the link in a new commit and
   note the changed digest (the history keeps the old bytes). The team decides;
   the inventory's public status is not recorded in this repository.
5. **Keywords.** The OpenReview form requires them. The list now in the draft's
   title block (`Keywords: predictive evaluation, item response theory, Brier
   score, cold start, PAIEC`) was chosen during the release work, not by the
   team, and the report's App I still says `TODO(team)` for it. That line is
   the one source: the PDF build writes it into the PDF's metadata (it is not
   printed), and `tests/test_build_report_pdf.py` checks that CITATION.cff's
   `keywords` repeat it. Confirm or change it there and in CITATION.cff.

## 4 History check

Run on 2026-10-02 over everything reachable from any ref (branches `master` at
`6e19f40` and `official-protocol` at `3e6b770`; no tags, no stash, no remote, no
merge commits): 19 commits, 239 paths ever tracked, 322 distinct blobs. Values
were never printed; matches were inspected masked.

| check | result |
|---|---|
| paths ever tracked: `.env*`, `kaggle.json`, `third_party/`, `data/`, `dist/`, `*.parquet`, archives (`.zip`, `.tar`, `.gz`, `.xz`, `.bz2`, `.7z`), `submission/prior.json`, key and certificate files, names containing credential, secret or token, `.DS_Store`, caches, `.pyc`, pickles, arrays and tensors, local tool state | none. Every path ever tracked lies under `docs/`, `experiments/`, `kaggle/`, `paiec/`, `results/`, `submission/` (`model.py`, `labeling.py`), `tests/`, `tools/`, or is one of the five root files |
| secret-shaped strings in every blob and every commit message: Hugging Face, GitHub, OpenAI, Anthropic, AWS, Slack and Google keys, private-key blocks, JWTs, `kaggle.json` content, bearer headers | none |
| generic assignments (`token=`, `password=`, `api_key=`, `secret=` followed by 16 or more characters) | one: the Dataverse preview link in `results/inventory.csv` (section 3.4) |
| largest blobs | `results/level_calibration.json` 19.5 MB, `results/testlike_check.json` 10.0 MB, `results/hier_eval.json` 5.7 MB; 8 blobs over 1 MB, none over 50 MB (GitHub warns at 50 MB and refuses 100 MB); about 21 MB of objects in all |
| measurement-db item text (48-character windows of every item's content and grading criterion, at several offsets, against every blob) | `results/rate2_truth.json` and `results/ctrl_truth.json` (section 3.1). Elsewhere only instruction lines shared by hundreds of matharena items ("Please reason step by step, ...", the proof and Kangaroo templates) in the Kaggle kit, `results/strong_llm_eval.json` and three tests: boilerplate, not item content |
| e-mail addresses in tracked content | the organisers' public contact in `docs/report/report_requirements.md`; the addresses inside quoted item text (section 3.1). Commit metadata: section 3.2 |
| notebook outputs | `kaggle/strong_probe/paiec_strong_probe.ipynb` has none |

Re-run it on the final commit, just before pushing (prints paths or counts only):

    # paths that must never have been committed
    git rev-list --all | while read c; do git ls-tree -r --name-only "$c"; done | sort -u \
      | grep -E '(^|/)\.env($|\.)|kaggle\.json|^third_party/|^data/|^dist/|\.parquet$|\.zip$|^submission/prior\.json$' \
      || echo "no forbidden paths"
    # key-shaped strings anywhere in history: a count, never the match (expect 0;
    # "github[_]pat_" is written so that this file does not match itself)
    git log --all -p | grep -c -E 'hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github[_]pat_|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-|AIza[0-9A-Za-z_-]{35}|BEGIN [A-Z ]*PRIVATE KEY'
    # generic assignments (expect 1: the Dataverse link of section 3.4, unless removed)
    git log --all -p | grep -c -i -E '(token|password|passwd|api_?key|secret)[[:space:]]*[=:][[:space:]]*["'\'']?[A-Za-z0-9_-]{16,}'
    # blobs over 10 MB (expect the two results files above)
    git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
      | awk '$1=="blob" && $2>10000000 {print $2, $3}'

These were run as written on 2026-10-02 and gave: no forbidden paths, 0, 1, and
the two files.

Tests: on 2026-10-03, `python -m pytest -q` in the working tree (`3e6b770` plus
the uncommitted release work, this revision's included) with the data and the
baseline repository present: 717 passed, none skipped, 1 warning, in 3 min 58 s
(717 collected tests over 24 test files; `git status` was the same before and
after). On 2026-10-02, in a `git archive` export of `3e6b770` without `data/`,
`third_party/` or git history: 687 passed, 12 skipped (16 min 37 s), so the
README's claim that the tests need neither holds. The release review's clean
clone (the release files copied in, no `data/` or `third_party/`) gave 704
passed, 10 skipped in 9 min 59 s, before this revision added three tests.

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
| `tests/test_kaggle_probe.py` | 1 | a scratch CSV, compared only if it exists |
| `docs/report/provenance_facts.md` | 1 | the interpreter under the home directory (`~/anaconda3`) |

Do not rewrite them. The results files' sha256 digests are recorded elsewhere:
in `docs/report/fig/manifest.json`, which `tests/test_report_figures.py` and
`python tools/report_figures.py --check` verify, and in `results/formative_run3.json`,
`results/gate_and_ci.json` and `docs/plans/p1a_regime_sensitivity.md`. Editing a
results file breaks those checks and the provenance chain of report App H, and
the old bytes would stay in the history anyway. The docs and the test are
records of how the work was done and are left as they are too. (The paths
inside quoted item text belong to section 3.1.)

## 6 The tag

- **Name:** `v1.0.0` (CITATION.cff's `version` is `1.0.0`). Annotated.
- **Commit:** the commit that holds the final release files, that is, the head of
  `official-protocol` once this checklist's files (LICENSE, README.md,
  CITATION.cff, .gitignore, CLAUDE.md, this file), the report's PDF tooling and
  the other release work are committed, and, if section 7 step 2 is followed,
  after the URL is filled in.
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
- **One tag only.** An alternative, `v1.0-submission` at `3e6b770`, was
  proposed during the release work and is dropped: the archive code is the same
  at both commits, and only the release head carries the license, README and
  CITATION.cff. The report's App I says "Tag the commit the final archive is
  built from", which reads as `4d2cc4f`; it should say to tag the release
  commit, whose archive code equals `4d2cc4f`'s (the report's text, for its
  owner to change).
- **Check before tagging** (prints the line only if nothing changed):

      git diff --quiet 4d2cc4f HEAD -- paiec/__init__.py paiec/hier.py paiec/prior.py \
        paiec/predict.py paiec/fitting.py paiec/irt.py paiec/subjects.py paiec/mcq.py \
        submission tools/build_submission.py requirements.txt \
        && echo "archive code unchanged since 4d2cc4f"

- At the tagged commit, recount the tests quoted in report App A.4 and App G.2
  (report App I): see section 4 for the counts of 2026-10-03.
- If the report or results change after the tag is pushed, tag the new commit
  `v1.0.1` rather than moving `v1.0.0`.

## 7 Publish

Before: section 3 settled (3.2 before any release commit is made) and
`python -m pytest -q` green. After step 0, `git status` on `official-protocol`
shows nothing but ignored files; re-run section 4's commands then, on the final
commit. The local default branch is `master` (at `6e19f40`), an ancestor of
`official-protocol` (17 commits behind), so `main` is a fast-forward.

    # 0. The release commit itself, with the identity of section 3.2. If the
    #    address must stay private, use GitHub's no-reply address (from GitHub's
    #    e-mail settings; do not guess it) for this and every later commit:
    git -c user.email=<no-reply address> commit ...
    #    Before tagging, also resolve or remove the draft's body TODO(...) markers
    #    (10 on 2026-10-03: App G.4's table, App G.6, and App I with the review
    #    map, which would print in the OpenReview PDF), and recount the tests of
    #    App A.4 and App G.2 (section 6). These are the report's text.

    # 1. Create the empty public repository: no README, license or .gitignore
    #    (they come from here). With the GitHub CLI, or on github.com.
    gh repo create doctawho42/PAIEC --public \
      --description "PAIEC (NeurIPS 2026): a hierarchical response model for a benchmark-level cold start"

    # 2. The URL (https://github.com/doctawho42/PAIEC) and the affiliation are
    #    already filled in (section 1). Before tagging:
    python -m pytest -q
    python tools/build_report_pdf.py --strict   # exit 2 while a body TODO(...) marker is
                                                # left; the PDF for OpenReview needs exit 0

    # 3. Remote, branches
    git remote add origin https://github.com/doctawho42/PAIEC.git
    git merge-base --is-ancestor master official-protocol && echo "fast-forward OK"
    git branch -m master main
    git branch -f main official-protocol     # fast-forward; run while official-protocol is checked out
    git push -u origin main                  # main becomes the default branch
    git push origin official-protocol        # the report names this branch

    # 4. Tag (section 6)
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
> its upstream licences; `results/rate2_truth.json` and `results/ctrl_truth.json`
> quote shortened item text, which is not under the MIT License (the README
> lists each benchmark's licence). The organisers' baseline repository has no
> license and is not included; clone it at commit 82d330dd.

(Drop the sentence on the two results files if section 3.1 removes them.) Do
not attach the submission archive or the data to the release; the report PDF
may be attached once it is final.

After pushing, check on GitHub that the default branch is `main`, that the
license is detected as MIT, and that "Cite this repository" appears (it reads
CITATION.cff). Check that secret scanning and push protection are on (the
repository's code security settings); both are free for public repositories.

## 8 After publishing

The repository URL https://github.com/doctawho42/PAIEC is already in
`README.md` (the clone command), `CITATION.cff` (`repository-code`, twice) and
the report's title block (`docs/report/draft.md`, "Code:"). The placeholder
`"OWNER" + "/REPO"` survives only, by design, in `tools/build_report_pdf.py`
(`PLACEHOLDERS`, what `--strict` flags) and in a synthetic title block of
`tests/test_build_report_pdf.py`; do not replace either.

Also, after publishing:

- `docs/report/draft.md`: the "Code:" line may point at the tag,
  `https://github.com/doctawho42/PAIEC/tree/v1.0.0`. Rebuild the PDF with
  `python tools/build_report_pdf.py --strict` from the tagged commit, or from a
  later one if those lines change after tagging.
- The OpenReview form's code link and keywords (the draft's "Keywords:" line,
  section 3.5), and the e-mail to the organisers naming the selected Codabench
  submission (`docs/report/report_requirements.md`, items 4 and 7).
