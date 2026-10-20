# Revision history of the technical report

This file holds the report's revision history and its commits. Up to report v1.0.1 it was the report's App H.4; from report v2 on it is kept in the repository, outside the PDF (`docs/plans/p2_final_plan.md` §1, decision 3). A seven-character commit id from before the history rewrite of 2026-10-03 resolves through a lightweight tag of the same name (`docs/release.md`, section 9).

## Report v1 (2026-09-28 to 2026-10-04)

Draft v1 is dated 2026-09-28. We revised it on 2026-10-01 (the 14B's entropy job, formative run 3 and the RS study) and on 2026-10-02 for the remaining P1 items of the internal review (its second priority level: what a sound, complete analysis needs). Those items were the missing baselines, the pooling decomposition, the gate's tightening and intervals, the single-subject breakout, the row export and the figures. On 2026-10-02 we also restructured the report for the review's P2 items, which concern clarity, presentation and length. We cut the main text to sections 1 to 7, moved detail to the appendices, and used one name per model and one SE format. We kept the pre-trim text as `docs/report/draft_v1_long.md`. A verification pass the same day restored caveats lost in the trim, labelled the headline table in plain words, completed the references and recorded the submission requirements with their sources (`docs/report/report_requirements.md`). The pass changed no reported result.

We wrote the report against branch `official-protocol`. Every script, results file and test it cites is committed there, and so are the report's own files. We committed the P0 revision (the answers to the review's items blocking submission) in `00bdf04`, and the Kaggle record in the commit after it. The entropy job's kit and rule are in `78e303e`, the entropy job's output and formative run 3 in `380fecf`, the RS study in `7fb7450` and the P1 records in `dd3e372`. The submission archive's code is that of `4d2cc4f`. We tagged the release commit `v1.0.0` and pushed it on 2026-10-03 (`docs/release.md`, sections 6 and 7).

On 2026-10-04 we revised the report once more, in a commit after the release commit (`d64ff8b`, tagged `v1.0.1`). The author decided that the row archives are not hosted and that run 3's headline score stays unrecorded, and left open how the raw-versus-graded judgement of App G.6 was made. The author also confirmed the statements that App G.6 marks as confirmed. We added the wall-time notes of App G.4, corrected the device in App G.1 and the archive count and offline-bank command in App G.4, and restructured App I. This revision changed no reported result.

## Report v2 (from 2026-10-06)

Plan P2 (`docs/plans/p2_final_plan.md`), written and committed in `6a629cb` on 2026-10-06 before any of its checks ran, rebuilt the report around the hidden formative runs (its R1 to R6) and fixed the gates of the checks S1 to S4. Its dated amendments, and the commits that carry the work:

- Amendment 1 (2026-10-06): S1 closed and S3 found no failure, so the code was frozen on archive-3; the hidden-runs analysis was ported to `experiments/hidden_runs.py`, and S4 was prepared (`6915f98`).
- Amendment 2 (2026-10-06, `a5ef016`): E1 declared before it ran, report v2's framing and title, and the reworded statement on run 3. S4's results followed in `aec3b2a`.
- Amendment 3 (2026-10-07, `31007b4`): the id-use wording of the run-3 statement, confirmed by the author.
- 2026-10-07: the P2 data scripts (E1, the factorial, the R5 ports, the R6 pins, the hidden-runs keys) in `9f73fd6`; the R5 results as scored on 2026-10-06 in `d6df0bb`; the P2 data results, re-run from the committed scripts, in `022ae32`; `tools/fetch_pool.py` in `a725df9`, with an option added in `2bfcd17` (2026-10-11).
- Amendment 4 (`3d38232`, 2026-10-11 UTC), with 4a (`d48f66b`), 4b (`fe2a7f4`) and 4c (`72cc96b`): the expanded public pool and a rule, committed and pushed in advance, for refitting hier's priors on it. The lock was committed in `831ef7d` (2026-10-13 UTC) and the selection in `365664c` (2026-10-14 UTC): no candidate was carried, and archive-3 stays the final submission. `docs/findings.md` gained the P2 sections and amendment 4's negative result in `e9b1010`.

Report v2 restructures the text. The main text is cut to at most 9,000 words (plan P2 §1, decision 3) and leads with the root-free reading of the formative tables (§3, §6.1, §6.2), with the submitted predictor in §5 and the pool evaluation in §7. Report v1's main-text detail moved to the appendices, and the new appendices B.7, B.8, C.5, C.6 and D.5 to D.10 hold the P2 checks, the level choice, the pool evaluation and the hidden-run analysis in full. App H.4, this history, and App I.2, the review map (`docs/report/review_map.md`), left the PDF and stay in the repository, and the main text's results keys moved to `docs/report/sources_main.md`.

We committed report v2's text in `2af8fcd` on 2026-10-14. Plan P2 §7 scheduled its release commit, tagged `v1.1.0`, for 2026-10-26 to 2026-10-27; on 2026-10-19 the author asked to release report v2 without waiting for those dates. The release commit, the commit after `2af8fcd`, is tagged `v1.1.0` (`docs/release.md`, section 10). It changes text only: CITATION.cff (report v2's title, version 1.1.0 and the abstract), README.md (the report's title, the list of what is here, the headline results from report v2's main text, the repository map, the data revision, the archive statement, the disclosures and the section references), `docs/release.md` (the header, a row or a note in each of sections 2 to 6, and section 10), this file, and the draft, which gains the v1.1.0 test counts (App A.4, App G.2) and states what changed in the build script since archive-3 (App C.6, App G.3). It changes no code, test or results file, and no reported result. Archive-3 stays the final submission: its members other than `prior.json` are unchanged since `4d2cc4f`, `tools/build_submission.py` has changed only by the lines that `831ef7d` added for its `--pool` path, and on 2026-10-19 the script's default path rebuilt archive-3 byte for byte.

## Report v2 in the NeurIPS 2026 template (v1.2.0, 2026-10-20)

On 2026-10-20 we read the competition page again. It asks for the NeurIPS template and about 4 to 8 pages of main text, references and appendices excluded (`docs/report/report_requirements.md`). Report v2 keeps its title, its results and its appendices' content, and is now set in that template. The commit `e29853f` (2026-10-20) did most of the work:

- **The build.** `tools/build_report_pdf.py` loads the official style file `neurips_2026.sty` (revision 2026-01-29) with the options `sglblindworkshop` and `final`. The style carries no license, so it is kept out of git and checked by its sha256. `--strict` holds the main text to 8 pages.
- **The cut.** The main text went from 15 template pages to 8.00, under the author's decisions of that day, and nothing it dropped was lost:
  - §1 and §2 lost their subsection headings, and §8 became "Limitations".
  - Tables 2 and 3 became one table, Table 2.
  - §6.4's Table 4 became Table F.1a (App F.1), and §4.3's gate figure, Figure 1, became Figure C1 (App C.4).
  - §7's selection table, Table 5, became Table C.6a. App C.6 discloses the move, which departs from the pool design's plan of one paragraph and one table in the main text.
  - The two author-confirmed C1m sentences of design section 11 moved from §5.3 to App G.6, which quotes them word for word but for their pointer to that appendix, cut and marked [...].
  - Each block that left the main text took its source keys with it (`docs/report/sources_main.md`).
- **The rules.** `docs/report/style_guide.md` replaced the 9,000-word rule of plan P2 §1, decision 3, with the page rule.

The release commit, after `e29853f`, is tagged `v1.2.0` (`docs/release.md`, section 11), under the author's decisions of 2026-10-20:

- **(a) Acknowledgments.** It adds the unnumbered section "Acknowledgments and Disclosure of Funding" before the References: "This work received no specific funding. The author declares no competing interests." As the NeurIPS instructions do not count acknowledgments as content pages, the build ends the main text at that heading.
- **(b) Approval.** The author approved the move of the C1m sentences to App G.6 and the layout of §1 and §2 (no subsection headings, Table 1 moved).
- **(c) Figure type.** Figures 1 and 2 have no type under 6 pt at print size, sub- and superscripts included, with the same data and size. Their labels write B0, Bk, B31 and log2(1+B) on the line, and Figure 2's unit as "in 0.001", because matplotlib sets a sub- or superscript at 0.7 of its label's size (`docs/report/figures.md`, "Type size").
- **(d) The release.** It adds the v1.2.0 test counts to App A.4 and App G.2 (809 test functions over 35 files, 1,113 collected tests) and the style file to App G.4's row for the PDF, ends the main text at the acknowledgments heading in `docs/report/style_guide.md` (item 10) and `docs/report/sources_main.md`, and updates CITATION.cff (version 1.2.0), README.md and `docs/release.md`.

It changes no file under `paiec/`, `submission/`, `experiments/` or `results/`, and no reported result. Archive-3 stays the final submission. Its archive code is unchanged since `v1.1.0`, and its members other than `prior.json` since `4d2cc4f`.
