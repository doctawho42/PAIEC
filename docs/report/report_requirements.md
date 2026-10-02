# Technical report: what the competition requires

Read 2026-10-02 from the official pages listed under Sources. Quotes are short and verbatim. Everything else is paraphrase. Where a page says nothing, this file says so; it does not guess.

## Summary

| item | requirement | source |
|---|---|---|
| deadline | "30 October 2026 at 23:59 Anywhere on Earth (AoE)", for the codebase and the report together. This is 2026-10-31 11:59 UTC, which matches the OpenReview due date and the Codabench phase end | competition page, Participation; OpenReview `duedate`; Codabench phase 29785 `end` |
| where | OpenReview, venue `NeurIPS.cc/2026/Workshop/PAIEC`: "submit the technical report, including links to these released materials, through OpenReview" | competition page, Presentation consideration |
| also required by the deadline | the codebase, and an e-mail to the organisers to "identify their selected Codabench submission" | competition page, Submissions |
| content | "clearly describe the data, submitted approach, implementation and experimental setup, results, and limitations", "with instructions sufficient to rerun the method" | competition page, Presentation consideration |
| releases to link | "release of the code used to develop and run the submission", "along with any training data that differ from measurement-db" | same |
| disclosures | training sources ("Disclose all training sources in the technical report."); pretrained models ("must disclose them in the technical report"); LLM coding assistants, a request, not a rule ("We kindly ask participants to disclose any such use"); shared contributions from other teams ("acknowledged in the technical report") | competition page, Disclosure, Conduct, Q&A |
| file | one PDF, at most 50 MB; no separate supplementary field, so appendices go in the same PDF and code goes by link | OpenReview submission form |
| form fields | title (at most 250 characters); authors, each with an OpenReview profile ("All authors must have an OpenReview profile prior to submitting a paper."); keywords (required); TL;DR (optional, at most 250 characters); abstract (required, at most 5,000 characters); PDF; two consent boxes, author-e-mail sharing with the programme chairs and public release on acceptance | OpenReview submission form |
| page or word limit | none stated | competition page, OpenReview form, Codabench |
| template or style file | none stated | same |
| required section headings | none stated; only the content list above | same |
| anonymity | not stated, and not blind in practice: the form takes named author profiles, its author field is not hidden, and acceptance means "accepted submissions, along with their author names, will be released to the public" | OpenReview form; competition page silent |
| how it is assessed | by the organisers, "criteria adapted from the NeurIPS 2026 Evaluations & Datasets reviewing guidelines"; emphasis on "technical soundness, clarity, reproducibility, and the quality of experimental analysis"; "Methodological novelty will be a secondary consideration."; "We do not currently plan to conduct formal peer review of technical reports." | competition page, Summative evaluation |
| what it decides | the summative evaluation combines Brier ALC on a common hidden test subset with manual grading of the codebase and the report; the organisers will "review and rerun the released method". The report's weight in presentation selection: "Technical report quality is an important factor in selection" for oral and poster slots (five orals of 60 minutes in total, then a 60-minute poster session). Selection also weighs predictive performance, scientific rigour and contributions to data curation and the community | competition page, Overview and Summative evaluation; launch post |
| session | in person, Sydney, Friday 11 December 2026, 08:00 to 10:15 Sydney time | competition page, Overview |

## What this means for the draft

1. The abstract does not fit the form. At dd3e372 the draft's abstract is 5,236 characters (882 words, Markdown included), and OpenReview takes at most 5,000. The OpenReview abstract needs cutting regardless of what the PDF's abstract says. A NeurIPS-style abstract (about 150 to 250 words) would also match how the E&D criteria read a paper (below).
2. Format is free, length is not policed, but clarity is graded. No page limit or template is stated. The report is graded on criteria "adapted from" the NeurIPS 2026 Evaluations & Datasets guidelines. That track's own rules, which do not apply here and are given only as context, are a 9-page main text in the official LaTeX style with double-blind review. A short main text with long appendices, in the NeurIPS style, is the safe reading. The draft now runs about 31,700 words in Markdown; the review's P2.19 trim is the lever. The PDF must be built from the Markdown before submission.
3. Names, not anonymity. Fill `Authors: TODO(team)`; every author needs an OpenReview profile before the form can be submitted. Do not anonymise. A double-blind version is neither asked for nor possible on this form.
4. The code link is mandatory for presentation consideration. `Code: TODO(link to the public release)` must become a public link before 2026-10-30 AoE. The report must link every released item. Training data other than measurement-db: §8.6 says none was used. The row export (data/release_rows, §8.4) and the LLM features (data/features/) are derived research data. Say in the report whether and where they are released, since the organisers will "review and rerun the released method".
5. Rerunnability is graded explicitly. §8 already holds the rebuild and rerun commands. Keep them in the PDF (an appendix is fine), with wall times.
6. The disclosures the page asks for are present in §8.6 and §8.2: training data (measurement-db only); pretrained models (Qwen3-Embedding-0.6B, Qwen3-4B-Instruct-2507, Qwen3-14B-AWQ, research only); the LLM coding assistant (Claude Code); and every formative-feedback use. One conduct point the team should read against the rules: §8.6's inventory scan cloned repositories of the 161-benchmark inventory, which includes test-pool candidates. The rules say participants "must use benchmarks in the public training pool for competition-specific training and curation", and must not "extract or share private test data". The draft discloses the scan in full. Whether to state outright that nothing from it entered training or the submission is for the team to decide.
7. E-mail the organisers (aimslab@cs.stanford.edu) with the selected Codabench submission. This step is separate from OpenReview, and the page sets it for the same deadline.
8. Timing. The OpenReview form opened 2026-09-15. Its due date is 2026-10-31 11:59 UTC; the form's expiry is 30 minutes later, but do not plan on that. Codabench's competition phase ends at the same instant. 28 days remain from the date read.

## What is not stated anywhere I could read

Page or word limit, template, font or margins, required section headings, a separate supplementary upload, anonymity rules, a camera-ready stage, a poster or slide format, and how the report is weighted against Brier ALC in the summative score. The organisers' contact is aimslab@cs.stanford.edu, and they offer office hours by appointment. Ask them if any of these matters.

## Sources (all read 2026-10-02)

- Competition page: https://aimslab.stanford.edu/competition (sections Overview, Participation, Evaluation → Summative evaluation, Resources, Q&A, Organizers). The "Submit on Codabench" link (/competition/submit) redirects to Codabench. Codabench's terms point to .../competition#rules, an anchor the page does not have; the rules are under Participation.
- Launch post, 2026-08-15: https://aimslab.stanford.edu/news/predictive-evaluation-competition-opens ("Technical report quality is an important factor in both.").
- OpenReview venue: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC. Read through the public API: api2.openreview.net/groups?id=NeurIPS.cc/2026/Workshop/PAIEC (public_submissions false, instructions empty, location Sydney) and api2.openreview.net/invitations?id=NeurIPS.cc/2026/Workshop/PAIEC/-/Submission (fields, limits, duedate 1793447940000 = 2026-10-31T11:59:00Z, expdate 2026-10-31T12:29:00Z).
- Codabench: https://www.codabench.org/competitions/17828/, read through www.codabench.org/api/competitions/17828/. Its overview defers to the competition page; phase "Competition" (id 29785) runs 2026-08-15T00:00Z to 2026-10-31T11:59Z. It says nothing about the report.
- NeurIPS 2026 E&D reviewer guidelines: https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines. The four criteria are Quality, Clarity ("detailed enough to enable reproducibility"), Significance and Originality. Track rules: a 9-page main text, the official LaTeX style, double-blind by default. Read through a fetch tool, because direct download failed on DNS from this machine; treat the wording as the tool reported it.
- Baseline repository README: https://github.com/aims-foundations/paiec_baseline. Nothing about the report.
