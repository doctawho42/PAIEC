# Editing plan for the technical report (review P2: items 18, 19, 22; m1 to m7, m9)

This is a plan for the editor. It is not a report and changes nothing in the draft. It was written against `docs/report/draft.md` at commit `dd3e372` (1,172 lines), `docs/report/review_v0.md`, `docs/report/figures.md` and `docs/report/fig/`. **Line numbers below are lines of `draft.md` at `dd3e372`**; re-anchor them with `git show dd3e372:docs/report/draft.md` once the draft starts moving.

Rules this plan keeps, and the editor must keep:

- **No number changes.** Every number stays as written. The plan only moves, cuts or rewords text around numbers. Prose may keep a rounded form only where the draft already uses it (for example 0.042 for -0.0419). The plan creates no new roundings and flips no signs.
- **Nothing true is deleted without a trace.** A cut sentence goes to an appendix of the report, or is dropped with a pointer to the `docs/findings.md` section that already holds it. Section 5 of this plan says which full SE triples were checked as present in findings. Any other cut must be checked the same way first (`grep -F` the number in `docs/findings.md`).
- **The figure check.** `tools/report_figures.py --check` and `tests/test_report_figures.py` do not read `figures.md`. Its only mention is in the script's docstring, line 23. Renumbering figures in the draft and editing the placement table in `figures.md` therefore cannot break the check. Run it anyway (section 9). Renaming legend labels is a figure-lane change; see 3.9.

---

## 0 The plan in brief

1. **Shrink the main text.** The abstract goes from 882 to at most 250 words. The prose of sections 1 to 7 goes from about 17,200 words to about 7,200 (ceiling 7,500), plus tables and figures. Nothing is dropped: detail moves to nine appendices, A to I (3.8).
2. **Tell the shipped configuration's story once, in a new §5.4** (selection, audit, confirmation, the pooled reading of run 2, and the test at the feedback's readings). §3.4 keeps only the method. Every other mention becomes a back-reference (section 2).
3. **Move the acceptance harness from §4.5 to §6.1**, so that §6 opens with its instrument. §5.5 (ablations), §5.4 (per-budget table) and almost all of §6.1 to §6.8 go to appendices. In the main text each study keeps one paragraph (3.6).
4. **Use one name per model and one table of names** (§1.5): LegacyP, hier-fit, hier-ship, hier-rec, Smooth-cal, 1PL-ship, and so on. Take review tags (P1a, P1.9, W3, Q5, D1, "commit D") out of the main text (section 4).
5. **Use one SE format**, defined once in §4.3. In the main text, ± and a bracketed SE after a difference mean the pair-cluster SE. The full run / cluster / stratified triple appears only in the confirmation table of §5.4 and in the appendices (section 5).
6. **Apply the wording fixes** for m3, m4 (the §2.4 scope sentence), m5 (§3.2, §3.3; the §3.4 item is already gone) and m6, quoted with their replacements (section 6).
7. **Give each repeated number one home.** Numbers repeated in three or more places each get one home and at most two short restatements (section 7).

---

## 1 Target structure and word budgets

Counts are words of prose. "Tables" and "captions" are counted apart and do not count against the budget. The counts are by `/private/tmp/claude-501/-Users-nikitapolomosnov-PycharmProjects-PAIEC/9df8ecfa-42c0-4508-919c-f794f1770909/scratchpad/p2/plan/wc.py`: lines beginning with `|` are table, `*Figure`/`![` are captions, and indented lines are code.

### 1.1 Main text

| new section | holds | from (old §, lines) | prose now | target |
|---|---|---|---|---|
| front matter | title; authors; code link; one sentence on provenance pointing to App H | 1-9 (306 words, incl. the "Names" paragraph) | 306 | 60 |
| Abstract | the candidate in 8.1 | 13-21 | 882 | ≤ 250 |
| 1.1 The task | as now; the platform-score reproduction sentence shortened | 27-33 | 183 | 170 |
| 1.2 What makes it hard | four bold items, each 2 to 3 sentences | 35-43 | 384 | 280 |
| 1.3 Contributions | the four rewritten in 8.2 | 45-50 | 225 | 200 |
| 1.4 Related work (new) | one-line pointers now spread over §6's *Literature* paragraphs; the literature lane's m10 additions go here | 751, 769, 792, 815, 835, 845, 865 (condensed) | 0 | 200 |
| 1.5 Names and notation (new) | Tables 4.1 and 4.2 of this plan | 9 (the "Names" paragraph) | 0 | 30 + two tables |
| 1.6 Headline numbers | 12-row table (8.4), a two-line note | 52-79 | 70 | 40 |
| 2.1 Data | first paragraph; data table; five one-line properties (m3 fix); two sentences on the hidden pool | 85-107 | 498 | 230 |
| 2.2 The protocol, verified | the six properties, plus the three verification facts of old §2.5 in one closing sentence | 109-118, 172-177 | 208 + 158 | 230 |
| 2.3 What our first replica got wrong | one sentence, three bullets, the pooling table and its note, one closing sentence (3.2) | 120-153 | 663 | 280 |
| 2.4 Unknowns kept as parameters | split scope in two sentences, plus the new scope sentence (6.2) | 155-170 | 294 | 120 |
| 3.1 Model | as now | 187-205 | 106 | 110 |
| 3.2 Inference | the first two paragraphs, with the m5 caveat; "Cost" goes to App B.1 | 207-213 | 360 | 220 |
| 3.3 Offline prior | as now, with the m5 caveat | 215-232 | 255 | 200 |
| 3.4 Calibrating the level prior | method only: signal, regime, candidates, selection rule and guard, conduct rule, then a pointer to §5.4 | 234-249 | 958 | 330 |
| 3.5 Run-time engineering | two sentences; the build checks go to App B.2 | 270-279 | 192 | 100 |
| 4.1 Runs | regime table (renamed, 4.2), four bullets on how test-like runs are built, two sentences on fit, one on thinness, the m6 sentence | 285-307 | 431 | 260 |
| 4.2 Leave-one-parent-out | as now | 309-317 | 112 | 90 |
| 4.3 Statistics | the SE convention of section 5 | 319-327 | 176 | 170 |
| 4.4 Selection discipline | halves; guard; the four gate conditions; one sentence pointing to App C.3 for when each rule was fixed | 329-340 | 550 | 120 |
| 5.1 Formative feedback | run table; run 1 at B0; runs 2 and 3 placed (table, Figure 1); "different draws"; leaderboard | 403-464 (part) | 1,712 | 420 |
| 5.2 Public runs: baselines | baselines table; three sentences | 466-487 (part) | 160 | 120 |
| 5.3 Moving the level | the unconstrained table; three bullets; Figure 2 | 489-515 | (in 2,872) | 250 |
| 5.4 The shipped configuration: chosen, audited, confirmed, tested | the one home of m2 (section 2) | 251-268, 447-462, 517-614, A.3 | (in 958 + 1,712 + 2,872) | 800 |
| 5.5 Calibration against modelling | the 1PL; two tables; four bullets; BLE in one sentence; caveat | 616-644, 507-511, 607-611 | (in 2,872) | 330 |
| 5.6 Where it loses, and what it costs | single-subject benchmark; platform composition; latency in one sentence | 646-652, 881, 701-703 | 342 + part | 180 |
| 6.1 The acceptance harness and its gate | from old §4.5: formula, slopes, merged gate table, four bullets, Figure 5 | 342-397 | 817 | 330 |
| 6.2 What was tried | the footing sentence, the compact table (3.6), Figure 6 | 707-735 | 158 | 90 |
| 6.3 Study by study | one run-in paragraph per family (nine), Figure 7 | 737-865 | 3,235 | 580 |
| 6.4 Why these fail | old §6.9, with the numbers duplicated from §6.1 replaced by back-references | 867-870 | 357 | 250 |
| 7 Limitations | ten bullets, each at most three sentences, back-referenced (3.7) | 874-888 | 1,430 | 550 |
| Code, data and conduct (unnumbered) | the statement in 8.3 | 892-1001 (condensed) | (1,980 in §8) | 190 |
| **main total** | | | **about 17,200** (sections 1 to 7) | **about 7,200** (ceiling 7,500) |

Section totals, now → target: §1 862 → 920; §2 1,821 → 860; §3 1,871 → 960; §4 1,269 (without 4.5) → 640; §5 5,496 → 2,100; §6 3,750 + 817 → 1,250; §7 1,430 → 550; conduct statement → 190.

### 1.2 Appendices (new lettering)

| appendix | holds | from |
|---|---|---|
| A Data, protocol and the legacy replica | A.1 data sources, the inventory classification and its validation (2.1 "Sources", 97; 107). A.2 the legacy replica's five differences, the order effect, the "Why the lever vanished" method, pooling by company, and the platform-composition reweighting (124-128, 130, 138, 147-153, 881 in part, 1.2's legacy sentence at 37). A.3 split-scope numbers on dense runs and other open points (157-170). A.4 verification and Figure A1 (174-181) | §1.2, §2, §7 |
| B Model, inference and run-time details | B.1 solver cost and floored fits (213). B.2 run-time engineering and the build checks (270-279). B.3 ablations of hier-fit and Figure B1 (670-697). B.4 the corrected floor and the floored-fit fix (699, 668). B.5 cost in full (701-703). B.6 library defects (883-886) | §3, §5.4, §5.5, §5.6, §7 |
| C Regimes, rules and when they were fixed | C.1 regime catalogue and realised levels: one table, built from 287-307 and 293-294, see 3.4. C.2 the decision rules: the level selection (247), the gate (333-337), the regime-sensitivity rule's five conditions (591-597), the reading rule (447). C.3 when each rule was fixed: the long paragraph at 339, "The reading was run four times" (458), the precision deviation (268, 599). C.4 the gate under mix/whole (379) and the full gate table with the mix/whole and public columns (355-366) | §3.4, §4, §5 |
| D Formative runs | D.1 scores, archives and the recurring pair (413, 415), old A.1's tables and ECE line (1030-1075). D.2 placement by budget: the full placement table (427-433), 435 and Figure D1 (`formative_runs`, 441-443). D.3 matched estimates (445). D.4 the pooled reading of runs 1 and 2: the rule, the table, the LEVEL-implied sds, BIC, and the B0-excess split (447-460); the continuity-corrected -1.51 / -0.71 (238) | §5.1, old A.1 |
| E Calibration details | E.1 hier-fit against LegacyP on public runs by setting and by company, and strict run-LOBO (478-487; old A.2 1077-1086; old A.5 1114-1116). E.2 sensitivity to the regime (544-555). E.3 the empirical-Bayes level and Figure E1 (557-561). E.4 the baseline study's run provenance, paired SEs and the B0 attribute-prior bullet (616, 638-641). E.5 BLE (642, and the P1.9 deferral at 1129). E.6 the single-subject benchmark among the regime-sensitivity alternatives (650). E.7 hier-ship across regimes by budget (656-668) | §5 |
| F The studies of §6 | F.1 the full summary table: old §6 table (711-731), plus a "within-pair r" column from old 383-391 and the known-sign rows from old A.4. F.2 text maps and Figure F1 (`transfer`; 737-751). F.3 language-model judgements (753-773). F.4 the itemsig layer (775-786, 792). F.5 meta-learned heads (794-803) and the harness's acceptance reproduction (353). F.6 known-sign cues (805-815; old A.4 1102-1112). F.7 subject side (817-835). F.8 legacy-replica studies (837-845). F.9 closed probes and the 14B (847-865) | §6, old §4.5, old A.4 |
| G Reproducibility | old §8.1 to §8.6, verbatim, with cross-references updated (892-1001); test counts from 177 | §8, §2.5 |
| H Provenance | H.1 the provenance convention and the glossary of internal labels (7; 4.4 of this plan). H.2 which code produced each number: old App C (1148-1172), the "Which code" paragraph (668) and the base-ALC sentence (6.4). H.3 provenance gaps (887). H.4 the commit list and revision history (3, 5) | header, §5.4, §7, old App C |
| I Open items | old App B (1120-1144), with resolved TODOs removed by the team, plus a map from review items to sections (4.4) | old App B |

### 1.3 If the requirements lane finds a hard limit

If the limit is tighter, cut in this order, moving the text to the named appendix and leaving a one-line back-reference:

1. §6.3 run-in paragraphs → compact table only (App F).
2. §1.4 related work → 100 words.
3. §2.4 → App A.3, leaving the scope sentence in §2.2.
4. §5.6 → two clauses in §7.
5. §3.2 → its first sentence of each paragraph.
6. §5.5's share table → App E.4 (keep the four bullets).

If the limit is looser, restore in reverse order. Never restore the review tags.

---

## 2 The shipped configuration's story, told once (m2)

### 2.1 Where it is told now

| place | what it tells | lines |
|---|---|---|
| header "Names" | defines shipped and neighbouring configurations | 9 |
| Abstract | gains by regime; the reading left LEVEL unchanged | 17, 19 |
| §1.4 | 17 rows, 9 of them the shipped config | 58-77 |
| §3.4 | "What shipped", the audit's three reasons, "After run 2", "Tested at the feedback's reading (P1a)" | 251-268 |
| §4.4 | the lock evidence for the reading and P1a; deviation D1 | 339 |
| §5.1 | "What the shipped configuration was expected to score"; the pooled reading; "What the pooled reading supports is modest" | 445-462 |
| §5.3 | the "four observations"; the shipped table and bullets; "Against the neighbouring configuration"; sensitivity; EB; P1a; calibration against modelling; single-subject | 506-652 |
| §5.5 | "the price of the bet on hidden levels far from the public centre" | 693 |
| §7 | the tuned-regime bullet; "a moved level, not hier"; selection optimism; single-subject; provenance (D1) | 876-881, 887 |
| §8.6 | feedback uses | 991-993 |
| old A.3 | the guarded configurations | 1088-1100 |

### 2.2 Its one home: new §5.4 "The shipped configuration: chosen, audited, confirmed, tested" (about 800 words)

The steps run in time order. Each one keeps only the numbers listed; everything else moves as given.

**(a) Chosen (about 100 words).** Source: 251, first two sentences, and old A.3. The selection rule's argmax was hier-argmax (mu0 -3.5, attr_scale 0.25). The level calibration recommended hier-rec on the guard's tie-break: 0.0004 behind on selection, level on confirmation.

Table: old A.3, merged here (review m9), with three changes:

- Rename the rows (section 4).
- Move LegacyP's own ALCs in the public columns (0.2068, 0.2038) into a footnote, so that each public column holds only differences against LegacyP.
- Add one footnote line with the paired differences now at 542: hier-ship minus hier-rec is +0.0030 (0.0009) on the selection half and +0.0022 (0.0011) on the confirmation half, and -0.0023 (0.0005) and -0.0024 (0.0004) on R1-bf and R1-pu runs 0-99 (cluster SEs).

The paragraph at 542 is then deleted. Its run and stratified SEs go to App E.2, next to the sensitivity table. The neighbouring configuration's -0.0418 ± 0.0024 / 0.0047 / 0.0042 and its +0.0008 / +0.0007 are already in the A.3 row.

**(b) Audited (about 150 words).** Source: 251, last sentence, and 253-255, condensed to three sentences:

1. Two of run 1's pairs (one benchmark) are more plausibly high-rate: 70% and 80% of their nearest replica pairs sit above a rate of 0.5. Against LegacyP, the aggressive setting loses 0.014 on test-like pairs at rates 0.5 to 0.7, 0.070 at 0.7 and above, and 0.015 to 0.017 on public matharena. Re-read that way, run 1's nine pairs average -0.98 (sd 2.05); this is the one restatement of 421.
2. The two configurations sit on a plateau: the costs and gains in (a)'s footnote, with no worse measured worst case.
3. The confirmation half repeats 82.2% of the selection half's pairs and 99.5% of its clusters, so it measures redrawing, not new parents.

**(c) Confirmed (about 200 words).**

- Table: old §5.3's shipped table (519-528). It is the one main-text table that keeps the full run / cluster / stratified triple and the parent-level column (section 5). Keep its code note (530) as a one-line footnote, with "App H.2" for the rest.
- Figure 3 (`level_fix_budgets`).
- Four sentences from 532-536:
  - Held out, 0.040 to 0.042; hier-ship beats LegacyP on 296 of 300 runs; leaving one parent out gives -0.047 to -0.032.
  - The parent-level SE is about 2.7 times the cluster SE; matharena gains least (-0.014 ± 0.009), multi_swebench most (-0.058 ± 0.004).
  - B0 and B1 carry 0.029 of the 0.042.
  - On public runs it is level with LegacyP (0.0015 to 0.0018 better, within about one cluster SE), losing at B0 and gaining from B7 on. It loses on public matharena (+0.010, +0.009) and on the single-subject pair (§5.6). Against Smooth it gains 0.016 to 0.018 in the tuned regime and 0.009 on public runs.

**(d) After run 2: the pooled reading (about 120 words).** Source: 257, 447, 456, 460, 462.

- A rule, written into the results file before the reading was computed but after both runs' tables had been seen (App C.3), could change mu0 and sigma_mu only, at a bar of 2 SEs over 7 benchmarks.
- The 17 pairs' level is -0.65 (K = 15) and -0.78 (K = 40), SE 0.51, which is 1.24 and 1.00 SEs above the tuned regime's realised -1.29. No candidate; LEVEL stays.
- Run 3 is not read.
- The full reading (table, BIC, B0-excess split, the four runs of the read stage) is in App D.4.
- §5.1 keeps none of this, only a pointer.

**(e) Tested at the feedback's readings (about 230 words).** Source: 259-268, 563-614.

- Design, two sentences: ten alternatives against hier-ship on identical runs of five test-like RS regimes (§4.1) and two public weightings. Seed 11 is fresh, but it redraws the catalogue and the public pairs that chose the level.
- Rule, one sentence: gain at least 0.002 in READING and AUDIT with the upper 95% limit below zero; lose at most 0.002 in the other test-like regimes and 0.001 on public runs. The five conditions go to App C.2.
- The table (571-583, renamed, cluster SE) and Figure 4 (`regime_sensitivity`).
- Results, five sentences:
  - No candidate. This rests on the precision deviation, which the team accepted (App C.3).
  - The closest alternatives (hier-ship σ3.5 and both hier-EB) sit within 0.0006 in READING and AUDIT.
  - hier-rec wins only in TUNED (-0.0034 (0.0010)) and loses elsewhere by +0.0005 to +0.0033, more as the level rises, on matharena.
  - σ5 is worse by 0.0022 to 0.0031 in five of seven regimes. hier-EB on hier-ship's level gains only in TUNED (-0.0014) and costs +0.0006 and +0.0014 on public runs.
  - Against LegacyP, hier-ship gains 0.043 in TUNED, 0.031 in AUDIT, 0.026 in MIXTURE, 0.025 in READING and 0.020 in FLAT. The gain shrinks as the hidden level rises (the level audit's milder regimes and the older sensitivities are in App E.2).
- Close, one sentence (614): four parents carry these numbers, the parent-level SE is two to four times the cluster SE, and the rule detects gains of about 0.003 or more and is a coin flip at 0.002.

The bullets on Smooth-cal (607-610) and hier-nosubj (611) move to §5.5, not here.

### 2.3 Back-references (exact sentences)

- **End of §3.4:** "Which configuration shipped, and why, is told once, in §5.4: the rule's choice, the audit that replaced it, its confirmation on held-out runs, the reading of the next feedback, and a test at the feedback's level readings."
- **§5.1, after the run-1 paragraph:** "This is the observation behind the level calibration (§3.4); §5.4 says how it was used and what the later runs changed."
- **§1.6 note under the headline table:** "Every row is hier-ship unless named; its selection, audit, confirmation and test are in §5.4, and the split between calibration and modelling is in §5.5."
- **§7**, wherever a shipped-config number is restated: "(§5.4)" or "(§5.5)". No new sentences.
- **App B.3** (the moved ablations), replacing the end of 693: "The deliberately widened level prior costs about 0.002 on public runs; why it was kept is in §5.4."
- **App G.6** (feedback uses): keep as is, adding "(§5.4)" after "moved the shipped level between two guarded configurations" and after "changed nothing".

---

## 3 What moves where

The marks: **keep** (main, as is or lightly trimmed), **condense** (main, shortened to what is stated), **→ X** (moved to appendix X, verbatim apart from renames and cross-references), and **cut ↦ F§ "…"** (dropped; the same content is in that findings section, checked).

### 3.1 Front matter, abstract, §1

- 1 (title): keep.
- 3 (draft version and revision dates) → App H.4.
- 5 (authors, code link, commit list): keep the authors and code link; commit list → App H.4.
- 7 (provenance convention) → App H.1. The front matter keeps: "Every number traces to `docs/findings.md`, `docs/protocol.md` or a committed `results/` file through the script that produced it (App H)."
- 9 ("Names") → replaced by the §1.5 tables (section 4).
- 13-21 (abstract) → the candidate in 8.1. What the shorter abstract drops is all in §1.6, §5 and §6: the parent-level SEs, the mix/whole 0.044, the Smooth and EmpMean gains, the 1PL shares, the run-2/3 z values, the reading's 1.0 to 1.2 SEs, and the entropy correlations.
- 29-33 (§1.1): keep. In 33, shorten "recomputed that way … do not" to "the platform's run score is the unweighted mean of per-pair ALCs (App D.1)".
- 37 (§1.2, level): keep the first three sentences. Condense the rest to: "The first formative run read far below the public centre, and later readings are more central (§5.1, §5.4)." Move "Centring hier's level prior at 0 instead costs 0.0011 on public runs (§5.5)" to App B.3, where that row lives. "On the legacy replica, mixing it … cost 0.004 (legacy)" → App A.2.
- 39 (per-item structure): keep.
- 41 (runs are small): keep the first four sentences; the last sentence becomes "at that composition pooling is worth 0.0004 to 0.001 ALC to our model (§2.3)".
- 43: keep.
- 45-50 (§1.3) → the text in 8.2.
- 52-79 (§1.4) → §1.6 with the rows in 8.4. The cut rows go to their homes: run-2 and run-3 placement → §5.1 table; confirmation half → §5.4(c) table; nearest alternatives → §5.4(e); the 0.1658 row → App E.7; leaderboard → §5.1; the footnote's single-run sd → §5.1.

### 3.2 §2

- 87: keep.
- 89-95 (table): keep.
- 97 ("Sources: …") → table footnote "Sources and the count script: App A.1", with the text → App A.1.
- 99-105 (properties): keep five one-line bullets. The first takes the m3 replacement (6.1). The matharena bullet keeps "a third have no task text" and "336 Kangaroo items are five-option multiple choice with options in an image"; its sources → App A.1.
- 107 (inventory) → condense to two sentences: the hidden benchmarks come from the organisers' inventory of 161, assigned at random; keyword rules on titles class 67% as AI evaluations, of which 39% text QA, 30% images, 15% agents, 6.5% code, 4.6% video, 2.8% math, 2.8% audio; the public benchmarks are not a representative sample. Agreement figures, the AI-agent labeller and the scripts → App A.1.
- 111-118 (§2.2): keep. Append the three verification facts of 174-177 as one sentence: "We checked the replica against the client decision for decision, with platform-like argument copies (bit-identical per-pair Brier), and against the empirical mean's analytic ALC (residual -0.0005 ± 0.0006, App A.4)." Figure 1 (`empirical_mean`) → Figure A1. The test counts (177) → App G.
- 120-153 (§2.3) → condense to:
  - One opening sentence: the legacy replica, built before the client was public, scored all 221 pairs pair-major in one session, showed each target only its own labels and leaked identifiers (the five differences in full → App A.2).
  - Three bullets:
    1. **It pointed the research at pooling.** Evaluation order alone moved the assembled predictor from 0.1898 to 0.1725, and its ladder made pooled item difficulty look worth about 0.020 ALC (legacy).
    2. **At formative size the lever nearly vanishes.** Over 600 R1 runs pooled difficulty adds about 0.0007 to LegacyP, which needs 64 distinct labelled items on a benchmark, against about two pairs per benchmark in a formative run. The baselines of those runs are in §5.2: the bullets at 134-135 move there.
    3. **Run size, not the information set, removed it.** Pooling switched on and off under both split scopes (table) is still worth 0.009 to 0.030 on dense runs, mostly item difficulty. At formative size it is worth 0.002 to 0.003 to hier and 0.0004 to 0.001 to LegacyP. Reweighted to the platform's composition (16 of 26 formative pair appearances alone on their benchmark, 10 with one companion), it is worth 0.0004 to 0.001 and exactly 0. Per-benchmark splits halve it on dense runs (0.47, 0.36).
  - Table 140-145 and note 147: keep, with cluster-SE format (section 5).
  - The closing sentence of 153 ("nothing in this report is quoted from the legacy replica except where marked 'legacy'"): keep.
  - → App A.2: 124-128, 130's citations, 138 (the method), 150's "grows with company … 0.0027 … 0.0053", and 153's first two sentences.
- 157-162 (§2.4) → condense to: one sentence on `split_scope`; one sentence on coverage (83% to 99.5% at B31 under 'pair' against 0% to 5% under 'benchmark'); and the new scope sentence (6.2). The dense numbers of 162 (matharena -0.0354 / -0.0159; researchcodebench ±0.004; hier-ship's dense gains) → App A.3. 164-170 → App A.3.

### 3.3 §3

- 189-205 (§3.1): keep.
- 209: keep. 211: keep with the m5 replacement (6.3). 213 ("Cost") → App B.1, leaving one clause in 211: "fits that Newton cannot settle are settled by trust-region Newton from three starts (App B.1)".
- 217-230: keep. 232: m5 replacement (6.3).
- 236-241 (§3.4, the signal) → condense:
  - The first formative feedback (§5.1) showed LegacyP confidently optimistic at budget 0.
  - Read through the lower roots of p(1-p) = B31, its pairs sit about 0.9 logit below the public centre (§5.1).
  - If the organisers' leaderboard entry is the empirical-mean baseline, inverting its ALC puts E[p(1-p)] near 0.128 against 0.181 on public runs.
  - Later readings are more central (§5.4).
  - The numbers -1.6 / -0.74 live in §5.1. The continuity-corrected -1.51 / -0.71 → App D.4. 241 → §5.4(b).
- 243, 245, 247: keep, one sentence each. 247's guard definition stays here; the glossary points to it.
- 249: keep the first three sentences. Replace the rest with: "Two later uses read the feedback per pair, still for global hyperparameters only (§5.4); App G.6 lists every use."
- 251-255 → §5.4(a), (b).
- 257 → §5.4(d).
- 259-268 → §5.4(e) and App C.2, C.3. The precision deviation goes once to App C.3, with one clause in §5.4(e).
- 270-279 (§3.5) → two sentences: "`predict` is a pure function of `(input, labeled)`, fits are cached under a content fingerprint of `labeled`, and nothing raises: a failed fit falls back to the prediction without labels, then to 0.5. The archive ships numpy-only modules renamed `paiec_rt/`, and the build writes it only if every check passes, including the organisers' validator and a bit-for-bit match with the in-repository predictor (App B.2)."

### 3.4 §4

- 287-294 (regime table): keep the rows R1, R2, TL, sensitivities (TL-1.2, TL-2.0, TL-mix, TL-noshift, LA-0.8, LA-flat) and RS regimes, renamed (4.2). The "P1 studies (2026-10-02)" row → App C.1.
- 296 (sources) → table footnote.
- 298-303 (how test-like runs are built): keep four bullets, one or two sentences each. The level-tilt bullet keeps the target (-1.6, sd 1.5) and the realised -1.29 (sd 1.70, cluster SE 0.16). Every other realised level goes to **one new table in App C.1**, with columns regime / knobs / target / realised mean (sd). Its rows: TL -1.29 (1.70); TL-1.2 → -1.10; TL-2.0 → -1.56; LA-0.8 → -0.77; LA-flat → -0.59; TUNED -1.28 (1.62); READING -0.63 (1.68), target -0.72, knobs -0.85 / 1.75; AUDIT -0.93 (2.05), target -1.10, knobs -1.8 / 2.1; MIXTURE -0.64 (2.13); FLAT -0.28 (2.23); public R1 -0.71 (1.42). All of these are already in the draft (302, 293, 453).
- 305: keep (in sample at B0, B1; B3 to B31 at z +0.09 to +0.50; R1 B0 at z +8.5; ALC 0.2073 / 0.2075 / 0.2113).
- 307: keep one sentence (46% against 55%), then the m6 sentence (6.4).
- 311-317 (§4.2): keep.
- 321-327 (§4.3) → the SE convention of section 5. Remove "On the headline it is about 2.7 times the cluster SE (§5.3)"; it lives in §5.4(c).
- 331-337: keep. 339 → App C.3, as a table with columns study / rule / where it was fixed / evidence / caveat. The rows: the harness gate (`f7e7d87`); itemsig (predates the gate; read as a plain null); the 14B's rubric and attempts (13.6 h after); the entropy job (rule `78e303e` before the data; 83.8 h); the pooled reading (hash and timestamp; after tables seen; four read runs, from 458); the regime-sensitivity study (hashes, local file times, the precision deviation, the dry run). §4.4 keeps: "These gates live in the studies' plan blocks. We call them fixed in the plans, not pre-registered; App C.3 shows, study by study, what the repository records about when each rule was fixed."
- 340 → App H.1.
- 342-397 (old §4.5) → new §6.1. See 3.6.

### 3.5 §5, paragraph by paragraph

| lines | paragraph | goes to |
|---|---|---|
| 405 | three runs; sources | **keep** the first sentence; sources → a "Sources" line (3.10) |
| 407-411 | run table | **keep** |
| 413 | recomputed ALCs; run 3's archive; regression-check purpose | → App D.1. Main keeps "run 3, the rebuilt archive, was uploaded as a regression and latency check only; its ALC is recomputed from its table (App D.1)" |
| 415 | the recurring pair; sd of a difference of two runs | → App D.1, except the last two sentences, which **keep**: the difference of two independent runs has an sd of about 0.04, against the observed 0.019 and 0.011; formative runs on different draws do not measure an improvement |
| 417-423 | run 1 at budget 0 | **condense** into one paragraph: 0.3589 against 0.25 for a constant, ECE up to 0.75, 0.1996 for answering 0.5 at B0 and B1; strong 2025-26 subjects placed near 0.75; lower roots 0.006 to 0.45, averaging -1.6 (sd 1.5) against public -0.74 (sd 1.50). 421's matched re-read (-0.98, sd 2.05) → §5.4(b). End with the back-reference of 2.3 |
| 425 | run 2 at B0; run 3 | **condense**: B0 0.237 against run 1's 0.359, on different pairs; run 3 B0 0.235. The test-like comparisons (0.2165, 0.2350, 0.03 to 0.045 above) → App D.2 |
| 427-433 | placement table | **keep** the columns regime / mean ALC / single-run sd / run 2's z / run 3's z. The per-budget z strings and the "share ≥ 0.1926" column → App D.2 (the full table) |
| 435 | reading the placement | **keep** two sentences: run 2 is within 0.8 sd of public R1 at every budget, while its B31 is 1.4 sd above the test-like mean; both runs lie within 1.02 single-run sds of every regime's mean ALC. The rest → App D.2 |
| 437-439 | Figure 3 `learning_curves` | **keep**, as Figure 1 |
| 441-443 | Figure 4 `formative_runs` | → App D.2, as Figure D1 |
| 445 | matched estimates | → App D.3 |
| 447-456 | the pooled reading: rule, table, result | → §5.4(d) (two sentences) and App D.4 (everything) |
| 458 | "The reading was run four times" | → App C.3 |
| 460 | run 3 not read | → §5.4(d), one clause |
| 462 | "What the pooled reading supports is modest" | → §5.4(d), first two sentences; the rest is already in §5.4(e) |
| 464 | leaderboard | **keep**, one sentence |
| 468-476 | baselines table | **keep** (§5.2) |
| 478-485 | hier-fit against LegacyP, four settings | → App E.1. §2.4's scope sentence points there |
| 487 | where hier-fit's gain comes from; first verdict | **condense** to three sentences: hier-fit beats LegacyP on public runs, by 0.0024 in the primary setting (all four settings in App E.1), mostly by pooling a benchmark's level across the subjects a run holds, and not at all for a target alone on its benchmark; the first verdict was therefore to keep LegacyP; §5.3 reversed it. Add the moved 2.3 bullets: EmpMean 0.2526 ± 0.0014 against 0.2500 (one label sends it to 0 or 1); LegacyP beats Smooth by 0.0026 ± 0.0009 with a strict prior or 0.0069 ± 0.0016 with a target-LOBO prior (pair-cluster SEs). The swe_rebench loss and the real-run weighting → App E.1 |
| 491-502 | unconstrained families | **keep** (§5.3) |
| 504 | Smooth-cal row's provenance | table footnote, one sentence; the rest → App E.4 |
| 506-511 | four observations | **condense** bullets 1 to 3: the plateau 0.156 to 0.157, 14 of 108 within 0.001, a calibrated smoothed mean reaching 0.159; the widest prior always won; the guard decides. `fit_ab`'s divergence → App B.6. Bullet 4 → one clause: "how much of this to call calibration is §5.5" |
| 513-515 | Figure 5 `level_surface` | **keep**, as Figure 2 |
| 517-530 | the shipped table | → §5.4(c) |
| 532-536 | five bullets | → §5.4(c), as four sentences (2.2) |
| 538-540 | Figure 6 `level_fix_budgets` | → §5.4(c), as Figure 3 |
| 542 | against the neighbouring configuration | → §5.4(a), the merged table's footnote; the triples → App E.2 |
| 544-553 | sensitivity table | → App E.2 |
| 555 | "the gain shrinks …" | → App E.2. The RS half of the sentence is in §5.4(e); the no-shift clause is in §5.4(c) |
| 557 | EB adapts the right way | → App E.3. §5.4(e) keeps the EB result |
| 559-561 | Figure 7 `eb_adaptation` | → App E.3, as Figure E1 |
| 563-567 | P1a: intro and three bullets | → §5.4(e), two sentences; "Priors and code" → App H.2 |
| 569-585 | P1a table | → §5.4(e). Caption gets m6's seed and library (6.4) |
| 587-589 | Figure 8 `regime_sensitivity` | → §5.4(e), as Figure 4 |
| 591-599 | the rule; D1 | → App C.2 (five conditions), C.3 (deviation); one sentence in §5.4(e) |
| 601-606 | four bullets of results | → §5.4(e), as four sentences |
| 607-610 | Smooth-cal recovers the tuned gain | → §5.5, first bullet |
| 611 | hier-nosubj | → §5.5, inside the public-runs bullet; the B0-centre detail (0.31 against 0.42) → App E.4 |
| 612 | against LegacyP by RS regime | → §5.4(e), last result sentence |
| 614 | four parents | → §5.4(e), closing sentence |
| 616 | the baseline study's intro | **condense** (§5.5): "On the RS study's TUNED and public runs we scored a plain 1PL (hier with attributes, identity link, groups, floor and slip off: a pooled level, an ability per pair, a difficulty per item, refitted at every checkpoint), at hier-ship's level and at its own leave-one-parent-out public level, and the organisers' empirical mean." Row provenance → App E.4 |
| 618-626 | ALC table | **keep** (§5.5); caption gets m6 |
| 628-634 | the shares, by order of the steps | **keep** (§5.5) |
| 636-639 | bullets 1 to 4 | **keep**, condensed by a third, with the Smooth-cal bullet from 607-610 merged into bullet 2 |
| 640 | B0 attribute prior | → App E.4; §5.5 keeps one clause: "at B0 the attribute prior pays once the centre is held level" |
| 641 | EmpMean paired | → App E.4. The ALCs are in the table and §1.6 carries the differences |
| 642 | BLE | **condense** to one sentence: "BLE, the organisers' language-model predictor, could not be run: each prediction is an agent run against a paid API, 569,838 of them on these runs alone (App E.5); the best leaderboard entry (0.1172) stays unexplained" |
| 644 | caveat | **keep** |
| 646-652 | single-subject | → §5.6: the first bullet's four numbers, "the loss sits at B0 to B3", and 652. 650 → App E.6 |
| 654-668 | hier-ship across regimes by budget | → App E.7 (the table); "Which code" (668) → App H.2 |
| 670-697 | ablations; Figure 9 | → App B.3, with Figure B1. §5.2 gains one sentence: "On public runs three of hier's components pull their weight: the attribute prior, the feature groups and the pooled level (App B.3)" |
| 699 | corrected floor | → App B.4 |
| 703 | cost | → App B.5. §5.6 keeps: "hier-ship takes 0.80 to 1.04 ms an evaluation call on the run-2 library and 1.34 ms on the current one (slowest call 0.56 s), and 7.5 to 9.4 ms on dense matharena; a formative run is about 3,000 calls, a few seconds against the 8-hour limit" |
| 881 (§7) | platform composition and swe_rebench | → §5.6, two sentences: at the platform's composition hier-ship is not distinguishable from LegacyP under per-pair splits (+0.0024 ± 0.0023) and behind it under per-benchmark splits (+0.0045 ± 0.0024); 87 of the 193 lone pairs are swe_rebench, and without it these become +0.0004 ± 0.0026 and +0.0017 ± 0.0022. The class detail → App A.2 |

### 3.6 §6, paragraph by paragraph

**New §6.1, the acceptance harness and its gate (from old §4.5):**

| lines | paragraph | goes to |
|---|---|---|
| 344-351 | the offset, the cap, the two slopes | **keep** |
| 353 | the acceptance reproduction; degraded oracles | **keep** the degraded-oracle definition; the reproduction (-0.0437 / -0.0557 against -0.0439 / -0.0558) → App F.5 |
| 355-366 + 370-375 | gate table and pass-probability table | **merge** into one table: honest r (within-pair r); transferred nested TL Δ (cluster SE, draw sd); draws passing (Jeffreys 95%); P(one draw clears -0.002), normal / t; P(full gate); per-pair nested TL Δ; per-pair draws passing. The mix/whole and public columns and the pre-recollection note → App C.4. The rows r = 0 and 0.1 stay with the transferred line only |
| 368 | "What a covariate needs" | **condense** to the bold heads of 377-378: transferred slope r ≈ 0.3 to 0.35, two times in three at 0.3; per-pair 0.46 |
| 377 | detail | → App C.4, except "of the two failing draws, one misses the bar and one fails the worst-parent condition", which **keeps** |
| 378 | per-pair | **keep** the first sentence; the forced costs → App C.4 |
| 379 | thinnest regime; mix/whole | **keep** the first two sentences; the rest → App C.4 |
| 381 | one correlation scale | **keep** three sentences: the reliability (0.84 to 0.95, so 0.5% to 1.6% inflation); the unit (within-pair, bar 0.25); a degraded oracle keeps 0.82 of its r within a pair and the 14B's covariates 0.57 to 0.65. The rest → App F.1's note |
| 383-393 | within-pair table | → becomes the "within-pair r" column of the §6.2 compact table; the full table → App F.1 |
| 395-397 | Figure 2 `gate_curve` | **keep**, as Figure 5 |

**New §6.2 and §6.3, what was tried, study by study:**

| lines | paragraph | goes to |
|---|---|---|
| 709 | intro | **condense** to the footing sentence (W7): which ideas were scored through the harness against hier-ship leave-one-parent-out, which are data-level correlations, and that §6.3's last paragraph is legacy-replica. "They share one reason for failing …" → §6.4 |
| 711-731 | table | → App F.1 in full (scripts, intervals). Main: a compact table with columns idea / footing / key number / within-pair r / verdict, one row per family. The known-sign row carries old A.4's decisive values: text length +0.28 / +0.36 / -0.43 within paper; stated size +0.50 on one benchmark (A.4 merged, see below) |
| 733-735 | Figure 10 `ideas_forest` | **keep**, as Figure 6 |
| 739-745 | text maps | **condense** to one run-in paragraph, **Text maps.** TF-IDF leave one benchmark out: -0.16 to 0.14 against Rasch difficulty, every 95% interval below 0.3, within a pair 0.07. Embeddings leave one benchmark out: -0.23 to -0.02 for ridge; within a benchmark they mostly identify the item_features group. → App F.2: all intervals, the matharena 0.80 / 0.58 and the note on the earlier 0.73 / 0.49, and the group-held-out values |
| 747-749 | Figure 11 `transfer` | → App F.2, as Figure F1 |
| 751, 769, 792, 815, 835, 845, 865 | *Literature* | one line each to §1.4; the full paragraphs stay with their study in App F |
| 755-773 | judges | **condense**, **Language-model judges.** The blind rater: 0.482 [0.25, 0.64] on matharena, -0.13 to 0.21 elsewhere; inconclusive outside mathematics at n = 45; its prompt is undocumented (conduct statement). The fully specified 4B judge was closed. The 14B rated every item and settles the power question. → App F.3: the table, the control, the rater paragraph, the power paragraph, the 4B paragraph |
| 777-786 | itemsig | **condense**, **A label-conditional layer.** Nested -0.00002 on TL; it cannot act at B0 or B1; at B31 it takes 0.5% of the 0.063 gap between the pair-rate and item oracles (Figure 7). → App F.4: the table (with m6's run labels) and the held-out researchcodebench loss |
| 788-790 | Figure 12 `item_gap` | **keep**, as Figure 7 |
| 796-803 | meta-heads | **condense** to two sentences in **Probes, heads and encoders**: nested every head is off in every fold (0); forced, they cost +0.0002 to +0.0006. → App F.5 |
| 807-813 | known-sign cues | **condense**, **Known-sign cues.** Most cannot be tested across benchmarks; text length changes sign; stated size has within-pair r +0.41 on one benchmark; nothing passes (0 to +0.00003). → App F.6, with old A.4 (1102-1112) merged as its sign-check table |
| 819-833 | subject side | **condense**, **The subject side.** Ordered effort is on in all four folds but gains -0.00097, half the bar, through the date slope under the synthetic shift; harness identity cannot be measured leave-one-parent-out; the date form is decided by the shift; the t level ties. → App F.7: the table and the bullets |
| 839-843 | legacy replica | **one paragraph** (review m9), **Measured on the legacy replica.** Acquisition: coverage-first -0.0000 ± 0.0013, so the submission ships no `labeling.py`. Calibration: temperature and slip lose 0.0007 ± 0.0011. Offline bank: 1 of 161 inventory benchmarks judged usable, closed on feasibility and on the conduct rule (conduct statement). → App F.8: the rest |
| 849-856 | six closed studies | → the compact table rows, plus one sentence in **Probes, heads and encoders** (4B attempts below the floor; entropy and hidden-state heads a replicated null; in-context learning r 0.206; pairwise q 0.540; fine-tuned encoder 0 of 4 parents at r 0.3). → App F.9 |
| 858-863 | the 14B | **condense**, **A 14B on a free GPU.** Its attempts' entropy reaches a within-competition Spearman of 0.372 [0.209, 0.516] on mathematics, the one signal past its correlation bar. Its reasoning entropy over all four parents orders difficulty on mathematics and research code (+0.29, +0.33 within group) but weakly or not on the agentic two (+0.12, -0.03). Within a pair it reaches 0.12 [0.09, 0.15], and it fails the gate under a rule fixed before the data existed. → App F.9: sessions, coverage, the 2026 contests, truncation, nested and forced lines |
| 869-870 | why these fail | → §6.4. Keep both points. In point 2, replace "The target does not flatter them … 0.57 to 0.65 (§4.5)" with "(§6.1: the target does not flatter them; the unit does)". In point 1, replace the B0/B1 Brier values of runs 2 and 3 with "(§5.1)" |

### 3.7 §7, §8, references and the old appendices

- **§7 (874-888) → ten bullets, about 550 words:**
  1. **Five public benchmarks**: keep the first two sentences of 876, with "(§5.4)".
  2. **A self-built regime** (877, condensed): tuned in sample to one run; realises -1.29, not -1.6; a synthetic date shift sets the size of the gain, which is 0.014 without it; most of the tuned headline is LegacyP's own optimism under the shift (§5.5); regimes at the feedback's readings give 0.025 to 0.031 but share catalogue and shift and realised milder levels (§5.4). The level-identification distances (0.63 / 0.37 / 0.46) and the mixture's two pseudo-benchmarks → App C.1.
  3. **What hier adds** (878): one sentence, "over a calibrated level, hier-ship's measured value is robustness to where the hidden level sits (§5.5)". The numbers stay in §5.5.
  4. **Baselines missing** (879): BLE, and no 1PL-specific level calibration, with "(§5.5, App E.5)".
  5. **Selection optimism** (880): two sentences; the hyperparameter list → App C.2.
  6. **Single-subject benchmarks** (881): one sentence, "(§5.6)". The composition detail → §5.6 and App A.2.
  7. **Unexplained** (882): two sentences; detail → App I.
  8. **Library defects** (883-886): one sentence → App B.6.
  9. **Provenance gaps** (887): one sentence → App H.3, which holds the precision deviation and the lock evidence.
  10. **Feedback is noisy** (888): keep.
- **§8 (892-1001)** → App G, verbatim apart from renames and cross-references. In 960, gloss the README heading "Коммит D" once as "('Commit D', the entropy job)". The unnumbered "Code, data and conduct" statement (8.3) replaces §8 in the main text.
- **References (1005-1024)**: stay after §7 and the statement, before the appendices. The `TODO(verify)` notes belong to the literature lane.
- **Old A.1** → App D.1. **Old A.2** → App E.1. **Old A.3** → main §5.4(a), merged. **Old A.4** → main §6.2's compact known-sign row, plus App F.6 in full. **Old A.5** → App E.1. **Old App B** → App I. **Old App C** → App H.2.

### 3.8 Cross-reference map (old → new)

The editor must update every `§`, "Appendix", "A.n" and "Figure n" reference. The draft holds 227 `§` references, so search for each old target.

| old | new | old | new |
|---|---|---|---|
| §1.4 | §1.6 | §5.1 | §5.1 (reading → §5.4(d), App D.4) |
| §2.3 | §2.3 (detail App A.2) | §5.2 | §5.2 (hier-fit table → App E.1) |
| §2.4 | §2.4 (numbers App A.3) | §5.3 | §5.3, §5.4, §5.5 by topic (3.5) |
| §2.5 | §2.2 / App A.4 | §5.4 | App E.7 |
| §3.4 "What shipped", "After run 2", "Tested …" | §5.4(a)-(e) | §5.5 | App B.3, B.4 |
| §3.5 | §3.5 / App B.2 | §5.6 | §5.6 / App B.5 |
| §4.4 (timing) | App C.3 | §6.1-§6.8 | §6.3 run-ins / App F.2-F.9 |
| §4.5 | §6.1 (detail App C.4) | §6.9 | §6.4 |
| §8.x | App G.x | Appendix A.1 / A.2 / A.3 / A.4 / A.5 | App D.1 / E.1 / §5.4(a) / F.6 / E.1 |
| Appendix B | App I | Appendix C | App H.2 |

### 3.9 Figures: which figure carries which claim

| new | file (`docs/report/fig/`) | figures.md | placed in | the claim it carries |
|---|---|---|---|---|
| Figure 1 | `learning_curves` | D1 | §5.1 | the formative runs sit within single-run noise of both the tuned and the public regimes; runs on different draws do not measure an improvement |
| Figure 2 | `level_surface` | D2 | §5.3 | a flat plateau; the widest level prior wins everywhere; the guard, not the selection half, decides; where hier-argmax, hier-rec and hier-ship sit |
| Figure 3 | `level_fix_budgets` | D6 | §5.4(c) | the tuned gain is at B0 and B1; on public runs hier-ship pays at B0 and recovers from B7 |
| Figure 4 | `regime_sensitivity` | D4 | §5.4(e) | at the feedback's readings no alternative beats hier-ship by the rule's 0.002; hier-rec wins only where it was chosen |
| Figure 5 | `gate_curve` | D3 | §6.1 | a transferred slope needs an honest r of about 0.3 to 0.35; no measured covariate reaches the gate; the per-benchmark correlations of the text maps, judges and 14B |
| Figure 6 | `ideas_forest` | D12 | §6.2 | no idea's nested line reaches -0.002 |
| Figure 7 | `item_gap` | D8 | §6.3 | the item-level headroom is large and the itemsig layer recovers 0.5% of it on TL |
| Figure A1 | `empirical_mean` | D10 | App A.4 | the replica agrees with the analytic ALC of the empirical mean; where the organisers' entry would sit |
| Figure B1 | `hier_ablations` | D11 | App B.3 | which of hier's components pull their weight on public runs |
| Figure D1 | `formative_runs` | D5 | App D.2 | the per-pair spread of each formative run |
| Figure E1 | `eb_adaptation` | D7 | App E.3 | the EB centre moves the right way, and the date shift moves it further down |
| Figure F1 | `transfer` | D9 | App F.2 | item difficulty from text maps does not transfer between benchmarks |

Editor's tasks:

- **Captions.** Update the draft's short captions: new numbers; names on first use as "LegacyP (legacy Predictor)" and "hier-ship (shipped hier)", because the drawn legends still say "legacy Predictor", "shipped hier" and "neighbouring config"; and the in-caption cross-reference at 515, "P1a's 3.5 and 5 are in Figure 8" → "the wider priors are in Figure 4".
- **`figures.md`.** Update the placement table (lines 28-41) with the new numbers and sections, and the section references at 43 and 47 (§6.1 → App F.2; "draft §5.4" → App E.7).
- **Figure lane (optional).** `tools/report_figures.py` keeps every legend name in `NAMES` and `FORMATIVE` (lines 71-100) and two axis labels (653, 812: "vs shipped hier"). Renaming them to the glossary means a rebuild, a new `manifest.json` and the check rerun. Map: ship → hier-ship; legacy → LegacyP; smooth → Smooth; aggr → hier-rec (−3.0 / 0.25); eb_fit → hier-fit; eb_adapt → hier-EB on −3.0 / 0.5; eb_ship → hier-EB on hier-ship; wide35 / wide50 → hier-ship, σ_μ 3.5 / 5.0; smcal → Smooth-cal; onepl → hier-nosubj.
- **New figures.** The figures.md candidates (a pooling-by-company panel, a decomposition bar for the baseline study) are not needed: the §2.3 and §5.5 tables carry those claims.

### 3.10 Sources in the main text

Sections 1 to 7 now cite F§ 93 times and `results/` 88 times, mostly mid-sentence. The rule for the edit:

- Main-text prose cites a source once per paragraph, at its end, in square brackets: `[F§ "Shipped configuration, confirmed"; results/ship_confirm.json]`.
- Each table and figure keeps its "Source:" line.
- A subsection whose numbers all come from one script ends with one "Sources." line instead of inline citations.
- No citation is deleted outright. A citation dropped from prose must remain in that subsection's Sources line, in its table's Source line, or in App H.2.

---

## 4 Names and notation (m1)

### 4.1 Models: the first table of new §1.5

| name | what it is |
|---|---|
| **LegacyP** | Our first submission's predictor (`paiec/predict.py`): a per-pair logistic model with a multiple-choice floor, subject ability around a relative attribute prior, and item difficulty from a joint IRT once a benchmark has 64 labelled items. Formative run 1. |
| **hier** | The hierarchical item-response model of §3 (`paiec/hier.py`). |
| **hier-fit** | hier with every hyperparameter from the empirical-Bayes fit, its level prior included (no LEVEL). |
| **hier-ship** | hier with the level prior LEVEL: mu0 -2.5, sigma_mu 2.5, attr_scale 0.5. Our submission; formative runs 2 and 3. |
| **hier-rec** | hier with mu0 -3.0, sigma_mu 2.5, attr_scale 0.25: the level calibration's recommendation, replaced by an audit before shipping. |
| **hier-argmax** | hier with mu0 -3.5, sigma_mu 2.5, attr_scale 0.25: the selection rule's choice. |
| **hier-ship σ3.5, σ5** | hier-ship with sigma_mu 3.5 or 5.0. |
| **hier-EB** | hier whose new-benchmark level centre is re-estimated by empirical Bayes at every checkpoint, on a stated base prior: "hier-EB on hier-ship", "hier-EB on −3.0 / 0.5", or the level calibration's EB-cs (tau 2) on the base its table names (App E.3). |
| **hier-nosubj** | hier-ship without its subject prior (attribute prior and identity link off). |
| **1PL-ship, 1PL-fit** | A plain 1PL (one pooled level per benchmark, one ability per pair, one difficulty per item, refitted from `labeled`) at hier-ship's level, or at its own leave-one-parent-out public level. |
| **Smooth** | The Beta(2,2)-smoothed mean of the pair's labels. |
| **Smooth-cal** | A smoothed mean whose Beta prior (mean 0.25, strength 2) was chosen by LEVEL's selection rule on the tuned regime, regardless of the guard. |
| **EmpMean** | The organisers' empirical-mean baseline. |
| **BLE** | The organisers' language-model predictor (not run, §5.5). |
| **LegacyP+fix** | LegacyP with a level fix (offset, scale, variance; values in the tables). |
| **oracles** | Base-rate oracle: every pair at its true rate, so E[p(1-p)]. Pair-rate oracle: every response at its pair's rate. Item oracle: in-sample Rasch probability per item. |

### 4.2 Terms: the second table of new §1.5 (the rest go to App H.1)

| term | meaning |
|---|---|
| subject, item, pair | an evaluated AI system (eight visible string fields); a benchmark item; a (subject, benchmark). ALC averages pairs equally. |
| appearance | one pair in one run. Replica runs redraw pairs, so a pair has many appearances. |
| budget Bk | the checkpoint after k labels per pair, k in {0, 1, 3, 7, 15, 31}. ALC = 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31. |
| level; LEVEL | a benchmark's mean pair-accuracy logit; hier-ship's prior on a new one. |
| R1 (R1-bf, R1-pu); R2 | public formative-like runs, benchmark-first or pair-uniform weighting; dense runs, every eligible pair of one benchmark. |
| parent; pseudo-benchmark | a public benchmark (four multi-subject parents carry every test-like number); a group of one parent's items given its own anonymous id in a test-like run. |
| TL; TL-mix; TL-noshift; TL-1.2, TL-2.0 | test-like runs at the defaults; with item groups merged at random and whole parents; without the date shift; with level_mean -1.2 or -2.0. |
| LA-0.8, LA-flat | the level audit's two extra regimes: level_mean -0.8, and no level tilt. |
| RS regimes | TUNED, READING, AUDIT, MIXTURE, FLAT: the regime-sensitivity study's test-like regimes (§4.1, App C.1). TUNED is TL redrawn at seed 11. |
| S, C, H halves | test-like runs 0-99 (selection), 100-199 (confirmation), 200-299 (held out, never used to choose the level). |
| date shift; level tilt | test-like runs shift visible release and access dates 1.25 years later; they tilt the draw of pairs toward a target level distribution. |
| split scope | whether the 50/50 acquisition / evaluation split is drawn per pair ('pair') or per benchmark item ('benchmark'). |
| LOPO; strict run-LOBO; nested LOPO | fitted without the target's parent; without every benchmark of the run; selection redone inside each held-out parent and bootstrap resample. |
| guard; gate; rule | the guard: lose at most 0.003 ALC to LegacyP on both R1 weightings. The gate: the four acceptance conditions of §4.4. The rule: the regime-sensitivity study's decision rule (App C.2). |
| honest r; within-pair r | correlation with item difficulty fitted on other subject folds; the same correlation within a test-like pair, the gate's scale. |
| single-run sd | the spread of one run's ALC over a regime's runs: the noise of one formative run. |
| lower-root reading; pooled reading | run 1's pair rates read from p(1-p) = B31 at the lower root; the reading of runs 1 and 2's 17 pairs against matched replica pairs (§5.4(d)). |

To App H.1, as a glossary of internal labels: seeds; archives (archive-1 `8e28d930…`, archive-2 `2c64eaad…`, archive-3 `4a882cc7…`); libraries ("run-2 library" = hier 70a3a81a with the old floor and pre-fix solver; "current library" = `4d2cc4f`); "legacy rows" and "current rows" of the harness; F§ and P§; "the precision deviation" (D1); "the entropy job" (commit D); "the 4B", "the 14B", "the blind rater".

### 4.3 Find-and-replace list (old → new)

Do not replace blindly: "legacy replica" is not LegacyP, and "smoothed mean" is sometimes generic.

| old (count at dd3e372) | new | notes |
|---|---|---|
| legacy Predictor (93); the Predictor (1); the first submission's predictor (2); our first predictor (1); "Predictor, target-LOBO prior" (§5.2 table) | LegacyP | keep "legacy replica" as is |
| legacy Predictor with a level fix; "legacy Predictor fix" | LegacyP+fix | |
| shipped configuration (29); shipped model (28); shipped hier (9); SHIP (31, in P1a text and tables); "the milder guarded configuration"; "hier G mu0 -2.5, sigma_mu 2.5, attr_scale 0.5 (shipped)" | hier-ship | keep the results-file keys (`hier G mu0=-2.50 …`, `ship`) inside code spans |
| neighbouring configuration (15); neighbouring (22 in all); "the -3.0/0.25 config"; "the aggressive setting" (253); "neighbouring (mu0 -3.0, attr_scale 0.25)" | hier-rec | |
| rule's argmax (2) | hier-argmax | |
| sigma_mu 3.5 / sigma_mu 5.0 (as configs) | hier-ship σ3.5 / σ5 | keep "sigma_mu" as the parameter name |
| hier's fitted defaults; hier, fitted defaults; hier at its fitted level (no LEVEL); fitted EB hyperparameters; hier's fitted level prior | hier-fit | |
| adaptive EB on SHIP's level; adaptive EB on mu0 -3.0; empirical-Bayes level / EB level (as configs); hier EB-cs | hier-EB on hier-ship; hier-EB on −3.0 / 0.5; hier-EB (EB-cs) on the base its row names | EB-cs sits on different bases in old A.3, the unconstrained table and Figure E1 |
| hier without its subject prior (10); hier, no subject prior (2); `onepl` (P1a key) | hier-nosubj | not the plain 1PL |
| plain 1PL at the shipped level; plain 1PL at its fitted public level | 1PL-ship; 1PL-fit | |
| smoothed Beta(2,2); the smoothed mean (as a model) | Smooth | |
| smoothed mean, calibrated prior (P1a); level-calibrated smoothed mean; calibrated smoothed mean; smcal | Smooth-cal | |
| organisers' empirical mean; empirical mean (official baseline) | EmpMean | keep "empirical mean" in the analytic-ALC sentences |
| test-like (primary); tuned test-like; tuned regime | TL (TUNED when seed 11) | |
| groups merged at random (and wholes); mix/whole (32) | TL-mix | define once in §4.1 |
| no date shift (as a regime) | TL-noshift | |
| level_mean -0.8 (40; realised -0.77); no level tilt (40; realised -0.59) | LA-0.8; LA-flat | avoids the clash with the RS regime AUDIT |
| R1 bench-first; benchmark-first (as a regime); r1b; R1B | R1-bf | keep "benchmark-first" in the definition |
| pair-uniform (as a regime); r1p; R1P | R1-pu | |
| pair appearance | appearance | |
| target-LOBO | LOPO (on public runs the parent is the benchmark) | |
| commit D | the entropy job | |
| deviation D1 | the precision deviation (App C.3) | |
| "the archive now selected" | archive-3 (`4a882cc7…`) | it may not be the final selection |

### 4.4 Review tags and internal labels: out of the main text

Sections 1 to 7 hold about 100 review tags: "P1a" 52 times, "P1.n" 49, W/Q 10.

- **Replace:**
  - P1a → "the regime-sensitivity study" on first use, then "the RS study"; "P1a regimes" → "RS regimes"; "P1a's runs" → "the RS runs (seed 11)".
  - P1.9 → "the baseline study". P1.11 → "the pooling decomposition". P1.17 → "the row export".
  - P1.10, P1.12, P1.13, P1.15, P1.16, W3, W4, W7, Q3, Q5, Q12, m-items, "P0 revision" → drop.
  - "v0", "v1", "since v0" → drop or "earlier".
- **Map kept in App I:** one table, review item → section that answers it.
- **Internal dates** ("P1 studies (2026-10-02)", revision dates) → App H.4.

---

## 5 One SE format (m7)

**Definition: replace §4.3's list (321-327) with this text.**

> All comparisons are paired: every candidate scores the same runs, checkpoints and acquisitions, and a run's ALC is the mean over its pairs. A difference Δ = A − B is negative when A is better. Three SEs describe it, written in full as "Δ ± a / b / c (run / cluster / stratified SE)":
>
> - The **run SE** is over runs. It overstates precision, because runs redraw the same pairs.
> - The **cluster SE** is a pair-cluster bootstrap (2,000 resamples, ratio estimator) over (benchmark, subject) clusters on public runs and (parent, subject) clusters on test-like runs.
> - The **stratified SE** is the same bootstrap, resampled within each parent.
>
> **In the main text, "±" and an SE in parentheses after a difference mean the cluster SE, unless marked otherwise.** The full triple is given only in the confirmation table of §5.4 and in the appendices. None of these SEs covers variation between benchmarks, so we also report the **parent-level** mean ± SE over the four multi-subject parents ("parent-level -0.038 ± 0.008"), which on the headline is about 2.7 times the cluster SE (§5.4). Other spreads are named where they occur:
>
> - "± SE over appearances": the single-subject benchmark;
> - "± SE across benchmarks": dense runs;
> - "± run SE": R1 baselines, where no cluster SE is stored;
> - "SE, benchmark bootstrap": the pooled reading.
>
> A **single-run sd** is the spread of one run, not an SE. [a, b] is a 95% interval. When many options are compared with one default we state the Bonferroni bar (2.95 SEs two-sided for sixteen comparisons).

Remove the "2.7 times" clause if the editor keeps it in §5.4(c) alone (section 7, row 20).

**Tables.**

- **Main-text tables.** Write each difference cell as "−0.0419 (0.0037)", with the column header "Δ ALC (cluster SE)" and the sign convention in the header, for example "hier-ship − LegacyP" or "alternative − hier-ship". The headline, merged-A.3, RS and baseline tables use different directions now (§1.4 "shipped minus comparator", P1a "X minus SHIP"). Keep each table's own direction, because flipping a sign would change a number, and state it in the header.
- **The exception.** The §5.4(c) confirmation table keeps "± a / b / c", because it is where the triple is shown next to its definition.
- **Appendix tables.** Keep their current format, with "(run / cluster / stratified SE)" in the header.
- **Cells with no stored cluster SE.** Write "(run SE)" in the header or "(no SE)" in the cell, as the draft already does for derived values.
- **Run SEs dropped from main text.** Every triple in §1.4 and §5.3 was checked present in `docs/findings.md`: 44 differences, those of the headline, confirmation, unconstrained-family, guarded-configuration, RS, baseline, single-subject and neighbouring rows. Dropping their run and stratified SEs from main-text cells is therefore a cut with a trace. Give each table a Source line naming the F§ section or results file.

**Tables to convert in the main text.**

- §1.6 headline (8.4): cluster SE only.
- §2.3 pooling table: formative columns "± run / cluster / stratified" → cluster; the dense columns keep "± SE across benchmarks", so label the header.
- §5.2 baselines: "ALC ± run SE" stays (no cluster SE stored).
- §5.3 unconstrained: triples → cluster.
- §5.4(a) merged A.3: the confirmation column's triple → cluster.
- §5.4(e) RS table: already cluster.
- §5.5 tables: already ALC or shares.
- §6.1 merged gate table: already (cluster SE, draw sd).
- §6.2 compact table: one number per row; the triple moves to App F.1.

---

## 6 Wording fixes (m3 to m6, the §2.4 scope sentence, the §3.2 and §3.3 caveats)

### 6.1 m3: subjects against canonical names (§2.1, line 101)

**Current:**
> - **Few subjects cross benchmarks.** Only 22 of 287 subjects appear on more than one benchmark. `interactors` is empty everywhere [P§ Data].

**Replacement:**
> - **Few subject ids cross benchmarks; more model names do.** Only 22 of 287 subjects appear on more than one benchmark, and 24 of the 221 pairs share all eight visible subject fields with a pair on another benchmark [P§ Data]. hier keys a subject's standing theta_s on the coarser canonical name (`normalized_name`, else the source model name), which links 117 of the 220 pairs of the four multi-subject benchmarks across benchmarks (F§ "The step-2 analyses behind the model"). On top of the attributes, that link adds almost nothing (§3.3). `interactors` is empty everywhere [P§ Data].

Notes: "24 of the 221" is in `docs/protocol.md` (Data); drop that clause if the editor wants no number new to the draft. "117 of 220" is in findings at line 847, as m3 asks. The canonical-name rule is `paiec/hier.py` line 424 and `paiec/prior.py` line 85.

### 6.2 m4 and the §2.4 scope sentence (line 162, last two sentences)

**Current:**
> We report both scopes for the baselines, for the comparison of hier with the legacy Predictor (§5.2) and for the pooling decomposition (§2.3). The other later studies ran split scope 'pair' only (§7).

This is wrong in one detail: the R1 baselines of §5.2 are 'pair' only (F§ "Formative-sized runs (R1)"). Only the dense R2 baselines have both scopes.

**Replacement:**
> Both scopes are reported in three places only: the dense R2 baselines (App A.3), hier-fit against LegacyP on public runs (App E.1), and the pooling decomposition on dense and formative-size runs (§2.3). Everything else ran split scope 'pair' only: the R1 baselines (§5.2), the level calibration and its public guard, hier-ship's confirmation (§5.4), the regime-sensitivity and baseline studies, the harness, and every official-protocol study of §6. Scope matters most on dense runs, where per-benchmark splits halve the value of pooling (§2.3); on formative-size runs both scopes leave it small.

Sources: the guard is 'pair' only (F§ "Calibrating for the hidden test", "What was cut", line 1397); the subject side likewise (F§ line 2583); the RS public regimes "R1B … split scope 'pair'" (F§ line 5663); `paiec/testlike.py` and `paiec/official.run_official` default to `'pair'`.

### 6.3 m5: three caveats

**(a) §3.3, tau2_res (line 232).**

**Current:**
> The subject's cross-benchmark identity carries almost nothing beyond its attributes. The shared part of a named model's attribute residual across benchmarks, tau2_res, is estimated at -0.002 on the public data and clipped to [0.01, 0.2]. That puts sigma_theta at its floor of 0.1 and the link weight across benchmarks at 0.0018 (`paiec/hier.py` docstring; F§ "The step-2 analyses behind the model").

**Replacement:**
> On current evidence, the subject's cross-benchmark identity carries almost nothing beyond its attributes. The shared part of a named model's attribute residual across benchmarks, tau2_res, is estimated at -0.002 on the public data and clipped to [0.01, 0.2]. That puts sigma_theta at its floor of 0.1 and the link weight across benchmarks at 0.0018 (`paiec/hier.py` docstring). The estimator is biased downward: the ridge that predicts a name's standing on one benchmark was fitted without that benchmark but saw the same name's standings on the others, which pulls its residuals on two benchmarks apart. Until the estimator is corrected and re-measured, the identity conclusion and the small link weight it sets are provisional (F§ "The step-2 analyses behind the model").

The source is findings lines 856-862.

**(b) §3.2, the exact marginal (line 211).**

**Current:**
> For one pair alone it is the exact marginal. With other subjects on the benchmark, the line ignores their skew.

**Replacement:**
> It is the exact marginal only when every label loads on x through a'x alone, as for one pair whose labelled items share the target item's group effects (the pinned test, `test_laplace_error_at_low_budgets`, has no groups, floor or slip). Labelled items in other item_features groups, or other subjects on the benchmark, add components that the line moves along their Gaussian conditional mean, ignoring their skew, so the read is then an approximation.

The next sentences stay. In "The docstring of `paiec/hier.py` documents the remaining approximation error case by case", add "for other subjects on the benchmark and for floored benchmarks; a single pair whose labelled items lie in other groups is not measured separately".

This was checked: the docstring (`paiec/hier.py` lines 1-200 at `dd3e372`) measures the other-subjects and floored cases and contains no group-only case. That answers review Q6 without a new number. The docstring's worst listed case, 0.028 for a target with labels of its own after others' extreme records, may be added only if the editor accepts a number new to the draft from a source the draft already cites.

**(c) §3.4, "It is also the only use of one noisy run that generalises".** It is no longer in the draft at `dd3e372` (no match). Nothing to do; do not reintroduce it.

### 6.4 m6: mixed run sets

The rule: every base ALC names its runs and seed, and its library where it differs.

- **Add to §4.1, after line 307** (every number is already in the draft):
  > Base ALCs differ between tables because run sets differ: hier-ship scores 0.1658 on TL runs 0-299 (seed 2) and 0.1690 on TUNED (seed 11, current library); on R1-bf it scores 0.2051 on runs 0-149 and 0.2057 on runs 0-199 (seed 0, run-2 library), and 0.2114 on 60 runs at seed 11 (current library). Every table names its runs and seed.
- **App F.4 (old §6.3 table, 779-784):** header note, "Base ALCs in brackets. R1-bf here is runs 0-199 (0.2057), not the 0-149 of App E.7 (0.2051)."
- **§5.4(e) RS table caption (585):** add "seed 11; library of `4d2cc4f`".
- **§5.5 ALC table (618):** header "TUNED and R1, seed 11 (the RS runs)".
- **App E.7 table (656-662):** the "runs" column → "runs (seed)": 0-299 (2); 0-199 (3); 0-99 (3); 0-149 (0); 0-99 (0). The seeds are those of `figures.md` D1's caption and §4.1/§8.5.
- **§1.6:** every row names its run set (8.4).

### 6.5 Other wording fixes found while reading

- **Regime-name clash.** "AUDIT" (an RS regime) and "the audit's" regimes (level_mean -0.8, untilted) collide. Rename the latter LA-0.8 and LA-flat (4.3).
- **Mixed difference directions.** §1.4 and the confirmation table use "shipped minus comparator"; the RS and §5.3 neighbour paragraphs use "X minus SHIP". State the direction in every table header (section 5). Do not flip signs.
- **"The archive now selected"** (411, 413, 668, 931): write "archive-3", since the final selection may change.
- **The draft-version line and dates** (3; "Draft v1", "revised", "2026-10-02") → App H.4.
- **Mid-sentence F§ citations** → 3.10.
- **TODOs in the main text.** `TODO(team)` appears in 5 (authors, code link) and inside §8.6. Keep the front-matter TODOs until the team fills them. §8.6's TODO moves with it to App G.6.

---

## 7 Numbers that appear more than twice, and their homes

"Home" holds the full form (table cell with SE). "Restate" lists the at most two places that may repeat it, in the draft's existing rounded form with a back-reference. Every other occurrence is cut or turned into a back-reference. Locations are section:line at `dd3e372`.

| # | claim and numbers | now at | home | restate | cut from (→ where it survives) |
|---|---|---|---|---|---|
| 1 | held-out TL gain over LegacyP: -0.0419 (200-299), -0.0396 (100-199); "0.040 to 0.042", "0.042" | Abs:17; 1.4:63-64; 5.1:462; 5.3:522-523, 532, 534, 555; 7:877; A.3:1092 | §5.4(c) table | abstract (0.042); §1.6 | 5.1:462 (→ §5.4(d)); 5.3:555 (→ App E.2); 7:877 becomes "(§5.4)" |
| 2 | TL-mix: -0.0436 | Abs:17; 1.4:65; 5.3:525, 548 | §5.4(c) table | §1.6 | abstract; 5.3:548 → App E.2 (duplicate allowed there) |
| 3 | TL-noshift: -0.0144, "0.014" | Abs:17; 1.4:70; 5.3:526, 540, 548, 555; 7:876-877 | §5.4(c) table | abstract; §7 | 5.3:555 → App E.2; Fig. 3 caption keeps 0.040 / 0.023 per budget |
| 4 | LA regimes: -0.0286, -0.0222, "0.022 to 0.029" | Abs:17; 1.4:66; 5.1:462; 5.3:548 | App E.2 table | §5.4(e) one clause | abstract; §1.6; 5.1:462 |
| 5 | RS READING / AUDIT: -0.0248, -0.0313, "0.025 to 0.031" | Abs:17; 1.4:67; 5.1:462; 5.3:583, 612; 7:877 | §5.4(e) RS table (LegacyP row) | abstract; §1.6; §7 | 5.1:462 |
| 6 | public R1 against LegacyP: -0.0017, -0.0018, "0.0015 to 0.0018" | Abs:17; 1.4:71; 5.3:527-528, 536, 649 | §5.4(c) table | abstract ("within noise"); §1.6 | 5.3:649 → §5.6 keeps "+0.0010 of R1-bf's -0.0017" |
| 7 | against Smooth: -0.0169 (TL 0-199), -0.0086 (R1-bf); "0.016 to 0.018", "0.009" | Abs:17; 1.4:72; 5.3:524, 527, 535 | §5.4(c) table | §1.6; §5.4(c) sentence | abstract |
| 8 | against EmpMean: -0.0339, -0.0464, -0.0449; "0.034 to 0.046" | Abs:17; 1.4:74; 5.3:641 | §1.6 (cluster SE); triples → App E.4 | §5.5 bullet | abstract |
| 9 | Smooth-cal: +0.0015 TUNED; -0.0043 to -0.0103 other TL; -0.0108 / -0.0100 public; "0.004 to 0.011" | Abs:17; 1.4:69; 5.3:511, 581, 608-609, 637; 7:878 | §5.4(e) RS table row; §5.5 bullet 2 | abstract (0.004 to 0.011); §1.6 | 5.3:511 (→ one clause); 7:878 (→ "(§5.5)") |
| 10 | 1PL-ship: +0.0017 TUNED; -0.0048, -0.0050 public; "0.005" | Abs:17; 1.4:73; 5.3:637, 639; 7:878 | §5.5 bullets 2 and 4 | §1.6 | abstract; §7 |
| 11 | shares 1.03 / 0.71 / 0.16 / 0.88; "0.16 to all" | Abs:17; 5.3:511, 632-636; 7:878 | §5.5 share table | none | abstract; 5.3:511; 7:878 |
| 12 | LegacyP's own optimism: 1PL-fit beats it by 0.038; EmpMean ahead by 0.0095; B0 mean prediction 0.68 against 0.42 | 5.3:611, 638; 7:877 | §5.5 bullet 3 | none | 7:877 (→ "(§5.5)"); 5.3:611's 0.31 / 0.42 → App E.4 |
| 13 | formative ALCs 0.2113 / 0.1926 / 0.1817 (recomputed); B0 0.3589 / 0.2368 / 0.2351 | Abs:19; 1.4:58-61; 4.1:305; 5.1:409-413, 419, 425, 427, 445; 5.4:663-664; 6.9:869; A.1; captions | §5.1 run table | §1.6; Figure 1 caption | abstract (→ "two formative runs"); 6.9:869 (→ "(§5.1)"); App E.7 and D.1 keep duplicates; 4.1:305 keeps 0.2113 (regime fit) |
| 14 | placement z: +0.84 / -0.44 (run 2), +0.50 / -0.82 (run 3) | Abs:19; 1.4:60, 62; 5.1:429-435; figures.md D1 | §5.1 placement table | abstract ("within about one single-run sd") | §1.6 |
| 15 | pooled reading: -0.65 / -0.78 (SE 0.51); 1.24 / 1.00 SEs ("1.0 to 1.2 SEs") | Abs:19; 1.2:37; 3.4:257; 5.1:451-456; 7:877 | §5.4(d); table → App D.4 | §7 (-0.65 to -0.78) | abstract; 1.2:37; 3.4:257 |
| 16 | lower-root reading: -1.6 (sd 1.5) against public -0.74 (1.50); audit re-read -0.98 (2.05) | 1.2:37; 3.4:238; 4.1:292, 302 (as a target); 5.1:421; 7:877 | §5.1, run-1 paragraph | §4.1 (the regime's target); §5.4(b) (-0.98) | 1.2:37 (→ "about 0.9 logit below"); 3.4:238 |
| 17 | TL realised level -1.29 (sd 1.70) | 4.1:302; 5.1:447, 453; 5.3:555; 7:877 | §4.1 | §5.4(d); §7 | 5.3:555 → App E.2 |
| 18 | pooling: ladder 0.020 (legacy); 0.0007 at formative size; 0.009 to 0.030 dense; 0.0004 to 0.001 at the platform's composition | Abs:15; 1.2:41; 1.3:47; 2.3:130-150 | §2.3 bullets and table | abstract; §1.2 (0.0004 to 0.001) | 1.3 (contributions carry no numbers) |
| 19 | single-run sd 0.026 to 0.034 ("0.02 to 0.04"); difference of two runs about 0.04 | 1.4:79; 5.1:415; 7:888 | §5.1 | §7 | §1.6 footnote |
| 20 | parent-level SE 0.009 (about 2.7 × cluster SE); per-parent -0.014 to -0.058 | Abs:17 (0.008 to 0.011); 4.3:327; 5.3:521-524, 533; 7:876 | §5.4(c) | §7 | abstract; 4.3:327 |
| 21 | gate: honest r 0.3 to 0.35; within-pair 0.25; two in three at 0.3; 0.87 at 0.4; per-pair 0.46 | Abs:21; 4.5:360-378; 6.2:769; 6.8:861; 6.9:870 | §6.1 merged table and bullets | §6.4 (one clause) | abstract; 6.2:769, 6.8:861 (→ "(§6.1)") |
| 22 | within-pair r: at most 0.15; TF-IDF 0.07; 4B 0.04 to 0.09; 14B 0.11 / 0.13 / 0.15; entropy 0.12 [0.09, 0.15]; attempts 0.32 | Abs:21; 4.5:383-393; 6:713-726; 6.8:861-862; 6.9:870 | §6.2 compact table column; full table App F.1 | §6.4 ("at most 0.15") | abstract (keeps "fails the gate" only) |
| 23 | 14B attempts' rho 0.372 [0.209, 0.516] | Abs:21; 6:725; 6.8:860; 6.9:870 | §6.3, 14B run-in | §6.4 (0.37) | abstract; App F.9 keeps the full paragraph |
| 24 | reasoning entropy within group +0.29 / +0.12 / -0.03 / +0.33 | Abs:21; 6:726; 6.8:862; 6.9:870 | §6.3, 14B run-in | §6.4 | abstract |
| 25 | item-oracle gap 0.063; oracle shares 46% / 55% | 4.1:307; 6.3:786; 6.9:870; Fig. 12 caption | §6.3 itemsig run-in and Figure 7 | §4.1 (46% / 55%) | 6.9:870 (→ "(§6.3)") |
| 26 | honest difficulty reliability 0.84 to 0.95; inflation 0.5% to 1.6%; unit ratio 0.82, 0.57 to 0.65 | 4.5:381; 6.9:870 | §6.1 | none | 6.9:870 (→ "(§6.1)") |
| 27 | hier-ship against hier-rec: +0.0030, +0.0022, -0.0023, -0.0024 | 3.4:254; 5.3:542; A.3 (as ALCs) | §5.4(a), merged table and footnote | §5.4(b) (prose, once) | 3.4:254; 5.3:542 (triples → App E.2) |
| 28 | single-subject: +0.0101 ± 0.0012 (130), +0.0094 ± 0.0016 (51); against Smooth +0.0130, +0.0117 | 1.4:75; 5.3:648; 7:881 | §5.6 | §1.6 | 7:881 (→ "(§5.6)") |
| 29 | RS closest alternatives within 0.0006 (cluster SEs 0.0003 to 0.0007) | 1.4:68; 5.3:603 | §5.4(e) | none | §1.6 (→ "no alternative passes the rule") |
| 30 | TL per-budget gains 0.135 (B0), 0.076 (B1); 0.029 of the 0.042 at B0 and B1 | 5.3:534; Fig. 6 caption | §5.4(c) sentence | Figure 3 caption | none |
| 31 | leaderboard 0.1801 / 0.1172; inversion 0.128 against 0.181 | 1.4:77; 2.5:181; 3.4:239; 5.1:464; 5.3:642 | §5.1 (one sentence) | §3.4 (0.128 against 0.181); §5.5 BLE sentence (0.1172) | §1.6; Figure A1 caption keeps 0.1801 |
| 32 | EmpMean on R1: 0.2526 against 0.2500; B1 0.3746 | 2.3:134; Fig. 1 caption:181; 5.2:473 | §5.2 table | §5.2 sentence | 2.3:134 (→ §5.2); Figure A1 caption keeps 0.2526 |
| 33 | LegacyP against Smooth: 0.0026 ± 0.0009; 0.0069 ± 0.0016 | 2.3:135 (once in the draft; also review) | §5.2 sentence (moved) | none | 2.3 |
| 34 | the precision deviation (D1) statement | 3.4:268; 4.4:339; 5.3:599; 7:887 | App C.3 | §5.4(e) (one clause) | 3.4, 4.4, 7 (→ App H.3 pointer) |
| 35 | "formative runs on different draws do not measure an improvement" | Abs:19; 5.1:415 (twice); Fig. 3 caption; 7:888 | §5.1 | abstract (implicitly); §7 | 5.1:415's second statement |
| 36 | "run 3: a regression and latency check only" | Abs:19; 1.4:61; 5.1:413; 7:887; 8.6:993; old B | §5.1 (one clause) | conduct statement | Abs; 7:887 (→ App H.3); 8.6 → App G.6 (keeps) |
| 37 | Bonferroni 2.95 / 2.955 for sixteen comparisons | 4.3:327; 5.5:693; Fig. 9 caption | §4.3 definition | App B.3 and Figure B1 (moved) | none |
| 38 | platform composition: 16 of 26 alone, 10 with one companion; per run 5, 6, 5 alone | 1.2:41; 2.3:138; 8.6:993 | §2.3 bullet 3 | §1.2 (per-run counts) | 8.6 → App G.6 (keeps) |
| 39 | run limits: 5 to 12 pairs, 1,000-item cap, 80 items | 1.1:29; 1.2:41; 2.2:117; 4.1:289 | §1.2 | §4.1 table | none (each use is local) |

---

## 8 Drafts the editor may use

Every number below is already in the draft.

### 8.1 Abstract (249 words)

> PAIEC asks for the probability that an AI system answers a benchmark item correctly, given visible attributes and 0 to 31 revealed labels, scored by Brier ALC on benchmarks absent from training. We rebuilt the organisers' streaming evaluator. An earlier replica had made pooling item difficulty across subjects look like the main lever; under the verified protocol pooling is still worth 0.009 to 0.030 ALC on dense runs, but 0.0004 to 0.001 at the platform's composition of one or two pairs per benchmark: run size, not the information set, removed it. Our predictor is a hierarchical Bayesian item-response model refitted from the revealed labels at every checkpoint. Its main lever is the prior on a new benchmark's level: three global hyperparameters, set on test-like runs built from public data to resemble the first formative feedback. On held-out test-like runs it beats our first submission by 0.042 ALC, but the gain is regime-conditional: 0.014 without the regime's synthetic date shift, 0.025 to 0.031 in regimes set to the feedback's level readings, and within noise on public runs. A smoothed mean with a calibrated prior matches it in the tuned regime; the model adds 0.004 to 0.011 over that mean in the other regimes and on public runs. Two formative runs of the shipped model lie within about one single-run sd of every regime's mean. An acceptance harness rejects every item-side signal we could transfer, from text maps to a 14B model's reasoning entropy; each negative result comes with its script.

Check the count with `wc -w` after pasting. The phrase "within about one single-run sd" restates 435's "within 1.02 single-run sds of every regime's mean ALC". If the editor prefers the exact form, use it (2 words more).

### 8.2 Contributions (§1.3, about 200 words)

> 1. **A verified replica of the official evaluator, and what the first one got wrong** (`paiec/official.py`). It reproduces the organisers' streaming client decision for decision. An earlier replica had made cross-subject pooling look like the main lever; at the platform's run composition, run size, not the information set, removes it (§2).
> 2. **A hierarchical item-response predictor fitted from `labeled` alone** (`paiec/hier.py`). It integrates every item residual exactly and reads the target along its exact posterior line (§3).
> 3. **A calibration of the new-benchmark level, and an account of what it buys** (`paiec/testlike.py`). It sets global hyperparameters only, on test-like runs built from public data, under a public guard. The shipped configuration is confirmed on held-out runs, tested at the feedback's level readings under a rule fixed in advance, and split into calibration and modelling against a calibrated smoothed mean and a plain 1PL (§3.4, §5).
> 4. **An acceptance harness and a catalogue of negative results** (`experiments/harness.py`). It maps a covariate's honest correlation with item difficulty to its chance of passing one gate. Every item-side idea we completed, each with 95% intervals on its transfer correlations, fails that gate (§6).

### 8.3 Code, data and conduct (unnumbered, after §7, about 190 words)

> **Code, data and conduct.**
>
> - *Data and code.* All results use measurement-db at a pinned revision and the organisers' client at a pinned commit (App G). Every number traces to a committed script and results file (App H), and the per-row files behind the tables are packaged for release, except the itemsig rows (App G.4).
> - *Formative feedback.* It set global hyperparameters only. Run 1 set the test-like regime's defaults and was read per pair by the audit. Run 2's pooled reading, under a rule written before it was computed but after both runs' tables had been seen, changed nothing. Run 3 was a regression check only (§5.4, App G.6). No input, prior or selection is keyed on an anonymous id.
> - *The inventory scan.* It cloned the organisers' inventory repositories to list and match file paths. Whether some files were opened is unrecorded, and nothing from the scan entered any model (App G.6).
> - *Language models.* The blind difficulty rater of §6 was the Claude model of our coding session, and its prompt is undocumented. The code was developed with Claude Code.

### 8.4 Headline table (§1.6): twelve rows

Header: "Every row is hier-ship unless named. Δ = hier-ship − comparator, negative favours hier-ship; cluster SE in parentheses (§4.3)." Then the §2.3 back-reference note of 2.3.

| what | value | where |
|---|---|---|
| formative runs, single draws: run 1 LegacyP; run 2 hier-ship; run 3 hier-ship's rebuilt archive (recomputed from its table) | ALC 0.2113; 0.1926; 0.1817 | §5.1 |
| Δ LegacyP, TL held-out runs 200-299 | -0.0419 (0.0037); parent-level -0.038 ± 0.008 | §5.4(c) |
| Δ LegacyP, TL-mix | -0.0436 (0.0039) | §5.4(c) |
| Δ LegacyP, TL-noshift | -0.0144 (0.0024) | §5.4(c) |
| Δ LegacyP, RS READING / AUDIT (realised level -0.63 / -0.93) | -0.0248 (0.0037) / -0.0313 (0.0049) | §5.4(e) |
| Δ LegacyP, R1-bf / R1-pu | -0.0017 (0.0021) / -0.0018 (0.0017) | §5.4(c) |
| Δ Smooth-cal, TUNED / R1-bf / R1-pu | +0.0015 (0.0019) / -0.0108 (0.0014) / -0.0100 (0.0018) | §5.4(e), §5.5 |
| Δ 1PL-ship, TUNED / R1-bf / R1-pu | +0.0017 (0.0013) / -0.0048 (0.0007) / -0.0050 (0.0007) | §5.5 |
| Δ Smooth, TL 0-199 / R1-bf | -0.0169 (0.0020) / -0.0086 (0.0030) | §5.4(c) |
| Δ EmpMean, TUNED / R1-bf / R1-pu | -0.0339 (0.0030) / -0.0464 (0.0027) / -0.0449 (0.0030) | §5.5 |
| alternatives at the feedback's readings | none gains the rule's 0.002 | §5.4(e) |
| Δ LegacyP, the single-subject benchmark | +0.0101 ± 0.0012 (SE over 130 appearances) | §5.6 |

All cluster SEs are the middle term of the triples now at 63-75, or the RS table's brackets.

---

## 9 Checks for the editor before handing back

1. **Numbers survive.** Every numeric token of the old draft must still be in the new draft, or be in `docs/findings.md` with a pointer left in the report:

   ```bash
   git show dd3e372:docs/report/draft.md > /tmp/old.md   # or the lane's scratch dir
   python - <<'EOF'
   import re
   tok = lambda s: set(re.findall(r'(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?', s))
   old, new = open('/tmp/old.md').read(), open('docs/report/draft.md').read()
   fnd = open('docs/findings.md').read()
   gone = sorted(tok(old) - tok(new))
   print(len(gone), 'tokens gone;', [t for t in gone if t not in fnd][:200], '<- not in findings either')
   EOF
   ```

   Each token on the last list must be restored somewhere in the report. Most will be in §8 or appendices if they were moved rather than cut.
2. **No sign or rounding changed.** For the 39 rows of section 7, the home value must equal the old value character for character.
3. **Budgets.** `python /private/tmp/claude-501/-Users-nikitapolomosnov-PycharmProjects-PAIEC/9df8ecfa-42c0-4508-919c-f794f1770909/scratchpad/p2/plan/wc.py` (the counter used here; it reads `docs/report/draft.md`) on the new draft gives sections 1 to 7 at or under 7,500 prose words and the abstract at or under 250 (`wc -w`).
4. **Cross-references.** No dangling `§`, `Appendix` or `Figure` reference: `grep -n "§[0-9]" docs/report/draft.md` against the new headings. No review tag in sections 1 to 7: `grep -nE "P1a|P1\.[0-9]|\b[WQ][0-9]+\b|\bD1\b|commit D"` lists only appendix lines.
5. **Tests and figures.** `python -m pytest -q` is green, and `python tools/report_figures.py --check` prints ok (also after any edit to `figures.md`, which the check does not read).
6. **Hard rules.** Nothing committed, tagged or pushed; `.env` never read; no training or model loading.
