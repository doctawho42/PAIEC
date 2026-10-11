# Amendment 4: the calendar revision of 2026-10-11/12

Written 2026-10-11 (Moscow evening; 17:45 to 18:30 UTC) against repository HEAD
2bfcd17. Nothing of amendment 4 was committed before this revision, so none of
its earlier dates was binding. This log covers `amendment4_short.md` ("A4"),
`p2_pool_eval.md` ("PE") and `impl_plan.md`. No git write, no ALC or Brier of
any predictor, no item text, and no formative feedback beyond what PE §11 lists
went into it. The pool files were hashed, not parsed.

## The calendar

The author's frame of 2026-10-07 was a gate decision by 2026-10-16 and the
regression submission by 2026-10-20. The new calendar falls inside it. Every
stop is 23:59 UTC, and missing any of them leaves archive-3 the final.

| stop | old (2026-10-07 draft) | new |
|---|---|---|
| amendment 4, PE and `docs/plans/pool_design/` committed | Thu 10-08 | Mon 10-12 |
| lock (digests of the implementation, the constants block, the bundles) | target 10-09, hard stop 10-10 | hard stop Tue 10-13 (no target) |
| S and RS scored, selection committed; C scored, the gate read once and committed | gate by 10-13 (planned 10-10 to 10-11) | both by Thu 10-15 (planned S 10-14, C 10-15) |
| on a pass: archive-4 built, through S3, announced (amendment 5) | 10-15 | Fri 10-16 |
| on a pass: the one upload | 10-16 | Sat 10-17 |
| final recorded | 10-20 | 10-20 (unchanged) |

## The dense run counts

**Old rule** (PE §5.4, 2026-10-07): 30 dense runs a regime in S and 50 in C.
If the lock slipped past 2026-10-09 23:59 UTC, which was four days before the
gate's hard stop, they fell to 20 and 40.

**Decision.** The counts are now 20 (S) and 40 (C) from the start, wherever the
lock lands, and the lock-slip rule is removed. This was decided before the lock
and before any task was scored, from PE §12's budget alone. No outcome exists
yet.

**Why.**

1. The old rule cut the counts whenever the lock left less than four days
   before the gate's hard stop.
   * On the new calendar the lock's hard stop leaves two days: one day for each
     of the two sequential scoring stages, and no slack.
   * Even the earliest realistic lock leaves less than four days. The
     implementation needs about 18 hours of session time (impl_plan, totals),
     and it starts after the amendment is committed on 2026-10-12.
   * So the old rule's condition is always met.
2. The budget gives the same answer.
   * The criterion: each stage's wall time at three times PE §12's 1.5x figure
     must fit in 24 hours. The factor of three is the critic's 2 to 3 for the
     loads seen on 2026-10-07. The criterion must hold for every dense cost up
     to the timing adjustment's 30-process-hour trigger.
   * Post-lock wall hours on 3 shards, each cell the 1.5x figure / three times
     it:

   | dense runs (S, C) | dense cost | S and RS | C core with `summarise` |
   |---|---|---|---|
   | 60, 100 | planned | 5.9 / 17.6 | 4.1 / 12.2 |
   | 60, 100 | 1.875x (the trigger) | 8.2 / **24.7** | 6.1 / 18.2 |
   | 40, 80 | planned | 5.0 / 14.9 | 3.6 / 10.9 |
   | 40, 80 | 2x | 6.8 / 20.3 | 5.4 / 16.3 |
   | 40, 80 | 2.577x (the trigger) | 7.8 / 23.4 | 6.5 / 19.5 |

   * At 30 and 50 runs a regime, S and RS run over the day just below the
     trigger, where no adjustment acts. At 20 and 40 both stages fit, all the
     way up to the trigger.
   * Core total: 67,050 process-seconds (18.6 process-hours), against 76,800
     (21.3) before. The dense share is 39% (was 47%). At twice the dense cost
     the core is 25.8 process-hours (was 31.2).
3. The timing adjustment stays.
   * If the projected core exceeds 30 process-hours, the dense item cap drops
     from 120 to 80.
   * Its second step, dense counts to 20 and 40, is now the plan itself.
   * Formative-size counts and the RS guard (520 tasks) are unchanged.

**Cost of the decision.** The D-READING and D-FLAT loss bounds rest on fewer
runs: (d) in C and (E2) in S, both at +0.001. A noisier point estimate fails a
bound more often when the true effect is null, which errs towards archive-3.
The 2026-10-07 design already accepted this for a slipped lock.

## Item-table facts recorded (PE §3.1, §3.7)

These come from the new `a4final/items_present.py`, which writes
`a4final/items_present.json` (counts, sizes and digests only), measured
2026-10-11 17:53 UTC.

* `data/pool/manifest.json`:
  * `fetched_utc` 2026-10-08T04:04:26Z;
  * sha256 `ff5482069780ea78…`;
  * `items_skipped` lists cseo_bench and qatch.
* Item tables cover 250 of the 255 entries:
  * missing: cseo_bench (ordinal, 5.69 GB) and qatch (continuous, 0.71 GB), both public and both outside L and E;
  * the three withheld entries (imagenet_hard, swe_together, visual_memory) have a catalogue row only.
* L (137 benchmarks; rule of PE §3.3 applied to `benchmarks.csv`):
  * items present for all 137;
  * all 548 files of L (four tables each) match the manifest in size and sha256 on disk.
* Item bytes: L 2,281,481,006, E 2,173,366,701, all 250 together 4,917,276,705.
* `data/pool` occupies 7.42 GB.
* The benchmarks, subjects and responses entries match the pre-items manifest (`a58a23fb…`) for all 255 entries.
* Free disk: 11,242,594,304 bytes (11.24 GB, 10.47 GiB).
* `--items-skip` is committed at 2bfcd17.

## Changes, file by file

### amendment4_short.md

* **Heading.** The date changes from (2026-10-07) to (2026-10-12), the day of the commit.
* **Preamble.** It now says the text was written on 2026-10-07, revised on 2026-10-11/12 and committed on 2026-10-12. The quoted sha256 of PE changes from `c5c976c6…` to `3e080097…` (full value below).
* **§4.1, heading of the decisions list.** It now notes that decision 6's dates were revised on 2026-10-11.
* **§4.1, decision 2.** It adds that the fetch is done: every item table of L is present, and only cseo_bench and qatch (both outside L) were skipped.
* **§4.1, decision 6.** The old text was: "the lock must be committed by 2026-10-10 23:59 UTC". The new text lists §4.5's hard stops: the lock 10-13, the selection and the gate 10-15, archive-4 10-16, the upload 10-17.
* **§4.2.** Three supersessions are added:
  * P2 §3's dates for archive-4: "by 2026-10-15" and "the submission of section 4 on 2026-10-16", beside "chosen on 2026-10-17 ... does not ship";
  * P2 §4's "no later than 2026-10-16", which gives way to the upload by 2026-10-17 23:59 UTC.

  Without these, the 10-17 upload would collide with the plan.
* **§4.3.** The dense counts change from "30 a D- regime in S, 50 in C" to "20 ... 40", citing PE §5.4 and §12.
* **§4.4, `RULE_TEXT` Reading 1.** It now carries every stop:
  * build, checks, validator, S3 and amendment 5 by 2026-10-16 23:59 UTC (this date was not in the rule before);
  * the upload "once, by 2026-10-17 23:59 UTC" (was 2026-10-16);
  * the final by 2026-10-20 23:59 UTC (unchanged).
* **§4.4, `RULE_TEXT` Reading 4.** The "not locked in time" stop changes from 2026-10-10 to 2026-10-13. The "not read in time" stop changes from 2026-10-13 to 2026-10-15, and now names the selection and then the gate.
* **§4.5, the table.** It is redated: Mon 10-12 commit, Tue 10-13 lock, Wed 10-14 to Thu 10-15 scoring and gate, Fri 10-16 archive-4, Sat 10-17 upload, by Tue 10-20 final.
* **§4.5, hard stops.** These are rewritten for the new calendar. The lock-slip parenthesis is replaced by the fixed dense counts.
* **§4.5, outcomes.** Two outcomes are added:
  * archive-4 not built, through S3 and announced by 10-16, or not uploaded by 10-17: archive-3 stays and nothing is submitted;
  * run 4 not read, or the final not recorded, by 10-20: archive-3 stays.

### p2_pool_eval.md

* **Preamble.** It now says "revised on 2026-10-11/12" and names the sections the revision touched.
* **§3.1.**
  * The fetch is now described as run with `--items-skip` for cseo_bench and qatch (committed at 2bfcd17).
  * The manifest's `fetched_utc` and `items_skipped` are named.
  * The three older tables are unchanged from the pre-items manifest.
  * The manifest sha256 `ff548206…` is recorded.
* **§3.7.**
  * The facts above are recorded, measured 2026-10-11 17:53 UTC and written 2026-10-11/12, from `a4final/items_present.py`.
  * The column header no longer reads "(*planning*)". Only the two multiple-choice rows keep *planning* marks.
  * The free-disk row is the measured value, with the 2026-10-07 figure kept beside it.
* **§5.4, the counts table.** The D- column changes from 30 each (60) to 20 each (40) in S, and from 50 each (100) to 40 each (80) in C.
* **§5.4, the lock-slip bullet.** It is replaced by "The dense counts follow the calendar", which gives the reasoning above.
* **§5.4, the adjustment.** It keeps only the item-cap step, and notes that its second step is now the plan.
* **§10.**
  * archive-4's sha256 is recorded by 2026-10-16 (was 10-15).
  * Steps 1 to 3 must be done by 2026-10-16 23:59 UTC, or archive-3 stays.
  * The upload happens once, by 2026-10-17 23:59 UTC (was 10-16), and notes the move of P2 §4's date.
  * A missed upload leaves archive-3.
* **§11, the App G.2 report text.** "fetched 2026-10-07 ..., with item tables" becomes "fetched ... on 2026-10-07 and 2026-10-08, with the item tables of every public benchmark but cseo_bench and qatch, which it does not use". The author confirms this wording in commit 1, as before.
* **§12.** The budget is redone at 40 and 80 dense runs:
  * 67,050 process-seconds (18.6 process-hours);
  * 5.7, 3.6 and 1.4 hours of wall;
  * dense share 39%, 25.8 process-hours at 2x the dense cost, trigger at about 2.6x.

  The 2026-10-07 figures are kept in a parenthesis, and a new "Wall time against the calendar" paragraph holds the table above.
* **§13.**
  * The schedule is redated: an added row for 10-08 to 10-11, commit 1 on 10-12, the lock 10-13, S 10-14, C and the gate 10-15, archive-4 10-16, upload 10-17, final by 10-20.
  * The `--pool` build path is written on 10-14, while stages run, because the old slack days are gone.
  * The hard stops are rewritten.
  * `dates_log.md` is added to the folder list.
* **§15, the recommended-changes row.** It now says the reduced dense counts are the plan on this calendar.

### impl_plan.md

* **Header.** It adds the revision of 2026-10-11/12 at HEAD 2bfcd17.
* **Step 0.1.** "Wait for the item fetch" becomes "The item fetch is done", with the facts above. The quoted PE sha256 changes from `c5c976c64ce96cf4…` to `3e0800974e9f5808…`.
* **Step 0.2.** The list of choices to confirm now includes the calendar and the dense counts.
* **Step 0.3.** Commit 1 is dated 2026-10-12.
* **Step 5, the constants block.** The counts change from S 80/30 and C 120/50 to S 80/20 and C 120/40. The adjustment is the item cap only.
* **Step 5, the `lock` stage.** The lock-slip check is replaced by a check of the lock commit's time against the 2026-10-13 hard stop.
* **Step 8.** The deadline is the hard stop 2026-10-13 23:59 UTC. There is no target and no slip rule.
* **Steps 9 and 10.** The wall estimates are redone (5 and 3.6 hours at 1.5x, 15 and 11 hours at three times that) and the steps are dated 10-14 and 10-15. Commit 3 is due no later than 10-15, in practice by the end of 10-14. Commit 4 is due by 2026-10-15 23:59 UTC (was 10-13).
* **Step 12.** The window is 2026-10-16 to 2026-10-17 (was 10-14 to 10-16). Steps 1 to 3 must be done by 10-16 23:59 UTC; the upload happens once, by 10-17 23:59 UTC.
* **Step 13.** It starts after the gate's commit (was "2026-10-12 onward").
* **Totals.** The table is redated, and the scoring wall time becomes 8.6 hours at 1.5x (26 hours at three times that).
* **Risks.** The item-fetch risk is closed. A new risk, "The calendar leaves no slack day", covers about 18 hours of work in two days before the lock, and one day per scoring stage.

## Digests

| file | before | after |
|---|---|---|
| p2_pool_eval.md | `c5c976c64ce96cf40148016d9b2d2b0fcb21f5ef1314a77b06c4cbcd416b2e7c` | `3e0800974e9f580823a0bc56a97c80e3b7dd769959d4b3de5d1fcceceb14f3c5` |
| amendment4_short.md | `70b0cc19ce13daa3ce39fa7a9d962b5279bb5996e56f9ee265e6956e4fbb6240` | `c655e4828a0bae8252ea4f8f4a3cabb08f26870ac657366452d3a20340558e3e` |
| impl_plan.md | `c568fc5b82ff7ca29f90ede6f3452a27d265d44d59bb66c076067c2f6dd6a4d7` | `1a95760d177e0ae42b5ac10c189e0fe72262faade6be804c7c123c384d8cac00` |

* A4 quotes the new PE digest in full, and impl_plan quotes its first 16 hex.
* PE quotes neither file's digest, so the chain has no cycle.
* Any later edit to PE needs a new digest in both files.

## Checks run

* Grepping the three files for the old calendar:
  * 2026-10-09 or 2026-10-10 appears once, in PE §5.4's description of the 2026-10-07 rule, which is kept on purpose;
  * 2026-10-08 appears only as the date of the fetch and its manifest (PE §3.1, §11 and §13; impl_plan step 0.1);
  * "lock (target)", "80/30" and "120/50" do not appear.
* Every hard stop reads the same in A4 §4.1 decision 6, `RULE_TEXT` Readings 1 and 4, A4 §4.5, PE §5.4, §10, §12 and §13, and impl_plan §§5, 8, 9, 10 and 12:
  * the lock 10-13;
  * the selection and the gate 10-15;
  * archive-4 built, through S3 and announced 10-16;
  * the upload 10-17;
  * the final 10-20.
* The P2 plan quotations in A4 §4.2 were checked against `docs/plans/p2_final_plan.md` lines 62-66 and 71-72.

## For the author to confirm (in commit 1)

* Decision 6 as rewritten. The calendar itself came from the workflow's task text.
* The amendment's heading date, 2026-10-12.
* The dense counts of 20 and 40, fixed from the budget, and the removal of the lock-slip rule.
* The App G.2 wording on the fetch dates and the two skipped tables.
* The expanded supersession of P2 §3's and §4's dates.
* That about 18 hours of implementation fit into 2026-10-12 and 2026-10-13 before the lock's hard stop.

## Not changed

* `eval.md`, `critic.md`, `data.md`, `code.md` and `amendment4_draft.md` keep their 2026-10-07 dates. They are the record of the superseded designs.
* `docs/plans/p2_final_plan.md` and every file outside this folder are untouched.
