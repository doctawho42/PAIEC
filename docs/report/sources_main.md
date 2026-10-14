# Source keys of the main text (report v2)

The main text of report v2 (the title block to the References) carries no HTML comments, so that the outline's word rule, whitespace tokens from the line after `## Abstract` to the line before `## References`, counts only what the PDF shows. This file holds the source keys of its numbers instead: one entry per paragraph, table or list, under the main-text section it belongs to, with the opening words of each block it covers. Most entries are the comments of the assembled main text of 2026-10-14, re-anchored where one comment covered several paragraphs; §7 and the conduct section were rewritten at the final assembly and keyed then.

Conventions. A key is a committed file and a key path in it; braces list sibling keys, `<R>` stands for each regime and `<C>` for each pool candidate (C1, C1m, C2), and a value in parentheses after a key is the stored value or a note on it. F§ is a section title of `docs/findings.md`, P§ a section of `docs/protocol.md`; "report v1.0.1" is the previous report, `docs/report/draft.md` at git tag `v1.0.1`; "derived" marks a number computed from stored values (App H.1). Numbers that the text sources inline (F§, P§, a named file or an appendix) need no entry here. No entry holds item text, an anonymous hidden-benchmark id or anything keyed on one.


## Abstract

- "PAIEC asks for the probability that an AI system answers": results/hidden_runs.json main.v_run{1,2,3}; decomposition.hier_pooled.alc_minus_v (0.0171, hier-ship's two runs pooled); main.ideal_lower_team_hier, main.ideal_prior_weighted_{lo,hi}; item_level.run3.share_range[0], item_level.run2.share_range[1]; main.legacy_minus_hier_vadj{,_se}; main.held_out_tl_shipped_minus_legacy (-0.041911, stored as hier-ship minus LegacyP), main.held_out_tl_cluster_se; main.v_run_strat_{mean,sd}; entry.leaderboard.best_entry; curve.shape.platform_steps_mean_se."B7->B31"; results/gate_and_ci.json scale; results/harness_thresholds.json thresholds.honest; the refit (last sentence): results/pool_fits.json selection.L (137 eligible benchmarks); results/pool_eval.json selection.outcome; F§ "Amendment 4: hier's priors refitted on the expanded pool; no candidate carried (P2)".

## 1.2 Why evaluation is the hard part

- "**The outside evidence is three small draws.**": results/hidden_runs.json tables.run*.{pairs, benchmarks}; results/pooling_decomposition.json platform_mix.counts.{alone, two} (16 of 26)

## 1.3 Contributions

- "1. **A root-free reading of formative scores**": contribution 1: results/hidden_runs.json replica.noise_summary.{ship_alc_sd_range, ship_alc_minus_v_sd_range}; contribution 2: replica.placement_z.R1B."run1 (LegacyP)".z_by_budget[0] (+8.87); contribution 3: results/factorial_rs.json summary.interaction.testlike.D_range; contribution 5: results/pool_eval.json lock_time.commit (831ef7d), selection.outcome; F§ "Amendment 4: hier's priors refitted on the expanded pool; no candidate carried (P2)"

## 2.1 Data

- "We use only measurement-db, the organisers' public training release": results/data_counts.json benchmarks.<b>.eligible_pairs; results/pins.json checks[1].measurement_db.sha; F§ "Amendment 4: hier's priors refitted on the expanded pool; no candidate carried (P2)" (252 public benchmarks, revision 54074453)

## 2.2 The protocol as verified

- "Our replica (`paiec/official.py`) matches the client": results/pins.json checks[0].{status, measurement_db.matches_pin, paiec_baseline.matches_pin} (the pins, App G.2); platform resources: the competition page [PAIEC organisers 2026], Runtime and Q&A, recorded in docs/report/report_requirements.md (runtime-resources row, read 2026-10-14), and docs/plans/p2_final_plan.md section 1, decision 1 (a disclosed exception to the convention, title block and App H.1)

## 2.3 What the first replica got wrong

- "Our first replica (`paiec/evaluator.py`": results/pooling_decomposition.json reading."ship - ship_own" (dense pair, dense benchmark: mean and SE over n = 4 benchmarks, -0.030022 (0.009934) and the benchmark scope's; platform mix: pair-cluster SE); platform_mix.counts

## 3.1 The identity

- "Run 1 checks both, since LegacyP's predictions moved": results/hidden_runs.json identity.run1.per_pair_range_max; identity.run1.pairs_within_rounding_bound
- "hier spreads its predictions within a pair": results/hidden_runs.json replica.validation_summary.{b0_bias_range, b0_sd_range, b31_bias_range}

## 3.3 The item-level share and what a run's composition allows

- "Pair-level tables do not identify the item oracle": results/hidden_runs.json item_level.ratio_range
- "A predictor that gives one probability per pair": results/hidden_runs.json main.v_run_strat_mean; main.v_run_strat_sd; main.v_run_iid_mean; main.v_run_iid_sd; main."P_v_le_0.117_strat"; main."P_v_le_0.117_iid"; run_spread.pairs.v; run_spread.benchmarks
- "On a synthetic population like the hidden pairs": results/hidden_runs.json requirement.summary.hidden_like_req_r_known; requirement.summary.hidden_like_req_r_cal (tau 2.0, 2.6, 3.2)
- "The value 0.117 is the leaderboard's best formative score": results/hidden_runs.json entry.leaderboard.best_entry; entry.leaderboard.read

## 3.4 What the reading can and cannot claim

- "Adjusted for $V$, runs on different pairs": results/hidden_runs.json replica.placement_summary.note (the sds are contribution 1's)
- "hier-ship's $V$ is an estimate": results/hidden_runs.json tables.run2.benchmarks; tables.run3.benchmarks; run_spread.distinct_pairs.{pairs_in_two_runs, repeat_runs}; decomposition.hier_pooled.pair_se_note

## 4.1 Test-like runs

- "We build test-like (TL) runs"; "The regime-sensitivity (RS) study redraws"; "TL is partly in sample for run 1": results/testlike_check.json summary.check.default.regime.pair_logit.{mean, sd, cluster_se}; summary.check.default.ppc.predictor.{3,7,15}.z (0.092, 0.093, 0.499); results/regime_sensitivity.json summary.realised.<R>.mean; paiec/testlike.py Regime.exclude (swe_rebench excluded, so four parents)

## 4.3 The acceptance harness

- "The gate's scale comes from degraded oracles": results/harness_thresholds.json thresholds.honest, acceptance; results/gate_and_ci.json gate (pass probability 0.68); report v1.0.1 §6.1 for the r = 0.2 comparison (harness_thresholds.json)
- "*Figure 1.*": panel (a) results/harness_thresholds.json (the degraded-oracle lines, their draws and passes, the honest oracle); panel (b), the 44 covariates: tools/report_figures.py COVARIATES, whose families read the `harness` blocks of results/llm4b_close.json (4), results/hidden_state_probe.json (9), results/strong_llm_eval.json (15, and 6 under `entropy`), results/finetune_encoder.json (4) and results/itemcov_eval.json (6); docs/report/fig/manifest.json figures.gate_curve.inputs lists those six files with their sha256; docs/report/figures.md D3

## 4.4 When the rules were fixed

- "The gate first appears in commit": results/gate_and_ci.json evidence.gate_first_commit; docs/plans/p2_final_plan.md (written and committed 2026-10-06 before any check ran); results/pool_eval.json lock_time.commit (831ef7d)

## 5.2 Inference

- "A joint mode over all effects": tests/test_hier.py test_item_residual_is_integrated_not_maximised (160 of 200, item variance 8.7, the joint mode's 0.70 and the integrated 0.8, each asserted to within 0.01); the shipped widths as in §5.3's key
- "The Laplace posterior"; "S1 checked hier's approximations": paiec/hier.py module docstring (the under-reaction, 0.013 to 0.02; 20 Gauss-Hermite nodes); results/ablations_at_level.json summary.regimes.<R>.vs_ship.noline.equal.{D, cluster_se}; results/dense_b1_check.json gate.{difference, interval_95_cluster, mc_se}

## 5.3 Offline prior and the level prior

- "The offline prior (`paiec/prior.py`)"; "Its level prior, set for the hidden test"; "The feedback may set *global* hyperparameters": the shipped hyperparameters (0.1, 2.382, 1.018, 2.671, 1.542, 0.01, 0.5): F§ "Verdict: ship hier with the level moved down" and results/pool_fits.json hyper.A3 (archive-3's bundle; submission/prior.json itself is gitignored); submission/model.py LEVEL; results/level_calibration.json summary.selected.order (hier-argmax), summary.ship.config (hier-rec); results/level_audit.json mild (hier-ship). The C1m sentences are the wording docs/plans/p2_pool_eval.md §11 fixes for report v1.0.1 §5.3 (author-confirmed in the commit of amendment 4).

## 5.4 Run time and robustness

- "`predict` caches its fits": results/regime_sensitivity.json summary.latency.all.ship.{mean_ms, max_s}; results/stress_archive.json gate.pooled.{calls, no_exception_escapes, finite_and_in_open_unit, p99_window_s, max_window_s}; gate.parts_alone.fuzz.p99_window_s

## 6.1 The hidden runs decomposed

- "hier-ship pays 0.0130 and 0.0208"; "Sources: `experiments/hidden_runs.py`": results/hidden_runs.json tables.run{1,2,3}.alc; decomposition.run{1,2,3}.{v, alc_minus_v, b0_term, b1_b15_terms, b31_term, share_b0_b1_b3}; vadjusted.run{1,2,3}.se; item_level.run{1,2,3}.share_range; item_level.ratio_range; replica.placement_raw.TUNED."runs 2+3 (hier)".hidden_alc; decomposition.hier_pooled.{v, alc_minus_v, pair_se.alc_minus_v_pair_se}
- "Pooled over runs 2 and 3": results/hidden_runs.json decomposition.hier_pooled.excess_by_budget; ideal.summary_hier.{lower_roots_team_prior, prior_weighted_range, observed_hier}
- "At $B_0$ hier-ship pays about what": results/hidden_runs.json ideal."<prior> | runs (2, 3)".mix; curve.shape."pairs_B3_gt_B1_plus_0.01"; nonmonotone."prior=truth logit N(-0.7,1.42)", nonmonotone."prior=truth logit N(-1.29,1.7)", nonmonotone."P(>=8 of 17 with B3>B1+0.01)".{0.2, 0.25}; derived pair-SE gaps 6.77 and 5.38 (App D.7)
- "What remains is mostly item level": results/hidden_runs.json item_level.run{2,3}.{v_minus_item_oracle_range, share_range}; late_gap.run{2,3}.{b31_minus_v, ece31_sq, r31, floor_pure_rate_learner_B31_b0_convention}

## 6.2 The test-like regimes checked against the hidden runs

- "Out of sample, run 1 rejects public runs": results/hidden_runs.json replica.placement_summary.{run1_vs_R1B_B0_B1, run1_vs_R1P_B0_B1, run1_vs_test_like_range}; replica.placement_z.<R1B|R1P>."run1 (LegacyP)".z_alc_minus_v; replica.placement_raw_summary.side_by_side_z_alc."run1 (LegacyP)"; replica.noise.<READING|TUNED>.legacy.alc_minus_v
- "hier-ship's ALC minus $V$, never a tuning target": results/hidden_runs.json replica.noise.<R>.ship.{v, alc, alc_minus_v}; replica.placement_z.<R>."runs 2+3 (hier)".{z_by_budget, z_alc_minus_v}; decomposition.hier_pooled.bins
- "| regime (realised level) | $V$ |"; "Sources: `results/hidden_runs.json` (`replica.noise`": results/hidden_runs.json replica.noise.<R>.{ship,legacy}.{v, alc, alc_sd, alc_minus_v, alc_minus_v_sd}; replica.placement_z.<R>.*.z_alc_minus_v; results/regime_sensitivity.json summary.realised.<R>.mean
- "Late budgets are where the regimes miss.": results/hidden_runs.json curve.shape.platform_steps_mean_se."B7->B31"; curve.test.replica.<tl|r1b|r1p>.{"B7->B31_mean_of17", "P(mean >= platform)"}; late_gap.vs_replica_summary.hier_pooled.hidden_minus_replica_range_test_like

## 6.3 The replica's results on one run set

- "Table 3 gives the replica's comparisons": results/ship_confirm.json table (tl, legacy, 200-299: stored as hier-ship minus LegacyP -0.041911, cluster SE 0.003736; parent mean and SE; per-parent range); table (tl no shift, legacy); results/regime_sensitivity.json summary.vs_ship.<R>.legacy.exact.{D, cluster_se}
- "Adjusted for $V$, the hidden runs give": results/hidden_runs.json vadjusted.{legacy_minus_hier.{v_adjusted, se, raw}, agreement.z}
- "| comparator | tuned (-1.28) |"; "Sources: `results/regime_sensitivity.json`": results/regime_sensitivity.json summary.vs_ship.<R>.{legacy, smooth, smcal, eb_fit}.exact.{D, cluster_se}; summary.configs.<R>.ship.ALC; results/baselines_p1.json summary.regimes.{TUNED,R1B,R1P}.vs_ship.{empmean, rasch}.{D, cluster_se}; results/e1_baselines_rs.json summary.t3.{empmean, rasch}.columns.<R>.{D, cluster_se} (AUDIT, MIXTURE, READING, FLAT; App E.4); results/hidden_runs.json vadjusted.legacy_minus_hier
- "In the tuned regime EmpMean is ahead": results/e1_baselines_rs.json summary.regimes.TUNED.vs_legacy.empmean.{D, cluster_se}; summary.t3.empmean.columns
- "Calibration and modelling form a paired": results/factorial_rs.json summary.{calibration_in_hier, calibration_in_smooth, model_at_default, model_at_calibrated, interaction}.by_regime.<R>.{D, cluster_se}; contrasts.<R>.interaction.per_parent; results/regime_sensitivity.json smcal.{selection, guard_failed}
- "Two tests found no replacement for LEVEL.": results/ablations_at_level.json summary.regimes.<R>.vs_ship.{sd_x085, sd_x07, sd_x05, sd_x07_noline, sd_x085_mp, wide35, wide50, eb_ship, sd085_abs}.{equal, hidden_v}.{D, cluster_se}; rule.{outcome, would_pass, shipped}; rule.thresholds.gain (App B.3)

## 6.4 What does not transfer

- "| family | ideas |"; "Sources: App F.1 and F.8": results/gate_and_ci.json scale, ci; results/harness_thresholds.json thresholds.honest; results/posthoc_family.json family.best_tl_lopo.{delta, se, variant}, family.verdicts; results/acquisition_official.json summary.range.policies_minus_two_arm_mean, summary.range.any_meets_tl_bar_vs_two_arm_mean (App F.8)
- "Two facts explain Table 4.": results/itemsig_eval.json (0.063, report v1.0.1 §6.4); results/hidden_runs.json item_level.run{1,2,3}.v_minus_item_oracle_range (0.0703 to 0.1010 over the three runs); results/posthoc_family.json family.parameter_spread

## 7 The pool evaluation

- "Benchmarks of the expanded release (§2.1) were assigned": F§ "Amendment 4: hier's priors refitted on the expanded pool; no candidate carried (P2)" (the random pool assignment, the candidates, the rule, the outcome, "What this does not show", "Deviation, disclosed: C-extra and describe-hidden did not run"); results/pool_fits.json selection.{L, E} (137; 132, L less the five original benchmarks, which enter only the RS guard: App C.6), hyper.{C1,C1m,C2}.{mu0, sigma_mu} (-2.5, -2.046, 0.131; 2.5, 2.35), mu0_m, b0_centre_matching_set_minus_A3.C1m."plus_1.25y:all" (0.00002, the matching); results/pool_eval.json lock_time.{commit, utc} (831ef7d, 2026-10-13T22:48:55Z; the rule itself first committed in 3d38232, F§ "Amendment 4", the record's timeline; the push: the clone's remote-tracking reflog, App C.3 and H.3); lock.{mu0_m, c2_literal}; split.summary.{S, C}.benchmarks (44, 88); split.rule.unit (family); selection.rule.{carry, twin, loss, dense, rs_mean, rs_public, rs_testlike}; selection.{outcome, Q, X_star, utc}; selection.candidates.<C>.{M, eligible, holds, E1.{mean, regimes.<R>.{D, bound, ok}}, E2.<R>.{D, bound, ok}, E3.ok, E4.ok} (C1 M -0.000712, C1m M -0.001739, C2 M +0.001926; C1m E2."P-AUDIT-DS".D +0.000166; C2 E1.mean +0.005174); selection.stats.C1m."P-AUDIT-DS".{SEg, SEf} (0.001031); selection.stats.C2."P-FLAT".D (-0.012232); selection.table.<C>.invalid (0); commit 365664c (the selection); the twin power (0.25 to 0.5 with no true effect): App C.6 "Power", from docs/plans/p2_pool_eval.md section 7 (a planning figure); the deviation and the author's decision of 2026-10-14: F§ "Deviation, disclosed: C-extra and describe-hidden did not run".
- "*Table 5: the selection on S"; "| regime (realised level) | condition |"; "Source: `experiments/pool_eval.py`": results/pool_eval.json selection.stats.<C>.<R>.{D, SEg, runs} (80 per formative-size regime and twin, 20 per dense regime); selection.candidates.<C>.{M, E1.mean, E1.regimes.<R>.ok, eligible}; selection.rule (the bounds); regimes.regimes.S.<R>.cal.mean (P-READING -0.644, P-AUDIT -1.019, P-TUNED -1.217, P-MIXTURE -0.609, P-FLAT -0.022, D-READING -0.716, D-FLAT -0.244); F§ "Amendment 4: hier's priors refitted on the expanded pool; no candidate carried (P2)" ("Checks and provenance": 520 RS tasks). Rounding: 4 decimals, half up, from the stored floats, except C1's P-AUDIT and P-TUNED, given to five decimals because they round to zero at four (selection.stats.C1."P-AUDIT".D +0.000033, selection.stats.C1."P-TUNED".D -0.000027).

## 8 Threats to validity

- "1. Five public benchmarks carry the replica": 1: results/ship_confirm.json table (tl, legacy, 200-299); 2: ship_confirm.json table (tl no shift); results/testlike_check.json summary.check.default.regime.pair_logit.mean; results/hidden_runs.json vadjusted.legacy_minus_hier.se; 3: F§ "Item signal from the pair's own labels", "Against the item oracle" (46%; public 55% benchmark-first, 56% pair-uniform); 5: results/e1_baselines_rs.json summary.regimes.TUNED.vs_legacy.empmean.{D, cluster_se, parent_level} (-0.00947 / 0.00491 = 1.9 cluster SEs, derived; parent level -0.0029 ± 0.0120); F§ "BLE cannot run here" (569,838); 6: results/gate_and_ci.json single.seed11_current_library.R1B.vs.legacy; 7: results/inventory_classes.json; 8: results/dense_b1_check.json gate; F§ "Pooling under the verified protocol", "Not explained" (+0.0043 ± 0.0016 and +0.0093 ± 0.0019 at B1 on dense multi_swebench, results/pooling_decomposition.json); results/stress_archive.json projection_full_dense_run.pair.{over_16_workers_s, platform_limit_s}; 9: docs/plans/p2_pool_eval.md §15 disclosure 1; 10: results/hidden_runs.json run_spread.alc_hier_{strat,iid}.sd, replica.noise_summary.ship_alc_sd_range, vadjusted.run3_minus_run2; the missing records: App H.3; 7: results/hidden_runs.json run_spread.benchmarks (7, the formative runs' benchmarks)

## Code, data and conduct

- "The formative feedback (§1.1) set global quantities only": results/hidden_runs.json entry.leaderboard.{organisers_entry, best_entry, read} (0.1801, 0.1172, 2026-09-24); about.id_use (the two uses of the anonymous ids); the bound restated in the abstract, §6.4 and App E.5 (App G.6, "The leaderboard"); F§ "Amendment 4: hier's priors refitted on the expanded pool; no candidate carried (P2)", "Deviation, disclosed: C-extra and describe-hidden did not run" (the gate on C not read, describe-hidden not run, results/hidden_runs.json never opened by the pool evaluation).
