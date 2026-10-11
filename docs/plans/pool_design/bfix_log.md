# Final-check blockers B1-B3, fixed 2026-10-12 by the orchestrator

- B1: PE section 9 'The check log': pre-lock checks are read by their latest entry and the lock refuses unless every latest entry passes; post-lock checks pass only if all entries pass or a recorded fix resolved the failure. PE section 5.1's 'or keeps this rule with the failure recorded' removed (a failed balance check leaves only the benchmark-unit rule by a dated amendment before scoring).
- B2: JK defined in RULE_TEXT (A4) and PE section 6 as the maximum over families f of sum(v_k, k outside f) / sum(n_k, k outside f), with w_k = 1/(pairs in run), v_k = w_k d_k, n_k = w_k.
- B3: narrow reading adopted. An invalid output gives 'archive-3 stays' for the candidate whose arm produced it and no other, unless A3's arm produced it (then for every candidate); (E4) does not require that arm on that run. Aligned in A4 (E4) and Reading 4, PE 5.2 and 9 item 11, impl_plan items 44/51 and section 6.
- PE sha256 5af21af6c818d8a43d22cbbb18876426949d801165cfec28bb4c1177a60d9591 (was 7806244ca4f5d273b788278e3a05086433403368b6011e8c38503f69fef19302); quoted in full in A4's preamble and as 5af21af6c818d8a4 in impl_plan 0.1.
Pre-fix copies: bfix_orig/.
