"""A stronger model on Kaggle: ingest the item features a stronger open model
computed on a free Kaggle GPU, and read them against honest difficulty and
through the acceptance harness.

Run on the first Kaggle commit (Qwen/Qwen3-14B-AWQ, the whole rubric and 270 of
819 attempt texts); docs/findings.md, "Strong model on Kaggle: Qwen3-14B rubric
and attempts", reports it from results/strong_llm_eval.json (`--stage show`
prints its tables): coverage and hash checks (ingest), the per-parent
within-group correlations and sign counts (signs), the LOBO heads (signs ->
heads), the harness lines and placebos (harness), the attempt call on the probe
texts and on every attempted text (attempts), the gate (verdict), and the
session's times, rates and plan (run). The attempt primary is set beside the 4B
probe's D2 lead only where attempts.reference_4b.comparable is true: vLLM's V0
engine returns log-probs after the presence penalty, temperature and
top-k/top-p, so without the notebook's raw recorder its entropies are top-5
lower bounds on a penalised, sharpened, truncated distribution, while D2 was
the full-vocabulary entropy of the raw one.

Why. Every local language-model item signal was null: the Qwen3-4B judge
(experiments/llm4b_close.py), its attempts (experiments/attempt_probe.py, FLOOR:
2-8% graded accuracy on matharena), its entropy profiles and hidden states
(experiments/hidden_state_probe.py), in-context learning and comparisons. The
plan's step 8 escalates to a stronger open model on a free GPU for two uses:
(a) chain-of-thought attempts on matharena, re-read with the attempt probe's
GO/KILL/FLOOR rule, and (b) an ADeLe/DeLeAn-style demand rubric (Zhou et al.,
arXiv:2503.06378) with Agent Psychometrics' task scales (Ge et al.,
arXiv:2604.00594): a handful of 0-5 scales read out as probability-weighted
soft scores, combined by a low-dimensional linear head fitted leave one parent
out. The notebook (task K1: kaggle/strong_probe/strong_probe.py) runs on
Kaggle, downloads measurement-db itself with the user's HF_TOKEN secret and
writes, per model, <model>/export/ in this script's schema; the user copies that
directory's contents to data/features/kaggle/ (or passes --kaggle DIR). This
script loads no language model and needs no GPU.

The entropy job (commit D: strong_probe.py's `entropy`, ENTROPY_VERSION e1.0; README
"Коммит D"). The attempts' reasoning entropy was the first item-side signal to pass its
correlation bar (tok_entropy within-competition rho 0.372 on the probe texts, ent_first1024
0.30-0.37), but on matharena only, so the harness could not fit a transferred slope leave one
parent out and the verdict was NULL for ALC. The harness reference says a covariate of
within-pair r ~0.3 on all four parents gives about -0.0025 test-like ALC, just past the -0.002
gate. Commit D measures that entropy cheaply on all four parents: one sample per unit (the
rubric's units), thinking on, no system prompt, the task text cut as the rubric cuts it and
"Think through how you would solve this task.", at most 1,024 new tokens with the attempts'
sampling, the raw full-vocabulary entropy and raw log-prob recorded per token. It is for the
report's claim; it goes into the submission only if it passes the gate and the organisers allow
a model at predict time. Its table (entropy/entropy.parquet, ENTROPY_COLUMNS) is its own kind
here (ENTROPY_KIND), never merged into the rubric's and the attempts' features: ingest joins it
by content hash like the rubric into WORK/entropy.parquet, reports its coverage per benchmark
under OUT's `entropy` section, and signs, harness, reference and verdict read it with --job
entropy, storing under that section. The decision was fixed before any entropy output was
read; ENTROPY_RULE is the declaration that governs (not the README, not kinds.entropy): the
declared primary is ent_first1024, kept only if a nested harness line passes harness.gate and its
declared sign holds on >= KEEP_SIGNS of the four parents (the step-8 prong, PRONG_FIELD), exactly
as the other declared primaries; ent_first256, lp_first256, lp_first1024, ent_n_tokens and
ent_closed are exploratory (ENTROPY_SIGNS: entropy + = harder, log-prob -, generated length +,
thinking closed -). The features are read as declared only when kinds.entropy says the token
statistics are the raw distribution's full-vocabulary ones (`logprobs` 'raw') and its recorder
check did not fail; otherwise they are excluded (reported, never kept), and the primary is kept
only on the version and job config the rule was fixed for (ENTROPY_VERSION e1.0, ENTROPY_CFG: the
kit's defaults, commit D's config). A unit whose text is degenerate (ent_degenerate: fp16
overflow's '!!!!') has its features set missing, counted per benchmark. The call is final only on
the complete export (every unit of the four parents; a partial one, from a commit that stopped
with exit 75, is labelled PRELIMINARY). Beside the verdict, not gating: the consistency stage
(matharena: this job's ent_first1024 against the attempts' ent_first1024 and tok_entropy, Spearman
over shared items, units and prompts; and the test-retest of units whose prompts are identical,
sampled with different seeds, as pairs and as ICC(1) over every such unit).

The rule is kept traceable in OUT (tests pin its digest): ingest writes entropy.rule with its
digest (entropy_rule_digest) and refuses to ingest under a rule whose digest differs from the one
stored (--accept-rule-change records the old rule in the append-only entropy.rule_history and
drops the results read under it); entropy.rule_first keeps the rule of the first ingest; every
ingest and verdict appends an entry to entropy.history (features digest, config, version, model,
units, completeness, rule digest; the call), which nothing drops. The verdict refuses a rule other
than the first ingest's (or reads it marked RULE CHANGED under --accept-rule-change), and signs,
harness, reference, consistency and verdict refuse a WORK/entropy.parquet other than the one
ingest recorded; the verdict also rebuilds every harness line's x from it and refuses a line
computed on another x (entropy_provenance), so another --work's results never mix with this OUT.

Commit D's export, with the previous Output attached (the recommended path), carries the first
commit's rubric and attempt tables with the same rows (the same bytes only if Kaggle's pyarrow is
the first commit's: parquet records its writer's version). Pass that export alone: ingest
re-writes the joined features with the local pyarrow and compares their digest, so it finds the
rubric-and-attempt features unchanged and leaves every stored section of theirs as it was
(nothing is rewritten). An export with the entropy table alone keeps the features already in
WORK; then pass both directories: --kaggle takes several (a job's shards are read from the first
directory that carries them; two directories with different rows of one job are an error, the
same rows in other bytes a warning).

Input schema (SCHEMA below; `--stage schema` prints it as JSON). It was defined
here first; the notebook's export (strong_probe.py VERSION k1.2, store LAYOUT 2:
eight rubric scales, solve_share, time_log_minutes and their _entropy / _mass
readouts, token counts, and chain-of-thought attempts with the ATTEMPT_EXTRA
columns) was reconciled against it, and tests/test_kaggle_probe.py runs
check_schema on a mock export and requires no error and no warning. The
raw-distribution attempt semantics were reconciled after a review of the
notebook's V0 log-probs: the notebook (strong_probe.py, its RawRecorder) writes
the raw full-vocabulary statistics into the standard columns and says so in
`logprobs`, and runs the forced readout for every attempt; the optional
<base>_raw columns, lp_forced and kinds.attempts.force_every_attempt are accepted
as well (check-schema warns "attempt columns not read" on any other spelling).
The notebook writes no correctness (no per-attempt flag, no per-item graded or
top_correct, the canonical answer only as a hash in _detail/): ingest grades the
answers here, against items.parquet. Its _detail/ tables, _summary.json and
_harness.json (one file on Kaggle, whose output keeps at most 500 files;
strong_probe.py split-harness writes the per-feature JSONs experiments/harness.py
reads) start with '_', so the shard loader, check-schema and ingest skip them.
Two stages read them: attempts reads _detail/attempt_units.parquet for the
notebook's probe flag (UNITS_DETAIL), and run reads _detail/{rubric_units,
attempt_units,attempt_samples}.parquet and _summary.json for the session's
facts. _harness.json is not read here. Run `--stage check-schema` on the real
directory before anything else: it lists every difference from SCHEMA.
Under KAGGLE_DIR (data/features/kaggle/, gitignored under data/):

  manifest.json   schema_version (SCHEMA_VERSION), model (repo, revision, dtype,
                  quantisation, engine), hash (must equal HASH_DEF), kinds
                  ({kind: prompt, scales, readout, sampling, signs, ...}),
                  shards ([{path, rows, sha256}] relative to KAGGLE_DIR);
                  optional created, wall_s, gpu, notebook_digest, logprobs
  <kind>/*.parquet
    item shards    one row per (benchmark, item_id): benchmark (the public name,
                   e.g. matharena), item_id (items.parquet's item_id, as str),
                   content_sha256 (HASH_DEF: sha256 hex of the utf-8 text
                   item_content + "\\n" + item_features, both as paiec.data.
                   load_pairs builds them; a prefix of at least 12 hex digits is
                   accepted), and float feature columns. Names carry the sign
                   (SIGN_RULES): rubric_<scale> an expected demand level 0..5
                   (+), time_<x> an estimated human time (+), solve_<x> a judged
                   share of strong systems that solve it (-); a suffix _entropy,
                   _mass, _tokens or _n marks a diagnostic (no sign). The
                   manifest's kinds.<kind>.signs overrides a name's sign. An
                   item_ids list column in place of item_id (one row per unique
                   text) is exploded. Non-numeric extra columns are ignored.
    attempt shards one row per (benchmark, item_id, attempt): the key columns and
                   attempt (int), answer (the extracted final answer, str or
                   null), n_tokens (int), capped (bool: hit the token cap),
                   tok_entropy and tok_lp (mean next-token entropy in nats and
                   mean token log-prob over the generated tokens, of the
                   distribution the manifest's `logprobs` names: 'raw', the raw
                   model distribution over the full vocabulary, when the
                   notebook's recorder ran (vLLM V0); else the engine's top-5
                   statistics, the entropy a lower bound, of the raw distribution
                   ('raw_topk', vLLM V1) or of the one after penalties,
                   temperature and top-k/top-p ('processed_topk', V0 without the
                   recorder)); optional
                   design (str, default 'cot'), lp_answer, ent0 (float), refuse,
                   forced (bool), and the notebook's per-attempt extras
                   (ATTEMPT_EXTRA: closed, n_think, entropy and log-prob over the
                   reasoning, the answer and the first 256 / 1024 tokens), and
                   (defined here first, for the notebook to adopt) raw-distribution
                   counterparts <base>_raw of the entropy and log-prob columns
                   (RAW_BASES; tok_entropy_raw: the mean full-vocabulary entropy of
                   the raw logits row, tok_lp_raw: the mean raw log-prob of the
                   sampled token, e.g. from a V0 per-request logits processor) and
                   lp_forced (the greedy forced "**Final Answer** \\boxed{" readout's
                   mean answer log-prob, run for every attempt: one raw
                   distribution for all attempts; when present it is the answer
                   log-prob, in place of lp_answer). A shard is an attempt shard
                   iff it has an `attempt` column. Aggregated per (item, design) by
                   attempt_probe.item_features (the extras: their mean) into
                   att_<design>_<feature>. The manifest's `logprobs` (optional;
                   logprob_semantics) says what the token statistics are: 'raw'
                   (the raw distribution, full vocabulary: the notebook's V0
                   recorder), 'raw_topk', 'processed_topk', 'mixed'; a failed
                   kinds.attempts.token_stats.recorder_check makes it unknown.
                   kinds.attempts.forced starting 'every attempt' (the notebook's
                   wording) or force_every_attempt: true says lp_answer is the
                   greedy forced readout for every attempt. Without one of them,
                   an lp_answer that mixes sampled (processed) and forced (greedy,
                   raw) readouts is excluded.
    entropy shard  entropy/*.parquet (ENTROPY_KIND): one row per (benchmark, item_id), the
                   key columns, the four features ENTROPY_FEATURES (float64: ent_first256,
                   ent_first1024 the mean raw full-vocabulary next-token entropy in nats over
                   the first 256 / 1024 generated tokens, lp_first256, lp_first1024 the mean raw
                   log-prob of the sampled tokens; NaN only for an empty sample) and the
                   diagnostics ent_n_tokens, ent_prompt_tokens, ent_task_tokens (int64),
                   ent_closed, ent_degenerate, ent_truncated (bool) (ENTROPY_COLUMNS). A unit
                   that stands for several item_ids repeats its values on each row. Optional:
                   an export without it passes as before. The manifest's kinds.entropy
                   (version, cfg, config, prompt, sampling, signs, primary, token_stats with
                   its recorder_check, logprobs, units) says what they are; its signs,
                   primary, version and cfg are checked against ENTROPY_SIGNS, ENTROPY_PRIMARY
                   and ENTROPY_RULE's config (a difference is a warning: this script's
                   declarations govern).
  Files or directories starting with '.' or '_' are not shards: the shard loader
  skips them (unfinished writes, and the _detail/ tables read by attempts and run).

Stages (results in OUT, derived tables in WORK = data/strong_llm_eval/):

  schema        print SCHEMA as JSON (for the notebook's author)
  check-schema  validate KAGGLE_DIR against SCHEMA: manifest keys, version, hash
                definition, every listed shard present with its row count and
                sha256, key and attempt columns and types, duplicate keys per
                feature, conflicting content hashes, unknown benchmarks; the
                entropy shard's columns and types and kinds.entropy (present, its
                signs, primary, version and job config, a failed recorder check:
                warnings); with several directories, each, and whether two carry
                different rows of one job (an error; the same rows in other bytes, a
                warning); exits 1 on any error
  ingest        read the shards, aggregate the attempts (graded against
                items.parquet's reference answer, a diagnostic only), join to the
                public items as paiec.data.load_pairs builds them, verify every
                row's content hash, carry a unique text's features to its other
                item_ids (matharena's 200 repeated texts; same predict.item_key),
                and report coverage per benchmark and feature, including the
                share of load_pairs responses whose item is covered and a check
                that load_pairs' item dicts give the same predict.item_key
                -> WORK/features.parquet (one row per covered item_id: key,
                key_official, text_key as paiec.llmfeat stores them, source
                'direct' | 'duplicate', features). Records the attempts' log-prob
                semantics (attempt_semantics: manifest `logprobs`, presence
                penalty, the answer log-prob's source and whether it is one
                distribution, the resolved attempt primary and whether it can be
                set beside the 4B D2 lead). When the features change it drops the
                downstream results and WORK/oof.json; when they come out
                byte-identical (the same digest, the same attempt semantics) it
                rewrites nothing, not even ingest and meta. An export without rubric
                or attempt shards leaves them as they are. The entropy table, when
                present: joined the same way (degenerate units' features missing)
                -> WORK/entropy.parquet and entropy.ingest (coverage per benchmark,
                the share of responses covered, per-benchmark means and rates,
                entropy_semantics, completeness against the expected units),
                entropy.meta, entropy.rule (with its digest; a different stored
                rule stops ingest before anything is written, unless
                --accept-rule-change), entropy.rule_first and an entropy.history
                entry; when it changes, entropy's downstream sections go
                (ENTROPY_DOWNSTREAM; history, rule_first and rule_history stay),
                and when the rubric-and-attempt features change,
                entropy.consistency goes.
  signs         every feature, oriented + = harder by its declared sign, against
                the harness's honest difficulty (Rasch b without each of five
                subject folds, averaged; llm4b_close.honest_targets, cached in
                WORK/targets.json) per parent: llm4b_close.corr_block's Spearman
                and Pearson over the benchmark, within item_features groups
                (competition for matharena, language for multi_swebench) and net
                of log length (and position on matharena), with 95% intervals
                from a bootstrap over groups; matharena's text-bearing items and
                its 2025/2026 contests separately; llm4b_close.sign_rule
                (declared sign and leave-one-unit-out agreement on 4 of 5 units,
                the transferred-slope rule) and the step-8 count (declared sign
                on >= KEEP_SIGNS of the 4 parents; agreement_prong: within group,
                and on matharena net of position too (PRONG_FIELD: the rubric
                prompt carries item_features, whose problem_idx is a position
                cue that tracks difficulty within a competition); a
                DerSimonian-Laird mean over parents with its prediction interval
                (hidden_state_probe.random_effects).
                Heads (head_specs): ridge over the rubric levels (rubric_ridge, the
                declared primary), the same with matharena's rubric levels
                residualised on position within competition (rubric_ridge_posfree,
                reported beside it: what the rubric carries beyond position),
                rubric levels and diagnostics, and every
                declared item feature, fitted leave one parent out with the
                penalty chosen by nested LOBO (hidden_state_probe.lobo: features
                and target standardised within benchmark, every benchmark weighs
                the same, no intercept); out-of-fold predictions on each parent,
                the most-chosen penalty refitted on all four for other
                benchmarks. rubric_sum, the unit-weight mean of the oriented
                standardised rubric levels, needs no fit.
                -> OUT signs, heads; WORK/oof.json (tied to the features'
                digest: harness and reference refuse heads from other features)
  harness       every oriented declared feature, rubric_sum and each head through
                experiments/harness.py on its stored rows of the shipped hier
                (llm4b_close.stage_harness's recipe: harness.eval_covariate, x
                standardised within benchmark, the B0 term on raw x;
                harness.score_covariate: nested transferred, per-pair, hybrid and
                B0 lines, the forced lines, per-pair from B7 at s = 0.1 and
                0.25), the within-pair r, and a placebo (x permuted within
                benchmark, N_PLACEBO draws). A covariate on one parent only (the
                attempts: matharena) cannot be switched on leave-one-parent-out;
                its forced per-pair lines are its reading. Resumable (--redo).
  reference     the honest difficulty degraded to r on exactly the covered items
                (0 elsewhere), N_REF draws: what the harness gives at this
                coverage (llm4b_close.stage_reference)
  attempts      the attempt probe's rule on matharena (A1, the plan's step 6):
                per design and label-free feature (attempt_probe.FEATURES and
                ATTEMPT_EXTRA, oriented), within-competition Spearman against
                honest b with a bootstrap over competitions
                (attempt_probe.within_rho), by
                contest year, against strong-tier b where
                data/attempt_probe/targets.parquet has it, and net of log length.
                GO if the best feature reaches rho >= GO_RHO with CI lower bound
                > GO_LO and >= GO_2026 on the 2026 contests; KILL if every
                feature is below KILL_RHO; FLOOR if graded accuracy is under
                FLOOR_ACC (checked first); WEAK otherwise. An answer log-prob
                that mixes readouts, or any token statistic when the manifest
                says 'mixed' (ingest), is reported apart, never a candidate. The
                attempt primary follows PRIMARY_ATTEMPT_RULE (fixed before any
                output existed, by columns and declared semantics: the raw
                full-vocabulary mean entropy the 4B lead D2 measured, as
                tok_entropy_raw or as tok_entropy under manifest logprobs 'raw',
                which the notebook's V0 logits-processor recorder writes; else
                ent_first1024, a fixed window, since a processed entropy with a
                presence penalty drifts with position and generation length).
                The 4B probe's decision is set beside it only when comparable
                (reference_4b), flagged for another quantity of the raw
                distribution, withheld for processed, mixed or undeclared
                log-probs. --attempt-units (ATTEMPT_UNITS) picks the units:
                'all' every attempted text (attempts' own fields), 'probe' the
                147 probe texts the rule was fixed for, flagged in the export's
                UNITS_DETAIL (attempts.probe_only, beside them), 'both' (the
                default) both; each keeps the other's stored result. Both
                readings record the features' and this script's digests; the
                probe reading also the flag table's sha256 (the export's manifest
                does not hash _detail/) and a check that the flag is membership
                in strong_probe.py's PROBE_IDS (a unit is a probe text iff one of
                its item_ids is listed; a mismatch stops the stage)
  run           the session's facts and costs from files alone: the export's
                manifest, _summary.json and _detail write times and token counts,
                and the notebook's root manifest (--run-manifest; by default the
                one under KAGGLE_RAW that matches the export): the sessions'
                measured seconds and rates, the plan's estimates, the progress,
                the attempt shards' duration and the hours left at it; and the
                session's log, if one was saved as *.log beside that manifest's
                copy (run.logs: path, sha256, whether it shows the prefix_prefill
                failure and the --no-prefix-caching retry; [] when none is)
  verdict       the harness gate (harness.gate: nested and acting in >= 3 of 4
                folds, test-like ALC difference <= -0.002, mix/whole of the same
                sign, no held-out parent above +0.002, neither public R1
                weighting above +0.001) on the nested lines, plus the step-8
                sign prong (declared sign on >= KEEP_SIGNS of 4 parents, within
                group and on matharena net of position; for a head, positive
                out-of-fold r, on matharena net of competition, log length and
                position). A declared primary (PRIMARY_HEAD; the resolved attempt
                primary for every attempt design) passing both is kept; others
                that pass are reported as exploratory, not kept.
  consistency   (the entropy job) on matharena, this job's ent_first1024 against
                the attempts' ent_first1024 and tok_entropy per design (Spearman
                over the items both cover, over one item per entropy unit, and per
                prompt: the mean over a prompt's units against the attempts' value
                for that text), with both sides' log-prob semantics; and the
                test-retest per parent: units whose prompts are identical (the same
                item_content under other metadata, so another unit and another
                seed), Spearman and Pearson of their ent_first1024 over one pair per
                prompt, and ICC(1) over every unit of those prompts. Reported,
                never gating -> entropy.consistency
  show          markdown tables of OUT (the entropy job's after the rest)

--job entropy (signs, harness, reference, verdict): the same stages on WORK/entropy.parquet,
stored under OUT's `entropy` section: signs without heads (entropy_registry: ENTROPY_SIGNS, the
ent_* diagnostics without a sign), the harness on the usable oriented features (four parents, so
the transferred slope is fitted leave one parent out), the reference on the items with a finite
ent_first1024, and the verdict by ENTROPY_RULE (with the rule's digests, the completeness, the
provenance check and a history entry; final only when complete under the unchanged rule). `run`
on an export with kinds.entropy writes entropy.run (the session's entropy shards, rates against
the plan's assumption, units left) and leaves `run` alone. The entropy features are read only
this way: strong_probe.py split-harness leaves the export's entropy_* entries out of the
per-covariate files experiments/harness.py reads.

Caveats built in, to state in the findings: the heads' transferred slope for a
held-out parent is fitted on other parents' out-of-fold x, whose heads saw the
held-out parent's items (a second-order leak that favours the covariate;
hidden_state_probe's caveat); x is standardised over every covered item of a
benchmark, a run-time predictor sees only the run's items; the honest target
leaves each subject fold out, but the features themselves are label-free.

Run (repo root; one process, at most ~0.75 GB (ingest's load_pairs), the rest
under 0.6 GB; no language model; timings from a smoke run on a synthetic
directory over every public item, beside a running LoRA job):
  python experiments/strong_llm_eval.py --stage schema
  python experiments/strong_llm_eval.py --stage check-schema
  python experiments/strong_llm_eval.py --stage ingest       # seconds
  python experiments/strong_llm_eval.py --stage signs        # ~0.5 min per signed feature, ~15 s per head
  python experiments/strong_llm_eval.py --stage attempts     # ~20 s per reading; --attempt-units all|probe|both
  python experiments/strong_llm_eval.py --stage harness      # ~1.3 min per covariate with --placebo 1,
                                                             # ~2 with the default 3; resumable;
                                                             # --feats NAME ... for a subset
  python experiments/strong_llm_eval.py --stage reference    # ~6 min
  python experiments/strong_llm_eval.py --stage verdict
  python experiments/strong_llm_eval.py --stage run          # seconds; reads KAGGLE_RAW's root manifest
  python experiments/strong_llm_eval.py --stage show
The entropy job (commit D's export copied to data/features/kaggle_d/; the same WORK and OUT):
  python experiments/strong_llm_eval.py --stage check-schema --kaggle data/features/kaggle_d
  python experiments/strong_llm_eval.py --stage ingest --kaggle data/features/kaggle_d
      # the Output was attached: kaggle_d alone; only when kaggle_d carries the entropy table alone:
      # --kaggle data/features/kaggle data/features/kaggle_d
  python experiments/strong_llm_eval.py --stage signs --job entropy        # ~0.5 min per feature
  python experiments/strong_llm_eval.py --stage harness --job entropy      # ~2 min per covariate
  python experiments/strong_llm_eval.py --stage reference --job entropy    # ~6 min
  python experiments/strong_llm_eval.py --stage consistency
  python experiments/strong_llm_eval.py --stage verdict --job entropy
  python experiments/strong_llm_eval.py --stage run --kaggle data/features/kaggle_d
Needs data/<benchmark>/, the Kaggle outputs, and the harness rows (python
experiments/harness.py --stage collect) for harness and reference.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import attempt_probe as AP  # noqa: E402
from experiments import harness as H  # noqa: E402
from experiments import hidden_state_probe as HS  # noqa: E402
from experiments import itemcov_eval as ICE  # noqa: E402
from experiments import llm4b_close as L4  # noqa: E402
from paiec import itemcov as IC  # noqa: E402
from paiec import llmfeat as F  # noqa: E402

KAGGLE_DIR = os.path.join(ROOT, "data", "features", "kaggle")
#: where the full Kaggle Output is copied (its root manifest strong_probe/manifest.json carries the
#: sessions' times, rates and the plan; the export's manifest only their totals)
KAGGLE_RAW = os.path.join(ROOT, "data", "features", "kaggle_raw")
WORK = os.path.join(ROOT, "data", "strong_llm_eval")
OUT = os.path.join(ROOT, "results", "strong_llm_eval.json")
BENCHES = L4.BENCHES
PARENTS = H.PARENTS
GROUP_KEY = ICE.GROUP_KEY
YEAR_2026 = L4.YEAR_2026

# --- the input schema (reconcile with kaggle/strong_probe/strong_probe.py) ---------------

SCHEMA_VERSION = 1
HASH_DEF = ("sha256 hex of utf-8(item_content + '\\n' + item_features), both as paiec.data.load_pairs "
            "builds them (paiec.llmfeat.item_text)")
KEY_COLS = ("benchmark", "item_id", "content_sha256")
MIN_HASH_HEX = 12
#: other spellings accepted for the key columns
ALIASES = {"bench": "benchmark", "benchmark_name": "benchmark", "content_hash": "content_sha256",
           "sha256": "content_sha256", "text_sha256": "content_sha256"}
ATTEMPT_COLS = {"attempt": "int", "answer": "any", "n_tokens": "int", "capped": "bool",
                "tok_entropy": "float", "tok_lp": "float"}
ATTEMPT_OPTIONAL = {"design": "str", "lp_answer": "float", "ent0": "float", "refuse": "bool", "forced": "bool",
                    "lp_forced": "float"}
#: optional per-attempt columns kaggle/strong_probe/strong_probe.py writes, averaged over an
#: item's attempts: column -> (aggregate name, declared sign, + = harder); the signs are the
#: notebook's ATTEMPT_SIGNS, declared there before any output existed
ATTEMPT_EXTRA = {"closed": ("closed_rate", -1), "n_think": ("mean_think_len", 1),
                 "tok_lp_think": ("tok_lp_think", -1), "tok_entropy_think": ("tok_entropy_think", 1),
                 "tok_lp_answer": ("tok_lp_answer", -1), "tok_entropy_answer": ("tok_entropy_answer", 1),
                 "ent_first256": ("ent_first256", 1), "ent_first1024": ("ent_first1024", 1),
                 "lp_first256": ("lp_first256", -1), "lp_first1024": ("lp_first1024", -1)}
#: entropy and log-prob columns whose raw-distribution counterpart <base>_raw an attempt shard
#: may carry (the full-vocabulary entropy of the raw logits row; the raw log-prob of the
#: sampled token), with the base's sign; raw whatever the manifest's `logprobs` say
RAW_BASES = {"tok_entropy": 1, "tok_lp": -1, "tok_entropy_think": 1, "tok_lp_think": -1,
             "tok_entropy_answer": 1, "tok_lp_answer": -1, "ent_first256": 1, "ent_first1024": 1,
             "lp_first256": -1, "lp_first1024": -1}
RAW_SUFFIX = "_raw"
ATTEMPT_EXTRA.update({b + RAW_SUFFIX: (b + RAW_SUFFIX, s) for b, s in RAW_BASES.items()})
#: attempt aggregate -> declared sign: attempt_probe.FEATURES, forced_rate (a forced
#: continuation: the model gave no answer of its own; the notebook's sign) and the extras
ATT_SIGNS = {**AP.FEATURES, "forced_rate": 1, **{n: sg for n, sg in ATTEMPT_EXTRA.values()}}
#: attempt aggregates read off the answer log-prob (excluded when it mixes readouts)
LP_ANSWER_FEATURES = ("lp_answer", "lp_answer_top")
#: the attempt primary, for every design: the first (feature, required log-prob semantics) the
#: export carries. Fixed before any output existed, by the columns and the declared semantics,
#: never by values: the raw full-vocabulary mean entropy is what the 4B probe's lead (D2)
#: measured, as tok_entropy_raw or as tok_entropy when the manifest says the token statistics
#: are the raw distribution's ('raw'); otherwise ent_first1024, a fixed window (a processed
#: entropy under a presence penalty drifts with position and with generation length, so a
#: mean over the whole attempt mixes that drift in)
PRIMARY_ATTEMPT_RULE = (("tok_entropy_raw", None), ("tok_entropy", "raw"), ("ent_first1024", None))
PRIMARY_ATTEMPT_ORDER = tuple(f if s is None else f"{f} (if {s})" for f, s in PRIMARY_ATTEMPT_RULE)
DEFAULT_DESIGN = "cot"
MANIFEST_KEYS = ("schema_version", "model", "hash", "kinds", "shards")
#: name -> declared sign against difficulty (+ = harder), fixed before any output was read.
#: Checked in order; the first match wins. The manifest's kinds.<kind>.signs overrides.
SIGN_RULES = (
    ("suffix", ("_entropy", "_mass", "_tokens", "_n"), 0),    # diagnostics of a readout
    ("prefix", ("rubric_",), 1),     # a demand level: more demand, harder (ADeLe/DeLeAn; Agent Psychometrics)
    ("prefix", ("time_",), 1),       # an estimated human time
    ("prefix", ("solve_",), -1),     # a judged share of strong systems that solve it
)
#: attempt aggregates that are not label-free (graded, top_correct: they need the reference
#: answer) or not features (n_distinct, k): never in the harness
ATT_DIAG = ("graded", "top_correct", "n_distinct", "k")

# --- the entropy job (commit D) --------------------------------------------------------------

#: the export's kind: its shards' directory, the manifest's kinds key and OUT's section
ENTROPY_KIND = "entropy"
#: the version of strong_probe.py's entropy job (its prompt, windows and readout: kinds.entropy.version)
#: the decision (ENTROPY_RULE) was fixed for
ENTROPY_VERSION = "e1.0"
#: the job config hash (kinds.entropy.cfg) the decision was fixed for: strong_probe.py's job_cfg with every default
#: (commit D's ARGS ["--jobs", "entropy", "--no-prefix-caching"]): Qwen/Qwen3-14B-AWQ at its pinned revision,
#: vllm, awq, task_tokens 3072, the template and instruction, 1,024 new tokens, the attempts' sampling with
#: presence penalty 1.5, the windows, seed 0 and its per-unit rule. Another model, seed, penalty or cut is another
#: config: its features are read, but the primary is kept only on this one
ENTROPY_CFG = "e21faa7f3d0bf929"
#: the four features and their declared signs against difficulty (+ = harder), fixed before any
#: output existed: on a harder item the model is less sure of its next token (entropy +) and its
#: sampled tokens are less likely (log-prob -)
ENTROPY_FEATURES = {"ent_first256": 1, "ent_first1024": 1, "lp_first256": -1, "lp_first1024": -1}
#: two diagnostics read as exploratory features, their signs declared here (kinds.entropy.signs gives
#: them 0): a harder item runs to the 1,024-token cap (generated length +) and closes its thinking
#: less often (-, as the attempts' closed_rate)
ENTROPY_EXPLORATORY = {"ent_n_tokens": 1, "ent_closed": -1}
ENTROPY_SIGNS = {**ENTROPY_FEATURES, **ENTROPY_EXPLORATORY}
ENTROPY_PRIMARY = "ent_first1024"
#: the entropy shard's columns after the key columns, with their types (strong_probe.py ENTROPY_COLS)
ENTROPY_COLUMNS = {**{f: "float" for f in ENTROPY_FEATURES}, "ent_n_tokens": "int", "ent_closed": "bool",
                   "ent_degenerate": "bool", "ent_prompt_tokens": "int", "ent_task_tokens": "int",
                   "ent_truncated": "bool"}
#: a unit whose text is degenerate (one character most of it: fp16 overflow) has its features missing
ENTROPY_DEGENERATE = "ent_degenerate"
#: the joined entropy table's columns that are neither join columns nor features: prompt_sha, sha256
#: of item_content (16 hex): the entropy prompt reads the task text alone, so equal prompt_sha means
#: an identical prompt (the test-retest pairs)
ENTROPY_JOIN_EXTRA = ("prompt_sha",)
ENTROPY_TABLE = "entropy.parquet"
#: the entropy section's parts computed from its table: dropped when the table changes
ENTROPY_DOWNSTREAM = ("signs", "harness", "reference", "consistency", "verdict")
#: consistency: the attempt aggregates set beside the entropy primary on matharena, and the fewest
#: shared items (or test-retest pairs) for a correlation
CONSISTENCY_WITH = ("ent_first1024", "tok_entropy")
CONSISTENCY_MIN = 5

SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "layout": {"manifest": "manifest.json",
               "shards": "<kind>/*.parquet; an attempt shard is one with an 'attempt' column; "
                         "paths starting with '.' or '_' are skipped"},
    "manifest_required": list(MANIFEST_KEYS),
    "manifest_example": {
        "schema_version": SCHEMA_VERSION,
        "model": {"repo": "<hf repo id>", "revision": "<commit sha>", "dtype": "float16",
                  "quantization": "<none|awq|gptq|...>", "engine": "<vllm x.y | transformers x.y>"},
        "hash": HASH_DEF,
        "kinds": {"rubric": {"prompt": "<full prompt text>", "scales": {"rubric_<scale>": "<definition>"},
                             "readout": "probability-weighted expected level over the tokens '0'..'5'",
                             "signs": {"rubric_<scale>": 1}},
                  "attempts": {"prompt": "<full prompt text>", "sampling": {"temperature": 0.7},
                               "k": 4, "max_new_tokens": 4096, "designs": [DEFAULT_DESIGN],
                               "force_every_attempt": "<optional bool: lp_answer is the forced readout "
                                                      "for every attempt>"}},
        "shards": [{"path": "rubric/part-0000.parquet", "rows": 512, "sha256": "<file sha256>"}],
        "logprobs": "<optional: 'raw model distribution' | 'processed (V0): after temperature, penalties "
                    "and top-k/top-p; ...'>",
        "created": "<ISO time>", "wall_s": 0, "gpu": "2x T4"},
    "key_columns": {"benchmark": "str: public benchmark name (matharena, multi_swebench, ...)",
                    "item_id": "str: items.parquet item_id (or item_ids: list of str, one row per unique text)",
                    "content_sha256": f"str: {HASH_DEF}; a prefix of >= {MIN_HASH_HEX} hex digits is accepted"},
    "item_shard_features": {"rubric_<scale>": "float, expected demand level 0..5 (sign +)",
                            "rubric_<scale>_entropy": "float, readout entropy (diagnostic, no sign)",
                            "time_<x>": "float, estimated human time, e.g. log10 minutes (sign +)",
                            "solve_<x>": "float, judged share of strong systems that solve it (sign -)"},
    "attempt_shard_columns": {**ATTEMPT_COLS, **{k + " (optional)": v for k, v in ATTEMPT_OPTIONAL.items()},
                              **{k + " (optional, averaged per item)": f"{'bool' if k == 'closed' else 'float'} "
                                 f"-> att_<design>_{n} (sign {sg:+d})" for k, (n, sg) in ATTEMPT_EXTRA.items()}},
    "entropy_shard": {
        "path": f"{ENTROPY_KIND}/*.parquet (optional; one row per (benchmark, item_id), a unit's values repeated "
                "on each of its item_ids)",
        "columns": {c: (f"{t} (sign {ENTROPY_SIGNS[c]:+d}{', the declared primary' if c == ENTROPY_PRIMARY else ''})"
                        if c in ENTROPY_SIGNS else f"{t} (diagnostic)") for c, t in ENTROPY_COLUMNS.items()},
        "required": list(ENTROPY_FEATURES),
        "manifest": {"kinds.entropy": {"version": ENTROPY_VERSION, "cfg": f"{ENTROPY_CFG} (the rule's config)",
                                       "signs": ENTROPY_FEATURES,
                                       "primary": ENTROPY_PRIMARY,
                                       "logprobs": "'raw' (the raw full-vocabulary statistics: read as declared)",
                                       "token_stats": {"recorder_check": {"status": "'ok' | 'FAILED' | 'n/a'"}}}}},
    "aliases": ALIASES,
}

# --- analysis constants --------------------------------------------------------------------

MIN_ITEMS = L4.MIN_ITEMS
BOOT = L4.BOOT
DIAG_BOOT = 500                  # bootstrap draws for a feature without a declared sign
#: results sections computed from the ingested features or the export behind them (dropped when
#: they change)
DOWNSTREAM = ("signs", "heads", "harness", "reference", "attempts", "verdict", "run")
N_PLACEBO = L4.N_PLACEBO
PLACEBO_BOOTS = L4.PLACEBO_BOOTS
FORCED_ALL = L4.FORCED_ALL
NESTED = L4.NESTED
REF_R = (0.2, 0.3, 0.5)
N_REF = 4
KEEP_SIGNS = 3                   # plan step 8: sign agreement on at least 3 of 4 parents
KEEP_R = HS.KEEP_R               # a head's mean LOBO r, reported beside the gate
PRIMARY_HEAD = "rubric_ridge"
POSFREE_HEAD = "rubric_ridge_posfree"   # the primary's columns, matharena's net of position
#: the sign prong's statistic per parent (default spearman_within): on matharena net of the
#: competition means, log length and position (problem_idx is in the rubric prompt's metadata)
PRONG_FIELD = {"matharena": "partial2_spearman"}
#: a head's prong: out-of-fold Pearson over the benchmark; on matharena net of position too
HEAD_PRONG_FIELD = {"matharena": "partial2_pearson"}
ATT_MIN_ITEMS = 8                # attempt_report: fewest matharena items for a design
#: the export's per-text attempt table (under KAGGLE_DIR; '_' paths are not shards): one row per unique
#: text with its item_ids and `probe`, the notebook's flag for the 147 probe texts (strong_probe.py
#: PROBE_IDS: the 160 attempt-probe items' texts, run first). The attempt rule was fixed for them
UNITS_DETAIL = os.path.join("_detail", "attempt_units.parquet")
#: --attempt-units: the units the attempt rule reads. 'all' writes attempts' all-units fields,
#: 'probe' attempts.probe_only (the probe texts only), 'both' (default) both; each keeps the other's
ATTEMPT_UNITS = ("all", "probe", "both")
#: the notebook, relative to ROOT: its PROBE_IDS (read with ast, never imported) check the probe flag
STRONG_PROBE = os.path.join("kaggle", "strong_probe", "strong_probe.py")
GO_RHO, GO_LO, GO_2026 = 0.35, 0.15, 0.25
KILL_RHO = 0.15
FLOOR_ACC = 0.10
ALPHAS = HS.ALPHAS

#: the entropy job's decision, fixed before any entropy output was read and written into OUT (entropy.rule
#: at ingest, entropy.verdict.rule), each time with its digest (entropy_rule_digest; tests pin it): nothing here
#: may be changed after commit D's export is read. Ingest refuses a rule whose digest differs from the one OUT
#: holds (--accept-rule-change records the old one in entropy.rule_history and drops the results read under it),
#: and the verdict refuses one that differs from the rule at the first ingest (entropy.rule_first), or marks its
#: call RULE CHANGED under --accept-rule-change
ENTROPY_RULE = {
    "fixed": ("2026-09-28, before any output of strong_probe.py's entropy job (version " + ENTROPY_VERSION
              + ") was downloaded or read"),
    "primary": ENTROPY_PRIMARY,
    "config": {"version": ENTROPY_VERSION, "cfg": ENTROPY_CFG,
               "what": ("kinds.entropy.cfg, strong_probe.py's job config hash with every default (commit D's ARGS "
                        "['--jobs', 'entropy', '--no-prefix-caching']): the model and its revision, the prompt, the "
                        "cut, the sampling, the windows and the seeds; the primary is kept only on this config and "
                        "version")},
    "keep": (f"{ENTROPY_PRIMARY} is kept only if a nested harness line passes harness.gate {H.GATE} (test-like ALC "
             f"difference <= {H.GATE['tl']}, nested and acting in >= {H.GATE['folds_on']} of 4 folds, mix/whole of "
             f"the same sign, no held-out parent above +{H.GATE['parent']}, neither public R1 weighting above "
             f"+{H.GATE['guard']}) and its declared sign holds on >= {KEEP_SIGNS} of the 4 parents (plan step 8: "
             f"within group, and on matharena net of competition, log length and position: {PRONG_FIELD}), exactly "
             "as the other declared primaries"),
    "exploratory": [f for f in ENTROPY_SIGNS if f != ENTROPY_PRIMARY],
    "signs": dict(ENTROPY_SIGNS),
    "readable": ("the features are read as declared only when the manifest's kinds.entropy says the token "
                 "statistics are the raw distribution's full-vocabulary ones (logprobs 'raw') and its recorder check "
                 "did not fail; otherwise every feature is excluded (reported, never kept). The primary is kept only "
                 f"on version {ENTROPY_VERSION} and job config {ENTROPY_CFG} (`config`), the ones this rule was fixed "
                 "for"),
    "degenerate": f"a unit whose text is degenerate ({ENTROPY_DEGENERATE}) has its features missing, counted per "
                  "benchmark",
    "complete": ("the call is read only on the complete export: every unit of the four parents (strong_probe.py's "
                 "entropy_units: the unique items by predict.item_key, 1,555 / 2,078 / 233 / 212 on the public data) "
                 "covered; on a partial export (a commit that stopped with exit 75) the verdict is labelled "
                 "PRELIMINARY, and the complete export's verdict replaces it"),
    "provenance": ("the verdict reads signs and harness lines computed on the table ingest recorded (their features "
                   "digest and x digests), under the rule ingest recorded, which must be the rule at the first "
                   "ingest"),
    "not_gating": ["consistency: matharena against the attempts' ent_first1024 and tok_entropy (over items, units "
                   "and prompts), and the test-retest of identical prompts (pairs, ICC(1))", "reference", "placebos",
                   "forced lines"],
    "submission": ("not for the submission unless the primary is kept and the organisers allow a language model at "
                   "predict time"),
}


def entropy_rule_digest(rule=None):
    """sha256 hex of a rule's canonical JSON (paiec.llmfeat.digest of H._jsonable), without its own `digest`
    key: ENTROPY_RULE's by default, what OUT's entropy.rule, entropy.ingest and entropy.verdict record."""
    r = ENTROPY_RULE if rule is None else {k: v for k, v in rule.items() if k != "digest"}
    return F.digest(H._jsonable(r))


def entropy_rule_record():
    """ENTROPY_RULE with its digest, as OUT's entropy.rule keeps it."""
    return {**ENTROPY_RULE, "digest": entropy_rule_digest()}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def content_sha256(item):
    """HASH_DEF for one item dict."""
    return hashlib.sha256(F.item_text(item).encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_sign(name, overrides=None):
    """(sign, source) of a feature name: the manifest's override, else SIGN_RULES,
    else 0 (no declared sign)."""
    if overrides and name in overrides:
        try:
            return int(np.sign(float(overrides[name]))), "manifest"
        except (TypeError, ValueError):
            pass
    for kind, pats, sign in SIGN_RULES:
        if (kind == "suffix" and name.endswith(pats)) or (kind == "prefix" and name.startswith(pats)):
            return sign, f"{kind} {'|'.join(pats)}"
    return 0, "none"


# --- reading the shards ----------------------------------------------------------------------

def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _norm(df):
    """Key columns under their SCHEMA names, item_ids exploded, keys as str."""
    df = df.rename(columns={c: ALIASES[c] for c in df.columns if c in ALIASES and ALIASES[c] not in df.columns})
    if "item_id" not in df.columns and "item_ids" in df.columns:
        df = df.explode("item_ids").rename(columns={"item_ids": "item_id"})
        df = df[df["item_id"].notna()].reset_index(drop=True)
    for c in ("benchmark", "item_id", "content_sha256"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: None if _missing(v) else str(v).strip())
    if "design" in df.columns:
        df["design"] = df["design"].map(lambda v: DEFAULT_DESIGN if _missing(v) else str(v))
    return df


def shard_files(kdir):
    """Relative paths of the parquet shards under kdir, sorted; any path part
    starting with '.' or '_' is skipped."""
    out = []
    for p in sorted(glob.glob(os.path.join(kdir, "**", "*.parquet"), recursive=True)):
        rel = os.path.relpath(p, kdir)
        if any(part.startswith((".", "_")) for part in rel.split(os.sep)):
            continue
        out.append(rel)
    return out


def shard_kind(rel, df):
    """'attempts' (an `attempt` column), 'entropy' (under ENTROPY_KIND/: the entropy job's table,
    never merged into the item features) or 'items'."""
    if "attempt" in df.columns:
        return "attempts"
    if rel.split(os.sep)[0] == ENTROPY_KIND:
        return ENTROPY_KIND
    return "items"


def load_shards(kdir):
    """(manifest or None, [(relative path, 'items' | 'attempts' | 'entropy', DataFrame)])."""
    import pandas as pd
    man = None
    mp = os.path.join(kdir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            man = json.load(fh)
    frames = []
    for rel in shard_files(kdir):
        df = _norm(pd.read_parquet(os.path.join(kdir, rel)))
        frames.append((rel, shard_kind(rel, df), df))
    return man, frames


def feature_columns(df):
    """The numeric non-key columns of an item shard (bool counts as numeric)."""
    import pandas as pd
    return [c for c in df.columns if c not in KEY_COLS and c != "item_ids"
            and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]


def _type_ok(series, t):
    import pandas as pd
    s = series.dropna()
    if not len(s):
        return True
    if t == "int":
        return pd.api.types.is_integer_dtype(series) or (pd.api.types.is_float_dtype(series)
                                                        and bool(np.all(np.mod(s, 1) == 0)))
    if t == "float":
        return pd.api.types.is_numeric_dtype(series)
    if t == "bool":
        return pd.api.types.is_bool_dtype(series) or set(pd.unique(s)) <= {0, 1, True, False}
    if t == "any":
        return True
    return bool(s.map(lambda v: isinstance(v, str)).all())


def check_schema(kdir, loaded=None, benches=BENCHES):
    """Validate a Kaggle output directory against SCHEMA: {'ok', 'errors',
    'warnings', 'shards', 'manifest_model'}."""
    man, frames = loaded if loaded is not None else load_shards(kdir)
    err, warn, shards = [], [], {}
    if man is None:
        err.append("manifest.json missing")
        man = {}
    else:
        miss = [k for k in MANIFEST_KEYS if k not in man]
        if miss:
            err.append(f"manifest lacks {miss}")
        if "schema_version" in man and man["schema_version"] != SCHEMA_VERSION:
            err.append(f"schema_version {man['schema_version']!r}, expected {SCHEMA_VERSION}")
        if "hash" in man and man["hash"] != HASH_DEF:
            err.append(f"hash definition differs: manifest {man['hash']!r}, expected {HASH_DEF!r}")
    if not frames:
        err.append("no parquet shards")
    listed = {}
    for s in man.get("shards") or []:
        if isinstance(s, dict) and "path" in s:
            listed[os.path.normpath(s["path"])] = s
    seen = set()
    item_parts, att_parts = [], []
    n_ent = 0
    for rel, kind, df in frames:
        seen.add(os.path.normpath(rel))
        info = {"kind": kind, "rows": int(len(df)), "columns": list(map(str, df.columns))}
        miss = [c for c in KEY_COLS if c not in df.columns]
        if miss:
            err.append(f"{rel}: lacks key columns {miss}")
        lst = listed.get(os.path.normpath(rel))
        if lst is None:
            warn.append(f"{rel}: not listed in the manifest's shards")
        else:
            if "rows" in lst and int(lst["rows"]) != len(df):
                err.append(f"{rel}: {len(df)} rows, manifest says {lst['rows']}")
            if lst.get("sha256") and lst["sha256"] != file_sha256(os.path.join(kdir, rel)):
                err.append(f"{rel}: sha256 differs from the manifest")
        if "benchmark" in df.columns:
            unk = sorted(set(df["benchmark"].dropna()) - set(benches))
            if unk:
                warn.append(f"{rel}: benchmarks outside {list(benches)}: {unk} (ignored)")
        for c in ("benchmark", "item_id"):
            if c in df.columns and df[c].isna().any():
                err.append(f"{rel}: {int(df[c].isna().sum())} rows without {c}")
        if "content_sha256" in df.columns:
            h = df["content_sha256"].dropna()
            bad = int((~h.str.fullmatch(r"[0-9a-fA-F]{%d,64}" % MIN_HASH_HEX)).sum())
            if bad:
                err.append(f"{rel}: {bad} content_sha256 values are not hex of >= {MIN_HASH_HEX} digits")
            if df["content_sha256"].isna().any():
                warn.append(f"{rel}: {int(df['content_sha256'].isna().sum())} rows without content_sha256 "
                            "(they cannot be verified and are dropped)")
        if kind == "attempts":
            for c, t in ATTEMPT_COLS.items():
                if c not in df.columns:
                    err.append(f"{rel}: attempt shard lacks {c}")
                elif not _type_ok(df[c], t):
                    err.append(f"{rel}: {c} is not {t}")
            for c, t in list(ATTEMPT_OPTIONAL.items()) + [(c, "bool" if c == "closed" else "float")
                                                          for c in ATTEMPT_EXTRA]:
                if c in df.columns and not _type_ok(df[c], t):
                    err.append(f"{rel}: {c} is not {t}")
            unread = [c for c in df.columns if c not in KEY_COLS and c not in ATTEMPT_COLS
                      and c not in ATTEMPT_OPTIONAL and c not in ATTEMPT_EXTRA]
            if unread:
                warn.append(f"{rel}: attempt columns not read: {unread}")
            att_parts.append(df)
        else:
            feats = feature_columns(df)
            other = [c for c in df.columns if c not in KEY_COLS and c not in feats and c != "item_ids"]
            if other:
                warn.append(f"{rel}: non-numeric columns ignored: {other}")
            if not feats:
                warn.append(f"{rel}: no numeric feature columns")
            info["features"] = feats
            nonfinite = {c: int((~np.isfinite(df[c].astype(float))).sum()) for c in feats}
            info["nonfinite"] = {c: n for c, n in nonfinite.items() if n}
            if kind == ENTROPY_KIND:
                n_ent += 1
                if lst is None or "rows" not in lst or not lst.get("sha256"):
                    err.append(f"{rel}: an entropy shard must be listed in the manifest's shards with its rows and "
                               "sha256, which are verified")
                miss = [c for c in ENTROPY_FEATURES if c not in df.columns]
                if miss:
                    err.append(f"{rel}: entropy shard lacks {miss}")
                for c, t in ENTROPY_COLUMNS.items():
                    if c in df.columns and not _type_ok(df[c], t):
                        err.append(f"{rel}: {c} is not {t}")
                unread = [c for c in feats if c not in ENTROPY_COLUMNS]
                if unread:
                    warn.append(f"{rel}: entropy columns not read: {unread}")
            else:
                ent_named = [c for c in feats if c in ENTROPY_COLUMNS]
                if ent_named:
                    warn.append(f"{rel}: entropy column names outside {ENTROPY_KIND}/: {ent_named} (read as item "
                                "features of the rubric's kind, not as the entropy job)")
            item_parts.append(df[[c for c in KEY_COLS if c in df.columns] + feats])
        shards[rel] = info
    for p in sorted(set(listed) - seen):
        err.append(f"{p}: listed in the manifest, not on disk")
    if n_ent:
        spec = (man.get("kinds") or {}).get(ENTROPY_KIND)
        if not isinstance(spec, dict):
            warn.append(f"an entropy shard, but no kinds.{ENTROPY_KIND} in the manifest: its prompt, sampling and "
                        "token statistics are undeclared, so its features are excluded")
        else:
            sem = entropy_semantics(man)
            if sem["sign_disagreements"]:
                warn.append(f"kinds.{ENTROPY_KIND}.signs differ from ENTROPY_SIGNS on {sem['sign_disagreements']}: "
                            "this script's declared signs are used")
            if sem["primary_manifest"] != ENTROPY_PRIMARY:
                warn.append(f"kinds.{ENTROPY_KIND}.primary is {sem['primary_manifest']!r}; the declared primary is "
                            f"{ENTROPY_PRIMARY!r} (ENTROPY_RULE)")
            if sem["version"] != ENTROPY_VERSION:
                warn.append(f"kinds.{ENTROPY_KIND}.version is {sem['version']!r}, the rule was fixed for "
                            f"{ENTROPY_VERSION!r}: the primary cannot be kept")
            elif sem["readable"] and not sem["primary_eligible"]:
                warn.append(f"kinds.{ENTROPY_KIND}: {sem['why_not_eligible']}: the primary cannot be kept")
            if not sem["readable"]:
                warn.append(f"entropy features excluded: {sem['why']}")
    if item_parts and all(all(c in d.columns for c in KEY_COLS[:2]) for d in item_parts):
        import pandas as pd
        allf = pd.concat(item_parts, ignore_index=True)
        for c in allf.columns:
            if c in KEY_COLS:
                continue
            sub = allf[allf[c].notna()]
            d = int(sub.duplicated(["benchmark", "item_id"]).sum())
            if d:
                err.append(f"feature {c}: {d} duplicate (benchmark, item_id) rows")
        _hash_conflicts(allf, err, "item shards")
    if att_parts and all(all(c in d.columns for c in ("benchmark", "item_id", "attempt")) for d in att_parts):
        import pandas as pd
        alla = pd.concat(att_parts, ignore_index=True)
        alla["design"] = alla["design"].fillna(DEFAULT_DESIGN) if "design" in alla.columns else DEFAULT_DESIGN
        d = int(alla.duplicated(["benchmark", "item_id", "design", "attempt"]).sum())
        if d:
            err.append(f"attempts: {d} duplicate (benchmark, item_id, design, attempt) rows")
        _hash_conflicts(alla, err, "attempt shards")
    model = man.get("model")
    return {"ok": not err, "errors": err, "warnings": warn, "shards": shards,
            "manifest_model": model, "schema_version": SCHEMA_VERSION}


def _hash_conflicts(df, err, what):
    if "content_sha256" not in df.columns:
        return
    n = df.dropna(subset=["content_sha256"]).groupby(["benchmark", "item_id"])["content_sha256"].nunique()
    bad = int((n > 1).sum())
    if bad:
        err.append(f"{what}: {bad} items carry more than one content_sha256")


#: the jobs an export's shards belong to: 'main' (the rubric's and the attempts' shards, the features
#: every stage without --job entropy reads) and the entropy job
JOB_KINDS = {"main": ("items", "attempts"), ENTROPY_KIND: (ENTROPY_KIND,)}


def kaggle_dirs(args):
    """--kaggle as a list of directories (one or several)."""
    k = args.kaggle
    return [k] if isinstance(k, str) else list(k)


def job_sources(exports):
    """Which export directory each job's shards are read from: exports is
    [(dir, (manifest, frames))]. -> ({job: (dir, manifest, frames)}, errors,
    warnings): a job's shards come from the first directory that carries any; a
    directory that carries a different set of them is an error; one that carries
    the same set (paths and sha256) is read once, and so is one with the same
    paths and the same rows in other bytes (parquet records its writer's pyarrow
    version: a Kaggle image with another pyarrow rewrites an attached table
    byte-differently), with a warning."""
    out, errs, warns = {}, [], []
    for job, kinds in JOB_KINDS.items():
        cands = []
        for d, (man, frames) in exports:
            mine = {rel: df for rel, kind, df in frames if kind in kinds}
            if mine:
                cands.append((d, man, frames, [(r, file_sha256(os.path.join(d, r))) for r in sorted(mine)], mine))
        if not cands:
            continue
        first, other, same_rows = cands[0], [], []
        for c in cands[1:]:
            if c[3] == first[3]:
                continue
            if sorted(c[4]) == sorted(first[4]) and all(c[4][r].equals(first[4][r]) for r in first[4]):
                same_rows.append(c[0])
            else:
                other.append(c[0])
        if other:
            errs.append(f"{job}: {first[0]} and {other} carry different shards of it "
                        f"({[r for r, _ in first[3]]} vs others); pass the directory to read")
        if same_rows:
            warns.append(f"{job}: {same_rows} carry the same rows as {first[0]} in other bytes (parquet records its "
                         f"writer's pyarrow version); read once, from {first[0]}")
        out[job] = first[:3]
    return out, errs, warns


def attempt_dir(dirs):
    """The export directory the attempts' _detail table is read from: the first
    that holds UNITS_DETAIL, else the first."""
    return next((d for d in dirs if os.path.exists(os.path.join(d, UNITS_DETAIL))), dirs[0])


def item_table(frames):
    """The item shards merged: one row per (benchmark, item_id), each feature's
    first non-null value, content_sha256 the first non-null."""
    import pandas as pd
    parts = [df[[c for c in KEY_COLS if c in df.columns] + feature_columns(df)]
             for _, kind, df in frames if kind == "items"]
    if not parts:
        return pd.DataFrame(columns=list(KEY_COLS))
    allf = pd.concat(parts, ignore_index=True)
    for c in allf.columns:
        if c not in KEY_COLS:
            allf[c] = allf[c].astype(float)
    return allf.groupby(["benchmark", "item_id"], sort=True, dropna=True).first().reset_index()


def _flag(v):
    return False if _missing(v) else bool(v)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def attempt_samples(g, lp_col="lp_answer"):
    """attempt_probe.item_features' sample dicts from one item's attempt rows
    (ans, tok_lp, tok_entropy, ent0, refuse, capped, forced, n_tok, and
    lp_answer, read from lp_col, where it is finite)."""
    out = []
    for d in g.sort_values("attempt").to_dict("records"):
        s = {"ans": None if _missing(d.get("answer")) else str(d["answer"]),
             "tok_lp": _num(d["tok_lp"]), "tok_entropy": _num(d["tok_entropy"]), "ent0": _num(d.get("ent0")),
             "refuse": _flag(d.get("refuse")), "capped": _flag(d["capped"]), "forced": _flag(d.get("forced")),
             "n_tok": int(_num(d["n_tokens"])) if math.isfinite(_num(d["n_tokens"])) else 0}
        lp = _num(d.get(lp_col))
        if math.isfinite(lp):
            s["lp_answer"] = lp
        out.append(s)
    return out


def logprob_semantics(manifest):
    """What the export's attempt token statistics are, from the manifest's
    `logprobs` (kaggle/strong_probe/strong_probe.py's logprobs_kind): 'raw' (the
    raw model distribution, full-vocabulary entropy: the 4B D2 definition),
    'raw_topk' (the engine's top-5 of the raw distribution: exact log-probs of
    the sampled tokens, entropy a lower bound; also an older notebook's 'raw
    model distribution ...'), 'processed' (vLLM V0's logprobs after the presence
    penalty, temperature and top-k/top-p), 'mixed' (more than one source) or
    'unknown' (none declared, or the notebook's recorder check failed:
    kinds.attempts.token_stats.recorder_check)."""
    man = manifest or {}
    rc = ((_attempt_spec(man).get("token_stats") or {}).get("recorder_check") or {}).get("status")
    if str(rc).upper() == "FAILED":
        return "unknown"
    s = str(man.get("logprobs") or "").strip().lower()
    if s in ("raw", "raw_full_vocab"):
        return "raw"
    if s == "raw_topk" or s.startswith("raw model distribution"):
        return "raw_topk"
    if "processed" in s:
        return "processed"
    if s == "mixed":
        return "mixed"
    return "unknown"


def _attempt_spec(manifest):
    spec = ((manifest or {}).get("kinds") or {}).get("attempts")
    return spec if isinstance(spec, dict) else {}


def lp_answer_plan(frames, manifest=None):
    """Where each attempt's answer log-prob comes from and whether it is one
    distribution for every attempt: {column, uniform, why}. A natural \\boxed{}
    answer's log-prob is read off the sampled distribution and a forced answer's
    off a greedy continuation; under V0 without the raw recorder the first is
    processed (presence penalty, temperature, top-k/top-p) and the second raw,
    and forced answers cluster on hard, truncated items, so a mixture gains a
    difficulty correlation from the engine alone. Uniform: an lp_forced column
    (the forced readout for every attempt: its column is used), the forced
    readout run for every attempt (the manifest's kinds.attempts.forced starting
    'every attempt', as the notebook writes it, or force_every_attempt: true),
    raw log-probs ('raw' or 'raw_topk': a sampled token's log-prob is exact), or
    every answer from one kind of readout (the `forced` flags)."""
    import pandas as pd
    sem = logprob_semantics(manifest)
    parts = [df for _, kind, df in frames if kind == "attempts"]
    if not parts:
        return {"column": "lp_answer", "uniform": True, "why": "no attempt shards"}
    cols = set().union(*(set(df.columns) for df in parts))
    if "lp_forced" in cols:
        return {"column": "lp_forced", "uniform": True, "readout": "forced",
                "why": "lp_forced: the greedy forced readout for every attempt, one raw distribution"}
    if "lp_answer" not in cols:
        return {"column": "lp_answer", "uniform": True, "why": "no answer log-probs"}
    spec = _attempt_spec(manifest)
    if spec.get("force_every_attempt") is True or str(spec.get("forced") or "").lower().startswith("every attempt"):
        return {"column": "lp_answer", "uniform": True, "readout": "forced",
                "why": "manifest: lp_answer is the greedy forced readout for every attempt (raw)"}
    if sem in ("raw", "raw_topk"):
        return {"column": "lp_answer", "uniform": True,
                "why": f"{sem} log-probs: sampled and forced readouts share one distribution"}
    if sem == "mixed":
        return {"column": "lp_answer", "uniform": False,
                "why": "token statistics from more than one source, and no forced readout for every attempt"}
    if "forced" not in cols:
        return {"column": "lp_answer", "uniform": False,
                "why": f"log-probs {sem} and no 'forced' column: sampled and forced readouts cannot be told apart"}
    alla = pd.concat([df[[c for c in ("lp_answer", "forced") if c in df.columns]] for df in parts],
                     ignore_index=True)
    have = np.isfinite(pd.to_numeric(alla["lp_answer"], errors="coerce").to_numpy(float))
    forced = alla["forced"].map(_flag).to_numpy(bool)
    nf, nn = int((have & forced).sum()), int((have & ~forced).sum())
    if nf and nn:
        return {"column": "lp_answer", "uniform": False, "sampled": nn, "forced": nf,
                "why": f"log-probs {sem}: {nn} answer log-probs read off the sampled (processed) distribution, "
                       f"{nf} off the greedy forced (raw) one"}
    return {"column": "lp_answer", "uniform": True, "sampled": nn, "forced": nf,
            "readout": "forced" if nf else "sampled",
            "why": f"every answer log-prob from the {'forced greedy' if nf else 'sampled'} readout"}


def attempt_table(frames, gold=None, lp_col="lp_answer"):
    """The attempt shards aggregated per (benchmark, item_id, design) by
    attempt_probe.item_features and spread wide as att_<design>_<feature>;
    graded (and top_correct) is NaN where the item has no reference answer.
    gold: {(benchmark, item_id): reference answer}; lp_col: the answer
    log-prob's column (lp_answer_plan)."""
    import pandas as pd
    parts = [df for _, kind, df in frames if kind == "attempts"]
    if not parts:
        return pd.DataFrame(columns=list(KEY_COLS))
    alla = pd.concat(parts, ignore_index=True)
    alla["design"] = alla["design"].fillna(DEFAULT_DESIGN) if "design" in alla.columns else DEFAULT_DESIGN
    gold = gold or {}
    rows = {}
    hashes = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (b, iid, dsg), g in alla.groupby(["benchmark", "item_id", "design"], sort=True):
            ref = gold.get((b, iid))
            f = AP.item_features(attempt_samples(g, lp_col), ref)
            f["k"] = float(len(g))
            for col, (name, _) in ATTEMPT_EXTRA.items():
                if col in g.columns:
                    v = g[col].to_numpy(float)
                    f[name] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            if ref is None:
                f["graded"] = float("nan")
                f["top_correct"] = float("nan")
            rec = rows.setdefault((b, iid), {})
            for k, v in f.items():
                rec[f"att_{dsg}_{k}"] = float(v)
            h = g["content_sha256"].dropna()
            if len(h):
                hashes[(b, iid)] = str(h.iloc[0])
    out = pd.DataFrame([{"benchmark": b, "item_id": i, "content_sha256": hashes.get((b, i)), **v}
                        for (b, i), v in rows.items()])
    return out


def combine(items, attempts):
    """Item and attempt tables on (benchmark, item_id); content hashes coalesced
    (a conflict is counted and the item's attempt features dropped)."""
    if not len(attempts):
        return items, 0
    if not len(items):
        return attempts, 0
    m = items.merge(attempts, on=["benchmark", "item_id"], how="outer", suffixes=("", "_att"))
    a, b = m["content_sha256"], m["content_sha256_att"]
    both = a.notna() & b.notna()
    conflict = both & (a.fillna("").str.lower() != b.fillna("").str.lower())
    att_cols = [c for c in attempts.columns if c not in KEY_COLS]
    m.loc[conflict, att_cols] = np.nan
    m["content_sha256"] = a.where(a.notna(), b)
    return m.drop(columns=["content_sha256_att"]).sort_values(["benchmark", "item_id"]).reset_index(drop=True), \
        int(conflict.sum())


# --- the entropy job (commit D) --------------------------------------------------------------

def entropy_semantics(manifest):
    """What the export says its entropy features are (kinds.entropy): the token
    statistics' source ('raw': the raw distribution's full-vocabulary entropy and
    log-prob, the features' definition), the recorder check, the version, the
    manifest's signs and primary against ENTROPY_SIGNS and ENTROPY_PRIMARY.
    readable: logprobs 'raw' and a recorder check that did not fail (else every
    feature is excluded, with why); primary_eligible: readable on the version and
    the job config the rule was fixed for (ENTROPY_RULE's config: ENTROPY_VERSION,
    ENTROPY_CFG), else why_not_eligible says which differs."""
    spec = ((manifest or {}).get("kinds") or {}).get(ENTROPY_KIND)
    spec = spec if isinstance(spec, dict) else {}
    ts = spec.get("token_stats") if isinstance(spec.get("token_stats"), dict) else {}
    lp = str(spec.get("logprobs") or ts.get("logprobs") or "").strip().lower() or None
    rc = (ts.get("recorder_check") or {}).get("status")
    signs = spec.get("signs") if isinstance(spec.get("signs"), dict) else {}
    disagree = {}
    for f, s in ENTROPY_FEATURES.items():
        if f in signs:
            try:
                ok = int(np.sign(float(signs[f]))) == s
            except (TypeError, ValueError):
                ok = False
            if not ok:
                disagree[f] = signs[f]
    why = None
    if not spec:
        why = f"no kinds.{ENTROPY_KIND} in the manifest: the token statistics are undeclared"
    elif str(rc).upper() == "FAILED":
        why = "the recorder check failed: the recorded statistics may not be the raw distribution's"
    elif lp != "raw":
        why = (f"token statistics {lp!r}, not the raw distribution's full-vocabulary ones ('raw') the features are "
               "defined on")
    want = ENTROPY_RULE["config"]
    not_eligible = why
    if why is None and spec.get("version") != want["version"]:
        not_eligible = f"version {spec.get('version')!r}, the rule was fixed for {want['version']!r}"
    elif why is None and spec.get("cfg") != want["cfg"]:
        not_eligible = (f"job config {spec.get('cfg')!r}, the rule was fixed for {want['cfg']!r} (strong_probe.py's "
                        "defaults: another model, seed, sampling or cut)")
    return {"logprobs": lp, "recorder_check": rc, "version": spec.get("version"), "expected_version": want["version"],
            "cfg": spec.get("cfg"), "expected_cfg": want["cfg"], "primary_manifest": spec.get("primary"),
            "signs_manifest": signs, "sign_disagreements": disagree, "sampling": spec.get("sampling"),
            "max_new_tokens": spec.get("max_new_tokens"), "units": spec.get("units"),
            "per_benchmark_units": spec.get("per_benchmark_units"), "readable": why is None, "why": why,
            "primary_eligible": not_eligible is None, "why_not_eligible": not_eligible}


def entropy_table(frames):
    """The entropy shards merged, one row per (benchmark, item_id) (each column's
    first non-null value), numeric columns as float; a degenerate unit's
    ENTROPY_SIGNS columns set missing. -> (table, {benchmark: degenerate rows})."""
    import pandas as pd
    parts = [df for _, kind, df in frames if kind == ENTROPY_KIND]
    if not parts:
        return pd.DataFrame(columns=list(KEY_COLS)), {}
    allf = pd.concat([df[[c for c in KEY_COLS if c in df.columns] + feature_columns(df)] for df in parts],
                     ignore_index=True)
    for c in allf.columns:
        if c not in KEY_COLS:
            allf[c] = allf[c].astype(float)
    deg = {}
    if ENTROPY_DEGENERATE in allf.columns:
        m = allf[ENTROPY_DEGENERATE].to_numpy(float) > 0.5
        deg = {str(b): int(n) for b, n in allf.loc[m, "benchmark"].value_counts().items()}
        cols = [c for c in ENTROPY_SIGNS if c in allf.columns]
        allf.loc[m, cols] = np.nan
    return allf.groupby(["benchmark", "item_id"], sort=True, dropna=True).first().reset_index(), deg


def entropy_nonfeature(c):
    return c in JOIN_COLS or c in ENTROPY_JOIN_EXTRA


def entropy_registry(joined, semantics=None):
    """{column: {sign, source, kind, role, label_free, usable, excluded, primary,
    distribution}} for the entropy table's columns: ENTROPY_SIGNS' columns are
    features (the primary and the exploratory ones), the other ent_* columns
    diagnostics (sign 0). Every feature is excluded (not usable) when semantics
    (entropy_semantics) is not readable."""
    sem = semantics or {}
    excluded = None if sem.get("readable") else (sem.get("why") or "the entropy table's semantics are unknown")
    reg = {}
    for c in joined.columns:
        if entropy_nonfeature(c):
            continue
        s = ENTROPY_SIGNS.get(c, 0)
        role = "primary" if c == ENTROPY_PRIMARY else "exploratory" if s else "diagnostic"
        reg[c] = {"sign": s, "source": "ENTROPY_SIGNS" if s else "diagnostic", "kind": ENTROPY_KIND if s else
                  "diagnostic", "role": role, "label_free": True, "usable": bool(s) and excluded is None,
                  "excluded": excluded if s else None, "primary": c == ENTROPY_PRIMARY,
                  "distribution": sem.get("logprobs")}
    return reg


def entropy_per_benchmark(joined, degenerate=None):
    """Per benchmark: the covered items, how many carry a finite primary, the
    degenerate rows set missing, and the means and rates of the columns."""
    out = {}
    for b, sub in joined.groupby("benchmark", sort=True):
        r = {"items": int(len(sub)), "primary_finite": int(np.isfinite(sub[ENTROPY_PRIMARY].to_numpy(float)).sum())
             if ENTROPY_PRIMARY in sub.columns else 0, "degenerate_excluded": int((degenerate or {}).get(b, 0))}
        for c in ENTROPY_COLUMNS:
            if c in sub.columns:
                v = sub[c].to_numpy(float)
                if np.isfinite(v).any():
                    r[("rate_" if ENTROPY_COLUMNS[c] == "bool" else "mean_") + c] = round(float(np.nanmean(v)), 5)
        out[str(b)] = r
    return out


# --- the public items and the join -----------------------------------------------------------

#: benchmarks whose reference answers grade the attempts (a diagnostic; swe_rebench's
#: grading criteria alone are 52 M characters, so they are not read elsewhere)
GOLD_BENCHES = ("matharena",)


def local_items(benches=BENCHES):
    """({bench: {item_id: item dict as paiec.data.load_pairs builds it}},
    {(bench, item_id): reference answer or None}) from data/<bench>/items.parquet;
    reference answers for GOLD_BENCHES only."""
    import pandas as pd
    from paiec import data as D
    items, gold = {}, {}
    for b in benches:
        path = os.path.join(D.DATA_DIR, b, "items.parquet")
        if not os.path.exists(path):
            continue
        cols = ["item_id", "benchmark_id", "content", "item_features"]
        try:
            it = pd.read_parquet(path, columns=cols + (["grading_criterion"] if b in GOLD_BENCHES else []))
        except Exception:
            it = pd.read_parquet(path, columns=cols)
        d = {}
        for r in it.itertuples(index=False):
            iid = str(r.item_id)
            d[iid] = {"item_content": D._clean(r.content), "item_features": D._clean(r.item_features),
                      "interactors": "", "benchmark_id": D._clean(r.benchmark_id)}
            ref = None
            gc = getattr(r, "grading_criterion", None)
            if isinstance(gc, str) and gc:
                try:
                    ref = json.loads(gc).get("reference_answer")
                except Exception:
                    ref = None
            gold[(b, iid)] = ref
        items[b] = d
    return items, gold


def item_keys(items):
    """{bench: {item_id: (key, key_official, text_key, sha256)}}: key is
    paiec.predict.item_key under the item's own benchmark_id (load_pairs' view),
    key_official under official's anonymous id (llmfeat's key / key_official /
    text_key), sha256 is HASH_DEF."""
    from paiec import data as D
    out = {}
    for b, d in items.items():
        anon = D.anon_id("benchmark", b)
        out[b] = {iid: (F.key_for(it, it["benchmark_id"]), F.key_for(it, anon), F.text_key(it), content_sha256(it))
                  for iid, it in d.items()}
    return out


def join(table, items, keys=None):
    """(joined DataFrame, report). One row per public item that gets features:
    rows whose content hash matches the local text ('direct'), and the other
    item_ids of the same text (same predict.item_key) where no row of their own
    exists ('duplicate', copied from the direct row with the smallest item_id).
    Rows of unknown benchmarks or item_ids, and rows whose hash is missing or
    does not match, are dropped and counted."""
    import pandas as pd
    keys = keys or item_keys(items)
    feats = [c for c in table.columns if c not in KEY_COLS]
    rep = {"per_benchmark": {}, "unknown_benchmark_rows": 0}
    direct = {}
    for b in items:
        rep["per_benchmark"][b] = {"items": len(items[b]), "unique_texts": len({v[0] for v in keys[b].values()}),
                                   "rows": 0, "unknown_item": 0, "hash_ok": 0, "hash_mismatch": 0,
                                   "hash_missing": 0}
    vals = table[feats].to_numpy(float) if feats else np.zeros((len(table), 0))
    hashes = table["content_sha256"] if "content_sha256" in table.columns else [None] * len(table)
    for i, (b, iid, h) in enumerate(zip(table["benchmark"], table["item_id"], hashes)):
        if b not in items:
            rep["unknown_benchmark_rows"] += 1
            continue
        r = rep["per_benchmark"][b]
        r["rows"] += 1
        if iid not in items[b]:
            r["unknown_item"] += 1
            continue
        h = "" if _missing(h) else str(h).strip().lower()
        if not h:
            r["hash_missing"] += 1
            continue
        if len(h) < MIN_HASH_HEX or not keys[b][iid][3].startswith(h):
            r["hash_mismatch"] += 1
            continue
        r["hash_ok"] += 1
        direct[(b, iid)] = vals[i]
    recs = []
    conflicts = {}
    for b in items:
        by_key = {}
        for iid, k in keys[b].items():
            by_key.setdefault(k[0], []).append(iid)
        n_dup = 0
        conflicts[b] = 0
        for k, ids in by_key.items():
            have = sorted(i for i in ids if (b, i) in direct)
            if not have:
                continue
            if len(have) > 1:
                v0 = direct[(b, have[0])]
                if any(not np.allclose(direct[(b, j)], v0, equal_nan=True) for j in have[1:]):
                    conflicts[b] += 1
            for iid in sorted(ids):
                src = "direct" if (b, iid) in direct else "duplicate"
                v = direct[(b, iid)] if src == "direct" else direct[(b, have[0])]
                n_dup += src == "duplicate"
                kk = keys[b][iid]
                recs.append({"benchmark": b, "item_id": iid, "key": kk[0], "key_official": kk[1],
                             "text_key": kk[2], "content_sha256": kk[3], "source": src,
                             **dict(zip(feats, v.tolist()))})
        r = rep["per_benchmark"][b]
        r["covered_direct"] = r["hash_ok"]
        r["covered_duplicate"] = n_dup
        r["covered_items"] = r["hash_ok"] + n_dup
        r["covered_unique_texts"] = len({keys[b][i][0] for (bb, i) in direct if bb == b})
        r["duplicate_texts_disagreeing"] = conflicts[b]
    cols = ["benchmark", "item_id", "key", "key_official", "text_key", "content_sha256", "source"] + feats
    joined = pd.DataFrame(recs, columns=cols)
    for b, r in rep["per_benchmark"].items():
        sub = joined[joined["benchmark"] == b]
        r["features_nonnull"] = {c: int(np.isfinite(sub[c].to_numpy(float)).sum()) for c in feats
                                 if np.isfinite(sub[c].to_numpy(float)).any()}
    tot = {k: sum(r.get(k, 0) for r in rep["per_benchmark"].values())
           for k in ("rows", "unknown_item", "hash_ok", "hash_mismatch", "hash_missing")}
    rep["totals"] = {**tot, "unknown_benchmark_rows": rep["unknown_benchmark_rows"]}
    return joined, rep


def response_coverage(joined, benches, keys):
    """Per benchmark, over the responses of paiec.data.load_pairs' pairs: the
    share whose item carries any feature (and any feature of each kind), and
    the count of responses whose item dict gives another predict.item_key than
    the join's (must be 0)."""
    kinds = {"rubric": [c for c in joined.columns if c.startswith("rubric_")],
             "attempts": [c for c in joined.columns if c.startswith("att_")]}
    return responses_coverage({"main": (joined, kinds)}, benches, keys)["main"]


def responses_coverage(parts, benches, keys):
    """response_coverage for several joined tables at once, reading each
    benchmark's pairs once: parts {label: (joined, {kind: [columns]})} ->
    {label: {benchmark: {pairs, responses, covered_share, covered_share_<kind>,
    item_key_mismatches}}}."""
    from paiec import data as D
    out = {lab: {} for lab in parts}
    for b in benches:
        cov = {}
        for lab, (joined, kinds) in parts.items():
            sub = joined[joined["benchmark"] == b]
            cov[lab] = (set(sub["item_id"]),
                        {k: set(sub.loc[np.isfinite(sub[cs].to_numpy(float)).any(1), "item_id"]) if cs else set()
                         for k, cs in kinds.items()})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pairs = D.load_pairs([b])
            except FileNotFoundError:
                continue
        n = mism = 0
        hit = {lab: 0 for lab in parts}
        hk = {lab: {k: 0 for k in kinds} for lab, (_, kinds) in parts.items()}
        kb = keys.get(b, {})
        for p in pairs:
            for r in p.responses:
                n += 1
                iid = str(r.item_key)
                for lab, (cov_any, cov_kind) in cov.items():
                    hit[lab] += iid in cov_any
                    for k in cov_kind:
                        hk[lab][k] += iid in cov_kind[k]
                if iid in kb and F.key_for(r.item, r.item["benchmark_id"]) != kb[iid][0]:
                    mism += 1
        for lab in parts:
            out[lab][b] = {"pairs": len(pairs), "responses": n, "covered_share": round(hit[lab] / max(n, 1), 4),
                           **{f"covered_share_{k}": round(v / max(n, 1), 4) for k, v in hk[lab].items()},
                           "item_key_mismatches": mism}
        del pairs
    return out


def stage_schema(args):
    print(json.dumps(SCHEMA, indent=1))


def check_exports(dirs, exports=None):
    """check_schema on each directory, and job_sources across them: {'ok',
    'dirs': {dir: report}, 'sources': {job: dir}, 'errors': [...]} (errors
    prefixed with their directory when there are several)."""
    exports = exports if exports is not None else [(d, load_shards(d)) for d in dirs]
    reps = {d: check_schema(d, loaded) for d, loaded in exports}
    src, src_err, src_warn = job_sources(exports)
    many = len(dirs) > 1
    errs = [f"{d}: {e}" if many else e for d, r in reps.items() for e in r["errors"]] + src_err
    warns = [f"{d}: {w}" if many else w for d, r in reps.items() for w in r["warnings"]] + src_warn
    return {"ok": not errs, "errors": errs, "warnings": warns, "dirs": reps,
            "sources": {j: v[0] for j, v in src.items()}}, src


def stage_check_schema(args):
    dirs = kaggle_dirs(args)
    rep, _ = check_exports(dirs)
    print(json.dumps(rep["dirs"][dirs[0]] if len(dirs) == 1 else rep, indent=1))
    if not rep["ok"]:
        log(f"check-schema: {len(rep['errors'])} error(s); the expected schema is `--stage schema`")
        raise SystemExit(1)
    n = sum(len(r["shards"]) for r in rep["dirs"].values())
    log(f"check-schema: ok ({n} shards, {len(rep['warnings'])} warnings; jobs from {rep['sources']})")


def _main_ingest(kdir, man, frames, items, gold, keys):
    """The rubric's and the attempts' features joined (ingest's first part):
    (joined, report) before the response coverage."""
    it = item_table(frames)
    plan = lp_answer_plan(frames, man)
    at = attempt_table(frames, gold, plan["column"])
    table, conflicts = combine(it, at)
    joined, rep = join(table, items, keys)
    rep["attempt_hash_conflicts"] = conflicts
    rep["attempt_semantics"] = attempt_semantics(man, joined, plan)
    tot = rep["totals"]
    if tot["rows"] and tot["hash_ok"] == 0:
        raise SystemExit(f"{kdir}: no row's content hash matches ({tot}); reconcile the hash definition: {HASH_DEF}")
    return joined, rep


def _entropy_ingest(kdir, man, frames, items, keys):
    """The entropy table joined (ingest's entropy part): (joined, report)."""
    table, deg = entropy_table(frames)
    joined, rep = join(table, items, keys)
    tot = rep["totals"]
    if tot["rows"] and tot["hash_ok"] == 0:
        raise SystemExit(f"{kdir}: no entropy row's content hash matches ({tot}); reconcile the hash definition: "
                         f"{HASH_DEF}")
    joined.insert(len(JOIN_COLS), "prompt_sha", [
        hashlib.sha256(str(items[b][i]["item_content"]).encode("utf-8")).hexdigest()[:16]
        for b, i in zip(joined["benchmark"], joined["item_id"])])
    rep["degenerate_rows"] = deg
    rep["per_benchmark_stats"] = entropy_per_benchmark(joined, deg)
    rep["semantics"] = entropy_semantics(man)
    rep["completeness"] = entropy_completeness(rep, rep["semantics"])
    return joined, rep


def entropy_completeness(rep, semantics=None):
    """Is the export complete (ENTROPY_RULE's `complete`)? Per parent: the units
    expected (the unique items by predict.item_key, as strong_probe.py's
    entropy_units counts them: join's unique_texts), the units covered (join's
    covered_unique_texts: keys with a row whose content hash matches), the export
    manifest's own count, and the items covered; complete when every parent's
    units are all covered."""
    per, pbu = {}, (semantics or {}).get("per_benchmark_units") or {}
    for b in PARENTS:
        r = (rep.get("per_benchmark") or {}).get(b)
        if r is None:
            per[b] = {"units_expected": None, "units_covered": 0, "complete": False, "why": "no local items"}
            continue
        covered = int(r.get("covered_unique_texts") or 0)
        per[b] = {"units_expected": int(r["unique_texts"]), "units_covered": covered,
                  "units_exported_manifest": pbu.get(b), "items": int(r["items"]),
                  "items_covered": int(r.get("covered_items") or 0),
                  "covered_share": round((r.get("covered_items") or 0) / r["items"], 4) if r["items"] else None,
                  "complete": covered >= int(r["unique_texts"])}
    return {"complete": all(v["complete"] for v in per.values()), "per_parent": per,
            "units_expected": sum(v.get("units_expected") or 0 for v in per.values()),
            "units_covered": sum(v["units_covered"] for v in per.values())}


def _unchanged(state, work, tmp, semantics):
    """Is the freshly written main table (tmp) what ingest recorded and what WORK
    holds, under the same attempt semantics?"""
    ing = state.get("ingest") or {}
    dig = file_sha256(tmp)[:16]
    return (ing.get("features_digest") == dig and features_digest(work) == dig
            and ing.get("attempt_semantics") == json.loads(json.dumps(H._jsonable(semantics))))


def stage_ingest(args):
    import gc
    t0 = time.time()
    dirs = kaggle_dirs(args)
    exports = [(d, load_shards(d)) for d in dirs]
    chk, src = check_exports(dirs, exports)
    for w in chk["warnings"]:
        log("warning:", w)
    if not chk["ok"] and not args.force:
        for e in chk["errors"]:
            log("error:", e)
        raise SystemExit("schema errors (above); fix the input, or --force to ingest what can be read")
    if not src:
        raise SystemExit(f"no rubric, attempt or entropy shards in {dirs}")
    state = H.load_json(args.out) or {}
    # before anything is read or written: the rule OUT's entropy results were read under must be this one
    rule_change = (entropy_rule_check(state.get(ENTROPY_KIND) or {}, getattr(args, "accept_rule_change", False))
                   if ENTROPY_KIND in src else None)
    items, gold = local_items()
    keys = item_keys(items)
    benches = list(items)
    os.makedirs(args.work, exist_ok=True)
    main = ent = None
    if "main" in src:
        kdir, man, frames = src["main"]
        joined, rep = _main_ingest(kdir, man, frames, items, gold, keys)
        path = os.path.join(args.work, "features.parquet")
        joined.to_parquet(path + ".tmp.parquet", index=False)
        main = {"dir": kdir, "man": man, "joined": joined, "rep": rep, "path": path,
                "unchanged": _unchanged(state, args.work, path + ".tmp.parquet", rep["attempt_semantics"])}
    else:
        log(f"ingest: no rubric or attempt shards in {dirs}: the features in {args.work} and their results are kept")
    if ENTROPY_KIND in src:
        kdir, man, frames = src[ENTROPY_KIND]
        joined_e, rep_e = _entropy_ingest(kdir, man, frames, items, keys)
        ent = {"dir": kdir, "man": man, "joined": joined_e, "rep": rep_e}
    del items, exports, src                       # load_pairs re-reads every column of items.parquet
    frames = None
    gc.collect()
    parts = {}
    if main and not main["unchanged"]:
        mj = main["joined"]
        parts["main"] = (mj, {"rubric": [c for c in mj.columns if c.startswith("rubric_")],
                              "attempts": [c for c in mj.columns if c.startswith("att_")]})
    if ent:
        parts[ENTROPY_KIND] = (ent["joined"], {ENTROPY_KIND: [c for c in (ENTROPY_PRIMARY,)
                                                              if c in ent["joined"].columns]})
    cov = responses_coverage(parts, benches, keys) if parts else {}
    if main:
        rep, path = main["rep"], main["path"]
        if main["unchanged"]:
            os.remove(path + ".tmp.parquet")
            log(f"ingest: the rubric and attempt features from {main['dir']} are byte-identical to those ingested "
                f"({(state.get('ingest') or {}).get('features_digest')}), under the same attempt semantics: "
                "ingest, meta and every result of theirs are kept as they are")
        else:
            os.replace(path + ".tmp.parquet", path)
            rep["responses"] = cov["main"]
            rep["features_digest"] = file_sha256(path)[:16]
            sch = chk["dirs"][main["dir"]]
            rep["schema"] = {k: sch[k] for k in ("ok", "errors", "warnings", "shards")}
            rep["wall_s"] = round(time.time() - t0, 1)
            drop_stale(state, args.work, rep["features_digest"], args.out, rep["attempt_semantics"])
            if "consistency" in (state.get(ENTROPY_KIND) or {}):
                del state[ENTROPY_KIND]["consistency"]
                log("ingest: the rubric and attempt features changed: entropy.consistency dropped")
            state["ingest"] = rep
            state["meta"] = meta(args, main["man"], main["dir"])
            _log_main_ingest(rep, path, len(main["joined"]))
    if ent:
        rep = ent["rep"]
        path = os.path.join(args.work, ENTROPY_TABLE)
        ent["joined"].to_parquet(path + ".tmp.parquet", index=False)
        os.replace(path + ".tmp.parquet", path)
        rep["responses"] = cov[ENTROPY_KIND]
        rep["features_digest"] = file_sha256(path)[:16]
        sch = chk["dirs"][ent["dir"]]
        rep["schema"] = {k: sch[k] for k in ("ok", "errors", "warnings", "shards")}
        rep["main_features"] = {
            "source": _rel(main["dir"]) if main else None,
            "status": ("byte-identical to those ingested before: kept, nothing rewritten" if main and main["unchanged"]
                       else "ingested" if main else "no rubric or attempt shards given: those in WORK kept"),
            "features_digest": (state.get("ingest") or {}).get("features_digest")}
        rep["wall_s"] = round(time.time() - t0, 1)
        est = state.get(ENTROPY_KIND) or {}
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        dropped = []
        if rule_change:                       # --accept-rule-change: the old rule and what was read under it
            est.setdefault("rule_history", []).append({**rule_change, "time": now})
            dropped = [k for k in ENTROPY_DOWNSTREAM if k in est]
            for k in dropped:
                del est[k]
            log(f"ingest: ENTROPY_RULE changed ({rule_change['digest'][:16]} -> {rule_change['replaced_by'][:16]}, "
                f"fields {rule_change['fields']}): the old rule is in entropy.rule_history; dropped entropy.{dropped}")
        dropped += drop_stale_entropy(est, rep["features_digest"], rep["semantics"])
        rule = entropy_rule_record()
        rep["rule_digest"] = rule["digest"]
        est["rule"] = rule
        est.setdefault("rule_first", {"digest": rule["digest"], "time": now, "features_digest": rep["features_digest"]})
        est["ingest"] = rep
        est["meta"] = entropy_meta(ent["man"], ent["dir"])
        sem, man_e = rep["semantics"], ent["man"] or {}
        est.setdefault("history", []).append({
            "stage": "ingest", "time": now, "features_digest": rep["features_digest"], "kaggle_dir": _rel(ent["dir"]),
            "version": sem.get("version"), "cfg": sem.get("cfg"), "model": man_e.get("model"),
            "notebook_digest": man_e.get("notebook_digest"), "created": man_e.get("created"),
            "units": sem.get("units"), "per_benchmark_units": sem.get("per_benchmark_units"),
            "complete": rep["completeness"]["complete"], "readable": sem.get("readable"),
            "primary_eligible": sem.get("primary_eligible"), "rule_digest": rule["digest"],
            "rule_changed": bool(rule_change), "dropped": dropped})
        state[ENTROPY_KIND] = est
        _log_entropy_ingest(rep, path, len(ent["joined"]))
    if (main and not main["unchanged"]) or ent:
        H.save_json(args.out, state)


def _rel(path):
    return os.path.relpath(path, ROOT) if os.path.abspath(path).startswith(ROOT) else path


def _log_main_ingest(rep, path, n):
    for b, r in rep["per_benchmark"].items():
        log(f"{b:18s} rows {r['rows']:5d}  hash ok {r['hash_ok']:5d}  mismatch {r['hash_mismatch']}  "
            f"missing {r['hash_missing']}  unknown {r['unknown_item']}  covered {r['covered_items']}/{r['items']} "
            f"items ({r['covered_duplicate']} by duplicate)  responses "
            f"{rep['responses'].get(b, {}).get('covered_share')}")
    sem = rep["attempt_semantics"]
    if sem["has_attempts"]:
        log(f"attempts: log-probs {sem['logprobs']} ({sem['logprobs_manifest']!r}), presence penalty "
            f"{sem['presence_penalty']}; answer log-prob: {sem['lp_answer']['why']}"
            + ("" if sem["lp_answer"]["uniform"] else " -> lp_answer excluded")
            + f"; primary {sem['primary_attempt']}; beside the 4B D2 lead: {sem['d2']['comparable']} "
              f"({sem['d2']['reason']})")
    log(f"ingest: {n} items with features -> {path}; {rep['wall_s']}s")


def _log_entropy_ingest(rep, path, n):
    for b, r in rep["per_benchmark"].items():
        st = rep["per_benchmark_stats"].get(b, {})
        log(f"entropy {b:18s} rows {r['rows']:5d}  hash ok {r['hash_ok']:5d}  mismatch {r['hash_mismatch']}  "
            f"missing {r['hash_missing']}  unknown {r['unknown_item']}  covered {r['covered_items']}/{r['items']} "
            f"items, {st.get('primary_finite', 0)} with {ENTROPY_PRIMARY} ({st.get('degenerate_excluded', 0)} "
            f"degenerate excluded)  responses {rep['responses'].get(b, {}).get('covered_share')}")
    sem = rep["semantics"]
    log(f"entropy: version {sem['version']} (rule fixed for {ENTROPY_VERSION}), log-probs {sem['logprobs']}, recorder "
        f"check {sem['recorder_check']}" + ("" if sem["readable"] else f" -> features excluded: {sem['why']}")
        + f"; primary {ENTROPY_PRIMARY} eligible: {sem['primary_eligible']}")
    log(f"ingest: {n} items with entropy features -> {path}; {rep['wall_s']}s")


def entropy_meta(man, kdir):
    """The entropy export's provenance for OUT's entropy section."""
    man = man or {}
    spec = (man.get("kinds") or {}).get(ENTROPY_KIND)
    return {"kaggle_dir": _rel(kdir), "script_digest_strong": H.digest(["experiments/strong_llm_eval.py"]),
            "manifest": {k: man.get(k) for k in ("schema_version", "model", "created", "wall_s", "sessions", "gpu",
                                                 "notebook_digest", "hash", "slug")},
            "kind": spec if isinstance(spec, dict) else None,
            "primary": ENTROPY_PRIMARY, "signs": dict(ENTROPY_SIGNS), "version": ENTROPY_VERSION,
            "prong_fields": {"feature": PRONG_FIELD}, "gate": H.GATE, "keep_signs": KEEP_SIGNS}


def entropy_rule_check(est, accept=False):
    """OUT's entropy.rule against ENTROPY_RULE, before ingest writes anything:
    None when they agree (or OUT holds no rule yet); when they differ (another
    digest, or a stored rule that no longer matches its own digest), a
    rule_history entry if accept (--accept-rule-change), else SystemExit."""
    prev = est.get("rule")
    if not isinstance(prev, dict):
        return None
    content = entropy_rule_digest(prev)
    stored = prev.get("digest") or content
    now = entropy_rule_digest()
    if stored == content == now:
        return None
    fields = sorted(k for k in set(prev) | set(ENTROPY_RULE) if k != "digest"
                    and json.dumps(H._jsonable(prev.get(k)), sort_keys=True)
                    != json.dumps(H._jsonable(ENTROPY_RULE.get(k)), sort_keys=True))
    what = (f"ENTROPY_RULE (digest {now[:16]}) differs from the rule this OUT's entropy results were read under "
            f"({stored[:16]}{'' if stored == content else ', which no longer matches its own digest'}; fields "
            f"{fields})")
    if not accept:
        raise SystemExit(what + ": the decision was fixed before the output was read. Restore the rule, or pass "
                         "--accept-rule-change to record the change (the old rule goes to entropy.rule_history, the "
                         "results read under it are dropped, and the verdict's call is marked RULE CHANGED)")
    return {"rule": prev, "digest": stored, "digest_of_content": content, "replaced_by": now, "fields": fields}


def drop_stale_entropy(est, digest, semantics=None):
    """When the entropy table (or what its manifest says it is) changes, drop the
    entropy section's ENTROPY_DOWNSTREAM parts. Returns the dropped names."""
    ing = est.get("ingest") or {}
    prev = ing.get("features_digest")
    sem_changed = (semantics is not None and "semantics" in ing
                   and ing.get("semantics") != json.loads(json.dumps(H._jsonable(semantics))))
    dropped = []
    if prev and (prev != digest or sem_changed):
        dropped = [k for k in ENTROPY_DOWNSTREAM if k in est]
        for k in dropped:
            del est[k]
    if dropped:
        log(f"ingest: the entropy features ({prev} -> {digest}) or their semantics changed; dropped entropy.{dropped}")
    return dropped


def drop_stale(state, work, digest, out=OUT, semantics=None):
    """When the ingested features change (another Kaggle session, a re-export),
    drop every result computed from the old ones: the DOWNSTREAM sections of
    state and WORK/oof.json (the heads' out-of-fold predictions); when only the
    attempts' semantics change (a manifest that now declares other log-probs),
    the DOWNSTREAM sections. Returns the dropped names."""
    ing = state.get("ingest") or {}
    prev = ing.get("features_digest")
    oof_path = os.path.join(work, "oof.json")
    oof = H.load_json(oof_path) if os.path.exists(oof_path) else None
    stale_oof = oof is not None and (not isinstance(oof, dict) or oof.get("features_digest") != digest)
    sem_changed = semantics is not None and ing.get("attempt_semantics") != json.loads(json.dumps(H._jsonable(semantics)))
    dropped = []
    if (prev and prev != digest) or (prev and sem_changed):
        dropped = [k for k in DOWNSTREAM if k in state]
        for k in dropped:
            del state[k]
    if stale_oof:
        os.remove(oof_path)
        dropped.append("oof.json")
    if dropped:
        log(f"ingest: the features ({prev} -> {digest}) or the attempts' semantics changed; dropped {dropped} "
            f"({out}, {work})")
    return dropped


# --- features, targets --------------------------------------------------------------------

def load_joined(work):
    import pandas as pd
    path = os.path.join(work, "features.parquet")
    if not os.path.exists(path):
        raise SystemExit("run --stage ingest first")
    return pd.read_parquet(path)


def load_entropy(work):
    """WORK/entropy.parquet: the entropy job's features joined to the public items."""
    import pandas as pd
    path = os.path.join(work, ENTROPY_TABLE)
    if not os.path.exists(path):
        raise SystemExit(f"no {path}: run --stage ingest on an export with the entropy table first")
    return pd.read_parquet(path)


def entropy_digest(work):
    """The digest --stage ingest records for WORK/entropy.parquet."""
    path = os.path.join(work, ENTROPY_TABLE)
    return file_sha256(path)[:16] if os.path.exists(path) else None


def entropy_state(state):
    """OUT's entropy section, which ingest must have written."""
    est = state.get(ENTROPY_KIND)
    if not isinstance(est, dict) or "ingest" not in est:
        raise SystemExit("no entropy section in OUT: run --stage ingest on an export with the entropy table first")
    return est


def entropy_work_checked(work, est):
    """(WORK/entropy.parquet, its digest), refused (SystemExit) unless it is the
    table --stage ingest recorded in this OUT's entropy section: the entropy
    stages never mix one --work's table with another --out's results."""
    joined = load_entropy(work)
    dig = entropy_digest(work)
    want = (est.get("ingest") or {}).get("features_digest")
    if dig != want:
        raise SystemExit(f"{os.path.join(work, ENTROPY_TABLE)} ({dig}) is not the table --stage ingest recorded in "
                         f"this OUT's entropy section ({want}): pass the --work and --out ingest used, or ingest again")
    return joined, dig


def entropy_stage_registry(joined, est):
    return entropy_registry(joined, (est.get("ingest") or {}).get("semantics"))


def _save_part_entropy(path, part, value):
    st = H.load_json(path) or {}
    st.setdefault(ENTROPY_KIND, {})[part] = value
    H.save_json(path, st)


def attempt_designs(columns):
    """{design: {feature: column}} from att_<design>_<feature> columns."""
    names = tuple(ATT_SIGNS) + ATT_DIAG
    out = {}
    for c in columns:
        if not c.startswith("att_"):
            continue
        rest = c[4:]
        for f in sorted(set(names), key=len, reverse=True):
            if rest.endswith("_" + f):
                out.setdefault(rest[:-len(f) - 1], {})[f] = c
                break
    return out


JOIN_COLS = ("benchmark", "item_id", "key", "key_official", "text_key", "content_sha256", "source")
#: attempt aggregates read off token log-probs (their meaning follows the manifest's `logprobs`,
#: except the <base>_raw ones)
LOGPROB_FEATURES = set(RAW_BASES) | {b + RAW_SUFFIX for b in RAW_BASES} | {"lp_answer", "lp_answer_top", "ent0"}


def primary_attempt(joined, logprobs="unknown"):
    """The attempt primary (PRIMARY_ATTEMPT_RULE): the first feature whose
    semantics requirement logprobs meets and that some design carries with a
    finite value on at least ATT_MIN_ITEMS items (a column the notebook writes but
    could not fill does not count); None without one."""
    designs = attempt_designs(joined.columns)
    for f, need in PRIMARY_ATTEMPT_RULE:
        if need is not None and logprobs != need:
            continue
        for fs in designs.values():
            if f in fs and int(np.isfinite(joined[fs[f]].to_numpy(float)).sum()) >= ATT_MIN_ITEMS:
                return f
    return None


def d2_comparable(primary, logprobs):
    """(comparable, reason): may the attempt primary be set beside the 4B attempt
    probe's lead (D2: the mean full-vocabulary entropy of the raw next-token
    distribution over a short chain of thought, experiments/attempt_probe.py)?"""
    if primary is None:
        return False, "no attempt primary"
    if primary.endswith(RAW_SUFFIX) or (primary == "tok_entropy" and logprobs == "raw"):
        return True, (f"{primary}: the raw distribution's full-vocabulary mean entropy, D2's definition "
                      "(D2's attempts were capped at 512 tokens)")
    if logprobs == "raw":
        return False, f"{primary}: a fixed window of the raw full-vocabulary entropy, not D2's whole-attempt mean"
    if logprobs == "raw_topk":
        return False, (f"{primary}: a top-5 lower bound on the raw distribution (D2: the full vocabulary); "
                       "flagged, not withheld")
    return False, (f"{primary}: a top-5 lower bound on a {logprobs} distribution (vLLM V0: after the presence "
                   "penalty, temperature and top-k/top-p); D2 is the full-vocabulary entropy of the raw one")


def attempt_semantics(manifest, joined, lp_plan):
    """What the attempt features measure: the manifest's log-prob semantics, the
    presence penalty, the answer log-prob's source (lp_answer_plan), the raw
    columns present, the resolved primary and whether it can be set beside D2."""
    man = manifest or {}
    spec = _attempt_spec(man)
    samp = spec.get("sampling") if isinstance(spec.get("sampling"), dict) else {}
    sem = logprob_semantics(man)
    designs = attempt_designs(joined.columns)
    prim = primary_attempt(joined, sem)
    comp, why = d2_comparable(prim, sem)
    pp = samp.get("presence_penalty")
    notes = []
    if designs and sem not in ("raw", "raw_topk") and pp:
        notes.append(f"presence penalty {pp} with {sem} log-probs: a per-token entropy or log-prob depends on "
                     "what the attempt generated before, so it drifts with position and generation length")
    if designs and not lp_plan.get("uniform", True):
        notes.append(f"{', '.join(LP_ANSWER_FEATURES)} excluded: {lp_plan['why']}")
    if designs and sem == "mixed":
        notes.append("token statistics from more than one source (manifest logprobs 'mixed'): every entropy "
                     "and log-prob feature but a uniform answer log-prob is excluded")
    ts = spec.get("token_stats") if isinstance(spec.get("token_stats"), dict) else {}
    return {"has_attempts": bool(designs), "logprobs": sem, "logprobs_manifest": man.get("logprobs"),
            "logprobs_engine": man.get("logprobs_engine"), "token_sources": ts.get("sources"),
            "recorder_check": (ts.get("recorder_check") or {}).get("status"),
            "presence_penalty": pp, "entropy": spec.get("entropy"), "lp_answer": lp_plan,
            "raw_features": sorted({f for fs in designs.values() for f in fs if f.endswith(RAW_SUFFIX)}),
            "primary_attempt": prim, "primary_order": list(PRIMARY_ATTEMPT_ORDER),
            "d2": {"comparable": comp, "reason": why}, "notes": notes}


def stage_semantics(state, joined):
    """The ingest's attempt_semantics; an ingest from before these checks cannot
    be read for attempts."""
    sem = (state.get("ingest") or {}).get("attempt_semantics")
    if sem is None and attempt_designs(joined.columns):
        raise SystemExit("the ingest predates the attempt log-prob checks: run --stage ingest again")
    return sem


def stage_registry(joined, state):
    return registry(joined, (state.get("meta") or {}).get("manifest"), stage_semantics(state, joined))


def registry(joined, manifest=None, semantics=None):
    """{feature: {sign, source, kind, label_free, usable}} for every feature
    column. usable: it may enter the harness (a declared sign, label-free, not
    excluded). Attempt aggregates take attempt_probe.FEATURES' signs; graded and
    top_correct need the reference answer (not label-free); n_distinct and k
    are descriptions, not features; lp_answer and lp_answer_top are excluded
    when semantics (attempt_semantics) says the answer log-prob mixes readouts,
    and every other token statistic (but the <base>_raw ones) when the manifest
    says 'mixed'. Attempt features carry primary (the resolved attempt primary)
    and, for log-prob features, the distribution they are read off."""
    over = {}
    for spec in ((manifest or {}).get("kinds") or {}).values():
        if isinstance(spec, dict) and isinstance(spec.get("signs"), dict):
            over.update(spec["signs"])
    reg = {}
    att = {c: (d, f) for d, fs in attempt_designs(joined.columns).items() for f, c in fs.items()}
    lp_plan = (semantics or {}).get("lp_answer") or {}
    sem = (semantics or {}).get("logprobs") or logprob_semantics(manifest)
    prim = primary_attempt(joined, sem) if att else None
    for c in joined.columns:
        if c in JOIN_COLS:
            continue
        if c in att:
            d, f = att[c]
            s = ATT_SIGNS.get(f, 0)
            src = ("attempt_probe.FEATURES" if f in AP.FEATURES else "ATTEMPT_EXTRA") if s else "description"
            excluded = None
            if f in LP_ANSWER_FEATURES and not lp_plan.get("uniform", True):
                excluded = "the answer log-prob mixes readouts: " + lp_plan.get("why", "")
            elif (f in LOGPROB_FEATURES and f not in LP_ANSWER_FEATURES and not f.endswith(RAW_SUFFIX)
                  and sem == "mixed"):
                excluded = "token statistics from more than one source (manifest logprobs 'mixed')"
            reg[c] = {"sign": s, "source": src, "kind": "attempt",
                      "design": d, "feature": f, "label_free": f not in ("graded", "top_correct"),
                      "usable": bool(s) and excluded is None, "excluded": excluded, "primary": f == prim}
            if f in LOGPROB_FEATURES:
                reg[c]["distribution"] = ("raw" if f.endswith(RAW_SUFFIX) else "raw (greedy forced readout)"
                                          if f in LP_ANSWER_FEATURES and lp_plan.get("readout") == "forced"
                                          else sem)
            continue
        s, src = declared_sign(c, over)
        kind = "rubric" if c.startswith("rubric_") else "item"
        reg[c] = {"sign": s, "source": src, "kind": kind if s else "diagnostic", "label_free": True,
                  "usable": bool(s)}
    return reg


def oriented_maps(joined, reg, names=None):
    """{feature: {item_id: sign * x}} (raw x where no sign is declared)."""
    out = {}
    for c in (names or reg):
        s = reg[c]["sign"] or 1
        v = joined[c].to_numpy(float)
        ok = np.isfinite(v)
        out[c] = dict(zip(joined["item_id"].to_numpy()[ok].tolist(), (s * v[ok]).tolist()))
    return out


def rubric_levels(reg):
    return [c for c, v in reg.items() if v["kind"] == "rubric" and v["sign"] != 0]


def rubric_sum(joined, reg):
    """The unit-weight mean of the oriented rubric levels, each standardised
    within benchmark over the items that carry every level: {item_id: x}."""
    cols = rubric_levels(reg)
    if len(cols) < 2:
        return {}
    X = joined[cols].to_numpy(float) * np.array([reg[c]["sign"] for c in cols], float)
    ok = np.isfinite(X).all(1)
    bench = joined["benchmark"].to_numpy()[ok]
    Z = HS.zscore_within(X[ok], bench)
    return dict(zip(joined["item_id"].to_numpy()[ok].tolist(), Z.mean(1).tolist()))


def _targets_digest():
    from paiec import data as D
    parts = [H.digest(["paiec/testlike.py", "paiec/official.py", "paiec/rasch.py", "paiec/data.py",
                       "paiec/evaluator.py"]), H.HONEST_SALT, str(H.HONEST_FOLDS)]
    for b in PARENTS:
        for f in ("response.parquet", "items.parquet"):
            p = os.path.join(D.DATA_DIR, b, f)
            parts.append(f"{b}/{f}:{os.path.getsize(p) if os.path.exists(p) else -1}")
    return F.digest(parts)[:16]


def targets(work=WORK, refresh=False):
    """(mean honest difficulty per parent {item_id: b}, the five fold maps, info):
    llm4b_close.honest_targets, cached in work/targets.json under a digest of the
    code and data it depends on."""
    path = os.path.join(work, "targets.json")
    dig = _targets_digest()
    c = H.load_json(path)
    if c and c.get("digest") == dig and not refresh:
        return c["mean"], c["folds"], c["info"]
    t0 = time.time()
    mean, folds, info = L4.honest_targets()
    H.save_json(path, {"digest": dig, "mean": mean, "folds": folds, "info": info})
    log(f"honest targets in {time.time() - t0:.0f}s -> {path}")
    return mean, folds, info


def unit_items(joined):
    """{bench: {item_id: item dict}} for the benchmarks in joined (the official
    item view of itemcov_eval.load_items, which is what corr_block's groups and
    controls read)."""
    return {b: ICE.load_items(b) for b in sorted(set(joined["benchmark"]))}


# --- signs --------------------------------------------------------------------------------

def _controls(items, ks):
    L = [IC.log_length(items[k]) for k in ks]
    pos = [IC.position(items[k]) for k in ks]
    if all(v is not None for v in pos):         # matharena's problem_idx
        return np.column_stack([L, pos])
    return np.asarray(L, float)


def agreement(per, field="spearman_within", by_unit=None):
    """Signs of an oriented statistic over the parents where the feature applies:
    how many are positive (the declared sign) and negative. by_unit: {parent:
    field} overriding field on those parents (PRONG_FIELD). A parent where the
    feature applies but the statistic is missing counts as a unit that is not
    positive (a missing position-net statistic never falls back to one that
    carries position)."""
    fields = {u: (by_unit or {}).get(u, field) for u in PARENTS}
    vals, missing = {}, []
    for u, v in per.items():
        if u not in PARENTS or not v.get("applies"):
            continue
        s = v.get(fields[u])
        if isinstance(s, dict) and s.get("est") is not None:
            vals[u] = s["est"]
        else:
            missing.append(u)
    pos = sum(1 for r in vals.values() if r > 0)
    neg = sum(1 for r in vals.values() if r < 0)
    return {"field": field, "fields": {u: fields[u] for u in list(vals) + missing}, "units": len(vals) + len(missing),
            "positive": pos, "negative": neg, "missing": missing, "same_sign": max(pos, neg),
            "declared_sign_ok": pos >= KEEP_SIGNS}


def random_effects_over(per, field):
    """hidden_state_probe.random_effects over the parents with the statistic."""
    rs, ns = [], []
    for u, v in per.items():
        if u in PARENTS and v.get("applies") and isinstance(v.get(field), dict):
            rs.append(v[field]["est"])
            ns.append(v["n"])
    if len(rs) < 2:
        return None
    return HS.random_effects(rs, ns)


def sign_entry(x_map, target, items, boots, seed, declared):
    """One feature's signs: per parent corr_block (+ matharena's text-bearing
    items and its contest years), sign rules, agreement, random effects."""
    per = {}
    for bi, b in enumerate(PARENTS):
        tgt = target.get(b, {})
        if b not in items:
            continue
        ks, x, dd = L4._vectors(x_map, tgt, list(items[b]))
        if len(ks) < MIN_ITEMS or np.std(x) == 0:
            per[b] = {"n": len(ks), "applies": False}
            continue
        g = [IC.features(items[b][k]).get(GROUP_KEY.get(b, ""), "") for k in ks]
        C = _controls(items[b], ks)
        per[b] = {"applies": True, **L4.corr_block(x, dd, g, C, boots, seed=seed + bi)}
        if b == "matharena":
            text = L4.text_bearing({k: items[b][k] for k in ks})
            comp = {k: IC.features(items[b][k]).get("competition", "") for k in ks}
            m = np.array([text[k] for k in ks])
            ga = np.array(g, object)
            Ca = np.asarray(C)
            if m.sum() >= MIN_ITEMS and np.std(x[m]) > 0:
                per["matharena text-bearing"] = L4.corr_block(x[m], dd[m], ga[m], Ca[m], boots, seed=seed + 7)
            for yr, sel in (("2025", lambda c: c not in YEAR_2026), ("2026", lambda c: c in YEAR_2026)):
                my = m & np.array([sel(comp[k]) for k in ks])
                if my.sum() >= MIN_ITEMS // 2 and np.std(x[my]) > 0:
                    per[f"matharena text-bearing {yr}"] = L4.corr_block(
                        x[my], dd[my], ga[my], Ca[my], boots, seed=seed + (8 if yr == "2025" else 9))
    entry = {"declared_sign": declared, "units": per,
             "agreement_prong": agreement(per, "spearman_within", PRONG_FIELD),
             "agreement_within": agreement(per, "spearman_within"),
             "agreement": agreement(per, "spearman"),
             "random_effects_within": random_effects_over(per, "spearman_within"),
             "random_effects": random_effects_over(per, "spearman")}
    # x is oriented already, so the declared sign llm4b_close.sign_rule checks is +1
    for key, field in (("sign_rule", "spearman"), ("sign_rule_within", "spearman_within")):
        have = {u: v for u, v in per.items() if isinstance(v.get(field), dict) or not v.get("applies")}
        entry[key] = L4.sign_rule(have, 1 if declared else 0, field)
    return entry


# --- heads --------------------------------------------------------------------------------

class _Rows:
    """The row set hidden_state_probe.lobo reads: bench, benches, y, n."""

    def __init__(self, bench, y, iid, g, C):
        self.bench, self.y, self.iid, self.g, self.C = bench, y, iid, g, C
        self.benches = [b for b in PARENTS if (bench == b).any()]
        self.n = len(y)


class RidgeHead:
    """Ridge on features standardised within benchmark (hidden_state_probe's
    ProfileHead on a given matrix); configs: the penalty."""

    def __init__(self, X, bench, alphas=ALPHAS):
        self.X = HS.zscore_within(X, bench)
        self.configs = [(a,) for a in alphas]

    def fit_predict(self, P, train, test, cfg, yz):
        w = HS.weights(P.bench[train])
        beta = HS.ridge(self.X[train], yz[train], w, cfg[0])
        return self.X[test] @ beta, {"beta": beta}


def head_specs(reg):
    """name -> (columns, primary?, position-free?) for the heads with at least
    two columns; POSFREE_HEAD is the primary's columns with matharena's
    residualised on position (reported beside the primary, never primary)."""
    lv = rubric_levels(reg)
    diag = [c for c, v in reg.items() if c.startswith("rubric_") and v["sign"] == 0]
    item = [c for c, v in reg.items() if v["kind"] in ("rubric", "item") and v["sign"] != 0]
    specs = {PRIMARY_HEAD: (lv, True, False), POSFREE_HEAD: (lv, False, True),
             "rubric_all_ridge": (lv + diag, False, False), "judge_ridge": (item, False, False)}
    out, seen = {}, set()
    for name, (cols, prim, posfree) in specs.items():
        if len(cols) >= 2 and (tuple(cols), posfree) not in seen:
            out[name] = (cols, prim, posfree)
            seen.add((tuple(cols), posfree))
    return out


def residualise_position(X, bench, iid, items):
    """X with each column's linear position component removed within group
    (GROUP_KEY: competition on matharena), on every benchmark whose items carry
    a position (IC.position: problem_idx); group means are kept. Rows without a
    position, or with a non-finite column, are left as they are. -> (X', info)."""
    X = np.array(X, float)
    info = {}
    for b in np.unique(bench):
        its = items.get(b)
        if not its:
            continue
        rows = np.flatnonzero(bench == b)
        pos = np.array([IC.position(its[i]) if i in its else None for i in iid[rows]], object)
        have = np.array([p is not None for p in pos]) & np.isfinite(X[rows]).all(1)
        if have.sum() < 3:
            continue
        r = rows[have]
        p = pos[have].astype(float)
        g = np.array([IC.features(its[i]).get(GROUP_KEY.get(b, ""), "") for i in iid[r]], object).astype(str)
        _, inv = np.unique(g, return_inverse=True)
        cnt = np.bincount(inv)

        def dm(v):
            return v - (np.bincount(inv, v) / cnt)[inv]

        pd_ = dm(p)
        den = float(pd_ @ pd_)
        if den <= 0:
            continue
        slopes = []
        for j in range(X.shape[1]):
            beta = float(pd_ @ dm(X[r, j])) / den
            X[r, j] = X[r, j] - beta * pd_
            slopes.append(round(beta, 6))
        info[str(b)] = {"rows": int(len(r)), "rows_without_position": int((~have).sum()), "slopes": slopes}
    return X, info


def fit_head(joined, cols, target, items, boots, seed=0, alphas=ALPHAS, posfree=False):
    """Nested leave-one-parent-out ridge (hidden_state_probe.lobo) on the parent
    items with every column and a target; out-of-fold predictions there, and on
    the other benchmarks' items the fit on all parents with the penalty the
    outer folds chose most often. posfree: the columns residualised on position
    first (residualise_position). -> (entry, {item_id: prediction})."""
    X = joined[cols].to_numpy(float)
    bench = joined["benchmark"].to_numpy().astype(str)
    iid = joined["item_id"].to_numpy().astype(str)
    pos_info = None
    if posfree:
        X, pos_info = residualise_position(X, bench, iid, items)
    ok = np.isfinite(X).all(1)
    y = np.array([target.get(b, {}).get(i, np.nan) for b, i in zip(bench, iid)], float)
    tr = ok & np.isfinite(y) & np.isin(bench, PARENTS)
    if len(set(bench[tr])) < 3:
        return {"skipped": f"parents with every column and a target: {sorted(set(bench[tr]))} (< 3)"}, {}
    g = np.array([IC.features(items[b][i]).get(GROUP_KEY.get(b, ""), "") for b, i in zip(bench[tr], iid[tr])],
                 object)
    P = _Rows(bench[tr], y[tr], iid[tr], g, None)
    head = RidgeHead(X[tr], P.bench, alphas)
    yz = HS.zscore_within(P.y[:, None], P.bench)[:, 0]
    oof, folds = HS.lobo(P, head, yz)
    per = {}
    for i, q in enumerate(P.benches):
        m = P.bench == q
        ks = list(P.iid[m])
        per[q] = {"applies": True, **L4.corr_block(oof[m], P.y[m], P.g[m], _controls(items[q], ks), boots,
                                                   seed=seed + 97 * i + 11)}
    # a statistic corr_block could not compute (a constant prediction) is NaN here
    pr = [(per[q].get("pearson") or {}).get("est", float("nan")) for q in P.benches]
    pw = [(per[q].get("pearson_within") or {}).get("est", float("nan")) for q in P.benches]
    ns = [per[q]["n"] for q in P.benches]
    # the penalty the outer folds chose most often (ties: the larger penalty)
    cfg = max((tuple(f["choice"]) for f in folds.values()),
              key=lambda c: (sum(tuple(f["choice"]) == c for f in folds.values()), c[0]))
    preds = dict(zip(P.iid.tolist(), oof.tolist()))
    other = ok & ~np.isin(bench, PARENTS)
    if other.any():
        beta = HS.ridge(head.X, yz, HS.weights(P.bench), cfg[0])
        Xo = HS.zscore_within(X[other], bench[other])
        preds.update(dict(zip(iid[other].tolist(), (Xo @ beta).tolist())))
    # the sign prong: out-of-fold r per parent, on matharena net of position (HEAD_PRONG_FIELD); a
    # parent whose statistic is missing is not positive
    prong = {q: (per[q].get(HEAD_PRONG_FIELD.get(q, "pearson")) or {}).get("est") for q in P.benches}
    entry = {"columns": cols, "rows": int(P.n), "posfree": bool(posfree), "position_residualised": pos_info,
             "folds": folds, "per_parent": per,
             "prong": {"fields": {q: HEAD_PRONG_FIELD.get(q, "pearson") for q in P.benches}, "values": prong},
             "positive_parents_prong": int(sum(1 for r in prong.values() if r is not None and r > 0)),
             "pearson": dict(zip(P.benches, pr)), "pearson_within_group": dict(zip(P.benches, pw)),
             "mean_pearson": round(float(np.nanmean(pr)), 4) if np.isfinite(pr).any() else None,
             "mean_pearson_within_group": round(float(np.nanmean(pw)), 4) if np.isfinite(pw).any() else None,
             "positive_parents": int(sum(r > 0 for r in pr)),
             "positive_parents_within_group": int(sum(r > 0 for r in pw)),
             "parents_r_at_least": int(sum(r >= KEEP_R for r in pr)),
             "random_effects": HS.random_effects(pr, ns) if len(pr) >= 3 else None,
             "random_effects_within_group": HS.random_effects(pw, ns) if len(pw) >= 3 else None,
             "config_other_benchmarks": list(cfg), "predicted_other_benchmarks": int(other.sum())}
    entry["r_prong"] = bool(entry["mean_pearson"] is not None and entry["mean_pearson"] >= KEEP_R
                            and entry["positive_parents"] >= KEEP_SIGNS)
    return entry, preds


def _log_sign(name, e, prefix=""):
    short = {"matharena text-bearing": "text", "matharena text-bearing 2025": "2025",
             "matharena text-bearing 2026": "2026"}
    log(f"{prefix}{name:30s} " + "  ".join(
        f"{short.get(u, u[:10])} {v['spearman_within']['est']:+.3f}" for u, v in e["units"].items()
        if isinstance(v.get("spearman_within"), dict)) + f"  | declared sign within on "
        f"{e['agreement_within']['positive']}/{e['agreement_within']['units']}, prong (matharena net of "
        f"position) {e['agreement_prong']['positive']}/{e['agreement_prong']['units']}")


def stage_signs_entropy(args):
    """--stage signs --job entropy: every column of the entropy table through
    sign_entry (the declared ones at --boots, the diagnostics at DIAG_BOOT), no
    heads -> entropy.signs."""
    t0 = time.time()
    state = H.load_json(args.out) or {}
    est = entropy_state(state)
    joined, dig = entropy_work_checked(args.work, est)
    reg = entropy_stage_registry(joined, est)
    target, _, tinfo = targets(args.work, args.refresh_targets)
    items = unit_items(joined)
    maps = oriented_maps(joined, reg)
    log(f"signs (entropy): {len(maps)} columns, targets and items in {time.time() - t0:.0f}s")
    res = {"registry": reg, "features": {}, "target_info": tinfo, "features_digest": dig,
           "prong_fields": {**{q: "spearman_within" for q in PARENTS}, **PRONG_FIELD}}
    for fi, (name, x) in enumerate(sorted(maps.items())):
        declared = 0 if reg[name].get("excluded") else reg[name]["sign"]
        boots = args.boots if declared else min(args.boots, DIAG_BOOT)
        e = res["features"][name] = sign_entry(x, target, items, boots, 100 * fi, declared)
        if reg[name].get("excluded"):
            e["excluded"] = reg[name]["excluded"]
        _log_sign(name, e, "entropy ")
    res["wall_s"] = round(time.time() - t0, 1)
    _save_part_entropy(args.out, "signs", res)
    log(f"signs (entropy): {res['wall_s']}s -> {args.out} [{ENTROPY_KIND}]")


def stage_signs(args):
    if getattr(args, "job", "main") == ENTROPY_KIND:
        return stage_signs_entropy(args)
    t0 = time.time()
    joined = load_joined(args.work)
    dig = features_digest(args.work)
    state = H.load_json(args.out) or {}
    ing = (state.get("ingest") or {}).get("features_digest")
    if ing != dig:
        log(f"warning: {os.path.join(args.work, 'features.parquet')} ({dig}) is not what --stage ingest recorded "
            f"({ing}); results follow the file")
    reg = stage_registry(joined, state)
    target, _, tinfo = targets(args.work, args.refresh_targets)
    items = unit_items(joined)
    maps = oriented_maps(joined, reg)
    rs = rubric_sum(joined, reg)
    if rs:
        maps["rubric_sum"] = rs
        reg["rubric_sum"] = {"sign": 1, "source": "composite of the oriented rubric levels", "kind": "composite",
                             "label_free": True, "usable": True}
    # attempt descriptions (k, n_distinct) are not features: no sign table; an excluded feature
    # (a mixed answer log-prob) is read as a diagnostic
    maps = {k: v for k, v in maps.items() if reg[k]["kind"] != "attempt" or reg[k]["sign"]
            or not reg[k]["label_free"]}
    log(f"signs: {len(maps)} features, targets and items in {time.time() - t0:.0f}s")
    res = {"registry": reg, "features": {}, "target_info": tinfo, "features_digest": dig,
           "prong_fields": {**{q: "spearman_within" for q in PARENTS}, **PRONG_FIELD}}
    for fi, (name, x) in enumerate(sorted(maps.items())):
        # a feature without a declared sign (or excluded) is a diagnostic: fewer bootstrap draws
        declared = 0 if reg[name].get("excluded") else reg[name]["sign"]
        boots = args.boots if declared else min(args.boots, DIAG_BOOT)
        res["features"][name] = sign_entry(x, target, items, boots, 100 * fi, declared)
        e = res["features"][name]
        if reg[name].get("excluded"):
            e["excluded"] = reg[name]["excluded"]
        _log_sign(name, e)
    heads, oof = {}, {}
    for hi, (name, (cols, prim, posfree)) in enumerate(head_specs(reg).items()):
        t1 = time.time()
        entry, preds = fit_head(joined, cols, target, items, args.boots, seed=1000 * (hi + 1), posfree=posfree)
        entry["primary"] = prim
        entry["wall_s"] = round(time.time() - t1, 1)
        heads[name] = entry
        if preds:
            oof[name] = preds
            log(f"head {name}: r " + "  ".join(f"{q[:12]} {r:+.3f}" for q, r in entry["pearson"].items())
                + f" | mean {_g(entry['mean_pearson'], 3)} (within group {_g(entry['mean_pearson_within_group'], 3)})"
                  f" | prong {entry['positive_parents_prong']}/{len(entry['pearson'])}"
                  f" | choices {[f['choice'] for f in entry['folds'].values()]}")
        else:
            log(f"head {name}: {entry.get('skipped')}")
    os.makedirs(args.work, exist_ok=True)
    H.save_json(os.path.join(args.work, "oof.json"), {"features_digest": dig, "heads": oof})
    res["wall_s"] = round(time.time() - t0, 1)
    state = H.load_json(args.out) or {}
    state["signs"] = res
    state["heads"] = heads
    H.save_json(args.out, state)
    log(f"signs: {res['wall_s']}s -> {args.out}")


# --- harness ------------------------------------------------------------------------------

def features_digest(work):
    """The digest --stage ingest records for WORK/features.parquet."""
    path = os.path.join(work, "features.parquet")
    return file_sha256(path)[:16] if os.path.exists(path) else None


def load_oof(work, digest=None):
    """The heads' out-of-fold predictions {head: {item_id: x}} from WORK/oof.json,
    only if they were fitted on the features now in WORK (their digest equals
    digest, default features_digest(work)); {} otherwise (logged), so a stale
    head never enters the harness."""
    path = os.path.join(work, "oof.json")
    c = H.load_json(path) if os.path.exists(path) else None
    if not c:
        return {}
    digest = digest or features_digest(work)
    if not isinstance(c, dict) or "heads" not in c or c.get("features_digest") != digest:
        got = c.get("features_digest") if isinstance(c, dict) else None
        log(f"warning: {path} was fitted on other features ({got}, now {digest}): heads left out; "
            "run --stage signs again")
        return {}
    return c["heads"]


def harness_maps(work, joined, reg, feats=None):
    """{name: raw x map} for the harness: every oriented declared label-free
    feature, rubric_sum and each head's out-of-fold predictions (only heads
    fitted on these features: load_oof)."""
    out = {}
    names = [c for c, v in reg.items() if v["usable"] and v["label_free"] and c in joined.columns]
    out.update(oriented_maps(joined, reg, names))
    rs = rubric_sum(joined, reg)
    if rs:
        out["rubric_sum"] = rs
    oof = load_oof(work)
    out.update({f"head {k}": v for k, v in oof.items()})
    if feats:
        out = {k: v for k, v in out.items() if k in feats}
    return out


def maps_digest(maps):
    return F.digest({k: sorted((i, round(float(x), 7)) for i, x in v.items()) for k, v in maps.items()})[:16]


def entropy_harness_maps(joined, reg, feats=None):
    """{name: raw x map} for --stage harness --job entropy: every usable oriented
    entropy feature (ENTROPY_SIGNS' columns, unless the semantics exclude them)."""
    names = [c for c, v in reg.items() if v["usable"] and v["label_free"] and c in joined.columns]
    out = oriented_maps(joined, reg, names) if names else {}       # oriented_maps reads every column given none
    return {k: v for k, v in out.items() if k in feats} if feats else out


def stage_harness(args):
    t0 = time.time()
    state = H.load_json(args.out) or {}
    if getattr(args, "job", "main") == ENTROPY_KIND:
        est = entropy_state(state)
        joined, _ = entropy_work_checked(args.work, est)
        maps = entropy_harness_maps(joined, entropy_stage_registry(joined, est), args.feats)
        res, where = est.get("harness", {}), ENTROPY_KIND

        def save(r):
            _save_part_entropy(args.out, "harness", r)
    else:
        joined = load_joined(args.work)
        reg = stage_registry(joined, state)
        maps = harness_maps(args.work, joined, reg, args.feats)
        res, where = state.get("harness", {}), "harness"

        def save(r):
            _save_part(args.out, "harness", r)
    run_harness(args, maps, res, save, t0)
    log(f"harness: {time.time() - t0:.0f}s -> {args.out} [{where}]")


def run_harness(args, maps, res, save, t0):
    """Every covariate in maps through the harness (stage_harness' recipe), kept
    where its x digest is unchanged (unless --redo); save(res) after each."""
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    items_bench = H.benchmark_items(rows, keys)
    _, honest, _ = targets(args.work)
    prov = H.provenance()
    log(f"harness: {len(maps)} covariates, rows and targets in {time.time() - t0:.0f}s")
    if res.get("_meta", {}).get("lib_digest") != prov["lib_digest"]:
        res = {}
    res["_meta"] = {"lib_digest": prov["lib_digest"], "rows_lib_digest": H.rows_digest(rows),
                    "forced": [H.cname(c) for c in FORCED_ALL], "n_placebo": args.placebo,
                    "placebo_boots": PLACEBO_BOOTS, "boots": H.BOOTS,
                    "input": "x standardised within benchmark (harness.eval_covariate), B0 term on raw x"}
    for name, raw in maps.items():
        dig = maps_digest({name: raw})
        if name in res and res[name].get("x_digest") == dig and not args.redo:
            log(f"{name}: kept")
            continue
        t1 = time.time()
        cov, info = H.eval_covariate(raw, items_bench, keys, name, standardise=True)
        cov.keys = keys
        covered = {reg_: float(np.mean(np.isin(np.array(keys, object)[R.ev_k], list(raw))))
                   for reg_, R in rows.items()}
        lines, _ = H.score_covariate(rows, cov, forced=FORCED_ALL)
        entry = {"x_digest": dig, "x": info, "coverage_eval_items": covered, **ICE.realised_r(rows, cov, honest),
                 "lines": {n: L4.compact(ln) for n, ln in lines.items()}}
        plac = {}
        for s in range(args.placebo):
            pm = ICE.permuted(raw, items_bench, s)
            cp, _ = H.eval_covariate(pm, items_bench, keys, name + " placebo", standardise=True)
            pl, _ = H.score_covariate(rows, cp, forced=FORCED_ALL, boots=PLACEBO_BOOTS, per=True)
            for n, ln in pl.items():
                plac.setdefault(n, []).append(L4.compact(ln))
        entry["placebo"] = {n: {k: float(np.mean([d[k] for d in ds])) if ds[0][k] is not None else None
                                for k in ("tl", "tl_benchmark_equal", "mix", "r1b", "r1p")}
                            | {"tl_draws": [d["tl"] for d in ds], "folds_on": [d["folds_on"] for d in ds]}
                            for n, ds in plac.items()}
        entry["wall_s"] = round(time.time() - t1, 1)
        res[name] = entry
        save(res)
        a = entry["lines"]
        log(f"{name}: {entry['wall_s']}s  r_within_pair {entry.get('r_within_pair_tl')}  " + "  ".join(
            f"{n.split()[0]} {a[n]['tl']:+.5f} (on {a[n]['folds_on']})" for n in NESTED)
            + "  forced: " + "  ".join(f"{n} {a[n]['tl']:+.5f}" for n in a if n.endswith("(forced)")))
    save(res)
    return res


def _save_part(path, part, value):
    st = H.load_json(path) or {}
    st[part] = value
    H.save_json(path, st)


def reference_covered(work, state, job="main"):
    """(covered item_ids, what): the items the reference degrades honest
    difficulty on. The rubric's job: the declared primary head's (else any
    declared feature's); the entropy job: the items with a finite
    ENTROPY_PRIMARY (degenerate units excluded), this job's coverage."""
    if job == ENTROPY_KIND:
        joined = load_entropy(work)
        v = joined[ENTROPY_PRIMARY].to_numpy(float) if ENTROPY_PRIMARY in joined.columns else np.zeros(0)
        return (set(joined.loc[np.isfinite(v), "item_id"]) if len(v) else set(),
                f"entropy job: items with a finite {ENTROPY_PRIMARY}")
    joined = load_joined(work)
    reg = stage_registry(joined, state)
    oof = load_oof(work)
    if PRIMARY_HEAD in oof:
        return set(oof[PRIMARY_HEAD]), f"head {PRIMARY_HEAD}"
    cols = rubric_levels(reg) or [c for c, v in reg.items() if v["sign"] != 0]
    covered = set(joined.loc[np.isfinite(joined[cols].to_numpy(float)).any(1), "item_id"]) if cols else set()
    return covered, "any declared feature"


def stage_reference(args):
    """The honest difficulty degraded to r on exactly the items the declared
    primary head (else any rubric level) covers, or with --job entropy the items
    with a finite ent_first1024, 0 elsewhere, N_REF draws per r."""
    t0 = time.time()
    state = H.load_json(args.out) or {}
    job = getattr(args, "job", "main")
    if job == ENTROPY_KIND:
        entropy_work_checked(args.work, entropy_state(state))
    covered, what = reference_covered(args.work, state, job)
    rows, keys = H.load_rows(args.rows, list(H.PLAN))
    target, honest, _ = targets(args.work)
    par_of_key = {k: par for par, d in target.items() for k in d}
    Z, has, _ = H.base_matrix(honest, keys, par_of_key)
    on = np.array([k in covered and k in par_of_key for k in keys])
    out = {"covered_by": what, "covered_keys_in_rows": int(on.sum()), "r": {}}
    for r in REF_R:
        runs = []
        for rep in range(N_REF):
            eps = np.random.default_rng([rep, 7919]).standard_normal(len(keys))
            X = np.where(on[None, :], H.degrade(np.where(has, Z, 0.0), r, eps[None, :]), 0.0)
            cov = H.Covariate(X, H.fold_of, f"ref r={r}", np.where(on[None, :], X, np.nan))
            runs.append(H.score_covariate(rows, cov, boots=PLACEBO_BOOTS, per=False))
        avg = H.average_lines(rows, runs, boots=PLACEBO_BOOTS)
        out["r"][f"r={r:g}"] = {
            n: {"tl": ln["regimes"]["tl"]["est"], "tl_benchmark_equal": ln["regimes"]["tl"]["parent_mean"],
                "tl_draws": ln["replicates"]["tl"], "mix": ln["regimes"]["mix"]["est"], "folds_on": ln.get("folds_on")}
            for n, ln in avg.items()}
        log(f"reference r={r:g}: " + "  ".join(f"{n} {v['tl']:+.5f}" for n, v in out["r"][f"r={r:g}"].items()))
    out["wall_s"] = round(time.time() - t0, 1)
    if job == ENTROPY_KIND:
        out["covered_items"] = len(covered)
        _save_part_entropy(args.out, "reference", out)
    else:
        _save_part(args.out, "reference", out)


# --- attempts -----------------------------------------------------------------------------

def attempt_call(rep, primary=None):
    """The attempt probe's rule (attempt_probe.verdict) over designs: FLOOR if the
    best design's graded accuracy is under FLOOR_ACC; GO if the best label-free
    feature reaches rho >= GO_RHO with CI lower bound > GO_LO and rho >= GO_2026
    on the 2026 contests; KILL if every feature is below KILL_RHO; WEAK
    otherwise. rep: {design: {accuracy, features_vs_honest: {f: within_rho},
    by_year_honest: {f: {'2025': .., '2026': ..}}}}; the resolved attempt
    primary's rows are reported beside the call (primary)."""
    accs = [v["accuracy"] for v in rep.values() if v.get("accuracy") is not None]
    acc = max(accs) if accs else None
    best = None
    for d, v in rep.items():
        for f, row in v["features_vs_honest"].items():
            if row and (best is None or row["rho"] > best[2]["rho"]):
                best = (d, f, row)
    if best is None:
        return {"call": "NO DATA", "best_graded_accuracy": acc}
    y26 = (rep[best[0]].get("by_year_honest", {}).get(best[1]) or {}).get("2026")
    all_below = all(row is None or row["rho"] < KILL_RHO for v in rep.values()
                    for row in v["features_vs_honest"].values())
    lo = best[2]["ci"][0] if best[2].get("ci") else None
    if acc is not None and acc < FLOOR_ACC:
        call = "FLOOR"
    elif best[2]["rho"] >= GO_RHO and lo is not None and lo > GO_LO and y26 is not None and y26["rho"] >= GO_2026:
        call = "GO"
    elif all_below:
        call = "KILL"
    else:
        call = "WEAK (between KILL and GO)"
    out = {"call": call, "best_design": best[0], "best_feature": best[1], "best": best[2], "best_2026": y26,
           "best_graded_accuracy": None if acc is None else round(acc, 3),
           "rule": (f"FLOOR if graded accuracy < {FLOOR_ACC}; GO if rho >= {GO_RHO}, CI lower bound > {GO_LO} "
                    f"and rho >= {GO_2026} on the 2026 contests; KILL if every feature < {KILL_RHO}")}
    if primary:
        out["primary"] = {"feature": primary, "designs": {
            d: {"vs_honest": v["features_vs_honest"].get(primary),
                "2026": (v.get("by_year_honest", {}).get(primary) or {}).get("2026")}
            for d, v in rep.items() if primary in v["features_vs_honest"]}}
    return out


def attempt_report(joined, target, items, boots=AP.BOOTS, strong=None, seed=0, exclude=None, primary=None):
    """Per attempt design on matharena: accuracy, rates, and every label-free
    feature (oriented + = harder) against honest b within competition, by
    contest year, against strong-tier b where given, and net of log length.
    Two statistics: features_vs_honest, by_year_honest and features_vs_strong
    are the attempt rule's (attempt_probe.within_rho: ranks within each
    competition); partial_on_log_length is the sign stage's
    (llm4b_close.corr_block's partial_spearman: ranks over the benchmark,
    demeaned within competition, net of log length), and
    spearman_within_corr_block its unadjusted counterpart (corr_block's
    spearman_within), so that the length adjustment is read in one statistic.
    exclude: {feature: reason} read apart (excluded, never a candidate of the
    call); primary: the resolved attempt primary (its relation to prompt length
    is reported)."""
    import pandas as pd
    exclude = exclude or {}
    mh = joined[joined["benchmark"] == "matharena"]
    tgt = target.get("matharena", {})
    it = items.get("matharena", {})
    rep = {}
    for d, cols in sorted(attempt_designs(joined.columns).items()):
        present = cols.get("k") or cols.get("tok_entropy") or next(iter(cols.values()))
        sub = mh[np.isfinite(mh[present].to_numpy(float))]
        sub = sub[sub["item_id"].isin(list(tgt)) & sub["item_id"].isin(list(it))]
        if len(sub) < ATT_MIN_ITEMS:
            rep[d] = {"n_items": int(len(sub)), "features_vs_honest": {},
                      "skipped": f"fewer than {ATT_MIN_ITEMS} items"}
            continue
        ids = sub["item_id"].tolist()
        y = pd.Series([tgt[i] for i in ids], index=ids)
        comp = pd.Series([IC.features(it[i]).get("competition", "") for i in ids], index=ids)
        year = pd.Series(np.where(comp.isin(YEAR_2026), "2026", "2025"), index=ids)
        col = lambda f: pd.Series(sub[cols[f]].to_numpy(float), index=ids)  # noqa: E731
        rng = np.random.default_rng(seed)
        v = {"n_items": int(len(sub)), "competitions": int(comp.nunique()), "k": float(np.nanmean(col("k")))
             if "k" in cols else None, "features_vs_honest": {}, "by_year_honest": {}, "features_vs_strong": {},
             "partial_on_log_length": {}, "spearman_within_corr_block": {}, "excluded": {}}
        if "graded" in cols:
            gr = col("graded")
            v["accuracy"] = None if gr.notna().sum() == 0 else round(float(gr.mean()), 4)
            v["accuracy_items_with_gold"] = int(gr.notna().sum())
            v["accuracy_by_year"] = {yy: round(float(gr[year == yy].mean()), 4) for yy in ("2025", "2026")
                                     if gr[year == yy].notna().any()}
            v["graded_vs_honest (diagnostic, needs the reference answer)"] = AP.within_rho(
                -gr, y, comp, rng, boots)
        if "top_correct" in cols:
            v["majority_accuracy"] = round(float(col("top_correct").mean()), 4)
        for f in ("fail_rate", "trunc_rate", "forced_rate", "closed_rate", "mean_len", "mean_think_len"):
            if f in cols:
                v[f] = round(float(col(f).mean()), 4)
        L = np.array([IC.log_length(it[i]) for i in ids])
        for f, sgn in ATT_SIGNS.items():
            if f not in cols:
                continue
            x = sgn * col(f)
            if x.nunique(dropna=True) < 2:
                continue
            if f in exclude:
                v["excluded"][f] = {"reason": exclude[f], "vs_honest": AP.within_rho(x, y, comp, rng, boots)}
                continue
            v["features_vs_honest"][f] = AP.within_rho(x, y, comp, rng, boots)
            v["by_year_honest"][f] = {yy: AP.within_rho(x[year == yy], y[year == yy], comp[year == yy], rng, boots)
                                      for yy in ("2025", "2026")}
            if strong is not None:
                s = pd.Series([strong.get(i, np.nan) for i in ids], index=ids)
                if s.notna().sum() >= 8:
                    v["features_vs_strong"][f] = AP.within_rho(x[s.notna()], s[s.notna()], comp[s.notna()], rng,
                                                               boots)
            ok = np.isfinite(x.to_numpy())
            if ok.sum() >= MIN_ITEMS:
                cb = L4.corr_block(x.to_numpy()[ok], y.to_numpy()[ok], comp.to_numpy()[ok], L[ok],
                                   min(boots, 500), seed=seed + 5)
                v["partial_on_log_length"][f] = cb.get("partial_spearman")
                v["spearman_within_corr_block"][f] = cb.get("spearman_within")
        plen = pd.Series([len(it[i]["item_content"]) for i in ids], index=ids, dtype=float)
        v["prompt_chars_vs_honest"] = AP.within_rho(plen, y, comp, rng, boots)
        for f in dict.fromkeys(("tok_entropy", primary)):
            if f and f in cols:
                v[f"{f}_vs_prompt_chars"] = AP.within_rho(col(f), plen, comp, rng, boots)
        rep[d] = v
    return rep


def reference_4b(sem):
    """The 4B attempt probe's decision (results/attempt_probe.json), set beside
    this export's attempt primary only when that is the same kind of quantity
    (attempt_semantics' d2): comparable with a raw full-vocabulary primary;
    flagged (not comparable, shown) when the primary is another quantity of the
    raw distribution (a fixed window, a top-5 bound); withheld when the
    log-probs are processed, mixed or undeclared."""
    sem = sem or {}
    d2 = sem.get("d2") or {"comparable": False, "reason": "no attempt semantics"}
    out = {"comparable": bool(d2["comparable"]), "reason": d2["reason"], "primary": sem.get("primary_attempt"),
           "logprobs": sem.get("logprobs")}
    if d2["comparable"] or sem.get("logprobs") in ("raw", "raw_topk"):
        out["decision"] = _ref_4b()
    else:
        out["decision"] = None
        out["withheld"] = ("the 4B probe's decision is not set beside a primary read off a processed (or "
                           "undeclared) distribution")
    return out


def strong_probe_ids(path=None):
    """The notebook's PROBE_IDS (the attempt probe's 160 matharena item_ids), read
    from strong_probe.py's source with ast, never imported: a frozenset of str,
    or None when the file or the assignment is missing."""
    import ast
    path = path or os.path.join(ROOT, STRONG_PROBE)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROBE_IDS"
                                                 for t in node.targets):
            try:
                txt = next(n.value for n in ast.walk(node.value)
                           if isinstance(n, ast.Constant) and isinstance(n.value, str))
            except StopIteration:
                return None
            return frozenset(txt.split())
    return None


def probe_units(kdir, probe_ids=None):
    """The notebook's probe flag per attempt text (UNITS_DETAIL under kdir: one row
    per unique text, its item_ids (or item_id) and `probe`): {'path', 'sha256'
    (of the table: the export's manifest does not hash _detail/), 'probe':
    {(benchmark, item_id)}, 'rest': {...}, 'units': {'probe': n, 'rest': n},
    'items': {...}, 'conflicting_items': n, 'probe_ids_check'}. An item_id under
    both a probe and a non-probe text (the notebook's units never share one) is in
    neither set. probe_ids (strong_probe.py's PROBE_IDS, as strong_probe_ids reads
    them): the notebook flags a text iff one of its item_ids is listed, so every
    unit's flag is checked against that ('probe_ids_check': units, mismatches,
    listed ids in flagged / unflagged / no attempted text, ok); None leaves it
    unchecked. None when the table or its flag is missing."""
    import pandas as pd
    path = os.path.join(kdir, UNITS_DETAIL)
    if not os.path.exists(path):
        return None
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    if "probe" not in have or "benchmark" not in have or not have & {"item_ids", "item_id"}:
        return None
    cols = [c for c in ("unit", "benchmark", "item_ids", "item_id", "probe") if c in have]
    df = _norm(pd.read_parquet(path, columns=cols))
    df = df[df["item_id"].notna() & df["benchmark"].notna()]
    flag = df["probe"].map(_flag).to_numpy(bool)
    keys = list(zip(df["benchmark"], df["item_id"]))
    probe = {k for k, f in zip(keys, flag) if f}
    rest = {k for k, f in zip(keys, flag) if not f}
    both = probe & rest
    units = {"probe": None, "rest": None}
    if "unit" in df.columns:
        u = df.drop_duplicates("unit")
        units = {"probe": int(u["probe"].map(_flag).sum()), "rest": int((~u["probe"].map(_flag)).sum())}
    check = None
    if probe_ids is not None:
        listed = pd.Series([str(i) in probe_ids for i in df["item_id"]], index=df.index)
        grp = df["unit"] if "unit" in df.columns else pd.Series(range(len(df)), index=df.index)
        per = pd.DataFrame({"u": grp.to_numpy(), "flag": flag, "listed": listed.to_numpy()}).groupby("u").agg(
            flag=("flag", "max"), listed=("listed", "max"))
        seen_p = {str(i) for i, f in zip(df["item_id"], flag) if f}
        seen_r = {str(i) for i, f in zip(df["item_id"], flag) if not f}
        mism = int((per["flag"] != per["listed"]).sum())
        check = {"what": ("the notebook's flag against strong_probe.py PROBE_IDS: a text is a probe text iff one of "
                          "its item_ids is listed" + ("" if "unit" in df.columns else " (no unit column: per row)")),
                 "probe_ids": len(probe_ids), "units_checked": int(len(per)), "units_flag_mismatch": mism,
                 "listed_in_flagged_units": len(probe_ids & seen_p),
                 "listed_in_unflagged_units": len(probe_ids & seen_r),
                 "listed_not_attempted": len(probe_ids - seen_p - seen_r), "ok": mism == 0}
    return {"path": UNITS_DETAIL, "sha256": file_sha256(path), "probe": probe - both, "rest": rest - both,
            "units": units, "items": {"probe": len(probe - both), "rest": len(rest - both)},
            "conflicting_items": len(both), "probe_ids_check": check}


def unit_rows(joined, keys):
    """The rows of joined whose (benchmark, item_id) is in keys."""
    m = np.fromiter(((b, i) in keys for b, i in zip(joined["benchmark"], joined["item_id"])), bool, len(joined))
    return joined[m]


def stage_attempts(args):
    import pandas as pd
    t0 = time.time()
    units = getattr(args, "attempt_units", "both")
    joined = load_joined(args.work)
    if not attempt_designs(joined.columns):
        log("attempts: no attempt features in the Kaggle outputs")
        _save_part(args.out, "attempts", {"call": "NO DATA"})
        return
    state = H.load_json(args.out) or {}
    sem = stage_semantics(state, joined)
    reg = stage_registry(joined, state)
    exclude = {v["feature"]: v["excluded"] for v in reg.values() if v["kind"] == "attempt" and v.get("excluded")}
    primary = sem.get("primary_attempt")
    kdir = attempt_dir(kaggle_dirs(args))
    pu = probe_units(kdir, strong_probe_ids()) if units in ("probe", "both") else None
    if units == "probe" and pu is None:
        raise SystemExit(f"attempts: no probe flag ({os.path.join(kdir, UNITS_DETAIL)} is missing or lacks "
                         "benchmark, item_ids and probe); --attempt-units all reads every unit")
    if pu is not None and pu["probe_ids_check"] is not None and not pu["probe_ids_check"]["ok"]:
        raise SystemExit(f"attempts: the probe flag in {os.path.join(kdir, UNITS_DETAIL)} is not membership "
                         f"in {STRONG_PROBE} PROBE_IDS ({json.dumps(pu['probe_ids_check'])}); the probe reading "
                         "would be on other texts than the rule was fixed for")
    target, _, _ = targets(args.work)
    items = {"matharena": ICE.load_items("matharena")}
    strong = None
    tp = os.path.join(ROOT, "data", "attempt_probe", "targets.parquet")
    if os.path.exists(tp):
        t = pd.read_parquet(tp)
        if "b_strong" in t.columns:
            strong = {str(k): float(v) for k, v in t["b_strong"].items() if np.isfinite(v)}
    old = state.get("attempts") or {}
    dig = features_digest(args.work)
    if units == "probe":
        out = {k: v for k, v in old.items() if k != "probe_only"}
    else:
        rep = attempt_report(joined, target, items, args.boots, strong, exclude=exclude, primary=primary)
        out = {"designs": rep,
               "decision": attempt_call({d: v for d, v in rep.items() if v.get("features_vs_honest")}, primary),
               "semantics": {k: sem.get(k) for k in ("logprobs", "logprobs_manifest", "presence_penalty",
                                                     "lp_answer", "raw_features", "primary_attempt", "primary_order",
                                                     "notes")},
               "strong_tier_items": 0 if strong is None else len(strong),
               "reference_4b": reference_4b(sem), "features_digest": dig,
               "script_digest": H.digest(["experiments/strong_llm_eval.py"]), "wall_s": round(time.time() - t0, 1)}
        log(f"attempts, all units: {json.dumps(out['decision'])}")
        log(f"attempts: primary {primary}; beside the 4B D2 lead: {out['reference_4b']['comparable']} "
            f"({out['reference_4b']['reason']})")
    if pu is not None:
        t1 = time.time()
        sub = unit_rows(joined, pu["probe"])
        rep = attempt_report(sub, target, items, args.boots, strong, exclude=exclude, primary=primary)
        out["probe_only"] = {
            "units": "probe",
            "what": ("the attempt rule on the probe texts only, the units it was fixed for before any output "
                     "existed (the notebook's probe flag, strong_probe.py PROBE_IDS: run first); the all-units "
                     "fields beside it read every attempted text"),
            "source": pu["path"], "source_sha256": pu["sha256"], "probe_ids_check": pu["probe_ids_check"],
            "flagged_units": pu["units"], "flagged_items": pu["items"],
            "conflicting_items": pu["conflicting_items"],
            "probe_digest": F.digest(sorted(f"{b}/{i}" for b, i in pu["probe"]))[:16],
            "designs": rep,
            "decision": attempt_call({d: v for d, v in rep.items() if v.get("features_vs_honest")}, primary),
            "features_digest": dig, "script_digest": H.digest(["experiments/strong_llm_eval.py"]),
            "wall_s": round(time.time() - t1, 1)}
        log(f"attempts, probe texts only ({pu['units']['probe']} texts, {pu['items']['probe']} item_ids): "
            f"{json.dumps(out['probe_only']['decision'])}")
    else:                                                   # 'all', or 'both' without a probe flag
        if units == "both":
            log(f"attempts: no probe flag in {os.path.join(kdir, UNITS_DETAIL)}; probe-only reading skipped")
        prev = old.get("probe_only")
        if prev is not None and prev.get("features_digest") == dig:
            out["probe_only"] = prev                        # a probe reading of these features is kept
        elif prev is not None:
            log("attempts: the stored probe-only reading was on other features: dropped")
    _save_part(args.out, "attempts", out)


# --- run facts ----------------------------------------------------------------------------

def _utc(t):
    return None if t is None or not math.isfinite(float(t)) else time.strftime("%Y-%m-%d %H:%M:%S",
                                                                               time.gmtime(float(t)))


def find_root_manifest(export_manifest, raw=KAGGLE_RAW):
    """The notebook's root manifest (<copy>/strong_probe/manifest.json of a Kaggle
    Output copied under raw) of the session that wrote this export: its
    script_sha256 is the export's notebook_digest and its wall_s_total the
    export's wall_s. None unless exactly one matches."""
    man = export_manifest or {}
    hits = []
    for p in sorted(glob.glob(os.path.join(raw, "**", "strong_probe", "manifest.json"), recursive=True)):
        try:
            with open(p) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if m.get("script_sha256") == man.get("notebook_digest") and m.get("wall_s_total") == man.get("wall_s"):
            hits.append(p)
    return hits[0] if len(hits) == 1 else None


#: what session_logs looks for in a saved log: the first engine start's failure and the kit's retry
LOG_MARKERS = {"prefix_prefill": "prefix_prefill", "f16_conversion": "Unsupported conversion from f16 to f16",
               "no_prefix_caching": "--no-prefix-caching"}


def session_logs(root_manifest):
    """The session's saved logs: every *.log under the Output copy that holds
    root_manifest (<copy>/strong_probe/manifest.json; Kaggle's log is not part
    of the Output, so it is there only if someone saved it), each as {path,
    bytes, sha256, mentions: {marker: bool} for LOG_MARKERS}. [] when none is."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(root_manifest)))
    out = []
    for p in sorted(glob.glob(os.path.join(base, "**", "*.log"), recursive=True)):
        with open(p, "rb") as fh:
            data = fh.read()
        txt = data.decode("utf-8", "replace")
        out.append({"path": os.path.relpath(p, ROOT) if p.startswith(ROOT) else p, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "mentions": {k: m in txt for k, m in LOG_MARKERS.items()}})
    return out


def run_facts(kdir, root_manifest=None):
    """What the Kaggle session did and cost, from files only: the export's
    manifest and _summary.json; the write times, prompt tokens and generated
    tokens of its _detail tables (rubric rows are stamped one by one as they are
    written, an attempt shard's rows share one stamp, so consecutive stamps
    time a shard); and, when given, the notebook's root manifest (sessions'
    start, wall time, status and measured rates, the plan's estimates, the
    progress) and any session log saved beside it (session_logs). Derived: the
    rubric's hours against the plan's, the attempt shards' duration, the
    generated tokens a second over the attempt phase (from the last rubric
    write to the last shard), and the hours the remaining attempt units would
    take at the median shard duration."""
    import pandas as pd
    import pyarrow.parquet as pq
    with open(os.path.join(kdir, "manifest.json")) as fh:
        man = json.load(fh)
    out = {"export": {k: man.get(k) for k in ("created", "wall_s", "sessions", "gpu", "notebook_digest", "slug")}}
    sp = os.path.join(kdir, "_summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            out["export"]["summary"] = json.load(fh)
    tl = {}

    def cols(path, want):
        have = set(pq.read_schema(path).names)
        return pd.read_parquet(path, columns=[c for c in want if c in have])

    rp = os.path.join(kdir, "_detail", "rubric_units.parquet")
    r_last = None
    if os.path.exists(rp):
        r = cols(rp, ("t", "prompt_tokens", "gen_tokens", "truncated"))
        if "t" in r.columns and len(r):
            r_last = float(r["t"].max())
            tl.update(rubric_units=int(len(r)), rubric_first_write_utc=_utc(r["t"].min()),
                      rubric_last_write_utc=_utc(r_last))
        if "prompt_tokens" in r.columns:
            tl["rubric_prompt_tokens"] = int(r["prompt_tokens"].sum())
        if "gen_tokens" in r.columns:
            tl["rubric_gen_tokens"] = int(r["gen_tokens"].sum())
    up = os.path.join(kdir, UNITS_DETAIL)
    ts = None
    if os.path.exists(up):
        u = cols(up, ("unit", "t", "probe", "item_ids"))
        if "probe" in u.columns:
            f = u["probe"].map(_flag)
            tl["attempt_units"] = {"probe": int(f.sum()), "rest": int((~f).sum())}
            if "item_ids" in u.columns:
                n = u["item_ids"].map(len)
                tl["attempt_item_ids"] = {"probe": int(n[f].sum()), "rest": int(n[~f].sum())}
        if "t" in u.columns and len(u):
            ts = np.sort(u["t"].unique().astype(float))
            d = np.diff(ts)
            tl.update(attempt_shards=int(len(ts)), units_per_shard=round(len(u) / len(ts), 2),
                      attempt_first_write_utc=_utc(ts[0]), attempt_last_write_utc=_utc(ts[-1]))
            if r_last is not None:
                tl["first_shard_after_rubric_s"] = round(float(ts[0] - r_last), 1)
                if "probe" in u.columns and u["probe"].map(_flag).any():
                    # the probe texts run first: the shard holding the last of them, after the rubric
                    tl["probe_done_after_rubric_h"] = round(
                        float(u.loc[u["probe"].map(_flag), "t"].max() - r_last) / 3600, 3)
            if len(d):
                tl["shard_s"] = {"median": round(float(np.median(d)), 1), "mean": round(float(d.mean()), 1),
                                 "min": round(float(d.min()), 1), "max": round(float(d.max()), 1),
                                 "n": int(len(d)), "what": "between consecutive shard writes"}
    ap = os.path.join(kdir, "_detail", "attempt_samples.parquet")
    if os.path.exists(ap):
        s = cols(ap, ("n_tokens", "capped", "closed", "forced"))
        if "n_tokens" in s.columns:
            tl["attempt_samples"] = int(len(s))
            tl["attempt_gen_tokens"] = int(s["n_tokens"].sum())
            if ts is not None and r_last is not None and ts[-1] > r_last:
                tl["attempt_gen_tok_s"] = round(tl["attempt_gen_tokens"] / float(ts[-1] - r_last), 1)
        for c in ("capped", "closed", "forced"):
            if c in s.columns:
                tl[f"attempt_{c}_share"] = round(float(s[c].map(_flag).mean()), 4)
    ep = os.path.join(kdir, "_detail", "entropy_units.parquet")
    if os.path.exists(ep):
        tl.update(entropy_timeline(cols(ep, ("t", "benchmark", "item_ids", "n_tokens", "prompt_tokens", "capped",
                                             "closed", "degenerate", "truncated"))))
    out["timeline"] = tl
    if not root_manifest:
        out["root_manifest"] = None
        return out
    with open(root_manifest) as fh:
        root = json.load(fh)
    slug = man.get("slug")
    out["root_manifest"] = os.path.relpath(root_manifest, ROOT) if root_manifest.startswith(ROOT) else root_manifest
    out["logs"] = session_logs(root_manifest)
    out["notebook"] = {k: root.get(k) for k in ("version", "vllm_pin", "transformers_pin", "wall_s_total")}
    eng = (root.get("models") or {}).get(slug) or {}
    out["notebook"]["engine"] = {k: eng.get(k) for k in ("model", "revision", "engine", "torch", "tp", "gpus",
                                                         "gpu_mem", "max_model_len", "k", "max_tokens",
                                                         "attempt_logprobs", "raw_recorder")}
    sess = []
    for se in root.get("sessions") or []:
        st = se.get("stats") or {}
        sess.append({"run": se.get("run"), "t0_utc": _utc(se.get("t0")), "wall_s": se.get("wall_s"),
                     "status": se.get("status"),
                     "rubric": {k: (st.get("rubric") or {}).get(k) for k in ("secs", "prompt_tok_s",
                                                                             "units_done_session")},
                     "attempts": {k: (st.get("attempts") or {}).get(k) for k in (
                         "secs", "gen_tok_s", "units_done_session", "graded_units", "recorder_check",
                         "stats_source")},
                     "files_at_end": (se.get("files") or {}).get("at_end")})
        if st.get(ENTROPY_KIND):
            sess[-1][ENTROPY_KIND] = {k: st[ENTROPY_KIND].get(k) for k in (
                "secs", "gen_tok_s", "units_done_session", "shards", "gen_tokens", "prompt_tokens", "stats_source",
                "recorder_check", "self_check")}
    out["sessions"] = sess
    plan = (root.get("plan") or {}).get(slug) or {}
    pr, pa = plan.get("rubric") or {}, plan.get("attempts") or {}
    out["plan"] = {"rates_tok_s": plan.get("rates_tok_s"), "slow_case": plan.get("slow_case"),
                   "rubric": {k: pr.get(k) for k in ("units", "prompt_tokens", "prompt_tokens_uncached",
                                                     "shared_prefix_tokens", "hours_est", "hours_est_slow")},
                   "attempts": {k: pa.get(k) for k in ("units", "probe_units", "rest_units", "item_ids",
                                                       "gen_tokens_est", "forced_cache_hit", "hours_est",
                                                       "hours_est_slow", "probe_hours_est", "probe_hours_est_slow",
                                                       "rest_hours_est", "rest_hours_est_slow")}}
    pe = plan.get(ENTROPY_KIND) or {}
    if pe:
        out["plan"][ENTROPY_KIND] = {k: pe.get(k) for k in (
            "units", "item_ids", "per_benchmark", "prompt_tokens", "gen_tokens_est", "shards_est", "rates_tok_s",
            "hours_est", "hours_est_slow", "commit_usable_hours", "fits_one_commit", "fits_one_commit_slow",
            "units_per_commit_slow")}
    prog = ((root.get("progress") or {}).get(slug) or {})
    out["progress"] = prog
    der = {}
    s0 = sess[0] if len(sess) == 1 else None
    if s0:
        rs = s0["rubric"].get("secs")
        der["session_hours"] = round(s0["wall_s"] / 3600, 2) if s0.get("wall_s") else None
        if rs:
            der["rubric_hours"] = round(rs / 3600, 3)
            if pr.get("hours_est"):
                der["rubric_hours_over_plan"] = round(rs / 3600 / pr["hours_est"], 2)
            if r_last is not None and root["sessions"][0].get("t0"):
                der["before_rubric_s"] = round(r_last - rs - float(root["sessions"][0]["t0"]), 1)
                der["before_rubric_what"] = ("from the notebook's first cell to the rubric's start (its last write "
                                             "less its measured seconds): installs, downloads, the plan, engine "
                                             "starts")
            if tl.get("rubric_prompt_tokens"):
                der["rubric_prompt_tok_s"] = round(tl["rubric_prompt_tokens"] / rs, 1)
        if s0["attempts"].get("secs"):
            der["attempt_hours"] = round(s0["attempts"]["secs"] / 3600, 3)
    pa_prog = prog.get("attempts") or {}
    if pa_prog.get("units") is not None and pa_prog.get("done") is not None:
        left = int(pa_prog["units"]) - int(pa_prog["done"])
        der["attempt_units_left"] = left
        if tl.get("shard_s") and tl.get("units_per_shard"):
            der["attempt_hours_left_at_median_shard"] = round(
                left / tl["units_per_shard"] * tl["shard_s"]["median"] / 3600, 2)
    es = [se[ENTROPY_KIND] for se in sess if se.get(ENTROPY_KIND)]
    pe_prog = prog.get(ENTROPY_KIND) or {}
    if es or pe_prog:
        der.update(entropy_derived(es, pe, pe_prog))
    out["derived"] = der
    return out


def entropy_timeline(e):
    """run_facts' entropy entries from _detail/entropy_units.parquet (one row per
    unit; a shard's rows share their write time t)."""
    tl = {"entropy_units": int(len(e))}
    if "benchmark" in e.columns:
        tl["entropy_units_per_benchmark"] = {str(b): int(n) for b, n in e["benchmark"].value_counts().items()}
    if "item_ids" in e.columns:
        tl["entropy_item_ids"] = int(e["item_ids"].map(len).sum())
    if "t" in e.columns and len(e):
        ts = np.sort(e["t"].unique().astype(float))
        d = np.diff(ts)
        tl.update(entropy_shards=int(len(ts)), entropy_first_write_utc=_utc(ts[0]),
                  entropy_last_write_utc=_utc(ts[-1]), entropy_units_per_shard=round(len(e) / len(ts), 2))
        if len(d):
            tl["entropy_shard_s"] = {"median": round(float(np.median(d)), 1), "mean": round(float(d.mean()), 1),
                                     "min": round(float(d.min()), 1), "max": round(float(d.max()), 1),
                                     "n": int(len(d)), "what": "between consecutive shard writes"}
    for c, name in (("n_tokens", "entropy_gen_tokens"), ("prompt_tokens", "entropy_prompt_tokens")):
        if c in e.columns:
            tl[name] = int(e[c].sum())
    for c in ("capped", "closed", "degenerate", "truncated"):
        if c in e.columns and len(e):
            tl[f"entropy_{c}_share"] = round(float(e[c].map(_flag).mean()), 4)
    return tl


def entropy_derived(sessions, plan, progress):
    """The entropy job's measured rate against the plan's assumption, and what is
    left: sessions [{secs, gen_tokens, units_done_session, ...}] from the root
    manifest, plan its plan[slug].entropy, progress its progress[slug].entropy."""
    der = {}
    secs = sum(float(s.get("secs") or 0) for s in sessions)
    gen = sum(int(s.get("gen_tokens") or 0) for s in sessions)
    done = sum(int(s.get("units_done_session") or 0) for s in sessions)
    if secs > 0:
        der["entropy_hours"] = round(secs / 3600, 3)
        der["entropy_gen_tok_s"] = round(gen / secs, 1)
        der["entropy_gen_tok_s_what"] = ("generated tokens over the shards' measured seconds, prefill and checks "
                                         "included")
        rt = (plan or {}).get("rates_tok_s") or {}
        for k in ("decode", "decode_slow"):
            if rt.get(k):
                der[f"entropy_gen_tok_s_over_plan_{k}"] = round(gen / secs / float(rt[k]), 3)
        if (plan or {}).get("hours_est") and (plan or {}).get("units") and done:
            der["entropy_hours_per_unit_over_plan"] = round(secs / done / (plan["hours_est"] * 3600 / plan["units"]),
                                                            3)
    if progress.get("units") is not None and progress.get("done") is not None:
        left = int(progress["units"]) - int(progress["done"])
        der["entropy_units_left"] = left
        if secs > 0 and done:
            der["entropy_hours_left_at_measured"] = round(left * secs / done / 3600, 2)
    return der


def stage_run(args):
    """run_facts for each export directory: into `run` for the first commit's kind of
    export, into entropy.run for one whose manifest carries kinds.entropy (so `run`
    keeps the session it describes)."""
    dirs = kaggle_dirs(args)
    if args.run_manifest and len(dirs) > 1:
        raise SystemExit("--run-manifest reads one export directory's session: pass one --kaggle directory")
    for kdir in dirs:
        mp = os.path.join(kdir, "manifest.json")
        if not os.path.exists(mp):
            raise SystemExit(f"run: no export manifest at {mp}")
        with open(mp) as fh:
            man = json.load(fh)
        root = args.run_manifest or find_root_manifest(man)
        if args.run_manifest is None and root is None:
            log(f"run: no root manifest under {KAGGLE_RAW} matches the export {kdir} (notebook_digest, wall_s); "
                "--run-manifest PATH for the sessions and the plan")
        out = run_facts(kdir, root)
        if isinstance((man.get("kinds") or {}).get(ENTROPY_KIND), dict):
            _save_part_entropy(args.out, "run", out)
            log(f"run: {kdir} carries the entropy job -> {args.out} [{ENTROPY_KIND}.run]")
        else:
            _save_part(args.out, "run", out)
        print(json.dumps(out, indent=1))


def _ref_4b():
    """The 4B attempt probe's decision, for comparison (results/attempt_probe.json)."""
    r = H.load_json(os.path.join(ROOT, "results", "attempt_probe.json")) or {}
    return r.get("decision")


# --- verdict ------------------------------------------------------------------------------

def gate_checks(ln):
    """harness.gate on a compact line (llm4b_close.compact)."""
    tl, mix = ln.get("tl"), ln.get("mix")
    on = ln.get("folds_on")
    g = H.GATE
    checks = {"selection_on": None if on is None else bool(on >= g["folds_on"]),
              "tl": tl is not None and tl <= g["tl"],
              "mix_same_sign": mix is not None and tl is not None and tl != 0 and np.sign(mix) == np.sign(tl),
              "no_parent_worse": ln.get("tl_worst_parent") is not None and ln["tl_worst_parent"] <= g["parent"],
              "guard": all(ln.get(k) is None or ln[k] <= g["guard"] for k in ("r1b", "r1p"))}
    rest = all(bool(v) for k, v in checks.items() if k != "selection_on")
    checks["pass_if_on"] = rest
    checks["pass"] = None if on is None else bool(rest and checks["selection_on"])
    return checks


def nested_gate(lines):
    """({nested line: gate_checks with tl and folds_on}, whether any passes)."""
    nested = {n: {**gate_checks(lines[n]), "tl": lines[n]["tl"], "folds_on": lines[n]["folds_on"]}
              for n in NESTED if n in lines}
    return nested, any(v["pass"] for v in nested.values())


def best_nested(nested):
    return min((v["tl"] for v in nested.values() if v["tl"] is not None), default=None)


def forced_per_pair(lines):
    return {n: lines[n]["tl"] for n in lines if n.startswith("per-pair") and n.endswith("(forced)")}


def verdict(state):
    """The harness gate on every nested line, the step-8 sign prong, and the call."""
    hz = {k: v for k, v in (state.get("harness") or {}).items() if not k.startswith("_")}
    sg = (state.get("signs") or {}).get("features", {})
    heads = state.get("heads") or {}
    reg = (state.get("signs") or {}).get("registry", {})
    per = {}
    for name, e in hz.items():
        lines = e["lines"]
        nested, gate_pass = nested_gate(lines)
        if name.startswith("head "):
            h = heads.get(name[5:], {})
            # the prong: positive out-of-fold r, on matharena net of position (HEAD_PRONG_FIELD)
            sign_ok = (h.get("positive_parents_prong") or 0) >= KEEP_SIGNS
            primary = name[5:] == PRIMARY_HEAD
            sign_info = {"positive_parents_prong": h.get("positive_parents_prong"),
                         "positive_parents": h.get("positive_parents"), "mean_pearson": h.get("mean_pearson")}
        else:
            # the prong: within group, on matharena net of position (PRONG_FIELD)
            a = (sg.get(name) or {}).get("agreement_prong") or {}
            sign_ok = bool(a.get("declared_sign_ok"))
            r = reg.get(name, {})
            primary = bool(r.get("kind") == "attempt" and r.get("primary"))
            sign_info = {"positive_parents_prong": a.get("positive"), "units": a.get("units"),
                         "positive_parents_within": ((sg.get(name) or {}).get("agreement_within") or {}).get(
                             "positive")}
        best, forced_pp = best_nested(nested), forced_per_pair(lines)
        per[name] = {"primary": primary, "gate_pass_nested": gate_pass, "sign_prong": sign_ok, **sign_info,
                     "best_nested_tl": best, "nested": nested, "forced_per_pair_tl": forced_pp,
                     "keep": bool(primary and gate_pass and sign_ok)}
    keep = sorted(n for n, v in per.items() if v["keep"])
    exploratory = sorted(n for n, v in per.items() if not v["primary"] and v["gate_pass_nested"] and v["sign_prong"])
    att = (state.get("attempts") or {}).get("decision")
    att_probe = ((state.get("attempts") or {}).get("probe_only") or {}).get("decision")
    sem = (state.get("ingest") or {}).get("attempt_semantics") or {}
    return {"rule": (f"keep a declared primary ({PRIMARY_HEAD}; the attempt primary, the first of "
                     f"{list(PRIMARY_ATTEMPT_ORDER)} the export carries, of every attempt design) only if a nested "
                     f"harness line passes the gate {H.GATE} (nested, acting in >= 3 of 4 folds, mix/whole of the "
                     f"same sign, public R1 guard) and its declared sign holds on >= {KEEP_SIGNS} of 4 parents "
                     f"(plan step 8; within group, and on matharena net of position: {PRONG_FIELD} for a feature, "
                     f"{HEAD_PRONG_FIELD} for a head's out-of-fold r); a covariate on one parent (the attempts) "
                     f"cannot pass the nested gate, and its forced per-pair lines and the attempt call are its "
                     f"reading (attempts_probe_only: on the probe texts the call was fixed for; attempts: on "
                     f"every attempted text)"),
            "features": per, "keep": keep, "exploratory_pass": exploratory, "attempts": att,
            "attempts_probe_only": att_probe,
            "attempt_primary": sem.get("primary_attempt"), "attempt_logprobs": sem.get("logprobs"),
            "attempts_vs_4b": (state.get("attempts") or {}).get("reference_4b"),
            "posfree_head": {k: per[f"head {POSFREE_HEAD}"].get(k) for k in ("gate_pass_nested", "sign_prong",
                                                                             "best_nested_tl")}
            if f"head {POSFREE_HEAD}" in per else None,
            "call": ("KEEP " + ", ".join(keep)) if keep else "NULL: no declared primary passes"}


def entropy_verdict(est):
    """ENTROPY_RULE on the entropy section: per harness covariate the nested gate
    lines and the step-8 sign prong (PRONG_FIELD, as for every declared feature);
    ENTROPY_PRIMARY is kept only if it passes both, is readable and on the rule's
    version and job config; any other feature that passes both is exploratory,
    never kept. The consistency check and the reference are set beside it, not
    gating. The call is final only on a complete export (ingest's completeness)
    under an unchanged rule (entropy.rule's digest, the one at the first ingest
    and ENTROPY_RULE's agree); otherwise it is prefixed PRELIMINARY or RULE
    CHANGED."""
    hz = {k: v for k, v in (est.get("harness") or {}).items() if not k.startswith("_")}
    sg = (est.get("signs") or {}).get("features", {})
    reg = (est.get("signs") or {}).get("registry", {})
    sem = (est.get("ingest") or {}).get("semantics") or {}
    eligible = bool(sem.get("primary_eligible"))
    per = {}
    for name, e in hz.items():
        lines = e["lines"]
        nested, gate_pass = nested_gate(lines)
        a = (sg.get(name) or {}).get("agreement_prong") or {}
        sign_ok = bool(a.get("declared_sign_ok"))
        r = reg.get(name, {})
        primary = name == ENTROPY_PRIMARY
        per[name] = {"primary": primary, "role": r.get("role"), "excluded": r.get("excluded"),
                     "gate_pass_nested": gate_pass, "sign_prong": sign_ok, "positive_parents_prong": a.get("positive"),
                     "units": a.get("units"),
                     "positive_parents_within": ((sg.get(name) or {}).get("agreement_within") or {}).get("positive"),
                     "random_effects_within": (sg.get(name) or {}).get("random_effects_within"),
                     "within_pair_r": e.get("r_within_pair_tl"), "best_nested_tl": best_nested(nested),
                     "nested": nested, "forced_per_pair_tl": forced_per_pair(lines),
                     "keep": bool(primary and eligible and gate_pass and sign_ok)}
    keep = sorted(n for n, v in per.items() if v["keep"])
    exploratory = sorted(n for n, v in per.items() if not v["primary"] and v["gate_pass_nested"] and v["sign_prong"])
    p = per.get(ENTROPY_PRIMARY)
    if keep:
        call = "KEEP " + ", ".join(keep)
    elif not sem.get("readable"):
        call = f"NOT READ: {sem.get('why') or 'the entropy semantics are unknown'}"
    elif p is None:
        call = f"NO DATA: {ENTROPY_PRIMARY} has no harness line (run --stage harness --job entropy)"
    elif not eligible:
        call = (f"NULL: {sem.get('why_not_eligible') or 'the export is not the one the rule was fixed for'}; the "
                "declared primary cannot be kept")
    else:
        why = [w for w, ok in (("the gate", p["gate_pass_nested"]), ("the sign prong", p["sign_prong"])) if not ok]
        call = f"NULL: the declared primary {ENTROPY_PRIMARY} fails " + " and ".join(why)
    # final only on a complete export under the rule of the first ingest
    rule_now = entropy_rule_digest()
    stored = est.get("rule") if isinstance(est.get("rule"), dict) else {}
    stored_dig = stored.get("digest")
    first = (est.get("rule_first") or {}).get("digest")
    rule_unchanged = bool(stored) and stored_dig == entropy_rule_digest(stored) == rule_now == first
    comp = (est.get("ingest") or {}).get("completeness") or {}
    complete = comp.get("complete") is True
    marks = []
    if not rule_unchanged:
        marks.append(f"RULE CHANGED (at the first ingest {(first or 'none')[:16]}, stored "
                     f"{(stored_dig or 'none')[:16]}, ENTROPY_RULE {rule_now[:16]})")
    if not complete:
        part = {b: f"{v.get('units_covered')}/{v.get('units_expected')}" for b, v in
                (comp.get("per_parent") or {}).items() if not v.get("complete")}
        marks.append("PRELIMINARY (" + (f"partial export, units covered/expected {part}" if comp else
                                        "completeness unknown: run --stage ingest again") + ")")
    if marks:
        call = "; ".join(marks) + ": " + call
    cons = est.get("consistency") or {}
    ref = est.get("reference") or {}
    return {"rule": entropy_rule_record(), "rule_digest": rule_now, "rule_digest_stored": stored_dig,
            "rule_digest_at_first_ingest": first, "rule_unchanged": rule_unchanged,
            "complete": complete, "completeness": comp.get("per_parent"), "final": complete and rule_unchanged,
            "features": per, "keep": keep, "exploratory_pass": exploratory,
            "primary": ENTROPY_PRIMARY, "primary_eligible": eligible,
            "semantics": {k: sem.get(k) for k in ("logprobs", "recorder_check", "version", "cfg", "readable", "why",
                                                  "why_not_eligible")},
            "consistency (not gating)": {k: cons.get(k) for k in ("attempts", "test_retest")} if cons else None,
            "reference (not gating)": ({"covered_by": ref.get("covered_by"), "covered_keys_in_rows":
                                        ref.get("covered_keys_in_rows"),
                                        "transferred_nested_tl": {r: (v.get("transferred nested") or {}).get("tl")
                                                                  for r, v in (ref.get("r") or {}).items()}}
                                       if ref else None),
            "call": call}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return round(L4.pearson(L4.ranks(x), L4.ranks(y)), 4) if len(x) >= CONSISTENCY_MIN else None


def icc1(groups):
    """One-way random-effects ICC(1) over groups of repeated measurements of
    unequal size (n0 = (N - sum n_i^2 / N) / (g - 1)); None under CONSISTENCY_MIN
    groups of at least two."""
    gs = [np.asarray(g, float) for g in groups if len(g) >= 2]
    k = len(gs)
    if k < CONSISTENCY_MIN:
        return None
    n = sum(len(g) for g in gs)
    grand = float(np.concatenate(gs).mean())
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs) / (k - 1)
    msw = sum(float(((g - g.mean()) ** 2).sum()) for g in gs) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in gs) / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    return round(float((msb - msw) / den), 4) if den > 0 else None


def consistency(je, jm=None, att_sem=None, ent_sem=None):
    """The entropy job beside the attempts on matharena, and its test-retest
    (entropy.consistency; reported, never gating). attempts: per design and
    CONSISTENCY_WITH feature, Spearman of this job's ENTROPY_PRIMARY against the
    attempts' aggregate over the matharena items both carry finite, over one
    item per entropy unit (key), and per prompt (the attempts' unit is the task
    text: the mean primary over a prompt's entropy units, every one of them,
    against the mean attempt value over its items). test_retest: per benchmark,
    the units whose prompts are identical (the same prompt_sha: the task text
    alone) but which are other units (another key: other metadata, so another
    seed): one pair per prompt (the first two keys), Spearman and Pearson of the
    primary; and the one-way ICC(1) over every unit of such prompts (icc1)."""
    out = {"what": ("matharena: this job's one sample (the neutral instruction, no metadata, 1,024 tokens) against "
                    "the mean over the attempts' k samples (the \\boxed instruction, 4,096 tokens) of the same "
                    "windowed statistics; and a single sample's test-retest on identical prompts"),
           "gating": False, "entropy_logprobs": (ent_sem or {}).get("logprobs"),
           "attempt_logprobs": (att_sem or {}).get("logprobs"),
           "same_definition": (ent_sem or {}).get("logprobs") == "raw" and (att_sem or {}).get("logprobs") == "raw"}
    ok = np.isfinite(je[ENTROPY_PRIMARY].to_numpy(float)) if ENTROPY_PRIMARY in je.columns else np.zeros(len(je), bool)
    e = je[ok]
    mh = e[e["benchmark"] == "matharena"]
    x = dict(zip(mh["item_id"], mh[ENTROPY_PRIMARY].to_numpy(float)))
    unit = dict(zip(mh["item_id"], mh["key"]))
    psha = dict(zip(mh["item_id"], mh["prompt_sha"])) if "prompt_sha" in mh.columns else {}
    px = {}                                           # prompt -> {unit: x}: every entropy unit of the prompt
    for i, p in psha.items():
        px.setdefault(p, {})[unit[i]] = x[i]
    att = {}
    if jm is None:
        out["attempts"] = None
        out["attempts_why"] = "no rubric-and-attempt features in WORK"
    else:
        mm = jm[jm["benchmark"] == "matharena"]
        for d, cols in sorted(attempt_designs(jm.columns).items()):
            for f in CONSISTENCY_WITH:
                if f not in cols:
                    continue
                y = dict(zip(mm["item_id"], mm[cols[f]].to_numpy(float)))
                ids = sorted(i for i in x if i in y and np.isfinite(y[i]))
                first = {}
                for i in ids:
                    first.setdefault(unit[i], i)
                us = sorted(first.values())
                py = {}
                for i in ids:
                    if i in psha:
                        py.setdefault(psha[i], []).append(y[i])
                ps = sorted(py)
                att[f"{d}/{f}"] = {"design": d, "attempt_feature": f, "n_items": len(ids), "n_units": len(us),
                                   "spearman": spearman([x[i] for i in ids], [y[i] for i in ids]),
                                   "spearman_units": spearman([x[i] for i in us], [y[i] for i in us]),
                                   "n_prompts": len(ps),
                                   "spearman_prompts": spearman([np.mean(list(px[p].values())) for p in ps],
                                                                [np.mean(py[p]) for p in ps])}
        out["attempts"] = att
    tr = {}
    for b, sub in e.groupby("benchmark", sort=True):
        if "prompt_sha" not in sub.columns:
            continue
        pairs, groups = [], []
        for _, g in sub.groupby("prompt_sha", sort=True):
            ks = g.drop_duplicates("key").sort_values("key")
            if len(ks) >= 2:
                v = ks[ENTROPY_PRIMARY].to_numpy(float)
                pairs.append(v[:2])
                groups.append(v)
        if pairs:
            P = np.array(pairs)
            tr[str(b)] = {"pairs": len(P), "spearman": spearman(P[:, 0], P[:, 1]),
                          "pearson": round(L4.pearson(P[:, 0], P[:, 1]), 4) if len(P) >= CONSISTENCY_MIN else None,
                          "mean_abs_diff": round(float(np.mean(np.abs(P[:, 0] - P[:, 1]))), 5),
                          "sd_over_units": round(float(np.std(sub.drop_duplicates("key")[ENTROPY_PRIMARY])), 5),
                          "units_in_repeated_prompts": int(sum(map(len, groups))), "icc1": icc1(groups)}
    out["test_retest"] = tr
    return out


def stage_consistency(args):
    import pandas as pd
    t0 = time.time()
    state = H.load_json(args.out) or {}
    est = entropy_state(state)
    je, _ = entropy_work_checked(args.work, est)
    p = os.path.join(args.work, "features.parquet")
    jm = pd.read_parquet(p) if os.path.exists(p) else None
    out = consistency(je, jm, (state.get("ingest") or {}).get("attempt_semantics"),
                      (est.get("ingest") or {}).get("semantics"))
    out["features_digest"] = entropy_digest(args.work)
    out["main_features_digest"] = features_digest(args.work)
    out["wall_s"] = round(time.time() - t0, 1)
    _save_part_entropy(args.out, "consistency", out)
    for k, v in (out.get("attempts") or {}).items():
        log(f"consistency, matharena: {ENTROPY_PRIMARY} vs attempts {k}: Spearman {v['spearman']} over {v['n_items']} "
            f"items ({v['spearman_units']} over {v['n_units']} units, {v['spearman_prompts']} over {v['n_prompts']} "
            "prompts)")
    for b, v in out["test_retest"].items():
        log(f"test-retest {b}: {v['pairs']} pairs of identical prompts, Spearman {v['spearman']}, Pearson "
            f"{v['pearson']}; ICC(1) {v['icc1']} over all {v['units_in_repeated_prompts']} units of those prompts")
    log(f"consistency: {out['wall_s']}s -> {args.out} [{ENTROPY_KIND}] (reported, not gating)")


def entropy_provenance(work, est):
    """The verdict's inputs against the table ingest recorded: WORK/entropy.parquet's
    digest, the signs' features digest, and every harness line's x digest against
    the map rebuilt from WORK on the ingested semantics (entropy_harness_maps) ->
    {ok, problems, features_digest, harness_lines_checked}."""
    want = (est.get("ingest") or {}).get("features_digest")
    probs = []
    dig = entropy_digest(work)
    if dig != want:
        probs.append(f"{os.path.join(work, ENTROPY_TABLE)} is {dig}, ingest recorded {want}")
    sd = (est.get("signs") or {}).get("features_digest")
    if sd != want:
        probs.append(f"entropy.signs were computed on {sd}, ingest recorded {want}")
    hz = {k: v for k, v in (est.get("harness") or {}).items() if not k.startswith("_")}
    n = 0
    if hz and dig == want:
        joined = load_entropy(work)
        maps = entropy_harness_maps(joined, entropy_stage_registry(joined, est))
        for name, e in sorted(hz.items()):
            if name not in maps:
                probs.append(f"entropy.harness[{name!r}] is not a usable feature of the ingested table")
            elif e.get("x_digest") != maps_digest({name: maps[name]}):
                probs.append(f"entropy.harness[{name!r}] was computed on another x ({e.get('x_digest')}, the ingested "
                             f"table gives {maps_digest({name: maps[name]})})")
            else:
                n += 1
    return {"ok": not probs, "problems": probs, "features_digest": want, "harness_lines_checked": n}


def stage_verdict_entropy(args):
    state = H.load_json(args.out) or {}
    est = entropy_state(state)
    if "harness" not in est or "signs" not in est:
        raise SystemExit("run --stage signs --job entropy and --stage harness --job entropy first")
    prov = entropy_provenance(args.work, est)
    if not prov["ok"]:
        raise SystemExit("the entropy verdict reads only what --stage ingest recorded: " + "; ".join(prov["problems"])
                         + " (run signs and harness --job entropy again with the --work and --out ingest used)")
    v = entropy_verdict(est)
    if not v["rule_unchanged"] and not getattr(args, "accept_rule_change", False):
        raise SystemExit(f"ENTROPY_RULE ({v['rule_digest'][:16]}) is not the rule of the first ingest "
                         f"({(v['rule_digest_at_first_ingest'] or 'none')[:16]}) or the one stored "
                         f"({(v['rule_digest_stored'] or 'none')[:16]}): the decision was fixed before the output was "
                         "read. Restore it, or pass --accept-rule-change to read the call marked RULE CHANGED")
    v["provenance"] = prov
    v["script_digest"] = H.digest(["experiments/strong_llm_eval.py"])
    v["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    est["verdict"] = v
    est.setdefault("history", []).append({
        "stage": "verdict", "time": v["time"], "call": v["call"], "keep": v["keep"],
        "exploratory_pass": v["exploratory_pass"], "primary": v["primary"], "primary_eligible": v["primary_eligible"],
        "complete": v["complete"], "final": v["final"], "rule_digest": v["rule_digest"],
        "rule_unchanged": v["rule_unchanged"], "features_digest": prov["features_digest"],
        "script_digest": v["script_digest"],
        "primary_line": {k: ((v["features"].get(ENTROPY_PRIMARY) or {}).get(k)) for k in (
            "gate_pass_nested", "sign_prong", "best_nested_tl", "within_pair_r")}})
    state[ENTROPY_KIND] = est
    H.save_json(args.out, state)
    print(json.dumps(H._jsonable({k: v[k] for k in ("call", "keep", "exploratory_pass", "primary_eligible", "complete",
                                                     "final", "rule_digest")}), indent=1))


def stage_verdict(args):
    if getattr(args, "job", "main") == ENTROPY_KIND:
        return stage_verdict_entropy(args)
    state = H.load_json(args.out)
    if not state or "harness" not in state or "signs" not in state:
        raise SystemExit("run --stage signs and --stage harness first")
    state["verdict"] = verdict(state)
    state["verdict"]["script_digest"] = H.digest(["experiments/strong_llm_eval.py"])
    H.save_json(args.out, state)
    v = state["verdict"]
    print(json.dumps({k: v[k] for k in ("call", "keep", "exploratory_pass", "attempts", "attempts_probe_only")},
                     indent=1))


# --- show ---------------------------------------------------------------------------------

def _c(v, nd=2):
    if not isinstance(v, dict) or v.get("est") is None:
        return ""
    ci = v.get("ci_group") or v.get("ci_item")
    return f"{v['est']:+.{nd}f}" + ("" if not ci else f" [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]")


def _g(x, nd=4):
    return "" if x is None else f"{x:+.{nd}f}"


def _pi(re_):
    """'mean [prediction interval]' of a random-effects summary."""
    if not re_ or re_.get("mean") is None:
        return ""
    lo, hi = (re_.get("prediction_interval") or [None, None])[:2]
    return f"{re_['mean']:+.2f}" + ("" if lo is None or hi is None else f" [{lo:+.2f}, {hi:+.2f}]")


def _rho(r):
    if not r:
        return ""
    ci = r.get("ci") or [None, None]
    return f"{r['rho']:+.2f}" + ("" if ci[0] is None else f" [{ci[0]:+.2f}, {ci[1]:+.2f}]")


def stage_show(args):
    s = H.load_json(args.out) or {}
    m = s.get("meta") or {}
    if m:
        print("model:", json.dumps((m.get("manifest") or {}).get("model")))
    ing = s.get("ingest")
    if ing:
        print("\n| benchmark | items | rows | hash ok | mismatch | missing | unknown | covered (by duplicate) | "
              "responses covered |\n|" + "---|" * 9)
        for b, r in ing["per_benchmark"].items():
            print(f"| {b} | {r['items']} | {r['rows']} | {r['hash_ok']} | {r['hash_mismatch']} | {r['hash_missing']} "
                  f"| {r['unknown_item']} | {r.get('covered_items')} ({r.get('covered_duplicate')}) | "
                  f"{ing.get('responses', {}).get(b, {}).get('covered_share')} |")
    sem = (ing or {}).get("attempt_semantics")
    if sem and sem.get("has_attempts"):
        print(f"\nattempt log-probs: {sem['logprobs']} ({sem['logprobs_manifest']!r}); presence penalty "
              f"{sem['presence_penalty']}; answer log-prob: {sem['lp_answer']['why']} (uniform "
              f"{sem['lp_answer']['uniform']}); raw features {sem['raw_features']}; primary {sem['primary_attempt']} "
              f"(order {sem['primary_order']}); beside the 4B D2 lead: {sem['d2']['comparable']} "
              f"({sem['d2']['reason']})")
        for n in sem.get("notes") or []:
            print("  note:", n)
    sg = s.get("signs")
    if sg:
        _show_signs(sg)
    hd = s.get("heads")
    if hd:
        print("\nheads, leave one parent out (Pearson over the benchmark / within group; the prong's statistic "
              f"{HEAD_PRONG_FIELD} on matharena)\n| head | primary | position-free | " + " | ".join(PARENTS)
              + " | mean | positive | prong | RE [PI] | choices |\n|" + "---|" * (len(PARENTS) + 8))
        for n, e in hd.items():
            if "pearson" not in e:
                print(f"| {n} | {e.get('primary')} | {e.get('posfree')} | {e.get('skipped')} |")
                continue
            cells = []
            for q in PARENTS:
                c = f"{e['pearson'].get(q, float('nan')):+.2f} / {e['pearson_within_group'].get(q, float('nan')):+.2f}"
                if q in HEAD_PRONG_FIELD and (e.get("prong") or {}).get("values", {}).get(q) is not None:
                    c += f" ({e['prong']['values'][q]:+.2f})"
                cells.append(c)
            print(f"| {n} | {e['primary']} | {e.get('posfree')} | " + " | ".join(cells)
                  + f" | {_g(e['mean_pearson'], 3)} | {e['positive_parents']}/4 | "
                  f"{e.get('positive_parents_prong')}/4 | {_pi(e.get('random_effects'))} | "
                  f"{[f['choice'] for f in e['folds'].values()]} |")
    hz = s.get("harness")
    if hz:
        _show_harness(hz)
    if s.get("reference"):
        _show_reference(s["reference"])
    at = s.get("attempts")
    for label, part in (("", at), (", probe texts only", (at or {}).get("probe_only"))):
        if not part or not part.get("designs"):
            continue
        if part.get("flagged_units"):
            print(f"\nprobe texts ({part['source']}, sha256 {str(part.get('source_sha256'))[:16]}): units "
                  f"{part['flagged_units']}, item_ids {part['flagged_items']}, conflicting {part['conflicting_items']}"
                  f"; against PROBE_IDS: {json.dumps(part.get('probe_ids_check'))}")
        for d, v in part["designs"].items():
            print(f"\nattempts{label}, design {d}: {v.get('n_items')} items, {v.get('competitions')} competitions, "
                  f"graded accuracy {v.get('accuracy')} (by year {v.get('accuracy_by_year')})\n"
                  "| feature | honest b | strong-tier b | 2025 | 2026 | sign-stage statistic | "
                  "the same, net of log length |\n|" + "---|" * 7)
            for f, r in v["features_vs_honest"].items():
                by = v["by_year_honest"].get(f, {})
                print(f"| {f} | {_rho(r)} | {_rho(v['features_vs_strong'].get(f))} | {_rho(by.get('2025'))} | "
                      f"{_rho(by.get('2026'))} | {_c((v.get('spearman_within_corr_block') or {}).get(f))} | "
                      f"{_c(v['partial_on_log_length'].get(f))} |")
            for f, e in (v.get("excluded") or {}).items():
                print(f"  excluded {f}: {_rho(e['vs_honest'])} ({e['reason']})")
        print(f"\nattempt decision{label}:", json.dumps(part.get("decision")))
    if at and at.get("designs"):
        print("beside the 4B probe (D2):", json.dumps(at.get("reference_4b")))
    if s.get("run"):
        print("\nrun:", json.dumps(s["run"], indent=1))
    if s.get("verdict"):
        v = s["verdict"]
        print("\nverdict:", json.dumps({k: v.get(k) for k in ("call", "keep", "exploratory_pass")}, indent=1))
    if s.get(ENTROPY_KIND):
        show_entropy(s[ENTROPY_KIND])


def _show_signs(sg):
    print("\nwithin-group Spearman, oriented (+ = harder), 95% group bootstrap; matharena net of competition, "
          "log length and position in brackets\n| feature | sign | " + " | ".join(PARENTS)
          + " | matharena text-bearing | 2026 | declared sign on (prong) | within group | RE mean [PI] |\n|"
          + "---|" * (len(PARENTS) + 7))
    for f, e in sg["features"].items():
        u = e["units"]
        cells = []
        for q in PARENTS:
            c = _c((u.get(q) or {}).get("spearman_within"))
            if q in PRONG_FIELD and (u.get(q) or {}).get(PRONG_FIELD[q]):
                c += f" ({_c(u[q][PRONG_FIELD[q]])})"
            cells.append(c)
        pr = e.get("agreement_prong") or {}
        print(f"| {f} | {'excluded' if e.get('excluded') else e['declared_sign']} | " + " | ".join(cells)
              + f" | {_c((u.get('matharena text-bearing') or {}).get('spearman_within'))} | "
              f"{_c((u.get('matharena text-bearing 2026') or {}).get('spearman_within'))} | "
              f"{pr.get('positive')}/{pr.get('units')} | "
              f"{e['agreement_within']['positive']}/{e['agreement_within']['units']} | "
              f"{_pi(e.get('random_effects_within'))} |")


def _show_harness(hz):
    print("\n| covariate, line | test-like ± cluster SE (sel) | benchmark-equal | worst parent | mix/whole | "
          "R1 b / p | folds on | gate | placebo test-like |\n|" + "---|" * 9)
    for f, v in hz.items():
        if f.startswith("_"):
            continue
        for n, ln in v["lines"].items():
            pl = v["placebo"].get(n, {})
            sel = ln.get("tl_sel_se")
            gate = ln["gate_pass"] if ln["gate_pass"] is not None else f"if on: {ln['gate_pass_if_on']}"
            print(f"| {f}, {n} | {_g(ln['tl'], 5)} ± {ln['tl_cluster_se']:.5f}"
                  + ("" if sel is None else f" ({sel:.5f})") + f" | {_g(ln['tl_benchmark_equal'], 5)} | "
                  f"{_g(ln['tl_worst_parent'], 5)} | {_g(ln['mix'], 5)} | {_g(ln['r1b'], 5)} / {_g(ln['r1p'], 5)}"
                  f" | {'' if ln['folds_on'] is None else str(ln['folds_on']) + '/4'} | {gate} | "
                  f"{_g(pl.get('tl'), 5)} |")
        print(f"  {f}: within-pair r {v.get('r_within_pair_tl')}, coverage {v['coverage_eval_items']}")


def _show_reference(ref):
    print("\nreference (honest difficulty degraded to r on the covered items):")
    for r, t in ref["r"].items():
        print(f"  {r}: " + "  ".join(f"{n} {v['tl']:+.5f}" for n, v in t.items()))


def show_entropy(est):
    """The entropy section's tables (show)."""
    rule = est.get("rule") if isinstance(est.get("rule"), dict) else {}
    cfg = rule.get("config") or {}
    now = entropy_rule_digest()
    print(f"\n## the entropy job ({ENTROPY_KIND}): primary {rule.get('primary')} (the stored rule, digest "
          f"{(rule.get('digest') or 'none')[:16]}; fixed for version {cfg.get('version')}, config {cfg.get('cfg')})"
          + ("" if rule and rule.get("digest") == entropy_rule_digest(rule) == now
             == (est.get("rule_first") or {}).get("digest") else
             f" -- WARNING: the stored rule's content is {entropy_rule_digest(rule)[:16] if rule else 'none'}, "
             f"ENTROPY_RULE {now[:16]}, "
             f"the first ingest's {((est.get('rule_first') or {}).get('digest') or 'none')[:16]}")
          + (f"; {len(est['rule_history'])} earlier rule(s) in rule_history" if est.get("rule_history") else ""))
    ing = est.get("ingest")
    if ing:
        sem = ing.get("semantics") or {}
        print(f"source {(est.get('meta') or {}).get('kaggle_dir')}; version {sem.get('version')}, log-probs "
              f"{sem.get('logprobs')}, recorder check {sem.get('recorder_check')}, readable {sem.get('readable')}"
              + ("" if sem.get("readable") else f" ({sem.get('why')})") + f"; primary eligible "
              f"{sem.get('primary_eligible')}; rubric-and-attempt features: {ing.get('main_features')}")
        print("\n| benchmark | items | rows | hash ok | mismatch | missing | unknown | covered | with the primary | "
              "degenerate excluded | responses covered | mean ent_first1024 | closed | truncated |\n|" + "---|" * 14)
        for b, r in (ing.get("per_benchmark") or {}).items():
            st = (ing.get("per_benchmark_stats") or {}).get(b, {})
            print(f"| {b} | {r['items']} | {r['rows']} | {r['hash_ok']} | {r['hash_mismatch']} | {r['hash_missing']} "
                  f"| {r['unknown_item']} | {r.get('covered_items')} | {st.get('primary_finite')} | "
                  f"{st.get('degenerate_excluded')} | {ing.get('responses', {}).get(b, {}).get('covered_share')} | "
                  f"{st.get('mean_' + ENTROPY_PRIMARY)} | {st.get('rate_ent_closed')} | "
                  f"{st.get('rate_ent_truncated')} |")
        comp = ing.get("completeness") or {}
        part = {b: f"{v.get('units_covered')}/{v.get('units_expected')}" for b, v in
                (comp.get("per_parent") or {}).items()}
        print(f"\ncomplete export: {comp.get('complete')} (units covered/expected {part})")
    if est.get("signs"):
        _show_signs(est["signs"])
    if est.get("harness"):
        _show_harness(est["harness"])
    if est.get("reference"):
        _show_reference(est["reference"])
    c = est.get("consistency")
    if c:
        print("\nconsistency (not gating):", json.dumps({k: c.get(k) for k in ("same_definition", "attempts",
                                                                                "test_retest")}, indent=1))
    if est.get("run"):
        print("\nentropy run:", json.dumps({k: est["run"].get(k) for k in ("timeline", "derived")}, indent=1))
    if est.get("verdict"):
        v = est["verdict"]
        print("\nentropy verdict:", json.dumps({k: v.get(k) for k in ("call", "keep", "exploratory_pass",
                                                                       "primary_eligible", "complete", "final",
                                                                       "rule_digest")}, indent=1))
    if est.get("history"):
        print(f"\nentropy history ({len(est['history'])} entries, append-only): " + "; ".join(
            f"{h.get('time')} {h.get('stage')} {h.get('features_digest')}"
            + (f" {h.get('call')!r}" if h.get("stage") == "verdict" else "") for h in est["history"][-6:]))


# --- io -----------------------------------------------------------------------------------

def meta(args, man, kdir=None):
    man = man or {}
    kdir = kdir if kdir is not None else kaggle_dirs(args)[0]
    return {**H.provenance(), "script_digest_strong": H.digest(["experiments/strong_llm_eval.py"]),
            "kaggle_dir": os.path.relpath(kdir, ROOT) if kdir.startswith(ROOT) else kdir,
            "schema_version": SCHEMA_VERSION, "hash_def": HASH_DEF,
            "manifest": {k: man.get(k) for k in ("schema_version", "model", "kinds", "created", "wall_s", "gpu",
                                                 "notebook_digest", "hash", "logprobs", "slug")},
            "primary": {"head": PRIMARY_HEAD, "posfree_head": POSFREE_HEAD,
                        "attempt_feature_order": list(PRIMARY_ATTEMPT_ORDER)},
            "prong_fields": {"feature": PRONG_FIELD, "head": HEAD_PRONG_FIELD},
            "rules": {"attempts": {"go_rho": GO_RHO, "go_lo": GO_LO, "go_2026": GO_2026, "kill": KILL_RHO,
                                   "floor_acc": FLOOR_ACC}, "gate": H.GATE, "keep_signs": KEEP_SIGNS,
                      "keep_r": KEEP_R},
            "sign_rules": [[k, list(p), s] for k, p, s in SIGN_RULES]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("schema", "check-schema", "ingest", "signs", "harness",
                                                        "reference", "attempts", "consistency", "verdict", "run",
                                                        "show"))
    ap.add_argument("--kaggle", nargs="+", default=[KAGGLE_DIR],
                    help="the Kaggle export directories (manifest.json + parquet shards); several are merged by job: "
                         "each job's shards from the first that carries them")
    ap.add_argument("--job", choices=("main", ENTROPY_KIND), default="main",
                    help="signs, harness, reference, verdict: the rubric's and the attempts' features (main) or the "
                         "entropy job's (commit D; OUT's entropy section)")
    ap.add_argument("--work", default=WORK, help="derived tables (features.parquet, targets.json, oof.json)")
    ap.add_argument("--rows", default=H.ROWS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boots", type=int, default=BOOT)
    ap.add_argument("--placebo", type=int, default=N_PLACEBO)
    ap.add_argument("--feats", nargs="+", default=None, help="harness: only these covariates (names as in show)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--force", action="store_true", help="ingest despite schema errors")
    ap.add_argument("--accept-rule-change", action="store_true",
                    help="the entropy job: ingest under an ENTROPY_RULE that differs from the one OUT's entropy "
                         "results were read under (the old rule goes to entropy.rule_history, those results are "
                         "dropped), and a verdict under a rule other than the first ingest's (its call marked RULE "
                         "CHANGED)")
    ap.add_argument("--refresh-targets", action="store_true")
    ap.add_argument("--attempt-units", choices=ATTEMPT_UNITS, default="both",
                    help="attempts: every attempted text ('all'), the probe texts the rule was fixed for "
                         f"('probe', flagged in KAGGLE_DIR/{UNITS_DETAIL}), or both; each keeps the other's result")
    ap.add_argument("--run-manifest", default=None,
                    help="run: the notebook's root manifest (strong_probe/manifest.json of the Kaggle Output); "
                         f"default: the one under {os.path.relpath(KAGGLE_RAW, ROOT)} that matches the export")
    args = ap.parse_args()
    if args.job == ENTROPY_KIND and args.stage == "attempts":
        ap.error("--job entropy applies to signs, harness, reference and verdict; attempts reads the attempts, and "
                 "check-schema, ingest, consistency, run and show handle both jobs by themselves")
    {"schema": stage_schema, "check-schema": stage_check_schema, "ingest": stage_ingest, "signs": stage_signs,
     "harness": stage_harness, "reference": stage_reference, "attempts": stage_attempts,
     "consistency": stage_consistency, "verdict": stage_verdict, "run": stage_run, "show": stage_show}[args.stage](args)


if __name__ == "__main__":
    main()
