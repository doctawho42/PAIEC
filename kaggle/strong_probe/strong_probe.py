#!/usr/bin/env python3
"""A stronger open model on a free Kaggle GPU: demand rubrics and solving attempts
for the public items of measurement-db (PAIEC, task K1).

Every local language-model item signal was null with Qwen3-4B (docs/findings.md:
"The 4B judge, closed out", "Attempting instead of judging", "Entropy profiles and
hidden-state probes", "Few-shot prompting"), and the 4B is too weak as a subject
(2-8% graded accuracy on matharena). This script runs the plan's step 8 on a free
Kaggle notebook (2x NVIDIA T4, 16 GB each, compute capability 7.5) with a dense
Qwen3 model in 4-bit AWQ under vLLM, and writes item features that
experiments/strong_llm_eval.py reads locally. It is self-contained: the notebook
paiec_strong_probe.ipynb carries this file verbatim (build_notebook.py writes it)
and runs it as a subprocess.

Jobs
  rubric    every unique item of the four multi-subject parents (matharena,
            multi_swebench, real_webagents, researchcodebench; swe_rebench with
            --rubric-benchmarks), unique as paiec.predict.item_key sees them
            (content, features, benchmark), thinking OFF, greedy, one short JSON
            answer read out from the next-token digit probabilities (top-20
            logprobs at the digit's position, renormalised over the allowed
            digits), not from the parsed text:
              (a) eight ADeLe/DeLeAn-style demand scales, 0-5 with anchors, the
                  benchmark-agnostic subset that fits any task type: reasoning,
                  knowledge, work, interaction, volume, atypicality, precision,
                  unguessability (exported as rubric_<scale>, expected level);
              (b) difficulty for strong 2025-26 AI systems as the share that fail,
                  in deciles 0-9 (exported as solve_share = 1 - expected fail
                  share at the deciles' midpoints);
              (c) a skilled human expert's time in ten log-spaced buckets
                  (exported as time_log_minutes, the expected natural log of the
                  bucket's geometric midpoint in minutes).
            The task text (item_content) is cut to --task-tokens (3072) by head
            and tail around MARKER; item_features follow as "Metadata" unless
            --rubric-no-meta, without position fields (META_DROP_RE: matharena's
            problem_idx is a within-competition position cue that would let a
            rating echo the slot instead of reading the task). No benchmark name
            is ever put into a prompt.
  attempts  matharena items with text (no image reference, >= 70 characters) and a
            short checkable reference answer (integer or expression in
            grading_criterion; proofs and Kangaroo's image multiple choice are
            out): 819 unique texts standing for 946 item_ids. The 147 texts of the
            attempt probe's 160 items (PROBE_IDS) go first, then the rest,
            round-robin over competitions so any prefix is spread. Thinking ON,
            k samples (--k 4) with the model card's settings (temperature 0.6,
            top-p 0.95, top-k 20, presence penalty 1.5 for quantised models),
            --max-tokens 4096 including the thinking (truncation is a feature:
            most hard problems end inside <think>). EVERY attempt then gets one
            greedy forced readout of its answer: the reasoning (through </think>,
            or all of it when the thinking did not close) + "... **Final Answer**
            $\\boxed{", at most 40 tokens, so lp_answer is one quantity (the raw
            greedy log-prob of the answer the reasoning leads to) for every
            attempt, whether or not the attempt boxed an answer itself. Per
            attempt: the answer (its own \\boxed{} after </think>, else the forced
            one), finish reason, length, whether the thinking closed, mean token
            log-prob and entropy over everything, the reasoning, the answer and
            the first 256 / 1024 tokens, and the per-token sequences (float16).
            --attempt-scope probe | rest | all picks the probe texts, the others,
            or both (probe first).
            Graded correctness against the reference answer (attempt_probe.canon's
            normal form) is a diagnostic kept in memory only: a shard's mean in the
            log and the session's mean in the manifest. Neither the reference
            answer nor any per-attempt or per-item correctness is written, and an
            attempt's canonical answer only as a hash (canon_sha): the model's own
            answers are in the output, but nothing there says which of them is
            right (LEAK_COLS; the local side grades against items.parquet).

Token statistics (tok_entropy, tok_lp and their spans, lp_boxed, lp_answer) are
those of the RAW model distribution, as the attempt probe's D2 lead measured them
(experiments/attempt_probe.Recorder): vLLM's V0 engine (0.9.x on a T4) returns
logprobs only after the presence penalty, temperature and top-k/top-p (the penalty
depends on the tokens generated so far, so their entropy drifts with position), so
every attempt and forced request carries a V0 per-request logits processor
(RawRecorder) that records, from the raw logits row, the full-vocabulary entropy
and the sampled token's log-prob. A greedy request's engine logprobs are raw too;
every shard compares the two on the forced readouts (recorder_check in the
manifest). Without the recorder (vLLM V1, >= 0.11) the export falls back to the
engine's top-5 logprobs, the remaining mass as one bucket (a lower bound), and the
export manifest says so: logprobs "raw" (full vocabulary, raw distribution: the
D2 definition), "raw_topk", "processed_topk" or "mixed". The engine's top-5
statistics are kept beside the raw ones as *_engine diagnostics in _detail/.

Model and engine (the reasoning is in README.md). T4 is Turing: no bfloat16, no
FP8, no Marlin kernels (sm80+), no FlashAttention-2. vLLM's plain AWQ GEMM kernel
runs on sm75, and vLLM 0.9.2 falls back to its V0 engine (xformers attention,
prefix caching) on compute capability < 8. Default: Qwen/Qwen3-14B-AWQ (9.98 GB of
safetensors, dense, not a matharena subject), tensor parallel over both T4s, fp16,
--max-model-len 6144, prefix caching (the rubric's long instruction block is a
shared prefix; the k attempts share their prompt). Qwen/Qwen3-32B-AWQ (19.33 GB)
fits with TP=2 but leaves ~3 GB of KV cache per GPU: rubric only. --backend hf is
a fallback for the rubric job without vLLM: transformers with the unquantised
Qwen/Qwen3-8B in fp16 split over both GPUs (device_map="auto").

Data. measurement-db is gated: the notebook downloads the core tables itself with
the user's token from the Kaggle Secret HF_TOKEN (kaggle_secrets), into
--data-dir (/tmp/paiec_data, never /kaggle/working, which becomes the notebook's
output). The token is passed to huggingface_hub only; it is never printed, logged,
written or put into the environment. No item text, no reference answer and no
correctness flag is written to the output (the model's reasoning may restate a
task: keep the notebook private).

Output (--out, /kaggle/working/strong_probe on Kaggle), per model. Kaggle saves
at most 500 files of a notebook's output, mounts at most 500 of an output attached
as an input, and `kaggle kernels output` lists at most 500, so a shard is not a
file: every run process appends its shards to one segment per kind, rewritten
atomically after each shard and rotated at SEG_BYTES, and compaction merges
segments into few files (at the start and end of `run`, and before any shard once
the saved output holds MAX_FILES entries; FileGuard). A full run leaves about 20
files; the tests drive the worst case (a file per shard, several sessions, retries,
new configs) and require the count to stay under the guard.
  <model>/rubric/items-<run>-<nnn>.parquet     one row per unique item (unit)
  <model>/attempts/items-<run>-<nnn>.parquet   one row per unique text
  <model>/attempts/samples-<run>-<nnn>.parquet one row per attempt, with its text
                                               and per-token sequences
      (<run>: the process's run id, c<run><n> for a compacted segment; the k1.2
      layout's items_NNNNN / samples_NNNNN.parquet, one file per shard, is read
      the same way and compacted into this one at the start of `run`)
  <model>/export/                              what the local side reads: copy
                                               its contents to data/features/kaggle/
    manifest.json      experiments/strong_llm_eval.py's schema v1: model, the
                       content hash's definition, the prompts, readouts, signs,
                       and every shard with its rows and sha256
    rubric/rubric.parquet      one row per (benchmark, item_id): content_sha256
                               (sha256 of item_content + "\\n" + item_features, as
                               paiec.llmfeat.item_text) and the features
    attempts/attempts.parquet  one row per (benchmark, item_id, attempt)
    _keys.parquet      (benchmark, item_id) -> key, text_key (paiec.predict.
                       item_key as paiec.llmfeat stores them), content_sha,
                       the dataset's content_hash
    _detail/           the unit tables (digit probabilities, aggregated attempt
                       features) and every attempt's text and token sequences
    _harness.json      {<job>_<feature>: {item_id: x}}, + = harder; split-harness
                       writes one _harness/<name>.json per feature for
                       python experiments/harness.py --stage eval --cov ...
  manifest.json      config, engine, versions, progress, throughput, file counts,
                     and the segment files compaction merged away (`absorbed`)

Resuming. Work is cut into shards; a unit counts as done when an items row with
the same job config hash holds it, so a crash of the script (an exception, the
watchdog) loses at most the shard in flight, and the notebook's RUN cell starts
it again in the same session. What a Kaggle commit saves is its /kaggle/working
when it ends: a commit that is cancelled, dies or passes Kaggle's 12 hours saves
nothing (hence --session-hours 11, and one piece of work per commit: README). On
Kaggle, attach the previous version's output as an input: `run` copies its
segments from any /kaggle/input/**/strong_probe (or --resume-from) before
starting, except those an attached output's manifest lists as absorbed, and
merges that output's session history into this one's manifest. The session clock
starts at the notebook's first cell (SP_T0): no shard starts that would not
finish by --session-hours, a watchdog kills the process at the hard limit (exit
75) and if the engine does not come up within --init-timeout-min (exit 76), and
`run` exits 75 when it stopped with work left. Self-checks stop a run whose output
would be useless (exit 77, not retried: the rubric's first shard parses under
RUBRIC_MIN_PARSE; the attempts' first shard fails the recorder check, is mostly
degenerate text, or yields no answer at all); setup errors exit 78.

Commands
  python strong_probe.py plan    [--jobs rubric,attempts] ...  data, items, prompts, budget (CPU)
  python strong_probe.py run     [--jobs rubric,attempts] ...  generate (GPU), resumable
  python strong_probe.py export                                 rebuild export/ from the shards
  python strong_probe.py split-harness --export-dir DIR         _harness.json -> _harness/<name>.json (local)

Local check (no GPU, no download): python -m pytest -q tests/test_kaggle_probe.py
runs the whole pipeline on real local items with a mock backend.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np
import pandas as pd

VERSION = "k1.2"                 # what the jobs compute (prompts, readouts); part of every job config hash
LAYOUT = 2                       # how the store keeps it (2: segments; 1: a file per shard); not in the hash
SCHEMA_VERSION = 1               # experiments/strong_llm_eval.py's input schema
HASH_DEF = ("sha256 hex of utf-8(item_content + '\\n' + item_features), both as paiec.data.load_pairs "
            "builds them (paiec.llmfeat.item_text)")
DATA_REPO = "aims-foundations/measurement-db"
PARENTS = ("matharena", "multi_swebench", "real_webagents", "researchcodebench")
BENCHES = PARENTS + ("swe_rebench",)
TABLES = ("items", "response", "subjects", "benchmarks")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")
ON_KAGGLE = os.path.isdir("/kaggle/working")
DEFAULT_OUT = "/kaggle/working/strong_probe" if ON_KAGGLE else os.path.join(os.getcwd(), "strong_probe_out")
DEFAULT_DATA = "/tmp/paiec_data" if ON_KAGGLE else os.environ.get("PAIEC_DATA", os.path.join(os.getcwd(), "data"))
if ON_KAGGLE and (not os.environ.get("HF_HOME") or os.path.abspath(os.environ["HF_HOME"]).startswith("/kaggle/working")):
    os.environ["HF_HOME"] = "/tmp/hf"    # model weights never land in /kaggle/working (the notebook's output)

#: install pins (the notebook's setup cell): a vLLM that falls back to its V0 engine on
#: Turing (V0 was removed in 0.11), and a transformers of the same month (a newer
#: major breaks 0.9.x's imports)
VLLM_PIN = "0.9.2"
TRANSFORMERS_PIN = "4.53.2"

#: repo -> revision (commit sha), size and shape from HfApi().model_info (2026-09-27),
#: and the throughput assumed on 2x T4 before a run measures its own (tokens/s,
#: aggregate: prefill; decode of long attempts; decode of the rubric's short answers)
MODELS = {
    "Qwen/Qwen3-14B-AWQ": dict(revision="31c69efc29464b6bb0aee1398b5a7b50a99340c3", quantization="awq",
                               weights_gb=9.98, layers=40, kv_heads=8, head_dim=128,
                               prefill=1200, decode=300, decode_short=600),
    "Qwen/Qwen3-32B-AWQ": dict(revision="0499c3ac83fdef8810b907a23894ba91e95eddd8", quantization="awq",
                               weights_gb=19.33, layers=64, kv_heads=8, head_dim=128,
                               prefill=500, decode=110, decode_short=250),
    "Qwen/Qwen3-8B-AWQ": dict(revision="4da05a8edb55c6046cce958586c33b61da07bb79", quantization="awq",
                              weights_gb=6.10, layers=36, kv_heads=8, head_dim=128,
                              prefill=2000, decode=450, decode_short=900),
    "Qwen/Qwen3-8B": dict(revision="b968826d9c46dd6066d109eabc6255188de91218", quantization=None,
                          weights_gb=16.38, layers=36, kv_heads=8, head_dim=128,
                          prefill=500, decode=60, decode_short=150),
}
DEFAULT_MODEL = "Qwen/Qwen3-14B-AWQ"
HF_FALLBACK_MODEL = "Qwen/Qwen3-8B"
WEEKLY_GPU_H = 30.0
SESSION_OVERHEAD_H = 0.4         # per commit: install, model download and load, export
SMOKE_H = 0.7                    # the smoke run (README step 6), in the quota arithmetic
#: exit codes: stopped at the session deadline with work left; the engine did not start (init watchdog);
#: a self-check stopped the run; a setup error (no GPU, no token, a failed download). The RUN cell retries
#: only other failures (and 76).
EXIT_DEADLINE, EXIT_INIT, EXIT_CHECK, EXIT_CONFIG = 75, 76, 77, 78


class ConfigError(SystemExit):
    """A setup error that a retry cannot fix; exits EXIT_CONFIG with the message."""

    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg

MARKER = "\n[... middle of the task omitted ...]\n"
TASK_PH, META_PH = "\u0000TASK\u0000", "\u0000META\u0000"
META_MAX_TOKENS = 256
#: item_features fields left out of the rubric's metadata: position within a set
#: (matharena's problem_idx rises with difficulty inside a competition, and the
#: within-competition statistics downstream do not net position out)
META_DROP_RE = re.compile(r"(?:^|_)(?:idx|index|position|pos|order|rank)$", re.I)

# --- the rubric prompt ------------------------------------------------------------------

SCALES = {
    "reasoning": ("depth of logical, mathematical or causal reasoning needed",
                  "0 none: recall or copy an answer; 1 one simple inference or calculation; 2 a few "
                  "routine steps (a standard exercise); 3 a multi-step argument with a non-obvious step; "
                  "4 a long chain of reasoning that needs a creative insight (hard competition level); "
                  "5 exceptional insight, at the level of the hardest olympiad or research problems"),
    "knowledge": ("specialised knowledge needed beyond general education",
                  "0 none; 1 everyday knowledge; 2 high-school level; 3 undergraduate or working-"
                  "professional level; 4 graduate or senior-expert level; 5 frontier research knowledge "
                  "or deep familiarity with a niche system or codebase"),
    "work": ("amount of work: the number of distinct steps, edits or actions a competent solver needs",
             "0 one step; 1 two or three steps; 2 about five to ten steps; 3 tens of steps, or a change "
             "in a few places; 4 about a hundred steps, or a change across several files or components; "
             "5 hundreds of steps or a large multi-part implementation"),
    "interaction": ("need to act in an environment (run code, use tools, browse, operate software) and "
                    "react to its feedback",
                    "0 none: the answer follows from the text alone; 1 a tool would help but is not "
                    "needed; 2 one tool use or code run; 3 several tool calls or navigation steps with "
                    "feedback; 4 extended exploration of a codebase, website or system; 5 long-horizon "
                    "interaction with a complex, stateful environment"),
    "volume": ("amount of material to read, understand and keep track of, in the task and in anything "
               "it refers to",
               "0 a sentence or two; 1 a short paragraph; 2 about a page; 3 several pages or a small "
               "codebase; 4 a long document or a medium codebase; 5 a very large collection of documents "
               "or a large codebase"),
    "atypicality": ("how unusual the task is compared with common tasks of its kind",
                    "0 a textbook or very common task; 1 a familiar variant; 2 an unusual twist; 3 clearly "
                    "unusual; 4 rare, unlike most tasks of its kind; 5 highly novel, with no close "
                    "precedent"),
    "precision": ("how exacting the success criterion is: how easily a nearly right attempt still fails",
                  "0 lenient, many answers count; 1 mostly lenient; 2 moderate; 3 an exact answer or "
                  "behaviour is required; 4 exact, with hidden checks or edge cases; 5 a single small "
                  "error anywhere fails the whole task"),
    "unguessability": ("how hard it is to succeed by guessing, pattern-matching or a shortcut instead of "
                       "really solving the task",
                       "0 trivially guessable (yes or no, two options); 1 a few options; 2 a small answer "
                       "space, or a common default often works; 3 a moderate answer space; 4 a large "
                       "answer space; 5 success is impossible without really solving the task"),
}
DIFFICULTY_DEF = ("how hard the task is for strong AI systems of 2025-2026 (frontier language models "
                  "and agents, with whatever tools the task allows). Imagine many of them each attempt "
                  "it once: what share of them fail? 0 under 10% fail, 1 10-20%, and so on up to 9 for "
                  "90% or more. Benchmark tasks are chosen to challenge the best systems and are often "
                  "harder than they look.")
HUMAN_TIME_DEF = ("how long a skilled human expert in the field would take to complete the task: 0 under "
                  "1 minute, 1 1-3 minutes, 2 3-10 minutes, 3 10-30 minutes, 4 30-60 minutes, 5 1-2 hours, "
                  "6 2-4 hours, 7 4-8 hours, 8 8-24 hours, 9 more than a day.")
#: geometric midpoints of the human-time buckets, minutes
HUMAN_MID = (0.5, 1.7, 5.5, 17.0, 42.0, 85.0, 170.0, 340.0, 830.0, 2880.0)
RUBRIC_KEYS = tuple(SCALES) + ("difficulty", "human_time")
RANGES = {**{s: 6 for s in SCALES}, "difficulty": 10, "human_time": 10}
RUBRIC_JSON = "{" + ", ".join(f'"{k}": <0-{RANGES[k] - 1}>' for k in RUBRIC_KEYS) + "}"
SYSTEM_RUBRIC = (
    "You are an expert in the psychometrics of AI evaluation. You read one task from an AI "
    "evaluation benchmark and rate what it demands, on fixed scales that apply to any kind of task: "
    "questions, math problems, programming and software-engineering tasks, tasks for agents acting "
    "on websites or in computer environments, and the text part of tasks that also involve images "
    "or documents. Very long tasks are shortened, with the middle omitted. Metadata about the task "
    "may follow its text. Rate the task as it is given; do not solve it and do not explain.\n\n"
    "Demand scales, each 0-5:\n\n"
    + "\n\n".join(f"{k}: {d}.\n{a}." for k, (d, a) in SCALES.items())
    + "\n\nTwo more ratings, each 0-9:\n\n"
    f"difficulty: {DIFFICULTY_DEF}\n\nhuman_time: {HUMAN_TIME_DEF}\n\n"
    "Answer with one JSON object on one line, with exactly these keys in this order and a single "
    f"digit for each value:\n{RUBRIC_JSON}")
USER_RUBRIC = ("<task>\n" + TASK_PH + "\n</task>\n" + META_PH
               + "\nRate the demands of this task. Answer with the JSON object only.")
RUBRIC_PREFILL = '{"reasoning": '
RUBRIC_MAX_TOKENS = 96
RUBRIC_GEN_EST = 72              # generated tokens per rubric answer, for the budget
#: exported feature -> (declared sign against difficulty, + = harder; definition)
RUBRIC_EXPORT = {**{f"rubric_{s}": (1, f"{d}; expected level 0-5 from the digit probabilities")
                    for s, (d, _) in SCALES.items()},
                 "solve_share": (-1, "1 - expected share of strong 2025-26 systems that fail, from the "
                                     "0-9 decile digit probabilities at the deciles' midpoints"),
                 "time_log_minutes": (1, "expected natural log of a skilled human expert's time in minutes, "
                                         "from the 0-9 bucket digit probabilities at geometric midpoints")}

# --- the attempt prompt ------------------------------------------------------------------

BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
THINK_END = "</think>"
FORCE_OPEN = "\n</think>\n\n**Final Answer**\n\n$\\boxed{"      # the thinking did not close
FORCE_CLOSED = "\n\n**Final Answer**\n\n$\\boxed{"              # closed, but no \boxed{} answer
FORCE_TOKENS = 40
ATTEMPT_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
ATTEMPT_GEN_FRAC = 0.9           # budget: mean generated share of --max-tokens (most runs hit the cap)
FORCED_FRAC, FORCED_CACHE_HIT = 1.0, 0.5     # every attempt gets a forced readout; half its prefill cached
SLOW_DECODE = 0.5                # the budget's slow case (the planning figure): decode at half the assumed rate
RECORDER_TOL = 1e-2              # recorder check: |raw - engine| log-prob on greedy tokens (nats)
#: self-checks on a process's first shard (--no-self-check turns them off): the rubric's parse rate, and the
#: attempts' share of degenerate texts (one character most of a long text: fp16 overflow's '!!!!')
RUBRIC_MIN_PARSE, MAX_DEGENERATE = 0.5, 0.5
#: attempt diagnostics that need the reference answer: kept in memory (log, session mean), never written
REF_DIAG = ("correct", "graded", "graded_natural", "top_correct")
MIN_TEXT = 70
IMAGE_RE = re.compile(r"\bsee (the )?(image|figure|picture|diagram)\b|!\[[^\]]*\]\(|<image\b|\[image\]"
                      r"|<img\b", re.I)
REFUSE_RE = re.compile(r"cannot (be )?(determined|determine|solve)|not enough information|"
                       r"I'm unable|I am unable|insufficient information", re.I)
#: attempt aggregate -> declared sign (+ = harder); graded / top_correct are diagnostics
ATTEMPT_SIGNS = {"top_share": -1, "ans_entropy": 1, "fail_rate": 1, "trunc_rate": 1, "closed_rate": -1,
                 "forced_rate": 1, "mean_len": 1, "mean_think_len": 1, "tok_lp": -1, "tok_entropy": 1,
                 "tok_lp_think": -1, "tok_entropy_think": 1, "tok_lp_answer": -1, "tok_entropy_answer": 1,
                 "ent_first256": 1, "ent_first1024": 1, "lp_first256": -1, "lp_first1024": -1,
                 "lp_boxed": -1, "lp_answer": -1}
#: where an attempt's token statistics come from, and the export manifest's `logprobs`
STATS_RAW, STATS_RAW_TOPK, STATS_PROC = "raw_full_vocab", "raw_topk", "processed_topk"
STATS_DEF = {STATS_RAW: "the raw model distribution (before penalties, temperature and top-k/top-p): "
                        "full-vocabulary entropy and the sampled token's log-prob, recorded by a V0 "
                        "per-request logits processor (the attempt probe's D2 definition)",
             STATS_RAW_TOPK: "the engine's top-5 logprobs of the raw distribution (vLLM V1): the sampled "
                             "token's raw log-prob; entropy a lower bound, the remaining mass as one outcome",
             STATS_PROC: "the engine's top-5 logprobs AFTER penalties, temperature and top-k/top-p (vLLM "
                         "V0 without the recorder): not comparable with D2"}
#: the attempt probe's 160 matharena items (attempt-signal design a7; 147 unique texts)
PROBE_IDS = frozenset("""
676126bc9dd5e0fa 98b1be1d26b06e39 190a332a629869f9 3ef28cfdfcaa878b a4da728a82994f33 6fabe6323b98e041
1f88043e716b4e57 bb6ddf3a0e7a0496 bc8450f7a094500e d6fd91839534dc84 0bd69a9aaf7cd437 62cd475bb78d8c27
7e66059f78831f9c c9a92a82693f3791 46992bb1d7678be7 7498d8531e4e2409 5f55882f9345c4da a36a33d01b1616ed
015c236cea533fa7 30209bb1b5037490 7de7d787b11828e5 0edd9780af89b3ad 6a486a4b45fe2157 16ccbeb34f68a1a3
b601a9fd0f8d3aa5 d7c87f73d6298b8c f89fb7f9eb8ddfcd c8d4b1784d63de6c 87dc2ae25e292dad 61cd0e9f7d534ce9
acf8fccf9145538e 6b68addfb1520e45 a6eb8f9be694c65f 33b0500be7f395d5 dfaec48e3d792468 319f11c9fbe35834
10ccd3d46ca361e1 93f9965ef0e8478e 51f7f857239535cb a56c664ead10c5c9 bd48b628a8a17fa4 e80d89dc32e3c87e
c75c5735f70f104e 799c03cdaf778610 8bc4fe3837d5fcea b8852ec6a86af318 34dac999fdd30939 aff1f7038cb29694
f9f40011df6e771d c4464ae4011ca2fa d3e8f310bce08612 1bade281cb9e7499 573ad175daf8ab34 8e757075a67ae36c
20c6db6ce805da92 6ba2564af146e1c1 23263128ad436d11 82fdf1ef11d4c56e c86465d054f7f540 88ebea713ae828ca
144b7f25d2640e1f 86b1590a7846ac25 0792dda3f77cbec3 eb2d56ef31331ab7 c1c610f5dd3f3729 f29986b09ac63b11
0f0f64346ebac436 c9548239675b063d 528dffacc1bbdfdb 8fbab8136fa69f31 3a0158de14328afc f2f61bab7dfb42e4
abc0d7ca298f482d b22885a09b3f3f2c 3d09607dee7b2dfb 16f28d89f6202ffd a478f765c1bc904a 45d0968bb6cb708c
e4a786be7f0a853b 2a2456266a4fe02f 17a1614f89c29a0e a4c3f17894713423 463727ab5cfd8c0c 431c83bab723762e
f925f2e74edd1565 4c5535c4eeb58cce 200e6d921d2fba35 02ba800c6ed91b3d 60a4656b3b11639d be80ec92ae5d19e0
ac1343c0d288fb65 ab4e0aa1ea107353 4298fa448980855c 0e3fdeb9110e1ffb 6e7f121b54944b46 34ebdaa4855c9ff4
27f76d5cbd4cf3b5 7b27954842cb912a 7733fd31a22e7155 288290c46f4dafa2 04bf624a9f4b6654 78ee6afa312992ef
563a0131892dddcf bb230777cfd8281c 7f7fec10552d2e0a 714809b4f2d0133a a118fbcc188ebb60 495770f1536fa549
15f0dc9de685fa2e 8ea7f133618384af 115dc02eca7d21aa ec49bb087dbd23ce 70d9205162dec331 a8b0c08a1f6d1831
cbf75354128e9d47 72ad81b11a36b101 7cbab604e044f27a 26e6ca892b6d51f6 bcb95cc8855c96e2 4b7921a89fac43b2
4f0e7d4fa5cc9ab6 773ff654de32cd81 0f6fd95e03a78d48 a1f0c655181eaf92 e61aa7a735c1d2ff ac77fa5e244eec43
36cbe257d6476672 5dccbc32f38c512a 89fc3e60d88ec5b0 2736227bb518c4b1 055901c39f304cdb e2931eabb8da681b
e3fd0a29b4f12a16 7ca8ecb770732172 3715721733bcbd43 7831ff00afb975fe 1fa79e6e223565e0 ed08a5bf3736cfed
09f958f90160a4d6 1236f6df18ee5a30 7cb9c7c766a28c29 1da3ad9f5354789a 8e413597bc6444bf fa3be8523651a578
0103753ba3e27fa7 206cdf7028fdc7f7 500fdae0a895650b d4375305311504ee 0a0d7f277b11e9bf c21223f046e441ed
d40657ff8d963ef1 256701a938ff5edc 4377054818ca4f07 8cf6a98329376819 916bcd4a92823ae1 f6fdaa3310d4cf76
7ca3cd463cd168b1 b3650849cc366393 b1ed80b5209fdc34 82f4157f9fca54c3
""".split())


# --- small utilities -----------------------------------------------------------------------

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(text: str, n: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:n]


def digest(obj) -> str:
    """sha256 of an object's canonical JSON (job config hashes)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
                          .encode()).hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def order_hash(*parts) -> str:
    return hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).hexdigest()


def _mean(a) -> float:
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def atomic_write(path: str, write) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    atomic_write(path, lambda p: df.to_parquet(p, index=False))


def write_json(path: str, obj) -> None:
    def w(p):
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
    atomic_write(path, w)


def read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def f16(a) -> bytes:
    """A float sequence as float16 bytes (per-token logprobs and entropies)."""
    return np.asarray(a, np.float32).astype(np.float16).tobytes()


def unf16(b) -> np.ndarray:
    return np.frombuffer(b, np.float16).astype(np.float32)


# --- keys ------------------------------------------------------------------------------------

def _text(v) -> str:
    """paiec.predict._text."""
    if isinstance(v, str):
        return v
    return "" if v is None else str(v)


def _clean(v) -> str:
    """paiec.data._clean: NaN and None -> ''."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def item_key_hex(item: dict) -> str:
    """paiec.predict.item_key(item).hex(): blake2b-128 over the blake2b-128 of each
    of item_content, item_features, interactors, benchmark_id."""
    h = hashlib.blake2b(digest_size=16)
    for f in ITEM_FIELDS:
        h.update(hashlib.blake2b(_text(item.get(f)).encode("utf-8", "surrogatepass"), digest_size=16).digest())
    return h.hexdigest()


def text_key_hex(item: dict) -> str:
    """paiec.llmfeat.text_key: the key with benchmark_id blank."""
    return item_key_hex({**item, "benchmark_id": ""})


def content_sha256(item: dict) -> str:
    """HASH_DEF (experiments/strong_llm_eval.py): sha256 of paiec.llmfeat.item_text."""
    return sha(f"{_text(item.get('item_content'))}\n{_text(item.get('item_features'))}")


def head_tail(ids, max_len: int, marker=()) -> tuple[list, bool]:
    """paiec.llmfeat.head_tail: `ids` if they fit, else head + marker + tail in
    exactly max_len tokens (the head gets the odd one)."""
    ids, marker = list(ids), list(marker)
    if len(ids) <= max_len:
        return ids, False
    keep = max_len - len(marker)
    if keep < 2:
        raise ValueError(f"max_len {max_len} leaves no room around a {len(marker)}-token marker")
    head = (keep + 1) // 2
    return ids[:head] + marker + ids[len(ids) - (keep - head):], True


# --- data ------------------------------------------------------------------------------------

def data_path(data_dir: str, bench: str, table: str) -> str:
    return os.path.join(data_dir, bench, f"{table}.parquet")


def kaggle_hf_token():
    """The user's Hugging Face token: the Kaggle Secret HF_TOKEN on Kaggle; off
    Kaggle the HF_TOKEN variable or None (huggingface_hub then uses a cached login)."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return os.environ.get("HF_TOKEN") or None
    return UserSecretsClient().get_secret("HF_TOKEN")


def download_data(data_dir: str, benches, tables=TABLES, token_getter=None, downloader=None) -> int:
    """Fetch the missing core tables of measurement-db into data_dir. The token goes
    to hf_hub_download and nowhere else; errors are reported without it."""
    missing = [(b, t) for b in benches for t in tables if not os.path.exists(data_path(data_dir, b, t))]
    if not missing:
        return 0
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader
    token = (token_getter or kaggle_hf_token)()
    if not token and ON_KAGGLE:
        raise ConfigError("no Hugging Face token: add a Kaggle Secret named HF_TOKEN (Add-ons -> Secrets) "
                          "and attach it to this notebook; the dataset is gated")
    log(f"downloading {len(missing)} tables of {DATA_REPO} into {data_dir}")
    try:
        for b, t in missing:
            downloader(DATA_REPO, f"{b}/{t}.parquet", repo_type="dataset", local_dir=data_dir, token=token)
    except Exception as e:                                   # noqa: BLE001 - reported without the token
        msg = str(e).splitlines()[0][:300] if str(e) else ""
        if token:
            msg = msg.replace(token, "***")
        raise ConfigError(f"download failed: {type(e).__name__}: {msg} (accepted the dataset's terms on "
                          f"huggingface.co/datasets/{DATA_REPO} with the token's account?)") from None
    finally:
        token = None                                         # noqa: F841 - drop the reference
    return len(missing)


def read_items(data_dir: str, bench: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = data_path(data_dir, bench, "items")
    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in ("item_id", "benchmark_id", "content", "item_features", "grading_criterion",
                        "content_hash") if c in have]
    df = pd.read_parquet(path, columns=cols)
    for c in ("benchmark_id", "content", "item_features", "grading_criterion", "content_hash"):
        if c not in df.columns:
            df[c] = ""
    return df


def item_dict(row) -> dict:
    """The item as paiec.data.load_pairs builds it."""
    return {"item_content": _clean(row.content), "item_features": _clean(row.item_features),
            "interactors": "", "benchmark_id": _clean(row.benchmark_id)}


def rubric_units(data_dir: str, benches, limit: int | None = None) -> list[dict]:
    """The unique items of each benchmark (by predict.item_key, as
    experiments/llm_features.unique_items), each with every item_id it stands for;
    interleaved over benchmarks in a fixed hash order (limit: per benchmark)."""
    out = []
    for b in benches:
        it = read_items(data_dir, b)
        by = {}
        for row in it.itertuples(index=False):
            item = item_dict(row)
            k = item_key_hex(item)
            u = by.get(k)
            if u is None:
                u = by[k] = dict(unit=k, benchmark=b, key=k, text_key=text_key_hex(item),
                                 content=item["item_content"], features=item["item_features"],
                                 content_sha256=content_sha256(item), content_sha=sha(item["item_content"], 16),
                                 content_hash=_clean(row.content_hash), item_ids=[])
            u["item_ids"].append(str(row.item_id))
        us = sorted(by.values(), key=lambda u: order_hash(b, u["key"]))
        for u in us:
            u["item_ids"].sort()
        out += us[:limit] if limit else us
        del it, by
    return sorted(out, key=lambda u: order_hash("rubric", u["key"]))


def reference_answer(gc):
    if not isinstance(gc, str) or not gc:
        return None
    try:
        return json.loads(gc).get("reference_answer")
    except (ValueError, AttributeError):
        return None


def answer_format(a) -> str:
    """attempt_probe.select_items' fmt."""
    if a is None:
        return "none"
    a = str(a).strip()
    return "mcq" if re.fullmatch(r"[A-E]", a) else "integer" if re.fullmatch(r"-?\d+", a) else "expression"


def attempt_eligible(content: str, gold) -> bool:
    return (answer_format(gold) in ("integer", "expression") and not IMAGE_RE.search(content or "")
            and len((content or "").strip()) >= MIN_TEXT)


def competition(features: str) -> str:
    m = re.search(r"competition=([^;]+)", features or "")
    return m.group(1) if m else ""


def attempt_units(data_dir: str, scope: str = "all", limit: int | None = None,
                  probe_ids=PROBE_IDS) -> list[dict]:
    """matharena's eligible items, one unit per unique item_content (an attempt
    reads the content alone), probe units first, then the rest; each group
    round-robin over competitions in a fixed hash order. scope: 'all', 'probe'
    (the probe units only) or 'rest' (the others only)."""
    it = read_items(data_dir, "matharena")
    nsub = {}
    rp = data_path(data_dir, "matharena", "response")
    if os.path.exists(rp):
        r = pd.read_parquet(rp, columns=["subject_id", "item_id"])
        nsub = r.groupby("item_id").subject_id.nunique().to_dict()
        del r
    by = {}
    for row in it.itertuples(index=False):
        item = item_dict(row)
        gold = reference_answer(row.grading_criterion)
        if not attempt_eligible(item["item_content"], gold):
            continue
        cs = sha(item["item_content"], 16)
        u = by.setdefault(cs, dict(unit=cs, benchmark="matharena", content=item["item_content"],
                                   content_sha=cs, members=[], comps=set(), golds=set(), probe=False,
                                   n_subjects=0))
        iid = str(row.item_id)
        u["members"].append((iid, item_key_hex(item), text_key_hex(item), content_sha256(item),
                             _clean(row.content_hash)))
        u["comps"].add(competition(item["item_features"]))
        u["golds"].add(str(gold).strip())
        u["probe"] |= iid in probe_ids
        u["n_subjects"] = max(u["n_subjects"], int(nsub.get(row.item_id, 0)))
    units = []
    for u in by.values():
        u["members"].sort()
        u["item_ids"] = [m[0] for m in u["members"]]
        u["comps"] = sorted(u["comps"])
        u["golds"] = sorted(u["golds"])
        units.append(u)

    def round_robin(group):
        per = {}
        for u in sorted(group, key=lambda u: order_hash("attempt", u["unit"])):
            per.setdefault(u["comps"][0], []).append(u)
        comps = sorted(per, key=lambda c: order_hash("comp", c))
        return [per[c][i] for i in range(max(map(len, per.values()), default=0)) for c in comps
                if i < len(per[c])]

    probe = [] if scope == "rest" else round_robin([u for u in units if u["probe"]])
    rest = [] if scope == "probe" else round_robin([u for u in units if not u["probe"]])
    out = probe + rest
    return out[:limit] if limit else out


# --- tokenizer adapter and prompts ----------------------------------------------------------

class HFTok:
    """The interface the prompts need, over a Hugging Face tokenizer."""

    def __init__(self, tok):
        self.tok = tok
        ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
        for t in ("<|im_end|>", "<|endoftext|>"):
            try:
                i = tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
                    ids.add(i)
            except Exception:                                # noqa: BLE001
                pass
        self.eos_ids = ids

    def enc(self, text: str) -> list:
        return list(self.tok.encode(text, add_special_tokens=False))

    def dec(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def chat(self, messages, enable_thinking: bool) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=enable_thinking)


class ChatPrompt:
    """A chat prompt rendered once around placeholders; the pieces around the task
    (and the metadata) are tokenized separately and joined as ids, so the task can
    be cut by head_tail to an exact token budget (paiec.llmfeat.RatingPrompt)."""

    def __init__(self, tok, system, user, enable_thinking, prefill="", max_task_tokens=3072):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = tok.chat(msgs, enable_thinking) + prefill
        if text.count(TASK_PH) != 1 or text.count(META_PH) > 1:
            raise ValueError("chat template did not keep the placeholders")
        self.tok, self.rendered, self.max_task_tokens = tok, text, max_task_tokens
        pre, rest = text.split(TASK_PH)
        mid, post = rest.split(META_PH) if META_PH in rest else (rest, None)
        self.pre, self.mid = tok.enc(pre), tok.enc(mid)
        self.post = None if post is None else tok.enc(post)
        self.marker = tok.enc(MARKER)

    def build(self, task: str, meta: str = ""):
        ids = self.tok.enc(task)
        cut, trunc = head_tail(ids, self.max_task_tokens, self.marker)
        out = self.pre + cut + self.mid
        if self.post is not None:
            m, _ = head_tail(self.tok.enc(meta), META_MAX_TOKENS, self.marker) if meta else ([], False)
            out = out + m + self.post
        return out, {"task_tokens": len(ids), "truncated": bool(trunc)}

    def shared_prefix(self) -> int:
        return len(self.pre)


def rubric_features(features: str) -> str:
    """item_features without position fields (META_DROP_RE), as `k=v;k=v`."""
    keep = []
    for part in (features or "").split(";"):
        k = part.split("=", 1)[0].strip() if "=" in part else ""
        if k and META_DROP_RE.search(k):
            continue
        keep.append(part)
    return ";".join(keep).strip().strip(";")


def rubric_meta(features: str, with_meta: bool = True) -> str:
    f = rubric_features(features) if with_meta else ""
    return f"Metadata: {f}\n" if f.strip() else ""


def rubric_prompt(tok, max_task_tokens):
    return ChatPrompt(tok, SYSTEM_RUBRIC, USER_RUBRIC, enable_thinking=False, prefill=RUBRIC_PREFILL,
                      max_task_tokens=max_task_tokens)


def attempt_prompt(tok, max_task_tokens):
    return ChatPrompt(tok, None, TASK_PH, enable_thinking=True, max_task_tokens=max_task_tokens)


def attempt_task(content: str) -> str:
    return content if "\\boxed" in content else f"{content}\n\n{BOXED_INSTRUCTION}"


# --- generation --------------------------------------------------------------------------------

@dataclasses.dataclass
class GenParams:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 64
    logprobs: int = 0
    stop: tuple = ()
    keep_top: bool = False       # keep the top-k alternatives per position (the rubric readout)
    record_raw: bool = False     # record the raw distribution's entropy and log-prob (RawRecorder)


@dataclasses.dataclass
class Sample:
    """One completion: token ids and their decoded strings (their concatenation is
    the text every parser reads), the engine's logprob of each sampled token and
    its top-k entropy bound, the finish reason, (keep_top) the top-k alternatives
    as [(decoded string, logprob)] per position, and (record_raw, where the engine
    allows it) the raw model distribution's log-prob of each sampled token and
    full-vocabulary entropy. engine_logprobs says what lp / ent are: 'raw' (a
    greedy request, or vLLM V1) or 'processed' (V0 after penalties, temperature,
    top-k/top-p)."""
    token_ids: list
    token_strs: list
    lp: np.ndarray
    ent: np.ndarray
    finish_reason: str
    top: list | None = None
    lp_raw: np.ndarray | None = None
    ent_raw: np.ndarray | None = None
    engine_logprobs: str = "processed"

    @property
    def text(self) -> str:
        return "".join(self.token_strs)

    @property
    def stats_source(self) -> str:
        if self.lp_raw is not None and self.ent_raw is not None:
            return STATS_RAW
        return STATS_RAW_TOPK if self.engine_logprobs == "raw" else STATS_PROC

    def stats(self) -> tuple:
        """(per-token log-prob, per-token entropy, source): the raw distribution's
        where recorded, else the engine's top-k."""
        if self.stats_source == STATS_RAW:
            return np.asarray(self.lp_raw, np.float64), np.asarray(self.ent_raw, np.float64), STATS_RAW
        return np.asarray(self.lp, np.float64), np.asarray(self.ent, np.float64), self.stats_source


class RawRecorder:
    """vLLM V0 per-request logits processor (SamplingParams.logits_processors):
    per generated token, the full-vocabulary entropy of the raw next-token
    distribution and the raw log-prob of the token sampled from it, as
    experiments/attempt_probe.Recorder. V0 calls it on the driver (in this
    process) with the raw logits row, before penalties, temperature and
    top-k/top-p, and the sequence's output tokens so far; the row at step t (t =
    len(past)) gives entropy[t], and the log-softmax kept from step t-1 gives the
    log-prob of past[-1]. Indexing by len(past) makes a recompute after preemption
    and V0's extra step after a stop (asynchronous output processing) harmless.
    Values stay on the GPU in preallocated buffers (no per-step synchronisation)
    and are read once by finish(). One recorder per request of n = 1; no clone():
    vLLM's defensive copy of the SamplingParams keeps this object."""

    def __init__(self, cap: int):
        self.cap, self.ent, self.lp, self.prev, self.prev_t, self.calls = int(cap), None, None, None, -2, 0

    def __call__(self, past_ids, logits):
        import torch
        t = len(past_ids)
        ls = torch.log_softmax(logits.float(), -1)               # a new tensor: logits is left as it is
        if self.ent is None:
            self.ent = torch.full((max(self.cap, t + 2),), float("nan"), dtype=torch.float32, device=ls.device)
            self.lp = torch.full_like(self.ent, float("nan"))
        if t + 1 >= self.ent.numel():
            pad = torch.full((self.ent.numel() + 256,), float("nan"), dtype=torch.float32, device=ls.device)
            self.ent, self.lp = torch.cat([self.ent, pad]), torch.cat([self.lp, pad.clone()])
        if t > 0 and self.prev_t == t - 1:
            self.lp[t - 1] = self.prev[int(past_ids[-1])]
        self.ent[t] = torch.special.entr(ls.exp()).sum()
        self.prev, self.prev_t = ls, t
        self.calls += 1
        return logits

    def finish(self, token_ids):
        """(lp, ent) float32 arrays over the n generated tokens, or (None, None) if
        the processor never ran."""
        import torch
        n = len(token_ids)
        if self.ent is None or not n or n > self.ent.numel():
            return None, None
        # V0 runs the model, and so __call__, under inference_mode: the buffers are inference tensors, which
        # may only be updated in place inside it
        with torch.inference_mode():
            if self.prev is not None and self.prev_t == n - 1:
                self.lp[n - 1] = self.prev[int(token_ids[-1])]
            lp = self.lp[:n].float().cpu().numpy().astype(np.float32)
            ent = self.ent[:n].float().cpu().numpy().astype(np.float32)
        self.ent = self.lp = self.prev = None
        return lp, ent


def topk_entropy(lps) -> float:
    """Entropy (nats) of a next-token distribution known through its top-k
    logprobs, the remaining mass taken as one outcome: a lower bound."""
    lp = np.asarray([x for x in lps if x is not None and np.isfinite(x)], np.float64)
    if not lp.size:
        return float("nan")
    p = np.exp(np.minimum(lp, 0.0))
    h = float(-(p * np.log(np.maximum(p, 1e-300))).sum())
    r = 1.0 - float(p.sum())
    if r > 1e-9:
        h -= r * math.log(r)
    return h


def _sample_from_vllm(co, tok, k: int, keep_top: bool, recorder: RawRecorder | None = None,
                      engine_logprobs: str = "processed") -> Sample:
    ids = list(co.token_ids)
    lps = co.logprobs
    n = len(ids)
    strs, top = [], ([] if keep_top else None)
    lp = np.full(n, np.nan, np.float32)
    ent = np.full(n, np.nan, np.float32)
    for t, tid in enumerate(ids):
        d = lps[t] if (lps is not None and t < len(lps)) else None
        s = None
        alts = []
        if d:
            ch = d.get(tid)
            if ch is not None:
                lp[t] = ch.logprob
                s = ch.decoded_token
            for key, v in d.items():
                rank = getattr(v, "rank", None)
                if rank is not None and rank > k:
                    continue                         # the sampled token, outside the top-k
                alts.append((v.decoded_token if v.decoded_token is not None else tok.dec([key]),
                             float(v.logprob)))
            ent[t] = topk_entropy([a[1] for a in alts])
        if s is None:
            s = tok.dec([tid])
        strs.append(s)
        if keep_top:
            top.append(alts)
    lp_raw, ent_raw = recorder.finish(ids) if recorder is not None else (None, None)
    return Sample(ids, strs, lp, ent, str(co.finish_reason), top, lp_raw, ent_raw, engine_logprobs)


class VLLMBackend:
    """vLLM's offline LLM, fp16, AWQ, tensor parallel; prompts as token ids."""

    def __init__(self, model, revision, quantization, tp, max_model_len, gpu_mem, max_num_seqs,
                 enforce_eager, prefix_caching, seed=0):
        from importlib.metadata import version
        vv = version("vllm")
        major, minor = (int(x) for x in re.findall(r"\d+", vv)[:2])
        if (major, minor) < (0, 11):
            os.environ.setdefault("VLLM_USE_V1", "0")    # V0: the engine that runs on Turing
        from vllm import LLM
        kw = dict(model=model, revision=revision, tokenizer_revision=revision, dtype="float16",
                  tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                  max_num_seqs=max_num_seqs, enforce_eager=enforce_eager, enable_prefix_caching=prefix_caching,
                  disable_custom_all_reduce=True, seed=seed, max_logprobs=20, trust_remote_code=False)
        if tp > 1:
            kw["distributed_executor_backend"] = "mp"   # V0: the driver worker (and RawRecorder) in this process
        if quantization:
            kw["quantization"] = quantization
        self.llm = LLM(**kw)
        self.tok = HFTok(self.llm.get_tokenizer())
        eng = type(self.llm.llm_engine).__module__
        self.v1 = eng.startswith("vllm.v1")
        import torch
        self.info = {"backend": "vllm", "engine": f"vllm {vv} ({'V1' if self.v1 else 'V0'})", "engine_module": eng,
                     "torch": torch.__version__, "raw_recorder": not self.v1,
                     "logprobs": ("engine logprobs: raw model distribution (V1); no per-request logits processors, "
                                  "attempt entropy is the top-5 lower bound" if self.v1 else
                                  "engine logprobs: processed (V0), after penalties, temperature and top-k/top-p, "
                                  "raw for greedy requests; attempts and forced readouts carry RawRecorder "
                                  "(raw full-vocabulary entropy and log-prob)"),
                     "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

    def sampling_kwargs(self, params: GenParams, seed=None, recorder=None) -> dict:
        kw = dict(n=params.n, temperature=params.temperature, max_tokens=params.max_tokens)
        if params.temperature > 0:
            kw.update(top_p=params.top_p, top_k=params.top_k, min_p=params.min_p)
        if params.presence_penalty:
            kw["presence_penalty"] = params.presence_penalty
        if params.logprobs:
            kw["logprobs"] = params.logprobs
        if seed is not None:
            kw["seed"] = int(seed)
        if params.stop:
            kw["stop"] = list(params.stop)
        if recorder is not None:
            kw["logits_processors"] = [recorder]
        return kw

    def generate(self, prompts, params: GenParams, seeds=None) -> list[list[Sample]]:
        from vllm import SamplingParams
        use_rec = params.record_raw and not self.v1 and params.n == 1
        recs = [RawRecorder(params.max_tokens + 8) if use_rec else None for _ in prompts]
        sps = [SamplingParams(**self.sampling_kwargs(params, None if seeds is None else seeds[i], recs[i]))
               for i in range(len(prompts))]
        outs = self.llm.generate([{"prompt_token_ids": list(p)} for p in prompts], sps, use_tqdm=False)
        kind = "raw" if (self.v1 or (params.temperature <= 0 and not params.presence_penalty)) else "processed"
        return [[_sample_from_vllm(co, self.tok, params.logprobs, params.keep_top, rec, kind) for co in ro.outputs]
                for ro, rec in zip(outs, recs)]


class HFBackend:
    """Fallback without vLLM, for the rubric job only: transformers, the
    unquantised model in fp16 over every GPU (device_map="auto"), greedy, top-k
    logprobs from the raw logits. Sampling k long attempts this way is too slow."""

    def __init__(self, model, revision, max_batch_tokens=24000, max_batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, revision=revision)
        tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, torch_dtype=torch.float16,
                                                          device_map="auto").eval()
        self.tok = HFTok(tok)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.max_batch_tokens, self.max_batch = max_batch_tokens, max_batch
        import transformers
        self.info = {"backend": "hf", "engine": f"transformers {transformers.__version__}",
                     "torch": torch.__version__, "logprobs": "raw model distribution",
                     "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

    def generate(self, prompts, params: GenParams, seeds=None) -> list[list[Sample]]:
        import torch
        if params.n != 1 or params.temperature > 0:
            raise NotImplementedError("the transformers fallback only runs greedy single answers (rubric)")
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        i = 0
        while i < len(order):
            j = i + 1
            while (j < len(order) and j - i < self.max_batch
                   and (j - i + 1) * (len(prompts[order[j]]) + params.max_tokens) <= self.max_batch_tokens):
                j += 1
            idx = order[i:j]
            L = max(len(prompts[q]) for q in idx)
            ids = torch.full((len(idx), L), self.pad, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, q in enumerate(idx):
                ids[r, L - len(prompts[q]):] = torch.tensor(prompts[q])
                mask[r, L - len(prompts[q]):] = 1
            dev = self.model.get_input_embeddings().weight.device
            with torch.inference_mode():
                g = self.model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), do_sample=False,
                                        max_new_tokens=params.max_tokens, pad_token_id=self.pad,
                                        output_logits=True, return_dict_in_generate=True)
            gen = g.sequences[:, L:].cpu()
            k = max(params.logprobs, 1)
            tvs, tis, chs = [], [], []
            for t, x in enumerate(g.logits):                          # one (B, V) step at a time
                ls = torch.log_softmax(x.float(), -1)
                top = ls.topk(k, -1)
                tvs.append(top.values.cpu())
                tis.append(top.indices.cpu())
                chs.append(ls.gather(1, gen[:, t:t + 1].to(ls.device))[:, 0].cpu())
            tv, ti, chosen = torch.stack(tvs, 1), torch.stack(tis, 1), torch.stack(chs, 1)
            del g
            for r, q in enumerate(idx):
                toks = gen[r].tolist()
                n = next((t for t, x in enumerate(toks) if x in self.tok.eos_ids), len(toks))
                toks = toks[:n + 1] if n < len(toks) else toks
                strs, prev = [], ""
                for t in range(len(toks)):
                    cur = self.tok.dec(toks[:t + 1])
                    strs.append(cur[len(prev):] if cur.startswith(prev) else self.tok.dec([toks[t]]))
                    prev = cur
                text = "".join(strs)
                cut = len(toks)
                for s in params.stop:                                      # cut after the stop string
                    c = text.find(s)
                    if c >= 0:
                        ends = np.cumsum([len(x) for x in strs])
                        cut = min(cut, int(np.searchsorted(ends, c, side="right")) + 1)
                toks, strs = toks[:cut], strs[:cut]
                lp = chosen[r, :len(toks)].numpy().astype(np.float32)
                alts = [[(self.tok.dec([int(a)]), float(v)) for a, v in zip(ti[r, t].tolist(), tv[r, t].tolist())]
                        for t in range(len(toks))]
                ent = np.array([topk_entropy([v for _, v in a]) for a in alts], np.float32)
                fin = "stop" if (cut < len(gen[r]) or n < len(gen[r].tolist())) else "length"
                out[q] = [Sample(toks, strs, lp, ent, fin, alts if params.keep_top else None, engine_logprobs="raw")]
            i = j
        return out


# --- readouts ------------------------------------------------------------------------------------

def digit_of(s: str):
    s = (s or "").strip()
    return int(s) if len(s) == 1 and s in "0123456789" else None


def read_rubric(sample: Sample, prefill: str = RUBRIC_PREFILL) -> dict:
    """Each key's digit distribution at the position of its generated value: the
    top-k alternatives there that are single allowed digits, renormalised. Returns
    per key the expected value, the generated digit, the entropy, the mass the
    digits held (of the whole next-token distribution) and the probabilities."""
    strs = sample.token_strs
    ends = np.cumsum([len(s) for s in strs]) if strs else np.zeros(0)
    full = prefill + "".join(strs)
    out, ok = {}, True
    for key in RUBRIC_KEYS:
        R = RANGES[key]
        m = re.search(r'"%s"\s*:\s*' % re.escape(key), full)
        c = m.end() if m else -1
        if not m or c >= len(full) or full[c] not in "0123456789" or c < len(prefill):
            out.update({key: np.nan, f"{key}_mode": np.nan, f"{key}_entropy": np.nan, f"{key}_mass": np.nan,
                        f"{key}_probs": [np.nan] * R})
            ok = False
            continue
        t = int(np.searchsorted(ends, c - len(prefill), side="right"))
        chosen = int(full[c])
        p = np.zeros(R)
        for s, v in ((sample.top[t] if sample.top and t < len(sample.top) else []) or []):
            d = digit_of(s)
            if d is not None and d < R and v is not None and np.isfinite(v):
                p[d] += math.exp(min(float(v), 0.0))
        mass = float(p.sum())
        if mass <= 0 and chosen < R:
            p[chosen] = 1.0
        probs = p / p.sum() if p.sum() > 0 else np.full(R, np.nan)
        ok &= chosen < R
        nz = probs[probs > 0]
        out.update({key: float(probs @ np.arange(R)), f"{key}_mode": float(chosen),
                    f"{key}_entropy": float(-(nz * np.log(nz)).sum()), f"{key}_mass": mass,
                    f"{key}_probs": probs.tolist()})
    ht = np.asarray(out["human_time_probs"], float)
    out["time_log_minutes"] = float(ht @ np.log(HUMAN_MID)) if np.isfinite(ht).all() else np.nan
    df = np.asarray(out["difficulty_probs"], float)
    out["solve_share"] = float(1.0 - df @ ((np.arange(10) + 0.5) / 10)) if np.isfinite(df).all() else np.nan
    out["parse_ok"] = bool(ok)
    return out


def boxed(text: str, start=None):
    """Content of the last \\boxed{...} (or of the brace opened at `start`), and
    its char span; (None, None) if absent or unclosed (attempt_probe.boxed)."""
    if start is None:
        i = text.rfind("\\boxed{")
        if i < 0:
            return None, None
        start = i + 7
    j, depth = start, 1
    while j < len(text):
        depth += (text[j] == "{") - (text[j] == "}")
        if depth == 0:
            return text[start:j], (start, j)
        j += 1
    return None, None


_SAFE = {"sqrt", "pi", "log", "binomial", "oo", "exp", "sin", "cos", "tan", "factorial"}
_TEXT = re.compile(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\{([^{}]*)\}")


def _latex_to_py(s):
    s = s.replace("\\left", "").replace("\\right", "").replace("\\displaystyle", "")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    for _ in range(6):
        s2 = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s2 = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", s2)
        s2 = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", s2)
        s2 = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s2)
        s2 = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s2)
        s2 = re.sub(r"\\binom\{([^{}]*)\}\{([^{}]*)\}", r"binomial(\1,\2)", s2)
        if s2 == s:
            break
        s = s2
    return (s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
            .replace("\\ln", "log").replace("\\log", "log").replace("\\infty", "oo")
            .replace("^", "**").replace("{", "(").replace("}", ")"))


def canon(a):
    """experiments/attempt_probe.canon: numbers by value (6 significant digits),
    anything else as a cleaned string or sympy's expanded form."""
    if a is None:
        return None
    s = str(a).strip().strip("$").strip()
    s = _TEXT.sub(r"\1", s)
    for t in ("\\,", "\\;", "\\!", "\\ ", "~", " ", "\\%", "%", "^{\\circ}", "^\\circ", "\\circ"):
        s = s.replace(t, "")
    s = s.rstrip(".").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    if not s:
        return None
    if re.fullmatch(r"[-+]?\d+", s):
        return "num:%.6g" % float(int(s)) if len(s) < 16 else "int:" + s.lstrip("+").lstrip("0")
    py = _latex_to_py(s)
    if (len(py) > 160 or not re.fullmatch(r"[0-9A-Za-z+\-*/().,]*", py)
            or py.count("**") > 2 or re.search(r"\*\*\(*\d{4,}", py)
            or re.search(r"factorial\(\d{5,}", py)
            or any(w not in _SAFE and not re.fullmatch(r"[A-Za-z]\d*", w)
                   for w in re.findall(r"[A-Za-z_]\w*", py))):
        return "str:" + s
    import signal
    use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()

    def _timeout(*_):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _timeout) if use_alarm else None
    if use_alarm:
        signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        import sympy
        from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                                parse_expr, standard_transformations)
        e = parse_expr(py, evaluate=True, transformations=standard_transformations
                       + (implicit_multiplication_application, convert_xor))
        if not e.free_symbols:
            v = complex(e.evalf(15))
            if abs(v.imag) < 1e-12 and math.isfinite(v.real):
                return "num:%.6g" % v.real
        return "sym:" + str(sympy.expand(e))
    except BaseException as exc:                             # noqa: BLE001 - parse errors, overflow, alarm
        if isinstance(exc, KeyboardInterrupt):
            raise
        return "str:" + s
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def _span_mean(arr, ends, c0, c1) -> float:
    """Mean of a per-token array over the tokens covering chars [c0, c1)."""
    if c1 <= c0 or not len(ends):
        return float("nan")
    t0 = int(np.searchsorted(ends, c0, side="right"))
    t1 = int(np.searchsorted(ends, c1 - 1, side="right")) + 1
    return _mean(arr[t0:t1])


def think_split(s: Sample) -> tuple:
    """(closed, n_think, char offset after </think>): n_think counts the tokens
    through the one that completes </think>, or all tokens if it never closed."""
    full = s.text
    c = full.find(THINK_END)
    if c < 0:
        return False, len(s.token_ids), len(full)
    ends = np.cumsum([len(x) for x in s.token_strs])
    return True, int(np.searchsorted(ends, c + len(THINK_END) - 1, side="right")) + 1, c + len(THINK_END)


def forced_prompt(prompt, s: Sample, eos=(), enc_open=(), enc_closed=()) -> list:
    """The forced readout's prompt: the attempt's prompt, its reasoning (through
    </think>; all of it, less a trailing end-of-turn, when the thinking did not
    close) and the Final Answer opener, so every attempt's answer is read the same
    way whether or not it boxed one itself."""
    closed, n_think, _ = think_split(s)
    body = list(s.token_ids[:n_think])
    if not closed:
        while body and body[-1] in eos:
            body.pop()
    return list(prompt) + body + list(enc_closed if closed else enc_open)


def recorder_gap(s: Sample | None) -> tuple:
    """(max |raw - engine| log-prob, tokens compared) on a greedy sample, whose
    engine logprobs are raw: the recorder's alignment check."""
    if s is None or s.lp_raw is None or s.engine_logprobs != "raw":
        return float("nan"), 0
    a, b = np.asarray(s.lp_raw, np.float64), np.asarray(s.lp, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return (float(np.abs(a[m] - b[m]).max()) if m.any() else float("nan")), int(m.sum())


def analyse_attempt(s: Sample, forced: Sample | None = None) -> dict:
    """One attempt: where the thinking ends, the natural \\boxed{} answer after it,
    the last boxed answer inside the thinking, the forced answer, and token
    statistics (the raw distribution's where recorded: Sample.stats) over
    everything, the reasoning, the answer and the first tokens. lp_answer is the
    forced readout's mean answer log-prob whenever forced readouts ran (for every
    attempt), else the natural boxed answer's."""
    strs = s.token_strs
    full = "".join(strs)
    ends = np.cumsum([len(x) for x in strs]) if strs else np.zeros(0, int)
    lp, ent, source = s.stats()
    lp_e, ent_e = np.asarray(s.lp, np.float64), np.asarray(s.ent, np.float64)
    n = len(strs)
    closed, n_think, a0 = think_split(s)
    nat, span = boxed(full[a0:])
    lp_boxed = _span_mean(lp, ends, a0 + span[0], a0 + span[1]) if span else float("nan")
    think_ans, _ = boxed(full[:a0 - len(THINK_END)] if closed else full)
    ans_forced, lp_forced, fsrc = None, float("nan"), None
    if forced is not None:
        ans_forced, fspan = boxed("\\boxed{" + forced.text, start=7)
        flp, _, fsrc = forced.stats()
        if fspan:
            fe = np.cumsum([len(x) for x in forced.token_strs])
            lp_forced = _span_mean(flp, fe, 0, max(fspan[1] - 7, 1))
    gap, gap_n = recorder_gap(forced)
    final = nat if nat is not None else ans_forced
    return dict(
        n_tokens=n, finish_reason=s.finish_reason, capped=s.finish_reason == "length", closed=closed,
        n_think=int(n_think), ans_natural=nat, ans_think=think_ans, ans_forced=ans_forced, answer=final,
        forced=nat is None and forced is not None,
        refuse=bool(REFUSE_RE.search(full[a0:] if closed else full[-2000:])),
        lp_answer=lp_forced if forced is not None else lp_boxed, lp_forced=lp_forced,
        tok_lp=_mean(lp), tok_entropy=_mean(ent),
        tok_lp_think=_mean(lp[:n_think]), tok_entropy_think=_mean(ent[:n_think]),
        tok_lp_answer=_mean(lp[n_think:]), tok_entropy_answer=_mean(ent[n_think:]),
        ent_first256=_mean(ent[:256]), ent_first1024=_mean(ent[:1024]),
        lp_first256=_mean(lp[:256]), lp_first1024=_mean(lp[:1024]), lp_boxed=lp_boxed,
        stats_source=source, forced_source=fsrc, recorder_gap=gap, recorder_gap_n=gap_n,
        tok_lp_engine=_mean(lp_e), tok_entropy_engine=_mean(ent_e), engine_logprobs=s.engine_logprobs,
        text=full, forced_text=forced.text if forced is not None else None,
        lp_seq=f16(lp), ent_seq=f16(ent), lp_engine_seq=f16(lp_e), ent_engine_seq=f16(ent_e))


def degenerate(text) -> bool:
    """Garbage such as fp16 overflow's '!!!!...': one character is most of a long text."""
    s = re.sub(r"\s+", "", text or "")
    return len(s) >= 64 and Counter(s).most_common(1)[0][1] > 0.5 * len(s)


def canon_sha(canonical) -> str | None:
    """What the samples keep of an attempt's canonical answer: a hash (answers agree
    iff their hashes do), not the normal form itself."""
    return None if canonical is None else sha(canonical, 16)


def attempt_aggregate(recs: list[dict], golds) -> dict:
    """Item features from k attempts (attempt_probe.item_features' definitions,
    plus the thinking and span statistics). REF_DIAG (graded, top_correct,
    graded_natural, per-attempt correct) and `canons` are for the log only:
    run_attempts pops them before anything is written."""
    k = len(recs)
    can = [canon(r["answer"]) for r in recs]
    cnt = Counter(x for x in can if x is not None)
    top, topn = cnt.most_common(1)[0] if cnt else (None, 0)
    cats = list(cnt.values()) + [1] * (k - sum(cnt.values()))
    p = np.array(cats, float) / k
    nat = Counter(x for x in (canon(r["ans_natural"]) for r in recs) if x is not None)
    gold = {canon(g) for g in golds} - {None}
    correct = [x is not None and x in gold for x in can]
    return dict(
        k=k, top_share=topn / k, top_share_natural=(nat.most_common(1)[0][1] / k) if nat else 0.0,
        ans_entropy=float(-(p * np.log(p)).sum()), n_distinct=len(cnt),
        fail_rate=float(np.mean([(x is None) or r["refuse"] for r, x in zip(recs, can)])),
        trunc_rate=float(np.mean([r["capped"] for r in recs])),
        closed_rate=float(np.mean([r["closed"] for r in recs])),
        forced_rate=float(np.mean([r["forced"] for r in recs])),
        mean_len=float(np.mean([r["n_tokens"] for r in recs])),
        mean_think_len=float(np.mean([r["n_think"] for r in recs])),
        **{f: _mean([r[f] for r in recs]) for f in
           ("tok_lp", "tok_entropy", "tok_lp_think", "tok_entropy_think", "tok_lp_answer", "tok_entropy_answer",
            "ent_first256", "ent_first1024", "lp_first256", "lp_first1024", "lp_boxed", "lp_answer",
            "tok_lp_engine", "tok_entropy_engine")},
        stats_source=",".join(sorted({r["stats_source"] for r in recs})),
        graded=float(np.mean(correct)) if gold else np.nan,
        graded_natural=float(np.mean([canon(r["ans_natural"]) in gold for r in recs
                                      if r["ans_natural"] is not None])) if gold and nat else np.nan,
        top_correct=float(top is not None and top in gold) if gold else np.nan,
        correct=correct, canons=can)


# --- shards, restore, manifest ---------------------------------------------------------------------

#: Kaggle saves at most 500 files of a notebook's output, mounts at most 500 of an output attached as an
#: input, and `kaggle kernels output` lists at most 500: the store keeps its files few. SEG_BYTES: a run
#: process's open segment is rewritten after each shard until it reaches this size, then a new one starts;
#: COMPACT_BYTES: the size of a compacted file; SEGS_PER_KIND: compact a job's items or samples at the start
#: and end of `run` when they hold more files than this; MAX_FILES: compact every store before a shard once
#: the saved output holds this many entries (files and directories); STOP_FILES: if compaction cannot bring
#: it below this (entries that are not the script's), stop the run (exit 77) rather than lose files.
SEG_BYTES = 48 << 20
COMPACT_BYTES = 1 << 30
SEGS_PER_KIND = 8
MAX_FILES, STOP_FILES, KAGGLE_FILE_CAP = 200, 400, 500
KAGGLE_WORKING = "/kaggle/working"
#: a store file's name: the k1.2 layout's items_NNNNN.parquet (a file per shard) or a segment
#: items-<run>-<nnn>.parquet
SEG_NAME_RE = re.compile(r"^(items|samples)(?:_\d{5}|-[a-z0-9]+-\d{3,})\.parquet$")
LEGACY_NAME_RE = re.compile(r"^(items|samples)_\d{5}\.parquet$")
SHARD_RE = re.compile(r"^[^/]+/(rubric|attempts)/((items|samples)(?:_\d{5}|-[a-z0-9]+-\d{3,})\.parquet)$")
#: columns that, next to the model's answers, would reveal the reference answer: never written, and dropped
#: from anything read back (a k1.2 store wrote them)
LEAK_COLS = ("correct", "canon", "graded", "graded_natural", "top_correct")


def new_run_id() -> str:
    """A run process's id in its segment names: time, then random (lowercase hex)."""
    return time.strftime("%y%m%d%H%M%S") + secrets.token_hex(3)


def count_entries(root: str) -> int:
    """Files and directories under root (what a cap on output items counts, conservatively)."""
    return sum(len(fs) + len(ds) for _, ds, fs in os.walk(root))


def saved_root(out_root: str) -> str:
    """The directory Kaggle saves as the notebook's output: /kaggle/working when --out is in it."""
    a = os.path.abspath(out_root)
    return KAGGLE_WORKING if ON_KAGGLE and (a + os.sep).startswith(KAGGLE_WORKING + os.sep) else a


class Store:
    """One job's units under root, in segments: items-<run>-<nnn>.parquet (one row
    per unit, with the job config hash `cfg` and the time `t`) and, for attempts,
    samples-<run>-<nnn>.parquet (one row per attempt, with `t`), each shard's
    samples written before its items. A unit is done when an items row of this cfg
    holds it. Each write rewrites the process's open segment atomically (tmp +
    os.replace: a crash leaves the previous version) and rotates it at SEG_BYTES;
    read() takes every segment and the k1.2 layout's items_NNNNN.parquet, keeps
    the latest row per unit (per attempt) and drops LEAK_COLS."""

    def __init__(self, root: str, cfg: str, run: str | None = None):
        self.root, self.cfg, self.run = root, cfg, run or new_run_id()
        self.open, self.ncompact = {}, 0

    def files(self, kind: str = "items") -> list:
        return sorted(p for p in glob.glob(os.path.join(self.root, f"{kind}*.parquet"))
                      if (m := SEG_NAME_RE.match(os.path.basename(p))) and m.group(1) == kind)

    def legacy(self) -> bool:
        return any(LEGACY_NAME_RE.match(os.path.basename(p)) for k in ("items", "samples") for p in self.files(k))

    def needs_compaction(self) -> bool:
        return self.legacy() or any(len(self.files(k)) > SEGS_PER_KIND for k in ("items", "samples"))

    def done(self) -> set:
        out = set()
        for p in self.files("items"):
            df = pd.read_parquet(p, columns=["unit", "cfg"])
            out |= set(df.loc[df.cfg == self.cfg, "unit"])
        return out

    def _append(self, kind: str, df: pd.DataFrame) -> None:
        seg = self.open.get(kind)
        if seg is None:
            os.makedirs(self.root, exist_ok=True)
            taken = {os.path.basename(p) for p in self.files(kind)}
            n = next(i for i in range(10 ** 6) if f"{kind}-{self.run}-{i:03d}.parquet" not in taken)
            seg = self.open[kind] = [os.path.join(self.root, f"{kind}-{self.run}-{n:03d}.parquet"), []]
        seg[1].append(df)
        write_parquet(pd.concat(seg[1], ignore_index=True) if len(seg[1]) > 1 else df, seg[0])
        if os.path.getsize(seg[0]) >= SEG_BYTES:
            del self.open[kind]                              # the next shard starts a new segment

    def write(self, items: pd.DataFrame, samples: pd.DataFrame | None = None) -> None:
        if samples is not None:
            self._append("samples", samples)
        self._append("items", items)

    def read(self, kind: str = "items", cfg: str | None = None) -> pd.DataFrame:
        parts = [pd.read_parquet(p) for p in self.files(kind)]
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])
        if cfg is not None:
            df = df[df.cfg == cfg]
        t = pd.to_numeric(df["t"], errors="coerce").fillna(0.0).to_numpy() if "t" in df.columns \
            else np.zeros(len(df))
        key = ["unit", "cfg"] if kind == "items" else ["unit", "cfg", "j"]
        # a unit done twice (an input attached twice, a stale input) or an attempt's samples written before a
        # crash and again by the redo: the latest rows win, the ones its items row was written with
        df = df.iloc[np.argsort(t, kind="stable")].drop_duplicates(key, keep="last")
        return df.sort_index().reset_index(drop=True)

    def compact(self) -> list:
        """Merge every file of each kind into as few as COMPACT_BYTES allows,
        deduplicated and without LEAK_COLS. The merged files are written before
        the old ones go, so a crash in between leaves duplicates, which read()
        drops. Returns the names merged away."""
        self.open.clear()
        gone = []
        for kind in ("samples", "items"):
            old = self.files(kind)
            if not old:
                continue
            df = self.read(kind)
            self.ncompact += 1
            new = []
            if len(df):
                n = max(1, math.ceil(float(df.memory_usage(deep=True).sum()) / COMPACT_BYTES))
                for i, idx in enumerate(np.array_split(np.arange(len(df)), n)):
                    path = os.path.join(self.root, f"{kind}-c{self.run}{self.ncompact}-{i:03d}.parquet")
                    write_parquet(df.iloc[idx].reset_index(drop=True), path)
                    new.append(path)
            for p in old:
                if p not in new:
                    os.remove(p)
                    gone.append(os.path.basename(p))
        return gone


def job_dirs(out_root: str) -> list:
    return sorted(d for d in glob.glob(os.path.join(out_root, "*", "*"))
                  if os.path.isdir(d) and os.path.basename(d) in ("rubric", "attempts"))


def note_absorbed(out_root: str, names) -> None:
    """Record segment files merged away (relative to out_root), so a later restore
    skips them in an input that still has them."""
    if names:
        path = os.path.join(out_root, "manifest.json")
        man = read_json(path)
        man["absorbed"] = sorted(set(man.get("absorbed") or []) | set(names))
        write_json(path, man)


def compact_all(out_root: str, stores=(), run: str | None = None, only_needed: bool = False) -> int:
    """Compact every job's store under out_root (the live stores in `stores` in
    place, so their next write opens a new segment); only_needed: only stores in
    the k1.2 layout or with more than SEGS_PER_KIND files of a kind. Leftover
    .tmp files of interrupted writes go too. Returns the files merged away."""
    live = {os.path.abspath(s.root): s for s in stores}
    run = run or new_run_id()
    gone = []
    for d in job_dirs(out_root):
        for p in glob.glob(os.path.join(d, "*.tmp")):
            os.remove(p)
        st = live.get(os.path.abspath(d)) or Store(d, "", run)
        if only_needed and not st.needs_compaction():
            continue
        rel = os.path.relpath(d, out_root).replace(os.sep, "/")
        gone += [f"{rel}/{n}" for n in st.compact()]
    note_absorbed(out_root, gone)
    return len(gone)


class FileGuard:
    """Keeps the saved output far below Kaggle's file cap: check() runs before
    every shard, counts the entries under the saved output (/kaggle/working on
    Kaggle) and at MAX_FILES compacts every store; if that cannot bring the count
    under STOP_FILES it returns False and the run stops (exit 77)."""

    def __init__(self, out_root: str, stores=(), run: str | None = None):
        self.out_root, self.stores, self.run = out_root, list(stores), run
        self.root = saved_root(out_root)
        self.max_seen = self.compactions = 0

    def count(self) -> int:
        n = count_entries(self.root)
        self.max_seen = max(self.max_seen, n)
        return n

    def check(self, reserve: int = 0) -> bool:
        n = self.count()
        if n + reserve < MAX_FILES:
            return True
        log(f"output holds {n} files and directories (Kaggle keeps {KAGGLE_FILE_CAP}): compacting")
        merged = compact_all(self.out_root, self.stores, self.run)
        self.compactions += 1
        n = count_entries(self.root)
        log(f"compacted {merged} segment files: {n} entries now")
        if n + reserve >= STOP_FILES:
            log(f"ERROR: {n} entries under {self.root} after compaction, most of them not this script's; "
                f"stopping before Kaggle's cap of {KAGGLE_FILE_CAP} drops files")
            return False
        return True

    def info(self) -> dict:
        return {"root": self.root, "max_seen": self.max_seen, "at_end": count_entries(self.root),
                "compactions": self.compactions, "cap": KAGGLE_FILE_CAP, "guard": MAX_FILES}


def resume_sources(spec: str, root: str = "/kaggle/input") -> list:
    """auto: every strong_probe/ with a manifest under root, at any depth (a
    notebook output attached as an input mounts at a depth Kaggle chooses)."""
    if spec == "none":
        return []
    if spec != "auto":
        return [p for p in spec.split(",") if p]
    return sorted({os.path.dirname(m) for m in glob.glob(os.path.join(root, "**", "strong_probe", "manifest.json"),
                                                         recursive=True)})


def merge_manifest(out_root: str, src_manifest: str) -> None:
    """Carry an earlier output's history into this one's manifest, which may exist
    already (plan writes it first): the sessions of both without exact duplicates,
    in start order, wall_s_total over them, and the union of `absorbed`; for
    models / plan / anything else this output's own entries win."""
    src = read_json(src_manifest)
    if not src:
        return
    path = os.path.join(out_root, "manifest.json")
    here = read_json(path)
    seen, sessions = set(), []
    for s in (src.get("sessions") or []) + (here.get("sessions") or []):
        key = json.dumps(s, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            sessions.append(s)
    sessions.sort(key=lambda s: float(s.get("t0") or 0.0) if isinstance(s, dict) else 0.0)
    merged = {**src, **here}
    for k in ("models", "plan"):
        if isinstance(src.get(k), dict) or isinstance(here.get(k), dict):
            merged[k] = {**(src.get(k) or {}), **(here.get(k) or {})}
    merged["sessions"] = sessions
    merged["wall_s_total"] = round(sum(float(s.get("wall_s") or 0.0) for s in sessions if isinstance(s, dict)), 1)
    merged["absorbed"] = sorted(set(src.get("absorbed") or []) | set(here.get("absorbed") or []))
    os.makedirs(out_root, exist_ok=True)
    write_json(path, merged)


def restore(out_root: str, sources, run: str | None = None) -> int:
    """Copy the store files of earlier sessions (other outputs' strong_probe/, in
    either layout) that are not here yet, a same-named file with other bytes
    under a new name, after merging every source's manifest into this output's
    (session history; `absorbed`: files a later compaction merged into another,
    which are skipped). Returns the number of files copied."""
    srcs = [s for s in sources if os.path.abspath(s) != os.path.abspath(out_root)]
    for src in srcs:
        man = os.path.join(src, "manifest.json")
        if os.path.exists(man):
            merge_manifest(out_root, man)
    absorbed = set(read_json(os.path.join(out_root, "manifest.json")).get("absorbed") or [])
    run = run or new_run_id()
    n = 0
    for src in srcs:
        for path in sorted(glob.glob(os.path.join(src, "*", "*", "*.parquet"))):
            rel = os.path.relpath(path, src).replace(os.sep, "/")
            m = SHARD_RE.match(rel)
            if not m or rel in absorbed:
                continue
            dst = os.path.join(out_root, rel)
            if os.path.exists(dst):
                if os.path.getsize(dst) == os.path.getsize(path) and file_sha256(dst) == file_sha256(path):
                    continue
                d, kind = os.path.dirname(dst), m.group(3)
                i = next(i for i in range(10 ** 6)
                         if not os.path.exists(os.path.join(d, f"{kind}-r{run}-{i:03d}.parquet")))
                dst = os.path.join(d, f"{kind}-r{run}-{i:03d}.parquet")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)
            n += 1
    return n


def update_manifest(out_root: str, **parts) -> dict:
    path = os.path.join(out_root, "manifest.json")
    man = read_json(path)
    for k, v in parts.items():
        if isinstance(v, dict) and isinstance(man.get(k), dict):
            man[k] = {**man[k], **v}
        else:
            man[k] = v
    write_json(path, man)
    return man


# --- configuration -----------------------------------------------------------------------------------

def model_slug(model: str, backend: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model) + ("__hf" if backend == "hf" else "")


def job_cfg(args, job: str) -> dict:
    """What changes a job's results (its hash marks the shards)."""
    base = dict(version=VERSION, job=job, model=args.model, revision=args.revision, backend=args.backend,
                quantization=args.quantization, task_tokens=args.task_tokens)
    if job == "rubric":
        return {**base, "system": SYSTEM_RUBRIC, "user": USER_RUBRIC, "prefill": RUBRIC_PREFILL,
                "meta": not args.rubric_no_meta, "meta_drop": META_DROP_RE.pattern,
                "max_tokens": RUBRIC_MAX_TOKENS, "logprobs": 20, "marker": MARKER}
    return {**base, "user": TASK_PH, "instruction": BOXED_INSTRUCTION, "k": args.k,
            "max_tokens": args.max_tokens, "sampling": {**ATTEMPT_SAMPLING, "presence_penalty": args.presence_penalty},
            "logprobs": args.attempt_logprobs, "record_raw": True, "force": not args.no_force,
            "force_every_attempt": True, "force_open": FORCE_OPEN, "force_closed": FORCE_CLOSED,
            "force_tokens": FORCE_TOKENS, "seed": args.seed}


def rates(args) -> dict:
    m = MODELS.get(args.model, dict(prefill=1000, decode=250, decode_short=500))
    return {"prefill": args.prefill_rate or m["prefill"], "decode": args.decode_rate or m["decode"],
            "decode_short": args.decode_rate_short or m["decode_short"]}


# --- budget ---------------------------------------------------------------------------------------

def rubric_secs(prompt_tokens: float, n_units: int, r: dict) -> float:
    return prompt_tokens / r["prefill"] + n_units * RUBRIC_GEN_EST / r["decode_short"]


def attempt_secs(n_units: int, prompt_tokens: float, k: int, max_tokens: int, force: bool, r: dict,
                 cache_hit: float = FORCED_CACHE_HIT) -> float:
    """cache_hit: the share of a forced readout's prefill that prefix caching
    serves (0 with --no-prefix-caching)."""
    gen = n_units * k * max_tokens * ATTEMPT_GEN_FRAC
    forced = (n_units * k * FORCED_FRAC * (prompt_tokens / max(n_units, 1) + max_tokens * ATTEMPT_GEN_FRAC)
              * (1 - cache_hit)) if force else 0.0
    return gen / r["decode"] + (prompt_tokens + forced) / r["prefill"]


def forced_cache_hit(args) -> float:
    return 0.0 if args.no_prefix_caching else FORCED_CACHE_HIT


def kv_tokens(args) -> float | None:
    """KV-cache capacity in tokens, from the model's shape and the memory left
    after the weights (an estimate; vLLM prints the real one at start)."""
    m = MODELS.get(args.model)
    if not m:
        return None
    per_gpu = 14.75 * args.gpu_mem - m["weights_gb"] / args.tp - 1.3          # GiB: T4 total, activations
    per_tok = 2 * m["layers"] * (m["kv_heads"] / args.tp) * m["head_dim"] * 2 / 2 ** 30
    return max(per_gpu, 0) / per_tok


#: the recommended split into commits (README, "Сессии"): each piece its own commit, so a lost commit (cancelled,
#: died, past 12 h: it saves nothing) costs one piece; the probe's result decides whether the rest is worth it
SESSION_PIECES = (("rubric", ["--jobs", "rubric"]),
                  ("probe", ["--jobs", "attempts", "--attempt-scope", "probe"]),
                  ("rest", ["--jobs", "attempts", "--attempt-scope", "rest"]))


def plan_budget(args, tok, units_r, units_a) -> dict:
    """Prompt tokens (exact with the model's tokenizer), expected generated tokens
    and hours per job at the assumed rates and in the slow case (decode at
    SLOW_DECODE of them: the planning figure), the recommended commits and the
    weekly quota with the smoke run and each commit's start-up."""
    r = rates(args)
    rs = {**r, "decode": r["decode"] * SLOW_DECODE, "decode_short": r["decode_short"] * SLOW_DECODE}
    kv = kv_tokens(args)
    out = {"rates_tok_s": r, "kv_cache_tokens_est": kv, "max_model_len": args.max_model_len}
    total = slow = 0.0
    pieces = {}                                                  # piece -> (hours, slow hours)
    if units_r is not None:
        p = rubric_prompt(tok, args.task_tokens)
        lens, trunc, per_b = [], 0, Counter()
        for u in units_r:
            ids, info = p.build(u["content"], rubric_meta(u["features"], not args.rubric_no_meta))
            lens.append(len(ids))
            trunc += info["truncated"]
            per_b[u["benchmark"]] += 1
        uncached = sum(lens) - (len(lens) - 1) * p.shared_prefix() if lens and not args.no_prefix_caching \
            else sum(lens)
        h, hs = rubric_secs(uncached, len(lens), r) / 3600, rubric_secs(uncached, len(lens), rs) / 3600
        out["rubric"] = {"units": len(lens), "per_benchmark": dict(per_b), "truncated": trunc,
                         "prompt_tokens": int(sum(lens)), "prompt_tokens_uncached": int(uncached),
                         "shared_prefix_tokens": p.shared_prefix(), "max_prompt": max(lens) if lens else 0,
                         "over_max_model_len": int(sum(x + RUBRIC_MAX_TOKENS > args.max_model_len for x in lens)),
                         "gen_tokens_est": len(lens) * RUBRIC_GEN_EST, "hours_est": round(h, 2),
                         "hours_est_slow": round(hs, 2)}
        total += h
        slow += hs
        pieces["rubric"] = (h, hs)
    if units_a is not None:
        p = attempt_prompt(tok, args.task_tokens)
        lens = [len(p.build(attempt_task(u["content"]))[0]) for u in units_a]
        force, hit = not args.no_force, forced_cache_hit(args)

        def hours(sel, rr):
            return attempt_secs(len(sel), sum(lens[i] for i in sel), args.k, args.max_tokens, force, rr, hit) / 3600

        every = list(range(len(units_a)))
        probe = [i for i in every if units_a[i]["probe"]]
        rest = [i for i in every if not units_a[i]["probe"]]
        h, hs = hours(every, r), hours(every, rs)
        # one shard's sequences at full length against the KV cache: beyond it V0 preempts by recompute
        shard_kv = args.attempt_shard * args.k * ((sum(lens) / max(len(lens), 1)) + args.max_tokens)
        out["attempts"] = {"units": len(lens), "probe_units": len(probe), "rest_units": len(rest),
                           "item_ids": sum(len(u["item_ids"]) for u in units_a),
                           "prompt_tokens": int(sum(lens)), "max_prompt": max(lens) if lens else 0,
                           "over_max_model_len": int(sum(x + args.max_tokens + FORCE_TOKENS + 16 > args.max_model_len
                                                         for x in lens)),
                           "gen_tokens_est": int(len(lens) * args.k * args.max_tokens * ATTEMPT_GEN_FRAC),
                           "forced_readouts": len(lens) * args.k if force else 0, "forced_cache_hit": hit,
                           "shard_kv_tokens": int(shard_kv),
                           "shard_fits_kv": bool(kv is None or shard_kv <= kv),
                           "hours_est": round(h, 2), "probe_hours_est": round(hours(probe, r), 2),
                           "rest_hours_est": round(hours(rest, r), 2),
                           "hours_est_slow": round(hs, 2), "probe_hours_est_slow": round(hours(probe, rs), 2),
                           "rest_hours_est_slow": round(hours(rest, rs), 2)}
        total += h
        slow += hs
        for name, sel in (("probe", probe), ("rest", rest)):
            if sel:
                pieces[name] = (hours(sel, r), hours(sel, rs))
    usable = max(args.session_hours - SESSION_OVERHEAD_H, 1.0)
    out["total_hours_est"] = round(total, 2)
    out["sessions_est"] = int(math.ceil(total / usable)) if total else 0
    out["weekly_quota_h"] = WEEKLY_GPU_H
    out["fits_weekly_quota"] = total + SESSION_OVERHEAD_H * out["sessions_est"] <= WEEKLY_GPU_H
    n_slow = int(math.ceil(slow / usable)) if slow else 0
    out["slow_case"] = {"decode_factor": SLOW_DECODE, "total_hours_est": round(slow, 2), "sessions_est": n_slow,
                        "fits_weekly_quota": slow + SESSION_OVERHEAD_H * n_slow <= WEEKLY_GPU_H}
    plan, fast_h, slow_h = [], SMOKE_H, SMOKE_H
    for name, piece_args in SESSION_PIECES:
        if name not in pieces:
            continue
        h, hs = pieces[name]
        n, n_fast = max(1, math.ceil(hs / usable)), max(1, math.ceil(h / usable))
        plan.append({"piece": name, "args": piece_args, "hours_est": round(h, 2), "hours_est_slow": round(hs, 2),
                     "commits_slow": n, "gpu_h_slow": round(hs + n * SESSION_OVERHEAD_H, 2)})
        slow_h += hs + n * SESSION_OVERHEAD_H
        fast_h += h + n_fast * SESSION_OVERHEAD_H
    out["session_plan"] = plan
    out["quota"] = {"smoke_h": SMOKE_H, "overhead_per_commit_h": SESSION_OVERHEAD_H,
                    "gpu_h_planning": round(slow_h, 2), "gpu_h_optimistic": round(fast_h, 2),
                    "weekly_quota_h": WEEKLY_GPU_H, "fits": slow_h <= WEEKLY_GPU_H}
    return out


# --- jobs ------------------------------------------------------------------------------------------

class Clock:
    def __init__(self, t0: float, hours: float, now=time.time, margin_s: float = 300.0):
        self.t0, self.now = t0, now
        self.stop_at = t0 + hours * 3600 - margin_s

    def left(self) -> float:
        return self.stop_at - self.now()

    def elapsed(self) -> float:
        return self.now() - self.t0


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _eta(done_units, secs, left_units):
    return (secs / done_units * left_units) if done_units else float("nan")


def run_rubric(args, backend, units, store: Store, clock: Clock, stats: dict, guard: FileGuard | None = None) -> str:
    """Returns 'done', 'deadline' (time or --max-shards), 'files' (the file guard) or
    'check_failed' (the first shard's parse rate is under RUBRIC_MIN_PARSE)."""
    p = rubric_prompt(backend.tok, args.task_tokens)
    done = store.done()
    todo = [u for u in units if u["unit"] not in done]
    log(f"[rubric] {len(units)} units, {len(done & {u['unit'] for u in units})} done, {len(todo)} to go")
    r = rates(args)
    params = GenParams(temperature=0.0, max_tokens=RUBRIC_MAX_TOKENS, logprobs=20, stop=("}",), keep_top=True)
    t_all, n_done, shards = 0.0, 0, 0
    for shard in _chunks(todo, args.rubric_shard):
        built = [p.build(u["content"], rubric_meta(u["features"], not args.rubric_no_meta)) for u in shard]
        ptok = sum(len(b[0]) for b in built)
        est = (t_all / n_done * len(shard)) if n_done else rubric_secs(ptok, len(shard), r)
        if (args.max_shards and shards >= args.max_shards) or clock.left() < 1.25 * est + 60:
            log(f"[rubric] stopping with {len(todo) - n_done} units left "
                f"(session left {clock.left() / 3600:.2f} h, next shard ~{est / 60:.1f} min)")
            return "deadline"
        if guard is not None and not guard.check():
            return "files"
        t = time.time()
        outs = backend.generate([b[0] for b in built], params)
        secs = time.time() - t
        rows = []
        for u, (ids, info), smp in zip(shard, built, outs):
            s = smp[0]
            rd = read_rubric(s)
            rows.append({"unit": u["unit"], "cfg": store.cfg, "benchmark": u["benchmark"], "key": u["key"],
                         "text_key": u["text_key"], "item_ids": u["item_ids"],
                         "content_sha256": u["content_sha256"], "content_sha": u["content_sha"],
                         "content_hash": u["content_hash"], "prompt_tokens": len(ids),
                         "task_tokens": info["task_tokens"], "truncated": info["truncated"],
                         "gen_tokens": len(s.token_ids), "finish_reason": s.finish_reason,
                         "gen_text": RUBRIC_PREFILL + s.text, **rd, "t": time.time()})
        store.write(pd.DataFrame(rows))
        t_all += secs
        n_done += len(shard)
        shards += 1
        gtok = sum(len(o[0].token_ids) for o in outs)
        parse = float(np.mean([x["parse_ok"] for x in rows]))
        stats["rubric"] = {"units_done_session": n_done, "secs": round(t_all, 1),
                           "prompt_tok_s": round(ptok / max(secs, 1e-9), 1)}
        log(f"[rubric] {n_done}/{len(todo)} units this session: shard of {len(shard)} in {secs:.0f}s, "
            f"{ptok} prompt + {gtok} gen tokens ({ptok / max(secs, 1e-9):.0f} prompt tok/s), "
            f"parse ok {parse:.2f} | eta "
            f"{_eta(n_done, t_all, len(todo) - n_done) / 3600:.2f} h, session left {clock.left() / 3600:.2f} h")
        if shards == 1 and not args.no_self_check and parse < RUBRIC_MIN_PARSE:
            stats["rubric"]["self_check"] = {"status": "FAILED", "parse_ok": parse, "min": RUBRIC_MIN_PARSE}
            log(f"[rubric] self-check FAILED: the first shard parses at {parse:.2f} (< {RUBRIC_MIN_PARSE}); the "
                "engine or the model's output is broken, stopping (exit 77; the shard is kept for a look, "
                "see README)")
            return "check_failed"
    return "done"


def attempt_seed(unit: str, j: int, seed: int) -> int:
    return int(hashlib.sha256(f"{unit}|{j}|{seed}".encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def attempts_self_check(recs: list[dict], recorder_status: str, force: bool) -> list:
    """What is wrong with a first attempt shard, if anything: the recorder check
    failed (the raw log-probs are misaligned), most texts are degenerate (fp16
    overflow), or no attempt yielded an answer although every one was forced."""
    bad = []
    if recorder_status == "FAILED":
        bad.append("the recorder check failed (raw log-probs misaligned with the engine's on greedy tokens)")
    deg = float(np.mean([degenerate(r["text"]) for r in recs])) if recs else 0.0
    if deg >= MAX_DEGENERATE:
        bad.append(f"{deg:.0%} of the texts are degenerate (one character most of the text)")
    if force and recs and all(r["answer"] is None for r in recs):
        bad.append("no attempt yielded an answer, forced readouts included")
    return bad


def run_attempts(args, backend, units, store: Store, clock: Clock, stats: dict, guard: FileGuard | None = None) -> str:
    """Returns 'done', 'deadline', 'files' or 'check_failed' (attempts_self_check on
    the first shard). Graded correctness stays in memory: the shard's mean in the
    log, the session's in stats; the store gets canon_sha, not the canonical
    answer, and no REF_DIAG."""
    p = attempt_prompt(backend.tok, args.task_tokens)
    enc_open, enc_closed = backend.tok.enc(FORCE_OPEN), backend.tok.enc(FORCE_CLOSED)
    eos = set(getattr(backend.tok, "eos_ids", set()))
    done = store.done()
    todo = [u for u in units if u["unit"] not in done]
    log(f"[attempts] {len(units)} units ({sum(u['probe'] for u in units)} probe), "
        f"{len(done & {u['unit'] for u in units})} done, {len(todo)} to go; k={args.k}, "
        f"max_tokens={args.max_tokens}")
    r = rates(args)
    params = GenParams(max_tokens=args.max_tokens, logprobs=args.attempt_logprobs, keep_top=False, record_raw=True,
                       **{**ATTEMPT_SAMPLING, "presence_penalty": args.presence_penalty})
    fparams = GenParams(temperature=0.0, max_tokens=FORCE_TOKENS, logprobs=1, record_raw=True)
    t_all, n_done, shards, gen_all = 0.0, 0, 0, 0
    sources, gap_max, gap_n, graded = Counter(), float("nan"), 0, []
    for shard in _chunks(todo, args.attempt_shard):
        built = [p.build(attempt_task(u["content"]))[0] for u in shard]
        est = (t_all / n_done * len(shard)) if n_done else attempt_secs(
            len(shard), sum(map(len, built)), args.k, args.max_tokens, not args.no_force, r, forced_cache_hit(args))
        if (args.max_shards and shards >= args.max_shards) or clock.left() < 1.25 * est + 60:
            log(f"[attempts] stopping with {len(todo) - n_done} units left "
                f"(session left {clock.left() / 3600:.2f} h, next shard ~{est / 60:.1f} min)")
            return "deadline"
        if guard is not None and not guard.check():
            return "files"
        t = time.time()
        prompts = [ids for ids in built for _ in range(args.k)]
        seeds = [attempt_seed(u["unit"], j, args.seed) for u in shard for j in range(args.k)]
        outs = [o[0] for o in backend.generate(prompts, params, seeds)]
        gen = sum(len(s.token_ids) for s in outs)
        forced = [None] * len(outs)
        if not args.no_force:                   # every attempt: lp_answer is one quantity for all
            fprompts = [forced_prompt(prompts[i], s, eos, enc_open, enc_closed) for i, s in enumerate(outs)]
            forced = [fo[0] for fo in backend.generate(fprompts, fparams)]
        secs = time.time() - t
        t_shard = time.time()
        items, samples, shard_recs, shard_graded = [], [], [], []
        for ui, u in enumerate(shard):
            recs = []
            for j in range(args.k):
                i = ui * args.k + j
                rec = analyse_attempt(outs[i], forced[i])
                recs.append(rec)
                sources[rec["stats_source"]] += 1
                if rec["recorder_gap_n"]:
                    gap_n += rec["recorder_gap_n"]
                    gap_max = rec["recorder_gap"] if not np.isfinite(gap_max) else max(gap_max, rec["recorder_gap"])
            agg = attempt_aggregate(recs, u["golds"])
            canons = agg.pop("canons")
            ref = {c: agg.pop(c) for c in REF_DIAG}            # needs the reference answer: memory only
            if np.isfinite(ref["graded"]):
                shard_graded.append(ref["graded"])
            shard_recs += recs
            for j, rec in enumerate(recs):
                samples.append({"unit": u["unit"], "cfg": store.cfg, "j": j,
                                "seed": attempt_seed(u["unit"], j, args.seed), "canon_sha": canon_sha(canons[j]),
                                **rec, "t": t_shard})
            items.append({"unit": u["unit"], "cfg": store.cfg, "benchmark": u["benchmark"],
                          "item_ids": [m[0] for m in u["members"]], "keys": [m[1] for m in u["members"]],
                          "text_keys": [m[2] for m in u["members"]],
                          "content_sha256s": [m[3] for m in u["members"]],
                          "content_hashes": [m[4] for m in u["members"]], "content_sha": u["content_sha"],
                          "comps": u["comps"], "probe": bool(u["probe"]), "n_subjects": int(u["n_subjects"]),
                          "prompt_tokens": len(built[ui]), **agg, "t": t_shard})
        store.write(pd.DataFrame(items), pd.DataFrame(samples))
        t_all += secs
        n_done += len(shard)
        shards += 1
        gen_all += gen
        graded += shard_graded
        check = ("ok" if gap_n and gap_max <= RECORDER_TOL else "FAILED" if gap_n else "n/a")
        stats["attempts"] = {"units_done_session": n_done, "secs": round(t_all, 1),
                             "gen_tok_s": round(gen_all / max(t_all, 1e-9), 1), "stats_source": dict(sources),
                             "graded": _mean(graded), "graded_units": len(graded),
                             "recorder_check": {"status": check, "max_abs_gap": None if not gap_n else gap_max,
                                                "tokens": gap_n, "tol": RECORDER_TOL}}
        if shards == 1 or check == "FAILED":
            log(f"[attempts] token statistics: {dict(sources)}; recorder check {check} (max |raw - engine| "
                f"log-prob {gap_max:.2e} over {gap_n} greedy tokens, tol {RECORDER_TOL})"
                + (" -- WARNING: the raw log-probs are misaligned; see README" if check == "FAILED" else ""))
        log(f"[attempts] {n_done}/{len(todo)} units this session: shard of {len(shard)} x {args.k} in "
            f"{secs:.0f}s, {gen} gen tokens ({gen / max(secs, 1e-9):.0f} tok/s), forced "
            f"{sum(r_['forced'] for r_ in samples)}/{len(samples)}, closed "
            f"{np.mean([x['closed_rate'] for x in items]):.2f}, graded {_mean(shard_graded):.2f} | eta "
            f"{_eta(n_done, t_all, len(todo) - n_done) / 3600:.2f} h, session left {clock.left() / 3600:.2f} h")
        if shards == 1 and not args.no_self_check:
            bad = attempts_self_check(shard_recs, check, not args.no_force)
            if bad:
                stats["attempts"]["self_check"] = {"status": "FAILED", "why": bad}
                log("[attempts] self-check FAILED on the first shard: " + "; ".join(bad) + ". Stopping (exit 77; "
                    "the shard is kept for a look, see README)")
                return "check_failed"
    return "done"


# --- export ---------------------------------------------------------------------------------------

RUBRIC_DIAG = [f"rubric_{s}_{d}" for s in SCALES for d in ("entropy", "mass")] + \
    ["solve_share_entropy", "solve_share_mass", "time_log_minutes_entropy", "time_log_minutes_mass",
     "prompt_tokens", "task_tokens", "gen_tokens", "truncated_n"]
ATTEMPT_COLS = ["attempt", "design", "answer", "n_tokens", "capped", "tok_entropy", "tok_lp", "lp_answer",
                "refuse", "forced", "closed", "n_think", "tok_entropy_think", "tok_lp_think", "tok_entropy_answer",
                "tok_lp_answer", "ent_first256", "ent_first1024", "lp_first256", "lp_first1024"]


def _majority_cfg(df: pd.DataFrame):
    """The config with the most units among the shards; on a tie the latest one."""
    if not len(df):
        return None
    g = df.groupby("cfg").agg(n=("unit", "size"), t=("t", "max")).sort_values(["n", "t"], ascending=False)
    if len(g) > 1:
        log(f"export: {len(g)} configs among the shards, exporting {g.index[0]} ({int(g.n.iloc[0])} units)")
    return g.index[0]


def rubric_export_rows(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in units.to_dict("records"):
        feats = {f"rubric_{s}": r[s] for s in SCALES}
        feats.update({f"rubric_{s}_entropy": r[f"{s}_entropy"] for s in SCALES})
        feats.update({f"rubric_{s}_mass": r[f"{s}_mass"] for s in SCALES})
        feats.update(solve_share=r["solve_share"], solve_share_entropy=r["difficulty_entropy"],
                     solve_share_mass=r["difficulty_mass"], time_log_minutes=r["time_log_minutes"],
                     time_log_minutes_entropy=r["human_time_entropy"], time_log_minutes_mass=r["human_time_mass"],
                     prompt_tokens=float(r["prompt_tokens"]), task_tokens=float(r["task_tokens"]),
                     gen_tokens=float(r["gen_tokens"]), truncated_n=float(r["truncated"]))
        for iid in r["item_ids"]:
            rows.append({"benchmark": r["benchmark"], "item_id": str(iid), "content_sha256": r["content_sha256"],
                         **{k: float(v) for k, v in feats.items()}})
    return pd.DataFrame(rows)


def attempt_export_rows(units: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = {u: g.sort_values("j") for u, g in samples.groupby("unit")}
    for r in units.to_dict("records"):
        g = by.get(r["unit"])
        if g is None:
            continue
        for iid, csh in zip(r["item_ids"], r["content_sha256s"]):
            for s in g.to_dict("records"):
                rows.append({"benchmark": r["benchmark"], "item_id": str(iid), "content_sha256": csh,
                             "attempt": int(s["j"]), "design": "cot",
                             "answer": _clean(s["answer"]) or None,
                             "n_tokens": int(s["n_tokens"]), "capped": bool(s["capped"]),
                             "tok_entropy": float(s["tok_entropy"]), "tok_lp": float(s["tok_lp"]),
                             "lp_answer": float(s["lp_answer"]), "refuse": bool(s["refuse"]),
                             "forced": bool(s["forced"]), "closed": bool(s["closed"]), "n_think": int(s["n_think"]),
                             **{c: float(s[c]) for c in ATTEMPT_COLS[12:]}})
    df = pd.DataFrame(rows, columns=["benchmark", "item_id", "content_sha256"] + ATTEMPT_COLS)
    return df.astype({"attempt": "int64", "n_tokens": "int64", "n_think": "int64", "capped": bool,
                      "refuse": bool, "forced": bool, "closed": bool})


def logprobs_kind(sources) -> str:
    """The export manifest's `logprobs`: 'raw' when every attempt's token
    statistics are the raw distribution's full-vocabulary ones (D2's definition;
    also when there are no attempts: the rubric is greedy, raw on every engine),
    else the one other source, or 'mixed'."""
    ks = {k for k, v in dict(sources).items() if v}
    if not ks or ks == {STATS_RAW}:
        return "raw"
    return next(iter(ks)) if len(ks) == 1 else "mixed"


def token_stats_info(sm: pd.DataFrame) -> dict:
    """The attempts' token-statistics provenance for the export manifest."""
    src = Counter(sm["stats_source"]) if "stats_source" in sm.columns else Counter({STATS_PROC: len(sm)})
    gap = sm["recorder_gap"].to_numpy(float) if "recorder_gap" in sm.columns else np.zeros(0)
    gap = gap[np.isfinite(gap)]
    n = int(sm["recorder_gap_n"].sum()) if "recorder_gap_n" in sm.columns else 0
    kind = logprobs_kind(src)
    return {"logprobs": kind, "sources": {k: int(v) for k, v in src.items()},
            "definitions": {k: STATS_DEF[k] for k in src if k in STATS_DEF},
            "comparable_to_d2": kind == "raw",
            "recorder_check": {"status": ("ok" if gap.size and gap.max() <= RECORDER_TOL else
                                          "FAILED" if gap.size else "n/a"),
                               "max_abs_gap": float(gap.max()) if gap.size else None, "tokens": n,
                               "tol": RECORDER_TOL,
                               "what": "max |recorder - engine| log-prob over the forced greedy readouts' "
                                       "tokens, where the engine's logprobs are raw"}}


def export(out_root: str) -> list:
    """Rebuild <model>/export/ from the shards, in experiments/strong_llm_eval.py's
    schema v1, for every model under out_root. Returns the export dirs."""
    man = read_json(os.path.join(out_root, "manifest.json"))
    dirs = []
    for mdir in sorted(glob.glob(os.path.join(out_root, "*"))):
        if not os.path.isdir(mdir) or not any(os.path.isdir(os.path.join(mdir, j)) for j in ("rubric", "attempts")):
            continue
        slug = os.path.basename(mdir)
        minfo = (man.get("models") or {}).get(slug, {})
        ex = os.path.join(mdir, "export")
        if os.path.isdir(ex):
            shutil.rmtree(ex)
        os.makedirs(ex)
        shards, kinds, keys, summary, harness, tstats = [], {}, [], {"model": slug}, {}, None
        ru = Store(os.path.join(mdir, "rubric"), "").read("items")
        if len(ru):
            cfg = _majority_cfg(ru)
            ru = ru[ru.cfg == cfg].reset_index(drop=True)
            flat = rubric_export_rows(ru)
            path = os.path.join(ex, "rubric", "rubric.parquet")
            write_parquet(flat, path)
            shards.append({"path": "rubric/rubric.parquet", "rows": int(len(flat)), "sha256": file_sha256(path)})
            kinds["rubric"] = {
                "prompt": {"system": SYSTEM_RUBRIC, "user": USER_RUBRIC, "assistant_prefill": RUBRIC_PREFILL,
                           "marker": MARKER, "rendered": minfo.get("rubric_rendered")},
                "scales": {f"rubric_{s}": f"{d}. {a}." for s, (d, a) in SCALES.items()},
                "ratings": {"solve_share": DIFFICULTY_DEF, "time_log_minutes": HUMAN_TIME_DEF,
                            "human_time_midpoints_minutes": list(HUMAN_MID)},
                "readout": ("greedy JSON answer, thinking off; at each value's position the top-20 logprobs "
                            "restricted to the allowed single-digit tokens and renormalised; rubric_<scale> is "
                            "the expected level 0-5, solve_share 1 - the expected fail share at the deciles' "
                            "midpoints, time_log_minutes the expected ln(minutes) at the buckets' geometric "
                            "midpoints; _mass is the digits' share of the whole next-token distribution"),
                "signs": {**{k: s for k, (s, _) in RUBRIC_EXPORT.items()}, **{c: 0 for c in RUBRIC_DIAG}},
                "cfg": cfg, "units": int(len(ru)), "parse_ok": float(ru.parse_ok.mean())}
            for r in ru.to_dict("records"):
                for iid in r["item_ids"]:
                    keys.append({"job": "rubric", "benchmark": r["benchmark"], "item_id": str(iid), "key": r["key"],
                                 "text_key": r["text_key"], "content_sha": r["content_sha"],
                                 "content_hash": r["content_hash"]})
            for f, (sign, _) in RUBRIC_EXPORT.items():
                harness[f"rubric_{f}" if not f.startswith("rubric_") else f] = {
                    str(i): sign * float(v) for i, v in zip(flat.item_id, flat[f]) if np.isfinite(v)}
            summary["rubric"] = {"units": int(len(ru)), "item_rows": int(len(flat)),
                                 "per_benchmark": flat.benchmark.value_counts().to_dict(),
                                 "parse_ok": float(ru.parse_ok.mean()),
                                 "means": {f: float(flat.groupby("benchmark")[f].mean().mean())
                                           for f in RUBRIC_EXPORT}}
        st = Store(os.path.join(mdir, "attempts"), "")
        au = st.read("items")
        if len(au):
            cfg = _majority_cfg(au)
            au = au[au.cfg == cfg].reset_index(drop=True)
            sm = st.read("samples", cfg)
            sm = sm[sm.unit.isin(set(au.unit))]
            flat = attempt_export_rows(au, sm)
            path = os.path.join(ex, "attempts", "attempts.parquet")
            write_parquet(flat, path)
            shards.append({"path": "attempts/attempts.parquet", "rows": int(len(flat)), "sha256": file_sha256(path)})
            write_parquet(sm, os.path.join(ex, "_detail", "attempt_samples.parquet"))
            write_parquet(au, os.path.join(ex, "_detail", "attempt_units.parquet"))
            tstats = token_stats_info(sm)
            n_forced = int(sm["forced_text"].notna().sum()) if "forced_text" in sm.columns else 0
            if tstats["recorder_check"]["status"] == "FAILED":
                log(f"export: WARNING recorder check failed: {tstats['recorder_check']}")
            kinds["attempts"] = {
                "prompt": {"system": None, "user": "<item_content>, plus the instruction below when the content "
                                                   "has no \\boxed", "instruction": BOXED_INSTRUCTION,
                           "thinking": True, "rendered": minfo.get("attempt_rendered"),
                           "force_open": FORCE_OPEN, "force_closed": FORCE_CLOSED},
                "sampling": minfo.get("attempt_sampling", ATTEMPT_SAMPLING),
                "k": int(au.k.iloc[0]), "max_new_tokens": minfo.get("max_tokens"), "designs": ["cot"],
                "forced": (f"every attempt: a greedy continuation of at most {FORCE_TOKENS} tokens after its "
                           "reasoning (through </think>, or all of it when the thinking did not close) and the "
                           "Final Answer opener; `answer` is the attempt's own \\boxed{} after </think>, else the "
                           "forced one (forced = True)" if n_forced else "none (--no-force)"),
                "lp_answer": ("mean log-prob of the forced readout's answer tokens (greedy: the raw distribution) "
                              "for every attempt; lp_boxed (in _detail and _harness.json) is the attempt's own "
                              "boxed answer, where it gave one" if n_forced else
                              "mean log-prob of the attempt's own boxed answer tokens (NaN without one)"),
                "entropy": ("tok_entropy, tok_lp and the span columns: " + STATS_DEF.get(
                    STATS_RAW if tstats["logprobs"] == "raw" else tstats["logprobs"],
                    "mixed sources, see token_stats.sources")),
                "token_stats": tstats,
                "cfg": cfg, "units": int(len(au)), "probe_units": int(au.probe.sum())}
            for r in au.to_dict("records"):
                for iid, k_, tk, ch in zip(r["item_ids"], r["keys"], r["text_keys"], r["content_hashes"]):
                    keys.append({"job": "attempts", "benchmark": r["benchmark"], "item_id": str(iid), "key": k_,
                                 "text_key": tk, "content_sha": r["content_sha"], "content_hash": ch})
            for f, sign in ATTEMPT_SIGNS.items():
                harness[f"attempts_{f}"] = {str(i): sign * float(v) for ids, v in zip(au.item_ids, au[f])
                                            if np.isfinite(v) for i in ids}
            summary["attempts"] = {"units": int(len(au)), "probe_units": int(au.probe.sum()),
                                   "attempt_rows": int(len(flat)), "logprobs": tstats["logprobs"],
                                   "recorder_check": tstats["recorder_check"]["status"],
                                   "closed_rate": float(au.closed_rate.mean()),
                                   "trunc_rate": float(au.trunc_rate.mean()), "forced_rate": float(au.forced_rate.mean()),
                                   "mean_len": float(au.mean_len.mean())}
        if len(ru):
            write_parquet(ru, os.path.join(ex, "_detail", "rubric_units.parquet"))
        write_parquet(pd.DataFrame(keys, columns=["job", "benchmark", "item_id", "key", "text_key", "content_sha",
                                                  "content_hash"]), os.path.join(ex, "_keys.parquet"))
        write_json(os.path.join(ex, "_harness.json"), harness)     # one file: Kaggle's output keeps <= 500
        manifest = {"schema_version": SCHEMA_VERSION, "hash": HASH_DEF, "kinds": kinds, "shards": shards,
                    "model": {"repo": minfo.get("model"), "revision": minfo.get("revision"), "dtype": "float16",
                              "quantization": minfo.get("quantization") or "none",
                              "engine": minfo.get("engine")},
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": man.get("wall_s_total"),
                    "sessions": len(man.get("sessions") or []),
                    "gpu": minfo.get("gpus"), "notebook_digest": man.get("script_sha256"),
                    "logprobs": tstats["logprobs"] if tstats else "raw",
                    "logprobs_engine": minfo.get("logprobs"), "slug": slug}
        write_json(os.path.join(ex, "manifest.json"), manifest)
        write_json(os.path.join(ex, "_summary.json"), summary)
        what = ", ".join("%s %d rows" % (s["path"], s["rows"]) for s in shards) or "nothing"
        log(f"export: {ex} ({what})")
        dirs.append(ex)
    root = saved_root(out_root)
    n = count_entries(root)
    log(f"output: {n} files and directories under {root} (Kaggle saves at most {KAGGLE_FILE_CAP})"
        + (" -- WARNING: close to the cap" if n >= STOP_FILES else ""))
    return dirs


def split_harness(export_dir: str) -> list:
    """Local: _harness.json ({name: {item_id: x}}) -> _harness/<name>.json, the
    one-covariate files python experiments/harness.py --stage eval --cov reads."""
    h = read_json(os.path.join(export_dir, "_harness.json"))
    if not h:
        raise ConfigError(f"no _harness.json in {export_dir}")
    paths = []
    for name, m in sorted(h.items()):
        paths.append(os.path.join(export_dir, "_harness", f"{name}.json"))
        write_json(paths[-1], m)
    log(f"split-harness: {len(paths)} files in {os.path.join(export_dir, '_harness')}")
    return paths


# --- watchdog and GPU check ----------------------------------------------------------------------

def _kill_children():
    """Kill this process's descendants (vLLM's workers) before it exits. The
    notebook also kills the run's whole process group once it is gone (sp in the
    notebook), which reaches workers orphaned by a SIGKILL of this process; the
    GPU process list of nvidia-smi is empty inside Kaggle's container."""
    try:
        import psutil
        procs = psutil.Process().children(recursive=True)
    except Exception:                                        # noqa: BLE001 - no psutil: direct children
        import multiprocessing
        procs = multiprocessing.active_children()
    for c in procs:
        try:
            c.kill()
        except Exception:                                    # noqa: BLE001
            pass


class Watchdog(threading.Thread):
    """Ends the process at the hard session limit, and if the engine is not up
    within the init timeout (a stuck NCCL or download would burn the session)."""

    def __init__(self, hard_at: float, init_timeout_s: float):
        super().__init__(daemon=True)
        self.hard_at, self.init_timeout_s, self.init_deadline = hard_at, init_timeout_s, None

    def arm_init(self):
        self.init_deadline = time.time() + self.init_timeout_s

    def init_done(self):
        self.init_deadline = None

    def run(self):
        while True:
            time.sleep(10)
            now = time.time()
            if now > self.hard_at:
                log("watchdog: hard session limit reached, exiting (finished shards are kept)")
                _kill_children()
                os._exit(EXIT_DEADLINE)
            if self.init_deadline is not None and now > self.init_deadline:
                log("watchdog: the engine did not start in time, exiting 76 (the RUN cell retries with "
                    "--nccl-p2p-disable, then also --enforce-eager and a smaller --gpu-mem)")
                _kill_children()
                os._exit(EXIT_INIT)


def gpu_check(tp: int) -> list:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise ConfigError(f"no GPU visible ({type(e).__name__}): set Accelerator to 'GPU T4 x2'")
    gpus = [[x.strip() for x in ln.split(",")] for ln in out.strip().splitlines() if ln.strip()]
    log("GPUs: " + "; ".join(", ".join(g) for g in gpus))
    if len(gpus) < tp:
        raise ConfigError(f"{len(gpus)} GPU(s) visible, --tp {tp} needs {tp}: set Accelerator to 'GPU T4 x2'")
    for g in gpus:
        try:
            cc = float(g[2])
        except (IndexError, ValueError):
            continue
        if cc < 7.5:
            raise ConfigError(f"{g[0]} has compute capability {cc}: vLLM's AWQ kernel needs >= 7.5 "
                              "(set Accelerator to 'GPU T4 x2', not P100)")
    return gpus


# --- commands --------------------------------------------------------------------------------------

def parse_args(argv):
    ap = argparse.ArgumentParser(description="PAIEC K1: rubric and attempts with a stronger model on Kaggle")
    ap.add_argument("command", choices=("plan", "run", "export", "split-harness"))
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--download", default=",".join(BENCHES), help="benchmarks whose core tables to fetch")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--jobs", default="rubric,attempts")
    ap.add_argument("--backend", default="vllm", choices=("vllm", "hf"))
    ap.add_argument("--model", default=None, help=f"default {DEFAULT_MODEL} (hf backend: {HF_FALLBACK_MODEL})")
    ap.add_argument("--revision", default=None, help="commit sha (default: the pinned one of MODELS)")
    ap.add_argument("--quantization", default=None, help="default: from MODELS (awq for the AWQ repos)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-num-seqs", type=int, default=48)
    ap.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs (less memory, slower decode)")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--nccl-p2p-disable", action="store_true", help="NCCL_P2P_DISABLE=1 (if TP init hangs)")
    ap.add_argument("--task-tokens", type=int, default=3072, help="task text cut to this many tokens (head+tail)")
    ap.add_argument("--rubric-benchmarks", default=",".join(PARENTS))
    ap.add_argument("--rubric-limit", type=int, default=0, help="units per benchmark (0: all)")
    ap.add_argument("--rubric-no-meta", action="store_true", help="leave item_features out of the rubric prompt")
    ap.add_argument("--rubric-shard", type=int, default=128)
    ap.add_argument("--attempt-scope", default="all", choices=("probe", "rest", "all"),
                    help="the probe texts, the others, or both (probe first)")
    ap.add_argument("--attempt-limit", type=int, default=0, help="units (0: all of the scope)")
    ap.add_argument("--attempt-shard", type=int, default=5,
                    help="units per generate call (x k sequences; 5 x 4 x ~4.3k tokens fits 14B's KV cache)")
    ap.add_argument("--attempt-logprobs", type=int, default=5,
                    help="engine top-k logprobs per attempt token (the *_engine diagnostics; the raw statistics "
                         "come from the recorder; 1 saves V0's per-token detokenisation of the alternatives). "
                         "Part of the job config: changing it after attempts started starts them over")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--presence-penalty", type=float, default=ATTEMPT_SAMPLING["presence_penalty"])
    ap.add_argument("--no-force", action="store_true", help="no forced answer after a truncated attempt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--session-hours", type=float, default=11.0, help="from SP_T0 (the notebook's first cell)")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--init-timeout-min", type=float, default=25.0)
    ap.add_argument("--resume-from", default="auto", help="auto | none | comma-separated strong_probe dirs")
    ap.add_argument("--max-shards", type=int, default=0, help="per job and session (0: no limit; for tests)")
    ap.add_argument("--no-self-check", action="store_true",
                    help="do not stop on a failed first-shard self-check (exit 77; README)")
    ap.add_argument("--export-dir", default=None, help="split-harness: the export directory")
    ap.add_argument("--prefill-rate", type=float, default=0.0)
    ap.add_argument("--decode-rate", type=float, default=0.0)
    ap.add_argument("--decode-rate-short", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.model is None:
        a.model = HF_FALLBACK_MODEL if a.backend == "hf" else DEFAULT_MODEL
    m = MODELS.get(a.model, {})
    a.revision = a.revision or m.get("revision")
    if a.quantization is None:
        a.quantization = None if a.backend == "hf" else m.get("quantization")
    a.jobs = [j.strip() for j in a.jobs.split(",") if j.strip()]
    bad = [j for j in a.jobs if j not in ("rubric", "attempts")]
    if bad:
        ap.error(f"unknown jobs {bad}")
    if a.backend == "hf" and "attempts" in a.jobs:
        log("hf backend: the attempts job needs vLLM, running the rubric only")
        a.jobs = [j for j in a.jobs if j == "rubric"]
    a.rubric_benchmarks = [b.strip() for b in a.rubric_benchmarks.split(",") if b.strip()]
    a.download = [b.strip() for b in a.download.split(",") if b.strip()]
    if a.t0 is None:
        a.t0 = float(os.environ.get("SP_T0") or time.time())
    return a


def load_units(args):
    ur = rubric_units(args.data_dir, args.rubric_benchmarks, args.rubric_limit or None) \
        if "rubric" in args.jobs else None
    ua = attempt_units(args.data_dir, args.attempt_scope, args.attempt_limit or None) \
        if "attempts" in args.jobs else None
    for name, us in (("rubric", ur), ("attempts", ua)):
        if us is not None:
            log(f"{name}: {len(us)} units standing for {sum(len(u['item_ids']) for u in us)} item_ids"
                + (f" ({sum(u['probe'] for u in us)} probe)" if name == "attempts" else ""))
    return ur, ua


def needed_benches(args) -> list:
    """--download (the five binary benchmarks by default) and whatever the jobs read."""
    need = set(args.download) | (set(args.rubric_benchmarks) if "rubric" in args.jobs else set())
    return sorted(need | ({"matharena"} if "attempts" in args.jobs else set()))


def check_prompts_agnostic():
    text = (SYSTEM_RUBRIC + USER_RUBRIC + RUBRIC_PREFILL + BOXED_INSTRUCTION + FORCE_OPEN).lower()
    hit = [b for b in BENCHES + ("mmdocrag", "measurement-db") if b in text]
    if hit:
        raise AssertionError(f"benchmark names in a prompt template: {hit}")


def cmd_plan(args, tok=None, token_getter=None, downloader=None) -> int:
    check_prompts_agnostic()
    download_data(args.data_dir, needed_benches(args), token_getter=token_getter, downloader=downloader)
    ur, ua = load_units(args)
    if tok is None:
        from transformers import AutoTokenizer
        tok = HFTok(AutoTokenizer.from_pretrained(args.model, revision=args.revision))
    b = plan_budget(args, tok, ur, ua)
    log("budget: " + json.dumps(b, indent=1))
    for job in ("rubric", "attempts"):
        if job in b and b[job].get("over_max_model_len"):
            log(f"WARNING: {b[job]['over_max_model_len']} {job} prompts exceed --max-model-len {args.max_model_len}")
    if "attempts" in b and not b["attempts"]["shard_fits_kv"]:
        log(f"WARNING: an attempt shard at full length ({b['attempts']['shard_kv_tokens']} tokens) exceeds the "
            f"estimated KV cache ({b['kv_cache_tokens_est']:.0f}): V0 will preempt; lower --attempt-shard")
    sc, q = b["slow_case"], b["quota"]
    log(f"budget, planning figure (decode at {SLOW_DECODE:g}x the assumed rate): {sc['total_hours_est']} h of "
        f"generation; optimistic (the assumed rates) {b['total_hours_est']} h")
    for piece in b["session_plan"]:
        log(f"  commit(s) {piece['piece']}: ARGS = {piece['args']}: {piece['hours_est_slow']} h planning "
            f"({piece['hours_est']} h optimistic), {piece['commits_slow']} commit(s), {piece['gpu_h_slow']} GPU-h")
    log(f"  weekly quota: smoke run {SMOKE_H} h + the commits, each with {SESSION_OVERHEAD_H} h of start-up = "
        f"{q['gpu_h_planning']} GPU-h planning ({q['gpu_h_optimistic']} optimistic) of {WEEKLY_GPU_H:.0f}: "
        + ("fits" if q["fits"] else "does NOT fit: run the probe, read it locally, then decide on the rest"))
    update_manifest(args.out, plan={model_slug(args.model, args.backend): b})
    return 0


def cmd_run(args, backend=None, token_getter=None, downloader=None, now=time.time) -> int:
    check_prompts_agnostic()
    clock = Clock(args.t0, args.session_hours, now)
    os.makedirs(args.out, exist_ok=True)
    run = new_run_id()
    log(f"run {run}: jobs {args.jobs}, model {args.model}@{(args.revision or '')[:12]}, backend {args.backend}, "
        f"session left {clock.left() / 3600:.2f} h")
    download_data(args.data_dir, needed_benches(args), token_getter=token_getter, downloader=downloader)
    src = resume_sources(args.resume_from)
    if src:
        log(f"restored {restore(args.out, src, run)} store files from {src}")
    ur, ua = load_units(args)
    slug = model_slug(args.model, args.backend)
    mroot = os.path.join(args.out, slug)
    cfgs = {j: job_cfg(args, j) for j in args.jobs}
    stores = {j: Store(os.path.join(mroot, j), digest(cfgs[j]), run) for j in args.jobs}
    n = compact_all(args.out, stores.values(), run, only_needed=True)
    if n:
        log(f"compacted {n} store files (the k1.2 layout, or more than {SEGS_PER_KIND} files of a kind)")
    guard = FileGuard(args.out, stores.values(), run)
    log(f"output: {guard.count()} files and directories under {guard.root} (guard {MAX_FILES}, "
        f"Kaggle's cap {KAGGLE_FILE_CAP})")
    units = {"rubric": ur, "attempts": ua}
    left = {j: len({u["unit"] for u in units[j]} - stores[j].done()) for j in args.jobs}
    log("to do: " + ", ".join(f"{j} {n}" for j, n in left.items()))
    if not any(left.values()):
        export(args.out)
        return 0
    if clock.left() < 600:
        log("less than 10 minutes of the session left, not starting")
        return EXIT_DEADLINE
    wd = None
    t_start = now()
    if backend is None:
        if args.nccl_p2p_disable:
            os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if args.backend == "vllm":
            gpu_check(args.tp)
        wd = Watchdog(args.t0 + args.session_hours * 3600 + 900, args.init_timeout_min * 60)
        wd.start()
        wd.arm_init()
        t = time.time()
        if args.backend == "hf":
            backend = HFBackend(args.model, args.revision)
        else:
            backend = VLLMBackend(args.model, args.revision, args.quantization, args.tp, args.max_model_len,
                                  args.gpu_mem, args.max_num_seqs, args.enforce_eager, not args.no_prefix_caching,
                                  seed=args.seed)
        wd.init_done()
        log(f"engine up in {time.time() - t:.0f}s: {backend.info}")
    info = dict(getattr(backend, "info", {}) or {})
    minfo = {"model": args.model, "revision": args.revision, "quantization": args.quantization,
             "backend": args.backend, "tp": args.tp, "max_model_len": args.max_model_len,
             "gpu_mem": args.gpu_mem, "cfg": {j: stores[j].cfg for j in args.jobs},
             "max_tokens": args.max_tokens, "k": args.k, "attempt_logprobs": args.attempt_logprobs,
             "attempt_sampling": {**ATTEMPT_SAMPLING, "presence_penalty": args.presence_penalty}, **info}
    if "rubric" in args.jobs:
        minfo["rubric_rendered"] = rubric_prompt(backend.tok, args.task_tokens).rendered
    if "attempts" in args.jobs:
        minfo["attempt_rendered"] = attempt_prompt(backend.tok, args.task_tokens).rendered
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            script_sha = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        script_sha = None
    update_manifest(args.out, script_sha256=script_sha, vllm_pin=VLLM_PIN, transformers_pin=TRANSFORMERS_PIN,
                    version=VERSION, store_layout=LAYOUT, models={slug: minfo})
    stats, status = {}, {}
    for j in args.jobs:
        if not left[j]:
            status[j] = "done"
            continue
        runner = run_rubric if j == "rubric" else run_attempts
        status[j] = runner(args, backend, units[j], stores[j], clock, stats, guard)
        if status[j] != "done":
            break
    compact_all(args.out, stores.values(), run, only_needed=True)     # the saved output: few files
    guard.check(reserve=16)                                            # and room for export/
    man = read_json(os.path.join(args.out, "manifest.json"))
    sessions = man.get("sessions", []) + [{"t0": args.t0, "run": run, "wall_s": round(now() - t_start, 1),
                                           "status": status, "stats": stats, "model": slug, "files": guard.info()}]
    prog = dict((man.get("progress") or {}).get(slug) or {})         # other jobs' progress (other commits) stays
    prog.update({j: {"units": len(units[j]), "done": len({u["unit"] for u in units[j]} & stores[j].done()),
                     **({"scope": args.attempt_scope} if j == "attempts" else {})} for j in args.jobs})
    update_manifest(args.out, sessions=sessions, wall_s_total=round(sum(s["wall_s"] for s in sessions), 1),
                    progress={slug: prog})
    export(args.out)
    all_done = all(status.get(j) == "done" for j in args.jobs)
    if any(v in ("check_failed", "files") for v in status.values()):
        log(f"run: stopped by a check ({status}); see the log above and README before running again")
        return EXIT_CHECK
    log("run: " + ("all jobs done" if all_done else f"stopped with work left ({status}); run again to resume"))
    return 0 if all_done else EXIT_DEADLINE


def main(argv=None, backend=None, tok=None, token_getter=None, downloader=None, now=time.time) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "plan":
        return cmd_plan(args, tok or (backend.tok if backend is not None else None), token_getter, downloader)
    if args.command == "run":
        return cmd_run(args, backend, token_getter, downloader, now)
    if args.command == "split-harness":
        split_harness(args.export_dir or os.path.join(os.getcwd(), "data", "features", "kaggle"))
        return 0
    export(args.out)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except ConfigError as e:                                 # a setup error: the RUN cell does not retry it
        print(e.msg, file=sys.stderr, flush=True)
        code = EXIT_CONFIG
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int) and e.code:
            print(e.code, file=sys.stderr, flush=True)
    except BaseException:                                    # noqa: BLE001 - print it before os._exit
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _kill_children()
        os._exit(code)
