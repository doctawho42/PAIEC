"""kaggle/strong_probe (task K1) without a GPU or a download: the whole pipeline on a
mock generation backend, on synthetic items in measurement-db's layout and, where
data/ holds them, on real local items. Checks prompt building, the digit readout,
attempt parsing and features, the output schema (experiments/strong_llm_eval.py's
check_schema where importable), resuming, restoring an earlier session's shards
(both store layouts), the output's file count against Kaggle's 500-file cap in the
worst case, that nothing written reveals the reference answer, the self-checks,
the token's handling, that the notebook carries the script verbatim, and the
notebook's own cells (install check, retries, process-group cleanup)."""
import dataclasses
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDIR = os.path.join(ROOT, "kaggle", "strong_probe")
DATA = os.environ.get("PAIEC_DATA", os.path.join(ROOT, "data"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SP = _load("strong_probe", os.path.join(KDIR, "strong_probe.py"))
NB = _load("strong_probe_build_notebook", os.path.join(KDIR, "build_notebook.py"))


# --- mocks --------------------------------------------------------------------------------------

class MockTok:
    """Qwen-like: every digit its own token, the chat markers single tokens, and a
    chat template that inserts the empty think block when thinking is off."""
    SPECIAL = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    PAT = re.compile("|".join(map(re.escape, SPECIAL)) + r"|\d|[A-Za-z]+|\s+|[^\sA-Za-z\d]")

    def __init__(self):
        self.vocab, self.inv = {}, []
        self.eos_ids = {self._id("<|im_end|>")}

    def _id(self, t):
        if t not in self.vocab:
            self.vocab[t] = len(self.inv)
            self.inv.append(t)
        return self.vocab[t]

    def enc(self, text):
        return [self._id(t) for t in self.PAT.findall(text)]

    def dec(self, ids):
        return "".join(self.inv[i] for i in ids)

    def chat(self, messages, enable_thinking):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")


P_CHOSEN, P_ALT, P_OTHER = 0.6, 0.3, 0.1      # a value digit, its neighbour, a non-digit token


def rubric_values(prompt_text):
    h = SP.sha(prompt_text)
    return {k: int(h[2 * i:2 * i + 2], 16) % SP.RANGES[k] for i, k in enumerate(SP.RUBRIC_KEYS)}


#: the mock's raw distribution (what RawRecorder would record) against the engine's processed top-k
RAW_LP, RAW_ENT = -0.7, 2.5


class MockBackend:
    """Canned completions by prompt kind: a rubric JSON with known digit
    distributions, attempts that close the thinking with an answer or run out of
    tokens inside it (by seed), forced answers, entropy samples (a short one that
    closes its thinking or a long one cut at max_tokens, by seed) and the entropy
    shards' greedy check. With raw=True (vLLM V0 with RawRecorder) record_raw
    requests also carry the raw distribution's per-token log-prob and entropy,
    distinct from the engine's processed ones."""

    def __init__(self, raw=True):
        self.tok = MockTok()
        self.raw = raw
        self.info = {"backend": "mock", "engine": "mock 0", "logprobs": "mock", "gpus": []}
        self.calls = []
        self.rubric_prompts, self.forced_prompts = [], []
        self.entropy_prompts, self.check_prompts = [], []

    def generate(self, prompts, params, seeds=None):
        self.calls.append({"n": len(prompts), "params": dataclasses.asdict(params),
                           "seeds": None if seeds is None else list(seeds)})
        out = []
        for i, ids in enumerate(prompts):
            text = self.tok.dec(ids)
            if "Rate the demands of this task" in text:
                assert text.endswith("<think>\n\n</think>\n\n" + SP.RUBRIC_PREFILL)   # thinking off
                self.rubric_prompts.append(text)
                s = self.rubric(text)
            elif text.endswith("\\boxed{"):
                # the reasoning and the opener only: never the attempt's own answer
                assert text.rsplit("</think>", 1)[1] == "\n\n**Final Answer**\n\n$\\boxed{"
                assert params.temperature == 0 and params.record_raw
                self.forced_prompts.append(text)
                s = self.forced()
            elif SP.ENTROPY_INSTRUCTION in text:
                # thinking on, the neutral instruction last, nothing after it
                assert text.endswith(SP.ENTROPY_INSTRUCTION + "<|im_end|>\n<|im_start|>assistant\n")
                if params.temperature <= 0:              # the shard's recorder check: greedy, one prompt, no seed
                    assert seeds is None and len(prompts) == 1 and params.record_raw
                    assert params.max_tokens == SP.ENTROPY_CHECK_TOKENS and not params.presence_penalty
                    self.check_prompts.append(text)
                    s = self.entropy_check()
                else:
                    self.entropy_prompts.append(text)
                    s = self.entropy(seeds[i], params.max_tokens)
            else:
                assert text.endswith("<|im_start|>assistant\n")                   # thinking on
                s = self.attempt(seeds[i], params.max_tokens)
            greedy = params.temperature <= 0 and not params.presence_penalty
            s.engine_logprobs = "raw" if greedy else "processed"
            if not (params.record_raw and self.raw):
                s.lp_raw = s.ent_raw = None
            out.append([s])
        return out

    def _sample(self, gen, lp_fn, ent_fn, finish, top_fn=None, raw_fn=None):
        ids = self.tok.enc(gen)
        strs = [self.tok.inv[i] for i in ids]
        lp = np.array([lp_fn(t, s) for t, s in enumerate(strs)], np.float32)
        ent = np.array([ent_fn(t, s) for t, s in enumerate(strs)], np.float32)
        top = [top_fn(t, s) for t, s in enumerate(strs)] if top_fn else None
        raw = raw_fn or (lambda t, s: (RAW_LP - 0.01 * (t % 7), RAW_ENT + 0.02 * (t % 3)))
        lr = np.array([raw(t, s)[0] for t, s in enumerate(strs)], np.float32)
        er = np.array([raw(t, s)[1] for t, s in enumerate(strs)], np.float32)
        return SP.Sample(ids, strs, lp, ent, finish, top, lr, er)

    def rubric(self, prompt_text):
        vals = rubric_values(prompt_text)
        keys = list(SP.RUBRIC_KEYS)
        gen = f"{vals[keys[0]]}" + "".join(f', "{k}": {vals[k]}' for k in keys[1:]) + "}"
        # the value positions, in order: every digit token in gen is a value
        order = iter(keys)
        dig = {}

        def top(t, s):
            if s.isdigit():
                k = next(order)
                v, R = vals[k], SP.RANGES[k]
                dig[t] = k
                return [(s, math.log(P_CHOSEN)), (str((v + 1) % R), math.log(P_ALT)), ("x", math.log(P_OTHER))]
            return [(s, 0.0)]

        return self._sample(gen, lambda t, s: math.log(P_CHOSEN) if s.isdigit() else 0.0,
                            lambda t, s: 0.9 if s.isdigit() else 0.0, "stop", top)

    def attempt(self, seed, max_tokens):
        kind = seed % 3
        if kind < 2:
            ans = 42 if kind == 0 else 41
            gen = f"<think>\nLet me think. 2+2 = 4, so the answer is {ans}.\n</think>\n\n" \
                  f"The answer is $\\boxed{{{ans}}}$.<|im_end|>"
        else:
            gen = "<think>\nHmm, this is long. " + "Try again. " * 200
        ids = self.tok.enc(gen)
        finish = "stop"
        if len(ids) > max_tokens:
            gen, finish = self.tok.dec(ids[:max_tokens]), "length"
        return self._sample(gen, lambda t, s: -0.1 * (1 + t % 3), lambda t, s: 0.5 + 0.01 * (t % 5), finish)

    def forced(self):
        # greedy: the engine's logprobs are the raw distribution's, as the recorder's
        s = self._sample("7}$", lambda t, s: -0.05, lambda t, s: 0.1, "stop", raw_fn=lambda t, s: (-0.05, 0.3))
        s.engine_logprobs = "raw"
        return s

    def entropy(self, seed, max_tokens):
        if seed % 2 == 0:                                   # the thinking closes early
            gen = "<think>\nFirst read the task, then plan the change.\n</think>\n\nI would start there.<|im_end|>"
        else:                                               # still thinking at the cap
            gen = "<think>\nLet me think about the approach. " + "Consider the next step carefully. " * 100
        ids = self.tok.enc(gen)
        finish = "stop"
        if len(ids) > max_tokens:
            gen, finish = self.tok.dec(ids[:max_tokens]), "length"
        return self._sample(gen, lambda t, s: -0.2 * (1 + t % 2), lambda t, s: 0.4 + 0.01 * (t % 4), finish)

    def entropy_check(self):
        # greedy: the engine's logprobs are the raw distribution's, as the recorder's
        s = self._sample("<think>\nOkay, so the task asks", lambda t, s: -0.3, lambda t, s: 0.2, "length",
                         raw_fn=lambda t, s: (-0.3, 1.7))
        s.engine_logprobs = "raw"
        return s


# --- synthetic data in measurement-db's layout ----------------------------------------------------

PROBE = sorted(SP.PROBE_IDS)[:2]


def gc(ans):
    return json.dumps({"reference_answer": ans, "rule": "match"})


def write_synthetic(root):
    items = {
        "matharena": [
            (PROBE[0], "Please reason step by step, and put your final answer within \\boxed{}.\nFind the "
                       "number of ordered pairs of integers with a sum of 42 and a product that is positive.",
             "competition=aime_2025;problem_idx=1", gc("42")),
            (PROBE[1], "Please reason step by step, and put your final answer within \\boxed{}.\nFind the "
                       "number of ordered pairs of integers with a sum of 42 and a product that is positive.",
             "competition=aime_2025_I;problem_idx=1", gc("42")),          # same text, other competition
            ("m3", "Compute the value of the expression 1/2 + 1/3 and give the answer as a reduced fraction "
                   "in lowest terms.", "competition=hmmt_feb_2025;problem_idx=4", gc("\\frac{5}{6}")),
            ("m4", "Let x be a real number such that x^2 = 2 and x is positive. What is x written in "
                   "simplest radical form please?", "competition=hmmt_feb_2025;problem_idx=7", gc("\\sqrt{2}")),
            ("m5", "See image. Which figure comes next in the sequence shown? Choose the right option from "
                   "the five answers given below.", "competition=kangaroo_2025;problem_idx=2", gc("C")),
            ("m6", "Prove that for every positive integer n the number n^3 - n is divisible by six, for all "
                   "cases considered here.", "competition=usamo_2025;problem_idx=1", gc(None)),
            ("m7", "short", "competition=aime_2025;problem_idx=9", gc("3")),
        ],
        "multi_swebench": [
            ("s1", "Fix the crash in the parser when the input file is empty. " * 3, "lang=c++", gc("diff")),
            ("s2", "Add a flag to disable colour output in the CLI. " * 2, "lang=go", gc("diff")),
            ("s3", "Add a flag to disable colour output in the CLI. " * 2, "lang=go", gc("diff")),   # duplicate
        ],
        "real_webagents": [("w1", "Book the cheapest room for two nights in May.", "website=staynb", gc(None))],
        "researchcodebench": [("r1", "\\documentclass{article} " + "word " * 5000 + "\\end{document}",
                               "paper=long-paper", gc("code"))],
    }
    return write_items(root, items)


def write_items(root, items):
    for b, rows in items.items():
        os.makedirs(os.path.join(root, b), exist_ok=True)
        pd.DataFrame({"item_id": [r[0] for r in rows], "benchmark_id": b, "raw_item_id": [r[0] for r in rows],
                      "content": [r[1] for r in rows], "asset_manifest": None,
                      "grading_criterion": [r[3] for r in rows], "verifier": None,
                      "content_hash": [SP.sha(r[1], 32) for r in rows], "item_features": [r[2] for r in rows]}
                     ).to_parquet(os.path.join(root, b, "items.parquet"), index=False)
        pd.DataFrame({"subject_id": ["a", "b"] * len(rows), "item_id": [r[0] for r in rows for _ in range(2)],
                      "response": 1.0}).to_parquet(os.path.join(root, b, "response.parquet"), index=False)
        for t in ("subjects", "benchmarks"):
            pd.DataFrame({"x": [1]}).to_parquet(os.path.join(root, b, f"{t}.parquet"), index=False)
    return items


def write_many(root, n_math=24, n_other=8):
    """Many small units: n_math attempt-eligible matharena texts (the first two probe items) and n_other
    items in each other parent."""
    items = {"matharena": [(PROBE[i] if i < 2 else f"m{i}", f"Compute {i} plus {i} and give the result as an "
                            f"integer; show the reasoning briefly and then the final answer {i}.",
                            f"competition=c{i % 3};problem_idx={i}", gc(str(2 * i))) for i in range(n_math)]}
    for j, b in enumerate(SP.PARENTS[1:]):
        items[b] = [(f"x{j}_{i}", f"Task {i}.{j}: change the program so that it handles case {i} correctly.",
                     f"lang=l{i % 2}", gc(None)) for i in range(n_other)]
    return write_items(root, items)


def run_args(data, out, *extra):
    return ["run", "--data-dir", str(data), "--out", str(out), "--download", "", "--resume-from", "none",
            "--k", "3", "--max-tokens", "64", "--task-tokens", "256", "--attempt-shard", "2", "--rubric-shard", "3",
            "--t0", "0", "--session-hours", "11", *extra]


def clock(t=3600.0):
    return lambda: t


@pytest.fixture
def synth(tmp_path):
    data = tmp_path / "data"
    items = write_synthetic(str(data))
    return data, tmp_path / "out", items


# --- pieces ----------------------------------------------------------------------------------------

def test_keys_match_paiec():
    from paiec import llmfeat as F
    from paiec.predict import item_key
    for item in ({"item_content": "Solve x", "item_features": "k=v", "interactors": "", "benchmark_id": "matharena"},
                 {"item_content": "é\u2063", "item_features": "", "interactors": "", "benchmark_id": ""},
                 {"item_content": None, "item_features": "a", "interactors": "", "benchmark_id": "b"}):
        assert SP.item_key_hex(item) == item_key(item).hex() == F.key_for(item, item["benchmark_id"])
        assert SP.text_key_hex(item) == F.text_key(item)
        assert SP.content_sha256(item) == SP.sha(F.item_text(item))


def test_head_tail_matches_llmfeat():
    from paiec import llmfeat as F
    rng = np.random.default_rng(0)
    for n, m in ((5, 10), (20, 10), (21, 9), (100, 7)):
        ids = rng.integers(0, 50, n).tolist()
        assert SP.head_tail(ids, m, [-1, -2]) == F.head_tail(ids, m, [-1, -2])


def test_hash_definition_matches_consumer():
    try:
        from experiments import strong_llm_eval as S
    except Exception as e:                                   # the consumer is optional here
        pytest.skip(f"experiments/strong_llm_eval.py not importable: {e}")
    assert SP.HASH_DEF == S.HASH_DEF and SP.SCHEMA_VERSION == S.SCHEMA_VERSION
    for name, (sign, _) in SP.RUBRIC_EXPORT.items():
        assert S.declared_sign(name)[0] == sign
    for name in SP.RUBRIC_DIAG:
        assert S.declared_sign(name, {name: 0})[0] == 0


def test_prompts_are_benchmark_agnostic_and_built_exactly():
    SP.check_prompts_agnostic()
    tok = MockTok()
    p = SP.rubric_prompt(tok, 64)
    for b in SP.BENCHES + ("mmdocrag",):
        assert b not in p.rendered.lower()
    assert p.rendered.endswith("<think>\n\n</think>\n\n" + SP.RUBRIC_PREFILL)
    for k in SP.RUBRIC_KEYS:
        assert f'"{k}"' in SP.SYSTEM_RUBRIC
    ids, info = p.build("short task", SP.rubric_meta("lang=go"))
    text = tok.dec(ids)
    assert "<task>\nshort task\n</task>\nMetadata: lang=go\n\nRate the demands" in text
    assert not info["truncated"] and info["task_tokens"] == len(tok.enc("short task"))
    long = " ".join(f"w{i}" for i in range(500))
    ids, info = p.build(long, "")
    text = tok.dec(ids)
    assert info["truncated"] and SP.MARKER in text and "w0 " in text and "w499" in text and "w250" not in text
    task = text.split("<task>\n")[1].split("\n</task>")[0]
    assert len(tok.enc(task)) == 64
    assert "Metadata:" not in text.split("<task>")[1] and SP.rubric_meta("  ") == ""
    # position fields never reach the rubric (matharena's problem_idx: a within-competition difficulty cue)
    assert SP.rubric_meta("competition=aime_2025;problem_idx=2;prompt_source=user_message") == \
        "Metadata: competition=aime_2025;prompt_source=user_message\n"
    assert SP.rubric_meta("problem_idx=3") == "" and SP.rubric_meta("lang=c++") == "Metadata: lang=c++\n"
    assert SP.rubric_meta("competition=x;problem_idx=2", with_meta=False) == ""
    a = SP.attempt_prompt(tok, 64)
    ids, _ = a.build(SP.attempt_task("What is 1+1?"))
    text = tok.dec(ids)
    assert text.endswith("<|im_start|>assistant\n") and SP.BOXED_INSTRUCTION in text
    assert SP.attempt_task("x \\boxed{} y") == "x \\boxed{} y"


def test_digit_readout():
    be = MockBackend()
    prompt = "any prompt"
    s = be.rubric(prompt)
    vals = rubric_values(prompt)
    rd = SP.read_rubric(s)
    assert rd["parse_ok"]
    for k in SP.RUBRIC_KEYS:
        R, v = SP.RANGES[k], vals[k]
        alt = (v + 1) % R
        assert rd[f"{k}_mode"] == v
        assert rd[k] == pytest.approx((P_CHOSEN * v + P_ALT * alt) / (P_CHOSEN + P_ALT))
        assert rd[f"{k}_mass"] == pytest.approx(P_CHOSEN + P_ALT)
        q = np.array([P_CHOSEN, P_ALT]) / (P_CHOSEN + P_ALT)
        assert rd[f"{k}_entropy"] == pytest.approx(float(-(q * np.log(q)).sum()))
        assert len(rd[f"{k}_probs"]) == R and sum(rd[f"{k}_probs"]) == pytest.approx(1)
    ht = np.array(rd["human_time_probs"])
    assert rd["time_log_minutes"] == pytest.approx(float(ht @ np.log(SP.HUMAN_MID)))
    df = np.array(rd["difficulty_probs"])
    assert rd["solve_share"] == pytest.approx(1 - float(df @ ((np.arange(10) + 0.5) / 10)))
    # a missing key, and an out-of-range digit
    bad = SP.Sample([0, 1], ["3", ', "knowledge": 7}'], np.zeros(2), np.zeros(2), "stop", [[("3", 0.0)], []])
    rd = SP.read_rubric(bad)
    assert not rd["parse_ok"] and rd["reasoning"] == 3 and math.isnan(rd["work"]) and math.isnan(rd["solve_share"])


def test_topk_entropy():
    p = np.array([0.5, 0.25, 0.25])
    assert SP.topk_entropy(np.log(p)) == pytest.approx(float(-(p * np.log(p)).sum()))
    assert SP.topk_entropy(np.log([0.5, 0.25])) == pytest.approx(float(-(p * np.log(p))[:2].sum() - 0.25 * math.log(0.25)))
    assert SP.topk_entropy([0.0, -np.inf]) == pytest.approx(0.0)
    assert math.isnan(SP.topk_entropy([]))


def test_boxed_and_canon():
    assert SP.boxed("a \\boxed{1} b \\boxed{\\frac{1}{2}} c") == ("\\frac{1}{2}", (21, 32))
    assert SP.boxed("\\boxed{12") == (None, None)
    assert SP.boxed("\\boxed{" + "7}$", start=7)[0] == "7"
    assert SP.canon("042") == SP.canon("42") == "num:42"
    assert SP.canon("\\frac{5}{6}") == SP.canon("\\dfrac{5}{6}") == SP.canon("5/6")
    assert SP.canon("1,000") == "num:1000" and SP.canon(None) is None and SP.canon("  ") is None


def test_attempt_analysis_and_aggregate():
    be = MockBackend()
    closed, trunc = be.attempt(0, 4096), be.attempt(2, 30)
    a = SP.analyse_attempt(closed)                           # no forced readout (--no-force)
    assert a["closed"] and a["answer"] == "42" and a["ans_natural"] == "42" and not a["forced"]
    assert a["n_think"] < a["n_tokens"] and a["finish_reason"] == "stop" and not a["capped"]
    assert a["lp_answer"] == pytest.approx(a["lp_boxed"]) and np.isfinite(a["lp_boxed"])
    # the token statistics are the raw distribution's; the engine's are kept as diagnostics
    assert a["stats_source"] == SP.STATS_RAW
    assert a["tok_lp"] == pytest.approx(float(np.mean(closed.lp_raw)))
    assert a["tok_entropy"] == pytest.approx(float(np.mean(closed.ent_raw)))
    assert a["tok_lp_engine"] == pytest.approx(float(np.mean(closed.lp)))
    assert a["tok_entropy_engine"] == pytest.approx(float(np.mean(closed.ent)))
    assert a["ent_first256"] == pytest.approx(float(np.mean(closed.ent_raw[:256])))
    assert np.allclose(SP.unf16(a["ent_seq"]), closed.ent_raw, atol=1e-2)
    assert np.allclose(SP.unf16(a["ent_engine_seq"]), closed.ent, atol=1e-3)
    ends = np.cumsum([len(x) for x in closed.token_strs])
    c = closed.text.rfind("\\boxed{") + 7
    t0, t1 = (int(np.searchsorted(ends, x, side="right")) for x in (c, c + 1))    # "4", "2": one token each
    assert t1 == t0 + 1 and closed.token_strs[t0] + closed.token_strs[t1] == "42"
    assert a["lp_boxed"] == pytest.approx(float(np.mean(closed.lp_raw[t0:t1 + 1])))   # the raw log-probs
    # with the forced readout (every attempt gets one), lp_answer is the forced answer's, also when it boxed its own
    a2 = SP.analyse_attempt(closed, be.forced())
    assert a2["answer"] == "42" and a2["ans_forced"] == "7" and not a2["forced"]
    assert a2["lp_answer"] == pytest.approx(-0.05) == a2["lp_forced"] and a2["lp_boxed"] == a["lp_boxed"]
    assert a2["recorder_gap"] == pytest.approx(0.0) and a2["recorder_gap_n"] == 3
    b = SP.analyse_attempt(trunc, be.forced())
    assert not b["closed"] and b["capped"] and b["forced"] and b["answer"] == "7" and b["ans_natural"] is None
    assert b["n_think"] == b["n_tokens"] == 30 and math.isnan(b["tok_lp_answer"])
    assert b["lp_answer"] == pytest.approx(-0.05)
    # without the recorder the engine's statistics stand in, and say what they are
    proc = dataclasses.replace(closed, lp_raw=None, ent_raw=None, engine_logprobs="processed")
    p = SP.analyse_attempt(proc)
    assert p["stats_source"] == SP.STATS_PROC and p["tok_entropy"] == pytest.approx(float(np.mean(closed.ent)))
    assert SP.analyse_attempt(dataclasses.replace(proc, engine_logprobs="raw"))["stats_source"] == SP.STATS_RAW_TOPK
    agg = SP.attempt_aggregate([a, a, b], ["42"])
    assert agg["top_share"] == pytest.approx(2 / 3) and agg["n_distinct"] == 2
    assert agg["graded"] == pytest.approx(2 / 3) and agg["top_correct"] == 1.0
    assert agg["closed_rate"] == pytest.approx(2 / 3) and agg["forced_rate"] == pytest.approx(1 / 3)
    assert agg["ans_entropy"] == pytest.approx(-(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)))
    assert agg["correct"] == [True, True, False]
    none = SP.attempt_aggregate([dict(b, answer=None, forced=False)] * 2, [])
    assert none["top_share"] == 0 and none["fail_rate"] == 1 and math.isnan(none["graded"])


def test_units_on_synthetic_items(synth):
    data, _, items = synth
    ur = SP.rubric_units(str(data), SP.PARENTS)
    assert len(ur) == sum(len(v) for v in items.values()) - 1            # s2 and s3 are one item
    dup = [u for u in ur if len(u["item_ids"]) > 1]
    assert len(dup) == 1 and dup[0]["item_ids"] == ["s2", "s3"]
    ua = SP.attempt_units(str(data))
    # eligible: the probe pair (one text), m3, m4; not the image, the proof, the short one
    assert sorted(sum((u["item_ids"] for u in ua), [])) == sorted(PROBE + ["m3", "m4"])
    assert len(ua) == 3 and ua[0]["probe"] and ua[0]["item_ids"] == PROBE
    assert ua[0]["comps"] == ["aime_2025", "aime_2025_I"] and ua[0]["golds"] == ["42"]
    assert [u["unit"] for u in SP.attempt_units(str(data), "probe")] == [ua[0]["unit"]]
    assert [u["unit"] for u in SP.attempt_units(str(data), "rest")] == [u["unit"] for u in ua[1:]]


# --- the pipeline ---------------------------------------------------------------------------------

def test_pipeline_end_to_end(synth):
    data, out, items = synth
    be = MockBackend()
    rc = SP.main(run_args(data, out), backend=be, now=clock())
    assert rc == 0
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    ex = out / slug / "export"
    man = json.load(open(ex / "manifest.json"))
    assert man["schema_version"] == 1 and man["hash"] == SP.HASH_DEF
    assert man["model"]["repo"] == SP.DEFAULT_MODEL and man["model"]["quantization"] == "awq"
    assert {s["path"] for s in man["shards"]} == {"rubric/rubric.parquet", "attempts/attempts.parquet"}
    for s in man["shards"]:
        assert SP.file_sha256(str(ex / s["path"])) == s["sha256"]
    ru = pd.read_parquet(ex / "rubric" / "rubric.parquet")
    all_ids = {(b, r[0]) for b, rows in items.items() for r in rows}
    assert set(zip(ru.benchmark, ru.item_id)) == all_ids and len(ru) == len(all_ids)
    for c in list(SP.RUBRIC_EXPORT) + SP.RUBRIC_DIAG:
        assert c in ru.columns and ru[c].dtype == float
    assert ru.rubric_reasoning.between(0, 5).all() and ru.solve_share.between(0, 1).all()
    assert (ru.loc[ru.benchmark == "researchcodebench", "truncated_n"] == 1).all()
    by_id = {(b, r[0]): r for b, rows in items.items() for r in rows}
    for row in ru.itertuples():
        _, content, feats, _ = by_id[(row.benchmark, row.item_id)]
        assert row.content_sha256 == SP.sha(f"{content}\n{feats}")
    at = pd.read_parquet(ex / "attempts" / "attempts.parquet")
    assert len(at) == 4 * 3 and set(at.item_id) == set(PROBE + ["m3", "m4"])
    assert at.attempt.dtype == np.int64 and at.capped.dtype == bool and set(at.design) == {"cot"}
    assert at.groupby("item_id").attempt.apply(lambda a: sorted(a) == [0, 1, 2]).all()
    # the probe pair shares its attempts, with each item's own content hash
    p0, p1 = (at[at.item_id == i].sort_values("attempt") for i in PROBE)
    assert list(p0.answer) == list(p1.answer) and p0.content_sha256.iloc[0] != p1.content_sha256.iloc[0]
    assert set(at.answer.dropna()) <= {"42", "41", "7"} and at.forced.any() and (~at.forced).any()
    keys = pd.read_parquet(ex / "_keys.parquet")
    assert set(keys.job) == {"rubric", "attempts"}
    for r in keys.itertuples():
        _, content, feats, _ = by_id[(r.benchmark, r.item_id)]
        item = {"item_content": content, "item_features": feats, "interactors": "", "benchmark_id": r.benchmark}
        assert r.key == SP.item_key_hex(item) and r.text_key == SP.text_key_hex(item)
    hj = json.load(open(ex / "_harness.json"))                # one file on Kaggle (the 500-file cap)
    h = hj["rubric_solve_share"]
    assert set(h) == {i for _, i in all_ids} and not (ex / "_harness").exists()
    assert h["m3"] == pytest.approx(-float(ru.loc[ru.item_id == "m3", "solve_share"].iloc[0]))
    assert set(hj["attempts_tok_entropy"]) == set(PROBE + ["m3", "m4"])
    # split-harness (local) writes the one-covariate files experiments/harness.py --cov reads
    assert SP.main(["split-harness", "--export-dir", str(ex)]) == 0
    assert json.load(open(ex / "_harness" / "rubric_solve_share.json")) == h
    assert len(os.listdir(ex / "_harness")) == len(hj)
    # the exported token statistics are the raw distribution's (D2's definition), and the manifest says so
    assert at.tok_entropy.between(RAW_ENT - 1e-3, RAW_ENT + 0.05).all() and at.tok_lp.between(RAW_LP - 0.07, RAW_LP).all()
    assert np.allclose(at.lp_answer, -0.05)                  # the forced greedy readout's, for every attempt
    assert man["logprobs"] == "raw" and man["logprobs_engine"] == be.info["logprobs"]
    ts = man["kinds"]["attempts"]["token_stats"]
    assert ts["comparable_to_d2"] and ts["sources"] == {SP.STATS_RAW: 3 * 3}       # 3 units x k=3 attempts
    assert ts["recorder_check"]["status"] == "ok" and ts["recorder_check"]["tokens"] > 0
    det = pd.read_parquet(ex / "_detail" / "attempt_samples.parquet")
    assert (det.tok_entropy_engine < 1).all() and (det.tok_entropy > 2).all()
    # every attempt got a forced readout, after its reasoning only; the rubric never saw problem_idx
    assert len(be.forced_prompts) == 3 * 3 and not any("The answer is" in t.rsplit("</think>", 1)[1]
                                                        for t in be.forced_prompts)
    assert be.rubric_prompts and not any("problem_idx" in t for t in be.rubric_prompts)
    assert any("Metadata: competition=hmmt_feb_2025\n" in t for t in be.rubric_prompts)
    # no item text, no reference answer and no correctness in any output (parquet read back: its bytes are
    # compressed): the model's answers are there (a correct one equals the reference), nothing says which
    texts = []
    for d, _, fs in os.walk(out):
        for f in fs:
            path = os.path.join(d, f)
            if f.endswith(".parquet"):
                df = pd.read_parquet(path)
                assert not set(df.columns) & set(SP.LEAK_COLS), (f, set(df.columns) & set(SP.LEAK_COLS))
                texts.append(df.apply(lambda col: col.map(str)).to_csv())
            else:
                texts.append(open(path, encoding="utf-8").read())
    assert "graded" not in json.load(open(ex / "_summary.json"))["attempts"]
    det = pd.read_parquet(ex / "_detail" / "attempt_samples.parquet")
    assert list(det.canon_sha) == [SP.canon_sha(SP.canon(a)) for a in det.answer] and det.canon_sha.notna().any()
    top = json.load(open(out / "manifest.json"))
    assert np.isfinite(top["sessions"][-1]["stats"]["attempts"]["graded"])     # a session mean, in memory only
    blob = "\n".join(texts)
    for rows in items.values():
        for _, content, _, _ in rows:
            if len(content) > 40:                            # the tail: the head can be our own instruction
                assert content[-40:] not in blob
    assert "frac{5}{6}" not in blob and "sqrt{2}" not in blob
    # prompts and sampling as configured
    att = [c for c in be.calls if c["seeds"] is not None]
    assert all(c["params"]["temperature"] == 0.6 and c["params"]["top_k"] == 20 and c["params"]["logprobs"] == 5
               and c["params"]["presence_penalty"] == 1.5 and c["params"]["max_tokens"] == 64 for c in att)
    seeds = [s for c in att for s in c["seeds"]]
    assert len(seeds) == len(set(seeds)) == 3 * 3
    rub = [c for c in be.calls if c["params"]["keep_top"]]
    assert all(c["params"]["temperature"] == 0 and c["params"]["logprobs"] == 20 for c in rub)
    top = json.load(open(out / "manifest.json"))
    assert top["progress"][slug]["rubric"] == {"units": len(all_ids) - 1, "done": len(all_ids) - 1}


def test_resume_and_config_change(synth):
    data, out, _ = synth
    be = MockBackend()
    rc = SP.main(run_args(data, out, "--max-shards", "1"), backend=be, now=clock())
    assert rc == SP.EXIT_DEADLINE
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    rub = SP.Store(str(out / slug / "rubric"), "").read()
    assert len(rub) == 3                                     # one shard of --rubric-shard 3
    be2 = MockBackend()
    rc = SP.main(run_args(data, out), backend=be2, now=clock())
    assert rc == 0
    rub = SP.Store(str(out / slug / "rubric"), "").read()
    n_units = len(SP.rubric_units(str(data), SP.PARENTS))
    assert len(rub) == n_units and rub.unit.is_unique
    done_first = sum(c["n"] for c in be.calls if c["params"]["keep_top"])
    assert sum(c["n"] for c in be2.calls if c["params"]["keep_top"]) == n_units - done_first
    # nothing left: a third run generates nothing
    be3 = MockBackend()
    assert SP.main(run_args(data, out), backend=be3, now=clock()) == 0 and not be3.calls
    # another k is another config: its attempts start over, the rubric is kept
    be4 = MockBackend()
    assert SP.main(run_args(data, out, "--k", "2"), backend=be4, now=clock()) == 0
    assert all(c["seeds"] is not None for c in be4.calls if c["params"]["temperature"] > 0)
    assert sum(len(c["seeds"]) for c in be4.calls if c["seeds"]) == 2 * 3
    at = pd.read_parquet(out / slug / "export" / "attempts" / "attempts.parquet")
    assert set(at.attempt) == {0, 1}                        # the export takes the larger config (k=2 = 3 units)


def test_deadline_stops_before_generating(synth):
    data, out, _ = synth
    be = MockBackend()
    rc = SP.main(run_args(data, out), backend=be, now=clock(11 * 3600 - 100))
    assert rc == SP.EXIT_DEADLINE and not be.calls


def test_restore_from_an_earlier_session(synth, tmp_path):
    data, out, _ = synth
    SP.main(run_args(data, out, "--max-shards", "1"), backend=MockBackend(), now=clock())
    new = tmp_path / "session2"
    be = MockBackend()
    args = run_args(data, new)
    args[args.index("--resume-from") + 1] = str(out)
    rc = SP.main(args, backend=be, now=clock())
    assert rc == 0
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    assert set(os.listdir(out / slug / "rubric")) <= set(os.listdir(new / slug / "rubric"))     # same names
    n_units = len(SP.rubric_units(str(data), SP.PARENTS))
    assert sum(c["n"] for c in be.calls if c["params"]["keep_top"]) == n_units - 3
    # restoring twice copies nothing new; the first session's history carried over
    assert SP.restore(str(new), [str(out)]) == 0
    assert len(json.load(open(new / "manifest.json"))["sessions"]) == 2


def test_session_history_survives_plan_before_restore(synth, tmp_path):
    """The notebook runs plan (which writes the manifest) before run restores: the
    earlier sessions and their wall time must still carry over."""
    data, out, _ = synth
    SP.main(run_args(data, out, "--max-shards", "1"), backend=MockBackend(), now=clock())
    man1 = json.load(open(out / "manifest.json"))
    assert len(man1["sessions"]) == 1
    new = tmp_path / "session2"
    plan = ["plan"] + run_args(data, new)[1:]
    assert SP.main(plan, tok=MockTok()) == 0 and (new / "manifest.json").exists()
    args = run_args(data, new)
    args[args.index("--resume-from") + 1] = str(out)
    assert SP.main(args, backend=MockBackend(), now=clock()) == 0
    man = json.load(open(new / "manifest.json"))
    assert len(man["sessions"]) == 2 and man["sessions"][0] == man1["sessions"][0]
    assert man["wall_s_total"] == pytest.approx(sum(s["wall_s"] for s in man["sessions"]))
    assert "plan" in man and SP.model_slug(SP.DEFAULT_MODEL, "vllm") in man["models"]
    ex = json.load(open(new / SP.model_slug(SP.DEFAULT_MODEL, "vllm") / "export" / "manifest.json"))
    assert ex["sessions"] == 2 and ex["wall_s"] == man["wall_s_total"]
    # a third session restoring from the second (which holds both) and the first: no duplicates
    third = tmp_path / "session3"
    SP.restore(str(third), [str(out), str(new)])
    assert len(json.load(open(third / "manifest.json"))["sessions"]) == 2


def test_resume_sources_at_any_depth(tmp_path):
    for d in ("a/strong_probe", "b/c/d/strong_probe", "e/strong_probe_not"):
        os.makedirs(tmp_path / d)
        (tmp_path / d / "manifest.json").write_text("{}")
    os.makedirs(tmp_path / "f" / "strong_probe")                    # no manifest: not a source
    got = SP.resume_sources("auto", root=str(tmp_path))
    assert got == sorted([str(tmp_path / "a" / "strong_probe"), str(tmp_path / "b" / "c" / "d" / "strong_probe")])
    assert SP.resume_sources("none") == [] and SP.resume_sources("x,y") == ["x", "y"]


def test_export_passes_the_consumers_schema_check(synth):
    try:
        from experiments import strong_llm_eval as S
    except Exception as e:
        pytest.skip(f"experiments/strong_llm_eval.py not importable: {e}")
    data, out, _ = synth
    SP.main(run_args(data, out), backend=MockBackend(), now=clock())
    ex = out / SP.model_slug(SP.DEFAULT_MODEL, "vllm") / "export"
    rep = S.check_schema(str(ex), benches=SP.BENCHES)
    assert rep["ok"], rep["errors"]
    assert not rep["warnings"], rep["warnings"]


def test_token_goes_to_the_downloader_only(tmp_path, capsys, monkeypatch):
    secret = "hf_" + "Zq9" * 11
    monkeypatch.delenv("HF_TOKEN", raising=False)
    seen = []

    def fake_download(repo, filename, repo_type, local_dir, token):
        seen.append((repo, filename, repo_type, token))
        b, t = filename.split("/")
        os.makedirs(os.path.join(local_dir, b), exist_ok=True)
        pd.DataFrame({"x": [1]}).to_parquet(os.path.join(local_dir, b, t), index=False)

    n = SP.download_data(str(tmp_path / "d"), ["matharena"], token_getter=lambda: secret, downloader=fake_download)
    assert n == 4 and {s[3] for s in seen} == {secret} and {s[0] for s in seen} == {SP.DATA_REPO}
    assert {s[1] for s in seen} == {f"matharena/{t}.parquet" for t in SP.TABLES}
    assert SP.download_data(str(tmp_path / "d"), ["matharena"], token_getter=lambda: 1 / 0) == 0   # nothing missing

    def failing(repo, filename, repo_type, local_dir, token):
        raise RuntimeError(f"401 Client Error for {filename} with token {token}")

    with pytest.raises(SystemExit) as e:
        SP.download_data(str(tmp_path / "e"), ["matharena"], token_getter=lambda: secret, downloader=failing)
    assert secret not in str(e.value) and "***" in str(e.value)
    cap = capsys.readouterr()
    assert secret not in cap.out + cap.err and secret not in json.dumps(dict(os.environ))
    for d, _, fs in os.walk(tmp_path):
        for f in fs:
            assert secret.encode() not in open(os.path.join(d, f), "rb").read()


def test_budget(synth):
    data, out, _ = synth
    args = SP.parse_args(run_args(data, out))
    ur, ua = SP.load_units(args)
    b = SP.plan_budget(args, MockTok(), ur, ua)
    r = SP.rates(args)
    assert b["rubric"]["units"] == len(ur) and b["attempts"]["units"] == len(ua) == 3
    assert b["rubric"]["hours_est"] == pytest.approx(
        SP.rubric_secs(b["rubric"]["prompt_tokens_uncached"], len(ur), r) / 3600, abs=0.01)
    assert b["rubric"]["prompt_tokens_uncached"] < b["rubric"]["prompt_tokens"]      # the shared prefix
    assert b["attempts"]["gen_tokens_est"] == int(3 * 3 * 64 * SP.ATTEMPT_GEN_FRAC)
    assert b["fits_weekly_quota"] and b["sessions_est"] == 1
    # every attempt gets a forced readout; the slow case halves the decode rate
    assert b["attempts"]["forced_readouts"] == 3 * 3
    rs = {**r, "decode": r["decode"] * SP.SLOW_DECODE, "decode_short": r["decode_short"] * SP.SLOW_DECODE}
    assert b["attempts"]["hours_est_slow"] == pytest.approx(
        SP.attempt_secs(3, b["attempts"]["prompt_tokens"], 3, 64, True, rs) / 3600, abs=0.01)
    assert b["slow_case"]["total_hours_est"] >= b["total_hours_est"] and b["slow_case"]["fits_weekly_quota"]
    assert b["attempts"]["shard_kv_tokens"] == int(2 * 3 * (b["attempts"]["prompt_tokens"] / 3 + 64))
    assert SP.main(["plan"] + run_args(data, out)[1:], tok=MockTok()) == 0
    assert "plan" in json.load(open(out / "manifest.json"))
    # the default shard at full length fits the 14B's estimated KV cache (no preemption by recompute)
    d = SP.parse_args(["run"])
    assert d.attempt_shard == 5 and d.attempt_shard * d.k * (200 + d.max_tokens) <= SP.kv_tokens(d)


def test_kv_estimate_and_rates():
    args = SP.parse_args(["run"])
    assert args.model == SP.DEFAULT_MODEL and args.quantization == "awq" and args.tp == 2
    kv = SP.kv_tokens(args)
    assert 60_000 < kv < 130_000                  # 14B-AWQ on 2x T4: room for ~20 attempts of 4.4k tokens
    hf = SP.parse_args(["run", "--backend", "hf"])
    assert hf.model == SP.HF_FALLBACK_MODEL and hf.quantization is None and hf.jobs == ["rubric"]


def test_hf_fallback_bookkeeping():
    """HFBackend.generate on a fake model (no weights): tokens, strings, logprobs,
    top-k, the stop string and the finish reason line up with the readout."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    tok = MockTok()
    tok.enc("0123456789")
    eos = next(iter(tok.eos_ids))
    cont = tok.enc('3, "knowledge": 2}') + [eos]
    V = len(tok.inv)

    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, input_ids, attention_mask, max_new_tokens, **kw):
            B, T = input_ids.shape[0], max_new_tokens
            seq = (cont + [0] * T)[:T]
            logits = []
            for t in range(T):
                x = torch.full((B, V), -5.0)
                x[:, seq[t]] = 5.0
                if tok.inv[seq[t]].isdigit():
                    x[:, tok.vocab[str((int(tok.inv[seq[t]]) + 1) % 6)]] = 4.0
                logits.append(x)
            return SimpleNamespace(sequences=torch.cat([input_ids, torch.tensor([seq] * B)], 1), logits=tuple(logits))

    be = SP.HFBackend.__new__(SP.HFBackend)
    be.model, be.tok, be.pad, be.max_batch_tokens, be.max_batch = FakeModel(), tok, 0, 10_000, 4
    out = be.generate([tok.enc("a b c"), tok.enc("a")], SP.GenParams(max_tokens=16, logprobs=5, stop=("}",),
                                                                     keep_top=True))
    for (s,) in out:
        assert s.text == '3, "knowledge": 2}' and s.finish_reason == "stop" and len(s.lp) == len(s.token_ids)
        ls = torch.log_softmax(FakeModel().generate(torch.zeros((1, 1), dtype=torch.long), None, 16).logits[0][0], -1)
        assert s.lp[0] == pytest.approx(float(ls[tok.vocab["3"]]), abs=1e-5)
        rd = SP.read_rubric(s)
        p = torch.softmax(ls[[tok.vocab[str(d)] for d in range(6)]], -1).numpy()
        assert rd["reasoning"] == pytest.approx(float(p @ np.arange(6)), abs=1e-4)   # top-5 holds 3, 4 and three others
        assert rd["knowledge_mode"] == 2 and not rd["parse_ok"]
    with pytest.raises(NotImplementedError):
        be.generate([[1]], SP.GenParams(temperature=0.6))


def _v0_run(rec, rows, tokens, extra_step=False, preempt_at=None):
    """Drive a RawRecorder the way vLLM V0 does: one call per generated token,
    under torch.inference_mode (V0's model runner), with the raw logits row and the
    output tokens so far; optionally V0's extra step after the stop (asynchronous
    output processing) and a recompute after a preemption (the step at preempt_at
    runs twice). finish() then runs outside inference mode, as in generate()."""
    import torch
    with torch.inference_mode():
        for t in range(len(tokens)):
            for _ in range(2 if t == preempt_at else 1):
                row = rows[t].clone()
                got = rec(tuple(tokens[:t]), row)
                assert got is row and (got == rows[t]).all()         # the logits pass through untouched
        if extra_step:
            rec(tuple(tokens), rows[len(tokens)].clone())
    return rec.finish(tokens)


def test_raw_recorder_matches_the_raw_distribution():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    V, T = 50, 12
    rows = [torch.randn(V, generator=g, dtype=torch.float32).half() * 3 for _ in range(T + 1)]
    tokens = [int(torch.randint(V, (1,), generator=g)) for _ in range(T)]
    ls = [torch.log_softmax(r.float(), -1) for r in rows]
    want_lp = np.array([float(ls[t][tokens[t]]) for t in range(T)])
    want_ent = np.array([float(-(ls[t].exp() * ls[t]).sum()) for t in range(T)])
    for kw in ({}, {"extra_step": True}, {"preempt_at": 5}, {"extra_step": True, "preempt_at": 0}):
        lp, ent = _v0_run(SP.RawRecorder(4), rows, tokens, **kw)          # cap 4 < T: the buffers grow
        assert lp.shape == ent.shape == (T,)
        assert np.allclose(lp, want_lp, atol=1e-5) and np.allclose(ent, want_ent, atol=1e-5), kw
    assert SP.RawRecorder(8).finish([1, 2]) == (None, None)              # never called: no statistics


class _Logprob:
    def __init__(self, logprob, rank, decoded_token):
        self.logprob, self.rank, self.decoded_token = logprob, rank, decoded_token


class FakeV0LLM:
    """vLLM V0's LLM.generate in miniature: per request and step, a raw logits row
    goes through the request's logits processors, then penalties / temperature /
    top-k give the processed distribution the engine's logprobs come from; the
    token is its argmax (deterministic), output as CompletionOutput-like objects."""

    def __init__(self, tok, V=40, steps=9):
        self.tok, self.V, self.steps, self.seen = tok, V, steps, []

    def generate(self, prompts, sps, use_tqdm=False):
        import torch
        from types import SimpleNamespace
        outs = []
        for p, sp in zip(prompts, sps):
            self.seen.append(sp)
            g = torch.Generator().manual_seed(len(p["prompt_token_ids"]))
            toks, lps = [], []
            for t in range(min(self.steps, sp.max_tokens)):
                raw = torch.randn(self.V, generator=g).half() * 2
                row = raw.clone()
                with torch.inference_mode():                         # V0's model runner
                    for f in sp.logits_processors or []:
                        row = f(tuple(toks), row)
                x = row.float()
                if sp.presence_penalty:
                    x[list(set(toks))] -= sp.presence_penalty
                if sp.temperature > 0:
                    x = x / sp.temperature
                    x[x < x.topk(sp.top_k).values[-1]] = -float("inf")
                ls = torch.log_softmax(x, -1)
                tid = int(ls.argmax())
                top = ls.topk(sp.logprobs or 1)
                d = {int(i): _Logprob(float(v), r + 1, self.tok.dec([int(i)]))
                     for r, (v, i) in enumerate(zip(top.values, top.indices))}
                d.setdefault(tid, _Logprob(float(ls[tid]), 99, self.tok.dec([tid])))
                toks.append(tid)
                lps.append(d)
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=toks, logprobs=lps, finish_reason="length")]))
        return outs


def test_vllm_backend_attaches_the_recorder(monkeypatch):
    """VLLMBackend.generate on a fake V0 engine: every record_raw request of n=1
    gets its own RawRecorder as a logits processor, the Sample's raw statistics
    are the raw rows' (not the engine's processed ones), a greedy request's engine
    logprobs agree with the recorder (the run's recorder check), and V1 gets none."""
    torch = pytest.importorskip("torch")
    from types import ModuleType
    fake = ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw
            for k, v in dict(n=1, temperature=1.0, top_k=-1, presence_penalty=0.0, logprobs=None,
                             logits_processors=None).items():
                setattr(self, k, kw.get(k, v))
            self.max_tokens = kw["max_tokens"]

    fake.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)
    tok = MockTok()
    tok.enc(" ".join("x" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(60)))     # > 40 distinct tokens
    be = SP.VLLMBackend.__new__(SP.VLLMBackend)
    be.llm, be.tok, be.v1 = FakeV0LLM(tok), tok, False
    sampled = SP.GenParams(max_tokens=64, logprobs=5, record_raw=True, **SP.ATTEMPT_SAMPLING)
    out = be.generate([[1, 2, 3], [4, 5]], sampled, seeds=[7, 8])
    recs = [sp.logits_processors[0] for sp in be.llm.seen]
    assert len({id(r) for r in recs}) == 2 and all(isinstance(r, SP.RawRecorder) for r in recs)
    assert [sp.kw["seed"] for sp in be.llm.seen] == [7, 8] and be.llm.seen[0].kw["presence_penalty"] == 1.5
    for (s,), p in zip(out, ([1, 2, 3], [4, 5])):
        g = torch.Generator().manual_seed(len(p))
        raw = [torch.log_softmax((torch.randn(40, generator=g).half() * 2).float(), -1) for _ in s.token_ids]
        assert s.stats_source == SP.STATS_RAW and s.engine_logprobs == "processed"
        assert np.allclose(s.lp_raw, [float(r[t]) for r, t in zip(raw, s.token_ids)], atol=1e-5)
        assert np.allclose(s.ent_raw, [float(-(r.exp() * r).sum()) for r in raw], atol=1e-5)
        assert not np.allclose(s.lp, s.lp_raw, atol=1e-3)                  # processed: temperature 0.6, top-k
    greedy = SP.GenParams(temperature=0.0, max_tokens=64, logprobs=1, record_raw=True)
    (fs,), = be.generate([[9, 9]], greedy)
    gap, n = SP.recorder_gap(fs)
    assert fs.engine_logprobs == "raw" and n == len(fs.token_ids) and gap < 1e-4
    # the rubric does not ask for it; V1 has no per-request logits processors
    be.llm.seen.clear()
    be.generate([[1]], SP.GenParams(temperature=0.0, max_tokens=4, logprobs=20, keep_top=True))
    be.v1 = True
    (v1s,), = be.generate([[1]], sampled, seeds=[1])
    assert all(sp.logits_processors is None for sp in be.llm.seen)
    assert v1s.lp_raw is None and v1s.engine_logprobs == "raw" and v1s.stats_source == SP.STATS_RAW_TOPK


def test_export_without_the_recorder_says_so(synth):
    """Without the recorder (vLLM V1, or a V0 that did not call it) the engine's
    top-k statistics are exported and the manifest marks them as not D2's."""
    data, out, _ = synth
    SP.main(run_args(data, out, "--jobs", "attempts"), backend=MockBackend(raw=False), now=clock())
    man = json.load(open(out / SP.model_slug(SP.DEFAULT_MODEL, "vllm") / "export" / "manifest.json"))
    ts = man["kinds"]["attempts"]["token_stats"]
    assert man["logprobs"] == SP.STATS_PROC and not ts["comparable_to_d2"] and set(ts["definitions"]) == {SP.STATS_PROC}
    assert "not comparable with D2" in man["kinds"]["attempts"]["entropy"]
    assert ts["recorder_check"]["status"] == "n/a"
    assert SP.logprobs_kind({SP.STATS_RAW: 3, SP.STATS_PROC: 1}) == "mixed" and SP.logprobs_kind({}) == "raw"


# --- the entropy job (commit D) ------------------------------------------------------------------------

#: the rubric's and the attempts' job config hashes of the first commit's shards (data/features/kaggle/
#: manifest.json, kinds.<job>.cfg): the entropy job must leave them as they were
FIRST_COMMIT_CFG = {"rubric": "6464211e3b22c5cc", "attempts": "8ac1dacdaa99b16b"}


def entropy_args(data, out, *extra):
    return run_args(data, out, "--jobs", "entropy", "--no-prefix-caching", *extra)


def test_entropy_prompt_template_and_truncation():
    tok = MockTok()
    assert SP.ENTROPY_USER == SP.TASK_PH + "\n\n" + SP.ENTROPY_INSTRUCTION
    p = SP.entropy_prompt(tok, 64)
    # no system prompt, thinking on (no empty think block), the instruction right after the task
    assert p.rendered == ("<|im_start|>user\n" + SP.TASK_PH + "\n\n" + SP.ENTROPY_INSTRUCTION
                          + "<|im_end|>\n<|im_start|>assistant\n")
    SP.check_prompts_agnostic()
    low = SP.ENTROPY_USER.lower()
    assert not [b for b in SP.BENCHES + ("mmdocrag", "measurement-db") if b in low]
    assert not [w for w in ("boxed", "answer", "json", "final", "code", "math", "website") if w in low]
    ids, info = p.build("What is 1+1?")
    assert tok.dec(ids) == ("<|im_start|>user\nWhat is 1+1?\n\n" + SP.ENTROPY_INSTRUCTION
                            + "<|im_end|>\n<|im_start|>assistant\n")
    assert info == {"task_tokens": len(tok.enc("What is 1+1?")), "truncated": False}
    # a long task is cut exactly as the rubric cuts it: the same head, marker and tail, token for token
    long = " ".join(f"w{i}" for i in range(500))
    r = SP.rubric_prompt(tok, 64)
    ids_r, info_r = r.build(long, SP.rubric_meta("lang=go"))
    ids_e, info_e = p.build(long)
    cut_e = ids_e[len(p.pre):len(ids_e) - len(p.mid)]
    assert cut_e == ids_r[len(r.pre):len(r.pre) + 64] == SP.head_tail(tok.enc(long), 64, tok.enc(SP.MARKER))[0]
    assert info_e == info_r and info_e["truncated"] and len(cut_e) == 64
    text = tok.dec(ids_e)
    assert SP.MARKER in text and "w0 " in text and "w499" in text and "w250" not in text
    assert "Metadata" not in text and "<task>" not in text
    # the template is in the job config, with the attempts' sampling
    cfg = SP.job_cfg(SP.parse_args(["run", "--task-tokens", "64"]), "entropy")
    assert cfg["user"] == SP.ENTROPY_USER and cfg["instruction"] == SP.ENTROPY_INSTRUCTION and cfg["system"] is None
    assert cfg["thinking"] and cfg["task_tokens"] == 64 and cfg["marker"] == SP.MARKER
    assert cfg["sampling"] == SP.ATTEMPT_SAMPLING and cfg["sampling"]["presence_penalty"] == 1.5
    assert cfg["max_tokens"] == 1024 and cfg["n"] == 1 and cfg["record_raw"] and not cfg["force"]
    assert cfg["windows"] == [256, 1024]


def _sample(ent, lp, strs=None, ent_e=None, lp_e=None, finish="length"):
    n = len(ent)
    strs = strs if strs is not None else [f"w{t % 50} " for t in range(n)]
    return SP.Sample(list(range(n)), strs, np.asarray(lp_e if lp_e is not None else np.full(n, -9.0), np.float32),
                     np.asarray(ent_e if ent_e is not None else np.full(n, 0.01), np.float32), finish,
                     None, np.asarray(lp, np.float32), np.asarray(ent, np.float32), "processed")


def test_entropy_features_on_known_token_entropies():
    n = 1500
    ent = np.arange(n) / 1000.0
    lp = -np.arange(n) / 2000.0
    f = SP.entropy_features(_sample(ent, lp))
    e32, l32 = ent.astype(np.float32).astype(np.float64), lp.astype(np.float32).astype(np.float64)
    assert f["ent_first256"] == pytest.approx(e32[:256].mean())
    assert f["ent_first1024"] == pytest.approx(e32[:1024].mean())
    assert f["lp_first256"] == pytest.approx(l32[:256].mean()) and f["lp_first1024"] == pytest.approx(l32[:1024].mean())
    assert f["ent_first256"] == pytest.approx(0.1275, abs=1e-6)
    assert f["ent_first1024"] == pytest.approx(0.5115, abs=1e-6)
    assert f["tok_entropy"] == pytest.approx(e32.mean()) and f["stats_source"] == SP.STATS_RAW
    assert f["tok_entropy_engine"] == pytest.approx(0.01) and f["tok_lp_engine"] == pytest.approx(-9.0)
    assert f["n_tokens"] == n and f["capped"] and not f["closed"] and f["n_think"] == n and not f["degenerate"]
    assert np.allclose(SP.unf16(f["ent_seq"]), ent, atol=2e-3) and len(SP.unf16(f["lp_seq"])) == n
    # fewer tokens than a window: the mean over all of them
    short = SP.entropy_features(_sample(ent[:100], lp[:100], finish="stop"))
    assert short["ent_first256"] == short["ent_first1024"] == pytest.approx(e32[:100].mean())
    assert short["lp_first256"] == short["lp_first1024"] == pytest.approx(l32[:100].mean())
    # the thinking closed within the cap; a missing recorder value is skipped, not averaged in as 0
    strs = [f"w{t} " for t in range(300)]
    strs[10] = "</think>"
    e2 = ent[:300].copy()
    e2[3] = np.nan
    c = SP.entropy_features(_sample(e2, lp[:300], strs=strs, finish="stop"))
    assert c["closed"] and c["n_think"] == 11
    assert c["ent_first256"] == pytest.approx(np.nanmean(e2.astype(np.float32).astype(np.float64)[:256]))
    # degenerate text; an empty sample; the engine's statistics when there is no recorder (and it says so)
    assert SP.entropy_features(_sample(ent[:80], lp[:80], strs=["!"] * 80))["degenerate"]
    empty = SP.entropy_features(_sample([], []))
    assert empty["n_tokens"] == 0 and math.isnan(empty["ent_first256"]) and math.isnan(empty["lp_first1024"])
    proc = SP.entropy_features(dataclasses.replace(_sample(ent, lp), lp_raw=None, ent_raw=None))
    assert proc["stats_source"] == SP.STATS_PROC and proc["ent_first1024"] == pytest.approx(0.01)
    # seeds: fixed per unit (and --seed), none equal to an attempt's
    assert SP.entropy_seed("u1", 0) == SP.entropy_seed("u1", 0) != SP.entropy_seed("u2", 0) != SP.entropy_seed("u2", 1)
    assert SP.entropy_seed("u1", 0) != SP.attempt_seed("u1", 0, 0) and 0 <= SP.entropy_seed("u1", 0) < 2 ** 31 - 1


def test_entropy_units_are_the_rubrics_round_robin(synth, tmp_path):
    data, _, _ = synth
    ue = SP.entropy_units(str(data), SP.PARENTS)
    ur = SP.rubric_units(str(data), SP.PARENTS)
    def units(us):
        return sorted((u["unit"], tuple(u["item_ids"])) for u in us)

    assert units(ue) == units(ur)
    m, s, w, r = SP.PARENTS                                          # 7 matharena, 2 swe (s2 = s3), 1, 1
    assert [u["benchmark"] for u in ue] == [m, s, w, r, m, s, m, m, m, m, m]
    for b in SP.PARENTS:                                             # each benchmark in its fixed hash order
        assert [u["unit"] for u in ue if u["benchmark"] == b] == [u["unit"] for u in SP.unique_items(str(data), b)]
    one = SP.entropy_units(str(data), SP.PARENTS, limit=1)
    assert [u["benchmark"] for u in one] == list(SP.PARENTS)
    assert {u["unit"] for u in one} == {u["unit"] for u in SP.rubric_units(str(data), SP.PARENTS, 1)}
    # any prefix of whole rounds covers the benchmarks evenly
    many = tmp_path / "many"
    write_many(str(many))
    ue = SP.entropy_units(str(many), SP.PARENTS)
    for k in range(1, 9):
        assert Counter(u["benchmark"] for u in ue[:4 * k]) == {b: k for b in SP.PARENTS}
    assert [u["benchmark"] for u in ue[32:]] == ["matharena"] * 16
    args = SP.parse_args(entropy_args(many, tmp_path / "o", "--entropy-limit", "3"))
    assert len(SP.load_entropy_units(args)) == 12 and SP.load_entropy_units(SP.parse_args(["run"])) is None


def test_entropy_config_hash_is_its_own(monkeypatch):
    args = SP.parse_args(["run"])
    h = {j: SP.digest(SP.job_cfg(args, j)) for j in SP.JOBS}
    # the first commit's rubric and attempt shards keep their config hashes
    assert {j: h[j] for j in FIRST_COMMIT_CFG} == FIRST_COMMIT_CFG
    cfg = SP.job_cfg(args, "entropy")
    assert "version" not in cfg and cfg["entropy_version"] == SP.ENTROPY_VERSION
    monkeypatch.setattr(SP, "VERSION", "k9.9")
    assert SP.digest(SP.job_cfg(args, "entropy")) == h["entropy"]
    assert all(SP.digest(SP.job_cfg(args, j)) != h[j] for j in FIRST_COMMIT_CFG)
    monkeypatch.undo()
    monkeypatch.setattr(SP, "ENTROPY_VERSION", "e9.9")
    assert SP.digest(SP.job_cfg(args, "entropy")) != h["entropy"]
    assert all(SP.digest(SP.job_cfg(args, j)) == h[j] for j in FIRST_COMMIT_CFG)
    monkeypatch.undo()
    # what does not change a unit's result leaves the hash alone; what does, changes it
    for extra in (["--entropy-limit", "5"], ["--entropy-shard", "8"], ["--entropy-shard-tokens", "9000"],
                  ["--no-prefix-caching"], ["--k", "2"], ["--max-tokens", "2048"], ["--attempt-logprobs", "1"],
                  ["--entropy-benchmarks", "matharena"], ["--max-num-seqs", "32"], ["--gpu-mem", "0.8"]):
        assert SP.digest(SP.job_cfg(SP.parse_args(["run", *extra]), "entropy")) == h["entropy"], extra
    for extra in (["--presence-penalty", "0"], ["--task-tokens", "2048"], ["--seed", "1"],
                  ["--model", "Qwen/Qwen3-8B-AWQ"]):
        assert SP.digest(SP.job_cfg(SP.parse_args(["run", *extra]), "entropy")) != h["entropy"], extra
    # the transformers fallback cannot sample with the recorder: the entropy job is dropped there
    assert SP.parse_args(["run", "--backend", "hf", "--jobs", "entropy,rubric"]).jobs == ["rubric"]
    with pytest.raises(SystemExit):
        SP.parse_args(["run", "--jobs", "entropie"])


def test_entropy_shards_by_count_and_kv_tokens():
    lens = [100, 3000, 200, 3000, 50, 90_000, 10]
    got = [[u for u in s] for s, _ in SP.entropy_shards(range(len(lens)), lambda i: (range(lens[i]), None), 3,
                                                         8000, gen=1024)]
    # the first shard closes at 3 units; the second before a unit that would pass 8000 tokens (5098 + 91024);
    # a unit over the budget alone is a shard of its own
    assert got == [[0, 1, 2], [3, 4], [5], [6]]
    d = SP.parse_args(["run"])
    assert SP.entropy_shard_limits(d) == (48, 80_000) and 80_000 <= 0.9 * SP.kv_tokens(d)
    assert SP.entropy_shard_limits(SP.parse_args(["run", "--max-num-seqs", "32"]))[0] == 32
    low = SP.parse_args(["run", "--gpu-mem", "0.80"])                  # a retry's smaller KV cache
    assert SP.entropy_shard_limits(low)[1] == int(0.9 * SP.kv_tokens(low)) < 80_000


def _entropy_export(out):
    return out / SP.model_slug(SP.DEFAULT_MODEL, "vllm") / "export"


def pin_entropy_cfg(S, monkeypatch, argv):
    """strong_llm_eval's ENTROPY_RULE is fixed for the kit's default entropy config (commit D's); these tests run the
    kit with --task-tokens 256, another config, which the consumer's check-schema warns about: pin the rule to it
    here, and check that it is not the default one."""
    cfg = SP.digest(SP.job_cfg(SP.parse_args(argv), "entropy"))
    assert cfg != S.ENTROPY_CFG == SP.digest(SP.job_cfg(SP.parse_args(["run"]), "entropy"))
    monkeypatch.setitem(S.ENTROPY_RULE["config"], "cfg", cfg)


def test_entropy_pipeline_export_schema_and_manifest(synth, monkeypatch):
    data, out, items = synth
    be = MockBackend()
    argv = entropy_args(data, out, "--entropy-shard", "4")
    assert SP.main(argv, backend=be, now=clock()) == 0
    args = SP.parse_args(argv)
    ue = SP.entropy_units(str(data), SP.PARENTS)
    ex = _entropy_export(out)
    man = json.load(open(ex / "manifest.json"))
    assert man["schema_version"] == 1 and man["hash"] == SP.HASH_DEF and man["logprobs"] == "raw"
    assert man["shards"] == [{"path": "entropy/entropy.parquet", "rows": 12,
                              "sha256": SP.file_sha256(str(ex / "entropy" / "entropy.parquet"))}]
    en = pd.read_parquet(ex / "entropy" / "entropy.parquet")
    assert list(en.columns) == ["benchmark", "item_id", "content_sha256"] + list(SP.ENTROPY_COLS)
    want = {"ent_first256": "float64", "ent_first1024": "float64", "lp_first256": "float64", "lp_first1024": "float64",
            "ent_n_tokens": "int64", "ent_closed": "bool", "ent_degenerate": "bool", "ent_prompt_tokens": "int64",
            "ent_task_tokens": "int64", "ent_truncated": "bool"}
    assert {c: str(en[c].dtype) for c in want} == want and SP.ENTROPY_COLS == want
    all_ids = {(b, r[0]) for b, rows in items.items() for r in rows}
    assert set(zip(en.benchmark, en.item_id)) == all_ids and len(en) == len(all_ids)
    by_id = {(b, r[0]): r for b, rows in items.items() for r in rows}
    for row in en.itertuples():
        _, content, feats, _ = by_id[(row.benchmark, row.item_id)]
        assert row.content_sha256 == SP.sha(f"{content}\n{feats}")
    # the raw statistics over the windows: the mock's long samples run to the cap, the short ones close
    long, short = en[en.ent_n_tokens == 1024], en[en.ent_n_tokens < 1024]
    assert len(long) and len(short) and not long.ent_closed.any() and short.ent_closed.all()
    assert en.ent_first1024.between(RAW_ENT, RAW_ENT + 0.04).all()
    assert en.lp_first256.between(RAW_LP - 0.07, RAW_LP).all() and not en.ent_degenerate.any()
    assert (en.loc[en.benchmark == "researchcodebench", "ent_truncated"]).all()
    assert (en.loc[en.benchmark == "researchcodebench", "ent_task_tokens"] > 256).all()
    s2, s3 = (en[en.item_id == i].iloc[0] for i in ("s2", "s3"))       # one unit, both item_ids
    assert s2.ent_first1024 == s3.ent_first1024 and s2.ent_prompt_tokens == s3.ent_prompt_tokens
    det = pd.read_parquet(ex / "_detail" / "entropy_units.parquet")
    assert len(det) == len(ue) and det.unit.is_unique and {"text", "lp_seq", "ent_seq", "seed"} <= set(det.columns)
    for r in det.itertuples():
        ent = SP.unf16(r.ent_seq)
        assert len(ent) == r.n_tokens
        assert r.ent_first1024 == pytest.approx(float(ent.astype(np.float64).mean()), abs=1e-3)
    # the manifest's job section
    k = man["kinds"]["entropy"]
    store_cfg = SP.digest(SP.job_cfg(args, "entropy"))
    assert k["cfg"] == k["cfg_hash"] == store_cfg and k["version"] == SP.ENTROPY_VERSION
    assert k["config"] == json.loads(json.dumps(SP.job_cfg(args, "entropy")))
    assert k["sampling"] == SP.ATTEMPT_SAMPLING and k["max_new_tokens"] == 1024 and k["n"] == 1
    assert k["prompt"]["template"] == "{task}\n\n" + SP.ENTROPY_INSTRUCTION and k["prompt"]["thinking"]
    assert k["prompt"]["rendered"] == SP.entropy_prompt(be.tok, 256).rendered and k["prompt"]["task_tokens"] == 256
    assert k["signs"] == {"ent_first256": 1, "ent_first1024": 1, "lp_first256": -1, "lp_first1024": -1,
                          **{c: 0 for c in SP.ENTROPY_DIAG}} and k["primary"] == "ent_first1024"
    assert k["dtypes"] == want and k["units"] == len(ue) and k["logprobs"] == "raw"
    ts = k["token_stats"]
    n_shards = len(be.check_prompts)
    assert n_shards == math.ceil(len(ue) / 4) and ts["sources"] == {SP.STATS_RAW: len(ue)}
    assert ts["recorder_check"]["status"] == "ok" and ts["recorder_check"]["max_abs_gap"] == 0
    assert ts["recorder_check"]["shards_checked"] == n_shards and ts["recorder_check"]["tokens"] == 12 * n_shards
    # the keys, the harness file, the summary
    keys = pd.read_parquet(ex / "_keys.parquet")
    assert set(keys.job) == {"entropy"} and len(keys) == len(all_ids)
    for r in keys.itertuples():
        _, content, feats, _ = by_id[(r.benchmark, r.item_id)]
        item = {"item_content": content, "item_features": feats, "interactors": "", "benchmark_id": r.benchmark}
        assert r.key == SP.item_key_hex(item) and r.text_key == SP.text_key_hex(item)
    hj = json.load(open(ex / "_harness.json"))
    assert set(hj) == {f"entropy_{f}" for f in SP.ENTROPY_EXPORT}
    assert hj["entropy_lp_first1024"]["m3"] == pytest.approx(-float(en.loc[en.item_id == "m3", "lp_first1024"].iloc[0]))
    m3 = en[en.item_id == "m3"].iloc[0]
    assert hj["entropy_ent_first1024"]["m3"] == pytest.approx(float(m3.ent_first1024))
    # split-harness does not hand them to experiments/harness.py one by one: strong_llm_eval --job entropy reads them
    assert SP.split_harness(str(ex)) == [] and not os.path.exists(ex / "_harness" / "entropy_ent_first1024.json")
    summ = json.load(open(ex / "_summary.json"))["entropy"]
    assert summ["units"] == len(ue) and summ["recorder_check"] == "ok" and 0 < summ["closed_rate"] < 1
    # the requests: one sample per unit with its own seed and the attempts' sampling; a greedy check per shard
    samp = [c for c in be.calls if c["seeds"] is not None]
    assert all(c["params"]["temperature"] == 0.6 and c["params"]["top_p"] == 0.95 and c["params"]["top_k"] == 20
               and c["params"]["presence_penalty"] == 1.5 and c["params"]["max_tokens"] == 1024
               and c["params"]["logprobs"] == 1 and c["params"]["record_raw"] and c["n"] <= 4 for c in samp)
    assert [s for c in samp for s in c["seeds"]] == [SP.entropy_seed(u["unit"], 0) for u in ue]
    chk = [c for c in be.calls if c["seeds"] is None]
    assert len(chk) == n_shards and all(c["n"] == 1 and c["params"]["temperature"] == 0 for c in chk)
    assert be.check_prompts == be.entropy_prompts[::4]                 # each shard's first prompt
    # the prompts, in round-robin order, carry the task and nothing of its metadata
    assert len(be.entropy_prompts) == len(ue)
    for u, text in zip(ue, be.entropy_prompts):
        assert "Metadata" not in text and "problem_idx" not in text and "competition=" not in text
        if len(be.tok.enc(u["content"])) <= 256:
            assert f"<|im_start|>user\n{u['content']}\n\n{SP.ENTROPY_INSTRUCTION}<|im_end|>" in text
    # the root manifest: the job config by its hash, progress, the session's stats
    top = json.load(open(out / "manifest.json"))
    assert top["job_configs"][store_cfg] == json.loads(json.dumps(SP.job_cfg(args, "entropy")))
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    assert top["progress"][slug]["entropy"] == {"units": len(ue), "done": len(ue)}
    st = top["sessions"][-1]["stats"]["entropy"]
    assert st["recorder_check"]["status"] == "ok" and st["shards"] == n_shards and st["gen_tokens"] > 0
    assert top["models"][slug]["entropy_rendered"] == k["prompt"]["rendered"]
    # the consumer's schema check: no error, no warning
    try:
        from experiments import strong_llm_eval as S
    except Exception as e:                                   # the consumer is optional here
        pytest.skip(f"experiments/strong_llm_eval.py not importable: {e}")
    rep = S.check_schema(str(ex), benches=SP.BENCHES)
    assert rep["ok"], rep["errors"]
    assert [w for w in rep["warnings"] if "job config" in w] == rep["warnings"] != []    # not the default config
    pin_entropy_cfg(S, monkeypatch, argv)
    rep = S.check_schema(str(ex), benches=SP.BENCHES)
    assert rep["ok"], rep["errors"]
    assert not rep["warnings"], rep["warnings"]
    assert set(rep["shards"]["entropy/entropy.parquet"]["features"]) == set(SP.ENTROPY_COLS)
    over = k["signs"]
    for f, (sign, _) in SP.ENTROPY_EXPORT.items():
        assert S.declared_sign(f, over)[0] == sign
    for c in SP.ENTROPY_DIAG:
        assert S.declared_sign(c, over)[0] == 0


def test_entropy_commit_keeps_the_attached_rubric_and_attempts(synth, tmp_path, monkeypatch):
    """Commit D with the previous version's Output attached: the new export holds the rubric's and the attempts'
    tables unchanged beside the entropy table, with their prompts and settings, and passes the consumer's check
    (the entropy diagnostics do not collide with the rubric's columns)."""
    data, out, _ = synth
    assert SP.main(run_args(data, out), backend=MockBackend(), now=clock()) == 0
    new = tmp_path / "commit_d"
    be = MockBackend()
    assert SP.main(resumed(entropy_args(data, new), out), backend=be, now=clock()) == 0
    assert not be.rubric_prompts and not be.forced_prompts and be.entropy_prompts
    old_ex, ex = _entropy_export(out), _entropy_export(new)
    man0, man = json.load(open(old_ex / "manifest.json")), json.load(open(ex / "manifest.json"))
    got = {s["path"]: s for s in man["shards"]}
    assert set(got) == {"rubric/rubric.parquet", "attempts/attempts.parquet", "entropy/entropy.parquet"}
    for s in man0["shards"]:
        # the same rows and, with one pyarrow as here, the same bytes (on Kaggle only if the image's pyarrow is the
        # first commit's: parquet records its writer's version)
        assert got[s["path"]] == s
    for kind in ("rubric", "attempts"):
        assert man["kinds"][kind] == man0["kinds"][kind]       # prompts, rendered text, sampling, cfg
    assert man["kinds"]["attempts"]["max_new_tokens"] == 64 and man["kinds"]["rubric"]["prompt"]["rendered"]
    keys = pd.read_parquet(ex / "_keys.parquet")
    assert set(keys.job) == {"rubric", "attempts", "entropy"}
    assert len(json.load(open(new / "manifest.json"))["sessions"]) == 2
    try:
        from experiments import strong_llm_eval as S
    except Exception as e:
        pytest.skip(f"experiments/strong_llm_eval.py not importable: {e}")
    pin_entropy_cfg(S, monkeypatch, entropy_args(data, new))
    rep = S.check_schema(str(ex), benches=SP.BENCHES)
    assert rep["ok"], rep["errors"]
    assert not rep["warnings"], rep["warnings"]


def test_entropy_resume(synth, monkeypatch):
    data, out, _ = synth
    be = MockBackend()
    assert SP.main(entropy_args(data, out, "--entropy-shard", "3", "--max-shards", "1"), backend=be,
                   now=clock()) == SP.EXIT_DEADLINE
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    assert len(SP.Store(str(out / slug / "entropy"), "").read()) == 3 and len(be.entropy_prompts) == 3
    ue = SP.entropy_units(str(data), SP.PARENTS)
    assert json.load(open(out / "manifest.json"))["progress"][slug]["entropy"] == {"units": len(ue), "done": 3}
    be2 = MockBackend()
    assert SP.main(entropy_args(data, out, "--entropy-shard", "3"), backend=be2, now=clock()) == 0
    done_first = {SP.entropy_seed(u["unit"], 0) for u in ue[:3]}
    seeds2 = [s for c in be2.calls if c["seeds"] for s in c["seeds"]]
    assert len(seeds2) == len(ue) - 3 and not done_first & set(seeds2)
    en = SP.Store(str(out / slug / "entropy"), "").read()
    assert len(en) == len(ue) and en.unit.is_unique
    # nothing left: no generation; other jobs' settings do not restart it
    be3 = MockBackend()
    assert SP.main(entropy_args(data, out, "--k", "2", "--max-tokens", "48"), backend=be3, now=clock()) == 0
    assert not be3.calls
    # another prompt version is another config: it starts over, and the export takes the larger config
    be4 = MockBackend()
    monkeypatch.setattr(SP, "ENTROPY_VERSION", "e-test")
    assert SP.main(entropy_args(data, out, "--max-shards", "1", "--entropy-shard", "2"), backend=be4,
                   now=clock()) == SP.EXIT_DEADLINE
    assert len(be4.entropy_prompts) == 2
    assert len(pd.read_parquet(_entropy_export(out) / "entropy" / "entropy.parquet")) == 12


class MisalignedEntropyRecorder(MockBackend):
    """The recorder's raw log-probs disagree with the engine's on the entropy shards' greedy check."""

    def entropy_check(self):
        s = super().entropy_check()
        s.lp_raw = s.lp_raw - 0.5
        return s


class GarbageEntropy(MockBackend):
    def entropy(self, seed, max_tokens):
        return self._sample("!" * max_tokens, lambda t, s: -0.1, lambda t, s: 0.1, "length")


def test_entropy_self_checks_stop_a_useless_run(synth):
    data, out, _ = synth
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    be = MisalignedEntropyRecorder()
    assert SP.main(entropy_args(data, out, "--entropy-shard", "2"), backend=be, now=clock()) == SP.EXIT_CHECK
    assert len(SP.Store(str(out / slug / "entropy"), "").read()) == 2 and len(be.check_prompts) == 1
    sess = json.load(open(out / "manifest.json"))["sessions"][-1]
    assert sess["stats"]["entropy"]["self_check"]["status"] == "FAILED"
    assert "recorder" in sess["stats"]["entropy"]["self_check"]["why"][0]
    assert sess["stats"]["entropy"]["recorder_check"]["max_abs_gap"] == pytest.approx(0.5)
    man = json.load(open(_entropy_export(out) / "manifest.json"))
    assert man["kinds"]["entropy"]["token_stats"]["recorder_check"]["status"] == "FAILED"
    # --no-self-check runs on
    assert SP.main(entropy_args(data, out, "--entropy-shard", "2", "--no-self-check"),
                   backend=MisalignedEntropyRecorder(), now=clock()) == 0
    # degenerate texts
    g = out.parent / "garbage"
    be = GarbageEntropy()
    assert SP.main(entropy_args(data, g, "--entropy-shard", "2"), backend=be, now=clock()) == SP.EXIT_CHECK
    why = json.load(open(g / "manifest.json"))["sessions"][-1]["stats"]["entropy"]["self_check"]["why"]
    assert len(why) == 1 and "degenerate" in why[0] and len(be.entropy_prompts) == 2
    # nothing generated at all
    assert SP.entropy_self_check([{"degenerate": False, "n_tokens": 0}] * 3, "ok") == ["no unit generated a token"]
    raw = {"degenerate": False, "n_tokens": 5, "stats_source": SP.STATS_RAW}
    assert SP.entropy_self_check([raw], "ok") == []


def test_entropy_self_check_stops_a_run_without_the_raw_statistics(synth):
    """No recorder (vLLM V1, or a recorder that never ran): the greedy check compares nothing (status n/a) and
    the samples' statistics are the engine's processed ones, which strong_llm_eval does not read; the first shard
    stops the run (exit 77) instead of spending the commit on them."""
    data, out, _ = synth
    be = MockBackend(raw=False)
    assert SP.main(entropy_args(data, out, "--entropy-shard", "2"), backend=be, now=clock()) == SP.EXIT_CHECK
    assert len(be.entropy_prompts) == 2 and len(be.check_prompts) == 1
    st = json.load(open(out / "manifest.json"))["sessions"][-1]["stats"]["entropy"]
    assert st["recorder_check"]["status"] == "n/a" and st["stats_source"] == {SP.STATS_PROC: 2}
    why = st["self_check"]["why"]
    assert len(why) == 2 and "did not run" in why[0] and "not the raw full-vocabulary" in why[1]
    man = json.load(open(_entropy_export(out) / "manifest.json"))
    assert man["kinds"]["entropy"]["logprobs"] == SP.STATS_PROC
    # the rule on a shard: every sample that generated a token carries the raw statistics; an empty one has none
    raw = {"degenerate": False, "n_tokens": 5, "stats_source": SP.STATS_RAW}
    one_off = SP.entropy_self_check([raw] * 9 + [{**raw, "stats_source": SP.STATS_RAW_TOPK}], "ok")
    assert len(one_off) == 1 and "10%" in one_off[0] and SP.STATS_RAW_TOPK in one_off[0]
    assert SP.entropy_self_check([raw, {**raw, "n_tokens": 0, "stats_source": SP.STATS_PROC}], "ok") == []
    assert SP.entropy_self_check([raw], "n/a")[0].startswith("the recorder check did not run")
    # the export counts the same way: an empty unit's source label is not a second source
    eu = pd.DataFrame({"stats_source": [SP.STATS_RAW, SP.STATS_PROC], "n_tokens": [5, 0],
                       "check_gap": [0.0, np.nan], "check_n": [3, 0]})
    ts = SP.entropy_token_stats(eu)
    assert ts["logprobs"] == "raw" and ts["sources"] == {SP.STATS_RAW: 1} and ts["empty_units"] == 1


def test_entropy_plan_numbers(synth):
    data, out, _ = synth
    args = SP.parse_args(["plan"] + entropy_args(data, out)[1:])
    ue = SP.load_entropy_units(args)
    b = SP.plan_budget(args, MockTok(), None, None, ue)
    e, r = b["entropy"], SP.rates(args)
    p = SP.entropy_prompt(MockTok(), args.task_tokens)
    lens = [len(p.build(u["content"])[0]) for u in ue]
    assert e["units"] == len(ue) and e["prompt_tokens"] == sum(lens) and e["max_prompt"] == max(lens)
    assert e["gen_tokens_est"] == len(ue) * 1024 and e["truncated"] == 1 and not e["prefix_caching"]
    assert e["shards_est"] == 1 and e["shard_limits"] == {"units": 48, "kv_tokens": 80_000}
    assert r["prefill_nocache"] == 1000 and r["decode_entropy"] == 180
    assert e["rates_tok_s"]["decode_slow"] == pytest.approx(180 * SP.ENTROPY_SLOW) == pytest.approx(135)
    assert e["hours_est"] == pytest.approx(SP.entropy_secs(sum(lens), len(ue), 1, r, 180) / 3600, abs=0.01)
    assert e["hours_est_slow"] == pytest.approx(
        SP.entropy_secs(sum(lens), len(ue), 1, r, 135, step_s=0.2) / 3600, abs=0.01)
    assert e["hours_est_slow"] >= e["hours_est"] and e["fits_one_commit_slow"] and e["units_per_commit_slow"] == len(ue)
    piece = b["session_plan"][-1]
    assert [p_["piece"] for p_ in b["session_plan"]] == ["entropy"]
    assert piece["args"] == ["--jobs", "entropy", "--no-prefix-caching"] and piece["commits_slow"] == 1
    pa = SP.parse_args(["run", *piece["args"]])
    assert pa.jobs == ["entropy"] and pa.no_prefix_caching
    assert b["slow_case"]["total_hours_est"] == pytest.approx(e["hours_est_slow"], abs=0.01)
    # the quota Kaggle must still show for this commit: the plan cannot see the hours already used this week
    q = b["quota"]
    assert q["commit_need_h"] == round(piece["gpu_h_slow"] + SP.QUOTA_MARGIN_H, 1) and not q["counts_hours_used"]
    # prefill is uncached whether or not --no-prefix-caching was given
    assert SP.plan_budget(SP.parse_args(["plan"] + run_args(data, out, "--jobs", "entropy")[1:]), MockTok(), None,
                          None, ue)["entropy"]["hours_est"] == e["hours_est"]
    # a commit that cannot hold the job says how many units it covers
    slow = SP.parse_args(["plan"] + entropy_args(data, out, "--session-hours", "1.4", "--decode-rate-entropy",
                                                 "0.5")[1:])
    s = SP.plan_budget(slow, MockTok(), None, None, ue)["entropy"]
    per_unit = SP.entropy_secs(sum(lens), len(ue), 1, SP.rates(slow), 0.5 * SP.ENTROPY_SLOW, step_s=0.2) / len(ue)
    assert not s["fits_one_commit_slow"] and s["commit_usable_hours"] == pytest.approx(1.0)
    assert s["units_per_commit_slow"] == int(3600 / per_unit) < len(ue)
    assert SP.main(["plan"] + entropy_args(data, out)[1:], tok=MockTok()) == 0
    assert "entropy" in json.load(open(out / "manifest.json"))["plan"][SP.model_slug(SP.DEFAULT_MODEL, "vllm")]
    # the real job (plan on the local items with the Qwen3 tokenizer, README "Коммит D"): 4,078 units, 1,938,214
    # uncached prompt tokens, 89 shards; at the 14B's rates it fits one commit even in the slow case
    d = SP.parse_args(["run", "--jobs", "entropy", "--no-prefix-caching"])
    rd = SP.rates(d)
    usable = d.session_hours - SP.SESSION_OVERHEAD_H
    fast = SP.entropy_secs(1_938_214, 4078, 89, rd, rd["decode_entropy"]) / 3600
    slow_h = SP.entropy_secs(1_938_214, 4078, 89, rd, rd["decode_entropy"] * SP.ENTROPY_SLOW,
                             step_s=SP.ENTROPY_CHECK_STEP_S / SP.ENTROPY_SLOW) / 3600
    assert fast == pytest.approx(7.11, abs=0.01) and slow_h == pytest.approx(9.30, abs=0.01) and slow_h <= usable
    # README: 9.7 GPU-h of the quota in the slow case, and Settings -> Quota must show at least 10.5 before it
    assert round(slow_h + SP.SESSION_OVERHEAD_H, 1) == 9.7
    assert round(slow_h + SP.SESSION_OVERHEAD_H + SP.QUOTA_MARGIN_H, 1) == 10.5
    # and the default shard fits the measured KV cache (6,292 blocks of 16 tokens)
    assert SP.entropy_shard_limits(d)[1] <= 6292 * 16


# --- real items (skipped without data/) --------------------------------------------------------------

needs_data = pytest.mark.skipif(not all(os.path.exists(os.path.join(DATA, b, "items.parquet")) for b in SP.PARENTS),
                                reason="measurement-db not in data/ (python -m paiec.fetch)")


@needs_data
def test_attempt_item_set_on_real_data():
    ua = SP.attempt_units(DATA)
    assert len(ua) == 819 and sum(len(u["item_ids"]) for u in ua) == 946
    probe = [u for u in ua if u["probe"]]
    assert len(probe) == 147 and ua[:147] == probe
    assert SP.PROBE_IDS <= {i for u in probe for i in u["item_ids"]}
    csv = ("/private/tmp/claude-501/-Users-nikitapolomosnov-PycharmProjects-PAIEC/9df8ecfa-42c0-4508-919c-f794f1770909"
           "/scratchpad/rethink2/attempt-signals/a7_probe_items.csv")
    if os.path.exists(csv):
        assert set(pd.read_csv(csv, index_col=0).index.astype(str)) == SP.PROBE_IDS


@needs_data
def test_pipeline_on_real_items(tmp_path):
    from paiec import data as D
    from paiec import llmfeat as F
    out = tmp_path / "out"
    be = MockBackend()
    args = ["run", "--data-dir", DATA, "--out", str(out), "--download", "", "--resume-from", "none",
            "--rubric-limit", "3", "--attempt-limit", "3", "--k", "2", "--max-tokens", "64", "--t0", "0"]
    assert SP.main(args, backend=be, now=clock()) == 0
    ex = out / SP.model_slug(SP.DEFAULT_MODEL, "vllm") / "export"
    ru = pd.read_parquet(ex / "rubric" / "rubric.parquet")
    at = pd.read_parquet(ex / "attempts" / "attempts.parquet")
    assert set(ru.benchmark) == set(SP.PARENTS) and len(set(at.item_id)) >= 3
    keys = pd.read_parquet(ex / "_keys.parquet")
    local = {}
    for b in SP.PARENTS:
        it = pd.read_parquet(os.path.join(DATA, b, "items.parquet"),
                             columns=["item_id", "benchmark_id", "content", "item_features"])
        want = set(keys.loc[keys.benchmark == b, "item_id"])
        for r in it[it.item_id.astype(str).isin(want)].itertuples(index=False):
            local[(b, str(r.item_id))] = {"item_content": D._clean(r.content), "item_features": D._clean(r.item_features),
                                          "interactors": "", "benchmark_id": D._clean(r.benchmark_id)}
    assert len(local) == len(set(zip(keys.benchmark, keys.item_id)))
    for r in keys.itertuples():
        item = local[(r.benchmark, r.item_id)]
        assert r.key == F.key_for(item, item["benchmark_id"]) and r.text_key == F.text_key(item)
    for r in ru.itertuples():
        assert r.content_sha256 == SP.sha(F.item_text(local[(r.benchmark, r.item_id)]))
    # researchcodebench's papers are cut to --task-tokens by head and tail
    assert (ru.loc[ru.benchmark == "researchcodebench", "truncated_n"] == 1).all()
    # benchmark names never enter a prompt through the templates
    p = SP.rubric_prompt(be.tok, 3072)
    bound = len(p.pre) + 3072 + len(p.mid) + SP.META_MAX_TOKENS + len(p.post)
    assert (ru.loc[ru.benchmark == "researchcodebench", "prompt_tokens"] <= bound).all()
    assert (ru.loc[ru.benchmark == "researchcodebench", "task_tokens"] > 3072).all()
    assert not [b for b in SP.BENCHES if b in p.rendered]


# --- the notebook and the kernel metadata -------------------------------------------------------------

def test_notebook_carries_the_script_verbatim():
    with open(os.path.join(KDIR, "strong_probe.py"), encoding="utf-8") as fh:
        src = fh.read()
    with open(NB.NOTEBOOK, encoding="utf-8") as fh:
        on_disk = fh.read()
    assert on_disk == NB.dumps(NB.build(src)), "run python kaggle/strong_probe/build_notebook.py"
    nb = json.loads(on_disk)
    assert NB.script_in(nb) == src
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert f'VLLM, TRANSFORMERS = "{SP.VLLM_PIN}", "{SP.TRANSFORMERS_PIN}"' in code
    rest = code.replace(src, "")
    assert "get_secret" not in rest and "UserSecretsClient" not in rest     # read inside the script only
    assert "/kaggle/working" not in NB.TARGET


def test_kernel_metadata():
    with open(os.path.join(KDIR, "kernel-metadata.json")) as fh:
        meta = json.load(fh)
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["code_file"] == os.path.basename(NB.NOTEBOOK) and meta["kernel_type"] == "notebook"
    assert meta["id"].endswith("/paiec-strong-probe") and not meta["dataset_sources"]


# --- the output's file count (Kaggle keeps at most 500 files) ------------------------------------------

class CrashingBackend(MockBackend):
    """A run process that dies (an exception out of the engine) at its `after`-th generate call."""

    def __init__(self, after):
        super().__init__()
        self.after = after

    def generate(self, prompts, params, seeds=None):
        if len(self.calls) + 1 >= self.after:
            raise RuntimeError("engine died")
        return super().generate(prompts, params, seeds)


def resumed(args, *sources):
    args = list(args)
    args[args.index("--resume-from") + 1] = ",".join(map(str, sources))
    return args


def _sessions(data, root, seen_by_write):
    """A worst-case week: session 1 crashes mid-run and its retry continues in the same output; session 2
    restores it and applies a slow-run remedy (--attempt-logprobs 1: a new attempts config); session 3
    restores session 2 and the stale session 1 and applies another (--max-tokens); session 4 finishes, the
    entropy job included. One unit per shard. Returns the four outputs."""
    base = ["--jobs", "rubric,attempts,entropy", "--rubric-shard", "1", "--attempt-shard", "1", "--entropy-shard", "1",
            "--k", "2"]
    s1, s2, s3, s4 = (root / f"s{i}" for i in range(1, 5))
    with pytest.raises(RuntimeError):
        SP.main(run_args(data, s1, *base), backend=CrashingBackend(after=30), now=clock())
    assert SP.main(run_args(data, s1, *base, "--max-shards", "20"), backend=MockBackend(), now=clock()) \
        == SP.EXIT_DEADLINE
    assert SP.main(resumed(run_args(data, s2, *base, "--attempt-logprobs", "1", "--max-shards", "20"), s1),
                   backend=MockBackend(), now=clock()) == SP.EXIT_DEADLINE
    assert SP.main(resumed(run_args(data, s3, *base, "--max-tokens", "48", "--max-shards", "20"), s2, s1),
                   backend=MockBackend(), now=clock()) == SP.EXIT_DEADLINE
    assert SP.main(resumed(run_args(data, s4, *base, "--max-tokens", "48"), s3), backend=MockBackend(),
                   now=clock()) == 0
    return s1, s2, s3, s4


def _count_writes(monkeypatch):
    seen = []
    orig = SP.Store.write

    def write(self, items, samples=None):
        orig(self, items, samples)
        seen.append(SP.count_entries(os.path.dirname(os.path.dirname(self.root))))

    monkeypatch.setattr(SP.Store, "write", write)
    return seen


def _all_units_exported(data, out):
    ex = out / SP.model_slug(SP.DEFAULT_MODEL, "vllm") / "export"
    ru = pd.read_parquet(ex / "rubric" / "rubric.parquet")
    at = pd.read_parquet(ex / "attempts" / "attempts.parquet")
    en = pd.read_parquet(ex / "entropy" / "entropy.parquet")
    n_items = sum(len(u["item_ids"]) for u in SP.rubric_units(str(data), SP.PARENTS))
    return (len(ru) == n_items and set(at.item_id) == {i for u in SP.attempt_units(str(data)) for i in u["item_ids"]}
            and set(zip(en.benchmark, en.item_id)) == set(zip(ru.benchmark, ru.item_id)))


def test_file_count_worst_case_stays_under_the_guard(tmp_path, monkeypatch):
    """Kaggle saves at most 500 files of a notebook's output and mounts at most 500 of an output attached as
    an input. Worst case: every shard its own file (SEG_BYTES tiny, as the k1.2 layout did), one unit per
    shard, a crash and a retry, several sessions restoring the last (and a stale one), new configs. With the
    guard lowered to 40, the entries under an output never pass it by more than one shard's two files, and
    every saved output ends small, complete and without duplicated work."""
    data = tmp_path / "data"
    write_many(str(data))
    monkeypatch.setattr(SP, "SEG_BYTES", 1)
    monkeypatch.setattr(SP, "MAX_FILES", 40)
    seen = _count_writes(monkeypatch)
    outs = _sessions(data, tmp_path, seen)
    assert len(seen) > 100 and max(seen) <= SP.MAX_FILES + 1, (len(seen), max(seen))
    for out in outs:
        n = SP.count_entries(str(out))
        assert n <= SP.MAX_FILES, (out, n)
        man = json.load(open(out / "manifest.json"))
        assert all(s["files"]["max_seen"] <= SP.MAX_FILES + 1 for s in man["sessions"] if "files" in s)
    assert any(s["files"]["compactions"] for s in json.load(open(outs[0] / "manifest.json"))["sessions"])
    assert _all_units_exported(data, outs[-1])
    # nothing was lost or redone across the sessions: every unit once per config
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    rub = SP.Store(str(outs[-1] / slug / "rubric"), "").read()
    assert rub.unit.is_unique and len(rub) == len(SP.rubric_units(str(data), SP.PARENTS))
    ent = SP.Store(str(outs[-1] / slug / "entropy"), "").read()
    assert ent.unit.is_unique and len(ent) == len(rub)
    assert len(SP.Store(str(outs[-1] / slug / "entropy"), "").files()) == 1


def test_file_count_with_the_default_layout(tmp_path, monkeypatch):
    """The same week with the default constants: a process writes a segment per kind, so an output holds
    about twenty entries whatever the number of shards."""
    data = tmp_path / "data"
    write_many(str(data))
    seen = _count_writes(monkeypatch)
    outs = _sessions(data, tmp_path, seen)
    assert max(seen) <= 40 and all(SP.count_entries(str(o)) <= 40 for o in outs)
    assert _all_units_exported(data, outs[-1])
    # the guard's own bound, far below Kaggle's cap, leaves room for export/ and a few foreign files
    assert SP.MAX_FILES + 2 + 20 < SP.STOP_FILES < SP.KAGGLE_FILE_CAP


def test_store_segments_rotate_and_compact(tmp_path, monkeypatch):
    st = SP.Store(str(tmp_path / "job"), "c1", run="r1")
    frame = lambda u, t: pd.DataFrame({"unit": [u], "cfg": ["c1"], "t": [t], "x": [1.0]})   # noqa: E731
    for i in range(5):
        st.write(frame(f"u{i}", float(i)))
    assert [os.path.basename(p) for p in st.files()] == ["items-r1-000.parquet"]      # one open segment
    monkeypatch.setattr(SP, "SEG_BYTES", 1)
    st.write(frame("u5", 5.0))                               # fills the open segment, which then rotates
    st.write(frame("u6", 6.0))
    assert len(st.files()) == 2 and st.done() == {f"u{i}" for i in range(7)}
    st.write(frame("u0", 9.0).assign(x=2.0))                 # a unit written again: the latest row wins
    df = st.read()
    assert len(st.files()) == 3 and len(df) == 7 and float(df.loc[df.unit == "u0", "x"].iloc[0]) == 2.0
    gone = st.compact()
    assert len(gone) == 3 and len(st.files()) == 1 and st.read().equals(df)
    assert not glob_tmp(tmp_path)


def glob_tmp(root):
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".tmp")]


def test_resume_reads_the_k12_layout_and_drops_its_reference_columns(synth, tmp_path):
    """An output of the k1.2 store (a file per shard, items_NNNNN / samples_NNNNN, with the per-attempt
    correct flag and canonical answer, and per-item graded / top_correct) resumes: its units count as done,
    it is compacted into segments without those columns, and a second restore copies nothing."""
    data, out, _ = synth
    assert SP.main(run_args(data, out), backend=MockBackend(), now=clock()) == 0
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    for job in ("rubric", "attempts"):
        st = SP.Store(str(out / slug / job), "")
        for kind in ("items", "samples"):
            df = st.read(kind)
            if not len(df):
                continue
            for f in st.files(kind):
                os.remove(f)
            if kind == "samples":
                df = df.assign(correct=True, canon="num:42")
            elif job == "attempts":
                df = df.assign(graded=1.0, top_correct=1.0, graded_natural=1.0)
            for i in range(len(df)):
                df.iloc[[i]].to_parquet(out / slug / job / f"{kind}_{i:05d}.parquet", index=False)
    assert SP.Store(str(out / slug / "attempts"), "").legacy()
    new = tmp_path / "s2"
    be = MockBackend()
    assert SP.main(resumed(run_args(data, new), out), backend=be, now=clock()) == 0
    assert not be.calls                                      # every unit done in the old layout
    for d, _, fs in os.walk(new):
        for f in fs:
            assert not SP.LEGACY_NAME_RE.match(f), f
            if f.endswith(".parquet"):
                assert not set(pd.read_parquet(os.path.join(d, f)).columns) & set(SP.LEAK_COLS), f
    ex = new / slug / "export"
    assert len(pd.read_parquet(ex / "attempts" / "attempts.parquet")) == 4 * 3
    absorbed = json.load(open(new / "manifest.json"))["absorbed"]
    assert any(a.endswith("samples_00000.parquet") for a in absorbed)
    assert SP.restore(str(new), [str(out)]) == 0             # the absorbed files are not copied back


def test_merge_manifest_unions_absorbed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    SP.write_json(str(a / "manifest.json"), {"absorbed": ["m/rubric/items_00000.parquet"]})
    SP.write_json(str(b / "manifest.json"), {"absorbed": ["m/rubric/items-x-000.parquet"]})
    SP.merge_manifest(str(b), str(a / "manifest.json"))
    assert json.load(open(b / "manifest.json"))["absorbed"] == ["m/rubric/items-x-000.parquet",
                                                                "m/rubric/items_00000.parquet"]


# --- self-checks ---------------------------------------------------------------------------------------

class MisalignedRecorder(MockBackend):
    """The recorder's raw log-probs disagree with the engine's on the greedy forced readouts."""

    def forced(self):
        s = super().forced()
        s.lp_raw = s.lp_raw - 0.5
        return s


class Garbage(MockBackend):
    """fp16 overflow: every completion is '!!!!...'."""

    def rubric(self, prompt_text):
        return self._sample("!" * 40, lambda t, s: -0.1, lambda t, s: 0.1, "length")

    def attempt(self, seed, max_tokens):
        return self._sample("!" * max_tokens, lambda t, s: -0.1, lambda t, s: 0.1, "length")


def test_self_checks_stop_a_useless_run(synth):
    data, out, _ = synth
    slug = SP.model_slug(SP.DEFAULT_MODEL, "vllm")
    # misaligned raw log-probs: the attempts stop after their first shard (kept for a look), exit 77
    rc = SP.main(run_args(data, out, "--jobs", "attempts"), backend=MisalignedRecorder(), now=clock())
    assert rc == SP.EXIT_CHECK
    assert len(SP.Store(str(out / slug / "attempts"), "").read()) == 2        # --attempt-shard 2
    sess = json.load(open(out / "manifest.json"))["sessions"][-1]
    assert sess["stats"]["attempts"]["self_check"]["status"] == "FAILED"
    assert "recorder" in sess["stats"]["attempts"]["self_check"]["why"][0]
    # --no-self-check runs on
    assert SP.main(run_args(data, out, "--jobs", "attempts", "--no-self-check"), backend=MisalignedRecorder(),
                   now=clock()) == 0
    # garbage: the rubric's first shard does not parse; attempts of '!!!!' are degenerate
    g = out.parent / "garbage"
    be = Garbage()
    assert SP.main(run_args(data, g), backend=be, now=clock()) == SP.EXIT_CHECK
    assert not any(c["seeds"] for c in be.calls)                              # stopped before the attempts
    assert SP.main(run_args(data, g, "--jobs", "attempts"), backend=Garbage(), now=clock()) == SP.EXIT_CHECK
    assert SP.degenerate("!" * 100) and not SP.degenerate("Let me think. " * 20) and not SP.degenerate("!!!")


def test_setup_errors_exit_78(tmp_path):
    """A ConfigError (no token, no GPU, a failed download, ...) exits EXIT_CONFIG, which the RUN cell does not
    retry; checked on the script as the notebook runs it (a subprocess)."""
    r = subprocess.run([sys.executable, os.path.join(KDIR, "strong_probe.py"), "split-harness", "--export-dir",
                        str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert r.returncode == SP.EXIT_CONFIG and "_harness.json" in r.stderr


def test_budget_pieces_and_quota(synth):
    data, out, _ = synth
    args = SP.parse_args(run_args(data, out))
    ur, ua = SP.load_units(args)
    b = SP.plan_budget(args, MockTok(), ur, ua)
    a = b["attempts"]
    assert a["probe_units"] == 1 and a["rest_units"] == 2
    assert a["probe_hours_est_slow"] + a["rest_hours_est_slow"] == pytest.approx(a["hours_est_slow"], abs=0.02)
    assert [p["piece"] for p in b["session_plan"]] == ["rubric", "probe", "rest"]
    for p in b["session_plan"]:
        assert SP.parse_args(["run", *p["args"]]).jobs == (["rubric"] if p["piece"] == "rubric" else ["attempts"])
    q = b["quota"]
    assert q["gpu_h_planning"] == pytest.approx(SP.SMOKE_H + sum(p["gpu_h_slow"] for p in b["session_plan"]),
                                                abs=0.02)
    assert q["gpu_h_planning"] >= q["gpu_h_optimistic"] > SP.SMOKE_H and q["fits"]
    # without prefix caching the forced readouts' prefill is not cached
    nc = SP.plan_budget(SP.parse_args(run_args(data, out, "--no-prefix-caching")), MockTok(), ur, ua)
    assert nc["attempts"]["forced_cache_hit"] == 0 and a["forced_cache_hit"] == SP.FORCED_CACHE_HIT
    r = SP.rates(args)
    assert nc["attempts"]["hours_est"] == pytest.approx(
        SP.attempt_secs(3, a["prompt_tokens"], 3, 64, True, r, 0.0) / 3600, abs=0.01)


# --- the notebook's own cells --------------------------------------------------------------------------

def _cell(prefix):
    nb = json.loads(open(NB.NOTEBOOK, encoding="utf-8").read())
    return next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix))


def _helpers_ns():
    ns = {"os": os, "subprocess": subprocess, "sys": sys, "time": time}
    exec(_cell("import collections"), ns)
    return ns


def run_cell(outcomes, args=()):
    """Execute the RUN cell with sp() scripted: outcomes are (exit code, log text) per run; returns the
    options each run got after ARGS."""
    ns = _helpers_ns()
    ns["ARGS"] = list(args)
    calls, it = [], iter(outcomes)

    def sp(*a):
        assert a[0] == "run" and list(a[1:1 + len(args)]) == list(args)
        calls.append(list(a[1 + len(args):]))
        rc, text = next(it)
        ns["LOG_TAIL"][:] = text.splitlines(keepends=True)
        return rc

    ns["sp"], ns["wait_gpus_free"] = sp, (lambda *a, **k: None)
    exec(_cell("# Generation."), ns)
    return calls, ns


def test_run_cell_retries_address_the_failure_seen():
    # the init watchdog (exit 76): NCCL without P2P first, only then eager with less memory
    calls, ns = run_cell([(76, ""), (76, ""), (0, "")], args=["--jobs", "rubric"])
    assert calls == [[], ["--nccl-p2p-disable"], ["--nccl-p2p-disable", "--enforce-eager", "--gpu-mem", "0.85"]]
    # out of memory, then a Triton error: each retry adds its own remedy and keeps the earlier one
    calls, _ = run_cell([(1, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"),
                         (1, "triton.runtime.errors.OutOfResources: out of resource: shared memory"), (0, "")])
    assert calls == [[], ["--enforce-eager", "--gpu-mem", "0.85"],
                     ["--enforce-eager", "--gpu-mem", "0.85", "--no-prefix-caching"]]
    # the latest error in the log decides (an NCCL line early, a KV-cache error last)
    calls, _ = run_cell([(1, "NCCL error in: foo\n...\nValueError: ... larger than the maximum number of tokens "
                             "that can be stored in KV cache (4000)"), (0, "")])
    assert calls == [[], ["--enforce-eager", "--gpu-mem", "0.93"]]
    # a later value of an option wins: less memory after out of memory twice
    calls, _ = run_cell([(1, "CUDA out of memory"), (1, "CUDA out of memory"), (0, "")])
    assert calls[-1] == ["--enforce-eager", "--gpu-mem", "0.80", "--max-num-seqs", "32"]
    # not retried: done, deadline, a self-check, a setup error, a bad option
    for rc in (0, 75, 77, 78, 2):
        assert run_cell([(rc, "")])[0] == [[]]
    # at most three runs; a failure whose remedies are used up is not retried
    assert len(run_cell([(76, "")] * 5)[0]) == 3
    assert run_cell([(1, "prefix_prefill OutOfResources")] * 3)[0] == [[], ["--no-prefix-caching"]]
    assert run_cell([(1, "Segmentation fault"), (1, "Segmentation fault")])[0] == \
        [[], ["--enforce-eager", "--gpu-mem", "0.85"]]
    # every option a retry can add is one the script knows
    for rems in ns["REMEDIES"].values():
        for rem in rems:
            SP.parse_args(["run", *ns["merged"](["--nccl-p2p-disable"], rem)])


def test_install_cell_stops_when_vllm_does_not_import(capsys):
    """pip succeeds but `import vllm` fails (a CUDA / driver mismatch): the cell prints the error and
    stops the notebook instead of going on to plan and run."""
    code = _cell("# vLLM")
    runs = []

    def fake_run(cmd, capture_output, text):
        runs.append(cmd)
        from types import SimpleNamespace
        if "pip" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="ImportError: libcudart.so.12: cannot open shared "
                                                                 "object file")

    from types import SimpleNamespace
    ns = {"subprocess": SimpleNamespace(run=fake_run), "sys": sys}
    with pytest.raises(SystemExit) as e:
        exec(code, ns)
    assert "import vllm failed" in str(e.value) and len(runs) == 2
    assert "libcudart" in capsys.readouterr().out


def _gone(psutil, pid, wait_s=5.0):
    t = time.time()
    while time.time() - t < wait_s:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
@pytest.mark.parametrize("own_session", [False, True])
def test_sp_kills_what_a_run_leaves_behind(tmp_path, own_session):
    """The notebook's sp(): a script that dies leaving a child that holds the output pipe (a vLLM worker
    after its driver was SIGKILLed) neither hangs the cell nor survives it. The child is killed through the
    command's process group, or, if it left the group (own_session), as a descendant psutil saw while the
    command ran. A later command is not touched by the earlier one's delayed kill."""
    psutil = pytest.importorskip("psutil")
    ns = _helpers_ns()
    pidfile, script = tmp_path / "child.pid", tmp_path / "fake_run.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        if sys.argv[1] == "run":
            c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                 start_new_session={own_session})             # inherits the pipe
            open({str(pidfile)!r}, "w").write(str(c.pid))
            print("worker started", flush=True)
            time.sleep(0.5)
            sys.exit(3)
        print("later command", flush=True)
        time.sleep(1.0)
    """))
    ns["SCRIPT"], ns["REAP_GRACE_S"], ns["WATCH_S"] = str(script), 0.2, 0.05
    t = time.time()
    assert ns["sp"]("run") == 3 and time.time() - t < 30
    assert "worker started" in "".join(ns["LOG_TAIL"])
    pid = int(pidfile.read_text())
    if not _gone(psutil, pid):
        os.kill(pid, 9)
        pytest.fail("the orphaned child survived sp()")
    # a command that ends normally arms no kill; one started right after it (the script again) runs to its end
    ns["REAP_GRACE_S"] = 0.3
    assert ns["sp"]("plan") == 0
    assert ns["sp"]("plan") == 0 and "later command" in "".join(ns["LOG_TAIL"])
